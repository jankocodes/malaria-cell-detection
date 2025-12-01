import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from .base import BaseDetectionModel
from torchvision.models.detection import RetinaNet
import torch
import torch.nn as nn


class RetinaNetWrapper(BaseDetectionModel):
    """
    Wrapper for RetinaNet model that provides:
    - Clean forward() method that returns raw predictions
    - Separate compute_loss() method for training
    - Target format conversion utilities
    - Output post-processing
    """

    def __init__(
        self,
        model: RetinaNet,
        num_classes: int,
        device,
        loss_fn: Optional[callable] = None,
    ):
        """
        Initialize RetinaNet wrapper.

        Args:
            model: Loaded RetinaNet model (from torch.hub.load)
            num_classes: Number of detection classes
            img_size: Input image size (default: 640)
            loss_fn: Custom loss function. If None, uses default RetinaNet loss
        """
        super().__init__(model=model, num_classes=num_classes, device=device)

        self.loss_fn = loss_fn

        # Initialize loss computer if not provided
        if self.loss_fn is None:
            self._init_default_loss()

    def _init_default_loss(self):
        """Initialize default RetinaNet loss function."""
        try:
            self.loss_fn = None
        except ImportError:
            print("Warning: Could not import RetinaNet loss. Set loss_fn manually.")
            self.loss_fn = None

    def _set_num_classes(self, num_classes: int):
        """
        Modify the number of classes in the model.

        This updates:
        - Model's nc (number of classes) attribute
        - Model's class names
        - Detection head output dimensions
        - Optionally reinitializes the detection head weights

        Args:
            num_classes: New number of classes
            reinit_head: If True, reinitialize detection head weights (default: True)
        """

        old_cls = self.model.head.classification_head

        num_anchors = old_cls.num_anchors
        in_channels = old_cls.cls_logits.in_channels

        new_cls = nn.Conv2d(
            in_channels,
            num_anchors * num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Replace the layer
        self.model.head.classification_head.cls_logits = new_cls
        self.model.head.classification_head.num_classes = num_classes

        # Init weights like torchvision
        torch.nn.init.normal_(new_cls.weight, std=0.01)

        prior_prob = 0.01
        nn.init.constant_(
            new_cls.bias, -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        )

        print(f"✅ Updated model to {num_classes} classes")

    def forward(self, images: torch.Tensor, targets=None) -> Dict[str, Any]:
        """
        Forward pass through RetinaNet.

        Returns a dict with:
            - 'loss': dict of losses (empty during eval)
            - 'predictions': list of detection dicts (always available)
        """
        # Convert target boxes to xyxy format if targets are provided
        coco_targets = (
            self._convert_targets_xyxy(targets) if targets is not None else None
        )

        # Run the model
        outputs = self.model(
            images, coco_targets
        )  # returns losses if training, detections if eval

        # Initialize return dict
        result = {}

        if self.model.training:
            # outputs is a dict of losses during training
            cls_loss = outputs["classification"]
            box_loss = outputs["bbox_regression"]
            total_loss = cls_loss + box_loss
            result["loss"] = total_loss

        else:
            # During eval, outputs are the predictions
            result["predictions"] = outputs

        return result

    # --- New function ---
    def _convert_targets_xyxy(
        self, targets: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Converts target boxes from COCO format [x, y, w, h] to [x1, y1, x2, y2].

        Args:
            targets: List of dicts, each containing 'boxes' and 'labels'.

        Returns:
            List of dicts with boxes in xyxy format.
        """
        converted = []
        for t in targets:
            if len(t["boxes"]) == 0:
                converted.append(t)
                continue

            boxes = t["boxes"].clone()  # [N, 4]
            # COCO -> xyxy
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h

            converted.append({"boxes": boxes, "labels": t["labels"]})

        return converted
