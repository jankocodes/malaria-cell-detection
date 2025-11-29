import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from .base import BaseDetectionModel


class YOLOv5Wrapper(BaseDetectionModel):
    """
    Wrapper for YOLOv5 model that provides:
    - Clean forward() method that returns raw predictions
    - Separate compute_loss() method for training
    - Target format conversion utilities
    - Output post-processing
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        device,
        loss_fn: Optional[callable] = None,
    ):
        """
        Initialize YOLOv5 wrapper.

        Args:
            model: Loaded YOLOv5 model (from torch.hub.load)
            num_classes: Number of detection classes
            img_size: Input image size (default: 640)
            loss_fn: Custom loss function. If None, uses default YOLOv5 loss
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.device = device

        # adapt number of classes
        self._set_num_classes(num_classes)

        # Initialize loss computer if not provided
        if self.loss_fn is None:
            self._init_default_loss()

    def _init_default_loss(self):
        """Initialize default YOLOv5 loss function."""
        try:
            from yolov5.utils.loss import ComputeLoss

            print(type(self.model.model))
            self.loss_fn = ComputeLoss(self.model.model)
        except ImportError:
            print("Warning: Could not import YOLOv5 loss. Set loss_fn manually.")
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
        self.num_classes = num_classes

        # Update model-level attributes
        self.model.nc = num_classes
        self.model.names = [f"class_{i}" for i in range(num_classes)]

        # Get the Detect layer (last layer in YOLOv5)
        detect_layer = self.model.model.model[-1]

        # Update detection layer class count
        detect_layer.nc = num_classes
        detect_layer.no = num_classes + 5  # x, y, w, h, obj + num_classes

        # Recreate output conv layers with new dimensions
        detect_layer.m = nn.ModuleList(
            [
                nn.Conv2d(
                    x.in_channels, detect_layer.no * len(detect_layer.anchors[i]), 1
                )
                for i, x in enumerate(detect_layer.m)
            ]
        )

        # Reinitialize weights if requested
        for m in detect_layer.m:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)

        print(f"✅ Updated model to {num_classes} classes")

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Standard PyTorch forward pass - returns raw predictions only.

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            Tuple of prediction tensors at 3 scales:
            - [B, 3, H/8, W/8, 5+num_classes]
            - [B, 3, H/16, W/16, 5+num_classes]
            - [B, 3, H/32, W/32, 5+num_classes]

            Or in inference mode (after model flattening):
            - [B, num_predictions, 5+num_classes]
        """
        return self.model(images)

    def compute_loss(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: List[Dict[str, torch.Tensor]],
        img_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute YOLOv5 loss separately from forward pass.

        Args:
            predictions: Output from forward() - tuple of prediction tensors
            targets: List of target dicts with:
                - 'boxes': [N, 4] in format [x1, y1, x2, y2] (normalized 0-1)
                - 'labels': [N] class indices

        Returns:
            Dict with losses: {'total_loss', 'box_loss', 'obj_loss', 'cls_loss'}
        """
        if self.loss_fn is None:
            raise ValueError("Loss function not initialized. Set loss_fn in __init__")

        # Convert targets to YOLOv5 format
        yolo_targets = self._convert_targets_to_yolo_format(
            targets, img_size, self.device
        )

        # Compute loss using YOLOv5's loss function
        loss, loss_items = self.loss_fn(predictions, yolo_targets)

        # Return structured loss dict
        return loss, loss_items

    def _convert_targets_to_yolo_format(self, targets, img_size, device="cpu"):
        """
        Format batch targets into a unified tensor representation for object detection.

        Converts target annotations from a list of dictionaries containing boxes and labels
        into a single tensor where each row represents an object with its metadata.

        """
        formatted = []

        for i, t in enumerate(targets):

            if len(t["boxes"]) == 0:
                continue

            boxes = t["boxes"]
            labels = t["labels"].unsqueeze(1)
            img_idx = torch.full((len(labels), 1), i)

            xywh = self._xyxy_to_xywh_norm(boxes, img_size, device)
            merged = torch.cat([img_idx, labels.to(device), xywh.to(device)], dim=1)

            formatted.append(merged)

        return torch.cat(formatted, dim=0) if len(formatted) else None

    def _xyxy_to_xywh_norm(self, boxes, img_shape_hw, device):
        """Convert xyxy boxes to normalized xywh"""
        h, w = img_shape_hw
        boxes = boxes.to(device)
        x1, y1, x2, y2 = boxes.T
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return torch.stack([cx, cy, bw, bh], dim=1)
