import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from .base import BaseDetectionModel
from transformers import DetrForObjectDetection
import torch
from typing import Dict, Any, List
from torchvision.ops import box_convert


class DetrWrapper(BaseDetectionModel):
    """
    DetrWrapper - PyTorch wrapper for Detr object detection models.

    A specialized wrapper that adapts Detr models to a standardized detection interface,
    handling model configuration, loss computation, and target format conversion.

    Attributes:
        model (nn.Module): The underlying Detr model instance
        num_classes (int): Number of object classes for detection
        loss_fn (callable): Loss function for training (optional)
        device: Computation device (CPU or GPU)

    Methods:
        _set_num_classes(num_classes): Adapt model architecture for target class count
        _forward_train(images, targets): Execute model forward pass and compute loss during training
        _forward_eval(images): Execute model forward pass for evaluation/inference
        _convert_targets_to_detr_format(targets, img_h, img_w): Convert targets from COCO to DETR format

    Example:
        >>> model = DetrWrapper(Detr_model, num_classes=80, device='cuda')
        >>> output = model(images, targets)
        >>> loss = output['loss']
    """

    def __init__(
        self,
        model: DetrForObjectDetection,
        num_classes: int,
        device,
        loss_fn: Optional[callable] = None,
    ):
        super().__init__(
            model=model,
            num_classes=num_classes,
            device=device,
        )
        self.loss_fn = loss_fn

    def _set_num_classes(self, num_classes: int):
        """
        Dynamically change the number of classes for DETR models without touching source code.

        Args:
            model: DetrForObjectDetection
            num_classes (int): number of target classes
            device: optional; ensures replaced layers are moved to correct device
        """

        # Extract old classifier for shape reference
        old_classifier = self.model.class_labels_classifier
        in_features = old_classifier.in_features  # usually 256

        # Create new classifier with correct number of classes
        new_classifier = nn.Linear(in_features, num_classes + 1)
        # +1 because DETR always includes a "no-object" class

        # Optional: better initialization
        nn.init.xavier_uniform_(new_classifier.weight)
        nn.init.constant_(new_classifier.bias, 0.0)

        # Assign back into model
        self.model.class_labels_classifier = new_classifier
        self.model.config.num_labels = num_classes

        # Make sure new layer is on the correct device
        if self.device is not None:
            self.model.class_labels_classifier.to(self.device)

        print(f"✅ Updated model to {num_classes} classes")

    def _forward_train(self, images: torch.Tensor, targets=None) -> dict:
        """
        Unified DETR forward pass that uses DETR's internal loss during training.
        """

        if targets is None:
            raise ValueError("targets must be provided during training")

        # Convert targets from COCO format to DETR format
        # COCO: [x, y, w, h] in pixels
        # DETR: [x_center, y_center, w, h] normalized to [0, 1]
        img_h, img_w = images.shape[2:]
        detr_targets = self._convert_targets_to_detr_format(targets, img_h, img_w)

        # HF DETR requires named arguments
        outputs = self.model(pixel_values=images, labels=detr_targets)

        # HF DETR returns a dict-like object containing loss and loss components
        total_loss = outputs.loss
        loss_dict = outputs.loss_dict

        return {
            "loss": total_loss,
            # "loss_dict": loss_dict, -> optional detailed losses
        }

    def _convert_targets_to_detr_format(
        self, targets: List[Dict[str, torch.Tensor]], img_h: int, img_w: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert targets from COCO format to DETR format.

        Args:
            targets: List of dicts with:
                - 'boxes': [N, 4] in COCO format [x, y, w, h] (pixels)
                - 'class_labels': [N] class indices
            img_h: Image height in pixels
            img_w: Image width in pixels

        Returns:
            List of dicts with:
                - 'boxes': [N, 4] in normalized [x_center, y_center, w, h] format
                - 'class_labels': [N] class indices
        """
        detr_targets = []

        for target in targets:
            boxes = target["boxes"].clone().float()  # [N, 4]
            class_labels = target["class_labels"].clone()

            # Convert COCO [x, y, w, h] to DETR [x_center, y_center, w, h]
            # and normalize to [0, 1]
            boxes[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) / img_w  # x_center
            boxes[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / img_h  # y_center
            boxes[:, 2] = boxes[:, 2] / img_w  # width
            boxes[:, 3] = boxes[:, 3] / img_h  # height

            detr_targets.append({"boxes": boxes, "class_labels": class_labels})

        return detr_targets

    def _forward_eval(self, images: torch.Tensor) -> Dict[str, Any]:
        """
        Evaluation forward pass for DETR with post-processing.

        Returns YOLO-style predictions:
        - boxes: xyxy (pixels)
        - scores: confidence scores
        - labels: class indices
        """
        with torch.no_grad():
            outputs = self.model(images)

        logits = outputs.logits  # [B, Q, C+1]
        boxes = outputs.pred_boxes  # [B, Q, 4] (cxcywh, normalized)

        # Convert class logits → probabilities (drop no-object)
        probs = logits.softmax(-1)[..., :-1]
        scores, labels = probs.max(-1)  # [B, Q]

        B, Q, _ = boxes.shape
        _, _, H, W = images.shape

        boxes_xyxy = box_convert(
            boxes.view(-1, 4), in_fmt="cxcywh", out_fmt="xyxy"
        ).view(B, Q, 4)

        # Scale to pixel coordinates
        scale = torch.tensor([W, H, W, H], device=boxes.device)
        boxes_xyxy = boxes_xyxy * scale

        outputs_pp: List[Dict[str, torch.Tensor]] = []

        conf_thresh = 0.25  # keep consistent with YOLO

        for b in range(B):
            keep = scores[b] > conf_thresh

            if keep.sum() == 0:
                outputs_pp.append(
                    {
                        "boxes": torch.empty((0, 4), device=boxes.device),
                        "scores": torch.empty((0,), device=boxes.device),
                        "labels": torch.empty(
                            (0,), device=boxes.device, dtype=torch.long
                        ),
                    }
                )
                continue

            outputs_pp.append(
                {
                    "boxes": boxes_xyxy[b][keep],
                    "scores": scores[b][keep],
                    "labels": labels[b][keep],
                }
            )

        return {"predictions": outputs_pp}
