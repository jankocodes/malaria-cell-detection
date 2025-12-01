import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from .base import BaseDetectionModel


class YOLOv5Wrapper(BaseDetectionModel):
    """
    YOLOv5Wrapper - PyTorch wrapper for YOLOv5 object detection models.

    A specialized wrapper that adapts YOLOv5 models to a standardized detection interface,
    handling model configuration, loss computation, and target format conversion.

    Attributes:
        model (nn.Module): The underlying YOLOv5 model instance
        num_classes (int): Number of object classes for detection
        loss_fn (callable): Loss function for training (ComputeLoss or custom)
        device: Computation device (CPU or GPU)

    Methods:
        forward(images, targets): Execute model forward pass and compute loss
        compute_loss(predictions, targets, img_size): Calculate detection losses
        _set_num_classes(num_classes): Adapt model architecture for target class count
        _init_default_loss(): Initialize YOLOv5's native loss function
        _convert_targets_to_yolo_format(targets, img_size, device): Transform target annotations

    Example:
        >>> model = YOLOv5Wrapper(yolov5_model, num_classes=80, device='cuda')
        >>> output = model(images, targets)
        >>> predictions, loss = output['predictions'], output['loss']
    """

    def __init__(
        self,
        model: nn.Module,
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

        # Initialize loss computer if not provided
        if self.loss_fn is None:
            self._init_default_loss()

    def _init_default_loss(self):
        """Initialize default YOLOv5 loss function."""
        try:
            from yolov5.utils.loss import ComputeLoss

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

    def forward(self, images: torch.Tensor, targets) -> dict:
        """
        Forward pass — returns raw model predictions and computed loss.

        Args:
            images (torch.Tensor): Batch of input images, shape [B, 3, H, W].
            targets (List[Dict], optional): Ground-truth annotations per image.
            Each dict should contain:
                - 'boxes': Tensor [N, 4] in xyxy pixel coords
                - 'labels': Tensor [N] class indices
            Required for training; may be None for inference.

        Returns:
            Dict with:
            'predictions' :
                - Training (YOLOv5 multi-scale): list of 3 tensors:
                  [B, 3, H/8,  W/8,  5 + num_classes],
                  [B, 3, H/16, W/16, 5 + num_classes],
                  [B, 3, H/32, W/32, 5 + num_classes]
                - Inference (flattened): [B, num_predictions, 5 + num_classes]
            'loss' :
                - Loss returned by compute_loss when targets are provided.
                - None if targets is omitted (inference mode).

        Notes:
            - 'predictions' are raw network outputs (anchor/logit space) and require
              decoding and NMS to produce final detections.
            - When targets are supplied, compute_loss() is invoked and its result
              is returned under 'loss'.
        """
        pred = self.model(images)

        if self.training:
            loss = self.compute_loss(pred, targets, images.shape[2:])

            return {
                "predictions": pred,
                "loss": loss,
            }

        return {"predictions": pred}

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

        def xyxy_to_xywh_norm(boxes, img_shape_hw, device):
            """Convert xyxy boxes to normalized xywh"""
            h, w = img_shape_hw
            boxes = boxes.to(device)
            x1, y1, x2, y2 = boxes.T
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            return torch.stack([cx, cy, bw, bh], dim=1)

        formatted = []

        for i, t in enumerate(targets):
            if len(t["boxes"]) == 0:
                continue

            boxes = t["boxes"]
            labels = t["labels"].unsqueeze(1)
            img_idx = torch.full((len(labels), 1), i)

            xywh = xyxy_to_xywh_norm(boxes, img_size, self.device)
            merged = torch.cat([img_idx, labels.to(device), xywh.to(device)], dim=1)

            formatted.append(merged)

        return torch.cat(formatted, dim=0) if len(formatted) else None
