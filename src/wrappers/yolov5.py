import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .base import BaseDetectionModel
from yolov5.models.yolo import DetectionModel
from yolov5.utils.general import non_max_suppression


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
        model: DetectionModel,
        num_classes: int,
        device: str,
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

            self.loss_fn = ComputeLoss(self.model)
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
        detect_layer = self.model.model[-1]

        # Update detection layer class count
        detect_layer.nc = num_classes
        detect_layer.no = num_classes + 5  # x, y, w, h, obj + num_classes

        # Recreate output conv layers with new dimensions
        detect_layer.m = nn.ModuleList(
            [
                nn.Conv2d(
                    x.in_channels, detect_layer.no * len(detect_layer.anchors[i]), 1
                ).to(self.device)
                for i, x in enumerate(detect_layer.m)
            ]
        )

        # Reinitialize weights if requested
        for m in detect_layer.m:
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            nn.init.zeros_(m.bias)

        print(f"✅ Updated model to {num_classes} classes")

    def _forward_train(self, images: torch.Tensor, targets) -> dict:
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
                - Inference (flattened): None
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
        loss, _ = self.compute_loss(pred, targets, images.shape[2:])

        return {
            "loss": loss,
        }

    def _forward_eval(
        self,
        images: torch.Tensor,
        iou_thresh=0.5,
        conf_thresh=0.005,
    ) -> dict:
        """
        Forward pass for evaluation — returns final detections.
        """
        with torch.no_grad():
            pred = self.model(images)[0]  # decoded predictions

            pred = non_max_suppression(
                pred,
                conf_thres=conf_thresh,
                iou_thres=iou_thresh,
            )

        outputs = []

        for p in pred:
            if p is None or len(p) == 0:
                outputs.append(
                    {
                        "boxes": torch.empty((0, 4), device=images.device),
                        "scores": torch.empty((0,), device=images.device),
                        "labels": torch.empty(
                            (0,), device=images.device, dtype=torch.long
                        ),
                    }
                )
                continue

            outputs.append(
                {
                    "boxes": p[:, :4],  # xyxy
                    "scores": p[:, 4],
                    "labels": p[:, 5].long(),
                }
            )

        return {"predictions": outputs}

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
        yolo_targets = self._convert_targets_to_yolov5_format(targets, img_size)

        # Compute loss using YOLOv5's loss function
        loss, loss_items = self.loss_fn(predictions, yolo_targets)

        # Return structured loss dict
        return loss, loss_items

    def predict(
        self,
        images: torch.Tensor,
        iou_thresh=0.5,
        conf_thresh=0.005,
    ) -> dict:
        """
        Run inference to obtain final detections.

        Args:
            images (torch.Tensor): Batch of input images, shape [B, 3, H, W].
            iou_thresh (float): IoU threshold for NMS.
            conf_thresh (float): Confidence threshold for filtering boxes.

        Returns:
            Dict with:
            'predictions' : List of dicts per image with:
                - 'boxes': Tensor [N, 4] in xyxy pixel coords
                - 'scores': Tensor [N] confidence scores
                - 'labels': Tensor [N] class indices
        """
        return self._forward_eval(
            images,
            iou_thresh=iou_thresh,
            conf_thresh=conf_thresh,
        )

    def _convert_targets_to_yolov5_format(self, targets, img_size):
        """
        Format batch targets into a unified tensor representation for object detection.

        Converts target annotations from a list of dictionaries containing boxes and labels
        into a single tensor where each row represents an object with its metadata.

        Args:
            targets: List of dicts with:
                - 'boxes': [N, 4] in COCO format [x, y, w, h] (pixels)
                - 'class_labels': [N] class indices
            img_size: (height, width) tuple
            device: target device

        Returns:
            Tensor of shape [total_objects, 6] with format [batch_idx, class, cx, cy, w, h]
            where cx, cy, w, h are normalized to [0, 1]
        """

        def coco_to_xywh_norm(boxes, img_shape_hw, device):
            """Convert COCO [x, y, w, h] pixels to normalized [cx, cy, w, h]"""
            h, w = img_shape_hw
            boxes = boxes.to(device).clone()

            # COCO format: [x, y, width, height] where (x,y) is top-left corner
            x, y, box_w, box_h = boxes.T

            # Convert to normalized center coordinates
            cx = (x + box_w / 2) / w  # center x normalized
            cy = (y + box_h / 2) / h  # center y normalized
            w_norm = box_w / w  # width normalized
            h_norm = box_h / h  # height normalized

            return torch.stack([cx, cy, w_norm, h_norm], dim=1)

        formatted = []

        for i, t in enumerate(targets):
            if len(t["boxes"]) == 0:
                continue

            boxes = t["boxes"]
            labels = t["class_labels"].unsqueeze(1)
            img_idx = torch.full((len(labels), 1), i)

            xywh = coco_to_xywh_norm(boxes, img_size, self.device)
            merged = torch.cat(
                [img_idx.to(self.device), labels.to(self.device), xywh.to(self.device)],
                dim=1,
            )

            formatted.append(merged)

        return torch.cat(formatted, dim=0) if len(formatted) else None
