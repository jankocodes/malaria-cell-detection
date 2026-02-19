import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from .base import BaseDetectionModel
from transformers import DeformableDetrForObjectDetection
import torch
from typing import Dict, Any, List
from torchvision.ops import box_convert


class DeformableDetrWrapper(BaseDetectionModel):
    """
    DeformableDetrWrapper - PyTorch wrapper for DeformableDetr object detection models.

    A specialized wrapper that adapts DeformableDetr models to a standardized detection interface,
    handling model configuration, loss computation, and target format conversion.

    Attributes:
        model (nn.Module): The underlying DeformableDetr model instance
        num_classes (int): Number of object classes for detection
        loss_fn (callable): Loss function for training (optional)
        device: Computation device (CPU or GPU)

    Methods:
        _set_num_classes(num_classes): Adapt model architecture for target class count
        _forward_train(images, targets): Execute model forward pass and compute loss during training
        _forward_eval(images): Execute model forward pass for evaluation/inference
        _convert_targets_to_DeformableDetr_format(targets, img_h, img_w): Convert targets from COCO to DeformableDetr format

    Example:
        >>> model = DeformableDetrWrapper(DeformableDetr_model, num_classes=80, device='cuda')
        >>> output = model(images, targets)
        >>> loss = output['loss']
    """

    def __init__(
        self,
        model: DeformableDetrForObjectDetection,
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
        Dynamically change the number of classes for DeformableDetr models without touching source code.

        Args:
            model: DeformableDetrForObjectDetection
            num_classes (int): number of target classes
            device: optional; ensures replaced layers are moved to correct device
        """
        # Num classes is adapted in factory class

        # Extract old classifier for shape reference
        # Deformable DETR uses class_embed (ModuleList of Linear layers, one per decoder layer)
        old_classifier = self.model.class_embed[0]
        in_features = old_classifier.in_features  # usually 256
        num_decoder_layers = len(self.model.class_embed)  # usually 6

        # Create new classifiers with correct number of classes for each decoder layer
        new_classifiers = nn.ModuleList()
        for _ in range(num_decoder_layers):
            new_classifier = nn.Linear(in_features, num_classes + 1)
            # +1 because DETR always includes a "no-object" class

            # Optional: better initialization
            nn.init.xavier_uniform_(new_classifier.weight)
            nn.init.constant_(new_classifier.bias, 0.0)

            new_classifiers.append(new_classifier)

        # Assign back into model
        self.model.class_embed = new_classifiers
        self.model.config.num_labels = num_classes

        # Make sure new layers are on the correct device
        if self.device is not None:
            self.model.class_embed.to(self.device)
        print(f"✅ Updated model to {num_classes} classes")

    def _forward(
        self, images: torch.Tensor, targets=None, return_predictions=False
    ) -> dict:
        """
        Unified DeformableDetr forward pass that uses DeformableDetr's internal loss during training.
        """

        result = {}
        # Convert targets from COCO format to DeformableDetr format
        # COCO: [x, y, w, h] in pixels
        # DeformableDetr: [x_center, y_center, w, h] normalized to [0, 1]
        if targets is not None:
            img_h, img_w = images.shape[2:]
            detr_targets = self._convert_targets_to_detr_format(targets, img_h, img_w)

            # HF DeformableDetr requires named arguments
            outputs = self.model(pixel_values=images, labels=detr_targets)

            # HF DeformableDetr returns a dict-like object containing loss and loss components
            total_loss = outputs.loss
            result["loss"] = total_loss

        if return_predictions or targets is None:
            predictions = self._predict(images)
            result["predictions"] = predictions

        return result

    def _predict(
        self, images: torch.Tensor, postprocess=True, conf_thresh=0.5, iou_thresh=0.5
    ):
        """
        Run inference and post-process outputs.

        Args:
            images (torch.Tensor): Batch of input images, shape [B, 3, H, W].
            conf_thresh (float): Confidence threshold for filtering boxes.
            iou_thresh (float): IoU threshold for NMS.

        Returns:
            List of dicts per image with keys 'boxes', 'scores', 'labels'.
        """
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)
        self.model.train()

        if postprocess:
            predictions = self._post_process_predictions(
                predictions, images.shape, conf_thresh, iou_thresh
            )
        return predictions

    def _post_process_predictions(
        self, outputs, img_size, conf_thresh=0.5, iou_thresh=0.5
    ):
        """
        Post-process raw model predictions to evaluator-ready format.

        Args:
            outputs: Raw outputs from the DeformableDetr model.
            img_size: Tuple (B, C, H, W) of the input images.
            conf_thresh: Confidence threshold for filtering predictions.
            iou_thresh: IoU threshold for NMS.

        Returns:
            List of dicts with 'boxes', 'scores', 'labels' per image.
        """
        _, _, H, W = img_size
        logits = outputs.logits  # [B, Q, C+1]
        boxes = outputs.pred_boxes  # [B, Q, 4] (cxcywh, normalized)
        B, Q, _ = boxes.shape

        # Convert class logits → probabilities (drop no-object)
        probs = logits.softmax(-1)[..., :-1]
        scores, labels = probs.max(-1)  # [B, Q]

        boxes_xyxy = box_convert(
            boxes.view(-1, 4), in_fmt="cxcywh", out_fmt="xyxy"
        ).view(B, Q, 4)

        # Scale to pixel coordinates
        scale = torch.tensor([W, H, W, H], device=boxes.device)
        boxes_xyxy = boxes_xyxy * scale

        predictions: List[Dict[str, torch.Tensor]] = []

        for b in range(B):
            keep = scores[b] > conf_thresh

            if keep.sum() == 0:
                predictions.append(
                    {
                        "boxes": torch.empty((0, 4), device=boxes.device),
                        "scores": torch.empty((0,), device=boxes.device),
                        "labels": torch.empty(
                            (0,), device=boxes.device, dtype=torch.long
                        ),
                    }
                )
                continue

            predictions.append(
                {
                    "boxes": boxes_xyxy[b][keep],
                    "scores": scores[b][keep],
                    "labels": labels[b][keep],
                }
            )
        return predictions

    def _convert_targets_to_detr_format(
        self, targets: List[Dict[str, torch.Tensor]], img_h: int, img_w: int
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert targets from COCO format to DeformableDetr format.

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
        DeformableDetr_targets = []

        for target in targets:
            boxes = target["boxes"].clone().float()  # [N, 4]
            class_labels = target["class_labels"].clone()

            # Convert COCO [x, y, w, h] to DeformableDetr [x_center, y_center, w, h]
            # and normalize to [0, 1]
            boxes[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) / img_w  # x_center
            boxes[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / img_h  # y_center
            boxes[:, 2] = boxes[:, 2] / img_w  # width
            boxes[:, 3] = boxes[:, 3] / img_h  # height

            DeformableDetr_targets.append(
                {"boxes": boxes, "class_labels": class_labels}
            )

        return DeformableDetr_targets
