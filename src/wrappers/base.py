import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from data.util import clamp_boxes_xyxy


class BaseDetectionModel(ABC, nn.Module):
    """Abstract base class for object detection models."""

    def __init__(self, model, num_classes, device):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.to(self.device)

        self.train()
        self._set_num_classes(num_classes)

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            images: Batch of images [B, C, H, W]
            targets: List of dicts with 'boxes' [N, 4] and 'labels' [N]
                    Only required during training

        Returns:
            During training: Dict with losses {'loss': total_loss, 'box_loss': ..., 'cls_loss': ...}
            During inference: Dict with predictions {'boxes': ..., 'scores': ..., 'labels': ...}
        """
        if self.training:
            if targets is None:
                raise ValueError("Targets must be provided in training mode.")
            return self._forward_train(images, targets)
        else:
            out_raw = self._forward_eval(images)

            return self._postprocess_predictions(out_raw, images.shape)

    def _postprocess_predictions(
        self, out_raw: List[Dict[str, torch.Tensor]], img_size: tuple
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Post-process raw model predictions to evaluator-ready format.

        Args:
            predictions: List of dicts with raw model outputs per image.
            img_size: Tuple (width, height) of the input images.

        Returns:
            List of dicts with 'boxes', 'scores', 'labels' per image.
        """
        B, _, H, W = img_size

        outputs_pp = []
        for b in range(B):
            out = out_raw["predictions"][b]

            # Clamp boxes to image boundaries
            if out["boxes"].numel() > 0:
                boxes = clamp_boxes_xyxy(out["boxes"], (W, H))
            else:
                boxes = out["boxes"]

            scores = out["scores"]
            labels = out["labels"]

            # Keep only valid boxes
            valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

            outputs_pp.append(
                {
                    "boxes": boxes[valid_mask],
                    "scores": scores[valid_mask],
                    "labels": labels[valid_mask],
                }
            )

        return {"predictions": outputs_pp}

    @abstractmethod
    def _set_num_classes(self, num_classes: int):
        pass

    def train(self, mode: bool = True):
        """Override train mode to ensure proper behavior."""
        super().train(mode)
        if self.model is not None:
            self.model.train(mode)
        return self

    def eval(self):
        super().eval()
        if self.model is not None:
            self.model.train(False)
        return self
