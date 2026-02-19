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

        # Enable gradients for all parameters (they may be frozen when loaded from YOLO)
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        return_predictions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Behavior is determined by arguments, not by model.train() / model.eval() state:
          - targets provided      → compute and return {"loss": ...}
          - return_predictions    → also include {"predictions": ...}
          - targets is None       → return predictions only

        Args:
            images: Batch of images [B, C, H, W]
            targets: List of dicts with 'boxes' [N, 4] and 'class_labels' [N].
                     When provided, loss is computed and returned.
            return_predictions: If True, predictions are included in the output
                                alongside the loss.

        Returns:
            Dict with one or both of:
              {'loss': tensor}
              {'predictions': list of per-image dicts with 'boxes', 'scores', 'labels'}
        """
        result = {}

        if targets is not None:
            # Pass return_predictions so _forward_train can optionally include
            # predictions in a single model call (e.g. YOLOv5 eval-mode tuple).
            result.update(self._forward(images, targets, return_predictions))
        else:
            result.update(self._predict(images))
        print(result.keys())

        if "predictions" in result:
            result["predictions"] = self._postprocess_predictions(
                result["predictions"], images.shape
            )

        return result

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
            out = out_raw[b]

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

        return outputs_pp

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
