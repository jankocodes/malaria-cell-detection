import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any


class BaseDetectionModel(ABC, nn.Module):
    """Abstract base class for object detection models."""

    def __init__(self, model, num_classes, device):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.device = device

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
            return self._forward_eval(images)

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
