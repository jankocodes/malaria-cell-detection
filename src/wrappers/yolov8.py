from typing import List, Dict, Tuple, Optional
import torch
from ultralytics.nn.tasks import DetectionModel

if __name__ == "__main__":
    from base import BaseDetectionModel
else:
    from .base import BaseDetectionModel


class YOLOv8Wrapper(BaseDetectionModel):
    """
    Unified wrapper for Ultralytics YOLOv8.

    Training mode:
        forward(images, targets) -> {"loss": total_loss}

    Eval mode:
        forward(images) -> {"boxes": ..., "scores": ..., "labels": ...}
    """

    def __init__(
        self,
        model: DetectionModel,
        num_classes: int,
        device="cuda" if torch.cuda.is_available() else "cpu",
        loss_fn: Optional[callable] = None,
    ):
        super().__init__(model=model, num_classes=num_classes, device=device)
        self.loss_fn = loss_fn
        if self.loss_fn is None:
            self._init_default_loss()

    def _init_default_loss(self):
        """Initialize default YOLOv8 loss function."""
        try:
            from ultralytics.utils.loss import v8DetectionLoss

            self.loss_fn = v8DetectionLoss(self.model)

            class HParams:
                def __init__(self, box=0.05, cls=0.5, dfl=1.5):
                    self.box = box
                    self.cls = cls
                    self.dfl = dfl

            self.loss_fn.hyp = HParams()
        except ImportError:
            print("Warning: Could not import YOLOv8 loss. Set loss_fn manually.")
            self.loss_fn = None

    def _set_num_classes(self, num_classes: int):

        # YOLOv8 number of classes are adapted in the factory class

        print(f"✅ Updated YOLOv8 model to {num_classes} classes )")

    def _forward_train(
        self, images: torch.Tensor, targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        DetectionModel in training mode returns raw predictions that need to be
        passed to the loss function.
        """

        # Forward through model - returns list of prediction tensors
        # Shape of each: [batch, num_anchors, grid_h, grid_w, 4+nc]
        pred = self.model(images)

        total_loss, loss_items = self.compute_loss(pred, targets)

        if total_loss.ndim > 0:
            total_loss = total_loss.sum()

        # ensure requires_grad
        total_loss = total_loss.to(self.device)
        if not total_loss.requires_grad:
            total_loss = total_loss.clone().detach().requires_grad_(True)

        return {"loss": total_loss, "loss_items": loss_items}

    def _forward_eval(self, images: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Evaluation forward pass.

        DetectionModel returns raw predictions that need to be decoded.
        The predictions are in the format of [P3, P4, P5] multi-scale outputs.
        """
        pred = self.model(images)

        # TODO: implement NMS and decoding if needed

        return pred

    def compute_loss(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: List[Dict[str, torch.Tensor]],
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

        formatted_preds, batch_dict = self.prepare_v8_loss_inputs(predictions, targets)
        # Compute loss using YOLOv5's loss function
        loss, loss_items = self.loss_fn(formatted_preds, batch_dict)

        # Return structured loss dict
        return loss, loss_items

    def prepare_v8_loss_inputs(self, preds, targets):
        """
        Convert YOLOv8 predictions and target annotations to the format expected by v8DetectionLoss.

        Args:
            preds (list[Tensor] or tuple): List of feature maps from the model or (None, feats)
                Each feature map: (B, nc + 4*reg_max, H, W)
            targets (list[dict] or Tensor): Ground truth for each image in the batch.
                If list of dict: each dict = {"boxes": (N,4), "labels": (N,)}
            batch_size (int): Number of images in the batch
            nc (int): Number of classes
            reg_max (int): DFL reg_max from the model

        Returns:
            formatted_preds: list of feature maps as expected by v8DetectionLoss
            batch_dict: dict with keys "batch_idx", "cls", "bboxes"
        """

        # ---- 1. Ensure preds is a list ----
        if isinstance(preds, tuple):
            feats = preds[1]  # ignore first element
        else:
            feats = preds

        formatted_preds = []
        for f in feats:
            # sanity check shape: (B, nc + 4*reg_max, H, W)
            formatted_preds.append(f)

        # ---- 2. Flatten targets into single tensor ----
        batch_idx_list = []
        cls_list = []
        bbox_list = []

        for b_idx, t in enumerate(targets):
            if isinstance(t, dict):
                boxes = t["boxes"]
                labels = t["class_labels"]
            else:
                # assume already tensor of shape (N,5) [cls, xyxy] or (N,6) [idx, cls, xyxy]
                boxes = t[:, 1:5]
                labels = t[:, 0].long()

            num_targets = boxes.shape[0]
            if num_targets == 0:
                continue

            batch_idx_list.append(
                torch.full((num_targets,), b_idx, dtype=torch.long, device=boxes.device)
            )
            cls_list.append(labels)
            bbox_list.append(boxes)

        if len(batch_idx_list):
            batch_idx = torch.cat(batch_idx_list, dim=0)
            cls = torch.cat(cls_list, dim=0)
            bboxes = torch.cat(bbox_list, dim=0)
        else:
            batch_idx = torch.tensor([], dtype=torch.long)
            cls = torch.tensor([], dtype=torch.long)
            bboxes = torch.tensor([], dtype=torch.float32).view(0, 4)

        batch_dict = {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

        return formatted_preds, batch_dict
