from typing import List, Dict, Tuple, Optional
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression

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

            # YOLOv8 loss expects model.args to be an object with .box, .cls, .dfl attributes
            class HParams:
                def __init__(self, box=7.5, cls=0.5, dfl=1.5):
                    self.box = box
                    self.cls = cls
                    self.dfl = dfl

            # Replace model.args with HParams object
            # (when loading from YOLO wrapper, model.args is a dict which doesn't work)
            self.model.args = HParams()

            self.loss_fn = v8DetectionLoss(self.model)
        except ImportError:
            print("Warning: Could not import YOLOv8 loss. Set loss_fn manually.")
            self.loss_fn = None

    def _set_num_classes(self, num_classes: int):

        # YOLOv8 number of classes are adapted in the factory class

        print(f"✅ Updated YOLOv8 model to {num_classes} classes )")

    def _forward(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]] = None,
        return_predictions: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass — computes loss and/or predictions depending on arguments.

        YOLOv8's DetectionModel returns raw feature maps in train mode and decoded
        predictions in eval mode, so loss and prediction passes use different modes.
        Both are encapsulated here; the training loop never needs to switch modes.

        Args:
            images: [B, 3, H, W]
            targets: list of per-image dicts. When provided, loss is computed in
                     train mode.
            return_predictions: when True, post-NMS predictions are included.

        Returns:
            Dict with any combination of 'loss' and 'predictions'.
        """
        result = {}

        if targets is not None:
            # Loss requires train-mode feature maps from the model.
            was_training = self.model.training
            if not was_training:
                self.model.train()

            pred = self.model(images)

            if not was_training:
                self.model.eval()

            total_loss, _ = self.compute_loss(pred, targets, images.shape[2:])
            result["loss"] = total_loss

        if return_predictions or targets is None:
            # Predictions require eval-mode decoded outputs.
            decoded_preds = self._predict(
                images, post_processing=False
            )  # get raw predictions without NMS

            nms_out = non_max_suppression(
                decoded_preds, conf_thres=0.005, iou_thres=0.5
            )
            predictions = self._format_predictions(nms_out, images.device)

            result["predictions"] = predictions

        return result

    def _predict(
        self,
        images: torch.Tensor,
        post_processing: bool = True,
        conf_thresh=0.005,
        iou_thresh=0.5,
    ):
        """
        Prediction method that ensures eval-mode inference regardless of wrapper state.

        Returns YOLO-style predictions:
        - boxes: xyxy (pixels)
        - scores: confidence scores
        - labels: class indices
        """
        was_training = self.model.training
        if was_training:
            self.model.eval()

        with torch.no_grad():
            predictions = self.model(images)

        if was_training:
            self.model.train()

        if post_processing:
            nms_out = non_max_suppression(
                predictions,
                conf_thres=conf_thresh,
                iou_thres=iou_thresh,
            )
            predictions = self._format_predictions(nms_out, images.device)

        return predictions

    def _format_predictions(self, nms_out, device):
        outputs = []
        for p in nms_out:
            if p is None or len(p) == 0:
                outputs.append(
                    {
                        "boxes": torch.empty((0, 4), device=device),
                        "scores": torch.empty((0,), device=device),
                        "labels": torch.empty((0,), device=device, dtype=torch.long),
                    }
                )
            else:
                outputs.append(
                    {
                        "boxes": p[:, :4],
                        "scores": p[:, 4],
                        "labels": p[:, 5].long(),
                    }
                )
        return outputs

    def compute_loss(
        self,
        predictions: Tuple[torch.Tensor, ...],
        targets: List[Dict[str, torch.Tensor]],
        img_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute YOLOv8 loss separately from forward pass.

        Args:
            predictions: Output from forward() - tuple of prediction tensors
            targets: List of target dicts with:
                - 'boxes': [N, 4] in format [x, y, w, h] (COCO format, pixel coords)
                - 'class_labels': [N] class indices
            img_size: (height, width) of input images

        Returns:
            Tuple of (total_loss, loss_items)
        """
        if self.loss_fn is None:
            raise ValueError("Loss function not initialized. Set loss_fn in __init__")

        formatted_preds, batch_dict = self.prepare_v8_loss_inputs(
            predictions, targets, img_size
        )
        # Compute loss using YOLOv8's loss function
        # Returns: loss tensor with [box_loss, cls_loss, dfl_loss] and detached loss_items
        loss, loss_items = self.loss_fn(formatted_preds, batch_dict)

        # Sum the loss components to get scalar loss for backpropagation
        total_loss = loss.sum()

        return total_loss, loss_items

    def prepare_v8_loss_inputs(self, preds, targets, img_size):
        """
        Convert YOLOv8 predictions and target annotations to the format expected by v8DetectionLoss.

        Args:
            preds (list[Tensor] or tuple): List of feature maps from the model or (None, feats)
                Each feature map: (B, nc + 4*reg_max, H, W)
            targets (list[dict]): Ground truth for each image in the batch.
                Each dict = {"boxes": (N,4) in [x, y, w, h] COCO format (pixels), "class_labels": (N,)}
            img_size (tuple): (height, width) of input images

        Returns:
            formatted_preds: list of feature maps as expected by v8DetectionLoss
            batch_dict: dict with keys "batch_idx", "cls", "bboxes" (normalized xywh)
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

        h, w = img_size

        for b_idx, t in enumerate(targets):
            if isinstance(t, dict):
                boxes = t["boxes"]  # [N, 4] in COCO format [x, y, w, h] pixels
                labels = t["class_labels"]
            else:
                # assume already tensor of shape (N,5) [cls, xyxy] or (N,6) [idx, cls, xyxy]
                boxes = t[:, 1:5]
                labels = t[:, 0].long()

            num_targets = boxes.shape[0]
            if num_targets == 0:
                continue

            # Convert COCO format [x, y, w, h] (pixels) to normalized [x_center, y_center, w, h]
            boxes = boxes.clone()
            boxes[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) / w  # x_center normalized
            boxes[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / h  # y_center normalized
            boxes[:, 2] = boxes[:, 2] / w  # width normalized
            boxes[:, 3] = boxes[:, 3] / h  # height normalized

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
            batch_idx = torch.tensor([], dtype=torch.long, device=self.device)
            cls = torch.tensor([], dtype=torch.long, device=self.device)
            bboxes = torch.tensor([], dtype=torch.float32, device=self.device).view(
                0, 4
            )

        batch_dict = {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

        return formatted_preds, batch_dict
