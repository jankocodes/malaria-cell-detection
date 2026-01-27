import torch
import numpy as np
from typing import List, Dict, Tuple


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes.

    Args:
        boxes1: Tensor of shape [N, 4] in format [x1, y1, x2, y2]
        boxes2: Tensor of shape [M, 4] in format [x1, y1, x2, y2]

    Returns:
        iou: Tensor of shape [N, M] containing pairwise IoU values
    """
    # Calculate intersection area
    x1_max = torch.max(boxes1[:, None, 0], boxes2[:, 0])  # [N, M]
    y1_max = torch.max(boxes1[:, None, 1], boxes2[:, 1])  # [N, M]
    x2_min = torch.min(boxes1[:, None, 2], boxes2[:, 2])  # [N, M]
    y2_min = torch.min(boxes1[:, None, 3], boxes2[:, 3])  # [N, M]

    intersection_w = (x2_min - x1_max).clamp(min=0)
    intersection_h = (y2_min - y1_max).clamp(min=0)
    intersection = intersection_w * intersection_h

    # Calculate union area
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]
    union = area1[:, None] + area2[None, :] - intersection

    # Calculate IoU
    iou = intersection / union.clamp(min=1e-6)

    return iou


def match_predictions(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Match predictions to ground truth boxes at a given IoU threshold.

    Args:
        pred_boxes: Predicted boxes [N, 4] in format [x1, y1, x2, y2]
        pred_scores: Prediction confidence scores [N]
        pred_labels: Predicted class labels [N]
        gt_boxes: Ground truth boxes [M, 4] in format [x1, y1, x2, y2]
        gt_labels: Ground truth class labels [M]
        iou_threshold: IoU threshold for considering a match

    Returns:
        matches: Binary array [N] indicating if each prediction is a true positive
        scores: Confidence scores [N] sorted by score
        num_gt: Number of ground truth boxes for this class
    """
    if pred_boxes.numel() == 0:
        return np.array([]), np.array([]), gt_boxes.shape[0]

    # Sort predictions by score (descending)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    pred_labels = pred_labels[sorted_indices]

    matches = np.zeros(len(pred_boxes), dtype=bool)

    if gt_boxes.numel() == 0:
        return matches, pred_scores.cpu().numpy(), 0

    # Calculate IoU between all predictions and ground truths
    ious = box_iou(pred_boxes, gt_boxes)  # [N, M]

    # Track which ground truths have been matched
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    # Match each prediction to ground truth
    for i in range(len(pred_boxes)):
        # Find best matching ground truth for this prediction
        if ious[i].numel() == 0:
            continue

        best_iou, best_gt_idx = ious[i].max(dim=0)
        best_gt_idx = best_gt_idx.item()
        best_iou = best_iou.item()

        # Check if IoU exceeds threshold and labels match
        if (
            best_iou >= iou_threshold
            and pred_labels[i] == gt_labels[best_gt_idx]
            and not gt_matched[best_gt_idx]
        ):
            matches[i] = True
            gt_matched[best_gt_idx] = True

    return matches, pred_scores.cpu().numpy(), len(gt_boxes)


def compute_ap(matches: np.ndarray, scores: np.ndarray, num_gt: int) -> float:
    """
    Compute Average Precision (AP) from matches and scores.

    Args:
        matches: Binary array indicating true positives [N]
        scores: Confidence scores [N]
        num_gt: Total number of ground truth boxes

    Returns:
        ap: Average Precision value
    """
    if num_gt == 0:
        return 0.0

    if len(matches) == 0:
        return 0.0

    # Sort by confidence score (should already be sorted, but ensure)
    sort_idx = np.argsort(-scores)
    matches = matches[sort_idx]

    # Compute cumulative true positives and false positives
    tp_cumsum = np.cumsum(matches)
    fp_cumsum = np.cumsum(~matches)

    # Compute precision and recall at each threshold
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / num_gt

    # Add sentinel values at the end
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision is monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Calculate AP as area under P-R curve
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])

    return float(ap)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format.

    Args:
        boxes: Tensor of shape [N, 4] in format [x, y, w, h]

    Returns:
        boxes: Tensor of shape [N, 4] in format [x1, y1, x2, y2]
    """
    boxes_xyxy = boxes.clone()
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2 = x + w
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2 = y + h
    return boxes_xyxy


def calculate_ap(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate Average Precision at a fixed IoU threshold.

    Args:
        predictions: List of prediction dicts per image with 'boxes', 'scores', 'labels'
                     boxes in [x1, y1, x2, y2] format
        targets: List of target dicts per image with 'boxes', 'class_labels'
                 boxes in [x, y, w, h] format (COCO format)
        num_classes: Total number of classes
        iou_threshold: IoU threshold for considering a match (default: 0.5)

    Returns:
        Dictionary containing:
            - 'mAP': Mean Average Precision across all classes
            - 'AP_class_i': Average Precision for each class i
    """
    # Accumulate predictions and ground truths per class
    all_pred_boxes = {c: [] for c in range(num_classes)}
    all_pred_scores = {c: [] for c in range(num_classes)}
    all_gt_boxes = {c: [] for c in range(num_classes)}
    all_gt_counts = {c: 0 for c in range(num_classes)}

    # Collect all predictions and ground truths across images
    for pred, target in zip(predictions, targets):
        pred_boxes = pred["boxes"].cpu()
        pred_scores = pred["scores"].cpu()
        pred_labels = pred["labels"].cpu()

        # Convert ground truth boxes from [x, y, w, h] to [x1, y1, x2, y2]
        gt_boxes = target["boxes"].cpu()
        if gt_boxes.numel() > 0:
            gt_boxes = xywh_to_xyxy(gt_boxes)
        gt_labels = target["class_labels"].cpu()

        # Group by class
        for c in range(num_classes):
            # Predictions for this class
            pred_mask = pred_labels == c
            if pred_mask.any():
                all_pred_boxes[c].append(pred_boxes[pred_mask])
                all_pred_scores[c].append(pred_scores[pred_mask])

            # Ground truths for this class
            gt_mask = gt_labels == c
            if gt_mask.any():
                all_gt_boxes[c].append(gt_boxes[gt_mask])
                all_gt_counts[c] += gt_mask.sum().item()

    # Calculate AP for each class
    ap_per_class = {}
    valid_classes = []

    for c in range(num_classes):
        # Concatenate all predictions and ground truths for this class
        if len(all_pred_boxes[c]) > 0:
            pred_boxes = torch.cat(all_pred_boxes[c], dim=0)
            pred_scores = torch.cat(all_pred_scores[c], dim=0)
        else:
            pred_boxes = torch.empty((0, 4))
            pred_scores = torch.empty((0,))

        # For matching, we need to process image by image
        # But for AP calculation, we need global sorting
        all_matches = []
        all_scores = []

        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            pred_boxes_img = pred["boxes"].cpu()
            pred_scores_img = pred["scores"].cpu()
            pred_labels_img = pred["labels"].cpu()

            # Convert ground truth boxes from [x, y, w, h] to [x1, y1, x2, y2]
            gt_boxes_img = target["boxes"].cpu()
            if gt_boxes_img.numel() > 0:
                gt_boxes_img = xywh_to_xyxy(gt_boxes_img)
            gt_labels_img = target["class_labels"].cpu()

            # Filter by class
            pred_mask = pred_labels_img == c
            gt_mask = gt_labels_img == c

            if pred_mask.any() or gt_mask.any():
                matches, scores, _ = match_predictions(
                    pred_boxes_img[pred_mask],
                    pred_scores_img[pred_mask],
                    torch.full_like(pred_scores_img[pred_mask], c, dtype=torch.long),
                    gt_boxes_img[gt_mask],
                    torch.full_like(gt_labels_img[gt_mask], c, dtype=torch.long),
                    iou_threshold,
                )

                if len(matches) > 0:
                    all_matches.append(matches)
                    all_scores.append(scores)

        # Compute AP for this class
        # Compute AP for this class
        num_gt = all_gt_counts[c]

        # No ground truth → AP undefined (will be ignored in mAP)
        if num_gt == 0:
            ap = float("nan")

        # Ground truth exists but no predictions → AP = 0
        elif len(all_matches) == 0:
            ap = 0.0

        # Normal case
        else:
            matches_class = np.concatenate(all_matches)
            scores_class = np.concatenate(all_scores)
            ap = compute_ap(matches_class, scores_class, num_gt)

        ap_per_class[f"AP_class_{c}"] = ap

        # Only include classes that have ground truth data in mAP
        if all_gt_counts[c] > 0:
            valid_classes.append(ap)

    # Calculate mean AP across classes with ground truth
    if len(valid_classes) > 0:
        mAP = float(np.mean(valid_classes))
    else:
        mAP = 0.0

    results = {"mAP": mAP, **ap_per_class}

    return results
