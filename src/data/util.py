import torch


def collate_fn(batch):
    images, targets = zip(*batch)  # unzip
    return torch.stack(images), list(targets)


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


def format_batch_targets(targets, img_size, device):
    """
    Format batch targets into a unified tensor representation for object detection.

    Converts target annotations from a list of dictionaries containing boxes and labels
    into a single tensor where each row represents an object with its metadata.

    """
    formatted = []

    for i, t in enumerate(targets):

        if len(t["boxes"]) == 0:
            continue

        boxes = t["boxes"]
        labels = t["labels"].unsqueeze(1)
        img_idx = torch.full((len(labels), 1), i)

        xywh = xyxy_to_xywh_norm(boxes, img_size, device)
        merged = torch.cat([img_idx, labels.to(device), xywh.to(device)], dim=1)

        formatted.append(merged)

    return torch.cat(formatted, dim=0) if len(formatted) else None
