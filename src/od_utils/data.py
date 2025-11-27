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
