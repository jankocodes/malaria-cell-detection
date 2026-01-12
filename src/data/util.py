import torch


def collate_fn(batch):
    images, targets = zip(*batch)  # unzip
    return torch.stack(images), list(targets)


def clamp_boxes_xyxy(boxes, image_size):
    img_w, img_h = image_size
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(0, img_w)  # x1
    boxes[:, 2] = boxes[:, 2].clamp(0, img_w)  # x2
    boxes[:, 1] = boxes[:, 1].clamp(0, img_h)  # y1
    boxes[:, 3] = boxes[:, 3].clamp(0, img_h)  # y2
    return boxes
