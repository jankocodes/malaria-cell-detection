import json
import torch
from torch.utils.data import DataLoader
from data.dataset import BloodCellDataset, DatasetType
import numpy as np
import cv2
import os
from yolov5.utils.augmentations import letterbox


def load_dataloader(
    data_path: str, dataset_type: DatasetType, img_size, batch_size, num_workers
):

    dataset = BloodCellDataset(
        annotations_file=f"{data_path}/{dataset_type.value}.json",
        img_dir=f"{data_path}/{dataset_type.value}",
        img_size=img_size,
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if dataset_type == DatasetType.TRAIN else False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return test_loader


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


def get_targets(img_ids, split, data_path):
    annotation_path = os.path.join(data_path, f"{split}.json")

    with open(annotation_path, "r") as f:
        annotations = json.load(f)["annotations"]

    targets = []
    for img_id in img_ids:
        anns = annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        targets.append(target)
    return targets


def load_sample_image(
    data_path: str,
    split: str,
    img_id: int,
    with_target: bool = False,
    img_size: int = 640,
    device: str | torch.device | None = None,
):
    """
    Load and preprocess a single image exactly like BloodCellDataset.

    Args:
        img_path (str): Path to the image file.
        img_size (int): Target image size for letterbox.
        device (optional): torch device to move the tensor to.

    Returns:
        img_tensor (torch.Tensor): Image tensor [1, 3, H, W], normalized to [0,1]
        meta (dict): Metadata useful for post-processing
            - original_shape
            - resized_shape
            - ratio
            - pad
    """

    img_path = os.path.join(data_path, split, f"{img_id}.png")
    assert os.path.exists(img_path), f"Image not found: {img_path}"
    # Read image (BGR)
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    original_shape = img.shape[:2]  # (h, w)

    # Letterbox (same as dataset)
    img_lb, ratio, (pad_w, pad_h) = letterbox(img, new_shape=img_size, stride=32)

    resized_shape = img_lb.shape[:2]

    # BGR → RGB, HWC → CHW
    img_lb = img_lb[:, :, ::-1].transpose(2, 0, 1).copy()

    # To tensor + normalize
    img_tensor = torch.from_numpy(img_lb).float() / 255.0

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    if device is not None:
        img_tensor = img_tensor.to(device)

    if with_target:
        pass

    return img_tensor
