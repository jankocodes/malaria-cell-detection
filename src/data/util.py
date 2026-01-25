import torch
from torch.utils.data import DataLoader
from data.dataset import BloodCellDataset


def load_train_val_loaders(data_path, img_size, batch_size, num_workers):
    train_dataset = BloodCellDataset(
        annotations_file=f"{data_path}/train.json",
        img_dir=f"{data_path}/train",
        img_size=img_size,
    )

    val_dataset = BloodCellDataset(
        annotations_file=f"{data_path}/val.json",
        img_dir=f"{data_path}/val",
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


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
