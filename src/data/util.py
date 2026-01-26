import torch
from torch.utils.data import DataLoader
from data.dataset import BloodCellDataset, DatasetType


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
