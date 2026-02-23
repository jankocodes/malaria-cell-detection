import json
import torch
from torch.utils.data import DataLoader
from data.dataset import BloodCellDataset, DatasetType
import numpy as np
import cv2
import os
from yolov5.utils.augmentations import letterbox
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
        data = json.load(f)
        annotations_list = data["annotations"]

    # Create mapping from image_id to annotations
    annotations_by_id = {}
    for ann in annotations_list:
        img_id = ann["image_id"]
        if img_id not in annotations_by_id:
            annotations_by_id[img_id] = []
        annotations_by_id[img_id].append(ann)

    targets = []
    for img_id in img_ids:
        anns = annotations_by_id.get(img_id + 1, [])
        boxes = []
        labels = []
        for ann in anns:
            boxes.append(ann["bbox"])
            # Convert category_id from 1-indexed to 0-indexed (matching BloodCellDataset behavior)
            labels.append(ann["category_id"] - 1)
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
        data_path (str): Path to the dataset.
        split (str): Dataset split ('train' or 'val').
        img_id (int): Image ID to load.
        with_target (bool): If True, also load targets for the image.
        img_size (int): Target image size for letterbox.
        device (optional): torch device to move the tensor to.

    Returns:
        img_tensor (torch.Tensor): Image tensor [1, 3, H, W], normalized to [0,1]
        If with_target=True, also returns:
            targets (dict): Dictionary with 'boxes' and 'labels' tensors
            meta (dict): Metadata useful for post-processing
                - original_shape
                - resized_shape
                - ratio
                - pad
    """
    # Load annotations to get the correct file_name
    annotation_path = os.path.join(data_path, f"{split}.json")
    with open(annotation_path, "r") as f:
        data = json.load(f)

    # Find the image entry for this img_id
    file_name = None
    for img_info in data["images"]:
        if img_info["id"] == img_id:
            file_name = img_info["file_name"]
            break

    assert file_name is not None, f"Image with id {img_id} not found in {split} split"

    img_path = os.path.join(data_path, split, file_name)
    assert os.path.exists(img_path), f"Image file not found: {img_path}"

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
        targets = get_targets([img_id], split, data_path)
        return (
            img_tensor,
            targets[0],
            {
                "original_shape": original_shape,
                "resized_shape": resized_shape,
                "ratio": ratio,
                "pad": (pad_w, pad_h),
            },
        )

    return img_tensor


def visualize_sample(
    data_path: str,
    split: str,
    img_id: int,
    without_target: bool = True,
    rare_only: bool = False,
    show_letterbox: bool = False,
    save_path: str = None,
):
    """
    Visualize an image with its bounding box targets.

    Args:
        data_path: Path to the dataset
        split: Dataset split ('train' or 'val')
        img_id: Image ID to visualize
        show_letterbox: If True, show the letterbox-resized version; if False, show original
        save_path: If provided, save the visualization to this path
    """
    # Load annotations to get class names and file info
    annotation_path = os.path.join(data_path, f"{split}.json")
    with open(annotation_path, "r") as f:
        data = json.load(f)

    # Get class names (convert from 1-indexed to 0-indexed for mapping)
    class_names = {cat["id"] - 1: cat["name"] for cat in data["categories"]}

    # Find the image file name
    file_name = None
    for img_info in data["images"]:
        if img_info["id"] == img_id:
            file_name = img_info["file_name"]
            break

    # Load image and targets
    img_tensor, targets, meta = load_sample_image(
        data_path=data_path,
        split=split,
        img_id=img_id,
        with_target=True,
        img_size=640,
    )

    boxes = targets["boxes"]  # [N, 4] in xywh format
    labels = targets["labels"]  # [N]

    # Denormalize image tensor to [0, 255]
    # img_tensor shape: [1, 3, H, W]
    img_np = (img_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype("uint8")
    # Convert RGB back to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    if show_letterbox:
        # Use the letterbox-resized image
        display_img = img_bgr

        # Transform boxes to letterbox space using the metadata
        ratio = meta["ratio"]  # (ratio_x, ratio_y)
        pad_w, pad_h = meta["pad"]

        display_boxes = boxes.clone()
        display_boxes[:, [0, 2]] *= ratio[0]  # Scale x and width
        display_boxes[:, [1, 3]] *= ratio[1]  # Scale y and height
        display_boxes[:, 0] += pad_w  # Shift x
        display_boxes[:, 1] += pad_h  # Shift y

        title_suffix = "(Letterbox 640x640)"
    else:
        # Load original image to match box coordinates
        img_path = os.path.join(data_path, split, f"{img_id}.png")
        display_img = cv2.imread(img_path)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        display_boxes = boxes
        title_suffix = "(Original Image)"

    # Convert boxes from xywh to xyxy format for drawing
    boxes_xyxy = (
        display_boxes.clone()
        if torch.is_tensor(display_boxes)
        else torch.tensor(display_boxes)
    )
    boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]  # x2 = x + w
    boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]  # y2 = y + h

    # Create figure and plot
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(display_img)
    # ax.set_title(f"Image {img_id} - Split: {split} {title_suffix}")
    ax.axis("off")

    # Draw bounding boxes
    colors = plt.cm.tab10(range(10))
    for i, (box, label) in enumerate(zip(boxes_xyxy, labels)):
        if (rare_only and label.item() == 0) or without_target:
            continue
        x1, y1, x2, y2 = box.tolist()
        w = x2 - x1
        h = y2 - y1

        # Draw rectangle
        color = "red" if rare_only else colors[label.item() % 10]
        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=4,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

        # Add label text with class name
        class_label = label.item()
        class_name = class_names.get(class_label, f"Unknown ({class_label})")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    # Print target information
    print(f"\nImage ID: {img_id}")
    print(f"File: {file_name}")
    print(f"Split: {split}")
    print(f"Number of objects: {len(boxes)}")
    if len(boxes) > 0:
        print(f"\nBounding boxes (xywh):")
        for i, (box, label) in enumerate(zip(boxes, labels)):
            class_id = label.item()
            class_name = class_names.get(class_id, f"Unknown ({class_id})")
            print(f"  Object {i}: bbox={box.tolist()}, class={class_id} ({class_name})")
