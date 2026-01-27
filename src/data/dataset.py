from enum import Enum
import os
import json
from torch.utils.data import Dataset
import cv2
import torch
from yolov5.utils.augmentations import letterbox
import numpy as np
from copy import deepcopy


class DatasetType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class BloodCellDataset(Dataset):
    """
    BloodCellDataset class for loading blood cell images and their annotations.

    This class extends PyTorch's Dataset class to load images and corresponding
    bounding box annotations from a JSON file. It's designed to work with datasets
    in COCO format. Images and targets are transformed using letterbox resizing.

    Attributes:
        annotations (dict): Dictionary containing image metadata and annotations loaded from JSON.
        images (list): List of image metadata dictionaries.
        bounding_boxes (dict): Mapping from image_id to list of annotation dictionaries.
        img_dir (str): Path to directory containing the image files.
        img_size (int): Target size for letterbox transformation.

    Args:
        annotations_file (str): Path to the JSON file containing COCO format annotations.
        img_dir (str): Path to the directory containing the image files.
        img_size (int, optional): Target image size for letterbox transformation. Default: 640

    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Returns the image and its corresponding bounding box annotations at index idx.

    Returns:
        tuple: (image, target) where image is the transformed image tensor and target contains
               adjusted bounding boxes and labels.
    """

    def __init__(
        self,
        annotations_file,
        img_dir,
        img_size=640,
    ):
        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.img_size = img_size

        self.images = self.annotations["images"]
        self.target = dict()
        self.annotation_categories = {"bbox", "category_id", "iscrowd", "area"}
        self.unique_labels = set()
        self.label_mapping = dict()

        # Create a mapping from image_id to its annotations
        for annotation in self.annotations["annotations"]:
            image_id = annotation["image_id"]

            # Initialize target entry if not present
            if image_id not in self.target:
                self.target[image_id] = dict()
                for cat in self.annotation_categories:
                    self.target[image_id][cat] = []

            # Append annotation details to the corresponding image_id
            for cat in self.annotation_categories:
                self.target[image_id][cat].append(annotation[cat])

            self.unique_labels.add(annotation["category_id"])

        # Create class_id mapping
        for label in self.unique_labels:
            self.label_mapping[label] = label - 1
        print(self.label_mapping)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx]["file_name"])
        image = cv2.imread(img_path)
        img_id = self.images[idx]["id"]  # unequal to idx (idx=0 -> img_id=1)
        target = deepcopy(self.target[img_id])  # Deep copy to avoid modifying original

        # Apply letterbox transformation to both image and targets
        image, target = letterbox_transform(image, target, self.img_size)

        # Convert target to tensor format
        target = {
            "boxes": torch.as_tensor(target["bbox"], dtype=torch.float32),
            "class_labels": torch.as_tensor(
                [self.label_mapping.get(id) for id in target["category_id"]],
                dtype=torch.int64,
            ),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(target["area"], dtype=torch.float32),
            "iscrowd": torch.as_tensor(target["iscrowd"], dtype=torch.int64),
        }

        return image, target


from yolov5.utils.augmentations import letterbox


def letterbox_transform(img, targets, img_size=640):
    """
    Apply letterbox transformation to image and adjust targets accordingly.

    Args:
        img: Input image (numpy array in BGR format)
        targets: Dictionary containing bounding boxes in COCO format [x, y, w, h]
        img_size: Target image size (default: 640)

    Returns:
        transformed_img: Transformed image tensor
        transformed_targets: Targets with adjusted bounding boxes
    """
    # Store original image dimensions
    original_h, original_w = img.shape[:2]

    # 1️⃣ Resize and pad to model input using letterbox
    # letterbox returns (image, ratio, (pad_w, pad_h))
    img_letterboxed, ratio, (pad_w, pad_h) = letterbox(
        img, new_shape=img_size, stride=32
    )

    # 2️⃣ Transform bounding boxes
    if len(targets["bbox"]) > 0:
        boxes = np.array(
            targets["bbox"], dtype=np.float32
        ).copy()  # [x, y, w, h] format

        # Scale boxes by the resize ratio
        boxes[:, [0, 2]] *= ratio[0]  # Scale x and width
        boxes[:, [1, 3]] *= ratio[1]  # Scale y and height

        # Shift boxes by padding offsets
        boxes[:, 0] += pad_w  # Shift x
        boxes[:, 1] += pad_h  # Shift y

        targets["bbox"] = boxes.tolist()

    # 3️⃣ BGR -> RGB, HWC -> CHW
    img_letterboxed = (
        img_letterboxed[:, :, ::-1].transpose(2, 0, 1).copy()
    )  # ⚠ make a copy to avoid negative strides

    # 4️⃣ Convert to float tensor and scale 0-1
    img_letterboxed = torch.from_numpy(img_letterboxed).float() / 255.0

    return img_letterboxed, targets
