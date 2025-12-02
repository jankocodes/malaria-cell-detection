import os
import json
from torch.utils.data import Dataset
import cv2
import torch
from yolov5.utils.augmentations import letterbox


def default_transform(img):
    # 1️⃣ Resize and pad to model input
    img = letterbox(img, new_shape=640, stride=32)[0]

    # 2️⃣ BGR -> RGB, HWC -> CHW
    img = (
        img[:, :, ::-1].transpose(2, 0, 1).copy()
    )  # ⚠ make a copy to avoid negative strides

    # 3️⃣ Convert to float tensor and scale 0-1
    img = torch.from_numpy(img).float() / 255.0

    return img


class BloodCellDataset(Dataset):
    """
    BloodCellDataset class for loading blood cell images and their annotations.

    This class extends PyTorch's Dataset class to load images and corresponding
    bounding box annotations from a JSON file. It's designed to work with datasets
    in COCO format.

    Attributes:
        annotations (dict): Dictionary containing image metadata and annotations loaded from JSON.
        images (list): List of image metadata dictionaries.
        bounding_boxes (dict): Mapping from image_id to list of annotation dictionaries.
        img_dir (str): Path to directory containing the image files.
        transform (callable, optional): Function to apply transformations to images.
        target_transform (callable, optional): Function to apply transformations to labels/annotations.

    Args:
        annotations_file (str): Path to the JSON file containing COCO format annotations.
        img_dir (str): Path to the directory containing the image files.
        transform (callable, optional): Optional transform to be applied on images. Default: None
        target_transform (callable, optional): Optional transform to be applied on labels. Default: None

    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Returns the image and its corresponding bounding box annotations at index idx.

    Returns:
        tuple: (image, label) where image is the loaded image array and label is the list of annotations.
    """

    def __init__(
        self,
        annotations_file,
        img_dir,
        transform=default_transform,
        target_transform=None,
    ):
        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

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
        for i, label in enumerate(self.unique_labels):
            self.label_mapping[label] = i

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx]["file_name"])
        image = cv2.imread(img_path)
        img_id = self.images[idx]["id"]  # unequal to idx (idx=0 -> img_id=1)
        target = self.target[img_id]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)

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
