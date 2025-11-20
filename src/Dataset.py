# TODO
import os
import json
from torch.utils.data import Dataset
import cv2


class BloodCellDataset(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)
        self.img_labels = self.annotations["images"]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx]["file_name"])
        image = cv2.imread(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
