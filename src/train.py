import torch
from tqdm import tqdm
from factory import ModelFactory
from data.dataset import BloodCellDataset
from torch.utils.data import DataLoader
from yolov5.utils.loss import ComputeLoss
from od_utils.training import train_one_epoch
from od_utils.data import collate_fn


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DATASET_PATH = "data/raw/vogelbacher23/dataset_segmentation"


if __name__ == "__main__":

    # --- Load Model ---
    model = ModelFactory().load_yolo_v5(pretrained=True, device=DEVICE)

    # --- Load Data ---
    train_dataset = BloodCellDataset(
        annotations_file=f"{DATASET_PATH}/train.json",
        img_dir=f"{DATASET_PATH}/train",
    )

    val_dataset = BloodCellDataset(
        annotations_file=f"{DATASET_PATH}/val.json",
        img_dir=f"{DATASET_PATH}/val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        result = train_one_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epoch=epoch,
        )
        print(result)
