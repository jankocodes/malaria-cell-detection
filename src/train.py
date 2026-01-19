import torch
from factory import ModelFactory
from data.dataset import BloodCellDataset
from torch.utils.data import DataLoader
from training.util import *
from data.util import collate_fn
from factory import ModelType
import argparse
import yaml


def main(cfg, data_path=None, show_progress=True, model_dir=None):
    set_random_seed(42)

    hyp = cfg["hyp"]
    model_cfg = cfg["model"]
    data_path = data_path if data_path is not None else cfg["data_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    factory = ModelFactory(device=device, num_classes=model_cfg["num_classes"])
    model = factory.load(
        model_type=ModelType(model_cfg["type"]),
        pretrained=model_cfg["pretrained"],
        model_dir=model_dir,
    )

    # --- Load Data ---
    print("Loading datasets...")
    img_size = hyp.get("img_size", 640)  # Default to 640 if not specified
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
        batch_size=hyp["batch_size"],
        shuffle=True,
        num_workers=hyp["num_workers"],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=hyp["batch_size"],
        shuffle=False,
        num_workers=hyp["num_workers"],
        collate_fn=collate_fn,
    )

    # --- Optimizer ---
    optimizer = get_optimizer(
        model=model,
        model_type=ModelType(model_cfg["type"]),
        lr=hyp.get("lr"),
    )

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(hyp["num_epochs"]):
        result = train_one_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epoch=epoch,
            show_progress=show_progress,
        )
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config file")
    parser.add_argument("--data_path", type=str, help="Override dataset path")
    parser.add_argument(
        "--show_progress", type=bool, default=True, help="Override progress flag"
    )
    parser.add_argument("--model_dir", type=str, help="Override model directory path")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(
        cfg,
        data_path=args.data_path,
        show_progress=args.show_progress,
        model_dir=args.model_dir,
    )
