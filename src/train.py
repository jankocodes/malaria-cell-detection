import torch
from factory import ModelFactory
from data.dataset import BloodCellDataset
from torch.utils.data import DataLoader
from training.util import train_one_epoch
from data.util import collate_fn
from factory import ModelType
import argparse
import yaml


def main(cfg):

    hyp = cfg["hyp"]
    model_cfg = cfg["model"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    factory = ModelFactory(device=device, num_classes=model_cfg["num_classes"])
    model = factory.load(
        model_type=ModelType(model_cfg["type"]),
        pretrained=model_cfg["pretrained"],
    )

    # --- Load Data ---
    print("Loading datasets...")
    train_dataset = BloodCellDataset(
        annotations_file=f"{cfg['data_path']}/train.json",
        img_dir=f"{cfg['data_path']}/train",
    )

    val_dataset = BloodCellDataset(
        annotations_file=f"{cfg['data_path']}/val.json",
        img_dir=f"{cfg['data_path']}/val",
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyp["learning_rate"])

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(hyp["num_epochs"]):
        result = train_one_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            epoch=epoch,
            show_progress=cfg["show_progress"],
        )
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
