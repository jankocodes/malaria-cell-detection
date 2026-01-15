import torch
from factory import ModelFactory
from data.dataset import BloodCellDataset
from torch.utils.data import DataLoader
from training.lr_finder import find_lr
from data.util import collate_fn  #
from training.util import freeze_detr_backbone
from factory import ModelType
import argparse
import yaml


def main(cfg, data_path=None, show_progress=True, model_dir=None):

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

    if model_cfg["type"] == "detr":
        freeze_detr_backbone(model)
        print("✅ Frozen DETR backbone for LR finder.")

    # --- Load Data ---
    print("Loading datasets...")
    img_size = hyp.get("img_size", 640)  # Default to 640 if not specified
    train_dataset = BloodCellDataset(
        annotations_file=f"{data_path}/train.json",
        img_dir=f"{data_path}/train",
        img_size=img_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyp["batch_size"],
        shuffle=True,
        num_workers=hyp["num_workers"],
        collate_fn=collate_fn,
    )

    # --- Training Loop ---
    print("Finding learning rate...")

    result = find_lr(
        model=model,
        min_lr=hyp["min_lr"],
        max_lr=hyp["max_lr"],
        train_loader=train_loader,
        show_progress=show_progress,
    )

    # --- Log Results and Save Plot ---
    print(f"Learning rate search completed. Results: {result.get_results()}")
    print(f"Suggested LR: {result.lr_suggestion()}")

    if hasattr(result, "plot"):
        result.plot()
        import matplotlib.pyplot as plt

        plt.savefig(
            f"lr_finder_plot_{model_cfg['type']}.png", dpi=150, bbox_inches="tight"
        )
        print("Plot saved as 'lr_finder_plot.png'")


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
