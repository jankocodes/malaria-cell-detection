import torch
from factory import ModelFactory
from data.dataset import BloodCellDataset
from torch.utils.data import DataLoader
from training.lr_finder import *
from data.util import collate_fn
from training.util import *
from factory import ModelType
import argparse
import yaml
import os
import json


def main(cfg, data_path=None, show_progress=True, model_dir=None):

    set_random_seed(42)

    hyp = cfg["hyp"]
    model_cfg = cfg["model"]
    data_path = data_path if data_path is not None else cfg["data_path"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    model_type_str = model_cfg["type"]
    factory = ModelFactory(device=device, num_classes=model_cfg["num_classes"])
    model = factory.load(
        model_type=ModelType(model_type_str),
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=hyp["batch_size"],
        shuffle=True,
        num_workers=hyp["num_workers"],
        collate_fn=collate_fn,
    )

    # --- Training Loop ---
    print("Finding learning rate...")

    base_lr = hyp.get("base_lr")  # Default to 0.001 if not specified

    if model_cfg["pretrained"]:  # lower LR for pretrained models
        base_lr /= 10

    min_lr = base_lr / 10
    max_lr = base_lr * 10

    # --- Optimizer ---
    optimizer = get_optimizer(
        model=model,
        model_type=ModelType(model_cfg["type"]),
        lr=min_lr,
    )

    lr_finder = find_lr(
        model=model,
        optimizer=optimizer,
        max_lr=max_lr,
        train_loader=train_loader,
        show_progress=show_progress,
    )

    # --- Log Results ---

    results = lr_finder.get_results()
    suggestion = lr_finder.lr_suggestion()
    print(f"Learning rate search completed. Results: {results}")
    print(f"Suggested LR: {suggestion}")

    # --- Save Results as JSON ---
    model_pretrained_str = "pretrained" if model_cfg["pretrained"] else "from_scratch"

    save_dir = os.path.join("results", "lr_finder", model_pretrained_str)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_type_str }_lr_finder_results.json")
    with open(save_path, "w") as f:
        json.dump({"results": results, "suggested_lr": suggestion}, f, indent=2)

    # --- Plot Results ---
    save_and_plot_lr_finder_results(
        lr_finder,
        model_name=model_type_str,
        range=(min_lr, max_lr),
        model_pretrained_str=model_pretrained_str,
    )


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
