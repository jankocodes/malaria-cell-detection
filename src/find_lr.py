import torch
from factory import ModelFactory
from data.dataset import BloodCellDataset, DatasetType
from torch.utils.data import DataLoader
from training.lr_finder import *
from data.util import collate_fn, load_dataloader
from training.util import *
from factory import ModelType
import argparse
import yaml
import os
import json


def main(cfg, data_path=None, model_dir=None):

    set_random_seed(42)

    base_cfg = cfg["base"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    model_type_str = model_cfg["type"]
    factory = ModelFactory(device=device, num_classes=base_cfg["num_classes"])
    model = factory.load(
        model_type=ModelType(model_type_str),
        pretrained=train_cfg["pretrained"],
        model_dir=model_dir,
    )

    # --- Load Data ---
    print("Loading datasets...")
    img_size = base_cfg.get("img_size", 640)  # Default to 640 if not specified

    train_loader = load_dataloader(
        data_path,
        dataset_type=DatasetType.TRAIN,
        img_size=img_size,
        batch_size=model_cfg["batch_size"],
        num_workers=base_cfg["num_workers"],
    )

    # --- Training Loop ---
    print("Finding learning rate...")

    base_lr = model_cfg.get("base_lr")  # Default to 0.001 if not specified

    if train_cfg["pretrained"]:  # lower LR for pretrained models
        base_lr /= 10

    min_lr = base_lr / 10
    max_lr = base_lr * 10

    # --- Optimizer ---
    optimizer = get_optimizer(
        model=model,
        model_type=ModelType(model_type_str),
        lr=min_lr,
    )

    lr_finder = find_lr(
        model=model,
        optimizer=optimizer,
        max_lr=max_lr,
        train_loader=train_loader,
    )

    # --- Log Results ---

    results = lr_finder.get_results()
    suggestion = lr_finder.lr_suggestion()
    print(f"Learning rate search completed. Results: {results}")
    print(f"Suggested LR: {suggestion}")

    # --- Save Results as JSON ---
    model_pretrained_str = "pretrained" if train_cfg["pretrained"] else "from_scratch"

    save_dir = os.path.join("results", "lr_finder", model_pretrained_str)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_type_str}_lr_finder_results.json")
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
    parser.add_argument(
        "--base_config", required=True, type=str, help="Path to base config file"
    )
    parser.add_argument(
        "--model_config", required=True, type=str, help="Path to model config file"
    )
    parser.add_argument(
        "--train_config", required=True, type=str, help="Path to train config file"
    )
    parser.add_argument("--data_path", type=str, help="Override dataset path")
    parser.add_argument("--model_dir", type=str, help="Override model directory path")

    args = parser.parse_args()

    cfg = load_config(
        base_config=args.base_config,
        model_config=args.model_config,
        train_config=args.train_config,
    )

    main(
        cfg,
        data_path=args.data_path,
        model_dir=args.model_dir,
    )
