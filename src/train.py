import torch
from factory import ModelFactory
from training.util import *
from data.util import *
from factory import ModelType
import argparse
from torch.optim.lr_scheduler import OneCycleLR
from data.dataset import DatasetType


def main(cfg, data_path=None, model_dir=None):
    set_random_seed(42)

    # --- Configs ---
    base_cfg = cfg["base"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load Model ---
    factory = ModelFactory(device=device, num_classes=base_cfg["num_classes"])
    model_type = ModelType(model_cfg["type"])
    model = factory.load(
        model_type=model_type,
        pretrained=train_cfg["pretrained"],
        model_dir=model_dir,
    )

    # --- Load Data ---
    print("Loading datasets...")
    train_loader = load_dataloader(
        data_path,
        dataset_type=DatasetType.TRAIN,
        img_size=base_cfg.get("img_size", 640),
        batch_size=model_cfg["batch_size"],
        num_workers=base_cfg["num_workers"],
    )

    val_loader = load_dataloader(
        data_path,
        dataset_type=DatasetType.VAL,
        img_size=base_cfg.get("img_size", 640),
        batch_size=model_cfg["batch_size"],
        num_workers=base_cfg["num_workers"],
    )

    # --- Optimizer ---
    lr = (
        model_cfg["pretrained_lr"]
        if train_cfg["pretrained"]
        else model_cfg["from_scratch_lr"]
    )

    optimizer = get_optimizer(
        model=model,
        model_type=model_type,
        lr=lr,
    )

    # --- LR Scheduler ---
    num_epochs = train_cfg["num_epochs"]

    steps_per_epoch = len(train_loader)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[lr * 0.1, lr] if model_cfg["type"] == "detr" else lr,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
    )

    # --- Training Loop ---
    print("Starting training...")

    results = {"train_loss": [], "val_loss": []}

    # Early stopping setup
    patience = train_cfg["patience"]
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        result = train_one_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )
        print(result)

        # Log results
        results["train_loss"].append(result.get("train_loss"))
        results["val_loss"].append(result.get("val_loss"))

        # Early stopping logic
        val_loss = result.get("val_loss")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs)"
                )
                break

    # --- Save Results as JSON ---
    save_and_plot_train_results(
        results,
        cfg,
        model_state_dict=best_model_state,
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
