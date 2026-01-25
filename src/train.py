import torch
from factory import ModelFactory
from training.util import *
from data.util import *
from factory import ModelType
import argparse
import yaml
from torch.optim.lr_scheduler import OneCycleLR


def main(cfg, data_path=None, model_dir=None):
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
    train_loader, val_loader = load_train_val_loaders(data_path, hyp)

    # --- Optimizer ---
    lr = hyp.get("lr")
    optimizer = get_optimizer(
        model=model,
        model_type=ModelType(model_cfg["type"]),
        lr=lr,
    )

    # --- LR Scheduler ---
    num_epochs = hyp["num_epochs"]

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
    patience = 10
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
        model_cfg,
        model_state_dict=best_model_state,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Path to config file")
    parser.add_argument("--data_path", type=str, help="Override dataset path")
    parser.add_argument("--model_dir", type=str, help="Override model directory path")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(
        cfg,
        data_path=args.data_path,
        model_dir=args.model_dir,
    )
