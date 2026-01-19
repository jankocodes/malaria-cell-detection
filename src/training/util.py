import torch
from tqdm import tqdm
from wrappers.detr import DetrWrapper
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from ignite.handlers import FastaiLRFinder
from factory import ModelType
from wrappers.base import BaseDetectionModel
from matplotlib.ticker import LogLocator
from matplotlib.ticker import NullLocator


def train_one_epoch(
    model, train_loader, val_loader, optimizer, epoch, show_progress=True
):
    device = model.device

    # --- Training ---
    model.train()
    total_train_loss = 0

    pbar = tqdm(
        train_loader, desc=f"Training - Epoch {epoch +1}", disable=not show_progress
    )

    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.float().to(device)

        if len(targets) == 0:
            continue

        # Ensure targets are also on the right device
        coco_targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        # Forward
        result = model(images, coco_targets)  # 3 x [B, A, H, W, no]

        train_loss = result["loss"]

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_train_loss += train_loss.item()

        pbar.set_postfix({"loss": f"{train_loss.item():.4f}"})

    # --- Validation ---
    with torch.no_grad():
        total_val_loss = 0

        pbar = tqdm(
            val_loader, desc=f"Validation - Epoch {epoch +1}", disable=not show_progress
        )

        # validation
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.float().to(device)

            if len(targets) == 0:
                continue

            coco_targets = [
                {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            # Forward
            loss = model(images, coco_targets)  # 3 x [B, A, H, W, no]

            val_loss = loss["loss"]
            total_val_loss += val_loss.item()

            pbar.set_postfix({"loss": f"{val_loss.item():.4f}"})

    result = {}
    result["train_loss"] = total_train_loss / len(train_loader)
    result["val_loss"] = total_val_loss / len(val_loader)

    print(
        f"[Epoch {epoch + 1}] Train Loss: {result['train_loss']:.4f}, Val Loss: {result['val_loss']:.4f}"
    )

    return result


def freeze_detr_backbone(detr: DetrWrapper):
    # Freeze the backbone for LR finder
    for param in detr.model.model.backbone.parameters():
        param.requires_grad = False


def plot_lr_finder_results(
    lr_finder: FastaiLRFinder,
    model_name: str,
    range: tuple,
    skip_start=0,
    skip_end=1,
    log_lr=True,
    display_suggestion=True,
    figsize=(8, 5),
):
    """
    Enhanced visualization for FastAI's LRFinder using its built-in plot().

    Args:
        lr_finder: FastaiLRFinder object (after lr_find).
        model_name: Name used in title and default filename.
        skip_start: Batches to skip at start.
        skip_end: Batches to skip at end (KEEP THIS 0 to avoid clipping).
        log_lr: Whether to use log-scale LR.
        display_suggestion: Show FastAI's suggested LR marker.
        figsize: Figure size passed to plt.subplots.
    """
    # --- Call FastAI plot ---
    ax = lr_finder.plot(
        skip_start=skip_start,
        skip_end=skip_end,
        log_lr=log_lr,
        display_suggestion=display_suggestion,
        figsize=figsize,
    )

    lr_min, lr_max = range

    # --- Improve title and labels ---
    ax.set_title(
        f"Learning Rate Finder – {model_name}",
        fontsize=14,
    )
    ax.set_xlabel("Learning Rate (log scale)" if log_lr else "Learning Rate")
    ax.set_ylabel("Loss")

    # --- Improve tick spacing for log-scale ---
    if log_lr:
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(NullLocator())  # <-- CRITICAL
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.xaxis.set_major_formatter(LogFormatterSciNotation())

    # --- Force LR range ---
    ax.set_xlim(lr_min, lr_max)

    # --- Improve grid aesthetics ---
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    # --- Improve line visibility ---
    for line in ax.lines:
        line.set_linewidth(2)

    # --- Explicit suggested LR annotation (optional clarity) ---
    if display_suggestion and hasattr(lr_finder, "lr_suggestion"):
        suggested_lr = lr_finder.lr_suggestion()
        ax.axvline(
            suggested_lr,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label=f"Suggested LR = {suggested_lr:.2e}",
        )
        ax.legend()

    # --- Save ---
    fig = ax.figure
    fig.tight_layout()

    save_path = f"lr_finder_plot_{model_name}.png"

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"LR finder plot saved to: {save_path}")

    return ax


def get_optimizer(
    model: BaseDetectionModel,
    model_type: ModelType,
    lr: float,
) -> torch.optim.Optimizer:
    """Get the optimizer from the model type."""
    if model_type in {ModelType.YOLOV5, ModelType.YOLOV8, ModelType.RETINANET}:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
        )
    elif model_type == ModelType.DETR:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"Using optimizer: {optimizer}")

    return optimizer
