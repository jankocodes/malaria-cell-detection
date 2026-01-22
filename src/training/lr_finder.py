import torch
import os
from torch.utils.data import DataLoader
from ignite.engine import Engine
from ignite.handlers import FastaiLRFinder
from ignite.contrib.handlers import ProgressBar
from matplotlib.ticker import LogFormatterSciNotation, LogLocator
from ignite.handlers import FastaiLRFinder
from matplotlib.ticker import LogLocator
from matplotlib.ticker import NullLocator

# ---------------------------------------------------------
# 1. Training step for Ignite (returns only loss tensor)
# ---------------------------------------------------------


def find_lr(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    train_loader: DataLoader,
    show_progress: bool = True,
) -> FastaiLRFinder:
    # ---------------------------------------------------------
    def step_fn(engine, batch):
        model.train()
        device = model.device

        images, targets = batch

        images = images.float().to(device)

        # Ensure targets are also on the right device
        coco_targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        # Forward pass — YOLOv5Wrapper returns dict {"loss": tensor, ...}
        out = model(images, coco_targets)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    # ---------------------------------------------------------
    # 3. Ignite trainer engine
    # ---------------------------------------------------------

    trainer = Engine(step_fn)

    # ---------------------------------------------------------
    # 4. IGNITE LR FINDER
    # ---------------------------------------------------------
    lr_finder = FastaiLRFinder()
    to_save = {"model": model, "optimizer": optimizer}

    print("📈 Running LR Finder...")

    with lr_finder.attach(
        trainer, to_save=to_save, end_lr=max_lr
    ) as trainer_with_finder:
        pbar = ProgressBar(disable=not show_progress)
        pbar.attach(trainer_with_finder)

        trainer_with_finder.run(train_loader)

    return lr_finder


def plot_lr_finder_results(
    lr_finder: FastaiLRFinder,
    model_name: str,
    range: tuple,
    model_pretrained_str: str,
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
        f"Learning Rate Finder – {model_name} ({model_pretrained_str})",
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

    save_dir = os.path.join("plots", "lr_finder", model_pretrained_str)
    save_path = os.path.join(save_dir, f"lr_finder_plot_{model_name}.png")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"LR finder plot saved to: {save_path}")

    return ax
