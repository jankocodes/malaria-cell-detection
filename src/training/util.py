import torch
from tqdm import tqdm
from factory import ModelType
from wrappers.base import BaseDetectionModel
from torch.utils.data import DataLoader
import numpy as np
import random


def train_one_epoch(
    model: BaseDetectionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch,
    show_progress=True,
):
    device = model.device

    # --- Training ---
    model.train()
    total_train_loss = 0

    training_pbar = tqdm(
        train_loader, desc=f"Training - Epoch {epoch +1}", disable=not show_progress
    )

    for batch_idx, (images, targets) in enumerate(training_pbar):
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
        scheduler.step()

        total_train_loss += train_loss.item()

        training_pbar.set_postfix({"loss": f"{train_loss.item():.4f}"})
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Param group {i} LR: {param_group['lr']:.6f}")

    # --- Validation ---
    with torch.no_grad():
        total_val_loss = 0

        val_pbar = tqdm(
            val_loader, desc=f"Validation - Epoch {epoch +1}", disable=not show_progress
        )

        # validation
        for batch_idx, (images, targets) in enumerate(val_pbar):
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

            val_pbar.set_postfix({"loss": f"{val_loss.item():.4f}"})

    result = {}
    result["train_loss"] = total_train_loss / len(train_loader)
    result["val_loss"] = total_val_loss / len(val_loader)

    print(
        f"[Epoch {epoch + 1}] Train Loss: {result['train_loss']:.4f}, Val Loss: {result['val_loss']:.4f}"
    )

    return result


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


def set_random_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility.

    Args:
        seed (int): The seed value to use.
        deterministic (bool): If True, enables deterministic CuDNN behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multiple GPUs

    if deterministic:
        # Force deterministic behavior in CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
