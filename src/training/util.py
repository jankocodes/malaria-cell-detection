import torch
from tqdm import tqdm
from factory import ModelType
from wrappers.base import BaseDetectionModel
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import json
import matplotlib.pyplot as plt
import yaml


def train_one_epoch(
    model: BaseDetectionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch,
):
    """
    Train and validate the model for one epoch.

    Performs a complete training and validation cycle for the given epoch.
    Trains the model on the training dataset, computes losses, performs backpropagation,
    and updates model weights. Then validates on the validation dataset without gradient computation.

    Args:
        model (BaseDetectionModel): The detection model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        epoch (int): Current epoch number (0-indexed).

    Returns:
        dict: Dictionary containing:
            - 'train_loss' (float): Average training loss across the epoch.
            - 'val_loss' (float): Average validation loss across the epoch.

    Side effects:
        - Updates model weights during training.
        - Updates learning rate scheduler.
        - Prints learning rates for each parameter group after training.
        - Prints epoch summary with train and validation losses.
    """
    device = model.device

    # --- Training ---
    model.train()
    total_train_loss = 0

    training_pbar = tqdm(
        train_loader,
        desc=f"Training - Epoch {epoch +1}",
    )

    for _, (images, targets) in enumerate(training_pbar):
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
            val_loader,
            desc=f"Validation - Epoch {epoch +1}",
        )

        # validation
        for _, (images, targets) in enumerate(val_pbar):
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
    seperate_backbone_lr: bool = False,
) -> torch.optim.Optimizer:
    """
    Get the optimizer from the model type.

    Selects and configures an appropriate optimizer based on the model architecture.
    Different optimizers and hyperparameters are used for different model types:
    - YOLO models use SGD with weight decay 5e-4
    - RetinaNet uses SGD with weight decay 1e-4
    - DETR uses AdamW with differential learning rates for backbone vs other components

    Args:
        model (BaseDetectionModel): The detection model instance.
        model_type (ModelType): The type of the model (YOLOV5, YOLOV8, RETINANET, or DETR).
        lr (float): Base learning rate for the optimizer.

    Returns:
        torch.optim.Optimizer: Configured optimizer for the model.

    Raises:
        ValueError: If the model type is not supported.
    """
    if model_type in {ModelType.YOLOV5, ModelType.YOLOV8}:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=5e-4,
            momentum=0.9,
        )
    elif model_type == ModelType.RETINANET:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            momentum=0.9,
        )
    elif model_type == ModelType.DETR:

        param_groups = (
            get_seperate_backbone_lr(model, lr)
            if seperate_backbone_lr
            else model.parameters()
        )

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=1e-4,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    print(f"Using optimizer: {optimizer}")

    return optimizer


def get_seperate_backbone_lr(model, lr: float):
    backbone_params = []
    other_params = []

    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue

        if "backbone" in name:
            backbone_params.append(param)
            print(f"Backbone param: {name}")
        else:
            other_params.append(param)
            print(f"Other param: {name}")

    param_groups = [
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": other_params, "lr": lr},
    ]
    return param_groups


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


def save_and_plot_train_results(
    results,
    cfg: dict,
    model_state_dict: torch.nn.Module = None,
):
    """
    Save training results and generate visualizations.

    This function saves the training results to a JSON file, optionally saves the model checkpoint,
    and generates a plot comparing training and validation losses across epochs.

    Args:
        results (dict): Dictionary containing 'train_loss' and 'val_loss' lists with loss values for each epoch.
        model_cfg (dict): Model configuration dictionary containing 'type' and 'pretrained' keys.
                         'type' is used for naming output files.
                         'pretrained' is a boolean indicating if the model was pretrained.
        model_state_dict (torch.nn.Module, optional): Model state dictionary to save as a checkpoint.
                                                      If None, no checkpoint is saved. Defaults to None.

    Returns:
        None

    Side effects:
        - Creates and saves results JSON file to 'results/train/{pretrained|from_scratch}/'
        - Creates and saves model checkpoint to 'checkpoints/{pretrained|from_scratch}/' if model_state_dict is provided
        - Creates and saves training/validation loss plot to 'plots/train/{pretrained|from_scratch}/'
        - Prints paths of saved files to console
    """
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    pretrained_str = "pretrained" if train_cfg["pretrained"] else "from_scratch"

    # Setup results directory
    results_dir = os.path.join(
        "results/train",
        pretrained_str,
    )
    os.makedirs(results_dir, exist_ok=True)

    # Save results to file
    results_file = os.path.join(results_dir, f"{model_cfg['type']}_train_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Training results saved to: {results_file}")

    # Save model checkpoint if provided
    if model_state_dict is not None:
        checkpoints_dir = os.path.join(
            "checkpoints",
            pretrained_str,
        )
        os.makedirs(checkpoints_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoints_dir, f"{model_cfg['type']}_model.pt")
        torch.save(model_state_dict, checkpoint_file)
        print(f"Model checkpoint saved to: {checkpoint_file}")

    plots_dir = os.path.join(
        "plots/train",
        pretrained_str,
    )
    os.makedirs(plots_dir, exist_ok=True)

    train_losses = results["train_loss"]
    val_losses = results["val_loss"]
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Val Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss - {model_cfg['type']} ({pretrained_str})")
    plt.legend()
    plt.grid(True)

    max_epoch = int(max(epochs))

    xticks = [1]
    xticks += list(range(5, max_epoch + 1, 5))

    plt.xticks(xticks)

    plot_file = os.path.join(plots_dir, f"{model_cfg['type']}_train_val_loss.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to: {plot_file}")


def load_config(base_config, model_config, train_config=None):
    with open(base_config, "r") as f:
        base_cfg = yaml.safe_load(f)

    with open(model_config, "r") as f:
        model_cfg = yaml.safe_load(f)

    if train_config is not None:
        with open(train_config, "r") as f:
            train_cfg = yaml.safe_load(f)
    else:
        train_cfg = {}

    cfg = {"base": base_cfg, "model": model_cfg, "train": train_cfg}

    return cfg
