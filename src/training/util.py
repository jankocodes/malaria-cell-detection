import torch
from tqdm import tqdm
from factory import ModelType
from wrappers.base import BaseDetectionModel
from torch.utils.data import DataLoader
from evaluation.metrics import calculate_ap
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

        if len(targets) == 0:
            continue

        # Ensure targets are also on the right device
        images, coco_targets = to_device(images, targets, device)

        # Forward
        result = model(images, coco_targets)  # 3 x [B, A, H, W, no]

        train_loss = result["loss"]

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step()

        total_train_loss += train_loss.item()

        training_pbar.set_postfix({"loss": f"{train_loss.item():.4f}"})

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        total_val_loss = 0

        val_pbar = tqdm(
            val_loader,
            desc=f"Validation - Epoch {epoch +1}",
        )

        all_predictions = []
        all_targets = []

        for _, (images, targets) in enumerate(val_pbar):

            if len(targets) == 0:
                continue

            images, coco_targets = to_device(images, targets, device)

            outputs = model(images, coco_targets, return_predictions=True)
            val_loss = outputs["loss"]
            predictions = outputs["predictions"]

            total_val_loss += val_loss.item()
            all_predictions.extend(predictions)
            all_targets.extend(coco_targets)

            val_pbar.set_postfix({"loss": f"{val_loss.item():.4f}"})

    model.train()

    result = {}
    result["mAP"] = calculate_ap(
        predictions=all_predictions,
        targets=all_targets,
        num_classes=model.num_classes,
    )
    result["train_loss"] = total_train_loss / len(train_loader)
    result["val_loss"] = total_val_loss / len(val_loader)

    print(
        f"[Epoch {epoch + 1}] Train Loss: {result['train_loss']:.4f}, Val Loss: {result['val_loss']:.4f}"
    )

    return result


def to_device(images, targets, device):
    images = images.float().to(device)
    coco_targets = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
        for t in targets
    ]
    return images, coco_targets


def get_optimizer(
    model: BaseDetectionModel,
    model_type: ModelType,
    lr_dict: dict,
    seperate_backbone_lr: bool = False,
) -> torch.optim.Optimizer:
    """
    Get the optimizer from the model type.

    Selects and configures an appropriate optimizer based on the model architecture.
    Different optimizers and hyperparameters are used for different model types:
    - YOLO models use SGD with weight decay 5e-4
    - RetinaNet uses SGD with weight decay 1e-4
    - DETR uses AdamW with differential learning rates for backbone vs other components
    - Deformable DETR uses Adam with differential learning rates including special params

    Args:
        model (BaseDetectionModel): The detection model instance.
        model_type (ModelType): The type of the model (YOLOV5, YOLOV8, RETINANET, DETR, or DEFORMABLE_DETR).
        lr_dict (dict): Dictionary containing learning rates ('model_lr', 'backbone_lr', 'special_lr').
        seperate_backbone_lr (bool): Whether to use separate learning rate for backbone parameters.

    Returns:
        torch.optim.Optimizer: Configured optimizer for the model.

    Raises:
        ValueError: If the model type is not supported.
    """
    lr = lr_dict["model_lr"]

    if model_type in {ModelType.YOLOV5, ModelType.YOLOV8}:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=5e-4,
            momentum=0.9,
        )
        print(f"Using SGD optimizer with LR={lr:.6f}")

    elif model_type == ModelType.RETINANET:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,
            momentum=0.9,
        )
        print(f"Using SGD optimizer with LR={lr:.6f}")

    elif model_type == ModelType.DETR:
        param_groups = get_param_groups(
            model,
            seperate_backbone_lr,
            model_type,
            lr_dict,
        )

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=1e-4,
        )
        print(f"Using AdamW optimizer with {len(param_groups)} parameter groups")

    elif model_type == ModelType.DEFORMABLE_DETR:
        param_groups = get_param_groups(
            model,
            seperate_backbone_lr,
            model_type,
            lr_dict,
        )

        optimizer = torch.optim.Adam(
            param_groups,
            weight_decay=1e-4,
        )
        print(f"Using Adam optimizer with {len(param_groups)} parameter groups")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return optimizer


def get_param_groups(model, seperate_backbone_lr, model_type, lr_dict: dict):
    """
    Create parameter groups with different learning rates.

    Groups model parameters into categories with different learning rates:
    - Backbone parameters: Lower LR when using pretrained weights
    - Special parameters (Deformable DETR): reference_points and sampling_offsets get special LR
    - Regular parameters: Standard model LR

    Args:
        model (BaseDetectionModel): The model to extract parameters from.
        seperate_backbone_lr (bool): Whether to use separate LR for backbone.
        model_type (ModelType): Type of model being trained.
        lr_dict (dict): Dictionary containing 'model_lr', 'backbone_lr', 'special_lr'.

    Returns:
        list: List of parameter group dictionaries for optimizer.
    """

    def match_special_params(name):
        """Check if parameter name matches special Deformable DETR parameters."""
        return "reference_points" in name or "sampling_offsets" in name

    # Extract learning rates from dict
    lr_model = lr_dict["model_lr"]
    lr_backbone = lr_dict["backbone_lr"]
    lr_special = lr_dict["special_lr"]

    # Collect parameters into groups
    backbone_params = []
    model_params = []
    special_params = []

    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue

        # Categorize parameter
        if "backbone" in name and seperate_backbone_lr:
            backbone_params.append(param)
            print(f"Backbone param: {name} (LR: {lr_backbone:.6f})")
        elif match_special_params(name) and model_type == ModelType.DEFORMABLE_DETR:
            special_params.append(param)
            print(f"Deformable DETR special param: {name} (LR: {lr_special:.6f})")
        else:
            model_params.append(param)
            print(f"Regular param: {name} (LR: {lr_model:.6f})")

    # Build parameter groups (only include non-empty groups)
    param_groups = []

    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    if model_params:
        param_groups.append({"params": model_params, "lr": lr_model})

    if special_params:
        param_groups.append({"params": special_params, "lr": lr_special})

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


def get_lr_dict(model_cfg, pretrained, finder=False):
    """
    Get learning rate dictionary for different parts of the model.

    Creates a dictionary with learning rates for:
    - model_lr: Main learning rate for most parameters
    - backbone_lr: Learning rate for backbone (typically 0.1x model_lr for pretrained DETR)
    - special_lr: Learning rate for special parameters in Deformable DETR (reference_points, sampling_offsets)

    Args:
        model_cfg (dict): Model configuration containing learning rate settings.
        pretrained (bool): Whether using pretrained weights.
        finder (bool): Whether running LR finder (uses base_lr if True).

    Returns:
        dict: Dictionary with 'model_lr', 'backbone_lr', and 'special_lr' keys.
    """
    # Determine the main model learning rate
    model_lr = (
        model_cfg["base_lr"]
        if finder
        else (
            model_cfg["pretrained_lr"] if pretrained else model_cfg["from_scratch_lr"]
        )
    )

    # Backbone LR: use explicit value or default to 0.1x model_lr
    backbone_lr = model_cfg.get("backbone_lr", model_lr * 0.1)

    # Special LR for Deformable DETR parameters
    if pretrained:
        special_lr = model_cfg.get(
            "pretrained_special_lr", model_cfg.get("special_lr", model_lr * 0.1)
        )
    else:
        special_lr = model_cfg.get(
            "from_scratch_special_lr", model_cfg.get("special_lr", model_lr * 0.1)
        )

    lr_dict = {
        "model_lr": model_lr,
        "backbone_lr": backbone_lr,
        "special_lr": special_lr,
    }

    return lr_dict


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
