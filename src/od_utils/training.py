import torch
from tqdm import tqdm
from od_utils.data import format_batch_targets


def train_one_epoch(model, train_loader, val_loader, optimizer, epoch):
    device = model.device

    # --- Training ---
    model.train()
    total_train_loss = 0

    pbar = tqdm(train_loader, desc=f"Training - Epoch {epoch}")

    for images, targets in pbar:
        images = images.float().to(device)

        if len(targets) == 0:
            continue

        # Forward
        result = model(images, targets)  # 3 x [B, A, H, W, no]

        train_loss = result["loss"]

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        total_train_loss += train_loss.item()

        pbar.set_postfix({"loss": f"{train_loss.item():.4f}"})

    # --- Training ---
    with torch.no_grad():
        total_val_loss = 0

        pbar = tqdm(val_loader, desc=f"Validation - Epoch {epoch}")

        # validation
        for images, targets in pbar:
            images = images.float().to(device)

            if len(targets) == 0:
                continue

            # Forward
            preds = model(images)  # 3 x [B, A, H, W, no]

            val_loss, (lbox, lobj, lcls) = model.compute_loss(
                preds, targets, images.shape[2:]
            )

            total_val_loss += val_loss.item()

            pbar.set_postfix({"loss": f"{val_loss.item():.4f}"})

    result = {}
    result["train_loss"] = total_train_loss / len(train_loader)
    result["val_loss"] = total_val_loss / len(val_loader)

    return result
