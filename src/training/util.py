import torch
from tqdm import tqdm


def train_one_epoch(model, train_loader, val_loader, optimizer, epoch):
    device = model.device

    # --- Training ---
    model.train()
    total_train_loss = 0

    pbar = tqdm(train_loader, desc=f"Training - Epoch {epoch +1}")

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

        pbar = tqdm(val_loader, desc=f"Validation - Epoch {epoch +1}")

        # validation
        for images, targets in pbar:
            images = images.float().to(device)

            if len(targets) == 0:
                continue

            # Forward
            loss = model(images, targets)  # 3 x [B, A, H, W, no]

            val_loss = loss["loss"]
            total_val_loss += val_loss.item()

            pbar.set_postfix({"loss": f"{val_loss.item():.4f}"})

    result = {}
    result["train_loss"] = total_train_loss / len(train_loader)
    result["val_loss"] = total_val_loss / len(val_loader)

    return result
