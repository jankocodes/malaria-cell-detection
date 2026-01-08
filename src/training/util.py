import torch
from tqdm import tqdm


def train_one_epoch(
    model, train_loader, val_loader, optimizer, epoch, show_progress=True
):
    device = model.device
    print(f"[DEBUG] Model device: {device}")

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
