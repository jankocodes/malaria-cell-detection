import torch
from tqdm import tqdm
from od_utils.data import xyxy_to_xywh_norm


def train_one_epoch(model, dataloader, loss_fn, optimizer, epoch):
    device = model.device

    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for images, targets in pbar:
        images = images.float().to(device)

        formatted_targets = []
        for i, t in enumerate(targets):

            # target has no bboxes
            if len(t["boxes"]) == 0:
                continue

            boxes = t["boxes"]  # xyxy format
            labels = t["labels"].unsqueeze(1)
            img_idx = torch.full((len(labels), 1), i)

            # Convert xyxy → normalized xywh
            xywh = xyxy_to_xywh_norm(boxes, images.shape[2:], device=device)
            merged = torch.cat([img_idx, labels.to(device), xywh.to(device)], dim=1)
            formatted_targets.append(merged)

        if len(formatted_targets) == 0:
            continue

        targets_tensor = torch.cat(formatted_targets, dim=0)

        # Forward
        preds = model(images)  # 3 x [B, A, H, W, no]

        loss, (lbox, lobj, lcls) = loss_fn(preds, targets_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)
