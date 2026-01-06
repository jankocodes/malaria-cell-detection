import torch
from torch.utils.data import DataLoader
from ignite.engine import Engine
from ignite.handlers import FastaiLRFinder
from ignite.contrib.handlers import ProgressBar

# ---------------------------------------------------------
# 1. Training step for Ignite (returns only loss tensor)
# ---------------------------------------------------------


def find_lr(
    model: torch.nn.Module,
    min_lr: float,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=min_lr)

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
