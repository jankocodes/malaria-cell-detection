import torch
from torch.utils.data import DataLoader
from factory import ModelFactory
from data.dataset import BloodCellDataset
from data.util import collate_fn
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

        images, targets = batch

        # Forward pass — YOLOv5Wrapper returns dict {"loss": tensor, ...}
        out = model(images, targets)
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
