import torch


def collate_fn(batch):
    images, targets = zip(*batch)  # unzip
    return torch.stack(images), list(targets)
