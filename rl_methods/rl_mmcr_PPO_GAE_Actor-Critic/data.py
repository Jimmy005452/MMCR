from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from mmcr.data import build_loader
from mmcr.models import build_model_transforms


def build_reward_batches(
    datasets: list[str],
    data_root: Path | str,
    arch: str,
    batch_size: int,
    batches_per_dataset: int,
    num_workers: int,
    split: str = "val",
    download: bool = True,
):
    if batches_per_dataset <= 0:
        raise ValueError("batches_per_dataset must be positive.")

    _, eval_transform, _ = build_model_transforms(arch, pretrained=False)
    reward_batches = {}

    for dataset in tqdm(datasets, desc="reward batches", leave=False):
        loader, _ = build_loader(
            dataset,
            data_root,
            split=split,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            train_transform=eval_transform,
            eval_transform=eval_transform,
            shuffle=False,
        )
        batches = []
        for batch_index, (images, targets) in enumerate(loader):
            if batch_index == batches_per_dataset:
                break
            batches.append((images.cpu(), targets.cpu()))
        if not batches:
            raise RuntimeError(f"No reward batches were collected for dataset '{dataset}'.")
        reward_batches[dataset] = batches

    return reward_batches
