from __future__ import annotations

from pathlib import Path

import torch
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

def build_synthetic_reward_batches(
    datasets: list[str],
    synthesis_root: Path | str,
    batch_size: int,
    batches_per_dataset: int,
    include_rejected: bool = False,
):
    if batches_per_dataset <= 0:
        raise ValueError("batches_per_dataset must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    root = Path(synthesis_root)
    reward_batches = {}
    max_samples = batch_size * batches_per_dataset

    for dataset in tqdm(datasets, desc="synthetic reward batches", leave=False):
        folder = root / dataset
        inputs_path = folder / "inputs.pt"
        logits_path = folder / "teacher_logits.pt"
        missing = [path for path in (inputs_path, logits_path) if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing synthetic file(s): " + ", ".join(str(path) for path in missing))

        inputs = torch.load(inputs_path, map_location="cpu")
        teacher_logits = torch.load(logits_path, map_location="cpu")
        if inputs.shape[0] != teacher_logits.shape[0]:
            raise ValueError(f"{dataset}: inputs and teacher_logits have different lengths.")

        mask_path = folder / "accepted_mask.pt"
        if not include_rejected and mask_path.exists():
            mask = torch.load(mask_path, map_location="cpu").bool()
            if mask.shape[0] != inputs.shape[0]:
                raise ValueError(f"{dataset}: accepted_mask length does not match inputs.")
            if not mask.any():
                raise ValueError(f"{dataset}: accepted_mask has zero accepted samples. Use --include-rejected-synthetic.")
            inputs = inputs[mask]
            teacher_logits = teacher_logits[mask]

        inputs = inputs[:max_samples].cpu()
        teacher_logits = teacher_logits[:max_samples].cpu()
        batches = []
        for start in range(0, inputs.shape[0], batch_size):
            if len(batches) == batches_per_dataset:
                break
            batches.append((inputs[start : start + batch_size], teacher_logits[start : start + batch_size]))
        if not batches:
            raise RuntimeError(f"No synthetic reward batches were collected for dataset '{dataset}'.")
        reward_batches[dataset] = batches

    return reward_batches

