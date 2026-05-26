from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from mmcr.data import build_loader
from mmcr.models import build_model_transforms


class RewardBatchPool(dict):
    def __init__(self, batches: dict, *, sampling_mode: str = "sequential", batches_per_eval: int | None = None):
        super().__init__(batches)
        self.sampling_mode = sampling_mode
        self.batches_per_eval = batches_per_eval


def _dataset_targets(dataset) -> list[int]:
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            if torch.is_tensor(values):
                return [int(value) for value in values.cpu().tolist()]
            return [int(value) for value in values]
    for attr in ("samples", "_samples"):
        if hasattr(dataset, attr):
            return [int(sample[1]) for sample in getattr(dataset, attr)]
    # Fallback for custom datasets. This can be slower because it calls __getitem__,
    # but it keeps the sampling mode usable for datasets without exposed labels.
    targets = []
    for index in range(len(dataset)):
        _image, target = dataset[index]
        targets.append(int(target))
    return targets


def _stratified_interleaved_indices(targets: list[int], pool_size: int, seed: int) -> list[int]:
    buckets = defaultdict(list)
    for index, target in enumerate(targets):
        buckets[int(target)].append(index)

    generator = torch.Generator().manual_seed(int(seed))
    class_ids = sorted(buckets)
    for class_id in class_ids:
        bucket = buckets[class_id]
        order = torch.randperm(len(bucket), generator=generator).tolist()
        buckets[class_id] = [bucket[i] for i in order]

    pointers = {class_id: 0 for class_id in class_ids}
    indices = []
    while len(indices) < pool_size:
        added = False
        for class_id in class_ids:
            pointer = pointers[class_id]
            bucket = buckets[class_id]
            if pointer >= len(bucket):
                continue
            indices.append(bucket[pointer])
            pointers[class_id] = pointer + 1
            added = True
            if len(indices) == pool_size:
                break
        if not added:
            break
    return indices


def _collect_batches(loader, max_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    for batch_index, (images, targets) in enumerate(loader):
        if batch_index == max_batches:
            break
        batches.append((images.cpu(), targets.cpu()))
    return batches


def build_reward_batches(
    datasets: list[str],
    data_root: Path | str,
    arch: str,
    batch_size: int,
    batches_per_dataset: int,
    num_workers: int,
    split: str = "val",
    download: bool = True,
    sampling_mode: str = "sequential",
    reward_pool_size: int = 0,
    seed: int = 0,
):
    if batches_per_dataset <= 0:
        raise ValueError("batches_per_dataset must be positive.")
    if sampling_mode not in {"sequential", "stratified_pool"}:
        raise ValueError("sampling_mode must be either 'sequential' or 'stratified_pool'.")
    if reward_pool_size < 0:
        raise ValueError("reward_pool_size must be non-negative.")

    _, eval_transform, _ = build_model_transforms(arch, pretrained=False)
    reward_batches = {}
    batches_per_eval = None

    for dataset_index, dataset in enumerate(tqdm(datasets, desc="reward batches", leave=False)):
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

        if sampling_mode == "sequential":
            batches = _collect_batches(loader, batches_per_dataset)
        else:
            eval_samples = batch_size * batches_per_dataset
            pool_size = reward_pool_size if reward_pool_size > 0 else eval_samples
            pool_size = max(pool_size, eval_samples)
            targets = _dataset_targets(loader.dataset)
            pool_size = min(pool_size, len(targets))
            indices = _stratified_interleaved_indices(targets, pool_size, seed + dataset_index)
            subset = Subset(loader.dataset, indices)
            pool_loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
            max_batches = max(1, (len(indices) + batch_size - 1) // batch_size)
            batches = _collect_batches(pool_loader, max_batches)
            batches_per_eval = batches_per_dataset

        if not batches:
            raise RuntimeError(f"No reward batches were collected for dataset '{dataset}'.")
        reward_batches[dataset] = batches

    return RewardBatchPool(reward_batches, sampling_mode=sampling_mode, batches_per_eval=batches_per_eval)


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

    return RewardBatchPool(reward_batches, sampling_mode="sequential")
