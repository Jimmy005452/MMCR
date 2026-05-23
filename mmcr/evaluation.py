from pathlib import Path

import torch
from tqdm import tqdm

from mmcr.checkpoints import HEAD_FILE, load_encoder, load_head
from mmcr.data import build_loader, get_num_classes, normalize_dataset_key
from mmcr.models import ImageClassifier, build_model_transforms


def resolve_head_path(dataset: str, checkpoint_root: Path | str, head_path: Path | str | None = None):
    if head_path is not None:
        return Path(head_path)

    checkpoint_root = Path(checkpoint_root)
    direct_path = checkpoint_root / dataset / HEAD_FILE
    if direct_path.exists():
        return direct_path

    normalized = normalize_dataset_key(dataset)
    normalized_path = checkpoint_root / normalized / HEAD_FILE
    if normalized_path.exists():
        return normalized_path

    return direct_path


@torch.inference_mode()
def evaluate_accuracy(model, loader, device, amp: bool = False):
    model.eval()
    correct = 0
    total = 0

    progress = tqdm(loader, desc="test", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(images)

        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.numel()
        progress.set_postfix(acc=f"{correct / max(total, 1):.4f}")

    return correct / max(total, 1)


def evaluate_dataset(
    encoder,
    dataset: str,
    head_path: Path | str,
    data_root: Path | str,
    batch_size: int,
    num_workers: int,
    eval_transform,
    device,
    amp: bool = False,
    download: bool = True,
):
    head = load_head(head_path, device=device)
    model = ImageClassifier(encoder, head).to(device)
    loader, _ = build_loader(
        dataset,
        data_root,
        split="test",
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        eval_transform=eval_transform,
    )
    acc = evaluate_accuracy(model, loader, device, amp=amp)
    return acc


def evaluate_encoder(
    encoder_path: Path | str,
    datasets: list[str],
    checkpoint_root: Path | str,
    data_root: Path | str,
    arch: str,
    batch_size: int,
    num_workers: int,
    device,
    amp: bool = False,
    download: bool = True,
    head_path: Path | str | None = None,
):
    if head_path is not None and len(datasets) != 1:
        raise ValueError("head_path can only be used when evaluating exactly one dataset.")

    encoder = load_encoder(encoder_path, arch=arch, device=device)
    _, eval_transform, _ = build_model_transforms(arch, pretrained=False)

    results = {}
    for dataset in datasets:
        current_head_path = resolve_head_path(dataset, checkpoint_root, head_path=head_path)
        acc = evaluate_dataset(
            encoder,
            dataset,
            current_head_path,
            data_root,
            batch_size,
            num_workers,
            eval_transform,
            device,
            amp=amp,
            download=download,
        )
        print(f"{dataset}: ACC={acc * 100:.2f}%")
        results[dataset] = {"acc": acc, "num_classes": get_num_classes(dataset)}

    if len(results) > 1:
        results["average"] = {"acc": sum(item["acc"] for item in results.values()) / len(datasets)}

    return results
