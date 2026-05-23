import json
import random
from pathlib import Path

import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy(logits, targets) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def build_device(gpu: int):
    if gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")

    if gpu >= torch.cuda.device_count():
        raise ValueError(f"GPU index {gpu} is unavailable. Found {torch.cuda.device_count()} CUDA device(s).")

    return torch.device(f"cuda:{gpu}")


def write_metrics(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
