from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mmcr.checkpoints import ENCODER_FILE
from mmcr.data import normalize_dataset_key
from mmcr.evaluation import evaluate_encoder
from mmcr.models import DEFAULT_ARCH
from mmcr.utils import build_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def resolve_source_encoder_path(checkpoint_root: Path, dataset: str) -> Path:
    direct_path = checkpoint_root / dataset / ENCODER_FILE
    if direct_path.exists():
        return direct_path
    normalized_path = checkpoint_root / normalize_dataset_key(dataset) / ENCODER_FILE
    return normalized_path if normalized_path.exists() else direct_path


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = build_device(args.gpu)
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print("Using device: cpu")

    checkpoint_root = Path(args.checkpoint_root)
    results = {}
    source_baseline_scores = {}
    for dataset in args.datasets:
        encoder_path = resolve_source_encoder_path(checkpoint_root, dataset)
        print(f"{dataset}: evaluating source encoder {encoder_path}")
        evaluated = evaluate_encoder(
            encoder_path=encoder_path,
            datasets=[dataset],
            checkpoint_root=args.checkpoint_root,
            data_root=args.data_root,
            arch=args.arch,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            amp=args.amp,
            download=not args.no_download,
        )
        results[dataset] = {**evaluated[dataset], "encoder": str(encoder_path)}
        source_baseline_scores[dataset] = float(evaluated[dataset]["acc"])

    average = sum(source_baseline_scores.values()) / len(source_baseline_scores)
    payload = {
        "type": "source_baseline_scores",
        "datasets": args.datasets,
        "checkpoint_root": args.checkpoint_root,
        "data_root": args.data_root,
        "arch": args.arch,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "split": "test",
        "source_baseline_scores": {**source_baseline_scores, "average": average},
        "results": {**results, "average": {"acc": average}},
    }
    write_json(Path(args.output_json), payload)
    print(f"Saved source baseline scores to {args.output_json}")


if __name__ == "__main__":
    main()
