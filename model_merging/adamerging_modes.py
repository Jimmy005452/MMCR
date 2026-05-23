import argparse
from pathlib import Path

import torch

from mmcr.adamerging_modes import (
    build_adamerging_loaders,
    build_adamerging_modes_model,
    train_adamerging_modes,
)
from mmcr.checkpoints import ENCODER_FILE
from mmcr.evaluation import resolve_head_path
from mmcr.models import DEFAULT_ARCH
from mmcr.utils import build_device, seed_everything, write_json


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--head-root", default=None)
    parser.add_argument("--zeroshot", default=None)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--lambda-mode", choices=["task", "tensor"], default="task")
    parser.add_argument("--top-k", type=float, default=20)
    parser.add_argument("--prior", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batches-per-dataset", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-json", default=None)
    parser.add_argument("--lambda-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    checkpoint_root = Path(args.checkpoint_root)
    head_root = Path(args.head_root) if args.head_root is not None else checkpoint_root
    zeroshot_path = Path(args.zeroshot) if args.zeroshot is not None else checkpoint_root / "zeroshot.pt"
    encoder_paths = [checkpoint_root / dataset / ENCODER_FILE for dataset in args.datasets]
    head_paths = [resolve_head_path(dataset, head_root) for dataset in args.datasets]

    for dataset, encoder_path, head_path in zip(args.datasets, encoder_paths, head_paths):
        print(f"{dataset}: encoder={encoder_path} head={head_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(encoder_path)
        if not head_path.exists():
            raise FileNotFoundError(head_path)
    if not zeroshot_path.exists():
        raise FileNotFoundError(zeroshot_path)

    device = build_device(args.gpu)
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print("Using device: cpu")

    model = build_adamerging_modes_model(
        arch=args.arch,
        datasets=args.datasets,
        head_paths=head_paths,
        zeroshot_path=zeroshot_path,
        encoder_paths=encoder_paths,
        device=device,
        lambda_mode=args.lambda_mode,
        top_k_percent=args.top_k,
        prior=args.prior,
    )
    loaders = build_adamerging_loaders(
        datasets=args.datasets,
        data_root=args.data_root,
        arch=args.arch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
    )

    history = train_adamerging_modes(
        model=model,
        loaders=loaders,
        epochs=args.epochs,
        batches_per_dataset=args.batches_per_dataset,
        lr=args.lr,
        amp=args.amp,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.export_merged_state(), output_path)
    print(f"Saved AdaMerging-{args.lambda_mode} encoder to {output_path}")

    history_path = Path(args.history_json) if args.history_json is not None else output_path.with_suffix(".history.json")
    write_json(
        history_path,
        {
            "datasets": args.datasets,
            "lambda_mode": args.lambda_mode,
            "checkpoint_root": str(checkpoint_root),
            "head_root": str(head_root),
            "zeroshot": str(zeroshot_path),
            "output": str(output_path),
            "top_k": args.top_k,
            "prior": args.prior,
            "epochs": args.epochs,
            "batches_per_dataset": args.batches_per_dataset,
            "history": history,
        },
    )
    print(f"Saved lambda history to {history_path}")

    lambda_path = Path(args.lambda_json) if args.lambda_json is not None else output_path.with_suffix(".lambdas.json")
    write_json(
        lambda_path,
        {
            "datasets": args.datasets,
            "lambda_mode": args.lambda_mode,
            "lambda_keys": model.lambda_keys if args.lambda_mode == "tensor" else None,
            "lambdas": model.export_lambdas(),
        },
    )
    print(f"Saved lambdas to {lambda_path}")


if __name__ == "__main__":
    main()
