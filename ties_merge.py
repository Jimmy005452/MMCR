import argparse
from pathlib import Path

import torch

from mmcr.checkpoints import ENCODER_FILE
from mmcr.ties import merge_encoder_checkpoints


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--zeroshot", default=None)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--top-k", type=float, default=20)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--merge-func", choices=["dis-sum", "dis-mean"], default="dis-mean")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    zeroshot_path = Path(args.zeroshot) if args.zeroshot is not None else checkpoint_root / "zeroshot.pt"
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    encoder_paths = [checkpoint_root / dataset / ENCODER_FILE for dataset in args.datasets]
    for dataset, path in zip(args.datasets, encoder_paths):
        print(f"Using {dataset}: {path}")

    merged_state = merge_encoder_checkpoints(
        zeroshot_path=zeroshot_path,
        finetuned_paths=encoder_paths,
        top_k_percent=args.top_k,
        scaling_coef=args.scale,
        merge_func=args.merge_func,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_state, output_path)
    print(f"Saved TIES merged encoder to {output_path}")


if __name__ == "__main__":
    main()
