import argparse
from pathlib import Path

import torch

from mmcr.checkpoints import ENCODER_FILE
from mmcr.task_vectors import TaskVector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--zeroshot", default=None)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--scale", type=float, default=0.3)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    zeroshot_path = Path(args.zeroshot) if args.zeroshot is not None else checkpoint_root / "zeroshot.pt"

    task_vectors = []
    for dataset in args.datasets:
        encoder_path = checkpoint_root / dataset / ENCODER_FILE
        print(f"Loading task vector: {dataset} ({encoder_path})")
        task_vectors.append(TaskVector.from_checkpoints(zeroshot_path, encoder_path))

    task_vector_sum = sum(task_vectors)
    merged_state = task_vector_sum.apply_to_checkpoint(zeroshot_path, scaling_coef=args.scale)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_state, output_path)
    print(f"Saved merged encoder to {output_path}")


if __name__ == "__main__":
    main()
