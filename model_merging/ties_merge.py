import argparse
from pathlib import Path

import torch

from mmcr.checkpoints import ENCODER_FILE
from mmcr.adamerging import load_ties_task_vectors


def merge_streaming_ties(
    zeroshot_path: Path,
    finetuned_paths: list[Path],
    top_k_percent: float,
    scaling_coef: float,
    merge_func: str,
) -> dict[str, torch.Tensor]:
    pretrained_state, task_vectors, _ = load_ties_task_vectors(
        zeroshot_path=zeroshot_path,
        finetuned_paths=finetuned_paths,
        top_k_percent=top_k_percent,
        map_location="cpu",
    )

    if not task_vectors:
        raise ValueError("No task vectors were loaded.")

    merged_state = {key: value.detach().cpu().clone() for key, value in pretrained_state.items()}
    merge_keys = sorted(set().union(*(task_vector.keys() for task_vector in task_vectors)))

    print(f"Streaming merge over {len(merge_keys)} TIES-selected tensors.", flush=True)
    for key in merge_keys:
        base_value = pretrained_state[key].detach().cpu()
        merged_delta = torch.zeros_like(base_value, dtype=torch.float32)

        if merge_func == "dis-sum":
            for task_vector in task_vectors:
                if key in task_vector:
                    merged_delta += task_vector[key].detach().cpu().float()
        elif merge_func == "dis-mean":
            counts = torch.zeros_like(base_value, dtype=torch.int16)
            for task_vector in task_vectors:
                if key not in task_vector:
                    continue
                delta = task_vector[key].detach().cpu().float()
                merged_delta += delta
                counts += (delta != 0).to(dtype=counts.dtype)
            merged_delta = merged_delta / counts.clamp(min=1).to(dtype=merged_delta.dtype)
        else:
            raise ValueError("merge_func must be one of: dis-sum, dis-mean")

        merged_state[key] = (base_value.float() + scaling_coef * merged_delta).to(dtype=base_value.dtype)

    return merged_state


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

    merged_state = merge_streaming_ties(
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
