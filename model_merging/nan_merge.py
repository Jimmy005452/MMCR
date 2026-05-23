import argparse
from pathlib import Path

import torch

from mmcr.checkpoints import ENCODER_FILE
from mmcr.nan import nan_merge_checkpoints
from mmcr.utils import write_json


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--zeroshot", default=None)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--merge-method", choices=["ta", "ties"], default="ta")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--top-k", type=float, default=20, help="Only used when --merge-method ties.")
    parser.add_argument("--merge-func", choices=["dis-sum", "dis-mean"], default="dis-mean")
    parser.add_argument("--norm-target", choices=["finetuned", "task-vector"], default="finetuned")
    parser.add_argument("--global-scale", choices=["m_half", "none"], default="m_half")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coefficients-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_root = Path(args.checkpoint_root)
    zeroshot_path = Path(args.zeroshot) if args.zeroshot is not None else checkpoint_root / "zeroshot.pt"
    output_path = Path(args.output)
    coefficients_path = (
        Path(args.coefficients_json)
        if args.coefficients_json is not None
        else output_path.with_suffix(".coefficients.json")
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
    if coefficients_path.exists() and not args.overwrite:
        raise FileExistsError(f"{coefficients_path} already exists. Use --overwrite to replace it.")

    encoder_paths = [checkpoint_root / dataset / ENCODER_FILE for dataset in args.datasets]
    for dataset, path in zip(args.datasets, encoder_paths):
        print(f"Using {dataset}: {path}")
        if not path.exists():
            raise FileNotFoundError(path)
    if not zeroshot_path.exists():
        raise FileNotFoundError(zeroshot_path)

    merged_state, metadata = nan_merge_checkpoints(
        zeroshot_path=zeroshot_path,
        finetuned_paths=encoder_paths,
        scaling_coef=args.scale,
        merge_method=args.merge_method,
        top_k_percent=args.top_k,
        merge_func=args.merge_func,
        norm_target=args.norm_target,
        global_scale=args.global_scale,
        eps=args.eps,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged_state, output_path)

    metadata.update(
        {
            "datasets": args.datasets,
            "zeroshot_path": str(zeroshot_path),
            "encoder_paths": [str(path) for path in encoder_paths],
            "output_path": str(output_path),
        }
    )
    write_json(coefficients_path, metadata)

    print(f"Saved NAN-{args.merge_method.upper()} encoder to {output_path}")
    print(f"Saved NAN coefficients to {coefficients_path}")
    for dataset, norm, coefficient in zip(args.datasets, metadata["norms"], metadata["coefficients"]):
        print(f"{dataset}: norm={norm:.6g}, coefficient={coefficient:.6g}")


if __name__ == "__main__":
    main()
