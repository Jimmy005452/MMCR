import argparse
import json
from pathlib import Path

import torch

from mmcr.evaluation import evaluate_encoder
from mmcr.models import DEFAULT_ARCH
from mmcr.utils import build_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--head", default=None, help="Optional head path for single-dataset evaluation.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--results-txt", default=None)
    return parser.parse_args()


def format_results(results):
    lines = []
    for dataset, metrics in results.items():
        if dataset == "average":
            continue
        lines.append(f"{dataset}: ACC={metrics['acc'] * 100:.2f}%")

    if "average" in results:
        lines.append(f"Average ACC={results['average']['acc'] * 100:.2f}%")

    return lines


def main():
    args = parse_args()
    device = build_device(args.gpu)
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print("Using device: cpu")

    results = evaluate_encoder(
        encoder_path=args.encoder,
        datasets=args.datasets,
        checkpoint_root=args.checkpoint_root,
        data_root=args.data_root,
        arch=args.arch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        amp=args.amp,
        download=not args.no_download,
        head_path=args.head,
    )

    result_lines = format_results(results)
    for line in result_lines:
        print(line)

    if args.results_json is not None:
        output_path = Path(args.results_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {output_path}")

    if args.results_txt is not None:
        output_path = Path(args.results_txt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
