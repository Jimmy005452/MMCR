import argparse
from pathlib import Path

import torch

from mmcr.models import DEFAULT_ARCH, build_image_encoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/zeroshot.pt")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    encoder = build_image_encoder(arch=args.arch, pretrained=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), output_path)
    print(f"Saved zeroshot encoder to {output_path}")


if __name__ == "__main__":
    main()
