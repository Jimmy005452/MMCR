from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from mmcr.checkpoints import ENCODER_FILE, load_encoder, load_head
from mmcr.data import get_num_classes, normalize_dataset_key
from mmcr.evaluation import resolve_head_path
from mmcr.models import DEFAULT_ARCH, build_model_transforms
from mmcr.utils import build_device, seed_everything, write_json


DEFAULT_DATASETS = [
    "sun397",
    "stanford_cars",
    "resisc45",
    "eurosat",
    "svhn",
    "gtsrb",
    "mnist",
    "dtd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic model-ready inputs from source models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--output-dir", default="synthesis_data/generated")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--samples-per-dataset", type=int, default=64)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="When positive, overrides --samples-per-dataset and generates this many images per class.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument("--consistency-coef", type=float, default=0.05)
    parser.add_argument("--tv-coef", type=float, default=2e-4)
    parser.add_argument("--l2-coef", type=float, default=1e-4)
    parser.add_argument("--aug-shift", type=int, default=8)
    parser.add_argument("--aug-noise-std", type=float, default=0.03)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--max-entropy", type=float, default=math.inf)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_source_encoder_path(checkpoint_root: Path, dataset: str) -> Path:
    direct_path = checkpoint_root / dataset / ENCODER_FILE
    if direct_path.exists():
        return direct_path

    normalized_path = checkpoint_root / normalize_dataset_key(dataset) / ENCODER_FILE
    return normalized_path if normalized_path.exists() else direct_path


def require_existing(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required checkpoint(s): " + ", ".join(str(path) for path in missing))


def stable_dataset_seed(seed: int, dataset: str) -> int:
    offset = sum((index + 1) * ord(char) for index, char in enumerate(dataset))
    return seed + offset


def build_balanced_targets(
    dataset: str,
    samples_per_dataset: int,
    samples_per_class: int,
    generator: torch.Generator,
) -> torch.Tensor:
    num_classes = get_num_classes(dataset)
    if samples_per_class > 0:
        targets = torch.arange(num_classes).repeat_interleave(samples_per_class)
    else:
        if samples_per_dataset <= 0:
            raise ValueError("--samples-per-dataset must be positive when --samples-per-class is 0.")
        repeats = math.ceil(samples_per_dataset / num_classes)
        targets = torch.arange(num_classes).repeat(repeats)[:samples_per_dataset]
    return targets[torch.randperm(targets.numel(), generator=generator)]


def normalize_pixels(pixels: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (pixels - mean) / std


def random_shift(pixels: torch.Tensor, max_shift: int) -> torch.Tensor:
    if max_shift <= 0:
        return pixels
    shift_y = int(torch.randint(-max_shift, max_shift + 1, (1,), device=pixels.device).item())
    shift_x = int(torch.randint(-max_shift, max_shift + 1, (1,), device=pixels.device).item())
    return torch.roll(pixels, shifts=(shift_y, shift_x), dims=(-2, -1))


def augment_pixels(pixels: torch.Tensor, max_shift: int, noise_std: float) -> torch.Tensor:
    augmented = random_shift(pixels, max_shift)
    if noise_std > 0:
        augmented = augmented + torch.randn_like(augmented) * noise_std
    return augmented.clamp(0.0, 1.0)


def total_variation(pixels: torch.Tensor) -> torch.Tensor:
    height_tv = (pixels[:, :, 1:, :] - pixels[:, :, :-1, :]).abs().mean()
    width_tv = (pixels[:, :, :, 1:] - pixels[:, :, :, :-1]).abs().mean()
    return height_tv + width_tv


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def symmetric_kl(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    left_log_probs = F.log_softmax(left_logits, dim=-1)
    right_log_probs = F.log_softmax(right_logits, dim=-1)
    left_probs = left_log_probs.exp()
    right_probs = right_log_probs.exp()
    left_to_right = F.kl_div(right_log_probs, left_probs.detach(), reduction="batchmean")
    right_to_left = F.kl_div(left_log_probs, right_probs.detach(), reduction="batchmean")
    return 0.5 * (left_to_right + right_to_left)


def classifier_forward(encoder, head, inputs: torch.Tensor, device: torch.device, amp: bool) -> torch.Tensor:
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        features = encoder(inputs)
        return head(features)


def synthesize_batch(
    encoder,
    head,
    targets: torch.Tensor,
    *,
    image_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    steps: int,
    lr: float,
    entropy_coef: float,
    consistency_coef: float,
    tv_coef: float,
    l2_coef: float,
    aug_shift: int,
    aug_noise_std: float,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.randn(targets.numel(), 3, image_size, image_size, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([raw], lr=lr)
    targets = targets.to(device)

    for _ in range(steps):
        pixels = raw.sigmoid()
        augmented = augment_pixels(pixels, max_shift=aug_shift, noise_std=aug_noise_std)
        logits = classifier_forward(encoder, head, normalize_pixels(augmented, mean, std), device, amp)
        entropy = entropy_from_logits(logits).mean()
        loss = F.cross_entropy(logits, targets) + entropy_coef * entropy
        loss = loss + tv_coef * total_variation(pixels)
        loss = loss + l2_coef * (pixels - 0.5).pow(2).mean()

        if consistency_coef > 0:
            augmented_again = augment_pixels(pixels, max_shift=aug_shift, noise_std=aug_noise_std)
            logits_again = classifier_forward(encoder, head, normalize_pixels(augmented_again, mean, std), device, amp)
            loss = loss + consistency_coef * symmetric_kl(logits, logits_again)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    pixels = raw.sigmoid().detach()
    inputs = normalize_pixels(pixels, mean, std)
    with torch.no_grad():
        teacher_logits = classifier_forward(encoder, head, inputs, device, amp).detach()
    return pixels.cpu(), teacher_logits.cpu()


def synthesize_dataset(args: argparse.Namespace, dataset: str, device: torch.device, data_config: dict) -> dict:
    checkpoint_root = Path(args.checkpoint_root)
    encoder_path = resolve_source_encoder_path(checkpoint_root, dataset)
    head_path = resolve_head_path(dataset, checkpoint_root)
    require_existing([encoder_path, head_path])

    output_dir = Path(args.output_dir) / dataset
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        print(f"Skipping {dataset}: {metadata_path} already exists. Pass --overwrite to regenerate.")
        return {"dataset": dataset, "skipped": True, "output_dir": str(output_dir)}

    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(encoder_path, arch=args.arch, device=device)
    head = load_head(head_path, device=device)
    encoder.eval().requires_grad_(False)
    head.eval().requires_grad_(False)

    image_size = int(data_config["input_size"][-1])
    mean = torch.tensor(data_config["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(data_config["std"], device=device).view(1, 3, 1, 1)

    generator = torch.Generator().manual_seed(stable_dataset_seed(args.seed, dataset))
    targets = build_balanced_targets(
        dataset,
        samples_per_dataset=args.samples_per_dataset,
        samples_per_class=args.samples_per_class,
        generator=generator,
    )

    pixels_all = []
    logits_all = []
    progress = tqdm(range(0, targets.numel(), args.batch_size), desc=f"synthesize {dataset}")
    for start in progress:
        batch_targets = targets[start : start + args.batch_size]
        pixels, teacher_logits = synthesize_batch(
            encoder,
            head,
            batch_targets,
            image_size=image_size,
            mean=mean,
            std=std,
            device=device,
            steps=args.steps,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            consistency_coef=args.consistency_coef,
            tv_coef=args.tv_coef,
            l2_coef=args.l2_coef,
            aug_shift=args.aug_shift,
            aug_noise_std=args.aug_noise_std,
            amp=args.amp,
        )
        pixels_all.append(pixels)
        logits_all.append(teacher_logits)

    pixels = torch.cat(pixels_all, dim=0)
    teacher_logits = torch.cat(logits_all, dim=0)
    inputs = normalize_pixels(pixels, mean.cpu(), std.cpu())
    log_probs = F.log_softmax(teacher_logits, dim=-1)
    probs = log_probs.exp()
    confidence, predictions = probs.max(dim=-1)
    entropy = entropy_from_logits(teacher_logits)
    accepted_mask = (confidence >= args.min_confidence) & (entropy <= args.max_entropy)

    torch.save(inputs, output_dir / "inputs.pt")
    torch.save(pixels, output_dir / "pixels.pt")
    torch.save(teacher_logits, output_dir / "teacher_logits.pt")
    torch.save(targets, output_dir / "pseudo_labels.pt")
    torch.save(accepted_mask, output_dir / "accepted_mask.pt")

    metadata = {
        "dataset": dataset,
        "normalized_dataset": normalize_dataset_key(dataset),
        "num_classes": get_num_classes(dataset),
        "num_samples": int(targets.numel()),
        "accepted_samples": int(accepted_mask.sum().item()),
        "accepted_ratio": float(accepted_mask.float().mean().item()),
        "target_match_rate": float((predictions == targets).float().mean().item()),
        "mean_confidence": float(confidence.mean().item()),
        "mean_entropy": float(entropy.mean().item()),
        "max_entropy": float(entropy.max().item()),
        "min_confidence": float(confidence.min().item()),
        "image_size": image_size,
        "arch": args.arch,
        "encoder_path": str(encoder_path),
        "head_path": str(head_path),
        "files": {
            "inputs": str(output_dir / "inputs.pt"),
            "pixels": str(output_dir / "pixels.pt"),
            "teacher_logits": str(output_dir / "teacher_logits.pt"),
            "pseudo_labels": str(output_dir / "pseudo_labels.pt"),
            "accepted_mask": str(output_dir / "accepted_mask.pt"),
        },
        "synthesis": {
            "steps": args.steps,
            "lr": args.lr,
            "entropy_coef": args.entropy_coef,
            "consistency_coef": args.consistency_coef,
            "tv_coef": args.tv_coef,
            "l2_coef": args.l2_coef,
            "aug_shift": args.aug_shift,
            "aug_noise_std": args.aug_noise_std,
            "samples_per_dataset": args.samples_per_dataset,
            "samples_per_class": args.samples_per_class,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
    }
    write_json(metadata_path, metadata)

    del encoder, head
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    seed_everything(args.seed)
    device = build_device(args.gpu)
    _, _, data_config = build_model_transforms(args.arch, pretrained=False)

    summaries = []
    for dataset in args.datasets:
        summaries.append(synthesize_dataset(args, dataset, device, data_config))

    summary_path = Path(args.output_dir) / "summary.json"
    write_json(summary_path, {"datasets": args.datasets, "results": summaries})
    print(f"Wrote synthesis summary to {summary_path}")


if __name__ == "__main__":
    main()
