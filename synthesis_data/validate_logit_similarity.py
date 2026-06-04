from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from mmcr.checkpoints import ENCODER_FILE, load_encoder, load_head
from mmcr.data import build_loader, get_num_classes, normalize_dataset_key
from mmcr.evaluation import resolve_head_path
from mmcr.models import DEFAULT_ARCH, build_model_transforms
from mmcr.utils import build_device, write_json


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
        description="Compare synthetic source-model logits with real same-class source-model logits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--synthesis-root", default="synthesis_data/generated")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--real-split", choices=["val", "test"], default="val")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--real-samples-per-class", type=int, default=16)
    parser.add_argument("--synthetic-samples-per-class", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--include-rejected", action="store_true")
    parser.add_argument(
        "--logit-space",
        choices=["logits", "log_probs", "probs"],
        default="logits",
        help="Representation used before cosine similarity.",
    )
    parser.add_argument("--output-json", default="results/synthetic_logit_similarity.json")
    parser.add_argument("--output-txt", default="results/synthetic_logit_similarity.txt")
    return parser.parse_args()


def resolve_source_encoder_path(checkpoint_root: Path, dataset: str) -> Path:
    direct_path = checkpoint_root / dataset / ENCODER_FILE
    if direct_path.exists():
        return direct_path
    normalized_path = checkpoint_root / normalize_dataset_key(dataset) / ENCODER_FILE
    return normalized_path if normalized_path.exists() else direct_path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def transform_logits(logits: torch.Tensor, logit_space: str) -> torch.Tensor:
    if logit_space == "logits":
        return logits.float()
    if logit_space == "log_probs":
        return F.log_softmax(logits.float(), dim=-1)
    if logit_space == "probs":
        return F.softmax(logits.float(), dim=-1)
    raise RuntimeError(f"Unknown logit_space: {logit_space}")


def normalize_rows(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), dim=-1, eps=1e-8)


def extract_targets(batch) -> torch.Tensor:
    targets = batch[1]
    if torch.is_tensor(targets):
        return targets.long()
    return torch.tensor(targets, dtype=torch.long)


@torch.no_grad()
def collect_real_logits_by_class(
    encoder,
    head,
    dataset: str,
    *,
    data_root: Path,
    split: str,
    arch: str,
    batch_size: int,
    num_workers: int,
    samples_per_class: int,
    device: torch.device,
    amp: bool,
    download: bool,
    logit_space: str,
) -> dict[int, torch.Tensor]:
    _, eval_transform, _ = build_model_transforms(arch, pretrained=False)
    loader, num_classes = build_loader(
        dataset,
        data_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
        eval_transform=eval_transform,
        shuffle=False,
    )
    buckets: dict[int, list[torch.Tensor]] = defaultdict(list)
    remaining = None if samples_per_class <= 0 else {class_id: samples_per_class for class_id in range(num_classes)}

    for batch in tqdm(loader, desc=f"real logits {dataset}", leave=False):
        images = batch[0].to(device, non_blocking=True)
        targets = extract_targets(batch).cpu()
        needed_positions = []
        needed_targets = []
        for index, target in enumerate(targets.tolist()):
            if remaining is not None and remaining.get(int(target), 0) <= 0:
                continue
            needed_positions.append(index)
            needed_targets.append(int(target))
            if remaining is not None:
                remaining[int(target)] -= 1

        if not needed_positions:
            if remaining is not None and all(value <= 0 for value in remaining.values()):
                break
            continue

        selected_images = images[needed_positions]
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = head(encoder(selected_images))
        logits = transform_logits(logits.detach().cpu(), logit_space)
        for row, target in zip(logits, needed_targets):
            buckets[int(target)].append(row)

        if remaining is not None and all(value <= 0 for value in remaining.values()):
            break

    return {class_id: torch.stack(rows) for class_id, rows in buckets.items() if rows}


def load_synthetic_dataset(
    synthesis_root: Path,
    dataset: str,
    *,
    include_rejected: bool,
    samples_per_class: int,
    logit_space: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    folder = synthesis_root / dataset
    required = [folder / "teacher_logits.pt", folder / "pseudo_labels.pt", folder / "metadata.json"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing synthetic file(s): " + ", ".join(str(path) for path in missing))

    teacher_logits = torch.load(folder / "teacher_logits.pt", map_location="cpu")
    pseudo_labels = torch.load(folder / "pseudo_labels.pt", map_location="cpu").long()
    metadata = load_json(folder / "metadata.json")
    if teacher_logits.shape[0] != pseudo_labels.shape[0]:
        raise ValueError(f"{dataset}: teacher_logits and pseudo_labels have different lengths.")

    keep = torch.ones(pseudo_labels.shape[0], dtype=torch.bool)
    mask_path = folder / "accepted_mask.pt"
    if not include_rejected and mask_path.exists():
        accepted = torch.load(mask_path, map_location="cpu").bool()
        if accepted.shape[0] != pseudo_labels.shape[0]:
            raise ValueError(f"{dataset}: accepted_mask length does not match pseudo_labels.")
        if not accepted.any():
            raise ValueError(f"{dataset}: accepted_mask has zero accepted samples. Use --include-rejected.")
        keep &= accepted

    if samples_per_class > 0:
        counts: dict[int, int] = defaultdict(int)
        class_keep = torch.zeros_like(keep)
        for index, label in enumerate(pseudo_labels.tolist()):
            if not keep[index]:
                continue
            if counts[int(label)] >= samples_per_class:
                continue
            class_keep[index] = True
            counts[int(label)] += 1
        keep &= class_keep

    teacher_logits = transform_logits(teacher_logits[keep], logit_space)
    pseudo_labels = pseudo_labels[keep]
    original_indices = torch.nonzero(keep, as_tuple=False).flatten()
    return teacher_logits, pseudo_labels, original_indices, metadata


def compute_dataset_metrics(
    real_by_class: dict[int, torch.Tensor],
    synthetic_logits: torch.Tensor,
    synthetic_labels: torch.Tensor,
    *,
    num_classes: int,
    top_k: int,
) -> tuple[dict, dict[int, dict]]:
    if synthetic_logits.shape[0] == 0:
        raise ValueError("No synthetic samples available after filtering.")

    real_norm_by_class = {class_id: normalize_rows(rows) for class_id, rows in real_by_class.items()}
    real_all_rows = []
    real_all_labels = []
    for class_id, rows in real_norm_by_class.items():
        real_all_rows.append(rows)
        real_all_labels.extend([class_id] * rows.shape[0])
    if not real_all_rows:
        raise ValueError("No real logits were collected.")
    real_all = torch.cat(real_all_rows, dim=0)
    real_all_labels_tensor = torch.tensor(real_all_labels, dtype=torch.long)

    centroids = torch.zeros(num_classes, synthetic_logits.shape[-1], dtype=torch.float32)
    centroid_mask = torch.zeros(num_classes, dtype=torch.bool)
    for class_id, rows in real_by_class.items():
        centroids[class_id] = rows.float().mean(dim=0)
        centroid_mask[class_id] = True
    centroids = normalize_rows(centroids)

    syn_norm = normalize_rows(synthetic_logits)
    rows = []
    per_class_rows: dict[int, list[dict]] = defaultdict(list)
    for index, (syn_row, label_tensor) in enumerate(zip(syn_norm, synthetic_labels)):
        label = int(label_tensor.item())
        if label not in real_norm_by_class:
            continue
        same = real_norm_by_class[label]
        same_sims = same @ syn_row
        same_nn = float(same_sims.max().item())
        topk_count = min(max(1, top_k), same_sims.numel())
        same_topk = float(torch.topk(same_sims, k=topk_count).values.mean().item())

        wrong_mask = real_all_labels_tensor != label
        wrong_sims = real_all[wrong_mask] @ syn_row
        wrong_nn = float(wrong_sims.max().item()) if wrong_sims.numel() else float("nan")

        centroid_sims = centroids @ syn_row
        centroid_sims[~centroid_mask] = -float("inf")
        centroid_pred = int(torch.argmax(centroid_sims).item())
        row = {
            "sample_index": int(index),
            "pseudo_label": label,
            "same_class_nn_cos": same_nn,
            "same_class_topk_cos": same_topk,
            "wrong_class_nn_cos": wrong_nn,
            "logit_margin": same_nn - wrong_nn,
            "top1_match_to_real_centroid": float(centroid_pred == label),
            "nearest_centroid_class": centroid_pred,
        }
        rows.append(row)
        per_class_rows[label].append(row)

    def summarize(items: list[dict]) -> dict:
        keys = [
            "same_class_nn_cos",
            "same_class_topk_cos",
            "wrong_class_nn_cos",
            "logit_margin",
            "top1_match_to_real_centroid",
        ]
        return {
            key: sum(float(item[key]) for item in items) / len(items)
            for key in keys
            if items
        } | {"num_synthetic": len(items)}

    per_class = {class_id: summarize(items) for class_id, items in sorted(per_class_rows.items())}
    summary = summarize(rows)
    summary["num_real"] = int(sum(rows.shape[0] for rows in real_by_class.values()))
    summary["num_real_classes"] = int(len(real_by_class))
    summary["num_synthetic_classes"] = int(len(per_class))
    return summary, per_class


def write_text_report(path: Path, output: dict) -> None:
    lines = ["# Synthetic Logit Similarity Validation", ""]
    lines.append("| Dataset | Real | Synthetic | Same NN | Same TopK | Wrong NN | Margin | Centroid Match |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for dataset, row in output["results"].items():
        lines.append(
            "| {dataset} | {num_real} | {num_synthetic} | {same_nn:.4f} | {same_topk:.4f} | {wrong_nn:.4f} | {margin:.4f} | {match:.4f} |".format(
                dataset=dataset,
                num_real=row["summary"]["num_real"],
                num_synthetic=row["summary"]["num_synthetic"],
                same_nn=row["summary"]["same_class_nn_cos"],
                same_topk=row["summary"]["same_class_topk_cos"],
                wrong_nn=row["summary"]["wrong_class_nn_cos"],
                margin=row["summary"]["logit_margin"],
                match=row["summary"]["top1_match_to_real_centroid"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.real_samples_per_class < 0:
        raise ValueError("--real-samples-per-class must be non-negative.")
    if args.synthetic_samples_per_class < 0:
        raise ValueError("--synthetic-samples-per-class must be non-negative.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    device = build_device(args.gpu)
    checkpoint_root = Path(args.checkpoint_root)
    synthesis_root = Path(args.synthesis_root)
    output = {
        "config": vars(args),
        "results": {},
    }

    for dataset in args.datasets:
        encoder_path = resolve_source_encoder_path(checkpoint_root, dataset)
        head_path = resolve_head_path(dataset, checkpoint_root)
        if not encoder_path.exists():
            raise FileNotFoundError(encoder_path)
        if not head_path.exists():
            raise FileNotFoundError(head_path)

        encoder = load_encoder(encoder_path, arch=args.arch, device=device)
        head = load_head(head_path, device=device)
        encoder.eval().requires_grad_(False)
        head.eval().requires_grad_(False)

        real_by_class = collect_real_logits_by_class(
            encoder,
            head,
            dataset,
            data_root=Path(args.data_root),
            split=args.real_split,
            arch=args.arch,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            samples_per_class=args.real_samples_per_class,
            device=device,
            amp=args.amp,
            download=not args.no_download,
            logit_space=args.logit_space,
        )
        synthetic_logits, synthetic_labels, original_indices, metadata = load_synthetic_dataset(
            synthesis_root,
            dataset,
            include_rejected=args.include_rejected,
            samples_per_class=args.synthetic_samples_per_class,
            logit_space=args.logit_space,
        )
        summary, per_class = compute_dataset_metrics(
            real_by_class,
            synthetic_logits,
            synthetic_labels,
            num_classes=get_num_classes(dataset),
            top_k=args.top_k,
        )
        output["results"][dataset] = {
            "summary": summary,
            "per_class": {str(class_id): values for class_id, values in per_class.items()},
            "metadata": metadata,
            "used_synthetic_indices": original_indices.tolist(),
        }

        del encoder, head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json(Path(args.output_json), output)
    write_text_report(Path(args.output_txt), output)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_txt}")


if __name__ == "__main__":
    main()
