from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from mmcr.checkpoints import load_encoder, load_head
from mmcr.evaluation import resolve_head_path
from mmcr.models import DEFAULT_ARCH
from mmcr.utils import build_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate synthetic inputs by scoring candidate encoders with teacher-logit proxy metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--synthesis-root", default="synthesis_data/generated")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME=ENCODER.pt",
        help="Candidate merged/source encoder to score. Can be passed multiple times.",
    )
    parser.add_argument(
        "--real-result",
        action="append",
        default=[],
        metavar="NAME=RESULT.json",
        help="Optional eval.py results JSON for a candidate. NAME must match --candidate.",
    )
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--include-rejected",
        action="store_true",
        help="Use all synthetic samples instead of filtering by accepted_mask.pt.",
    )
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument("--agreement-weight", type=float, default=0.5)
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    parser.add_argument("--output-json", default="results/synthetic_proxy_scores.json")
    parser.add_argument("--output-txt", default="results/synthetic_proxy_scores.txt")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_named_paths(items: list[str], option_name: str) -> dict[str, Path]:
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{option_name} must use NAME=PATH format, got: {item}")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ValueError(f"{option_name} must use NAME=PATH format, got: {item}")
        if name in parsed:
            raise ValueError(f"Duplicate {option_name} name: {name}")
        parsed[name] = Path(value)
    return parsed


def infer_datasets(synthesis_root: Path) -> list[str]:
    summary_path = synthesis_root / "summary.json"
    if summary_path.exists():
        payload = load_json(summary_path)
        if isinstance(payload.get("datasets"), list):
            return [str(dataset) for dataset in payload["datasets"]]
        if isinstance(payload.get("results"), list):
            return [str(row["dataset"]) for row in payload["results"] if "dataset" in row]

    datasets = sorted(path.name for path in synthesis_root.iterdir() if (path / "metadata.json").exists())
    if not datasets:
        raise ValueError(f"Could not infer datasets from {synthesis_root}. Pass --datasets explicitly.")
    return datasets


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def batched_proxy_metrics(
    encoder,
    head,
    inputs: torch.Tensor,
    teacher_logits: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    if inputs.numel() == 0 or inputs.shape[0] == 0:
        raise ValueError("No synthetic samples available after masking.")

    total_kl = 0.0
    total_agreement = 0.0
    total_entropy = 0.0
    total_samples = 0

    for start in range(0, inputs.shape[0], batch_size):
        batch_inputs = inputs[start : start + batch_size].to(device, non_blocking=True)
        batch_teacher = teacher_logits[start : start + batch_size].to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            merged_logits = head(encoder(batch_inputs))

        teacher_probs = F.softmax(batch_teacher, dim=-1)
        per_sample_kl = F.kl_div(
            F.log_softmax(merged_logits, dim=-1),
            teacher_probs,
            reduction="none",
        ).sum(dim=-1)
        agreement = (merged_logits.argmax(dim=-1) == batch_teacher.argmax(dim=-1)).float()
        merged_entropy = entropy_from_logits(merged_logits)

        count = batch_inputs.shape[0]
        total_kl += float(per_sample_kl.sum().item())
        total_agreement += float(agreement.sum().item())
        total_entropy += float(merged_entropy.sum().item())
        total_samples += count

    return {
        "kl": total_kl / total_samples,
        "agreement": total_agreement / total_samples,
        "merged_entropy": total_entropy / total_samples,
        "num_samples": total_samples,
    }


def load_synthetic_dataset(synthesis_root: Path, dataset: str, include_rejected: bool):
    folder = synthesis_root / dataset
    required = [folder / "inputs.pt", folder / "teacher_logits.pt", folder / "metadata.json"]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing synthetic file(s): " + ", ".join(str(path) for path in missing))

    inputs = torch.load(folder / "inputs.pt", map_location="cpu")
    teacher_logits = torch.load(folder / "teacher_logits.pt", map_location="cpu")
    metadata = load_json(folder / "metadata.json")
    if inputs.shape[0] != teacher_logits.shape[0]:
        raise ValueError(f"{dataset}: inputs and teacher_logits have different lengths.")

    if not include_rejected and (folder / "accepted_mask.pt").exists():
        mask = torch.load(folder / "accepted_mask.pt", map_location="cpu").bool()
        if mask.shape[0] != inputs.shape[0]:
            raise ValueError(f"{dataset}: accepted_mask length does not match inputs.")
        if mask.any():
            inputs = inputs[mask]
            teacher_logits = teacher_logits[mask]
        else:
            raise ValueError(f"{dataset}: accepted_mask has zero accepted samples. Use --include-rejected or regenerate data.")

    return inputs, teacher_logits, metadata


def extract_accuracy(payload, dataset: str) -> float | None:
    if isinstance(payload, dict):
        if dataset in payload and isinstance(payload[dataset], dict) and "acc" in payload[dataset]:
            return float(payload[dataset]["acc"])
        results = payload.get("results")
        if isinstance(results, dict) and dataset in results and isinstance(results[dataset], dict) and "acc" in results[dataset]:
            return float(results[dataset]["acc"])
        comparison = payload.get("comparison")
        if isinstance(comparison, dict):
            rows = comparison.get("rows")
            if isinstance(rows, list):
                for row in rows:
                    if row.get("dataset") == dataset and "merged_acc" in row:
                        return float(row["merged_acc"])
    return None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + end - 1) / 2.0 + 1.0
        for original_index, _ in indexed[index:end]:
            result[original_index] = rank
        index = end
    return result


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(ranks(xs), ranks(ys))


def build_correlation(proxy_results: dict, real_results: dict[str, dict], datasets: list[str]) -> dict:
    correlations = {}
    for dataset in [*datasets, "average"]:
        rows = []
        for candidate_name, candidate_results in proxy_results.items():
            real_by_dataset = real_results.get(candidate_name, {})
            real_acc = real_by_dataset.get(dataset)
            proxy_row = candidate_results.get(dataset)
            if proxy_row is not None and real_acc is not None:
                rows.append(
                    {
                        "candidate": candidate_name,
                        "proxy_score": float(proxy_row["proxy_score"]),
                        "real_acc": float(real_acc),
                    }
                )
        xs = [row["proxy_score"] for row in rows]
        ys = [row["real_acc"] for row in rows]
        correlations[dataset] = {
            "num_candidates": len(rows),
            "pearson": pearson(xs, ys),
            "spearman": spearman(xs, ys),
            "rows": rows,
        }
    return correlations


def metric_value(row: dict, metric: str) -> float:
    if metric == "neg_kl":
        return -float(row["kl"])
    if metric == "agreement":
        return float(row["agreement"])
    if metric == "neg_entropy":
        return -float(row["merged_entropy"])
    if metric == "proxy_score":
        return float(row["proxy_score"])
    raise ValueError(f"Unknown metric: {metric}")


def build_metric_correlations(proxy_results: dict, real_results: dict[str, dict], datasets: list[str]) -> dict:
    metrics = ["proxy_score", "neg_kl", "agreement", "neg_entropy"]
    correlations = {}
    for metric in metrics:
        correlations[metric] = {}
        for dataset in [*datasets, "average"]:
            rows = []
            for candidate_name, candidate_results in proxy_results.items():
                real_acc = real_results.get(candidate_name, {}).get(dataset)
                proxy_row = candidate_results.get(dataset)
                if proxy_row is not None and real_acc is not None:
                    rows.append(
                        {
                            "candidate": candidate_name,
                            "metric_value": metric_value(proxy_row, metric),
                            "real_acc": float(real_acc),
                        }
                    )
            xs = [row["metric_value"] for row in rows]
            ys = [row["real_acc"] for row in rows]
            correlations[metric][dataset] = {
                "num_candidates": len(rows),
                "pearson": pearson(xs, ys),
                "spearman": spearman(xs, ys),
                "rows": rows,
            }
    return correlations


def format_table(payload: dict, datasets: list[str]) -> str:
    lines = [
        "# Synthetic Proxy Validation",
        "",
        "| Candidate | Dataset | KL | Agreement | Entropy | Proxy Score | Samples | Real ACC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    real_results = payload.get("real_results", {})
    for candidate_name, candidate_results in payload["proxy_results"].items():
        for dataset in [*datasets, "average"]:
            row = candidate_results[dataset]
            real_acc = real_results.get(candidate_name, {}).get(dataset)
            real_text = "" if real_acc is None else f"{real_acc * 100:.2f}%"
            samples_text = "" if dataset == "average" else str(row["num_samples"])
            lines.append(
                f"| {candidate_name} | {dataset} | {row['kl']:.4f} | {row['agreement']:.4f} | "
                f"{row['merged_entropy']:.4f} | {row['proxy_score']:.4f} | {samples_text} | {real_text} |"
            )

    correlations = payload.get("correlations")
    if correlations:
        lines.extend([
            "",
            "## Correlation",
            "",
            "| Dataset | Candidates | Pearson | Spearman |",
            "| --- | ---: | ---: | ---: |",
        ])
        for dataset in [*datasets, "average"]:
            row = correlations[dataset]
            pearson_text = "n/a" if row["pearson"] is None else f"{row['pearson']:.4f}"
            spearman_text = "n/a" if row["spearman"] is None else f"{row['spearman']:.4f}"
            lines.append(f"| {dataset} | {row['num_candidates']} | {pearson_text} | {spearman_text} |")

    metric_correlations = payload.get("metric_correlations")
    if metric_correlations:
        lines.extend([
            "",
            "## Metric Correlation",
            "",
            "| Metric | Dataset | Candidates | Pearson | Spearman |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for metric, metric_rows in metric_correlations.items():
            for dataset in [*datasets, "average"]:
                row = metric_rows[dataset]
                pearson_text = "n/a" if row["pearson"] is None else f"{row['pearson']:.4f}"
                spearman_text = "n/a" if row["spearman"] is None else f"{row['spearman']:.4f}"
                lines.append(
                    f"| {metric} | {dataset} | {row['num_candidates']} | {pearson_text} | {spearman_text} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    synthesis_root = Path(args.synthesis_root)
    datasets = args.datasets or infer_datasets(synthesis_root)
    candidates = parse_named_paths(args.candidate, "--candidate")
    real_paths = parse_named_paths(args.real_result, "--real-result")
    unknown_real = sorted(set(real_paths) - set(candidates))
    if unknown_real:
        raise ValueError("--real-result names without matching --candidate: " + ", ".join(unknown_real))

    for name, path in candidates.items():
        if not path.exists():
            raise FileNotFoundError(f"Candidate {name} encoder does not exist: {path}")
    for name, path in real_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Real result for {name} does not exist: {path}")

    device = build_device(args.gpu)
    proxy_results = {}
    synthetic_quality = {}

    synthetic_cache = {}
    for dataset in datasets:
        inputs, teacher_logits, metadata = load_synthetic_dataset(synthesis_root, dataset, args.include_rejected)
        synthetic_cache[dataset] = (inputs, teacher_logits)
        synthetic_quality[dataset] = {
            key: metadata.get(key)
            for key in (
                "num_samples",
                "accepted_samples",
                "accepted_ratio",
                "mean_confidence",
                "mean_entropy",
                "target_match_rate",
            )
        }
        synthetic_quality[dataset]["used_samples"] = int(inputs.shape[0])

    for candidate_name, encoder_path in candidates.items():
        encoder = load_encoder(encoder_path, arch=args.arch, device=device)
        encoder.eval().requires_grad_(False)
        candidate_rows = {}

        for dataset in tqdm(datasets, desc=f"proxy {candidate_name}"):
            inputs, teacher_logits = synthetic_cache[dataset]
            head = load_head(resolve_head_path(dataset, args.checkpoint_root), device=device)
            head.eval().requires_grad_(False)
            metrics = batched_proxy_metrics(
                encoder,
                head,
                inputs,
                teacher_logits,
                batch_size=args.batch_size,
                device=device,
                amp=args.amp,
            )
            metrics["proxy_score"] = (
                -args.kl_weight * metrics["kl"]
                + args.agreement_weight * metrics["agreement"]
                - args.entropy_weight * metrics["merged_entropy"]
            )
            candidate_rows[dataset] = metrics
            del head

        count = len(datasets)
        candidate_rows["average"] = {
            "kl": sum(candidate_rows[dataset]["kl"] for dataset in datasets) / count,
            "agreement": sum(candidate_rows[dataset]["agreement"] for dataset in datasets) / count,
            "merged_entropy": sum(candidate_rows[dataset]["merged_entropy"] for dataset in datasets) / count,
            "proxy_score": sum(candidate_rows[dataset]["proxy_score"] for dataset in datasets) / count,
        }
        proxy_results[candidate_name] = candidate_rows
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    real_results = {}
    for candidate_name, path in real_paths.items():
        payload = load_json(path)
        rows = {}
        for dataset in datasets:
            acc = extract_accuracy(payload, dataset)
            if acc is not None:
                rows[dataset] = acc
        if rows:
            rows["average"] = sum(rows.values()) / len(rows)
        real_results[candidate_name] = rows

    output = {
        "synthesis_root": str(synthesis_root),
        "datasets": datasets,
        "candidates": {name: str(path) for name, path in candidates.items()},
        "real_result_paths": {name: str(path) for name, path in real_paths.items()},
        "weights": {
            "kl_weight": args.kl_weight,
            "agreement_weight": args.agreement_weight,
            "entropy_weight": args.entropy_weight,
        },
        "include_rejected": args.include_rejected,
        "synthetic_quality": synthetic_quality,
        "proxy_results": proxy_results,
        "real_results": real_results,
    }
    if real_results:
        output["correlations"] = build_correlation(proxy_results, real_results, datasets)
        output["metric_correlations"] = build_metric_correlations(proxy_results, real_results, datasets)

    output_json = Path(args.output_json)
    output_txt = Path(args.output_txt)
    write_json(output_json, output)
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    output_txt.write_text(format_table(output, datasets), encoding="utf-8")
    print(f"Saved proxy validation JSON to {output_json}")
    print(f"Saved proxy validation table to {output_txt}")


if __name__ == "__main__":
    main()
