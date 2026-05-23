import argparse
import json
from pathlib import Path

import torch

from mmcr.checkpoints import ENCODER_FILE
from mmcr.data import normalize_dataset_key
from mmcr.evaluation import evaluate_encoder, resolve_head_path
from mmcr.models import DEFAULT_ARCH
from mmcr.utils import build_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument(
        "--run-results-json",
        default=None,
        help="Optional run results.json used to infer --datasets when omitted.",
    )
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
    parser.add_argument(
        "--single-model-results-json",
        default=None,
        help="Cache JSON for per-dataset single-source model accuracies.",
    )
    parser.add_argument(
        "--refresh-single-model-results",
        action="store_true",
        help="Recompute single-source model accuracies even if the cache exists.",
    )
    parser.add_argument(
        "--comparison-json",
        default=None,
        help="Optional path for merged-vs-single comparison JSON.",
    )
    parser.add_argument(
        "--comparison-txt",
        default=None,
        help="Optional path for merged-vs-single comparison table text.",
    )
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


def resolve_source_encoder_path(checkpoint_root: Path, dataset: str) -> Path:
    direct_path = checkpoint_root / dataset / ENCODER_FILE
    if direct_path.exists():
        return direct_path

    normalized_path = checkpoint_root / normalize_dataset_key(dataset) / ENCODER_FILE
    return normalized_path if normalized_path.exists() else direct_path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def extract_datasets_from_run_results(payload) -> list[str] | None:
    config = payload.get("config")
    if isinstance(config, dict) and isinstance(config.get("datasets"), list):
        return [str(dataset) for dataset in config["datasets"]]
    if isinstance(payload.get("datasets"), list):
        return [str(dataset) for dataset in payload["datasets"]]
    return None


def infer_run_results_path(args) -> Path | None:
    if args.run_results_json is not None:
        return Path(args.run_results_json)

    candidate = Path(args.encoder).parent / "results.json"
    return candidate if candidate.exists() else None


def resolve_datasets(args) -> list[str]:
    if args.datasets:
        return args.datasets

    run_results_path = infer_run_results_path(args)
    if run_results_path is None:
        raise ValueError(
            "--datasets was not provided and no run results.json was found next to --encoder. "
            "Pass --datasets or --run-results-json."
        )
    if not run_results_path.exists():
        raise FileNotFoundError(run_results_path)

    datasets = extract_datasets_from_run_results(load_json(run_results_path))
    if not datasets:
        raise ValueError(f"Could not find datasets in {run_results_path}. Expected config.datasets or datasets.")

    print(f"Inferred datasets from {run_results_path}: {' '.join(datasets)}")
    return datasets


def evaluate_single_model_results(args, device):
    if args.single_model_results_json is None:
        return None

    cache_path = Path(args.single_model_results_json)
    cached = None if args.refresh_single_model_results or not cache_path.exists() else load_json(cache_path)
    results = {} if cached is None else dict(cached.get("results", {}))
    checkpoint_root = Path(args.checkpoint_root)

    for dataset in args.datasets:
        if dataset in results and "acc" in results[dataset]:
            print(f"{dataset}: loaded single-model baseline from {cache_path}")
            continue

        encoder_path = resolve_source_encoder_path(checkpoint_root, dataset)
        head_path = resolve_head_path(dataset, checkpoint_root)
        print(f"{dataset}: evaluating single-model baseline encoder={encoder_path} head={head_path}")
        evaluated = evaluate_encoder(
            encoder_path=encoder_path,
            datasets=[dataset],
            checkpoint_root=args.checkpoint_root,
            data_root=args.data_root,
            arch=args.arch,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            amp=args.amp,
            download=not args.no_download,
        )
        results[dataset] = {
            **evaluated[dataset],
            "encoder": str(encoder_path),
            "head": str(head_path),
        }

    selected_results = {dataset: results[dataset] for dataset in args.datasets}
    average = sum(row["acc"] for row in selected_results.values()) / len(selected_results)
    payload = {
        "type": "single_model_baselines",
        "datasets": args.datasets,
        "checkpoint_root": args.checkpoint_root,
        "data_root": args.data_root,
        "arch": args.arch,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "results": {**results, "average": {"acc": average}},
    }
    write_json(cache_path, payload)
    print(f"Saved single-model baseline results to {cache_path}")
    return payload["results"]


def build_comparison(single_results, merged_results, datasets):
    rows = []
    for dataset in datasets:
        single_acc = float(single_results[dataset]["acc"])
        merged_acc = float(merged_results[dataset]["acc"])
        rows.append(
            {
                "dataset": dataset,
                "single_acc": single_acc,
                "merged_acc": merged_acc,
                "diff": merged_acc - single_acc,
            }
        )

    single_average = sum(row["single_acc"] for row in rows) / len(rows)
    merged_average = sum(row["merged_acc"] for row in rows) / len(rows)
    return {
        "rows": rows,
        "average": {
            "single_acc": single_average,
            "merged_acc": merged_average,
            "diff": merged_average - single_average,
        },
    }


def format_comparison_table(comparison):
    lines = [
        "| Dataset | Single ACC | Merged ACC | Difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"| {row['dataset']} | {row['single_acc'] * 100:.2f}% | "
            f"{row['merged_acc'] * 100:.2f}% | {row['diff'] * 100:+.2f}% |"
        )

    average = comparison["average"]
    lines.append(
        f"| Average | {average['single_acc'] * 100:.2f}% | "
        f"{average['merged_acc'] * 100:.2f}% | {average['diff'] * 100:+.2f}% |"
    )
    return lines


def main():
    args = parse_args()
    args.datasets = resolve_datasets(args)
    device = build_device(args.gpu)
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print("Using device: cpu")

    single_results = evaluate_single_model_results(args, device)

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
        write_json(output_path, results)
        print(f"Saved results to {output_path}")

    if args.results_txt is not None:
        output_path = Path(args.results_txt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
        print(f"Saved results to {output_path}")

    if single_results is not None:
        comparison = build_comparison(single_results, results, args.datasets)
        comparison_lines = format_comparison_table(comparison)
        print("\nMerged vs single-model accuracy:")
        for line in comparison_lines:
            print(line)

        if args.comparison_json is not None:
            output_path = Path(args.comparison_json)
            write_json(
                output_path,
                {
                    "single_model_results_json": args.single_model_results_json,
                    "run_results_json": str(infer_run_results_path(args)) if infer_run_results_path(args) else None,
                    "merged_encoder": args.encoder,
                    "datasets": args.datasets,
                    "comparison": comparison,
                },
            )
            print(f"Saved comparison JSON to {output_path}")

        if args.comparison_txt is not None:
            output_path = Path(args.comparison_txt)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("\n".join(comparison_lines) + "\n", encoding="utf-8")
            print(f"Saved comparison table to {output_path}")


if __name__ == "__main__":
    main()
