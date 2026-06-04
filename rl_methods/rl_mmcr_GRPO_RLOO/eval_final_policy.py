from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from mmcr.evaluation import evaluate_dataset, resolve_head_path
from mmcr.models import DEFAULT_ARCH, build_image_encoder, build_model_transforms
from mmcr.utils import build_device

ppo_train = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.train")
ppo_merge = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.merge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GRPO results.json policy without writing a merged encoder checkpoint."
    )
    parser.add_argument("--results-json", required=True, help="Path to a GRPO results.json file.")
    parser.add_argument(
        "--policy",
        choices=["final_policy", "best_sample", "exported_policy"],
        default="final_policy",
        help="Which policy/coefficient block to evaluate from results.json.",
    )
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--zeroshot", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--top-k-percent", type=float, default=None)
    parser.add_argument("--task-vector-mode", choices=["ties", "raw"], default=None)
    parser.add_argument("--coefficient-granularity", choices=["layer", "tensor"], default=None)
    parser.add_argument("--coefficient-mode", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-txt", default=None)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_arg(args: argparse.Namespace, config: dict, name: str, default=None):
    value = getattr(args, name.replace("-", "_"), None)
    if value is not None:
        return value
    return config.get(name.replace("-", "_"), default)


def extract_policy(payload: dict, policy_name: str) -> dict:
    if policy_name == "exported_policy":
        exported = payload.get("exported_policy")
        if not isinstance(exported, dict):
            raise ValueError("results.json does not contain exported_policy.")
        exported_type = exported.get("type")
        if exported_type == "final_deterministic":
            policy_name = "final_policy"
        elif exported_type == "best_group_sample":
            policy_name = "best_sample"
        else:
            raise ValueError(f"Cannot resolve exported_policy type: {exported_type!r}")

    policy = payload.get(policy_name)
    if not isinstance(policy, dict):
        raise ValueError(f"results.json does not contain {policy_name}.")
    return policy


def extract_coefficients(policy: dict) -> torch.Tensor:
    coefficients = policy.get("expanded_coefficients")
    if coefficients is None:
        coefficients = policy.get("coefficients")
    if coefficients is None:
        raise ValueError("Selected policy does not contain expanded_coefficients or coefficients.")
    return torch.tensor(coefficients, dtype=torch.float32)


def build_merged_encoder(
    payload: dict,
    datasets: list[str],
    checkpoint_root: Path,
    zeroshot_path: Path,
    arch: str,
    top_k_percent: float,
    task_vector_mode: str,
    coefficient_granularity: str,
    coefficient_mode: str,
    coefficients: torch.Tensor,
    device: torch.device,
):
    encoder_paths = [ppo_train.resolve_source_encoder_path(checkpoint_root, dataset) for dataset in datasets]
    ppo_train.require_existing([zeroshot_path, *encoder_paths])
    layered_task_vectors = ppo_merge.load_layered_task_vectors(
        zeroshot_path=zeroshot_path,
        finetuned_paths=encoder_paths,
        task_names=datasets,
        mode=task_vector_mode,
        top_k_percent=top_k_percent,
        granularity=coefficient_granularity,
    )
    if coefficients.ndim == 1:
        coefficients = coefficients.unsqueeze(0)
    if tuple(coefficients.shape) == (1, layered_task_vectors.num_models):
        coefficients = coefficients.expand(layered_task_vectors.num_layers, layered_task_vectors.num_models).clone()
    merged_state = ppo_merge.merge_state_with_layer_coefficients(
        layered_task_vectors,
        coefficients,
        coefficient_mode=coefficient_mode,
    )
    encoder = build_image_encoder(arch=arch, pretrained=False)
    encoder.load_state_dict(merged_state)
    return encoder.to(device), coefficients


def evaluate_in_memory_encoder(
    encoder,
    datasets: list[str],
    checkpoint_root: Path,
    data_root: str,
    arch: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
    download: bool,
) -> dict:
    _, eval_transform, _ = build_model_transforms(arch, pretrained=False)
    results = {}
    for dataset in datasets:
        head_path = resolve_head_path(dataset, checkpoint_root)
        acc = evaluate_dataset(
            encoder,
            dataset,
            head_path,
            data_root,
            batch_size,
            num_workers,
            eval_transform,
            device,
            amp=amp,
            download=download,
        )
        print(f"{dataset}: ACC={acc * 100:.2f}%")
        results[dataset] = {"acc": acc}
    if len(datasets) > 1:
        results["average"] = {"acc": sum(results[dataset]["acc"] for dataset in datasets) / len(datasets)}
        print(f"Average ACC={results['average']['acc'] * 100:.2f}%")
    return results


def format_results(results: dict) -> str:
    lines = []
    for dataset, row in results.items():
        if dataset == "average":
            continue
        lines.append(f"{dataset}: ACC={float(row['acc']) * 100:.2f}%")
    if "average" in results:
        lines.append(f"Average ACC={float(results['average']['acc']) * 100:.2f}%")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_path = Path(args.results_json)
    payload = load_json(results_path)
    config = payload.get("config", {})
    if not isinstance(config, dict):
        config = {}

    datasets = args.datasets or config.get("datasets")
    if not datasets:
        raise ValueError("Datasets were not provided and config.datasets is missing from results.json.")
    datasets = [str(dataset) for dataset in datasets]

    checkpoint_root = Path(args.checkpoint_root or config.get("checkpoint_root", "checkpoints"))
    zeroshot_path = Path(args.zeroshot or config.get("zeroshot") or checkpoint_root / "zeroshot.pt")
    data_root = args.data_root or config.get("data_root", "data")
    arch = args.arch or config.get("arch", DEFAULT_ARCH)
    top_k_percent = float(args.top_k_percent if args.top_k_percent is not None else config.get("top_k_percent", 20.0))
    task_vector_mode = args.task_vector_mode or config.get("task_vector_mode", "ties")
    coefficient_granularity = args.coefficient_granularity or config.get("coefficient_granularity", "layer")
    coefficient_mode = args.coefficient_mode or config.get("coefficient_mode", "positive")
    batch_size = int(args.batch_size if args.batch_size is not None else config.get("reward_batch_size", 64))
    num_workers = int(args.num_workers if args.num_workers is not None else config.get("num_workers", 4))
    gpu = int(args.gpu if args.gpu is not None else config.get("gpu", 0))
    amp = bool(args.amp or config.get("amp", False))

    policy = extract_policy(payload, args.policy)
    coefficients = extract_coefficients(policy)
    device = build_device(gpu)
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"Evaluating {args.policy} from {results_path}")

    encoder, expanded_coefficients = build_merged_encoder(
        payload=payload,
        datasets=datasets,
        checkpoint_root=checkpoint_root,
        zeroshot_path=zeroshot_path,
        arch=arch,
        top_k_percent=top_k_percent,
        task_vector_mode=task_vector_mode,
        coefficient_granularity=coefficient_granularity,
        coefficient_mode=coefficient_mode,
        coefficients=coefficients,
        device=device,
    )
    results = evaluate_in_memory_encoder(
        encoder,
        datasets=datasets,
        checkpoint_root=checkpoint_root,
        data_root=data_root,
        arch=arch,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        amp=amp,
        download=not args.no_download,
    )

    output_payload = {
        "type": "grpo_policy_in_memory_eval",
        "results_json": str(results_path),
        "policy": args.policy,
        "datasets": datasets,
        "checkpoint_root": str(checkpoint_root),
        "zeroshot": str(zeroshot_path),
        "data_root": str(data_root),
        "arch": arch,
        "top_k_percent": top_k_percent,
        "task_vector_mode": task_vector_mode,
        "coefficient_granularity": coefficient_granularity,
        "coefficient_mode": coefficient_mode,
        "coefficient_shape": list(expanded_coefficients.shape),
        "results": results,
    }
    if args.output_json:
        write_json(Path(args.output_json), output_payload)
        print(f"Saved JSON results to {args.output_json}")
    if args.output_txt:
        txt_path = Path(args.output_txt)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(format_results(results) + "\n", encoding="utf-8")
        print(f"Saved text results to {args.output_txt}")


if __name__ == "__main__":
    main()
