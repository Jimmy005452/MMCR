from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from mmcr.evaluation import evaluate_dataset, resolve_head_path
from mmcr.models import DEFAULT_ARCH, build_image_encoder, build_model_transforms
from mmcr.utils import build_device, seed_everything, write_json

from .grpo import PositiveSoftplusPolicy
from .train import deterministic_policy_result

# Reuse the PPO/GRPO shared environment builder so transfer evaluation sees the
# same state construction, task-vector preprocessing, and reward batches as training.
ppo_train = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a saved GRPO policy network to a different target dataset combination."
    )
    parser.add_argument("--policy-checkpoint", required=True, help="Path to best_policy.pt or final_policy.pt from a GRPO run.")
    parser.add_argument("--target-datasets", nargs="+", required=True, help="Datasets to merge with the transferred policy.")
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--zeroshot", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--synthesis-root", default=None)
    parser.add_argument("--arch", default=None)
    parser.add_argument("--top-k-percent", type=float, default=None)
    parser.add_argument("--task-vector-mode", choices=["ties", "raw"], default=None)
    parser.add_argument("--merge-granularity", choices=["global", "layer"], default=None)
    parser.add_argument("--coefficient-granularity", choices=["layer", "tensor"], default=None)
    parser.add_argument("--state-mode", choices=["minimal", "full_coefficients"], default=None)
    parser.add_argument("--coefficient-mode", choices=["positive"], default=None)
    parser.add_argument("--coefficient-init", type=float, default=None)
    parser.add_argument("--reward-mode", choices=["accuracy_retention", "entropy", "synthetic_entropy", "synthetic_proxy"], default=None)
    parser.add_argument("--reward-split", choices=["val", "test"], default=None)
    parser.add_argument("--reward-batch-size", type=int, default=None)
    parser.add_argument("--reward-batches-per-dataset", type=int, default=None)
    parser.add_argument("--reward-sampling-mode", choices=["sequential", "stratified_pool"], default=None)
    parser.add_argument("--reward-pool-size", type=int, default=None)
    parser.add_argument("--reward-pool-position", type=int, default=0, help="Reward pool window used while replaying the transferred policy.")
    parser.add_argument("--source-baseline-json", default=None, help="Optional target-dataset source baseline cache. Not inherited from source policy config.")
    parser.add_argument("--include-rejected-synthetic", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Batch size for final full-dataset accuracy evaluation.")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache-task-vectors-device", action="store_true")
    parser.add_argument("--batched-reward-eval", action="store_true")
    parser.add_argument("--batched-reward-max-samples", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-txt", default=None)
    return parser.parse_args()


def load_policy_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing policy checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "policy_state_dict" not in payload:
        raise ValueError(f"{path} is not a GRPO policy checkpoint with policy_state_dict.")
    return payload


def pick(args: argparse.Namespace, config: dict, name: str, default=None):
    value = getattr(args, name)
    if value is not None:
        return value
    return config.get(name, default)


def build_target_args(args: argparse.Namespace, source_config: dict) -> SimpleNamespace:
    # Build the minimal argparse-like object expected by ppo_train.build_environment.
    # Most architectural/training choices are inherited from the source policy so the
    # transferred network sees the same state/action format. Dataset paths and
    # target datasets come from this transfer command.
    checkpoint_root = pick(args, source_config, "checkpoint_root", "checkpoints")
    return SimpleNamespace(
        datasets=[str(dataset) for dataset in args.target_datasets],
        checkpoint_root=checkpoint_root,
        zeroshot=args.zeroshot if args.zeroshot is not None else source_config.get("zeroshot"),
        data_root=pick(args, source_config, "data_root", "data"),
        synthesis_root=pick(args, source_config, "synthesis_root", "synthesis_data/generated"),
        reward_split=pick(args, source_config, "reward_split", "val"),
        reward_batch_size=int(pick(args, source_config, "reward_batch_size", 64)),
        reward_batches_per_dataset=int(pick(args, source_config, "reward_batches_per_dataset", 1)),
        reward_sampling_mode=pick(args, source_config, "reward_sampling_mode", "sequential"),
        reward_pool_size=int(pick(args, source_config, "reward_pool_size", 0)),
        top_k_percent=float(pick(args, source_config, "top_k_percent", 20.0)),
        task_vector_mode=pick(args, source_config, "task_vector_mode", "ties"),
        no_download=bool(args.no_download or source_config.get("no_download", False)),
        source_baseline_json=args.source_baseline_json,
        include_rejected_synthetic=bool(args.include_rejected_synthetic or source_config.get("include_rejected_synthetic", False)),
        arch=pick(args, source_config, "arch", DEFAULT_ARCH),
        policy_hidden_dim=int(source_config.get("policy_hidden_dim", 128)),
        merge_granularity=pick(args, source_config, "merge_granularity", "layer"),
        coefficient_granularity=pick(args, source_config, "coefficient_granularity", "layer"),
        state_mode=pick(args, source_config, "state_mode", "minimal"),
        coefficient_mode=pick(args, source_config, "coefficient_mode", "positive"),
        coefficient_init=float(pick(args, source_config, "coefficient_init", 0.3)),
        log_std_min=float(source_config.get("log_std_min", -5.0)),
        log_std_max=float(source_config.get("log_std_max", 1.0)),
        terminal_bonus=float(source_config.get("terminal_bonus", 1.0)),
        reward_scale=float(source_config.get("reward_scale", 1.0)),
        reward_mode=pick(args, source_config, "reward_mode", "accuracy_retention"),
        kl_weight=float(source_config.get("kl_weight", 1.0)),
        agreement_weight=float(source_config.get("agreement_weight", 0.5)),
        entropy_weight=float(source_config.get("entropy_weight", 0.1)),
        activation_reward_coef=0.0,
        step_reward_coef=float(source_config.get("step_reward_coef", 0.25)),
        accuracy_imbalance_coef=float(source_config.get("accuracy_imbalance_coef", 0.5)),
        retention_worst_coef=float(source_config.get("retention_worst_coef", 0.5)),
        retention_drop_coef=float(source_config.get("retention_drop_coef", 1.0)),
        reward_eval_interval=int(source_config.get("reward_eval_interval", 1)),
        episode_reward_only=bool(source_config.get("episode_reward_only", False)),
        output_dir=str(Path(args.output_json).parent),
        gpu=int(pick(args, source_config, "gpu", 0)),
        num_workers=int(pick(args, source_config, "num_workers", 4)),
        seed=int(pick(args, source_config, "seed", 2026)),
        amp=bool(args.amp or source_config.get("amp", False)),
        cache_task_vectors_device=bool(args.cache_task_vectors_device or source_config.get("cache_task_vectors_device", False)),
        batched_reward_eval=bool(args.batched_reward_eval or source_config.get("batched_reward_eval", False)),
        batched_reward_max_samples=int(pick(args, source_config, "batched_reward_max_samples", 128)),
        skip_final_eval=True,
    )


def validate_transfer_shapes(policy: PositiveSoftplusPolicy, env) -> None:
    if policy.action_dim != env.num_models:
        raise ValueError(
            f"Policy action_dim={policy.action_dim} but target env num_models={env.num_models}. "
            "Direct policy transfer requires the same number of source models/tasks."
        )


def build_policy_from_checkpoint(policy_payload: dict, env, target_args: SimpleNamespace) -> PositiveSoftplusPolicy:
    policy = PositiveSoftplusPolicy(
        env.state_dim,
        env.num_models,
        hidden_dim=target_args.policy_hidden_dim,
        log_std_min=target_args.log_std_min,
        log_std_max=target_args.log_std_max,
        initial_coefficient=target_args.coefficient_init,
    ).to(env.device)
    try:
        policy.load_state_dict(policy_payload["policy_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Policy checkpoint is not shape-compatible with the target environment. "
            "Check that num_models, merge_granularity, coefficient_granularity, state_mode, "
            "and policy_hidden_dim match between source and target."
        ) from exc
    policy.eval()
    validate_transfer_shapes(policy, env)
    return policy


def evaluate_transferred_encoder(env, result: dict, args: argparse.Namespace, target_args: SimpleNamespace) -> dict:
    coefficients = torch.tensor(result["expanded_coefficients"], dtype=torch.float32)
    encoder_state = env.export_merged_state(coefficients)
    encoder = build_image_encoder(arch=target_args.arch, pretrained=False).to(env.device)
    encoder.load_state_dict(encoder_state)

    eval_batch_size = int(args.eval_batch_size or target_args.reward_batch_size)
    _, eval_transform, _ = build_model_transforms(target_args.arch, pretrained=False)
    results = {}
    for dataset in target_args.datasets:
        head_path = resolve_head_path(dataset, target_args.checkpoint_root)
        acc = evaluate_dataset(
            encoder,
            dataset,
            head_path,
            target_args.data_root,
            eval_batch_size,
            target_args.num_workers,
            eval_transform,
            env.device,
            amp=target_args.amp,
            download=not target_args.no_download,
        )
        print(f"{dataset}: ACC={acc * 100:.2f}%")
        results[dataset] = {"acc": acc}
    if len(target_args.datasets) > 1:
        results["average"] = {"acc": sum(results[dataset]["acc"] for dataset in target_args.datasets) / len(target_args.datasets)}
        print(f"Average ACC={results['average']['acc'] * 100:.2f}%")
    return results


def format_results(results: dict) -> str:
    lines = []
    for dataset, row in results.items():
        if dataset == "average":
            continue
        if isinstance(row, dict) and "acc" in row:
            lines.append(f"{dataset}: ACC={float(row['acc']) * 100:.2f}%")
    if "average" in results and isinstance(results["average"], dict) and "acc" in results["average"]:
        lines.append(f"Average ACC={float(results['average']['acc']) * 100:.2f}%")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    policy_path = Path(args.policy_checkpoint)
    policy_payload = load_policy_checkpoint(policy_path)
    source_config = policy_payload.get("config", {})
    if not isinstance(source_config, dict):
        source_config = {}

    target_args = build_target_args(args, source_config)
    seed_everything(target_args.seed)
    device = build_device(target_args.gpu)
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"Loading source policy from {policy_path}")
    print("Target datasets: " + ", ".join(target_args.datasets))

    env = ppo_train.build_environment(target_args, device)
    if hasattr(env, "set_reward_pool_position"):
        env.set_reward_pool_position(args.reward_pool_position)
    policy = build_policy_from_checkpoint(policy_payload, env, target_args)

    # Deterministic replay answers the transfer question: given target states,
    # what coefficients does the source-trained policy network choose?
    transfer_result = deterministic_policy_result(env, policy)
    eval_results = evaluate_transferred_encoder(env, transfer_result, args, target_args)

    output = {
        "policy_checkpoint": str(policy_path),
        "policy_type": policy_payload.get("type"),
        "source_config": source_config,
        "target_config": vars(target_args),
        "transfer_policy_result": transfer_result,
        "eval_results": eval_results,
    }
    write_json(Path(args.output_json), output)
    print(format_results(eval_results))
    print(f"Saved transfer evaluation to {args.output_json}")

    if args.output_txt:
        txt_path = Path(args.output_txt)
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        txt_path.write_text(format_results(eval_results) + "\n", encoding="utf-8")
        print(f"Saved text summary to {txt_path}")


if __name__ == "__main__":
    main()
