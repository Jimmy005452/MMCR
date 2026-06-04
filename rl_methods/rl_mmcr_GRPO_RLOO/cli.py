from __future__ import annotations

import argparse

DEFAULT_ARCH = "vit_large_patch14_clip_224.openai"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    data = parser.add_argument_group("data")
    data.add_argument("--datasets", nargs="+", required=True)
    data.add_argument("--checkpoint-root", default="checkpoints")
    data.add_argument("--zeroshot", default=None)
    data.add_argument("--data-root", default="data")
    data.add_argument("--synthesis-root", default="synthesis_data/generated")
    data.add_argument("--reward-split", choices=["val", "test"], default="val")
    data.add_argument("--reward-batch-size", type=int, default=32)
    data.add_argument("--reward-batches-per-dataset", type=int, default=4)
    data.add_argument("--reward-sampling-mode", choices=["sequential", "stratified_pool"], default="sequential", help="Reward subset construction. sequential keeps the original first-N batches; stratified_pool builds a fixed class-interleaved pool and rotates chunks during training.")
    data.add_argument("--reward-pool-size", type=int, default=0, help="Number of samples per dataset in stratified_pool mode. 0 uses reward_batch_size * reward_batches_per_dataset.")
    data.add_argument("--selection-reward-pool-position", type=int, default=0, help="Fixed stratified reward-pool chunk used for best-checkpoint selection and deterministic logging. Set to -1 to use the current rotating position.")
    data.add_argument("--selection-candidates", type=int, default=1, help="Number of top rotating-pool group samples to re-score on the fixed selection subset before best-checkpoint selection.")
    data.add_argument("--top-k-percent", type=float, default=20.0)
    data.add_argument("--task-vector-mode", choices=["ties", "raw"], default="ties", help="Task-vector preprocessing. ties uses TIES sign/top-k selection; raw uses full finetuned - zeroshot deltas.")
    data.add_argument("--no-download", action="store_true")
    data.add_argument("--source-baseline-json", default=None, help="Optional cached source-model baseline accuracies used as retention denominators.")
    data.add_argument("--include-rejected-synthetic", action="store_true", help="Use all synthetic samples instead of accepted_mask-filtered samples.")

    model = parser.add_argument_group("model")
    model.add_argument("--arch", default=DEFAULT_ARCH)
    model.add_argument("--policy-hidden-dim", type=int, default=128)
    model.add_argument("--merge-granularity", choices=["global", "layer"], default="global")
    model.add_argument(
        "--coefficient-granularity",
        choices=["layer", "tensor"],
        default="layer",
        help="Grouping used when merge-granularity=layer. layer shares one coefficient vector per transformer layer; tensor uses one coefficient vector per floating-point tensor.",
    )
    model.add_argument(
        "--state-mode",
        choices=["minimal", "full_coefficients"],
        default="minimal",
        help="State representation used by policy/value networks.",
    )
    model.add_argument("--coefficient-mode", choices=["positive"], default="positive")
    model.add_argument("--coefficient-init", type=float, default=0.3)
    model.add_argument("--export-policy", choices=["best", "final"], default="best")

    train = parser.add_argument_group("training")
    train.add_argument("--iterations", type=int, default=300)
    train.add_argument("--group-size", type=int, default=8)
    train.add_argument("--grpo-epochs", type=int, default=4)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--clip-eps", type=float, default=0.2)
    train.add_argument("--clip-eps-low", type=float, default=None, help="Lower PPO/DAPO-lite clipping epsilon. Defaults to --clip-eps.")
    train.add_argument("--clip-eps-high", type=float, default=None, help="Upper PPO/DAPO-lite clipping epsilon. Defaults to --clip-eps. Larger values implement DAPO-style Clip-Higher.")
    train.add_argument("--policy-loss-mode", choices=["step", "trajectory"], default="step", help="step is original per-layer ratio clipping; trajectory is GSPO-like sequence/trajectory ratio clipping.")
    train.add_argument("--entropy-coef", type=float, default=0.01)
    train.add_argument("--target-kl", type=float, default=0.03)
    train.add_argument("--advantage-mode", choices=["rloo", "rloo_no_std", "zscore", "rank"], default="rloo", help="rloo_no_std is a Dr.GRPO-like ablation that removes group reward std normalization.")
    train.add_argument("--min-concentration", type=float, default=0.05, help="Kept for compatibility with older Dirichlet GRPO runs; unused by the positive softplus-normal policy.")
    train.add_argument("--log-std-min", type=float, default=-5.0)
    train.add_argument("--log-std-max", type=float, default=1.0)
    train.add_argument("--terminal-bonus", type=float, default=1.0)
    train.add_argument("--reward-scale", type=float, default=1.0)
    train.add_argument("--reward-mode", choices=["accuracy_retention", "entropy", "synthetic_entropy", "synthetic_proxy"], default="accuracy_retention")
    train.add_argument("--kl-weight", type=float, default=1.0)
    train.add_argument("--agreement-weight", type=float, default=0.5)
    train.add_argument("--entropy-weight", type=float, default=0.1)
    train.add_argument("--activation-reward-coef", type=float, default=0.0, help="Dense layer-wise reward coefficient for cosine similarity between merged and source-model activations.")
    train.add_argument("--step-reward-coef", type=float, default=0.25)
    train.add_argument("--accuracy-imbalance-coef", type=float, default=0.5)
    train.add_argument("--retention-worst-coef", type=float, default=0.5)
    train.add_argument("--retention-drop-coef", type=float, default=1.0)
    train.add_argument("--reward-eval-interval", type=int, default=1)
    train.add_argument("--episode-reward-only", action="store_true")
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--dynamic-sampling-min-std", type=float, default=0.0, help="DAPO-lite dynamic sampling: resample a group before update when reward std is below this threshold. 0 disables it.")
    train.add_argument("--dynamic-sampling-max-resamples", type=int, default=0, help="Maximum DAPO-lite group resamples per iteration when reward std is too small.")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--output-dir", default="rl_mmcr_GRPO_RLOO_runs/default")
    runtime.add_argument("--gpu", type=int, default=0)
    runtime.add_argument("--num-workers", type=int, default=4)
    runtime.add_argument("--seed", type=int, default=2026)
    runtime.add_argument("--amp", action="store_true")
    runtime.add_argument("--cache-task-vectors-device", action="store_true")
    runtime.add_argument("--batched-reward-eval", action="store_true", help="Evaluate reward batches from multiple datasets in concatenated encoder forwards.")
    runtime.add_argument("--batched-reward-max-samples", type=int, default=128, help="Maximum concatenated image count per batched reward encoder forward. Use 0 to concatenate all datasets.")
    runtime.add_argument("--trajectory-devices", nargs="+", default=None, help="CUDA device ids for multi-GPU layer-wise trajectory rollout, for example: --trajectory-devices 0 1 or --trajectory-devices 0,1.")
    runtime.add_argument("--skip-final-eval", action="store_true")
    args = parser.parse_args()
    args.trajectory_devices = parse_trajectory_devices(args.trajectory_devices)
    return args


def parse_trajectory_devices(values: list[str] | None) -> list[int]:
    if not values:
        return []
    devices = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                devices.append(int(item))
    return devices


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "--iterations": args.iterations,
        "--group-size": args.group_size,
        "--grpo-epochs": args.grpo_epochs,
        "--log-every": args.log_every,
        "--reward-eval-interval": args.reward_eval_interval,
        "--reward-batches-per-dataset": args.reward_batches_per_dataset,
        "--reward-batch-size": args.reward_batch_size,
        "--reward-pool-size": max(1, args.reward_pool_size) if args.reward_sampling_mode == "stratified_pool" else 1,
        "--selection-candidates": args.selection_candidates,
        "--dynamic-sampling-max-resamples": args.dynamic_sampling_max_resamples + 1,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.group_size < 2:
        raise ValueError("--group-size must be at least 2 for group-relative advantages.")
    if args.coefficient_init <= 0:
        raise ValueError("--coefficient-init must be positive.")
    if args.min_concentration <= 0:
        raise ValueError("--min-concentration must be positive.")
    if args.clip_eps <= 0:
        raise ValueError("--clip-eps must be positive.")
    if args.clip_eps_low is not None and args.clip_eps_low <= 0:
        raise ValueError("--clip-eps-low must be positive when provided.")
    if args.clip_eps_high is not None and args.clip_eps_high <= 0:
        raise ValueError("--clip-eps-high must be positive when provided.")
    low = args.clip_eps if args.clip_eps_low is None else args.clip_eps_low
    high = args.clip_eps if args.clip_eps_high is None else args.clip_eps_high
    if 1.0 - low <= 0 or 1.0 - low >= 1.0 + high:
        raise ValueError("Invalid clipping range from --clip-eps-low/--clip-eps-high.")
    if args.dynamic_sampling_min_std < 0:
        raise ValueError("--dynamic-sampling-min-std must be non-negative.")
    if args.dynamic_sampling_max_resamples < 0:
        raise ValueError("--dynamic-sampling-max-resamples must be non-negative.")
    if args.log_std_min >= args.log_std_max:
        raise ValueError("--log-std-min must be smaller than --log-std-max.")
    if args.activation_reward_coef < 0:
        raise ValueError("--activation-reward-coef must be non-negative.")
    if args.reward_pool_size < 0:
        raise ValueError("--reward-pool-size must be non-negative.")
    if args.selection_reward_pool_position < -1:
        raise ValueError("--selection-reward-pool-position must be -1 or non-negative.")
    if args.kl_weight < 0:
        raise ValueError("--kl-weight must be non-negative.")
    if args.agreement_weight < 0:
        raise ValueError("--agreement-weight must be non-negative.")
    if args.entropy_weight < 0:
        raise ValueError("--entropy-weight must be non-negative.")
    if args.reward_mode != "accuracy_retention" and args.activation_reward_coef != 0:
        raise ValueError("--activation-reward-coef is only supported with --reward-mode accuracy_retention for now.")
    if len(args.trajectory_devices) != len(set(args.trajectory_devices)):
        raise ValueError("--trajectory-devices must not contain duplicate device ids.")
