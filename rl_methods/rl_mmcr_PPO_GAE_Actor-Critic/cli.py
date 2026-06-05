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
    data.add_argument("--reward-split", choices=["val", "test"], default="val")
    data.add_argument("--reward-batch-size", type=int, default=16)
    data.add_argument("--reward-batches-per-dataset", type=int, default=1)
    data.add_argument("--reward-sampling-mode", choices=["sequential", "stratified_pool"], default="sequential", help="Reward subset construction. sequential keeps the original first-N batches; stratified_pool builds a fixed class-interleaved pool and rotates chunks during training.")
    data.add_argument("--reward-pool-size", type=int, default=0, help="Number of samples per dataset in stratified_pool mode. 0 uses reward_batch_size * reward_batches_per_dataset.")
    data.add_argument("--top-k-percent", type=float, default=20.0)
    data.add_argument("--task-vector-mode", choices=["ties", "raw"], default="ties", help="Task-vector preprocessing. ties uses TIES sign/top-k selection; raw uses full finetuned - zeroshot deltas.")
    data.add_argument("--no-download", action="store_true")

    model = parser.add_argument_group("model")
    model.add_argument("--arch", default=DEFAULT_ARCH)
    model.add_argument("--policy-hidden-dim", type=int, default=64)
    model.add_argument(
        "--merge-granularity",
        choices=["layer", "global"],
        default="layer",
        help="Use layer for layer-wise coefficients or global for one coefficient vector shared by all layers.",
    )
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
    model.add_argument("--coefficient-init", type=float, default=0.3)
    model.add_argument("--log-std-min", type=float, default=-5.0)
    model.add_argument("--log-std-max", type=float, default=1.0)
    model.add_argument("--export-policy", choices=["best", "final"], default="best")

    train = parser.add_argument_group("training")
    train.add_argument("--episodes", type=int, default=300)
    train.add_argument("--gamma", type=float, default=1.0)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--entropy-coef", type=float, default=0.01)
    train.add_argument("--value-coef", type=float, default=0.5)
    train.add_argument("--terminal-bonus", type=float, default=1.0)
    train.add_argument("--reward-scale", type=float, default=1.0)
    train.add_argument("--reward-mode", choices=["entropy"], default="entropy")
    train.add_argument("--step-reward-coef", type=float, default=0.25)
    train.add_argument("--score-imbalance-coef", type=float, default=0.5)
    train.add_argument("--reward-eval-interval", type=int, default=1)
    train.add_argument("--episode-reward-only", action="store_true")
    train.add_argument("--log-every", type=int, default=10)
    train.add_argument("--rollouts-per-update", type=int, default=4)
    train.add_argument("--ppo-epochs", type=int, default=4)
    train.add_argument("--clip-eps", type=float, default=0.2)
    train.add_argument("--gae-lambda", type=float, default=0.95)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--target-kl", type=float, default=0.03)
    train.add_argument("--ppo-minibatch-size", type=int, default=0)
    train.add_argument("--no-advantage-norm", action="store_true")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--output-dir", default="rl_mmcr_PPO_GAE_Actor-Critic_runs/default")
    runtime.add_argument("--gpu", type=int, default=0)
    runtime.add_argument("--num-workers", type=int, default=4)
    runtime.add_argument("--seed", type=int, default=2026)
    runtime.add_argument("--amp", action="store_true")
    runtime.add_argument("--cache-task-vectors-device", action="store_true")
    runtime.add_argument("--batched-reward-eval", action="store_true", help="Evaluate reward batches from multiple datasets in concatenated encoder forwards.")
    runtime.add_argument("--batched-reward-max-samples", type=int, default=128, help="Maximum concatenated image count per batched reward encoder forward. Use 0 to concatenate all datasets.")
    runtime.add_argument("--skip-final-eval", action="store_true")
    args = parser.parse_args()
    args.coefficient_mode = "positive"
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "--episodes": args.episodes,
        "--log-every": args.log_every,
        "--rollouts-per-update": args.rollouts_per_update,
        "--ppo-epochs": args.ppo_epochs,
        "--reward-eval-interval": args.reward_eval_interval,
        "--reward-batches-per-dataset": args.reward_batches_per_dataset,
        "--reward-batch-size": args.reward_batch_size,
        "--reward-pool-size": max(1, args.reward_pool_size) if args.reward_sampling_mode == "stratified_pool" else 1,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.coefficient_init <= 0:
        raise ValueError("--coefficient-init must be positive.")
    if args.log_std_min >= args.log_std_max:
        raise ValueError("--log-std-min must be smaller than --log-std-max.")
    if args.reward_pool_size < 0:
        raise ValueError("--reward-pool-size must be non-negative.")
    if args.clip_eps <= 0:
        raise ValueError("--clip-eps must be positive.")
    if args.gae_lambda < 0 or args.gae_lambda > 1:
        raise ValueError("--gae-lambda must be in [0, 1].")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be positive.")
    if args.ppo_minibatch_size < 0:
        raise ValueError("--ppo-minibatch-size must be non-negative.")
