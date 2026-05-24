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
    data.add_argument("--reward-batch-size", type=int, default=32)
    data.add_argument("--reward-batches-per-dataset", type=int, default=4)
    data.add_argument("--top-k-percent", type=float, default=20.0)
    data.add_argument("--no-download", action="store_true")
    data.add_argument("--source-baseline-json", default=None, help="Optional cached source-model baseline accuracies used as retention denominators.")

    model = parser.add_argument_group("model")
    model.add_argument("--arch", default=DEFAULT_ARCH)
    model.add_argument("--policy-hidden-dim", type=int, default=128)
    model.add_argument("--merge-granularity", choices=["global"], default="global")
    model.add_argument("--coefficient-mode", choices=["positive"], default="positive")
    model.add_argument("--coefficient-init", type=float, default=0.3)
    model.add_argument("--export-policy", choices=["best", "final"], default="best")

    train = parser.add_argument_group("training")
    train.add_argument("--iterations", type=int, default=300)
    train.add_argument("--group-size", type=int, default=8)
    train.add_argument("--grpo-epochs", type=int, default=4)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--clip-eps", type=float, default=0.2)
    train.add_argument("--entropy-coef", type=float, default=0.01)
    train.add_argument("--target-kl", type=float, default=0.03)
    train.add_argument("--advantage-mode", choices=["rloo", "zscore", "rank"], default="rloo")
    train.add_argument("--min-concentration", type=float, default=0.05, help="Kept for compatibility with older Dirichlet GRPO runs; unused by the positive softplus-normal policy.")
    train.add_argument("--log-std-min", type=float, default=-5.0)
    train.add_argument("--log-std-max", type=float, default=1.0)
    train.add_argument("--terminal-bonus", type=float, default=1.0)
    train.add_argument("--reward-scale", type=float, default=1.0)
    train.add_argument("--activation-reward-coef", type=float, default=0.0, help="Dense layer-wise reward coefficient for cosine similarity between merged and source-model activations.")
    train.add_argument("--step-reward-coef", type=float, default=0.25)
    train.add_argument("--accuracy-imbalance-coef", type=float, default=0.5)
    train.add_argument("--retention-worst-coef", type=float, default=0.5)
    train.add_argument("--retention-drop-coef", type=float, default=1.0)
    train.add_argument("--reward-eval-interval", type=int, default=1)
    train.add_argument("--episode-reward-only", action="store_true")
    train.add_argument("--log-every", type=int, default=10)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--output-dir", default="rl_mmcr_GRPO_RLOO_runs/default")
    runtime.add_argument("--gpu", type=int, default=0)
    runtime.add_argument("--num-workers", type=int, default=4)
    runtime.add_argument("--seed", type=int, default=2026)
    runtime.add_argument("--amp", action="store_true")
    runtime.add_argument("--cache-task-vectors-device", action="store_true")
    runtime.add_argument("--skip-final-eval", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "--iterations": args.iterations,
        "--group-size": args.group_size,
        "--grpo-epochs": args.grpo_epochs,
        "--log-every": args.log_every,
        "--reward-eval-interval": args.reward_eval_interval,
        "--reward-batches-per-dataset": args.reward_batches_per_dataset,
        "--reward-batch-size": args.reward_batch_size,
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
    if args.log_std_min >= args.log_std_max:
        raise ValueError("--log-std-min must be smaller than --log-std-max.")
    if args.activation_reward_coef < 0:
        raise ValueError("--activation-reward-coef must be non-negative.")
