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

    model = parser.add_argument_group("model")
    model.add_argument("--arch", default=DEFAULT_ARCH)
    model.add_argument("--policy-hidden-dim", type=int, default=128)
    model.add_argument("--critic-hidden-dim", type=int, default=256)
    model.add_argument("--merge-granularity", choices=["layer", "global"], default="layer")
    model.add_argument("--coefficient-mode", choices=["softmax"], default="softmax")
    model.add_argument("--coefficient-init", type=float, default=1.0)
    model.add_argument("--export-policy", choices=["best", "final"], default="best")

    train = parser.add_argument_group("training")
    train.add_argument("--episodes", type=int, default=300)
    train.add_argument("--gamma", type=float, default=1.0)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--critic-lr", type=float, default=3e-4)
    train.add_argument("--alpha-lr", type=float, default=3e-4)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--replay-size", type=int, default=50000)
    train.add_argument("--random-steps", type=int, default=200)
    train.add_argument("--updates-per-step", type=int, default=1)
    train.add_argument("--tau", type=float, default=0.005)
    train.add_argument("--alpha", type=float, default=0.2)
    train.add_argument("--auto-alpha", action="store_true")
    train.add_argument("--target-entropy", type=float, default=None)
    train.add_argument("--min-concentration", type=float, default=0.05)
    train.add_argument("--terminal-bonus", type=float, default=1.0)
    train.add_argument("--reward-scale", type=float, default=1.0)
    train.add_argument("--step-reward-coef", type=float, default=0.25)
    train.add_argument("--accuracy-imbalance-coef", type=float, default=0.5)
    train.add_argument("--retention-worst-coef", type=float, default=0.5)
    train.add_argument("--retention-drop-coef", type=float, default=1.0)
    train.add_argument("--reward-eval-interval", type=int, default=4)
    train.add_argument("--episode-reward-only", action="store_true")
    train.add_argument("--log-every", type=int, default=10)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--output-dir", default="rl_mmcr_SAC_Actor_Critic_runs/default")
    runtime.add_argument("--gpu", type=int, default=0)
    runtime.add_argument("--num-workers", type=int, default=4)
    runtime.add_argument("--seed", type=int, default=2026)
    runtime.add_argument("--amp", action="store_true")
    runtime.add_argument("--cache-task-vectors-device", action="store_true")
    runtime.add_argument("--skip-final-eval", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "--episodes": args.episodes,
        "--log-every": args.log_every,
        "--reward-eval-interval": args.reward_eval_interval,
        "--reward-batches-per-dataset": args.reward_batches_per_dataset,
        "--reward-batch-size": args.reward_batch_size,
        "--batch-size": args.batch_size,
        "--replay-size": args.replay_size,
        "--updates-per-step": args.updates_per_step,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.random_steps < 0:
        raise ValueError("--random-steps must be non-negative.")
    if args.tau <= 0 or args.tau > 1:
        raise ValueError("--tau must be in (0, 1].")
    if args.min_concentration <= 0:
        raise ValueError("--min-concentration must be positive.")
