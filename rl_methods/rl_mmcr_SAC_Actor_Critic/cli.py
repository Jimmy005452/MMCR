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
    model.add_argument("--critic-hidden-dim", type=int, default=256)
    model.add_argument("--merge-granularity", choices=["layer", "global"], default="layer")
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
    train.add_argument("--episodes", type=int, default=300)
    train.add_argument("--gamma", type=float, default=1.0)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--critic-lr", type=float, default=3e-4)
    train.add_argument("--alpha-lr", type=float, default=3e-4)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--replay-size", type=int, default=50000)
    train.add_argument(
        "--random-steps",
        type=int,
        default=200,
        help="Number of environment steps sampled from independent Uniform(0, 1) positive coefficients before SAC updates use the actor.",
    )
    train.add_argument("--updates-per-step", type=int, default=1)
    train.add_argument("--actor-update-delay", type=int, default=1, help="Update the actor once every N critic updates. Values >1 make SAC more TD3-like and reduce critic exploitation.")
    train.add_argument("--freeze-actor-during-random-steps", action="store_true", help="Collect random replay and train only critics until --random-steps is exhausted.")
    train.add_argument("--action-anchor-coef", type=float, default=0.0, help="Penalty coefficient for keeping actor-sampled coefficients near --coefficient-init.")
    train.add_argument("--cql-coef", type=float, default=0.0, help="Conservative Q penalty coefficient. Penalizes actor-sampled actions whose Q exceeds replay-action Q.")
    train.add_argument("--tau", type=float, default=0.005)
    train.add_argument("--alpha", type=float, default=0.02)
    train.add_argument("--auto-alpha", action="store_true")
    train.add_argument("--target-entropy", type=float, default=None)
    train.add_argument("--min-concentration", type=float, default=0.05, help="Kept for compatibility with older Dirichlet SAC runs; unused by the positive softplus-normal actor.")
    train.add_argument("--log-std-min", type=float, default=-5.0)
    train.add_argument("--log-std-max", type=float, default=1.0)
    train.add_argument("--terminal-bonus", type=float, default=1.0)
    train.add_argument("--reward-scale", type=float, default=1.0)
    train.add_argument("--activation-reward-coef", type=float, default=0.0, help="Dense layer-wise reward coefficient for cosine similarity between merged and source-model activations.")
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
        "--actor-update-delay": args.actor_update_delay,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.random_steps < 0:
        raise ValueError("--random-steps must be non-negative.")
    if args.coefficient_mode == "positive" and args.coefficient_init <= 0:
        raise ValueError("--coefficient-init must be positive when --coefficient-mode positive.")
    if args.tau <= 0 or args.tau > 1:
        raise ValueError("--tau must be in (0, 1].")
    if args.min_concentration <= 0:
        raise ValueError("--min-concentration must be positive.")
    if args.log_std_min >= args.log_std_max:
        raise ValueError("--log-std-min must be smaller than --log-std-max.")
    if args.activation_reward_coef < 0:
        raise ValueError("--activation-reward-coef must be non-negative.")
    if args.action_anchor_coef < 0:
        raise ValueError("--action-anchor-coef must be non-negative.")
    if args.cql_coef < 0:
        raise ValueError("--cql-coef must be non-negative.")
