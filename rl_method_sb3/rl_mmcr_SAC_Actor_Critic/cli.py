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
        help="State representation used by the underlying MMCR environment.",
    )
    model.add_argument("--coefficient-mode", choices=["positive"], default="positive")
    model.add_argument("--coefficient-init", type=float, default=0.3, help="Initial coefficients used by the MMCR environment before the first action.")
    model.add_argument("--action-max", type=float, default=10.0, help="Upper bound for each positive coefficient in the SB3 Box action space.")
    model.add_argument("--log-std-init", type=float, default=-2.0, help="Initial log standard deviation used by the SB3 SAC policy.")
    model.add_argument("--export-policy", choices=["best", "final"], default="best")

    train = parser.add_argument_group("sb3 sac training")
    train.add_argument("--episodes", type=int, default=300)
    train.add_argument("--gamma", type=float, default=1.0)
    train.add_argument("--lr", type=float, default=3e-4, help="SB3 SAC learning_rate for policy, critics, and entropy coefficient optimizer.")
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--buffer-size", type=int, default=50000)
    train.add_argument("--learning-starts", type=int, default=200, help="Number of environment steps before SB3 starts gradient updates.")
    train.add_argument("--train-freq", type=int, default=1, help="SB3 train_freq in environment steps.")
    train.add_argument("--gradient-steps", type=int, default=1, help="SB3 gradient_steps per train_freq.")
    train.add_argument("--tau", type=float, default=0.005)
    train.add_argument("--ent-coef", default="0.02", help="SB3 SAC ent_coef: a float, 'auto', or 'auto_<initial_value>'.")
    train.add_argument("--target-entropy", default="auto", help="SB3 SAC target_entropy: 'auto' or a float.")
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
    runtime.add_argument("--output-dir", default="rl_method_sb3_runs/rl_mmcr_SAC_Actor_Critic/default")
    runtime.add_argument("--gpu", type=int, default=0)
    runtime.add_argument("--num-workers", type=int, default=4)
    runtime.add_argument("--seed", type=int, default=2026)
    runtime.add_argument("--amp", action="store_true")
    runtime.add_argument("--cache-task-vectors-device", action="store_true")
    runtime.add_argument("--skip-final-eval", action="store_true")
    return parser.parse_args()


def parse_float_or_auto(value: str, *, allow_auto_prefix: bool = False) -> str | float:
    lowered = value.lower()
    if lowered == "auto":
        return "auto"
    if allow_auto_prefix and lowered.startswith("auto_"):
        float(lowered.split("_", maxsplit=1)[1])
        return lowered
    return float(value)


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "--episodes": args.episodes,
        "--log-every": args.log_every,
        "--reward-eval-interval": args.reward_eval_interval,
        "--reward-batches-per-dataset": args.reward_batches_per_dataset,
        "--reward-batch-size": args.reward_batch_size,
        "--batch-size": args.batch_size,
        "--buffer-size": args.buffer_size,
        "--train-freq": args.train_freq,
        "--gradient-steps": args.gradient_steps,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.learning_starts < 0:
        raise ValueError("--learning-starts must be non-negative.")
    if args.coefficient_mode == "positive" and args.coefficient_init <= 0:
        raise ValueError("--coefficient-init must be positive when --coefficient-mode positive.")
    if args.action_max <= 0:
        raise ValueError("--action-max must be positive.")
    if args.tau <= 0 or args.tau > 1:
        raise ValueError("--tau must be in (0, 1].")
    if args.activation_reward_coef < 0:
        raise ValueError("--activation-reward-coef must be non-negative.")

    args.ent_coef = parse_float_or_auto(args.ent_coef, allow_auto_prefix=True)
    args.target_entropy = parse_float_or_auto(args.target_entropy)
