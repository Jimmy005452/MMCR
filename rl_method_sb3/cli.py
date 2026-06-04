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
    data.add_argument("--top-k-percent", type=float, default=20.0)
    data.add_argument("--no-download", action="store_true")
    data.add_argument("--source-baseline-json", default=None)
    data.add_argument("--include-rejected-synthetic", action="store_true")

    model = parser.add_argument_group("model")
    model.add_argument("--arch", default=DEFAULT_ARCH)
    model.add_argument("--policy-hidden-dim", type=int, default=128)
    model.add_argument("--merge-granularity", choices=["global", "layer"], default="global")
    model.add_argument(
        "--state-mode",
        choices=["minimal", "full_coefficients"],
        default="minimal",
        help="State representation used by SB3 policies and value/Q networks.",
    )
    model.add_argument("--coefficient-mode", choices=["positive"], default="positive")
    model.add_argument("--coefficient-init", type=float, default=0.3)
    model.add_argument("--action-max", type=float, default=2.0, help="Upper bound for each positive coefficient in the SB3 Box action space.")
    model.add_argument("--export-policy", choices=["best", "final"], default="best")

    train = parser.add_argument_group("training")
    train.add_argument("--algo", choices=["ppo", "sac"], default="ppo", help="Stable-Baselines3 algorithm.")
    train.add_argument("--episodes", type=int, default=None, help="Exact number of completed MMCR episodes. If omitted, uses --iterations * --group-size for GRPO compatibility.")
    train.add_argument("--iterations", type=int, default=300, help="Compatibility with the old GRPO CLI.")
    train.add_argument("--group-size", type=int, default=8, help="Compatibility with the old GRPO CLI; also used to choose the default PPO rollout length.")
    train.add_argument("--grpo-epochs", type=int, default=4, help="Compatibility with the old GRPO CLI; used as PPO --n-epochs when --n-epochs is omitted.")
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--gamma", type=float, default=1.0)
    train.add_argument("--gae-lambda", type=float, default=0.95)
    train.add_argument("--clip-eps", type=float, default=0.2)
    train.add_argument("--entropy-coef", type=float, default=0.01)
    train.add_argument("--target-kl", type=float, default=0.03, help="Compatibility with the old GRPO CLI; passed to SB3 PPO target_kl.")
    train.add_argument("--advantage-mode", choices=["rloo", "zscore", "rank"], default="rloo", help="Compatibility with the old GRPO CLI; SB3 computes advantages internally.")
    train.add_argument("--min-concentration", type=float, default=0.05, help="Compatibility with older Dirichlet GRPO runs; unused by SB3.")
    train.add_argument("--log-std-min", type=float, default=-5.0, help="Compatibility with the old GRPO CLI; unused by SB3 policies.")
    train.add_argument("--log-std-max", type=float, default=1.0, help="Compatibility with the old GRPO CLI; unused by SB3 policies.")
    train.add_argument("--value-coef", type=float, default=0.5)
    train.add_argument("--max-grad-norm", type=float, default=0.5)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--n-steps", type=int, default=None, help="PPO rollout steps. Defaults to group_size * episode_length.")
    train.add_argument("--n-epochs", type=int, default=None, help="PPO epochs per update. Defaults to --grpo-epochs.")
    train.add_argument("--learning-starts", type=int, default=100, help="SAC warmup timesteps.")
    train.add_argument("--buffer-size", type=int, default=50000, help="SAC replay buffer size.")
    train.add_argument("--tau", type=float, default=0.005, help="SAC target-network update coefficient.")
    train.add_argument("--train-freq", type=int, default=1, help="SAC train_freq in environment steps.")
    train.add_argument("--gradient-steps", type=int, default=1, help="SAC gradient steps per train_freq.")
    train.add_argument("--terminal-bonus", type=float, default=1.0)
    train.add_argument("--reward-scale", type=float, default=1.0)
    train.add_argument("--reward-mode", choices=["accuracy_retention", "entropy", "synthetic_entropy", "synthetic_proxy"], default="accuracy_retention")
    train.add_argument("--kl-weight", type=float, default=1.0)
    train.add_argument("--agreement-weight", type=float, default=0.5)
    train.add_argument("--entropy-weight", type=float, default=0.1)
    train.add_argument("--activation-reward-coef", type=float, default=0.0)
    train.add_argument("--step-reward-coef", type=float, default=0.25)
    train.add_argument("--accuracy-imbalance-coef", type=float, default=0.5)
    train.add_argument("--retention-worst-coef", type=float, default=0.5)
    train.add_argument("--retention-drop-coef", type=float, default=1.0)
    train.add_argument("--reward-eval-interval", type=int, default=1)
    train.add_argument("--episode-reward-only", action="store_true")
    train.add_argument("--log-every", type=int, default=10)

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--output-dir", default="rl_method_sb3_runs/default")
    runtime.add_argument("--gpu", type=int, default=0)
    runtime.add_argument("--num-workers", type=int, default=4)
    runtime.add_argument("--seed", type=int, default=2026)
    runtime.add_argument("--amp", action="store_true")
    runtime.add_argument("--cache-task-vectors-device", action="store_true")
    runtime.add_argument("--batched-reward-eval", action="store_true")
    runtime.add_argument("--batched-reward-max-samples", type=int, default=128)
    runtime.add_argument("--skip-final-eval", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.episodes = args.episodes if args.episodes is not None else args.iterations * args.group_size
    args.n_epochs = args.n_epochs if args.n_epochs is not None else args.grpo_epochs

    positive_ints = {
        "--episodes": args.episodes,
        "--iterations": args.iterations,
        "--group-size": args.group_size,
        "--grpo-epochs": args.grpo_epochs,
        "--n-epochs": args.n_epochs,
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

    if args.n_steps is not None and args.n_steps <= 1:
        raise ValueError("--n-steps must be greater than 1 for PPO.")
    if args.learning_starts < 0:
        raise ValueError("--learning-starts must be non-negative.")
    if args.target_kl is not None and args.target_kl <= 0:
        raise ValueError("--target-kl must be positive when set.")
    if args.min_concentration <= 0:
        raise ValueError("--min-concentration must be positive.")
    if args.log_std_min >= args.log_std_max:
        raise ValueError("--log-std-min must be smaller than --log-std-max.")
    if args.coefficient_init <= 0:
        raise ValueError("--coefficient-init must be positive.")
    if args.action_max <= 0:
        raise ValueError("--action-max must be positive.")
    if args.activation_reward_coef < 0:
        raise ValueError("--activation-reward-coef must be non-negative.")
    if args.kl_weight < 0:
        raise ValueError("--kl-weight must be non-negative.")
    if args.agreement_weight < 0:
        raise ValueError("--agreement-weight must be non-negative.")
    if args.entropy_weight < 0:
        raise ValueError("--entropy-weight must be non-negative.")
    if args.reward_mode != "accuracy_retention" and args.activation_reward_coef != 0:
        raise ValueError("--activation-reward-coef is only supported with --reward-mode accuracy_retention for now.")
