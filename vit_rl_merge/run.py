import argparse
from pathlib import Path

import torch

from mmcr.checkpoints import ENCODER_FILE
from mmcr.evaluation import resolve_head_path
from mmcr.models import DEFAULT_ARCH
from mmcr.utils import build_device, seed_everything, write_json

from vit_rl_merge.env import ViTTensorMergeEnv
from vit_rl_merge.reward import format_float_dict, format_percent_dict
from vit_rl_merge.trainer import train_actor_critic


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--checkpoint-root", default="checkpoints")
    parser.add_argument("--head-root", default=None)
    parser.add_argument("--zeroshot", default=None)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--arch", default=DEFAULT_ARCH)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Extra multiplier after RL coefficients. Use 1.0 to let the policy fully control merge strength.",
    )
    parser.add_argument("--decision-level", choices=["task", "group", "layer", "tensor"], default="group")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--rollouts-per-update",
        type=int,
        default=1,
        help="Number of complete episodes sampled before one policy update.",
    )
    parser.add_argument(
        "--no-normalize-advantages",
        action="store_true",
        help="Disable advantage normalization across rollouts in the same update.",
    )
    parser.add_argument("--batches-per-dataset", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument(
        "--coefficient-mode",
        choices=["softmax", "sigmoid", "positive", "unconstrained"],
        default="sigmoid",
    )
    parser.add_argument(
        "--coefficient-init",
        type=float,
        default=0.9,
        help="Initial coefficient value before exploration. Ignored by softmax.",
    )
    parser.add_argument(
        "--objective",
        choices=["supervised", "entropy"],
        default="supervised",
        help="supervised uses label-based retention reward; entropy is label-free.",
    )
    parser.add_argument("--worst-weight", type=float, default=0.5)
    parser.add_argument("--std-weight", type=float, default=0.25)
    parser.add_argument(
        "--reward-mode",
        choices=["balanced", "worst", "mean_worst", "harmonic"],
        default="balanced",
        help="Reward objective computed from per-dataset capability retention.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to the reward for stronger policy-gradient signal.",
    )
    parser.add_argument("--policy-hidden-dim", type=int, default=128)
    parser.add_argument(
        "--init-log-std",
        type=float,
        default=-0.5,
        help="Initial log standard deviation for coefficient exploration.",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--export-policy", choices=["final", "best"], default="best")
    parser.add_argument("--debug-decisions", action="store_true")
    parser.add_argument("--debug-all-episodes", action="store_true")
    parser.add_argument(
        "--no-print-selected-decisions",
        action="store_true",
        help="Do not print the selected best/final coefficient table after training.",
    )
    parser.add_argument(
        "--max-printed-decisions",
        type=int,
        default=120,
        help="Maximum decision rows to print. Use 0 or a negative value to print all rows.",
    )
    parser.add_argument("--history-json", default=None)
    parser.add_argument("--lambda-json", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def coefficient_summary(coefficients, datasets, decision_groups):
    return {
        "mean_by_dataset": dict(zip(datasets, coefficients.mean(dim=0).tolist())),
        "min_by_dataset": dict(zip(datasets, coefficients.min(dim=0).values.tolist())),
        "max_by_dataset": dict(zip(datasets, coefficients.max(dim=0).values.tolist())),
        "per_decision": {
            group["name"]: {
                "num_tensors": len(group["keys"]),
                "coefficients": dict(zip(datasets, coefficients[index].tolist())),
            }
            for index, group in enumerate(decision_groups)
        },
    }


def print_decision_table(title, coefficients, datasets, decision_groups, max_rows: int):
    print(title)
    if coefficients.numel() == 0:
        print("  no coefficients recorded")
        return

    rows_to_print = len(decision_groups)
    if max_rows > 0:
        rows_to_print = min(rows_to_print, max_rows)

    for index, group in enumerate(decision_groups[:rows_to_print]):
        coeff_text = ", ".join(
            f"{dataset}: {value:.4f}"
            for dataset, value in zip(datasets, coefficients[index].tolist())
        )
        print(
            f"  step={index:03d} group={group['name']} "
            f"num_tensors={len(group['keys'])} coeffs={{{coeff_text}}}"
        )

    if rows_to_print < len(decision_groups):
        remaining = len(decision_groups) - rows_to_print
        print(f"  ... skipped {remaining} rows; full coefficients are saved in the lambda JSON.")


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    checkpoint_root = Path(args.checkpoint_root)
    head_root = Path(args.head_root) if args.head_root is not None else checkpoint_root
    zeroshot_path = Path(args.zeroshot) if args.zeroshot is not None else checkpoint_root / "zeroshot.pt"
    encoder_paths = [checkpoint_root / dataset / ENCODER_FILE for dataset in args.datasets]
    head_paths = [resolve_head_path(dataset, head_root) for dataset in args.datasets]

    for dataset, encoder_path, head_path in zip(args.datasets, encoder_paths, head_paths):
        print(f"{dataset}: encoder={encoder_path} head={head_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(encoder_path)
        if not head_path.exists():
            raise FileNotFoundError(head_path)
    if not zeroshot_path.exists():
        raise FileNotFoundError(zeroshot_path)

    device = build_device(args.gpu)
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print("Using device: cpu")

    env = ViTTensorMergeEnv(
        arch=args.arch,
        datasets=args.datasets,
        head_paths=head_paths,
        zeroshot_path=zeroshot_path,
        encoder_paths=encoder_paths,
        data_root=args.data_root,
        device=device,
        scale=args.scale,
        decision_level=args.decision_level,
        batches_per_dataset=args.batches_per_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
        download=not args.no_download,
        objective=args.objective,
        worst_weight=args.worst_weight,
        std_weight=args.std_weight,
        reward_mode=args.reward_mode,
        reward_scale=args.reward_scale,
    )
    wa_baseline = env.evaluate_weight_average_baseline()
    if args.objective == "supervised":
        print(f"Source accuracies: {format_percent_dict(env.source_accuracies)}")
        print(
            f"WA baseline reward={wa_baseline['reward']:.4f} "
            f"accuracies={format_percent_dict(wa_baseline['accuracies'])} "
            f"retentions={format_percent_dict(wa_baseline['retentions'])}"
        )
    else:
        print(
            f"WA baseline reward={wa_baseline['reward']:.4f} "
            f"entropies={format_float_dict(wa_baseline['entropies'])}"
        )
    result = train_actor_critic(
        env=env,
        episodes=args.episodes,
        rollouts_per_update=args.rollouts_per_update,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        hidden_dim=args.policy_hidden_dim,
        init_log_std=args.init_log_std,
        coefficient_mode=args.coefficient_mode,
        coefficient_init=args.coefficient_init,
        debug_decisions=args.debug_decisions,
        debug_first_episode_only=not args.debug_all_episodes,
        normalize_advantages=not args.no_normalize_advantages,
    )

    selected = result[args.export_policy]
    best_message = f"Best episode={result['best']['episode']} reward={result['best']['reward']:.4f}"
    final_message = f"Final policy reward={result['final']['reward']:.4f}"
    if args.objective == "supervised":
        best_message += (
            f" accuracies={format_percent_dict(result['best']['info']['accuracies'])}"
            f" retentions={format_percent_dict(result['best']['info']['retentions'])}"
        )
        final_message += (
            f" accuracies={format_percent_dict(result['final']['info']['accuracies'])}"
            f" retentions={format_percent_dict(result['final']['info']['retentions'])}"
        )
    else:
        best_message += f" entropies={format_float_dict(result['best']['info']['entropies'])}"
        final_message += f" entropies={format_float_dict(result['final']['info']['entropies'])}"
    print(best_message)
    print(final_message)
    if not args.no_print_selected_decisions:
        print_decision_table(
            title=f"Selected {args.export_policy} decisions:",
            coefficients=selected["coefficients"],
            datasets=args.datasets,
            decision_groups=env.decision_groups,
            max_rows=args.max_printed_decisions,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(selected["state_dict"], output_path)
    print(f"Saved ViT RL merged encoder to {output_path} (export_policy={args.export_policy})")

    history_path = Path(args.history_json) if args.history_json is not None else output_path.with_suffix(".history.json")
    write_json(
        history_path,
        {
            "datasets": args.datasets,
            "checkpoint_root": str(checkpoint_root),
            "head_root": str(head_root),
            "zeroshot": str(zeroshot_path),
            "output": str(output_path),
            "export_policy": args.export_policy,
            "scale": args.scale,
            "objective": args.objective,
            "decision_level": args.decision_level,
            "coefficient_mode": args.coefficient_mode,
            "coefficient_init": args.coefficient_init,
            "init_log_std": args.init_log_std,
            "reward_mode": args.reward_mode,
            "reward_scale": args.reward_scale,
            "worst_weight": args.worst_weight,
            "std_weight": args.std_weight,
            "episodes": args.episodes,
            "rollouts_per_update": args.rollouts_per_update,
            "normalize_advantages": not args.no_normalize_advantages,
            "batches_per_dataset": args.batches_per_dataset,
            "source_accuracies": env.source_accuracies,
            "weight_average_baseline": wa_baseline,
            "final": {
                "reward": result["final"]["reward"],
                "accuracies": result["final"]["info"]["accuracies"],
                "retentions": result["final"]["info"]["retentions"],
                "entropies": result["final"]["info"]["entropies"],
            },
            "best": {
                "episode": result["best"]["episode"],
                "reward": result["best"]["reward"],
                "accuracies": result["best"]["info"]["accuracies"],
                "retentions": result["best"]["info"]["retentions"],
                "entropies": result["best"]["info"]["entropies"],
            },
            "history": result["history"],
        },
    )
    print(f"Saved ViT RL history to {history_path}")

    lambda_path = Path(args.lambda_json) if args.lambda_json is not None else output_path.with_suffix(".lambdas.json")
    selected_summary = coefficient_summary(selected["coefficients"], args.datasets, env.decision_groups)
    write_json(
        lambda_path,
        {
            "datasets": args.datasets,
            "decision_level": args.decision_level,
            "coefficient_mode": args.coefficient_mode,
            "coefficient_init": args.coefficient_init,
            "decision_groups": [
                {"name": group["name"], "keys": group["keys"]}
                for group in env.decision_groups
            ],
            "selected_policy": args.export_policy,
            "lambda_summary": selected_summary,
            "selected": selected_summary,
            "best": coefficient_summary(result["best"]["coefficients"], args.datasets, env.decision_groups),
            "final": coefficient_summary(result["final"]["coefficients"], args.datasets, env.decision_groups),
        },
    )
    print(f"Saved ViT RL lambdas to {lambda_path}")


if __name__ == "__main__":
    main()
