from __future__ import annotations

from pathlib import Path
import json

import torch
from tqdm import tqdm

from mmcr.checkpoints import ENCODER_FILE, load_head
from mmcr.data import normalize_dataset_key
from mmcr.evaluation import evaluate_encoder, resolve_head_path
from mmcr.models import build_image_encoder
from mmcr.utils import build_device, seed_everything, write_json
from .cli import parse_args, validate_args
from .data import build_reward_batches
from .env import RLMMCREnv
from .merge import load_layered_ties_task_vectors
from .plotting import plot_reward_curves, plot_training_curves
from .policy import HybridActorCritic
from .ppo import collect_rollout, deterministic_policy_result, update_policy


def resolve_source_encoder_path(checkpoint_root: Path, dataset: str) -> Path:
    direct_path = checkpoint_root / dataset / ENCODER_FILE
    if direct_path.exists():
        return direct_path

    normalized_path = checkpoint_root / normalize_dataset_key(dataset) / ENCODER_FILE
    return normalized_path if normalized_path.exists() else direct_path



def load_source_baseline_scores(path: str | None, datasets: list[str]) -> dict[str, float] | None:
    if path is None:
        return None

    baseline_path = Path(path)
    with baseline_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload.get("source_baseline_scores"), dict):
        raw_scores = payload["source_baseline_scores"]
    elif isinstance(payload.get("results"), dict):
        raw_scores = {
            dataset: row["acc"]
            for dataset, row in payload["results"].items()
            if dataset != "average" and isinstance(row, dict) and "acc" in row
        }
    else:
        raw_scores = {
            dataset: row["acc"]
            for dataset, row in payload.items()
            if dataset != "average" and isinstance(row, dict) and "acc" in row
        }

    scores = {}
    missing = []
    for dataset in datasets:
        normalized = normalize_dataset_key(dataset)
        if dataset in raw_scores:
            scores[dataset] = float(raw_scores[dataset])
        elif normalized in raw_scores:
            scores[dataset] = float(raw_scores[normalized])
        else:
            missing.append(dataset)
    if missing:
        raise ValueError(f"{baseline_path} is missing source baseline dataset(s): {', '.join(missing)}")

    print(f"Loaded source baseline scores from {baseline_path}")
    return scores

def require_existing(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required checkpoint(s): " + ", ".join(str(path) for path in missing))


def format_retention(value: float) -> str:
    return f"{value:.4f}x"


def format_scores(scores: dict[str, float], datasets: list[str]) -> str:
    parts = [f"{dataset}={scores[dataset] * 100:.1f}%" for dataset in datasets if dataset in scores]
    return " acc={" + ", ".join(parts) + "}" if parts else ""


def build_environment(args, device: torch.device) -> RLMMCREnv:
    checkpoint_root = Path(args.checkpoint_root)
    zeroshot_path = Path(args.zeroshot) if args.zeroshot else checkpoint_root / "zeroshot.pt"
    encoder_paths = [resolve_source_encoder_path(checkpoint_root, dataset) for dataset in args.datasets]
    head_paths = [resolve_head_path(dataset, checkpoint_root) for dataset in args.datasets]
    require_existing([zeroshot_path, *encoder_paths, *head_paths])

    layered_task_vectors = load_layered_ties_task_vectors(
        zeroshot_path=zeroshot_path,
        finetuned_paths=encoder_paths,
        task_names=args.datasets,
        top_k_percent=args.top_k_percent,
    )
    reward_batches = build_reward_batches(
        datasets=args.datasets,
        data_root=args.data_root,
        arch=args.arch,
        batch_size=args.reward_batch_size,
        batches_per_dataset=args.reward_batches_per_dataset,
        num_workers=args.num_workers,
        split=args.reward_split,
        download=not args.no_download,
    )
    heads = {
        dataset: load_head(head_path, device=device)
        for dataset, head_path in zip(args.datasets, head_paths)
    }

    source_baseline_scores = load_source_baseline_scores(args.source_baseline_json, args.datasets)

    return RLMMCREnv(
        encoder=build_image_encoder(arch=args.arch, pretrained=False).to(device),
        heads=heads,
        layered_task_vectors=layered_task_vectors,
        reward_batches=reward_batches,
        device=device,
        amp=args.amp,
        terminal_bonus=args.terminal_bonus,
        reward_scale=args.reward_scale,
        step_reward_coef=args.step_reward_coef,
        accuracy_imbalance_coef=args.accuracy_imbalance_coef,
        retention_worst_coef=args.retention_worst_coef,
        retention_drop_coef=args.retention_drop_coef,
        reward_eval_interval=args.reward_eval_interval,
        episode_reward_only=args.episode_reward_only,
        coefficient_mode=args.coefficient_mode,
        coefficient_init=args.coefficient_init,
        cache_task_vectors_device=args.cache_task_vectors_device,
        merge_granularity=args.merge_granularity,
        source_baseline_scores=source_baseline_scores,
        activation_reward_coef=args.activation_reward_coef,
        state_mode=args.state_mode,
    )


def summarize_rollout(rollout: dict, episode: int, best_sample: dict) -> dict:
    terminal = rollout["infos"][-1]
    reward_stats = terminal["reward_stats"]
    objective = float(terminal["objective"])
    reward_sum = float(sum(rollout["rewards"]))

    if objective > best_sample["objective"]:
        best_sample.update(
            selected=rollout["selected"],
            coefficients=rollout["coefficients"],
            objective=objective,
            mean_accuracy=float(terminal["average"]),
            mean_retention=float(reward_stats["mean_retention"]),
            scores=terminal["scores"],
            reward_stats=reward_stats,
        )

    return {
        "episode": episode,
        "sample_average": float(terminal["average"]),
        "sample_retention": float(reward_stats["mean_retention"]),
        "sample_objective": objective,
        "best_objective": float(best_sample["objective"]),
        "best_retention": float(best_sample["mean_retention"]),
        "reward_sum": reward_sum,
        "final_scores": terminal["scores"],
        "reward_stats": reward_stats,
    }


def mean_terminal(rollouts: list[dict], key: str) -> float:
    return sum(float(rollout["infos"][-1][key]) for rollout in rollouts) / len(rollouts)


def mean_reward_stat(rollouts: list[dict], key: str) -> float:
    return sum(float(rollout["infos"][-1]["reward_stats"][key]) for rollout in rollouts) / len(rollouts)


def train(args, env: RLMMCREnv, model: HybridActorCritic, optimizer: torch.optim.Optimizer):
    best_sample = {"objective": float("-inf"), "mean_retention": float("-inf")}
    update_history = []
    episode_history = []
    episodes_completed = 0
    progress = tqdm(total=args.episodes, desc="RL-MMCR")

    while episodes_completed < args.episodes:
        rollouts = []
        rollouts_this_update = min(args.rollouts_per_update, args.episodes - episodes_completed)
        for _ in range(rollouts_this_update):
            rollout = collect_rollout(env, model)
            rollouts.append(rollout)
            episode_history.append(summarize_rollout(rollout, episodes_completed, best_sample))
            episodes_completed += 1
            progress.update(1)

        stats = update_policy(
            model,
            optimizer,
            rollouts,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            ppo_epochs=args.ppo_epochs,
            target_kl=args.target_kl,
            max_grad_norm=args.max_grad_norm,
            minibatch_size=args.ppo_minibatch_size,
            normalize_advantages=not args.no_advantage_norm,
            device=env.device,
        )

        history_row = {
            "update": len(update_history) + 1,
            "episodes_completed": episodes_completed,
            "sample_average": mean_terminal(rollouts, "average"),
            "sample_retention": mean_reward_stat(rollouts, "mean_retention"),
            "sample_objective": mean_terminal(rollouts, "objective"),
            "best_objective": float(best_sample["objective"]),
            "best_retention": float(best_sample["mean_retention"]),
            "reward_sum": sum(sum(rollout["rewards"]) for rollout in rollouts) / len(rollouts),
            "deterministic_average": None,
            "deterministic_retention": None,
            "deterministic_objective": None,
            "deterministic_scores": None,
            **stats.__dict__,
        }

        should_log = episodes_completed % args.log_every == 0 or episodes_completed == args.episodes
        if should_log:
            deterministic = deterministic_policy_result(env, model, args.gate_threshold)
            deterministic_retention = deterministic["infos"][-1]["reward_stats"]["mean_retention"]
            history_row.update(
                deterministic_average=float(deterministic["average"]),
                deterministic_retention=float(deterministic_retention),
                deterministic_objective=float(deterministic["objective"]),
                deterministic_scores=deterministic["scores"],
            )
            progress.write(
                f"episodes={episodes_completed} retention sample={format_retention(history_row['sample_retention'])} "
                f"best={format_retention(best_sample['mean_retention'])} "
                f"deterministic={format_retention(deterministic_retention)} "
                f"{format_scores(rollouts[-1]['infos'][-1]['scores'], args.datasets)} "
                f"reward={history_row['reward_sum']:.4f} loss={stats.loss:.4f} kl={stats.approx_kl:.4f}"
            )

        update_history.append(history_row)
        progress.set_postfix(
            sample=format_retention(history_row["sample_retention"]),
            best=format_retention(best_sample["mean_retention"]),
            reward=f"{history_row['reward_sum']:.4f}",
            loss=f"{stats.loss:.4f}",
            kl=f"{stats.approx_kl:.4f}",
        )

    progress.close()
    return best_sample, update_history, episode_history


def export_results(args, env: RLMMCREnv, model: HybridActorCritic, best_sample: dict, history: list[dict], episodes: list[dict]):
    output_dir = Path(args.output_dir)
    final_policy = deterministic_policy_result(env, model, args.gate_threshold)
    final_coefficients = torch.tensor(final_policy["coefficients"], dtype=torch.float32)
    best_coefficients = torch.tensor(best_sample["coefficients"], dtype=torch.float32)

    if args.export_policy == "best":
        export_coefficients = best_coefficients
        exported_policy = {"type": "best_sample", **{key: best_sample[key] for key in (
            "objective",
            "mean_accuracy",
            "mean_retention",
            "scores",
            "reward_stats",
        )}}
    else:
        terminal_stats = final_policy["infos"][-1]["reward_stats"]
        export_coefficients = final_coefficients
        exported_policy = {
            "type": "final_deterministic",
            "objective": float(final_policy["objective"]),
            "mean_accuracy": float(final_policy["average"]),
            "mean_retention": float(terminal_stats["mean_retention"]),
            "scores": final_policy["scores"],
            "reward_stats": terminal_stats,
        }

    encoder_path = output_dir / "encoder.pt"
    torch.save(env.export_merged_state(export_coefficients), encoder_path)
    plot_path = plot_training_curves(history, episodes, output_dir / "training_curves.png")
    reward_plot_path = plot_reward_curves(episodes, output_dir / "reward_curves.png")

    payload = {
        "config": vars(args),
        "source_baseline_scores": env.source_baseline_scores,
        "state_dim": env.state_dim,
        "num_models": env.num_models,
        "num_layers": env.num_layers,
        "layer_names": env.layer_names,
        "action_type": args.action_mode,
        "merge_granularity": args.merge_granularity,
        "exported_policy": exported_policy,
        "final_policy": final_policy,
        "best_sample": best_sample,
        "history": history,
        "episode_history": episodes,
        "encoder_path": str(encoder_path),
        "plot_path": str(plot_path),
        "reward_plot_path": str(reward_plot_path),
    }

    if not args.skip_final_eval:
        payload["final_eval"] = evaluate_encoder(
            encoder_path=encoder_path,
            datasets=args.datasets,
            checkpoint_root=Path(args.checkpoint_root),
            data_root=args.data_root,
            arch=args.arch,
            batch_size=args.reward_batch_size,
            num_workers=args.num_workers,
            device=env.device,
            amp=args.amp,
            download=not args.no_download,
        )

    write_json(output_dir / "results.json", payload)
    return encoder_path, plot_path, reward_plot_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = build_device(args.gpu)
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    env = build_environment(args, device)
    model = HybridActorCritic(
        env.state_dim,
        env.num_models,
        hidden_dim=args.policy_hidden_dim,
        coefficient_mode=args.coefficient_mode,
        coefficient_init=args.coefficient_init,
        action_mode=args.action_mode,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_sample, history, episode_history = train(args, env, model, optimizer)
    encoder_path, plot_path, reward_plot_path = export_results(
        args,
        env,
        model,
        best_sample,
        history,
        episode_history,
    )
    print(f"Saved RL-MMCR results to {output_dir / 'results.json'}")
    print(f"Saved merged encoder to {encoder_path}")
    print(f"Saved training curves to {plot_path}")
    print(f"Saved reward curves to {reward_plot_path}")


if __name__ == "__main__":
    main()
