from __future__ import annotations

import importlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from tqdm import tqdm

from mmcr.evaluation import evaluate_encoder
from mmcr.utils import build_device, seed_everything, write_json

from .cli import parse_args, validate_args
from .grpo import PositiveSoftplusPolicy, compute_advantages, update_policy

ppo_train = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.train")
plotting = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.plotting")
coefficients_to_dict = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.merge").coefficients_to_dict


def format_retention(value: float) -> str:
    return f"{value:.4f}x"


def format_scores(scores: dict[str, float], datasets: list[str]) -> str:
    parts = [f"{dataset}={scores[dataset] * 100:.1f}%" for dataset in datasets if dataset in scores]
    return " acc={" + ", ".join(parts) + "}" if parts else ""


def same_torch_device(left: torch.device, right: torch.device) -> bool:
    if left.type != right.type:
        return False
    if left.type != "cuda":
        return True
    return left.index == right.index


def sync_policy_to_device(source: PositiveSoftplusPolicy, target: PositiveSoftplusPolicy, device: torch.device) -> None:
    state = {key: value.detach().to(device, non_blocking=True) for key, value in source.state_dict().items()}
    target.load_state_dict(state)


def build_trajectory_contexts(args, env, policy: PositiveSoftplusPolicy) -> list[dict]:
    if env.merge_granularity != "layer" or not args.trajectory_devices:
        return []

    contexts = []
    for device_id in args.trajectory_devices:
        device = build_device(device_id)
        if same_torch_device(device, env.device):
            worker_env = env.fork()
        else:
            worker_env = ppo_train.build_environment(args, device)
        worker_policy = PositiveSoftplusPolicy(
            worker_env.state_dim,
            worker_env.num_models,
            hidden_dim=args.policy_hidden_dim,
            log_std_min=args.log_std_min,
            log_std_max=args.log_std_max,
            initial_coefficient=args.coefficient_init,
        ).to(device)
        sync_policy_to_device(policy, worker_policy, device)
        contexts.append({"device": device, "env": worker_env, "policy": worker_policy})

    if env.device.type == "cuda":
        torch.cuda.set_device(env.device)
    return contexts


@torch.no_grad()
def evaluate_action(env, coefficients: torch.Tensor) -> dict:
    state = env.reset().to(env.device)
    selected = torch.ones_like(coefficients)
    next_state, reward, done, info = env.step(selected, coefficients)
    if not done:
        raise RuntimeError("evaluate_action expects a global one-step environment.")
    expanded = env.expand_coefficients(coefficients.unsqueeze(0))
    return {
        "state": state.detach().cpu(),
        "next_state": next_state.detach().cpu(),
        "selected": [selected.detach().cpu().tolist()],
        "coefficients": [coefficients.detach().cpu().tolist()],
        "expanded_coefficients": expanded.tolist(),
        "reward": float(reward),
        "average": float(info["average"]),
        "objective": float(info["objective"]),
        "scores": info["scores"],
        "reward_stats": info["reward_stats"],
        "infos": [info],
        "coefficients_by_layer": coefficients_to_dict(
            expanded,
            env.layer_names,
            env.layered_task_vectors.task_names,
        ),
    }


@torch.no_grad()
def evaluate_trajectory(env, policy: PositiveSoftplusPolicy, deterministic: bool = False) -> dict:
    state = env.reset().to(env.device)
    states = []
    actions = []
    raw_actions = []
    log_probs = []
    old_mean = []
    old_log_std = []
    entropies = []
    selected_history = []
    coefficient_history = []
    rewards = []
    infos = []
    done = False

    while not done:
        states.append(state.detach())
        if deterministic:
            coefficients = policy.deterministic(state.unsqueeze(0)).squeeze(0)
            mean, log_std = policy.distribution_params(state.unsqueeze(0))
            raw = mean.squeeze(0)
            log_prob = policy.log_prob(raw.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
            entropy = torch.zeros((), device=env.device)
        else:
            coefficients_batch, raw_batch, log_prob_batch, entropy_batch, mean, log_std = policy.sample(state.unsqueeze(0))
            coefficients = coefficients_batch.squeeze(0)
            raw = raw_batch.squeeze(0)
            log_prob = log_prob_batch.squeeze(0)
            entropy = entropy_batch.squeeze(0)
        selected = torch.ones_like(coefficients)
        next_state, reward, done, info = env.step(selected, coefficients.detach())
        next_state = next_state.to(env.device)

        actions.append(coefficients.detach())
        raw_actions.append(raw.detach())
        log_probs.append(log_prob.detach())
        old_mean.append(mean.squeeze(0).detach())
        old_log_std.append(log_std.squeeze(0).detach())
        entropies.append(entropy.detach())
        selected_history.append(selected.detach().cpu().tolist())
        coefficient_history.append(coefficients.detach().cpu().tolist())
        rewards.append(float(reward))
        infos.append(info)
        state = next_state

    expanded = env.expand_coefficients(torch.tensor(coefficient_history, dtype=torch.float32))
    terminal = infos[-1]
    return {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "raw_actions": torch.stack(raw_actions),
        "old_log_probs": torch.stack(log_probs),
        "old_mean": torch.stack(old_mean),
        "old_log_std": torch.stack(old_log_std),
        "entropies": torch.stack(entropies),
        "selected": selected_history,
        "coefficients": coefficient_history,
        "expanded_coefficients": expanded.tolist(),
        "reward": float(sum(rewards)),
        "average": float(terminal["average"]),
        "objective": float(terminal["objective"]),
        "scores": terminal["scores"],
        "reward_stats": terminal["reward_stats"],
        "infos": infos,
        "coefficients_by_layer": coefficients_to_dict(
            expanded,
            env.layer_names,
            env.layered_task_vectors.task_names,
        ),
    }


@torch.no_grad()
def deterministic_policy_result(env, policy: PositiveSoftplusPolicy) -> dict:
    if env.merge_granularity == "global":
        state = env.reset().to(env.device)
        coefficients = policy.deterministic(state.unsqueeze(0)).squeeze(0)
        result = evaluate_action(env, coefficients)
    else:
        result = evaluate_trajectory(env, policy, deterministic=True)
    return {
        "selected": result["selected"],
        "coefficients": result["coefficients"],
        "expanded_coefficients": result["expanded_coefficients"],
        "average": result["average"],
        "objective": result["objective"],
        "scores": result["scores"],
        "infos": result["infos"],
        "coefficients_by_layer": result["coefficients_by_layer"],
    }


def fixed_selection_position(args) -> int:
    return int(getattr(args, "selection_reward_pool_position", -1))


def with_reward_pool_position(env, position: int, fn):
    if position < 0 or not hasattr(env, "set_reward_pool_position"):
        return fn()
    previous = getattr(env, "_reward_eval_counter", None)
    env.set_reward_pool_position(position)
    try:
        return fn()
    finally:
        if previous is not None:
            env.set_reward_pool_position(previous)


@torch.no_grad()
def evaluate_coefficients(env, coefficients_by_layer) -> dict:
    coefficients = torch.tensor(coefficients_by_layer, dtype=torch.float32, device=env.device)
    state = env.reset().to(env.device)
    selected_history = []
    coefficient_history = []
    rewards = []
    infos = []

    if env.merge_granularity == "global":
        if coefficients.ndim == 2:
            coefficients = coefficients[0]
        selected = torch.ones_like(coefficients)
        next_state, reward, done, info = env.step(selected, coefficients.detach())
        if not done:
            raise RuntimeError("Global coefficient evaluation did not terminate in one step.")
        selected_history.append(selected.detach().cpu().tolist())
        coefficient_history.append(coefficients.detach().cpu().tolist())
        rewards.append(float(reward))
        infos.append(info)
    else:
        coefficients = env.expand_coefficients(coefficients.detach().cpu()).to(env.device)
        done = False
        for layer_coefficients in coefficients:
            selected = torch.ones_like(layer_coefficients)
            next_state, reward, done, info = env.step(selected, layer_coefficients.detach())
            selected_history.append(selected.detach().cpu().tolist())
            coefficient_history.append(layer_coefficients.detach().cpu().tolist())
            rewards.append(float(reward))
            infos.append(info)
            state = next_state.to(env.device)
        if not done:
            raise RuntimeError("Layer coefficient evaluation finished before the environment terminated.")

    expanded = env.expand_coefficients(torch.tensor(coefficient_history, dtype=torch.float32))
    terminal = infos[-1]
    return {
        "state": state.detach().cpu(),
        "selected": selected_history,
        "coefficients": coefficient_history,
        "expanded_coefficients": expanded.tolist(),
        "reward": float(sum(rewards)),
        "average": float(terminal["average"]),
        "objective": float(terminal["objective"]),
        "scores": terminal["scores"],
        "reward_stats": terminal["reward_stats"],
        "infos": infos,
        "coefficients_by_layer": coefficients_to_dict(
            expanded,
            env.layer_names,
            env.layered_task_vectors.task_names,
        ),
    }


def evaluate_coefficients_for_selection(env, args, coefficients_by_layer) -> dict:
    position = fixed_selection_position(args)
    return with_reward_pool_position(
        env,
        position,
        lambda: evaluate_coefficients(env, coefficients_by_layer),
    )


def deterministic_selection_result(env, args, policy: PositiveSoftplusPolicy) -> dict:
    position = fixed_selection_position(args)
    return with_reward_pool_position(
        env,
        position,
        lambda: deterministic_policy_result(env, policy),
    )


def collect_global_group(env, policy: PositiveSoftplusPolicy, group_size: int) -> dict:
    base_state = env.reset().to(env.device)
    states = base_state.unsqueeze(0).expand(group_size, -1).contiguous()
    actions, raw_actions, log_probs, entropies, old_mean, old_log_std = policy.sample(states)

    results = []
    rewards = []
    objectives = []
    retentions = []
    for action in actions:
        result = evaluate_action(env, action.detach())
        results.append(result)
        rewards.append(result["reward"])
        objectives.append(result["objective"])
        retentions.append(result["reward_stats"]["mean_retention"])

    return {
        "states": states.detach(),
        "actions": actions.detach(),
        "raw_actions": raw_actions.detach(),
        "old_log_probs": log_probs.detach(),
        "old_mean": old_mean.detach(),
        "old_log_std": old_log_std.detach(),
        "entropies": entropies.detach(),
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=env.device),
        "objectives": objectives,
        "retentions": retentions,
        "results": results,
    }


def _pack_layer_trajectories(env, trajectories: list[dict]) -> dict:
    rewards = torch.tensor([trajectory["reward"] for trajectory in trajectories], dtype=torch.float32, device=env.device)
    objectives = [trajectory["objective"] for trajectory in trajectories]
    retentions = [trajectory["reward_stats"]["mean_retention"] for trajectory in trajectories]

    def to_train_device(name: str) -> torch.Tensor:
        return torch.cat([trajectory[name].to(env.device, non_blocking=True) for trajectory in trajectories], dim=0).detach()

    return {
        "states": to_train_device("states"),
        "actions": to_train_device("actions"),
        "raw_actions": to_train_device("raw_actions"),
        "old_log_probs": to_train_device("old_log_probs"),
        "old_mean": to_train_device("old_mean"),
        "old_log_std": to_train_device("old_log_std"),
        "entropies": to_train_device("entropies"),
        "trajectory_lengths": [int(trajectory["states"].shape[0]) for trajectory in trajectories],
        "rewards": rewards,
        "objectives": objectives,
        "retentions": retentions,
        "results": trajectories,
    }


def collect_layer_group(
    env,
    policy: PositiveSoftplusPolicy,
    group_size: int,
    trajectory_contexts: list[dict] | None = None,
    *,
    seed: int = 0,
    iteration: int = 0,
) -> dict:
    if not trajectory_contexts:
        trajectories = []
        for _ in range(group_size):
            if hasattr(env, "set_reward_pool_position"):
                env.set_reward_pool_position(iteration)
            trajectories.append(evaluate_trajectory(env, policy, deterministic=False))
        return _pack_layer_trajectories(env, trajectories)

    for context in trajectory_contexts:
        sync_policy_to_device(policy, context["policy"], context["device"])

    active_contexts = trajectory_contexts[: min(len(trajectory_contexts), group_size)]
    buckets = [[] for _ in active_contexts]
    for index in range(group_size):
        buckets[index % len(active_contexts)].append(index)

    def rollout_bucket(context: dict, indices: list[int]) -> list[tuple[int, dict]]:
        device = context["device"]
        if device.type == "cuda":
            torch.cuda.set_device(device)
        items = []
        for index in indices:
            rollout_seed = int(seed) + int(iteration) * 100000 + index
            if device.type == "cuda":
                torch.cuda.manual_seed(rollout_seed)
            else:
                torch.manual_seed(rollout_seed)
            if hasattr(context["env"], "set_reward_pool_position"):
                context["env"].set_reward_pool_position(iteration)
            items.append((index, evaluate_trajectory(context["env"], context["policy"], deterministic=False)))
        return items

    indexed = []
    with ThreadPoolExecutor(max_workers=len(active_contexts)) as executor:
        futures = [
            executor.submit(rollout_bucket, context, bucket)
            for context, bucket in zip(active_contexts, buckets)
            if bucket
        ]
        for future in futures:
            indexed.extend(future.result())

    if env.device.type == "cuda":
        torch.cuda.set_device(env.device)

    trajectories = [result for _, result in sorted(indexed, key=lambda item: item[0])]
    return _pack_layer_trajectories(env, trajectories)


def collect_group(
    env,
    policy: PositiveSoftplusPolicy,
    group_size: int,
    trajectory_contexts: list[dict] | None = None,
    *,
    seed: int = 0,
    iteration: int = 0,
) -> dict:
    if env.merge_granularity == "global":
        return collect_global_group(env, policy, group_size)
    return collect_layer_group(
        env,
        policy,
        group_size,
        trajectory_contexts=trajectory_contexts,
        seed=seed,
        iteration=iteration,
    )


def summarize_best(group: dict, best_sample: dict, env, args) -> dict:
    candidate_count = min(int(getattr(args, "selection_candidates", 1)), len(group["results"]))
    top_indices = torch.topk(group["rewards"], k=candidate_count).indices.detach().cpu().tolist()
    fixed_results = []
    for rank, best_index in enumerate(top_indices):
        train_result = group["results"][int(best_index)]
        result = evaluate_coefficients_for_selection(env, args, train_result["expanded_coefficients"])
        result["selection_rank"] = int(rank)
        result["selection_group_index"] = int(best_index)
        result["train_objective"] = float(train_result["objective"])
        result["train_mean_retention"] = float(train_result["reward_stats"]["mean_retention"])
        result["train_scores"] = train_result["scores"]
        result["train_reward_stats"] = train_result["reward_stats"]
        result["selection_reward_pool_position"] = fixed_selection_position(args)
        fixed_results.append(result)
    result = max(fixed_results, key=lambda item: item["objective"])
    if result["objective"] > best_sample["objective"]:
        best_sample.update(
            selected=result["selected"],
            coefficients=result["coefficients"],
            expanded_coefficients=result["expanded_coefficients"],
            objective=float(result["objective"]),
            mean_accuracy=float(result["average"]),
            mean_retention=float(result["reward_stats"]["mean_retention"]),
            scores=result["scores"],
            reward_stats=result["reward_stats"],
            train_objective=result["train_objective"],
            train_mean_retention=result["train_mean_retention"],
            train_scores=result["train_scores"],
            train_reward_stats=result["train_reward_stats"],
            selection_reward_pool_position=result["selection_reward_pool_position"],
            selection_rank=result["selection_rank"],
            selection_group_index=result["selection_group_index"],
        )
    return result


def train(args, env, policy: PositiveSoftplusPolicy, optimizer: torch.optim.Optimizer, trajectory_contexts: list[dict] | None = None):
    best_sample = {"objective": float("-inf"), "mean_retention": float("-inf")}
    update_history = []
    episode_history = []
    progress = tqdm(total=args.iterations, desc="RL-MMCR-GRPO")

    for iteration in range(args.iterations):
        group = collect_group(
            env,
            policy,
            args.group_size,
            trajectory_contexts=trajectory_contexts,
            seed=args.seed,
            iteration=iteration,
        )
        trajectory_advantages = compute_advantages(group["rewards"], args.advantage_mode)
        advantages = trajectory_advantages
        if "trajectory_lengths" in group:
            advantages = torch.repeat_interleave(
                trajectory_advantages,
                torch.tensor(group["trajectory_lengths"], dtype=torch.long, device=env.device),
            )
        stats = update_policy(
            policy,
            optimizer,
            group["states"],
            group["actions"],
            group["raw_actions"],
            group["old_log_probs"],
            group["old_mean"],
            group["old_log_std"],
            advantages,
            clip_eps=args.clip_eps,
            entropy_coef=args.entropy_coef,
            grpo_epochs=args.grpo_epochs,
            target_kl=args.target_kl,
        )
        group_best = summarize_best(group, best_sample, env, args)

        episode_history.append({
            "episode": iteration,
            "sample_average": float(group_best["average"]),
            "sample_retention": float(group_best["reward_stats"]["mean_retention"]),
            "sample_objective": float(group_best["objective"]),
            "best_objective": float(best_sample["objective"]),
            "best_retention": float(best_sample["mean_retention"]),
            "reward_sum": float(group["rewards"].mean().item()),
            "final_scores": group_best["scores"],
            "reward_stats": group_best["reward_stats"],
        })

        history_row = {
            "update": iteration + 1,
            "episode": iteration,
            "episodes_completed": iteration + 1,
            "group_reward_mean": float(group["rewards"].mean().item()),
            "group_reward_std": float(group["rewards"].std(unbiased=False).item()),
            "group_objective_mean": float(sum(group["objectives"]) / len(group["objectives"])),
            "group_retention_mean": float(sum(group["retentions"]) / len(group["retentions"])),
            "trajectory_lengths": group.get("trajectory_lengths", [1 for _ in group["results"]]),
            "sample_average": float(group_best["average"]),
            "sample_retention": float(group_best["reward_stats"]["mean_retention"]),
            "sample_objective": float(group_best["objective"]),
            "best_objective": float(best_sample["objective"]),
            "best_retention": float(best_sample["mean_retention"]),
            "reward_sum": float(group["rewards"].mean().item()),
            "deterministic_average": None,
            "deterministic_retention": None,
            "deterministic_objective": None,
            "deterministic_scores": None,
            "value_loss": None,
            "entropy_bonus": float(-args.entropy_coef * stats.entropy),
            **stats.__dict__,
        }

        should_log = (iteration + 1) % args.log_every == 0 or iteration + 1 == args.iterations
        if should_log:
            deterministic = deterministic_selection_result(env, args, policy)
            deterministic_retention = deterministic["infos"][-1]["reward_stats"]["mean_retention"]
            history_row.update(
                deterministic_average=float(deterministic["average"]),
                deterministic_retention=float(deterministic_retention),
                deterministic_objective=float(deterministic["objective"]),
                deterministic_scores=deterministic["scores"],
            )
            progress.write(
                f"iters={iteration + 1} retention group_best={format_retention(history_row['sample_retention'])} "
                f"best={format_retention(best_sample['mean_retention'])} "
                f"deterministic={format_retention(deterministic_retention)} "
                f"{format_scores(group_best['scores'], args.datasets)} "
                f"reward_mean={history_row['group_reward_mean']:.4f} loss={stats.loss:.4f} kl={stats.approx_kl:.4f}"
            )

        update_history.append(history_row)
        progress.update(1)
        progress.set_postfix(
            group_best=format_retention(history_row["sample_retention"]),
            best=format_retention(best_sample["mean_retention"]),
            reward=f"{history_row['group_reward_mean']:.4f}",
            kl=f"{stats.approx_kl:.4f}",
        )

    progress.close()
    return best_sample, update_history, episode_history


def export_results(args, env, policy: PositiveSoftplusPolicy, best_sample: dict, history: list[dict], episodes: list[dict]):
    output_dir = Path(args.output_dir)
    final_policy = deterministic_selection_result(env, args, policy)
    final_coefficients = torch.tensor(final_policy["coefficients"], dtype=torch.float32)
    best_coefficients = torch.tensor(best_sample["coefficients"], dtype=torch.float32)

    if args.export_policy == "best":
        export_coefficients = best_coefficients
        exported_policy = {"type": "best_group_sample", **{key: best_sample[key] for key in (
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
    plot_path = plotting.plot_training_curves(
        history,
        episodes,
        output_dir / "training_curves.png",
        loss_keys=(
            ("loss", "Total Loss"),
            ("policy_loss", "Policy Loss"),
            ("entropy_bonus", "Entropy Bonus"),
        ),
    )
    reward_plot_path = plotting.plot_reward_curves(episodes, output_dir / "reward_curves.png")

    payload = {
        "config": vars(args),
        "source_baseline_scores": env.source_baseline_scores,
        "state_dim": env.state_dim,
        "num_models": env.num_models,
        "num_layers": env.num_layers,
        "layer_names": env.layer_names,
        "action_type": f"{args.merge_granularity}_grpo_rloo_positive_softplus_coefficients",
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

    env = ppo_train.build_environment(args, device)
    policy = PositiveSoftplusPolicy(
        env.state_dim,
        env.num_models,
        hidden_dim=args.policy_hidden_dim,
        log_std_min=args.log_std_min,
        log_std_max=args.log_std_max,
        initial_coefficient=args.coefficient_init,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    trajectory_contexts = build_trajectory_contexts(args, env, policy)
    if trajectory_contexts:
        devices = ", ".join(str(context["device"]) for context in trajectory_contexts)
        print(f"Using trajectory devices: {devices}")

    best_sample, history, episode_history = train(args, env, policy, optimizer, trajectory_contexts)
    encoder_path, plot_path, reward_plot_path = export_results(args, env, policy, best_sample, history, episode_history)
    print(f"Saved GRPO/RLOO results to {output_dir / 'results.json'}")
    print(f"Saved merged encoder to {encoder_path}")
    print(f"Saved training curves to {plot_path}")
    print(f"Saved reward curves to {reward_plot_path}")


if __name__ == "__main__":
    main()
