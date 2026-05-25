from __future__ import annotations

import importlib
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


def collect_layer_group(env, policy: PositiveSoftplusPolicy, group_size: int) -> dict:
    trajectories = [evaluate_trajectory(env, policy, deterministic=False) for _ in range(group_size)]
    rewards = torch.tensor([trajectory["reward"] for trajectory in trajectories], dtype=torch.float32, device=env.device)
    objectives = [trajectory["objective"] for trajectory in trajectories]
    retentions = [trajectory["reward_stats"]["mean_retention"] for trajectory in trajectories]
    return {
        "states": torch.cat([trajectory["states"] for trajectory in trajectories], dim=0).detach(),
        "actions": torch.cat([trajectory["actions"] for trajectory in trajectories], dim=0).detach(),
        "raw_actions": torch.cat([trajectory["raw_actions"] for trajectory in trajectories], dim=0).detach(),
        "old_log_probs": torch.cat([trajectory["old_log_probs"] for trajectory in trajectories], dim=0).detach(),
        "old_mean": torch.cat([trajectory["old_mean"] for trajectory in trajectories], dim=0).detach(),
        "old_log_std": torch.cat([trajectory["old_log_std"] for trajectory in trajectories], dim=0).detach(),
        "entropies": torch.cat([trajectory["entropies"] for trajectory in trajectories], dim=0).detach(),
        "trajectory_lengths": [int(trajectory["states"].shape[0]) for trajectory in trajectories],
        "rewards": rewards,
        "objectives": objectives,
        "retentions": retentions,
        "results": trajectories,
    }


def collect_group(env, policy: PositiveSoftplusPolicy, group_size: int) -> dict:
    if env.merge_granularity == "global":
        return collect_global_group(env, policy, group_size)
    return collect_layer_group(env, policy, group_size)


def summarize_best(group: dict, best_sample: dict) -> dict:
    best_index = int(torch.argmax(group["rewards"]).item())
    result = group["results"][best_index]
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
        )
    return result


def train(args, env, policy: PositiveSoftplusPolicy, optimizer: torch.optim.Optimizer):
    best_sample = {"objective": float("-inf"), "mean_retention": float("-inf")}
    update_history = []
    episode_history = []
    progress = tqdm(total=args.iterations, desc="RL-MMCR-GRPO")

    for iteration in range(args.iterations):
        group = collect_group(env, policy, args.group_size)
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
        group_best = summarize_best(group, best_sample)

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
            "value_loss": 0.0,
            **stats.__dict__,
        }

        should_log = (iteration + 1) % args.log_every == 0 or iteration + 1 == args.iterations
        if should_log:
            deterministic = deterministic_policy_result(env, policy)
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
    final_policy = deterministic_policy_result(env, policy)
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
    plot_path = plotting.plot_training_curves(history, episodes, output_dir / "training_curves.png")
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

    best_sample, history, episode_history = train(args, env, policy, optimizer)
    encoder_path, plot_path, reward_plot_path = export_results(args, env, policy, best_sample, history, episode_history)
    print(f"Saved GRPO/RLOO results to {output_dir / 'results.json'}")
    print(f"Saved merged encoder to {encoder_path}")
    print(f"Saved training curves to {plot_path}")
    print(f"Saved reward curves to {reward_plot_path}")


if __name__ == "__main__":
    main()
