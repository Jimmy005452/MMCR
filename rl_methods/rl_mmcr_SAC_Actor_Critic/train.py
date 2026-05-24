from __future__ import annotations

import importlib
from pathlib import Path

import torch
from tqdm import tqdm

from mmcr.evaluation import evaluate_encoder
from mmcr.utils import build_device, seed_everything, write_json

from .cli import parse_args, validate_args
from .sac import ReplayBuffer, SACAgent, SACStats

ppo_train = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.train")
plotting = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.plotting")
coefficients_to_dict = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.merge").coefficients_to_dict


def format_retention(value: float) -> str:
    return f"{value:.4f}x"


def format_scores(scores: dict[str, float], datasets: list[str]) -> str:
    parts = [f"{dataset}={scores[dataset] * 100:.1f}%" for dataset in datasets if dataset in scores]
    return " acc={" + ", ".join(parts) + "}" if parts else ""


@torch.no_grad()
def deterministic_policy_result(env, agent: SACAgent) -> dict:
    state = env.reset().to(env.device)
    selected_history = []
    coefficient_history = []
    infos = []
    done = False

    while not done:
        coefficients = agent.deterministic_action(state)
        selected = torch.ones_like(coefficients)
        state, _, done, info = env.step(selected, coefficients)
        selected_history.append(selected.detach().cpu().tolist())
        coefficient_history.append(coefficients.detach().cpu().tolist())
        infos.append(info)
        state = state.to(env.device)

    final_coefficients = torch.tensor(coefficient_history, dtype=torch.float32)
    expanded_coefficients = env.expand_coefficients(final_coefficients)
    terminal_info = infos[-1]
    return {
        "selected": selected_history,
        "coefficients": coefficient_history,
        "expanded_coefficients": expanded_coefficients.tolist(),
        "average": float(env.previous_average),
        "objective": float(terminal_info["objective"]),
        "scores": dict(env.previous_scores),
        "infos": infos,
        "coefficients_by_layer": coefficients_to_dict(
            expanded_coefficients,
            env.layer_names,
            env.layered_task_vectors.task_names,
        ),
    }


def zero_stats(alpha: float) -> SACStats:
    return SACStats(
        actor_loss=0.0,
        critic_loss=0.0,
        alpha_loss=0.0,
        alpha=float(alpha),
        q_mean=0.0,
        target_q_mean=0.0,
        log_prob_mean=0.0,
    )


def collect_episode(env, agent: SACAgent, replay: ReplayBuffer, random_steps_remaining: int) -> tuple[dict, int]:
    state = env.reset().to(env.device)
    selected_history = []
    coefficient_history = []
    rewards = []
    infos = []
    done = False
    steps = 0

    while not done:
        use_random = random_steps_remaining > 0
        coefficients = agent.sample_action(state, random=use_random)
        selected = torch.ones_like(coefficients)
        next_state, reward, done, info = env.step(selected, coefficients)
        next_state = next_state.to(env.device)
        replay.add(state, coefficients, reward, next_state, done)

        selected_history.append(selected.detach().cpu().tolist())
        coefficient_history.append(coefficients.detach().cpu().tolist())
        rewards.append(float(reward))
        infos.append(info)
        state = next_state
        steps += 1
        random_steps_remaining = max(0, random_steps_remaining - 1)

    terminal = infos[-1]
    return {
        "selected": selected_history,
        "coefficients": coefficient_history,
        "rewards": rewards,
        "infos": infos,
        "average": float(terminal["average"]),
        "objective": float(terminal["objective"]),
        "scores": terminal["scores"],
        "reward_stats": terminal["reward_stats"],
        "reward_sum": float(sum(rewards)),
        "steps": steps,
    }, random_steps_remaining


def summarize_episode(episode: dict, episode_index: int, best_sample: dict) -> dict:
    reward_stats = episode["reward_stats"]
    objective = float(episode["objective"])
    if objective > best_sample["objective"]:
        best_sample.update(
            selected=episode["selected"],
            coefficients=episode["coefficients"],
            objective=objective,
            mean_accuracy=float(episode["average"]),
            mean_retention=float(reward_stats["mean_retention"]),
            scores=episode["scores"],
            reward_stats=reward_stats,
        )

    return {
        "episode": episode_index,
        "sample_average": float(episode["average"]),
        "sample_retention": float(reward_stats["mean_retention"]),
        "sample_objective": objective,
        "best_objective": float(best_sample["objective"]),
        "best_retention": float(best_sample["mean_retention"]),
        "reward_sum": float(episode["reward_sum"]),
        "final_scores": episode["scores"],
        "reward_stats": reward_stats,
    }


def train(args, env, agent: SACAgent, replay: ReplayBuffer):
    best_sample = {"objective": float("-inf"), "mean_retention": float("-inf")}
    update_history = []
    episode_history = []
    random_steps_remaining = args.random_steps
    total_steps = 0
    last_stats = zero_stats(args.alpha)
    progress = tqdm(total=args.episodes, desc="RL-MMCR-SAC")

    for episode_index in range(args.episodes):
        episode, random_steps_remaining = collect_episode(env, agent, replay, random_steps_remaining)
        total_steps += int(episode["steps"])
        episode_history.append(summarize_episode(episode, episode_index, best_sample))

        updates = 0
        if replay.size >= args.batch_size:
            for _ in range(args.updates_per_step * int(episode["steps"])):
                last_stats = agent.update(replay, args.batch_size)
                updates += 1

        history_row = {
            "update": episode_index + 1,
            "episode": episode_index,
            "episodes_completed": episode_index + 1,
            "total_steps": total_steps,
            "replay_size": replay.size,
            "updates": updates,
            "sample_average": float(episode["average"]),
            "sample_retention": float(episode["reward_stats"]["mean_retention"]),
            "sample_objective": float(episode["objective"]),
            "best_objective": float(best_sample["objective"]),
            "best_retention": float(best_sample["mean_retention"]),
            "reward_sum": float(episode["reward_sum"]),
            "deterministic_average": None,
            "deterministic_retention": None,
            "deterministic_objective": None,
            "deterministic_scores": None,
            "loss": last_stats.actor_loss + last_stats.critic_loss,
            "policy_loss": last_stats.actor_loss,
            "value_loss": last_stats.critic_loss,
            **last_stats.__dict__,
        }

        should_log = (episode_index + 1) % args.log_every == 0 or episode_index + 1 == args.episodes
        if should_log:
            deterministic = deterministic_policy_result(env, agent)
            deterministic_retention = deterministic["infos"][-1]["reward_stats"]["mean_retention"]
            history_row.update(
                deterministic_average=float(deterministic["average"]),
                deterministic_retention=float(deterministic_retention),
                deterministic_objective=float(deterministic["objective"]),
                deterministic_scores=deterministic["scores"],
            )
            progress.write(
                f"episodes={episode_index + 1} retention sample={format_retention(history_row['sample_retention'])} "
                f"best={format_retention(best_sample['mean_retention'])} "
                f"deterministic={format_retention(deterministic_retention)} "
                f"{format_scores(episode['scores'], args.datasets)} "
                f"reward={history_row['reward_sum']:.4f} q={last_stats.q_mean:.4f} alpha={last_stats.alpha:.4f}"
            )

        update_history.append(history_row)
        progress.update(1)
        progress.set_postfix(
            sample=format_retention(history_row["sample_retention"]),
            best=format_retention(best_sample["mean_retention"]),
            reward=f"{history_row['reward_sum']:.4f}",
            q=f"{last_stats.q_mean:.4f}",
            alpha=f"{last_stats.alpha:.4f}",
        )

    progress.close()
    return best_sample, update_history, episode_history


def export_results(args, env, agent: SACAgent, best_sample: dict, history: list[dict], episodes: list[dict]):
    output_dir = Path(args.output_dir)
    final_policy = deterministic_policy_result(env, agent)
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
    plot_path = plotting.plot_training_curves(history, episodes, output_dir / "training_curves.png")
    reward_plot_path = plotting.plot_reward_curves(episodes, output_dir / "reward_curves.png")

    payload = {
        "config": vars(args),
        "source_baseline_scores": env.source_baseline_scores,
        "state_dim": env.state_dim,
        "num_models": env.num_models,
        "num_layers": env.num_layers,
        "layer_names": env.layer_names,
        "action_type": "sac_dirichlet_continuous_coefficients",
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
    agent = SACAgent(
        env.state_dim,
        env.num_models,
        actor_hidden_dim=args.policy_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        actor_lr=args.lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
        target_entropy=args.target_entropy,
        min_concentration=args.min_concentration,
        device=device,
    )
    replay = ReplayBuffer(env.state_dim, env.num_models, args.replay_size)

    best_sample, history, episode_history = train(args, env, agent, replay)
    encoder_path, plot_path, reward_plot_path = export_results(args, env, agent, best_sample, history, episode_history)
    print(f"Saved SAC RL-MMCR results to {output_dir / 'results.json'}")
    print(f"Saved merged encoder to {encoder_path}")
    print(f"Saved training curves to {plot_path}")
    print(f"Saved reward curves to {reward_plot_path}")


if __name__ == "__main__":
    main()
