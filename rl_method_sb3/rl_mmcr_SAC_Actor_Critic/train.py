from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from rl_method_sb3.env import SB3MMCREnv
from .cli import parse_args, validate_args

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as exc:  # pragma: no cover
    raise ImportError("rl_method_sb3 SAC requires stable-baselines3. Install it with `pip install stable-baselines3`.") from exc

SB3_SAC_NOTES = [
    "This method uses Stable-Baselines3 SAC with a bounded tanh-squashed Gaussian policy over Box(0, action_max).",
    "SB3 SAC uses --lr as the optimizer learning rate for policy, critics, and entropy coefficient optimizer.",
    "SB3 --learning-starts controls when gradient updates begin; before that, SB3 samples random actions from the Box action space.",
    "SB3 --gradient-steps controls the number of gradient updates per train_freq interval.",
]


def coefficients_to_dict(*args, **kwargs):
    merge = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.merge")
    return merge.coefficients_to_dict(*args, **kwargs)


def plot_training_curves(*args, **kwargs):
    plotting = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.plotting")
    return plotting.plot_training_curves(*args, **kwargs)


def plot_reward_curves(*args, **kwargs):
    plotting = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.plotting")
    return plotting.plot_reward_curves(*args, **kwargs)


def format_retention(value: float) -> str:
    return f"{value:.4f}x"


def format_scores(scores: dict[str, float], datasets: list[str]) -> str:
    parts = [f"{dataset}={scores[dataset] * 100:.1f}%" for dataset in datasets if dataset in scores]
    return " acc={" + ", ".join(parts) + "}" if parts else ""


def logger_value(model: Any, *keys: str, default: float = 0.0) -> float:
    values = getattr(model.logger, "name_to_value", {})
    for key in keys:
        if key in values:
            value = values[key]
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
    return float(default)


def summarize_episode(mmcr_env: Any, episode: dict, episode_index: int, best_sample: dict) -> dict:
    reward_stats = episode["reward_stats"]
    objective = float(episode["objective"])
    coefficients = torch.tensor(episode["coefficients"], dtype=torch.float32)
    expanded = mmcr_env.expand_coefficients(coefficients)

    if objective > best_sample["objective"]:
        best_sample.update(
            selected=episode["selected"],
            coefficients=episode["coefficients"],
            expanded_coefficients=expanded.tolist(),
            objective=objective,
            mean_accuracy=float(episode["average"]),
            mean_retention=float(reward_stats["mean_retention"]),
            scores=episode["scores"],
            reward_stats=reward_stats,
            coefficients_by_layer=coefficients_to_dict(
                expanded,
                mmcr_env.layer_names,
                mmcr_env.layered_task_vectors.task_names,
            ),
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


class EpisodeRecorderCallback(BaseCallback):
    def __init__(self, args, sb3_env: SB3MMCREnv, progress: tqdm):
        super().__init__()
        self.args = args
        self.sb3_env = sb3_env
        self.progress = progress
        self.best_sample = {"objective": float("-inf"), "mean_retention": float("-inf")}
        self.update_history: list[dict] = []
        self.episode_history: list[dict] = []
        self._seen_episodes = 0

    def _on_step(self) -> bool:
        while self._seen_episodes < len(self.sb3_env.episode_records):
            episode = self.sb3_env.episode_records[self._seen_episodes]
            row = summarize_episode(self.sb3_env.mmcr_env, episode, self._seen_episodes, self.best_sample)
            self.episode_history.append(row)

            actor_loss = logger_value(self.model, "train/actor_loss", default=0.0)
            critic_loss = logger_value(self.model, "train/critic_loss", default=0.0)
            ent_coef_loss = logger_value(self.model, "train/ent_coef_loss", default=0.0)
            ent_coef = logger_value(self.model, "train/ent_coef", default=initial_ent_coef(self.args.ent_coef))
            steps = len(episode["rewards"])
            past_warmup = self.num_timesteps >= self.args.learning_starts
            updates = self.args.gradient_steps * steps if past_warmup else 0
            history_row = {
                "update": len(self.update_history) + 1,
                "episode": self._seen_episodes,
                "episodes_completed": self._seen_episodes + 1,
                "total_steps": self.num_timesteps,
                "replay_size": min(self.num_timesteps, self.args.buffer_size),
                "updates": updates,
                "actor_updates": updates,
                "sample_average": row["sample_average"],
                "sample_retention": row["sample_retention"],
                "sample_objective": row["sample_objective"],
                "best_objective": row["best_objective"],
                "best_retention": row["best_retention"],
                "reward_sum": row["reward_sum"],
                "deterministic_average": None,
                "deterministic_retention": None,
                "deterministic_objective": None,
                "deterministic_scores": None,
                "loss": actor_loss + critic_loss,
                "policy_loss": actor_loss,
                "value_loss": critic_loss,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "alpha_loss": ent_coef_loss,
                "alpha": ent_coef,
                "q_mean": 0.0,
                "target_q_mean": 0.0,
                "log_prob_mean": 0.0,
                "actor_updated": updates > 0,
                "action_anchor_loss": 0.0,
                "cql_loss": 0.0,
                "bellman_loss": critic_loss,
            }
            self.update_history.append(history_row)
            self._seen_episodes += 1
            self.progress.update(1)
            self.progress.set_postfix(
                sample=format_retention(row["sample_retention"]),
                best=format_retention(row["best_retention"]),
                reward=f"{row['reward_sum']:.4f}",
                alpha=f"{ent_coef:.4f}",
            )

            if self._seen_episodes % self.args.log_every == 0 or self._seen_episodes == self.args.episodes:
                self.progress.write(
                    f"episodes={self._seen_episodes} retention sample={format_retention(row['sample_retention'])} "
                    f"best={format_retention(row['best_retention'])} "
                    f"{format_scores(episode['scores'], self.args.datasets)} "
                    f"reward={row['reward_sum']:.4f} vloss={critic_loss:.4f} "
                    f"ploss={actor_loss:.4f} au={updates} alpha={ent_coef:.4f}"
                )

            if self._seen_episodes >= self.args.episodes:
                return False
        return True


@torch.no_grad()
def deterministic_policy_result(env: Any, model: SAC) -> dict:
    state = env.reset().to(env.device)
    selected_history = []
    coefficient_history = []
    infos = []
    done = False

    while not done:
        obs = state.detach().cpu().numpy().astype(np.float32, copy=False)
        action, _ = model.predict(obs, deterministic=True)
        coefficients = torch.as_tensor(np.asarray(action, dtype=np.float32), dtype=torch.float32)
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


def initial_ent_coef(ent_coef: str | float) -> float:
    if isinstance(ent_coef, str):
        if ent_coef == "auto":
            return 1.0
        if ent_coef.startswith("auto_"):
            return float(ent_coef.split("_", maxsplit=1)[1])
        return float(ent_coef)
    return float(ent_coef)


def build_model(args, sb3_env: SB3MMCREnv, device: torch.device) -> SAC:
    policy_kwargs = {
        "net_arch": {
            "pi": [args.policy_hidden_dim, args.policy_hidden_dim],
            "qf": [args.critic_hidden_dim, args.critic_hidden_dim],
        },
        "log_std_init": args.log_std_init,
    }
    return SAC(
        "MlpPolicy",
        sb3_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        target_entropy=args.target_entropy,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=device,
        verbose=0,
    )


def train(args, sb3_env: SB3MMCREnv, model: SAC):
    episode_len = 1 if sb3_env.mmcr_env.merge_granularity == "global" else sb3_env.mmcr_env.num_layers
    total_timesteps = max(args.episodes * episode_len, 1)
    progress = tqdm(total=args.episodes, desc="RL-MMCR-SB3-SAC")
    callback = EpisodeRecorderCallback(args, sb3_env, progress)
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=True)
    finally:
        progress.close()
    return callback.best_sample, callback.update_history, callback.episode_history


def export_results(args, env: Any, model: SAC, best_sample: dict, history: list[dict], episodes: list[dict]):
    output_dir = Path(args.output_dir)
    final_policy = deterministic_policy_result(env, model)

    if not best_sample["objective"] > float("-inf"):
        terminal_stats = final_policy["infos"][-1]["reward_stats"]
        best_sample = {
            "selected": final_policy["selected"],
            "coefficients": final_policy["coefficients"],
            "expanded_coefficients": final_policy["expanded_coefficients"],
            "objective": float(final_policy["objective"]),
            "mean_accuracy": float(final_policy["average"]),
            "mean_retention": float(terminal_stats["mean_retention"]),
            "scores": final_policy["scores"],
            "reward_stats": terminal_stats,
            "coefficients_by_layer": final_policy["coefficients_by_layer"],
        }

    final_coefficients = torch.tensor(final_policy["coefficients"], dtype=torch.float32)
    best_coefficients = torch.tensor(best_sample["coefficients"], dtype=torch.float32)

    if args.export_policy == "best":
        export_coefficients = best_coefficients
        exported_policy = {"type": "best_sb3_sac_sample", **{key: best_sample[key] for key in (
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
    model_path = output_dir / "sb3_sac_model"
    model.save(model_path)

    if history:
        terminal_stats = final_policy["infos"][-1]["reward_stats"]
        history[-1].update(
            deterministic_average=float(final_policy["average"]),
            deterministic_retention=float(terminal_stats["mean_retention"]),
            deterministic_objective=float(final_policy["objective"]),
            deterministic_scores=final_policy["scores"],
        )

    plot_path = plot_training_curves(history, episodes, output_dir / "training_curves.png")
    reward_plot_path = plot_reward_curves(episodes, output_dir / "reward_curves.png")

    payload = {
        "config": vars(args),
        "source_baseline_scores": env.source_baseline_scores,
        "state_dim": env.state_dim,
        "num_models": env.num_models,
        "num_layers": env.num_layers,
        "layer_names": env.layer_names,
        "action_type": "stable_baselines3_sac_positive_box_coefficients",
        "merge_granularity": args.merge_granularity,
        "action_max": args.action_max,
        "sb3_sac_notes": SB3_SAC_NOTES,
        "exported_policy": exported_policy,
        "final_policy": final_policy,
        "best_sample": best_sample,
        "history": history,
        "episode_history": episodes,
        "encoder_path": str(encoder_path),
        "model_path": str(model_path) + ".zip",
        "plot_path": str(plot_path),
        "reward_plot_path": str(reward_plot_path),
    }

    if not args.skip_final_eval:
        from mmcr.evaluation import evaluate_encoder

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

    from mmcr.utils import write_json

    write_json(output_dir / "results.json", payload)
    return encoder_path, model_path.with_suffix(".zip"), plot_path, reward_plot_path


def main() -> None:
    args = parse_args()
    validate_args(args)
    from mmcr.utils import build_device, seed_everything

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = build_device(args.gpu)
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"Stable-Baselines3 SAC episodes: {args.episodes}")

    ppo_train = importlib.import_module("rl_methods.rl_mmcr_PPO_GAE_Actor-Critic.train")
    env = ppo_train.build_environment(args, device)
    sb3_env = SB3MMCREnv(env, action_max=args.action_max)
    model = build_model(args, sb3_env, device)

    best_sample, history, episode_history = train(args, sb3_env, model)
    encoder_path, model_path, plot_path, reward_plot_path = export_results(args, env, model, best_sample, history, episode_history)
    print(f"Saved SB3 SAC RL-MMCR results to {output_dir / 'results.json'}")
    print(f"Saved merged encoder to {encoder_path}")
    print(f"Saved SB3 SAC model to {model_path}")
    print(f"Saved training curves to {plot_path}")
    print(f"Saved reward curves to {reward_plot_path}")


if __name__ == "__main__":
    main()
