from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _ewma(values: list[float], alpha: float) -> list[float]:
    if not values:
        return []
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1.0 - alpha) * smoothed[-1])
    return smoothed


def plot_training_curves(update_history: list[dict], episode_history: list[dict], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    if episode_history:
        episodes = [row["episode"] for row in episode_history]
        axes[0].plot(episodes, [row["sample_retention"] for row in episode_history], label="Sample Retention")
        axes[0].plot(episodes, [row["best_retention"] for row in episode_history], label="Best Retention")

    deterministic = [row for row in update_history if row["deterministic_retention"] is not None]
    if deterministic:
        axes[0].plot(
            [row["episodes_completed"] for row in deterministic],
            [row["deterministic_retention"] for row in deterministic],
            label="Deterministic Retention",
        )

    axes[0].set(title="Retention", xlabel="Episode", ylabel="Retention ratio")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if update_history:
        updates = [row["update"] for row in update_history]
        for key, label in (("loss", "Total Loss"), ("policy_loss", "Policy Loss"), ("value_loss", "Value Loss")):
            axes[1].plot(updates, [row[key] for row in update_history], label=label)

    axes[1].set(title="Loss", xlabel="Update", ylabel="Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_reward_curves(episode_history: list[dict], output_path: Path | str, ewma_alpha: float = 0.1) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.8), constrained_layout=True)

    if episode_history:
        episodes = [row["episode"] for row in episode_history]
        rewards = [row["reward_sum"] for row in episode_history]
        ax.plot(episodes, rewards, label="Episode Reward", linewidth=1.0, alpha=0.55)
        ax.plot(episodes, _ewma(rewards, ewma_alpha), label=f"EWMA Reward (alpha={ewma_alpha:g})", linewidth=2.0)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set(title="Episode Reward", xlabel="Episode", ylabel="Reward sum")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
