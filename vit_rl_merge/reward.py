import torch


def softmax_entropy(logits: torch.Tensor):
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def capability_retention_reward(
    accuracies,
    source_accuracies,
    worst_weight: float = 0.5,
    std_weight: float = 0.25,
    reward_mode: str = "balanced",
    reward_scale: float = 1.0,
):
    retentions = {}
    for dataset, accuracy in accuracies.items():
        reference = max(source_accuracies[dataset], 1e-6)
        retentions[dataset] = accuracy / reference

    values = torch.tensor(list(retentions.values()), dtype=torch.float32)
    mean_retention = values.mean()
    worst_retention = values.min()
    std_retention = values.std(unbiased=False)

    if reward_mode == "balanced":
        performance_reward = mean_retention + worst_weight * worst_retention - std_weight * std_retention
    elif reward_mode == "worst":
        performance_reward = worst_retention
    elif reward_mode == "mean_worst":
        performance_reward = 0.5 * mean_retention + 0.5 * worst_retention - std_weight * std_retention
    elif reward_mode == "harmonic":
        performance_reward = values.numel() / torch.clamp((1.0 / values.clamp(min=1e-6)).sum(), min=1e-6)
        performance_reward = performance_reward - std_weight * std_retention
    else:
        raise ValueError("reward_mode must be one of: balanced, worst, mean_worst, harmonic")

    raw_reward = performance_reward
    reward = raw_reward * reward_scale
    stats = {
        "reward_mode": reward_mode,
        "reward_scale": reward_scale,
        "performance_reward": performance_reward.item(),
        "raw_reward": raw_reward.item(),
        "scaled_reward": reward.item(),
        "mean_retention": mean_retention.item(),
        "worst_retention": worst_retention.item(),
        "std_retention": std_retention.item(),
    }
    return reward.item(), retentions, stats


def entropy_reward(entropies, reward_scale: float = 1.0):
    values = torch.tensor(list(entropies.values()), dtype=torch.float32)
    mean_entropy = values.mean()
    reward = -mean_entropy * reward_scale
    stats = {
        "reward_mode": "entropy",
        "reward_scale": reward_scale,
        "mean_entropy": mean_entropy.item(),
        "raw_reward": (-mean_entropy).item(),
        "scaled_reward": reward.item(),
    }
    return reward.item(), stats


def format_percent_dict(values):
    return "{" + ", ".join(f"{key}: {value * 100:.2f}%" for key, value in values.items()) + "}"


def format_float_dict(values):
    return "{" + ", ".join(f"{key}: {value:.4f}" for key, value in values.items()) + "}"
