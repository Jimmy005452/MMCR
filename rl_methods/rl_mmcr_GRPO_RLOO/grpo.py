from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


class PositiveSoftplusPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        initial_coefficient: float = 0.3,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.constant_(self.mean.bias, inverse_softplus(initial_coefficient))
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -2.0)

    def distribution_params(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(states)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_params(states)
        std = log_std.exp()
        raw = mean + std * torch.randn_like(mean)
        actions = torch.nn.functional.softplus(raw)
        log_probs = log_prob_from_raw(raw, mean, log_std)
        entropy = (log_std + 0.5 * (1.0 + math.log(2.0 * math.pi)) + mean).sum(dim=-1)
        return actions, raw, log_probs, entropy, mean, log_std

    def log_prob(self, raw_actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.distribution_params(states)
        return log_prob_from_raw(raw_actions, mean, log_std)

    @torch.no_grad()
    def deterministic(self, states: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution_params(states)
        return torch.nn.functional.softplus(mean)


def log_prob_from_raw(raw: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    std = log_std.exp()
    log_two_pi = math.log(2.0 * math.pi)
    normal_log_prob = -0.5 * (((raw - mean) / std).pow(2) + 2.0 * log_std + log_two_pi)
    log_jacobian = torch.nn.functional.logsigmoid(raw)
    return (normal_log_prob - log_jacobian).sum(dim=-1)


def independent_normal_kl(
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_log_std: torch.Tensor,
) -> torch.Tensor:
    old_var = (2.0 * old_log_std).exp()
    new_var = (2.0 * new_log_std).exp()
    kl = new_log_std - old_log_std + (old_var + (old_mean - new_mean).pow(2)) / (2.0 * new_var) - 0.5
    return kl.sum(dim=-1)


@dataclass
class GRPOStats:
    loss: float
    policy_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    grpo_epochs_ran: int


def compute_advantages(rewards: torch.Tensor, mode: str) -> torch.Tensor:
    rewards = rewards.float()
    if mode == "rloo":
        group_size = rewards.numel()
        leave_one_out = (rewards.sum() - rewards) / max(group_size - 1, 1)
        advantages = rewards - leave_one_out
        return advantages / advantages.std(unbiased=False).clamp(min=1e-8)
    if mode == "zscore":
        return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-8)
    if mode == "rank":
        order = torch.argsort(torch.argsort(rewards))
        ranks = order.float() / max(rewards.numel() - 1, 1)
        return (ranks - 0.5) * 2.0
    raise ValueError(f"Unknown advantage mode: {mode}")


def update_policy(
    policy: PositiveSoftplusPolicy,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    raw_actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_mean: torch.Tensor,
    old_log_std: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float,
    entropy_coef: float,
    grpo_epochs: int,
    target_kl: float,
) -> GRPOStats:
    last = {"loss": None, "policy": None, "entropy": None, "kl": None, "clip": None, "epochs": 0}

    for epoch in range(grpo_epochs):
        mean, log_std = policy.distribution_params(states)
        log_probs = log_prob_from_raw(raw_actions, mean, log_std)
        entropy = (log_std + 0.5 * (1.0 + math.log(2.0 * math.pi)) + mean).sum(dim=-1).mean()
        ratios = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
        policy_loss = -torch.min(ratios * advantages, clipped * advantages).mean()
        loss = policy_loss - entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            new_mean, new_log_std = policy.distribution_params(states)
            updated_log_probs = log_prob_from_raw(raw_actions, new_mean, new_log_std)
            updated_ratios = torch.exp(updated_log_probs - old_log_probs)
            approx_kl = independent_normal_kl(old_mean, old_log_std, new_mean, new_log_std).mean()
            clip_fraction = (updated_ratios.sub(1.0).abs() > clip_eps).float().mean()
        last.update(loss=loss, policy=policy_loss, entropy=entropy, kl=approx_kl, clip=clip_fraction, epochs=epoch + 1)
        if approx_kl.item() > target_kl:
            break

    return GRPOStats(
        loss=float(last["loss"].item()),
        policy_loss=float(last["policy"].item()),
        entropy=float(last["entropy"].item()),
        approx_kl=float(last["kl"].item()),
        clip_fraction=float(last["clip"].item()),
        grpo_epochs_ran=int(last["epochs"]),
    )
