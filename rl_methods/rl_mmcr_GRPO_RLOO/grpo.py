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
    if mode in {"rloo", "rloo_no_std"}:
        group_size = rewards.numel()
        leave_one_out = (rewards.sum() - rewards) / max(group_size - 1, 1)
        advantages = rewards - leave_one_out
        if mode == "rloo_no_std":
            return advantages
        return advantages / advantages.std(unbiased=False).clamp(min=1e-8)
    if mode == "zscore":
        return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp(min=1e-8)
    if mode == "rank":
        order = torch.argsort(torch.argsort(rewards))
        ranks = order.float() / max(rewards.numel() - 1, 1)
        return (ranks - 0.5) * 2.0
    raise ValueError(f"Unknown advantage mode: {mode}")


def _segment_sum(values: torch.Tensor, lengths: list[int] | tuple[int, ...]) -> torch.Tensor:
    pieces = []
    offset = 0
    for length in lengths:
        next_offset = offset + int(length)
        pieces.append(values[offset:next_offset].sum())
        offset = next_offset
    if offset != int(values.shape[0]):
        raise ValueError(f"Trajectory lengths sum to {offset}, but tensor has {values.shape[0]} rows.")
    return torch.stack(pieces)


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
    policy_loss_mode: str = "step",
    trajectory_lengths: list[int] | None = None,
    clip_eps_low: float | None = None,
    clip_eps_high: float | None = None,
) -> GRPOStats:
    if policy_loss_mode not in {"step", "trajectory"}:
        raise ValueError("policy_loss_mode must be either 'step' or 'trajectory'.")
    lower_clip = 1.0 - float(clip_eps if clip_eps_low is None else clip_eps_low)
    upper_clip = 1.0 + float(clip_eps if clip_eps_high is None else clip_eps_high)
    if lower_clip <= 0 or upper_clip <= lower_clip:
        raise ValueError("Invalid clipping range. Expected 0 < 1 - clip_eps_low < 1 + clip_eps_high.")
    if policy_loss_mode == "trajectory":
        if trajectory_lengths is None:
            trajectory_lengths = [1 for _ in range(int(states.shape[0]))]
        if len(advantages) != len(trajectory_lengths):
            raise ValueError("Trajectory-level policy loss expects one advantage per trajectory.")
    elif len(advantages) != int(states.shape[0]):
        raise ValueError("Step-level policy loss expects one advantage per action step.")

    last = {"loss": None, "policy": None, "entropy": None, "kl": None, "clip": None, "epochs": 0}

    for epoch in range(grpo_epochs):
        mean, log_std = policy.distribution_params(states)
        log_probs = log_prob_from_raw(raw_actions, mean, log_std)
        entropy = (log_std + 0.5 * (1.0 + math.log(2.0 * math.pi)) + mean).sum(dim=-1).mean()

        if policy_loss_mode == "trajectory":
            new_traj_log_probs = _segment_sum(log_probs, trajectory_lengths)
            old_traj_log_probs = _segment_sum(old_log_probs, trajectory_lengths)
            ratios = torch.exp(new_traj_log_probs - old_traj_log_probs)
        else:
            ratios = torch.exp(log_probs - old_log_probs)

        clipped = torch.clamp(ratios, lower_clip, upper_clip)
        policy_loss = -torch.min(ratios * advantages, clipped * advantages).mean()
        loss = policy_loss - entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            new_mean, new_log_std = policy.distribution_params(states)
            updated_log_probs = log_prob_from_raw(raw_actions, new_mean, new_log_std)
            step_kl = independent_normal_kl(old_mean, old_log_std, new_mean, new_log_std)
            if policy_loss_mode == "trajectory":
                updated_traj_log_probs = _segment_sum(updated_log_probs, trajectory_lengths)
                updated_ratios = torch.exp(updated_traj_log_probs - old_traj_log_probs)
                approx_kl = _segment_sum(step_kl, trajectory_lengths).mean()
            else:
                updated_ratios = torch.exp(updated_log_probs - old_log_probs)
                approx_kl = step_kl.mean()
            clip_fraction = ((updated_ratios < lower_clip) | (updated_ratios > upper_clip)).float().mean()
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
