from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirichletPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, min_concentration: float = 0.05):
        super().__init__()
        self.min_concentration = float(min_concentration)
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.concentration = nn.Linear(hidden_dim, action_dim)

    def distribution(self, states: torch.Tensor) -> torch.distributions.Dirichlet:
        hidden = self.backbone(states)
        concentration = F.softplus(self.concentration(hidden)) + self.min_concentration
        return torch.distributions.Dirichlet(concentration)

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        actions = distribution.rsample()
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return actions, log_probs, entropy

    @torch.no_grad()
    def deterministic(self, states: torch.Tensor) -> torch.Tensor:
        concentration = self.distribution(states).concentration
        return concentration / concentration.sum(dim=-1, keepdim=True).clamp(min=1e-8)


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
    policy: DirichletPolicy,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_eps: float,
    entropy_coef: float,
    grpo_epochs: int,
    target_kl: float,
) -> GRPOStats:
    last = {"loss": None, "policy": None, "entropy": None, "kl": None, "clip": None, "epochs": 0}

    for epoch in range(grpo_epochs):
        distribution = policy.distribution(states)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()
        ratios = torch.exp(log_probs - old_log_probs)
        clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
        policy_loss = -torch.min(ratios * advantages, clipped * advantages).mean()
        loss = policy_loss - entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            approx_kl = (old_log_probs - log_probs).mean()
            clip_fraction = (ratios.sub(1.0).abs() > clip_eps).float().mean()
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
