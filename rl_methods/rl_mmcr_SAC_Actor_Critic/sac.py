from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionBatch(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int):
        self.capacity = int(capacity)
        self.states = torch.empty((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.empty((capacity, action_dim), dtype=torch.float32)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32)
        self.next_states = torch.empty((capacity, state_dim), dtype=torch.float32)
        self.dones = torch.empty((capacity, 1), dtype=torch.float32)
        self.position = 0
        self.size = 0

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool) -> None:
        index = self.position
        self.states[index].copy_(state.detach().cpu().float())
        self.actions[index].copy_(action.detach().cpu().float())
        self.rewards[index, 0] = float(reward)
        self.next_states[index].copy_(next_state.detach().cpu().float())
        self.dones[index, 0] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> TransitionBatch:
        indices = torch.randint(self.size, (batch_size,))
        return TransitionBatch(
            self.states[indices].to(device),
            self.actions[indices].to(device),
            self.rewards[indices].to(device),
            self.next_states[indices].to(device),
            self.dones[indices].to(device),
        )


class DirichletActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, min_concentration: float = 0.05):
        super().__init__()
        self.min_concentration = float(min_concentration)
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.concentration = nn.Linear(hidden_dim, action_dim)

    def distribution(self, states: torch.Tensor) -> torch.distributions.Dirichlet:
        hidden = self.backbone(states)
        concentration = F.softplus(self.concentration(hidden)) + self.min_concentration
        return torch.distributions.Dirichlet(concentration)

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        actions = distribution.rsample()
        log_probs = distribution.log_prob(actions).unsqueeze(-1)
        return actions, log_probs

    @torch.no_grad()
    def deterministic(self, states: torch.Tensor) -> torch.Tensor:
        distribution = self.distribution(states)
        concentration = distribution.concentration
        return concentration / concentration.sum(dim=-1, keepdim=True).clamp(min=1e-8)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([states, actions], dim=-1))


@dataclass
class SACStats:
    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float
    q_mean: float
    target_q_mean: float
    log_prob_mean: float


class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        actor_hidden_dim: int,
        critic_hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        alpha: float,
        auto_alpha: bool,
        target_entropy: float | None,
        min_concentration: float,
        device: torch.device,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.auto_alpha = bool(auto_alpha)
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(action_dim)

        self.actor = DirichletActor(state_dim, action_dim, actor_hidden_dim, min_concentration).to(device)
        self.critic1 = Critic(state_dim, action_dim, critic_hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, critic_hidden_dim).to(device)
        self.target_critic1 = Critic(state_dim, action_dim, critic_hidden_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim, critic_hidden_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
        )
        self.log_alpha = torch.tensor(float(alpha)).log().to(device).requires_grad_(True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def sample_action(self, state: torch.Tensor, random: bool = False) -> torch.Tensor:
        if random:
            concentration = torch.ones(self.actor.concentration.out_features, device=self.device)
            return torch.distributions.Dirichlet(concentration).sample()
        action, _ = self.actor.sample(state.unsqueeze(0).to(self.device))
        return action.squeeze(0)

    @torch.no_grad()
    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor.deterministic(state.unsqueeze(0).to(self.device)).squeeze(0)

    def update(self, replay: ReplayBuffer, batch_size: int) -> SACStats:
        batch = replay.sample(batch_size, self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(batch.next_states)
            target_q1 = self.target_critic1(batch.next_states, next_actions)
            target_q2 = self.target_critic2(batch.next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_probs
            target = batch.rewards + (1.0 - batch.dones) * self.gamma * target_q

        current_q1 = self.critic1(batch.states, batch.actions)
        current_q2 = self.critic2(batch.states, batch.actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actions, log_probs = self.actor.sample(batch.states)
        q1 = self.critic1(batch.states, actions)
        q2 = self.critic2(batch.states, actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha.detach() * log_probs - q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.zeros((), device=self.device)

        self._soft_update_targets()
        return SACStats(
            actor_loss=float(actor_loss.item()),
            critic_loss=float(critic_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.detach().item()),
            q_mean=float(q.detach().mean().item()),
            target_q_mean=float(target.detach().mean().item()),
            log_prob_mean=float(log_probs.detach().mean().item()),
        )

    @torch.no_grad()
    def _soft_update_targets(self) -> None:
        for target, source in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target.mul_(1.0 - self.tau).add_(source, alpha=self.tau)
        for target, source in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target.mul_(1.0 - self.tau).add_(source, alpha=self.tau)
