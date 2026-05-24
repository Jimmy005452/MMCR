from __future__ import annotations

from dataclasses import dataclass
import math
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


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


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


class PositiveSoftplusActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
        initial_coefficient: float = 1.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.constant_(self.mean.bias, inverse_softplus(initial_coefficient))
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -2.0)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(states)
        mean = torch.nan_to_num(self.mean(hidden), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        log_std = torch.nan_to_num(
            self.log_std(hidden),
            nan=-2.0,
            posinf=self.log_std_max,
            neginf=self.log_std_min,
        ).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(states)
        std = log_std.exp()
        raw = mean + std * torch.randn_like(mean)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        actions = F.softplus(raw).clamp(min=0.0, max=10.0)
        log_two_pi = torch.tensor(math.log(2.0 * math.pi), device=states.device)
        normal_log_prob = -0.5 * (((raw - mean) / std).pow(2) + 2.0 * log_std + log_two_pi)
        log_jacobian = F.logsigmoid(raw)
        log_probs = (normal_log_prob - log_jacobian).sum(dim=-1, keepdim=True)
        log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        return actions, log_probs

    @torch.no_grad()
    def deterministic(self, states: torch.Tensor) -> torch.Tensor:
        mean, _ = self(states)
        return F.softplus(mean).clamp(min=0.0, max=10.0)


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
        values = self.net(torch.cat([states, actions], dim=-1))
        return torch.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)


@dataclass
class SACStats:
    actor_loss: float
    critic_loss: float
    alpha_loss: float
    alpha: float
    q_mean: float
    target_q_mean: float
    log_prob_mean: float
    actor_updated: bool = False
    action_anchor_loss: float = 0.0
    cql_loss: float = 0.0
    bellman_loss: float = 0.0


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
        log_std_min: float,
        log_std_max: float,
        initial_coefficient: float,
        action_anchor_coef: float,
        cql_coef: float,
        device: torch.device,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.auto_alpha = bool(auto_alpha)
        self.target_entropy = float(target_entropy) if target_entropy is not None else -float(action_dim)
        self.action_anchor_coef = float(action_anchor_coef)
        self.cql_coef = float(cql_coef)
        self.initial_action = torch.full((1, action_dim), float(initial_coefficient), device=device)

        self.actor = PositiveSoftplusActor(
            state_dim,
            action_dim,
            actor_hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            initial_coefficient=initial_coefficient,
        ).to(device)
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
            return torch.empty(self.actor.action_dim, device=self.device).uniform_(0.0, 1.0)
        action, _ = self.actor.sample(state.unsqueeze(0).to(self.device))
        return action.squeeze(0).clamp(min=0.0, max=10.0)

    @torch.no_grad()
    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor.deterministic(state.unsqueeze(0).to(self.device)).squeeze(0).clamp(min=0.0, max=10.0)

    def update(self, replay: ReplayBuffer, batch_size: int, update_actor: bool = True) -> SACStats:
        batch = replay.sample(batch_size, self.device)
        batch = TransitionBatch(
            batch.states,
            torch.nan_to_num(batch.actions, nan=0.0, posinf=10.0, neginf=0.0).clamp(min=0.0, max=10.0),
            torch.nan_to_num(batch.rewards, nan=0.0, posinf=10.0, neginf=-10.0).clamp(min=-10.0, max=10.0),
            batch.next_states,
            batch.dones,
        )

        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(batch.next_states)
            next_actions = torch.nan_to_num(next_actions, nan=0.0, posinf=10.0, neginf=0.0).clamp(min=0.0, max=10.0)
            next_log_probs = torch.nan_to_num(next_log_probs, nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)
            target_q1 = self.target_critic1(batch.next_states, next_actions)
            target_q2 = self.target_critic2(batch.next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_probs
            target_q = torch.nan_to_num(target_q, nan=0.0, posinf=10.0, neginf=-10.0).clamp(min=-10.0, max=10.0)
            target = batch.rewards + (1.0 - batch.dones) * self.gamma * target_q
            target = torch.nan_to_num(target, nan=0.0, posinf=10.0, neginf=-10.0).clamp(min=-10.0, max=10.0)

        current_q1 = self.critic1(batch.states, batch.actions)
        current_q2 = self.critic2(batch.states, batch.actions)
        bellman_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)
        cql_loss = torch.zeros((), device=self.device)
        if self.cql_coef > 0.0:
            with torch.no_grad():
                cql_actions, _ = self.actor.sample(batch.states)
                cql_actions = torch.nan_to_num(cql_actions, nan=0.0, posinf=10.0, neginf=0.0).clamp(min=0.0, max=10.0)
            cql_q1 = self.critic1(batch.states, cql_actions)
            cql_q2 = self.critic2(batch.states, cql_actions)
            cql_loss = (
                F.softplus(cql_q1 - current_q1.detach()).mean()
                + F.softplus(cql_q2 - current_q2.detach()).mean()
            )
        critic_loss = bellman_loss + self.cql_coef * cql_loss
        if not torch.isfinite(critic_loss):
            return SACStats(
                0.0,
                float("nan"),
                0.0,
                float(self.alpha.detach().item()),
                0.0,
                float(target.mean().item()),
                0.0,
                False,
                0.0,
                float(cql_loss.detach().item()),
                float(bellman_loss.detach().item()) if torch.isfinite(bellman_loss) else float("nan"),
            )
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(self.critic1.parameters()) + list(self.critic2.parameters()), max_norm=5.0)
        self.critic_optimizer.step()

        actions, log_probs = self.actor.sample(batch.states)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=10.0, neginf=0.0).clamp(min=0.0, max=10.0)
        log_probs = torch.nan_to_num(log_probs, nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)
        q1 = self.critic1(batch.states, actions)
        q2 = self.critic2(batch.states, actions)
        q = torch.nan_to_num(torch.min(q1, q2), nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)
        action_anchor_loss = (actions - self.initial_action.expand_as(actions)).pow(2).mean()
        actor_loss = (self.alpha.detach() * log_probs - q).mean() + self.action_anchor_coef * action_anchor_loss
        alpha_loss = torch.zeros((), device=self.device)
        actor_updated = False

        if update_actor:
            if not torch.isfinite(actor_loss):
                return SACStats(
                    0.0,
                    float(critic_loss.item()),
                    0.0,
                    float(self.alpha.detach().item()),
                    float(q.mean().item()),
                    float(target.mean().item()),
                    float(log_probs.mean().item()),
                    False,
                    float(action_anchor_loss.detach().item()),
                    float(cql_loss.detach().item()),
                    float(bellman_loss.detach().item()),
                )
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
            self.actor_optimizer.step()
            actor_updated = True

            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()

        self._soft_update_targets()
        return SACStats(
            actor_loss=float(actor_loss.item()) if actor_updated else 0.0,
            critic_loss=float(critic_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.detach().item()),
            q_mean=float(q.detach().mean().item()),
            target_q_mean=float(target.detach().mean().item()),
            log_prob_mean=float(log_probs.detach().mean().item()),
            actor_updated=actor_updated,
            action_anchor_loss=float(action_anchor_loss.detach().item()),
            cql_loss=float(cql_loss.detach().item()),
            bellman_loss=float(bellman_loss.detach().item()),
        )

    @torch.no_grad()
    def _soft_update_targets(self) -> None:
        for target, source in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target.mul_(1.0 - self.tau).add_(source, alpha=self.tau)
        for target, source in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target.mul_(1.0 - self.tau).add_(source, alpha=self.tau)
