from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


COEFFICIENT_MODES = {"positive"}


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


class PositiveActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_models: int,
        hidden_dim: int = 64,
        coefficient_mode: str = "positive",
        coefficient_init: float = 0.3,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
    ):
        super().__init__()
        if coefficient_mode not in COEFFICIENT_MODES:
            raise ValueError(f"coefficient_mode must be one of: {sorted(COEFFICIENT_MODES)}")

        self.num_models = int(num_models)
        self.coefficient_mode = coefficient_mode
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, num_models)
        self.log_std = nn.Linear(hidden_dim, num_models)
        self.value = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.mean.weight)
        nn.init.constant_(self.mean.bias, inverse_softplus(coefficient_init))
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -2.0)

    def distribution_params(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(states)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def value_fn(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(states)
        return self.value(hidden).squeeze(-1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(states)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(self.log_std_min, self.log_std_max)
        values = self.value(hidden).squeeze(-1)
        return mean, log_std, values


def transform_coefficients(raw_weights: torch.Tensor, coefficient_mode: str) -> torch.Tensor:
    if coefficient_mode == "positive":
        return F.softplus(raw_weights)
    raise RuntimeError(f"Unsupported coefficient_mode: {coefficient_mode}")


def log_prob_from_raw(raw_weights: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    std = log_std.exp()
    log_two_pi = math.log(2.0 * math.pi)
    normal_log_prob = -0.5 * (((raw_weights - mean) / std).pow(2) + 2.0 * log_std + log_two_pi)
    log_jacobian = F.logsigmoid(raw_weights)
    return (normal_log_prob - log_jacobian).sum(dim=-1)


def entropy_from_params(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    return (log_std + 0.5 * (1.0 + math.log(2.0 * math.pi)) + mean).sum(dim=-1)


def sample_positive_action(model: PositiveActorCritic, state: torch.Tensor):
    mean, log_std, value = model(state)
    std = log_std.exp()
    raw_weights = mean + std * torch.randn_like(mean)
    coefficients = transform_coefficients(raw_weights, model.coefficient_mode)
    log_prob = log_prob_from_raw(raw_weights, mean, log_std)
    entropy = entropy_from_params(mean, log_std)
    selected = torch.ones_like(coefficients)
    return selected, coefficients, raw_weights, log_prob, entropy, value


def evaluate_positive_action(
    model: PositiveActorCritic,
    states: torch.Tensor,
    raw_weights: torch.Tensor,
):
    mean, log_std, values = model(states)
    log_prob = log_prob_from_raw(raw_weights, mean, log_std)
    entropy = entropy_from_params(mean, log_std)
    return log_prob, entropy, values


@torch.no_grad()
def deterministic_positive_action(model: PositiveActorCritic, state: torch.Tensor):
    mean, _ = model.distribution_params(state)
    selected = torch.ones_like(mean)
    coefficients = transform_coefficients(mean, model.coefficient_mode)
    return selected, coefficients
