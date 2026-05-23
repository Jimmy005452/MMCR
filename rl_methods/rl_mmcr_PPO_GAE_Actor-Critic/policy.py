from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


COEFFICIENT_MODES = {"softmax", "sigmoid", "positive", "unconstrained"}


def _initial_raw_value(coefficient_mode: str, coefficient_init: float) -> float:
    if coefficient_mode == "softmax":
        return 0.0
    if coefficient_mode == "sigmoid":
        value = min(max(coefficient_init, 1e-4), 1.0 - 1e-4)
        return torch.logit(torch.tensor(value)).item()
    if coefficient_mode == "positive":
        value = max(coefficient_init, 1e-4)
        return torch.log(torch.expm1(torch.tensor(value))).item()
    if coefficient_mode == "unconstrained":
        return float(coefficient_init)
    raise ValueError(f"coefficient_mode must be one of: {sorted(COEFFICIENT_MODES)}")


class HybridActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_models: int,
        hidden_dim: int = 64,
        coefficient_mode: str = "softmax",
        coefficient_init: float = 1.0,
    ):
        super().__init__()
        if coefficient_mode not in COEFFICIENT_MODES:
            raise ValueError(f"coefficient_mode must be one of: {sorted(COEFFICIENT_MODES)}")

        self.num_models = num_models
        self.coefficient_mode = coefficient_mode
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gate_logits = nn.Linear(hidden_dim, num_models)
        self.weight_mean = nn.Linear(hidden_dim, num_models)
        self.log_std = nn.Parameter(torch.full((num_models,), -0.5))
        self.value = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.weight_mean.weight)
        nn.init.constant_(self.weight_mean.bias, _initial_raw_value(coefficient_mode, coefficient_init))

    def forward(self, state: torch.Tensor):
        hidden = self.backbone(state)
        return (
            self.gate_logits(hidden),
            self.weight_mean(hidden),
            self.log_std.clamp(min=-4.0, max=1.0),
            self.value(hidden).squeeze(-1),
        )


def transform_coefficients(raw_weights: torch.Tensor, coefficient_mode: str) -> torch.Tensor:
    if coefficient_mode == "softmax":
        return F.softmax(raw_weights, dim=-1)
    if coefficient_mode == "sigmoid":
        return torch.sigmoid(raw_weights)
    if coefficient_mode == "positive":
        return F.softplus(raw_weights)
    if coefficient_mode == "unconstrained":
        return raw_weights
    raise RuntimeError(f"Unsupported coefficient_mode: {coefficient_mode}")


def coefficients_from_action(
    gate_action: torch.Tensor,
    raw_weights: torch.Tensor,
    coefficient_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = gate_action.float()
    if active.sum().item() == 0:
        active = torch.ones_like(active)

    if coefficient_mode == "softmax":
        masked_weights = raw_weights.masked_fill(active == 0, -1e9)
        coefficients = F.softmax(masked_weights, dim=0) * active
        coefficients = coefficients / coefficients.sum().clamp(min=1e-8)
    else:
        coefficients = transform_coefficients(raw_weights, coefficient_mode) * active
    return active, coefficients


def sample_hybrid_action(model: HybridActorCritic, state: torch.Tensor):
    gate_logits, weight_mean, log_std, value = model(state)
    gate_distribution = torch.distributions.Bernoulli(logits=gate_logits)
    weight_distribution = torch.distributions.Normal(weight_mean, log_std.exp())

    gate_action = gate_distribution.sample()
    raw_weights = weight_distribution.rsample()
    active, coefficients = coefficients_from_action(gate_action, raw_weights, model.coefficient_mode)
    log_prob = gate_distribution.log_prob(gate_action).sum() + weight_distribution.log_prob(raw_weights).sum()
    entropy = gate_distribution.entropy().sum() + weight_distribution.entropy().sum()
    return active, coefficients, gate_action, raw_weights, log_prob, entropy, value


def evaluate_hybrid_action(
    model: HybridActorCritic,
    states: torch.Tensor,
    gate_actions: torch.Tensor,
    raw_weights: torch.Tensor,
):
    gate_logits, weight_mean, log_std, values = model(states)
    gate_distribution = torch.distributions.Bernoulli(logits=gate_logits)
    weight_distribution = torch.distributions.Normal(weight_mean, log_std.exp())

    log_prob = gate_distribution.log_prob(gate_actions).sum(dim=-1)
    log_prob = log_prob + weight_distribution.log_prob(raw_weights).sum(dim=-1)
    entropy = gate_distribution.entropy().sum(dim=-1) + weight_distribution.entropy().sum(dim=-1)
    return log_prob, entropy, values


@torch.no_grad()
def deterministic_hybrid_action(model: HybridActorCritic, state: torch.Tensor, gate_threshold: float):
    gate_logits, weight_mean, _, _ = model(state)
    gate_action = (torch.sigmoid(gate_logits) >= gate_threshold).float()
    return coefficients_from_action(gate_action, weight_mean, model.coefficient_mode)
