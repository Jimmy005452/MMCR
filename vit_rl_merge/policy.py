import torch
import torch.nn as nn
import torch.nn.functional as F


def initial_raw_value(coefficient_mode: str, coefficient_init: float):
    if coefficient_mode == "softmax":
        return 0.0
    if coefficient_mode == "sigmoid":
        value = min(max(coefficient_init, 1e-4), 1.0 - 1e-4)
        return torch.logit(torch.tensor(value)).item()
    if coefficient_mode == "positive":
        value = max(coefficient_init, 1e-4)
        return torch.log(torch.expm1(torch.tensor(value))).item()
    if coefficient_mode == "unconstrained":
        return coefficient_init
    raise ValueError("coefficient_mode must be one of: softmax, sigmoid, positive, unconstrained")


class TensorCoefficientPolicy(nn.Module):
    """Policy that maps tensor states to source-model coefficients."""

    def __init__(
        self,
        state_dim: int,
        num_sources: int,
        hidden_dim: int = 128,
        init_log_std: float = -0.5,
        coefficient_mode: str = "sigmoid",
        coefficient_init: float = 1.0,
    ):
        super().__init__()
        if coefficient_mode not in {"softmax", "sigmoid", "positive", "unconstrained"}:
            raise ValueError("coefficient_mode must be one of: softmax, sigmoid, positive, unconstrained")
        self.num_sources = num_sources
        self.coefficient_mode = coefficient_mode
        self.coefficient_init = coefficient_init
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, num_sources)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((num_sources,), init_log_std))

        # Start from a controlled coefficient value. In the usual RL setting,
        # the outer merge scale is 1.0 and these coefficients control strength.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, initial_raw_value(coefficient_mode, coefficient_init))

    def transform_action(self, raw_action):
        if self.coefficient_mode == "softmax":
            return F.softmax(raw_action, dim=-1)
        if self.coefficient_mode == "sigmoid":
            return torch.sigmoid(raw_action)
        if self.coefficient_mode == "positive":
            return F.softplus(raw_action)
        if self.coefficient_mode == "unconstrained":
            return raw_action
        raise RuntimeError(f"Unsupported coefficient_mode: {self.coefficient_mode}")

    def forward(self, state):
        hidden = self.backbone(state)
        mean = self.mean_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        log_std = self.log_std.clamp(min=-5.0, max=1.0)
        return mean, log_std, value

    def sample_action(self, state):
        mean, log_std, value = self(state)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw_action = distribution.sample()
        coefficients = self.transform_action(raw_action)
        log_prob = distribution.log_prob(raw_action).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return coefficients, log_prob, entropy, value

    @torch.no_grad()
    def deterministic_action(self, state):
        mean, _, value = self(state)
        return self.transform_action(mean), value
