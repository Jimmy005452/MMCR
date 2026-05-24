from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .env import RLMMCREnv
from .merge import coefficients_to_dict
from .policy import (
    HybridActorCritic,
    deterministic_hybrid_action,
    evaluate_hybrid_action,
    sample_hybrid_action,
)


@dataclass
class PPOStats:
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    ppo_epochs_ran: int


@torch.no_grad()
def collect_rollout(env: RLMMCREnv, model: HybridActorCritic) -> dict:
    state = env.reset().to(env.device)
    rollout = {
        "states": [],
        "selected": [],
        "coefficients": [],
        "gate_actions": [],
        "raw_weights": [],
        "log_probs": [],
        "values": [],
        "entropies": [],
        "rewards": [],
        "infos": [],
    }

    done = False
    while not done:
        active, coefficients, gate_action, raw_weights, log_prob, entropy, value = sample_hybrid_action(model, state)
        next_state, reward, done, info = env.step(active, coefficients)

        rollout["states"].append(state.detach().clone())
        rollout["selected"].append(active.detach().cpu().tolist())
        rollout["coefficients"].append(coefficients.detach().cpu().tolist())
        rollout["gate_actions"].append(gate_action.detach().clone())
        rollout["raw_weights"].append(raw_weights.detach().clone())
        rollout["log_probs"].append(log_prob.detach())
        rollout["values"].append(value.detach())
        rollout["entropies"].append(entropy.detach())
        rollout["rewards"].append(reward)
        rollout["infos"].append(info)
        state = next_state.to(env.device)

    for key in ("states", "gate_actions", "raw_weights", "log_probs", "values", "entropies"):
        rollout[key] = torch.stack(rollout[key])
    return rollout


def compute_gae(rewards: list[float], values: torch.Tensor, gamma: float, gae_lambda: float):
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=values.device)
    advantages = torch.zeros_like(values)
    gae = torch.zeros((), dtype=torch.float32, device=values.device)

    for step in reversed(range(len(rewards))):
        next_value = values[step + 1] if step + 1 < len(values) else torch.zeros_like(values[step])
        delta = rewards_tensor[step] + gamma * next_value - values[step]
        gae = delta + gamma * gae_lambda * gae
        advantages[step] = gae
    return advantages, advantages + values


def update_policy(
    model: HybridActorCritic,
    optimizer: torch.optim.Optimizer,
    rollouts: list[dict],
    *,
    gamma: float,
    gae_lambda: float,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    ppo_epochs: int,
    target_kl: float,
    max_grad_norm: float,
    minibatch_size: int,
    normalize_advantages: bool,
    device: torch.device,
) -> PPOStats:
    states = torch.cat([rollout["states"] for rollout in rollouts], dim=0)
    gate_actions = torch.cat([rollout["gate_actions"] for rollout in rollouts], dim=0)
    raw_weights = torch.cat([rollout["raw_weights"] for rollout in rollouts], dim=0)
    old_log_probs = torch.cat([rollout["log_probs"] for rollout in rollouts], dim=0).detach()

    advantages, returns = [], []
    for rollout in rollouts:
        rollout_advantages, rollout_returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
            gamma,
            gae_lambda,
        )
        advantages.append(rollout_advantages)
        returns.append(rollout_returns)
    advantages = torch.cat(advantages, dim=0)
    returns = torch.cat(returns, dim=0)
    if normalize_advantages:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp(min=1e-8)

    batch_size = states.shape[0]
    minibatch_size = batch_size if minibatch_size <= 0 else min(minibatch_size, batch_size)
    last = {"loss": None, "policy": None, "value": None, "entropy": None, "kl": None, "clip": None, "epochs": 0}

    for epoch in range(ppo_epochs):
        permutation = torch.randperm(batch_size, device=device)
        kl_terms = []
        clip_terms = []

        for start in range(0, batch_size, minibatch_size):
            indices = permutation[start : start + minibatch_size]
            new_log_probs, entropy, values = evaluate_hybrid_action(
                model,
                states[indices],
                gate_actions[indices],
                raw_weights[indices],
            )
            ratios = torch.exp(new_log_probs - old_log_probs[indices])
            clipped = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
            policy_loss = -torch.min(ratios * advantages[indices], clipped * advantages[indices]).mean()
            value_loss = F.mse_loss(values, returns[indices])
            entropy_term = entropy.mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_term

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl_terms.append((old_log_probs[indices] - new_log_probs).mean())
                clip_terms.append((ratios.sub(1.0).abs() > clip_eps).float().mean())

        last.update(
            loss=loss,
            policy=policy_loss,
            value=value_loss,
            entropy=entropy_term,
            kl=torch.stack(kl_terms).mean(),
            clip=torch.stack(clip_terms).mean(),
            epochs=epoch + 1,
        )
        if last["kl"].item() > target_kl:
            break

    return PPOStats(
        loss=float(last["loss"].item()),
        policy_loss=float(last["policy"].item()),
        value_loss=float(last["value"].item()),
        entropy=float(last["entropy"].item()),
        approx_kl=float(last["kl"].item()),
        clip_fraction=float(last["clip"].item()),
        ppo_epochs_ran=int(last["epochs"]),
    )


@torch.no_grad()
def deterministic_policy_result(env: RLMMCREnv, model: HybridActorCritic, gate_threshold: float) -> dict:
    state = env.reset().to(env.device)
    selected_history = []
    coefficient_history = []
    infos = []
    done = False

    while not done:
        selected, coefficients = deterministic_hybrid_action(model, state, gate_threshold=gate_threshold)
        state, _, done, info = env.step(selected, coefficients)
        selected_history.append(selected.detach().cpu().tolist())
        coefficient_history.append(coefficients.detach().cpu().tolist())
        infos.append(info)
        state = state.to(env.device)

    final_coefficients = torch.tensor(coefficient_history, dtype=torch.float32)
    expanded_coefficients = env.expand_coefficients(final_coefficients)
    terminal_info = infos[-1]
    return {
        "selected": selected_history,
        "coefficients": coefficient_history,
        "expanded_coefficients": expanded_coefficients.tolist(),
        "average": float(env.previous_average),
        "objective": float(terminal_info["objective"]),
        "scores": dict(env.previous_scores),
        "infos": infos,
        "coefficients_by_layer": coefficients_to_dict(
            expanded_coefficients,
            env.layer_names,
            env.layered_task_vectors.task_names,
        ),
    }
