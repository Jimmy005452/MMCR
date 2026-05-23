from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from .merge import (
    LayeredTaskVectors,
    build_layer_gram_matrices,
    initial_coefficients,
    merge_state_with_layer_coefficients,
    normalize_coefficients,
)


class RLMMCREnv:
    """Layer-wise model-merging environment with retention-based rewards."""

    def __init__(
        self,
        encoder: nn.Module,
        heads: dict[str, nn.Module],
        layered_task_vectors: LayeredTaskVectors,
        reward_batches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
        device: torch.device,
        amp: bool = False,
        terminal_bonus: float = 1.0,
        reward_scale: float = 1.0,
        step_reward_coef: float = 0.25,
        accuracy_imbalance_coef: float = 0.25,
        retention_worst_coef: float = 0.25,
        retention_drop_coef: float = 0.5,
        reward_eval_interval: int = 1,
        episode_reward_only: bool = False,
        coefficient_mode: str = "softmax",
        coefficient_init: float = 1.0,
        cache_task_vectors_device: bool = False,
    ):
        if reward_eval_interval <= 0:
            raise ValueError("reward_eval_interval must be positive.")

        self.encoder = encoder
        self.heads = nn.ModuleDict(heads)
        self.layered_task_vectors = layered_task_vectors
        self.device = device
        self.amp = amp
        self.terminal_bonus = terminal_bonus
        self.reward_scale = reward_scale
        self.step_reward_coef = step_reward_coef
        self.accuracy_imbalance_coef = accuracy_imbalance_coef
        self.retention_worst_coef = retention_worst_coef
        self.retention_drop_coef = retention_drop_coef
        self.reward_eval_interval = reward_eval_interval
        self.episode_reward_only = episode_reward_only
        self.coefficient_mode = coefficient_mode

        self.num_layers = layered_task_vectors.num_layers
        self.num_models = layered_task_vectors.num_models
        self.layer_names = layered_task_vectors.layer_names
        self.layer_grams = build_layer_gram_matrices(layered_task_vectors)
        self.layer_similarity_features = [self._pairwise_distance_features(gram) for gram in self.layer_grams]

        self._pretrained_state_device = {
            key: value.detach().to(device, non_blocking=True)
            for key, value in layered_task_vectors.pretrained_state.items()
        }
        self._reward_batches_device = {
            dataset: [(images.to(device), targets.to(device)) for images, targets in batches]
            for dataset, batches in reward_batches.items()
        }
        self._task_vectors_device = self._cache_task_vectors(device) if cache_task_vectors_device else None
        self._merged_state_device = {key: value.detach().clone() for key, value in self._pretrained_state_device.items()}

        self.encoder.eval().requires_grad_(False)
        for head in self.heads.values():
            head.eval().requires_grad_(False)

        self.source_baseline_scores = self._evaluate_source_baseline_scores()
        self._initial_coefficients_by_layer = initial_coefficients(
            self.num_layers,
            self.num_models,
            coefficient_mode,
            coefficient_init,
        )
        self._cache_initial_state()
        self.reset()

    @property
    def state_dim(self) -> int:
        return 1 + self.layer_similarity_features[0].numel() + 4 + self.num_models

    def reset(self) -> torch.Tensor:
        self.layer_index = 0
        self.coefficients_by_layer = self._initial_coefficients_by_layer.clone()
        self.selection_counts = torch.zeros(self.num_models, dtype=torch.float32)
        self.last_layer_norm_ratio = torch.tensor(1.0, dtype=torch.float32)
        self._merged_state_device = dict(self._initial_merged_state_device)
        self.previous_average = self._initial_average
        self.previous_scores = dict(self._initial_scores)
        self.previous_objective = self._initial_objective
        self.previous_reward_stats = dict(self._initial_reward_stats)
        return self._state()

    def step(self, selected: torch.Tensor, coefficients: torch.Tensor):
        if self.layer_index >= self.num_layers:
            raise RuntimeError("Episode is already done. Call reset().")

        selected = selected.detach().cpu().float()
        coefficients = self._prepare_coefficients(coefficients)
        if tuple(selected.shape) != (self.num_models,) or tuple(coefficients.shape) != (self.num_models,):
            raise ValueError(f"selected and coefficients must have shape ({self.num_models},).")

        layer_name = self.layer_names[self.layer_index]
        self._apply_layer_coefficients(self.layer_index, coefficients)
        self.coefficients_by_layer[self.layer_index] = coefficients
        self.selection_counts += selected
        self.last_layer_norm_ratio = self._layer_norm_ratio(self.layer_index, coefficients)

        done = self.layer_index + 1 == self.num_layers
        reward, average, scores, objective, reward_stats, reward_evaluated = self._reward_for_step(done)
        self.layer_index += 1

        info = {
            "layer_name": layer_name,
            "average": average,
            "objective": objective,
            "scores": scores,
            "reward_stats": reward_stats,
            "reward_evaluated": reward_evaluated,
            "selected": selected.tolist(),
            "coefficients": coefficients.tolist(),
            "coefficients_by_layer": self.coefficients_by_layer.tolist(),
        }
        return self._state(), float(reward), done, info

    @torch.no_grad()
    def export_merged_state(self, coefficients_by_layer: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if coefficients_by_layer is None:
            return {key: value.detach().cpu().clone() for key, value in self._merged_state_device.items()}
        return merge_state_with_layer_coefficients(
            self.layered_task_vectors,
            coefficients_by_layer,
            coefficient_mode=self.coefficient_mode,
        )

    def _cache_task_vectors(self, device: torch.device) -> list[dict[str, torch.Tensor]]:
        return [
            {key: value.detach().to(device, non_blocking=True) for key, value in task_vector.items()}
            for task_vector in self.layered_task_vectors.task_vectors
        ]

    @torch.no_grad()
    def _cache_initial_state(self) -> None:
        for layer_index, coefficients in enumerate(self._initial_coefficients_by_layer):
            self._apply_layer_coefficients(layer_index, coefficients)
        self._initial_average, self._initial_scores = self._evaluate_current_average()
        self._initial_objective, self._initial_reward_stats = self._retention_objective(self._initial_scores)
        self._initial_merged_state_device = dict(self._merged_state_device)

    def _state(self) -> torch.Tensor:
        if self.layer_index >= self.num_layers:
            layer_progress = torch.tensor([1.0], dtype=torch.float32)
            layer_similarity = torch.zeros_like(self.layer_similarity_features[0])
        else:
            layer_progress = torch.tensor([(self.layer_index + 1) / max(1, self.num_layers)], dtype=torch.float32)
            layer_similarity = self.layer_similarity_features[self.layer_index]
        history = self.selection_counts / max(1, self.num_layers)
        return torch.cat([layer_progress, layer_similarity, self._merged_model_statistics(), history], dim=0)

    def _reward_for_step(self, done: bool):
        reward_evaluated = done or (
            not self.episode_reward_only
            and (self.layer_index + 1) % self.reward_eval_interval == 0
        )
        if reward_evaluated:
            average, scores = self._evaluate_current_average()
            objective, reward_stats = self._retention_objective(scores)
            reward = self.step_reward_coef * (objective - self.previous_objective)
            self.previous_average = average
            self.previous_scores = scores
            self.previous_objective = objective
            self.previous_reward_stats = reward_stats
        else:
            average = self.previous_average
            scores = dict(self.previous_scores)
            objective = self.previous_objective
            reward_stats = dict(self.previous_reward_stats)
            reward = 0.0

        if done:
            reward += self.terminal_bonus * objective
        return reward * self.reward_scale, average, scores, objective, reward_stats, reward_evaluated

    def _pairwise_distance_features(self, gram: torch.Tensor) -> torch.Tensor:
        distances = []
        for left in range(self.num_models):
            for right in range(left + 1, self.num_models):
                distance_sq = gram[left, left] + gram[right, right] - 2.0 * gram[left, right]
                distances.append(torch.sqrt(distance_sq.clamp(min=0.0)))
        features = torch.stack(distances).float() if distances else torch.zeros(1, dtype=torch.float32)
        return features / features.abs().max().clamp(min=1e-8)

    def _merged_model_statistics(self) -> torch.Tensor:
        if self.layer_index == 0:
            return torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float32)

        ratios = torch.stack([
            self._layer_norm_ratio(layer_index, self.coefficients_by_layer[layer_index])
            for layer_index in range(self.layer_index)
        ]).float()
        history = self.coefficients_by_layer[: self.layer_index].clamp(min=1e-8)
        entropy = -(history * history.log()).sum(dim=1)
        entropy = entropy / torch.log(torch.tensor(float(self.num_models))).clamp(min=1e-8)
        return torch.stack([ratios.mean(), ratios.std(unbiased=False), self.last_layer_norm_ratio, entropy.mean()])

    def _layer_norm_ratio(self, layer_index: int, coefficients: torch.Tensor) -> torch.Tensor:
        gram = self.layer_grams[layer_index]
        coefficients = coefficients.detach().cpu().float()
        merged_norm = (coefficients @ gram @ coefficients).clamp(min=0.0).sqrt()
        source_norm = torch.diagonal(gram).clamp(min=0.0).sqrt().mean().clamp(min=1e-8)
        return (merged_norm / source_norm).clamp(max=10.0)

    def _prepare_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        coefficients = coefficients.detach().cpu().float()
        if self.coefficient_mode == "softmax" and coefficients.sum().abs().item() == 0:
            return torch.full((self.num_models,), 1.0 / self.num_models, dtype=torch.float32)
        return normalize_coefficients(coefficients, self.coefficient_mode)

    @torch.no_grad()
    def _evaluate_source_baseline_scores(self) -> dict[str, float]:
        scores = {}
        for model_index, dataset in enumerate(self.layered_task_vectors.task_names):
            source_state = {
                key: value.to(self.device, non_blocking=True)
                for key, value in self.layered_task_vectors.finetuned_states[model_index].items()
            }
            scores[dataset] = self._evaluate_dataset_state(source_state, dataset)
        return scores

    @torch.no_grad()
    def _evaluate_current_average(self) -> tuple[float, dict[str, float]]:
        scores = {
            dataset: self._evaluate_dataset_state(self._merged_state_device, dataset)
            for dataset in self._reward_batches_device
        }
        return sum(scores.values()) / len(scores), scores

    @torch.no_grad()
    def _evaluate_dataset_state(self, device_state: dict[str, torch.Tensor], dataset: str) -> float:
        correct = 0
        total = 0
        head = self.heads[dataset]
        autocast_enabled = self.amp and self.device.type == "cuda"

        for images, targets in self._reward_batches_device[dataset]:
            with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                features = functional_call(self.encoder, device_state, (images,))
                logits = head(features)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
        return correct / max(total, 1)

    @torch.no_grad()
    def _apply_layer_coefficients(self, layer_index: int, coefficients: torch.Tensor) -> None:
        layer_name = self.layer_names[layer_index]
        coefficients = self._prepare_coefficients(coefficients)
        for key in self.layered_task_vectors.layer_to_keys[layer_name]:
            base_value = self._pretrained_state_device[key]
            merged_value = base_value.float()
            for model_index, task_vector in enumerate(self.layered_task_vectors.task_vectors):
                task_value = (
                    task_vector[key].to(self.device, non_blocking=True)
                    if self._task_vectors_device is None
                    else self._task_vectors_device[model_index][key]
                )
                merged_value = merged_value + float(coefficients[model_index].item()) * task_value.float()
            self._merged_state_device[key] = merged_value.to(dtype=base_value.dtype)

    def _retention_objective(self, scores: dict[str, float]) -> tuple[float, dict]:
        values = torch.tensor(list(scores.values()), dtype=torch.float32)
        baselines = torch.tensor(
            [max(self.source_baseline_scores.get(dataset, 0.0), 1e-6) for dataset in scores],
            dtype=torch.float32,
        )
        retention = values / baselines
        mean_accuracy = values.mean()
        std_retention = retention.std(unbiased=False)
        worst_retention = retention.min()
        retention_shortfall = torch.clamp(1.0 - retention, min=0.0).mean()
        objective = (
            retention.mean()
            + self.retention_worst_coef * worst_retention
            - self.accuracy_imbalance_coef * std_retention
            - self.retention_drop_coef * retention_shortfall
        )
        return float(objective.item()), {
            "mean_accuracy": float(mean_accuracy.item()),
            "std_accuracy": float(values.std(unbiased=False).item()),
            "worst_accuracy": float(values.min().item()),
            "best_accuracy": float(values.max().item()),
            "accuracy_gap": float((values.max() - values.min()).item()),
            "source_baseline_scores": dict(self.source_baseline_scores),
            "retention_ratios": {
                dataset: float(score / max(self.source_baseline_scores.get(dataset, 0.0), 1e-6))
                for dataset, score in scores.items()
            },
            "relative_improvements": {
                dataset: float(
                    (score - self.source_baseline_scores.get(dataset, 0.0))
                    / max(self.source_baseline_scores.get(dataset, 0.0), 1e-6)
                )
                for dataset, score in scores.items()
            },
            "mean_retention": float(retention.mean().item()),
            "std_retention": float(std_retention.item()),
            "worst_retention": float(worst_retention.item()),
            "best_retention": float(retention.max().item()),
            "retention_shortfall": float(retention_shortfall.item()),
            "accuracy_imbalance_coef": float(self.accuracy_imbalance_coef),
            "retention_worst_coef": float(self.retention_worst_coef),
            "retention_drop_coef": float(self.retention_drop_coef),
            "objective": float(objective.item()),
        }
