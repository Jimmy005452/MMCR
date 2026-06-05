from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """Layer-wise model-merging environment with entropy-based rewards."""

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
        score_imbalance_coef: float = 0.25,
        reward_eval_interval: int = 1,
        episode_reward_only: bool = False,
        coefficient_mode: str = "positive",
        coefficient_init: float = 1.0,
        cache_task_vectors_device: bool = False,
        merge_granularity: str = "layer",
        state_mode: str = "minimal",
        reward_mode: str = "entropy",
        batched_reward_eval: bool = False,
        batched_reward_max_samples: int = 128,
    ):
        if reward_eval_interval <= 0:
            raise ValueError("reward_eval_interval must be positive.")
        if merge_granularity not in {"layer", "global"}:
            raise ValueError("merge_granularity must be either 'layer' or 'global'.")
        if state_mode not in {"minimal", "full_coefficients"}:
            raise ValueError("state_mode must be either 'minimal' or 'full_coefficients'.")
        if reward_mode != "entropy":
            raise ValueError("reward_mode must be entropy.")

        self.encoder = encoder
        self.heads = nn.ModuleDict(heads)
        self.layered_task_vectors = layered_task_vectors
        self.device = device
        self.amp = amp
        self.terminal_bonus = terminal_bonus
        self.reward_scale = reward_scale
        self.step_reward_coef = step_reward_coef
        self.score_imbalance_coef = score_imbalance_coef
        self.reward_eval_interval = reward_eval_interval
        self.episode_reward_only = episode_reward_only
        self.coefficient_mode = coefficient_mode
        self.merge_granularity = merge_granularity
        self.state_mode = state_mode
        self.reward_mode = reward_mode
        self.batched_reward_eval = bool(batched_reward_eval)
        self.batched_reward_max_samples = int(batched_reward_max_samples)

        self.num_layers = layered_task_vectors.num_layers
        self.num_models = layered_task_vectors.num_models
        self.layer_names = layered_task_vectors.layer_names
        self.layer_grams = build_layer_gram_matrices(layered_task_vectors)
        self.layer_geometry_features = [self._layer_geometry_features(gram) for gram in self.layer_grams]
        self.global_geometry_feature = torch.stack(self.layer_geometry_features).mean(dim=0)

        self._pretrained_state_device = {
            key: value.detach().to(device, non_blocking=True)
            for key, value in layered_task_vectors.pretrained_state.items()
        }
        self.reward_sampling_mode = getattr(reward_batches, "sampling_mode", "sequential")
        self.reward_batches_per_eval = getattr(reward_batches, "batches_per_eval", None)
        self._reward_eval_counter = 0
        self._reward_batches_device = {
            dataset: [(images.to(device), targets.to(device)) for images, targets in batches]
            for dataset, batches in reward_batches.items()
        }
        self._task_vectors_stacked_device = self._cache_stacked_task_vectors(device) if cache_task_vectors_device else None
        self._merged_state_device = {key: value.detach().clone() for key, value in self._pretrained_state_device.items()}

        self.encoder.eval().requires_grad_(False)
        for head in self.heads.values():
            head.eval().requires_grad_(False)

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
        if self.state_mode == "minimal":
            return 3 * self.num_models + 1
        if self.state_mode == "full_coefficients":
            return 1 + 2 * self.num_models + self.num_layers * self.num_models + self.num_layers
        raise RuntimeError(f"Unknown state_mode: {self.state_mode}")

    def reset(self) -> torch.Tensor:
        self.layer_index = 0
        self.coefficients_by_layer = self._initial_coefficients_by_layer.clone()
        self._merged_state_device = dict(self._initial_merged_state_device)
        self.previous_average = self._initial_average
        self.previous_scores = dict(self._initial_scores)
        self.previous_objective = self._initial_objective
        self.previous_reward_stats = dict(self._initial_reward_stats)
        return self._state()

    def set_reward_pool_position(self, position: int) -> None:
        self._reward_eval_counter = max(0, int(position))

    def fork(self) -> "RLMMCREnv":
        """Create a lightweight rollout copy with independent mutable episode state."""
        clone = object.__new__(type(self))
        clone.__dict__ = self.__dict__.copy()
        clone.reset()
        return clone

    def step(self, selected: torch.Tensor, coefficients: torch.Tensor):
        if self.layer_index >= self.num_layers:
            raise RuntimeError("Episode is already done. Call reset().")

        selected = selected.detach().cpu().float()
        coefficients = self._prepare_coefficients(coefficients)
        if tuple(selected.shape) != (self.num_models,) or tuple(coefficients.shape) != (self.num_models,):
            raise ValueError(f"selected and coefficients must have shape ({self.num_models},).")

        if self.merge_granularity == "global":
            layer_name = "global"
            for layer_index in range(self.num_layers):
                self._apply_layer_coefficients(layer_index, coefficients)
            self.coefficients_by_layer[:] = coefficients.unsqueeze(0).expand_as(self.coefficients_by_layer)
            done = True
            reward, average, scores, objective, reward_stats, reward_evaluated = self._reward_for_step(done)
            self.layer_index = self.num_layers
        else:
            layer_name = self.layer_names[self.layer_index]
            self._apply_layer_coefficients(self.layer_index, coefficients)
            self.coefficients_by_layer[self.layer_index] = coefficients

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
    def expand_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        coefficients = coefficients.detach().cpu().float()
        if coefficients.ndim == 1:
            coefficients = coefficients.unsqueeze(0)
        if tuple(coefficients.shape) == (1, self.num_models):
            return coefficients.expand(self.num_layers, self.num_models).clone()
        return coefficients

    @torch.no_grad()
    def export_merged_state(self, coefficients_by_layer: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if coefficients_by_layer is None:
            return {key: value.detach().cpu().clone() for key, value in self._merged_state_device.items()}
        return merge_state_with_layer_coefficients(
            self.layered_task_vectors,
            self.expand_coefficients(coefficients_by_layer),
            coefficient_mode=self.coefficient_mode,
        )

    def _cache_task_vectors(self, device: torch.device) -> list[dict[str, torch.Tensor]]:
        return [
            {key: value.detach().to(device, non_blocking=True) for key, value in task_vector.items()}
            for task_vector in self.layered_task_vectors.task_vectors
        ]

    def _cache_stacked_task_vectors(self, device: torch.device) -> dict[str, torch.Tensor]:
        stacked = {}
        for key in self.layered_task_vectors.task_vectors[0]:
            stacked[key] = torch.stack([
                task_vector[key].detach().to(device, non_blocking=True)
                for task_vector in self.layered_task_vectors.task_vectors
            ], dim=0)
        return stacked

    @torch.no_grad()
    def _cache_initial_state(self) -> None:
        for layer_index, coefficients in enumerate(self._initial_coefficients_by_layer):
            self._apply_layer_coefficients(layer_index, coefficients)
        self._initial_average, self._initial_scores = self._evaluate_current_average()
        self._initial_objective, self._initial_reward_stats = self._objective(self._initial_scores)
        self._initial_merged_state_device = dict(self._merged_state_device)

    def _state(self) -> torch.Tensor:
        if self.merge_granularity == "global":
            layer_progress = torch.tensor([1.0 if self.layer_index >= self.num_layers else 0.0], dtype=torch.float32)
            layer_geometry = self.global_geometry_feature if self.layer_index < self.num_layers else torch.zeros(2 * self.num_models, dtype=torch.float32)
        elif self.layer_index >= self.num_layers:
            layer_progress = torch.tensor([1.0], dtype=torch.float32)
            layer_geometry = torch.zeros(2 * self.num_models, dtype=torch.float32)
        else:
            layer_progress = torch.tensor([(self.layer_index + 1) / max(1, self.num_layers)], dtype=torch.float32)
            layer_geometry = self.layer_geometry_features[self.layer_index]

        if self.state_mode == "minimal":
            parts = [
                layer_progress,
                layer_geometry,
                self._mean_coefficients_so_far(),
            ]
        elif self.state_mode == "full_coefficients":
            filled_mask = torch.zeros(self.num_layers, dtype=torch.float32)
            filled_mask[: min(self.layer_index, self.num_layers)] = 1.0
            parts = [
                layer_progress,
                layer_geometry,
                self.coefficients_by_layer.float().reshape(-1),
                filled_mask,
            ]
        else:
            raise RuntimeError(f"Unknown state_mode: {self.state_mode}")

        return torch.cat(parts, dim=0)

    def _reward_for_step(self, done: bool):
        reward_evaluated = done or (
            not self.episode_reward_only
            and (self.layer_index + 1) % self.reward_eval_interval == 0
        )
        if reward_evaluated:
            average, scores = self._evaluate_current_average()
            objective, reward_stats = self._objective(scores)
            reward = 0.0 if self.episode_reward_only else self.step_reward_coef * (objective - self.previous_objective)
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

    def _layer_geometry_features(self, gram: torch.Tensor) -> torch.Tensor:
        norms = torch.diagonal(gram).clamp(min=0.0).sqrt().float()
        relative_norms = norms / norms.mean().clamp(min=1e-8)

        dot_to_average = gram.mean(dim=1).float()
        average_norm = gram.mean().clamp(min=0.0).sqrt().float()
        cos_to_average = dot_to_average / (norms * average_norm).clamp(min=1e-8)
        return torch.cat([relative_norms, cos_to_average.clamp(min=-1.0, max=1.0)], dim=0)

    def _mean_coefficients_so_far(self) -> torch.Tensor:
        if self.layer_index == 0:
            return self._initial_coefficients_by_layer[0].float()
        return self.coefficients_by_layer[: self.layer_index].float().mean(dim=0)

    @torch.no_grad()

    @torch.no_grad()

    @torch.no_grad()

    @torch.no_grad()

    def _layer_norm_ratio(self, layer_index: int, coefficients: torch.Tensor) -> torch.Tensor:
        gram = self.layer_grams[layer_index]
        coefficients = coefficients.detach().cpu().float()
        merged_norm = (coefficients @ gram @ coefficients).clamp(min=0.0).sqrt()
        source_norm = torch.diagonal(gram).clamp(min=0.0).sqrt().mean().clamp(min=1e-8)
        return (merged_norm / source_norm).clamp(max=10.0)

    def _prepare_coefficients(self, coefficients: torch.Tensor) -> torch.Tensor:
        coefficients = coefficients.detach().cpu().float()
        return normalize_coefficients(coefficients, self.coefficient_mode)

    @torch.no_grad()

    def _active_reward_batches_device(self) -> dict[str, list[tuple[torch.Tensor, torch.Tensor]]]:
        if self.reward_sampling_mode != "stratified_pool" or not self.reward_batches_per_eval:
            return self._reward_batches_device
        active = {}
        window = max(1, int(self.reward_batches_per_eval))
        for dataset, batches in self._reward_batches_device.items():
            if len(batches) <= window:
                active[dataset] = batches
                continue
            start = (self._reward_eval_counter * window) % len(batches)
            active[dataset] = [batches[(start + offset) % len(batches)] for offset in range(window)]
        return active

    def _advance_reward_batches(self) -> None:
        if self.reward_sampling_mode == "stratified_pool" and self.reward_batches_per_eval:
            self._reward_eval_counter += 1

    @torch.no_grad()
    def _evaluate_current_average(self) -> tuple[float, dict[str, float]]:
        active_batches = self._active_reward_batches_device()
        if self.batched_reward_eval:
            scores = self._evaluate_current_scores_batched(self._merged_state_device, active_batches)
        else:
            scores = {
                dataset: self._evaluate_dataset_score(self._merged_state_device, dataset, batches)
                for dataset, batches in active_batches.items()
            }
        self._advance_reward_batches()
        return sum(scores.values()) / len(scores), scores

    @torch.no_grad()
    def _evaluate_current_scores_batched(self, device_state: dict[str, torch.Tensor], reward_batches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]]) -> dict[str, float]:
        accum = {dataset: {"entropy": 0.0, "total": 0} for dataset in reward_batches}
        autocast_enabled = self.amp and self.device.type == "cuda"
        max_batches = max(len(batches) for batches in reward_batches.values())

        for batch_index in range(max_batches):
            items = []
            for dataset, batches in reward_batches.items():
                if batch_index < len(batches):
                    images, targets = batches[batch_index]
                    items.append((dataset, images, targets))
            if not items:
                continue

            chunks = []
            current = []
            current_samples = 0
            max_samples = self.batched_reward_max_samples if self.batched_reward_max_samples > 0 else sum(int(item[1].shape[0]) for item in items)
            for item in items:
                item_samples = int(item[1].shape[0])
                if current and current_samples + item_samples > max_samples:
                    chunks.append(current)
                    current = []
                    current_samples = 0
                current.append(item)
                current_samples += item_samples
            if current:
                chunks.append(current)

            for chunk in chunks:
                images_all = torch.cat([item[1] for item in chunk], dim=0)
                with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                    features_all = functional_call(self.encoder, device_state, (images_all,))

                offset = 0
                for dataset, images, _targets in chunk:
                    count = int(images.shape[0])
                    features = features_all[offset : offset + count]
                    offset += count
                    with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                        logits = self.heads[dataset](features)
                    accum[dataset]["entropy"] += float(self._entropy_from_logits(logits).sum().item())
                    accum[dataset]["total"] += count

        return {
            dataset: -row["entropy"] / max(int(row["total"]), 1)
            for dataset, row in accum.items()
        }

    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    @torch.no_grad()
    def _evaluate_dataset_score(self, device_state: dict[str, torch.Tensor], dataset: str, batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None) -> float:
        return self._evaluate_dataset_neg_entropy(device_state, dataset, batches)

    @torch.no_grad()

    @torch.no_grad()
    def _evaluate_dataset_neg_entropy(self, device_state: dict[str, torch.Tensor], dataset: str, batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None) -> float:
        total_entropy = 0.0
        total = 0
        head = self.heads[dataset]
        autocast_enabled = self.amp and self.device.type == "cuda"

        batches = self._reward_batches_device[dataset] if batches is None else batches
        for images, _targets in batches:
            with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                features = functional_call(self.encoder, device_state, (images,))
                logits = head(features)
            entropy = self._entropy_from_logits(logits)
            total_entropy += float(entropy.sum().item())
            total += int(images.shape[0])
        return -total_entropy / max(total, 1)

    @torch.no_grad()

    @torch.no_grad()
    def _apply_layer_coefficients(self, layer_index: int, coefficients: torch.Tensor) -> None:
        layer_name = self.layer_names[layer_index]
        coefficients = self._prepare_coefficients(coefficients)
        device_coefficients = coefficients.to(self.device, non_blocking=True)
        for key in self.layered_task_vectors.layer_to_keys[layer_name]:
            base_value = self._pretrained_state_device[key]
            if self._task_vectors_stacked_device is not None:
                stacked = self._task_vectors_stacked_device[key].float()
                view_shape = (self.num_models,) + (1,) * (stacked.ndim - 1)
                merged_delta = (device_coefficients.view(view_shape) * stacked).sum(dim=0)
                merged_value = base_value.float() + merged_delta
            else:
                merged_value = base_value.float()
                for model_index, task_vector in enumerate(self.layered_task_vectors.task_vectors):
                    task_value = task_vector[key].to(self.device, non_blocking=True)
                    merged_value = merged_value + float(coefficients[model_index].item()) * task_value.float()
            self._merged_state_device[key] = merged_value.to(dtype=base_value.dtype)

    def _objective(self, scores: dict[str, float]) -> tuple[float, dict]:
        values = torch.tensor(list(scores.values()), dtype=torch.float32)
        mean_score = values.mean()
        std_score = values.std(unbiased=False)
        worst_score = values.min()
        objective = mean_score - self.score_imbalance_coef * std_score
        return float(objective.item()), {
            "reward_mode": self.reward_mode,
            "scores": dict(scores),
            "mean_score": float(mean_score.item()),
            "std_score": float(std_score.item()),
            "worst_score": float(worst_score.item()),
            "best_score": float(values.max().item()),
            "score_gap": float((values.max() - values.min()).item()),
            "score_imbalance_coef": float(self.score_imbalance_coef),
            "objective": float(objective.item()),
        }
