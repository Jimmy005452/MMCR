from __future__ import annotations

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from mmcr.task_vectors import load_state_dict

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
        coefficient_mode: str = "positive",
        coefficient_init: float = 1.0,
        cache_task_vectors_device: bool = False,
        merge_granularity: str = "layer",
        source_baseline_scores: dict[str, float] | None = None,
        activation_reward_coef: float = 0.0,
        state_mode: str = "minimal",
    ):
        if reward_eval_interval <= 0:
            raise ValueError("reward_eval_interval must be positive.")
        if merge_granularity not in {"layer", "global"}:
            raise ValueError("merge_granularity must be either 'layer' or 'global'.")
        if state_mode not in {"minimal", "full_coefficients"}:
            raise ValueError("state_mode must be either 'minimal' or 'full_coefficients'.")

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
        self.merge_granularity = merge_granularity
        self.activation_reward_coef = float(activation_reward_coef)
        self.state_mode = state_mode

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
        self._reward_batches_device = {
            dataset: [(images.to(device), targets.to(device)) for images, targets in batches]
            for dataset, batches in reward_batches.items()
        }
        self._task_vectors_device = self._cache_task_vectors(device) if cache_task_vectors_device else None
        self._merged_state_device = {key: value.detach().clone() for key, value in self._pretrained_state_device.items()}

        self.encoder.eval().requires_grad_(False)
        for head in self.heads.values():
            head.eval().requires_grad_(False)

        self.activation_modules = self._resolve_activation_modules()
        self._source_activation_refs = (
            self._precompute_source_activation_refs()
            if self.activation_reward_coef != 0.0 and self.merge_granularity == "layer"
            else None
        )

        self.source_baseline_scores = (
            self._validate_source_baseline_scores(source_baseline_scores)
            if source_baseline_scores is not None
            else self._evaluate_source_baseline_scores()
        )
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
            activation_score, activation_reward = self._activation_reward_for_layer(self.layer_index)
            activation_reward *= self.reward_scale
            reward += activation_reward
            self.layer_index += 1

        if self.merge_granularity == "global":
            activation_score = None
            activation_reward = 0.0

        info = {
            "layer_name": layer_name,
            "average": average,
            "objective": objective,
            "scores": scores,
            "reward_stats": reward_stats,
            "reward_evaluated": reward_evaluated,
            "activation_score": activation_score,
            "activation_reward": activation_reward,
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

    @torch.no_grad()
    def _cache_initial_state(self) -> None:
        for layer_index, coefficients in enumerate(self._initial_coefficients_by_layer):
            self._apply_layer_coefficients(layer_index, coefficients)
        self._initial_average, self._initial_scores = self._evaluate_current_average()
        self._initial_objective, self._initial_reward_stats = self._retention_objective(self._initial_scores)
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
            objective, reward_stats = self._retention_objective(scores)
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

    def _activation_module_name(self, layer_name: str) -> str | None:
        module_names = set(dict(self.encoder.named_modules()))
        candidates = []
        if layer_name == "embeddings":
            candidates = ["patch_embed", "pos_drop", "patch_drop"]
        elif layer_name == "norm":
            candidates = ["norm", "fc_norm"]
        else:
            candidates = [layer_name]
        for candidate in candidates:
            if candidate in module_names:
                return candidate
        return None

    def _resolve_activation_modules(self) -> list[str | None]:
        modules = [self._activation_module_name(layer_name) for layer_name in self.layer_names]
        if self.activation_reward_coef != 0.0 and self.merge_granularity == "layer":
            missing = [layer for layer, module in zip(self.layer_names, modules) if module is None]
            if missing:
                print("Activation reward will skip unresolved layer(s): " + ", ".join(missing))
        return modules

    def _pool_activation(self, activation: torch.Tensor) -> torch.Tensor:
        if isinstance(activation, (tuple, list)):
            activation = activation[0]
        activation = activation.detach().float()
        if activation.ndim <= 1:
            return activation.reshape(1, -1)
        return activation.reshape(activation.shape[0], -1)

    @torch.no_grad()
    def _capture_layer_activation(
        self,
        device_state: dict[str, torch.Tensor],
        images: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor | None:
        module_name = self.activation_modules[layer_index]
        if module_name is None:
            return None
        modules = dict(self.encoder.named_modules())
        module = modules[module_name]
        captured = {}

        def hook(_module, _inputs, output):
            captured["activation"] = self._pool_activation(output)

        handle = module.register_forward_hook(hook)
        try:
            functional_call(self.encoder, device_state, (images,))
        finally:
            handle.remove()
        return captured.get("activation")

    @torch.no_grad()
    def _capture_all_layer_activations(
        self,
        device_state: dict[str, torch.Tensor],
        images: torch.Tensor,
    ) -> list[torch.Tensor | None]:
        modules = dict(self.encoder.named_modules())
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer_index, module_name in enumerate(self.activation_modules):
            if module_name is None:
                continue

            def make_hook(index: int):
                def hook(_module, _inputs, output):
                    captured[index] = self._pool_activation(output).cpu()
                return hook

            handles.append(modules[module_name].register_forward_hook(make_hook(layer_index)))
        try:
            functional_call(self.encoder, device_state, (images,))
        finally:
            for handle in handles:
                handle.remove()
        return [captured.get(layer_index) for layer_index in range(self.num_layers)]

    @torch.no_grad()
    def _precompute_source_activation_refs(self) -> list[list[torch.Tensor | None]]:
        refs: list[list[torch.Tensor | None]] = [
            [None for _ in range(self.num_models)]
            for _ in range(self.num_layers)
        ]
        print("Precomputing source activation references for activation reward.")
        for model_index, (dataset, path) in enumerate(zip(self.layered_task_vectors.task_names, self.layered_task_vectors.finetuned_paths)):
            cpu_state = load_state_dict(path, map_location="cpu")
            source_state = {
                key: value.to(self.device, non_blocking=True)
                for key, value in cpu_state.items()
            }
            images = self._reward_batches_device[dataset][0][0]
            activations = self._capture_all_layer_activations(source_state, images)
            for layer_index, activation in enumerate(activations):
                refs[layer_index][model_index] = activation
            del source_state, cpu_state, activations
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        return refs

    @torch.no_grad()
    def _activation_reward_for_layer(self, layer_index: int) -> tuple[float | None, float]:
        if self.activation_reward_coef == 0.0 or self._source_activation_refs is None:
            return None, 0.0
        similarities = []
        for model_index, dataset in enumerate(self.layered_task_vectors.task_names):
            reference = self._source_activation_refs[layer_index][model_index]
            if reference is None:
                continue
            images = self._reward_batches_device[dataset][0][0]
            merged_activation = self._capture_layer_activation(self._merged_state_device, images, layer_index)
            if merged_activation is None:
                continue
            reference = reference.to(self.device, non_blocking=True)
            if reference.shape != merged_activation.shape:
                min_batch = min(reference.shape[0], merged_activation.shape[0])
                min_dim = min(reference.shape[1], merged_activation.shape[1])
                reference = reference[:min_batch, :min_dim]
                merged_activation = merged_activation[:min_batch, :min_dim]
            similarity = F.cosine_similarity(merged_activation, reference, dim=1).mean()
            similarities.append(similarity)
        if not similarities:
            return None, 0.0
        score = torch.stack(similarities).mean().item()
        return float(score), float(self.activation_reward_coef * score)

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


    def _validate_source_baseline_scores(self, scores: dict[str, float]) -> dict[str, float]:
        validated = {}
        missing = []
        for dataset in self.layered_task_vectors.task_names:
            if dataset not in scores:
                missing.append(dataset)
                continue
            value = float(scores[dataset])
            if value <= 0.0:
                raise ValueError(f"Cached source baseline for {dataset} must be positive, got {value}.")
            validated[dataset] = value
        if missing:
            raise ValueError("Cached source baseline is missing dataset(s): " + ", ".join(missing))
        return validated

    @torch.no_grad()
    def _evaluate_source_baseline_scores(self) -> dict[str, float]:
        scores = {}
        for dataset, path in zip(self.layered_task_vectors.task_names, self.layered_task_vectors.finetuned_paths):
            cpu_state = load_state_dict(path, map_location="cpu")
            source_state = {
                key: value.to(self.device, non_blocking=True)
                for key, value in cpu_state.items()
            }
            scores[dataset] = self._evaluate_dataset_state(source_state, dataset)
            del source_state, cpu_state
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
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
