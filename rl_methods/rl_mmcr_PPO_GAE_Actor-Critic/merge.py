from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from mmcr.adamerging import load_ties_task_vectors


@dataclass(frozen=True)
class LayeredTaskVectors:
    pretrained_state: dict[str, torch.Tensor]
    finetuned_paths: list[Path]
    task_vectors: list[dict[str, torch.Tensor]]
    task_names: list[str]
    layer_names: list[str]
    layer_to_keys: dict[str, list[str]]

    @property
    def num_layers(self) -> int:
        return len(self.layer_names)

    @property
    def num_models(self) -> int:
        return len(self.task_vectors)


def _layer_name(key: str) -> str:
    if key.startswith("blocks."):
        return ".".join(key.split(".")[:2])
    if key.startswith("patch_embed.") or key in {"cls_token", "pos_embed", "dist_token"}:
        return "embeddings"
    if key.startswith("norm."):
        return "norm"
    return key.split(".", maxsplit=1)[0]


def _group_layers(task_vectors: list[dict[str, torch.Tensor]]) -> tuple[list[str], dict[str, list[str]]]:
    if not task_vectors:
        raise ValueError("task_vectors must not be empty.")

    layer_to_keys: dict[str, list[str]] = {}
    for key in task_vectors[0]:
        layer_to_keys.setdefault(_layer_name(key), []).append(key)
    return list(layer_to_keys), layer_to_keys


def load_layered_ties_task_vectors(
    zeroshot_path: Path | str,
    finetuned_paths: list[Path | str],
    task_names: list[str],
    top_k_percent: float = 20,
    map_location: str = "cpu",
) -> LayeredTaskVectors:
    if len(finetuned_paths) != len(task_names):
        raise ValueError("finetuned_paths and task_names must have the same length.")

    pretrained_state, task_vectors, _ = load_ties_task_vectors(
        zeroshot_path=zeroshot_path,
        finetuned_paths=finetuned_paths,
        top_k_percent=top_k_percent,
        map_location=map_location,
    )
    layer_names, layer_to_keys = _group_layers(task_vectors)
    return LayeredTaskVectors(
        pretrained_state=pretrained_state,
        finetuned_paths=[Path(path) for path in finetuned_paths],
        task_vectors=task_vectors,
        task_names=task_names,
        layer_names=layer_names,
        layer_to_keys=layer_to_keys,
    )


def initial_coefficients(
    num_layers: int,
    num_models: int,
    coefficient_mode: str,
    coefficient_init: float,
) -> torch.Tensor:
    if coefficient_mode == "softmax":
        value = 1.0 / max(num_models, 1)
        return torch.full((num_layers, num_models), value, dtype=torch.float32)
    return torch.full((num_layers, num_models), coefficient_init, dtype=torch.float32)


def normalize_coefficients(coefficients: torch.Tensor, coefficient_mode: str) -> torch.Tensor:
    if coefficient_mode != "softmax":
        return coefficients
    if coefficients.ndim == 1:
        return coefficients / coefficients.sum().clamp(min=1e-8)
    return coefficients / coefficients.sum(dim=1, keepdim=True).clamp(min=1e-8)


def flatten_layer(task_vector: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    return torch.cat([task_vector[key].detach().cpu().float().reshape(-1) for key in keys])


def build_layer_gram_matrices(layered_task_vectors: LayeredTaskVectors) -> list[torch.Tensor]:
    grams = []
    for layer_name in layered_task_vectors.layer_names:
        flats = [
            flatten_layer(task_vector, layered_task_vectors.layer_to_keys[layer_name])
            for task_vector in layered_task_vectors.task_vectors
        ]
        gram = torch.stack(flats) @ torch.stack(flats).T
        grams.append(gram.float())
    return grams


def merge_state_with_layer_coefficients(
    layered_task_vectors: LayeredTaskVectors,
    coefficients_by_layer: torch.Tensor,
    coefficient_mode: str = "softmax",
) -> dict[str, torch.Tensor]:
    expected_shape = (layered_task_vectors.num_layers, layered_task_vectors.num_models)
    coefficients_by_layer = coefficients_by_layer.detach().cpu().float()
    if tuple(coefficients_by_layer.shape) != expected_shape:
        raise ValueError(f"Expected coefficients shape {expected_shape}, got {tuple(coefficients_by_layer.shape)}.")

    coefficients_by_layer = normalize_coefficients(coefficients_by_layer, coefficient_mode)
    merged_state = {key: value.detach().cpu().clone() for key, value in layered_task_vectors.pretrained_state.items()}

    for layer_index, layer_name in enumerate(layered_task_vectors.layer_names):
        coefficients = coefficients_by_layer[layer_index]
        for key in layered_task_vectors.layer_to_keys[layer_name]:
            base_value = layered_task_vectors.pretrained_state[key].detach().cpu()
            merged_value = base_value.float()
            for model_index, task_vector in enumerate(layered_task_vectors.task_vectors):
                merged_value = merged_value + coefficients[model_index] * task_vector[key].detach().cpu().float()
            merged_state[key] = merged_value.to(dtype=base_value.dtype)
    return merged_state


def coefficients_to_dict(
    coefficients_by_layer: torch.Tensor,
    layer_names: list[str],
    task_names: list[str],
) -> dict[str, dict[str, float]]:
    coefficients_by_layer = coefficients_by_layer.detach().cpu().float()
    return {
        layer_name: {
            task_name: float(coefficients_by_layer[layer_index, task_index].item())
            for task_index, task_name in enumerate(task_names)
        }
        for layer_index, layer_name in enumerate(layer_names)
    }
