from pathlib import Path

import torch

from mmcr.task_vectors import load_state_dict
from mmcr.ties import get_merge_keys


TENSOR_TYPES = ["patch", "attn_qkv", "attn_proj", "mlp_fc1", "mlp_fc2", "norm", "other"]


def tensor_type(key: str):
    if key.startswith("patch_embed"):
        return "patch"
    if ".attn.qkv." in key:
        return "attn_qkv"
    if ".attn.proj." in key:
        return "attn_proj"
    if ".mlp.fc1." in key:
        return "mlp_fc1"
    if ".mlp.fc2." in key:
        return "mlp_fc2"
    if "norm" in key:
        return "norm"
    return "other"


def layer_group(key: str):
    if key.startswith("patch_embed") or key in {"cls_token", "pos_embed"}:
        return "embedding"
    if key.startswith("blocks."):
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            return f"block_{parts[1]}"
    if key.startswith("norm."):
        return "final_norm"
    return "other"


def decision_group(key: str, decision_level: str):
    if decision_level == "tensor":
        return key
    if decision_level == "layer":
        return layer_group(key)
    if decision_level == "group":
        return tensor_type(key)
    if decision_level == "task":
        return "all_tensors"
    raise ValueError("decision_level must be one of: task, group, layer, tensor")


def build_decision_groups(keys: list[str], decision_level: str):
    group_to_keys = {}
    for key in keys:
        group = decision_group(key, decision_level)
        group_to_keys.setdefault(group, []).append(key)
    groups = [{"name": group, "keys": group_keys} for group, group_keys in group_to_keys.items()]
    return groups


def one_hot_tensor_type(key: str):
    values = torch.zeros(len(TENSOR_TYPES), dtype=torch.float32)
    values[TENSOR_TYPES.index(tensor_type(key))] = 1.0
    return values


def load_merge_inputs(zeroshot_path: Path | str, encoder_paths: list[Path | str], map_location="cpu"):
    zeroshot_state = load_state_dict(zeroshot_path, map_location=map_location)
    finetuned_states = [load_state_dict(path, map_location=map_location) for path in encoder_paths]
    keys = get_merge_keys(zeroshot_state, finetuned_states)
    task_vectors = []
    for finetuned_state in finetuned_states:
        task_vectors.append(
            {
                key: finetuned_state[key].detach().cpu() - zeroshot_state[key].detach().cpu()
                for key in keys
            }
        )
    return zeroshot_state, finetuned_states, task_vectors, keys


def cosine_or_zero(left: torch.Tensor, right: torch.Tensor):
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if left_norm.item() == 0 or right_norm.item() == 0:
        return torch.tensor(0.0)
    return torch.dot(left, right) / (left_norm * right_norm)


def build_tensor_features(task_vectors: list[dict[str, torch.Tensor]], keys: list[str]):
    num_tasks = len(task_vectors)
    rows = []
    for key_index, key in enumerate(keys):
        progress = torch.tensor([key_index / max(1, len(keys) - 1)], dtype=torch.float32)
        log_numel = torch.tensor([torch.log1p(torch.tensor(float(task_vectors[0][key].numel())))])
        norms = torch.stack(
            [torch.log1p(torch.linalg.vector_norm(task_vector[key].float())) for task_vector in task_vectors]
        )

        cosines = []
        for left in range(num_tasks):
            for right in range(left + 1, num_tasks):
                cosines.append(cosine_or_zero(task_vectors[left][key], task_vectors[right][key]))
        cosine_features = torch.stack(cosines).float() if cosines else torch.empty(0)
        rows.append(torch.cat([progress, log_numel.float(), norms.float(), cosine_features, one_hot_tensor_type(key)]))

    features = torch.stack(rows)
    return normalize_features(features)


def build_group_features(task_vectors: list[dict[str, torch.Tensor]], groups: list[dict]):
    rows = []
    num_groups = len(groups)
    num_tasks = len(task_vectors)

    for group_index, group in enumerate(groups):
        group_keys = group["keys"]
        progress = torch.tensor([group_index / max(1, num_groups - 1)], dtype=torch.float32)
        total_numel = sum(task_vectors[0][key].numel() for key in group_keys)
        log_numel = torch.tensor([torch.log1p(torch.tensor(float(total_numel)))])

        norms = []
        for task_vector in task_vectors:
            total = torch.tensor(0.0)
            for key in group_keys:
                total = total + task_vector[key].float().pow(2).sum()
            norms.append(torch.log1p(total.sqrt()))
        norms = torch.stack(norms)

        cosines = []
        for left in range(num_tasks):
            for right in range(left + 1, num_tasks):
                left_flat = torch.cat([task_vectors[left][key].float().reshape(-1) for key in group_keys])
                right_flat = torch.cat([task_vectors[right][key].float().reshape(-1) for key in group_keys])
                cosines.append(cosine_or_zero(left_flat, right_flat))
        cosine_features = torch.stack(cosines).float() if cosines else torch.empty(0)

        type_counts = torch.zeros(len(TENSOR_TYPES), dtype=torch.float32)
        for key in group_keys:
            type_counts[TENSOR_TYPES.index(tensor_type(key))] += 1.0
        type_counts = type_counts / type_counts.sum().clamp(min=1.0)
        rows.append(torch.cat([progress, log_numel.float(), norms.float(), cosine_features, type_counts]))

    return normalize_features(torch.stack(rows))


def normalize_features(features: torch.Tensor):
    mean = features.mean(dim=0, keepdim=True)
    if features.shape[0] <= 1:
        std = torch.ones_like(mean)
    else:
        std = features.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    return (features - mean) / std


def merge_tensor(base_tensor, task_vectors, key: str, coefficients: torch.Tensor, scale: float):
    delta = None
    coefficients = coefficients.detach().cpu()
    for task_index, task_vector in enumerate(task_vectors):
        weighted = coefficients[task_index] * task_vector[key].detach().cpu()
        delta = weighted if delta is None else delta + weighted
    return base_tensor.detach().cpu().clone() + scale * delta.to(dtype=base_tensor.dtype)
