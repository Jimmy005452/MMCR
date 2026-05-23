from pathlib import Path

import torch

from mmcr.task_vectors import load_state_dict


def get_merge_keys(pretrained_state: dict[str, torch.Tensor], finetuned_states: list[dict[str, torch.Tensor]]):
    keys = []
    for key, value in pretrained_state.items():
        if not torch.is_floating_point(value):
            continue
        if not all(key in state for state in finetuned_states):
            print(f"Warning: {key} is missing from at least one fine-tuned checkpoint; skipping.")
            continue
        if not all(state[key].shape == value.shape for state in finetuned_states):
            print(f"Warning: {key} has mismatched shapes across checkpoints; skipping.")
            continue
        keys.append(key)
    return keys


def state_dict_to_vector(state: dict[str, torch.Tensor], keys: list[str]):
    return torch.cat([state[key].detach().cpu().reshape(-1) for key in keys])


def vector_to_state_dict(vector: torch.Tensor, reference_state: dict[str, torch.Tensor], keys: list[str]):
    merged_state = {key: value.detach().cpu().clone() for key, value in reference_state.items()}
    offset = 0
    for key in keys:
        reference_value = reference_state[key]
        numel = reference_value.numel()
        merged_state[key] = vector[offset : offset + numel].reshape_as(reference_value).to(dtype=reference_value.dtype)
        offset += numel

    if offset != vector.numel():
        raise ValueError(f"Vector has {vector.numel()} values, but only consumed {offset}.")
    return merged_state


def topk_trim(task_vectors: torch.Tensor, top_k_percent: float):
    if not 0 < top_k_percent <= 100:
        raise ValueError("top_k_percent must be in (0, 100].")
    if top_k_percent == 100:
        return task_vectors

    values_per_task = task_vectors.shape[1]
    keep = max(1, int(values_per_task * top_k_percent / 100))
    trimmed = torch.zeros_like(task_vectors)

    for task_idx in range(task_vectors.shape[0]):
        _, indices = torch.topk(task_vectors[task_idx].abs(), k=keep, largest=True)
        trimmed[task_idx, indices] = task_vectors[task_idx, indices]
    return trimmed


def elect_sign(task_vectors: torch.Tensor):
    positive = torch.where(task_vectors > 0, task_vectors.abs(), torch.zeros_like(task_vectors)).sum(dim=0)
    negative = torch.where(task_vectors < 0, task_vectors.abs(), torch.zeros_like(task_vectors)).sum(dim=0)
    return torch.where(positive >= negative, torch.ones_like(positive), -torch.ones_like(negative))


def disjoint_merge(task_vectors: torch.Tensor, elected_sign: torch.Tensor, merge_func: str = "dis-sum"):
    sign_matches = torch.sign(task_vectors) == elected_sign.unsqueeze(0)
    nonzero = task_vectors != 0
    mask = sign_matches & nonzero & (elected_sign.unsqueeze(0) != 0)
    selected = task_vectors * mask

    if merge_func == "dis-sum":
        return selected.sum(dim=0)

    if merge_func == "dis-mean":
        counts = mask.sum(dim=0).clamp(min=1)
        return selected.sum(dim=0) / counts

    raise ValueError("merge_func must be one of: dis-sum, dis-mean")


def ties_select_task_vectors(task_vectors: torch.Tensor, top_k_percent: float = 20):
    """Return the per-task TIES-selected entries before summing them."""
    trimmed = topk_trim(task_vectors, top_k_percent=top_k_percent)
    elected = elect_sign(trimmed)
    sign_matches = torch.sign(trimmed) == elected.unsqueeze(0)
    nonzero = trimmed != 0
    mask = sign_matches & nonzero & (elected.unsqueeze(0) != 0)
    return trimmed * mask


def ties_merge(task_vectors: torch.Tensor, top_k_percent: float = 20, merge_func: str = "dis-sum"):
    trimmed = topk_trim(task_vectors, top_k_percent=top_k_percent)
    elected = elect_sign(trimmed)
    return disjoint_merge(trimmed, elected, merge_func=merge_func)


def merge_encoder_checkpoints(
    zeroshot_path: Path | str,
    finetuned_paths: list[Path | str],
    top_k_percent: float = 20,
    scaling_coef: float = 0.3,
    merge_func: str = "dis-sum",
    map_location="cpu",
):
    pretrained_state = load_state_dict(zeroshot_path, map_location=map_location)
    finetuned_states = [load_state_dict(path, map_location=map_location) for path in finetuned_paths]
    keys = get_merge_keys(pretrained_state, finetuned_states)

    print(f"Flattening {len(finetuned_states)} checkpoints over {len(keys)} floating-point tensors.")
    flat_pretrained = state_dict_to_vector(pretrained_state, keys)
    flat_finetuned = torch.vstack([state_dict_to_vector(state, keys) for state in finetuned_states])

    task_vectors = flat_finetuned - flat_pretrained.unsqueeze(0)
    merged_task_vector = ties_merge(task_vectors, top_k_percent=top_k_percent, merge_func=merge_func)
    merged_vector = flat_pretrained + scaling_coef * merged_task_vector
    return vector_to_state_dict(merged_vector, pretrained_state, keys)
