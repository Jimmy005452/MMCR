from pathlib import Path

import torch

from mmcr.task_vectors import load_state_dict
from mmcr.ties import get_merge_keys, state_dict_to_vector, ties_merge, vector_to_state_dict


def drop_and_rescale(task_vectors: torch.Tensor, drop_rate: float, seed: int | None = None):
    """Apply DARE random drop and rescale to flattened task vectors."""
    if not 0 <= drop_rate < 1:
        raise ValueError("drop_rate must be in [0, 1).")
    if drop_rate == 0:
        return task_vectors

    keep_prob = 1.0 - drop_rate
    generator = None
    if seed is not None:
        generator = torch.Generator(device=task_vectors.device)
        generator.manual_seed(seed)

    mask = torch.rand(task_vectors.shape, generator=generator, device=task_vectors.device) < keep_prob
    return task_vectors * mask.to(dtype=task_vectors.dtype) / keep_prob


def load_flat_task_vectors(zeroshot_path: Path | str, finetuned_paths: list[Path | str], map_location="cpu"):
    pretrained_state = load_state_dict(zeroshot_path, map_location=map_location)
    finetuned_states = [load_state_dict(path, map_location=map_location) for path in finetuned_paths]
    keys = get_merge_keys(pretrained_state, finetuned_states)

    print(f"Flattening {len(finetuned_states)} checkpoints over {len(keys)} floating-point tensors.")
    flat_pretrained = state_dict_to_vector(pretrained_state, keys)
    flat_finetuned = torch.vstack([state_dict_to_vector(state, keys) for state in finetuned_states])
    task_vectors = flat_finetuned - flat_pretrained.unsqueeze(0)
    return pretrained_state, flat_pretrained, task_vectors, keys


def dare_merge_checkpoints(
    zeroshot_path: Path | str,
    finetuned_paths: list[Path | str],
    drop_rate: float,
    scaling_coef: float,
    merge_method: str = "ta",
    top_k_percent: float = 20,
    merge_func: str = "dis-sum",
    seed: int | None = None,
    map_location="cpu",
):
    pretrained_state, flat_pretrained, task_vectors, keys = load_flat_task_vectors(
        zeroshot_path=zeroshot_path,
        finetuned_paths=finetuned_paths,
        map_location=map_location,
    )
    dared_task_vectors = drop_and_rescale(task_vectors, drop_rate=drop_rate, seed=seed)

    if merge_method == "ta":
        merged_task_vector = dared_task_vectors.sum(dim=0)
    elif merge_method == "ties":
        merged_task_vector = ties_merge(
            dared_task_vectors,
            top_k_percent=top_k_percent,
            merge_func=merge_func,
        )
    else:
        raise ValueError("merge_method must be one of: ta, ties")

    merged_vector = flat_pretrained + scaling_coef * merged_task_vector
    return vector_to_state_dict(merged_vector, pretrained_state, keys)
