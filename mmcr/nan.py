from pathlib import Path

import torch

from mmcr.dare import load_flat_task_vectors
from mmcr.ties import ties_merge, vector_to_state_dict


def compute_nan_coefficients(norms, global_scale: str = "m_half", eps: float = 1e-12):
    """Compute Norm-Aware mergiNg coefficients from per-model norms."""
    if eps <= 0:
        raise ValueError("eps must be positive.")

    norms = torch.as_tensor(norms, dtype=torch.float64)
    if norms.ndim != 1 or norms.numel() == 0:
        raise ValueError("norms must be a non-empty 1D sequence.")
    if not torch.isfinite(norms).all():
        raise ValueError("norms must be finite.")
    if (norms < 0).any():
        raise ValueError("norms must be non-negative.")

    inv_norms = 1.0 / (norms + eps)
    coefficients = inv_norms / inv_norms.sum()

    if global_scale == "m_half":
        coefficients = coefficients * (norms.numel() / 2.0)
    elif global_scale == "none":
        pass
    else:
        raise ValueError("global_scale must be one of: m_half, none")

    return coefficients


def compute_vector_norms(vectors: torch.Tensor):
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D tensor.")
    return torch.linalg.vector_norm(vectors.float(), ord=2, dim=1)


def nan_merge_checkpoints(
    zeroshot_path: Path | str,
    finetuned_paths: list[Path | str],
    scaling_coef: float,
    merge_method: str = "ta",
    top_k_percent: float = 20,
    merge_func: str = "dis-sum",
    norm_target: str = "finetuned",
    global_scale: str = "m_half",
    eps: float = 1e-12,
    map_location="cpu",
):
    pretrained_state, flat_pretrained, task_vectors, keys = load_flat_task_vectors(
        zeroshot_path=zeroshot_path,
        finetuned_paths=finetuned_paths,
        map_location=map_location,
    )

    if norm_target == "finetuned":
        norm_vectors = flat_pretrained.unsqueeze(0) + task_vectors
    elif norm_target == "task-vector":
        norm_vectors = task_vectors
    else:
        raise ValueError("norm_target must be one of: finetuned, task-vector")

    norms = compute_vector_norms(norm_vectors)
    coefficients = compute_nan_coefficients(norms, global_scale=global_scale, eps=eps).to(
        dtype=task_vectors.dtype,
        device=task_vectors.device,
    )
    weighted_task_vectors = coefficients.unsqueeze(1) * task_vectors

    if merge_method == "ta":
        merged_task_vector = weighted_task_vectors.sum(dim=0)
    elif merge_method == "ties":
        merged_task_vector = ties_merge(
            weighted_task_vectors,
            top_k_percent=top_k_percent,
            merge_func=merge_func,
        )
    else:
        raise ValueError("merge_method must be one of: ta, ties")

    merged_vector = flat_pretrained + scaling_coef * merged_task_vector
    metadata = {
        "merge_method": merge_method,
        "norm_target": norm_target,
        "global_scale": global_scale,
        "scale": scaling_coef,
        "top_k": top_k_percent if merge_method == "ties" else None,
        "merge_func": merge_func if merge_method == "ties" else None,
        "eps": eps,
        "num_models": len(finetuned_paths),
        "num_tensors": len(keys),
        "num_parameters": int(flat_pretrained.numel()),
        "norms": [float(value) for value in norms.tolist()],
        "coefficients": [float(value) for value in coefficients.tolist()],
    }
    return vector_to_state_dict(merged_vector, pretrained_state, keys), metadata
