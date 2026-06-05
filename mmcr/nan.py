from pathlib import Path
import gc

import torch

from mmcr.dare import _load_task_vector, _stream_merge_keys, _trim_vector
from mmcr.task_vectors import load_state_dict
from mmcr.ties import state_dict_to_vector, vector_to_state_dict


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


def _compute_streaming_norms(
    paths: list[Path],
    keys: list[str],
    flat_pretrained: torch.Tensor,
    norm_target: str,
    map_location="cpu",
):
    norms = []
    for index, path in enumerate(paths, start=1):
        print(f"NAN norm pass {index}/{len(paths)}: {path}", flush=True)
        task_vector = _load_task_vector(path, keys, flat_pretrained, map_location=map_location)
        if norm_target == "finetuned":
            norm_vector = flat_pretrained + task_vector
        elif norm_target == "task-vector":
            norm_vector = task_vector
        else:
            raise ValueError("norm_target must be one of: finetuned, task-vector")
        norms.append(float(torch.linalg.vector_norm(norm_vector.float(), ord=2).item()))
        del task_vector, norm_vector
        gc.collect()
    return torch.tensor(norms, dtype=torch.float64)


def _weighted_task_vector(
    path: Path,
    keys: list[str],
    flat_pretrained: torch.Tensor,
    coefficient: torch.Tensor,
    map_location="cpu",
):
    task_vector = _load_task_vector(path, keys, flat_pretrained, map_location=map_location)
    return task_vector * coefficient.to(dtype=task_vector.dtype, device=task_vector.device)


def _nan_task_arithmetic(
    paths: list[Path],
    keys: list[str],
    flat_pretrained: torch.Tensor,
    coefficients: torch.Tensor,
    map_location="cpu",
):
    merged_task_vector = torch.zeros_like(flat_pretrained)
    for index, (path, coefficient) in enumerate(zip(paths, coefficients), start=1):
        print(f"NAN-TA merge pass {index}/{len(paths)}: {path}", flush=True)
        task_vector = _weighted_task_vector(path, keys, flat_pretrained, coefficient, map_location=map_location)
        merged_task_vector += task_vector
        del task_vector
        gc.collect()
    return merged_task_vector


def _nan_ties(
    paths: list[Path],
    keys: list[str],
    flat_pretrained: torch.Tensor,
    coefficients: torch.Tensor,
    top_k_percent: float,
    merge_func: str,
    map_location="cpu",
):
    if merge_func not in {"dis-sum", "dis-mean"}:
        raise ValueError("merge_func must be one of: dis-sum, dis-mean")

    positive = torch.zeros_like(flat_pretrained)
    negative = torch.zeros_like(flat_pretrained)
    for index, (path, coefficient) in enumerate(zip(paths, coefficients), start=1):
        print(f"NAN-TIES sign pass {index}/{len(paths)}: {path}", flush=True)
        task_vector = _weighted_task_vector(path, keys, flat_pretrained, coefficient, map_location=map_location)
        trimmed = _trim_vector(task_vector, top_k_percent=top_k_percent)
        positive += trimmed.clamp(min=0).abs()
        negative += trimmed.clamp(max=0).abs()
        del task_vector, trimmed
        gc.collect()

    elected = torch.where(positive >= negative, torch.ones_like(positive), -torch.ones_like(negative))
    del positive, negative
    gc.collect()

    selected_sum = torch.zeros_like(flat_pretrained)
    counts = torch.zeros_like(flat_pretrained, dtype=torch.long) if merge_func == "dis-mean" else None
    for index, (path, coefficient) in enumerate(zip(paths, coefficients), start=1):
        print(f"NAN-TIES select pass {index}/{len(paths)}: {path}", flush=True)
        task_vector = _weighted_task_vector(path, keys, flat_pretrained, coefficient, map_location=map_location)
        trimmed = _trim_vector(task_vector, top_k_percent=top_k_percent)
        mask = (torch.sign(trimmed) == elected) & (trimmed != 0) & (elected != 0)
        selected_sum += trimmed * mask
        if counts is not None:
            counts += mask.to(dtype=counts.dtype)
        del task_vector, trimmed, mask
        gc.collect()

    if counts is not None:
        selected_sum = selected_sum / counts.clamp(min=1).to(dtype=selected_sum.dtype)
    del elected, counts
    gc.collect()
    return selected_sum


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
    pretrained_state = load_state_dict(zeroshot_path, map_location=map_location)
    paths = [Path(path) for path in finetuned_paths]
    keys = _stream_merge_keys(pretrained_state, paths, map_location=map_location)

    print(f"Streaming NAN-{merge_method.upper()} over {len(paths)} checkpoints and {len(keys)} floating-point tensors.", flush=True)
    flat_pretrained = state_dict_to_vector(pretrained_state, keys).float()

    norms = _compute_streaming_norms(
        paths=paths,
        keys=keys,
        flat_pretrained=flat_pretrained,
        norm_target=norm_target,
        map_location=map_location,
    )
    coefficients = compute_nan_coefficients(norms, global_scale=global_scale, eps=eps)

    if merge_method == "ta":
        merged_task_vector = _nan_task_arithmetic(
            paths=paths,
            keys=keys,
            flat_pretrained=flat_pretrained,
            coefficients=coefficients,
            map_location=map_location,
        )
    elif merge_method == "ties":
        merged_task_vector = _nan_ties(
            paths=paths,
            keys=keys,
            flat_pretrained=flat_pretrained,
            coefficients=coefficients,
            top_k_percent=top_k_percent,
            merge_func=merge_func,
            map_location=map_location,
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
        "num_models": len(paths),
        "num_tensors": len(keys),
        "num_parameters": int(flat_pretrained.numel()),
        "norms": [float(value) for value in norms.tolist()],
        "coefficients": [float(value) for value in coefficients.tolist()],
    }
    return vector_to_state_dict(merged_vector, pretrained_state, keys), metadata
