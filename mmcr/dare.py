from pathlib import Path
import gc

import torch

from mmcr.task_vectors import load_state_dict
from mmcr.ties import state_dict_to_vector, vector_to_state_dict


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


def _stream_merge_keys(pretrained_state: dict[str, torch.Tensor], finetuned_paths: list[Path | str], map_location="cpu"):
    keys = [key for key, value in pretrained_state.items() if torch.is_floating_point(value)]
    for path in finetuned_paths:
        state = load_state_dict(path, map_location=map_location)
        kept = []
        for key in keys:
            if key not in state:
                print(f"Warning: {key} is missing from {path}; skipping.", flush=True)
                continue
            if state[key].shape != pretrained_state[key].shape:
                print(f"Warning: {key} has mismatched shape in {path}; skipping.", flush=True)
                continue
            kept.append(key)
        keys = kept
        del state
        gc.collect()
    return keys


def _trim_vector(vector: torch.Tensor, top_k_percent: float) -> torch.Tensor:
    if not 0 < top_k_percent <= 100:
        raise ValueError("top_k_percent must be in (0, 100].")
    if top_k_percent == 100:
        return vector

    keep = max(1, int(vector.numel() * top_k_percent / 100))
    trimmed = torch.zeros_like(vector)
    _, indices = torch.topk(vector.abs(), k=keep, largest=True)
    trimmed[indices] = vector[indices]
    return trimmed


def _make_generator(seed: int | None):
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _drop_and_rescale_vector(vector: torch.Tensor, drop_rate: float, generator: torch.Generator | None):
    if not 0 <= drop_rate < 1:
        raise ValueError("drop_rate must be in [0, 1).")
    if drop_rate == 0:
        return vector

    keep_prob = 1.0 - drop_rate
    mask = torch.rand(vector.shape, generator=generator, device=vector.device) < keep_prob
    return vector * mask.to(dtype=vector.dtype) / keep_prob


def _load_task_vector(path: Path | str, keys: list[str], flat_pretrained: torch.Tensor, map_location="cpu"):
    state = load_state_dict(path, map_location=map_location)
    task_vector = state_dict_to_vector(state, keys).float() - flat_pretrained
    del state
    gc.collect()
    return task_vector



def _dare_task_arithmetic(
    paths: list[Path],
    keys: list[str],
    flat_pretrained: torch.Tensor,
    drop_rate: float,
    seed: int | None,
    map_location="cpu",
):
    generator = _make_generator(seed)
    merged_task_vector = torch.zeros_like(flat_pretrained)
    for index, path in enumerate(paths, start=1):
        print(f"DARE-TA pass {index}/{len(paths)}: {path}", flush=True)
        task_vector = _load_task_vector(path, keys, flat_pretrained, map_location=map_location)
        task_vector = _drop_and_rescale_vector(task_vector, drop_rate, generator)
        merged_task_vector += task_vector
        del task_vector
        gc.collect()
    return merged_task_vector


def _dare_ties(
    paths: list[Path],
    keys: list[str],
    flat_pretrained: torch.Tensor,
    drop_rate: float,
    top_k_percent: float,
    merge_func: str,
    seed: int | None,
    map_location="cpu",
):
    if merge_func not in {"dis-sum", "dis-mean"}:
        raise ValueError("merge_func must be one of: dis-sum, dis-mean")

    sign_generator = _make_generator(seed)
    positive = torch.zeros_like(flat_pretrained)
    negative = torch.zeros_like(flat_pretrained)

    for index, path in enumerate(paths, start=1):
        print(f"DARE-TIES sign pass {index}/{len(paths)}: {path}", flush=True)
        task_vector = _load_task_vector(path, keys, flat_pretrained, map_location=map_location)
        task_vector = _drop_and_rescale_vector(task_vector, drop_rate, sign_generator)
        trimmed = _trim_vector(task_vector, top_k_percent=top_k_percent)
        positive += trimmed.clamp(min=0).abs()
        negative += trimmed.clamp(max=0).abs()
        del task_vector, trimmed
        gc.collect()

    elected = torch.where(positive >= negative, torch.ones_like(positive), -torch.ones_like(negative))
    del positive, negative
    gc.collect()

    select_generator = _make_generator(seed)
    selected_sum = torch.zeros_like(flat_pretrained)
    counts = torch.zeros_like(flat_pretrained, dtype=torch.long) if merge_func == "dis-mean" else None

    for index, path in enumerate(paths, start=1):
        print(f"DARE-TIES select pass {index}/{len(paths)}: {path}", flush=True)
        task_vector = _load_task_vector(path, keys, flat_pretrained, map_location=map_location)
        task_vector = _drop_and_rescale_vector(task_vector, drop_rate, select_generator)
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
    pretrained_state = load_state_dict(zeroshot_path, map_location=map_location)
    paths = [Path(path) for path in finetuned_paths]
    keys = _stream_merge_keys(pretrained_state, paths, map_location=map_location)

    print(f"Streaming DARE-{merge_method.upper()} over {len(paths)} checkpoints and {len(keys)} floating-point tensors.", flush=True)
    flat_pretrained = state_dict_to_vector(pretrained_state, keys).float()

    if merge_method == "ta":
        merged_task_vector = _dare_task_arithmetic(
            paths=paths,
            keys=keys,
            flat_pretrained=flat_pretrained,
            drop_rate=drop_rate,
            seed=seed,
            map_location=map_location,
        )
    elif merge_method == "ties":
        merged_task_vector = _dare_ties(
            paths=paths,
            keys=keys,
            flat_pretrained=flat_pretrained,
            drop_rate=drop_rate,
            top_k_percent=top_k_percent,
            merge_func=merge_func,
            seed=seed,
            map_location=map_location,
        )
    else:
        raise ValueError("merge_method must be one of: ta, ties")

    merged_vector = flat_pretrained + scaling_coef * merged_task_vector
    return vector_to_state_dict(merged_vector, pretrained_state, keys)
