from pathlib import Path

import torch

from mmcr.task_vectors import load_state_dict


def get_common_float_keys(states: list[dict[str, torch.Tensor]]):
    """Return keys that are floating tensors with the same shape in every state_dict."""
    common_keys = set(states[0].keys())
    for state in states[1:]:
        common_keys &= set(state.keys())

    keys = []
    for key in sorted(common_keys):
        first = states[0][key]
        if not torch.is_floating_point(first):
            continue
        if all(torch.is_floating_point(state[key]) and state[key].shape == first.shape for state in states[1:]):
            keys.append(key)
    return keys


def task_vector_similarity(zeroshot_path: Path | str, encoder_paths: list[Path | str], map_location="cpu"):
    zeroshot = load_state_dict(zeroshot_path, map_location=map_location)
    finetuned_states = [load_state_dict(path, map_location=map_location) for path in encoder_paths]
    keys = get_common_float_keys([zeroshot, *finetuned_states])
    if not keys:
        raise ValueError("No common floating-point tensors were found.")

    count = len(finetuned_states)
    dots = torch.zeros((count, count), dtype=torch.float64)
    norm_squares = torch.zeros(count, dtype=torch.float64)

    for key in keys:
        base = zeroshot[key].detach().float().cpu()
        deltas = [(state[key].detach().float().cpu() - base).reshape(-1) for state in finetuned_states]

        for i, delta_i in enumerate(deltas):
            norm_squares[i] += torch.dot(delta_i, delta_i).double()
            for j in range(i, count):
                dots[i, j] += torch.dot(delta_i, deltas[j]).double()

    dots = dots + dots.T - torch.diag(torch.diag(dots))
    norms = torch.sqrt(norm_squares)
    denominator = norms[:, None] * norms[None, :]
    similarities = torch.zeros_like(dots)
    valid = denominator > 0
    similarities[valid] = dots[valid] / denominator[valid]

    return similarities.float(), norms.tolist(), keys
