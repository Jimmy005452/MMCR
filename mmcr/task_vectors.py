from pathlib import Path

import torch


def load_state_dict(path: Path | str, map_location="cpu"):
    try:
        state = torch.load(path, map_location=map_location, weights_only=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint: {path}\n"
            "If this is zeroshot.pt, it may be a partial/corrupted file from a failed save. "
            "Delete it and regenerate it."
        ) from exc
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state_dict at {path}, got {type(state)}")
    return state


class TaskVector:
    """Difference between a fine-tuned encoder and the pretrained encoder."""

    def __init__(self, vector: dict[str, torch.Tensor]):
        self.vector = vector

    @classmethod
    def from_checkpoints(cls, pretrained_path: Path | str, finetuned_path: Path | str, map_location="cpu"):
        pretrained = load_state_dict(pretrained_path, map_location=map_location)
        finetuned = load_state_dict(finetuned_path, map_location=map_location)
        vector = {}

        for key, pretrained_value in pretrained.items():
            if key not in finetuned:
                print(f"Warning: {key} is missing from fine-tuned checkpoint.")
                continue
            if not torch.is_floating_point(pretrained_value):
                continue

            finetuned_value = finetuned[key]
            if pretrained_value.shape != finetuned_value.shape:
                print(f"Warning: shape mismatch for {key}; skipping.")
                continue

            vector[key] = finetuned_value - pretrained_value

        return cls(vector)

    def __add__(self, other):
        vector = {}
        for key, value in self.vector.items():
            if key not in other.vector:
                print(f"Warning: {key} is missing from the other task vector.")
                continue
            vector[key] = value + other.vector[key]
        return TaskVector(vector)

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __neg__(self):
        return TaskVector({key: -value for key, value in self.vector.items()})

    def scale(self, coefficient: float):
        return TaskVector({key: coefficient * value for key, value in self.vector.items()})

    @staticmethod
    def weighted_sum(task_vectors, coefficients):
        if len(task_vectors) != len(coefficients):
            raise ValueError("task_vectors and coefficients must have the same length.")

        merged = {}
        for task_vector, coefficient in zip(task_vectors, coefficients):
            for key, value in task_vector.vector.items():
                if key not in merged:
                    merged[key] = coefficient * value
                else:
                    merged[key] = merged[key] + coefficient * value
        return TaskVector(merged)

    def apply_to_state_dict(self, pretrained_state: dict[str, torch.Tensor], scaling_coef: float = 1.0):
        merged_state = {}
        for key, pretrained_value in pretrained_state.items():
            if key in self.vector:
                merged_state[key] = pretrained_value + scaling_coef * self.vector[key]
            else:
                merged_state[key] = pretrained_value
        return merged_state

    def apply_to_checkpoint(self, pretrained_path: Path | str, scaling_coef: float = 1.0, map_location="cpu"):
        pretrained = load_state_dict(pretrained_path, map_location=map_location)
        return self.apply_to_state_dict(pretrained, scaling_coef=scaling_coef)
