"""PPO-GAE Actor-Critic RL-MMCR package built on the shared ``mmcr`` modules."""

__all__ = [
    "PositiveActorCritic",
    "LayeredTaskVectors",
    "RLMMCREnv",
    "coefficients_to_dict",
    "load_layered_ties_task_vectors",
    "merge_state_with_layer_coefficients",
]


def __getattr__(name: str):
    if name == "RLMMCREnv":
        from .env import RLMMCREnv

        return RLMMCREnv
    if name == "PositiveActorCritic":
        from .policy import PositiveActorCritic

        return PositiveActorCritic
    if name in {
        "LayeredTaskVectors",
        "coefficients_to_dict",
        "load_layered_ties_task_vectors",
        "merge_state_with_layer_coefficients",
    }:
        from . import merge

        return getattr(merge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
