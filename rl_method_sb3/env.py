from __future__ import annotations

from typing import Any

import numpy as np
import torch

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError("rl_method_sb3 requires gymnasium. Install stable-baselines3 first.") from exc


class SB3MMCREnv(gym.Env):
    """Gymnasium adapter for the existing torch-based RL-MMCR environment."""

    metadata = {"render_modes": []}

    def __init__(self, mmcr_env: Any, action_max: float = 2.0):
        super().__init__()
        self.mmcr_env = mmcr_env
        self.action_max = float(action_max)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(mmcr_env.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.zeros(mmcr_env.num_models, dtype=np.float32),
            high=np.full(mmcr_env.num_models, self.action_max, dtype=np.float32),
            dtype=np.float32,
        )
        self.episode_records: list[dict] = []
        self._reset_buffers()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_buffers()
        state = self.mmcr_env.reset()
        return self._to_numpy(state), {}

    def step(self, action):
        coefficients = np.asarray(action, dtype=np.float32).reshape(self.mmcr_env.num_models)
        coefficients = np.clip(coefficients, 0.0, self.action_max)
        selected = torch.ones(self.mmcr_env.num_models, dtype=torch.float32)
        coefficient_tensor = torch.from_numpy(coefficients)

        next_state, reward, done, info = self.mmcr_env.step(selected, coefficient_tensor)
        self._selected_history.append(selected.tolist())
        self._coefficient_history.append(coefficients.tolist())
        self._reward_history.append(float(reward))
        self._info_history.append(info)

        if done:
            self.episode_records.append(self._finalize_episode())

        return self._to_numpy(next_state), float(reward), bool(done), False, info

    def _reset_buffers(self) -> None:
        self._selected_history: list[list[float]] = []
        self._coefficient_history: list[list[float]] = []
        self._reward_history: list[float] = []
        self._info_history: list[dict] = []

    def _finalize_episode(self) -> dict:
        terminal = self._info_history[-1]
        return {
            "selected": list(self._selected_history),
            "coefficients": list(self._coefficient_history),
            "rewards": list(self._reward_history),
            "infos": list(self._info_history),
            "average": float(terminal["average"]),
            "objective": float(terminal["objective"]),
            "scores": terminal["scores"],
            "reward_stats": terminal["reward_stats"],
            "reward_sum": float(sum(self._reward_history)),
        }

    @staticmethod
    def _to_numpy(state: torch.Tensor) -> np.ndarray:
        return state.detach().cpu().numpy().astype(np.float32, copy=False)
