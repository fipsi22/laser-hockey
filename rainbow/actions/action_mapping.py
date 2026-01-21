import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ActionMapper:
    def __init__(self, action_set, keep_mode=True):
        """
        action_set: list[DiscreteAction]
        """
        self.action_set = action_set
        self.keep_mode = keep_mode

    def __len__(self):
        return len(self.action_set)

    def map(self, action_idx: int) -> np.ndarray:
        if not (0 <= action_idx < len(self.action_set)):
            raise IndexError(f"Invalid action index: {action_idx}")

        a = self.action_set[action_idx]

        if self.keep_mode:
            return np.array(
                [a.dx, a.dy, a.dtheta, a.shoot],
                dtype=np.float32
            )
        else:
            return np.array(
                [a.dx, a.dy, a.dtheta],
                dtype=np.float32
            )
