import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ActionMapper:
    def __init__(self, action_set, keep_mode=True):
        self.action_set = action_set
        self.keep_mode = keep_mode

    def to_continuous(self, action_idx):
        a = self.action_set[action_idx]
        cont = [a.dx, a.dy, a.dtheta]
        if self.keep_mode:
            cont.append(a.shoot)
        return np.array(cont, dtype=np.float32)

    def n_actions(self):
        return len(self.action_set)


class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env, bins=5):
        super().__init__(env)
        self.bins = bins
        self.orig_action_space = env.action_space
        self.action_space = spaces.Discrete(self.bins)

    def action(self, action):
        return self.orig_action_space.low + action / (self.bins - 1.0) * (
                self.orig_action_space.high - self.orig_action_space.low)
