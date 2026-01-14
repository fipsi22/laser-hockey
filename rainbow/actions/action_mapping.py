import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GenericDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env, mapper):
        super().__init__(env)
        self.mapper = mapper
        self.action_space = gym.spaces.Discrete(self.mapper.n_actions())

    def action(self, act_idx):
        cont = np.array(self.mapper.to_continuous(int(act_idx)), dtype=np.float32).flatten()
        return cont


class ActionMapper:
    def __init__(self, action_set, keep_mode=True):
        self.action_set = action_set
        self.keep_mode = keep_mode

    def to_continuous(self, action_idx):
        action_idx = int(action_idx)
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


class HockeyDQNWrapper(gym.ActionWrapper):
    def __init__(self, env, keep_mode=True):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8)
        self.keep_mode = keep_mode

    def action(self, act_idx):
        cont = self.env.discrete_to_continous_action(int(act_idx))
        cont = np.array(cont, dtype=np.float32).flatten()
        full_action = np.zeros(8, dtype=np.float32)
        full_action[:len(cont)] = cont
        return full_action
