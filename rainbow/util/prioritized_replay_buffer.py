import torch
import numpy as np
import random
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ReplayData:
    """Structure to hold a batch of transitions."""
    observations: torch.Tensor
    next_observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class PrioritizedReplayBuffer:
    def __init__(
            self,
            observation_shape: int,
            action_shape: int,
            buffer_size: int,
            device: torch.device = torch.device("cpu"),
            alpha: float = 0.6,
            beta: float = 0.4,
    ) -> None:
        self.device = device
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False

        # PER parameters
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

        self.observations = torch.zeros((buffer_size, observation_shape), dtype=torch.float32)
        self.next_observations = torch.zeros((buffer_size, observation_shape), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_shape), dtype=torch.long)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)

        tree_capacity = 1
        while tree_capacity < buffer_size:
            tree_capacity *= 2

        self.sum_tree = np.zeros(2 * tree_capacity - 1)
        self.min_tree = np.full(2 * tree_capacity - 1, float('inf'))
        self.tree_capacity = tree_capacity

    def add(self, obs, next_obs, action, reward, done) -> None:
        idx = self.pos

        self.observations[idx] = torch.as_tensor(obs)
        self.next_observations[idx] = torch.as_tensor(next_obs)
        self.actions[idx] = torch.as_tensor(action)
        self.rewards[idx] = torch.as_tensor(reward)
        self.dones[idx] = torch.as_tensor(done)

        self._update_tree(idx, self.max_priority ** self.alpha)

        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def _update_tree(self, idx: int, priority: float) -> None:
        """Helper to update both sum and min trees."""
        tree_idx = idx + self.tree_capacity - 1
        self.sum_tree[tree_idx] = priority
        self.min_tree[tree_idx] = priority

        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            left = 2 * tree_idx + 1
            right = 2 * tree_idx + 2
            self.sum_tree[tree_idx] = self.sum_tree[left] + self.sum_tree[right]
            self.min_tree[tree_idx] = min(self.min_tree[left], self.min_tree[right])

    def sample(self, batch_size: int) -> Tuple[ReplayData, torch.Tensor, list]:
        total_priority = self.sum_tree[0]
        segment = total_priority / batch_size

        indices = []
        priorities = []

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            value = random.uniform(a, b)

            idx = self._retrieve(value)
            indices.append(idx)
            priorities.append(self.sum_tree[idx + self.tree_capacity - 1])

        current_size = self.buffer_size if self.full else self.pos
        probs = np.array(priorities) / total_priority
        weights = (current_size * probs) ** (-self.beta)

        max_weight = (current_size * (self.min_tree[0] / total_priority)) ** (-self.beta)
        weights = torch.tensor(weights / max_weight, dtype=torch.float32)

        data = ReplayData(
            observations=self.observations[indices],
            next_observations=self.next_observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            dones=self.dones[indices]
        )

        return data, weights, indices

    def _retrieve(self, value: float) -> int:
        """Search the sum tree for the index corresponding to the prefix sum value."""
        idx = 0
        while idx < self.tree_capacity - 1:
            left = 2 * idx + 1
            right = 2 * idx + 2
            if value <= self.sum_tree[left]:
                idx = left
            else:
                value -= self.sum_tree[left]
                idx = right
        return idx - (self.tree_capacity - 1)

    def update_priorities(self, indices: list, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            priority = max(priority, 1e-6)  # Ensure non-zero
            self.max_priority = max(self.max_priority, priority)
            self._update_tree(idx, priority ** self.alpha)