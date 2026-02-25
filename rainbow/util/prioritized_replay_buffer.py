import numpy as np
import torch
import random
from dataclasses import dataclass


# SegmentTrees based on the Segment Tree implementation from OpenAI Baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
# Used for efficient Prioritized Experience Replay (PER) in Reinforcement Learning.

class SumSegmentTree:
    def __init__(self, capacity: int):
        assert capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)

    def __setitem__(self, idx: int, value: float):
        tree = self.tree
        idx += self.capacity
        tree[idx] = value
        idx >>= 1
        while idx:
            tree[idx] = tree[idx << 1] + tree[(idx << 1) + 1]
            idx >>= 1

    def __getitem__(self, idx: int) -> float:
        return self.tree[idx + self.capacity]

    def total(self) -> float:
        return self.tree[1]

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        tree = self.tree
        idx = 1
        capacity = self.capacity
        while idx < capacity:
            left = idx << 1
            if tree[left] >= prefixsum:
                idx = left
            else:
                prefixsum -= tree[left]
                idx = left + 1
        return idx - capacity


class MinSegmentTree:
    def __init__(self, capacity: int):
        assert capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.tree = np.full(2 * capacity, np.inf, dtype=np.float32)

    def __setitem__(self, idx: int, value: float):
        tree = self.tree
        idx += self.capacity
        tree[idx] = value
        idx >>= 1
        while idx:
            tree[idx] = min(tree[idx << 1], tree[(idx << 1) + 1])
            idx >>= 1

    def min(self) -> float:
        return self.tree[1]




@dataclass
class ReplayData:
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
            device: torch.device,
            alpha: float = 0.6,
            beta: float = 0.4,
    ):
        self.device = device
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

        self.pos = 0
        self.full = False

        self.observations = torch.zeros((buffer_size, observation_shape), dtype=torch.float32)
        self.next_observations = torch.zeros((buffer_size, observation_shape), dtype=torch.float32)
        self.actions = torch.zeros((buffer_size, action_shape), dtype=torch.long)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)

        tree_capacity = 1
        while tree_capacity < buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            priority = float(priority)
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

    def add(self, obs, next_obs, action, reward, done, priority_override=None):
        idx = self.pos

        self.observations[idx] = torch.as_tensor(obs, dtype=torch.float32)
        self.next_observations[idx] = torch.as_tensor(next_obs, dtype=torch.float32)
        self.actions[idx] = torch.as_tensor(action, dtype=torch.long)
        self.rewards[idx] = reward
        self.dones[idx] = float(done)

        if priority_override is not None:
            priority = priority_override
        else:
            priority = self.max_priority ** self.alpha

        self.sum_tree[idx] = priority
        self.min_tree[idx] = priority

        self.pos = (self.pos + 1) % self.buffer_size
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size: int):
        end = self.buffer_size if self.full else self.pos
        if end == 0: return None
        total_priority = self.sum_tree.total()
        segment = total_priority / batch_size

        indices = []
        priorities = []

        for i in range(batch_size):
            prefixsum = (i + random.random()) * segment
            idx = self.sum_tree.find_prefixsum_idx(prefixsum)
            idx = min(idx, end - 1)
            indices.append(idx)
            priorities.append(self.sum_tree[idx])

        probs = np.array(priorities) / (total_priority + 1e-8)
        weights = (probs * (end + 1)) ** (-self.beta)

        min_prob = self.min_tree.min() / total_priority
        max_weight = (min_prob * (end + 1)) ** (-self.beta)
        weights /= max_weight

        weights = torch.tensor(weights, device=self.device, dtype=torch.float32)

        batch = ReplayData(
            observations=self.observations[indices].to(self.device),
            next_observations=self.next_observations[indices].to(self.device),
            actions=self.actions[indices].to(self.device),
            rewards=self.rewards[indices].to(self.device),
            dones=self.dones[indices].to(self.device),
        )

        return batch, weights, indices

    def log_per_stats(self):
        if self.full:
            size = self.buffer_size
        else:
            size = self.pos
        all_priorities = np.array([self.sum_tree[i] for i in range(size)])
        print(
            f"PER stats — min: {all_priorities.min():.4f}, mean: {all_priorities.mean():.4f}, max: {all_priorities.max():.4f}")
