import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
from typing import List
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from networks.noisy_layers import FactorizedNoisyLinear


class DuelingDQN(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            hidden_layers: List[int],
            activation: str = "relu",
            use_noisy_linear: bool = True,
            device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.use_noisy_linear = use_noisy_linear

        # Activation
        if activation.lower() == "relu":
            Act = nn.ReLU
        elif activation.lower() == "tanh":
            Act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Feature extractor
        layers = []
        last_dim = observation_dim
        for dim in hidden_layers:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(Act())
            last_dim = dim

        self.feature_extractor = nn.Sequential(*layers)

        # Dueling heads
        Linear = FactorizedNoisyLinear if use_noisy_linear else nn.Linear
        self.value_head = Linear(last_dim, 1)
        self.advantage_head = Linear(last_dim, action_dim)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)

        features = self.feature_extractor(x)

        value = self.value_head(features)
        advantage = self.advantage_head(features)

        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        if not self.use_noisy_linear:
            return
        for m in self.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()
