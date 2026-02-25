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
            use_quantile_regression: bool = True,
            num_quantiles: int = 32,
            device: str = "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.num_quantiles = num_quantiles
        self.use_noisy_linear = use_noisy_linear
        self.use_quantile_regression = use_quantile_regression

        if activation.lower() == "relu":
            Act = nn.ReLU
        elif activation.lower() == "tanh":
            Act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

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
        if self.use_quantile_regression:
            self.value_head = Linear(last_dim, num_quantiles)
            self.advantage_head = Linear(last_dim, action_dim * num_quantiles)
        else:
            self.value_head = Linear(last_dim, 1) 
            self.advantage_head = Linear(last_dim, action_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)
        
        if self.use_noisy_linear:
            nn.init.orthogonal_(self.advantage_head.weight_mu, gain=0.01)
            nn.init.orthogonal_(self.value_head.weight_mu, gain=1.0)

        self.to(self.device)

    def forward(self, x: torch.Tensor, return_opp: bool = False):
        device = next(self.parameters()).device
        x = x.to(device)
        features = self.feature_extractor(x)

        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        if self.use_quantile_regression:
            advantage = advantage.view(x.size(0), self.action_dim, self.num_quantiles)
            value = value.view(x.size(0), 1, self.num_quantiles)
            q_quantiles = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_quantiles
        else:
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values

    def reset_noise(self):
        if not self.use_noisy_linear:
            return
        for m in self.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()
