import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, obs_dim, num_actions, hidden_sizes=[128, 128]):
        super(QNetwork, self).__init__()
        layers = []
        last_size = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last_size, h))
            layers.append(nn.ReLU())
            last_size = h
        layers.append(nn.Linear(last_size, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: torch tensor of shape (batch_size, obs_dim)
        returns: Q-values of shape (batch_size, num_actions)
        """
        return self.net(x)
