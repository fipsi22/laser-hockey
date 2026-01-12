import torch
import torch.nn as nn
import numpy as np


class DuelingQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[128, 128], learning_rate=0.0002):
        super().__init__()

        # Feature extractor (shared by both streams)
        self.feature_layer = nn.Sequential(
            nn.Linear(observation_dim, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh()
        )

        # Value stream: V(s) - How good is the state?
        self.value_stream = nn.Linear(hidden_sizes[1], 1)

        # Advantage stream: A(s, a) - How much better is this action than others?
        self.advantage_stream = nn.Linear(hidden_sizes[1], action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            # Handle single observations by adding a batch dimension
            if x.ndim == 1:
                x = x[None, :]  # Change shape from (N,) to (1, N)

            # Convert to tensor and run forward pass
            input_tensor = torch.from_numpy(x.astype(np.float32))
            result = self.forward(input_tensor).numpy()

            return result

    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)