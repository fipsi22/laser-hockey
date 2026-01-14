import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from networks.noisy_layers import FactorizedNoisyLinear


class DuelingQNetwork(nn.Module):
    def __init__(
            self,
            observation_dim: int,
            action_dim: int,
            hidden_layers: list[int] = [128, 128],
            activation: str = "relu",
            use_noisy_linear: bool = False,
    ):
        super().__init__()

        if activation.lower() == "tanh":
            self.activation = nn.Tanh
        elif activation.lower() == "relu":
            self.activation = nn.ReLU
        else:
            raise NotImplementedError(f"Activation '{activation}' not implemented")

        self.use_noisy_linear = use_noisy_linear
        self.hidden_layers = self._build_hidden_layers(observation_dim, hidden_layers)

        # Last hidden layer size
        last_hidden_dim = hidden_layers[-1]

        # Value and Advantage streams
        self.value_stream = self._build_stream(last_hidden_dim, 1)
        self.advantage_stream = self._build_stream(last_hidden_dim, action_dim)

    def _linear(self, in_dim, out_dim):
        if self.use_noisy_linear:
            return FactorizedNoisyLinear(in_dim, out_dim)
        else:
            return nn.Linear(in_dim, out_dim)

    def _build_hidden_layers(self, input_dim, hidden_layers):
        layers = []
        last_dim = input_dim
        for dim in hidden_layers:
            layers.append(self._linear(last_dim, dim))
            layers.append(self.activation())
            last_dim = dim
        return nn.Sequential(*layers)

    def _build_stream(self, input_dim, output_dim):
        return nn.Sequential(
            self._linear(input_dim, input_dim),
            self.activation(),
            self._linear(input_dim, output_dim)
        )

    def reset_noise(self):
        if self.use_noisy_linear:
            for stream in [self.value_stream, self.advantage_stream]:
                for layer in stream:
                    if isinstance(layer, FactorizedNoisyLinear):
                        layer.reset_noise()

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden_layers(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Dueling Q formula
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    # Helper functions
    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if x.ndim == 1:
                x = x[None, :]
            input_tensor = torch.from_numpy(x.astype(np.float32))
            return self.forward(input_tensor).numpy()

    def greedyAction(self, observations: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict(observations), axis=-1)
