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
            hidden_layers: list[int] = [256, 256],
            activation: str = "relu",
            use_noisy_linear: bool = False,
            device: str = "cpu"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.use_noisy_linear = use_noisy_linear
        self.activation_fn = nn.ReLU if activation.lower() == "relu" else nn.Tanh

        layers = []
        last_dim = observation_dim
        for dim in hidden_layers:
            layers.append(nn.Linear(last_dim, dim))
            layers.append(self.activation_fn())
            last_dim = dim
        self.feature_extractor = nn.Sequential(*layers)

        self.value_stream = self._build_stream(last_dim, 1, use_noisy_linear)
        self.advantage_stream = self._build_stream(last_dim, action_dim, use_noisy_linear)

        scaling_values = [1.0, 1.0, 0.5, 4.0, 4.0, 4.0,
                          1.0, 1.0, 0.5, 4.0, 4.0, 4.0,
                          2.0, 2.0, 10.0, 10.0, 4.0, 4.0]
        self.register_buffer("scaling", torch.tensor(scaling_values).float())

        self.to(self.device)

    def _linear(self, in_dim, out_dim):
        if self.use_noisy_linear:
            return FactorizedNoisyLinear(in_dim, out_dim)
        return nn.Linear(in_dim, out_dim)

    def _build_hidden_layers(self, input_dim, hidden_layers):
        layers = []
        last_dim = input_dim
        for dim in hidden_layers:
            layers.append(self._linear(last_dim, dim))
            layers.append(self.activation())
            last_dim = dim
        return nn.Sequential(*layers)

    def _build_stream(self, input_dim, output_dim, use_noisy):
        def linear_layer(in_f, out_f):
            return FactorizedNoisyLinear(in_f, out_f) if use_noisy else nn.Linear(in_f, out_f)

        return nn.Sequential(
            linear_layer(input_dim, input_dim),
            self.activation_fn(),
            linear_layer(input_dim, output_dim)
        )

    def reset_noise(self):
        if self.use_noisy_linear:
            for m in self.modules():
                if isinstance(m, FactorizedNoisyLinear):
                    m.reset_noise()

    def forward(self, x: Tensor) -> Tensor:
        if x.device != self.device:
            x = x.to(self.device)

        if hasattr(self, "scaling"):
            x = x * self.scaling

        x = self.feature_extractor(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            if x.ndim == 1:
                x = x[None, :]
            if isinstance(x, np.ndarray):
                input_tensor = torch.from_numpy(x).float().to(self.device)
            else:
                input_tensor = x.float().to(self.device)
            # input_tensor = input_tensor * scaling_tensor.to(self.device)

            return self.forward(input_tensor).cpu().numpy()

    def greedyAction(self, observations: np.ndarray, scaling_tensor: Tensor) -> int:
        q_values = self.predict(observations)
        return int(np.argmax(q_values, axis=-1))
