import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import util.memory as mem
from util.prioritized_replay_buffer import PrioritizedReplayBuffer
from networks.feed_forward import Feedforward
from networks.dueling_network import DuelingQNetwork


class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100, 100], learning_rate=0.0002):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes, output_size=action_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-6)
        self.loss = torch.nn.SmoothL1Loss()

    def fit(self, observations, actions, targets):
        self.train()
        self.optimizer.zero_grad()
        acts = torch.from_numpy(actions)
        pred = self.forward(torch.from_numpy(observations).float()).gather(1, acts[:, None])
        loss = self.loss(pred, torch.from_numpy(targets).float())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def maxQ(self, observations):
        return np.max(self.predict(observations), axis=-1, keepdims=True)

    def greedyAction(self, observations):
        return np.argmax(self.predict(observations), axis=-1)


class DQNAgent:
    def __init__(self, observation_space, action_space, **userconfig):
        self._observation_space = observation_space
        self._action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._config = {
            "eps": 0.0,  # disabled when using noisy nets
            "use_per": True,
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 32,
            "learning_rate": 1e-4,
            "update_target_every": 2_000,
            "train_every": 4,
            "use_target_net": True,
            "use_double_dqn": True,
            "use_noisy_linear": False,
            "hidden_layers": [512, 512],
            "tau": 0.005,
            "use_soft_update": False,
            "per_alpha": 0.6,
            "per_beta": 0.4,
            "per_beta_inc": 0.000009
        }
        self._config.update(userconfig)

        self.use_noisy = self._config["use_noisy_linear"]


        self.buffer = PrioritizedReplayBuffer(
            observation_shape=observation_space.shape[0],
            action_shape=1,
            buffer_size=self._config["buffer_size"],
            device=self.device,  # Optional: helps if you want to store tensors on GPU directly
            alpha=self._config.get("alpha", 0.6),
            beta=self._config.get("beta", 0.4)
        )


        self.Q = DuelingQNetwork(
            observation_space.shape[0],
            action_space.n,
            hidden_layers=self._config["hidden_layers"],
            activation="ReLU",
            use_noisy_linear=self.use_noisy
        ).to(self.device)

        self.Q_target = DuelingQNetwork(
            observation_space.shape[0],
            action_space.n,
            hidden_layers=self._config["hidden_layers"],
            activation="ReLU",
            use_noisy_linear=self.use_noisy
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.Q.parameters(), lr=self._config["learning_rate"]
        )

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[6000, 10000, 15_000],
            gamma=0.5
        )

        '''self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5
        )'''

        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.train_iter = 0

        self.scaling = torch.tensor(
            [1.0, 1.0, 0.5, 4.0, 4.0, 4.0,
             1.0, 1.0, 0.5, 4.0, 4.0, 4.0,
             2.0, 2.0, 10.0, 10.0, 4.0, 4.0],
            device=self.device
        ).float()

        self._update_target_net()

    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def _soft_update_target_net(self):
        tau = self._config["tau"]
        for tp, lp in zip(self.Q_target.parameters(), self.Q.parameters()):
            tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)

    def act(self, observation, eps=None):
        obs = torch.from_numpy(observation).float().to(self.device)
        if self.use_noisy:
            self.Q.reset_noise()
            eps = 0.0
        else:
            eps = eps if eps is not None else self._config["eps"]
        if np.random.random() > eps:
            return self.Q.greedyAction(obs, self.scaling)
        else:
            return self._action_space.sample()

    def train(self, iter_fit=32, beta=0.4):
        losses = []
        self.train_iter += 1

        # 1. Target Network Update Logic
        if not self._config["use_soft_update"]:
            if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
                self._update_target_net()

        for _ in range(iter_fit):
            # 2. Optimized Sampling
            # The buffer now returns a ReplayData object, weights (Tensor), and indices (list)
            # Note: I'm passing beta directly to the buffer's sample method
            self.buffer.beta = beta
            replay_data, weights_tensor, indices = self.buffer.sample(self._config["batch_size"])

            # Move all tensors to the correct device at once
            s = replay_data.observations.to(self.device)
            a = replay_data.actions.to(self.device)
            r = replay_data.rewards.to(self.device).unsqueeze(1)
            sp = replay_data.next_observations.to(self.device)
            d = replay_data.dones.to(self.device).unsqueeze(1)
            weights_tensor = weights_tensor.to(self.device)

            if self.use_noisy:
                self.Q.reset_noise()
                self.Q_target.reset_noise()

            # 3. Compute TD Target
            with torch.no_grad():
                if self._config["use_double_dqn"]:
                    # Double DQN: Use Q-net to select action, Target-net to evaluate it
                    next_actions = self.Q(sp).argmax(dim=1, keepdim=True)
                    v_prime = self.Q_target(sp).gather(1, next_actions)
                else:
                    # Standard DQN: Max over Target-net actions
                    v_prime = self.Q_target(sp).max(dim=1, keepdim=True)[0]

                td_target = r + self._config["discount"] * (1.0 - d) * v_prime

            self.optimizer.zero_grad()
            current_q = self.Q(s).gather(1, a)

            elementwise_loss = self.loss_fn(current_q, td_target)

            loss = (weights_tensor.unsqueeze(1) * elementwise_loss).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 1.0)
            self.optimizer.step()

            td_error = (current_q - td_target).abs().detach().cpu().numpy().flatten()
            self.buffer.update_priorities(indices, td_error)

            losses.append(loss.item())

        if self._config["use_soft_update"]:
            self._soft_update_target_net()

        return losses

    def save(self, path):
        torch.save(
            {
                "q_network": self.Q.state_dict(),
                "q_target": self.Q_target.state_dict(),
                "config": self._config,
            },
            path
        )

    def load(self, path, load_target=True):
        checkpoint = torch.load(path, map_location=self.device)
        self.Q.load_state_dict(checkpoint["q_network"])
        self.Q.to(self.device).eval()
        if load_target and "q_target" in checkpoint:
            self.Q_target.load_state_dict(checkpoint["q_target"])
            self.Q_target.to(self.device).eval()

    @property
    def config(self):
        return self._config
