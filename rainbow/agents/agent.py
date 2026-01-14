import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import util.memory as mem
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
        self._config = {
            "eps": 0.5, "discount": 0.95, "buffer_size": int(1e5),
            "batch_size": 128, "learning_rate": 0.0002,
            "update_target_every": 20, "use_target_net": True,
            "use_double_dqn": True, "use_noisy_linear": True,
            "hidden_layers": [265, 265], "tau": 0.005, "use_soft_update": True
        }
        self._config.update(userconfig)
        self.use_noisy = self._config.get("use_noisy_linear", False)
        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        self.Q = DuelingQNetwork(observation_space.shape[0], action_space.n,
                                 hidden_layers=self._config["hidden_layers"],
                                 activation="ReLu", use_noisy_linear=self.use_noisy)
        self.Q_target = DuelingQNetwork(observation_space.shape[0], action_space.n,
                                        hidden_layers=self._config["hidden_layers"],
                                        activation="ReLu", use_noisy_linear=self.use_noisy)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self._config["learning_rate"])
        self.loss_fn = nn.SmoothL1Loss()
        self.train_iter = 0
        self._update_target_net()
        self.scaling = torch.tensor([1.0, 1.0, 0.5, 4.0, 4.0, 4.0,
                                     1.0, 1.0, 0.5, 4.0, 4.0, 4.0,
                                     2.0, 2.0, 10.0, 10.0, 4.0, 4.0]).float()

    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def _soft_update_target_net(self):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        tau = self._config["tau"]
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def act(self, observation, eps=None):
        if self.use_noisy:
            self.Q.reset_noise()
            eps = 0.0
        else:
            eps = eps if eps is not None else self._config['eps']
        if np.random.random() > eps:
            return self.Q.greedyAction(observation)
        return self._action_space.sample()

    def train(self, iter_fit=32):
        losses = []
        self.train_iter += 1
        if not self._config.get("use_soft_update", False):
            if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
                self._update_target_net()

        for _ in range(iter_fit):
            if self.use_noisy:
                self.Q.reset_noise()
                self.Q_target.reset_noise()
            data = self.buffer.sample(batch=self._config['batch_size'])

            s = np.array(list(data[:, 0]), dtype=np.float32)
            a = np.array([int(x) for x in data[:, 1]], dtype=np.int64)
            rew = np.array(list(data[:, 2]), dtype=np.float32)
            s_p = np.array(list(data[:, 3]), dtype=np.float32)
            done = np.array(list(data[:, 4]), dtype=np.float32)

            # Convert to Tensors
            s_tensor = torch.from_numpy(s).float()
            s_tensor = s_tensor * self.scaling
            a_tensor = torch.from_numpy(a).long()
            rew_tensor = torch.from_numpy(rew).float()[:, None]
            sp_tensor = torch.from_numpy(s_p).float()
            sp_tensor = sp_tensor * self.scaling
            done_tensor = torch.from_numpy(done).float()[:, None]

            # Double DQN: Q-online to pick action, Q-target to evaluate
            with torch.no_grad():
                if self._config["use_double_dqn"]:
                    next_actions = self.Q(sp_tensor).argmax(dim=1, keepdim=True)
                    v_prime = self.Q_target(sp_tensor).gather(1, next_actions)
                else:
                    v_prime = self.Q_target(sp_tensor).max(dim=1, keepdim=True)[0]

                td_target = rew_tensor + self._config['discount'] * (1.0 - done_tensor) * v_prime

            # Optimization Step
            self.Q.train()
            self.optimizer.zero_grad()
            current_q = self.Q(s_tensor).gather(1, a_tensor[:, None])
            loss = self.loss_fn(current_q, td_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self._config.get("use_soft_update", False):
                self._soft_update_target_net()

            losses.append(loss.item())
        return losses

    def save(self, path: str):
        torch.save({
            "q_network": self.Q.state_dict(),
            "q_target": self.Q_target.state_dict(),
            "config": self._config,
        }, path)

    def load(self, path: str, load_target: bool = True):
        checkpoint = torch.load(path, map_location="cpu")

        self.Q.load_state_dict(checkpoint["q_network"])
        self.Q.eval()

        if load_target and "q_target" in checkpoint:
            self.Q_target.load_state_dict(checkpoint["q_target"])
            self.Q_target.eval()
