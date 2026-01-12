import numpy as np
import torch
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
            "eps": 0.05, "discount": 0.95, "buffer_size": int(1e5),
            "batch_size": 128, "learning_rate": 0.0002,
            "update_target_every": 20, "use_target_net": True,
            "use_double_dqn": True
        }
        self._config.update(userconfig)
        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        self.Q = DuelingQNetwork(observation_space.shape[0], action_space.n, learning_rate=self._config["learning_rate"])
        self.Q_target = DuelingQNetwork(observation_space.shape[0], action_space.n, learning_rate=0)
        self.train_iter = 0
        self._update_target_net()

    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, observation, eps=None):
        eps = eps if eps is not None else self._config['eps']
        if np.random.random() > eps:
            return self.Q.greedyAction(observation)
        return self._action_space.sample()

    def train(self, iter_fit=32):
        losses = []
        self.train_iter += 1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net()

        for _ in range(iter_fit):
            data = self.buffer.sample(batch=self._config['batch_size'])

            s = np.array(list(data[:, 0]), dtype=np.float32)
            a = np.array([int(x) for x in data[:, 1]], dtype=np.int64)
            rew = np.array(list(data[:, 2]), dtype=np.float32)
            s_p = np.array(list(data[:, 3]), dtype=np.float32)
            done = np.array(list(data[:, 4]), dtype=np.float32)

            # Convert to Tensors
            s_tensor = torch.from_numpy(s).float()
            a_tensor = torch.from_numpy(a).long()
            rew_tensor = torch.from_numpy(rew).float()[:, None]
            sp_tensor = torch.from_numpy(s_p).float()
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
            self.Q.optimizer.zero_grad()
            current_q = self.Q(s_tensor).gather(1, a_tensor[:, None])
            loss = self.Q.loss_fn(current_q, td_target)
            loss.backward()
            self.Q.optimizer.step()

            losses.append(loss.item())
        return losses


