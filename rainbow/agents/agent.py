import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from util.prioritized_replay_buffer import PrioritizedReplayBuffer
from networks.dueling_network import DuelingDQN



class DQNAgent:
    def __init__(self, observation_space, action_space, mapper=None, **userconfig):
        self._observation_space = observation_space
        self._action_space = action_space
        self.mapper = mapper
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'cuda: {self.device}')
        self._config = {
            "eps": 0.1,  # disabled when using noisy nets
            "use_per": True,
            "discount": 0.99,
            "buffer_size": int(1e6),
            "batch_size": 32,
            "learning_rate": 1e-4,
            "lr_scheduler_steps": [200_000, 500_000],
            "lr_scheduler_factor": 0.5,
            "update_target_every": 2_000,
            "train_every": 4,
            "use_target_net": True,
            "use_double_dqn": True,
            "use_noisy_linear": False,
            "hidden_layers": [512, 512],
            "tau": 0.005,
            "use_soft_update": False,
            "per_alpha": 0.5,
            "per_beta": 0.4,
            "per_beta_inc": 0.000009,
            "n_step": 1,  # 1= Standard DQN
            "use_n_step": False,
            "sequence_length": 8,
            "use_quantile_regression": True,
            "num_quantiles": 32,
            "use_symmetry_augmentation": True,
            "use_symmetry_regularization":False,
            "symmetry_lambda": 0.005,
        }
        self._config.update(userconfig)
        self.use_double_dqn = self._config["use_double_dqn"]
        self.use_noisy = self._config["use_noisy_linear"]
        self.use_quantile_regression= self._config["use_quantile_regression"]
        self.num_quantiles = self._config["num_quantiles"]
        self.buffer = PrioritizedReplayBuffer(
            observation_shape=observation_space.shape[0],
            action_shape=1,
            buffer_size=self._config["buffer_size"],
            device=self.device,
            alpha=self._config["per_alpha"],
            beta=self._config["per_beta"]
        )

        self.Q = DuelingDQN(
            observation_space.shape[0],
            action_space.n,
            hidden_layers=self._config["hidden_layers"],
            use_noisy_linear=self.use_noisy,
            use_quantile_regression=self.use_quantile_regression,
            num_quantiles=self.num_quantiles,
            device=self.device
        ).to(self.device)
        self.Q.to(self.device)

        self.Q_target = DuelingDQN(
            observation_space.shape[0],
            action_space.n,
            hidden_layers=self._config["hidden_layers"],
            use_noisy_linear=self.use_noisy,
            use_quantile_regression=self.use_quantile_regression,
            num_quantiles = self.num_quantiles,
            device=self.device
        ).to(self.device)
        self.Q_target = self.Q_target.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.Q.parameters(), lr=self._config["learning_rate"]
        )

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self._config["lr_scheduler_steps"],
            gamma=self._config["lr_scheduler_factor"]
        )

        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.train_iter = 1

        self.scaling = torch.tensor(
            [1.0, 1.0, 0.5, 4.0, 4.0, 4.0,
             1.0, 1.0, 0.5, 4.0, 4.0, 4.0,
             2.0, 2.0, 10.0, 10.0, 4.0, 4.0],
            device=self.device
        ).float()

        self._update_target_net()

        self.gamma = self._config["discount"]
        if self._config["use_n_step"]:
            self.effective_discount = self.gamma ** self._config["n_step"]
        else:
            self.effective_discount = self.gamma

        self.seq_len = self._config["sequence_length"]
        self.last_hidden = None

        
        self.mirror_action_indices = torch.tensor(
            [self.mapper.get_mirrored_index(i) for i in range(len(self.mapper))],
            device=self.device,
            dtype=torch.long
        )


    def _update_target_net(self):
        self.Q_target.load_state_dict(self.Q.state_dict())

    def _soft_update_target_net(self):
        tau = self._config["tau"]
        for tp, lp in zip(self.Q_target.parameters(), self.Q.parameters()):
            tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)

    def act(self, observation, eps=None, reset_noise=True):
        obs = torch.from_numpy(observation).float().to(self.device).unsqueeze(0) * self.scaling

        if self.use_noisy:
            if reset_noise:
                self.Q.reset_noise()
            eps = eps if eps is not None else 0.0
        else:
            eps = eps if eps is not None else self._config["eps"]

        if np.random.random() > eps:
            with torch.no_grad():                
                q_values = self.Q(obs)
                if self.use_quantile_regression:
                    q_mean = q_values.mean(dim=-1)
                    action = q_mean.argmax(dim=1).item()
                else:
                    action = q_values.argmax(dim=1).item()
            return action
        else:
            return self._action_space.sample()

    def train(self, iter_fit=1, beta=0.4, burn_in=2):
        losses = []
        action_gaps = []
        self.train_iter += 1
        self.buffer.beta = beta

        for _ in range(iter_fit):
            sample = self.buffer.sample(self._config["batch_size"])
            if sample is None: continue

            replay_data, weights_tensor, indices = sample
            weights_tensor = weights_tensor.to(self.device)

            s = replay_data.observations.to(self.device) * self.scaling
            sp = replay_data.next_observations.to(self.device) * self.scaling
            a = replay_data.actions.to(self.device)
            r = replay_data.rewards.to(self.device).unsqueeze(1)
            d = replay_data.dones.to(self.device).unsqueeze(1)
            w = weights_tensor

            perm = torch.as_tensor(self.mirror_action_indices, device=self.device)
            use_aug = self._config.get("use_symmetry_augmentation", False) 
            use_reg = self._config.get("use_symmetry_regularization", False)

            if use_aug:
                with torch.no_grad():
                    s_sym = self.mapper.mirror_obs_tensor(s)
                    sp_sym = self.mapper.mirror_obs_tensor(sp)
                    a_sym = perm[a]
                
                s = torch.cat([s, s_sym], dim=0)
                sp = torch.cat([sp, sp_sym], dim=0)
                a = torch.cat([a, a_sym], dim=0)
                r = torch.cat([r, r], dim=0)
                d = torch.cat([d, d], dim=0)
                w = torch.cat([w, w], dim=0)

            q_out = self.Q(s)

            with torch.no_grad():
                if self.use_quantile_regression:
                    q_mean = q_out.mean(dim=-1)
                    top_two = torch.topk(q_mean, 2, dim=1).values
                    mean_gap = (top_two[:, 0] - top_two[:, 1]).mean()
                else:
                    top_two = torch.topk(q_out, 2, dim=1).values
                    mean_gap = (top_two[:, 0] - top_two[:, 1]).mean()

            action_gaps.append(mean_gap.item())

            with torch.no_grad():
                target_out = self.Q_target(sp)
                if self._config.get("use_double_dqn", True):
                    selection_q = self.Q(sp)
                else:
                    selection_q = target_out
                
                if self.use_quantile_regression:
                    next_actions = selection_q.mean(dim=-1).argmax(dim=1, keepdim=True)
                    next_actions_exp = next_actions.unsqueeze(-1).expand(-1, -1, self.num_quantiles)
                    next_quantiles = target_out.gather(1, next_actions_exp).squeeze(1)
                    td_target = r + self.effective_discount * (1.0 - d) * next_quantiles
                else:
                    next_actions = selection_q.argmax(dim=1, keepdim=True)
                    v_prime = target_out.gather(1, next_actions)
                    td_target = r + self.effective_discount * (1.0 - d) * v_prime

            if self.use_quantile_regression:
                current_quantiles = q_out.gather(1, a.unsqueeze(-1).expand(-1, -1, self.num_quantiles)).squeeze(1)
                td_loss, td_error = self.quantile_huber_loss(current_quantiles, td_target, w)
            else:
                current_q = q_out.gather(1, a)
                elementwise_loss = self.loss_fn(current_q, td_target)
                td_loss = (w * elementwise_loss).mean()
                td_error = elementwise_loss.detach().squeeze(1)

            total_loss = td_loss
            if use_reg:
                if use_aug:
                    batch_size = self._config["batch_size"]
                    q_orig = q_out[:batch_size]
                    q_sym = q_out[batch_size:]
                else:
                    q_orig = q_out
                    q_sym = self.Q(self.mirror_obs_tensor(s))

                q_orig_vec = q_orig.mean(dim=-1) if self.use_quantile_regression else q_orig
                q_sym_vec = q_sym.mean(dim=-1) if self.use_quantile_regression else q_sym
                q_orig_permuted = q_orig_vec.index_select(-1, perm)
                sym_loss = torch.nn.functional.mse_loss(q_sym_vec, q_orig_permuted)
                
                total_loss += self._config.get("symmetry_lambda", 1e-3) * sym_loss


            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 1.0)
            self.optimizer.step()

            priority = td_error[:self._config["batch_size"]].abs().add(1e-6).cpu().numpy()
            self.buffer.update_priorities(indices, priority.flatten())

            losses.append(total_loss.item())

        self.lr_scheduler.step()
        if self._config["use_soft_update"]:
            self._soft_update_target_net()
        elif self.train_iter % self._config["update_target_every"] == 0:
            self._update_target_net()

        final_loss = np.mean(losses) if losses else 0.0
        final_gap = np.mean(action_gaps) if action_gaps else 0.0
        return final_loss, final_gap

    def update_n_step_config(self, use_n_step, n_step):
        self._config["use_n_step"] = use_n_step
        self._config["n_step"] = n_step
        if use_n_step:
            self.effective_discount = self._config["discount"] ** n_step
        else:
            self.effective_discount = self._config["discount"]

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
        self.scaling = self.scaling.to(self.device)
        self.Q.load_state_dict(checkpoint["q_network"])
        self.Q.to(self.device).eval()
        if load_target and "q_target" in checkpoint:
            self.Q_target.load_state_dict(checkpoint["q_target"])
            self.Q_target.to(self.device).eval()

    @property
    def config(self):
        return self._config


    def set_eval_mode(self):
        self.Q.eval()

    def set_train_mode(self):
        self.Q.train()
    

    def quantile_huber_loss(self, pred, target, weights):
        B, N = pred.shape
        taus = torch.linspace(0.0, 1.0, N + 1, device=pred.device)[1:] - 0.5 / N
        taus = taus.view(1, N, 1)
        diff = target.unsqueeze(1) - pred.unsqueeze(2)
        abs_diff = diff.abs()
        kappa = 1.0
        huber = torch.where(abs_diff <= kappa, 0.5 * diff.pow(2), kappa * (abs_diff - 0.5 * kappa))
        quantile_loss = torch.abs(taus - (diff.detach() < 0).float()) * huber
        loss = quantile_loss.mean(dim=1).sum(dim=1)
        weighted_loss = loss * weights.squeeze()
        total_loss = weighted_loss.mean()
        td_error_for_priorities = (target - pred).abs().mean(dim=-1)

        return total_loss, td_error_for_priorities.detach()

