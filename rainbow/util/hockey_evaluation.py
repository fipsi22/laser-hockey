import numpy as np
import torch
import hockey.hockey_env as h_env


class HockeyEvaluator:
    def __init__(self, env, agent, action_mapper, use_quantile=True):
        self.env = env
        self.agent = agent
        self.mapper = action_mapper
        self.use_quantile = use_quantile

    def evaluate(self, num_games=50, opponents=None):
        if opponents is None:
            opponents = [h_env.BasicOpponent(weak=True)]

        results = {"wins": 0, "losses": 0, "ties": 0}
        q_value_list = []
        gap_list = []
        self.agent.Q.eval()

        for _ in range(num_games):
            obs, _ = self.env.reset()
            opp = np.random.choice(opponents)
            done = False

            while not done:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device=self.agent.device, dtype=torch.float32).unsqueeze(0)
                    q_vals = self.agent.Q(obs_t)

                    top_two = torch.topk(q_vals, 2, dim=1).values
                    gap = (top_two[:, 0] - top_two[:, 1]).mean().item()
                    
                    q_value_list.append(q_vals.mean().item())
                    gap_list.append(gap)
                    a1_idx = self.agent.act(obs, eps=0.0, reset_noise=False)

                a1_idx = int(a1_idx.item() if hasattr(a1_idx, "item") else a1_idx)
                a1 = self.mapper.map(a1_idx)

                obs_agent2 = self.env.obs_agent_two()
                a2 = np.array(opp.act(obs_agent2)).flatten()

                combined_action = np.hstack([a1, a2])

                obs, _, terminated, truncated, info = self.env.step(combined_action)
                done = terminated or truncated

            w = info.get("winner", 0)
            if w == 1:
                results["wins"] += 1
            elif w == -1:
                results["losses"] += 1
            else:
                results["ties"] += 1

        eval_stats = {
            "mean_q": np.mean(q_value_list),
            "max_q": np.max(q_value_list),
            "mean_gap": np.mean(gap_list)
        }

        self.agent.Q.train()
        return results["wins"] / num_games, results, eval_stats


