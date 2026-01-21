import numpy as np
import torch
import hockey.hockey_env as h_env


class HockeyEvaluator:
    def __init__(self, env, agent, action_mapper):
        self.env = env
        self.agent = agent
        self.mapper = action_mapper

    def evaluate(self, num_games=50, opponents=None):
        if opponents is None:
            opponents = [h_env.BasicOpponent(weak=True)]

        results = {"wins": 0, "losses": 0, "ties": 0}
        self.agent.Q.eval()

        for _ in range(num_games):
            obs, _ = self.env.reset()
            opp = np.random.choice(opponents)
            done = False

            while not done:
                with torch.no_grad():
                    a1_idx = self.agent.act(obs, eps=0.0)

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

        self.agent.Q.train()
        return results["wins"] / num_games, results
