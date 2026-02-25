from __future__ import annotations

import argparse
import uuid

import gymnasium as gym
import hockey.hockey_env as h_env
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from comprl.client import Agent, launch_client
from actions.action_mapping import ActionMapper
from actions.action_sets import baseline_action_set, compound_action_set
from agents.agent import DQNAgent


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        super().__init__()

        self.mapper = ActionMapper(compound_action_set(), keep_mode=True)
        env = h_env.HockeyEnv()
        self.model = DQNAgent(
            observation_space=env.observation_space,
            action_space=gym.spaces.Discrete(len(self.mapper)),
            mapper=self.mapper,
            hidden_layers=[128, 256, 256], use_noisy_linear=True)
        
        checkpoint = torch.load(model_path, map_location=self.model.device)
        if isinstance(checkpoint, dict) and "q_network" in checkpoint:
            self.model.Q.load_state_dict(checkpoint["q_network"])
        else:
            self.model.Q.load_state_dict(checkpoint)

        self.model.Q.eval()
        self.model.Q.to(self.model.device)



    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        obs = np.array(observation, dtype=np.float32)

        with torch.no_grad():
            a1_idx = self.model.act(obs, eps=0.0, reset_noise=False)
        a1_idx = int(a1_idx.item() if hasattr(a1_idx, "item") else a1_idx)
        a1 = self.mapper.map(a1_idx)
 
        #continuous_action = h_env.HockeyEnv.discrete_to_continous_action(a1)
        return a1.tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    model_path = '/home/stud396/laser-hockey/checkpoints/20260224_011221_DQN/final_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return HockeyAgent(model_path=model_path, device=device)


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()  