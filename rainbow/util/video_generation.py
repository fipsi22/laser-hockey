import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import DQNAgent, DiscreteActionWrapper
import torch


def watch_agent(env_name='Pendulum-v1', video_folder='videos'):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda x: True)
    env = DiscreteActionWrapper(env, bins=5)

    agent = DQNAgent(env.observation_space, env.action_space)
    # agent.Q.load_state_dict(torch.load("dqn_model.pth"))

    ob, _ = env.reset()
    done = False

    print("Recording episode...")
    while not done:
        # low epsilon (0.01) so the agent acts on its knowledge
        action = agent.act(ob, eps=0.0)
        ob, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    print(f"Video saved in the folder: {video_folder}")


if __name__ == "__main__":
    watch_agent()
