import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import DQNAgent
from actions.action_mapping import DiscreteActionWrapper


def watch_agent(
        env_name="Pendulum-v1",
        model_path="../checkpoints/final_model.pth",
        video_folder="videos",
        bins=5,
):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda x: True,
        name_prefix="dqn_eval",
    )

    env = DiscreteActionWrapper(env, bins=bins)

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
    )

    agent.load(model_path)
    agent.Q.eval()

    obs, _ = env.reset()
    done = False

    print(f"Recording episode using model: {model_path}")

    while not done:
        # Greedy action (no exploration)
        action = agent.act(obs, eps=0.0)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    print(f"Video saved to: {video_folder}")


if __name__ == "__main__":
    watch_agent()
