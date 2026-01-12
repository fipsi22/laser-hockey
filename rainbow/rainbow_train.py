import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import json
from agents.agent import DQNAgent
from actions.action_mapping import DiscreteActionWrapper


def run_training(env_name='Pendulum-v1', max_episodes=600):
    env = gym.make(env_name)
    if isinstance(env.action_space, gym.spaces.Box):
        env = DiscreteActionWrapper(env, bins=5)

    agent = DQNAgent(env.observation_space, env.action_space, eps=0.2)

    stats = []
    all_losses = []

    print(f"Starting training on {env_name}...")
    for i in range(max_episodes):
        total_reward = 0
        ob, _ = env.reset()
        steps = 0

        for t in range(500):
            action = agent.act(ob)
            ob_new, reward, done, trunc, _ = env.step(action)
            agent.buffer.add_transition((ob, action, reward, ob_new, done))
            total_reward += reward
            ob = ob_new
            steps = t + 1
            if done or trunc:
                break

        if agent.buffer.size > agent._config['batch_size']:
            episode_losses = agent.train(32)
            all_losses.extend(episode_losses)

        stats.append({"episode": i, "reward": float(total_reward), "steps": steps})

        if i % 20 == 0:
            print(f"Episode {i}: Reward: {total_reward:.2f}, Steps: {steps}")

    # Save results for analysis
    with open("results.json", "w") as f:
        json.dump(stats, f)

    plot_results(stats, all_losses)


def plot_results(stats, losses):
    rewards = [s['reward'] for s in stats]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title("Reward per Episode")
    plt.xlabel("Episode")

    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Loss Training")
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("training_performance.png")
    print("Results saved to results.json and training_performance.png")


if __name__ == "__main__":
    run_training()