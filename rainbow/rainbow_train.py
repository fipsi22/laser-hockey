import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import json
import hockey.hockey_env as h_env
from agents.agent import DQNAgent
from actions.action_mapping import DiscreteActionWrapper, ActionMapper, HockeyDQNWrapper, GenericDiscreteWrapper
from actions.action_sets import baseline_action_set
import os


def run_training(
        env,
        max_episodes=600,
        max_steps=500,
        action_set=None,
        wrapper_fn=None,
        keep_mode=True,
        save_dir="checkpoints",
        save_every=50,
        save_best=True,
):
    """
    Trains a DQN agent on any Gymnasium environment.

    Args:
        env: Gym environment
        max_episodes (int): number of episodes
        max_steps (int): max steps per episode
        action_set (list): optional list of DiscreteAction primitives (for discrete wrapper)
        keep_mode (bool): whether shooting/extra actions are used in discrete action mapper
    """

    if wrapper_fn is not None and isinstance(env.action_space, gym.spaces.Box):
        env = wrapper_fn(env)

    agent = DQNAgent(
        env.observation_space,
        env.action_space,
        eps=0.2,
        discount=0.95,
        buffer_size=int(5e5),
        batch_size=128,
        learning_rate=0.0002,
        update_target_every=20,
        use_double_dqn=True
    )

    stats = []
    all_losses = []
    total_steps = 0
    best_reward = -float("inf")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting training...")

    for episode in range(max_episodes):
        obs, info = env.reset()
        obs = np.clip(obs, -10.0, 10.0)  # optional normalization
        total_reward = 0.0
        steps = 0

        for t in range(max_steps):
            eps = max(agent._config['eps'] * (1 - total_steps / 2e5), 0.05)
            action = agent.act(obs, eps=eps)

            obs_new, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs_new = np.clip(obs_new, -10.0, 10.0)  # optional normalization
            agent.buffer.add_transition((obs, action, reward, obs_new, done))

            obs = obs_new
            total_reward += reward
            steps += 1
            total_steps += 1

            if done:
                break

        # Train if buffer is large enough
        if agent.buffer.size >= agent._config['batch_size']:
            losses = agent.train(iter_fit=32)
            all_losses.extend(losses)

        # Logging
        stats.append({
            "episode": episode,
            "reward": float(total_reward),
            "steps": steps
        })

        # Save best model
        if save_best and total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(save_dir, "best_model.pth"))

        # Periodic checkpoint
        if episode % save_every == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_ep{episode}.pth"))

        if episode % 20 == 0:
            print(f"Episode {episode:4d} | Reward: {total_reward:8.2f} | Steps: {steps}")

    env.close()

    agent.save(os.path.join(save_dir, "final_model.pth"))

    # Save results
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
    env = gym.make('Pendulum-v1')
    env = DiscreteActionWrapper(env, bins=5)

    # env = gym.make('Hockey-v0')
    # env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    #mapper = ActionMapper(baseline_action_set())
    #env = GenericDiscreteWrapper(env, mapper)

    run_training(
        env=env,
        max_episodes=600,
        max_steps=200,
        wrapper_fn=lambda e: DiscreteActionWrapper(e)
    )
