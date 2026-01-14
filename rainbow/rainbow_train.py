import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import json
import hockey.hockey_env as h_env
from agents.agent import DQNAgent
from actions.action_mapping import DiscreteActionWrapper, ActionMapper, HockeyDQNWrapper, GenericDiscreteWrapper
from actions.action_sets import baseline_action_set
import os
import torch
from util.rewards import HockeyRewardManager


def run_training(
        env,
        max_episodes=1500,
        max_steps=500,
        action_set=None,
        wrapper_fn=None,
        keep_mode=True,
        save_dir="checkpoints",
        save_every=1000,
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

    opponent = h_env.BasicOpponent(weak=True)

    # if wrapper_fn is not None and isinstance(env.action_space, gym.spaces.Box):
    #    env = wrapper_fn(env)

    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=gym.spaces.Discrete(8),
        eps=0.2,
        discount=0.95,
        buffer_size=int(5e5),
        batch_size=256,
        learning_rate=5e-5,
        update_target_every=20,
        use_double_dqn=True
    )

    # Observation scaling constants from the notebook
    scaling = np.array([1.0, 1.0, 0.5, 4.0, 4.0, 4.0,
                        1.0, 1.0, 0.5, 4.0, 4.0, 4.0,
                        2.0, 2.0, 10.0, 10.0, 4.0, 4.0])

    stats = []
    all_losses = []
    total_steps = 0
    total_puck_touches = 0
    reward_manager = HockeyRewardManager(
        puck_touch_bonus=10.0,
        proximity_multiplier=0.05,
        direction_multiplier=5.0
    )
    best_reward = -float("inf")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting training...")

    for episode in range(max_episodes):
        obs, info = env.reset()
        # obs = np.clip(obs, -10.0, 10.0)  # optional normalization
        obs = obs * scaling
        total_reward = 0.0

        should_render = (episode % 20 == 0)

        for t in range(max_steps):
            obs_agent2 = env.obs_agent_two()
            a2 = opponent.act(obs_agent2)
            if should_render:
                env.render()  # This opens the window

            eps = max(0.1, 1.0 * (0.9954 ** episode))

            a1_idx = agent.act(obs, eps=eps)

            scalar_idx = int(a1_idx.item() if hasattr(a1_idx, 'item') else a1_idx)

            a1 = np.array(env.discrete_to_continous_action(scalar_idx)).flatten()
            a2 = np.array(a2).flatten()
            combined_action = np.hstack([a1, a2])
            obs_new, env_reward, terminated, truncated, info = env.step(combined_action)
            # new_reward = custom_reward(reward, info)
            done = terminated or truncated
            obs_new = obs_new * scaling
            shaped_reward = reward_manager.get_shaped_reward(env_reward, info, obs_new)

            agent.buffer.add_transition((obs, a1_idx, shaped_reward, obs_new, done))

            obs = obs_new
            total_reward += shaped_reward
            total_steps += 1
            if info.get("reward_touch_puck", 0) > 0:
                total_puck_touches += 1
                print(f"PUCK TOUCHED! (Episode: {episode} | step: {t} | Total this session: {total_puck_touches})")

            if done: break

        # Train if buffer is large enough
        if agent.buffer.size >= agent._config['batch_size']:
            losses = agent.train(iter_fit=32)
            all_losses.extend(losses)

        # Logging
        stats.append({
            "episode": episode,
            "reward": float(total_reward),
            "eps": float(eps)
        })

        # Save best model
        if save_best and total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(save_dir, "best_model.pth"))

        # Periodic checkpoint
        if episode % save_every == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_ep{episode}.pth"))

        if episode % 20 == 0:
            print(f"Episode {episode:4d} | Reward: {total_reward:8.2f} | eps: {eps:8.2f}")

        """        # .unsqueeze(0) adds the 'batch' dimension at index 0
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).cpu()

        with torch.no_grad():
            q_values = agent.Q(obs_tensor)

        # Use .squeeze() to remove the batch dimension for printing
        print(f"Action Q-Values: {q_values.squeeze().numpy()}")"""

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
    # env = gym.make('Pendulum-v1')
    # env = DiscreteActionWrapper(env, bins=5)

    env = gym.make('Hockey-v0')
    # env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
    # mapper = ActionMapper(baseline_action_set())
    # env = GenericDiscreteWrapper(env, mapper)

    run_training(
        env=env,
        max_episodes=4000,
        max_steps=500,
        wrapper_fn=None
    )
