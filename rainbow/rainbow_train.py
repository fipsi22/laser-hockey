import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import hockey.hockey_env as h_env
from datetime import datetime
from agents.agent import DQNAgent
from collections import deque
from actions.action_mapping import ActionMapper
from actions.action_sets import baseline_action_set, compound_action_set
import os
import torch
from util.rewards import HockeyRewardManager
from util.hockey_evaluation import HockeyEvaluator
from util.logging import TrainingTracker, save_run_config


def run_training(
        env,
        max_episodes=1500,
        max_steps=250,
        action_set=None,
        action_mapper=None,
        keep_mode=True,
        save_dir="checkpoints",
        save_every=5000,
        save_best=True,
        curriculum_threshold=5000
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

    # if wrapper_fn is not None and isinstance(env.action_space, gym.spaces.Box):
    #    env = wrapper_fn(env)

    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=gym.spaces.Discrete(len(mapper)),
        use_per=True,
        eps=0.05,
        discount=0.99,
        buffer_size=int(1e6),
        batch_size=32,
        learning_rate=3e-4,
        use_double_dqn=True,
        hidden_layers=[512, 512],
        use_noisy_linear=False
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{agent.config.get('run_name', 'DQN')}"
    save_dir = os.path.join("checkpoints", run_name)
    os.makedirs(save_dir, exist_ok=True)

    save_run_config(agent.config, save_dir)
    tracker = TrainingTracker(save_dir)

    stats = []
    all_losses = []
    losses = []
    total_steps = 0
    total_puck_touches = 0
    reward_manager = HockeyRewardManager(
        puck_touch_bonus=2.0,
        proximity_multiplier=1.0,
        direction_multiplier=0.1
    )
    best_reward = -float("inf")
    gamma = agent._config.get('discount', 0.95)
    train_every = agent._config.get('train_every', 4)

    weak_opp = h_env.BasicOpponent(weak=True)
    hard_opp = h_env.BasicOpponent(weak=False)
    opponent_pool = [weak_opp, hard_opp]

    evaluator = HockeyEvaluator(env, agent, mapper)
    evaluation_history = []
    eval_opponent = h_env.BasicOpponent(weak=False)

    total_max_steps = 12_000_000
    beta_start = 0.4
    beta_end = 1.0

    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting training...")

    for episode in range(max_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        should_render = (episode % 20 == 0)

        if episode < curriculum_threshold:
            current_opponent = hard_opp
        else:
            current_opponent = random.choice(opponent_pool)

        for t in range(max_steps):

            progress = min(1.0, total_steps / total_max_steps)
            current_beta = beta_start + progress * (beta_end - beta_start)

            obs_agent2 = env.obs_agent_two()
            a2 = current_opponent.act(obs_agent2)
            if should_render:
                env.render()  # This opens the window

            if agent.use_noisy:
                eps = 0
            else:
                eps = max(0.1, 1.0 - (episode / 4_000))

            a1_idx = agent.act(obs, eps=eps)
            a1_idx = int(a1_idx.item() if hasattr(a1_idx, "item") else a1_idx)

            a1_cont = mapper.map(a1_idx)

            a2 = np.array(current_opponent.act(obs_agent2)).flatten()

            combined_action = np.hstack([a1_cont, a2])

            obs_new, env_reward, terminated, truncated, info = env.step(combined_action)
            done = terminated or truncated
            shaped_reward = reward_manager.get_shaped_reward(env_reward, info, obs_new, episode)

            agent.buffer.add(obs, obs_new, a1_idx, shaped_reward, done)

            obs = obs_new
            total_reward += shaped_reward
            total_steps += 1
            if info.get("reward_touch_puck", 0) > 0:
                total_puck_touches += 1
                # print(f"PUCK TOUCHED! (Episode: {episode} | step: {t} | Total this session: {total_puck_touches})")

            # Train if buffer is large enough
            current_buffer_size = agent.buffer.buffer_size if agent.buffer.full else agent.buffer.pos
            if current_buffer_size >= 80_000 and total_steps % 4 == 0:
                losses = agent.train(iter_fit=2, beta=current_beta)
                all_losses.extend(losses)

            if done:
                break
        agent.lr_scheduler.step()

        # Logging
        current_lr = agent.optimizer.param_groups[0]['lr']
        stats.append({
            "episode": episode,
            "reward": float(total_reward),
            "eps": float(eps)
        })
        episode_stats = {
            "episode": episode,
            "lr": float(current_lr),
            "reward": total_reward,
            "loss": np.mean(losses) if losses else 0,
            "eps": eps,
            "steps": t,
            "touch_count": total_puck_touches
        }

        # Save best model
        if save_best and total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(save_dir, "best_model.pth"))

        # Periodic checkpoint
        if episode % save_every == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_ep{episode}.pth"))

        if episode % 20 == 0:
            print(f"Episode {episode:4d} | Reward: {total_reward:8.2f} | eps: {eps:8.2f} | lr: {current_lr}")

        if episode % 500 == 0 and episode > 0:
            print(f"\n--- Running Evaluation at Episode {episode} ---")
            win_rate, detailed_results = evaluator.evaluate(num_games=500, opponents=[eval_opponent])

            #agent.lr_scheduler.step(win_rate)

            evaluation_history.append({"episode": episode, "win_rate": win_rate})
            episode_stats["win_rate"] = win_rate
            episode_stats["eval_wins"] = detailed_results['wins']
            episode_stats["eval_losses"] = detailed_results['losses']
            episode_stats["eval_ties"] = detailed_results['ties']
            print(f"Win Rate: {win_rate:.2%} | Wins: {detailed_results['wins']} | Losses: {detailed_results['losses']}")
        tracker.log_episode(episode_stats)

        # .unsqueeze(0) adds the 'batch' dimension at index 0
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).cpu()

        #with torch.no_grad():
            #q_values = agent.Q(obs_tensor)

        # Use .squeeze() to remove the batch dimension for printing
        # print(f"Action Q-Values: {q_values.squeeze().numpy()}, reward: {float(total_reward)}")
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
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    # env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
    mapper = ActionMapper(compound_action_set())
    #env = GenericDiscreteWrapper(env, mapper)

    run_training(
        env=env,
        max_episodes=20_000,
        max_steps=250,
        action_mapper=mapper
    )
