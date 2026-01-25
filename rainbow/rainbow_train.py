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


def set_seeds():
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


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
        curriculum_threshold=10_000,
        n_step=3,
        use_n_step=True,
):
    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=gym.spaces.Discrete(len(mapper)),
        use_per=True,
        per_alpha=0.4,
        discount=0.99,
        buffer_size=int(1e5),
        batch_size=128,
        learning_rate=5e-4,
        use_double_dqn=True,
        hidden_layers=[512],
        use_noisy_linear=True,
        n_step=5,
        use_n_step=True,
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
    reward_manager = HockeyRewardManager()
    best_reward = -float("inf")
    n_step_buffer = deque(maxlen=n_step)
    gamma = agent._config.get('discount', 0.99)
    if use_n_step:
        agent.discount = gamma ** n_step
    else:
        agent.discount = gamma
    train_every = agent._config.get('train_every', 4)

    weak_opp = h_env.BasicOpponent(weak=True)
    hard_opp = h_env.BasicOpponent(weak=False)
    opponent_pool = [weak_opp, hard_opp]

    evaluator = HockeyEvaluator(env, agent, mapper)
    evaluation_history = []
    eval_opponent = h_env.BasicOpponent(weak=False)

    total_steps_count = 0
    total_max_steps = 12_000_000
    beta_start = 0.4
    beta_end = 1.0

    eps_start = 0.1
    eps_end = 0.05
    eps_decay_steps = 1_000_000

    print("Starting training...")

    for episode in range(max_episodes):
        reward_manager.reset()
        obs, info = env.reset()
        n_step_buffer.clear()
        total_reward = 0.0
        loss = None
        episode_losses = []
        should_render = (episode % 20 == 0)

        current_opponent = weak_opp if episode < curriculum_threshold else random.choice(opponent_pool)

        for t in range(max_steps):

            eps_progress = min(1.0, total_steps / eps_decay_steps)
            #current_eps = eps_end + (eps_start - eps_end) * (1.0 - eps_progress)
            current_eps=0

            progress = min(1.0, total_steps / 3_000_000)
            current_beta = beta_start + progress * (beta_end - beta_start)

            if should_render:
                env.render()

            a1_idx = agent.act(obs, eps=current_eps)  # eps always zero
            a1_idx = int(a1_idx.item() if hasattr(a1_idx, "item") else a1_idx)

            a1_cont = mapper.map(a1_idx)
            obs_agent2 = env.obs_agent_two()
            a2 = np.array(current_opponent.act(obs_agent2)).flatten()
            combined_action = np.hstack([a1_cont, a2])

            obs_new, env_reward, terminated, truncated, info = env.step(combined_action)
            done = terminated or truncated

            shaped_reward = reward_manager.get_basic_reward_shaped(env, obs_new, info, env_reward)

            if use_n_step:
                # Add to local temporary buffer
                n_step_buffer.append((obs, a1_idx, shaped_reward))

                # Only add to PER if we have enough steps OR we are at the end of the game
                if len(n_step_buffer) == n_step:
                    # Calculate discounted sum: R = r1 + γ*r2 + γ^2*r3
                    sum_reward = sum([n_step_buffer[i][2] * (gamma ** i) for i in range(n_step)])
                    state_n_ago, action_n_ago, _ = n_step_buffer[0]

                    agent.buffer.add(state_n_ago, obs_new, action_n_ago, sum_reward, done)
            else:
                agent.buffer.add(obs, obs_new, a1_idx, shaped_reward, done)


            total_reward += shaped_reward
            total_steps += 1

            if info.get("reward_touch_puck", 0) > 0:
                total_puck_touches += 1

            current_buffer_size = agent.buffer.buffer_size if agent.buffer.full else agent.buffer.pos
            if current_buffer_size >= 20_000 and total_steps % train_every == 0:
                loss = agent.train(iter_fit=1, beta=current_beta)
                all_losses.append(loss)
                episode_losses.append(loss)

            if done:
                if use_n_step and len(n_step_buffer) > 0:
                    while len(n_step_buffer) > 0:
                        sum_reward = sum([n_step_buffer[i][2] * (gamma ** i) for i in range(len(n_step_buffer))])
                        state_n_ago, action_n_ago, _ = n_step_buffer.popleft()
                        # For these final steps, next_state is obs_new (the terminal state)
                        agent.buffer.add(state_n_ago, obs_new, action_n_ago, sum_reward, True)
                break
            obs = obs_new

        current_lr = agent.optimizer.param_groups[0]['lr']

        episode_stats = {
            "episode": episode,
            "reward": float(total_reward),
            "loss": float(np.mean(episode_losses)) if episode_losses else 0.0,
            "lr": float(current_lr),
            "steps": t,
            "touch_count": total_puck_touches
        }

        stats.append(episode_stats)

        if save_best and total_reward > best_reward:
            best_reward = total_reward
            agent.save(os.path.join(save_dir, "best_model.pth"))

        if episode % save_every == 0:
            agent.save(os.path.join(save_dir, f"checkpoint_ep{episode}.pth"))

        if episode % 20 == 0:
            print(f"Episode {episode:4d} | Reward: {total_reward:8.2f} | LR: {current_lr} | steps: {total_steps} | loss: {loss}" )
            agent.buffer.log_per_stats()

        if episode % 500 == 0 and episode > 0:
            print(f"\n--- Running Evaluation at Episode {episode} ---")

            old_use_noisy = agent.use_noisy
            agent.use_noisy = False
            was_training = agent.Q.training
            agent.Q.eval()

            win_rate, detailed_results = evaluator.evaluate(
                num_games=500,
                opponents=[eval_opponent]
            )

            if was_training:
                agent.Q.train()
            agent.use_noisy = old_use_noisy

            evaluation_history.append({"episode": episode, "win_rate": win_rate})
            episode_stats.update({
                "win_rate": win_rate,
                "eval_wins": detailed_results['wins'],
                "eval_losses": detailed_results['losses'],
                "eval_ties": detailed_results['ties']
            })

            print(
                f"Win Rate: {win_rate:.2%} | wins: {detailed_results['wins']} | losses: {detailed_results['losses']} | ties: {detailed_results['ties']}")
            # Example inside your evaluator or test run
            with torch.no_grad():
                obs_tensor = torch.tensor(obs).unsqueeze(0).float().to(agent.device)
                q_vals = agent.Q(obs_tensor)
                print("Q min/max/mean:", q_vals.min().item(), q_vals.max().item(), q_vals.mean().item())

        tracker.log_episode(episode_stats)

    env.close()
    agent.save(os.path.join(save_dir, "final_model.pth"))

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
    set_seeds()
    # env = gym.make('Pendulum-v1')
    # env = DiscreteActionWrapper(env, bins=5)

    env = gym.make('Hockey-v0')
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    # env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
    mapper = ActionMapper(compound_action_set())
    # env = GenericDiscreteWrapper(env, mapper)

    run_training(
        env=env,
        max_episodes=20_000,
        max_steps=250,
        action_mapper=mapper
    )
