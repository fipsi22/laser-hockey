import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import copy
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
from util.logging_utils import TrainingTracker, save_run_config


def set_seeds():
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def add_episode_to_buffer(agent, episode_cache, mapper, n_step, gamma):
    states = [t['state'] for t in episode_cache]
    actions = [t['action'] for t in episode_cache]
    rewards = [t['reward'] for t in episode_cache]
    dones = [t['done'] for t in episode_cache]

    for i in range(len(states)):
        n_step_reward = 0
        is_done = False
        for n in range(n_step):
            if i + n < len(rewards):
                n_step_reward += (gamma ** n) * rewards[i + n]
                if dones[i + n]:
                    is_done = True
                    break
            else:
                is_done = True
                break

        next_s = states[i + 1] if i + 1 < len(states) else states[i]
        agent.buffer.add(states[i], next_s, actions[i], n_step_reward, is_done)



def run_training(
        env,
        max_episodes=1500,
        max_steps=250,
        action_set=None,
        action_mapper=None,
        keep_mode=True,
        save_dir="checkpoints",
        save_every=5_000,
        save_best=True,
        curriculum_threshold=2_000,
        n_step=5,
        use_n_step=True,
        iter_fit=8
):

    shared_config = {
        "use_per": True,
        "per_alpha": 0.6,
        "discount": 0.98,
        "buffer_size": int(2e5),
        "batch_size": 32,
        "learning_rate": 2e-4,
        "lr_scheduler_steps": "[200000, 350000, 700000]",
        "lr_scheduler_factor": 0.5,
        "train_every": 4,
        "use_double_dqn": True,
        "hidden_layers": [128, 256, 256],
        "use_noisy_linear": True,
        "tau": 0.001,
        "use_soft_update": True,
        "n_step": 5,
        "use_n_step": True,
        "use_quantile_regression": True,
        "num_quantiles": 32,
        "use_symmetry_augmentation": True,
        "use_symmetry_regularization":True,
        "symmetry_lambda": 0.0001,
    }

    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=gym.spaces.Discrete(len(action_mapper)),
        mapper=action_mapper,
        **shared_config
    )

    opponent_agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=gym.spaces.Discrete(len(action_mapper)),
        mapper=action_mapper,
        **shared_config
    )
    opponent_agent.Q.eval()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{agent.config.get('run_name', 'DQN')}"
    print(run_name, flush=True)
    save_dir = os.path.join("checkpoints", run_name)
    os.makedirs(save_dir, exist_ok=True)
    tracker = TrainingTracker(save_dir)

    stats = []
    all_losses = []
    all_gaps = []
    total_steps = 0
    total_puck_touches = 0
    reward_manager = HockeyRewardManager()
    best_reward = -float("inf")
    
    train_every = agent._config.get('train_every', 4)

    use_quantile_regression = agent._config.get('use_quantile_regression', True)
    num_quantiles = agent._config.get('num_quantiles', 32)
    use_recurrent = agent._config.get('use_recurrent', True)

    weak_opp = h_env.BasicOpponent(weak=True)
    hard_opp = h_env.BasicOpponent(weak=False)

    evaluator = HockeyEvaluator(env, agent, action_mapper, use_quantile=use_quantile_regression)
    evaluation_history = []
    eval_opponent = h_env.BasicOpponent(weak=False)

    use_self_play = True
    self_play_start_episode = 15_000
    shared_config["self_play_start_episode"] = self_play_start_episode
    self_play_save_every = 1_000
    shared_config["self_play_save_every"] = self_play_save_every
    opponent_probs = [0.2, 0.3, 0.5]
    shared_config["opponent_probs"] = opponent_probs
    self_play_pool = []

    beta_total_max_steps = 1_500_000  
    shared_config["beta_total_max_steps"] = beta_total_max_steps
    beta_start = 0.4
    shared_config["beta_start"] = beta_start
    beta_end = 1.0
    shared_config["beta_end"] = beta_end

    eps_start = 0.2
    shared_config["eps_start"] = eps_start
    eps_end = 0.05
    shared_config["eps_end"] = eps_end
    eps_decay_steps = 1_500_000  
    shared_config["eps_decay_steps"] = eps_decay_steps

    gamma_start = 0.98
    shared_config["gamma_start"] = gamma_start
    gamma_end = 0.99
    shared_config["gamma_end"] = gamma_end
    gamma_growth_steps = 1_500_000    
    shared_config["gamma_growth_steps"] = gamma_growth_steps
    save_run_config(shared_config, save_dir)

    for episode in range(max_episodes + 1):
        reward_manager.reset()
        obs, info = env.reset()
        total_reward = 0.0
        loss = 0.0
        episode_losses = []
        episode_gaps =  []
        episode_cache = []

        if episode < curriculum_threshold:
            current_opponent = weak_opp
            op_label = "WeakBot"
        elif not use_self_play or episode < self_play_start_episode or len(self_play_pool) == 0:
            op_type = random.choices(['weak', 'hard'], weights=[0.3, 0.7], k=1)[0]
            if op_type == 'weak':
                current_opponent = weak_opp
                op_label = "WeakBot"
            elif op_type == 'hard':
                current_opponent = hard_opp
                op_label = "HardBot"
        else:
            op_type = random.choices(['weak', 'hard', 'self'], weights=opponent_probs, k=1)[0]
            if op_type == 'weak':
                current_opponent = weak_opp
                op_label = "WeakBot"
            elif op_type == 'hard':
                current_opponent = hard_opp
                op_label = "HardBot"
            else:
                selected_weights = random.choice(self_play_pool)
                opponent_agent.Q.load_state_dict(selected_weights)
                current_opponent = opponent_agent
                op_label = "Self"

        for t in range(max_steps):

            eps_progress = min(1.0, total_steps / eps_decay_steps)
            eps_threshold = max(0.05, eps_start - (total_steps / eps_decay_steps))
            #current_eps = 0

            gamma_progress = min(1.0, total_steps / gamma_growth_steps)
            current_gamma = gamma_start + gamma_progress * (gamma_end - gamma_start)

            if use_n_step:
                agent.effective_discount = current_gamma ** n_step
            else:
                agent.effective_discount = current_gamma

            progress = min(1.0, total_steps / beta_total_max_steps)
            current_beta = beta_start + progress * (beta_end - beta_start)

            a1_idx = agent.act(obs, eps=eps_threshold)
            a1_idx = int(a1_idx.item() if hasattr(a1_idx, "item") else a1_idx)
            a1_cont = action_mapper.map(a1_idx)

            obs_agent2 = env.obs_agent_two()
            if op_label == "Self":
                a2_idx = current_opponent.act(obs_agent2, eps=0.0)
                a2_cont = action_mapper.map(a2_idx)
            else:
                a2_cont = current_opponent.act(obs_agent2)

            combined_action = np.hstack([a1_cont, a2_cont])
            obs_new, env_reward, terminated, truncated, info = env.step(combined_action)
            done = terminated or truncated

            shaped_reward = reward_manager.get_basic_reward_shaped(env, obs_new, info, env_reward)

            episode_cache.append({
                'state': obs,
                'action': a1_idx,
                'reward': shaped_reward,
                'done': done
            })

            total_reward += shaped_reward
            total_steps += 1

            if info.get("reward_touch_puck", 0) > 0:
                total_puck_touches += 1

            current_buffer_size = agent.buffer.buffer_size if agent.buffer.full else agent.buffer.pos

            if current_buffer_size >= 20_000:
                if total_steps % train_every == 0:
                    loss, action_gap = agent.train(iter_fit=iter_fit, beta=current_beta)
                    all_losses.append(loss)
                    episode_losses.append(loss)
                    episode_gaps.append(action_gap)
            
            obs = obs_new
            if done:
                break
        add_episode_to_buffer(agent, episode_cache, action_mapper, n_step, current_gamma)    

        if use_self_play and episode >= self_play_start_episode:
            if episode % self_play_save_every == 0:
                self_play_pool.append(copy.deepcopy(agent.Q.state_dict()))
                if len(self_play_pool) > 20:
                    self_play_pool.pop(0)

        current_lr = agent.optimizer.param_groups[0]['lr']
        lr_steps = agent.lr_scheduler.last_epoch

        avg_episode_loss = np.mean(episode_losses) if episode_losses else 0.0
        avg_episode_gap = np.mean(episode_gaps) if episode_gaps else 0.0

        episode_stats = {
            "episode": episode,
            "reward": float(total_reward),
            "loss": avg_episode_loss,
            "action_gap": avg_episode_gap,
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
            print(f"Episode {episode:4d} | Reward: {total_reward:8.2f} | Gap: {avg_episode_gap:.4f} | loss: {avg_episode_loss:.4f} | | LR: {current_lr} | steps: {total_steps} | gradient steps: {lr_steps}", flush=True)

        if episode % 1_000 == 0 and episode > 0:
            print(f"\n--- Running Evaluation at Episode {episode} ---", flush=True)

            old_use_noisy = agent.use_noisy
            agent.use_noisy = False
            was_training = agent.Q.training
            agent.Q.eval()

            win_rate, detailed_results, eval_stats = evaluator.evaluate(
                num_games=1_000,
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
                "eval_ties": detailed_results['ties'],
                "eval_mean_q": eval_stats['mean_q'],
                "eval_mean_gap": eval_stats['mean_gap']
            })

            print(f"Win Rate: {win_rate:.2%} | wins: {detailed_results['wins']} | losses: {detailed_results['losses']} | ties: {detailed_results['ties']}", flush=True)
            print(f"Global Eval Q Mean/Max: {eval_stats['mean_q']:.3f} / {eval_stats['max_q']:.3f} | Global Action Gap (Certainty): {eval_stats['mean_gap']:.4f}", flush=True)
        
        tracker.log_episode(episode_stats)

    env.close()
    agent.save(os.path.join(save_dir, "final_model.pth"))


if __name__ == "__main__":
    set_seeds()
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    mapper = ActionMapper(compound_action_set(), keep_mode=True)
    run_training(
        env=env,
        max_episodes=30_000,
        action_mapper=mapper
    )