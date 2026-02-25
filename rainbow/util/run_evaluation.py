import torch
import hockey.hockey_env as h_env
import gymnasium as gym
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import DQNAgent
from actions.action_mapping import ActionMapper
from actions.action_sets import baseline_action_set, compound_action_set
from util.hockey_evaluation import HockeyEvaluator


def run_agent_evaluation(model_path, env, agent, mapper, num_games=50, weak_opponent=False):
    """
    Loads a saved DQN and runs a full evaluation suite.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "q_network" in checkpoint:
        agent.Q.load_state_dict(checkpoint["q_network"])
    else:
        agent.Q.load_state_dict(checkpoint)

    agent.Q.to(device)

    opponents = [h_env.BasicOpponent(weak=weak_opponent)]

    evaluator = HockeyEvaluator(env, agent, mapper)
    win_rate, results, stats = evaluator.evaluate(num_games=num_games, opponents=opponents)

    print(f"\n--- Evaluation Finished ---")
    print(f"Path: {model_path}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Results: {stats}")

    return win_rate, stats


if __name__ == "__main__":
    env = gym.make('Hockey-v0')
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    # env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
    mapper = ActionMapper(compound_action_set(), keep_mode=True)
    shared_config = {
        "use_per": True,
        "per_alpha": 0.6,
        "discount": 0.98,
        "buffer_size": int(2e5),
        "batch_size": 32,
        "learning_rate": 3e-4,
        "lr_scheduler_steps": [350_000, 700_000],
        "lr_scheduler_factor": 0.5,
        "train_every": 4,
        "use_double_dqn": True,
        "hidden_layers": [128, 256, 256],
        "use_noisy_linear": True,
        "tau": 0.001,
        "use_soft_update": True,
        "use_recurrent": False,
        "sequence_length": 25,
        "n_step": 3,
        "use_n_step": True,
        "use_quantile_regression": True,
        "num_quantiles": 32,
        "use_quantile_regression": True,
        "num_quantiles": 32,
        "use_symmetry_regularization": True,
        "symmetry_lambda": 5e-05,
    }

    agent = DQNAgent(
        observation_space=env.observation_space,
        action_space=gym.spaces.Discrete(len(mapper)),
        mapper=mapper,
        **shared_config
    )

    win_rate, stats = run_agent_evaluation(
        model_path="/home/stud396/laser-hockey/checkpoints/20260218_215228_DQN/final_model.pth",
        env=env,
        agent=agent,
        mapper=mapper,
        num_games=1_000
    )
