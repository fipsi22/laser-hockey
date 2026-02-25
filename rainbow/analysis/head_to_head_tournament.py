import os
import sys
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import gymnasium as gym
import hockey.hockey_env as h_env
from tueplots import bundles
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.agent import DQNAgent
from actions.action_mapping import ActionMapper
from actions.action_sets import baseline_action_set, compound_action_set



def run_h2h_tournament(agent_configs, num_games=50):
    """
    agent_configs: List of dicts with {'path': 'path/to/model.pth', 'label': 'Name', 'obj': AgentClass}
    """

    weak_opp = h_env.BasicOpponent(weak=True)
    hard_opp = h_env.BasicOpponent(weak=False)

    competitor_labels = [c['label'] for c in agent_configs] + ["Basic (Weak)", "Basic (Hard)"]
    n = len(competitor_labels)
    win_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                win_matrix[i, j] = 0.5
                continue
            
            print(f"Match: {competitor_labels[i]} vs {competitor_labels[j]}")
        
            agent1 = get_competitor_obj(i, agent_configs, weak_opp, hard_opp)
            agent2 = get_competitor_obj(j, agent_configs, weak_opp, hard_opp)
            
            wr, _, _ = evaluate_match(agent1, agent2, num_games)
            win_matrix[i, j] = wr

    return win_matrix, competitor_labels

def evaluate_match(agent1, agent2, num_games):
    """Modified version of your evaluate function to support two arbitrary agents"""
    env = h_env.HockeyEnv()
    results = {"wins": 0, "losses": 0, "ties": 0}
    
    for _ in range(num_games):
        obs, _ = env.reset()
        done = False
        while not done:
            a1 = get_action(agent1, obs) 
            obs_agent2 = env.obs_agent_two()
            a2 = get_action(agent2, obs_agent2)
            combined_action = np.hstack([a1, a2])
            obs, _, terminated, truncated, info = env.step(combined_action)
            done = terminated or truncated

        w = info.get("winner", 0)
        if w == 1: results["wins"] += 1
        elif w == -1: results["losses"] += 1
        else: results["ties"] += 1
            
    return results["wins"] / num_games, results, {}

def get_action(agent, obs):
    if hasattr(agent, 'mapper'):
        with torch.no_grad():
            a1_idx = agent.act(obs, eps=0.0, reset_noise=False)
            a1_idx = int(a1_idx.item() if hasattr(a1_idx, "item") else a1_idx)
            a1 = mapper.map(a1_idx)
        return a1
    else:
        return np.array(agent.act(obs)).flatten()

def get_competitor_obj(idx, configs, weak, hard):
    if idx < len(configs):
        return configs[idx]['obj']
    elif idx == len(configs):
        return weak
    else:
        return hard

def plot_tournament_heatmap(matrix, labels):
    new_labels = [
        'Rainbow-DQN',
        'Rainbow-DQN\n+ Quantile',
        'Rainbow-DQN\n+ Aug',
        'Rainbow-DQN\n + Quantile\n+ Aug',
        'Rainbow-DQN\n + Quantile\n+ Aug + Reg',
        'Basic\n(Weak)',
        'Basic\n(Hard)'
    ]

    exclude_list = ["Basic (Weak)", "Basic (Hard)"]
    indices_to_keep = [i for i, l in enumerate(labels) if l not in exclude_list]

    filtered_matrix = matrix[indices_to_keep, :]
    row_labels = [new_labels[i] for i in indices_to_keep]
    col_labels = new_labels

    plt.rcParams.update(bundles.icml2024(column='full', usetex=False))
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    fig, ax = plt.subplots(figsize=(plt.rcParams["figure.figsize"][0], 2.4), layout="constrained")
    fig.subplots_adjust(right=5)
    cmap = sns.diverging_palette(20, 220, s=40, l=50, as_cmap=True)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)

    heatmap = sns.heatmap(
        filtered_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        xticklabels=col_labels,
        yticklabels=row_labels,
        square=False,
        linewidths=.8,
        linecolor='white',
        cbar=True,
        cbar_ax=cax,
        vmin=0.0,
        vmax=1.0,
        annot_kws={"size": 7},
        ax=ax
    )
    cbar = heatmap.collections[0].colorbar
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
    cbar.ax.tick_params(labelsize=8)

    cbar.set_label("Win Rate", rotation=270, labelpad=10, fontsize=10)
    ax.set_xticklabels(new_labels, ha='center', va='top', rotation=0)
    ax.set_yticklabels(row_labels, va='center', ha='center', rotation=0)
    ax.tick_params(axis='y', pad=30)
    ax.set_ylabel("Evaluating Agent", fontsize=10, labelpad=10)
    ax.set_xlabel("Opponent Agent", fontsize=10)

    plt.show()
    fig.savefig("h2h_tournament_matrix.pdf", bbox_inches='tight')

def load_agent_model(agent, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "q_network" in checkpoint:
        state_dict = checkpoint["q_network"]
    else:
        state_dict = checkpoint
    
    agent.Q.load_state_dict(state_dict)
    agent.Q.to(device)
    agent.Q.eval()
    return agent


if __name__ == "__main__":
    checkpoint_path = '/home/stud396/laser-hockey/checkpoints'

    mapper = ActionMapper(compound_action_set(), keep_mode=True)
    env = h_env.HockeyEnv()
    agent_dqn = DQNAgent(
        observation_space=env.observation_space,
        action_space=gym.spaces.Discrete(len(mapper)),
        mapper=mapper,
        hidden_layers=[128, 256, 256], use_noisy_linear=True, 
        use_quantile_regression=False)
    agent_dqn_quantile = DQNAgent(
        observation_space=env.observation_space,
        action_space=gym.spaces.Discrete(len(mapper)),
        mapper=mapper,
        hidden_layers=[128, 256, 256], use_noisy_linear=True, 
        use_quantile_regression=True, num_quantiles=32)
    '''
    configs = [
        {'label': 'DQN (Rainbow)', 'obj': load_agent_model(agent_dqn, checkpoint_path + '/20260218_221231_DQN/final_model.pth')},
        {'label': 'DQN + Quantile', 'obj': load_agent_model(agent_dqn_quantile, checkpoint_path + '/20260218_215301_DQN/final_model.pth')},
        {'label': 'DQN + Aug', 'obj': load_agent_model(agent_dqn, checkpoint_path + '/20260218_215058_DQN/final_model.pth')},
        {'label': 'DQN + Quantile + Aug', 'obj': load_agent_model(agent_dqn_quantile, checkpoint_path + '/20260218_220628_DQN/final_model.pth')},
        {'label': 'DQN + Quantile + Aug + Reg', 'obj': load_agent_model(agent_dqn_quantile, checkpoint_path + '/20260218_215228_DQN/final_model.pth')},
    ]
    '''
    #matrix, labels = run_h2h_tournament(configs, num_games=100)
    #np.save("h2h_matrix.npy", matrix)
    #np.save("h2h_labels.npy", labels)
    labels = np.load(r"C:\GitHub\laser-hockey\rainbow\util\h2h_labels.npy")
    matrix = np.load(r"C:\GitHub\laser-hockey\rainbow\util\h2h_matrix.npy")


    print(f'matrix.shape: {matrix.shape}, labels.shape: {labels.shape}')
    plot_tournament_heatmap(matrix, labels)

