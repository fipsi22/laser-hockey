import pandas as pd
import csv
import matplotlib.pyplot as plt
from tueplots import bundles

import pandas as pd
import matplotlib.pyplot as plt
import csv
import pickle
import numpy as np


def load_pkl_as_df(path, window=1000):
    """
    Loads pkl data and converts raw rewards into a
    win-rate DataFrame comparable to the CSV format.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    raw_rewards = np.array(data['rewards'])
    wins = (raw_rewards >= 3.0).astype(float)
    win_rate_series = pd.Series(wins).rolling(window=window, min_periods=1).mean()

    df = pd.DataFrame({
        'episode': np.arange(len(win_rate_series)),
        'win_rate': win_rate_series
    })
    return df.iloc[::1_000, :][1:]

def plot_multiple_win_rates(csv_configs, show_curriculum=False, save_path=''):
    """
    csv_configs: List of dicts, e.g.,
    [{'path': 'dqn.csv', 'label': 'DQN (Rainbow)', 'color': '#2c3e50'}, ...]
    """
    plt.rcParams.update(bundles.icml2024(column='full', usetex=False))
    plt.rcParams.update({
        "font.size": 10,  # default text
        "axes.labelsize": 11,  # x and y labels
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, ax = plt.subplots(figsize=(plt.rcParams["figure.figsize"][0], 3))
    global_max_ep = 0

    for config in csv_configs:
        path = config['path']

        if path.endswith('.pkl'):
            df = load_pkl_as_df(path)
        else:
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 11:
                        try:
                            data.append([float(row[0]), float(row[7])])
                        except (ValueError, IndexError):
                            continue
            if not data:
                continue
            df = pd.DataFrame(data, columns=['episode', 'win_rate'])

        global_max_ep = max(global_max_ep, df['episode'].max())

        ax.plot(
            df['episode'],
            df['win_rate'],
            marker='o',
            markersize=3,
            linewidth=1.5,
            color=config.get('color'),
            label=config['label']
        )


    if show_curriculum:
        ax.axvspan(0, 2000, color='grey', alpha=0.1)
        ax.text(1000, 1.02, r'$\mathbb{P} =$' + '\n' + '$\{O_{weak}\}$', ha='center', fontsize=8)

        ax.axvspan(2000, 15000, color='blue', alpha=0.03)
        ax.text(8500, 1.02, r'$\mathbb{P} = \{0.4 O_{weak}, 0.6 O_{hard}\}$', ha='center', fontsize=8)

        if global_max_ep > 15000:
            ax.axvspan(15000, max(32000, global_max_ep), color='green', alpha=0.03)
            ax.text(23500, 1.02, r'$\mathbb{P} = \{0.5 O_{self}, 0.3 O_{hard}, 0.2 O_{weak}\}$', ha='center', fontsize=8)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(0, max(32000, global_max_ep))
    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        frameon=True,
        facecolor='white',
        edgecolor='#cccccc',
        framealpha=0.9,
        fontsize=8,
        columnspacing=1.0
    )
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.show()
    fig.savefig(save_path, bbox_inches='tight')


if __name__ == "__main__":
    model_folder_path = r'/rainbow/models'

    configs = [
        {'path': model_folder_path + '/20260218_221231_DQN-rainbow/log.csv', 'label': 'DQN (Rainbow)', 'color': '#AA3377'},
        {'path': model_folder_path + '/20260219_130526_DQN-quantile/log.csv', 'label': 'DQN (Rainbow) + Quantile', 'color': '#CCBB44'},
        {'path': model_folder_path + '/20260218_215058_DQN-aug/log.csv', 'label': 'DQN (Rainbow) + Aug.', 'color': '#228833'},
        {'path': model_folder_path + '/20260218_220628_DQN-quantile-aug/log.csv', 'label': 'DQN (Rainbow) + Quantile + Aug.', 'color': '#66CCEE'},
        #{'path': model_folder_path + '/20260218_215301_DQN-quantile-reg/log.csv', 'label': 'DQN (Rainbow) + Quantile + Reg.', 'color': '#8d3da6'},
        {'path': model_folder_path + '/20260218_215228_DQN-quantile-aug-reg/log.csv', 'label': 'DQN (Rainbow) + Quantile + Aug + Reg.', 'color': '#4477AA'}
    ]

    plot_multiple_win_rates(configs, show_curriculum=True, save_path="comparison_training_curve.pdf")
