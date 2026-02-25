import pandas as pd
import csv
import matplotlib.pyplot as plt
from tueplots import bundles

import pandas as pd
import matplotlib.pyplot as plt
import csv
import pickle
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np


def plot_multiple_win_rates(csv_configs, show_curriculum=False, save_path='', window=100):
    plt.rcParams.update(bundles.icml2024(column='full', usetex=False))
    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig, ax = plt.subplots(figsize=(plt.rcParams["figure.figsize"][0], 3))
    ax_reward = ax.twinx()

    global_max_ep = 0

    for config in csv_configs:
        path = config['path']
        train_data = []
        eval_data = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) == 7:
                    try:
                        train_data.append([float(row[0]), float(row[1])])
                    except (ValueError, IndexError):
                        continue
                elif len(row) == 13:
                    try:
                        eval_data.append([float(row[0]), float(row[7])])
                    except (ValueError, IndexError):
                        continue

        if not eval_data: continue

        df_eval = pd.DataFrame(eval_data, columns=['episode', 'win_rate'])
        df_train = pd.DataFrame(train_data, columns=['episode', 'reward'])
        global_max_ep = max(global_max_ep, df_eval['episode'].max())

        if not df_train.empty:
            df_train['reward_smooth'] = df_train['reward'].rolling(window=window, min_periods=1).mean()
            ax_reward.fill_between(
                df_train['episode'],
                df_train['reward_smooth'],
                color=config.get('color'),
                alpha=0
            )
            ax_reward.plot(
                df_train['episode'],
                df_train['reward_smooth'],
                color=config.get('color'),
                linestyle=':',
                linewidth=0.5,
                alpha=0.3
            )

        ax.plot(
            df_eval['episode'],
            df_eval['win_rate'],
            marker='o',
            markersize=4,
            linewidth=2,
            color=config.get('color'),
            label=config['label']
        )

    ax_reward.set_ylabel("Shaped Reward (Avg. $w=100$)", color='grey', fontsize=10)
    ax_reward.tick_params(axis='y', labelcolor='grey')
    ax_reward.spines['top'].set_visible(False)
    ax_reward.set_ylim(bottom=None, top=None)

    if show_curriculum:
        ax.axvspan(0, 2000, color='grey', alpha=0.1)
        ax.text(1050, 1.02, r'$\mathbb{P} =$' + '\n' + '$\{O_{weak}\}$', ha='center', fontsize=8)
        ax.axvspan(2000, 15000, color='blue', alpha=0.03)
        ax.text(8500, 1.02, r'$\mathbb{P} = \{0.4 O_{weak}, 0.6 O_{hard}\}$', ha='center', fontsize=8)
        if global_max_ep > 15000:
            ax.axvspan(15000, max(32000, global_max_ep), color='green', alpha=0.03)
            ax.text(23500, 1.02, r'$\mathbb{P} = \{0.5 O_{self}, 0.3 O_{hard}, 0.2 O_{weak}\}$', ha='center',
                    fontsize=8)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1.15)
    ax.set_xlim(0, max(32000, global_max_ep))

    ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.58, 0.02),
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

    plt.tight_layout()
    plt.show()
    fig.savefig(save_path, bbox_inches='tight')

if __name__ == "__main__":
    model_folder_path = r'\rainbow\models'

    configs = [

        {'path': r'C:\GitHub\laser-hockey\rainbow\models\20260218_221231_DQN-rainbow\log.csv', 'label': 'Rainbow-DQN',
         'color': '#AA3377'},
        {'path': r'C:\GitHub\laser-hockey\rainbow\models\20260218_215058_DQN-aug\log.csv',
         'label': 'Rainbow-DQN + Aug.', 'color': '#228833'},
        {'path': r'C:\GitHub\laser-hockey\rainbow\models\20260219_130526_DQN-quantile\log.csv',
         'label': 'Rainbow-DQN + QR-DQN', 'color': '#EEAA22'},
        {'path':  r'C:\GitHub\laser-hockey\rainbow\models\20260218_220628_DQN-quantile-aug\log.csv', 'label': 'Rainbow-DQN + QR-DQN + Aug.', 'color': '#66CCEE'},
        #{'path':  r'C:\GitHub\laser-hockey\rainbow\models\20260218_215301_DQN-quantile-reg\log.csv', 'label': 'DQN (Rainbow) + Quantile + Reg.', 'color': '#8d3da6'},
        {'path': r'C:\GitHub\laser-hockey\rainbow\models\20260218_215228_DQN-quantile-aug-reg\log.csv', 'label': 'Rainbow-DQN + QR-DQN + Aug + Reg.', 'color': '#4477AA'},
        {'path': r'C:\GitHub\laser-hockey\rainbow\models\20260223_194615_DQN-quantile-aug-reg_weak_bot\log.csv',
         'label': 'Rainbow-DQN + Quantile + Aug + Reg. vs. $O_{weak}$',
         'color': '#848587'},




    ]

    plot_multiple_win_rates(configs, show_curriculum=True, save_path="comparison_training_curve.pdf")