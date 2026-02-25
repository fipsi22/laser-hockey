import json
import os
import pandas as pd



def save_run_config(config, save_dir):
    """Saves the hyperparams to a JSON file."""
    config_path = os.path.join(save_dir, "config.json")
    serializable_config = {k: str(v) for k, v in config.items()}
    with open(config_path, "w") as f:
        json.dump(serializable_config, f, indent=4)


class TrainingTracker:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.csv_path = os.path.join(save_dir, "log.csv")
        self.history = []

    def log_episode(self, episode_data):
        """Appends a single episode's stats to a CSV file."""
        self.history.append(episode_data)

        df = pd.DataFrame([episode_data])
        df.to_csv(self.csv_path, mode='a', header=not os.path.exists(self.csv_path), index=False)

    def save_summary(self):
        """Saves final stats for easy plotting later."""
        summary_path = os.path.join(self.save_dir, "summary.json")