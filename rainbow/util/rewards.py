import numpy as np


class HockeyRewardManager:
    def __init__(self,
                 win_bonus=10.0,
                 puck_touch_bonus=0.5,
                 proximity_multiplier=1.0,
                 direction_multiplier=2.0):
        self.win_bonus = win_bonus
        self.puck_touch_bonus = puck_touch_bonus
        self.proximity_multiplier = proximity_multiplier
        self.direction_multiplier = direction_multiplier

    def get_shaped_reward(self, env_reward, info, obs):
        """
        Calculates a custom reward based on environment feedback.
        """
        # Goal: +10, Loss: -10
        total_reward = info.get("winner", 0)
        total_reward += info.get("reward_closeness_to_puck", 0) * self.proximity_multiplier
        total_reward += info.get("reward_puck_direction", 0) * self.direction_multiplier

        if info.get("reward_touch_puck", 0) > 0:
            total_reward += self.puck_touch_bonus

        #total_reward = np.clip(total_reward, -2.0, self.win_bonus)

        return float(total_reward)

    def update_multipliers(self, total_steps):
        """
        Reward Decay: Slowly reduce reward shaping as the agent gets better.
        """
        if total_steps > 500000:
            self.proximity_multiplier *= 0.999
            self.puck_touch_bonus *= 0.999
