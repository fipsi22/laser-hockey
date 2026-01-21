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

    def get_shaped_reward(self, env_reward, info, obs, episode_num):
        total_reward = env_reward
        '''
        winner = info.get("winner", 0)
        if winner == 1:
            total_reward += self.win_bonus
        elif winner == -1:
            total_reward -= self.win_bonus
        '''

        shaping_weight = max(0.1, 1.0 - (episode_num / 8_000))

        closeness = info.get("reward_closeness_to_puck", 0) * self.proximity_multiplier
        direction = info.get("reward_puck_direction", 0) * self.direction_multiplier

        total_reward += (closeness + direction) * shaping_weight

        if info.get("reward_touch_puck", 0) > 0:
            total_reward += self.puck_touch_bonus * shaping_weight

        # Clip and rescale reward for stability
        total_reward = np.clip(total_reward, -20.0, 20.0)
        #total_reward /= 10.0

        return float(total_reward)

    def update_multipliers(self, total_steps):
        """
        Reward Decay: Slowly reduce reward shaping as the agent gets better.
        """
        if total_steps > 500000:
            self.proximity_multiplier *= 0.999
            self.puck_touch_bonus *= 0.999
