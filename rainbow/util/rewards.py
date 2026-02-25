import numpy as np


class HockeyRewardManager:
    def __init__(self):
        self.reset()
        self.max_puck_speed = 20.0

    def reset(self):
        self.last_touch_frame = 0
        self.frame_count = 0
        self.prev_dist = None

    def get_shaped_reward(self, env, obs, info, env_reward):
        self.frame_count += 1
        reward = 0.0

        p1_pos = obs[0:2]
        p1_vel = obs[3:5]
        puck_pos = obs[12:14]
        if len(obs) > 16:  # keep_mode=True
            p1_has_puck = obs[16]

        reward += info.get("reward_closeness_to_puck", 0) * 0.02
        dist_to_puck = np.linalg.norm(p1_pos - puck_pos)
        if self.prev_dist is not None:
            delta_dist = self.prev_dist - dist_to_puck
            reward += np.clip(delta_dist * 0.1, -1.0, 1.0)
        self.prev_dist = dist_to_puck

        reward += 0.05 * info.get("reward_puck_direction", 0)

        if p1_has_puck >= 5:
            if (self.frame_count - self.last_touch_frame) > 10:
                reward += 0.15
                self.last_touch_frame = self.frame_count

        win_status = info.get("winner", 0)
        frames_since_touch = self.frame_count - self.last_touch_frame

        if win_status == 1:
            # Check for direct shot vs accidental goal
            reward += 10.0 if frames_since_touch < 30 else 7.0
        elif win_status == -1:
            reward -= 10.0
            # reward -= 10.0 if frames_since_touch > 15 else 5.0

        return reward

    def get_basic_reward_shaped(self, env, obs, info, env_reward):
        reward = env_reward
        reward += info.get("reward_puck_direction", 0) * 1.0
        reward += info.get("reward_touch_puck", 0) * 0.2
        return reward
