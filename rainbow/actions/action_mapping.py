import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ActionMapper:
    def __init__(self, action_set, keep_mode=True):
        """
        action_set: list[DiscreteAction]
        """
        self.action_set = action_set
        self.keep_mode = keep_mode
        self.mirror_map = self._create_mirror_map()

    def __len__(self):
        return len(self.action_set)

    def map(self, action_idx: int) -> np.ndarray:
        if not (0 <= action_idx < len(self.action_set)):
            raise IndexError(f"Invalid action index: {action_idx}")

        a = self.action_set[action_idx]

        if self.keep_mode:
            return np.array(
                [a.dx, a.dy, a.dtheta, a.shoot],
                dtype=np.float32
            )
        else:
            return np.array(
                [a.dx, a.dy, a.dtheta],
                dtype=np.float32
            )

    def _create_mirror_map(self):
        """Finds the index of the mirrored version for every action."""
        mapping = {}
        for i, a in enumerate(self.action_set):
            for j, b in enumerate(self.action_set):
                if (np.isclose(a.dx, b.dx) and
                        np.isclose(a.dy, -b.dy) and
                        np.isclose(a.dtheta, -b.dtheta) and
                        np.isclose(a.shoot, b.shoot)):
                    mapping[i] = j
                    break
        return mapping

    def get_mirrored_index(self, action_idx):
        return self.mirror_map.get(action_idx, action_idx)

    def get_mirrored_obs(self, obs):
        """
        Reflects observation across the horizontal midline.
        Assumes indices: [p1_x, p1_y, p1_angle, p1_vx, p1_vy, p1_w, ...]
        """
        m_obs = obs.copy()
        # Negate Y-related components for P1, P2, and Puck
        # Indices: 1(p1_y), 2(p1_angle), 4(p1_vy), 5(p1_w),
        #          7(p2_y), 8(p2_angle), 10(p2_vy), 11(p2_w),
        #          13(puck_y), 15(puck_vy)
        negate_indices = [1, 2, 4, 5, 7, 8, 10, 11, 13, 15]
        m_obs[negate_indices] *= -1
        return m_obs
    
    def mirror_obs_tensor(self, obs):
        """ Vectorized mirroring on GPU """
        m_obs = obs.clone()
        negate_indices = [1, 2, 4, 5, 7, 8, 10, 11, 13, 15]
        m_obs[:, negate_indices] *= -1
        return m_obs

    
    def get_mirrored_obs_batch(self, obs_batch):
        """
        obs_batch: numpy array of shape [B, obs_dim]
        """
        m_obs = obs_batch.copy()
        negate_indices = [1, 2, 4, 5, 7, 8, 10, 11, 13, 15]
        m_obs[:, negate_indices] *= -1
        return m_obs

