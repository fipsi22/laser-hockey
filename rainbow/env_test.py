import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
from importlib import reload
import time

from actions.action_sets import baseline_action_set, compound_action_set
from actions.action_mapping import ActionMapper

np.set_printoptions(suppress=True)

env = h_env.HockeyEnv()
env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)

obs, info = env.reset()

mapper = ActionMapper(compound_action_set(), keep_mode=True)

for i in range(mapper.n_actions()):
    print(i, mapper.to_continuous(i))

'''
for t in range(200):
    env.render(mode="human")

    # test discrete action
    if t < 100:
        action_idx = 1
    elif t < 150:
        action_idx = 0
    else:
        action_idx = 13
    a1 = mapper.to_continuous(action_idx)

    # opponent does nothing
    a2 = np.zeros_like(a1)

    action = np.hstack([a1, a2])
    obs, r, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break
'''



for _ in range(200):
    env.render(mode="human")

    # Move forward aggressively
    action = np.array([1.0, 0.0, 0.0, 0.0,   # player 1
                       0.0, 0.0, 0.0, 0.0])  # player 2
    obs, r, d, t, info = env.step(action)

    print("has puck:", env.player1_has_puck)
env.close()
