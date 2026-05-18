import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np

from env.grid_world import GridWorld
from algorithms.q_learning import Q

env = GridWorld()

def state_to_index(state):
    return state[0] * 4 + state[1]

state = env.reset()

done = False

while not done:

    env.render()

    state_idx = state_to_index(state)

    action = np.argmax(Q[state_idx])

    next_state, reward, done = env.step(action)

    state = next_state

    time.sleep(1)