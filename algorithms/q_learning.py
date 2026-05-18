import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.grid_world import GridWorld

env = GridWorld()

num_states = 12
num_actions = 4

Q = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 500

def state_to_index(state):
    return state[0] * 4 + state[1]

q_rewards_per_episode = []

for episode in range(episodes):

    state = env.reset()
    state_idx = state_to_index(state)

    total_reward = 0
    done = False

    while not done:

        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)

        else:
            action = np.argmax(Q[state_idx])

        next_state, reward, done = env.step(action)

        total_reward += reward

        next_state_idx = state_to_index(next_state)

        Q[state_idx, action] += alpha * (
            reward + gamma * np.max(Q[next_state_idx])
            - Q[state_idx, action]
        )

        state_idx = next_state_idx

    q_rewards_per_episode.append(total_reward)

print(Q)