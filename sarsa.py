import numpy as np
from grid_world import GridWorld

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

sarsa_rewards_per_episode = []

for episode in range(episodes):
    state = env.reset()
    state_idx = state_to_index(state)
    
    # أول action
    if np.random.rand() < epsilon:
        action = np.random.choice(num_actions)
    else:
        action = np.argmax(Q[state_idx])
    
    total_reward = 0
    done = False
    
    while not done:
        next_state, reward, done = env.step(action)
        total_reward += reward
        next_state_idx = state_to_index(next_state)
        
        # action الجاية
        if np.random.rand() < epsilon:
            next_action = np.random.choice(num_actions)
        else:
            next_action = np.argmax(Q[next_state_idx])
        
        # SARSA update
        Q[state_idx, action] = Q[state_idx, action] + alpha * (
            reward + gamma * Q[next_state_idx, next_action] - Q[state_idx, action]
        )
        
        state_idx = next_state_idx
        action = next_action
        
        
        total_reward += reward
    
    sarsa_rewards_per_episode.append(total_reward)

print("Q-Table (SARSA):")
print(Q)

# رسم الجراف
import matplotlib.pyplot as plt

window = 50
smoothed = np.convolve(sarsa_rewards_per_episode, np.ones(window)/window, mode='valid')

plt.plot(smoothed)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("SARSA Learning Curve")
plt.show()