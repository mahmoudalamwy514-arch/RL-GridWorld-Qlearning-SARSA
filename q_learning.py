import numpy as np
from grid_world import GridWorld

env = GridWorld()

# عدد الحالات
num_states = 3 * 4  # grid size

# عدد الأكشنز
num_actions = 4

# Q-table
Q = np.zeros((num_states, num_actions))

# parameters
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
        # epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.choice(num_actions)
        else:
            action = np.argmax(Q[state_idx])
        
        next_state, reward, done = env.step(action)
        total_reward += reward
        
        next_state_idx = state_to_index(next_state)
        
        # update Q-table
        Q[state_idx, action] = Q[state_idx, action] + alpha * (
            reward + gamma * np.max(Q[next_state_idx]) - Q[state_idx, action]
        )
        
        state_idx = next_state_idx

    q_rewards_per_episode.append(total_reward)
print("Q-Table:")
print(Q)


#Step 6: نختبر الـ Agent (Policy Testing)

print("\nTesting the learned policy:\n")

state = env.reset()
done = False

steps = 0

while not done and steps < 20:
    state_idx = state_to_index(state)
    
    # أفضل أكشن
    action = np.argmax(Q[state_idx])
    
    next_state, reward, done = env.step(action)
    
    print(f"State: {state} -> Action: {action} -> Next: {next_state} -> Reward: {reward}")
    
    state = next_state
    steps += 1
    
#رسم الجراف

import matplotlib.pyplot as plt

#smoothing للجراف
window = 50
smoothed = np.convolve(q_rewards_per_episode, np.ones(window)/window, mode='valid')

plt.plot(smoothed)

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curve")
plt.show()