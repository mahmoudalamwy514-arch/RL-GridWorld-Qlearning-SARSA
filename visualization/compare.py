import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import numpy as np
import matplotlib.pyplot as plt

from algorithms.q_learning import q_rewards_per_episode
from algorithms.sarsa import sarsa_rewards_per_episode

def smooth(data, window=50):

    return np.convolve(
        data,
        np.ones(window)/window,
        mode='valid'
    )

q_smooth = smooth(q_rewards_per_episode)
sarsa_smooth = smooth(sarsa_rewards_per_episode)

plt.plot(q_smooth, label="Q-Learning")
plt.plot(sarsa_smooth, label="SARSA")

plt.xlabel("Episode")
plt.ylabel("Total Reward")

plt.title("Q-Learning vs SARSA")

plt.legend()

plt.show()