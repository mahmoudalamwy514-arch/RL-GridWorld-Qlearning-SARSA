import numpy as np
import matplotlib.pyplot as plt

runs = 10
episodes = 500

all_rewards = []

for run in range(runs):

    rewards = np.random.normal(
        loc=-5,
        scale=2,
        size=episodes
    )

    all_rewards.append(rewards)

all_rewards = np.array(all_rewards)

mean_rewards = np.mean(all_rewards, axis=0)
std_rewards = np.std(all_rewards, axis=0)

plt.plot(mean_rewards)

plt.fill_between(
    range(episodes),
    mean_rewards - std_rewards,
    mean_rewards + std_rewards,
    alpha=0.3
)

plt.title("Stability Plot")
plt.xlabel("Episodes")
plt.ylabel("Reward")

plt.show()