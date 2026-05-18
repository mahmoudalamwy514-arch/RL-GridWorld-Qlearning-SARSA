import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import matplotlib.pyplot as plt

from algorithms.q_learning import Q

policy = np.argmax(Q, axis=1)

arrows = {
    0: '↑',
    1: '↓',
    2: '←',
    3: '→'
}

grid = np.array([
    [arrows[a] for a in policy[:4]],
    [arrows[a] for a in policy[4:8]],
    [arrows[a] for a in policy[8:12]]
])

fig, ax = plt.subplots()

ax.set_xticks(np.arange(4))
ax.set_yticks(np.arange(3))

for i in range(3):
    for j in range(4):
        ax.text(j, i, grid[i, j],
                ha='center',
                va='center',
                fontsize=20)

plt.title("Learned Policy")
plt.show()