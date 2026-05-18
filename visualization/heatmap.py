
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from algorithms.q_learning import Q

values = np.max(Q, axis=1)
values = values.reshape((3, 4))

sns.heatmap(values, annot=True, cmap="viridis")

plt.title("Q-Value Heatmap")
plt.show()