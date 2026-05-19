import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class GridWorld:

    def __init__(self):

        self.grid = np.array([
            ['S', '.', '.', 'X'],
            ['.', 'X', '.', '.'],
            ['.', '.', '.', 'G']
        ])

        self.rows = 3
        self.cols = 4

        self.start = (0, 0)
        self.goal = (2, 3)

        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):

        x, y = self.state

        # up
        if action == 0:
            x -= 1

        # down
        elif action == 1:
            x += 1

        # left
        elif action == 2:
            y -= 1

        # right
        elif action == 3:
            y += 1

        x = max(0, min(x, self.rows - 1))
        y = max(0, min(y, self.cols - 1))

        next_state = (x, y)

        if self.grid[x][y] == 'G':
            reward = 10
            done = True

        elif self.grid[x][y] == 'X':
            reward = -10
            done = True

        else:
            reward = -1
            done = False

        self.state = next_state

        return next_state, reward, done

    def render(self):

        visual = np.zeros((self.rows, self.cols))

        for i in range(self.rows):
            for j in range(self.cols):

                if self.grid[i][j] == 'X':
                    visual[i][j] = -1

                elif self.grid[i][j] == 'G':
                    visual[i][j] = 2

        x, y = self.state
        visual[x][y] = 1

        cmap = colors.ListedColormap(
            ['black', 'white', 'blue', 'green']
        )

        bounds = [-1, 0, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(visual, cmap=cmap, norm=norm)

        plt.grid(True)
        plt.pause(2.5)
        plt.clf()