#Environment
import numpy as np

class GridWorld:
    def __init__(self):
        self.grid = np.array([
            ['S', '.', '.', 'X'],
            ['.', 'X', '.', '.'],
            ['.', '.', '.', 'G']
        ])
        
        self.start = (0, 0)
        self.goal = (2, 3)
        self.state = self.start
        
    def reset(self):
        self.state = self.start
        return self.state
    
    def step(self, action):
        x, y = self.state
        
        # movements
        if action == 0:  # up
            x -= 1
        elif action == 1:  # down
            x += 1
        elif action == 2:  # left
            y -= 1
        elif action == 3:  # right
            y += 1
        
        # check boundaries
        x = max(0, min(x, 2))
        y = max(0, min(y, 3))
        
        next_state = (x, y)
        
        # reward
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
    
    
    #هنعمل تيست علشان نشوف environment شغالة ولا لا
    
if __name__ == "__main__":
        env = GridWorld()
        
        state = env.reset()
        print("Start:", state)
        
        for _ in range(5):
            action = np.random.choice([0,1,2,3])
            next_state, reward, done = env.step(action)
            
            print("Action:", action)
            print("Next State:", next_state)
            print("Reward:", reward)
            print("Done:", done)
            print("-----")
            
            if done:
                break