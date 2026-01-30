import gym
from gym import spaces
import numpy as np
import random


class GridWorldEnv(gym.Env):
    def __init__(self, grid_size=5):
        super().__init__()

        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=3,
            shape=(grid_size * grid_size,),
            dtype=np.int32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent_pos = [0, 0]
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]
        self.obstacles = {(1, 1), (2, 2)}

        return self._get_obs(), {}

    def step(self, action):
        x, y = self.agent_pos

        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        elif action == 3:
            y += 1

        x = np.clip(x, 0, self.grid_size - 1)
        y = np.clip(y, 0, self.grid_size - 1)

        self.agent_pos = [x, y]

        reward = -0.1
        done = False

        if tuple(self.agent_pos) in self.obstacles:
            reward = -1

        if self.agent_pos == self.goal_pos:
            reward = 5
            done = True

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        for o in self.obstacles:
            grid[o] = 2

        grid[tuple(self.agent_pos)] = 1
        grid[tuple(self.goal_pos)] = 3

        return grid.flatten()

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".", dtype=str)

        for o in self.obstacles:
            grid[o] = "X"

        grid[tuple(self.agent_pos)] = "A"
        grid[tuple(self.goal_pos)] = "G"

        print("\n".join(" ".join(row) for row in grid))
        print()
