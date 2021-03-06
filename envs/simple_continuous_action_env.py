from gym import spaces
import random
import numpy as np


class SimpleContinuousActionEnv(object):
    def __init__(self, action_scale, bias):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.info = None
        self.observation = np.zeros(1)
        self.reward = np.zeros((1, 1))
        self.action_scale = action_scale
        self.bias = bias
        random.seed(0)

    def reset(self):
        self.state = 2 * random.random() - 1.0
        self.step_count = 0
        self.observation[0] = self.state
        return self.observation

    def step(self, action):
        if action > 1.0:
            action = 1.0
        if action < -1.0:
            action = -1.0
        self.state += self.action_scale * action + self.bias
        self.reward[0, 0] = -1 * abs(self.state)
        self.step_count += 1
        terminal = False if self.step_count < 2 else True
        self.observation[0] = self.state
        return (self.observation, self.reward, terminal, self.info)
