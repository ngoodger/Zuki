from gym import spaces
import random
import numpy as np


class TestContinuousActionEnv(object):
    def __init__(self):
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.info = None

    def reset(self):
        self.state = random.random()
        self.step_count = 0
        return self.state

    def step(self, action):
        self.state += action
        reward = np.array((-1 * abs(self.state)))
        self.step_count += 1
        terminal = False if self.step_count < 10 else True
        return (self.state, reward, terminal, self.info)
