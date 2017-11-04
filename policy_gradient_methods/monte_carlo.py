import tensorflow as tf
import numpy as np
import time
import numpy as np


def observation_state(observation):
    return np.reshape(observation, (1, 3))


class MonteCarloPolicyGradient:
    def __init__(self, env, learning_rate, PolicyClass,
            render=False, saved_policy:str=""):
        self.env = env
        action_space = env.action_space.shape[0]
        observation_space = env.observation_space.shape[0]
        self.episode_memory = []
        policy = PolicyClass(observation_space, action_space,
                             learning_rate)
        #policy.loss = (policy.normal_dist.log_prob(policy.action) *
        #               policy.target - 1e-1 * policy.normal_dist.entropy())
        policy.loss = (policy.normal_dist.log_prob(policy.action) *
                       policy.target)
        policy.train = (tf.train.AdamOptimizer(learning_rate)
                        .minimize(policy.loss))
        policy.setup(saved_policy)
        self.policy = policy
        self.render = render

    def run(self):
        rewards = []
        for episode_num in range(1000000):
            print(episode_num)
            self.episode_memory = []
            observation = self.env.reset()
            state = observation_state(observation)
            terminal = False
            step_count = 0
            while not terminal:
                step_count += 1
                action = self.policy.choose_action(state)
                state_old = state
                observation, reward, terminal, info = self.env.step(action)
                self.env.render() if self.render else None
                state = observation_state(observation)
                self.episode_memory.append((state_old, action, reward[0]))
            step_return = 0.0
            for j in range(len(self.episode_memory) - 1, 0, -1):
                state = self.episode_memory[j][0]
                action = self.episode_memory[j][1]
                step_return += self.episode_memory[j][2][0]
                #print("values")
                #print(state)
                #print(len(step_return))
                #print(action)
                self.policy.adjust(state, step_return, action)
            #rewards.append(step_return)
            if (len(rewards) > 1000):
                rewards.pop(0)
            #print("rewards")
            #print(episode_num)
            #print(step_return)
            self.policy.save_tensorboard(step_return)
