import tensorflow as tf
import numpy as np
import time
import numpy as np
from collections import namedtuple
from typing import Union, List

EpisodeMemory = namedtuple('EpisodeMemory', 'state action reward')

class MonteCarloPolicyGradient:
    def __init__(self, env, learning_rate: float, PolicyClass,
                 render: bool=False, entropy_weight: float=1e-2, saved_policy: str="",
                 random_seed: Union[None, int]=None, reward_sma_len: Union[None, int]=None,
                 hidden_size:List[int]=[]):
        self.reward_sma_len = reward_sma_len
        if reward_sma_len is not None:
            self.reward_sma_array = np.ones(reward_sma_len)
            self.reward_sma_idx = 0
        self.env = env
        self.action_space = env.action_space.shape[0]
        self.observation_space = env.observation_space.shape[0]
        policy = PolicyClass(self.observation_space, self.action_space, random_seed=random_seed,
                             hidden_size=hidden_size)
        policy.loss = -(policy.normal_dist.log_prob(policy.applied_action) *
                       policy.target + entropy_weight * policy.normal_dist.entropy())
        policy.train = (tf.train.AdamOptimizer(learning_rate)
                        .minimize(policy.loss))
        policy.setup(saved_policy)
        self.policy = policy
        self.render = render
        self.first_episode = True

    def observation_state(self, observation):
        return np.reshape(observation, (1, self.observation_space))

    def run(self, episodes):
        rewards = []
        for episode_num in range(episodes):
            print(episode_num)
            episode_memories = []
            observation = self.env.reset()
            state = self.observation_state(observation)
            terminal = False
            step_count = 0
            while not terminal:
                step_count += 1
                action, action_clipped = self.policy.choose_action(state)
                #print("numpy shape " + str(action.shape))
                # print("action: " + str(action))
                state_old = np.copy(state)
                # print("state_old: " + str(state_old))
                observation, reward, terminal, info = self.env.step(action_clipped)
                self.env.render() if self.render else None
                state = self.observation_state(observation)
                episode_memories.append(EpisodeMemory(state=state_old, action=action, reward=reward[0]))
            step_return = 0.0
            for step_memory in reversed(episode_memories):
                state = step_memory.state
                action = step_memory.action
                step_return += step_memory.reward[0]
                #print("state " + str(state))
                #print("step_return " + str(step_return))
                #print("action " + str(action))
                self.policy.adjust(state, step_return, action)
            if self.reward_sma_len is not None:
                if self.first_episode:
                    self.reward_sma_array *= step_return
                self.reward_sma_array[self.reward_sma_idx] = step_return
                self.reward_sma_idx += 1
                if self.reward_sma_idx >= self.reward_sma_len:
                    self.reward_sma_idx = 0
            self.policy.save_tensorboard(step_return)
            self.first_episode = False
        return np.mean(self.reward_sma_array)
