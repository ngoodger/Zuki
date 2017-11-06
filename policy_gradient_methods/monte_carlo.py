import tensorflow as tf
import numpy as np
import time
import numpy as np
from collections import namedtuple

EpisodeMemory = namedtuple('EpisodeMemory', 'state action reward')

class MonteCarloPolicyGradient:
    def __init__(self, env, learning_rate, PolicyClass,
            render=False, saved_policy: str=""):
        self.env = env
        self.action_space = env.action_space.shape[0]
        self.observation_space = env.observation_space.shape[0]
        policy = PolicyClass(self.observation_space, self.action_space)
        #policy.loss = (policy.normal_dist.log_prob(policy.action) *
        #               policy.target - 1e-1 * policy.normal_dist.entropy())
        #policy.loss = -(policy.normal_dist.log_prob(policy.applied_action) *
        #               policy.target + 1e-1 * policy.normal_dist.entropy())
        policy.loss = -(policy.normal_dist.log_prob(policy.applied_action) *
                       policy.target)
        policy.train = (tf.train.AdamOptimizer(learning_rate)
                        .minimize(policy.loss))
        policy.setup(saved_policy)
        self.policy = policy
        self.render = render

    def observation_state(self, observation):
        return np.reshape(observation, (1, self.observation_space))

    def run(self):
        rewards = []
        for episode_num in range(1000000):
            print(episode_num)
            episode_memories = []
            observation = self.env.reset()
            state = self.observation_state(observation)
            terminal = False
            step_count = 0
            while not terminal:
                step_count += 1
                action = self.policy.choose_action(state)
                print("action: " + str(action))
                state_old = np.copy(state)
                print("state_old: " + str(state_old))
                observation, reward, terminal, info = self.env.step(action)
                self.env.render() if self.render else None
                state = self.observation_state(observation)
                episode_memories.append(EpisodeMemory(state=state_old, action=action, reward=reward[0]))
                print(reward.shape)
            step_return = 0.0
            for j in range(len(episode_memories) - 1, -1, -1):
                #print(episode_memories)
                time.sleep(1)
                state = episode_memories[j].state
                action = episode_memories[j].action
                step_return += episode_memories[j].reward[0]
                print("j " + str(j))
                print("state " + str(state))
                print("step_return " + str(step_return))
                print("action " + str(action))
                #print("values")
                #print(state)
                #print(action)
                self.policy.adjust(state, step_return, action)
            #rewards.append(step_return)
            if (len(rewards) > 1000):
                rewards.pop(0)
            #print("rewards")
            #print(episode_num)
            #print(step_return)
            self.policy.save_tensorboard(step_return)
