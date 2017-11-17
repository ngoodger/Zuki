import tensorflow as tf
import numpy as np
from typing import Union
from zuki.helpers.helpers import variable_summaries
from zuki.policies.policy_base import PolicyBase


class FeedForwardPolicy(PolicyBase):
    def __init__(self, state_size: int, action_size: int, hidden_size: list=[],
                 random_seed: Union[None, int]=None,
                 init_value_max_magnitude: float=0.1) -> None:
        tf.reset_default_graph()
        if random_seed is not None:
            print("using non random seed")
            tf.set_random_seed(random_seed)
        """
        PLACEHOLDERS
        """
        self.episode_reward = tf.placeholder(tf.float32, shape=[],
                                             name="episode_reward")
        self.state = tf.placeholder(tf.float32, [1, state_size], name="state")
        self.target = tf.placeholder(tf.float32, name="target")
        self.applied_action = tf.placeholder(tf.float32, shape=[action_size],
                                             name="applied_action")

        self.mean_hidden, self.stddev_hidden = [], []
        with tf.variable_scope('variables', reuse=False):
            for i, layer_size in enumerate(hidden_size):
                hidden_mean_in = (self.state if i == 0
                                  else self.hidden_mean_output[-1])
                hidden_stddev_in = (self.state if i == 0
                                    else self.hidden_stddev_output[-1])
                mean_layer_name = "mean_hidden_{}".format(i)
                stddev_layer_name = "stddev_hidden_{}".format(i)
                mean_layer = tf.layers.dense(inputs=hidden_mean_in,
                                             units=layer_size,
                                             activation=tf.nn.relu,
                                             name=mean_layer_name)
                self.mean_hidden.append(mean_layer)
                stddev_layer = tf.layers.dense(inputs=hidden_stddev_in,
                                               units=layer_size,
                                               activation=tf.nn.relu,
                                               name=stddev_layer_name)
                self.stddev_hidden.append(stddev_layer)
            """
            OUTPUTS
            """
            mean_input = (self.state if len(hidden_size) == 0
                          else self.hidden_mean_output[-1])
            stddev_input = (self.state if len(hidden_size) == 0
                            else self.hidden_stddev_output[-1])
            self.mean = tf.layers.dense(inputs=mean_input,
                                        units=action_size, name="mean")
            self.stddev = tf.layers.dense(inputs=stddev_input,
                                          units=action_size, name="stddev")
        self.normal_dist = tf.contrib.distributions.Normal(self.mean,
                                                           self.stddev)
        self.action = self.normal_dist._sample_n(1)
        self.action_clipped = tf.clip_by_value(self.action,
                                               clip_value_min=-1.0,
                                               clip_value_max=1.0,
                                               name="action")
        self.save_idx = 0

    def setup(self, saved_policy_path: str=""):
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        if (saved_policy_path is ""):
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, "saved_policy_path")
        tf.summary.scalar('episode_reward', self.episode_reward)
        with tf.variable_scope('variables', reuse=True):
            mean_weights = tf.get_variable("mean/kernel")
            mean_bias = tf.get_variable("mean/bias")
            stddev_weights = tf.get_variable("stddev/kernel")
            stddev_bias = tf.get_variable("stddev/bias")
        variable_summaries(mean_weights)
        variable_summaries(mean_bias)
        variable_summaries(stddev_weights)
        variable_summaries(stddev_bias)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("./train",
                                                  self.sess.graph)

    def choose_action(self, state: np.array) -> np.array:
        return self.sess.run((self.action, self.action_clipped),
                             {self.state: state})

    def adjust(self, state: np.array, target: float, action: np.array):
        feed_dict = {self.state: state, self.target: target,
                     self.applied_action: action[0][0]}
        try:
            ops = (self.train, self.loss)
        except AttributeError:
            print("Must define policy.loss and policy.train to adjust policy")
            raise
        _, step_loss = self.sess.run(ops, feed_dict)
        return step_loss

    def save_tensorboard(self, episode_reward: float) -> None:
        feed_dict = {self.episode_reward: episode_reward}
        summary = self.sess.run(self.merged, feed_dict)
        self.train_writer.add_summary(summary, self.save_idx)
        self.save_idx += 1
