import tensorflow as tf
import numpy as np
from typing import Union

def variable_summaries(var):
    name = var.op.name
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def init_var(name: str, size: list, init_value_max_magnitude: float):
    print("HEELLLO" + str(size))
    return tf.Variable(tf.random_uniform(size,
                       -init_value_max_magnitude,
                       init_value_max_magnitude),
                       dtype=tf.float32,
                       name=name)


class FeedForwardPolicy():
    def __init__(self, state_size: int,
            action_size: int, hidden_size: list=[],
                 random_seed: Union[None, int]=None,
                  init_value_max_magnitude: float=0.1) -> None:
        tf.reset_default_graph()
        if random_seed is not None:
            print("using non random seed")
            tf.set_random_seed(random_seed)
        init_value_magnitude_max = 0.1 

        """
        PLACEHOLDERS
        """
        self.episode_reward= tf.placeholder(tf.float32, shape=[],
                                    name="episode_reward")
        self.state = tf.placeholder(tf.float32, [1, state_size], name="state")
        self.target = tf.placeholder(tf.float32, name="target")
        self.applied_action = tf.placeholder(tf.float32,shape=[action_size], name="applied_action")
        """
        HIDDEN LAYERS 
        """
        self.bias_mean_hidden = [None] * len(hidden_size)
        self.bias_stddev_hidden = [None] * len(hidden_size)
        self.weights_mean_hidden = [None] * len(hidden_size)
        self.weights_stddev_hidden = [None] * len(hidden_size)
        self.hidden_mean_logits = [None] * len(hidden_size)
        self.hidden_mean_output = [None] * len(hidden_size)
        self.hidden_stddev_logits = [None] * len(hidden_size)
        self.hidden_stddev_output = [None] * len(hidden_size)
        for i, layer_size in enumerate(hidden_size):
            fanin_size = state_size if i == 0 else hidden_size[i - 1]
            self.bias_mean_hidden[i] = init_var("bias_mean_hidden_{}".format(i), [layer_size], init_value_max_magnitude)
            self.weights_mean_hidden[i] = init_var("weights_mean_hidden_{}".format(i), [fanin_size, layer_size], init_value_max_magnitude)
            self.hidden_mean_logits[i] = tf.add(tf.multiply(self.state, self.weights_mean_hidden[i]),
                                           self.bias_mean_hidden[i], name="hidden_mean_logit_{}".format(i))
            self.hidden_mean_output[i] = tf.nn.relu(self.hidden_mean_logits[i], name="hidden_mean_output_{}".format(i))

            self.bias_stddev_hidden[i] = init_var("bias_stddev_hidden_{}".format(i), [layer_size], init_value_max_magnitude)
            self.weights_stddev_hidden[i] = init_var("weights_stddev_hidden_{}".format(i), [fanin_size, layer_size], init_value_max_magnitude)
            self.hidden_stddev_logits[i] = tf.add(tf.multiply(self.state, self.weights_stddev_hidden[i]),
                                           self.bias_stddev_hidden[i], name="hidden_stddev_logit_{}".format(i))
            self.hidden_stddev_output[i] = tf.nn.relu(self.hidden_stddev_logits[i], name="hidden_stddev_output_{}".format(i))
        """
        OUTPUTS 
        """
        fanin_size = state_size if len(hidden_size) == 0 else hidden_size[i - 1]
        self.bias_mean = init_var("bias_mean", [action_size], init_value_max_magnitude)
        self.weights_mean = init_var("weight_mean", [fanin_size, action_size], init_value_max_magnitude)
        self.bias_stddev = init_var("bias_stddev", [action_size], init_value_max_magnitude)
        self.weights_stddev = init_var("weight_stddev", [fanin_size, action_size], init_value_max_magnitude)
        mean_input = self.state if len(hidden_size) == 0 else self.hidden_mean_output[-1]
        print(mean_input)
        print(self.weights_mean)
        stddev_input = self.state if len(hidden_size) == 0 else self.hidden_stddev_output[-1]
        self.mean = tf.add(tf.matmul(mean_input, self.weights_mean),
                           self.bias_mean)
        self.stddev = tf.add(tf.matmul(stddev_input, self.weights_stddev),
                             self.bias_stddev)
        self.normal_dist = tf.contrib.distributions.Normal(self.mean,
                                                           self.stddev)

        self.action = self.normal_dist._sample_n(1)

        self.action_clipped = tf.clip_by_value(self.action,
                                       clip_value_min=-1000.0,
                                       clip_value_max=1000.0, name="action")
        self.save_idx = 0

    def setup(self, saved_policy_path: str=""):
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        if (saved_policy_path is ""):
            self.sess.run(tf.global_variables_initializer())
        else:
            self.saver.restore(self.sess, "saved_policy_path")
        tf.summary.scalar('episode_reward', self.episode_reward)
        variable_summaries(self.weights_mean)
        variable_summaries(self.weights_stddev)
        variable_summaries(self.bias_mean)
        variable_summaries(self.bias_stddev)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('./train',
                                                      self.sess.graph)

    def choose_action(self, state: np.array):
        return self.sess.run((self.action, self.action_clipped), {self.state: state})


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

    def save_tensorboard(self, episode_reward):
        feed_dict={self.episode_reward: episode_reward}
        summary = self.sess.run(self.merged, feed_dict)
        self.train_writer.add_summary(summary, self.save_idx)
        self.save_idx += 1

    def get_model_parameters(self):
       # Get trainable parameter names. 
       variable_names = [var.name for var in 
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
       # Get trainable parameter values. 
       snapshot = self.sess.run(variable_names)  
       model_parameters = dict(zip(variable_names, snapshot))
       return model_parameters 
