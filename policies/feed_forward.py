import tensorflow as tf
import numpy as np

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

class FeedForwardPolicy():
    def __init__(self, state_size: int,
                 action_size: int, learning_rate: float, hidden_size: int=0) -> None:
        tf.reset_default_graph()
        self.episode_reward= tf.placeholder(tf.float32, shape=[],
                                    name="episode_reward")
        self.state = tf.placeholder(tf.float32, shape=[state_size],
                                    name="state")
        self.target = tf.placeholder(tf.float32, name="target")
        out_fanin_size = (state_size if hidden_size == 0
                          else hidden_size[-1])
        self.weights_mean = tf.Variable(tf.random_uniform([out_fanin_size,
                                                           action_size], -0.1,
                                                           0.1),
                                         dtype=tf.float32,
                                         name="w_mean")
        self.weights_stddev = tf.Variable(tf.random_uniform([out_fanin_size,
                                           action_size],
                                           -0.1, 0.1), dtype=tf.float32,
                                           name="w_stddev")
        self.bias_mean = tf.Variable(tf.random_uniform([action_size], -0.1,
                                                        0.1),
                                      dtype=tf.float32, name="b_mean")
        self.bias_stddev = tf.Variable(tf.random_uniform([action_size], -0.1,
                                        0.1),
                                        dtype=tf.float32, name="b_stddev")
        self.state = tf.placeholder(tf.float32, [1, state_size], name="state")


        if hidden_size == 0:
            self.mean = tf.add(tf.matmul(self.state, self.weights_mean),
                               self.bias_mean)
            self.stddev = tf.add(tf.matmul(self.state, self.weights_stddev),
                                 self.bias_stddev)
            self.normal_dist = tf.contrib.distributions.Normal(self.mean,
                                                               self.stddev)

        self.action_unclipped = self.normal_dist._sample_n(1)

        self.action = tf.clip_by_value(self.action_unclipped,
                                       clip_value_min=-1.0,
                                       clip_value_max=1.0)
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
        return self.sess.run(self.action, {self.state: state})


    def adjust(self, state: np.array, target: float, action: np.array):
        feed_dict = {self.state: state, self.target: target,
                     self.action: action}
        ops = [self.train, 
               self.loss]
        _, step_loss = self.sess.run(ops, feed_dict)
        return step_loss

    def save_tensorboard(self, episode_reward):
        feed_dict={self.episode_reward: episode_reward[0]}
        print("rewards" + str(episode_reward))
        summary = self.sess.run(self.merged, feed_dict)
        #print("rewards")
        #print(episode_reward)
        #print(episode_reward_ret)
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
