from zuki.function_approximators.feed_forward import FeedForwardPolicy
from zuki.policy_gradient_methods.monte_carlo import MonteCarloPolicyGradient
from zuki.envs.simple_continuous_action_env import SimpleContinuousActionEnv
from typing import List


def test_continuous_action_env_positive():
    """
    Test continuous action environment with positive action scale
    Solution weight_mean = -1.0, bias_mean = 0.0
    """
    learning_rate = 3e-4
    env = SimpleContinuousActionEnv(action_scale=1.0, bias=0.0)
    pg = MonteCarloPolicyGradient(env, learning_rate,
                                  FeedForwardPolicy,
                                  render=False,
                                  random_seed=0, reward_sma_len=1000,
                                  hidden_size=[], entropy_weight=1e-3)
    avg_return = pg.run(50000)
    print("reward mean" + str(avg_return))
    assert(avg_return > -0.4)


def test_continuous_action_env_negative():
    """
    Test continuous action environment with positive action scale
    Solution weight_mean = 1.0, bias_mean = 0.0
    """
    env = SimpleContinuousActionEnv(action_scale=-2.0, bias=0.0)
    learning_rate = 3e-4
    pg = MonteCarloPolicyGradient(env, learning_rate,
                                  FeedForwardPolicy,
                                  render=False,
                                  random_seed=0, reward_sma_len=1000,
                                  hidden_size=[], entropy_weight=1e-3)
    avg_return = pg.run(50000)
    print("reward mean" + str(avg_return))
    assert(avg_return > -0.4)


def test_continuous_action_env_biased():
    """
    Test continuous action environment with positive action scale
    Solution weight_mean = -0.5, bias_mean = -0.25
    """
    env = SimpleContinuousActionEnv(action_scale=2.0, bias=0.5)
    learning_rate = 3e-4
    pg = MonteCarloPolicyGradient(env, learning_rate,
                                  FeedForwardPolicy,
                                  render=False,
                                  random_seed=0, reward_sma_len=1000,
                                  hidden_size=[], entropy_weight=1e-3)
    avg_return = pg.run(50000)
    print("reward mean" + str(avg_return))
    assert(avg_return > -0.4)


if __name__ == '__main__':
    test_continuous_action_env_biased()
