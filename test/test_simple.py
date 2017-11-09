from zuki.policies.feed_forward import FeedForwardPolicy
from zuki.policy_gradient_methods.monte_carlo import MonteCarloPolicyGradient
from zuki.envs.simple_continuous_action_env import SimpleContinuousActionEnv


def test_continuous_action_env_positive():
    """
    Test continuous action environment with positive action scale
    """
    env = SimpleContinuousActionEnv(action_scale=1.0, bias=0.0)
    learning_rate = 1e-3
    pg = MonteCarloPolicyGradient(env, learning_rate,
                                  FeedForwardPolicy,
                                  render=False,
                                  random_seed=0, reward_sma_len=1000)
    avg_return = pg.run(200000)
    print("reward mean" + str(avg_return))
    assert(avg_return > -0.4)


if __name__ == '__main__':
    test_continuous_action_env_positive()
