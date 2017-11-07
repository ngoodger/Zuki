from zuki.policies.feed_forward import FeedForwardPolicy
from zuki.policy_gradient_methods.monte_carlo import MonteCarloPolicyGradient
from zuki.test.test_continuous_action_env import TestContinuousActionEnv


def main():
    env = TestContinuousActionEnv(1.0, 0.0)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)
    print(env.action_space)
    print(env.action_space.high)
    print(env.action_space.low)
    learning_rate = 3e-4
    pg = MonteCarloPolicyGradient(env, learning_rate,
                                  FeedForwardPolicy,
                                  render=False,
                                  random_seed=0)
    pg.run()


if __name__ == "__main__":
    main()
