import gym
from zuki.policies.feed_forward import FeedForwardPolicy
from zuki.policy_gradient_methods.monte_carlo import MonteCarloPolicyGradient


def main():
    env = gym.make('Pendulum-v0')
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)
    print(env.action_space)
    print(env.action_space.high)
    print(env.action_space.low)
    learning_rate = 3e-2
    pg = MonteCarloPolicyGradient(env, learning_rate,
                                  FeedForwardPolicy, render=False)
    pg.run(1000000)


if __name__ == "__main__":
    main()
