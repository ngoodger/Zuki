# zuki
Reinforcement Learning Library

policy gradient methods
monte-carlo policy gradient


policies
feed forward

value functions

algorithms
reinforce(PolicyClass, ValueFunctionClass)
policy(function_approximator=feedforward)
value function (function_approximator=feed_forward)

subclass or object?

policy and value functions are subclasses of function approximator
policy_feed_forward(env, function_approximator)
value_function_feed_forward_function(env, function_approximator)
reinforce(environment, policy, value_function)
