import abc
import numpy as np


class PolicyBase():
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def setup(self, saved_policy_path: str=""):
        """ Setup policy. Including initialization of variables and DAG. """
        pass

    @abc.abstractmethod
    def choose_action(self, state: np.array):
        """ Choose action based on state input """
        pass

    @abc.abstractmethod
    def adjust(self, state: np.array, target: float, action: np.array):
        """ Adjust policy based on state, target and action """
        pass
