import abc
from typing import Union, Dict, List

import jax.numpy as jnp
import tensorflow as tf
import torch

Array = Union[jnp.ndarray, torch.Tensor, tf.Tensor]
ArrayDict = Union[Array, Dict[str, "ArrayDict"]]
Info = Dict[str, float]


class Algorithm:
    """
    Training:
    >>> for inputs, targets in train_data:
    >>>     for algorithm in algorithms:
    >>>         info = algorithm.update(inputs, targets)
    >>>         if log_interval():
    >>>             log(info)

    Evaluation:
    >>> for inputs, targets in eval_data:
    >>>     for algorithm in algorithms:
    >>>         evaluate(algorithm.infer(inputs), targets)
    """

    @abc.abstractmethod
    def update(self, inputs: ArrayDict, targets: ArrayDict, info: Info) -> Info:
        """
        We assume that all supervised learning algorithms learn some function from inputs to targets.
        This method assumes that inputs and targets are given in batch form and that the
        algorithm is agnostic to sampling method. The class is responsible for managing parameters and
         other state.

        :param inputs: batched input values
        :param targets: batched target values
        :param info: any metadata relevant to the update
        :return: the dictionary returned from this method should include information such as loss values
        that will get logged at certain intervals.
        """

    @abc.abstractmethod
    def infer(self, inputs: ArrayDict) -> ArrayDict:
        """
        Perform inference on given inputs. This method is primarily used for evaluation.

        :param inputs: batched input values
        :return: inferred target values
        """


class RLAlgorithm:
    """
    Observation:
    >>> while training:
    >>>     s1 = env.reset()
    >>>     a, log_p, h2 = policy(s1, h1)
    >>>     q = Q(s1, a)
    >>>     s2 = env.step(a)
    >>>     s2, r, t, i = env.step(a)
    >>>     for algorithm in algorithms:
    >>>         algorithm.observe(s1, h1, a, log_p, r, t, i, q)
    >>>     s1 = s2
    >>>     h1 = h2

    Updating (may vary per algorithm):
    >>> for algorithm in algorithms:
    >>>     info = algorithm.update()
    >>>     if log_interval():
    >>>         log(info)
    """

    @abc.abstractmethod
    def observe(
        self,
        observation: ArrayDict,
        hidden_state: Array,
        action: ArrayDict,
        action_log_prob: Array,
        reward: Array,
        done: Array,
        info: Info,
        value_estimate: Array = None,
    ):
        """
        We assume that all RL learning algorithms aggregate information from interactions with an MDP.
        This method assumes that arguments are given in batch form. The class is responsible for managing
        any internal state, such as replay buffers.

        :param observation: batch observation from the environment
        :param hidden_state: any internal hidden state maintained by the agent/value-estimator
        :param action: batch actions taken by agent conditioned on observation
        :param action_log_prob: log probability of actions taken by agent under policy induced by distribution
        :param reward: batch reward received for observation/action tuple
        :param done: whether action led to a terminal state
        :param info: metadata from the environment
        :param value_estimate: optional value estimate provided by external algorithm.
        """

    @abc.abstractmethod
    def update(self) -> Info:
        """
        This method performs updates to the parameters of the learned function, presumably based on the information
        aggregated via the `observe` method.

        :return: Information to be logged at certain intervals, such as loss. Cumulative reward, elapsed time-steps,
        and other common values will be logged elsewhere and should not be returned from this method.
        """
