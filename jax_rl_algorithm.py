from algorithm import RLAlgorithm, ArrayDict, Array, Info
from typing import Any, Callable, Mapping, Text

import jax
import jax.numpy as jnp
import optax
import rlax


# Batch variant of q_learning.
_batch_q_learning = jax.vmap(rlax.q_learning)


class DQN(RLAlgorithm):
    """Deep Q-Network agent."""

    def __init__(
        self,
        preprocessor,
        sample_network_input: jnp.ndarray,
        network,
        optimizer: optax.GradientTransformation,
        transition_accumulator: Any,
        replay,
        batch_size: int,
        exploration_epsilon: Callable[[int], float],
        min_replay_capacity_fraction: float,
        learn_period: int,
        target_network_update_period: int,
        grad_error_bound: float,
        rng_key,
    ):
        self._preprocessor = preprocessor
        self._replay = replay
        self._transition_accumulator = transition_accumulator
        self._batch_size = batch_size
        self._exploration_epsilon = exploration_epsilon
        self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
        self._learn_period = learn_period
        self._target_network_update_period = target_network_update_period

        # Initialize network parameters and optimizer.
        self._rng_key, network_rng_key = jax.random.split(rng_key)
        self._online_params = network.init(
            network_rng_key, sample_network_input[None, ...]
        )
        self._target_params = self._online_params
        self._opt_state = optimizer.init(self._online_params)

        # Other agent state: last action, frame count, etc.
        self._action = None
        self._frame_t = -1  # Current frame index.

        # Define jitted loss, update, and policy functions here instead of as
        # class methods, to emphasize that these are meant to be pure functions
        # and should not access the agent object's state via `self`.

        def loss_fn(online_params, target_params, transitions, rng_key):
            """Calculates loss given network parameters and transitions."""
            _, online_key, target_key = jax.random.split(rng_key, 3)
            q_tm1 = network.apply(online_params, online_key, transitions.s_tm1).q_values
            q_target_t = network.apply(
                target_params, target_key, transitions.s_t
            ).q_values
            td_errors = _batch_q_learning(
                q_tm1,
                transitions.a_tm1,
                transitions.r_t,
                transitions.discount_t,
                q_target_t,
            )
            td_errors = rlax.clip_gradient(
                td_errors, -grad_error_bound, grad_error_bound
            )
            losses = rlax.l2_loss(td_errors)
            assert losses.shape == (self._batch_size,)
            loss = jnp.mean(losses)
            return loss

        def update(rng_key, opt_state, online_params, target_params, transitions):
            """Computes learning update from batch of replay transitions."""
            rng_key, update_key = jax.random.split(rng_key)
            d_loss_d_params = jax.grad(loss_fn)(
                online_params, target_params, transitions, update_key
            )
            updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
            new_online_params = optax.apply_updates(online_params, updates)
            return rng_key, new_opt_state, new_online_params

        self._update = jax.jit(update)

        def select_action(rng_key, network_params, s_t, exploration_epsilon):
            """Samples action from eps-greedy policy wrt Q-values at given state."""
            rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
            q_t = network.apply(network_params, apply_key, s_t[None, ...]).q_values[0]
            a_t = rlax.epsilon_greedy().sample(policy_key, q_t, exploration_epsilon)
            return rng_key, a_t

        self._select_action = jax.jit(select_action)

    def observe(
        self,
        observation: ArrayDict,
        action: ArrayDict,
        reward: Array,
        done: Array,
        info: Info,
    ):
        if self.last is not None:
            self._replay.add(*self.last, reward, done, observation)

        self.last = observation, action

    def update(self):
        transitions = self._replay.sample(self._batch_size)
        self._rng_key, self._opt_state, self._online_params = self._update(
            self._rng_key,
            self._opt_state,
            self._online_params,
            self._target_params,
            transitions,
        )
