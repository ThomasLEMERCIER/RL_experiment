from functools import partial
from collections import namedtuple
from typing import Callable, TypeAlias
import numpy as np
import jax
from jax import numpy as jnp, random as jrandom
import haiku as hk
import optax

Array: TypeAlias = jnp.DeviceArray

class Agent:

  def __init__(self, net_apply: Callable[[hk.Params, Array], Array]) -> None:
    self.net = jax.jit(net_apply)
    self.discount_factor = 0.999
    self.n_actions = 2
    self.eps_end = 0.05
    self.eps_start = 0.9
    self.eps_decay = 200
    self.steps_done = 0
  
  @partial(jax.jit, static_argnums=(0,))
  def step(self, params_policy: hk.Params, state: Array) -> Array:
    Qvalues = self.net(params_policy, state)
    action: Array = jnp.argmax(Qvalues)
    return action

  def explore(self, params_policy: hk.Params, rng_key, state: Array) -> Array:
    sample = jrandom.uniform(rng_key, (1,))
    eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * jnp.exp(-1. * self.steps_done / self.eps_decay)
    self.steps_done += 1
    if sample > eps_threshold:
      return self.step(params_policy, state)
    return jrandom.randint(rng_key, (), minval=0, maxval=self.n_actions)
  
  @partial(jax.jit, static_argnums=(0,))
  def loss(self, params_policy: hk.Params, params_target: hk.Params, states: Array, actions: Array, rewards: Array, dones: Array, next_states: Array):
    expected_state_action_values = self.evaluate(params_target, rewards, next_states, dones)
    state_action_values = jnp.take_along_axis(self.net(params_policy, states), jnp.expand_dims(actions, axis=1), axis=1)
    return jnp.mean(optax.huber_loss(state_action_values, expected_state_action_values))

  @partial(jax.jit, static_argnums=(0,))
  def evaluate(self, params_target: hk.Params, rewards: Array, next_states: Array, dones: Array) -> Array:
    expected_state_action_values = rewards
    next_state_action_values = jnp.max(self.net(params_target, next_states), axis=1)
    expected_state_action_values += next_state_action_values * self.discount_factor * (1 - dones)
    return expected_state_action_values
