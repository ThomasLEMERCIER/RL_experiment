from functools import partial
import haiku as hk
import optax
from collections import namedtuple
import jax
from jax import value_and_grad as vgrad
from typing import TypeAlias

Array: TypeAlias = jax.numpy.ndarray

class Learner:
  """Slim wrapper around an agent/optimizer pair."""

  def __init__(self, agent, optimizer: optax.GradientTransformation):
    self.agent = agent
    self.optimizer = optimizer

  @partial(jax.jit, static_argnums=(0,))
  def update(self, params_policy: hk.Params, params_target: hk.Params, opt_state: optax.OptState, states: Array, actions: Array, rewards: Array, dones: Array, next_states: Array):
    loss, grads = jax.jit(vgrad(self.agent.loss))(params_policy, params_target, states, actions, rewards, dones, next_states)
    updates, opt_state = self.optimizer.update(grads, opt_state, params_policy)
    params_policy = optax.apply_updates(params_policy, updates)
    return params_policy, opt_state, loss
