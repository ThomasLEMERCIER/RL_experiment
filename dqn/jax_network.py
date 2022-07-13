from functools import partial
from typing import TypeAlias, Optional, Mapping

import jax
from jax import numpy as jnp, value_and_grad as vgrad, random as jrandom
import haiku as hk
import optax
import numpy as np



@hk.without_apply_rng
@hk.transform
def forward(x: Array) -> Array:
    model = MLP()
    return model(x)

@partial(jax.jit, static_argnums=(1))
def apply_gradients(params: dict, optimizer: optax.GradientTransformation, opt_state: dict, grads: dict):
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, new_opt_state
