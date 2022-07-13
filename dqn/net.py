from typing import TypeAlias

import jax
from jax import numpy as jnp
import haiku as hk

import numpy as np
Array: TypeAlias = jnp.DeviceArray

class MLP:
  def __init__(self, units: list[int]) -> None:
    super().__init__()
    self.units = units

  def __call__(self, x: Array) -> Array:
    for unit in self.units[:-1]:
      x = hk.Linear(unit)(x)
      x = jax.nn.relu(x)
    return hk.Linear(self.units[-1])(x)
