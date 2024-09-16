from flax import linen as nn
import jax.numpy as jnp

from typing import Callable, Sequence, Tuple

class DDPGCritic(nn.Module):

    hidden_dim : Sequence[int]

    @nn.compact
    def call(self, obs : jnp.ndarray, action : jnp.ndarray) -> jnp.ndarray:
        pass