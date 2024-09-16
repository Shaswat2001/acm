from flax import linen as nn
import jax.numpy as jnp

from typing import Callable, Sequence, Tuple
from acm_planner.network.base_net import default_init

class DDPGCritic(nn.Module):

    hidden_dim : Sequence[int]

    @nn.compact
    def __call__(self, obs : jnp.ndarray, action : jnp.ndarray) -> jnp.ndarray:
        
        input = jnp.concatenate([obs, action], axis=1)

        for i, size in enumerate(self.hidden_dim):
            input = nn.Dense(size, kernel_init=default_init())(input)
            input = nn.relu(input)
        
        return input



