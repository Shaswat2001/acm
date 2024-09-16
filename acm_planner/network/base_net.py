from typing import Optional
import jax.numpy as jnp
import flax.linen as nn

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)
