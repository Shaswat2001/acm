from typing import Optional
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)

class ContinuousMLP(nn.Module):

    hidden_dim: Sequence[int]
    n_action: int
    bound: Sequence[float]

    @nn.compact
    def __call__(self, obs : jnp.ndarray):

        for i, size in enumerate(self.hidden_dim):
            obs = nn.Dense(size, kernel_init=default_init())(obs)
            obs = nn.relu(obs)

        obs = nn.Dense(self.n_action, kernel_init=default_init())(obs)
        obs = nn.sigmoid(obs)
        obs = jnp.multiply(obs,self.bound)

        return obs