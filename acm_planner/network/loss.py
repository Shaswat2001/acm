import jax
import jax.numpy as jnp
from typing import Sequence,Any
import flax
import flax.linen as nn

@jax.jit
def policy_loss(buffer: Sequence[Any],network: Sequence[nn.Module]):

    state, _, _, _, _ = buffer
    PolicyNetwork, QNetwork, _, _ = network
    actions = PolicyNetwork(state)
    critic_value = QNetwork(state,actions)
    actor_loss = -jnp.mean(critic_value)

    return actor_loss

@jax.jit
def critic_loss(buffer: Sequence[Any],network: Sequence[nn.Module]):

    state, _, _, _, _ = buffer
    PolicyNetwork, QNetwork, _, _ = network
    actions = PolicyNetwork(state)
    critic_value = QNetwork(state,actions)
    actor_loss = -jnp.mean(critic_value)

    return actor_loss