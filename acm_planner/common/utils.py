from collections import namedtuple
import functools
import jax 
import jax.numpy as jnp
import numpy as np
import torch
import optax
import flax.linen as nn
from flax.training import train_state
from typing import Sequence

EPS = 1e-8

PolicyOps = namedtuple('PolicyOps', 'raw_mean mean log_std pi log_prob_pi')

@functools.partial(jax.jit, static_argnums=1)
def init_params(key: jax.Array, input_dims: Sequence[int], model: nn.Module):
    init_shape = jnp.ones(input_dims, jnp.float32)
    initial_params = model.init(key, init_shape)['params']
    return initial_params

def create_train_state(params, model: nn.Module, learning_rate: float) -> train_state.TrainState:
    
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state