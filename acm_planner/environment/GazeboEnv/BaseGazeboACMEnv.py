import jax.numpy as jnp
from jax import random
import jax
from gym import spaces

class BaseGazeboACMEnv:
   
   def __init__(self):
      self.random_limit = 0.05
   
   def _get_obsv(self, state):
      return state
   
   def _maybe_reset(self, env_state, done):
      key = env_state[1]
      return jax.lax.cond(done, self._reset, lambda key: env_state, key)
   
   def _reset(self, key):
      new_state = random.uniform(key, minval=-self.random_limit, 
                                      maxval=self.random_limit, shape=(4,))
      new_key = random.split(key)[0]
      return new_state, new_key

   def step(self, env_state, action):
      state, key = env_state
      new_state = state + action
      
      reward, done, info = 1., False, None
      
      env_state = new_state, key
      env_state = self._maybe_reset(env_state, done)
      new_state = env_state[0]
      return env_state, self._get_obsv(new_state), reward, done, info
      
   def reset(self, key):
      env_state = self._reset(key)
      new_state = env_state[0]
      return env_state, self._get_obsv(new_state)
   
key = random.PRNGKey(0)
env = BaseGazeboACMEnv()
env_state, inital_obsv = env.reset(key)
action = 1
env_state, obsv, reward, done, info = env.step(env_state, 1)