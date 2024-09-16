import unittest
from jax import random, numpy as jnp
from acm_planner.network.ddpg_critic import DDPGCritic


class TestCriticMethods(unittest.TestCase):
    def create_input(self):

        key1, key2 = random.split(random.key(0), 2)
        key_x1, key_x2 = random.split(key1, 2)
        x1 = random.uniform(key_x1, (4, 4))
        x2 = random.uniform(key_x2, (4, 4))

        return x1, x2, key2

    def test_ddpg(self):
        
        x1, x2, key = self.create_input()
        model = DDPGCritic(hidden_dim=[3, 4, 5])
        params = model.init(key, x1, x2)
        y = model.apply(params, x1, x2)

        self.assertEqual(params["params"]["Dense_0"]["bias"].shape[0], 3)
        self.assertEqual(params["params"]["Dense_0"]["kernel"].shape[0], 8)
        self.assertEqual(params["params"]["Dense_0"]["kernel"].shape[1], 3)

        self.assertEqual(y.shape[0], 4)
        self.assertEqual(y.shape[1], 5)


if __name__ == "__main__":
    unittest.main()
