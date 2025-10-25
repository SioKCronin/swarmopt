import unittest
import numpy as np
from context import Particle, Swarm
from functions import sphere

class TestParticle(unittest.TestCase):
    def setUp(self):
        np.random.seed(3)

        self.dims = 2
        self.w = 2
        self.c1 = 0.7
        self.c2 = 0.4
        self.val_min = 1
        self.val_max = 10
        self.max_start_velocity = 2
        self.obj_func = sphere
        self.swarm = Swarm(
            1, # n_particles
            self.dims,
            self.c1,
            self.c2,
            self.w,
            1, # epochs
            self.obj_func,
            (10, 50) # v_clamp
        )
        self.particle = Particle(
            self.swarm
        )

    def test_init(self):
        self.assertNotEqual(self.particle.pos[0], 0.0)
        self.assertNotEqual(self.particle.pos[1], 0.0)
        self.assertEqual(list(self.particle.best_pos), list(self.particle.pos))
        self.assertNotEqual(self.particle.best_cost, float('inf'))

    def test_cognitive_weight(self):
        self.particle.best_pos = np.random.uniform(self.val_min, self.val_max, self.dims)
        cognitive = self.particle.cognitive_weight()
        self.assertNotEqual(cognitive[0], 0.0)
        self.assertNotEqual(cognitive[1], 0.0)

    def test_social_weight(self):
        swarm_best_pos = np.random.uniform(self.val_min, self.val_max, self.dims)
        social = self.particle.social_weight()
        # self.assertNotEqual(social[0], 1.0)
        self.assertNotEqual(social[1], 1.0)

    def test_update(self):
        pos, velocity = np.copy(self.particle.pos), np.copy(self.particle.velocity)
        self.particle.update()
        self.assertNotEqual(self.particle.velocity[0], velocity[1])
        self.assertNotEqual(self.particle.pos[0], pos[0])
        self.assertNotEqual(self.particle.pos[1], pos[1])

    def tearDown(self):
        np.random.seed()

if __name__ == "__main__":
    unittest.main()
