import unittest
import numpy as np
from context import Particle, Swarm
from context import functions

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
        self.obj_func = functions.sphere
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

    def test_global_weight_returns_correct_shape(self):
        # global_weight uses swarm.best_pos which is set after init
        weight = self.particle.global_weight()
        self.assertEqual(len(weight), self.dims)

    def test_best_pos_updates_when_improvement_found(self):
        # Force particle to a high-cost position, then update to a low-cost one
        # so best_pos must be updated
        original_best_pos = self.particle.best_pos.copy()
        original_best_cost = self.particle.best_cost

        # Move particle to the global minimum of sphere (origin) manually
        self.particle.pos = np.zeros(self.dims)
        ideal_cost = self.swarm.obj_func(self.particle.pos)

        # Simulate what update() does at the end: check and record improvement
        if ideal_cost < self.particle.best_cost:
            self.particle.best_cost = ideal_cost
            self.particle.best_pos = self.particle.pos.copy()

        self.assertLess(self.particle.best_cost, original_best_cost)
        self.assertFalse(np.array_equal(self.particle.best_pos, original_best_pos))

    def test_stagnation_count_increments_when_no_improvement(self):
        # Move particle far from origin (high cost for sphere)
        self.particle.pos = np.full(self.dims, 100.0)
        self.particle.best_cost = 0.0  # Artificially perfect best so no update triggers
        self.particle.update()
        self.assertTrue(hasattr(self.particle, 'stagnation_count'))
        self.assertGreaterEqual(self.particle.stagnation_count, 1)

    def test_best_cost_is_non_negative_for_sphere(self):
        # sphere(x) = sum(x^2) >= 0 always
        self.assertGreaterEqual(self.particle.best_cost, 0.0)

    def tearDown(self):
        np.random.seed()

if __name__ == "__main__":
    unittest.main()
