import unittest
from context import Particle
from functions import sphere_func

class TestParticle(unittest.TestCase):
    def setUp(self):
        self.dims = 2
        self.val_min = 1
        self.val_max = 10
        self.max_start_velocity = 2
        self.obj_func = sphere_func

    def test_init(self):
        particle = Particle(
            self.val_min, self.val_max, self.max_start_velocity, self.dims, self.obj_func, seed=3
        )
        self.assertAlmostEqual(particle.pos[0], 5.9571811)
        self.assertAlmostEqual(particle.pos[1], 7.3733304)
        self.assertAlmostEqual(particle.best_pos[0], 5.9571811)
        self.assertAlmostEqual(particle.best_pos[1], 7.3733304)
        self.assertAlmostEqual(particle.best_cost, 89.85400817)


if __name__ == "__main__":
    unittest.main()
