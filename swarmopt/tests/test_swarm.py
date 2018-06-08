import unittest
from context import Swarm
from functions import sphere_func

class TestSwarm(unittest.TestCase):
    def setUp(self):
        self.n_particles = 5
        self.dims = 2
        self.c1 = 0.7
        self.c2 = 0.4
        self.w = 2
        self.epochs = 500
        self.obj_func = sphere_func
        self.v_clamp = (10, 50)

    def test_initialize_swarm(self):
        s = Swarm(
            self.n_particles,
            self.dims,
            self.c1,
            self.c2,
            self.w,
            self.epochs,
            self.obj_func,
            self.v_clamp
        )
        self.assertIsInstance(s, Swarm)
        self.assertEqual(s.shape(), [5, 2])

    def test_optimize(self):
        pass
        s = Swarm(
            self.n_particles,
            self.dims,
            self.c1,
            self.c2,
            self.w,
            self.epochs,
            self.obj_func,
            self.v_clamp
        )
        s.optimize()
        self.assertEqual(s.best_cost, 0)
        self.assertEqual(s.best_pos, [1,1])

if __name__ == "__main__":
    unittest.main()
