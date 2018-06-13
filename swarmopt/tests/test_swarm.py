import unittest
from context import Swarm
from functions import sphere_func

class TestSwarm(unittest.TestCase):
    def setUp(self):
        self.n_particles = 30
        self.dims = 2
        self.c1 = 0.5
        self.c2 = 0.3
        self.w = 0.9
        self.epochs = 5
        self.obj_func = sphere_func
        self.v_clamp = [-5.12, 5.12]

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
        self.assertEqual(s.shape(), [30, 2])
        self.assertNotEqual(s.best_pos[0], 0.0)
        self.assertNotEqual(s.best_pos[1], 0.0)
        self.assertNotEqual(s.best_cost, float('inf')) 

    def test_optimize(self):
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
        #self.assertLess(s.best_cost, 1)
        #self.assertEqual(s.best_pos, [1,1])


if __name__ == "__main__":
    unittest.main()
