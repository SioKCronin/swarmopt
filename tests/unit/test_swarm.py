import unittest
import numpy as np
from context import Swarm
from context import functions

class TestSwarm(unittest.TestCase):
    def setUp(self):
        self.n_particles = 30
        self.dims = 2
        self.c1 = 0.5
        self.c2 = 0.3
        self.w = 0.9
        self.epochs = 5
        self.obj_func = functions.sphere
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

    def test_sa_initializes_global_worst_position(self):
        s = Swarm(
            n_particles=5,
            dims=self.dims,
            c1=self.c1,
            c2=self.c2,
            w=self.w,
            epochs=self.epochs,
            obj_func=self.obj_func,
            algo='sa',
            velocity_clamp=self.v_clamp,
        )
        worst_particle = max(s.swarm, key=lambda particle: particle.best_cost)

        self.assertEqual(s.worst_cost, worst_particle.best_cost)
        np.testing.assert_allclose(s.worst_pos, worst_particle.best_pos)

    def test_global_worst_position_is_recomputed(self):
        s = Swarm(
            n_particles=2,
            dims=self.dims,
            c1=self.c1,
            c2=self.c2,
            w=self.w,
            epochs=self.epochs,
            obj_func=self.obj_func,
            algo='sa',
            velocity_clamp=self.v_clamp,
        )
        s.swarm[0].best_cost = 10.0
        s.swarm[0].best_pos = np.array([10.0, 10.0])
        s.swarm[1].best_cost = 5.0
        s.swarm[1].best_pos = np.array([5.0, 5.0])
        s.update_global_worst_pos()
        self.assertEqual(s.worst_cost, 10.0)

        s.swarm[0].best_cost = 1.0
        s.swarm[0].best_pos = np.array([1.0, 1.0])
        s.update_global_worst_pos()

        self.assertEqual(s.worst_cost, 5.0)
        np.testing.assert_allclose(s.worst_pos, np.array([5.0, 5.0]))

    def test_swarm_with_velocity_clamping(self):
        # Test with different velocity clamping functions
        clamping_types = ['none', 'basic', 'adaptive']
        for clamp_type in clamping_types:
            with self.subTest(clamp_type=clamp_type):
                s = Swarm(
                    n_particles=10,
                    dims=3,
                    c1=2.0,
                    c2=2.0,
                    w=0.8,
                    epochs=3,
                    obj_func=functions.sphere,
                    velocity_clamp=self.v_clamp,
                    velocity_clamp_func=clamp_type
                )
                s.optimize()
                self.assertIsNotNone(s.best_cost)
                self.assertFalse(np.isnan(s.best_cost))

    def test_swarm_with_variation(self):
        # Test with different variation strategies
        variation_strategies = ['gaussian', 'adaptive', 'boundary']
        for strategy in variation_strategies:
            with self.subTest(strategy=strategy):
                s = Swarm(
                    n_particles=10,
                    dims=2,
                    c1=2.0,
                    c2=2.0,
                    w=0.8,
                    epochs=3,
                    obj_func=functions.sphere,
                    velocity_clamp=self.v_clamp,
                    variation_strategy=strategy,
                    variation_rate=0.1,
                    variation_strength=0.05
                )
                s.optimize()
                self.assertIsNotNone(s.best_cost)
                self.assertFalse(np.isnan(s.best_cost))

    def test_swarm_with_combined_features(self):
        # Test with both velocity clamping and variation
        s = Swarm(
            n_particles=15,
            dims=3,
            c1=1.5,
            c2=1.5,
            w=0.7,
            epochs=4,
            obj_func=functions.rosenbrock,
            velocity_clamp=self.v_clamp,
            velocity_clamp_func='adaptive',
            variation_strategy='gaussian',
            variation_rate=0.2,
            variation_strength=0.1
        )
        s.optimize()
        self.assertIsNotNone(s.best_cost)
        self.assertFalse(np.isnan(s.best_cost))
        self.assertEqual(len(s.best_pos), 3)

if __name__ == "__main__":
    unittest.main()
