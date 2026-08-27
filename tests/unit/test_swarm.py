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

    def test_explicit_respect_boundary_keyword(self):
        target = np.array([0.0, 0.0])

        def distance_to_target(pos):
            return np.linalg.norm(pos - target)

        with self.assertWarns(UserWarning):
            s = Swarm(
                n_particles=10,
                dims=2,
                c1=1.5,
                c2=1.5,
                w=0.7,
                epochs=3,
                obj_func=distance_to_target,
                velocity_clamp=(-5.0, 5.0),
                target_position=target,
                respect_boundary=2.0,
            )

        self.assertTrue(s.use_respect_boundary)
        self.assertEqual(s.respect_boundary, 2.0)
        s.optimize()
        self.assertGreaterEqual(np.linalg.norm(s.best_pos - target), 2.0)

    def test_automatic_respect_boundary_still_defaults_from_search_space(self):
        target = np.array([0.0, 0.0])

        with self.assertWarns(UserWarning):
            s = Swarm(
                n_particles=5,
                dims=2,
                c1=1.0,
                c2=1.0,
                w=0.5,
                epochs=1,
                obj_func=functions.sphere,
                velocity_clamp=(-10.0, 10.0),
                target_position=target,
            )

        expected = 0.1 * np.sqrt(2 * (20.0 ** 2))
        self.assertAlmostEqual(s.respect_boundary, expected)

    def test_respect_boundary_requires_target_position(self):
        with self.assertRaises(ValueError):
            Swarm(
                n_particles=5,
                dims=2,
                c1=1.0,
                c2=1.0,
                w=0.5,
                epochs=1,
                obj_func=functions.sphere,
                respect_boundary=2.0,
            )

if __name__ == "__main__":
    unittest.main()
