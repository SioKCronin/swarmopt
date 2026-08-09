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

    def test_respect_boundary_accepts_explicit_distance(self):
        s = Swarm(
            n_particles=10,
            dims=2,
            c1=2.0,
            c2=2.0,
            w=0.8,
            epochs=2,
            obj_func=functions.sphere,
            velocity_clamp=self.v_clamp,
            target_position=[0.0, 0.0],
            respect_boundary=1.25,
        )

        self.assertTrue(s.use_respect_boundary)
        self.assertEqual(s.respect_boundary, 1.25)

    def test_respect_boundary_validates_target_dimensions(self):
        with self.assertRaisesRegex(ValueError, "target_position must contain exactly 3 values"):
            Swarm(
                n_particles=10,
                dims=3,
                c1=2.0,
                c2=2.0,
                w=0.8,
                epochs=2,
                obj_func=functions.sphere,
                velocity_clamp=self.v_clamp,
                target_position=[0.0, 0.0],
            )

    def test_respect_boundary_rejects_unsupported_optimizer_handoffs(self):
        cases = [
            {"algo": "cpso"},
            {"algo": "hhoa"},
            {"ppso_enabled": True},
            {"multiobjective": True},
        ]

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "Respect boundary is only supported"):
                    Swarm(
                        n_particles=10,
                        dims=2,
                        c1=2.0,
                        c2=2.0,
                        w=0.8,
                        epochs=2,
                        obj_func=functions.sphere,
                        velocity_clamp=self.v_clamp,
                        target_position=[0.0, 0.0],
                        **kwargs,
                    )

    def test_respect_boundary_penalty_does_not_reward_negative_objectives(self):
        def negative_objective(_):
            return -100.0

        s = Swarm(
            n_particles=10,
            dims=2,
            c1=2.0,
            c2=2.0,
            w=0.8,
            epochs=2,
            obj_func=negative_objective,
            velocity_clamp=self.v_clamp,
            target_position=[0.0, 0.0],
            respect_boundary=1.0,
        )

        outside_cost = s.objective_with_respect_boundary(np.array([2.0, 0.0]))
        inside_cost = s.objective_with_respect_boundary(np.array([0.0, 0.0]))

        self.assertEqual(outside_cost, -100.0)
        self.assertGreater(inside_cost, outside_cost)

if __name__ == "__main__":
    unittest.main()
