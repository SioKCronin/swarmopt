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

    def assertOutsideRespectBoundary(self, swarm, position):
        distance = np.linalg.norm(np.asarray(position) - swarm.target_position)
        self.assertGreaterEqual(distance + 1e-12, swarm.respect_boundary)

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

    def test_respect_boundary_penalty_is_positive_at_target(self):
        target = np.array([0.0, 0.0])

        def distance_to_target(x):
            return np.linalg.norm(x - target)

        s = Swarm(
            n_particles=5,
            dims=2,
            c1=1.0,
            c2=1.0,
            w=0.5,
            epochs=1,
            obj_func=distance_to_target,
            velocity_clamp=(-1.0, 1.0),
            target_position=target,
        )

        self.assertGreater(s.objective_with_respect_boundary(target), 0.0)

    def test_respect_boundary_applies_to_delegated_single_objective_optimizers(self):
        target = np.array([0.0, 0.0])

        def distance_to_target(x):
            return np.linalg.norm(x - target)

        cases = [
            ("ppso", {"ppso_enabled": True}),
            ("hhoa", {"algo": "hhoa"}),
            ("cpso", {"algo": "cpso", "n_swarms": 2}),
        ]

        for seed, (name, kwargs) in enumerate(cases):
            with self.subTest(name=name):
                np.random.seed(seed)
                swarm = Swarm(
                    n_particles=8,
                    dims=2,
                    c1=1.2,
                    c2=1.2,
                    w=0.5,
                    epochs=3,
                    obj_func=distance_to_target,
                    velocity_clamp=(-1.0, 1.0),
                    target_position=target,
                    **kwargs,
                )

                swarm.optimize()
                self.assertOutsideRespectBoundary(swarm, swarm.best_pos)

                if name == "ppso":
                    for particle in swarm.ppso.particles:
                        self.assertOutsideRespectBoundary(swarm, particle.pos)
                        self.assertOutsideRespectBoundary(swarm, particle.best_pos)
                elif name == "hhoa":
                    for horse in swarm.hhoa.horses:
                        self.assertOutsideRespectBoundary(swarm, horse.pos)
                        self.assertOutsideRespectBoundary(swarm, horse.best_pos)
                elif name == "cpso":
                    self.assertOutsideRespectBoundary(swarm, swarm.cpso.global_context)
                    self.assertOutsideRespectBoundary(swarm, swarm.cpso.global_best_pos)

    def test_respect_boundary_applies_to_multiobjective_optimizer(self):
        target = np.array([0.0, 0.0])

        def objectives(x):
            return np.array([
                np.linalg.norm(x - target),
                np.sum((x - np.array([0.5, -0.5])) ** 2),
            ])

        np.random.seed(11)
        swarm = Swarm(
            n_particles=8,
            dims=2,
            c1=1.2,
            c2=1.2,
            w=0.5,
            epochs=3,
            obj_func=objectives,
            velocity_clamp=(-1.0, 1.0),
            target_position=target,
            multiobjective=True,
            archive_size=20,
        )

        swarm.optimize()
        self.assertOutsideRespectBoundary(swarm, swarm.best_pos)
        for particle in swarm.mo_optimizer.particles:
            self.assertOutsideRespectBoundary(swarm, particle["pos"])
            self.assertOutsideRespectBoundary(swarm, particle["best_pos"])
        for solution in swarm.mo_optimizer.archive:
            self.assertOutsideRespectBoundary(swarm, solution["pos"])
            self.assertOutsideRespectBoundary(swarm, solution["best_pos"])

if __name__ == "__main__":
    unittest.main()
