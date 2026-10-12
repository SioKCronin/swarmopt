import unittest
import warnings
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

    def test_respect_boundary_applies_to_standalone_optimizers(self):
        target = np.zeros(2)
        boundary = 0.75
        multiobjective_func = lambda x: np.array([np.sum(x ** 2), np.sum((x - 1) ** 2)])
        configs = {
            "cpso": {"algo": "cpso", "n_swarms": 2},
            "hhoa": {"algo": "hhoa"},
            "ppso": {"ppso_enabled": True},
            "multiobjective": {"multiobjective": True, "obj_func": multiobjective_func},
        }

        for name, overrides in configs.items():
            cfg = overrides.copy()
            obj_func = cfg.pop("obj_func", functions.sphere)
            with self.subTest(mode=name), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = Swarm(
                    n_particles=20,
                    dims=2,
                    c1=2.0,
                    c2=2.0,
                    w=0.9,
                    epochs=20,
                    obj_func=obj_func,
                    velocity_clamp=(0, 2),
                    target_position=target,
                    respect_boundary=boundary,
                    seed=1,
                    **cfg
                )
                s.optimize()

            self.assertIsNotNone(s.best_pos)
            self.assertGreaterEqual(np.linalg.norm(s.best_pos - target), boundary)
            if name == "ppso":
                for particle in s.ppso.particles:
                    self.assertGreaterEqual(np.linalg.norm(particle.pos - target), boundary)
                    self.assertGreaterEqual(np.linalg.norm(particle.best_pos - target), boundary)
            elif name == "hhoa":
                for horse in s.hhoa.horses:
                    self.assertGreaterEqual(np.linalg.norm(horse.pos - target), boundary)
                    self.assertGreaterEqual(np.linalg.norm(horse.best_pos - target), boundary)
            elif name == "multiobjective":
                for solution in s.mo_optimizer.archive:
                    self.assertGreaterEqual(np.linalg.norm(solution["pos"] - target), boundary)

if __name__ == "__main__":
    unittest.main()
