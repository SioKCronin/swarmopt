"""Same seed must give the same run, without touching global random state."""

import random
import unittest
import warnings

import numpy as np

from swarmopt import Swarm
from swarmopt.functions import rastrigin, sphere

BASE = dict(n_particles=12, dims=4, c1=2.0, c2=2.0, w=0.9, epochs=15, obj_func=rastrigin)

CONFIGS = {
    "global": dict(algo="global"),
    "local": dict(algo="local"),
    "unified": dict(algo="unified"),
    "sa": dict(algo="sa"),
    "cpso": dict(algo="cpso"),
    "hhoa": dict(algo="hhoa"),
    "multiobjective": dict(
        multiobjective=True,
        obj_func=lambda x: np.array([np.sum(x ** 2), np.sum((x - 2) ** 2)]),
    ),
    "random_inertia": dict(inertia_func="random"),
    "variation": dict(variation_strategy="gaussian"),
    "diversity": dict(diversity_monitoring=True, variation_strategy="gaussian"),
    "ppso": dict(ppso_enabled=True),
    "respect_boundary": dict(target_position=[1.0, 1.0, 1.0, 1.0], respect_boundary=0.5),
}


def run(seed, **overrides):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        swarm = Swarm(**{**BASE, **overrides}, seed=seed)
        swarm.optimize()
    return swarm


class TestReproducibility(unittest.TestCase):

    def test_same_seed_same_result(self):
        for name, cfg in CONFIGS.items():
            with self.subTest(config=name):
                a, b = run(7, **cfg), run(7, **cfg)
                np.testing.assert_array_equal(np.asarray(a.best_cost), np.asarray(b.best_cost))
                np.testing.assert_array_equal(np.asarray(a.best_pos), np.asarray(b.best_pos))

    def test_different_seed_different_result(self):
        a, b = run(1), run(2)
        self.assertFalse(np.array_equal(np.asarray(a.best_pos), np.asarray(b.best_pos)))

    def test_does_not_consume_global_random_state(self):
        np.random.seed(0)
        random.seed(0)
        expected = (np.random.random(), random.random())
        np.random.seed(0)
        random.seed(0)
        run(3, variation_strategy="gaussian", inertia_func="random")
        self.assertEqual((np.random.random(), random.random()), expected)

    def test_accepts_generator(self):
        a = run(np.random.default_rng(5))
        b = run(np.random.default_rng(5))
        self.assertEqual(a.best_cost, b.best_cost)


class TestMultiswarmUnavailable(unittest.TestCase):

    def test_multiswarm_raises_clearly(self):
        with self.assertRaises(NotImplementedError):
            run(0, algo="multiswarm")


class TestRespectBoundary(unittest.TestCase):

    def test_explicit_distance(self):
        s = run(0, obj_func=sphere, target_position=[0, 0, 0, 0], respect_boundary=1.5)
        self.assertEqual(s.respect_boundary, 1.5)
        self.assertGreaterEqual(np.linalg.norm(s.best_pos), 1.5 - 1e-9)

    def test_default_distance_is_ten_percent_of_diagonal(self):
        s = run(0, target_position=[0, 0, 0, 0])
        diagonal = np.sqrt(s.dims * (s.val_max - s.val_min) ** 2)
        self.assertAlmostEqual(s.respect_boundary, 0.1 * diagonal)

    def test_rejects_invalid_distance(self):
        for bad in (0, -1, float("inf"), float("nan")):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                run(0, target_position=[0, 0, 0, 0], respect_boundary=bad)

    def test_rejects_boundary_without_target(self):
        with self.assertRaises(ValueError):
            run(0, respect_boundary=1.0)


if __name__ == "__main__":
    unittest.main()
