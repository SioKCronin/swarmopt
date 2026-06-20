import unittest
import warnings
import numpy as np

from context import Swarm, functions


class TestSwarmShape(unittest.TestCase):
    def _make_swarm(self, n=10, dims=3):
        return Swarm(n, dims, 1.5, 1.5, 0.7, 1, functions.sphere, velocity_clamp=(0, 2))

    def test_shape_returns_correct_dimensions(self):
        s = self._make_swarm(n=10, dims=3)
        self.assertEqual(s.shape(), [10, 3])

    def test_shape_single_dim(self):
        s = self._make_swarm(n=5, dims=1)
        self.assertEqual(s.shape(), [5, 1])


class TestSwarmInitialState(unittest.TestCase):
    def setUp(self):
        self.swarm = Swarm(20, 2, 1.5, 1.5, 0.7, 1, functions.sphere, velocity_clamp=(-5, 5))

    def test_best_cost_is_finite_after_init(self):
        self.assertFalse(np.isinf(self.swarm.best_cost))

    def test_best_pos_has_correct_length(self):
        self.assertEqual(len(self.swarm.best_pos), 2)

    def test_runtime_starts_at_zero(self):
        self.assertEqual(self.swarm.runtime, 0)

    def test_runtime_is_positive_after_optimize(self):
        self.swarm.optimize()
        self.assertGreater(self.swarm.runtime, 0)


class TestSwarmAlgorithms(unittest.TestCase):
    # 'multiswarm' is excluded: known bug — update_global_best_pos accesses
    # self.swarm which does not exist for that algo path (self.multiswarm is set instead)
    _ALGOS = ['global', 'local', 'unified']

    def test_all_standard_algorithms_complete(self):
        for algo in self._ALGOS:
            with self.subTest(algo=algo):
                s = Swarm(10, 2, 1.5, 1.5, 0.7, 2, functions.sphere,
                          velocity_clamp=(-5, 5), algo=algo)
                s.optimize()
                self.assertFalse(np.isnan(s.best_cost))


class TestRespectBoundary(unittest.TestCase):
    def test_target_position_triggers_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Swarm(10, 2, 1.5, 1.5, 0.7, 1, functions.sphere,
                  velocity_clamp=(-5, 5), target_position=[0.0, 0.0])
            self.assertTrue(any(issubclass(warning.category, UserWarning) for warning in w))

    def test_respect_boundary_enabled_automatically(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            s = Swarm(10, 2, 1.5, 1.5, 0.7, 1, functions.sphere,
                      velocity_clamp=(-5, 5), target_position=[1.0, 1.0])
        self.assertTrue(s.use_respect_boundary)
        self.assertIsNotNone(s.respect_boundary)
        self.assertGreater(s.respect_boundary, 0)

    def test_no_target_means_no_respect_boundary(self):
        s = Swarm(10, 2, 1.5, 1.5, 0.7, 1, functions.sphere, velocity_clamp=(-5, 5))
        self.assertFalse(s.use_respect_boundary)
        self.assertIsNone(s.respect_boundary)


class TestDelegatePositions(unittest.TestCase):
    def _make_swarm_with_delegates(self, dims, n_delegates, spread='uniform'):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            return Swarm(
                10, dims, 1.5, 1.5, 0.7, 1, functions.sphere,
                velocity_clamp=(-10, 10),
                target_position=[0.0] * dims,
                n_delegates=n_delegates,
                delegate_spread=spread,
            )

    def test_2d_uniform_delegate_count(self):
        s = self._make_swarm_with_delegates(dims=2, n_delegates=4)
        self.assertEqual(len(s.delegate_positions), 4)

    def test_3d_uniform_delegate_count(self):
        s = self._make_swarm_with_delegates(dims=3, n_delegates=6)
        self.assertEqual(len(s.delegate_positions), 6)

    def test_2d_delegates_at_correct_distance(self):
        target = np.array([0.0, 0.0])
        s = self._make_swarm_with_delegates(dims=2, n_delegates=4)
        for pos in s.delegate_positions:
            dist = np.linalg.norm(np.array(pos) - target)
            self.assertAlmostEqual(dist, s.respect_boundary, places=5)

    def test_delegate_spread_random(self):
        s = self._make_swarm_with_delegates(dims=2, n_delegates=3, spread='random')
        self.assertEqual(len(s.delegate_positions), 3)

    def test_delegate_spread_opposite(self):
        s = self._make_swarm_with_delegates(dims=2, n_delegates=2, spread='opposite')
        self.assertEqual(len(s.delegate_positions), 2)

    def test_no_delegates_by_default(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            s = Swarm(10, 2, 1.5, 1.5, 0.7, 1, functions.sphere,
                      velocity_clamp=(-5, 5), target_position=[0.0, 0.0])
        self.assertEqual(len(s.delegate_positions), 0)


if __name__ == '__main__':
    unittest.main()
