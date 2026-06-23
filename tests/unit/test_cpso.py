import unittest
import numpy as np
from context import Swarm


class TestCPSO(unittest.TestCase):
    def test_cpso_initializes_with_swarm_search_bounds(self):
        calls = []

        def bounded_objective(position):
            calls.append(np.array(position))
            return float(np.sum(position ** 2))

        swarm = Swarm(
            n_particles=4,
            dims=4,
            c1=1.5,
            c2=1.5,
            w=0.7,
            epochs=0,
            obj_func=bounded_objective,
            algo='cpso',
            velocity_clamp=(100.0, 110.0),
            n_swarms=2
        )

        swarm.optimize()

        self.assertGreater(len(calls), 0)
        for position in calls:
            self.assertTrue(np.all(position >= 100.0))
            self.assertTrue(np.all(position <= 110.0))
        self.assertTrue(np.all(swarm.cpso.global_context >= 100.0))
        self.assertTrue(np.all(swarm.cpso.global_context <= 110.0))

    def test_cpso_uses_symmetric_velocity_bounds_from_swarm(self):
        swarm = Swarm(
            n_particles=4,
            dims=4,
            c1=1.5,
            c2=1.5,
            w=0.7,
            epochs=0,
            obj_func=lambda position: float(np.sum(position ** 2)),
            algo='cpso',
            velocity_clamp=(100.0, 110.0),
            n_swarms=2
        )

        self.assertEqual(swarm.cpso.bounds, (100.0, 110.0))
        self.assertEqual(swarm.cpso.velocity_clamp, (-2.0, 2.0))
        for cooperative_swarm in swarm.cpso.swarms:
            self.assertEqual(cooperative_swarm.bounds, (100.0, 110.0))
            self.assertEqual(cooperative_swarm.velocity_clamp, (-2.0, 2.0))


if __name__ == "__main__":
    unittest.main()
