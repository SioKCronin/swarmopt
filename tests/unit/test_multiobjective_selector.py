import unittest

import numpy as np

from context import Swarm
from swarmopt.utils.multiobjective import NSGA2PSO, SPEA2PSO
from swarmopt.utils.simple_multiobjective import SimpleMultiObjectivePSO, zdt1


class TestMultiObjectiveSelector(unittest.TestCase):
    def _make_swarm(self, mo_algorithm):
        np.random.seed(7)
        return Swarm(
            n_particles=6,
            dims=3,
            c1=1.0,
            c2=1.0,
            w=0.5,
            epochs=2,
            obj_func=zdt1,
            velocity_clamp=(0.0, 1.0),
            multiobjective=True,
            mo_algorithm=mo_algorithm,
            archive_size=12,
        )

    def test_nsga2_selector_uses_nsga2_optimizer(self):
        swarm = self._make_swarm("nsga2")

        swarm.optimize()

        self.assertIsInstance(swarm.mo_optimizer, NSGA2PSO)
        self.assertGreater(len(swarm.mo_optimizer.archive), 0)
        self.assertIsNotNone(swarm.best_pos)
        self.assertEqual(len(swarm.best_cost), 2)

    def test_spea2_selector_uses_spea2_optimizer(self):
        swarm = self._make_swarm("spea2")

        swarm.optimize()

        self.assertIsInstance(swarm.mo_optimizer, SPEA2PSO)
        self.assertGreater(len(swarm.mo_optimizer.archive), 0)
        self.assertIsNotNone(swarm.best_pos)
        self.assertEqual(len(swarm.best_cost), 2)

    def test_simple_selector_remains_available(self):
        swarm = self._make_swarm("simple")

        swarm.optimize()

        self.assertIsInstance(swarm.mo_optimizer, SimpleMultiObjectivePSO)
        self.assertGreater(len(swarm.mo_optimizer.archive), 0)
        self.assertIsNotNone(swarm.best_pos)

    def test_unknown_selector_raises(self):
        swarm = self._make_swarm("not-a-real-algorithm")

        with self.assertRaisesRegex(ValueError, "Unknown multiobjective algorithm"):
            swarm.optimize()


if __name__ == "__main__":
    unittest.main()
