"""The reported best position must actually score the reported best cost.

Regression test: particles used to share one array between ``pos`` and
``best_pos``, so the in-place ``pos += velocity`` update dragged ``best_pos``
away from the point that earned ``best_cost``. Anything that persists or acts
on ``best_pos`` (a checkpoint, a drone steering to it) needs this invariant.
"""

import unittest

import numpy as np

from swarmopt import Swarm
from swarmopt.functions import rastrigin, sphere

ALGOS = ["global", "local", "unified", "sa", "hhoa", "cpso"]


class TestBestPositionIntegrity(unittest.TestCase):
    def test_best_pos_scores_best_cost(self):
        for func in (sphere, rastrigin):
            for algo in ALGOS:
                for seed in range(8):
                    with self.subTest(func=func.__name__, algo=algo, seed=seed):
                        s = Swarm(n_particles=10, dims=3, c1=2.0, c2=2.0, w=0.9,
                                  epochs=25, obj_func=func, algo=algo, seed=seed)
                        s.optimize()
                        self.assertTrue(np.isclose(func(s.best_pos), s.best_cost))

    def test_particle_arrays_are_independent(self):
        s = Swarm(n_particles=4, dims=3, c1=2.0, c2=2.0, w=0.9, epochs=1,
                  obj_func=sphere, seed=0)
        for p in s.swarm:
            self.assertFalse(np.shares_memory(p.pos, p.best_pos))
            self.assertFalse(np.shares_memory(p.pos, p.local_best_pos))
        self.assertFalse(any(np.shares_memory(s.best_pos, p.best_pos) for p in s.swarm))


if __name__ == "__main__":
    unittest.main()
