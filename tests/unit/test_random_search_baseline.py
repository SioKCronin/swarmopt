"""Every optimizer should beat random search on a problem it can't game.

The optimum is moved away from the origin (and the box centre), and random
search gets exactly as many objective evaluations as the optimizer used.
Marked ``slow``: run with ``pytest -m slow``.
"""

import warnings

import numpy as np
import pytest

import swarmopt
from swarmopt import Swarm

DIMS = 10
LO, HI = -5.12, 5.12
SHIFT = np.linspace(-3.5, 3.5, DIMS)
SEEDS = range(3)


class Counted:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return float(np.sum((np.asarray(x, dtype=float) - SHIFT) ** 2))


def _random_search(evaluations, seed):
    rng = np.random.default_rng(10_000 + seed)
    points = rng.uniform(LO, HI, (evaluations, DIMS))
    return float(np.min(np.sum((points - SHIFT) ** 2, axis=1)))


# Known bug: worse than random search at an equal budget. Strict, so a fix
# shows up as an unexpected pass. da, fa and iwdco also lose their
# best-so-far (see test_invariants.py), which explains part of it.
WORSE_THAN_RANDOM = {"aea", "cfa", "da", "fa", "gsa", "gso", "iwdco", "oio", "sda"}


@pytest.mark.slow
@pytest.mark.parametrize("algo", [
    pytest.param(a, marks=pytest.mark.xfail(strict=True, reason="worse than random search"))
    if a in WORSE_THAN_RANDOM else a
    for a in sorted(swarmopt.algorithms())
])
def test_beats_random_search_on_a_shifted_sphere(algo):
    wins = 0
    for seed in SEEDS:
        objective = Counted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            s = Swarm(30, DIMS, 1.5, 1.5, 0.7, 100, objective, algo=algo,
                      velocity_clamp=(LO, HI), inertia_func="constant", seed=seed)
        s.optimize()
        wins += s.best_cost < _random_search(objective.calls, seed)
    assert wins == len(SEEDS)
