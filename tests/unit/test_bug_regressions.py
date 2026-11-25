"""Regression tests for bugs found in the November 2026 review.

Each test names the bug it guards. ``TestOriginBias`` records nio
optimizers that still have a known bug: they are strict xfails, so fixing
one of those algorithms makes its test pass and fail the suite until the
marker is removed.
"""

import time
import warnings

import numpy as np
import pytest

from swarmopt import Swarm
from swarmopt.functions import sphere

BOUNDS = (-5.0, 5.0)


def _swarm(**kwargs):
    params = dict(n_particles=12, dims=3, c1=2.0, c2=2.0, w=0.9, epochs=15,
                  obj_func=sphere, velocity_clamp=BOUNDS, seed=0)
    params.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Swarm(**params)


def downhill(x):
    """Unbounded below: the optimum is wherever the box stops the search."""
    return -float(np.sum(x))


# -- O(n^3) epoch ------------------------------------------------------------

class TestNeighbourScans:
    @pytest.mark.parametrize("algo", ["global", "sa"])
    def test_global_best_algorithms_skip_the_neighbour_scan(self, algo, monkeypatch):
        calls = []
        original = Swarm.update_local_best_pos
        monkeypatch.setattr(Swarm, "update_local_best_pos",
                            lambda self: calls.append(1) or original(self))
        _swarm(algo=algo).optimize()
        assert calls == []

    def test_local_still_refreshes_neighbours_after_each_particle(self, monkeypatch):
        calls = []
        original = Swarm.update_local_best_pos
        monkeypatch.setattr(Swarm, "update_local_best_pos",
                            lambda self: calls.append(1) or original(self))
        s = _swarm(algo="local", n_particles=6, epochs=4)
        s.optimize()
        assert len(calls) == 6 * 4

    def test_global_epoch_cost_grows_linearly(self):
        def timed(n):
            s = _swarm(n_particles=n, dims=5, epochs=10)
            start = time.perf_counter()
            s.optimize()
            return time.perf_counter() - start

        timed(20)  # warm up
        small, large = min(timed(40) for _ in range(3)), min(timed(160) for _ in range(3))
        # 4x the particles: linear is ~4x, the old cubic loop was ~64x.
        assert large / small < 16


# -- particles leaving the search box ----------------------------------------

class TestSearchBox:
    @pytest.mark.parametrize("algo", ["global", "local", "unified", "sa"])
    def test_every_particle_stays_in_bounds(self, algo):
        s = _swarm(algo=algo, obj_func=downhill, epochs=30)
        seen = []
        s.epoch_callback = lambda epoch, swarm: seen.extend(p.pos.copy() for p in swarm.swarm)
        s.optimize()
        positions = np.array(seen + [p.pos for p in s.swarm])
        assert positions.min() >= BOUNDS[0] and positions.max() <= BOUNDS[1]

    @pytest.mark.parametrize("algo", ["global", "local", "unified", "sa", "cpso", "ppso", "hhoa"])
    def test_best_position_is_in_bounds(self, algo):
        s = _swarm(algo=algo, obj_func=downhill, dims=4, epochs=30)
        s.optimize()
        assert np.all(s.best_pos >= BOUNDS[0]) and np.all(s.best_pos <= BOUNDS[1])


# -- respect-boundary penalty -------------------------------------------------

class TestRespectPenalty:
    def _bounded(self, func):
        return _swarm(obj_func=func, dims=2, target_position=[0.0, 0.0],
                      respect_boundary=1.0)

    @pytest.mark.parametrize("cost", [-1.0, 0.0, 1.0])
    def test_inside_the_boundary_always_costs_more(self, cost):
        s = self._bounded(lambda x: cost)
        inside = s.objective_with_respect_boundary(np.array([0.1, 0.0]))
        outside = s.objective_with_respect_boundary(np.array([3.0, 0.0]))
        assert outside == cost
        assert inside > outside

    def test_penalty_grows_toward_the_target(self):
        s = self._bounded(lambda x: -2.0)
        costs = [s.objective_with_respect_boundary(np.array([r, 0.0]))
                 for r in (0.9, 0.5, 0.1)]
        assert costs[0] < costs[1] < costs[2]


# -- NaN objectives -----------------------------------------------------------

class TestNaNObjective:
    def test_all_nan_run_finishes_with_a_position(self):
        s = _swarm(obj_func=lambda x: float("nan"))
        s.optimize()
        assert s.best_pos is not None
        assert s.best_cost == float("inf")

    def test_nan_region_is_avoided(self):
        def half_nan(x):
            return float("nan") if x[0] > 0 else float(np.sum(x ** 2))

        s = _swarm(obj_func=half_nan, epochs=40)
        s.optimize()
        assert np.isfinite(s.best_cost)
        assert s.best_pos[0] <= 0


# -- input validation ---------------------------------------------------------

class TestValidation:
    @pytest.mark.parametrize("kwargs", [
        dict(n_particles=0), dict(n_particles=-3), dict(n_particles=2.5),
        dict(dims=0), dict(epochs=0), dict(epochs=-1), dict(epochs=True),
        dict(velocity_clamp=(5, -5)), dict(velocity_clamp=(1, 1)),
        dict(velocity_clamp=(-np.inf, 5)),
        dict(inertia_func="lineer"), dict(velocity_clamp_func="basci"),
    ])
    def test_rejects(self, kwargs):
        with pytest.raises(ValueError):
            _swarm(**kwargs)

    @pytest.mark.parametrize("kwargs", [dict(n_particles=0), dict(epochs=-1),
                                        dict(inertia_func="nope")])
    def test_rejects_for_nio_algorithms_too(self, kwargs):
        with pytest.raises(ValueError):
            _swarm(algo="gwo", **kwargs)

    def test_numpy_integers_are_accepted(self):
        _swarm(n_particles=np.int64(5), dims=np.int32(2), epochs=np.int64(3)).optimize()


# -- 'sa' never restarting its worst particle ---------------------------------

class TestSelfAdaptive:
    def test_worst_particle_is_tracked(self):
        s = _swarm(algo="sa")
        s.optimize()
        assert s.worst_pos is not None
        costs = [p.best_cost for p in s.swarm]
        assert s.worst_cost == max(costs)

    def test_sa_is_no_longer_identical_to_global(self):
        results = []
        for algo in ("global", "sa"):
            s = _swarm(algo=algo, epochs=25)
            s.optimize()
            results.append(s.best_cost)
        assert results[0] != results[1]


# -- nio optimizers biased toward the origin ----------------------------------

def _median_cost(algo, func, seeds=5):
    costs = []
    for seed in range(seeds):
        s = _swarm(algo=algo, obj_func=func, n_particles=30, dims=10,
                   epochs=100, velocity_clamp=(-5.12, 5.12), seed=seed)
        s.optimize()
        costs.append(s.best_cost)
    return float(np.median(costs))


SHIFT = np.full(10, 3.0)


def shifted_sphere(x):
    return float(np.sum((np.asarray(x) - SHIFT) ** 2))


ORIGIN_BIASED = ["cfa", "acroa", "gwo", "bat", "bha", "eho"]


class TestOriginBias:
    """Moving the sphere's optimum from 0 to (3, ..., 3) should not matter.

    These optimizers reach far better results when the optimum is at the
    origin, which usually means an update rule pulls toward zero (for
    example by scaling a position instead of a step). Each needs checking
    against its paper.
    """

    @pytest.mark.parametrize("algo", [
        pytest.param(a, marks=pytest.mark.xfail(strict=True, reason="origin bias"))
        for a in ORIGIN_BIASED
    ] + ["de", "global"])
    def test_shifting_the_optimum_does_not_change_the_result(self, algo):
        origin = _median_cost(algo, sphere)
        shifted = _median_cost(algo, shifted_sphere)
        assert shifted <= 1e3 * max(origin, 1e-8)
