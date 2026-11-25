"""Properties every optimizer behind ``Swarm`` must have, whatever the algorithm.

These run over ``swarmopt.algorithms()``, so a newly registered optimizer is
covered without editing this file. The hypothesis tests vary the search
space for the core PSO family, where most of the per-particle code lives.
"""

import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import swarmopt
from swarmopt import Swarm
from swarmopt.functions import rastrigin, sphere

ALGOS = sorted(swarmopt.algorithms())
# Algorithms whose run goes through the per-epoch loop or nio.drive, so
# epoch_callback fires. cpso, ppso and hhoa run their own loop.
CALLBACK_ALGOS = [a for a in ALGOS if a not in ("cpso", "hhoa")]
CORE_LOOP = ["global", "local", "unified", "sa"]

# Known bug: these report the best of the current population, not the best
# found so far, so a run can end worse than it was earlier. Strict, so a fix
# shows up as an unexpected pass.
LOSES_BEST = {"da", "fa", "iwdco"}


def _known(algos, broken, reason):
    return [pytest.param(a, marks=pytest.mark.xfail(strict=True, reason=reason))
            if a in broken else a for a in algos]
LO, HI = -5.12, 5.12


def _swarm(algo, obj=rastrigin, dims=4, epochs=12, n=10, seed=0, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Swarm(n, dims, 1.5, 1.5, 0.7, epochs, obj, algo=algo,
                     velocity_clamp=(LO, HI), seed=seed, **kwargs)


@pytest.fixture(scope="module")
def finished_runs():
    runs = {}
    for algo in ALGOS:
        s = _swarm(algo)
        s.optimize()
        runs[algo] = s
    return runs


@pytest.mark.parametrize("algo", ALGOS)
def test_best_cost_is_the_cost_of_best_pos(algo, finished_runs):
    s = finished_runs[algo]
    assert s.best_cost == pytest.approx(rastrigin(np.asarray(s.best_pos)), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("algo", ALGOS)
def test_best_pos_is_inside_the_search_box(algo, finished_runs):
    pos = np.asarray(finished_runs[algo].best_pos, dtype=float)
    assert pos.shape == (4,)
    assert np.all(pos >= LO) and np.all(pos <= HI)


@pytest.mark.parametrize("algo", ALGOS)
def test_result_is_finite(algo, finished_runs):
    s = finished_runs[algo]
    assert np.isfinite(s.best_cost)
    assert np.all(np.isfinite(s.best_pos))


@pytest.mark.parametrize("algo", _known(CALLBACK_ALGOS, LOSES_BEST, "loses its best-so-far"))
def test_best_so_far_never_gets_worse(algo):
    s = _swarm(algo, epochs=15)
    history = []
    s.epoch_callback = lambda epoch, swarm: history.append(
        swarm.nio_optimizer.best_fitness if swarm.nio_algo else swarm.best_cost)
    s.optimize()
    history.append(s.best_cost)
    assert len(history) == 16
    assert all(b <= a for a, b in zip(history, history[1:]))


@pytest.mark.parametrize("algo", _known(ALGOS, {"da"}, "loses its best-so-far"))
def test_does_better_than_its_starting_population(algo):
    # Not a quality bar: only that the run moved at all on an easy problem.
    s = _swarm(algo, obj=sphere, epochs=30, n=15)
    s.optimize()
    rng = np.random.default_rng(0)
    start = rng.uniform(LO, HI, (15, 4))
    assert s.best_cost < np.median([sphere(x) for x in start])


# -- property-based: core PSO family ------------------------------------------

bounds = st.tuples(st.floats(-100, 99), st.floats(0.5, 100)).map(lambda t: (t[0], t[0] + t[1]))
SETTINGS = settings(max_examples=25, deadline=None,
                    suppress_health_check=[HealthCheck.too_slow])


def _objective(kind):
    return {
        "sphere": sphere,
        "downhill": lambda x: -float(np.sum(x)),          # optimum on the box
        "negative": lambda x: -1.0 - float(np.sum(x ** 2)),  # costs below zero
    }[kind]


@SETTINGS
@given(algo=st.sampled_from(CORE_LOOP), dims=st.integers(1, 6), box=bounds,
       n=st.integers(2, 12), seed=st.integers(0, 2**32 - 1),
       kind=st.sampled_from(["sphere", "downhill", "negative"]))
def test_every_particle_stays_in_the_box_every_epoch(algo, dims, box, n, seed, kind):
    s = Swarm(n, dims, 2.0, 2.0, 0.9, 8, _objective(kind), algo=algo,
              velocity_clamp=box, seed=seed)
    seen = []
    s.epoch_callback = lambda epoch, swarm: seen.extend(p.pos.copy() for p in swarm.swarm)
    s.optimize()
    pos = np.array(seen + [p.pos for p in s.swarm])
    assert pos.min() >= box[0] and pos.max() <= box[1]
    for p in s.swarm:
        assert p.best_cost == s._cost(p.best_pos)


@SETTINGS
@given(algo=st.sampled_from(CORE_LOOP), dims=st.integers(2, 5), seed=st.integers(0, 2**32 - 1),
       radius=st.floats(0.2, 2.0), kind=st.sampled_from(["sphere", "downhill", "negative"]))
def test_no_particle_ever_enters_the_respect_boundary(algo, dims, seed, radius, kind):
    target = np.zeros(dims)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        s = Swarm(10, dims, 2.0, 2.0, 0.9, 10, _objective(kind), algo=algo,
                  velocity_clamp=(-5, 5), target_position=target,
                  respect_boundary=radius, seed=seed)
    closest = []
    s.epoch_callback = lambda epoch, swarm: closest.append(
        min(np.linalg.norm(p.pos - target) for p in swarm.swarm))
    s.optimize()
    closest.append(min(np.linalg.norm(p.pos - target) for p in s.swarm))
    assert min(closest) >= radius
    assert np.linalg.norm(s.best_pos - target) >= radius


# -- objectives that return unusual types --------------------------------------

@pytest.mark.parametrize("algo", ["global", "local", "cpso", "gwo", "de"])
@pytest.mark.parametrize("wrap", [
    pytest.param(lambda v: int(round(v)), id="int"),
    pytest.param(np.float32, id="float32"),
    pytest.param(lambda v: np.array(v), id="0d-array"),
])
def test_objective_return_types(algo, wrap):
    s = _swarm(algo, obj=lambda x: wrap(float(np.sum(np.asarray(x) ** 2))))
    s.optimize()
    assert np.isfinite(float(s.best_cost))


@pytest.mark.parametrize("algo", ["global", "local", "sa", "cpso", "gwo"])
def test_objective_errors_propagate(algo):
    class Boom(RuntimeError):
        pass

    def explode(x):
        raise Boom

    with pytest.raises(Boom):
        _swarm(algo, obj=explode).optimize()
