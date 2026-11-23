"""Every nio optimizer shares one step contract, checkpoints, and runs from Swarm."""

from __future__ import annotations

import math

import numpy as np
import pytest

import swarmopt
from swarmopt import Swarm
from swarmopt.nio.compare import budgeted_run
from swarmopt.nio.contract import StateError, drive, snapshot
from swarmopt.nio.registry import FAMILIES, FAMILY_NAMES, OPTIMIZERS, by_family, family
from swarmopt.persist import CheckpointMismatch, decode, encode


def rastrigin(x):
    return 10 * len(x) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in x)


class MemoryCheckpoint:
    """The SwarmCheckpoint interface, in memory, through the real JSON codec."""

    def __init__(self):
        self.blob = None
        self.saves = 0

    def save_state(self, state):
        self.blob = encode(state)
        self.saves += 1

    def load_state(self):
        return None if self.blob is None else decode(self.blob)


class Interrupt(Exception):
    pass


def build(name, seed=7):
    return OPTIMIZERS[name](objective=rastrigin, bounds=[(-5.12, 5.12)] * 3,
                            population_size=10, seed=seed)


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_step_contract(name):
    optimizer = build(name)
    optimizer.iterations = 3
    optimizer.initialise()
    for i in range(3):
        optimizer.iteration = i
        optimizer.step()
    assert len(optimizer.best_position) == 3
    assert math.isfinite(optimizer.best_fitness)


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_drive_is_reproducible(name):
    assert drive(build(name), 8) == drive(build(name), 8)


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_resume_matches_uninterrupted_run(name):
    expected = drive(build(name), 12)
    store = MemoryCheckpoint()

    def stop(_optimizer, iteration):
        if iteration == 6:
            raise Interrupt

    with pytest.raises(Interrupt):
        drive(build(name), 12, checkpoint=store, checkpoint_every=5, callback=stop)
    assert store.saves == 1
    assert drive(build(name), 12, checkpoint=store, checkpoint_every=5) == expected


def test_checkpoint_refuses_a_different_run():
    store = MemoryCheckpoint()
    drive(build("gwo"), 5, checkpoint=store)
    with pytest.raises(CheckpointMismatch):
        drive(build("de"), 5, checkpoint=store)
    with pytest.raises(CheckpointMismatch):
        drive(build("gwo"), 6, checkpoint=store)


def test_snapshot_refuses_foreign_objects():
    optimizer = build("gwo")
    optimizer.initialise()
    optimizer.extra = object()
    with pytest.raises(StateError):
        snapshot(optimizer, 5, next_iteration=0)


def test_families_cover_the_registry():
    assert set(FAMILIES) == set(OPTIMIZERS)
    assert set(FAMILIES.values()) == set(FAMILY_NAMES)
    assert family("GWO") == "swarm"
    assert family("de") == "evolutionary"
    assert set(by_family("physical")) >= {"gsa", "bha", "wca"}


def test_budgeted_run_respects_the_budget():
    for name in ("gwo", "fwa", "bfo", "de"):
        _, value, used = budgeted_run(name, rastrigin, [(-5.12, 5.12)] * 3, 400, seed=1)
        assert used <= 400
        assert math.isfinite(value)


# ---------------------------------------------------------------- Swarm

def np_rastrigin(x):
    return float(10 * len(x) + np.sum(x * x - 10 * np.cos(2 * np.pi * x)))


def make_swarm(algo, **kw):
    base = dict(n_particles=12, dims=3, c1=2.0, c2=2.0, w=0.9, epochs=15,
                obj_func=np_rastrigin, algo=algo, velocity_clamp=(-5.12, 5.12), seed=3)
    base.update(kw)
    return Swarm(**base)


def test_swarm_lists_every_algorithm():
    names = swarmopt.algorithms()
    assert "global" in names and "gwo" in names
    assert "pso" not in names and "ppso" not in names
    assert len(names) == len(swarmopt.CORE_ALGORITHMS) + len(OPTIMIZERS) - 2
    assert set(swarmopt.algorithms("evolutionary")) == set(by_family("evolutionary"))


@pytest.mark.parametrize("algo", sorted(set(OPTIMIZERS) - {"pso", "ppso"}))
def test_swarm_runs_every_nio_algorithm(algo):
    swarm = make_swarm(algo)
    swarm.optimize()
    assert swarm.best_pos.shape == (3,)
    assert np.all(swarm.best_pos >= -5.12) and np.all(swarm.best_pos <= 5.12)
    assert math.isclose(np_rastrigin(swarm.best_pos), swarm.best_cost, rel_tol=1e-9, abs_tol=1e-12)


def test_swarm_nio_run_is_reproducible():
    a, b = make_swarm("gwo"), make_swarm("gwo")
    a.optimize()
    b.optimize()
    assert a.best_cost == b.best_cost
    np.testing.assert_array_equal(a.best_pos, b.best_pos)


def test_swarm_seed_sequence_is_reproducible():
    a = make_swarm("de", seed=np.random.SeedSequence(5))
    b = make_swarm("de", seed=np.random.SeedSequence(5))
    a.optimize()
    b.optimize()
    assert a.best_cost == b.best_cost


def test_swarm_passes_algo_params():
    swarm = make_swarm("wca", algo_params={"num_rivers": 2})
    swarm.optimize()
    assert swarm.nio_optimizer.num_rivers == 2
    with pytest.raises(ValueError):
        make_swarm("global", algo_params={"num_rivers": 2})


def test_pso_and_ppso_route_to_the_core():
    pso = make_swarm("pso")
    assert pso.algo == "global" and pso.nio_algo is None
    ppso = make_swarm("ppso")
    assert ppso.algo == "global" and ppso.ppso_enabled and ppso.nio_algo is None


def test_swarm_nio_resume_matches_uninterrupted():
    expected = make_swarm("gwo")
    expected.optimize()

    store = MemoryCheckpoint()
    interrupted = make_swarm("gwo")

    def stop(epoch, _swarm):
        if epoch == 9:
            raise Interrupt

    interrupted.epoch_callback = stop
    with pytest.raises(Interrupt):
        interrupted.optimize(checkpoint=store, checkpoint_every=4)

    resumed = make_swarm("gwo")
    resumed.optimize(checkpoint=store, checkpoint_every=4)
    assert resumed.best_cost == expected.best_cost
    np.testing.assert_array_equal(resumed.best_pos, expected.best_pos)


@pytest.mark.parametrize("kw", [
    dict(target_position=[0.0, 0.0, 0.0]),
    dict(diversity_monitoring=True),
    dict(variation_strategy="gaussian"),
    dict(topology="ring"),
])
def test_swarm_refuses_pso_only_features_for_nio(kw):
    with pytest.raises(ValueError):
        make_swarm("gwo", **kw)
