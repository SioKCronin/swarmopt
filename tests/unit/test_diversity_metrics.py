"""Every diversity metric must tell a spread-out swarm from a collapsed one."""

import numpy as np
import pytest

from swarmopt.utils.diversity import calculate_swarm_diversity

METHODS = ["euclidean", "manhattan", "variance", "radius", "entropy"]


def _swarm(scale, seed=0):
    rng = np.random.default_rng(seed)
    return list(rng.uniform(-1, 1, (20, 3)) * scale)


@pytest.mark.parametrize("method", [
    pytest.param(m, marks=pytest.mark.xfail(
        strict=True,
        reason="known bug: histogram bins span the swarm's own min..max, so the "
               "metric is scale-invariant and never sees the swarm shrink"))
    if m == "entropy" else m
    for m in METHODS
])
def test_shrinking_the_swarm_lowers_diversity(method):
    wide = calculate_swarm_diversity(_swarm(5.0), method)
    narrow = calculate_swarm_diversity(_swarm(0.01), method)
    assert narrow < wide


@pytest.mark.parametrize("method", METHODS)
def test_identical_particles_have_no_diversity(method):
    same = [np.array([1.0, 2.0, 3.0])] * 6
    assert calculate_swarm_diversity(same, method) == pytest.approx(0.0, abs=1e-9)
