"""Recorded seeded results for every algorithm.

``test_reproducibility.py`` checks that a seed repeats on one machine. This
checks the result against a value recorded on Linux, so CI on macOS and
Windows catches platform drift and any change that silently alters a seeded
run shows up in review.

After an intended change, regenerate with::

    SWARMOPT_REGEN_GOLDEN=1 python -m pytest tests/unit/test_golden.py

and commit ``golden_results.json`` with a note on why the results moved.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

import swarmopt
from swarmopt import Swarm
from swarmopt.functions import rastrigin

GOLDEN = Path(__file__).with_name("golden_results.json")
ALGOS = sorted(swarmopt.algorithms())
REGEN = os.environ.get("SWARMOPT_REGEN_GOLDEN") == "1"


def _run(algo):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        s = Swarm(8, 3, 1.5, 1.5, 0.7, 5, rastrigin, algo=algo,
                  velocity_clamp=(-5.12, 5.12), seed=0)
    s.optimize()
    return {"best_cost": float(s.best_cost),
            "best_pos": [float(x) for x in np.asarray(s.best_pos)]}


def _recorded():
    if not GOLDEN.exists():
        return {}
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="module", autouse=True)
def _write_after_regen():
    results = {}
    yield results
    if REGEN:
        GOLDEN.write_text(json.dumps(dict(sorted(results.items())), indent=1) + "\n")


def test_every_algorithm_has_a_recording():
    if REGEN:
        pytest.skip("regenerating")
    assert sorted(_recorded()) == ALGOS


@pytest.mark.parametrize("algo", ALGOS)
def test_seeded_result_matches_the_recording(algo, _write_after_regen):
    result = _run(algo)
    if REGEN:
        _write_after_regen[algo] = result
        return
    expected = _recorded()[algo]
    # Tight, but not bit-exact: libm may differ in the last bit across
    # platforms. A real change moves these by far more.
    assert result["best_cost"] == pytest.approx(expected["best_cost"], rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(result["best_pos"], expected["best_pos"], rtol=1e-9, atol=1e-12)
