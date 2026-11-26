"""Turn the integration scripts' returned results into assertions.

These files began as scripts that print results and return them. Under
pytest, every value a test returns is now checked: a returned ``False``
fails, every number in it must be finite, and every ``Swarm`` in it must
hold a finite cost, a position inside the search box, and, for
single-objective runs, a cost that matches its position. A test that
returns nothing must make its own asserts.
"""

import numbers

import numpy as np
import pytest

from swarmopt import Swarm


# Numbers under these keys may legitimately be inf or NaN (a percentage of
# a zero baseline, or a boundary point's crowding distance).
UNCHECKED_KEYS = {"improvement", "crowding_distance", "crowding_distances"}


def _swarms_and_costs(value, depth=0, key=None):
    if depth > 4 or key in UNCHECKED_KEYS:
        return
    if isinstance(value, Swarm):
        yield "swarm", value
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        yield "cost", value
    elif isinstance(value, dict):
        for k, item in value.items():
            yield from _swarms_and_costs(item, depth + 1, k)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _swarms_and_costs(item, depth + 1, key)


def check_swarm(s):
    if s.multiobjective:
        assert s.best_pos is not None, "multiobjective run produced no Pareto front"
        return
    assert np.isfinite(s.best_cost), f"{s.algo}: best_cost is {s.best_cost}"
    pos = np.asarray(s.best_pos, dtype=float)
    assert pos.shape == (s.dims,)
    assert np.all(pos >= s.val_min) and np.all(pos <= s.val_max), \
        f"{s.algo}: best_pos {pos} left the box [{s.val_min}, {s.val_max}]"
    # Absolute slack scales with the coordinates: re-projecting a best point
    # that sits on a respect boundary moves it by a few ulps.
    slack = 1e-12 * max(1.0, float(np.max(np.abs(pos))))
    assert s.best_cost == pytest.approx(float(s._cost(pos)), rel=1e-9, abs=slack), \
        f"{s.algo}: best_cost does not match best_pos"


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    args = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    result = pyfuncitem.obj(**args)
    # Several scripts catch their own errors and return False.
    assert result is not False, "test reported failure by returning False"
    found = list(_swarms_and_costs(result))
    for kind, value in found:
        if kind == "swarm":
            check_swarm(value)
        else:
            assert np.isfinite(value), f"result cost is {value}"
    pyfuncitem.user_properties.append(("checked_results", len(found)))
    return True
