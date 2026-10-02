"""Random number generation for SwarmOpt.

Every stochastic step in the library draws from a ``numpy.random.Generator``
instead of NumPy's or Python's global random state. A ``Swarm`` owns one
generator (built from its ``seed``) and activates it while it runs, so helper
modules pick it up through :func:`get_rng` without it being threaded through
every call signature. Using a context variable keeps concurrent swarms in
different threads or async tasks independent of each other.
"""

import contextlib
import contextvars

import numpy as np

_active_rng = contextvars.ContextVar("swarmopt_active_rng", default=None)
_fallback_rng = np.random.default_rng()


def make_rng(seed=None):
    """Return a Generator from an int seed, a SeedSequence, a Generator, or None."""
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def get_rng():
    """Return the generator of the running swarm, or an unseeded fallback."""
    rng = _active_rng.get()
    return _fallback_rng if rng is None else rng


@contextlib.contextmanager
def using_rng(rng):
    """Make ``rng`` the active generator for the duration of the block."""
    token = _active_rng.set(rng)
    try:
        yield rng
    finally:
        _active_rng.reset(token)
