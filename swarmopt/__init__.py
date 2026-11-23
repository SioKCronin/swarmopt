"""SwarmOpt: reproducible particle swarm optimization."""

from .swarm import Swarm

__version__ = "0.3.0"

#: The core PSO family, run by ``Swarm`` itself.
CORE_ALGORITHMS = ("global", "local", "unified", "sa", "cpso", "hhoa")


def algorithms(family=None):
    """Every ``algo`` name ``Swarm`` accepts, mapped to its family.

    ``pso`` and ``ppso`` are not listed separately: ``Swarm`` runs them as the
    core ``global`` PSO and proactive PSO. Pass ``family`` (``swarm``,
    ``evolutionary``, ``ecological`` or ``physical``) to filter.
    """
    from .nio.registry import FAMILIES, FAMILY_NAMES

    if family is not None and family not in FAMILY_NAMES:
        raise ValueError(f"family must be one of {FAMILY_NAMES}, not {family!r}")
    names = dict.fromkeys(CORE_ALGORITHMS, "swarm")
    names.update({k: v for k, v in FAMILIES.items() if k not in ("pso", "ppso")})
    return {k: v for k, v in sorted(names.items()) if family is None or v == family}


__all__ = ["Swarm", "algorithms", "CORE_ALGORITHMS", "__version__"]
