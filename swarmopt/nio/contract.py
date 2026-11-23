"""One step contract for every nio optimizer, and the loop that drives it.

Every optimizer in :data:`swarmopt.nio.registry.OPTIMIZERS` supports:

* ``initialise()``: build the population and the best record.
* ``step()``: one iteration, with no arguments. The driver sets
  ``iteration`` and ``iterations`` first, so schedules that depend on
  progress read them from the optimizer.
* ``best_position`` and ``best_fitness``: the best point seen so far.

:func:`drive` runs that loop. It adds what each optimizer used to do on its
own (calling ``objective.update`` for moving targets) and what none of them
did before: a per-iteration callback, and checkpoint and resume. A resumed
run finishes exactly as the uninterrupted run would, because the saved state
includes the optimizer's random generator.

State is stored as strict JSON through :mod:`swarmopt.persist` (never
pickle). Dataclasses and enums are rebuilt only from classes defined inside
``swarmopt.nio``; a checkpoint can't name anything else.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Tuple

Vector = List[float]

#: Bumped when the layout of a saved nio state changes.
NIO_STATE_VERSION = 1

# Never saved: the objective is the caller's, and the generator is saved
# separately through getstate().
_NOT_STATE = ("objective", "random")


class StateError(TypeError):
    """An optimizer holds a value the checkpoint format can't store."""


def best_of(optimizer) -> Tuple[Vector, float]:
    """Best position and value under the shared contract."""
    return list(optimizer.best_position), float(optimizer.best_fitness)


def drive(
    optimizer,
    iterations: int,
    *,
    checkpoint=None,
    checkpoint_every: Optional[int] = None,
    callback: Optional[Callable[[Any, int], None]] = None,
) -> Tuple[Vector, float]:
    """Run ``optimizer`` for ``iterations`` steps and return its best point.

    checkpoint:
        Any store with ``save_state(dict)`` and ``load_state() -> dict | None``,
        for example :class:`swarmopt.persist.SwarmCheckpoint`. If it already
        holds a state for this same run, the run resumes from it.
    checkpoint_every:
        Iterations between saves (default 10). A final save is always made.
    callback:
        Called as ``callback(optimizer, iteration)`` after every step.
    """
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if checkpoint is None and checkpoint_every is not None:
        raise ValueError("checkpoint_every needs a checkpoint store")
    if checkpoint is not None:
        checkpoint_every = 10 if checkpoint_every is None else int(checkpoint_every)
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least 1")

    start = 0
    saved = checkpoint.load_state() if checkpoint is not None else None
    if saved is not None:
        start = restore(optimizer, saved, iterations)
    else:
        optimizer.iterations = iterations
        optimizer.initialise()

    objective = optimizer.objective
    for iteration in range(start, iterations):
        optimizer.iteration = iteration
        optimizer.iterations = iterations
        if hasattr(objective, "update"):
            objective.update(iteration)
        optimizer.step()
        if callback is not None:
            callback(optimizer, iteration)
        done = iteration + 1
        if checkpoint is not None and (done % checkpoint_every == 0 or done == iterations):
            checkpoint.save_state(snapshot(optimizer, iterations, next_iteration=done))

    position, value = best_of(optimizer)
    if not position:
        raise RuntimeError("optimizer did not record a best solution")
    return position, value


# --------------------------------------------------------------- snapshots

def run_config(optimizer, iterations: int) -> Dict[str, Any]:
    """What identifies a run. A checkpoint only resumes the same run."""
    from .. import __version__

    func = optimizer.objective
    named = func if hasattr(func, "__qualname__") else type(func)
    return {
        "optimizer": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "bounds": [list(b) for b in optimizer.bounds],
        "population_size": int(optimizer.population_size),
        "iterations": int(iterations),
        "objective": f"{getattr(named, '__module__', '?')}.{named.__qualname__}",
        "swarmopt_version": __version__,
    }


def snapshot(optimizer, iterations: int, *, next_iteration: int) -> Dict[str, Any]:
    """Everything needed to continue ``optimizer`` from ``next_iteration``."""
    from ..persist import CHECKPOINT_SCHEMA, fingerprint

    config = run_config(optimizer, iterations)
    state = {}
    for key, value in vars(optimizer).items():
        if key in _NOT_STATE:
            continue
        if callable(value) and not dataclasses.is_dataclass(value):
            continue
        state[key] = _to_plain(value, key)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "kind": "nio",
        "nio_state_version": NIO_STATE_VERSION,
        "fingerprint": fingerprint(config),
        "config": config,
        "next_iteration": int(next_iteration),
        "rng": _to_plain(optimizer.random.getstate(), "rng"),
        "state": state,
    }


def restore(optimizer, saved: Dict[str, Any], iterations: int) -> int:
    """Load a :func:`snapshot` into ``optimizer`` and return the next iteration."""
    from ..persist import CheckpointMismatch, fingerprint

    if saved.get("kind") != "nio" or saved.get("nio_state_version") != NIO_STATE_VERSION:
        raise CheckpointMismatch("checkpoint was not written by this version of swarmopt.nio")
    if saved.get("fingerprint") != fingerprint(run_config(optimizer, iterations)):
        raise CheckpointMismatch(
            "checkpoint was written by a run with a different configuration; "
            "use a fresh checkpoint directory for this run")
    next_iteration = int(saved["next_iteration"])
    if not 0 <= next_iteration <= iterations:
        raise CheckpointMismatch("checkpoint iteration is outside this run")
    optimizer.random.setstate(_rng_state(_from_plain(saved["rng"])))
    for key, value in saved["state"].items():
        setattr(optimizer, key, _from_plain(value))
    return next_iteration


def _rng_state(value):
    # random.Random.setstate wants (version, tuple_of_ints, gauss_next).
    version, internal, gauss_next = value
    return int(version), tuple(int(x) for x in internal), gauss_next


# ------------------------------------------------- plain-value conversion

_DC = "_nio_dataclass"
_ENUM = "_nio_enum"
_ENUM_DICT = "_nio_enum_dict"


def _to_plain(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, enum.Enum):
        return {_ENUM: _class_path(type(value), path), "name": value.name}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {f.name: _to_plain(getattr(value, f.name), f"{path}.{f.name}")
                  for f in dataclasses.fields(value)}
        return {_DC: _class_path(type(value), path), "fields": fields}
    if isinstance(value, tuple):
        return tuple(_to_plain(v, f"{path}[{i}]") for i, v in enumerate(value))
    if isinstance(value, list):
        return [_to_plain(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict) and value and all(isinstance(k, enum.Enum) for k in value):
        return {_ENUM_DICT: [[_to_plain(k, path), _to_plain(v, f"{path}.{k.name}")]
                             for k, v in value.items()]}
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise StateError(f"{path}: dict keys must be str, got {type(k).__name__}")
            out[k] = _to_plain(v, f"{path}.{k}")
        return out
    raise StateError(f"{path}: can't store a {type(value).__name__}")


def _from_plain(value: Any) -> Any:
    if isinstance(value, list):
        return [_from_plain(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_from_plain(v) for v in value)
    if not isinstance(value, dict):
        return value
    if _ENUM_DICT in value:
        return {_from_plain(k): _from_plain(v) for k, v in value[_ENUM_DICT]}
    if _ENUM in value:
        cls = _allowed_classes()[value[_ENUM]]
        return cls[value["name"]]
    if _DC in value:
        cls = _allowed_classes()[value[_DC]]
        kwargs = {k: _from_plain(v) for k, v in value["fields"].items()}
        init = {f.name for f in dataclasses.fields(cls) if f.init}
        obj = cls(**{k: v for k, v in kwargs.items() if k in init})
        for k, v in kwargs.items():
            if k not in init:
                setattr(obj, k, v)
        return obj
    return {k: _from_plain(v) for k, v in value.items()}


def _class_path(cls: type, path: str) -> str:
    name = f"{cls.__module__}.{cls.__qualname__}"
    if name not in _allowed_classes():
        raise StateError(f"{path}: {name} is not a swarmopt.nio class")
    return name


_ALLOWED: Optional[Dict[str, type]] = None


def _allowed_classes() -> Dict[str, type]:
    """Dataclasses and enums defined in swarmopt.nio: the only ones a state may name."""
    global _ALLOWED
    if _ALLOWED is None:
        import swarmopt.nio as package

        found: Dict[str, type] = {}
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            if info.name.endswith(("__main__", "visualize_ca")):
                continue
            module = importlib.import_module(info.name)
            for obj in vars(module).values():
                if not isinstance(obj, type) or obj.__module__ != module.__name__:
                    continue
                if dataclasses.is_dataclass(obj) or issubclass(obj, enum.Enum):
                    found[f"{obj.__module__}.{obj.__qualname__}"] = obj
        _ALLOWED = found
    return _ALLOWED
