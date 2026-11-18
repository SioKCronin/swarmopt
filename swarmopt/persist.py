"""Durable run logs and resumable checkpoints for SwarmOpt.

Storage goes through the Rust ``swarmopt-persist`` extension, which handles
power loss, bit flips, storage budgets and replication (see
``rust/persist/README.md``). This module adds:

* a strict, self-describing JSON encoding for payloads, with NumPy arrays
  stored as exact little-endian bytes. It never uses pickle, so loading a
  checkpoint or log can't execute code.
* :class:`RunLog`, for appending JSON events with a sequence number.
* :class:`SwarmCheckpoint`, used by ``Swarm.optimize(checkpoint=...)`` to
  save state every N epochs and resume a run so that it finishes exactly as
  if it had never been interrupted.

Install the extension with::

    pip install maturin
    cd rust/persist && maturin build --release && pip install target/wheels/*.whl
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import swarmopt_persist as _store
except ImportError:  # pragma: no cover - exercised only without the extension
    _store = None

__all__ = [
    "available", "encode", "decode", "RunLog", "read_run_log",
    "SwarmCheckpoint", "CheckpointMismatch", "StateEncodingError",
]

ENCODING_VERSION = 1
CHECKPOINT_SCHEMA = "swarmopt.checkpoint/1"
EVENT_KIND = 1  # record kind used for JSON events in a RunLog

# Only plain numeric dtypes can come back out of a payload: an "object"
# array is exactly the kind of thing that turns deserialisation into code
# execution elsewhere.
_ALLOWED_DTYPES = {
    "bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
    "uint64", "float16", "float32", "float64", "complex64", "complex128",
}
_MAX_ARRAY_BYTES = 64 * 1024 * 1024


class StateEncodingError(TypeError):
    """A value can't be represented in the strict payload encoding."""


class CheckpointMismatch(ValueError):
    """The checkpoint on disk belongs to a differently configured run."""


def available() -> bool:
    """True if the Rust storage extension is installed."""
    return _store is not None


def _require_store():
    if _store is None:
        raise ImportError(
            "swarmopt.persist needs the swarmopt-persist extension: "
            "cd rust/persist && maturin build --release && "
            "pip install target/wheels/*.whl"
        )
    return _store


# ----------------------------------------------------------------- encoding

def _enc(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.ndarray):
        dt = value.dtype.name
        if dt not in _ALLOWED_DTYPES:
            raise StateEncodingError(f"{path}: arrays of dtype {dt} can't be stored")
        raw = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<")).tobytes()
        return {"__nd__": [dt, list(value.shape), base64.b64encode(raw).decode("ascii")]}
    if isinstance(value, np.generic):
        dt = value.dtype.name
        if dt not in _ALLOWED_DTYPES:
            raise StateEncodingError(f"{path}: scalars of dtype {dt} can't be stored")
        raw = np.asarray(value, dtype=value.dtype.newbyteorder("<")).tobytes()
        return {"__ns__": [dt, base64.b64encode(raw).decode("ascii")]}
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"__f__": "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
    if isinstance(value, (list, tuple)):
        items = [_enc(v, f"{path}[{i}]") for i, v in enumerate(value)]
        return {"__tuple__": items} if isinstance(value, tuple) else items
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise StateEncodingError(f"{path}: dict keys must be str, got {type(k).__name__}")
            if k.startswith("__") and k.endswith("__"):
                raise StateEncodingError(f"{path}: key {k!r} is reserved")
            out[k] = _enc(v, f"{path}.{k}")
        return out
    raise StateEncodingError(f"{path}: can't store a {type(value).__name__}")


def _dec(value: Any) -> Any:
    if isinstance(value, list):
        return [_dec(v) for v in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        (tag, body), = value.items()
        if tag == "__nd__":
            dt, shape, b64 = body
            if dt not in _ALLOWED_DTYPES:
                raise ValueError(f"refusing to load array of dtype {dt!r}")
            raw = base64.b64decode(b64, validate=True)
            dtype = np.dtype(dt).newbyteorder("<")
            shape = tuple(int(s) for s in shape)
            if any(s < 0 for s in shape) or len(raw) > _MAX_ARRAY_BYTES:
                raise ValueError("array header is out of bounds")
            if int(np.prod(shape, dtype=np.int64)) * dtype.itemsize != len(raw):
                raise ValueError("array shape doesn't match its data")
            return np.frombuffer(raw, dtype=dtype).reshape(shape).astype(np.dtype(dt), copy=True)
        if tag == "__ns__":
            dt, b64 = body
            if dt not in _ALLOWED_DTYPES:
                raise ValueError(f"refusing to load scalar of dtype {dt!r}")
            raw = base64.b64decode(b64, validate=True)
            dtype = np.dtype(dt).newbyteorder("<")
            if len(raw) != dtype.itemsize:
                raise ValueError("scalar size doesn't match its dtype")
            return np.frombuffer(raw, dtype=dtype).astype(np.dtype(dt))[0]
        if tag == "__f__":
            if body not in ("nan", "inf", "-inf"):
                raise ValueError(f"bad float tag {body!r}")
            return float(body)
        if tag == "__tuple__":
            return tuple(_dec(v) for v in body)
    return {k: _dec(v) for k, v in value.items()}


def _reject_constant(name: str):
    raise ValueError(f"non-standard JSON constant {name}")


def encode(obj: Any) -> bytes:
    """Encode ``obj`` as canonical strict JSON bytes (no NaN tokens, sorted keys)."""
    doc = {"v": ENCODING_VERSION, "data": _enc(obj, "$")}
    return json.dumps(doc, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode(payload: bytes) -> Any:
    """Inverse of :func:`encode`. Raises ``ValueError`` on anything unexpected."""
    doc = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant)
    if not isinstance(doc, dict) or doc.get("v") != ENCODING_VERSION or "data" not in doc:
        raise ValueError("not a swarmopt payload, or an unsupported encoding version")
    return _dec(doc["data"])


# ------------------------------------------------------------------ run log

class RunLog:
    """Append-only event log on the Rust store.

    ``stream_id`` should identify the vehicle or mission, so logs from two
    vehicles can't be mixed by accident. Use ``replicas >= 2`` on hardware
    where storage can be damaged.
    """

    def __init__(self, path, *, stream_id: int = 0, replicas: int = 2,
                 sync="always", on_full: str = "reject",
                 max_total_bytes: int = 256 * 1024 * 1024,
                 segment_bytes: int = 4 * 1024 * 1024,
                 max_record_bytes: int = 1024 * 1024):
        st = _require_store()
        self._w = st.LogWriter(
            str(path), stream_id=stream_id, replicas=replicas, sync=sync,
            on_full=on_full, max_total_bytes=max_total_bytes,
            segment_bytes=segment_bytes, max_record_bytes=max_record_bytes,
        )

    def append(self, event: Dict[str, Any]) -> int:
        """Append one event (a dict) and return its sequence number."""
        return self._w.append(EVENT_KIND, encode(event))

    def sync(self) -> None:
        self._w.sync()

    def prune_through(self, seq: int) -> int:
        """Drop segments holding only events ``<= seq`` (e.g. after downlink)."""
        return self._w.prune_through(seq)

    @property
    def next_seq(self) -> int:
        return self._w.next_seq

    @property
    def recovery(self) -> Dict[str, int]:
        return self._w.recovery

    def close(self) -> None:
        self._w.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def read_run_log(path, *, stream_id: Optional[int] = None) -> Tuple[List[Tuple[int, Any]], Dict[str, int]]:
    """Read ``(seq, event)`` pairs from a run log, plus damage/repair stats.

    Events whose payload fails to decode are left out and counted in
    ``stats["undecodable"]``; with CRC checks in front this should only
    happen if a different program wrote into the log.
    """
    st = _require_store()
    records, stats = st.read_log(str(path), stream_id=stream_id)
    out, bad = [], 0
    for seq, kind, payload in records:
        if kind != EVENT_KIND:
            continue
        try:
            out.append((seq, decode(payload)))
        except (ValueError, UnicodeDecodeError):
            bad += 1
    stats["undecodable"] = bad
    return out, stats


# --------------------------------------------------------------- checkpoints

class SwarmCheckpoint:
    """Checkpoint slot pair for one optimisation run.

    Pass it to ``Swarm.optimize(checkpoint=..., checkpoint_every=N)``. If a
    checkpoint already exists for the same configuration, the run resumes
    from it; if one exists for a *different* configuration,
    :class:`CheckpointMismatch` is raised rather than silently mixing runs.
    """

    def __init__(self, path, *, max_bytes: int = 16 * 1024 * 1024, log: Optional[RunLog] = None):
        st = _require_store()
        self._store = st.CheckpointStore(str(path), max_bytes=max_bytes)
        self.log = log
        self.last_generation: Optional[int] = None
        self.fell_back = False

    def save_state(self, state: Dict[str, Any]) -> int:
        gen = self._store.save(encode(state))
        self.last_generation = gen
        return gen

    def load_state(self) -> Optional[Dict[str, Any]]:
        loaded = self._store.load()
        if loaded is None:
            return None
        self.last_generation = loaded["generation"]
        self.fell_back = bool(loaded["fell_back"])
        state = decode(bytes(loaded["payload"]))
        if not isinstance(state, dict) or state.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint payload is not a swarm checkpoint")
        return state

    def close(self) -> None:
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def fingerprint(config: Dict[str, Any]) -> str:
    """Stable hash of a run's configuration."""
    return hashlib.sha256(encode(config)).hexdigest()


def iter_events(records) -> Iterator[Dict[str, Any]]:
    for _, event in records:
        yield event
