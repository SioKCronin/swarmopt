"""Small, dependency-free helpers for writing result files safely.

These cover the plain-text outputs (CSV, JSON, HTML) that the benchmark and
example tooling writes. Durable mission data, checkpoints and logs go through
:mod:`swarmopt.persist`, which is backed by the Rust ``swarmopt-persist`` crate.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Union

PathLike = Union[str, "os.PathLike[str]"]

# A cell starting with one of these is treated as a formula by Excel,
# LibreOffice and Google Sheets (CWE-1236, "CSV injection").
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: PathLike, text: str, *, overwrite: bool = True,
                      encoding: str = "utf-8") -> Path:
    """Write ``text`` to ``path`` so readers see the old file or the new one, never half.

    The data goes to a temporary file in the same directory, is fsynced, and is
    then renamed into place; the directory is fsynced so the rename survives a
    power cut. With ``overwrite=False`` an existing file is left alone and
    ``FileExistsError`` is raised.
    """
    path = Path(path)
    directory = path.parent if str(path.parent) else Path(".")
    fd, tmp_name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp",
                                    dir=directory)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        if overwrite:
            os.replace(tmp, path)
        else:
            # link() fails if the target exists, so two writers can't clobber each other.
            os.link(tmp, path)
            tmp.unlink()
        _fsync_dir(directory)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def unique_path(directory: PathLike, stem: str, suffix: str) -> Path:
    """Return ``directory/stem.suffix``, or ``stem-1.suffix`` etc. if it is taken."""
    directory = Path(directory)
    candidate = directory / f"{stem}{suffix}"
    n = 1
    while candidate.exists():
        candidate = directory / f"{stem}-{n}{suffix}"
        n += 1
    return candidate


def csv_safe(value):
    """Neutralise spreadsheet formulas in a string cell; leave numbers alone."""
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        try:
            float(value)  # "-1.5" is a number, not a formula
            return value
        except ValueError:
            return "'" + value
    return value


def finite_float(text: str) -> float:
    """Parse a float, rejecting NaN and infinities (``ValueError``)."""
    x = float(text)
    if not math.isfinite(x):
        raise ValueError(f"non-finite value: {text!r}")
    return x
