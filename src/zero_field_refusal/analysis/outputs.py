"""Versioned output directories of the analysis builds and their file conventions.

``write_new_directory(out, write)`` has ``write`` fill a temporary directory next to ``out`` and
renames it to ``out`` when complete, so the outputs appear whole or not at all; an existing
directory is never overwritten. Numbers are written unrounded (shortest exact decimal); non-finite
numbers are empty in CSV files and null in JSON. Used by ``analysis.build`` and ``analysis.gate1``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def write_new_directory(out: Path, write: Callable[[Path], None]) -> Path:
    """Create ``out`` with the files that ``write(directory)`` writes; ``FileExistsError`` if it
    exists. On any error nothing is left behind."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists: results are versioned, never overwritten")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=out.parent))
    try:
        write(partial)
        # mkdtemp creates the directory 0700; give it the mode a plain mkdir would.
        partial.chmod(0o777 & ~_umask())
        if out.exists():
            raise FileExistsError(f"{out} appeared while it was written; it is left as it is")
        partial.rename(out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return out


def write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, lineterminator="\n")


def read_csv(path: Path) -> pd.DataFrame:
    """A table written by ``write_csv``, with every float read back exactly."""
    return pd.read_csv(path, float_precision="round_trip")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(plain(data), indent=2, allow_nan=False) + "\n")


def plain(value: Any) -> Any:
    """Plain JSON values: string keys, lists for tuples, null for non-finite numbers."""
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _umask() -> int:
    """The process umask (reading it means setting it; it is restored at once)."""
    mask = os.umask(0o022)
    os.umask(mask)
    return mask
