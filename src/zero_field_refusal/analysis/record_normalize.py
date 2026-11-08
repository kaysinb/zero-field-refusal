"""What differs between two runs of the same code, removed from a run's records before they are
compared: shared by the golden single-process fixture (``tests/test_golden_single_process.py``)
and study 5's engineering check (``scripts/study5_engineering_check.py``).

``normalize`` drops every key containing time, seconds, elapsed, written_at, timestamp or run_id,
and the keys in ``VOLATILE_KEYS`` (absolute paths, machine, package and git identity, the
path-check fingerprint's library and device versions), at every depth of a JSON value. Nothing
else is changed. It reads values only; it opens no file.
"""

from __future__ import annotations

from typing import Any

VOLATILE_PARTS = ("time", "seconds", "elapsed", "written_at", "timestamp", "run_id")
VOLATILE_KEYS = {
    "path", "hardware", "python", "packages", "git", "wandb", "host", "argv", "cwd", "dir",
    "started_at", "finished_at", "created", "version_info",
    "torch", "transformers", "cuda", "cudnn", "device_name",
}  # fmt: skip


def normalize(value: Any, dropped: set[str] | None = None) -> Any:
    """Drop what differs between two runs of the same code: every key containing time,
    seconds, elapsed, written_at, timestamp or run_id, and the keys in ``VOLATILE_KEYS``
    (absolute paths, machine, package and git identity, the path-check fingerprint's library and
    device versions). ``dropped`` collects the names removed."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in VOLATILE_KEYS or any(part in key.lower() for part in VOLATILE_PARTS):
                if dropped is not None:
                    dropped.add(key)
                continue
            out[key] = normalize(item, dropped)
        return out
    if isinstance(value, list):
        return [normalize(item, dropped) for item in value]
    return value
