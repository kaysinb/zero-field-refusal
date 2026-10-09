"""One table of every run: ``meta.json``, ``config.yaml`` and ``summary.json`` flattened.

Failed and unfinished runs stay in the table with their status, so an analysis can account for
them explicitly instead of losing them. Nested keys are joined with dots; lists become JSON.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

META_KEYS = ("run_id", "experiment", "name", "config_hash", "status", "started_at",
             "finished_at", "git", "launch", "inputs")  # fmt: skip


def index_runs(root: Path) -> pd.DataFrame:
    """Rows sorted by run directory; ``run_dir`` is relative to ``root``."""
    rows = []
    for meta_path in sorted(root.glob("*/*/meta.json")):
        run_dir = meta_path.parent
        meta = json.loads(meta_path.read_text())
        row: dict[str, Any] = {"run_dir": str(run_dir.relative_to(root))}
        _flatten("meta", {key: meta.get(key) for key in META_KEYS}, row)
        config_path = run_dir / "config.yaml"
        if config_path.is_file():
            _flatten("config", OmegaConf.to_container(OmegaConf.load(config_path)), row)
        summary_path = run_dir / "summary.json"
        row["has_summary"] = summary_path.is_file()
        if row["has_summary"]:
            _flatten("summary", json.loads(summary_path.read_text()), row)
        rows.append(row)
    return pd.DataFrame(rows)


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}", item, out)
    elif isinstance(value, list):
        out[prefix] = json.dumps(value)
    else:
        out[prefix] = value
