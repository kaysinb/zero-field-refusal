"""Run the missing runs of a campaign one after another, each in a fresh process.

A campaign file (YAML) lists cells::

    name: stage1-1.5b                    # names the launch log
    cells:
      - name: adamw-5e-5
        config: experiments/stage1_main_1.5b
        entry: train                     # train (scripts/train.py) | theta0 (evaluate_theta0.py)
        overrides: [optimizer.lr=5.0e-5]
        seeds: [0, 1, 2, 3, 4]

A run is one cell at one seed: ``CONFIG *overrides seed=N`` (and ``run_type=theta0`` for the θ₀
entry). Every run's config is loaded and checked against the protocol before anything starts, so
a campaign with a wrong cell starts nothing. A run is skipped when ``RUNS_DIR/<experiment>/``
already holds a completed run (``meta.json`` status ``completed``) with the same config hash;
failed runs do not count. Runs execute in order, each in its own process; a failed run is logged
and the next one starts. The launch log ``RUNS_DIR/launch/<campaign name>.jsonl`` gets one record
per run start and end, and each run's output goes to ``RUNS_DIR/launch/<campaign name>/``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .config import config_hash
from .paths import REPO_ROOT
from .protocol import load_run_config

ENTRIES = {"train": "scripts/train.py", "theta0": "scripts/evaluate_theta0.py"}
CELL_KEYS = {"name", "config", "entry", "overrides", "seeds"}


@dataclass(frozen=True)
class Run:
    cell: str
    entry: str
    config: str
    overrides: tuple[str, ...]  # the cell's overrides, then seed=N (and run_type for θ₀)
    seed: int
    experiment: str
    config_hash: str

    @property
    def label(self) -> str:
        return f"{self.cell}/s{self.seed}"

    def command(self) -> list[str]:
        script = ENTRIES[self.entry]
        args = [o for o in self.overrides if o != "run_type=theta0"]  # the script adds it
        return [sys.executable, str(REPO_ROOT / script), self.config, *args]


def load_campaign(path: str | Path) -> tuple[str, list[Run]]:
    """The campaign's name and its runs in order, every config loaded and checked."""
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        raise ValueError(f"{path}: a campaign needs a name and a list of cells")
    cells = data.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{path}: a campaign needs a non-empty list of cells")
    runs, names = [], set()
    for cell in cells:
        runs += _cell_runs(cell, names)
    labels = [run.label for run in runs]
    hashes = [(run.experiment, run.config_hash) for run in runs]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{path}: two runs of the campaign have the same config")
    assert len(set(labels)) == len(labels)
    return data["name"], runs


def _cell_runs(cell: Any, names: set[str]) -> list[Run]:
    if not isinstance(cell, dict) or not {"name", "config", "seeds"} <= set(cell):
        raise ValueError(f"A cell needs name, config and seeds: {cell!r}")
    unknown = set(cell) - CELL_KEYS
    if unknown:
        raise ValueError(f"Cell {cell['name']!r} has unknown keys {sorted(unknown)}")
    name, entry = cell["name"], cell.get("entry", "train")
    if name in names:
        raise ValueError(f"Two cells are named {name!r}")
    names.add(name)
    if entry not in ENTRIES:
        raise ValueError(f"Cell {name!r}: entry must be one of {sorted(ENTRIES)}")
    seeds, overrides = cell["seeds"], cell.get("overrides", [])
    if not (isinstance(seeds, list) and seeds and all(type(s) is int for s in seeds)):
        raise ValueError(f"Cell {name!r}: seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Cell {name!r} lists a seed twice")
    if any(o.split("=", 1)[0] in ("seed", "run_type") for o in overrides):
        raise ValueError(f"Cell {name!r}: seed and run_type come from seeds and entry")
    runs = []
    for seed in seeds:
        args = (*overrides, f"seed={seed}", *(["run_type=theta0"] if entry == "theta0" else []))
        cfg = load_run_config(cell["config"], args)
        runs.append(Run(name, entry, cell["config"], tuple(args), seed, str(cfg.experiment),
                        config_hash(cfg)))  # fmt: skip
    return runs


def completed_hashes(runs_root: Path, experiment: str) -> set[str]:
    """Config hashes of the completed runs in ``runs_root/<experiment>/``."""
    done = set()
    for meta in sorted((Path(runs_root) / experiment).glob("*/meta.json")):
        try:
            data = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # unreadable: not a completed run
        if isinstance(data, dict) and data.get("status") == "completed":
            done.add(data.get("config_hash"))
    return done


def run_campaign(
    name: str,
    runs: Sequence[Run],
    runs_root: Path,
    *,
    execute: Callable[[Run, Path], int] | None = None,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Run the missing runs in order; returns the labels of skipped, done and failed runs.

    ``execute(run, log_file)`` runs one run and returns its exit code (default: a subprocess).
    """
    execute = execute or _subprocess
    out_dir = Path(runs_root) / "launch" / name
    log = Path(runs_root) / "launch" / f"{name}.jsonl"
    result: dict[str, list[str]] = {"skipped": [], "done": [], "failed": [], "pending": []}
    for run in runs:
        if run.config_hash in completed_hashes(runs_root, run.experiment):
            result["skipped"].append(run.label)
            continue
        if dry_run:
            result["pending"].append(run.label)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = out_dir / f"{run.cell}-s{run.seed}-{stamp}.log"
        _log(log, {"event": "start", "run": run.label, "config_hash": run.config_hash,
                   "command": run.command()[1:], "output": output.name})  # fmt: skip
        code = execute(run, output)
        _log(log, {"event": "end", "run": run.label, "returncode": code})
        result["done" if code == 0 else "failed"].append(run.label)
    return result


def _subprocess(run: Run, output: Path) -> int:
    with output.open("w") as f:
        return subprocess.run(run.command(), stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT,
                              check=False).returncode  # fmt: skip


def _log(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {"time": datetime.now(UTC).isoformat(timespec="seconds"), **record}
    with path.open("a") as f:
        f.write(json.dumps(stamped) + "\n")
