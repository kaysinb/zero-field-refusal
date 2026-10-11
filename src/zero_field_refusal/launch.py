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
entry). Every run's config is loaded and checked against the protocol before anything starts.

Existing run directories with the run's config hash in ``RUNS_DIR/<experiment>/`` decide what
happens (``assess``):
- a *qualifying* completed run — its ``tracking.gate`` as the campaign's config says, a recorded
  commit and a clean tree when ``tracking.require_clean_git`` is set, a ``summary.json`` with a
  finite K (positive for training runs) and its responses file — makes the run done: skipped;
- a ``diverged`` run (training stopped on a non-finite loss) is an outcome: never rerun;
- every other directory (``failed``, still ``running`` after a kill, unreadable) is a technical
  attempt; at most ``MAX_ATTEMPTS`` attempts are made in all, across invocations;
- a completed run that does not qualify stops the whole campaign before anything starts: running
  it again would put a second run of the seed into the analysis.

Each attempt gets its own ``RUN_ID`` (never the launcher's), so its directory is known and its
outcome read from its ``meta.json``. The launch log ``RUNS_DIR/launch/<campaign name>.jsonl`` gets
one record per attempt start and end; each attempt's output goes to
``RUNS_DIR/launch/<campaign name>/``.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .config import config_hash
from .paths import REPO_ROOT
from .protocol import load_run_config

ENTRIES = {"train": "scripts/train.py", "theta0": "scripts/evaluate_theta0.py"}
CELL_KEYS = {"name", "config", "entry", "overrides", "seeds"}
MAX_ATTEMPTS = 3  # the first attempt and two reruns after technical failures


class CampaignError(ValueError):
    """The campaign cannot start as it stands."""


@dataclass(frozen=True)
class Run:
    cell: str
    entry: str
    config: str
    overrides: tuple[str, ...]  # the cell's overrides, then seed=N (and run_type for θ₀)
    seed: int
    experiment: str
    config_hash: str
    gate: int | None = None
    require_clean_git: bool = False
    prompt_set: str = ""

    @property
    def label(self) -> str:
        return f"{self.cell}/s{self.seed}"

    def command(self) -> list[str]:
        script = ENTRIES[self.entry]
        args = [o for o in self.overrides if o != "run_type=theta0"]  # the script adds it
        return [sys.executable, str(REPO_ROOT / script), self.config, *args]


@dataclass
class Assessment:
    done: Path | None = None  # a qualifying completed run
    diverged: Path | None = None
    attempts: int = 0  # technical attempts so far
    rejected: dict[str, str] = field(default_factory=dict)  # completed runs that do not qualify


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
    hashes = [(run.experiment, run.config_hash) for run in runs]
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{path}: two runs of the campaign have the same config")
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
                        config_hash(cfg), cfg.tracking.get("gate"),
                        bool(cfg.tracking.get("require_clean_git")),
                        str(cfg.eval.harmful.prompt_set)))  # fmt: skip
    return runs


def assess(run: Run, runs_root: Path) -> Assessment:
    """What the existing run directories with this run's config hash say about it."""
    result = Assessment()
    for run_dir in sorted(p for p in (Path(runs_root) / run.experiment).glob("*") if p.is_dir()):
        meta = _read_json(run_dir / "meta.json")
        if meta is None or meta.get("config_hash") != run.config_hash:
            if meta is None and run.config_hash[:8] in run_dir.name:
                result.attempts += 1  # killed before its metadata was written
            continue
        status = meta.get("status")
        if status == "completed":
            problem = _disqualification(run, run_dir, meta)
            if problem is None:
                result.done = result.done or run_dir
            else:
                result.rejected[run_dir.name] = problem
        elif status == "diverged":
            result.diverged = result.diverged or run_dir
        else:
            result.attempts += 1
    return result


def _disqualification(run: Run, run_dir: Path, meta: Mapping[str, Any]) -> str | None:
    """Why a completed run cannot stand for this campaign run (None: it can)."""
    try:
        cfg = OmegaConf.load(run_dir / "config.yaml")
    except (OSError, ValueError):
        return "config.yaml is missing or unreadable"
    gate = OmegaConf.select(cfg, "tracking.gate", default=None)
    if gate != run.gate:
        return f"tracking.gate is {gate!r}, the campaign's config says {run.gate!r}"
    git = meta.get("git") or {}
    if run.require_clean_git and (not git.get("commit") or git.get("dirty") is not False):
        return "not run from a clean checkout of a recorded commit"
    summary = _read_json(run_dir / "summary.json")
    k = None if summary is None else summary.get("K_harmful")
    finite = isinstance(k, int | float) and not isinstance(k, bool) and math.isfinite(k)
    if not finite or (run.entry == "train" and k <= 0):
        return f"summary.json lacks a valid K_harmful ({k!r})"
    if not (run_dir / "responses" / f"{run.prompt_set}.jsonl").is_file():
        return f"responses/{run.prompt_set}.jsonl is missing"
    return None


def run_campaign(
    name: str,
    runs: list[Run],
    runs_root: Path,
    *,
    execute: Callable[[Run, str, Path], int] | None = None,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Run the missing runs in order; returns the labels by outcome.

    Outcomes: skipped (a qualifying run exists), done, diverged, exhausted (``MAX_ATTEMPTS``
    technical attempts without success) and, for a dry run, pending. ``execute(run, run_id,
    output)`` runs one attempt under that run id and returns its exit code (default: a
    subprocess). Raises ``CampaignError`` before starting anything if a completed run with a
    campaign run's config does not qualify.
    """
    execute = execute or _subprocess
    assessments = {run.label: assess(run, runs_root) for run in runs}
    rejected = {f"{run.experiment}/{run_dir}": problem for run in runs
                for run_dir, problem in assessments[run.label].rejected.items()
                if assessments[run.label].done is None}  # fmt: skip
    if rejected:
        lines = "\n".join(f"- {where}: {problem}" for where, problem in sorted(rejected.items()))
        raise CampaignError(f"Completed runs that cannot stand for this campaign; move them out "
                            f"of the experiment or fix them first:\n{lines}")  # fmt: skip
    out_dir = Path(runs_root) / "launch" / name
    log = Path(runs_root) / "launch" / f"{name}.jsonl"
    outcomes = ("skipped", "done", "diverged", "exhausted", "pending")
    result: dict[str, list[str]] = {key: [] for key in outcomes}
    for run in runs:
        state = assessments[run.label]
        if state.done is not None:
            result["skipped"].append(run.label)
            continue
        if state.diverged is not None:
            result["diverged"].append(run.label)
            continue
        if dry_run:
            result["pending" if state.attempts < MAX_ATTEMPTS else "exhausted"].append(run.label)
            continue
        attempts = state.attempts
        outcome = "exhausted"
        while attempts < MAX_ATTEMPTS:
            attempts += 1
            outcome = _attempt(run, attempts, runs_root, out_dir, log, execute)
            if outcome != "technical":
                break
        result[outcome if outcome != "technical" else "exhausted"].append(run.label)
    return result


def _attempt(run: Run, attempt: int, runs_root: Path, out_dir: Path, log: Path,
             execute: Callable[[Run, str, Path], int]) -> str:  # fmt: skip
    """One attempt; returns done, diverged or technical."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = _run_id(stamp, run, attempt)
    output = out_dir / f"{run_id}.log"
    _log(log, {"event": "start", "run": run.label, "attempt": attempt, "run_id": run_id,
               "config_hash": run.config_hash, "command": run.command()[1:]})  # fmt: skip
    code = execute(run, run_id, output)
    run_dir = Path(runs_root) / run.experiment / run_id
    meta = _read_json(run_dir / "meta.json")
    status = None if meta is None else meta.get("status")
    if status == "completed":
        problem = _disqualification(run, run_dir, meta)
        outcome = "done" if problem is None else "technical"
    elif status == "diverged":
        outcome, problem = "diverged", None
    else:
        outcome, problem = "technical", f"status {status!r}"
    _log(log, {"event": "end", "run": run.label, "attempt": attempt, "run_id": run_id,
               "returncode": code, "outcome": outcome, "problem": problem})  # fmt: skip
    return outcome


def _run_id(stamp: str, run: Run, attempt: int) -> str:
    """A run id the tracker accepts: time, cell, seed, config hash and attempt."""
    cell = re.sub(r"[^A-Za-z0-9_.-]", "_", run.cell)[:28]
    return f"{stamp}-{cell}-s{run.seed}-{run.config_hash[:8]}-a{attempt}"


def _subprocess(run: Run, run_id: str, output: Path) -> int:
    env = {**os.environ, "RUN_ID": run_id}
    with output.open("w") as f:
        return subprocess.run(run.command(), stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT,
                              env=env, check=False).returncode  # fmt: skip


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _log(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = {"time": datetime.now(UTC).isoformat(timespec="seconds"), **record}
    with path.open("a") as f:
        f.write(json.dumps(stamped) + "\n")
