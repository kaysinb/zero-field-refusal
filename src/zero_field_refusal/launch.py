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

What already exists for a run — run directories in ``RUNS_DIR/<experiment>/`` with its config hash
and attempts recorded in the launch logs ``RUNS_DIR/launch/*.jsonl`` — decides what happens
(``assess``):
- one *qualifying* completed run — its ``tracking.gate`` as the campaign's config says, a recorded
  commit and a clean tree when ``tracking.require_clean_git`` is set, a ``summary.json`` with a
  finite, non-negative K, and its responses file; for a study-2 schedule, its terminal record —
  makes the run done: skipped;
- a ``diverged`` run (a non-finite training loss or K) is an outcome: never rerun;
- a completed run that does not qualify, or two completed runs, stop the whole campaign before
  anything starts: running again would leave two runs of the seed for the analysis;
- every other attempt (a ``failed`` or still ``running`` directory, an unreadable one, or a logged
  start that left no directory) is technical. At most ``MAX_ATTEMPTS`` are made in all, across
  invocations; a *recovery round* (``recovery=REASON``, once per cell, logged with its reason, after
  the technical cause is fixed) allows ``MAX_ATTEMPTS`` more to the cell's runs that used theirs up.
  Diverged runs never enter it.

Each attempt gets its own ``RUN_ID`` (never the launcher's), so its directory is known and its
outcome read from its ``meta.json``. An attempt that completes without qualifying stops the
campaign. The launch log ``RUNS_DIR/launch/<campaign name>.jsonl`` gets one record per attempt
start and end and per recovery round; each attempt's output goes to
``RUNS_DIR/launch/<campaign name>/``.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .analysis.stage_a import TrajectoryError, load_trajectory
from .config import config_hash
from .paths import REPO_ROOT
from .protocol import load_run_config

ENTRIES = {"train": "scripts/train.py", "theta0": "scripts/evaluate_theta0.py"}
CELL_KEYS = {"name", "config", "entry", "overrides", "seeds"}
MAX_ATTEMPTS = 3  # the first attempt and two reruns after technical failures


class CampaignError(ValueError):
    """The campaign cannot start, or go on, as it stands."""


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
    done: Path | None = None  # the qualifying completed run
    diverged: Path | None = None
    technical: set[str] = field(default_factory=set)  # ids of technical attempts so far
    conflicts: dict[str, str] = field(default_factory=dict)  # run dir -> why it cannot stand


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


def launch_records(runs_root: Path) -> list[dict[str, Any]]:
    """Every record of every launch log under ``runs_root/launch/``."""
    records = []
    for log in sorted((Path(runs_root) / "launch").glob("*.jsonl")):
        for line in log.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue  # a line cut short by a crash
            if isinstance(record, dict):
                records.append(record)
    return records


def assess(run: Run, runs_root: Path, records: Sequence[Mapping[str, Any]] = ()) -> Assessment:
    """What the run directories and the logged attempts say about this run."""
    result = Assessment()
    completed = []
    experiment_dir = Path(runs_root) / run.experiment
    for run_dir in sorted(p for p in experiment_dir.glob("*") if p.is_dir()):
        meta = _read_json(run_dir / "meta.json")
        if meta is None:
            if run.config_hash[:8] in run_dir.name:
                result.technical.add(run_dir.name)  # killed before its metadata was written
            continue
        if meta.get("config_hash") != run.config_hash:
            continue
        status = meta.get("status")
        if status == "completed" or _published(run_dir):
            completed.append(run_dir)
        elif status == "diverged":
            result.diverged = result.diverged or run_dir
        else:
            result.technical.add(run_dir.name)
    for run_dir in completed:
        problem = _disqualification(run, run_dir, _read_json(run_dir / "meta.json") or {})
        if problem is not None:
            result.conflicts[run_dir.name] = problem
    qualifying = [d for d in completed if d.name not in result.conflicts]
    if len(qualifying) > 1:
        for run_dir in qualifying:
            result.conflicts[run_dir.name] = "one of several completed runs of this config"
    elif qualifying and not result.conflicts:
        result.done = qualifying[0]
    settled = {d.name for d in completed} | ({result.diverged.name} if result.diverged else set())
    for record in records:  # starts that left no directory, e.g. a crash before the tracker ran
        if (record.get("event") == "start" and record.get("experiment") == run.experiment
                and record.get("config_hash") == run.config_hash
                and record.get("run_id") not in settled):  # fmt: skip
            result.technical.add(record["run_id"])
    return result


def recovered_cells(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """(experiment, cell) pairs that have had their recovery round."""
    return {(r.get("experiment"), r.get("cell")) for r in records if r.get("event") == "recovery"}


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
    if _scheduled_training(cfg):
        # A study-2 run is done once its whole records agree with its published terminal
        # record; a plateau or an instability stop, with or without evaluations, is an outcome.
        try:
            load_trajectory(run_dir)
        except TrajectoryError as exc:
            return f"a scheduled run's records are incomplete: {exc}"
        return None
    summary = _read_json(run_dir / "summary.json")
    k = None if summary is None else summary.get("K_harmful")
    finite = isinstance(k, int | float) and not isinstance(k, bool) and math.isfinite(k)
    if not finite or k < 0:  # zero is legitimate (soft targets, θ₀); fits decide on log K
        return f"summary.json lacks a finite, non-negative K_harmful ({k!r})"
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
    recovery: str | None = None,
) -> dict[str, list[str]]:
    """Run the missing runs in order; returns the labels by outcome.

    Outcomes: skipped (a qualifying run exists), done, diverged, exhausted (no attempt left) and,
    for a dry run, pending. ``execute(run, run_id, output)`` runs one attempt under that run id and
    returns its exit code (default: a subprocess). ``recovery`` names the fixed technical cause and
    opens the recovery round of every cell with an exhausted run that has not had one. Raises
    ``CampaignError`` before starting anything if a run's existing directories conflict, and stops
    at an attempt that completes without qualifying.
    """
    execute = execute or _subprocess
    records = launch_records(runs_root)
    assessments = {run.label: assess(run, runs_root, records) for run in runs}
    conflicts = {f"{run.experiment}/{run_dir}": problem for run in runs
                 for run_dir, problem in assessments[run.label].conflicts.items()}  # fmt: skip
    if conflicts:
        lines = "\n".join(f"- {where}: {problem}" for where, problem in sorted(conflicts.items()))
        raise CampaignError(f"Completed runs that cannot stand for this campaign; move them out "
                            f"of the experiment or fix them first:\n{lines}")  # fmt: skip
    out_dir = Path(runs_root) / "launch" / name
    log = Path(runs_root) / "launch" / f"{name}.jsonl"
    recovered = recovered_cells(records)
    if recovery and not dry_run:
        for experiment, cell in sorted({(r.experiment, r.cell) for r in runs}):
            exhausted = [r for r in runs if (r.experiment, r.cell) == (experiment, cell)
                         and _open(assessments[r.label]) and
                         len(assessments[r.label].technical) >= MAX_ATTEMPTS]  # fmt: skip
            if exhausted and (experiment, cell) not in recovered:
                _log(log, {"event": "recovery", "experiment": experiment, "cell": cell,
                           "reason": recovery, "runs": [r.label for r in exhausted]})  # fmt: skip
                recovered.add((experiment, cell))
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
        limit = MAX_ATTEMPTS * (2 if (run.experiment, run.cell) in recovered else 1)
        attempts = len(state.technical)
        if dry_run:
            result["pending" if attempts < limit else "exhausted"].append(run.label)
            continue
        outcome = "exhausted"
        while attempts < limit:
            attempts += 1
            outcome = _attempt(run, attempts, runs_root, out_dir, log, execute)
            if outcome != "technical":
                break
        result[outcome if outcome != "technical" else "exhausted"].append(run.label)
    return result


def _open(state: Assessment) -> bool:
    return state.done is None and state.diverged is None


def _attempt(run: Run, attempt: int, runs_root: Path, out_dir: Path, log: Path,
             execute: Callable[[Run, str, Path], int]) -> str:  # fmt: skip
    """One attempt; returns done, diverged or technical; stops the campaign on a conflict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = _run_id(stamp, run, attempt)
    output = out_dir / f"{run_id}.log"
    _log(log, {"event": "start", "run": run.label, "attempt": attempt, "run_id": run_id,
               "experiment": run.experiment, "config_hash": run.config_hash,
               "command": run.command()[1:]})  # fmt: skip
    code = execute(run, run_id, output)
    run_dir = Path(runs_root) / run.experiment / run_id
    meta = _read_json(run_dir / "meta.json")
    status = None if meta is None else meta.get("status")
    problem = None
    if status == "completed" or _published(run_dir):
        problem = _disqualification(run, run_dir, meta)
        outcome = "done" if problem is None else "conflict"
    elif status == "diverged":
        outcome = "diverged"
    else:
        outcome, problem = "technical", f"status {status!r}"
    _log(log, {"event": "end", "run": run.label, "attempt": attempt, "run_id": run_id,
               "returncode": code, "outcome": outcome, "problem": problem})  # fmt: skip
    if outcome == "conflict":
        raise CampaignError(f"{run.experiment}/{run_id} completed but cannot stand for "
                            f"{run.label}: {problem}; the campaign stops")  # fmt: skip
    return outcome


def _scheduled_training(cfg: Any) -> bool:
    """A training run under a study-2 schedule; a θ₀ evaluation of the same config is not."""
    return (
        OmegaConf.select(cfg, "schedule", default=None) is not None
        and OmegaConf.select(cfg, "run_type", default="train") == "train"
    )


def _published(run_dir: Path) -> bool:
    """A scheduled training run whose terminal outcome was published and whose records are
    whole, whatever its status says: a failure of logging or finalization after the outcome must
    not rerun an observed stop."""
    if not (run_dir / "terminal.json").is_file():
        return False
    try:
        if not _scheduled_training(OmegaConf.load(run_dir / "config.yaml")):
            return False
    except (OSError, ValueError):
        return False
    try:
        load_trajectory(run_dir)
    except TrajectoryError:
        return False
    return True


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
