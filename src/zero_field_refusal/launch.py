"""Run the missing runs of a campaign one after another, each in a fresh process.

A campaign file (YAML) lists cells::

    name: stage1-1.5b                    # names the launch log
    header: {w_max: 0.47}                # optional: logged with every start (``campaign_header``)
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
  finite, non-negative K, and its responses file; for a study-2 schedule, its terminal record (and,
  with ``eval.rayleigh`` enabled, a ``rayleigh.json`` with R or the reason it is null, a failed
  measurement included) — makes the run done: skipped;
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
campaign. A stop request (``StopRequest``: ``SIGTERM`` to the launcher) lets the running attempt
finish and starts no other; the runs left are reported as stopped, and the campaign started again
carries on from there. The launch log ``RUNS_DIR/launch/<campaign name>.jsonl`` gets one record
per attempt start and end, per recovery round and per stop; each attempt's output goes to
``RUNS_DIR/launch/<campaign name>/``. A start record carries the campaign's ``header`` when it
has one (study 2's campaigns record the w_max they were written under, which the replay reads
back: the final review of gate B, I4). An end the launcher writes after the fact, for a start it
finds without one, carries ``"settled": true`` (``run_campaign``, "A start found without its end").
Ends, ``superseded`` and ``held`` records are matched to their starts by run id, and a label's
attempts are ordered by their attempt number, then the start's time (``label_starts``), never by
the journal file that holds them.

**One label at a time** (``only=LABEL``; study 3's numerical qualification, which runs a case's
launcher run, its harness parts and the deletion of its checkpoints before the next case): that
run alone is assessed and attempted; every other run of the campaign is neither read nor started,
so a finished case whose checkpoint was deleted is never assessed again. Its start and end
records carry ``only``, and a recovery round opened under it is that run's own (once per run,
logged with the label), never its cell's. ``max_attempts`` (default ``MAX_ATTEMPTS``) bounds the
attempts of a run, and of a recovery round; the qualification passes 1, so that a failed launcher
run is attempted again only after a recovery round with its recorded reason. A start record
carries ``max_attempts`` when it is not the default.

**Superseded attempts** (``supersede``). A ``superseded`` record in the journal (written for study
3's qualification by a case restart or an engineering cycle's supersession) names an attempt
that no longer counts for its label: its directory and records stay, it is a used attempt (a
technical one), and it is neither done, diverged nor a conflict; the run is then attempted again
through its recovery round. A ``held`` record (``hold``: an outcome a study-3 engineering cycle
could not resolve) makes the run a conflict: nothing of the campaign starts, its recovery round
included, in a dry run too. The hold applies to the run's label (its experiment and
``<cell>/s<seed>``) whatever the config hash it was written with, so a campaign whose cell now
has another config hash is refused it as well; nothing here lifts a hold. A checkpoint run
(``tracking.endpoint_checkpoint``) is done only if its recorded numerics setup is what the
campaign's config asks for (``expected_numerics``: determinism, TF32 and the matmul precision,
the memory limit; read without touching CUDA).

**The journal** (``_log``). A record is appended as one line: the log is opened for appending
(created with mode 0o644), the whole line goes out in one ``os.write`` (a short write raises) and
is fsynced; creating a log fsyncs its folder too. A crash in the middle of an append can still
leave a *torn line*: the bytes after the log's last newline. The next append to that log repairs
it first: a newline, then a ``torn`` record naming the torn line's exact bytes (``offset`` of its
first byte, ``length``, ``sha256``) and its ``kind``, then the record. The kind is the event the
torn bytes show through the strict pattern ``"event": "<lower-case word>"`` (``torn_kind``), or
null; nothing else of the torn bytes is copied anywhere. ``torn_lines`` lists every unparseable
line of every log under ``RUNS_DIR/launch/`` (a valid JSON line that is not an object included,
and a torn last line, which has no ``torn`` record yet), and whether a ``torn_resolved`` record
names it (by log, offset and sha256); ``launch_records`` skips them, as before.

While any log holds a torn line that no ``torn_resolved`` record names, ``run_campaign`` starts
nothing, settles nothing and opens no recovery round, in a dry run too: it raises
``TornJournalError``, and scripts/launch.py prints each such line (log, offset, length, kind and
the sha256 its resolution names) and exits ``TORN_JOURNAL`` (12). No caller uses that code: the
launcher's other exits are 0, 1 and argparse's 2, and study 3's qualification driver, which
passes the launcher's exit on, uses 1, 3 and 8 to 11. Every log is read, since a label's held and
superseded lines can live in any of them. What a lost line costs, by its kind, and what restores
it before the resolution:

- ``end``: the start stands without its end. Nothing is rerun; once the line is resolved, the
  launcher settles the start from its run directory, as any start found without its end.
- ``start``: a run directory may stand with no start (study 3's analysis lists it and blocks).
  The launcher counts that directory as any directory of the run's config and never reuses its
  run id: a new attempt's id carries its own time and attempt number, and the tracker never
  writes into an existing run directory.
- ``held``, ``superseded``: their writers run again first (``qualify.py supersede`` with the same
  record, ``qualify.py restart-case`` again for the case), so that a held run is never rerun and
  a superseded one never counts. Each writes only the missing lines: ``hold`` and ``supersede``
  write a line only when no identical one (the same event, label, experiment, run id and reason)
  is in the journal.
- ``recovery``, ``stop``: once the line is resolved, the operator relaunches with the same
  command; a recovery round's reason is recorded again.
- unknown (null): the operator reads the run directories of the campaign's labels and records
  the resolution.

``resolve_torn`` (``scripts/launch.py --resolve-torn LOG --offset N --sha256 HEX --reason TEXT``)
appends ``{"event": "torn_resolved", "log", "offset", "sha256", "reason"}`` to that log through
``_log``. It refuses an offset and sha256 that name no torn line of that log, a line already
resolved and an empty reason. It does not recover the lost record: it records that the operator
restored what the journal needs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from yaml import YAMLError

from .analysis.stage_a import TrajectoryError, load_trajectory
from .config import config_hash
from .paths import REPO_ROOT
from .protocol import load_run_config
from .schedule import INSTABILITY
from .tensor_hash import file_sha256

ENTRIES = {"train": "scripts/train.py", "theta0": "scripts/evaluate_theta0.py"}
CELL_KEYS = {"name", "config", "entry", "overrides", "seeds"}
MAX_ATTEMPTS = 3  # the first attempt and two reruns after technical failures
TORN_JOURNAL = 12  # scripts/launch.py's exit while a torn journal line is unresolved
# The one part of a torn line's bytes that a record copies: its kind (module docstring)
TORN_KIND = re.compile(rb'"event": "([a-z][a-z_]{0,31})"')


class CampaignError(ValueError):
    """The campaign cannot start, or go on, as it stands."""


@dataclass(frozen=True)
class TornLine:
    """An unparseable line of a launch log (``torn_lines``): its log's file name, the offset of
    its first byte, its length in bytes and their sha256, its kind (``torn_kind``), whether a
    ``torn`` record of that log names exactly those bytes and whether a ``torn_resolved`` record
    names it (by log, offset and sha256)."""

    log: str
    offset: int
    length: int
    sha256: str
    kind: str | None
    repaired: bool
    resolved: bool


class TornJournalError(CampaignError):
    """A launch log holds a torn line that no ``torn_resolved`` record names (module docstring,
    "The journal"): the launcher starts nothing, settles nothing and opens no recovery round."""

    def __init__(self, lines: Sequence[TornLine]) -> None:
        self.lines = list(lines)
        listed = "\n".join(f"- {x.log}: offset {x.offset}, length {x.length}, kind "
                           f"{x.kind or 'null'}, sha256 {x.sha256}"
                           for x in self.lines)  # fmt: skip
        super().__init__(
            "The launch journal holds torn lines that no torn_resolved record names; nothing "
            "starts, is settled or opens a recovery round until each is resolved: restore what "
            'its kind lost (docs/operations.md, "Launching runs"), then run scripts/launch.py '
            f"--resolve-torn LOG --offset N --sha256 HEX --reason TEXT.\n{listed}"
        )


class StopRequest:
    """Set by a signal: the campaign starts no further attempt, and the running one, a child
    process the signal does not reach, finishes and is recorded first."""

    def __init__(self) -> None:
        self.requested = False
        self.previous: Any = None

    def __call__(self) -> bool:
        return self.requested

    def _handle(self, signum: int, frame: Any) -> None:
        self.requested = True

    @classmethod
    def on_sigterm(cls) -> StopRequest:
        request = cls()
        request.previous = signal.signal(signal.SIGTERM, request._handle)
        return request


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
    superseded: set[str] = field(default_factory=set)  # ids a supersession record names
    # run dir -> (outcome, problem) as the launcher ends that attempt: done, diverged, conflict
    # (the problem is why it cannot stand) or technical; an attempt without a directory is absent
    outcomes: dict[str, tuple[str, str | None]] = field(default_factory=dict)


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


def campaign_header(path: str | Path) -> dict[str, Any] | None:
    """The campaign's optional ``header``: a mapping of names to plain values (numbers, strings,
    booleans, null) that every start record of the campaign carries; None without one."""
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    header = data.get("header") if isinstance(data, dict) else None
    if header is None:
        return None
    plain = (int, float, str, bool, type(None))
    if not (isinstance(header, dict) and all(isinstance(k, str) and isinstance(v, plain)
                                             for k, v in header.items())):  # fmt: skip
        raise ValueError(f"{path}: a campaign's header maps names to plain values: {header!r}")
    return header


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


def torn_kind(data: bytes) -> str | None:
    """The event a torn line's bytes show through the strict pattern ``TORN_KIND``
    (``"event": "<lower-case word>"``), or None; nothing else of the bytes is read out."""
    match = TORN_KIND.search(data)
    return match[1].decode("ascii") if match else None


def torn_lines(runs_root: Path) -> list[TornLine]:
    """Every unparseable line of every launch log under ``runs_root/launch/``, a valid JSON line
    that is not an object included, in file name and offset order (module docstring, "The
    journal"). A torn last line, with no newline after it, is listed unrepaired: no record
    follows it yet."""
    logs = {log.name: _journal_lines(log)
            for log in sorted((Path(runs_root) / "launch").glob("*.jsonl"))}  # fmt: skip
    resolved = {(r.get("log"), r.get("offset"), r.get("sha256"))
                for lines in logs.values() for _, _, r in lines
                if r is not None and r.get("event") == "torn_resolved"
                and _count(r.get("offset"))}  # fmt: skip
    found = []
    for name, lines in logs.items():
        named = {(r.get("offset"), r.get("length"), r.get("sha256")) for _, _, r in lines
                 if r is not None and r.get("event") == "torn"
                 and _count(r.get("offset")) and _count(r.get("length"))}  # fmt: skip
        for offset, data, record in lines:
            if record is not None:
                continue
            sha = hashlib.sha256(data).hexdigest()
            found.append(TornLine(name, offset, len(data), sha, torn_kind(data),
                                  (offset, len(data), sha) in named,
                                  (name, offset, sha) in resolved))  # fmt: skip
    return found


def resolve_torn(runs_root: Path, log: str, *, offset: int, sha256: str,
                 reason: str) -> dict[str, Any]:  # fmt: skip
    """Record that the operator restored what the journal needs after the torn line of ``log``
    (a file name under ``runs_root/launch/``) at ``offset`` with ``sha256`` lost its record (module
    docstring, "The journal"): a ``torn_resolved`` record appended to that log through ``_log``,
    which is returned. Raises ``ValueError``, writing nothing, for an empty reason, an offset and
    sha256 that name no torn line of that log, and a line already resolved."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A resolution needs its reason: what was restored, and from what")
    if Path(log).name != log or not log.endswith(".jsonl"):
        raise ValueError(f"{log!r}: name the launch log by its file name under "
                         "RUNS_DIR/launch/, as the launcher's refusal prints it")  # fmt: skip
    named = [x for x in torn_lines(runs_root)
             if (x.log, x.offset, x.sha256) == (log, offset, sha256)]  # fmt: skip
    if not named:
        raise ValueError(f"{log}: no torn line at offset {offset} with sha256 {sha256}")
    if named[0].resolved:
        raise ValueError(f"{log}: the torn line at offset {offset} is already resolved")
    return _log(Path(runs_root) / "launch" / log,
                {"event": "torn_resolved", "log": log, "offset": offset, "sha256": sha256,
                 "reason": reason})  # fmt: skip


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _journal_lines(log: Path) -> list[tuple[int, bytes, dict[str, Any] | None]]:
    """Each line of a launch log, split at its newlines: its offset, its bytes and the record it
    holds (None when it is not a JSON object)."""
    data, lines, offset = log.read_bytes(), [], 0
    while offset < len(data):
        end = data.find(b"\n", offset)
        end = len(data) if end < 0 else end
        try:
            record = json.loads(data[offset:end])
        except ValueError:
            record = None
        lines.append((offset, data[offset:end], record if isinstance(record, dict) else None))
        offset = end + 1
    return lines


def superseded_runs(run: Run, records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The run ids of this run that the journal's ``superseded`` records name (``supersede``),
    with their reasons."""
    return {r["run_id"]: str(r.get("reason")) for r in records
            if r.get("event") == "superseded" and r.get("run") == run.label
            and r.get("experiment") == run.experiment and r.get("config_hash") == run.config_hash
            and isinstance(r.get("run_id"), str)}  # fmt: skip


def held_runs(run: Run, records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The run ids of this run's label (its experiment and ``<cell>/s<seed>``) that the
    journal's ``held`` records name (``hold``), with their reasons, whatever their config hash:
    a hold applies to the label, so a campaign whose cell now has another config hash (a fix
    that changed a hashed key) is refused it too."""
    return {r["run_id"]: str(r.get("reason")) for r in records
            if r.get("event") == "held" and r.get("run") == run.label
            and r.get("experiment") == run.experiment
            and isinstance(r.get("run_id"), str)}  # fmt: skip


def attempt_key(record: Mapping[str, Any]) -> tuple[int, str]:
    """How a start record sorts among its label's attempts: by its attempt number, then by its
    time (``_log``'s ISO stamp), never by the journal file that holds it. A start without an
    attempt number sorts first."""
    attempt = record.get("attempt")
    number = attempt if isinstance(attempt, int) and not isinstance(attempt, bool) else -1
    return number, str(record.get("time") or "")


def label_starts(run: Run, records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The start records of the run's attempts (its label, experiment and config hash, with a
    run id), in attempt order (``attempt_key``). ``launch_records`` reads the journals in file
    name order, so a later attempt logged under another campaign's name may come first there."""
    starts = [r for r in records
              if r.get("event") == "start" and r.get("run") == run.label
              and r.get("experiment") == run.experiment
              and r.get("config_hash") == run.config_hash
              and isinstance(r.get("run_id"), str)]  # fmt: skip
    return sorted(starts, key=attempt_key)


def open_start(run: Run, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The start of the run's latest attempt that no ``end``, ``superseded`` or ``held`` record
    names by its run id (``run_campaign``, "A start found without its end"), or None. Attempts
    are ordered by their number, then the start's time (``label_starts``), whatever journal file
    holds each record; a record closes the start whose run id it names, wherever it is."""
    closed = {r.get("run_id") for r in records if r.get("event") in ("end", "superseded", "held")}
    pending = [r for r in label_starts(run, records) if r["run_id"] not in closed]
    return pending[-1] if pending else None


def hold(
    runs_root: Path, campaign: str, run: Run, run_id: str, *, reason: str, record: str
) -> bool:
    """Log that an attempt of ``run`` has an outcome nobody could resolve (module docstring,
    "Superseded attempts"): it is kept as it is, and the run's label is never attempted again,
    under any config hash, its recovery round included, while the record stands. Written only
    when no identical line is in the journal (``_journal_names``); returns whether it was."""
    if _journal_names(runs_root, "held", run, run_id, reason):
        return False
    _log(Path(runs_root) / "launch" / f"{campaign}.jsonl",
         {"event": "held", "run": run.label, "run_id": run_id, "experiment": run.experiment,
          "config_hash": run.config_hash, "reason": reason, "record": record})  # fmt: skip
    return True


def supersede(
    runs_root: Path, campaign: str, run: Run, run_id: str, *, reason: str, record: str
) -> bool:
    """Log that an attempt of ``run`` no longer counts for its label (module docstring,
    "Superseded attempts"): its directory and its records stay, and ``record`` (the path of the
    record that decided it, a case restart or an engineering cycle's supersession) says why.
    Written only when no identical line is in the journal (``_journal_names``); returns whether
    it was."""
    if _journal_names(runs_root, "superseded", run, run_id, reason):
        return False
    _log(Path(runs_root) / "launch" / f"{campaign}.jsonl",
         {"event": "superseded", "run": run.label, "run_id": run_id,
          "experiment": run.experiment, "config_hash": run.config_hash, "reason": reason,
          "record": record})  # fmt: skip
    return True


def _journal_names(runs_root: Path, event: str, run: Run, run_id: str, reason: str) -> bool:
    """Whether any launch log holds an identical ``held`` or ``superseded`` line: the same event,
    label, experiment, run id and reason (module docstring, "The journal": a writer that lost its
    line runs again and writes only what is missing)."""
    return any(r.get("event") == event and r.get("run") == run.label
               and r.get("experiment") == run.experiment and r.get("run_id") == run_id
               and r.get("reason") == reason for r in launch_records(runs_root))  # fmt: skip


def assess(run: Run, runs_root: Path, records: Sequence[Mapping[str, Any]] = ()) -> Assessment:
    """What the run directories and the logged attempts say about this run."""
    result = Assessment()
    completed = []
    named = superseded_runs(run, records)
    result.superseded = set(named)
    experiment_dir = Path(runs_root) / run.experiment
    for run_dir in sorted(p for p in experiment_dir.glob("*") if p.is_dir()):
        meta = _read_json(run_dir / "meta.json")
        if meta is None:
            if run.config_hash[:8] in run_dir.name:
                result.technical.add(run_dir.name)  # killed before its metadata was written
                result.outcomes[run_dir.name] = ("technical", "meta.json is missing or unreadable")
            continue
        if meta.get("config_hash") != run.config_hash:
            continue
        if run_dir.name in result.superseded:  # kept, and a used attempt; it no longer counts
            result.technical.add(run_dir.name)
            result.outcomes[run_dir.name] = (
                "technical", f"named superseded in the journal: {named[run_dir.name]}")  # fmt: skip
            continue
        status = meta.get("status")
        if status == "completed" or _published(run_dir):
            completed.append(run_dir)
        elif status == "diverged":
            result.diverged = result.diverged or run_dir
            result.outcomes[run_dir.name] = ("diverged", None)
        else:
            result.technical.add(run_dir.name)
            result.outcomes[run_dir.name] = ("technical", f"status {status!r}")
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
    for run_id, reason in held_runs(run, records).items():  # never attempted again
        result.conflicts[run_id] = f"held, its outcome unresolved: {reason}"
    result.outcomes |= {run_dir.name: ("done", None) for run_dir in completed}
    result.outcomes |= {name: ("conflict", why) for name, why in result.conflicts.items()}
    settled = {d.name for d in completed} | ({result.diverged.name} if result.diverged else set())
    for record in records:  # starts that left no directory, e.g. a crash before the tracker ran
        if (record.get("event") == "start" and record.get("experiment") == run.experiment
                and record.get("config_hash") == run.config_hash
                and record.get("run_id") not in settled):  # fmt: skip
            result.technical.add(record["run_id"])
    return result


def recovered_cells(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """(experiment, cell) pairs that have had their recovery round (a run's own round, opened
    under ``only``, is not its cell's)."""
    return {(r.get("experiment"), r.get("cell")) for r in records
            if r.get("event") == "recovery" and r.get("only") is None}  # fmt: skip


def recovered_runs(records: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    """(experiment, label) pairs that have had a recovery round of their own (under ``only``)."""
    return {(r.get("experiment"), r.get("only")) for r in records
            if r.get("event") == "recovery" and r.get("only") is not None}  # fmt: skip


def _disqualification(run: Run, run_dir: Path, meta: Mapping[str, Any]) -> str | None:
    """Why a completed run cannot stand for this campaign run (None: it can)."""
    try:
        cfg = OmegaConf.load(run_dir / "config.yaml")
    except (OSError, ValueError, YAMLError):
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
        problem = _checkpoint_problem(cfg, run_dir) or _numerics_problem(run, cfg, meta)
        if problem is not None:
            return problem
        if OmegaConf.select(cfg, "eval.rayleigh.enabled", default=False):
            return _rayleigh_problem(_read_json(run_dir / "rayleigh.json"))
        return None
    summary = _read_json(run_dir / "summary.json")
    k = None if summary is None else summary.get("K_harmful")
    finite = isinstance(k, int | float) and not isinstance(k, bool) and math.isfinite(k)
    if not finite or k < 0:  # zero is legitimate (soft targets, θ₀); fits decide on log K
        return f"summary.json lacks a finite, non-negative K_harmful ({k!r})"
    if not (run_dir / "responses" / f"{run.prompt_set}.jsonl").is_file():
        return f"responses/{run.prompt_set}.jsonl is missing"
    return None


def _checkpoint_problem(cfg: Any, run_dir: Path) -> str | None:
    """A run whose config sets ``tracking.endpoint_checkpoint`` is done only if its summary holds
    the checkpoint record and the file at the recorded path exists with the recorded sha256
    (docs/study3-preregistration.md, "The execution path: harness and bridge"); anything else is
    a conflict, never silently reused.

    The exception is a declared instability stop (``schedule.INSTABILITY`` in the published
    terminal record): that is the run's scientific outcome, not a defect, and a rerun to replace
    it is not allowed. Such a run is done whether its summary records no checkpoint (it stopped
    before the last update, or on a non-finite weight at it) or a checkpoint (it stopped inside
    the step-25 ``after_step``), and the file is not required. The file's existence is not the
    run's success: the terminal record is read separately."""
    if OmegaConf.select(cfg, "tracking.endpoint_checkpoint", default=None) is None:
        return None
    summary = _read_json(run_dir / "summary.json") or {}
    stop = ((_read_json(run_dir / "terminal.json") or {}).get("stop") or {}).get("reason")
    if "endpoint_checkpoint" in summary and stop in INSTABILITY:
        return None
    record = summary.get("endpoint_checkpoint")
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        return "summary.json holds no endpoint-checkpoint record"
    path = Path(record["path"])
    if not path.is_file():
        return f"the endpoint checkpoint {path} is missing"
    if file_sha256(path) != record.get("sha256"):
        return f"the endpoint checkpoint {path} does not match its recorded sha256"
    return None


def expected_numerics(cfg: Any) -> dict[str, Any]:
    """What ``runtime.configure_numerics`` records for ``cfg``, its device-independent part, read
    from the config without touching CUDA: determinism (``execution.deterministic``), TF32 and
    the float32 matmul precision it implies (``model.tf32``, as ``models.set_numerics`` sets
    them) and the memory-limit policy (``execution.memory_limit_gb``). The memory fraction and
    the device index depend on the device: the child process checks them."""
    tf32 = bool(OmegaConf.select(cfg, "model.tf32", default=False))
    limit = OmegaConf.select(cfg, "execution.memory_limit_gb", default=None)
    return {"deterministic": bool(OmegaConf.select(cfg, "execution.deterministic", default=False)),
            "tf32_matmul": tf32, "tf32_cudnn": tf32,
            "matmul_precision": "high" if tf32 else "highest",
            "memory_limit_gb": None if limit is None else float(limit)}  # fmt: skip


def _numerics_problem(run: Run, cfg: Any, meta: Mapping[str, Any]) -> str | None:
    """A checkpoint run (``tracking.endpoint_checkpoint``) is done only if its recorded numerics
    setup (``meta.json``: ``inputs.numerics_setup``) is what the campaign's config asks for
    (``expected_numerics``; ``execution`` is outside the config hash, so the campaign's config is
    read, not the run's); a departure is a conflict, found before the harness would find it."""
    if OmegaConf.select(cfg, "tracking.endpoint_checkpoint", default=None) is None:
        return None
    recorded = (meta.get("inputs") or {}).get("numerics_setup")
    if not isinstance(recorded, Mapping):
        return "meta.json records no numerics setup (inputs.numerics_setup)"
    try:
        wanted = expected_numerics(load_run_config(run.config, run.overrides))
    except Exception as exc:  # no config to compare with: it cannot stand
        return f"the campaign's config cannot be read: {type(exc).__name__}: {exc}"
    departs = {key: {"recorded": recorded.get(key), "config": value}
               for key, value in wanted.items() if recorded.get(key) != value}  # fmt: skip
    if departs:
        return f"its numerics setup departs from the campaign's config: {departs}"
    return None


def run_campaign(
    name: str,
    runs: list[Run],
    runs_root: Path,
    *,
    execute: Callable[[Run, str, Path, Mapping[str, Any]], int] | None = None,
    dry_run: bool = False,
    recovery: str | None = None,
    stop: Callable[[], bool] | None = None,
    header: Mapping[str, Any] | None = None,
    only: str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict[str, list[str]]:
    """Run the missing runs in order; returns the labels by outcome.

    Outcomes: skipped (a qualifying run exists), done, diverged, exhausted (no attempt left),
    stopped (``stop()`` was true before its next attempt; logged once) and, for a dry run,
    pending. ``execute(run, run_id, output, campaign)`` runs one attempt under that run id and
    returns its exit code (default: a subprocess); ``campaign`` is the campaign's name, the run's
    label, the attempt and the header, which the child reads from ``ZFR_CAMPAIGN``.
    ``recovery`` names the fixed technical cause and
    opens the recovery round of every cell with an exhausted run that has not had one.
    ``header`` (``campaign_header``) goes into every start record. ``only`` runs that label alone
    and ``max_attempts`` bounds the attempts (module docstring). Raises ``CampaignError``
    before starting anything, its recovery round included and in a dry run too, if a run's
    existing directories conflict or its label has a ``held`` run in any journal, whatever that
    run's config hash (the conflict names it and the hold's reason), or if ``only`` names no
    run; and stops at an attempt that completes without qualifying.

    **A start found without its end** (the launcher died while its child ran, or after the child
    published and before it judged the run). When an assessed run has a start that no ``end``,
    ``superseded`` or ``held`` record names by its run id (``open_start``: the latest such
    attempt, the attempts ordered by their number and then the start's time, whatever journal
    file holds each record), the launcher writes that end first, before it skips the run,
    raises a conflict or starts a new attempt: the same run id and attempt, ``returncode`` null,
    ``"settled": true`` (judged after the fact, not when the process ended), ``only`` as any end
    carries it, and the outcome ``assess`` gives the attempt's directory
    (``Assessment.outcomes``): ``done``, ``diverged``, ``conflict`` with the problem that stops
    the campaign, or ``technical`` (no directory, a missing or unreadable ``meta.json``, any
    other status). A ``done`` end keeps the launcher's meaning, a valid terminal outcome (a
    declared instability stop included), never a qualification. An end that exists is never
    rewritten. Nothing is written for a run that is not assessed (another label under ``only``,
    a run left out of ``runs``) and nothing in a dry run; the end goes to this campaign's log.
    Attempts are counted from the starts, as before, so the new end changes no count.

    **A torn journal line** (module docstring, "The journal"). Before anything else, while any
    log under ``runs_root/launch/`` holds a torn line that no ``torn_resolved`` record names
    (``torn_lines``), it raises ``TornJournalError`` naming each such line: nothing starts,
    nothing is settled or written, no recovery round opens, in a dry run too.
    """
    unresolved = [line for line in torn_lines(runs_root) if not line.resolved]
    if unresolved:
        raise TornJournalError(unresolved)
    execute = execute or _subprocess
    if not (isinstance(max_attempts, int) and max_attempts >= 1):
        raise CampaignError(f"max_attempts must be a positive integer, not {max_attempts!r}")
    if only is not None:
        runs = [run for run in runs if run.label == only]
        if not runs:
            raise CampaignError(f"The campaign has no run labelled {only!r}")
    records = launch_records(runs_root)
    assessments = {run.label: assess(run, runs_root, records) for run in runs}
    out_dir = Path(runs_root) / "launch" / name
    log = Path(runs_root) / "launch" / f"{name}.jsonl"
    extra = {"only": only} if only is not None else {}
    if max_attempts != MAX_ATTEMPTS:
        extra["max_attempts"] = max_attempts
    if not dry_run:  # before a run is skipped, a conflict raised or an attempt started
        for run in runs:
            _settle(run, assessments[run.label], records, log, extra)
    conflicts = {f"{run.experiment}/{run_dir}": problem for run in runs
                 for run_dir, problem in assessments[run.label].conflicts.items()}  # fmt: skip
    if conflicts:
        lines = "\n".join(f"- {where}: {problem}" for where, problem in sorted(conflicts.items()))
        raise CampaignError(f"Completed runs that cannot stand for this campaign; move them out "
                            f"of the experiment or fix them first:\n{lines}")  # fmt: skip
    recovered = recovered_cells(records)
    own = recovered_runs(records)  # the rounds opened under ``only``

    def used_up(run: Run) -> bool:
        state = assessments[run.label]
        return _open(state) and len(state.technical) >= max_attempts

    if recovery and not dry_run and only is not None:
        run = runs[0]
        if (used_up(run) and (run.experiment, run.cell) not in recovered
                and (run.experiment, run.label) not in own):  # fmt: skip
            _log(log, {"event": "recovery", "experiment": run.experiment, "cell": run.cell,
                       "reason": recovery, "runs": [run.label], "only": only})  # fmt: skip
            own.add((run.experiment, run.label))
    elif recovery and not dry_run:
        for experiment, cell in sorted({(r.experiment, r.cell) for r in runs}):
            exhausted = [r for r in runs if (r.experiment, r.cell) == (experiment, cell)
                         and used_up(r)]  # fmt: skip
            if exhausted and (experiment, cell) not in recovered:
                _log(log, {"event": "recovery", "experiment": experiment, "cell": cell,
                           "reason": recovery, "runs": [r.label for r in exhausted]})  # fmt: skip
                recovered.add((experiment, cell))
    outcomes = ("skipped", "done", "diverged", "exhausted", "stopped", "pending")
    result: dict[str, list[str]] = {key: [] for key in outcomes}
    requested = stop or (lambda: False)
    for run in runs:
        state = assessments[run.label]
        if state.done is not None:
            result["skipped"].append(run.label)
            continue
        if state.diverged is not None:
            result["diverged"].append(run.label)
            continue
        rounds = ((run.experiment, run.cell) in recovered
                  or (run.experiment, run.label) in own)  # fmt: skip
        limit = max_attempts * (2 if rounds else 1)
        attempts = len(state.technical)
        if dry_run:
            result["pending" if attempts < limit else "exhausted"].append(run.label)
            continue
        outcome = "exhausted"
        while attempts < limit:
            if requested():
                outcome = "stopped"
                break
            attempts += 1
            outcome = _attempt(run, attempts, runs_root, out_dir, log, execute, header, name,
                               extra)  # fmt: skip
            if outcome != "technical":
                break
        if outcome == "stopped" and not result["stopped"]:
            _log(log, {"event": "stop", "run": run.label, "reason": "stop requested", **extra})
        result[outcome if outcome != "technical" else "exhausted"].append(run.label)
    return result


def _open(state: Assessment) -> bool:
    return state.done is None and state.diverged is None


def _settle(run: Run, state: Assessment, records: Sequence[Mapping[str, Any]], log: Path,
            extra: Mapping[str, Any]) -> None:  # fmt: skip
    """Write the end of the run's start that has none (``open_start``), as ``run_campaign``
    says: the attempt's own run id and number, no return code (the launcher did not see the
    process end), the outcome and problem ``assess`` read for its directory (technical, with
    the directory missing, when it has none) and ``"settled": true``."""
    start = open_start(run, records)
    if start is None:
        return
    outcome, problem = state.outcomes.get(start["run_id"], ("technical", "no run directory"))
    _log(log, {"event": "end", "run": run.label, "attempt": start.get("attempt"),
               "run_id": start["run_id"], "returncode": None, "outcome": outcome,
               "problem": problem, "settled": True,
               **({"only": extra["only"]} if "only" in extra else {})})  # fmt: skip


def _attempt(run: Run, attempt: int, runs_root: Path, out_dir: Path, log: Path,
             execute: Callable[[Run, str, Path, Mapping[str, Any]], int],
             header: Mapping[str, Any] | None, name: str,
             extra: Mapping[str, Any] | None = None) -> str:  # fmt: skip
    """One attempt; returns done, diverged or technical; stops the campaign on a conflict.
    ``name`` is the campaign's (the child's ``ZFR_CAMPAIGN``); ``extra`` (``only``,
    ``max_attempts``) goes into its start and end records."""
    extra = dict(extra or {})
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = _run_id(stamp, run, attempt)
    output = out_dir / f"{run_id}.log"
    _log(log, {"event": "start", "run": run.label, "attempt": attempt, "run_id": run_id,
               "experiment": run.experiment, "config_hash": run.config_hash,
               "command": run.command()[1:],
               **({"header": dict(header)} if header is not None else {}), **extra})  # fmt: skip
    campaign = {"name": name, "label": run.label, "attempt": attempt,
                "header": None if header is None else dict(header)}  # fmt: skip
    code = execute(run, run_id, output, campaign)
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
               "returncode": code, "outcome": outcome, "problem": problem,
               **({"only": extra["only"]} if "only" in extra else {})})  # fmt: skip
    if outcome == "conflict":
        raise CampaignError(f"{run.experiment}/{run_id} completed but cannot stand for "
                            f"{run.label}: {problem}; the campaign stops")  # fmt: skip
    return outcome


def _rayleigh_problem(record: Mapping[str, Any] | None) -> str | None:
    """Why a run's endpoint Rayleigh record cannot stand (None: it can). R is descriptive and
    follows the published outcome: a null R with its reason (a zero or non-finite displacement,
    a non-finite product, a failed measurement) counts as done, as a measured R does. Only a
    missing or unreadable record, or one with neither a finite R nor a reason, does not."""
    if record is None:
        return "eval.rayleigh is enabled, but rayleigh.json is missing or unreadable"
    r, reason = record.get("R"), record.get("reason")
    measured = (isinstance(r, int | float) and not isinstance(r, bool) and math.isfinite(r)
                and reason is None)  # fmt: skip
    explained = r is None and isinstance(reason, str) and bool(reason)
    if not (measured or explained):
        return (f"rayleigh.json holds neither a finite R nor the reason for its absence "
                f"(R={r!r}, reason={reason!r})")  # fmt: skip
    return None


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
    except (OSError, ValueError, YAMLError):
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


def _subprocess(run: Run, run_id: str, output: Path,
                campaign: Mapping[str, Any] | None = None) -> int:  # fmt: skip
    env = {**os.environ, "RUN_ID": run_id}
    env.pop("ZFR_CAMPAIGN", None)  # a stale value of the launcher's own would be recorded
    if campaign is not None:
        env["ZFR_CAMPAIGN"] = json.dumps(campaign)
    with output.open("w") as f:
        return subprocess.run(run.command(), stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT,
                              env=env, check=False).returncode  # fmt: skip


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _log(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Append ``record``, stamped with the time, to a launch log as one line (module docstring,
    "The journal"): one ``os.write`` of the whole line (a short write raises ``OSError``), then
    an fsync, and the folder's fsync when the log is created. A torn last line is named first: a
    newline, then its ``torn`` record. Returns the stamped record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    stamped = {"time": now, **record}
    created = not path.exists()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        torn = _torn_tail(path)
        if torn is not None:  # the newline that ends the torn line, then the record naming it
            _write(fd, path, b"\n" + _line({"time": now, "event": "torn", **torn}))
        _write(fd, path, _line(stamped))
        os.fsync(fd)
    finally:
        os.close(fd)
    if created:
        folder = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(folder)
        finally:
            os.close(folder)
    return stamped


def _line(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record) + "\n").encode("utf-8")


def _write(fd: int, path: Path, data: bytes) -> None:
    written = os.write(fd, data)
    if written != len(data):
        raise OSError(f"{path}: a short write ({written} of {len(data)} bytes); the line is torn, "
                      "and the log's next append names it")  # fmt: skip


def _torn_tail(path: Path) -> dict[str, Any] | None:
    """The log's torn last line as its ``torn`` record names it (``offset``, ``length``,
    ``sha256``, ``kind``), or None when the log is empty or its last byte is a newline."""
    with path.open("rb") as f:
        if f.seek(0, os.SEEK_END) == 0:
            return None
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return None
        f.seek(0)
        data = f.read()
    offset = data.rfind(b"\n") + 1
    torn = data[offset:]
    return {"offset": offset, "length": len(torn), "sha256": hashlib.sha256(torn).hexdigest(),
            "kind": torn_kind(torn)}  # fmt: skip
