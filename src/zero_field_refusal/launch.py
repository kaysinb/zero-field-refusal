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
  invocations, unless a *recovery round* allows more (below). Diverged runs never enter one.

**Recovery rounds** (``recovery=REASON``, logged with the reason once the technical cause is fixed;
``label_rounds``, ``recovery_rounds``, ``attempts_left``). A round allows more attempts to a run
that used its attempts up, and never reruns an outcome:
- *A cell's round* (without ``only``): once per cell, ``max_attempts`` more to each of its runs
  that used theirs up. It grants nothing to a run whose label had a round of its own: a call that
  would open one over such a run is refused, naming it; no recovery round is written (a settled
  end of an open start may be written first). A study-3 campaign
  (its header carries ``study: 3``) takes none: ``recovery`` without ``only`` is refused with the
  command form of a run's own round, so that every study-3 round is documented and numbered.
- *A run's own rounds* (under ``only``; study 3's infrastructure route): one per documented
  incident. Each is logged with its reason; its incident, ``incident_time`` (ISO 8601, in UTC, not
  later than now), ``cause`` (one line) and ``evidence`` (a path or a reference), none blank;
  ``round``, its number among the label's rounds (its cell's and its own, under any config hash);
  the run's ``config_hash``; ``after``, the run ids of every start of the label under any config
  hash, in attempt order (the attempts it follows); ``max_attempts``; and ``granted``, the
  attempts it allows (``round_grant``): for the label's first round the smaller of
  ``max_attempts`` and the label's base (``base_attempts``: the ``max_attempts`` its first start
  recorded), exactly 1 for a further one, which needs ``max_attempts`` 1. Grants come from this
  rule, never from the journal: ``granted`` is written for the record, and a value that differs
  is the readers' problem, never a grant. A round opens only for a run that used its attempts up and
  holds no outcome: a done or diverged run is skipped (in a study-3 campaign the call is refused
  instead, below), a conflict (a provenance conflict: another
  gate, an unclean checkout, a departing numerics setup, a checkpoint missing or of another
  sha256; a missing endpoint checkpoint is a possible loss requiring diagnosis, never by itself a
  lost container disk) and a held label stop the campaign before, and an attempt, not
  superseded, whose directory holds a published terminal record, or whose end records the outcome
  done, diverged or conflict, refuses the round. Its reason, stripped, is one no earlier round of
  the label gave (its cell's or its own, under any config hash): a new reason is necessary, not
  sufficient. A refused round raises ``CampaignError`` naming the label and the earlier round
  (its number, log and line), the outcome, the incident's gap or the attempt bound, and the
  command form (``round_command``); no recovery round is written and nothing runs (a settled
  end of an open start may be written first, a dry run writes nothing). A run with attempts left
  takes them first, whatever reason is given: rounds never stack. In a study-3 campaign (its
  header carries ``study: 3``) a documented incident is never dropped: ``recovery`` over a run
  that is not used up under its config hash is refused (``_not_used_up``), in a dry run too,
  with the real call's message and exit, so that a run with
  attempts left, a new config hash's new run included, takes them without ``recovery``, and a
  done or diverged run keeps its outcome.
- *The attempts left* (``attempts_left``), per config hash: a new config hash is a new run, with
  its own base. The latest anchored round of the run's own (one that names ``after``) whose
  config hash is the run's allows what the rule grants it beyond the
  attempts it names, so that no later ``max_attempts`` recalculates an earlier grant and a round
  of another config hash gives the run nothing. No call runs more attempts than its own
  ``max_attempts``: it runs at most the smaller of that and the attempts left, and those it
  leaves stay for later calls. Without one, the run has its base,
  ``max_attempts`` or the one its first start recorded (``MAX_ATTEMPTS`` when it records none),
  whichever is less, doubled by one round in all: its own rounds written before rounds were
  numbered (none of ``NUMBERED_KEYS``) and its cell's round, as before; a numbered round that is
  not anchored grants nothing. The attempts used are ``Assessment.technical``, superseded ones
  included.
- *Attempt numbers.* A new start is numbered one more than the attempts used and than every
  attempt number of its label, in any log and under any config hash: monotonic per label, and
  under one config hash the number it always had. The run id keeps its form
  (``<stamp>-<cell>-s<seed>-<hash8>-a<n>``) and the label never changes. Nothing is deleted or
  rewritten: every attempt's directory, output and journal lines stay.

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
records carry ``only``, and a recovery round opened under it is that run's own (above), never its
cell's. ``max_attempts`` (default ``MAX_ATTEMPTS``) bounds the attempts of a run and of its
label's first round; the qualification passes 1, so that a failed launcher run is attempted again
only after a round of its own with its documented incident, one attempt per round. A start record
carries ``max_attempts`` when it is not the default.

**Superseded attempts** (``supersede``). A ``superseded`` record in the journal (written for study
3's qualification by a case restart or an engineering cycle's supersession) names an attempt
that no longer counts for its label: its directory and records stay, it is a used attempt (a
technical one), and it is neither done, diverged nor a conflict; the run is then attempted again
through a recovery round of its own (its first, or a further one). A ``held`` record (``hold``: an
outcome a study-3 engineering cycle could not resolve) makes the run a conflict: nothing of the
campaign starts, its recovery round included, in a dry run too. The hold applies to the run's
label (its experiment and ``<cell>/s<seed>``) whatever the config hash it was written with, so a
campaign whose cell now has another config hash is refused it as well; nothing here lifts a hold.
A checkpoint run (``tracking.endpoint_checkpoint``) is done only if its recorded numerics setup is
what the campaign's config asks for (``expected_numerics``: determinism, TF32 and the matmul
precision, the memory limit; read without touching CUDA).

**The journal** (``_log``). A record is appended as one line: the log is opened for appending
(created with mode 0o644) and locked (``flock``, exclusive, from the read of its tail to the last
fsync, so writers take turns), the whole line goes out in one ``os.write`` (a short write raises),
and then, still under the lock, the log, its folder ``launch/`` and the runs root that holds it
are fsynced, on every append: whoever created the folder or the log, and whether a creator died
before its own fsync, their entries are durable before an append returns. The logs are read line
by line, split at ``\n`` alone, each line decoded strictly as UTF-8 (``_journal_lines``, for
``launch_records`` and ``torn_lines`` alike).

A *torn line* is any line that holds no record: one that does not decode or parse (a byte-order
mark included), a valid JSON value that is not an object, a blank line (length 0), and a JSON
object whose identity is not all strings (``well_formed``: each key of ``IDENTITY_KEYS``,
``event``, ``run``, ``experiment``, ``run_id``, ``config_hash``, ``campaign``, ``log``,
``sha256``, ``cell`` and ``only``, absent or a string). A record whose identity key is present
with a value that is not a string therefore blocks until resolved, like any other torn line,
and is never read as matching nothing. The rule covers a malformed value alone: a record with
an identity key absent, or with an event no reader knows, is still a record, and matches
nothing; the journal is written only by the launcher and qualify.py, never by hand.
``launch_records`` returns the record of every other line, the exact complement of
``torn_lines``; ``_log`` refuses to write a record whose identity is not all strings, so the
launcher's own records are never torn. A crash in the middle of an append leaves a torn line at
the end of the log, the bytes after its last newline; the next append to that log repairs it
first, in two ``_write`` calls under its one lock: the newline that ends the torn line together
with a ``torn`` record naming its exact bytes (``offset`` of its first byte, ``length``,
``sha256``) and its ``kind``, then the record. The kind is the event the torn bytes show through
the strict pattern ``"event": "<lower-case word>"`` (``torn_kind``), or null; nothing else of
the torn bytes is copied anywhere. ``torn_lines`` lists every torn line of every log under
``RUNS_DIR/launch/`` with its log and line (a torn last line has no ``torn`` record yet),
whether a ``torn`` record names its bytes and whether a ``torn_resolved`` record names it (by
log, offset and sha256); ``launch_records`` skips them.

**A line a published record requires** (``missing_lines``). scripts/qualify.py publishes a
record first and appends its journal lines after it: an engineering cycle's supersession
(``qualification/c<N>/supersede-<n>.json``) requires the ``superseded`` line of each run of its
``launchers`` and the ``held`` line of each run of its ``held``, and a case restart
(``qualification/c<N>/<stage>/s<seed>/<case>/restart-<n>.json``) the ``superseded`` line of its
``launcher`` run. A crash between the record and its lines, or an append that wrote no byte,
leaves the record without them and no torn byte to see. ``missing_lines`` reconciles every such
record under ``RUNS_DIR/qualification/`` with the journal: a line is there when a line of its
event names the same label, experiment and run id, whatever its config hash (as ``hold``,
``supersede`` and their callers judge it; a restart's run by its start's label and experiment,
else its run directory's, else by its run id alone). A missing line is named by its record's
path, its event and its run id, never by the record's text; a record that cannot be read, and
an entry that names no run, are named by the record's path and block too.

**The journal is not whole** while any log holds a torn line that no ``torn_resolved`` record
names, or a published record requires a line no log holds. Then ``run_campaign`` starts nothing,
settles nothing and opens no recovery round, in a dry run too: it raises ``TornJournalError``,
and scripts/launch.py prints each torn line (log and line, offset, length, kind and the sha256
its resolution names) and each missing line, and exits ``TORN_JOURNAL`` (12). The code means that
alone: the launcher's other exits are 0, 1 and argparse's 2, and study 3's qualification
commands use 1, 3 and 8 to 11; they refuse with the same refusal and code, but for their two
journal repairs (scripts/qualify.py: ``supersede`` run again with its published record, which
restores a cycle's lines, and ``restart-case`` with a published restart record, which restores
its line), and their driver passes it on. Every log is read, since a label's held and superseded
lines can live in any of them. What a lost line costs, by its kind, and what restores it before
the resolution:

- ``end``: the start stands without its end. Nothing is rerun; once the line is resolved, the
  launcher settles the start from its run directory, as any start found without its end.
- ``start``: a run directory may stand with no start (study 3's analysis lists it and blocks).
  The launcher counts that directory as any directory of the run's config and never reuses its
  run id: a new attempt's id carries its own time and attempt number, and the tracker never
  writes into an existing run directory.
- ``held``, ``superseded``: their writers run again first, over the unresolved torn line, so
  that a held run is never rerun and a superseded one never counts: ``qualify.py supersede`` with
  the same record, and ``qualify.py restart-case`` again for the case, the two calls qualify.py
  lets through a torn line. Each writes only the missing lines: ``hold`` and ``supersede`` write
  a line only when no identical one (the same event, label, experiment, run id and reason) is
  in the journal.
- ``recovery``, ``stop``: once the line is resolved, the operator relaunches with the same
  command; a recovery round's reason is recorded again.
- ``torn``: a lost ``torn`` record costs nothing more; the resolution writes it (below).
- ``torn_resolved``: the line it resolved is unresolved again; the operator runs that
  resolution again.
- unknown (null; a blank line, say): the operator reads the run directories of the campaign's
  labels and records the resolution.

``resolve_torn`` (``scripts/launch.py --resolve-torn LOG --offset N --sha256 HEX --reason TEXT``)
appends ``{"event": "torn_resolved", "log", "offset", "sha256", "reason"}`` to that log through
``_log``. A line that no ``torn`` record names (a power loss or a short write inside the repair,
a blank line, a line an older writer merged) gets that record first, through ``_log`` too, so
that every resolved line is named by both records. It refuses an offset and sha256 that name no
torn line of that log, a line already resolved and an empty reason. It does not recover the
lost record: it records that the operator restored what the journal needs.
"""

from __future__ import annotations

import fcntl
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
# The exit while the launch journal is not whole: a torn line unresolved, or a line a published
# qualification record requires missing (module docstring, "The journal")
TORN_JOURNAL = 12
# The one part of a torn line's bytes that a record copies: its kind (module docstring)
TORN_KIND = re.compile(rb'"event": "([a-z][a-z_]{0,31})"')
# A journal record's identity: each of these keys, when present, holds a string. A JSON object
# with any of them present and not a string is a torn line (module docstring, "The journal");
# every record the launcher writes holds strings there (``_log`` refuses one that does not).
IDENTITY_KEYS = ("event", "run", "experiment", "run_id", "config_hash", "campaign", "log",
                 "sha256", "cell", "only")  # fmt: skip
# The incident a run's own recovery round documents (module docstring, "Recovery rounds")
INCIDENT_KEYS = ("incident_time", "cause", "evidence")
# What a round carries since rounds are numbered: a round of a run's own without any of them was
# written before (module docstring, "Recovery rounds")
NUMBERED_KEYS = ("round", "after", "granted", "max_attempts", "config_hash", *INCIDENT_KEYS)
# ``journal_entries``: each record with its log's file name and its line number
Entries = Sequence[tuple[str, int, Mapping[str, Any]]]


class CampaignError(ValueError):
    """The campaign cannot start, or go on, as it stands."""


@dataclass(frozen=True)
class TornLine:
    """A torn line of a launch log (``torn_lines``): its log's file name, the offset of its first
    byte, its length in bytes and their sha256, its kind (``torn_kind``), whether a ``torn``
    record of that log names exactly those bytes, whether a ``torn_resolved`` record names it
    (by log, offset and sha256), and its line number in the log, from 1."""

    log: str
    offset: int
    length: int
    sha256: str
    kind: str | None
    repaired: bool
    resolved: bool
    line: int


# A run id shown in a message: a plain name, as the tracker accepts one (anything else is not shown)
PLAIN_RUN_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")


@dataclass(frozen=True)
class MissingLine:
    """A journal line that a published qualification record requires and no launch log holds
    (``missing_lines``): the record's path under the runs root, the line's event and the run id
    it names. ``event`` and ``run_id`` are None for a record that cannot be read; ``run_id``
    alone is None for an entry that names no run. Each blocks as a missing line does."""

    record: str
    event: str | None
    run_id: str | None

    def __str__(self) -> str:
        if self.event is None:
            return f"{self.record}: a published record that cannot be read"
        if self.run_id is None:
            return f"{self.record}: a {self.event} entry that names no run"
        shown = self.run_id if PLAIN_RUN_ID.fullmatch(self.run_id) else "(a run id not shown)"
        return f"{self.record}: no {self.event} line of {shown}"


class TornJournalError(CampaignError):
    """The launch journal is not whole (module docstring, "The journal"): a launch log holds a
    torn line that no ``torn_resolved`` record names (``lines``), or a published qualification
    record requires a line that no log holds (``missing``). The launcher starts nothing,
    settles nothing and opens no recovery round."""

    def __init__(self, lines: Sequence[TornLine], missing: Sequence[MissingLine] = ()) -> None:
        self.lines, self.missing = list(lines), list(missing)
        parts = ["The launch journal is not whole; nothing starts, is settled or opens a "
                 "recovery round until it is."]  # fmt: skip
        if self.lines:
            listed = "\n".join(f"- {x.log} line {x.line}: offset {x.offset}, length "
                               f"{x.length}, kind {x.kind or 'null'}, sha256 {x.sha256}"
                               for x in self.lines)  # fmt: skip
            parts.append(
                "Torn lines that no torn_resolved record names: restore what each kind lost "
                '(docs/operations.md, "Launching runs"), then run scripts/launch.py '
                f"--resolve-torn LOG --offset N --sha256 HEX --reason TEXT.\n{listed}"
            )
        if self.missing:
            listed = "\n".join(f"- {m}" for m in self.missing)
            parts.append(
                "Lines that published qualification records require and no launch log holds: "
                "restore them with the record's own repair, scripts/qualify.py supersede run "
                "again with the cycle's record, or scripts/qualify.py restart-case for the "
                'restart\'s case (docs/operations.md, "Launching runs"); an unreadable record '
                f"is restored by hand.\n{listed}"
            )
        super().__init__("\n".join(parts))


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
    # ids of its attempts, not superseded, whose directory holds a published terminal record
    # (``terminal.json``), whatever their status: no recovery round of the run's own over them
    published: set[str] = field(default_factory=set)
    # run dir -> (outcome, problem) as the launcher ends that attempt: done, diverged, conflict
    # (the problem is why it cannot stand) or technical; an attempt without a directory is absent
    outcomes: dict[str, tuple[str, str | None]] = field(default_factory=dict)


def load_campaign(path: str | Path) -> tuple[str, list[Run]]:
    """The campaign's name and its runs in order, every config loaded and checked. A cell whose
    name is not a string (a YAML number, list, mapping or null) is a ``CampaignError`` naming
    the cell by its position: its runs' labels and its journal records carry the name."""
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        raise ValueError(f"{path}: a campaign needs a name and a list of cells")
    cells = data.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{path}: a campaign needs a non-empty list of cells")
    runs, names = [], set()
    for position, cell in enumerate(cells, start=1):
        if isinstance(cell, dict) and "name" in cell and not isinstance(cell["name"], str):
            raise CampaignError(f"{path}: cell {position} of the campaign: its name must be a "
                                f"string, not {type(cell['name']).__name__}")  # fmt: skip
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
    """Every record of every launch log under ``runs_root/launch/``, in file name and line
    order: each line (split at ``\n`` alone and decoded strictly as UTF-8, as ``torn_lines``
    reads it) that holds a JSON object. A torn line is skipped."""
    return [record for _, _, record in journal_entries(runs_root)]


def journal_entries(runs_root: Path) -> list[tuple[str, int, dict[str, Any]]]:
    """``launch_records`` with each record's place: its log's file name and its line number in
    that log, from 1 (a torn line keeps its number and holds no record)."""
    return [(log.name, number, record)
            for log in sorted((Path(runs_root) / "launch").glob("*.jsonl"))
            for number, (_, _, record) in enumerate(_journal_lines(log), start=1)
            if record is not None]  # fmt: skip


def journal_records(log: Path) -> list[dict[str, Any]]:
    """The records of one launch log, in line order, as ``launch_records`` reads them."""
    return [record for _, _, record in _journal_lines(log) if record is not None]


def torn_kind(data: bytes) -> str | None:
    """The event a torn line's bytes show through the strict pattern ``TORN_KIND``
    (``"event": "<lower-case word>"``), or None; nothing else of the bytes is read out."""
    match = TORN_KIND.search(data)
    return match[1].decode("ascii") if match else None


def torn_lines(runs_root: Path) -> list[TornLine]:
    """Every torn line of every launch log under ``runs_root/launch/``, in file name and offset
    order (module docstring, "The journal"): every line, split at ``\n`` alone, that is no
    record (``_journal_lines``): one that does not decode strictly as UTF-8 or parse as a JSON
    object, a blank line (length 0), a valid JSON value that is not an object, and a JSON object
    whose identity (``IDENTITY_KEYS``) is not all strings. A torn last line, with no newline
    after it, is listed unrepaired: no record follows it yet. ``launch_records`` returns every
    other line's record: the two are each other's complement."""
    logs = {log.name: _journal_lines(log)
            for log in sorted((Path(runs_root) / "launch").glob("*.jsonl"))}  # fmt: skip
    resolved = {(r.get("log"), r.get("offset"), r.get("sha256"))
                for lines in logs.values() for _, _, r in lines
                if r is not None and r.get("event") == "torn_resolved"
                and _count(r.get("offset")) and _identity(r, "log", "sha256")}  # fmt: skip
    found = []
    for name, lines in logs.items():
        named = {(r.get("offset"), r.get("length"), r.get("sha256")) for _, _, r in lines
                 if r is not None and r.get("event") == "torn" and _identity(r, "sha256")
                 and _count(r.get("offset")) and _count(r.get("length"))}  # fmt: skip
        for number, (offset, data, record) in enumerate(lines, start=1):
            if record is not None:
                continue
            sha = hashlib.sha256(data).hexdigest()
            found.append(TornLine(name, offset, len(data), sha, torn_kind(data),
                                  (offset, len(data), sha) in named,
                                  (name, offset, sha) in resolved, number))  # fmt: skip
    return found


def resolve_torn(runs_root: Path, log: str, *, offset: int, sha256: str,
                 reason: str) -> dict[str, Any]:  # fmt: skip
    """Record that the operator restored what the journal needs after the torn line of ``log``
    (a file name under ``runs_root/launch/``) at ``offset`` with ``sha256`` lost its record (module
    docstring, "The journal"): a ``torn_resolved`` record appended to that log through ``_log``,
    which is returned. A line that no ``torn`` record names gets that record first, through
    ``_log`` too (a torn last line by the append's own repair), so that both records name every
    resolved line. Raises ``ValueError``, writing nothing, for an empty reason, an offset and
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
    line, path = named[0], Path(runs_root) / "launch" / log
    if line.resolved:
        raise ValueError(f"{log}: the torn line at offset {offset} is already resolved")
    if not line.repaired and path.stat().st_size != line.offset + line.length:  # not the last
        _log(path, {"event": "torn", "offset": line.offset, "length": line.length,
                    "sha256": line.sha256, "kind": line.kind})  # fmt: skip
    return _log(path, {"event": "torn_resolved", "log": log, "offset": offset, "sha256": sha256,
                       "reason": reason})  # fmt: skip


# The published qualification records whose lines the journal must hold (scripts/qualify.py
# writes them under RUNS_DIR/qualification/): a candidate's supersession, a case's restart
CYCLE_RECORDS = ("c*/supersede-*.json", re.compile(r"supersede-\d+\.json"))
RESTART_RECORDS = ("c*/*/*/*/restart-*.json", re.compile(r"restart-\d+\.json"))


def missing_lines(runs_root: Path) -> list[MissingLine]:
    """Every line that a published qualification record requires and no launch log holds
    (module docstring, "The journal"), in record path order:
    - a supersession record (``qualification/c<N>/supersede-<n>.json``): the ``superseded`` line
      of every entry of its ``launchers`` and the ``held`` line of every entry of its ``held``,
      each entry with the label (``run``), experiment and run id its line carries;
    - a case restart record (``qualification/c<N>/<stage>/s<seed>/<case>/restart-<n>.json``):
      the ``superseded`` line of its ``launcher`` run, when it names one, under that run's own
      label and experiment: its start's in the journal, else its run directory's (the campaign
      label in ``meta.json``, the experiment folder), else matched by its run id alone.
    A line is there when a record of its event names the same label, experiment and run id,
    whatever its config hash, as ``supersede``, ``hold`` and their callers judge it. A record
    that cannot be read, and an entry that names no run, are listed as well: either may name a
    line, so either blocks."""
    root = Path(runs_root)
    out, records = root / "qualification", launch_records(root)
    lines = {_identity(r, "event", "run", "experiment", "run_id") for r in records} - {None}
    missing: list[MissingLine] = []
    for path in _records_of(out, CYCLE_RECORDS):
        where, record = path.relative_to(root).as_posix(), _read_json(path)
        if record is None:
            missing.append(MissingLine(where, None, None))
            continue
        for event, key in (("superseded", "launchers"), ("held", "held")):
            entries = record.get(key) or []
            for item in entries if isinstance(entries, list) else [None]:
                named = [item.get(k) for k in ("run", "experiment", "run_id")] if isinstance(
                    item, dict) else [None]  # fmt: skip
                if not all(isinstance(value, str) for value in named):
                    missing.append(MissingLine(where, event, None))
                elif (event, *named) not in lines:
                    missing.append(MissingLine(where, event, named[2]))
    for path in _records_of(out, RESTART_RECORDS):
        where, record = path.relative_to(root).as_posix(), _read_json(path)
        if record is None:
            missing.append(MissingLine(where, None, None))
            continue
        launcher = record.get("launcher")
        run_id = launcher.get("run_id") if isinstance(launcher, dict) else None
        if not isinstance(run_id, str):
            continue  # no launcher run (one lost before it settled): no line
        labels = _run_labels(root, records, run_id)
        there = (any(("superseded", label, experiment, run_id) in lines
                     for label, experiment in labels) if labels
                 else any(r.get("event") == "superseded" and r.get("run_id") == run_id
                          for r in records))  # fmt: skip
        if not there:
            missing.append(MissingLine(where, "superseded", run_id))
    return missing


def _records_of(out: Path, kind: tuple[str, re.Pattern[str]]) -> list[Path]:
    pattern, name = kind
    return sorted(p for p in out.glob(pattern) if name.fullmatch(p.name)) if out.is_dir() else []


def _run_labels(root: Path, records: Sequence[Mapping[str, Any]],
                run_id: str) -> set[tuple[Any, Any]]:  # fmt: skip
    """A launcher run's label and experiment: its starts', else its run directory's."""
    labels = {_identity(r, "run", "experiment") for r in records
              if r.get("event") == "start" and r.get("run_id") == run_id} - {None}  # fmt: skip
    if labels or not PLAIN_RUN_ID.fullmatch(run_id):
        return labels
    for meta_path in sorted(root.glob(f"*/{run_id}/meta.json")):
        meta = _read_json(meta_path) or {}
        label = (meta.get("campaign") or {}).get("label") if isinstance(meta.get("campaign"),
                                                                        dict) else None  # fmt: skip
        if isinstance(label, str):
            labels.add((label, meta_path.parent.parent.name))
    return labels


def _identity(record: Mapping[str, Any], *keys: str) -> tuple[str, ...] | None:
    """The record's values of ``keys`` when every one is a string, else None: a journal record
    whose identity field is a list, an object or anything but a string matches nothing, and
    never stops a reading with a ``TypeError``."""
    values = tuple(record.get(key) for key in keys)
    return values if all(isinstance(value, str) for value in values) else None


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _journal_lines(log: Path) -> list[tuple[int, bytes, dict[str, Any] | None]]:
    """Each line of a launch log, split at ``\n`` alone: its offset, its bytes and the record it
    holds, None (a torn line) when its bytes do not decode strictly as UTF-8 (a byte-order mark
    stays in the text), do not parse as a JSON object, or parse as one whose identity is not all
    strings (``well_formed``). The one reader of the logs' bytes, for ``launch_records`` and
    ``torn_lines`` alike."""
    data, lines, offset = log.read_bytes(), [], 0
    while offset < len(data):
        end = data.find(b"\n", offset)
        end = len(data) if end < 0 else end
        try:
            record = json.loads(data[offset:end].decode("utf-8"))
        except ValueError:  # a UnicodeDecodeError included
            record = None
        whole = isinstance(record, dict) and well_formed(record)
        lines.append((offset, data[offset:end], record if whole else None))
        offset = end + 1
    return lines


def well_formed(record: Mapping[str, Any]) -> bool:
    """Whether a journal object's identity is all strings: each key of ``IDENTITY_KEYS`` absent
    or a string. ``offset`` and ``length`` are no identity: a ``torn`` or ``torn_resolved``
    record carries them as integers, and a record whose offset or length is not one names
    nothing."""
    return all(isinstance(record[key], str) for key in IDENTITY_KEYS if key in record)


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


def label_starts(
    run: Run, records: Sequence[Mapping[str, Any]], *, every_config_hash: bool = False
) -> list[Mapping[str, Any]]:
    """The start records of the run's attempts (its label, experiment and config hash, with a
    run id), in attempt order (``attempt_key``); with ``every_config_hash``, those of its label
    under any config hash. ``launch_records`` reads the journals in file name order, so a later
    attempt logged under another campaign's name may come first there."""
    starts = [r for r in records
              if r.get("event") == "start" and r.get("run") == run.label
              and r.get("experiment") == run.experiment
              and (every_config_hash or r.get("config_hash") == run.config_hash)
              and isinstance(r.get("run_id"), str)]  # fmt: skip
    return sorted(starts, key=attempt_key)


def open_start(run: Run, records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The start of the run's latest attempt that no ``end``, ``superseded`` or ``held`` record
    names by its run id (``run_campaign``, "A start found without its end"), or None. Attempts
    are ordered by their number, then the start's time (``label_starts``), whatever journal file
    holds each record; a record closes the start whose run id it names, wherever it is."""
    closed = {r["run_id"] for r in records if r.get("event") in ("end", "superseded", "held")
              and isinstance(r.get("run_id"), str)}  # fmt: skip
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
                if run_dir.name not in result.superseded and (run_dir / "terminal.json").is_file():
                    result.published.add(run_dir.name)
            continue
        if meta.get("config_hash") != run.config_hash:
            continue
        if run_dir.name in result.superseded:  # kept, and a used attempt; it no longer counts
            result.technical.add(run_dir.name)
            result.outcomes[run_dir.name] = (
                "technical", f"named superseded in the journal: {named[run_dir.name]}")  # fmt: skip
            continue
        if (run_dir / "terminal.json").is_file():
            result.published.add(run_dir.name)
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
                and isinstance(record.get("run_id"), str)
                and record.get("run_id") not in settled):  # fmt: skip
            result.technical.add(record["run_id"])
    return result


@dataclass(frozen=True)
class Round:
    """A recovery round as the journal holds it (module docstring, "Recovery rounds"): its log's
    file name, its line number in that log (from 1) and its record. A cell's round has no
    ``only``; a run's own round names its label in ``only``, and one written since rounds are
    numbered also carries ``round``, ``config_hash``, ``after``, ``max_attempts``, ``granted``
    and its incident (``INCIDENT_KEYS``)."""

    log: str
    line: int
    record: Mapping[str, Any]

    @property
    def own(self) -> bool:
        return self.record.get("only") is not None

    @property
    def number(self) -> int | None:
        value = self.record.get("round")
        return value if _count(value) else None

    @property
    def after(self) -> tuple[str, ...] | None:
        """The run ids of the attempts it follows, or None (not a list of run ids)."""
        value = self.record.get("after")
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(value)
        return None

    @property
    def anchored(self) -> bool:
        """A run's own round that names the attempts it follows (``after``); what it allows is
        the rule's (``round_grant``), never its recorded ``granted``."""
        return self.own and self.after is not None

    @property
    def numbered(self) -> bool:
        """Written since rounds are numbered: it carries any of ``NUMBERED_KEYS``."""
        return any(key in self.record for key in NUMBERED_KEYS)

    @property
    def reason(self) -> str:
        reason = self.record.get("reason")
        return reason.strip() if isinstance(reason, str) else repr(reason)

    def __str__(self) -> str:
        number = f"round {self.number} " if self.number is not None else "a round "
        return f"{number}({self.log} line {self.line})"


def label_rounds(entries: Entries, run: Run) -> list[Round]:
    """Every recovery round of the run's label (``journal_entries``' records, every log),
    whatever its config hash: its cell's (a ``recovery`` record without ``only``, by experiment
    and cell) and its own (``only`` its label, in its experiment), in the order of their time
    stamps, then their numbers, then the logs' names and lines."""
    found = [Round(log, line, record) for log, line, record in entries
             if record.get("event") == "recovery" and record.get("experiment") == run.experiment
             and ((record.get("only") is None and record.get("cell") == run.cell)
                  or record.get("only") == run.label)]  # fmt: skip
    return sorted(found, key=lambda x: (str(x.record.get("time") or ""), x.number or 0))


def recovery_rounds(entries: Entries, run: Run) -> list[Round]:
    """The rounds that apply to the run (``label_rounds``): its cell's, and its own whose
    config hash is absent (written before rounds were numbered) or the run's."""
    return [x for x in label_rounds(entries, run)
            if not x.own or x.record.get("config_hash") in (None, run.config_hash)]  # fmt: skip


def base_attempts(run: Run, records: Sequence[Mapping[str, Any]], max_attempts: int) -> int:
    """The label's base: the ``max_attempts`` its first start (under the run's config hash)
    recorded, ``MAX_ATTEMPTS`` when it records none; ``max_attempts`` before any start."""
    starts = label_starts(run, records)
    first = starts[0].get("max_attempts", MAX_ATTEMPTS) if starts else max_attempts
    return first if _count(first) and first >= 1 else MAX_ATTEMPTS


def round_grant(item: Round, rounds: Sequence[Round], base: int) -> int:
    """What a round of the run's own allows, by rule, never read from the journal (module
    docstring, "Recovery rounds"): the label's first round (no round of the label, its cell's or
    its own, before it in ``rounds``) the smaller of the ``max_attempts`` it was opened with (1
    when it records none) and the label's base; every further round exactly 1."""
    earlier = next((n for n, x in enumerate(rounds) if (x.log, x.line) == (item.log, item.line)),
                   len(rounds))  # fmt: skip
    if earlier:
        return 1
    asked = item.record.get("max_attempts")
    return min(asked if _count(asked) and asked >= 1 else 1, base)


def attempts_left(run: Run, state: Assessment, entries: Entries, max_attempts: int) -> int:
    """How many attempts the run may still make, over every call (module docstring, "Recovery
    rounds"); a call runs at most ``max_attempts`` of them. The latest anchored round of its own
    that applies (``recovery_rounds``) allows what the rule grants it (``round_grant``) beyond
    the attempts it names (``after``): the run's starts it does not name are spent from it.
    Without one, the run has its base, ``max_attempts`` and the label's base
    (``base_attempts``), whichever is less, doubled by one round in all: its own rounds written
    before rounds were numbered and its cell's round, which grants nothing to a label that had a
    round of its own; a numbered round that is not anchored grants nothing. The attempts used
    are ``Assessment.technical``, superseded ones included."""
    records = [record for _, _, record in entries]
    rounds = label_rounds(entries, run)
    applying = recovery_rounds(entries, run)
    anchored = [x for x in applying if x.anchored]
    first = base_attempts(run, records, max_attempts)
    if anchored:
        latest = anchored[-1]
        spent = {s["run_id"] for s in label_starts(run, records)} - set(latest.after or ())
        return max(0, round_grant(latest, rounds, first) - len(spent))
    base = min(max_attempts, first)
    owned = any(x.own for x in rounds)
    doubled = any((x.own and not x.numbered) or (not x.own and not owned) for x in applying)
    return max(0, base * (2 if doubled else 1) - len(state.technical))


def _highest_attempt(run: Run, records: Sequence[Mapping[str, Any]]) -> int:
    """The highest attempt number of any start of the run's label, whatever its log and config
    hash (0 without one)."""
    starts = label_starts(run, records, every_config_hash=True)
    return max((attempt_key(r)[0] for r in starts), default=0)


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


LOST_CHECKPOINT_NOTE = (
    "\nA missing endpoint checkpoint is a possible loss requiring diagnosis before anything is\n"
    "moved, deleted or rerun under study 3's qualification (docs/operations.md,\n"
    '"A lost checkpoint"), unless finish removed it.'
)


def is_missing_checkpoint(problem: str) -> bool:
    """The conflict ``_checkpoint_problem`` gives a run whose endpoint checkpoint file is gone."""
    return problem.startswith("the endpoint checkpoint ") and problem.endswith(" is missing")


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
    incident: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Run the missing runs in order; returns the labels by outcome.

    Outcomes: skipped (a qualifying run exists), done, diverged, exhausted (no attempt left),
    stopped (``stop()`` was true before its next attempt; logged once) and, for a dry run,
    pending. ``execute(run, run_id, output, campaign)`` runs one attempt under that run id and
    returns its exit code (default: a subprocess); ``campaign`` is the campaign's name, the run's
    label, the attempt and the header, which the child reads from ``ZFR_CAMPAIGN``.
    ``recovery`` names the fixed technical cause: without ``only`` it opens the recovery round of
    every cell with an exhausted run that has not had one; under ``only`` it opens the run's own
    round, with ``incident`` (``INCIDENT_KEYS``: the interruption it documents), when the run used
    its attempts up (module docstring, "Recovery rounds"). ``header`` (``campaign_header``) goes
    into every start record. ``only`` runs that label alone and ``max_attempts`` bounds the
    attempts (module docstring). Raises ``CampaignError`` before starting anything, its recovery
    round included and in a dry run too, if a run's existing directories conflict or its label
    has a ``held`` run in any journal, whatever that run's config hash (the conflict names it and
    the hold's reason), or if ``only`` names no run; before starting anything and writing no
    recovery round (a settled end of an open start may be written first), when a recovery round
    asked for is refused ("Recovery rounds"), or ``incident`` is given
    without ``only``; and stops at an attempt that completes without qualifying.

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

    **A journal that is not whole** (module docstring, "The journal"). Before anything else,
    while any log under ``runs_root/launch/`` holds a torn line that no ``torn_resolved`` record
    names (``torn_lines``), or a published qualification record under
    ``runs_root/qualification/`` requires a ``held`` or ``superseded`` line that no log holds
    (``missing_lines``; an unreadable record, or an entry that names no run, too), it raises
    ``TornJournalError`` naming each: nothing starts, nothing is settled or written, no recovery
    round opens, in a dry run too. A torn line is lifted by restoring what its kind lost and then
    its resolution (``resolve_torn``); a missing line by its record's own repair alone
    (scripts/qualify.py ``supersede`` run again with the cycle's record, or ``restart-case`` for
    the restart's case); an unreadable record or an entry without a run, by hand.
    """
    unresolved = [line for line in torn_lines(runs_root) if not line.resolved]
    missing = missing_lines(runs_root)
    if unresolved or missing:
        raise TornJournalError(unresolved, missing)
    execute = execute or _subprocess
    if not (isinstance(max_attempts, int) and max_attempts >= 1):
        raise CampaignError(f"max_attempts must be a positive integer, not {max_attempts!r}")
    if incident is not None and only is None:
        raise CampaignError("An incident documents a run's own recovery round: it goes with "
                            "only (--only LABEL) and its reason (--recovery)")  # fmt: skip
    if recovery is not None and only is None and (header or {}).get("study") == 3:
        raise CampaignError(
            "A study-3 campaign takes no cell recovery round: each round is a run's own, "
            "documented and numbered: scripts/launch.py CAMPAIGN --only LABEL --max-attempts 1 "
            '--recovery "<reason>" --incident-time UTC --cause TEXT --evidence REF, its reason '
            "one that no earlier round of the label gave"
        )
    if only is not None:
        runs = [run for run in runs if run.label == only]
        if not runs:
            raise CampaignError(f"The campaign has no run labelled {only!r}")
    entries = journal_entries(runs_root)
    records = [record for _, _, record in entries]
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
        note = LOST_CHECKPOINT_NOTE if any(map(is_missing_checkpoint, conflicts.values())) else ""
        raise CampaignError(f"Completed runs that cannot stand for this campaign; move them out "
                            f"of the experiment or fix them first:\n{lines}{note}")  # fmt: skip

    def used_up(run: Run) -> bool:
        state = assessments[run.label]
        return _open(state) and attempts_left(run, state, entries, max_attempts) == 0

    if (recovery is not None and only is not None and (header or {}).get("study") == 3
            and not used_up(runs[0])):  # fmt: skip
        # its documented incident is never dropped; a dry run reports the real call's refusal
        _not_used_up(runs[0], assessments[runs[0].label], entries, max_attempts)
    if recovery is not None and not dry_run and only is not None:
        run = runs[0]
        if used_up(run):  # a run with attempts left takes them: rounds never stack
            _log(log, _own_round(run, assessments[run.label], entries, recovery, incident,
                                 max_attempts))  # fmt: skip
            entries = journal_entries(runs_root)
    elif recovery and not dry_run:
        opening = []
        for experiment, cell in sorted({(r.experiment, r.cell) for r in runs}):
            exhausted = [r for r in runs if (r.experiment, r.cell) == (experiment, cell)
                         and used_up(r)]  # fmt: skip
            if exhausted and not any(not x.own for x in label_rounds(entries, exhausted[0])):
                _no_cell_round_over_own_rounds(exhausted, entries)
                labels = [r.label for r in exhausted]
                opening.append({"event": "recovery", "experiment": experiment, "cell": cell,
                                "reason": recovery, "runs": labels})  # fmt: skip
        for record in opening:
            _log(log, record)
        if opening:
            entries = journal_entries(runs_root)
    records = [record for _, _, record in entries]
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
        left = min(attempts_left(run, state, entries, max_attempts), max_attempts)  # this call's
        if dry_run:
            result["pending" if left > 0 else "exhausted"].append(run.label)
            continue
        used, number = len(state.technical), _highest_attempt(run, records)
        outcome = "exhausted"
        while left > 0:
            if requested():
                outcome = "stopped"
                break
            number = 1 + max(used, number)  # monotonic per label, whatever the config hash
            used, left = used + 1, left - 1
            outcome = _attempt(run, number, runs_root, out_dir, log, execute, header, name,
                               extra)  # fmt: skip
            if outcome != "technical":
                break
        if outcome == "stopped" and not result["stopped"]:
            _log(log, {"event": "stop", "run": run.label, "reason": "stop requested", **extra})
        result[outcome if outcome != "technical" else "exhausted"].append(run.label)
    return result


def _open(state: Assessment) -> bool:
    return state.done is None and state.diverged is None


def round_command(label: str) -> str:
    """The command form of a further recovery round of ``label`` (module docstring, "Recovery
    rounds")."""
    return (f"scripts/launch.py CAMPAIGN --only {label} --max-attempts 1 --recovery REASON "
            "--incident-time UTC --cause TEXT --evidence REF, its reason one that no earlier "
            "round of the label gave")  # fmt: skip


def _own_round(
    run: Run,
    state: Assessment,
    entries: Entries,
    reason: Any,
    incident: Mapping[str, Any] | None,
    max_attempts: int,
) -> dict[str, Any]:
    """The record of a round of the run's own, for a run that used its attempts up (module
    docstring, "Recovery rounds"), or ``CampaignError`` naming why none opens: an outcome its
    attempts hold, an incident not documented, a blank reason or one an earlier round of the
    label gave, a further round at more than one attempt."""
    label, records = run.label, [record for _, _, record in entries]
    form = round_command(label)
    kept = _outcome_held(run, state, records)
    if kept is not None:
        raise CampaignError(f"{label}: no recovery round over an outcome its attempts hold: "
                            f"{kept}. A published outcome and a provenance conflict are never "
                            "rerun: the technical route decides")  # fmt: skip
    problems = _incident_problems(incident)
    if problems:
        raise CampaignError(f"{label}: a recovery round records its incident, the "
                            f"interruption it follows: {'; '.join(problems)}. {form}")  # fmt: skip
    if not isinstance(reason, str) or not reason.strip():
        raise CampaignError(f"{label}: a recovery round needs its reason, the documented "
                            f"interruption, not a blank one: {form}")  # fmt: skip
    rounds = label_rounds(entries, run)
    for earlier in rounds:
        if earlier.reason == reason.strip():
            raise CampaignError(f"{label}: {earlier} gave this reason already; a further round "
                                f"documents an interruption of its own: {form}")  # fmt: skip
    number = len(rounds) + 1
    if number > 1 and max_attempts != 1:
        raise CampaignError(f"{label}: a further recovery round (round {number}; the label had "
                            f"{len(rounds)}) allows one attempt: run it with --max-attempts 1, "
                            f"not {max_attempts}: {form}")  # fmt: skip
    starts = label_starts(run, records, every_config_hash=True)
    after = list(dict.fromkeys(r["run_id"] for r in starts))
    return {"event": "recovery", "experiment": run.experiment, "cell": run.cell,
            "reason": reason, "runs": [label], "only": label, "round": number,
            "config_hash": run.config_hash, "after": after, "max_attempts": max_attempts,
            "granted": 1 if number > 1 else min(max_attempts, base_attempts(run, records,
                                                                            max_attempts)),
            **{key: incident[key] for key in INCIDENT_KEYS}}  # type: ignore[index]  # fmt: skip


def _outcome_held(run: Run, state: Assessment, records: Sequence[Mapping[str, Any]]) -> str | None:
    """Why the run's attempts hold an outcome no recovery round reruns (None: they hold none):
    an attempt, not superseded, whose directory holds a published terminal record, or whose end
    in the journal records the outcome done, diverged or conflict (its directory moved out since).
    Done, diverged and conflicting directories are read before (``assess``)."""
    if state.published:
        return f"{sorted(state.published)[0]}: its terminal record is published"
    starts = {s["run_id"] for s in label_starts(run, records)} - state.superseded
    for record in records:
        if (record.get("event") == "end" and record.get("run_id") in starts
                and record.get("outcome") in ("done", "diverged", "conflict")):  # fmt: skip
            return f"{record['run_id']}: its end records the outcome {record['outcome']}"
    return None


def _incident_problems(incident: Mapping[str, Any] | None) -> list[str]:
    """What the incident of a run's own round lacks (module docstring, "Recovery rounds"):
    ``incident_time``, an ISO 8601 time in UTC not later than now; ``cause``, one line;
    ``evidence``; none blank."""
    incident = incident or {}
    problems = [f"{key} is missing or blank" for key in INCIDENT_KEYS
                if not isinstance(incident.get(key), str) or not incident[key].strip()]  # fmt: skip
    when = incident.get("incident_time")
    if isinstance(when, str) and when.strip():
        try:
            moment = datetime.fromisoformat(when.strip())
        except ValueError:
            moment = None
        if moment is None or moment.utcoffset() is None or moment.utcoffset().total_seconds():
            problems.append("incident_time is no ISO 8601 time in UTC (+00:00 or Z)")
        elif moment > datetime.now(UTC):
            problems.append("incident_time is later than now")
    for key in ("cause", "evidence"):
        if isinstance(incident.get(key), str) and len(incident[key].splitlines()) > 1:
            problems.append(f"{key} is more than one line")
    return problems


def _not_used_up(run: Run, state: Assessment, entries: Entries, max_attempts: int) -> None:
    """A study-3 call with ``recovery`` over a run that is not used up under its config hash
    (module docstring, "Recovery rounds") raises ``CampaignError``, in a dry run too, with the
    same message: no round opens, and its incident would be recorded nowhere. A run with
    attempts left (a new config hash is a new run) takes them without ``recovery``; a done or
    diverged run holds its outcome, named as ``assess`` found it, with its directory."""
    held = [f"{outcome} ({where.name})" for outcome, where in (("done", state.done),
                                                               ("diverged", state.diverged))
            if where is not None]  # fmt: skip
    if held:
        raise CampaignError(f"{run.label}: no recovery round: the run is {' and '.join(held)} "
                            "under its config hash, an outcome that is never rerun; no recovery "
                            "round is written")  # fmt: skip
    left = attempts_left(run, state, entries, max_attempts)
    raise CampaignError(
        f"{run.label}: no recovery round: the run is not used up under its config hash ({left} "
        "attempt(s) left; a new config hash is a new run), so a round would record its incident "
        "nowhere. Run its attempts without --recovery: scripts/launch.py CAMPAIGN --only "
        f"{run.label} --max-attempts 1"
    )


def _no_cell_round_over_own_rounds(exhausted: Sequence[Run], entries: Entries) -> None:
    """A cell's round grants nothing to a label that had a round of its own (module docstring,
    "Recovery rounds"): a call that would open one over such a label is refused, naming it."""
    owned = {run.label: [x for x in label_rounds(entries, run) if x.own] for run in exhausted}
    named = [f"- {label}: {', '.join(map(str, rounds))}; {round_command(label)}"
             for label, rounds in owned.items() if rounds]  # fmt: skip
    if named:
        lines = "\n".join(named)
        raise CampaignError("A cell's recovery round grants nothing to a run that had a round of "
                            "its own; its further round is its own, at one attempt. No recovery "
                            "round is written (a settled end of an open start may be "
                            f"written first):\n{lines}")  # fmt: skip


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
    "The journal"): under an exclusive ``flock`` held from the tail's read to the last fsync,
    one ``os.write`` of the whole line (a short write raises ``OSError``), then the fsync of the
    log, of its folder ``launch/`` and of the runs root that holds it, on every append. A torn
    last line is named first, by its own ``_write`` under the same lock: the newline that ends
    it and its ``torn`` record, then the record. Returns the stamped record."""
    path = Path(path)
    for key in IDENTITY_KEYS:  # never a record its own reader would take for a torn line
        if key in record and not isinstance(record[key], str):
            raise ValueError(f"{path}: a journal record's {key} must be a string")
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat(timespec="seconds")
    stamped = {"time": now, **record}
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)  # one writer at a time, until the descriptor is closed
        torn = _torn_tail(path)
        if torn is not None:  # the newline that ends the torn line, then the record naming it
            _write(fd, path, b"\n" + _line({"time": now, "event": "torn", **torn}))
        _write(fd, path, _line(stamped))
        os.fsync(fd)
        for folder in (path.parent, path.parent.parent):  # launch/ and the runs root: whoever
            _fsync_folder(folder)  # created them, their entries are durable before we return
    finally:
        os.close(fd)
    return stamped


def _fsync_folder(folder: Path) -> None:
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
