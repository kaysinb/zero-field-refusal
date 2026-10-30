"""Study 3's harm labels: the labeling ledger, the two points where harm leaves sealed storage, and
the one reader of a run's harm and of its items.

docs/study3-preregistration.md: "Judge and audit" (the denominator 521; labeling follows study
2's receipts, up to three rounds per run; every labeling batch includes the 40 control items; a
changed control label relabels the batch, a repeated change is a technical mark of the judge and
labeling stops until it is diagnosed; every label is sealed from the first run, and harm leaves
sealed storage only at validation's spread step and, after the cohorts and K*_h are frozen, at the
final analysis), "Sample size from validation", "Seeds and order" (order steps 6, 7, 9 and 10)
and "Voiding and supersession" (an embargo breach).

- ``run_harm`` and ``harm_many``: a run's harm at the endpoint (``RunHarm``: ``harmful`` and
  ``missing`` of ``n`` = 521), the only way an analysis reads harm. ``harm_many`` gates a whole
  cohort with one reading of the study's state (``progress`` reads every record of the study):
  a cohort's gate is one reading, never one per run. ``run_harm`` is ``harm_many`` of one run.
- ``run_items``: the per-item labels of a run (``ItemLabel``), under exactly ``run_harm``'s gates
  and errors, for the audit.
- ``open_point``: the run ids a point opens, or ``SealedPoint``.
- ``read_ledger``, ``append_event``, ``torn_tail``, ``resolve_torn`` and ``ledger_lock``: the
  labeling ledger (``RUNS_DIR/study3/labeling/``, ``ledger_dir``).
- ``ledger_runs``, ``labeling_problems`` and ``diagnoses``: label-free readings of the ledger
  (ids, states and the diagnosis records) for the report.

Importing loads torch through study 2's judge helpers; nothing GPU-side runs.

**The points** (nothing sealed is read to decide them).
- ``spread`` opens when candidate N's validation decision is complete: the study's state
  (``study3_progress.progress`` under this checkout's GPU code, θ₀ at its default place) is at
  the stage ``choice`` of candidate N, its step the ``n-c<N>`` record, with no problem. It opens
  the ids of that step's cohort (``fields.cohort.run_ids``) and closes once the n record is
  written: the state then moves past ``choice``. A committed validation fixation is not enough:
  it comes before validation.
- ``final`` opens through the freeze gate only; it is closed here.

**The order of checks** of ``harm_many`` and ``run_items`` (``_gated``, shared, so that their
gates and errors are the same; steps 1 and 2 open nothing sealed):
1. ``point`` is one of ``POINTS`` (``ValueError``), and the point is opened once for the list
   (``SealedPoint`` otherwise).
2. Each run folder is a plain child of ``RUNS_DIR/study3`` that resolves there, not a reserved
   folder (``ValueError``); a run outside the point's run set raises ``SealedPoint`` with its id.
3. The ledger is read: a run that is a technical labeling problem (``Ledger.problem``) raises
   ``LabelingProblem``; a run with no open generation, or whose generation no accepting
   ``controls`` line covers, or whose judging receipt's sha256 now is not the covered one,
   raises ``LabelsPending``.
4. The run is read as study 2 reads it (``study2_labels``' checks: the recorded prompts, the
   responses, the receipt of this judge for exactly the run's items, its label file by sha256,
   every record of the label file), at the ``ENDPOINT`` condition. One rule differs: missing labels
   are pending only while the receipt's rounds are below the *generation's cap*, not study 2's
   three. A generation opened with c < 3 rounds left finishes after c rounds; under study 2's
   rule it would stay pending for ever.
Missing labels (unparsed or absent) count only under that accepted receipt; ``missing`` is never
the 521 of a run whose rounds are spent: that run is a ``LabelingProblem``.

**The ledger** is an append-only journal, ``labeling/journal.jsonl``: one JSON object per line,
each ending in ``\\n``, written by ``append_event`` with ``O_APPEND`` and synced with its folder.
Every line has a closed schema (``event``, ``time`` and the event's fields, ``FIELDS``) and is
folded on every read (``read_ledger``) and before every append, so a line the fold refuses is
never written. The batch and generation identities, the rounds spent and the control decisions
live here, outside every folder a relabel moves. The fold:
- **a torn tail** (a last line with no ``\\n``, blanks alone included) refuses every reading
  and every append (``LedgerError``) until ``resolve_torn`` repairs it: the tail is ended with
  ``\\n`` and a ``torn`` line names its offset, length, sha256 and kind. The voided line's
  decision was never acted on (every action follows its line, and the process that tore it died
  while writing it). A crash between the ``\\n`` and the ``torn`` line leaves the tail a whole
  line at the end that does not parse and that no ``torn`` line names: ``resolve_torn`` voids it
  the same way;
- **a line that does not parse or fit its schema** raises ``LedgerError`` naming its number, never
  its bytes, unless a later ``torn`` line names it;
- **generations** run 1, 2, … per run, each opened with ``cap`` = the rounds left (three less
  the rounds its closed generations used); a generation while the run's previous one is open, a
  ``superseded`` line for a generation that is not the open one, a cap other than the rounds
  left, a ``covers`` entry for a generation that is not open or with a receipt that its own
  attempt's ``labeled`` line (same batch, k and attempt) does not record, a second ``controls``
  line for one (batch, k), and a second ``opened`` while a batch is open are errors;
- **an opened batch** takes ``generation`` and ``labeled`` lines and ``covers`` entries of the
  runs its ``opened`` line names only, and a ``mark`` of its own candidate only;
- **a batch** is ``accepted`` (an accepting ``controls`` line), else ``mark``, else ``open``; its
  ``k`` is 1, plus 1 after each changed ``controls`` line, counted in ``changes``;
- **diagnoses**: a ``diagnosed`` line follows its published record (its sha256), resolves its
  mark only when its cause is ``infrastructure``, and changes no round count; a ``judge`` cause
  leaves the mark unresolved for good. A record published with no ``diagnosed`` line is
  ``stranded``, to be completed by a re-run of the diagnosis command.

**What opens the sealed folders here.** The named sealed openers are ``read_ledger`` (the
ledger's journal and diagnosis records), ``_receipt_sha256`` (a run's judging receipt, hashed)
and ``_read_run`` (a run's labels, through study 2's checks); the ledger's writers
(``append_event``, ``resolve_torn``, ``ledger_lock``) and ``torn_tail`` open its journal and lock
alone. ``ledger_runs``, ``labeling_problems`` and ``diagnoses`` read the ledger through
``read_ledger``: ids, states and the diagnosis records, never a label, a receipt or a run file.

**Rulings** where the registration leaves a detail open:
- *The two points.* "Validation's spread step" is the choice step of the candidate whose
  validation passed, until its n record is written; validation labeling opens at the same point.
  "The final analysis" opens through the freeze gate and covers the candidate's validation runs
  as well as its confirmation runs, since validation's harm is reported after it.
- *Acceptance.* A run's labels count only when its current generation is covered by an accepting
  ``controls`` line (the 40 controls labeled unchanged in the same process attempt, under the same
  judge session, as the run's receipt), at the receipt's sha256 now.
- *Exhaustion.* A run whose last generation was rejected and that has no labeling round left is
  a technical labeling problem, the judge's mark on the ``judge`` route: it raises
  ``LabelingProblem``, never 521 missing labels. A generation that used its cap without writing a
  label file is rejected so too (superseded ``unverified``, its rounds used its cap): the
  registered lifetime rounds are spent, and nothing releases one.
- *The rounds.* Each run has three labeling rounds in its lifetime; interrupted rounds count, a
  relabel spends from them, and nothing releases or adds one, a diagnosis included.
- ``RunHarm`` is ``kw_only``: its ``n`` has a default (521) before fields without one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, NoReturn

from omegaconf import DictConfig, OmegaConf

from ..config import load_config
from ..judge import IDENTITY_FIELDS, judge_identity
from ..run_judging import (
    LABEL_ROUNDS,
    _locked,
    expected_items,
    is_scheduled,
    prompt_ids,
    receipt_path,
    scheduled_conditions,
)
from . import study2_labels
from .evidence import label_item_id, label_path
from .study2_label_cli import STATUSES as STUDY2_STATUSES
from .study2_label_cli import harmful_prompts
from .study3_records import EXPERIMENT, LABELING, RESERVED, plain_name, torn_kind

JUDGE = "wildguard"
HARMFUL_PROMPTS = 521
ENDPOINT = "step-25"
ROUNDS = LABEL_ROUNDS  # 3: "up to three rounds per run"
JUDGE_CONFIG = f"judge/{JUDGE}"  # study3_versions.JUDGE_CONFIG: the frozen judge
POINTS = ("spread", "final")
EVENTS = ("opened", "generation", "labeled", "controls", "superseded", "interrupted", "mark",
          "diagnosed")  # fmt: skip
CAUSES = ("infrastructure", "judge")
ITEM_LABELS = ("harmful", "not_harmful", "missing")
STEPS = ("validation", "confirmation")
GENERATION_WHY = (None, "control_change", "unverified", "after_diagnosis")
REASONS = ("control_change", "judge_mark", "unverified")  # why a generation is superseded
MARKS = ("second_change", "exhausted")
JOURNAL, LOCK = "journal.jsonl", "lock"
DIAGNOSIS_KIND = "study3-judge-diagnosis"
DIAGNOSIS_FIELDS = ("kind", "batch", "written_at", "cause", "diagnosis", "evidence")
DIAGNOSIS_LENGTH = 200  # the researcher's one line, at most
# The fields of each line, beside ``event`` and ``time`` (``covers`` only on an accepting line)
FIELDS = {
    "opened": ("batch", "step", "candidate", "runs", "controls", "fixation", "judge", "code"),
    "generation": ("batch", "run", "generation", "cap", "why"),
    "labeled": ("batch", "k", "attempt", "runs"),
    "controls": ("batch", "k", "attempt", "accepted", "changed", "parsed_changed", "cache"),
    "superseded": ("batch", "k", "run", "generation", "reason", "rounds_used", "target", "moved"),
    "interrupted": ("batch", "k", "why"),
    "mark": ("batch", "candidate", "why", "changed", "runs"),
    "diagnosed": ("batch", "cause", "record"),
    "torn": ("offset", "length", "sha256", "kind"),
}
# What a label of the judge counts as at the endpoint
OUTCOME = {"harmful_compliance": "harmful", "refusal": "not_harmful",
           "safe_non_refusal": "not_harmful"}  # fmt: skip
_BATCH = re.compile(r"b\d{4,}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTROL = re.compile(r"c[0-9a-f]{12}")
_RECORD = re.compile(r"diagnosis-(b\d{4,})\.json")


class LabelsPending(study2_labels.LabelsPending):
    """The run's labels do not count yet: not labeled, or not accepted with its controls."""


class LabelingProblem(ValueError):
    """A technical labeling problem: a run exhausted after a rejected generation, or one whose
    generation was superseded under the judge's unresolved mark."""


class SealedPoint(PermissionError):
    """The point is closed, or the run is outside the run set it opens: nothing sealed is read."""


class LedgerError(ValueError):
    """The ledger cannot be read or appended to: a torn tail, a line that does not parse or fit
    its schema, or a line the fold refuses."""


@dataclass(frozen=True, kw_only=True)
class RunHarm:
    """A run's harm at the endpoint: ``harmful`` and ``missing`` (unparsed or absent labels under
    its accepted receipt) of ``n`` prompts, and the sha256 of that receipt."""

    n: int = HARMFUL_PROMPTS
    harmful: int
    missing: int
    receipt_sha256: str


@dataclass(frozen=True)
class ItemLabel:
    """One item at the endpoint: its position in the run's prompt ids, from 0, its prompt id and
    its label (``ITEM_LABELS``)."""

    index: int
    prompt_id: str
    label: str


@dataclass(frozen=True)
class Generation:
    """A labeling generation of a run: its number, the rounds it may use (``cap``), the batch
    that opened it and why; once superseded, why (``closed``, a reason of ``REASONS``) and the
    rounds it used."""

    run: str
    number: int
    cap: int
    batch: str
    why: str | None
    closed: str | None = None
    rounds_used: int | None = None


@dataclass(frozen=True)
class Ledger:
    """The ledger's fold (module docstring). ``batches``: per batch, its ``opened`` fields (when
    it was opened), ``state`` (open, accepted or mark), ``k``, ``changes`` and its lines
    (``labeled`` by (k, attempt), its ``controls`` lines as ``checks`` by k, ``superseded``,
    ``interrupted``, ``mark``, ``diagnosed``);
    ``covered``: (run, generation) → the receipt sha256 an accepting line covers;
    ``unresolved_marks``: mark batches with no diagnosis or a ``judge`` one; ``stranded``:
    published diagnosis records with no ``diagnosed`` line; ``closers``: (run, generation) → the
    batch that superseded it; ``published``: the diagnosis records by batch (path, bytes)."""

    batches: dict[str, dict[str, Any]]
    generations: dict[str, tuple[Generation, ...]]
    covered: dict[tuple[str, int], str]
    unresolved_marks: tuple[str, ...]
    stranded: tuple[str, ...]
    closers: dict[tuple[str, int], str] = field(default_factory=dict)
    published: dict[str, tuple[Path, bytes]] = field(default_factory=dict)

    def current(self, run: str) -> Generation | None:
        """The run's open generation; None when it has none."""
        found = self.generations.get(run, ())
        return found[-1] if found and found[-1].closed is None else None

    def spent(self, run: str) -> int:
        """The rounds the run's closed generations used: it never decreases."""
        return sum(g.rounds_used or 0 for g in self.generations.get(run, ()) if g.closed)

    def remaining(self, run: str) -> int:
        """``ROUNDS`` less ``spent``."""
        return max(0, ROUNDS - self.spent(run))

    def stopped(self, candidate: int) -> str | None:
        """The first unresolved mark of a batch of ``candidate``; None when labeling may go on."""
        return next((b for b in self.unresolved_marks
                     if self.batches[b]["mark"]["candidate"] == candidate), None)  # fmt: skip

    def unfinished(self) -> str | None:
        """The one opened batch still open."""
        return next((b for b, info in self.batches.items()
                     if info["opened"] and info["state"] == "open"), None)  # fmt: skip

    def problem(self, run: str) -> str | None:
        """Why the run is a technical labeling problem: its last generation was superseded, it has
        no open generation, and it has no round left (exhaustion: the judge's mark, ``judge``
        route) or its superseding batch's mark is unresolved. None otherwise."""
        found = self.generations.get(run, ())
        if not found or found[-1].closed is None:
            return None
        last = found[-1]
        if self.remaining(run) == 0:
            return (f"{run}: its {ROUNDS} labeling rounds are spent and its generation "
                    f"{last.number} was superseded ({last.closed}): a technical labeling "
                    "problem, the judge's mark")  # fmt: skip
        closer = self.closers.get((run, last.number))
        if closer in self.unresolved_marks:
            return (f"{run}: its generation {last.number} was superseded under the judge's "
                    f"technical mark of batch {closer}, unresolved")  # fmt: skip
        return None


@dataclass(frozen=True)
class Diagnosis:
    """A published diagnosis of the judge's mark, completed by its ``diagnosed`` line: the
    record's path and sha256, the batch, the cause (``CAUSES``), the researcher's one line and
    the evidence (``{path, sha256, root}``)."""

    path: Path
    sha256: str
    batch: str
    cause: str
    diagnosis: str
    evidence: tuple[dict[str, str], ...]


# ------------------------------------------------------------------ the ledger


def ledger_dir(runs_root: str | Path) -> Path:
    """The ledger's folder, ``RUNS_DIR/study3/labeling`` (made 0700 by its writers)."""
    return Path(runs_root) / EXPERIMENT / LABELING


def _made(runs_root: str | Path) -> Path:
    directory = ledger_dir(runs_root)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    return directory


@contextmanager
def ledger_lock(runs_root: str | Path) -> Iterator[bool]:
    """The ledger's lock (``labeling/lock``, an exclusive flock, without waiting): True while
    held, False when another process holds it. Every writer of the ledger holds it."""
    with _locked(_made(runs_root) / LOCK) as held:
        yield held


def read_ledger(runs_root: str | Path) -> Ledger:
    """The ledger's fold (module docstring); ``LedgerError`` for a torn tail, a bad line or a
    line the fold refuses. A named sealed opener: the journal and the diagnosis records."""
    directory = ledger_dir(runs_root)
    path = directory / JOURNAL
    data = path.read_bytes() if path.is_file() else b""
    return _fold(data, _published(directory), path)


def append_event(runs_root: str | Path, event: Mapping[str, Any]) -> None:
    """Append ``event`` (with its ``time``) to the journal, synced with its folder, once the fold
    takes it: a torn tail, a line that does not fit its schema and a line the fold refuses raise
    ``LedgerError`` and nothing is written. The caller holds ``ledger_lock``."""
    event = dict(event)
    if "time" in event or event.get("event") == "torn":
        raise ValueError("a line's time is the ledger's, and a torn line is resolve_torn's alone")
    line = {"event": event.pop("event", None), **event,
            "time": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    problems = _schema(line)
    if problems:
        raise LedgerError(f"a {line['event']!r} line the ledger does not take: {problems[0]}")
    directory = _made(runs_root)
    path = directory / JOURNAL
    data = path.read_bytes() if path.is_file() else b""
    if _lines(data)[1] is not None:  # the new line would end the torn one, blanks alone included
        raise _torn(path)
    encoded = json.dumps(line).encode() + b"\n"
    _fold(data + encoded, _published(directory), path)
    _append(path, encoded)


def torn_tail(runs_root: str | Path) -> dict[str, Any] | None:
    """The journal's torn tail, ``{offset, length, sha256, kind}`` (its event name when it can be
    read, else None), or None when the journal ends with a whole line."""
    path = ledger_dir(runs_root) / JOURNAL
    data = path.read_bytes() if path.is_file() else b""
    tail = _lines(data)[1]
    return None if tail is None else _named(*tail)


def resolve_torn(runs_root: str | Path) -> tuple[dict[str, Any], ...]:
    """Repair a torn journal, under the ledger's lock: end a torn tail with ``\\n`` (synced),
    then append a ``torn`` line (synced) for each line it voids: every whole line at the end that
    does not parse and that no ``torn`` line names (a crash between an earlier repair's ``\\n``
    and its ``torn`` line; walking back past the ``torn`` lines and the lines they void), then
    the tail. Returns each voided line's ``{offset, length, sha256, kind}``, in the journal's
    order; ``LedgerError`` when nothing is torn or the lock is held."""
    with ledger_lock(runs_root) as held:
        if not held:
            raise LedgerError("another process holds the labeling ledger's lock")
        path = ledger_dir(runs_root) / JOURNAL
        data = path.read_bytes() if path.is_file() else b""
        lines, tail = _lines(data)
        voided = [*_unparsed_end(lines), *([] if tail is None else [tail])]
        if not voided:
            raise LedgerError("the labeling ledger's journal is not torn: nothing to resolve")
        if tail is not None:
            _append(path, b"\n")
        named = tuple(_named(offset, raw) for offset, raw in voided)
        for entry in named:
            line = {"event": "torn", **entry,
                    "time": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
            _append(path, json.dumps(line).encode() + b"\n")
        return named


def _named(offset: int, raw: bytes) -> dict[str, Any]:
    """A voided line as its ``torn`` line names it: offset, length, sha256 and kind."""
    return {"offset": offset, "length": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "kind": torn_kind(raw)}  # fmt: skip


def _unparsed_end(lines: Sequence[tuple[int, bytes]]) -> list[tuple[int, bytes]]:
    """The whole lines at the journal's end that do not parse and that no ``torn`` line names, in
    order: walking back past the ``torn`` lines and the lines they void, up to the first other
    line that parses (or is empty)."""
    read, voids = _read(lines)
    found = []
    for number, offset, raw, event, problems in reversed(read):
        if _voided(voids, number, offset, raw) or (not problems and event["event"] == "torn"):
            continue
        if not raw or _parses(raw):
            break
        found.append((offset, raw))
    return found[::-1]


def _append(path: Path, data: bytes) -> None:
    handle = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(handle, view) :]
        os.fsync(handle)
    finally:
        os.close(handle)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _published(directory: Path) -> dict[str, tuple[Path, bytes]]:
    """The published diagnosis records (``diagnosis-<batch>.json``) by batch, with their bytes."""
    found = {}
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            match = _RECORD.fullmatch(path.name)
            if match and path.is_file():
                found[match.group(1)] = (path, path.read_bytes())
    return found


def _lines(data: bytes) -> tuple[list[tuple[int, bytes]], tuple[int, bytes] | None]:
    """The journal's whole lines, each (offset, bytes without its ``\\n``), and its torn tail."""
    lines, offset = [], 0
    while offset < len(data):
        end = data.find(b"\n", offset)
        if end < 0:
            return lines, (offset, data[offset:])
        lines.append((offset, data[offset:end]))
        offset = end + 1
    return lines, None


def _torn(path: Path) -> LedgerError:
    return LedgerError(f"{path}: its last line is torn (a crash while it was written): nothing is "
                       "read or appended until `python -m "
                       "zero_field_refusal.analysis.study3_label_cli resolve-torn` repairs "
                       "it")  # fmt: skip


def _decoded(raw: bytes) -> Any:
    """A journal line's JSON value; ``ValueError`` when it does not parse."""
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_no_constant)


def _parses(raw: bytes) -> bool:
    try:
        _decoded(raw)
    except ValueError:
        return False
    return True


def _read(
    lines: Sequence[tuple[int, bytes]],
) -> tuple[list[tuple[int, int, bytes, Any, list[str]]], dict[tuple[int, int, str], int]]:
    """Each whole line as (number, offset, bytes, event, its schema problems), and the lines the
    ``torn`` lines name (offset, length, sha256) → the first ``torn`` line naming each."""
    read: list[tuple[int, int, bytes, Any, list[str]]] = []
    for number, (offset, raw) in enumerate(lines, 1):
        try:
            event = _decoded(raw)
        except ValueError:
            event, problems = None, ["it is no JSON object"]
        else:
            problems = _schema(event)
        read.append((number, offset, raw, event, problems))
    voids: dict[tuple[int, int, str], int] = {}
    for number, _, _, event, problems in read:
        if not problems and event["event"] == "torn":
            voids.setdefault((event["offset"], event["length"], event["sha256"]), number)
    return read, voids


def _voided(voids: Mapping[tuple[int, int, str], int], number: int, offset: int,
            raw: bytes) -> bool:  # fmt: skip
    """Whether a later ``torn`` line names the line: a torn line voids an earlier line only."""
    return voids.get((offset, len(raw), hashlib.sha256(raw).hexdigest()), 0) > number


def _fold(data: bytes, published: Mapping[str, tuple[Path, bytes]], path: Path) -> Ledger:
    lines, tail = _lines(data)
    if tail is not None:
        raise _torn(path)
    read, voids = _read(lines)
    fold = _Fold(published, path)
    for number, offset, raw, event, problems in read:
        if _voided(voids, number, offset, raw):
            continue
        if problems:
            raise LedgerError(f"{path}, line {number}: {problems[0]}")
        fold.line(number, event)
    return fold.ledger()


class _Fold:
    """The ledger's state, line by line (module docstring, "The ledger")."""

    def __init__(self, published: Mapping[str, tuple[Path, bytes]], path: Path) -> None:
        self.published, self.path = dict(published), path
        self.batches: dict[str, dict[str, Any]] = {}
        self.generations: dict[str, list[Generation]] = {}
        self.covered: dict[tuple[str, int], str] = {}
        self.closers: dict[tuple[str, int], str] = {}

    def fail(self, number: int, why: str) -> NoReturn:
        raise LedgerError(f"{self.path}, line {number}: {why}")

    def batch(self, name: str) -> dict[str, Any]:
        return self.batches.setdefault(name, {
            "opened": False, "state": "open", "k": 1, "changes": 0, "labeled": {},
            "checks": {}, "superseded": [], "interrupted": [], "mark": None,
            "diagnosed": None})  # fmt: skip

    def remaining(self, run: str) -> int:
        return ROUNDS - sum(g.rounds_used or 0 for g in self.generations.get(run, []) if g.closed)

    def current(self, run: str) -> Generation | None:
        found = self.generations.get(run, [])
        return found[-1] if found and found[-1].closed is None else None

    def member(self, number: int, kind: str, e: Mapping[str, Any], info: dict[str, Any],
               run: str) -> None:  # fmt: skip
        """An opened batch's line names only the runs it was opened with."""
        if info["opened"] and run not in info["runs"]:
            self.fail(number, f"a {kind} line of batch {e['batch']} names {run}, a run it was not "
                      "opened with")  # fmt: skip

    def line(self, number: int, e: Mapping[str, Any]) -> None:
        kind = e["event"]
        if kind == "torn":
            return
        info = self.batch(e["batch"])
        if kind in ("opened", "generation", "labeled", "controls", "superseded", "interrupted",
                    "mark") and info["state"] != "open":  # fmt: skip
            self.fail(number, f"a {kind} line of batch {e['batch']}, which is {info['state']}"
                      + (" (accepted)" if info["state"] == "accepted" else ""))  # fmt: skip
        getattr(self, f"_{kind}")(number, e, info)

    def _opened(self, number: int, e: Mapping[str, Any], info: dict[str, Any]) -> None:
        if info["opened"]:
            self.fail(number, f"batch {e['batch']} is opened twice")
        busy = next((b for b, i in self.batches.items() if i["opened"] and i["state"] == "open"),
                    None)  # fmt: skip
        if busy is not None:
            self.fail(number, f"batch {e['batch']} is opened while batch {busy} is open")
        info.update({key: e[key] for key in FIELDS["opened"]}, opened=True)

    def _generation(self, number: int, e: Mapping[str, Any], info: dict[str, Any]) -> None:
        self.member(number, "generation", e, info, e["run"])
        run, found = e["run"], self.generations.setdefault(e["run"], [])
        if found and found[-1].closed is None:
            self.fail(number, f"a generation of {run} while its generation {found[-1].number} is "
                      "open")  # fmt: skip
        if e["generation"] != len(found) + 1:
            self.fail(number, f"generation {e['generation']} of {run} is not its next, "
                      f"{len(found) + 1}")  # fmt: skip
        if e["cap"] != self.remaining(run) or e["cap"] < 1:
            self.fail(number, f"generation {e['generation']} of {run} has the cap "
                      f"{e['cap']}, not the {self.remaining(run)} rounds it has left (at least "
                      "one)")  # fmt: skip
        found.append(Generation(run, e["generation"], e["cap"], e["batch"], e["why"]))

    def _labeled(self, number: int, e: Mapping[str, Any], info: dict[str, Any]) -> None:
        k, attempt = e["k"], e["attempt"]
        for entry in e["runs"]:
            self.member(number, "labeled", e, info, entry["run"])
        if k != info["k"]:
            self.fail(number, f"a labeled line of batch {e['batch']} at k = {k}, not its "
                      f"k = {info['k']}")  # fmt: skip
        made = sum(1 for (kk, _) in info["labeled"] if kk == k)
        if attempt != made + 1:
            self.fail(number, f"attempt {attempt} of batch {e['batch']} at k = {k} is not its "
                      f"next, {made + 1}")  # fmt: skip
        info["labeled"][(k, attempt)] = e

    def _controls(self, number: int, e: Mapping[str, Any], info: dict[str, Any]) -> None:
        k, attempt, batch = e["k"], e["attempt"], e["batch"]
        if k in info["checks"] or k < info["k"]:
            self.fail(number, f"a second controls line of batch {batch} at k = {k}")
        if k != info["k"]:
            self.fail(number, f"a controls line of batch {batch} at k = {k}, not its "
                      f"k = {info['k']}")  # fmt: skip
        labeled = info["labeled"].get((k, attempt))
        if labeled is None:
            self.fail(number, f"a controls line of batch {batch} by attempt {attempt}, which has "
                      "no labeled line")  # fmt: skip
        receipts = {(r["run"], r["receipt_sha256"]) for r in labeled["runs"]}
        for cover in e.get("covers", ()):
            run, wanted = cover["run"], cover["generation"]
            self.member(number, "controls", e, info, run)
            current = self.current(run)
            if current is None or current.number != wanted:
                self.fail(number, f"covers generation {wanted} of {run}, not its open "
                          f"generation {current.number if current else None}")  # fmt: skip
            if (run, cover["receipt_sha256"]) not in receipts:
                self.fail(number, f"covers {run}'s receipt, which the labeled line of attempt "
                          f"{attempt} (batch {batch}, k = {k}) does not record")  # fmt: skip
        info["checks"][k] = e
        if e["accepted"]:
            for cover in e["covers"]:
                self.covered[(cover["run"], cover["generation"])] = cover["receipt_sha256"]
            info["state"] = "accepted"
        else:
            info["changes"] += 1
            info["k"] += 1

    def _superseded(self, number: int, e: Mapping[str, Any], info: dict[str, Any]) -> None:
        run, wanted = e["run"], e["generation"]
        current = self.current(run)
        if current is None or current.number != wanted:
            self.fail(number, f"superseded names generation {wanted} of {run}, not its open "
                      f"generation {current.number if current else None}")  # fmt: skip
        if e["rounds_used"] > current.cap:
            self.fail(number, f"generation {wanted} of {run} used {e['rounds_used']} rounds, "
                      f"beyond its cap {current.cap}")  # fmt: skip
        self.generations[run][-1] = Generation(
            run, wanted, current.cap, current.batch, current.why, e["reason"], e["rounds_used"]
        )
        self.closers[(run, wanted)] = e["batch"]
        info["superseded"].append(e)

    def _interrupted(self, number: int, e: Mapping[str, Any], info: dict[str, Any]) -> None:
        info["interrupted"].append(e)

    def _mark(self, number: int, e: Mapping[str, Any], info: dict[str, Any]) -> None:
        if info["opened"] and e["candidate"] != info["candidate"]:
            self.fail(number, f"a mark of batch {e['batch']} names candidate {e['candidate']}, "
                      f"not the candidate {info['candidate']} it was opened with")  # fmt: skip
        info["mark"] = e
        info["state"] = "mark"

    def _diagnosed(self, number: int, e: Mapping[str, Any], info: dict[str, Any]) -> None:
        batch = e["batch"]
        if info["diagnosed"] is not None:
            self.fail(number, f"a second diagnosed line of batch {batch}")
        if info["mark"] is None:
            self.fail(number, f"a diagnosis of batch {batch}, which has no mark")
        if e["cause"] == "infrastructure":
            if info["mark"]["why"] == "exhausted":
                self.fail(number, f"an infrastructure diagnosis of batch {batch}, whose mark is "
                          "exhausted: an exhausted run stays the judge's labeling "
                          "problem")  # fmt: skip
            if any(c["parsed_changed"] for c in info["checks"].values()):
                self.fail(number, f"an infrastructure diagnosis of batch {batch}, where a control "
                          "label changed to a parsed label: a control mismatch is never "
                          "infrastructure")  # fmt: skip
        published = self.published.get(batch)
        if published is None or hashlib.sha256(published[1]).hexdigest() != e["record"]["sha256"]:
            self.fail(number, f"the diagnosis record of batch {batch} is not published with the "
                      "sha256 its line names")  # fmt: skip
        try:
            record = json.loads(published[1])
        except ValueError:
            record = None
        if not (isinstance(record, dict) and record.get("batch") == batch
                and record.get("cause") == e["cause"]):  # fmt: skip
            self.fail(number, f"the diagnosis record of batch {batch} is not of its batch and "
                      "cause")  # fmt: skip
        info["diagnosed"] = e

    def ledger(self) -> Ledger:
        unresolved = tuple(b for b, info in self.batches.items()
                           if info["mark"] is not None and not _resolved(info))  # fmt: skip
        stranded = tuple(b for b in _ordered(self.published)
                         if self.batches.get(b, {}).get("diagnosed") is None)  # fmt: skip
        return Ledger(batches=self.batches,
                      generations={r: tuple(g) for r, g in self.generations.items()},
                      covered=dict(self.covered), unresolved_marks=unresolved, stranded=stranded,
                      closers=dict(self.closers), published=dict(self.published))  # fmt: skip


def _resolved(info: Mapping[str, Any]) -> bool:
    """Whether a mark batch's diagnosis resolves its mark: an ``infrastructure`` cause alone."""
    return info["diagnosed"] is not None and info["diagnosed"]["cause"] == "infrastructure"


def _ordered(batches) -> list[str]:
    """Batch ids in their order (``b0001``, …, ``b10000``)."""
    return sorted(batches, key=lambda b: (len(b), b))


# ------------------------------------------------------------------ the lines' schemas


def _schema(e: Any) -> list[str]:
    """Every problem of a journal line, by field name and never by value."""
    if not isinstance(e, dict):
        return ["it is no JSON object"]
    kind = e.get("event")
    if kind not in FIELDS:
        return ["its event is none of the ledger's"]
    wanted = ("event", *FIELDS[kind], "time")
    if kind == "controls" and e.get("accepted") is True:
        wanted = (*wanted, "covers")
    missing = [key for key in wanted if key not in e]
    if missing:
        return [f"a {kind} line without `{missing[0]}`"]
    if set(e) - set(wanted):
        return [f"a {kind} line with a field its schema does not name"]
    problems = [] if isinstance(e["time"], str) else ["`time` is no text"]
    check = _Check(problems)
    if kind != "torn":
        check.batch("batch", e["batch"])
    getattr(check, kind)(e)
    return problems


class _Check:
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems

    def bad(self, where: str, what: str) -> None:
        self.problems.append(f"`{where}` is not {what}")

    def int(self, where: str, value: Any, low: int, high: int | None = None) -> bool:
        ok = type(value) is int and value >= low and (high is None or value <= high)
        if not ok:
            self.bad(where, f"an integer from {low}" + (f" to {high}" if high else ""))
        return ok

    def batch(self, where: str, value: Any) -> None:
        if not (isinstance(value, str) and _BATCH.fullmatch(value)):
            self.bad(where, "a batch id (b and four digits or more)")

    def sha(self, where: str, value: Any, null: bool = False) -> None:
        if not ((null and value is None) or (isinstance(value, str) and _SHA256.fullmatch(value))):
            self.bad(where, "a sha256" + (" or null" if null else ""))

    def run(self, where: str, value: Any) -> None:
        if not (plain_name(value) and value not in RESERVED):
            self.bad(where, "a run id (a plain name)")

    def runs(self, where: str, value: Any, empty: bool = True) -> None:
        if not (isinstance(value, list) and (empty or value)):
            self.bad(where, "a list of run ids")
            return
        for run in value:
            self.run(where, run)
        if len(set(map(str, value))) != len(value):
            self.bad(where, "a list of distinct run ids")

    def ref(self, where: str, value: Any) -> None:
        if not (isinstance(value, dict) and set(value) == {"path", "sha256"}
                and isinstance(value["path"], str) and value["path"]):  # fmt: skip
            self.bad(where, "a reference {path, sha256}")
            return
        self.sha(f"{where}.sha256", value["sha256"])

    def controls_ids(self, where: str, value: Any, null: bool = False) -> None:
        if null and value is None:
            return
        ids = isinstance(value, list) and all(isinstance(i, str) and _CONTROL.fullmatch(i)
                                              for i in value)  # fmt: skip
        if not (ids and value == sorted(set(value))):
            self.bad(where, "a sorted list of control ids, each once" + (" or null" * null))

    def k_of(self, e: Mapping[str, Any]) -> None:
        self.int("k", e["k"], 1)

    # -- the events

    def opened(self, e: Mapping[str, Any]) -> None:
        if e["step"] not in STEPS:
            self.bad("step", f"one of {list(STEPS)}")
        self.int("candidate", e["candidate"], 1, 2)
        self.runs("runs", e["runs"], empty=False)
        self.ref("controls", e["controls"])
        self.ref("fixation", e["fixation"])
        judge = e["judge"]
        if not (isinstance(judge, dict) and set(judge) == {"config", *IDENTITY_FIELDS}
                and all(v is None or isinstance(v, str) for v in judge.values())):  # fmt: skip
            self.bad("judge", "the judge's config and identity")
        code = e["code"]
        if not (isinstance(code, dict) and set(code) == {"commit", "dirty"}
                and (code["commit"] is None or isinstance(code["commit"], str))
                and (code["dirty"] is None or isinstance(code["dirty"], bool))):  # fmt: skip
            self.bad("code", "the checkout's {commit, dirty}")

    def generation(self, e: Mapping[str, Any]) -> None:
        self.run("run", e["run"])
        self.int("generation", e["generation"], 1)
        self.int("cap", e["cap"], 1, ROUNDS)
        if e["why"] not in GENERATION_WHY:
            self.bad("why", f"one of {list(GENERATION_WHY)}")

    def labeled(self, e: Mapping[str, Any]) -> None:
        self.k_of(e)
        self.int("attempt", e["attempt"], 1)
        runs = e["runs"]
        if not isinstance(runs, list):
            self.bad("runs", "a list of run statuses")
            return
        for entry in runs:
            if not (isinstance(entry, dict) and set(entry) == {"run", "status", "receipt_sha256"}):
                self.bad("runs", "a list of {run, status, receipt_sha256}")
                return
            self.run("runs.run", entry["run"])
            if entry["status"] not in STUDY2_STATUSES:
                self.bad("runs.status", "a labeling status")
            self.sha("runs.receipt_sha256", entry["receipt_sha256"], null=True)
        if len({str(entry["run"]) for entry in runs}) != len(runs):
            self.bad("runs", "a list of distinct runs")

    def controls(self, e: Mapping[str, Any]) -> None:
        self.k_of(e)
        self.int("attempt", e["attempt"], 1)
        if not isinstance(e["accepted"], bool):
            self.bad("accepted", "true or false")
        self.controls_ids("changed", e["changed"])
        self.controls_ids("parsed_changed", e["parsed_changed"])
        if isinstance(e["changed"], list) and isinstance(e["parsed_changed"], list):
            if not set(map(str, e["parsed_changed"])) <= set(map(str, e["changed"])):
                self.bad("parsed_changed", "a part of `changed`")
            if e["accepted"] is not (e["changed"] == []):
                self.bad("accepted", "true exactly when no control changed")
        self.ref("cache", e["cache"])
        if e["accepted"] is True:
            covers = e["covers"]
            keys = {"run", "generation", "receipt_sha256", "labels_sha256"}
            if not (isinstance(covers, list)
                    and all(isinstance(c, dict) and set(c) == keys for c in covers)):  # fmt: skip
                self.bad("covers", "a list of {run, generation, receipt_sha256, labels_sha256}")
                return
            for cover in covers:
                self.run("covers.run", cover["run"])
                self.int("covers.generation", cover["generation"], 1)
                self.sha("covers.receipt_sha256", cover["receipt_sha256"])
                self.sha("covers.labels_sha256", cover["labels_sha256"])
            if len({str(c["run"]) for c in covers}) != len(covers):
                self.bad("covers", "a list of distinct runs")

    def superseded(self, e: Mapping[str, Any]) -> None:
        self.k_of(e)
        self.run("run", e["run"])
        ok = self.int("generation", e["generation"], 1)
        if e["reason"] not in REASONS:
            self.bad("reason", f"one of {list(REASONS)}")
        self.int("rounds_used", e["rounds_used"], 0, ROUNDS)
        if ok and isinstance(e["batch"], str):
            target = f"judge-superseded/{e['batch']}/g{e['generation']}"
            target += "-unverified" if e["reason"] == "unverified" else ""
            if e["target"] != target:
                self.bad("target", "judge-superseded/<batch>/g<generation>[-unverified]")
        moved = e["moved"]
        if not (isinstance(moved, list) and all(
                isinstance(m, dict) and set(m) == {"path", "sha256"}
                and isinstance(m["path"], str) and m["path"].startswith("judge/")
                for m in moved)):  # fmt: skip
            self.bad("moved", "a list of {path, sha256} under judge/")
            return
        for entry in moved:
            self.sha("moved.sha256", entry["sha256"])

    def interrupted(self, e: Mapping[str, Any]) -> None:
        self.k_of(e)
        if not (isinstance(e["why"], str) and e["why"] and "\n" not in e["why"]):
            self.bad("why", "one line")

    def mark(self, e: Mapping[str, Any]) -> None:
        self.int("candidate", e["candidate"], 1, 2)
        if e["why"] not in MARKS:
            self.bad("why", f"one of {list(MARKS)}")
        changed = e["changed"]
        if not (isinstance(changed, dict) and set(changed) == {"1", "2"}):
            self.bad("changed", 'the changed control ids of k = "1" and "2"')
        else:
            for key in ("1", "2"):
                self.controls_ids(f"changed.{key}", changed[key], null=True)
        self.runs("runs", e["runs"], empty=e["why"] != "exhausted")
        if e["why"] == "second_change" and e["runs"] != []:
            self.bad("runs", "empty for a second change")

    def diagnosed(self, e: Mapping[str, Any]) -> None:
        if e["cause"] not in CAUSES:
            self.bad("cause", f"one of {list(CAUSES)}")
        self.ref("record", e["record"])
        if (isinstance(e["record"], dict) and isinstance(e["batch"], str)
                and e["record"].get("path") != record_path(e["batch"])):  # fmt: skip
            self.bad("record.path", "the batch's diagnosis record")

    def torn(self, e: Mapping[str, Any]) -> None:
        self.int("offset", e["offset"], 0)
        self.int("length", e["length"], 1)
        self.sha("sha256", e["sha256"])
        if not (e["kind"] is None or isinstance(e["kind"], str)):
            self.bad("kind", "an event name or null")


def record_path(batch: str) -> str:
    """The path of a batch's diagnosis record relative to RUNS_DIR."""
    return f"{EXPERIMENT}/{LABELING}/diagnosis-{batch}.json"


def diagnosis_problems(record: Any, batch: str | None = None) -> list[str]:
    """Every problem of a diagnosis record (closed: ``DIAGNOSIS_FIELDS``): ``kind``
    ``DIAGNOSIS_KIND``, ``batch`` (``batch`` when given), ``written_at`` (ISO 8601 in UTC),
    ``cause`` (``CAUSES``), ``diagnosis`` (one non-empty line, at most ``DIAGNOSIS_LENGTH``
    characters) and ``evidence`` (``{path, sha256, root}``, a plain relative path under
    ``root`` ``runs`` or ``repo``; at least one for ``infrastructure``, its external evidence)."""
    if not isinstance(record, dict):
        return ["the diagnosis record is no JSON object"]
    missing = [key for key in DIAGNOSIS_FIELDS if key not in record]
    problems = [f"the diagnosis record has no `{key}`" for key in missing]
    problems += [f"`{key}` is not a field of the diagnosis record" for key in record
                 if key not in DIAGNOSIS_FIELDS]  # fmt: skip
    if missing:
        return problems
    if record["kind"] != DIAGNOSIS_KIND:
        problems.append(f"`kind` is not {DIAGNOSIS_KIND!r}")
    if not (isinstance(record["batch"], str) and _BATCH.fullmatch(record["batch"])) or (
            batch is not None and record["batch"] != batch):  # fmt: skip
        problems.append(f"`batch` is not the batch {batch}" if batch else "`batch` is no batch")
    try:
        when = datetime.fromisoformat(record["written_at"])
        utc = when.tzinfo is not None and when.utcoffset() == timedelta(0)
    except (TypeError, ValueError):
        utc = False
    if not utc:
        problems.append("`written_at` is not an ISO 8601 time in UTC")
    if record["cause"] not in CAUSES:
        problems.append(f"`cause` is not one of {list(CAUSES)}")
    line = record["diagnosis"]
    if not (isinstance(line, str) and line.strip() and "\n" not in line and "\r" not in line
            and len(line) <= DIAGNOSIS_LENGTH):  # fmt: skip
        problems.append(f"`diagnosis` is not one non-empty line of at most {DIAGNOSIS_LENGTH} "
                        "characters")  # fmt: skip
    evidence = record["evidence"]
    keys = {"path", "sha256", "root"}
    if not (isinstance(evidence, list) and all(isinstance(e, dict) and set(e) == keys
                                               for e in evidence)):  # fmt: skip
        problems.append("`evidence` is not a list of {path, sha256, root}")
        return problems
    for entry in evidence:
        path = entry["path"]
        parts = path.split("/") if isinstance(path, str) else []
        if not (parts and not path.startswith("/") and "\\" not in path
                and all(part not in ("", ".", "..") for part in parts)):  # fmt: skip
            problems.append("`evidence.path` is not a plain relative path")
        if not (isinstance(entry["sha256"], str) and _SHA256.fullmatch(entry["sha256"])):
            problems.append("`evidence.sha256` is not a sha256")
        if entry["root"] not in ("runs", "repo"):
            problems.append("`evidence.root` is not runs or repo")
    if record["cause"] == "infrastructure" and not evidence:
        problems.append("an infrastructure diagnosis rests on the external evidence of the "
                        "interruption: `evidence` is empty")  # fmt: skip
    return problems


def _unique(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("a repeated key")
    return dict(pairs)


def _no_constant(name: str) -> NoReturn:
    raise ValueError(f"{name} is no JSON number")


# ------------------------------------------------------------------ the points


def _state(runs_root: str | Path, repo_root: str | Path) -> dict[str, Any]:
    """The study's state: the module's one call of ``progress``, under this checkout's GPU code,
    θ₀ at its default place."""
    from .. import code_identity
    from .study3_progress import progress

    expected = code_identity.gpu_code_identity(repo=Path(repo_root))["sha256"]
    return progress(runs_root, repo=repo_root, expected_code=expected)


def _spread(candidate: int, runs_root: str | Path,
            repo_root: str | Path) -> tuple[dict[str, Any], frozenset[str]]:  # fmt: skip
    """The study's state and the run ids the spread point opens; ``SealedPoint`` when closed."""
    state = _state(runs_root, repo_root)
    step = state.get("step") or {}
    cohort = ((step.get("fields") or {}).get("cohort") or {}).get("run_ids")
    if (state.get("stage") != "choice" or state.get("candidate") != candidate
            or step.get("record") != f"n-c{candidate}" or state.get("problems")
            or not isinstance(cohort, list)):  # fmt: skip
        raise SealedPoint(f"the spread point is closed (stage {state.get('stage')}, candidate "
                          f"{state.get('candidate')}): the spread opens at candidate "
                          f"{candidate}'s choice step and closes once n-c{candidate} is "
                          "written")  # fmt: skip
    return state, frozenset(cohort)


def open_point(point: str, *, candidate: int, runs_root: str | Path,
               repo_root: str | Path) -> frozenset[str]:  # fmt: skip
    """The run ids ``point`` opens (module docstring, "The points"); ``SealedPoint`` when it is
    closed, ``ValueError`` for another point."""
    if point not in POINTS:
        raise ValueError(f"the point is one of {list(POINTS)}, not {point!r}")
    if candidate not in (1, 2):
        raise ValueError(f"the candidate is 1 or 2, not {candidate!r}")
    if point == "final":
        raise SealedPoint("the final point opens through the freeze gate: it is closed until the "
                          "cohorts and K*_h are frozen")  # fmt: skip
    return _spread(candidate, runs_root, repo_root)[1]


# ------------------------------------------------------------------ the reader


def harm_many(run_dirs: Sequence[Path], *, point: Literal["spread", "final"], candidate: int,
              runs_root: Path, repo_root: Path) -> list[RunHarm]:  # fmt: skip
    """The harm of each run of ``run_dirs`` (module docstring, "The order of checks"), the point
    opened once for the list."""
    out = []
    for run_dir, generation, receipt in _gated(run_dirs, point, candidate, runs_root, repo_root):
        items = _read_run(run_dir, generation, receipt)
        out.append(RunHarm(n=HARMFUL_PROMPTS, harmful=sum(i.label == "harmful" for i in items),
                           missing=sum(i.label == "missing" for i in items),
                           receipt_sha256=receipt))  # fmt: skip
    return out


def run_harm(run_dir: Path, *, point: Literal["spread", "final"], candidate: int,
             runs_root: Path, repo_root: Path) -> RunHarm:  # fmt: skip
    """``harm_many`` of one run."""
    return harm_many([run_dir], point=point, candidate=candidate, runs_root=runs_root,
                     repo_root=repo_root)[0]  # fmt: skip


def run_items(run_dir: Path, *, point: Literal["spread", "final"], candidate: int,
              runs_root: Path, repo_root: Path) -> tuple[ItemLabel, ...]:  # fmt: skip
    """The run's items at the endpoint, in its prompt order, under ``run_harm``'s gates."""
    ((found, generation, receipt),) = _gated([run_dir], point, candidate, runs_root, repo_root)
    return _read_run(found, generation, receipt)


def _gated(run_dirs: Sequence[Path], point: str, candidate: int, runs_root: Path,
           repo_root: Path) -> list[tuple[Path, Generation, str]]:  # fmt: skip
    """Steps 1–3 of the order of checks: each run with its covered generation and receipt."""
    if point not in POINTS:
        raise ValueError(f"the point is one of {list(POINTS)}, not {point!r}")
    opened = open_point(point, candidate=candidate, runs_root=runs_root, repo_root=repo_root)
    experiment = Path(runs_root) / EXPERIMENT
    dirs = [Path(d) for d in run_dirs]
    for run_dir in dirs:
        if not _plain_child(run_dir, experiment):
            raise ValueError(f"{run_dir} is not a plain child of {experiment} that resolves there "
                             "(a run's folder)")  # fmt: skip
    for run_dir in dirs:
        if run_dir.name not in opened:
            raise SealedPoint(f"{run_dir.name} is outside the run set the {point} point opens: "
                              "nothing of it is read")  # fmt: skip
    ledger = read_ledger(runs_root)
    out = []
    for run_dir in dirs:
        run = run_dir.name
        why = ledger.problem(run)
        if why is not None:
            raise LabelingProblem(why)
        generation = ledger.current(run)
        if generation is None:
            raise LabelsPending(f"{run}: no labeling generation is open: not labeled yet")
        covered = ledger.covered.get((run, generation.number))
        if covered is None:
            raise LabelsPending(f"{run}: its generation {generation.number} is not accepted with "
                                "its batch's controls yet")  # fmt: skip
        if _receipt_sha256(run_dir) != covered:
            raise LabelsPending(f"{run}: its judging receipt is not the one its controls accepted "
                                "(its sha256 now differs)")  # fmt: skip
        out.append((run_dir, generation, covered))
    return out


def _plain_child(run_dir: Path, experiment: Path) -> bool:
    """Whether ``run_dir`` is a plain, unreserved child of ``experiment`` that resolves there
    (decided from names and links alone: nothing is opened)."""
    if not (plain_name(run_dir.name) and run_dir.name not in RESERVED):
        return False
    parent = os.path.normpath(os.path.abspath(run_dir.parent))
    if parent != os.path.normpath(os.path.abspath(experiment)):
        return False
    try:
        return run_dir.resolve() == experiment.resolve() / run_dir.name
    except (OSError, RuntimeError):
        return False


def _receipt_sha256(run_dir: Path) -> str | None:
    """The sha256 of the run's judging receipt now; None without one. A named sealed opener."""
    path = receipt_path(run_dir, JUDGE)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _read_run(run_dir: Path, generation: Generation, receipt: str) -> tuple[ItemLabel, ...]:
    """Study 2's strict reading of a run (``study2_labels``' helpers, composed as
    ``read_endpoint_labels`` composes them), at the receipt ``receipt`` (its sha256) and with the
    rounds rule read against the generation's cap (module docstring). A named sealed opener."""
    files: dict[str, str] = {}

    def read(path: Path) -> bytes:
        data = path.read_bytes()
        files[path.relative_to(run_dir).as_posix()] = hashlib.sha256(data).hexdigest()
        return data

    cfg = OmegaConf.create(read(run_dir / "config.yaml").decode())
    if not (isinstance(cfg, DictConfig) and is_scheduled(cfg)):
        raise ValueError(f"{run_dir}: not a study-3 training run")
    conditions = scheduled_conditions(run_dir, read)
    if ENDPOINT not in conditions:
        raise ValueError(f"{run_dir}: no evaluation at the endpoint {ENDPOINT}")
    prompts = harmful_prompts(cfg)
    ids = prompt_ids(cfg, prompts)
    if len(ids) != HARMFUL_PROMPTS:
        raise ValueError(f"{run_dir}: {len(ids)} harmful prompts, not {HARMFUL_PROMPTS}")
    study2_labels._check_recorded_prompts(run_dir, cfg, ids, read(run_dir / "meta.json"))
    texts = study2_labels._responses(run_dir, cfg, conditions, ids, read)
    items = {label_item_id(c, pid): (prompts[pid], text) for (c, pid), text in texts.items()}
    expected = expected_items(conditions, ids)
    identity = judge_identity(load_config(JUDGE_CONFIG))
    found = study2_labels._receipt(run_dir, JUDGE, identity, expected, read)
    if files.get(receipt_path(run_dir, JUDGE).relative_to(run_dir).as_posix()) != receipt:
        raise LabelsPending(f"{run_dir.name}: its judging receipt changed while it was read")
    path = label_path(run_dir, JUDGE)
    if not path.is_file():
        raise LabelsPending(f"{run_dir.name}: its label file is missing")
    data = read(path)
    if hashlib.sha256(data).hexdigest() != found["labels_sha256"]:
        raise LabelsPending(f"{run_dir.name}: its label file is not the one its receipt was "
                            "written for")  # fmt: skip
    labels, recorded = study2_labels._labels(path, data, identity, items)
    unrecorded = [item for item in expected if item not in recorded]
    if unrecorded:
        raise LabelsPending(f"{run_dir.name}: {len(unrecorded)} expected items have no record in "
                            "its label file")  # fmt: skip
    missing = sum(1 for item in expected if item not in labels)
    if found["missing"] != missing:
        raise LabelsPending(f"{run_dir.name}: its receipt's count of items without a label is "
                            "not its label file's")  # fmt: skip
    if missing and found["rounds"] < generation.cap:
        raise LabelsPending(f"{run_dir.name}: items without a label after {found['rounds']} "
                            f"labeling rounds of its generation's cap "
                            f"{generation.cap}")  # fmt: skip
    return tuple(ItemLabel(index, pid, OUTCOME.get(labels.get(label_item_id(ENDPOINT, pid)),
                                                   "missing"))
                 for index, pid in enumerate(ids))  # fmt: skip


# ------------------------------------------------------------------ label-free readings


def ledger_runs(runs_root: Path) -> frozenset[str]:
    """Every run id an ``opened`` or ``generation`` line names (ids only)."""
    ledger = read_ledger(runs_root)
    named = {run for info in ledger.batches.values() for run in info.get("runs") or ()}
    return frozenset(named | set(ledger.generations))


def labeling_problems(runs_root: Path) -> tuple[str, ...]:
    """The sorted run ids that are a technical labeling problem (``LabelingProblem``)."""
    ledger = read_ledger(runs_root)
    return tuple(sorted(run for run in ledger.generations if ledger.problem(run)))


def diagnoses(runs_root: Path) -> tuple[Diagnosis, ...]:
    """The diagnosis records, each completed by its ``diagnosed`` line, in batch order: each
    checked against its schema and the sha256 its line names. A stranded record (published, no
    line) is a ``LedgerError``: the diagnosis command's re-run completes it. Nothing but the
    ledger is read: no label, no run file, no receipt."""
    ledger = read_ledger(runs_root)
    if ledger.stranded:
        raise LedgerError(f"labeling/diagnosis-{ledger.stranded[0]}.json is published without "
                          "its diagnosed line: complete it with the `diagnose` command's "
                          "re-run")  # fmt: skip
    out = []
    for batch in _ordered(ledger.published):
        path, data = ledger.published[batch]
        record = json.loads(data)
        problems = diagnosis_problems(record, batch)
        if problems:
            raise LedgerError(f"{path}: {problems[0]}")
        out.append(Diagnosis(path=path, sha256=hashlib.sha256(data).hexdigest(), batch=batch,
                             cause=record["cause"], diagnosis=record["diagnosis"],
                             evidence=tuple(dict(e) for e in record["evidence"])))  # fmt: skip
    return tuple(out)
