"""Study 3's run records as endpoints, each bound to the cell its campaign issued, read from the
numbers alone.

docs/study3-preregistration.md: "Protocol" (B = 20, exactly 25 updates, the stops before step 25),
"Questions and estimands" (usable runs), "Utility" (the losses against θ₀), "Failure routes",
"Seeds and order", "Voiding and supersession" and amendment 1 (what K-only runs measure). Pure
functions of what lies under RUNS_DIR: the launch journal (``launch/<campaign>.jsonl``), the run
directories of the experiment ``study3`` (both configs, regular and K-only) and the
qualification's restart and supersession records (``qualification/c<N>/``).

- ``read_endpoints``: every attempt of a study-3 run (and every study-3 run directory), each
  bound to its issued cell, with its state and step-25 endpoint, and the refusals by name
  (``problems``). Nothing may be computed from records with a refusal: they are ambiguous.
- ``read_holds``: every held launcher run, by its label under any config hash, with the records
  that hold it and their sha256 (what a closure record cites).
- ``SEEDS`` and ``candidate_of``: the registered seeds per role and candidate.
- ``step_key``: the key of a step's prescribed cells, ``c<candidate>/<step>[/r<round>]``.
- ``read_theta0``: θ₀'s accuracies, the utility losses' baseline (study 2's θ₀ evaluation).

**The sealing boundary.** Of a run directory only ``config.yaml``, ``meta.json``,
``terminal.json``, ``summary.json``, ``evaluations.jsonl``, ``metrics.jsonl`` and
``monitor.jsonl`` are read; nothing under ``responses/`` or ``observations/`` is ever opened, so
no response text, harm label or judge output enters a selection ("Seeds and order": harm labels
never enter a selection). The structural check of the responses that the launcher makes
(``stage_a.load_trajectory``) is not repeated here; the launcher's verdict on it is its end
record, whose outcome is read and whose free text is not copied.

**The issued cell** ("Seeds and order": provenance from the first day; "Voiding and supersession").
Every attempt is bound, before any outcome is read, to the cell its campaign issued: its start in
the launch journal (run id, label, attempt, config hash, header) and, when its directory has them,
its ``meta.json`` (the same campaign name, label, attempt and header; the same config hash) and
its ``config.yaml`` (the config whose hash is the start's; without a readable one, the start's
command loaded again must give that hash). The header names study 3, the candidate (1 or 2), the
step (``STEP_ROLES``) and its round (``round``, present, None only when written so), and for
validation and confirmation the versions record and its sha256. From the config: the arm
(``ARMS``), the rate, the seed, the role (``study3.role``, the step's) and whether the run is
K-only (the step's kind), at B = 20 and 25 updates under the fixed schedule evaluated at step 25.
Refused by name, before any outcome is read: a run directory with no start in the journal, a run
id started twice, a start without a header that issues such a cell, a recorded campaign that is
not the journal's, a config hash that is not the cell's, an unknown arm, a role or kind that is
not the step's, another schedule, a seed outside its role's registered seeds for the header's
candidate (confirmation seeds are bound to their candidate by the header alone), a label that is
not its seed's, an (arm, rate, seed) that ``prescribed`` does not hold under its step key, and an
unreadable restart or supersession record (it may name a run). A refused attempt is listed with
the state ``refused``, its refusal as its reason, and what of its cell could be bound. A start
whose config can no longer be bound (no readable ``config.yaml``, and its command no longer gives
its config hash under this checkout's configs) is not refused but ``technical``, with the reason:
a recovery round or a supersession clears it. Authorized historical rounds stay in
``prescribed`` and are listed with their cells; which step's cells enter a median, a support or a
gate is the caller's choice (``Cell.key``).

**Attempts** ("Failure routes": an infrastructure interruption is rerun under a new attempt id;
"Voiding and supersession"). The journal is every log under ``launch/`` read as one: no rule
depends on a log's name or a line's place in it, but for a line that cannot be parsed: it is
skipped silently only as the last line of its log (cut short by a crash, the launcher's rule),
and before the last it is a problem, named by its log and line number. An attempt is a start;
its campaign is the log that holds its start, and the attempts of a label are its starts in any
log, in the order of their attempt numbers (then of their starts' time stamps). Ends,
``superseded`` and ``held`` lines name their attempt by run id, in any log: the end the
launcher writes after the fact for a start it found without one (``"settled": true``) included,
and an end ``done`` is the outcome whatever stop its terminal record holds. An attempt that no
longer counts is listed with its reason, never counted:
- ``superseded``: named by a ``superseded`` line, by a case restart
  (``c<N>/<stage>/<case>/restart-<n>.json``, its ``launcher``) or by an engineering cycle's
  supersession (``c<N>/supersede-<n>.json``, its ``launchers``);
- ``held``: named by a ``held`` line or by a supersession's ``held``: an outcome the engineering
  cycle could not resolve, never attempted again (the launcher refuses its label under any
  config hash: ``read_holds``);
- ``recovered``: an attempt without a published terminal record before a recovery round of its
  label (``--recovery REASON``, its ``runs``, in any log): the documented interruption
  supersedes it, and a later attempt decides. Before the round are the label's attempts numbered
  up to the highest number started before the round's time stamp (the launcher numbers a run's
  attempts in order, and logs a round before the attempt it allows).
Every other attempt that is not refused counts. Two counting attempts of one scientific cell (the
step key and the prescribed (arm, rate, seed) both matched), or of one launcher run (one config
hash, whatever its campaign or label), are refused by name: a retry replaces an attempt only
through a recorded recovery round or supersession, and never an attempt with a published
outcome. A start without a readable time stamp is refused; a recovery round without one is a
problem of its own (its attempts cannot be placed).

**States** of a counting attempt:
- ``pending``: the launcher has not ended it (no end, or an end ``stopped``), whether or not a
  terminal record is published, unless ``meta.json`` recorded the run failed or finished without
  one: the step waits;
- ``technical``: an unexplained failure (an end ``technical``, ``exhausted``, ``conflict`` or
  any other but ``done``, ``diverged`` or ``stopped``; ends that disagree; no published terminal
  record after an end, or after ``meta.json`` recorded the run ``failed``, or finished; a config
  that can no longer be bound), or a record that does not conform (below): the technical route
  ("Failure routes"), never pending;
- ``scientific``: a declared instability stop (``INSTABILITY``, the schedule's) on a conforming
  record: no endpoint ("Protocol": a scientific endpoint failure);
- ``usable``: 25 updates with finite records and a step-25 evaluation; a K-only run needs a
  positive, finite K, a regular run an MMLU loss of at most 15 pp ("Usable runs");
- ``unusable`` (``unusable_reason``): a K-only run whose K is not positive (``nonpositive_k``),
  or a regular run that lost more than 15 pp of MMLU (``gross_collapse``, decided on utility
  before any label); it removes support and is no route by itself.

**A conforming record**, read from the numbers alone (the launcher's and ``stage_a``'s integrity
rules restricted to the files above): a readable terminal record that ``summary.json`` repeats
when it holds one, with the stop's own detail; a stop that is either the 25 updates (``steps``)
or an instability rule before or at step 25; one training row per update taken; a monitor
reading every ``eval.monitor.every`` steps up to the last step (the one before for
``nonfinite_weight``); the evaluation at step 25 for a run that took its 25 updates and none
after an instability stop; the training stream's recorded count equal to the responses used; a
K-only run that records ``k_only`` and no harmful prompts; every number finite (the terminal
record but a refused batch's description, the evaluations, the training rows, the readings and
the logged rows but the last step's of a ``nonfinite_k`` stop); and in the step-25 evaluation a
finite K_benign and, for a regular run, K_harmful and both accuracies.

**Endpoint values**: K = K_benign and, for a regular run, K_harmful, both of the step-25
evaluation; the utility losses θ₀ − run accuracy, in pp, for MMLU (``benchmark_accuracy``) and
the factual questions (``factual_accuracy``); steps and stop from the terminal record.

**Rulings** where the registration leaves a detail open, fixed here before any study-3 record:
- *The 15 pp bound* is inclusive: a loss of exactly 15 pp is usable; the comparison allows
  10⁻⁹ pp for the floating-point difference of two accuracies.
- *A K-only run's utility* (amendment 1: recorded, descriptive only) enters no state: its
  accuracies and held-out NLL are left out of the finiteness check, in the evaluations and in
  the rows the schedule logs from them (``eval/<key>`` in ``metrics.jsonl``), and a non-finite or
  missing one gives a loss of None.
- *Prescribed rates* compare as numbers (relative tolerance 10⁻⁹), so that the spelling of a rate
  in a campaign does not matter; the step keys are exact. The prescribed (arm, rate, seed) a run
  matched is its scientific cell.
- *An outcome is read only once the launcher has ended its attempt* (an end ``done`` or
  ``diverged``): a published terminal record without that end waits for it.
- *A recovery round and a start in the same second* (the journal's stamps are in seconds): the
  start is taken to follow the round, as the launcher logs a round before the attempt it allows.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..config import config_hash
from ..protocol import load_run_config

EXPERIMENT = "study3"
ROLES = ("engineering", "search", "numerical", "validation", "confirmation")
K_ONLY_ROLES = ("engineering", "search", "numerical")
# "Seeds and order": validation to 10339 / 10379 with 40 blocks; confirmation in order, its
# candidate named by its campaign's header.
SEEDS: dict[str, dict[int, range]] = {
    "engineering": {1: range(10000, 10004)},
    "search": {1: range(10100, 10104), 2: range(10110, 10114)},
    "numerical": {1: range(10200, 10203), 2: range(10210, 10213)},
    "validation": {1: range(10300, 10340), 2: range(10340, 10380)},
    "confirmation": {1: range(10400, 10600), 2: range(10400, 10600)},
}
ARMS = {"adamw": "A", "adam_frozen": "F", "adamw_prewarmed": "P"}  # optimizer.name -> arm
# A campaign header's step and the role of its runs: the repair's rates are qualified as
# numerical runs.
STEP_ROLES = {"engineering": "engineering", "search": "search", "numerical": "numerical",
              "repair": "numerical", "validation": "validation",
              "confirmation": "confirmation"}  # fmt: skip
VERSIONED_STEPS = ("validation", "confirmation")
BATCH, UPDATES = 20, 25
# schedule.INSTABILITY: the declared instability rules (the schedule itself is a GPU path)
INSTABILITY = ("nonfinite_loss", "nonfinite_gradient", "nonfinite_weight", "nonfinite_k",
               "k_above_max", "gradient_spike", "contamination")  # fmt: skip
STOP_DETAIL = {"gradient_spike": {"grad_norm", "reference_median"},
               "contamination": {"K_scoring", "bound"},
               "k_above_max": {"K_monitor", "k_max"}}  # fmt: skip
GROSS_COLLAPSE_PP = 15.0
TOLERANCE_PP = 1e-9
RATE_TOLERANCE = 1e-9
GROSS_COLLAPSE, NONPOSITIVE_K = "gross_collapse", "nonpositive_k"  # unusable_reason
# Utility (amendment 1: descriptive for a K-only run), in an evaluation and as the schedule logs
# it (``eval/<key>``)
UTILITY_KEYS = ("benchmark_accuracy", "factual_accuracy", "heldout_nll_per_token")
LOGGED_UTILITY_KEYS = tuple(f"eval/{key}" for key in UTILITY_KEYS)
STATES = ("pending", "recovered", "superseded", "held", "refused", "technical", "scientific",
          "usable", "unusable")  # fmt: skip
COUNTING = ("pending", "technical", "scientific", "usable", "unusable")
OUTCOME_ENDS = ("done", "diverged")  # ends after which the records decide
OPEN_ENDS = (None, "stopped")  # the launcher has not ended the attempt


@dataclass(frozen=True)
class Cell:
    """The cell a campaign issued for a run: the campaign (its journal's name) and the label,
    the header's candidate, step and round (and versions record), the config's arm, rate and
    seed, the config hash of its start, and the prescribed (arm, rate, seed) it matched. A field
    is None where the records do not bind it (a refused or unbound attempt)."""

    campaign: str | None
    label: str | None
    candidate: int | None
    step: str | None
    round: int | None
    arm: str | None
    rate: float | None
    seed: int | None
    config_hash: str | None
    versions_record: str | None = None
    versions_record_sha256: str | None = None
    prescribed: tuple[str, float, int] | None = None

    @property
    def key(self) -> str | None:
        """The step key of ``prescribed``."""
        if self.candidate is None or self.step is None:
            return None
        return step_key(self.candidate, self.step, self.round)

    @property
    def scientific(self) -> tuple[str, str, float, int] | None:
        """The scientific cell: the step key and the prescribed (arm, rate, seed) matched."""
        key = self.key
        return None if key is None or self.prescribed is None else (key, *self.prescribed)


@dataclass(frozen=True)
class Endpoint:
    """One attempt of a study-3 run (or a study-3 run directory with no start): its issued cell,
    its state and reasons, and its step-25 endpoint (None where it has none); losses in pp
    against θ₀. The cell's fields are None where the records do not bind them."""

    run_id: str
    run_dir: Path
    cell: Cell
    arm: str | None
    lr: float | None
    seed: int | None
    role: str | None
    candidate: int | None
    k_only: bool | None
    attempt: int | None
    label: str | None
    state: str  # STATES
    reasons: tuple[str, ...] = ()
    steps: int | None = None
    stop: str | None = None
    K: float | None = None
    K_harmful: float | None = None
    mmlu_loss_pp: float | None = None
    factual_loss_pp: float | None = None
    unusable_reason: str | None = None  # GROSS_COLLAPSE or NONPOSITIVE_K when unusable

    @property
    def counts(self) -> bool:
        """Whether this attempt is its label's: the others are listed, never counted."""
        return self.state in COUNTING


@dataclass
class _Attempt:
    campaign: str | None
    started: datetime | None  # its start's time stamp
    run_id: str
    run_dir: Path
    cell: Cell
    number: int | None  # the attempt's number in its start
    role: str | None = None
    k_only: bool | None = None
    every: int | None = None  # eval.monitor.every
    unbound: str | None = None  # why its config can no longer be bound (technical)
    state: str | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class _Journal:
    """The study-3 records of every log; nothing here depends on a log's name or a line's
    place."""

    starts: list[tuple[str, dict[str, Any]]] = field(default_factory=list)  # campaign, start
    ends: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    recoveries: list[dict[str, Any]] = field(default_factory=list)
    corrupt: list[str] = field(default_factory=list)  # "<log> line <n>": an unparseable line


def step_key(candidate: int, step: str, round_: int | None = None) -> str:
    """``c<candidate>/<step>``, with ``/r<round>`` when the header names a round."""
    return f"c{candidate}/{step}" + ("" if round_ is None else f"/r{round_}")


def candidate_of(seed: int, role: str) -> int | None:
    """The candidate whose registered seeds of ``role`` hold ``seed``; None when none does, or
    when both do (confirmation: its campaign's header binds the candidate)."""
    owners = [c for c, seeds in SEEDS.get(role, {}).items() if seed in seeds]
    return owners[0] if len(owners) == 1 else None


def read_theta0(path: str | Path) -> dict[str, float]:
    """θ₀'s accuracies from its evaluation's ``summary.json`` (the run directory or the file):
    each a finite fraction in [0, 1]; ``ValueError`` otherwise."""
    path = Path(path)
    file = path / "summary.json" if path.is_dir() else path
    try:
        summary = json.loads(file.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"θ₀: {file} is unreadable ({type(exc).__name__})") from exc
    out = {}
    for key in ("benchmark_accuracy", "factual_accuracy"):
        value = summary.get(key) if isinstance(summary, dict) else None
        if not (_number(value) and math.isfinite(value) and 0.0 <= value <= 1.0):
            raise ValueError(f"θ₀: {key} is {value!r}, not a fraction in [0, 1] ({file})")
        out[key] = float(value)
    return out


def read_endpoints(
    runs_root: str | Path,
    *,
    theta0: str | Path,
    prescribed: Mapping[str, Collection[tuple[str, float, int]]],
) -> tuple[list[Endpoint], list[str]]:
    """Every attempt of a study-3 run under ``runs_root`` and every study-3 run directory, bound
    to its issued cell, with its state and endpoint; and the refusals, by name (module
    docstring). ``theta0``: θ₀'s evaluation (``read_theta0``); ``prescribed``: per step key
    (``step_key``), the (arm, rate, seed) cells issued so far, historical rounds included."""
    root = Path(runs_root)
    baseline = read_theta0(theta0)
    voided: dict[str, tuple[str, str]] = {}  # run id -> superseded or held, and why
    journal = _read_journal(root, voided)
    problems = _voiding_records(root, voided)
    problems += [f"{where}: not valid JSON, and not the last line of its log (only a line cut "
                 "short by a crash is skipped)" for where in journal.corrupt]  # fmt: skip
    started = Counter(start.get("run_id") for _, start in journal.starts)
    attempts: list[_Attempt] = []
    for campaign, start in journal.starts:
        if not isinstance(start.get("run_id"), str):
            problems.append(f"{campaign}: a start in the launch journal without a run id")
            continue
        attempts.append(_bind(root, campaign, start, prescribed,
                              twice=started[start["run_id"]] > 1))  # fmt: skip
    experiment = root / EXPERIMENT
    for run_dir in sorted(experiment.iterdir()) if experiment.is_dir() else []:
        if run_dir.is_dir() and run_dir.name not in started:
            attempts.append(_Attempt(
                campaign=None, started=None, run_id=run_dir.name, run_dir=run_dir,
                cell=Cell(*[None] * 9), number=None, state="refused",
                reasons=["a study-3 run directory with no start in the launch journal: no "
                         "issued cell"]))  # fmt: skip
    problems += _select(attempts, journal.recoveries, voided)
    _refuse_duplicates(attempts)
    problems += [f"{a.run_id}: {'; '.join(a.reasons)}" for a in attempts if a.state == "refused"]
    endpoints = [_endpoint(a, journal.ends.get(a.run_id, []), baseline) for a in attempts]
    endpoints.sort(key=_order)
    return endpoints, sorted(set(problems))


def _order(e: Endpoint) -> tuple:
    """The listing's order: by cell, then by attempt number (the campaign's name only breaks
    ties)."""
    return (e.candidate or 0, e.cell.step or "", -1 if e.cell.round is None else e.cell.round,
            e.arm or "", e.lr or 0.0, e.seed or 0, e.attempt or 0, e.cell.campaign or "",
            e.run_id)  # fmt: skip


# ------------------------------------------------------------------ journal and voiding records


def _read_journal(root: Path, voided: dict[str, tuple[str, str]]) -> _Journal:
    """The launch journal's study-3 records from every log: the starts with their campaign (the
    log's name), every end by run id, the recovery rounds, and the ``superseded`` and ``held``
    lines into ``voided`` by run id. A line cut short by a crash is skipped only as the last line
    of its log (the launcher's rule); an unparseable line before the last is skipped too but
    listed in ``corrupt`` with its log's name and its number, never its content."""
    journal = _Journal()
    for log in sorted((root / "launch").glob("*.jsonl")):
        campaign = log.stem
        lines = log.read_text(encoding="utf-8").splitlines()
        last = max((i for i, line in enumerate(lines) if line.strip()), default=-1)
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                if i != last:
                    journal.corrupt.append(f"launch/{log.name} line {i + 1}")
                continue
            if not isinstance(record, dict):
                continue
            event, run_id = record.get("event"), record.get("run_id")
            if event == "start" and record.get("experiment") == EXPERIMENT:
                journal.starts.append((campaign, record))
            elif event == "end" and isinstance(run_id, str):
                journal.ends[run_id].append(record)
            elif event == "recovery" and record.get("experiment") == EXPERIMENT:
                journal.recoveries.append(record)
            elif event in ("superseded", "held") and isinstance(run_id, str):
                _void(voided, run_id, event,
                      f"{event} in the launch journal: {record.get('reason')}")  # fmt: skip
    return journal


def _voiding_records(root: Path, voided: dict[str, tuple[str, str]]) -> list[str]:
    """The launcher runs that the qualification's case restarts and engineering-cycle
    supersessions name (scripts/qualify.py, "A case restart", "Supersession"), into ``voided``:
    superseded or held, with the reason. An unreadable record is refused: it may name a run."""
    problems: list[str] = []
    out = root / "qualification"
    for path in sorted(out.glob("c*/*/s*/*/restart-*.json")):
        record = _json(path)
        if record is None:
            problems.append(f"{path}: an unreadable case-restart record")
            continue
        launcher = record.get("launcher")
        if isinstance(launcher, Mapping) and isinstance(launcher.get("run_id"), str):
            _void(voided, launcher["run_id"], "superseded",
                  f"case restart {path}: {record.get('reason')}")  # fmt: skip
    for path in sorted(out.glob("c*/supersede-*.json")):
        record = _json(path)
        if record is None:
            problems.append(f"{path}: an unreadable supersession record")
            continue
        why = f"engineering cycle {path}: {record.get('diagnosis')}; fix: {record.get('fix')}"
        for state, entries in (("superseded", "launchers"), ("held", "held")):
            for entry in record.get(entries) or []:
                if isinstance(entry, Mapping) and isinstance(entry.get("run_id"), str):
                    reason = why if state == "superseded" else (
                        f"held by {path}: {entry.get('reason')}")  # fmt: skip
                    _void(voided, entry["run_id"], state, reason)
    return problems


def _void(voided: dict[str, tuple[str, str]], run_id: str, state: str, reason: str) -> None:
    """Record that an attempt no longer counts: superseded, or held, which prevails."""
    if voided.get(run_id, ("",))[0] != "held":
        voided[run_id] = (state, reason)


def read_holds(runs_root: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Every held launcher run (``launch.hold``: an outcome an engineering cycle could not
    resolve, kept as it is and never attempted again), once per run id, with every record that
    holds it. A hold applies to its label, its ``experiment`` and ``run`` (``<cell>/s<seed>``),
    whatever the config hash it was written with: the launcher refuses that label under any
    config hash (``launch.held_runs``), so a caller lists a cell's holds by its label and
    experiment, never by its config hash.

    Each hold: ``run_id``; ``run``, ``experiment``, ``config_hash`` and ``reason`` as the first
    record that names it gives them; and ``records``, each an evidence reference (``path``
    relative to RUNS_DIR, ``root`` ``"runs"``, the file's ``sha256`` as it is now) with its
    ``kind``: ``journal``, a ``held`` line of a launch log (``line``: its number in the log,
    from 1; ``line_sha256``: the sha256 of the line's UTF-8 text, which stays when the log
    grows, as a log's own sha256 does not), and ``supersession``, an engineering cycle's record
    (``qualification/c<N>/supersede-<n>.json``) whose ``held`` names the run. Journal lines come
    first, by log name and line, then supersession records by path. The second value names the
    records that cannot be read (a log or a supersession record): a hold they hold cannot be
    listed. Of a record only its numbers, ids, paths, hashes and the hold's reason are kept."""
    root = Path(runs_root)
    holds: dict[str, dict[str, Any]] = {}
    problems: list[str] = []

    def add(entry: Mapping[str, Any], reference: dict[str, Any]) -> None:
        run_id = entry["run_id"]
        hold = holds.setdefault(run_id, {
            "run_id": run_id, "run": entry.get("run"), "experiment": entry.get("experiment"),
            "config_hash": entry.get("config_hash"), "reason": entry.get("reason"),
            "records": []})  # fmt: skip
        hold["records"].append(reference)

    for log in sorted((root / "launch").glob("*.jsonl")):
        where = log.relative_to(root).as_posix()
        try:
            data = log.read_bytes()
            lines = data.decode("utf-8").splitlines()
        except (OSError, ValueError):
            problems.append(f"{where}: an unreadable launch journal")
            continue
        sha = hashlib.sha256(data).hexdigest()
        for number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except ValueError:  # read_endpoints names such a line, unless it is the last
                continue
            if (isinstance(record, dict) and record.get("event") == "held"
                    and isinstance(record.get("run_id"), str)):  # fmt: skip
                add(record, {"kind": "journal", "path": where, "root": "runs", "line": number,
                             "line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                             "sha256": sha})  # fmt: skip
    for path in sorted((root / "qualification").glob("c*/supersede-*.json")):
        where = path.relative_to(root).as_posix()
        try:
            data = path.read_bytes()
            record = json.loads(data)
        except (OSError, ValueError):
            problems.append(f"{where}: an unreadable supersession record")
            continue
        entries = record.get("held") if isinstance(record, dict) else None
        reference = {"kind": "supersession", "path": where, "root": "runs",
                     "sha256": hashlib.sha256(data).hexdigest()}  # fmt: skip
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, Mapping) and isinstance(entry.get("run_id"), str):
                add(entry, dict(reference))
    return list(holds.values()), problems


# ------------------------------------------------------------------ the issued cell


def _bind(
    root: Path,
    campaign: str,
    start: Mapping[str, Any],
    prescribed: Mapping[str, Collection[tuple[str, float, int]]],
    *,
    twice: bool,
) -> _Attempt:
    """The attempt bound to its issued cell (module docstring, "The issued cell"): refused, with
    its refusals as its reasons, when it cannot be; with the reason it can no longer be bound
    (``unbound``) when its config cannot. Reads the run's ``meta.json`` and ``config.yaml``
    only."""
    run_id = start["run_id"]
    run_dir = root / EXPERIMENT / run_id
    label, number, chash = start.get("run"), start.get("attempt"), start.get("config_hash")
    label = label if isinstance(label, str) else None
    number = number if type(number) is int else None
    chash = chash if isinstance(chash, str) else None
    head = start.get("header")
    issued = dict(head) if isinstance(head, Mapping) else {}
    refusals = ["its run id is started more than once in the launch journal"] if twice else []
    if None in (label, number, chash):
        refusals.append("its start in the launch journal lacks its label, attempt or config hash")
    started = _time(start.get("time"))
    if started is None:
        refusals.append(
            f"its start in the launch journal has no time stamp ({start.get('time')!r})"
        )
    problem = _header_problem(head)
    if problem is not None:
        refusals.append(problem)
    meta = _json(run_dir / "meta.json")
    if meta is not None:
        recorded = meta.get("campaign")
        wanted = {"name": campaign, "label": label, "attempt": number, "header": head}
        if not isinstance(recorded, Mapping) or {k: recorded.get(k) for k in wanted} != wanted:
            refusals.append(f"its meta.json records another campaign than its start ({campaign}, "
                            f"{label}, attempt {number})")  # fmt: skip
        if meta.get("config_hash") != chash:
            refusals.append(f"its config hash {meta.get('config_hash')!r} is not its cell's "
                            f"{chash!r}")  # fmt: skip
    cfg, unbound, refusal = _config(run_dir, start, chash)
    if refusal is not None:
        refusals.append(refusal)
    facts: dict[str, Any] = {}
    if cfg is not None:
        facts, found = _facts(cfg, label, issued if problem is None else {}, prescribed)
        refusals += found
    candidate, step = issued.get("candidate"), issued.get("step")
    cell = Cell(
        campaign=campaign, label=label,
        candidate=candidate if candidate in (1, 2) and type(candidate) is int else None,
        step=step if step in STEP_ROLES else None,
        round=issued.get("round") if type(issued.get("round")) is int else None,
        arm=facts.get("arm"), rate=facts.get("rate"), seed=facts.get("seed"), config_hash=chash,
        versions_record=_text(issued.get("versions_record")),
        versions_record_sha256=_text(issued.get("versions_record_sha256")),
        prescribed=facts.get("prescribed"),
    )  # fmt: skip
    return _Attempt(campaign=campaign, started=started, run_id=run_id, run_dir=run_dir,
                    cell=cell, number=number, role=facts.get("role"),
                    k_only=facts.get("k_only"), every=facts.get("every"), unbound=unbound,
                    state="refused" if refusals else None, reasons=refusals)  # fmt: skip


def _facts(
    cfg: Any,
    label: str | None,
    issued: Mapping[str, Any],
    prescribed: Mapping[str, Collection[tuple[str, float, int]]],
) -> tuple[dict[str, Any], list[str]]:
    """What a bound config says of its run, and why its cell is not one the header issued
    (``issued``: a valid header, or empty when it is not one)."""
    problems = []
    name = OmegaConf.select(cfg, "optimizer.name", default=None)
    rate, seed = OmegaConf.select(cfg, "optimizer.lr", default=None), _value(cfg, "seed")
    role, every = _value(cfg, "study3.role"), _value(cfg, "eval.monitor.every")
    facts = {"arm": ARMS.get(name), "rate": float(rate) if _number(rate) else None,
             "seed": seed if type(seed) is int else None,
             "role": role if role in ROLES else None,
             "k_only": OmegaConf.select(cfg, "eval.k_only", default=False) is True,
             "every": every if type(every) is int and every >= 1 else None}  # fmt: skip
    if name not in ARMS:
        problems.append(f"an unknown arm: optimizer {name!r}")
    if facts["rate"] is None or facts["seed"] is None or facts["every"] is None:
        problems.append(f"its config has no rate, seed or monitor interval ({rate!r}, {seed!r}, "
                        f"{every!r})")  # fmt: skip
    evaluate_at = OmegaConf.select(cfg, "schedule.evaluate_at", default=None)
    protocol = (OmegaConf.select(cfg, "train.batch_size", default=None),
                OmegaConf.select(cfg, "train.steps", default=None),
                OmegaConf.select(cfg, "schedule.policy", default=None),
                list(evaluate_at or []))  # fmt: skip
    if protocol != (BATCH, UPDATES, "fixed", [UPDATES]):
        problems.append(f"not the registered schedule (batch, steps, policy, evaluations: "
                        f"{protocol})")  # fmt: skip
    if label is not None and not label.endswith(f"/s{seed}"):
        problems.append(f"its label {label!r} is not that of seed {seed}")
    if not issued or problems:
        return facts, problems
    step, candidate, round_ = issued["step"], issued["candidate"], issued["round"]
    if role != STEP_ROLES[step]:
        problems.append(f"role {role!r} under the header's step {step!r}")
    if facts["k_only"] != (role in K_ONLY_ROLES):
        problems.append(f"{'a K-only' if facts['k_only'] else 'a regular'} run in the role "
                        f"{role!r}")  # fmt: skip
    seeds = SEEDS.get(role, {}).get(candidate) if role in ROLES else None
    if seeds is None or seed not in seeds:
        span = f"{seeds[0]}–{seeds[-1]}" if seeds else "none"
        problems.append(f"seed {seed} is outside the seeds of {role!r} for candidate {candidate} "
                        f"({span})")  # fmt: skip
    key = step_key(candidate, step, round_)
    matched = sorted(cell for cell in prescribed.get(key, ())
                     if cell[0] == facts["arm"] and cell[2] == seed and _number(cell[1])
                     and math.isclose(cell[1], facts["rate"], rel_tol=RATE_TOLERANCE))  # fmt: skip
    if matched:
        facts["prescribed"] = tuple(matched[0])
    else:
        problems.append(f"({facts['arm']}, {facts['rate']:g}, {seed}) is not prescribed for {key}")
    return facts, problems


def _header_problem(head: Any) -> str | None:
    """Why a campaign header issues no study-3 cell (None: it does)."""
    if not isinstance(head, Mapping) or head.get("study") != 3:
        return f"its campaign header is not a study-3 header ({head!r})"
    if head.get("candidate") not in (1, 2) or type(head.get("candidate")) is not int:
        return f"its campaign header names no candidate 1 or 2 ({head.get('candidate')!r})"
    if head.get("step") not in STEP_ROLES:
        return f"its campaign header names no study-3 step ({head.get('step')!r})"
    if "round" not in head:
        return "its campaign header carries no round (None only when written so)"
    round_ = head["round"]
    if round_ is not None and (type(round_) is not int or round_ < 0):
        return f"its campaign header's round is {round_!r}"
    if head["step"] in VERSIONED_STEPS:
        sha = head.get("versions_record_sha256")
        if not (isinstance(head.get("versions_record"), str) and head["versions_record"]
                and isinstance(sha, str) and len(sha) == 64):  # fmt: skip
            return "its campaign header names no versions record with its sha256"
    return None


def _config(run_dir: Path, start: Mapping[str, Any], chash: str | None) -> tuple[Any, Any, Any]:
    """The run's config, whose hash is its cell's: its ``config.yaml``, or without a readable
    one its start's command loaded again. Returns (config, why it can no longer be bound, why it
    is refused), the config None unless it binds."""
    path = run_dir / "config.yaml"
    unreadable = None
    if path.is_file():
        try:
            cfg = OmegaConf.load(path)
            found = config_hash(cfg)
        except Exception as exc:  # an unreadable config binds nothing
            unreadable = f"its config.yaml cannot be read ({type(exc).__name__})"
        else:
            if found != chash:
                return None, None, (f"its config.yaml's config hash {found!r} is not its cell's "
                                    f"{chash!r}")  # fmt: skip
            return cfg, None, None
    command = start.get("command")
    try:
        if not (isinstance(command, list) and len(command) >= 2
                and all(isinstance(c, str) for c in command)):  # fmt: skip
            raise ValueError("no command")
        cfg = load_run_config(command[1], command[2:])
        found = config_hash(cfg)
    except Exception as exc:  # no config to bind it under this checkout's configs
        why = f"its start's command gives no config hash ({type(exc).__name__})"
    else:
        if found == chash:
            return cfg, unreadable, None
        why = f"its start's command gives the config hash {found!r}, not its cell's {chash!r}"
    return None, f"{unreadable or 'no config.yaml'}, and {why}", None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _time(value: Any) -> datetime | None:
    """A journal record's time stamp (the launcher's: ISO 8601, UTC, seconds), None without a
    readable one."""
    try:
        when = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo is not None else None


# ------------------------------------------------------------------ attempts


def _select(
    attempts: list[_Attempt],
    recoveries: Sequence[Mapping[str, Any]],
    voided: Mapping[str, tuple[str, str]],
) -> list[str]:
    """Mark the attempts that no longer count (module docstring, "Attempts"); a refused one
    stays refused. Returns the recovery rounds that cannot be placed (no time stamp)."""
    by_label: dict[str, list[_Attempt]] = defaultdict(list)  # across every log
    for attempt in attempts:
        if attempt.cell.label is not None and attempt.number is not None:
            by_label[attempt.cell.label].append(attempt)
        if attempt.state is None and attempt.run_id in voided:
            attempt.state, reason = voided[attempt.run_id]
            attempt.reasons.append(reason)
    problems = []
    for record in recoveries:
        when = _time(record.get("time"))
        labels = record.get("runs")
        if when is None:
            problems.append(f"a recovery round of {labels!r} without a time stamp "
                            f"({record.get('time')!r}): its attempts cannot be placed")  # fmt: skip
            continue
        for label in labels if isinstance(labels, list) else []:
            mine = by_label.get(label, [])
            # the attempts before the round: numbered up to the last one started before it
            before = [a.number for a in mine if a.started is not None and a.started < when]
            for attempt in (a for a in mine if before and a.number <= max(before)):
                if attempt.state is None and not (attempt.run_dir / "terminal.json").is_file():
                    attempt.state = "recovered"
                    attempt.reasons.append(f"a recovery round followed it: {record.get('reason')}")
    return problems


def _refuse_duplicates(attempts: list[_Attempt]) -> None:
    """Refuse every attempt that counts beside another of its scientific cell or of its launcher
    run (its config hash, whatever the campaign or label: the launcher's run), in any log."""
    groups: dict[str, dict[Any, list[_Attempt]]] = {"cell": defaultdict(list),
                                                    "run": defaultdict(list)}  # fmt: skip
    for attempt in attempts:
        if attempt.state is None:
            if attempt.cell.scientific is not None:
                groups["cell"][attempt.cell.scientific].append(attempt)
            if attempt.cell.config_hash is not None:
                groups["run"][attempt.cell.config_hash].append(attempt)
    seen: set[frozenset[str]] = set()
    for kind, by_key in groups.items():
        for key, group in by_key.items():
            ids = frozenset(a.run_id for a in group)
            if len(group) < 2 or ids in seen:
                continue
            seen.add(ids)
            what = (f"one cell ({key[0]}, {key[1]}, {key[2]:g}, {key[3]})" if kind == "cell"
                    else f"one launcher run (config hash {key})")  # fmt: skip
            group.sort(key=lambda a: (a.number or 0, a.started or datetime.min, a.run_id))
            names = ", ".join(f"{a.run_id} ({a.campaign}, {a.cell.label}, attempt {a.number})"
                              for a in group)  # fmt: skip
            why = (
                f"{len(group)} attempts count for {what}: {names}; only a recorded recovery "
                "round or supersession replaces an attempt, and never a published outcome"
            )
            for attempt in group:
                attempt.state = "refused"
                attempt.reasons.append(why)


# ------------------------------------------------------------------ outcomes


def _endpoint(attempt: _Attempt, ends: Sequence[Mapping[str, Any]],
              theta0: Mapping[str, float]) -> Endpoint:  # fmt: skip
    cell = attempt.cell
    common = {"run_id": attempt.run_id, "run_dir": attempt.run_dir, "cell": cell,
              "arm": cell.arm, "lr": cell.rate, "seed": cell.seed, "role": attempt.role,
              "candidate": cell.candidate, "k_only": attempt.k_only,
              "attempt": attempt.number, "label": cell.label}  # fmt: skip
    if attempt.state is not None:
        return Endpoint(**common, state=attempt.state, reasons=tuple(attempt.reasons))
    return Endpoint(**common, **_outcome(attempt, ends, theta0))


def _outcome(attempt: _Attempt, ends: Sequence[Mapping[str, Any]],
             theta0: Mapping[str, float]) -> dict[str, Any]:  # fmt: skip
    """A counting attempt's state, reasons and endpoint (module docstring, "States"); ``ends``:
    its end records, from any log."""
    if attempt.unbound is not None:
        return {"state": "technical", "reasons": (attempt.unbound,)}
    outcomes = sorted({str(end.get("outcome")) for end in ends})
    if len(outcomes) > 1:  # an end is never rewritten: two verdicts cannot both stand
        why = f"the launch journal holds ends that disagree: {outcomes}"
        return {"state": "technical", "reasons": (why,)}
    run_dir = attempt.run_dir
    ended = ends[0].get("outcome") if ends else None
    published = (run_dir / "terminal.json").is_file()
    status = None if published else (_json(run_dir / "meta.json") or {}).get("status")
    finished = status not in (None, "running")  # failed, or finished without its outcome
    if ended in OPEN_ENDS and not finished:
        why = "the launcher has not ended it" if ended is None else "the launcher's end: stopped"
        return {"state": "pending", "reasons": (why,)}
    if not published:
        reasons = [f"the launcher's end record: {ended}"] if ended not in OPEN_ENDS else []
        reasons += [f"meta.json: status {status}"] if finished else []
        return {"state": "technical", "reasons": (*reasons, "no published terminal record")}
    if ended not in OUTCOME_ENDS:  # the launcher's verdict; its free text is not copied
        return {"state": "technical", "reasons": (f"the launcher's end record: {ended}",)}
    problems, terminal, row = _records(attempt)
    if problems:
        return {"state": "technical", "reasons": tuple(problems)}
    steps, stop = terminal["steps"], terminal["stop"]["reason"]
    out: dict[str, Any] = {"steps": steps, "stop": stop}
    if stop in INSTABILITY:
        return {**out, "state": "scientific", "reasons": (f"instability stop {stop} after "
                                                          f"{steps} updates",)}  # fmt: skip
    k = float(row["K_benign"])
    mmlu = _loss(theta0["benchmark_accuracy"], row.get("benchmark_accuracy"))
    factual = _loss(theta0["factual_accuracy"], row.get("factual_accuracy"))
    out |= {"K": k, "mmlu_loss_pp": mmlu, "factual_loss_pp": factual}
    if attempt.k_only:
        if not k > 0:
            return {**out, "state": "unusable", "unusable_reason": NONPOSITIVE_K,
                    "reasons": (f"K {k!r} is not positive",)}  # fmt: skip
        return {**out, "state": "usable", "reasons": ()}
    out["K_harmful"] = float(row["K_harmful"])
    if mmlu > GROSS_COLLAPSE_PP + TOLERANCE_PP:
        return {**out, "state": "unusable", "unusable_reason": GROSS_COLLAPSE,
                "reasons": (f"gross collapse: MMLU loss {mmlu:.4g} pp > 15 pp",)}  # fmt: skip
    return {**out, "state": "usable", "reasons": ()}


def _records(attempt: _Attempt) -> tuple[list[str], dict[str, Any] | None, dict[str, Any] | None]:
    """A run's records with a published terminal record, checked from their numbers (module
    docstring, "A conforming record"): the problems, the terminal record and the step-25
    evaluation (None without one)."""
    run_dir = attempt.run_dir
    try:
        meta = _load(run_dir / "meta.json")
        terminal = _load(run_dir / "terminal.json")
        summary = _load(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else {}
        evaluations = _jsonl(run_dir / "evaluations.jsonl")
        metrics = _jsonl(run_dir / "metrics.jsonl")
        readings = _jsonl(run_dir / "monitor.jsonl")
        stop, steps = terminal["stop"]["reason"], terminal["steps"]
        eval_steps = [row["step"] for row in evaluations]
        monitor_steps = [row["step"] for row in readings]
        training = [row for row in metrics if "loss" in row]
        training_steps = [row["step"] for row in training]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"unreadable records ({type(exc).__name__}: {exc})"], None, None
    if not (isinstance(stop, str) and type(steps) is int and 0 <= steps <= UPDATES):
        return [f"a terminal record without a stop and its steps ({stop!r}, {steps!r})"], None, None
    problems = []
    if "terminal" in summary and summary["terminal"] != terminal:
        problems.append("summary.json's terminal record differs from terminal.json")
    if set(terminal["stop"].get("detail") or {}) != STOP_DETAIL.get(stop, set()):
        problems.append(f"the terminal record's detail is not that of a {stop} stop")
    if stop == "steps":
        if steps != UPDATES:
            problems.append(f"a stop by its steps after {steps} updates, not {UPDATES}")
    elif stop not in INSTABILITY:
        problems.append(f"a stop that is neither its {UPDATES} updates nor an instability rule: "
                        f"{stop!r}")  # fmt: skip
    if training_steps != list(range(steps)):
        problems.append("training metrics are not one row per update taken")
    read_to = steps - 1 if stop == "nonfinite_weight" else steps  # no reading after a bad update
    if monitor_steps != list(range(attempt.every, read_to + 1, attempt.every)):
        problems.append("monitor readings are not every reading up to the last step")
    wanted = [UPDATES] if stop == "steps" else []
    if eval_steps != wanted:
        problems.append(f"evaluations at steps {eval_steps}, not {wanted}")
    inputs = meta.get("inputs") or {}
    count = ((inputs.get("streams") or {}).get("training") or {}).get("count")
    if count != steps * BATCH:
        problems.append(f"the recorded training stream holds {count!r} responses, the run used "
                        f"{steps * BATCH}")  # fmt: skip
    if attempt.k_only and (inputs.get("k_only") is not True or "harmful_prompts" in inputs):
        problems.append("a K-only run whose inputs do not record k_only, or record harmful prompts")
    last = steps if stop == "nonfinite_k" else None  # its last reading may be its cause
    stop_record = {k: v for k, v in terminal["stop"].items() if k != "rejected"}
    if _nonfinite({**terminal, "stop": stop_record}):
        problems.append("terminal record: a non-finite number")
    descriptive = (*UTILITY_KEYS, *LOGGED_UTILITY_KEYS) if attempt.k_only else ()  # amendment 1
    logged = [_without(row, descriptive) for row in metrics if "loss" not in row]
    problems += _finiteness("training metrics", training, None)
    problems += _finiteness("monitor readings", readings, last)
    problems += _finiteness("logged metrics", logged, None if last is None else last - 1)
    problems += _finiteness("evaluations", [_without(row, descriptive) for row in evaluations],
                            None)  # fmt: skip
    row = evaluations[0] if eval_steps == [UPDATES] else None
    if stop == "steps" and row is not None:
        needed = ("K_benign",) if attempt.k_only else (
            "K_benign", "K_harmful", "benchmark_accuracy", "factual_accuracy")  # fmt: skip
        problems += [f"the step-25 evaluation lacks {key}" for key in needed
                     if not _number(row.get(key))]  # fmt: skip
    return problems, terminal, row


# ------------------------------------------------------------------ helpers


def _loss(theta0: float, accuracy: Any) -> float | None:
    """θ₀'s accuracy minus the run's, in pp; None without a finite accuracy."""
    if not (_number(accuracy) and math.isfinite(accuracy)):
        return None
    return 100.0 * (theta0 - accuracy)


def _without(row: Mapping[str, Any], keys: Collection[str]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in keys}


def _value(cfg: Any, key: str) -> Any:
    """A config value, None when absent or unset (``???``)."""
    try:
        return OmegaConf.select(cfg, key, default=None, throw_on_missing=True)
    except Exception:  # a ``???`` value
        return None


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _nonfinite(value: Any) -> bool:
    """Whether a value holds a non-finite number anywhere (nested lists and mappings)."""
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_nonfinite(v) for v in value.values())
    if isinstance(value, list | tuple):
        return any(_nonfinite(v) for v in value)
    return False


def _finiteness(table: str, rows: Sequence[Mapping[str, Any]], allowed: int | None) -> list[str]:
    """Rows of ``table`` holding a non-finite number; ``allowed`` names the one step whose rows
    may."""
    bad = sorted({str(row.get("step")) for row in rows if _nonfinite(row)
                  and (allowed is None or row.get("step") != allowed)})  # fmt: skip
    return [f"{table}: a non-finite number at step {b}" for b in bad]


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return data


def _json(path: Path) -> dict[str, Any] | None:
    try:
        return _load(path)
    except (OSError, ValueError):
        return None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    """The rows of a JSON-lines record; none when it is absent."""
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name} holds a row that is not a JSON object")
    return rows
