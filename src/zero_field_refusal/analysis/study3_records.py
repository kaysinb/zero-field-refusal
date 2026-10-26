"""Study 3's run records as endpoints, each bound to the cell its campaign issued, read from the
numbers alone.

docs/study3-preregistration.md: "Protocol" (B = 20, exactly 25 updates, the stops before step 25),
"Questions and estimands" (usable runs), "Utility" (the losses against θ₀), "Failure routes",
"Seeds and order", "Voiding and supersession" and amendment 1 (what K-only runs measure). Pure
functions of what lies under RUNS_DIR: the launch journal (``launch/<campaign>.jsonl``), the run
directories of the experiment ``study3`` (both configs, regular and K-only) and the
qualification's restart and supersession records (``qualification/c<N>/``).

- ``read_endpoints``: every attempt of a study-3 run, each bound to its issued cell, with its state
  and step-25 endpoint, and the refusals by name (``problems``). Nothing may be computed from
  records with a refusal: they are ambiguous.
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
its ``config.yaml`` (the config whose hash is the start's; without a directory or a config, the
start's command loaded again must give that hash). The header names study 3, the candidate (1 or
2), the step (``STEP_ROLES``) and its round, and for validation and confirmation the versions
record and its sha256. From the config: the arm (``ARMS``), the rate, the seed, the role
(``study3.role``, the step's) and whether the run is K-only (the step's kind), at B = 20 and 25
updates under the fixed schedule evaluated at step 25. Refused by name, before any outcome is
read: a run directory with no start in the journal, a run id started twice, a start without a
header that issues such a cell, a recorded campaign that is not the journal's, a config hash that
is not the cell's, an unknown arm, a role or kind that is not the step's, another schedule, a
seed outside its role's registered seeds for the header's candidate (confirmation seeds are bound
to their candidate by the header alone), a label that is not its seed's, an (arm, rate, seed)
that ``prescribed`` does not hold under its step key, and an unreadable restart or supersession
record (it may name a run). Authorized historical
rounds stay in ``prescribed`` and are listed with their cells; which step's cells enter a median,
a support or a gate is the caller's choice (``Cell.key``).

**Attempts** ("Failure routes": an infrastructure interruption is rerun under a new attempt id;
"Voiding and supersession"). The attempts of a label are its starts in its campaign's journal,
in order. One that no longer counts is listed with its reason, never counted:
- ``superseded``: named by a ``superseded`` line of the journal, by a case restart
  (``c<N>/<stage>/<case>/restart-<n>.json``, its ``launcher``) or by an engineering cycle's
  supersession (``c<N>/supersede-<n>.json``, its ``launchers``);
- ``held``: named by a ``held`` line of the journal or by a supersession's ``held``: an outcome
  the engineering cycle could not resolve, never attempted again;
- ``recovered``: an attempt without a published terminal record that the launcher's recovery
  round (``--recovery REASON``, logged for its label) followed: the documented interruption
  supersedes it, and the next attempt decides.
Every other attempt counts. Two counting attempts of one scientific cell (step key, arm, rate,
seed), within a label or across campaigns, are refused by name: a retry replaces an attempt only
through a recorded recovery round or supersession, and never an attempt with a published outcome.

**States** of a counting attempt:
- ``pending``: no published terminal record, the launcher has not ended it (no end, or an end
  ``stopped``) and ``meta.json`` (if any) says it is running: the step waits;
- ``technical``: an unexplained failure (an end ``technical``, ``exhausted``, ``conflict`` or
  any other but ``done``, ``diverged`` or ``stopped``; no published terminal record after an end,
  or after ``meta.json`` recorded the run ``failed``, or finished), or a record that does not
  conform (below): the technical route ("Failure routes"), never pending;
- ``scientific``: a declared instability stop (``INSTABILITY``, the schedule's) on a conforming
  record: no endpoint ("Protocol": a scientific endpoint failure);
- ``usable``: 25 updates with finite records and a step-25 evaluation; a K-only run needs a
  positive, finite K, a regular run an MMLU loss of at most 15 pp ("Usable runs");
- ``unusable``: a K-only run whose K is not positive, or a regular run that lost more than 15 pp
  of MMLU (``gross collapse``, decided on utility before any label); it removes support and is no
  route by itself.

**A conforming record**, read from the numbers alone (the launcher's and ``stage_a``'s integrity
rules restricted to the files above): a readable terminal record that ``summary.json`` repeats
when it holds one, with the stop's own detail; a stop that is either the 25 updates (``steps``)
or an instability rule before or at step 25; one training row per update taken; a monitor
reading every ``eval.monitor.every`` steps up to the last step (the one before for
``nonfinite_weight``); the evaluation at step 25 for a run that took its 25 updates and none at
or after an instability stop; the training stream's recorded count equal to the responses used;
a K-only run that records ``k_only`` and no harmful prompts; every number finite (the terminal
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
  accuracies and held-out NLL are left out of the finiteness check, and a non-finite or missing
  one gives a loss of None.
- *Prescribed rates* compare as numbers (relative tolerance 10⁻⁹), so that the spelling of a rate
  in a campaign does not matter; the step keys are exact.
- *The launcher's end record* decides only its outcome; a published terminal record is the
  outcome whatever ``meta.json``'s status says (the launcher's own rule).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
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
UTILITY_KEYS = ("benchmark_accuracy", "factual_accuracy", "heldout_nll_per_token")
STATES = ("pending", "recovered", "superseded", "held", "technical", "scientific", "usable",
          "unusable")  # fmt: skip
COUNTING = ("pending", "technical", "scientific", "usable", "unusable")
SETTLED_ENDS = (None, "done", "diverged", "stopped")  # ends that leave the outcome to the records


@dataclass(frozen=True)
class Cell:
    """The cell a campaign issued for a run: the campaign (its journal's name) and the label,
    the header's candidate, step and round (and versions record), the config's arm, rate and
    seed, and the config hash of its start."""

    campaign: str
    label: str
    candidate: int
    step: str
    round: int | None
    arm: str
    rate: float
    seed: int
    config_hash: str
    versions_record: str | None = None
    versions_record_sha256: str | None = None

    @property
    def key(self) -> str:
        """The step key of ``prescribed``."""
        return step_key(self.candidate, self.step, self.round)

    @property
    def scientific(self) -> tuple[str, str, str, int]:
        """The scientific cell: step key, arm, rate (as a number) and seed."""
        return (self.key, self.arm, f"{self.rate:.9g}", self.seed)


@dataclass(frozen=True)
class Endpoint:
    """One attempt of a study-3 run: its issued cell, its state and reasons, and its step-25
    endpoint (None where it has none); losses in pp against θ₀."""

    run_id: str
    run_dir: Path
    cell: Cell
    arm: str
    lr: float
    seed: int
    role: str
    candidate: int
    k_only: bool
    attempt: int
    label: str
    state: str  # STATES
    reasons: tuple[str, ...] = ()
    steps: int | None = None
    stop: str | None = None
    K: float | None = None
    K_harmful: float | None = None
    mmlu_loss_pp: float | None = None
    factual_loss_pp: float | None = None

    @property
    def counts(self) -> bool:
        """Whether this attempt is its label's: the others are listed, never counted."""
        return self.state in COUNTING


@dataclass
class _Attempt:
    campaign: str
    position: int  # the start's line in its campaign's journal
    run_id: str
    run_dir: Path
    cell: Cell
    number: int  # the attempt's number in its start
    role: str
    k_only: bool
    every: int  # eval.monitor.every
    state: str | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class _Journal:
    starts: list[tuple[str, int, dict[str, Any]]] = field(default_factory=list)
    ends: dict[str, dict[str, Any]] = field(default_factory=dict)  # run id -> its last end
    recoveries: list[tuple[str, int, dict[str, Any]]] = field(default_factory=list)


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
    """Every attempt of a study-3 run under ``runs_root``, bound to its issued cell, with its
    state and endpoint; and the refusals, by name (module docstring). ``theta0``: θ₀'s
    evaluation (``read_theta0``); ``prescribed``: per step key (``step_key``), the (arm, rate,
    seed) cells issued so far, historical rounds included. A refused attempt is not listed."""
    root = Path(runs_root)
    baseline = read_theta0(theta0)
    voided: dict[str, tuple[str, str]] = {}  # run id -> superseded or held, and why
    journal = _read_journal(root, voided)
    problems = _voiding_records(root, voided)
    started = Counter(start.get("run_id") for _, _, start in journal.starts)
    attempts: list[_Attempt] = []
    for campaign, position, start in journal.starts:
        run_id = start.get("run_id")
        if not isinstance(run_id, str) or started[run_id] > 1:
            problems.append(f"{campaign}: a start in the launch journal without its own run id "
                            f"({run_id!r})")  # fmt: skip
            continue
        bound = _bind(root, campaign, position, start, prescribed)
        if isinstance(bound, str):
            problems.append(f"{run_id}: {bound}")
        else:
            attempts.append(bound)
    experiment = root / EXPERIMENT
    for run_dir in sorted(experiment.iterdir()) if experiment.is_dir() else []:
        if run_dir.is_dir() and run_dir.name not in started:
            problems.append(f"{run_dir.name}: a study-3 run directory with no start in the launch "
                            "journal: no issued cell")  # fmt: skip
    _select(attempts, journal.recoveries, voided)
    counting: dict[tuple, list[_Attempt]] = defaultdict(list)
    for attempt in attempts:
        if attempt.state is None:
            counting[attempt.cell.scientific].append(attempt)
    refused: set[str] = set()
    for (key, arm, rate, seed), group in counting.items():
        if len(group) > 1:
            names = ", ".join(f"{a.run_id} ({a.campaign}, {a.cell.label}, attempt {a.number})"
                              for a in group)  # fmt: skip
            problems.append(f"{names}: {len(group)} attempts count for one cell ({key}, {arm}, "
                            f"{rate}, {seed}): only a recorded recovery round or supersession "
                            "replaces an attempt, and never a published outcome")  # fmt: skip
            refused |= {a.run_id for a in group}
    endpoints = [_endpoint(a, journal.ends.get(a.run_id), baseline)
                 for a in attempts if a.run_id not in refused]  # fmt: skip
    endpoints.sort(key=lambda e: (e.candidate, e.cell.step, e.cell.round or 0, e.arm, e.lr,
                                  e.seed, e.cell.campaign, e.attempt, e.run_id))  # fmt: skip
    return endpoints, sorted(problems)


# ------------------------------------------------------------------ journal and voiding records


def _read_journal(root: Path, voided: dict[str, tuple[str, str]]) -> _Journal:
    """The launch journal's study-3 records, per campaign in order, its ``superseded`` and
    ``held`` lines into ``voided``; a line cut short by a crash is skipped (the launcher's
    rule)."""
    journal = _Journal()
    for log in sorted((root / "launch").glob("*.jsonl")):
        campaign = log.stem
        for position, line in enumerate(log.read_text().splitlines()):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            event, run_id = record.get("event"), record.get("run_id")
            if event == "start" and record.get("experiment") == EXPERIMENT:
                journal.starts.append((campaign, position, record))
            elif event == "end" and isinstance(run_id, str):
                journal.ends[run_id] = record
            elif event == "recovery" and record.get("experiment") == EXPERIMENT:
                journal.recoveries.append((campaign, position, record))
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


# ------------------------------------------------------------------ the issued cell


def _bind(
    root: Path,
    campaign: str,
    position: int,
    start: Mapping[str, Any],
    prescribed: Mapping[str, Collection[tuple[str, float, int]]],
) -> _Attempt | str:
    """The attempt bound to its issued cell, or why it cannot be (module docstring, "The issued
    cell"). Reads the run's ``meta.json`` and ``config.yaml`` only."""
    run_id = start["run_id"]
    label, attempt, chash = start.get("run"), start.get("attempt"), start.get("config_hash")
    head = start.get("header")
    if not (isinstance(label, str) and type(attempt) is int and isinstance(chash, str)):
        return "its start in the launch journal lacks its label, attempt or config hash"
    problem = _header_problem(head)
    if problem is not None:
        return problem
    run_dir = root / EXPERIMENT / run_id
    meta = _json(run_dir / "meta.json")
    if meta is not None:
        recorded = meta.get("campaign")
        issued = {"name": campaign, "label": label, "attempt": attempt, "header": head}
        if not isinstance(recorded, Mapping) or {k: recorded.get(k) for k in issued} != issued:
            return (f"its meta.json records another campaign than its start ({campaign}, "
                    f"{label}, attempt {attempt})")  # fmt: skip
        if meta.get("config_hash") != chash:
            return f"its config hash {meta.get('config_hash')!r} is not its cell's {chash!r}"
    cfg = _config(run_dir, start, chash)
    if isinstance(cfg, str):
        return cfg
    name = OmegaConf.select(cfg, "optimizer.name", default=None)
    if name not in ARMS:
        return f"an unknown arm: optimizer {name!r}"
    arm, rate = ARMS[name], OmegaConf.select(cfg, "optimizer.lr", default=None)
    seed, role = OmegaConf.select(cfg, "seed", default=None), _select_key(cfg, "study3.role")
    k_only = OmegaConf.select(cfg, "eval.k_only", default=False) is True
    if not (_number(rate) and type(seed) is int):
        return f"its config has no rate or seed ({rate!r}, {seed!r})"
    step, candidate, round_ = head["step"], head["candidate"], head.get("round")
    problems = []
    if role != STEP_ROLES[step]:
        problems.append(f"role {role!r} under the header's step {step!r}")
    if k_only != (role in K_ONLY_ROLES):
        problems.append(f"{'a K-only' if k_only else 'a regular'} run in the role {role!r}")
    evaluate_at = OmegaConf.select(cfg, "schedule.evaluate_at", default=None)
    protocol = (OmegaConf.select(cfg, "train.batch_size", default=None),
                OmegaConf.select(cfg, "train.steps", default=None),
                OmegaConf.select(cfg, "schedule.policy", default=None),
                list(evaluate_at or []))  # fmt: skip
    if protocol != (BATCH, UPDATES, "fixed", [UPDATES]):
        problems.append(f"not the registered schedule (batch, steps, policy, evaluations: "
                        f"{protocol})")  # fmt: skip
    if not label.endswith(f"/s{seed}"):
        problems.append(f"its label {label!r} is not that of seed {seed}")
    seeds = SEEDS.get(role, {}).get(candidate) if isinstance(role, str) else None
    if seeds is None or seed not in seeds:
        span = f"{seeds[0]}–{seeds[-1]}" if seeds else "none"
        problems.append(f"seed {seed} is outside the seeds of {role!r} for candidate {candidate} "
                        f"({span})")  # fmt: skip
    key = step_key(candidate, step, round_)
    if not any(a == arm and _number(r) and math.isclose(r, rate, rel_tol=RATE_TOLERANCE)
               and s == seed for a, r, s in prescribed.get(key, ())):  # fmt: skip
        problems.append(f"({arm}, {rate:g}, {seed}) is not prescribed for {key}")
    if problems:
        return "; ".join(problems)
    cell = Cell(campaign=campaign, label=label, candidate=candidate, step=step, round=round_,
                arm=arm, rate=float(rate), seed=seed, config_hash=chash,
                versions_record=head.get("versions_record"),
                versions_record_sha256=head.get("versions_record_sha256"))  # fmt: skip
    every = OmegaConf.select(cfg, "eval.monitor.every", default=None)
    if type(every) is not int or every < 1:
        return f"its config has no monitor interval ({every!r})"
    return _Attempt(campaign=campaign, position=position, run_id=run_id, run_dir=run_dir,
                    cell=cell, number=attempt, role=role, k_only=k_only, every=every)  # fmt: skip


def _header_problem(head: Any) -> str | None:
    """Why a campaign header issues no study-3 cell (None: it does)."""
    if not isinstance(head, Mapping) or head.get("study") != 3:
        return f"its campaign header is not a study-3 header ({head!r})"
    if head.get("candidate") not in (1, 2) or type(head.get("candidate")) is not int:
        return f"its campaign header names no candidate 1 or 2 ({head.get('candidate')!r})"
    if head.get("step") not in STEP_ROLES:
        return f"its campaign header names no study-3 step ({head.get('step')!r})"
    round_ = head.get("round")
    if round_ is not None and (type(round_) is not int or round_ < 0):
        return f"its campaign header's round is {round_!r}"
    if head["step"] in VERSIONED_STEPS:
        sha = head.get("versions_record_sha256")
        if not (isinstance(head.get("versions_record"), str) and head["versions_record"]
                and isinstance(sha, str) and len(sha) == 64):  # fmt: skip
            return "its campaign header names no versions record with its sha256"
    return None


def _config(run_dir: Path, start: Mapping[str, Any], chash: str) -> Any:
    """The run's config, whose hash is its cell's: its ``config.yaml``, or without one its
    start's command loaded again; or why there is none."""
    path = run_dir / "config.yaml"
    if path.is_file():
        try:
            cfg = OmegaConf.load(path)
            found = config_hash(cfg)
        except Exception as exc:  # an unreadable config binds nothing
            return f"its config.yaml cannot be read ({type(exc).__name__})"
        if found != chash:
            return f"its config.yaml's config hash {found!r} is not its cell's {chash!r}"
        return cfg
    command = start.get("command")
    try:
        if not (isinstance(command, list) and len(command) >= 2
                and all(isinstance(c, str) for c in command)):  # fmt: skip
            raise ValueError("no command")
        cfg = load_run_config(command[1], command[2:])
        found = config_hash(cfg)
    except Exception as exc:  # no config to bind it: refused
        return (f"no config.yaml, and its start's command gives no config hash "
                f"({type(exc).__name__})")  # fmt: skip
    if found != chash:
        return (f"no config.yaml, and its start's command gives the config hash {found!r}, not "
                f"its cell's {chash!r}")  # fmt: skip
    return cfg


# ------------------------------------------------------------------ attempts


def _select(
    attempts: list[_Attempt],
    recoveries: Sequence[tuple[str, int, Mapping[str, Any]]],
    voided: Mapping[str, tuple[str, str]],
) -> None:
    """Mark the attempts that no longer count (module docstring, "Attempts")."""
    by_label: dict[tuple[str, str], list[_Attempt]] = defaultdict(list)
    for attempt in sorted(attempts, key=lambda a: (a.campaign, a.position)):
        by_label[(attempt.campaign, attempt.cell.label)].append(attempt)
        if attempt.run_id in voided:
            attempt.state, reason = voided[attempt.run_id]
            attempt.reasons.append(reason)
    for campaign, position, record in recoveries:
        labels = record.get("runs")
        for label in labels if isinstance(labels, list) else []:
            earlier = [a for a in by_label.get((campaign, label), []) if a.position < position]
            if not earlier:
                continue
            latest = earlier[-1]  # a round reruns the label's latest attempt
            if latest.state is None and not (latest.run_dir / "terminal.json").is_file():
                latest.state = "recovered"
                latest.reasons.append(f"a recovery round followed it: {record.get('reason')}")


# ------------------------------------------------------------------ outcomes


def _endpoint(attempt: _Attempt, end: Mapping[str, Any] | None,
              theta0: Mapping[str, float]) -> Endpoint:  # fmt: skip
    cell = attempt.cell
    common = {"run_id": attempt.run_id, "run_dir": attempt.run_dir, "cell": cell,
              "arm": cell.arm, "lr": cell.rate, "seed": cell.seed, "role": attempt.role,
              "candidate": cell.candidate, "k_only": attempt.k_only,
              "attempt": attempt.number, "label": cell.label}  # fmt: skip
    if attempt.state is not None:
        return Endpoint(**common, state=attempt.state, reasons=tuple(attempt.reasons))
    return Endpoint(**common, **_outcome(attempt, end, theta0))


def _outcome(attempt: _Attempt, end: Mapping[str, Any] | None,
             theta0: Mapping[str, float]) -> dict[str, Any]:  # fmt: skip
    """A counting attempt's state, reasons and endpoint (module docstring, "States")."""
    run_dir = attempt.run_dir
    ended = None if end is None else end.get("outcome")
    if not (run_dir / "terminal.json").is_file():
        status = (_json(run_dir / "meta.json") or {}).get("status")
        finished = status not in (None, "running")  # failed, or ended without its outcome
        if ended in (None, "stopped") and not finished:
            return {"state": "pending", "reasons": ("no published terminal record yet",)}
        reasons = [f"the launcher's end record: {ended}"] if ended not in (None, "stopped") else []
        reasons += [f"meta.json: status {status}"] if finished else []
        return {"state": "technical", "reasons": (*reasons, "no published terminal record")}
    if ended not in SETTLED_ENDS:  # the launcher's verdict; its free text is not copied
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
            return {**out, "state": "unusable", "reasons": (f"K {k!r} is not positive",)}
        return {**out, "state": "usable", "reasons": ()}
    out["K_harmful"] = float(row["K_harmful"])
    if mmlu > GROSS_COLLAPSE_PP + TOLERANCE_PP:
        return {**out, "state": "unusable",
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
    wanted = [UPDATES] if stop == "steps" else [t for t in (UPDATES,) if t < steps]
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
    problems += _finiteness("training metrics", training, None)
    problems += _finiteness("monitor readings", readings, last)
    problems += _finiteness("logged metrics", [row for row in metrics if "loss" not in row],
                            None if last is None else last - 1)  # fmt: skip
    descriptive = UTILITY_KEYS if attempt.k_only else ()  # amendment 1: decides nothing
    problems += _finiteness("evaluations", [{k: v for k, v in row.items() if k not in descriptive}
                                            for row in evaluations], None)  # fmt: skip
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


def _select_key(cfg: Any, key: str) -> Any:
    """A config value, None when absent or unset."""
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
