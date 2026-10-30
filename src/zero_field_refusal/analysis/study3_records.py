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
- ``EndpointReader``: one snapshot of the same records, read under any number of prescriptions
  (``read_endpoints`` is one reading of a fresh snapshot): every file is read once; its
  ``resolved`` lists the journal's resolved torn lines, each with its kind. With a journal
  ``prefix`` (an engineering cycle's pinned prefix, ``study3_cycle``), each launch log is read up
  to its pinned number of lines (``prefixed``) and a run folder whose start lies beyond the
  prefix is no run of that view: the records as they stood when the cycle was written.
- ``read_holds``: every held launcher run, by its label under any config hash, with the records
  that hold it and their sha256 (what a closure record cites).
- ``journal_lines``: every launch log, line by line, each line with its own sha256 and, for a
  torn line, its kind and its standing (the last of its log, resolved or unresolved): the one
  reading of a cited journal line (``read_holds``, the closure records of ``study3_decisions``)
  and the one rule for what is corrupt (an unresolved torn line before the last of its log), for
  every reader of the journal here, ``study3_versions``' and the qualification's.
- ``log_lines``: a log's bytes line by line, as the launcher reads them (``launch``'s
  ``_journal_lines``, mirrored: this module loads no GPU path).
- ``SEEDS`` and ``candidate_of``: the registered seeds per role and candidate.
- ``step_key``: the key of a step's prescribed cells, ``c<candidate>/<step>[/r<round>]``.
- ``read_theta0``: θ₀'s accuracies, the utility losses' baseline (study 2's θ₀ evaluation).

**The decision records' folder** (``RUNS_DIR/study3/decisions/``, ``DECISIONS``) lies in the
experiment's folder and is reserved: it is no run directory (no run id is ever named so: they are
``<stamp>-<cell>-s<seed>-<hash>-a<attempt>``), and every walk of the experiment's folder skips it.

**The sealing boundary.** Of a run directory only ``config.yaml``, ``meta.json``,
``terminal.json``, ``summary.json``, ``evaluations.jsonl``, ``metrics.jsonl`` and
``monitor.jsonl`` are read (``RECORD_FILES``); nothing under ``responses/`` or ``observations/`` is
ever opened, so no response text, harm label or judge output enters a selection ("Seeds and
order": harm labels never enter a selection). Each record file is checked by the one sealed-path
rule (``under_sealed``, which ``study3_decisions`` and the scripts use too) before it is opened,
with the run's folder as the base, and the run's folder itself with the experiment's: a record
file or a run folder that resolves, through a link, under a ``responses/`` or ``observations/``
folder is never opened, and its attempt is refused by name before anything of the run is read.
A run folder is a plain child of the experiment's folder: a start whose run id is not a plain
name (``plain_name``: a separator, ``.``, ``..``, empty) binds no run, a refusal by name, and
nothing under it is opened. A path the rule cannot resolve (a link loop) counts as sealed, fail
closed, and its refusal says so (``LOOP_WHY``: it cannot be resolved), not a sealed folder. θ₀'s
folder must have no ``responses`` or ``observations`` component, and its ``summary.json`` is
checked with that folder as the base (``read_theta0`` refuses it otherwise, unopened). Every
other file read under RUNS_DIR, a launch log or a record of the qualification (a restart or
supersession record), is checked by the same rule with RUNS_DIR as the base: one that resolves
under a ``responses/`` or ``observations/`` folder of any run is never opened, and is a problem
named by its path (it blocks: what it holds cannot be shown). The structural check of the
responses that the launcher makes (``stage_a.load_trajectory``) is not repeated here; the
launcher's verdict on it is its end record, whose outcome is read and whose free text is not
copied.

**A start's command** is composed again (when the run has no readable ``config.yaml``) only from
a config the launcher composes study 3's runs from, ``experiments/study3`` or
``experiments/study3_k_only`` (``STUDY3_CONFIGS``), and only with section overrides that name a
config file by its plain name (``optimizer=adamw``). Any other config name (another config, a path
with ``..``, an absolute path, another folder) or override gives no config, its reason named, and
nothing of it is loaded.

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
From the same ``meta.json`` the endpoint keeps the commit its run ran and whether its checkout
was dirty (``git``, the tracker's ``git.commit`` and ``git.dirty``), which
``study3_versions.uncounted_runs`` compares with the commit the header's versions record pins.
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
"Voiding and supersession"). The journal is every log under ``launch/`` read as one, each log
as the launcher reads it (``log_lines``: its bytes split at ``\n`` alone, each line decoded
strictly as UTF-8): no rule depends on a log's name or a line's place in it, but for a *torn
line*, one that holds no record: it does not decode or parse (a byte-order mark included), it is
a JSON value that is no object or a blank line, or it is a JSON object whose identity
(``IDENTITY_KEYS``: ``event``, ``run``, ``experiment``, ``run_id``, ``config_hash``,
``campaign``, ``log``, ``sha256``, ``cell``, ``only``) is not all strings, as
``launch.torn_lines`` lists them. The log's last
line, with no newline after it (cut short by a crash), is skipped silently: the launcher's rule.
One before it is *resolved* when a ``torn`` record of its log names its offset, length and sha256
and a ``torn_resolved`` record names its log, offset and sha256 (``launch.resolve_torn``,
``scripts/launch.py --resolve-torn``: the operator restored what its kind lost, docs/operations.md,
"Launching runs"): it is then listed with its kind (``EndpointReader.resolved``) and is no
problem. Any other is a problem, named by its log and line number, never by its bytes. The
``torn`` and ``torn_resolved`` records are no problem and void nothing. An attempt is a start;
its campaign is the log that holds its start, and the attempts of a label are its starts in any
log, in the order of their attempt numbers (then of their starts' time stamps). Ends,
``superseded`` and ``held`` lines name their attempt by run id, in any log: the end the
launcher writes after the fact for a start it found without one (``"settled": true``) included,
and an end ``done`` is the outcome whatever stop its terminal record holds. An attempt that no
longer counts is listed with its reason, never counted:
- ``superseded``: named by a ``superseded`` line, by a case restart
  (``c<N>/<stage>/<case>/restart-<n>.json``, its ``launcher``: its own case's launcher run
  alone, a start of the label its case's issued run has, under candidate N, in the
  qualification's experiment, ``qualification_records.restart_checks``; any other is a problem
  and voids nothing) or by an engineering cycle's supersession (``c<N>/supersede-<n>.json``, its
  ``launchers``);
- ``held``: named by a ``held`` line or by a supersession's ``held``: an outcome the engineering
  cycle could not resolve, never attempted again (the launcher refuses its label under any
  config hash: ``read_holds``);
- a supersession record supersedes or holds only once the one check accepted it
  (``qualification_records.cycle_checks``, loaded only when a record is there); every problem
  of the check is a problem here;
- a journal line alone never takes a run out of the counts. A ``superseded`` or ``held`` line
  is study 3's by the run it names (a study-3 start), whatever its own ``experiment`` field, and
  it counts only with study 3's experiment, its run's label and a record that names its run
  (``_unbacked``): an accepted supersession record that names it under ``launchers`` or
  ``held``, or, for a ``superseded`` line, a readable case restart record of a candidate's
  qualification folder whose launcher run it is, a start of its own case (``qualify.py``
  publishes its records before their lines). Any other study-3 line counts for nothing and is a
  problem by its log and line, never by its free text; a line that names no study-3 start takes
  no study-3 run out of the counts and is no problem;
- ``recovered``: an attempt without a published terminal record that a recovery round of its
  label follows (``--recovery REASON``, its ``runs``, in any log; ``round_follows``, the one rule
  ``study3_decisions`` reads too): the documented interruption supersedes it, and a later
  attempt decides; the first round that follows it names it, with its number and reason. A round
  written since rounds are numbered (``launch``, "Recovery rounds": a run's own round, one per
  documented incident, the label's first allowing the smaller of its ``max_attempts`` and the
  label's base, the first start's ``max_attempts`` under its config hash, and each further one
  exactly one) follows exactly the attempts its ``after`` names. A round without ``after``
  (written before) follows the label's attempts numbered up to the highest number started before
  its time stamp (the launcher numbers a run's attempts in order, and logs a round before the
  attempt it allows). The rounds of a label are its own (``only``) and its cell's, under any
  config hash (``label_rounds``); those that apply to a run, its cell's and its own under its
  config hash or none (``RecoveryRound.applies``); what the launcher would still allow a run,
  ``attempts_left`` (the launcher's rule, mirrored: grants by ``round_grant``, never read from
  ``granted``), the one helper ``study3_progress`` reads too. A numbered round (one with any of
  ``NUMBERED_KEYS``) that does not read as the launcher writes it is a problem, named by its log
  and line, never by its reason: one without its number, its ``after`` list, ``config_hash``,
  ``max_attempts`` or a reason that is not blank; without its incident (``incident_time`` in
  UTC and not later than the round's own stamp, ``cause`` and ``evidence``, none blank, each on
  one line); round 1 whose ``granted`` is not the smaller of its ``max_attempts`` and the
  label's base; a round numbered 2 or more without ``only``, or not at ``granted`` and
  ``max_attempts`` 1; an ``after`` that names a run id that is no start of its label, leaves out
  a start of the label stamped before the round, or names an attempt, not superseded, whose end
  records done, diverged or conflict; a label's rounds not numbered 1, 2, … in time order (ties
  by number); a numbered round that gives the reason of an earlier round of its label (spaces
  stripped). Rounds written before rounds were numbered read as before: they raise no problem
  of their own, and among them the first in file order names an attempt they follow.
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
  start is taken to follow the round, as the launcher logs a round before the attempt it allows;
  this applies to a round without ``after``.
- *A round's place is its ``after`` list* when it has one: the attempts it names, whatever their
  time stamps; its time stamp only places the attempts it leaves out (a problem when stamped
  before it).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from omegaconf import OmegaConf

from ..config import config_hash
from ..protocol import load_run_config

EXPERIMENT = "study3"
DECISIONS = "decisions"  # RUNS_DIR/study3/decisions/: the decision records, reserved, no run
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
# The sealing boundary: a run's folders of responses, labels and judge outputs, never opened; and
# the run's record files, the only ones read
SEALED = ("responses", "observations")
RECORD_FILES = ("config.yaml", "meta.json", "terminal.json", "summary.json", "evaluations.jsonl",
                "metrics.jsonl", "monitor.jsonl")  # fmt: skip
SEALED_WHY = "under a responses or observations folder"
LOOP_WHY = "cannot be resolved (a link loop)"  # a path the rule counts as sealed: why
# The configs the launcher composes study 3's runs from (a start's command, composed again)
STUDY3_CONFIGS = ("experiments/study3", "experiments/study3_k_only")
# launch.TORN_KIND: the one part of a torn line's bytes a record copies, its kind (the launcher
# itself is a GPU path, never imported here)
TORN_KIND = re.compile(rb'"event": "([a-z][a-z_]{0,31})"')
# launch.IDENTITY_KEYS: a journal record's identity, each key absent or a string; a JSON object
# with any of them present and not a string is a torn line, as the launcher reads it
IDENTITY_KEYS = ("event", "run", "experiment", "run_id", "config_hash", "campaign", "log",
                 "sha256", "cell", "only")  # fmt: skip
# launch.INCIDENT_KEYS: the incident a run's own recovery round documents
INCIDENT_KEYS = ("incident_time", "cause", "evidence")
# The fields of a recovery round written since rounds are numbered (``launch``, "Recovery
# rounds"): a round with any of them is read as one, and needs them all
NUMBERED_KEYS = ("round", "after", "granted", "max_attempts", "config_hash", *INCIDENT_KEYS)
# launch.MAX_ATTEMPTS: a start's attempts when it records no max_attempts
MAX_ATTEMPTS = 3
# The ends after which the launcher opens no round over an attempt (``launch``, "Recovery rounds")
HELD_ENDS = ("done", "diverged", "conflict")


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
    # The commit its run ran and whether its checkout was dirty, as its meta.json's ``git``
    # records them (``commit``, ``dirty``: None where not recorded as a string or a bool); None
    # without a readable meta.json
    git: tuple[str | None, bool | None] | None = None

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
    git: tuple[str | None, bool | None] | None = None  # meta.json's commit and dirty flag


@dataclass(frozen=True)
class RecoveryRound:
    """A study-3 ``recovery`` line of the launch journal (``launch.Round``, mirrored: the
    launcher is a GPU path, never imported here): its log's path under RUNS_DIR, its line number
    and its record. A cell's round has no ``only``; a run's own round names its label there, and
    one written since rounds are numbered carries ``NUMBERED_KEYS`` (module docstring,
    "Attempts")."""

    path: str
    line: int
    record: Mapping[str, Any] = field(hash=False)

    @property
    def where(self) -> str:
        return f"{self.path} line {self.line}"

    @property
    def time(self) -> datetime | None:
        return _time(self.record.get("time"))

    @property
    def only(self) -> Any:
        return self.record.get("only")

    @property
    def cell(self) -> Any:
        return self.record.get("cell")

    @property
    def number(self) -> int | None:
        value = self.record.get("round")
        return value if type(value) is int and value >= 1 else None

    @property
    def numbered(self) -> bool:
        """Written since rounds are numbered: it carries any of ``NUMBERED_KEYS``."""
        return any(key in self.record for key in NUMBERED_KEYS)

    @property
    def after(self) -> tuple[str, ...] | None:
        value = self.record.get("after")
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(value)
        return None

    @property
    def labels(self) -> list[str]:
        """The labels its ``runs`` names."""
        runs = self.record.get("runs")
        return [x for x in runs if isinstance(x, str)] if isinstance(runs, list) else []

    @property
    def reason(self) -> str | None:
        reason = self.record.get("reason")
        return reason.strip() if isinstance(reason, str) else None

    def of_label(self, label: str) -> bool:
        """A round of ``label``: its own (``only``), or its cell's (no ``only``)."""
        if self.only is not None:
            return self.only == label
        return isinstance(label, str) and "/s" in label and self.cell == label.rsplit("/s", 1)[0]

    def applies(self, label: str, chash: Any) -> bool:
        """A round that applies to the run of ``label`` under config hash ``chash``, as the
        launcher counts it (``launch.recovery_rounds``): its cell's, or its own whose config
        hash is absent or ``chash``."""
        return self.of_label(label) and (
            self.only is None or self.record.get("config_hash") in (None, chash)
        )


def round_order(item: RecoveryRound) -> tuple:
    """The order of a label's rounds, as the launcher numbers them: their time stamps, then
    their numbers, then their logs and lines."""
    return (str(item.record.get("time") or ""), item.number or 0, item.path, item.line)


def label_rounds(rounds: Sequence[RecoveryRound], label: str) -> list[RecoveryRound]:
    """The rounds of ``label`` (``launch.label_rounds``, mirrored): its cell's and its own,
    whatever their config hash, in ``round_order``."""
    return sorted((x for x in rounds if x.of_label(label)), key=round_order)


def round_follows(
    record: Mapping[str, Any],
    run_id: Any,
    number: Any,
    started: datetime | None,
    starts: Sequence[tuple[Any, datetime | None]],
) -> bool | None:
    """Whether a recovery round (its record) follows an attempt of its label (module docstring,
    "Attempts"): the one rule of every reader here, ``study3_decisions``' included. A round
    with ``after`` follows exactly the attempts it names (by run id); one without (written
    before rounds were numbered) follows the attempts numbered up to the highest number started
    before its time stamp (``starts``: the label's attempts' numbers and start times; a start in
    the round's second follows it). None when it cannot be placed: an ``after`` that is no list
    of run ids or, without one, a time stamp (the round's or the attempt's) or a number that
    cannot be read."""
    if "after" in record:
        after = record["after"]
        if not (isinstance(after, list) and all(isinstance(item, str) for item in after)):
            return None
        return run_id in after
    when = _time(record.get("time"))
    if when is None or started is None or type(number) is not int:
        return None
    before = [n for n, at in starts if type(n) is int and at is not None and at < when]
    return bool(before) and number <= max(before)


def _starts_of(starts: Sequence[Mapping[str, Any]], label: Any, chash: Any) -> list[Mapping]:
    """The starts of ``label`` under config hash ``chash`` (with a run id), in attempt order."""
    found = [s for s in starts if s.get("run") == label and s.get("config_hash") == chash
             and isinstance(s.get("run_id"), str)]  # fmt: skip
    return sorted(found, key=_attempt_key)


def base_attempts(starts: Sequence[Mapping[str, Any]], label: Any, chash: Any,
                  max_attempts: Any) -> int:  # fmt: skip
    """The label's base (``launch.base_attempts``, mirrored): the ``max_attempts`` its first
    start under ``chash`` recorded, ``MAX_ATTEMPTS`` when it records none; ``max_attempts``
    before any start."""
    mine = _starts_of(starts, label, chash)
    first = mine[0].get("max_attempts", MAX_ATTEMPTS) if mine else max_attempts
    return first if type(first) is int and first >= 1 else MAX_ATTEMPTS


def round_grant(item: RecoveryRound, rounds: Sequence[RecoveryRound], base: int) -> int:
    """What a round of a run's own allows, by the launcher's rule (``launch.round_grant``,
    mirrored), never read from its ``granted``: the label's first round (no round of the label
    before it in ``rounds``) the smaller of the ``max_attempts`` it was opened with (1 when it
    records none) and the label's base; every further round exactly 1."""
    earlier = next((n for n, x in enumerate(rounds) if (x.path, x.line) == (item.path, item.line)),
                   len(rounds))  # fmt: skip
    if earlier:
        return 1
    asked = item.record.get("max_attempts")
    return min(asked if type(asked) is int and asked >= 1 else 1, base)


def attempts_left(
    rounds: Sequence[RecoveryRound],
    starts: Sequence[Mapping[str, Any]],
    label: str,
    chash: Any,
    max_attempts: int = 1,
) -> int:
    """How many attempts the launcher would still make for the run of ``label`` under config
    hash ``chash`` (``launch.attempts_left``, mirrored: the one helper of every reader here),
    at ``max_attempts`` (1, the command a way out gives): the latest round of its own that
    applies and names the attempts it follows grants by rule (``round_grant``) beyond them;
    without one, the base (``max_attempts`` and the label's, whichever is less) doubled by one
    round in all, a round of its own written before rounds were numbered or its cell's round (none
    for a label that had a round of its own). The attempts used are its starts under ``chash``."""
    mine = _starts_of(starts, label, chash)
    of_label = label_rounds(rounds, label)
    applying = [x for x in of_label if x.applies(label, chash)]
    anchored = [x for x in applying if x.only is not None and x.after is not None]
    first = base_attempts(starts, label, chash, max_attempts)
    if anchored:
        latest = anchored[-1]
        spent = {s["run_id"] for s in mine} - set(latest.after or ())
        return max(0, round_grant(latest, of_label, first) - len(spent))
    owned = any(x.only is not None for x in of_label)
    doubled = any((x.only is not None and not x.numbered) or (x.only is None and not owned)
                  for x in applying)  # fmt: skip
    used = len({s["run_id"] for s in mine})
    return max(0, min(max_attempts, first) * (2 if doubled else 1) - used)


@dataclass
class _Journal:
    """The study-3 records of every log; nothing here depends on a log's name or a line's
    place."""

    starts: list[tuple[str, dict[str, Any]]] = field(default_factory=list)  # campaign, start
    ends: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    recoveries: list[RecoveryRound] = field(default_factory=list)
    corrupt: list[str] = field(default_factory=list)  # "<log> line <n>": an unresolved torn line
    resolved: list[dict[str, Any]] = field(default_factory=list)  # the resolved torn lines
    sealed: list[str] = field(default_factory=list)  # logs under a sealed folder, never opened
    unbacked: list[str] = field(default_factory=list)  # lines that count for nothing: why
    later: set[str] = field(default_factory=set)  # run ids started beyond a pinned prefix


def step_key(candidate: int, step: str, round_: int | None = None) -> str:
    """``c<candidate>/<step>``, with ``/r<round>`` when the header names a round."""
    return f"c{candidate}/{step}" + ("" if round_ is None else f"/r{round_}")


def candidate_of(seed: int, role: str) -> int | None:
    """The candidate whose registered seeds of ``role`` hold ``seed``; None when none does, or
    when both do (confirmation: its campaign's header binds the candidate)."""
    owners = [c for c, seeds in SEEDS.get(role, {}).items() if seed in seeds]
    return owners[0] if len(owners) == 1 else None


def under_sealed(path: Path, base: Path, named: Path | None = None) -> bool:
    """Whether ``path`` (as written, ``named``, or where its links lead) is under a ``responses``
    or ``observations`` folder, below ``base`` (a folder of the base itself does not count).
    Decided from the names alone: nothing is opened. A path that cannot be resolved (a link
    loop) counts as sealed: where it leads cannot be shown. The one sealed-path rule of study 3's
    analysis."""
    try:
        resolved, root = Path(path).resolve(), Path(base).resolve()
    except (OSError, RuntimeError):  # a link loop
        return True
    below = resolved.relative_to(root) if resolved.is_relative_to(root) else resolved
    return any(part in SEALED for part in (*(named.parts if named else ()), *below.parts))


def unresolvable(path: Path, base: Path | None = None) -> bool:
    """Whether ``path`` (or ``base``) cannot be resolved: a link loop, which the sealed-path rule
    counts as sealed (``under_sealed``). Nothing is opened."""
    try:
        Path(path).resolve()
        if base is not None:
            Path(base).resolve()
    except (OSError, RuntimeError):
        return True
    return False


def sealed_why(path: Path, base: Path | None = None, folder: str = f"resolves {SEALED_WHY}") -> str:
    """Why the sealed-path rule refuses ``path`` (with ``base``): it cannot be resolved (a link
    loop, ``LOOP_WHY``), or it lies under a sealed folder (``folder``, the wording of that
    case)."""
    return LOOP_WHY if unresolvable(path, base) else folder


class SealedRecord(PermissionError):
    """A run's record file that resolves under a sealed folder (or cannot be resolved): refused,
    never opened."""


def sealed_record(what: str, path: Path | None = None, base: Path | None = None) -> str:
    """The problem that names a record under RUNS_DIR (``what``: "a launch log", say) refused by
    the rule: it resolves under a sealed folder, or ``path`` cannot be resolved (a link loop),
    and it is never opened."""
    why = f"resolves {SEALED_WHY}" if path is None else sealed_why(path, base)
    return f"{what} that {why}: never opened"


def plain_name(value: Any) -> bool:
    """A single path component: a string that names nothing in another folder (no separator, not
    empty, ``.`` or ``..``). The one rule for a run id, an experiment's name and a config file
    named by an override."""
    return (isinstance(value, str) and value not in ("", ".", "..") and "\\" not in value
            and Path(value).name == value)  # fmt: skip


def _sealed_folder(folder: Path, base: Path | None) -> bool:
    """Whether a run's folder resolves under a sealed folder: against its parent, and against
    ``base`` (the experiment's folder, or the runs root) when given."""
    return under_sealed(folder, folder.parent) or (base is not None and under_sealed(folder, base))


def _sealed_file(path: Path, base: Path | None = None) -> bool:
    """Whether a run's record file (``<run folder>/<name>``) resolves under a sealed folder: the
    file with its run's folder as the base, or the run's folder itself (a link) with the
    experiment's folder (``base``, or the folder's parent) as the base."""
    folder = path.parent
    return under_sealed(path, folder) or _sealed_folder(folder, base)


def _sealed_records(run_dir: Path, base: Path | None = None) -> str | None:
    """Why nothing of a run is read: its folder (checked against ``base``, the experiment's
    folder, as well as its parent), or record files of it, resolve under a sealed folder (named);
    None when none does."""
    if _sealed_folder(run_dir, base):
        return f"its run folder {sealed_why(run_dir, base)}: nothing of it is opened"
    found = [name for name in RECORD_FILES if under_sealed(run_dir / name, run_dir)]
    if not found:
        return None
    loops = [name for name in found if unresolvable(run_dir / name)]
    sealed = [name for name in found if name not in loops]
    out = []
    if sealed:
        verb = "resolves" if len(sealed) == 1 else "resolve"
        out.append(f"its {_listed(sealed)} {verb} {SEALED_WHY}: never opened")
    if loops:
        out.append(f"its {_listed(loops)} {LOOP_WHY}: never opened")
    return "; ".join(out)


def _listed(names: Sequence[str]) -> str:
    """``a``, or ``a, b and c``."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def read_theta0(path: str | Path) -> dict[str, float]:
    """θ₀'s accuracies from its evaluation's ``summary.json`` (the run directory or the file):
    each a finite fraction in [0, 1]; ``ValueError`` otherwise, and, unopened, for a θ₀ folder
    whose resolved path has a ``responses`` or ``observations`` component or a file that resolves
    under a sealed folder below it."""
    path = Path(path)
    folder = path if path.is_dir() else path.parent
    file = folder / "summary.json" if path.is_dir() else path
    try:
        parts = folder.resolve().parts
    except (OSError, RuntimeError):  # a link loop
        parts = SEALED
    if any(part in SEALED for part in parts) or under_sealed(file, folder):
        why = LOOP_WHY if unresolvable(file, folder) else f"resolves {SEALED_WHY}"
        raise ValueError(f"θ₀: {file} {why}: never opened")
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
    return EndpointReader(runs_root, theta0=theta0).read(prescribed)


class EndpointReader:
    """One snapshot of the run records under ``runs_root``: θ₀'s evaluation, the launch journal,
    the qualification's restart and supersession records, the experiment's folders and, as they
    are first needed, each run's files. ``read(prescribed)`` gives what ``read_endpoints`` gives
    under ``prescribed``; any number of readings read every file once (a caller that derives the
    cells to prescribe from the records reads them once). ``resolved``: the journal's resolved
    torn lines (module docstring, "Attempts"), each its log's ``path`` under RUNS_DIR, its
    ``line`` number, its ``offset``, ``length`` and ``sha256`` and its ``kind``; listed, no
    problem."""

    def __init__(
        self,
        runs_root: str | Path,
        *,
        theta0: str | Path,
        cycles: Mapping[str, Mapping[str, Any]] | None = None,
        restarts: Mapping[str, Mapping[str, Any]] | None = None,
        prefix: Mapping[str, int] | None = None,
    ) -> None:
        self.root = Path(runs_root)
        self.baseline = read_theta0(theta0)
        self.voided: dict[str, tuple[str, str]] = {}  # run id -> superseded or held, and why
        cycles = _cycle_checks(self.root, cycles)
        restarts = _restart_checks(self.root, restarts)
        self.journal = _read_journal(self.root, self.voided, cycles, restarts, prefix)
        self.problems = _voiding_records(self.root, self.voided, cycles, restarts)
        self.problems += [f"{where}: {TORN_WHY}" for where in self.journal.corrupt]
        # the resolved torn lines, each with its kind: listed, no problem
        self.resolved = list(self.journal.resolved)
        self.problems += [f"{where}: {sealed_record('a launch log', self.root / where)}"
                          for where in self.journal.sealed]  # fmt: skip
        self.problems += self.journal.unbacked
        experiment = self.root / EXPERIMENT
        # a run folder whose start lies beyond a pinned prefix is a later run: no run of the view
        self.folders = [d for d in (sorted(experiment.iterdir()) if experiment.is_dir() else [])
                        if d.is_dir() and d.name != DECISIONS
                        and d.name not in self.journal.later]  # fmt: skip
        self.files = _Files(experiment)

    def read(
        self, prescribed: Mapping[str, Collection[tuple[str, float, int]]]
    ) -> tuple[list[Endpoint], list[str]]:
        """``read_endpoints`` under ``prescribed``, from this snapshot."""
        problems = list(self.problems)
        started = Counter(start["run_id"] for _, start in self.journal.starts
                          if isinstance(start.get("run_id"), str))  # fmt: skip
        attempts: list[_Attempt] = []
        for campaign, start in self.journal.starts:
            if not isinstance(start.get("run_id"), str):
                problems.append(f"{campaign}: a start in the launch journal without a run id")
                continue
            attempts.append(_bind(self.root, campaign, start, prescribed,
                                  twice=started[start["run_id"]] > 1,
                                  files=self.files))  # fmt: skip
        for run_dir in self.folders:
            if run_dir.name not in started:
                attempts.append(_Attempt(
                    campaign=None, started=None, run_id=run_dir.name, run_dir=run_dir,
                    cell=Cell(*[None] * 9), number=None, state="refused",
                    reasons=["a study-3 run directory with no start in the launch journal: no "
                             "issued cell"]))  # fmt: skip
        problems += _select(attempts, self.journal.recoveries, self.voided)
        problems += _round_problems(self.journal.recoveries, self.journal.starts,
                                    self.journal.ends, self.voided)  # fmt: skip
        _refuse_duplicates(attempts)
        problems += [f"{a.run_id}: {'; '.join(a.reasons)}" for a in attempts
                     if a.state == "refused"]  # fmt: skip
        endpoints = [_endpoint(a, self.journal.ends.get(a.run_id, []), self.baseline,
                               self.files) for a in attempts]  # fmt: skip
        endpoints.sort(key=_order)
        return endpoints, sorted(set(problems))

    def meta_status(self, run_dir: Path) -> Any:
        """A run's status as its ``meta.json`` records it (None without a readable one), from
        this snapshot."""
        return (_json(run_dir / "meta.json", self.files) or {}).get("status")

    def attempts_left(self, label: str, chash: Any, max_attempts: int = 1) -> int:
        """``attempts_left`` of the run of ``label`` under ``chash``, from this snapshot."""
        starts = [start for _, start in self.journal.starts]
        return attempts_left(self.journal.recoveries, starts, label, chash, max_attempts)

    def outcome_held(self, label: str, chash: Any) -> bool:
        """Whether an attempt of the run of ``label`` under ``chash``, not superseded, has an end
        that records done, diverged or conflict: the launcher opens no round over it."""
        starts = [start for _, start in self.journal.starts]
        return any(self.voided.get(s["run_id"], ("",))[0] != "superseded"
                   and any(end.get("outcome") in HELD_ENDS
                           for end in self.journal.ends.get(s["run_id"], []))
                   for s in _starts_of(starts, label, chash))  # fmt: skip

    def latest_start(self, label: str) -> Mapping[str, Any] | None:
        """The start of the label's latest attempt in any log (by its attempt number, then its
        time stamp, as the launcher orders them), or None."""
        starts = [start for _, start in self.journal.starts if start.get("run") == label]
        return max(starts, key=_attempt_key) if starts else None


class _Files:
    """The run files a snapshot has read: each file's text (or the error reading it raised) and
    each run's bound config, read once."""

    def __init__(self, base: Path | None = None) -> None:
        self.base = base  # the experiment's folder: each run folder is checked against it too
        self.texts: dict[Path, str | Exception] = {}
        self.configs: dict[tuple, tuple[Any, Any, Any]] = {}

    def read_text(self, path: Path) -> str:
        if path not in self.texts:
            try:
                if _sealed_file(path, self.base):
                    raise SealedRecord(f"{path} {sealed_why(path)}: never opened")
                self.texts[path] = path.read_text()
            except (OSError, ValueError) as exc:
                self.texts[path] = exc
        found = self.texts[path]
        if isinstance(found, Exception):
            raise found
        return found

    def config(self, run_dir: Path, start: Mapping[str, Any], chash: str | None) -> tuple:
        key = (run_dir, chash, json.dumps(start.get("command"), sort_keys=True, default=str))
        if key not in self.configs:
            self.configs[key] = _config(run_dir, start, chash, self.base)
        return self.configs[key]


def _order(e: Endpoint) -> tuple:
    """The listing's order: by cell, then by attempt number (the campaign's name only breaks
    ties)."""
    return (e.candidate or 0, e.cell.step or "", -1 if e.cell.round is None else e.cell.round,
            e.arm or "", e.lr or 0.0, e.seed or 0, e.attempt or 0, e.cell.campaign or "",
            e.run_id)  # fmt: skip


# ------------------------------------------------------------------ journal and voiding records


def _read_journal(
    root: Path,
    voided: dict[str, tuple[str, str]],
    cycles: Mapping[str, Mapping[str, Any]] | None = None,
    restarts: Mapping[str, Mapping[str, Any]] | None = None,
    prefix: Mapping[str, int] | None = None,
) -> _Journal:
    """The launch journal's study-3 records from every log (``journal_lines``' reading): the
    starts with their campaign (the log's name), every end by run id, the recovery rounds, and the
    ``superseded`` and ``held`` lines into ``voided`` by run id; with ``cycles`` (the one check of
    every supersession record), a line counts only through a record that names its run
    (``_unbacked``: an accepted supersession record, or for a study-3 ``superseded`` line a case
    restart record whose launcher run it is): any other is listed in ``unbacked``, with why, and
    counts for nothing. A line is study 3's by the run it names (a study-3 start), whatever its
    own ``experiment``; one that names no study-3 start takes no study-3 run out of the counts. A
    torn line (module docstring, "Attempts") holds no record: the last of its log is skipped
    silently (the launcher's rule), a resolved one is listed in ``resolved`` with its kind, and
    any other is listed in ``corrupt`` with its log's name and its number, never its bytes. A
    log that resolves under a sealed folder is never opened: it is listed in ``sealed``, by its
    path under RUNS_DIR. A log that cannot be read raises its ``OSError``. ``prefix``: each log
    read up to its pinned number of lines, a log the map does not name as empty (``prefixed``);
    the run ids started beyond it are listed in ``later``."""
    journal = _Journal()
    lines: list[tuple[str, int, dict[str, Any]]] = []  # the superseded and held lines
    logs, unread = _journal_logs(root)
    for log, error in unread:
        if prefix is not None and f"launch/{log.name}" not in prefix:
            continue
        if error is not None:
            raise error
        journal.sealed.append(f"launch/{log.name}")
    if prefix is not None:
        every = {line.record.get("run_id") for log in logs for line in log.lines
                 if line.record is not None and line.record.get("event") == "start"}  # fmt: skip
        logs = prefixed(logs, prefix)
        within = {line.record.get("run_id") for log in logs for line in log.lines
                  if line.record is not None and line.record.get("event") == "start"}  # fmt: skip
        journal.later = {r for r in every - within if isinstance(r, str)}
    for log in logs:
        campaign = PurePosixPath(log.path).stem
        journal.corrupt += [f"{log.path} line {number}" for number in log.corrupt]
        journal.resolved += log.resolved
        for line in log.lines:
            record, number = line.record, line.line
            if record is None:
                continue
            event, run_id = record.get("event"), record.get("run_id")
            if event == "start" and record.get("experiment") == EXPERIMENT:
                journal.starts.append((campaign, record))
            elif event == "end" and isinstance(run_id, str):
                journal.ends[run_id].append(record)
            elif event == "recovery" and record.get("experiment") == EXPERIMENT:
                journal.recoveries.append(RecoveryRound(log.path, number, record))
            elif event in ("superseded", "held") and isinstance(run_id, str):
                lines.append((log.path, number, record))
    labels = study3_labels(journal.starts)  # every log read: a line's run may start in any
    for where, number, record in lines:
        why = _unbacked(record, cycles, restarts or {}, labels)
        if why is not None:
            journal.unbacked.append(_unbacked_line(where, number, record, why))
            continue
        _void(voided, record["run_id"], record["event"],
              f"{record['event']} in the launch journal: {record.get('reason')}")  # fmt: skip
    return journal


def study3_labels(starts: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, set[Any]]:
    """Each study-3 start's run id, with the labels (strings) its starts name: a ``superseded``
    or ``held`` line is study 3's when it names one of these runs. A start whose label is no
    string names no label (its binding refuses it by name)."""
    out: dict[str, set[Any]] = {}
    for _, start in starts:
        if isinstance(start.get("run_id"), str):
            labels = out.setdefault(start["run_id"], set())
            if isinstance(start.get("run"), str):
                labels.add(start["run"])
    return out


def log_lines(data: bytes) -> list[tuple[int, bytes, dict[str, Any] | None]]:
    """A launch log's bytes line by line, as the launcher reads them (``launch._journal_lines``,
    mirrored: the launcher is a GPU path, never imported here): split at ``\n`` alone, each line's
    offset, its bytes and its record, None (a torn line) when its bytes do not decode strictly as
    UTF-8 (a byte-order mark stays in the text), do not parse as a JSON object, or parse as one
    whose identity (``IDENTITY_KEYS``) is not all strings (``launch.well_formed``)."""
    lines, offset = [], 0
    while offset < len(data):
        end = data.find(b"\n", offset)
        end = len(data) if end < 0 else end
        try:
            record = json.loads(data[offset:end].decode("utf-8"))
        except ValueError:  # a UnicodeDecodeError included
            record = None
        whole = isinstance(record, dict) and all(
            isinstance(record[key], str) for key in IDENTITY_KEYS if key in record)  # fmt: skip
        lines.append((offset, data[offset:end], record if whole else None))
        offset = end + 1
    return lines


def torn_kind(data: bytes) -> str | None:
    """The event a torn line's bytes show through the strict pattern ``TORN_KIND``
    (``launch.torn_kind``), or None; nothing else of the bytes is read out."""
    match = TORN_KIND.search(data)
    return match[1].decode("ascii") if match else None


def _voiding_records(
    root: Path,
    voided: dict[str, tuple[str, str]],
    cycles: Mapping[str, Mapping[str, Any]],
    restarts: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """The launcher runs that the qualification's case restarts and engineering-cycle
    supersessions name (scripts/qualify.py, "A case restart", "Supersession"), into ``voided``:
    superseded or held, with the reason; a supersession record only when the one check accepted
    it (``cycles``: ``qualification_records.cycle_checks``), and a case restart record only for
    its own case's launcher run (``restarts``: ``qualification_records.restart_checks``); every
    problem of either check is one of the returned problems (it blocks at every step). An
    unreadable record is refused: it may name a run; so is one that resolves under a sealed
    folder, never opened."""
    problems: list[str] = []
    out = root / "qualification"
    for path in sorted(out.glob("c*/*/s*/*/restart-*.json")):
        if not _RESTART_NAME.fullmatch(path.name):
            continue  # no case restart record's name: its problem is the check's (``restarts``)
        if under_sealed(path, root):
            problems.append(f"{path}: {sealed_record('a case-restart record', path, root)}")
            continue
        record = _json(path)
        if record is None:
            problems.append(f"{path}: an unreadable case-restart record")
            continue
        launcher = record.get("launcher")
        check = restarts.get(path.relative_to(root).as_posix()) or {}
        mine = check.get("launchers", ()) if check.get("accepted") else ()
        if isinstance(launcher, Mapping) and launcher.get("run_id") in mine:
            _void(voided, launcher["run_id"], "superseded",
                  f"case restart {path}: {record.get('reason')}")  # fmt: skip
    for path in sorted(out.glob("c*/supersede-*.json")):
        where = path.relative_to(root).as_posix()  # named as ``read_holds`` names it: once
        if under_sealed(path, root):
            problems.append(f"{where}: {sealed_record('a supersession record', path, root)}")
            continue
        record = _json(path)
        if record is None:
            problems.append(f"{where}: an unreadable supersession record")
            continue
        if not (cycles.get(where) or {}).get("accepted"):
            continue  # rejected: it voids and holds nothing (its problems, below)
        why = f"engineering cycle {path}: {record.get('diagnosis')}; fix: {record.get('fix')}"
        for state, entries in (("superseded", "launchers"), ("held", "held")):
            for entry in record.get(entries) or []:
                if isinstance(entry, Mapping) and isinstance(entry.get("run_id"), str):
                    reason = why if state == "superseded" else (
                        f"held by {path}: {entry.get('reason')}")  # fmt: skip
                    _void(voided, entry["run_id"], state, reason)
    for check in (*cycles.values(), *restarts.values()):
        problems += [p for p in check.get("problems") or [] if p not in problems]
    return problems


def _restart_checks(
    root: Path, restarts: Mapping[str, Mapping[str, Any]] | None
) -> Mapping[str, Mapping[str, Any]]:
    """The check of every case restart record (``qualification_records.restart_checks``: what it
    voids, its own case's launcher run alone), as given, or read here when there is a record to
    check (that module loads the verdicts, and torch with them, only then)."""
    if restarts is not None:
        return restarts
    if not any((root / "qualification").glob("c*/*/s*/*/restart-*.json")):
        return {}
    from . import qualification_records  # it loads torch (qualification_verdicts): here only

    return qualification_records.restart_checks(root)


def _cycle_checks(
    root: Path, cycles: Mapping[str, Mapping[str, Any]] | None
) -> Mapping[str, Mapping[str, Any]]:
    """The one check of every supersession record (``qualification_records.cycle_checks``), as
    given, or read here when there is a record to check (that module loads the verdicts, and
    torch with them, only then)."""
    if cycles is not None:
        return cycles
    if not any((root / "qualification").glob("c*/supersede-*.json")):
        return {}
    from . import qualification_records  # it loads torch (qualification_verdicts): here only

    return qualification_records.cycle_checks(root)


_CYCLE_PATH = re.compile(r"qualification/[^/]+/supersede-[^/]*\.json")
# A case restart record of a candidate's qualification folder (named exactly c<N>)
_RESTART_PATH = re.compile(r"qualification/c[1-9]\d*/[^/]+/[^/]+/[^/]+/restart-\d+\.json")
_RESTART_NAME = re.compile(r"restart-\d+\.json")  # a case restart record's name (qualify.py's)
_LINE_KEYS = {"superseded": "launchers", "held": "held"}  # what a line's record names it under


def _unbacked(
    line: Mapping[str, Any],
    cycles: Mapping[str, Mapping[str, Any]] | None,
    restarts: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, Collection[Any]],
) -> str | None:
    """Why a ``superseded`` or ``held`` line counts for nothing, or None when it counts: a run
    stops counting only through a record that names it ("Voiding and supersession"; every
    attempt is kept). A line is study 3's by the run it names: one whose run id is a study-3
    start (``labels``: ``study3_labels``), whatever its own ``experiment`` field says. A line
    that names no study-3 start takes no study-3 run out of the counts, and is no problem (the
    launcher, too, applies a line by its experiment and label). A study-3 line counts only when
    its ``experiment`` is study 3's, its label (``run``) is its run's, and a record names its run;
    its ``record`` is a path of the machine that wrote it, read by its trailing components, so
    that a Pod's root resolves:
    - a supersession record (``qualification/<folder>/supersede-*.json``) the one check
      (``cycles``) has an entry for and accepted, that names the run under ``launchers`` (a
      ``superseded`` line) or ``held`` (a ``held`` line); ``qualify.py`` publishes the record
      before its lines;
    - for a ``superseded`` line, a case restart record of a candidate's qualification folder
      (``qualification/c<N>/<stage>/s<seed>/<case>/restart-<n>.json``, the folder named exactly
      ``c<N>``) that is there, is readable, resolves under no sealed folder and voids the line's
      run: its launcher run, a start of its own case (``restarts``:
      ``qualification_records.restart_checks``).
    Without ``cycles``, every line counts. The reason names ids and the paths of records that
    are there, never the line's free text."""
    if cycles is None or line.get("run_id") not in labels:
        return None
    if line.get("experiment") != EXPERIMENT:
        return f"it names a run of {EXPERIMENT}, and its experiment is not {EXPERIMENT}'s"
    if not isinstance(line.get("run"), str) or line["run"] not in labels[line["run_id"]]:
        return "its label is not its run's label"
    cited = line.get("record")
    parts = PurePosixPath(cited.replace("\\", "/")).parts if isinstance(cited, str) else ()
    where = "/".join(parts[-3:])
    if _CYCLE_PATH.fullmatch(where):
        check = cycles.get(where)
        if check is None:
            return ("it cites a supersession record that no check covers (none is there, or it "
                    "lies in no candidate's folder)")  # fmt: skip
        if not check.get("accepted"):
            return f"it cites {where}, a supersession record the check did not accept"
        key = _LINE_KEYS[line["event"]]
        if line.get("run_id") not in (check.get(key) or ()):
            return f"its run is not among the {key} of {where}, the record it cites"
        return None
    if line.get("event") == "held":
        return ("a held line counts only through an accepted supersession record whose held "
                "names its run")  # fmt: skip
    if not isinstance(cited, str):
        return "it cites no record, and a run stops counting only through a record that names it"
    where = "/".join(parts[-6:])
    check = restarts.get(where) if _RESTART_PATH.fullmatch(where) else None
    if check is None:
        return ("it cites no supersession record and no case restart record of a candidate's "
                "qualification folder that is there")  # fmt: skip
    if not check.get("readable"):
        return (f"it cites {where}, a case restart record that cannot be read or resolves under "
                "a sealed folder")  # fmt: skip
    if line.get("run_id") not in (check.get("launchers") if check.get("accepted") else ()):
        return (f"it cites {where}, a case restart record that does not void its run (its "
                "launcher run is another, or no start of its own case)")  # fmt: skip
    return None


def _unbacked_line(log: str, number: int, line: Mapping[str, Any], why: str) -> str:
    """The problem of a line that counts for nothing (``_unbacked``), by its log and line: its
    event and its run id (a plain name), never its free text."""
    run_id = line.get("run_id")
    run = f" of {run_id}" if plain_name(run_id) else ""
    return f"{log} line {number}: a {line.get('event')} line{run} counts for nothing: {why}"


def voided_runs(
    runs_root: str | Path,
    *,
    cycles: Mapping[str, Mapping[str, Any]] | None = None,
    restarts: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """The launcher runs that no longer count, by run id (superseded or held, and why), as
    ``read_endpoints`` reads them: the journal's ``superseded`` and ``held`` lines that count,
    through a record that names their run (``_unbacked``), the case restarts and the accepted
    supersession records (``cycles``:
    ``qualification_records.cycle_checks``, read here by default); and the problems: the
    voiding records that cannot be read, every problem of the one check, and every line that
    counts for nothing."""
    root = Path(runs_root)
    cycles = _cycle_checks(root, cycles)
    restarts = _restart_checks(root, restarts)
    voided: dict[str, tuple[str, str]] = {}
    journal = _read_journal(root, voided, cycles, restarts)
    return voided, [*_voiding_records(root, voided, cycles, restarts), *journal.unbacked]


def _void(voided: dict[str, tuple[str, str]], run_id: str, state: str, reason: str) -> None:
    """Record that an attempt no longer counts: superseded, or held, which prevails."""
    if voided.get(run_id, ("",))[0] != "held":
        voided[run_id] = (state, reason)


def read_holds(
    runs_root: str | Path, *, cycles: Mapping[str, Mapping[str, Any]] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
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
    records that cannot be read (a log or a supersession record, or one that resolves under a
    sealed folder, never opened): a hold they hold cannot be listed. Of a record only its numbers,
    ids, paths, hashes and the hold's reason are kept. A supersession record holds only once the
    one check accepted it (``cycles``: ``qualification_records.cycle_checks``, read here by
    default; its problems are the run records', ``read_endpoints``), and a ``held`` line that
    names a study-3 start holds only with study 3's experiment, its run's label and an accepted
    supersession record whose ``held`` names its run (``_unbacked``): any other is named among
    the problems and holds nothing. A held line that names no study-3 start is listed as the
    launcher reads it (it holds no study-3 run: ``study3_progress`` places holds by run)."""
    root = Path(runs_root)
    cycles = _cycle_checks(root, cycles)
    holds: dict[str, dict[str, Any]] = {}

    def add(entry: Mapping[str, Any], reference: dict[str, Any]) -> None:
        run_id = entry["run_id"]
        hold = holds.setdefault(run_id, {
            "run_id": run_id, "run": entry.get("run"), "experiment": entry.get("experiment"),
            "config_hash": entry.get("config_hash"), "reason": entry.get("reason"),
            "records": []})  # fmt: skip
        hold["records"].append(reference)

    logs, problems = journal_lines(root)
    labels = study3_labels([
        ("", line.record) for log in logs for line in log.lines
        if line.record is not None and line.record.get("event") == "start"
        and line.record.get("experiment") == EXPERIMENT])  # fmt: skip
    for log in logs:
        for line in log.lines:
            record = line.record  # a torn line: read_endpoints names it, unless resolved
            if (record is not None and record.get("event") == "held"
                    and isinstance(record.get("run_id"), str)):  # fmt: skip
                why = _unbacked(record, cycles, {}, labels)  # never through a restart
                if why is not None:
                    problems.append(_unbacked_line(log.path, line.line, record, why))
                    continue
                add(record, {"kind": "journal", "path": log.path, "root": "runs",
                             "line": line.line, "line_sha256": line.line_sha256,
                             "sha256": log.sha256})  # fmt: skip
    for path in sorted((root / "qualification").glob("c*/supersede-*.json")):
        where = path.relative_to(root).as_posix()
        if under_sealed(path, root):
            problems.append(f"{where}: {sealed_record('a supersession record', path, root)}")
            continue
        try:
            data = path.read_bytes()
            record = json.loads(data)
        except (OSError, ValueError):
            problems.append(f"{where}: an unreadable supersession record")
            continue
        entries = record.get("held") if isinstance(record, dict) else None
        if not (cycles.get(where) or {}).get("accepted"):
            entries = None  # rejected: it holds nothing
        reference = {"kind": "supersession", "path": where, "root": "runs",
                     "sha256": hashlib.sha256(data).hexdigest()}  # fmt: skip
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, Mapping) and isinstance(entry.get("run_id"), str):
                add(entry, dict(reference))
    return list(holds.values()), problems


# A torn line's standing (module docstring, "Attempts"): the last of its log, with no newline
# after it (skipped, the launcher's rule); resolved (listed, no problem); unresolved (a problem)
FINAL, RESOLVED, UNRESOLVED = "final", "resolved", "unresolved"
# The one term every reader names an unresolved torn line before the last of its log with
TORN_LINE = ("a torn line (no JSON object, or one whose identity is not all strings) before the "
             "last line of its log that no torn and torn_resolved records both name")  # fmt: skip
TORN_WHY = (f"{TORN_LINE}: what it held cannot be shown (resolve it: docs/operations.md, "
            '"Launching runs")')  # fmt: skip


@dataclass(frozen=True)
class JournalLine:
    """One line of a launch log, as the launcher reads it (``log_lines``): its number from 1, the
    offset of its first byte and its length, the sha256 of its bytes (for a line that decodes,
    the sha256 of its UTF-8 text: its own bytes, which stay when the log grows, and the bytes a
    ``torn`` record names) and its record (None unless a JSON object). A torn line (no record)
    has its ``kind`` (``torn_kind``) and its standing ``torn``: ``final``, ``resolved`` or
    ``unresolved`` (module docstring, "Attempts"); a record has neither. Nothing else of a
    line's bytes is kept."""

    line: int
    offset: int
    length: int
    line_sha256: str
    record: dict[str, Any] | None
    kind: str | None = None
    torn: str | None = None


@dataclass(frozen=True)
class JournalLog:
    """A launch log: its path under RUNS_DIR, its sha256 now and its lines."""

    path: str
    sha256: str
    lines: tuple[JournalLine, ...]

    @property
    def corrupt(self) -> tuple[int, ...]:
        """The numbers of its unresolved torn lines (each a problem: what it held cannot be
        shown)."""
        return tuple(line.line for line in self.lines if line.torn == UNRESOLVED)

    @property
    def refused(self) -> tuple[int, ...]:
        """The numbers of the torn lines the launcher refuses (``launch.torn_lines``: every one
        that is not resolved, the last of the log included)."""
        return tuple(line.line for line in self.lines if line.torn in (UNRESOLVED, FINAL))

    @property
    def resolved(self) -> tuple[dict[str, Any], ...]:
        """Its resolved torn lines, each with its kind: listed, no problem."""
        return tuple({"path": self.path, "line": line.line, "offset": line.offset,
                      "length": line.length, "sha256": line.line_sha256, "kind": line.kind}
                     for line in self.lines if line.torn == RESOLVED)  # fmt: skip


def journal_lines(runs_root: str | Path) -> tuple[list[JournalLog], list[str]]:
    """Every launch log under ``runs_root/launch/``, by its name, line by line as the launcher
    reads it, each line with its own sha256 and its record, and each torn line with its kind and
    its standing (``JournalLine``); and the logs that cannot be read or resolve under a sealed
    folder (never opened), named. The one reading of a cited journal line, and of what is
    corrupt: a log's ``corrupt`` lines."""
    root = Path(runs_root)
    logs, unread = _journal_logs(root)
    problems = [f"{log.relative_to(root).as_posix()}: "
                + (sealed_record("a launch log", log, root) if error is None
                   else "an unreadable launch journal") for log, error in unread]  # fmt: skip
    return logs, problems


def prefixed(logs: Sequence[JournalLog], prefix: Mapping[str, int] | None) -> list[JournalLog]:
    """The logs as a pinned journal prefix reads them (an engineering cycle's, ``study3_cycle``):
    each log the map names (by its path under RUNS_DIR), up to its pinned number of lines; a log
    it does not name reads as empty, left out. Without a map, the logs as they are."""
    if prefix is None:
        return list(logs)
    return [JournalLog(log.path, log.sha256, log.lines[: max(0, int(prefix[log.path]))])
            for log in logs if log.path in prefix]  # fmt: skip


def _journal_logs(root: Path) -> tuple[list[JournalLog], list[tuple[Path, OSError | None]]]:
    """Every launch log under ``root/launch/`` that can be read, each line as the launcher reads
    it (``log_lines``) with its standing as ``launch.torn_lines`` gives it: a ``torn_resolved``
    record of any log read names a line by its log's name, offset and sha256, and a ``torn``
    record names a line of its own log by offset, length and sha256; a torn line is resolved
    only when both name it, and is ``final`` when no newline follows it. And the logs not read,
    in the order of their paths: one that resolves under a sealed folder (None: never opened) or
    that cannot be read (its error)."""
    read, unread = [], []
    for log in sorted((root / "launch").glob("*.jsonl")):
        if under_sealed(log, root):
            unread.append((log, None))
            continue
        try:
            data = log.read_bytes()
        except OSError as exc:
            unread.append((log, exc))
            continue
        read.append((log, data, log_lines(data)))
    resolutions = {(r["log"], r["offset"], r["sha256"]) for _, _, lines in read
                   for _, _, r in lines if r is not None and r.get("event") == "torn_resolved"
                   and isinstance(r.get("log"), str) and _count(r.get("offset"))
                   and isinstance(r.get("sha256"), str)}  # fmt: skip
    logs = []
    for log, data, lines in read:
        named = {(r["offset"], r["length"], r["sha256"]) for _, _, r in lines
                 if r is not None and r.get("event") == "torn" and _count(r.get("offset"))
                 and _count(r.get("length")) and isinstance(r.get("sha256"), str)}  # fmt: skip
        found = []
        for number, (offset, line, record) in enumerate(lines, start=1):
            sha, kind, torn = hashlib.sha256(line).hexdigest(), None, None
            if record is None:
                kind = torn_kind(line)
                if offset + len(line) == len(data):  # no newline after it
                    torn = FINAL
                elif ((offset, len(line), sha) in named
                      and (log.name, offset, sha) in resolutions):  # fmt: skip
                    torn = RESOLVED
                else:
                    torn = UNRESOLVED
            found.append(JournalLine(number, offset, len(line), sha, record, kind, torn))
        logs.append(JournalLog(log.relative_to(root).as_posix(), hashlib.sha256(data).hexdigest(),
                               tuple(found)))  # fmt: skip
    return logs, unread


def _count(value: Any) -> bool:
    """An offset or a length as ``launch`` counts one: an integer, not a bool."""
    return isinstance(value, int) and not isinstance(value, bool)


# ------------------------------------------------------------------ the issued cell


def _bind(
    root: Path,
    campaign: str,
    start: Mapping[str, Any],
    prescribed: Mapping[str, Collection[tuple[str, float, int]]],
    *,
    twice: bool,
    files: _Files | None = None,
) -> _Attempt:
    """The attempt bound to its issued cell (module docstring, "The issued cell"): refused, with
    its refusals as its reasons, when it cannot be; with the reason it can no longer be bound
    (``unbound``) when its config cannot. Reads the run's ``meta.json`` and ``config.yaml``
    only, and nothing of a run whose folder or record files resolve under a sealed folder (a
    refusal)."""
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
    files = files or _Files(root / EXPERIMENT)
    if plain_name(run_id):
        sealed = _sealed_records(run_dir, root / EXPERIMENT)
    else:  # a folder anywhere below the experiment's: never read
        sealed = (f"its run id {run_id!r} is not a plain name: it names no run folder, and "
                  "nothing under it is opened")  # fmt: skip
    if sealed is not None:
        refusals.append(sealed)
    meta = None if sealed else _json(run_dir / "meta.json", files)
    if meta is not None:
        recorded = meta.get("campaign")
        wanted = {"name": campaign, "label": label, "attempt": number, "header": head}
        if not isinstance(recorded, Mapping) or {k: recorded.get(k) for k in wanted} != wanted:
            refusals.append(f"its meta.json records another campaign than its start ({campaign}, "
                            f"{label}, attempt {number})")  # fmt: skip
        if meta.get("config_hash") != chash:
            refusals.append(f"its config hash {meta.get('config_hash')!r} is not its cell's "
                            f"{chash!r}")  # fmt: skip
    cfg, unbound, refusal = (None, None, None) if sealed else files.config(run_dir, start, chash)
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
        step=step if isinstance(step, str) and step in STEP_ROLES else None,
        round=issued.get("round") if type(issued.get("round")) is int else None,
        arm=facts.get("arm"), rate=facts.get("rate"), seed=facts.get("seed"), config_hash=chash,
        versions_record=_text(issued.get("versions_record")),
        versions_record_sha256=_text(issued.get("versions_record_sha256")),
        prescribed=facts.get("prescribed"),
    )  # fmt: skip
    return _Attempt(campaign=campaign, started=started, run_id=run_id, run_dir=run_dir,
                    cell=cell, number=number, role=facts.get("role"),
                    k_only=facts.get("k_only"), every=facts.get("every"), unbound=unbound,
                    state="refused" if refusals else None, reasons=refusals,
                    git=None if meta is None else _git_of(meta))  # fmt: skip


def _git_of(meta: Mapping[str, Any]) -> tuple[str | None, bool | None]:
    """The commit and the dirty flag a run's ``meta.json`` records (the tracker's ``git``:
    ``commit`` and ``dirty``), each None where it is not recorded as a string or a bool."""
    git = meta.get("git")
    git = git if isinstance(git, Mapping) else {}
    commit, dirty = git.get("commit"), git.get("dirty")
    return (commit if isinstance(commit, str) else None,
            dirty if isinstance(dirty, bool) else None)  # fmt: skip


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
    if not isinstance(head.get("step"), str) or head["step"] not in STEP_ROLES:
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


def _config(run_dir: Path, start: Mapping[str, Any], chash: str | None,
            base: Path | None = None) -> tuple[Any, Any, Any]:  # fmt: skip
    """The run's config, whose hash is its cell's: its ``config.yaml``, or without a readable
    one its start's command loaded again (``_command_config``). Returns (config, why it can no
    longer be bound, why it is refused), the config None unless it binds; a ``config.yaml`` that
    resolves under a sealed folder (its run folder checked against ``base``, the experiment's
    folder, too) is refused, never opened."""
    path = run_dir / "config.yaml"
    if _sealed_file(path, base):
        return None, None, f"its config.yaml {sealed_why(path)}: never opened"
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
    unissued = _unissued(command) if _is_command(command) else None
    if unissued is not None:  # never loaded: the reason is named
        return None, f"{unreadable or 'no config.yaml'}, and its start's command {unissued}", None
    try:
        cfg = _command_config(command)
        found = config_hash(cfg)
    except Exception as exc:  # no config to bind it under this checkout's configs
        why = f"its start's command gives no config hash ({type(exc).__name__})"
    else:
        if found == chash:
            return cfg, unreadable, None
        why = f"its start's command gives the config hash {found!r}, not its cell's {chash!r}"
    return None, f"{unreadable or 'no config.yaml'}, and {why}", None


def _is_command(command: Any) -> bool:
    """A journal start's command as the launcher logs it: the script, the config's name and its
    overrides, all strings."""
    return (isinstance(command, list) and len(command) >= 2
            and all(isinstance(c, str) for c in command))  # fmt: skip


def _unissued(command: list[str]) -> Exception | None:
    """Why a start's command is composed by no study-3 cell of the launcher, as the error its
    composition is refused with (its message the reason, ``"<what>: not loaded"``): a config that
    is none of ``STUDY3_CONFIGS`` (``FileNotFoundError``: no such study-3 config), or a section
    override (``key=value``, the key a top-level name) whose value is no plain file name
    (``ValueError``); None for a command the launcher issues for study 3."""
    name, overrides = command[1], command[2:]
    if name not in STUDY3_CONFIGS:
        return FileNotFoundError(f"names the config {name!r}, not a study-3 config "
                                 f"({', '.join(STUDY3_CONFIGS)}): not loaded")  # fmt: skip
    for item in overrides:
        key, sep, value = item.partition("=")
        if sep and "." not in key and not key.startswith("+") and not plain_name(value):
            return ValueError(f"has the override {item!r}, which names no config file by its "
                              "plain name: not loaded")  # fmt: skip
    return None


def _command_config(command: Any) -> Any:
    """The config a journal start's command (the script, the config's name, its overrides)
    composes under this checkout's configs, as the launcher composes a campaign's cell; raises
    ``ValueError`` for what is no command, the refusal of ``_unissued`` for a command no study-3
    cell gives (nothing of it is loaded), and whatever the composition raises."""
    if not _is_command(command):
        raise ValueError("no command")
    unissued = _unissued(command)
    if unissued is not None:
        raise unissued
    return load_run_config(command[1], command[2:])


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
    recoveries: Sequence[RecoveryRound],
    voided: Mapping[str, tuple[str, str]],
) -> list[str]:
    """Mark the attempts that no longer count (module docstring, "Attempts"); a refused one
    stays refused. A round recovers the attempts of the labels its ``runs`` names that it
    follows (``round_follows``) and that have no published terminal record; the first round to
    follow an attempt names it: the rounds written before rounds were numbered in file order,
    as always, then the numbered ones in time order (``round_order``). Returns the recovery
    rounds that cannot be placed (no time stamp)."""
    by_label: dict[str, list[_Attempt]] = defaultdict(list)  # across every log
    for attempt in attempts:
        if attempt.cell.label is not None and attempt.number is not None:
            by_label[attempt.cell.label].append(attempt)
        if attempt.state is None and attempt.run_id in voided:
            attempt.state, reason = voided[attempt.run_id]
            attempt.reasons.append(reason)
    problems = []
    ordered = [x for x in recoveries if not x.numbered]
    ordered += sorted((x for x in recoveries if x.numbered), key=round_order)
    for item in ordered:
        record = item.record
        if item.time is None:
            problems.append(f"a recovery round of {record.get('runs')!r} without a time stamp "
                            f"({record.get('time')!r}): its attempts cannot be placed")  # fmt: skip
            continue
        named = ("a recovery round" if item.number is None or not item.numbered
                 else f"recovery round {item.number}")  # fmt: skip
        for label in item.labels:
            mine = by_label.get(label, [])
            starts = [(a.number, a.started) for a in mine]
            for attempt in mine:
                if attempt.state is not None or (attempt.run_dir / "terminal.json").is_file():
                    continue
                if round_follows(record, attempt.run_id, attempt.number, attempt.started, starts):
                    attempt.state = "recovered"
                    attempt.reasons.append(f"{named} followed it: {record.get('reason')}")
    return problems


def _round_problems(
    recoveries: Sequence[RecoveryRound],
    starts: Sequence[tuple[str, Mapping[str, Any]]],
    ends: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    voided: Mapping[str, tuple[str, str]] | None = None,
) -> list[str]:
    """The numbered recovery rounds that do not read as the launcher writes them (module
    docstring, "Attempts"), each named by its log and line, never by its reason: what
    ``_round_fields`` finds in its record; round 1 whose ``granted`` is not the rule's
    (``round_grant``: the smaller of its ``max_attempts`` and the label's base); an ``after``
    that names a run id that is no start of its label, leaves out a start of the label stamped
    before the round, or names an attempt, not superseded, whose end records done, diverged or
    conflict; a label's rounds not numbered 1, 2, … in time order (ties by number); a numbered
    round that gives the reason of an earlier round of its label. Rounds written before rounds
    were numbered read as before: no problem of their own."""
    ends, voided = ends or {}, voided or {}
    every = [start for _, start in starts]
    problems: list[str] = []
    for item in recoveries:
        if item.numbered:
            problems += [f"{item.where}: a recovery round {why}" for why in _round_fields(item)]
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for _, start in starts:
        if isinstance(start.get("run"), str) and isinstance(start.get("run_id"), str):
            by_label[start["run"]].append(start)
    for label, mine in by_label.items():
        rounds = label_rounds(recoveries, label)
        reasons: dict[str, RecoveryRound] = {}
        for position, item in enumerate(rounds, start=1):
            if item.numbered and item.reason is not None and item.reason in reasons:
                problems.append(f"{item.where}: a recovery round that gives the reason of the "
                                f"label's round at {reasons[item.reason].where}: each round "
                                "documents an interruption of its own")  # fmt: skip
            if item.reason:
                reasons.setdefault(item.reason, item)
            if not item.numbered or item.number is None:
                continue
            if item.number != position:
                problems.append(f"{item.where}: a recovery round numbered {item.number}, the "
                                f"label's round {position} in time order: a label's rounds are "
                                "numbered 1, 2, … in time order")  # fmt: skip
            if item.only != label:
                continue
            chash = item.record.get("config_hash")
            granted = item.record.get("granted")
            if item.number == 1:
                rule = round_grant(
                    item,
                    rounds,
                    base_attempts(every, label, chash, item.record.get("max_attempts")),
                )
                if type(granted) is not int or granted != rule:
                    problems.append(f"{item.where}: a recovery round 1 whose granted is not the "
                                    f"rule's ({rule}: the smaller of its max_attempts and the "
                                    "label's base)")  # fmt: skip
            problems += [f"{item.where}: a recovery round {why}"
                         for why in _round_attempts(item, mine, ends, voided)]  # fmt: skip
    return problems


def _round_fields(item: RecoveryRound) -> list[str]:
    """What a numbered round lacks of what the launcher writes, from its record alone: its
    number, its ``after`` list, ``config_hash`` and ``max_attempts``, a reason that is not
    blank, its incident (a UTC time not later than the round's own stamp, a cause and the
    evidence, each on one line), ``only`` and one attempt (``granted`` and ``max_attempts`` 1)
    from round 2 on."""
    record, found = item.record, []
    if item.number is None:
        found.append("without its number (round)")
    if item.after is None:
        found.append("whose after list is no list of run ids")
    if not isinstance(record.get("config_hash"), str):
        found.append("without its config_hash")
    if type(record.get("max_attempts")) is not int or record["max_attempts"] < 1:
        found.append("without its max_attempts")
    if not item.reason:
        found.append("without its reason: a blank one documents nothing")
    for key in INCIDENT_KEYS:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            found.append(f"without its incident: {key} is missing or blank")
        elif key != "incident_time" and len(value.splitlines()) > 1:
            found.append(f"whose {key} is more than one line")
    when = record.get("incident_time")
    if isinstance(when, str) and when.strip() and not _utc(when):
        found.append("whose incident_time is no ISO 8601 time in UTC")
    elif (
        isinstance(when, str)
        and when.strip()
        and item.time is not None
        # the launcher stamps a round to the second: compared at whole seconds
        and (datetime.fromisoformat(when.strip()).replace(microsecond=0) > item.time)
    ):
        found.append("whose incident_time is later than the round's own time stamp")
    if item.number is not None and item.number >= 2:
        if item.only is None:
            found.append(f"numbered {item.number} without only: a further round is a run's own")
        if record.get("granted") != 1 or record.get("max_attempts") != 1:
            found.append(f"numbered {item.number} that does not allow exactly one attempt "
                         "(granted and max_attempts 1)")  # fmt: skip
    return found


def _round_attempts(
    item: RecoveryRound,
    starts: Sequence[Mapping[str, Any]],
    ends: Mapping[str, Sequence[Mapping[str, Any]]],
    voided: Mapping[str, tuple[str, str]],
) -> list[str]:
    """How a round of a label's own disagrees with the label's starts (``starts``, any config
    hash): ``after`` names a run id that is no start of the label, leaves out one stamped before
    the round, or names an attempt under the round's config hash, not superseded, whose end
    records done, diverged or conflict (``ends``, ``voided``; ``launch._outcome_held``). A
    later start under another config hash, at any attempts, is the launcher's own sequence."""
    after, when, found = item.after, item.time, []
    if after is None:
        return found
    ids = {start["run_id"] for start in starts}
    strangers = [run_id for run_id in after if run_id not in ids]
    if strangers:
        found.append(f"whose after list names {len(strangers)} run id(s) that are no start of "
                     "its label")  # fmt: skip
    earlier = [s for s in starts if s["run_id"] not in after and when is not None
               and (at := _time(s.get("time"))) is not None and at < when]  # fmt: skip
    if earlier:
        found.append(f"whose after list leaves out {len(earlier)} start(s) of its label stamped "
                     "before it")  # fmt: skip
    chash = item.record.get("config_hash")  # the launcher judges outcomes under the run's hash
    mine = {s["run_id"] for s in starts if s.get("config_hash") == chash}
    held = sorted({str(end.get("outcome")) for run_id in after
                   if run_id in mine and voided.get(run_id, ("",))[0] != "superseded"
                   for end in ends.get(run_id, []) if end.get("outcome") in HELD_ENDS})  # fmt: skip
    if held:
        found.append(f"over an attempt whose end records the outcome {', '.join(held)}, not "
                     "superseded: the launcher opens no round over an outcome")  # fmt: skip
    return found


def _utc(value: str) -> bool:
    """An ISO 8601 time in UTC (``+00:00`` or ``Z``), as the launcher takes an incident's."""
    try:
        moment = datetime.fromisoformat(value.strip())
    except ValueError:
        return False
    offset = moment.utcoffset()
    return offset is not None and not offset.total_seconds()


def _attempt_key(start: Mapping[str, Any]) -> tuple[int, str]:
    """``launch.attempt_key``, mirrored: a start's attempt number, then its time stamp."""
    attempt = start.get("attempt")
    return (attempt if type(attempt) is int else -1, str(start.get("time") or ""))


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
              theta0: Mapping[str, float], files: _Files | None = None) -> Endpoint:  # fmt: skip
    cell = attempt.cell
    common = {"run_id": attempt.run_id, "run_dir": attempt.run_dir, "cell": cell,
              "arm": cell.arm, "lr": cell.rate, "seed": cell.seed, "role": attempt.role,
              "candidate": cell.candidate, "k_only": attempt.k_only,
              "attempt": attempt.number, "label": cell.label, "git": attempt.git}  # fmt: skip
    if attempt.state is not None:
        return Endpoint(**common, state=attempt.state, reasons=tuple(attempt.reasons))
    return Endpoint(**common, **_outcome(attempt, ends, theta0, files or _Files()))


def _outcome(attempt: _Attempt, ends: Sequence[Mapping[str, Any]],
             theta0: Mapping[str, float], files: _Files) -> dict[str, Any]:  # fmt: skip
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
    status = None if published else (_json(run_dir / "meta.json", files) or {}).get("status")
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
    problems, terminal, row = _records(attempt, files)
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


def _records(
    attempt: _Attempt, files: _Files | None = None
) -> tuple[list[str], dict[str, Any] | None, dict[str, Any] | None]:
    """A run's records with a published terminal record, checked from their numbers (module
    docstring, "A conforming record"): the problems, the terminal record and the step-25
    evaluation (None without one)."""
    run_dir = attempt.run_dir
    files = files or _Files()
    try:
        meta = _load(run_dir / "meta.json", files)
        terminal = _load(run_dir / "terminal.json", files)
        summary = (_load(run_dir / "summary.json", files)
                   if (run_dir / "summary.json").is_file() else {})  # fmt: skip
        evaluations = _jsonl(run_dir / "evaluations.jsonl", files)
        metrics = _jsonl(run_dir / "metrics.jsonl", files)
        readings = _jsonl(run_dir / "monitor.jsonl", files)
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


def _load(path: Path, files: _Files | None = None) -> dict[str, Any]:
    data = json.loads(path.read_text() if files is None else files.read_text(path))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return data


def _json(path: Path, files: _Files | None = None) -> dict[str, Any] | None:
    try:
        return _load(path, files)
    except (OSError, ValueError):
        return None


def _jsonl(path: Path, files: _Files | None = None) -> list[dict[str, Any]]:
    """The rows of a JSON-lines record; none when it is absent."""
    if not path.is_file():
        return []
    text = path.read_text() if files is None else files.read_text(path)
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name} holds a row that is not a JSON object")
    return rows
