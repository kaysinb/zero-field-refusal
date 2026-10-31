"""Study 3's decision records, each bound to the evidence it rests on.

docs/study3-preregistration.md: "Simulation before validation" (the coverage screen, the level L
and the validation size), "Validation, confirmation and stopping" ("Sample size from validation":
N, the cap, the budget decision) and "Seeds and order" (harm labels never enter a selection; the
order of the stages). The records are small JSON files under
``RUNS_DIR/study3/decisions/<kind>.json``; this module fixes their schemas, reads them and
checks them, and writes them once. Later packages (the simulation, the spread script) write the
records; the state machine consumes only what this module accepts.

- ``read_decision``: one record if it is present and valid, and its problems: a schema or value
  off its set, an evidence file that is missing or whose sha256 is no longer the recorded one,
  an unrounded spread, a record that contradicts the one it answers. A record with any problem
  is not returned.
- ``decisions``: the records a candidate's route needs (engineering, simulation, ``n-c<N>`` and,
  when N asks for one, ``budget-c<N>``), each with its state and problems, the immutability check
  of ``immutable`` included; a record is invalid when one it rests on is. An n or budget record is
  the link in force of its chain of successors (below, "Successors").
- ``chain``, ``successor_kind``, ``next_successor`` and ``link_of``: an n or budget record's
  chain of numbered successors; ``confirmation_started``: whether the candidate's confirmation
  has started (a start of one of its blocks in the launch journal).
- ``immutable``: a record whose sha256 differs from the one a started campaign's header named.
- ``write_decision``: validates and writes a record atomically, once; a second version is refused.
- ``cap_of``, ``budget_reason`` and ``budget_needed``: the cap N + max(10, ⌈0.12·N⌉), and why (if
  at all) N asks for a budget decision.
- ``closures``: the closure records of a candidate's arms, each with its state and problems, as
  ``decisions`` lists the others; ``hold_reasons``: the ids of the reasons a hold line states;
  ``end_lines`` and ``start_lines``: each run's last ``end`` line and its start line in the
  launch journal, as a closure cites them; ``end_reason``: the id an end's outcome would give a
  closure's reason (``launcher_<outcome>``), which a closure may not carry (``NOT_ACCEPTED``);
  ``invalidate``: an entry made invalid, with why.
- ``cycle_path`` and ``bridge_name``: a candidate's cycle record and its bridge (the supersession
  record ``scripts/qualify.py`` reads); ``serialized``: a record's bytes as written;
  ``one_line``: the rule of a cycle's free text; ``pinned_end``: where a launch log's pinned
  prefix ends.

**Kinds**: ``engineering``, ``simulation``, ``n-c<N>`` (one per candidate), ``budget-c<N>``,
their numbered successors ``n-c<N>-<k>`` and ``budget-c<N>-<k>`` (k = 2, 3, …; below,
"Successors"), ``cycle-c<N>`` (one per candidate: "Failure routes") and ``closure-c<N>-<arm>``
(one per candidate and arm A, F or P).
Every record carries ``kind`` (equal to its file's), ``state`` (``passed`` or ``failed``),
``written_at`` (ISO 8601 with a UTC offset) and ``code``: ``commit`` (40 hexadecimal digits) and
``gpu_code_sha256`` (the sha256 of ``code_identity.gpu_code_identity`` at that commit). The
schemas are closed: a field the schema does not name is refused, and so is a simulation gate or
input whose name is not one of the fixed gate names or a plain lowercase word, so that no mean,
sign, per-seed value, unrounded spread or exploratory engineering diagnostic (e_eng, path
maxima) can ride in a record or gate anything.

**Evidence** is a reference ``{"path", "sha256"}`` (and, for a path that is not absolute,
``"root"``: ``"runs"``, the default, for a path relative to RUNS_DIR, or ``"repo"`` for one
relative to the checkout). It is checked against the file as it is now, byte for byte, whenever
the record is read; a missing file or another sha256 is a problem. A reference to a file under a
``responses/`` or ``observations/`` folder is refused and never opened: nothing a decision rests
on is a response or an observation, and this module opens none.

**The schemas**
- ``engineering``: ``checks`` with the four prescribed ones, ``dry_run``, ``bridge``,
  ``theta0_utility`` (``scripts/check_theta0_utility.py``'s output) and ``stage0``, each the
  evidence of the record it rests on and ``passed``; a ``passed`` record needs all four, with a
  record each and each passed. In a ``failed`` record a check that left no record is null.
- ``simulation``: ``fixation`` (the simulation's fixation) and ``inputs`` (name → evidence; a
  name is a lowercase word of letters and underscores, at most 40, no digits, and holds none of
  the words that mark a mean, a sign, a per-seed value, a spread or an e_eng), the commit and GPU
  code in ``code``; ``gates`` (name → ``value``, ``requirement``, ``passed``), closed to
  ``safe_pass_rate`` (the pass rate at the safe alternative of the 28-block validation, required
  ≥ 0.95) and ``power_af`` (the lower of the A–F powers at the two planning spreads, required
  ≥ 0.80), both present, the requirement no lower than the registered one and ``passed`` equal
  to value ≥ requirement, and four gates that are only reported, with a null requirement and a
  null ``passed``: ``empirical_pass_rate`` (study 2's empirical point), ``power_pf``,
  ``detection_power_comparison`` and ``detection_power_joint`` (the detection power at the
  actual J); ``power_af`` also has ``power_short`` (true when its requirement is not met) and
  ``n_min``, the smallest N at which the simulated power reaches the requirement (an integer, at
  least 60, when the power is short; else null); ``coverage``: one entry per level tested, in
  ascending order, with ``level``, ``tests``, ``J``, ``M`` (exactly 10 000), ``counts`` and
  ``thresholds`` (each ``comparison`` and ``joint``, the non-coverage counts X and the thresholds
  c; each X is the largest count over the level's ``tests``, the N grid and the scenarios, and,
  for ``comparison``, over both comparisons A–F and P–F: the worst test decides, so the level
  passes exactly when every X is at or under its c, which is how its ``passed`` is checked; the
  record carries no count per test, so that aggregate itself cannot be checked here) and
  ``passed``, all with the same J, the one frozen number of coverage tests across the
  grid, the levels and the calibration and verification streams: the ``tests`` of every level
  and of the verification add up to at most J; ``level`` (L, one of 0.975, 0.980, 0.985, 0.990,
  or null when no level passed) and ``validation_blocks`` (28 or 40); ``verification``
  (``level``, ``passed``, its ``tests``, the ``calibration_stream`` and ``verification_stream``,
  two different integers). L is the first level whose tests passed (the levels below it tested
  and failed); a null L means every level was tested and failed.
  **The state follows the coverage screen and its independent verification, and nothing else**:
  ``passed`` exactly when a level passed and its verification passed ("If no level passes, or the
  verification fails, the confirmation cannot start"). A pass rate below its requirement is not a
  failure but the 40-block validation, chosen once: ``validation_blocks`` is 40 when
  ``safe_pass_rate`` is below 0.95 and 28 otherwise. A power shortfall is not a failure either: it
  is recorded (``power_short``, ``n_min``) and N and the cap are raised, with a decision on the
  money (see ``n-c<N>`` and ``budget-c<N>``). A failed record may have a null L or a null size.
- ``n-c<N>``: ``candidate`` (N of the kind), ``rates`` (per arm A, F and, when P–F is active, P:
  ``lr_l`` and ``lr_u``), ``active`` (``["A-F"]`` or ``["A-F", "P-F"]``), ``level`` (the
  simulation record's), ``cohort`` (the validation block ``seeds`` and the ``run_ids`` used, at
  least four per block: A and F at both rates), ``spread`` (the evidence of the precommitted
  spread script, ``script``, and of its ``output`` record), ``spread_pp`` (the A–F spread
  rounded to 0.5 pp, and nothing else about it), ``simulation`` (the evidence of the simulation
  record itself, at ``study3/decisions/simulation.json`` under the runs directory, so the n
  record rests on the exact simulation it follows), ``N`` (an integer, at least 60, and at least
  the simulation record's ``n_min`` when its power is short) and ``cap`` (N + max(10,
  ⌈0.12·N⌉)). N is never cut: a cap beyond the 200 reserved confirmation seeds (10400–10599) is
  recorded, with state ``failed`` (the confirmation cannot start within the registered seeds); the
  state is ``passed`` otherwise.
- ``budget-c<N>``: ``candidate``, ``N`` and ``cap`` it answers (those of the ``n-c<N>`` record),
  ``decision`` (``proceed``, with state ``passed``, or ``stop``, with state ``failed``) and
  ``answers``, the evidence of the n record itself, a link of the candidate's n chain
  (``study3/decisions/n-c<N>.json``, or a successor ``n-c<N>-<k>.json``) under the runs directory
  and no other file (so its sha256 is the recorded one); a budget successor answers an n
  successor.
  It is needed when a passed ``n`` record has N above 90 and when N is raised for power (the
  simulation record's ``power_short``), even if N is 90 or less: the extra blocks are the
  researcher's decision; ``decisions`` lists the record as ``missing``, with that as its problem,
  until it is written. A failed ``n`` record needs none. One for an N that needs none is valid
  and read as written.
- ``closure-c<N>-<arm>``: an unresolved technical failure of the arm closed as the terminal
  technical outcome it is ("Failure routes", "After a technical mark"): a launcher run that the
  candidate's engineering cycle held (an outcome nobody could resolve, never attempted again), a
  run under another code that the readers count as a technical mark and that no recovery round
  reran, or a technical mark outside the qualification still unresolved after the candidate's
  engineering cycle. ``candidate`` and ``arm`` (the kind's), ``rate``,
  ``seed`` (a registered seed of the candidate, the one the cited line's label names),
  ``run_id``, ``line`` (the launch-journal line that shows the failure: ``kind``, ``path``,
  ``launch/<log>.jsonl`` under the runs directory, ``line``, its number from 1, and
  ``line_sha256``, the sha256 of the line's UTF-8 text: the line's own bytes, which stay when the
  launcher appends to the log, as the log's sha256 does not), ``cycle``, ``reason`` and
  ``decision``: ``technical_feasibility_report`` for A or F (the study ends with the technical
  feasibility report, and candidate 2 is not taken), ``close_pf`` for P (P–F closes, and A–F goes
  on); any other pairing is refused. The state is ``failed``: a terminal technical outcome.
  - ``kind: "held"``: a ``held`` line of the launch journal naming the run, whose reasons state
    ``reason`` (an id of ``HOLD_REASONS``: never the evidence's free text); ``cycle`` is the
    evidence of the candidate's supersession record (``qualification/c<N>/supersede-<n>.json``
    under the runs directory), whose ``held`` names the run.
  - ``kind: "end"``: the run's last ``end`` line in the journal (by its time stamp, in any log),
    whatever its outcome.
  - ``kind: "start"``: the run's start line, when the journal holds no end line of the run.
  - An end or start line closes one of two runs, by its ``reason``: a run under a code other
    than the one it counts under, with its uncounted reason id (``UNCOUNTED_REASONS``: a
    validation or confirmation run under a commit other than the pinned one,
    ``study3_versions.uncounted_runs``; a counting run of a stage a cycle reruns under a GPU code
    other than the fix's, ``other_gpu_code``), whatever the end's outcome, and ``cycle`` null;
    or a technical mark outside the qualification left unresolved after the candidate's one
    engineering cycle ("Failure routes"), with ``UNRESOLVED_AFTER_CYCLE`` and ``cycle`` the
    evidence of the candidate's supersession record (``qualification/c<N>/supersede-<n>.json``
    under the runs directory, the candidate's). That the run is one, with that reason, is the
    state machine's to check. A closure with an end's or a start line's own reason
    (``launcher_<outcome>``: ``end_reason``; ``no_end_line``: ``START_REASON``) carries the
    launcher's outcome, never its route's reason: it is refused (``NOT_ACCEPTED``).
  An end or start closure is refused for a run that a later attempt of its label, a recovery
  round of its label at or after the cited line, a ``superseded`` line or a ``held`` line
  followed: the readers would not count it as a technical mark. That the run lies outside the
  qualification and that the readers count it as a technical mark is the state machine's to
  check (``study3_progress``). The record is checked against the files as
  they are now, through ``journal_lines`` (the one reading of a cited journal line, each log
  split at ``\n`` alone as the launcher reads it, so a line's number counts every line before it,
  a resolved torn line included; a launch log that cannot be read, or an unresolved torn line
  before the last of a log, fails the check closed): the line's bytes, the supersession record's
  bytes, and the run id both name. The failure's evidence stays as it is; a closure removes
  nothing.
- ``cycle-c<N>``: the candidate's engineering cycle for a technical mark outside the
  qualification (its fields, what they list and the bridge it names: ``study3_cycle``'s
  docstring). Checked here: the form, the state ``passed``, ``stage`` one of ``MARKED_STAGES``,
  ``old_code`` a sha256 that is not the fix's (``code.gpu_code_sha256``: a fix that keeps the
  GPU code is no cycle), ``diagnosis`` and ``fix`` of one line each (``one_line``: not blank, at
  most ``TEXT_MAX`` characters), the marks (run, its label, a step of the stage, arm, rate,
  registered seed, its cited end or start line), the claims (``stages``: every stage up to the
  marks' claimed, each ``reuse`` with no arm or ``rerun`` with its arms in the order A, F, P,
  the qualification's all three, each with a justification of one line; the marks' stage a
  rerun of every marked arm), ``fixation`` null or evidence in the checkout, ``journal`` (each
  log's pinned prefix), the affected runs (their files as evidence or absence markers, their
  start and end lines), the qualification's records and launcher runs, ``kept`` (``KEPT_WHY``)
  and ``bridge`` (``bridge_name``); every list in canonical order (by run id, then path) and
  naming nothing twice. Then, as the files are now: every evidence file of its sha256, every
  absence marker's file still absent, every pinned prefix of its sha256 (a rewritten journal is
  a problem) and every cited line of its bytes, the event and run it names. What the record
  lists against the records it rests on is ``study3_cycle``'s to check.

**Successors** ("Voiding and supersession": a superseded record stays, listed with its reason;
"Validation, confirmation and stopping", "Sample size from validation"). A candidate's
engineering cycle moves its GPU code, so a write-once record of the old code goes stale: its
successor is a new file, ``n-c<N>-<k>`` or ``budget-c<N>-<k>`` (k = 2, 3, …), and the record it
succeeds stays as it was. A successor carries ``supersedes``: ``{"path", "sha256", "reason",
"cycle"}``, the record before it (``n-c<N>.json`` for k = 2, else ``n-c<N>-<k-1>.json``, under
the runs directory) by its path and its sha256 now, the reason ``cycle_c<N>`` and the evidence
of the candidate's cycle record (``study3/decisions/cycle-c<N>.json``); the first record carries
none. A successor is accepted only after a stale record: the cycle is one ``study3_cycle.
read_cycle`` accepts, its old code is the predecessor's ``code.gpu_code_sha256``, it lists the
predecessor among the records it makes stale (``affected.records``, by path and sha256: the cycle
was written after it), and the successor carries the cycle's fix as its code. The record in force
is the last link of an unbroken chain from the first: a fork (a link that names another than the
one before it), a gap (a missing number), a link changed since its successor named it, or a
successor of a record that is not stale is a problem of the chain. Only the link in force is
checked in full; the earlier links for their integrity (there, of the sha256 their successor
names, unchanged under the campaigns that named them: ``immutable``). A budget successor
answers the n record's successor. Once a candidate has a cycle record, ``write_decision``
writes its n and budget records only under the cycle's fix, the code in force for its runs, and
no record under the old code of any candidate's cycle (``after_cycle_code``): the records a cycle
lists as stale then change after it only when a file is tampered with.

*The confirmation boundary* ("Validation, confirmation and stopping", "Confirmation": the
confirmation never triggers a new candidate). Once the candidate's confirmation has started (a
study-3 start of the candidate in the launch journal whose header names the step
``confirmation``: the guard lets a block start only under its supplement), the candidate, the
rates, the started seeds and the cohort's order never change: an n or budget successor differs
from its predecessor only in ``kind``, ``code``, ``written_at``, ``supersedes`` and, for a budget
record, ``answers`` (the n record in force). Any other change, of N, the cap or the cohort among
them, needs an explicit prospective amendment, which the code never infers: it is refused.

**Rulings** where the registration leaves a detail open:
- *Cross-record checks*: an ``n-c<N>`` record is valid only with a valid ``simulation`` record
  that passed, of the same L, whose validation size its cohort fits (seeds among the first 28 or
  40 of the candidate's validation seeds, at least 22 or 31 of them: the gate's minimum) and whose
  ``n_min``, when its power is short, N reaches; a
  ``budget-c<N>`` record only with a valid ``n-c<N>`` of its N and cap.
- *The thresholds of the coverage screen* are recomputed, c = BinomQuantile(1 − 0.05/J; M, p₀)
  with p₀ = 0.025 per comparison and 0.05 jointly, and a record whose c is another is refused; J
  is one number for all levels, and the tests of the levels and the verification fit within it,
  so a level cannot pass on a J raised for it; a level's ``passed`` must agree with its counts and
  thresholds. Whether that J is the one the simulation's fixation froze would need the fixation's
  content, which is not read here.
- *The reserved seeds*: the cap of an ``n-c<N>`` record is within the 200 confirmation seeds
  exactly when its state is ``passed``.
- *Immutability* (``immutable``): a campaign's header names a record by a reference
  ``{"path", "sha256"}`` anywhere in it, or by a string path with a sibling ``<key>_sha256``
  (the form ``versions_record`` takes; the launcher's headers are flat); paths compare by their
  last three components (``study3/decisions/<kind>.json``), so a record named from another
  machine's root is found, and a path among the items of a list is named without a sha256. A
  record named with a sha256 other than its own now, or named without a readable one, or no
  longer there, is a problem; a missing journal, or one with a torn line before the last line of
  its log that is not resolved (``study3_records``' rule), fails closed. A successor's name
  (``n-c<N>-<k>.json``) is matched the same way. A
  record that resolves (a link, say) under a ``responses/`` or ``observations/`` folder is a
  problem and is never opened. The journal is read as ``study3_records`` reads it: through its
  private ``_read_journal``, which this module keeps importing (one journal reader, no copy).
- *Nothing is computed from the evidence*: its content is the writer's; this module binds the
  record to the file's bytes and does not read the evidence's meaning. A cycle record binds the
  journal lines it cites and its pinned prefix by their bytes (each cited line's event and run
  id included). The closure is the one exception: the launch journal and the supersession
  record are read for the run ids, labels, attempts, outcomes and time stamps they name and the
  reasons a held line states, as ids, and nothing else (no free text is copied).
- *The reasons of a hold* are those ``scripts/qualify.py`` gives an unresolved launcher run
  (``resolve_launcher_run``), joined with "; " in the hold line: each has an id here, by the text
  it starts with (``HOLD_REASONS``); a test reads the script's source so that the two stay one
  list.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

from ..paths import REPO_ROOT
from .study3_records import DECISIONS as DECISIONS_FOLDER
from .study3_records import (
    EXPERIMENT,
    LOOP_WHY,
    SEEDS,
    TORN_LINE,
    _read_journal,
    _time,
    journal_lines,
    log_lines,
    round_follows,
    sealed_why,
    unresolvable,
)
from .study3_records import under_sealed as _under_sealed  # the one sealed-path rule

DIRECTORY = (EXPERIMENT, DECISIONS_FOLDER)  # reserved in the experiment's folder
STATES = ("passed", "failed")

# "Simulation before validation": the candidate levels, the validation sizes and the gates
LEVELS = (0.975, 0.980, 0.985, 0.990)
VALIDATION_BLOCKS = (28, 40)
SUPPORTED_AT = {28: 22, 40: 31}  # the validation gate's minimum of supported blocks
REQUIRED_GATES = {"safe_pass_rate": 0.95, "power_af": 0.80}
# Gates that are only reported: the pass rate at study 2's empirical point, P-F's power, and the
# detection power of the coverage screen at the actual J (per comparison and joint)
REPORTED_GATES = ("empirical_pass_rate", "power_pf", "detection_power_comparison",
                  "detection_power_joint")  # fmt: skip
COVERAGE_RUNS = 10_000  # M, simulated studies per coverage test
FAMILY_ALPHA = 0.05
COVERAGE_P0 = {"comparison": 0.025, "joint": 0.05}
# "Sample size from validation"
MIN_N = 60
BUDGET_ABOVE = 90  # N above this needs the researcher's budget decision
CAP_FLOOR, CAP_SHARE_PERCENT = 10, 12
ACTIVE = (["A-F"], ["A-F", "P-F"])
SPREAD_STEP_PP = 0.5
CHECKS = ("dry_run", "bridge", "theta0_utility", "stage0")
DECISIONS = ("proceed", "stop")
# The closure of a held run: the terminal decision that applies to each arm ("Failure routes")
CLOSURE_DECISIONS = {"A": "technical_feasibility_report", "F": "technical_feasibility_report",
                     "P": "close_pf"}  # fmt: skip
# The reasons ``qualify.py`` holds an unresolved launcher run with (``resolve_launcher_run``), each
# by its id and the text the hold line states it with (the start of one "; "-joined reason)
HOLD_REASONS = {
    "records_unreadable": "its records are unreadable",
    "no_campaign_header": "no campaign header in its meta.json",
    "no_numerics_setup": "no numerics setup",
    "no_theta0_hash": "no hash of θ₀'s trainable parameters",
    "no_v0_hash": "no hash of v₀",
    "no_recorded_commit": "no recorded commit",
    "gpu_code_unknown": "the GPU code of its commit",
    "issued_config_not_composed": "what it was issued cannot be composed",
    "issued_config_hash_differs": "its issued config, composed from the configs of its commit",
    "numerics_setup_key_missing": "no numerics_setup.",
    "no_terminal_record": "no whole terminal record",
}
# The line a closure cites: a held line, the run's last end line, or its start line when the
# launcher never ended it (an end or start line of a run under another commit alone)
LINE_KINDS = ("held", "end", "start")
# A start line's own reason (the launcher never ended the run): a closure may not carry it
START_REASON = "no_end_line"
# A technical mark outside the qualification left unresolved after its candidate's one
# engineering cycle ("Failure routes"): the reason of its closure, which cites that cycle
UNRESOLVED_AFTER_CYCLE = "unresolved_after_cycle"
# A closure whose reason is an end's outcome or the start line's: refused
NOT_ACCEPTED = ("a closure carries its route's reason (an uncounted reason, or "
                "unresolved_after_cycle), never the launcher's outcome")  # fmt: skip
# Why a run does not count for the code it ran: a validation or confirmation run under a commit
# other than the pinned one (``study3_versions.uncounted_runs``), or a counting run of a stage a
# cycle reruns under a GPU code other than the fix's (``other_gpu_code``); the reason of its
# closure, an end or a start line
UNCOUNTED_REASONS = ("no_commit", "dirty_checkout", "dirty_unrecorded", "unknown_commit",
                     "no_versions_record", "other_commit", "other_gpu_code")  # fmt: skip
# The engineering cycle ("Failure routes"): the stages a cycle claims and their steps (the
# qualification's are its numerical runs and its repair), the stages a mark opens one from, the
# claims, the arms, why a run keeps counting, and the bound on its free text
CYCLE_STAGES = ("search", "qualification", "validation", "confirmation")
CYCLE_STEPS = {"search": ("search",), "qualification": ("numerical", "repair"),
               "validation": ("validation",), "confirmation": ("confirmation",)}  # fmt: skip
MARKED_STAGES = ("search", "validation", "confirmation")
CLAIMS = ("reuse", "rerun")
CLAIM_ARMS = ("A", "F", "P")
KEPT_WHY = ("scientific_failure", "held", "closed_arm", "reused_stage")
TEXT_MAX = 1000  # characters of a diagnosis, a fix or a claim's justification
_OUTCOME = re.compile(r"[a-z][a-z_]{0,40}")
COMMON = ("kind", "state", "written_at", "code")
FIELDS = {
    "engineering": ("checks",),
    "simulation": ("fixation", "inputs", "gates", "coverage", "level", "validation_blocks",
                   "verification"),
    "n": ("candidate", "rates", "active", "level", "cohort", "spread", "spread_pp", "N", "cap",
          "simulation"),
    "budget": ("candidate", "N", "cap", "decision", "answers"),
    "closure": ("candidate", "arm", "rate", "seed", "run_id", "line", "cycle", "reason",
                "decision"),
    "cycle": ("candidate", "stage", "old_code", "marks", "diagnosis", "fix", "stages",
              "fixation", "journal", "affected", "kept", "bridge"),
}  # fmt: skip

_KIND = re.compile(
    r"(engineering|simulation|(?:n|budget)-c([12])(?:-([2-9]|[1-9][0-9]+))?"
    r"|cycle-c([12])|closure-c([12])-([AFP]))"
)
# A successor may differ from its predecessor in these fields alone once the confirmation has
# started (module docstring, "The confirmation boundary")
SUCCESSOR_FREE = ("kind", "code", "written_at", "supersedes", "answers")
OPTIONAL = {"n": ("supersedes",), "budget": ("supersedes",)}  # a successor's field
_REASON_ID = re.compile(r"[a-z][a-z0-9_]{0,63}")
_N_LINK = re.compile(r"study3/decisions/n-c([12])(?:-(?:[2-9]|[1-9][0-9]+))?\.json")
_CYCLE = re.compile(r"qualification/c([12])/supersede-\d+\.json")
_INPUT_NAME = re.compile(r"[a-z](?:[a-z_]{0,38}[a-z])?")  # letters and underscores, no digits
_COMMIT, _SHA256 = re.compile(r"[0-9a-f]{40}"), re.compile(r"[0-9a-f]{64}")
# Words that mark a field as one no record carries (the hint on an unknown field)
_HIDDEN = re.compile(
    r"mean|(?<!de)sign|per[_-]?seed|(^|_)sd($|_)|std|e_eng|path_max|unrounded|spread"
)


class DecisionRefused(ValueError):
    """A record that is not valid, or one that is already written."""


class _Ref(NamedTuple):
    where: str
    path: str
    sha256: str
    root: str


Starts = Sequence[tuple[str, Mapping[str, Any]]]  # the journal's starts: (campaign, record)
SIMULATION_RECORD = "/".join((*DIRECTORY, "simulation.json"))  # an n record's reference to it


# ------------------------------------------------------------------ the kinds and their files


def kind_of(kind: str) -> tuple[str, int | None]:
    """(``engineering``, ``simulation``, ``n``, ``budget``, ``cycle`` or ``closure``, the
    candidate) of a kind (a successor's too: ``n-c1-2`` is ``n`` of candidate 1);
    ``ValueError`` for another name."""
    found = _KIND.fullmatch(kind) if isinstance(kind, str) else None
    if found is None:
        raise ValueError(f"{kind!r} is not a decision kind (engineering, simulation, n-c<N>, "
                         "budget-c<N> and their successors n-c<N>-<k> and budget-c<N>-<k>, "
                         "cycle-c<N>, closure-c<N>-<arm>, N = 1 or 2, k = 2, 3, ..., arm A, F or "
                         "P)")  # fmt: skip
    if found.group(5):
        return "closure", int(found.group(5))
    if found.group(4):
        return "cycle", int(found.group(4))
    return (kind.split("-")[0], int(found.group(2))) if found.group(2) else (kind, None)


def link_of(kind: str) -> tuple[str, int]:
    """(the first record's kind, k) of an n or budget record's kind (module docstring,
    "Successors"): ``("n-c1", 1)`` for ``n-c1``, ``("n-c1", 2)`` for ``n-c1-2``; ``ValueError``
    for another kind."""
    found = _KIND.fullmatch(kind) if isinstance(kind, str) else None
    if found is None or not found.group(2):
        raise ValueError(f"{kind!r} is not an n or budget record's kind")
    return (kind.rsplit("-", 1)[0], int(found.group(3))) if found.group(3) else (kind, 1)


def successor_kind(kind: str, k: int) -> str:
    """The kind of the k-th link (k ≥ 2) of the chain of ``kind``, an n or budget record's first
    kind: ``successor_kind("n-c1", 2) == "n-c1-2"``."""
    if link_of(kind)[1] != 1 or type(k) is not int or k < 2:
        raise ValueError(f"no successor {k!r} of {kind!r}: a successor numbers k = 2, 3, ... of "
                         "an n-c<N> or budget-c<N> record")  # fmt: skip
    return f"{kind}-{k}"


def chain(
    runs_root: str | Path, kind: str, *, repo: str | Path = REPO_ROOT
) -> list[dict[str, Any]]:
    """The chain of an n or budget record (module docstring, "Successors"): every link there, the
    first record and its successors, in the order of their numbers, each ``{"kind", "path",
    "sha256"}`` (the sha256 None for a link that resolves under a sealed folder, never opened, or
    that cannot be read). A gap shows as a missing number. ``kind``: the first record's kind;
    ``repo`` is the checkout the records' evidence is read in (unused here: the chain is read
    from RUNS_DIR alone)."""
    base, own = link_of(kind)
    if own != 1:
        raise ValueError(f"{kind!r} is a successor: a chain is named by its first record's kind")
    root = Path(runs_root)
    folder = root.joinpath(*DIRECTORY)
    found = []
    for path in sorted(folder.glob(f"{base}*.json")) if folder.is_dir() else []:
        try:
            first, number = link_of(path.name[: -len(".json")])
        except ValueError:
            continue
        if first == base:
            found.append((number, path))
    out = []
    for _, path in sorted(found):
        sha = None
        if not _under_sealed(path, root) and path.is_file():
            try:
                sha = _digest(path)
            except OSError:
                sha = None
        out.append({"kind": path.name[: -len(".json")], "path": str(path), "sha256": sha})
    return out


def next_successor(runs_root: str | Path, kind: str) -> str:
    """The kind the next successor of ``kind`` (an n or budget record's first kind) takes: one
    above the highest link there. ``ValueError`` when there is no record to succeed."""
    links = chain(runs_root, kind)
    if not links:
        raise ValueError(f"{kind}: no record is there to succeed")
    return successor_kind(kind, max(link_of(link["kind"])[1] for link in links) + 1)


def confirmation_started(runs_root: str | Path, candidate: int) -> bool:
    """Whether candidate N's confirmation has started (module docstring, "The confirmation
    boundary"): a study-3 start of the candidate in the launch journal whose campaign header
    names the step ``confirmation``. A journal that holds logs and cannot be read fails closed
    (True): that the confirmation has not started cannot be shown."""
    root = Path(runs_root)
    if not any((root / "launch").glob("*.jsonl")):
        return False
    try:
        journal = _read_journal(root, {})
    except (OSError, ValueError):
        return True
    for _, start in journal.starts:
        header = start.get("header") if isinstance(start, Mapping) else None
        if (isinstance(header, Mapping) and header.get("study") == 3
                and type(header.get("candidate")) is int and header["candidate"] == candidate
                and header.get("step") == "confirmation"):  # fmt: skip
            return True
    return False


def end_reason(outcome: Any) -> str | None:
    """The reason id an end line with the launcher's ``outcome`` would give a closure
    (``launcher_<outcome>``), None for an outcome that names none: a closure may not carry it
    (``NOT_ACCEPTED``)."""
    return (
        f"launcher_{outcome}" if isinstance(outcome, str) and _OUTCOME.fullmatch(outcome) else None
    )


def _end_reason_id(reason: Any) -> bool:
    """Whether ``reason`` is the reason id of an end's outcome (``launcher_<outcome>``)."""
    return (
        isinstance(reason, str)
        and reason.startswith("launcher_")
        and end_reason(reason[len("launcher_") :]) == reason
    )


def hold_reasons(text: Any) -> list[str]:
    """The ids (``HOLD_REASONS``) of the reasons a hold line states, in the order it states them:
    its reasons are joined with "; ", and each is known by the text it starts with."""
    if not isinstance(text, str):
        return []
    out: list[str] = []
    for item in text.split("; "):
        for key, start in HOLD_REASONS.items():
            if item.startswith(start) and key not in out:
                out.append(key)
    return out


def decision_path(runs_root: str | Path, kind: str) -> Path:
    """``RUNS_DIR/study3/decisions/<kind>.json``."""
    kind_of(kind)
    return Path(runs_root).joinpath(*DIRECTORY, f"{kind}.json")


def cycle_path(runs_root: str | Path, candidate: int) -> Path:
    """The candidate's cycle record, ``RUNS_DIR/study3/decisions/cycle-c<N>.json``."""
    return decision_path(runs_root, f"cycle-c{candidate}")


def bridge_name(candidate: int) -> str:
    """The cycle's bridge under RUNS_DIR, ``qualification/c<N>/supersede-1.json``: the
    supersession record ``scripts/qualify.py`` reads."""
    return f"qualification/c{candidate}/supersede-1.json"


def one_line(value: Any) -> str | None:
    """Why ``value`` is no diagnosis, fix or justification (one line, not blank, at most
    ``TEXT_MAX`` characters), or None."""
    if not isinstance(value, str):
        return f"is {value!r}, not a line of text"
    if not value.strip():
        return "is blank"
    if "\n" in value or "\r" in value:
        return "is more than one line"
    if len(value) > TEXT_MAX:
        return f"is longer than {TEXT_MAX} characters"
    return None


def pinned_end(lines: Sequence[tuple[int, bytes, Any]], count: Any) -> int | None:
    """The end of a launch log's first ``count`` whole lines (``log_lines``' lines: each ended by
    its newline, which the prefix includes), or None when the log holds fewer whole lines."""
    if type(count) is not int or count < 0 or count > len(lines):
        return None
    return lines[count - 1][0] + len(lines[count - 1][1]) + 1 if count else 0


def serialized(record: Mapping[str, Any]) -> bytes:
    """A decision record's bytes as ``write_decision`` writes them: their sha256 is the file's."""
    return (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def cap_of(n: int) -> int:
    """N + max(10, ⌈0.12·N⌉), in whole numbers."""
    return n + max(CAP_FLOOR, -(-CAP_SHARE_PERCENT * n // 100))


def budget_reason(n: int, simulation: Mapping[str, Any] | None = None) -> str | None:
    """Why N asks for the researcher's decision on the extra blocks and their price: N above 90,
    or N raised for power (the simulation record's ``power_af`` is short, so N is at least its
    ``n_min``), even when N is 90 or less. None: no budget decision is needed."""
    reasons = []
    if n > BUDGET_ABOVE:
        reasons.append(f"N = {n} is above {BUDGET_ABOVE}")
    gate = ((simulation or {}).get("gates") or {}).get("power_af")
    if isinstance(gate, Mapping) and gate.get("power_short") is True:
        reasons.append(f"N is raised to at least {gate.get('n_min')} for the simulated power")
    return "; ".join(reasons) or None


def budget_needed(n: int, simulation: Mapping[str, Any] | None = None) -> bool:
    """Whether a budget decision must answer N (``budget_reason``)."""
    return budget_reason(n, simulation) is not None


# ------------------------------------------------------------------ reading


def read_decision(
    runs_root: str | Path, kind: str, *, repo: str | Path = REPO_ROOT
) -> tuple[dict[str, Any] | None, list[str]]:
    """The record ``kind`` if it is present and valid, and its problems. Absent (no file, and no
    link in its place): (None, []); a link that cannot be resolved (a link loop) is refused
    unopened, a problem. With any problem the record is None: nothing downstream may consume it.
    ``repo``: the checkout that evidence with ``"root": "repo"`` is relative to."""
    path = decision_path(runs_root, kind)
    if not (path.exists() or path.is_symlink()):  # a link that leads nowhere is no absence
        return None, []
    if _under_sealed(path, Path(runs_root)):
        return None, [_sealed_problem(path)]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, [f"{path}: unreadable ({type(exc).__name__})"]
    problems = _problems(Path(runs_root), kind, record, Path(repo))
    return (None, [f"{path}: {p}" for p in problems]) if problems else (record, [])


def decisions(
    runs_root: str | Path, candidate: int, *, repo: str | Path = REPO_ROOT
) -> dict[str, dict[str, Any]]:
    """The decision records of a candidate's route, by kind: ``engineering``, ``simulation``,
    ``n-c<candidate>`` and ``budget-c<candidate>``. Each is ``{"kind", "path", "state",
    "record", "problems"}``: the state is ``passed`` or ``failed`` for a valid record,
    ``missing`` for none, ``invalid`` for one with a problem (its record None), the immutability
    check against the launch journal included.

    The n and budget entries are their chains' links in force (module docstring,
    "Successors"), each with ``chain``, the path of every link: the entry's kind, path and record
    the link's, its problems those of the chain too.

    A record rests on the ones before it: the ``n`` entry is ``invalid`` when the simulation entry
    is, and the budget entry when the ``n`` entry is (a budget decision that the ``n`` record
    would have asked for cannot then be told, so it is listed, invalid, rather than dropped). The
    budget entry is listed when a passed ``n`` record asks for a budget decision
    (``budget_reason``), or one is written; a needed one that is not written is ``missing`` with
    that as its problem. A failed ``n`` record (its cap beyond the reserved seeds) needs none."""
    if candidate not in (1, 2) or isinstance(candidate, bool):
        raise ValueError(f"candidate {candidate!r} is not 1 or 2")
    root, repo = Path(runs_root), Path(repo)
    starts, corrupt, sealed = _starts(root)
    kind_n, budget = f"n-c{candidate}", f"budget-c{candidate}"
    out = {kind: _entry(root, kind, starts, corrupt, sealed, repo)
           for kind in ("engineering", "simulation")}  # fmt: skip
    out[kind_n] = _linked(root, kind_n, starts, corrupt, sealed, repo)
    simulation, n = out["simulation"], out[kind_n]
    if simulation["state"] == "invalid" and n["state"] in STATES:
        invalidate(n, f"the simulation record it rests on is invalid ({simulation['path']})")
    reason = None
    if n["record"] is not None and n["record"]["state"] == "passed":
        reason = budget_reason(n["record"]["N"], simulation["record"])
    if reason is not None or n["state"] == "invalid" or chain(root, budget):
        out[budget] = _linked(root, budget, starts, corrupt, sealed, repo)
        if n["state"] == "invalid":
            invalidate(out[budget], f"the n-c{candidate} record it answers is invalid "
                                     f"({n['path']})")  # fmt: skip
        elif reason is not None and out[budget]["state"] == "missing":
            out[budget]["problems"] = [f"{out[budget]['path']}: a budget decision answering the "
                                       f"n-c{candidate} record is needed ({reason}) and is not "
                                       "written"]  # fmt: skip
    return out


def closures(
    runs_root: str | Path, candidate: int, *, repo: str | Path = REPO_ROOT
) -> dict[str, dict[str, Any]]:
    """The closure records of a candidate, one per arm (``closure-c<N>-A``, ``-F``, ``-P``),
    each ``{"kind", "path", "state", "record", "problems"}`` as ``decisions`` lists the others:
    ``failed`` for a valid record (a closure is a terminal technical outcome), ``missing`` for none,
    ``invalid`` for one with a problem, the immutability check included."""
    if candidate not in (1, 2) or isinstance(candidate, bool):
        raise ValueError(f"candidate {candidate!r} is not 1 or 2")
    root, repo = Path(runs_root), Path(repo)
    starts, corrupt, sealed = _starts(root)
    return {kind: _entry(root, kind, starts, corrupt, sealed, repo)
            for kind in (f"closure-c{candidate}-{arm}" for arm in CLOSURE_DECISIONS)}  # fmt: skip


def invalidate(entry: dict[str, Any], why: str) -> None:
    """Mark a ``decisions`` entry invalid, with why: its record is no longer consumed."""
    entry.update(state="invalid", record=None,
                 problems=[*entry["problems"], f"{entry['path']}: {why}"])  # fmt: skip


def _entry(root: Path, kind: str, starts: Starts | None, corrupt: Sequence[str],
           sealed: Sequence[str], repo: Path) -> dict[str, Any]:  # fmt: skip
    path = decision_path(root, kind)
    record, problems = read_decision(root, kind, repo=repo)
    problems = problems + [p for p in immutable(path, starts, corrupt=corrupt, sealed=sealed)
                           if p not in problems]  # fmt: skip
    state = "invalid" if problems else "missing" if record is None else record["state"]
    return {"kind": kind, "path": str(path), "state": state,
            "record": None if problems else record, "problems": problems}  # fmt: skip


def _linked(root: Path, kind: str, starts: Starts | None, corrupt: Sequence[str],
            sealed: Sequence[str], repo: Path) -> dict[str, Any]:  # fmt: skip
    """The entry of an n or budget record's chain (module docstring, "Successors"): its link in
    force checked in full (``_entry``), every earlier link for its integrity (there, of the
    sha256 the link after it names, ``immutable``), a gap a problem; ``chain``: every link's
    path."""
    links = chain(root, kind)
    if len(links) <= 1:
        return {**_entry(root, kind, starts, corrupt, sealed, repo),
                "chain": [link["path"] for link in links]}  # fmt: skip
    problems: list[str] = []
    numbers = [link_of(link["kind"])[1] for link in links]
    if numbers != list(range(1, len(links) + 1)):
        problems.append(f"{decision_path(root, kind)}: its chain {[x['kind'] for x in links]} "
                        "has a gap: each successor follows the record before it")  # fmt: skip
    for before, after in zip(links, links[1:], strict=False):
        problems += [p for p in immutable(before["path"], starts, corrupt=corrupt, sealed=sealed)
                     if p not in problems]  # fmt: skip
        cited = None
        if not _under_sealed(Path(after["path"]), root):
            try:
                data = json.loads(Path(after["path"]).read_text(encoding="utf-8"))
                cited = (data.get("supersedes") or {}).get("sha256")
            except (OSError, ValueError, AttributeError):
                cited = None
        if before["sha256"] is None or cited != before["sha256"]:
            problems.append(f"{after['path']}: it names {before['kind']} with sha256 {cited!r}, "
                            f"not its sha256 now {before['sha256']!r}: a link is unchanged once "
                            "its successor names it")  # fmt: skip
    entry = _entry(root, links[-1]["kind"], starts, corrupt, sealed, repo)
    problems += [p for p in entry["problems"] if p not in problems]
    return {**entry, "state": "invalid" if problems else entry["state"],
            "record": None if problems else entry["record"], "problems": problems,
            "chain": [link["path"] for link in links]}  # fmt: skip


def _starts(root: Path) -> tuple[Starts | None, list[str], list[str]]:
    """The launch journal's study-3 starts, its corrupt lines and the logs that resolve under a
    sealed folder, never opened (by their paths under RUNS_DIR: ``study3_records``' reading of
    it); (None, [], []) when there is no journal or it cannot be read."""
    if not any((root / "launch").glob("*.jsonl")):
        return None, [], []
    try:
        journal = _read_journal(root, {})
    except (OSError, ValueError):
        return None, [], []
    return journal.starts, journal.corrupt, journal.sealed


def immutable(
    record_path: str | Path,
    journal: Starts | None,
    *,
    corrupt: Sequence[str] = (),
    sealed: Sequence[str] = (),
) -> list[str]:
    """Problems with a record that a started campaign named and that has since changed: the
    campaign's header named the record (module docstring) with a sha256 that is not the file's
    now, or with none that can be read, or the file is gone. A record no campaign named is free
    of problems. ``journal``: the launch journal's starts, ``(campaign, start record)``; None
    (a missing or unreadable journal) fails closed for a record that exists, and so do
    ``corrupt`` lines (``"<log> line <n>"``, the journal's unresolved torn lines before the last
    of their log) and ``sealed`` logs (their paths under RUNS_DIR: a log that resolves under a
    sealed folder, never opened): a campaign that named the record may be what they lost."""
    path = Path(record_path)
    if path.exists() and _under_sealed(path, _runs_of(path)):
        return [_sealed_problem(path)]
    if journal is None:
        return ([f"{path}: the launch journal is missing or unreadable, so whether a campaign "
                 "has named this record cannot be shown"] if path.exists() else [])  # fmt: skip
    problems = set()
    if corrupt and path.exists():
        problems.add(f"{path}: the launch journal has {TORN_LINE} ({', '.join(corrupt)}), so "
                     "whether a campaign has named this record cannot be shown")  # fmt: skip
    if sealed and path.exists():
        root = _runs_of(path)
        loops = [where for where in sealed if unresolvable(root / where)]
        for logs, why in (([w for w in sealed if w not in loops], "resolves under a sealed folder"),
                          (loops, LOOP_WHY)):  # fmt: skip
            if logs:
                problems.add(f"{path}: a launch log {why} ({', '.join(logs)}), so whether a "
                             "campaign has named this record cannot be shown")  # fmt: skip
    now = _digest(path) if path.is_file() else None
    for campaign, start in journal:
        header = start.get("header") if isinstance(start, Mapping) else None
        for _, sha in _named(header, path):
            if now is None:
                problems.add(f"{path}: campaign {campaign} named this record, which is no "
                             "longer there")  # fmt: skip
            elif not (isinstance(sha, str) and _SHA256.fullmatch(sha)):
                problems.add(f"{path}: campaign {campaign} named this record without a "
                             f"readable sha256 ({sha!r})")  # fmt: skip
            elif sha != now:
                problems.add(f"{path}: campaign {campaign} named this record with sha256 "
                             f"{sha}, and it is now {now}")  # fmt: skip
    return sorted(problems)


def _named(header: Any, path: Path) -> list[tuple[str, Any]]:
    """The references to ``path`` in a campaign header, anywhere in it: (path as named, sha256
    as named or None). A reference is a string path with its sha256 beside it, in ``sha256`` when
    the key is ``path`` and in ``<key>_sha256`` otherwise; a path among a list's items has none
    (so it is named without a readable sha256)."""
    found: list[tuple[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if isinstance(value, str) and _same(value, path):
                    found.append((value, node.get("sha256" if key == "path" else f"{key}_sha256")))
                else:
                    walk(value)
        elif isinstance(node, list | tuple):
            for value in node:
                if isinstance(value, str) and _same(value, path):
                    found.append((value, None))  # a list holds no sha256 beside its strings
                else:
                    walk(value)

    walk(header)
    return found


def _same(named: str, path: Path) -> bool:
    """Whether a named path is the record's, by the last three components."""
    tail = PurePosixPath(named.replace("\\", "/")).parts[-3:]
    return len(tail) == 3 and tail == PurePosixPath(path.as_posix()).parts[-3:]


# ------------------------------------------------------------------ writing


def write_decision(
    runs_root: str | Path, kind: str, record: Mapping[str, Any], *, repo: str | Path = REPO_ROOT
) -> Path:
    """Validate ``record`` as ``kind`` (evidence read as it is now, the records it answers read
    from ``runs_root``) and write it atomically, once. ``DecisionRefused`` for an invalid record
    (its problems named) or for one already written: a second version is never written. Returns
    the path."""
    path = decision_path(runs_root, kind)
    if _under_sealed(path, Path(runs_root)):
        raise DecisionRefused(_sealed_problem(path))
    if path.exists():
        raise DecisionRefused(f"{path} is already written: a decision record is never rewritten")
    try:
        record = json.loads(json.dumps(record, allow_nan=False))  # what a reader will see
    except (TypeError, ValueError) as exc:
        raise DecisionRefused(f"{kind}: not a JSON record ({exc})") from exc
    problems = _problems(Path(runs_root), kind, record, Path(repo))
    family, candidate = kind_of(kind) if _KIND.fullmatch(str(kind)) else (None, None)
    if family in ("n", "budget"):  # a record a cycle lists as stale: under the code in force
        code = record.get("code") if isinstance(record, dict) else None
        sha = code.get("gpu_code_sha256") if isinstance(code, dict) else None
        problems += after_cycle_code(runs_root, sha, candidate=candidate, repo=repo)
    if problems:
        raise DecisionRefused(f"{kind}: not written, {len(problems)} problem(s): "
                              + "; ".join(problems))  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish(path, serialized(record))
    return path


def after_cycle_code(
    runs_root: str | Path,
    code: Any,
    *,
    candidate: int | None = None,
    repo: str | Path = REPO_ROOT,
) -> list[str]:
    """Why a write-once record under GPU code ``code`` is not written now (module docstring,
    "Successors"): once ``candidate`` has a cycle record outside the qualification, its record
    carries the cycle's fix, the code in force for its runs (a cycle record that is there and not
    valid refuses it until it is valid); and no record is written under the old code of any
    candidate's valid cycle record. The records a cycle lists as stale (``affected.records``:
    the fixations, the n and budget records and the versions record of its old code there when it
    was written) then change after it only when a file is tampered with. No problem before any
    cycle."""
    out: list[str] = []
    for c in (1, 2):
        path = cycle_path(runs_root, c)
        if not (path.exists() or path.is_symlink()):
            continue
        record, problems = read_decision(runs_root, f"cycle-c{c}", repo=repo)
        if record is None:
            if c == candidate:
                out.append(f"candidate {c}'s cycle record {path} is not valid ("
                           + "; ".join(problems[:3]) + f"): no record of candidate {c} is "
                           "written until it is")  # fmt: skip
            continue
        fix, old = record["code"]["gpu_code_sha256"], record["old_code"]
        if c == candidate and code != fix:
            out.append(f"its GPU code {code} is not candidate {c}'s code in force, {fix} (the fix "
                       f"of its cycle): after the cycle a record of candidate {c} carries the "
                       "fix")  # fmt: skip
        elif code == old:
            out.append(f"its GPU code {code} is the old code of candidate {c}'s cycle: no record "
                       "is written under it after the cycle")  # fmt: skip
    return out


def _publish(path: Path, data: bytes) -> None:
    """The file at ``path``, whole or not at all, and only if there is none: written beside it
    and linked into place, which fails when the record exists. Where the filesystem has no hard
    links, created exclusively and written in place."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    taken = DecisionRefused(f"{path} is already written: a decision record is never rewritten")
    with temporary.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise taken from exc
        except OSError:
            try:
                with path.open("xb") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            except FileExistsError as exc:
                raise taken from exc
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


# ------------------------------------------------------------------ validation


def _problems(root: Path, kind: str, record: Any, repo: Path) -> list[str]:
    """Every problem of ``record`` as ``kind``: its schema, the evidence now, and the records it
    answers."""
    family, candidate = kind_of(kind)
    check = _Schema(kind, family, candidate)
    check.record(record)
    problems = list(check.problems)
    if problems:
        return problems  # nothing is read from a record that is not well formed
    problems += _evidence(check.refs, root, repo)
    if family == "n":
        problems += _against_simulation(root, record, candidate, repo)
    elif family == "budget":
        problems += _against_n(root, kind, record, candidate, repo)
    if family in ("n", "budget") and link_of(kind)[1] > 1:
        problems += _against_predecessor(root, kind, record, candidate, repo)
    elif family == "closure":
        problems += _against_line(root, record, candidate)
    elif family == "cycle":
        problems += _still_absent(check.absent, root, repo)
        problems += _against_journal(root, record)
    return problems


class _Schema:
    """The closed schema of one kind: ``problems`` and the evidence ``refs`` of a record."""

    def __init__(self, kind: str, family: str, candidate: int | None) -> None:
        self.kind, self.family, self.candidate = kind, family, candidate
        self.problems: list[str] = []
        self.refs: list[_Ref] = []
        self.absent: list[_Ref] = []  # the absence markers: files that must not be there

    # -- the building blocks

    def add(self, where: str, text: str) -> None:
        self.problems.append(f"{where}: {text}" if where else text)

    def object(self, where, value, required, optional=()) -> dict[str, Any] | None:
        """``value`` as an object with the fields: None when it is not one or misses one; a field
        that is not named is a problem (the schemas are closed)."""
        if not isinstance(value, dict):
            self.add(where, f"is not an object ({type(value).__name__})")
            return None
        complete = True
        for key in required:
            if key not in value:
                self.add(where, f"has no field `{key}`")
                complete = False
        for key in value:
            if key not in (*required, *optional):
                hint = (" (no mean, sign, per-seed value, unrounded spread or exploratory "
                        "diagnostic is recorded)" if _HIDDEN.search(str(key)) else "")  # fmt: skip
                self.add(where, f"`{key}` is not a field of this record{hint}")
        return value if complete else None

    def integer(self, where, value, low=None, high=None) -> bool:
        ok = (
            type(value) is int and (low is None or value >= low) and (high is None or value <= high)
        )
        if not ok:
            bound = (
                ""
                if low is None
                else f" at least {low}"
                if high is None
                else f" in [{low}, {high}]"
            )
            self.add(where, f"is {value!r}, not an integer{bound}")
        return ok

    def real(self, where, value, *, positive=False) -> bool:
        ok = (isinstance(value, int | float) and not isinstance(value, bool)
              and math.isfinite(value) and (value > 0 if positive else True))  # fmt: skip
        if not ok:
            self.add(where, f"is {value!r}, not a finite {'positive ' if positive else ''}number")
        return ok

    def boolean(self, where, value) -> bool:
        if not isinstance(value, bool):
            self.add(where, f"is {value!r}, not true or false")
        return isinstance(value, bool)

    def text(self, where, value) -> bool:
        ok = isinstance(value, str) and value.strip() != ""
        if not ok:
            self.add(where, f"is {value!r}, not a non-empty string")
        return ok

    def level(self, where, value, *, null=False) -> float | None:
        """The ladder's level that ``value`` is (None for a null where it is allowed, or a
        problem)."""
        if value is None and null:
            return None
        found = _level(value)
        if found is None:
            self.add(
                where, f"is {value!r}, not one of {list(LEVELS)}" + (" or null" if null else "")
            )
        return found

    def ref(self, where, value, extra=()) -> dict[str, Any] | None:
        """Evidence by path and sha256 (and the fields ``extra``); the reference is kept to be
        checked against the file."""
        entry = self.object(where, value, ("path", "sha256", *extra), ("root",))
        if entry is None:
            return None
        ok = self.text(f"{where}.path", entry["path"])
        if not (isinstance(entry["sha256"], str) and _SHA256.fullmatch(entry["sha256"])):
            self.add(f"{where}.sha256", f"is {entry['sha256']!r}, not a sha256 (64 hexadecimal "
                     "digits)")  # fmt: skip
            ok = False
        root = entry.get("root", "runs")
        if root not in ("runs", "repo"):
            self.add(f"{where}.root", f"is {root!r}, not runs or repo")
            ok = False
        if ok:
            self.refs.append(_Ref(where, entry["path"], entry["sha256"], root))
        return entry

    # -- the common fields

    def record(self, record: Any) -> None:
        top = self.object("", record, (*COMMON, *FIELDS[self.family]),
                          OPTIONAL.get(self.family, ()))  # fmt: skip
        if top is None:
            return
        if top["kind"] != self.kind:
            self.add("kind", f"is {top['kind']!r}, not {self.kind!r}")
        if top["state"] not in STATES:
            self.add("state", f"is {top['state']!r}, not one of {list(STATES)}")
        self._when(top["written_at"])
        code = self.object("code", top["code"], ("commit", "gpu_code_sha256"))
        if code is not None:
            for key, pattern, what in (("commit", _COMMIT, "a commit (40 hexadecimal digits)"),
                                       ("gpu_code_sha256", _SHA256, "a sha256")):  # fmt: skip
                if not (isinstance(code[key], str) and pattern.fullmatch(code[key])):
                    self.add(f"code.{key}", f"is {code[key]!r}, not {what}")
        getattr(self, f"_{self.family}")(top)

    def _when(self, value: Any) -> None:
        try:
            when = datetime.fromisoformat(value)
            ok = when.tzinfo is not None and when.utcoffset() == timedelta(0)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            self.add("written_at", f"is {value!r}, not an ISO 8601 time in UTC (Z or +00:00)")

    # -- engineering

    def _engineering(self, r: dict[str, Any]) -> None:
        checks = self.object("checks", r["checks"], CHECKS)
        for name in CHECKS if checks is not None else ():
            where, entry = f"checks.{name}", checks[name]
            if entry is None:
                if r["state"] == "passed":
                    self.add(where, "is null: a passed record rests on a record of every check")
                continue
            entry = self.ref(where, entry, ("passed",))
            if entry is None or not self.boolean(f"{where}.passed", entry["passed"]):
                continue
            if r["state"] == "passed" and not entry["passed"]:
                self.add(where, "did not pass, and the record says it passed")

    # -- simulation

    def _simulation(self, r: dict[str, Any]) -> None:
        self.ref("fixation", r["fixation"])
        inputs = r["inputs"]
        if not isinstance(inputs, dict) or not inputs:
            self.add("inputs", "is not a non-empty object of evidence")
        else:
            for name, value in inputs.items():
                if not _INPUT_NAME.fullmatch(name) or _HIDDEN.search(name):
                    self.add("inputs", f"the name {name!r} is not a lowercase word of letters "
                             "and underscores (at most 40, no digits) that names an input and "
                             "no mean, sign, per-seed value, spread or diagnostic")  # fmt: skip
                self.ref(f"inputs.{name}", value)
        safe = self._gates(r)
        entries = self._coverage(r)
        level = self.level("level", r["level"], null=True)
        blocks, failed = r["validation_blocks"], r["state"] == "failed"
        if not (blocks in VALIDATION_BLOCKS and type(blocks) is int or blocks is None and failed):
            self.add("validation_blocks", f"is {blocks!r}, not one of {list(VALIDATION_BLOCKS)}"
                     + (" or null in a failed record" if failed else ""))  # fmt: skip
        if safe is not None and blocks in VALIDATION_BLOCKS and type(blocks) is int:
            wanted = VALIDATION_BLOCKS[0] if safe["passed"] else VALIDATION_BLOCKS[1]
            if blocks != wanted:
                self.add("validation_blocks", f"is {blocks}, but the pass rate {safe['value']} at "
                         f"the safe alternative {'meets' if safe['passed'] else 'is below'} "
                         f"{safe['requirement']}: the validation has {wanted} blocks")  # fmt: skip
        verified = self._verification(r, level)
        if entries is not None:
            self._ladder(level, entries)
            self._tests(entries, r["verification"])
        # The state follows the coverage screen and its verification, and nothing else: a pass rate
        # below its requirement sets the validation size, a power shortfall is recorded (power_af)
        if r["state"] == "passed":
            if level is None:
                self.add("level", "is null in a passed record")
            if verified is not True:
                self.add("verification", "did not pass, and the record says it passed")
        elif level is not None and verified is True:
            self.add("state", f"is {r['state']!r}, but the coverage screen passed at {level} and "
                     "so did its verification: the state follows both, and no gate")  # fmt: skip

    def _tests(self, entries: list[dict[str, Any]], verification: Any) -> None:
        """The coverage tests of every level and of the verification are among the J frozen
        tests."""
        tests = verification.get("tests") if isinstance(verification, dict) else None
        if type(tests) is not int or tests < 1:
            return
        used = sum(e["tests"] for e in entries)
        if used + tests > entries[0]["J"]:
            self.add("coverage", f"the tests of the levels ({used}) and of the verification "
                     f"({tests}) add up to {used + tests}, above the frozen J = "
                     f"{entries[0]['J']}")  # fmt: skip

    def _gates(self, r: dict[str, Any]) -> dict[str, Any] | None:
        """Checks the gates' entries; returns the ``safe_pass_rate`` entry if it is well formed.
        A gate has ``value``, ``requirement`` and ``passed`` (null for both of the last two when
        it is only reported); ``power_af`` also records ``power_short`` (its requirement not met)
        and, then, ``n_min``, the smallest N at which the simulated power reaches the
        requirement."""
        gates, safe = r["gates"], None
        if not isinstance(gates, dict):
            self.add("gates", "is not an object")
            return None
        for name in REQUIRED_GATES:
            if name not in gates:
                self.add("gates", f"has no gate `{name}`")
        for name, gate in gates.items():
            where = f"gates.{name}"
            if name not in REQUIRED_GATES and name not in REPORTED_GATES:
                self.add(where, f"is not a gate of the simulation record (the gates are "
                         f"{[*REQUIRED_GATES, *REPORTED_GATES]})")  # fmt: skip
                continue
            extra = ("power_short", "n_min") if name == "power_af" else ()
            gate = self.object(where, gate, ("value", "requirement", "passed", *extra))
            if gate is None:
                continue
            value_ok = self.real(f"{where}.value", gate["value"])
            required = gate["requirement"]
            if required is None:
                if gate["passed"] is not None:
                    self.add(f"{where}.passed", "must be null for a gate that is only reported")
                if name in REQUIRED_GATES:
                    self.add(where, "is only reported, and a required gate has its requirement")
                continue
            if name in REPORTED_GATES:
                self.add(f"{where}.requirement", f"is {required!r}: this gate is only reported, "
                         "its requirement is null")  # fmt: skip
                continue
            if not (self.real(f"{where}.requirement", required)
                    and self.boolean(f"{where}.passed", gate["passed"]) and value_ok):  # fmt: skip
                continue
            if name in REQUIRED_GATES and required < REQUIRED_GATES[name]:
                self.add(f"{where}.requirement", f"is {required}, below the registered "
                         f"{REQUIRED_GATES[name]}")  # fmt: skip
            if gate["passed"] != (gate["value"] >= required):
                self.add(where, f"says passed={gate['passed']} for {gate['value']} against "
                         f"{required}")  # fmt: skip
            if name == "power_af":
                self._power(where, gate)
            if name == "safe_pass_rate":
                safe = gate
        return safe

    def _power(self, where: str, gate: dict[str, Any]) -> None:
        """``power_short`` is the requirement not met; ``n_min`` is then an N (at least 60)."""
        short = gate["power_short"]
        if not self.boolean(f"{where}.power_short", short):
            return
        if short == gate["passed"]:
            self.add(f"{where}.power_short", f"is {short}, but passed is {gate['passed']}: the "
                     "power is short when its requirement is not met")  # fmt: skip
        if short:
            self.integer(f"{where}.n_min", gate["n_min"], MIN_N)
        elif gate["n_min"] is not None:
            self.add(f"{where}.n_min", f"is {gate['n_min']!r}: it is null when the power is not "
                     "short")  # fmt: skip

    def _coverage(self, r: dict[str, Any]) -> list[dict[str, Any]] | None:
        """The coverage entries, or None when one is not well formed or its thresholds are not
        the recomputed ones."""
        rows = r["coverage"]
        if not isinstance(rows, list) or not rows:
            self.add("coverage", "is not a non-empty list of levels tested")
            return None
        ok, before, frozen = True, None, None
        for i, row in enumerate(rows):
            where = f"coverage[{i}]"
            row = self.object(where, row, ("level", "tests", "J", "M", "counts", "thresholds",
                                           "passed"))  # fmt: skip
            if row is None:
                ok = False
                continue
            level = self.level(f"{where}.level", row["level"])
            if level is not None and before is not None and not before < level:
                self.add(f"{where}.level", "is not above the level before it (each level is "
                         "listed once, in ascending order)")  # fmt: skip
            before = level if level is not None else before
            good = [level is not None,
                    self.integer(f"{where}.tests", row["tests"], 1),
                    self.integer(f"{where}.J", row["J"], 1),
                    self.integer(f"{where}.M", row["M"]),
                    self.boolean(f"{where}.passed", row["passed"])]  # fmt: skip
            if good[3] and row["M"] != COVERAGE_RUNS:
                self.add(f"{where}.M", f"is {row['M']}, not the registered {COVERAGE_RUNS} "
                         "simulated studies per test")  # fmt: skip
                good[3] = False
            if good[2]:
                frozen = row["J"] if frozen is None else frozen
                if row["J"] != frozen:
                    self.add(f"{where}.J", f"is {row['J']}, not {frozen} of the first level: J is "
                             "one frozen number for the whole screen")  # fmt: skip
                    good[2] = False
            counts = self.object(f"{where}.counts", row["counts"], tuple(COVERAGE_P0))
            thresholds = self.object(f"{where}.thresholds", row["thresholds"], tuple(COVERAGE_P0))
            if not all(good) or counts is None or thresholds is None:
                ok = False
                continue
            if row["tests"] > row["J"]:
                self.add(f"{where}.tests", f"{row['tests']} tests at one level exceed the total "
                         f"J = {row['J']}")  # fmt: skip
            valid, agree = True, True
            for name, p0 in COVERAGE_P0.items():
                if not self.integer(f"{where}.counts.{name}", counts[name], 0, row["M"]):
                    valid = False
                    continue
                expected = coverage_threshold(row["J"], row["M"], p0)
                if thresholds[name] != expected or type(thresholds[name]) is not int:
                    self.add(f"{where}.thresholds.{name}", f"is {thresholds[name]!r}, but "
                             f"BinomQuantile(1 − {FAMILY_ALPHA}/{row['J']}; {row['M']}, {p0}) "
                             f"is {expected}")  # fmt: skip
                    valid = False
                else:
                    agree = agree and counts[name] <= expected
            if valid and row["passed"] is not agree:
                self.add(f"{where}.passed", "disagrees with the counts and the thresholds")
                valid = False
            ok = ok and valid
        return rows if ok else None

    def _ladder(self, level: float | None, entries: list[dict[str, Any]]) -> None:
        """L is the first level whose tests passed, the ones below it tested; with no passing
        level, L is null and every level was tested."""
        first = next((e["level"] for e in entries if e["passed"]), None)
        tested = [_level(e["level"]) for e in entries]
        if level is None:
            if first is not None:
                self.add("level", f"is null, but the tests at level {first} passed")
            elif tested != list(LEVELS):
                self.add("coverage", f"a null level needs every level tested, not {tested}")
            return
        if first is None or not math.isclose(first, level, rel_tol=1e-12):
            self.add("level", f"is {level}, but the first level whose tests passed is {first}")
        below = [lv for lv in LEVELS if lv <= level]
        if tested[: len(below)] != below:
            self.add("coverage", f"does not list every level up to {level} ({tested})")

    def _verification(self, r: dict[str, Any], level: float | None) -> bool | None:
        """Whether the verification passed (None when it is not well formed)."""
        fields = ("level", "passed", "tests", "calibration_stream", "verification_stream")
        v = self.object("verification", r["verification"], fields)
        if v is None:
            return None
        self.integer("verification.tests", v["tests"], 1)
        found = self.level("verification.level", v["level"], null=True)
        valid = self.boolean("verification.passed", v["passed"])
        if (found is None) != (level is None) or (
            found is not None and level is not None and not math.isclose(found, level)
        ):
            self.add("verification.level", f"is {v['level']!r}, not the record's level {level}")
        streams = [
            self.integer(f"verification.{name}", v[name])
            for name in ("calibration_stream", "verification_stream")
        ]
        if all(streams) and v["calibration_stream"] == v["verification_stream"]:
            self.add("verification", "the verification stream is the calibration stream: it is "
                     "not independent")  # fmt: skip
        if valid and level is None and v["passed"]:
            self.add("verification.passed", "is true, but there is no level to verify")
        return v["passed"] if valid else None

    # -- n

    def _n(self, r: dict[str, Any]) -> None:
        if r["candidate"] != self.candidate or type(r["candidate"]) is not int:
            self.add("candidate", f"is {r['candidate']!r}, not {self.candidate}")
        self.level("level", r["level"])
        active = r["active"]
        if active not in ACTIVE:
            self.add("active", f"is {active!r}, not {ACTIVE}")
        self._rates(r["rates"], "P-F" in active if isinstance(active, list) else False)
        self._cohort(r["cohort"])
        spread = self.object("spread", r["spread"], ("script", "output"))
        for name in ("script", "output") if spread is not None else ():
            self.ref(f"spread.{name}", spread[name])
        self._spread_pp(r["spread_pp"])
        self._exact_ref("simulation", r["simulation"], SIMULATION_RECORD)
        if self.integer("N", r["N"], MIN_N):
            cap = cap_of(r["N"])
            if r["cap"] != cap or type(r["cap"]) is not int:
                self.add("cap", f"is {r['cap']!r}, not N + max({CAP_FLOOR}, ⌈0.12·N⌉) = {cap}")
            else:
                self._cap_state(r["state"], cap)
        else:
            self.integer("cap", r["cap"], MIN_N + CAP_FLOOR)
        self._supersedes(r)

    def _supersedes(self, r: dict[str, Any]) -> None:
        """A successor's ``supersedes`` (module docstring, "Successors"): the record before it,
        by its path under the runs directory and its sha256 now, the reason ``cycle_c<N>`` and
        the candidate's cycle record as evidence; the first record carries none."""
        base, k = link_of(self.kind)
        if k == 1:
            if "supersedes" in r:
                self.add("supersedes", f"is a successor's field ({base}-<k>): the first record "
                         "supersedes none")  # fmt: skip
            return
        if "supersedes" not in r:
            self.add("", f"has no field `supersedes`: a successor ({self.kind}) names the record "
                     "before it and the candidate's cycle")  # fmt: skip
            return
        entry = self.ref("supersedes", r["supersedes"], ("reason", "cycle"))
        if entry is None:
            return
        named = "/".join((*DIRECTORY, f"{base if k == 2 else f'{base}-{k - 1}'}.json"))
        if entry["path"] != named or entry.get("root", "runs") != "runs":
            self.add("supersedes.path", f"is {entry['path']!r}, not {named!r} under the runs "
                     "directory, the record before it: a successor names its predecessor (no "
                     "fork, no gap)")  # fmt: skip
        if entry["reason"] != f"cycle_c{self.candidate}":
            self.add("supersedes.reason", f"is {entry['reason']!r}, not 'cycle_c{self.candidate}':"
                     " a successor follows the candidate's cycle")  # fmt: skip
        self._exact_ref("supersedes.cycle", entry["cycle"],
                        "/".join((*DIRECTORY, f"cycle-c{self.candidate}.json")))  # fmt: skip

    def _cap_state(self, state: Any, cap: int) -> None:
        """The state is ``passed`` when the cap fits the reserved confirmation seeds and
        ``failed`` when it does not: the confirmation cannot start within them, and N is never
        cut."""
        reserved = len(SEEDS["confirmation"][self.candidate])
        expected = "passed" if cap <= reserved else "failed"
        if state in STATES and state != expected:
            self.add("state", f"is {state!r}, but the cap {cap} "
                     f"{'fits' if cap <= reserved else 'exceeds'} the {reserved} confirmation "
                     f"seeds reserved, so it is {expected!r}"
                     + ("" if cap <= reserved else ": the confirmation cannot start within the "
                        "registered seeds, and N is not cut"))  # fmt: skip

    def _exact_ref(self, where: str, value: Any, path: str) -> None:
        """Evidence of the record at ``path`` under the runs directory, and no other file."""
        entry = self.ref(where, value)
        if entry is None:
            return
        if entry["path"] != path or entry.get("root", "runs") != "runs":
            self.add(where, f"is {entry['path']!r} under {entry.get('root', 'runs')!r}, not the "
                     f"record {path!r} under the runs directory")  # fmt: skip

    def _rates(self, rates: Any, with_p: bool) -> None:
        arms = ("A", "F", "P") if with_p else ("A", "F")
        rates = self.object("rates", rates, arms)
        for arm in arms if rates is not None else ():
            pair = self.object(f"rates.{arm}", rates[arm], ("lr_l", "lr_u"))
            for key in ("lr_l", "lr_u") if pair is not None else ():
                self.real(f"rates.{arm}.{key}", pair[key], positive=True)

    def _cohort(self, cohort: Any) -> None:
        cohort = self.object("cohort", cohort, ("seeds", "run_ids"))
        if cohort is None:
            return
        seeds, runs = cohort["seeds"], cohort["run_ids"]
        if not (isinstance(seeds, list) and seeds
                and all(type(s) is int for s in seeds)):  # fmt: skip
            self.add("cohort.seeds", "is not a non-empty list of block seeds")
        elif seeds != sorted(set(seeds)):
            self.add("cohort.seeds", "is not in ascending order without repeats")
        else:
            allowed = SEEDS["validation"][self.candidate]
            outside = [s for s in seeds if s not in allowed]
            if outside:
                self.add("cohort.seeds", f"{outside[:3]} are not validation seeds of candidate "
                         f"{self.candidate} ({allowed[0]}–{allowed[-1]})")  # fmt: skip
        if not (isinstance(runs, list) and runs and all(isinstance(x, str) and x for x in runs)):
            self.add("cohort.run_ids", "is not a non-empty list of run ids")
        elif len(set(runs)) != len(runs):
            self.add("cohort.run_ids", "names a run twice")
        elif isinstance(seeds, list) and len(runs) < 4 * len(seeds):
            self.add("cohort.run_ids", f"{len(runs)} runs for {len(seeds)} blocks: each block "
                     "has at least four (A and F at both rates)")  # fmt: skip

    def _spread_pp(self, value: Any) -> None:
        """The spread rounded to 0.5 pp. A refusal names the field and the rule, never the value:
        only the rounded spread is ever printed."""
        number = (isinstance(value, int | float) and not isinstance(value, bool)
                  and math.isfinite(value))  # fmt: skip
        if not (number and 0 <= value <= 100 and (value / SPREAD_STEP_PP).is_integer()):
            self.add("spread_pp", f"is not a finite multiple of {SPREAD_STEP_PP} pp between 0 and "
                     "100: the spread is recorded rounded, and nothing finer (its value is not "
                     "shown)")  # fmt: skip

    # -- budget

    def _budget(self, r: dict[str, Any]) -> None:
        if r["candidate"] != self.candidate or type(r["candidate"]) is not int:
            self.add("candidate", f"is {r['candidate']!r}, not {self.candidate}")
        self.integer("N", r["N"], MIN_N)
        self.integer("cap", r["cap"], MIN_N + CAP_FLOOR)
        decision = r["decision"]
        if decision not in DECISIONS:
            self.add("decision", f"is {decision!r}, not one of {list(DECISIONS)}")
        elif (decision == "proceed") != (r["state"] == "passed"):
            self.add("state", f"is {r['state']!r} for the decision {decision!r} (proceed is "
                     "passed, stop is failed)")  # fmt: skip
        entry = self.ref("answers", r["answers"])
        named = None if entry is None else entry["path"]
        if entry is not None and (entry.get("root", "runs") != "runs" or not (
                isinstance(named, str) and _N_LINK.fullmatch(named)
                and int(_N_LINK.fullmatch(named).group(1)) == self.candidate)):  # fmt: skip
            self.add("answers", f"is {named!r} under {entry.get('root', 'runs')!r}, not the "
                     f"record study3/decisions/n-c{self.candidate}.json (or a successor "
                     f"n-c{self.candidate}-<k>.json) under the runs directory")  # fmt: skip
        self._supersedes(r)

    # -- closure

    def _closure(self, r: dict[str, Any]) -> None:
        arm = self.kind.rsplit("-", 1)[1]
        if r["candidate"] != self.candidate or type(r["candidate"]) is not int:
            self.add("candidate", f"is {r['candidate']!r}, not {self.candidate}")
        if r["arm"] != arm:
            self.add("arm", f"is {r['arm']!r}, not the kind's {arm}")
        if r["state"] != "failed":
            self.add("state", f"is {r['state']!r}: a closure is a terminal technical outcome, "
                     "failed")  # fmt: skip
        self.real("rate", r["rate"], positive=True)
        seeds = {s for role in SEEDS.values() for s in role.get(self.candidate, ())}
        if type(r["seed"]) is not int or r["seed"] not in seeds:
            self.add("seed", f"is {r['seed']!r}, not a registered seed of candidate "
                     f"{self.candidate}")  # fmt: skip
        run_id = r["run_id"]
        if not (isinstance(run_id, str) and run_id not in ("", ".", "..")
                and PurePosixPath(run_id).name == run_id and "\\" not in run_id):  # fmt: skip
            self.add("run_id", f"is {run_id!r}, not a run id")
        kind = self._line(r["line"])
        reason = r["reason"]
        # a held line, or an end or start line of a mark left unresolved after the cycle, cites
        # the candidate's supersession record; the closure of a run under another code, none
        cited = kind == "held" or (kind in ("end", "start") and reason == UNRESOLVED_AFTER_CYCLE)
        if kind in ("end", "start") and not cited and r["cycle"] is not None:
            self.add("cycle", f"is {r['cycle']!r}: the closure of a run's {kind} line under "
                     "another code cites no cycle (null)")  # fmt: skip
        if cited and r["cycle"] is None:
            self.add("cycle", "is null: the closure of a held line, or of a mark left unresolved "
                     "after the candidate's engineering cycle, cites the candidate's "
                     "supersession record")  # fmt: skip
        cycle = self.ref("cycle", r["cycle"]) if cited and r["cycle"] is not None else None
        if cycle is not None and (cycle.get("root", "runs") != "runs" or not (
                isinstance(cycle["path"], str) and _CYCLE.fullmatch(cycle["path"])
                and _CYCLE.fullmatch(cycle["path"]).group(1) == str(self.candidate))):  # fmt: skip
            self.add("cycle", f"is {cycle['path']!r} under {cycle.get('root', 'runs')!r}, not "
                     f"candidate {self.candidate}'s supersession record "
                     f"qualification/c{self.candidate}/supersede-<n>.json under the runs "
                     "directory")  # fmt: skip
        if kind == "held" and not (isinstance(reason, str) and reason in HOLD_REASONS):
            self.add("reason", f"is {reason!r}, not the id of a reason a hold states "
                     f"({sorted(HOLD_REASONS)})")  # fmt: skip
        elif kind in ("end", "start") and (_end_reason_id(reason) or reason == START_REASON):
            self.add("reason", f"is {reason!r}: {NOT_ACCEPTED}")
        elif kind in ("end", "start") and reason not in (*UNCOUNTED_REASONS,
                                                         UNRESOLVED_AFTER_CYCLE):  # fmt: skip
            self.add("reason", f"is {reason!r}, not the id of a run under another code "
                     f"{UNCOUNTED_REASONS} nor {UNRESOLVED_AFTER_CYCLE}: an end or start line "
                     "closes such a run, or a mark left unresolved after the cycle, "
                     "alone")  # fmt: skip
        elif kind is None and not (isinstance(reason, str) and _REASON_ID.fullmatch(reason)):
            self.add("reason", f"is {reason!r}, not an id")
        if r["decision"] != CLOSURE_DECISIONS[arm]:
            self.add("decision", f"is {r['decision']!r}: a closure of {arm} decides "
                     f"{CLOSURE_DECISIONS[arm]!r}")  # fmt: skip

    def _line(self, value: Any) -> str | None:
        """The cited journal line: its kind (returned, None when it is not one of
        ``LINE_KINDS``), ``launch/<log>.jsonl`` under the runs directory, its number and its own
        sha256."""
        line = self.object("line", value, ("kind", "path", "line", "line_sha256"))
        if line is None:
            return None
        kind = line["kind"] if line["kind"] in LINE_KINDS else None
        if kind is None:
            self.add("line.kind", f"is {line['kind']!r}, not one of {list(LINE_KINDS)}")
        path = line["path"]
        parts = PurePosixPath(path).parts if isinstance(path, str) else ()
        if not (len(parts) == 2 and parts[0] == "launch" and parts[1].endswith(".jsonl")
                and "\\" not in path and parts[1] not in (".", "..")):  # fmt: skip
            self.add("line.path", f"is {path!r}, not a launch log launch/<log>.jsonl under the "
                     "runs directory")  # fmt: skip
        self.integer("line.line", line["line"], 1)
        if not (isinstance(line["line_sha256"], str) and _SHA256.fullmatch(line["line_sha256"])):
            self.add("line.line_sha256", f"is {line['line_sha256']!r}, not a sha256")
        return kind

    # -- cycle

    def _cycle(self, r: dict[str, Any]) -> None:
        """The cycle record's form (``study3_cycle`` checks what it lists against the records it
        rests on)."""
        candidate = self.candidate
        if r["candidate"] != candidate or type(r["candidate"]) is not int:
            self.add("candidate", f"is {r['candidate']!r}, not {candidate}")
        if r["state"] != "passed":
            self.add("state", f"is {r['state']!r}: a cycle record is passed (the "
                     "re-qualification is read from the qualification's records)")  # fmt: skip
        stage = r["stage"] if r["stage"] in MARKED_STAGES else None
        if stage is None:
            self.add("stage", f"is {r['stage']!r}, not one of {list(MARKED_STAGES)}: the stage "
                     "of the marks outside the qualification")  # fmt: skip
        old = r["old_code"]
        if not (isinstance(old, str) and _SHA256.fullmatch(old)):
            self.add("old_code", f"is {old!r}, not a sha256")
        elif isinstance(r["code"], dict) and r["code"].get("gpu_code_sha256") == old:
            self.add("old_code", "is the fix's GPU code (code.gpu_code_sha256): a fix that keeps "
                     "the GPU code is no engineering cycle")  # fmt: skip
        for name in ("diagnosis", "fix"):
            why = one_line(r[name])
            if why is not None:
                self.add(name, why)
        marked = self._marks(r["marks"], stage)
        self._claims(r["stages"], stage, marked)
        if r["fixation"] is not None:
            entry = self.ref("fixation", r["fixation"])
            if entry is not None and entry.get("root") != "repo":
                self.add("fixation", "is not evidence in the checkout (root: repo): the "
                         "validation fixation is a committed record")  # fmt: skip
        self._journal(r["journal"])
        self._affected(r["affected"])
        self._kept(r["kept"])
        if r["bridge"] != bridge_name(candidate):
            self.add("bridge", f"is {r['bridge']!r}, not {bridge_name(candidate)!r}")

    def _marks(self, value: Any, stage: str | None) -> set[str]:
        """The marks of the stage: each run's id, label, step (the stage's), arm, rate, seed and
        its cited end or start line. Returns the arms marked."""
        if not isinstance(value, list) or not value:
            self.add("marks", "is not a non-empty list of the marks of the stage")
            return set()
        arms, keys = set(), []
        for number, item in enumerate(value):
            where = f"marks[{number}]"
            entry = self.object(where, item, ("run_id", "label", "step", "arm", "rate", "seed",
                                              "line"))  # fmt: skip
            if entry is None:
                continue
            self._run(where, entry, CYCLE_STEPS[stage] if stage is not None else None)
            self._cited(f"{where}.line", entry["line"], ("end", "start"))
            arms.add(entry["arm"])
            keys.append(str(entry["run_id"]))
        self._canonical("marks", keys)
        return arms

    def _run(self, where: str, entry: Mapping[str, Any], steps: Sequence[str] | None) -> None:
        """A run as a cycle names it: its id (a plain name), label, step (one of ``steps``),
        arm, rate and seed (a registered seed of the candidate)."""
        run_id = entry["run_id"]
        if not (isinstance(run_id, str) and run_id not in ("", ".", "..")
                and PurePosixPath(run_id).name == run_id and "\\" not in run_id):  # fmt: skip
            self.add(f"{where}.run_id", f"is {run_id!r}, not a run id")
        if not (isinstance(entry["label"], str) and entry["label"]):
            self.add(f"{where}.label", f"is {entry['label']!r}, not a label")
        if steps is not None and entry["step"] not in steps:
            self.add(f"{where}.step", f"is {entry['step']!r}, not one of {list(steps)}")
        if entry["arm"] not in CLAIM_ARMS:
            self.add(f"{where}.arm", f"is {entry['arm']!r}, not one of {list(CLAIM_ARMS)}")
        self.real(f"{where}.rate", entry["rate"], positive=True)
        seeds = {s for role in SEEDS.values() for s in role.get(self.candidate, ())}
        if type(entry["seed"]) is not int or entry["seed"] not in seeds:
            self.add(f"{where}.seed", f"is {entry['seed']!r}, not a registered seed of candidate "
                     f"{self.candidate}")  # fmt: skip

    def _cited(self, where: str, value: Any, kinds: Sequence[str] | None = None) -> None:
        """A cited launch-journal line: ``launch/<log>.jsonl`` under the runs directory, its
        number from 1 and its own sha256, and its ``kind`` (one of ``kinds``) when it has
        one."""
        fields = ("kind", "path", "line", "line_sha256") if kinds else ("path", "line",
                                                                        "line_sha256")  # fmt: skip
        line = self.object(where, value, fields)
        if line is None:
            return
        if kinds and line["kind"] not in kinds:
            self.add(f"{where}.kind", f"is {line['kind']!r}, not one of {list(kinds)}")
        self._log(f"{where}.path", line["path"])
        self.integer(f"{where}.line", line["line"], 1)
        if not (isinstance(line["line_sha256"], str) and _SHA256.fullmatch(line["line_sha256"])):
            self.add(f"{where}.line_sha256", f"is {line['line_sha256']!r}, not a sha256")

    def _log(self, where: str, path: Any) -> None:
        parts = PurePosixPath(path).parts if isinstance(path, str) else ()
        if not (len(parts) == 2 and parts[0] == "launch" and parts[1].endswith(".jsonl")
                and "\\" not in path and parts[1] not in (".", "..")):  # fmt: skip
            self.add(where, f"is {path!r}, not a launch log launch/<log>.jsonl under the runs "
                     "directory")  # fmt: skip

    def _claims(self, value: Any, stage: str | None, marked: set[str]) -> None:
        """A claim per stage: ``reuse`` (no arm) or ``rerun`` with its arms (all three for the
        qualification: all or nothing for the candidate), and a justification of one line;
        every stage up to the marks' claimed, the marks' a rerun of every marked arm."""
        if not isinstance(value, dict):
            self.add("stages", "is not an object of a claim per stage")
            return
        for name, claim in value.items():
            where = f"stages.{name}"
            if name not in CYCLE_STAGES:
                self.add(where, f"is no stage a cycle claims ({list(CYCLE_STAGES)})")
                continue
            entry = self.object(where, claim, ("action", "arms", "justification"))
            if entry is None:
                continue
            why = one_line(entry["justification"])
            if why is not None:
                self.add(f"{where}.justification", why)
            action, arms = entry["action"], entry["arms"]
            if action not in CLAIMS:
                self.add(f"{where}.action", f"is {action!r}, not one of {list(CLAIMS)}")
            elif not (isinstance(arms, list) and all(a in CLAIM_ARMS for a in arms)
                      and arms == [a for a in CLAIM_ARMS if a in arms]):  # fmt: skip
                self.add(f"{where}.arms", f"is {arms!r}, not distinct arms in the order "
                         f"{list(CLAIM_ARMS)}")  # fmt: skip
            elif action == "reuse" and arms:
                self.add(f"{where}.arms", f"is {arms!r}: a reuse reruns no arm")
            elif action == "rerun" and not arms:
                self.add(f"{where}.arms", "is empty: a rerun names the arms it reruns")
            elif name == "qualification" and action == "rerun" and arms != list(CLAIM_ARMS):
                self.add(f"{where}.arms", f"is {arms!r}: the qualification's claim is all or "
                         "nothing for the candidate (a rerun reruns A, F and P)")  # fmt: skip
        if stage is None:
            return
        for name in CYCLE_STAGES[: CYCLE_STAGES.index(stage) + 1]:
            if name not in value:
                self.add("stages", f"has no claim for the {name}, a stage started at or before "
                         f"the marks' {stage}")  # fmt: skip
        claim = value.get(stage) if isinstance(value.get(stage), dict) else {}
        arms = claim.get("arms") if isinstance(claim.get("arms"), list) else []
        left = sorted(marked - set(arms)) if claim.get("action") == "rerun" else sorted(marked)
        if stage in value and left:
            self.add(f"stages.{stage}", f"does not rerun every marked arm ({left} left): the "
                     "stage of the marks is a rerun of every marked arm")  # fmt: skip

    def _journal(self, value: Any) -> None:
        """The pinned prefix of each launch log: its path, its whole lines and their sha256."""
        if not isinstance(value, list) or not value:
            self.add("journal", "is not a non-empty list of the launch logs' pinned prefixes")
            return
        keys = []
        for number, item in enumerate(value):
            where = f"journal[{number}]"
            entry = self.object(where, item, ("path", "lines", "sha256"))
            if entry is None:
                continue
            self._log(f"{where}.path", entry["path"])
            self.integer(f"{where}.lines", entry["lines"], 0)
            if not (isinstance(entry["sha256"], str) and _SHA256.fullmatch(entry["sha256"])):
                self.add(f"{where}.sha256", f"is {entry['sha256']!r}, not a sha256")
            keys.append(str(entry["path"]))
        self._canonical("journal", keys)

    def _affected(self, value: Any) -> None:
        """The affected runs (each with its records and its start and end lines), the
        qualification's part records, launcher runs and the case restart records counted before
        the cycle, and the stale write-once records."""
        entry = self.object("affected", value, ("runs", "qualification", "records"))
        if entry is None:
            return
        runs = entry["runs"]
        if not isinstance(runs, list):
            self.add("affected.runs", "is not a list")
        else:
            keys = []
            for number, item in enumerate(runs):
                where = f"affected.runs[{number}]"
                run = self.object(where, item, ("run_id", "label", "step", "arm", "rate", "seed",
                                                "gpu_code", "records", "start", "end"))  # fmt: skip
                if run is None:
                    continue
                self._run(where, run, MARKED_STAGES)
                gpu = run["gpu_code"]
                if gpu is not None and not (isinstance(gpu, str) and _SHA256.fullmatch(gpu)):
                    self.add(f"{where}.gpu_code", f"is {gpu!r}, not a sha256 or null")
                self._files(f"{where}.records", run["records"], absent=True, empty=False)
                self._cited(f"{where}.start", run["start"])
                if run["end"] is not None:
                    self._cited(f"{where}.end", run["end"])
                keys.append(str(run["run_id"]))
            self._canonical("affected.runs", keys)
        qualification = self.object("affected.qualification", entry["qualification"],
                                    ("records", "launchers", "restarts"))  # fmt: skip
        if qualification is not None:
            self._files("affected.qualification.records", qualification["records"])
            self._files("affected.qualification.restarts", qualification["restarts"])
            launchers = qualification["launchers"]
            if not isinstance(launchers, list):
                self.add("affected.qualification.launchers", "is not a list")
            else:
                keys = []
                for number, item in enumerate(launchers):
                    where = f"affected.qualification.launchers[{number}]"
                    launcher = self.object(where, item, ("run_id", "meta"))
                    if launcher is None:
                        continue
                    if not (isinstance(launcher["run_id"], str) and launcher["run_id"]):
                        self.add(f"{where}.run_id", f"is {launcher['run_id']!r}, not a run id")
                    self._file(f"{where}.meta", launcher["meta"], absent=True)
                    keys.append(str(launcher["run_id"]))
                self._canonical("affected.qualification.launchers", keys)
        self._files("affected.records", entry["records"])

    def _kept(self, value: Any) -> None:
        """What keeps counting: runs with their evidence and why, and per reused stage the
        number of its counting runs."""
        if not isinstance(value, list):
            self.add("kept", "is not a list")
            return
        keys = []
        for number, item in enumerate(value):
            where = f"kept[{number}]"
            if isinstance(item, dict) and item.get("why") == "reused_stage":
                entry = self.object(where, item, ("why", "stage", "runs"))
                if entry is None:
                    continue
                if entry["stage"] not in CYCLE_STAGES:
                    self.add(f"{where}.stage", f"is {entry['stage']!r}, not a stage")
                self.integer(f"{where}.runs", entry["runs"], 1)
                keys.append(("", "", str(entry["stage"])))
                continue
            entry = self.object(where, item, ("run_id", "why", "evidence"))
            if entry is None:
                continue
            if entry["why"] not in KEPT_WHY:
                self.add(f"{where}.why", f"is {entry['why']!r}, not one of {list(KEPT_WHY)}")
            if not (isinstance(entry["run_id"], str) and entry["run_id"]):
                self.add(f"{where}.run_id", f"is {entry['run_id']!r}, not a run id")
            evidence = self._file(f"{where}.evidence", entry["evidence"], absent=True)
            keys.append((str(entry["run_id"]), str((evidence or {}).get("path")), ""))
        self._canonical("kept", keys)

    def _files(self, where: str, value: Any, *, absent: bool = False,
               empty: bool = True) -> None:  # fmt: skip
        """A list of evidence (or, with ``absent``, absence markers), by path."""
        if not isinstance(value, list) or (not empty and not value):
            self.add(where, "is not a " + ("" if empty else "non-empty ") + "list of evidence")
            return
        keys = []
        for number, item in enumerate(value):
            entry = self._file(f"{where}[{number}]", item, absent=absent)
            keys.append(str((entry or {}).get("path")))
        self._canonical(where, keys)

    def _file(self, where: str, value: Any, *, absent: bool = False) -> dict[str, Any] | None:
        """Evidence ``{path, sha256}`` of a file that is there or, with ``absent``, an absence
        marker ``{path, absent: true}`` of one that is not: checked to be still absent."""
        if not (absent and isinstance(value, dict) and "absent" in value):
            return self.ref(where, value)
        entry = self.object(where, value, ("path", "absent"), ("root",))
        if entry is None:
            return None
        ok = self.text(f"{where}.path", entry["path"])
        if entry["absent"] is not True:
            self.add(f"{where}.absent", f"is {entry['absent']!r}, not true")
        root = entry.get("root", "runs")
        if root not in ("runs", "repo"):
            self.add(f"{where}.root", f"is {root!r}, not runs or repo")
            ok = False
        if ok:
            self.absent.append(_Ref(where, entry["path"], "", root))
        return entry

    def _canonical(self, where: str, keys: list[Any]) -> None:
        """A list in canonical order (by run id, then path) that names nothing twice."""
        twice = [key for number, key in enumerate(keys) if key in keys[:number]]
        if twice:
            self.add(where, f"names {twice[0]!r} twice: a cycle's list names nothing twice")
        elif keys != sorted(keys):
            self.add(where, "is not in canonical order (by run id, then path)")


def coverage_threshold(tests: int, runs: int, p0: float) -> int:
    """c = BinomQuantile(1 − 0.05/J; M, p₀): the largest non-coverage count that a correctly
    calibrated procedure passes across J tests of M simulated studies each."""
    from scipy.stats import binom

    return int(binom.ppf(1 - FAMILY_ALPHA / tests, runs, p0))


# ------------------------------------------------------------------ evidence and cross-checks


def _evidence(refs: Sequence[_Ref], root: Path, repo: Path) -> list[str]:
    """The problems of the evidence files as they are now: sealed, missing or changed."""
    problems = []
    for ref in refs:
        named = Path(ref.path)
        base = repo if ref.root == "repo" else root
        path = named if named.is_absolute() else base / named
        if _under_sealed(path, base, named):
            why = sealed_why(path, base, "is under a responses or observations folder")
            problems.append(f"{ref.where}: {path} {why}: never evidence, never opened")
        elif not path.is_file():
            problems.append(f"{ref.where}: the evidence file {path} is missing")
        else:
            try:
                now = _digest(path)
            except OSError as exc:
                problems.append(f"{ref.where}: the evidence file {path} is unreadable "
                                f"({type(exc).__name__})")  # fmt: skip
                continue
            if now != ref.sha256:
                problems.append(f"{ref.where}: the evidence file {path} has sha256 {now} now, "
                                f"not the recorded {ref.sha256}")  # fmt: skip
    return problems


def _still_absent(refs: Sequence[_Ref], root: Path, repo: Path) -> list[str]:
    """The problems of absence markers as the files are now: a file that has appeared since, or
    a marker under a sealed folder (never opened)."""
    problems = []
    for ref in refs:
        named = Path(ref.path)
        base = repo if ref.root == "repo" else root
        path = named if named.is_absolute() else base / named
        if _under_sealed(path, base, named):
            why = sealed_why(path, base, "is under a responses or observations folder")
            problems.append(f"{ref.where}: {path} {why}: never evidence, never opened")
        elif path.exists() or path.is_symlink():
            problems.append(f"{ref.where}: {path} is there now, and the record marks it absent")
    return problems


def _against_journal(root: Path, record: Mapping[str, Any]) -> list[str]:
    """A cycle record against the launch journal as it is now: each pinned prefix (its whole
    lines' sha256: a rewritten journal is a problem), and each cited line (a mark's, an affected
    run's start and end) by its bytes, of the event and run it names. A log that cannot be read,
    or that resolves under a sealed folder (never opened), fails closed."""
    logs, unreadable = journal_lines(root)
    problems = [f"journal: {problem}" for problem in unreadable]
    by_path = {log.path: log for log in logs}
    for number, entry in enumerate(record["journal"]):
        where, path = f"journal[{number}]", root / entry["path"]
        if entry["path"] not in by_path:
            problems.append(f"{where}: the launch log {path} whose prefix the cycle pinned is gone "
                            "or cannot be read")  # fmt: skip
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            problems.append(f"{where}: the launch log {path} cannot be read ({type(exc).__name__})")
            continue
        end = pinned_end(log_lines(data), entry["lines"])
        now = None if end is None else hashlib.sha256(data[:end]).hexdigest()
        if now != entry["sha256"]:
            problems.append(f"{where}: the pinned prefix of {entry['lines']} lines of {path} has "
                            f"sha256 {now} now, not {entry['sha256']}: the journal was "
                            "rewritten")  # fmt: skip
    cited = [(f"marks[{n}].line", m["line"], m["line"]["kind"], m["run_id"])
             for n, m in enumerate(record["marks"])]  # fmt: skip
    for n, run in enumerate(record["affected"]["runs"]):
        cited.append((f"affected.runs[{n}].start", run["start"], "start", run["run_id"]))
        if run["end"] is not None:
            cited.append((f"affected.runs[{n}].end", run["end"], "end", run["run_id"]))
    for where, line, event, run_id in cited:
        path = root / line["path"]
        log = by_path.get(line["path"])
        if log is None:
            problems.append(f"{where}: the launch log {path} cannot be read")
        elif line["line"] > len(log.lines):
            problems.append(f"{where}: the launch log {path} has no line {line['line']}")
        elif log.lines[line["line"] - 1].line_sha256 != line["line_sha256"]:
            problems.append(f"{where}: line {line['line']} of {path} has line_sha256 "
                            f"{log.lines[line['line'] - 1].line_sha256} now, not the recorded "
                            f"{line['line_sha256']}")  # fmt: skip
        else:
            found = log.lines[line["line"] - 1].record or {}
            if found.get("event") != event or found.get("run_id") != run_id:
                problems.append(f"{where}: line {line['line']} of {path} is not the {event} line "
                                f"of {run_id}")  # fmt: skip
    return problems


def _against_simulation(root: Path, record: Mapping[str, Any], candidate: int,
                        repo: Path) -> list[str]:  # fmt: skip
    """An ``n-c<N>`` record against the simulation record it follows."""
    simulation, problems = read_decision(root, "simulation", repo=repo)
    if simulation is None:
        return ["the simulation record it follows is " + ("missing" if not problems else
                f"not valid ({problems[0]})")]  # fmt: skip
    found = []
    if simulation["state"] != "passed":
        found.append("the simulation record did not pass")
    if simulation["level"] is None or not math.isclose(simulation["level"], record["level"],
                                                       rel_tol=1e-12):  # fmt: skip
        found.append(f"L is {record['level']}, not the simulation record's {simulation['level']}")
    power = simulation["gates"]["power_af"]
    if power["power_short"] and record["N"] < power["n_min"]:
        found.append(f"N is {record['N']}, below n_min {power['n_min']}, the smallest N at which "
                     f"the simulated power reaches {power['requirement']}")  # fmt: skip
    blocks = simulation["validation_blocks"]
    seeds = record["cohort"]["seeds"]
    if blocks in SUPPORTED_AT:
        first = SEEDS["validation"][candidate][0]
        late = [s for s in seeds if s >= first + blocks]
        if late:
            found.append(f"cohort seeds {late[:3]} lie beyond the {blocks} validation blocks the "
                         "simulation chose")  # fmt: skip
        if len(seeds) < SUPPORTED_AT[blocks]:
            found.append(f"the cohort has {len(seeds)} blocks, below the {SUPPORTED_AT[blocks]} "
                         f"supported that the {blocks}-block validation needs")  # fmt: skip
    return found


def _against_line(root: Path, record: Mapping[str, Any], candidate: int) -> list[str]:
    """A ``closure-c<N>-<arm>`` record against the files as they are now (``journal_lines``,
    the one reading of a cited line): the cited line's bytes and what it is (a held line, the
    run's last end, or its start when it has no end; of the record's run and seed, stating the
    record's reason), nothing in the journal after it that the readers would read as the run's
    next attempt, and the supersession record it cites: for a held line, its ``held`` naming
    the run; for a mark left unresolved after the cycle, the candidate's. A launch log that
    cannot be read, or an unresolved torn line before the last of a log, fails
    closed: what it holds cannot be shown. The cycle's sha256 is checked with the other
    evidence."""
    run_id, cited = record["run_id"], record["line"]
    path = root / cited["path"]
    if _under_sealed(path, root, Path(cited["path"])):
        why = sealed_why(path, root, "is under a responses or observations folder")
        return [f"line: {path} {why}: never opened"]
    logs, unreadable = journal_lines(root)
    found = [f"line: {problem}: the journal cannot show what follows the run"
             for problem in unreadable]  # fmt: skip
    found += [f"line: {log.path} line {number}: {TORN_LINE}: the journal cannot show what "
              "follows the run" for log in logs for number in log.corrupt]  # fmt: skip
    log = next((log for log in logs if log.path == cited["path"]), None)
    where = f"line: line {cited['line']} of {path}"
    if log is None:
        found.append(f"line: the launch log {path} cannot be read")
    elif cited["line"] > len(log.lines):
        found.append(f"line: the launch log {path} has no line {cited['line']}")
    else:
        line = log.lines[cited["line"] - 1]
        if line.line_sha256 != cited["line_sha256"]:
            found.append(f"{where} has line_sha256 {line.line_sha256} now, not the recorded "
                         f"{cited['line_sha256']}")  # fmt: skip
        else:
            check = {"held": _held_line, "end": _end_line, "start": _start_line}[cited["kind"]]
            found += check(root, line.record or {}, record, where, logs)
    if record["cycle"] is None:
        return found
    cycle = root / record["cycle"]["path"]
    if not _under_sealed(cycle, root) and cycle.is_file():
        try:
            data = json.loads(cycle.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        entries = data.get("held") if isinstance(data, dict) else None
        if cited["kind"] == "held" and (not isinstance(entries, list) or not any(
                isinstance(e, Mapping) and e.get("run_id") == run_id
                for e in entries)):  # fmt: skip
            found.append(f"run_id: the cycle record {cycle} holds no run {run_id}")
        elif isinstance(data, dict) and data.get("candidate", candidate) != candidate:
            found.append(f"cycle: the record {cycle} is candidate {data.get('candidate')!r}'s, "
                         f"not {candidate}'s")  # fmt: skip
    return found


def _of_run(event: Mapping[str, Any], record: Mapping[str, Any]) -> list[str]:
    """A cited line names the record's run, and its label the record's seed."""
    found = []
    if event.get("run_id") != record["run_id"]:
        found.append(f"run_id: is {record['run_id']!r}, not the run its line names "
                     f"({event.get('run_id')!r})")  # fmt: skip
    label = event.get("run")
    if not (isinstance(label, str) and label.endswith(f"/s{record['seed']}")):
        found.append(f"seed: {record['seed']} is not the seed of its line's run")
    return found


def _held_line(root: Path, event: Mapping[str, Any], record: Mapping[str, Any], where: str,
               logs: Sequence[Any]) -> list[str]:  # fmt: skip
    """A cited held line: the ``held`` event of the record's run, its label the record's seed's,
    stating the record's reason."""
    if event.get("event") != "held":
        return [f"{where} is not a held line of the launch journal"]
    found = _of_run(event, record)
    stated = hold_reasons(event.get("reason"))
    if record["reason"] not in stated:
        found.append(f"reason: {record['reason']} is not one of the reasons the held line "
                     f"states ({stated})")  # fmt: skip
    return found


def _end_line(root: Path, event: Mapping[str, Any], record: Mapping[str, Any], where: str,
              logs: Sequence[Any]) -> list[str]:  # fmt: skip
    """A cited end line: the ``end`` of the record's run, its last in the journal, whatever its
    outcome (the record's reason is the run's uncounted reason, not the outcome's: the schema
    checks it), of a run nothing followed (``_followed``)."""
    if event.get("event") != "end":
        return [f"{where} is not an end line of the launch journal"]
    found = _of_run(event, record)
    last = end_lines(root, logs).get(record["run_id"])
    if last is None or (last["path"], last["line"]) != (
        record["line"]["path"],
        record["line"]["line"],
    ):
        found.append(f"line: it is not the run's last end line in the launch journal "
                     f"({None if last is None else (last['path'], last['line'])})")  # fmt: skip
    return found + _followed(event, record, logs)


def _start_line(root: Path, event: Mapping[str, Any], record: Mapping[str, Any], where: str,
                logs: Sequence[Any]) -> list[str]:  # fmt: skip
    """A cited start line: the ``start`` of the record's run, which has no end line in the
    journal, nothing following it (``_followed``)."""
    if event.get("event") != "start":
        return [f"{where} is not a start line of the launch journal"]
    found = _of_run(event, record)
    if record["run_id"] in end_lines(root, logs):
        found.append(f"line: {record['run_id']} has an end line in the launch journal: its "
                     "closure cites its last end line")  # fmt: skip
    return found + _followed(event, record, logs)


def _followed(event: Mapping[str, Any], record: Mapping[str, Any],
              logs: Sequence[Any]) -> list[str]:  # fmt: skip
    """What in the journal makes a run no technical mark the readers list: a later attempt of
    its label, a recovery round of its label that follows the run (``study3_records.
    round_follows``, the records reader's rule: a round with ``after`` follows the attempts it
    names; one without, the attempts started before its time stamp, a start in its second
    following it; one that cannot be placed counts as following), a ``superseded`` or a ``held``
    line naming the run."""
    run_id, label, attempt = record["run_id"], event.get("run"), event.get("attempt")
    lines = [line.record for log in logs for line in log.lines if line.record is not None]
    found = []
    later = sorted({r["run_id"] if isinstance(r.get("run_id"), str) else "(no run id)"
                    for r in lines if r.get("event") == "start"
                    and r.get("run") == label and type(r.get("attempt")) is int
                    and type(attempt) is int and r["attempt"] > attempt})  # fmt: skip
    if later:
        found.append(f"line: a later attempt of its label {label} was started ({later}): the "
                     "next attempt decides")  # fmt: skip
    for name in ("superseded", "held"):
        if any(r.get("event") == name and r.get("run_id") == run_id for r in lines):
            found.append(f"run_id: a {name} line of the launch journal names {run_id}")
    start = (
        event
        if event.get("event") == "start"
        else next(
            (r for r in lines if r.get("event") == "start" and r.get("run_id") == run_id), event
        )
    )
    started = _time(start.get("time"))
    starts = [(r.get("attempt"), _time(r.get("time"))) for r in lines
              if r.get("event") == "start" and r.get("run") == label]  # fmt: skip
    if any(r.get("event") == "recovery" and label in (r.get("runs") or [])
           and round_follows(r, run_id, attempt, started, starts) is not False
           for r in lines):  # fmt: skip
        found.append(f"line: a recovery round of its label {label} followed it: the next "
                     "attempt decides")  # fmt: skip
    return found


def end_lines(runs_root: str | Path, logs: Sequence[Any] | None = None) -> dict[str, dict]:
    """Each run's last ``end`` line in the launch journal (by its time stamp, then its log and
    its number), as a closure record cites it: ``path`` (under RUNS_DIR), ``line``,
    ``line_sha256`` (the line's own bytes), and its ``outcome``, ``attempt`` and ``run`` (the
    label); never the launcher's free text. ``logs``: ``journal_lines``' logs, when read
    already."""
    found: dict[str, tuple[tuple, dict[str, Any]]] = {}
    for log in journal_lines(runs_root)[0] if logs is None else logs:
        for line in log.lines:
            record = line.record or {}
            run_id = record.get("run_id")
            if record.get("event") != "end" or not isinstance(run_id, str):
                continue
            when = _time(record.get("time"))
            key = (when is not None, when.isoformat() if when else "", log.path, line.line)
            if run_id not in found or key > found[run_id][0]:
                found[run_id] = (key, {
                    "path": log.path, "line": line.line, "line_sha256": line.line_sha256,
                    "outcome": record.get("outcome"), "attempt": record.get("attempt"),
                    "run": record.get("run")})  # fmt: skip
    return {run_id: value for run_id, (_, value) in found.items()}


def start_lines(runs_root: str | Path, logs: Sequence[Any] | None = None) -> dict[str, dict]:
    """Each run's start line in the launch journal (its first), as a closure record cites it
    when the launcher never ended the run: ``path``, ``line``, ``line_sha256``, ``attempt`` and
    ``run`` (the label)."""
    found: dict[str, dict[str, Any]] = {}
    for log in journal_lines(runs_root)[0] if logs is None else logs:
        for line in log.lines:
            record = line.record or {}
            run_id = record.get("run_id")
            if record.get("event") == "start" and isinstance(run_id, str) and run_id not in found:
                found[run_id] = {
                    "path": log.path,
                    "line": line.line,
                    "line_sha256": line.line_sha256,
                    "attempt": record.get("attempt"),
                    "run": record.get("run"),
                }
    return found


def _against_n(root: Path, kind: str, record: Mapping[str, Any], candidate: int,
               repo: Path) -> list[str]:  # fmt: skip
    """A ``budget-c<N>`` record (or a successor) against the n record it answers: its N and its
    cap; a successor answers an n successor (module docstring, "Successors")."""
    named = PurePosixPath(record["answers"]["path"]).name[: -len(".json")]
    n, problems = read_decision(root, named, repo=repo)
    if n is None:
        return [f"the {named} record it answers is "
                + ("missing" if not problems else f"not valid ({problems[0]})")]  # fmt: skip
    found = [f"{name} {record[name]} is not the {named} record's {n[name]}"
             for name in ("N", "cap") if record[name] != n[name]]  # fmt: skip
    if link_of(kind)[1] > 1 and link_of(named)[1] == 1:
        found.append(f"answers: {named} is the first n record: a budget successor answers the n "
                     "record's successor")  # fmt: skip
    return found


def _against_predecessor(root: Path, kind: str, record: Mapping[str, Any], candidate: int,
                         repo: Path) -> list[str]:  # fmt: skip
    """A successor against the record before it and the candidate's cycle (module docstring,
    "Successors", "The confirmation boundary")."""
    from .study3_cycle import read_cycle  # it reads this module's records: here only

    base, k = link_of(kind)
    before = base if k == 2 else f"{base}-{k - 1}"
    predecessor, problems = read_decision(root, before, repo=repo)
    if predecessor is None:
        return [f"supersedes: the record before it, {before}, is "
                + ("missing" if not problems else f"not valid ({problems[0]})")]  # fmt: skip
    found = read_cycle(root, candidate, repo=repo)
    if found["state"] != "valid" or found["record"] is None:
        why = "; ".join(found["problems"][:2]) or "none is there"
        return [f"supersedes.cycle: candidate {candidate}'s cycle is not accepted ({why}): a "
                "successor follows a stale record through an accepted cycle"]  # fmt: skip
    cycle, out = found["record"], []
    old, fix = cycle["old_code"], cycle["code"]["gpu_code_sha256"]
    own = predecessor["code"]["gpu_code_sha256"]
    where = "/".join((*DIRECTORY, f"{before}.json"))
    listed = {(entry.get("path"), entry.get("sha256")) for entry in cycle["affected"]["records"]}
    pinned = (where, _digest(decision_path(root, before))) in listed
    if own != old or not pinned:
        out.append(f"supersedes: {before} is not stale through candidate {candidate}'s cycle (its "
                   f"code {own}, the cycle's old code {old}; listed among the records the cycle "
                   f"makes stale: {pinned}): a successor follows a stale record alone")  # fmt: skip
    if record["code"]["gpu_code_sha256"] != fix:
        out.append(f"code: its GPU code {record['code']['gpu_code_sha256']} is not the cycle's "
                   f"fix {fix}: a successor carries the fix")  # fmt: skip
    if confirmation_started(root, candidate):
        changed = sorted(key for key in {*record, *predecessor} if key not in SUCCESSOR_FREE
                         and record.get(key, _ABSENT) != predecessor.get(key, _ABSENT))  # fmt: skip
        if changed:
            out.append(f"{', '.join(changed)}: after the confirmation has started, N, the cap and "
                       "the cohort never change without an explicit prospective amendment (a "
                       "successor differs from its predecessor only in its code, written_at, "
                       "supersedes and the reference to the n record in force)")  # fmt: skip
    return out


_ABSENT = object()


# ------------------------------------------------------------------ small helpers


def _runs_of(record_path: Path) -> Path:
    """The runs directory of a record at ``RUNS_DIR/study3/decisions/<kind>.json``."""
    parents = record_path.parents
    return parents[2] if len(parents) > 2 else Path(record_path.anchor or ".")


def _sealed_problem(path: Path) -> str:
    """A record path the sealed-path rule refuses: under a sealed folder, or a link loop."""
    return f"{path}: {sealed_why(path)}: never opened"


def _level(value: Any) -> float | None:
    """The ladder's level that ``value`` is."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return next((lv for lv in LEVELS if math.isclose(value, lv, rel_tol=1e-12)), None)
    return None


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()
