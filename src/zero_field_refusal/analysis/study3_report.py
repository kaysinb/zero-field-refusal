"""Study 3's final report: the facts of every terminal outcome and their tables, label-free.

docs/study3-preregistration.md: "Seeds and order" (order step 13, the final report; both
candidates failing scientifically end the study with a report of the specific reasons), "Failure
routes" (a technical dead end and scientific infeasibility reported separately), "Voiding and
supersession" (a superseded verdict kept and reported; a run under another commit kept and
reported; an embargo breach disclosed with the stage it affected), "Candidates and rate search"
(the spacing, a reported quantity; the curvature sensitivity at the realized widths), "Numerical
qualification" (e_eng and the path maxima; the fallback per case), "Simulation before validation"
(the gates, the coverage per level, L and the size; per test the non-coverage with its 95 %
interval and the detection power at the actual J), "Utility" (every endpoint's utility, every
exclusion and the worst-case values over the started blocks), "Validation, confirmation and
stopping" (P–F's power; the unsupported and excluded started blocks), "Questions and estimands",
"Inference", "Judge and audit", "Disclosure", and amendments 4 (the dry run in two parts and every
re-run, in the reports the engineering record names), 5 (each stage's incidents with their
prices, and any stop) and 6 (every replay, in the same reports).

The facts are numbers, ids, run ids, paths, hashes and the registered words; the prose of the
report is the researcher's. Nothing here opens a response, a label or a judge output, and nothing
imports a label reader: the report's command (``scripts/study3_analysis.py report``) passes in what
the labeling ledger gives (the runs with a technical labeling problem, the judge's diagnosis), and
the analysis, validation's harm and the audit's estimate as their records hold them.

- ``report_facts``: the facts (``SECTIONS``), for the standard variant at ``done`` or a terminal
  outcome (``OUTCOMES``), or the ``judge_technical`` variant (``VARIANTS``).
- ``tables``: the facts as markdown tables, harm proportions shown in pp, no prose.
- ``read_billing`` and ``read_embargo``: the researcher's billing and embargo files, closed
  schemas.
- ``journal_incidents`` and ``qualification_incidents``: the incidents the records hold, each of
  which needs a billed price.
- ``simulation_results``: the simulation's per-test rows (the one adapter of the simulation's
  results, below).

**The variants.**
- *standard*: at ``done``, with the judge-based analysis, validation's harm and the audit's
  estimate, all three required; at a terminal outcome (``study3_progress.TERMINAL``), with what
  the study reached.
- *judge_technical*: the registered technical feasibility report for the harm comparison, written
  when the judge's technical mark is unresolved (a diagnosis record whose cause is ``judge``) and
  labeling stays stopped: no Δ, no interval and no audit (``analysis``, ``validation_harm`` and
  ``audit`` are ``"not reached: the judge's technical mark"``); the freeze with its cohorts when
  it exists; every diagnosis whose cause is ``judge`` (``judge_mark``), each also a technical dead
  end; the runs with a technical labeling problem; every label-free section as in the standard
  variant. An ``infrastructure`` diagnosis changes nothing
  here: its lost rounds are released, and labeling resumes. The variants do not mix.

**The sections** (``SECTIONS``): each present in every outcome; one the study did not reach is
``"not reached"``; ``judge_mark`` is null in the standard variant; ``labeling``, ``not_counted``
and ``embargo`` are lists, ``[]`` when empty.
- ``outcome``: the variant, stage, candidate, outcome, reason, reasons, readings, active arms and
  comparisons, route and closures, as the state gives them; the dead ends apart, ``technical``
  and ``scientific``, each with its reason id.
- ``dry_run``: the engineering record's checks, each its evidence's path and sha256 (the dry run's
  aggregate report among them, which names each part's report, every re-run and every replay),
  and the registration at HEAD with the numbers of its amendments.
- ``engineering``: e_eng, the repeats, the path maxima and the margins of the engineering
  verdict's summary, with the verdict's path and sha256.
- ``simulation``: the simulation record's gates (the pass rates, the powers, the detection powers
  at J), coverage per level (its X and c), verification, L and size; per test its X, c, pass and
  non-coverage with its 95 % interval, from the simulation's results.
- ``qualification``: per candidate and rate its state, fallback cases and warnings; the
  supersession records (path, sha256) with the verdicts they superseded, and the case restarts.
- ``not_counted``: the state's runs that do not count for the commit they ran.
- ``validation``: the gates and supported counts, the utility bounds at α = 0.05 per bound, and
  the spacing: the registered ln 1.25 and, per arm and rate, the SD of ln K over the blocks, with
  the ratio ln 1.25 / SD.
- ``n``: N, the cap, the spread in pp and P–F's power (the simulation's ``power_pf``); the budget
  decision, when one was written.
- ``confirmation``: the started blocks, counters, stop and readings; per comparison every
  started block outside its analysis cohort with its state and the arms' reasons (unsupported,
  or supported beyond the cohort).
- ``utility``: each validation and confirmation endpoint's MMLU and factual loss and state, the
  exclusions, and the registered worst case: per arm, the largest losses among the ended
  endpoints of the started confirmation blocks, with the run.
- ``analysis``: per comparison the three computations (the missing-labels envelope with both its
  sources), the readings and the reading, the wording's row and numbers, n, status and the
  missing labels; the worst cases apart (``worst_case_unknown_blocks``, ``observed_extremes``);
  the utility claim with its gates; A − P (n = 0 shown as such); K*_h's matched analysis with
  the frozen K*_h, its blocks and those left out; the spread of K; the curvature sensitivity with
  the simulation fixation's κ and its source.
- ``validation_harm``: per comparison the three computations and the reading (descriptive).
- ``audit``: B_E per estimable comparison (missing at random and the worst case), per arm the
  sensitivity and the false-positive rate (both ways), the missing references and the
  undetermined flags; the non-estimable comparisons. Per comparison, whether B_E's MAR estimate
  lies outside its worst-case bound, with why it can (``BIAS_NOTE``): the MAR estimate averages
  over the resolved draws alone, while each missing draw's term in the bound is fixed in sign by
  its known judge label.
- ``labeling``: the run ids with a technical labeling problem.
- ``freeze``: the freeze record's path and sha256 and its cohorts as frozen (seeds and run ids per
  comparison, K*_h, A − P).
- ``judge_mark``: the judge's diagnoses, each its path, sha256, batch, cause, diagnosis and
  evidence.
- ``money``: actual spending by stage, one row per incident (its id, stage, incident time, cause,
  billed price and evidence list), and the stages an internal spending policy stopped, by name
  alone.
- ``embargo``: the breaches the researcher's embargo file lists (stage, what).

**Money** ("Budget and timing"; amendment 5, "Infrastructure incidents": each incident is counted
once, with its price). The report states actual spending by stage and each incident with its
billed price, from the researcher's billing file. One incident can leave several records: a Pod
stop can leave a launch-journal recovery round, a case restart and a part record made with
``--infrastructure``. The records' incidents are keyed by what they are: a launch-journal
incident (a study-3 ``recovery`` line carrying ``INCIDENT_KEYS``) by its log and line, a
qualification record made with ``--infrastructure`` (a part record carrying ``infrastructure``,
or a case restart) by its path under RUNS_DIR. Each billed incident names its records in its
evidence (``{"journal": log, "line": n}``, ``{"qualification": path}``), and may add a provider
reference (``{"pod": ref}``), which names no record. The report is refused unless every record
is named by exactly one billed incident, every record an incident names is one of them, the ids
are distinct, every incident names some evidence, and each incident's stage is its records'
where shown (a journal line's: the step of its log's study-3 starts; a qualification record's:
its stage folder, ``numerical`` covering the repair). Every stage the route reached (the stage of
a route event) or an incident is billed at has its spending entry, ``0`` included; otherwise the
report is refused. No other money figure is read or written.

**The simulation's results** are read through one adapter, ``simulation_results``: the per-test
rows of ``RUNS_DIR/study3/simulation/results.json`` (``SIMULATION_RESULTS``), each with the row
fields of the simulation's results. The simulation's own reader (its fixation and results
checks, and the decisions recomputed from the rows) takes its place when it is in the package.

**Rulings** where the registration leaves a detail open:
- *The prose is the researcher's.* The facts and tables carry no sentence of interpretation.
- *Harm in pp.* Every harm difference, interval and estimate is a proportion in the records and
  in the facts; ``tables`` alone multiplies it by 100.
- *The dead ends.* A route's move to candidate 2 and a P–F closure for a scientific failure or a
  failed validation gate are scientific (``scientific_failure``, ``validation_gate``); a technical
  report, a P–F closure after the engineering cycle (``unresolved_after_cycle``) or by a closure
  record (its reason id) are technical, and so is the judge's technical mark in the
  ``judge_technical`` variant (``judge_mark``, at the labeling, one per diagnosis with its
  batch).
- *The registered utility worst case* is over the started confirmation blocks, per arm: the
  largest MMLU and factual losses among the endpoints that ended (usable, unusable or failed
  scientifically), with the run.
- *One incident, one price.* An incident is matched to its records by their keys alone; its
  cause is the researcher's line, shown and never matched, so that two recovery rounds of one
  stop whose cause texts differ are still one incident.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from .study3_blocks import PAIRS, SUPPORTED, VALIDATION_ALPHA, block_state, utility_gate
from .study3_progress import (
    COMPARISONS,
    ENDED,
    TERMINAL,
)
from .study3_records import (
    EXPERIMENT,
    INCIDENT_KEYS,
    RATE_TOLERANCE,
    SEEDS,
    SIMULATION,
    journal_lines,
    under_sealed,
)
from .study3_search import ARM_ORDER

OUTCOMES = ("done", *TERMINAL)
VARIANTS = ("standard", "judge_technical")
SECTIONS = (
    "outcome",
    "dry_run",
    "engineering",
    "simulation",
    "qualification",
    "not_counted",
    "validation",
    "n",
    "confirmation",
    "utility",
    "analysis",
    "validation_harm",
    "audit",
    "labeling",
    "freeze",
    "judge_mark",
    "money",
    "embargo",
)
MONEY_STAGES = (
    "dry_run",
    "engineering",
    "simulation",
    "search",
    "numerical",
    "repair",
    "validation",
    "confirmation",
    "labeling",
    "audit",
)
CURRENCY = "USD"
NOT_REACHED = "not reached"
JUDGE_NOT_REACHED = "not reached: the judge's technical mark"
SPACING = math.log(1.25)  # "Candidates and rate search": the spacing between targets
SIMULATION_RESULTS = "/".join((EXPERIMENT, SIMULATION, "results.json"))
# The simulation's per-test row (its results' ``tests``), as the simulation writes it
ROW_FIELDS = (
    "test",
    "level",
    "due",
    "p0",
    "studies",
    "draws",
    "reason",
    "X",
    "c",
    "passed",
    "rate",
    "interval",
)
BILLING_FIELDS = ("currency", "stages", "incidents", "policy_stops")
STAGE_FIELDS = ("stage", "spent")
INCIDENT_FIELDS = ("id", "stage", "incident_time", "cause", "billed", "evidence")
# An incident's evidence item, by its kind: a launch-journal line, a qualification record (both
# records of the incident) or a provider reference (no record)
EVIDENCE_FIELDS = {"journal": ("journal", "line"), "qualification": ("qualification",),
                   "pod": ("pod",)}  # fmt: skip
# A qualification record's stage folder, and the stages its incident is billed at: the repair's
# rates are qualified as numerical runs
QUALIFICATION_STAGES = {"engineering": ("engineering",), "numerical": ("numerical", "repair")}
INCIDENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
EMBARGO_FIELDS = ("stage", "what")
LINE = 200  # a one-line text (a cause, an evidence reference, a breach's what), at most
FINAL_THREE = ("analysis", "validation", "audit")
# B_E: why its MAR estimate need not lie inside its worst-case bound
BIAS_NOTE = (
    "B_E's MAR estimate averages over the draws with a resolved reference only, while "
    "each missing draw's term in the worst-case bound is fixed in sign by its known "
    "judge label: the MAR estimate can lie outside the worst-case bound"
)


# ------------------------------------------------------------------ the researcher's files


def read_billing(path: str | Path) -> dict[str, Any]:
    """The billing file (closed schema): ``currency`` (``USD``); ``stages``, each ``{"stage",
    "spent"}`` (a ``MONEY_STAGES`` id, once, and a finite amount ≥ 0); ``incidents``, each
    ``{"id", "stage", "incident_time", "cause", "billed", "evidence"}`` (the id ``INCIDENT_ID``,
    once; the time ISO 8601 in UTC; the cause one line; the billed price a finite amount ≥ 0; the
    evidence a non-empty list of items, each ``{"journal": <log under RUNS_DIR>, "line": n}``
    (n ≥ 1), ``{"qualification": <record path under RUNS_DIR>}`` or ``{"pod": <provider
    reference>}``, each text one line, and no journal line or qualification record named twice,
    within an incident or across incidents); ``policy_stops``, each ``{"stage"}``. ``ValueError``
    naming every problem, an unknown field included."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = billing_problems(data)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return data


def billing_problems(data: Any) -> list[str]:
    """Every problem of a billing record (``read_billing``'s schema)."""
    if not isinstance(data, dict):
        return ["the billing file is no JSON object"]
    problems = _closed(data, BILLING_FIELDS, "the billing file")
    if problems:
        return problems
    if data["currency"] != CURRENCY:
        problems.append(f"currency is not {CURRENCY!r}")
    for key in ("stages", "incidents", "policy_stops"):
        if not isinstance(data[key], list):
            problems.append(f"{key} is not a list")
    if problems:
        return problems
    seen: set[Any] = set()
    for i, entry in enumerate(data["stages"]):
        where = f"stages[{i}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} is no object")
            continue
        found = _closed(entry, STAGE_FIELDS, where)
        if not found:
            found += _stage(entry["stage"], where)
            found += _amount(entry["spent"], f"{where}.spent")
            if entry["stage"] in seen:
                found.append(f"{where}: stage {entry['stage']!r} is listed twice")
            seen.add(entry["stage"])
        problems += found
    ids: set[str] = set()
    named: dict[tuple[Any, ...], str] = {}  # a record named → the incident naming it
    for i, entry in enumerate(data["incidents"]):
        where = f"incidents[{i}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} is no object")
            continue
        found = _closed(entry, INCIDENT_FIELDS, where)
        if not found:
            if not (isinstance(entry["id"], str) and INCIDENT_ID.fullmatch(entry["id"])):
                found.append(f"{where}.id is not an id (a letter or digit, then up to 63 "
                             "letters, digits, '.', '_' or '-')")  # fmt: skip
            elif entry["id"] in ids:
                found.append(f"{where}: id {entry['id']!r} is listed twice")
            else:
                ids.add(entry["id"])
            found += _stage(entry["stage"], where)
            if _instant(entry["incident_time"]) is None:
                found.append(f"{where}.incident_time is no ISO 8601 time in UTC")
            if not _one_line(entry["cause"]):
                found.append(f"{where}.cause is not one non-empty line")
            found += _amount(entry["billed"], f"{where}.billed")
            found += _evidence(entry["evidence"], where, named)
        problems += found
    for i, entry in enumerate(data["policy_stops"]):
        where = f"policy_stops[{i}]"
        if not isinstance(entry, dict):
            problems.append(f"{where} is no object")
            continue
        found = _closed(entry, ("stage",), where)
        problems += found or _stage(entry["stage"], where)
    return problems


def read_embargo(path: str | Path) -> list[dict[str, str]]:
    """The researcher's embargo file (closed schema): ``{"breaches": [{"stage", "what"}]}``, each
    breach a harm label opened outside the two scheduled points, with the stage it affected (a
    ``MONEY_STAGES`` id) and what was opened (one line); ``[]`` states that there was none.
    ``ValueError`` naming every problem."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the embargo file is no JSON object")
    problems = _closed(data, ("breaches",), "the embargo file")
    if not problems and not isinstance(data["breaches"], list):
        problems.append("breaches is not a list")
    if not problems:
        for i, entry in enumerate(data["breaches"]):
            where = f"breaches[{i}]"
            if not isinstance(entry, dict):
                problems.append(f"{where} is no object")
                continue
            found = _closed(entry, EMBARGO_FIELDS, where)
            if not found:
                found += _stage(entry["stage"], where)
                if not _one_line(entry["what"]):
                    found.append(f"{where}.what is not one non-empty line")
            problems += found
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return [dict(entry) for entry in data["breaches"]]


def _closed(data: Mapping[str, Any], fields: Sequence[str], where: str) -> list[str]:
    found = [f"{where} has no {key!r}" for key in fields if key not in data]
    return found + [f"{where}: {key!r} is not a field of it" for key in data if key not in fields]


def _stage(value: Any, where: str) -> list[str]:
    return [] if value in MONEY_STAGES else [f"{where}.stage is not one of {list(MONEY_STAGES)}"]


def _amount(value: Any, where: str) -> list[str]:
    ok = (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(value)
        and value >= 0
    )
    return [] if ok else [f"{where} is not a finite amount of at least 0"]


def _one_line(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and "\n" not in value
        and "\r" not in value
        and len(value) <= LINE
    )


def _evidence(items: Any, where: str, named: dict[tuple[Any, ...], str]) -> list[str]:
    """The problems of an incident's evidence (``read_billing``'s schema); each record it names
    is entered in ``named`` (its key → the incident), and one named before is a problem."""
    if not isinstance(items, list) or not items:
        return [f"{where}.evidence is not a non-empty list"]
    found = []
    for j, item in enumerate(items):
        at = f"{where}.evidence[{j}]"
        kind = _evidence_kind(item)
        if kind is None:
            found.append(f'{at} is not {{"journal", "line"}}, {{"qualification"}} or {{"pod"}}')
            continue
        if not _one_line(item[kind]):
            found.append(f"{at}.{kind} is not one non-empty line")
            continue
        line = item.get("line")
        if kind == "journal" and (isinstance(line, bool) or not isinstance(line, int) or line < 1):
            found.append(f"{at}.line is not a line number (an integer from 1)")
            continue
        key = _record_key(item)
        if key is None:
            continue
        if key in named:
            found.append(f"{at}: {_shown(key)} is named twice (by {named[key]} too)")
        else:
            named[key] = where
    return found


def _evidence_kind(item: Any) -> str | None:
    """The kind of an evidence item (``EVIDENCE_FIELDS``), None for anything else."""
    if not isinstance(item, dict):
        return None
    return next((kind for kind, keys in EVIDENCE_FIELDS.items() if set(item) == set(keys)), None)


def _record_key(item: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """The record an evidence item names: ``("journal", log, line)`` or ``("qualification",
    path)``; None for a provider reference (``pod``), which names no record."""
    if "journal" in item:
        return ("journal", item["journal"], item["line"])
    if "qualification" in item:
        return ("qualification", item["qualification"])
    return None


def _shown(key: tuple[Any, ...]) -> str:
    """A record's key as the refusals name it."""
    if key[0] == "journal":
        return f"the launch journal's {key[1]} line {key[2]}"
    return f"the qualification's --infrastructure record {key[1]}"


def _instant(value: Any) -> datetime | None:
    """An ISO 8601 time in UTC as an instant; None for anything else."""
    if not isinstance(value, str):
        return None
    try:
        when = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return when if when.tzinfo is not None and when.utcoffset() == timedelta(0) else None


# ------------------------------------------------------------------ the incidents of the records


def journal_incidents(runs_root: str | Path) -> list[dict[str, Any]]:
    """Every study-3 ``recovery`` line of the launch journal that carries an incident
    (``INCIDENT_KEYS``: ``incident_time``, ``cause``, ``evidence``), with its log (its path under
    RUNS_DIR), line number and ``stage``: the one step its log's study-3 starts name in their
    campaign header (a ``MONEY_STAGES`` id; None when they name none or several). Read through
    ``study3_records.journal_lines``. ``ValueError`` when a log cannot be read, resolves under a
    sealed folder or holds a torn line that is not resolved: what it held cannot be shown."""
    logs, problems = journal_lines(runs_root)
    torn = [f"{log.path} line {number}" for log in logs for number in log.corrupt]
    if problems or torn:
        raise ValueError(
            "the launch journal's incidents cannot be shown: "
            + "; ".join([*problems, *(f"{t}: a torn line" for t in torn)])
        )
    found = []
    for log in logs:
        steps = {_step(line.record) for line in log.lines if line.record is not None
                 and line.record.get("event") == "start"
                 and line.record.get("experiment") == EXPERIMENT}  # fmt: skip
        stage = next(iter(steps)) if len(steps) == 1 and steps <= set(MONEY_STAGES) else None
        for line in log.lines:
            record = line.record
            if (
                record is None
                or record.get("event") != "recovery"
                or record.get("experiment") != EXPERIMENT
                or not any(key in record for key in INCIDENT_KEYS)
            ):
                continue
            found.append(
                {
                    "log": log.path,
                    "line": line.line,
                    "stage": stage,
                    **{key: record.get(key) for key in INCIDENT_KEYS},
                }
            )
    return found


def _step(start: Mapping[str, Any]) -> Any:
    """The step a start's campaign header names (None without one)."""
    header = start.get("header")
    step = header.get("step") if isinstance(header, Mapping) else None
    return step if isinstance(step, str) else None


def qualification_incidents(runs_root: str | Path) -> list[dict[str, Any]]:
    """Every qualification record made with ``--infrastructure``: a part record
    (``<part>-attempt-<n>.json``) carrying ``infrastructure``, and every case restart
    (``restart-<n>.json``, the records ``qualification_records.restart_checks`` checks), of any
    candidate and stage, each by its path under RUNS_DIR (``path``), its ``kind`` (``part`` or
    ``restart``) and its sha256. Read through ``qualification_records``' reader of a record on
    disk, which opens nothing under a sealed folder. ``ValueError`` for a part record that cannot
    be read: whether it was made with ``--infrastructure`` cannot be shown."""
    from . import qualification_records as records  # it loads torch (its verdicts): here only

    root = Path(runs_root)
    found = []
    restarts = set(records.restart_checks(root))
    for path in sorted((root / "qualification").glob("c*/*/s*/*/*.json")):
        where = path.relative_to(root).as_posix()
        if where in restarts and records.RESTART_FILE.fullmatch(path.name):
            found.append(
                {"path": where, "kind": "restart", "sha256": records._file_sha256(path, root)}
            )
        elif records.PART_FILE.fullmatch(path.name):
            record = records._object(path, root)
            if record is None:
                raise ValueError(
                    f"{where} cannot be read (or resolves under a sealed folder): "
                    "whether it was made with --infrastructure cannot be "
                    "shown"
                )
            if record.get("infrastructure") is not None:
                found.append(
                    {"path": where, "kind": "part", "sha256": records._file_sha256(path, root)}
                )
    return found


def simulation_results(runs_root: str | Path) -> dict[str, Any] | None:
    """The simulation's per-test rows (module docstring, "The simulation's results"): its results
    file's path under RUNS_DIR, its sha256 and its ``tests``, each row with ``ROW_FIELDS``; None
    when there is no results file. ``ValueError`` for a file under a sealed folder, one that is no
    JSON record of the simulation's results, or a row off its fields."""
    root = Path(runs_root)
    path = root / SIMULATION_RESULTS
    if not (path.exists() or path.is_symlink()):
        return None
    if under_sealed(path, root):
        raise ValueError(f"{SIMULATION_RESULTS} resolves under a sealed folder: never opened")
    data = path.read_bytes()
    results = json.loads(data)
    if not (
        isinstance(results, dict)
        and results.get("kind") == "simulation-results"
        and isinstance(results.get("tests"), list)
    ):
        raise ValueError(f"{SIMULATION_RESULTS} is no record of the simulation's results")
    rows = results["tests"]
    off = [
        i
        for i, row in enumerate(rows)
        if not (isinstance(row, dict) and set(row) == set(ROW_FIELDS))
    ]
    if off:
        raise ValueError(
            f"{SIMULATION_RESULTS}: tests{off[:5]} are not rows with {list(ROW_FIELDS)}"
        )
    return {
        "path": SIMULATION_RESULTS,
        "sha256": hashlib.sha256(data).hexdigest(),
        "tests": [dict(row) for row in rows],
    }


# ------------------------------------------------------------------ the facts


def report_facts(
    state: Mapping[str, Any],
    *,
    records: Mapping[str, Any],
    billing: Mapping[str, Any],
    incidents: Mapping[str, Sequence[Mapping[str, Any]]],
    labeling_problems: Sequence[str],
    analysis: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    audit: Mapping[str, Any] | None = None,
    embargo: Sequence[Mapping[str, Any]] = (),
    judge_diagnoses: Sequence[Any] = (),
) -> dict[str, Any]:
    """The report's facts (module docstring), strict JSON, one entry per ``SECTIONS``.

    ``state``: ``study3_progress.read_state``'s. ``records`` (each optional; absent, its section
    is not reached): ``endpoints`` (the state's), ``engineering``, ``simulation``, ``decisions``
    (``n-c<N>`` and ``budget-c<N>`` by kind) and ``freeze``, each ``{"path", "sha256",
    "record"}``; ``policy`` (``study3_versions.registered_policy`` at HEAD);
    ``engineering_verdict`` (``{"path", "sha256", "summary"}``); ``simulation_results``
    (``simulation_results``'); ``qualification`` (candidate → ``read_qualification``'s reading).
    ``billing``: ``read_billing``'s; ``incidents``: ``{"journal": journal_incidents,
    "qualification": qualification_incidents}``; ``labeling_problems``: the runs with a technical
    labeling problem (the label reader's ledger); ``analysis``, ``validation`` and ``audit``: the
    analysis, validation's harm and the audit's estimate, each ``{"path", "sha256", "record"}``;
    ``embargo``: ``read_embargo``'s; ``judge_diagnoses``: every diagnosis record of the labeling
    ledger whose cause is ``judge`` (each ``path``, ``sha256``, ``batch``, ``cause``,
    ``diagnosis``, ``evidence``), given exactly for the ``judge_technical`` variant.

    ``ValueError``: the ``judge_technical`` variant with a diagnosis whose cause is not
    ``judge``, or with the analysis, validation's harm or the audit; the standard variant at a
    stage that is neither ``done`` nor terminal, at ``done`` without the analysis, validation's
    harm or the audit's estimate (naming the runs with a technical labeling problem, and that a
    judge diagnosis calls for the ``judge_technical`` variant), or with them before ``done``; a
    billing record off its schema; an incident of the records (a launch-journal line, a
    qualification ``--infrastructure`` record) named by no billed incident, a record an incident
    names that is none of them, or an incident billed at a stage that is not its records'; a
    stage the route or an incident reached with no spending entry (module docstring, "Money")."""
    diagnoses = list(judge_diagnoses)
    variant = _variant(
        state,
        labeling_problems,
        diagnoses,
        analysis=analysis,
        validation=validation,
        audit=audit,
    )
    money = _money(billing, incidents, state.get("route") or ())
    endpoints = list(records.get("endpoints") or ())
    uncounted = {entry.get("run_id") for entry in state.get("not_counted") or ()}
    kept = [e for e in endpoints if e.run_id not in uncounted]
    simulation = records.get("simulation")
    validation_ = _validation(state, kept)
    confirmation = _confirmation(state, kept)
    judged = variant == "judge_technical"
    facts = {
        "outcome": _outcome(state, variant, diagnoses),
        "dry_run": _dry_run(records.get("engineering"), records.get("policy")),
        "engineering": _engineering(records.get("engineering_verdict")),
        "simulation": _simulation(simulation, records.get("simulation_results")),
        "qualification": _qualification(records.get("qualification")),
        "not_counted": [dict(entry) for entry in state.get("not_counted") or ()],
        "validation": validation_,
        "n": _n(records.get("decisions") or {}, state.get("candidate"), simulation),
        "confirmation": confirmation,
        "utility": _utility(state, kept, endpoints, validation_, confirmation),
        "analysis": (JUDGE_NOT_REACHED if judged else _analysis(analysis, records.get("freeze"))),
        "validation_harm": JUDGE_NOT_REACHED if judged else _validation_harm(validation),
        "audit": JUDGE_NOT_REACHED if judged else _audit(audit),
        "labeling": sorted(labeling_problems),
        "freeze": _freeze(records.get("freeze")),
        "judge_mark": [_judge_mark(d) for d in diagnoses] if judged else None,
        "money": money,
        "embargo": [{key: entry[key] for key in EMBARGO_FIELDS} for entry in embargo],
    }
    out = _plain(facts)
    json.dumps(out, allow_nan=False)  # strict JSON, or a ValueError here
    return out


def _variant(
    state: Mapping[str, Any],
    labeling_problems: Sequence[str],
    diagnoses: Sequence[Any],
    **final: Any,
) -> str:
    """The variant the inputs call for, or the refusal (``report_facts``)."""
    given = [name for name in FINAL_THREE if final.get(name) is not None]
    stage = state.get("stage")
    if diagnoses:
        causes = [_field(d, "cause") for d in diagnoses]
        if any(cause != "judge" for cause in causes):
            raise ValueError(
                f"the judge_technical variant needs the judge's diagnosis records, each "
                f"of cause judge, not {causes!r}: an infrastructure diagnosis releases "
                "its rounds, and labeling resumes"
            )
        if given:
            raise ValueError(
                f"a judge diagnosis with {', '.join(given)}: the judge_technical "
                "variant has no Δ, no interval and no audit, and the variants do not "
                "mix"
            )
        return "judge_technical"
    if stage != "done" and stage not in TERMINAL:
        raise ValueError(
            f"the study is at {stage!r}: the report is written at done or at a "
            f"terminal outcome ({', '.join(TERMINAL)}), or as the judge_technical "
            "variant after a judge diagnosis"
        )
    missing = [name for name in FINAL_THREE if final.get(name) is None]
    if stage == "done" and missing:
        problems = ", ".join(labeling_problems) or "none"
        raise ValueError(
            f"at done the report needs the analysis, validation's harm and the "
            f"audit's estimate: {', '.join(missing)} not written. Runs with a "
            f"technical labeling problem: {problems}; a judge diagnosis (cause judge) "
            "calls for the judge_technical variant"
        )
    if stage != "done" and given:
        raise ValueError(f"{', '.join(given)} at {stage!r}: the analysis follows done")
    return "standard"


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)


def _link(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return None if entry is None else {"path": entry.get("path"), "sha256": entry.get("sha256")}


# -- money


def _money(
    billing: Mapping[str, Any],
    incidents: Mapping[str, Sequence[Mapping[str, Any]]],
    route: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Money (module docstring): the billing as it is, each incident of the records named by one
    billed incident of its stage, nothing else named, and a spending entry for every stage the
    route (its events' stages) or an incident reached."""
    problems = billing_problems(billing)
    if problems:
        raise ValueError("the billing file: " + "; ".join(problems))
    billed = list(billing["incidents"])
    # each record of an incident → the stages its incident can be billed at (None: not shown)
    records: dict[tuple[Any, ...], tuple[str, ...] | None] = {}
    for row in incidents.get("journal") or ():
        stage = row.get("stage")
        records[("journal", row.get("log"), row.get("line"))] = None if stage is None else (stage,)
    for row in incidents.get("qualification") or ():
        records[("qualification", row.get("path"))] = _qualification_stages(row.get("path"))
    found, named = [], set()
    for entry in billed:
        for item in entry["evidence"]:
            key = _record_key(item)
            if key is None:
                continue
            named.add(key)
            if key not in records:
                found.append(f"incident {entry['id']}: {_shown(key)} is no incident of the "
                             "records")  # fmt: skip
                continue
            stages = records[key]
            if stages is not None and entry["stage"] not in stages:
                found.append(f"incident {entry['id']} is billed at {entry['stage']}, but "
                             f"{_shown(key)} is of {' or '.join(stages)}")  # fmt: skip
    found += [
        f"{_shown(key)} is named by no billed incident" for key in records if key not in named
    ]
    if found:
        raise ValueError("every incident of the records is billed once, at its stage: "
                         + "; ".join(found))  # fmt: skip
    reached = {e.get("stage") for e in route} | {entry["stage"] for entry in billed}
    spent = {entry["stage"] for entry in billing["stages"]}
    unlisted = [stage for stage in MONEY_STAGES if stage in reached and stage not in spent]
    if unlisted:
        raise ValueError(f"the billing file has no spending entry for {', '.join(unlisted)}: every "
                         "stage the route or an incident reached is listed with what it "
                         "spent")  # fmt: skip
    return {
        "currency": billing["currency"],
        "by_stage": [dict(entry) for entry in billing["stages"]],
        "incidents": [
            {
                **{key: entry[key] for key in INCIDENT_FIELDS},
                "evidence": [dict(item) for item in entry["evidence"]],
            }
            for entry in billed
        ],
        "policy_stops": [entry["stage"] for entry in billing["policy_stops"]],
    }


def _qualification_stages(path: Any) -> tuple[str, ...] | None:
    """The stages a qualification record's incident is billed at, by its stage folder
    (``qualification/c<N>/<stage>/...``, ``QUALIFICATION_STAGES``); None when not shown."""
    parts = PurePosixPath(path).parts if isinstance(path, str) else ()
    if len(parts) < 3 or parts[0] != "qualification":
        return None
    return QUALIFICATION_STAGES.get(parts[2])


# -- the outcome


def _outcome(state: Mapping[str, Any], variant: str, diagnoses: Sequence[Any]) -> dict[str, Any]:
    closures = {}
    for kind, entry in sorted((state.get("closures") or {}).items()):
        record = entry.get("record") or {}
        closures[kind] = {
            "state": entry.get("state"),
            "path": entry.get("path"),
            **{key: record.get(key) for key in ("run_id", "arm", "reason", "decision")},
        }
    return {
        "variant": variant,
        "stage": state.get("stage"),
        "candidate": state.get("candidate"),
        "outcome": state.get("outcome"),
        "reason": state.get("reason"),
        "reasons": list(state.get("reasons") or ()),
        "readings": state.get("readings"),
        "active": state.get("active"),
        "route": [dict(e) for e in state.get("route") or ()],
        "closures": closures,
        "dead_ends": _dead_ends(state, diagnoses),
    }


def _dead_ends(
    state: Mapping[str, Any], diagnoses: Sequence[Any] = ()
) -> dict[str, list[dict[str, Any]]]:
    """The technical dead ends and the scientific failures apart (module docstring,
    "Rulings"), each with its candidate, stage, arms and reason id; the judge's technical mark
    (``diagnoses``, the ``judge_technical`` variant's) a technical one per diagnosis, at the
    labeling, with its batch."""
    technical: list[dict[str, Any]] = []
    scientific: list[dict[str, Any]] = []
    closures = state.get("closures") or {}
    for event in state.get("route") or ():
        kind, why = event.get("event"), str(event.get("why") or "")
        entry = {key: event.get(key) for key in ("candidate", "stage", "arms")}
        gate = why.startswith(("the A–F validation gate", "the P–F validation gate"))
        failed = gate or why.startswith("a scientific failure")
        if kind == "candidate" or (kind == "close_pf" and failed):
            scientific.append(
                {**entry, "reason": "validation_gate" if gate else "scientific_failure"}
            )
        elif kind == "close_pf" and why.startswith("unresolved after"):
            technical.append({**entry, "reason": "unresolved_after_cycle"})
        elif kind == "close_pf":  # a closure record: its kind
            record = (closures.get(why) or {}).get("record") or {}
            technical.append({**entry, "reason": record.get("reason") or "closure", "closure": why})
        elif kind == "report":
            technical.append({**entry, "reason": state.get("reason")})
    for diagnosis in diagnoses:
        technical.append(
            {
                "candidate": state.get("candidate"),
                "stage": "labeling",
                "arms": [],
                "reason": "judge_mark",
                "batch": _field(diagnosis, "batch"),
            }
        )
    return {"technical": technical, "scientific": scientific}


# -- the stages before the confirmation


def _dry_run(engineering: Mapping[str, Any] | None, policy: Mapping[str, Any] | None) -> Any:
    if engineering is None:
        return NOT_REACHED
    record = engineering.get("record") or {}
    checks = {}
    for name, check in (record.get("checks") or {}).items():
        checks[name] = (
            None
            if not isinstance(check, Mapping)
            else {key: check.get(key) for key in ("path", "sha256", "root", "passed")}
        )
    return {
        "record": _link(engineering),
        "state": record.get("state"),
        "checks": checks,
        "policy": None
        if policy is None
        else {key: policy.get(key) for key in ("path", "commit", "sha256", "amendments")},
    }


def _engineering(verdict: Mapping[str, Any] | None) -> Any:
    if verdict is None:
        return NOT_REACHED
    summary = verdict.get("summary") or {}
    return {
        "verdict": _link(verdict),
        "e_eng": summary.get("e_eng"),
        "repeats": summary.get("repeats"),
        "path_maxima": dict(summary.get("path_maxima") or {}),
        "margins": dict(summary.get("margins") or {}),
    }


def _simulation(link: Mapping[str, Any] | None, results: Mapping[str, Any] | None) -> Any:
    if link is None:
        return NOT_REACHED
    record = link.get("record") or {}
    gates = {
        name: {
            key: gate.get(key)
            for key in ("value", "requirement", "passed", "power_short", "n_min")
            if key in gate
        }
        for name, gate in (record.get("gates") or {}).items()
    }
    coverage = []
    for row in record.get("coverage") or ():
        counts, limits = row.get("counts") or {}, row.get("thresholds") or {}
        coverage.append(
            {
                "level": row.get("level"),
                "tests": row.get("tests"),
                "J": row.get("J"),
                "M": row.get("M"),
                "passed": row.get("passed"),
                **{
                    kind: {"X": counts.get(kind), "c": limits.get(kind)}
                    for kind in ("comparison", "joint")
                },
            }
        )
    tests: Any = NOT_REACHED
    if results is not None:
        tests = [
            {
                "test": list(row["test"]),
                "level": row["level"],
                "due": row["due"],
                "studies": row["studies"],
                "X": row["X"],
                "c": row["c"],
                "passed": row["passed"],
                "non_coverage": row["rate"],
                "interval": row["interval"],
                "reason": row["reason"],
            }
            for row in results.get("tests") or ()
        ]
    return {
        "record": _link(link),
        "state": record.get("state"),
        "level": record.get("level"),
        "validation_blocks": record.get("validation_blocks"),
        "gates": gates,
        "coverage": coverage,
        "verification": record.get("verification"),
        "results": _link(results),
        "tests": tests,
    }


def _qualification(readings: Mapping[int, Mapping[str, Any]] | None) -> Any:
    if not readings:
        return NOT_REACHED
    out = {}
    for candidate, reading in sorted(readings.items()):
        rates = {}
        for key, rate in (reading.get("rates") or {}).items():
            verdict = rate.get("verdict") or {}
            rates[key] = {
                "arm": rate.get("arm"),
                "lr": rate.get("lr"),
                "state": rate.get("state"),
                "fallbacks": list(verdict.get("fallbacks") or ()),
                "warnings": list(verdict.get("warnings") or ()),
            }
        supersessions = []
        for entry in reading.get("supersessions") or ():
            old = entry.get("old_record")
            supersessions.append(
                {
                    "path": entry.get("path"),
                    "sha256": entry.get("sha256"),
                    "old_record": (
                        {"path": old.get("path"), "sha256": old.get("sha256")}
                        if isinstance(old, Mapping)
                        else old
                    ),
                    "cases_requalified": entry.get("cases_requalified"),
                }
            )
        restarts = [
            {key: entry.get(key) for key in ("path", "sha256", "case", "launcher_run_id")}
            for entry in reading.get("restarts") or ()
        ]
        out[f"c{candidate}"] = {
            "rates": rates,
            "supersessions": supersessions,
            "restarts": restarts,
            "problems": len(reading.get("problems") or ()),
        }
    return out


def _by_seed(endpoints: Iterable[Any]) -> dict[int | None, list[Any]]:
    by_seed: dict[int | None, list[Any]] = defaultdict(list)
    for e in endpoints:
        by_seed[e.seed].append(e)
    return by_seed


def _block(
    by_seed: Mapping[int | None, Sequence[Any]], seed: int, candidate: int, arms: Sequence[str]
) -> dict[str, Any]:
    return block_state([*by_seed.get(seed, ()), *by_seed.get(None, ())], seed, candidate, arms)


def _validation(state: Mapping[str, Any], kept: Sequence[Any]) -> Any:
    found = state.get("validation") or {}
    gates = found.get("gates")
    if not gates:
        return NOT_REACHED
    candidate = state.get("candidate")
    arms = [arm for arm in ARM_ORDER if any(arm in PAIRS[name] for name in gates)]
    by_seed = _by_seed(kept)
    blocks = [
        _block(by_seed, seed, candidate, arms)
        for seed in SEEDS["validation"][candidate][: found["blocks"]]
    ]
    utility = {name: _gate(utility_gate(blocks, name, VALIDATION_ALPHA)) for name in gates}
    return {
        "blocks": found["blocks"],
        "gates": {name: dict(g) for name, g in gates.items()},
        "utility": utility,
        "spacing": _spacing(kept, candidate, arms),
    }


def _gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    """A utility gate as the report shows it: per arm and metric its n, mean, upper bound, limit
    (the registered 5 or 10 pp) and pass, in pp."""
    bounds = {
        arm: {
            metric: {
                "n": b.get("n"),
                "mean_pp": b.get("mean"),
                "upper_pp": b.get("upper"),
                "limit_pp": b.get("threshold"),
                "passes": b.get("passes"),
            }
            for metric, b in per_arm.items()
        }
        for arm, per_arm in (gate.get("bounds") or {}).items()
    }
    out = {key: gate.get(key) for key in ("comparison", "alpha", "n", "computed", "passes")}
    out |= {
        "excluded": list(gate.get("excluded") or ()),
        "waiting": [dict(w) for w in gate.get("waiting") or ()],
        "bounds": bounds,
    }
    if gate.get("reason") is not None:
        out["reason"] = gate["reason"]
    return out


def _spacing(kept: Sequence[Any], candidate: int, arms: Sequence[str]) -> dict[str, Any]:
    """The spacing (module docstring): per arm and rate, the SD of ln K over the validation
    blocks and the ratio of the registered spacing to it."""
    out: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        rates: dict[float, list[float]] = {}
        for e in kept:
            if not (
                e.cell.step == "validation"
                and e.candidate == candidate
                and e.arm == arm
                and e.counts
                and e.lr is not None
                and isinstance(e.K, int | float)
                and math.isfinite(e.K)
                and e.K > 0
            ):
                continue
            key = next((r for r in rates if math.isclose(r, e.lr, rel_tol=RATE_TOLERANCE)), e.lr)
            rates.setdefault(key, []).append(math.log(e.K))
        rows = []
        for lr, values in sorted(rates.items()):
            sd = statistics.stdev(values) if len(values) > 1 else None
            rows.append(
                {
                    "lr": lr,
                    "blocks": len(values),
                    "sd_ln_k": sd,
                    "ratio": SPACING / sd if sd else None,
                }
            )
        out[arm] = rows
    return {"registered": SPACING, "arms": out}


def _n(decisions: Mapping[str, Any], candidate: Any, simulation: Mapping[str, Any] | None) -> Any:
    entry = decisions.get(f"n-c{candidate}")
    if entry is None:
        return NOT_REACHED
    record = entry.get("record") or {}
    gates = ((simulation or {}).get("record") or {}).get("gates") or {}
    budget = decisions.get(f"budget-c{candidate}")
    return {
        "record": _link(entry),
        "state": record.get("state"),
        "N": record.get("N"),
        "cap": record.get("cap"),
        "spread_pp": record.get("spread_pp"),
        "level": record.get("level"),
        "active": record.get("active"),
        "cohort_blocks": len((record.get("cohort") or {}).get("seeds") or ()),
        "power_pf": (gates.get("power_pf") or {}).get("value"),
        "budget": None
        if budget is None
        else {"record": _link(budget), "decision": (budget.get("record") or {}).get("decision")},
    }


def _confirmation(state: Mapping[str, Any], kept: Sequence[Any]) -> Any:
    found = state.get("confirmation")
    if not found:
        return NOT_REACHED
    candidate, active, started = found["candidate"], list(found["active"]), list(found["started"])
    arms = [arm for arm in ARM_ORDER if arm != "P" or "P-F" in active]
    by_seed = _by_seed(kept)
    outside = {}
    for comparison in active:
        cohort = set(
            (found.get("analysis") or {}).get(comparison)
            or (found.get("cohorts") or {}).get(comparison)
            or ()
        )
        rows = []
        for seed in started:
            if seed in cohort:
                continue
            pair = _block(by_seed, seed, candidate, arms)[COMPARISONS[comparison]]
            rows.append(
                {
                    "seed": seed,
                    "state": pair["state"],
                    "reason": "beyond_cohort" if pair["state"] == SUPPORTED else pair["state"],
                    "arms": dict(pair["reasons"]),
                }
            )
        outside[comparison] = rows
    return {
        "candidate": candidate,
        "N": found.get("N"),
        "cap": found.get("cap"),
        "started": len(started),
        "seeds": [started[0], started[-1]] if started else [],
        "counters": found.get("counters"),
        "supported": found.get("supported"),
        "stop": found.get("stop"),
        "readings": found.get("readings"),
        "outside": outside,
    }


def _utility(
    state: Mapping[str, Any],
    kept: Sequence[Any],
    endpoints: Sequence[Any],
    validation: Any,
    confirmation: Any,
) -> Any:
    rows = [
        {
            "run_id": e.run_id,
            "step": e.cell.step,
            "candidate": e.candidate,
            "arm": e.arm,
            "lr": e.lr,
            "seed": e.seed,
            "attempt": e.attempt,
            "state": e.state,
            "unusable_reason": e.unusable_reason,
            "mmlu_loss_pp": e.mmlu_loss_pp,
            "factual_loss_pp": e.factual_loss_pp,
        }
        for e in endpoints
        if e.cell.step in ("validation", "confirmation")
    ]
    if not rows:
        return NOT_REACHED
    rows.sort(
        key=lambda r: (
            r["step"] != "validation",
            r["candidate"] or 0,
            r["seed"] or 0,
            ARM_ORDER.index(r["arm"]) if r["arm"] in ARM_ORDER else 9,
            r["lr"] or 0.0,
            r["attempt"] or 0,
            r["run_id"],
        )
    )
    exclusions: dict[str, Any] = {}
    if isinstance(validation, Mapping):
        exclusions["validation"] = {
            name: gate["excluded"] for name, gate in validation["utility"].items()
        }
    if isinstance(confirmation, Mapping):
        exclusions["confirmation"] = {
            c: [r["seed"] for r in found if r["state"] != SUPPORTED]
            for c, found in confirmation["outside"].items()
        }
    worst = None
    found = state.get("confirmation")
    if found:
        started = set(found["started"])
        ended = [
            e
            for e in kept
            if e.cell.step == "confirmation"
            and e.candidate == found["candidate"]
            and e.seed in started
            and e.state in ENDED
        ]
        worst = {}
        for arm in ARM_ORDER:
            mine = [e for e in ended if e.arm == arm]
            if not mine:
                continue
            worst[arm] = {}
            for metric, field in (("mmlu", "mmlu_loss_pp"), ("factual", "factual_loss_pp")):
                valued = [e for e in mine if getattr(e, field) is not None]
                top = max(valued, key=lambda e, f=field: getattr(e, f), default=None)
                worst[arm][metric] = (
                    None
                    if top is None
                    else {"loss_pp": getattr(top, field), "run_id": top.run_id, "seed": top.seed}
                )
    return {"endpoints": rows, "exclusions": exclusions, "worst": worst}


# -- the final analysis, validation's harm, the audit and the freeze


INTERVALS = ("primary", "missing_labels", "all_missing_harmful")


def _comparison(entry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if entry is None:
        return None
    keys = (
        "n",
        "status",
        *INTERVALS,
        "readings",
        "reading",
        "wording",
        "missing",
        "worst_case_unknown_blocks",
        "observed_extremes",
    )
    return {key: entry[key] for key in keys if key in entry}


def _analysis(link: Mapping[str, Any] | None, freeze: Mapping[str, Any] | None) -> Any:
    if link is None:
        return NOT_REACHED
    record = link["record"]
    utility = record.get("utility") or {}
    harmful = ((freeze or {}).get("record") or {}).get("k_harmful")
    kappa = record.get("kappa") or {}
    return {
        "record": _link(link),
        "freeze": record.get("freeze"),
        "supersedes": record.get("superseded"),
        "level": record.get("level"),
        "bootstrap": record.get("bootstrap"),
        "comparisons": {
            key: _comparison(entry) for key, entry in (record.get("comparisons") or {}).items()
        },
        "utility": {
            "claim": utility.get("claim"),
            "gates": {name: _gate(g) for name, g in (utility.get("gates") or {}).items()},
        },
        "matched_k_harmful": None
        if harmful is None
        else {
            "kstar_h": harmful.get("kstar_h"),
            "n": len(harmful.get("blocks") or ()),
            "left_out": list(harmful.get("left_out") or ()),
        },
        "k_spread": record.get("k_spread"),
        "curvature": {
            "comparisons": record.get("curvature"),
            "kappa": kappa.get("values"),
            "unit": kappa.get("unit"),
            "source": kappa.get("source"),
            "fixation": _link(kappa) if kappa else None,
        },
    }


def _validation_harm(link: Mapping[str, Any] | None) -> Any:
    if link is None:
        return NOT_REACHED
    record = link["record"]
    return {
        "record": _link(link),
        "analysis": record.get("analysis"),
        "comparisons": {
            key: _comparison(entry) for key, entry in (record.get("comparisons") or {}).items()
        },
    }


def _audit(link: Mapping[str, Any] | None) -> Any:
    if link is None:
        return NOT_REACHED
    record = link["record"]
    outside = {}
    for e, bias in (record.get("bias") or {}).items():
        estimate, bound = (bias.get("mar") or {}).get("estimate"), bias.get("worst_case")
        outside[e] = (
            None
            if estimate is None or not bound or None in bound
            else not bound[0] <= estimate <= bound[1]
        )
    return {
        "record": _link(link),
        **{
            key: record.get(key)
            for key in ("selection", "bias", "error_rates", "non_estimable", "missing")
        },
        "mar_outside_worst_case": outside,
        "bias_note": BIAS_NOTE,
    }


def _freeze(link: Mapping[str, Any] | None) -> Any:
    if link is None:
        return NOT_REACHED
    record = link["record"]

    def cohort(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        runs = {
            arm[end] for block in blocks for arm in block["arms"].values() for end in ("lo", "hi")
        }
        return {"seeds": [block["seed"] for block in blocks], "run_ids": sorted(runs)}

    harmful = record.get("k_harmful") or {}
    return {
        "path": link.get("path"),
        "sha256": link.get("sha256"),
        "candidate": record.get("candidate"),
        "confirmation": record.get("confirmation"),
        "comparisons": {
            key: {
                "reading": entry.get("reading"),
                "rule": entry.get("rule"),
                **cohort(entry.get("blocks") or ()),
                "unknown": list(entry.get("unknown") or ()),
            }
            for key, entry in (record.get("comparisons") or {}).items()
        },
        "a_minus_p": cohort((record.get("a_minus_p") or {}).get("blocks") or ()),
        "kstar_h": harmful.get("kstar_h"),
        "matched": len(harmful.get("blocks") or ()),
        "left_out": list(harmful.get("left_out") or ()),
        "validation": {
            key: None if entry is None else cohort(entry.get("blocks") or ())
            for key, entry in (record.get("validation") or {}).items()
        },
    }


def _judge_mark(diagnosis: Any) -> dict[str, Any]:
    out = {key: _field(diagnosis, key) for key in ("path", "sha256", "batch", "cause", "diagnosis")}
    out["evidence"] = [dict(e) for e in _field(diagnosis, "evidence") or ()]
    return out


def _plain(value: Any) -> Any:
    """``value`` as strict JSON: mappings with string keys, tuples and sets as lists, paths as
    strings, a dataclass as its fields, NaN as null and ±∞ as "+inf"/"-inf"."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, set | frozenset):
        return [_plain(v) for v in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("+inf" if value > 0 else "-inf")
    return value


# ------------------------------------------------------------------ the tables


def tables(facts: Mapping[str, Any]) -> str:
    """The facts as markdown tables (module docstring): one heading per section, harm
    proportions shown in pp, no prose."""
    lines: list[str] = []
    for section in SECTIONS:
        value = facts[section]
        lines += ["", f"## {section}", ""]
        if isinstance(value, str):
            lines += _table(("status",), [(value,)])
            continue
        lines += _RENDER[section](value)
    return "\n".join(lines).lstrip("\n") + "\n"


def _cell(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list | tuple):
        return ", ".join(_cell(v) for v in value)
    if isinstance(value, Mapping):
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    else:
        text = str(value)
    return re.sub(r"\s+", " ", text).replace("|", "\\|")


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return lines + [""]


def _pp(value: Any) -> str:
    """A proportion in pp (a "+inf"/"-inf" or null as it is)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return _cell(value)
    return f"{100 * value:.3f}"


def _interval_pp(entry: Any) -> str:
    if not isinstance(entry, Mapping):
        return "null"
    if entry.get("undetermined"):
        return "undetermined"
    return f"[{_pp(entry.get('lower'))}, {_pp(entry.get('upper'))}]"


def _pairs(mapping: Mapping[str, Any]) -> list[str]:
    return _table(("field", "value"), mapping.items())


def _render_outcome(o: Mapping[str, Any]) -> list[str]:
    lines = _pairs(
        {
            k: o[k]
            for k in ("variant", "stage", "candidate", "outcome", "reason", "readings", "active")
        }
    )
    lines += _table(("reasons",), [(r,) for r in o["reasons"]])
    lines += _table(
        ("candidate", "stage", "event", "arms", "why"),
        [
            (e.get("candidate"), e.get("stage"), e.get("event"), e.get("arms"), e.get("why"))
            for e in o["route"]
        ],
    )
    lines += _table(
        ("closure", "state", "path", "run_id", "arm", "reason", "decision"),
        [
            (k, c["state"], c["path"], c["run_id"], c["arm"], c["reason"], c["decision"])
            for k, c in o["closures"].items()
        ],
    )
    lines += _table(
        ("dead end", "candidate", "stage", "arms", "reason", "batch"),
        [
            (
                kind,
                e.get("candidate"),
                e.get("stage"),
                e.get("arms"),
                e.get("reason"),
                e.get("batch"),
            )
            for kind in ("technical", "scientific")
            for e in o["dead_ends"][kind]
        ],
    )
    return lines


def _render_dry_run(d: Mapping[str, Any]) -> list[str]:
    lines = _pairs(
        {"record": d["record"]["path"], "sha256": d["record"]["sha256"], "state": d["state"]}
    )
    lines += _table(
        ("check", "path", "sha256", "root", "passed"),
        [
            (name, *((c or {}).get(k) for k in ("path", "sha256", "root", "passed")))
            for name, c in d["checks"].items()
        ],
    )
    policy = d["policy"] or {}
    return lines + _table(
        ("registration", "commit", "sha256", "amendments"),
        [
            (
                policy.get("path"),
                policy.get("commit"),
                policy.get("sha256"),
                policy.get("amendments"),
            )
        ],
    )


def _render_engineering(e: Mapping[str, Any]) -> list[str]:
    lines = _pairs(
        {
            "verdict": e["verdict"]["path"],
            "sha256": e["verdict"]["sha256"],
            "e_eng": e["e_eng"],
            "repeats": e["repeats"],
        }
    )
    lines += _table(("path maximum", "value"), e["path_maxima"].items())
    return lines + _table(("margin", "value"), e["margins"].items())


def _render_simulation(s: Mapping[str, Any]) -> list[str]:
    lines = _pairs(
        {
            "record": s["record"]["path"],
            "sha256": s["record"]["sha256"],
            "state": s["state"],
            "L": s["level"],
            "validation_blocks": s["validation_blocks"],
            "results": (s["results"] or {}).get("path"),
            "results_sha256": (s["results"] or {}).get("sha256"),
        }
    )
    lines += _table(
        ("gate", "value", "requirement", "passed", "power_short", "n_min"),
        [
            (
                name,
                g.get("value"),
                g.get("requirement"),
                g.get("passed"),
                g.get("power_short"),
                g.get("n_min"),
            )
            for name, g in s["gates"].items()
        ],
    )
    lines += _table(
        (
            "level",
            "tests",
            "J",
            "M",
            "X comparison",
            "c comparison",
            "X joint",
            "c joint",
            "passed",
        ),
        [
            (
                r["level"],
                r["tests"],
                r["J"],
                r["M"],
                r["comparison"]["X"],
                r["comparison"]["c"],
                r["joint"]["X"],
                r["joint"]["c"],
                r["passed"],
            )
            for r in s["coverage"]
        ],
    )
    lines += _pairs({f"verification {k}": v for k, v in (s["verification"] or {}).items()})
    if isinstance(s["tests"], str):
        return lines + _table(("tests",), [(s["tests"],)])
    return lines + _table(
        (
            "test",
            "level",
            "due",
            "studies",
            "X",
            "c",
            "passed",
            "non-coverage",
            "95 % interval",
            "reason",
        ),
        [
            (
                "/".join(str(x) for x in t["test"]),
                t["level"],
                t["due"],
                t["studies"],
                t["X"],
                t["c"],
                t["passed"],
                t["non_coverage"],
                t["interval"],
                t["reason"],
            )
            for t in s["tests"]
        ],
    )


def _render_qualification(q: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for candidate, found in q.items():
        lines += _table(
            ("candidate", "rate", "arm", "lr", "state", "fallbacks", "warnings"),
            [
                (candidate, key, r["arm"], r["lr"], r["state"], r["fallbacks"], r["warnings"])
                for key, r in found["rates"].items()
            ],
        )
        lines += _table(
            (
                "candidate",
                "supersession",
                "sha256",
                "superseded verdict",
                "its sha256",
                "cases re-qualified",
            ),
            [
                (
                    candidate,
                    s["path"],
                    s["sha256"],
                    _field(s["old_record"] or {}, "path"),
                    _field(s["old_record"] or {}, "sha256"),
                    s["cases_requalified"],
                )
                for s in found["supersessions"]
            ],
        )
        lines += _table(
            ("candidate", "case restart", "sha256", "case", "launcher run"),
            [
                (candidate, r["path"], r["sha256"], r["case"], r["launcher_run_id"])
                for r in found["restarts"]
            ],
        )
        lines += _pairs({f"{candidate} problems": found["problems"]})
    return lines


def _render_not_counted(found: Sequence[Mapping[str, Any]]) -> list[str]:
    return _table(
        ("run_id", "label", "candidate", "step", "reason", "commit"),
        [
            (
                e.get("run_id"),
                e.get("label"),
                e.get("candidate"),
                e.get("step"),
                e.get("reason"),
                e.get("commit"),
            )
            for e in found
        ],
    )


def _bound_rows(name: str, gate: Mapping[str, Any]) -> list[tuple]:
    return [
        (name, arm, metric, b["n"], b["mean_pp"], b["upper_pp"], b["limit_pp"], b["passes"])
        for arm, per_arm in gate["bounds"].items()
        for metric, b in per_arm.items()
    ]


BOUND_HEADER = (
    "comparison",
    "arm",
    "metric",
    "n",
    "mean (pp)",
    "upper (pp)",
    "limit (pp)",
    "passes",
)


def _render_validation(v: Mapping[str, Any]) -> list[str]:
    lines = _table(
        ("comparison", "supported", "minimum", "utility", "passes"),
        [
            (name, g.get("supported"), g.get("minimum"), g.get("utility"), g.get("passes"))
            for name, g in v["gates"].items()
        ],
    )
    lines += _table(
        BOUND_HEADER, [row for name, g in v["utility"].items() for row in _bound_rows(name, g)]
    )
    lines += _table(
        ("comparison", "excluded blocks"),
        [(name, g["excluded"]) for name, g in v["utility"].items()],
    )
    lines += _pairs({"blocks": v["blocks"], "registered spacing": v["spacing"]["registered"]})
    return lines + _table(
        ("arm", "lr", "blocks", "SD of ln K", "spacing / SD"),
        [
            (arm, r["lr"], r["blocks"], r["sd_ln_k"], r["ratio"])
            for arm, rows in v["spacing"]["arms"].items()
            for r in rows
        ],
    )


def _render_n(n: Mapping[str, Any]) -> list[str]:
    budget = n["budget"] or {}
    return _pairs(
        {
            "record": n["record"]["path"],
            "sha256": n["record"]["sha256"],
            "state": n["state"],
            "N": n["N"],
            "cap": n["cap"],
            "spread (pp)": n["spread_pp"],
            "L": n["level"],
            "active": n["active"],
            "cohort blocks": n["cohort_blocks"],
            "P-F power": n["power_pf"],
            "budget record": (budget.get("record") or {}).get("path"),
            "budget decision": budget.get("decision"),
        }
    )


def _render_confirmation(c: Mapping[str, Any]) -> list[str]:
    lines = _pairs(
        {
            k: c[k]
            for k in (
                "candidate",
                "N",
                "cap",
                "started",
                "seeds",
                "counters",
                "supported",
                "stop",
                "readings",
            )
        }
    )
    return lines + _table(
        ("comparison", "seed", "state", "reason", "arms"),
        [
            (name, r["seed"], r["state"], r["reason"], r["arms"])
            for name, rows in c["outside"].items()
            for r in rows
        ],
    )


def _render_utility(u: Mapping[str, Any]) -> list[str]:
    lines = _table(
        (
            "run_id",
            "step",
            "candidate",
            "arm",
            "lr",
            "seed",
            "attempt",
            "state",
            "unusable",
            "MMLU loss (pp)",
            "factual loss (pp)",
        ),
        [
            (
                r["run_id"],
                r["step"],
                r["candidate"],
                r["arm"],
                r["lr"],
                r["seed"],
                r["attempt"],
                r["state"],
                r["unusable_reason"],
                r["mmlu_loss_pp"],
                r["factual_loss_pp"],
            )
            for r in u["endpoints"]
        ],
    )
    lines += _table(
        ("stage", "comparison", "excluded blocks"),
        [
            (stage, name, seeds)
            for stage, found in u["exclusions"].items()
            for name, seeds in found.items()
        ],
    )
    worst = u["worst"] or {}
    return lines + _table(
        ("arm", "metric", "largest loss (pp)", "run_id", "seed"),
        [
            (arm, metric, (w or {}).get("loss_pp"), (w or {}).get("run_id"), (w or {}).get("seed"))
            for arm, found in worst.items()
            for metric, w in found.items()
        ],
    )


def _harm_rows(comparisons: Mapping[str, Any]) -> list[tuple]:
    rows = []
    for key, c in comparisons.items():
        if c is None:
            rows.append((key, *(["null"] * 10)))
            continue
        wording = c.get("wording") or {}
        missing = c.get("missing") or {}
        rows.append(
            (
                key,
                c.get("n"),
                c.get("status"),
                wording.get("row"),
                _interval_pp(c.get("primary")),
                _interval_pp(c.get("missing_labels")),
                _interval_pp(c.get("all_missing_harmful")),
                _pp(wording.get("low")),
                _pp(wording.get("high")),
                missing.get("runs"),
                missing.get("labels"),
            )
        )
    return rows


HARM_HEADER = (
    "comparison",
    "n",
    "status",
    "wording",
    "primary (pp)",
    "missing labels (pp)",
    "all missing harmful (pp)",
    "x (pp)",
    "y (pp)",
    "runs missing a label",
    "labels missing",
)


def _render_harm(comparisons: Mapping[str, Any]) -> list[str]:
    lines = _table(HARM_HEADER, _harm_rows(comparisons))
    return lines + _table(
        ("comparison", "missing labels: d_low source (pp)", "missing labels: d_high source (pp)"),
        [
            (
                key,
                _interval_pp((c.get("missing_labels") or {}).get("low")),
                _interval_pp((c.get("missing_labels") or {}).get("high")),
            )
            for key, c in comparisons.items()
            if c is not None
        ],
    )


def _render_analysis(a: Mapping[str, Any]) -> list[str]:
    lines = _pairs(
        {
            "record": a["record"]["path"],
            "sha256": a["record"]["sha256"],
            "freeze": a["freeze"],
            "supersedes": a["supersedes"],
            "L": a["level"],
            "bootstrap": a["bootstrap"],
        }
    )
    lines += _render_harm(a["comparisons"])
    rows = []
    for key, c in a["comparisons"].items():
        unknown, extremes = c.get("worst_case_unknown_blocks"), c.get("observed_extremes")
        if unknown is None and extremes is None:
            continue
        unknown, extremes = unknown or {}, extremes or {}
        smallest, largest = extremes.get("smallest") or {}, extremes.get("largest") or {}
        envelope = unknown.get("envelope")
        rows.append(
            (
                key,
                unknown.get("unknown"),
                _pp(unknown.get("low_mean")),
                _pp(unknown.get("high_mean")),
                _interval_pp(envelope) if envelope is not None else "null",
                _pp(smallest.get("d")),
                smallest.get("seed"),
                _pp(largest.get("d")),
                largest.get("seed"),
            )
        )
    lines += _table(
        (
            "comparison",
            "worst_case_unknown_blocks: unknown",
            "worst_case_unknown_blocks: low mean (pp)",
            "worst_case_unknown_blocks: high mean (pp)",
            "worst_case_unknown_blocks: envelope (pp)",
            "observed_extremes: smallest (pp)",
            "observed_extremes: its seed",
            "observed_extremes: largest (pp)",
            "observed_extremes: its seed",
        ),
        rows,
    )
    utility = a["utility"]
    lines += _pairs({"utility claim": utility["claim"]})
    lines += _table(
        BOUND_HEADER, [row for name, g in utility["gates"].items() for row in _bound_rows(name, g)]
    )
    matched = a["matched_k_harmful"] or {}
    lines += _pairs(
        {
            "K*_h": matched.get("kstar_h"),
            "matched blocks": matched.get("n"),
            "left out": len(matched.get("left_out") or ()),
        }
    )
    lines += _table(
        ("left out", "reason"),
        [(e.get("seed"), e.get("reason") or e.get("arms")) for e in matched.get("left_out") or ()],
    )
    curvature = a["curvature"]
    lines += _pairs(
        {
            "κ": curvature["kappa"],
            "unit": curvature["unit"],
            "source": curvature["source"],
            "fixation": (curvature["fixation"] or {}).get("path"),
        }
    )
    lines += _table(
        ("comparison", "bias (pp)", "largest (pp)", "widths"),
        [
            (key, c.get("bias_pp"), c.get("largest_pp"), c.get("widths"))
            for key, c in (curvature["comparisons"] or {}).items()
        ],
    )
    return lines + _table(
        ("run_id", "q0", "q0.1", "q0.25", "q0.5", "q0.75", "q0.9", "q1"),
        [(run, *values) for run, values in (a["k_spread"] or {}).items()],
    )


def _render_validation_harm(v: Mapping[str, Any]) -> list[str]:
    lines = _pairs(
        {"record": v["record"]["path"], "sha256": v["record"]["sha256"], "analysis": v["analysis"]}
    )
    return lines + _render_harm(v["comparisons"])


def _render_audit(a: Mapping[str, Any]) -> list[str]:
    lines = _pairs(
        {
            "record": a["record"]["path"],
            "sha256": a["record"]["sha256"],
            "selection": a["selection"],
            "missing": a["missing"],
        }
    )
    lines += _table(
        (
            "E",
            "undetermined",
            "MAR B_E (pp)",
            "SE (pp)",
            "worst low (pp)",
            "worst high (pp)",
            "MAR outside the worst case",
            "missing draws",
            "missing items",
        ),
        [
            (
                e,
                b.get("undetermined"),
                _pp((b.get("mar") or {}).get("estimate")),
                _pp((b.get("mar") or {}).get("se")),
                _pp((b.get("worst_case") or [None, None])[0]),
                _pp((b.get("worst_case") or [None, None])[1]),
                a["mar_outside_worst_case"].get(e),
                b.get("missing_draws"),
                b.get("missing_items"),
            )
            for e, b in (a["bias"] or {}).items()
        ],
    )
    lines += _table(("B_E: the MAR estimate and the worst case",), [(a["bias_note"],)])
    rows = []
    for arm, r in (a["error_rates"] or {}).items():
        sens, fpr, counts = r.get("sensitivity") or {}, r.get("fpr") or {}, r.get("counts") or {}
        rows.append(
            (
                arm,
                r.get("draws"),
                r.get("distinct"),
                counts.get("tp"),
                counts.get("fn"),
                counts.get("fp"),
                counts.get("tn"),
                r.get("missing_draws"),
                r.get("harmful_by_reference"),
                sens.get("mar"),
                sens.get("worst_case"),
                fpr.get("mar"),
                fpr.get("worst_case"),
                r.get("undetermined"),
            )
        )
    lines += _table(
        (
            "arm",
            "draws",
            "distinct",
            "tp",
            "fn",
            "fp",
            "tn",
            "missing draws",
            "harmful by reference",
            "sensitivity",
            "sensitivity worst case",
            "FPR",
            "FPR worst case",
            "undetermined",
        ),
        rows,
    )
    return lines + _table(("non-estimable", "why"), (a["non_estimable"] or {}).items())


def _render_labeling(runs: Sequence[str]) -> list[str]:
    return _table(("run with a technical labeling problem",), [(r,) for r in runs])


def _render_freeze(f: Mapping[str, Any]) -> list[str]:
    lines = _pairs(
        {
            "path": f["path"],
            "sha256": f["sha256"],
            "candidate": f["candidate"],
            "confirmation": f["confirmation"],
            "K*_h": f["kstar_h"],
            "matched blocks": f["matched"],
            "left out": len(f["left_out"]),
        }
    )
    rows = [
        (key, c["reading"], c["rule"], len(c["seeds"]), len(c["run_ids"]), c["unknown"])
        for key, c in f["comparisons"].items()
    ]
    rows.append(
        ("A-P", None, None, len(f["a_minus_p"]["seeds"]), len(f["a_minus_p"]["run_ids"]), None)
    )
    rows += [
        (
            f"validation {key}",
            None,
            None,
            None if c is None else len(c["seeds"]),
            None if c is None else len(c["run_ids"]),
            None,
        )
        for key, c in f["validation"].items()
    ]
    return lines + _table(("cohort", "reading", "rule", "blocks", "runs", "unknown blocks"), rows)


def _render_judge_mark(marks: Sequence[Mapping[str, Any]] | None) -> list[str]:
    if marks is None:
        return _table(("judge mark",), [(None,)])
    keys = ("path", "sha256", "batch", "cause", "diagnosis")
    lines = _table(keys, [tuple(j[k] for k in keys) for j in marks])
    return lines + _table(
        ("batch", "evidence", "sha256", "root"),
        [
            (j["batch"], e.get("path"), e.get("sha256"), e.get("root"))
            for j in marks
            for e in j["evidence"]
        ],
    )


def _render_money(m: Mapping[str, Any]) -> list[str]:
    lines = _table(
        ("stage", f"spent ({m['currency']})"), [(s["stage"], s["spent"]) for s in m["by_stage"]]
    )
    lines += _table(
        ("incident", "stage", "incident_time", "cause", f"billed ({m['currency']})", "evidence"),
        [
            (*(i[k] for k in INCIDENT_FIELDS[:-1]), "; ".join(_item(e) for e in i["evidence"]))
            for i in m["incidents"]
        ],
    )
    return lines + _table(
        ("stage stopped by the internal spending policy",), [(s,) for s in m["policy_stops"]]
    )


def _item(item: Mapping[str, Any]) -> str:
    """An evidence item as the table shows it."""
    if "journal" in item:
        return f"{item['journal']} line {item['line']}"
    return item.get("qualification") or f"pod {item.get('pod')}"


def _render_embargo(found: Sequence[Mapping[str, Any]]) -> list[str]:
    return _table(("stage", "what"), [(e["stage"], e["what"]) for e in found])


_RENDER = {
    "outcome": _render_outcome,
    "dry_run": _render_dry_run,
    "engineering": _render_engineering,
    "simulation": _render_simulation,
    "qualification": _render_qualification,
    "not_counted": _render_not_counted,
    "validation": _render_validation,
    "n": _render_n,
    "confirmation": _render_confirmation,
    "utility": _render_utility,
    "analysis": _render_analysis,
    "validation_harm": _render_validation_harm,
    "audit": _render_audit,
    "labeling": _render_labeling,
    "freeze": _render_freeze,
    "judge_mark": _render_judge_mark,
    "money": _render_money,
    "embargo": _render_embargo,
}
