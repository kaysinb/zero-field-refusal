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
  of ``immutable`` included; a record is invalid when one it rests on is.
- ``immutable``: a record whose sha256 differs from the one a started campaign's header named.
- ``write_decision``: validates and writes a record atomically, once; a second version is refused.
- ``cap_of``, ``budget_reason`` and ``budget_needed``: the cap N + max(10, ⌈0.12·N⌉), and why (if
  at all) N asks for a budget decision.

**Kinds**: ``engineering``, ``simulation``, ``n-c<N>`` (one per candidate) and ``budget-c<N>``.
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
  c) and ``passed``, all with the same J, the one frozen number of coverage tests across the
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
  ``answers``, the evidence of the ``n-c<N>`` record itself, at ``study3/decisions/n-c<N>.json``
  under the runs directory and no other file (so its sha256 is the recorded one).
  It is needed when a passed ``n`` record has N above 90 and when N is raised for power (the
  simulation record's ``power_short``), even if N is 90 or less: the extra blocks are the
  researcher's decision; ``decisions`` lists the record as ``missing``, with that as its problem,
  until it is written. A failed ``n`` record needs none. One for an N that needs none is valid
  and read as written.

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
  longer there, is a problem; a missing journal, or one with an unparseable line that is not the
  last of its log, fails closed. A record that resolves (a link, say) under a ``responses/`` or
  ``observations/`` folder is a problem and is never opened. The journal is read as
  ``study3_records`` reads it: through its private ``_read_journal``, which this module keeps
  importing (one journal reader, no copy).
- *Nothing is computed from the evidence*: its content is the writer's; this module binds the
  record to the file's bytes and does not read the evidence's meaning.
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
from .study3_records import SEEDS, _read_journal

DIRECTORY = ("study3", "decisions")
STATES = ("passed", "failed")
SEALED = ("responses", "observations")  # a run's folders no decision rests on

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
COMMON = ("kind", "state", "written_at", "code")
FIELDS = {
    "engineering": ("checks",),
    "simulation": ("fixation", "inputs", "gates", "coverage", "level", "validation_blocks",
                   "verification"),
    "n": ("candidate", "rates", "active", "level", "cohort", "spread", "spread_pp", "N", "cap",
          "simulation"),
    "budget": ("candidate", "N", "cap", "decision", "answers"),
}  # fmt: skip

_KIND = re.compile(r"(engineering|simulation|(?:n|budget)-c([12]))")
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
    """(``engineering``, ``simulation``, ``n`` or ``budget``, the candidate) of a kind;
    ``ValueError`` for another name."""
    found = _KIND.fullmatch(kind) if isinstance(kind, str) else None
    if found is None:
        raise ValueError(f"{kind!r} is not a decision kind (engineering, simulation, n-c<N>, "
                         "budget-c<N>, N = 1 or 2)")  # fmt: skip
    return (kind.split("-")[0], int(found.group(2))) if found.group(2) else (kind, None)


def decision_path(runs_root: str | Path, kind: str) -> Path:
    """``RUNS_DIR/study3/decisions/<kind>.json``."""
    kind_of(kind)
    return Path(runs_root).joinpath(*DIRECTORY, f"{kind}.json")


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
    """The record ``kind`` if it is present and valid, and its problems. Absent: (None, []).
    With any problem the record is None: nothing downstream may consume it. ``repo``: the
    checkout that evidence with ``"root": "repo"`` is relative to."""
    path = decision_path(runs_root, kind)
    if not path.exists():
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

    A record rests on the ones before it: the ``n`` entry is ``invalid`` when the simulation entry
    is, and the budget entry when the ``n`` entry is (a budget decision that the ``n`` record
    would have asked for cannot then be told, so it is listed, invalid, rather than dropped). The
    budget entry is listed when a passed ``n`` record asks for a budget decision
    (``budget_reason``), or one is written; a needed one that is not written is ``missing`` with
    that as its problem. A failed ``n`` record (its cap beyond the reserved seeds) needs none."""
    if candidate not in (1, 2) or isinstance(candidate, bool):
        raise ValueError(f"candidate {candidate!r} is not 1 or 2")
    root, repo = Path(runs_root), Path(repo)
    starts, corrupt = _starts(root)
    kind_n, budget = f"n-c{candidate}", f"budget-c{candidate}"
    out = {kind: _entry(root, kind, starts, corrupt, repo)
           for kind in ("engineering", "simulation", kind_n)}  # fmt: skip
    simulation, n = out["simulation"], out[kind_n]
    if simulation["state"] == "invalid" and n["state"] in STATES:
        _invalidate(n, f"the simulation record it rests on is invalid ({simulation['path']})")
    reason = None
    if n["record"] is not None and n["record"]["state"] == "passed":
        reason = budget_reason(n["record"]["N"], simulation["record"])
    if reason is not None or n["state"] == "invalid" or decision_path(root, budget).exists():
        out[budget] = _entry(root, budget, starts, corrupt, repo)
        if n["state"] == "invalid":
            _invalidate(out[budget], f"the n-c{candidate} record it answers is invalid "
                                     f"({n['path']})")  # fmt: skip
        elif reason is not None and out[budget]["state"] == "missing":
            out[budget]["problems"] = [f"{out[budget]['path']}: a budget decision answering the "
                                       f"n-c{candidate} record is needed ({reason}) and is not "
                                       "written"]  # fmt: skip
    return out


def _invalidate(entry: dict[str, Any], why: str) -> None:
    entry.update(state="invalid", record=None,
                 problems=[*entry["problems"], f"{entry['path']}: {why}"])  # fmt: skip


def _entry(root: Path, kind: str, starts: Starts | None, corrupt: Sequence[str],
           repo: Path) -> dict[str, Any]:  # fmt: skip
    path = decision_path(root, kind)
    record, problems = read_decision(root, kind, repo=repo)
    problems = problems + [p for p in immutable(path, starts, corrupt=corrupt) if p not in problems]
    state = "invalid" if problems else "missing" if record is None else record["state"]
    return {"kind": kind, "path": str(path), "state": state,
            "record": None if problems else record, "problems": problems}  # fmt: skip


def _starts(root: Path) -> tuple[Starts | None, list[str]]:
    """The launch journal's study-3 starts and its corrupt lines (``study3_records``' reading of
    it); (None, []) when there is no journal or it cannot be read."""
    if not any((root / "launch").glob("*.jsonl")):
        return None, []
    try:
        journal = _read_journal(root, {})
    except (OSError, ValueError):
        return None, []
    return journal.starts, journal.corrupt


def immutable(
    record_path: str | Path, journal: Starts | None, *, corrupt: Sequence[str] = ()
) -> list[str]:
    """Problems with a record that a started campaign named and that has since changed: the
    campaign's header named the record (module docstring) with a sha256 that is not the file's
    now, or with none that can be read, or the file is gone. A record no campaign named is free
    of problems. ``journal``: the launch journal's starts, ``(campaign, start record)``; None
    (a missing or unreadable journal) fails closed for a record that exists, and so do
    ``corrupt`` lines (``"<log> line <n>"``, the journal's unparseable lines before the last of
    their log): a campaign that named the record may be what they lost."""
    path = Path(record_path)
    if path.exists() and _under_sealed(path, _runs_of(path)):
        return [_sealed_problem(path)]
    if journal is None:
        return ([f"{path}: the launch journal is missing or unreadable, so whether a campaign "
                 "has named this record cannot be shown"] if path.exists() else [])  # fmt: skip
    problems = set()
    if corrupt and path.exists():
        problems.add(f"{path}: the launch journal has an unparseable line that is not the last "
                     f"of its log ({', '.join(corrupt)}), so whether a campaign has named this "
                     "record cannot be shown")  # fmt: skip
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
    if problems:
        raise DecisionRefused(f"{kind}: not written, {len(problems)} problem(s): "
                              + "; ".join(problems))  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    _publish(path, data)
    return path


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
    return problems


class _Schema:
    """The closed schema of one kind: ``problems`` and the evidence ``refs`` of a record."""

    def __init__(self, kind: str, family: str, candidate: int | None) -> None:
        self.kind, self.family, self.candidate = kind, family, candidate
        self.problems: list[str] = []
        self.refs: list[_Ref] = []

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
        top = self.object("", record, (*COMMON, *FIELDS[self.family]))
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
        if self.real("spread_pp", value) and not (
                0 <= value <= 100 and (value / SPREAD_STEP_PP).is_integer()):  # fmt: skip
            self.add("spread_pp", f"is {value!r}: the spread is a multiple of {SPREAD_STEP_PP} pp "
                     "between 0 and 100, and nothing finer")  # fmt: skip

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
        self._exact_ref(
            "answers", r["answers"], "/".join((*DIRECTORY, f"n-c{self.candidate}.json"))
        )


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
            problems.append(f"{ref.where}: {path} is under a responses or observations folder: "
                            "never evidence, never opened")  # fmt: skip
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


def _against_n(root: Path, kind: str, record: Mapping[str, Any], candidate: int,
               repo: Path) -> list[str]:  # fmt: skip
    """A ``budget-c<N>`` record against the ``n-c<N>`` record it answers."""
    n, problems = read_decision(root, f"n-c{candidate}", repo=repo)
    if n is None:
        return [f"the n-c{candidate} record it answers is "
                + ("missing" if not problems else f"not valid ({problems[0]})")]  # fmt: skip
    found = [f"{name} {record[name]} is not the n-c{candidate} record's {n[name]}"
             for name in ("N", "cap") if record[name] != n[name]]  # fmt: skip
    return found


# ------------------------------------------------------------------ small helpers


def _under_sealed(path: Path, base: Path, named: Path | None = None) -> bool:
    """Whether ``path`` (as written, ``named``, or where its links lead) is under a ``responses``
    or ``observations`` folder, below ``base`` (a folder of the base itself does not count).
    Decided from the names alone: nothing is opened."""
    resolved, root = path.resolve(), base.resolve()
    below = resolved.relative_to(root) if resolved.is_relative_to(root) else resolved
    return any(part in SEALED for part in (*(named.parts if named else ()), *below.parts))


def _runs_of(record_path: Path) -> Path:
    """The runs directory of a record at ``RUNS_DIR/study3/decisions/<kind>.json``."""
    parents = record_path.parents
    return parents[2] if len(parents) > 2 else Path(record_path.anchor or ".")


def _sealed_problem(path: Path) -> str:
    return f"{path}: resolves under a responses or observations folder: never opened"


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
