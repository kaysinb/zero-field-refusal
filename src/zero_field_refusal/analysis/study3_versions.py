"""Study 3's versions record, its staged fixation and the journal check.

docs/study3-preregistration.md: "Seeds and order" (the versions record pins one commit for the
validation and the confirmation, as in study 2; the check fails closed: a run without its start in
the launch journal does not count until the journal is restored; harm labels never enter a
selection: after the fixation there are no new candidates, rates or K*), "Simulation before
validation" (L and the validation size, recorded before validation), "Validation", "Sample size
from validation" (the candidate, the rates, the numerical policy and the judge sealed; N, the cap
and the decision on the money), "Confirmation" and "Voiding and supersession" (a validation or
confirmation run under a commit other than the pinned one does not count). Study 2's versions
record (``study2_short.versions_problems`` and ``read_versions_record``,
docs/study2-short-versions*.json) is the model. Study 3 fixes in two stages: a record before a
candidate's validation, and before its confirmation a supplement that names the first.

- ``validation_fixation``, ``confirmation_fixation`` and ``versions_record``: the builders. Each
  returns a record from records already written: the decision records through
  ``study3_decisions.read_decision``, the numerical qualification through
  ``qualification_records.read_qualification`` (recomputed from its records), the checkout
  through Git. A built record is checked as its reader checks it, and ``RecordRefused`` names
  every problem. No builder writes a file (``scripts/study3_summary.py`` does).
- ``read_validation_fixation``, ``read_confirmation_fixation`` and ``read_versions``: a record
  and its problems, ``(record | None, problems)`` as ``study3_decisions.read_decision`` gives
  them: an absent record is (None, []); with any problem the record is None.
- ``versions_problems``: how the checkout and the launch journal depart from a versions record;
  none: it holds.
- ``qualification_logs``: the launch logs a candidate's qualification rests on, each with its
  sha256 now; ``registered_policy``: the registration at a commit, as a fixation pins it;
  ``instruments``: the model and the judge as this checkout's configs give them.
- ``fixation_name`` and ``versions_name``: the records' paths in the checkout.

**The records** lie in the checkout (``docs/``). Their schemas are closed: a field the schema does
not name is refused, and they hold numbers, ids, paths and hashes only. Evidence under RUNS_DIR is
named as ``study3_decisions`` names it, ``{"path", "sha256"}`` with the path relative to RUNS_DIR
(``"root": "repo"`` for a path relative to the checkout), and checked against the file as it is
now, byte for byte, whenever the record is read; a file under a ``responses/`` or
``observations/`` folder is refused and never opened. ``written_at`` is ISO 8601 in UTC; ``code``
is ``commit`` (40 hexadecimal digits) and ``gpu_code_sha256`` (``code_identity.gpu_code_identity``
of that commit); ``model`` is the model's ``name`` and ``revision`` and ``judge`` the judge's
``config`` and identity (``judge.IDENTITY_FIELDS``), as study 2's records give them, equal to this
checkout's configs (``experiments/study3`` and ``judge/wildguard``, the frozen judge).

1. **The validation fixation**, ``docs/study3-fixation-validation-c<N>.json``, written once before
   candidate N's validation: ``kind`` (``fixation-validation``) and ``candidate``;
   ``written_at``; ``policy``: the registration (``path``, docs/study3-preregistration.md), the
   ``commit`` it is read at, the file's ``sha256`` at that commit and its ``amendments``, the
   numbers of all its "Amendment <n>" headings there; ``rates``: the selected (or repaired) rates
   per active arm, A, F and, while P–F is active, P, each ``lr_l`` and ``lr_u``;
   ``qualification``: the evidence of the numerical verdict file (``verdict``,
   ``qualification/c<N>/verdict.json``) and of every launch log the qualification rests on
   (``logs``, ``qualification_logs``, in the order of their paths), and the ``states`` of the
   fixed rates (per arm, ``lr_l`` and ``lr_u``), each ``qualified``; ``simulation`` and
   ``engineering``: the evidence of the decision records ``study3/decisions/simulation.json`` and
   ``study3/decisions/engineering.json``; ``validation_blocks`` (28 or 40) and ``level`` (L), the
   simulation record's; ``model``, ``judge`` and ``code``.
2. **The confirmation supplement**, ``docs/study3-fixation-confirmation-c<N>.json``, written once
   before candidate N's confirmation: ``kind`` (``fixation-confirmation``) and ``candidate``;
   ``written_at``; ``validation_fixation``: the first record's evidence (``"root": "repo"``);
   ``n``: the evidence of ``study3/decisions/n-c<N>.json``; ``N`` and ``cap``, the n record's;
   ``budget``: the evidence of ``study3/decisions/budget-c<N>.json`` when
   ``study3_decisions.budget_reason`` says N needs a budget decision (its decision proceed), else
   null; ``code``.
3. **The versions record**, ``docs/study3-versions.json`` and any later one
   ``docs/study3-versions-<k>.json`` (k = 2, 3, …): ``what`` (the fixed text ``WHAT``),
   ``written_at``, ``code`` (its ``commit`` alone), ``model``, ``judge``, ``fixation`` (the
   fixation record its runs are under, ``{"path", "sha256"}`` relative to the checkout, or null),
   ``supersedes`` (``{"path", "sha256", "reason"}``: the record before it; null for the first) and
   ``reason`` (why it is written). A reason is an id (a lowercase letter, then lowercase letters,
   digits or underscores, at most 64). A later record is a new file; nothing is overwritten.

**What each reader checks.**
- The validation fixation: its schema; its evidence now; the policy at its commit, which HEAD
  descends from; that the qualification rests now on exactly the logs it pins; the simulation
  record valid and passed, of the fixed L and size; the engineering record valid and passed; the
  model and the judge; the code: a commit HEAD descends from, whose GPU code is
  ``gpu_code_sha256``, the code in force for the candidate's runs (the chain of engineering cycles
  from the engineering record, ``study3_progress.code_chain``).
- The confirmation supplement: its schema; its evidence now; its candidate's validation fixation
  valid; the n record valid and passed, of its N and cap, of the fixed L, of the fixed rates for
  the arms it names and of the fixation's GPU code; the budget record named exactly when N needs
  one, valid, its decision proceed; its code, checked as the validation fixation's is (so both
  carry the code in force: one GPU code).
- The versions record: its schema and the evidence it names now (its fixation, the record it
  supersedes).

**The journal check** (``versions_problems``), as study 2's, failing closed:
- the record lies in the checkout under its name, Git tracks it, and it is unchanged at HEAD and
  since the commit that added it (a record is never overwritten);
- HEAD is the recorded commit or its record-only child, read as study 2 reads it: a descendant
  whose changes from the recorded commit touch the record alone; no tracked file has an
  uncommitted change, and no file is untracked unless the checkout's ``.gitignore`` ignores it;
- its model and judge are this checkout's configs'; its fixation, when it names one, is committed,
  of the named sha256 now and at the recorded commit, valid (its reader), and of the recorded
  commit's GPU code;
- the succession: the first record names none; every later one names the one before it, by path,
  sha256 and reason; each earlier record is read from Git at the commit that added it, and a
  record added more than once, or changed since, is a problem;
- every validation and confirmation run (a study-3 start of the launch journal whose campaign
  header names that step) started under a campaign whose header names this record or one it
  supersedes (``versions_record``: its path, or a path from another checkout that ends with it),
  with that record's sha256 as it was added (``versions_record_sha256``): a supersession keeps the
  earlier runs valid under the record they started with. A header naming no record, a record
  outside the chain or another sha256 is refused, and so is a study-3 start whose header names no
  study-3 step: whether it is a validation or confirmation run cannot be shown;
- the record each run is under stands for it: a validation run's record names its candidate's
  validation fixation, a confirmation run's its candidate's confirmation supplement; a record whose
  fixation is null stands for none. The fixation of each record a run is under is checked as the
  record's own is;
- a missing launch journal, a log that cannot be read, an unparseable line before the last of its
  log, and a study-3 run directory without a start in the journal that is a validation or
  confirmation run (its ``config.yaml``'s role, else its ``meta.json``'s campaign header), or whose
  role cannot be read, are problems: what the journal does not show cannot be shown.
With ``step`` and ``candidate``, the record must also stand for a run of that step and candidate:
the check before a campaign is written under it.

**The sealing boundary.** Of a run directory only ``config.yaml`` and ``meta.json`` are read, and
only for a directory with no start in the journal; a file that resolves under a ``responses/`` or
``observations/`` folder is never opened. No response text, harm label or judge output is read.

**Rulings** where the registration leaves a detail open:
- *The registration's commit* is, by default, the last commit that changed the registration, which
  the checkout holds as committed; the amendments are named by number, all of them.
- *The logs the qualification rests on* are every launch log that holds one of its lines: a start
  of a ``numerical`` or ``repair`` campaign of the candidate, or a line naming one of those runs (an
  ``end``, ``superseded`` or ``held`` line by run id, a ``superseded`` or ``held`` line by label, a
  recovery round naming a label). They are pinned whole, by sha256: a later line in any of them, or
  a new log naming a run of the qualification, changes the evidence.
- *The reader does not recompute the qualification*; the builder does (``read_qualification``,
  under the code in force). The record is bound to the verdict file and the logs by their bytes,
  and the state machine recomputes the qualification at every step.
- *The two reasons*: ``reason`` says why the record is written (the stage it serves, for example
  ``validation_c1``), ``supersedes.reason`` why the record before it no longer serves.
- *A run's own commit* (``meta.json``'s ``git``) is not compared with the record's commit here: the
  check of the record before each launch stands for it, as in study 2.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from omegaconf import OmegaConf

from .. import code_identity
from ..config import load_config
from ..judge import IDENTITY_FIELDS, judge_identity
from ..paths import REPO_ROOT
from ..tracking import code_changes, git_state, tracked, worktree_changes
from .study3_decisions import (
    CAP_FLOOR,
    DIRECTORY,
    MIN_N,
    SIMULATION_RECORD,
    VALIDATION_BLOCKS,
    _digest,
    _evidence,
    _Ref,
    _Schema,
    _sealed_problem,
    _under_sealed,
    budget_reason,
    cap_of,
    decision_path,
    read_decision,
)
from .study3_records import (
    DECISIONS,
    EXPERIMENT,
    RATE_TOLERANCE,
    STEP_ROLES,
    VERSIONED_STEPS,
    JournalLog,
    _read_journal,
    journal_lines,
)

REGISTRATION = "docs/study3-preregistration.md"
VERSIONS_FIRST = "docs/study3-versions.json"
STAGES = VERSIONED_STEPS  # the steps a fixation stands for: validation and confirmation
KINDS = {"validation": "fixation-validation", "confirmation": "fixation-confirmation"}
MODEL_CONFIG = ("experiments/study3", ("study3.role=validation",))  # the runs' model
JUDGE_CONFIG = "judge/wildguard"  # "The judge is frozen"
# A versions record's fixed text: never changed, since every earlier record is read with it
WHAT = ("The versions of study 3's validation and confirmation runs: the commit, the model, the "
        "judge and the fixation they run under.")  # fmt: skip
QUALIFIED = "qualified"
LABELS = ("lr_l", "lr_u")
# The campaign steps whose runs are qualification cases (qualification_records.STEPS; that module
# loads torch, and is imported only where it is read)
QUALIFICATION_STEPS = ("numerical", "repair")
ENGINEERING_RECORD = "/".join((*DIRECTORY, "engineering.json"))
FIELDS = {
    "validation": ("kind", "candidate", "written_at", "policy", "rates", "qualification",
                   "simulation", "validation_blocks", "level", "engineering", "model", "judge",
                   "code"),
    "confirmation": ("kind", "candidate", "written_at", "validation_fixation", "n", "N", "cap",
                     "budget", "code"),
    "versions": ("what", "written_at", "code", "model", "judge", "fixation", "supersedes",
                 "reason"),
}  # fmt: skip

_VERSIONS_NAME = re.compile(r"docs/study3-versions(?:-([2-9]|[1-9][0-9]+))?\.json")
_FIXATION_NAME = re.compile(r"docs/study3-fixation-(validation|confirmation)-c([12])\.json")
_LOG = re.compile(r"launch/[^/\\]+\.jsonl")
_AMENDMENT = re.compile(r"^### Amendment (\d+)\b", re.MULTILINE)
_ID = re.compile(r"[a-z][a-z0-9_]{0,63}")
_COMMIT, _SHA256 = re.compile(r"[0-9a-f]{40}"), re.compile(r"[0-9a-f]{64}")


class RecordRefused(ValueError):
    """A fixation or versions record that cannot be built: its problems named."""


# ------------------------------------------------------------------ names


def fixation_name(step: str, candidate: int) -> str:
    """``docs/study3-fixation-<step>-c<candidate>.json``, relative to the checkout."""
    _stage(step)
    _candidate(candidate)
    return f"docs/study3-fixation-{step}-c{candidate}.json"


def versions_name(k: int = 1) -> str:
    """The k-th versions record: ``docs/study3-versions.json``, then
    ``docs/study3-versions-<k>.json``, relative to the checkout."""
    if type(k) is not int or k < 1:
        raise ValueError(f"{k!r} is not the number of a versions record (1, 2, …)")
    return VERSIONS_FIRST if k == 1 else f"docs/study3-versions-{k}.json"


def _previous(name: str) -> str | None:
    """The versions record before ``name`` (a versions record's name); None for the first."""
    number = _VERSIONS_NAME.fullmatch(name).group(1)
    return None if number is None else versions_name(int(number) - 1)


def _stage(step: Any) -> None:
    if step not in STAGES:
        raise ValueError(f"{step!r} is not a step a fixation stands for ({list(STAGES)})")


def _candidate(candidate: Any) -> None:
    if type(candidate) is not int or candidate not in (1, 2):
        raise ValueError(f"candidate {candidate!r} is not 1 or 2")


# ------------------------------------------------------------------ the checkout


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=True).stdout


def _blob(repo: Path, commit: str, path: str) -> bytes | None:
    """The bytes of ``path`` at ``commit``; None when the commit does not hold it."""
    try:
        return _git(repo, "show", f"{commit}:{path}")
    except (OSError, subprocess.CalledProcessError):
        return None


def _descends(repo: Path, commit: str) -> bool:
    """Whether HEAD is ``commit`` or descends from it."""
    try:
        _git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _added(repo: Path, path: str) -> tuple[str | None, str | None]:
    """The commit of HEAD's history that added ``path``, or why there is none (never added, or
    added more than once: overwritten)."""
    try:
        commits = _git(repo, "log", "--diff-filter=A", "--format=%H", "--", path).decode().split()
    except (OSError, subprocess.CalledProcessError):
        return None, f"{path}: its history cannot be read"
    if not commits:
        return None, f"{path} is not committed"
    if len(commits) > 1:
        return None, (f"{path} was added {len(commits)} times: a record is never overwritten (a "
                      "later one is a new file)")  # fmt: skip
    return commits[0], None


def _gpu_code(repo: Path, commit: str) -> str | None:
    """The GPU code of ``commit`` (``code_identity.gpu_code_identity``); None when it cannot be
    computed."""
    try:
        return code_identity.gpu_code_identity(commit, repo)["sha256"]
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def _in_checkout(path: str | Path, repo: Path) -> str | None:
    """A path relative to the checkout (a relative path is read from it); None outside it."""
    path = Path(path)
    resolved = (path if path.is_absolute() else repo / path).resolve()
    root = repo.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else None


def registered_policy(repo: str | Path = REPO_ROOT, commit: str | None = None) -> dict[str, Any]:
    """The registration as a validation fixation pins it: its ``path``, the ``commit`` it is read
    at (by default the last commit that changed it, the checkout holding it as committed), its
    ``sha256`` there and the numbers of its ``amendments``. ``ValueError`` otherwise."""
    repo = Path(repo)
    if commit is None:
        try:
            changes = worktree_changes(repo)
            commit = _git(repo, "log", "-1", "--format=%H", "--", REGISTRATION).decode().strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"{repo} is no Git checkout of {REGISTRATION}") from exc
        if REGISTRATION in (*changes["uncommitted"], *changes["untracked"]):
            raise ValueError(f"{REGISTRATION} has uncommitted changes: the policy is the "
                             "registration as committed")  # fmt: skip
        if not commit:
            raise ValueError(f"{REGISTRATION} is not committed")
    blob = _blob(repo, commit, REGISTRATION)
    if blob is None:
        raise ValueError(f"{REGISTRATION} cannot be read at commit {commit}")
    resolved = _git(repo, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    return {"path": REGISTRATION, "commit": resolved, "sha256": hashlib.sha256(blob).hexdigest(),
            "amendments": _amendments(blob)}  # fmt: skip


def _amendments(blob: bytes) -> list[int]:
    """The numbers of the registration's "Amendment <n>" headings, in the order it holds them."""
    return [int(n) for n in _AMENDMENT.findall(blob.decode("utf-8", errors="replace"))]


@functools.cache
def _configured() -> str:
    cfg = load_config(MODEL_CONFIG[0], MODEL_CONFIG[1])
    model = {"name": str(cfg.model.name), "revision": str(cfg.model.revision)}
    judge = {"config": JUDGE_CONFIG, **judge_identity(load_config(JUDGE_CONFIG))}
    return json.dumps({"model": model, "judge": judge})


def instruments() -> tuple[dict[str, Any], dict[str, Any]]:
    """The model (``name``, ``revision``) and the judge (``config`` and its identity) as this
    checkout's configs give them: the runs' model and the frozen judge."""
    found = json.loads(_configured())
    return found["model"], found["judge"]


# ------------------------------------------------------------------ the qualification's logs


def qualification_logs(
    runs_root: str | Path, candidate: int
) -> tuple[list[dict[str, str]], list[str]]:
    """The launch logs a candidate's numerical qualification rests on (module docstring,
    "Rulings"), in the order of their paths, each ``{"path", "sha256"}`` with its path under
    RUNS_DIR and its sha256 now; and the problems: a log that cannot be read, or an unparseable
    line before the last of its log (it may be a line of the qualification)."""
    _candidate(candidate)
    logs, unreadable = journal_lines(runs_root)
    problems = [f"{problem}: it may hold a line of the qualification" for problem in unreadable]
    ids, labels = set(), set()
    for log in logs:
        for line in log.lines:
            record = line.record
            if record is not None and _qualification_start(record, candidate):
                if isinstance(record.get("run_id"), str):
                    ids.add(record["run_id"])
                key = _label(record)
                if key is not None:
                    labels.add(key)
    pinned = []
    for log in logs:
        problems += [f"{log.path} line {number}: not valid JSON, and not the last line of its "
                     "log: it may be a line of the qualification"
                     for number in _corrupt(log)]  # fmt: skip
        if any(line.record is not None and _of_qualification(line.record, candidate, ids, labels)
               for line in log.lines):  # fmt: skip
            pinned.append({"path": log.path, "sha256": log.sha256})
    return pinned, problems


def _qualification_start(record: Mapping[str, Any], candidate: int) -> bool:
    """A start of the candidate's qualification (``qualification_records``' issued runs)."""
    head = record.get("header")
    return (record.get("event") == "start" and isinstance(head, Mapping) and head.get("study") == 3
            and type(head.get("candidate")) is int and head.get("candidate") == candidate
            and head.get("step") in QUALIFICATION_STEPS)  # fmt: skip


def _label(record: Mapping[str, Any]) -> tuple[str, str] | None:
    key = (record.get("run"), record.get("experiment"))
    return key if all(isinstance(part, str) for part in key) else None


def _of_qualification(record: Mapping[str, Any], candidate: int, ids: set[str],
                      labels: set[tuple[str, str]]) -> bool:  # fmt: skip
    """Whether a journal line is one of the qualification's: a start of it, a line naming one of
    its runs by run id or by label, or a recovery round naming one of its labels."""
    event, run_id = record.get("event"), record.get("run_id")
    if event == "start":
        return _qualification_start(record, candidate)
    if event in ("end", "superseded", "held") and isinstance(run_id, str) and run_id in ids:
        return True
    if event in ("superseded", "held") and _label(record) in labels:
        return True
    runs, experiment = record.get("runs"), record.get("experiment")
    return (event == "recovery" and isinstance(runs, list) and isinstance(experiment, str)
            and any(isinstance(label, str) and (label, experiment) in labels
                    for label in runs))  # fmt: skip


def _corrupt(log: JournalLog) -> list[int]:
    """The numbers of a log's unparseable lines before its last (a last line cut short by a
    crash is skipped, the launcher's rule)."""
    last = max((line.line for line in log.lines if line.text.strip()), default=0)
    out = []
    for line in log.lines:
        if line.text.strip() and line.line != last:
            try:
                json.loads(line.text)
            except ValueError:
                out.append(line.line)
    return out


# ------------------------------------------------------------------ the schemas


class _Check(_Schema):
    """The closed schemas of the fixation and versions records, built on the blocks of
    ``study3_decisions``' schemas (one vocabulary of problems); ``refs`` is the evidence to check
    against the files."""

    def __init__(self, kind: str, candidate: int | None = None) -> None:
        super().__init__(kind, kind, candidate)

    def pattern(self, where: str, value: Any, pattern: re.Pattern, what: str) -> bool:
        ok = isinstance(value, str) and pattern.fullmatch(value) is not None
        if not ok:
            self.add(where, f"is {value!r}, not {what}")
        return ok

    def ident(self, where: str, value: Any) -> None:
        self.pattern(where, value, _ID, "an id (a lowercase letter, then lowercase letters, digits "
                     "or underscores, at most 64)")  # fmt: skip

    def repo_ref(self, where: str, value: Any, extra: Sequence[str] = ()) -> dict[str, Any] | None:
        """Evidence relative to the checkout, ``{"path", "sha256"}`` (and ``extra``)."""
        entry = self.object(where, value, ("path", "sha256", *extra))
        if entry is None:
            return None
        ok = self.text(f"{where}.path", entry["path"])
        ok = self.pattern(f"{where}.sha256", entry["sha256"], _SHA256, "a sha256") and ok
        if ok:
            self.refs.append(_Ref(where, entry["path"], entry["sha256"], "repo"))
        return entry

    def code(self, value: Any, keys: Sequence[str] = ("commit", "gpu_code_sha256")) -> None:
        code = self.object("code", value, keys)
        if code is None:
            return
        self.pattern("code.commit", code["commit"], _COMMIT, "a commit (40 hexadecimal digits)")
        if "gpu_code_sha256" in keys:
            self.pattern("code.gpu_code_sha256", code["gpu_code_sha256"], _SHA256, "a sha256")

    def instruments(self, top: Mapping[str, Any]) -> None:
        model = self.object("model", top["model"], ("name", "revision"))
        for key in ("name", "revision") if model is not None else ():
            self.text(f"model.{key}", model[key])
        judge = self.object("judge", top["judge"], ("config", *IDENTITY_FIELDS))
        if judge is not None:
            self.text("judge.config", judge["config"])
            for key in IDENTITY_FIELDS:
                if not (judge[key] is None or isinstance(judge[key], str)):
                    self.add(f"judge.{key}", f"is {judge[key]!r}, not a string or null")

    def common(self, top: Mapping[str, Any]) -> None:
        if top["kind"] != self.kind:
            self.add("kind", f"is {top['kind']!r}, not {self.kind!r}")
        if type(top["candidate"]) is not int or top["candidate"] != self.candidate:
            self.add("candidate", f"is {top['candidate']!r}, not {self.candidate}")
        self._when(top["written_at"])

    def rates(self, value: Any) -> tuple[str, ...] | None:
        """The fixed rates, per active arm (A and F, and P while P–F is active); their arms, or
        None when they are not well formed."""
        before = len(self.problems)
        with_p = isinstance(value, dict) and "P" in value
        self._rates(value, with_p)
        return (("A", "F", "P") if with_p else ("A", "F")) if len(self.problems) == before else None

    # -- the validation fixation

    def validation(self, record: Any) -> None:
        top = self.object("", record, FIELDS["validation"])
        if top is None:
            return
        self.common(top)
        self.policy(top["policy"])
        self.qualification(top["qualification"], self.rates(top["rates"]))
        self._exact_ref("simulation", top["simulation"], SIMULATION_RECORD)
        self._exact_ref("engineering", top["engineering"], ENGINEERING_RECORD)
        blocks = top["validation_blocks"]
        if not (type(blocks) is int and blocks in VALIDATION_BLOCKS):
            self.add("validation_blocks", f"is {blocks!r}, not one of {list(VALIDATION_BLOCKS)}")
        self.level("level", top["level"])
        self.instruments(top)
        self.code(top["code"])

    def policy(self, value: Any) -> None:
        policy = self.object("policy", value, ("path", "commit", "sha256", "amendments"))
        if policy is None:
            return
        if policy["path"] != REGISTRATION:
            self.add("policy.path", f"is {policy['path']!r}, not {REGISTRATION!r}")
        self.pattern("policy.commit", policy["commit"], _COMMIT, "a commit (40 hexadecimal digits)")
        self.pattern("policy.sha256", policy["sha256"], _SHA256, "a sha256")
        amendments = policy["amendments"]
        if not (isinstance(amendments, list)
                and all(type(a) is int and a >= 1 for a in amendments)
                and amendments == sorted(set(amendments))):  # fmt: skip
            self.add("policy.amendments", f"is {amendments!r}, not the amendments' numbers in "
                     "ascending order, each once")  # fmt: skip

    def qualification(self, value: Any, arms: tuple[str, ...] | None) -> None:
        found = self.object("qualification", value, ("verdict", "logs", "states"))
        if found is None:
            return
        verdict = f"qualification/c{self.candidate}/verdict.json"
        self._exact_ref("qualification.verdict", found["verdict"], verdict)
        logs = found["logs"]
        if not isinstance(logs, list) or not logs:
            self.add("qualification.logs", "is not a non-empty list of launch logs")
        else:
            paths = []
            for i, entry in enumerate(logs):
                where = f"qualification.logs[{i}]"
                entry = self.ref(where, entry)
                if entry is None:
                    continue
                path = entry["path"]
                if entry.get("root", "runs") != "runs" or not (
                        isinstance(path, str) and _LOG.fullmatch(path)):  # fmt: skip
                    self.add(where, f"is {path!r}, not a launch log launch/<log>.jsonl under the "
                             "runs directory")  # fmt: skip
                paths.append(path)
            if all(isinstance(p, str) for p in paths) and paths != sorted(set(paths)):
                self.add("qualification.logs", "is not in the order of its paths, each once")
        if arms is None:
            return  # the rates' problems are named
        states = self.object("qualification.states", found["states"], arms)
        for arm in arms if states is not None else ():
            pair = self.object(f"qualification.states.{arm}", states[arm], LABELS)
            for label in LABELS if pair is not None else ():
                if pair[label] != QUALIFIED:
                    self.add(f"qualification.states.{arm}.{label}", f"is {pair[label]!r}, not "
                             f"{QUALIFIED!r}: a validation runs qualified rates only")  # fmt: skip

    # -- the confirmation supplement

    def confirmation(self, record: Any) -> None:
        top = self.object("", record, FIELDS["confirmation"])
        if top is None:
            return
        self.common(top)
        name = fixation_name("validation", self.candidate)
        entry = self.ref("validation_fixation", top["validation_fixation"])
        if entry is not None and (entry["path"] != name or entry.get("root", "runs") != "repo"):
            self.add(
                "validation_fixation",
                f"is {entry['path']!r} under "
                f"{entry.get('root', 'runs')!r}, not {name!r} under 'repo', the checkout",
            )
        self._exact_ref("n", top["n"], "/".join((*DIRECTORY, f"n-c{self.candidate}.json")))
        if self.integer("N", top["N"], MIN_N):
            cap = cap_of(top["N"])
            if top["cap"] != cap or type(top["cap"]) is not int:
                self.add("cap", f"is {top['cap']!r}, not N + max({CAP_FLOOR}, ⌈0.12·N⌉) = {cap}")
        else:
            self.integer("cap", top["cap"], MIN_N + CAP_FLOOR)
        if top["budget"] is not None:
            budget = "/".join((*DIRECTORY, f"budget-c{self.candidate}.json"))
            self._exact_ref("budget", top["budget"], budget)
        self.code(top["code"])

    # -- the versions record

    def versions(self, record: Any) -> None:
        top = self.object("", record, FIELDS["versions"])
        if top is None:
            return
        if top["what"] != WHAT:
            self.add("what", f"is {top['what']!r}, not the record's fixed text {WHAT!r}")
        self._when(top["written_at"])
        self.code(top["code"], ("commit",))
        self.instruments(top)
        if top["fixation"] is not None:
            entry = self.repo_ref("fixation", top["fixation"])
            if entry is not None and not (isinstance(entry["path"], str)
                                          and _FIXATION_NAME.fullmatch(entry["path"])):  # fmt: skip
                self.add("fixation.path", f"is {entry['path']!r}, not a fixation record "
                         "docs/study3-fixation-<validation|confirmation>-c<N>.json")  # fmt: skip
        if top["supersedes"] is not None:
            entry = self.repo_ref("supersedes", top["supersedes"], ("reason",))
            if entry is not None:
                if not (isinstance(entry["path"], str) and _VERSIONS_NAME.fullmatch(entry["path"])):
                    self.add("supersedes.path", f"is {entry['path']!r}, not a versions record")
                self.ident("supersedes.reason", entry["reason"])
        self.ident("reason", top["reason"])


# ------------------------------------------------------------------ reading the fixations


def read_validation_fixation(
    candidate: int, *, runs_root: str | Path, repo: str | Path = REPO_ROOT
) -> tuple[dict[str, Any] | None, list[str]]:
    """Candidate N's validation fixation (``docs/study3-fixation-validation-c<N>.json`` in the
    checkout ``repo``) if it is present and valid, and its problems; absent: (None, []). Its
    evidence under RUNS_DIR is read from ``runs_root``."""
    return _read_fixation("validation", candidate, Path(runs_root), Path(repo))


def read_confirmation_fixation(
    candidate: int, *, runs_root: str | Path, repo: str | Path = REPO_ROOT
) -> tuple[dict[str, Any] | None, list[str]]:
    """Candidate N's confirmation supplement (``docs/study3-fixation-confirmation-c<N>.json``),
    as ``read_validation_fixation`` reads the first."""
    return _read_fixation("confirmation", candidate, Path(runs_root), Path(repo))


def _read_fixation(step: str, candidate: int, root: Path,
                   repo: Path) -> tuple[dict[str, Any] | None, list[str]]:  # fmt: skip
    path = repo / fixation_name(step, candidate)
    if not (path.exists() or path.is_symlink()):
        return None, []
    if _under_sealed(path, repo):
        return None, [_sealed_problem(path)]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, [f"{path}: unreadable ({type(exc).__name__})"]
    check = _validation_problems if step == "validation" else _confirmation_problems
    problems = check(record, candidate, root, repo)
    return (None, [f"{path}: {p}" for p in problems]) if problems else (record, [])


def _validation_problems(record: Any, candidate: int, root: Path, repo: Path) -> list[str]:
    """Every problem of a validation fixation (module docstring, "What each reader checks")."""
    check = _Check(KINDS["validation"], candidate)
    check.validation(record)
    if check.problems:
        return check.problems  # nothing is read from a record that is not well formed
    problems = _evidence(check.refs, root, repo)
    problems += _policy_problems(record["policy"], repo)
    problems += _logs_problems(record["qualification"]["logs"], root, candidate)
    simulation, found = read_decision(root, "simulation", repo=repo)
    problems += [f"simulation: {p}" for p in _passed("simulation", simulation, found)]
    if simulation is not None and simulation["state"] == "passed":
        if record["validation_blocks"] != simulation["validation_blocks"]:
            problems.append(f"validation_blocks: is {record['validation_blocks']}, not the "
                            f"simulation record's {simulation['validation_blocks']}")  # fmt: skip
        if not math.isclose(record["level"], simulation["level"], rel_tol=1e-12):
            problems.append(f"level: is {record['level']}, not the simulation record's L "
                            f"{simulation['level']}")  # fmt: skip
    engineering, found = read_decision(root, "engineering", repo=repo)
    problems += [f"engineering: {p}" for p in _passed("engineering", engineering, found)]
    problems += _instrument_problems(record)
    problems += _code_problems(record["code"], candidate, root, repo, engineering)
    return problems


def _passed(kind: str, record: Mapping[str, Any] | None, problems: Sequence[str]) -> list[str]:
    """A decision record that is not there, not valid or not passed."""
    if record is None:
        return [f"the {kind} record is " + ("missing" if not problems else
                                            f"not valid ({problems[0]})")]  # fmt: skip
    return [] if record["state"] == "passed" else [f"the {kind} record did not pass"]


def _policy_problems(policy: Mapping[str, Any], repo: Path) -> list[str]:
    """The policy against the registration at its commit."""
    commit = policy["commit"]
    blob = _blob(repo, commit, REGISTRATION)
    if blob is None:
        return [f"policy: {REGISTRATION} cannot be read at commit {commit}"]
    found = []
    if not _descends(repo, commit):
        found.append(f"policy: the commit {commit} is not in the history of HEAD")
    digest = hashlib.sha256(blob).hexdigest()
    if digest != policy["sha256"]:
        found.append(f"policy: {REGISTRATION} has sha256 {digest} at commit {commit}, not the "
                     f"recorded {policy['sha256']}")  # fmt: skip
    amendments = _amendments(blob)
    if policy["amendments"] != amendments:
        found.append(f"policy: the amendments are {policy['amendments']}, but the registration at "
                     f"commit {commit} has {amendments}")  # fmt: skip
    return found


def _logs_problems(pinned: Sequence[Mapping[str, Any]], root: Path, candidate: int) -> list[str]:
    """The logs the qualification rests on now against the ones pinned."""
    logs, problems = qualification_logs(root, candidate)
    found = [f"qualification.logs: {p}" for p in problems]
    now, named = {e["path"] for e in logs}, {e["path"] for e in pinned}
    if now - named:
        found.append(f"qualification.logs: the qualification rests now on {sorted(now - named)}, "
                     "which the record does not pin")  # fmt: skip
    if named - now:
        found.append(f"qualification.logs: the record pins {sorted(named - now)}, which the "
                     "qualification no longer rests on")  # fmt: skip
    return found


def _instrument_problems(record: Mapping[str, Any]) -> list[str]:
    """The model and the judge against this checkout's configs."""
    model, judge = instruments()
    found = []
    if record["model"] != model:
        found.append(f"model: is {record['model']}, not the configs' {model}")
    if record["judge"] != judge:
        differ = sorted(key for key in judge if record["judge"].get(key) != judge[key])
        found.append(f"judge: differs from the identity of {JUDGE_CONFIG} in {differ}")
    return found


def _code_problems(code: Mapping[str, Any], candidate: int, root: Path, repo: Path,
                   engineering: Mapping[str, Any] | None) -> list[str]:  # fmt: skip
    """A fixation's code: a commit HEAD descends from, whose GPU code is the recorded one, the
    code in force for the candidate's runs."""
    commit, gpu = code["commit"], code["gpu_code_sha256"]
    found = []
    if not _descends(repo, commit):
        found.append(f"code: the commit {commit} is not in the history of HEAD")
    else:
        identity = _gpu_code(repo, commit)
        if identity != gpu:
            found.append(f"code: the GPU code of commit {commit} is {identity}, not the recorded "
                         f"{gpu}")  # fmt: skip
    if engineering is None:
        return found  # the engineering record's absence is named: no chain to read
    in_force, breaks = _in_force(root, candidate, engineering)
    found += [f"code: {b}" for b in breaks]
    if in_force != gpu:
        found.append(f"code: the GPU code {gpu} is not the one in force for candidate "
                     f"{candidate}'s runs ({in_force}), the chain of engineering cycles from the "
                     "engineering record's")  # fmt: skip
    return found


def _in_force(root: Path, candidate: int,
              engineering: Mapping[str, Any]) -> tuple[str | None, list[str]]:  # fmt: skip
    """The GPU code in force for the candidate's runs and the breaks of the chain up to it
    (``study3_progress.code_chain``, from the engineering record's code through the supersession
    records of the candidate and of those before it: a later candidate's cycle leaves it as it
    is)."""
    from . import qualification_records  # it loads torch (qualification_verdicts): here only
    from .study3_progress import code_chain

    cycles = {c: qualification_records.supersessions(root, c) for c in (1, 2) if c <= candidate}
    _, in_force, breaks = code_chain(engineering["code"]["gpu_code_sha256"], cycles)
    return in_force.get(candidate), breaks


def _confirmation_problems(record: Any, candidate: int, root: Path, repo: Path) -> list[str]:
    """Every problem of a confirmation supplement (module docstring, "What each reader
    checks")."""
    check = _Check(KINDS["confirmation"], candidate)
    check.confirmation(record)
    if check.problems:
        return check.problems
    problems = _evidence(check.refs, root, repo)
    fixation, found = read_validation_fixation(candidate, runs_root=root, repo=repo)
    if fixation is None:
        problems.append(_no_fixation(candidate, found))
    kind = f"n-c{candidate}"
    n, found = read_decision(root, kind, repo=repo)
    problems += [f"n: {p}" for p in _passed(kind, n, found)]
    if n is not None:
        problems += [f"{name} {record[name]} is not the {kind} record's {n[name]}"
                     for name in ("N", "cap") if record[name] != n[name]]  # fmt: skip
        if fixation is not None:
            problems += _against_fixation(n, fixation, kind)
        simulation, _ = read_decision(root, "simulation", repo=repo)
        problems += _budget_problems(record, n, simulation, root, repo, candidate)
    # the supplement's code is in force for the candidate, as the valid validation fixation's is:
    # both are one GPU code
    engineering, _ = read_decision(root, "engineering", repo=repo)
    problems += _code_problems(record["code"], candidate, root, repo, engineering)
    return problems


def _no_fixation(candidate: int, problems: Sequence[str]) -> str:
    name = fixation_name("validation", candidate)
    return f"validation_fixation: the validation fixation {name} is " + (
        "missing" if not problems else f"not valid ({problems[0]})")  # fmt: skip


def _against_fixation(n: Mapping[str, Any], fixation: Mapping[str, Any], kind: str) -> list[str]:
    """The n record of the fixed L, rates (for the arms it names) and GPU code."""
    found = []
    if not math.isclose(n["level"], fixation["level"], rel_tol=1e-12):
        found.append(f"n: the {kind} record's L {n['level']} is not the fixed {fixation['level']}")
    for arm, pair in n["rates"].items():
        fixed = fixation["rates"].get(arm)
        if fixed is None or any(not math.isclose(pair[k], fixed[k], rel_tol=RATE_TOLERANCE)
                                for k in LABELS):  # fmt: skip
            found.append(f"n: the {kind} record's rates of {arm} {pair} are not the fixed {fixed}")
    gpu = n["code"]["gpu_code_sha256"]
    if gpu != fixation["code"]["gpu_code_sha256"]:
        found.append(f"n: the {kind} record's GPU code {gpu} is not the validation fixation's "
                     f"{fixation['code']['gpu_code_sha256']}")  # fmt: skip
    return found


def _budget_problems(record: Mapping[str, Any], n: Mapping[str, Any],
                     simulation: Mapping[str, Any] | None, root: Path, repo: Path,
                     candidate: int) -> list[str]:  # fmt: skip
    """The budget record, named exactly when N needs one (``budget_reason``), its decision
    proceed."""
    kind = f"budget-c{candidate}"
    reason = budget_reason(n["N"], simulation) if n["state"] == "passed" else None
    if reason is None:
        return (
            []
            if record["budget"] is None
            else [f"budget: is named, but N = {n['N']} needs no budget decision: it is null"]
        )
    if record["budget"] is None:
        return [f"budget: is null, but a budget decision is needed ({reason}): the {kind} record"]
    budget, found = read_decision(root, kind, repo=repo)
    if budget is None:
        return [f"budget: {p}" for p in _passed(kind, budget, found)]
    if budget["decision"] != "proceed":
        return [f"budget: the {kind} record decides {budget['decision']}: the confirmation does "
                "not start"]  # fmt: skip
    return []


# ------------------------------------------------------------------ the builders


def validation_fixation(
    runs_root: str | Path,
    candidate: int,
    *,
    rates: Mapping[str, Mapping[str, float]],
    code: Mapping[str, str],
    repo: str | Path = REPO_ROOT,
    settings: Mapping[str, Any] | None = None,
    policy_commit: str | None = None,
    written_at: str | None = None,
) -> dict[str, Any]:
    """Candidate N's validation fixation (module docstring) from the records under ``runs_root``
    and the checkout ``repo``. ``rates``: the selected (or repaired) rates per active arm, as the
    search and repair state gives them; ``code``: the ``commit`` it is written at and that commit's
    GPU code (``gpu_code_sha256``), the code in force for the candidate's runs, under which the
    qualification is recomputed (``read_qualification``; ``settings``:
    configs/experiments/qualification.yaml as loaded, by default this checkout's);
    ``policy_commit``: the registration's commit (``registered_policy``). ``RecordRefused`` while
    a fixed rate is not qualified, the simulation or engineering record is not there, not valid
    or not passed, or for any problem its reader would name; and once the record is written."""
    _candidate(candidate)
    root, repo = Path(runs_root), Path(repo)
    _not_written(repo / fixation_name("validation", candidate))
    simulation, found = read_decision(root, "simulation", repo=repo)
    problems = _passed("simulation", simulation, found)
    engineering, found = read_decision(root, "engineering", repo=repo)
    problems += _passed("engineering", engineering, found)
    check = _Check(KINDS["validation"], candidate)
    arms = check.rates(rates)
    check.code(code)
    problems += check.problems
    policy = None
    try:
        policy = registered_policy(repo, policy_commit)
    except ValueError as exc:
        problems.append(f"policy: {exc}")
    logs, found = qualification_logs(root, candidate)
    problems += [f"qualification.logs: {p}" for p in found]
    if not logs:
        problems.append(f"qualification.logs: no launch log holds candidate {candidate}'s "
                        "qualification")  # fmt: skip
    verdict = root / "qualification" / f"c{candidate}" / "verdict.json"
    if not verdict.is_file():
        problems.append(f"qualification.verdict: {verdict} is missing")
    states: dict[str, dict[str, Any]] = {}
    if arms is not None and not check.problems:
        states, found = _qualified(root, candidate, rates, arms, code["gpu_code_sha256"], settings)
        problems += found
    if problems:
        raise RecordRefused(_refusal(KINDS["validation"], candidate, problems))
    model, judge = instruments()
    record = {
        "kind": KINDS["validation"], "candidate": candidate, "written_at": written_at or _now(),
        "policy": policy,
        "rates": {arm: {label: float(rates[arm][label]) for label in LABELS} for arm in arms},
        "qualification": {"verdict": {"path": verdict.relative_to(root).as_posix(),
                                      "sha256": _digest(verdict)},
                          "logs": logs, "states": states},
        "simulation": {"path": SIMULATION_RECORD,
                       "sha256": _digest(decision_path(root, "simulation"))},
        "validation_blocks": simulation["validation_blocks"], "level": simulation["level"],
        "engineering": {"path": ENGINEERING_RECORD,
                        "sha256": _digest(decision_path(root, "engineering"))},
        "model": model, "judge": judge,
        "code": {"commit": code["commit"], "gpu_code_sha256": code["gpu_code_sha256"]},
    }  # fmt: skip
    record = _plain(record, KINDS["validation"], candidate)
    problems = _validation_problems(record, candidate, root, repo)
    if problems:
        raise RecordRefused(_refusal(KINDS["validation"], candidate, problems))
    return record


def _qualified(
    root: Path,
    candidate: int,
    rates: Mapping[str, Mapping[str, float]],
    arms: Sequence[str],
    expected_code: str,
    settings: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The fixed rates' states, recomputed from the qualification's records under the code in
    force (``read_qualification``), and why a rate is not qualified (a reading with a problem
    decides nothing)."""
    from . import qualification_records  # it loads torch (qualification_verdicts): here only

    if settings is None:
        settings = load_config("experiments/qualification")
    try:
        reading = qualification_records.read_qualification(
            root, candidate, expected_code=expected_code, settings=settings)  # fmt: skip
    except (OSError, ValueError) as exc:
        return {}, [f"qualification: it cannot be read ({type(exc).__name__}: {exc})"]
    problems = [f"qualification: {p}" for p in reading["problems"]]
    states: dict[str, dict[str, Any]] = {}
    for arm in arms:
        states[arm] = {}
        for label in LABELS:
            lr = float(rates[arm][label])
            rate = qualification_records.rate_of(reading, arm, lr)
            state = None if rate is None else rate["state"]
            states[arm][label] = state
            if state != QUALIFIED:
                shown = state or "not in the qualification"
                problems.append(f"qualification: {arm}'s {label} {lr:g} is {shown}, not qualified")
    return states, problems


def confirmation_fixation(
    runs_root: str | Path,
    candidate: int,
    *,
    code: Mapping[str, str],
    repo: str | Path = REPO_ROOT,
    written_at: str | None = None,
) -> dict[str, Any]:
    """Candidate N's confirmation supplement (module docstring): its validation fixation, its n
    record and, when N needs one, its budget record, from the records under ``runs_root`` and the
    checkout ``repo``; ``code``: the commit it is written at and that commit's GPU code.
    ``RecordRefused`` while the validation fixation is not valid, the n record is not there, not
    valid or not passed, a needed budget record is not there or does not proceed, or for any
    problem its reader would name; and once the record is written."""
    _candidate(candidate)
    root, repo = Path(runs_root), Path(repo)
    _not_written(repo / fixation_name("confirmation", candidate))
    fixation, found = read_validation_fixation(candidate, runs_root=root, repo=repo)
    problems = [] if fixation is not None else [_no_fixation(candidate, found)]
    kind = f"n-c{candidate}"
    n, found = read_decision(root, kind, repo=repo)
    problems += [f"n: {p}" for p in _passed(kind, n, found)]
    check = _Check(KINDS["confirmation"], candidate)
    check.code(code)
    problems += check.problems
    budget = None
    if n is not None and n["state"] == "passed":
        simulation, _ = read_decision(root, "simulation", repo=repo)
        reason = budget_reason(n["N"], simulation)
        if reason is not None:
            answer = f"budget-c{candidate}"
            entry, found = read_decision(root, answer, repo=repo)
            if entry is None:
                problems.append(f"budget: N needs a budget decision ({reason}), and "
                                + _passed(answer, entry, found)[0])  # fmt: skip
            elif entry["decision"] != "proceed":
                problems.append(f"budget: the {answer} record decides {entry['decision']}: the "
                                "confirmation does not start")  # fmt: skip
            else:
                budget = {"path": "/".join((*DIRECTORY, f"{answer}.json")),
                          "sha256": _digest(decision_path(root, answer))}  # fmt: skip
    if problems:
        raise RecordRefused(_refusal(KINDS["confirmation"], candidate, problems))
    name = fixation_name("validation", candidate)
    record = {
        "kind": KINDS["confirmation"], "candidate": candidate, "written_at": written_at or _now(),
        "validation_fixation": {"path": name, "sha256": _digest(repo / name), "root": "repo"},
        "n": {"path": "/".join((*DIRECTORY, f"{kind}.json")),
              "sha256": _digest(decision_path(root, kind))},
        "N": n["N"], "cap": n["cap"], "budget": budget,
        "code": {"commit": code["commit"], "gpu_code_sha256": code["gpu_code_sha256"]},
    }  # fmt: skip
    record = _plain(record, KINDS["confirmation"], candidate)
    problems = _confirmation_problems(record, candidate, root, repo)
    if problems:
        raise RecordRefused(_refusal(KINDS["confirmation"], candidate, problems))
    return record


def versions_record(
    path: str | Path,
    *,
    runs_root: str | Path,
    fixation: str | Path | None,
    reason: str,
    supersedes_reason: str | None = None,
    repo: str | Path = REPO_ROOT,
    written_at: str | None = None,
) -> dict[str, Any]:
    """The versions record to be written at ``path`` (a versions record's name in the checkout
    ``repo``; a relative path is read from it): the HEAD of a clean checkout, the model and the
    judge, the ``fixation`` record its runs are under (a committed fixation record, valid as its
    reader reads it against ``runs_root``; None: no fixation yet, and the record stands for no
    validation or confirmation run), the ``reason`` it is written; a later record supersedes the
    one before it, committed, with ``supersedes_reason``. ``RecordRefused`` for any problem, and
    once a record is written at ``path``."""
    repo, root = Path(repo), Path(runs_root)
    own = _in_checkout(path, repo)
    if own is None or not _VERSIONS_NAME.fullmatch(own):
        raise RecordRefused(f"{path} is not a versions record of the checkout {repo} "
                            f"({VERSIONS_FIRST}, docs/study3-versions-<k>.json)")  # fmt: skip
    _not_written(repo / own)
    state = git_state(repo)
    if not state.get("commit") or state.get("dirty") is not False:
        raise RecordRefused(f"the versions record needs a clean checkout of a known commit, not "
                            f"{state}")  # fmt: skip
    problems = []
    previous, supersedes = _previous(own), None
    if previous is None:
        if supersedes_reason is not None:
            problems.append(f"{own} is the first versions record: it supersedes none")
    elif supersedes_reason is None:
        problems.append(f"{own} is a later versions record: it supersedes {previous}, with the "
                        "reason (supersedes_reason)")  # fmt: skip
    elif not ((repo / previous).is_file() and tracked(previous, repo)):
        problems.append(f"{own} supersedes {previous}, which is not a committed file of the "
                        "checkout")  # fmt: skip
    else:
        supersedes = {"path": previous, "sha256": _digest(repo / previous),
                      "reason": supersedes_reason}  # fmt: skip
    pinned = None
    if fixation is not None:
        name = _in_checkout(fixation, repo)
        if name is None or not _FIXATION_NAME.fullmatch(name):
            problems.append(f"fixation: {fixation} is not a fixation record of the checkout "
                            "(docs/study3-fixation-<step>-c<N>.json)")  # fmt: skip
        elif not ((repo / name).is_file() and tracked(name, repo)):
            problems.append(f"fixation: {name} is not a committed file of the checkout")
        else:
            pinned = {"path": name, "sha256": _digest(repo / name)}
    if problems:
        raise RecordRefused(_refusal("versions", None, problems))
    model, judge = instruments()
    record = _plain({"what": WHAT, "written_at": written_at or _now(),
                     "code": {"commit": state["commit"]}, "model": model, "judge": judge,
                     "fixation": pinned, "supersedes": supersedes, "reason": reason},
                    "versions", None)  # fmt: skip
    check = _Check("versions")
    check.versions(record)
    problems = check.problems
    if not problems and supersedes is not None:
        problems = read_versions(repo / previous, repo=repo)[1]
    if not problems:
        problems = _pins(record, repo, _Fixations(root, repo))
    if problems:
        raise RecordRefused(_refusal("versions", None, problems))
    return record


def _not_written(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise RecordRefused(f"{path} is already written: a record is never rewritten (a later "
                            "versions record is a new file)")  # fmt: skip


def _plain(record: dict[str, Any], kind: str, candidate: int | None) -> dict[str, Any]:
    """The record as a reader will see it: JSON, with no NaN or infinity."""
    try:
        return json.loads(json.dumps(record, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RecordRefused(f"{_label_of(kind, candidate)}: not a JSON record ({exc})") from exc


def _refusal(kind: str, candidate: int | None, problems: Sequence[str]) -> str:
    return (f"{_label_of(kind, candidate)}: not built, {len(problems)} problem(s): "
            + "; ".join(problems))  # fmt: skip


def _label_of(kind: str, candidate: int | None) -> str:
    return kind if candidate is None else f"{kind}-c{candidate}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ------------------------------------------------------------------ the versions record


def read_versions(
    path: str | Path, *, repo: str | Path = REPO_ROOT
) -> tuple[dict[str, Any] | None, list[str]]:
    """A versions record if it is present and well formed, and its problems: its schema and the
    files it names (its fixation, the record it supersedes) as they are now in the checkout
    ``repo``; absent: (None, []). ``versions_problems`` checks it against the checkout and the
    journal."""
    path, repo = Path(path), Path(repo)
    if not (path.exists() or path.is_symlink()):
        return None, []
    if _under_sealed(path, repo):
        return None, [_sealed_problem(path)]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, [f"{path}: unreadable ({type(exc).__name__})"]
    check = _Check("versions")
    check.versions(record)
    problems = check.problems or _evidence(check.refs, repo, repo)
    return (None, [f"{path}: {p}" for p in problems]) if problems else (record, [])


@dataclass(frozen=True)
class _Link:
    """A versions record of a chain: its name, its sha256 (as it was added, or now for the record
    checked) and its content."""

    name: str
    sha256: str
    record: Mapping[str, Any]


class _Fixations:
    """The fixation records a check names, each read once by its reader: the record, and its
    problems as named (committed, its sha256 now, valid)."""

    def __init__(self, root: Path, repo: Path) -> None:
        self.root, self.repo = root, repo
        self.seen: dict[tuple[str, str], tuple[dict[str, Any] | None, list[str]]] = {}

    def get(self, ref: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        key = (ref["path"], ref["sha256"])
        if key not in self.seen:
            self.seen[key] = self._read(ref)
        return self.seen[key]

    def _read(self, ref: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        name = ref["path"]
        step, candidate = _FIXATION_NAME.fullmatch(name).groups()
        path, where = self.repo / name, f"fixation {name}"
        if _under_sealed(path, self.repo):
            return None, [_sealed_problem(path)]
        if not path.is_file():
            return None, [f"{where} is missing"]
        problems = [] if tracked(name, self.repo) else [f"{where} is not committed"]
        now = _digest(path)
        if now != ref["sha256"]:
            problems.append(f"{where} has sha256 {now} now, not the named {ref['sha256']}")
        record, found = _read_fixation(step, int(candidate), self.root, self.repo)
        problems += found
        return (None if problems else record), problems


def _pins(record: Mapping[str, Any], repo: Path, fixations: _Fixations) -> list[str]:
    """A versions record's fixation: committed, of the named sha256 now and at the recorded
    commit, valid, and of the recorded commit's GPU code."""
    ref = record["fixation"]
    if ref is None:
        return []
    fixation, found = fixations.get(ref)
    found = list(found)
    commit = record["code"]["commit"]
    blob = _blob(repo, commit, ref["path"])
    if blob is None or hashlib.sha256(blob).hexdigest() != ref["sha256"]:
        found.append(f"fixation: the recorded commit {commit} does not hold {ref['path']} with "
                     f"sha256 {ref['sha256']}: a versions record pins the commit that holds its "
                     "fixation")  # fmt: skip
    if fixation is not None:
        gpu, identity = fixation["code"]["gpu_code_sha256"], _gpu_code(repo, commit)
        if identity != gpu:
            found.append(f"code: the GPU code of the recorded commit {commit} is {identity}, not "
                         f"its fixation's {gpu}")  # fmt: skip
    return found


def versions_problems(
    path: str | Path,
    *,
    runs_root: str | Path,
    repo: str | Path = REPO_ROOT,
    step: str | None = None,
    candidate: int | None = None,
) -> list[str]:
    """How the checkout ``repo`` and the launch journal under ``runs_root`` depart from the
    versions record at ``path`` (module docstring, "The journal check"); none: it holds. With
    ``step`` (validation or confirmation) and ``candidate``, the record must also stand for a run
    of that step and candidate; ``ValueError`` for another step or candidate."""
    if step is not None or candidate is not None:
        _stage(step)
        _candidate(candidate)
    path, repo, root = Path(path), Path(repo), Path(runs_root)
    if not path.is_file():
        return [f"{path} is missing: write the versions record (scripts/study3_summary.py "
                "versions) and commit it first"]  # fmt: skip
    own = _in_checkout(path, repo)
    if own is None:
        return [f"{path} is not in the checkout {repo}: the record must be committed"]
    if not _VERSIONS_NAME.fullmatch(own):
        return [f"{own} is not a versions record of study 3 ({VERSIONS_FIRST}, "
                "docs/study3-versions-<k>.json)"]  # fmt: skip
    committed = tracked(own, repo)
    problems = [] if committed else [f"{path} is not committed: commit the versions record first"]
    try:
        data = path.read_bytes()
        record = json.loads(data)
    except (OSError, ValueError) as exc:
        return [*problems, f"{path}: unreadable ({type(exc).__name__})"]
    check = _Check("versions")
    check.versions(record)
    if check.problems:
        return [*problems, *(f"{own}: {p}" for p in check.problems)]
    problems += _checkout_problems(record, own, path, repo, committed)
    problems += [f"{own}: {p}" for p in _instrument_problems(record)]
    fixations = _Fixations(root, repo)
    problems += [f"{own}: {p}" for p in _pins(record, repo, fixations)]
    chain, found = _chain(own, record, hashlib.sha256(data).hexdigest(), repo)
    problems += found
    problems += _runs_problems(root, repo, chain, fixations)
    if step is not None:
        why = _stands_for(record["fixation"], step, candidate)
        if why is not None:
            problems.append(f"{own}: {why}")
    return list(dict.fromkeys(problems))


def _checkout_problems(record: Mapping[str, Any], own: str, path: Path, repo: Path,
                       committed: bool) -> list[str]:  # fmt: skip
    """HEAD against the recorded commit, as study 2 reads it (its record-only child), the
    checkout's changes, and the record unchanged since it was added."""
    commit, problems = record["code"]["commit"], []
    head = git_state(repo).get("commit")
    changes = code_changes(commit, repo)
    if head is None:
        problems.append("the running code is not a Git checkout: its commit is unknown")
    elif changes is None:
        problems.append(f"the running HEAD {head} is neither the recorded commit {commit} nor a "
                        "descendant of it")  # fmt: skip
    else:
        others = [name for name in changes["committed"] if name != own]
        uncommitted = changes["uncommitted"]
        if own in uncommitted:
            problems.append(f"{path} differs from its committed version at HEAD")
            uncommitted = [name for name in uncommitted if name != own]
        # an untracked record is named above, as not committed
        untracked = [name for name in changes["untracked"] if name != own]
        if others:
            problems.append(f"the running HEAD {head} differs from the recorded commit {commit} "
                            f"in {len(others)} files (first: {others[0]}): only the record "
                            "itself may follow it")  # fmt: skip
        if uncommitted:
            problems.append(f"the checkout has {len(uncommitted)} uncommitted changes (first: "
                            f"{uncommitted[0]})")  # fmt: skip
        if untracked:
            problems.append(f"the checkout has {len(untracked)} untracked files (first: "
                            f"{untracked[0]})")  # fmt: skip
    if committed:
        added, why = _added(repo, own)
        if added is None:
            problems.append(why)
        elif _blob(repo, "HEAD", own) not in (None, _blob(repo, added, own)):
            problems.append(
                f"{own} changed after it was added at {added}: a versions record is "
                "never overwritten (a later one is a new file that supersedes it)"
            )
    return problems


def _chain(own: str, record: Mapping[str, Any], digest: str,
           repo: Path) -> tuple[list[_Link], list[str]]:  # fmt: skip
    """The record and every record it supersedes, each earlier one read from Git at the commit
    that added it; and the problems of the succession."""
    chain, problems = [_Link(own, digest, record)], []
    name, current = own, record
    while True:
        previous, supersedes = _previous(name), current["supersedes"]
        if previous is None:
            if supersedes is not None:
                problems.append(f"{name} is the first versions record: it supersedes none")
            return chain, problems
        if supersedes is None:
            problems.append(f"{name} is a later versions record: it must name the one before it, "
                            f"{previous} (supersedes: its path, sha256 and reason)")  # fmt: skip
            return chain, problems
        if supersedes["path"] != previous:
            problems.append(f"{name} supersedes {supersedes['path']}, not the one before it, "
                            f"{previous}")  # fmt: skip
            return chain, problems
        added, why = _added(repo, previous)
        if added is None:
            problems.append(f"{name} supersedes {previous}: {why}")
            return chain, problems
        blob = _blob(repo, added, previous)
        then = None if blob is None else hashlib.sha256(blob).hexdigest()
        if then != supersedes["sha256"]:
            problems.append(f"{name} supersedes {previous} with sha256 {supersedes['sha256']}, but "
                            f"that record as added at {added} has {then}")  # fmt: skip
            return chain, problems
        now = repo / previous
        if not now.is_file() or _digest(now) != then:
            problems.append(f"{previous} changed after it was added at {added}: a versions record "
                            "is never overwritten (a later one is a new file that supersedes "
                            "it)")  # fmt: skip
        try:
            earlier = json.loads(blob)
        except ValueError:
            earlier = None
        check = _Check("versions")
        check.versions(earlier)
        if check.problems:
            problems.append(f"{previous} as added at {added} is not a versions record "
                            f"({check.problems[0]})")  # fmt: skip
            return chain, problems
        chain.append(_Link(previous, then, earlier))
        name, current = previous, earlier


def _runs_problems(root: Path, repo: Path, chain: Sequence[_Link],
                   fixations: _Fixations) -> list[str]:  # fmt: skip
    """The validation and confirmation runs of the launch journal against the chain, and the run
    directories the journal does not show (module docstring, "The journal check")."""
    launch, journal, problems = root / "launch", None, []
    if not any(launch.glob("*.jsonl")):
        problems.append(f"{launch}: the launch journal is missing, so the versions record each "
                        "validation and confirmation run started under cannot be shown (fails "
                        "closed): restore it")  # fmt: skip
    else:
        try:
            journal = _read_journal(root, {})
        except (OSError, ValueError) as exc:
            problems.append(f"{launch}: the launch journal cannot be read ({type(exc).__name__}), "
                            "so the versions record each validation and confirmation run started "
                            "under cannot be shown (fails closed)")  # fmt: skip
    started: set[str] = set()
    named: dict[tuple[str, str], list[str]] = defaultdict(list)
    pinned = {chain[0].name: []}  # the record's own fixation is checked already
    if journal is not None:
        if journal.corrupt:
            problems.append(f"the launch journal has an unparseable line that is not the last of "
                            f"its log ({', '.join(journal.corrupt)}): a start it held cannot be "
                            "shown (fails closed)")  # fmt: skip
        for campaign, start in journal.starts:
            run_id = start.get("run_id")
            if isinstance(run_id, str):
                started.add(run_id)
            why = _start_problem(start, chain, repo, fixations, pinned)
            if why is not None:
                named[(campaign, why)].append(f"{start.get('run')} ({run_id})")
    for (campaign, why), runs in sorted(named.items()):
        more = f" and {len(runs) - 1} other run(s)" if len(runs) > 1 else ""
        problems.append(f"{campaign}: {runs[0]}{more}: {why}")
    for name, found in pinned.items():
        problems += [f"{name}: {p}" for p in found]
    return problems + _unstarted(root, started)


def _start_problem(start: Mapping[str, Any], chain: Sequence[_Link], repo: Path,
                   fixations: _Fixations, pinned: dict[str, list[str]]) -> str | None:  # fmt: skip
    """Why a study-3 start is not a run of the chain's records that its record stands for (None:
    it is, or it is no validation or confirmation run). The fixation of each record a run is under
    is checked once, into ``pinned``."""
    header = start.get("header")
    if not (isinstance(header, Mapping) and header.get("study") == 3
            and header.get("step") in STEP_ROLES):  # fmt: skip
        return ("its campaign header names no study-3 step: whether it is a validation or "
                "confirmation run cannot be shown")  # fmt: skip
    step = header["step"]
    if step not in STAGES:
        return None
    candidate = header.get("candidate")
    if type(candidate) is not int or candidate not in (1, 2):
        return f"its campaign header names no candidate 1 or 2 ({candidate!r})"
    named, sha = header.get("versions_record"), header.get("versions_record_sha256")
    if not (isinstance(named, str) and named and isinstance(sha, str) and _SHA256.fullmatch(sha)):
        return f"a {step} run whose campaign header names no versions record with its sha256"
    link = next((link for link in chain if _names(named, link.name)), None)
    if link is None:
        return (f"its campaign header names the versions record {named}, outside the chain of "
                f"{chain[0].name} ({[link.name for link in chain]})")  # fmt: skip
    if sha != link.sha256:
        return (f"its campaign header names {link.name} with sha256 {sha}, not that record's "
                f"{link.sha256} as it was added")  # fmt: skip
    why = _stands_for(link.record["fixation"], step, candidate)
    if why is not None:
        return f"its record {link.name}: {why}"
    if link.name not in pinned:
        pinned[link.name] = _pins(link.record, repo, fixations)
    return None


def _names(named: str, name: str) -> bool:
    """Whether a header's path names the record ``name`` (a checkout's ``docs/<file>``): by its
    last components, so that a path from another checkout names it."""
    parts = PurePosixPath(name).parts
    return PurePosixPath(named.replace("\\", "/")).parts[-len(parts) :] == parts


def _stands_for(fixation: Mapping[str, Any] | None, step: str, candidate: int) -> str | None:
    """Why a record whose ``fixation`` is this cannot stand for a run of ``step`` and
    ``candidate`` (None: it can)."""
    wanted = fixation_name(step, candidate)
    what = "validation fixation" if step == "validation" else "confirmation supplement"
    if fixation is None:
        return (f"its fixation is null: a versions record without a fixation stands for no "
                f"validation or confirmation run (a {step} run of candidate {candidate} needs "
                f"{wanted}, its candidate's {what})")  # fmt: skip
    if fixation["path"] != wanted:
        return (f"its fixation is {fixation['path']}, not {wanted}: a {step} run of candidate "
                f"{candidate} needs its candidate's {what}")  # fmt: skip
    return None


def _unstarted(root: Path, started: set[str]) -> list[str]:
    """The study-3 run directories without a start in the journal that are validation or
    confirmation runs, or whose role cannot be read (fails closed)."""
    experiment = root / EXPERIMENT
    try:
        folders = sorted(experiment.iterdir()) if experiment.is_dir() else []
    except OSError as exc:
        return [f"{experiment}: the run directories cannot be listed ({type(exc).__name__})"]
    found: dict[str, list[str]] = defaultdict(list)
    for folder in folders:
        if folder.name == DECISIONS or folder.name in started or not folder.is_dir():
            continue
        role = _role(folder, root)
        if role is None:
            found["whose role cannot be read"].append(folder.name)
        elif role in STAGES:
            found["a validation or confirmation run"].append(folder.name)
    problems = []
    for what, names in sorted(found.items()):
        more = f" and {len(names) - 1} other(s)" if len(names) > 1 else ""
        problems.append(
            f"the study-3 run {names[0]}{more} has no start in the launch journal, "
            f"and is {what}: the versions record it was launched under cannot be "
            "shown (fails closed): restore the launch journal that started it"
        )
    return problems


def _role(folder: Path, root: Path) -> str | None:
    """A run directory's role: its ``config.yaml``'s ``study3.role``, else the role of its
    ``meta.json``'s campaign step; None when neither can be read. A file that resolves under a
    responses or observations folder is not opened."""
    config, meta = folder / "config.yaml", folder / "meta.json"
    if config.is_file() and not _under_sealed(config, root):
        try:
            role = OmegaConf.select(OmegaConf.load(config), "study3.role", default=None)
        except Exception:  # an unreadable config tells no role
            role = None
        if isinstance(role, str):
            return role
    if meta.is_file() and not _under_sealed(meta, root):
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        campaign = data.get("campaign") if isinstance(data, dict) else None
        header = campaign.get("header") if isinstance(campaign, Mapping) else None
        step = header.get("step") if isinstance(header, Mapping) else None
        if isinstance(step, str) and step in STEP_ROLES:
            return STEP_ROLES[step]
    return None
