"""Study 3's versions record, its staged fixation and the journal check.

docs/study3-preregistration.md: "Seeds and order" (the versions record pins one commit for the
validation and the confirmation, as in study 2; the check fails closed: a run without its start in
the launch journal does not count until the journal is restored; harm labels never enter a
selection: after the fixation there are no new candidates, rates or K*), "Simulation before
validation" (L and the validation size, recorded before validation), "Validation", "Sample size
from validation" (the candidate, the rates, the numerical policy and the judge sealed; N, the cap
and the decision on the money), "Inference" (the bootstrap's fixed seed, recorded in the fixation
record before confirmation), "The audit" (the draw's seed and procedure, in the fixation record
before any confirmation label is opened; the agreements checked on a draw with a recorded seed),
"Confirmation" and "Voiding and supersession" (a validation or confirmation run under a commit
other than the pinned one does not count). Study 2's versions record
(``study2_short.versions_problems`` and ``read_versions_record``, docs/study2-short-versions*.json)
is the model. Study 3 fixes in two stages: a record before a candidate's validation, and before
its confirmation a supplement that names the first.

- ``validation_fixation``, ``confirmation_fixation`` and ``versions_record``: the builders. Each
  returns a record from records already written: the stage and its rates through
  ``study3_progress.progress``, the decision records through ``study3_decisions.read_decision``, the
  numerical qualification through ``qualification_records.read_qualification`` (recomputed from
  its records), the checkout through Git. A built record is checked as its reader checks it, and
  ``RecordRefused`` names every problem. No builder writes a file (``scripts/study3_summary.py``
  does).
- ``read_validation_fixation``, ``read_confirmation_fixation`` and ``read_versions``: a record
  and its problems, ``(record | None, problems)`` as ``study3_decisions.read_decision`` gives
  them: an absent record is (None, []); with any problem the record is None.
- ``versions_problems``: how the checkout and the launch journal depart from a versions record;
  none: it holds.
- ``uncounted_runs``: the validation and confirmation runs that do not count for the commit they
  ran, each with its reason; ``committed_changes``: the one reading of a record-only descendant,
  for the running HEAD and for a run's own commit.
- ``qualification_logs``: the launch logs a candidate's qualification rests on, each with its
  sha256 now; ``registered_policy``: the registration at a commit, as a fixation pins it;
  ``instruments``: the model and the judge as this checkout's configs give them.
- ``fixation_name`` and ``versions_name``: the records' paths in the checkout;
  ``fixation_chain`` and ``fixation_number``: a fixation record and its successors there, and a
  link's number; ``dropped_rates``: the rates of an arm an earlier link of the validation
  fixation chain fixed and its link in force leaves out (P, closed after a cycle).

**The records** lie in the checkout (``docs/``). Their schemas are closed: a field the schema does
not name is refused, and they hold numbers, ids, paths and hashes only (the model's and the
judge's names and revisions are repository ids: letters, digits and ``._@+-/``, no space). Evidence
under RUNS_DIR is named as ``study3_decisions`` names it, ``{"path", "sha256"}`` with the path
relative to RUNS_DIR (``"root": "repo"`` for a path relative to the checkout), and checked against
the file as it is now, byte for byte, whenever the record is read; a file under a ``responses/``
or ``observations/`` folder is refused and never opened. ``written_at`` is ISO 8601 in UTC;
``code`` is ``commit`` (40 hexadecimal digits) and ``gpu_code_sha256``
(``code_identity.gpu_code_identity`` of that commit); ``model`` is the model's ``name`` and
``revision`` and ``judge`` the judge's ``config`` and identity (``judge.IDENTITY_FIELDS``), as
study 2's records give them, equal to this checkout's configs (``experiments/study3`` and
``judge/wildguard``, the frozen judge).

1. **The validation fixation**, ``docs/study3-fixation-validation-c<N>.json``, written once before
   candidate N's validation: ``kind`` (``fixation-validation``) and ``candidate``;
   ``written_at``; ``policy``: the registration (``path``, docs/study3-preregistration.md), the
   ``commit`` it is read at, the file's ``sha256`` at that commit and its ``amendments``, the
   numbers of all its "Amendment <n>" headings there; ``rates``: the active arms of candidate N's
   validation step (A, F and, while P–F is open, P) and their ``lr_l`` and ``lr_u``, as
   ``study3_progress`` gives them; ``qualification``: ``logs``, the evidence of every launch log
   the qualification rests on (``qualification_logs``), ``ledger``, the evidence of every record
   file ``read_qualification`` takes as an input for the candidate (its ``inputs``: the part
   attempt records, the trajectory results, the case restart and supersession records; never the
   verdict cache, which package 1 rewrites, nor a ``checkpoints-removed`` record), both in the
   order of their paths, and the ``states`` of the fixed rates (per arm, ``lr_l`` and ``lr_u``),
   each ``qualified`` or ``admitted`` (a failing rate amendment 10 admits); ``simulation`` and
   ``engineering``: the evidence of the decision records
   ``study3/decisions/simulation.json`` and ``study3/decisions/engineering.json``;
   ``validation_blocks`` (28 or 40) and ``level`` (L), the simulation record's; ``model``,
   ``judge`` and ``code``.
2. **The confirmation supplement**, ``docs/study3-fixation-confirmation-c<N>.json``, written once
   before candidate N's confirmation: ``kind`` (``fixation-confirmation``) and ``candidate``;
   ``written_at``; ``validation_fixation``: the first record's evidence (``"root": "repo"``);
   ``n``: the evidence of ``study3/decisions/n-c<N>.json``; ``N`` and ``cap``, the n record's;
   ``budget``: the evidence of ``study3/decisions/budget-c<N>.json`` when
   ``study3_decisions.budget_reason`` says N needs a budget decision (its decision proceed), else
   null; ``bootstrap``: the interval's fixed ``seed`` and its ``resamples`` (100 000);
   ``audit``: the draw's ``seed``, the ``agreement_seed`` of the draw of agreements the researcher
   checks, and the ``procedure``, the fixed id ``AUDIT_PROCEDURE`` that names the registration's
   "Draw" (probability proportional to size, with replacement; 400 A, 400 F and 200 P draws), run
   by the pinned ``code``; seeds are integers in [0, 2³²); ``code``.
**Successors of a fixation** ("Voiding and supersession"; "Seeds and order": after the fixation
there are no new candidates, rates or K*). A candidate's engineering cycle after a fixation
record moves its GPU code: the record goes stale, and its successor is a new file,
``docs/study3-fixation-<step>-c<N>-<k>.json`` (k = 2, 3, …), the record before it kept as it was.
A successor carries ``supersedes``: ``{"path", "sha256", "reason", "cycle"}``, the record before
it in the checkout (``-<k-1>``, or the first) by its path and its sha256 now, the reason
``cycle_c<N>`` and the evidence of the candidate's cycle record (``study3/decisions/
cycle-c<N>.json`` under RUNS_DIR); the first record carries none. The record in force is the
last link of an unbroken chain; a successor is accepted only after a stale record (the cycle
``study3_cycle.read_cycle`` accepts has the predecessor's code as its old code and lists the
predecessor among the records it makes stale; the successor carries the cycle's fix). A
validation fixation's successor keeps the candidate and the rates of the one before it: A's and
F's as they were, and P's while P–F is open. P alone is left out once the route closed it after
the cycle: P's re-qualification of a rate the record before it fixed failing, before the
confirmation has started (after it, a scientific failure closes no arm: ``confirmation_fixed``; a
technical mark closes nothing, amendment 7). The record before it stays as it was, P's rates and
states in it as their evidence (``dropped_rates``). A supplement's successor names the
validation fixation and the n record in force; once the confirmation has started
(``study3_decisions.confirmation_started``), its N, cap, bootstrap and audit seeds are its
predecessor's: they never change without an explicit prospective amendment, and the n record in
force keeps P's rates and P–F as N was decided, a P closed since fixed by the earlier link alone.
3. **The versions record**, ``docs/study3-versions.json`` and any later one
   ``docs/study3-versions-<k>.json`` (k = 2, 3, …): ``what`` (the fixed text ``WHAT``),
   ``written_at``, ``code`` (its ``commit`` alone), ``model``, ``judge``, ``fixation`` (the
   fixation record its runs are under, ``{"path", "sha256"}`` relative to the checkout, or null),
   ``supersedes`` (``{"path", "sha256", "reason"}``: the record before it; null for the first) and
   ``reason`` (why it is written). A reason is an id (a lowercase letter, then lowercase letters,
   digits or underscores, at most 64). A later record is a new file; nothing is overwritten.

**What each reader checks.** A fixation's reader reads the link in force of its chain in full,
and every earlier link for its integrity, as an earlier versions record is read: committed and
unchanged since it was added, of the sha256 the link after it names; a gap or a fork (a link
that names another than the one before it) is a problem, and so is a successor of a record that
is not stale (above, "Successors of a fixation").
- The validation fixation: its schema; its evidence now (the logs and the ledger included, each
  unchanged); the policy at its commit, which HEAD descends from; that the qualification rests
  now on exactly the logs it pins; the simulation record valid and passed, of the fixed L and
  size; the engineering record valid and passed; the model and the judge; the code: a commit HEAD
  descends from, whose GPU code is ``gpu_code_sha256``, the code in force for the candidate's runs
  (the chain of engineering cycles from the engineering record, ``study3_progress.code_chain``);
  and the qualification recomputed under the code it counts under (``read_qualification`` under
  ``study3_progress.qualification_code``: the code in force, or the old code of a cycle that
  reuses the qualification), without a
  problem, its fixed rates' states the pinned ones and no input added that bears on them (a record
  of a fixed rate's case, or one of the candidate's in no case, as a supersession record is). A
  later restart or supersession so shows as a changed state, a new input or a changed code in
  force; a rewritten verdict cache, a new ``checkpoints-removed`` record or a record added to the
  case of a rate the fixation does not fix (a closed arm's) changes nothing. Every pinned file
  stays unchanged, whatever its case.
- The confirmation supplement: its schema; its evidence now; its candidate's validation fixation
  valid (its full reader) and the one in force, the n record the one in force
  (``study3_decisions.decisions``) and valid and passed, of its N and cap, of the fixed L, of the
  fixed rates for the arms it names (once the confirmation has started, P's those of the earlier
  link that fixed P, when the link in force leaves P out) and of the fixation's GPU code; the
  budget record named exactly when N needs one, valid, its decision proceed; the seeds' form; its
  code, checked as the validation fixation's is (so both carry the code in force: one GPU code).
- The versions record: its schema and the evidence it names now (its fixation, the record it
  supersedes).

**The journal check** (``versions_problems``), as study 2's, failing closed:
- the record lies in the checkout under its name, Git tracks it, and it is unchanged at HEAD and
  since the commit that added it (a record is never overwritten);
- HEAD is the recorded commit or its record-only child, read as study 2 reads it: a descendant
  whose changes from the recorded commit touch the record alone; no tracked file has an
  uncommitted change, and no file is untracked unless the checkout's ``.gitignore`` ignores it;
- its model and judge are this checkout's configs'; its fixation, when it names one, is committed,
  of the named sha256 now and at the recorded commit, valid (its full reader), and of the recorded
  commit's GPU code;
- the succession: the first record names none; every later one names the one before it, by path,
  sha256 and reason; each earlier record is read from Git at the commit that added it, and a
  record added more than once, changed since or deleted is a problem;
- a record's fixation, the record checked's, is read in full when it is its chain's link in
  force, and for its integrity when it is an earlier link (the runs it stands for keep counting);
- every validation and confirmation run (a study-3 start of the launch journal whose campaign
  header names that step) started under a campaign whose header names this record or one it
  supersedes (``versions_record``: its path, or a path from another checkout that ends with it),
  with that record's sha256 as it was added (``versions_record_sha256``): a supersession keeps the
  earlier runs valid under the record they started with. A header naming no record, a record
  outside the chain or another sha256 is refused, and so is a study-3 start whose header names no
  study-3 step: whether it is a validation or confirmation run cannot be shown;
- the record each run is under stands for it: a validation run's record names a link of its
  candidate's validation fixation chain, a confirmation run's one of its supplement's; a record
  whose
  fixation is null stands for none. For an earlier record, as study 2 checks it, that fixation
  file is committed and of the sha256 the record pins (its full reader runs for the record checked
  only);
- each such run's arm is active in its fixation and its rate one of the two the fixation fixes for
  that arm (a validation fixation's ``rates``; for a supplement, the rates of the n record it pins:
  the arms still open at confirmation), read from the run's cell as ``study3_records`` binds it;
- a missing launch journal, a log that cannot be read, a torn line before the last line of its
  log that no ``torn`` and ``torn_resolved`` records both name (``study3_records``' rule; a
  resolved one is no problem), and a study-3 run directory without a start in the journal that
  is a validation or confirmation run (its ``config.yaml``'s role, else its ``meta.json``'s
  campaign header), or whose role cannot be read, are problems: what the journal does not show
  cannot be shown.
With ``step`` and ``candidate``, the record must also stand for a run of that step and candidate,
its fixation the link in force: the check before a campaign is written under it.

**The sealing boundary.** Of a run directory only ``config.yaml`` and ``meta.json`` are read (a
directory with no start in the journal here; a started run's cell through ``study3_records``); a
file that resolves under a ``responses/`` or ``observations/`` folder is checked for that before
any read or hash of a record path, and never opened. No response text, harm label or judge output
is read.

**Rulings** where the registration leaves a detail open:
- *The registration's commit* is, by default, the last commit that changed the registration, which
  the checkout holds as committed; the amendments are named by number, all of them.
- *The logs the qualification rests on* are every launch log that holds one of its lines: a start
  of a ``numerical`` or ``repair`` campaign of the candidate, or a line naming one of those runs (an
  ``end``, ``superseded`` or ``held`` line by run id, a ``superseded`` or ``held`` line by label, a
  recovery round naming a label). They are pinned whole, by sha256: no registered route writes to
  them after the fixation (the campaign writer refuses a reused campaign name). Any change to a
  pinned log breaks the fixation, the launcher's repair of a torn line and its resolution
  written after the fixation included. So no log that holds a torn line the launcher refuses
  (every one not resolved, the last line of its log included, which the other readers skip) is
  pinned: it is a problem by its log and line, in any log, until its resolution
  (``scripts/launch.py --resolve-torn``) is written, before the fixation; the builder refuses
  meanwhile. Once the fixation exists, its reader names such a line by its log and line with the
  command that resolves it, and, in a log the fixation pins, adds that the resolution changes
  that log, so the fixation breaks.
- *The fixed rates* are the stage's: the validation step's cells when it carries them, each at one
  of the arm's two rates, and otherwise the rates of the reading the cells come from (``progress``'
  ``rates`` and ``active``); P is fixed exactly while P–F is open.
- *The two reasons*: ``reason`` says why the record is written (the stage it serves, for example
  ``validation_c1``), ``supersedes.reason`` why the record before it no longer serves.
- *A run's own commit* (``meta.json``'s ``git.commit`` and ``git.dirty``, kept by
  ``study3_records``) is compared with the commit the versions record its header names pins, as
  study 2 reads a run's commit (``uncounted_runs``): the pinned commit or a record-only
  descendant of it, from a clean checkout. A run under any other commit, a dirty checkout, a
  commit this checkout does not hold or no recorded commit does not count; it is kept and listed,
  and ``study3_progress`` blocks on it as a technical mark of its arm, until its way out: a
  recovery round where the launcher would open one. The check of the record before each launch
  (``scripts/study3_summary.py check``, and the confirmation's guard) prevents such a run; it
  does not stand for the comparison, which also covers a resumption or a recovery round launched
  from another checkout.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from omegaconf import OmegaConf

from .. import code_identity
from ..config import load_config
from ..judge import IDENTITY_FIELDS, judge_identity
from ..paths import REPO_ROOT
from ..tracking import git_state, tracked, worktree_changes
from .study3_decisions import (
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
    confirmation_started,
    decision_path,
    decisions,
    kind_of,
    read_decision,
)
from .study3_decisions import UNCOUNTED_REASONS as UNCOUNTED_REASONS  # ``uncounted_runs``' reasons
from .study3_records import (
    COUNTING,
    EXPERIMENT,
    RATE_TOLERANCE,
    RESERVED,
    STEP_ROLES,
    TORN_LINE,
    VERSIONED_STEPS,
    Endpoint,
    _bind,
    _Files,
    _read_journal,
    _sealed_file,
    _sealed_records,
    journal_lines,
    plain_name,
    sealed_record,
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
ADMITTED = "admitted"  # a failing rate amendment 10 admits to validation (its eligibility)
LABELS = ("lr_l", "lr_u")
ARMS = ("A", "F", "P")
# A launch log that holds a torn line the launcher refuses: the fixation pins none (``Rulings``).
# While the fixation is built (``qualification_logs``), the line is named so:
REFUSED_TORN = ("a torn line the launcher refuses: resolve it (scripts/launch.py --resolve-torn) "
                "before the fixation; its resolution changes the log")  # fmt: skip
# Once the fixation exists, its reader names the line so, and in a log the fixation pins adds why
# the fixation breaks
FIXED_TORN = "a torn line the launcher refuses: resolve it (scripts/launch.py --resolve-torn)"
PINNED_TORN = "its resolution changes a log the fixation pins, so the fixation breaks"
# The campaign steps whose runs are qualification cases (qualification_records.STEPS; that module
# loads torch, and is imported only where it is read)
QUALIFICATION_STEPS = ("numerical", "repair")
ENGINEERING_RECORD = "/".join((*DIRECTORY, "engineering.json"))
# "Inference": 100 000 resamples of blocks; "The audit", "Draw": the registered procedure
BOOTSTRAP_RESAMPLES = 100_000
AUDIT_PROCEDURE = "pps-with-replacement-400-400-200"
SEED_LIMIT = 2**32  # a seed is an integer in [0, 2³²)
FIELDS = {
    "validation": ("kind", "candidate", "written_at", "policy", "rates", "qualification",
                   "simulation", "validation_blocks", "level", "engineering", "model", "judge",
                   "code"),
    "confirmation": ("kind", "candidate", "written_at", "validation_fixation", "n", "N", "cap",
                     "budget", "bootstrap", "audit", "code"),
    "versions": ("what", "written_at", "code", "model", "judge", "fixation", "supersedes",
                 "reason"),
}  # fmt: skip

_VERSIONS_NAME = re.compile(r"docs/study3-versions(?:-([2-9]|[1-9][0-9]+))?\.json")
_FIXATION_NAME = re.compile(
    r"docs/study3-fixation-(validation|confirmation)-c([12])"
    r"(?:-([2-9]|[1-9][0-9]+))?\.json"
)
_LINK_RECORD = re.compile(r"study3/decisions/(n|budget)-c([12])(?:-(?:[2-9]|[1-9][0-9]+))?\.json")
_LOG = re.compile(r"launch/[^/\\]+\.jsonl")
_AMENDMENT = re.compile(r"^### Amendment (\d+)\b", re.MULTILINE)
_ID = re.compile(r"[a-z][a-z0-9_]{0,63}")
# A repository id or revision (a model's or judge's name, a revision, a prompt version)
_PART = r"[A-Za-z0-9][A-Za-z0-9._@+-]{0,127}"
_NAME = re.compile(rf"{_PART}(?:/{_PART}){{0,3}}")
_COMMIT, _SHA256 = re.compile(r"[0-9a-f]{40}"), re.compile(r"[0-9a-f]{64}")
# Why a validation or confirmation run does not count for the commit it ran (``uncounted_runs``):
# ``UNCOUNTED_REASONS``, kept with the closure schema that names them (``study3_decisions``)


class RecordRefused(ValueError):
    """A fixation or versions record that cannot be built: its problems named."""


# ------------------------------------------------------------------ names


def fixation_name(step: str, candidate: int, k: int = 1) -> str:
    """``docs/study3-fixation-<step>-c<candidate>.json``, relative to the checkout; its k-th
    link (k ≥ 2, a successor) ``docs/study3-fixation-<step>-c<candidate>-<k>.json``."""
    _stage(step)
    _candidate(candidate)
    if type(k) is not int or k < 1:
        raise ValueError(f"{k!r} is not the number of a fixation record (1, 2, ...)")
    return f"docs/study3-fixation-{step}-c{candidate}" + ("" if k == 1 else f"-{k}") + ".json"


def fixation_chain(step: str, candidate: int, *, repo: str | Path = REPO_ROOT) -> list[str]:
    """The links of candidate N's fixation chain of ``step`` in the checkout ``repo`` (module
    docstring, "Successors of a fixation"): the first record and its successors there, in the
    order of their numbers (a gap shows as a missing number)."""
    _stage(step)
    _candidate(candidate)
    folder, found = Path(repo) / "docs", []
    paths = folder.glob(f"study3-fixation-{step}-c{candidate}*.json") if folder.is_dir() else ()
    for path in paths:
        match = _FIXATION_NAME.fullmatch(f"docs/{path.name}")
        if match and match.group(1) == step and int(match.group(2)) == candidate:
            found.append((int(match.group(3) or 1), f"docs/{path.name}"))
    return [name for _, name in sorted(found)]


def fixation_number(name: str) -> int:
    """A fixation record's number in its chain (1 for the first)."""
    return int(_FIXATION_NAME.fullmatch(name).group(3) or 1)


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


def committed_changes(base: str, target: str, repo: str | Path = REPO_ROOT) -> list[str] | None:
    """The paths changed from the commit ``base`` to ``target`` (a commit, or ``HEAD``) when
    ``target`` is ``base`` or descends from it; None otherwise, or when either is unknown to the
    checkout ``repo``. A record-only descendant of ``base`` is one whose changes touch the record
    alone: the one reading of it, for the running HEAD (``versions_problems``) and for a run's own
    commit (``uncounted_runs``), as study 2 reads it."""
    try:
        _git(Path(repo), "merge-base", "--is-ancestor", base, target)
        out = _git(Path(repo), "diff", "--name-only", "-z", "--no-renames", base, target)
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted(set(out.decode("utf-8", "replace").split("\0")) - {""})


def _known(repo: Path, commit: str) -> bool:
    """Whether ``commit`` (40 hexadecimal digits) is a commit of the checkout ``repo``."""
    if not _COMMIT.fullmatch(commit):
        return False
    try:
        _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
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
    resolved = _from(path, repo).resolve()
    root = repo.resolve()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else None


def _from(path: str | Path, repo: Path) -> Path:
    """``path``, a relative one read from the checkout ``repo``."""
    path = Path(path)
    return path if path.is_absolute() else repo / path


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
    RUNS_DIR and its sha256 now; and the problems: a log that cannot be read, and each torn line
    the launcher refuses (``study3_records.journal_lines``' ``refused``: every torn line that is
    not resolved, the last of its log included), by its log and line, in any log: it may be a line
    of the qualification, and its one resolution appends to its log (``REFUSED_TORN``: resolve it
    before the fixation). A log that holds one is not pinned; a resolved one is no problem."""
    pinned, problems, refused = _qualification_logs(runs_root, candidate)
    return pinned, problems + [f"{where} line {number}: {REFUSED_TORN}"
                               for where, number in refused]  # fmt: skip


def _qualification_logs(
    runs_root: str | Path, candidate: int
) -> tuple[list[dict[str, str]], list[str], list[tuple[str, int]]]:
    """``qualification_logs``' logs, its logs that cannot be read (as problems), and each torn
    line the launcher refuses, by its log's path under RUNS_DIR and its number: named by the
    builder (``qualification_logs``) and by the reader of an existing fixation
    (``_logs_problems``), each in its own words."""
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
    pinned, refused = [], []
    for log in logs:
        refused += [(log.path, number) for number in log.refused]
        if not log.refused and any(
                line.record is not None and _of_qualification(line.record, candidate, ids, labels)
                for line in log.lines):  # fmt: skip
            pinned.append({"path": log.path, "sha256": log.sha256})
    return pinned, problems, refused


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


# ------------------------------------------------------------------ the schemas


class _Check(_Schema):
    """The closed schemas of the fixation and versions records, built on the blocks of
    ``study3_decisions``' schemas (one vocabulary of problems); ``refs`` is the evidence to check
    against the files."""

    def __init__(self, kind: str, candidate: int | None = None, k: int = 1) -> None:
        super().__init__(kind, kind, candidate)
        self.k = k  # a fixation's number in its chain

    def pattern(self, where: str, value: Any, pattern: re.Pattern, what: str) -> bool:
        ok = isinstance(value, str) and pattern.fullmatch(value) is not None
        if not ok:
            self.add(where, f"is {value!r}, not {what}")
        return ok

    def ident(self, where: str, value: Any) -> None:
        self.pattern(where, value, _ID, "an id (a lowercase letter, then lowercase letters, digits "
                     "or underscores, at most 64)")  # fmt: skip

    def name(self, where: str, value: Any) -> None:
        self.pattern(where, value, _NAME, "a repository id or revision (letters, digits and "
                     "._@+-/, no space)")  # fmt: skip

    def seed(self, where: str, value: Any) -> None:
        if not (type(value) is int and 0 <= value < SEED_LIMIT):
            self.add(where, f"is {value!r}, not a seed (an integer in [0, 2³²))")

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
            self.name(f"model.{key}", model[key])
        judge = self.object("judge", top["judge"], ("config", *IDENTITY_FIELDS))
        if judge is not None:
            self.name("judge.config", judge["config"])
            for key in IDENTITY_FIELDS:
                if judge[key] is not None:
                    self.name(f"judge.{key}", judge[key])

    def common(self, top: Mapping[str, Any]) -> None:
        if top["kind"] != self.kind:
            self.add("kind", f"is {top['kind']!r}, not {self.kind!r}")
        if type(top["candidate"]) is not int or top["candidate"] != self.candidate:
            self.add("candidate", f"is {top['candidate']!r}, not {self.candidate}")
        self._when(top["written_at"])

    def rates(self, value: Any) -> tuple[str, ...] | None:
        """The fixed rates, per active arm (A and F, and P while P–F is open); their arms, or
        None when they are not well formed."""
        before = len(self.problems)
        with_p = isinstance(value, dict) and "P" in value
        self._rates(value, with_p)
        return (("A", "F", "P") if with_p else ("A", "F")) if len(self.problems) == before else None

    def refs_list(self, where: str, value: Any, path: re.Pattern, what: str) -> None:
        """A non-empty list of evidence under the runs directory, each path ``path``'s, in the
        order of its paths, each once."""
        if not isinstance(value, list) or not value:
            self.add(where, f"is not a non-empty list of {what}")
            return
        paths = []
        for i, entry in enumerate(value):
            entry = self.ref(f"{where}[{i}]", entry)
            if entry is None:
                continue
            named = entry["path"]
            if entry.get("root", "runs") != "runs" or not (
                    isinstance(named, str) and path.fullmatch(named)):  # fmt: skip
                self.add(f"{where}[{i}]", f"is {named!r}, not one of the {what} under the runs "
                         "directory")  # fmt: skip
            paths.append(named)
        if all(isinstance(p, str) for p in paths) and paths != sorted(set(paths)):
            self.add(where, "is not in the order of its paths, each once")

    # -- the validation fixation

    def supersedes(self, top: Mapping[str, Any], step: str) -> None:
        """A successor's ``supersedes`` (module docstring, "Successors of a fixation"): the
        record before it in the checkout, by its path and sha256 now, the reason ``cycle_c<N>``
        and the candidate's cycle record as evidence; the first record carries none."""
        if self.k == 1:
            if "supersedes" in top:
                self.add("supersedes", "is a successor's field: the first record supersedes none")
            return
        if "supersedes" not in top:
            self.add("", "has no field `supersedes`: a successor names the record before it and "
                     "the candidate's cycle")  # fmt: skip
            return
        entry = self.repo_ref("supersedes", top["supersedes"], ("reason", "cycle"))
        if entry is None:
            return
        named = fixation_name(step, self.candidate, self.k - 1)
        if entry["path"] != named:
            self.add("supersedes.path", f"is {entry['path']!r}, not {named!r}, the record before "
                     "it: a successor names its predecessor (no fork, no gap)")  # fmt: skip
        if entry["reason"] != f"cycle_c{self.candidate}":
            self.add("supersedes.reason", f"is {entry['reason']!r}, not 'cycle_c{self.candidate}':"
                     " a successor follows the candidate's cycle")  # fmt: skip
        self._exact_ref("supersedes.cycle", entry["cycle"],
                        "/".join((*DIRECTORY, f"cycle-c{self.candidate}.json")))  # fmt: skip

    def validation(self, record: Any) -> None:
        top = self.object("", record, FIELDS["validation"], ("supersedes",))
        if top is None:
            return
        self.common(top)
        self.supersedes(top, "validation")
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
        found = self.object("qualification", value, ("logs", "ledger", "states"))
        if found is None:
            return
        self.refs_list("qualification.logs", found["logs"], _LOG, "launch logs launch/<log>.jsonl")
        self.refs_list("qualification.ledger", found["ledger"], _ledger_path(self.candidate),
                       f"records of candidate {self.candidate}'s qualification (no verdict cache, "
                       "no checkpoints-removed record)")  # fmt: skip
        if arms is None:
            return  # the rates' problems are named
        states = self.object("qualification.states", found["states"], arms)
        for arm in arms if states is not None else ():
            pair = self.object(f"qualification.states.{arm}", states[arm], LABELS)
            for label in LABELS if pair is not None else ():
                if pair[label] not in (QUALIFIED, ADMITTED):
                    self.add(f"qualification.states.{arm}.{label}", f"is {pair[label]!r}, not "
                             f"{QUALIFIED!r} or {ADMITTED!r}: a validation runs qualified rates "
                             "and those amendment 10 admits only")  # fmt: skip

    # -- the confirmation supplement

    def confirmation(self, record: Any) -> None:
        top = self.object("", record, FIELDS["confirmation"], ("supersedes",))
        if top is None:
            return
        self.common(top)
        self.supersedes(top, "confirmation")
        name = fixation_name("validation", self.candidate)
        entry = self.ref("validation_fixation", top["validation_fixation"])
        found = (_FIXATION_NAME.fullmatch(entry["path"]) if entry is not None
                 and isinstance(entry["path"], str) else None)  # fmt: skip
        if entry is not None and (found is None or found.group(1) != "validation"
                                  or int(found.group(2)) != self.candidate
                                  or entry.get("root", "runs") != "repo"):  # fmt: skip
            self.add("validation_fixation", f"is {entry['path']!r} under "
                     f"{entry.get('root', 'runs')!r}, not {name!r} (or a successor of it) under "
                     "'repo', the checkout")  # fmt: skip
        self.link_ref("n", top["n"], "n")
        if self.integer("N", top["N"], MIN_N):
            cap = cap_of(top["N"])
            if top["cap"] != cap or type(top["cap"]) is not int:
                self.add("cap", f"is {top['cap']!r}, not C(N) = {cap}")
        else:
            self.integer("cap", top["cap"], cap_of(MIN_N))
        if top["budget"] is not None:
            self.link_ref("budget", top["budget"], "budget")
        bootstrap = self.object("bootstrap", top["bootstrap"], ("seed", "resamples"))
        if bootstrap is not None:
            self.seed("bootstrap.seed", bootstrap["seed"])
            resamples = bootstrap["resamples"]
            if type(resamples) is not int or resamples != BOOTSTRAP_RESAMPLES:
                self.add("bootstrap.resamples", f"is {resamples!r}, not the registered "
                         f"{BOOTSTRAP_RESAMPLES}")  # fmt: skip
        audit = self.object("audit", top["audit"], ("seed", "agreement_seed", "procedure"))
        if audit is not None:
            self.seed("audit.seed", audit["seed"])
            self.seed("audit.agreement_seed", audit["agreement_seed"])
            if audit["procedure"] != AUDIT_PROCEDURE:
                self.add("audit.procedure", f"is {audit['procedure']!r}, not the registered "
                         f"draw {AUDIT_PROCEDURE!r}")  # fmt: skip
        self.code(top["code"])

    def link_ref(self, where: str, value: Any, family: str) -> None:
        """Evidence of a link of the candidate's n or budget chain (``n-c<N>.json`` or a
        successor ``n-c<N>-<k>.json``) under the runs directory, and no other file."""
        entry = self.ref(where, value)
        if entry is None:
            return
        found = _LINK_RECORD.fullmatch(entry["path"]) if isinstance(entry["path"], str) else None
        if (found is None or found.group(1) != family or int(found.group(2)) != self.candidate
                or entry.get("root", "runs") != "runs"):  # fmt: skip
            named = "/".join((*DIRECTORY, f"{family}-c{self.candidate}.json"))
            self.add(where, f"is {entry['path']!r} under {entry.get('root', 'runs')!r}, not the "
                     f"record {named!r} (or a successor of it) under the runs "
                     "directory")  # fmt: skip

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


def _ledger_path(candidate: int | None) -> re.Pattern:
    """A record file of the candidate's qualification as the ledger names it: under
    ``qualification/c<N>/``, a JSON file that is neither the verdict cache nor a
    ``checkpoints-removed`` record."""
    return re.compile(rf"qualification/c{candidate}/(?:[^/\\]+/)*"
                      r"(?!verdict\.json$)(?!checkpoints-removed-)[^/\\]+\.json")  # fmt: skip


# ------------------------------------------------------------------ reading the fixations


def read_validation_fixation(
    candidate: int,
    *,
    runs_root: str | Path,
    repo: str | Path = REPO_ROOT,
    settings: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Candidate N's validation fixation (``docs/study3-fixation-validation-c<N>.json`` in the
    checkout ``repo``) if it is present and valid, and its problems; absent: (None, []). Its
    evidence under RUNS_DIR is read from ``runs_root``; ``settings``:
    configs/experiments/qualification.yaml as loaded (by default this checkout's), under which the
    qualification is recomputed."""
    return _read_fixation("validation", candidate, Path(runs_root), Path(repo), settings)


def read_confirmation_fixation(
    candidate: int,
    *,
    runs_root: str | Path,
    repo: str | Path = REPO_ROOT,
    settings: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Candidate N's confirmation supplement (``docs/study3-fixation-confirmation-c<N>.json``),
    as ``read_validation_fixation`` reads the first (which it reads in full)."""
    return _read_fixation("confirmation", candidate, Path(runs_root), Path(repo), settings)


Reading = tuple[dict[str, Any] | None, list[str]]  # a record (None with a problem), its problems


def _read_fixation(step: str, candidate: int, root: Path, repo: Path,
                   settings: Mapping[str, Any] | None) -> Reading:  # fmt: skip
    """The fixation chain's link in force read in full, every earlier link for its integrity
    (module docstring, "What each reader checks")."""
    names = fixation_chain(step, candidate, repo=repo)
    if not names:
        return None, []
    path = repo / names[-1]
    problems = _chain_problems(names, repo)
    if _under_sealed(path, repo):
        return None, [*problems, _sealed_problem(path)]
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, [*problems, f"{path}: unreadable ({type(exc).__name__})"]
    check = _validation_problems if step == "validation" else _confirmation_problems
    problems += [f"{path}: {p}" for p in check(record, candidate, root, repo, settings,
                                                k=fixation_number(names[-1]))]  # fmt: skip
    return (None, problems) if problems else (record, [])


def _chain_problems(names: Sequence[str], repo: Path) -> list[str]:
    """The integrity of a fixation chain's earlier links (module docstring, "What each reader
    checks"): no gap, each committed and unchanged since it was added, of the sha256 the link
    after it names."""
    found = []
    if [fixation_number(name) for name in names] != list(range(1, len(names) + 1)):
        found.append(f"{names[-1]}: the chain {list(names)} has a gap: each successor follows the "
                     "record before it")  # fmt: skip
    for before, after in zip(names, names[1:], strict=False):
        path = repo / before
        if _under_sealed(path, repo):
            found.append(f"{before}: {_sealed_problem(path)}")
            continue
        added, why = _added(repo, before)
        now = path.read_bytes() if path.is_file() else None
        if added is None:
            found.append(why)
        elif now is None or _blob(repo, added, before) != now:
            found.append(f"{before} changed after it was added at {added}: a fixation record is "
                         "never overwritten (a successor is a new file)")  # fmt: skip
        later = repo / after
        try:
            cited = (json.loads(later.read_text(encoding="utf-8")).get("supersedes") or {}).get(
                "sha256")  # fmt: skip
        except (OSError, ValueError, AttributeError):
            cited = None
        if now is None or cited != hashlib.sha256(now).hexdigest():
            found.append(f"{after}: it names {before} with sha256 {cited!r}, not its sha256 now: "
                         "a link is unchanged once its successor names it")  # fmt: skip
    return found


def _succession(step: str, record: Mapping[str, Any], candidate: int, k: int, root: Path,
                repo: Path, reading: Mapping[str, Any] | None = None) -> list[str]:  # fmt: skip
    """A fixation successor against the record before it and the candidate's cycle (module
    docstring, "Successors of a fixation"); ``reading``: the candidate's qualification under the
    code it counts under (``_recomputed``'s), for a validation fixation's successor."""
    from .study3_cycle import read_cycle  # it reads the decision records: here only

    before = fixation_name(step, candidate, k - 1)
    path = repo / before
    predecessor = None
    if not _under_sealed(path, repo) and path.is_file():
        try:
            predecessor = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            predecessor = None
    if not isinstance(predecessor, dict):
        return [f"supersedes: the record before it, {before}, is not there or cannot be read"]
    found = read_cycle(root, candidate, repo=repo)
    if found["state"] != "valid" or found["record"] is None:
        why = "; ".join(found["problems"][:2]) or "none is there"
        return [f"supersedes.cycle: candidate {candidate}'s cycle is not accepted ({why}): a "
                "successor follows a stale record through an accepted cycle"]  # fmt: skip
    cycle, out = found["record"], []
    old, fix = cycle["old_code"], cycle["code"]["gpu_code_sha256"]
    own = (predecessor.get("code") or {}).get("gpu_code_sha256")
    listed = {(e.get("path"), e.get("sha256"), e.get("root", "runs"))
              for e in cycle["affected"]["records"]}  # fmt: skip
    pinned = (before, _digest(path), "repo") in listed
    if own != old or not pinned:
        out.append(f"supersedes: {before} is not stale through candidate {candidate}'s cycle (its "
                   f"code {own}, the cycle's old code {old}; listed among the records the cycle "
                   f"makes stale: {pinned}): a successor follows a stale record alone")  # fmt: skip
    if record["code"]["gpu_code_sha256"] != fix:
        out.append(f"code: its GPU code {record['code']['gpu_code_sha256']} is not the cycle's "
                   f"fix {fix}: a successor carries the fix")  # fmt: skip
    if step == "validation":
        out += _kept_rates(record, predecessor, before, candidate, root, repo, reading)
    elif confirmation_started(root, candidate):
        changed = [key for key in ("N", "cap", "bootstrap", "audit")
                   if record.get(key) != predecessor.get(key)]  # fmt: skip
        if changed:
            out.append(f"{', '.join(changed)}: after the confirmation has started, N, the cap and "
                       "the seeds never change without an explicit prospective amendment (a "
                       "successor differs from its predecessor only in its code, written_at, "
                       "supersedes and the references to the records in force)")  # fmt: skip
    return out


def _kept_rates(record: Mapping[str, Any], predecessor: Mapping[str, Any], before: str,
                candidate: int, root: Path, repo: Path,
                reading: Mapping[str, Any] | None) -> list[str]:  # fmt: skip
    """A validation fixation's successor keeps the candidate and the rates of the record before
    it (no new candidate, rate or K*): A's and F's as they were, and P's while P–F is open. P
    alone may be left out, once the route closed it after the cycle (``_closed_p``); the record
    before it keeps P's rates as its evidence (module docstring, "Successors of a fixation")."""
    rates, other = record["rates"], predecessor.get("rates")
    kept = isinstance(other, Mapping) and set(rates) <= set(other)
    if (record["candidate"] != predecessor.get("candidate") or not kept
            or not _same_rates(rates, {arm: other[arm] for arm in rates})
            or set(other) - set(rates) - {"P"}):  # fmt: skip
        return [f"rates: the candidate and the rates are not those of {before}: after the "
                "validation fixation a cycle keeps the candidate and the rates (no new candidate, "
                "rate or K*), P's left out only once P is closed"]  # fmt: skip
    if "P" in other and "P" not in rates and not _closed_p(candidate, other["P"], root, repo,
                                                            reading):  # fmt: skip
        return [f"rates: P is left out, and P is not closed: {before} fixes P, and a successor "
                "leaves P out only once the route closed it after the cycle (P's "
                "re-qualification of a fixed rate failing before the confirmation has "
                "started)"]  # fmt: skip
    return []


def _closed_p(candidate: int, rates: Mapping[str, Any], root: Path, repo: Path,
              reading: Mapping[str, Any] | None) -> bool:  # fmt: skip
    """Whether the route closes P at the candidate's re-qualification after its cycle: before the
    confirmation has started ("Validation, confirmation and stopping": after it, a scientific
    failure closes no arm the confirmation holds), a rate the record before it fixed for P
    (``rates``) failing under the code it counts under (``reading``). A technical mark or a rate
    still waiting closes nothing (amendment 7, "Technical causes")."""
    from . import qualification_records  # it loads torch (qualification_verdicts): here only
    from .study3_progress import _condition

    if reading is None or confirmation_started(root, candidate):
        return False
    for label in LABELS:
        try:
            rate = qualification_records.rate_of(reading, "P", float(rates[label]))
        except (KeyError, TypeError, ValueError):
            return False
        if rate is not None and _condition(rate, reading)[0] == "failing":
            return True
    return False


def dropped_rates(candidate: int, repo: str | Path = REPO_ROOT) -> dict[str, dict[str, Any]]:
    """The rates of every arm that an earlier link of the candidate's validation fixation chain
    fixes and its link in force leaves out (P, once closed after a cycle: module docstring,
    "Successors of a fixation"), as the last link that fixed it gives them; none without such a
    link. The links are read as written: the chain's reader checks their integrity."""
    repo = Path(repo)
    names = fixation_chain("validation", candidate, repo=repo)
    links = []
    for name in names:
        path = repo / name
        data = None
        if not _under_sealed(path, repo):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = None
        rates = data.get("rates") if isinstance(data, dict) else None
        links.append(rates if isinstance(rates, Mapping) else {})
    if not links:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for rates in links[:-1]:
        for arm, pair in rates.items():
            if arm not in links[-1] and isinstance(pair, Mapping):
                out[arm] = dict(pair)
    return out


def _same_rates(rates: Any, other: Any) -> bool:
    """The same arms with the same two rates each (relative tolerance 10⁻⁹)."""
    if not (isinstance(rates, Mapping) and isinstance(other, Mapping)) or set(rates) != set(other):
        return False
    try:
        return all(math.isclose(rates[arm][k], other[arm][k], rel_tol=RATE_TOLERANCE)
                   for arm in rates for k in LABELS)  # fmt: skip
    except (KeyError, TypeError):
        return False


def _validation_problems(record: Any, candidate: int, root: Path, repo: Path,
                         settings: Mapping[str, Any] | None, *,
                         k: int = 1) -> list[str]:  # fmt: skip
    """Every problem of a validation fixation, the k-th link of its chain (module docstring,
    "What each reader checks")."""
    check = _Check(KINDS["validation"], candidate, k)
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
    reading = None
    if engineering is not None:
        found, reading = _recomputed(record, candidate, root, engineering, settings, repo)
        problems += found
    if k > 1:
        problems += _succession("validation", record, candidate, k, root, repo, reading)
    return problems


def _recomputed(record: Mapping[str, Any], candidate: int, root: Path,
                engineering: Mapping[str, Any], settings: Mapping[str, Any] | None,
                repo: Path) -> tuple[list[str], dict[str, Any] | None]:  # fmt: skip
    """The qualification recomputed under the code it counts under for the candidate
    (``_qualification_code``): without a problem, the fixed rates' states the pinned ones, and no
    input that the ledger does not pin bearing on the fixed rates (``_bears_on``). Returns the
    problems and the reading (None when there is none)."""
    in_force = _qualification_code(root, repo, candidate, engineering)
    if in_force is None:
        return [], None  # no code in force: named with the code
    reading, problems = _reading(root, candidate, in_force, settings)
    if reading is None:
        return problems, None
    pinned = record["qualification"]
    for arm, pair in _states(reading, record["rates"]).items():
        for label, state in pair.items():
            if state != pinned["states"][arm][label]:
                problems.append(f"qualification.states.{arm}.{label}: the qualification "
                                f"recomputed under the code in force gives "
                                f"{state or 'no such rate'}, not the pinned "
                                f"{pinned['states'][arm][label]}")  # fmt: skip
    entries, found = _ledger(reading, root)
    problems += found
    new = {e["path"] for e in entries} - {e["path"] for e in pinned["ledger"]}
    fixed = _fixed_folders(reading, record["rates"], root)
    added = sorted(path for path in new if _bears_on(path, candidate, fixed))
    if added:
        problems.append(f"qualification.ledger: the qualification takes now as inputs {added}, "
                        "which the record does not pin")  # fmt: skip
    return problems, reading


def _fixed_folders(reading: Mapping[str, Any], rates: Mapping[str, Mapping[str, float]],
                   root: Path) -> set[str]:  # fmt: skip
    """The case folders of the fixed rates in a qualification reading, under the runs
    directory."""
    from . import qualification_records  # it loads torch (qualification_verdicts): here only

    folders = set()
    for arm, pair in rates.items():
        for label in LABELS:
            rate = qualification_records.rate_of(reading, arm, float(pair[label]))
            for case in (rate or {}).get("cases") or []:
                try:
                    folders.add(Path(case["folder"]).relative_to(root).as_posix())
                except (KeyError, TypeError, ValueError):
                    continue
    return folders


def _bears_on(path: str, candidate: int, fixed: set[str]) -> bool:
    """Whether a record the fixation does not pin bears on the fixed rates: a record of one of
    their cases (in its folder, ``qualification/c<N>/numerical/s<seed>/<case>/``), or one of
    the candidate's that lies in no case (a supersession record: it touches every case). A
    record of another rate's case (a closed arm's, an unfixed rate's) does not."""
    parts = PurePosixPath(path).parts
    stage = ("qualification", f"c{candidate}", "numerical")
    if len(parts) >= 6 and parts[:3] == stage:
        return PurePosixPath(*parts[:5]).as_posix() in fixed
    return True


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
    """The logs the qualification rests on now against the ones pinned. A torn line the launcher
    refuses is named by its log and line with the command that resolves it (``FIXED_TORN``),
    and, in a log the fixation pins, with why the fixation breaks (``PINNED_TORN``): the builder's
    "before the fixation" no longer applies once the fixation exists."""
    logs, problems, refused = _qualification_logs(root, candidate)
    named = {e["path"] for e in pinned}
    found = [f"qualification.logs: {p}" for p in problems]
    found += [f"qualification.logs: {where} line {number}: {FIXED_TORN}"
              + (f"; {PINNED_TORN}" if where in named else "")
              for where, number in refused]  # fmt: skip
    now = {e["path"] for e in logs}
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


def _qualification_code(root: Path, repo: Path, candidate: int,
                        engineering: Mapping[str, Any]) -> str | None:  # fmt: skip
    """The code the candidate's qualification counts under (``study3_progress.
    qualification_code``): the old code of the candidate's valid cycle record outside the
    qualification when it reuses the qualification, the code in force otherwise."""
    from . import qualification_records  # it loads torch (qualification_verdicts): here only
    from .study3_cycle import read_cycle
    from .study3_progress import Records, qualification_code

    cycles = {c: qualification_records.supersessions(root, c) for c in (1, 2) if c <= candidate}
    records = Records(decisions={1: {"engineering": {"state": "passed", "record": engineering}}},
                      cycles=cycles,
                      cycle_records={candidate: read_cycle(root, candidate,
                                                           repo=repo)})  # fmt: skip
    return qualification_code(records, candidate)


def _reading(root: Path, candidate: int, expected_code: str,
             settings: Mapping[str, Any] | None) -> Reading:  # fmt: skip
    """The candidate's qualification recomputed from its records under ``expected_code``
    (``read_qualification``), and its problems (a reading with a problem decides nothing)."""
    from . import qualification_records  # it loads torch (qualification_verdicts): here only

    if settings is None:
        settings = load_config("experiments/qualification")
    try:
        reading = qualification_records.read_qualification(
            root, candidate, expected_code=expected_code, settings=settings)  # fmt: skip
    except (OSError, ValueError) as exc:
        return None, [f"qualification: it cannot be read ({type(exc).__name__}: {exc})"]
    return reading, [f"qualification: {p}" for p in reading["problems"]]


def _states(reading: Mapping[str, Any], rates: Mapping[str, Mapping[str, float]]) -> dict:
    """The fixed rates' states in a qualification reading (None for a rate it does not name):
    ``admitted`` for a ``failing`` rate whose eligibility is ``qualified`` (amendment 10), its
    state otherwise."""
    from . import qualification_records  # it loads torch (qualification_verdicts): here only

    states: dict[str, dict[str, Any]] = {}
    for arm, pair in rates.items():
        states[arm] = {}
        for label in LABELS:
            rate = qualification_records.rate_of(reading, arm, float(pair[label]))
            states[arm][label] = None if rate is None else (
                ADMITTED if rate["state"] == "failing" and rate.get("eligibility") == QUALIFIED
                else rate["state"])  # fmt: skip
    return states


def _ledger(reading: Mapping[str, Any], root: Path) -> tuple[list[dict[str, str]], list[str]]:
    """The record files the reading took as inputs (``inputs``), each ``{"path", "sha256"}``
    under the runs directory with its sha256 now, in the order of their paths."""
    entries, problems = [], []
    for name in reading.get("inputs") or []:
        path = Path(name)
        try:
            relative = path.relative_to(root)
        except ValueError:
            problems.append(f"qualification.ledger: {path} is not under the runs directory")
            continue
        if _under_sealed(path, root):
            problems.append(f"qualification.ledger: {_sealed_problem(path)}")
            continue
        try:
            entries.append({"path": relative.as_posix(), "sha256": _digest(path)})
        except OSError as exc:
            problems.append(f"qualification.ledger: {path} is unreadable ({type(exc).__name__})")
    return sorted(entries, key=lambda e: e["path"]), problems


def _confirmation_problems(record: Any, candidate: int, root: Path, repo: Path,
                           settings: Mapping[str, Any] | None, *,
                           k: int = 1) -> list[str]:  # fmt: skip
    """Every problem of a confirmation supplement, the k-th link of its chain (module docstring,
    "What each reader checks")."""
    check = _Check(KINDS["confirmation"], candidate, k)
    check.confirmation(record)
    if check.problems:
        return check.problems
    problems = _evidence(check.refs, root, repo)
    fixation, found = read_validation_fixation(candidate, runs_root=root, repo=repo,
                                               settings=settings)  # fmt: skip
    if fixation is None:
        problems.append(_no_fixation(candidate, found))
    in_force = fixation_chain("validation", candidate, repo=repo)
    if in_force and record["validation_fixation"]["path"] != in_force[-1]:
        problems.append(f"validation_fixation: is {record['validation_fixation']['path']}, not "
                        f"the validation fixation in force, {in_force[-1]}")  # fmt: skip
    kind = f"n-c{candidate}"
    n, found, named = _in_force_record(root, kind, repo)
    problems += [f"n: {p}" for p in _passed(kind, n, found)]
    if named is not None and record["n"]["path"] != named:
        problems.append(f"n: is {record['n']['path']}, not the n record in force, {named}")
    if n is not None:
        problems += [f"{name} {record[name]} is not the {kind} record's {n[name]}"
                     for name in ("N", "cap") if record[name] != n[name]]  # fmt: skip
        if fixation is not None:
            # once the confirmation has started, the n record keeps P as N was decided: a P
            # closed since is fixed by an earlier link of the chain alone
            frozen = dropped_rates(candidate, repo) if confirmation_started(root, candidate) else {}
            problems += _against_fixation(n, fixation, kind, frozen)
        simulation, _ = read_decision(root, "simulation", repo=repo)
        problems += _budget_problems(record, n, simulation, root, repo, candidate)
    # the supplement's code is in force for the candidate, as the valid validation fixation's is:
    # both are one GPU code
    engineering, _ = read_decision(root, "engineering", repo=repo)
    problems += _code_problems(record["code"], candidate, root, repo, engineering)
    if k > 1:
        problems += _succession("confirmation", record, candidate, k, root, repo)
    return problems


def _in_force_record(
    root: Path, kind: str, repo: Path
) -> tuple[dict[str, Any] | None, list[str], str | None]:
    """The link in force of an n or budget record's chain (``study3_decisions.decisions``): its
    record (None unless valid), its problems and its path under RUNS_DIR (None when there is
    none)."""
    family, candidate = kind_of(kind)
    entry = decisions(root, candidate, repo=repo).get(kind)
    if entry is None or entry["state"] == "missing":
        return None, [], None
    named = "/".join((*DIRECTORY, Path(entry["path"]).name))
    return entry["record"], list(entry["problems"]), named


def _no_fixation(candidate: int, problems: Sequence[str]) -> str:
    name = fixation_name("validation", candidate)
    return f"validation_fixation: the validation fixation {name} is " + (
        "missing" if not problems else f"not valid ({problems[0]})")  # fmt: skip


def _against_fixation(n: Mapping[str, Any], fixation: Mapping[str, Any], kind: str,
                      frozen: Mapping[str, Any] | None = None) -> list[str]:  # fmt: skip
    """The n record of the fixed L, rates (for the arms it names) and GPU code; ``frozen``: the
    rates of an arm the fixation in force no longer fixes (``dropped_rates``: P, closed after the
    confirmation started), which a frozen n record keeps as the earlier link fixed them."""
    found = []
    if not math.isclose(n["level"], fixation["level"], rel_tol=1e-12):
        found.append(f"n: the {kind} record's L {n['level']} is not the fixed {fixation['level']}")
    for arm, pair in n["rates"].items():
        fixed = fixation["rates"].get(arm) or (frozen or {}).get(arm)
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
        if record["budget"] is None:
            return []
        return [f"budget: is named, but N = {n['N']} needs no budget decision: it is null"]
    if record["budget"] is None:
        return [f"budget: is null, but a budget decision is needed ({reason}): the {kind} record"]
    budget, found, named = _in_force_record(root, kind, repo)
    if budget is None:
        return [f"budget: {p}" for p in _passed(kind, budget, found)]
    if record["budget"]["path"] != named:
        return [f"budget: is {record['budget']['path']}, not the budget record in force, {named}"]
    if budget["decision"] != "proceed":
        return [f"budget: the {kind} record decides {budget['decision']}: the confirmation does "
                "not start"]  # fmt: skip
    return []


# ------------------------------------------------------------------ the builders


def validation_fixation(
    runs_root: str | Path,
    candidate: int,
    *,
    code: Mapping[str, str],
    repo: str | Path = REPO_ROOT,
    theta0: str | Path | None = None,
    settings: Mapping[str, Any] | None = None,
    policy_commit: str | None = None,
    written_at: str | None = None,
    supersedes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Candidate N's validation fixation (module docstring) from the records under ``runs_root``
    and the checkout ``repo``. The active arms and their rates are the stage's: ``progress``
    (``theta0`` and ``settings`` passed to it) at candidate N's validation, the step's cells when
    it carries them; ``code``: the ``commit`` it is written at and that commit's GPU code
    (``gpu_code_sha256``), the code in force for the candidate's runs, under which the stage is read
    and the qualification recomputed (``read_qualification``; ``settings``:
    configs/experiments/qualification.yaml as loaded, by default this checkout's);
    ``policy_commit``: the registration's commit (``registered_policy``). ``RecordRefused`` while
    the study is not at candidate N's validation, P is fixed while P–F is closed or not fixed while
    it is open, a cell's rate is not the stage's, a fixed rate is not qualified, the simulation or
    engineering record is not there, not valid or not passed, or for any problem its reader would
    name; and once the record is written. ``supersedes``: the successor of the fixation in force
    after the candidate's cycle (module docstring, "Successors of a fixation"), with the same
    candidate and A's and F's rates (P's while P–F is open), written as the chain's next link."""
    _candidate(candidate)
    root, repo = Path(runs_root), Path(repo)
    k = _next_link("validation", candidate, repo, supersedes)
    simulation, found = read_decision(root, "simulation", repo=repo)
    problems = _passed("simulation", simulation, found)
    engineering, found = read_decision(root, "engineering", repo=repo)
    problems += _passed("engineering", engineering, found)
    check = _Check(KINDS["validation"], candidate)
    check.code(code)
    problems += check.problems
    if problems:  # no stage is read without its decision records and a code
        raise RecordRefused(_refusal(KINDS["validation"], candidate, problems))
    if settings is None:
        settings = load_config("experiments/qualification")
    rates, problems = _stage_rates(root, candidate, code, repo, theta0, settings)
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
    states: dict[str, dict[str, Any]] = {}
    ledger: list[dict[str, str]] = []
    if rates is not None:
        reading_code = _qualification_code(root, repo, candidate, engineering) or code[
            "gpu_code_sha256"]  # fmt: skip
        reading, found = _reading(root, candidate, reading_code, settings)
        problems += found
        if reading is not None:
            states = _states(reading, rates)
            problems += [f"qualification: {arm}'s {label} {rates[arm][label]:g} is "
                         f"{state or 'not in the qualification'}, neither qualified nor "
                         "admitted (amendment 10)"
                         for arm, pair in states.items() for label, state in pair.items()
                         if state not in (QUALIFIED, ADMITTED)]  # fmt: skip
            ledger, found = _ledger(reading, root)
            problems += found
            if not ledger:
                problems.append("qualification.ledger: the qualification took no record as input")
    if problems:
        raise RecordRefused(_refusal(KINDS["validation"], candidate, problems))
    model, judge = instruments()
    record = {
        "kind": KINDS["validation"], "candidate": candidate, "written_at": written_at or _now(),
        "policy": policy, "rates": rates,
        "qualification": {"logs": logs, "ledger": ledger, "states": states},
        "simulation": {"path": SIMULATION_RECORD,
                       "sha256": _pin(decision_path(root, "simulation"), root)},
        "validation_blocks": simulation["validation_blocks"], "level": simulation["level"],
        "engineering": {"path": ENGINEERING_RECORD,
                        "sha256": _pin(decision_path(root, "engineering"), root)},
        "model": model, "judge": judge,
        "code": {"commit": code["commit"], "gpu_code_sha256": code["gpu_code_sha256"]},
        **({} if supersedes is None else {"supersedes": dict(supersedes)}),
    }  # fmt: skip
    record = _plain(record, KINDS["validation"], candidate)
    problems = _validation_problems(record, candidate, root, repo, settings, k=k)
    if problems:
        raise RecordRefused(_refusal(KINDS["validation"], candidate, problems))
    return record


Fixed = tuple[dict[str, dict[str, float]] | None, list[str]]  # fixed rates (None: none), problems


def _stage_rates(root: Path, candidate: int, code: Mapping[str, str], repo: Path,
                 theta0: str | Path | None, settings: Mapping[str, Any]) -> Fixed:  # fmt: skip
    """The active arms and their rates at candidate N's validation, as ``progress`` gives them
    (module docstring, "Rulings"), or why there are none."""
    from . import study3_progress

    try:
        state = study3_progress.progress(root, repo=repo, expected_code=code["gpu_code_sha256"],
                                         theta0=theta0, settings=settings)  # fmt: skip
    except (OSError, ValueError) as exc:
        return None, [f"stage: the study's state cannot be read ({type(exc).__name__}: {exc})"]
    if state["problems"]:
        return None, [f"stage: {p}" for p in state["problems"]]
    if (state["stage"], state["candidate"]) != ("validation", candidate):
        return None, [f"stage: the study is at {state['stage']!r} of candidate "
                      f"{state['candidate']} ({state['step'].get('kind')}), not at candidate "
                      f"{candidate}'s validation, where its validation fixation is "
                      "written"]  # fmt: skip
    return _fixed_rates(state)


def _fixed_rates(state: Mapping[str, Any]) -> Fixed:
    """The fixed rates of a stage at validation: its active arms and their two rates; the step's
    cells, when it carries them, each at one of its arm's two rates."""
    active, rates = state["active"], state["rates"]
    arms, open_pf = list(active["arms"]), "P-F" in active["comparisons"]
    problems = []
    if "P" in arms and not open_pf:
        problems.append("stage: P is active while P–F is closed: P is fixed only while P–F is open")
    if "P" not in arms and open_pf:
        problems.append("stage: P–F is open and P is not active: P is fixed while P–F is open")
    if not {"A", "F"} <= set(arms) or not set(arms) <= set(ARMS):
        problems.append(f"stage: the active arms {arms} are not A and F (and P)")
    fixed: dict[str, dict[str, float]] = {}
    for arm in (a for a in ARMS if a in arms):
        pair = rates.get(arm) or {}
        values = [pair.get(k) for k in ("l", "u")]
        if not all(isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
                   and v > 0 for v in values):  # fmt: skip
            problems.append(f"stage: {arm}'s rates {pair} are not its two rates")
            continue
        fixed[arm] = {"lr_l": float(values[0]), "lr_u": float(values[1])}
    step = state["step"]
    if step.get("kind") == "campaign" and step.get("campaign") == "validation":
        for arm, rate, seed in step.get("cells") or []:
            pair = fixed.get(arm)
            if pair is None:
                problems.append(f"stage: the validation step issues {arm} at {rate:g} on {seed}, "
                                "an arm that is not fixed")  # fmt: skip
            elif not any(math.isclose(rate, pair[k], rel_tol=RATE_TOLERANCE) for k in LABELS):
                problems.append(f"stage: the validation step issues {arm} at {rate:g} on {seed}, "
                                f"not one of the stage's rates of {arm} "
                                f"({pair['lr_l']:g}, {pair['lr_u']:g})")  # fmt: skip
    return (None if problems else fixed), problems


def confirmation_fixation(
    runs_root: str | Path,
    candidate: int,
    *,
    code: Mapping[str, str],
    bootstrap_seed: int,
    audit_seed: int,
    agreement_seed: int,
    repo: str | Path = REPO_ROOT,
    settings: Mapping[str, Any] | None = None,
    written_at: str | None = None,
    supersedes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Candidate N's confirmation supplement (module docstring): its validation fixation, its n
    record and, when N needs one, its budget record, from the records under ``runs_root`` and the
    checkout ``repo``; the bootstrap's seed, the audit draw's seed and the seed of the draw of
    agreements the researcher checks (integers in [0, 2³²)), with the registered resamples and
    draw; ``code``: the commit it is written at and that commit's GPU code. ``RecordRefused``
    while the validation fixation is not valid, the n record is not there, not valid or not passed,
    a needed budget record is not there or does not proceed, a seed is not one, or for any problem
    its reader would name; and once the record is written. It names the validation fixation and
    the n and budget records in force. ``supersedes``: the successor of the supplement in force
    after the candidate's cycle (module docstring, "Successors of a fixation"): once the
    confirmation has started, with the same N, cap and seeds."""
    _candidate(candidate)
    root, repo = Path(runs_root), Path(repo)
    k = _next_link("confirmation", candidate, repo, supersedes)
    fixation, found = read_validation_fixation(candidate, runs_root=root, repo=repo,
                                               settings=settings)  # fmt: skip
    problems = [] if fixation is not None else [_no_fixation(candidate, found)]
    kind = f"n-c{candidate}"
    n, found, n_path = _in_force_record(root, kind, repo)
    problems += [f"n: {p}" for p in _passed(kind, n, found)]
    check = _Check(KINDS["confirmation"], candidate)
    check.code(code)
    for where, seed in (("bootstrap.seed", bootstrap_seed), ("audit.seed", audit_seed),
                        ("audit.agreement_seed", agreement_seed)):  # fmt: skip
        check.seed(where, seed)
    problems += check.problems
    budget = None
    if n is not None and n["state"] == "passed":
        simulation, _ = read_decision(root, "simulation", repo=repo)
        reason = budget_reason(n["N"], simulation)
        if reason is not None:
            answer = f"budget-c{candidate}"
            entry, found, named = _in_force_record(root, answer, repo)
            if entry is None:
                problems.append(f"budget: N needs a budget decision ({reason}), and "
                                + _passed(answer, entry, found)[0])  # fmt: skip
            elif entry["decision"] != "proceed":
                problems.append(f"budget: the {answer} record decides {entry['decision']}: the "
                                "confirmation does not start")  # fmt: skip
            else:
                budget = {"path": named, "sha256": _pin(root / named, root)}
    if problems:
        raise RecordRefused(_refusal(KINDS["confirmation"], candidate, problems))
    name = fixation_chain("validation", candidate, repo=repo)[-1]
    record = {
        "kind": KINDS["confirmation"], "candidate": candidate, "written_at": written_at or _now(),
        "validation_fixation": {"path": name, "sha256": _digest(repo / name), "root": "repo"},
        "n": {"path": n_path, "sha256": _pin(root / n_path, root)},
        "N": n["N"], "cap": n["cap"], "budget": budget,
        "bootstrap": {"seed": bootstrap_seed, "resamples": BOOTSTRAP_RESAMPLES},
        "audit": {"seed": audit_seed, "agreement_seed": agreement_seed,
                  "procedure": AUDIT_PROCEDURE},
        "code": {"commit": code["commit"], "gpu_code_sha256": code["gpu_code_sha256"]},
        **({} if supersedes is None else {"supersedes": dict(supersedes)}),
    }  # fmt: skip
    record = _plain(record, KINDS["confirmation"], candidate)
    problems = _confirmation_problems(record, candidate, root, repo, settings, k=k)
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
    settings: Mapping[str, Any] | None = None,
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
    elif _under_sealed(repo / previous, repo):
        problems.append(f"{own} supersedes {previous}: {_sealed_problem(repo / previous)}")
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
        elif _under_sealed(repo / name, repo):
            problems.append(f"fixation: {_sealed_problem(repo / name)}")
        elif not ((repo / name).is_file() and tracked(name, repo)):
            problems.append(f"fixation: {name} is not a committed file of the checkout")
        else:
            step, number = _FIXATION_NAME.fullmatch(name).group(1, 2)
            in_force = fixation_chain(step, int(number), repo=repo)[-1]
            if name != in_force:
                problems.append(f"fixation: {name} is not the fixation in force, {in_force}: a "
                                "new versions record names its chain's last link")  # fmt: skip
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
        problems = _pins(record, repo, _Fixations(root, repo, settings))
    if problems:
        raise RecordRefused(_refusal("versions", None, problems))
    return record


def _next_link(step: str, candidate: int, repo: Path,
               supersedes: Mapping[str, Any] | None) -> int:  # fmt: skip
    """The number of the record a builder writes: the first, or with ``supersedes`` the chain's
    next link; ``RecordRefused`` once it is written."""
    names = fixation_chain(step, candidate, repo=repo)
    k = 1 if supersedes is None else (fixation_number(names[-1]) + 1 if names else 2)
    _not_written(repo / fixation_name(step, candidate, k))
    return k


def _pin(path: Path, base: Path) -> str:
    """The sha256 of a record a fixation pins, the sealed-path rule applied again where it is read
    (below ``base``): ``RecordRefused`` for one that resolves under a sealed folder now, never
    opened."""
    if _under_sealed(path, base):
        raise RecordRefused(_sealed_problem(path))
    return _digest(path)


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
    ``repo`` (a relative ``path`` is read from it); absent: (None, []). ``versions_problems``
    checks it against the checkout and the journal."""
    repo = Path(repo)
    path = _from(path, repo)
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
    """The fixation records a check names, each read once: in full by its reader (the record
    checked), or as study 2 checks an earlier record's (committed, of the pinned sha256); and the
    rates each fixes."""

    def __init__(self, root: Path, repo: Path, settings: Mapping[str, Any] | None = None) -> None:
        self.root, self.repo, self.settings = root, repo, settings
        self.full_seen: dict[tuple[str, str], tuple[dict[str, Any] | None, list[str]]] = {}
        self.rates_seen: dict[tuple[str, str], tuple[dict[str, Any] | None, list[str]]] = {}

    def light(self, ref: Mapping[str, Any]) -> list[str]:
        """The fixation file committed, and of the sha256 the record pins (study 2's check of a
        record's fixation)."""
        name = ref["path"]
        path, where = self.repo / name, f"fixation {name}"
        if _under_sealed(path, self.repo):
            return [_sealed_problem(path)]
        if not path.is_file():
            return [f"{where} is missing"]
        problems = [] if tracked(name, self.repo) else [f"{where} is not committed"]
        now = _digest(path)
        if now != ref["sha256"]:
            problems.append(f"{where} has sha256 {now} now, not the named {ref['sha256']}")
        return problems

    def full(self, ref: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        """The fixation as its reader reads it, after the light check: in full when it is its
        chain's link in force; an earlier link for its integrity alone (the chain's), its
        content as pinned."""
        key = (ref["path"], ref["sha256"])
        if key not in self.full_seen:
            step, candidate, _ = _FIXATION_NAME.fullmatch(ref["path"]).groups()
            problems = self.light(ref)
            names = fixation_chain(step, int(candidate), repo=self.repo)
            if names and ref["path"] != names[-1]:
                problems += _chain_problems(names, self.repo)
                record = _pinned_json(self.repo / ref["path"], self.repo, ref["sha256"])
            else:
                record, found = _read_fixation(step, int(candidate), self.root, self.repo,
                                               self.settings)  # fmt: skip
                problems += found
            self.full_seen[key] = ((None if problems else record), problems)
        return self.full_seen[key]

    def rates(self, ref: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        """The rates the fixation fixes, per active arm: a validation fixation's ``rates``; a
        supplement's, those of the n record it pins (the arms open at confirmation). None, with
        why, when they cannot be shown; nothing is said of a fixation file that is not the pinned
        one (``light`` names it)."""
        key = (ref["path"], ref["sha256"])
        if key not in self.rates_seen:
            self.rates_seen[key] = self._rates(ref)
        return self.rates_seen[key]

    def _rates(self, ref: Mapping[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        step = _FIXATION_NAME.fullmatch(ref["path"]).group(1)
        data = _pinned_json(self.repo / ref["path"], self.repo, ref["sha256"])
        if data is None:
            return None, []
        where = f"fixation {ref['path']}"
        if step == "confirmation":
            n = data.get("n") if isinstance(data, dict) else None
            if not (isinstance(n, Mapping) and isinstance(n.get("path"), str)
                    and isinstance(n.get("sha256"), str)):  # fmt: skip
                return None, [f"{where}: its n record cannot be read: its fixed arms cannot be "
                              "shown"]  # fmt: skip
            data = _pinned_json(self.root / n["path"], self.root, n["sha256"])
            if data is None:
                return None, [f"{where}: the n record {n['path']} it pins is not as pinned: its "
                              "fixed arms cannot be shown"]  # fmt: skip
        rates = data.get("rates") if isinstance(data, dict) else None
        check = _Check("rates")
        if check.rates(rates) is None:
            return None, [f"{where}: its fixed rates cannot be read"]
        return rates, []


def _pinned_json(path: Path, base: Path, sha256: str) -> Any:
    """The JSON content of ``path`` when its bytes are the pinned ones; None otherwise (sealed,
    missing, another sha256, unreadable)."""
    if _under_sealed(path, base) or not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if hashlib.sha256(data).hexdigest() != sha256:
        return None
    try:
        return json.loads(data)
    except ValueError:
        return None


def _pins(record: Mapping[str, Any], repo: Path, fixations: _Fixations) -> list[str]:
    """A versions record's fixation: committed, of the named sha256 now and at the recorded
    commit, valid (its full reader), and of the recorded commit's GPU code."""
    ref = record["fixation"]
    if ref is None:
        return []
    fixation, found = fixations.full(ref)
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
    settings: Mapping[str, Any] | None = None,
) -> list[str]:
    """How the checkout ``repo`` and the launch journal under ``runs_root`` depart from the
    versions record at ``path`` (a relative path is read from the checkout; module docstring,
    "The journal check"); none: it holds. With ``step`` (validation or confirmation) and
    ``candidate``, the record must also stand for a run of that step and candidate; ``ValueError``
    for another step or candidate. ``settings``: configs/experiments/qualification.yaml as loaded
    (by default this checkout's), under which the record's fixation recomputes the
    qualification."""
    if step is not None or candidate is not None:
        _stage(step)
        _candidate(candidate)
    repo, root = Path(repo), Path(runs_root)
    path = _from(path, repo)
    if not path.is_file():
        return [f"{path} is missing: write the versions record (scripts/study3_summary.py "
                "versions) and commit it first"]  # fmt: skip
    if _under_sealed(path, repo):
        return [_sealed_problem(path)]
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
    fixations = _Fixations(root, repo, settings)
    problems += [f"{own}: {p}" for p in _pins(record, repo, fixations)]
    chain, found = _chain(own, record, hashlib.sha256(data).hexdigest(), repo)
    problems += found
    problems += _runs_problems(root, chain, fixations)
    if step is not None:
        why = _stands_for(record["fixation"], step, candidate, repo=repo)
        if why is not None:
            problems.append(f"{own}: {why}")
    return list(dict.fromkeys(problems))


def _checkout_problems(record: Mapping[str, Any], own: str, path: Path, repo: Path,
                       committed: bool) -> list[str]:  # fmt: skip
    """HEAD against the recorded commit, as study 2 reads it (its record-only child), the
    checkout's changes, and the record unchanged since it was added."""
    commit, problems = record["code"]["commit"], []
    head = git_state(repo).get("commit")
    changes = _head_changes(commit, repo)
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
            problems.append(f"{own} changed after it was added at {added}: a versions record is "
                            "never overwritten (a later one is a new file that supersedes "
                            "it)")  # fmt: skip
    return problems


def _head_changes(commit: str, repo: Path) -> dict[str, list[str]] | None:
    """How the checkout departs from ``commit``: ``committed``, the paths changed between it and
    HEAD (``committed_changes``), and the uncommitted and untracked files
    (``tracking.worktree_changes``); None when HEAD is neither ``commit`` nor a descendant of it
    (or ``repo`` is no checkout)."""
    committed = committed_changes(commit, "HEAD", repo)
    if committed is None:
        return None
    try:
        return {"committed": committed, **worktree_changes(repo)}
    except (OSError, subprocess.CalledProcessError):
        return None


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
        if _under_sealed(now, repo):
            problems.append(f"{previous}: {_sealed_problem(now)}")
        elif not (now.exists() or now.is_symlink()):
            problems.append(f"{previous} was deleted after it was added at {added}: a versions "
                            "record is kept as it was written")  # fmt: skip
        elif not now.is_file() or _digest(now) != then:
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


def _runs_problems(root: Path, chain: Sequence[_Link], fixations: _Fixations) -> list[str]:
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
    pinned = {chain[0].name: []}  # the record's own fixation is read in full already
    if journal is not None:
        if journal.corrupt:
            problems.append(f"the launch journal has {TORN_LINE} ({', '.join(journal.corrupt)}): "
                            "a start it held cannot be shown (fails closed)")  # fmt: skip
        problems += [f"{where}: {sealed_record('a launch log', root / where, root)}: a start it "
                     "held cannot be shown (fails closed)" for where in journal.sealed]  # fmt: skip
        files = _Files(root / EXPERIMENT)
        for campaign, start in journal.starts:
            run_id = start.get("run_id")
            if isinstance(run_id, str):
                started.add(run_id)
            why = _start_problem(root, campaign, start, chain, fixations, pinned, files)
            if why is not None:
                named[(campaign, why)].append(f"{start.get('run')} ({run_id})")
    for (campaign, why), runs in sorted(named.items()):
        more = f" and {len(runs) - 1} other run(s)" if len(runs) > 1 else ""
        problems.append(f"{campaign}: {runs[0]}{more}: {why}")
    for name, found in pinned.items():
        problems += [f"{name}: {p}" for p in found]
    return problems + _unstarted(root, started)


def _start_problem(root: Path, campaign: str, start: Mapping[str, Any], chain: Sequence[_Link],
                   fixations: _Fixations, pinned: dict[str, list[str]],
                   files: _Files) -> str | None:  # fmt: skip
    """Why a study-3 start is not a run of the chain's records that its record stands for, at a
    rate its fixation fixes (None: it is, or it is no validation or confirmation run). The
    fixation of each earlier record a run is under is checked once, into ``pinned``."""
    header = start.get("header")
    if not (isinstance(header, Mapping) and header.get("study") == 3
            and isinstance(header.get("step"), str) and header["step"] in STEP_ROLES):  # fmt: skip
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
    ref = link.record["fixation"]
    why = _stands_for(ref, step, candidate)
    if why is not None:
        return f"its record {link.name}: {why}"
    if link.name not in pinned:  # an earlier record: as study 2 checks it
        pinned[link.name] = fixations.light(ref)
    fixed, found = fixations.rates(ref)
    pinned[link.name] += [p for p in found if p not in pinned[link.name]]
    if fixed is None:
        return None  # why its rates cannot be shown is named with its record
    if not isinstance(start.get("run_id"), str):
        return "its start names no run id: its arm and rate cannot be bound"
    if not plain_name(start["run_id"]):
        return (f"its run id {start['run_id']!r} is not a plain name: it names no run folder, "
                "and nothing under it is opened")  # fmt: skip
    cell = _bind(root, campaign, start, {}, twice=False, files=files).cell
    if cell.arm is None or cell.rate is None:
        return ("its arm and rate cannot be bound from its records: whether its fixation fixes "
                "them cannot be shown")  # fmt: skip
    if cell.arm not in fixed:
        return (f"its arm {cell.arm} is not active in its fixation {ref['path']} (active: "
                f"{sorted(fixed)})")  # fmt: skip
    pair = fixed[cell.arm]
    if not any(math.isclose(cell.rate, pair[k], rel_tol=RATE_TOLERANCE) for k in LABELS):
        return (f"its rate {cell.rate:g} of {cell.arm} is not one of the two its fixation "
                f"{ref['path']} fixes ({pair['lr_l']:g}, {pair['lr_u']:g})")  # fmt: skip
    return None


def _names(named: str, name: str) -> bool:
    """Whether a header's path names the record ``name`` (a checkout's ``docs/<file>``): by its
    last components, so that a path from another checkout names it."""
    parts = PurePosixPath(name).parts
    return PurePosixPath(named.replace("\\", "/")).parts[-len(parts) :] == parts


def _stands_for(fixation: Mapping[str, Any] | None, step: str, candidate: int, *,
                repo: Path | None = None) -> str | None:  # fmt: skip
    """Why a record whose ``fixation`` is this cannot stand for a run of ``step`` and
    ``candidate`` (None: it can): its fixation is a link of the candidate's chain of that step
    (a run keeps the record it started under); with ``repo`` (the check before a launch), the
    link in force there."""
    wanted = fixation_name(step, candidate)
    what = "validation fixation" if step == "validation" else "confirmation supplement"
    if fixation is None:
        return (f"its fixation is null: a versions record without a fixation stands for no "
                f"validation or confirmation run (a {step} run of candidate {candidate} needs "
                f"{wanted}, its candidate's {what})")  # fmt: skip
    found = _FIXATION_NAME.fullmatch(fixation["path"])
    if found is None or found.group(1) != step or int(found.group(2)) != candidate:
        return (f"its fixation is {fixation['path']}, not {wanted}: a {step} run of candidate "
                f"{candidate} needs its candidate's {what}")  # fmt: skip
    names = [] if repo is None else fixation_chain(step, candidate, repo=repo)
    if names and fixation["path"] != names[-1]:
        return (f"its fixation is {fixation['path']}, not the one in force, {names[-1]}: a "
                f"{step} of candidate {candidate} is launched under its candidate's {what} in "
                "force")  # fmt: skip
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
        if folder.name in RESERVED or folder.name in started or not folder.is_dir():
            continue
        if _sealed_records(folder, experiment) is not None:  # nothing of it is opened
            found["a run whose record files resolve under a sealed folder or cannot be resolved "
                  "(never opened)"].append(folder.name)  # fmt: skip
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
    responses or observations folder is not opened (the rule, with the run's folder as the
    base)."""
    config, meta = folder / "config.yaml", folder / "meta.json"
    if config.is_file() and not _sealed_file(config):
        try:
            role = OmegaConf.select(OmegaConf.load(config), "study3.role", default=None)
        except Exception:  # an unreadable config tells no role
            role = None
        if isinstance(role, str):
            return role
    if meta.is_file() and not _sealed_file(meta):
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


# ------------------------------------------------------------------ each run's own commit


def uncounted_runs(
    endpoints: Iterable[Endpoint], *, repo: str | Path = REPO_ROOT
) -> dict[str, dict[str, Any]]:
    """The validation and confirmation runs that do not count for the commit they ran ("Voiding
    and supersession", Versions: a run under a commit other than the pinned one does not count;
    it is kept and reported), by run id. A run counts only under the commit the versions record
    its campaign header names pins (``code.commit``), or a record-only descendant of it whose
    changes touch that record alone (``committed_changes``, as the running HEAD is read), from a
    clean checkout. Checked: every attempt of a validation or confirmation cell that counts
    (``study3_records.COUNTING``) and whose ``meta.json`` was read (``Endpoint.git``); an attempt
    without one has recorded no commit yet (it has not started), and one that no longer counts is
    listed by the readers already. Each entry: ``run_id``, ``label``, ``candidate``, ``step``,
    ``reason`` (``UNCOUNTED_REASONS``: ``no_commit``, its ``meta.json`` records no commit;
    ``dirty_checkout``, its checkout was dirty; ``dirty_unrecorded``, its ``meta.json`` does not
    record whether it was, which the tracker always records; ``unknown_commit``, a commit the
    checkout ``repo`` does not hold; ``no_versions_record``, the record its header names is not in
    the checkout with the named sha256, so the pinned commit cannot be shown; ``other_commit``,
    neither the pinned commit nor a record-only descendant of it), ``commit`` as recorded (None
    unless it is a commit id: no free text of a record is copied) and ``dirty``,
    ``versions_record`` (the record's name in the checkout, or as the header names it) and
    ``why``. Only the record's ``code.commit`` is read here; ``versions_problems`` checks the
    record itself. Such a run is a technical mark outside the qualification of its arm
    (``study3_progress``, "A run under another commit"): a recovery round where the launcher
    would open one, or its closure record, its reason this one."""
    repo = Path(repo)
    pinned: dict[tuple[Any, Any], tuple[str | None, str | None]] = {}
    known: dict[str, bool] = {}
    changes: dict[tuple[str, str], list[str] | None] = {}
    out: dict[str, dict[str, Any]] = {}
    for e in endpoints:
        if e.cell.step not in STAGES or e.state not in COUNTING or e.git is None:
            continue
        commit, dirty = e.git
        named, sha = e.cell.versions_record, e.cell.versions_record_sha256
        if (named, sha) not in pinned:
            pinned[(named, sha)] = _pinned_commit(named, sha, repo)
        name, base = pinned[(named, sha)]
        if commit is not None and commit not in known:
            known[commit] = _known(repo, commit)
        if (commit is not None and base is not None and known[commit]
                and (base, commit) not in changes):  # fmt: skip
            changes[(base, commit)] = committed_changes(base, commit, repo)
        found = _commit_problem(commit, dirty, name or named, sha, base,
                                known.get(commit, False), changes.get((base, commit)))  # fmt: skip
        if found is not None:
            reason, why = found
            shown = commit if commit is not None and _COMMIT.fullmatch(commit) else None
            out[e.run_id] = {"run_id": e.run_id, "label": e.label, "candidate": e.candidate,
                             "step": e.cell.step, "reason": reason, "commit": shown,
                             "dirty": dirty, "versions_record": name or named,
                             "why": why}  # fmt: skip
    return out


def _pinned_commit(named: Any, sha: Any, repo: Path) -> tuple[str | None, str | None]:
    """The versions record a campaign header names (``versions_record``, a path in this checkout
    or another, read by its last two components) as its name in the checkout, and the commit it
    pins (``code.commit``), when the file is there with the named sha256; None for each that
    cannot be shown. A file that resolves under a sealed folder is not opened."""
    if not (isinstance(named, str) and isinstance(sha, str)):
        return None, None
    name = "/".join(PurePosixPath(named.replace("\\", "/")).parts[-2:])
    if not _VERSIONS_NAME.fullmatch(name):
        return None, None
    data = _pinned_json(repo / name, repo, sha)
    code = data.get("code") if isinstance(data, dict) else None
    commit = code.get("commit") if isinstance(code, Mapping) else None
    return name, commit if isinstance(commit, str) and _COMMIT.fullmatch(commit) else None


def _commit_problem(commit: str | None, dirty: bool | None, record: Any, sha: Any,
                    base: str | None, known: bool,
                    changes: list[str] | None) -> tuple[str, str] | None:  # fmt: skip
    """Why a run under ``commit`` (``dirty``) does not count against the commit ``base`` that the
    versions record ``record`` pins (``changes``: the paths changed from ``base`` to ``commit``),
    as (reason, why); None when it counts."""
    if commit is None:
        return "no_commit", "its meta.json records no commit (git.commit)"
    if not _COMMIT.fullmatch(commit):  # not copied: no free text of a record is shown
        return "unknown_commit", "its meta.json records a commit that is no commit id"
    if dirty is True:
        return "dirty_checkout", f"it ran on a checkout of {commit} with uncommitted changes"
    if dirty is None:
        return ("dirty_unrecorded", f"its meta.json does not record whether its checkout of "
                f"{commit} was clean (git.dirty)")  # fmt: skip
    if not known:
        return "unknown_commit", f"its commit {commit} is not a commit of this checkout"
    if base is None:
        return ("no_versions_record", f"the versions record its header names ({record}, sha256 "
                f"{sha}) is not in this checkout as named: the commit it pins cannot be "
                "shown")  # fmt: skip
    if changes is None:
        return ("other_commit", f"its commit {commit} is neither the commit {base} that {record} "
                "pins nor a descendant of it")  # fmt: skip
    others = [path for path in changes if path != record]
    if others:
        return ("other_commit", f"its commit {commit} changes {len(others)} files from the commit "
                f"{base} that {record} pins (first: {others[0]}): only the record itself may "
                "follow it")  # fmt: skip
    return None
