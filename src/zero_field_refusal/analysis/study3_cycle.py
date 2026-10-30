"""Study 3's engineering cycle for a technical mark outside the numerical qualification.

docs/study3-preregistration.md: "Failure routes" (one engineering cycle per candidate: diagnosis,
a fix, re-qualification of the affected cases and reruns of the affected runs; a mark still
unresolved after it routes to the technical feasibility report for A or F, and closes P–F for P)
and "Voiding and supersession" (a superseded record stays, listed with its reason; a scientific
failure keeps counting). A search, validation or confirmation run that the readers count as a
technical mark opens the cycle; analysis code records it, and ``scripts/qualify.py``'s own
journal repair and re-qualification carry it out, unchanged.

- ``STAGES``, ``STAGE_STEPS`` and ``ACTIONS``: the stages a cycle claims (the qualification's
  steps are its numerical runs and its repair), and the two claims a stage takes.
- ``parse_claims``: the claims per stage, ``STAGE=reuse|rerun[:A,F,P]``, each with its
  justification, ``STAGE=TEXT``.
- ``run_gpu_codes``: each run's GPU code, from the commit its ``meta.json`` records.
- ``journal_seen`` and ``seen_starts``: the journal prefix a cycle pins (each launch log's line
  count and the sha256 of those lines), and the starts that lie in it.

**Claims** ("Failure routes": each earlier stage is reused or rerun according to what the fix
changes). A stage is ``reuse`` (its runs keep counting under the old code) or ``rerun`` with the
arms it reruns, every arm by default; the qualification's claim is all or nothing for the
candidate, so it takes no arms. Every claim carries a justification: one line, not blank, at
most ``JUSTIFICATION_MAX`` characters. The readers check that a claim and its justification are
there, never what the justification says.

**A run's GPU code** is ``code_identity.gpu_code_identity`` of the commit its ``meta.json``
records (``Endpoint.git``), computed once per commit (``study3_versions._gpu_code``). A run of a
dirty checkout, of a checkout whose cleanness is not recorded, of a commit that is no commit id
or that the checkout cannot place has none (None): its code cannot be shown. A run without a
readable ``meta.json`` (``Endpoint.git`` None) is left out.

**The journal prefix.** A cycle is judged on the journal as it stood when the cycle was written:
for each launch log, the number of its whole lines (each ended by its newline; a last line still
being written is not one) and the sha256 of those lines' bytes, newlines included. Never a time
stamp: the clocks of the machines that write the journal and the record need not agree. A start
belongs to the prefix when its line lies within the pinned lines of its log; one appended later
is never the cycle's. A log whose pinned lines no longer have their sha256, or that is gone, is
a refusal: the journal was rewritten. The logs are read as every reader here reads them
(``study3_records.journal_lines`` and ``log_lines``), and one that resolves under a sealed folder
or cannot be read is a refusal, never opened.

**The cycle record** (``RUNS_DIR/study3/decisions/cycle-c<N>.json``, a decision record of
``study3_decisions``, written once): ``kind``, ``state`` (``passed``: the re-qualification is
read from the qualification), ``written_at`` and ``code`` (the fix: a clean checkout's commit and
its GPU code), and
- ``candidate`` and ``stage``: N and the stage of the marks (``search``, ``validation`` or
  ``confirmation``);
- ``old_code``: the GPU code in force for N before the cycle; the fix's is another;
- ``marks``: every mark of the stage (one cycle for all of them), each ``run_id``, ``label``,
  ``step``, ``arm``, ``rate``, ``seed`` and ``line``, its last end line or, without one, its start
  line in the journal (``kind``, ``path``, ``line``, ``line_sha256``);
- ``diagnosis`` and ``fix``: one line each, not blank;
- ``stages``: a claim per stage of N started when the record is written (``action``, ``arms``,
  ``justification``); the marks' stage is ``rerun`` with every marked arm;
- ``fixation``: the validation fixation in force (evidence, ``root: repo``), or null;
- ``journal``: the prefix it pins (``journal_seen``);
- ``affected``: ``runs``, every run of N whose start lies in the prefix, in a stage and arm the
  claims rerun, counting, under a GPU code other than the fix's (or none), and neither a
  scientific failure, held, of a closed arm nor cited by a closure: its ``run_id``, ``label``,
  ``step``, ``arm``, ``rate``, ``seed``, ``gpu_code`` (or null), ``records`` (its ``config.yaml``,
  ``meta.json`` and ``terminal.json``, each evidence ``{path, sha256}`` under RUNS_DIR, or an
  absence marker ``{path, absent: true}`` for a file that is not there) and its journal ``start``
  and ``end`` lines (``{path, line, line_sha256}``; ``end`` null without one), which bind the
  command and config hash it was issued; ``qualification``, the part records the bridge voids
  (``records``, evidence) and the launcher runs it supersedes (``launchers``, each ``run_id`` and
  ``meta``, its ``meta.json`` as evidence or an absence marker); ``records``, the write-once
  records the cycle makes stale;
- ``kept``: what keeps counting, each with ``why``: a run that is a ``scientific_failure``,
  ``held``, of a ``closed_arm`` (or cited by a closure), each ``run_id`` with its ``evidence``
  (its ``terminal.json``, else its ``meta.json``, else an absence marker); and per stage whose
  counting runs the claims reuse, one ``reused_stage`` entry with the ``stage`` and the number of
  its ``runs``;
- ``bridge``: ``qualification/c<N>/supersede-1.json``.
Every list is in canonical order (by run id, then path) and names nothing twice.

**The bridge** (``RUNS_DIR/qualification/c<N>/supersede-1.json``) is a supersession record in
``scripts/qualify.py``'s own format, so that its journal repair and its re-qualification read it
unchanged: ``candidate``, ``supersession`` (1), ``old_record`` (the cycle record: its absolute
``path``, its ``sha256`` and ``kind``), ``diagnosis``, ``fix``, ``old_gpu_code`` (``{sha256}``),
``fix_gpu_code`` (the fix's ``gpu_code_identity``), ``cases_requalified``, ``attempts``,
``kept_cases``, ``kept`` (``qualify.py supersede``'s entries for the numerical qualification,
under a ``rerun`` of the qualification; empty otherwise), ``launchers`` (the qualification's
launcher runs it supersedes and every affected run, each ``run_id``, ``campaign``, ``run``,
``experiment`` and ``config_hash``), ``held`` (empty), ``host``, ``superseded_at`` and
``provenance`` (the fix's commit and the writer).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from ..paths import REPO_ROOT
from .study3_decisions import CLAIMS, CYCLE_STAGES, CYCLE_STEPS, TEXT_MAX, one_line, pinned_end
from .study3_records import Endpoint, journal_lines, log_lines, under_sealed
from .study3_search import ARM_ORDER
from .study3_versions import _gpu_code

STAGES = CYCLE_STAGES  # search, qualification, validation, confirmation
STAGE_STEPS = CYCLE_STEPS  # the qualification's steps: its numerical runs and its repair
ACTIONS = CLAIMS  # reuse, rerun
JUSTIFICATION_MAX = TEXT_MAX  # characters of a diagnosis, a fix or a claim's justification
_COMMIT = re.compile(r"[0-9a-f]{40}")


class Claim(NamedTuple):
    """A stage's claim: ``reuse`` or ``rerun``, the arms it reruns (none for ``reuse``) and its
    justification."""

    action: str
    arms: tuple[str, ...]
    justification: str


class CycleRefused(ValueError):
    """A cycle that cannot be recorded or read as it stands: every reason, named."""

    def __init__(self, reasons: str | Sequence[str]) -> None:
        self.reasons = [reasons] if isinstance(reasons, str) else list(reasons)
        super().__init__("; ".join(self.reasons))


# ------------------------------------------------------------------ claims


def parse_claims(stages: Sequence[str], justifications: Sequence[str]) -> dict[str, Claim]:
    """The claims ``STAGE=reuse|rerun[:A,F,P]`` with their justifications ``STAGE=TEXT``, by
    stage in ``STAGES`` order (module docstring, "Claims"). ``CycleRefused``, naming every
    reason: an unknown stage or action, an arm that is not A, F or P (or one named twice, or
    none after the colon), arms on a ``reuse`` or on the qualification's ``rerun`` (all or
    nothing for the candidate), a stage claimed twice, and a justification that is missing,
    blank, more than one line, too long, given twice or given for a stage with no claim."""
    reasons: list[str] = []
    claimed: dict[str, tuple[str, tuple[str, ...]]] = {}
    for item in stages:
        stage, sep, value = str(item).partition("=")
        action, colon, listed = value.partition(":")
        if not sep or stage not in STAGES:
            reasons.append(f"{item!r}: the stage is not one of {list(STAGES)} (STAGE=reuse|"
                           "rerun[:A,F,P])")  # fmt: skip
            continue
        if stage in claimed:
            reasons.append(f"{item!r}: the {stage} is claimed twice")
            continue
        if action not in ACTIONS:
            reasons.append(f"{item!r}: the claim is not one of {list(ACTIONS)}")
            continue
        arms: tuple[str, ...] = ()
        if colon:
            names = listed.split(",")
            if action == "reuse":
                reasons.append(f"{item!r}: a reuse takes no arms")
                continue
            if stage == "qualification":
                reasons.append(f"{item!r}: the qualification's claim is all or nothing for the "
                               "candidate: it takes no arms")  # fmt: skip
                continue
            if not all(name in ARM_ORDER for name in names) or len(set(names)) != len(names):
                reasons.append(f"{item!r}: the arms are not distinct arms of {list(ARM_ORDER)}")
                continue
            arms = tuple(arm for arm in ARM_ORDER if arm in names)
        elif action == "rerun":
            arms = tuple(ARM_ORDER)
        claimed[stage] = (action, arms)
    given: dict[str, str] = {}
    for item in justifications:
        stage, sep, text = str(item).partition("=")
        if not sep or stage not in STAGES:
            reasons.append(f"a justification {str(item)[:60]!r}: the stage is not one of "
                           f"{list(STAGES)} (STAGE=TEXT)")  # fmt: skip
        elif stage in given:
            reasons.append(f"the {stage}'s justification is given twice")
        else:
            given[stage] = text
    for stage in STAGES:
        if stage in claimed and stage not in given:
            reasons.append(f"the {stage}'s claim has no justification (--justification "
                           f"{stage}=TEXT)")  # fmt: skip
        elif stage in given and (why := one_line(given[stage])) is not None:
            reasons.append(f"the {stage}'s justification {why}")
        if stage in given and not any(str(i).partition("=")[0] == stage for i in stages):
            reasons.append(f"a justification for the {stage}, which no claim names")
    if reasons:
        raise CycleRefused(reasons)
    return {stage: Claim(*claimed[stage], given[stage].strip()) for stage in STAGES
            if stage in claimed}  # fmt: skip


# ------------------------------------------------------------------ each run's GPU code


def run_gpu_codes(
    endpoints: Iterable[Endpoint], *, repo: str | Path = REPO_ROOT
) -> dict[str, str | None]:
    """Each run's GPU code by run id (module docstring, "A run's GPU code"): the sha256 of
    ``code_identity.gpu_code_identity`` of the commit its ``meta.json`` records, computed once
    per commit; None for a dirty checkout, one whose cleanness is not recorded, a commit that is
    no commit id or one the checkout ``repo`` cannot place. Endpoints without a readable
    ``meta.json`` are left out."""
    known: dict[str, str | None] = {}
    out: dict[str, str | None] = {}
    for e in endpoints:
        if e.git is None:
            continue
        commit, dirty = e.git
        if dirty is not False or not (isinstance(commit, str) and _COMMIT.fullmatch(commit)):
            out[e.run_id] = None
            continue
        if commit not in known:
            try:
                known[commit] = _gpu_code(Path(repo), commit)
            except LookupError:  # a commit the identity cannot place: its code is unknown
                known[commit] = None
        out[e.run_id] = known[commit]
    return out


# ------------------------------------------------------------------ the journal prefix


def journal_seen(runs_root: str | Path) -> list[dict[str, Any]]:
    """The journal prefix a cycle pins (module docstring, "The journal prefix"): per launch log,
    ``{"path": "launch/<name>.jsonl", "lines": <its whole lines>, "sha256": <of those lines'
    bytes, newlines included>}``, sorted by path. ``CycleRefused`` for a log that resolves under
    a sealed folder or cannot be read."""
    root = Path(runs_root)
    out = []
    for log in _logs(root):
        data = _log_bytes(root, log)
        lines = log_lines(data)
        count = sum(1 for offset, line, _ in lines if offset + len(line) < len(data))
        end = pinned_end(lines, count) or 0
        out.append({"path": log, "lines": count,
                    "sha256": hashlib.sha256(data[:end]).hexdigest()})  # fmt: skip
    return sorted(out, key=lambda entry: entry["path"])


def seen_starts(runs_root: str | Path, seen: Sequence[Mapping[str, Any]]) -> set[str]:
    """The run ids whose start line lies within the pinned prefix of its log (``seen``:
    ``journal_seen``'s list). ``CycleRefused``, naming each log, when a pinned prefix is no
    longer there with its sha256 (the journal was rewritten, or the log is gone or cannot be
    read)."""
    return set(_seen_starts(Path(runs_root), seen))


def _seen_starts(root: Path, seen: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """``seen_starts`` with each start's record (its first in the prefix)."""
    reasons: list[str] = []
    found: dict[str, dict[str, Any]] = {}
    logs = set(_logs(root, strict=False))
    for entry in seen:
        path, count, sha = entry.get("path"), entry.get("lines"), entry.get("sha256")
        if path not in logs:
            reasons.append(f"{path}: the log whose prefix of {count} lines the cycle pinned is "
                           "gone, unreadable or under a sealed folder")  # fmt: skip
            continue
        try:
            data = _log_bytes(root, path)
        except CycleRefused as exc:
            reasons += exc.reasons
            continue
        lines = log_lines(data)
        end = pinned_end(lines, count)
        if end is None:
            reasons.append(f"{path}: it holds {len(lines)} lines, fewer than the pinned prefix "
                           f"of {count!r}")  # fmt: skip
            continue
        now = hashlib.sha256(data[:end]).hexdigest()
        if end > len(data) or now != sha:
            reasons.append(f"{path}: its pinned prefix of {count} lines has sha256 {now} now, "
                           f"not {sha}: the journal was rewritten")  # fmt: skip
            continue
        for _, _, record in lines[:count]:
            run_id = (record or {}).get("run_id")
            if (record or {}).get("event") == "start" and isinstance(run_id, str):
                found.setdefault(run_id, record)
    if reasons:
        raise CycleRefused(reasons)
    return found


def _logs(root: Path, *, strict: bool = True) -> list[str]:
    """The launch logs (``launch/<name>.jsonl``) as ``journal_lines`` lists them; with
    ``strict``, ``CycleRefused`` for one it cannot read or that resolves under a sealed
    folder."""
    logs, unread = journal_lines(root)
    if unread and strict:
        raise CycleRefused([f"{problem}: the journal prefix cannot be pinned"
                            for problem in unread])  # fmt: skip
    return [log.path for log in logs]


def _log_bytes(root: Path, path: str) -> bytes:
    """A launch log's bytes, the sealed-path rule applied before it is opened."""
    file = root / path
    if under_sealed(file, root):
        raise CycleRefused(f"{path}: a launch log under a sealed folder: never opened")
    try:
        return file.read_bytes()
    except OSError as exc:
        raise CycleRefused(f"{path}: an unreadable launch log ({type(exc).__name__})") from exc
