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
- ``started_stages``, ``affected_and_kept`` and ``failed``: the stages a candidate has started,
  the runs a cycle affects and those that keep counting, and the readers' own predicate of a
  scientific failure (``study3_blocks.block_state``'s failed endpoint, imported).
- ``derive_cycle``, ``build_bridge`` and ``publish_cycle``: the cycle record and its bridge,
  derived from the records before the cycle, and published once each, the record first.

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

**Refusals** (``CycleRefused``, naming every reason) where a cycle is derived: a diagnosis or fix
that is not one line; the fix's GPU code the old one; a claim missing for a stage started, or
one for a stage not started; the marks' stage not a rerun of every marked arm; a mark under a
known GPU code other than the old one (its code unknown, it is a mark of the code in force); a
start in the pinned prefix that the records read do not hold; a record under a sealed folder or
present and unreadable (a missing one is an absence marker, never a refusal); and a launcher run
of the qualification that no part record classified and that the readers cannot settle
(``qualification_records.cycle_entries``: ``qualify.py``'s own resolution of such a run needs
GPU paths, so the cycle names it instead). ``qualification_records`` (and torch with it) is
imported only where the qualification is rerun.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, NamedTuple

from ..paths import REPO_ROOT
from .study3_blocks import BLOCK_STEPS, block_state
from .study3_decisions import (
    CLAIMS,
    CYCLE_STAGES,
    CYCLE_STEPS,
    MARKED_STAGES,
    TEXT_MAX,
    DecisionRefused,
    _publish,
    bridge_name,
    cycle_path,
    one_line,
    pinned_end,
    serialized,
    write_decision,
)
from .study3_records import (
    COUNTING,
    EXPERIMENT,
    GROSS_COLLAPSE,
    Endpoint,
    _sealed_file,
    journal_lines,
    log_lines,
    plain_name,
    under_sealed,
)
from .study3_search import ARM_ORDER
from .study3_versions import _gpu_code

if TYPE_CHECKING:  # the state machine's records: imported where they are built
    from .qualification_records import CycleEntries
    from .study3_progress import Records

STAGES = CYCLE_STAGES  # search, qualification, validation, confirmation
STAGE_STEPS = CYCLE_STEPS  # the qualification's steps: its numerical runs and its repair
ACTIONS = CLAIMS  # reuse, rerun
JUSTIFICATION_MAX = TEXT_MAX  # characters of a diagnosis, a fix or a claim's justification
_COMMIT = re.compile(r"[0-9a-f]{40}")
WRITER = "scripts/study3_summary.py cycle"  # the bridge's provenance: what wrote it
RUN_FILES = ("config.yaml", "meta.json", "terminal.json")  # an affected run's records


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


def _seen_starts(
    root: Path, seen: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[str, dict[str, Any]]]:
    """``seen_starts`` with each start's log and record (its first in the prefix)."""
    reasons: list[str] = []
    found: dict[str, tuple[str, dict[str, Any]]] = {}
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
                found.setdefault(run_id, (path, record))
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


# ------------------------------------------------------------------ deriving the cycle


def started_stages(records: Records, candidate: int) -> list[str]:
    """The stages (``STAGES``, in order) with any endpoint of the candidate in their steps."""
    steps = {e.cell.step for e in records.endpoints if e.cell.candidate == candidate}
    return [stage for stage in STAGES if steps & set(STAGE_STEPS[stage])]


def affected_and_kept(
    records: Records,
    candidate: int,
    claims: Mapping[str, Claim],
    *,
    run_codes: Mapping[str, str | None],
    fix_code: str,
    seen: set[str],
    closed_arms: Collection[str],
    cited: Collection[str],
) -> tuple[list[dict], list[dict]]:
    """The affected runs and what keeps counting (module docstring, "The cycle record"), in
    canonical order. Of the candidate's runs in the search, the validation and the confirmation
    whose start lies in the journal prefix (``seen``; one started later is never listed), in
    this order: a ``held`` run is kept; a run of a closed arm (``closed_arms``) or one a closure
    cites (``cited``) is kept as ``closed_arm``; a scientific failure is kept with its evidence
    (``failed``: the readers' own predicate); a run that does not count (superseded, recovered,
    refused) is left out; a counting run of a stage or arm the claims reuse counts for its
    stage's ``reused_stage`` entry; one of the fix's GPU code (``run_codes``) is left out; every
    other is affected, with its records (its ``config.yaml``, ``meta.json`` and
    ``terminal.json``, each evidence or an absence marker) and its journal start and end lines
    (``records.starts``, ``records.ends``). ``CycleRefused`` for a stage with a run in the prefix
    and no claim, a run without its start line, and a record under a sealed folder or that
    cannot be read."""
    affected: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    reused: Counter[str] = Counter()
    blocks: dict[tuple[str, int], dict[str, Any]] = {}
    mine = [e for e in records.endpoints if e.cell.candidate == candidate]
    for stage in MARKED_STAGES:
        claim = claims.get(stage)
        for e in sorted(mine, key=lambda e: e.run_id):
            if e.cell.step not in STAGE_STEPS[stage] or e.run_id not in seen:
                continue
            if claim is None:
                raise CycleRefused(f"the {stage} has runs in the journal prefix and no claim")
            if e.state == "held":
                kept.append(_kept(e, "held"))
            elif e.arm in closed_arms or e.run_id in cited:
                kept.append(_kept(e, "closed_arm"))
            elif failed(e, mine, candidate, blocks):
                kept.append(_kept(e, "scientific_failure"))
            elif e.state not in COUNTING:
                continue
            elif claim.action == "reuse" or e.arm not in claim.arms:
                reused[stage] += 1
            elif run_codes.get(e.run_id) != fix_code:
                affected.append(_affected_run(e, records, run_codes))
    kept += [{"why": "reused_stage", "stage": stage, "runs": count}
             for stage, count in reused.items() if count]  # fmt: skip
    return sorted(affected, key=lambda r: r["run_id"]), sorted(kept, key=kept_key)


def failed(e: Endpoint, endpoints: Sequence[Endpoint], candidate: int,
           blocks: dict[tuple[str, int], dict[str, Any]] | None = None) -> bool:  # fmt: skip
    """Whether a run is a scientific failure, as the readers read one: a declared instability
    stop (the state ``scientific``, the search's failure among them), or a failed endpoint that
    ``study3_blocks.block_state`` counts in its block, a gross collapse (``GROSS_COLLAPSE``).
    ``blocks``: the blocks read already, by (step, seed)."""
    if e.state == "scientific":
        return True
    if (e.state != "unusable" or e.unusable_reason != GROSS_COLLAPSE
            or e.cell.step not in BLOCK_STEPS or e.seed is None):  # fmt: skip
        return False
    blocks = {} if blocks is None else blocks
    key = (e.cell.step, e.seed)
    if key not in blocks:
        arms = [arm for arm in ARM_ORDER if any(x.arm == arm and x.seed == e.seed
                                                and x.cell.step == e.cell.step
                                                for x in endpoints)]  # fmt: skip
        try:
            blocks[key] = block_state(endpoints, e.seed, candidate, arms)
        except ValueError as exc:
            raise CycleRefused(f"{e.run_id}: its block {e.seed} cannot be read ({exc})") from exc
    support = blocks[key]["arms"].get(e.arm) or {}
    return any(x.run_id == e.run_id for x in support.get("failed", ()))


def kept_key(entry: Mapping[str, Any]) -> tuple[str, str, str]:
    """A kept entry's place in canonical order: its run id, its evidence's path, its stage."""
    return (str(entry.get("run_id", "")), str((entry.get("evidence") or {}).get("path", "")),
            str(entry.get("stage", "")))  # fmt: skip


def _kept(e: Endpoint, why: str) -> dict[str, Any]:
    """A run that keeps counting, with why and its evidence: its terminal record, else its
    ``meta.json``, else an absence marker of its terminal record."""
    files = {name: _file(e, name) for name in ("terminal.json", "meta.json")}
    evidence = next((f for f in files.values() if "sha256" in f), files["terminal.json"])
    return {"run_id": e.run_id, "why": why, "evidence": evidence}


def _file(e: Endpoint, name: str) -> dict[str, Any]:
    """A run's record file as evidence ``{path, sha256}`` under RUNS_DIR, or an absence marker
    ``{path, absent: true}``. ``CycleRefused`` for a file (or a run folder) that resolves under a
    sealed folder, never opened, and for one that is there and cannot be read."""
    if not plain_name(e.run_id):
        raise CycleRefused(f"{e.run_id!r}: no run id: no run folder is named so")
    where = f"{EXPERIMENT}/{e.run_id}/{name}"
    path = Path(e.run_dir) / name
    if _sealed_file(path, Path(e.run_dir).parent):
        raise CycleRefused(f"{where}: under a sealed folder: never evidence, never opened")
    if not (path.exists() or path.is_symlink()):
        return {"path": where, "absent": True}
    try:
        return {"path": where, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    except OSError as exc:
        raise CycleRefused(f"{where}: there, and unreadable ({type(exc).__name__})") from exc


def _affected_run(e: Endpoint, records: Records,
                  run_codes: Mapping[str, str | None]) -> dict[str, Any]:  # fmt: skip
    """An affected run as the cycle record lists it."""
    start = records.starts.get(e.run_id)
    if start is None:
        raise CycleRefused(f"{e.run_id}: no start line in the launch journal")
    end = records.ends.get(e.run_id)
    line = ("path", "line", "line_sha256")
    return {"run_id": e.run_id, "label": e.label, "step": e.cell.step, "arm": e.arm,
            "rate": e.lr, "seed": e.seed, "gpu_code": run_codes.get(e.run_id),
            "records": [_file(e, name) for name in RUN_FILES],
            "start": {k: start[k] for k in line},
            "end": None if end is None else {k: end[k] for k in line}}  # fmt: skip


def build_bridge(
    runs_root: str | Path,
    candidate: int,
    *,
    cycle_file: Path,
    cycle_sha256: str,
    diagnosis: str,
    fix: str,
    old_code: str,
    fix_identity: Mapping[str, Any],
    entries: CycleEntries | None,
    stage_runs: Sequence[Mapping[str, Any]],
    host: str,
    at: str,
    commit: str,
) -> dict:
    """The bridge (module docstring, "The bridge"): a supersession record in ``qualify.py``'s
    format whose ``old_record`` is the cycle record (``cycle_file``, its resolved absolute path,
    and ``cycle_sha256``), its qualification entries ``entries``' (``qualification_records.
    cycle_entries``; none without a rerun of the qualification), its ``launchers`` theirs and
    each affected run's (``stage_runs``: ``run_id``, ``campaign``, ``run``, ``experiment``,
    ``config_hash``, from its journal start), in canonical order, ``held`` empty.
    ``CycleRefused`` for a cycle file that is not the candidate's record and a run named
    twice."""
    if Path(cycle_file).resolve() != cycle_path(runs_root, candidate).resolve():
        raise CycleRefused(f"{cycle_file} is not candidate {candidate}'s cycle record "
                           f"({cycle_path(runs_root, candidate)})")  # fmt: skip
    keys = ("run_id", "campaign", "run", "experiment", "config_hash")
    launchers = [*(entries.launchers if entries is not None else []),
                 *({key: run[key] for key in keys} for run in stage_runs)]  # fmt: skip
    ids = [entry["run_id"] for entry in launchers]
    twice = sorted({run_id for run_id in ids if ids.count(run_id) > 1})
    if twice:
        raise CycleRefused(f"the bridge would name {twice} twice among its launcher runs")
    return {
        "candidate": candidate, "supersession": 1,
        "old_record": {"path": str(Path(cycle_file).resolve()), "sha256": cycle_sha256,
                       "kind": f"cycle-c{candidate}"},
        "diagnosis": diagnosis, "fix": fix,
        "old_gpu_code": {"sha256": old_code}, "fix_gpu_code": dict(fix_identity),
        "cases_requalified": list(entries.cases_requalified) if entries is not None else [],
        "attempts": list(entries.attempts) if entries is not None else [],
        "kept_cases": list(entries.kept_cases) if entries is not None else [],
        "kept": list(entries.kept) if entries is not None else [],
        "launchers": sorted(launchers, key=lambda entry: entry["run_id"]), "held": [],
        "host": host, "superseded_at": at,
        "provenance": {"commit": commit, "writer": WRITER},
    }  # fmt: skip


def derive_cycle(
    runs_root: str | Path,
    candidate: int,
    *,
    records: Records,
    marks: Sequence[str],
    stage: str,
    closed_arms: Collection[str],
    old_code: str,
    fixation: Mapping[str, Any] | None,
    claims: Mapping[str, Claim],
    diagnosis: str,
    fix: str,
    code: Mapping[str, str],
    fix_identity: Mapping[str, Any],
    settings: Mapping[str, Any],
    repo: str | Path,
    written_at: str,
    host: str,
) -> tuple[dict, dict]:
    """The cycle record and its bridge (module docstring), derived from ``records`` (the view
    before the cycle: ``pre_cycle_records``) and the journal prefix pinned now
    (``journal_seen``). ``marks``: the run ids of the marks of ``stage``; ``closed_arms``: the
    arms the route has closed; ``fixation``: the validation fixation in force, or None;
    ``code``: the fix (its commit and GPU code); ``fix_identity``: its ``gpu_code_identity``;
    ``settings``: the qualification's; ``repo``: the checkout the runs' commits are read in. The
    bridge names the record by the sha256 of its bytes as ``publish_cycle`` writes them.
    ``CycleRefused``, naming every reason: a diagnosis or fix that is not one line; a stage that
    is not one of the marks'; the fix's GPU code the old one; a claim missing for a stage
    started, or one for a stage not started, the marks' stage not a rerun of every marked arm
    (``parse_claims`` checked the claims' own form); a mark that is no run of the stage, or has no
    line in the journal; a mark under a known GPU code other than ``old_code`` (an
    ``other_gpu_code`` mark, no cycle case); a start in the prefix that ``records`` does not hold
    (the records are not of this journal: read them again); a launcher run of the qualification
    the readers cannot settle (``qualification_records.cycle_entries``' ``unsettled``); and what
    ``affected_and_kept`` refuses."""
    root = Path(runs_root)
    reasons: list[str] = []
    for name, value in (("diagnosis", diagnosis), ("fix", fix)):
        why = one_line(value)
        if why is not None:
            reasons.append(f"the {name} {why}")
    if stage not in MARKED_STAGES:
        reasons.append(f"the stage {stage!r} is not one of {list(MARKED_STAGES)}")
    fix_code = code.get("gpu_code_sha256")
    if fix_code == old_code:
        reasons.append(f"the fix's GPU code is the old one ({old_code}): a fix that keeps the "
                       "GPU code is no engineering cycle")  # fmt: skip
    started = started_stages(records, candidate)
    reasons += [f"the {name}, started, has no claim" for name in started if name not in claims]
    reasons += [f"a claim for the {name}, which has not started" for name in claims
                if name not in started]  # fmt: skip
    by_id = {e.run_id: e for e in records.endpoints}
    found = []
    for run_id in marks:
        e = by_id.get(run_id)
        if e is None or e.cell.candidate != candidate or stage not in MARKED_STAGES or (
                e.cell.step not in STAGE_STEPS[stage]):  # fmt: skip
            reasons.append(f"the mark {run_id} is no run of candidate {candidate}'s {stage}")
        else:
            found.append(e)
    claim = claims.get(stage)
    left = sorted({e.arm for e in found} - set(claim.arms if claim and claim.action == "rerun"
                                               else ()), key=str)  # fmt: skip
    if stage in MARKED_STAGES and (not found or left):
        reasons.append(f"the {stage}'s claim does not rerun every marked arm ({left} left): the "
                       "stage of the marks is a rerun of every marked arm")  # fmt: skip
    run_codes = run_gpu_codes(records.endpoints, repo=repo)
    for e in found:
        known = run_codes.get(e.run_id)
        if known is not None and known != old_code:
            reasons.append(f"the mark {e.run_id} ran under the GPU code {known}, not the old "
                           f"{old_code}: an other_gpu_code mark, no cycle case")  # fmt: skip
    if reasons:
        raise CycleRefused(reasons)
    journal = journal_seen(root)
    starts = _seen_starts(root, journal)
    unread = sorted(run_id for run_id, (_, start) in starts.items()
                    if start.get("experiment") == EXPERIMENT and run_id not in by_id)  # fmt: skip
    if unread:
        raise CycleRefused(f"the journal prefix starts {unread[:5]}, which the records read do "
                           "not hold: read the records of this journal again")  # fmt: skip
    cited = {entry["record"]["run_id"] for entry in records.closures.values()
             if entry.get("record") is not None
             and entry["record"].get("candidate") == candidate}  # fmt: skip
    affected, kept = affected_and_kept(records, candidate, claims, run_codes=run_codes,
                                       fix_code=fix_code, seen=set(starts),
                                       closed_arms=closed_arms, cited=cited)  # fmt: skip
    entries = None
    qualification: dict[str, list[dict[str, Any]]] = {"records": [], "launchers": []}
    if "qualification" in claims and claims["qualification"].action == "rerun":
        from .qualification_records import cycle_entries  # it loads torch: here only

        entries = cycle_entries(root, candidate, old_code=old_code, fix_code=fix_code,
                                settings=settings, run_codes=run_codes,
                                prefix={j["path"]: j["lines"] for j in journal})  # fmt: skip
        if entries.unsettled:
            raise CycleRefused(f"the qualification's launcher runs {entries.unsettled} were "
                               "classified by no part record, and the analysis readers cannot "
                               "settle them: the cycle cannot supersede or keep them (resolve "
                               "them first)")  # fmt: skip
        qualification = _qualification(root, entries)
    record = {
        "kind": f"cycle-c{candidate}", "state": "passed", "written_at": written_at,
        "code": {"commit": code.get("commit"), "gpu_code_sha256": fix_code},
        "candidate": candidate, "stage": stage, "old_code": old_code,
        "marks": sorted((_mark(e, records) for e in found), key=lambda m: m["run_id"]),
        "diagnosis": diagnosis, "fix": fix,
        "stages": {name: {"action": c.action, "arms": list(c.arms),
                          "justification": c.justification} for name, c in claims.items()},
        "fixation": None if fixation is None else dict(fixation),
        "journal": journal,
        "affected": {"runs": affected, "qualification": qualification, "records": []},
        "kept": kept, "bridge": bridge_name(candidate),
    }  # fmt: skip
    record = json.loads(json.dumps(record, allow_nan=False))  # what a reader will see
    stage_runs = []
    for run in affected:
        log, start = starts[run["run_id"]]
        stage_runs.append({"run_id": run["run_id"], "campaign": PurePosixPath(log).stem,
                           "run": start.get("run"), "experiment": start.get("experiment"),
                           "config_hash": start.get("config_hash")})  # fmt: skip
    bridge = build_bridge(root, candidate, cycle_file=cycle_path(root, candidate),
                          cycle_sha256=hashlib.sha256(serialized(record)).hexdigest(),
                          diagnosis=diagnosis, fix=fix, old_code=old_code,
                          fix_identity=fix_identity, entries=entries, stage_runs=stage_runs,
                          host=host, at=written_at, commit=str(code.get("commit")))  # fmt: skip
    return record, json.loads(json.dumps(bridge, allow_nan=False))


def _mark(e: Endpoint, records: Records) -> dict[str, Any]:
    """A mark as the record lists it: its run and its last end line, else its start line."""
    end, start = records.ends.get(e.run_id), records.starts.get(e.run_id)
    kind, line = ("end", end) if end is not None else ("start", start)
    if line is None:
        raise CycleRefused(f"the mark {e.run_id} has no line in the launch journal")
    return {"run_id": e.run_id, "label": e.label, "step": e.cell.step, "arm": e.arm,
            "rate": e.lr, "seed": e.seed,
            "line": {"kind": kind, **{k: line[k] for k in ("path", "line",
                                                           "line_sha256")}}}  # fmt: skip


def _qualification(root: Path, entries: CycleEntries) -> dict[str, list[dict[str, Any]]]:
    """The qualification's side of the affected records: the part records the bridge voids (by
    path under RUNS_DIR and sha256) and the launcher runs it supersedes (each its ``meta.json``
    as evidence or an absence marker)."""
    base = root.absolute()
    found = {}
    for entry in entries.attempts:
        if entry.get("state") == "recorded" and isinstance(entry.get("sha256"), str):
            where = Path(entry["path"]).relative_to(base).as_posix()
            found[where] = {"path": where, "sha256": entry["sha256"]}
    launchers = []
    for entry in entries.launchers:
        where = f"{entry['experiment']}/{entry['run_id']}/meta.json"
        path = root / where
        if under_sealed(path, root):
            raise CycleRefused(f"{where}: under a sealed folder: never evidence, never opened")
        meta: dict[str, Any] = {"path": where, "absent": True}
        if path.exists():
            meta = {"path": where, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        launchers.append({"run_id": entry["run_id"], "meta": meta})
    return {"records": [found[k] for k in sorted(found)],
            "launchers": sorted(launchers, key=lambda entry: entry["run_id"])}  # fmt: skip


def publish_cycle(
    runs_root: str | Path, candidate: int, record: Mapping, bridge: Mapping
) -> tuple[Path, Path]:
    """The cycle record, then its bridge, each written once (``study3_decisions._publish``; the
    record validated by ``write_decision``, the bridge in ``qualify.py``'s format). Run again
    with byte-identical content, it writes only a file that is missing. ``CycleRefused`` when a
    file is published with other content (it is never rewritten), when the bridge does not name
    the record by its bytes' sha256, and for a record ``write_decision`` refuses. Returns both
    paths."""
    root = Path(runs_root)
    path, bridge_path = cycle_path(root, candidate), root / bridge_name(candidate)
    data = serialized(record)
    for where in (path, bridge_path):
        if under_sealed(where, root):
            raise CycleRefused(f"{where}: under a sealed folder: never written, never opened")
    published = path.exists() or path.is_symlink()
    if published and path.read_bytes() != data:
        raise CycleRefused(f"{path} is published with other content: a cycle record is never "
                           "rewritten")  # fmt: skip
    if hashlib.sha256(data).hexdigest() != (bridge.get("old_record") or {}).get("sha256"):
        raise CycleRefused("the bridge does not name the cycle record by its sha256")
    if not published:
        try:
            write_decision(root, f"cycle-c{candidate}", record)
        except DecisionRefused as exc:
            raise CycleRefused(str(exc)) from exc
    content = (json.dumps(bridge, indent=1) + "\n").encode("utf-8")
    if bridge_path.exists() or bridge_path.is_symlink():
        if bridge_path.read_bytes() != content:
            raise CycleRefused(f"{bridge_path} is published with other content: a supersession "
                               "record is never rewritten")  # fmt: skip
        return path, bridge_path
    bridge_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _publish(bridge_path, content)
    except DecisionRefused as exc:
        raise CycleRefused(str(exc)) from exc
    return path, bridge_path
