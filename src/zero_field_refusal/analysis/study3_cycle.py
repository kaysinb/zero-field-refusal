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
- ``in_flight``: the candidate's runs still in flight in the journal prefix, which refuse a
  cycle.
- ``derive_cycle``, ``build_bridge`` and ``publish_cycle``: the cycle record and its bridge,
  derived from the records before the cycle, and published once each, the record first.
- ``pre_cycle_records``: the records as they stood when the cycle was written (the pre-cycle
  view), from which the cycle is derived and against which it is checked.
- ``read_cycle``, ``bridge_problems`` and ``cycle_problems``: the published cycle read back
  (reading no run record), its bridge against it, and its lists against their derivation in the
  pre-cycle view.
- ``rebuild_bridge``: the bridge of a published cycle record whose bridge is missing.
- ``stale_records``: the write-once records a cycle makes stale.
- ``journal_unread`` and ``journal_problems``: why the launch journal cannot be read whole, and
  why it is not whole, as the launcher refuses it.

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
- ``fixation``: the validation fixation in force, its chain's last link (evidence, ``root:
  repo``), or null;
- ``journal``: the prefix it pins (``journal_seen``);
- ``affected``: ``runs``, every run of N whose start lies in the prefix, in a stage and arm the
  claims rerun, counting, under a GPU code other than the fix's (or none), and neither a
  scientific failure, held, of a closed arm nor cited by a closure: its ``run_id``, ``label``,
  ``step``, ``arm``, ``rate``, ``seed``, ``gpu_code`` (or null), ``records`` (its ``config.yaml``,
  ``meta.json`` and ``terminal.json``, each evidence ``{path, sha256}`` under RUNS_DIR, or an
  absence marker ``{path, absent: true}`` for a file that is not there) and its journal ``start``
  and ``end`` lines (``{path, line, line_sha256}``; ``end`` null without one), which bind the
  command and config hash it was issued; ``qualification``, the part records the bridge voids
  (``records``, evidence), the launcher runs it supersedes (``launchers``, each ``run_id`` and
  ``meta``, its ``meta.json`` as evidence or an absence marker) and the case restart records
  counted as written before the cycle (``restarts``, evidence: every one there when the cycle is
  written; a restart is never dated by its clock); ``records``, the write-once records the cycle
  makes stale (``stale_records``: of the candidate's validation fixation, confirmation supplement,
  n and budget records, each chain's last link of the old code, and the standing versions record,
  the last whose commit has the old code; each as evidence, the checkout's ``root: repo``), those
  there when it is written;
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
one for a stage not started; no mark; the marks' stage not a rerun of every marked arm; a mark
under a known GPU code other than the old one (its code unknown, it is a mark of the code in
force); once the candidate's confirmation has started (a run of it in the records), a claim that
reruns the validation or the search ("Validation, confirmation and stopping": validation is not
reopened without an explicit prospective amendment); a mark that is not among the affected
runs; a journal whose prefix moved since the records were read, or a start in the pinned prefix
that the records read do not hold; a run of the candidate in flight (``in_flight``: its start in
the pinned prefix, no end line in it, and the state ``pending``, so that nothing settles it), in
any stage and arm, named: its records would change after the cycle is written (a kept run's
evidence is its terminal record once there, an affected run lists its files), so the cycle waits
for it to end; a record
under a sealed folder or present and unreadable (a missing one is an absence marker, never a
refusal); a launcher run of the qualification that no part record classified and that the
readers cannot settle (``qualification_records.cycle_entries``: ``qualify.py``'s own
resolution of such a run needs GPU paths, so the cycle names it instead); and case restarts that
disagree with the journal prefix (below). It is published (``publish_cycle``) only when no other
supersession record of the candidate is there and no file of it is there with other content:
every refusal comes before anything is written. ``qualification_records`` (and torch with it)
is imported only where the qualification is rerun.

**Reading a cycle back** ("Voiding and supersession": every superseded record stays, listed with
its reason). ``read_cycle`` checks, in order: the record as a decision record (its form, every
evidence file of its sha256, every absence marker's file still absent, its pinned prefix and
cited lines of their bytes: ``study3_decisions``); its old code, the chain's when given; the
pair, a record without its bridge or a bridge without its record being a problem that names
``study3_summary.py cycle``, which writes only the missing file (``qualify.py``'s own
supersession record, which names a part record and no cycle record, is no bridge: no cycle outside
the qualification, and no problem); and the bridge against the
record (``bridge_problems``): its candidate and number, its ``old_record`` the record by its
bytes' sha256, the record's diagnosis, fix and codes, its ``launchers`` exactly the affected runs
and the qualification's affected launcher runs, its ``attempts`` exactly the qualification's
affected records, nothing named twice, its launchers in canonical order (by run id), and
``held`` empty. It reads no run record. The bridge's
``old_record.path`` is the exact record path the journal repair takes (``repair_record``), shown
as it is: it names the machine the cycle was written on.

**The pre-cycle view** (``pre_cycle_records``). The records are read with the journal truncated
to the prefix the cycle pins (each prefix's sha256 checked first: a rewritten journal is a
refusal) and with the cycle's own bridge read as absent (``without``): the runs it supersedes
count as they did, its journal lines lie beyond the prefix, and a run started later is no run of
the view. Every bound cell is read (no cell the route would issue later decides a run's state).
Where the state is read (``study3_progress.read_records``), the view reads the run files its
reader read already (one snapshot: each run file is read once).
The qualification's reading is not part of the view: the cycle reads the qualification through
``qualification_records.cycle_entries``, with the same prefix and without the same bridge. The
writer derives from this view with ``without=None``; the reader re-derives with ``without`` this
cycle's bridge, before and after the repair alike, and with the case restarts the record pins
(only those count as written before the cycle; no time stamp is compared, the bridge's
``superseded_at`` included). ``cycle_problems`` then requires a claim for every stage started
and none for a stage not started (no rerun of the validation or the search once the
confirmation has started), the record's ``fixation`` the candidate's validation fixation of the
old code that the stale records list (null when they list none) and the search reused once a
fixation exists, ``affected`` (its runs and its stale records) and ``kept`` equal to
their derivation exactly
(no omission, no extra, nothing twice, canonical order), every mark among the affected runs, the
qualification's side equal to ``cycle_entries``', no listed run a scientific failure, and the
bridge equal to its derivation (``_bridge_against``: its ``launchers`` the qualification's
launcher runs and the affected runs, each field but the identity, in canonical order; under a
rerun of the qualification its ``attempts``, interrupted ones included, ``cases_requalified``,
``kept`` and ``kept_cases`` exactly ``cycle_entries``'). The pinned case restarts agree with the
journal prefix (``_restart_problems``): every restart record of the candidate that a
``superseded`` line inside the prefix cites is pinned, and none that only lines beyond it cite.
A run kept as of a closed arm is re-derived from the arms closed before the cycle: by a valid
closure of the candidate (of its arm, or naming it) whose cited line lies in the pinned prefix,
or by the route itself (``inactive_arms``: the state machine's walk over the records before the
cycle, under its old code and with the qualification read as it stood before the cycle, shows the
arm inactive: P–F closed by its validation gate or by a scientific failure, with no closure
record). A run kept as of an arm closed by neither is a problem.
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
    _digest,
    _publish,
    bridge_name,
    closures,
    cycle_path,
    decisions,
    end_lines,
    one_line,
    pinned_end,
    read_decision,
    serialized,
    start_lines,
    write_decision,
)
from .study3_records import (
    COUNTING,
    EXPERIMENT,
    GROSS_COLLAPSE,
    Endpoint,
    EndpointReader,
    _Files,
    _sealed_file,
    journal_lines,
    log_lines,
    plain_name,
    prefixed,
    read_holds,
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
# The checkout a cycle record's evidence there (the validation fixation) is read in, where a caller
# names none (``read_cycle``; the qualification's one check of a bridge)
REPO = REPO_ROOT
REOPENED = ("after the confirmation has started, validation is not reopened without an explicit "
            "prospective amendment: validation=rerun and search=rerun are refused")  # fmt: skip
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
    journal: Sequence[Mapping[str, Any]],
) -> tuple[dict, dict]:
    """The cycle record and its bridge (module docstring), derived from ``records`` (the view
    before the cycle: ``pre_cycle_records``) and ``journal``, the prefix those records were read
    with (``journal_seen``, pinned before they were read). ``marks``: the run ids of the marks of
    ``stage``; ``closed_arms``: the
    arms the route has closed; ``fixation``: the validation fixation in force, or None;
    ``code``: the fix (its commit and GPU code); ``fix_identity``: its ``gpu_code_identity``;
    ``settings``: the qualification's; ``repo``: the checkout the runs' commits are read in. The
    bridge names the record by the sha256 of its bytes as ``publish_cycle`` writes them.
    ``CycleRefused``, naming every reason: a diagnosis or fix that is not one line; a stage that
    is not one of the marks'; the fix's GPU code the old one; a claim missing for a stage
    started, or one for a stage not started, the marks' stage not a rerun of every marked arm
    (``parse_claims`` checked the claims' own form); no mark at all; a mark that is no run of the
    stage, or has no line in the journal; a mark under a known GPU code other than ``old_code``
    (an ``other_gpu_code`` mark, no cycle case); a journal whose prefix is not ``journal`` now,
    or a start in it that ``records`` does not hold (the records are not of this journal: read
    them again); a run of the candidate in flight in the prefix (``in_flight``), named; a mark
    that is not among the affected runs; a launcher run of the qualification
    the readers cannot settle (``qualification_records.cycle_entries``' ``unsettled``); case
    restarts the journal prefix cites and the cycle does not pin (``_restart_problems``); and
    what ``affected_and_kept`` refuses."""
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
    reasons += reopened(claims, started)
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
    if stage in MARKED_STAGES and not marks:
        reasons.append(f"no mark: a cycle is opened by the technical marks of its stage, and none "
                       f"of the {stage} is given")  # fmt: skip
    elif stage in MARKED_STAGES and left:
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
    now = journal_seen(root)
    if now != [dict(entry) for entry in journal]:
        raise CycleRefused("the launch journal is not the one the records were read with (its "
                           "prefix has moved since): read the records again")  # fmt: skip
    starts = _seen_starts(root, now)
    unread = sorted(run_id for run_id, (_, start) in starts.items()
                    if start.get("experiment") == EXPERIMENT and run_id not in by_id)  # fmt: skip
    if unread:
        raise CycleRefused(f"the journal prefix starts {unread[:5]}, which the records read do "
                           "not hold: read the records of this journal again")  # fmt: skip
    running = in_flight(records, candidate, set(starts))
    if running:
        raise CycleRefused(f"runs of candidate {candidate} started in the pinned journal prefix "
                           f"have not ended (no end line in it, nothing settles them): "
                           f"{running}: the cycle waits for them, since their records would "
                           "change after it is written")  # fmt: skip
    cited = {entry["record"]["run_id"] for entry in records.closures.values()
             if entry.get("record") is not None
             and entry["record"].get("candidate") == candidate}  # fmt: skip
    affected, kept = affected_and_kept(records, candidate, claims, run_codes=run_codes,
                                       fix_code=fix_code, seen=set(starts),
                                       closed_arms=closed_arms, cited=cited)  # fmt: skip
    unmarked = sorted({e.run_id for e in found} - {run["run_id"] for run in affected})
    if unmarked:
        raise CycleRefused(f"the marks {unmarked} are not among the affected runs: a mark is a "
                           "run the cycle reruns")  # fmt: skip
    entries = None
    qualification: dict[str, list[dict[str, Any]]] = {"records": [], "launchers": [],
                                                      "restarts": []}  # fmt: skip
    rerun = "qualification" in claims and claims["qualification"].action == "rerun"
    if rerun:
        from .qualification_records import cycle_entries  # it loads torch: here only

        entries = cycle_entries(root, candidate, old_code=old_code, fix_code=fix_code,
                                settings=settings, run_codes=run_codes,
                                prefix={j["path"]: j["lines"] for j in now})  # fmt: skip
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
        "journal": now,
        "affected": {"runs": affected, "qualification": qualification,
                     "records": stale_records(root, candidate, repo=repo, old_code=old_code)},
        "kept": kept, "bridge": bridge_name(candidate),
    }  # fmt: skip
    record = json.loads(json.dumps(record, allow_nan=False))  # what a reader will see
    anchored = _restart_problems(root, record) if rerun else []
    if anchored:
        raise CycleRefused(anchored)
    stage_runs = _runs_of(affected, starts)
    bridge = build_bridge(root, candidate, cycle_file=cycle_path(root, candidate),
                          cycle_sha256=hashlib.sha256(serialized(record)).hexdigest(),
                          diagnosis=diagnosis, fix=fix, old_code=old_code,
                          fix_identity=fix_identity, entries=entries, stage_runs=stage_runs,
                          host=host, at=written_at, commit=str(code.get("commit")))  # fmt: skip
    return record, json.loads(json.dumps(bridge, allow_nan=False))


def in_flight(records: Records, candidate: int, seen: Collection[str]) -> list[str]:
    """The candidate's runs still in flight in the view before the cycle (module docstring,
    "Refusals"), by run id: a start in the pinned prefix (``seen``), no end line in it
    (``records.ends``) and the state ``pending`` (the launcher has not ended it, and neither its
    ``meta.json`` nor a record or a recovery round settles it: ``study3_records``, "States")."""
    return sorted(e.run_id for e in records.endpoints
                  if e.cell.candidate == candidate and e.run_id in seen
                  and e.state == "pending" and e.run_id not in records.ends)  # fmt: skip


def reopened(claims: Mapping[str, Claim], started: Collection[str]) -> list[str]:
    """Why the claims reopen the validation once the confirmation has started (module
    docstring, "Refusals"): a rerun of the validation or the search."""
    rerun = [name for name in ("search", "validation")
             if name in claims and claims[name].action == "rerun"]  # fmt: skip
    return [f"{REOPENED} ({', '.join(f'{name}=rerun' for name in rerun)})"] if (
        "confirmation" in started and rerun) else []  # fmt: skip


def stale_records(
    runs_root: str | Path, candidate: int, *, repo: str | Path, old_code: str
) -> list[dict[str, Any]]:
    """The write-once records a cycle of ``old_code`` makes stale (module docstring, "The cycle
    record"), in canonical order: of the candidate's validation fixation, confirmation supplement,
    n and budget records, each chain's last link whose GPU code is ``old_code`` (its successors,
    written after the cycle, carry the fix), and the last versions record whose commit has
    ``old_code`` (``study3_versions._gpu_code``); each as evidence, the checkout's with ``root:
    repo``. Read as written (their readers check them): a file under a sealed folder or that
    cannot be read is passed over."""
    from .study3_decisions import chain
    from .study3_versions import fixation_chain, versions_name

    root, repo = Path(runs_root), Path(repo)
    out = []
    for step in ("validation", "confirmation"):
        for name in reversed(fixation_chain(step, candidate, repo=repo)):
            data = _plain_json(repo / name, repo)
            if _code_of(data) == old_code:
                out.append({"path": name, "sha256": _sha(repo / name), "root": "repo"})
                break
    for family in ("n", "budget"):
        for link in reversed(chain(root, f"{family}-c{candidate}")):
            if _code_of(_plain_json(Path(link["path"]), root)) == old_code:
                out.append({"path": "/".join((EXPERIMENT, "decisions", Path(link["path"]).name)),
                            "sha256": _sha(Path(link["path"]))})  # fmt: skip
                break
    names = []
    while (repo / versions_name(len(names) + 1)).is_file():
        names.append(versions_name(len(names) + 1))
    for name in reversed(names):
        data = _plain_json(repo / name, repo)
        code = data.get("code") if isinstance(data, dict) else None
        commit = code.get("commit") if isinstance(code, Mapping) else None
        if isinstance(commit, str) and _COMMIT.fullmatch(commit) and _gpu_code(
                repo, commit) == old_code:  # fmt: skip
            out.append({"path": name, "sha256": _sha(repo / name), "root": "repo"})
            break
    return sorted(out, key=lambda entry: entry["path"])


def _plain_json(path: Path, base: Path) -> Any:
    """A record's JSON as written, None under a sealed folder or unreadable (never opened)."""
    if under_sealed(path, base) or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _code_of(data: Any) -> Any:
    """A record's ``code.gpu_code_sha256``, or None."""
    code = data.get("code") if isinstance(data, dict) else None
    return code.get("gpu_code_sha256") if isinstance(code, Mapping) else None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    path under RUNS_DIR and sha256), the launcher runs it supersedes (each its ``meta.json`` as
    evidence or an absence marker) and the case restart records counted before the cycle, each
    pinned by its path and sha256 (``_evidence``)."""
    base = root.absolute()
    found = {}
    for entry in entries.attempts:
        if entry.get("state") == "recorded" and isinstance(entry.get("sha256"), str):
            where = Path(entry["path"]).relative_to(base).as_posix()
            found[where] = {"path": where, "sha256": entry["sha256"]}
    launchers = [{"run_id": entry["run_id"],
                  "meta": _evidence(root, f"{entry['experiment']}/{entry['run_id']}/meta.json")}
                 for entry in entries.launchers]  # fmt: skip
    return {"records": [found[k] for k in sorted(found)],
            "launchers": sorted(launchers, key=lambda entry: entry["run_id"]),
            "restarts": [_evidence(root, where) for where in sorted(entries.restarts)]}  # fmt: skip


def _evidence(root: Path, where: str) -> dict[str, Any]:
    """A file under RUNS_DIR (``where``) as evidence ``{path, sha256}``, or an absence marker
    ``{path, absent: true}`` when nothing is there (no file, no link). ``CycleRefused`` for one
    that resolves under a sealed folder (never opened) and for one that is there and cannot be
    read (a link to nothing, a folder)."""
    path = root / where
    if under_sealed(path, root):
        raise CycleRefused(f"{where}: under a sealed folder: never evidence, never opened")
    if not (path.exists() or path.is_symlink()):
        return {"path": where, "absent": True}
    try:
        return {"path": where, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    except OSError as exc:
        raise CycleRefused(f"{where}: there, and unreadable ({type(exc).__name__})") from exc


def _runs_of(
    runs: Sequence[Mapping[str, Any]], starts: Mapping[str, tuple[str, Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """The bridge's entries of the affected runs: each run's ``run_id`` and, from its start in
    the pinned prefix (``starts``: ``_seen_starts``), its ``campaign`` (its log's name),
    ``run``, ``experiment`` and ``config_hash``. ``CycleRefused`` for a run with no start
    there."""
    out = []
    for run in runs:
        found = starts.get(run["run_id"])
        if found is None:
            raise CycleRefused(f"{run['run_id']}: no start line in the pinned journal prefix")
        log, start = found
        out.append({"run_id": run["run_id"], "campaign": PurePosixPath(log).stem,
                    "run": start.get("run"), "experiment": start.get("experiment"),
                    "config_hash": start.get("config_hash")})  # fmt: skip
    return out


def publish_cycle(
    runs_root: str | Path,
    candidate: int,
    record: Mapping,
    bridge: Mapping,
    *,
    repo: str | Path = REPO_ROOT,
) -> tuple[Path, Path]:
    """The cycle record, then its bridge, each written once (``study3_decisions._publish``; the
    record validated by ``write_decision``, its evidence in the checkout, the validation fixation,
    read in ``repo``; the bridge in ``qualify.py``'s format). Run again
    with byte-identical content, it writes only a file that is missing. ``CycleRefused``, before
    anything is written, when another supersession record of the candidate is there
    (``c<N>/supersede-*.json``), when a file is published with other content (it is never
    rewritten), when the bridge does not name the record by its bytes' sha256, and for a record
    ``write_decision`` refuses. Returns both paths."""
    root = Path(runs_root)
    path, bridge_path = cycle_path(root, candidate), root / bridge_name(candidate)
    data = serialized(record)
    for where in (path, bridge_path):
        if under_sealed(where, root):
            raise CycleRefused(f"{where}: under a sealed folder: never written, never opened")
    others = sorted(found for found in bridge_path.parent.glob("supersede-*.json")
                    if found.name != bridge_path.name)  # fmt: skip
    if others:
        raise CycleRefused(f"{others[0]}: a supersession record of candidate {candidate} is "
                           "there already: one engineering cycle per candidate, and nothing is "
                           "written")  # fmt: skip
    published = path.exists() or path.is_symlink()
    if published and _there(path) != data:
        raise CycleRefused(f"{path} is published with other content: a cycle record is never "
                           "rewritten")  # fmt: skip
    if hashlib.sha256(data).hexdigest() != (bridge.get("old_record") or {}).get("sha256"):
        raise CycleRefused("the bridge does not name the cycle record by its sha256")
    content = (json.dumps(bridge, indent=1) + "\n").encode("utf-8")
    bridged = bridge_path.exists() or bridge_path.is_symlink()
    if bridged and _there(bridge_path) != content:
        raise CycleRefused(f"{bridge_path} is published with other content: a supersession "
                           "record is never rewritten, and nothing is written")  # fmt: skip
    if not published:
        try:
            write_decision(root, f"cycle-c{candidate}", record, repo=repo)
        except DecisionRefused as exc:
            raise CycleRefused(str(exc)) from exc
    if bridged:
        return path, bridge_path
    bridge_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _publish(bridge_path, content)
    except DecisionRefused as exc:
        raise CycleRefused(str(exc)) from exc
    return path, bridge_path


def _there(path: Path) -> bytes:
    """The bytes of a file that is there (a link included); ``CycleRefused`` when they cannot be
    read (a dangling link, a folder, no permission): nothing is written over it."""
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CycleRefused(f"{path} is there and cannot be read ({type(exc).__name__}): nothing "
                           "is written over it") from exc  # fmt: skip


def rebuild_bridge(
    runs_root: str | Path,
    record: Mapping[str, Any],
    *,
    fix_identity: Mapping[str, Any],
    run_codes: Mapping[str, str | None],
    settings: Mapping[str, Any],
    host: str,
) -> dict:
    """The bridge of a published cycle record (a valid decision record) whose bridge is missing,
    as ``derive_cycle`` built it: the record's diagnosis, fix, codes and time (``written_at``,
    the bridge's ``superseded_at``), its affected runs from their starts in its pinned prefix and,
    under a rerun of the qualification, the qualification's entries re-derived as the readers
    re-derive them (``_requalified``). ``CycleRefused`` when the fix's identity is not the
    record's code, or the qualification's side is not the record's any more (a launcher run the
    readers cannot settle, or other records): the record then stands without its bridge."""
    root = Path(runs_root)
    candidate = record["candidate"]
    if fix_identity.get("sha256") != record["code"]["gpu_code_sha256"]:
        raise CycleRefused(f"this checkout's GPU code {fix_identity.get('sha256')} is not the "
                           f"cycle record's fix, {record['code']['gpu_code_sha256']}")  # fmt: skip
    entries = None
    if record["stages"].get("qualification", {}).get("action") == "rerun":
        entries = _requalified(root, record, run_codes=run_codes, settings=settings)
        if entries.unsettled or _qualification(root, entries) != record["affected"][
                "qualification"]:  # fmt: skip
            raise CycleRefused("the qualification's side of the cycle is not the record's any "
                               "more: its bridge cannot be written from it")  # fmt: skip
    bridge = build_bridge(root, candidate, cycle_file=cycle_path(root, candidate),
                          cycle_sha256=hashlib.sha256(serialized(record)).hexdigest(),
                          diagnosis=record["diagnosis"], fix=record["fix"],
                          old_code=record["old_code"], fix_identity=fix_identity,
                          entries=entries, stage_runs=_stage_runs(root, record), host=host,
                          at=record["written_at"], commit=record["code"]["commit"])  # fmt: skip
    return json.loads(json.dumps(bridge, allow_nan=False))


_SUPERSESSION = re.compile(r"supersede-\d+\.json")
_CASE_RESTART = re.compile(r"restart-\d+\.json")


def journal_unread(runs_root: str | Path) -> list[str]:
    """Why the launch journal cannot be read whole: a launch log that cannot be read or resolves
    under a sealed folder, and a torn line that no resolution names, the last of its log included
    (``JournalLog.refused``). The first part of ``journal_problems``: what makes every reader of
    the journal, the decision records' checks included, read less than is there."""
    logs, unread = journal_lines(Path(runs_root))
    return [*unread, *(f"{log.path} line {number}: a torn line that no resolution names"
                       for log in logs for number in log.refused)]  # fmt: skip


def journal_problems(runs_root: str | Path) -> list[str]:
    """Why the launch journal is not whole, as the launcher and ``qualify.py`` refuse it (exit
    12): what ``journal_unread`` names; and a line that a published qualification record requires
    and no log holds (``launch.missing_lines``, mirrored): a supersession record's ``superseded``
    line of each run of its ``launchers`` and ``held`` line of each run of its ``held`` (the same
    label, experiment and run id), a case restart record's ``superseded`` line of its launcher
    run, under that run's label and experiment (its start's in the journal, else its run
    folder's: the campaign label its ``meta.json`` records and the experiment's folder), else by
    its run id alone. A record that cannot be read may name a line, and is named too."""
    root = Path(runs_root)
    logs, _ = journal_lines(root)
    problems = journal_unread(root)
    records = [line.record for log in logs for line in log.lines if line.record is not None]
    lines = {tuple(r.get(k) for k in ("event", "run", "experiment", "run_id")) for r in records}
    superseded = {r.get("run_id") for r in records if r.get("event") == "superseded"}
    out = root / "qualification"
    found = [(path, "supersession") for path in sorted(out.glob("c*/supersede-*.json"))
             if _SUPERSESSION.fullmatch(path.name)]  # fmt: skip
    found += [(path, "restart") for path in sorted(out.glob("c*/*/s*/*/restart-*.json"))
              if _CASE_RESTART.fullmatch(path.name)]  # fmt: skip
    for path, kind in found:
        where = path.relative_to(root).as_posix()
        record = None
        if not under_sealed(path, root):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = None
        if not isinstance(record, dict):
            problems.append(f"{where}: a record that cannot be read may name a journal line")
            continue
        if kind == "restart":
            launcher = record.get("launcher")
            run_id = launcher.get("run_id") if isinstance(launcher, Mapping) else None
            if not isinstance(run_id, str):
                continue
            labels = _run_labels(root, records, run_id)
            there = (any(("superseded", label, experiment, run_id) in lines
                         for label, experiment in labels) if labels
                     else run_id in superseded)  # fmt: skip
            if not there:
                problems.append(f"{where}: no superseded line of {run_id}")
            continue
        for event, key in (("superseded", "launchers"), ("held", "held")):
            entries = record.get(key) or []
            for entry in entries if isinstance(entries, list) else [None]:
                named = tuple(entry.get(k) for k in ("run", "experiment", "run_id")) if isinstance(
                    entry, Mapping) else (None,)  # fmt: skip
                if (event, *named) not in lines:
                    shown = named[-1] if isinstance(named[-1], str) else "a run"
                    problems.append(f"{where}: no {event} line of {shown}")
    return problems


def _run_labels(root: Path, records: Sequence[Mapping[str, Any]],
                run_id: str) -> set[tuple[str, str]]:  # fmt: skip
    """A launcher run's label and experiment, as ``launch`` finds them for a case restart's
    line: its starts' in the journal, else its run folder's (``RUNS_DIR/<experiment>/<run
    id>/meta.json``'s campaign label; a ``meta.json`` under a sealed folder or unreadable gives
    none)."""
    labels = {(r["run"], r["experiment"]) for r in records if r.get("event") == "start"
              and r.get("run_id") == run_id and isinstance(r.get("run"), str)
              and isinstance(r.get("experiment"), str)}  # fmt: skip
    if labels or not plain_name(run_id):
        return labels
    for meta in sorted(root.glob(f"*/{run_id}/meta.json")):
        if under_sealed(meta, root):
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        campaign = data.get("campaign") if isinstance(data, dict) else None
        label = campaign.get("label") if isinstance(campaign, Mapping) else None
        if isinstance(label, str):
            labels.add((label, meta.parent.parent.name))
    return labels


# ------------------------------------------------------------------ reading the cycle back


def pre_cycle_records(
    runs_root: str | Path,
    candidate: int,
    *,
    journal: Sequence[Mapping[str, Any]],
    without: str | None,
    theta0: str | Path,
    repo: str | Path,
    settings: Mapping[str, Any],
    files: _Files | None = None,
) -> Records:
    """The records as they stood when the cycle was written (module docstring, "The pre-cycle
    view"): ``journal``, the pinned prefix (``journal_seen``'s list; ``CycleRefused`` when it is
    no longer there with its sha256); ``without``, the cycle's bridge (its path under RUNS_DIR),
    read as absent, or None (the writer's view); ``theta0``, θ₀'s evaluation; ``repo``, the
    checkout of the decision records' evidence and the runs' commits; ``settings``, the
    qualification's; ``files``, the run files a reader of the same records read already
    (``EndpointReader.files``: one snapshot, each run file read once), or None. ``candidate`` is
    the cycle's: every candidate's records are read."""
    from . import qualification_records  # it loads torch (qualification_verdicts): here only
    from .study3_progress import Records, _bound, _headers, _record_digest, _recoverable
    from .study3_versions import uncounted_runs

    root, repo = Path(runs_root), Path(repo)
    seen_starts(root, journal)  # the prefix is there, of its sha256
    prefix = {entry["path"]: entry["lines"] for entry in journal}
    checks = {where: check for where, check in
              qualification_records.cycle_checks(root, settings=settings).items()
              if where != without}  # fmt: skip
    restarts = qualification_records.restart_checks(root)
    holds, hold_problems = read_holds(root, cycles=checks)
    entries = {c: decisions(root, c, repo=repo) for c in (1, 2)}
    found = {k: v for c in (1, 2) for k, v in closures(root, c, repo=repo).items()}
    valid = {k for found_ in (*entries.values(), found) for k, e in found_.items()
             if e["state"] in ("passed", "failed")}  # fmt: skip
    cycles = {c: [x for x in qualification_records.supersessions(root, c)
                  if Path(x["path"]).relative_to(root).as_posix() != without]
              for c in (1, 2)}  # fmt: skip
    logs = prefixed(journal_lines(root)[0], prefix)
    reader = EndpointReader(root, theta0=theta0, cycles=checks, restarts=restarts, prefix=prefix,
                            files=files)  # fmt: skip
    endpoints, problems = reader.read(_bound(reader.read({})[0]))
    return Records(
        endpoints=tuple(endpoints), problems=tuple(problems), holds=tuple(holds),
        hold_problems=tuple(hold_problems), decisions=entries, closures=found,
        headers=_headers(root), digests={k: _record_digest(root, k) for k in valid},
        ends=end_lines(root, logs), starts=start_lines(root, logs), cycles=cycles,
        recoverable=_recoverable(reader, endpoints),
        uncounted=uncounted_runs(endpoints, repo=repo),
        resolved=tuple({"log": r["path"], "line": r["line"], "kind": r["kind"]}
                       for log in logs for r in log.resolved),
    )  # fmt: skip


def read_cycle(
    runs_root: str | Path,
    candidate: int,
    *,
    repo: str | Path | None = None,
    chain_old: str | None = None,
) -> dict[str, Any]:
    """The candidate's cycle as published (module docstring, "Reading a cycle back"), reading no
    run record: ``state`` (``missing``: neither file, or no cycle record and ``qualify.py``'s own
    supersession record, whose old record is no cycle record: the qualification's cycle, no
    problem here; ``valid``; ``invalid``: any problem),
    ``path`` and ``bridge_path``, the ``record`` (None unless it is a valid decision record) and
    the ``bridge`` (None unless readable), the ``claims`` by stage, the ``affected`` run ids, the
    ``kept`` entries, ``repair_record`` (the bridge's ``old_record.path``: the exact ``--record``
    of the journal repair) and the ``problems``. ``repo``: the checkout of the fixation's
    evidence (``REPO`` by default); ``chain_old``: the code in force before the cycle on the chain
    of engineering cycles, when known."""
    root, repo = Path(runs_root), REPO if repo is None else repo
    path, bridge_path = cycle_path(root, candidate), root / bridge_name(candidate)
    out: dict[str, Any] = {"state": "missing", "path": str(path), "bridge_path": str(bridge_path),
                           "record": None, "bridge": None, "claims": {}, "affected": [],
                           "kept": [], "repair_record": None, "problems": []}  # fmt: skip
    has_record = path.exists() or path.is_symlink()
    has_bridge = bridge_path.exists() or bridge_path.is_symlink()
    if not (has_record or has_bridge):
        return out
    problems: list[str] = []
    bridge = None
    if has_bridge:
        if under_sealed(bridge_path, root):
            problems.append(f"{bridge_path}: under a sealed folder: never opened")
        else:
            try:
                bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                problems.append(f"{bridge_path}: an unreadable bridge ({type(exc).__name__})")
            if bridge is not None and not isinstance(bridge, dict):
                problems.append(f"{bridge_path}: the bridge is no JSON object")
                bridge = None
    if not has_record and bridge is not None and not _names_cycle_record(bridge):
        return out  # qualify.py's own cycle, from a part record: no cycle outside it
    record = None
    if has_record:
        record, found = read_decision(root, f"cycle-c{candidate}", repo=repo)
        problems += found
    else:
        problems.append(f"{bridge_path}: the bridge is there without its cycle record {path}: "
                        "run study3_summary.py cycle again, which writes only the missing "
                        "file")  # fmt: skip
    if not has_bridge:
        problems.append(f"{path}: the cycle record is published without its bridge "
                        f"{bridge_path}: run study3_summary.py cycle again, which writes only "
                        "the missing file")  # fmt: skip
    if record is not None:
        if chain_old is not None and record["old_code"] != chain_old:
            problems.append(f"{path}: its old code {record['old_code']} is not the code in force "
                            f"before it on the chain of engineering cycles, "
                            f"{chain_old}")  # fmt: skip
        if bridge is not None:
            problems += [f"{bridge_path}: {p}" for p in
                         _bridge_problems(candidate, bridge, record, _digest(path))]  # fmt: skip
    old = bridge.get("old_record") if isinstance(bridge, dict) else None
    out.update(
        state="invalid" if problems else "valid", record=record, bridge=bridge,
        claims={} if record is None else {
            stage: Claim(c["action"], tuple(c["arms"]), c["justification"])
            for stage, c in record["stages"].items()},
        affected=[] if record is None else [r["run_id"] for r in record["affected"]["runs"]],
        kept=[] if record is None else list(record["kept"]),
        repair_record=(old.get("path") if isinstance(old, Mapping)
                       and isinstance(old.get("path"), str) else None),
        problems=problems,
    )  # fmt: skip
    return out


def _names_cycle_record(bridge: Mapping[str, Any]) -> bool:
    """Whether a supersession record's ``old_record.path`` names a cycle record
    (``study3/decisions/cycle-c<N>.json``, by its last components): the bridge of a cycle outside
    the qualification. ``qualify.py``'s own record names a part record of the qualification."""
    old = bridge.get("old_record")
    named = old.get("path") if isinstance(old, Mapping) else None
    if not isinstance(named, str):
        return False  # it names no cycle record: no bridge (its own check names what it lacks)
    parts = PurePosixPath(named.replace("\\", "/")).parts[-3:]
    return len(parts) == 3 and parts[:2] == (EXPERIMENT, "decisions") and bool(
        re.fullmatch(r"cycle-c\d+\.json", parts[2]))  # fmt: skip


def bridge_problems(
    runs_root: str | Path,
    candidate: int,
    bridge: Mapping[str, Any],
    *,
    repo: str | Path = REPO_ROOT,
    cycle: Mapping[str, Any] | None = None,
) -> list[str]:
    """The problems of a bridge against the candidate's cycle record as it is now (module
    docstring, "Reading a cycle back"); the record itself must be a valid decision record.
    ``cycle``: the record as ``read_cycle`` accepted it, when read already (it is then not read
    again); its file's sha256 is taken now."""
    root = Path(runs_root)
    record, problems = (cycle, []) if cycle is not None else read_decision(
        root, f"cycle-c{candidate}", repo=repo)  # fmt: skip
    if record is None:
        return problems or [f"{cycle_path(root, candidate)}: no cycle record of candidate "
                            f"{candidate} is published"]  # fmt: skip
    return _bridge_problems(candidate, bridge, record, _digest(cycle_path(root, candidate)))


def _bridge_problems(candidate: int, bridge: Any, record: Mapping[str, Any],
                     sha: str) -> list[str]:  # fmt: skip
    """The bridge against its record (whose file has sha256 ``sha``)."""
    if not isinstance(bridge, Mapping):
        return ["the bridge is no JSON object"]
    found = []
    if bridge.get("candidate") != candidate or type(bridge.get("candidate")) is not int:
        found.append(f"it names candidate {bridge.get('candidate')!r}, not {candidate}")
    if bridge.get("supersession") != 1:
        found.append(f"its supersession is {bridge.get('supersession')!r}, not 1")
    old = bridge.get("old_record") if isinstance(bridge.get("old_record"), Mapping) else {}
    named = old.get("path")
    tail = PurePosixPath(named.replace("\\", "/")).parts[-3:] if isinstance(named, str) else ()
    if tail != (EXPERIMENT, "decisions", f"cycle-c{candidate}.json"):
        found.append(f"its old record {named!r} is not the cycle record cycle-c{candidate}.json")
    if old.get("sha256") != sha:
        found.append(f"its old record names sha256 {old.get('sha256')!r}, not the cycle record's "
                     f"{sha}")  # fmt: skip
    if old.get("kind") != f"cycle-c{candidate}":
        found.append(f"its old record's kind is {old.get('kind')!r}, not cycle-c{candidate}")
    for key in ("diagnosis", "fix"):
        if bridge.get(key) != record[key]:
            found.append(f"its {key} is not the cycle record's")
    codes = (("old_gpu_code", record["old_code"]),
             ("fix_gpu_code", record["code"]["gpu_code_sha256"]))  # fmt: skip
    for key, code in codes:
        given = bridge.get(key) if isinstance(bridge.get(key), Mapping) else {}
        if given.get("sha256") != code:
            found.append(f"its {key} is {given.get('sha256')!r}, not the cycle record's {code}")
    affected = record["affected"]
    wanted = {r["run_id"] for r in affected["runs"]} | {
        r["run_id"] for r in affected["qualification"]["launchers"]}  # fmt: skip
    launchers = bridge.get("launchers")
    ids = [e.get("run_id") if isinstance(e, Mapping) else None
           for e in (launchers if isinstance(launchers, list) else [None])]  # fmt: skip
    found += [f"its launchers name {run_id!r} twice" for run_id in sorted(
        {str(r) for r in ids if ids.count(r) > 1})]  # fmt: skip
    if len(set(ids)) == len(ids) and ids != sorted(ids, key=str):
        found.append("its launchers are not in canonical order (by run id)")
    if set(ids) != wanted:
        found.append(f"its launchers are not the affected runs and the qualification's affected "
                     f"launcher runs (missing {sorted(wanted - set(ids), key=str)[:5]}, other "
                     f"{sorted(set(ids) - wanted, key=str)[:5]})")  # fmt: skip
    attempts = bridge.get("attempts")
    if not isinstance(attempts, list) or not all(isinstance(e, Mapping) for e in attempts):
        found.append("its attempts are not a list of attempt entries")
        attempts = []
    shas = [e.get("sha256") for e in attempts if e.get("sha256") is not None]  # recorded ones
    records = {r["sha256"] for r in affected["qualification"]["records"]}
    found += [f"its attempts name the record {x!r} twice" for x in sorted(
        {str(x) for x in shas if shas.count(x) > 1})]  # fmt: skip
    if set(shas) != records:
        found.append("its attempts are not the qualification's affected records of the cycle "
                     "record")  # fmt: skip
    if bridge.get("held") != []:
        found.append("its held is not empty: a cycle outside the qualification holds no run")
    if record["stages"].get("qualification", {}).get("action") != "rerun":
        for key in ("cases_requalified", "attempts", "kept_cases", "kept"):
            if bridge.get(key) != []:
                found.append(f"its {key} is not empty, and the cycle reuses the qualification")
    return found


def cycle_problems(
    found: Mapping[str, Any],
    *,
    runs_root: str | Path,
    theta0: str | Path,
    repo: str | Path,
    settings: Mapping[str, Any],
    files: _Files | None = None,
) -> list[str]:
    """The problems of a published cycle (``read_cycle``'s ``found``) against the records before
    it (module docstring, "The pre-cycle view"); each makes the cycle invalid. None for a cycle
    whose record is not a valid decision record (``read_cycle`` names why). ``files``: the run
    files the caller's reader read already (``pre_cycle_records``), or None."""
    record = found.get("record")
    if record is None:
        return []
    root = Path(runs_root)
    where = found.get("path") or str(cycle_path(root, record["candidate"]))
    candidate = record["candidate"]
    without = bridge_name(candidate)
    try:
        records = pre_cycle_records(root, candidate, journal=record["journal"], without=without,
                                    theta0=theta0, repo=repo, settings=settings,
                                    files=files)  # fmt: skip
        seen = seen_starts(root, record["journal"])
    except CycleRefused as exc:
        return [f"{where}: {reason}" for reason in exc.reasons]
    except (OSError, ValueError) as exc:
        return [f"{where}: the records before the cycle cannot be read ({type(exc).__name__}: "
                f"{exc})"]  # fmt: skip
    claims = {stage: Claim(c["action"], tuple(c["arms"]), c["justification"])
              for stage, c in record["stages"].items()}  # fmt: skip
    started = started_stages(records, candidate)
    problems = [f"{where}: the {stage}, started when the cycle was written, has no claim (with its "
                "justification)" for stage in started if stage not in claims]  # fmt: skip
    problems += [f"{where}: a claim for the {stage}, which had not started" for stage in claims
                 if stage not in started]  # fmt: skip
    problems += [f"{where}: {why}" for why in reopened(claims, started)]
    stale = stale_records(root, candidate, repo=repo, old_code=record["old_code"])
    problems += _compare(where, "affected.records", record["affected"]["records"], stale,
                         lambda r: r["path"])  # fmt: skip
    problems += [f"{where}: {why}" for why in _fixation_problems(record, claims, stale, repo)]
    run_codes = run_gpu_codes(records.endpoints, repo=repo)
    by_id = {e.run_id: e for e in records.endpoints}
    closed = [k["run_id"] for k in record["kept"] if k.get("why") == "closed_arm"]
    # a closure the cycle may rest on cites a line of the pinned prefix: written before it
    valid = [e["record"] for e in records.closures.values() if e.get("record") is not None
             and e["record"].get("candidate") == candidate
             and _in_prefix(e["record"].get("line"), record["journal"])]  # fmt: skip
    cited = {c["run_id"] for c in valid if isinstance(c.get("run_id"), str)}
    closed_arms = {c["arm"] for c in valid if isinstance(c.get("arm"), str)}
    uncovered = [
        run_id
        for run_id in closed
        if run_id not in cited
        and (by_id[run_id].arm if run_id in by_id else None) not in closed_arms
    ]
    if uncovered:  # an arm the route itself closed, with no closure record
        closed_arms |= inactive_arms(records, record, runs_root=root, settings=settings)
    for run_id in uncovered:
        if (by_id[run_id].arm if run_id in by_id else None) not in closed_arms:
            problems.append(f"{where}: {run_id} is kept as of a closed arm, and neither the route "
                            "before the cycle closed its arm nor a valid closure of the "
                            "candidate closes its arm or names it")  # fmt: skip
    try:
        affected, kept = affected_and_kept(records, candidate, claims, run_codes=run_codes,
                                           fix_code=record["code"]["gpu_code_sha256"], seen=seen,
                                           closed_arms=closed_arms, cited=cited)  # fmt: skip
    except CycleRefused as exc:
        return problems + [f"{where}: {reason}" for reason in exc.reasons]
    problems += _compare(where, "affected.runs", record["affected"]["runs"], affected,
                         lambda r: r["run_id"])  # fmt: skip
    problems += _compare(where, "kept", record["kept"], kept, kept_key)
    listed = {run["run_id"] for run in record["affected"]["runs"]}
    problems += [f"{where}: the mark {mark['run_id']} is not among the affected runs: a mark is a "
                 "run the cycle reruns" for mark in record["marks"]
                 if mark["run_id"] not in listed]  # fmt: skip
    entries = None
    wanted: dict[str, list[dict[str, Any]]] = {"records": [], "launchers": [], "restarts": []}
    try:
        if claims.get("qualification", Claim("reuse", (), "")).action == "rerun":
            entries = _requalified(root, record, run_codes=run_codes, settings=settings)
            problems += [f"{where}: the qualification's launcher run {run_id} was classified by "
                         "no part record, and the readers cannot settle it" for run_id in
                         entries.unsettled]  # fmt: skip
            wanted = _qualification(root, entries)
            problems += [f"{where}: {p}" for p in _restart_problems(root, record)]
        stage_runs = _stage_runs(root, record)
    except CycleRefused as exc:
        return problems + [f"{where}: {reason}" for reason in exc.reasons]
    for key, by in (("records", lambda r: r["path"]), ("launchers", lambda r: r["run_id"]),
                    ("restarts", lambda r: r["path"])):  # fmt: skip
        problems += _compare(where, f"affected.qualification.{key}",
                             record["affected"]["qualification"][key], wanted[key],
                             by)  # fmt: skip
    problems += _bridge_against(where, found.get("bridge"), entries, stage_runs)
    mine = [e for e in records.endpoints if e.cell.candidate == candidate]
    for run in record["affected"]["runs"]:
        e = by_id.get(run["run_id"])
        if e is not None and failed(e, mine, candidate):
            problems.append(f"{where}: {e.run_id} is a scientific failure, and it is listed as "
                            "affected: a scientific failure keeps counting")  # fmt: skip
    return problems


def _fixation_problems(
    record: Mapping[str, Any],
    claims: Mapping[str, Claim],
    stale: Sequence[Mapping[str, Any]],
    repo: str | Path,
) -> list[str]:
    """The record's validation fixation against the stale records (module docstring, "The
    pre-cycle view"): its ``fixation`` is the candidate's validation fixation of the old code in
    force when it was written (the link ``stale_records`` lists: its path, sha256 and root), or
    null when there is none; and once a fixation exists the search is reused, never rerun."""
    from .study3_versions import fixation_chain

    names = set(fixation_chain("validation", record["candidate"], repo=repo))
    derived = next((dict(entry) for entry in stale if entry.get("path") in names), None)
    found = []
    if record.get("fixation") != derived:
        found.append(f"its fixation {record.get('fixation')} is not the validation fixation of its "
                     f"old code in force when it was written, {derived}")  # fmt: skip
    search = claims.get("search")
    if (record.get("fixation") is not None or derived is not None) and (
            search is not None and search.action == "rerun"):  # fmt: skip
        found.append("the validation fixation exists: the search is never rerun after it "
                      "(search=rerun; search=reuse, with its justification)")  # fmt: skip
    return found


def inactive_arms(records: Records, record: Mapping[str, Any], *, runs_root: str | Path,
                  settings: Mapping[str, Any]) -> set[str]:  # fmt: skip
    """The arms the route had closed for the cycle's candidate before the cycle (module
    docstring, "The pre-cycle view"): those that ``study3_progress.advance`` over the records
    before it (``records``), under the cycle's old code, does not show active once its walk is at
    the candidate. The qualification is read as ``records`` give it, or, when they give none
    (``pre_cycle_records`` reads no qualification), as it stood before the cycle
    (``qualification_records.read_qualification`` with the cycle's journal prefix): each
    candidate's under the code in force for its runs on the chain before the cycle, the cycle's
    candidate's under its old code, its bridge read as absent (its old records count, none voided
    by it) and, when the cycle reruns the qualification, only the case restarts it pinned (one
    that reuses it leaves its records as they are). So an arm the route alone closed at or after
    the qualification (P–F's validation gate, or a scientific failure) stays shown after the
    cycle's repair and its re-qualification as before them."""
    from dataclasses import replace

    from .study3_progress import _engineering_code, advance, code_chain

    candidate, old = record["candidate"], record["old_code"]
    if not records.qualification:
        from . import qualification_records  # it loads torch: here only

        engineering = (records.decisions.get(1) or {}).get("engineering") or {}
        _, in_force, _ = code_chain(_engineering_code(engineering), records.cycles)
        prefix = {j["path"]: j["lines"] for j in record["journal"]}
        rerun = (record["stages"].get("qualification") or {}).get("action") == "rerun"
        pinned = [r["path"] for r in record["affected"]["qualification"]["restarts"]]
        readings = {}
        for c in (1, 2):
            code = old if c == candidate else in_force.get(c)
            if code is None:
                continue
            own = c == candidate
            readings[c] = qualification_records.read_qualification(
                runs_root, c, expected_code=code, settings=settings, prefix=prefix,
                without=bridge_name(candidate) if own else None,
                restarts=pinned if own and rerun else None)  # fmt: skip
        records = replace(records, qualification=readings)
    result = advance(records, expected_code=old)
    if result.get("candidate") != candidate:
        return set()
    return {arm for arm in ARM_ORDER if arm not in result["active"]["arms"]}


def _requalified(root: Path, record: Mapping[str, Any], *, run_codes: Mapping[str, str | None],
                 settings: Mapping[str, Any]) -> CycleEntries:  # fmt: skip
    """The qualification's entries of a published cycle as the readers re-derive them
    (``qualification_records.cycle_entries``): under the record's codes, its own bridge read as
    absent, the journal up to its pinned prefix and the case restarts it pinned (never a time
    stamp)."""
    from .qualification_records import cycle_entries  # it loads torch: here only

    candidate = record["candidate"]
    return cycle_entries(root, candidate, old_code=record["old_code"],
                         fix_code=record["code"]["gpu_code_sha256"], settings=settings,
                         run_codes=run_codes, without=bridge_name(candidate),
                         prefix={j["path"]: j["lines"] for j in record["journal"]},
                         restarts=[r["path"] for r in
                                   record["affected"]["qualification"]["restarts"]])  # fmt: skip


def _stage_runs(root: Path, record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The bridge's entries of the record's affected runs, from their starts in its pinned
    prefix (``_runs_of``)."""
    return _runs_of(record["affected"]["runs"], _seen_starts(root, record["journal"]))


ATTEMPT_KEYS = ("case", "stage", "part", "attempt", "state", "sha256")  # an attempt compared
BRIDGE_LISTS = {"attempts": ATTEMPT_KEYS, "cases_requalified": ("case", "stage"),
                "kept": ("case", "stage", "sha256"),
                "kept_cases": ("case", "stage", "why")}  # fmt: skip


def _bridge_against(where: str, bridge: Any, entries: CycleEntries | None,
                    stage_runs: Sequence[Mapping[str, Any]]) -> list[str]:  # fmt: skip
    """The bridge against its derivation (module docstring, "The pre-cycle view"): its
    ``launchers`` exactly the qualification's launcher runs (``entries``, under a rerun of the
    qualification) and the affected runs (``stage_runs``), every field but the identity, in
    canonical order; its ``attempts`` (interrupted ones included), ``cases_requalified``,
    ``kept`` and ``kept_cases`` exactly ``entries``' (empty without a rerun), in their order. A
    bridge that cannot be read is ``read_cycle``'s problem."""
    if not isinstance(bridge, Mapping):
        return []

    def bare(entry: Any) -> Any:
        return ({k: v for k, v in entry.items() if k != "identity"}
                if isinstance(entry, Mapping) else entry)  # fmt: skip

    def rows(value: Any, keys: Sequence[str]) -> list[Any]:
        items = value if isinstance(value, list) else [value]
        return [tuple(e.get(k) for k in keys) if isinstance(e, Mapping) else e for e in items]

    derived = [*(bare(e) for e in (entries.launchers if entries is not None else [])),
               *(dict(run) for run in stage_runs)]  # fmt: skip
    launchers = bridge.get("launchers")
    found = []
    if [bare(e) for e in (launchers if isinstance(launchers, list) else [launchers])] != sorted(
            derived, key=lambda e: str(e["run_id"])):  # fmt: skip
        found.append(f"{where}: the bridge's launchers are not their derivation (the "
                     "qualification's launcher runs and the affected runs, each field but the "
                     "identity, in canonical order)")  # fmt: skip
    for key, keys in BRIDGE_LISTS.items():
        wanted = list(getattr(entries, key)) if entries is not None else []
        if rows(bridge.get(key), keys) != rows(wanted, keys):
            found.append(f"{where}: the bridge's {key} are not their derivation "
                         "(qualification_records.cycle_entries)")  # fmt: skip
    return found


_RESTART = re.compile(r"qualification/c\d+/[^/]+/s[^/]+/[^/]+/restart-\d+\.json")


def _restart_problems(root: Path, record: Mapping[str, Any]) -> list[str]:
    """The case restarts a cycle pins against the journal prefix it pins (module docstring, "The
    pre-cycle view"): every case restart record of the candidate's qualification that a
    ``superseded`` line inside the prefix cites (by its path's last components) is pinned, and
    none that only lines beyond the prefix cite."""
    candidate = record["candidate"]
    pinned = {r["path"] for r in record["affected"]["qualification"]["restarts"]}
    counts = {j["path"]: j["lines"] for j in record["journal"]}
    inside: set[str] = set()
    beyond: set[str] = set()
    for log in journal_lines(root)[0]:
        for line in log.lines:
            found = line.record or {}
            cited = found.get("record")
            if found.get("event") != "superseded" or not isinstance(cited, str):
                continue
            parts = PurePosixPath(cited.replace("\\", "/")).parts[-6:]
            where = "/".join(parts)
            if _RESTART.fullmatch(where) and parts[1] == f"c{candidate}":
                (inside if line.line <= counts.get(log.path, 0) else beyond).add(where)
    problems = [f"the case restart {where} is cited by a superseded line inside the pinned "
                "prefix and is not pinned" for where in sorted(inside - pinned)]  # fmt: skip
    problems += [f"the case restart {where} is pinned and cited only by a superseded line "
                 "beyond the pinned prefix"
                 for where in sorted(pinned & beyond - inside)]  # fmt: skip
    return problems


def _in_prefix(line: Any, journal: Sequence[Mapping[str, Any]]) -> bool:
    """Whether a cited line (``{path, line}``) lies within the pinned prefix of its log."""
    if not isinstance(line, Mapping):
        return False
    count = {j.get("path"): j.get("lines") for j in journal}.get(line.get("path"))
    number = line.get("line")
    return type(number) is int and type(count) is int and 1 <= number <= count


def _compare(where: str, name: str, recorded: Sequence[Mapping[str, Any]],
             derived: Sequence[Mapping[str, Any]], key: Any) -> list[str]:  # fmt: skip
    """A recorded list against its derivation, exactly: nothing derived and not listed, nothing
    listed and not derived, nothing twice, each entry as derived, in canonical order."""
    keys = [key(x) for x in recorded]
    expected = {key(x): x for x in derived}
    problems = [f"{where}: {k} is derived and not listed in {name}" for k in expected
                if k not in keys]  # fmt: skip
    problems += [f"{where}: {k} is listed in {name} and not derived" for k in dict.fromkeys(keys)
                 if k not in expected]  # fmt: skip
    problems += [f"{where}: {k} is listed twice in {name}" for k in dict.fromkeys(keys)
                 if keys.count(k) > 1]  # fmt: skip
    problems += [f"{where}: {key(x)} in {name} differs from its derivation" for x in recorded
                 if key(x) in expected and expected[key(x)] != x]  # fmt: skip
    if len(set(keys)) == len(keys) and keys != sorted(keys):
        problems.append(f"{where}: {name} is not in canonical order (by run id, then path)")
    return problems
