"""Label study-3 run directories with the frozen judge, every run sealed, in batches bracketed by
the control items and accepted only once the controls after them passed
(docs/study3-preregistration.md, "Judge and audit"; amendment 7, "Labeling").

    python -m zero_field_refusal.analysis.study3_label_cli --judge wildguard --controls RECORD \
        [--batch-size 20] [--workers N] [--sealed-dir DIR] [--reconcile REASON] RUN_DIR ...

Study 3's only labeling command. It labels as ``analysis.study2_label_cli`` labels a sealed run,
and reuses its machinery unchanged: the setup (``.env``, the judge config configs/judge/NAME.yaml,
the WildGuard token check, each run's harmful prompt set, a ``JudgeClient`` per label file),
``run_judging.label_runs`` inside the capture (stdout, stderr and logging into the sealed record),
the allowlisted statuses ``{"run_id", "sealed": true, "status"}`` (``sealed_status``: technical
facts only) and the sealed record (``write_sealed_record`` under ``sealed_directory``).
``run_judging`` is a GPU path and does not change: it labels a study-3 run (a scheduled training
run) at the conditions of the evaluations it recorded, ``step-25``, over the run's 521 harmful
prompts, with its receipts and rounds records.

**Which runs** (``refusal``, from ``config.yaml`` and, for a confirmation run, ``meta.json``):
every run given must be a regular study-3 run, ``experiment: study3`` with a ``schedule`` and the
role ``validation`` or ``confirmation`` (``study3.role``), not K-only; and a confirmation run only
once its candidate's analysis freeze is open (``study3_freeze.freeze_open``: no confirmation label
before the freeze; the candidate is the one its campaign header names in ``meta.json``). Any other
run, a K-only run (it generated no harmful response) or one whose records do not read, is refused,
fail closed: the command then labels nothing and prints each refused run's id and why.

**The ledger** (``<sealed dir>/study3-ledger/``, 0700, files 0600 written whole and synced) keeps
the labeling's state across invocations, so that an interruption, a restart or a quarantine
changes nothing that was decided:
- ``batch-<id>.json`` per batch: its runs (fixed once the batch is made), its control ``changes``
  (0, 1, 2), its ``judge_mark``, per run the ``rounds`` ``charged`` (answered rounds of label files
  moved to the quarantine) and ``lost`` (rounds that failed on the judge, ``RoundFailed``), every
  attempt (counted, ``end`` null, before anything is sent), its ``reconciliations`` and its
  ``base``: the sha256 of every run's judge files (receipt, label file, rounds record) as the last
  ended attempt or reconciliation left them, null while an attempt runs;
- ``accepted-<id>-<attempt>.json``, the **acceptance**, written once and only after the controls
  after the attempt passed, before the batch file records the attempt's end: per run whose labeling
  completed (``sealed_status`` labeled) the sha256 of its label file and of its receipt.
A run's labels count only through an acceptance that names the run with the judge and both
sha256 as the files are now (``accepted``); ``study3_spread`` requires it too.

**Batches.** A run already in a ledger batch is labeled with that batch (all its runs, as the
ledger names them); the other runs given form new batches of ``--batch-size`` (20), in the order
given. A run of the batch already accepted, its files unchanged, is labeled and left alone. The
others go on only when the batch can be reconciled with its files: its ``base`` is set (no attempt
was cut off) and their judge files are the base's. Otherwise (a process stopped mid-attempt, a file
changed since) the batch is an **unfinished batch, paused for reconciliation**: its open runs are
``UNFINISHED``, nothing is sent, moved or deleted. ``--reconcile REASON`` (one line) records a
reconciliation of the paused batches of the runs given (the reason and the files as they are; the
base becomes those files) and lets them go on. A run with judge files that no ledger batch holds
(labels made outside this command) is ``UNFINISHED`` too, and ``--reconcile`` does not take it.
An unresolved **judge mark** in any batch refuses the whole invocation: labeling stays stopped
until it is diagnosed (amendment 7: a documented repair; not this command's).

**An attempt**: 1. the control items (``study3_controls.load_controls``: 40 archived pairs, each
verified against its two sha256), judged by a client of the same judge config through a temporary
label file in the sealed directory, removed after (``judge_controls``); 2. the labeling, each run
with the rounds left to it (``LABEL_ROUNDS`` less its ``charged`` rounds; none left: not labeled,
"not complete"), by ``label_runs`` per group of equal rounds; 3. the controls again, with the
WildGuard token check first (the re-probe). A **change** is a control whose label differs from its
expected one on the two-class outcome, harmful (``harmful_compliance``) or not (an unparsed output
is a change); the exact three-class agreement is recorded, gating nothing. Then:
- *passed*: the acceptance, the statuses (``sealed_status``);
- *a change* (before or after): ``changes`` + 1; the judge files of the batch's open runs (a
  ``busy`` run apart) move, receipt first, to ``<sealed dir>/quarantine/<UTC stamp>-<batch
  id>-attempt<a>/<run id>/``, each listed with its sha256, and their answered rounds are charged
  (the rounds record's ``used``), so that no quarantine resets a run's budget of three. The first
  change after the labeling relabels the batch once, at once; the first change before it stops
  with ``CONTROL_CHANGED`` (the next invocation is the one relabeling); the second change of the
  batch, in this invocation or a later one, is the judge's technical mark: its runs are "not
  complete" and labeling stops. No retry waits for the controls to agree;
- *the judge failed* (a failed round, a control judgment or token check that fails: "not
  complete"; an exception of the call: ``ERROR``): labeling stops, and nothing is moved: the
  answered rounds and parsed labels stay, the failed round is counted ``lost`` with its traceback
  in the sealed record (``run_judging`` charges it; giving it back needs a documented fault, a
  decision this command does not take). The next invocation resumes the batch from its base.
The batches after a stop are "not attempted".

**Printed**, nothing else of a run: per control check of a batch ``{"batch": id, "controls": C}``,
C ``controls ok``, ``controls changed`` or ``controls not judged``; per run its allowlisted line;
then ``{"runs", "token_check"}`` and the sealed record's path.

**The sealed record**, one per invocation (kind ``study3-labeling``, 0600 in a 0700 directory Git
cannot track): the command, the code, the judge, the control record (path and sha256), the batch
size, the invocation's stamp (the ledger's attempts name it), per batch each attempt (the control
labels by item id, the changes and the exact disagreements, the summaries, the captured output
and any traceback, what it quarantined) and every run's status.

**Exit.** Non-zero when a run is refused, a judge mark stands or any run's status is not
"labeled", with a message that names the statuses, the batch of a judge mark and the record's
path, never a count.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..config import load_config
from ..judge import JudgeClient, check_wildguard_tokens
from ..paths import REPO_ROOT, load_env, runs_root
from ..run_judging import (
    LABEL_ROUNDS,
    NOT_ATTEMPTED,
    is_scheduled,
    label_runs,
    receipt_path,
    rounds_path,
)
from ..tracking import git_state
from .evidence import HARMFUL, label_path
from .study2_label_cli import (
    ERROR,
    INCOMPLETE,
    LABELED,
    NOT_ATTEMPTED_STATUS,
    STATUSES,
    _captured,
    _round_failed,
    harmful_prompts,
    sealed_directory,
    sealed_status,
    write_sealed_record,
)
from .study3_controls import load_controls
from .study3_records import EXPERIMENT, ROLES

CONTROL_CHANGED = "control changed: the batch is not labeled (details sealed)"
UNFINISHED = "unfinished batch: paused for reconciliation, nothing deleted (details sealed)"
STATUSES3 = (*STATUSES, CONTROL_CHANGED, UNFINISHED)
CONTROLS_OK, CONTROLS_CHANGED, CONTROLS_UNJUDGED = (
    "controls ok", "controls changed", "controls not judged")  # fmt: skip
REGULAR_ROLES = ("validation", "confirmation")
QUARANTINE = "quarantine"  # under the sealed directory
LEDGER = "study3-ledger"  # under the sealed directory
KIND = "study3-labeling"
BATCH_KIND, ACCEPTANCE_KIND = "study3-labeling-batch", "study3-labeling-acceptance"
DEFAULT_BATCH = 20
ROUNDS_USED = "its labeling rounds are used, quarantined ones included: nothing is sent"
# the kinds of an attempt's end
OK, CHANGED_BEFORE, CHANGED_AFTER, UNJUDGED, FAILED = (
    "ok", "changed_before", "changed_after", "judge_failed", "exception")  # fmt: skip


class LedgerError(ValueError):
    """A ledger file that does not read: the invocation refuses (fail closed)."""


def refusal(run_dir: Path, freeze_gate: Callable[[int], Sequence[str]]) -> str | None:
    """Why a run is refused (None: a run to label; see the module docstring). ``freeze_gate``:
    a candidate's analysis-freeze problems (``study3_freeze.freeze_open``), none when open."""
    try:
        cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
        if not isinstance(cfg, DictConfig):
            return "unclassifiable: its config.yaml is no mapping"
        experiment, scheduled = cfg.get("experiment"), is_scheduled(cfg)
        role = OmegaConf.select(cfg, "study3.role", default=None)
        k_only = OmegaConf.select(cfg, "eval.k_only", default=False)
    except Exception:  # a config that does not read: fail closed
        return "unclassifiable: its config.yaml does not read"
    if experiment != EXPERIMENT or not scheduled or role not in ROLES:
        return "not a study-3 run"
    if k_only is not False or role not in REGULAR_ROLES:
        return f"a K-only study-3 run (role {role}): it has no harmful responses to label"
    if role == "confirmation":
        candidate = _candidate(Path(run_dir))
        if candidate is None:
            return "unclassifiable: a confirmation run whose meta.json names no candidate"
        if freeze_gate(candidate):
            return (f"a confirmation run of candidate {candidate} before its analysis freeze: no "
                    "confirmation label is made before it")  # fmt: skip
    return None


def _candidate(run_dir: Path) -> int | None:
    """The candidate a run's campaign header names in its ``meta.json``, or None."""
    try:
        header = json.loads((run_dir / "meta.json").read_text())["campaign"]["header"]
        candidate = header["candidate"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return candidate if type(candidate) is int and candidate in (1, 2) else None


# ------------------------------------------------------------------ the controls


def judge_controls(
    make_client: Callable[[Path], JudgeClient], directory: Path, controls: Sequence[Mapping]
) -> dict[str, str | None]:
    """The label of each control item, by item id: one request per item through a client from
    ``make_client`` writing to a label file in a temporary folder of ``directory`` (the sealed
    directory), removed after, so that every check judges anew. A request that fails raises."""
    items = [{key: c[key] for key in ("item_id", "prompt", "response")} for c in controls]
    with tempfile.TemporaryDirectory(prefix="controls-", dir=directory) as folder:
        records = make_client(Path(folder) / "controls.jsonl").label_items(items)
    return {r["item_id"]: r.get("label") for r in records}


def changed_controls(labels: Mapping[str, str | None], controls: Sequence[Mapping]) -> list[str]:
    """The ids of the control items whose label differs from the expected one on the two-class
    outcome, harmful or not; an unparsed output (no label) is a change."""
    return [c["item_id"] for c in controls
            if labels.get(c["item_id"]) is None
            or (labels[c["item_id"]] == HARMFUL) != (c["expected_label"] == HARMFUL)]  # fmt: skip


def exact_mismatches(labels: Mapping[str, str | None], controls: Sequence[Mapping]) -> list[str]:
    """The ids whose three-class label is not the expected one: a diagnostic, gating nothing."""
    return [c["item_id"] for c in controls if labels.get(c["item_id"]) != c["expected_label"]]


# ------------------------------------------------------------------ files and the ledger


def judge_files(run_dir: Path, judge: str) -> list[Path]:
    """A run's judge files, in the order they are quarantined: receipt, label file, rounds."""
    return [receipt_path(run_dir, judge), label_path(run_dir, judge), rounds_path(run_dir, judge)]


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def files_now(run_dir: Path, judge: str) -> list[str | None]:
    """The sha256 of a run's judge files (``judge_files``' order), None for one that is absent."""
    return [_digest(path) for path in judge_files(run_dir, judge)]


def label_hashes(run_dir: Path, judge: str) -> dict[str, str | None]:
    """The sha256 of a run's label file and of its receipt, as an acceptance binds them."""
    return {"labels_sha256": _digest(label_path(run_dir, judge)),
            "receipt_sha256": _digest(receipt_path(run_dir, judge))}  # fmt: skip


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _write_whole(path: Path, data: Mapping[str, Any], *, once: bool = False) -> None:
    """``data`` as JSON at ``path`` (0600), whole or not at all, synced; ``once``: never over
    another file (``FileExistsError``)."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    handle = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w") as out:
        os.fchmod(out.fileno(), 0o600)
        out.write(json.dumps(data, indent=1) + "\n")
        out.flush()
        os.fsync(out.fileno())
    if once:
        os.link(partial, path)  # fails when the file is there
        partial.unlink()
    else:
        os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _ledger_files(sealed_dir: Path, prefix: str) -> list[dict[str, Any]]:
    folder = Path(sealed_dir) / LEDGER
    found = []
    for path in sorted(folder.glob(f"{prefix}-*.json")) if folder.is_dir() else []:
        try:
            found.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            raise LedgerError(f"{path} does not read: the ledger is refused") from None
    return found


def read_batches(sealed_dir: Path) -> list[dict[str, Any]]:
    """Every batch of the ledger, in the order of their ids (their making)."""
    return _ledger_files(sealed_dir, "batch")


def acceptances(sealed_dir: Path) -> list[dict[str, Any]]:
    """Every acceptance of the ledger."""
    return _ledger_files(sealed_dir, "accepted")


def accepted(sealed_dir: Path, run_dir: Path, judge: str,
             found: Sequence[Mapping[str, Any]] | None = None) -> bool:  # fmt: skip
    """Whether an acceptance names the run with this judge and the sha256 of its label file and
    receipt as they are now (``found``: the acceptances, read once by a caller of many runs)."""
    now = label_hashes(Path(run_dir), judge)
    if None in now.values():
        return False
    found = acceptances(sealed_dir) if found is None else found
    return any(a.get("judge") == judge and (a.get("runs") or {}).get(Path(run_dir).name) == now
               for a in found)  # fmt: skip


def _write_batch(sealed_dir: Path, batch: Mapping[str, Any]) -> None:
    _write_whole(Path(sealed_dir) / LEDGER / f"batch-{batch['batch']}.json", batch)


def _new_batch(runs: Sequence[Path], judge: str) -> dict[str, Any]:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    digest = hashlib.sha256(",".join(sorted(r.name for r in runs)).encode()).hexdigest()[:8]
    return {"kind": BATCH_KIND, "batch": f"{stamp}-{digest}", "judge": judge,
            "created_at": _now(), "runs": [{"run_id": r.name, "run_dir": str(r)} for r in runs],
            "changes": 0, "judge_mark": False,
            "rounds": {r.name: {"charged": 0, "lost": 0} for r in runs},
            "attempts": [], "reconciliations": [],
            "base": {"from": "made",
                     "files": {r.name: files_now(r, judge) for r in runs}}}  # fmt: skip


def _used(run_dir: Path, judge: str) -> int:
    """The rounds a run's rounds record counts (0 without one that reads)."""
    try:
        used = json.loads(rounds_path(run_dir, judge).read_text()).get("used")
    except (OSError, ValueError, AttributeError):
        return 0
    return used if type(used) is int and used >= 0 else 0


def quarantine(sealed_dir: Path, judge: str, runs: Iterable[Path], batch: str,
               attempt: int) -> list[dict[str, Any]]:  # fmt: skip
    """Move the judge files of ``runs`` into the batch attempt's quarantine folder (module
    docstring), receipt first; each moved file with its sha256."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    base = Path(sealed_dir) / QUARANTINE / f"{stamp}-{batch}-attempt{attempt}"
    moved = []
    for run in runs:
        for path in judge_files(run, judge):
            if not path.is_file():
                continue
            target = base / run.name / path.name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            digest = _digest(path)
            shutil.move(path, target)
            moved.append({"run_id": run.name, "from": str(path), "to": str(target),
                          "sha256": digest})  # fmt: skip
    return moved


# ------------------------------------------------------------------ an attempt


@dataclass
class Attempt:
    """One attempt of a batch: how it ended (``OK``, ``CHANGED_BEFORE``, ``CHANGED_AFTER``,
    ``UNJUDGED``, ``FAILED``), what is printed of its control checks, and what is sealed."""

    number: int
    end: str = OK
    shown: list[str] = field(default_factory=list)
    summaries: list[dict[str, Any] | None] = field(default_factory=list)
    sealed: dict[str, Any] = field(default_factory=dict)


def _check(attempt: Attempt, where: str, output: io.StringIO, make_client, sealed_dir, controls,
           probe=None) -> bool | None:  # fmt: skip
    """One control check, captured: True when no control changed on the two-class outcome,
    False when one did, None when the judge failed (its traceback sealed)."""
    try:
        with _captured(output):
            if probe is not None:
                probe()
            labels = judge_controls(make_client, sealed_dir, controls)
    except BaseException:  # a request, a timeout or the token check failed: sealed
        attempt.sealed[f"controls_{where}"] = {"traceback": traceback.format_exc()}
        attempt.shown.append(CONTROLS_UNJUDGED)
        return None
    changed = changed_controls(labels, controls)
    exact = exact_mismatches(labels, controls)  # a diagnostic
    attempt.sealed[f"controls_{where}"] = {"labels": labels, "changed": changed,
                                           "exact_mismatches": exact}  # fmt: skip
    attempt.shown.append(CONTROLS_CHANGED if changed else CONTROLS_OK)
    return not changed


def _label(batch: Sequence[Path], rounds: Mapping[Path, int], judge: str, load_prompts,
           make_client) -> list[dict[str, Any]]:  # fmt: skip
    """``label_runs`` per group of runs with the same rounds left, in the batch's order of the
    groups; a run with none left is not labeled; after a failed round, the rest not attempted."""
    found: dict[Path, dict[str, Any]] = {}
    failed = False
    for left in dict.fromkeys(rounds[run] for run in batch):
        group = [run for run in batch if rounds[run] == left]
        if failed:
            found |= {run: {"run_id": run.name, "skipped": NOT_ATTEMPTED} for run in group}
        elif left < 1:
            found |= {run: {"run_id": run.name, "skipped": ROUNDS_USED} for run in group}
        else:
            summaries = label_runs(group, judge, load_prompts, make_client,
                                   rounds=min(left, LABEL_ROUNDS))  # fmt: skip
            found |= dict(zip(group, summaries, strict=True))
            failed = _round_failed(summaries)
    return [found[run] for run in batch]


def run_attempt(number: int, batch: Sequence[Path], rounds: Mapping[Path, int], judge: str,
                load_prompts, make_client, controls, sealed_dir: Path,
                probe=None) -> Attempt:  # fmt: skip
    """Controls before, the labeling (``rounds``: each run's rounds left), controls after
    (module docstring, "An attempt")."""
    attempt, output = Attempt(number), io.StringIO()
    try:
        before = _check(attempt, "before", output, make_client, sealed_dir, controls)
        if before is not True:
            attempt.end = UNJUDGED if before is None else CHANGED_BEFORE
            return attempt
        attempt.summaries = [None] * len(batch)
        try:
            with _captured(output):
                attempt.summaries = _label(batch, rounds, judge, load_prompts, make_client)
        except BaseException:  # every exception of the call: its details are sealed
            attempt.sealed["traceback"] = traceback.format_exc()
            attempt.end = FAILED
        if attempt.end == OK and _round_failed(attempt.summaries):
            attempt.end = UNJUDGED
        if attempt.end == OK:
            after = _check(attempt, "after", output, make_client, sealed_dir, controls, probe)
            attempt.end = OK if after else UNJUDGED if after is None else CHANGED_AFTER
        return attempt
    finally:
        attempt.sealed |= {"attempt": number, "end": attempt.end, "summaries": attempt.summaries,
                           "output": output.getvalue()}  # fmt: skip


# ------------------------------------------------------------------ a batch


@dataclass
class Outcome:
    """What a batch gives: each run's status, the control lines, the sealed attempts and whether
    labeling stops."""

    batch: str | None
    statuses: dict[Path, str] = field(default_factory=dict)
    shown: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    stop: bool = False
    mark: bool = False


def label_batch(batch: dict[str, Any], judge: str, load_prompts, make_client, controls,
                sealed_dir: Path, *, invocation: str, probe=None, reconcile: str | None = None,
                found: Sequence[Mapping[str, Any]] = ()) -> Outcome:  # fmt: skip
    """One ledger batch (module docstring, "Batches"): its accepted runs left alone, the others
    paused, reconciled or labeled through attempts until it is accepted, paused or stopped; the
    ledger written before each attempt and at its end."""
    members = [Path(r["run_dir"]) for r in batch["runs"]]
    out = Outcome(batch["batch"])
    out.statuses = {run: LABELED for run in members if accepted(sealed_dir, run, judge, found)}
    open_ = [run for run in members if run not in out.statuses]
    if not open_:
        return out
    base = batch["base"]
    if base is None or any(files_now(r, judge) != base["files"].get(r.name) for r in open_):
        if reconcile is None:
            out.statuses |= {run: UNFINISHED for run in open_}
            return out
        files = {r.name: files_now(r, judge) for r in members}
        batch["reconciliations"].append({"at": _now(), "invocation": invocation,
                                         "reason": reconcile, "files": files})  # fmt: skip
        batch["base"] = {"from": "reconciliation", "files": files}
        _write_batch(sealed_dir, batch)
    while True:
        number = len(batch["attempts"]) + 1
        batch["attempts"].append({"number": number, "invocation": invocation,
                                  "started_at": _now(), "end": None})  # fmt: skip
        batch["base"] = None  # counted before anything is sent
        _write_batch(sealed_dir, batch)
        left = {run: LABEL_ROUNDS - batch["rounds"][run.name]["charged"] for run in open_}
        attempt = run_attempt(number, open_, left, judge, load_prompts, make_client, controls,
                              sealed_dir, probe)  # fmt: skip
        entry = batch["attempts"][-1]
        summaries = dict(zip(open_, attempt.summaries or [None] * len(open_), strict=True))
        for run, summary in summaries.items():
            if str((summary or {}).get("error", "")).startswith("RoundFailed:"):
                batch["rounds"][run.name]["lost"] += 1
        if attempt.end in (CHANGED_BEFORE, CHANGED_AFTER):
            batch["changes"] += 1
            held = [run for run in open_ if not (summaries[run] or {}).get("busy")
                    and any(path.is_file() for path in judge_files(run, judge))]  # fmt: skip
            for run in held:  # answered rounds stay charged: no quarantine resets the budget
                batch["rounds"][run.name]["charged"] += _used(run, judge)
            moved = quarantine(sealed_dir, judge, held, batch["batch"], number)
            attempt.sealed["quarantined"] = entry["quarantined"] = moved
            batch["judge_mark"] = batch["changes"] >= 2
        files = {r.name: files_now(r, judge) for r in members}
        entry |= {"end": attempt.end, "ended_at": _now(), "files": files}
        if attempt.end == OK:
            statuses = {run: sealed_status(summaries[run], run, judge) for run in open_}
            hashes = {run: label_hashes(run, judge) for run in open_}
            runs = {run.name: hashes[run] for run, status in statuses.items()
                    if status == LABELED and None not in hashes[run].values()}  # fmt: skip
            if runs:  # first the acceptance, then the attempt's end
                name = f"accepted-{batch['batch']}-{number}.json"
                _write_whole(Path(sealed_dir) / LEDGER / name,
                             {"kind": ACCEPTANCE_KIND, "batch": batch["batch"], "attempt": number,
                              "judge": judge, "accepted_at": _now(), "runs": runs},
                             once=True)  # fmt: skip
            entry["accepted"] = sorted(runs)
        batch["base"] = {"from": f"attempt {number}", "files": files}
        _write_batch(sealed_dir, batch)
        out.attempts.append(attempt.sealed)
        out.shown += attempt.shown
        if attempt.end == OK:
            out.statuses |= statuses
            return out
        if batch["judge_mark"]:
            out.statuses |= {run: INCOMPLETE for run in open_}
            out.stop = out.mark = True
            return out
        if attempt.end == CHANGED_AFTER:
            continue  # the first change after the labeling: relabeled once, at once
        status = {CHANGED_BEFORE: CONTROL_CHANGED, FAILED: ERROR}.get(attempt.end, INCOMPLETE)
        out.statuses |= {run: status for run in open_}
        out.stop = True
        return out


# ------------------------------------------------------------------ the invocation


def label_study3(
    run_dirs: Iterable[Path],
    judge: str,
    load_prompts: Callable[[DictConfig], Mapping[str, str]],
    make_client: Callable[[Path], JudgeClient],
    controls: Sequence[Mapping[str, str]],
    *,
    sealed_dir: Path,
    freeze_gate: Callable[[int], Sequence[str]],
    batch_size: int = DEFAULT_BATCH,
    token_check: Mapping[str, Any] | None = None,
    probe: Callable[[], Any] | None = None,
    reconcile: str | None = None,
    controls_record: Mapping[str, str] | None = None,
    command: list[str] | None = None,
) -> str | None:
    """Label ``run_dirs`` (module docstring), print what it allows, write the sealed record;
    returns the exit message (None: exit 0). ``load_prompts`` and ``make_client`` are
    ``run_judging.label_runs``'; ``controls``: ``load_controls``' items; ``probe()`` (the token
    check, None: none) raises when the judge does not answer; ``reconcile``: the reason of a
    reconciliation of the paused batches; ``controls_record``: the control record's path and
    sha256, for the sealed record."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError(f"the batch size must be a positive integer, not {batch_size!r}")
    runs = list(dict.fromkeys(Path(run_dir) for run_dir in run_dirs))
    refused = [(run, why) for run in runs if (why := refusal(run, freeze_gate)) is not None]
    if refused:
        for run, why in refused:
            print(json.dumps({"run_id": run.name, "refused": why}))
        return (f"{len(refused)} of {len(runs)} runs refused: not regular study-3 runs this "
                "command may label; nothing was labeled")  # fmt: skip
    directory = Path(sealed_dir)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    try:
        batches, found = read_batches(directory), acceptances(directory)
    except LedgerError as exc:
        return f"{exc}; nothing was labeled"
    marked = [b["batch"] for b in batches if b.get("judge_mark")]
    if marked:
        return (f"The judge's technical mark of batch {marked[0]} is unresolved: labeling stays "
                "stopped until it is diagnosed; nothing was labeled")  # fmt: skip
    holder = {r["run_id"]: b for b in batches for r in b["runs"] if b.get("judge") == judge}
    existing: dict[str, dict[str, Any]] = {}
    loose: list[Path] = []
    new: list[Path] = []
    for run in runs:
        batch = holder.get(run.name)
        if batch is not None:
            existing.setdefault(batch["batch"], batch)
        elif any(path.exists() for path in judge_files(run, judge)):
            loose.append(run)  # labels no batch explains
        else:
            new.append(run)
    invocation = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    planned: list[dict[str, Any] | list[Path]] = [*existing.values(), *(
        new[i : i + batch_size] for i in range(0, len(new), batch_size))]  # fmt: skip
    outcomes: list[Outcome] = [Outcome(None, {run: UNFINISHED for run in loose})] if loose else []
    stopped = False
    for item in planned:
        if stopped:
            members = [Path(r["run_dir"]) for r in item["runs"]] if isinstance(item, dict) else item
            outcomes.append(Outcome(None, {run: NOT_ATTEMPTED_STATUS for run in members}))
            continue
        batch = item if isinstance(item, dict) else _new_batch(item, judge)
        if not isinstance(item, dict):
            _write_batch(directory, batch)
        outcome = label_batch(batch, judge, load_prompts, make_client, controls, directory,
                              invocation=invocation, probe=probe, reconcile=reconcile,
                              found=found)  # fmt: skip
        outcomes.append(outcome)
        found = acceptances(directory)
        stopped = outcome.stop
    flat = [(run, status) for o in outcomes for run, status in o.statuses.items()]
    marks = [o.batch for o in outcomes if o.mark]
    path = write_sealed_record(directory, KIND, judge, {
        "what": "The sealed record of a study-3 labeling (analysis.study3_label_cli): per batch "
                "each attempt's control labels, summaries, captured output and tracebacks, and "
                "what it quarantined; every run's status. The ledger holds the batches' state.",
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "invocation": invocation,
        "command": command,
        "code": git_state(),
        "judge": judge,
        "rounds": LABEL_ROUNDS,
        "batch_size": batch_size,
        "reconcile": reconcile,
        "controls": dict(controls_record) if controls_record is not None else None,
        "batches": [{"batch": o.batch, "run_ids": [run.name for run in o.statuses],
                     "attempts": o.attempts, "judge_mark": o.mark} for o in outcomes],
        "runs": [{"run_dir": str(run), "run_id": run.name, "status": status}
                 for run, status in flat],
    })  # fmt: skip
    for outcome in outcomes:
        for shown in outcome.shown:
            print(json.dumps({"batch": outcome.batch, "controls": shown}))
        for run, status in outcome.statuses.items():
            print(json.dumps({"run_id": run.name, "sealed": True, "status": status}))
    print(json.dumps({"runs": len(flat), "token_check": token_check}))
    print(json.dumps({"sealed_record": str(path)}))
    failures = []
    if marks:
        failures.append(f"The judge's technical mark: a control changed twice in batch "
                        f"{marks[0]}; labeling stops until it is diagnosed")  # fmt: skip
    statuses = {status for _, status in flat}
    unfinished = [status for status in STATUSES3 if status != LABELED and status in statuses]
    if unfinished:
        failures.append(f"Runs not labeled yet ({'; '.join(unfinished)}): their details are "
                        f"sealed in {path}")  # fmt: skip
    return "\n".join(failures) or None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m zero_field_refusal.analysis.study3_label_cli",
                                     description=__doc__.splitlines()[0])  # fmt: skip
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--judge", required=True, help="a judge config in configs/judge/")
    parser.add_argument("--controls", required=True, type=Path,
                        help="the control record (docs/study3-judge-controls.json)")  # fmt: skip
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sealed-dir", type=Path,
                        help="where the sealed records and the ledger go (default "
                             "$RUNS_DIR/sealed)")  # fmt: skip
    parser.add_argument("--reconcile", metavar="REASON",
                        help="reconcile the paused batches of these runs (one line)")  # fmt: skip
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.reconcile is not None and (not args.reconcile.strip() or "\n" in args.reconcile
                                       or len(args.reconcile) > 1000):  # fmt: skip
        parser.error("--reconcile takes one line of text, at most 1000 characters")
    load_env()
    try:
        sealed_dir = sealed_directory(args.sealed_dir)
        data = args.controls.read_bytes()
        record = json.loads(data)
        controls = load_controls(record, runs_root() / record["source"]["directory"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    cfg = load_config(f"judge/{args.judge}")
    client = JudgeClient.from_config(cfg)
    if record["identity"] != client.identity:
        parser.error(f"the control items' expected labels are of the judge {record['identity']}, "
                     f"not of {args.judge}'s {client.identity}")  # fmt: skip
    tokens, probe = None, None
    if client.backend == "wildguard":  # before any item: a doubled BOS would change every label

        def probe() -> dict[str, Any]:
            return check_wildguard_tokens(os.environ["JUDGE_API_BASE"], client.model,
                                          os.environ.get("JUDGE_API_KEY"))  # fmt: skip

        tokens = probe()

    @functools.cache  # one reading of the records per candidate, not one per run
    def freeze_gate(candidate: int) -> tuple[str, ...]:
        from .study3_freeze import freeze_open  # it reads the route: here only

        return freeze_open(candidate, runs_root(), REPO_ROOT)

    failure = label_study3(
        args.run_dirs,
        args.judge,
        harmful_prompts,
        lambda path: JudgeClient.from_config(cfg, cache_path=path, max_workers=args.workers),
        controls,
        sealed_dir=sealed_dir,
        freeze_gate=freeze_gate,
        batch_size=args.batch_size,
        token_check=tokens,
        probe=probe,
        reconcile=args.reconcile,
        controls_record={"path": str(args.controls), "sha256": hashlib.sha256(data).hexdigest()},
        command=sys.argv if argv is None else [parser.prog, *argv],
    )
    if failure:
        sys.exit(failure)


if __name__ == "__main__":
    main()
