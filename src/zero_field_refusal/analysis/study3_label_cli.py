"""Label study-3 run directories with the frozen judge, every run sealed, in batches bracketed by
the control items (docs/study3-preregistration.md, "Judge and audit").

    python -m zero_field_refusal.analysis.study3_label_cli --judge wildguard --controls RECORD \
        [--batch-size 20] [--workers N] [--sealed-dir DIR] RUN_DIR [RUN_DIR ...]

Study 3's only labeling command. It labels as ``analysis.study2_label_cli`` labels a sealed run,
and reuses its machinery unchanged: the setup (``.env``, the judge config configs/judge/NAME.yaml,
the WildGuard token check, each run's harmful prompt set, a ``JudgeClient`` per label file),
``run_judging.label_runs`` at three rounds inside the capture (stdout, stderr and logging into the
sealed record), the allowlisted statuses ``{"run_id", "sealed": true, "status"}``
(``sealed_status``: technical facts only) and the sealed record (``write_sealed_record`` under
``sealed_directory``). ``run_judging`` is a GPU path and does not change: it labels a study-3 run
(a scheduled training run) at the conditions of the evaluations it recorded, ``step-25``, over
the run's 521 harmful prompts, with its receipts and rounds records.

**Which runs** (``refusal``, from ``config.yaml`` and, for a confirmation run, ``meta.json``):
every run given must be a regular study-3 run, ``experiment: study3`` with a ``schedule`` and the
role ``validation`` or ``confirmation`` (``study3.role``), not K-only; and a confirmation run only
once its candidate's analysis freeze is open (``study3_freeze.freeze_open``: no confirmation label
before the freeze; the candidate is the one its campaign header names in ``meta.json``). Any other
run, a K-only run (it generated no harmful response) or one whose records do not read, is refused,
fail closed: the command then labels nothing and prints each refused run's id and why.

**Batches.** The runs are labeled in the order given, ``--batch-size`` at a time (20 by default).
The control items of the control record (``study3_controls.load_controls``: 40 archived pairs,
each verified against its two sha256) are judged by a client of the same judge config right before
and right after each batch's labeling (``judge_controls``), through a temporary label file in the
sealed directory, removed after; their labels go to the sealed record only. Per attempt of a batch:
1. *Controls before.* A label other than its expected one (an unparsed output included): the
   batch is not labeled, its runs ``CONTROL_CHANGED``, and labeling stops.
2. *The labeling*, inside the capture, as study 2's sealed call.
3. *Controls after*, inside the capture, with the WildGuard token check first (the re-probe).
A change after (or a change before the second attempt) quarantines the runs' judge files written
by the attempt (``quarantine``) and the batch is relabeled once; a second change is a technical
mark of the judge: labeling stops, every run of the batch is "not complete" and the sealed record
names the batch (``judge_mark``). A judge failure (a failed round, an exception of the call, a
control judgment or token check that fails) makes every run of the batch "not complete" (an
exception of the call: ``ERROR``) and stops; the batches after a stop are "not attempted".

**Quarantine.** A batch's labels count only when the controls after it passed. After an attempt
whose controls after did not pass, the judge files (label file, receipt, rounds record) of every
run of the batch that the attempt wrote (their sha256 before and after it differ; a ``busy`` run,
which another labeling holds, is left alone) are moved, receipt first, into
``<sealed dir>/quarantine/<UTC stamp>-batch<k>-attempt<a>/<run id>/``, each listed in the sealed
record with its sha256: recorded, never deleted. A run whose completed receipt was only kept is
not touched. A relabeling then starts each such run afresh (no label file, no rounds record).

**Printed**, nothing else of a run: per control check of a batch ``{"batch": k, "controls": C}``,
C ``controls ok``, ``controls changed`` or ``controls not judged`` (the judge failed); per run its
allowlisted line; then ``{"runs", "token_check"}`` and the sealed record's path.

**The sealed record**, one per invocation (kind ``study3-labeling``, 0600 in a 0700 directory
Git cannot track): the command, the code, the judge, the control record (path and sha256), the
batch size, and per batch its run ids, each attempt (the control labels before and after by item
id, the summaries, the captured output and any traceback, what it quarantined) and the judge
mark; and every run's status. Opened in the analysis that may open its labels, never before.

**Exit.** Non-zero when a run is refused or any run's status is not "labeled", with a message
that names the statuses, the batch of a judge mark and the record's path, never a count.
"""

from __future__ import annotations

import argparse
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
from ..run_judging import LABEL_ROUNDS, is_scheduled, label_runs, receipt_path, rounds_path
from ..tracking import git_state
from .evidence import label_path
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
STATUSES3 = (*STATUSES, CONTROL_CHANGED)
CONTROLS_OK, CONTROLS_CHANGED, CONTROLS_UNJUDGED = (
    "controls ok", "controls changed", "controls not judged")  # fmt: skip
REGULAR_ROLES = ("validation", "confirmation")
QUARANTINE = "quarantine"  # under the sealed directory
KIND = "study3-labeling"
DEFAULT_BATCH = 20
# the kinds of an attempt's end
OK, CHANGED_BEFORE, CHANGED_AFTER, UNJUDGED, FAILED = (
    "ok", "changed_before", "changed_after", "judge_failed", "exception")  # fmt: skip


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
    """The ids of the control items whose label is not their expected one."""
    return [c["item_id"] for c in controls if labels.get(c["item_id"]) != c["expected_label"]]


def judge_files(run_dir: Path, judge: str) -> list[Path]:
    """A run's judge files, in the order they are quarantined: receipt, label file, rounds."""
    return [receipt_path(run_dir, judge), label_path(run_dir, judge), rounds_path(run_dir, judge)]


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _snapshot(runs: Sequence[Path], judge: str) -> dict[Path, list[str | None]]:
    return {run: [_digest(p) for p in judge_files(run, judge)] for run in runs}


def quarantine(sealed_dir: Path, judge: str, runs: Iterable[Path], batch: int,
               attempt: int) -> list[dict[str, Any]]:  # fmt: skip
    """Move the judge files of ``runs`` into the batch attempt's quarantine folder (module
    docstring), receipt first; each moved file with its sha256."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    base = Path(sealed_dir) / QUARANTINE / f"{stamp}-batch{batch}-attempt{attempt}"
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


@dataclass
class Attempt:
    """One attempt of a batch: how it ended (``OK``, ``CHANGED_BEFORE``, ``CHANGED_AFTER``,
    ``UNJUDGED``, ``FAILED``), what is printed of its control checks, and what is sealed."""

    number: int
    end: str = OK
    shown: list[str] = field(default_factory=list)
    summaries: list[dict[str, Any] | None] = field(default_factory=list)
    written: list[Path] = field(default_factory=list)  # the runs whose judge files it wrote
    sealed: dict[str, Any] = field(default_factory=dict)


def _check(attempt: Attempt, where: str, output: io.StringIO, make_client, sealed_dir, controls,
           probe=None) -> bool | None:  # fmt: skip
    """One control check, captured: True when every label is its expected one, False when one
    changed, None when the judge failed (its traceback sealed)."""
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
    attempt.sealed[f"controls_{where}"] = {"labels": labels, "changed": changed}
    attempt.shown.append(CONTROLS_CHANGED if changed else CONTROLS_OK)
    return not changed


def run_attempt(number: int, batch: Sequence[Path], judge: str, load_prompts, make_client,
                controls, sealed_dir: Path, probe=None) -> Attempt:  # fmt: skip
    """Controls before, the labeling, controls after (module docstring, "Batches")."""
    attempt, output = Attempt(number), io.StringIO()
    try:
        before = _check(attempt, "before", output, make_client, sealed_dir, controls)
        if before is not True:
            attempt.end = UNJUDGED if before is None else CHANGED_BEFORE
            return attempt
        files = _snapshot(batch, judge)
        attempt.summaries = [None] * len(batch)
        try:
            with _captured(output):
                attempt.summaries = list(label_runs(batch, judge, load_prompts, make_client,
                                                    rounds=LABEL_ROUNDS))  # fmt: skip
        except BaseException:  # every exception of the call: its details are sealed
            attempt.sealed["traceback"] = traceback.format_exc()
            attempt.end = FAILED
        if attempt.end == OK and _round_failed(attempt.summaries):
            attempt.end = UNJUDGED
        if attempt.end == OK:
            after = _check(attempt, "after", output, make_client, sealed_dir, controls, probe)
            attempt.end = OK if after else UNJUDGED if after is None else CHANGED_AFTER
        now = _snapshot(batch, judge)
        attempt.written = [run for run, summary in zip(batch, attempt.summaries, strict=True)
                           if now[run] != files[run]
                           and not (summary or {}).get("busy")]  # fmt: skip
        return attempt
    finally:
        attempt.sealed |= {"attempt": number, "end": attempt.end, "summaries": attempt.summaries,
                           "output": output.getvalue()}  # fmt: skip


def label_batch(
    number: int, batch: Sequence[Path], judge: str, load_prompts, make_client, controls,
    sealed_dir: Path, probe=None,
) -> tuple[list[str], dict[str, Any]]:  # fmt: skip
    """A batch's statuses, in order, and its sealed entry (``judge_mark``, ``stop``, the
    attempts and the lines shown)."""
    entry: dict[str, Any] = {
        "batch": number,
        "run_ids": [run.name for run in batch],
        "attempts": [],
        "shown": [],
        "judge_mark": False,
        "stop": False,
    }
    for count in (1, 2):
        attempt = run_attempt(count, batch, judge, load_prompts, make_client, controls,
                              sealed_dir, probe)  # fmt: skip
        if attempt.end != OK and attempt.written:  # its labels do not count: kept apart
            attempt.sealed["quarantined"] = quarantine(sealed_dir, judge, attempt.written,
                                                       number, count)  # fmt: skip
        entry["attempts"].append(attempt.sealed)
        entry["shown"] += attempt.shown
        if attempt.end == OK:
            return [sealed_status(summary, run, judge) for summary, run in
                    zip(attempt.summaries, batch, strict=True)], entry  # fmt: skip
        entry["stop"] = True
        if attempt.end == CHANGED_AFTER and count == 1:
            entry["stop"] = False
            continue  # relabeled once
        if attempt.end in (CHANGED_BEFORE, CHANGED_AFTER) and count == 2:
            entry["judge_mark"] = True  # a repeated change
            return [INCOMPLETE] * len(batch), entry
        status = {CHANGED_BEFORE: CONTROL_CHANGED, FAILED: ERROR}.get(attempt.end, INCOMPLETE)
        return [status] * len(batch), entry
    raise AssertionError("unreachable")


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
    controls_record: Mapping[str, str] | None = None,
    command: list[str] | None = None,
) -> str | None:
    """Label ``run_dirs`` in batches (module docstring), print what it allows, write the sealed
    record; returns the exit message (None: exit 0). ``load_prompts`` and ``make_client`` are
    ``run_judging.label_runs``'; ``controls``: ``load_controls``' items; ``probe()`` (the token
    check, None: none) raises when the judge does not answer; ``controls_record``: the control
    record's path and sha256, for the sealed record."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError(f"the batch size must be a positive integer, not {batch_size!r}")
    runs = [Path(run_dir) for run_dir in run_dirs]
    refused = [(run, why) for run in runs if (why := refusal(run, freeze_gate)) is not None]
    if refused:
        for run, why in refused:
            print(json.dumps({"run_id": run.name, "refused": why}))
        return (f"{len(refused)} of {len(runs)} runs refused: not regular study-3 runs this "
                "command may label; nothing was labeled")  # fmt: skip
    directory = Path(sealed_dir)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    batches = [runs[i : i + batch_size] for i in range(0, len(runs), batch_size)]
    statuses: list[list[str]] = []
    entries: list[dict[str, Any]] = []
    for number, batch in enumerate(batches, 1):
        if entries and entries[-1]["stop"]:
            statuses.append([NOT_ATTEMPTED_STATUS] * len(batch))
            continue
        found, entry = label_batch(number, batch, judge, load_prompts, make_client, controls,
                                   directory, probe)  # fmt: skip
        statuses.append(found)
        entries.append(entry)
    flat = [status for found in statuses for status in found]
    marked = [entry["batch"] for entry in entries if entry["judge_mark"]]
    path = write_sealed_record(directory, KIND, judge, {
        "what": "The sealed record of a study-3 labeling (analysis.study3_label_cli): per batch "
                "its runs, each attempt's control labels, summaries, captured output and "
                "tracebacks, and what it quarantined; every run's status. Opened in the "
                "analysis that may open its labels, never before.",
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": command,
        "code": git_state(),
        "judge": judge,
        "rounds": LABEL_ROUNDS,
        "batch_size": batch_size,
        "controls": dict(controls_record) if controls_record is not None else None,
        "batches": entries,
        "judge_mark": marked,
        "runs": [{"run_dir": str(run), "run_id": run.name, "status": status}
                 for run, status in zip(runs, flat, strict=True)],
    })  # fmt: skip
    for number, (batch, found) in enumerate(zip(batches, statuses, strict=True), 1):
        for shown in next((e["shown"] for e in entries if e["batch"] == number), []):
            print(json.dumps({"batch": number, "controls": shown}))
        for run, status in zip(batch, found, strict=True):
            print(json.dumps({"run_id": run.name, "sealed": True, "status": status}))
    print(json.dumps({"runs": len(runs), "token_check": token_check}))
    print(json.dumps({"sealed_record": str(path)}))
    failures = []
    if marked:
        failures.append(f"The judge's technical mark: a control label changed twice in batch "
                        f"{marked[0]}; labeling stops until it is diagnosed")  # fmt: skip
    unfinished = [status for status in STATUSES3 if status != LABELED and status in flat]
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
                        help="where the sealed record goes (default $RUNS_DIR/sealed)")  # fmt: skip
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
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
        controls_record={"path": str(args.controls), "sha256": hashlib.sha256(data).hexdigest()},
        command=sys.argv if argv is None else [parser.prog, *argv],
    )
    if failure:
        sys.exit(failure)


if __name__ == "__main__":
    main()
