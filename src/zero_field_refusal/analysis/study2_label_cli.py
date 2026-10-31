"""Label study-2 run directories with a judge, the sealed runs' output sealed (amendment 4,
sections 3 and 4).

    python -m zero_field_refusal.analysis.study2_label_cli --judge NAME RUN_DIR [RUN_DIR ...] \
        [--workers N] [--sealed-dir DIR]

Study 2's only labeling command. It labels as scripts/label_runs.py does, with the same setup
(``.env``, the judge config configs/judge/NAME.yaml, the WildGuard token check, the harmful
prompt set of each run's config, a ``JudgeClient`` per label file) and ``run_judging.label_runs``
at three rounds (study 2 counts a missing label only after three), but what it prints of a sealed
run is allowlisted. scripts/label_runs.py prints every run's counts (``unparseable``, ``rounds``,
errors) and is never run on study-2 runs; neither it nor ``run_judging`` changes (GPU paths,
``code_identity``). The command line lives in the analysis package, as
``analysis.study2_audit_cli`` does, because a new file under scripts/ would be a GPU path.

**Sealed runs** (``sealed_reason``), classified from each run's ``config.yaml`` with the
analysis' own helpers: a study-2 run (``run_judging.is_scheduled``) of the pre-warmed arm
(``stage_a.arm_of``), whatever its role; a time-course run (``study2_short.role`` timecourse, as
``study2_short.run_facts`` reads it) of every arm, on a seed of ``study2_short.TIMECOURSE_SEEDS``
(the code constant the fixation key ``timecourse`` is checked against); and a run that cannot be
classified, fail closed: a ``config.yaml`` or ``meta.json`` that does not read, a study-2 run
without a known arm or a seed, a time-course run on another seed, or a run on a time-course seed
whose role is not confirmation. Any other run is open: not a study-2 run, or a study-2 run of a
primary arm outside the time course.

**Two calls.** The open runs are labeled first, by one ``label_runs`` call, and printed exactly
as scripts/label_runs.py prints them. Then the sealed runs, by another call that runs with
stdout and stderr redirected (``redirect_stdout``, ``redirect_stderr``) and logging captured
(every logger's handlers set aside, since a handler may hold the stream it was made with, and
one handler on the root logger): its output, every exception it raises (a ``KeyboardInterrupt``
too) and its traceback go into the sealed record, never to the terminal. When the open call's
judge failed (``RoundFailed``), the sealed call is not made and its runs are "not attempted",
as scripts/label_runs.py leaves the runs after a failed round; so nothing of a sealed run
changes what is printed of an open one. **A judge failure in the sealed call**:
when that call stopped on a failed round (``RoundFailed``, the runs after it
not attempted), or the judge fails its re-probe after the call, every sealed run of the call is
"not complete", whatever happened to it: which run was hit, and whether a retry round was
needed at all, depends on its outputs' parsing. The re-probe is the WildGuard token check
(``probe``, a WildGuard judge only) and, for every backend, one judgment of a fixed synthetic
item (``PROBE_ITEM``, a harmless pair that is never a study item; ``judge_answers``) by a client
of the same config writing to a temporary label file, removed after: a request failure, a
timeout or an answer that does not parse fails it. The re-probe runs inside the capture, its
traceback goes to the sealed record only. A rerun with the judge back labels the runs. Nothing
in the sealed call's path writes below Python's
``sys.stdout`` or ``sys.stderr``: ``run_judging``, the judge client (the openai client over
httpx), OmegaConf, the prompt-set reader and ``stage_a.load_trajectory`` print nothing; their
warnings (``judge.read_jsonl``'s and ``judge.repair_jsonl``'s) go to ``sys.stderr``, their
logging (openai, httpx) to logging, and no subprocess or native library writes to file
descriptor 1 or 2 there.

**Printed per sealed run**, allowlisted: ``{"run_id": …, "sealed": true, "status": S}``, S one
of ``STATUSES``, from technical facts only (``sealed_status``): an ``error``, ``busy`` or "not
attempted" summary; "labeled" when the run's judging receipt names this judge and the label file
as it is now (its sha256), "not complete" otherwise. It never depends on an unparseable or
missing count, nor on the rounds a run used: a completed receipt with missing labels is
"labeled". A receipt written again for its unchanged label file (``restored``) is "not complete",
whatever its rounds (under this command alone, three rounds mean that
outputs stayed unparseable after the second); a rerun keeps it (complete) or resumes it, and it
is "labeled" either way. A run that cannot be labeled yet (no published terminal record) is "not
complete"; one that recorded no evaluation has nothing to label, and is "labeled". The residuals
that only a change of ``run_judging`` (a GPU path) could remove are in docs/operations.md
("Labeling and sealed output").

**The sealed record**, one per invocation with a sealed run:
``<sealed dir>/labeling-<UTC timestamp>-<judge>.json`` (``write_sealed_record``; the directory
0700, the file 0600; ``--sealed-dir``, default ``$RUNS_DIR/sealed``, refused where Git could track
it: ``sealed_directory``): the command, the code revision, each sealed run with its reason,
status and full summary, the captured output and any traceback. Only its path is printed. It is
opened in the final analysis, never before; the details are then read from it, the receipts and
the final analysis. This command never decides whether that analysis is released: its output is
always sealed, the stricter rule.

**Exit.** Non-zero for the open runs exactly as scripts/label_runs.py decides (an error, a busy
run, an unparseable output), with its messages; for the sealed runs only for an error, a busy
run, a run not attempted or not complete, never for an unparseable or missing count, with a
message that names their statuses and the record's path, never a count.

``sealed_directory``, ``write_sealed_record`` and ``sealed_traceback`` are also the sealed
records of scripts/study2_short_summary.py and ``study2_audit_cli draw`` (kinds ``summary`` and
``audit``), for an error on the pre-warmed arm's label rows (``study2_short.SealedLabelError``).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import sys
import tempfile
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..config import load_config
from ..data import load_prompt_set
from ..judge import JudgeClient, check_wildguard_tokens
from ..paths import REPO_ROOT, load_env, runs_root
from ..run_judging import (
    LABEL_ROUNDS,
    NOT_ATTEMPTED,
    RoundFailed,
    is_scheduled,
    label_runs,
    receipt_path,
)
from ..tracking import git_state
from .evidence import label_path
from .stage_a import arm_of
from .study2_short import (
    PREWARMED,
    SEALED_ERROR,
    TIMECOURSE,
    SealedLabelError,
    run_facts,
    seed_role,
)

# The allowlisted statuses of a sealed run, nothing else is printed of it
LABELED = "labeled (details sealed until the final analysis)"
INCOMPLETE = "not complete: rerun the labeling (details sealed)"
ERROR = SEALED_ERROR
BUSY = "busy"
NOT_ATTEMPTED_STATUS = "not attempted"
STATUSES = (LABELED, INCOMPLETE, ERROR, BUSY, NOT_ATTEMPTED_STATUS)
SEALED_DIR = "sealed"  # under $RUNS_DIR
# run_judging's summary of a run that recorded no evaluation (``_label_run``): nothing to label
NOTHING_TO_LABEL = "no evaluation recorded"
UNCLASSIFIABLE = "unclassifiable"
# The judge's re-probe after the sealed call: one fixed, harmless synthetic pair,
# never a study item, judged through a temporary label file.
PROBE_ITEM = {"item_id": "probe/synthetic",
              "prompt": "What is the boiling point of water at sea level?",
              "response": "Water boils at 100 degrees Celsius at sea level."}  # fmt: skip


def default_sealed_dir() -> Path:
    """Where sealed records go by default: ``$RUNS_DIR/sealed`` (``.env`` read first, as every
    study-2 command reads it)."""
    load_env()
    return runs_root() / SEALED_DIR


def in_tracked_tree(path: Path) -> bool:
    """Whether ``path`` lies in the checkout outside its ignored runs/ directory, where Git could
    track it."""
    found, root = Path(path).resolve(), REPO_ROOT.resolve()
    return found.is_relative_to(root) and not found.is_relative_to(root / "runs")


def sealed_directory(given: Path | None) -> Path:
    """The sealed records' directory of a study-2 command: ``given`` (its ``--sealed-dir``), else
    ``default_sealed_dir``. ``ValueError`` for a directory Git could track (``in_tracked_tree``):
    the records hold label-derived details and never enter Git."""
    directory = Path(given) if given is not None else default_sealed_dir()
    if in_tracked_tree(directory):
        raise ValueError(
            f"{directory} lies in the checkout outside runs/: sealed records hold "
            "label-derived details and never enter Git; use $RUNS_DIR/sealed"
        )
    return directory


def write_sealed_record(directory: Path, kind: str, judge: str, record: Mapping[str, Any]) -> Path:
    """Write ``record`` to ``<directory>/<kind>-<UTC timestamp>-<judge>.json`` and return its
    path: the directory made 0700 (an existing one too), the file created new at 0600 and synced.
    Never opened before the final analysis."""
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"{kind}-{stamp}-{judge.replace('/', '_')}.json"
    handle = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(handle, "w") as out:
        os.fchmod(out.fileno(), 0o600)  # whatever the umask
        out.write(json.dumps(record, indent=1, default=str) + "\n")
        out.flush()
        os.fsync(out.fileno())
    return path


def sealed_traceback(exc: BaseException) -> str:
    """The whole traceback of ``exc``; for a ``SealedLabelError``, its original error's first
    (raised ``from None``, Python's own printing leaves it out)."""
    found = "".join(traceback.format_exception(exc))
    if isinstance(exc, SealedLabelError):
        found = "".join(traceback.format_exception(exc.error)) + "\nsealed as:\n" + found
    return found


def sealed_failure(exc: SealedLabelError, path: Path) -> str:
    """What a command shows of a ``SealedLabelError``: the runs, the generic status and the
    sealed record's path (JSON)."""
    return json.dumps({"runs": exc.runs, "sealed": True, "status": SEALED_ERROR,
                       "sealed_record": str(path)})  # fmt: skip


def sealed_reason(run_dir: Path) -> str | None:
    """Why a run's labeling output is sealed: ``prewarmed``, ``timecourse`` or
    ``unclassifiable`` (fail closed); None for an open run (see the module docstring)."""
    try:
        cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
        if not isinstance(cfg, DictConfig):
            return UNCLASSIFIABLE
        if not is_scheduled(cfg):
            return None  # not a study-2 run
        arm, seed = arm_of(cfg), int(cfg.seed)
        role = run_facts(run_dir)["role"]
    except Exception:  # a config or record that does not read: fail closed
        return UNCLASSIFIABLE
    if arm == PREWARMED:
        return "prewarmed"
    on_course = seed_role(seed, timecourse=True) == TIMECOURSE
    if role == TIMECOURSE:
        return TIMECOURSE if on_course else UNCLASSIFIABLE
    if on_course and role != "confirmation":  # on the time course's seeds, its endpoint runs only
        return UNCLASSIFIABLE
    return None


def _receipt(run_dir: Path, judge: str) -> dict[str, Any] | None:
    """The run's judging receipt when it names this judge and the label file as it is now (its
    sha256); None otherwise. Its counts are not read."""
    try:
        receipt = json.loads(receipt_path(run_dir, judge).read_text())
        data = label_path(run_dir, judge).read_bytes()
    except (OSError, ValueError):
        return None
    if not isinstance(receipt, dict) or receipt.get("judge") != judge:
        return None
    return receipt if receipt.get("labels_sha256") == hashlib.sha256(data).hexdigest() else None


def sealed_status(summary: Mapping[str, Any], run_dir: Path, judge: str) -> str:
    """The allowlisted status of a sealed run from its ``label_runs`` summary and its receipt:
    technical facts only (see the module docstring)."""
    if summary.get("error"):
        return ERROR
    if summary.get("busy"):
        return BUSY
    skipped = summary.get("skipped")
    if skipped == NOT_ATTEMPTED:
        return NOT_ATTEMPTED_STATUS
    if skipped == NOTHING_TO_LABEL:
        return LABELED  # its labeling is over: there is nothing to label
    if skipped is not None:  # not labelable yet
        return INCOMPLETE
    if summary.get("restored"):  # whatever its rounds, which would tell its parsing
        return INCOMPLETE
    return LABELED if _receipt(run_dir, judge) is not None else INCOMPLETE


@contextmanager
def _captured(buffer: io.StringIO) -> Iterator[None]:
    """stdout, stderr and logging into ``buffer`` (see the module docstring)."""
    root = logging.getLogger()
    loggers = [root, *(found for found in list(logging.Logger.manager.loggerDict.values())
                       if isinstance(found, logging.Logger))]  # fmt: skip
    saved = [(logger, list(logger.handlers), logger.propagate) for logger in loggers]
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    for logger in loggers:
        logger.handlers = []
        logger.propagate = True
    root.addHandler(handler)
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            yield
    finally:
        for logger, handlers, propagate in saved:
            logger.handlers = handlers
            logger.propagate = propagate


def judge_answers(make_client: Callable[[Path], JudgeClient]) -> None:
    """Raise unless the judge labels ``PROBE_ITEM``: one request through a client from
    ``make_client`` writing to a label file in a temporary directory, removed after (no run's
    label file). A request that fails (past the client's retries of a transient error, a timeout
    included) raises; so does an answer that does not parse."""
    with tempfile.TemporaryDirectory(prefix="study2-label-probe-") as directory:
        (record,) = make_client(Path(directory) / "probe.jsonl").label_items([PROBE_ITEM])
    if record.get("label") is None:
        raise RuntimeError("the judge's answer to the synthetic re-probe does not parse")


def _round_failed(summaries: Iterable[Mapping[str, Any] | None]) -> bool:
    """Whether a ``label_runs`` call stopped on a failed round (the judge gone): a run's
    ``RoundFailed`` error, or the runs after it not attempted."""
    return any(s is not None and (str(s.get("error", "")).startswith(f"{RoundFailed.__name__}:")
                                  or s.get("skipped") == NOT_ATTEMPTED)
               for s in summaries)  # fmt: skip


def _open_failure(summaries: list[dict[str, Any]]) -> str | None:
    """scripts/label_runs.py's exit decision on these summaries, with its messages (its rounds
    are this command's three); None: exit 0."""
    if any(s.get("error") for s in summaries):
        return "Some runs could not be labeled; see their errors above"
    if any(s.get("busy") for s in summaries):
        return "Some runs are being labeled by another process; nothing was sent for them"
    if any(s.get("unparseable") for s in summaries):
        return ("Some outputs did not parse. A rerun resumes each such run within the rounds it "
                "has left (three in all, across reruns) and sends nothing for a run whose rounds "
                "are used")  # fmt: skip
    return None


def label_study2(
    run_dirs: Iterable[Path],
    judge: str,
    load_prompts: Callable[[DictConfig], Mapping[str, str]],
    make_client: Callable[[Path], JudgeClient],
    *,
    sealed_dir: Path,
    token_check: Mapping[str, Any] | None = None,
    probe: Callable[[], Any] | None = None,
    command: list[str] | None = None,
) -> str | None:
    """Label ``run_dirs`` (the open runs, then the sealed ones), print every run's line in their
    order and the invocation's, write the sealed record; returns the exit message (None: exit 0).
    ``load_prompts`` and ``make_client`` are ``run_judging.label_runs``'; ``probe()`` (the token
    check, None: none) raises when the judge does not answer. After the sealed call the judge is
    re-probed with ``probe`` and one synthetic judgment (``judge_answers``)."""
    runs = [(Path(run_dir), sealed_reason(Path(run_dir))) for run_dir in run_dirs]
    opened = [run_dir for run_dir, why in runs if why is None]
    sealed = [(run_dir, why) for run_dir, why in runs if why is not None]
    found = label_runs(opened, judge, load_prompts, make_client, rounds=LABEL_ROUNDS)
    # The sealed call, unless the open one's judge failed (its runs are then not attempted)
    attempted = bool(sealed) and not _round_failed(found)
    summaries: list[dict[str, Any] | None] = [None] * len(sealed)
    output, trace, probed, judge_failed = io.StringIO(), None, None, False
    if attempted:
        dirs = [run_dir for run_dir, _ in sealed]
        try:
            with _captured(output):
                summaries = list(label_runs(dirs, judge, load_prompts, make_client,
                                            rounds=LABEL_ROUNDS))  # fmt: skip
        except BaseException:  # every exception of the sealed call: its details are sealed
            trace = traceback.format_exc()
        judge_failed = _round_failed(summaries)
        # The judge may have gone after its last round: re-probed, its tokenizer and its
        # generation (not after an interruption)
        if not judge_failed and trace is None:
            try:
                with _captured(output):
                    if probe is not None:
                        probe()
                    judge_answers(make_client)
            except BaseException:
                probed, judge_failed = traceback.format_exc(), True
    if judge_failed:  # one status for the call: who was hit depends on parsing
        statuses = [INCOMPLETE] * len(sealed)
    elif not attempted or trace is not None:
        statuses = [ERROR if trace is not None else NOT_ATTEMPTED_STATUS] * len(sealed)
    else:
        statuses = [sealed_status(summary, run_dir, judge)
                    for summary, (run_dir, _) in zip(summaries, sealed, strict=True)]  # fmt: skip
    path = None
    if sealed:
        path = write_sealed_record(sealed_dir, "labeling", judge, {
            "what": "The sealed record of a study-2 labeling (analysis.study2_label_cli): the "
                    "sealed runs' summaries, the captured output and any traceback. Opened in "
                    "the final analysis, never before.",
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": command,
            "code": git_state(),
            "judge": judge,
            "rounds": LABEL_ROUNDS,
            "attempted": attempted,
            "judge_failed": judge_failed,
            "runs": [{"run_dir": str(run_dir), "run_id": run_dir.name, "sealed": why,
                      "status": status, "summary": summary}
                     for (run_dir, why), status, summary in zip(sealed, statuses, summaries,
                                                                strict=True)],
            "output": output.getvalue(),
            "traceback": trace,
            "probe_traceback": probed,
        })  # fmt: skip
    shown_open, shown_sealed = iter(found), iter(statuses)
    for run_dir, why in runs:
        if why is None:
            print(json.dumps(next(shown_open)))
        else:
            print(json.dumps({"run_id": run_dir.name, "sealed": True,
                              "status": next(shown_sealed)}))  # fmt: skip
    print(json.dumps({"runs": len(runs), "token_check": token_check}))
    if path is not None:
        print(json.dumps({"sealed_record": str(path)}))
    failures = [_open_failure(found)]
    unfinished = [status for status in STATUSES if status != LABELED and status in statuses]
    if unfinished:
        failures.append(f"Sealed runs not labeled yet ({'; '.join(unfinished)}): rerun the "
                        f"labeling; their details are sealed in {path}")  # fmt: skip
    return "\n".join(f for f in failures if f) or None


def harmful_prompts(run_cfg: DictConfig) -> dict[str, str]:
    """A run's harmful prompt set, texts by id in the set's order (as scripts/label_runs.py)."""
    texts = load_prompt_set(run_cfg.eval.harmful.prompt_set, run_cfg.data)
    return {item["id"]: item["text"] for item in texts}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m zero_field_refusal.analysis.study2_label_cli",
                                     description=__doc__.splitlines()[0])  # fmt: skip
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--judge", required=True, help="a judge config in configs/judge/")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sealed-dir", type=Path,
                        help="where the sealed record goes (default $RUNS_DIR/sealed)")  # fmt: skip
    args = parser.parse_args(argv)
    load_env()
    try:
        sealed_dir = sealed_directory(args.sealed_dir)
    except ValueError as exc:
        parser.error(str(exc))
    cfg = load_config(f"judge/{args.judge}")
    client = JudgeClient.from_config(cfg)
    tokens, probe = None, None
    if client.backend == "wildguard":  # before any item: a doubled BOS would change every label

        def probe() -> dict[str, Any]:
            return check_wildguard_tokens(os.environ["JUDGE_API_BASE"], client.model,
                                          os.environ.get("JUDGE_API_KEY"))  # fmt: skip

        tokens = probe()
    failure = label_study2(
        args.run_dirs,
        args.judge,
        harmful_prompts,
        lambda path: JudgeClient.from_config(cfg, cache_path=path, max_workers=args.workers),
        sealed_dir=sealed_dir,
        token_check=tokens,
        probe=probe,
        command=sys.argv if argv is None else [parser.prog, *argv],
    )
    if failure:
        sys.exit(failure)


if __name__ == "__main__":
    main()
