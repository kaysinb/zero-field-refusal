"""Study 3's labeling command: the control list's one-time draw, the persistent labeling batch
with its 40 controls, one relabel and the judge's mark, and the mark's diagnosis.

    python -m zero_field_refusal.analysis.study3_label_cli controls-draw --calibration DIR \
        --seed S --archive URI [--runs-root DIR]
    python -m zero_field_refusal.analysis.study3_label_cli label --runs-root DIR \
        --step {validation,confirmation} --candidate N --controls-source DIR [--workers N] \
        [--sealed-dir DIR] [RUN_ID ...]          (no run ids: resume the unfinished batch)
    python -m zero_field_refusal.analysis.study3_label_cli diagnose --runs-root DIR --batch ID \
        --from FILE                              (a re-run completes a stranded diagnosis)
    python -m zero_field_refusal.analysis.study3_label_cli resolve-torn --runs-root DIR

docs/study3-preregistration.md: "Judge and audit" (labeling follows study 2's receipts, up to
three rounds per run; every labeling batch includes 40 fixed control items: immutable archived
prompt–response pairs from study 1's judge calibration, drawn once by a recorded seed, 20 that
WildGuard labeled harmful there and 20 not, each pinned by its id, the sha256 of its prompt and of
its response, and its expected label; an item whose content hash differs is refused; a changed
control label leads to relabeling the batch; a repeated change is a technical mark of the judge,
and labeling stops until it is diagnosed; every label is sealed from the first run), "Sample size
from validation" and "Seeds and order" (validation is labeled at the spread step).

**Where study 1's calibration set lives.** Locally, ``RUNS_DIR/judge_calibration/calibration/``
(study 1's Gate-1 command; docs/reports/gate-1.md): ``sample.jsonl``, the blind sample
``{item_id, prompt, response}`` with ids ``c`` and 12 hexadecimal digits, whose sha256 is fixed
in code (``analysis.gate1.CALIBRATION_SAMPLE_SHA256``); ``judge-wildguard.jsonl``, WildGuard's
labels, the latest record per id, as ``gate1._agreement`` reads them; and the reference labels.
It is archived in the create-only bucket; at labeling time the folder is restored from there,
read only, and given as ``--controls-source``. Every item is checked by id and both hashes.

**The draw** (``draw_controls``; ``controls-draw`` publishes it once at
``docs/study3-judge-controls.json``, ``study3_pins.CONTROLS``, refused when it exists; the
researcher commits it alone, before the validation fixation, which pins it):
1. the sample's sha256 must be the fixed one;
2. the latest label record per id, each carrying the identity of ``configs/judge/wildguard.yaml``;
3. the sample's items with a non-null label split into harmful (``harmful_compliance``) and not
   harmful, at least 20 of each;
4. ``random.Random(seed)``: 20 of each, each drawn from the sorted ids, listed by id;
5. each item with the sha256 of its prompt's and of its response's UTF-8 bytes, its study-1 label
   and its expected label; the record checked by ``study3_pins.control_list_problems``.
The record holds ids, hashes and labels, never a text; the command prints the counts, the path
and the sha256.

**The fetch** (``fetch_controls``) refuses, before any request: a source sample whose sha256 is
not the list's, and (``ControlMismatch``, naming the ids, never a text) an id that is missing or
doubled, or whose prompt or response hash differs. **The check**: ``label_controls`` labels all
40 on a cache file of its own, which must not exist (no control label is reused), in up to three
rounds while a label is null; ``changed_controls`` lists the ids whose two-class outcome
(``harmful_compliance`` or not) differs from the expected one, a null counting as changed, and
those whose label parsed (a parsed mismatch: a control mismatch).

**The persistent batch** (``label_batch``). One invocation advances one batch from the phase its
ledger lines show (``study3_labels``, "The ledger"): every decision is a line, written once,
before what it decides is done (write-ahead), and never recomputed; a crash anywhere leaves the
batch ``open``, resumed next time under the same id (an unfinished batch is resumed before any
other: the same command, or no run ids). In order (steps 1–5 send nothing and write no line):
1. *the lock*: the ledger's lock, without waiting;
2. *the ledger*: a ``LedgerError`` refuses (a torn tail names ``resolve-torn``), a stranded
   diagnosis refuses (naming the ``diagnose`` re-run), every ``superseded`` line whose move is
   not done is completed (the source's hashes checked first; any other state is ambiguous and
   refuses), and the candidate's unresolved mark refuses every batch of it;
3. *which batch*: the unfinished one (its step, candidate and runs), or a new one of the run ids;
4. *the point*, every invocation: validation runs while the spread point is open
   (``study3_labels.open_point``'s rule), each run id a plain name bound by the launch journal as
   an attempt of the step and candidate, not refused (``_bound``); confirmation runs are refused
   until the final point opens through the freeze gate;
5. *the pins*, every invocation: the candidate's in-force validation fixation
   (``study3_versions.read_validation_fixation``) without a problem, its control list's sha256
   the checkout's list now (and, on a resume, the batch's), the list clean, the fixation's judge
   this checkout's, and every control item fetched by id and hash;
6. *a new batch's runs*: a run whose open generation is accepted is left out ("accepted
   earlier"), one with a technical labeling problem too ("rounds spent": no diagnosis adds
   rounds); ``opened`` (``b%04d``), then a ``generation`` line per run without an open one, its
   ``cap`` the rounds the run has left;
7. *the labels of control generation k*, attempt n (1 plus the ``labeled`` lines of the batch at
   k). First the *unverified outputs*: a run whose open generation already holds a label file
   was written by an earlier attempt that died before its controls finished (or by an earlier
   batch that left it uncovered); its ``judge/`` is moved aside unopened to
   ``judge-superseded/<batch>/g<g>-unverified/`` (``superseded``, ``unverified``, its rounds used
   counted) and the run relabeled in its next generation. A run whose earlier attempt left only a
   rounds record keeps its generation while that record shows rounds left under its cap; once it
   shows the cap used (or does not read), the generation is superseded ``unverified`` the same
   way, its rounds used its cap: the run has no round left, so it is the judge's mark
   (``exhausted``, as in step 10) and is never sent again. Then the runs are labeled, grouped by
   their generation's cap, one ``run_judging.label_runs`` call per group at ``rounds=cap`` inside
   study 2's capture, stopping after a failed round; the judge is re-probed (the token check and
   one synthetic judgment); ``labeled`` records each run's allowlisted status and the receipt
   sha256 of the runs completed in this process. A failed round, a failed re-probe or an
   exception appends ``interrupted``: the batch is ``unfinished``, to be resumed with the same
   command;
8. *the controls of k*: all 40 labeled anew in this attempt, on its own
   ``labeling/batches/<batch>/controls-k<k>-a<n>.jsonl``; one ``controls`` line, which, when
   accepted, ``covers`` each run whose receipt this attempt's ``labeled`` line records and which
   is unchanged now;
9. *accepted*: the batch is done; a run not covered (an error, busy) goes to a later batch;
10. *the first change*: every run's open generation is superseded (``control_change``: its
    rounds used read from its rounds record, every file of its ``judge/`` hashed, the line
    written, then the folder renamed to ``judge-superseded/<batch>/g<g>``); a run left with no
    round is the judge's mark (``exhausted``); otherwise every run's next generation opens and
    the controls are labeled again at k = 2;
11. *the second change*: the open generations are superseded (``judge_mark``), then the mark
    (``second_change``); no generation opens;
12. *every return*: the sealed record (``study3-labeling``: the command, the code, the batch,
    each run's status and summary, the checks, the captured output and the tracebacks), one
    printed ``{"run_id", "status"}`` per run and ``{"batch", "outcome", "runs", "checks",
    "sealed_record"}``; exit 0 only when the batch is accepted and every run labeled.

**The rounds.** Each run has three labeling rounds in its lifetime. ``label_runs(rounds=r)`` caps
a run's rounds in all, counted by ``judge/<judge>.rounds.json``, which moves with ``judge/``; so
each generation gets ``cap = 3 −`` the rounds its run's closed generations used, journaled when it
opens and passed unchanged on every call, and a run with none left is not sent. Interrupted rounds
count, a relabel spends from them, and nothing releases or adds a round: no diagnosis does. A
bounded recovery of spent rounds would need a prospective registration amendment, with external
evidence for every affected round.

**A receipt counts only with its own controls**: those labeled in the same process attempt, under
the same judge session. Outputs of an attempt that died before its controls finished are moved
aside unverified and relabeled, within the run's rounds.

**The judge's mark** is a second control change in one batch, or a run exhausted after a rejected
generation (no round left: never 521 missing labels, a ``LabelingProblem``), a generation that
used its cap without a label file included (step 7): the registered lifetime rounds are spent,
with no release. Labeling of that candidate stops until the researcher's diagnosis
(``write_diagnosis``, under the ledger's lock), a closed record
``labeling/diagnosis-<batch>.json``: ``kind``, ``batch``, ``written_at``,
``cause`` (``infrastructure``: a documented external interruption of the GPU, the Pod or the
process; or ``judge``), ``diagnosis`` (one line, at most 200 characters) and ``evidence``
(``{path, sha256, root}``, rehashed on write, a sealed path refused; for ``infrastructure`` the
external evidence). It is written only for a mark and once. ``infrastructure`` is refused when a
control label of the batch changed to a parsed label other than the expected one (a control
mismatch) and for an ``exhausted`` mark. It resolves the mark and adds no round: the next batch
opens new generations within each run's remaining rounds (``after_diagnosis``), and a run with
none left stays a problem. ``judge`` leaves the mark unresolved: every later batch of the
candidate is refused and the runs stay labeling problems. The order is recoverable: the record is
published once, then its ``diagnosed`` line appended; a re-run that finds the record published
without its line appends the line when every field but ``written_at`` is the new file's, and
refuses otherwise. **A torn tail** of the ledger refuses ``label``, ``diagnose`` and every reader
until ``resolve-torn`` repairs it (it prints ``{offset, length, sha256, kind}`` of each line it
voids: the tail, and a whole line at the end that does not parse, left by a crash between an
earlier repair's ``\\n`` and its ``torn`` line).

**What opens the sealed folders here.** The named sealed openers are ``_label_pass`` (the runs'
labeling and their receipts), ``_rounds_used`` (a run's rounds record, which step 7's
``_unverified`` reads through it), ``_supersede`` (the control-change, mark and unverified moves
alike, through ``_hashed``, which hashes a run's ``judge/`` without parsing it and which the
completion of a move uses too), ``label_controls``
(the controls' cache, which ``_controls`` then hashes beside the receipts it re-hashes through
``study3_labels._receipt_sha256``) and ``write_diagnosis`` (the ledger's records); the ledger's
journal is read through ``study3_labels``. What is printed is allowlisted: ids, statuses, counts,
control ids, paths and hashes; never a text, a label or a harmful count.

**Rulings** where the registration leaves a detail open:
- *The two-class outcome.* The registration's judge has a two-class outcome; a control changes
  when its outcome differs from the expected one. The three-class label is kept as information.
- *A null control label* (an output that did not parse within the rounds) counts as changed, as
  a parsed mismatch does; only the parsed ones are a control mismatch in the sense that rules out
  an infrastructure cause.
- *The relabel* is of the whole batch, within each run's lifetime rounds; a change after a
  restart in the relabel is the second change, as the journal shows it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from ..config import load_config
from ..judge import (
    IDENTITY_FIELDS,
    LABELS,
    JudgeClient,
    check_wildguard_tokens,
    count_unparseable,
    judge_identity,
)
from ..paths import REPO_ROOT, load_env, runs_root
from ..run_judging import NOT_ATTEMPTED, label_runs, receipt_path, rounds_path
from ..tracking import git_state
from . import study3_versions
from .evidence import JUDGE_DIR, label_path
from .study2_label_cli import (
    BUSY,
    ERROR,
    INCOMPLETE,
    LABELED,
    NOT_ATTEMPTED_STATUS,
    _captured,
    _round_failed,
    harmful_prompts,
    judge_answers,
    sealed_directory,
    sealed_status,
    write_sealed_record,
)
from .study3_decisions import _publish
from .study3_labels import (
    JUDGE,
    JUDGE_CONFIG,
    ROUNDS,
    Generation,
    LedgerError,
    SealedPoint,
    _receipt_sha256,
    _spread,
    append_event,
    diagnosis_problems,
    ledger_dir,
    ledger_lock,
    read_ledger,
    record_path,
    resolve_torn,
    torn_tail,
)
from .study3_pins import (
    CONTROL_ITEMS,
    CONTROLS,
    CONTROLS_KIND,
    CONTROLS_WHAT,
    EXPECTED,
    HARMFUL,
    PER_CLASS,
    SOURCE_FILES,
    control_list_problems,
    read_control_list,
)
from .study3_records import EXPERIMENT, RESERVED, EndpointReader, plain_name, under_sealed

PROG = "python -m zero_field_refusal.analysis.study3_label_cli"
CONTROL_PREFIX = "control/"  # a control item's id as the judge client sees it
SUPERSEDED = "judge-superseded"  # a run's judge outputs moved aside, unopened
ACCEPTED_EARLIER = "accepted earlier"
ROUNDS_SPENT = "rounds spent"
SUPERSEDED_MARK = "superseded after the judge's mark"
# The allowlisted statuses of a run: study 2's, and the batch's own
STATUSES = (LABELED, INCOMPLETE, ERROR, BUSY, NOT_ATTEMPTED_STATUS, ACCEPTED_EARLIER, ROUNDS_SPENT,
            SUPERSEDED_MARK)  # fmt: skip
OUTCOMES = ("accepted", "unfinished", "mark", "refused")
# How a run's next generation is opened, by how its last one closed
NEXT_WHY = {"control_change": "control_change", "unverified": "unverified",
            "judge_mark": "after_diagnosis"}  # fmt: skip


class ControlMismatch(ValueError):
    """Control items of the list that the source does not hold once, or holds with another
    prompt or response: refused before any request, by id."""

    def __init__(self, ids: Sequence[str], why: str) -> None:
        self.ids = list(ids)
        super().__init__(f"control items {self.ids} {why}: refused before any request")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(path: Path, data: bytes) -> list[dict[str, Any]]:
    """Every line of a JSONL file as an object; ``ValueError`` naming the line's number."""
    rows = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}, line {number}, is not JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}, line {number}, is no object")
        rows.append(row)
    return rows


def _source_file(folder: Path, name: str, root: Path) -> tuple[Path, bytes]:
    path = folder / name
    if under_sealed(path, root):
        raise ValueError(f"{path} resolves under a sealed folder: never opened")
    try:
        return path, path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{path} cannot be read ({type(exc).__name__})") from exc


# ------------------------------------------------------------------ the draw


def draw_controls(calibration: Path, *, seed: int, archive: str, runs_root: Path,
                  code: Mapping[str, str], written_at: str | None = None) -> dict:  # fmt: skip
    """The control list drawn from study 1's judge-calibration folder ``calibration`` (under
    ``runs_root``, RUNS_DIR) by ``seed`` (module docstring, "The draw"); ``archive``: the bucket
    URI of the folder; ``code``: ``{"commit"}`` of the checkout. ``ValueError`` for every
    refusal. Nothing of a text is kept."""
    from . import gate1  # the fixed sample sha256 (its module loads the statistics stack)

    calibration, root = Path(calibration), Path(runs_root)
    try:
        folder = calibration.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"{calibration} is not under RUNS_DIR ({root}): the list names study 1's "
                         "files by their paths relative to it") from None  # fmt: skip
    sample_path, sample = _source_file(calibration, SOURCE_FILES["sample"], root)
    labels_path, labels = _source_file(calibration, SOURCE_FILES["labels"], root)
    if _sha256(sample) != gate1.CALIBRATION_SAMPLE_SHA256:
        raise ValueError(f"{sample_path} has sha256 {_sha256(sample)}, not study 1's fixed "
                         f"{gate1.CALIBRATION_SAMPLE_SHA256}")  # fmt: skip
    identity = judge_identity(load_config(JUDGE_CONFIG))
    latest: dict[str, Any] = {}
    for row in _jsonl(labels_path, labels):
        if {key: row.get(key) for key in IDENTITY_FIELDS} != identity:
            raise ValueError(f"{labels_path}: a label record of another judge identity than "
                             f"{JUDGE_CONFIG}'s")  # fmt: skip
        if row.get("label") is not None and row.get("label") not in LABELS:
            raise ValueError(f"{labels_path}: a label outside {list(LABELS)}")
        latest[row.get("item_id")] = row.get("label")
    items = _jsonl(sample_path, sample)
    ids = [item.get("item_id") for item in items]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"{sample_path}: an item id is missing or doubled")
    by_id = {item["item_id"]: item for item in items}
    harmful = sorted(i for i in ids if latest.get(i) == HARMFUL)
    not_harmful = sorted(i for i in ids if latest.get(i) is not None and latest[i] != HARMFUL)
    if len(harmful) < PER_CLASS or len(not_harmful) < PER_CLASS:
        raise ValueError(f"{len(harmful)} harmful and {len(not_harmful)} not harmful labeled "
                         f"items: fewer than {PER_CLASS} of each to draw from")  # fmt: skip
    rng = random.Random(seed)
    chosen = sorted(rng.sample(harmful, PER_CLASS) + rng.sample(not_harmful, PER_CLASS))
    record = {
        "kind": CONTROLS_KIND, "what": CONTROLS_WHAT,
        "written_at": written_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "code": {"commit": code.get("commit")}, "seed": seed,
        "source": {
            "sample": {"path": (folder / SOURCE_FILES["sample"]).as_posix(),
                       "sha256": _sha256(sample)},
            "labels": {"path": (folder / SOURCE_FILES["labels"]).as_posix(),
                       "sha256": _sha256(labels)},
            "archive": archive},
        "judge": identity,
        "items": [{"id": i,
                   "prompt_sha256": _sha256(str(by_id[i]["prompt"]).encode("utf-8")),
                   "response_sha256": _sha256(str(by_id[i]["response"]).encode("utf-8")),
                   "expected": EXPECTED[0] if latest[i] == HARMFUL else EXPECTED[1],
                   "label": latest[i]} for i in chosen],
    }  # fmt: skip
    problems = control_list_problems(record)
    if problems:
        raise ValueError(f"the drawn control list is not one: {problems[0]}")
    return record


# ------------------------------------------------------------------ the fetch and the check


def fetch_controls(record: Mapping, source: Path) -> list[dict[str, str]]:
    """The control items of ``record`` (a control list) from the restored calibration folder
    ``source``, in the list's order: ``{"item_id": "control/<id>", "prompt", "response"}``
    (module docstring, "The fetch")."""
    source = Path(source)
    path, data = _source_file(source, SOURCE_FILES["sample"], source)
    pinned = record["source"]["sample"]["sha256"]
    if _sha256(data) != pinned:
        raise ValueError(f"{path} has sha256 {_sha256(data)}, not the control list's source "
                         f"sample {pinned}: not the calibration set the list was drawn "
                         "from")  # fmt: skip
    rows = _jsonl(path, data)
    count = Counter(row.get("item_id") for row in rows)
    wanted = [item["id"] for item in record["items"]]
    once = [i for i in wanted if count[i] != 1]
    if once:
        raise ControlMismatch(once, "are missing from the source or held more than once")
    by_id = {row["item_id"]: row for row in rows if row.get("item_id") in set(wanted)}
    differ = [item["id"] for item in record["items"]
              if _sha256(str(by_id[item["id"]].get("prompt")).encode("utf-8"))
              != item["prompt_sha256"]
              or _sha256(str(by_id[item["id"]].get("response")).encode("utf-8"))
              != item["response_sha256"]]  # fmt: skip
    if differ:
        raise ControlMismatch(differ, "have another prompt or response hash than the list pins")
    return [{"item_id": f"{CONTROL_PREFIX}{i}", "prompt": str(by_id[i]["prompt"]),
             "response": str(by_id[i]["response"])} for i in wanted]  # fmt: skip


def label_controls(items: Sequence[Mapping[str, str]], make_client: Callable[[Path], JudgeClient],
                   cache: Path) -> dict[str, str | None]:  # fmt: skip
    """The judge's label of each control item, by control id (null: unparsed), from a client of
    ``make_client`` writing to ``cache``, created new here (an existing file refuses: no control
    label is reused), in up to ``ROUNDS`` rounds while a label is null. A named sealed opener:
    the ledger's cache file."""
    cache = Path(cache)
    cache.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.close(os.open(cache, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError:
        raise ValueError(f"{cache} exists: the controls are labeled anew on a cache of their own, "
                         "and no earlier control label is reused") from None  # fmt: skip
    client = make_client(cache)
    records: list[dict[str, Any]] = []
    for _ in range(ROUNDS):
        records = client.label_items(list(items))
        if not count_unparseable(records):
            break
    return {str(r["item_id"]).removeprefix(CONTROL_PREFIX): r["label"] for r in records}


def changed_controls(
    record: Mapping, labels: Mapping[str, str | None]
) -> tuple[list[str], list[str]]:
    """``(changed, parsed_changed)``: the ids whose two-class outcome differs from the expected
    one (a null counts as changed), and those of them whose label parsed; both sorted."""
    changed, parsed = [], []
    for item in record["items"]:
        label = labels.get(item["id"])
        outcome = EXPECTED[0] if label == HARMFUL else EXPECTED[1]
        if label is None or outcome != item["expected"]:
            changed.append(item["id"])
            if label is not None:
                parsed.append(item["id"])
    return sorted(changed), sorted(parsed)


# ------------------------------------------------------------------ the persistent batch


class BatchRefused(ValueError):
    """The batch is refused before any label: why is named, never a text."""


@dataclass(frozen=True)
class BatchResult:
    """An invocation's result: the batch, its ``outcome`` (``OUTCOMES``), each run's status, the
    checks (per k: ``accepted`` and the ``changed`` control ids), the sealed record's path and
    the exit message (None: exit 0). A refusal raises ``BatchRefused`` instead."""

    batch: str | None
    outcome: str
    statuses: dict[str, str]
    checks: list[dict]
    sealed_record: Path | None
    failure: str | None


def label_batch(
    runs_root: Path,
    step: str,
    candidate: int,
    run_ids: Sequence[str] | None,
    *,
    controls_source: Path,
    make_client: Callable[[Path], JudgeClient],
    load_prompts: Callable[[DictConfig], Mapping[str, str]],
    sealed_dir: Path,
    repo: Path = REPO_ROOT,
    probe: Callable[[], Any] | None = None,
    command: list[str] | None = None,
) -> BatchResult:
    """Advance one persistent labeling batch (module docstring, "The persistent batch"):
    ``run_ids`` names a new batch's runs, None resumes the unfinished one. ``make_client`` and
    ``load_prompts`` are ``run_judging.label_runs``'; ``probe()`` (the token check, None: none)
    raises when the judge does not answer. ``BatchRefused`` for every refusal."""
    if step not in ("validation", "confirmation") or candidate not in (1, 2):
        raise BatchRefused(f"the step is validation or confirmation and the candidate 1 or 2, "
                           f"not {step!r} and {candidate!r}")  # fmt: skip
    with ledger_lock(runs_root) as held:
        if not held:
            raise BatchRefused("another labeling holds the labeling ledger's lock: nothing is "
                               "sent")  # fmt: skip
        batch = _Batch(Path(runs_root), step, candidate, run_ids, Path(controls_source),
                       make_client, load_prompts, Path(sealed_dir), Path(repo), probe,
                       command)  # fmt: skip
        return batch.run()


class _Batch:
    """One invocation of ``label_batch``, under the ledger's lock."""

    def __init__(self, root: Path, step: str, candidate: int, run_ids: Sequence[str] | None,
                 source: Path, make_client: Callable[[Path], JudgeClient],
                 load_prompts: Callable[[DictConfig], Mapping[str, str]], sealed_dir: Path,
                 repo: Path, probe: Callable[[], Any] | None,
                 command: list[str] | None) -> None:  # fmt: skip
        self.root, self.step, self.candidate, self.source = root, step, candidate, source
        self.run_ids = None if run_ids is None else list(run_ids)
        self.make_client, self.load_prompts, self.probe = make_client, load_prompts, probe
        self.sealed_dir, self.repo, self.command = sealed_dir, repo, command
        self.batch: str | None = None
        self.runs: list[str] = []  # the batch's runs
        self.extra: list[str] = []  # runs named but left out, with their statuses
        self.statuses: dict[str, str] = {}
        self.summaries: dict[str, Any] = {}
        self.traces: dict[str, str] = {}
        self.output = io.StringIO()

    def run_dir(self, run: str) -> Path:
        return self.root / EXPERIMENT / run

    # -- steps 2 to 6: nothing is sent

    def run(self) -> BatchResult:
        try:
            ledger = read_ledger(self.root)
        except LedgerError as exc:
            if torn_tail(self.root) is not None:
                raise BatchRefused(f"the labeling ledger's last line is torn: run `{PROG} "
                                   f"resolve-torn --runs-root {self.root}` "
                                   "first") from None  # fmt: skip
            raise BatchRefused(f"the labeling ledger cannot be read: {exc}") from None
        if ledger.stranded:
            batch = ledger.stranded[0]
            raise BatchRefused(f"labeling/diagnosis-{batch}.json is published without its "
                               f"diagnosed line: complete it first with `{PROG} diagnose "
                               f"--runs-root {self.root} --batch {batch} --from FILE`")  # fmt: skip
        self._complete_moves(ledger)
        stopped = ledger.stopped(self.candidate)
        if stopped is not None:
            if ledger.batches[stopped]["diagnosed"] is not None:
                raise BatchRefused(f"the judge's technical mark of batch {stopped} was diagnosed "
                                   "`judge`: labeling stays stopped")  # fmt: skip
            raise BatchRefused(f"the judge's technical mark of batch {stopped} waits for "
                               f"labeling/diagnosis-{stopped}.json")  # fmt: skip
        batch, named = self._which(ledger)
        self._gate(named)
        self._pins(ledger, batch)
        if batch is None:
            batch = self._open(ledger, named)
        self.batch = batch
        outcome, failure = self._advance()
        return self._finish(outcome, failure)

    def _which(self, ledger) -> tuple[str | None, list[str]]:
        """Step 3: the unfinished batch and its runs, or None and the runs named; every run id
        the invocation names."""
        given = self.run_ids
        if given is not None and (not given or len(set(given)) != len(given)):
            raise BatchRefused("name each run to label once")
        unfinished = ledger.unfinished()
        if unfinished is None:
            if given is None:
                raise BatchRefused("no batch is unfinished: name the runs to label")
            return None, list(given)
        info = ledger.batches[unfinished]
        runs = list(info["runs"])
        extra = [] if given is None else [r for r in given if r not in runs]
        same = (info["step"] == self.step and info["candidate"] == self.candidate
                and (given is None or (set(runs) <= set(given)
                                       and all(_left_out(ledger, r) for r in extra))))  # fmt: skip
        if not same:
            raise BatchRefused(f"batch {unfinished} is unfinished (candidate {info['candidate']}'s "
                               f"{info['step']} runs {runs}): resume it first, with the same "
                               "command or no run ids")  # fmt: skip
        self.runs, self.extra = runs, extra
        for run in extra:
            self.statuses[run] = _left_out(ledger, run)
        return unfinished, [*runs, *extra]

    def _gate(self, named: Sequence[str]) -> None:
        """Step 4: the point, and each run bound as an attempt of the step and candidate."""
        bad = [r for r in named if not plain_name(r) or r in RESERVED]
        if bad:
            raise BatchRefused(f"{bad[0]!r} is not a plain run id")
        if self.step == "confirmation":
            raise BatchRefused("the final point opens through the freeze gate: confirmation runs "
                               "are labeled once it is open")  # fmt: skip
        try:
            state, _ = _spread(self.candidate, self.root, self.repo)
        except SealedPoint as exc:
            raise BatchRefused(f"validation runs are labeled while the spread point is open: "
                               f"{exc}") from None  # fmt: skip
        _bound(self.root, state, self.step, self.candidate, named)

    def _pins(self, ledger, batch: str | None) -> None:
        """Step 5: the fixation, the control list and its items (fetched by id and hash)."""
        fixation, problems = study3_versions.read_validation_fixation(
            self.candidate, runs_root=self.root, repo=self.repo
        )
        if fixation is None:
            why = f"invalid: {problems[0]}" if problems else "missing"
            raise BatchRefused(f"candidate {self.candidate}'s validation fixation is {why}: it "
                               "pins the control list labeling runs with")  # fmt: skip
        record, problems = read_control_list(self.repo)
        if record is None:
            raise BatchRefused(f"the control list: {problems[0]}")
        data = (self.repo / CONTROLS).read_bytes()
        record, problems = read_control_list(self.repo, data=data)
        if record is None:
            raise BatchRefused(f"the control list: {problems[0]}")
        digest = _sha256(data)
        if fixation["judge_controls"]["sha256"] != digest:
            raise BatchRefused(f"{CONTROLS} has sha256 {digest}, not the one candidate "
                               f"{self.candidate}'s validation fixation pins")  # fmt: skip
        if batch is not None and ledger.batches[batch]["controls"]["sha256"] != digest:
            raise BatchRefused(f"{CONTROLS} is not the list batch {batch} opened with")
        judge = study3_versions.instruments()[1]
        if fixation.get("judge") != judge:
            raise BatchRefused("the validation fixation's judge is not this checkout's judge")
        named = study3_versions.fixation_name("validation", self.candidate)
        try:
            fixed = _sha256((self.repo / named).read_bytes())
        except OSError:
            raise BatchRefused(f"{named} cannot be read") from None
        try:
            self.items = fetch_controls(record, self.source)
        except ControlMismatch as exc:
            raise BatchRefused(str(exc)) from None
        except (OSError, ValueError) as exc:
            raise BatchRefused(f"the controls' source: {exc}") from None
        self.record = record
        self.pinned = {"controls": {"path": CONTROLS, "sha256": digest},
                       "fixation": {"path": named, "sha256": fixed}, "judge": judge}  # fmt: skip

    def _open(self, ledger, named: Sequence[str]) -> str:
        """Step 6: a new batch of the named runs that are left to label, and their generations."""
        for run in named:
            status = _left_out(ledger, run)
            if status is None:
                self.runs.append(run)
            else:
                self.extra.append(run)
                self.statuses[run] = status
        if not self.runs:
            raise BatchRefused(f"no run of {list(named)} is left to label: "
                               f"{sorted(set(self.statuses.values()))}")  # fmt: skip
        batch = f"b{sum(1 for info in ledger.batches.values() if info['opened']) + 1:04d}"
        code = git_state(self.repo)
        append_event(self.root, {"event": "opened", "batch": batch, "step": self.step,
                                 "candidate": self.candidate, "runs": self.runs, **self.pinned,
                                 "code": {"commit": code["commit"],
                                          "dirty": code["dirty"]}})  # fmt: skip
        self.batch = batch
        for run in self.runs:
            if read_ledger(self.root).current(run) is None:
                self._open_generation(run)
        return batch

    # -- steps 7 to 11

    def _advance(self) -> tuple[str, str | None]:
        while True:
            info = read_ledger(self.root).batches[self.batch]
            if info["changes"] >= 2:
                return self._second_change()
            if info["changes"] == 1:  # the first change's moves, each run once
                done = {s["run"] for s in info["superseded"]
                        if s["k"] == 1 and s["reason"] == "control_change"}  # fmt: skip
                ledger = read_ledger(self.root)
                for run in self.runs:
                    if run not in done and ledger.current(run) is not None:
                        self._supersede(run, 1, "control_change")
            marked = self._settle()
            if marked is not None:
                return marked
            k = info["k"]
            ledger = read_ledger(self.root)
            for run in self.runs:  # unverified outputs of an earlier attempt, or its cap spent
                if self._unverified(run, ledger.current(run)):
                    self._supersede(run, k, "unverified")
            marked = self._settle()
            if marked is not None:
                return marked
            info = read_ledger(self.root).batches[self.batch]
            attempt = 1 + sum(1 for (at, _) in info["labeled"] if at == k)
            receipts = self._label_pass(k, attempt)
            if receipts is None:
                return "unfinished", self._unfinished(k)
            done = self._controls(k, attempt, receipts)
            if done is not None:
                return done

    def _unverified(self, run: str, generation: Generation | None) -> bool:
        """Whether the run's open generation is superseded ``unverified`` (step 7): it holds a
        label file of an earlier attempt, or its rounds record shows its cap used with no label
        file (no round of it is left: ``_settle`` then makes the run the judge's mark)."""
        if generation is None:
            return False
        run_dir = self.run_dir(run)
        if label_path(run_dir, JUDGE).exists():
            return True
        return _rounds_used(run_dir, generation.cap) >= generation.cap

    def _settle(self) -> tuple[str, str] | None:
        """Every run of the batch without an open generation gets its next one, unless a run is
        left with no round: then the judge's mark (``exhausted``)."""
        ledger = read_ledger(self.root)
        waiting = [run for run in self.runs if ledger.current(run) is None]
        exhausted = [run for run in waiting if ledger.remaining(run) == 0]
        if exhausted:
            return self._mark("exhausted", exhausted)
        for run in waiting:
            self._open_generation(run)
        return None

    def _open_generation(self, run: str) -> None:
        ledger = read_ledger(self.root)
        found = ledger.generations.get(run, ())
        why = NEXT_WHY[found[-1].closed] if found else None
        append_event(self.root, {"event": "generation", "batch": self.batch, "run": run,
                                 "generation": len(found) + 1, "cap": ledger.remaining(run),
                                 "why": why})  # fmt: skip

    def _second_change(self) -> tuple[str, str]:
        for run in self.runs:
            if read_ledger(self.root).current(run) is not None:
                self._supersede(run, 2, "judge_mark")
        return self._mark("second_change", [])

    def _mark(self, why: str, exhausted: Sequence[str]) -> tuple[str, str]:
        ledger = read_ledger(self.root)
        controls = ledger.batches[self.batch]["checks"]
        changed = {str(k): controls[k]["changed"] if k in controls else None for k in (1, 2)}
        append_event(self.root, {"event": "mark", "batch": self.batch,
                                 "candidate": self.candidate, "why": why, "changed": changed,
                                 "runs": list(exhausted)})  # fmt: skip
        for run in self.runs:
            if run in exhausted:
                self.statuses[run] = ROUNDS_SPENT
            elif ledger.current(run) is None:
                self.statuses[run] = SUPERSEDED_MARK
            else:
                self.statuses.setdefault(run, NOT_ATTEMPTED_STATUS)
        stop = (f"labeling stops until labeling/diagnosis-{self.batch}.json is written "
                f"(`{PROG} diagnose`)")  # fmt: skip
        if why == "exhausted":
            return "mark", (f"run(s) {', '.join(exhausted)} exhausted after a rejected "
                            f"generation: the judge's technical mark; {stop}")  # fmt: skip
        return "mark", (f"a second control change in batch {self.batch}: the judge's technical "
                        f"mark; {stop}")  # fmt: skip

    def _unfinished(self, k: int) -> str:
        return (f"the judge failed during batch {self.batch} (controls generation {k}): its runs "
                "are not labeled yet; resume with the same command")  # fmt: skip

    def _supersede(self, run: str, k: int, reason: str) -> None:
        """A run's open generation superseded: its rounds used and the hash of every file of its
        ``judge/`` (unparsed) journaled, then the folder moved aside. A named sealed opener."""
        generation = read_ledger(self.root).current(run)
        assert generation is not None
        run_dir = self.run_dir(run)
        target = f"{SUPERSEDED}/{self.batch}/g{generation.number}"
        target += "-unverified" if reason == "unverified" else ""
        append_event(self.root, {"event": "superseded", "batch": self.batch, "k": k, "run": run,
                                 "generation": generation.number, "reason": reason,
                                 "rounds_used": _rounds_used(run_dir, generation.cap),
                                 "target": target, "moved": _hashed(run_dir)})  # fmt: skip
        _move(run_dir, target)

    def _complete_moves(self, ledger) -> None:
        """Every ``superseded`` line whose move is not done, completed: its source's files
        hashed first (unparsed), and the folder moved; any other state refuses."""
        for info in ledger.batches.values():
            for line in info["superseded"]:
                run_dir = self.run_dir(line["run"])
                if (run_dir / line["target"]).exists():
                    continue
                source = run_dir / JUDGE_DIR
                held = source.exists() if line["moved"] else not source.exists()
                if held and _hashed(run_dir) == line["moved"]:  # as the line names it
                    _move(run_dir, line["target"])
                    continue
                raise BatchRefused(f"{line['run']}: its generation {line['generation']} was "
                                   f"superseded in batch {line['batch']} but neither moved nor "
                                   "held as the line names it: ambiguous, decide it by "
                                   "hand")  # fmt: skip

    def _label_pass(self, k: int, attempt: int) -> dict[str, tuple[str, str]] | None:
        """Step 7's labeling: every run with an open generation, grouped by cap, then the
        re-probe and the ``labeled`` line; the receipts (and label files) of the runs completed
        in this process, or None when the judge failed (``interrupted``). A named sealed
        opener."""
        ledger = read_ledger(self.root)
        groups: dict[int, list[str]] = {}
        for run in self.runs:
            generation = ledger.current(run)
            if generation is not None:
                groups.setdefault(generation.cap, []).append(run)
        labeled = [run for group in groups.values() for run in group]
        summaries: dict[str, Any] = {}
        why = None
        try:
            with _captured(self.output):
                for cap, group in groups.items():
                    found = label_runs([self.run_dir(r) for r in group], JUDGE,
                                       self.load_prompts, self.make_client, rounds=cap)  # fmt: skip
                    summaries.update(zip(group, found, strict=True))
                    if _round_failed(found):
                        why = "a labeling round failed: the judge did not answer"
                        break
        except BaseException:  # its details are sealed
            self.traces[f"labeling k{k} a{attempt}"] = traceback.format_exc()
            why = "an exception while the runs were labeled"
        if why is None:
            try:
                with _captured(self.output):
                    if self.probe is not None:
                        self.probe()
                    judge_answers(self.make_client)
            except BaseException:
                self.traces[f"probe k{k} a{attempt}"] = traceback.format_exc()
                why = "the judge failed its re-probe"
        missing = {"skipped": NOT_ATTEMPTED}
        statuses = {run: INCOMPLETE if why else sealed_status(summaries.get(run, missing),
                                                               self.run_dir(run), JUDGE)
                    for run in labeled}  # fmt: skip
        receipts: dict[str, tuple[str, str]] = {}
        for run in labeled if why is None else ():
            summary = summaries.get(run) or {}
            path = receipt_path(self.run_dir(run), JUDGE)
            if (statuses[run] == LABELED and not summary.get("kept")
                    and not summary.get("restored") and path.is_file()):  # fmt: skip
                receipts[run] = (
                    _sha256(path.read_bytes()),
                    _sha256(label_path(self.run_dir(run), JUDGE).read_bytes()),
                )
        self.statuses.update(statuses)
        self.summaries.update({run: summaries.get(run) for run in labeled})
        append_event(self.root, {"event": "labeled", "batch": self.batch, "k": k,
                                 "attempt": attempt, "runs": [
                                     {"run": run, "status": statuses[run],
                                      "receipt_sha256": receipts.get(run, (None,))[0]}
                                     for run in labeled]})  # fmt: skip
        if why is not None:
            append_event(self.root, {"event": "interrupted", "batch": self.batch, "k": k,
                                     "why": why})  # fmt: skip
            return None
        return receipts

    def _controls(
        self, k: int, attempt: int, receipts: Mapping[str, tuple[str, str]]
    ) -> tuple[str, str | None] | None:
        """Step 8: the 40 controls of k labeled anew on this attempt's own cache, and the one
        ``controls`` line: accepted (the outcome), or changed (None: the batch goes on)."""
        cache = ledger_dir(self.root) / "batches" / self.batch / f"controls-k{k}-a{attempt}.jsonl"
        try:
            with _captured(self.output):
                labels = label_controls(self.items, self.make_client, cache)
        except BaseException:
            self.traces[f"controls k{k} a{attempt}"] = traceback.format_exc()
            append_event(self.root, {"event": "interrupted", "batch": self.batch, "k": k,
                                     "why": "the judge failed while the controls were "
                                            "labeled"})  # fmt: skip
            return "unfinished", self._unfinished(k)
        changed, parsed = changed_controls(self.record, labels)
        line: dict[str, Any] = {
            "event": "controls", "batch": self.batch, "k": k, "attempt": attempt,
            "accepted": not changed, "changed": changed, "parsed_changed": parsed,
            "cache": {"path": cache.relative_to(self.root).as_posix(),
                      "sha256": _sha256(cache.read_bytes())}}  # fmt: skip
        if changed:
            append_event(self.root, line)
            return None
        ledger = read_ledger(self.root)
        covers = []
        for run, (receipt, labels_sha256) in receipts.items():
            generation = ledger.current(run)
            if generation is not None and _receipt_sha256(self.run_dir(run)) == receipt:
                covers.append({"run": run, "generation": generation.number,
                               "receipt_sha256": receipt,
                               "labels_sha256": labels_sha256})  # fmt: skip
        append_event(self.root, {**line, "covers": covers})
        covered = {cover["run"] for cover in covers}
        for run in self.runs:
            if run not in covered and self.statuses.get(run) == LABELED:
                self.statuses[run] = INCOMPLETE
        named = [*self.runs, *self.extra]
        left = sorted({self.statuses.get(run, NOT_ATTEMPTED_STATUS) for run in named}
                      - {LABELED, ACCEPTED_EARLIER})  # fmt: skip
        if left:
            return "accepted", (f"batch {self.batch} accepted with runs not labeled "
                                f"({'; '.join(left)}): label them in a later batch")  # fmt: skip
        return "accepted", None

    # -- step 12

    def _finish(self, outcome: str, failure: str | None) -> BatchResult:
        info = read_ledger(self.root).batches[self.batch]
        checks = [{"k": k, "accepted": line["accepted"], "changed": line["changed"]}
                  for k, line in sorted(info["checks"].items())]  # fmt: skip
        runs = [*self.runs, *self.extra]
        statuses = {run: self.statuses.get(run, NOT_ATTEMPTED_STATUS) for run in runs}
        path = write_sealed_record(self.sealed_dir, "study3-labeling", JUDGE, {
            "what": "The sealed record of a study-3 labeling batch (analysis.study3_label_cli): "
                    "each run's status and summary, the checks, the captured output and the "
                    "tracebacks. Opened in the final analysis, never before.",
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": self.command, "code": git_state(self.repo), "batch": self.batch,
            "step": self.step, "candidate": self.candidate, "outcome": outcome,
            "runs": {run: {"status": statuses[run], "summary": self.summaries.get(run)}
                     for run in runs},
            "checks": checks, "output": self.output.getvalue(), "tracebacks": self.traces,
        })  # fmt: skip
        for run in runs:
            print(json.dumps({"run_id": run, "status": statuses[run]}))
        print(json.dumps({"batch": self.batch, "outcome": outcome, "runs": len(runs),
                          "checks": checks, "sealed_record": str(path)}))  # fmt: skip
        return BatchResult(self.batch, outcome, statuses, checks, path, failure)


def _left_out(ledger, run: str) -> str | None:
    """Why a run is left out of a new batch: its open generation accepted, or a technical
    labeling problem (no diagnosis adds rounds); None: it is labeled."""
    current = ledger.current(run)
    if current is not None and (run, current.number) in ledger.covered:
        return ACCEPTED_EARLIER
    return ROUNDS_SPENT if ledger.problem(run) else None


def _bound(runs_root: Path, state: Mapping[str, Any], step: str, candidate: int,
           run_ids: Sequence[str], theta0: Path | None = None) -> None:  # fmt: skip
    """Raise ``BatchRefused`` unless every run id names an attempt the launch journal binds to
    ``step`` and ``candidate``, not refused (``study3_records.EndpointReader`` under the state's
    prescribed cells; θ₀ at its default place)."""
    from .study3_progress import THETA0_RUN

    root = Path(runs_root)
    try:
        reader = EndpointReader(root, theta0=root / THETA0_RUN if theta0 is None else theta0)
        endpoints, _ = reader.read(state.get("prescribed") or {})
    except (OSError, ValueError) as exc:
        raise BatchRefused(f"the run records cannot be read ({type(exc).__name__})") from None
    found = {e.run_id: e for e in endpoints}
    for run in run_ids:
        e = found.get(run)
        if e is None or e.state == "refused" or e.cell.step != step or (
                e.cell.candidate != candidate):  # fmt: skip
            raise BatchRefused(f"{run} is not an attempt of candidate {candidate}'s {step} that "
                               "the launch journal binds (or it is refused): nothing is "
                               "labeled")  # fmt: skip


def _rounds_used(run_dir: Path, cap: int) -> int:
    """The rounds a run's open generation used, from its rounds record: 0 without one and without
    a label file, the cap without one beside a label file (the rounds spent on it are unknown) or
    when it does not read, never more than the cap. A named sealed opener."""
    path = rounds_path(run_dir, JUDGE)
    if not path.exists():
        return cap if label_path(run_dir, JUDGE).exists() else 0
    try:
        used = json.loads(path.read_text())["used"]
    except (OSError, ValueError, KeyError, TypeError):
        return cap
    return min(used, cap) if type(used) is int and used >= 0 else cap


def _hashed(run_dir: Path) -> list[dict[str, str]]:
    """Every file of the run's ``judge/``, by its path in the run's folder, with its sha256 (not
    parsed); in path order."""
    source = run_dir / JUDGE_DIR
    if not source.is_dir():
        return []
    files = sorted(p for p in source.rglob("*") if p.is_file())
    return [{"path": p.relative_to(run_dir).as_posix(), "sha256": _sha256(p.read_bytes())}
            for p in files]  # fmt: skip


def _rename(source: Path, target: Path) -> None:
    os.rename(source, target)


def _move(run_dir: Path, target: str) -> None:
    """The run's ``judge/`` renamed to ``target`` (an empty folder made when there is none), both
    parents synced."""
    destination = run_dir / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = run_dir / JUDGE_DIR
    if source.exists():
        _rename(source, destination)
    else:
        destination.mkdir()
    for folder in (run_dir, destination.parent, destination.parent.parent):
        handle = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


# ------------------------------------------------------------------ the diagnosis


def write_diagnosis(runs_root: Path, batch: str, record: Mapping[str, Any]) -> Path:
    """Publish the diagnosis of batch ``batch``'s mark, then its ``diagnosed`` line, under the
    ledger's lock (module docstring, "The judge's mark"); a re-run completes a stranded one.
    ``ValueError`` for every refusal; returns the record's path."""
    root = Path(runs_root)
    with ledger_lock(root) as held:
        if not held:
            raise ValueError("another process holds the labeling ledger's lock")
        ledger = read_ledger(root)
        info = ledger.batches.get(batch)
        if info is None or info["mark"] is None:
            raise ValueError(f"batch {batch} has no judge's mark: a diagnosis is written for a "
                             "mark only")  # fmt: skip
        if info["diagnosed"] is not None:
            raise ValueError(f"batch {batch} is diagnosed already: a diagnosis is written once")
        others = [b for b in ledger.stranded if b != batch]
        if others:
            raise ValueError(f"complete batch {others[0]}'s stranded diagnosis first")
        problems = diagnosis_problems(record, batch)
        if problems:
            raise ValueError(f"the diagnosis record: {problems[0]}")
        for entry in record["evidence"]:
            base = root if entry["root"] == "runs" else REPO_ROOT
            path = base / entry["path"]
            if under_sealed(path, base):
                raise ValueError(f"evidence {entry['path']} resolves under a sealed folder: never "
                                 "opened")  # fmt: skip
            if not path.is_file() or _sha256(path.read_bytes()) != entry["sha256"]:
                raise ValueError(f"evidence {entry['path']} is not the file its sha256 names")
        if record["cause"] == "infrastructure":
            parsed = sorted({i for line in info["checks"].values()
                             for i in line["parsed_changed"]})  # fmt: skip
            if parsed:
                raise ValueError(f"batch {batch}: control labels {parsed} changed to parsed "
                                 "labels other than the expected ones, a control mismatch: "
                                 "never infrastructure")  # fmt: skip
            if info["mark"]["why"] == "exhausted":
                raise ValueError(f"batch {batch}'s mark is exhausted: an exhausted run stays the "
                                 "judge's labeling problem, never infrastructure")  # fmt: skip
        path = ledger_dir(root) / f"diagnosis-{batch}.json"
        if batch in ledger.stranded:
            data = ledger.published[batch][1]
            published = json.loads(data)
            if ({k: v for k, v in published.items() if k != "written_at"}
                    != {k: v for k, v in record.items() if k != "written_at"}):  # fmt: skip
                raise ValueError(f"{path} is published with other fields than the given record: "
                                 "a published record is never rewritten")  # fmt: skip
        else:
            data = (json.dumps(dict(record), indent=1) + "\n").encode()
            _publish(path, data)
        reference = {"path": record_path(batch), "sha256": _sha256(data)}
        append_event(root, {"event": "diagnosed", "batch": batch, "cause": record["cause"],
                            "record": reference})  # fmt: skip
        return path


# ------------------------------------------------------------------ the commands


def _draw_command(args: argparse.Namespace) -> None:
    if args.runs_root is None:
        load_env()
    root = Path(args.runs_root) if args.runs_root is not None else runs_root()
    commit = git_state(REPO_ROOT)["commit"]
    if not isinstance(commit, str):
        sys.exit(f"{PROG} controls-draw: {REPO_ROOT} is not a Git checkout with a commit")
    try:
        record = draw_controls(args.calibration, seed=args.seed, archive=args.archive,
                               runs_root=root, code={"commit": commit})  # fmt: skip
        data = (json.dumps(record, indent=2) + "\n").encode()
        _publish(REPO_ROOT / CONTROLS, data)
    except ValueError as exc:
        sys.exit(f"{PROG} controls-draw: {exc}")
    harmful = sum(item["expected"] == EXPECTED[0] for item in record["items"])
    print(json.dumps({"controls": CONTROL_ITEMS, "harmful": harmful,
                      "not_harmful": CONTROL_ITEMS - harmful, "path": CONTROLS,
                      "sha256": _sha256(data)}))  # fmt: skip


def _label_command(args: argparse.Namespace, prog: str, argv: list[str] | None) -> None:
    load_env()
    try:
        sealed_dir = sealed_directory(args.sealed_dir)
    except ValueError as exc:
        sys.exit(f"{PROG} label: {exc}")
    cfg = load_config(JUDGE_CONFIG)
    client = JudgeClient.from_config(cfg)
    probe = None
    if client.backend == "wildguard":  # before any item: a doubled BOS would change every label

        def probe() -> dict[str, Any]:
            return check_wildguard_tokens(os.environ["JUDGE_API_BASE"], client.model,
                                          os.environ.get("JUDGE_API_KEY"))  # fmt: skip

        probe()
    try:
        result = label_batch(
            args.runs_root, args.step, args.candidate, args.run_ids or None,
            controls_source=args.controls_source,
            make_client=lambda path: JudgeClient.from_config(cfg, cache_path=path,
                                                             max_workers=args.workers),
            load_prompts=harmful_prompts, sealed_dir=sealed_dir, probe=probe,
            command=sys.argv if argv is None else [prog, *argv])  # fmt: skip
    except (BatchRefused, LedgerError, OSError) as exc:
        sys.exit(f"study3_label_cli label: {exc}")
    if result.failure:
        sys.exit(result.failure)


def _diagnose_command(args: argparse.Namespace) -> None:
    try:
        record = json.loads(Path(args.from_file).read_text(encoding="utf-8"))
        path = write_diagnosis(args.runs_root, args.batch, record)
    except (OSError, ValueError) as exc:
        sys.exit(f"study3_label_cli diagnose: {exc}")
    print(json.dumps({"batch": args.batch, "cause": record["cause"], "path": str(path),
                      "sha256": _sha256(path.read_bytes())}))  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    draw = commands.add_parser("controls-draw", help="draw the control list once")
    draw.add_argument("--calibration", required=True, type=Path,
                      help="study 1's judge-calibration folder under RUNS_DIR")  # fmt: skip
    draw.add_argument("--seed", required=True, type=int)
    draw.add_argument("--archive", required=True, help="the folder's bucket URI (gs://)")
    draw.add_argument("--runs-root", type=Path, help="RUNS_DIR (default: from .env)")
    label = commands.add_parser("label", help="advance a persistent labeling batch")
    label.add_argument("--runs-root", required=True, type=Path)
    label.add_argument("--step", required=True, choices=("validation", "confirmation"))
    label.add_argument("--candidate", required=True, type=int, choices=(1, 2))
    label.add_argument("--controls-source", required=True, type=Path,
                       help="study 1's calibration folder, restored read only")  # fmt: skip
    label.add_argument("--workers", type=int, default=8)
    label.add_argument("--sealed-dir", type=Path,
                       help="where the sealed record goes (default $RUNS_DIR/sealed)")  # fmt: skip
    label.add_argument("run_ids", nargs="*", help="a new batch's runs; none: resume")
    diagnose = commands.add_parser("diagnose", help="the researcher's diagnosis of a mark")
    diagnose.add_argument("--runs-root", required=True, type=Path)
    diagnose.add_argument("--batch", required=True)
    diagnose.add_argument("--from", dest="from_file", required=True, type=Path)
    torn = commands.add_parser("resolve-torn", help="repair the ledger's torn tail")
    torn.add_argument("--runs-root", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "controls-draw":
        _draw_command(args)
    elif args.command == "label":
        _label_command(args, parser.prog, argv)
    elif args.command == "diagnose":
        _diagnose_command(args)
    else:
        try:
            voided = resolve_torn(args.runs_root)
        except LedgerError as exc:
            sys.exit(f"study3_label_cli resolve-torn: {exc}")
        for line in voided:
            print(json.dumps(line))


if __name__ == "__main__":
    main()
