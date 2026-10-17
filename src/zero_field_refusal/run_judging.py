"""Label the responses of run directories with a judge, into the label files the analysis reads.

``judge_items(run_dir, prompts)`` turns a run's ``responses/<eval.harmful.prompt_set>.jsonl`` into
judge items ``{item_id, prompt, response}``, one per response and condition, keyed as
``analysis.evidence`` expects (``<condition>/<prompt id>``). It first checks that the file holds
exactly the responses the run's config implies: every prompt of the set (the first
``eval.harmful.limit`` when set) under every condition (``response_conditions``), once each; a
truncated or mixed file stops that run. ``label_runs`` labels each labelable run into
``<run dir>/judge/<judge>.jsonl``, which is also the judge's cache: a round sends only the items
not yet labeled, including the ones whose output did not parse. A run that cannot be labeled is
reported with its error and the next run goes on.

Which runs are labelable (``unlabelable``): a study-2 run once its terminal outcome was published
and its records are whole (``analysis.stage_a.load_trajectory`` accepts them), whatever
``meta.json``'s status says, as the launcher decides it: a measurement that fails after the
terminal record (the endpoint Rayleigh quotient, say) leaves the status ``failed`` and the
endpoint stands. Any other run once its status is ``completed``.

**Rounds.** A round is one ``JudgeClient.label_items`` call that sends requests. It is made only
while some expected item has no finished (non-null) label in the label file, and it sends one
request per distinct prompt and response among those items (the client's retries of a transient
error belong to that request). A reply already in the label file, under any item id, is reused
and is no new round; a pass that finds every item labeled sends nothing and counts no round. The
rounds of a run and judge are limited in all, across every invocation, to ``label_runs``'
``rounds`` (at most ``LABEL_ROUNDS``). The rounds record (``rounds_path``:
``<run dir>/judge/<judge>.rounds.json``: ``run_id``, ``judge``, ``identity``, ``used``,
``limit``, ``updated_at``) counts each round before any of its requests is sent, written whole
(a rename) and synced to disk, so an interrupted round counts as used. A rerun resumes with the
rounds left, and sends nothing once they are used. Nothing lowers or deletes the record; deleting
or invalidating a receipt leaves it as it is. A label file without a rounds record (labeled before
the record existed) has unknown rounds behind it: its run is refused and nothing is sent, unless
its receipt is complete (below). So is a record that does not read or names another judge.

**The judging receipt** (``receipt_path``: ``<run dir>/judge/<judge>.receipt.json``). When the
labeling of a run is over (every item labeled, or the rounds used), ``label_runs`` writes a JSON
object: ``run_id`` (the directory's name), ``judge`` (the config name) and ``identity``
(``judge.IDENTITY_FIELDS``, the client's), ``expected_items`` (the item ids ``judge_items``
expected, in order: each recorded condition × each prompt id of the set), ``items`` (their
number), ``rounds`` (the rounds used in all: the rounds record's count, earlier invocations and
interrupted rounds included), ``rounds_allowed`` (this invocation's ``rounds``), ``missing`` (the
expected items still without a non-null label), ``labels_sha256`` (the sha256 of the label file
at completion) and ``completed_at``. It is written whole (a rename), and the receipt of an earlier
labeling is deleted before a round starts, so an interrupted round leaves no receipt: the analysis
(``analysis.study2_labels``) counts no label of a run without a valid receipt. A run whose rounds
are used but has no valid receipt (its last round interrupted, its receipt deleted) gets the
receipt of its label file as it stands, nothing sent.

**A completed receipt is kept.** A receipt that the analysis would read (this judge and identity,
exactly the expected items, its counts: ``study2_labels`` checks the same) and that matches the
label file's sha256 is left alone, nothing sent and nothing written, when it records no missing
label or the rounds used reach ``rounds`` (so always after ``LABEL_ROUNDS``). One with missing
labels and rounds left (labeled with ``rounds`` below ``LABEL_ROUNDS``) is resumed with them.

Study-2 runs. A training run (``run_type`` train) whose config has a ``schedule`` section
(``schedule.Schedule``; θ₀ and curvature runs may carry one and are not study-2 runs) answers
its harmful prompts at every evaluation it recorded, under the condition ``step-<N>`` of that
evaluation's step N. ``response_conditions(cfg,
run_dir)`` reads them, in order, from ``<run dir>/evaluations.jsonl`` (the rows
``schedule.Schedule._evaluate`` writes; ``terminal.json`` lists only their steps, which must
agree) and raises for a condition that is not ``step-<its step>``, a repeated step or a record
that disagrees with ``terminal.json``. Which evaluations a run recorded does not depend on its stop
reason, so every one is labeled; which of them enter an analysis is decided elsewhere, never from
labels. A completed run that stopped before its first evaluation has no ``evaluations.jsonl`` (its
``terminal.json`` lists no step), nothing to label, and is skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .analysis.evidence import JUDGE_DIR, TRAINED, label_item_id, label_path
from .analysis.stage_a import TrajectoryError, load_trajectory
from .evaluation import line_scan_condition, random_condition
from .judge import IDENTITY_FIELDS, JudgeClient, count_unparseable

THETA0 = "theta0"  # the condition of a θ₀ evaluation (train.evaluate_theta0)
# The labeling rounds study 2 allows in all, per run and judge across every invocation (an output
# that does not parse is rerun, up to three rounds); scripts/label_runs.py's default and maximum.
LABEL_ROUNDS = 3


def receipt_path(run_dir: Path, judge: str) -> Path:
    """Where the judging receipt of the judge named ``judge`` lives in a run directory."""
    return Path(run_dir) / JUDGE_DIR / f"{judge}.receipt.json"


def rounds_path(run_dir: Path, judge: str) -> Path:
    """Where the rounds the judge named ``judge`` used on a run are counted (the rounds record)."""
    return Path(run_dir) / JUDGE_DIR / f"{judge}.rounds.json"


def is_scheduled(cfg: DictConfig) -> bool:
    """Whether a run config is a study-2 training run: ``run_type`` train (the default) with a
    ``schedule`` section. A θ₀ run and a curvature run may carry one too, and are not."""
    return cfg.get("run_type", "train") == "train" and cfg.get("schedule") is not None


def scheduled_conditions(
    run_dir: Path, read: Callable[[Path], bytes] = Path.read_bytes
) -> list[str]:
    """The conditions of the evaluations a study-2 run recorded, in order: ``step-<step>`` each.

    From ``evaluations.jsonl``, which a run creates at its first evaluation: a run that stopped
    before it has no such file and records ``[]`` in ``terminal.json``, which every completed run
    has and which must list the same steps. ``read`` gives the bytes of a file (the reader hashes
    what it parses)."""
    run_dir = Path(run_dir)
    path, terminal = run_dir / "evaluations.jsonl", run_dir / "terminal.json"
    if not terminal.is_file():
        raise ValueError(f"{terminal} is missing: the evaluations of the run are unknown")
    lines = read(path).decode().splitlines() if path.is_file() else []
    rows = [json.loads(line) for line in lines if line.strip()]
    conditions = []
    for number, row in enumerate(rows, 1):
        step = row.get("step") if isinstance(row, dict) else None
        condition = row.get("condition") if isinstance(row, dict) else None
        if not isinstance(step, int) or isinstance(step, bool) or condition != f"step-{step}":
            raise ValueError(
                f"{path}, evaluation {number}: condition {condition!r} is not 'step-<step>' "
                f"for step {step!r}"
            )
        conditions.append(condition)
    if len(set(conditions)) != len(conditions):
        repeated = sorted(c for c, n in Counter(conditions).items() if n > 1)
        raise ValueError(f"{path} records the evaluation {repeated[0]} twice")
    steps = json.loads(read(terminal).decode()).get("evaluations")
    if steps != [int(c.removeprefix("step-")) for c in conditions]:
        raise ValueError(f"{path} and {terminal} record different evaluations")
    return conditions


def response_conditions(cfg: DictConfig, run_dir: Path | None = None) -> list[str]:
    """The conditions under which a run answers its harmful prompt set. A study-2 run
    (``is_scheduled``) needs ``run_dir``: its conditions are its recorded evaluations
    (``scheduled_conditions``), none for a run that stopped before the first."""
    if cfg.get("run_type", "train") == THETA0:
        return [THETA0]
    if is_scheduled(cfg):
        if run_dir is None:
            raise ValueError("the conditions of a scheduled (study 2) run need its run directory")
        return scheduled_conditions(run_dir)
    conditions = [TRAINED]
    controls = cfg.eval.controls
    if controls.line_scan.enabled:
        conditions += [line_scan_condition(sign * a) for sign in controls.line_scan.signs
                       for a in controls.line_scan.coefficients]  # fmt: skip
    if controls.random_direction.enabled:
        conditions += [random_condition(kind) for kind in controls.random_direction.kinds]
    return conditions


def check_responses(
    path: Path, records: list[dict[str, Any]], conditions: list[str], ids: list[str]
) -> None:
    """Raise unless ``records`` (the rows of the responses file ``path``) hold exactly one
    response to each of ``ids`` under each of ``conditions``, and no other."""
    found = Counter((r["condition"], r["item_id"]) for r in records)
    twice = sorted(label_item_id(*key) for key, n in found.items() if n > 1)
    if twice:
        raise ValueError(f"{path} answers {twice[0]} twice ({len(twice)} such responses)")
    expected = {(c, pid) for c in conditions for pid in ids}
    missing, extra = expected - set(found), set(found) - expected
    if missing or extra:
        raise ValueError(
            f"{path} does not hold the expected {len(expected)} responses: {len(missing)} missing "
            f"(first: {label_item_id(*min(missing)) if missing else None}), {len(extra)} "
            f"unexpected (first: {label_item_id(*min(extra)) if extra else None})"
        )


def judge_items(run_dir: Path, prompts: Mapping[str, str]) -> list[dict[str, str]]:
    """Items for every response of the run's harmful prompt set; ``prompts``: the whole set's texts
    by prompt id, in the set's order. A scheduled run that stopped before its first evaluation has
    none (its responses file is absent; one that exists is checked and fails)."""
    return _judge_items(run_dir, prompts)[0]


def prompt_ids(cfg: DictConfig, prompts: Mapping[str, str]) -> list[str]:
    """The prompt ids a run answers: those of ``prompts`` (the whole set, in its order), the first
    ``eval.harmful.limit`` when that is set."""
    limit = cfg.eval.harmful.get("limit")
    return list(prompts)[:limit] if limit else list(prompts)


def expected_items(conditions: Iterable[str], ids: Iterable[str]) -> list[str]:
    """The item ids of a run's labeling, in order: each condition × each prompt id."""
    ids = list(ids)
    return [label_item_id(condition, pid) for condition in conditions for pid in ids]


def _judge_items(
    run_dir: Path, prompts: Mapping[str, str]
) -> tuple[list[dict[str, str]], list[str]]:
    """``judge_items`` and the item ids it expected (``expected_items``)."""
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    assert isinstance(cfg, DictConfig)
    path = Path(run_dir) / "responses" / f"{cfg.eval.harmful.prompt_set}.jsonl"
    conditions = response_conditions(cfg, run_dir)
    if not conditions and not path.is_file():
        return [], []
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = prompt_ids(cfg, prompts)
    check_responses(path, records, conditions, ids)
    items = [{"item_id": label_item_id(r["condition"], r["item_id"]),
              "prompt": prompts[r["item_id"]], "response": r["text"]} for r in records]  # fmt: skip
    return items, expected_items(conditions, ids)


def unlabelable(run_dir: Path, cfg: DictConfig) -> str | None:
    """Why a run cannot be labeled (None: it can). A study-2 run (``is_scheduled``) can once its
    terminal record was published with whole records (``stage_a.load_trajectory``, as the
    launcher decides it), whatever its status; any other run once its status is completed."""
    if is_scheduled(cfg):
        try:
            load_trajectory(run_dir)
        except TrajectoryError as exc:
            return f"no published terminal record with whole records ({exc})"
        return None
    status = json.loads((Path(run_dir) / "meta.json").read_text()).get("status")
    return None if status == "completed" else f"status {status!r}"


def label_runs(
    run_dirs: Iterable[Path],
    judge: str,
    load_prompts: Callable[[DictConfig], Mapping[str, str]],
    make_client: Callable[[Path], JudgeClient],
    rounds: int = 1,
) -> list[dict[str, Any]]:
    """Label every labelable run in ``run_dirs`` (``unlabelable``) with the judge named ``judge``.

    A run with unparseable outputs left is labeled again, at most ``rounds`` rounds in all (1 to
    ``LABEL_ROUNDS``), counted across invocations by the run's rounds record; each round sends
    only the items still without a label (see the module docstring). Once the rounds are over, the
    run's judging receipt is written; a run with a completed receipt is left alone.

    ``load_prompts(cfg)`` gives the run's harmful prompt texts by id; ``make_client(cache_path)``
    the judge client writing to that label file. Returns one summary per run, counts only:
    ``items``, ``unparseable`` (the items left without a label) and, when not 1, ``rounds`` (the
    rounds used in all); ``kept`` when a completed receipt was left alone. A run that cannot be
    labeled is skipped with the reason, and a run whose labeling fails gets an ``error`` and no
    receipt.
    """
    if not 1 <= rounds <= LABEL_ROUNDS:
        raise ValueError(f"rounds must be 1 to {LABEL_ROUNDS}, not {rounds!r}")
    summaries = []
    for run_dir in map(Path, run_dirs):
        try:
            summaries.append(_label_run(run_dir, judge, load_prompts, make_client, rounds))
        except Exception as exc:  # reported per run; the other runs go on
            summaries.append({"run_id": run_dir.name, "error": f"{type(exc).__name__}: {exc}"})
    return summaries


def _label_run(
    run_dir: Path,
    judge: str,
    load_prompts: Callable[[DictConfig], Mapping[str, str]],
    make_client: Callable[[Path], JudgeClient],
    rounds: int,
) -> dict[str, Any]:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    assert isinstance(cfg, DictConfig)
    why = unlabelable(run_dir, cfg)
    if why is not None:
        return {"run_id": run_dir.name, "skipped": why}
    items, expected = _judge_items(run_dir, load_prompts(cfg))
    if not items:
        return {"run_id": run_dir.name, "skipped": "no evaluation recorded"}
    path = label_path(run_dir, judge)
    path.parent.mkdir(exist_ok=True)
    receipt, record = receipt_path(run_dir, judge), rounds_path(run_dir, judge)
    client = make_client(path)
    used = _rounds_used(record, judge, client.identity)
    done = _completed_receipt(run_dir, judge, client.identity, expected, path)
    spent = max(used or 0, done["rounds"] if done else 0)
    summary: dict[str, Any] = {"run_id": run_dir.name, "items": len(items)}
    if done is not None and (done["missing"] == 0 or spent >= rounds):
        # The labeling is over: nothing is sent, the receipt and the labels stay as they are.
        summary.update(unparseable=done["missing"], kept="its completed judging receipt")
        return summary if done["rounds"] == 1 else {**summary, "rounds": done["rounds"]}
    if used is None and path.exists():
        raise ValueError(f"{path} has no rounds record ({record.name}): the rounds already used "
                         "on it are unknown, so none is sent")  # fmt: skip
    records = client.label_items(items, send=False)  # the labels already there; no request
    while count_unparseable(records) and spent < rounds:
        receipt.unlink(missing_ok=True)  # a round starts: no label of the run counts meanwhile
        spent += 1
        _write_json(record, {  # the round counts as used before any of its requests is sent
            "what": "The labeling rounds of this run by this judge, each counted before its "
                    "requests are sent (run_judging.label_runs); never reset.",
            "run_id": run_dir.name,
            "judge": judge,
            "identity": client.identity,
            "used": spent,
            "limit": LABEL_ROUNDS,
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        })  # fmt: skip
        records = client.label_items(items)
    if not path.exists():  # no round could run, and nothing was labeled before
        raise ValueError(f"{path} is missing and the run's {spent} labeling rounds of {rounds} "
                         "are used: nothing is sent")  # fmt: skip
    missing = count_unparseable(records)
    _write_json(receipt, {
        "what": "The judging receipt: the labeling of this run by this judge completed.",
        "run_id": run_dir.name,
        "judge": judge,
        "identity": client.identity,
        "expected_items": expected,
        "items": len(expected),
        "rounds": spent,
        "rounds_allowed": rounds,
        "missing": missing,
        "labels_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })  # fmt: skip
    summary["unparseable"] = missing
    return summary if spent == 1 else {**summary, "rounds": spent}


def _rounds_used(path: Path, judge: str, identity: Mapping[str, Any]) -> int | None:
    """The rounds the rounds record at ``path`` counts; None without a record. Raises
    ``ValueError`` for a record that does not read or counts another judge's rounds."""
    if not path.exists():
        return None
    try:
        found = json.loads(path.read_text())
    except ValueError as exc:
        raise ValueError(f"{path} is not a rounds record: nothing is sent") from exc
    used = found.get("used") if isinstance(found, dict) else None
    if not isinstance(used, int) or isinstance(used, bool) or used < 0:
        raise ValueError(f"{path} is not a rounds record: nothing is sent")
    recorded = found.get("identity")
    if (found.get("judge") != judge or not isinstance(recorded, dict)
            or any(recorded.get(key) != identity.get(key) for key in IDENTITY_FIELDS)):  # fmt: skip
        raise ValueError(f"{path} counts the rounds of another judge: nothing is sent")
    return used


def _completed_receipt(
    run_dir: Path, judge: str, identity: Mapping[str, Any], expected: list[str], labels: Path
) -> dict[str, Any] | None:
    """The run's judging receipt when the analysis would read it (``study2_labels``' checks of the
    receipt) and it was written for the label file as it is (its sha256); None otherwise."""
    from .analysis.study2_labels import LabelsPending, _receipt  # it imports this module

    try:
        receipt = _receipt(run_dir, judge, identity, expected, Path.read_bytes)
    except LabelsPending:
        return None
    if not labels.is_file():
        return None
    digest = hashlib.sha256(labels.read_bytes()).hexdigest()
    return receipt if receipt["labels_sha256"] == digest else None


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write ``data`` whole or not at all (a rename), on disk once this returns (synced)."""
    partial = path.with_name(path.name + ".partial")
    with partial.open("w") as handle:
        handle.write(json.dumps(data, indent=1) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
