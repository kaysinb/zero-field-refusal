"""Label the responses of run directories with a judge, into the label files the analysis reads.

``judge_items(run_dir, prompts)`` turns a run's ``responses/<eval.harmful.prompt_set>.jsonl`` into
judge items ``{item_id, prompt, response}``, one per response and condition, keyed as
``analysis.evidence`` expects (``<condition>/<prompt id>``). It first checks that the file holds
exactly the responses the run's config implies: every prompt of the set (the first
``eval.harmful.limit`` when set) under every condition (``response_conditions``), once each; a
truncated or mixed file stops that run. ``label_runs`` labels each labelable run into
``<run dir>/judge/<judge>.jsonl``, which is also the judge's cache: a rerun sends only the items
not yet labeled, including the ones whose output did not parse. A run that cannot be labeled is
reported with its error and the next run goes on.

Which runs are labelable (``unlabelable``): a study-2 run once its terminal outcome was published
and its records are whole (``analysis.stage_a.load_trajectory`` accepts them), whatever
``meta.json``'s status says, as the launcher decides it: a measurement that fails after the
terminal record (the endpoint Rayleigh quotient, say) leaves the status ``failed`` and the
endpoint stands. Any other run once its status is ``completed``.

**The judging receipt** (``receipt_path``: ``<run dir>/judge/<judge>.receipt.json``). When the
labeling of a run finishes, every round included, ``label_runs`` writes a JSON object: ``run_id``
(the directory's name), ``judge`` (the config name) and ``identity`` (``judge.IDENTITY_FIELDS``,
the client's), ``expected_items`` (the item ids ``judge_items`` expected, in order: each recorded
condition × each prompt id of the set), ``items`` (their number), ``rounds`` (the rounds used),
``rounds_allowed``, ``missing`` (the expected items still without a non-null label),
``labels_sha256`` (the sha256 of the label file at completion) and ``completed_at``. The receipt
of an earlier labeling is deleted before a new one starts, and the new one is written (whole, by
a rename) only once it is done, so an interrupted labeling leaves no receipt: the analysis
(``analysis.study2_labels``) counts no label of a run without a valid receipt.

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
from .judge import JudgeClient, count_unparseable

THETA0 = "theta0"  # the condition of a θ₀ evaluation (train.evaluate_theta0)
# The labeling rounds study 2 allows in all (an output that does not parse is rerun, up to three
# rounds); scripts/label_runs.py's default and maximum.
LABEL_ROUNDS = 3


def receipt_path(run_dir: Path, judge: str) -> Path:
    """Where the judging receipt of the judge named ``judge`` lives in a run directory."""
    return Path(run_dir) / JUDGE_DIR / f"{judge}.receipt.json"


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

    A run with unparseable outputs left is labeled again, at most ``rounds`` times in all (each
    round sends only the items still without a label); its summary then has ``rounds``, the
    number of rounds used, and its ``unparseable`` is what remains missing. Once its rounds are
    over, the run's judging receipt is written (see the module docstring).

    ``load_prompts(cfg)`` gives the run's harmful prompt texts by id; ``make_client(cache_path)``
    the judge client writing to that label file. Returns one summary per run, counts only: a run
    that cannot be labeled is skipped with the reason, and a run whose labeling fails gets an
    ``error`` and no receipt.
    """
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
    receipt = receipt_path(run_dir, judge)
    receipt.unlink(missing_ok=True)  # labeling in progress: no label of the run counts meanwhile
    client = make_client(path)
    used = 0
    while True:
        missing = count_unparseable(client.label_items(items))
        used += 1
        if not missing or used >= rounds:
            break
    _write_receipt(receipt, {
        "what": "The judging receipt: the labeling of this run by this judge completed.",
        "run_id": run_dir.name,
        "judge": judge,
        "identity": client.identity,
        "expected_items": expected,
        "items": len(expected),
        "rounds": used,
        "rounds_allowed": rounds,
        "missing": missing,
        "labels_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "completed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })  # fmt: skip
    summary = {"run_id": run_dir.name, "items": len(items), "unparseable": missing}
    return summary if used == 1 else {**summary, "rounds": used}


def _write_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    """Write the receipt whole or not at all (a rename)."""
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(receipt, indent=1) + "\n")
    os.replace(partial, path)
