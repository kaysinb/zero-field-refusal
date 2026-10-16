"""Label the responses of run directories with a judge, into the label files the analysis reads.

``judge_items(run_dir, prompts)`` turns a run's ``responses/<eval.harmful.prompt_set>.jsonl`` into
judge items ``{item_id, prompt, response}``, one per response and condition, keyed as
``analysis.evidence`` expects (``<condition>/<prompt id>``). It first checks that the file holds
exactly the responses the run's config implies: every prompt of the set (the first
``eval.harmful.limit`` when set) under every condition (``response_conditions``), once each; a
truncated or mixed file stops that run. ``label_runs`` labels each completed run into
``<run dir>/judge/<judge>.jsonl``, which is also the judge's cache: a rerun sends only the items
not yet labeled, including the ones whose output did not parse. A run that cannot be labeled is
reported with its error and the next run goes on.

Study-2 runs. A run whose config has a ``schedule`` section (``schedule.Schedule``; a θ₀ run keeps
its own ``run_type`` and is not one) answers its harmful prompts at every evaluation it recorded,
under the condition ``step-<N>`` of that evaluation's step N. ``response_conditions(cfg,
run_dir)`` reads them, in order, from ``<run dir>/evaluations.jsonl`` (the rows
``schedule.Schedule._evaluate`` writes; ``terminal.json`` lists only their steps, which must
agree) and raises for a condition that is not ``step-<its step>``, a repeated step or a record
that disagrees with ``terminal.json``. Which evaluations a run recorded does not depend on its stop
reason, so every one is labeled; which of them enter an analysis is decided elsewhere, never from
labels. A completed run with no evaluation has nothing to label and is skipped.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .analysis.evidence import TRAINED, label_item_id, label_path
from .evaluation import line_scan_condition, random_condition
from .judge import JudgeClient, count_unparseable

THETA0 = "theta0"  # the condition of a θ₀ evaluation (train.evaluate_theta0)


def scheduled_conditions(run_dir: Path) -> list[str]:
    """The conditions of the evaluations a study-2 run recorded, in order: ``step-<step>`` each."""
    run_dir = Path(run_dir)
    path = run_dir / "evaluations.jsonl"
    if not path.is_file():
        raise ValueError(f"{path} is missing: the evaluations of the run are unknown")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
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
    terminal = run_dir / "terminal.json"
    if terminal.is_file():
        steps = json.loads(terminal.read_text()).get("evaluations")
        if steps != [int(c.removeprefix("step-")) for c in conditions]:
            raise ValueError(f"{path} and {terminal} record different evaluations")
    return conditions


def response_conditions(cfg: DictConfig, run_dir: Path | None = None) -> list[str]:
    """The conditions under which a run answers its harmful prompt set. A study-2 run (a
    ``schedule`` section, and not a θ₀ run) needs ``run_dir``: its conditions are its recorded
    evaluations (``scheduled_conditions``)."""
    if cfg.get("run_type", "train") == THETA0:
        return [THETA0]
    if cfg.get("schedule") is not None:
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


def judge_items(run_dir: Path, prompts: Mapping[str, str]) -> list[dict[str, str]]:
    """Items for every response of the run's harmful prompt set; ``prompts``: the whole set's texts
    by prompt id, in the set's order."""
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    assert isinstance(cfg, DictConfig)
    prompt_set, limit = cfg.eval.harmful.prompt_set, cfg.eval.harmful.get("limit")
    path = Path(run_dir) / "responses" / f"{prompt_set}.jsonl"
    conditions = response_conditions(cfg, run_dir)
    if not conditions:  # a scheduled run that recorded no evaluation answered nothing
        return []
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    found = Counter((r["condition"], r["item_id"]) for r in records)
    twice = sorted(label_item_id(*key) for key, n in found.items() if n > 1)
    if twice:
        raise ValueError(f"{path} answers {twice[0]} twice ({len(twice)} such responses)")
    ids = list(prompts)[:limit] if limit else list(prompts)
    expected = {(c, pid) for c in conditions for pid in ids}
    missing, extra = expected - set(found), set(found) - expected
    if missing or extra:
        raise ValueError(
            f"{path} does not hold the expected {len(expected)} responses: {len(missing)} missing "
            f"(first: {label_item_id(*min(missing)) if missing else None}), {len(extra)} "
            f"unexpected (first: {label_item_id(*min(extra)) if extra else None})"
        )
    return [{"item_id": label_item_id(r["condition"], r["item_id"]),
             "prompt": prompts[r["item_id"]], "response": r["text"]} for r in records]  # fmt: skip


def label_runs(
    run_dirs: Iterable[Path],
    judge: str,
    load_prompts: Callable[[DictConfig], Mapping[str, str]],
    make_client: Callable[[Path], JudgeClient],
    rounds: int = 1,
) -> list[dict[str, Any]]:
    """Label every completed run in ``run_dirs`` with the judge named ``judge``.

    A run with unparseable outputs left is labeled again, at most ``rounds`` times in all (each
    round sends only the items still without a label); its summary then has ``rounds``, the
    number of rounds used, and its ``unparseable`` is what remains missing.

    ``load_prompts(cfg)`` gives the run's harmful prompt texts by id; ``make_client(cache_path)``
    the judge client writing to that label file. Returns one summary per run, counts only: runs
    that are not completed are skipped, and a run that cannot be labeled gets an ``error``.
    """
    summaries = []
    for run_dir in map(Path, run_dirs):
        try:
            summary = _label_run(run_dir, judge, load_prompts, make_client)
            used = 1
            while summary.get("unparseable") and used < rounds:
                summary, used = _label_run(run_dir, judge, load_prompts, make_client), used + 1
            summaries.append(summary if used == 1 else {**summary, "rounds": used})
        except Exception as exc:  # reported per run; the other runs go on
            summaries.append({"run_id": run_dir.name, "error": f"{type(exc).__name__}: {exc}"})
    return summaries


def _label_run(
    run_dir: Path,
    judge: str,
    load_prompts: Callable[[DictConfig], Mapping[str, str]],
    make_client: Callable[[Path], JudgeClient],
) -> dict[str, Any]:
    status = json.loads((run_dir / "meta.json").read_text()).get("status")
    if status != "completed":
        return {"run_id": run_dir.name, "skipped": f"status {status!r}"}
    cfg = OmegaConf.load(run_dir / "config.yaml")
    assert isinstance(cfg, DictConfig)
    items = judge_items(run_dir, load_prompts(cfg))
    if not items:
        return {"run_id": run_dir.name, "skipped": "no evaluation recorded"}
    path = label_path(run_dir, judge)
    path.parent.mkdir(exist_ok=True)
    records = make_client(path).label_items(items)
    return {"run_id": run_dir.name, "items": len(items), "unparseable": count_unparseable(records)}
