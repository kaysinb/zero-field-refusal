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


def response_conditions(cfg: DictConfig) -> list[str]:
    """The conditions under which a run answers its harmful prompt set."""
    if cfg.get("run_type", "train") == THETA0:
        return [THETA0]
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
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    found = Counter((r["condition"], r["item_id"]) for r in records)
    twice = sorted(label_item_id(*key) for key, n in found.items() if n > 1)
    if twice:
        raise ValueError(f"{path} answers {twice[0]} twice ({len(twice)} such responses)")
    ids = list(prompts)[:limit] if limit else list(prompts)
    expected = {(c, pid) for c in response_conditions(cfg) for pid in ids}
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
) -> list[dict[str, Any]]:
    """Label every completed run in ``run_dirs`` with the judge named ``judge``.

    ``load_prompts(cfg)`` gives the run's harmful prompt texts by id; ``make_client(cache_path)``
    the judge client writing to that label file. Returns one summary per run, counts only: runs
    that are not completed are skipped, and a run that cannot be labeled gets an ``error``.
    """
    summaries = []
    for run_dir in map(Path, run_dirs):
        try:
            summaries.append(_label_run(run_dir, judge, load_prompts, make_client))
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
    path = label_path(run_dir, judge)
    path.parent.mkdir(exist_ok=True)
    records = make_client(path).label_items(items)
    return {"run_id": run_dir.name, "items": len(items), "unparseable": count_unparseable(records)}
