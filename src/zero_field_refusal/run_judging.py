"""Label the responses of run directories with a judge, into the label files the analysis reads.

``judge_items(run_dir, prompts)`` turns a run's ``responses/<eval.harmful.prompt_set>.jsonl`` into
judge items ``{item_id, prompt, response}``, one per response and condition, keyed as
``analysis.evidence`` expects (``<condition>/<prompt id>``). ``label_runs`` labels each completed
run into ``<run dir>/judge/<judge>.jsonl``, which is also the judge's cache: a rerun sends only
the items not yet labeled, including the ones whose output did not parse.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .analysis.evidence import label_item_id, label_path
from .judge import JudgeClient, count_unparseable


def judge_items(run_dir: Path, prompts: Mapping[str, str]) -> list[dict[str, str]]:
    """Items for every response of the run's harmful prompt set; ``prompts``: prompt id -> text."""
    cfg = OmegaConf.load(Path(run_dir) / "config.yaml")
    prompt_set = cfg.eval.harmful.prompt_set
    path = Path(run_dir) / "responses" / f"{prompt_set}.jsonl"
    items, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        item_id = label_item_id(record["condition"], record["item_id"])
        if item_id in seen:
            raise ValueError(f"{path} answers {item_id} twice")
        if record["item_id"] not in prompts:
            raise ValueError(f"{path}: prompt {record['item_id']} is not in {prompt_set}")
        seen.add(item_id)
        items.append({"item_id": item_id, "prompt": prompts[record["item_id"]],
                      "response": record["text"]})  # fmt: skip
    return items


def label_runs(
    run_dirs: Iterable[Path],
    judge: str,
    load_prompts: Callable[[DictConfig], Mapping[str, str]],
    make_client: Callable[[Path], JudgeClient],
) -> list[dict[str, Any]]:
    """Label every completed run in ``run_dirs`` with the judge named ``judge``.

    ``load_prompts(cfg)`` gives the run's harmful prompt texts by id; ``make_client(cache_path)``
    the judge client writing to that label file. Returns one summary per run, counts only: runs
    that are not completed are skipped and say so.
    """
    summaries = []
    for run_dir in map(Path, run_dirs):
        status = json.loads((run_dir / "meta.json").read_text()).get("status")
        if status != "completed":
            summaries.append({"run_id": run_dir.name, "skipped": f"status {status!r}"})
            continue
        cfg = OmegaConf.load(run_dir / "config.yaml")
        assert isinstance(cfg, DictConfig)
        items = judge_items(run_dir, load_prompts(cfg))
        path = label_path(run_dir, judge)
        path.parent.mkdir(exist_ok=True)
        records = make_client(path).label_items(items)
        summaries.append({"run_id": run_dir.name, "items": len(items),
                          "unparseable": count_unparseable(records)})  # fmt: skip
    return summaries
