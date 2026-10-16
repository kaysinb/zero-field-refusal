"""Judge-label counts at the evaluations (endpoints) of a study-2 run, read strictly.

    read_endpoint_labels(run_dir, judge, identity, prompts) -> list[dict]

``run_dir`` is a study-2 run directory (a training run with a ``schedule`` in its config),
``judge`` the name of the judge (its labels are ``<run dir>/judge/<judge>.jsonl``, written by
``scripts/label_runs.py``),
``identity`` the judge's identity (``judge.judge_identity``), and ``prompts`` the texts of the
whole harmful prompt set by prompt id, in the set's order (``data.load_prompt_set``, as
``scripts/label_runs.py`` loads them; tests inject them). The first ``eval.harmful.limit`` ids are
the run's set when that is set. Nothing is computed by a model: this is counting.

One row per evaluation the run recorded (``run_judging.response_conditions``), in order, with
``condition`` (``step-<N>``), ``step``, ``prompts`` (the set's size), ``labeled`` (prompts with a
label), ``missing`` (``prompts - labeled``), ``harmful``, ``refusal``, ``safe_non_refusal`` (the
labeled prompts by label), ``harmful_rate`` (``harmful / labeled``, null when nothing is labeled),
``harmful_low`` (``harmful / prompts``: every missing label counts as not harmful) and
``harmful_high`` (``(harmful + missing) / prompts``: every missing label counts as harmful), and
``files``: the sha256 of every file read, by name relative to the run directory. A null label
(unparseable judge output) labels nothing; an item without a non-null label is missing. Missing
labels are reported through these bounds, never imputed. A run that stopped before its first
evaluation (no ``evaluations.jsonl``, none in ``terminal.json``) has no rows.

Each check raises ``ValueError`` naming the run directory and the first offending item:
- the run recorded its harmful prompts (``meta.json`` ``inputs.harmful_prompts``: set, count, ids
  hash) as ``prompts`` gives them, and the responses file holds exactly one response for each
  prompt id under each evaluation condition, and no response under another condition;
- every label record carries exactly ``identity`` (``judge.IDENTITY_FIELDS``);
- every label record names a response of the file and its ``content_sha256`` is the hash
  (``judge._content_hash``) of that prompt and the response *as the file holds it now*: a label
  of an older response text, or a record without the hash, fails;
- an item has no two different labels, and no label is outside ``judge.LABELS`` (null apart).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..judge import IDENTITY_FIELDS, LABELS, _content_hash
from ..run_judging import check_responses, is_scheduled, scheduled_conditions
from .evidence import HARMFUL, label_item_id, label_path

Reader = Callable[[Path], bytes]


def read_endpoint_labels(
    run_dir: Path, judge: str, identity: Mapping[str, Any], prompts: Mapping[str, str]
) -> list[dict[str, Any]]:
    """The label counts of every evaluation of a study-2 run; see the module docstring."""
    run_dir = Path(run_dir)
    files: dict[str, str] = {}

    def read(path: Path) -> bytes:
        data = path.read_bytes()
        files[path.relative_to(run_dir).as_posix()] = hashlib.sha256(data).hexdigest()
        return data

    cfg = OmegaConf.create(read(run_dir / "config.yaml").decode())
    assert isinstance(cfg, DictConfig)
    if not is_scheduled(cfg):
        raise ValueError(f"{run_dir}: not a study-2 training run")
    conditions = scheduled_conditions(run_dir, read)
    if not conditions:
        return []
    limit = cfg.eval.harmful.get("limit")
    ids = list(prompts)[:limit] if limit else list(prompts)
    _check_recorded_prompts(run_dir, cfg, ids, read(run_dir / "meta.json"))
    texts = _responses(run_dir, cfg, conditions, ids, read)
    items = {label_item_id(c, pid): (prompts[pid], text) for (c, pid), text in texts.items()}
    labels = _labels(label_path(run_dir, judge), identity, items, read)

    rows = []
    for condition in conditions:
        found = [labels.get(label_item_id(condition, pid)) for pid in ids]
        count = Counter(label for label in found if label is not None)
        labeled = sum(count.values())
        missing = len(ids) - labeled
        rows.append({
            "condition": condition, "step": int(condition.removeprefix("step-")),
            "prompts": len(ids), "labeled": labeled, "missing": missing,
            "harmful": count[HARMFUL], "refusal": count["refusal"],
            "safe_non_refusal": count["safe_non_refusal"],
            "harmful_rate": count[HARMFUL] / labeled if labeled else None,
            "harmful_low": count[HARMFUL] / len(ids),
            "harmful_high": (count[HARMFUL] + missing) / len(ids),
        })  # fmt: skip
    return [{**row, "files": dict(files)} for row in rows]


def _check_recorded_prompts(run_dir: Path, cfg: DictConfig, ids: list[str], meta: bytes) -> None:
    recorded = json.loads(meta).get("inputs", {}).get("harmful_prompts")
    wanted = {"set": cfg.eval.harmful.prompt_set, "count": len(ids),
              "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}  # fmt: skip
    if not isinstance(recorded, dict):
        raise ValueError(f"{run_dir}/meta.json does not record the run's harmful prompts")
    for key, value in wanted.items():
        if recorded.get(key) != value:
            raise ValueError(
                f"{run_dir}/meta.json: the run recorded harmful prompts with {key} "
                f"{recorded.get(key)!r}, the prompt set given has {value!r}"
            )


def _responses(
    run_dir: Path, cfg: DictConfig, conditions: list[str], ids: list[str], read: Reader
) -> dict[tuple[str, str], str]:
    """The text of each response by (condition, prompt id); exactly the expected ones, once."""
    path = run_dir / "responses" / f"{cfg.eval.harmful.prompt_set}.jsonl"
    rows = [json.loads(line) for line in read(path).decode().splitlines() if line.strip()]
    check_responses(path, rows, conditions, ids)
    return {(r["condition"], r["item_id"]): r["text"] for r in rows}


def _labels(
    path: Path, identity: Mapping[str, Any], items: dict[str, tuple[str, str]], read: Reader
) -> dict[str, str]:
    """The label of each labeled item. Null labels are left out; every record is checked."""
    if not path.is_file():
        raise ValueError(f"{path} is missing: the run is not labeled by this judge")
    lines = [line for line in read(path).decode().splitlines() if line.strip()]
    labels: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}, line {number}, is not valid JSON") from exc
        item = record.get("item_id") if isinstance(record, dict) else None
        if not isinstance(item, str):
            raise ValueError(f"{path}, line {number}, names no item")
        if any(record.get(key) != identity.get(key) for key in IDENTITY_FIELDS):
            raise ValueError(f"{path}: {item} is labeled by another judge than the one asked for")
        if item not in items:
            raise ValueError(f"{path}: {item} names no response of the run")
        if record.get("content_sha256") != _content_hash(*items[item]):
            raise ValueError(f"{path}: the label of {item} is not of the response stored now")
        if "label" not in record:
            raise ValueError(f"{path}: the record of {item} has no label field")
        label = record["label"]
        if label is None:
            continue
        if label not in LABELS:
            raise ValueError(f"{path}: {item} has the label {label!r}, not one of {LABELS}")
        if labels.setdefault(item, label) != label:
            raise ValueError(f"{path}: {item} has two different labels")
    return labels
