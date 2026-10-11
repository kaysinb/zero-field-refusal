"""The runs table of one experiment, read from its run directories, with every exclusion explained.

``collect_runs(root, experiment, judge, identity)`` reads ``<root>/<experiment>/<run id>/`` of
every run, failed ones included, and returns the runs table that ``analyze`` takes (``runs``)
together with one row per run directory (``considered``): its status and, for a run left out,
a reason code and a detail. Every file read is hashed. Nothing is computed by a model or a judge.

A run's row in the runs table:
- ``arm``: ``optimizer.name``, with ``sgd`` at momentum > 0 as ``sgd_momentum`` (configs/optimizer/
  sgd_momentum.yaml), followed by ``+<control>`` when the run declares a control;
- ``lr``, ``seed``: ``optimizer.lr`` and ``seed`` of ``config.yaml``;
- ``K``: ``K_harmful`` of ``summary.json``, in nats;
- ``judged``: the trained model's responses in ``responses/<eval.harmful.prompt_set>.jsonl``
  (condition ``trained``); ``harmful``: those the judge labels ``harmful_compliance``.

Judge labels. A run's labels from the judge named ``judge`` (a config in ``configs/judge/``) are
the label records ``JudgeClient`` writes, in ``<run dir>/judge/<judge>.jsonl``, one per response
of ``responses/<eval.harmful.prompt_set>.jsonl``, keyed by ``label_item_id(condition, prompt
id)``, e.g. ``trained/judged:17``: the in-process controls (line scans, random directions) answer
the same prompts under other conditions, so the prompt id alone is ambiguous. Every record of the
file must carry the judge's identity (``judge.IDENTITY_FIELDS``). A record whose label is null
(unparseable output, which the judge retries) labels nothing; an item is labeled when one of its
records has a label.

Exclusion reasons, checked in this order (the first that applies is recorded):
- ``no_meta``: no readable ``meta.json``; ``no_config``: no ``config.yaml``;
- ``not_completed``: ``meta.json`` status other than ``completed`` (failed or still running);
- ``not_training``: ``run_type`` other than ``train`` (e.g. a θ₀ evaluation);
- ``unknown_arm``: an arm outside ``runs.ARMS``, such as a control run;
- ``no_summary``: no ``summary.json``; ``no_k``: no positive, finite ``K_harmful``;
- ``no_responses``: no readable responses of the trained model, or one prompt answered twice;
- ``no_labels``: no readable label file;
- ``judge_identity``: a label record from another judge identity;
- ``incomplete_labels``: a response of the trained model without a label;
- ``invalid_labels``: an item with two different labels, a label outside ``judge.LABELS``, or a
  record that names no response of the run.

The included runs must form one analysis, otherwise ``collect_runs`` raises instead of choosing:
one condition (the config without the optimizer section, the seed and the keys that do not
change what a run computes: model, protocol version, data, training and evaluation settings),
one optimizer configuration per arm apart from the learning rate, at most one run per seed in a
cell, and shared-teacher pairing: every run of a seed trained on the same training stream
(``meta.json`` inputs).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..config import NON_IDENTITY_KEYS
from ..judge import IDENTITY_FIELDS, LABELS
from .runs import ARMS, validate_runs

TRAINED = "trained"  # the condition of the trained model's responses (train.Evaluator)
HARMFUL = "harmful_compliance"
JUDGE_DIR = "judge"

CONSIDERED_COLUMNS = ("run_id", "status", "included", "reason", "detail", "arm", "lr", "seed",
                      "K", "harmful", "judged", "protocol_version", "config_hash",
                      "training_stream")  # fmt: skip
INTEGER_COLUMNS = ("seed", "harmful", "judged", "protocol_version")


def label_item_id(condition: str, prompt_id: str) -> str:
    """The ``item_id`` of the judge label of one response: ``<condition>/<prompt id>``."""
    return f"{condition}/{prompt_id}"


def label_path(run_dir: Path, judge: str) -> Path:
    """Where the labels of the judge named ``judge`` live in a run directory."""
    return run_dir / JUDGE_DIR / f"{judge}.jsonl"


def arm_of(cfg: DictConfig) -> str:
    """The arm of a run config: the optimizer, then ``+<control>`` for a control run."""
    arm = str(cfg.optimizer.name)
    if arm == "sgd" and (cfg.optimizer.get("momentum") or 0) > 0:
        arm = "sgd_momentum"
    control = cfg.get("control")
    return f"{arm}+{control}" if control else arm


@dataclass(frozen=True)
class Evidence:
    experiment: str
    judge: str
    judge_identity: dict[str, Any]
    runs: pd.DataFrame  # the runs table of the included runs (runs.validate_runs)
    considered: pd.DataFrame  # one row per run directory, CONSIDERED_COLUMNS, by run_id
    files: dict[str, str]  # every file read: path relative to the runs root -> sha256, sorted


def collect_runs(root: Path, experiment: str, judge: str, identity: Mapping[str, Any]) -> Evidence:
    """The runs table of ``<root>/<experiment>`` with the labels of ``judge``, whose label records
    must carry ``identity`` (``judge.judge_identity`` of the judge's config)."""
    root = Path(root)
    run_dirs = sorted(p for p in (root / experiment).glob("*") if p.is_dir())
    if not run_dirs:
        raise ValueError(f"There are no run directories in experiment {experiment!r}")
    files = _Files(root)
    rows, configs = [], {}
    for run_dir in run_dirs:
        row, cfg = _read_run(run_dir, judge, identity, files)
        rows.append(row)
        if row["included"]:
            configs[row["run_id"]] = cfg
    considered = pd.DataFrame(rows, columns=list(CONSIDERED_COLUMNS))
    for column in INTEGER_COLUMNS:
        considered[column] = considered[column].astype("Int64")
    considered["lr"] = considered["lr"].astype(float)
    considered["K"] = considered["K"].astype(float)

    included = considered[considered["included"]]
    if included.empty:
        reasons = Counter(considered["reason"])
        raise ValueError(f"No run of experiment {experiment!r} is included: {dict(reasons)}")
    _check_one_analysis(included, configs)
    runs = validate_runs(included.loc[:, ["run_id", "seed", "arm", "lr", "K", "harmful",
                                          "judged"]])  # fmt: skip
    return Evidence(
        experiment=experiment,
        judge=judge,
        judge_identity=dict(identity),
        runs=runs,
        considered=considered,
        files=dict(sorted(files.sha256.items())),
    )


class _Files:
    """Reads files under the runs root and records the sha256 of each."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.sha256: dict[str, str] = {}

    def text(self, path: Path) -> str | None:
        if not path.is_file():
            return None
        data = path.read_bytes()
        self.sha256[path.relative_to(self.root).as_posix()] = hashlib.sha256(data).hexdigest()
        return data.decode()


class _Excluded(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason, self.detail = reason, detail


def _read_run(
    run_dir: Path, judge: str, identity: Mapping[str, Any], files: _Files
) -> tuple[dict[str, Any], DictConfig | None]:
    row: dict[str, Any] = dict.fromkeys(CONSIDERED_COLUMNS) | {"run_id": run_dir.name}
    cfg = None
    try:
        cfg = _assess(run_dir, judge, identity, files, row)
    except _Excluded as excluded:
        row.update(included=False, reason=excluded.reason, detail=excluded.detail)
    else:
        row.update(included=True, reason="", detail="")
    return row, cfg


def _assess(
    run_dir: Path, judge: str, identity: Mapping[str, Any], files: _Files, row: dict[str, Any]
) -> DictConfig:
    """Fill ``row`` with what the run directory holds; raise ``_Excluded`` to leave it out."""
    meta = _json(files, run_dir / "meta.json", "no_meta")
    row["status"] = meta.get("status")
    row["config_hash"] = meta.get("config_hash")
    streams = (meta.get("inputs") or {}).get("streams") or {}
    row["training_stream"] = (streams.get("training") or {}).get("sha256")
    text = files.text(run_dir / "config.yaml")
    if text is None:
        raise _Excluded("no_config", "config.yaml is missing")
    cfg = OmegaConf.create(text)
    # Descriptive columns first, so that an excluded run still shows its cell and seed.
    row.update(arm=arm_of(cfg), lr=cfg.optimizer.get("lr"), seed=cfg.get("seed"),
               protocol_version=cfg.get("protocol_version"))  # fmt: skip

    if row["status"] != "completed":
        raise _Excluded("not_completed", f"meta.json status is {row['status']!r}")
    run_type = cfg.get("run_type", "train")
    if run_type != "train":
        raise _Excluded("not_training", f"run_type is {run_type!r}")
    if row["arm"] not in ARMS:
        raise _Excluded("unknown_arm", f"arm {row['arm']!r} is not one of {ARMS}")
    summary = _json(files, run_dir / "summary.json", "no_summary")
    k = summary.get("K_harmful")
    if not (isinstance(k, int | float) and math.isfinite(k) and k > 0):
        raise _Excluded("no_k", f"K_harmful in summary.json is {k!r}")
    row["K"] = float(k)

    prompt_set = cfg.eval.harmful.prompt_set
    responses = _jsonl(files, run_dir / "responses" / f"{prompt_set}.jsonl", "no_responses",
                       ("condition", "item_id"))  # fmt: skip
    items = [label_item_id(r["condition"], r["item_id"]) for r in responses]
    trained = [label_item_id(TRAINED, r["item_id"]) for r in responses
               if r["condition"] == TRAINED]  # fmt: skip
    if not trained:
        raise _Excluded("no_responses", f"no responses of the trained model to {prompt_set}")
    if len(set(items)) != len(items):
        raise _Excluded("no_responses", f"a prompt of {prompt_set} is answered twice")

    records = _jsonl(files, label_path(run_dir, judge), "no_labels", ("item_id", "label"))
    _check_identity(records, identity)
    labels: dict[str, set[str]] = {}
    for record in records:
        if record["label"] is not None:
            labels.setdefault(record["item_id"], set()).add(record["label"])
    unlabeled = [item for item in trained if item not in labels]
    if unlabeled:
        raise _Excluded(
            "incomplete_labels",
            f"{len(unlabeled)} of {len(trained)} responses of the trained model have no label "
            f"(first: {unlabeled[0]})",
        )
    stray = sorted({record["item_id"] for record in records} - set(items))
    if stray:
        raise _Excluded("invalid_labels", f"{len(stray)} label records name no response "
                                          f"(first: {stray[0]})")  # fmt: skip
    conflicting = sorted(item for item in trained if len(labels[item]) > 1)
    if conflicting:
        raise _Excluded("invalid_labels", f"{len(conflicting)} responses have two labels "
                                          f"(first: {conflicting[0]})")  # fmt: skip
    unknown = sorted({label for item in trained for label in labels[item]} - set(LABELS))
    if unknown:
        raise _Excluded("invalid_labels", f"labels {unknown} are not in {LABELS}")

    row["judged"] = len(trained)
    row["harmful"] = sum(labels[item] == {HARMFUL} for item in trained)
    return cfg


def _check_identity(records: list[dict[str, Any]], identity: Mapping[str, Any]) -> None:
    wanted = tuple(identity[key] for key in IDENTITY_FIELDS)
    others = Counter(
        tuple(record.get(key) for key in IDENTITY_FIELDS)
        for record in records
        if tuple(record.get(key) for key in IDENTITY_FIELDS) != wanted
    )
    if others:
        names = ", ".join(f"{b}/{m}@{r}" for b, _, m, r, _ in sorted(others, key=repr))
        raise _Excluded(
            "judge_identity",
            f"{others.total()} of {len(records)} label records come from another judge ({names})",
        )


def _json(files: _Files, path: Path, reason: str) -> dict[str, Any]:
    text = files.text(path)
    if text is None:
        raise _Excluded(reason, f"{path.name} is missing")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _Excluded(reason, f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _Excluded(reason, f"{path.name} does not hold an object")
    return data


def _jsonl(files: _Files, path: Path, reason: str, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """The records of a JSONL file; each must be an object with ``keys``."""
    text = files.text(path)
    name = path.relative_to(path.parents[1]).as_posix()
    if text is None:
        raise _Excluded(reason, f"{name} is missing")
    records = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _Excluded(reason, f"{name}, line {number}, is not valid JSON") from exc
        if not (isinstance(record, dict) and all(key in record for key in keys)):
            raise _Excluded(reason, f"{name}, line {number}, is not an object with {keys}")
        records.append(record)
    return records


def _check_one_analysis(included: pd.DataFrame, configs: Mapping[str, DictConfig]) -> None:
    """Raise unless the included runs form one analysis (see the module docstring)."""
    run_ids = included["run_id"].tolist()
    conditions = {run_id: _condition(configs[run_id]) for run_id in run_ids}
    differing = _differing_keys(conditions.values())
    if differing:
        raise ValueError(
            f"The included runs differ in {differing}: one analysis needs one condition, so "
            "runs of other conditions belong to another experiment"
        )
    for arm, group in included.groupby("arm", sort=True):
        optimizers = [_optimizer(configs[run_id]) for run_id in group["run_id"]]
        differing = _differing_keys(optimizers)
        if differing:
            raise ValueError(f"Runs of arm {arm!r} differ in optimizer settings {differing}")
    cells = included.groupby(["arm", "lr", "seed"], sort=True)["run_id"]
    for (arm, lr, seed), group in cells:
        if len(group) > 1:
            raise ValueError(
                f"More than one run of seed {seed} in cell ({arm}, {lr:g}): {sorted(group)}"
            )
    for seed, group in included.groupby("seed", sort=True):
        streams = group["training_stream"]
        if streams.isna().any() or streams.nunique() != 1:
            recorded = dict(zip(group["run_id"], streams, strict=True))
            raise ValueError(
                f"The runs of seed {seed} do not share one recorded training stream (meta.json "
                f"inputs.streams.training.sha256): {recorded}"
            )


def _condition(cfg: DictConfig) -> dict[str, Any]:
    data = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(data, dict)
    for key in (*NON_IDENTITY_KEYS, "optimizer", "seed"):
        data.pop(key, None)
    return data


def _optimizer(cfg: DictConfig) -> dict[str, Any]:
    data = OmegaConf.to_container(cfg.optimizer, resolve=True)
    assert isinstance(data, dict)
    data.pop("lr", None)
    return data


def _differing_keys(configs: Iterable[Mapping[str, Any]]) -> list[str]:
    """Dotted keys whose values are not the same in every config."""
    flat = [_flatten(config) for config in configs]
    keys = sorted({key for f in flat for key in f})
    return [key for key in keys if len({json.dumps(f.get(key), sort_keys=True) for f in flat}) > 1]


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping) and value:
        out: dict[str, Any] = {}
        for key, item in value.items():
            out |= _flatten(item, f"{prefix}{key}.")
        return out
    return {prefix.removesuffix("."): value}
