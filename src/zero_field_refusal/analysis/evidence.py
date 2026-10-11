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

Exclusion reasons, checked in this order (the first that applies is recorded). A file that is
missing, not UTF-8, not valid JSON or YAML, or lacks a required field excludes only its run:
- ``no_meta``: no readable ``meta.json`` object;
- ``not_completed``: ``meta.json`` status other than ``completed`` (failed or still running);
- ``no_config``: no ``config.yaml``; ``invalid_config``: a ``config.yaml`` that is not a YAML
  mapping or lacks one of ``CONFIG_KEYS`` with its type;
- ``not_training``: ``run_type`` other than ``train`` (e.g. a θ₀ evaluation);
- ``unknown_arm``: an arm outside ``runs.ARMS``, such as a control run;
- ``no_summary``: no readable ``summary.json`` object; ``no_k``: no positive, finite ``K_harmful``;
- ``no_responses``: no readable responses of the trained model, or one prompt answered twice;
- ``no_labels``: no readable label file;
- ``judge_identity``: a label record from another judge identity;
- ``incomplete_labels``: a response of the trained model without a label;
- ``invalid_labels``: an item with two different labels, a label outside ``judge.LABELS``, or a
  record that names no response of the run.

The included runs must form one analysis; otherwise ``collect_runs`` raises instead of choosing:
- one condition: every key of ``config.yaml`` matches across the included runs, except the
  ``optimizer`` section, ``seed`` and the keys outside the config hash (``config.
  NON_IDENTITY_KEYS``: experiment, name, tracking, execution, checks). Model, protocol version,
  control, data, teacher, training and evaluation settings must all match, evaluation-only ones
  such as ``eval.controls`` included;
- one optimizer section per arm apart from ``optimizer.lr``;
- at most one run per seed in a cell (arm, lr);
- shared-teacher pairing: every run of a seed records the same training stream (``meta.json``
  ``inputs.streams.training.sha256``), and different seeds record different streams, since the
  bootstrap treats seeds as independent clusters.
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
# Keys every included run's config.yaml must hold, with their types.
CONFIG_KEYS = {"seed": int, "optimizer.name": str, "optimizer.lr": int | float,
               "eval.harmful.prompt_set": str}  # fmt: skip


def label_item_id(condition: str, prompt_id: str) -> str:
    """The ``item_id`` of the judge label of one response: ``<condition>/<prompt id>``."""
    return f"{condition}/{prompt_id}"


def label_path(run_dir: Path, judge: str) -> Path:
    """Where the labels of the judge named ``judge`` live in a run directory."""
    return run_dir / JUDGE_DIR / f"{judge}.jsonl"


def arm_of(cfg: DictConfig) -> str:
    """The arm of a run config: the optimizer, then ``+<control>`` for a control run."""
    arm = str(cfg.optimizer.name)
    momentum = _value(cfg, "optimizer.momentum", int | float)
    if arm == "sgd" and momentum is not None and momentum > 0:
        arm = "sgd_momentum"
    control = _select(cfg, "control")
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

    def text(self, path: Path, reason: str) -> str | None:
        """The file's text, None if it is missing; bytes that are not UTF-8 exclude the run."""
        if not path.is_file():
            return None
        data = path.read_bytes()
        relative = path.relative_to(self.root)
        self.sha256[relative.as_posix()] = hashlib.sha256(data).hexdigest()
        try:
            return data.decode()
        except UnicodeDecodeError as exc:
            in_run = Path(*relative.parts[2:]).as_posix()  # below <experiment>/<run id>/
            raise _Excluded(reason, f"{in_run} is not UTF-8 text") from exc


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
    row["training_stream"] = _nested(meta, "inputs", "streams", "training", "sha256")
    cfg, problem = _config(files, run_dir)
    if cfg is not None:  # descriptive columns, so that an excluded run still shows its cell
        named = _value(cfg, "optimizer.name", str) is not None
        row.update(arm=arm_of(cfg) if named else None,
                   lr=_value(cfg, "optimizer.lr", int | float), seed=_value(cfg, "seed", int),
                   protocol_version=_value(cfg, "protocol_version", int))  # fmt: skip

    if row["status"] != "completed":
        raise _Excluded("not_completed", f"meta.json status is {row['status']!r}")
    if problem is not None:
        raise problem
    assert cfg is not None
    run_type = _select(cfg, "run_type", "train")
    if run_type != "train":
        raise _Excluded("not_training", f"run_type is {run_type!r}")
    if row["arm"] not in ARMS:
        raise _Excluded("unknown_arm", f"arm {row['arm']!r} is not one of {ARMS}")
    summary = _json(files, run_dir / "summary.json", "no_summary")
    k = summary.get("K_harmful")
    if isinstance(k, bool) or not (isinstance(k, int | float) and math.isfinite(k) and k > 0):
        raise _Excluded("no_k", f"K_harmful in summary.json is {k!r}")
    row["K"] = float(k)

    prompt_set = _select(cfg, "eval.harmful.prompt_set")
    responses = _jsonl(files, run_dir / "responses" / f"{prompt_set}.jsonl", "no_responses",
                       {"condition": str, "item_id": str})  # fmt: skip
    items = [label_item_id(r["condition"], r["item_id"]) for r in responses]
    trained = [label_item_id(TRAINED, r["item_id"]) for r in responses
               if r["condition"] == TRAINED]  # fmt: skip
    if not trained:
        raise _Excluded("no_responses", f"no responses of the trained model to {prompt_set}")
    if len(set(items)) != len(items):
        raise _Excluded("no_responses", f"a prompt of {prompt_set} is answered twice")

    records = _jsonl(files, label_path(run_dir, judge), "no_labels",
                     {"item_id": str, "label": str | None})  # fmt: skip
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


def _config(files: _Files, run_dir: Path) -> tuple[DictConfig | None, _Excluded | None]:
    """``config.yaml`` if it parses, and the exclusion it causes (None: complete)."""
    try:
        text = files.text(run_dir / "config.yaml", "invalid_config")
    except _Excluded as excluded:
        return None, excluded
    if text is None:
        return None, _Excluded("no_config", "config.yaml is missing")
    try:
        cfg = OmegaConf.create(text)
    except Exception as exc:  # YAML syntax or OmegaConf errors
        return None, _Excluded(
            "invalid_config", f"config.yaml does not parse ({type(exc).__name__})"
        )
    if not isinstance(cfg, DictConfig):
        return None, _Excluded("invalid_config", "config.yaml does not hold a mapping")
    bad = [key for key, kind in CONFIG_KEYS.items() if _value(cfg, key, kind) is None]
    if bad:
        return cfg, _Excluded("invalid_config", f"config.yaml lacks {bad} or has a wrong type")
    return cfg, None


def _select(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """A dotted key's value; ``default`` if it is missing, ``???`` or cannot be resolved."""
    return OmegaConf.select(cfg, key, default=default, throw_on_resolution_failure=False)


def _value(cfg: DictConfig, key: str, kind: Any) -> Any:
    """A dotted key's value if it has type ``kind`` (never a bool), otherwise None."""
    value = _select(cfg, key)
    return value if isinstance(value, kind) and not isinstance(value, bool) else None


def _nested(data: Any, *keys: str) -> Any:
    """``data[k1][k2]...``, None where a level is missing or not an object."""
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def _check_identity(records: list[dict[str, Any]], identity: Mapping[str, Any]) -> None:
    def of(record: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(record.get(key) for key in IDENTITY_FIELDS)

    wanted = of(identity)
    others = Counter(of(record) for record in records if of(record) != wanted)
    if others:
        judges = [dict(zip(IDENTITY_FIELDS, other, strict=True)) for other in others]
        names = sorted(f"{j['backend']}/{j['judge_model']}@{j['judge_revision']} "
                       f"({j['prompt_version']})" for j in judges)  # fmt: skip
        raise _Excluded(
            "judge_identity",
            f"{others.total()} of {len(records)} label records come from another judge "
            f"({', '.join(names)})",
        )


def _json(files: _Files, path: Path, reason: str) -> dict[str, Any]:
    text = files.text(path, reason)
    if text is None:
        raise _Excluded(reason, f"{path.name} is missing")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _Excluded(reason, f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _Excluded(reason, f"{path.name} does not hold an object")
    return data


def _jsonl(
    files: _Files, path: Path, reason: str, fields: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """The records of a JSONL file; each must be an object with ``fields`` of their types."""
    text = files.text(path, reason)
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
        valid = isinstance(record, dict) and all(
            key in record and isinstance(record[key], kind) for key, kind in fields.items()
        )
        if not valid:
            raise _Excluded(reason, f"{name}, line {number}: needs {sorted(fields)} with their "
                                    "types")  # fmt: skip
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
    # Different seeds must train on different streams: the bootstrap resamples seeds as
    # independent clusters.
    stream_of_seed = included.groupby("seed", sort=True)["training_stream"].first()
    shared = stream_of_seed[stream_of_seed.duplicated(keep=False)]
    if not shared.empty:
        raise ValueError(
            f"Seeds {[int(s) for s in shared.index]} record the same training stream "
            f"{sorted(set(shared))}: seeds are resampled as independent clusters"
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
