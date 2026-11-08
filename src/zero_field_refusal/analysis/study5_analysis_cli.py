"""Study 5's analysis on a CPU: the retrospective predictor before the registration, and after
the Pods the confirmatory test and the descriptive measurements of docs/study5-preregistration.md,
from the campaigns' records and the label files, each written once as one record (the
computations are ``analysis.study5_analysis``'s).

    python -m zero_field_refusal.analysis.study5_analysis_cli retrospective --runs-root DIR
        --pod1 DIR --out FILE --study3-runs-root DIR --study3-repo DIR [--study3-sealed-dir DIR]
        [--fixation docs/study5-fixation.json]
    python -m zero_field_refusal.analysis.study5_analysis_cli full --runs-root DIR --pod1 DIR
        --pod2 DIR --out FILE [--fixation docs/study5-fixation.json]
        (--study3-runs-root DIR --study3-repo DIR [--study3-sealed-dir DIR] --retrospective FILE
         | --without-study3)
        [--reference-input DIR --reference NAME=FILE [--reference NAME=FILE]]

**retrospective** (registration order step 2, before the registration; no pod2 input) writes the
retrospective record (kind ``study5-retrospective``): ρ(p_s, F's harm at K*) over study 3's 60
A–F blocks with the fixation record's block bootstrap interval, ρ(p_s, F's refusal share at K*),
the stream features against the harm, the per-block inputs, and every input's sha256. **full**
recomputes the retrospective from the same inputs and refuses unless it agrees with that
record's ``retrospective`` (``--retrospective``, named by its sha256 in the analysis, with the
largest numeric deviation): the same structure, ids, seeds, counts and states, numbers within a
relative 1e-12 (1e-15 near zero), reasons not compared (``same_tree``). The registered
retrospective result is the one the analysis reports. Both run from a
clean checkout of a known commit (its commit in the record).

It sits in the analysis package, outside the GPU code identity, so that adding it changes no GPU
path (``code_identity``: every file under ``scripts/`` not named in ``ANALYSIS_SCRIPTS`` is one).

**Inputs.** --runs-root is the local runs root holding both Pods' archives as the campaigns wrote
them (``study5/<ID>/runs/`` of each restored under it: the run directories, ``launch/`` and
``study5_records/<ID>/``), with the label files the labeling wrote into the run directories.
--pod1 and --pod2 are the two campaign folders (``plan.json`` and ``campaigns/*.yaml``, each
campaign's sha256 the plan's). From the plans: pod2's n and seeds (the fixation record's fresh
seeds, or its n = 24 alternative), each fresh seed's F runs at η_l and η_u (experiments/study5,
``study5.role=fresh``), the controls (study5_control_order, study5_control_v0), θ₀'s
evaluation (pod1, experiments/study5_theta0), and the records each step wrote (``predictions/
record``, ``predictions/publish``, ``streams/s<seed>/features``, ``fresh/s<seed>/surrogate-
endpoint``, ``fresh/s<seed>/scan``, ``controls/s<seed>/ladder-<rung>``; pod1's
``retrospective/s<seed>/predict`` and ``features``), their Pod paths read relative to the plan's
``runs``. A seed's line scan is its scan run (the record's experiment and run id, the record of
the seed's η_l run): h(+1) is the η_l run's step-25 evaluation, h(0) θ₀'s run, each other point
its row of the scan run (``line_scan.jsonl``) and its labels; a point not evaluated, a scan run
missing or not labeled leaves those points missing by name, the seed in every other quantity. A
run is its launcher campaign's last end of its label (``launch/<campaign>.jsonl``) with the
outcome ``done`` or ``diverged`` (its records then decide), its folder the start's experiment and
run id; its config must give the cell's seed, rate and ``adam_frozen``. The prompt sets come
from ``DATA_DIR``.

**Labels.** A study-5 run's labels are read as study 2 reads a run (``study2_labels``' checks,
composed as ``study3_labels._read_run`` composes them: the recorded prompts, exactly the
responses expected, the judging receipt of this judge for exactly the run's items, the label file
by its sha256, every record checked, study 2's rounds rule) over the conditions
``run_judging.response_conditions`` gives the run: ``step-25`` for a fresh or control F run, the
points ``line_scan_{a:+g}`` for a scan run, ``ladder_<rung>`` for a rung's run, ``theta0`` for θ₀.
``study2_labels.read_endpoint_labels`` itself reads ``scheduled_conditions`` alone and would refuse
a scan run's, a ladder rung's or θ₀'s. WildGuard alone gives the outcome; per condition the counts
of the three classes (refusal, safe non-refusal, harmful compliance) and the missing labels, of
521.

**Study 3's parts** (the retrospective and the extreme blocks) are read as the fixation record
read them: study 3's analysis freeze of candidate 1 (``study3_freeze.read_freeze``, no problem),
the freeze's sha256 the record's, --study3-repo a clean checkout whose GPU code is the freeze's
(study 3's label gate opens labels under that code only); the harm of every F run of the 60 A–F
blocks and of the 40 relabeled runs through study 3's final reader (``study3_final.cohort_harm``,
one gated reading), each receipt's sha256 the record's. Only after that reading, and bound to its
accepted receipt (sha256 and counts), a run's three classes are read with the study-5 reader
above (study 3's reader gives harmful or not alone), and Llama Guard's labels of the 40 relabeled
runs (``llama_guard``, two classes). Study 3's gate is not changed or bypassed: nothing of a study-3
run is read before ``cohort_harm`` opened it. --without-study3 leaves these parts out, said so in
the record.

**The reference sample.** --reference-input is the reference annotators' input folder
(``scripts/study5_fixation.py reference-input``): its ``manifest.json`` names this fixation record
by sha256 and its ``key.json`` by sha256; the key maps each item id to its cell. Each
--reference NAME=FILE is one annotator's labels in study 3's audit-label form, ``{item_id: label
| null}`` over exactly the key's items; with two, each is reported and their agreement.

**The record.** ``analysis.json`` (--out, refused when it exists, never overwritten; written whole
by a link): the fixation record's sha256, the code, the seeds, the predictions record (its sha256
that of the receipt and the sidecar; its bucket copy's creation time beside the first run start
in pod2's launcher journals, and whether it is earlier), ``confirmatory`` (with the excluded seeds
counted by state), ``outcomes`` (per fresh seed p, support,
h, the three classes or the reason), ``retrospective``, ``descriptive`` (the refusal share at K*,
the surrogate at the endpoints, the line scans, the controls, the ladder, θ₀'s baseline, the
stream features and the extreme blocks) and ``inputs``, every file read by its root and relative
path with its sha256 (a run's ``summary.json`` and the other files ``response_conditions`` reads
among them; study 3's label files are bound through the receipts the fixation record names).
Strict JSON, sorted keys, a non-finite number copied from a record as ``{"missing":
"nonfinite"}``. It holds numbers, ids, states and the registered words, never a text. stdout
carries numbers, counts and the record's sha256 only. The ladder's rungs are every
``controls/s<seed>/ladder-<rung>`` step of pod2's plan.

**Refusals** exit with ``study5_analysis_cli: <why>`` and write nothing: a dirty checkout, a
retrospective record that is not its recomputation, a plan or campaign that
does not read or is not its plan's, a predictions record that is not the published one, a study-3
input that is not the fixation record's, study 3's labels pending, a reference input of another
record. A missing run, record or label file of one seed is no refusal: it is named in each
quantity it leaves, and the seed counts nowhere there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

import yaml
from omegaconf import DictConfig, OmegaConf

from .. import code_identity
from ..config import load_config
from ..evaluation import ladder_condition, line_scan_condition
from ..judge import judge_identity
from ..paths import REPO_ROOT
from ..run_judging import (
    LABEL_ROUNDS,
    expected_items,
    prompt_ids,
    receipt_path,
    response_conditions,
)
from ..tracking import git_state
from . import study2_labels
from . import study5_analysis as sa
from .evidence import HARMFUL, label_item_id, label_path
from .study2_label_cli import harmful_prompts

PROG = "study5_analysis_cli"
KIND, VERSION = "study5-analysis", 1
FIXATION = "docs/study5-fixation.json"
ETA_L, ETA_U = 1.3e-05, 1.83e-05  # study 3's candidate-1 rates of F (scripts/study5_campaigns.py)
ENDPOINT = "step-25"
FRESH, FRESH_ROLE = "experiments/study5", "study5.role=fresh"  # the fresh F runs' cells
THETA0 = "theta0"
DONE = ("done", "diverged")  # a launcher end after which the run's records decide
WILDGUARD, LLAMA_GUARD = "wildguard", "llama_guard"
CANDIDATE = 1
OPTIMIZER = "adam_frozen"
RETRO_KIND = "study5-retrospective"
REL_TOLERANCE, ABS_TOLERANCE = 1e-12, 1e-15  # a recomputed retrospective number against its record
FREE_TEXT = ("reason", "missing")  # keys whose strings are reasons: not compared
LADDER_STEP = re.compile(r"controls/s(?P<seed>[0-9]+)/ladder-(?P<rung>[a-z_]+)")
# the files run_judging.response_conditions reads of a run (scheduled, line scan, ladder)
CONDITION_FILES = ("evaluations.jsonl", "terminal.json", "line_scan.jsonl", "summary.json")


class Refused(Exception):
    """The analysis cannot stand: nothing is written."""


def refuse(why: str) -> NoReturn:
    raise Refused(why)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(record: Any) -> bytes:
    return (json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


# ------------------------------------------------------------------ reading, hashed


class Reader:
    """Every file read, by ``<root name>:<path relative to the root>``, with its sha256."""

    def __init__(self, roots: Mapping[str, Path]) -> None:
        self.given = {name: Path(root) for name, root in roots.items()}
        self.roots = {name: Path(root).resolve() for name, root in roots.items()}
        self.hashes: dict[str, str] = {}
        self.prompt_sets: dict[str, str] = {}  # a prompt set's name -> sha256 of its ids and texts

    def scrub(self, text: str) -> str:
        """``text`` with every root's local path replaced by its name (``<runs>``): the root's
        absolute forms only (resolved, and absolute as given), each matched at path boundaries
        (``/tmp/runs`` not inside ``/x/tmp/runs`` or ``/tmp/runs2``), the longest first. A root as
        given relative on the command line is never matched: it could rewrite unrelated text."""
        forms = {(str(form), name) for name in self.roots
                 for form in (self.roots[name], self.given[name].absolute())
                 if form.is_absolute() and str(form) != "/"}  # fmt: skip
        for form, name in sorted(forms, key=lambda f: -len(f[0])):
            pattern = r"(?<![\w./-])" + re.escape(form) + r"(?![\w.-])"
            text = re.sub(pattern, f"<{name}>", text)
        return text

    def name(self, path: Path) -> str:
        found = Path(path).resolve()
        for name, root in self.roots.items():
            if found.is_relative_to(root):
                return f"{name}:{found.relative_to(root).as_posix()}"
        return f"other:{found.name}"

    def bytes(self, path: Path) -> bytes:
        data = Path(path).read_bytes()
        self.hashes[self.name(path)] = sha256(data)
        return data

    def json(self, path: Path) -> Any:
        return json.loads(self.bytes(path))

    def jsonl(self, path: Path) -> list[Any]:
        return [json.loads(line) for line in self.bytes(path).decode().splitlines()
                if line.strip()]  # fmt: skip


def _why(reader: Reader, exc: Exception) -> str:
    """A reason from an error: its type and message, local paths replaced by their roots' names
    (the readers' messages name ids, counts and paths, never a label or a text)."""
    return reader.scrub(f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------ labels of a run


def run_labels(
    reader: Reader,
    run_dir: Path,
    judge: str,
    identity: Mapping[str, Any],
    prompts_of: Callable[[DictConfig], Mapping[str, str]],
) -> dict[str, Any]:
    """The label counts of every condition of a run (module docstring, "Labels"): per condition
    ``n``, ``labeled``, ``missing``, ``harmful`` and, for a three-class judge, ``refusal`` and
    ``safe_non_refusal`` (``not_harmful`` for ``llama_guard``); and the receipt's sha256
    (``receipt_sha256``). ``study2_labels.LabelsPending`` or ``ValueError`` otherwise."""
    run_dir = Path(run_dir)
    read = reader.bytes
    cfg = OmegaConf.create(read(run_dir / "config.yaml").decode())
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"{run_dir.name}: its config does not read")
    conditions = response_conditions(cfg, run_dir)
    for name in CONDITION_FILES:  # what response_conditions read, hashed as an input
        if (run_dir / name).is_file():
            read(run_dir / name)
    if not conditions:
        raise study2_labels.LabelsPending(f"{run_dir.name}: no condition answered")
    prompts = prompts_of(cfg)
    ids = prompt_ids(cfg, prompts)
    if len(ids) != sa.DENOMINATOR:
        raise ValueError(f"{run_dir.name}: {len(ids)} harmful prompts, not {sa.DENOMINATOR}")
    study2_labels._check_recorded_prompts(run_dir, cfg, ids, read(run_dir / "meta.json"))
    texts = study2_labels._responses(run_dir, cfg, conditions, ids, read)
    items = {label_item_id(c, pid): (prompts[pid], text) for (c, pid), text in texts.items()}
    del texts
    expected = expected_items(conditions, ids)
    receipt = study2_labels._receipt(run_dir, judge, identity, expected, read)
    receipt_sha = reader.hashes[reader.name(receipt_path(run_dir, judge))]
    path = label_path(run_dir, judge)
    if not path.is_file():
        raise study2_labels.LabelsPending(f"{run_dir.name}: its {judge} label file is missing")
    data = read(path)
    if sha256(data) != receipt["labels_sha256"]:
        raise study2_labels.LabelsPending(f"{run_dir.name}: its {judge} label file is not the "
                                          "one its receipt was written for")  # fmt: skip
    labels, recorded = study2_labels._labels(path, data, identity, items)
    del items
    unrecorded = [item for item in expected if item not in recorded]
    if unrecorded:
        raise study2_labels.LabelsPending(f"{run_dir.name}: {len(unrecorded)} expected items have "
                                          f"no record in its {judge} label file")  # fmt: skip
    missing = sum(1 for item in expected if item not in labels)
    if receipt["missing"] != missing:
        raise study2_labels.LabelsPending(f"{run_dir.name}: its receipt's count of items without a "
                                          "label is not its label file's")  # fmt: skip
    if missing and receipt["rounds"] < LABEL_ROUNDS:
        raise study2_labels.LabelsPending(f"{run_dir.name}: items without a label after "
                                          f"{receipt['rounds']} labeling rounds of "
                                          f"{LABEL_ROUNDS}")  # fmt: skip
    out: dict[str, Any] = {"receipt_sha256": receipt_sha, "conditions": {}}
    for condition in conditions:
        count = Counter(labels.get(label_item_id(condition, pid)) for pid in ids)
        labeled = len(ids) - count[None]
        row = {"n": len(ids), "labeled": labeled, "missing": count[None],
               "harmful": count[HARMFUL]}  # fmt: skip
        if identity.get("backend") == "llama_guard":
            row["not_harmful"] = labeled - count[HARMFUL]
        else:
            row |= {"refusal": count["refusal"], "safe_non_refusal": count["safe_non_refusal"]}
        out["conditions"][condition] = row
    return out


class Labels:
    """The WildGuard labels of study-5 runs, each run read once, a failure kept as its reason."""

    def __init__(self, reader: Reader, prompts_of, judge: str = WILDGUARD) -> None:
        self.reader, self.prompts_of, self.judge = reader, prompts_of, judge
        self.identity = judge_identity(load_config(f"judge/{judge}"))
        self.found: dict[Path, dict[str, Any] | str] = {}

    def of(self, run_dir: Path) -> dict[str, Any] | str:
        if run_dir not in self.found:
            try:
                self.found[run_dir] = run_labels(self.reader, run_dir, self.judge, self.identity,
                                                 self.prompts_of)  # fmt: skip
            except (OSError, ValueError, KeyError) as exc:
                self.found[run_dir] = f"labels: {_why(self.reader, exc)}"
        return self.found[run_dir]

    def at(self, run_dir: Path | str, condition: str) -> dict[str, int] | str:
        """A run's counts at one condition, or the reason they are missing."""
        if isinstance(run_dir, str):
            return run_dir
        found = self.of(run_dir)
        if isinstance(found, str):
            return found
        return found["conditions"].get(condition, f"labels: no condition {condition}")


def prompt_sets(reader: Reader) -> Callable[[DictConfig], Mapping[str, str]]:
    """The run's harmful prompt texts (``DATA_DIR``), each set loaded once, its sha256 (of its ids
    and texts, in order) kept as an input; never printed."""
    cache: dict[str, Mapping[str, str]] = {}

    def of(cfg: DictConfig) -> Mapping[str, str]:
        name = str(cfg.eval.harmful.prompt_set)
        if name not in cache:
            cache[name] = harmful_prompts(cfg)
            blob = json.dumps(list(cache[name].items()), ensure_ascii=False).encode("utf-8")
            reader.prompt_sets[name] = sha256(blob)
        return cache[name]

    return of


# ------------------------------------------------------------------ the plans


class Plan:
    """A campaign folder's plan: its steps' outputs read relative to its runs root, its launcher
    campaigns' cells (each campaign the plan's by sha256)."""

    def __init__(self, reader: Reader, folder: Path, runs: Path) -> None:
        self.folder, self.runs = Path(folder), Path(runs)
        try:
            self.data = reader.json(self.folder / "plan.json")
        except (OSError, ValueError) as exc:
            refuse(f"{self.folder.name}/plan.json does not read ({type(exc).__name__})")
        self.pod_runs = PurePosixPath(self.data["runs"])
        self.steps = {s["name"]: s for b in self.data["blocks"] for s in b["steps"]}
        self.cells: list[tuple[str, dict[str, Any]]] = []
        for name, digest in sorted(self.data["campaigns"].items()):
            path = self.folder / "campaigns" / f"{name}.yaml"
            try:
                data = reader.bytes(path)
            except OSError:
                refuse(f"{self.folder.name}: the launcher campaign {name} is missing")
            if sha256(data) != digest:
                refuse(f"{self.folder.name}: the launcher campaign {name} is not the plan's")
            for cell in yaml.safe_load(data)["cells"]:
                self.cells.append((name, cell))

    def output(self, step: str) -> Path | None:
        """A step's first output, under the local runs root; None for a step not in the plan."""
        found = self.steps.get(step)
        if not found or not found.get("outputs"):
            return None
        rel = PurePosixPath(found["outputs"][0]).relative_to(self.pod_runs)
        return self.runs / Path(*rel.parts)

    def find(self, config: str, seed: int, lr: float | None = None,
             overrides: Sequence[str] = ()) -> tuple[str, str] | None:  # fmt: skip
        """The (campaign, run label) of the cell of ``config`` and ``seed`` (and the rate and
        overrides given)."""
        for name, cell in self.cells:
            given = cell.get("overrides") or []
            if (cell.get("config") == config and cell.get("seeds") == [seed]
                    and (lr is None or f"optimizer.lr={lr!r}" in given)
                    and all(o in given for o in overrides)):  # fmt: skip
                return name, f"{cell['name']}/s{seed}"
        return None


def journal(reader: Reader, runs: Path, campaign: str) -> list[dict[str, Any]]:
    path = runs / "launch" / f"{campaign}.jsonl"
    if not path.is_file():
        return []
    out = []
    for line in reader.bytes(path).decode().splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict):
            out.append(entry)
    return out


def completed(reader: Reader, runs: Path, found: tuple[str, str] | None,
              what: str) -> Path | str:  # fmt: skip
    """The run directory of a cell's run (module docstring), or why there is none."""
    if found is None:
        return f"{what}: no cell in the plan"
    campaign, label = found
    events = journal(reader, runs, campaign)
    ends = [e for e in events if e.get("event") == "end" and e.get("run") == label]
    if not ends or ends[-1].get("outcome") not in DONE:
        outcome = ends[-1].get("outcome") if ends else "no end"
        return f"{what}: no completed run in {campaign} ({outcome})"
    run_id = ends[-1].get("run_id")
    starts = [e for e in events if e.get("event") == "start" and e.get("run_id") == run_id]
    if not starts or not isinstance(starts[-1].get("experiment"), str):
        return f"{what}: no start of {run_id} in {campaign}"
    run_dir = runs / starts[-1]["experiment"] / str(run_id)
    return run_dir if run_dir.is_dir() else f"{what}: the run directory {run_id} is missing"


# ------------------------------------------------------------------ a run's records


def f_run(reader: Reader, run_dir: Path | str, seed: int, lr: float) -> dict[str, Any] | str:
    """An F run's records: its stop and its step-25 evaluation row; its config the cell's seed,
    rate and optimizer. The reason when they do not read."""
    if isinstance(run_dir, str):
        return run_dir
    try:
        cfg = OmegaConf.create(reader.bytes(run_dir / "config.yaml").decode())
        if (int(cfg.seed) != seed or not math.isclose(float(cfg.optimizer.lr), lr, rel_tol=1e-9)
                or cfg.optimizer.name != OPTIMIZER):  # fmt: skip
            return f"{run_dir.name}: its config is not seed {seed}, rate {lr!r}, {OPTIMIZER}"
        terminal = reader.json(run_dir / "terminal.json")
        rows = (reader.jsonl(run_dir / "evaluations.jsonl")
                if (run_dir / "evaluations.jsonl").is_file() else [])  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"{Path(run_dir).name}: its records do not read ({type(exc).__name__})"
    endpoint = [r for r in rows if r.get("condition") == ENDPOINT]
    return {"run_dir": run_dir, "run_id": run_dir.name, "seed": seed, "lr": lr,
            "stop": (terminal.get("stop") or {}).get("reason"), "steps": terminal.get("steps"),
            "row": endpoint[0] if len(endpoint) == 1 else None}  # fmt: skip


def scan_rows(reader: Reader, run_dir: Path) -> dict[float, dict[str, Any]]:
    """A scan run's rows (``line_scan.run_scan``) by point a; ``OSError``/``ValueError`` when
    they do not read."""
    rows = reader.jsonl(Path(run_dir) / "line_scan.jsonl")
    return {float(r["a"]): r for r in rows if isinstance(r.get("a"), int | float)}


def scan_points(reader: Reader, labels: Labels, found: Mapping[str, Any] | str, runs: Path,
                lower_id: str) -> dict[float, Any]:  # fmt: skip
    """The scan's points of one seed, a ∈ ``SCAN_POINTS``: each point's values, or the reason it
    is missing, by name (``found``: the seed's scan record, or why there is none)."""
    out: dict[float, Any] = {}
    run_dir: Path | None = None
    why = found if isinstance(found, str) else None
    if why is None and (found.get("f_run") or {}).get("run_id") != lower_id:
        why = "scan: the record is not of the seed's η_l run"
    if why is None:
        run_dir = runs / str(found.get("experiment")) / str(found.get("run_id"))
        if not run_dir.is_dir():
            why = "scan: its run directory is missing"
    rows: dict[float, dict[str, Any]] = {}
    if why is None:
        try:
            rows = scan_rows(reader, run_dir)
        except (OSError, ValueError) as exc:
            why = f"scan: its rows do not read ({type(exc).__name__})"
    for a in sa.SCAN_POINTS:
        condition = line_scan_condition(a)
        row = rows.get(a)
        if why is not None:
            out[a] = f"{condition}: {why}"
        elif row is None:
            out[a] = f"{condition}: no scan row"
        elif not row.get("evaluated"):
            out[a] = f"{condition}: not evaluated ({row.get('reason')})"
        else:
            out[a] = sa.point_values(labels.at(run_dir, condition), row)
    return out


def record(reader: Reader, path: Path | None, kind: str, seed: int | None,
           what: str) -> dict[str, Any] | str:  # fmt: skip
    """A study-5 script's record of one seed, or why it is missing."""
    if path is None:
        return f"{what}: not in the plan"
    if not path.is_file():
        return f"{what}: no record ({path.name})"
    try:
        found = reader.json(path)
    except (OSError, ValueError) as exc:
        return f"{what}: {path.name} does not read ({type(exc).__name__})"
    if found.get("kind") != kind or (seed is not None and found.get("seed") != seed):
        return f"{what}: {path.name} is not the {kind} record of seed {seed}"
    return found


# ------------------------------------------------------------------ the predictions record


def predictions(reader: Reader, plan: Plan, seeds: Sequence[int]) -> tuple[dict, dict]:
    """The published predictions record: its sha256 the receipt's and the sidecar's, exactly the
    plan's seeds in order and n; p by seed and what binds it."""
    path, receipt_path_ = plan.output("predictions/record"), plan.output("predictions/publish")
    if path is None or receipt_path_ is None or not path.is_file() or not receipt_path_.is_file():
        refuse("the predictions record or its receipt is missing")
    try:
        data = reader.bytes(path)
        receipt = reader.json(receipt_path_)
        sidecar = reader.bytes(path.with_name(path.name + ".sha256")).decode()
    except (OSError, ValueError) as exc:
        refuse(f"the predictions record, its receipt or sidecar does not read "
               f"({type(exc).__name__})")  # fmt: skip
    digest = sha256(data)
    if receipt.get("sha256") != digest or sidecar != f"{digest}  {path.name}\n":
        refuse("the predictions record is not the published one (its receipt's and sidecar's "
               "sha256)")  # fmt: skip
    found = json.loads(data)
    if (found.get("kind") != "study5-predictions" or found.get("seeds") != list(seeds)
            or found.get("n") != len(seeds) or found.get("lr") != ETA_L
            or [e.get("seed") for e in found.get("predictions") or []] != list(seeds)):  # fmt: skip
        refuse("the predictions record is not of the plan's seeds, in order, at η_l")
    p = {e["seed"]: float(e["p"]) for e in found["predictions"]}
    return p, {"record": path.name, "sha256": digest, "receipt": receipt_path_.name,
               "created": receipt.get("created"), "n": found["n"]}  # fmt: skip


# ------------------------------------------------------------------ study 3's parts


def study3_parts(args: argparse.Namespace, fixation: Mapping[str, Any], reader: Reader,
                 prompts_of) -> dict[str, Any]:  # fmt: skip
    """Study 3's freeze, the harm of its F runs and relabeled runs, their three classes and Llama
    Guard's labels (module docstring, "Study 3's parts")."""
    from . import study3_final
    from .study3_freeze import freeze_name, read_freeze
    from .study3_labels import LabelingProblem, LedgerError, SealedPoint
    from .study3_records import EXPERIMENT

    root, repo = Path(args.study3_runs_root), Path(args.study3_repo)
    state = git_state(repo)
    if not state.get("commit") or state.get("dirty") is not False:
        refuse("study 3's checkout (--study3-repo) needs a clean checkout of a known commit")
    identity = code_identity.gpu_code_identity(state["commit"], repo)["sha256"]
    freeze, problems = read_freeze(CANDIDATE, runs_root=root, repo=repo)
    if freeze is None:
        refuse(f"study 3's analysis freeze has {len(problems)} problem(s): " + "; ".join(problems))
    if identity != freeze["code"]["gpu_code_sha256"]:
        refuse(f"study 3's checkout runs GPU code {identity[:12]}, not the freeze's: its readers "
               "open no label under another code")  # fmt: skip
    freeze_sha = sha256(reader.bytes(repo / freeze_name(CANDIDATE)))
    if freeze_sha != fixation["inputs"]["study3_freeze"]["sha256"]:
        refuse("study 3's freeze is not the one the fixation record names")
    blocks = freeze["comparisons"]["A-F"]["blocks"]
    f_runs = sorted({b["arms"]["F"][end] for b in blocks for end in ("lo", "hi")})
    relabel = list(fixation["extreme_blocks"]["relabel"]["runs"])
    runs = sorted(set(f_runs) | set(relabel))
    experiment = root / EXPERIMENT
    try:
        harm = study3_final.cohort_harm({run: experiment / run for run in runs},
                                        candidate=CANDIDATE, runs_root=root, repo=repo,
                                        sealed_dir=args.study3_sealed_dir)  # fmt: skip
    except (study3_final.Pending, SealedPoint) as exc:
        refuse(str(exc))
    except (LabelingProblem, LedgerError) as exc:
        refuse(f"study 3's labels are not read ({type(exc).__name__})")
    except (OSError, ValueError) as exc:
        refuse(f"study 3's harm cannot be read ({type(exc).__name__})")
    fixed = fixation["inputs"]["labels"]["runs"]
    other = [run for run in runs if fixed.get(run, {}).get("receipt_sha256")
             != harm[run].receipt_sha256]  # fmt: skip
    if other:
        refuse(f"{len(other)} of study 3's runs have another receipt than the fixation record's "
               f"(first: {other[0]})")  # fmt: skip
    # opened by the gate above; each bound to the receipt that reading accepted
    wildguard = Labels(reader, prompts_of)
    three: dict[str, dict[str, int] | str] = {}
    for run in f_runs:
        found = wildguard.of(experiment / run)
        if isinstance(found, str):
            three[run] = found
            continue
        counts = found["conditions"].get(ENDPOINT)
        bound = (found["receipt_sha256"] == harm[run].receipt_sha256 and counts is not None
                 and (counts["harmful"], counts["missing"]) == (harm[run].harmful,
                                                                harm[run].missing))  # fmt: skip
        three[run] = counts if bound else "labels: not the counts of the accepted receipt"
    llama = Labels(reader, prompts_of, LLAMA_GUARD)
    llama_shares: dict[str, sa.Value] = {}
    for run in relabel:
        counts = llama.at(experiment / run, ENDPOINT)
        llama_shares[run] = (counts["harmful"] / sa.DENOMINATOR if isinstance(counts, Mapping)
                             else f"{LLAMA_GUARD} {counts}")  # fmt: skip
    return {"blocks": blocks, "harm": harm, "three": three, "llama": llama_shares,
            "freeze_sha256": freeze_sha,
            "checkout": {"commit": state["commit"], "gpu_code_sha256": identity}}  # fmt: skip


def retrospective(parts: Mapping[str, Any], pod1: Plan, reader: Reader,
                  fixation: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """ρ(p_s, F's harm at K*) over study 3's 60 A–F blocks with the record's block bootstrap, and
    ρ(p_s, F's refusal share at K*), and the stream features against the harm; retrospective,
    no test."""
    blocks, harm, three = parts["blocks"], parts["harm"], parts["three"]
    seeds = [int(b["seed"]) for b in blocks]
    h: dict[int, sa.Value] = {}
    refusal: dict[int, sa.Value] = {}
    for block in blocks:
        seed = int(block["seed"])
        h[seed] = sa._arm(block, sa.ARM, harm)[0]
        runs = [block["arms"][sa.ARM][end] for end in ("lo", "hi")]
        absent = [f"{run}: {three[run]}" for run in runs if isinstance(three[run], str)]
        if absent:
            refusal[seed] = "; ".join(absent)
        else:
            counts = {
                run: sa.Counts(
                    n=three[run]["n"], harmful=three[run]["refusal"], missing=three[run]["missing"]
                )
                for run in runs
            }
            refusal[seed] = sa._arm(block, sa.ARM, counts)[0]
    p: dict[int, sa.Value] = {}
    features: dict[str, dict[int, sa.Value]] = {f: {} for f in sa.FEATURES}
    for seed in seeds:
        found = record(reader, pod1.output(f"retrospective/s{seed}/predict"),
                       "study5-prediction", seed, "p")  # fmt: skip
        if isinstance(found, str):
            p[seed] = found
        elif found.get("lr") != ETA_L:
            p[seed] = f"p: its rate {found.get('lr')!r} is not η_l"
        else:
            p[seed] = float(found["p"])
        found = record(reader, pod1.output(f"retrospective/s{seed}/features"),
                       "study5-stream-features", seed, "features")  # fmt: skip
        for name in sa.FEATURES:
            features[name][seed] = found if isinstance(found, str) else found["features"][name]
    harm_rho = sa.association(p, h, seeds, ("p", "h"))
    used, xs, ys, _ = sa.pair(p, h, seeds, ("p", "h"))
    interval = (sa.bootstrap_interval(xs, ys, fixation["retrospective_interval"])
                if len(used) >= 2 else None)  # fmt: skip
    return {
        "computed": True, "label": "retrospective: an association, not a test",
        "blocks": len(blocks), "freeze_sha256": parts["freeze_sha256"],
        "study3_checkout": parts["checkout"],
        "harm": {**harm_rho, "interval": interval},
        "refusal": sa.association(p, refusal, seeds, ("p", "refusal")),
        "stream_features": {name: sa.association(features[name], h, seeds, (name, "h"))
                            for name in sa.FEATURES},
        "per_block": [{"seed": s, "p": sa._json_value(p[s]), "h": sa._json_value(h[s]),
                       "refusal": sa._json_value(refusal[s])} for s in seeds],
    }  # fmt: skip


def extreme_blocks(parts: Mapping[str, Any] | None, fixation: Mapping[str, Any],
                   reference: Mapping[str, Any] | None) -> dict[str, Any]:  # fmt: skip
    """The extreme blocks' contrast by judge and in the reference sample (the record's rule)."""
    cells = fixation["reference_sample"]["cells_drawn"]
    out: dict[str, Any] = {"rule": fixation["extreme_blocks"]["contrast"]}
    if parts is None:
        out |= {"computed": False, "reason": "--without-study3"}
    else:
        harm = parts["harm"]
        wildguard = {c["run_id"]: harm[c["run_id"]].harmful / sa.DENOMINATOR for c in cells}
        out |= {"computed": True, "wildguard": sa.judge_contrast(cells, wildguard),
                "llama_guard": sa.judge_contrast(cells, parts["llama"])}  # fmt: skip
    out["reference"] = (
        reference
        if reference is not None
        else {"computed": False, "reason": "no reference labels given"}
    )
    return out


def reference_part(args: argparse.Namespace, fixation_data: bytes, fixation: Mapping[str, Any],
                   reader: Reader) -> dict[str, Any] | None:  # fmt: skip
    """The reference sample's estimator per annotator and their agreement."""
    if not args.reference:
        return None
    if args.reference_input is None:
        refuse("--reference needs --reference-input (the blind key)")
    folder = Path(args.reference_input)
    manifest = reader.json(folder / "manifest.json")
    key_data = reader.bytes(folder / "key.json")
    if (manifest.get("fixation") or {}).get("sha256") != sha256(fixation_data):
        refuse("the reference input is not of this fixation record")
    if (manifest.get("files") or {}).get("key.json", {}).get("sha256") != sha256(key_data):
        refuse("the reference input's key.json is not its manifest's")
    key = json.loads(key_data)
    cells = fixation["reference_sample"]["cells_drawn"]
    out: dict[str, Any] = {"computed": True, "annotators": {}}
    labels: dict[str, dict[str, Any]] = {}
    for given in args.reference:
        name, _, path = given.partition("=")
        if not name or not path or name in labels:
            refuse(f"--reference {given!r}: NAME=FILE, each name once")
        try:
            found = reader.json(Path(path))
        except (OSError, ValueError) as exc:
            refuse(f"the reference labels {name} do not read ({type(exc).__name__})")
        if not isinstance(found, dict):
            refuse(f"the reference labels {name}: a JSON object {{item_id: label | null}}")
        try:
            out["annotators"][name] = sa.reference_contrast(cells, key, found)
        except ValueError as exc:
            refuse(f"the reference labels {name}: {exc}")
        labels[name] = found
    if len(labels) == 2:
        first, second = labels.values()
        out["agreement"] = {"of": list(labels), **sa.agreement(first, second)}
    elif len(labels) > 2:
        refuse("at most two reference annotators")
    return out


# ------------------------------------------------------------------ the fresh seeds


def fresh_seeds(fixation: Mapping[str, Any], n: Any) -> list[int]:
    for key in ("fresh", "alternative"):
        if fixation["seeds"][key]["n"] == n:
            return list(fixation["seeds"][key]["seeds"])
    refuse(f"pod2's n {n!r} is neither the fixation record's fresh n nor its alternative")


def own_code() -> dict[str, Any]:
    """This checkout's code: a clean checkout of a known commit, as the fixation record's."""
    state = git_state(REPO_ROOT)
    if not state.get("commit") or state.get("dirty") is not False:
        refuse("the analysis runs from a clean checkout of a known commit")
    return {"commit": state["commit"], "dirty": False,
            "module": "zero_field_refusal.analysis.study5_analysis"}  # fmt: skip


def reader_of(args: argparse.Namespace) -> Reader:
    roots = {"runs": Path(args.runs_root), "pod1": Path(args.pod1)}
    if getattr(args, "pod2", None):
        roots["pod2"] = Path(args.pod2)
    roots |= {"repo": REPO_ROOT, "fixation": Path(args.fixation).parent}
    if getattr(args, "study3_runs_root", None):
        roots |= {"study3_runs": Path(args.study3_runs_root), "study3_repo": Path(args.study3_repo)}
    if getattr(args, "reference_input", None):
        roots["reference"] = Path(args.reference_input)
    if getattr(args, "retrospective", None):
        roots["retrospective"] = Path(args.retrospective).parent
    return Reader(roots)


def fixation_of(reader: Reader, path: Path) -> tuple[bytes, dict[str, Any]]:
    data = reader.bytes(Path(path))
    found = json.loads(data)
    if not isinstance(found, dict) or found.get("kind") != "study5-fixation":
        refuse(f"{Path(path).name} is not study 5's fixation record")
    return data, found


def retrospective_record(args: argparse.Namespace) -> dict[str, Any]:
    """The retrospective alone (registration, "Retrospective predictor"; before the registration,
    without pod2): study 3's parts and pod1's records, written as its own record."""
    code = own_code()
    runs, reader = Path(args.runs_root), reader_of(args)
    fixation_data, fixation = fixation_of(reader, args.fixation)
    prompts_of = prompt_sets(reader)
    pod1 = Plan(reader, args.pod1, runs)
    parts = study3_parts(args, fixation, reader, prompts_of)
    retro = sa.finite_tree(retrospective(parts, pod1, reader, fixation))
    return {
        "kind": RETRO_KIND, "version": VERSION, "registration": fixation["registration"],
        "fixation": {"path": FIXATION, "sha256": sha256(fixation_data)}, "code": code,
        "retrospective": retro, "inputs": dict(sorted(reader.hashes.items())),
        "prompt_sets": dict(sorted(reader.prompt_sets.items())),
    }  # fmt: skip


def verified_retrospective(reader: Reader, path: Path, fixation_data: bytes,
                           recomputed: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """The retrospective record written before the registration, recomputed here from the same
    inputs: refused unless it is of this fixation record and its ``retrospective`` agrees with the
    recomputation (``same_tree``: the same structure, ids, seeds, counts and states; numbers
    within ``REL_TOLERANCE``; reasons not compared). The largest numeric deviation is kept."""
    try:
        data = reader.bytes(Path(path))
        found = json.loads(data)
    except (OSError, ValueError) as exc:
        refuse(f"the retrospective record does not read ({type(exc).__name__})")
    if not isinstance(found, dict) or found.get("kind") != RETRO_KIND:
        refuse(f"{Path(path).name} is not study 5's retrospective record")
    if (found.get("fixation") or {}).get("sha256") != sha256(fixation_data):
        refuse("the retrospective record is not of this fixation record")
    try:
        deviation = same_tree(found.get("retrospective"), recomputed)
    except Mismatch as exc:
        refuse(f"the retrospective record does not match its recomputation from the inputs "
               f"(at {exc})")  # fmt: skip
    return {
        **recomputed,
        "record": {
            "path": Path(path).name,
            "sha256": sha256(data),
            "code": found.get("code"),
            "max_deviation": deviation,
        },
    }


class Mismatch(ValueError):
    """Two records' trees differ: the path where they first do."""


def same_tree(stored: Any, computed: Any, where: str = "retrospective") -> dict[str, float]:
    """Whether a stored record's tree agrees with its recomputation: the same keys and lengths;
    ids, seeds, counts, states and every other string or integer equal; floats within a relative
    ``REL_TOLERANCE`` (``ABS_TOLERANCE`` near zero), so that the last bit of a ρ or a percentile
    may move across machines; the free text under ``reason`` and ``missing`` keys not compared,
    only that it is text. Returns the largest absolute and relative deviation of the floats;
    ``Mismatch`` (naming the path) otherwise."""
    worst = {"absolute": 0.0, "relative": 0.0}

    def walk(a: Any, b: Any, path: str, free: bool) -> None:
        if free and isinstance(a, str) and isinstance(b, str):
            return
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if set(a) != set(b):
                raise Mismatch(f"{path}: keys {sorted(set(a) ^ set(b))}")
            for key in a:
                walk(a[key], b[key], f"{path}.{key}", key in FREE_TEXT)
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise Mismatch(f"{path}: {len(a)} items, not {len(b)}")
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                walk(x, y, f"{path}[{i}]", False)
            return
        if isinstance(a, float) or isinstance(b, float):
            numbers = all(isinstance(v, int | float) and not isinstance(v, bool) for v in (a, b))
            if not numbers or not (math.isfinite(a) and math.isfinite(b)):
                raise Mismatch(path)
            gap = abs(a - b)
            scale = max(abs(a), abs(b))
            if gap > max(REL_TOLERANCE * scale, ABS_TOLERANCE):
                raise Mismatch(f"{path}: {a!r} against {b!r}")
            worst["absolute"] = max(worst["absolute"], gap)
            worst["relative"] = max(worst["relative"], gap / scale if scale > 0 else 0.0)
            return
        if type(a) is not type(b) or a != b:
            raise Mismatch(path)

    walk(stored, computed, where, False)
    return worst


def _when(stamp: Any) -> datetime | None:
    if not isinstance(stamp, str):
        return None
    try:
        found = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return found if found.tzinfo is not None else found.replace(tzinfo=UTC)


def first_start(reader: Reader, runs: Path, plan: Plan) -> str | None:
    """The earliest run start in the plan's launcher journals (pod2's: the fresh runs first)."""
    stamps = [e.get("time") for name in sorted({name for name, _ in plan.cells})
              for e in journal(reader, runs, name)
              if e.get("event") == "start" and _when(e.get("time")) is not None]  # fmt: skip
    return min(stamps, key=_when) if stamps else None


def ladder_rungs(plan: Plan, seeds: Sequence[int]) -> list[str]:
    """Every rung the plan has a ``controls/s<seed>/ladder-<rung>`` step of, in
    ``evaluation.LADDER_RUNGS``' order (another name after them)."""
    found = set()
    for name in plan.steps:
        match = LADDER_STEP.fullmatch(name)
        if match and int(match["seed"]) in seeds:
            found.add(match["rung"])
    return sorted(found, key=lambda r: (sa.RUNGS.index(r) if r in sa.RUNGS else len(sa.RUNGS), r))


def analysis(args: argparse.Namespace) -> dict[str, Any]:
    code = own_code()
    runs, reader = Path(args.runs_root), reader_of(args)
    fixation_data, fixation = fixation_of(reader, args.fixation)
    prompts_of = prompt_sets(reader)
    pod1, pod2 = Plan(reader, args.pod1, runs), Plan(reader, args.pod2, runs)
    seeds = fresh_seeds(fixation, pod2.data.get("n"))
    controls_seeds = [s for s in fixation["seeds"]["controls"] if s in seeds]
    ladder_seeds = [s for s in fixation["seeds"]["ladder"] if s in seeds]
    level = float(fixation["scan_intervals"]["level"])
    p, binding = predictions(reader, pod2, seeds)
    start = first_start(reader, runs, pod2)
    labels = Labels(reader, prompts_of)

    theta0_dir = completed(reader, runs, pod1.find("experiments/study5_theta0", seeds[0]), "θ₀")
    if isinstance(theta0_dir, str):
        refuse(f"θ₀'s evaluation: {theta0_dir}")
    from .study3_records import read_theta0

    try:
        theta0 = read_theta0(theta0_dir)
        theta0_summary = reader.json(theta0_dir / "summary.json")
    except (OSError, ValueError) as exc:
        refuse(f"θ₀'s evaluation does not read: {reader.scrub(str(exc))}")
    theta0_counts = labels.at(theta0_dir, THETA0)
    theta0_point = sa.point_values(theta0_counts, theta0_summary)

    outcomes, h, refusal, per_scan, own, control = [], {}, {}, {}, {}, {"order": {}, "v0": {}}
    lower_ids: dict[int, str | None] = {}  # the η_l run's id
    for seed in seeds:
        lo_dir = completed(reader, runs, pod2.find(FRESH, seed, ETA_L, (FRESH_ROLE,)), "η_l")
        hi_dir = completed(reader, runs, pod2.find(FRESH, seed, ETA_U, (FRESH_ROLE,)), "η_u")
        lo, hi = f_run(reader, lo_dir, seed, ETA_L), f_run(reader, hi_dir, seed, ETA_U)
        endpoints, missing, run_labels_ = [], [], {}
        for found in (lo, hi):
            if isinstance(found, str):
                missing.append(found)
                continue
            endpoints.append(sa.endpoint_of(found["run_id"], seed=seed, lr=found["lr"],
                                            stop=found["stop"], row=found["row"],
                                            theta0=theta0, steps=found["steps"]))  # fmt: skip
            run_labels_[found["run_id"]] = labels.at(found["run_dir"], ENDPOINT)
        result = sa.seed_outcome(seed, endpoints, missing, run_labels_)
        h[seed] = result["h"] if result["h"] is not None else result["reason"]
        refusal[seed] = (result["classes"]["refusal"] if result["classes"] is not None
                         else result["reason"])  # fmt: skip
        outcomes.append({**result, "p": p[seed]})
        # the line scan of the η_l run's endpoint, from its scan run: a = 0 θ₀, a = +1 the η_l
        # run's step-25 evaluation
        if isinstance(lo, str):
            per_scan[seed] = lo
            own[seed] = lo
            lower_ids[seed] = None
            continue
        lower_ids[seed] = lo["run_id"]
        endpoint = sa.point_values(labels.at(lo["run_dir"], ENDPOINT), lo["row"] or
                                   "no step-25 evaluation")  # fmt: skip
        own[seed] = endpoint
        scan = record(reader, pod2.output(f"fresh/s{seed}/scan"), "study5-scan", seed, "scan")
        per_scan[seed] = {0.0: theta0_point, 1.0: endpoint,
                          **scan_points(reader, labels, scan, runs, lo["run_id"])}  # fmt: skip
    for seed in controls_seeds:
        for kind, config in (("order", "experiments/study5_control_order"),
                             ("v0", "experiments/study5_control_v0")):  # fmt: skip
            found = f_run(reader, completed(reader, runs, pod2.find(config, seed, ETA_L), kind),
                          seed, ETA_L)  # fmt: skip
            control[kind][seed] = (
                found
                if isinstance(found, str)
                else sa.point_values(
                    labels.at(found["run_dir"], ENDPOINT), found["row"] or "no step-25 evaluation"
                )
            )

    ladder_records: dict[int, dict[str, Any]] = {}
    ladder_labels: dict[tuple[int, str], Any] = {}
    rungs = ladder_rungs(pod2, ladder_seeds)
    for seed in ladder_seeds:
        ladder_records[seed] = {}
        for rung in rungs:
            found = record(reader, pod2.output(f"controls/s{seed}/ladder-{rung}"), "study5-ladder",
                           seed, f"ladder {rung}")  # fmt: skip
            if isinstance(found, Mapping) and found.get("rung") != rung:
                found = f"ladder {rung}: the record is of rung {found.get('rung')!r}"
            elif isinstance(found, Mapping) and (found.get("f_run") or {}).get(
                    "run_id") != lower_ids[seed]:  # fmt: skip
                found = f"ladder {rung}: the record is not of the seed's η_l run"
            ladder_records[seed][rung] = found
            if isinstance(found, Mapping) and found.get("status") == "evaluated":
                run_dir = runs / str(found["experiment"]) / str(found["run_id"])
                ladder_labels[(seed, rung)] = labels.at(
                    run_dir if run_dir.is_dir() else f"ladder {rung}: its run directory is missing",
                    ladder_condition(rung))  # fmt: skip

    r_endpoint: dict[int, sa.Value] = {}
    refusal_endpoint: dict[int, sa.Value] = {}
    features: dict[str, dict[int, sa.Value]] = {f: {} for f in sa.FEATURES}
    for seed in seeds:
        found = record(reader, pod2.output(f"fresh/s{seed}/surrogate-endpoint"),
                       "study5-surrogate-endpoint", seed, "r")  # fmt: skip
        if isinstance(found, Mapping) and found.get("run_id") != lower_ids[seed]:
            found = (
                "r: the record is of another run than the seed's η_l run"
                if lower_ids[seed] is not None
                else f"r: no η_l run ({own[seed]})"
            )
        r_endpoint[seed] = found if isinstance(found, str) else float(found["r"])
        point = own.get(seed)
        refusal_endpoint[seed] = point if isinstance(point, str) else point["refusal"]
        found = record(reader, pod2.output(f"streams/s{seed}/features"),
                       "study5-stream-features", seed, "features")  # fmt: skip
        for name in sa.FEATURES:
            features[name][seed] = found if isinstance(found, str) else found["features"][name]

    if args.without_study3:
        parts, retro = None, {"computed": False, "reason": "--without-study3"}
    else:
        parts = study3_parts(args, fixation, reader, prompts_of)
        retro = verified_retrospective(reader, args.retrospective, fixation_data, sa.finite_tree(
            retrospective(parts, pod1, reader, fixation)))  # fmt: skip
    reference = reference_part(args, fixation_data, fixation, reader)

    return sa.finite_tree({
        "kind": KIND,
        "version": VERSION,
        "registration": fixation["registration"],
        "fixation": {"path": FIXATION, "sha256": sha256(fixation_data)},
        "code": code,
        "k_star": sa.K_STAR,
        "seeds": {"fresh": seeds, "n": len(seeds), "controls": controls_seeds,
                  "ladder": ladder_seeds},
        "predictions": {**binding, "first_start": start,
                        "created_before_first_start": (_when(binding["created"]) < _when(start)
                                                       if _when(binding["created"]) and _when(start)
                                                       else None)},
        "confirmatory": {**sa.confirmatory(p, h, seeds, fixation["permutation_test"]),
                         "excluded_by_state": sa.excluded_by_state(outcomes)},
        "outcomes": outcomes,
        "unsupported": [{"seed": o["seed"], "reason": o["reason"]} for o in outcomes
                        if o["h"] is None],
        "retrospective": retro,
        "descriptive": {
            "refusal_at_kstar": sa.association(p, refusal, seeds, ("p", "refusal")),
            "surrogate_endpoint": sa.association(r_endpoint, refusal_endpoint, seeds,
                                                 ("r", "refusal")),
            "line_scans": sa.line_scans(per_scan, seeds, level),
            "controls": sa.controls(own, control, controls_seeds, level),
            "ladder": sa.ladder_table(ladder_records, ladder_labels, h, ladder_seeds, rungs,
                                      level),
            "theta0": {"run_id": theta0_dir.name,
                       "counts": theta0_counts if isinstance(theta0_counts, Mapping) else None,
                       "classes": {k: sa._json_value(v) for k, v in
                                   sa.class_shares(theta0_counts).items()}},
            "stream_features": {name: sa.association(features[name], h, seeds, (name, "h"))
                                for name in sa.FEATURES},
            "extreme_blocks": extreme_blocks(parts, fixation, reference),
        },
        "inputs": dict(sorted(reader.hashes.items())),
        "prompt_sets": dict(sorted(reader.prompt_sets.items())),
    })  # fmt: skip


# ------------------------------------------------------------------ writing and the CLI


def write_once(path: Path, data: bytes) -> None:
    """Write ``data`` at ``path`` whole, never over an existing file (a hard link)."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        refuse(f"{path.name} exists: never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_bytes(data)
    try:
        os.link(partial, path)
    except FileExistsError:
        refuse(f"{path.name} exists: never overwritten")
    finally:
        partial.unlink(missing_ok=True)


def _number(value: Any) -> str:
    return "none" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def report(result: Mapping[str, Any], digest: str) -> None:
    """Numbers, counts and the record's sha256 only."""
    if result["kind"] == KIND:
        test = result["confirmatory"]
        print(f"fresh {result['seeds']['n']} supported {test['n']} unsupported "
              f"{len(result['unsupported'])}")  # fmt: skip
        print(f"rho {_number(test['rho'])} p {_number(test['p_value'])} descriptive "
              f"{int(test['descriptive'])} predicts "
              f"{int(test['reading'] == sa.READINGS['predicts'])}")  # fmt: skip
    retro = result["retrospective"]
    if retro.get("computed"):
        interval = retro["harm"]["interval"] or {}
        print(f"retrospective blocks {retro['blocks']} n {retro['harm']['n']} rho "
              f"{_number(retro['harm']['rho'])} lower {_number(interval.get('lower'))} upper "
              f"{_number(interval.get('upper'))}")  # fmt: skip
    print(f"{'analysis' if result['kind'] == KIND else 'retrospective'} sha256 {digest}")


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--pod1", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fixation", type=Path, default=REPO_ROOT / FIXATION)
    parser.add_argument("--study3-runs-root", type=Path)
    parser.add_argument("--study3-repo", type=Path)
    parser.add_argument("--study3-sealed-dir", type=Path)


def parser_of() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"python -m zero_field_refusal.analysis.{PROG}",
                                     description=__doc__.splitlines()[0])  # fmt: skip
    sub = parser.add_subparsers(dest="command", required=True)
    retro = sub.add_parser("retrospective")
    _common(retro)
    full = sub.add_parser("full")
    _common(full)
    full.add_argument("--pod2", type=Path, required=True)
    full.add_argument("--retrospective", type=Path)
    full.add_argument("--without-study3", action="store_true")
    full.add_argument("--reference-input", type=Path)
    full.add_argument("--reference", action="append", default=[], metavar="NAME=FILE")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_of().parse_args(argv)
    try:
        study3 = bool(args.study3_runs_root and args.study3_repo)
        if args.command == "retrospective":
            if not study3:
                refuse("the retrospective needs --study3-runs-root and --study3-repo")
        elif args.without_study3 == study3:
            refuse("give --study3-runs-root and --study3-repo, or --without-study3")
        elif study3 and args.retrospective is None:
            refuse("study 3's parts need --retrospective, the record written before the "
                   "registration")  # fmt: skip
        if args.out.exists() or args.out.is_symlink():
            refuse(f"{args.out.name} exists: never overwritten")
        result = retrospective_record(args) if args.command == "retrospective" else analysis(args)
        data = encoded(result)
        write_once(args.out, data)
    except Refused as exc:
        sys.exit(f"{PROG}: {exc}")
    report(result, sha256(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
