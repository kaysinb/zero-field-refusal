"""Study 5's analysis on a CPU: the retrospective predictor before the registration, and after
the Pods the confirmatory test and the descriptive measurements of docs/study5-preregistration.md,
from the campaigns' records and the label files, each written once as one record (the
computations are ``analysis.study5_analysis``'s).

    python -m zero_field_refusal.analysis.study5_analysis_cli retrospective --runs-root DIR
        --pod1 DIR --out FILE --study3-runs-root DIR --study3-repo DIR [--study3-sealed-dir DIR]
        [--fixation docs/study5-fixation.json]
    python -m zero_field_refusal.analysis.study5_analysis_cli ready --runs-root DIR --pod1 DIR
        --pod2 DIR [--fixation docs/study5-fixation.json]
    python -m zero_field_refusal.analysis.study5_analysis_cli full --runs-root DIR --pod1 DIR
        --pod2 DIR --out FILE [--fixation docs/study5-fixation.json]
        (--study3-runs-root DIR --study3-repo DIR [--study3-sealed-dir DIR] --retrospective FILE
         | --without-study3)
        [--reference-input DIR --reference NAME=FILE [--reference NAME=FILE]]
    python -m zero_field_refusal.analysis.study5_analysis_cli consensus-plan --sonnet FILE
        --grok FILE --gpt FILE --out DIR --reference-input DIR --stops FILE --batches DIR
        --seed N
    python -m zero_field_refusal.analysis.study5_analysis_cli consensus --sonnet FILE --grok FILE
        --gpt FILE --out DIR --manifest FILE --arbiter FILE --arbiter-stops FILE

**ready** (the pre-label check, before any fresh label; ``readiness``) reads the analysis's
inputs and no label, and exits 1 naming each open technical gap (a plan fresh run not ended
``done`` or ``diverged`` or its records unreadable, study 3's open or unresolved support state, p
not bound to an endpoint or its predict record missing, pod 2's completion record not standing);
with none it writes a ready-to-label record (kind ``study5-ready-to-label``: its time, the
completion record's sha256, the runs it certifies) once, under ``study5_records/<pod 2's
ID>/ready-to-label/`` of the runs root. Fresh labeling starts only after it passed. A repair
after labeling began (a technical gap found late) is allowed: the check runs again, writes a
newer record, and every pod-2 run is labeled again; labels from before the latest record are
superseded (listed, never used).

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
| null}`` over exactly the key's items; each is reported, and the first two's agreement: two
annotators and their consensus (``consensus``), at most three.

**consensus-plan** and **consensus** build the reference consensus of study 3's amendments 11
and 12 (``analysis.study5_reference``): the plan and the arbiter's blind input once, before the
arbiter's first call, with the vote order's seed in its manifest; then, from the same inputs and
the arbiter's decisions, ``CONSENSUS.json`` and its source record once.

**The gates and the bindings** (``full``). The fixation record read has the sha256 the
registration at this commit names (``registered``). Pod 1's engineering-check record is equal (at
least one pair, every rerun from a clean checkout of a commit of pod 2's plan's GPU code identity,
the record's own identity that one) and pod 1's surrogate checks passed (``gates``); the
predictions record's GPU code identity is pod 2's plan's; one surrogate sha256 runs through the
checks, both plans, every predictor record (fresh and retrospective) and every endpoint record.
Each is a refusal otherwise. Each fresh seed's p is bound to both its F runs, η_l and η_u: its
predict record the one the predictions record names (sha256), its v₀ and θ₀ hashes each run's,
each run's training stream a prefix of the prediction's by content (the same stream, its chunks
the first of the prediction's: a run stopped early read fewer).

**No inference while a seed is unresolved.** The support rule is the only registered exclusion
(an unsupported seed: no h, reported, not replaced). Every other gap leaves the seed
*technically unresolved* (``unresolved``: ``binding`` for a mismatch on either endpoint,
``runs`` for a run missing, not ended or unreadable, ``technical`` for study 3's unresolved
state, ``labels`` for a supported seed whose labels are missing, pending or unreadable), never
excluded (``seeds_by_state`` keeps the two kinds apart). While any of the plan's seeds is, or a
rule of the order does not hold, the test gives no inference (``sa.confirmatory``: ``status``
``withheld``, no ρ, p-value or reading, ``reading_withheld`` naming each unresolved seed with its
reason and each other reason; ρ over the resolved supported seeds under ``descriptive_partial``
alone); the descriptive quantities stay as they are, partial, the missing named. The rules of the
order: the predictions' bucket copy created before the first fresh start (``order_violated``
otherwise); pod 2's completion record (``fresh/completion``: every fresh run its launcher
journals show as started, with its end) present, listing exactly the plan's fresh runs and every
fresh run read here as ended (``completion``); the latest ready-to-label record
(``ready_to_label``) of that completion record, written after it, certifying every fresh run
read; and the seeds the
committed choice (``docs/study5-confirmatory-seeds.json``, its sha256 named by the registration
at this commit, its seeds pod 2's: ``confirmatory_seeds``). Every run's GPUs as its
``meta.json`` records them are reported (``hardware``), never refused. --runs-root and
--study3-runs-root must not overlap.

**Superseded labels.** Of a pod-2 run, only labels whose labeling started (the earliest
``created_at`` of its label rows) after the latest ready-to-label record are used, their receipt
completed after it too (``labels_order``). Earlier ones are superseded: listed by run id
(``superseded_labels``, with ``labels_order.superseded``; a ``judge-superseded/`` folder, labels
moved aside by a repair, listed by its existence alone), never used in any quantity; the run then
awaits labels, a technical gap of its seed, not an order violation. θ₀'s run (pod 1) is apart.

**The surrogate overlap** (descriptive, predeclared). The judged prompts the fixation record's
``surrogate_overlap.evaluation_ids_same_or_close`` names match a surrogate-panel prompt by
content; h_s is recomputed on the other 482 by the same support rule and interpolation, and
ρ(p_s, h_s) reported on them, without a p-value or a reading (``descriptive.surrogate_overlap``),
and the same for the retrospective predictor on study 3's blocks
(``retrospective.surrogate_overlap``).

**The record.** ``analysis.json`` (--out, refused when it exists, never overwritten; written whole
by a link): the fixation record's sha256, the registration's, the code, the gates, the seeds,
the predictions record (its sha256 that of the receipt and the sidecar; its bucket copy's
creation time beside the first run start in pod2's launcher journals, and whether it is earlier),
the committed seeds, the completion record, the labels' order, the runs' GPUs, ``confirmatory``
(its ``status``, the seeds counted by state), ``outcomes`` (per fresh seed p, support, h, the
three classes or the reason, ``unresolved``), ``unsupported`` and ``technically_unresolved``,
``retrospective``, ``descriptive`` (the refusal share at K*, the surrogate at the endpoints, the
line scans, the controls, the ladder, θ₀'s baseline, the stream features and the extreme blocks)
and ``inputs``, every file read by its root and relative
path with its sha256 (a run's ``summary.json`` and the other files ``response_conditions`` reads
among them; study 3's label files are bound through the receipts the fixation record names).
Strict JSON, sorted keys, a non-finite number copied from a record as ``{"missing":
"nonfinite"}``. It holds numbers, ids, states and the registered words, never a text. stdout
carries numbers, counts and the record's sha256 only. The ladder's rungs are every
``controls/s<seed>/ladder-<rung>`` step of pod2's plan.

**Refusals** exit with ``study5_analysis_cli: <why>`` and write nothing: a dirty checkout, a
retrospective record that is not its recomputation, a plan or campaign that does not read or is
not its plan's, a predictions record that is not the published one, a fixation record the
registration does not name, a gate that does not stand, two surrogates, a study-3 input that is
not the fixation record's, study 3's labels pending, a reference input of another record,
overlapping runs roots. A missing run, record or label file of one seed is no refusal: it is
named in each quantity it leaves, the seed counts nowhere there, and it withholds the test's
inference.
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
from . import study3_audit as s3a
from . import study5_analysis as sa
from . import study5_reference as sr
from .evidence import HARMFUL, label_item_id, label_path
from .study2_label_cli import harmful_prompts

PROG = "study5_analysis_cli"
KIND, VERSION = "study5-analysis", 1
FIXATION = "docs/study5-fixation.json"
REGISTRATION = "docs/study5-preregistration.md"
SEEDS_FILE = "docs/study5-confirmatory-seeds.json"  # the n chosen before generation, committed
REGISTERED = re.compile(r"^Status: `registered`", re.MULTILINE)
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
    exclude: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """The label counts of every condition of a run (module docstring, "Labels"): per condition
    ``n``, ``labeled``, ``missing``, ``harmful`` and, for a three-class judge, ``refusal`` and
    ``safe_non_refusal`` (``not_harmful`` for ``llama_guard``); with ``exclude`` (prompt ids)
    also ``subset``: ``n``, ``harmful`` and ``missing`` over the run's prompts not excluded, and
    ``excluded``, how many of its prompts were; and the receipt's sha256 (``receipt_sha256``).
    ``study2_labels.LabelsPending`` or ``ValueError`` otherwise."""
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
    out: dict[str, Any] = {"receipt_sha256": receipt_sha, "conditions": {},
                           "started_at": labeling_started(data),
                           "completed_at": receipt.get("completed_at")}  # fmt: skip
    for condition in conditions:
        count = Counter(labels.get(label_item_id(condition, pid)) for pid in ids)
        labeled = len(ids) - count[None]
        row = {"n": len(ids), "labeled": labeled, "missing": count[None],
               "harmful": count[HARMFUL]}  # fmt: skip
        if identity.get("backend") == "llama_guard":
            row["not_harmful"] = labeled - count[HARMFUL]
        else:
            row |= {"refusal": count["refusal"], "safe_non_refusal": count["safe_non_refusal"]}
        if exclude:
            kept = Counter(labels.get(label_item_id(condition, pid)) for pid in ids
                           if pid not in exclude)  # fmt: skip
            n = sum(kept.values())
            row["subset"] = {"n": n, "harmful": kept[HARMFUL], "missing": kept[None],
                             "excluded": len(ids) - n}  # fmt: skip
        out["conditions"][condition] = row
    return out


def labeling_started(data: bytes) -> str | None:
    """When a run's labeling started: the earliest ``created_at`` over its label file's rows
    (the judge stamps each reply, ``judge.JudgeClient``); None when no row carries one."""
    stamps = []
    for line in data.decode("utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        stamp = row.get("created_at") if isinstance(row, dict) else None
        if _when(stamp) is not None:
            stamps.append(stamp)
    return min(stamps, key=_when) if stamps else None


class Labels:
    """The WildGuard labels of study-5 runs, each run read once, a failure kept as its reason."""

    def __init__(self, reader: Reader, prompts_of, judge: str = WILDGUARD,
                 exclude: frozenset[str] = frozenset(), ready: str | None = None,
                 exempt: Sequence[Path] = ()) -> None:  # fmt: skip
        """``ready``: the latest ready-to-label record's time; a run's labels whose labeling
        started at or before it are superseded (never used: the run awaits its labels), except
        for the ``exempt`` runs (θ₀'s, pod 1's)."""
        self.reader, self.prompts_of, self.judge = reader, prompts_of, judge
        self.exclude = frozenset(exclude)
        self.ready, self.exempt = _when(ready), {Path(d) for d in exempt}
        self.superseded: dict[str, dict[str, Any]] = {}  # run id → where and when
        self.identity = judge_identity(load_config(f"judge/{judge}"))
        self.found: dict[Path, dict[str, Any] | str] = {}

    def of(self, run_dir: Path) -> dict[str, Any] | str:
        if run_dir not in self.found:
            try:
                self.found[run_dir] = run_labels(self.reader, run_dir, self.judge, self.identity,
                                                 self.prompts_of, self.exclude)  # fmt: skip
            except (OSError, ValueError, KeyError) as exc:
                self.found[run_dir] = f"labels: {_why(self.reader, exc)}"
            self._supersede(Path(run_dir))
        return self.found[run_dir]

    def _supersede(self, run_dir: Path) -> None:
        """Labels from before the latest ready-to-label record are superseded: kept where they
        are, listed by run id, never used; the run awaits labels made after it. Labels moved
        aside (a ``judge-superseded/`` folder: its existence only, never its contents) are
        listed too."""
        if self.ready is None or run_dir in self.exempt:
            return
        archived = run_dir / "judge-superseded"
        if archived.is_dir() and any(archived.iterdir()):
            self.superseded.setdefault(run_dir.name, {})["archived"] = True
        found = self.found[run_dir]
        if not isinstance(found, Mapping):
            return
        started = _when(found.get("started_at"))
        if started is None or started <= self.ready:
            self.superseded.setdefault(run_dir.name, {}).update(
                started_at=found.get("started_at"), completed_at=found.get("completed_at")
            )
            self.found[run_dir] = ("labels: superseded: its labeling started before the latest "
                                   "ready-to-label record; the run awaits labels made after "
                                   "it")  # fmt: skip

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
        meta = reader.json(run_dir / "meta.json")
        rows = (reader.jsonl(run_dir / "evaluations.jsonl")
                if (run_dir / "evaluations.jsonl").is_file() else [])  # fmt: skip
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"{Path(run_dir).name}: its records do not read ({type(exc).__name__})"
    endpoint = [r for r in rows if r.get("condition") == ENDPOINT]
    return {"run_dir": run_dir, "run_id": run_dir.name, "seed": seed, "lr": lr,
            "stop": (terminal.get("stop") or {}).get("reason"), "steps": terminal.get("steps"),
            "row": endpoint[0] if len(endpoint) == 1 else None,
            "inputs": meta.get("inputs") or {}}  # fmt: skip


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
               "created": receipt.get("created"), "n": found["n"],
               "gpu_code_sha256": (found.get("code") or {}).get("gpu_code_sha256"),
               "entries": {e["seed"]: e for e in found["predictions"]}}  # fmt: skip


# ------------------------------------------------------------------ the gates and the bindings


def identity_of(commit: Any) -> str | None:
    """The GPU code identity of ``commit`` in this checkout's history (None: it cannot be told)."""
    if not isinstance(commit, str) or not commit:
        return None
    try:
        return code_identity.gpu_code_identity(commit, REPO_ROOT)["sha256"]
    except Exception:
        return None


def registration_text() -> str:
    """The registration at this checkout's commit (``own_code``: a clean checkout)."""
    return (REPO_ROOT / REGISTRATION).read_text(encoding="utf-8")


def registration_part(fixation_data: bytes) -> dict[str, Any]:
    """The registration binds the fixation record: the record read has the sha256 the
    registration at this commit names; refused otherwise."""
    text = registration_text()
    digest = sha256(fixation_data)
    if digest not in text:
        refuse(f"the fixation record read ({digest[:12]}) is not the one the registration names")
    return {"path": REGISTRATION, "sha256": sha256(text.encode("utf-8")),
            "registered": bool(REGISTERED.search(text)), "fixation_sha256": digest}  # fmt: skip


def gate_records(reader: Reader, pod1: Plan, gpu: Any) -> tuple[dict[str, Any], str]:
    """Pod 1's gates, reported, and the surrogate's sha256 of its checks: the engineering check
    equal (at least one pair, every rerun from a clean checkout of a commit of GPU code ``gpu``,
    its own ``code.gpu_code_sha256`` ``gpu``) and the surrogate checks passed; refused
    otherwise. ``gpu``: pod 2's plan's GPU code identity, the predictions record's."""
    found = {}
    for step, kind in (("engineering/check", "study5-engineering-check"),
                       ("surrogate/checks", "study5-surrogate-checks")):  # fmt: skip
        path = pod1.output(step)
        if path is None or not path.is_file():
            refuse(f"pod 1's {step} record is missing")
        try:
            data = reader.bytes(path)
            record = json.loads(data)
        except (OSError, ValueError) as exc:
            refuse(f"pod 1's {step} record does not read ({type(exc).__name__})")
        if not isinstance(record, dict) or record.get("kind") != kind:
            refuse(f"pod 1's {step} record is not a {kind} record")
        found[step] = (path, sha256(data), record)
    path, digest, eng = found["engineering/check"]
    pairs = eng.get("pairs") or []
    if eng.get("equal") is not True or not pairs:
        refuse("pod 1's engineering check is not equal")
    if (eng.get("code") or {}).get("gpu_code_sha256") != gpu:
        refuse("pod 1's engineering check is not of pod 2's GPU code identity")
    for pair in pairs:
        commit = pair.get("rerun_commit")
        identity = identity_of(commit)
        if identity is None:
            refuse(f"pod 1's engineering rerun {pair.get('rerun')}: commit {commit} not in "
                   "this checkout's history")  # fmt: skip
        if pair.get("rerun_dirty") is not False or identity != gpu:
            refuse(f"pod 1's engineering rerun {pair.get('rerun')} is not of a clean checkout of "
                   "pod 2's GPU code identity")  # fmt: skip
    path_c, digest_c, checks = found["surrogate/checks"]
    if checks.get("passes") is not True:
        refuse("pod 1's surrogate checks did not pass")
    spec = checks.get("surrogate_sha256")
    if not isinstance(spec, str):
        refuse("pod 1's surrogate checks record no surrogate sha256")
    return {
        "engineering": {"record": path.name, "sha256": digest, "equal": True,
                        "pairs": len(pairs), "gpu_code_sha256": gpu,
                        "rerun_commits": sorted({str(p.get("rerun_commit")) for p in pairs})},
        "surrogate_checks": {"record": path_c.name, "sha256": digest_c, "passes": True,
                             "surrogate_sha256": spec, "commit": (checks.get("git") or {}).get(
                                 "commit")},
    }, spec  # fmt: skip


def one_surrogate(spec: str, records: Mapping[str, Any]) -> None:
    """Every predictor record and every endpoint record of the surrogate of the checks
    (``records``: name → a record, or why it is missing); refused otherwise."""
    other = sorted(name for name, found in records.items()
                   if isinstance(found, Mapping)
                   and found.get("surrogate_sha256") != spec)  # fmt: skip
    if other:
        refuse(f"{len(other)} records are of another surrogate than the checks' (first: "
               f"{other[0]}): one surrogate sha256 across the checks, every predictor and every "
               "endpoint record")  # fmt: skip


def stream_prefix_problem(run: Any, predicted: Any) -> str | None:
    """The run's training stream against the one its prediction read, by content: the same
    stream, the run's chunks a prefix of the prediction's (a run stopped early read fewer)."""
    if not isinstance(run, Mapping) or not isinstance(predicted, Mapping):
        return "a training stream is not recorded"
    chunks, theirs = run.get("chunks"), predicted.get("chunks")
    if run.get("path") != predicted.get("path") or not isinstance(chunks, list) or not isinstance(
            theirs, list):  # fmt: skip
        return "the training streams are not the same stream"
    if (
        not chunks
        or run.get("count", 0) > predicted.get("count", 0)
        or theirs[: len(chunks)] != chunks
    ):
        return "the run's training stream is not a prefix of the prediction's"
    return None


def binding_problem(reader: Reader, entry: Mapping[str, Any] | None, path: Path | None,
                    run: Mapping[str, Any]) -> tuple[str | None, Any]:  # fmt: skip
    """A fresh seed's p bound to one of its two F runs (η_l and η_u, each checked): its predict
    record the one the predictions record names (sha256), and its v₀, θ₀ and training stream the
    run's (``run``: ``f_run``). The problem (None: bound) and the predict record (or why there is
    none)."""
    if entry is None:
        return "the predictions record has no entry of the seed", "no entry"
    if path is None or not path.is_file():
        return "its predict record is missing", "p: no record"
    data = reader.bytes(path)
    if sha256(data) != entry.get("sha256"):
        return "its predict record is not the one the predictions record names", "p: other record"
    record = json.loads(data)
    inputs = run["inputs"]
    for key, what in (("v0_sha256", "v₀"), ("theta0_trainable_sha256", "θ₀")):
        if record.get(key) is None or record.get(key) != inputs.get(key):
            return f"its prediction's {what} is not the run's", record
    found = stream_prefix_problem((inputs.get("streams") or {}).get("training"),
                                  (record.get("streams") or {}).get("training"))  # fmt: skip
    return found, record


def hardware_of(reader: Reader, run_dir: Path | str) -> Any:
    """A run's GPUs as its meta.json records them (``tracking.hardware``), or why not."""
    if isinstance(run_dir, str):
        return run_dir
    try:
        meta = reader.json(Path(run_dir) / "meta.json")
    except (OSError, ValueError) as exc:
        return f"meta.json does not read ({type(exc).__name__})"
    return list((meta.get("hardware") or {}).get("gpus") or [])


def labels_order(found: Mapping[Path, Any], pod2_runs: Sequence[Path], ready: str | None,
                 superseded: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """Labels of fresh runs opened only after every fresh run ended and every technical gap was
    resolved (registration, "Seeds and order"): the labels used are those whose labeling started
    (its label rows' earliest ``created_at``) after the latest ready-to-label record
    (``Labels``: earlier ones are superseded, listed, never used); each one's receipt completed
    (``completed_at``) after it too. ``after`` None without a ready-to-label record, False when a
    receipt's time does not follow it or does not read; each run's times are reported."""
    end = _when(ready)
    runs = {Path(d).name: found[d] for d in pod2_runs if isinstance(found.get(d), Mapping)}
    times = {r: f.get("completed_at") for r, f in runs.items()}
    earlier = sorted(r for r, t in times.items() if end is not None and _when(t) is not None
                     and _when(t) <= end)  # fmt: skip
    unknown = sorted(r for r, t in times.items() if _when(t) is None)
    after = None if end is None else not (earlier or unknown)
    return {"ready_written": ready, "runs": len(times), "after": after, "earlier": earlier,
            "unknown": unknown, "superseded": dict(sorted(superseded.items())),
            "labeling": {r: {"started_at": f.get("started_at"),
                             "completed_at": f.get("completed_at")}
                         for r, f in sorted(runs.items())}}  # fmt: skip


READY_KIND = "study5-ready-to-label"
READY_FOLDER = "ready-to-label"  # under the runs root's study5_records/<pod 2's ID>/


def ready_folder(runs: Path, pod2: Plan) -> Path:
    return Path(runs) / "study5_records" / str(pod2.data.get("archive_id")) / READY_FOLDER


def ready_records(reader: Reader, runs: Path, pod2: Plan) -> dict[str, Any]:
    """Every ready-to-label record the pre-label check wrote (``ready``), and the latest by its
    time, whose time is the one fresh labels must follow."""
    out: dict[str, Any] = {"records": [], "latest": None, "written_utc": None, "record": None,
                           "problems": []}  # fmt: skip
    folder = ready_folder(runs, pod2)
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            data = reader.bytes(path)
            found = json.loads(data)
        except (OSError, ValueError) as exc:
            out["problems"].append(f"the ready-to-label record {path.name} does not read "
                                   f"({type(exc).__name__})")  # fmt: skip
            continue
        if not isinstance(found, dict) or found.get("kind") != READY_KIND or _when(
                found.get("written_utc")) is None:  # fmt: skip
            out["problems"].append(f"{path.name} is not a ready-to-label record")
            continue
        out["records"].append({"record": path.name, "sha256": sha256(data),
                               "written_utc": found["written_utc"]})  # fmt: skip
        if out["record"] is None or _when(found["written_utc"]) > _when(out["written_utc"]):
            out |= {"latest": path.name, "written_utc": found["written_utc"], "record": found}
    return out


def ready_part(ready: Mapping[str, Any], completion: Mapping[str, Any],
               used: Mapping[tuple[str, str], str]) -> dict[str, Any]:  # fmt: skip
    """The latest ready-to-label record against pod 2's completion record and the runs read: of
    that completion record (its sha256), written after it, certifying every fresh run read. Its
    problems withhold the reading."""
    problems = list(ready["problems"])
    found = ready["record"]
    if found is None:
        problems.append("no ready-to-label record: the pre-label check (study5_analysis_cli "
                        "ready) has not passed")  # fmt: skip
    else:
        mine = found.get("completion") or {}
        if mine.get("sha256") != completion.get("sha256"):
            problems.append("the latest ready-to-label record is not of pod 2's completion record")
        end, own = _when(completion.get("written_utc")), _when(found.get("written_utc"))
        if end is None or own is None or own <= end:
            problems.append("the latest ready-to-label record does not postdate pod 2's "
                            "completion record")  # fmt: skip
        certified = {(r.get("campaign"), r.get("run"), r.get("run_id"))
                     for r in found.get("runs") or [] if isinstance(r, Mapping)}  # fmt: skip
        other = sorted(f"{c}: {r}" for (c, label), r in used.items()
                       if (c, label, r) not in certified)  # fmt: skip
        if other:
            problems.append(f"{len(other)} fresh runs read are not the ones the latest "
                            f"ready-to-label record certifies (first: {other[0]})")  # fmt: skip
    return {"records": ready["records"], "latest": ready["latest"],
            "written_utc": ready["written_utc"], "stands": not problems,
            "problems": problems}  # fmt: skip


def fresh_cells(plan: Plan) -> set[tuple[str, str]]:
    """The plan's fresh F runs (experiments/study5, ``study5.role=fresh``): (campaign, label)."""
    return {(name, f"{cell['name']}/s{seed}") for name, cell in plan.cells
            if cell.get("config") == FRESH and FRESH_ROLE in (cell.get("overrides") or [])
            for seed in cell.get("seeds") or []}  # fmt: skip


def completion_part(reader: Reader, pod2: Plan, used: Mapping[tuple[str, str], str]
                    ) -> dict[str, Any]:  # fmt: skip
    """Pod 2's completion record (``fresh/completion``, written once its last fresh group ended:
    every fresh run its launcher journals show as started, with its end): it must list exactly
    the plan's fresh runs and, ended, every run read here (``used``: (campaign, label) → run id).
    ``stands`` and its ``written_utc``; the problems otherwise (each withholds the reading)."""
    path = pod2.output("fresh/completion")
    out: dict[str, Any] = {"record": None, "sha256": None, "written_utc": None, "stands": False,
                           "problems": []}  # fmt: skip
    if path is None or not path.is_file():
        return out | {"problems": ["pod 2's completion record of the fresh runs is missing"]}
    try:
        data = reader.bytes(path)
        record = json.loads(data)
    except (OSError, ValueError) as exc:
        return out | {"problems": [f"pod 2's completion record does not read "
                                   f"({type(exc).__name__})"]}  # fmt: skip
    out |= {"record": path.name, "sha256": sha256(data)}
    if not isinstance(record, dict) or record.get("kind") != "study5-fresh-completion":
        return out | {"problems": ["pod 2's completion record is not a study5-fresh-completion "
                                   "record"]}  # fmt: skip
    listed = [r for r in record.get("runs") or [] if isinstance(r, Mapping)]
    labels = {(r.get("campaign"), r.get("run")) for r in listed}
    ended = {(r.get("campaign"), r.get("run"), r.get("run_id")) for r in listed
             if r.get("ended") is not None and r.get("outcome") is not None}  # fmt: skip
    problems = []
    planned = fresh_cells(pod2)
    if labels != planned:
        problems.append(f"the completion record lists {len(labels)} fresh runs, not exactly the "
                        f"plan's {len(planned)}")  # fmt: skip
    absent = sorted(f"{c}: {run_id}" for (c, label), run_id in used.items()
                    if (c, label, run_id) not in ended)  # fmt: skip
    if absent:
        problems.append(f"{len(absent)} fresh runs read are not ended in the completion record "
                        f"(first: {absent[0]})")  # fmt: skip
    if _when(record.get("written_utc")) is None:
        problems.append("the completion record's time does not read")
    return out | {"written_utc": record.get("written_utc"), "runs": len(listed),
                  "stands": not problems, "problems": problems}  # fmt: skip


def seeds_file_bytes() -> bytes:
    """``SEEDS_FILE`` at this checkout's commit (``own_code``: a clean checkout)."""
    return (REPO_ROOT / SEEDS_FILE).read_bytes()


def confirmatory_seeds(seeds: Sequence[int], text: str) -> dict[str, Any]:
    """The n chosen before generation (``SEEDS_FILE``): its sha256 named by the registration and
    its seeds pod 2's plan's. ``problems`` (each withholds the reading) otherwise."""
    try:
        data = seeds_file_bytes()
        chosen = json.loads(data)
    except (OSError, ValueError) as exc:
        return {"path": SEEDS_FILE, "sha256": None, "problems": [
            f"{SEEDS_FILE} does not read ({type(exc).__name__})"]}  # fmt: skip
    digest, problems = sha256(data), []
    if digest not in text:
        problems.append(f"the registration does not name the sha256 of {SEEDS_FILE}")
    if not isinstance(chosen, dict) or chosen.get("seeds") != list(seeds) or chosen.get(
            "n") != len(seeds):  # fmt: skip
        problems.append(f"pod 2's seeds are not the committed choice of {SEEDS_FILE}")
    return {"path": SEEDS_FILE, "sha256": digest, "n": (chosen or {}).get("n")
            if isinstance(chosen, dict) else None, "problems": problems}  # fmt: skip


def overlap_ids(fixation: Mapping[str, Any]) -> frozenset[str]:
    """The judged prompts that match a surrogate-panel prompt by content (the fixation record's
    ``surrogate_overlap.evaluation_ids_same_or_close``), left out by the descriptive check."""
    return frozenset((fixation.get("surrogate_overlap") or {}).get(
        "evaluation_ids_same_or_close") or ())  # fmt: skip


def without_overlap(block: Mapping[str, Any] | None, runs: Mapping[str, Any],
                    excluded: int) -> sa.Value:  # fmt: skip
    """h_s over the prompts not in the overlap (``sa.subset_share``), or why it is missing:
    ``runs``: run id → its counts at the endpoint (with ``subset``) or why they are missing."""
    if block is None:
        return "no supported block"
    counts = {}
    for end in ("lo", "hi"):
        run = block["arms"][sa.ARM][end]
        found = runs.get(run)
        if not isinstance(found, Mapping):
            return f"{run}: {found}"
        subset = found.get("subset")
        if not isinstance(subset, Mapping) or subset["n"] != sa.DENOMINATOR - excluded:
            return f"{run}: the overlap's {excluded} prompts are not all among its prompts"
        counts[run] = subset
    return sa.subset_share(block, counts)


OVERLAP_RULE = (
    "descriptive: h_s recomputed by the same support rule and interpolation on the "
    "judged prompts not in the fixation record's "
    "surrogate_overlap.evaluation_ids_same_or_close, and Spearman rho(p_s, h_s) on "
    "them; no p-value and no reading: the confirmatory test stays on the 521"
)


def overlap_check(p: Mapping[int, sa.Value], h_overlap: Mapping[int, sa.Value],
                  seeds: Sequence[int], excluded: frozenset[str]) -> dict[str, Any]:  # fmt: skip
    """The predeclared check of the surrogate overlap (registration, "The surrogate and the
    predictor"): ρ(p_s, h_s) with h_s on the 521 − ``len(excluded)`` prompts; descriptive."""
    if not excluded:
        return {"computed": False, "reason": "the fixation record names no overlap"}
    return {"computed": True, "rule": OVERLAP_RULE, "excluded_prompts": len(excluded),
            "prompts": sa.DENOMINATOR - len(excluded),
            "h": {str(s): sa._json_value(h_overlap.get(s, "no value")) for s in seeds},
            "association": sa.association(p, h_overlap, seeds, ("p", "h"))}  # fmt: skip


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
    wildguard = Labels(reader, prompts_of, exclude=overlap_ids(fixation))
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
    excluded = overlap_ids(fixation)
    h_overlap = {int(b["seed"]): without_overlap(b, three, len(excluded)) for b in blocks}
    used, xs, ys, _ = sa.pair(p, h, seeds, ("p", "h"))
    interval = (sa.bootstrap_interval(xs, ys, fixation["retrospective_interval"])
                if len(used) >= 2 else None)  # fmt: skip
    return {
        "computed": True, "label": "retrospective: an association, not a test",
        "blocks": len(blocks), "freeze_sha256": parts["freeze_sha256"],
        "study3_checkout": parts["checkout"],
        "harm": {**harm_rho, "interval": interval},
        "refusal": sa.association(p, refusal, seeds, ("p", "refusal")),
        "surrogate_overlap": overlap_check(p, h_overlap, seeds, excluded),
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
    if len(labels) > 3:
        refuse("at most three reference labels (two annotators and their consensus)")
    if len(labels) >= 2:
        first, second = list(labels.values())[:2]
        out["agreement"] = {"of": list(labels)[:2], **sa.agreement(first, second)}
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


def first_start(reader: Reader, runs: Path, plan: Plan, event: str = "start",
                last: bool = False) -> str | None:  # fmt: skip
    """The earliest run start in the plan's launcher journals (pod2's: the fresh runs first); with
    ``event`` "end" and ``last``, the last run end."""
    stamps = [e.get("time") for name in sorted({name for name, _ in plan.cells})
              for e in journal(reader, runs, name)
              if e.get("event") == event and _when(e.get("time")) is not None]  # fmt: skip
    if not stamps:
        return None
    return (max if last else min)(stamps, key=_when)


def overlapping(first: Path, second: Path) -> bool:
    a, b = Path(first).resolve(), Path(second).resolve()
    return a == b or a.is_relative_to(b) or b.is_relative_to(a)


def ladder_rungs(plan: Plan, seeds: Sequence[int]) -> list[str]:
    """Every rung the plan has a ``controls/s<seed>/ladder-<rung>`` step of, in
    ``evaluation.LADDER_RUNGS``' order (another name after them)."""
    found = set()
    for name in plan.steps:
        match = LADDER_STEP.fullmatch(name)
        if match and int(match["seed"]) in seeds:
            found.add(match["rung"])
    return sorted(found, key=lambda r: (sa.RUNGS.index(r) if r in sa.RUNGS else len(sa.RUNGS), r))


def theta0_of(reader: Reader, runs: Path, pod1: Plan, seeds: Sequence[int]
              ) -> tuple[Path, Mapping[str, float], Any]:  # fmt: skip
    """θ₀'s evaluation run (pod 1), its accuracies (gross collapse is checked against them) and
    its summary; refused when it does not read."""
    from .study3_records import read_theta0

    theta0_dir = completed(reader, runs, pod1.find("experiments/study5_theta0", seeds[0]), "θ₀")
    if isinstance(theta0_dir, str):
        refuse(f"θ₀'s evaluation: {theta0_dir}")
    try:
        return theta0_dir, read_theta0(theta0_dir), reader.json(theta0_dir / "summary.json")
    except (OSError, ValueError) as exc:
        refuse(f"θ₀'s evaluation does not read: {reader.scrub(str(exc))}")


def seed_runs(reader: Reader, runs: Path, pod2: Plan, seed: int, theta0: Mapping[str, float],
              entries: Mapping[int, Any]) -> dict[str, Any]:  # fmt: skip
    """A fresh seed's two F runs, read without any label: each run (``f_run``) or why it is
    missing, the plan's cell of each, the endpoints (``sa.endpoint_of``), the missing runs'
    reasons, F's support (study 3's rule, on K alone), and p's binding to both endpoints (the
    problems, each named by its endpoint, and the predict record)."""
    cells = {name: pod2.find(FRESH, seed, lr, (FRESH_ROLE,))
             for name, lr in (("η_l", ETA_L), ("η_u", ETA_U))}  # fmt: skip
    found = {name: f_run(reader, completed(reader, runs, cells[name], name), seed, lr)
             for name, lr in (("η_l", ETA_L), ("η_u", ETA_U))}  # fmt: skip
    endpoints, missing = [], []
    for run in found.values():
        if isinstance(run, str):
            missing.append(run)
            continue
        endpoints.append(sa.endpoint_of(run["run_id"], seed=seed, lr=run["lr"], stop=run["stop"],
                                        row=run["row"], theta0=theta0,
                                        steps=run["steps"]))  # fmt: skip
    predict_path = pod2.output(f"streams/s{seed}/predict")
    problems, predicted = [], None
    for name, run in found.items():
        if isinstance(run, str):
            continue
        problem, predicted = binding_problem(reader, entries.get(seed), predict_path, run)
        if problem is not None:
            problems.append(f"{name}: {problem}")
    return {"cells": cells, "lo": found["η_l"], "hi": found["η_u"], "endpoints": endpoints,
            "missing": missing, "support": sa.f_support(seed, endpoints)[0],
            "binding": problems, "predicted": predicted,
            "predict_path": predict_path}  # fmt: skip


def gap_kind(binding: Sequence[str], missing: Sequence[str], state: Any) -> str | None:
    """A seed's technical gap that is not its labels' (``sa.seeds_by_state``' kinds): ``binding``
    on either endpoint, ``runs`` (a run missing, not ended, unreadable, study 3's open state),
    ``technical`` (study 3's unresolved state); None when there is none."""
    return ("binding" if binding else "runs" if missing or state == "open"
            else "technical" if state == "unresolved" else None)  # fmt: skip


def analysis(args: argparse.Namespace) -> dict[str, Any]:
    code = own_code()
    runs, reader = Path(args.runs_root), reader_of(args)
    fixation_data, fixation = fixation_of(reader, args.fixation)
    registration = registration_part(fixation_data)
    prompts_of = prompt_sets(reader)
    pod1, pod2 = Plan(reader, args.pod1, runs), Plan(reader, args.pod2, runs)
    seeds = fresh_seeds(fixation, pod2.data.get("n"))
    controls_seeds = [s for s in fixation["seeds"]["controls"] if s in seeds]
    ladder_seeds = [s for s in fixation["seeds"]["ladder"] if s in seeds]
    level = float(fixation["scan_intervals"]["level"])
    p, published = predictions(reader, pod2, seeds)
    entries = published.pop("entries")
    gpu = pod2.data.get("gpu_code")
    if published["gpu_code_sha256"] != gpu:
        refuse("the predictions record's GPU code identity is not pod 2's plan's")
    gates, spec = gate_records(reader, pod1, gpu)
    if pod2.data.get("surrogate_sha256") not in (None, spec) or pod1.data.get(
            "surrogate_sha256") not in (None, spec):  # fmt: skip
        refuse("a plan names another surrogate than the checks'")
    start = first_start(reader, runs, pod2)
    excluded_ids = overlap_ids(fixation)
    theta0_dir, theta0, theta0_summary = theta0_of(reader, runs, pod1, seeds)
    # only labels started after the latest ready-to-label record count; θ₀'s (pod 1) apart
    ready = ready_records(reader, runs, pod2)
    labels = Labels(reader, prompts_of, exclude=excluded_ids, ready=ready["written_utc"],
                    exempt=(theta0_dir,))  # fmt: skip
    surrogate_records: dict[str, Any] = {}  # every predictor and endpoint record, by name
    pod2_runs: list[Path] = []  # the fresh seeds' runs read, for the labels' order
    gpus: dict[str, Any] = {}  # every run's GPUs, by run id
    theta0_counts = labels.at(theta0_dir, THETA0)
    theta0_point = sa.point_values(theta0_counts, theta0_summary)
    gpus[theta0_dir.name] = hardware_of(reader, theta0_dir)

    outcomes, h, refusal, per_scan, own, control = [], {}, {}, {}, {}, {"order": {}, "v0": {}}
    lower_ids: dict[int, str | None] = {}  # the η_l run's id
    unresolved: dict[int, str] = {}  # the technically unresolved seeds: no inference while any
    used: dict[tuple[str, str], str] = {}  # the fresh runs read: (campaign, label) → run id
    h_overlap: dict[int, sa.Value] = {}  # h_s without the prompts of the surrogate overlap
    for seed in seeds:
        # the two runs and p's binding to both endpoints (its record the published one, its v₀,
        # θ₀ and training stream each endpoint's), as the pre-label check reads them
        found = seed_runs(reader, runs, pod2, seed, theta0, entries)
        lo, hi, missing, problems = found["lo"], found["hi"], found["missing"], found["binding"]
        run_labels_ = {}
        for name, run in (("η_l", lo), ("η_u", hi)):
            if isinstance(run, str):
                continue
            used[found["cells"][name]] = run["run_id"]
            pod2_runs.append(run["run_dir"])
            gpus[run["run_id"]] = hardware_of(reader, run["run_dir"])
            run_labels_[run["run_id"]] = labels.at(run["run_dir"], ENDPOINT)
        result = {**sa.seed_outcome(seed, found["endpoints"], missing, run_labels_),
                  "binding": None, "unresolved": None}  # fmt: skip
        predict_path = found["predict_path"]
        if found["predicted"] is not None:
            surrogate_records[f"predict s{seed}"] = found["predicted"]
        elif predict_path is not None and predict_path.is_file():
            surrogate_records[f"predict s{seed}"] = reader.json(predict_path)
        # every gap but the support rule is technical: the seed is unresolved, not excluded
        kind = gap_kind(problems, missing, result["support"]["state"]) or (
            "labels" if result["block"] is not None and result["h"] is None else None
        )
        if problems:
            result |= {"binding": "; ".join(problems), "reason": "binding: " + "; ".join(problems)}
        if kind is not None:
            result |= {"h": None, "classes": None, "unresolved": kind}
            unresolved[seed] = f"{kind}: {result['reason']}"
        h[seed] = result["h"] if result["h"] is not None else result["reason"]
        h_overlap[seed] = (without_overlap(result["block"], run_labels_, len(excluded_ids))
                           if result["h"] is not None else result["reason"])  # fmt: skip
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
        if isinstance(scan, Mapping):
            scan_dir = runs / str(scan.get("experiment")) / str(scan.get("run_id"))
            if scan_dir.is_dir():
                pod2_runs.append(scan_dir)
                gpus[scan_dir.name] = hardware_of(reader, scan_dir)
    for seed in controls_seeds:
        for kind, config in (("order", "experiments/study5_control_order"),
                             ("v0", "experiments/study5_control_v0")):  # fmt: skip
            found = f_run(reader, completed(reader, runs, pod2.find(config, seed, ETA_L), kind),
                          seed, ETA_L)  # fmt: skip
            if not isinstance(found, str):
                pod2_runs.append(found["run_dir"])
                gpus[found["run_id"]] = hardware_of(reader, found["run_dir"])
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
                if run_dir.is_dir():
                    pod2_runs.append(run_dir)
                    gpus[run_dir.name] = hardware_of(reader, run_dir)
                ladder_labels[(seed, rung)] = labels.at(
                    run_dir if run_dir.is_dir() else f"ladder {rung}: its run directory is missing",
                    ladder_condition(rung))  # fmt: skip

    r_endpoint: dict[int, sa.Value] = {}
    refusal_endpoint: dict[int, sa.Value] = {}
    features: dict[str, dict[int, sa.Value]] = {f: {} for f in sa.FEATURES}
    for seed in seeds:
        found = record(reader, pod2.output(f"fresh/s{seed}/surrogate-endpoint"),
                       "study5-surrogate-endpoint", seed, "r")  # fmt: skip
        surrogate_records[f"endpoint s{seed}"] = found
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

    for step in pod1.steps:  # the retrospective predictor's records: the same surrogate
        if re.fullmatch(r"retrospective/s[0-9]+/predict", step):
            path = pod1.output(step)
            if path is not None and path.is_file():
                surrogate_records[step] = reader.json(path)
    one_surrogate(spec, surrogate_records)
    completion = completion_part(reader, pod2, used)
    readiness = ready_part(ready, completion, used)
    order = labels_order(labels.found, pod2_runs, ready["written_utc"], labels.superseded)
    chosen = confirmatory_seeds(seeds, registration_text())
    created = (_when(published["created"]) < _when(start)
               if _when(published["created"]) and _when(start) else None)  # fmt: skip
    withheld = [why for why, bad in (
        ("the predictions' bucket copy is not known to precede the first fresh run's start",
         created is not True),
        ("a fresh run's label receipt does not follow the latest ready-to-label record",
         readiness["stands"] and order["after"] is not True)) if bad]  # fmt: skip
    withheld += completion["problems"] + readiness["problems"] + chosen["problems"]
    test = {**sa.confirmatory(p, h, seeds, fixation["permutation_test"], unresolved=unresolved,
                              withheld=withheld),
            "seeds_by_state": sa.seeds_by_state(outcomes),
            "order_violated": created is False}  # fmt: skip

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
        "registered": registration,
        "fixation": {"path": FIXATION, "sha256": sha256(fixation_data)},
        "code": code,
        "k_star": sa.K_STAR,
        "gates": gates,
        "seeds": {"fresh": seeds, "n": len(seeds), "controls": controls_seeds,
                  "ladder": ladder_seeds},
        "predictions": {**published, "first_start": start, "created_before_first_start": created},
        "confirmatory_seeds": chosen,
        "completion": completion,
        "ready_to_label": readiness,
        "labels_order": order,
        "superseded_labels": sorted(labels.superseded),
        "hardware": {"runs": dict(sorted(gpus.items())),
                     "models": sorted({g for v in gpus.values() if isinstance(v, list)
                                       for g in v})},
        "confirmatory": test,
        "outcomes": outcomes,
        "unsupported": [{"seed": o["seed"], "reason": o["reason"]} for o in outcomes
                        if o["h"] is None and not o["unresolved"]],
        "technically_unresolved": [{"seed": o["seed"], "kind": o["unresolved"],
                                    "reason": o["reason"]} for o in outcomes
                                   if o["unresolved"]],
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
            "surrogate_overlap": overlap_check(p, h_overlap, seeds, excluded_ids),
        },
        "inputs": dict(sorted(reader.hashes.items())),
        "prompt_sets": dict(sorted(reader.prompt_sets.items())),
    })  # fmt: skip


# ------------------------------------------------------------------ the pre-label check


def readiness(args: argparse.Namespace) -> tuple[list[str], dict[str, Any], Path]:
    """The pre-label check (``ready``): before any fresh label, every technical gap that is not
    a label's must be resolved, read from the analysis's own inputs without any label: each plan
    fresh run ended with its records readable (its launcher campaign's last end ``done`` or
    ``diverged``, no conflict, its run directory there), study 3's support state neither open nor
    unresolved, p bound to both endpoints with its predict record present, and pod 2's
    completion record standing. The open gaps (each named), and the ready-to-label record to
    write when there are none: its time, the completion record's sha256 and the runs it
    certifies. A repair later (a technical gap found after labeling began) is followed by this
    check again and the labeling redone: labels from before the latest record are superseded."""
    code = own_code()
    runs, reader = Path(args.runs_root), reader_of(args)
    _, fixation = fixation_of(reader, args.fixation)
    pod1, pod2 = Plan(reader, args.pod1, runs), Plan(reader, args.pod2, runs)
    seeds = fresh_seeds(fixation, pod2.data.get("n"))
    _, published = predictions(reader, pod2, seeds)
    entries = published.pop("entries")
    _, theta0, _ = theta0_of(reader, runs, pod1, seeds)
    gaps, used = [], {}
    for seed in seeds:
        found = seed_runs(reader, runs, pod2, seed, theta0, entries)
        for name, key in (("η_l", "lo"), ("η_u", "hi")):
            if isinstance(found[key], Mapping):
                used[found["cells"][name]] = found[key]["run_id"]
        state = found["support"]["state"]
        kind = gap_kind(found["binding"], found["missing"], state)
        if kind is not None:
            why = "; ".join(found["binding"] or found["missing"]) or (
                f"{state}: {found['support']['reason']}"
            )
            gaps.append(f"seed {seed}: {kind}: {reader.scrub(why)}")
    completion = completion_part(reader, pod2, used)
    gaps += completion["problems"]
    record = {
        "kind": READY_KIND, "version": VERSION, "written_utc": _now_utc(), "code": code,
        "pod2": pod2.data.get("archive_id"), "seeds": seeds,
        "completion": {key: completion[key] for key in ("record", "sha256", "written_utc")},
        "runs": [{"campaign": c, "run": label, "run_id": run_id}
                 for (c, label), run_id in sorted(used.items())],
        "inputs": dict(sorted(reader.hashes.items())),
    }  # fmt: skip
    return gaps, record, ready_folder(runs, pod2)


def _now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ready_main(args: argparse.Namespace) -> int:
    """Exit 1 naming each open gap, writing nothing; else the ready-to-label record, once, in
    ``ready_folder`` (``ready-<UTC>.json``)."""
    gaps, record, folder = readiness(args)
    if gaps:
        for gap in gaps:
            print(f"open gap: {gap}")
        print(f"not ready to label: {len(gaps)} open gaps")
        return 1
    stamp = record["written_utc"].replace(":", "").replace("-", "").replace("+0000", "Z")
    path = folder / f"ready-{stamp}.json"
    data = encoded(record)
    write_once(path, data)
    print(
        f"ready to label: {len(record['runs'])} fresh runs, completion "
        f"{record['completion']['sha256']}, record {path.name} sha256 {sha256(data)}"
    )
    return 0


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
        print(f"status {test['status']} unresolved {len(result['technically_unresolved'])}")
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


# ------------------------------------------------------------------ the reference consensus


def _labels_file(path: Path, name: str) -> tuple[bytes, dict[str, Any]]:
    try:
        data = Path(path).read_bytes()
        found = json.loads(data)
    except (OSError, ValueError) as exc:
        refuse(f"the {name} labels do not read ({type(exc).__name__})")
    if not isinstance(found, dict):
        refuse(f"the {name} labels: a JSON object {{item_id: label | null}}")
    return data, found


def _stopped_items(stops: Path, batches: Path) -> list[str]:
    """The items of every Claude call the stops record names (``model`` claude-*, ``batch``
    the batch file's name in ``batches``)."""
    items: list[str] = []
    for line in Path(stops).read_text().splitlines():
        if not line.strip():
            continue
        stop = json.loads(line)
        if str(stop.get("model", "")).startswith("claude"):
            batch = Path(batches) / f"{stop['batch']}.jsonl"
            items += [json.loads(row)["item_id"] for row in batch.read_text().splitlines() if row]
    return sorted(set(items))


def _consensus_inputs(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    hashes, labels = {}, {}
    for name in ("sonnet", "grok", "gpt"):
        data, labels[name] = _labels_file(getattr(args, name), name)
        hashes[name] = sha256(data)
    return hashes, labels


def consensus_plan_main(args: argparse.Namespace) -> int:
    """The consensus's manifest (its seed, inputs by sha256, the plan) and the arbiter's blind
    input, each written once, before the arbiter's first call."""
    folder = Path(args.reference_input)
    unique = (folder / "unique.jsonl").read_bytes()
    hashes, labels = _consensus_inputs(args)
    stopped = _stopped_items(args.stops, args.batches)
    try:
        planned = sr.plan(labels["sonnet"], labels["grok"], labels["gpt"], stopped=stopped,
                          seed=args.seed)  # fmt: skip
    except ValueError as exc:
        refuse(f"the consensus's plan: {exc}")
    rows = {json.loads(line)["item_id"]: json.loads(line) for line in unique.decode().splitlines()}
    blind = b""
    for item in planned["to_arbiter"]:
        row = {"item_id": item, "prompt": rows[item]["prompt"], "response": rows[item]["response"],
               "votes": planned["votes"][item]}  # fmt: skip
        s3a.blind_checked(row, ("item_id", "prompt", "response", "votes"))
        blind += (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    manifest = {
        "kind": "study5-reference-consensus-plan",
        "rule": "zero_field_refusal.analysis.study5_reference (study 3's amendments 11 and 12)",
        "seed": args.seed,
        "inputs": {
            "unique.jsonl": sha256(unique),
            "manifest.json": sha256((folder / "manifest.json").read_bytes()),
            "stops": sha256(Path(args.stops).read_bytes()),
            **hashes,
        },
        "stopped_items": stopped,
        "plan": planned,
        "arbiter_input": {"sha256": sha256(blind), "items": len(planned["to_arbiter"])},
        "code": own_code(),
        "written_utc": _now_utc(),
    }
    out = Path(args.out)
    write_once(out / "arbiter-input.jsonl", blind)
    data = encoded(manifest)
    write_once(out / "consensus-manifest.json", data)
    print(f"consensus plan: items {planned['items']} agreements {len(planned['agreements'])} "
          f"third {len(planned['third'])} to arbiter {len(planned['to_arbiter'])} not sent "
          f"{len(planned['not_sent'])} unvoted {len(planned['unvoted'])} manifest sha256 "
          f"{sha256(data)}")  # fmt: skip
    return 0


def consensus_main(args: argparse.Namespace) -> int:
    """The reference labels (``CONSENSUS.json``) and their source record, each written once, from
    the manifest's plan (recomputed from the same inputs and seed) and the arbiter's decisions."""
    manifest_data = Path(args.manifest).read_bytes()
    manifest = json.loads(manifest_data)
    hashes, labels = _consensus_inputs(args)
    for name, digest in hashes.items():
        if manifest["inputs"][name] != digest:
            refuse(f"the {name} labels are not the manifest's")
    planned = sr.plan(labels["sonnet"], labels["grok"], labels["gpt"],
                      stopped=manifest["stopped_items"], seed=manifest["seed"])  # fmt: skip
    if json.loads(json.dumps(planned)) != manifest["plan"]:
        refuse("the plan recomputed from the manifest's inputs is not the manifest's")
    arbiter_data, arbiter = _labels_file(args.arbiter, "arbiter")
    stopped = json.loads(Path(args.arbiter_stops).read_text())
    try:
        built = sr.consensus(planned, labels["sonnet"], arbiter, arbiter_stopped=stopped)
    except ValueError as exc:
        refuse(f"the consensus: {exc}")
    out = Path(args.out)
    labels_data = encoded(built["labels"])
    record = {"kind": "study5-reference-consensus", "manifest": sha256(manifest_data),
              "arbiter": sha256(arbiter_data), "arbiter_stopped": sorted(stopped),
              "consensus": sha256(labels_data), "counts": built["counts"],
              "missing": built["missing"], "sources": built["sources"], "code": own_code(),
              "written_utc": _now_utc()}  # fmt: skip
    write_once(out / "CONSENSUS.json", labels_data)
    data = encoded(record)
    write_once(out / "consensus-record.json", data)
    print("consensus: " + " ".join(f"{k.replace(' ', '_')} {v}" for k, v in built["counts"].items())
          + f" missing {built['missing']} labels sha256 {sha256(labels_data)} record sha256 "
          f"{sha256(data)}")  # fmt: skip
    return 0


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
    ready = sub.add_parser("ready")  # the pre-label check: no label read, its record written
    ready.add_argument("--runs-root", type=Path, required=True)
    ready.add_argument("--pod1", type=Path, required=True)
    ready.add_argument("--pod2", type=Path, required=True)
    ready.add_argument("--fixation", type=Path, default=REPO_ROOT / FIXATION)
    for name in ("consensus-plan", "consensus"):  # the reference consensus (amendments 11-12)
        step = sub.add_parser(name)
        for annotator in ("sonnet", "grok", "gpt"):
            step.add_argument(f"--{annotator}", type=Path, required=True)
        step.add_argument("--out", type=Path, required=True)
    plan_step = sub.choices["consensus-plan"]
    plan_step.add_argument("--reference-input", type=Path, required=True)
    plan_step.add_argument("--stops", type=Path, required=True)
    plan_step.add_argument("--batches", type=Path, required=True)
    plan_step.add_argument("--seed", type=int, required=True)
    build = sub.choices["consensus"]
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--arbiter", type=Path, required=True)
    build.add_argument("--arbiter-stops", type=Path, required=True)
    return parser


STEPS = {"ready": ready_main, "consensus-plan": consensus_plan_main, "consensus": consensus_main}


def main(argv: list[str] | None = None) -> int:
    args = parser_of().parse_args(argv)
    if args.command in STEPS:
        try:
            return STEPS[args.command](args)
        except Refused as exc:
            sys.exit(f"{PROG}: {exc}")
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
        if study3 and overlapping(args.runs_root, args.study3_runs_root):
            refuse("--runs-root and --study3-runs-root overlap: study 5's runs and study 3's are "
                   "read from separate roots")  # fmt: skip
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
