"""Study 5's fixation record, before the registration, and the reference annotators' input for
study 3's extreme blocks (docs/study5-preregistration.md, "The confirmatory test", "Descriptive
measurements"; the computations are ``analysis.study5_fixation``'s).

    python scripts/study5_fixation.py fix --runs-root DIR --study3-repo DIR --theta0 RUN_DIR
        --sealed-dir DIR --surrogate-overlap FILE [--out docs/study5-fixation.json]
    python scripts/study5_fixation.py reference-input --runs-root DIR --study3-repo DIR
        --out FOLDER [--fixation docs/study5-fixation.json]

--runs-root is study 3's runs root (its ``study3/`` runs, ``study3-final/c1/``). --study3-repo is
a clean checkout whose HEAD's GPU code is the one study 3's analysis freeze records: it supplies
the gate's GPU-code identity (study 3's label gate opens the final point only under that code,
``study3_labels``, "The points"), the committed freeze and, through the freeze, its validation
runs; the readers themselves run from this checkout, the record's ``code.commit``. --theta0 is
θ₀'s evaluation run (``study3_progress.read_state``); --sealed-dir the labeling's sealed
directory, whose ledger accepts the labels; --surrogate-overlap the classification of the
surrogate panel's prompts against the 521 judged ones by content (ids, numbers and the method
only; ``study5_fixation.surrogate_overlap``), recorded as given, its sha256 among the inputs.
The prompt sets are read from ``DATA_DIR`` (or the checkout's ``data/``).

**fix** writes the fixation record (``study5_fixation.fixation_record``) from this checkout with
its ``src/``, ``scripts/`` and ``configs/`` committed unchanged; the record's ``code`` is the last
commit that changed the fixation's own two files, with their blobs, so that the commit adding the
record keeps it:
1. study 3's analysis freeze of candidate 1 (``study3_freeze.read_freeze``, with no problem), its
   60 A–F blocks;
2. the state's endpoints (``study3_progress.read_state`` under the freeze's GPU code): each A–F
   run's rate and arm;
3. the harm of every run of those blocks through the final analysis's label reader
   (``study3_final.cohort_harm``, one gated reading), and the sha256 of each run's label file and
   receipt as the ledger's acceptance binds them (``study3_label_cli.label_hashes``; the receipt's
   must be the reading's), and the sha256 of every file of the labeling's ledger
   (``<sealed dir>/study3-ledger/``, hashed, never parsed here);
4. the block means at K* (``study3_estimates._arm``, study 3's per-arm interpolation), the 5
   highest and 5 lowest blocks, the 40 runs Llama Guard relabels, and, for each of those runs, its
   521 prompt ids in its prompt order (``run_judging.prompt_ids`` of its config's harmful set, as
   study 3's text reader takes them; the sha256 of its ``config.yaml``);
5. the record, strict JSON with sorted keys and no time: rerun on the same inputs and the same
   code it is byte-identical, and an existing file is left as it is; a different existing file
   is refused, never overwritten.
It prints the extreme blocks' seeds and means, counts and the record's sha256.

**reference-input** builds, from the committed fixation record, the input of the reference
annotators of study 3's amendments 11 and 12 in study 3's audit form (``scripts/study3_audit.py``,
``draw``): ``unique.jsonl``, one ``{item_id, prompt, response}`` row per item (study 3's
``BLIND`` fields, checked by ``study3_audit.blind_checked``, encoded as study 3 encodes them), in
the order and with the opaque ids of study 3's ``ordered`` under the record's order seed; the
texts are the prompt and the step-25 response of the item's run, read through study 3's final
reader (``study3_final.blind_texts``, which opens them only once study 3's analysis, the latest
``study3-final/c1/analysis*.json``, names the committed freeze). Beside it ``key.json`` (per item
id its cell, block, arm, rate, run, prompt id and weight) and ``manifest.json`` (the fixation
record's and the analysis's sha256, the order seed, per cell its item ids, and each file's item
count and sha256), neither with a text. The folder is written once, 0700, its files 0600 (study
3's ``write_folder``), never where Git could track it, inside study 3's checkout outside its
``runs/``, or under a sealed folder. It prints the folder, counts and the manifest's sha256.

**Refusals** exit with ``study5_fixation.py <command>: <why>``, writing nothing more; the messages
name ids, counts, paths and hashes, never a label or a text.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

from omegaconf import OmegaConf

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import study3_audit, study3_final
from zero_field_refusal.analysis import study5_fixation as fx
from zero_field_refusal.analysis.study2_label_cli import harmful_prompts, in_tracked_tree
from zero_field_refusal.analysis.study3_freeze import freeze_name, read_freeze
from zero_field_refusal.analysis.study3_label_cli import LEDGER, label_hashes
from zero_field_refusal.analysis.study3_labels import (
    JUDGE,
    LabelingProblem,
    LedgerError,
    SealedPoint,
    sealed_dir_of,
)
from zero_field_refusal.analysis.study3_progress import read_state
from zero_field_refusal.analysis.study3_records import EXPERIMENT, under_sealed
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.run_judging import prompt_ids
from zero_field_refusal.tracking import git_state

REPO = REPO_ROOT  # this checkout: the record's code and place
CANDIDATE = 1  # study 3's candidate whose confirmation the extreme blocks are drawn from
FIXATION = "docs/study5-fixation.json"
CODE_FILES = ("scripts/study5_fixation.py", "src/zero_field_refusal/analysis/study5_fixation.py")
COMMITTED = ("src", "scripts", "configs")  # committed unchanged when the record is written
INPUT_KIND = "study5-reference-input"


def refuse(command: str, why: str) -> NoReturn:
    sys.exit(f"study5_fixation.py {command}: {why}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(record: Any) -> bytes:
    return (json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def audit_script():
    """scripts/study3_audit.py as a module: its blind order and its write-once folder."""
    spec = importlib.util.spec_from_file_location(
        "study3_audit_script", REPO_ROOT / "scripts" / "study3_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    return done.stdout if done.returncode == 0 else ""


def record_code(command: str, repo: Path) -> dict[str, Any]:
    """The record's code: the last commit that changed the fixation's own files (``CODE_FILES``)
    and their blobs, with every file under ``COMMITTED`` committed unchanged at HEAD (the readers
    it runs included). A commit that only adds the record keeps it, so that a rerun is
    byte-identical."""
    if _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *COMMITTED).strip():
        refuse(command, "the record's code needs a clean checkout: "
               + ", ".join(f"{p}/" for p in COMMITTED) + " committed unchanged")  # fmt: skip
    commit = _git(repo, "log", "-1", "--format=%H", "--", *CODE_FILES).strip()
    blobs = {path: _git(repo, "rev-parse", f"HEAD:{path}").strip() for path in CODE_FILES}
    if not commit or not all(blobs.values()):
        refuse(command, f"the fixation's code ({', '.join(CODE_FILES)}) is not committed")
    return {"commit": commit, "files": blobs,
            "rule": "the last commit that changed these files; src/, scripts/ and configs/ "
                    "committed unchanged at HEAD when the record was written"}  # fmt: skip


def clean_commit(command: str, repo: Path, what: str) -> str:
    state = git_state(Path(repo))
    if not state.get("commit") or state.get("dirty") is not False:
        refuse(command, f"{what} needs a clean checkout of a known commit")
    return state["commit"]


def study3_checkout(command: str, repo: Path) -> dict[str, Any]:
    """Study 3's checkout: clean, its HEAD's GPU code that of the freeze. It supplies the label
    gate's GPU-code identity, the committed freeze and the freeze's validation runs; the readers
    run from this checkout."""
    commit = clean_commit(command, repo, "study 3's checkout (--study3-repo)")
    identity = code_identity.gpu_code_identity(commit, Path(repo))
    return {"commit": commit, "gpu_code_sha256": identity["sha256"]}


def ledger_hashes(command: str, sealed_dir: Path) -> dict[str, Any]:
    """The sha256 of every file of the labeling's ledger, by its name (hashed, never parsed)."""
    folder = Path(sealed_dir) / LEDGER
    files = sorted(p for p in folder.iterdir() if p.is_file()) if folder.is_dir() else []
    if not files:
        refuse(command, f"the labeling's ledger {LEDGER}/ holds no file")
    found = {p.name: sha256(p.read_bytes()) for p in files}
    return {"path": LEDGER, "root": "sealed_dir", "count": len(found),
            "sha256": sha256(encoded(found)), "files": found}  # fmt: skip


def rel_to(path: Path, base: Path) -> str:
    """``path`` relative to ``base`` when inside it, else its name alone (no local path)."""
    found, root = Path(path).resolve(), Path(base).resolve()
    return found.relative_to(root).as_posix() if found.is_relative_to(root) else found.name


def run_prompt_ids(run_dir: Path) -> tuple[list[str], str]:
    """A run's harmful prompt ids in its prompt order, as study 3's text reader takes them, and
    the sha256 of its config.yaml (the ids alone are kept)."""
    data = (run_dir / "config.yaml").read_bytes()
    cfg = OmegaConf.create(data.decode("utf-8"))
    return prompt_ids(cfg, harmful_prompts(cfg)), sha256(data)


def ids_sha256(ids: Sequence[str]) -> str:
    return sha256(json.dumps(list(ids)).encode())


# ------------------------------------------------------------------ fix


def af_harm(freeze: Mapping[str, Any], root: Path, repo: Path, sealed_dir: Path | None) -> dict:
    """The harm of every run of the frozen A–F blocks, one gated reading."""
    blocks = freeze["comparisons"]["A-F"]["blocks"]
    experiment = root / EXPERIMENT
    runs = {run: experiment / run for run in study3_final.runs_of(blocks)}
    try:
        return study3_final.cohort_harm(runs, candidate=CANDIDATE, runs_root=root, repo=repo,
                                        sealed_dir=sealed_dir)  # fmt: skip
    except (study3_final.Pending, SealedPoint) as exc:
        refuse("fix", str(exc))
    except (LabelingProblem, LedgerError) as exc:
        refuse("fix", f"the labels are not read ({type(exc).__name__})")
    except (OSError, ValueError) as exc:
        refuse("fix", f"the harm cannot be read ({type(exc).__name__})")


def rates_of(freeze: Mapping[str, Any], endpoints: Sequence[Any]) -> dict[str, float]:
    """Each A–F run's rate as the records bind it, its arm the frozen one."""
    by_id = {e.run_id: e for e in endpoints}
    rates = {}
    for block in freeze["comparisons"]["A-F"]["blocks"]:
        for arm, entry in block["arms"].items():
            for side in fx.SIDES:
                run = entry[side]
                e = by_id.get(run)
                if e is None or e.arm != arm or not isinstance(e.lr, float):
                    refuse("fix", f"{run}: no endpoint of arm {arm} with its rate in the state")
                rates[run] = e.lr
    return rates


def fix(args: argparse.Namespace) -> int:
    root, study3_repo = Path(args.runs_root), Path(args.study3_repo)
    code = record_code("fix", REPO)
    try:
        overlap_data = Path(args.surrogate_overlap).read_bytes()
        overlap = fx.surrogate_overlap(json.loads(overlap_data))
    except (OSError, ValueError) as exc:
        refuse("fix", f"the surrogate overlap does not read or does not hold: {exc}")
    checkout = study3_checkout("fix", study3_repo)
    freeze, problems = read_freeze(CANDIDATE, runs_root=root, repo=study3_repo)
    if freeze is None:
        refuse("fix", f"study 3's analysis freeze has {len(problems)} problem(s): "
               + "; ".join(problems))  # fmt: skip
    expected = freeze["code"]["gpu_code_sha256"]
    if checkout["gpu_code_sha256"] != expected:
        refuse("fix", f"study 3's checkout runs GPU code {checkout['gpu_code_sha256'][:12]}, "
               f"not the freeze's {expected[:12]}: its readers open no label under another "
               "code")  # fmt: skip
    try:
        _, endpoints = read_state(root, repo=study3_repo, expected_code=expected,
                                  theta0=args.theta0)  # fmt: skip
    except (OSError, ValueError) as exc:
        refuse("fix", f"study 3's state cannot be read ({type(exc).__name__})")
    rates = rates_of(freeze, endpoints)
    harm = af_harm(freeze, root, study3_repo, args.sealed_dir)
    ledger = ledger_hashes("fix", sealed_dir_of(root, args.sealed_dir))
    labels = {}
    for run in sorted(harm):
        hashes = label_hashes(root / EXPERIMENT / run, JUDGE)
        if hashes["receipt_sha256"] != harm[run].receipt_sha256 or not hashes["labels_sha256"]:
            refuse("fix", f"{run}: its label files are not the ones the reading accepted")
        labels[run] = hashes
    blocks = freeze["comparisons"]["A-F"]["blocks"]
    try:
        chosen = fx.chosen_blocks(fx.extreme_blocks(fx.block_means(blocks, harm)), blocks)
    except ValueError as exc:
        refuse("fix", f"the extreme blocks cannot be chosen: {exc}")
    population, configs = None, {}
    for run in fx.relabel_runs(chosen):
        try:
            ids, configs[run] = run_prompt_ids(root / EXPERIMENT / run)
        except FileNotFoundError:
            refuse("fix", f"{run}: its config or its harmful prompt set is not found: the prompt "
                   "sets are read from DATA_DIR (or the checkout's data/); set it")  # fmt: skip
        except (OSError, ValueError, KeyError) as exc:
            refuse("fix", f"{run}: its prompt ids cannot be read ({type(exc).__name__})")
        if population is not None and ids != population:
            refuse("fix", f"{run}: its prompt ids are not the other relabeled runs'")
        population = ids
    freeze_data = (study3_repo / freeze_name(CANDIDATE)).read_bytes()
    inputs = {
        "study3_freeze": {"path": freeze_name(CANDIDATE), "sha256": sha256(freeze_data),
                          "root": "study3 checkout"},
        "study3_checkout": {**checkout,
                            "supplies": "the label gate's GPU-code identity, the committed freeze "
                                        "and its validation runs; the readers run from "
                                        f"code.commit {code['commit']}"},
        "state": {"bound_by": "the freeze's sha256 (its blocks' run ids, weights and code) and, "
                              "per cell, the run id and the lr the state's records give it, "
                              "recorded in reference_sample.cells_drawn; the state's run records "
                              "are not hashed one by one"},
        "theta0": {"run": rel_to(args.theta0, root), "root": "runs",
                   "bound_by": "its run id (the state's baseline for its utility checks); no "
                               "value of it enters the record"},
        "ledger": ledger,
        "labels": {"judge": JUDGE, "sealed_dir": Path(args.sealed_dir).name if args.sealed_dir
                   else "sealed", "count": len(labels), "sha256": sha256(encoded(labels)),
                   "runs": labels},
        "run_configs": configs,
        "prompt_ids": {"count": len(population), "sha256": ids_sha256(population)},
        "surrogate_overlap": {"path": Path(args.surrogate_overlap).name,
                              "sha256": sha256(overlap_data), "root": "given"},
    }  # fmt: skip
    try:
        record = fx.fixation_record(freeze=freeze, harm=harm, rates=rates, prompt_ids=population,
                                    inputs=inputs, code=code, overlap=overlap)  # fmt: skip
    except ValueError as exc:
        refuse("fix", f"the record cannot be built: {exc}")
    data = encoded(record)
    out = Path(args.out)
    if out.exists() or out.is_symlink():
        if out.read_bytes() != data:
            refuse("fix", f"{out.name} exists and is different: never overwritten")
        print(f"{out.name} unchanged, sha256 {sha256(data)}")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.name}.partial")
    partial.write_bytes(data)
    partial.replace(out)
    for group in fx.GROUPS:
        for block in record["extreme_blocks"][group]:
            print(f"{group} {block['seed']} mean {block['mean']:.6f} A {block['A']:.6f} "
                  f"F {block['F']:.6f}")  # fmt: skip
    print(f"blocks {len(blocks)} runs {len(harm)} relabel {len(fx.relabel_runs(chosen))} "
          f"cells {len(record['reference_sample']['cells_drawn'])}")  # fmt: skip
    print(f"{out.name} sha256 {sha256(data)}")
    return 0


# ------------------------------------------------------------------ reference-input


def input_folder(folder: Path, root: Path, study3_repo: Path) -> Path:
    """The input's folder: never where Git could track it (this checkout, or study 3's outside
    its runs/), never under a sealed folder, written once."""
    found = folder.resolve()
    other = study3_repo.resolve()
    if in_tracked_tree(folder) or (found.is_relative_to(other)
                                   and not found.is_relative_to(other / "runs")):  # fmt: skip
        refuse("reference-input", f"{folder} lies where Git could track it: the input holds model "
               "responses and never enters Git")  # fmt: skip
    if under_sealed(folder, root):
        refuse("reference-input", f"{folder} resolves under a sealed folder: never written")
    if folder.exists() or folder.is_symlink():
        refuse("reference-input", f"{folder} exists: the input is written once")
    return folder


def latest_analysis(root: Path) -> Path:
    """Study 3's latest analysis of candidate 1 (analysis.json, then analysis-<k>.json)."""
    folder = root / study3_final.FINAL / f"c{CANDIDATE}"

    def name(k: int) -> str:
        return "analysis.json" if k == 1 else f"analysis-{k}.json"

    k = 0
    while (folder / name(k + 1)).is_file():
        k += 1
    if not k:
        refuse("reference-input", "the blind texts wait for study 3's final analysis: no "
               f"{study3_final.FINAL}/c{CANDIDATE}/analysis.json")  # fmt: skip
    return folder / name(k)


def reference_input(args: argparse.Namespace) -> int:
    root, study3_repo = Path(args.runs_root), Path(args.study3_repo)
    folder = input_folder(Path(args.out), root, study3_repo)
    fixation_data = Path(args.fixation).read_bytes()
    record = json.loads(fixation_data)
    if not (isinstance(record, dict) and record.get("kind") == fx.KIND):
        refuse("reference-input", f"{args.fixation} is not study 5's fixation record")
    sample = record["reference_sample"]
    cells = sample["cells_drawn"]
    prompts: dict[str, list[str]] = {}
    for cell in cells:
        if cell["run_id"] in prompts:
            refuse("reference-input", f"{cell['run_id']}: two cells of one run")
        prompts[cell["run_id"]] = list(cell["prompt_ids"])
    analysis = latest_analysis(root)
    try:
        texts = study3_final.blind_texts(prompts, candidate=CANDIDATE, runs_root=root,
                                         repo=study3_repo, analysis=analysis)  # fmt: skip
    except SealedPoint as exc:
        refuse("reference-input", str(exc))
    except (OSError, ValueError) as exc:
        refuse("reference-input", f"the blind texts cannot be read ({type(exc).__name__})")
    items = [{"cell": i, "prompt": pid} for i, cell in enumerate(cells)
             for pid in cell["prompt_ids"]]  # fmt: skip
    audit = audit_script()
    named = audit.ordered(items, sample["order"]["seed"])
    unique, key = [], {}
    item_ids: dict[int, list[str]] = {i: [] for i in range(len(cells))}
    for item_id, entry in named:
        cell = cells[entry["cell"]]
        text = texts[cell["run_id"]][entry["prompt"]]
        row = {"item_id": item_id, "prompt": text["prompt"], "response": text["response"]}
        try:
            study3_audit.blind_checked(row, study3_audit.BLIND)
        except ValueError as exc:
            refuse("reference-input", str(exc))
        unique.append(row)
        key[item_id] = {"cell": entry["cell"], "prompt": entry["prompt"],
                        **{k: cell[k] for k in ("group", "rank", "seed", "arm", "side", "lr",
                                                "run_id", "weight")}}  # fmt: skip
        item_ids[entry["cell"]].append(item_id)
    files = {
        "unique.jsonl": "".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n"
                                for u in unique).encode("utf-8"),
        "key.json": encoded(key),
    }  # fmt: skip
    analysis_data = analysis.read_bytes()
    manifest = {
        "kind": INPUT_KIND,
        "fixation": {"path": FIXATION, "sha256": sha256(fixation_data)},
        "analysis": {"path": f"{study3_final.FINAL}/c{CANDIDATE}/{analysis.name}",
                     "sha256": sha256(analysis_data), "root": "runs"},
        "order": sample["order"],
        "items": len(unique),
        "cells": [{"cell": i, **{k: cell[k] for k in ("group", "rank", "seed", "arm", "side",
                                                       "lr", "run_id")},
                   "item_ids": item_ids[i]} for i, cell in enumerate(cells)],
        "files": {name: {"items": len(key) if name == "key.json" else data.count(b"\n"),
                         "sha256": sha256(data)} for name, data in files.items()},
        "annotation": sample["annotation"],
    }  # fmt: skip
    files["manifest.json"] = encoded(manifest)
    audit.write_folder(folder, files)
    print(folder)
    print(f"cells {len(cells)} items {len(unique)} files {len(files)}")
    print(f"manifest.json sha256 {sha256(files['manifest.json'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fix")
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--study3-repo", type=Path, required=True)
    p.add_argument("--theta0", type=Path, required=True)
    p.add_argument("--sealed-dir", type=Path)
    p.add_argument("--surrogate-overlap", type=Path, required=True)
    p.add_argument("--out", type=Path, default=REPO / FIXATION)
    p = sub.add_parser("reference-input")
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--study3-repo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fixation", type=Path, default=REPO / FIXATION)
    args = parser.parse_args(argv)
    return {"fix": fix, "reference-input": reference_input}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
