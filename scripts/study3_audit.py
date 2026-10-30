"""Draw study 3's audit of the judge once the judge-based analysis is written, then take the
reference labellers' agreements and estimate the judge's implied bias and error rates.

    python scripts/study3_audit.py draw       --runs-root DIR --candidate N
    python scripts/study3_audit.py agreements --runs-root DIR --candidate N --gpt FILE --grok FILE
    python scripts/study3_audit.py estimate   --runs-root DIR --candidate N --reference FILE

docs/study3-preregistration.md: "Judge and audit" ("The audit", descriptive: one terminal draw
after the judge-based analysis, its frame, its draw, the blind reference labels, the researcher's
resolution and check of 10 % of the agreements, the estimates, the undetermined rates, no
reference-harm claim) and "Seeds and order" (order step 12, under the confirmation supplement's
audit seed and agreement seed). The computations are ``analysis.study3_audit``'s; the labels and
the texts are read through the one label reader of the final analysis (``study3_final``).

The audit's folder is RUNS_DIR/study3-final/c<N>/audit/, beside the analysis.

**draw** draws the audit once:
1. the analysis freeze, read by the gate's own reading (``study3_freeze.read_freeze``), with no
   problem; a clean checkout of a known commit (the records' code); the latest analysis
   (``analysis.json``, or a recomputation's ``analysis-<k>.json``), candidate N's, naming the
   committed freeze: the one terminal draw comes after the judge-based analysis;
2. the in-force confirmation supplement's ``audit`` (``study3_versions.read_confirmation_fixation``;
   its link, ``study3_freeze.in_force_supplement``): the seed, the agreement seed and the
   registered procedure;
3. the per-item labels of every bracket run of the frozen comparisons (``study3_final.item_labels``,
   their counts those of the runs' harm), the frame (``study3_audit.audit_frame``) and, per
   estimable comparison, its judge point Σ c_i·y_J,i, which must be the analysis's primary mean to
   1e-12 (a comparison without frame items: the analysis has no primary mean either);
4. the draw (``study3_audit.draw``) under the audit seed, and the blind texts of the distinct
   items drawn (``study3_final.blind_texts``: the prompt and the step-25 response), in the order
   of the stream ``stream(seed, "audit", "order")``, each with an opaque id from that stream;
5. the files, written into a temporary sibling folder, each synced, then renamed to audit/:
   - ``unique.jsonl``: ``{item_id, prompt, response}`` per distinct item drawn, in that order: the
     only file the reference labellers see, blind to arm, run, block and judge label;
   - ``key.json``: per item id its origin (frame index, arm, seed, run, side, prompt id, weight),
     its judge label, cohorts, coefficients, a_i, q_i, A_a and multiplicity;
   - ``frame.jsonl``: every frame item, the same fields, its multiplicity (0 when not drawn) and
     its item id when drawn;
   - ``selection.json``: the record ``kind`` (``study3-audit-selection``), ``candidate``,
     ``written_at``, ``code``, ``freeze``, ``analysis`` (its path under RUNS_DIR, sha256 and
     ``root: runs``), ``supplement``, ``audit``, ``sizes``, ``cohorts`` (blocks per comparison),
     per arm its frame, A_a, draws and distinct items or the reason it is not drawn (``arms``),
     the draws, the judge points, the non-estimable comparisons with their reasons, and the item
     count and sha256 of every other file (``files``).
   It prints counts only.

**agreements** reads GPT's and Grok's labels (``--gpt``, ``--grok``: JSON objects ``{item_id:
label | null}`` over exactly unique.jsonl's item ids; a label one of refusal, safe_non_refusal or
harmful_compliance; null a failed output) and writes ``agreements.json`` once: the agreements,
the disagreements the researcher resolves, the failed outputs, and the check, ⌈10 % ×
agreements⌉ of the agreements drawn under the agreement seed (``study3_audit.agreements``). It
prints the counts and the item ids to resolve and to check.

**estimate** reads the resolved reference labels (``--reference``: ``{item_id: label | null}``
over every item id; null a missing reference, failed or unresolved), once the agreements are
written. It checks the draw's files against selection.json's sha256 and the in-force audit
against selection.json's, draws again from frame.jsonl (the draw must be the stored one, its
items and multiplicities key.json's, its judge points selection.json's), and writes
``estimate.json`` once: B_E per estimable comparison (missing at random, and its worst-case
bound), the error rates per arm drawn, the non-estimable comparisons and the missing references
(``study3_audit.estimates``), naming selection.json, agreements.json and the reference file by
sha256. It prints those numbers.

**Refusals** exit with ``study3_audit.py <command>: <why>``, writing nothing more: the audit's
folder where Git could track it or under a sealed folder; a draw already made; no analysis, or one
of another freeze or candidate; a freeze with a problem, a dirty checkout, the in-force
supplement missing or not the registered procedure; a cohort run pending or a technical labeling
problem; per-item labels whose counts are not the run's harm; a judge point that is not the
analysis's mean; a file the draw did not write; labels that do not cover exactly the items; a
record already written. Messages name ids, counts, paths and hashes, never a label or a text.

**The sealing boundary.** The labels, the prompts and the responses are opened inside
``study3_final`` alone, behind the freeze gate; the blind texts only once the analysis is written.
Only unique.jsonl holds texts: the audit's folder lies under RUNS_DIR, never where Git could track
it (``study2_label_cli.in_tracked_tree``), its folder 0700 and its files 0600. Nothing printed
holds a text.

**Rulings** where the registration leaves a detail open:
- *Item ids* are opaque, ``c`` and 12 hexadecimal digits from the order stream: neither the id nor
  the order tells the arm, run, block or judge label.
- *One item per drawn (run, prompt)*: each distinct item drawn is one line of unique.jsonl, labeled
  once, counted once per draw.
- *The estimate follows the agreements*: it is refused until agreements.json is written, and names
  it.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import study2_labels, study3_audit, study3_final
from zero_field_refusal.analysis.study2_label_cli import in_tracked_tree
from zero_field_refusal.analysis.study3_decisions import _under_sealed
from zero_field_refusal.analysis.study3_freeze import freeze_name, in_force_supplement, read_freeze
from zero_field_refusal.analysis.study3_inference import Interval, stream
from zero_field_refusal.analysis.study3_labels import (
    LabelingProblem,
    LedgerError,
    SealedPoint,
)
from zero_field_refusal.analysis.study3_records import EXPERIMENT
from zero_field_refusal.analysis.study3_versions import read_confirmation_fixation
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.tracking import git_state

REPO = REPO_ROOT  # the checkout: the freeze record and the supplement
FINAL = study3_final.FINAL  # RUNS_DIR/study3-final/c<N>/: the analysis and the audit
AUDIT = "audit"
DRAWN_FILES = ("unique.jsonl", "key.json", "frame.jsonl")  # selection.json names each
SELECTION, AGREEMENTS, ESTIMATE = "selection.json", "agreements.json", "estimate.json"
KINDS = {"selection": "study3-audit-selection", "agreements": "study3-audit-agreements",
         "estimate": "study3-audit-estimate"}  # fmt: skip
TOLERANCE = 1e-12  # a judge point against the analysis's primary mean
DRAWN = ("q", "A", "multiplicity", "item_id")  # frame.jsonl's fields of the draw
KEY_FIELDS = ("arm", "seed", "run", "side", "prompt", "w", "judge", "y_judge", "cohorts", "c",
              "a")  # fmt: skip


def refuse(command: str, why: str) -> NoReturn:
    sys.exit(f"study3_audit.py {command}: {why}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def encoded(record: Any) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def head_code(command: str) -> dict[str, str]:
    """HEAD's commit and its GPU code, as a record's ``code`` names them, from a clean
    checkout."""
    state = git_state(REPO)
    if not state.get("commit") or state.get("dirty") is not False:
        refuse(command, f"the record's code needs a clean checkout of a known commit, not {state}")
    identity = code_identity.gpu_code_identity(state["commit"], REPO)
    return {"commit": state["commit"], "gpu_code_sha256": identity["sha256"]}


def written(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path: Path, data: bytes) -> None:
    """``data`` at ``path``, created new (0600), synced; ``FileExistsError`` if it exists."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view) :]
        os.fsync(fd)
    finally:
        os.close(fd)
    _fsync_dir(path.parent)


def publish(command: str, path: Path, data: bytes) -> None:
    try:
        write_new(path, data)
    except FileExistsError:
        refuse(command, f"{path.name} is written once: it exists already")


# ------------------------------------------------------------------ the audit's folder


def audit_folder(command: str, root: Path, candidate: int) -> Path:
    """RUNS_DIR/study3-final/c<N>/audit/, refused where Git could track it or under a sealed
    folder."""
    folder = root / FINAL / f"c{candidate}" / AUDIT
    found, repo = folder.resolve(), REPO.resolve()
    tracked = in_tracked_tree(folder) or (found.is_relative_to(repo)
                                          and not found.is_relative_to(repo / "runs"))  # fmt: skip
    if tracked:
        refuse(command, f"{folder} lies where Git could track it: the audit's files hold model "
               "responses and never enter Git; keep them under RUNS_DIR")  # fmt: skip
    if _under_sealed(folder, root):
        refuse(command, f"{folder} resolves under a sealed folder: never written")
    return folder


def supplement(command: str, root: Path, candidate: int) -> tuple[dict[str, Any], dict]:
    """The in-force supplement's ``audit`` (its seed, agreement seed and procedure, the
    registered one) and its link."""
    record, problems = read_confirmation_fixation(candidate, runs_root=root, repo=REPO)
    if record is None:
        refuse(command, f"candidate {candidate}'s confirmation supplement is "
               + (f"not valid ({problems[0]})" if problems else "missing")
               + ": it fixes the audit's seeds")  # fmt: skip
    link, problems = in_force_supplement(candidate, runs_root=root, repo=REPO)
    if link is None:
        refuse(command, "the in-force confirmation supplement cannot be named: "
               + ("; ".join(problems) or "missing"))  # fmt: skip
    audit = record["audit"]
    if audit.get("procedure") != study3_audit.PROCEDURE:
        refuse(command, f"the supplement's audit procedure {audit.get('procedure')!r} is not the "
               f"registered {study3_audit.PROCEDURE!r}")  # fmt: skip
    return {key: audit[key] for key in ("seed", "agreement_seed", "procedure")}, link


# ------------------------------------------------------------------ draw


def latest_analysis(root: Path, candidate: int,
                    frozen: Mapping[str, str]) -> tuple[Path, dict, bytes]:  # fmt: skip
    """The latest analysis of candidate N (analysis.json, then analysis-<k>.json), naming the
    committed freeze ``frozen``."""
    folder = root / FINAL / f"c{candidate}"

    def name(k: int) -> str:
        return "analysis.json" if k == 1 else f"analysis-{k}.json"

    k = 0
    while written(folder / name(k + 1)):
        k += 1
    if not k:
        refuse("draw", f"the audit's one terminal draw comes after the judge-based analysis: no "
               f"{FINAL}/c{candidate}/analysis.json yet")  # fmt: skip
    path = folder / name(k)
    try:
        data = path.read_bytes()
        record = json.loads(data)
    except (OSError, ValueError) as exc:
        refuse("draw", f"{path} cannot be read ({type(exc).__name__})")
    same = (isinstance(record, dict) and record.get("kind") == study3_final.ANALYSIS_KIND
            and record.get("candidate") == candidate
            and record.get("freeze") == frozen)  # fmt: skip
    if not same:
        refuse("draw", f"{path} is not candidate {candidate}'s analysis of the committed freeze "
               f"{frozen['path']}: the draw comes after the judge-based analysis")  # fmt: skip
    return path, record, data


def frame_labels(freeze_: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """The per-item labels of every bracket run of the frozen comparisons, through the final
    reader; every refusal names ids and counts, or an error's type alone."""
    blocks = [b for entry in freeze_["comparisons"].values() for b in entry["blocks"]]
    experiment = root / EXPERIMENT
    runs = {run: experiment / run for run in study3_final.runs_of(blocks)}
    try:
        return study3_final.item_labels(runs, candidate=freeze_["candidate"], runs_root=root,
                                        repo=REPO)  # fmt: skip
    except (study3_final.Pending, study3_final.ItemsRefused, SealedPoint) as exc:
        refuse("draw", str(exc))
    except (study2_labels.LabelsPending, LabelingProblem, LedgerError) as exc:
        refuse("draw", f"the labels are not read ({type(exc).__name__})")
    except (OSError, ValueError) as exc:
        refuse("draw", f"the per-item labels cannot be read ({type(exc).__name__})")


def judge_points(frame: Sequence[Mapping[str, Any]], analysis: Mapping[str, Any]) -> dict:
    """Per estimable comparison its judge point, which must be the analysis's primary mean to
    1e-12; a comparison without frame items must have no primary mean in the analysis."""
    held = study3_audit.held(frame)
    out = {}
    for e, key in study3_audit.FROZEN.items():
        entry = (analysis.get("comparisons") or {}).get(key) or {}
        primary = entry.get("primary")
        mean = None if primary is None else Interval.from_json(primary).mean
        if e not in held:
            if mean is not None:
                refuse("draw", f"{key} has no frame item, but the analysis has its primary mean")
            continue
        point = study3_audit.judge_point(frame, e)
        if mean is None or not abs(point - mean) <= TOLERANCE:
            refuse("draw", f"the frame's judge point of {key}, {point!r}, is not the analysis's "
                   f"primary mean {mean!r}")  # fmt: skip
        out[e] = point
    return out


def ordered(items: Sequence[Mapping[str, Any]], seed: int) -> list[tuple[str, Mapping]]:
    """The distinct items drawn, in the order of ``stream(seed, "audit", "order")``, each with an
    opaque id from the same stream."""
    rng = np.random.default_rng(stream(seed, "audit", "order"))
    out, used = [], set()
    for j in rng.permutation(len(items)):
        item_id = f"c{int(rng.integers(0, 2**48)):012x}"
        while item_id in used:
            item_id = f"c{int(rng.integers(0, 2**48)):012x}"
        used.add(item_id)
        out.append((item_id, items[int(j)]))
    return out


def draw(args: argparse.Namespace) -> None:
    """The audit's draw (module docstring, "draw")."""
    root, candidate = Path(args.runs_root), args.candidate
    folder = audit_folder("draw", root, candidate)
    if written(folder):
        refuse("draw", f"the audit has one terminal draw: {folder} exists")
    # 1. the freeze, a clean checkout, the analysis
    freeze_, problems = read_freeze(candidate, runs_root=root, repo=REPO)
    if freeze_ is None:
        refuse("draw", f"the analysis freeze has {len(problems)} problem(s): "
               + "; ".join(problems))  # fmt: skip
    code = head_code("draw")
    name = freeze_name(candidate)
    frozen = {"path": name, "sha256": sha256((REPO / name).read_bytes()), "root": "repo"}
    analysis_path, analysis, analysis_data = latest_analysis(root, candidate, frozen)
    # 2. the supplement's audit
    audit, link = supplement("draw", root, candidate)
    # 3. the labels, the frame and its judge points
    try:
        frame = study3_audit.audit_frame(freeze_, frame_labels(freeze_, root))
    except ValueError as exc:  # the frame's refusals name run ids and counts
        refuse("draw", f"the audit's frame cannot be built: {exc}")
    points = judge_points(frame, analysis)
    # 4. the draw and the blind texts
    sample = study3_audit.draw(frame, seed=audit["seed"])
    named = ordered(sample["items"], audit["seed"])
    prompts: dict[str, list[str]] = {}
    for _, entry in sorted(named, key=lambda pair: pair[1]["index"]):
        item = frame[entry["index"]]
        prompts.setdefault(item["run"], []).append(item["prompt"])
    try:
        texts = study3_final.blind_texts(prompts, candidate=candidate, runs_root=root, repo=REPO,
                                         analysis=analysis_path)  # fmt: skip
    except SealedPoint as exc:
        refuse("draw", str(exc))
    except (OSError, ValueError) as exc:
        refuse("draw", f"the blind texts cannot be read ({type(exc).__name__})")
    # 5. the files
    unique, key = [], {}
    for item_id, entry in named:
        item = frame[entry["index"]]
        text = texts[item["run"]][item["prompt"]]
        unique.append({"item_id": item_id, "prompt": text["prompt"],
                       "response": text["response"]})  # fmt: skip
        key[item_id] = {"frame_index": item["index"], **{k: item[k] for k in KEY_FIELDS},
                        **{k: entry[k] for k in ("q", "A", "multiplicity")}}  # fmt: skip
    drawn = {entry["frame_index"]: (item_id, entry) for item_id, entry in key.items()}
    rows = []
    for item in frame:
        total = sample["arms"][item["arm"]]["A"]
        item_id, entry = drawn.get(item["index"], (None, None))
        rows.append({**item, "A": total, "q": item["a"] / total if total else None,
                     "multiplicity": 0 if entry is None else entry["multiplicity"],
                     "item_id": item_id})  # fmt: skip
    files = {
        "unique.jsonl": "".join(json.dumps(u, ensure_ascii=False, sort_keys=True) + "\n"
                                for u in unique).encode("utf-8"),
        "key.json": encoded(key),
        "frame.jsonl": "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                               for row in rows).encode("utf-8"),
    }  # fmt: skip
    cohorts = {e: len(((freeze_["comparisons"].get(key_) or {}).get("blocks")) or ())
               for e, key_ in study3_audit.FROZEN.items()}  # fmt: skip
    selection = {
        "kind": KINDS["selection"], "candidate": candidate, "written_at": now(), "code": code,
        "freeze": frozen,
        "analysis": {"path": f"{FINAL}/c{candidate}/{analysis_path.name}",
                     "sha256": sha256(analysis_data), "root": "runs"},
        "supplement": link, "audit": audit, "sizes": sample["sizes"], "cohorts": cohorts,
        "arms": sample["arms"], "draws": sample["draws"], "judge_points": points,
        "non_estimable": study3_audit.non_estimable(frame),
        "files": {name_: {"items": len(key) if name_ == "key.json"
                          else data.count(b"\n"), "sha256": sha256(data)}
                  for name_, data in files.items()},
    }  # fmt: skip
    files[SELECTION] = encoded(selection)
    write_folder(folder, files)
    # what was drawn: counts only
    print(folder)
    for arm, entry in sample["arms"].items():
        found = (f"{entry['draws']} draws of {entry['frame']} items, {entry['distinct']} distinct"
                 if entry["draws"] else entry["reason"])  # fmt: skip
        print(f"{arm}: {found}")
    print(f"unique.jsonl: {len(unique)} items for the reference labellers")


def write_folder(folder: Path, files: Mapping[str, bytes]) -> None:
    """The files in a temporary sibling folder (0700), each written new and synced, then the
    folder renamed to ``folder``, refused if it exists; on any failure the temporary folder is
    removed."""
    folder.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{folder.name}-", dir=folder.parent))
    try:
        for name, data in files.items():
            write_new(temporary / name, data)
        if written(folder):
            refuse("draw", f"the audit has one terminal draw: {folder} exists")
        os.rename(temporary, folder)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    _fsync_dir(folder.parent)


# ------------------------------------------------------------------ the draw's files


def read_draw(command: str, folder: Path) -> dict[str, Any]:
    """selection.json, its sha256 and key.json, each drawn file checked against selection.json's
    sha256."""
    path = folder / SELECTION
    if not path.is_file():
        refuse(command, f"no draw yet: {path} is missing")
    try:
        data = path.read_bytes()
        selection = json.loads(data)
        for name in DRAWN_FILES:
            if sha256((folder / name).read_bytes()) != selection["files"][name]["sha256"]:
                refuse(command, f"{folder / name} is not the file the draw wrote")
        key = json.loads((folder / "key.json").read_bytes())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        refuse(command, f"the draw's files cannot be read ({type(exc).__name__})")
    return {"selection": selection, "sha256": sha256(data), "key": key}


def labels_file(command: str, path: Path, root: Path, ids: set[str], what: str) -> dict:
    """A labels file ``{item_id: label | null}`` over exactly ``ids``."""
    if _under_sealed(path, root):
        refuse(command, f"{path} resolves under a sealed folder: never opened")
    try:
        labels = json.loads(Path(path).read_bytes())
    except (OSError, ValueError) as exc:
        refuse(command, f"{path} cannot be read ({type(exc).__name__})")
    if not isinstance(labels, dict):
        refuse(command, f"{path}: a JSON object {{item_id: label | null}}")
    if set(labels) != ids:
        absent, extra = len(ids - set(labels)), len(set(labels) - ids)
        refuse(command, f"{what} must give every item of unique.jsonl, null when missing, and no "
               f"other: {absent} absent, {extra} unknown")  # fmt: skip
    return labels


def checked_audit(command: str, root: Path, candidate: int, selection: Mapping) -> dict:
    audit, _ = supplement(command, root, candidate)
    if audit != selection.get("audit") or selection.get("candidate") != candidate:
        refuse(command, "the in-force supplement's audit is not the one the draw used")
    return audit


def file_link(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(Path(path).read_bytes())}


# ------------------------------------------------------------------ agreements


def agreements(args: argparse.Namespace) -> None:
    """The reference labellers' agreements (module docstring, "agreements")."""
    root, candidate = Path(args.runs_root), args.candidate
    folder = audit_folder("agreements", root, candidate)
    drawn = read_draw("agreements", folder)
    audit = checked_audit("agreements", root, candidate, drawn["selection"])
    path = folder / AGREEMENTS
    if written(path):
        refuse("agreements", f"{AGREEMENTS} is written once: it exists already")
    code = head_code("agreements")
    ids = set(drawn["key"])
    gpt = labels_file("agreements", args.gpt, root, ids, "--gpt")
    grok = labels_file("agreements", args.grok, root, ids, "--grok")
    try:
        found = study3_audit.agreements(gpt, grok, seed=audit["agreement_seed"])
    except ValueError as exc:  # an item id and a label outside the judge's
        refuse("agreements", str(exc))
    counts = {k: len(found[k]) for k in ("agreements", "disagreements", "failed", "check")}
    record = {"kind": KINDS["agreements"], "candidate": candidate, "written_at": now(),
              "code": code,
              "selection": {"path": f"{FINAL}/c{candidate}/{AUDIT}/{SELECTION}",
                            "sha256": drawn["sha256"], "root": "runs"},
              "gpt": file_link(args.gpt), "grok": file_link(args.grok), "counts": counts,
              **found}  # fmt: skip
    publish("agreements", path, encoded(record))
    print(path)
    print(json.dumps(counts))
    print("to resolve: " + " ".join(found["disagreements"] + found["failed"]))
    print("to check: " + " ".join(found["check"]))


# ------------------------------------------------------------------ estimate


def estimate(args: argparse.Namespace) -> None:
    """The audit's estimates (module docstring, "estimate")."""
    root, candidate = Path(args.runs_root), args.candidate
    folder = audit_folder("estimate", root, candidate)
    drawn = read_draw("estimate", folder)
    selection, key = drawn["selection"], drawn["key"]
    audit = checked_audit("estimate", root, candidate, selection)
    path = folder / ESTIMATE
    if written(path):
        refuse("estimate", f"{ESTIMATE} is written once: it exists already")
    agreed = folder / AGREEMENTS
    if not agreed.is_file():
        refuse("estimate", f"the estimate follows the agreements: {AGREEMENTS} is not written")
    code = head_code("estimate")
    resolved = labels_file("estimate", args.reference, root, set(key), "--reference")
    try:
        values = study3_audit.reference_values(resolved)
    except ValueError as exc:  # an item's label outside the judge's
        refuse("estimate", str(exc))
    frame, sample = redrawn(folder, selection, key, audit)
    reference = {entry["frame_index"]: values[item_id] for item_id, entry in key.items()}
    found = study3_audit.estimates(frame, sample, reference)
    record = {"kind": KINDS["estimate"], "candidate": candidate, "written_at": now(),
              "code": code,
              "selection": {"path": f"{FINAL}/c{candidate}/{AUDIT}/{SELECTION}",
                            "sha256": drawn["sha256"], "root": "runs"},
              "agreements": {"path": f"{FINAL}/c{candidate}/{AUDIT}/{AGREEMENTS}",
                             "sha256": sha256(agreed.read_bytes()), "root": "runs"},
              "reference": file_link(args.reference), **found}  # fmt: skip
    publish("estimate", path, encoded(record))
    print(path)
    print(json.dumps(found, indent=1, allow_nan=False))


def redrawn(folder: Path, selection: Mapping[str, Any], key: Mapping[str, Any],
            audit: Mapping[str, Any]) -> tuple[list[dict], dict]:  # fmt: skip
    """The frame read from frame.jsonl and its draw made again under the audit seed: refused
    unless the draw, its items' multiplicities and its judge points are the stored ones."""
    try:
        rows = [json.loads(line) for line in (folder / "frame.jsonl").read_bytes().splitlines()]
    except (OSError, ValueError) as exc:
        refuse("estimate", f"frame.jsonl cannot be read ({type(exc).__name__})")
    frame = [{k: v for k, v in row.items() if k not in DRAWN} for row in rows]
    try:
        sample = study3_audit.draw(frame, seed=audit["seed"])
    except (ValueError, KeyError, TypeError) as exc:
        refuse("estimate", f"frame.jsonl does not draw ({type(exc).__name__})")
    again = {item["index"]: item["multiplicity"] for item in sample["items"]}
    stored = {entry["frame_index"]: entry["multiplicity"] for entry in key.values()}
    if (sample["arms"], sample["draws"], again) != (selection["arms"], selection["draws"], stored):
        refuse("estimate", "the draw made again from frame.jsonl is not the stored draw")
    for e, point in selection["judge_points"].items():
        if study3_audit.judge_point(frame, e) != point:
            refuse("estimate", f"frame.jsonl's judge point of {e} is not selection.json's")
    return frame, sample


# ------------------------------------------------------------------ the command


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    drawn = commands.add_parser("draw", help="draw the audit once, after the analysis")
    agreed = commands.add_parser("agreements", help="the agreements the researcher checks")
    estimated = commands.add_parser("estimate", help="the bias and the error rates")
    for command in (drawn, agreed, estimated):
        command.add_argument(
            "--runs-root",
            type=Path,
            required=True,
            help="RUNS_DIR: the run records, the labels and the analysis",
        )
        command.add_argument("--candidate", type=int, choices=(1, 2), required=True)
    agreed.add_argument("--gpt", type=Path, required=True, help="GPT's labels (JSON)")
    agreed.add_argument("--grok", type=Path, required=True, help="Grok's labels (JSON)")
    estimated.add_argument("--reference", type=Path, required=True,
                           help="the resolved reference labels (JSON)")  # fmt: skip
    args = parser.parse_args(argv)
    {"draw": draw, "agreements": agreements, "estimate": estimate}[args.command](args)


if __name__ == "__main__":
    main()
