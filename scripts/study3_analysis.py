"""Freeze the cohorts and K*_h of study 3's candidate N, then analyze its harm on them, and write
the final report's facts.

    python scripts/study3_analysis.py freeze  --runs-root DIR --candidate N [--theta0 RUN_DIR]
    python scripts/study3_analysis.py analyze --runs-root DIR --candidate N [--theta0 RUN_DIR]
        [--supersedes-reason ID]
    python scripts/study3_analysis.py report  --runs-root DIR --billing FILE --embargo FILE
        [--theta0 RUN_DIR] [--out DIR]

docs/study3-preregistration.md: "Seeds and order" (order steps 9–11 and 13: the cohorts and K*_h
frozen before any confirmation label is opened, the labeling of the confirmation, the judge-based
analysis, the final report), "Questions and estimands" (the comparisons and the descriptives,
validation's harm after the final analysis and apart from it), "Inference", "Utility" (the claim at
confirmation, at α = 0.05/8 per bound), "Validation, confirmation and stopping" ("Confirmation"),
"Judge and audit" (the denominator 521, the three computations, "Sealing"), "Voiding and
supersession", "Disclosure" and amendments 4–6 (what the final report states).

**freeze** writes candidate N's analysis freeze (``study3_freeze.analysis_freeze``: the cohorts,
A − P's blocks, K*_h and the matched blocks, validation's blocks, from the study's state at
``done``, θ₀ from --theta0; the labeling ledger's run ids, ``study3_labels.ledger_runs``, for the
check that no confirmation label exists yet) at docs/study3-analysis-freeze-c<N>.json, once. It
needs a clean checkout of a known commit: the record's code is HEAD's commit and that commit's GPU
code. It prints the record's path; the record is then committed alone, before any confirmation
label. It reads no label: of the sealed folders it reads the labeling ledger's run ids alone
(through ``ledger_runs``), and nothing of a run's.

**analyze** runs the judge-based analysis on the frozen cohorts:
1. the analysis freeze, read by the gate's own reading (``study3_freeze.read_freeze``), with no
   problem, then a clean checkout of a known commit (the records' code);
2. the in-force records, each through its reader: the confirmation supplement and its link from
   one reading (``study3_freeze.in_force_supplement``, by
   ``study3_versions.read_confirmation_fixation``: the bootstrap's seed and resamples), the
   validation fixation (``study3_versions.read_validation_fixation``: the level L) and the n
   record (``study3_decisions.decisions``: validation's A–F cohort, which must be the frozen
   one); κ per arm from the simulation fixation's ``curvature`` (``simulation_curvature``);
3. the outputs so far (below: a resumed or a superseding run);
4. the utility gates over each comparison's frozen blocks at α = 0.05/8 per bound
   (``study3_blocks.utility_gate``, from the state's endpoints, the frozen runs checked), which
   read no label; then the labels, through the one label reader (``study3_final.gather``): the
   spread of K of every bracket run of the comparisons, then the harm of every run of the frozen
   comparisons, A − P's blocks and the matched blocks; no validation run is read yet;
5. ``study3_estimates.analysis``;
6. ``analysis.json``, written; it prints its path, and each comparison's n and the row of the
   registered wording;
7. only then validation's labels: the harm of validation's frozen runs, read through the final
   reader (``study3_final.cohort_harm``), and validation's harm
   (``study3_estimates.validation_harm``) in ``validation-harm.json``, which names the analysis
   by its sha256; it prints that path.

**report** writes the final report's facts (``study3_report.report_facts``) and their tables
(``study3_report.tables``) once, into RUNS_DIR/study3-final/report-<UTC>/ (or ``--out``, a folder
that does not exist yet): ``facts.json`` and ``tables.md``. The prose of the report is the
researcher's. Its inputs, gathered label-free:
1. the researcher's files: the billing file (``--billing``, ``study3_report.read_billing``: actual
   spending by stage, each incident with its billed price and evidence, the stages an internal
   spending policy stopped) and the embargo file (``--embargo``, ``study3_report.read_embargo``:
   the breaches, or none);
2. the study's state under this checkout's GPU code (``read_state``) with its endpoints; the
   decision records (``read_decision``: engineering, simulation, the candidate's ``n-c<N>`` and
   ``budget-c<N>``), the registration at HEAD (``registered_policy``), the engineering verdict's
   summary (``qualification/c1/engineering/verdict.json``), the simulation's per-test results
   (``study3_report.simulation_results``) and each candidate's qualification
   (``qualification_records.read_qualification`` under the GPU code in force for its runs);
3. the analysis freeze (``read_freeze``, no problem) when it is written;
4. the labeling ledger's answers, through the label reader's ledger readers alone:
   ``labeling_problems`` (the runs with a technical labeling problem) and ``diagnoses`` (the
   diagnosis records; a ``judge`` one makes the report the ``judge_technical`` variant);
5. the latest analysis and its validation harm (each naming the committed freeze by its sha256,
   validation's harm naming the analysis by its sha256), and the audit's estimate (naming its
   selection and agreements by their sha256, the selection naming the freeze and an analysis of
   the candidate by theirs): the records as written, never a label;
6. the incidents of the records: the launch journal's (``study3_report.journal_incidents``) and
   the qualification's ``--infrastructure`` records (``study3_report.qualification_incidents``),
   each of which needs a billed incident.
It refuses a stage that is neither ``done`` nor terminal (unless a judge diagnosis calls for the
``judge_technical`` variant), ``done`` without the analysis, validation's harm and the audit's
estimate, an incident with no billed price, and a record that does not name what it rests on. It
prints the two files' paths. It opens no label, no response and nothing of the labeling ledger but
through ``labeling_problems`` and ``diagnoses``.

**Refusals** exit with ``study3_analysis.py <command>: <why>``, writing nothing more: no analysis
freeze, one with a problem (not committed, rewritten, not the current records' …) or a dirty
checkout; an in-force record missing or invalid; a pending run or a technical labeling problem in
the cohorts (``<k> runs pending: <ids>; <j> runs with a technical labeling problem: <ids>``: the
analysis waits for both, and neither counts as missing labels); a label that cannot be read (its
exception's type alone); a spread of K whose samples are not the endpoint's K; a frozen run no
longer the state's. The messages name ids, counts, paths and hashes, never a label or a text.

**Outputs**, under RUNS_DIR/study3-final/c<N>/, outside RUNS_DIR/study3/ (whose walk would read a
new folder as a run's): ``analysis.json``, the record ``kind`` (``study3-analysis``),
``candidate``, ``written_at``, ``code``, ``freeze`` (its path, sha256 and ``root: repo``),
``supplement``, ``level``, ``bootstrap`` (``seed``, ``resamples``), ``kappa`` (the simulation
fixation's path, sha256 and root, κ per arm, its unit and source), ``superseded`` (null) and the
analysis (``comparisons``, ``utility``, ``curvature``, ``k_spread``); ``validation-harm.json``:
``kind`` (``study3-validation-harm``), ``candidate``, ``written_at``, ``code``, ``freeze``,
``analysis`` (its path under RUNS_DIR, sha256 and ``root: runs``), ``level``, ``bootstrap`` and
``comparisons``. Every interval is ``Interval.to_json``'s and the missing-labels envelope its own
``to_json``; every record is strict JSON (``allow_nan=False``), written whole and only where none
is (``study3_decisions._publish``).

**Restart and supersession.** Each record is written once. A run with the analysis written and
validation's harm not yet written (a validation run pending at step 7, say) writes validation's
harm alone, from the same freeze and in-force records. Once both are written, ``analyze`` refuses
unless --supersedes-reason gives an id (a lowercase letter, then lowercase letters, digits or
underscores): a recomputation after a fix of the analysis code, with no change of the GPU code, is
written as ``analysis-<k>.json`` and ``validation-harm-<k>.json`` (k = 2, 3, …), its ``superseded``
naming its predecessor by path, sha256 and the reason.

**The sealing boundary.** The labels, the K rows and nothing else of a run's sealed folders are
opened, inside ``study3_final`` alone, and only once the freeze gate opens the final point; of the
labeling ledger, only run ids and states are read. What is written and printed holds numbers, ids,
run ids, paths, hashes, times and the simulation fixation's ``source``.

**Rulings** where the registration leaves a detail open:
- *Outside the experiment's folder.* The outputs are under RUNS_DIR/study3-final/c<N>/.
- *Validation's harm* is A–F on the n record's cohort and P–F on the validation blocks supported
  for P and F while P–F is active, computed as the analysis is, descriptive, after the analysis
  and apart from it: validation's labels are read only once ``analysis.json`` is written, and its
  file names the analysis by its sha256.
- *Written once.* The analysis is ``analysis.json``; a recomputation after a fix without a change
  of the GPU code is ``analysis-<k>.json``, naming its predecessor by path, sha256 and a reason id.
- *κ* is the simulation fixation's base ``curvature`` (docs/study3-simulation-fixation.json, in the
  checkout, committed once and unchanged at HEAD), pinned in the analysis by its sha256.
- *The report* ("Disclosure", amendment 5): the facts are
  numbers, ids, paths and hashes, the prose is the researcher's, and money is actual spending by
  stage plus each incident with its billed price; the report says which stages an internal
  spending policy stopped, by name alone. It is written once per folder, outside
  RUNS_DIR/study3/.
"""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import study3_estimates, study3_final, study3_report
from zero_field_refusal.analysis.study3_blocks import (
    CONFIRMATION_ALPHA,
    PAIRS,
    block_state,
    utility_gate,
)
from zero_field_refusal.analysis.study3_decisions import (  # its write-once publication
    DecisionRefused,
    _publish,
    _under_sealed,
    decision_path,
    decisions,
    read_decision,
)
from zero_field_refusal.analysis.study3_estimates import CURVATURE_UNIT
from zero_field_refusal.analysis.study3_freeze import (
    FreezeRefused,
    analysis_freeze,
    freeze_name,
    in_force_supplement,
    read_freeze,
)
from zero_field_refusal.analysis.study3_labels import (
    LabelingProblem,
    LabelsPending,
    LedgerError,
    SealedPoint,
    diagnoses,
    labeling_problems,
    ledger_runs,
    record_path,
)
from zero_field_refusal.analysis.study3_progress import COMPARISONS, code_chain, read_state
from zero_field_refusal.analysis.study3_records import EXPERIMENT
from zero_field_refusal.analysis.study3_search import ARM_ORDER
from zero_field_refusal.analysis.study3_versions import (
    _added,
    _blob,
    read_validation_fixation,
    registered_policy,
)
from zero_field_refusal.config import load_config
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.tracking import git_state

REPO = REPO_ROOT  # the checkout: the freeze record, the fixations, the records' repo evidence
FINAL = "study3-final"  # RUNS_DIR/study3-final/c<N>/: the analysis's outputs
SIMULATION_FIXATION = "docs/study3-simulation-fixation.json"
SIMULATION_KIND = "simulation-fixation"
VALIDATION_KIND = "study3-validation-harm"
REASON = re.compile(r"[a-z][a-z0-9_]{0,63}")  # a reason id, as the versions record's
ENGINEERING_VERDICT = "qualification/c1/engineering/verdict.json"  # its summary, under RUNS_DIR
AUDIT = "audit"  # RUNS_DIR/study3-final/c<N>/audit/: the audit's files
FACTS, TABLES = "facts.json", "tables.md"  # the report's files


def refuse(command: str, why: str) -> NoReturn:
    sys.exit(f"study3_analysis.py {command}: {why}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def head_code(command: str) -> dict[str, str]:
    """HEAD's commit and its GPU code (``code_identity.gpu_code_identity`` of that commit), as a
    record's ``code`` names them, from a clean checkout."""
    state = git_state(REPO)
    if not state.get("commit") or state.get("dirty") is not False:
        refuse(command, f"the record's code needs a clean checkout of a known commit, not {state}")
    identity = code_identity.gpu_code_identity(state["commit"], REPO)
    return {"commit": state["commit"], "gpu_code_sha256": identity["sha256"]}


def encoded(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def publish(command: str, path: Path, base: Path, data: bytes) -> None:
    """``data`` at ``path``, written once (``_publish``), never under a sealed folder of
    ``base``."""
    if _under_sealed(path, base):
        refuse(command, f"{path} resolves under a sealed folder: never written")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _publish(path, data)
    except DecisionRefused:
        refuse(command, f"{path} is written once: it exists already")


# ------------------------------------------------------------------ freeze


def freeze(args: argparse.Namespace) -> None:
    """The analysis freeze (module docstring, "freeze")."""
    name = freeze_name(args.candidate)
    path = REPO / name
    if path.exists() or path.is_symlink():
        refuse("freeze", f"{name} is written once: it exists already")
    code = head_code("freeze")
    try:
        ledger = ledger_runs(args.runs_root)
    except LedgerError as exc:
        refuse("freeze", f"the labeling ledger cannot be read: {exc}")
    try:
        record = analysis_freeze(args.runs_root, args.candidate, code=code, repo=REPO,
                                 theta0=args.theta0, ledger_runs=ledger)  # fmt: skip
    except FreezeRefused as exc:
        refuse("freeze", str(exc))
    publish("freeze", path, REPO, encoded(record))
    print(f"{name}: commit it alone before any confirmation label")


# ------------------------------------------------------------------ the in-force records


def simulation_curvature() -> dict[str, Any]:
    """κ per arm (module docstring, "Rulings"): the simulation fixation's base ``curvature``, its
    ``A``, ``F`` and ``P`` finite numbers in pp per (ln K)², its ``unit`` and ``source``, with the
    fixation's path, sha256 and root (``kappa``). The fixation is the file in the checkout,
    committed once and unchanged at HEAD."""
    name = SIMULATION_FIXATION
    path = REPO / name
    if _under_sealed(path, REPO):
        refuse("analyze", f"{name} resolves under a sealed folder: never opened")
    if not path.is_file():
        refuse("analyze", f"the simulation fixation {name} is missing: it gives κ per arm")
    added, why = _added(REPO, name)
    if added is None:
        refuse("analyze", f"the simulation fixation: {why}")
    data = path.read_bytes()
    if _blob(REPO, "HEAD", name) != data:
        refuse("analyze", f"the simulation fixation {name} is not HEAD's")
    try:
        record = json.loads(data)
    except ValueError:
        refuse("analyze", f"the simulation fixation {name} is no JSON record")
    curvature = record.get("curvature") if isinstance(record, dict) else None
    if not (isinstance(record, dict) and record.get("kind") == SIMULATION_KIND
            and isinstance(curvature, dict)):  # fmt: skip
        refuse("analyze", f"{name} is no simulation fixation with a curvature")
    values = {arm: curvature.get(arm) for arm in ARM_ORDER}
    finite = all(not isinstance(v, bool) and isinstance(v, int | float) and math.isfinite(v)
                 for v in values.values())  # fmt: skip
    if not (finite and curvature.get("unit") == CURVATURE_UNIT
            and isinstance(curvature.get("source"), str)):  # fmt: skip
        refuse("analyze", f"{name}: its curvature is not κ of A, F and P in {CURVATURE_UNIT!r} "
               "with its source")  # fmt: skip
    return {"path": name, "sha256": sha256(data), "root": "repo",
            "values": {arm: float(v) for arm, v in values.items()},
            "unit": curvature["unit"], "source": curvature["source"]}  # fmt: skip


def in_force(root: Path, candidate: int, freeze_: Mapping[str, Any]) -> dict[str, Any]:
    """Step 2: the supplement's bootstrap and link, L, κ, and the n record's validation cohort
    checked against the frozen validation blocks of A–F."""
    supplement, link, problems = in_force_supplement(candidate, runs_root=root, repo=REPO)
    if supplement is None or link is None:
        refuse("analyze", f"candidate {candidate}'s confirmation supplement is "
               + (f"not valid ({problems[0]})" if problems else "missing")
               + ": it fixes the bootstrap's seed")  # fmt: skip
    fixation, problems = read_validation_fixation(candidate, runs_root=root, repo=REPO)
    if fixation is None:
        refuse("analyze", f"candidate {candidate}'s validation fixation is "
               + (f"not valid ({problems[0]})" if problems else "missing")
               + ": it fixes the level L")  # fmt: skip
    entry = decisions(root, candidate, repo=REPO).get(f"n-c{candidate}") or {}
    n = entry.get("record")
    if n is None:
        refuse("analyze", f"the in-force n-c{candidate} record is {entry.get('state', 'missing')}: "
               "it gives validation's A-F cohort")  # fmt: skip
    frozen = [block["seed"] for block in freeze_["validation"]["A-F"]["blocks"]]
    if frozen != list(n["cohort"]["seeds"]):
        refuse("analyze", f"the frozen A-F validation blocks are not the n-c{candidate} record's "
               "cohort")  # fmt: skip
    bootstrap = supplement["bootstrap"]
    return {"supplement": link, "level": fixation["level"],
            "bootstrap": {"seed": bootstrap["seed"], "resamples": bootstrap["resamples"]},
            "kappa": simulation_curvature()}  # fmt: skip


# ------------------------------------------------------------------ the outputs so far


def names(k: int) -> tuple[str, str]:
    """The k-th computation's analysis and validation harm: analysis.json and
    validation-harm.json, then analysis-<k>.json and validation-harm-<k>.json."""
    suffix = "" if k == 1 else f"-{k}"
    return f"analysis{suffix}.json", f"validation-harm{suffix}.json"


def written(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def latest(folder: Path) -> int:
    """The number of the last analysis written (0 for none)."""
    k = 0
    while written(folder / names(k + 1)[0]):
        k += 1
    return k


def predecessor(path: Path, frozen: Mapping[str, str], inputs: Mapping[str, Any],
                candidate: int) -> tuple[dict[str, Any], bytes]:  # fmt: skip
    """An analysis written before: candidate N's record of this freeze, its level and bootstrap
    those in force now; refused otherwise."""
    try:
        data = path.read_bytes()
        record = json.loads(data)
    except (OSError, ValueError) as exc:
        refuse("analyze", f"{path} cannot be read ({type(exc).__name__})")
    same = (isinstance(record, dict) and record.get("kind") == study3_final.ANALYSIS_KIND
            and record.get("candidate") == candidate and record.get("freeze") == frozen
            and record.get("level") == inputs["level"]
            and record.get("bootstrap") == inputs["bootstrap"])  # fmt: skip
    if not same:
        refuse("analyze", f"{path} is not candidate {candidate}'s analysis of the freeze "
               f"{frozen['path']} under the in-force level and bootstrap")  # fmt: skip
    return record, data


# ------------------------------------------------------------------ the analysis


def labels(freeze_: Mapping[str, Any], endpoints: Sequence[Any], root: Path,
           run_dirs: Mapping[str, Path] | None = None) -> tuple[dict, dict]:  # fmt: skip
    """Step 4 (``study3_final.gather``), or, with ``run_dirs``, the harm of those runs alone
    (``study3_final.cohort_harm``: step 7's); every refusal names ids, counts and types only."""
    try:
        if run_dirs is not None:
            return study3_final.cohort_harm(run_dirs, candidate=freeze_["candidate"],
                                            runs_root=root, repo=REPO), {}  # fmt: skip
        return study3_final.gather(freeze_, endpoints, runs_root=root, repo=REPO)
    except study3_final.Pending as exc:
        refuse("analyze", str(exc))
    except (study3_final.SpreadRefused, SealedPoint) as exc:
        refuse("analyze", str(exc))
    except (LabelsPending, LabelingProblem, LedgerError) as exc:
        refuse("analyze", f"the labels are not read ({type(exc).__name__})")
    except (OSError, ValueError) as exc:
        refuse("analyze", f"the labels or the K rows cannot be read ({type(exc).__name__})")


def utility_gates(freeze_: Mapping[str, Any], state: Mapping[str, Any],
                  endpoints: Sequence[Any]) -> dict[str, dict[str, Any]]:  # fmt: skip
    """Step 4's gates, which read no label: per frozen comparison, ``utility_gate`` at
    α = 0.05/8 over its frozen blocks as the state's endpoints give them now (a run under another
    commit left out, as the freeze leaves it out), each arm's runs the frozen ones."""
    candidate = freeze_["candidate"]
    uncounted = {entry.get("run_id") for entry in state.get("not_counted") or ()}
    kept = [e for e in endpoints if e.run_id not in uncounted]
    active = freeze_["confirmation"]["active"]
    arms = [arm for arm in ARM_ORDER if arm != "P" or "P-F" in active]
    gates = {}
    for key, entry in freeze_["comparisons"].items():
        name = COMPARISONS[key]
        blocks = []
        for frozen_block in entry["blocks"]:
            seed = frozen_block["seed"]
            try:
                found = block_state(kept, seed, candidate, arms)
            except ValueError as exc:
                refuse("analyze", f"{key}: block {seed}: {exc}")
            for arm in PAIRS[name]:
                support, frozen = found["arms"][arm], frozen_block["arms"][arm]
                runs = ((support["lo"].run_id, support["hi"].run_id) if support["supported"]
                        else None)  # fmt: skip
                if runs != (frozen["lo"], frozen["hi"]):
                    refuse("analyze", f"{key}: block {seed}'s {arm} is not the frozen one now")
            blocks.append(found)
        gates[key] = utility_gate(blocks, name, CONFIRMATION_ALPHA)
    return gates


def validation_record(candidate: int, code: Mapping[str, str], frozen: Mapping[str, str],
                      inputs: Mapping[str, Any], freeze_: Mapping[str, Any],
                      harm: Mapping[str, Any], analysis: Path,
                      data: bytes) -> dict[str, Any]:  # fmt: skip
    """Validation's harm, naming the analysis ``analysis`` (its bytes ``data``) by its sha256."""
    bootstrap = inputs["bootstrap"]
    try:
        comparisons = study3_estimates.validation_harm(
            freeze_, harm, level=inputs["level"], seed=bootstrap["seed"],
            resamples=bootstrap["resamples"])  # fmt: skip
    except ValueError as exc:  # the estimates' refusals name run ids and counts
        refuse("analyze", f"validation's harm cannot be computed: {exc}")
    return {"kind": VALIDATION_KIND, "candidate": candidate, "written_at": now(), "code": code,
            "freeze": frozen,
            "analysis": {"path": f"{FINAL}/c{candidate}/{analysis.name}", "sha256": sha256(data),
                         "root": "runs"},
            "level": inputs["level"], "bootstrap": bootstrap,
            "comparisons": comparisons}  # fmt: skip


def analyze(args: argparse.Namespace) -> None:
    """The judge-based analysis (module docstring, "analyze")."""
    root, candidate = Path(args.runs_root), args.candidate
    reason = args.supersedes_reason
    if reason is not None and not REASON.fullmatch(reason):
        refuse("analyze", "--supersedes-reason is an id (a lowercase letter, then lowercase "
               "letters, digits or underscores)")  # fmt: skip
    # 1. the freeze, then a clean checkout
    freeze_, problems = read_freeze(candidate, runs_root=root, repo=REPO)
    if freeze_ is None:
        refuse("analyze", f"the analysis freeze has {len(problems)} problem(s): "
               + "; ".join(problems))  # fmt: skip
    code = head_code("analyze")
    name = freeze_name(candidate)
    frozen = {"path": name, "sha256": sha256((REPO / name).read_bytes()), "root": "repo"}
    # 2. the in-force records
    inputs = in_force(root, candidate, freeze_)
    # 3. the outputs so far
    folder = root / FINAL / f"c{candidate}"
    if _under_sealed(folder, root):
        refuse("analyze", f"{folder} resolves under a sealed folder: never written")
    k = latest(folder)
    superseded = None
    if k:
        previous, data = predecessor(folder / names(k)[0], frozen, inputs, candidate)
        if not written(folder / names(k)[1]):
            if reason is not None:
                refuse("analyze", f"{names(k)[0]} has no {names(k)[1]} yet: complete it without "
                       "--supersedes-reason first")  # fmt: skip
            resume(root, candidate, code, frozen, inputs, freeze_, folder, k, data)
            return
        if reason is None:
            refuse("analyze", f"{names(k)[0]} and {names(k)[1]} are written: a recomputation "
                   "after a fix supersedes them, with --supersedes-reason ID")  # fmt: skip
        if previous["code"]["gpu_code_sha256"] != code["gpu_code_sha256"]:
            refuse("analyze", f"the GPU code changed since {names(k)[0]}: a recomputation is one "
                   "without a change of the GPU code")  # fmt: skip
        superseded = {"path": f"{FINAL}/c{candidate}/{names(k)[0]}", "sha256": sha256(data),
                      "reason": reason}  # fmt: skip
    elif reason is not None:
        refuse("analyze", "--supersedes-reason names a recomputation: nothing to supersede yet")
    analysis_path, validation_path = (folder / part for part in names(k + 1))
    if written(validation_path):
        refuse("analyze", f"{validation_path} is written already, without its analysis")
    # 4. the label-free utility gates, then the labels of the confirmation (no validation run)
    try:
        state, endpoints = read_state(root, repo=REPO,
                                      expected_code=freeze_["code"]["gpu_code_sha256"],
                                      theta0=args.theta0)  # fmt: skip
    except (OSError, ValueError) as exc:
        refuse("analyze", f"the study's state cannot be read ({type(exc).__name__})")
    gates = utility_gates(freeze_, state, endpoints)
    harm, spread = labels(freeze_, endpoints, root)
    # 5. the analysis
    bootstrap = inputs["bootstrap"]
    kappa = inputs["kappa"]
    try:
        result = study3_estimates.analysis(
            freeze_, harm, level=inputs["level"], seed=bootstrap["seed"],
            resamples=bootstrap["resamples"], gates=gates,
            curvature={**kappa["values"], "unit": kappa["unit"], "source": kappa["source"]},
            spread=spread)  # fmt: skip
    except ValueError as exc:  # the estimates' refusals name run ids and counts
        refuse("analyze", f"the analysis cannot be computed: {exc}")
    record = {"kind": study3_final.ANALYSIS_KIND, "candidate": candidate, "written_at": now(),
              "code": code, "freeze": frozen, "supplement": inputs["supplement"],
              "level": inputs["level"], "bootstrap": bootstrap, "kappa": kappa,
              "superseded": superseded, **result}  # fmt: skip
    # 6. the analysis, written
    data = encoded(record)
    publish("analyze", analysis_path, root, data)
    print(analysis_path)
    for key, entry in result["comparisons"].items():
        print(f"{key}: n {entry['n']}, {entry['wording']['row']}")
    # 7. only then validation's labels, and validation's harm
    resume(root, candidate, code, frozen, inputs, freeze_, folder, k + 1, data)


def resume(root: Path, candidate: int, code: Mapping[str, str], frozen: Mapping[str, str],
           inputs: Mapping[str, Any], freeze_: Mapping[str, Any], folder: Path, k: int,
           data: bytes) -> None:  # fmt: skip
    """Validation's harm, once the k-th analysis (its bytes ``data``) is written (step 7, and
    the restart of module docstring, "Restart and supersession"): the harm of validation's frozen
    runs, read through the final reader only now, then ``validation-harm.json`` naming the
    analysis."""
    blocks = [block for entry in freeze_["validation"].values() if entry is not None
              for block in entry["blocks"]]  # fmt: skip
    experiment = root / EXPERIMENT
    run_dirs = {run: experiment / run for run in study3_final.runs_of(blocks)}
    harm, _ = labels(freeze_, (), root, run_dirs)
    analysis, path = (folder / part for part in names(k))
    validation = validation_record(candidate, code, frozen, inputs, freeze_, harm, analysis, data)
    publish("analyze", path, root, encoded(validation))
    print(path)


# ------------------------------------------------------------------ the report


def report(args: argparse.Namespace) -> None:
    """The final report's facts and tables (module docstring, "report")."""
    root = Path(args.runs_root)
    # 1. the researcher's files
    billing = researchers("--billing", study3_report.read_billing, args.billing)
    embargo = researchers("--embargo", study3_report.read_embargo, args.embargo)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out is not None else root / FINAL / f"report-{stamp}"
    if written(out):
        refuse("report", f"{out} exists: a report is written once, into a folder of its own")
    if _under_sealed(out, root):
        refuse("report", f"{out} resolves under a sealed folder: never written")
    # 2. the state and the label-free records
    expected = code_identity.gpu_code_identity(repo=REPO)["sha256"]
    try:
        state, endpoints = read_state(root, repo=REPO, expected_code=expected,
                                      theta0=args.theta0)  # fmt: skip
    except (OSError, ValueError) as exc:
        refuse("report", f"the study's state cannot be read ({type(exc).__name__})")
    candidate = state.get("candidate")
    try:
        policy = registered_policy(REPO)
    except ValueError as exc:
        refuse("report", f"the registration at HEAD cannot be read: {exc}")
    try:
        results = study3_report.simulation_results(root)
    except (OSError, ValueError) as exc:
        refuse("report", f"the simulation's results cannot be read: {exc}")
    kinds = ("engineering", "simulation", *(f"{k}-c{candidate}" for k in ("n", "budget")
                                            if candidate in (1, 2)))  # fmt: skip
    links = {kind: decision_link(root, kind) for kind in kinds}
    # 3. the analysis freeze
    frozen = frozen_link(root, candidate)
    records = {
        "endpoints": endpoints, "engineering": links["engineering"], "policy": policy,
        "engineering_verdict": engineering_verdict(root), "simulation": links["simulation"],
        "simulation_results": results, "qualification": qualification_readings(root, expected),
        "decisions": {kind: found for kind, found in links.items()
                      if kind.startswith(("n-", "budget-")) and found is not None},
        "freeze": frozen,
    }  # fmt: skip
    # 4. the labeling ledger's answers: the runs with a technical problem, the diagnoses
    try:
        problems = labeling_problems(root)
        found = diagnoses(root)
    except LedgerError as exc:
        refuse("report", f"the labeling ledger cannot be read: {exc}")
    judged = [d for d in found if d.cause == "judge"]
    judge = None if not judged else {
        "path": record_path(judged[0].batch), "sha256": judged[0].sha256,
        "batch": judged[0].batch, "cause": judged[0].cause, "diagnosis": judged[0].diagnosis,
        "evidence": [dict(e) for e in judged[0].evidence]}  # fmt: skip
    # 5. the analysis, validation's harm and the audit's estimate, as written
    analysis, validation, audit = final_outputs(root, candidate, frozen)
    # 6. the incidents of the records
    try:
        incidents = {"journal": study3_report.journal_incidents(root),
                     "qualification": study3_report.qualification_incidents(root)}  # fmt: skip
    except (OSError, ValueError) as exc:
        refuse("report", f"the incidents cannot be read: {exc}")
    try:
        facts = study3_report.report_facts(
            state, records=records, billing=billing, incidents=incidents,
            labeling_problems=problems, analysis=analysis, validation=validation, audit=audit,
            embargo=embargo, judge_diagnosis=judge)  # fmt: skip
    except ValueError as exc:
        refuse("report", str(exc))
    data = json.dumps(facts, indent=2, allow_nan=False, ensure_ascii=False) + "\n"  # in order
    write_report(out, {FACTS: data.encode(), TABLES: study3_report.tables(facts).encode()})
    print(out / FACTS)
    print(out / TABLES)


def researchers(option: str, reader: Any, path: Path) -> Any:
    """The researcher's file at ``path``, read by ``reader`` (its closed schema)."""
    try:
        return reader(path)
    except (OSError, ValueError) as exc:
        refuse("report", f"{option}: {exc}")


def decision_link(root: Path, kind: str) -> dict[str, Any] | None:
    """A decision record as the report names it: its path under RUNS_DIR, its sha256 and its
    record (``read_decision``); None when it is not written; refused with a problem."""
    record, problems = read_decision(root, kind, repo=REPO)
    if problems:
        refuse("report", f"the decision record {kind}: {problems[0]}")
    if record is None:
        return None
    path = decision_path(root, kind)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path.read_bytes()),
            "record": record}  # fmt: skip


def engineering_verdict(root: Path) -> dict[str, Any] | None:
    """The engineering verdict's summary (e_eng, the path maxima, the margins) with the verdict's
    path and sha256; None before the engineering stage's verdict is written."""
    path = root / ENGINEERING_VERDICT
    if not written(path):
        return None
    if _under_sealed(path, root):
        refuse("report", f"{ENGINEERING_VERDICT} resolves under a sealed folder: never opened")
    try:
        data = path.read_bytes()
        record = json.loads(data)
    except (OSError, ValueError) as exc:
        refuse("report", f"{ENGINEERING_VERDICT} cannot be read ({type(exc).__name__})")
    summary = record.get("summary") if isinstance(record, dict) else None
    if not isinstance(summary, dict):
        refuse("report", f"{ENGINEERING_VERDICT} holds no engineering summary")
    return {"path": ENGINEERING_VERDICT, "sha256": sha256(data), "summary": summary}


def qualification_readings(root: Path, expected: str) -> dict[int, dict[str, Any]]:
    """Each candidate's numerical qualification (``read_qualification``), read under the GPU code
    in force for its runs (the chain of engineering cycles from the engineering record, as the
    state reads it), for each candidate whose qualification folder exists."""
    from zero_field_refusal.analysis import qualification_records  # it loads torch: here only

    settings = load_config("experiments/qualification")
    engineering = decisions(root, 1, repo=REPO).get("engineering") or {}
    record = engineering.get("record") if engineering.get("state") in ("passed", "failed") else None
    start = ((record or {}).get("code") or {}).get("gpu_code_sha256")
    cycles = {c: qualification_records.supersessions(root, c) for c in (1, 2)}
    in_force = code_chain(start, cycles)[1]
    found = [c for c in (1, 2) if (root / "qualification" / f"c{c}" / "numerical").is_dir()]
    return {c: qualification_records.read_qualification(
                root, c, expected_code=in_force.get(c) or expected, settings=settings)
            for c in found}  # fmt: skip


def frozen_link(root: Path, candidate: Any) -> dict[str, Any] | None:
    """The committed analysis freeze, read by the gate's own reading with no problem, with its
    path and sha256; None while it is not written."""
    if candidate not in (1, 2):
        return None
    name = freeze_name(candidate)
    if not written(REPO / name):
        return None
    record, problems = read_freeze(candidate, runs_root=root, repo=REPO)
    if record is None:
        refuse("report", f"the analysis freeze has {len(problems)} problem(s): "
               + "; ".join(problems))  # fmt: skip
    return {"path": name, "sha256": sha256((REPO / name).read_bytes()), "record": record}


def output(root: Path, path: Path) -> dict[str, Any]:
    """A record written under RUNS_DIR/study3-final/: its path under RUNS_DIR, its sha256 and its
    content."""
    where = path.relative_to(root).as_posix()
    if _under_sealed(path, root):
        refuse("report", f"{where} resolves under a sealed folder: never opened")
    try:
        data = path.read_bytes()
        record = json.loads(data)
    except (OSError, ValueError) as exc:
        refuse("report", f"{where} cannot be read ({type(exc).__name__})")
    if not isinstance(record, dict):
        refuse("report", f"{where} is no JSON record")
    return {"path": where, "sha256": sha256(data), "record": record}


def final_outputs(root: Path, candidate: Any, frozen: Mapping[str, Any] | None
                  ) -> tuple[dict | None, dict | None, dict | None]:  # fmt: skip
    """The latest analysis, its validation harm and the audit's estimate, each checked by the
    sha256s it names (module docstring, "report", step 5); None for what is not written."""
    folder = root / FINAL / f"c{candidate}"
    if candidate not in (1, 2) or not folder.is_dir():
        return None, None, None
    k = latest(folder)
    analysis = validation = audit = None
    named = None if frozen is None else {"path": frozen["path"], "sha256": frozen["sha256"],
                                         "root": "repo"}  # fmt: skip
    if k:
        analysis = output(root, folder / names(k)[0])
        if named is None or analysis["record"].get("freeze") != named:
            refuse("report", f"{analysis['path']} does not name the committed analysis freeze by "
                   "its sha256")  # fmt: skip
        if written(folder / names(k)[1]):
            validation = output(root, folder / names(k)[1])
            own = {"path": analysis["path"], "sha256": analysis["sha256"], "root": "runs"}
            if (validation["record"].get("analysis") != own
                    or validation["record"].get("freeze") != named):  # fmt: skip
                refuse("report", f"{validation['path']} does not name {analysis['path']} and the "
                       "freeze by their sha256")  # fmt: skip
    if written(folder / AUDIT / "estimate.json"):
        audit = output(root, folder / AUDIT / "estimate.json")
        if analysis is None:
            refuse("report", f"{audit['path']} is written without an analysis")
        for name in ("selection", "agreements"):
            found = output(root, folder / AUDIT / f"{name}.json")
            if audit["record"].get(name) != {"path": found["path"], "sha256": found["sha256"],
                                             "root": "runs"}:  # fmt: skip
                refuse("report", f"{audit['path']} does not name {found['path']} by its sha256")
            if name == "selection":
                drawn = found["record"].get("analysis") or {}
                analyses = {f"{FINAL}/c{candidate}/{names(j)[0]}" for j in range(1, k + 1)}
                target = root / str(drawn.get("path"))
                if (found["record"].get("freeze") != named or drawn.get("path") not in analyses
                        or drawn.get("root") != "runs"
                        or sha256(target.read_bytes()) != drawn.get("sha256")):  # fmt: skip
                    refuse("report", f"{found['path']} does not name the freeze and an analysis "
                           f"of candidate {candidate} by their sha256")  # fmt: skip
    return analysis, validation, audit


def write_report(out: Path, files: Mapping[str, bytes]) -> None:
    """The report's files into a new folder: written into a temporary sibling, each synced, then
    renamed to ``out`` once; the temporary folder removed on any failure."""
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        for name, data in files.items():
            with open(temporary / name, "xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        if written(out):
            refuse("report", f"{out} exists: a report is written once, into a folder of its own")
        temporary.rename(out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


# ------------------------------------------------------------------ the command


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze", help="write the analysis freeze (once)")
    analyzed = commands.add_parser("analyze", help="the judge-based analysis on the frozen cohorts")
    reported = commands.add_parser("report", help="the final report's facts and tables (once)")
    for command in (frozen, analyzed, reported):
        command.add_argument("--runs-root", type=Path, required=True,
                             help="RUNS_DIR: the run records, the labels and the decision "
                             "records")  # fmt: skip
        command.add_argument("--theta0", type=Path,
                             help="θ₀'s evaluation (by default study 2's under "
                             "RUNS_DIR)")  # fmt: skip
    for command in (frozen, analyzed):
        command.add_argument("--candidate", type=int, choices=(1, 2), required=True)
    analyzed.add_argument("--supersedes-reason",
                          help="a recomputation after a fix: why the analysis before it no longer "
                          "serves (an id)")  # fmt: skip
    reported.add_argument("--billing", type=Path, required=True,
                          help="the researcher's billing file: spending by stage and each "
                          "incident with its billed price")  # fmt: skip
    reported.add_argument("--embargo", type=Path, required=True,
                          help="the researcher's embargo file: the breaches, or none")  # fmt: skip
    reported.add_argument("--out", type=Path,
                          help="the report's folder, new (by default "
                          "RUNS_DIR/study3-final/report-<UTC>/)")  # fmt: skip
    args = parser.parse_args(argv)
    {"freeze": freeze, "analyze": analyze, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
