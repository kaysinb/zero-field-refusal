"""Freeze the cohorts and K*_h of study 3's candidate N, then analyze its harm on them.

    python scripts/study3_analysis.py freeze  --runs-root DIR --candidate N [--theta0 RUN_DIR]
    python scripts/study3_analysis.py analyze --runs-root DIR --candidate N [--theta0 RUN_DIR]
        [--supersedes-reason ID]

docs/study3-preregistration.md: "Seeds and order" (order steps 9–11: the cohorts and K*_h frozen
before any confirmation label is opened, the labeling of the confirmation, the judge-based
analysis), "Questions and estimands" (the comparisons and the descriptives, validation's harm after
the final analysis and apart from it), "Inference", "Utility" (the claim at confirmation, at
α = 0.05/8 per bound), "Validation, confirmation and stopping" ("Confirmation") and "Judge and
audit" (the denominator 521, the three computations, "Sealing").

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
2. the in-force records, each through its reader: the confirmation supplement
   (``study3_versions.read_confirmation_fixation``: the bootstrap's seed and resamples; its link,
   ``study3_freeze.in_force_supplement``), the validation fixation
   (``study3_versions.read_validation_fixation``: the level L) and the n record
   (``study3_decisions.decisions``: validation's A–F cohort, which must be the frozen one); κ per
   arm from the simulation fixation's ``curvature`` (``simulation_curvature``);
3. the outputs so far (below: a resumed or a superseding run);
4. the labels, through the one label reader (``study3_final.gather``): the harm of every run the
   freeze names, then the spread of K of every bracket run of its comparisons;
5. the utility gates over each comparison's frozen blocks at α = 0.05/8 per bound
   (``study3_blocks.utility_gate``, from the state's endpoints, the frozen runs checked), then
   ``study3_estimates.analysis``;
6. ``analysis.json``, then validation's harm (``study3_estimates.validation_harm``) in
   ``validation-harm.json``, which names the analysis by its sha256;
7. it prints both paths, and each comparison's n and the row of the registered wording.

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
validation's harm not yet written writes validation's harm alone, from the same freeze and
in-force records. Once both are written, ``analyze`` refuses unless --supersedes-reason gives an
id (a lowercase letter, then lowercase letters, digits or underscores): a recomputation after a
fix of the analysis code, with no change of the GPU code, is written as ``analysis-<k>.json`` and
``validation-harm-<k>.json`` (k = 2, 3, …), its ``superseded`` naming its predecessor by path,
sha256 and the reason.

**The sealing boundary.** The labels, the K rows and nothing else of a run's sealed folders are
opened, inside ``study3_final`` alone, and only once the freeze gate opens the final point; of the
labeling ledger, only run ids and states are read. What is written and printed holds numbers, ids,
run ids, paths, hashes, times and the simulation fixation's ``source``.

**Rulings** where the registration leaves a detail open:
- *Outside the experiment's folder.* The outputs are under RUNS_DIR/study3-final/c<N>/.
- *Validation's harm* is A–F on the n record's cohort and P–F on the validation blocks supported
  for P and F while P–F is active, computed as the analysis is, descriptive, after the analysis
  and in its own file, which names the analysis by its sha256.
- *Written once.* The analysis is ``analysis.json``; a recomputation after a fix without a change
  of the GPU code is ``analysis-<k>.json``, naming its predecessor by path, sha256 and a reason id.
- *κ* is the simulation fixation's base ``curvature`` (docs/study3-simulation-fixation.json, in the
  checkout, committed once and unchanged at HEAD), pinned in the analysis by its sha256.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import study3_estimates, study3_final
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
    decisions,
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
    ledger_runs,
)
from zero_field_refusal.analysis.study3_progress import COMPARISONS, read_state
from zero_field_refusal.analysis.study3_records import EXPERIMENT
from zero_field_refusal.analysis.study3_search import ARM_ORDER
from zero_field_refusal.analysis.study3_versions import (
    _added,
    _blob,
    read_confirmation_fixation,
    read_validation_fixation,
)
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.tracking import git_state

REPO = REPO_ROOT  # the checkout: the freeze record, the fixations, the records' repo evidence
FINAL = "study3-final"  # RUNS_DIR/study3-final/c<N>/: the analysis's outputs
SIMULATION_FIXATION = "docs/study3-simulation-fixation.json"
SIMULATION_KIND = "simulation-fixation"
VALIDATION_KIND = "study3-validation-harm"
REASON = re.compile(r"[a-z][a-z0-9_]{0,63}")  # a reason id, as the versions record's


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
    supplement, problems = read_confirmation_fixation(candidate, runs_root=root, repo=REPO)
    if supplement is None:
        refuse("analyze", f"candidate {candidate}'s confirmation supplement is "
               + (f"not valid ({problems[0]})" if problems else "missing")
               + ": it fixes the bootstrap's seed")  # fmt: skip
    link, problems = in_force_supplement(candidate, runs_root=root, repo=REPO)
    if link is None:
        refuse("analyze", "the in-force confirmation supplement cannot be named: "
               + ("; ".join(problems) or "missing"))  # fmt: skip
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
    (``study3_final.cohort_harm``); every refusal names ids, counts and types only."""
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
    """Step 5's gates: per frozen comparison, ``utility_gate`` at α = 0.05/8 over its frozen
    blocks as the state's endpoints give them now (a run under another commit left out, as the
    freeze leaves it out), each arm's runs the frozen ones."""
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
    # 4. the labels
    try:
        state, endpoints = read_state(root, repo=REPO,
                                      expected_code=freeze_["code"]["gpu_code_sha256"],
                                      theta0=args.theta0)  # fmt: skip
    except (OSError, ValueError) as exc:
        refuse("analyze", f"the study's state cannot be read ({type(exc).__name__})")
    harm, spread = labels(freeze_, endpoints, root)
    # 5. the gates, then the analysis
    bootstrap = inputs["bootstrap"]
    kappa = inputs["kappa"]
    gates = utility_gates(freeze_, state, endpoints)
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
    # 6. the analysis, then validation's harm
    data = encoded(record)
    publish("analyze", analysis_path, root, data)
    validation = validation_record(candidate, code, frozen, inputs, freeze_, harm, analysis_path,
                                   data)  # fmt: skip
    publish("analyze", validation_path, root, encoded(validation))
    # 7. what was written
    print(analysis_path)
    print(validation_path)
    for key, entry in result["comparisons"].items():
        print(f"{key}: n {entry['n']}, {entry['wording']['row']}")


def resume(root: Path, candidate: int, code: Mapping[str, str], frozen: Mapping[str, str],
           inputs: Mapping[str, Any], freeze_: Mapping[str, Any], folder: Path, k: int,
           data: bytes) -> None:  # fmt: skip
    """Validation's harm alone, after the k-th analysis (its bytes ``data``) is written without
    it (module docstring, "Restart and supersession"): the harm of validation's frozen runs, read
    through the final reader."""
    blocks = [block for entry in freeze_["validation"].values() if entry is not None
              for block in entry["blocks"]]  # fmt: skip
    experiment = root / EXPERIMENT
    run_dirs = {run: experiment / run for run in study3_final.runs_of(blocks)}
    harm, _ = labels(freeze_, (), root, run_dirs)
    analysis, path = (folder / part for part in names(k))
    validation = validation_record(candidate, code, frozen, inputs, freeze_, harm, analysis, data)
    publish("analyze", path, root, encoded(validation))
    print(path)


# ------------------------------------------------------------------ the command


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("freeze", help="write the analysis freeze (once)")
    analyzed = commands.add_parser("analyze", help="the judge-based analysis on the frozen cohorts")
    for command in (frozen, analyzed):
        command.add_argument("--runs-root", type=Path, required=True,
                             help="RUNS_DIR: the run records, the labels and the decision "
                             "records")  # fmt: skip
        command.add_argument("--candidate", type=int, choices=(1, 2), required=True)
        command.add_argument("--theta0", type=Path,
                             help="θ₀'s evaluation (by default study 2's under "
                             "RUNS_DIR)")  # fmt: skip
    analyzed.add_argument("--supersedes-reason",
                          help="a recomputation after a fix: why the analysis before it no longer "
                          "serves (an id)")  # fmt: skip
    args = parser.parse_args(argv)
    {"freeze": freeze, "analyze": analyze}[args.command](args)


if __name__ == "__main__":
    main()
