"""Study 3's simulation before validation: its inputs, its fixation, the run, the independent
verification and the decision record.

    python scripts/study3_simulate.py inputs [--roots DIR ...] [--theta0 RUN_DIR] [--out PATH]
    python scripts/study3_simulate.py fixation [--workers 16]
    python scripts/study3_simulate.py run --runs-root DIR [--workers 16]
    python scripts/study3_simulate.py verify --runs-root DIR [--workers 16]
    python scripts/study3_simulate.py record --runs-root DIR

The rules are docs/study3-preregistration.md's "Simulation before validation" (the grid, the
coverage screen, the level L, the gates, the verification) and "Seeds and order" (after the
engineering stage, before the search); the models and the values the researcher decided are
``analysis.study3_simulation``'s (its rulings, M1–M27), the record's checks
``analysis.study3_decisions``' (the J check). The order:

1. **inputs** writes study 2's whole seed vectors (``analysis.study3_seed_vectors``) to --out
   (``docs/study3-simulation-inputs.json`` in the checkout by default) from --roots (by default
   ``$RUNS_DIR/study2_short`` and ``$RUNS_DIR/study2_stage_a``) and θ₀'s evaluation --theta0
   (study 2's, under RUNS_DIR, by default), each under RUNS_DIR and named in the record
   relative to it (refused otherwise); it prints the vectors, the exclusions and the record's
   sha256. The researcher commits it.
2. **fixation** needs a clean checkout of a known commit with the inputs record tracked and
   unchanged; it calibrates every scenario at every candidate it can run (through the
   selection, ``calibrate_all``, over --workers processes), enumerates the coverage tests and
   writes ``docs/study3-simulation-fixation.json`` (``build_fixation``: the code, the machine,
   the registration, the inputs by sha256, the sizes, the streams, the rules, the curvature,
   the scenarios with their calibrations, the tests with their count and sha256, J, the
   thresholds and the detection powers); it prints J, the tests' sha256 and the thresholds.
   The researcher commits it alone, before the first draw.
3. **run** refuses before any draw unless the fixation is tracked, committed once and alone in
   HEAD or an ancestor of it and unchanged since (``read_fixation``), its inputs' sha256 is
   the pinned one, its tests are its scenarios' identities, the checkout is clean, its sizes
   are this code's, no ``results.json`` and no ``screen.json`` exist, no part of another
   fixation or commit exists (each named), and the calibration recomputes (integers, flags and
   identities exactly, floats within a relative 1e-9 or their solver's precision,
   ``study3_simulation.CALIBRATION_FLOORS``; a different machine is printed, not refused). It
   then runs the gates, the screen and the n_min scan (``run_screen``) and writes
   ``screen.json``; it prints L (or "no level passed"), the validation size and the gates'
   values. Interrupted, it resumes from its finished parts on the same streams, to the same
   bytes.
4. **verify** refuses without ``screen.json``, with one of another commit than the clean HEAD,
   without a recomputed L (the verification is not due), or with ``verification.json``
   present (it is never rerolled); it runs the binding verification at L
   (``run_verification``) and writes ``verification.json``.
5. **record** needs ``screen.json`` (and ``verification.json`` when L exists) of the clean
   HEAD; it writes ``results.json`` (``results_record``: its time is the code commit's, so a
   resumed run's results are an uninterrupted one's, byte for byte) after the J check of the
   results (``study3_decisions.results_problems``), then the decision record
   ``study3/decisions/simulation.json`` through ``write_decision`` (its ``inputs``: the results
   and the seed vectors, each by sha256; the J check runs again in its reader); it prints the
   record's path, its state, L and the size. A second record is refused, as
   ``scripts/study3_summary.py record simulation`` refuses it.

The files of the run lie under ``RUNS_DIR/study3/simulation/`` (reserved: no run directory):
``parts/<phase>/<scenario>/<first>.json``, ``screen.json``, ``verification.json`` and
``results.json``; every file is JSON, written whole and once. The script opens no response,
harm label or judge output: of study 2's runs only their config, terminal record and step-25
evaluation (K and the utility), and nothing of study 3's runs. It prints numbers, ids, paths
and hashes only; a refusal exits with ``study3_simulate.py <command>: <why>``.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import study3_decisions as decisions
from zero_field_refusal.analysis import study3_simulation as sim
from zero_field_refusal.analysis.study3_decisions import (
    SIMULATION_FIXATION,
    SIMULATION_RESULTS,
    DecisionRefused,
    decision_path,
    write_decision,
)
from zero_field_refusal.analysis.study3_progress import THETA0_RUN
from zero_field_refusal.analysis.study3_records import read_theta0
from zero_field_refusal.analysis.study3_seed_vectors import (
    INPUTS_PATH,
    STUDY2_ROOTS,
    inputs_record,
    read_inputs,
    seed_vectors,
    study2_points,
)
from zero_field_refusal.paths import REPO_ROOT, runs_root
from zero_field_refusal.tracking import git_state

REPO = REPO_ROOT  # the checkout: the inputs record, the fixation and the GPU code
WORKERS = 16  # the local machine's cores


def refuse(command: str, why: str) -> NoReturn:
    sys.exit(f"study3_simulate.py {command}: {why}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def head_code(command: str) -> dict[str, str]:
    """HEAD's commit and its GPU code (``code_identity.gpu_code_identity`` of that commit), as
    a record's ``code`` names them; refused unless the checkout is clean."""
    state = git_state(REPO)
    if not state.get("commit") or state.get("dirty") is not False:
        refuse(command, f"the checkout is not clean, of a known commit ({state})")
    identity = code_identity.gpu_code_identity(state["commit"], REPO)
    return {"commit": state["commit"], "gpu_code_sha256": identity["sha256"]}


def tracked(command: str, path: str, what: str) -> None:
    """``path`` is tracked by the checkout (committed), or the command is refused."""
    try:
        git("ls-files", "--error-unmatch", "--", path)
    except subprocess.CalledProcessError:
        refuse(command, f"{what} {path} is not tracked: it is committed first")


def context(command: str):
    """The committed fixation with its inputs (``study3_simulation.load_context``), checked as
    ``run`` needs it before any draw; and HEAD's code."""
    tracked(command, SIMULATION_FIXATION, "the fixation")
    try:
        found = sim.load_context(REPO)
    except ValueError as exc:
        refuse(command, str(exc))
    code = head_code(command)
    sizes = {"screen": sim.SCREEN_RESAMPLES, "verification": sim.VERIFICATION_RESAMPLES}
    if found.fixation["resamples"] != sizes:
        refuse(command, f"the fixation froze the resamples {found.fixation['resamples']}, not "
               f"this code's {sizes}")  # fmt: skip
    return found, code


def write(command: str, path: Path, data: Any) -> None:
    try:
        sim._write_once(path, data)
    except sim.SimulationError as exc:
        refuse(command, str(exc))


# ------------------------------------------------------------------ the commands


def inputs(args: argparse.Namespace) -> None:
    """Study 2's whole seed vectors (module docstring, "inputs")."""
    out = args.out if args.out.is_absolute() else REPO / args.out
    if out.exists():
        refuse("inputs", f"{out} exists: the inputs record is written once")
    runs = runs_root()
    roots = args.roots or [runs / name for name in STUDY2_ROOTS]
    theta0 = args.theta0 or runs / THETA0_RUN

    def under_runs(path: Path) -> str:
        """``path`` relative to RUNS_DIR, as the record names it (no home directory)."""
        try:
            return path.resolve().relative_to(runs.resolve()).as_posix()
        except ValueError:
            refuse("inputs", f"{path} is not under RUNS_DIR ({runs}): the inputs record names "
                   "study 2's runs relative to it")  # fmt: skip

    named = {"theta0": under_runs(theta0), "sources": [under_runs(r) for r in roots]}
    try:
        points, excluded = study2_points(roots, theta0)
        vectors, skipped = seed_vectors(points)
        baseline = read_theta0(theta0)
        record = inputs_record(vectors, {**excluded, **skipped},
                               theta0={"path": named["theta0"], **baseline},
                               sources=named["sources"])  # fmt: skip
    except ValueError as exc:
        refuse("inputs", str(exc))
    write("inputs", out, record)
    print(f"vectors {len(vectors)}")
    for reason, count in sorted(record["excluded"].items()):
        print(f"excluded {reason} {count}")
    print(f"duplicates {record['duplicates']['count']}")
    print(f"{out} {sha256(out)}")


def fixation(args: argparse.Namespace) -> None:
    """The fixation (module docstring, "fixation")."""
    path = REPO / SIMULATION_FIXATION
    if path.exists():
        refuse("fixation", f"{path} exists: the fixation is written once")
    tracked("fixation", INPUTS_PATH, "the inputs record")
    code = head_code("fixation")
    try:
        vectors, slopes, _ = read_inputs(REPO / INPUTS_PATH)
    except ValueError as exc:
        refuse("fixation", str(exc))
    setup = sim.Setup(vectors=tuple(vectors), slopes=slopes, calibrations={},
                      M=decisions.COVERAGE_RUNS, levels=decisions.LEVELS,
                      validation_blocks=decisions.VALIDATION_BLOCKS[0])  # fmt: skip
    scenarios = list(sim.SCENARIOS)
    calibrations = sim.calibrate_all(setup, sim.STREAMS["calibration"],
                                     pairs=sim.calibration_pairs(scenarios), workers=args.workers,
                                     studies=sim.CALIBRATION_STUDIES)  # fmt: skip
    record = sim.build_fixation(REPO / INPUTS_PATH, scenarios, calibrations, code=code,
                                streams=sim.STREAMS, repo=REPO)  # fmt: skip
    write("fixation", path, record)
    print(f"J {record['J']}")
    print(f"tests {record['tests']['count']} {record['tests']['sha256']}")
    for name, c in sorted(record["thresholds"].items()):
        print(f"threshold {name} {c}")
    print(f"{path} {sha256(path)}")


def run(args: argparse.Namespace) -> None:
    """The gates, the screen and the scan (module docstring, "run")."""
    found, code = context("run")
    root = args.runs_root
    for path in (root / SIMULATION_RESULTS, sim.screen_path(root)):
        if path.exists():
            refuse("run", f"{path} exists: the screen is written once")
    problems = sim.part_problems(root, found.ref, code["commit"])
    if problems:
        refuse("run", "; ".join(problems))
    machine = sim.machine_record()
    if found.fixation["machine"] != machine:
        print(f"machine {json.dumps(machine, sort_keys=True)}, not the fixation's "
              f"{json.dumps(found.fixation['machine'], sort_keys=True)}")  # fmt: skip
    frozen = found.setup.calibrations
    setup = replace(found.setup, calibrations={})
    recomputed = sim.calibrate_all(setup, found.fixation["streams"]["calibration"],
                                   pairs=list(frozen), workers=args.workers,
                                   studies=sim.CALIBRATION_STUDIES)  # fmt: skip
    problems = sim.calibration_problems(recomputed, frozen)
    if problems:
        refuse("run", "the calibration does not recompute: " + "; ".join(problems[:20]))
    try:
        path = sim.run_screen(found, root, code["commit"], workers=args.workers)
    except sim.SimulationError as exc:
        refuse("run", str(exc))
    screen = json.loads(path.read_text(encoding="utf-8"))
    print(f"{path} {sha256(path)}")
    level = screen["level"]
    print("no level passed" if level is None else f"level {level}")
    print(f"validation_blocks {screen['validation_blocks']}")
    m, gates = screen["M"], screen["gates"]
    for name in ("safe_pass_rate", "empirical_pass_rate"):
        print(f"{name} {gates[name]['passed'] / m}")
    at = decisions.level_slot(decisions.POWER_LEVEL if level is None else level)
    for comparison, name in (("A-F", "power_af"), ("P-F", "power_pf")):
        by = gates["power"][comparison][at]
        print(f"{name} {min(c['correct'] for c in by.values()) / m}")
    scan = gates["n_scan"]
    print(f"n_min {scan[-1]['n'] if scan else 'null'}")


def verify(args: argparse.Namespace) -> None:
    """The binding verification (module docstring, "verify")."""
    root = args.runs_root
    if not sim.screen_path(root).exists():
        refuse("verify", f"{sim.screen_path(root)} is missing: the screen runs first")
    found, code = context("verify")
    try:
        sim.read_screen(root, found, code["commit"])
        path = sim.run_verification(found, root, code["commit"], workers=args.workers)
    except sim.SimulationError as exc:
        refuse("verify", str(exc))
    verification = json.loads(path.read_text(encoding="utf-8"))
    tests = verification["tests"]
    print(f"{path} {sha256(path)}")
    print(f"level {verification['level']}")
    print(f"tests {len(tests)} passed {sum(t['passed'] is True for t in tests)}")


def commit_time(commit: str) -> str:
    """The commit's time, UTC, ISO 8601 (``results.json``'s ``written_at``)."""
    seconds = int(git("show", "-s", "--format=%ct", commit))
    return datetime.fromtimestamp(seconds, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def record(args: argparse.Namespace) -> None:
    """The results and the decision record (module docstring, "record")."""
    root = args.runs_root
    found, code = context("record")
    target = decision_path(root, "simulation")
    if target.exists():
        refuse("record", f"{target} is written once: a second record is refused")
    try:
        screen = sim.read_screen(root, found, code["commit"])
    except sim.SimulationError as exc:
        refuse("record", str(exc))
    level = sim.selected_level(screen["tests"], found.fixation["J"], decisions.COVERAGE_RUNS)
    verification = None
    if level is not None:
        path = sim.verification_path(root)
        if not path.exists():
            refuse("record", f"level {level} passed the screen: {path} is missing (verify first)")
        verification = json.loads(path.read_text(encoding="utf-8"))
    try:
        results = sim.results_record(screen, verification, found, code,
                                     written_at=commit_time(code["commit"]))  # fmt: skip
    except sim.SimulationError as exc:
        refuse("record", str(exc))
    decided, problems = decisions.results_problems(results, found.fixation, found.ref["commit"],
                                                   REPO)  # fmt: skip
    if problems or decided is None:
        refuse("record", "the results do not pass the J check: " + "; ".join(problems))
    path = sim.results_path(root)
    data = sim._json_bytes(results)
    if path.exists() and path.read_bytes() != data:
        refuse("record", f"{path} exists and is not these results: results.json is written once")
    if not path.exists():
        write("record", path, results)
    pinned = found.fixation["inputs"]
    entry = {
        "kind": "simulation",
        "state": decided["state"],
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code,
        "fixation": {"path": SIMULATION_FIXATION, "sha256": found.ref["sha256"], "root": "repo"},
        "inputs": {"results": {"path": SIMULATION_RESULTS, "sha256": sha256(path)},
                   "seed_vectors": {"path": pinned["path"], "sha256": pinned["sha256"],
                                    "root": "repo"}},
        "gates": decided["gates"],
        "coverage": decided["coverage"],
        "level": decided["level"],
        "validation_blocks": decided["validation_blocks"],
        "verification": decided["verification"],
    }  # fmt: skip
    try:
        written = write_decision(root, "simulation", entry, repo=REPO)
    except DecisionRefused as exc:
        refuse("record", str(exc))
    print(f"{path} {sha256(path)}")
    print(f"{written} {sha256(written)}")
    print(f"state {entry['state']}")
    print(f"level {'null' if entry['level'] is None else entry['level']}")
    print(f"validation_blocks {entry['validation_blocks']}")


# ------------------------------------------------------------------ the command line


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    made = commands.add_parser("inputs", help="write study 2's whole seed vectors")
    fixed = commands.add_parser("fixation", help="calibrate the grid and write the fixation")
    ran = commands.add_parser("run", help="the gates, the screen and the n_min scan")
    verified = commands.add_parser("verify", help="the binding verification at L")
    recorded = commands.add_parser("record", help="the results and the decision record")
    made.add_argument("--roots", type=Path, nargs="+",
                      help="study 2's roots (default: RUNS_DIR/study2_short and "
                      "RUNS_DIR/study2_stage_a)")  # fmt: skip
    made.add_argument("--theta0", type=Path, help="θ₀'s evaluation (default: study 2's)")
    out = f"the inputs record (default: {INPUTS_PATH} in the checkout)"
    made.add_argument("--out", type=Path, default=Path(INPUTS_PATH), help=out)
    for command in (fixed, ran, verified):
        command.add_argument("--workers", type=int, default=WORKERS,
                             help="processes (the results do not depend on them)")  # fmt: skip
    runs = "RUNS_DIR: the simulation's files and the decision records"
    for command in (ran, verified, recorded):
        command.add_argument("--runs-root", type=Path, required=True, help=runs)
    args = parser.parse_args(argv)
    if getattr(args, "workers", 1) < 1:
        parser.error("--workers is at least 1")
    {"inputs": inputs, "fixation": fixation, "run": run, "verify": verify,
     "record": record}[args.command](args)  # fmt: skip


if __name__ == "__main__":
    main()
