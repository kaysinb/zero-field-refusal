"""Run the numerical check of study 2's selected rates (amendment 4, section 1).

    python scripts/numerical_check.py plan RATES --batch B --upper U [--runs-root DIR] [--out DIR]
        [--no-stream-check] [--dry-run]
    python scripts/numerical_check.py control RATES --batch B --upper U [--retry]
        [--runs-root DIR] [--out DIR] [--device D] [--dry-run]
    python scripts/numerical_check.py twins RATES --batch B --upper U [--retry]
        [--keep-checkpoints] [--runs-root DIR] [--out DIR] [--device D] [--dry-run]
    python scripts/numerical_check.py verdict RATES --batch B --upper U [--retry-needed]
        [--runs-root DIR] [--out DIR] [--dry-run]
    python scripts/numerical_check.py driver RATES --batch B --upper U --commit SHA --out FILE
        [--dry-run]
    python scripts/numerical_check.py void --batch B --upper U [--runs-root DIR] [--dry-run]

RATES is ``--rates OPTIMIZER=R1,R2,R3`` once per arm (adamw, adam_frozen, adamw_prewarmed; or
the primary arms adamw and adam_frozen alone, a selection without the pre-warmed arm), with
``--given-up CASE`` once per case whose regular run the launcher gave up on and ``--request
FILE``, the check request that names them, or ``--selection FILE``: a JSON record with
``rates``, mapping each arm (by optimizer name, or by the analysis's arm name) to its selected
rates, and optionally ``batch`` and ``U``, which must equal ``--batch`` and ``--upper``, and
``given_up``, the cases whose regular run the launcher gave up on (each named as the check names
a case, ``s<seed>-<optimizer>-<lr>``; the selection's check request writes it); any other key is
refused. The rates are never written in the code. A ``--request`` must name the same rates and
cases given up as ``--rates`` and ``--given-up`` (the driver passes all three).

Outside the confirmation and blind to outcomes: only K, KL between twins and norms are computed
and written, every evaluation on benign panels; no harmful prompt set, stream or generation is
touched (src/zero_field_refusal/fa_control_diagnostic.py), and every prompt set whose name
contains "harmful" raises if any code asks for it (``guard_prompt_sets``). The settings are the
tracked configs/experiments/numerical_check.yaml: ``control``, ``twins``, ``verdict`` and
``driver`` refuse a ``--config`` that resolves to any other file (a file of the same name in the
working directory included, which ``load_config`` would read first) unless they only print
(``--dry-run``, ``--retry-needed``).

- The regular runs re-examined are found under ``--runs-root`` (RUNS_DIR) by their archived
  configs: experiment study2_short with ``study2_short.role=numerical``, at the case's seed,
  arm, rate and B. Their streams must be in DATA_DIR (the plan checks them).
- Every part writes under ``--out`` (default RUNS_DIR/numerical/b{B}-u{U}/):
  ``controls/<lr>/attempt-<n>.json``, ``twins/s<seed>/<optimizer>-<lr>/attempt-<n>.json``,
  ``verdicts/<selection digest>.json`` (one per selection, kept: ``selection_digest``) and
  ``verdict.json`` (a copy of the latest).
- **Which records count** (``RecordReader``): those written from a clean checkout at a commit
  whose GPU code this checkout's equals (``code_identity.reuse_check``), under the frozen
  settings and the policy's perturbations, of the regular run located now. A record that counts
  is kept: running again skips it, at this commit or a later one with the same GPU code. Only
  a real run of ``control`` or ``twins``, from a clean checkout and after the code check, moves
  a record that does not count aside, to ``stale/<its commit>-<time>/``, before redoing it.
  Every ``--dry-run``, ``plan`` and ``--retry-needed`` only reads.
- ``control`` runs the first attempt of the same-path float64 soft control at each selected FA
  rate on the first numerical seed that has none; ``twins`` the first attempt of every case that
  has none. With ``--retry``, each runs the second attempt (a fresh process) of every control or
  case whose first is incomplete (a case not exceeding, a control not rejected). A case or
  control that fails is recorded (incomplete) and the part goes on; the part then exits 6.
  Both refuse a checkout that is not clean.
- ``verdict`` writes the selection's verdict record (and refuses a checkout that is not clean):
  per rate qualified, failing (a pair exceeding, or the regular run stopped by an instability
  rule: evidence against it), not_qualified (technical: incomplete after its rerun, never
  called instability) or incomplete (its rerun not yet done); per FA control accepted, rejected
  (exceeding, at the bound, or its run stopped by instability), not_qualified or incomplete;
  with the numerical policy, the seeds each rate was tested on and the cases whose reference
  fell back. ``--retry-needed`` exits 0 when a rerun (of a case or a control) is due, 3 when
  none is, and writes nothing.
- Every part runs under the memory rule (``memory_rule_gb``: 46 GB reserved, 1e9 bytes): the
  process's CUDA memory is capped there; an out-of-memory error under the cap makes the float64
  reference fall back (recorded), and fails the case or control it hits (recorded).
- ``--dry-run`` prints what the part would run, with the cost estimate, without loading a model
  or writing or moving anything (``driver --dry-run`` prints the driver). ``plan`` also checks
  the DATA_DIR streams, unless ``--no-stream-check``, and exits 1 when a run or a stream is
  missing, or a run departs from the prescribed one. A run that an instability rule stopped
  (and that conforms otherwise) is found, with a note: the parts record its rate as failing
  and its FA control rejected. A missing run of a case named given up (``given_up``)
  is no problem either, with a note: the parts record that case
  incomplete, and not qualified after its rerun (and, for an FA run, its same-path control and
  every case of its seed when it was the seed's only FA run, which sets v₀ and p1, p2 up).
- **A case given up**: ``control`` and ``twins`` record a case (or same-path
  control) named given up whose regular run is not located without a GPU, before any model or
  device is touched (``given_up_twins``, ``given_up_controls``): incomplete, with the reason
  "given up by the launcher" and, as its evidence, the check request's path and sha256
  (``--selection``, or ``--request``; ``plan`` and the parts refuse a case given up without
  it). Its rerun is recorded the same way, and the verdict holds it not qualified (technical),
  never failing; a part whose every due item is such a case builds no environment. When every
  case is given up and no run is located, the GPU-code and run-environment checks have nothing
  to compare: they are skipped with a note (``NONE_LOCATED``), recorded in the verdict's
  ``code_check`` and in the records (``provenance.checks_skipped``), and the verdict's policy
  records the check's own GPU code.
- **GPU code**: every part compares the GPU-code identity
  (``code_identity.gpu_code_identity``) of the commit each regular run recorded with this
  checkout's (``code_check``). On a mismatch ``plan`` exits 1, ``control`` and ``twins`` run
  nothing and exit 7, and the verdict counts no record. The verdict's policy records that
  identity (``gpu_code``) and the commits that wrote the records (``tool_commit``).
- **The run environment**: ``plan``, ``control`` and ``twins``
  compare each regular run's recorded environment (meta.json: the GPU names, the driver,
  ``torch_cuda``, the Python version, the package versions, and the container image when one is
  recorded) with this process's (``environment_check``). On a mismatch, as for the GPU code,
  ``plan`` exits 1 and ``control`` and ``twins`` record nothing and exit 7, naming each field.
  Twin A can be expected to reproduce its run (numerically consistent with it, within the
  policy's ``twin_a_consistency``) only in the environment that made it: the check
  runs on the Pod (or the image, GPU and driver) of the numerical runs.
- **Host memory**: every control and twin record holds the
  process's peak resident memory and the free host memory when it was written
  (``provenance.host_memory``); the driver starts only with ``HOST_MEMORY_GB`` free.
- ``driver`` writes the bash driver of all parts for a Pod, from the source checkout of SHA. It
  checks the checkout, that nothing else uses the GPU, the free host memory, clears the twins'
  checkpoints that an interrupted case left (records are kept), checks the free disk, then runs
  the plan and the parts. Its exits: 1 a precondition or the plan; 5 a verdict with an
  incomplete rate or control; 6 a part failed (recorded); 0 complete. With cases given up, it
  passes the check request's resolved path to every part (``--request``), and is written only
  with one (``--selection FILE``, or ``--request FILE``).
- ``void`` moves, never deletes, what a GPU-path change after
  the check left unusable: every numerical run at B (role numerical) whose recorded commit's GPU
  code is not HEAD's, ``RUNS_DIR/numerical/b{B}-u{U}/``, and the launcher's record of the
  campaigns that made those runs (``launch/<campaign>.jsonl``, ``launch/<campaign>/`` and the
  campaign's file under ``RUNS_DIR/campaigns/`` with its ``-driver.sh``), to
  ``RUNS_DIR/void/<the old GPU code's sha256, first 12>/``, with an inventory of every file
  moved (path, size, sha256) and a record (``void_plan``, ``void_numerical``). It refuses on a
  checkout that is not clean; when a run of the role validation, confirmation or timecourse
  exists at B (S3 has started: a GPU-path change after that is outside this procedure and goes
  to the researcher, since voiding those runs means spending on S3 again); when a check
  directory of another U exists at B (the numerical runs at B serve it too, and its verdicts
  would go unread: to the researcher); when no run's GPU code differs; when it would move a run
  of another role (or one whose GPU code cannot be read); and when a launcher log to move also
  names a run it leaves in place. Without ``numerical/b{B}-u{U}/`` (a GPU-path change between
  the K-only runs and their check) it moves the rest and records "no check directory". The selection
  then calls for the numerical runs again, and their check; the fresh campaign at the same
  rates starts with all its attempts. ``--dry-run`` prints what it would move.

Cost on one L40S at 1.5B (fp32), scaled from the FA diagnostic's records at B 5 (≈ 0.17 s per
response and step with the monitor's readings, a D_twin reading ≈ 30 s, a scoring K ≈ 14.5 s, a
checkpoint save ≈ 5 s); estimates, ``plan`` prints them at the given B. At B 20:
- a seed's setup (models, v₀'s 50 batch gradients, the residual's replay) ≈ 5 min;
- a case ≈ 25 min: 6.5 twin-equivalents (twin A, p1-p4, the float64 reference ≈ 1.5 twins),
  30 D_twin readings, 6 saves;
- an allowance of ≈ 5 min per AdamW and pre-warmed case for a full-float64 reference lost to
  the memory cap before its fallback;
- a control ≈ 6 min (two forwards per step in float64), after ≈ 5 min of setup.
Fifteen cases and three controls ≈ 6.8 GPU hours, ≈ 7.6 with the allowance; a rerun ≈ 25 min
more per incomplete case. Peaks: ≈ 31-38 GB for the twins, ≈ 38 GB for FA's float64 reference
and the controls, ≈ 44 GB for the AdamW and pre-warmed references, which fall back to float64
loss and accumulation if they meet the cap. 31.4 GB of checkpoints per case on the volume,
deleted when the case is done.
"""

import argparse
import hashlib
import json
import shlex
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

from zero_field_refusal.config import _locate, load_config
from zero_field_refusal.fa_control_diagnostic import (
    ATTEMPTS,
    DiagnosticError,
    NumericalError,
    NumericalScope,
    RecordReader,
    apply_memory_rule,
    attempt_state,
    code_check,
    conformance,
    control_attempt_state,
    control_case,
    control_path,
    control_retry_due,
    environment_check,
    git_state,
    given_up_controls,
    given_up_twins,
    guarded_prompt_sets,
    instability_stop,
    load_numerical,
    locate_case,
    numerical_cases,
    numerical_root,
    numerical_verdict,
    output_commit,
    output_dirty,
    retry_due,
    run_config,
    run_numerical_controls,
    run_numerical_twins,
    selected_rates,
    set_aside,
    stream_inputs,
    void_numerical,
    void_plan,
    write_verdict,
)
from zero_field_refusal.paths import CONFIG_ROOT, runs_root
from zero_field_refusal.tracking import CLEAN_CHECKOUT

CONFIG = "experiments/numerical_check"
SCRIPT = 'project-run "$SRC" python scripts/numerical_check.py'
NOT_DUE = 3  # ``verdict --retry-needed``: no rerun is due
INCOMPLETE = 5  # the driver: the verdict holds an incomplete rate
PART_FAILED = 6  # a case or control failed (recorded); the driver: a part did
# The regular runs' GPU code or recorded run
# environment is not the check's; nothing runs
CODE_MISMATCH = 7
GPU_USED_MIB = 1000  # the driver starts only below this much GPU memory in use
DISK_GB = 40  # the free space a case needs: its checkpoints (31.4 GB at 1.5B) and a margin
# The free host memory the driver needs (GB of 1e9 bytes): θ₀,
# v₀, the second model and the p1 direction on the host (≈ 25-30 GB at 1.5B) and a checkpoint
# save's transient (≈ 5 GB), × 1.5
HOST_MEMORY_GB = 1.5 * (30 + 5)
# The analysis's arm names (analysis.study2_short) and the optimizers they run.
ARM_OPTIMIZERS = {"adamw_25": "adamw", "frozen_adam_25": "adam_frozen",
                  "prewarmed_adamw_25": "adamw_prewarmed"}  # fmt: skip
SELECTION_KEYS = {"rates", "batch", "U", "given_up"}  # the keys a selection record may hold
# The driver's programs (python -c, through project-run): a field of the verdict, whether it
# holds an incomplete rate or control (exit 0), and whether the free host memory
# (``drift.host_memory``: MemAvailable, or the cgroup's free memory when smaller) reaches the
# requirement (exit 0). No $, backtick, backslash or double quote inside.
READ_FIELD = "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])"
HAS_INCOMPLETE = ("import json, sys; v = json.load(open(sys.argv[1])); "
                  "sys.exit(0 if 'incomplete' in [*v['rates'].values(), *v['controls'].values()] "
                  "else 1)")  # fmt: skip
HOST_FREE = ("import sys; from zero_field_refusal.drift import host_memory; m = host_memory(); "
             "print('free host memory (GB):', m); "
             "sys.exit(0 if (m['available_gb'] or 0) >= float(sys.argv[1]) else 1)")  # fmt: skip
# A plan's note on a case whose regular run the launcher gave up on
GIVEN_UP = "given up by the launcher: the parts will record it not qualified"
# Why the code and environment checks compare nothing, when they do not
NONE_LOCATED = ("no regular run is located: the check request names every case given up by the "
                "launcher, so there is nothing to compare; skipped")  # fmt: skip


def comment(text: str) -> list[str]:
    return textwrap.wrap(text, 100, initial_indent="# ", subsequent_indent="# ",
                         break_long_words=False, break_on_hyphens=False)  # fmt: skip


def guard_prompt_sets(env):
    """The environment with every prompt set whose name contains "harmful" refused: the check
    is K-only on benign panels, and this makes any path to a harmful set raise."""
    return guarded_prompt_sets(env, "harmful", "The numerical check")


def parse_rates(items: list[str]) -> dict[str, list[float]]:
    """``OPTIMIZER=R1,R2,R3`` items, one per arm."""
    out: dict[str, list[float]] = {}
    for item in items:
        name, _, listed = item.partition("=")
        if not listed or name in out:
            raise NumericalError(f"--rates {item!r}: OPTIMIZER=R1,R2,R3, once per arm")
        out[name] = [float(x) for x in listed.split(",")]
    return out


def given_up_names(names: object, where: str) -> list[str]:
    """The cases named given up: distinct case names (``s<seed>-<optimizer>-<lr>``)."""
    if not (isinstance(names, list) and all(isinstance(n, str) and n for n in names)
            and len(set(names)) == len(names)):  # fmt: skip
        raise NumericalError(f"{where}: 'given_up' must list distinct case names, got {names!r}")
    return list(names)


def read_selection(path: Path, batch: int, upper: float) -> tuple[dict[str, list[float]], list]:
    """The rates of a selection record, ``rates`` (required), and the cases it names given up
    by the launcher, ``given_up`` (optional); ``batch`` and
    ``U`` must equal B and U when present. Any other key, or a malformed one, is refused."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise NumericalError(f"{path}: not a readable JSON selection record ({exc})") from exc
    return parse_selection(data, path, batch, upper)


def parse_selection(
    data: bytes, path: Path, batch: int, upper: float
) -> tuple[dict[str, list[float]], list]:  # fmt: skip
    """``read_selection`` of the bytes ``data`` read from ``path``."""
    try:
        record = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NumericalError(f"{path}: not a readable JSON selection record ({exc})") from exc
    if not isinstance(record, dict) or "rates" not in record:
        raise NumericalError(f"{path}: a selection record needs 'rates' (and may hold 'batch', "
                             "'U' and 'given_up')")  # fmt: skip
    if set(record) - SELECTION_KEYS:
        raise NumericalError(f"{path}: unexpected keys {sorted(set(record) - SELECTION_KEYS)}; "
                             f"a selection record holds only {sorted(SELECTION_KEYS)}")  # fmt: skip
    for key, value in (("batch", batch), ("U", upper)):
        if key in record and record[key] != value:
            raise NumericalError(f"{path}: {key} {record[key]!r}, not {value}")
    rates = record["rates"]
    if not isinstance(rates, dict) or not all(
        isinstance(values, list) and all(isinstance(r, int | float) and not isinstance(r, bool)
                                         for r in values)
        for values in rates.values()
    ):  # fmt: skip
        raise NumericalError(f"{path}: 'rates' must map each arm to a list of numbers")
    given_up = given_up_names(record.get("given_up", []), str(path))
    return {ARM_OPTIMIZERS.get(arm, arm): [float(r) for r in values]
            for arm, values in rates.items()}, given_up  # fmt: skip


def rates_of(args: argparse.Namespace) -> tuple[dict[str, list[float]], list[str]]:
    """The selected rates, and the cases given up by the launcher: from ``--rates`` and
    ``--given-up``, or from a selection record (``--selection``), which names its own."""
    if bool(args.rates) == bool(args.selection):
        raise NumericalError("Give the rates by --rates (once per arm) or by --selection FILE")
    if args.rates:
        return parse_rates(args.rates), given_up_names(args.given_up, "--given-up")
    if args.given_up:
        raise NumericalError("--given-up goes with --rates: a selection record names its own "
                             "cases given up ('given_up')")  # fmt: skip
    return read_selection(args.selection, args.batch, args.upper)


def check_given_up(given_up: list[str], cases) -> None:
    """Every case named given up is one of the check's (a name the check gives a case)."""
    unknown = sorted(set(given_up) - {case.name for case in cases})
    if unknown:
        raise NumericalError(f"given up: {unknown} name no case of this check "
                             f"({[case.name for case in cases]})")  # fmt: skip


def request_of(args: argparse.Namespace, scope: NumericalScope, rates: dict, given_up: list):
    """The check request that names the cases given up, which their records
    state as evidence: its resolved path and the sha256 of its bytes, as read now. With
    ``--selection FILE``, that file; with ``--rates`` and ``--given-up``, ``--request FILE``
    (the driver passes it), which must name the same rates and the same cases given up; without
    either, None."""
    if args.request is not None and args.selection is not None:
        raise NumericalError("--request goes with --rates and --given-up: a selection record is "
                             "its own request")  # fmt: skip
    path = args.selection or args.request
    if path is None:
        return None
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise NumericalError(f"{path}: the check request cannot be read ({exc})") from exc
    named, named_given_up = parse_selection(data, path, args.batch, args.upper)
    if selected_rates(scope, named) != rates or sorted(named_given_up) != sorted(given_up):
        raise NumericalError(f"{path}: the check request names other rates or cases given up "
                             f"({named}, given up {sorted(named_given_up)}) than these "
                             f"({rates}, given up {sorted(given_up)})")  # fmt: skip
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def given_up_unlocated(part: str, items, given_up, located, scope, batch: int, upper: float):
    """The items of ``part`` (cases for ``twins``, FA rates for ``control``)
    whose case the request names given up and whose regular run is not located (``located``,
    by case name): recorded without a GPU (``given_up_twins``, ``given_up_controls``)."""

    def name(item) -> str:
        return control_case(scope, item, batch, upper).name if part == "control" else item.name

    return [item for item in items if name(item) in given_up and name(item) not in located]


TWINS_PER_CASE = 6.5  # twin A, p1-p4, and the float64 reference ≈ 1.5 twins
FALLBACK_ARMS = ("adamw", "adamw_prewarmed")  # their full-float64 reference meets the cap


def costs(batch: int, scope: NumericalScope) -> dict[str, float]:
    """GPU minutes on one L40S at 1.5B (module docstring), at batch ``batch``. ``fallback`` is
    the allowance per AdamW or pre-warmed case for a full-float64 reference lost to the memory
    cap before its fallback (at worst a whole reference: 1.5 twins and its readings)."""
    step = 0.17 * batch  # seconds per training step, monitor readings included
    twin = scope.steps * step
    readings = len(scope.eval_at)
    return {
        "setup": (50 * step + 120) / 60,  # v₀'s batch gradients; models, streams, the replay
        "case": (TWINS_PER_CASE * twin + 5 * readings * 30 + readings * 5) / 60,
        "fallback": (1.5 * twin + readings * 30) / 60,
        "control": (2 * twin + readings * 14.5 + scope.steps / 5 * 2.7 + 60) / 60,
    }


def total_cost(batch: int, scope: NumericalScope, cases, controls: int) -> tuple[float, float]:
    """GPU minutes of ``cases`` and ``controls`` controls, without and with the allowance."""
    cost = costs(batch, scope)
    seeds = len({c.seed for c in cases})
    minutes = (seeds * cost["setup"] + len(cases) * cost["case"]
               + (cost["setup"] + controls * cost["control"] if controls else 0))  # fmt: skip
    fallback = sum(cost["fallback"] for c in cases if c.optimizer in FALLBACK_ARMS)
    return minutes, minutes + fallback


def find_runs(
    root: Path, scope: NumericalScope, cases, given_up=()
) -> tuple[dict, dict, dict]:  # fmt: skip
    """The regular run of each case found; why a case has none, or one that cannot be twin A
    (it departs from the prescribed run); and the notes of runs that an instability rule
    stopped. Such a run is found and no problem: the parts record its rate as
    failing. A run that departs otherwise is a problem even when it stopped by instability. A
    case named in ``given_up`` without a run is
    no problem either, with a note: the parts record it incomplete, not qualified after its
    rerun; so is every found case of a seed whose control-arm runs are all given up, which the
    parts record incomplete too (that run sets the seed's v₀ and p1, p2 up)."""
    found, problems, notes = {}, {}, {}
    for case in cases:
        try:
            path = locate_case(root, scope, case)
        except DiagnosticError as exc:
            if case.name in given_up:
                control = case.optimizer == scope.control_arm and case.seed == scope.control_seed
                notes[case.name] = GIVEN_UP + (" (and its same-path control)" if control else "")
            else:
                problems[case.name] = str(exc)
            continue
        found[case.name] = path
        cfg = run_config(path)
        departures = conformance(cfg, path, scope, case, stop=False)
        if not departures:
            unstable = instability_stop(path)
            if unstable is not None:
                notes[case.name] = ("instability: the parts will record the rate as failing ("
                                    f"{unstable.removeprefix('instability: ')})")  # fmt: skip
                continue
            departures = conformance(cfg, path, scope, case)
        if departures:
            problems[case.name] = "does not conform: " + "; ".join(departures)
    for seed in dict.fromkeys(case.seed for case in cases):
        controls = [c for c in cases if c.seed == seed and c.optimizer == scope.control_arm]
        if controls and all(c.name in given_up and c.name not in found for c in controls):
            for case in cases:
                if case.seed == seed and case.name in found and case.name not in notes:
                    notes[case.name] = (f"the seed's {scope.control_arm} runs are given up by "
                                        "the launcher: the parts will record it incomplete (that "
                                        "run sets v₀ and p1, p2 up), not qualified after its "
                                        "rerun")  # fmt: skip
    return found, problems, notes


def plan(settings, args, cases, root: Path, out: Path, given_up=()) -> tuple[str, bool]:
    """What the check would run, its cost, and whether every run it needs is there (a case
    named given up needs none)."""
    scope = NumericalScope.of(settings)
    found, problems, notes = find_runs(root, scope, cases, given_up)
    upper = cases[0].upper
    control_rates = sorted({c.lr for c in cases if c.optimizer == scope.control_arm
                            and c.seed == scope.control_seed})  # fmt: skip
    lines = [f"output: {out}",
             f"B {args.batch}, U {upper:g}: K* = {scope.k_star(upper):.6g}, bound "
             f"{scope.bound(upper):.6g}; perturbation norm {scope.norm:g}; steps "
             f"{list(scope.eval_at)}", f"cases ({len(cases)}):"]  # fmt: skip
    for case in cases:
        if case.name in problems:
            lines.append(f"  {case.name}: PROBLEM: {problems[case.name]}")
        elif case.name in notes:
            lines.append(f"  {case.name}: {found.get(case.name, 'no regular run')}; NOTE: "
                         f"{notes[case.name]}")  # fmt: skip
        else:
            lines.append(f"  {case.name}: {found[case.name]}")
    lines.append(f"same-path controls on seed {scope.control_seed}: "
                 f"{[f'{lr:g}' for lr in control_rates]}")  # fmt: skip
    cost = costs(args.batch, scope)
    minutes, allowed = total_cost(args.batch, scope, cases, len(control_rates))
    lines.append(f"cost ≈ {minutes:.0f} GPU min at B {args.batch}, ≈ {allowed:.0f} with a lost "
                 f"full-float64 reference per AdamW and pre-warmed case (a setup ≈ "
                 f"{cost['setup']:.0f} min per seed and for the controls, a case ≈ "
                 f"{cost['case']:.0f} min, the allowance ≈ {cost['fallback']:.0f} min, a control "
                 f"≈ {cost['control']:.0f} min); reserved memory capped at "
                 f"{scope.memory_rule_gb:g} GB; a rerun ≈ {cost['case']:.0f} min per incomplete "
                 "case")  # fmt: skip
    return "\n".join(lines), not problems


def check_streams(
    settings, cases, root: Path, device: str, given_up=()
) -> tuple[list[str], list[str]]:  # fmt: skip
    """The DATA_DIR streams of each seed, from a control-arm run of the seed (``stream_inputs``).
    A seed whose control-arm runs were all given up by the launcher has none to locate them by:
    noted among those found, its cases recorded incomplete by the parts (``find_runs``)."""
    from zero_field_refusal.runtime import environment

    scope = NumericalScope.of(settings)
    found, missing = [], []
    for seed in dict.fromkeys(c.seed for c in cases):
        controls = [c for c in cases if c.seed == seed and c.optimizer == scope.control_arm]
        cfg = None
        for case in controls:
            try:
                cfg = run_config(locate_case(root, scope, case))
                break
            except DiagnosticError:
                continue
        if cfg is None:
            if controls and all(c.name in given_up for c in controls):
                found.append(f"seed {seed}: no stream checked: its {scope.control_arm} runs are "
                             "given up by the launcher")  # fmt: skip
            else:
                missing.append(f"seed {seed}: no {scope.control_arm} run to locate its streams by")
            continue
        have, lack = stream_inputs(cfg, guard_prompt_sets(environment(cfg, device)))
        found += [f"seed {seed}: {f}" for f in have]
        missing += [f"seed {seed}: {m}" for m in lack]
    return found, missing


def environment_config(root: Path, scope: NumericalScope, cases):
    """The config of the first case's regular run found: the environment's model and data (the
    same for every regular run of the stage; the parts list the cases they run first, then the
    selection's, so that a rerun due only for cases whose seed's FA runs were given up still
    finds one; a part whose every due case was given up builds no environment)."""
    for case in cases:
        try:
            return run_config(locate_case(root, scope, case))
        except DiagnosticError:
            continue
    sys.exit("No regular run of the check's cases is found: nothing can run")


def due_items(part, out: Path, cases, control_rates, *, retry: bool, reader: RecordReader,
              batch: int, upper: float) -> tuple[list, list[str]]:  # fmt: skip
    """What is due, read only: the cases (``twins``) or control rates
    (``control``) whose attempt is due, and the record files a real run moves aside first.
    Without ``retry``: those without a first attempt that counts (``RecordReader``), with their
    first attempt's file and any rerun's; with it: those whose counted first attempt is
    incomplete (a case not exceeding, a control not rejected) with no counted rerun, with the
    rerun's file."""
    attempts, controls = load_numerical(out, cases, control_rates)
    if part == "control":
        items = list(control_rates)
        paths = {lr: [control_path(lr, n) for n in ATTEMPTS] for lr in items}

        def present(lr) -> bool:
            return control_attempt_state(controls[lr][ATTEMPTS[0]], rate=lr, batch=batch,
                                         upper=upper, reader=reader)["present"]  # fmt: skip

        reruns = (control_retry_due(controls, control_rates, batch=batch, upper=upper,
                                    reader=reader) if retry else [])  # fmt: skip
    else:
        items = list(cases)
        paths = {case: [case.attempt_path(n) for n in ATTEMPTS] for case in items}

        def present(case) -> bool:
            return attempt_state(attempts[case.name][ATTEMPTS[0]], case, reader=reader)["present"]

        reruns = retry_due(attempts, cases, reader=reader) if retry else []
    if retry:
        aside = [paths[item][1] for item in reruns if (out / paths[item][1]).exists()]
        return reruns, aside
    due = [item for item in items if not present(item)]
    return due, [path for item in due for path in paths[item] if (out / path).exists()]


def move_aside(out: Path, relatives: list[str]) -> None:
    """Move records that do not count aside, to ``stale/<their commit>-<time>/`` (kept, never
    read), before a real run redoes them."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for relative in relatives:
        try:
            record = json.loads((out / relative).read_text())
            label = f"{output_commit(record)}{'-dirty' if output_dirty(record) else ''}"
        except (OSError, json.JSONDecodeError):
            label = "unreadable"
        set_aside(out, relative, f"{label}-{stamp}")


def code_summary(code: dict) -> str:
    """The code check (``code_check``) in one line."""
    if code["consistent"] and code.get("note"):  # nothing to compare
        identity = code["gpu_code"] or {}
        return (f"GPU code: {code['note']}; the check's own is recorded "
                f"({identity.get('files')} files, sha256 {identity.get('sha256')}, at "
                f"{identity.get('commit')})")  # fmt: skip
    if code["consistent"]:
        identity = code["gpu_code"] or {}
        return (f"GPU code: the regular runs' equals the check's ({identity.get('files')} files, "
                f"sha256 {identity.get('sha256')}, at {identity.get('commit')})")  # fmt: skip
    return "GPU CODE MISMATCH: " + "; ".join(code["problems"])


def environment_summary(check: dict) -> str:
    """The environment check (``environment_check``) in one line."""
    if check["consistent"] and check.get("note"):  # nothing to compare
        return f"run environment: {check['note']}"
    if check["consistent"]:
        here = check["here"]
        return (f"run environment: the regular runs' equals this one ({here['hardware.gpus']}, "
                f"driver {here['hardware.driver']}, CUDA {here['hardware.torch_cuda']}, Python "
                f"{here['python']})")  # fmt: skip
    return "RUN ENVIRONMENT MISMATCH: " + "; ".join(check["problems"])


def void(args: argparse.Namespace, settings) -> None:
    """``void`` (module docstring): print what it moves, refuse (exit 1) on any problem, move it
    unless ``--dry-run``."""
    root = args.runs_root or runs_root()
    found = void_plan(root, NumericalScope.of(settings), args.batch, args.upper, git_state())
    print(json.dumps({key: found[key] for key in ("batch", "U", "paths", "check_directory",
                                                  "kept", "destination")}
                     | {"old_gpu_code": found["old_gpu_code"], "head_gpu_code": found["head"]},
                     indent=1))  # fmt: skip
    if found["problems"]:
        sys.exit("void refuses: " + "; ".join(found["problems"]))
    if args.dry_run:
        return
    print(void_numerical(root, found, git_state()))


def require_clean() -> None:
    """A part that writes runs only from a clean checkout of a commit."""
    state = git_state()
    if state.get("commit") is None or state.get("dirty") is not False:
        sys.exit("The checkout is not clean (tracking.git_state): records made from it never "
                 "count, and none is written or moved from it; commit first")  # fmt: skip


def config_path(config: str) -> Path:
    """The settings file ``--config`` names, found as ``config.load_config`` finds it: a file
    relative to the working directory first, then one under configs/."""
    return _locate(config, CONFIG_ROOT)


def tracked_config(config: str) -> bool:
    """Whether ``--config`` resolves to the tracked settings (configs/experiments/
    numerical_check.yaml, unmodified when the checkout is clean), whatever the working
    directory: a file of that name there shadows the name, and is not them."""
    return config_path(config) == (CONFIG_ROOT / f"{CONFIG}.yaml").resolve()


def failed(records: dict) -> bool:
    """Whether any record of this invocation holds an error."""
    return any("traceback" in record for record in records.values())


def driver_script(
    commit: str, written_at: str, batch: int, upper: float, rates: dict, given_up=(),
    request: str | None = None,
) -> str:  # fmt: skip
    """The bash driver of the check on a Pod (module docstring). ``request``: the check
    request's path, passed to every part (``--request``) when cases are given up."""
    stamp = "$(date -u +%FT%TZ)"
    run = SCRIPT
    selection = " ".join([*(f"--rates {arm}={','.join(repr(r) for r in values)}"
                            for arm, values in rates.items()),
                          *([f"--request {shlex.quote(request)}"] if request else []),
                          *(f"--given-up {name}" for name in given_up)])  # fmt: skip
    lines = [
        "#!/usr/bin/env bash",
        *comment(f"Written by scripts/numerical_check.py driver at {written_at}; do not edit."),
        *comment(f"The numerical check of study 2's selected rates (amendment 4, section 1) at "
                 f"B {batch} and U {upper:g}, from commit {commit}: the plan, the same-path "
                 "controls at the FA rates, the twins' first attempts, the verdict, then one "
                 "rerun, in a fresh process, of every control and case left incomplete, and the "
                 "verdict again. K-only and blind to outcomes: no harmful prompt set, stream or "
                 "generation."),
        *comment("Before it runs, DATA_DIR holds the numerical seeds' training and v₀ streams at "
                 "this B and the panels (restored with scripts/pod_setup.sh), and RUNS_DIR the "
                 "regular numerical runs (study2_short, role numerical), made in this run "
                 "environment (the plan compares the GPU, driver, CUDA, Python, packages and "
                 "image each run recorded). The GPU must be free (no other compute process), the "
                 f"host must have {HOST_MEMORY_GB:g} GB of free memory, and the volume must hold "
                 f"{DISK_GB} GB for a case's checkpoints, once those an interrupted case left are "
                 "cleared."),
        *comment("Outputs: $OUT (controls, twins, verdicts/<selection digest>.json and its copy "
                 "verdict.json), with this driver's log. It can be run again: every record that "
                 "counts (written at a commit with this GPU code) is kept. Before any "
                 "non-zero exit the verdict is recorded. A control or twin that fails is "
                 "recorded and the driver goes on; it then ends with exit "
                 f"{PART_FAILED}. A verdict that holds an incomplete rate or control ends it with "
                 f"exit {INCOMPLETE}. A verdict that is not qualified is a result, not a "
                 "failure."),
        *(comment(f"Given up by the launcher, as the check request (--request) names them: "
                  f"{', '.join(given_up)}. The parts record each without a GPU, with the "
                  "request's path and sha256, and the verdict holds it not qualified "
                  "(technical) after the rerun.") if given_up else []),
        "set -euo pipefail",
        ". /etc/rp_environment",
        ". /root/zfr-env.sh",
        f"SRC=/workspace/code/zero-field-refusal/{commit}/source",
        'cd "$SRC"',
        f'test "$(git rev-parse HEAD)" = {commit}',
        f'project-run "$SRC" python -c "{CLEAN_CHECKOUT}"',
        f'OUT="${{RUNS_DIR:?}}/numerical/b{batch}-u{upper:g}"',
        'mkdir -p "$OUT"',
        'exec > >(tee -a "$OUT/driver-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1',
        f'step() {{ echo "== {stamp} start: $*"; "$@"; echo "== {stamp} done: $*"; }}',
        f"ARGS=(--batch {batch} --upper {upper:g} {selection})",
        "# Every stop records the verdict first (incomplete, with why)",
        f'stop() {{ echo "== {stamp} $1" >&2; {run} verdict "${{ARGS[@]}}" || true; exit "$2"; }}',
        "# No GPU job beside the check: no compute process, and (as a container may not see",
        "# other processes) no GPU memory in use",
        'if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then',
        '  stop "a process holds the GPU; the check does not start" 1',
        "fi",
        "used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)",
        f"if echo \"$used\" | awk '$1 + 0 >= {GPU_USED_MIB} {{busy = 1}} END {{exit !busy}}'; then",
        '  stop "GPU memory in use ($used MiB); the check does not start" 1',
        "fi",
        "# The host holds theta_0, v_0, the second model and the p1 direction, and a checkpoint",
        "# save's transient (the free memory: MemAvailable, or the cgroup's when smaller)",
        f'if ! project-run "$SRC" python -c "{HOST_FREE}" {HOST_MEMORY_GB:g}; then',
        f'  stop "{HOST_MEMORY_GB:g} GB of free host memory are needed; the check does not '
        'start" 1',
        "fi",
        "# The twins keep one case's checkpoints at a time; a case the deadline or a kill",
        "# interrupted left its own: cleared here (the records are kept)",
        'rm -rf -- "$OUT/twins/checkpoints"',
        'free_kb=$(df -Pk "$OUT" | awk \'NR == 2 {print $4}\')',
        f'if [ "$free_kb" -lt {DISK_GB * 1_000_000} ]; then',
        f'  stop "{DISK_GB} GB free are needed under $OUT, $free_kb KB are" 1',
        "fi",
        "set +e",
        f'{run} plan "${{ARGS[@]}}"',
        "rc=$?",
        "set -e",
        "# The plan fails on a missing run (unless given up) or stream, or on GPU code or a run",
        "# environment unlike the regular runs'",
        'if [ "$rc" -ne 0 ]; then stop "the plan found a problem (exit $rc)" 1; fi',
        "# The controls, then the twins' first attempts: a failure is recorded (incomplete) and",
        "# the driver goes on",
        "set +e",
        f'{run} control "${{ARGS[@]}}"',
        "control=$?",
        f'{run} twins "${{ARGS[@]}}"',
        "twins=$?",
        "set -e",
        f'step {run} verdict "${{ARGS[@]}}"',
        "# One rerun, in a fresh process, of every control and case whose first attempt is",
        "# incomplete",
        "set +e",
        f'{run} verdict "${{ARGS[@]}}" --retry-needed',
        "due=$?",
        "set -e",
        "rerun_control=0",
        "rerun_twins=0",
        'case "$due" in',
        "  0)",
        "    set +e",
        f'    {run} control "${{ARGS[@]}}" --retry',
        "    rerun_control=$?",
        f'    {run} twins "${{ARGS[@]}}" --retry',
        "    rerun_twins=$?",
        "    set -e",
        f'    step {run} verdict "${{ARGS[@]}}"',
        "    ;;",
        f'  {NOT_DUE}) echo "== {stamp} no rerun is due" ;;',
        '  *) stop "verdict --retry-needed failed (exit $due)" "$due" ;;',
        "esac",
        f'qualified=$(project-run "$SRC" python -c "{READ_FIELD}" "$OUT/verdict.json" '
        "qualified)",
        f'echo "== {stamp} the verdict: qualified $qualified ($OUT/verdict.json)"',
        f'if project-run "$SRC" python -c "{HAS_INCOMPLETE}" "$OUT/verdict.json"; then',
        f'  echo "== {stamp} the verdict holds an incomplete rate or control" >&2',
        f"  exit {INCOMPLETE}",
        "fi",
        'failures="$control $twins $rerun_control $rerun_twins"',
        'if [ "$failures" != "0 0 0 0" ]; then',
        f'  echo "== {stamp} a part failed (control, twins, their reruns: $failures); recorded" '
        ">&2",
        f"  exit {PART_FAILED}",
        "fi",
        f'echo "== {stamp} the numerical check is complete: $OUT"',
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def main(*, untracked_config_for_tests: bool = False) -> None:
    """The command line (module docstring). ``untracked_config_for_tests`` lets the tests run
    the writing parts on settings other than the tracked ones; nothing on the command line
    does."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["plan", "control", "twins", "verdict", "driver", "void"])
    parser.add_argument("--rates", action="append", default=[], metavar="OPTIMIZER=R1,R2,R3")
    parser.add_argument("--given-up", action="append", default=[], metavar="CASE",
                        help="with --rates: a case whose regular run the launcher gave up "
                             "on")  # fmt: skip
    parser.add_argument("--selection", type=Path, help="a selection record (JSON)")
    parser.add_argument("--request", type=Path, metavar="FILE",
                        help="with --rates and --given-up: the check request (selection record) "
                             "naming them, the evidence of the cases given up")  # fmt: skip
    parser.add_argument("--batch", type=int, required=True, help="the selected B")
    parser.add_argument("--upper", type=float, required=True, help="the selected U")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--runs-root", type=Path, default=None, help="RUNS_DIR")
    parser.add_argument("--out", type=Path, default=None, help="output directory (driver: file)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-stream-check", action="store_true", help="plan: skip DATA_DIR")
    parser.add_argument("--retry", action="store_true", help="control, twins: the reruns")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--retry-needed", action="store_true")
    parser.add_argument("--commit")
    args = parser.parse_args()

    settings = load_config(config_path(args.config))  # the file tracked_config judges
    scope = NumericalScope.of(settings)
    if args.part == "void":  # no rates: every numerical run at B and the check of (B, U)
        if args.rates or args.selection or args.given_up or args.request:
            sys.exit("void takes --batch and --upper only: it moves every numerical run at B "
                     "whose GPU code is not HEAD's")  # fmt: skip
        if not (args.dry_run or tracked_config(args.config) or untracked_config_for_tests):
            sys.exit(f"void moves runs only under the tracked settings ({CONFIG}), not "
                     f"{config_path(args.config)}; --dry-run only prints")  # fmt: skip
        try:
            void(args, settings)
        except NumericalError as exc:
            sys.exit(str(exc))
        return
    try:
        selected, given_up = rates_of(args)
        rates = selected_rates(scope, selected)
        cases = numerical_cases(settings, rates, args.upper, batch=args.batch)
        check_given_up(given_up, cases)
        default_out = numerical_root(args.batch, args.upper, args.runs_root)
        # The evidence of the cases given up, for the parts that record them
        request = None if args.part == "verdict" else request_of(args, scope, rates, given_up)
    except NumericalError as exc:
        sys.exit(str(exc))
    missing_request = ("the check request naming the cases given up (--selection FILE, or "
                       "--request FILE beside --rates and --given-up): their records state its "
                       "path and sha256")  # fmt: skip
    if args.part == "driver":
        if not (args.dry_run or tracked_config(args.config) or untracked_config_for_tests):
            sys.exit(f"driver writes a driver only under the tracked settings ({CONFIG}), not "
                     f"{config_path(args.config)}; --dry-run only prints")  # fmt: skip
        script = driver_script(args.commit or "<SHA>", git_state().get("commit")
                               or datetime.now(UTC).isoformat(timespec="seconds"),
                               args.batch, args.upper, rates, given_up,
                               request["path"] if request and given_up else None)  # fmt: skip
        if args.dry_run:
            print(script, end="")
            return
        if given_up and request is None:
            sys.exit(f"driver: cases are given up, and the driver needs {missing_request}")
        if not args.commit or not args.out or args.out.exists():
            sys.exit("driver needs --commit SHA and an --out FILE that does not exist")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(script)
        print(args.out)
        return

    out = args.out or default_out
    root = args.runs_root or runs_root()
    state = git_state()
    reader = RecordReader(scope, state, root)
    writes = args.part in ("control", "twins", "verdict") and not (
        args.dry_run or args.retry_needed
    )

    def refuse_untracked() -> None:  # no record under other settings
        if writes and not tracked_config(args.config) and not untracked_config_for_tests:
            sys.exit(f"{args.part} writes records only under the tracked settings ({CONFIG}), "
                     f"not {config_path(args.config)}; --dry-run only prints")  # fmt: skip

    # The regular runs' GPU code against the check's; for the parts that run the check, their
    # recorded run environment against this one.
    # With no run located because every case is given up, both are skipped
    # with a note, and the check's own GPU code is recorded.
    located = find_runs(root, scope, cases, given_up)[0]
    found_runs = list(located.values())
    every_given_up = all(case.name in given_up for case in cases)
    skipped = NONE_LOCATED if not found_runs and every_given_up else None
    code = code_check(found_runs, state.get("commit"), skipped=skipped)
    machine = (environment_check(found_runs, skipped=skipped)
               if args.part in ("plan", "control", "twins") else None)  # fmt: skip
    absent = [case for case in cases if case.name in given_up and case.name not in located]
    if args.part == "verdict":
        verdict = numerical_verdict(settings, out, rates=rates, batch=args.batch,
                                    upper=args.upper, code=code, root=root,
                                    state=state)  # fmt: skip
        print(json.dumps({key: verdict[key] for key in ("qualified", "rates", "controls")}
                         | {"retry_due": verdict["provenance"]["retry_due"]}))  # fmt: skip
        if args.retry_needed:
            due = verdict["provenance"]["retry_due"]
            sys.exit(0 if due["cases"] or due["controls"] else NOT_DUE)
        if args.dry_run:
            return
        require_clean()
        refuse_untracked()
        print(write_verdict(out, verdict, rates))
        return
    if args.part == "plan":
        text, complete = plan(settings, args, cases, root, out, given_up)
        print(text)
        print(code_summary(code))
        print(environment_summary(machine))
        if request is not None:
            print(f"check request: {request['path']} (sha256 {request['sha256']})")
        if args.dry_run:
            return
        missing = [] if code["consistent"] else ["the GPU code"]
        missing += [] if machine["consistent"] else ["the run environment"]
        missing += [missing_request] if absent and request is None else []
        if not args.no_stream_check:
            found, streams = check_streams(settings, cases, root, "cpu", given_up)
            missing += streams
            print("\n".join([*(f"found: {f}" for f in found),
                             *(f"MISSING: {m}" for m in missing)]))  # fmt: skip
        if missing or not complete:
            sys.exit(1)
        return

    control_rates = rates[scope.control_arm]
    attempt = ATTEMPTS[1] if args.retry else ATTEMPTS[0]
    due, aside = due_items(args.part, out, cases, control_rates, retry=args.retry,
                           reader=reader, batch=args.batch, upper=args.upper)  # fmt: skip
    names = [f"{lr:g}" for lr in due] if args.part == "control" else [c.name for c in due]
    print(json.dumps({"part": args.part, "attempt": attempt, "due": names, "set_aside": aside}))
    # What is due for a case given up whose run is not located is recorded
    # without a GPU, with the check request as its evidence; the rest runs as before
    gone = given_up_unlocated(args.part, due, given_up, located, scope, args.batch, args.upper)
    rest = [item for item in due if item not in gone]
    if args.dry_run:
        if gone:
            shown = [f"{lr:g}" for lr in gone] if args.part == "control" else [c.name for c in gone]
            print(f"given up by the launcher, recorded without a GPU: {shown}")
        if args.part == "control":
            minutes = allowed = total_cost(args.batch, scope, [], len(rest))[0]
        else:
            minutes, allowed = total_cost(args.batch, scope, rest, 0)
        print(f"≈ {minutes:.0f} GPU min, ≈ {allowed:.0f} with a lost full-float64 reference per "
              f"AdamW and pre-warmed case; output {out}")  # fmt: skip
        print(code_summary(code))
        print(environment_summary(machine))
        return
    if not due:
        return
    require_clean()
    refuse_untracked()
    if not code["consistent"]:  # nothing runs against other GPU code
        print(code_summary(code), file=sys.stderr)
        sys.exit(CODE_MISMATCH)
    if not machine["consistent"]:  # nor in another environment
        print(environment_summary(machine), file=sys.stderr)
        sys.exit(CODE_MISMATCH)
    if gone and request is None:
        sys.exit(f"{args.part}: cases are given up, and their records need {missing_request}")
    move_aside(out, aside)  # only now: a real run, clean, under the check's GPU code

    done: dict = {}
    noted = {"checks_skipped": code["note"]} if code.get("note") else {}
    if gone and args.part == "control":
        done |= given_up_controls(settings, root, out, rates=gone, batch=args.batch,
                                  upper=args.upper, attempt=attempt, code=code, request=request,
                                  **noted)  # fmt: skip
    elif gone:
        done |= given_up_twins(settings, root, out, gone, attempt=attempt, code=code,
                               request=request, **noted)  # fmt: skip
    if rest:  # the environment, and the GPU, only for these
        from zero_field_refusal.runtime import environment

        env_cases = (
            [c for c in cases if c.seed == scope.control_seed] if args.part == "control" else rest
        )
        cfg = environment_config(root, scope, [*env_cases, *cases])
        env = guard_prompt_sets(environment(cfg, args.device))
        rule = apply_memory_rule(env.device, scope.memory_rule_gb)
        print(json.dumps({"memory_rule": rule}))
        if args.part == "control":
            done |= run_numerical_controls(settings, env, root, out, rates=rest,
                                           batch=args.batch, upper=args.upper, attempt=attempt,
                                           code=code)  # fmt: skip
        else:
            done |= run_numerical_twins(settings, env, root, out, rest, attempt=attempt,
                                        code=code, keep_checkpoints=args.keep_checkpoints,
                                        setups=cases)  # fmt: skip
    if args.part == "control":
        print(json.dumps({lr: {key: r.get(key) for key in ("status", "state", "max_K_scoring",
                                                           "reason")}
                          for lr, r in done.items()}))  # fmt: skip
    else:
        print(json.dumps({name: {key: r.get(key) for key in ("status", "pair_states",
                                                             "max_D_twin", "reason")}
                          for name, r in done.items()}))  # fmt: skip
    if failed(done):
        sys.exit(PART_FAILED)


if __name__ == "__main__":
    main()
