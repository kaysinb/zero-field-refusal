"""Run the diagnostic of the FA soft-control failure (docs/study2-fa-control-diagnostic.md).

    python scripts/fa_control_diagnostic.py plan [--runs-root DIR] [--no-stream-check]
    python scripts/fa_control_diagnostic.py (d1 | d2) [--runs-root DIR] [--out DIR] [--device D]
        [--dry-run]
    python scripts/fa_control_diagnostic.py d3 [--expansion] [--keep-checkpoints] [--runs-root DIR]
        [--out DIR] [--device D] [--dry-run]
    python scripts/fa_control_diagnostic.py branch [--expansion-needed] [--out DIR]
    python scripts/fa_control_diagnostic.py driver --commit SHA --out FILE

Outside the confirmation and blind to outcomes: only K, KL between twins, norms, eigenvalues and
v₀ statistics are computed and written, every evaluation on benign panels; no harmful prompt set,
stream or generation is touched (src/zero_field_refusal/fa_control_diagnostic.py), and every
prompt set whose name contains "harmful" raises if any code asks for it (``guard_prompt_sets``).
The settings are configs/experiments/fa_control_diagnostic.yaml (``--config``).

- The runs re-examined are found under ``--runs-root`` (RUNS_DIR) by their archived configs:
  study2_short (the regular runs of each case) and study2_short_fa_controls (the FA soft
  controls). Their streams must be in DATA_DIR, as for the controls (docs/operations.md).
- Each part writes under ``--out`` (default RUNS_DIR/diagnostics/fa_control/): ``d1/result.json``,
  ``d2/result.json``, ``d3/determinism.json`` and ``d3/<case>.json``, ``branch/branch.json``. A
  part already written at this commit is kept: running again skips it (D3 case by case). One
  written at another commit is moved aside to ``stale/<its commit>/`` and redone, and ``branch``
  never reads one.
- Every part runs under the memory rule (``memory_rule_gb``: 46 GB reserved, in the GB of the
  run records' peaks, 1e9 bytes): the process's CUDA memory is capped there. An out-of-memory
  error under the cap makes the float64 reference fall back (recorded); any other part fails
  (exit 4).
- ``d3`` runs the determinism check, then the cases in the settings' order; with
  ``--expansion`` only the expansion case (FA 1.46e-5), which the specification adds once when
  the three cases give the ambiguous branch. A failed determinism check stops D3 (exit 2).
- ``d2`` records any failure in ``d2/failed.json`` (the error, the peaks, the commit) before it
  exits; a later success renames that record, which is kept. No branch reads D2.
- ``branch`` reads the outputs and writes the branch; ``--expansion-needed`` exits 0 when the
  expansion case is due, 3 when it is not; ``--expansion-failed CODE`` records in the facts that
  the expansion case failed, the outcome staying the three cases'.
- ``--dry-run`` (and ``plan`` for every part) finds the runs and their records and prints what
  would run, with the cost estimate, without loading a model. ``plan`` also checks the DATA_DIR
  streams D1–D3 read, unless ``--no-stream-check``, and exits 1 when one is missing.
- ``driver`` writes the bash driver of all parts for a Pod, from the source checkout of SHA.

Cost on one L40S at 1.5B (fp32), from the targeted round's records (≈ 0.85 s per training step
at B 5 with its monitor readings; the step-25 scoring diagnostic 14.5 s with its cache and a
monitor reading 2.7 s; 30.7 GB allocated at that state) and the Stage-A curvature run (≈ 1.8 s
per Hessian–vector product at B 5). Peaks are allocated GB; reserved memory is capped at 46 GB.
- D1 ≈ 12 min, ≈ 38 GB.
- D2 ≈ 50 min: ≈ 1250 products (≈ 38 min), the Ritz vectors through host memory (≈ 7 min) and
  the setup; ≈ 40 GB, or ≈ 45 when a Ritz vector fits beside the Lanczos vectors under the cap.
- D3: the setup and the determinism check ≈ 8 min; ≈ 25 min per case: 30 D_twin readings of
  ≈ 30 s each (two uncached forwards over 400 responses, a 5.2 GB checkpoint load, two 6.2 GB
  moves), 6 twins of 25 steps, the checkpoint saves. ≈ 31 GB for the twins, ≈ 38 GB for FA's
  float64 reference, ≈ 44 GB for AdamW's, which falls back to float64 loss and accumulation if
  it meets the cap. Three cases ≈ 83 min; the expansion ≈ 30 more, its own setup included.
- In all ≈ 145 GPU min (≈ 2.4 h), ≈ 175 (≈ 2.9 h) with the expansion.
- 31.4 GB of checkpoints per case on the volume, deleted when the case is done; host memory up
  to ≈ 40 GB in D3.
"""

import argparse
import dataclasses
import json
import sys
import textwrap
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch

from zero_field_refusal.config import load_config
from zero_field_refusal.fa_control_diagnostic import (
    DeterminismFailure,
    apply_memory_rule,
    cases_of,
    check_inputs,
    decide_branch,
    kept,
    load_outputs,
    locate_runs,
    output_root,
    peak_memory,
    provenance,
    run_config,
    run_d1,
    run_d2,
    run_d3,
    write_json,
)
from zero_field_refusal.paths import runs_root
from zero_field_refusal.tracking import CLEAN_CHECKOUT, git_state

CONFIG = "experiments/fa_control_diagnostic"
SCRIPT = 'project-run "$SRC" python scripts/fa_control_diagnostic.py'
NOT_DUE = 3  # ``branch --expansion-needed``: the expansion case is not due
DETERMINISM_FAILED = 2  # ``d3``: the re-run differs from its archive
OUT_OF_MEMORY = 4  # a part met the memory rule's cap
INCOMPLETE = 5  # the driver: the branch is incomplete
D2_FAILED = 6  # the driver: D2 failed (recorded), the rest ran
GPU_USED_MIB = 1000  # the driver starts only below this much GPU memory in use
# The driver's programs (python -c, through project-run): the branch written, and a D2 failure
# record when the D2 process left none (killed, say). No $, backtick or backslash inside.
READ_BRANCH = "import json, sys; print(json.load(open(sys.argv[1]))['branch'])"
D2_FAILURE_RECORD = (
    "import json, sys; from datetime import UTC, datetime; "
    "from zero_field_refusal.tracking import git_state; "
    "json.dump({'part': 'd2', 'error': 'exit ' + sys.argv[2] + ', no failure record of its own', "
    "'written_at': datetime.now(UTC).isoformat(timespec='seconds'), "
    "'provenance': {'code': git_state()}}, open(sys.argv[1], 'w'), indent=1)"
)
COSTS = {  # GPU minutes and peak allocated GB on one L40S (module docstring)
    "d1": (12, 38), "d2": (50, 45), "d3-setup": (8, 31), "d3-expansion-setup": (5, 31),
    "d3-case": (25, 44),
}  # fmt: skip
DISK_GB = 40  # the free space D3 needs: one case's checkpoints (31.4 GB at 1.5B) and a margin
# The inventories a Pod restores before the diagnostic: every Pod's of the short-regime stage
# after the prep of 2026-10-16, as for the FA controls (scripts/study2_short_campaigns.py,
# docs/operations.md "Prerequisites on a Pod"); the diagnostic reads seed 2000's streams only.
INVENTORIES = (
    "inventory-2026-10-11.tsv",
    "inventory-2026-10-14-study2-prompts.tsv",
    "inventory-2026-10-14-study2-a-phase0.tsv",
    "inventory-2026-10-15-study2-monitor-v2.tsv",
    "inventory-2026-10-14-study2-a-preconditioner-s2000.tsv",
    "inventory-2026-10-14-study2-a-preconditioner-s2001.tsv",
    "inventory-2026-10-14-study2-a-preconditioner-s2002.tsv",
    "inventory-2026-10-16-study2-a-stream-s2000-final.tsv",
    "inventory-2026-10-15-study2-a-stream-s2001-search-s2001-2.tsv",
    "inventory-2026-10-16-study2-a-stream-s2002-final.tsv",
    "inventory-2026-10-16-study2-stage0-v3-record.tsv",
    "inventory-2026-10-16-study2-curvature-panel.tsv",
)


def comment(text: str) -> list[str]:
    return textwrap.wrap(text, 100, initial_indent="# ", subsequent_indent="# ",
                         break_long_words=False, break_on_hyphens=False)  # fmt: skip


def guard_prompt_sets(env):
    """The environment with every prompt set whose name contains "harmful" refused: the
    diagnostic is K-only on benign panels, and this makes any path to a harmful set raise."""
    original = env.prompt_set

    def prompt_set(name: str):
        if "harmful" in name:
            raise RuntimeError(f"The FA control diagnostic must not read the prompt set {name!r}")
        return original(name)

    return dataclasses.replace(env, prompt_set=prompt_set)


def driver_script(commit: str, written_at: str) -> str:
    """The bash driver of the diagnostic on a Pod (module docstring)."""
    stamp = "$(date -u +%FT%TZ)"
    lines = [
        "#!/usr/bin/env bash",
        *comment(f"Written by scripts/fa_control_diagnostic.py driver at {written_at}; do not "
                 "edit."),
        *comment(f"The diagnostic of the FA soft-control failure (docs/study2-fa-control-"
                 f"diagnostic.md), from commit {commit}: D4/D1, D2, then D3 (the determinism "
                 "check, then FA 1.21e-5, FA 1e-5 and AdamW 1.32e-5), the branch, and FA 1.46e-5 "
                 "only if the branch is ambiguous. K-only and blind to outcomes: no harmful "
                 "prompt set, stream or generation."),
        *comment("Before it runs, DATA_DIR is restored with scripts/pod_setup.sh from "
                 f"{', '.join(INVENTORIES)} under runs/archive/, and RUNS_DIR holds the records "
                 "of the targeted round (study2_short) and of the FA soft controls "
                 "(study2_short_fa_controls), as on P1. The GPU must be free (no other compute "
                 f"process), and the volume must hold {DISK_GB} GB for D3's checkpoints."),
        *comment("Outputs: $RUNS_DIR/diagnostics/fa_control/ (d1, d2, d3, branch), with this "
                 "driver's log. It can be run again: every part already written at this commit "
                 "is kept. It stops at the first failure of D1 or D3, except that a failed "
                 "determinism check or expansion case still records the branch first. A D2 "
                 "failure is recorded (d2/failed.json) and the driver goes on, since no branch "
                 f"reads D2, then exits {D2_FAILED} at the end. An incomplete branch ends the "
                 f"driver with exit {INCOMPLETE}."),
        "set -euo pipefail",
        ". /etc/rp_environment",
        ". /root/zfr-env.sh",
        f"SRC=/workspace/code/zero-field-refusal/{commit}/source",
        'cd "$SRC"',
        f'test "$(git rev-parse HEAD)" = {commit}',
        f'project-run "$SRC" python -c "{CLEAN_CHECKOUT}"',
        'OUT="${RUNS_DIR:?}/diagnostics/fa_control"',
        'mkdir -p "$OUT"',
        'exec > >(tee -a "$OUT/driver-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1',
        f'step() {{ echo "== {stamp} start: $*"; "$@"; echo "== {stamp} done: $*"; }}',
        "# No GPU job beside the diagnostic: no compute process, and (as a container may not see",
        "# other processes) no GPU memory in use",
        'if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then',
        f'  echo "== {stamp} a process holds the GPU; the diagnostic does not start" >&2',
        "  exit 1",
        "fi",
        "used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)",
        f"if echo \"$used\" | awk '$1 + 0 >= {GPU_USED_MIB} {{busy = 1}} END {{exit !busy}}'; then",
        f'  echo "== {stamp} GPU memory in use ($used MiB); the diagnostic does not start" >&2',
        "  exit 1",
        "fi",
        "# D3 keeps one case's checkpoints at a time",
        'free_kb=$(df -Pk "$OUT" | awk \'NR == 2 {print $4}\')',
        f'if [ "$free_kb" -lt {DISK_GB * 1_000_000} ]; then',
        f'  echo "== {stamp} {DISK_GB} GB free are needed under $OUT, $free_kb KB are" >&2',
        "  exit 1",
        "fi",
        "# The steps, in the specification's order",
        f"step {SCRIPT} plan",
        f"step {SCRIPT} d1",
        "# D2: a failure is recorded and the driver goes on; no branch reads D2",
        f'echo "== {stamp} start: d2"',
        "set +e",
        f"{SCRIPT} d2",
        "d2=$?",
        "set -e",
        'if [ "$d2" -ne 0 ]; then',
        f'  echo "== {stamp} d2 failed with code $d2; recorded, D3 goes on" >&2',
        '  if [ ! -f "$OUT/d2/failed.json" ]; then',
        '    mkdir -p "$OUT/d2"',
        f'    project-run "$SRC" python -c "{D2_FAILURE_RECORD}" "$OUT/d2/failed.json" "$d2"',
        "  fi",
        "else",
        f'  echo "== {stamp} done: d2"',
        "fi",
        f'echo "== {stamp} start: d3"',
        "set +e",
        f"{SCRIPT} d3",
        "rc=$?",
        "set -e",
        'if [ "$rc" -ne 0 ]; then',
        f'  if [ "$rc" -eq {DETERMINISM_FAILED} ]; then step {SCRIPT} branch; fi',
        '  exit "$rc"',
        "fi",
        f'echo "== {stamp} done: d3"',
        "set +e",
        f"{SCRIPT} branch --expansion-needed",
        "due=$?",
        "set -e",
        'case "$due" in',
        "  0)",
        f'    echo "== {stamp} start: d3 --expansion"',
        "    set +e",
        f"    {SCRIPT} d3 --expansion",
        "    rc=$?",
        "    set -e",
        '    if [ "$rc" -ne 0 ]; then',
        f'      step {SCRIPT} branch --expansion-failed "$rc"',
        '      exit "$rc"',
        "    fi",
        f'    echo "== {stamp} done: d3 --expansion"',
        "    ;;",
        f'  {NOT_DUE}) echo "== {stamp} the expansion case is not due" ;;',
        '  *) exit "$due" ;;',
        "esac",
        f"step {SCRIPT} branch",
        f'outcome=$(project-run "$SRC" python -c "{READ_BRANCH}" "$OUT/branch/branch.json")',
        f'echo "== {stamp} the branch: $outcome"',
        f'if [ "$outcome" = incomplete ]; then exit {INCOMPLETE}; fi',
        f'if [ "$d2" -ne 0 ]; then echo "== {stamp} d2 failed: $OUT/d2/failed.json" >&2; '
        f"exit {D2_FAILED}; fi",
        f'echo "== {stamp} the diagnostic is complete: $OUT"',
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def plan(settings, runs, parts: list[str], out: Path) -> str:
    """What the parts would run, and their cost; nothing is loaded."""
    d = settings.diagnostic
    cases = cases_of(settings)
    lines = [f"output: {out}", "runs:"]
    for name, path in runs.regular.items():
        cfg = run_config(path)
        lines.append(f"  {name}: {path} ({cfg.optimizer.name}, lr {cfg.optimizer.lr}, "
                     f"{cfg.train.steps} steps, B {cfg.train.batch_size})")  # fmt: skip
    for rate, path in runs.controls.items():
        lines.append(f"  control {rate:g}: {path}")
    for path in [*runs.regular.values(), *runs.controls.values()]:
        for record in ("metrics.jsonl", "monitor.jsonl"):
            if not (path / record).is_file():
                raise SystemExit(f"{path / record} is missing")
    minutes = gb = 0.0
    for part in parts:
        if part == "d1":
            lines.append(f"d1: shared state of {d.d1.pair} and its control; v₀ census; "
                         "residual (i)-(iii); same-path float64 control at "
                         f"{d.d1.same_path_control.rate:g}")  # fmt: skip
            cost = [COSTS["d1"]]
        elif part == "d2":
            lines.append(f"d2: {d.d2.batches} batches × (exact soft-loss Hessian, sampled-NLL "
                         f"Hessian) Lanczos {d.d2.lanczos.steps} steps + Ritz vectors; "
                         f"linearized runs at {list(d.d2.rates)}")  # fmt: skip
            cost = [COSTS["d2"]]
        else:
            expansion = part == "d3-expansion"
            names = [str(d.d3.expansion)] if expansion else list(d.d3.order)
            listed = [(n, cases[n].lr, cases[n].control_lr) for n in names]
            steps = list(d.d3.eval_at)
            lines.append(f"{part}: determinism on {d.d3.determinism}; cases {listed}; pairs "
                         f"p1-p4 and the float64 reference at steps {steps}")  # fmt: skip
            setup = COSTS["d3-expansion-setup" if expansion else "d3-setup"]
            cost = [setup] + [COSTS["d3-case"]] * len(names)
        minutes += sum(c[0] for c in cost)
        gb = max(gb, *(c[1] for c in cost))
        lines.append(f"  cost ≈ {sum(c[0] for c in cost)} GPU min, peak ≈ "
                     f"{max(c[1] for c in cost)} GB allocated")  # fmt: skip
    rule, k_star, bound = float(d.memory_rule_gb), float(d.k_star), float(d.bound)
    lines.append(f"total ≈ {minutes:.0f} GPU min, peak ≈ {gb:.0f} GB allocated, reserved capped "
                 f"at {rule:g} GB (K* = {k_star:.4g}, bound {bound:.4g})")  # fmt: skip
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["plan", "d1", "d2", "d3", "branch", "driver"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--runs-root", type=Path, default=None, help="RUNS_DIR")
    parser.add_argument("--out", type=Path, default=None, help="output directory (driver: file)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-stream-check", action="store_true", help="plan: skip DATA_DIR")
    parser.add_argument("--expansion", action="store_true", help="d3: the expansion case only")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--expansion-needed", action="store_true")
    parser.add_argument("--expansion-failed", type=int, default=None, metavar="EXIT_CODE",
                        help="branch: the expansion case failed with this exit code")  # fmt: skip
    parser.add_argument("--commit")
    args = parser.parse_args()

    if args.part == "driver":
        if not args.commit or not args.out or args.out.exists():
            sys.exit("driver needs --commit SHA and an --out FILE that does not exist")
        stamp = git_state().get("commit") or datetime.now(UTC).isoformat(timespec="seconds")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(driver_script(args.commit, stamp))
        print(args.out)
        return

    settings = load_config(args.config)
    d = settings.diagnostic
    commit = git_state().get("commit")
    out = args.out or output_root(args.runs_root)
    if args.part == "branch":
        d1, determinism, cases = load_outputs(out)
        decision = decide_branch(settings, d1, determinism, cases, commit=commit)
        if args.expansion_failed is not None:  # the outcome stays the three cases'
            decision["facts"]["expansion_failure"] = {
                "case": str(d.d3.expansion), "exit_code": args.expansion_failed,
                "note": "the expansion case did not complete; the outcome is the three cases'",
            }  # fmt: skip
        print(json.dumps({**{k: decision[k] for k in ("branch", "expand", "missing")},
                          "incomplete": decision["facts"]["incomplete"]}))  # fmt: skip
        if args.expansion_needed:
            sys.exit(0 if decision["expand"] else NOT_DUE)
        write_json(out / "branch" / "branch.json", decision)
        return

    from zero_field_refusal.runtime import environment

    root = args.runs_root or runs_root()
    runs = locate_runs(settings, root)
    cfg = run_config(runs.regular[str(d.d3.determinism)])
    parts = {"plan": ["d1", "d2", "d3"], "d1": ["d1"], "d2": ["d2"],
             "d3": ["d3-expansion" if args.expansion else "d3"]}[args.part]  # fmt: skip
    if args.part == "plan" or args.dry_run:
        print(plan(settings, runs, parts, out))
        if args.part == "plan" and not args.no_stream_check:
            found, missing = check_inputs(settings, guard_prompt_sets(environment(cfg, "cpu")),
                                          runs)  # fmt: skip
            print("\n".join([*(f"found: {f}" for f in found),
                             *(f"MISSING: {m}" for m in missing)]))  # fmt: skip
            if missing:
                sys.exit(1)
        return
    if args.part in ("d1", "d2") and kept(out, f"{args.part}/result.json", commit) is not None:
        print(f"kept: {out / args.part / 'result.json'} was written at {commit}")
        return
    names = [str(d.d3.expansion)] if args.expansion else [str(c) for c in d.d3.order]
    if args.part == "d3":
        record = kept(out, "d3/determinism.json", commit)
        if record is not None and not record["equal"]:
            print(f"D3 stopped: {out / 'd3' / 'determinism.json'} records a re-run that differs",
                  file=sys.stderr)  # fmt: skip
            sys.exit(DETERMINISM_FAILED)
        if record is not None and all(kept(out, f"d3/{n}.json", commit) is not None
                                      for n in names):  # fmt: skip
            print(f"kept: {names} were written under {out / 'd3'} at {commit}")
            return

    env = guard_prompt_sets(environment(cfg, args.device))
    rule = apply_memory_rule(env.device, float(d.memory_rule_gb))
    print(json.dumps({"memory_rule": rule}))
    try:
        if args.part == "d1":
            result = run_d1(settings, env, runs, out)
            print(json.dumps({"status": result["status"], "reasons": result.get("reasons", []),
                              "residual_norms": result["residual"]["norms"],
                              "path_criterion": result["residual"]["path_criterion"]["state"],
                              "same_path_control": result["same_path_control"]["state"],
                              "same_path_control_max_K":
                                  result["same_path_control"]["max_K_scoring"],
                              "v0_equal": result["shared_state"]["v0"]["equal"]}))  # fmt: skip
        elif args.part == "d2":
            failed = out / "d2" / "failed.json"
            try:
                result = run_d2(settings, env, runs, out)
            except BaseException as exc:  # recorded whatever it is, then raised as it was
                write_json(failed, {
                    "part": "d2", "error": f"{type(exc).__name__}: {str(exc)[:4000]}",
                    "traceback": traceback.format_exc()[-8000:], **peak_memory(env.device),
                    "provenance": provenance(settings, runs),
                })  # fmt: skip
                raise
            if failed.exists():  # an earlier attempt's failure, kept under another name
                failed.rename(failed.with_name(f"failed-before-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
                                               ".json"))  # fmt: skip
            print(json.dumps({kind: [round(r["top"], 6) for r in rows]
                              for kind, rows in result["spectra"].items()}))  # fmt: skip
        else:
            try:
                done = run_d3(settings, env, runs, out, cases=names,
                              keep_checkpoints=args.keep_checkpoints)  # fmt: skip
            except DeterminismFailure as exc:
                print(f"D3 stopped: {exc}", file=sys.stderr)
                sys.exit(DETERMINISM_FAILED)
            print(json.dumps({name: {"status": r.get("status"),
                                     "pair_states": r.get("pair_states"),
                                     "max_D_twin": r.get("max_D_twin")}
                              for name, r in done.items()}))  # fmt: skip
    except torch.OutOfMemoryError as exc:
        first = str(exc).splitlines()[0] if str(exc) else ""
        print(f"{args.part} failed: out of memory under the {rule['limit_gb']:g} GB rule: {first}",
              file=sys.stderr)  # fmt: skip
        sys.exit(OUT_OF_MEMORY)


if __name__ == "__main__":
    main()
