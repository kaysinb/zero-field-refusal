"""Run the diagnostic of the FA soft-control failure (docs/study2-fa-control-diagnostic.md).

    python scripts/fa_control_diagnostic.py plan [--runs-root DIR]
    python scripts/fa_control_diagnostic.py (d1 | d2) [--runs-root DIR] [--out DIR] [--device D]
        [--dry-run]
    python scripts/fa_control_diagnostic.py d3 [--expansion] [--keep-checkpoints] [--runs-root DIR]
        [--out DIR] [--device D] [--dry-run]
    python scripts/fa_control_diagnostic.py branch [--expansion-needed] [--out DIR]
    python scripts/fa_control_diagnostic.py driver --commit SHA --out FILE

Outside the confirmation and blind to outcomes: only K, KL between twins, norms, eigenvalues and
v₀ statistics are computed and written, every evaluation on benign panels; no harmful prompt set,
stream or generation is touched (src/zero_field_refusal/fa_control_diagnostic.py). The settings
are configs/experiments/fa_control_diagnostic.yaml (``--config``).

- The runs re-examined are found under ``--runs-root`` (RUNS_DIR) by their archived configs:
  study2_short (the regular runs of each case) and study2_short_fa_controls (the FA soft
  controls). Their streams must be in DATA_DIR, as for the controls (docs/operations.md).
- Each part writes under ``--out`` (default RUNS_DIR/diagnostics/fa_control/): ``d1/result.json``,
  ``d2/result.json``, ``d3/determinism.json`` and ``d3/<case>.json``, ``branch/branch.json``. A
  part already written is kept: running again skips it (D3 case by case).
- ``d3`` runs the determinism check, then the cases in the settings' order; with
  ``--expansion`` only the expansion case (FA 1.46e-5), which the specification adds once when
  the three cases give the ambiguous branch. A failed determinism check stops D3 (exit 2).
- ``branch`` reads the outputs and writes the branch; ``--expansion-needed`` exits 0 when the
  expansion case is due, 3 when it is not.
- ``--dry-run`` (and ``plan`` for every part) finds the runs and their records and prints what
  would run, with the cost estimate, without loading a model.
- ``driver`` writes the bash driver of all parts for a Pod, from the source checkout of SHA.

Cost on one L40S at 1.5B (fp32), from the targeted round's records (≈ 0.85 s per training step
at B 5 with its monitor readings, peaks 30.7 GB allocated at the step-25 diagnostic) and the
Stage-A curvature run (≈ 1.8 s per Hessian–vector product at B 5): D1 ≈ 10 min, peak ≈ 38 GB;
D2 ≈ 35–45 min (≈ 1250 products), peak ≈ 40 GB; D3 ≈ 20 min per case and ≈ 5 min for the
determinism check, ≈ 65 min for three cases (+20 with the expansion), peak ≈ 31 GB for the
twins, ≈ 38 GB for FA's float64 reference and ≈ 43 GB for AdamW's (which falls back to float64
loss and accumulation if it does not fit); 31 GB of checkpoints per case on the volume, deleted
when the case is done; host memory up to ≈ 40 GB in D3.
"""

import argparse
import json
import math
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

from zero_field_refusal.config import load_config
from zero_field_refusal.fa_control_diagnostic import (
    DeterminismFailure,
    cases_of,
    decide_branch,
    load_outputs,
    locate_runs,
    output_root,
    run_config,
    run_d1,
    run_d2,
    run_d3,
    write_json,
)
from zero_field_refusal.paths import runs_root
from zero_field_refusal.tracking import git_state

CONFIG = "experiments/fa_control_diagnostic"
SCRIPT = 'project-run "$SRC" python scripts/fa_control_diagnostic.py'
NOT_DUE = 3  # ``branch --expansion-needed``: the expansion case is not due
COSTS = {  # GPU minutes and peak GB on one L40S (module docstring)
    "d1": (10, 38), "d2": (40, 40), "d3-determinism": (5, 31), "d3-case": (20, 43),
}  # fmt: skip
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
                 "(study2_short_fa_controls), as on P1."),
        *comment("Outputs: $RUNS_DIR/diagnostics/fa_control/ (d1, d2, d3, branch). It stops at "
                 "the first failure and can be run again: every part already written is kept."),
        "set -euo pipefail",
        ". /etc/rp_environment",
        ". /root/zfr-env.sh",
        f"SRC=/workspace/code/zero-field-refusal/{commit}/source",
        'cd "$SRC"',
        f'test "$(git rev-parse HEAD)" = {commit}',
        'test -z "$(git status --porcelain)"',
        'OUT="${RUNS_DIR:?}/diagnostics/fa_control"',
        'mkdir -p "$OUT"',
        f'step() {{ echo "== {stamp} start: $*"; "$@"; echo "== {stamp} done: $*"; }}',
        "# The steps, in the specification's order",
        f"step {SCRIPT} plan",
        f"step {SCRIPT} d1",
        f"step {SCRIPT} d2",
        f"step {SCRIPT} d3",
        "set +e",
        f"{SCRIPT} branch --expansion-needed",
        "due=$?",
        "set -e",
        'case "$due" in',
        f"  0) step {SCRIPT} d3 --expansion ;;",
        f'  {NOT_DUE}) echo "== {stamp} the expansion case is not due" ;;',
        '  *) exit "$due" ;;',
        "esac",
        f"step {SCRIPT} branch",
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
            lines.append(f"d2: {d.d2.batches} batches × (soft, nll) Lanczos "
                         f"{d.d2.lanczos.steps} steps + Ritz vectors; linearized runs at "
                         f"{list(d.d2.rates)}")  # fmt: skip
            cost = [COSTS["d2"]]
        else:
            names = [str(d.d3.expansion)] if part == "d3-expansion" else list(d.d3.order)
            listed = [(n, cases[n].lr, cases[n].control_lr) for n in names]
            steps = list(d.d3.eval_at)
            lines.append(f"{part}: determinism on {d.d3.determinism}; cases {listed}; pairs "
                         f"p1-p4 and the float64 reference at steps {steps}")  # fmt: skip
            cost = [COSTS["d3-determinism"]] + [COSTS["d3-case"]] * len(names)
        minutes += sum(c[0] for c in cost)
        gb = max(gb, *(c[1] for c in cost))
        lines.append(f"  cost ≈ {sum(c[0] for c in cost)} GPU min, peak ≈ "
                     f"{max(c[1] for c in cost)} GB")  # fmt: skip
    lines.append(f"total ≈ {minutes:.0f} GPU min, peak ≈ {gb:.0f} GB of 46")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["plan", "d1", "d2", "d3", "branch", "driver"])
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--runs-root", type=Path, default=None, help="RUNS_DIR")
    parser.add_argument("--out", type=Path, default=None, help="output directory (driver: file)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--expansion", action="store_true", help="d3: the expansion case only")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--expansion-needed", action="store_true")
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
    out = args.out or output_root(args.runs_root)
    if args.part == "branch":
        d1, determinism, cases = load_outputs(out)
        decision = decide_branch(settings, d1, determinism, cases)
        print(json.dumps({k: decision[k] for k in ("branch", "expand", "missing")}))
        if args.expansion_needed:
            sys.exit(0 if decision["expand"] else NOT_DUE)
        write_json(out / "branch" / "branch.json", decision)
        return

    root = args.runs_root or runs_root()
    runs = locate_runs(settings, root)
    parts = {"plan": ["d1", "d2", "d3"], "d1": ["d1"], "d2": ["d2"],
             "d3": ["d3-expansion" if args.expansion else "d3"]}[args.part]  # fmt: skip
    if args.part == "plan" or args.dry_run:
        print(plan(settings, runs, parts, out))
        return
    if args.part in ("d1", "d2") and (out / args.part / "result.json").exists():
        print(f"kept: {out / args.part / 'result.json'} exists")
        return
    d = settings.diagnostic
    names = [str(d.d3.expansion)] if args.expansion else [str(c) for c in d.d3.order]
    if args.part == "d3":
        record = out / "d3" / "determinism.json"
        if record.exists() and not json.loads(record.read_text())["equal"]:
            print(f"D3 stopped: {record} records a re-run that differs", file=sys.stderr)
            sys.exit(2)
        if record.exists() and all((out / "d3" / f"{n}.json").exists() for n in names):
            print(f"kept: {names} are written under {out / 'd3'}")
            return

    from zero_field_refusal.runtime import environment

    cfg = run_config(runs.regular[settings.diagnostic.d3.determinism])
    env = environment(cfg, args.device)
    if args.part == "d1":
        result = run_d1(settings, env, runs, out)
        print(json.dumps({"residual_norms": result["residual"]["norms"],
                          "path_dominates": result["residual"]["path_dominates"],
                          "same_path_control_max_K": result["same_path_control"]["max_K_scoring"],
                          "v0_equal": result["shared_state"]["v0"]["equal"]}))  # fmt: skip
    elif args.part == "d2":
        result = run_d2(settings, env, runs, out)
        print(json.dumps({kind: [round(r["top"], 6) for r in rows]
                          for kind, rows in result["spectra"].items()}))  # fmt: skip
    else:
        try:
            done = run_d3(settings, env, runs, out, cases=names,
                          keep_checkpoints=args.keep_checkpoints)  # fmt: skip
        except DeterminismFailure as exc:
            print(f"D3 stopped: {exc}", file=sys.stderr)
            sys.exit(2)
        print(json.dumps({name: (r["max_D_twin"] if math.isfinite(r["max_D_twin"]) else "inf")
                          for name, r in done.items()}))  # fmt: skip


if __name__ == "__main__":
    main()
