"""Write the campaigns of study 2's short-regime stage (amendment 3, design A) from the run records,
and the shell driver of their prerequisites.

    python scripts/study2_short_campaigns.py prep --seeds S [S ...] --batch B --commit SHA \
        [--streams-only] [--stage0-run-id ID] --out FILE
    python scripts/study2_short_campaigns.py KIND --runs DIR... --theta0 RUN \
        (--w-max W | --fixation FILE) [--seeds S ...] [--cells NAME ...] --out FILE
    python scripts/study2_short_campaigns.py confirmation --blocks FIRST LAST --runs DIR... \
        --theta0 RUN (--w-max W | --fixation FILE) [--cells NAME ...] --out FILE

KIND is the campaign the stage's state machine names (analysis.study2_short.progress), and each
kind refuses when it names another step, or when the records are not whole: ``targeted`` or
``repair`` (a round of discovery runs at the rates it prescribes, on seeds 2000-2002, of the
primary arms or of the pre-warmed arm), ``controls`` (one FA soft-target control per selected FA
rate on seed 2000, experiments/study2_short_fa_controls with schedule.instability.contamination_k
= 0.01 × K*), ``prewarmed`` (the pre-warmed arm's first discovery runs, at the union of the frozen
AdamW and FA rates), ``validation`` (seeds 2100-2111, every arm at its frozen rates) and
``confirmation`` (the blocks FIRST to LAST, 1-80, in the preassigned seed order 3000-3079, every
arm at its frozen rates, once the validation passed; blocks 1-10 run
experiments/study2_short_rayleigh, which measures the endpoint Rayleigh quotient). A campaign
holds the runs still missing; ``--seeds`` and ``--cells`` only split it over Pods. --runs holds
every run so far: Stage A's (study2_stage_a, study2_controls), whose 25-step grids are the first
discovery runs and whose version-1 discovery set fixes the stream time, and this stage's
(study2_short, study2_short_fa_controls). --theta0 is the θ₀ run (summary.json), the utility
baseline. w_max comes from --w-max or from the simulation's fixation record (--fixation,
``read_fixation``). Nothing here reads a judge label or K_harmful; the looks
(scripts/study2_short_summary.py --judge) tell when no further confirmation block is started.

``prep`` writes a bash driver of the prerequisites of the runs of the given seeds at batch B, from
the source checkout of commit SHA on a Pod, as the Stage-A drivers: the stage-0 checks under
protocol version 3 and the curvature panel (once; ``--streams-only`` leaves them out), then per
seed the training stream (25 × B responses: the stream is appendable, and its default length,
that of a 4000-step run, is not needed) and v₀'s preconditioner stream at B. It stops at the
first failure; a stream that exists already is kept, since generate_teacher.py never regenerates
one. FILE must not exist; write campaigns under an ignored directory such as $RUNS_DIR/campaigns/.
"""

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from zero_field_refusal.analysis.stage_a import BATCHES, load_trajectories, rate, theta0_problems
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    ARMS,
    CAP,
    CONFIRMATION_SEEDS,
    DISCOVERY_SEEDS,
    FA,
    FA_CONTROL_ARM,
    FA_CONTROLS,
    PREWARMED,
    RAYLEIGH,
    REGULAR,
    STEPS,
    VALIDATION_SEEDS,
    measures_rayleigh,
    progress,
    read_fixation,
    run_facts,
    seed_role,
)

KINDS = ("targeted", "repair", "controls", "prewarmed", "validation", "confirmation")
OPTIMIZER = {ADAMW: [], FA: ["optimizer=adam_frozen"], PREWARMED: ["optimizer=adamw_prewarmed"]}
STAGE_SEEDS = (*DISCOVERY_SEEDS, *VALIDATION_SEEDS, *CONFIRMATION_SEEDS)
TEACHER = 'project-run "$SRC" python scripts/generate_teacher.py'
STAGE0 = 'project-run "$SRC" python scripts/stage0_checks.py experiments/stage0_checks'
# generate_teacher.py's refusal of a stream that exists (runtime.generate_stream).
EXISTS = "streams are the record and are never regenerated"


def cell(name: str, config: str, overrides: list[str], seeds: list[int]) -> dict:
    return {"name": name, "config": config, "overrides": overrides, "seeds": seeds}


def regular_cells(runs: list[dict], suffix: str = "") -> list[dict]:
    """One cell per arm, rate and config (the first 10 confirmation blocks measure R), over its
    seeds, with the role of its seeds. The launcher grants a recovery round once per experiment
    and cell name, so a name never recurs in another of the stage's campaigns: ``suffix`` names
    the validation or the confirmation blocks."""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for r in runs:
        seed = r["seed"]
        key = (ARMS.index(r["arm"]), r["batch"], r["lr"], measures_rayleigh(seed), seed_role(seed))
        groups[key].append(seed)
    items = []
    for (arm_index, batch, lr, rayleigh, role), seeds in sorted(groups.items()):
        arm = ARMS[arm_index]
        overrides = [*OPTIMIZER[arm], f"train.batch_size={batch}", f"optimizer.lr={lr:g}",
                     f"study2_short.role={role}"]  # fmt: skip
        name = f"{arm}-b{batch}-{lr:g}{suffix}" + ("-rayleigh" if rayleigh else "")
        items.append(cell(name, RAYLEIGH if rayleigh else REGULAR, overrides, sorted(seeds)))
    return items


def control_cells(runs: list[dict], upper: float) -> list[dict]:
    """One FA soft-target control per rate on seed 2000, its bound 0.01 × K* written in full;
    a rate may have a control under each U, so the cell's name carries U."""
    items = []
    for r in sorted(runs, key=lambda r: r["lr"]):
        overrides = [f"train.batch_size={r['batch']}", f"optimizer.lr={r['lr']:g}",
                     f"schedule.instability.contamination_k={r['contamination_k']!r}"]  # fmt: skip
        items.append(cell(f"fa-control-b{r['batch']}-{r['lr']:g}-u{upper:g}", FA_CONTROLS,
                          overrides, [r["seed"]]))  # fmt: skip
    return items


def prep_driver(seeds: list[int], batch: int, commit: str, stage0_run_id: str | None) -> str:
    """The bash driver of the runs' prerequisites (see the module docstring)."""
    what = "the training and preconditioner streams"
    if stage0_run_id:
        what = f"the stage-0 checks under protocol version 3, the curvature panel, {what}"
    stamp = "$(date -u +%FT%TZ)"
    lines = [
        "#!/usr/bin/env bash",
        "# Written by scripts/study2_short_campaigns.py prep; do not edit.",
        f"# Study 2's short-regime stage, from commit {commit}: {what}",
        f"# of seeds {' '.join(map(str, seeds))} at batch {batch}. Stops at the first failure; a",
        "# stream that exists already is kept (generate_teacher.py never regenerates one).",
        "set -euo pipefail",
        ". /etc/rp_environment",
        ". /root/zfr-env.sh",
        f"SRC=/workspace/code/zero-field-refusal/{commit}/source",
        'cd "$SRC"',
        f'test "$(git rev-parse HEAD)" = {commit}',
        'test -z "$(git status --porcelain)"',
        f'step() {{ echo "== {stamp} start: $*"; "$@"; echo "== {stamp} done: $*"; }}',
        "kept() {",
        '  local log',
        '  log=$(mktemp)',
        f'  echo "== {stamp} start: $*"',
        f'  if "$@" 2>&1 | tee "$log"; then echo "== {stamp} done: $*"',
        f'  elif grep -q "{EXISTS}" "$log"; then echo "== {stamp} kept, it exists: $*"',
        '  else rm -f "$log"; return 1; fi',
        '  rm -f "$log"',
        "}",
    ]  # fmt: skip
    if stage0_run_id:
        lines += [
            f"step env RUN_ID={stage0_run_id} {STAGE0} \\",
            "  model=qwen2.5-1.5b-instruct protocol_version=3",
            f"kept {TEACHER} {RAYLEIGH} --stream curvature_panel",
        ]
    for seed in seeds:
        lines += [
            f"step {TEACHER} experiments/study2_stage_a --stream training \\",
            f"  --seeds {seed} --responses {STEPS * batch}",
            f"kept {TEACHER} experiments/study2_stage_a \\",
            f"  optimizer=adam_frozen train.batch_size={batch} --stream preconditioner \\",
            f"  --seeds {seed}",
        ]
    lines.append(f'echo "== {stamp} the prerequisites are complete"')
    return "\n".join(lines) + "\n"


def prep(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    if not args.seeds or args.batch is None or not args.commit:
        parser.error("prep needs --seeds, --batch and --commit")
    if len(args.commit) != 40 or set(args.commit) - set("0123456789abcdef"):
        parser.error("--commit takes the full 40-character commit hash")
    if set(args.seeds) - set(STAGE_SEEDS) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds: distinct seeds of the stage (2000-2002, 2100-2111, 3000-3079)")
    run_id = None
    if not args.streams_only:
        today = datetime.now(UTC).strftime("%Y%m%d")
        run_id = args.stage0_run_id or f"stage0-v3-qwen1.5b-l40s-{today}-01"
    return prep_driver(args.seeds, args.batch, args.commit, run_id)


def w_max_of(parser: argparse.ArgumentParser, args: argparse.Namespace) -> float:
    w_max = args.w_max
    if args.fixation is not None:
        try:
            fixed = read_fixation(args.fixation)["w_max"]
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if w_max is not None and w_max != fixed:
            parser.error(f"--w-max {w_max} differs from the fixation record's {fixed}")
        w_max = fixed
    if w_max is None or not w_max > 0:
        parser.error("w_max is needed: --w-max or --fixation")
    return w_max


def campaign(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict:
    if args.theta0 is None or not args.runs:
        parser.error(f"{args.kind} needs --runs and --theta0")
    if (args.kind == "confirmation") != (args.blocks is not None):
        parser.error("--blocks goes with confirmation, and confirmation needs it")
    if args.kind in ("confirmation", "controls") and args.seeds:
        parser.error("--seeds splits discovery and validation campaigns only")
    w_max = w_max_of(parser, args)
    summary = json.loads((args.theta0 / "summary.json").read_text())
    theta0 = {key: summary.get(key) for key in ("benchmark_accuracy", "factual_accuracy")}
    if theta0_problems(theta0):
        parser.error("; ".join(theta0_problems(theta0)))
    runs = load_trajectories(*args.runs)
    facts = {r.run_id: run_facts(r.run_dir) for r in runs if r.arm in (*ARMS, FA_CONTROL_ARM)}
    state = progress(runs, theta0, w_max, facts=facts)
    step = state["next"]
    if state["problems"]:
        parser.error("the stage's records are not whole: " + "; ".join(state["problems"]))
    if step["campaign"] != args.kind:
        shown = {k: v for k, v in step.items() if k != "runs"}
        parser.error(f"the stage's next step is {step['campaign']}, not {args.kind}: {shown}")
    batch, upper = step["batch"], step["U"]
    if args.kind == "controls":
        return {"name": f"study2-short-controls-b{batch}-u{upper:g}",
                "cells": control_cells(step["runs"], upper)}  # fmt: skip
    if args.kind == "confirmation":
        first, last = args.blocks
        if not 1 <= first <= last <= CAP:
            parser.error(f"--blocks: 1 <= FIRST <= LAST <= {CAP}")
        seeds = CONFIRMATION_SEEDS[first - 1 : last]
        present = {(r.arm, r.batch_size, rate(r.lr), r.seed) for r in runs
                   if r.arm in ARMS and r.seed in seeds}  # fmt: skip
        wanted = [{"arm": arm, "batch": batch, "lr": rate(lr), "seed": seed}
                  for seed in seeds for arm, lrs in step["rates"].items() for lr in lrs
                  if (arm, batch, rate(lr), seed) not in present]  # fmt: skip
        if not wanted:
            parser.error(f"blocks {first}-{last} have every run already")
        return {"name": f"study2-short-confirmation-{first}-{last}",
                "cells": regular_cells(wanted, f"-blocks-{first}-{last}")}  # fmt: skip
    wanted = step["runs"]
    if args.seeds:
        unknown = sorted(set(args.seeds) - {r["seed"] for r in wanted})
        if unknown:
            parser.error(f"no run of seeds {unknown} in this campaign")
        wanted = [r for r in wanted if r["seed"] in args.seeds]
    # The pre-warmed arm's own rounds, apart from the primary arms' of the same batch and U.
    own = args.kind in ("targeted", "repair") and {r["arm"] for r in wanted} == {PREWARMED}
    kind = f"{args.kind}-prewarmed" if own else args.kind
    suffix = "-validation" if args.kind == "validation" else ""
    return {"name": f"study2-short-{kind}-b{batch}-u{upper:g}",
            "cells": regular_cells(wanted, suffix)}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("prep", *KINDS))
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--theta0", type=Path, help="the θ₀ run directory (summary.json)")
    parser.add_argument("--w-max", type=float, help="the realized-width bound w_max")
    parser.add_argument("--fixation", type=Path, help="the simulation's fixation record (JSON)")
    parser.add_argument("--seeds", type=int, nargs="+", help="keep only these seeds (one Pod)")
    parser.add_argument("--cells", nargs="+", help="keep only these cells (one Pod each)")
    parser.add_argument("--blocks", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="confirmation: the blocks FIRST to LAST, 1-80")  # fmt: skip
    parser.add_argument("--batch", type=int, choices=BATCHES, help="prep: the batch size")
    parser.add_argument("--commit", help="prep: the commit whose source checkout the Pod runs")
    parser.add_argument(
        "--streams-only",
        action="store_true",
        help="prep: leave out the stage-0 checks and the curvature panel",
    )
    parser.add_argument("--stage0-run-id", help="prep: RUN_ID of the stage-0 checks")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists")
    if args.kind == "prep":
        text = prep(parser, args)
    else:
        written = campaign(parser, args)
        if args.cells:  # a campaign split over Pods, cell by cell, without choosing anything
            unknown = sorted(set(args.cells) - {c["name"] for c in written["cells"]})
            if unknown:
                parser.error(f"no such cells in this campaign: {unknown}")
            written["cells"] = [c for c in written["cells"] if c["name"] in args.cells]
        header = f"# Written by scripts/study2_short_campaigns.py {args.kind}; do not edit.\n"
        text = header + yaml.safe_dump(written, sort_keys=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(args.out)


if __name__ == "__main__":
    main()
