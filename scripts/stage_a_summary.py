"""Summarize Stage A of study 2 and apply the preregistered selection rule; reads no harm label.

    python scripts/stage_a_summary.py --runs DIR [DIR ...] --theta0 RUN_DIR
        [--seeds 2000 2001 2002] --out OUT.json

Each DIR holds scheduled runs (experiments under RUNS_DIR, or any parents of run directories;
overlapping DIRs count each run once). They are split by what they are: soft-target controls of
100 steps form the screen, longer ones the full-length controls, every other run discovery.
--theta0 is the θ₀ evaluation whose utility the losses are measured against.

OUT.json, which must not exist, records the command and the code revision; one summary per arm,
batch and rate; the gradient rule's margin and possible false alarms; the monitor's agreement;
and the selection with the screen, its candidates and the numerical acceptance
(docs/study2-preregistration.md, "Selection rule"). The selection is ``incomplete`` unless the
whole prescribed search is there.
"""

import argparse
import json
import sys
from pathlib import Path

from zero_field_refusal.analysis.stage_a import (
    SCREEN_STEPS,
    agreement,
    cell_summary,
    cells,
    instability_summary,
    load_trajectories,
    select,
)
from zero_field_refusal.tracking import git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--theta0", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001, 2002])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists; a Stage-A summary is never overwritten")

    every = load_trajectories(*args.runs)
    soft = [run for run in every if run.arm == "soft_control"]
    screen = [run for run in soft if run.planned_steps == SCREEN_STEPS]
    controls = [run for run in soft if run.planned_steps != SCREEN_STEPS]
    runs = [run for run in every if run.arm != "soft_control"]
    summary = json.loads((args.theta0 / "summary.json").read_text())
    theta0 = {key: summary[key] for key in ("benchmark_accuracy", "factual_accuracy")}
    result = {
        "command": sys.argv,
        "code": git_state(),
        "theta0": {"run": args.theta0.name, **theta0},
        "cells": [cell_summary(cell, theta0, args.seeds) for cell in cells(runs).values()],
        "instability": instability_summary(runs),
        "monitor_agreement": agreement(runs),
        "selection": select(runs, screen, controls, theta0, args.seeds),
    }
    args.out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
