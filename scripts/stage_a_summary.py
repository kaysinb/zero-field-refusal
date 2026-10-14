"""Summarize Stage A of study 2 and apply the preregistered selection rule; reads no harm label.

    python scripts/stage_a_summary.py --discovery DIR [DIR ...] --theta0 RUN_DIR
        [--screen DIR ...] [--controls DIR ...] [--seeds 2000 2001 2002] --out OUT.json

DIR is a directory of scheduled runs (an experiment under RUNS_DIR, or any parent of run
directories). --screen holds the 100-step soft-target controls of the SGD rate grid, --controls
the full-length controls of the candidate rates; --theta0 is the θ₀ evaluation whose utility the
losses are measured against. OUT.json (which must not exist) records the command and the code
revision, the screen and its candidates, the numerical acceptance, one summary per arm, batch and
rate, the gradient rule's firings and false alarms, the monitor's agreement, the next rates of the
25-step grids at batch 5, and the selection (docs/study2-preregistration.md, "Selection rule").
"""

import argparse
import json
import sys
from pathlib import Path

from zero_field_refusal.analysis.stage_a import (
    agreement,
    cell_summary,
    cells,
    instability_summary,
    load_trajectories,
    numerical_acceptance,
    screen_candidates,
    select,
    short_grid_extension,
)
from zero_field_refusal.tracking import git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--discovery", type=Path, nargs="+", required=True)
    parser.add_argument("--screen", type=Path, nargs="*", default=[])
    parser.add_argument("--controls", type=Path, nargs="*", default=[])
    parser.add_argument("--theta0", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001, 2002])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists; a Stage-A summary is never overwritten")

    def load(dirs):
        return [run for d in dirs for run in load_trajectories(d)]

    runs, screen, controls = load(args.discovery), load(args.screen), load(args.controls)
    summary = json.loads((args.theta0 / "summary.json").read_text())
    theta0 = {key: summary[key] for key in ("benchmark_accuracy", "factual_accuracy")}
    acceptance = numerical_acceptance(controls, runs)
    result = {
        "command": sys.argv,
        "code": git_state(),
        "theta0": {"run": args.theta0.name, **theta0},
        "screen": screen_candidates(screen),
        "acceptance": {f"B={b} lr={lr:g}": a for (b, lr), a in acceptance.items()},
        "cells": [cell_summary(cell, theta0, args.seeds) for cell in cells(runs).values()],
        "instability": instability_summary(runs),
        "monitor_agreement": agreement(runs),
        "short_grid_extension": {
            arm: short_grid_extension(runs, arm) for arm in ("adamw_25", "frozen_adam_25")
        },
        "selection": select(runs, theta0, acceptance, args.seeds),
    }
    args.out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
