"""Summarize Stage A of study 2 and apply the preregistered selection rule; reads no harm label.

    python scripts/stage_a_summary.py --runs DIR [DIR ...] --theta0 RUN_DIR --out OUT.json

Each DIR holds scheduled runs (experiments under RUNS_DIR, or any parents of run directories;
overlapping DIRs count each run once). They are split by the role their config records
(``stage_a.role``: screen, full_control or discovery); a run without one stops the summary.
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
    BATCHES,
    DISCOVERY_SEEDS,
    ROLES,
    agreement,
    cell_summary,
    cells,
    instability_summary,
    load_trajectories,
    monitor_qualification,
    select,
)
from zero_field_refusal.tracking import git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--theta0", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists; a Stage-A summary is never overwritten")

    every = load_trajectories(*args.runs)
    by_role = {role: [run for run in every if run.role == role] for role in ROLES}
    runs, screen, controls = by_role["discovery"], by_role["screen"], by_role["full_control"]
    if len(every) != sum(map(len, by_role.values())):
        parser.error("runs without a Stage-A role (stage_a.role) among the inputs")
    summary = json.loads((args.theta0 / "summary.json").read_text())
    theta0 = {key: summary[key] for key in ("benchmark_accuracy", "factual_accuracy")}
    result = {
        "command": sys.argv,
        "code": git_state(),
        "theta0": {"run": args.theta0.name, **theta0},
        "cells": [cell_summary(cell, theta0, DISCOVERY_SEEDS) for cell in cells(runs).values()],
        "instability": instability_summary(runs),
        "monitor_agreement": agreement(runs),
        "monitor_qualification_by_batch": {  # the selection's is that of its batch
            b: monitor_qualification(
                [r for r in [*runs, *by_role["diagnostic"]] if r.batch_size == b]
            )
            for b in BATCHES
        },
        "selection": select(runs, screen, controls, theta0, by_role["diagnostic"]),
    }
    args.out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
