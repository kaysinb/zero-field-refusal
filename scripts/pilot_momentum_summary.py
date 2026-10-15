"""The SGD-momentum pilot's outcome and the check of its recorded prediction; reads no harm label
(docs/study2-pilot-momentum.md).

    python scripts/pilot_momentum_summary.py --runs DIR [DIR ...] --stage-a DIR [DIR ...] \
        --theta0 RUN_DIR --g-table FILE.json --out OUT.json

--runs holds the pilot's runs (study2_pilot_momentum, study2_pilot_momentum_controls), --stage-a
Stage A's (study2_stage_a, study2_controls), --g-table the table frozen before the pilot's first
run. OUT.json, which must not exist, records the command and the code revision; the outcome
(analysis.pilot_momentum.outcome); and, for every regular run of the pilot, the median ratio of its
scoring K to the law at its effective rate (analysis.zero_field_law.check), with the median over
the runs that have one. The prediction is contradicted when that median lies outside 0.7–1.4.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

from zero_field_refusal.analysis.pilot_momentum import ROLES, outcome
from zero_field_refusal.analysis.stage_a import load_trajectories
from zero_field_refusal.analysis.zero_field_law import check
from zero_field_refusal.tracking import git_state

BAND = (0.7, 1.4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--stage-a", type=Path, nargs="+", required=True)
    parser.add_argument("--theta0", type=Path, required=True)
    parser.add_argument("--g-table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists")
    summary = json.loads((args.theta0 / "summary.json").read_text())
    theta0 = {key: summary[key] for key in ("benchmark_accuracy", "factual_accuracy")}
    runs = load_trajectories(*args.runs)
    by_role = {role: [r for r in runs if r.role == role] for role in ROLES}
    result = outcome(by_role["screen"], by_role["discovery"], by_role["full_control"], theta0,
                     load_trajectories(*args.stage_a))  # fmt: skip
    table = json.loads(args.g_table.read_text())
    checks = [check(run, table) for run in sorted(by_role["discovery"], key=lambda r: r.run_id)]
    ratios = [c["median_ratio"] for c in checks if c["median_ratio"] is not None]
    middle = statistics.median(ratios) if ratios else None
    prediction = {
        "g_table": str(args.g_table),
        "runs": checks,
        "median_ratio": middle,
        "band": BAND,
        "contradicted": None if middle is None else not BAND[0] <= middle <= BAND[1],
    }
    out = {"command": sys.argv, "code": git_state(), "outcome": result, "prediction": prediction}
    args.out.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
