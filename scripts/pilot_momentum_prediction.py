"""Freeze the g table of the SGD-momentum pilot's recorded prediction from Stage A's version-2
SGD-long runs; reads K only (docs/study2-pilot-momentum.md, "Recorded prediction").

    python scripts/pilot_momentum_prediction.py --runs DIR [DIR ...] --out FILE.json

Writes the table of analysis.zero_field_law.g_table (bins of ηt with g and the number of runs, and
the runs it used) with the command and the code revision. Run it once Stage A's search has
resolved and commit FILE before the pilot's first run; FILE must not exist.
"""

import argparse
import json
import sys
from pathlib import Path

from zero_field_refusal.analysis.stage_a import load_trajectories
from zero_field_refusal.analysis.zero_field_law import g_table
from zero_field_refusal.tracking import git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists")
    runs = [r for r in load_trajectories(*args.runs) if r.role == "discovery"]
    table = g_table(runs)
    if not table["bins"]:
        parser.error("no version-2 SGD-long runs to estimate g from")
    out = {"command": sys.argv, "code": git_state(), **table}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
