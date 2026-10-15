"""Freeze the g table of the SGD-momentum pilot's recorded prediction; reads K only
(docs/study2-pilot-momentum.md, "Recorded prediction").

    python scripts/pilot_momentum_prediction.py --stage-a DIR [DIR ...] --theta0 RUN_DIR \
        --out FILE.json

The source runs come from Stage A's resolved search alone (analysis.zero_field_law.freeze):
every SGD-long run it tested on every discovery seed, from Stage A's records checked as its
selection checks them (--stage-a: study2_stage_a and study2_controls). FILE records the table,
the runs' IDs with the SHA-256 of the files it reads, the table's hash, the command and the
code revision. Run it once Stage A's search has resolved; commit FILE, and its table hash in the
pilot's document, before the pilot's first run. FILE must not exist.
"""

import argparse
import json
import sys
from pathlib import Path

from zero_field_refusal.analysis.stage_a import load_trajectories
from zero_field_refusal.analysis.zero_field_law import freeze
from zero_field_refusal.tracking import git_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage-a", type=Path, nargs="+", required=True)
    parser.add_argument("--theta0", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists")
    summary = json.loads((args.theta0 / "summary.json").read_text())
    theta0 = {key: summary[key] for key in ("benchmark_accuracy", "factual_accuracy")}
    try:
        frozen = freeze(load_trajectories(*args.stage_a), theta0)
    except ValueError as exc:
        parser.error(str(exc))
    out = {"command": sys.argv, "code": git_state(), **frozen}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(args.out, frozen["table_sha256"])


if __name__ == "__main__":
    main()
