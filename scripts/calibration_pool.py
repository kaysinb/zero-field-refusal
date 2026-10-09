"""Gather calibration responses from run directories into one pool for the blind sample.

    python scripts/calibration_pool.py RUN_DIR [RUN_DIR ...] --out POOL.jsonl

Then: make_calibration_sample.py POOL.jsonl SAMPLE.jsonl KEY.json --strata model,state --seed N.
The pool holds model responses to harmful prompts: keep it out of git.
"""

import argparse
import json
from pathlib import Path

from zero_field_refusal.calibration import CALIBRATION_SET, calibration_pool
from zero_field_refusal.data import load_prompt_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pool = calibration_pool(args.run_dirs, lambda cfg: load_prompt_set(CALIBRATION_SET, cfg.data))
    with args.out.open("w") as f:
        for item in pool:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    print(f"{len(pool)} items -> {args.out}")


if __name__ == "__main__":
    main()
