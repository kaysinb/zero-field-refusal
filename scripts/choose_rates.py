"""Choose the stage-1 sweep rates from the seed-0 pilots (zero_field_refusal.analysis.sweeps).

    python scripts/choose_rates.py RUNS_ROOT --model 1.5b

Reads K, divergence and utility of the runs in RUNS_ROOT/stage1_main_<model> (and, on 1.5B,
RUNS_ROOT/stage1_sgd100x5_1.5b); never a judge label. Prints, per sweep (SGD, frozen Adam on 1.5B,
SGD under 100 x 5 on 1.5B), the candidates, the target K values and either the chosen rates or the
extension rates to pilot first, as JSON.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from zero_field_refusal.analysis.sweeps import (
    adamw_range,
    choose,
    pilots_of,
    read_pilots,
    theta0_utility,
)

GRIDS = {"sgd": [0.03, 0.1, 0.3, 1.0, 3.0], "adam_frozen": [1e-5, 2e-5, 5e-5, 1e-4],
         "sgd_100x5": [0.01, 0.03, 0.1, 0.3, 1.0]}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--model", choices=["0.5b", "1.5b"], required=True)
    args = parser.parse_args(argv)
    main_rows = read_pilots(args.runs_root, f"stage1_main_{args.model}")
    theta0 = theta0_utility(main_rows)
    k_min, k_max = adamw_range(main_rows)
    out = {"model": args.model, "theta0": theta0, "adamw_k_range": [k_min, k_max], "sweeps": {}}
    sweeps = [("sgd", "sgd", main_rows, GRIDS["sgd"])]
    if args.model == "1.5b":
        sweeps.append(("adam_frozen", "adam_frozen", main_rows, GRIDS["adam_frozen"]))
        x5 = read_pilots(args.runs_root, "stage1_sgd100x5_1.5b")
        if x5:
            sweeps.append(("sgd_100x5", "sgd", x5, GRIDS["sgd_100x5"]))
    for name, arm, rows, grid in sweeps:
        choice = choose(arm, grid, pilots_of(rows, arm), theta0, k_min, k_max)
        out["sweeps"][name] = asdict(choice)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
