"""Flatten every run under RUNS_DIR into one CSV table.

python scripts/index_runs.py [--root RUNS_DIR] [--out FILE]
"""

import argparse
from pathlib import Path

from zero_field_refusal.paths import runs_root
from zero_field_refusal.runindex import index_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root or runs_root()
    table = index_runs(root)
    out = args.out or root / "index.csv"
    table.to_csv(out, index=False)
    print(f"{len(table)} runs -> {out}")


if __name__ == "__main__":
    main()
