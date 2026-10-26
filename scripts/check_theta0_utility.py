"""Check that a new θ₀ evaluation reproduces the reference's utility (study 3, "Numerical
qualification": the starting model's utility check).

    python scripts/check_theta0_utility.py NEW_RUN REFERENCE_RUN

Each argument is a θ₀ evaluation's run directory (or its ``summary.json``). Exits 0 when
``benchmark_accuracy`` and ``factual_accuracy`` of the two summaries are exactly equal, printing
both; exits 1 otherwise, or when a summary or a value is missing: a technical mark, not a
scientific outcome.
"""

import argparse
import json
import sys
from pathlib import Path

MEASURES = ("benchmark_accuracy", "factual_accuracy")


def read_summary(path: Path) -> dict:
    file = path / "summary.json" if path.is_dir() else path
    return json.loads(file.read_text())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new_run", type=Path)
    parser.add_argument("reference_run", type=Path)
    args = parser.parse_args(argv)
    try:
        new, reference = read_summary(args.new_run), read_summary(args.reference_run)
    except (OSError, ValueError) as error:
        print(f"unreadable summary: {error}")
        return 1
    equal = True
    for name in MEASURES:
        values = (new.get(name), reference.get(name))
        print(f"{name}: new={values[0]!r} reference={values[1]!r}")
        equal &= all(isinstance(v, int | float) and not isinstance(v, bool) for v in values)
        equal &= values[0] == values[1]
    print("equal" if equal else "NOT equal")
    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
