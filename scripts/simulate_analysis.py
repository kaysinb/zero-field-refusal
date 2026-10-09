"""Full-analysis simulation of Gate 2: error rates and power of the whole sequential analysis.

Writes to --out (default: $RUNS_DIR/analysis/simulation/<UTC timestamp>/):
- results.csv: one row per scenario × CI method × CI level (the grid);
- resamples.csv: the resample-count comparison, one row per scenario × resamples × method × level;
- summary.json: settings, κ per σ, versions, acceptance verdicts and the recommended CI method,
  level and number of resamples.
"""

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from zero_field_refusal.analysis.simulate import (
    acceptance,
    full_settings,
    quick_settings,
    recommend,
    run_simulation,
)
from zero_field_refusal.paths import runs_root
from zero_field_refusal.tracking import git_state

PACKAGES = ("numpy", "scipy", "pandas", "statsmodels", "zero-field-refusal")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke run with a tiny grid")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=0, help="base seed of all randomness")
    args = parser.parse_args(argv)

    settings = quick_settings(args.seed) if args.quick else full_settings(args.seed)
    started = datetime.now(UTC)
    out = args.out or runs_root() / "analysis" / "simulation" / started.strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done == total or done % max(1, total // 50) == 0:
            elapsed = time.perf_counter() - clock
            print(f"{done}/{total} chunks, {elapsed / 60:.1f} min", file=sys.stderr, flush=True)

    result = run_simulation(settings, workers=args.workers, progress=progress)
    verdicts = acceptance(result.results)
    recommendation = recommend(verdicts, result.comparison, settings)
    result.results.to_csv(out / "results.csv", index=False)
    result.comparison.to_csv(out / "resamples.csv", index=False)
    summary = {
        "started": started.isoformat(timespec="seconds"),
        "runtime_s": time.perf_counter() - clock,
        "mode": "quick" if args.quick else "full",
        "workers": args.workers,
        "settings": asdict(settings) | {"scenario_names": [s.name for s in settings.scenarios]},
        "kappa": {f"{sigma:.2f}": value for sigma, value in result.kappa.items()},
        "versions": {
            "python": platform.python_version(),
            **{name: metadata.version(name) for name in PACKAGES},
            "git": git_state(),
        },
        "acceptance": verdicts,
        "recommendation": recommendation,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out}")
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
