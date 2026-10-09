"""Full-analysis simulation of Gate 2: error rates and power of the whole sequential analysis.

Writes to --out (default: $RUNS_DIR/analysis/simulation/<UTC timestamp>/):
- results.csv: the grid, one row per scenario × CI method × CI level;
- resamples.csv and exclusion.csv: the resample-count comparison and the sensitivity to the
  small-K exclusion factor, one row per scenario × variant × method × level;
- summary.json: settings, κ per σ and base curve, versions, and either the acceptance verdicts,
  the exclusion-factor worst cases and the recommended CI method, level and number of resamples,
  or (with --confirm) the worst-case null rates of the confirmation run.

--confirm reruns only the binding σ = 0.22 null scenarios; give it a fresh --seed for an
out-of-sample check of the chosen level.
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
    PLANNING_LEVEL,
    acceptance,
    confirm_settings,
    full_settings,
    h2_label_frequencies,
    null_worst_cases,
    quick_confirm_settings,
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
    parser.add_argument("--confirm", action="store_true", help="rerun the σ = 0.22 nulls only")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=0, help="base seed of all randomness")
    args = parser.parse_args(argv)

    if args.confirm:
        mode = "confirm"
        settings = (quick_confirm_settings if args.quick else confirm_settings)(args.seed)
    else:
        mode = "quick" if args.quick else "full"
        settings = (quick_settings if args.quick else full_settings)(args.seed)
    # Recorded at launch: the source of a long run is the one it started with.
    versions = {
        "python": platform.python_version(),
        **{name: metadata.version(name) for name in PACKAGES},
        "git": git_state(),
    }
    started = datetime.now(UTC)
    out = args.out or runs_root() / "analysis" / "simulation" / started.strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()

    def progress(done: int, total: int) -> None:
        if done == total or done % max(1, total // 50) == 0:
            elapsed = time.perf_counter() - clock
            print(f"{done}/{total} chunks, {elapsed / 60:.1f} min", file=sys.stderr, flush=True)

    result = run_simulation(settings, workers=args.workers, progress=progress)
    for name, table in result.tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    grid = result.tables[settings.grid.name]
    summary = {
        "started": started.isoformat(timespec="seconds"),
        "runtime_s": time.perf_counter() - clock,
        "mode": mode,
        "workers": args.workers,
        "settings": asdict(settings)
        | {"scenario_names": {p.name: [s.name for s in p.scenarios] for p in settings.parts}},
        "kappa": result.kappa,
        "versions": versions,
    }
    if args.confirm:
        summary["null_worst_cases"] = null_worst_cases(grid)
    else:
        verdicts = acceptance(grid)
        comparison = result.tables[settings.resample_comparison.name]
        summary["acceptance"] = verdicts
        summary["recommendation"] = recommend(verdicts, comparison, settings)
        summary["exclusion"] = null_worst_cases(
            result.tables[settings.exclusion_sensitivity.name],
            by=("exclusion_factor", "method", "level"),
        )
        # Descriptive: H2 labels at the recommended method and level (the planning level of the
        # percentile CI if nothing passes).
        method = summary["recommendation"]["method"] or "percentile"
        level = summary["recommendation"]["level"] or PLANNING_LEVEL
        summary["h2_boundary"] = {
            "method": method,
            "level": level,
            "rows": h2_label_frequencies(result.tables[settings.h2_boundary.name], method, level),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out}")
    if "recommendation" in summary:
        print(json.dumps(summary["recommendation"], indent=2))


if __name__ == "__main__":
    main()
