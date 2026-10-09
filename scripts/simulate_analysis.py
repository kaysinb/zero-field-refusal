"""Full-analysis simulation of Gate 2: error rates and power of the whole sequential analysis.

Writes to --out (default: $RUNS_DIR/analysis/simulation/<UTC timestamp>/):
- results.csv: the grid, one row per scenario × CI method × CI level;
- resamples.csv, exclusion.csv and h2_boundary.csv: the resample-count comparison, the sensitivity
  to the small-K exclusion factor and the descriptive H2 boundary scenarios, one row per
  scenario × variant × method × level;
- summary.json: settings, κ per σ and base curve, versions, and either the acceptance verdicts,
  the exclusion-factor worst cases, the recommended CI method, level and number of resamples and
  the H2 labels of the boundary scenarios, or (with --confirm) the worst-case null rates.

--confirm reruns only the binding σ = 0.22 null scenarios; give it a fresh --seed for an
out-of-sample check of the chosen level. --resummarize DIR recomputes the verdicts in DIR's
summary.json from its tables with the current code, after checking that the run's settings are
the code's, and records the code version under "resummarized".
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

import pandas as pd

from zero_field_refusal.analysis.simulate import (
    PLANNING_LEVEL,
    SimulationSettings,
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
SETTINGS = {
    "full": full_settings,
    "quick": quick_settings,
    "confirm": confirm_settings,
    "quick-confirm": quick_confirm_settings,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smoke run with a tiny grid")
    parser.add_argument("--confirm", action="store_true", help="rerun the σ = 0.22 nulls only")
    parser.add_argument("--resummarize", type=Path, help="recompute the verdicts of a saved run")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=0, help="base seed of all randomness")
    args = parser.parse_args(argv)

    # Recorded at launch: the source of a long run is the one it started with.
    versions = {
        "python": platform.python_version(),
        **{name: metadata.version(name) for name in PACKAGES},
        "git": git_state(),
    }
    if args.resummarize:
        resummarize(args.resummarize, versions)
        return

    mode = ("quick-" if args.quick else "") + ("confirm" if args.confirm else "full")
    mode = "quick" if mode == "quick-full" else mode
    settings = SETTINGS[mode](args.seed)
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
    summary = {
        "started": started.isoformat(timespec="seconds"),
        "runtime_s": time.perf_counter() - clock,
        "mode": mode,
        "workers": args.workers,
        "settings": _settings_record(settings),
        "kappa": result.kappa,
        "versions": versions,
        **verdicts(mode, settings, result.tables),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out}")
    if "recommendation" in summary:
        print(json.dumps(summary["recommendation"], indent=2))


def verdicts(mode: str, settings: SimulationSettings, tables: dict[str, pd.DataFrame]) -> dict:
    """The summary sections computed from the result tables."""
    grid = tables[settings.grid.name]
    if mode.endswith("confirm"):
        return {"null_worst_cases": null_worst_cases(grid)}
    accepted = acceptance(grid)
    recommendation = recommend(accepted, tables[settings.resample_comparison.name], settings)
    # Descriptive: H2 labels at the recommended method and level (the planning level of the
    # percentile CI if nothing passes).
    method = recommendation["method"] or "percentile"
    level = recommendation["level"] or PLANNING_LEVEL
    return {
        "acceptance": accepted,
        "recommendation": recommendation,
        "exclusion": null_worst_cases(
            tables[settings.exclusion_sensitivity.name],
            by=("exclusion_factor", "method", "level"),
        ),
        "h2_boundary": {
            "method": method,
            "level": level,
            "rows": h2_label_frequencies(tables[settings.h2_boundary.name], method, level),
        },
    }


def resummarize(directory: Path, versions: dict) -> None:
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    settings = SETTINGS[summary["mode"]](summary["settings"]["base_seed"])
    if _settings_record(settings) != summary["settings"]:
        raise SystemExit(f"The settings recorded in {path} are not the code's settings")
    tables = {part.name: pd.read_csv(directory / f"{part.name}.csv") for part in settings.parts}
    summary |= verdicts(summary["mode"], settings, tables)
    summary["resummarized"] = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "versions": versions,
    }
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Rewrote {path}")


def _settings_record(settings: SimulationSettings) -> dict:
    """The settings as they appear in summary.json (after a JSON round trip)."""
    record = asdict(settings) | {
        "scenario_names": {p.name: [s.name for s in p.scenarios] for p in settings.parts}
    }
    return json.loads(json.dumps(record))


if __name__ == "__main__":
    main()
