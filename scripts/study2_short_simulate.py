"""Simulation of study 2's short-regime confirmation (amendment 3): levels, validation and bias.

Modes (the model and the procedure: src/zero_field_refusal/analysis/study2_short_sim.py):

- calibrate: the central model at every target SD × Δ, on the ``calibration`` stream; evaluates
  the planning levels and the O'Brien–Fleming pair of every α on the grid, and selects, per SE
  floor, the largest α meeting the error constraints over the binding SDs (3, 5, 10 pp).
- validate: the validation grid on the independent ``validation`` stream, at the planning levels
  and at the α's given by --levels-from (a calibrate run's summary.json) or --alphas.
- bias: paired interpolation bias over curve family × selection miss × w_max (``bias`` stream),
  and the bias allowance per w_max.
- smoke: one reduced scenario per shape and per revision-1 addition, at the planning levels.

Writes to --out (default: $RUNS_DIR/analysis/study2_short_sim/<mode>-<UTC timestamp>/):
- summary.json: settings, versions, run time, the spread calibrations, every scenario (its
  settings, parameters, block diagnostics, compute time and one result row per level set × SE
  floor) and the mode's verdicts (selection, checks, bias allowance);
- results.md: the same as markdown tables.

summary.json is standard JSON: non-finite numbers are written as null (a w_max of null: no width
rule). --list prints the selected scenarios' names and exits; --scenarios takes fnmatch patterns
on those names; --reps overrides every scenario's replicate count.
"""

import argparse
import fnmatch
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from zero_field_refusal.analysis import study2_short_sim as sim
from zero_field_refusal.paths import runs_root
from zero_field_refusal.tracking import git_state

PACKAGES = ("numpy", "scipy", "zero-field-refusal")
MAX_WORKERS = 8
GRIDS = {
    "calibrate": sim.calibration_grid,
    "validate": sim.validation_grid,
    "bias": sim.bias_grid,
    "smoke": sim.smoke_grid,
}
STREAMS = {"calibrate": "calibration", "validate": "validation", "bias": "bias", "smoke": "smoke"}
DESIGNS = ("interp", "interp_grid")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=tuple(GRIDS))
    parser.add_argument("--designs", nargs="+", choices=DESIGNS, help="keep these designs only")
    parser.add_argument("--scenarios", nargs="+", metavar="PATTERN", help="fnmatch on names")
    parser.add_argument("--reps", type=int, help="replicates per scenario (overrides the grid)")
    parser.add_argument("--levels-from", type=Path, help="validate: a calibrate summary.json")
    parser.add_argument("--alphas", type=float, nargs="+", help="validate: OBF α's to evaluate")
    parser.add_argument("--workers", type=int, default=min(MAX_WORKERS, os.cpu_count() or 1))
    parser.add_argument("--chunk", type=int, default=250, help="replicates per worker task")
    parser.add_argument("--spread-replicates", type=int, default=100)
    parser.add_argument("--spread-blocks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0, help="base seed of all randomness")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument("--list", action="store_true", help="print the scenario names and exit")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must lie in 1..{MAX_WORKERS}")

    scenarios = select(GRIDS[args.mode](), args.designs, args.scenarios, args.reps)
    if not scenarios:
        parser.error("No scenario selected")
    if args.list:
        for scenario in scenarios:
            print(f"{scenario.name}  reps={scenario.reps}  group={scenario.group}")
        return

    level_sets, alphas = level_choice(args.mode, args.levels_from, args.alphas)
    settings = sim.RunSettings(
        stream=STREAMS[args.mode],
        level_sets=level_sets,
        alphas=alphas,
        base_seed=args.seed,
        workers=args.workers,
        chunk=args.chunk,
        spread_replicates=args.spread_replicates,
        spread_blocks=args.spread_blocks,
    )
    # Recorded at launch: the source of a long run is the one it started with.
    versions = {
        "python": platform.python_version(),
        **{name: metadata.version(name) for name in PACKAGES},
        "git": git_state(),
    }
    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out = args.out or runs_root() / "analysis" / "study2_short_sim" / f"{args.mode}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()

    def progress(stage: str, done: int, total: int) -> None:
        if done == total or done % max(1, total // 20) == 0:
            minutes = (time.perf_counter() - clock) / 60
            print(f"{stage}: {done}/{total}, {minutes:.1f} min", file=sys.stderr, flush=True)

    result = sim.run_scenarios(scenarios, settings, progress)
    summary = {
        "mode": args.mode,
        "started": started.isoformat(timespec="seconds"),
        "runtime_s": time.perf_counter() - clock,
        "settings": {
            "stream": settings.stream,
            "base_seed": settings.base_seed,
            "workers": settings.workers,
            "chunk": settings.chunk,
            "spread_replicates": settings.spread_replicates,
            "spread_blocks": settings.spread_blocks,
            "floors": list(settings.floors),
            "level_sets": {name: list(levels) for name, levels in level_sets.items()},
            "alphas": alphas,
            "levels_from": None if args.levels_from is None else str(args.levels_from),
            "constants": constants(),
        },
        "versions": versions,
        **result,
        **verdicts(args.mode, result["scenarios"]),
    }
    (out / "summary.json").write_text(json_text(summary))
    (out / "results.md").write_text(markdown(summary))
    print(f"Wrote {out}")


def select(
    scenarios: list[sim.Scenario],
    designs: list[str] | None,
    patterns: list[str] | None,
    reps: int | None,
) -> list[sim.Scenario]:
    if designs:
        scenarios = [s for s in scenarios if s.design in designs]
    if patterns:
        scenarios = [s for s in scenarios if any(fnmatch.fnmatch(s.name, p) for p in patterns)]
    if reps is not None:
        scenarios = [replace(s, reps=reps) for s in scenarios]
    return scenarios


def level_choice(
    mode: str, levels_from: Path | None, alphas: list[float] | None
) -> tuple[dict[str, tuple[float, float]], dict[str, float | None]]:
    """calibrate: the whole α grid; otherwise the planning levels plus the requested α's."""
    if mode == "calibrate":
        return sim.calibration_level_sets()
    requested = set(alphas or [])
    if levels_from is not None:
        selection = json.loads(levels_from.read_text())["selection"]
        requested |= {s["alpha"] for s in selection if s["alpha"] is not None}
    level_sets: dict[str, tuple[float, float]] = {"planning": sim.PLANNING_LEVELS}
    names: dict[str, float | None] = {"planning": None}
    for alpha in sorted(requested):
        name = f"obf_{alpha:.4f}"
        level_sets[name], names[name] = sim.obf_levels(alpha), alpha
    return level_sets, names


def constants() -> dict:
    return {
        "prompts": sim.PROMPTS,
        "cap": sim.CAP,
        "looks": list(sim.LOOKS),
        "margin": sim.MARGIN,
        "wide_margin": sim.WIDE_MARGIN,
        "secondary_level": sim.SECONDARY_LEVEL,
        "positions": {k: [list(p) for p in v] for k, v in sim.POSITIONS.items()},
        "families": {k: asdict(v) for k, v in sim.FAMILIES.items()},
        "rare": sim.RARE,
        "gamma_shape": sim.GAMMA_SHAPE,
        "failure_slope": sim.FAILURE_SLOPE,
        "chunk_blocks": sim.CHUNK_BLOCKS,
    }


def verdicts(mode: str, scenarios: list[dict]) -> dict:
    found = {"bias_allowance": sim.bias_allowance(scenarios)}
    if mode == "calibrate":
        found["selection"] = sim.select_levels(scenarios)
    else:
        found["checks"] = sim.validation_checks(scenarios)
    return found


# Markdown


def pp(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    return f"{100 * value:.{digits}f}"


def share(value: float | None) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    return f"{value:.4f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def headline_sets(summary: dict) -> set[tuple[str, float]]:
    """The (level set, floor) pairs shown in the result table."""
    floors = summary["settings"]["floors"]
    pairs = {("planning", floor) for floor in floors}
    for s in summary.get("selection", []):
        if s["level_set"] is not None:
            pairs.add((s["level_set"], s["floor"]))
    if summary["mode"] != "calibrate":
        pairs |= {(name, floor) for name in summary["settings"]["level_sets"] for floor in floors}
    return pairs


def markdown(summary: dict) -> str:
    parts = [
        f"# Study 2 short-regime simulation: {summary['mode']}",
        "",
        f"Started {summary['started']}, {summary['runtime_s'] / 60:.1f} min with "
        f"{summary['settings']['workers']} workers, stream `{summary['settings']['stream']}`, "
        f"base seed {summary['settings']['base_seed']}. Rates in %, MC SE in brackets; SDs, "
        "biases and bounds in pp.",
    ]
    if "selection" in summary:
        parts += ["", "## Selected levels", "", selection_table(summary["selection"])]
    if "checks" in summary:
        parts += ["", "## Constraint checks", "", checks_table(summary["checks"])]
    if summary["bias_allowance"]:
        parts += ["", "## Bias allowance", "", allowance_table(summary["bias_allowance"])]
    parts += ["", "## Results", "", results_table(summary)]
    parts += ["", "## Block diagnostics", "", diagnostics_table(summary["scenarios"])]
    return "\n".join(parts) + "\n"


def selection_table(selection: list[dict]) -> str:
    rows = []
    for s in selection:
        c = s["constraints"] or {}
        planning = s["planning"] or {}

        def value(summary: dict, key: str) -> str:
            item = summary.get(key) or {}
            return pp(item.get("value"))

        rows.append(
            [
                s["design"],
                pp(s["floor"], 0),
                "–" if s["alpha"] is None else f"{s['alpha']:.4f}",
                "–" if s["levels"] is None else " / ".join(f"{100 * v:.2f}" for v in s["levels"]),
                value(c, "max_false_positive"),
                value(c, "max_false_negative"),
                value(c, "max_false_equivalence_5pp"),
                value(c, "min_power_equivalence_at_0"),
                value(c, "min_power_direction_at_5pp"),
                "yes" if planning.get("meets") else "no",
            ]
        )
    header = ["design", "floor", "α", "levels %", "max dir+ @0", "max dir− @0", "max eq @5",
              "min pow eq @0", "min pow dir @5", "planning meets"]  # fmt: skip
    return table(header, rows)


def checks_table(checks: list[dict]) -> str:
    rows = []
    for check in checks:
        for group, c in check["groups"].items():
            rows.append(
                [
                    check["level_set"],
                    pp(check["floor"], 0),
                    group,
                    pp(c["max_false_positive"]["value"]),
                    pp(c["max_false_negative"]["value"]),
                    pp(c["max_false_equivalence_5pp"]["value"]),
                    pp(c["min_power_equivalence_at_0"]["value"]),
                    pp(c["min_power_direction_at_5pp"]["value"]),
                    "yes" if c["meets"] else "no",
                ]
            )
    header = ["level set", "floor", "group", "max dir+ @0", "max dir− @0", "max eq @5",
              "min pow eq @0", "min pow dir @5", "meets"]  # fmt: skip
    return table(header, rows)


def allowance_table(allowance: list[dict]) -> str:
    rows = []
    for a in allowance:
        for family, f in a["per_family"].items():
            rows.append(
                [
                    "∞" if not math.isfinite(a["w_max"]) else f"{a['w_max']:.2f}",
                    family,
                    f"{pp(f['bias'], 3)} ({pp(f['se'], 3)})",
                    f["scenario"],
                    "**max**" if family == a["family"] else "",
                ]
            )
    return table(["w_max", "family", "largest |bias| (SE)", "at scenario", "allowance"], rows)


def results_table(summary: dict) -> str:
    shown = headline_sets(summary)
    rows = []
    for s in summary["scenarios"]:
        for r in s["results"]:
            if (r["level_set"], r["floor"]) not in shown:
                continue
            cover = r["secondary_coverage"]
            rows.append(
                [
                    s["name"],
                    r["level_set"],
                    pp(r["floor"], 0),
                    f"{pp(r['p_equivalence'])} ({pp(r['se_equivalence'])})",
                    f"{pp(r['p_positive'])} ({pp(r['se_positive'])})",
                    f"{pp(r['p_negative'])} ({pp(r['se_negative'])})",
                    pp(r["p_inconclusive"]),
                    pp(r["p_inside_wide"]),
                    pp(r["power"]),
                    f"{r['mean_started']:.1f}",
                    f"{r['mean_supported']:.1f}",
                    pp(r["p_cap_exhausted"]),
                    pp(r["p_look1_stop"]),
                    pp(r["bias"], 3),
                    pp(r["bias_vs_block_mean"], 3),
                    pp(r["cohort_bound_width"], 1),
                    pp(r["p_cohort_bounds_straddle_5pp"]),
                    pp(cover["P-F"]["truth"], 1),
                    pp(cover["A-P"]["truth"], 1),
                    pp(r["three_arm_attrition"], 1),
                    str(s["reps"]),
                ]
            )
    header = ["scenario", "levels", "floor", "eq", "dir+", "dir−", "inc", "±8 in", "power",
              "E started", "E supp", "cap", "look-1 stop", "bias", "bias vs block mean",
              "cohort width", "straddle ±5", "cov P−F", "cov A−P", "3-arm attr",
              "reps"]  # fmt: skip
    return table(header, rows)


def diagnostics_table(scenarios: list[dict]) -> str:
    rows = []
    for s in scenarios:
        d, p = s["diagnostics"], s["parameters"]
        block = d["unsupported_per_block"]
        arms = d["unsupported_per_arm"]
        calibration = s["calibration"] or {}
        rows.append(
            [
                s["name"],
                f"{p['spread']:.4f}",
                pp(calibration.get("attained_sd")),
                pp(calibration.get("noise_sd")),
                pp(d["attained_sd"]),
                f"{pp(d['block_bias'], 3)} ({pp(d['block_bias_se'], 3)})",
                " / ".join(pp(block[c], 1) for c in sim.CAUSES),
                " / ".join(pp(arms["A"][c], 1) for c in sim.CAUSES),
                " / ".join(pp(arms["F"][c], 1) for c in sim.CAUSES),
                " / ".join(pp(arms["P"][c], 1) for c in sim.CAUSES),
                pp(d["three_arm_attrition"], 1),
                f"{calibration['cpu_seconds']:.0f}" if calibration else "–",
                f"{s['cpu_seconds']:.0f}",
                f"{s['seconds']:.0f}",
                f"{1000 * s['cpu_seconds'] / s['reps']:.1f}",
            ]
        )
    header = ["scenario", "spread", "cal SD", "noise SD", "run SD", "block bias (SE)",
              "unsupp/block b/t/w", "A b/t/w", "F b/t/w", "P b/t/w", "3-arm attr",
              "cal CPU s", "CPU s", "wall s", "CPU ms/rep"]  # fmt: skip
    return table(header, rows)


def json_text(data: dict) -> str:
    """Standard JSON: non-finite numbers become null."""
    return json.dumps(_finite(data), indent=2, allow_nan=False) + "\n"


def _finite(value):
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
