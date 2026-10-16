"""Simulation of study 2's short-regime confirmation (amendment 3): levels, validation and bias.

Modes (the model and the procedure: src/zero_field_refusal/analysis/study2_short_sim.py). Every
mode evaluates every look design (24/50, 36/60, one look at 50, 24/50 with equivalence only at
the terminal look) and every SE floor on the same replicates:

- calibrate: the central model at every target SD × Δ, on the ``calibration`` stream, at every α
  of the grid; selects per look design and SE floor the largest α meeting the error constraints
  over the binding SDs (3, 5, 10 pp).
- bias: paired interpolation bias over curve family × selection miss × w_max (``bias`` stream),
  and the allowance b over w_max 0.35 and 0.47.
- validate: the validation grid on the independent ``validation`` stream, at α = 0.05 and at the
  α's selected by --levels-from (a calibrate run's summary.json) or given by --alphas; with
  --allowance-from (a bias run's summary.json) or --allowance it adds the pointwise readings
  with b.
- validate-bootstrap: revision 3's grid (every shape × SD × Δ ∈ {0, ±2.5, ±5, ±5.5}) on the
  ``validation`` stream, sized by --boundary-reps (Δ = 0, ±5 pp) and --other-reps; meant with
  --looks 24_50 36_60 --intervals t bootstrap_t.
- validate-stress: revision 4's grid on the ``validation`` stream: every non-shape group of the
  validation grid (cap with correlated and independent failures), and the heavy shape over every
  SD × Δ ∈ {0, ±2.5, ±5, ±5.5}; sized by --boundary-reps and --other-reps.
- confirm: revision 4's fresh confirmation of the decided design: every shape × SD at
  Δ ∈ {0, ±5} on the new ``confirmation`` stream (--reps per scenario, default 20 000).
- decided-bias, decided, decided-gate: revision 5 at the decided w_max 0.47, on the
  ``confirmation`` stream: the bias allowance grid; kink and stress_slopes at SD 5, the shape grid
  and the cap scenarios; and the cap scenarios at Δ = ±5 prescreened on the validation gate
  (``reps`` = gates drawn; only passing studies are simulated).
- smoke: one reduced scenario per shape and per revision-1 addition.

--looks restricts the look designs evaluated (default: all four); --intervals picks t and/or
bootstrap_t (the studentized bootstrap of the mean with --bootstrap resamples per look and
replicate, computed in the workers). A calibrate run selects α per look design × interval ×
SE floor. --rules adds ``eq_min36`` (equivalence only at a look with ≥ 36 supported blocks);
--floors restricts the SE floors; --exact-alphas evaluates exactly --alphas (otherwise α = 0.05
and the selected α's are added). Every mode also simulates the 12-block validation gate.

Writes to --out (default: $RUNS_DIR/analysis/study2_short_sim/<mode>-<UTC timestamp>/):
- summary.json: settings, versions, run time, the spread calibrations, every scenario (its
  settings, parameters with the population curves, block diagnostics, compute time and one result
  row per look design × α × SE floor) and the mode's verdicts (selection, checks, allowance);
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
    "validate-bootstrap": sim.bootstrap_validation_grid,
    "validate-stress": sim.stress_validation_grid,
    "confirm": sim.confirmation_grid,
    "decided-bias": sim.decided_bias_grid,
    "decided": sim.decided_grid,
    "decided-gate": sim.decided_gate_grid,
    "bias": sim.bias_grid,
    "smoke": sim.smoke_grid,
}
SIZED = ("validate-bootstrap", "validate-stress")  # modes taking --boundary-reps, --other-reps
STREAMS = {
    "calibrate": "calibration",
    "validate": "validation",
    "validate-bootstrap": "validation",
    "validate-stress": "validation",
    "confirm": "confirmation",
    "decided-bias": "confirmation",
    "decided": "confirmation",
    "decided-gate": "confirmation",
    "bias": "bias",
    "smoke": "smoke",
}
DESIGNS = ("interp", "interp_grid")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=tuple(GRIDS))
    parser.add_argument("--designs", nargs="+", choices=DESIGNS, help="keep these designs only")
    parser.add_argument(
        "--looks", nargs="+", choices=tuple(sim.DESIGNS_BY_NAME), help="look designs to evaluate"
    )
    parser.add_argument("--intervals", nargs="+", choices=sim.INTERVALS, default=["t"])
    parser.add_argument("--bootstrap", type=int, default=sim.BOOTSTRAP_RESAMPLES, help="resamples")
    parser.add_argument("--rules", nargs="+", choices=sim.RULES, default=["standard"])
    parser.add_argument("--floors", type=float, nargs="+", help="SE floors (default: all)")
    parser.add_argument("--exact-alphas", action="store_true", help="evaluate exactly --alphas")
    parser.add_argument("--boundary-reps", type=int, help=f"{', '.join(SIZED)}: Δ = 0, ±5 pp")
    parser.add_argument("--other-reps", type=int, help=f"{', '.join(SIZED)}: other Δ")
    parser.add_argument("--scenarios", nargs="+", metavar="PATTERN", help="fnmatch on names")
    parser.add_argument("--reps", type=int, help="replicates per scenario (overrides the grid)")
    parser.add_argument("--levels-from", type=Path, help="a calibrate run's summary.json")
    parser.add_argument("--alphas", type=float, nargs="+", help="extra α's to evaluate")
    allowance = parser.add_mutually_exclusive_group()
    allowance.add_argument("--allowance-from", type=Path, help="a bias run's summary.json")
    allowance.add_argument("--allowance", type=float, help="the bias allowance b (fraction)")
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

    grid = GRIDS[args.mode]
    if args.mode in SIZED:
        sizes = {"boundary_reps": args.boundary_reps, "other_reps": args.other_reps}
        scenarios = grid(**{k: v for k, v in sizes.items() if v is not None})
    elif args.boundary_reps is not None or args.other_reps is not None:
        parser.error(f"--boundary-reps and --other-reps apply to {', '.join(SIZED)} only")
    else:
        scenarios = grid()
    scenarios = select(scenarios, args.designs, args.scenarios, args.reps)
    if not scenarios:
        parser.error("No scenario selected")
    if args.list:
        for scenario in scenarios:
            print(f"{scenario.name}  reps={scenario.reps}  group={scenario.group}")
        return

    selection = selected_levels(args.levels_from)
    if args.exact_alphas and not args.alphas:
        parser.error("--exact-alphas needs --alphas")
    alphas = tuple(sorted(args.alphas)) if args.exact_alphas else None
    alphas = alphas or alpha_choice(args.mode, selection, args.alphas)
    b = args.allowance
    if args.allowance_from is not None:
        found = json.loads(args.allowance_from.read_text())["bias_allowance"]["allowance"]
        b = None if found is None else found["value"]
    settings = sim.RunSettings(
        stream=STREAMS[args.mode],
        alphas=alphas,
        allowance=b,
        base_seed=args.seed,
        workers=args.workers,
        chunk=args.chunk,
        spread_replicates=args.spread_replicates,
        spread_blocks=args.spread_blocks,
        designs=tuple(args.looks) if args.looks else tuple(d.name for d in sim.LOOK_DESIGNS),
        intervals=tuple(dict.fromkeys(args.intervals)),
        resamples=args.bootstrap,
        rules=tuple(dict.fromkeys(args.rules)),
        floors=tuple(args.floors) if args.floors else sim.SE_FLOORS,
        prescreen=args.mode == "decided-gate",
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
            "looks": list(settings.designs),
            "intervals": list(settings.intervals),
            "rules": list(settings.rules),
            "prescreen": settings.prescreen,
            "resamples": settings.resamples,
            "alphas": list(settings.alphas),
            "allowance": settings.allowance,
            "levels_from": None if args.levels_from is None else str(args.levels_from),
            "allowance_from": None if args.allowance_from is None else str(args.allowance_from),
            "constants": constants(),
        },
        "versions": versions,
        **result,
        **verdicts(args.mode, result["scenarios"], selection),
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


def selected_levels(path: Path | None) -> dict[tuple[str, str, float], float | None]:
    """(look design, interval, floor) -> the α a calibrate run selected (None if none passed).
    Runs before revision 3 selected for the t interval only."""
    if path is None:
        return {}
    selection = json.loads(path.read_text())["selection"]
    return {(s["looks"], s.get("interval", "t"), s["floor"]): s["alpha"] for s in selection}


def alpha_choice(
    mode: str, selection: dict[tuple[str, str, float], float | None], extra: list[float] | None
) -> tuple[float, ...]:
    """calibrate: the whole α grid; otherwise α = 0.05, the selected α's and any extra ones."""
    if mode == "calibrate":
        return sim.ALPHAS
    alphas = {sim.PLANNING_ALPHA, *(extra or [])}
    alphas |= {alpha for alpha in selection.values() if alpha is not None}
    return tuple(sorted(alphas))


def constants() -> dict:
    return {
        "prompts": sim.PROMPTS,
        "cap": sim.CAP,
        "look_designs": [asdict(d) for d in sim.LOOK_DESIGNS],
        "generate_until": sim.GENERATE_UNTIL,
        "margin": sim.MARGIN,
        "wide_margin": sim.WIDE_MARGIN,
        "secondary_level": sim.SECONDARY_LEVEL,
        "positions": {k: [list(p) for p in v] for k, v in sim.POSITIONS.items()},
        "families": {k: asdict(v) for k, v in sim.FAMILIES.items()},
        "fit_grid": [float(x) for x in sim.FIT_GRID],
        "rare": sim.RARE,
        "gamma_shape": sim.GAMMA_SHAPE,
        "failure_slope": sim.FAILURE_SLOPE,
        "chunk_blocks": sim.CHUNK_BLOCKS,
    }


def verdicts(mode: str, scenarios: list[dict], selection: dict) -> dict:
    found = {"bias_allowance": sim.bias_allowance(scenarios)}
    if mode == "calibrate":
        found["selection"] = sim.select_levels(scenarios)
    else:
        found["checks"] = sim.validation_checks(scenarios, selection)
    return found


# Markdown


def pp(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "–"
    return f"{100 * value:.{digits}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def shown_alpha(summary: dict) -> dict[tuple[str, str, float], float]:
    """(look design, interval, floor) -> the α whose results the tables show: the selected α
    (from this calibrate run or from --levels-from), else α = 0.05 if it was evaluated, else the
    smallest α evaluated."""
    if "selection" in summary:
        chosen = {
            (s["looks"], s.get("interval", "t"), s["floor"]): s["alpha"]
            for s in summary["selection"]
        }
    else:
        path = summary["settings"]["levels_from"]
        chosen = selected_levels(Path(path)) if path else {}
    floors = summary["settings"]["floors"]
    evaluated = summary["settings"]["alphas"]
    default = sim.PLANNING_ALPHA if sim.PLANNING_ALPHA in evaluated else min(evaluated)
    return {
        (d.name, interval, floor): chosen.get((d.name, interval, floor)) or default
        for d in sim.LOOK_DESIGNS
        for interval in sim.INTERVALS
        for floor in floors
    }


def markdown(summary: dict) -> str:
    settings = summary["settings"]
    allowance = settings["allowance"]
    parts = [
        f"# Study 2 short-regime simulation: {summary['mode']}",
        "",
        f"Started {summary['started']}, {summary['runtime_s'] / 60:.1f} min with "
        f"{settings['workers']} workers, stream `{settings['stream']}`, base seed "
        f"{settings['base_seed']}. Rates in %, MC SE in brackets; SDs, biases and bounds in pp. "
        "Tables show each look design at its selected α (else α = 0.05) and floor 0"
        + (
            f"; pointwise readings use b = {pp(allowance, 3)} pp." if allowance is not None else "."
        ),
    ]
    if "selection" in summary:
        parts += ["", "## Selected levels", "", selection_table(summary["selection"])]
    if "checks" in summary:
        parts += ["", "## Constraint checks", "", checks_table(summary["checks"])]
    if summary["bias_allowance"]["per_w_max"]:
        parts += ["", "## Bias allowance", "", allowance_table(summary["bias_allowance"])]
    parts += ["", "## Results", "", results_table(summary)]
    parts += ["", "## Block diagnostics", "", diagnostics_table(summary["scenarios"])]
    return "\n".join(parts) + "\n"


def _value(summary: dict | None, key: str) -> str:
    item = (summary or {}).get(key) or {}
    return pp(item.get("value"))


def selection_table(selection: list[dict]) -> str:
    rows = []
    for s in selection:
        c = s["constraints"]
        rows.append(
            [
                s["looks"],
                s.get("interval", "t"),
                pp(s["floor"], 0),
                "–" if s["alpha"] is None else f"{s['alpha']:.4f}",
                "–" if s["levels"] is None else " / ".join(f"{100 * v:.2f}" for v in s["levels"]),
                _value(c, "max_false_positive"),
                _value(c, "max_false_negative"),
                _value(c, "max_false_equivalence_5pp"),
                _value(c, "min_power_equivalence_at_0"),
                _value(c, "min_power_direction_at_5pp"),
                "yes" if (s["at_alpha_0.05"] or {}).get("meets") else "no",
            ]
        )
    header = ["looks", "interval", "floor", "α", "levels %", "max dir+ @0", "max dir− @0",
              "max eq @5", "min pow eq @0", "min pow dir @5", "α 0.05 meets"]  # fmt: skip
    return table(header, rows)


def checks_table(checks: list[dict]) -> str:
    rows = []
    for check in checks:
        for group, c in check["groups"].items():
            rows.append(
                [
                    check["looks"],
                    check.get("interval", "t"),
                    check.get("rule", "standard"),
                    pp(check["floor"], 0),
                    f"{check['which']} ({check['alpha']:.4f})",
                    group,
                    _value(c, "max_false_positive"),
                    _value(c, "max_false_negative"),
                    _value(c, "max_false_equivalence_5pp"),
                    _value(c, "min_power_equivalence_at_0"),
                    _value(c, "min_power_direction_at_5pp"),
                    _value(c, "max_cap_exhausted"),
                    "–" if c["max_mean_started"] is None else f"{c['max_mean_started']:.1f}",
                    "yes" if c["meets"] else "no",
                ]
            )
    header = ["looks", "interval", "rule", "floor", "α", "group", "max dir+ @0", "max dir− @0",
              "max eq @5", "min pow eq @0", "min pow dir @5", "max cap", "max E started",
              "meets"]  # fmt: skip
    return table(header, rows)


def allowance_table(allowance: dict) -> str:
    rows = []
    for entry in allowance["per_w_max"]:
        for family, f in entry["per_family"].items():
            rows.append(
                [
                    "∞" if not math.isfinite(entry["w_max"]) else f"{entry['w_max']:.2f}",
                    family,
                    f"{pp(f['bias'], 3)} ({pp(f['se'], 3)})",
                    f["scenario"],
                    "**max**" if family == entry["family"] else "",
                ]
            )
    text = table(["w_max", "family", "largest |bias| (SE)", "at scenario", "largest"], rows)
    b = allowance["allowance"]
    if b is not None:
        text += (
            f"\n\nAllowance b = {pp(b['value'], 3)} pp ({b['family']}, w_max {b['w_max']:.2f}, "
            f"{b['scenario']}; over w_max {b['over_w_max']} and all families)."
        )
    return text


def results_table(summary: dict) -> str:
    shown = shown_alpha(summary)
    rows = []
    for s in summary["scenarios"]:
        for r in s["results"]:
            interval = r.get("interval", "t")
            floor = min(summary["settings"]["floors"])
            if r["floor"] != floor or shown.get((r["looks"], interval, floor)) != r["alpha"]:
                continue
            gated = r.get("gated") or {}
            cover = r["secondary_coverage"]
            point = r.get("pointwise") or {}
            rows.append(
                [
                    s["name"],
                    r["looks"],
                    interval,
                    r.get("rule", "standard"),
                    f"{r['alpha']:.4f}",
                    pp(r.get("attained_sd"), 1),
                    f"{pp(r['p_equivalence'])} ({pp(r['se_equivalence'])})",
                    f"{pp(r['p_positive'])} ({pp(r['se_positive'])})",
                    f"{pp(r['p_negative'])} ({pp(r['se_negative'])})",
                    pp(r["p_inconclusive"]),
                    pp(r["p_inside_wide"]),
                    pp(r["power"]),
                    f"{r['mean_started']:.1f}",
                    f"{r['mean_supported']:.1f}",
                    pp(r["p_cap_exhausted"]),
                    pp(gated.get("p_equivalence")),
                    pp(r["p_look1_stop"]),
                    pp(r["bias"], 3),
                    pp(r["bias_vs_block_mean"], 3),
                    " / ".join(
                        pp(point.get(f"p_{k}")) for k in ("equivalence", "positive", "negative")
                    ),
                    pp(r["cohort_bound_width"], 1),
                    pp(r["p_cohort_bounds_straddle_5pp"]),
                    pp(cover["P-F"]["truth"], 1),
                    pp(cover["A-P"]["truth"], 1),
                    str(s["reps"]),
                ]
            )
    header = ["scenario", "looks", "interval", "rule", "α", "SD", "eq", "dir+", "dir−", "inc",
              "±8 in", "power",
              "E started", "E supp", "cap", "eq (gate passed)", "look-1 stop", "bias vs Δ",
              "bias vs interp",
              "pointwise eq/dir+/dir−", "cohort width", "straddle ±5", "cov P−F", "cov A−P",
              "reps"]  # fmt: skip
    return table(header, rows)


def diagnostics_table(scenarios: list[dict]) -> str:
    rows = []
    for s in scenarios:
        d, p = s["diagnostics"], s["parameters"]
        block = d["unsupported_per_block"]
        arms = d["unsupported_per_arm"]
        curve = p["population_curve"]
        calibration = s["calibration"] or {}

        rows.append(
            [
                s["name"],
                f"{p['spread']:.4f}",
                pp(calibration.get("noise_sd")),
                pp(d["attained_sd"]),
                f"{pp(d['block_bias'], 3)} ({pp(d['block_bias_se'], 3)})",
                attained(curve["A"]),
                attained(curve["F"]),
                " / ".join(pp(block[c], 1) for c in sim.CAUSES),
                " / ".join(pp(arms["A"][c], 1) for c in sim.CAUSES),
                " / ".join(pp(arms["F"][c], 1) for c in sim.CAUSES),
                pp(d["three_arm_attrition"], 1),
                pp(d.get("p_gate_fail")),
                f"{calibration['cpu_seconds']:.0f}" if calibration else "–",
                f"{s['cpu_seconds']:.0f}",
                f"{1000 * s['cpu_seconds'] / s['reps']:.1f}",
            ]
        )
    header = ["scenario", "spread", "noise SD", "run SD", "block bias vs Δ (SE)",
              "A h̄′, h̄″ attained/target", "F h̄′, h̄″ attained/target", "unsupp/block b/t/w",
              "A b/t/w", "F b/t/w", "3-arm attr", "gate fail", "cal CPU s", "CPU s",
              "CPU ms/rep"]  # fmt: skip
    return table(header, rows)


def attained(curve: dict) -> str:
    """Population slope and curvature at K*, attained/target."""
    a, t = curve["attained"], curve["target"]
    return f"{a['slope']:.3f}/{t['slope']:.3f}, {a['curvature']:+.3f}/{t['curvature']:+.3f}"


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
