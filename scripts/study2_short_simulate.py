"""Simulation of study 2's short-regime confirmation (amendment 3): levels, validation and bias.

Modes (the model and the procedure: src/zero_field_refusal/analysis/study2_short_sim.py). Every
mode but the decided ones evaluates by default every look design (24/50, 36/60, one look at 50,
24/50 with equivalence only at the terminal look) and every SE floor on the same replicates:

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
  (``reps`` = gates drawn; only passing studies are simulated). Without flags they evaluate the
  decided design: look design 36/60 (``DECIDED_DESIGN``), the bootstrap-t interval with 2 000
  resamples, α 0.040 (``DECIDED_ALPHA``) only, SE floor 0 (``DECIDED_SE_FLOOR``), the rules
  ``standard`` and ``eq_min36`` (the decided one, ``DECIDED_RULE``), w_max 0.47 (the grids'
  ``DECIDED_W_MAX``), and for decided and decided-gate the pointwise readings with
  b = ``DECIDED_ALLOWANCE``; decided-bias measures b and takes none, as revision 5 did.
  decided-gate runs 5 000 replicates per worker task.
- decided-secondary: the secondary check of 2026-10-17, on the new ``secondary`` stream: the
  decided design over every shape × SD at Δ ∈ {0, ±5} (10 000 replicates each), with the
  secondary contrasts' t and bootstrap-t intervals side by side (--secondary-intervals); the
  verdict ``secondary_checks`` gives per shape × SD the worst coverage over Δ of each contrast
  and of both jointly, against 97.5 % and 95 %.
- smoke: one reduced scenario per shape and per revision-1 addition.

Amendment 4 (batch 20 and the mechanism cohort; the inputs record
docs/study2-short-b20-inputs.json, read by ``sim.read_inputs``):

- mechanism: the mechanism cohort's simulation (``sim.mechanism_grid``: the required grid at
  10 000 replicates per scenario and the stress grid at --stress-reps, default 2 000) on the
  ``mechanism-calibration`` stream, every level of ``sim.MECHANISM_LEVELS`` (0.95 … 0.99)
  evaluated from the same bootstrap draws, with --bootstrap resamples per replicate and interval
  (default 2 000). Coverage is of each replicate's conditional truth (review fa24, Q1: fresh
  supported seed blocks given its prompt panel and selection misses, ``sim.PanelTruth``), a
  missing interval not covering. Its verdict ``checks`` (``sim.mechanism_checks``) applies the
  calibration rule: every required scenario (normal and skewed at SD 5 and 10 pp; 1 620 with
  the five attrition modes) against 0.95 − 2·√(0.95·0.05/R); the smallest level at which every
  one holds; and, predeclared there, the borderline scenarios (under 95 % at that level).
- mechanism-check: the required grid on the separate ``mechanism-check`` stream at one level,
  --level or the level of --level-from (a mechanism run's summary.json), once: the borderline
  scenarios of --level-from at ``sim.PRECISION_REPS`` (50 000; --precision-reps) against their
  floor 0.95 − 2·√(0.95·0.05/50 000), the others at 10 000.
- inputs: measures the inputs record on 25-step runs under --runs-root, from K and support only
  (``sim.measure_inputs``), at --batch on --seeds with each arm's selected rates (--rates
  A=r1,r2,r3 F=r1,r2,r3 P=r1,r2,r3), and writes it to --out (a JSON file).
- compare: the decision rates of a rerun (--new, a summary.json) against archived runs (--old),
  scenario by scenario, matched on every setting but the per-arm inputs; a difference beyond two
  MC SEs of the difference is flagged.
- final: the level-fixing run after S1 in one command (below).

--conditional adds to the primary modes each replicate's conditional truth of Δ (its
interpolation contrast over fresh primary-supported blocks given its panel and misses, and its
pointwise contrast) and the rows' ``conditional`` rates and ``conditional_checks`` against it,
beside the population figures; no fixed level depends on them.

--inputs applies an inputs record to the primary modes too (``sim.inputs_scenario``: per-arm K
spreads, and per-arm failure rates where a scenario has none of its own); the mechanism modes
read docs/study2-short-b20-inputs.json by default. Every simulation copies the record it read to
inputs.json, records its sha256 in summary.json, and writes inventory.tsv: the path, size and
sha256 of every file it wrote.

--upper U (amendment 5; every mode) simulates at U: it sets K* = √(5U), at which the inputs
measure support, and the nominal half-span ln √(U/5) of the endpoints, and nothing else
(``sim.set_upper``; the workers inherit it). The default, 8, gives every earlier run's constants
(K* = √40, half-span 0.235). A candidate at U = 7 or U = 6 is forecast with --upper 7 or 6 at its
rates before its confirmation. summary.json records U, K* and the half-span in its constants.

The final run after S1, once S1's K-only data exist (the rates: each arm's selected triple, from
the selection's record; both roots, because a selected rate may be one of Stage A's grid rates,
FA's 2e-5 at U = 8, and a selected rate without a run counts as a failure):

    python scripts/study2_short_simulate.py final \\
        --runs-root $RUNS_DIR/study2_short $RUNS_DIR/study2_stage_a \\
        --rates A=r1,r2,r3 F=r1,r2,r3 P=r1,r2,r3 --workers 8 --write \\
        --out $RUNS_DIR/analysis/study2_short_sim/<date>-b20-final

It measures the inputs (B = 20, seeds 2000-2002; the K spreads over every complete rate under
both roots, Stage A's grid and S1's rates) into OUT/inputs.json, then runs, binding results
first: mechanism on the required grid (OUT/mechanism), one mechanism-check at the level it
chose, 0.95 included, with its borderline scenarios at 50 000 (OUT/check), the primary rerun
decided (with --conditional), decided-gate and decided-bias with the inputs (OUT/decided, OUT/gate,
OUT/bias) and their comparison with amendment 3's archived runs (OUT/compare-*), and mechanism on
the stress grid at the fixed level (OUT/stress, reported only). It writes OUT/final.json with the
fixed level and the flagged rates. A level is fixed only if it holds on the calibration stream
and passes the single binding check at it, 0.95 included; otherwise none is, with no repetition,
and final.json and the log warn that docs/study2-short-fixation.json keeps its placeholder
mechanism.level 0.95, which must not be used. With --write and a fixed level it also writes the
record to docs/study2-short-b20-inputs.json and the level to docs/study2-short-fixation.json
(mechanism.level); without a fixed level it writes neither. Without --runs-root it uses --inputs
as it is. A written level other than 0.95 also needs the tracked record's pin in
tests/test_study2_short_campaigns.py (test_fixation_requires_the_mechanism_key) updated.

--looks restricts the look designs evaluated; --intervals picks t and/or bootstrap_t (the
studentized bootstrap of the mean with --bootstrap resamples per look and replicate, computed in
the workers); --secondary-intervals does the same for the secondary contrasts (default t;
decided-secondary: t and bootstrap_t, with --bootstrap resamples per contrast). A calibrate run
selects α per look design × interval × SE floor. --rules adds ``eq_min36`` (equivalence only at
a look with ≥ 36 supported blocks); --floors restricts the SE floors; --exact-alphas evaluates
exactly --alphas (otherwise α = 0.05, or the decided α in the decided modes, and the selected
α's are added). Every mode also simulates the 12-block validation gate.

Revision 5, the decided design, ran on 2026-10-16 (archived under
runs/analysis/study2_short_sim/2026-10-16/, inventory
runs/archive/inventory-2026-10-16-study2-short-sim.tsv). Its four runs are reproduced, from the
same seeds, by:

    python scripts/study2_short_simulate.py decided-bias --workers 8 --out OUT/bias
    python scripts/study2_short_simulate.py decided --workers 8 --out OUT/main
    python scripts/study2_short_simulate.py decided --workers 8 \\
        --scenarios 'interp_normal_kink_*' 'interp_normal_stress_slopes_*' --out OUT/families
    python scripts/study2_short_simulate.py decided-gate --workers 8 --out OUT/gate

bias and main ran from ec2c7c8 through a scratchpad driver with exactly these scenarios and
settings, before the modes existed. families and gate ran from e900179 through these modes, with
the flags that are now their defaults (gate with --chunk 5000). Every decision rate is
reproduced bit for bit (checked on 2026-10-17 for one scenario of bias, families and gate;
archived in runs/analysis/study2_short_sim/2026-10-17/). The
pointwise readings can differ in two ways only: revision 5 used the bias run's unrounded b,
0.010841274718211662, where the modes use the recorded DECIDED_ALLOWANCE 0.010841 (for kink at
Δ = 0 one replicate of 20 000 then differs, pointwise equivalence 76.775 against 76.770 %;
``--allowance 0.010841274718211662`` restores it exactly); and main's readings predate the
equivalence mask (review N1), for which families reran kink and stress_slopes.

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
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from zero_field_refusal.analysis import study2_short_sim as sim
from zero_field_refusal.paths import REPO_ROOT, runs_root
from zero_field_refusal.tracking import git_state

PACKAGES = ("numpy", "scipy", "zero-field-refusal")
DEFAULT_WORKERS = 12  # of the 16 cores of the shared laptop (the lead allows up to 12)
GRIDS = {
    "calibrate": sim.calibration_grid,
    "validate": sim.validation_grid,
    "validate-bootstrap": sim.bootstrap_validation_grid,
    "validate-stress": sim.stress_validation_grid,
    "confirm": sim.confirmation_grid,
    "decided-bias": sim.decided_bias_grid,
    "decided": sim.decided_grid,
    "decided-gate": sim.decided_gate_grid,
    "decided-secondary": sim.decided_secondary_grid,
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
    "decided-secondary": "secondary",
    "bias": "bias",
    "smoke": "smoke",
}
DESIGNS = ("interp", "interp_grid")
# The decided design (revision 5's modes and the secondary check).
DECIDED_MODES = ("decided-bias", "decided", "decided-gate", "decided-secondary")
ALLOWANCE_MODES = ("decided", "decided-gate", "decided-secondary")  # b = DECIDED_ALLOWANCE
SECONDARY_MODES = ("decided-secondary",)  # t and bootstrap-t secondaries by default
CHUNK = 250  # replicates per worker task
GATE_CHUNK = 5000  # decided-gate: most replicates are a validation gate only
# Amendment 4.
MECHANISM_MODES = {"mechanism": "calibration", "mechanism-check": "check"}
OTHER_MODES = ("inputs", "compare", "final")
INPUTS_RECORD = REPO_ROOT / sim.INPUTS_RECORD
FIXATION_RECORD = REPO_ROOT / "docs" / "study2-short-fixation.json"
# Amendment 3's archived decided runs (revision 5), which the primary rerun is compared with.
ARCHIVED = {
    "decided": "analysis/study2_short_sim/2026-10-16/full-r5/main/summary.json",
    "decided-gate": "analysis/study2_short_sim/2026-10-16/full-r5/gate/summary.json",
    "decided-bias": "analysis/study2_short_sim/2026-10-16/full-r5/bias/summary.json",
}
FINAL_PRIMARY = {"decided": "decided", "decided-gate": "gate", "decided-bias": "bias"}
INVENTORY = "inventory.tsv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=(*GRIDS, *MECHANISM_MODES, *OTHER_MODES))
    parser.add_argument("--designs", nargs="+", choices=DESIGNS, help="keep these designs only")
    parser.add_argument(
        "--looks",
        nargs="+",
        choices=tuple(sim.DESIGNS_BY_NAME),
        help="look designs to evaluate (default: all; decided modes: the decided one)",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        choices=sim.INTERVALS,
        help="default: t; decided modes: bootstrap_t",
    )
    parser.add_argument(
        "--secondary-intervals",
        nargs="+",
        choices=sim.INTERVALS,
        help="secondary contrasts' intervals (default: t; decided-secondary: t bootstrap_t)",
    )
    parser.add_argument("--bootstrap", type=int, default=sim.BOOTSTRAP_RESAMPLES, help="resamples")
    parser.add_argument(
        "--rules", nargs="+", choices=sim.RULES, help="default: standard; decided modes: both"
    )
    parser.add_argument(
        "--floors", type=float, nargs="+", help="SE floors (default: all; decided modes: 0)"
    )
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
    parser.add_argument("--workers", type=int, default=min(DEFAULT_WORKERS, os.cpu_count() or 1))
    parser.add_argument(
        "--chunk",
        type=int,
        help=f"replicates per worker task (default: {CHUNK}; decided-gate: {GATE_CHUNK})",
    )
    parser.add_argument("--spread-replicates", type=int, default=100)
    parser.add_argument("--spread-blocks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0, help="base seed of all randomness")
    parser.add_argument("--out", type=Path, help="output directory")
    parser.add_argument("--list", action="store_true", help="print the scenario names and exit")
    # Amendment 4.
    parser.add_argument(
        "--inputs",
        type=Path,
        help=f"an inputs record (mechanism modes: default {sim.INPUTS_RECORD})",
    )
    parser.add_argument("--stress-reps", type=int, help="mechanism: replicates per stress scenario")
    parser.add_argument(
        "--groups", nargs="+", choices=("required", "stress"), help="mechanism: the grid's groups"
    )
    parser.add_argument("--levels", type=float, nargs="+", help="mechanism: the levels evaluated")
    parser.add_argument("--level", type=float, help="mechanism-check: the level checked")
    parser.add_argument("--level-from", type=Path, help="mechanism-check: a mechanism summary.json")
    parser.add_argument(
        "--precision-reps",
        type=int,
        help=f"mechanism-check: replicates of a borderline scenario (default {sim.PRECISION_REPS})",
    )
    parser.add_argument(
        "--readings-at", type=float, help="mechanism: the level the readings tables show"
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="primary modes: rates against each replicate's conditional truth of Δ",
    )
    parser.add_argument("--runs-root", type=Path, nargs="+", help="inputs, final: run directories")
    parser.add_argument(
        "--rates",
        nargs="+",
        metavar="ARM=R1,R2,R3",
        help="inputs, final: each arm's selected rates (A, F, P)",
    )
    parser.add_argument("--batch", type=int, default=20, help="inputs, final: the batch size")
    parser.add_argument("--upper", type=float, default=sim.DEFAULT_UPPER,
                        help="U: K* = sqrt(5U) and the nominal half-span ln sqrt(U/5) only "
                             "(default 8, every earlier run's)")  # fmt: skip
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[2000, 2001, 2002], help="inputs, final: seeds"
    )
    parser.add_argument("--new", type=Path, help="compare: the rerun's summary.json")
    parser.add_argument("--old", type=Path, nargs="+", help="compare: archived summary.json files")
    parser.add_argument(
        "--write",
        action="store_true",
        help="final: write the record and the fixed level into docs/",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:  # U before anything is simulated or measured; the workers inherit it
        sim.set_upper(args.upper)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.workers <= (os.cpu_count() or 1):  # a dedicated CPU Pod may use every core
        parser.error(f"--workers must lie in 1..{os.cpu_count() or 1}")
    if args.mode in MECHANISM_MODES:
        return mechanism_main(args, parser)
    if args.mode == "inputs":
        return inputs_main(args, parser)
    if args.mode == "compare":
        return compare_main(args, parser)
    if args.mode == "final":
        return final_main(args, parser)

    grid = GRIDS[args.mode]
    if args.mode in SIZED:
        sizes = {"boundary_reps": args.boundary_reps, "other_reps": args.other_reps}
        scenarios = grid(**{k: v for k, v in sizes.items() if v is not None})
    elif args.boundary_reps is not None or args.other_reps is not None:
        parser.error(f"--boundary-reps and --other-reps apply to {', '.join(SIZED)} only")
    else:
        scenarios = grid()
    inputs = None if args.inputs is None else sim.read_inputs(args.inputs)
    if inputs is not None:
        scenarios = [sim.inputs_scenario(s, inputs) for s in scenarios]
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
    settings = run_settings(args, selection)
    # Recorded at launch: the source of a long run is the one it started with. An uninstalled
    # checkout (a Pod's project-run) reads None as the project's version.
    versions = versions_record()
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
            "conditional": settings.conditional,
            **(
                {"secondary_intervals": list(settings.secondary_intervals)}
                if settings.secondary_intervals != ("t",)
                else {}
            ),
            "resamples": settings.resamples,
            "alphas": list(settings.alphas),
            "allowance": settings.allowance,
            "levels_from": None if args.levels_from is None else str(args.levels_from),
            "allowance_from": None if args.allowance_from is None else str(args.allowance_from),
            "constants": constants(),
        },
        "versions": versions,
        **({} if inputs is None else {"inputs": inputs.summary()}),
        **result,
        **verdicts(args.mode, result["scenarios"], selection),
    }
    if inputs is not None:
        shutil.copyfile(inputs.path, out / "inputs.json")
    (out / "summary.json").write_text(json_text(summary))
    (out / "results.md").write_text(markdown(summary))
    write_inventory(out)
    print(f"Wrote {out}")


def run_settings(
    args: argparse.Namespace, selection: dict[tuple[str, str, float], float | None]
) -> sim.RunSettings:
    """The run's settings from the parsed arguments. The decided modes default to the decided
    design (see the module docstring); every other mode to t, the standard rule, every look
    design and SE floor, and α = 0.05 with the selected α's."""
    decided = args.mode in DECIDED_MODES
    alphas = tuple(sorted(args.alphas)) if args.exact_alphas else None
    alphas = alphas or alpha_choice(args.mode, selection, args.alphas)
    b = args.allowance
    if args.allowance_from is not None:
        found = json.loads(args.allowance_from.read_text())["bias_allowance"]["allowance"]
        b = None if found is None else found["value"]
    elif b is None and args.mode in ALLOWANCE_MODES:
        b = sim.DECIDED_ALLOWANCE
    if args.looks:
        designs = tuple(args.looks)
    else:
        designs = (sim.DECIDED_DESIGN,) if decided else tuple(d.name for d in sim.LOOK_DESIGNS)
    intervals = args.intervals or (["bootstrap_t"] if decided else ["t"])
    rules = args.rules or (["standard", sim.DECIDED_RULE] if decided else ["standard"])
    if args.floors:
        floors = tuple(args.floors)
    else:
        floors = (sim.DECIDED_SE_FLOOR,) if decided else sim.SE_FLOORS
    chunk = args.chunk or (GATE_CHUNK if args.mode == "decided-gate" else CHUNK)
    secondary = args.secondary_intervals or (
        ["t", "bootstrap_t"] if args.mode in SECONDARY_MODES else ["t"]
    )
    return sim.RunSettings(
        stream=STREAMS[args.mode],
        alphas=alphas,
        allowance=b,
        base_seed=args.seed,
        workers=args.workers,
        chunk=chunk,
        spread_replicates=args.spread_replicates,
        spread_blocks=args.spread_blocks,
        designs=designs,
        intervals=tuple(dict.fromkeys(intervals)),
        resamples=args.bootstrap,
        rules=tuple(dict.fromkeys(rules)),
        floors=floors,
        prescreen=args.mode == "decided-gate",
        secondary_intervals=tuple(dict.fromkeys(secondary)),
        conditional=args.conditional,
    )


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
    """calibrate: the whole α grid; otherwise α = 0.05 (the decided modes: the decided α), the
    selected α's and any extra ones."""
    if mode == "calibrate":
        return sim.ALPHAS
    base = sim.DECIDED_ALPHA if mode in DECIDED_MODES else sim.PLANNING_ALPHA
    alphas = {base, *(extra or [])}
    alphas |= {alpha for alpha in selection.values() if alpha is not None}
    return tuple(sorted(alphas))


def upper() -> dict:
    """The simulated U with what it sets (``sim.set_upper``)."""
    return {"upper": sim.U, "k_star": sim.K_STAR, "half_span": sim.HALF_SPAN}


def constants() -> dict:
    return {
        **upper(),
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
    secondary = sim.secondary_checks(scenarios)
    if secondary:  # runs with secondary intervals other than t alone
        found["secondary_checks"] = secondary
    conditional = sim.conditional_checks(scenarios)
    if conditional:  # runs with --conditional
        found["conditional_checks"] = conditional
    return found


# Amendment 4: the mechanism cohort, the inputs record, comparisons and the final run


def _version(name: str) -> str | None:
    """A package's installed version; None for one imported from a checkout without being
    installed (the project on a Pod, through ``project-run``), whose source ``git`` records."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def versions_record() -> dict:
    """Recorded at launch: the source of a long run is the one it started with."""
    return {
        "python": platform.python_version(),
        **{name: _version(name) for name in PACKAGES},
        "git": git_state(),
    }


def progress_printer(clock: float):
    def progress(stage: str, done: int, total: int) -> None:
        if done == total or done % max(1, total // 20) == 0:
            minutes = (time.perf_counter() - clock) / 60
            print(f"{stage}: {done}/{total}, {minutes:.1f} min", file=sys.stderr, flush=True)

    return progress


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def write_inventory(out: Path) -> Path:
    """``out``/inventory.tsv: the path (relative to ``out``), size and sha256 of every file under
    ``out`` but this inventory, sorted by path."""
    target = out / INVENTORY
    lines = ["path\tsize\tsha256"]
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p != target):
        rel = path.relative_to(out).as_posix()
        lines.append(f"{rel}\t{path.stat().st_size}\t{file_sha256(path)}")
    target.write_text("\n".join(lines) + "\n")
    return target


def checked_level(path: Path) -> float | None:
    """The level a mechanism run's calibration rule implies (None: no level passed)."""
    return json.loads(path.read_text())["checks"]["level"]


def mechanism_constants() -> dict:
    return {
        **upper(),
        "target": sim.MECHANISM_TARGET,
        "cap": sim.MECHANISM_CAP,
        "minimum": sim.MECHANISM_MINIMUM,
        "comparison_level": sim.COMPARISON_LEVEL,
        "effects": list(sim.MECHANISM_EFFECTS),
        "coverage_target": sim.MECHANISM_COVERAGE,
        "w_max": sim.MECHANISM_W_MAX,
        "component_means": [list(c) for c in sim.COMPONENT_MEANS],
        "sd_ratios": {k: list(v) for k, v in sim.SD_RATIOS.items()},
        "rhos": list(sim.COMPONENT_RHOS),
        "attritions": list(sim.ATTRITIONS),
        "required": {"shapes": list(sim.REQUIRED_SHAPES), "sds": list(sim.REQUIRED_SDS)},
        "stress_sds": list(sim.STRESS_SDS),
        "tied_baselines": list(sim.TIED_BASELINES),
        "estimand_geometries": sim.ESTIMAND_GEOMETRIES,
        "copula_nodes": sim.COPULA_NODES,
        "failure_slope": sim.FAILURE_SLOPE,
        "chunk_blocks": sim.CHUNK_BLOCKS,
        "prompts": sim.PROMPTS,
    }


RESAMPLES_REASON = (
    "run time: the required grid alone is 9.72 million replicates; per replicate on one idle "
    "core the blocks cost ~7 ms and the three bootstrap sets ~1.6 ms at 2 000 resamples against "
    "~14.6 ms at 10 000, so 10 000 would about triple the run on a shared machine; amendment 3's "
    "simulation also used 2 000"
)


def mechanism_main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    which = MECHANISM_MODES[args.mode]
    inputs = sim.read_inputs(args.inputs or INPUTS_RECORD)
    default = ("required", "stress") if which == "calibration" else ("required",)
    groups = tuple(args.groups or default)
    if which == "check":
        if (args.level is None) == (args.level_from is None):
            parser.error("mechanism-check needs exactly one of --level and --level-from")
        level = args.level if args.level is not None else checked_level(args.level_from)
        if level is None:
            parser.error(f"{args.level_from}: no level passed the calibration rule")
        levels = (level,)
    else:
        levels = tuple(sorted(args.levels)) if args.levels else sim.MECHANISM_LEVELS
    grid = sim.mechanism_grid(
        stress_reps=args.stress_reps or sim.MECHANISM_STRESS_REPS, stress="stress" in groups
    )
    grid = [m for m in grid if m.group in groups]
    if args.scenarios:
        grid = [m for m in grid if any(fnmatch.fnmatch(m.name, p) for p in args.scenarios)]
    if args.reps is not None:
        grid = [replace(m, reps=args.reps) for m in grid]
    # The predeclared precision check (review fa24, Q2): the calibration run's borderline
    # scenarios, once, at PRECISION_REPS replicates on the check stream. Only a check from a
    # calibration summary (--level-from) over the whole required grid at its replicates is
    # binding; a borderline scenario missing from the grid is an error, never dropped.
    borderline = []
    precision = args.precision_reps or sim.PRECISION_REPS
    if which == "check" and args.level_from is not None:
        borderline = json.loads(args.level_from.read_text())["checks"].get("borderline") or []
        names = {m.name for m in grid}
        absent = [name for name in borderline if name not in names]
        if absent:
            parser.error(f"borderline scenarios of {args.level_from} not in the grid: {absent}")
        grid = [replace(m, reps=precision) if m.name in borderline else m for m in grid]
    binding = (which == "check" and args.level_from is not None and not args.scenarios
               and args.reps is None and groups == ("required",)
               and precision == sim.PRECISION_REPS)  # fmt: skip
    if not grid:
        parser.error("No scenario selected")
    if args.list:
        for m in grid:
            print(f"{m.name}  reps={m.reps}  group={m.group}")
        return
    settings = sim.MechanismSettings(
        stream=sim.MECHANISM_STREAMS[which],
        inputs=inputs,
        levels=levels,
        resamples=args.bootstrap,
        base_seed=args.seed,
        workers=args.workers,
        chunk=args.chunk or CHUNK,
        spread_replicates=args.spread_replicates,
        spread_blocks=args.spread_blocks,
    )
    versions = versions_record()
    started = datetime.now(UTC)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out = args.out or runs_root() / "analysis" / "study2_short_sim" / f"{args.mode}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    clock = time.perf_counter()
    result = sim.run_mechanism(grid, settings, progress_printer(clock))
    if args.readings_at is not None and args.readings_at not in levels:
        parser.error(f"--readings-at {args.readings_at} is not an evaluated level {levels}")
    checks = sim.mechanism_checks(
        result["scenarios"], args.readings_at, predeclare=which == "calibration"
    )
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
            "levels": list(settings.levels),
            "resamples": settings.resamples,
            "resamples_reason": RESAMPLES_REASON,
            "groups": list(groups),
            "level_from": None if args.level_from is None else str(args.level_from),
            "readings_at": args.readings_at,
            "constants": mechanism_constants(),
        },
        "inputs": inputs.summary(),
        "versions": versions,
        **result,
        "checks": checks,
    }
    if which == "check":
        cells = checks["per_level"][0]["required"]["cells"]
        summary["check"] = {
            "level": levels[0],
            "passes": checks["per_level"][0]["required"]["meets"],
            "binding": binding,
            "from": summary["settings"]["level_from"],
            "borderline": borderline,  # predeclared on the calibration stream
            "precision_reps": precision,
            "precision_floor": sim.coverage_floor(precision),
            "floor": sim.coverage_floor(sim.MECHANISM_REPS),
            "short": [name for c in cells for name in c["short_scenarios"]],
        }
    shutil.copyfile(inputs.path, out / "inputs.json")
    (out / "summary.json").write_text(json_text(summary))
    (out / "results.md").write_text(mechanism_markdown(summary))
    write_inventory(out)
    print(f"Wrote {out}")


def parse_rates(items: list[str] | None) -> dict[str, list[float]]:
    """--rates A=r1,r2,r3 … as {arm: [rates]}."""
    rates: dict[str, list[float]] = {}
    for item in items or ():
        arm, _, values = item.partition("=")
        if arm not in sim.ARMS or not values:
            raise ValueError(f"--rates takes ARM=r1,r2,r3 with ARM one of {sim.ARMS}: {item!r}")
        rates[arm] = [float(v) for v in values.split(",")]
    return rates


def measured_record(args: argparse.Namespace) -> dict:
    """The inputs record measured on the runs under --runs-root (K and support only)."""
    from zero_field_refusal.analysis.stage_a import load_trajectories

    runs = load_trajectories(*args.runs_root)
    measured = sim.measure_inputs(runs, batch=args.batch, seeds=tuple(args.seeds),
                                  rates=parse_rates(args.rates))  # fmt: skip
    seeds = f"{min(args.seeds)}-{max(args.seeds)}"
    where = f"runs under {', '.join(str(p) for p in args.runs_root)}, seeds {seeds}"
    return sim.inputs_record(measured, batch=args.batch, where=where)


def inputs_main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.runs_root or not args.rates or args.out is None:
        parser.error("inputs needs --runs-root, --rates and --out (the record's path)")
    record = measured_record(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    sim.read_inputs(args.out)
    print(f"Wrote {args.out}")


# Settings that carry the inputs, which a comparison across inputs ignores, and rates compared.
INPUT_SETTINGS = ("sigma_u_arms", "sigma_e_arms", "failure_arms", "group", "reps")
COMPARED_RATES = (("p_positive", "se_positive"), ("p_negative", "se_negative"),
                  ("p_equivalence", "se_equivalence"), ("power", "se_power"),
                  ("p_cap_exhausted", "se_cap_exhausted"))  # fmt: skip
COMPARED_GATED = (("p_positive", "se_positive"), ("p_negative", "se_negative"),
                  ("p_equivalence", "se_equivalence"), ("power", "se_power"))  # fmt: skip


def _match_key(settings: dict) -> str:
    return json.dumps({k: v for k, v in settings.items() if k not in INPUT_SETTINGS},
                      sort_keys=True)  # fmt: skip


def _difference(what: str, new: float, new_se: float, old: float, old_se: float) -> dict:
    se = math.hypot(new_se, old_se)
    z = (new - old) / se if se > 0 else (0.0 if new == old else math.inf)
    return {"rate": what, "new": new, "new_se": new_se, "old": old, "old_se": old_se,
            "difference": new - old, "se": se, "z": z, "beyond_2se": abs(z) > 2}  # fmt: skip


def compare_runs(new: dict, old: list[dict]) -> dict:
    """Per scenario matched on every setting but the per-arm inputs (and replicates), and per
    result row (look design, interval, rule, α, floor): each decision rate of the rerun against
    the archived one, the gated rates where both have them, and the block bias and gate failure;
    the difference, its MC SE √(SE_new² + SE_old²) (independent streams), z, and whether it is
    beyond two SEs. With many rates some exceed two SEs by chance (about 4.6 % of them)."""
    archived = {_match_key(s["settings"]): s for summary in old for s in summary["scenarios"]}
    entries, unmatched = [], []
    for scenario in new["scenarios"]:
        match = archived.get(_match_key(scenario["settings"]))
        if match is None:
            unmatched.append(scenario["name"])
            continue
        rows = {(r["looks"], r["interval"], r["rule"], r["alpha"], r["floor"]): r
                for r in match["results"]}  # fmt: skip
        found = []
        for row in scenario["results"]:
            key = (row["looks"], row["interval"], row["rule"], row["alpha"], row["floor"])
            if key not in rows:
                continue
            past = rows[key]
            label = f"{row['looks']} {row['interval']} {row['rule']} α {row['alpha']:g}"
            for rate, se in COMPARED_RATES:
                found.append({"row": label, **_difference(rate, row[rate], row[se], past[rate],
                                                          past[se])})  # fmt: skip
            new_gated, old_gated = row.get("gated") or {}, past.get("gated") or {}
            for rate, se in COMPARED_GATED:
                if rate in new_gated and rate in old_gated:
                    found.append({"row": label, **_difference(
                        f"gated {rate}", new_gated[rate], new_gated[se], old_gated[rate],
                        old_gated[se])})  # fmt: skip
        d_new, d_old = scenario["diagnostics"], match["diagnostics"]
        found.append({"row": "diagnostics", **_difference(
            "block_bias", d_new["block_bias"], d_new["block_bias_se"], d_old["block_bias"],
            d_old["block_bias_se"])})  # fmt: skip
        if "p_gate_fail" in d_new and "p_gate_fail" in d_old:
            found.append({"row": "diagnostics", **_difference(
                "p_gate_fail", d_new["p_gate_fail"], d_new["se_gate_fail"], d_old["p_gate_fail"],
                d_old["se_gate_fail"])})  # fmt: skip
        entries.append({"scenario": scenario["name"], "archived": match["name"],
                        "reps": [scenario["reps"], match["reps"]], "rates": found})  # fmt: skip
    rates = [r for e in entries for r in e["rates"]]
    flagged = [{"scenario": e["scenario"], **r} for e in entries for r in e["rates"]
               if r["beyond_2se"]]  # fmt: skip
    finite = [abs(r["z"]) for r in rates if math.isfinite(r["z"])]
    return {
        "compared": len(rates),
        "beyond_2se": len(flagged),
        "expected_by_chance": round(0.0455 * len(rates), 1),
        "max_abs_z": max(finite, default=None),
        "flagged": flagged,
        "unmatched": unmatched,
        "scenarios": entries,
    }


def compare_main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.new is None or not args.old:
        parser.error("compare needs --new and --old")
    comparison = compare_runs(json.loads(args.new.read_text()),
                              [json.loads(p.read_text()) for p in args.old])  # fmt: skip
    comparison = {"new": str(args.new), "new_sha256": file_sha256(args.new),
                  "old": [{"path": str(p), "sha256": file_sha256(p)} for p in args.old],
                  **comparison}  # fmt: skip
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or runs_root() / "analysis" / "study2_short_sim" / f"compare-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json_text(comparison))
    (out / "comparison.md").write_text(comparison_markdown(comparison))
    write_inventory(out)
    print(f"Wrote {out}: {comparison['beyond_2se']} of {comparison['compared']} rates beyond 2 SE")


def write_fixation_level(path: Path, level: float) -> None:
    """Set the fixation record's ``mechanism.level``, leaving every other byte as it is, and
    check the result with the analysis's reader."""
    from zero_field_refusal.analysis.study2_short import read_fixation

    text = path.read_text()
    pattern = re.compile(r'("mechanism":\s*\{[^{}]*?"level":\s*)(-?[0-9.eE+-]+)')
    updated, count = pattern.subn(lambda m: m.group(1) + json.dumps(level), text)
    if count != 1:
        raise ValueError(f"{path}: no single mechanism level to set")
    before, after = json.loads(text), json.loads(updated)
    before["mechanism"]["level"] = level
    if before != after:
        raise ValueError(f"{path}: setting the level changed something else")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(updated)
    read_fixation(temporary)
    temporary.replace(path)


def final_main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """The level-fixing run after S1 (module docstring, "The final run after S1")."""
    if bool(args.runs_root) != bool(args.rates):
        parser.error("final measures the inputs with both --runs-root and --rates, or neither")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or runs_root() / "analysis" / "study2_short_sim" / f"final-{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    record = out / "inputs.json"
    if args.runs_root:
        record.write_text(json.dumps(measured_record(args), indent=2) + "\n")
    else:
        shutil.copyfile(args.inputs or INPUTS_RECORD, record)
    inputs = sim.read_inputs(record)
    common = ["--inputs", str(record), "--workers", str(args.workers), "--seed", str(args.seed)]
    if args.reps is not None:
        common += ["--reps", str(args.reps)]
    tuning = ["--bootstrap", str(args.bootstrap), "--spread-replicates",
              str(args.spread_replicates), "--spread-blocks", str(args.spread_blocks)]  # fmt: skip
    mechanism = [*common, *tuning]
    if args.scenarios:
        mechanism += ["--scenarios", *args.scenarios]
    # What binds first: the required grid, its check, the primary rerun; the stress grid last.
    main(["mechanism", *mechanism, "--groups", "required", "--out", str(out / "mechanism")])
    calibration = out / "mechanism" / "summary.json"
    level = checked_level(calibration)
    check = None
    if level is not None:
        # One check at the chosen level, 0.95 included; the borderline scenarios at
        # PRECISION_REPS; no repetition (review fa24, Q2).
        precision = (
            [] if args.precision_reps is None else ["--precision-reps", str(args.precision_reps)]
        )
        main(["mechanism-check", *mechanism, *precision, "--level-from", str(calibration),
              "--out", str(out / "check")])  # fmt: skip
        check = json.loads((out / "check" / "summary.json").read_text())["check"]
    fixed = (level if level is not None and check is not None and check["passes"]
             and check.get("binding", False) else None)  # fmt: skip
    comparisons = {}
    for mode, name in FINAL_PRIMARY.items():
        primary = [*common, "--spread-replicates", str(args.spread_replicates),
                   "--spread-blocks", str(args.spread_blocks)]  # fmt: skip
        if mode == "decided":  # the recheck against the conditional truth of Δ (fa24, Q1)
            primary.append("--conditional")
        main([mode, *primary, "--out", str(out / name)])
        archived = runs_root() / ARCHIVED[mode]
        if archived.is_file():
            target = out / f"compare-{name}"
            main(["compare", "--new", str(out / name / "summary.json"), "--old", str(archived),
                  "--out", str(target)])  # fmt: skip
            found = json.loads((target / "comparison.json").read_text())
            comparisons[name] = {
                k: found[k] for k in ("compared", "beyond_2se", "expected_by_chance", "max_abs_z")
            }
        else:
            comparisons[name] = {"archived": str(archived), "missing": True}
    stress = [*mechanism, "--groups", "stress", "--out", str(out / "stress")]
    if fixed is not None:  # the stress readings at the fixed level
        stress += ["--readings-at", str(fixed)]
    if args.stress_reps is not None:
        stress += ["--stress-reps", str(args.stress_reps)]
    names = [m.name for m in sim.mechanism_grid() if m.group == "stress"]
    if not args.scenarios or any(fnmatch.fnmatch(n, p) for n in names for p in args.scenarios):
        main(["mechanism", *stress])
    final = {
        "inputs": inputs.summary(),
        "level_on_calibration_stream": level,
        "check": check,
        "fixed_level": fixed,
        "rule": "a level is fixed only if it holds on the calibration stream (the smallest level "
        "on the grid at which every required scenario meets its floor) and passes the single "
        "independent check at it on the check stream, 0.95 included (the predeclared borderline "
        "scenarios at 50 000 replicates); otherwise no level is fixed",
        "comparisons": comparisons,
        "written": [],
    }
    if fixed is None:
        final["warning"] = (
            "no mechanism level is fixed: docs/study2-short-fixation.json keeps its placeholder "
            "mechanism.level 0.95, which must not be used"
        )
        print(f"WARNING: {final['warning']}", file=sys.stderr, flush=True)
    if args.write and fixed is not None:  # the record and its level go in together, or nothing
        if args.runs_root:
            shutil.copyfile(record, INPUTS_RECORD)
            final["written"].append(str(INPUTS_RECORD))
        write_fixation_level(FIXATION_RECORD, fixed)
        final["written"].append(f"{FIXATION_RECORD}: mechanism.level {fixed}")
    (out / "final.json").write_text(json_text(final))
    write_inventory(out)
    print(f"Wrote {out}: fixed level {fixed}")


def _rate_cell(item: dict | None, digits: int = 2) -> str:
    if not item:
        return "–"
    return f"{pp(item['value'], digits)} ({pp(item['se'], digits)})"


def mechanism_markdown(summary: dict) -> str:
    settings, checks, inputs = summary["settings"], summary["checks"], summary["inputs"]
    shown = checks["readings_at"]
    levels = settings["levels"]
    index = levels.index(shown) if shown in levels else 0
    if settings.get("readings_at") is not None:
        readings = f"at the fixed level {shown}"
    elif checks["level"] is not None and "check" not in summary:
        readings = f"at the level the rule implies on this stream, {shown}"
    else:
        readings = f"at {shown} (no level fixed on this run)"
    coverage = (
        "Simultaneous max-t coverage of both components' conditional truths (each replicate's: "
        "fresh supported seed blocks given its prompt panel and misses), against "
        "0.95 − 2·√(0.95·0.05/R) per scenario at its own R. A missing interval does not cover; a "
        "whole-line one (q = ∞) is uninformative, counted as covering as the brief rules and "
        "reported separately; the analysis itself returns no interval for it. Required: normal "
        "and skewed at SD 5 and 10 pp. At exactly 95 % coverage the expected number below the "
        "floor is 2.3 % of a cell's scenarios and the chance of at least one is nearly 1 (in "
        "brackets)."
    )
    if "check" in summary:
        rule = ["## The check", "", check_text(summary["check"]), "", coverage, "",
                rule_table(checks)]  # fmt: skip
    else:
        borderline = checks.get("borderline") or []
        rule = ["## Calibration rule", "", coverage + f" Level the rule implies: "
                f"**{checks['level']}**; predeclared borderline scenarios there (under 95 %, "
                f"checked once at {sim.PRECISION_REPS} on the check stream): {len(borderline)}.",
                "", rule_table(checks)]  # fmt: skip
    parts = [
        f"# Study 2 mechanism simulation: {summary['mode']}",
        "",
        f"Started {summary['started']}, {summary['runtime_s'] / 60:.1f} min with "
        f"{settings['workers']} workers, stream `{settings['stream']}`, base seed "
        f"{settings['base_seed']}, {settings['resamples']} bootstrap resamples per replicate and "
        f"interval. Inputs `{inputs['path']}` (sha256 {inputs['sha256']}, {inputs['status']}). "
        "Rates in %, MC SE in brackets; SDs in pp.",
        "",
        "## Inputs",
        "",
        inputs_table(summary),
        "",
        *rule,
        "",
        f"## Readings {readings}",
        "",
        "Direction power at |C| = 5 and 2.5 pp at the declared sign and at the conditional "
        "truth's (smallest / mean over scenarios and components with that component SD; "
        "unconditional, over every replicate, a reading or not), the largest false direction "
        "against the conditional truth (at C = 0 and anywhere), the bounded rate at C = 0, the "
        "largest false bounded rate at |C| = 5 pp, and per cell the smallest reach of the "
        "36-block minimum, the largest shares ending at the cap and under 36, the smallest "
        "shares of usable intervals (both finite) and of readable ones (finite with at least 36 "
        "blocks: a reading may be made), and the largest of missing and whole-line ones. Below "
        "36 blocks the intervals are descriptive only: with 20 % failures in every arm most "
        "cohorts end near 27 blocks, so those scenarios' coverage comes from descriptive "
        "intervals (the scenario table splits it).",
        "",
        readings_table(checks),
        "",
        f"## Scenarios {readings}",
        "",
        scenario_table(summary["scenarios"], index),
    ]
    return "\n".join(parts) + "\n"


def check_text(check: dict) -> str:
    """The single check (review fa24, Q2) as its record has it: the level, whether it binds,
    the borderline set predeclared on the calibration stream with its replicates and floor, the
    floor of the others, the verdict and the scenarios short of their floor."""
    borderline = check.get("borderline") or []
    short = check.get("short") or []
    binds = "binding" if check.get("binding") else "not binding (a partial or level-only run)"
    return (
        f"Check of level **{check['level']}** on the separate check stream, from "
        f"`{check.get('from')}`; {binds}. Predeclared borderline scenarios (under 95 % on the "
        f"calibration stream): {len(borderline)}, each at {check['precision_reps']} replicates "
        f"against {pp(check['precision_floor'], 3)} %; the others at {sim.MECHANISM_REPS} against "
        f"{pp(check.get('floor', sim.coverage_floor(sim.MECHANISM_REPS)), 3)} %. Verdict: "
        f"**{'passes' if check['passes'] else 'fails'}**; scenarios short of their floor: "
        f"{len(short)}" + (": " + ", ".join(short) if short else ".")
    )


def inputs_table(summary: dict) -> str:
    inputs = summary["inputs"]
    simulated = {}
    for s in summary["scenarios"]:
        if s["settings"]["attrition"] == "none" and (s["parameters"]["estimand"] or {}).get(
            "support"
        ):
            simulated = s["parameters"]["estimand"]["support"]
            break
    rows = [
        [arm, f"{inputs['sigma_u'][arm]:.4f}", f"{inputs['sigma_e'][arm]:.4f}",
         pp(inputs["support_rate"][arm]), pp(simulated.get(arm)), pp(inputs["failure_rate"][arm])]
        for arm in sim.ARMS
    ]  # fmt: skip
    text = table(["arm", "σ_u", "σ_e", "support (record)", "support (simulated, no attrition)",
                  "failure rate"], rows)  # fmt: skip
    if inputs["provisional"]:
        text += "\n\nProvisional: " + "; ".join(inputs["provisional"]) + "."
    return text


def rule_table(checks: dict) -> str:
    rows = []
    for entry in checks["per_level"]:
        for group in ("required", "stress"):
            for c in entry[group]["cells"]:
                rows.append([
                    f"{entry['level']:.3f}", group, c["shape"],
                    "–" if c["shape"] in ("rare_only", "tied") else pp(c["sd"], 0),
                    f"{pp(c['min_coverage'])} ({pp(c['mc_se'])})", pp(c.get("mean_coverage")),
                    pp(c["floor"]), "yes" if c["meets"] else "**no**",
                    f"{c['short']}/{c['scenarios']} ({c.get('short_by_chance', '–')}; "
                    f"{pp(c.get('any_short_by_chance'), 1)} %)",
                    c["scenario"],
                ])  # fmt: skip
    header = ["level", "group", "shape", "SD", "min coverage", "mean", "floor", "meets",
              "short (expected at 95 %; P(any))", "worst scenario"]  # fmt: skip
    return table(header, rows)


def readings_table(checks: dict) -> str:
    rows = []
    for group, entries in checks["readings"].items():
        for e in entries:

            def power(key: str, e: dict = e) -> str:
                item = e.get(key)
                return "–" if not item else f"{pp(item['min'], 1)} / {pp(item['mean'], 1)}"

            bounded = e.get("bounded_at_0")
            rows.append([
                group, e["shape"], "–" if e["shape"] in ("rare_only", "tied") else pp(e["sd"], 0),
                "larger" if e["component_sd"] == 1.0 else "half",
                power("power_5pp"), power("power_5pp_conditional"),
                power("power_2.5pp"), power("power_2.5pp_conditional"),
                f"{pp(e['max_false_direction_at_0'])} / {pp(e['max_false_direction'])}",
                "–" if not bounded else f"{pp(bounded['min'], 1)} / {pp(bounded['mean'], 1)}",
                pp(e.get("max_false_bounded_at_5pp")),
                pp(e["min_minimum_met"]), pp(e["max_ended_at_cap"]), pp(e["max_under_minimum"]),
                pp(e["min_usable"]), pp(e.get("min_readable")), pp(e["max_missing"]),
                pp(e["max_whole_line"]),
            ])  # fmt: skip
    header = ["group", "shape", "SD", "component SD", "power @5", "power @5 cond", "power @2.5",
              "power @2.5 cond", "max false dir @0 / any", "bounded @0", "max false bounded @5",
              "min reach 36", "max at cap", "max under 36", "min usable", "min readable",
              "max missing", "max whole line"]  # fmt: skip
    return table(header, rows)


def scenario_table(scenarios: list[dict], index: int) -> str:
    rows = []
    for s in scenarios:
        at, first = s["levels"][index], s["levels"][0]
        comps = at["components"]
        check = s["estimand_check"]
        rows.append([
            s["name"], s["settings"]["group"], _rate_cell(first["simultaneous"]),
            _rate_cell(at["simultaneous"]),
            pp(comps["C1"]["coverage"]["value"]), pp(comps["C2"]["coverage"]["value"]),
            pp(s["bonferroni"]["C1"]["value"]), pp(s["bonferroni"]["C2"]["value"]),
            " / ".join(pp((comps[c]["power"] or {}).get("value"), 1) for c in sim.COMPONENTS),
            " / ".join(pp(comps[c]["bounded"]["value"], 1) for c in sim.COMPONENTS),
            pp(s["minimum_met"]["value"], 1), pp(at["infinite_q"]["value"]),
            f"{pp(at['missing']['value'])} / {pp(at['whole_line']['value'])} / "
            f"{pp(at['usable']['value'])} / {pp(at['readable']['value'])}",
            " / ".join("–" if at[k]["value"] is None else
                       f"{pp(at[k]['value'])} ({pp(at[k]['share'], 0)})"
                       for k in ("simultaneous_readable", "simultaneous_descriptive")),
            _rate_cell(at["simultaneous_population"]),
            " / ".join(f"{pp(s['conditional_truth'][c]['mean'], 2)}±"
                       f"{pp(s['conditional_truth'][c]['sd'], 2)}" for c in sim.COMPONENTS),
            " / ".join(pp(s["zero_se"][c]["value"], 1) for c in sim.COMPONENTS),
            " / ".join(pp(s["attained_sd"][c], 1) for c in sim.COMPONENTS),
            "–" if s["mean_correlation"] is None else f"{s['mean_correlation']:+.2f}",
            " / ".join("–" if check[c]["z"] is None else f"{check[c]['z']:+.1f}"
                       for c in sim.COMPONENTS),
            " / ".join("–" if check[c].get("conditional_z") is None else
                       f"{check[c]['conditional_z']:+.1f}" for c in sim.COMPONENTS),
            str(s["reps"]),
        ])  # fmt: skip
    header = ["scenario", "group", "sim cov @0.95", "sim cov", "C1 cov", "C2 cov", "Bonf C1",
              "Bonf C2", "power C1/C2", "bounded C1/C2", "reach 36", "q = ∞",
              "missing / whole line / usable / readable",
              "sim cov readable / descriptive (share)", "sim cov (population)",
              "conditional truth C1/C2 (mean ± SD, pp)", "zero SE C1/C2", "SD C1/C2", "corr",
              "population z C1/C2", "conditional z C1/C2", "reps"]  # fmt: skip
    return table(header, rows)


def comparison_markdown(comparison: dict) -> str:
    rows = [
        [f["scenario"], f["row"], f["rate"], pp(f["new"]), pp(f["old"]), pp(f["difference"]),
         pp(f["se"]), f"{f['z']:+.2f}"]
        for f in comparison["flagged"]
    ]  # fmt: skip
    parts = [
        "# Primary rerun against the archived run",
        "",
        f"New `{comparison['new']}`; archived "
        + ", ".join(f"`{o['path']}`" for o in comparison["old"])
        + f". {comparison['compared']} rates compared, {comparison['beyond_2se']} beyond two "
        f"MC SEs of the difference (about {comparison['expected_by_chance']} expected by chance "
        f"at independent streams); largest |z| {comparison['max_abs_z']}. Rates in %.",
        "",
        table(["scenario", "row", "rate", "new", "archived", "difference", "SE", "z"], rows)
        if rows
        else "No rate differs by more than two MC SEs.",
    ]
    if comparison["unmatched"]:
        parts += ["", "Unmatched: " + ", ".join(comparison["unmatched"])]
    return "\n".join(parts) + "\n"


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
    if "secondary_checks" in summary:
        parts += [
            "",
            "## Secondary coverage",
            "",
            "Coverage of the secondary contrasts' pointwise truths at the look where the primary "
            f"stopped, {100 * sim.SECONDARY_LEVEL:.2f} % intervals per contrast and look; the "
            f"worst over Δ per shape × SD (Δ of the worst in brackets). Targets "
            f"{100 * sim.SECONDARY_MARGINAL_TARGET:.1f} % per contrast and "
            f"{100 * sim.SECONDARY_JOINT_TARGET:.1f} % jointly; * marks a value below its "
            "target, ** one more than 2 MC SE below it.",
            "",
            secondary_checks_table(summary["secondary_checks"]),
            "",
            "Per scenario (coverage of the truth; A−P also of its block mean):",
            "",
            secondary_results_table(summary),
        ]
    if "conditional_checks" in summary:
        parts += [
            "",
            "## Against the conditional truth",
            "",
            "Each replicate's truth is the expected interpolated contrast over fresh supported "
            "seed blocks, conditional on its 521-prompt panel and selection misses (review fa24, "
            "Q1). Conditional rate / population rate; coverage of the stopping look's interval "
            "(a look without an interval does not cover). Reported only: no fixed level changes.",
            "",
            conditional_checks_table(summary["conditional_checks"]),
            "",
            conditional_results_table(summary),
        ]
    parts += ["", "## Results", "", results_table(summary)]
    parts += ["", "## Block diagnostics", "", diagnostics_table(summary["scenarios"])]
    return "\n".join(parts) + "\n"


def conditional_checks_table(checks: list[dict]) -> str:
    rows = []
    for check in checks:
        for group, g in check["groups"].items():
            p = g["population"]
            point = g.get("pointwise") or {}
            rows.append([
                check["looks"], check["rule"], f"{check['alpha']:.4f}", group,
                f"{_value(g, 'max_false_positive')} / {_value(p, 'max_false_positive')}",
                f"{_value(g, 'max_false_negative')} / {_value(p, 'max_false_negative')}",
                f"{_value(g, 'max_false_equivalence_5pp')} / "
                f"{_value(p, 'max_false_equivalence_5pp')}",
                _value(g, "min_coverage"), _value(g, "max_missing"),
                " / ".join(_value(point, k) for k in ("max_false_positive", "max_false_negative",
                                                      "max_false_equivalence_5pp")),
                "yes" if g["meets"] else "**no**",
            ])  # fmt: skip
    header = ["looks", "rule", "α", "group", "max dir+ @0", "max dir− @0", "max eq @5",
              "min coverage", "max no interval", "pointwise dir+/dir−/eq", "meets"]  # fmt: skip
    return table(header, rows)


def conditional_results_table(summary: dict) -> str:
    rows = []
    for s in summary["scenarios"]:
        for r in s["results"]:
            c = r.get("conditional")
            if not c:
                continue
            point = c.get("pointwise") or {}
            rows.append([
                s["name"], r["rule"], f"{pp(c['truth_mean'], 3)} ± {pp(c['truth_sd'], 3)}",
                _rate_cell(c["coverage"]), pp(c["missing"]["value"]),
                f"{pp(c['false_positive']['value'])} / {pp(r['p_positive'])}",
                f"{pp(c['false_negative']['value'])} / {pp(r['p_negative'])}",
                f"{pp(c['false_equivalence']['value'])} / {pp(r['p_equivalence'])}",
                pp(c["right_direction"]["value"]),
                " / ".join(pp((point.get(k) or {}).get("value")) for k in
                           ("false_positive", "false_negative", "false_equivalence")),
            ])  # fmt: skip
    header = ["scenario", "rule", "truth mean ± SD (pp)", "coverage", "no interval",
              "dir+ cond / any", "dir− cond / any", "eq cond / any", "right direction",
              "pointwise dir+/dir−/eq"]  # fmt: skip
    return table(header, rows)


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


def _worst_cell(item: dict) -> str:
    mark = "**" if item["short_2se"] else "*" if not item["meets"] else ""
    return f"{pp(item['value'])}{mark} ({pp(item['delta'], 0)})"


def secondary_checks_table(checks: list[dict]) -> str:
    """Per primary design × rule and shape × SD, the worst coverage over Δ of each secondary
    interval side by side."""
    names = list(dict.fromkeys(c["secondary_interval"] for c in checks))
    primary = list(dict.fromkeys(
        (c["looks"], c["interval"], c["rule"], c["alpha"], c["floor"]) for c in checks
    ))  # fmt: skip
    rows = []
    for key in primary:
        mine = {
            c["secondary_interval"]: c
            for c in checks
            if (c["looks"], c["interval"], c["rule"], c["alpha"], c["floor"]) == key
        }
        cells = {name: {(c["shape"], c["sd"]): c for c in mine[name]["cells"]} for name in mine}
        for shape, sd in cells[names[0]]:
            row = [key[0], key[1], key[2], f"{key[3]:.4f}", shape, "–" if shape == "rare_only"
                   else pp(sd, 0)]  # fmt: skip
            for name in names:
                cell = cells[name][(shape, sd)]
                row += [_worst_cell(cell[which]) for which in (*sim.SECONDARY_CONTRASTS, "joint")]
            rows.append(row)
    header = ["looks", "interval", "rule", "α", "shape", "SD"]
    for name in names:
        header += [f"{name} P−F", f"{name} A−P", f"{name} joint"]
    return table(header, rows)


def secondary_results_table(summary: dict) -> str:
    rows = []
    for s in summary["scenarios"]:
        for r in s["results"]:
            for name, c in (r.get("secondary_coverage_by_interval") or {}).items():
                rows.append(
                    [
                        s["name"],
                        r["rule"],
                        name,
                        f"{pp(c['P-F']['truth'])} ({pp(c['P-F']['truth_se'])})",
                        f"{pp(c['A-P']['truth'])} ({pp(c['A-P']['truth_se'])})",
                        f"{pp(c['joint']['truth'])} ({pp(c['joint']['truth_se'])})",
                        pp(c["A-P"]["block_mean"]),
                        pp(r["p_look1_stop"]),
                        f"{r['mean_three_arm_blocks']:.1f}",
                        str(s["reps"]),
                    ]
                )
    header = ["scenario", "rule", "secondary", "P−F", "A−P", "joint", "A−P (block mean)",
              "look-1 stop", "E 3-arm blocks", "reps"]  # fmt: skip
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
