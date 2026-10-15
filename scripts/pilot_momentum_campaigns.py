"""Write the campaigns of study 2's SGD-momentum pilot, the data-dependent ones from its records.

    python scripts/pilot_momentum_campaigns.py screen --out FILE
    python scripts/pilot_momentum_campaigns.py search --runs DIR... [--theta0 RUN] --seed S \
        --out FILE
    python scripts/pilot_momentum_campaigns.py controls --runs DIR... --theta0 RUN --out FILE

The rules are docs/study2-pilot-momentum.md's; nothing is chosen by hand. ``screen``: the
100-step soft-target controls with momentum 0.9 of every rate of the pilot's grid at both batches
on seed 2000. ``search``: this seed's runs of the rates the pilot's downward search needs next
(its first step, the largest candidate of each batch, needs only the screen); ``--theta0`` (the
θ₀ run) once long runs exist. ``controls``: one full-length control on seed 2000 for each rate
whose three seeds have run and whose control is missing, as long as the longest of those runs
and evaluated at seed 2000's evaluation steps. A campaign holds one seed, since each seed's
stream has one writer, its Pod. The inputs are checked with the pilot's own checks
(analysis.pilot_momentum), and anything not whole is refused. FILE must not exist; write it under
an ignored directory such as $RUNS_DIR/campaigns/.
"""

import argparse
import json
from pathlib import Path

import yaml

from zero_field_refusal.analysis.pilot_momentum import (
    ARM,
    CONTROL_SEED,
    CONTROLS,
    GRID,
    MAIN,
    ROLES,
    progress,
)
from zero_field_refusal.analysis.stage_a import (
    BATCHES,
    DISCOVERY_SEEDS,
    SCREEN_STEPS,
    control_schedule,
    load_trajectories,
)


def cell(name: str, config: str, overrides: list[str], seeds: list[int]) -> dict:
    return {"name": name, "config": config, "overrides": overrides, "seeds": seeds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=("screen", "search", "controls"))
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--theta0", type=Path, help="the θ₀ run directory (summary.json)")
    parser.add_argument("--seed", type=int, choices=DISCOVERY_SEEDS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists")
    if args.kind == "search" and args.seed is None:
        parser.error("search needs --seed")
    runs = load_trajectories(*args.runs) if args.runs else []
    theta0 = None
    if args.theta0 is not None:
        summary = json.loads((args.theta0 / "summary.json").read_text())
        theta0 = {key: summary[key] for key in ("benchmark_accuracy", "factual_accuracy")}
    by_role = {role: [r for r in runs if r.role == role] for role in ROLES}

    def state() -> dict:
        try:
            found, problems = progress(by_role["screen"], by_role["discovery"],
                                       by_role["full_control"], theta0)  # fmt: skip
        except ValueError as exc:
            parser.error(str(exc))
        if problems:
            parser.error("the pilot's records are not whole: " + "; ".join(problems))
        return found

    if args.kind == "screen":
        name = "study2-pilot-momentum-screen"
        items = [
            cell(
                f"screen-b{b}-{lr:g}",
                CONTROLS,
                [
                    f"train.batch_size={b}",
                    f"optimizer.lr={lr:g}",
                    f"train.steps={SCREEN_STEPS}",
                    f"schedule.evaluate_at=[{SCREEN_STEPS}]",
                    "stage_a.role=screen",
                ],
                [CONTROL_SEED],
            )
            for b in BATCHES
            for lr in GRID
        ]
    elif args.kind == "search":
        manifest = state()["manifest"]
        name, items = f"study2-pilot-momentum-search-s{args.seed}", []
        for b in (20, 5):  # B = 20 first: the run that needs the most GPU memory shows it early
            pending = manifest["searches"][f"{ARM}/B={b}"]["pending"] or {}
            if args.seed in pending.get("seeds", []):
                lr = pending["rate"]
                overrides = [f"train.batch_size={b}", f"optimizer.lr={lr:g}"]
                items.append(cell(f"{ARM}-b{b}-{lr:g}", MAIN, overrides, [args.seed]))
        if not items:
            waiting = {k: e["pending"] for k, e in manifest["searches"].items() if e["pending"]}
            parser.error(f"the search needs no run of seed {args.seed} now; waiting: {waiting}")
    else:  # controls
        found = state()
        name, items = "study2-pilot-momentum-controls", []
        for entry in found["manifest"]["searches"].values():
            if "control" not in (entry["pending"] or {}):
                continue
            b, lr = entry["batch"], entry["pending"]["rate"]
            group = [found["present"][(ARM, b, lr, s)] for s in DISCOVERY_SEEDS]
            plan = control_schedule(group)  # the longest over the three seeds
            at = plan["seeds"][CONTROL_SEED]
            overrides = [
                f"train.batch_size={b}",
                f"optimizer.lr={lr:g}",
                f"train.steps={plan['steps']}",
                f"schedule.evaluate_at=[{','.join(map(str, at))}]",
            ]
            items.append(cell(f"control-b{b}-{lr:g}", CONTROLS, overrides, [CONTROL_SEED]))
        if not items:
            parser.error("no rate of the pilot's search waits for its full control")
    header = f"# Written by scripts/pilot_momentum_campaigns.py {args.kind}; do not edit.\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + yaml.safe_dump({"name": name, "cells": items}, sort_keys=False))
    print(args.out)


if __name__ == "__main__":
    main()
