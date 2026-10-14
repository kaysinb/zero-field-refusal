"""Write the campaigns of study 2's Stage A, the data-dependent ones from the run records.

    python scripts/stage_a_campaigns.py setup --out FILE              # θ₀ evaluation, seed 2000
    python scripts/stage_a_campaigns.py screen --out FILE             # 10 SGD rates × 2 batches
    python scripts/stage_a_campaigns.py grids --seed S --out FILE     # AdamW-long, 25-step grids
    python scripts/stage_a_campaigns.py sgd --runs DIR... --seed S --out FILE
    python scripts/stage_a_campaigns.py extension --runs DIR... --seed S --out FILE
    python scripts/stage_a_campaigns.py controls --runs DIR... --out FILE

A campaign holds one seed, since each seed's stream has one writer, its Pod. ``sgd`` takes the two
candidates per batch of the screen (analysis.stage_a.screen_candidates, refused on an incomplete
screen); ``extension`` the next rates of the 25-step grids at batch 5 that the runs of all
discovery seeds call for; ``controls`` one full-length control per candidate on seed 2000, as long
as the longest regular run of its rate and batch and evaluated at that seed's evaluation steps.
The rules are those of docs/study2-preregistration.md; nothing is chosen by hand. FILE must not
exist.
"""

import argparse
from pathlib import Path

import yaml

from zero_field_refusal.analysis.stage_a import (
    ADAMW_LONG_GRID,
    BATCHES,
    CONTROL_SEED,
    DISCOVERY_SEEDS,
    SCREEN_STEPS,
    SGD_GRID,
    SHORT_ARMS,
    SHORT_GRID,
    cells,
    control_schedule,
    load_trajectories,
    screen_candidates,
    short_grid_extension,
)

MAIN, CONTROLS = "experiments/study2_stage_a", "experiments/study2_controls"
SHORT = {
    "adamw_25": ["schedule=short"],
    "frozen_adam_25": ["schedule=short", "optimizer=adam_frozen"],
}


def cell(name: str, config: str, overrides: list[str], seeds: list[int], **extra) -> dict:
    return {"name": name, "config": config, "overrides": overrides, "seeds": seeds, **extra}


def long_cells(arm: str, rates, seed: int) -> list[dict]:
    optimizer = ["optimizer=sgd"] if arm == "sgd_long" else []
    return [
        cell(
            f"{arm}-b{b}-{lr:g}",
            MAIN,
            [*optimizer, f"train.batch_size={b}", f"optimizer.lr={lr:g}"],
            [seed],
        )
        for b in BATCHES
        for lr in rates
    ]


def short_cells(arm: str, batch: int, rates, seed: int) -> list[dict]:
    return [
        cell(
            f"{arm}-b{batch}-{lr:g}",
            MAIN,
            [*SHORT[arm], f"train.batch_size={batch}", f"optimizer.lr={lr:g}"],
            [seed],
        )
        for lr in rates
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "kind", choices=("setup", "screen", "grids", "sgd", "extension", "controls")
    )
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--seed", type=int, choices=DISCOVERY_SEEDS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists")
    if args.kind in ("grids", "sgd", "extension") and args.seed is None:
        parser.error(f"{args.kind} needs --seed")
    runs = load_trajectories(*args.runs) if args.runs else []

    if args.kind == "setup":
        name, items = "study2-a-setup", [cell("theta0", MAIN, [], [CONTROL_SEED], entry="theta0")]
    elif args.kind == "screen":
        name = "study2-a-screen"
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
            for lr in SGD_GRID
        ]
    elif args.kind == "grids":
        name = f"study2-a-grids-s{args.seed}"
        items = long_cells("adamw_long", ADAMW_LONG_GRID, args.seed)
        for arm in SHORT_ARMS:
            for b in BATCHES:
                items += short_cells(arm, b, SHORT_GRID, args.seed)
    elif args.kind == "sgd":
        screen = [r for r in runs if r.role == "screen"]
        result, problems = screen_candidates(screen)
        if problems:
            parser.error("the screen is incomplete: " + "; ".join(problems))
        name, items = f"study2-a-sgd-s{args.seed}", []
        for b in BATCHES:
            items += [
                c
                for c in long_cells("sgd_long", result[b]["candidates"], args.seed)
                if c["name"].startswith(f"sgd_long-b{b}-")
            ]
        if not items:
            parser.error("no SGD rate passes the screen at either batch: numerics fail")
    elif args.kind == "extension":
        name, items = f"study2-a-extension-s{args.seed}", []
        for arm in SHORT_ARMS:
            present = [
                r for r in runs if r.role == "discovery" and r.arm == arm and r.batch_size == 5
            ]
            nxt = short_grid_extension(present, arm)
            if nxt is not None:
                items += short_cells(arm, 5, [nxt], args.seed)
        if not items:
            parser.error("no extension is called for")
    else:  # controls
        name, items = "study2-a-controls", []
        for (arm, b, lr), group in cells(r for r in runs if r.role == "discovery").items():
            if arm != "sgd_long":
                continue
            plan = control_schedule(group)
            at = plan["seeds"].get(CONTROL_SEED)
            if at is None:
                parser.error(f"no regular run of seed {CONTROL_SEED} at batch {b}, rate {lr:g}")
            items.append(
                cell(
                    f"control-b{b}-{lr:g}",
                    CONTROLS,
                    [
                        f"train.batch_size={b}",
                        f"optimizer.lr={lr:g}",
                        f"train.steps={plan['steps']}",
                        f"schedule.evaluate_at=[{','.join(map(str, at))}]",
                    ],
                    [CONTROL_SEED],
                )
            )
    header = f"# Written by scripts/stage_a_campaigns.py {args.kind}; do not edit.\n"
    args.out.write_text(header + yaml.safe_dump({"name": name, "cells": items}, sort_keys=False))
    print(args.out)


if __name__ == "__main__":
    main()
