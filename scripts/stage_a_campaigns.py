"""Write the campaigns of study 2's Stage A, the data-dependent ones from the run records.

    python scripts/stage_a_campaigns.py setup --out FILE              # θ₀ evaluation, seed 2000
    python scripts/stage_a_campaigns.py screen --out FILE             # 10 SGD rates × 2 batches
    python scripts/stage_a_campaigns.py grids --seed S --out FILE     # AdamW-long, 25-step grids
    python scripts/stage_a_campaigns.py extension --runs DIR... --seed S --out FILE
    python scripts/stage_a_campaigns.py search --runs DIR... [--theta0 RUN] --seed S --out FILE
    python scripts/stage_a_campaigns.py controls --runs DIR... --theta0 RUN --out FILE
    python scripts/stage_a_campaigns.py diagnostic --runs DIR... --theta0 RUN --seed S --out FILE

A campaign holds one seed, since each seed's stream has one writer, its Pod (the full controls
hold seed 2000's). The data-dependent ones check their inputs with the selection's own checks
(analysis.stage_a) and refuse anything not whole. ``extension``: the rates of the 25-step grids
at batch 5 that the runs of all three discovery seeds call for and this seed lacks (write it again
once those runs are done: a second halving may follow). Under amendment 1 the long arms are
searched downward under monitor version 2 (``analysis.stage_a.search_manifest``): ``search``
writes this seed's runs of the rates the search needs next (its first step, the largest rate of
each long arm at each batch, needs only the screen); ``controls`` one full-length control on seed
2000 for each SGD rate whose three seeds have run and whose control is missing, as long as the
longest of those runs and evaluated at seed 2000's evaluation steps (``--control-rule 2``:
amendment 2's rerun of every such control under the amended gradient rule, in
study2_controls_amended, once each); ``diagnostic`` this seed's
diagnostic run of the rate the qualification needs, when the selected batch has fewer than ten
diagnostic states for an optimizer. ``--theta0`` (the θ₀ run) is needed once version-2 runs exist,
for the selection rule's utility. The rules are those of docs/study2-preregistration.md; nothing
is chosen by hand; ``--cells`` only splits a campaign over Pods. FILE must not exist; write it
outside the checkout or under an ignored
directory, such as $RUNS_DIR/campaigns/, so that runs from a clean checkout stay clean.
"""

import argparse
import json
from pathlib import Path

import yaml

from zero_field_refusal.analysis.stage_a import (
    ADAMW_LONG_GRID,
    BATCHES,
    CONTROL_SEED,
    DISCOVERY_SEEDS,
    ROLES,
    SCREEN_STEPS,
    SGD_GRID,
    SHORT_ARMS,
    SHORT_GRID,
    checked_discovery,
    control_rule,
    control_schedule,
    load_trajectories,
    prescribed_short_rates,
    rate,
    search_progress,
    select,
)

MAIN, CONTROLS = "experiments/study2_stage_a", "experiments/study2_controls"
CONTROLS_AMENDED = "experiments/study2_controls_amended"  # amendment 2
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
    kinds = ("setup", "screen", "grids", "sgd", "extension", "search", "controls", "diagnostic")
    parser.add_argument("kind", choices=kinds)
    parser.add_argument("--runs", type=Path, nargs="*", default=[])
    parser.add_argument("--theta0", type=Path, help="the θ₀ run directory (summary.json)")
    parser.add_argument("--seed", type=int, choices=DISCOVERY_SEEDS)
    parser.add_argument(
        "--control-rule",
        type=int,
        choices=(1, 2),
        default=1,
        help="controls: 2 reruns them under amendment 2's gradient rule",
    )
    parser.add_argument("--cells", nargs="+", help="keep only these cells (one Pod each)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"{args.out} exists")
    if args.kind in ("grids", "extension", "search", "diagnostic") and args.seed is None:
        parser.error(f"{args.kind} needs --seed")
    if args.kind == "sgd":
        parser.error("superseded by the downward search of amendment 1: use search")
    runs = load_trajectories(*args.runs) if args.runs else []
    theta0 = None
    if args.theta0 is not None:
        summary = json.loads((args.theta0 / "summary.json").read_text())
        theta0 = {key: summary[key] for key in ("benchmark_accuracy", "factual_accuracy")}
    by_role = {role: [r for r in runs if r.role == role] for role in ROLES}
    # The search follows the preregistered full controls; amendment 2's reruns stand beside them.
    amended = [r for r in by_role["full_control"] if control_rule(r) == 2]
    by_role["full_control"] = [r for r in by_role["full_control"] if control_rule(r) == 1]

    def progress() -> dict:
        try:
            state, problems = search_progress(by_role["screen"], by_role["discovery"],
                                              by_role["full_control"], theta0)  # fmt: skip
        except ValueError as exc:
            parser.error(str(exc))
        if problems:
            parser.error("the search's records are not whole: " + "; ".join(problems))
        return state

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
    elif args.kind == "search":
        manifest = progress()["manifest"]
        name, items = f"study2-a-search-s{args.seed}", []
        # AdamW-long at B = 20 first: the run that needs the most GPU memory shows it early.
        order = [f"{arm}/B={b}" for arm in ("adamw_long", "sgd_long") for b in (20, 5)]
        for entry in (manifest["searches"][key] for key in order):
            pending = entry["pending"] or {}
            if args.seed in pending.get("seeds", []):
                arm, b, lr = entry["arm"], entry["batch"], pending["rate"]
                items += [c for c in long_cells(arm, [lr], args.seed)
                          if c["name"].startswith(f"{arm}-b{b}-")]  # fmt: skip
        if not items:
            waiting = {k: e["pending"] for k, e in manifest["searches"].items() if e["pending"]}
            parser.error(f"the search needs no run of seed {args.seed} now; waiting: {waiting}")
    elif args.kind == "extension":
        discovery = [r for r in runs if r.role == "discovery" and r.batch_size == 5]
        present, problems = checked_discovery(discovery, arms=SHORT_ARMS)
        name, items = f"study2-a-extension-s{args.seed}", []
        for arm in SHORT_ARMS:
            grid, missing = prescribed_short_rates(present, arm)
            problems += missing
            new = [lr for lr in grid if (arm, 5, lr, args.seed) not in present]
            items += short_cells(arm, 5, new, args.seed)
        if problems:
            parser.error(
                "the 25-step runs at batch 5 of every seed are needed: " + "; ".join(problems)
            )
        if not items:
            parser.error("no extension is called for")
    elif args.kind == "controls":
        state = progress()
        if args.control_rule == 1:  # the controls the search waits for
            name, config = "study2-a-controls", CONTROLS
            searches = state["manifest"]["searches"].values()
            due = [(e["batch"], e["pending"]["rate"]) for e in searches
                   if e["arm"] == "sgd_long" and "control" in (e["pending"] or {})]  # fmt: skip
        else:  # amendment 2: every SGD rate run on all three seeds, rerun once
            name, config = "study2-a-controls-amended", CONTROLS_AMENDED
            rerun = {(r.batch_size, rate(r.lr)) for r in amended}
            due = [(b, lr) for b in BATCHES for lr in state["candidates"].get(b, [])
                   if all(("sgd_long", b, lr, s) in state["present"] for s in DISCOVERY_SEEDS)
                   and (b, lr) not in rerun]  # fmt: skip
        items = []
        for b, lr in due:
            group = [state["present"][("sgd_long", b, lr, s)] for s in DISCOVERY_SEEDS]
            plan = control_schedule(group)  # the longest over the three seeds
            at = plan["seeds"][CONTROL_SEED]
            items.append(
                cell(
                    f"control-b{b}-{lr:g}",
                    config,
                    [
                        f"train.batch_size={b}",
                        f"optimizer.lr={lr:g}",
                        f"train.steps={plan['steps']}",
                        f"schedule.evaluate_at=[{','.join(map(str, at))}]",
                    ],
                    [CONTROL_SEED],
                )
            )
        if not items:
            parser.error("no SGD rate of the search waits for its full control")

    else:  # diagnostic
        if theta0 is None:
            parser.error("diagnostic needs --theta0")
        result = select(by_role["discovery"], by_role["screen"], by_role["full_control"],
                        theta0, by_role["diagnostic"])  # fmt: skip
        following = result.get("diagnostic_next")
        if not following:
            parser.error(f"no diagnostic run is called for (outcome {result['outcome']})")
        name, items = f"study2-a-diagnostic-s{args.seed}", []
        b = result["batch_size"]
        for arm, lr in following.items():
            if any(r.arm == arm and r.batch_size == b and rate(r.lr) == lr and r.seed == args.seed
                   for r in by_role["diagnostic"]):  # fmt: skip
                continue
            items += [{**c, "overrides": [*c["overrides"], "stage_a.role=diagnostic"]}
                      for c in long_cells(arm, [lr], args.seed)
                      if c["name"].startswith(f"{arm}-b{b}-")]  # fmt: skip
        if not items:
            parser.error(f"seed {args.seed} has its diagnostic runs already")
    if args.cells:  # a campaign split over Pods, cell by cell, without choosing anything
        unknown = sorted(set(args.cells) - {c["name"] for c in items})
        if unknown:
            parser.error(f"no such cells in this campaign: {unknown}")
        items = [c for c in items if c["name"] in args.cells]
    header = f"# Written by scripts/stage_a_campaigns.py {args.kind}; do not edit.\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(header + yaml.safe_dump({"name": name, "cells": items}, sort_keys=False))
    print(args.out)


if __name__ == "__main__":
    main()
