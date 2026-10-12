"""Eligible cells, which alone define the primary comparison (amendment of 2026-10-12).

A cell (arm, learning rate) of a stage-1 experiment is *eligible* when seeds 0–4 all completed,
none of its runs diverged, every completed run has a positive K and valid accuracies, and the cell
has not collapsed: its mean factual accuracy is at most 10 pp and its mean benchmark accuracy at
most 5 pp below θ₀'s (the cell-level definition of collapse; one poor seed does not by itself
collapse a cell). Eligibility is read from K, divergence and utility alone, never from a judge
label: ``cell_eligibility`` reads each logical run once, held to its experiment's committed
condition (``sweeps.read_logical_runs``), and θ₀'s accuracies from the experiment's θ₀ run.

Only eligible cells define the primary common range, K\\*, the decisive cells and the primary
curves; ``restrict`` leaves every other run out of the primary analysis with the reason
``ineligible_cell``. The analysis over every cell is kept and reported beside it
(``analysis.build``). Eligibility is fixed once, with K\\*, from the stage-1 cells: later
instability excludes no seed and reselects nothing.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from .evidence import Evidence
from .sweeps import (
    BENCHMARK_DROP,
    FACTUAL_DROP,
    TOLERANCE,
    _accuracy,
    read_logical_runs,
    theta0_utility,
)

SEEDS = (0, 1, 2, 3, 4)
INELIGIBLE = "ineligible_cell"  # the exclusion reason of a run outside the eligible cells
RULE = (
    "seeds 0-4 completed, no run diverged, every completed run with a positive K and valid "
    "accuracies, mean factual accuracy at most 10 pp and mean benchmark accuracy at most 5 pp "
    "below theta0's"
)
COLUMNS = ["arm", "lr", "runs", "completed", "diverged", "mean_log_k", "factual", "benchmark",
           "collapse", "eligible", "detail"]  # fmt: skip


def cell_eligibility(
    root: Path, experiment: str, theta0: Mapping[str, float] | None = None
) -> pd.DataFrame:
    """Every training cell of ``root/experiment`` with its eligibility (``COLUMNS``).

    ``theta0`` gives θ₀'s ``factual`` and ``benchmark`` accuracies; None reads them from the
    experiment's one completed θ₀ run. ``detail`` says why a cell is not eligible.
    """
    logical = read_logical_runs(root, experiment)
    if theta0 is None:
        theta0 = theta0_utility(logical)
    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for run in logical:
        if run["run_type"] == "train":
            groups.setdefault((run["arm"], run["lr"]), []).append(run)
    rows = []
    for (arm, lr), runs in sorted(groups.items()):
        done = [r for r in runs if r["status"] == "completed" and r["seed"] in SEEDS]
        valid = [r for r in done if r["k"] is not None and math.isfinite(r["k"]) and r["k"] > 0
                 and _accuracy(r["factual"]) and _accuracy(r["benchmark"])]  # fmt: skip
        diverged = any(r["status"] == "diverged" for r in runs)
        complete = sorted(r["seed"] for r in done) == list(SEEDS)
        if valid:
            factual = sum(r["factual"] for r in valid) / len(valid)
            benchmark = sum(r["benchmark"] for r in valid) / len(valid)
            mean_log_k = sum(math.log(r["k"]) for r in valid) / len(valid)
        else:
            factual = benchmark = mean_log_k = math.nan
        collapse = [name for name, drop, limit in
                    (("factual", theta0["factual"] - factual, FACTUAL_DROP),
                     ("benchmark", theta0["benchmark"] - benchmark, BENCHMARK_DROP))
                    if drop > limit + TOLERANCE]  # fmt: skip
        reasons = []
        if not complete:
            seeds = " ".join(str(s) for s in sorted(r["seed"] for r in done)) or "none"
            reasons.append(f"seeds completed: {seeds} (needs 0-4)")
        if diverged:
            reasons.append("a run diverged")
        if len(valid) != len(done):
            reasons.append("a completed run without a positive K or valid accuracies")
        if collapse:
            reasons.append("collapse: " + "+".join(collapse))
        rows.append({"arm": arm, "lr": float(lr), "runs": len(runs), "completed": len(done),
                     "diverged": diverged, "mean_log_k": mean_log_k, "factual": factual,
                     "benchmark": benchmark, "collapse": "+".join(collapse),
                     "eligible": not reasons, "detail": "; ".join(reasons)})  # fmt: skip
    return pd.DataFrame(rows, columns=COLUMNS)


def eligible_cells(table: pd.DataFrame) -> list[dict[str, Any]]:
    """The eligible cells of a ``cell_eligibility`` table, as ``{arm, lr}`` records."""
    chosen = table[table["eligible"].astype(bool)]
    return [{"arm": str(arm), "lr": float(lr)}
            for arm, lr in zip(chosen["arm"], chosen["lr"], strict=True)]  # fmt: skip


def restrict(
    evidence: Evidence, cells: list[Mapping[str, Any]], details: Mapping[tuple[str, float], str]
) -> Evidence:
    """``evidence`` with only the runs of ``cells`` included; every other included run is left
    out with the reason ``ineligible_cell`` and its cell's entry of ``details`` as the detail."""
    keep = {(str(c["arm"]), float(c["lr"])) for c in cells}
    considered = evidence.considered.copy()
    for index, row in considered.iterrows():
        key = (str(row["arm"]), float(row["lr"]))
        if row["included"] and key not in keep:
            considered.loc[index, ["included", "reason", "detail"]] = [
                False, INELIGIBLE, details.get(key, "not an eligible cell")]  # fmt: skip
    runs = evidence.runs
    in_cells = [(str(a), float(lr)) in keep for a, lr in zip(runs["arm"], runs["lr"], strict=True)]
    return dataclasses.replace(evidence, runs=runs[in_cells].reset_index(drop=True),
                               considered=considered)  # fmt: skip
