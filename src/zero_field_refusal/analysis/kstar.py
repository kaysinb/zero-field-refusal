"""K\\*, the common KL range and the decisive cells, from KL values only.

All operations are on the log scale. A cell mean is the mean of log K over the cell's runs (the log
of the geometric mean of K). An arm's range spans its cell means; the common range [C_lo, C_hi] is
the overlap of the ranges of SGD, frozen Adam and AdamW. If frozen Adam's range misses the overlap
of SGD and AdamW, the common range is that overlap and H2 is descriptive only; if SGD and AdamW do
not overlap, K\\* is undefined. Ranges and overlaps are closed intervals. K\\* is the geometric
midpoint √(C_lo·C_hi); the auxiliary levels are the quarter points of the range on the log scale.

The selection is made once, from the sweep runs before look 1, and passed unchanged to every
later analysis: K\\* is not re-estimated in bootstrap resamples or at look 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .runs import PRIMARY_ARMS, validate_runs

# Runs with K below C_lo/EXCLUSION_FACTOR, a decade under the common range, leave the curve fits.
# The factor is part of the selection; math.inf keeps every run.
EXCLUSION_FACTOR = 10.0
LEVEL_NAMES = ("lower_quarter", "k_star", "upper_quarter")


class CommonRangeUndefined(ValueError):
    """The K ranges of SGD and AdamW do not overlap, so K\\* is undefined."""


@dataclass(frozen=True)
class KStarSelection:
    """K\\* and the decisive cells, fixed before look 1. K values in nats."""

    arm_ranges: dict[str, tuple[float, float]]  # arm -> (smallest, largest) geometric cell mean
    c_lo: float
    c_hi: float
    k_star: float
    lower_quarter: float
    upper_quarter: float
    h2_descriptive_only: bool  # frozen Adam misses the SGD-AdamW overlap
    decisive: dict[str, tuple[float, ...]]  # arm -> learning rates, ordered by cell mean
    exclusion_factor: float = EXCLUSION_FACTOR

    @property
    def levels(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in LEVEL_NAMES}

    @property
    def exclusion_threshold(self) -> float:
        """Runs with K below this leave the curve fits."""
        return self.c_lo / self.exclusion_factor

    @property
    def exclusion_reason(self) -> str:
        if math.isinf(self.exclusion_factor):
            return "no exclusion"
        return f"K below C_lo/{self.exclusion_factor:g}"


def cell_means(runs: pd.DataFrame) -> pd.DataFrame:
    """Columns ``arm``, ``lr``, ``n_runs`` and ``mean_log_k`` (mean of log K), one row per cell."""
    table = validate_runs(runs, outcomes=False).assign(log_k=lambda t: np.log(t["K"]))
    grouped = table.groupby(["arm", "lr"], sort=True)["log_k"]
    return grouped.agg(n_runs="size", mean_log_k="mean").reset_index()


def common_range(cells: pd.DataFrame) -> tuple[float, float, bool]:
    """(C_lo, C_hi, h2_descriptive_only) from cell means; raises if K\\* is undefined."""
    ranges = _log_ranges(cells)
    missing = [arm for arm in PRIMARY_ARMS if arm not in ranges]
    if missing:
        raise ValueError(f"K* needs cells of every primary arm; missing {missing}")
    lo = max(ranges["sgd"][0], ranges["adamw"][0])
    hi = min(ranges["sgd"][1], ranges["adamw"][1])
    if lo > hi:
        raise CommonRangeUndefined(
            f"SGD and AdamW K ranges do not overlap: sgd {_k_range(ranges['sgd'])}, "
            f"adamw {_k_range(ranges['adamw'])}"
        )
    frozen_lo, frozen_hi = max(lo, ranges["adam_frozen"][0]), min(hi, ranges["adam_frozen"][1])
    if frozen_lo > frozen_hi:
        return math.exp(lo), math.exp(hi), True
    return math.exp(frozen_lo), math.exp(frozen_hi), False


def k_star_levels(c_lo: float, c_hi: float) -> tuple[float, float, float]:
    """(lower quarter point, K\\*, upper quarter point) of [C_lo, C_hi] on the log scale."""
    log_lo, log_hi = math.log(c_lo), math.log(c_hi)
    return tuple(math.exp(log_lo + q * (log_hi - log_lo)) for q in (0.25, 0.5, 0.75))


def decisive_cells(cells: pd.DataFrame, k_star: float) -> dict[str, tuple[float, ...]]:
    """Per arm, the learning rates of the cells nearest to K\\* on the log scale.

    The nearest cell at or below K\\* and the nearest above it; if one side has no cell, the single
    closest cell. A cell mean exactly at K\\* counts as below.
    """
    target = math.log(k_star)
    decisive = {}
    for arm, group in cells.groupby("arm", sort=True):
        below = group[group["mean_log_k"] <= target]
        above = group[group["mean_log_k"] > target]
        chosen = []
        if not below.empty:
            chosen.append(below.loc[below["mean_log_k"].idxmax(), "lr"])
        if not above.empty:
            chosen.append(above.loc[above["mean_log_k"].idxmin(), "lr"])
        decisive[str(arm)] = tuple(float(lr) for lr in chosen)
    return decisive


def select_k_star(
    sweep_runs: pd.DataFrame, *, exclusion_factor: float = EXCLUSION_FACTOR
) -> KStarSelection:
    """K\\*, its auxiliary levels and the decisive cells from the sweep runs' KL values."""
    cells = cell_means(sweep_runs)
    c_lo, c_hi, descriptive_only = common_range(cells)
    lower, k_star, upper = k_star_levels(c_lo, c_hi)
    return KStarSelection(
        arm_ranges={arm: _k_range(r) for arm, r in _log_ranges(cells).items()},
        c_lo=c_lo,
        c_hi=c_hi,
        k_star=k_star,
        lower_quarter=lower,
        upper_quarter=upper,
        h2_descriptive_only=descriptive_only,
        decisive=decisive_cells(cells, k_star),
        exclusion_factor=exclusion_factor,
    )


def _log_ranges(cells: pd.DataFrame) -> dict[str, tuple[float, float]]:
    grouped = cells.groupby("arm", sort=True)["mean_log_k"]
    return {str(arm): (float(group.min()), float(group.max())) for arm, group in grouped}


def _k_range(log_range: tuple[float, float]) -> tuple[float, float]:
    return math.exp(log_range[0]), math.exp(log_range[1])
