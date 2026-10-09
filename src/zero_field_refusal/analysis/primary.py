"""Analysis of one Gate 2 look: curves h(K), Δ(K\\*), H2, and the unequal-displacement fallback.

Inputs are the runs of the look (sweep runs of every cell plus the decisive cells' seeds of the
look) and the ``KStarSelection`` fixed from the sweep before look 1 (``select_k_star``). Per arm,
h(K) is fitted to all of the arm's runs except those below the selection's exclusion threshold
(C_lo/10 by default). The point estimate and every paired-bootstrap and jackknife replicate are
fitted at once; contrasts are linear combinations of arms' h at a level (K\\* or a quarter
point). Numbers are returned unrounded.

SGD and AdamW must keep runs after the exclusion. Any other arm whose runs are all excluded, or
that is absent, has no curve: it is listed in ``unfitted_arms`` and, for frozen Adam, the H2
comparisons are ``unavailable`` while Δ(K\\*) is still estimated.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bootstrap import (
    CI_METHODS,
    BootstrapDraws,
    SeedResampling,
    confidence_interval,
    dropped_replicates,
    paired_resampling,
)
from .curves import LogisticFit, cluster_influence, fit_logistic
from .decisions import (
    EQUIVALENCE,
    EQUIVALENCE_MARGIN,
    UNAVAILABLE,
    h2_comparison,
    h2_mechanism,
    look_decision,
)
from .kstar import LEVEL_NAMES, KStarSelection
from .runs import validate_runs

PRIMARY_CONTRAST = {"adamw": 1.0, "sgd": -1.0}
FROZEN_VS_SGD = {"adam_frozen": 1.0, "sgd": -1.0}
FROZEN_VS_ADAMW = {"adam_frozen": 1.0, "adamw": -1.0}
CONTRAST_ARMS = ("sgd", "adamw")  # needed for Δ(K*)


@dataclass(frozen=True)
class ArmCurve:
    """One arm's fits for every weight row (point, bootstrap resamples, jackknife)."""

    runs_used: int
    runs_excluded: int
    fit: LogisticFit
    h: np.ndarray  # (rows, levels), levels ordered as LEVEL_NAMES
    influence: np.ndarray  # (rows, seeds, levels), see curves.cluster_influence


@dataclass(frozen=True)
class CurveDraws:
    """Curves of all arms at one look, with the resampling that produced them."""

    selection: KStarSelection
    resampling: SeedResampling
    curves: dict[str, ArmCurve]
    unfitted: dict[str, int]  # arm -> runs, for arms whose runs were all excluded

    def contrast(self, coefficients: Mapping[str, float], level: str = "k_star") -> BootstrapDraws:
        """Σ coefficient·h_arm at a level, e.g. ``{"adamw": 1, "sgd": -1}`` at ``"k_star"``."""
        missing = [arm for arm in coefficients if arm not in self.curves]
        if missing:
            raise ValueError(f"No curve for arm(s) {missing}")
        j = LEVEL_NAMES.index(level)
        values = sum(c * self.curves[arm].h[:, j] for arm, c in coefficients.items())
        influence = sum(c * self.curves[arm].influence[:, :, j] for arm, c in coefficients.items())
        return _draws(values, influence, self.resampling)


@dataclass(frozen=True)
class Interval:
    estimate: float
    ci_low: float
    ci_high: float
    se: float  # cluster-robust delta-method SE at the estimate
    dropped_replicates: int  # bootstrap replicates the CI method could not use


@dataclass(frozen=True)
class ArmResult:
    intercept: float  # logit h = intercept + slope·log K, K in nats
    slope: float
    h: dict[str, Interval]  # at K* and the quarter points
    runs_used: int
    runs_excluded: int  # selection.exclusion_reason
    separated: dict[str, int]  # fits without a unique finite estimate: point, resamples, jackknife
    not_converged: dict[str, int]


@dataclass(frozen=True)
class H2Result:
    vs_sgd: Interval | None  # h_FA(K*) − h_SGD(K*); None when unavailable
    vs_adamw: Interval | None  # h_FA(K*) − h_AdamW(K*)
    vs_sgd_label: str
    vs_adamw_label: str
    between: bool | None
    mechanism: str


@dataclass(frozen=True)
class AnalysisResult:
    selection: KStarSelection
    level: float
    ci_method: str
    resamples: int
    seed: int
    margin: float
    seeds: list[int]
    n_seeds: int
    seeds_per_decisive_cell: dict[str, dict[float, int]]
    runs_total: int
    runs_excluded: int
    exclusion_reason: str
    unfitted_arms: dict[str, int]  # arm -> runs, all excluded: no curve
    arms: dict[str, ArmResult]
    primary: dict[str, Interval]  # Δ = h_AdamW − h_SGD at K* and the quarter points
    decision: str  # from the CI of Δ(K*)
    equivalence_at_all_levels: bool
    h2: H2Result


@dataclass(frozen=True)
class FallbackResult:
    cell: tuple[str, float]
    reference: tuple[str, float]
    contrast: Interval  # mean harmful rate of cell minus that of reference
    decision: str
    level: float
    ci_method: str
    resamples: int
    seed: int
    margin: float
    n_seeds: int
    runs: dict[str, int]  # {"cell": n, "reference": n}


def curve_draws(
    runs: pd.DataFrame, selection: KStarSelection, *, resamples: int, seed: int
) -> CurveDraws:
    """Fit every arm's curve for the point estimate and all paired resamples of one look."""
    return _curve_draws(validate_runs(runs), selection, resamples, seed)


def analyze(
    runs: pd.DataFrame,
    selection: KStarSelection,
    *,
    level: float,
    ci_method: str,
    resamples: int,
    seed: int,
    margin: float = EQUIVALENCE_MARGIN,
) -> AnalysisResult:
    """The Gate 2 analysis of one look at a fixed K\\* (``selection``)."""
    if ci_method not in CI_METHODS:
        raise ValueError(f"Unknown CI method {ci_method!r}; expected one of {CI_METHODS}")
    table = validate_runs(runs)
    draws = _curve_draws(table, selection, resamples, seed)

    def interval(coefficients: Mapping[str, float], name: str = "k_star") -> Interval:
        contrast = draws.contrast(coefficients, name)
        low, high = confidence_interval(contrast, ci_method, level)
        dropped = dropped_replicates(contrast, ci_method)
        return Interval(contrast.estimate, low, high, contrast.se, dropped)

    primary = {name: interval(PRIMARY_CONTRAST, name) for name in LEVEL_NAMES}
    decision = look_decision(primary["k_star"].ci_low, primary["k_star"].ci_high, margin)
    arms = {
        arm: _arm_result(curve, {name: interval({arm: 1.0}, name) for name in LEVEL_NAMES}, draws)
        for arm, curve in draws.curves.items()
    }
    return AnalysisResult(
        selection=selection,
        level=level,
        ci_method=ci_method,
        resamples=resamples,
        seed=seed,
        margin=margin,
        seeds=[int(s) for s in draws.resampling.seeds],
        n_seeds=len(draws.resampling.seeds),
        seeds_per_decisive_cell=_decisive_seed_counts(table, selection),
        runs_total=len(table),
        runs_excluded=sum(curve.runs_excluded for curve in draws.curves.values())
        + sum(draws.unfitted.values()),
        exclusion_reason=selection.exclusion_reason,
        unfitted_arms=draws.unfitted,
        arms=arms,
        primary=primary,
        decision=decision,
        equivalence_at_all_levels=all(
            look_decision(i.ci_low, i.ci_high, margin) == EQUIVALENCE for i in primary.values()
        ),
        h2=_h2(draws, interval, decision, margin),
    )


def fallback_contrast(
    runs: pd.DataFrame,
    cell: tuple[str, float],
    reference: tuple[str, float],
    *,
    level: float,
    ci_method: str,
    resamples: int,
    seed: int,
    margin: float = EQUIVALENCE_MARGIN,
) -> FallbackResult:
    """Gate 2 fallback: mean harmful rate of ``cell`` minus that of ``reference``.

    Cells are (arm, lr). The mean rate of a cell is the mean of its runs' harmful/judged. The CI
    comes from the same paired bootstrap over the seeds present in either cell.
    """
    table = validate_runs(runs)
    masks = [(table["arm"] == arm) & (table["lr"] == lr) for arm, lr in (cell, reference)]
    for mask, name in zip(masks, ("cell", "reference"), strict=True):
        if not mask.any():
            raise ValueError(f"No runs in the {name} cell")
    resampling = paired_resampling(
        table.loc[masks[0] | masks[1], "seed"], resamples=resamples, seed=seed
    )
    (cell_mean, cell_influence), (ref_mean, ref_influence) = (
        _mean_rate(table[mask], resampling) for mask in masks
    )
    draws = _draws(cell_mean - ref_mean, cell_influence - ref_influence, resampling)
    low, high = confidence_interval(draws, ci_method, level)
    return FallbackResult(
        cell=cell,
        reference=reference,
        contrast=Interval(
            draws.estimate, low, high, draws.se, dropped_replicates(draws, ci_method)
        ),
        decision=look_decision(low, high, margin),
        level=level,
        ci_method=ci_method,
        resamples=resamples,
        seed=seed,
        margin=margin,
        n_seeds=len(resampling.seeds),
        runs={"cell": int(masks[0].sum()), "reference": int(masks[1].sum())},
    )


def _curve_draws(
    table: pd.DataFrame, selection: KStarSelection, resamples: int, seed: int
) -> CurveDraws:
    missing = [arm for arm in CONTRAST_ARMS if arm not in set(table["arm"])]
    if missing:
        raise ValueError(f"The analysis needs runs of SGD and AdamW; missing {missing}")
    resampling = paired_resampling(table["seed"], resamples=resamples, seed=seed)
    log_levels = np.log([selection.levels[name] for name in LEVEL_NAMES])
    kept = table["K"] >= selection.exclusion_threshold
    curves, unfitted = {}, {}
    for arm, group in table.groupby("arm", sort=True):
        used = group[kept[group.index]]
        if used.empty:
            if arm in CONTRAST_ARMS:
                raise ValueError(
                    f"Every run of arm {arm!r} has {selection.exclusion_reason}: "
                    "Δ(K*) cannot be estimated"
                )
            unfitted[str(arm)] = len(group)
            continue
        log_k = np.log(used["K"].to_numpy())
        harmful, judged = used["harmful"].to_numpy(), used["judged"].to_numpy()
        seed_index = resampling.seed_index(used["seed"].to_numpy())
        weights = resampling.weights[:, seed_index]
        fit = fit_logistic(log_k, harmful, judged, weights)
        influence = cluster_influence(
            fit, log_k, harmful, judged, weights, seed_index, len(resampling.seeds), log_levels
        )
        curves[str(arm)] = ArmCurve(
            runs_used=len(used),
            runs_excluded=len(group) - len(used),
            fit=fit,
            h=fit.predict(log_levels),
            influence=influence,
        )
    return CurveDraws(selection=selection, resampling=resampling, curves=curves, unfitted=unfitted)


def _draws(values: np.ndarray, influence: np.ndarray, resampling: SeedResampling) -> BootstrapDraws:
    """Split per-row values into point, bootstrap and jackknife; SE from per-seed influence."""
    se = np.sqrt((resampling.weights * influence**2).sum(axis=1))
    bootstrap = slice(1, 1 + resampling.resamples)
    return BootstrapDraws(
        estimate=float(values[0]),
        se=float(se[0]),
        replicates=values[bootstrap],
        replicate_se=se[bootstrap],
        jackknife=values[1 + resampling.resamples :],
    )


def _mean_rate(cell: pd.DataFrame, resampling: SeedResampling) -> tuple[np.ndarray, np.ndarray]:
    """Weighted mean rate per row and the influence of one copy of each seed on it."""
    rate = (cell["harmful"] / cell["judged"]).to_numpy()
    seed_index = resampling.seed_index(cell["seed"].to_numpy())
    weights = resampling.weights[:, seed_index]
    total = weights.sum(axis=1)
    membership = np.zeros((len(rate), len(resampling.seeds)))
    membership[np.arange(len(rate)), seed_index] = 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = weights @ rate / total
        influence = ((rate - mean[:, None]) / total[:, None]) @ membership
    return mean, influence


def _arm_result(curve: ArmCurve, h: dict[str, Interval], draws: CurveDraws) -> ArmResult:
    parts = {
        "point": slice(0, 1),
        "resamples": slice(1, 1 + draws.resampling.resamples),
        "jackknife": slice(1 + draws.resampling.resamples, None),
    }
    fit = curve.fit
    return ArmResult(
        intercept=float(fit.intercept[0]),
        slope=float(fit.slope[0]),
        h=h,
        runs_used=curve.runs_used,
        runs_excluded=curve.runs_excluded,
        separated={name: int(fit.separated[rows].sum()) for name, rows in parts.items()},
        not_converged={name: int((~fit.converged[rows]).sum()) for name, rows in parts.items()},
    )


def frozen_adam_between(draws: CurveDraws) -> bool:
    """Whether the estimate of h_FA(K*) lies between those of SGD and AdamW (bounds included)."""
    j = LEVEL_NAMES.index("k_star")
    h_sgd, h_adamw, h_frozen = (
        draws.curves[arm].h[0, j] for arm in ("sgd", "adamw", "adam_frozen")
    )
    return bool(min(h_sgd, h_adamw) <= h_frozen <= max(h_sgd, h_adamw))


def _h2(
    draws: CurveDraws,
    interval: Callable[[Mapping[str, float]], Interval],
    decision: str,
    margin: float,
) -> H2Result:
    if "adam_frozen" not in draws.curves:
        return H2Result(None, None, UNAVAILABLE, UNAVAILABLE, None, UNAVAILABLE)
    vs_sgd, vs_adamw = interval(FROZEN_VS_SGD), interval(FROZEN_VS_ADAMW)
    between = frozen_adam_between(draws)
    sgd_label = h2_comparison(vs_sgd.ci_low, vs_sgd.ci_high, margin)
    adamw_label = h2_comparison(vs_adamw.ci_low, vs_adamw.ci_high, margin)
    return H2Result(
        vs_sgd=vs_sgd,
        vs_adamw=vs_adamw,
        vs_sgd_label=sgd_label,
        vs_adamw_label=adamw_label,
        between=between,
        mechanism=h2_mechanism(
            sgd_label,
            adamw_label,
            between,
            descriptive_only=draws.selection.h2_descriptive_only,
            primary_decision=decision,
        ),
    )


def _decisive_seed_counts(
    table: pd.DataFrame, selection: KStarSelection
) -> dict[str, dict[float, int]]:
    counts = table.groupby(["arm", "lr"])["seed"].nunique()
    return {
        arm: {lr: int(counts.get((arm, lr), 0)) for lr in lrs}
        for arm, lrs in selection.decisive.items()
    }
