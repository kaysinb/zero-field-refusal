import dataclasses
import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from zero_field_refusal.analysis import (
    analyze,
    fallback_contrast,
    fit_logistic,
    select_k_star,
    validate_runs,
)
from zero_field_refusal.analysis.bootstrap import paired_resampling
from zero_field_refusal.analysis.primary import PRIMARY_CONTRAST, ArmCurve, CurveDraws

PLACEMENT = {
    "sgd": (0.0, 1.0, 2.0),
    "adamw": (0.5, 1.5, 2.5),
    "adam_frozen": (0.25, 1.25, 2.25),
    "sgd_momentum": (0.0, 1.0, 2.0),
}

LEARNING_RATES = (1e-5, 2e-5, 3e-5, 4e-5)


def _runs(
    delta: float, n_seeds: int = 12, seed: int = 0, placement: dict = PLACEMENT
) -> pd.DataFrame:
    """Runs with true curves expit(-2.5 + 0.8 log K), shifted up by delta for AdamW."""
    rng = np.random.default_rng(seed)
    shift = {"sgd": 0.0, "adamw": delta, "adam_frozen": delta / 2, "sgd_momentum": 0.0}
    rows = []
    for arm, means in placement.items():
        for cell, mean_log_k in enumerate(means):
            for s in range(n_seeds):
                log_k = mean_log_k + rng.normal(0.0, 0.2)
                rate = expit(-2.5 + 0.8 * log_k) + shift[arm] + rng.normal(0.0, 0.03)
                rows.append(
                    {
                        "run_id": f"{arm}-{cell}-{s}",
                        "seed": s,
                        "arm": arm,
                        "lr": LEARNING_RATES[cell],
                        "K": float(np.exp(log_k)),
                        "harmful": int(rng.binomial(300, np.clip(rate, 0.001, 0.999))),
                        "judged": 300,
                    }
                )
    return pd.DataFrame(rows)


def _analyze(runs: pd.DataFrame, method: str = "percentile", seed: int = 1, **kwargs):
    return analyze(
        runs,
        select_k_star(runs),
        level=0.95,
        ci_method=method,
        resamples=300,
        seed=seed,
        **kwargs,
    )


def test_a_large_shift_gives_a_positive_result() -> None:
    result = _analyze(_runs(delta=0.25))
    primary = result.primary["k_star"]
    assert result.decision == "positive"
    assert primary.ci_low < primary.estimate < primary.ci_high
    assert primary.estimate == pytest.approx(0.25, abs=0.05)
    assert set(result.primary) == {"lower_quarter", "k_star", "upper_quarter"}
    assert set(result.arms) == set(PLACEMENT)
    assert result.n_seeds == 12
    assert result.h2 is not None and result.h2.mechanism in {"both", "none"}
    for arm, summary in result.arms.items():
        assert summary.runs_used + summary.runs_excluded == 3 * 12, arm


def test_equal_curves_give_equivalence_with_enough_seeds() -> None:
    result = _analyze(_runs(delta=0.0, n_seeds=30))
    assert result.decision == "equivalence"
    assert result.h2 is not None and result.h2.mechanism == "not_applicable"


@pytest.mark.parametrize("method", ["percentile", "bca", "bootstrap_t"])
def test_results_are_reproducible(method: str) -> None:
    runs = _runs(delta=0.1)
    first, second = _analyze(runs, method), _analyze(runs, method)
    assert first.primary == second.primary
    assert first.primary != _analyze(runs, method, seed=2).primary


def test_k_star_stays_fixed_and_small_k_runs_are_excluded() -> None:
    runs = _runs(delta=0.1)
    selection = select_k_star(runs)
    low = {
        "run_id": "sgd-low",
        "seed": 0,
        "arm": "sgd",
        "lr": 1e-6,
        "K": selection.exclusion_threshold / 2,
        "harmful": 0,
        "judged": 300,
    }
    with_low = pd.concat([runs, pd.DataFrame([low])], ignore_index=True)
    result = analyze(with_low, selection, level=0.95, ci_method="percentile", resamples=100, seed=1)
    sgd = result.arms["sgd"]
    assert (sgd.runs_used, sgd.runs_excluded) == (36, 1)
    kept = runs[runs.arm == "sgd"]
    fit = fit_logistic(np.log(kept.K), kept.harmful, kept.judged)
    assert sgd.intercept == pytest.approx(fit.intercept[0], abs=1e-9)
    assert sgd.slope == pytest.approx(fit.slope[0], abs=1e-9)
    assert result.selection.k_star == selection.k_star


def test_fallback_contrast_is_a_difference_of_mean_rates() -> None:
    runs = _runs(delta=0.25)
    cell, reference = ("adamw", 2e-5), ("sgd", 3e-5)
    result = fallback_contrast(
        runs, cell, reference, level=0.95, ci_method="bca", resamples=300, seed=1
    )
    rate = runs.harmful / runs.judged
    expected = (
        rate[(runs.arm == "adamw") & (runs.lr == 2e-5)].mean()
        - rate[(runs.arm == "sgd") & (runs.lr == 3e-5)].mean()
    )
    assert result.contrast.estimate == pytest.approx(expected)
    assert result.contrast.ci_low < expected < result.contrast.ci_high
    assert result.decision == "positive"
    assert result.n_seeds == 12


def test_the_primary_arms_are_required() -> None:
    runs = _runs(delta=0.1)
    selection = select_k_star(runs)
    with pytest.raises(ValueError, match="adamw"):
        analyze(
            runs[runs.arm != "adamw"],
            selection,
            level=0.95,
            ci_method="percentile",
            resamples=50,
            seed=0,
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("K", 0.0, "K"),
        ("harmful", 301, "harmful"),
        ("arm", "adam", "arm"),
        ("run_id", "sgd-0-1", "run_id"),
        ("seed", 1.5, "seed"),
    ],
)
def test_invalid_runs_are_rejected(column: str, value: object, message: str) -> None:
    runs = _runs(delta=0.0, n_seeds=2)
    runs[column] = runs[column].astype(object)
    runs.loc[0, column] = value
    with pytest.raises(ValueError, match=message):
        validate_runs(runs)


def test_a_fully_excluded_frozen_adam_leaves_h2_unavailable_and_keeps_the_primary() -> None:
    runs = _runs(delta=0.25, placement=PLACEMENT | {"adam_frozen": (-5.0, -4.5, -4.0, -3.5)})
    selection = select_k_star(runs)
    assert selection.h2_descriptive_only
    assert -2.0 < math.log(selection.exclusion_threshold) < -1.5  # cells at -5 to -3.5
    result = analyze(runs, selection, level=0.95, ci_method="percentile", resamples=200, seed=1)
    assert result.decision == "positive"
    assert result.unfitted_arms == {"adam_frozen": 48}
    assert "adam_frozen" not in result.arms
    assert result.runs_excluded == 48
    assert result.h2.mechanism == "unavailable"
    assert (result.h2.vs_sgd, result.h2.vs_adamw) == (None, None)
    assert result.h2.vs_sgd_label == result.h2.vs_adamw_label == "unavailable"


def test_frozen_adam_may_be_absent_from_a_look() -> None:
    runs = _runs(delta=0.25)
    selection = select_k_star(runs)
    result = analyze(
        runs[runs.arm != "adam_frozen"],
        selection,
        level=0.95,
        ci_method="percentile",
        resamples=100,
        seed=1,
    )
    assert result.decision == "positive"
    assert result.unfitted_arms == {}
    assert result.h2.mechanism == "unavailable"


def test_a_fully_excluded_sgd_arm_is_an_error() -> None:
    runs = _runs(delta=0.1)
    selection = select_k_star(runs)
    low = runs.assign(K=np.where(runs.arm == "sgd", runs.K * 1e-6, runs.K))
    with pytest.raises(ValueError, match="sgd"):
        analyze(low, selection, level=0.95, ci_method="percentile", resamples=50, seed=0)


def test_the_exclusion_factor_travels_with_the_selection() -> None:
    runs = _runs(delta=0.1)
    selection = select_k_star(runs, exclusion_factor=30.0)
    assert selection.exclusion_threshold == pytest.approx(selection.c_lo / 30)
    assert selection.exclusion_reason == "K below C_lo/30"
    keep_all = dataclasses.replace(selection, exclusion_factor=math.inf)
    assert keep_all.exclusion_threshold == 0.0
    result = analyze(runs, keep_all, level=0.95, ci_method="percentile", resamples=50, seed=0)
    assert result.runs_excluded == 0
    assert result.exclusion_reason == "no exclusion"


def test_dropped_replicates_are_counted_per_interval() -> None:
    result = _analyze(_runs(delta=0.1), method="bootstrap_t")
    assert result.primary["k_star"].dropped_replicates == 0
    assert all(i.dropped_replicates == 0 for i in result.arms["sgd"].h.values())


def test_contrast_se_pairs_the_arms_influences_per_seed() -> None:
    resampling = paired_resampling(np.arange(4), resamples=5, seed=0)
    rows = resampling.weights.shape[0]
    per_seed = np.broadcast_to(np.array([1.0, -1.0, 2.0, -2.0])[None, :, None], (rows, 4, 3))

    def curve(scale: float) -> ArmCurve:
        h = np.full((rows, 3), 0.2 * scale)
        return ArmCurve(runs_used=4, runs_excluded=0, fit=None, h=h, influence=scale * per_seed)

    def contrast_se(adamw_scale: float) -> float:
        curves = {"sgd": curve(1.0), "adamw": curve(adamw_scale)}
        draws = CurveDraws(selection=None, resampling=resampling, curves=curves, unfitted={})
        return draws.contrast(PRIMARY_CONTRAST).se

    # Identical per-seed influences cancel: the shared seed effect drops out of the difference.
    assert contrast_se(1.0) == 0.0
    # Otherwise the SE is that of the per-seed difference: sqrt(1 + 1 + 4 + 4).
    assert contrast_se(2.0) == pytest.approx(np.sqrt(10.0))
