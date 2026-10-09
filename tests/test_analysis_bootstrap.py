import numpy as np
import pytest
import scipy.stats

from zero_field_refusal.analysis.bootstrap import (
    BootstrapDraws,
    confidence_interval,
    confidence_intervals,
    dropped_replicates,
    paired_resampling,
)

# Runs of three arms: seeds 0-4 in every arm, seeds 5-7 only in the last arm.
RUN_SEEDS = np.array([0, 1, 2, 3, 4] * 2 + [0, 1, 2, 3, 4, 5, 6, 7])


def test_all_runs_of_a_seed_share_its_weight() -> None:
    resampling = paired_resampling(RUN_SEEDS, resamples=50, seed=0)
    assert resampling.seeds.tolist() == list(range(8))
    weights = resampling.run_weights(RUN_SEEDS)
    assert weights.shape == (1 + 50 + 8, len(RUN_SEEDS))
    for seed in range(8):
        columns = weights[:, seed == RUN_SEEDS]
        assert (columns == columns[:, :1]).all()
    np.testing.assert_array_equal(weights[:, RUN_SEEDS == 5][:, 0], resampling.weights[:, 5])


def test_rows_are_point_resamples_and_leave_one_seed_out() -> None:
    resampling = paired_resampling(RUN_SEEDS, resamples=50, seed=0)
    assert (resampling.point == 1).all()
    assert (resampling.bootstrap.sum(axis=1) == 8).all()
    np.testing.assert_array_equal(resampling.jackknife, 1 - np.eye(8))


def test_resamples_are_reproducible() -> None:
    first = paired_resampling(RUN_SEEDS, resamples=50, seed=3).weights
    np.testing.assert_array_equal(first, paired_resampling(RUN_SEEDS, resamples=50, seed=3).weights)
    assert (first != paired_resampling(RUN_SEEDS, resamples=50, seed=4).weights).any()


def _mean_draws(values: np.ndarray, resamples: int, seed: int) -> BootstrapDraws:
    """Draws of the mean of per-seed values, with the per-seed influence for the SE."""
    resampling = paired_resampling(np.arange(len(values)), resamples=resamples, seed=seed)
    weights = resampling.weights
    means = weights @ values / weights.sum(axis=1)
    se = np.sqrt((weights * (values - means[:, None]) ** 2).sum(axis=1)) / weights.sum(axis=1)
    rows = (slice(0, 1), slice(1, 1 + resamples), slice(1 + resamples, None))
    return BootstrapDraws(
        estimate=float(means[0]),
        se=float(se[0]),
        replicates=means[rows[1]],
        replicate_se=se[rows[1]],
        jackknife=means[rows[2]],
    )


def test_percentile_interval_is_the_quantile_pair() -> None:
    draws = _mean_draws(np.random.default_rng(0).normal(size=30), 999, seed=1)
    low, high = confidence_interval(draws, "percentile", 0.9)
    assert (low, high) == pytest.approx(tuple(np.quantile(draws.replicates, [0.05, 0.95])))


def test_bca_matches_percentile_on_a_symmetric_case() -> None:
    half = np.random.default_rng(5).normal(size=20)
    draws = _mean_draws(np.concatenate([half, -half]), 20000, seed=6)
    levels = np.array([0.9, 0.95, 0.9706])
    percentile = confidence_intervals(draws, "percentile", levels)
    bca = confidence_intervals(draws, "bca", levels)
    width = percentile[:, 1] - percentile[:, 0]
    assert np.abs(bca - percentile).max() < 0.05 * width.min()


def test_bootstrap_t_with_constant_se_is_the_basic_interval() -> None:
    draws = _mean_draws(np.random.default_rng(0).normal(size=30), 999, seed=1)
    constant = BootstrapDraws(
        estimate=draws.estimate,
        se=0.2,
        replicates=draws.replicates,
        replicate_se=np.full(999, 0.2),
        jackknife=draws.jackknife,
    )
    low, high = confidence_interval(constant, "bootstrap_t", 0.9)
    q_low, q_high = np.quantile(draws.replicates, [0.05, 0.95])
    assert low == pytest.approx(2 * draws.estimate - q_high)
    assert high == pytest.approx(2 * draws.estimate - q_low)


def test_intervals_widen_with_the_level_and_ignore_invalid_replicates() -> None:
    draws = _mean_draws(np.random.default_rng(0).normal(size=30), 999, seed=1)
    for method in ("percentile", "bca", "bootstrap_t"):
        intervals = confidence_intervals(draws, method, np.array([0.8, 0.9, 0.99]))
        assert (np.diff(intervals[:, 0]) < 0).all() and (np.diff(intervals[:, 1]) > 0).all()
    with_nan = BootstrapDraws(
        estimate=draws.estimate,
        se=draws.se,
        replicates=np.append(draws.replicates, np.nan),
        replicate_se=np.append(draws.replicate_se, np.nan),
        jackknife=draws.jackknife,
    )
    assert confidence_interval(with_nan, "percentile", 0.9) == confidence_interval(
        draws, "percentile", 0.9
    )


def test_unknown_method_is_rejected() -> None:
    draws = _mean_draws(np.random.default_rng(0).normal(size=10), 99, seed=1)
    with pytest.raises(ValueError, match="method"):
        confidence_interval(draws, "normal", 0.95)


def test_bca_matches_scipy_on_a_skewed_case() -> None:
    values = np.random.default_rng(7).exponential(size=30)
    draws = _mean_draws(values, 40000, seed=8)
    for level in (0.9, 0.95):
        reference = scipy.stats.bootstrap(
            (values,),
            np.mean,
            confidence_level=level,
            method="BCa",
            n_resamples=40000,
            rng=np.random.default_rng(9),
        ).confidence_interval
        width = reference.high - reference.low
        bca = confidence_interval(draws, "bca", level)
        percentile = confidence_interval(draws, "percentile", level)
        assert bca == pytest.approx((reference.low, reference.high), abs=0.02 * width)
        # Skewed enough that BCa and percentile differ well beyond the tolerance.
        assert abs(bca[1] - percentile[1]) > 0.05 * width


def test_bootstrap_t_studentizes_each_replicate_by_its_own_se() -> None:
    rng = np.random.default_rng(3)
    replicates = 0.1 + rng.exponential(0.02, 999) - 0.02  # skewed, so orientation matters
    replicate_se = rng.uniform(0.01, 0.03, 999)
    draws = BootstrapDraws(
        estimate=0.1,
        se=0.02,
        replicates=replicates,
        replicate_se=replicate_se,
        jackknife=np.ones(5),
    )
    low, high = confidence_interval(draws, "bootstrap_t", 0.9)
    q_low, q_high = np.quantile((replicates - 0.1) / replicate_se, [0.05, 0.95])
    assert (low, high) == pytest.approx((0.1 - q_high * 0.02, 0.1 - q_low * 0.02))
    assert q_high != pytest.approx(-q_low, rel=0.1)


def test_dropped_replicates_depend_on_the_method() -> None:
    draws = BootstrapDraws(
        estimate=0.0,
        se=1.0,
        replicates=np.array([0.1, np.nan, 0.2, 0.3]),
        replicate_se=np.array([1.0, 1.0, 0.0, np.inf]),
        jackknife=np.ones(3),
    )
    assert dropped_replicates(draws, "percentile") == 1
    assert dropped_replicates(draws, "bca") == 1
    assert dropped_replicates(draws, "bootstrap_t") == 3
