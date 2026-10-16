import dataclasses
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import t as student_t

from zero_field_refusal.analysis import study2_short_sim as sim
from zero_field_refusal.analysis.study2_short_sim import (
    BRACKETING,
    CAP,
    FAMILIES,
    NOT_STARTED,
    POSITIONS,
    SUPPORTED,
    TECHNICAL,
    WIDTH,
    Parameters,
    Replicates,
    Scenario,
    calibrate_spread,
    calibration_key,
    draw_constants,
    interpolate_at_zero,
    look_statistics,
    obf_levels,
    paired_sd,
    sequential_decisions,
    simulate_replicates,
    solve_parameters,
    t_interval,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "study2_short_simulate.py"
NONE_FAILED = np.zeros((1, 3), dtype=bool)


def interpolate(x, h, failed=NONE_FAILED, w_max=math.inf):
    estimate, cause = interpolate_at_zero(np.array([x]), np.array([h]), np.array(failed), w_max)
    return float(estimate[0]), int(cause[0])


def test_interpolation_uses_the_nearest_bracketing_endpoints() -> None:
    # Nearest below is −0.1, nearest at or above is 0.2: 0.2 + 0.3 · 0.1/0.3.
    estimate, cause = interpolate([-0.3, -0.1, 0.2], [0.1, 0.2, 0.5])
    assert estimate == pytest.approx(0.3) and cause == SUPPORTED
    # The order of the endpoints does not matter.
    assert interpolate([0.2, -0.3, -0.1], [0.5, 0.1, 0.2])[0] == pytest.approx(0.3)
    # An endpoint exactly at K* is the upper end: its harm is the estimate.
    assert interpolate([-0.2, 0.0, 0.2], [0.1, 0.15, 0.3])[0] == pytest.approx(0.15)
    # A failed endpoint is skipped: the bracket becomes (−0.3, 0.2).
    failed = [[False, True, False]]
    assert interpolate([-0.3, -0.1, 0.2], [0.1, 0.2, 0.5], failed)[0] == pytest.approx(
        0.1 + 0.4 * 0.6
    )


def test_support_causes_bracketing_technical_and_width() -> None:
    estimate, cause = interpolate([-0.3, -0.2, -0.1], [0.1, 0.1, 0.1])
    assert math.isnan(estimate) and cause == BRACKETING
    # Bracketing comes first, even when runs also failed.
    assert interpolate([-0.3, -0.2, -0.1], [0.1] * 3, [[True, True, False]])[1] == BRACKETING
    # The only endpoint below K* failed.
    assert interpolate([-0.2, 0.05, 0.25], [0.1] * 3, [[True, False, False]])[1] == TECHNICAL
    # A failed endpoint that is not needed does not matter.
    assert interpolate([-0.2, 0.05, 0.25], [0.1] * 3, [[False, False, True]])[1] == SUPPORTED
    # Without the middle endpoint the bracket is the outer pair, 0.47 wide.
    middle_failed = [[False, True, False]]
    nominal = list(POSITIONS["nominal"][0])
    assert interpolate(nominal, [0.1] * 3, middle_failed, w_max=0.35)[1] == WIDTH
    assert interpolate(nominal, [0.1] * 3, middle_failed, w_max=0.48)[1] == SUPPORTED
    assert interpolate(nominal, [0.1] * 3, w_max=0.30)[1] == SUPPORTED  # 0.235 wide


def test_a_block_supports_the_primary_contrast_when_a_and_f_are_supported() -> None:
    estimates = np.full((1, CAP, 3), np.nan)
    causes = np.full((1, CAP, 3), NOT_STARTED, dtype=np.int8)
    rows = [  # causes of A, F, P per started block
        (SUPPORTED, SUPPORTED, SUPPORTED),
        (SUPPORTED, SUPPORTED, WIDTH),
        (WIDTH, TECHNICAL, SUPPORTED),
        (SUPPORTED, BRACKETING, SUPPORTED),
        (WIDTH, SUPPORTED, SUPPORTED),
    ]
    for b, row in enumerate(rows):
        causes[0, b] = row
        estimates[0, b] = [0.2 if c == SUPPORTED else np.nan for c in row]
    data = Replicates(estimates, causes, np.array([len(rows)]))
    assert np.isfinite(data.primary[0, :5]).tolist() == [True, True, False, False, False]
    # The secondary contrasts need all three arms.
    assert np.isfinite(data.secondary[0, :5, 0]).tolist() == [True, False, False, False, False]
    assert np.isfinite(data.secondary[0, :5, 1]).tolist() == [True, False, False, False, False]
    truth = {"A-F": 0.0, "P-F": 0.0, "A-P": 0.0}
    flat = ((0.0, 0.0, 0.0),) * 3
    params = Parameters(
        (0.0, 0.0, 0.0), flat, (0.0, 0.0, 0.0), 0.0, (True, True, True), (-math.inf, 0.0), truth
    )
    diagnostics = sim.block_diagnostics(params, data)
    per_block = diagnostics["unsupported_per_block"]
    # Per block the first cause in order counts: block 3 is technical, not width.
    assert per_block == pytest.approx(
        {"bracketing": 0.2, "technical": 0.2, "width": 0.2, "any": 0.6}
    )
    assert diagnostics["unsupported_per_arm"]["A"]["width"] == pytest.approx(0.4)
    assert diagnostics["unsupported_per_arm"]["P"]["width"] == pytest.approx(0.2)
    assert diagnostics["three_arm_attrition"] == pytest.approx(0.5)


def sequence(values: list[float], gaps: tuple[int, ...] = ()) -> np.ndarray:
    """Started-order Δ̂_b of one replicate, unsupported (NaN) at the ``gaps`` and past the end."""
    row = np.full(CAP, np.nan)
    supported = [b for b in range(CAP) if b not in gaps]
    for b, value in zip(supported, values, strict=False):
        row[b] = value
    return row[None, :]


def test_looks_happen_at_the_supported_minimum_postponed_up_to_the_cap() -> None:
    values = list(np.linspace(-0.05, 0.05, 80))
    stats = look_statistics(sequence(values, gaps=(3, 10)))
    assert stats.reached.tolist() == [[True, True]]
    assert stats.n.tolist() == [[24, 50]]
    assert stats.started.tolist() == [[26, 52]]  # two unsupported blocks before each look
    assert stats.mean[0, 0] == pytest.approx(np.mean(values[:24]))
    assert stats.sd[0, 1] == pytest.approx(np.std(values[:50], ddof=1))
    assert not stats.at_cap[0]
    # A later first look (postponed to 30 supported).
    assert look_statistics(sequence(values), sizes=(30, 50)).started.tolist() == [[30, 50]]
    # Only 45 supported blocks by the cap: the terminal look happens at the cap.
    capped = look_statistics(sequence(values, gaps=tuple(range(40, 75))))
    assert capped.n.tolist() == [[24, 45]] and capped.started.tolist() == [[24, CAP]]
    assert capped.at_cap[0]
    # Fewer than 24 by the cap: no look 1, one terminal look with whatever is supported.
    sparse = look_statistics(sequence(values, gaps=tuple(range(20, 80))))
    assert sparse.reached.tolist() == [[False, True]]
    assert sparse.n[0, 1] == 20 and sparse.started[0, 1] == CAP


def test_look_1_stops_on_equivalence_or_direction_otherwise_look_2_decides() -> None:
    noisy = [0.11, -0.11] * 12  # mean 0, too wide for equivalence at 99.53 %
    calm = noisy + [0.0] * 26  # look 2: equivalence
    shifted = noisy + [0.06] * 26  # look 2: direction
    clear = [0.01, -0.01] * 40  # equivalence at look 1
    stats = look_statistics(np.concatenate([sequence(v) for v in (calm, shifted, clear)]))
    d = sequential_decisions(stats, sim.PLANNING_LEVELS)
    assert d.look.tolist() == [2, 2, 1]
    assert d.equivalence.tolist() == [True, False, True]
    assert d.positive.tolist() == [False, True, False]
    assert d.inconclusive.tolist() == [False, False, False]
    assert d.n.tolist() == [50, 50, 24]
    # The ±8 reading never stops: a look-1 interval inside ±8 but not ±5 goes on.
    low1, high1 = t_interval(stats.mean[0, 0], stats.sd[0, 0], 24, sim.PLANNING_LEVELS[0])
    assert -0.08 < low1 < -0.05 and 0.05 < high1 < 0.08


def test_equal_block_differences_and_the_se_floor() -> None:
    equal = look_statistics(sequence([0.04] * 80))
    collapsed = sequential_decisions(equal, sim.PLANNING_LEVELS, floor=0.0)
    assert collapsed.low[0] == pytest.approx(0.04) and collapsed.high[0] == pytest.approx(0.04)
    assert collapsed.look[0] == 1 and collapsed.equivalence[0] and collapsed.positive[0]
    floored = sequential_decisions(equal, sim.PLANNING_LEVELS, floor=0.02)
    half = student_t.ppf(1 - (1 - sim.PLANNING_LEVELS[0]) / 2, 23) * 0.02 / math.sqrt(24)
    assert floored.low[0] == pytest.approx(0.04 - half)
    assert floored.high[0] == pytest.approx(0.04 + half)
    assert floored.positive[0] and not floored.equivalence[0]
    # All zero: no floor gives the collapsed interval [0, 0] (equivalence, no direction).
    zero = sequential_decisions(look_statistics(sequence([0.0] * 80)), sim.PLANNING_LEVELS)
    assert zero.low[0] == 0 and zero.high[0] == 0
    assert zero.equivalence[0] and not (zero.positive[0] or zero.negative[0])


def test_t_interval_is_unbounded_with_fewer_than_two_blocks() -> None:
    low, high = t_interval(np.array([0.01, 0.01]), np.array([np.nan, 0.02]), np.array([1, 2]), 0.95)
    assert low[0] == -math.inf and high[0] == math.inf
    assert math.isfinite(low[1]) and math.isfinite(high[1])


def test_secondary_intervals_are_read_at_the_stopping_look() -> None:
    primary = sequence([0.01, -0.01] * 40)
    secondary = np.stack([primary, primary + 0.02], axis=-1)
    secondary[0, 0] = np.nan  # P unsupported in the first block
    stats = look_statistics(primary, secondary)
    d = sequential_decisions(stats, sim.PLANNING_LEVELS)
    assert d.look[0] == 1 and d.secondary_n[0] == 23
    low, high = t_interval(
        stats.secondary_mean[0, 0], stats.secondary_sd[0, 0], 23, sim.SECONDARY_LEVEL
    )
    assert d.secondary_low[0].tolist() == pytest.approx(low.tolist())
    assert d.secondary_high[0].tolist() == pytest.approx(high.tolist())


def decide(rows: list[np.ndarray], design: str, alpha: float = 0.05, postpone: int = 0):
    d = sim.DESIGNS_BY_NAME[design]
    stats = look_statistics(np.concatenate(rows), sizes=d.realized(postpone))
    return sequential_decisions(stats, d.levels(alpha), equivalence_at_first=d.equivalence_at_first)


def test_look_designs_one_look_later_looks_and_equivalence_only_at_the_end() -> None:
    clear = sequence([0.01, -0.01] * 40)  # equivalence at any look
    shifted = sequence([0.03 + 0.002 * (-1) ** b for b in range(80)])  # inside ±5 and positive
    # One look at 50: no look 1.
    one = decide([clear, shifted], "50")
    assert one.look.tolist() == [2, 2] and one.n.tolist() == [50, 50]
    assert one.equivalence.tolist() == [True, True] and one.positive.tolist() == [False, True]
    # 36/60: look 1 at the 36th supported block.
    later = decide([clear], "36_60")
    assert later.look.tolist() == [1] and later.n.tolist() == [36]
    # 24/50 stops at look 1 on either reading; with equivalence only at the terminal look, the
    # equivalent study goes on to look 2, and a direction stop at look 1 reports no equivalence.
    planning = decide([clear, shifted], "24_50")
    assert planning.look.tolist() == [1, 1]
    assert planning.equivalence.tolist() == [True, True]
    terminal = decide([clear, shifted], "24_50_eq_terminal")
    assert terminal.look.tolist() == [2, 1] and terminal.n.tolist() == [50, 24]
    assert terminal.equivalence.tolist() == [True, False]
    assert terminal.positive.tolist() == [False, True]
    # A postponed look 1 happens later; the one-look design ignores it.
    assert decide([clear], "24_50", postpone=6).n.tolist() == [30]
    assert decide([clear], "50", postpone=6).n.tolist() == [50]
    assert sim.DESIGNS_BY_NAME["50"].levels(0.05) == (0.95,)


def test_pointwise_readings_use_the_bias_allowance() -> None:
    # Final intervals [0.4, 1.0] pp and [−4.7, −0.2] pp, with b = 0.5 pp.
    stats = look_statistics(np.concatenate([sequence([0.007] * 80), sequence([-0.0245] * 80)]))
    d = sequential_decisions(stats, sim.PLANNING_LEVELS)
    d = sim.Decisions(
        **{**vars(d), "low": np.array([0.004, -0.047]), "high": np.array([0.01, -0.002])}
    )
    readings = sim.pointwise_readings(d, 0.005)
    assert readings["positive"].tolist() == [False, False]  # 0.4 pp does not clear b
    assert readings["negative"].tolist() == [False, False]  # −0.2 pp does not clear −b
    assert readings["equivalence"].tolist() == [True, False]  # −4.7 pp lies outside ±4.5 pp


def test_bias_allowance_is_the_largest_pointwise_bias_at_w_max_035_and_047() -> None:
    def summary(family: str, w_max: float, bias: float, **changes) -> dict:
        scenario = Scenario(family=family, w_max=w_max, **changes)
        return {
            "name": scenario.name,
            "settings": dataclasses.asdict(scenario),
            "diagnostics": {"block_bias": bias, "block_bias_se": 0.0001},
        }

    scenarios = [
        summary("study1", 0.35, 0.0002),
        summary("kink", 0.35, 0.006, miss="common"),
        summary("kink", 0.47, -0.007, delta=0.05),
        summary("stress_slopes", 0.30, 0.02),  # w_max 0.30 is reported, not counted
        summary("high", 0.35, 0.05, failure=0.2),  # not a bias scenario: other settings differ
    ]
    found = sim.bias_allowance(scenarios)
    assert [entry["w_max"] for entry in found["per_w_max"]] == [0.30, 0.35, 0.47]
    b = found["allowance"]
    assert b["value"] == pytest.approx(0.007) and b["family"] == "kink" and b["w_max"] == 0.47


PROBABILITIES_95 = np.array([0.025, 0.975])


def misses(draw, n: int, samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Share of 95 % intervals below (high < 0) and above (low > 0) a true mean of 0: t, then
    bootstrap-t."""
    rng = np.random.default_rng(seed)
    t_miss, boot_miss = np.zeros(2), np.zeros(2)
    for _ in range(samples):
        x = draw(rng, n)
        mean, sd = x.mean(), x.std(ddof=1)
        low, high = t_interval(np.array([mean]), np.array([sd]), np.array([n]), 0.95)
        t_miss += [high[0] < 0, low[0] > 0]
        q, _ = sim.bootstrap_t_quantiles(x, PROBABILITIES_95, (0.0,), 999, rng)
        se = sd / math.sqrt(n)
        boot_miss += [mean - q[0, 0] * se < 0, mean - q[0, 1] * se > 0]
    return t_miss / samples, boot_miss / samples


def test_bootstrap_t_matches_t_coverage_on_normal_data() -> None:
    t_miss, boot_miss = misses(lambda rng, n: rng.standard_normal(n), 30, 2000, seed=1)
    se = math.sqrt(2 * 0.05 * 0.95 / 2000)  # MC SE of a coverage difference, at most
    assert 1 - boot_miss.sum() == pytest.approx(1 - t_miss.sum(), abs=3 * se)
    assert 1 - boot_miss.sum() == pytest.approx(0.95, abs=3 * se)


def test_bootstrap_t_corrects_the_left_tail_of_right_skewed_differences() -> None:
    # Exp(1) − 1: mean 0, skewness 2. The t interval's upper end falls below the mean too often
    # (a false negative direction); the bootstrap-t brings that side back towards 2.5 %.
    t_miss, boot_miss = misses(lambda rng, n: rng.exponential(1.0, n) - 1.0, 24, 4000, seed=2)
    assert t_miss[0] > 0.05
    assert boot_miss[0] < t_miss[0] - 0.02 and boot_miss[0] < 0.045
    assert abs(boot_miss[0] - boot_miss[1]) < abs(t_miss[0] - t_miss[1])


def test_bootstrap_t_floor_drops_and_collapsed_intervals() -> None:
    rng = np.random.default_rng(0)
    equal = np.full(24, 0.04)
    q, dropped = sim.bootstrap_t_quantiles(equal, PROBABILITIES_95, (0.0, 0.01), 500, rng)
    assert dropped.tolist() == [500, 0]  # se* = 0: dropped without a floor, floored otherwise
    assert np.isnan(q[0]).all() and np.isfinite(q[1]).all()
    # The look's interval is [mean, mean] without a floor (zero se), and also with one: every
    # resample has mean* = mean, so T* ≡ 0. The floor widens the t interval of such a sample, not
    # the bootstrap-t (it cannot recover variation the sample does not show).
    design = sim.DESIGNS_BY_NAME["50"]
    stats = look_statistics(sequence([0.04] * 80), sizes=design.sizes)
    quantiles = {"50:1": np.stack([q])}  # (1 replicate, 2 floors, 2 probabilities)
    probabilities = {"50:1": PROBABILITIES_95}
    flat = sim.bootstrap_bounds(stats, design, (0.95,), 0, 0.0, quantiles, probabilities)
    assert flat[0] is None and flat[1][0][0] == pytest.approx(0.04) == flat[1][1][0]
    floored = sim.bootstrap_bounds(stats, design, (0.95,), 1, 0.01, quantiles, probabilities)
    assert floored[1][0][0] == pytest.approx(0.04) == floored[1][1][0]
    low, high = t_interval(stats.mean[:, 1], stats.sd[:, 1], stats.n[:, 1], 0.95, 0.01)
    assert low[0] < 0.04 < high[0]
    # Two distinct values: the floor raises se* where a resample happens to be constant.
    two = np.array([0.0] * 23 + [0.1])
    q, dropped = sim.bootstrap_t_quantiles(two, PROBABILITIES_95, (0.0, 0.05), 2000, rng)
    assert dropped[0] > 0 and dropped[1] == 0 and np.isfinite(q).all()
    # Sorted-array quantiles agree with numpy's.
    values = np.sort(rng.standard_normal(301))
    probs = np.array([0.0, 0.0012, 0.025, 0.5, 0.975, 1.0])
    np.testing.assert_allclose(sim._sorted_quantiles(values, probs), np.quantile(values, probs))


def test_bootstrap_tables_are_reproducible_per_replicate() -> None:
    scenario = Scenario(delta=0.05)
    params = solve_parameters(scenario, 0.3)
    data = simulate_replicates(scenario, params, range(3), base_seed=0, stream="test")
    designs = (sim.DESIGNS_BY_NAME["24_50"],)
    kwargs = dict(
        designs=designs, alphas=(0.05,), floors=(0.0,), resamples=200, base_seed=0, stream="test"
    )
    whole, _ = sim.bootstrap_tables(scenario, data.primary, range(3), **kwargs)
    last, _ = sim.bootstrap_tables(scenario, data.primary[2:], range(2, 3), **kwargs)
    np.testing.assert_array_equal(whole["24_50:1"][2:], last["24_50:1"])
    assert set(whole) == {"24_50:0", "24_50:1"}


def test_bootstrap_resamples_do_not_depend_on_the_other_designs() -> None:
    scenario = Scenario(delta=0.05)
    params = solve_parameters(scenario, 0.3)
    data = simulate_replicates(scenario, params, range(2), base_seed=0, stream="test")
    kwargs = dict(alphas=(0.04,), floors=(0.0,), resamples=200, base_seed=0, stream="test")
    both, _ = sim.bootstrap_tables(
        scenario,
        data.primary,
        range(2),
        designs=(sim.DESIGNS_BY_NAME["24_50"], sim.DESIGNS_BY_NAME["36_60"]),
        **kwargs,
    )
    alone, _ = sim.bootstrap_tables(
        scenario, data.primary, range(2), designs=(sim.DESIGNS_BY_NAME["36_60"],), **kwargs
    )
    for key in ("36_60:0", "36_60:1"):
        np.testing.assert_array_equal(both[key], alone[key])


def test_streams_are_seeded_from_the_full_digest() -> None:
    a, b = sim.seed_sequence(0, "validation", "x", 1), sim.seed_sequence(0, "validation", "x", 1)
    assert np.array_equal(a.entropy, b.entropy) and a.entropy.size == 8  # 256 bits
    assert not np.array_equal(sim.seed_sequence(0, "validation", "x", 2).entropy, a.entropy)
    first = sim.generator(0, "s", 1).random(3)
    np.testing.assert_array_equal(first, sim.generator(0, "s", 1).random(3))


def test_heavy_shape_is_a_symmetric_scale_mixture_calibrated_to_the_target_sd() -> None:
    values, weights = sim.effect_distribution("heavy", 0.4)
    assert weights.sum() == pytest.approx(1.0)
    assert weights @ values == pytest.approx(0.0, abs=1e-12)
    assert weights @ values**2 == pytest.approx(0.16)  # E[S²] = 1
    assert (weights @ values**4) / 0.16**2 - 3 == pytest.approx(5.33, abs=0.01)
    # Both arms carry it, with one scale per block.
    params = solve_parameters(Scenario(shape="heavy"), 0.4)
    assert params.shifted == (True, True, True) and params.truth["A-F"] == pytest.approx(0.0)
    latent = np.ones((4, 2))
    tail = np.array([0.05, 0.5, 0.05, 0.9])
    effect = sim.block_effects(Scenario(shape="heavy"), params, latent, (0, 1), tail)
    np.testing.assert_allclose(effect[:, 0], effect[:, 1])
    np.testing.assert_allclose(effect[[0, 2], 0] / effect[1, 0], sim.HEAVY_RATIO)
    # Its spread hits the target paired SD on fresh draws.
    scenario = Scenario(shape="heavy", sd=0.05)
    calibration = calibrate_spread(scenario, base_seed=0, replicates=20, blocks=100)
    assert calibration.attained_sd == pytest.approx(0.05, abs=5e-4)
    fresh = paired_sd(
        calibration_key(scenario), calibration.spread, seed=999, replicates=40, blocks=100
    )
    assert fresh == pytest.approx(0.05, abs=4e-3)


def test_equivalence_rule_at_a_cap_look() -> None:
    # 30 supported blocks by the cap: one look at the cap, below 36.
    gaps = tuple(range(30, 80))
    calm = sequence([0.01, -0.01] * 40, gaps=gaps)  # equivalence, no direction
    clear = sequence([0.03 + 0.002 * (-1) ** b for b in range(80)], gaps=gaps)  # both readings
    design = sim.DESIGNS_BY_NAME["36_60"]
    stats = look_statistics(np.concatenate([calm, clear]), sizes=design.sizes)
    assert stats.reached[:, 0].tolist() == [False, False] and stats.n[:, 1].tolist() == [30, 30]
    levels = design.levels(0.04)
    standard = sequential_decisions(stats, levels)
    assert standard.equivalence.tolist() == [True, True] and standard.at_cap.tolist() == [True] * 2
    ruled = sequential_decisions(stats, levels, equivalence_min_blocks=36)
    assert ruled.equivalence.tolist() == [False, False]  # not concluded below 36 blocks
    assert ruled.positive.tolist() == [False, True]  # a direction still is
    assert ruled.inconclusive.tolist() == [True, False]
    # At a full look 1 (36 blocks) the rule changes nothing.
    full = look_statistics(sequence([0.01, -0.01] * 40), sizes=design.sizes)
    assert sequential_decisions(full, levels, equivalence_min_blocks=36).look.tolist() == [1]
    assert sequential_decisions(full, levels, equivalence_min_blocks=36).equivalence.tolist() == [
        True
    ]


def test_pointwise_equivalence_respects_the_equivalence_rule() -> None:
    # 30 supported blocks by the cap, tightly around 0: the interval lies inside ±(5 − b).
    design = sim.DESIGNS_BY_NAME["36_60"]
    calm = sequence([0.004, -0.004] * 40, gaps=tuple(range(30, 80)))
    clear = sequence([0.03 + 0.002 * (-1) ** b for b in range(80)], gaps=tuple(range(30, 80)))
    stats = look_statistics(np.concatenate([calm, clear]), sizes=design.sizes)
    levels = design.levels(0.04)
    standard = sim.pointwise_readings(sequential_decisions(stats, levels), 0.01)
    ruled_decisions = sequential_decisions(stats, levels, equivalence_min_blocks=36)
    ruled = sim.pointwise_readings(ruled_decisions, 0.01)
    assert standard["equivalence"].tolist() == [True, True]
    assert ruled["equivalence"].tolist() == [False, False]  # not at a cap look below 36 blocks
    assert ruled["positive"].tolist() == [False, True] == standard["positive"].tolist()
    assert ruled_decisions.equivalence_allowed.tolist() == [False, False]
    # A full look keeps equivalence allowed, so the readings agree with the standard rule.
    full = look_statistics(sequence([0.004, -0.004] * 40), sizes=design.sizes)
    decisions = sequential_decisions(full, levels, equivalence_min_blocks=36)
    assert sim.pointwise_readings(decisions, 0.01)["equivalence"].tolist() == [True]


def test_gate_prescreen_keeps_the_passing_studies_unchanged() -> None:
    scenario = Scenario(failure=0.1, failure_correlated=True)
    params = solve_parameters(scenario, 0.3)
    kwargs = dict(base_seed=0, stream="test")
    everything = simulate_replicates(scenario, params, range(12), **kwargs)
    screened = simulate_replicates(scenario, params, range(12), prescreen=True, **kwargs)
    passed = np.flatnonzero(everything.gate_passed)
    assert screened.rep_ids.tolist() == passed.tolist() and screened.screened == 12
    np.testing.assert_array_equal(screened.estimates, everything.estimates[passed])
    np.testing.assert_array_equal(screened.gate_supported, everything.gate_supported[passed])
    gate = sim._gate_failure(screened)
    assert gate["p_gate_fail"] == pytest.approx(1 - passed.size / 12)
    assert gate["gates_drawn"] == 12 and gate["gates_passed"] == passed.size


def test_decided_grids() -> None:
    for grid in (sim.decided_bias_grid(), sim.decided_grid(), sim.decided_gate_grid()):
        names = [s.name for s in grid]
        assert len(set(names)) == len(names)
        assert all(s.w_max == 0.47 for s in grid)
    assert len(sim.decided_bias_grid()) == 4 * 3 * 2
    decided = sim.decided_grid()
    assert len(decided) == 2 * 3 + (4 * 4 + 1) * 3 + 2 * 2 * 3
    assert {s.reps for s in decided if s.group == "cap"} == {10_000}
    gates = sim.decided_gate_grid()
    assert {(s.failure_correlated, s.reps) for s in gates} == {(True, 150_000), (False, 400_000)}
    assert all(abs(s.delta) == 0.05 for s in gates)


FIXATION = Path(__file__).resolve().parents[1] / "docs" / "study2-short-fixation.json"


def test_the_fixation_record_is_the_simulations_decided_design() -> None:
    """The committed fixation record, the analysis's constants and the simulation's decided
    design agree, so that the confirmation runs the procedure that was calibrated."""
    from zero_field_refusal.analysis import study2_short

    raw = json.loads(FIXATION.read_text())
    record = study2_short.read_fixation(FIXATION)
    decided = sim.DESIGNS_BY_NAME[sim.DECIDED_DESIGN]
    assert record["w_max"] == sim.DECIDED_W_MAX
    fraction = decided.sizes[0] / decided.sizes[1]
    assert raw["levels"] == list(obf_levels(sim.DECIDED_ALPHA, fraction))  # bit for bit
    assert record["se_floor"] == sim.DECIDED_SE_FLOOR
    assert record["bias_allowance"] == sim.DECIDED_ALLOWANCE
    assert record["interval"] == "bootstrap_t" and "bootstrap_t" in sim.INTERVALS
    assert record["bootstrap_resamples"] >= sim.BOOTSTRAP_RESAMPLES
    assert tuple(record["looks"]) == decided.sizes == study2_short.LOOK_SIZES
    assert record["cap"] == CAP == study2_short.CAP
    assert record["equivalence_min_supported"] == sim.RULE_MIN_BLOCKS[sim.DECIDED_RULE]
    gate = (len(study2_short.VALIDATION_SEEDS), study2_short.VALIDATION_MINIMUM)
    assert gate == (sim.GATE_BLOCKS, sim.GATE_MIN_SUPPORTED)
    assert study2_short.SECONDARY_LEVEL == sim.SECONDARY_LEVEL
    assert (study2_short.EQUIVALENCE, study2_short.NESTED) == (sim.MARGIN, sim.WIDE_MARGIN)
    assert study2_short.ZERO_SE == sim.ZERO_SE


def test_the_decided_modes_default_to_the_decided_design() -> None:
    """Without flags the revision-5 modes run revision 5's settings (its summary.json files)."""
    script = load_script()

    def settings(*argv: str) -> sim.RunSettings:
        return script.run_settings(script.build_parser().parse_args(list(argv)), {})

    for mode in script.DECIDED_MODES:
        found = settings(mode)
        assert (found.stream, found.designs, found.intervals, found.resamples) == (
            "confirmation", ("36_60",), ("bootstrap_t",), 2000
        )  # fmt: skip
        assert (found.alphas, found.floors, found.rules) == (
            (0.04,), (0.0,), ("standard", "eq_min36")
        )  # fmt: skip
        assert found.allowance == (None if mode == "decided-bias" else sim.DECIDED_ALLOWANCE)
        assert found.prescreen == (mode == "decided-gate")
        assert found.chunk == (5000 if mode == "decided-gate" else 250)
    # Flags still override, and the other modes keep their own defaults.
    assert settings("decided", "--intervals", "t", "--allowance", "0.01").intervals == ("t",)
    assert settings("decided", "--alphas", "0.05").alphas == (0.04, 0.05)
    other = settings("validate")
    assert other.designs == tuple(d.name for d in sim.LOOK_DESIGNS)
    assert (other.intervals, other.rules, other.alphas) == (("t",), ("standard",), (0.05,))
    assert other.floors == sim.SE_FLOORS and other.allowance is None and other.chunk == 250


def test_validation_gate_counts_jointly_supported_blocks() -> None:
    params = solve_parameters(Scenario(), 0.3)
    data = simulate_replicates(Scenario(), params, range(20), base_seed=0, stream="test")
    assert ((data.gate_supported >= 0) & (data.gate_supported <= sim.GATE_BLOCKS)).all()
    # No technical failures, but width failures cluster within a replicate (fixed misses).
    assert data.gate_passed.mean() > 0.6
    failing = Scenario(failure=0.2, failure_correlated=True)
    params = solve_parameters(failing, 0.3)
    poor = simulate_replicates(failing, params, range(20), base_seed=0, stream="test")
    assert poor.gate_supported.mean() < data.gate_supported.mean()
    # The gate has its own stream: the study's blocks are those of a run without it.
    rows = sim.summarize_scenario(
        failing, params, poor, (0.04,), (0.0,), designs=(sim.DESIGNS_BY_NAME["36_60"],)
    )
    assert rows["diagnostics"]["p_gate_fail"] == pytest.approx(1 - poor.gate_passed.mean())
    assert rows["results"][0]["gated"]["passed"] == int(poor.gate_passed.sum())


def test_revision_4_grids() -> None:
    stress = sim.stress_validation_grid()
    names = [s.name for s in stress]
    assert len(set(names)) == len(names)
    assert {s.group for s in stress} == {
        "postponed", "cap", "width", "k_spread", "miss", "families", "stress", "shapes"
    }  # fmt: skip
    cap = [s for s in stress if s.group == "cap"]
    assert {(s.failure_correlated, s.sd, s.delta) for s in cap} == {
        (c, sd, d) for c in (True, False) for sd in (0.05, 0.10) for d in (0.0, 0.05, -0.05)
    }
    assert {s.shape for s in stress if s.group == "shapes"} == {"heavy"}
    assert all(s.reps == 20_000 for s in stress if s.delta == 0 or abs(s.delta) == 0.05)
    confirm = sim.confirmation_grid()
    assert len(confirm) == (4 * 4 + 1) * 3 and all(s.reps == 20_000 for s in confirm)


def test_obf_spending_at_alpha_5_percent_gives_the_planning_levels() -> None:
    level1, level2 = obf_levels(0.05)
    assert level1 == pytest.approx(sim.PLANNING_LEVELS[0], abs=5e-5)
    assert level2 == pytest.approx(sim.PLANNING_LEVELS[1], abs=5e-5)
    assert obf_levels(0.04)[1] > level2  # smaller α, wider intervals


def population_delta_by_monte_carlo(scenario: Scenario, params: Parameters, n: int = 2_000_000):
    rng = np.random.default_rng(1)
    v = rng.normal(0, math.hypot(scenario.sigma_alpha, scenario.sigma_s), (n, 1))
    latent = rng.standard_normal((n, 3))
    effect = sim.block_effects(scenario, params, latent, (0, 1, 2))
    h = expit(np.array(params.offsets) + v + effect).mean(axis=0)
    return h[0] - h[1], h[1]


@pytest.mark.parametrize(
    ("shape", "spread", "delta"),
    [("normal", 0.4, 0.05), ("skewed", 0.8, -0.05), ("rare_centered", 4.0, 0.0)],
)
def test_offsets_give_the_population_delta_and_baseline(shape, spread, delta) -> None:
    scenario = Scenario(shape=shape, delta=delta)
    params = solve_parameters(scenario, spread)
    assert params.truth["A-F"] == pytest.approx(delta, abs=1e-9)
    mc_delta, mc_f = population_delta_by_monte_carlo(scenario, params)
    assert mc_delta == pytest.approx(delta, abs=1.5e-3)
    assert mc_f == pytest.approx(0.15, abs=1.5e-3)
    assert params.truth["P-F"] == pytest.approx(0.0, abs=1e-12)


def test_rare_only_puts_the_whole_difference_in_rare_blocks() -> None:
    up = solve_parameters(Scenario(shape="rare_only", sd=0.0, delta=0.05))
    assert up.shifted == (True, False, False) and up.offsets[0] == up.offsets[1]
    assert population_delta_by_monte_carlo(Scenario(shape="rare_only"), up)[0] == pytest.approx(
        0.05, abs=1.5e-3
    )
    down = solve_parameters(Scenario(shape="rare_only", sd=0.0, delta=-0.05))
    assert down.shifted == (False, True, True) and down.truth["A-F"] == pytest.approx(-0.05)
    assert solve_parameters(Scenario(shape="rare_only", sd=0.0)).spread == 0.0
    with pytest.raises(ValueError, match="cannot reach"):
        solve_parameters(Scenario(shape="rare_only", family="high", sd=0.0, delta=0.08))


def test_study1_targets_are_arm_specific_population_curves() -> None:
    study1 = FAMILIES["study1"]
    x = np.array([-0.3, 0.0, 0.3])
    np.testing.assert_allclose(study1.target(0, x), expit(logit(0.15) + 1.20 * x - 0.13 * x**2))
    np.testing.assert_allclose(study1.target(1, x), expit(logit(0.15) + 1.39 * x - 0.24 * x**2))
    assert FAMILIES["high"].slope == study1.slope and FAMILIES["high"].baseline == 0.40
    # Without prompt and block heterogeneity the fit returns the target's own coefficients.
    single = Scenario(sigma_alpha=0.0, sigma_s=0.0)
    params = solve_parameters(single, 0.0)
    np.testing.assert_allclose(params.coefficients[0], (1.20, -0.13, 0.0), atol=1e-6)
    np.testing.assert_allclose(params.coefficients[1], (1.39, -0.24, 0.0), atol=1e-6)


@pytest.mark.parametrize("shape", ["normal", "rare_centered"])
def test_population_curves_match_the_planning_values(shape: str) -> None:
    # Heterogeneity flattens the population curve, so the per-prompt coefficients are steeper;
    # the population curve has the planning h̄′ 0.153 / 0.177 and h̄″ 0.096 / 0.112 at h̄ = 0.15.
    scenario = Scenario(shape=shape)
    params = solve_parameters(scenario, 0.34 if shape == "normal" else 1.3)
    curve = sim.population_curve(scenario, params)
    for arm, slope, curvature in (("A", 0.153, 0.095), ("F", 0.177, 0.111)):
        attained = curve[arm]["attained"]
        assert attained["h"] == pytest.approx(0.15, abs=1e-9)
        assert attained["slope"] == pytest.approx(slope, abs=2e-3)
        assert attained["curvature"] == pytest.approx(curvature, abs=5e-3)
        assert curve[arm]["fit_rmse"] < 1e-3
    assert params.coefficients[0][0] > 1.20 * 1.2  # steeper per prompt than in the population
    # The stress pair: population logistics with slopes 4 and 2.
    stress = Scenario(family="stress_slopes")
    curve = sim.population_curve(stress, solve_parameters(stress, 0.34))
    assert curve["A"]["attained"]["slope"] == pytest.approx(0.15 * 0.85 * 4, abs=3e-3)
    assert curve["F"]["attained"]["slope"] == pytest.approx(0.15 * 0.85 * 2, abs=3e-3)


def test_selection_miss_modes() -> None:
    rng = np.random.default_rng(0)
    _, common = draw_constants(Scenario(miss="common"), rng)
    assert np.unique(common).size == 1 and abs(common[0, 0]) <= 0.05
    _, arm = draw_constants(Scenario(miss="arm"), rng)
    assert (arm == arm[:, :1]).all() and np.unique(arm[:, 0]).size == 3
    _, rate = draw_constants(Scenario(miss="rate", tau=0.1), rng)
    assert np.unique(rate).size == 9 and np.abs(rate).max() <= 0.1


def test_coarse_discovery_brackets_reproduce_the_reviewers_spurious_contrast() -> None:
    # fa2: curves equal at K* (h = 0.15) with slopes 4 and 2, interpolated at the median brackets.
    c = logit(0.15)
    estimates = []
    for arm, slope in ((0, 4.0), (1, 2.0)):
        x = np.array(POSITIONS["grid"][arm])
        estimates.append(interpolate(x, expit(c + slope * x))[0])
    assert estimates[0] == pytest.approx(0.1908, abs=5e-4)
    assert estimates[1] == pytest.approx(0.1610, abs=5e-4)
    assert estimates[0] - estimates[1] == pytest.approx(0.03, abs=1e-3)


def test_spread_calibration_hits_its_target() -> None:
    scenario = Scenario(sd=0.05)
    calibration = calibrate_spread(scenario, base_seed=0, replicates=20, blocks=100)
    assert calibration.noise_sd < 0.05 and calibration.spread > 0
    assert calibration.attained_sd == pytest.approx(0.05, abs=5e-4)
    # Fresh draws at the calibrated spread: within Monte Carlo error of the target.
    fresh = paired_sd(
        calibration_key(scenario), calibration.spread, seed=12345, replicates=40, blocks=100
    )
    assert fresh == pytest.approx(0.05, abs=3.5e-3)
    # Below the noise floor the spread is 0 and the attained SD is reported.
    tiny = calibrate_spread(Scenario(sd=0.005), base_seed=0, replicates=5, blocks=40)
    assert tiny.spread == 0 and tiny.attained_sd == tiny.noise_sd > 0.005


def test_replicates_are_reproducible_and_independent_of_chunking() -> None:
    scenario = Scenario(sd=0.05, delta=0.05, failure=0.1)
    params = solve_parameters(scenario, 0.3)
    kwargs = dict(base_seed=7, stream="test")
    whole = simulate_replicates(scenario, params, range(3), **kwargs)
    again = simulate_replicates(scenario, params, range(3), **kwargs)
    parts = [simulate_replicates(scenario, params, r, **kwargs) for r in (range(1), range(1, 3))]
    np.testing.assert_array_equal(whole.estimates, again.estimates)
    np.testing.assert_array_equal(whole.causes, again.causes)
    np.testing.assert_array_equal(whole.estimates, np.concatenate([p.estimates for p in parts]))
    other = simulate_replicates(scenario, params, range(3), base_seed=7, stream="other")
    assert not np.array_equal(whole.estimates, other.estimates, equal_nan=True)
    # Blocks start until 60 (the largest look) are supported or the cap is reached, in chunks.
    supported = np.isfinite(whole.primary).sum(axis=1)
    assert sim.GENERATE_UNTIL == 60
    assert ((supported >= 60) | (whole.started == CAP)).all()
    assert (whole.started % sim.CHUNK_BLOCKS == 0).all()
    assert ((whole.causes[..., 0] != NOT_STARTED).sum(axis=1) == whole.started).all()


def test_scenario_names_carry_the_non_default_settings() -> None:
    assert Scenario().name == "interp_normal_study1_sd5_d+0"
    assert (
        Scenario(delta=-0.055, miss="arm", w_max=0.47, failure=0.2, failure_correlated=True).name
        == "interp_normal_study1_sd5_d-5.5_miss-arm_f0.2_fcorr_w0.47"
    )
    assert Scenario(shape="rare_only", sd=0.0).name == "interp_rare_only_study1_sdna_d+0"
    assert Scenario(positions="grid", w_max=math.inf).name.startswith("interp_grid_normal")
    assert Scenario(group="a") == Scenario(group="b")
    assert Scenario(postpone=6).name == "interp_normal_study1_sd5_d+0_postpone6"
    with pytest.raises(ValueError, match="postpone"):
        Scenario(postpone=24)  # look 1 would reach look 2 in 36/60 and 24/50


def test_grids_have_unique_names_and_the_required_replicates() -> None:
    for grid in (sim.calibration_grid(), sim.validation_grid(), sim.bias_grid(), sim.smoke_grid()):
        names = [s.name for s in grid]
        assert len(set(names)) == len(names)
    for s in sim.calibration_grid() + sim.validation_grid():
        boundary = s.delta == 0 or math.isclose(abs(s.delta), 0.05)
        assert s.reps == (20_000 if boundary else 5_000)
    bias = sim.bias_grid()
    assert {(s.family, s.miss, s.w_max) for s in bias if s.positions == "nominal"} == {
        (f, m, w) for f in FAMILIES for m in sim.MISSES for w in sim.W_MAXES
    }


def load_script():
    spec = importlib.util.spec_from_file_location("study2_short_simulate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_calibrates_then_validates_on_a_tiny_grid(tmp_path: Path) -> None:
    script = load_script()
    small = ["--reps", "6", "--workers", "1", "--spread-replicates", "3", "--spread-blocks", "30"]
    calibrate = tmp_path / "calibrate"
    script.main(
        ["calibrate", "--scenarios", "*_sd5_d+0", "*_sd5_d?5", *small, "--out", str(calibrate)]
    )
    summary = json.loads((calibrate / "summary.json").read_text())
    assert [s["name"] for s in summary["scenarios"]] == [
        "interp_normal_study1_sd5_d+0",
        "interp_normal_study1_sd5_d+5",
        "interp_normal_study1_sd5_d-5",
    ]
    designs = [d.name for d in sim.LOOK_DESIGNS]
    rows = summary["scenarios"][0]["results"]
    assert len(rows) == len(designs) * len(sim.ALPHAS) * len(sim.SE_FLOORS)
    # The one-look design uses 1 − α; the two-look ones OBF spending at their own fraction.
    one = next(r for r in rows if r["looks"] == "50" and r["alpha"] == 0.05)
    assert one["levels"] == pytest.approx([0.95])
    later = next(r for r in rows if r["looks"] == "36_60" and r["alpha"] == 0.05)
    assert later["levels"] == pytest.approx(list(obf_levels(0.05, 0.6)))
    # Power is the equivalence rate at Δ = 0 and the rate of the right direction otherwise.
    null, positive, negative = (s["results"] for s in summary["scenarios"])
    assert all(r["power"] == r["p_equivalence"] for r in null)
    assert all(r["power"] == r["p_positive"] for r in positive)
    assert all(r["power"] == r["p_negative"] for r in negative)
    selection = summary["selection"]
    assert [(s["looks"], s["floor"]) for s in selection] == [
        (d, f) for d in designs for f in sim.SE_FLOORS
    ]
    assert "## Selected levels" in (calibrate / "results.md").read_text()

    validate = tmp_path / "validate"
    script.main(
        ["validate", "--scenarios", "*rare_only*_d+5", "--alphas", "0.04", "--allowance", "0.005",
         "--levels-from", str(calibrate / "summary.json"), *small, "--out", str(validate)]
    )  # fmt: skip
    summary = json.loads((validate / "summary.json").read_text())
    (scenario,) = summary["scenarios"]
    assert scenario["calibration"] is None and scenario["parameters"]["spread"] > 0
    assert {r["alpha"] for r in scenario["results"]} >= {0.04, 0.05}
    assert all(r["pointwise"]["allowance"] == 0.005 for r in scenario["results"])
    check = summary["checks"][0]
    assert check["groups"]["all"]["max_false_positive"]["value"] is None
    assert "## Constraint checks" in (validate / "results.md").read_text()


def test_script_runs_bootstrap_t_beside_t(tmp_path: Path) -> None:
    script = load_script()
    small = ["--workers", "1", "--spread-replicates", "3", "--spread-blocks", "30"]
    boot = ["--looks", "24_50", "36_60", "--intervals", "t", "bootstrap_t", "--bootstrap", "200"]
    calibrate = tmp_path / "calibrate"
    script.main(
        ["calibrate", "--scenarios", "*_sd5_d+0", "*_sd5_d?5", "--reps", "6", *small, *boot,
         "--out", str(calibrate)]
    )  # fmt: skip
    summary = json.loads((calibrate / "summary.json").read_text())
    rows = summary["scenarios"][0]["results"]
    assert len(rows) == 2 * 2 * len(sim.ALPHAS) * len(sim.SE_FLOORS)
    assert {(r["looks"], r["interval"]) for r in rows} == {
        (d, i) for d in ("24_50", "36_60") for i in ("t", "bootstrap_t")
    }
    assert [(s["looks"], s["interval"]) for s in summary["selection"][::3]] == [
        ("24_50", "t"), ("24_50", "bootstrap_t"), ("36_60", "t"), ("36_60", "bootstrap_t")
    ]  # fmt: skip
    dropped = summary["scenarios"][0]["diagnostics"]["bootstrap_dropped"]
    assert set(dropped) == {"24_50:0", "24_50:1", "36_60:0", "36_60:1"}

    validate = tmp_path / "validate"
    script.main(
        ["validate-bootstrap", "--scenarios", "*skewed*_sd5_d+0", "--boundary-reps", "5",
         "--levels-from", str(calibrate / "summary.json"), *small, *boot, "--out", str(validate)]
    )  # fmt: skip
    summary = json.loads((validate / "summary.json").read_text())
    (scenario,) = summary["scenarios"]
    assert scenario["reps"] == 5 and scenario["settings"]["shape"] == "skewed"
    assert {r["interval"] for r in scenario["results"]} == {"t", "bootstrap_t"}
    assert {(c["looks"], c["interval"]) for c in summary["checks"]} == {
        (d, i) for d in ("24_50", "36_60") for i in ("t", "bootstrap_t")
    }
    assert len(sim.bootstrap_validation_grid()) == 3 * 4 * 7 + 7


def test_script_lists_and_filters_scenarios(capsys) -> None:
    load_script().main(["bias", "--designs", "interp_grid", "--list"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(FAMILIES) and all(line.startswith("interp_grid_") for line in lines)
