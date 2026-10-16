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
    params = Parameters((0.0, 0.0, 0.0), 0.0, (True, True, True), (-math.inf, 0.0), truth)
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


def test_study1_family_is_arm_specific() -> None:
    study1 = FAMILIES["study1"]
    x = np.array([[0.3], [0.3], [0.3]])
    np.testing.assert_allclose(
        study1.offset(x, (0, 1, 2))[:, 0],
        [1.20 * 0.3 - 0.13 * 0.09, 1.39 * 0.3 - 0.24 * 0.09, 1.39 * 0.3 - 0.24 * 0.09],
    )
    # Planning slopes on the probability scale at h = 0.15: h' = h(1 − h)·s.
    assert 0.15 * 0.85 * study1.slope[0] == pytest.approx(0.153, abs=5e-4)
    assert 0.15 * 0.85 * study1.slope[1] == pytest.approx(0.177, abs=5e-4)
    assert FAMILIES["high"].slope == study1.slope and FAMILIES["high"].baseline == 0.40
    # Without prompt and block heterogeneity the population curve is that single logistic:
    # h' = h(1 − h)·s and h'' = h(1 − h)·((1 − 2h)·s² + 2q), the planning values.
    single = Scenario(sigma_alpha=0.0, sigma_s=0.0)
    curve = sim.population_curve(single, solve_parameters(single, 0.0))
    assert curve["A"]["slope"] == pytest.approx(0.153, abs=5e-4)
    assert curve["F"]["slope"] == pytest.approx(0.177, abs=5e-4)
    assert curve["A"]["curvature"] == pytest.approx(0.096, abs=2e-3)
    assert curve["F"]["curvature"] == pytest.approx(0.112, abs=2e-3)
    # Heterogeneity flattens it.
    central = Scenario()
    flattened = sim.population_curve(central, solve_parameters(central, 0.3))
    assert flattened["A"]["slope"] < 0.8 * curve["A"]["slope"]


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
    # Blocks start until 50 are supported or the cap is reached, in chunks of 10.
    supported = np.isfinite(whole.primary).sum(axis=1)
    assert ((supported >= 50) | (whole.started == CAP)).all()
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
    with pytest.raises(ValueError, match="first_look"):
        Scenario(first_look=50)


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
    assert len(summary["scenarios"][0]["results"]) == (1 + len(sim.ALPHAS)) * len(sim.SE_FLOORS)
    # Power is the equivalence rate at Δ = 0 and the rate of the right direction otherwise.
    null, positive, negative = (s["results"] for s in summary["scenarios"])
    assert all(r["power"] == r["p_equivalence"] for r in null)
    assert all(r["power"] == r["p_positive"] for r in positive)
    assert all(r["power"] == r["p_negative"] for r in negative)
    assert [s["floor"] for s in summary["selection"]] == list(sim.SE_FLOORS)
    assert "## Selected levels" in (calibrate / "results.md").read_text()

    validate = tmp_path / "validate"
    script.main(
        ["validate", "--scenarios", "*rare_only*_d+5", "--alphas", "0.05",
         "--levels-from", str(calibrate / "summary.json"), *small, "--out", str(validate)]
    )  # fmt: skip
    summary = json.loads((validate / "summary.json").read_text())
    (scenario,) = summary["scenarios"]
    assert scenario["calibration"] is None and scenario["parameters"]["spread"] > 0
    assert {r["level_set"] for r in scenario["results"]} >= {"planning", "obf_0.0500"}
    assert summary["checks"][0]["groups"]["all"]["max_false_positive"]["value"] is None


def test_script_lists_and_filters_scenarios(capsys) -> None:
    load_script().main(["bias", "--designs", "interp_grid", "--list"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(FAMILIES) and all(line.startswith("interp_grid_") for line in lines)
