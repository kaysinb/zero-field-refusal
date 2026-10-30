"""Study 3's simulation before validation (analysis/study3_simulation.py): the scenario models
and the exact truth, checked against study 1's curves, large samples and a brute-force simulation
of fresh blocks, on synthetic seed vectors."""

import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from scipy.stats import norm

from zero_field_refusal.analysis import study3_simulation as sim
from zero_field_refusal.analysis.study3_blocks import arm_support
from zero_field_refusal.analysis.study3_records import Cell, Endpoint
from zero_field_refusal.analysis.study3_search import CANDIDATES
from zero_field_refusal.analysis.study3_seed_vectors import Point, Vector

START_RATES = CANDIDATES[1]["start"]
KSTAR = CANDIDATES[1]["targets"]["star"]
# Utility slopes (pp per ln K) beyond the points: the MMLU's, then the factual loss's
slopes = {"A": (4.0, 2.0), "F": (4.5, 2.2), "P": (4.2, 2.1)}
# The registered numbers, written out independently of the module
BETA = {"A": 15.300, "F": 17.7225, "P": 17.7225}
KAPPA = {"A": 9.5370, "F": 11.1240, "P": 11.1240}
JUDGE = {
    "equal": {"A": (0.93, 0.036), "F": (0.93, 0.036), "P": (0.93, 0.036)},
    "differential": {"A": (0.93, 0.036), "F": (0.75, 0.036), "P": (0.75, 0.036)},
}
SIGMA_E = {"A": 0.013, "F": 0.032, "P": 0.027}


def make_vector(seed: int, arms: dict) -> Vector:
    """A vector whose arms have points at the given (rate, x = ln(K/K*), MMLU loss) triples;
    the factual loss is half the MMLU loss."""
    return Vector(
        seed=seed,
        arms={
            arm: tuple(
                Point(
                    rate=r,
                    K=KSTAR * math.exp(x),
                    mmlu_loss_pp=m,
                    factual_loss_pp=m / 2,
                    run_id=f"r{seed}{arm}{i}",
                )
                for i, (r, x, m) in enumerate(points)
            )
            for arm, points in arms.items()
        },
    )


def at_start(lo: float, hi: float, m_lo: float = 2.5, m_hi: float = 3.5, arm: str = "A"):
    """An arm's two points at candidate 1's starting rates, at x = lo and hi."""
    return [(START_RATES[arm]["l"], lo, m_lo), (START_RATES[arm]["u"], hi, m_hi)]


@pytest.fixture
def tiny_vectors() -> list[Vector]:
    """Three whole vectors around K*: brackets of about ±0.22 in ln K with seed effects, one
    with a third, middle rate."""
    out = []
    for seed, shift in ((2100, -0.08), (2101, 0.0), (2102, 0.11)):
        arms = {}
        for arm, (lo, hi) in {"A": (-0.23, 0.22), "F": (-0.21, 0.24), "P": (-0.22, 0.23)}.items():
            arms[arm] = at_start(lo + shift, hi + shift, 2.0 + shift, 4.0 + shift, arm)
        if seed == 2102:
            mid = math.sqrt(START_RATES["A"]["l"] * START_RATES["A"]["u"])
            arms["A"] = sorted([*arms["A"], (mid, 0.05, 3.2)])
        out.append(make_vector(seed, arms))
    return out


def calibration_for(
    sc,
    failure=0.0,
    sigma_af=0.01,
    sigma_pf=0.01,
    offset_a=0.0,
    offset_p=0.0,
    candidate=None,
    mmlu_shift=None,
    factual_shift=None,
) -> sim.Calibration:
    """A calibration with the given failure, SDs and offsets (the attained values are not the
    truth's concern)."""
    return sim.Calibration(
        candidate=candidate or sc.candidate,
        failure=failure,
        realized_support=0.9,
        support_flagged=False,
        mmlu_shift=mmlu_shift or {},
        factual_shift=factual_shift or {},
        utility_realized={},
        utility_flagged=False,
        sigma_af=sigma_af,
        sigma_pf=sigma_pf,
        offset_a=offset_a,
        offset_p=offset_p,
        attained_af_pp=0.0,
        attained_pf_pp=0.0,
        attained_delta_pp={},
    )


class Brute(NamedTuple):
    mean: float
    se: float
    blocks: int


def curve(arm: str, x: np.ndarray, g: float) -> np.ndarray:
    return (15 + BETA[arm] * x + 0.5 * g * KAPPA[arm] * x**2) / 100


def brute_force_truth(
    sc, cal, vectors, rates, blocks, *, comparison="A-F", seed=11, utility_slopes=None
) -> Brute:
    """The mean interpolated difference of the primary judged share over simulated fresh
    supported blocks (each block one vector, with replacement; each run its jitter, failure and
    usability; each arm its block effect), the label noise integrated out exactly (the judged
    share given h), written from the registration's text."""
    rng = np.random.default_rng(seed)
    use = utility_slopes or slopes
    a, b = {"A-F": ("A", "F"), "P-F": ("P", "F")}[comparison]
    kstar_ln = math.log(CANDIDATES[cal.candidate]["targets"]["star"])
    vectors = sim.shifted_vectors(vectors, cal)
    total = total_sq = 0.0
    count = 0
    chunk = 500_000
    for start in range(0, blocks, chunk):
        size = min(chunk, blocks - start)
        which = rng.integers(0, len(vectors), size)
        ok = np.ones(size, dtype=bool)
        value = {}
        for arm in (a, b):
            ln = []
            for label in ("l", "u"):
                means = np.array([v.ln_k(arm, rates[arm][label]) for v in vectors])
                x = means[which].copy()
                if sc.jitter:
                    x += SIGMA_E[arm] * rng.standard_normal(size)
                ln.append(x)
            usable = np.ones(size, dtype=bool)
            for x in ln:
                mmlu = np.empty(size)
                for i, v in enumerate(vectors):
                    here = which == i
                    mmlu[here] = v.losses_at(arm, x[here], use)[0]
                usable &= mmlu <= 15 + 1e-9
                usable &= rng.random(size) >= cal.failure
            x1, x2 = ln[0] - kstar_ln, ln[1] - kstar_ln
            lo, hi = np.minimum(x1, x2), np.maximum(x1, x2)
            supported = usable & (hi > lo) & (lo <= 0) & (hi >= 0) & (hi - lo <= 0.60 + 1e-9)
            w = np.where(supported, -lo / np.where(hi > lo, hi - lo, 1.0), 0.0)
            sd = {"A": cal.sigma_af, "F": 0.0, "P": cal.sigma_pf}[arm]
            offset = {"A": cal.offset_a, "F": 0.0, "P": cal.offset_p}[arm]
            effect = offset + (sd * sim.draw_effects(sc.shape, rng, size) if sd else 0.0)
            se, fpr = JUDGE[sc.judge][arm]
            m_h, m_n = (sc.missing_rate, sc.missing_rate if sc.missing == "MCAR" else 0.0)

            def share(x, effect=effect, arm=arm, se=se, fpr=fpr, m_h=m_h, m_n=m_n):
                h = np.clip(curve(arm, x, sc.curvature_scale) + effect, 0, 1)
                return (1 - m_h) * se * h + (1 - m_n) * fpr * (1 - h)

            value[arm] = (1 - w) * share(lo) + w * share(hi)
            ok &= supported
        d = (value[a] - value[b])[ok]
        total += float(d.sum())
        total_sq += float((d * d).sum())
        count += int(d.size)
    mean = total / count
    var = total_sq / count - mean**2
    return Brute(mean, math.sqrt(var / count), count)


# ------------------------------------------------------------------ study 1's curvature


def test_study1_curvature_at_the_nominal_bracket() -> None:
    assert pytest.approx({"A": 9.5370, "F": 11.1240, "P": 11.1240}, abs=1e-4) == sim.CURVATURE_PP
    assert pytest.approx(BETA, abs=1e-4) == sim.SLOPE_PP
    a, f = (100 * sim.bracket_bias(arm, 0.223, 1.0) for arm in ("A", "F"))
    assert (a, f, a - f) == pytest.approx((0.2371, 0.2766, -0.0395), abs=2e-4)
    assert 100 * sim.bracket_bias("A", 0.223, 3.0) == pytest.approx(3 * a)  # C11 triples it
    assert pytest.approx(-0.0395, abs=1e-4) == sim.NOMINAL_BIAS_AF_PP  # M4's disclosure


def test_the_curve_is_study1s_at_kstar() -> None:
    """At K* the curve is the baseline 0.15 with study 1's slope and curvature: the logistic
    curves' local values (h′ = p(1−p)s, h″ = p(1−p)(1−2p)s² + 2p(1−p)q)."""
    from zero_field_refusal.analysis import study2_short_sim as s2

    family = s2.FAMILIES["study1"]
    for i, arm in enumerate(("A", "F", "P")):
        eps = 1e-4
        h = [float(family.target(i, np.array(x))) for x in (-eps, 0.0, eps)]
        assert h[1] == pytest.approx(0.15)
        assert 100 * (h[2] - h[0]) / (2 * eps) == pytest.approx(sim.SLOPE_PP[arm], rel=1e-6)
        assert 100 * (h[2] - 2 * h[1] + h[0]) / eps**2 == pytest.approx(
            sim.CURVATURE_PP[arm], rel=1e-4
        )
        assert float(sim.harm_curve(arm, np.array(0.0), 1.0)) == pytest.approx(0.15)


def test_the_constants_written_out_are_study2s() -> None:
    """The constants this module writes out (so that it loads no GPU path through study 2's
    analysis) are study 2's simulation's."""
    from zero_field_refusal.analysis import study2_short_sim as s2

    assert {
        arm: (s2.STUDY1["slope"][i], s2.STUDY1["curvature"][i])
        for i, arm in enumerate(("A", "F", "P"))
    } == sim.STUDY1_LOGISTIC
    assert s2.STUDY1["kink"] == (0.0, 0.0, 0.0) and s2.FAMILIES["study1"].baseline == sim.BASELINE
    assert (sim.HEAVY_SHARE, sim.HEAVY_RATIO, sim.HEAVY_SCALES) == (
        s2.HEAVY_SHARE,
        s2.HEAVY_RATIO,
        s2.HEAVY_SCALES,
    )
    assert (sim.GAMMA_SHAPE, sim.RARE, sim.THREAD_VARIABLES) == (
        s2.GAMMA_SHAPE,
        s2.RARE,
        s2.THREAD_VARIABLES,
    )
    defaults = s2.Scenario.__dataclass_fields__
    assert (defaults["rho_a"].default, defaults["rho_b"].default) == (sim.RHO_ARM, sim.RHO_BLOCK)
    assert sim.PROMPTS == s2.PROMPTS == 521


def test_the_module_loads_no_gpu_path_and_no_study2_reader() -> None:
    code = (
        "import sys; import zero_field_refusal.analysis.study3_simulation; "
        "bad = ('torch', 'transformers', 'zero_field_refusal.analysis.stage_a', "
        "'zero_field_refusal.analysis.study2_short', "
        "'zero_field_refusal.analysis.study2_short_sim', "
        "'zero_field_refusal.analysis.study2_labels'); "
        "sys.exit(any(m == b or m.startswith(b + '.') for m in sys.modules for b in bad))"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ------------------------------------------------------------------ the shapes and the clip


@pytest.mark.parametrize("shape", sim.SHAPES)
def test_each_shape_has_mean_0_and_sd_1(shape) -> None:
    e = sim.draw_effects(shape, np.random.default_rng(1), 2_000_000)
    assert abs(e.mean()) < 4e-3 and e.std() == pytest.approx(1.0, abs=4e-3)


def test_the_shapes_are_as_registered() -> None:
    rng = np.random.default_rng(2)
    rare = sim.draw_effects("rare", rng, 200_000)
    assert set(np.round(rare, 9)) == {round(0.9 / 0.3, 9), round(-0.1 / 0.3, 9)}
    skewed = sim.draw_effects("skewed", rng, 2_000_000)
    assert ((skewed - skewed.mean()) ** 3).mean() == pytest.approx(2 / math.sqrt(2), abs=0.05)
    heavy = sim.draw_effects("heavy", rng, 2_000_000)
    assert (heavy**4).mean() / (heavy**2).mean() ** 2 - 3 == pytest.approx(
        3 * (0.1 * 81 + 0.9) / 1.8**2 - 3, abs=0.2
    )
    with pytest.raises(ValueError):
        sim.draw_effects("lognormal", rng, 3)


@pytest.mark.parametrize("shape", sim.SHAPES)
def test_the_clipped_mean_matches_a_large_sample(shape) -> None:
    rng = np.random.default_rng(3)
    e = sim.draw_effects(shape, rng, 2_000_000)
    for mean, sigma in (
        (0.15, 0.05),
        (0.02, 0.05),
        (0.97, 0.05),
        (0.15, 0.3),
        (-0.1, 0.2),
        (1.05, 0.2),
        (0.5, 0.0),
        (0.0, 0.08),
    ):
        sample = np.clip(mean + sigma * e, 0, 1)
        exact = float(sim.expected_clipped(np.array(mean), sigma, shape))
        se = sample.std() / math.sqrt(sample.size)
        assert abs(exact - sample.mean()) <= 4 * se + 1e-12, (mean, sigma)


def test_the_clipped_mean_is_elementwise_and_refuses_a_negative_sigma() -> None:
    means = np.array([-0.2, 0.0, 0.15, 0.9, 1.2])
    for shape in sim.SHAPES:
        at = sim.expected_clipped(means, 0.1, shape)
        assert at == pytest.approx(
            [float(sim.expected_clipped(np.array(m), 0.1, shape)) for m in means]
        )
    with pytest.raises(ValueError):
        sim.expected_clipped(means, -0.1, "normal")


# ------------------------------------------------------------------ the labels


@pytest.mark.parametrize("name", ["C02", "C13", "C14"])
def test_labels_have_the_expected_judged_share(name) -> None:
    """(1 − m_h)·Se·h + (1 − m_n)·FPR·(1 − h), within 4 SE over many blocks."""
    sc = sim.SCENARIOS_BY_NAME[name]
    rng = np.random.default_rng(4)
    h = np.array([0.05, 0.12, 0.15, 0.2, 0.35, 0.6])
    arms = ["A", "A", "F", "F", "P", "P"]
    shares = np.stack([sim.endpoint_labels(h, arms, sc, rng) for _ in range(3000)])
    for j, arm in enumerate(arms):
        se, fpr = JUDGE[sc.judge][arm]
        m_h, m_n = (sc.missing_rate, sc.missing_rate if sc.missing == "MCAR" else 0.0)
        want = (1 - m_h) * se * h[j] + (1 - m_n) * fpr * (1 - h[j])
        got = shares[:, j]
        assert abs(got.mean() - want) <= 4 * got.std() / math.sqrt(got.size), (arm, j)
        assert float(sim.judged_share(arm, np.array(h[j]), sc)) == pytest.approx(want)


def test_the_labels_copula_ties_an_arms_endpoints_more_than_the_arms() -> None:
    """ρ_a 0.7 within an arm's two endpoints, ρ_b 0.3 between arms: the shares of one arm's
    endpoints move together more than those of two arms."""
    sc = replace(sim.SCENARIOS_BY_NAME["C02"], missing_rate=0.0, judge="equal")
    rng = np.random.default_rng(5)
    shares = np.stack(
        [sim.endpoint_labels(np.full(4, 0.5), ["A", "A", "F", "F"], sc, rng) for _ in range(4000)]
    )
    within = np.corrcoef(shares[:, 0], shares[:, 1])[0, 1]
    across = np.corrcoef(shares[:, 0], shares[:, 2])[0, 1]
    assert within > across > 0
    with pytest.raises(ValueError):
        sim.endpoint_labels(np.full(3, 0.5), ["A", "A", "F", "F"], sc, rng)


# ------------------------------------------------------------------ the quadrature


def test_a_normal_threshold_is_integrated_exactly() -> None:
    # Gauss–Hermite at 20 nodes gives 0.5 here; the breakpoint rule must give Φ(0.3125)
    value = sim.normal_piecewise(lambda z: (z < 0.3125).astype(float), [0.3125])
    assert value == pytest.approx(0.6226697, abs=1e-6)
    hermite, weights = np.polynomial.hermite_e.hermegauss(20)
    assert float(((hermite < 0.3125) * weights).sum() / weights.sum()) == pytest.approx(0.5)


def test_the_quadrature_integrates_smooth_functions_and_refuses_a_hidden_step() -> None:
    # bounded integrands, as the truth's are (shares and supports)
    assert sim.normal_piecewise(lambda z: np.exp(-(z**2)), []) == pytest.approx(
        1 / math.sqrt(3), abs=1e-6
    )
    assert sim.normal_piecewise(lambda z: norm.cdf(z) ** 2, [0.0]) == pytest.approx(1 / 3, abs=1e-6)
    # integrands that vary far into the tails: the fixed splits keep the rule fast there
    assert sim.normal_piecewise(lambda z: z**2, []) == pytest.approx(1.0, abs=1e-6)
    assert sim.normal_piecewise(np.cos, []) == pytest.approx(math.exp(-0.5), abs=1e-6)
    # a smooth piece between breakpoints: E[Z²; |Z| < 1] = 2Φ(1) − 1 − 2φ(1)
    inside = sim.normal_piecewise(lambda z: np.where(np.abs(z) < 1, z**2, 0.0), [-1.0, 1.0])
    assert inside == pytest.approx(2 * norm.cdf(1) - 1 - 2 * norm.pdf(1), abs=1e-6)
    with pytest.raises(sim.TruthNotConverged):  # a step the breakpoints do not name
        sim.normal_piecewise(lambda z: (z < 0.3125).astype(float), [])
    with pytest.raises(sim.TruthNotConverged):  # too few nodes to compare n with 2n
        sim.normal_piecewise(lambda z: np.exp(-(z**2)), [], max_nodes=16)


# ------------------------------------------------------------------ the support


def test_the_truths_support_is_the_registered_support() -> None:
    """The truth's support and weight are ``study3_blocks.arm_support``'s, on random brackets."""
    rng = np.random.default_rng(6)
    for _ in range(400):
        x = rng.uniform(-0.7, 0.7, 2)
        usable = rng.random(2) > 0.1
        mine, lo, hi, w = sim._support(
            np.array(x[0]), np.array(x[1]), np.array(usable[0]), np.array(usable[1])
        )
        ends = []
        for i, (xi, ok) in enumerate(zip(x, usable, strict=True)):
            rate = (1.45e-5, 2.11e-5)[i]
            cell = Cell(
                campaign="c",
                label=f"A-{rate:g}/s10300",
                candidate=1,
                step="validation",
                round=None,
                arm="A",
                rate=rate,
                seed=10300,
                config_hash="0",
            )
            ends.append(
                Endpoint(
                    run_id=f"r{i}",
                    run_dir=Path(f"r{i}"),
                    cell=cell,
                    arm="A",
                    lr=rate,
                    seed=10300,
                    role="validation",
                    candidate=1,
                    k_only=False,
                    attempt=1,
                    label=cell.label,
                    state="usable" if ok else "unusable",
                    steps=25,
                    stop="steps",
                    K=KSTAR * math.exp(xi),
                    mmlu_loss_pp=3.0 if ok else 20.0,
                    factual_loss_pp=1.0,
                    unusable_reason=None if ok else "gross_collapse",
                )
            )
        registered = arm_support(ends, KSTAR)
        assert bool(mine) == registered["supported"], (x, usable, registered["reason"])
        if registered["supported"]:
            assert float(w) == pytest.approx(registered["weight"], abs=1e-12)


# ------------------------------------------------------------------ the truth


def test_the_truth_without_jitter_is_the_mean_over_fresh_supported_blocks(tiny_vectors) -> None:
    sc = sim.SCENARIOS_BY_NAME["C11"]  # g = 3: the curvature counts
    cal = calibration_for(
        sc, failure=0.05, sigma_af=0.03, sigma_pf=0.03, offset_a=0.01, offset_p=-0.004
    )
    exact = sim.truth(sc, cal, tiny_vectors, slopes, START_RATES)
    assert set(exact) == {"A-F", "P-F"}
    for comparison in ("A-F", "P-F"):
        brute = brute_force_truth(
            sc, cal, tiny_vectors, START_RATES, 400_000, comparison=comparison
        )
        assert exact[comparison] == pytest.approx(brute.mean, abs=4 * brute.se)


def test_the_truth_is_the_mean_over_fresh_supported_blocks(tiny_vectors) -> None:
    sc = replace(sim.SCENARIOS_BY_NAME["C13"], jitter=True)  # differential judge, jitter
    cal = calibration_for(sc, failure=0.02, sigma_af=0.03, sigma_pf=0.03, offset_a=0.004)
    exact = sim.truth(sc, cal, tiny_vectors, slopes, START_RATES)
    brute = brute_force_truth(sc, cal, tiny_vectors, START_RATES, blocks=400_000)
    assert exact["A-F"] == pytest.approx(brute.mean, abs=4 * brute.se)


@pytest.mark.parametrize("shape", ["heavy", "skewed", "rare"])
def test_the_truth_takes_each_shapes_clipped_mean(tiny_vectors, shape) -> None:
    sc = replace(sim.SCENARIOS_BY_NAME["C02"], shape=shape)
    cal = calibration_for(sc, sigma_af=0.12, offset_a=-0.1)  # the clip binds often
    exact = sim.truth(sc, cal, tiny_vectors, slopes, START_RATES)
    brute = brute_force_truth(sc, cal, tiny_vectors, START_RATES, 400_000)
    assert exact["A-F"] == pytest.approx(brute.mean, abs=4 * brute.se)


def test_two_resolutions_agree_or_the_truth_refuses(tiny_vectors) -> None:
    sc = replace(sim.SCENARIOS_BY_NAME["C12"], jitter=True)
    cal = calibration_for(sc)  # test helper
    fine = sim.truth(sc, cal, tiny_vectors, slopes, START_RATES)  # converged at 1e-6
    assert math.isfinite(fine["A-F"]) and math.isfinite(fine["P-F"])
    finer = sim.truth(sc, cal, tiny_vectors, slopes, START_RATES, tol=1e-9)
    assert finer == pytest.approx(fine, abs=1e-6)
    with pytest.raises(sim.TruthNotConverged):
        sim.truth(sc, cal, tiny_vectors, slopes, START_RATES, max_nodes=8)


def test_a_truth_that_does_not_converge_is_refused(tiny_vectors) -> None:
    """A tolerance no node count up to the largest can meet: the doubling runs out of nodes
    (16 against 32, then 32 against 64) and the truth refuses, at every call."""
    sc = replace(sim.SCENARIOS_BY_NAME["C12"], jitter=True)
    cal = calibration_for(sc)
    with pytest.raises(sim.TruthNotConverged):
        sim.truth(sc, cal, tiny_vectors, slopes, START_RATES, tol=1e-14, max_nodes=64)


def boundary_case(where: str):
    """A jitter scenario and two vectors: one whose A and F endpoints sit within 0.01 in ln K
    of a support boundary (the bracket's low or high end, the 0.60 width, or a gross collapse),
    and one far from every boundary with another interpolated difference, so that how often the
    first is supported moves the truth."""
    sc = sim.SCENARIOS_BY_NAME["C12"]
    if where == "bracket_low":
        a, f = (-0.004, 0.25), (-0.006, 0.22)
    elif where == "bracket_high":
        a, f = (-0.25, 0.004), (-0.22, 0.007)
    elif where == "width":
        a, f = (-0.30, 0.295), (-0.31, 0.296)
    else:
        a, f = (-0.2, 0.2), (-0.21, 0.19)
    if where == "collapse":  # the MMLU loss reaches 15 pp just above the high endpoint
        near = {
            "A": at_start(*a, 5.0, 14.95, "A"),
            "F": at_start(*f, 5.0, 14.9, "F"),
        }
    else:
        near = {"A": at_start(*a, arm="A"), "F": at_start(*f, arm="F")}
    far = {"A": at_start(-0.3, 0.27, arm="A"), "F": at_start(-0.03, 0.5, arm="F")}
    vectors = [make_vector(2100, near), make_vector(2101, far)]
    rates = {arm: START_RATES[arm] for arm in ("A", "F")}
    return sc, vectors, rates


BOUNDARY_SLOPES = {"A": (25.0, 12.0), "F": (25.0, 12.0), "P": (25.0, 12.0)}


@pytest.mark.parametrize("where", ["bracket_low", "bracket_high", "width", "collapse"])
def test_the_truth_is_accurate_at_each_support_boundary(where) -> None:
    # vectors placed so that the jittered endpoint sits within 0.01 in ln K of the boundary
    sc, vectors, rates = boundary_case(where)  # test helper
    use = BOUNDARY_SLOPES
    exact = sim.truth(sc, calibration_for(sc), vectors, use, rates)
    brute = brute_force_truth(
        sc, calibration_for(sc), vectors, rates, blocks=4_000_000, utility_slopes=use
    )
    assert exact["A-F"] == pytest.approx(brute.mean, abs=4 * brute.se)
    assert brute.se < 2e-5
    # the boundary binds: the support is neither certain nor impossible
    assert 0.05 * 4_000_000 < brute.blocks < 0.95 * 4_000_000
    # the endpoint sits within 0.01 in ln K of the boundary, and the breakpoints name it
    v, sd = vectors[0], sim.JITTER_SD["A"]
    m1 = v.ln_k("A", rates["A"]["l"]) - math.log(KSTAR)
    m2 = v.ln_k("A", rates["A"]["u"]) - math.log(KSTAR)
    outer = [m1 + sd * z for z in sim.support_breakpoints(v, "A", rates, sc, None, slopes=use)]
    inner = [m2 + sd * z for z in sim.support_breakpoints(v, "A", rates, sc, 0.0, slopes=use)]
    width = 0.60 + 1e-9
    if where == "bracket_low":
        assert abs(m1) < 0.01 and min(abs(x) for x in outer) < 1e-12
    elif where == "bracket_high":
        assert abs(m2) < 0.01 and min(abs(x) for x in inner) < 1e-12
    elif where == "width":
        assert abs(m2 - m1 - 0.60) < 0.01
        assert min(abs(x - (m1 + width)) for x in inner) < 1e-12
    else:
        (crossing,) = v.mmlu_crossings("A", 15 + 1e-9, use)
        c = crossing - math.log(KSTAR)
        assert 0 < c - m2 < 0.01
        assert min(abs(x - c) for x in outer) < 1e-12 and min(abs(x - c) for x in inner) < 1e-12


def test_the_truth_applies_the_calibrations_shifts_and_candidate(tiny_vectors) -> None:
    """The safe alternative's shifts move the collapse, and so the truth's support; the
    calibration's candidate gives K*."""
    sc = sim.SCENARIOS_BY_NAME["C02"]
    shift = {
        arm: {v.seed: (12.5 if v.seed == 2102 else 0.0) for v in tiny_vectors}
        for arm in ("A", "F", "P")
    }
    cal = calibration_for(
        sc, sigma_af=0.02, offset_a=0.01, mmlu_shift=shift, factual_shift=dict.fromkeys("AFP", 0.0)
    )
    plain = sim.truth(
        sc, calibration_for(sc, sigma_af=0.02, offset_a=0.01), tiny_vectors, slopes, START_RATES
    )
    shifted = sim.truth(sc, cal, tiny_vectors, slopes, START_RATES)
    without = sim.truth(
        sc, calibration_for(sc, sigma_af=0.02, offset_a=0.01), tiny_vectors[:2], slopes, START_RATES
    )
    # seed 2102's high endpoints lose more than 15 pp once shifted: it supports nothing
    assert shifted["A-F"] == pytest.approx(without["A-F"]) and shifted != plain
    brute = brute_force_truth(sc, cal, tiny_vectors, START_RATES, 400_000)
    assert shifted["A-F"] == pytest.approx(brute.mean, abs=4 * brute.se)
    with pytest.raises(ValueError):
        sim.shifted_vectors(
            tiny_vectors,
            calibration_for(sc, mmlu_shift={"A": {2100: 1.0}}, factual_shift={"A": 0.0}),
        )
    # the calibration's candidate gives K* (candidate 2's is 5.0): fewer brackets hold it
    second = calibration_for(sc, sigma_af=0.02, offset_a=0.01, candidate=2)
    at_second = sim.truth(sc, second, tiny_vectors, slopes, START_RATES)
    assert at_second["A-F"] != pytest.approx(plain["A-F"], abs=1e-4)
    brute = brute_force_truth(sc, second, tiny_vectors, START_RATES, 400_000)
    assert at_second["A-F"] == pytest.approx(brute.mean, abs=4 * brute.se)
    # rates at which no vector brackets K*: no fresh block is supported, the truth undefined
    far = {arm: {k: 3 * r for k, r in pair.items()} for arm, pair in START_RATES.items()}
    assert all(math.isnan(v) for v in sim.truth(sc, cal, tiny_vectors, slopes, far).values())


def test_the_truth_without_p_gives_a_f_only(tiny_vectors) -> None:
    sc = sim.SCENARIOS_BY_NAME["C02"]
    rates = {arm: START_RATES[arm] for arm in ("A", "F")}
    out = sim.truth(sc, calibration_for(sc), tiny_vectors, slopes, rates)
    assert set(out) == {"A-F"}
    assert out["A-F"] == pytest.approx(
        sim.truth(sc, calibration_for(sc), tiny_vectors, slopes, START_RATES)["A-F"]
    )


# ------------------------------------------------------------------ the grid


def test_every_scenario_parameter_has_a_decided_source() -> None:
    for name in sim.Scenario.parameters():
        assert sim.SOURCES[name].endswith("decided"), name
        assert " M" in sim.SOURCES[name].rsplit(";", 1)[-1], name
    assert set(sim.SOURCES) == set(sim.Scenario.parameters())
    for sc in sim.SCENARIOS:
        assert dict(sc.sources) == sim.SOURCES


def test_the_grid_is_the_registered_one() -> None:
    names = [s.name for s in sim.SCENARIOS]
    assert names == [f"C{i:02d}" for i in range(1, 18)] + ["P18", "P19", "G1", "G2"]
    coverage = [s for s in sim.SCENARIOS if s.coverage]
    assert len(coverage) == 17  # J = 17 × 105
    assert [s.name for s in sim.SCENARIOS if s.power] == ["C16", "C17", "P18", "P19"]
    by = sim.SCENARIOS_BY_NAME
    assert [by[f"C{i:02d}"].spread_pp for i in range(1, 9)] == [3.5, 5.0] * 4
    assert [by[f"C{i:02d}"].shape for i in range(1, 9)] == [
        "normal",
        "normal",
        "heavy",
        "heavy",
        "skewed",
        "skewed",
        "rare",
        "rare",
    ]
    assert (by["C09"].support, by["C10"].support, by["C11"].curvature_scale) == (0.88, 0.95, 3.0)
    assert by["C12"].jitter and not any(s.jitter for s in sim.SCENARIOS if s.name != "C12")
    assert by["C13"].judge == "differential"
    assert (by["C14"].missing, by["C14"].missing_rate) == ("MNAR", 0.05)
    assert by["C15"].candidate == 2 and {s.candidate for s in sim.SCENARIOS} == {1, 2}
    for name, spread, delta in (
        ("C16", 3.5, 2.0),
        ("C17", 5.0, 2.0),
        ("P18", 3.5, -2.0),
        ("P19", 5.0, -2.0),
    ):
        assert (by[name].spread_pp, by[name].delta_af_pp, by[name].delta_pf_pp) == (
            spread,
            delta,
            delta,
        )
    assert (by["G1"].support, by["G1"].utility, by["G2"].support, by["G2"].utility) == (
        0.95,
        "safe",
        0.90,
        "empirical",
    )
    common = {s.name for s in sim.SCENARIOS if s.role != "gate"}
    assert all(by[n].utility == "safe" for n in common)
    assert all(by[n].missing_rate == 0.01 for n in common - {"C14"})


def test_a_scenario_round_trips_through_json_and_refuses_another_field() -> None:
    import json

    for sc in sim.SCENARIOS:
        text = json.dumps(sc.to_json(), allow_nan=False, sort_keys=True)
        assert sim.Scenario.from_json(json.loads(text)) == sc
    with pytest.raises(ValueError):
        sim.Scenario.from_json({**sim.SCENARIOS[0].to_json(), "harm": 1})
    with pytest.raises(ValueError):
        replace(sim.SCENARIOS[0], shape="lognormal")
    with pytest.raises(ValueError):
        replace(sim.SCENARIOS[0], sources=(("shape", "x"),))
    with pytest.raises(ValueError):
        replace(sim.SCENARIOS[0], jitter=1)
    with pytest.raises(ValueError):
        replace(sim.SCENARIOS[0], support=0.9 + 0.2)
