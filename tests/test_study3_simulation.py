"""Study 3's simulation before validation (analysis/study3_simulation.py): the scenario models
and the exact truth, checked against study 1's curves, large samples and a brute-force simulation
of fresh blocks, on synthetic seed vectors; one simulated study through the registered functions
and the calibration through the selection, on synthetic whole seed vectors."""

import math
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from scipy.stats import norm

from zero_field_refusal.analysis import study3_simulation as sim
from zero_field_refusal.analysis.study3_blocks import arm_support, block_state
from zero_field_refusal.analysis.study3_decisions import LEVELS, cap_of, sample_size
from zero_field_refusal.analysis.study3_inference import Interval, stream
from zero_field_refusal.analysis.study3_progress import confirmation_state
from zero_field_refusal.analysis.study3_records import SEEDS, Cell, Endpoint
from zero_field_refusal.analysis.study3_search import (
    ARM_ORDER,
    CANDIDATES,
    LABELS,
    SLOPES,
    search_state,
)
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


def test_no_quantile_is_taken_in_the_simulation_modules() -> None:
    """No call takes a quantile or a percentile in the simulation's modules (order statistics
    are type-1 ranks: ``study3_inference.tail_ranks``), and the module's rulings list every
    decided value M1–M27 that it applies."""
    import ast

    from zero_field_refusal.paths import REPO_ROOT

    paths = [Path(sim.__file__), Path(sim.__file__).with_name("study3_seed_vectors.py"),
             REPO_ROOT / "scripts" / "study3_simulate.py"]  # fmt: skip
    for path in paths:
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                assert not any(w in name.lower() for w in ("quantile", "percentile", "median")), (
                    path.name, name)  # fmt: skip
    rulings = sim.__doc__.split("**Rulings**", 1)[1]
    for n in (*range(1, 9), *range(10, 28)):
        assert f"- M{n}," in rulings, n


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


def nested_moments(vector, arm, rates, sc, cal, kstar_ln, tol=1e-6, max_nodes=512):
    """(E[1_a], E[1_a·I_a]) by the one nested quadrature of the whole integrand (run 1's jitter
    outer, run 2's inner, both shares inside), the form before the quadrature was split."""
    m = sim._arm_means(vector, arm, rates, kstar_ln)
    se = sim.JITTER_SD[arm]
    terms = sim._arm_terms(vector, arm, m, se, kstar_ln, slopes, sc, cal)
    crossings = sim._crossings(vector, arm, slopes, kstar_ln)

    def outer(z1):
        flat = z1.reshape(-1)
        breaks = (sim._x_breaks_inner(m[0] + se * flat, crossings) - m[1]) / se
        values = sim._piecewise(lambda z2: np.stack(terms(flat[:, None, None], z2)), breaks,
                                tol=tol, max_nodes=max_nodes)  # fmt: skip
        return values.reshape((2, *z1.shape))

    breaks = np.array([[(x - m[0]) / se for x in sim._x_breaks_outer(crossings)]])
    s, t = sim._piecewise(outer, breaks, tol=tol, max_nodes=max_nodes)[:, 0]
    return float(s), float(t)


def test_the_jitter_quadrature_is_reused_across_sigma_and_offsets(tiny_vectors,
                                                                  monkeypatch) -> None:  # fmt: skip
    """C12's moments at a new σ or offset reuse the jitter quadrature, which depends on neither:
    no inner integral is computed again, the value is a fresh quadrature's and the nested
    quadrature's of the whole integrand, and the outer rule is checked at every evaluation."""
    sc = sim.SCENARIOS_BY_NAME["C12"]
    kstar_ln = math.log(KSTAR)
    rates = {"l": START_RATES["A"]["l"] * 1.05, "u": START_RATES["A"]["u"] * 0.97}
    v = tiny_vectors[1]
    sim.clear_caches()
    inner = []
    real = sim._piecewise
    monkeypatch.setattr(sim, "_piecewise", lambda *a, **k: inner.append(1) or real(*a, **k))
    first = sim._arm_moments(v, "A", rates, sc, calibration_for(sc, sigma_af=0.03), slopes,
                             kstar_ln, tol=1e-6, max_nodes=512)  # fmt: skip
    built = len(inner)
    assert built > 0
    moved = calibration_for(sc, sigma_af=0.045, offset_a=0.02)
    second = sim._arm_moments(v, "A", rates, sc, moved, slopes, kstar_ln, tol=1e-6,
                              max_nodes=512)  # fmt: skip
    assert len(inner) == built  # the inner integrals are reused
    assert second[0] == first[0] and second[1] != first[1]  # support: σ and offset free
    monkeypatch.setattr(sim, "_piecewise", real)
    sim.clear_caches()
    assert sim._arm_moments(v, "A", rates, sc, moved, slopes, kstar_ln, tol=1e-6,
                            max_nodes=512) == second  # fmt: skip
    for value, whole in zip(second, nested_moments(v, "A", rates, sc, moved, kstar_ln),
                            strict=True):  # fmt: skip
        assert abs(value - whole) <= 1e-8
    # the outer rule is checked at every evaluation: a kept quadrature refuses a tolerance
    # no node count meets
    grid = sim._grid(v, "A", rates, slopes, kstar_ln, tol=1e-6, max_nodes=512)
    grid.tol, grid.max_nodes = 1e-15, 64
    with pytest.raises(sim.TruthNotConverged):
        grid.moments(sim._share("A", sc, moved))


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


# ------------------------------------------------------------------ one study

SEED = stream(20261101, "support", "test", 1)  # a calibration stream
WORLD_BIAS = 0.12  # ln K at the starting rates lies above the targets: the search moves them
SEED_SD = {"A": 0.07, "F": 0.08, "P": 0.07}


def world_vectors(count: int = 25, seed: int = 7) -> list[Vector]:
    """Synthetic whole seed vectors: per arm three rates around candidate 1's starting rates, ln K
    in ln η with the registered slope, a seed effect and a bias above the targets (so that the
    search adjusts the rates), the MMLU loss rising with ln K; one vector's F lies far above K*
    (no bracket at any selected rate), so that the natural support lies below 1."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        utility = rng.normal(0.0, 0.8)
        arms = {}
        for arm in ARM_ORDER:
            lo, hi = START_RATES[arm]["l"], START_RATES[arm]["u"]
            mid = math.sqrt(lo * hi)
            effect = 0.30 if (i == 0 and arm == "F") else rng.normal(0.0, SEED_SD[arm])
            points = []
            for r in (lo * 0.8, mid, hi * 1.25):
                x = SLOPES[arm] * math.log(r / mid) + WORLD_BIAS + effect + rng.normal(0, 0.02)
                points.append((r, x, 2.5 + 4.0 * x + utility + rng.normal(0.0, 0.1)))
            arms[arm] = points
        out.append(make_vector(2100 + i, arms))
    return out


def hand_calibration(candidate: int, failure: float = 0.02, sigma: float = 0.03, **change):
    return sim.Calibration(
        candidate=candidate,
        failure=failure,
        realized_support=0.9,
        support_flagged=False,
        mmlu_shift=change.pop("mmlu_shift", {}),
        factual_shift=change.pop("factual_shift", {}),
        utility_realized={},
        utility_flagged=False,
        sigma_af=sigma,
        sigma_pf=sigma,
        offset_a=change.pop("offset_a", 0.0),
        offset_p=change.pop("offset_p", 0.0),
        attained_af_pp=0.0,
        attained_pf_pp=0.0,
        attained_delta_pp={},
    )


def af_fails_at_candidate_1() -> sim.Scenario:
    return replace(sim.SCENARIOS_BY_NAME["C02"], name="AF1")


def pf_gate_fails() -> sim.Scenario:
    return replace(sim.SCENARIOS_BY_NAME["C02"], name="PF0")


def small_spread() -> sim.Scenario:
    return replace(sim.SCENARIOS_BY_NAME["C01"], name="S60")


@pytest.fixture(scope="module")
def vectors() -> list[Vector]:
    return world_vectors()


@pytest.fixture(scope="module")
def setup(vectors) -> sim.Setup:
    """The synthetic vectors, their slopes and hand calibrations of every scenario the tests run
    (the calibration's own tests calibrate)."""
    from zero_field_refusal.analysis.study3_seed_vectors import utility_slopes

    cal = {}
    for sc in sim.SCENARIOS:
        for c in (1, 2):
            cal[(sc, c)] = hand_calibration(c)
    cal[(replace(sim.SCENARIOS_BY_NAME["C02"], support=0.5), 1)] = hand_calibration(1, 0.4)
    cal[(af_fails_at_candidate_1(), 1)] = hand_calibration(1, 0.6)
    cal[(af_fails_at_candidate_1(), 2)] = hand_calibration(2, 0.0)
    shift = {arm: {v.seed: (6.0 if arm == "P" else 0.0) for v in vectors} for arm in ARM_ORDER}
    for c in (1, 2):  # P's MMLU loss far above 5 pp: P–F's utility bounds fail, A–F's hold
        cal[(pf_gate_fails(), c)] = hand_calibration(
            c, mmlu_shift=shift, factual_shift=dict.fromkeys(ARM_ORDER, 0.0)
        )
        cal[(small_spread(), c)] = hand_calibration(c, sigma=0.0)
    return sim.Setup(
        vectors=tuple(vectors),
        slopes=utility_slopes(vectors),
        calibrations=cal,
        M=10_000,
        levels=LEVELS,
        validation_blocks=28,
        n_min=None,
    )


def test_synthetic_round0_passes_the_registered_search(setup) -> None:
    eps = [
        sim.synthetic_endpoint(
            1,
            "search",
            0,
            arm,
            CANDIDATES[1]["start"][arm][lab],
            s,
            K=5.5,
            mmlu=None,
            factual=None,
            k_only=True,
        )
        for arm in ARM_ORDER
        for lab in LABELS
        for s in SEEDS["search"][1]
    ]
    state = search_state(eps, 1, problems=())
    assert state["next"] in ("round1", "numerical") and state["problems"] == []


def test_a_synthetic_block_passes_the_registered_block_state() -> None:
    eps = [
        sim.synthetic_endpoint(
            1,
            "validation",
            None,
            arm,
            START_RATES[arm][lab],
            10300,
            K=k,
            mmlu=3.0,
            factual=1.5,
            k_only=False,
        )
        for arm in ARM_ORDER
        for lab, k in (("l", 5.0), ("u", 7.0))
    ]
    eps[0] = sim.synthetic_endpoint(
        1, "validation", None, "A", START_RATES["A"]["l"], 10300, K=None, mmlu=None,
        factual=None, state="scientific", k_only=False,
    )  # fmt: skip
    block = block_state(eps, 10300, 1, ARM_ORDER)
    assert block["state"] == "resolved" and block["blockers"] == []
    assert block["arms"]["A"]["reason"] == "failed endpoint" and block["PF"]["supported"]
    state = confirmation_state(
        [
            sim.synthetic_endpoint(
                1, "confirmation", None, arm, START_RATES[arm][lab], s, K=k, mmlu=3.0,
                factual=1.5, k_only=False,
            )
            for s in SEEDS["confirmation"][1][:3]
            for arm in ("A", "F")
            for lab, k in (("l", 5.0), ("u", 7.0))
        ],
        1,
        60,
        ["A-F"],
    )  # fmt: skip
    assert state["started"] == [10400, 10401, 10402] and state["counters"]["A-F"] == 3
    assert state["problems"] == [] and state["may_start_next"]


def test_a_study_depends_only_on_its_stream_scenario_and_index(setup) -> None:
    sc = sim.SCENARIOS_BY_NAME["C02"]
    a = sim.simulate_study(sc, setup, 11, 7, resamples=200, levels=LEVELS)
    other = sim.simulate_study(sc, setup, 11, 3, resamples=200, levels=LEVELS)
    assert sim.simulate_study(sc, setup, 11, 7, resamples=200, levels=LEVELS) == a
    assert other != a and any(a.confirmed.values())
    assert sim.simulate_study(sc, setup, 12, 7, resamples=200, levels=LEVELS) != a


def test_the_n_point_cohort_at_n_is_the_confirmations_cohort(setup) -> None:
    """At N = 60 (a small spread), the N point 60's blocks are the confirmation's cohort: the
    first 60 supported blocks by seed order."""
    seen = 0
    for index in range(4):
        o = sim.simulate_study(small_spread(), setup, 11, index, resamples=200, levels=LEVELS)
        for level in LEVELS:
            if not o.confirmed[level] or o.N[level] != 60:
                continue
            seen += 1
            for comparison in ("A-F", "P-F"):
                cohort = o.cohorts[(level, comparison)]
                assert len(cohort) == 60
                assert o.point_seeds[(60, comparison)] == cohort
                assert cohort == tuple(sorted(cohort)) and cohort[0] >= 10400
    assert seen


def test_p_closed_leaves_the_af_analysis_set_unchanged(setup) -> None:
    """A study whose P–F gate fails: the screen computes P–F as if active, the procedure closes
    P; A–F's blocks, analysis sets, intervals and readings are the same."""
    sc = pf_gate_fails()
    for index in range(3):
        screen = sim.simulate_study(sc, setup, 13, index, resamples=200, levels=LEVELS)
        procedure = sim.simulate_study(
            sc, setup, 13, index, mode="procedure", resamples=200, levels=LEVELS
        )
        assert screen.gates == procedure.gates and screen.gates["PF"] is False
        assert screen.active == ["A-F", "P-F"] and procedure.active == ["A-F"]
        assert screen.truth["A-F"] == procedure.truth["A-F"]
        for level in LEVELS:
            if not screen.confirmed[level]:
                continue
            key = (level, "A-F")
            assert screen.analysis[key] == procedure.analysis[key]
            assert screen.direction[key] == procedure.direction[key]
            for point in sim.N_POINTS:
                assert (
                    screen.covered[(point, level, "A-F")]
                    == procedure.covered[(point, level, "A-F")]
                )
                assert procedure.covered[(point, level, "P-F")] is None


def test_n_follows_the_rule_at_each_level(setup) -> None:
    o = sim.simulate_study(sim.SCENARIOS_BY_NAME["C02"], setup, 11, 0, resamples=200, levels=LEVELS)
    assert o.m >= 22 and o.spread_pp > 0
    for level in LEVELS:
        assert o.N[level] == sample_size(o.spread_pp, o.m, level)  # spread kept in memory only
        assert o.cap[level] == cap_of(o.N[level])


def test_a_failed_af_gate_is_unconfirmed_and_counted(setup) -> None:
    o = sim.simulate_study(
        replace(sim.SCENARIOS_BY_NAME["C02"], support=0.5),
        setup,
        11,
        0,
        resamples=200,
        levels=LEVELS,
    )
    assert not any(o.confirmed.values()) and set(o.why.values()) == {"af_gate"}
    assert o.gates["AF"] is False and o.covered == {}


def test_a_cap_beyond_the_seeds_is_unconfirmed_at_its_level(setup) -> None:
    o = sim.simulate_study(
        sim.SCENARIOS_BY_NAME["C02"], replace(setup, n_min=190), 13, 0, mode="procedure",
        resamples=200, levels=(0.975,),
    )  # fmt: skip
    assert o.N[0.975] >= 190 and o.cap[0.975] > 200
    assert o.confirmed == {0.975: False} and o.why == {0.975: "cap_beyond_seeds"}


def test_an_undetermined_interval_is_a_miss(setup, monkeypatch) -> None:
    def undetermined(d, *, level, resamples, seed, chunk):
        return Interval(len(d), 0.0, 0.0, 0.0, float(level), resamples, math.nan, math.nan,
                        0, 0, 0, True)  # fmt: skip

    monkeypatch.setattr(sim, "bootstrap_t", undetermined)
    o = sim.simulate_study(sim.SCENARIOS_BY_NAME["C02"], setup, 11, 0, resamples=200,
                           levels=LEVELS)  # fmt: skip
    assert o.covered and not any(o.covered.values())
    assert all(v is None for v in o.direction.values())


def test_the_outcome_row_round_trips(setup) -> None:
    import json

    for sc, mode, s in (
        (sim.SCENARIOS_BY_NAME["C02"], "screen", setup),
        (pf_gate_fails(), "procedure", setup),
        (replace(sim.SCENARIOS_BY_NAME["C02"], support=0.5), "screen", setup),
    ):
        o = sim.simulate_study(sc, s, 11, 1, mode=mode, resamples=200, levels=LEVELS)
        text = json.dumps(sim.outcome_json(o), allow_nan=False, sort_keys=True)
        assert "spread" not in text
        assert sim.outcome_from_json(json.loads(text)) == o
    with pytest.raises(ValueError):
        sim.outcome_from_json({**json.loads(text), "spread_pp": 3.0})


def test_the_screen_counts_a_joint_miss_when_either_comparison_misses(setup) -> None:
    o = sim.simulate_study(sim.SCENARIOS_BY_NAME["C02"], setup, 11, 2, resamples=200,
                           levels=LEVELS)  # fmt: skip
    for (point, level, comparison), value in o.covered.items():
        if comparison == "joint":
            assert value == (o.covered[(point, level, "A-F")] and o.covered[(point, level, "P-F")])


def test_procedure_mode_falls_back_to_candidate_2(setup) -> None:
    o = sim.simulate_study(af_fails_at_candidate_1(), setup, 13, 0, mode="procedure",
                           resamples=200, levels=(0.975,))  # fmt: skip
    assert o.candidate == 2 and o.rates_at_candidate(2) and o.rates_at_candidate(1)
    assert o.confirmed[0.975]
    screen = sim.simulate_study(af_fails_at_candidate_1(), setup, 13, 0, resamples=200,
                                levels=(0.975,))  # fmt: skip
    assert screen.candidate == 1 and screen.why == {0.975: "af_gate"}
    assert not screen.rates_at_candidate(2)


def test_procedure_mode_ends_in_the_scientific_report_when_candidate_2_fails(setup) -> None:
    sc = af_fails_at_candidate_1()
    failing = replace(setup, calibrations={**setup.calibrations, (sc, 2): hand_calibration(2, 0.6)})
    o = sim.simulate_study(sc, failing, 13, 0, mode="procedure", resamples=200,
                           levels=(0.975,))  # fmt: skip
    assert o.candidate == 2 and o.why == {0.975: "scientific_report"} and not o.covered


def test_procedure_mode_counts_pf_only_where_its_gate_passed(setup) -> None:
    o = sim.simulate_study(pf_gate_fails(), setup, 13, 0, mode="procedure", resamples=200,
                           levels=(0.975,))  # fmt: skip
    assert o.active == ["A-F"] and all(v is None for k, v in o.covered.items() if k[2] == "P-F")
    for (point, level, comparison), value in o.covered.items():
        if comparison == "joint":
            assert value == o.covered[(point, level, "A-F")]


def test_procedure_mode_raises_n_to_n_min(setup) -> None:
    o = sim.simulate_study(sim.SCENARIOS_BY_NAME["C02"], replace(setup, n_min=120), 13, 0,
                           mode="procedure", resamples=200, levels=(0.975,))  # fmt: skip
    assert o.N[0.975] >= 120
    screen = sim.simulate_study(sim.SCENARIOS_BY_NAME["C02"], replace(setup, n_min=120), 13, 0,
                                resamples=200, levels=(0.975,))  # fmt: skip
    assert screen.N[0.975] < 120  # the screen takes N from the rule alone


# ------------------------------------------------------------------ the calibration


def support_at_starting_rates(sc, setup, f) -> float:
    """The expected A–F support of a validation block at candidate 1's starting rates (no
    search): each block one vector, uniform with replacement, its four runs surviving f."""
    supported = []
    for v in setup.vectors:
        eps = [
            sim.synthetic_endpoint(1, "validation", None, arm, START_RATES[arm][lab], 10300,
                                   K=math.exp(v.ln_k(arm, START_RATES[arm][lab])), mmlu=1.0,
                                   factual=1.0, k_only=False)
            for arm in ("A", "F") for lab in LABELS
        ]  # fmt: skip
        supported.append(block_state(eps, 10300, 1, ["A", "F"])["AF"]["supported"])
    return float(np.mean(supported)) * (1 - f) ** 4


@pytest.mark.parametrize("name", ["C02", "C09", "C10", "G1"])
def test_the_calibrated_failure_gives_the_support_at_the_selected_rates(setup, name) -> None:
    sc = sim.SCENARIOS_BY_NAME[name]
    f, realized, flagged = sim.calibrate_failure(sc, setup, 1, SEED, studies=2_000)
    support, se = sim.realized_support(sc, setup, 1, f, SEED, studies=2_000)
    assert not flagged and 0.0 <= f < 1.0
    assert support == realized and abs(support - sc.support) <= 2 * se + 1e-4
    # the selection matters: the starting rates' support at this f is not what is calibrated
    assert support_at_starting_rates(sc, setup, f) != pytest.approx(support, abs=1e-12)


def test_the_realized_support_is_the_registered_block_states(setup) -> None:
    """The support the calibration counts is the A–F support ``block_state`` gives the studies'
    validation blocks with the failures applied (the same streams)."""
    sc = sim.SCENARIOS_BY_NAME["C02"]
    support, _ = sim.realized_support(sc, setup, 1, 0.07, SEED, studies=40)
    failing = replace(
        setup, calibrations={**setup.calibrations, (sc, 1): hand_calibration(1, 0.07)}
    )
    blocks = [
        sim.simulate_study(
            sc, failing, SEED, i, resamples=200, levels=LEVELS, validation_only=True, candidate=1
        ).m  # fmt: skip
        for i in range(40)
    ]
    assert support == pytest.approx(sum(blocks) / (40 * 28), abs=1e-12)


def test_natural_failures_beyond_one_minus_s_give_f_0_and_a_flag(setup) -> None:
    sc = replace(sim.SCENARIOS_BY_NAME["C10"], support=0.999)  # above the natural support
    f, realized, flagged = sim.calibrate_failure(sc, setup, 1, SEED, studies=2_000)
    assert (f, flagged) == (0.0, True) and realized < 0.999


def interpolated_utility_moments(sc, cal, setup, seed) -> dict:
    """Per arm, the interpolated MMLU loss's mean and SD and the factual loss's mean over the
    supported validation blocks of the same studies, from ``simulate_study`` itself."""
    use = replace(setup, calibrations={**setup.calibrations, (sc, 1): cal})
    values = {arm: [] for arm in ARM_ORDER}
    for i in range(2_000):
        o = sim.simulate_study(sc, use, seed, i, resamples=200, levels=LEVELS,
                               validation_only=True, candidate=1)  # fmt: skip
        for arm in ARM_ORDER:
            values[arm] += [(m, f) for _, m, f in o.utility[arm]]
    out = {}
    for arm, pairs in values.items():
        m = np.array([p[0] for p in pairs])
        f = np.array([p[1] for p in pairs])
        out[arm] = (float(m.mean()), float(m.std()), float(f.mean()))
    return out


def test_the_safe_alternative_holds_on_the_supported_interpolated_utility(setup) -> None:
    sc = sim.SCENARIOS_BY_NAME["G1"]
    cal = sim.calibrate_safe(sc, setup, 1, SEED, studies=2_000)
    for arm in ("A", "F", "P"):
        mean, sd, factual = cal.utility_realized[arm]
        assert (mean, sd, factual) == pytest.approx((3.0, 2.0, 2.0), abs=1e-6)
    # recomputed independently from the same studies' supported blocks
    again = interpolated_utility_moments(sc, cal, setup, SEED)
    for arm in ARM_ORDER:
        assert again[arm] == pytest.approx(cal.utility_realized[arm], abs=1e-9)
    assert not cal.utility_flagged and set(cal.mmlu_shift) == set(ARM_ORDER)
    assert {v.seed for v in setup.vectors} == set(cal.mmlu_shift["A"])


def paired_spread_at_selected_rates(sc, setup, cal, seed, studies) -> tuple[float, float, float]:
    """Study by study, the observed A–F s² with simulated labels at the Δ = 0 offsets (o⁰)
    against the calibration's exact E[s² | blocks] of the same study (pp²): the mean paired
    difference, its SE, and the simulated spread (the square root of the mean s²)."""
    use = replace(setup, calibrations={**setup.calibrations, (sc, 1): cal.at_null()})
    draws = sim._draws(sc, use, 1, seed, studies)
    exact, keep = sim.study_variances(sc, draws, cal.failure, cal.sigma_af, "A-F",
                                      cal.null_offset_a)  # fmt: skip
    simulated = []
    for i in np.flatnonzero(keep):
        o = sim.simulate_study(sc, use, seed, int(i), resamples=200, levels=LEVELS,
                               validation_only=True, candidate=1)  # fmt: skip
        assert o.m >= 2
        simulated.append(o.spread_pp**2)
    diff = np.array(simulated) - 1e4 * exact
    return (
        float(diff.mean()),
        float(diff.std(ddof=1) / math.sqrt(diff.size)),
        math.sqrt(float(np.mean(simulated))),
    )


@pytest.mark.parametrize("name", ["C02", "C07", "C13", "C16", "P18"])
def test_the_spread_and_delta_hold_at_the_selected_rates(setup, name) -> None:
    """σ is calibrated at Δ = 0, the offsets o⁰ at which the mean truth is 0 (C13's
    differential judge makes the truth at offset 0 nonzero): the exact spread at o⁰ is the
    planning spread, and each study's simulated s² matches its exact E[s² | blocks] (paired)."""
    sc = sim.SCENARIOS_BY_NAME[name]
    cal = sim.calibrate_scenario(sc, setup, 1, SEED, studies=2_000)
    assert cal.attained_af_pp == pytest.approx(sc.spread_pp, abs=1e-6)
    assert cal.attained_pf_pp == pytest.approx(sc.spread_pp, abs=1e-6)
    mean, se, spread = paired_spread_at_selected_rates(sc, setup, cal, SEED, 2_000)
    assert abs(mean) <= 3 * se and abs(spread - sc.spread_pp) <= 0.1
    # o⁰ gives the mean truth 0; the offsets give Δ
    use = replace(setup, calibrations={**setup.calibrations, (sc, 1): cal})
    draws = sim._draws(sc, use, 1, SEED, 2_000)
    for comparison, target in (("A-F", sc.delta_af_pp), ("P-F", sc.delta_pf_pp)):
        null, _ = sim._mean_truth(sc, setup, cal.at_null(), draws, comparison)
        assert abs(null) <= 1e-12
        mean, se = cal.attained_delta_pp[comparison]
        assert abs(mean - target) <= 1e-9
    if sc.delta_af_pp == 0:
        assert (cal.offset_a, cal.offset_p) == (cal.null_offset_a, cal.null_offset_p)
    if name == "C13":  # the truth at offset 0 is not 0: o⁰ is not 0
        assert abs(cal.null_offset_a) > 1e-4


def test_the_calibration_is_recomputed_identically(setup) -> None:
    sc = sim.SCENARIOS_BY_NAME["C16"]
    first = sim.calibrate_scenario(sc, setup, 1, SEED, studies=2_000)
    sim.clear_caches()
    assert sim.calibrate_scenario(sc, setup, 1, SEED, studies=2_000) == first


def test_sigma_is_brents_root_from_any_bracket(setup, monkeypatch) -> None:
    """σ is the root of the spread at the planning spread to Brent's tolerance, whether its
    bracket is [0, 0.5] or starts from a previous σ below, at or above it (each round of the
    σ/o⁰ alternation starts from the last), in a few evaluations; a bisection to a width of
    1e-13 finds the same σ."""
    sc = sim.SCENARIOS_BY_NAME["C02"]
    cal = hand_calibration(1)
    draws = sim._draws(sc, replace(setup, calibrations={**setup.calibrations, (sc, 1): cal}), 1,
                       SEED, 400)  # fmt: skip
    real, calls = sim._spread, [0]

    def counted(*args, **kwargs):
        calls[0] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sim, "_spread", counted)
    fresh = sim._solve_sigma(sc, draws, cal.failure, "A-F", 0.0)
    assert fresh > 0 and calls[0] <= 15
    assert real(sc, draws, cal.failure, fresh, "A-F")[0] == pytest.approx(sc.spread_pp, abs=1e-9)
    for start in (fresh, fresh * 0.97, fresh * 1.03, fresh * 0.2, fresh * 5):
        calls[0] = 0
        again = sim._solve_sigma(sc, draws, cal.failure, "A-F", 0.0, start)
        assert abs(again - fresh) <= 2 * sim.sigma_tolerance(fresh), start
        assert calls[0] <= 15
    lo, hi = 0.0, 0.5  # the bisection's σ, as it was solved before
    while hi - lo > 1e-13:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if real(sc, draws, cal.failure, mid, "A-F")[0] < sc.spread_pp else (
            lo, mid)  # fmt: skip
    assert abs((lo + hi) / 2 - fresh) <= 5e-14 + sim.sigma_tolerance(fresh)
    # σ is 0 when the labels and the geometry alone reach the spread
    low = replace(sc, spread_pp=0.1)
    assert sim._solve_sigma(low, draws, cal.failure, "A-F", 0.0, fresh) == 0.0


def test_the_sigma_offset_rounds_stop_within_the_solvers_precision() -> None:
    sigma, null = 0.054, -0.031
    tolerance = sim.sigma_tolerance(sigma)
    assert sim._settled((sigma + 2 * tolerance * 0.99, null), (sigma, null))
    assert sim._settled((sigma, null + sim.OFFSET_PRECISION), (sigma, null))
    assert not sim._settled((sigma + 1e-12, null), (sigma, null))
    assert not sim._settled((sigma, null + 3 * sim.OFFSET_PRECISION), (sigma, null))
    assert sim.SIGMA_PRECISION <= 1e-14  # Brent's tolerance is no looser than the bisection's


@pytest.mark.parametrize("name", ["C02", "C13", "C14"])
def test_the_label_variance_is_the_simulated_labels(setup, name) -> None:
    """The calibration takes the label noise of a block's difference through its exact variance
    (prompts independent given the block, pairs of endpoints through the bivariate normal); it
    matches the simulated labels' variance, an endpoint at h = 0 included, for the equal and
    the differential judge and for labels missing at random or among harmful responses."""
    sc = sim.SCENARIOS_BY_NAME[name]
    rng = np.random.default_rng(9)
    h = np.array([0.12, 0.2, 0.0, 0.18, 0.05, 0.3])
    arms = ["A", "A", "F", "F", "P", "P"]
    c = np.array([0.4, 0.6, -0.3, -0.7, 0.0, 0.0])
    draws = np.array([c @ sim.endpoint_labels(h, arms, sc, rng) for _ in range(20_000)])
    want = sim.label_variance(h[None, :4], c[None, :4], ("A", "A", "F", "F"), sc)[0]
    assert draws.var(ddof=1) == pytest.approx(want, rel=0.04)
    from scipy.stats import multivariate_normal

    for t1, t2, rho in ((-1.0, 0.3, 0.7), (0.5, -2.0, 0.3), (-40.0, 0.2, 0.7)):
        exact = multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([t1, t2])
        assert float(sim.bivariate_normal(np.array(t1), np.array(t2), rho)) == pytest.approx(
            exact, abs=1e-7
        )


# ------------------------------------------------------------------ the fixation


def checkout_with_registration(tmp_path: Path) -> Path:
    """A Git checkout holding a registration and an inputs record, committed."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "study3-preregistration.md").write_text(
        "# Study 3 preregistration\n\n### Amendment 1: x\n\n### Amendment 2: y\n"
    )
    (repo / "docs" / "study3-simulation-inputs.json").write_text('{"kind": "inputs"}\n')
    run_git(repo, "init", "-q")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-qm", "the registration and the inputs")
    return repo


def run_git(repo: Path, *args: str) -> str:
    command = ["git", "-c", "user.name=test", "-c", "user.email=test@example.org",
               "-c", "commit.gpgsign=false", *args]  # fmt: skip
    return subprocess.run(command, cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def a_calibration(candidate: int) -> sim.Calibration:
    return replace(
        hand_calibration(candidate, mmlu_shift={"A": {2100: 0.5}, "F": {2100: -0.25},
                                                "P": {2100: 1.0}}),
        utility_realized={"A": (3.0, 2.0, 2.0)}, utility_lambda={"A": 1.5},
        attained_delta_pp={"A-F": (0.0, 0.01)}, attained_spread_se_pp={"A-F": 0.02},
    )  # fmt: skip


def test_a_calibration_round_trips_through_json() -> None:
    import json

    cal = a_calibration(2)
    text = json.dumps(sim.calibration_json(cal), allow_nan=False, sort_keys=True)
    assert sim.calibration_from_json(json.loads(text)) == cal
    with pytest.raises(ValueError):
        sim.calibration_from_json({**json.loads(text), "extra": 1})


def test_the_fixation_is_built_committed_and_read_back(tmp_path, monkeypatch) -> None:
    import json

    from zero_field_refusal.analysis import study3_decisions as decisions

    repo = checkout_with_registration(tmp_path)
    scenarios = [sim.SCENARIOS_BY_NAME[n] for n in ("C01", "C15", "P18", "G1")]
    calibrations = {(sc, c): a_calibration(c) for sc in scenarios for c in (1, 2)
                    if c == sc.candidate or sc.candidate == 1}  # fmt: skip
    code = {"commit": run_git(repo, "rev-parse", "HEAD"), "gpu_code_sha256": "b" * 64}
    inputs = repo / "docs" / "study3-simulation-inputs.json"
    fixation = sim.build_fixation(
        inputs, scenarios, calibrations, code=code, streams=sim.STREAMS, repo=repo
    )
    assert fixation["J"] == 2 * 105 and fixation["tests"]["count"] == 210
    assert fixation["streams"] == {"calibration": 20261101, "verification": 20261102}
    assert fixation["policy"]["path"] == "docs/study3-preregistration.md"
    assert fixation["policy"]["amendments"] == [1, 2]
    assert fixation["inputs"]["path"] == "docs/study3-simulation-inputs.json"
    assert fixation["curvature"] == {
        "A": 9.537, "F": 11.124, "P": 11.124, "unit": "pp per (ln K)^2",
        "nominal_bias_af_pp": -0.0395,
        "source": "study 1's fitted curves (study2_short_sim.STUDY1) at K*, baseline 0.15; "
                  "M4, decided",
        "note": "the registration states about 0.06 pp, a fitted estimate with partial "
                "cancellation between arms",
    }  # fmt: skip
    assert set(fixation["rules"]) == {"screen_simplifications", "pf_verification",
                                      "joint_verification", "draw_bound"}  # fmt: skip
    assert fixation["rules"]["joint_verification"]["source"] == (
        "counted over the studies confirmed for A-F; a study misses jointly if A-F misses, or if "
        "P-F is active in it and misses; M27, decided"
    )
    for rule in fixation["rules"].values():
        assert rule["source"].endswith("decided") and " M" in rule["source"].rsplit(";", 1)[-1]
    assert fixation["max_draws_factor"] == 4 and fixation["gate_replicates"] == fixation["M"]
    assert fixation["calibration"] == {
        "studies": 10_000, "blocks": 28, "rounds": 10,
        "tolerances": {"support": 1e-4, "sigma": 1e-14, "sigma_relative": sim.SIGMA_RTOL,
                       "offset": 1e-12},
    }  # fmt: skip
    assert [s["name"] for s in fixation["scenarios"]] == ["C01", "C15", "P18", "G1"]
    assert set(fixation["scenarios"][1]["calibration"]) == {"2"}  # C15 starts at candidate 2
    path = repo / decisions.SIMULATION_FIXATION
    text = json.dumps(fixation, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text)
    with pytest.raises(ValueError, match="not committed"):
        sim.read_fixation(path, repo)
    run_git(repo, "add", decisions.SIMULATION_FIXATION)
    run_git(repo, "commit", "-qm", "the fixation")
    found = sim.read_fixation(decisions.SIMULATION_FIXATION, repo)
    assert found["fixation"] == fixation and found["commit"] == run_git(repo, "rev-parse", "HEAD")
    assert found["scenarios"] == scenarios
    assert found["calibrations"] == {(sc, c): cal for (sc, c), cal in calibrations.items()}
    # modified after its commit, or its tests edited: refused
    path.write_text(text.replace('"J": 210', '"J": 211'))
    with pytest.raises(ValueError, match="changed since its commit"):
        sim.read_fixation(path, repo)
    path.write_text(text)
    # a later commit that leaves the fixation alone keeps it readable (HEAD's ancestor holds it)
    (repo / "other.txt").write_text("x\n")
    run_git(repo, "add", "other.txt")
    run_git(repo, "commit", "-qm", "another file")
    assert sim.read_fixation(path, repo)["commit"] == found["commit"]
    # a code whose calibration's size or tolerances are not the fixation's does not read it
    monkeypatch.setattr(sim, "CALIBRATION_STUDIES", 1_000)
    with pytest.raises(ValueError, match="calibration"):
        sim.read_fixation(path, repo)
    monkeypatch.setattr(sim, "CALIBRATION_STUDIES", 10_000)
    monkeypatch.setattr(sim, "SIGMA_PRECISION", 1e-13)
    with pytest.raises(ValueError, match="calibration"):
        sim.read_fixation(path, repo)


def test_the_fixations_schema_is_closed(tmp_path) -> None:
    import json

    from zero_field_refusal.analysis import study3_decisions as decisions

    repo = checkout_with_registration(tmp_path)
    scenarios = [sim.SCENARIOS_BY_NAME["C02"]]
    inputs = repo / "docs" / "study3-simulation-inputs.json"
    code = {"commit": "a" * 40, "gpu_code_sha256": "b" * 64}
    with pytest.raises(ValueError, match="candidate 2"):  # the verification's fallback
        sim.build_fixation(inputs, scenarios, {(scenarios[0], 1): a_calibration(1)}, code=code,
                           streams=sim.STREAMS, repo=repo)  # fmt: skip
    fixation = sim.build_fixation(inputs, scenarios,
                                  {(scenarios[0], c): a_calibration(c) for c in (1, 2)},
                                  code=code, streams=sim.STREAMS, repo=repo)  # fmt: skip
    assert fixation["machine"] == sim.machine_record()
    assert set(fixation["machine"]) == {"platform", "cpu", "numpy"}
    only_first = {**fixation["scenarios"][0], "calibration": {
        "1": fixation["scenarios"][0]["calibration"]["1"]}}  # fmt: skip
    for change, word in (
        ({"extra": 1}, "fields"),
        ({"scenarios": [only_first]}, "candidate 2"),
        ({"curvature": {**fixation["curvature"], "A": 9.6}}, "curvature"),
        ({"curvature": {**fixation["curvature"], "note": "about 0.04 pp"}}, "curvature"),
        ({"rules": {**fixation["rules"], "draw_bound": {
            **fixation["rules"]["draw_bound"], "rule": "each test draws at most 5 M"}}}, "rules"),
        ({"machine": {"platform": "x"}}, "machine"),
        ({"calibration": {**fixation["calibration"], "studies": 0}}, "calibration.studies"),
        ({"calibration": {**fixation["calibration"], "rounds": 2.5}}, "calibration.rounds"),
        ({"calibration": {**fixation["calibration"], "tolerances": {"support": 1e-4}}},
         "calibration.tolerances"),
        ({"calibration": {**fixation["calibration"], "tolerances": {
            **fixation["calibration"]["tolerances"], "sigma": -1.0}}}, "calibration.tolerances"),
        ({"calibration": {"studies": 10_000}}, "calibration"),
        ({"J": 106}, "J"),
        ({"thresholds": {"comparison": 300, "joint": 573}}, "thresholds"),
        ({"streams": {"calibration": 5, "verification": 5}}, "streams"),
        ({"max_draws_factor": 5}, "max_draws_factor"),
        ({"tests": {**fixation["tests"], "list": fixation["tests"]["list"][:-1]}}, "tests.list"),
        ({"rules": {**fixation["rules"], "draw_bound": {"rule": "x", "source": "y"}}},
         "draw_bound"),
    ):  # fmt: skip
        problems = decisions._fixation_schema({**fixation, **change})
        assert problems and any(word in p for p in problems), (change, problems)
    assert decisions._fixation_schema(json.loads(json.dumps(fixation))) == []


def test_the_grid_of_the_proposal_has_its_thresholds() -> None:
    from zero_field_refusal.analysis import study3_decisions as decisions

    names = [s.name for s in sim.SCENARIOS if s.coverage]
    j = len(decisions.test_identities(names))
    assert j == 1785
    assert {n: decisions.coverage_threshold(j, 10_000, p) for n, p in
            decisions.COVERAGE_P0.items()} == {"comparison": 315, "joint": 590}  # fmt: skip
    assert sim.N_POINTS is decisions.N_POINTS and sim.TESTED is decisions.TESTED
    assert sim.MAX_DRAWS_FACTOR is decisions.MAX_DRAWS_FACTOR


# ------------------------------------------------------------------ the engine

ENGINE_COMMIT = "c" * 40
ENGINE_POLICY = {"path": "docs/study3-preregistration.md", "commit": "d" * 40,
                 "sha256": "e" * 64, "amendments": [1, 2, 3, 4, 5, 6]}  # fmt: skip
ENGINE_MACHINE = {"platform": "test", "cpu": "test", "numpy": "test"}


class Interrupted(Exception):
    """A run stopped partway (the test's interruption)."""


class Tiny(NamedTuple):
    """A tiny grid's context (two scenarios and the two gates, M = 6, 200 and 400 resamples:
    with 20, the interval is the range of 20 t*, and the joint test misses one study in five)
    and the test's monkeypatch."""

    context: sim.Context
    monkeypatch: pytest.MonkeyPatch


def engine_calibrations(scenarios, never: str | None = None) -> dict:
    """Hand calibrations: f 0.02 and σ 0.03; a power scenario's offsets ±0.06 (a large Δ: its
    power is not short); scenario ``never``'s f 0.6 (its A–F gate never passes)."""
    out = {}
    for sc in scenarios:
        for c in (1, 2) if sc.coverage and sc.candidate == 1 else (sc.candidate,):
            cal = hand_calibration(c, 0.6 if sc.name == never else 0.02)
            if sc.power:
                sign = 1 if sc.delta_af_pp > 0 else -1
                cal = replace(cal, offset_a=0.06 * sign, offset_p=0.06 * sign)
            out[(sc, c)] = cal
    return out


def engine_context(base: Path, scenarios, vectors, *, never: str | None = None) -> sim.Context:
    """The context of a fixation built (not committed) for ``scenarios`` in a folder."""
    import hashlib
    import json

    from zero_field_refusal.analysis.study3_seed_vectors import inputs_record, utility_slopes

    repo = base / "repo"
    (repo / "docs").mkdir(parents=True)
    inputs = repo / "docs" / "study3-simulation-inputs.json"
    inputs.write_text(json.dumps(inputs_record(vectors, {}, theta0={"path": "x"}, sources=[])))
    fixation = sim.build_fixation(
        inputs, scenarios, engine_calibrations(scenarios, never),
        code={"commit": ENGINE_COMMIT, "gpu_code_sha256": "b" * 64},
        streams={"calibration": 31, "verification": 32}, repo=repo, policy=ENGINE_POLICY,
        written_at="2026-10-30T00:00:00Z", machine=ENGINE_MACHINE)  # fmt: skip
    text = json.dumps(fixation, allow_nan=False, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    ref = {"path": "docs/study3-simulation-fixation.json", "sha256": digest,
           "commit": ENGINE_COMMIT}  # fmt: skip
    return sim.context_of(json.loads(text), ref, vectors, utility_slopes(vectors))


def tiny_grid(never: str | None = None) -> list:
    first = replace(sim.SCENARIOS_BY_NAME["C02"], name=never) if never else (
        sim.SCENARIOS_BY_NAME["C02"])  # fmt: skip
    names = ("C16", "G1", "G2")
    return [first, *(sim.SCENARIOS_BY_NAME[n] for n in names)]


@pytest.fixture
def tiny(tmp_path, monkeypatch, vectors) -> Tiny:
    from zero_field_refusal.analysis import study3_decisions as decisions

    monkeypatch.setattr(decisions, "COVERAGE_RUNS", 6)
    monkeypatch.setattr(sim, "SCREEN_RESAMPLES", 200)
    monkeypatch.setattr(sim, "VERIFICATION_RESAMPLES", 400)
    return Tiny(engine_context(tmp_path / "tiny", tiny_grid(), vectors), monkeypatch)


def tiny_never_confirms(tiny: Tiny, base: Path, vectors) -> Tiny:
    """The tiny grid with its coverage scenario NC, whose A–F gate never passes."""
    return Tiny(engine_context(base, tiny_grid("NC"), vectors, never="NC"), tiny.monkeypatch)


def run_screen(tiny: Tiny, root: Path, *, workers: int, unit: int,
               stop_after_units: int | None = None) -> Path | None:  # fmt: skip
    """The screen (``run_screen``: gates, screen, scan) in ``root`` with ``unit`` studies per
    part; with ``stop_after_units``, interrupted once that many parts are written."""
    tiny.monkeypatch.setattr(sim, "UNIT", unit)
    if stop_after_units is None:
        return sim.run_screen(tiny.context, root, ENGINE_COMMIT, workers=workers)
    real, left = sim.execute, [stop_after_units]

    def limited(units, *args, **kwargs):
        units = list(units)
        take = units[: left[0]]
        real(take, *args, **kwargs)
        left[0] -= len(take)
        if left[0] <= 0:
            raise Interrupted

    tiny.monkeypatch.setattr(sim, "execute", limited)
    with pytest.raises(Interrupted):
        sim.run_screen(tiny.context, root, ENGINE_COMMIT, workers=workers)
    tiny.monkeypatch.setattr(sim, "execute", real)
    assert not (root / "study3" / "simulation" / "screen.json").exists()
    return None


def rows_of(path: Path) -> list[dict]:
    import json

    return json.loads(path.read_text())["tests"]


def test_the_screen_does_not_depend_on_workers_or_unit_size(tiny, tmp_path) -> None:
    a = run_screen(tiny, tmp_path / "a", workers=1, unit=2)
    b = run_screen(tiny, tmp_path / "b", workers=3, unit=5)
    c = run_screen(tiny, tmp_path / "c", workers=4, unit=2)
    assert a.read_bytes() == b.read_bytes() == c.read_bytes()
    assert a == tmp_path / "a" / "study3" / "simulation" / "screen.json"


def test_an_interrupted_screen_resumes_to_the_same_bytes(tiny, tmp_path) -> None:
    whole = run_screen(tiny, tmp_path / "whole", workers=2, unit=2)
    run_screen(tiny, tmp_path / "cut", workers=2, unit=2, stop_after_units=3)  # raises partway
    parts = sorted((tmp_path / "cut" / "study3" / "simulation" / "parts").rglob("*.json"))
    assert len(parts) == 3
    again = run_screen(tiny, tmp_path / "cut", workers=2, unit=2)
    assert again.read_bytes() == whole.read_bytes()
    for part in parts:  # the finished parts were kept, not drawn again
        assert part.read_bytes() == (tmp_path / "whole" / part.relative_to(tmp_path / "cut")
                                     ).read_bytes()  # fmt: skip


def test_a_part_of_another_commit_or_fixation_is_named(tiny, tmp_path) -> None:
    ctx = tiny.context
    unit = sim.Unit("gates", "G1", 0, 2, 31, 200, sim.LEVELS, 28, 60)
    sim.execute([unit], tmp_path, ctx.ref, ENGINE_COMMIT, workers=1, context=ctx)
    path = sim.part_path(tmp_path, "gates", "G1", 0)
    assert path.is_file()
    found, problems = sim.existing_parts(tmp_path, "gates", ctx.ref, ENGINE_COMMIT)
    assert found == {("G1", 0): path} and problems == []
    found, problems = sim.existing_parts(tmp_path, "gates", ctx.ref, "d" * 40)
    assert found == {} and any(str(path) in p and "commit" in p for p in problems)
    other = {**ctx.ref, "sha256": "f" * 64}
    _, problems = sim.existing_parts(tmp_path, "gates", other, ENGINE_COMMIT)
    assert any(str(path) in p and "fixation" in p for p in problems)
    with pytest.raises(sim.SimulationError, match=str(path)):
        sim.run_screen(ctx, tmp_path, "d" * 40, workers=1)
    assert not (tmp_path / "study3" / "simulation" / "screen.json").exists()
    with sim.Workers(1, ctx), pytest.raises(sim.SimulationError, match="written once"):
        sim.run_unit((unit, tmp_path, ctx.ref, ENGINE_COMMIT))  # a part is never drawn again
    with pytest.raises(sim.SimulationError, match="no context"):  # outside a run's workers
        sim.run_unit((unit, tmp_path, ctx.ref, ENGINE_COMMIT))


def outcome(index: int, confirmed: bool) -> sim.StudyOutcome:
    level = 0.975
    return sim.StudyOutcome(index=index, mode="screen", candidate=1, active=["A-F", "P-F"],
                            levels=(level,), rates={}, gates={}, m=0,
                            confirmed={level: confirmed}, why={level: None}, N={level: None},
                            cap={level: None}, blocks={level: None}, covered={}, direction={},
                            readings={}, utility_claim={}, truth={})  # fmt: skip


def test_the_first_m_confirmed_by_index_are_used() -> None:
    rows = [outcome(i, i % 3 != 0) for i in range(12)]
    shuffled = rows[::2] + rows[1::2]
    assert [o.index for o in sim.first_confirmed(shuffled, 4, 0.975)] == [1, 2, 4, 5]
    assert [o.index for o in sim.first_confirmed(rows, 100, 0.975)] == [
        i for i in range(12) if i % 3]  # fmt: skip


def test_the_blocks_quantiles_are_type_1_ranks() -> None:
    values = list(range(1, 11))
    assert [sim.order_statistic(values, q) for q in (0.1, 0.5, 0.9)] == [1, 5, 9]
    assert sim.order_statistic([7], 0.9) == 7 and sim.order_statistic([3, 1, 2], 0.5) == 2


def test_the_results_hold_every_identity_once(tiny, tmp_path) -> None:
    from zero_field_refusal.analysis import study3_decisions as decisions

    ctx = tiny.context
    tiny.monkeypatch.setattr(sim, "UNIT", 4)
    screen = sim.run_screen(ctx, tmp_path, ENGINE_COMMIT, workers=2)
    verification = sim.run_verification(ctx, tmp_path, ENGINE_COMMIT, workers=2)
    import json

    s, v = json.loads(screen.read_text()), json.loads(verification.read_text())
    code = {"commit": ENGINE_COMMIT, "gpu_code_sha256": "b" * 64}
    results = sim.results_record(s, v, ctx, code, written_at="2026-10-30T01:00:00Z")
    assert set(results) == set(decisions.RESULTS_FIELDS)
    assert [tuple(r["test"]) for r in results["tests"]] == [
        tuple(i) for i in ctx.fixation["tests"]["list"]]  # fmt: skip
    assert decisions._identities(results["tests"], ctx.fixation) == []
    decided, problems = decisions.coverage_decisions(results, ctx.fixation)
    assert problems == [] and decided["level"] == results["selected_calibration_level"]
    assert decided["level"] is not None and decided["validation_blocks"] in (28, 40)
    assert results["M"] == 6 and results["J"] == 210
    assert all(r["studies"] == 6 for r in results["tests"] if r["due"])
    # the verification's rows are at L, and every comparison counts its own studies
    due = [r for r in results["tests"] if r["test"][0] == "verification"]
    assert {r["level"] for r in due} == {decided["level"]}
    assert set(results["reported"]) == {"C02", "C16", "G1", "G2"}
    assert set(results["reported"]["C02"]["blocks"]["0.975"]) == {"q10", "q50", "q90"}
    # the verification does not depend on the workers either
    other = tmp_path / "other"
    sim.run_screen(ctx, other, ENGINE_COMMIT, workers=1)
    again = sim.run_verification(ctx, other, ENGINE_COMMIT, workers=4)
    assert again.read_bytes() == verification.read_bytes()


def test_a_scenario_that_never_confirms_stops_at_4m_and_is_not_attainable(
    tiny, tmp_path, vectors
) -> None:
    never = tiny_never_confirms(tiny, tmp_path / "never", vectors)
    rows = run_screen(never, tmp_path, workers=2, unit=2)  # A–F gate never passes
    test = rows_of(rows)[0]
    assert test["test"][:2] == ["calibration", "NC"]
    assert (test["draws"], test["studies"], test["reason"], test["passed"]) == \
        (4 * 6, 0, "not_attainable", False)  # fmt: skip
    import json

    screen = json.loads(rows.read_text())
    assert screen["level"] is None  # every level fails: X = M counted
    assert screen["reported"]["NC"]["unconfirmed"] == {"af_gate": 24}


def test_the_scan_stops_at_the_first_n_with_the_power(tiny, tmp_path) -> None:
    """A power scenario whose cap readings detect only from N = 63 (one in three before): the
    power is short, the scan runs n = 60, 61, … and stops at 63, its first n with the power."""
    from zero_field_refusal.analysis import study3_decisions as decisions

    real = sim.simulate_study

    def slow_power(scenario, setup, base, index, *, n_floor=60, **kw):
        o = real(scenario, setup, base, index, n_floor=n_floor, **kw)
        if scenario.name != "C16" or not o.direction:
            return o
        hit = n_floor >= 63 or index % 3 == 0
        direction = {k: (("positive" if hit else None) if k[1] == "A-F" else v)
                     for k, v in o.direction.items()}  # fmt: skip
        return replace(o, direction=direction)

    tiny.monkeypatch.setattr(sim, "simulate_study", slow_power)
    path = run_screen(tiny, tmp_path, workers=1, unit=3)
    import json

    screen = json.loads(path.read_text())
    scan = screen["gates"]["n_scan"]
    assert [entry["n"] for entry in scan] == [60, 61, 62, 63]
    assert [min(c["correct"] for c in e["by_scenario"].values()) for e in scan][-1] == 6
    assert all(min(c["correct"] for c in e["by_scenario"].values()) < 5 for e in scan[:-1])
    assert decisions._n_min(scan, True, ["C16"]) == (63, [])


def test_the_calibration_does_not_depend_on_the_workers(vectors, monkeypatch) -> None:
    """Every scenario's calibration, one pair per worker and C12's truths over a pool of their
    own, is the same with one worker and with four; C12's calibration builds each of its jitter
    quadratures once (kept across its σ and offset steps) and forgets them when it ends."""
    from zero_field_refusal.analysis.study3_seed_vectors import utility_slopes

    few = world_vectors(count=6)
    setup = sim.Setup(vectors=tuple(few), slopes=utility_slopes(few), calibrations={}, M=6,
                      levels=LEVELS, validation_blocks=28)  # fmt: skip
    pairs = [(sim.SCENARIOS_BY_NAME[n], 1) for n in ("C02", "C12", "G1")]
    sim.clear_caches()
    built = []
    real = sim._JitterGrid.__init__

    def counted(self, vector, arm, rates, *args, **kwargs):
        built.append((vector.seed, vector.arms[arm], arm, rates["l"], rates["u"]))
        real(self, vector, arm, rates, *args, **kwargs)

    monkeypatch.setattr(sim._JitterGrid, "__init__", counted)
    one = sim.calibrate_all(setup, 41, pairs=pairs, workers=1, studies=400)
    monkeypatch.setattr(sim._JitterGrid, "__init__", real)
    assert built and len(built) == len(set(built))  # each built once, none again
    assert sim._GRIDS == {} and sim._GRIDS_LIMIT == sim.GRIDS_KEPT
    sim.clear_caches()
    four = sim.calibrate_all(setup, 41, pairs=pairs, workers=4, studies=400)
    assert one == four and set(one) == set(pairs)
    assert sim.calibration_problems(one, one) == []


def test_the_recomputed_calibration_is_compared_at_a_relative_1e9() -> None:
    cal = a_calibration(1)
    sc = sim.SCENARIOS_BY_NAME["C01"]
    near = replace(cal, sigma_af=cal.sigma_af * (1 + 1e-12))
    far = replace(cal, sigma_af=cal.sigma_af * (1 + 1e-6))
    assert sim.calibration_problems({(sc, 1): near}, {(sc, 1): cal}) == []
    assert any("sigma_af" in p for p in sim.calibration_problems({(sc, 1): far}, {(sc, 1): cal}))
    flagged = replace(cal, support_flagged=True)
    assert any("support_flagged" in p
               for p in sim.calibration_problems({(sc, 1): flagged}, {(sc, 1): cal}))  # fmt: skip
    seeds = replace(cal, mmlu_shift={"A": {2101: 0.5}, "F": {2100: -0.25}, "P": {2100: 1.0}})
    assert any("mmlu_shift" in p
               for p in sim.calibration_problems({(sc, 1): seeds}, {(sc, 1): cal}))  # fmt: skip
    assert any("missing" in p for p in sim.calibration_problems({}, {(sc, 1): cal}))


def test_a_result_near_zero_is_compared_within_its_solvers_precision() -> None:
    """A near-zero result (the attained Δ at Δ = 0, an offset o⁰ near 0) moved in its last bits
    on another machine is the same result; a move beyond its field's floor is refused."""
    sc = sim.SCENARIOS_BY_NAME["C02"]
    near = {"A-F": (-2.5426366452122966e-11, 0.0015577865794015396),
            "P-F": (1.3574346218829063e-11, 0.0017754666554159167)}  # fmt: skip
    cal = replace(a_calibration(1), null_offset_p=-3.2e-14, attained_delta_pp=near)

    def moved(**change) -> list[str]:
        return sim.calibration_problems({(sc, 1): replace(cal, **change)}, {(sc, 1): cal})

    delta = cal.attained_delta_pp
    assert moved(attained_delta_pp={c: (v[0] + 1e-15, v[1]) for c, v in delta.items()}) == []
    beyond = moved(attained_delta_pp={c: (v[0] + 2e-9, v[1]) for c, v in delta.items()})
    assert len(beyond) == 2 and all("attained_delta_pp" in p for p in beyond)
    assert moved(null_offset_p=cal.null_offset_p + 1e-13) == []
    assert any("null_offset_p" in p for p in moved(null_offset_p=cal.null_offset_p + 1e-10))
    assert moved(failure=cal.failure + 5e-12) == []
    assert any("failure" in p for p in moved(failure=cal.failure + 1e-9))
    # every float the calibration holds has its floor; the integers and flags have none
    floats = {f.name for f in fields(sim.Calibration)} - {"candidate", "support_flagged",
                                                          "utility_flagged"}  # fmt: skip
    assert set(sim.CALIBRATION_FLOORS) == floats
    pp = ("attained_delta_pp", "attained_af_pp", "attained_pf_pp")
    proportions = ("failure", "offset_a", "offset_p", "null_offset_a", "null_offset_p")
    assert {sim.CALIBRATION_FLOORS[n] for n in pp} == {1e-9}
    assert {sim.CALIBRATION_FLOORS[n] for n in proportions} == {1e-11}
