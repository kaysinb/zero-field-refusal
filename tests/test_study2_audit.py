"""Study 2's terminal audit of the judge (amendment 4; the audit design v2.1): the frame, the
allocation and its SRSWOR draw, the HT estimate and its variance, the hypergeometric worst-case
interval with its sign-aware missing references, the reference harm, the fixation record's
``audit`` key, and the command line's (analysis.study2_audit_cli) draw (behind the summary's final
analysis, with D7's blinding) and estimate. Synthetic frames, labels and texts only: no real
label, prompt or response."""

import json
import math
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from test_study2_labels import IDENTITY, IDS, JUDGE, PROMPTS, make_run, record, write_labels
from test_study2_short_campaigns import FIXATION, LOOK_SETTINGS, summary_module

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import study2_audit, study2_audit_cli
from zero_field_refusal.analysis.study2_audit import (
    ESTIMANDS,
    SIDES,
    agreement_check,
    allocate,
    audit_cohorts,
    audit_estimates,
    audit_frame,
    bias_bounds,
    bias_estimate,
    bracket_runs,
    coefficients,
    draw_audit,
    estimands,
    hypergeometric_lower,
    hypergeometric_upper,
    judge_point,
    reference_harm,
    reference_readings,
    reference_value,
)
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    ARMS,
    AUDIT_FIXED,
    AUDIT_KEYS,
    FA,
    FIXATION_KEYS,
    PREWARMED,
    PRIMARY_ARMS,
    look_blocks,
    looks,
    mechanism_state,
    read_fixation,
)
from zero_field_refusal.paths import REPO_ROOT

HARMFUL = "harmful_compliance"
SIZES = FIXATION["audit"]["sizes"]
SEED = FIXATION["audit"]["seed"]
LEVEL = FIXATION["audit"]["bias_level"]
K_STAR = 6.3
RATES = [1.2e-5, 1.5e-5, 1.8e-5]


# ------------------------------------------------------------------ synthetic worlds


def cohort_blocks(rng, blocks: int, arms, prompts: int, harm=None, first: int = 1):
    """``progress``-like confirmation blocks, every arm supported with a random bracket around
    K*, and the judge's synthetic label of every prompt of each bracket run (2 % unparsed)."""
    harm = harm or {ADAMW: 0.3, FA: 0.3, PREWARMED: 0.35}
    confirmation, labels = [], {}
    for number in range(first, first + blocks):
        supports = {}
        for arm in arms:
            u, v = rng.uniform(0.05, 0.4, 2)
            runs = {side: f"{arm}-b{number}-{side}" for side in SIDES}
            bracket = {"lo": {"run": runs["lo"], "rate": RATES[0], "K": K_STAR * math.exp(-u)},
                       "hi": {"run": runs["hi"], "rate": RATES[1], "K": K_STAR * math.exp(v)},
                       "width": u + v}  # fmt: skip
            supports[arm] = {"supported": True, "cause": None, "bracket": bracket}
            for run in runs.values():
                draws = rng.random(prompts)
                labels[run] = {f"p{j:03d}": (None if x < 0.02 else HARMFUL if x < 0.02 + harm[arm]
                                             else "refusal")
                               for j, x in enumerate(draws)}  # fmt: skip
        confirmation.append({"block": number, "seed": 2999 + number, "runs": 3 * len(arms),
                             "complete": True, "supported": True,
                             "three_arm_supported": len(arms) == 3, "supports": supports,
                             "exhausted": []})  # fmt: skip
    return confirmation, labels


def label_rows(labels: dict) -> dict:
    """The label rows (``look_blocks``' counts) of item labels."""
    return {run: [{"condition": "step-25", "step": 25, "prompts": len(found),
                   "harmful": sum(1 for x in found.values() if x == HARMFUL),
                   "missing": sum(1 for x in found.values() if x is None)}]
            for run, found in labels.items()}  # fmt: skip


def world(monkeypatch, *, three_arm=True, blocks=60, prompts=20, seed=0, harm=None):
    """The summary's state, looks and final analysis over synthetic blocks, computed by Task 3's
    functions (``look_blocks``, ``looks``, ``mechanism_state``, the summary's ``final_analysis``),
    and the judge's item labels of every bracket run."""
    rng = np.random.default_rng(seed)
    arms = ARMS if three_arm else PRIMARY_ARMS
    confirmation, labels = cohort_blocks(rng, blocks, arms, prompts, harm)
    rows = label_rows(labels)
    module = summary_module()
    monkeypatch.setattr(module, "read_timecourse_labels", lambda state, courses, judge: ({}, {}))
    state = {"next": {"campaign": "confirmation"},
             "frozen": {"three_arm": three_arm, "K_star": K_STAR, "batch": 20,
                        "rates": {arm: RATES for arm in arms}},
             "confirmation": confirmation,
             "mechanism": mechanism_state(confirmation, three_arm=three_arm),
             "timecourse": {"trajectories": []}}  # fmt: skip
    entries = look_blocks(confirmation, rows, K_STAR)
    found = {"available": True, "looks": looks(entries, FIXATION["levels"], **LOOK_SETTINGS),
             "blocks": entries}  # fmt: skip
    final = module.final_analysis(state, [], [], rows, FIXATION, found, JUDGE, {})
    return state, found, final, labels


def flat_frame(rng, *, blocks=10, prompts=40, arms=PRIMARY_ARMS, harm=0.3):
    """A frame for the statistical tests (one cohort, ``primary``), from ``audit_frame``."""
    confirmation, labels = cohort_blocks(rng, blocks, arms, prompts,
                                         dict.fromkeys(ARMS, harm))  # fmt: skip
    cohort = {"K_star": K_STAR, "primary": confirmation, "mechanism": None}
    if len(arms) == 3:
        cohort["mechanism"] = confirmation
    return audit_frame(cohort, labels)


def truth(rng, frame, rates, *, adversarial=None) -> np.ndarray:
    """A reference label of every frame item: errors at ``rates[judge_label]`` per stratum, at
    random or, with ``adversarial`` (a coefficient vector), on the items whose σ c is largest."""
    y = np.array([item["y_judge"] for item in frame])
    errors = np.zeros(len(frame), dtype=int)
    for name in {item["stratum"] for item in frame}:
        members = [item["index"] for item in frame if item["stratum"] == name]
        label = name.rsplit("/", 1)[1]
        count = round(rates[label] * len(members))
        if adversarial is None:
            chosen = rng.choice(members, count, replace=False)
        else:
            sigma = study2_audit.SIGMA[label]
            chosen = sorted(members, key=lambda i: -sigma * adversarial[i])[:count]
        errors[np.asarray(chosen, dtype=int)] = 1
    return np.where(errors == 1, 1 - y, y)


def true_bias(frame, c, reference) -> float:
    return math.fsum(c[i] * (item["y_judge"] - reference[i]) for i, item in enumerate(frame))


def sampled_reference(sample, reference, missing=None) -> dict:
    return {e["index"]: None if missing is not None and e["index"] in missing
            else int(reference[e["index"]]) for e in sample["items"]}  # fmt: skip


# ------------------------------------------------------------------ the frame


def test_the_frame_reproduces_the_final_analysis(monkeypatch) -> None:
    """The frame's judge-rated Δ, C1 and C2 (Σ c_i y_J,i) are the final analysis's to 1e-12; an
    item of a block in both cohorts is counted once; each arm's coefficients sum to its sign."""
    state, found, final, labels = world(monkeypatch, harm={ADAMW: 0.3, FA: 0.3, PREWARMED: 0.36})
    cohorts = audit_cohorts(state, found, final)
    stopping = found["looks"]["looks"][-1]
    assert [b["block"] for b in cohorts["primary"]] == list(range(1, stopping["started"] + 1))
    assert [b["block"] for b in cohorts["mechanism"]] == list(range(1, 61))
    assert set(bracket_runs(cohorts)) == set(labels)
    frame = audit_frame(cohorts, labels)
    assert len(frame) == 60 * 3 * 2 * 20  # deduplicated: the primary's blocks are counted once
    assert len({(i["run"], i["prompt"]) for i in frame}) == len(frame)
    both = [i for i in frame if i["block"] <= stopping["started"] and i["arm"] != PREWARMED]
    assert all(i["cohorts"] == ["primary", "mechanism"] for i in both)
    assert {tuple(i["cohorts"]) for i in frame if i["arm"] == PREWARMED} == {("mechanism",)}
    for name, expected in cohorts["judge"].items():
        assert judge_point(frame, coefficients(frame, name)) == pytest.approx(expected, abs=1e-12)
    assert cohorts["judge"]["delta"] == stopping["mean"]
    assert cohorts["judge"]["C1"] == final["mechanism"]["C1"]["mean"]
    for name in estimands(frame):
        c = coefficients(frame, name)
        for arm, sign in ESTIMANDS[name]["signs"].items():
            total = math.fsum(c[i["index"]] for i in frame if i["arm"] == arm)
            assert total == pytest.approx(sign, abs=1e-12)
    # The weights are interpolate's: w_lo + w_hi = 1, in [0, 1].
    pairs = {}
    for item in frame:
        pairs.setdefault((item["arm"], item["block"]), {})[item["side"]] = item["w"]
    assert all(0 <= w["lo"] <= 1 and w["lo"] + w["hi"] == pytest.approx(1) for w in pairs.values())
    # An arm's own harm on the mechanism cohort is the mean of its blocks' h.
    h = [b["h"][PREWARMED] for b in final["mechanism"]["blocks"]]
    point = judge_point(frame, coefficients(frame, f"{PREWARMED}@mechanism"))
    assert point == pytest.approx(math.fsum(h) / len(h), abs=1e-12)


def test_the_frame_refuses_what_it_cannot_read(monkeypatch) -> None:
    state, found, final, labels = world(monkeypatch, blocks=60, prompts=4)
    cohorts = audit_cohorts(state, found, final)
    run = cohorts["primary"][0]["supports"][ADAMW]["bracket"]["lo"]["run"]
    with pytest.raises(ValueError, match="no judge labels"):
        audit_frame(cohorts, {k: v for k, v in labels.items() if k != run})
    with pytest.raises(ValueError, match="not a judge label"):
        audit_frame(cohorts, {**labels, run: {"p000": "unsure"}})
    with pytest.raises(ValueError, match="after the final analysis"):
        audit_cohorts(state, found, {"available": False, "reason": "not yet: x"})


# ------------------------------------------------------------------ allocation and the draw


def test_allocation_is_equal_and_reallocates_a_short_stratum() -> None:
    def counts(lo_h, lo_n, hi_h, hi_n):
        return {("lo", "harmful"): lo_h, ("lo", "not_harmful"): lo_n,
                ("hi", "harmful"): hi_h, ("hi", "not_harmful"): hi_n}  # fmt: skip

    assert allocate(counts(500, 900, 400, 1000), 400) == counts(100, 100, 100, 100)
    assert allocate(counts(500, 900, 400, 1000), 200) == counts(50, 50, 50, 50)
    # A short stratum is taken whole; its remainder goes to the other label of its side...
    assert allocate(counts(30, 900, 400, 1000), 400) == counts(30, 170, 100, 100)
    # ... then to the other side, split equally, ...
    assert allocate(counts(30, 120, 400, 1000), 400) == counts(30, 120, 125, 125)
    # ... spilling to the stratum with room.
    assert allocate(counts(30, 120, 110, 1000), 400) == counts(30, 120, 110, 140)
    assert allocate(counts(500, 900, 0, 60), 400) == counts(170, 170, 0, 60)
    # A stratum of N_h < n_h on each side; everything short: the whole frame.
    assert allocate(counts(10, 20, 30, 40), 400) == counts(10, 20, 30, 40)
    assert allocate(counts(0, 0, 0, 0), 400) == counts(0, 0, 0, 0)
    with pytest.raises(ValueError):
        allocate(counts(1, 1, 1, 1), 402)


def test_every_sampled_item_has_pi_n_over_N(monkeypatch) -> None:
    """π = n_h/N_h for every item; a stratum with N_h < n_h is taken whole (π = 1); the strata
    partition each arm's frame; the draw is reproducible and its streams are per stratum."""
    rng = np.random.default_rng(1)
    frame = flat_frame(rng, blocks=8, prompts=30, arms=ARMS, harm=0.12)
    drawn = draw_audit(frame, SIZES, SEED)
    assert drawn == draw_audit(frame, SIZES, SEED)
    for arm, key in ((ADAMW, "adamw"), (FA, "fa"), (PREWARMED, "prewarmed")):
        strata = [s for s in drawn["strata"] if s["arm"] == arm]
        assert len(strata) == 4
        assert sorted(i for s in strata for i in s["members"]) == [
            i["index"] for i in frame if i["arm"] == arm]  # fmt: skip
        assert sum(s["n"] for s in strata) == SIZES[key] == drawn["arms"][arm]["drawn"]
        harmful = [s for s in strata if s["judge_label"] == "harmful"]
        assert all(s["N"] < SIZES[key] // 4 and s["pi"] == 1 for s in harmful if arm != PREWARMED)
    by_index = {e["index"]: e for e in drawn["items"]}
    for stratum in drawn["strata"]:
        assert len(stratum["sampled"]) == len(set(stratum["sampled"])) == stratum["n"]
        assert set(stratum["sampled"]) <= set(stratum["members"])
        assert stratum["stream_seed"] == study2_audit.derive_seed(
            SEED, stratum["arm"], stratum["side"], stratum["judge_label"])  # fmt: skip
        for index in stratum["sampled"]:
            assert by_index[index]["pi"] == stratum["n"] / stratum["N"]
            assert frame[index]["stratum"] == stratum["stratum"] == by_index[index]["stratum"]
    other = draw_audit(frame, SIZES, SEED + 1)
    assert [e["index"] for e in other["items"]] != [e["index"] for e in drawn["items"]]


def test_the_two_arm_case_draws_no_prewarmed_item(monkeypatch) -> None:
    state, found, final, labels = world(monkeypatch, three_arm=False, prompts=20)
    cohorts = audit_cohorts(state, found, final)
    assert cohorts["mechanism"] is None and set(cohorts["judge"]) == {"delta"}
    frame = audit_frame(cohorts, labels)
    assert {i["arm"] for i in frame} == {ADAMW, FA}
    assert {tuple(i["cohorts"]) for i in frame} == {("primary",)}
    drawn = draw_audit(frame, SIZES, SEED)
    assert drawn["arms"][PREWARMED] == {
        "size": 200,
        "frame": 0,
        "drawn": 0,
        "reason": "no frame (a two-arm selection): not drawn",
    }
    assert {s["arm"] for s in drawn["strata"]} == {ADAMW, FA}
    assert len(drawn["items"]) == 800
    assert estimands(frame) == ["delta", f"{ADAMW}@primary", f"{FA}@primary"]
    reference = {e["index"]: frame[e["index"]]["y_judge"] for e in drawn["items"]}
    found = audit_estimates(frame, drawn, reference, LEVEL)
    assert set(found) == set(estimands(frame)) and "reference_harm" in found["delta"]


# ------------------------------------------------------------------ the HT estimate


def test_ht_is_unbiased_and_its_variance_is_srswor() -> None:
    """Over repeated draws (new seeds) on a fixed synthetic frame with known d, the HT estimate's
    mean is B and its variance the stratified SRSWOR variance, which V̂ estimates."""
    rng = np.random.default_rng(2)
    frame = flat_frame(rng, blocks=6, prompts=60)
    c = coefficients(frame, "delta")
    reference = truth(rng, frame, {"harmful": 0.15, "not_harmful": 0.06})
    target = true_bias(frame, c, reference)
    estimates, variances = [], []
    for seed in range(2000):
        drawn = draw_audit(frame, SIZES, seed)
        found = bias_estimate(drawn, sampled_reference(drawn, reference), c, level=LEVEL)
        assert found["lower"] == found["upper"]  # no missing reference
        estimates.append(found["lower"]["estimate"])
        variances.append(found["lower"]["se"] ** 2)
    drawn = draw_audit(frame, SIZES, 0)
    exact = 0.0
    for stratum in drawn["strata"]:
        big, small = stratum["N"], stratum["n"]
        values = [c[i] * (frame[i]["y_judge"] - reference[i]) for i in stratum["members"]]
        exact += big * big * (1 - small / big) * np.var(values, ddof=1) / small
    mean, spread = float(np.mean(estimates)), float(np.std(estimates, ddof=1))
    assert abs(mean - target) < 4 * spread / math.sqrt(len(estimates))
    assert float(np.var(estimates, ddof=1)) / exact == pytest.approx(1, abs=0.15)
    assert float(np.mean(variances)) / exact == pytest.approx(1, abs=0.05)


def test_ht_collapses_without_errors_and_reports_both_completions() -> None:
    rng = np.random.default_rng(3)
    frame = flat_frame(rng, blocks=4, prompts=60)
    c = coefficients(frame, "delta")
    drawn = draw_audit(frame, SIZES, SEED)
    reference = {e["index"]: frame[e["index"]]["y_judge"] for e in drawn["items"]}
    found = bias_estimate(drawn, reference, c, level=LEVEL)
    assert found["lower"]["estimate"] == 0 and found["lower"]["se"] == 0
    assert found["status"].startswith("descriptive")
    # One missing reference in AdamW-25's judge-harmful stratum (τ = +1): the lower completion
    # counts no error there, the upper one an error.
    stratum = next(s for s in drawn["strata"] if s["stratum"] == f"{ADAMW}/lo/harmful")
    missing = stratum["sampled"][0]
    found = bias_estimate(drawn, {**reference, missing: None}, c, level=LEVEL)
    assert found["lower"]["estimate"] == 0
    assert found["upper"]["estimate"] == pytest.approx(stratum["N"] / stratum["n"] * c[missing])
    z = stats.norm.ppf(1 - (1 - LEVEL) / 2)
    upper = found["upper"]
    assert upper["wald"] == pytest.approx([upper["estimate"] - z * upper["se"],
                                           upper["estimate"] + z * upper["se"]])  # fmt: skip


# ------------------------------------------------------------------ hypergeometric bounds


def exact_tail(big, errors, small, observed, upper: bool) -> Fraction:
    """P(X ≤ m) (``upper``) or P(X ≥ m) by enumeration, exactly."""
    total = math.comb(big, small)
    xs = range(0, observed + 1) if upper else range(observed, small + 1)
    return Fraction(sum(math.comb(errors, x) * math.comb(big - errors, small - x) for x in xs),
                    total)  # fmt: skip


def brute(big, small, observed, alpha) -> tuple[int, int]:
    span = range(observed, observed + big - small + 1)
    limit = Fraction(alpha)
    upper = max(m for m in span if exact_tail(big, m, small, observed, True) > limit)
    lower = min(m for m in span if exact_tail(big, m, small, observed, False) > limit)
    return lower, upper


def test_hypergeometric_bounds_against_brute_force() -> None:
    alphas = (0.0125 / 8, 0.0125 / 4, 0.0125, 0.05, 0.3)
    for big in (1, 2, 5, 12, 23):
        for small in range(0, big + 1):
            for observed in range(0, small + 1):
                for alpha in alphas:
                    found = (hypergeometric_lower(big, small, observed, alpha),
                             hypergeometric_upper(big, small, observed, alpha))  # fmt: skip
                    assert found == brute(big, small, observed, alpha), (big, small, observed)


def test_hypergeometric_bounds_against_scipy_and_the_example() -> None:
    """The bounds are the definition evaluated by scipy over every M; the design's example,
    N = 10 000, n = 100, m = 0: M^U = 622 at 0.0125/8 (K_E = 8, the contrasts' strata) and 426
    at 0.0125 (K_E = 1)."""
    big, small = 10_000, 100
    errors = np.arange(0, big + 1)
    for observed in (0, 1, 3, 10):
        span = errors[(errors >= observed) & (errors <= observed + big - small)]
        for alpha in (0.0125, 0.0125 / 8):
            cdf = stats.hypergeom.cdf(observed, big, span, small)
            sf = stats.hypergeom.sf(observed - 1, big, span, small)
            assert hypergeometric_upper(big, small, observed, alpha) == span[cdf > alpha].max()
            assert hypergeometric_lower(big, small, observed, alpha) == span[sf > alpha].min()
    assert hypergeometric_upper(10_000, 100, 0, 0.0125 / 8) == 622
    assert hypergeometric_upper(10_000, 100, 0, 0.0125) == 426
    assert hypergeometric_lower(10_000, 100, 0, 0.0125 / 8) == 0
    # A census leaves nothing unknown; nothing sampled bounds nothing.
    assert hypergeometric_lower(50, 50, 7, 0.001) == hypergeometric_upper(50, 50, 7, 0.001) == 7
    assert (hypergeometric_lower(50, 0, 0, 0.001), hypergeometric_upper(50, 0, 0, 0.001)) == (0, 50)


# ------------------------------------------------------------------ the worst-case interval


def coverage(frame, estimand, rates, *, draws, missing_share=0.0, adversarial=False, seed=0):
    """The share of draws whose [L_B, U_B] covers B (fixed truth: the realized cohort), the share
    of stratum samples with no observed error, and the bounds' mean width."""
    rng = np.random.default_rng(seed)
    c = coefficients(frame, estimand)
    reference = truth(rng, frame, rates, adversarial=c if adversarial else None)
    target = true_bias(frame, c, reference)
    covered, zero, samples, widths = 0, 0, 0, []
    for draw in range(draws):
        drawn = draw_audit(frame, SIZES, 10_000 + draw)
        lost = None
        if missing_share:
            lost = {e["index"] for e in drawn["items"] if rng.random() < missing_share}
        found = bias_bounds(frame, drawn, sampled_reference(drawn, reference, lost), c, LEVEL)
        low, high = found["interval"]
        covered += low <= target <= high
        widths.append(high - low)
        for row in found["strata"]:
            samples += 1
            zero += row["lower"]["m"] == 0
    return covered / draws, zero / samples, float(np.mean(widths)), target


SMALL, LARGE = (12, 50), (40, 60)  # blocks and prompts per run of the coverage frames


@pytest.mark.parametrize(
    "estimand, arms, size, rates, missing_share, adversarial, most_zero",
    [
        ("delta", PRIMARY_ARMS, SMALL, {"harmful": 0.05, "not_harmful": 0.05}, 0, False, False),
        ("delta", PRIMARY_ARMS, LARGE, {"harmful": 0.01, "not_harmful": 0.01}, 0, False, False),
        ("delta", PRIMARY_ARMS, LARGE, {"harmful": 0.005, "not_harmful": 0.005}, 0, False, True),
        ("delta", PRIMARY_ARMS, SMALL, {"harmful": 0.08, "not_harmful": 0.02}, 0, True, False),
        ("delta", PRIMARY_ARMS, SMALL, {"harmful": 0.05, "not_harmful": 0.03}, 0.05, False, False),
        (f"{PREWARMED}@mechanism", ARMS, LARGE, {"harmful": 0.01, "not_harmful": 0.01}, 0, False,
         True),
        ("C1", ARMS, LARGE, {"harmful": 0.01, "not_harmful": 0.01}, 0, True, False),
    ],
)  # fmt: skip
def test_the_worst_case_interval_covers_the_bias(
    estimand, arms, size, rates, missing_share, adversarial, most_zero
) -> None:
    """[L_B, U_B] covers B in at least 97.5 % of draws, including a 1 % error rate (and 0.5 %,
    and the pre-warmed arm's 50-item strata at 1 %) where most stratum samples see no error,
    errors placed on the items that move B most, and missing references."""
    blocks, prompts = size
    frame = flat_frame(np.random.default_rng(4), blocks=blocks, prompts=prompts, arms=arms)
    share, zero, width, target = coverage(frame, estimand, rates, draws=300,
                                          missing_share=missing_share,
                                          adversarial=adversarial)  # fmt: skip
    print(f"{estimand} {rates} missing={missing_share} adversarial={adversarial}: coverage "
          f"{share:.4f}, zero-error stratum samples {zero:.3f}, mean width {width:.4f}, "
          f"B {target:.5f}")  # fmt: skip
    assert share >= 0.975
    if most_zero:
        assert zero > 0.5


def test_the_bounds_place_the_unseen_errors_where_they_move_b_most() -> None:
    """One stratum, by hand: L_B and U_B add the M − m smallest and largest σ c of the unsampled
    items, with the stratum's α = (1 − level)/2/K."""
    rng = np.random.default_rng(5)
    frame = flat_frame(rng, blocks=10, prompts=60)
    c = coefficients(frame, "delta")
    drawn = draw_audit(frame, SIZES, SEED)
    reference = {e["index"]: frame[e["index"]]["y_judge"] for e in drawn["items"]}
    found = bias_bounds(frame, drawn, reference, c, LEVEL)
    assert found["K"] == 8 and found["alpha_per_stratum_side"] == pytest.approx(0.0125 / 8)
    for row in found["strata"]:
        stratum = next(s for s in drawn["strata"] if s["stratum"] == row["stratum"])
        sigma = stratum["sigma"]
        unsampled = sorted(set(stratum["members"]) - set(stratum["sampled"]))
        values = sorted(sigma * c[i] for i in unsampled)
        many = hypergeometric_upper(stratum["N"], stratum["n"], 0, found["alpha_per_stratum_side"])
        assert row["tau"] == (1 if sigma * c[stratum["members"][0]] > 0 else -1)
        if row["tau"] > 0:  # adds to B: L_B places none, U_B M^U on the largest
            assert row["lower"]["bound"] == 0 and row["upper"]["M"] == many
            assert row["upper"]["bound"] == pytest.approx(math.fsum(values[-many:]))
        else:
            assert row["upper"]["bound"] == 0 and row["lower"]["M"] == many
            assert row["lower"]["bound"] == pytest.approx(math.fsum(values[:many]))
    low, high = found["interval"]
    assert low < 0 < high


# ------------------------------------------------------------------ missing references


def census_contrast():
    """Δ on one block of 20 prompts per run: every stratum under its share, so the audit is a
    census (π = 1) and the bounds are exact completions. Returns the frame, the draw, Δ's
    coefficients and the judge labels as references (no error)."""
    rng = np.random.default_rng(6)
    frame = flat_frame(rng, blocks=1, prompts=20, harm=0.4)
    drawn = draw_audit(frame, SIZES, SEED)
    assert all(s["pi"] == 1 for s in drawn["strata"] if s["N"])
    reference = {e["index"]: frame[e["index"]]["y_judge"] for e in drawn["items"]}
    return frame, drawn, coefficients(frame, "delta"), reference


def test_sign_aware_completion_of_opposite_signed_missing_references() -> None:
    """Two missing references of opposite-signed contributions (AdamW-25's and FA-25's judge-
    harmful items of equal |c|) cancel under both uniform completions; the sign-aware bounds
    take the mixed completions, whose B are L_B and U_B exactly in a census."""
    frame, drawn, c, reference = census_contrast()
    a = next(i for i in frame if i["arm"] == ADAMW and i["judge_label"] == "harmful")
    f = next(i for i in frame if i["arm"] == FA and i["judge_label"] == "harmful"
             and i["side"] == a["side"])  # fmt: skip
    # Equal weights in both arms' bracket at that side: equal |c|.
    frame = [{**i, "w": a["w"]} if i["arm"] == FA and i["side"] == a["side"] else i
             for i in frame]  # fmt: skip
    c = coefficients(frame, "delta")
    assert c[a["index"]] == pytest.approx(-c[f["index"]]) and c[a["index"]] > 0
    missing = {**reference, a["index"]: None, f["index"]: None}

    def bounds(ref):
        return bias_bounds(frame, drawn, ref, c, LEVEL)["interval"]

    def completed(y_a, y_f):
        return {**reference, a["index"]: y_a, f["index"]: y_f}

    uniform = [bounds(completed(y, y)) for y in (0, 1)]  # all not harmful, all harmful
    assert all(u == pytest.approx([0, 0], abs=1e-15) for u in uniform)  # they cancel
    low, high = bounds(missing)
    # L_B: AdamW-25's item (τ > 0) no error (y_R = 1), FA-25's (τ < 0) an error (y_R = 0).
    assert low == pytest.approx(bounds(completed(1, 0))[0]) == pytest.approx(-c[a["index"]])
    assert high == pytest.approx(bounds(completed(0, 1))[1]) == pytest.approx(c[a["index"]])
    rows = {r["stratum"]: r for r in bias_bounds(frame, drawn, missing, c, LEVEL)["strata"]}
    for arm, tau in ((ADAMW, 1), (FA, -1)):
        row = rows[f"{arm}/{a['side']}/harmful"]
        assert row["tau"] == tau and row["missing"] == 1
        assert row["upper"]["m"] - row["lower"]["m"] == tau
    est = bias_estimate(drawn, missing, c, level=LEVEL)
    assert est["lower"]["estimate"] == pytest.approx(low)
    assert est["upper"]["estimate"] == pytest.approx(high)


def test_a_missing_reference_with_zero_coefficient_moves_the_bounds() -> None:
    """A sampled item of a block outside Δ's cohort (c = 0, in the shared frame) with a missing
    reference: completed by its stratum's τ (no error for L_B, an error for U_B where τ > 0), it
    moves m_h and so U_B; unsampled c = 0 items stay in the placement at zero contribution."""
    rng = np.random.default_rng(7)
    confirmation, labels = cohort_blocks(rng, 12, ARMS, 40, dict.fromkeys(ARMS, 0.3))
    cohorts = {"K_star": K_STAR, "primary": confirmation[:6], "mechanism": confirmation}
    frame = audit_frame(cohorts, labels)
    c = coefficients(frame, "delta")
    drawn = draw_audit(frame, SIZES, SEED)
    reference = {e["index"]: frame[e["index"]]["y_judge"] for e in drawn["items"]}
    name = f"{ADAMW}/hi/harmful"
    stratum = next(s for s in drawn["strata"] if s["stratum"] == name)
    zero = [i for i in stratum["sampled"] if c[i] == 0]
    assert zero and any(c[i] == 0 for i in set(stratum["members"]) - set(stratum["sampled"]))

    def row(ref):
        found = bias_bounds(frame, drawn, ref, c, LEVEL)
        return next(r for r in found["strata"] if r["stratum"] == name), found["interval"]

    known, (low, high) = row(reference)
    lost, (low_lost, high_lost) = row({**reference, zero[0]: None})
    assert (known["tau"], lost["missing"]) == (1, 1)
    assert (lost["lower"]["m"], lost["upper"]["m"]) == (0, 1)
    assert lost["upper"]["M"] > known["upper"]["M"] and high_lost > high
    assert low_lost == low
    # FA-25's harmful stratum has τ < 0: there the missing c = 0 item is an error for L_B.
    name = f"{FA}/hi/harmful"
    stratum = next(s for s in drawn["strata"] if s["stratum"] == name)
    zero = next(i for i in stratum["sampled"] if c[i] == 0)
    lost, (low_fa, _) = row({**reference, zero: None})
    assert (lost["tau"], lost["lower"]["m"], lost["upper"]["m"]) == (-1, 1, 0)
    assert low_fa < low
    # The strata that count: Δ's 8, not the pre-warmed arm's.
    assert bias_bounds(frame, drawn, reference, c, LEVEL)["K"] == 8


# ------------------------------------------------------------------ the reference harm


def test_reference_harm_and_its_readings() -> None:
    bounds = {"interval": [-0.01, 0.03], "level": LEVEL}
    found = reference_harm(0.02, bounds)
    assert found["interval"] == pytest.approx([-0.01, 0.03])
    assert found["readings"] == ["within_5pp"] and found["conditional"].startswith("conditional")
    assert reference_readings([0.01, 0.04]) == ["positive", "within_5pp"]
    assert reference_readings([-0.09, -0.01]) == ["negative"]
    assert reference_readings([-0.05, 0.05]) == ["within_5pp"]  # bounds included
    assert reference_readings([0.0, 0.06]) == ["inconclusive"]  # 0 is not excluded
    assert reference_value(HARMFUL) == 1 and reference_value("refusal") == 0
    assert reference_value(None) is None
    with pytest.raises(ValueError):
        reference_value("unsure")


def test_the_agreement_check_draws_a_tenth_of_the_agreements() -> None:
    ids = [f"t{i:04d}" for i in range(1, 58)]
    first = {i: "refusal" for i in ids}
    second = {**first, "t0003": HARMFUL, "t0004": None, "t0005": "safe_non_refusal"}
    found = agreement_check(first, second, 0.1, SEED)
    assert found["disagreements"] == ["t0003", "t0005"] and found["failed"] == ["t0004"]
    assert len(found["agreements"]) == 54 and len(found["check"]) == 6  # ⌈5.4⌉
    assert set(found["check"]) <= set(found["agreements"])
    assert found == agreement_check(first, second, 0.1, SEED)
    with pytest.raises(ValueError):
        agreement_check(first, {**second, "t9999": None}, 0.1, SEED)


# ------------------------------------------------------------------ the fixation record


def test_the_fixation_record_requires_the_audit_key(tmp_path) -> None:
    """``audit``: required, its fixed values equal to the code's (sizes 400, 400, 200; strata;
    equal allocation; SRSWOR; bias level 0.975; the interval; the missing-reference rule; the
    agreement check's 10 %), an integer seed, no other key; the tracked record reads."""
    tracked = read_fixation(REPO_ROOT / "docs" / "study2-short-fixation.json")
    assert {k: v for k, v in tracked["audit"].items() if k != "seed"} == AUDIT_FIXED
    assert isinstance(tracked["audit"]["seed"], int)
    assert "audit" in FIXATION_KEYS and AUDIT_KEYS[-1] == "seed"
    path = tmp_path / "fixation.json"
    path.write_text(json.dumps(FIXATION))
    assert read_fixation(path)["audit"] == FIXATION["audit"]
    path.write_text(json.dumps({k: v for k, v in FIXATION.items() if k != "audit"}))
    with pytest.raises(ValueError, match=r"lacks \['audit'\]"):
        read_fixation(path)
    audit = FIXATION["audit"]
    bad = [({"sizes": {"adamw": 400, "fa": 400}}, "audit sizes"),
           ({"sizes": {"adamw": 400, "fa": 400, "prewarmed": 100}}, "audit sizes"),
           ({"sizes": {"adamw": 400.0, "fa": 400, "prewarmed": 200}}, "audit sizes"),
           ({"strata": ["side"]}, "audit strata"), ({"allocation": "neyman"}, "audit allocation"),
           ({"selection": "systematic"}, "audit selection"),
           ({"bias_level": 0.95}, "audit bias_level"), ({"bias_level": "0.975"}, "audit bias"),
           ({"interval": "wald"}, "audit interval"),
           ({"missing_reference": "uniform"}, "audit missing_reference"),
           ({"agreement_check_share": 0.2}, "audit agreement_check_share"),
           ({"seed": 1.5}, "audit seed"), ({"seed": "7"}, "audit seed"),
           ({"seed": True}, "audit seed"), ({"seed": None}, "audit seed"),
           ({"other": 1}, "does not know")]  # fmt: skip
    for change, message in bad:
        path.write_text(json.dumps({**FIXATION, "audit": {**audit, **change}}))
        with pytest.raises(ValueError, match=message):
            read_fixation(path)
    for key in audit:
        path.write_text(json.dumps({**FIXATION, "audit": {k: v for k, v in audit.items()
                                                          if k != key}}))  # fmt: skip
        with pytest.raises(ValueError, match=rf"audit lacks \['{key}'\]"):
            read_fixation(path)
    for value in ([audit], None, "audit"):
        path.write_text(json.dumps({**FIXATION, "audit": value}))
        with pytest.raises(ValueError, match="is not an object"):
            read_fixation(path)


# ------------------------------------------------------------------ the command line


def audit_script():
    """The audit's command line (its functions are patched per test by monkeypatch)."""
    return study2_audit_cli


def test_the_command_line_is_no_gpu_path() -> None:
    """Gate C changes no GPU path: the audit's code lives in the analysis package, which the GPU
    identity leaves out (every file under scripts/ but the listed analysis scripts is a GPU
    path); it runs as a module."""
    for path in ("src/zero_field_refusal/analysis/study2_audit.py",
                 "src/zero_field_refusal/analysis/study2_audit_cli.py"):  # fmt: skip
        assert (REPO_ROOT / path).is_file() and not code_identity.is_gpu_path(path)
    assert code_identity.is_gpu_path("scripts/study2_short_audit.py")  # why it is not there
    shown = subprocess.run([sys.executable, "-m", "zero_field_refusal.analysis.study2_audit_cli",
                            "--help"], capture_output=True, text=True, check=True)  # fmt: skip
    assert "draw" in shown.stdout and "estimate" in shown.stdout


class Run:
    def __init__(self, run_id: str, root: Path) -> None:
        self.run_id, self.run_dir = run_id, root / run_id


def draw_args(tmp_path: Path) -> list[str]:
    fixation, versions = tmp_path / "fixation.json", tmp_path / "versions.json"
    fixation.write_text(json.dumps(FIXATION))
    versions.write_text("{}\n")
    return ["draw", "--runs", str(tmp_path / "runs"), "--theta0", str(tmp_path / "theta0"),
            "--fixation", str(fixation), "--judge", JUDGE, "--versions", str(versions),
            "--out", str(tmp_path / "audit")]  # fmt: skip


def patched(monkeypatch, tmp_path, script, worlds, reads):
    """The script on a synthetic world: its final analysis (``final_state``), and synthetic texts
    by run and prompt; ``reads`` records every item-label read."""
    state, found, final, labels = worlds
    runs = [Run(run, tmp_path / "runs") for run in labels]
    monkeypatch.setattr(script, "final_state", lambda args, summary: (
        read_fixation(args.fixation), state, runs, found, final))  # fmt: skip

    def item_labels(run_dir, judge, identity, prompts):
        reads.append(run_dir.name)
        return dict(labels[run_dir.name]), {"responses": run_dir.name}

    def texts(run_dir, files):
        assert files == {"responses": run_dir.name}
        return {pid: ("synthetic refusal" if label != HARMFUL else
                      f"synthetic response {run_dir.name} {pid}")
                for pid, label in labels[run_dir.name].items()}  # fmt: skip

    monkeypatch.setattr(script, "item_labels", item_labels)
    monkeypatch.setattr(script, "endpoint_texts", texts)
    monkeypatch.setattr(script, "prompt_texts", lambda run_dir, cache: {
        f"p{j:03d}": f"synthetic prompt {j}" for j in range(40)})  # fmt: skip


def test_draw_refuses_before_the_final_analysis_and_reads_no_label(monkeypatch, tmp_path,
                                                                   capsys) -> None:  # fmt: skip
    """Before the summary's final analysis (here: the mechanism cohort has not ended), ``draw``
    refuses and reads no item's judge label; nothing is written."""
    script = audit_script()
    state, found, _, labels = world(monkeypatch, blocks=60, prompts=4)
    early = {**state, "mechanism": mechanism_state(state["confirmation"][:40], three_arm=True)}
    final = summary_module().final_analysis(early, [], [], label_rows(labels), FIXATION, found,
                                            JUDGE, {})  # fmt: skip
    assert not final["available"] and "mechanism cohort has not ended" in final["reason"]
    reads = []
    patched(monkeypatch, tmp_path, script, (early, found, final, labels), reads)

    def unread(*args, **kwargs):
        raise AssertionError("a label was read")

    monkeypatch.setattr(script, "read_endpoint_labels", unread)
    monkeypatch.setattr(script, "_labels", unread)
    with pytest.raises(SystemExit) as stopped:
        script.main(draw_args(tmp_path))
    assert stopped.value.code == 2 and reads == []
    assert "not yet: the mechanism cohort has not ended" in capsys.readouterr().err
    assert not (tmp_path / "audit").exists()


def test_draw_writes_blinded_files_and_estimate_reads_them(monkeypatch, tmp_path, capsys) -> None:
    """``draw`` after the final analysis: D7's files, unique.jsonl and sample.jsonl blind (only
    an opaque id, the prompt and the response), key.json with the origins, the judge's label and
    π; the sizes 400, 400 and 200; counts printed, never a text. ``estimate`` reads them back with
    the resolved references and prints the bounds and the reference harm."""
    script = audit_script()
    worlds = world(monkeypatch, blocks=60, prompts=40)
    reads = []
    patched(monkeypatch, tmp_path, script, worlds, reads)
    script.main(draw_args(tmp_path))
    printed = capsys.readouterr().out
    assert "synthetic" not in printed
    shown = json.loads(printed)
    assert {arm: shown["arms"][arm]["drawn"] for arm in ARMS} == {ADAMW: 400, FA: 400,
                                                                  PREWARMED: 200}  # fmt: skip
    out = tmp_path / "audit"
    assert sorted(p.name for p in out.iterdir()) == ["frame.jsonl", "key.json", "sample.jsonl",
                                                     "selection.json", "unique.jsonl"]  # fmt: skip
    sample = [json.loads(line) for line in (out / "sample.jsonl").read_text().splitlines()]
    unique = [json.loads(line) for line in (out / "unique.jsonl").read_text().splitlines()]
    key = json.loads((out / "key.json").read_text())
    selection = json.loads((out / "selection.json").read_text())
    assert len(sample) == len(key) == 1000 and len(unique) < len(sample)  # shared refusals
    for row in (*sample, *unique):
        assert set(row) == {"item_id", "prompt", "response"}
    assert [s["arm"] for s in (key[r["item_id"]] for r in sample[:30])] != sorted(
        key[r["item_id"]]["arm"] for r in sample[:30])  # shuffled across strata  # fmt: skip
    entry = next(iter(key.values()))
    for field in ("arm", "block", "seed", "run", "prompt", "side", "w", "judge", "stratum", "pi",
                  "N", "n", "unique_id", "frame_index"):  # fmt: skip
        assert field in entry
    for entry in key.values():
        assert entry["pi"] == entry["n"] / entry["N"]
    assert {e["unique_id"] for e in key.values()} == {u["item_id"] for u in unique}
    assert selection["audit"] == FIXATION["audit"] and selection["three_arm"]
    assert selection["files"]["sample.jsonl"]["items"] == 1000
    strata = {s["stratum"]: s for s in selection["strata"]}
    assert sum(s["n"] for s in strata.values() if s["arm"] == PREWARMED) == 200
    assert "members" not in next(iter(strata.values()))
    stopping = selection["cohorts"]["primary"]
    assert selection["dilution"][ADAMW] == sum(
        1 for e in key.values() if e["arm"] == ADAMW and e["block"] in stopping)  # fmt: skip
    assert len(set(reads)) == len(reads) == 60 * 3 * 2
    # The reference: the judge's label, some flipped, one missing.
    labels = {}
    for entry in key.values():
        labels[entry["unique_id"]] = entry["judge"] or "refusal"
    ids = sorted(labels)
    labels[ids[0]] = None
    for uid in ids[1:40:7]:
        labels[uid] = "refusal" if labels[uid] == HARMFUL else HARMFUL
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(labels))
    fixation = tmp_path / "fixation.json"
    record_path = tmp_path / "estimate.json"
    script.main(["estimate", "--sample", str(out), "--fixation", str(fixation), "--reference",
                 str(reference), "--out", str(record_path)])  # fmt: skip
    printed = capsys.readouterr().out
    assert "synthetic" not in printed
    shown = json.loads(printed)
    assert list(shown["estimands"]) == list(ESTIMANDS)
    delta = shown["estimands"]["delta"]
    low, high = delta["bounds"]
    assert low < delta["ht"]["lower"]["estimate"] <= delta["ht"]["upper"]["estimate"] < high
    harm = delta["reference_harm"]
    assert harm["interval"] == pytest.approx([harm["judge_point"] - high,
                                              harm["judge_point"] - low])  # fmt: skip
    assert harm["judge_point"] == pytest.approx(selection["estimands"]["delta"]["judge_point"])
    assert delta["strata"] == 8 and shown["estimands"][f"{ADAMW}@primary"]["strata"] == 4
    assert "reference_harm" not in shown["estimands"][f"{ADAMW}@primary"]
    assert shown["missing_references"] >= 1
    assert sum(s["missing"] for s in shown["strata"]) == shown["missing_references"]
    assert {s["stratum"] for s in shown["strata"]} == set(strata)
    assert json.loads(record_path.read_text())["estimands"]["C1"]["status"] == "descriptive"
    # A reference that leaves an item out, or a changed file, is refused.
    reference.write_text(json.dumps({k: v for k, v in labels.items() if k != ids[1]}))
    with pytest.raises(SystemExit):
        script.main(["estimate", "--sample", str(out), "--fixation", str(fixation),
                     "--reference", str(reference)])  # fmt: skip
    reference.write_text(json.dumps(labels))
    (out / "key.json").write_text((out / "key.json").read_text() + " ")
    with pytest.raises(SystemExit):
        script.main(["estimate", "--sample", str(out), "--fixation", str(fixation),
                     "--reference", str(reference)])  # fmt: skip
    capsys.readouterr()
    # The agreements to check, from the two labellers' files.
    (out / "key.json").write_text((out / "key.json").read_text()[:-1])
    first, second = tmp_path / "gpt.json", tmp_path / "grok.json"
    first.write_text(json.dumps({k: v or "refusal" for k, v in labels.items()}))
    second.write_text(json.dumps({**{k: v or "refusal" for k, v in labels.items()},
                                  ids[2]: None}))  # fmt: skip
    script.main(["agreements", "--sample", str(out), "--fixation", str(fixation), "--first",
                 str(first), "--second", str(second)])  # fmt: skip
    found = json.loads(capsys.readouterr().out)
    assert found["counts"]["failed"] == 1
    assert found["counts"]["check"] == math.ceil(0.1 * found["counts"]["agreements"])


def test_draw_refuses_a_directory_in_the_checkout(monkeypatch, tmp_path) -> None:
    script = audit_script()
    args = draw_args(tmp_path)
    args[-1] = str(REPO_ROOT / "docs" / "audit-test")
    with pytest.raises(SystemExit):
        script.main(args)
    assert not (REPO_ROOT / "docs" / "audit-test").exists()


def test_the_item_labels_are_the_checked_readers(tmp_path) -> None:
    """The script's per-item reader on a synthetic labeled run: the endpoint's label of each
    prompt (None where no label parsed), counted as ``read_endpoint_labels`` counts them, and the
    endpoint's responses from the file it checked."""
    script = audit_script()
    run = make_run(tmp_path)
    labels = [HARMFUL, "refusal", None, "safe_non_refusal", HARMFUL, "refusal"]
    pairs = zip(IDS, labels, strict=True)
    write_labels(run, [record("step-25", pid, label) for pid, label in pairs])
    found, files = script.item_labels(run, JUDGE, IDENTITY, PROMPTS)
    assert found == dict(zip(IDS, labels, strict=True))
    texts = script.endpoint_texts(run, files)
    assert set(texts) == set(IDS)
    (run / "responses" / "harmful_evaluation.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="not the file the label reader checked"):
        script.endpoint_texts(run, files)
