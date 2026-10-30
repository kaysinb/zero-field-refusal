"""Study 3's judge-based estimates (analysis/study3_estimates.py), as pure functions of frozen
blocks and harm counts: the block differences of the three computations, their intervals,
readings and wording, the worst cases, matched K_harmful, the curvature sensitivity and the spread
of K. Every block, harm count and endpoint here is synthetic; no file is read."""

import json
import math
import re
import statistics
from types import SimpleNamespace as H

import numpy as np
import pytest

from zero_field_refusal.analysis import study3_estimates as est
from zero_field_refusal.analysis import study3_inference as inf
from zero_field_refusal.analysis.study3_blocks import CONFIRMATION_ALPHA

UNIT = "pp per (ln K)^2"


def blocks(n, w=0.5, width=0.446):
    return [
        {
            "seed": 10400 + i,
            "arms": {
                a: {"lo": f"{a}{i}l", "hi": f"{a}{i}h", "w": w, "width": width} for a in "AFP"
            },
        }
        for i in range(n)
    ]


def test_block_differences_follow_the_registered_sensitivities() -> None:
    harm = {
        "A0l": H(n=521, harmful=52, missing=0),
        "A0h": H(n=521, harmful=104, missing=10),
        "F0l": H(n=521, harmful=26, missing=5),
        "F0h": H(n=521, harmful=78, missing=0),
    }
    d = est.block_differences(blocks(1), harm, ("A", "F"))
    a_low, a_high = (52 + 104) / 2 / 521, (52 + 114) / 2 / 521
    f_low, f_high = (26 + 78) / 2 / 521, (31 + 78) / 2 / 521
    assert d["primary"][0] == pytest.approx(a_low - f_low)
    assert d["missing_low"][0] == pytest.approx(a_low - f_high)
    assert d["missing_high"][0] == pytest.approx(a_high - f_low)
    assert d["all_missing_harmful"][0] == pytest.approx(a_high - f_high)
    with pytest.raises(ValueError, match="521"):
        est.block_differences(
            blocks(1), {**harm, "A0l": H(n=520, harmful=1, missing=0)}, ("A", "F")
        )


@pytest.mark.parametrize(
    "bad",
    [
        H(n=521, harmful=-1, missing=0),
        H(n=521, harmful=1, missing=-2),
        H(n=521, harmful=500, missing=22),
    ],
)
def test_block_differences_refuse_impossible_counts(bad) -> None:
    harm = {f"{a}0{e}": H(n=521, harmful=10, missing=0) for a in "AF" for e in "lh"}
    with pytest.raises(ValueError):
        est.block_differences(blocks(1), {**harm, "F0h": bad}, ("A", "F"))


def test_the_three_computations_share_one_stream() -> None:
    rng = np.random.default_rng(5)
    d = {k: rng.normal(0, 0.02, 40) for k in est.COMPUTATIONS}
    seed = inf.stream(7, "analysis", "A-F")
    got = est.three_computations(d, level=0.975, resamples=2000, seed=seed)
    for key in est.COMPUTATIONS:
        assert got[key] == inf.bootstrap_t(d[key], level=0.975, resamples=2000, seed=seed)
    env = got["missing_labels"]
    assert isinstance(env, est.SensitivityEnvelope) and not isinstance(env, inf.Interval)
    assert (env.low, env.high) == (got["missing_low"], got["missing_high"])
    assert (env.lower, env.upper) == (env.low.lower, env.high.upper)
    assert got["reading"] == est.three_way_reading(got["primary"], env, got["all_missing_harmful"])


def test_a_constant_d_low_with_a_varying_d_high_is_an_undetermined_envelope() -> None:
    d = {
        "primary": np.linspace(-0.02, 0.01, 30),
        "missing_low": np.full(30, -0.01),
        "missing_high": np.linspace(-0.01, 0.03, 30),
        "all_missing_harmful": np.zeros(30) + 0.01 + np.linspace(0, 1e-3, 30),
    }
    got = est.three_computations(
        d, level=0.975, resamples=2000, seed=inf.stream(7, "analysis", "A-F")
    )  # no exception
    env = got["missing_labels"]
    assert env.low.undetermined and not env.high.undetermined and env.undetermined
    assert math.isnan(env.lower) and math.isnan(env.upper)
    assert env.high.infinite_low == got["missing_high"].infinite_low  # counts stay with sources
    assert got["reading"].direction is None and got["reading"].margin is None
    assert json.loads(json.dumps(env.to_json(), allow_nan=False))["low"] == env.low.to_json()


def test_an_envelope_keeps_its_sources_and_reads_back_from_its_json() -> None:
    low = iv(-3, -1)
    high = inf.Interval(
        n=60,
        mean=0.0,
        sd=0.03,
        se=0.004,
        level=0.975,
        resamples=100_000,
        lower=-0.005,
        upper=math.inf,
        infinite_low=2,
        infinite_high=0,
        zero_se=2,
        undetermined=False,
    )
    env = est.envelope(low, high)
    assert (env.lower, env.upper, env.undetermined) == (-0.03, math.inf, False)
    data = json.loads(json.dumps(env.to_json(), allow_nan=False))
    assert data["upper"] == "+inf" and data["high"] == high.to_json()
    assert est.SensitivityEnvelope.from_json(data) == env
    with pytest.raises(ValueError):
        est.SensitivityEnvelope.from_json({**data, "lower": -0.04})
    with pytest.raises(ValueError):
        est.SensitivityEnvelope(low, high, -0.04, math.inf, False)
    assert est.envelope_reading(env) == inf.Reading(None, None, None, True)


@pytest.mark.parametrize("seed", range(20))
def test_the_three_way_rule_is_stream_0s_on_plain_intervals(seed) -> None:
    rng = np.random.default_rng(seed)
    a, b, c = (iv(*sorted(rng.uniform(-7, 7, 2))) for _ in range(3))
    assert est.three_way_reading(a, est.envelope(b, b), c) == inf.combined_reading([a, b, c])


def test_the_three_way_rule_combines_only_one_level_resamples_and_n() -> None:
    other = inf.Interval(**{**iv(-2, -1).__dict__, "n": 59})
    with pytest.raises(ValueError, match="share"):
        est.three_way_reading(iv(-2, -1), est.envelope(other, other), iv(-2, -1))


def iv(lo_pp, hi_pp, undetermined=False):
    return inf.Interval(
        n=60,
        mean=(lo_pp + hi_pp) / 200,
        sd=0.03,
        se=0.004,
        level=0.975,
        resamples=100_000,
        lower=lo_pp / 100,
        upper=hi_pp / 100,
        infinite_low=0,
        infinite_high=0,
        zero_se=0,
        undetermined=undetermined,
    )


@pytest.mark.parametrize(
    "three,row,direction,low,high",
    [
        ([(-2, -1), (-2.2, -0.5), (-2.4, -0.8)], "direction_below_2_5", "less", 0.005, 0.024),
        ([(-4, -1), (-3, -0.5), (-2, -0.8)], "direction_below_5", "less", 0.005, 0.04),
        ([(-7, -1.5), (-6, -1), (-8, -2)], "direction_beyond_5", "less", 0.01, None),
        ([(1, 2), (0.5, 2.4), (0.8, 2.2)], "direction_below_2_5", "more", 0.005, 0.024),
        ([(-2, 1), (-1, 2), (-2.4, 0.2)], "below_2_5_direction_undetermined", None, None, None),
        ([(-4, 1), (-1, 2), (-2, 0.2)], "below_5_direction_undetermined", None, None, None),
        ([(-6, 1), (-1, 2), (-2, 0.2)], "undetermined", None, None, None),
    ],
)
def test_the_wording_follows_the_registered_table(three, row, direction, low, high) -> None:
    intervals = [iv(*pair) for pair in three]
    got = est.wording(inf.combined_reading(intervals), intervals)
    assert (got["row"], got["direction"]) == (row, direction)
    assert got["low"] == (pytest.approx(low) if low else None)
    assert got["high"] == (pytest.approx(high) if high else None)
    assert got["row"] in est.WORDING_ROWS


def test_an_undetermined_computation_makes_the_wording_undetermined() -> None:
    d = {
        "primary": np.linspace(-0.03, -0.01, 30),
        "missing_low": np.linspace(-0.035, -0.012, 30),
        "missing_high": np.linspace(-0.028, -0.008, 30),
        "all_missing_harmful": np.full(30, -0.02),
    }  # s = 0
    got = est.three_computations(
        d, level=0.975, resamples=2000, seed=inf.stream(7, "analysis", "A-F")
    )
    assert est.reading(got["primary"]).direction == "negative"
    members = [got["primary"], got["missing_labels"], got["all_missing_harmful"]]
    worded = est.wording(got["reading"], members)
    assert worded == {
        "row": "undetermined",
        "direction": None,
        "low": None,
        "high": None,
        "magnitude_lower": None,
        "margin": None,
    }


def test_the_worst_case_takes_unknown_blocks_at_minus_and_plus_one() -> None:
    got = est.worst_case_unknown(
        np.array([0.01, 0.03]), 2, level=0.975, resamples=500, seed=inf.stream(7, "worst", "A-F")
    )
    assert (got["low_mean"], got["high_mean"]) == (pytest.approx(-0.49), pytest.approx(0.51))
    assert got["envelope"].low.n == got["envelope"].high.n == 4


# ------------------------------------------------------------------ matched K_harmful


def test_the_worst_case_counts_every_started_block_outside_the_cohort() -> None:
    # 70 started blocks, A-F supported on all, N = 60; P-F reaches only 30: the blocks beyond N
    # are unknown like the unsupported ones
    freeze = frozen(60, unknown=tuple(range(10460, 10470)))
    freeze["comparisons"]["P-F"]["blocks"] = freeze["comparisons"]["P-F"]["blocks"][:30]
    freeze["comparisons"]["P-F"]["unknown"] = list(range(10430, 10470))
    freeze["comparisons"]["A-F"]["unknown"] = list(range(10460, 10470))
    result = analyzed(freeze, harm_of(60))["comparisons"]
    assert result["A-F"]["worst_case_unknown_blocks"]["unknown"] == 10
    assert result["P-F"]["worst_case_unknown_blocks"]["unknown"] == 40
    got = est.worst_case_unknown(
        np.zeros(60), 10, level=0.975, resamples=999, seed=np.random.SeedSequence(1)
    )
    assert got["high_mean"] == pytest.approx(10 / 70) and got["low_mean"] == pytest.approx(-10 / 70)
    assert round(100 * got["high_mean"], 2) == 14.29


def harmful_ks(n, a=(4.8, 7.5), f=(1.2 * 4.8, 1.2 * 7.5)) -> dict:
    """Each block's endpoints by run id, with A's K_harmful ``a`` and F's ``f`` at (lo, hi)."""
    out = {}
    for i in range(n):
        for arm, (lo, hi) in (("A", a), ("F", f)):
            out[f"{arm}{i}l"], out[f"{arm}{i}h"] = H(K_harmful=lo), H(K_harmful=hi)
    return out


def test_kstar_h_is_the_geometric_mean_of_k_harmful_at_kstar_over_a_and_f() -> None:
    by_id = harmful_ks(2)
    assert est.kstar_h(blocks(2), by_id) == (pytest.approx(math.sqrt(6.0 * 7.2)), [])
    for value in (None, 0.0, -1.0, math.inf, math.nan):
        by_id["A1l"] = H(K_harmful=value)
        kstar, left_out = est.kstar_h(blocks(2), by_id)
        assert kstar == pytest.approx(math.sqrt(6.0 * 7.2))
        assert left_out == [{"seed": 10401, "reason": "no k_harmful"}]
    by_id["A0h"] = H(K_harmful=None)
    assert est.kstar_h(blocks(2), by_id) == (
        None,
        [{"seed": 10400, "reason": "no k_harmful"}, {"seed": 10401, "reason": "no k_harmful"}],
    )


def test_matched_blocks_order_by_k_harmful_and_count_every_left_out_block() -> None:
    by_id = harmful_ks(4)
    kstar = math.sqrt(6.0 * 7.2)
    by_id["A0l"], by_id["A0h"] = H(K_harmful=7.5), H(K_harmful=4.8)  # reversed against K
    by_id["F1l"] = by_id["F1h"] = H(K_harmful=7.0)  # tied
    by_id["F2l"], by_id["F2h"] = H(K_harmful=8.0), H(K_harmful=9.0)  # above K*_h
    by_id["A3l"] = H(K_harmful=None)
    matched, left_out = est.matched_blocks(blocks(4), by_id, kstar)
    assert [b["seed"] for b in matched] == [10400]
    a = matched[0]["arms"]["A"]
    assert (a["lo"], a["hi"]) == ("A0h", "A0l")
    assert a["w"] == pytest.approx((math.log(kstar) - math.log(4.8)) / math.log(7.5 / 4.8))
    assert left_out == [
        {"seed": 10401, "arms": {"F": "tied"}},
        {"seed": 10402, "arms": {"F": "no bracket"}},
        {"seed": 10403, "reason": "no k_harmful"},
    ]


# ------------------------------------------------------------------ descriptives


def test_curvature_bias_follows_the_realized_widths() -> None:
    found = blocks(2)
    found[1]["arms"]["F"] |= {"w": 0.25, "width": 0.6}
    got = est.curvature_bias(found, ("A", "F"), {"A": 2.4, "F": 1.2, "unit": UNIT})
    a = 2.4 / 2 * 0.25 * 0.446**2
    per_block = [a - 1.2 / 2 * 0.25 * 0.446**2, a - 1.2 / 2 * 0.25 * 0.75 * 0.6**2]
    assert got["bias_pp"] == pytest.approx(statistics.fmean(per_block))
    assert got["largest_pp"] == pytest.approx(max(abs(x) for x in per_block))
    assert got["widths"]["F"] == pytest.approx({"min": 0.446, "median": 0.523, "max": 0.6})
    assert got["widths"]["A"] == pytest.approx({"min": 0.446, "median": 0.446, "max": 0.446})
    with pytest.raises(ValueError, match="unit"):
        est.curvature_bias(found, ("A", "F"), {"A": 2.4, "F": 1.2, "unit": "pp per K"})


def test_the_spread_and_the_observed_extremes() -> None:
    assert est.spread_quantiles([1.0, 2.0, 3.0, 4.0, 5.0]) == pytest.approx(
        [1.0, 1.4, 2.0, 3.0, 4.0, 4.6, 5.0]
    )
    with pytest.raises(ValueError):
        est.spread_quantiles([])
    got = est.observed_extremes(blocks(3), np.array([0.01, -0.02, 0.03]))
    assert got == {"smallest": {"seed": 10401, "d": -0.02}, "largest": {"seed": 10402, "d": 0.03}}
    assert est.observed_extremes([], np.array([])) == {"smallest": None, "largest": None}


# ------------------------------------------------------------------ the analysis


def frozen(n=30, *, pf="descriptive", unknown=(10500, 10501), a_minus_p=None, matched=None):
    """A freeze record's label-dependent part: A–F and P–F on ``n`` blocks, A − P and the matched
    blocks on the same seeds unless given."""
    found = blocks(n)

    def of(arms, rows=found):
        return [{"seed": b["seed"], "arms": {a: dict(b["arms"][a]) for a in arms}} for b in rows]

    return {
        "comparisons": {
            "A-F": {
                "reading": "confirmatory",
                "rule": "n_reached",
                "blocks": of("AF"),
                "unknown": [],
            },
            "P-F": {"reading": pf, "rule": "cap", "blocks": of("PF"), "unknown": list(unknown)},
        },
        "a_minus_p": {"blocks": of("AP") if a_minus_p is None else a_minus_p},
        "k_harmful": {
            "kstar_h": 6.5,
            "left_out": [],
            "blocks": [
                {
                    "seed": b["seed"],
                    "arms": {a: {k: b["arms"][a][k] for k in ("lo", "hi", "w")} for a in "AF"},
                }
                for b in found
            ]
            if matched is None
            else matched,
        },
    }


def harm_of(n=30, seed=3, *, equal=False) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        for arm in "AFP":
            for end in "lh":
                harmful = 60 if equal else int(rng.integers(20, 200))
                missing = 0 if equal else int(rng.integers(0, 3))
                out[f"{arm}{i}{end}"] = H(n=521, harmful=harmful, missing=missing)
    return out


def gate(name, passes=True, computed=True) -> dict:
    bound = {"n": 30, "mean": 1.0, "upper": 1.5, "threshold": 5.0, "passes": passes}
    return {
        "comparison": name,
        "alpha": CONFIRMATION_ALPHA,
        "n": 30,
        "excluded": (),
        "waiting": (),
        "computed": computed,
        "passes": passes,
        "bounds": {arm: {"mmlu": bound} for arm in (name[0], "F")},
    }


def analyzed(freeze=None, harm=None, **kw):
    freeze = frozen() if freeze is None else freeze
    harm = harm_of() if harm is None else harm
    args = {
        "level": 0.975,
        "seed": 20261101,
        "resamples": 999,
        "gates": {"A-F": gate("AF"), "P-F": gate("PF")},
        "curvature": {"A": 9.537, "F": 11.124, "P": 11.124, "unit": UNIT, "source": "s"},
        "spread": {"A0l": [4.0, 5.0, 6.5], "F0h": [7.0, 7.5, 8.0]},
        **kw,
    }
    return est.analysis(freeze, harm, **args)


ID = re.compile(r"^[A-Za-z0-9_.:/@+-]+$")


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from strings(item)


def test_the_analysis_is_numbers_and_ids_only() -> None:
    result = analyzed()
    text = json.dumps(result, allow_nan=False)
    assert all(ID.fullmatch(s) for s in strings(json.loads(text))), [
        s for s in strings(json.loads(text)) if not ID.fullmatch(s)
    ]
    found = result["comparisons"]
    assert set(found) == {"A-F", "P-F", "A-P", "A-F@K_harmful"}
    assert [found[k]["status"] for k in ("A-F", "P-F", "A-P", "A-F@K_harmful")] == [
        "confirmatory",
        "descriptive",
        "descriptive",
        "descriptive",
    ]
    assert found["A-F"]["n"] == 30 and found["A-F"]["wording"]["row"] in est.WORDING_ROWS
    seed = inf.stream(20261101, "analysis", "A-F")
    d = est.block_differences(frozen()["comparisons"]["A-F"]["blocks"], harm_of(), ("A", "F"))
    for key in est.COMPUTATIONS:
        assert inf.Interval.from_json(found["A-F"][key]) == inf.bootstrap_t(
            d[key], level=0.975, resamples=999, seed=seed
        )
    missing = {
        r
        for b in frozen()["comparisons"]["A-F"]["blocks"]
        for a in "AF"
        for r in (b["arms"][a]["lo"], b["arms"][a]["hi"])
        if harm_of()[r].missing
    }
    assert found["A-F"]["missing"] == {
        "runs": len(missing),
        "labels": sum(harm_of()[r].missing for r in missing),
    }
    assert result["utility"]["claim"] is True and set(result["utility"]["gates"]) == {"A-F", "P-F"}
    assert set(result["curvature"]) == {"A-F", "P-F"}
    assert result["k_spread"]["A0l"] == est.spread_quantiles([4.0, 5.0, 6.5])
    equal = analyzed(harm=harm_of(equal=True))["comparisons"]["A-F"]
    assert equal["reading"]["undetermined"] and equal["wording"]["row"] == "undetermined"


def test_the_analysis_serializes_every_interval_through_to_json(monkeypatch) -> None:
    def fake(d, *, level, resamples, seed):
        d = np.asarray(d, dtype=float)
        return inf.Interval(
            n=int(d.size),
            mean=float(d.mean()),
            sd=0.03,
            se=0.004,
            level=level,
            resamples=resamples,
            lower=-math.inf,
            upper=0.01,
            infinite_low=0,
            infinite_high=3,
            zero_se=3,
            undetermined=False,
        )

    monkeypatch.setattr(est, "bootstrap_t", fake)
    result = analyzed()
    json.dumps(result, allow_nan=False)
    entry = result["comparisons"]["A-F"]
    d = est.block_differences(frozen()["comparisons"]["A-F"]["blocks"], harm_of(), ("A", "F"))
    expected = fake(d["primary"], level=0.975, resamples=999, seed=None)
    assert entry["primary"] == expected.to_json() and entry["primary"]["lower"] == "-inf"
    assert inf.Interval.from_json(entry["primary"]) == expected
    assert entry["missing_labels"]["low"]["lower"] == "-inf"
    assert est.SensitivityEnvelope.from_json(entry["missing_labels"]).lower == -math.inf


def test_the_analysis_labels_the_worst_cases_apart() -> None:
    result = analyzed()["comparisons"]
    pf, af = result["P-F"], result["A-F"]
    assert pf["worst_case_unknown_blocks"]["unknown"] == 2
    assert pf["worst_case_unknown_blocks"]["low_mean"] < pf["primary"]["mean"]
    assert pf["worst_case_unknown_blocks"]["high_mean"] > pf["primary"]["mean"]
    assert set(pf["worst_case_unknown_blocks"]["envelope"]) == {
        "low",
        "high",
        "lower",
        "upper",
        "undetermined",
    }
    worst = af["worst_case_unknown_blocks"]
    assert worst["unknown"] == 0
    assert worst["low_mean"] == pytest.approx(af["primary"]["mean"])
    assert worst["high_mean"] == pytest.approx(af["primary"]["mean"])
    d = est.block_differences(frozen()["comparisons"]["A-F"]["blocks"], harm_of(), ("A", "F"))
    assert af["observed_extremes"] == est.observed_extremes(
        frozen()["comparisons"]["A-F"]["blocks"], d["primary"]
    )
    assert "worst_case_unknown_blocks" not in result["A-P"]
    assert "observed_extremes" not in result["A-F@K_harmful"]


def test_an_empty_a_minus_p_is_reported_undetermined() -> None:
    entry = analyzed(frozen(a_minus_p=[]))["comparisons"]["A-P"]
    assert entry["n"] == 0 and entry["status"] == "descriptive"
    assert all(entry[key] is None for key in (*est.COMPUTATIONS, "missing_labels"))
    assert entry["reading"]["undetermined"] and all(
        r["undetermined"] for r in entry["readings"].values()
    )
    assert entry["wording"]["row"] == "undetermined"
    assert entry["missing"] == {"runs": 0, "labels": 0}


def test_the_utility_claim_needs_every_gate_computed_and_passed() -> None:
    assert (
        analyzed(gates={"A-F": gate("AF"), "P-F": gate("PF", passes=False)})["utility"]["claim"]
        is False
    )
    short = {**gate("PF", passes=False, computed=False), "bounds": {}}
    assert analyzed(gates={"A-F": gate("AF"), "P-F": short})["utility"]["claim"] is False
    with pytest.raises(ValueError, match="gate"):
        analyzed(gates={"A-F": gate("AF")})
    with pytest.raises(ValueError, match="alpha"):
        analyzed(gates={"A-F": {**gate("AF"), "alpha": 0.05}, "P-F": gate("PF")})
