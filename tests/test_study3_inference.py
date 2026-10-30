import json
import math
import subprocess
import sys
from fractions import Fraction

import numpy as np
import pytest

from zero_field_refusal.analysis import study3_inference as inf

SEED = inf.stream(20261031, "test")


def ranks(level, m):
    """Type 1 by hand: the zero-based rank ceil(M·p) − 1, with exact arithmetic."""
    lvl = Fraction(str(level))
    return (math.ceil(m * (1 - lvl) / 2) - 1, math.ceil(m * (1 + lvl) / 2) - 1)


@pytest.mark.parametrize(
    "level,m,expected",
    [
        (0.975, 2000, (24, 1974)),
        (0.975, 100_000, (1249, 98749)),
        (0.98, 2000, (19, 1979)),
        (0.98, 100_000, (999, 98999)),
        (0.985, 2000, (14, 1984)),
        (0.985, 100_000, (749, 99249)),
        (0.99, 2000, (9, 1989)),
        (0.99, 100_000, (499, 99499)),
    ],
)
def test_tail_ranks_are_exact(level, m, expected):
    assert inf.tail_ranks(level, m) == expected == ranks(level, m)


def reference(d, level, resamples, seed):
    """The registered rule written out per resample (Inference: t* = (d̄* − d̄)/(s*/√n); s* = 0
    gives 0 when d̄* = d̄ exactly, else ±∞), then the order statistics at the exact ranks."""
    d = np.asarray(d, float)
    n = d.size
    mean = d.mean()
    se = d.std(ddof=1) / math.sqrt(n)
    idx = np.random.default_rng(seed).integers(0, n, size=(resamples, n))
    t = []
    for row in d[idx]:
        if row.max() == row.min():
            t.append(0.0 if row[0] == mean else math.copysign(math.inf, row[0] - mean))
        else:
            t.append((row.mean() - mean) / (row.std(ddof=1) / math.sqrt(n)))
    t = np.sort(np.array(t))
    k_lo, k_hi = ranks(level, resamples)
    return mean - t[k_hi] * se, mean - t[k_lo] * se


@pytest.mark.parametrize("level", [0.975, 0.98, 0.985, 0.99])
def test_bootstrap_t_matches_the_rule_written_out(level):
    d = np.random.default_rng(1).normal(0.01, 0.04, 9)
    got = inf.bootstrap_t(d, level=level, resamples=2000, seed=SEED, chunk=700)
    assert (got.lower, got.upper) == pytest.approx(reference(d, level, 2000, SEED), abs=1e-13)
    assert got.n == 9 and got.resamples == 2000 and not got.undetermined


def test_the_result_does_not_depend_on_the_chunk():
    d = np.random.default_rng(2).normal(0, 0.03, 12)
    a = inf.bootstrap_t(d, level=0.98, resamples=5000, seed=SEED, chunk=5000)
    b = inf.bootstrap_t(d, level=0.98, resamples=5000, seed=SEED, chunk=333)
    assert a == b
    d9 = np.random.default_rng(4).normal(0, 0.03, 9)
    whole = inf.bootstrap_t(d9, level=0.98, resamples=5000, seed=SEED, chunk=5000)
    assert whole == inf.bootstrap_t(d9, level=0.98, resamples=5000, seed=SEED, chunk=333)
    small = inf.bootstrap_t(d9, level=0.9, resamples=60, seed=SEED, chunk=60)
    assert small == inf.bootstrap_t(d9, level=0.9, resamples=60, seed=SEED, chunk=1)


def test_an_observed_zero_sd_records_exact_zeros():
    got = inf.bootstrap_t([0.1] * 3, level=0.975, resamples=100, seed=SEED)
    assert got.sd == 0.0 and got.se == 0.0 and got.undetermined


@pytest.mark.parametrize("seed", [np.random.default_rng(1), 5])
def test_a_seed_that_is_not_a_seed_sequence_is_refused(seed):
    with pytest.raises(ValueError):
        inf.bootstrap_t([0.0, 0.01, 0.02], level=0.975, resamples=100, seed=seed)


def test_two_blocks_give_infinite_tails():
    # resamples (0,0) → −∞, (1,1) → +∞, mixed → 0: each infinite tail holds about a quarter
    got = inf.bootstrap_t([0.0, 0.01], level=0.975, resamples=4000, seed=SEED)
    assert got.lower == -math.inf and got.upper == math.inf
    assert got.zero_se == got.infinite_low + got.infinite_high
    assert 800 < got.infinite_low < 1200 and 800 < got.infinite_high < 1200


def test_a_zero_se_resample_exactly_at_the_mean_gives_zero():
    # d̄ = 0.5 exactly: all-0.5 resamples give 0; all-0 give −∞ and all-1 give +∞
    got = inf.bootstrap_t([0.0, 0.5, 1.0], level=0.975, resamples=9000, seed=SEED)
    at_mean = got.zero_se - got.infinite_low - got.infinite_high
    assert 250 < at_mean < 420 and 250 < got.infinite_low < 420 and 250 < got.infinite_high < 420


def test_no_tolerance_near_the_mean():
    # 35 differences of 0.01 and one of 0.01 + 1e-11: d̄ is above 0.01, so an all-0.01 resample
    # (probability (35/36)^36 ≈ 0.363) gives −∞, not 0 (the registered rule has no tolerance)
    d = [0.01] * 35 + [0.01 + 1e-11]
    got = inf.bootstrap_t(d, level=0.975, resamples=10_000, seed=SEED)
    assert got.zero_se == got.infinite_low + got.infinite_high
    assert 3300 < got.infinite_low < 3950


def test_an_observed_zero_sd_is_undetermined():
    got = inf.bootstrap_t([0.02] * 5, level=0.975, resamples=1000, seed=SEED)
    assert got.undetermined and math.isnan(got.lower) and math.isnan(got.upper)
    assert inf.reading(got) == inf.Reading(None, None, None, True)


def test_one_block_is_undetermined():
    assert inf.bootstrap_t([0.01], level=0.975, resamples=10, seed=SEED).undetermined


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d": []},
        {"d": [[0.1, 0.2], [0.3, 0.4]]},
        {"d": [0.1, math.nan]},
        {"d": [0.1, math.inf]},
        {"level": 0.0},
        {"level": 1.0},
        {"level": -0.5},
        {"resamples": 0},
        {"resamples": 2.5},
        {"resamples": True},
        {"chunk": 0},
        {"chunk": False},
    ],
)
def test_malformed_inputs_are_refused(kwargs):
    args = {"d": [0.0, 0.01, 0.02], "level": 0.975, "resamples": 100, "chunk": 10} | kwargs
    with pytest.raises(ValueError):
        inf.bootstrap_t(args.pop("d"), seed=SEED, **args)


def test_the_same_seed_shifts_with_the_data():
    d = np.random.default_rng(3).normal(0, 0.03, 15)
    a = inf.bootstrap_t(d, level=0.975, resamples=4000, seed=SEED)
    b = inf.bootstrap_t(d + 0.007, level=0.975, resamples=4000, seed=SEED)
    assert b.lower - a.lower == pytest.approx(0.007, abs=1e-12)
    assert b.upper - a.upper == pytest.approx(0.007, abs=1e-12)


def at(lower_pp, upper_pp):
    finite = [x for x in (lower_pp, upper_pp) if math.isfinite(x)]
    return inf.Interval(
        n=60,
        mean=(sum(finite) / len(finite) if finite else 0.0) / 100,
        sd=0.03,
        se=0.004,
        level=0.975,
        resamples=100_000,
        lower=lower_pp / 100,
        upper=upper_pp / 100,
        infinite_low=int(upper_pp == math.inf),
        infinite_high=int(lower_pp == -math.inf),
        zero_se=int(upper_pp == math.inf) + int(lower_pp == -math.inf),
        undetermined=False,
    )


UNDETERMINED = inf.Interval(
    n=60,
    mean=0.02,
    sd=0.0,
    se=0.0,
    level=0.975,
    resamples=100_000,
    lower=math.nan,
    upper=math.nan,
    infinite_low=0,
    infinite_high=0,
    zero_se=0,
    undetermined=True,
)


@pytest.mark.parametrize(
    "lo,hi,direction,margin,magnitude",
    [
        (-2.0, -1.0, "negative", 0.025, 0.01),
        (-4.0, -1.0, "negative", 0.05, 0.01),
        (-7.0, -1.5, "negative", None, 0.015),
        (1.0, 2.5, "positive", 0.025, 0.01),
        (-2.0, 1.0, None, 0.025, None),
        (-4.0, 3.0, None, 0.05, None),
        (-6.0, 1.0, None, None, None),
        (-2.5, 2.5, None, 0.025, None),
        (-5.0, 0.0, None, 0.05, None),
        (-math.inf, -1.0, "negative", None, 0.01),
        (1.0, math.inf, "positive", None, 0.01),
        (-math.inf, 1.0, None, None, None),
    ],
)
def test_readings_follow_the_registered_table(lo, hi, direction, margin, magnitude):
    r = inf.reading(at(lo, hi))
    assert (r.direction, r.margin) == (direction, margin)
    assert r.magnitude_lower == (pytest.approx(magnitude) if magnitude else None)
    assert r.undetermined == (direction is None and margin is None)


def test_combined_reading_reports_direction_and_margin_independently():
    r = inf.combined_reading([at(-2, -1), at(-3, -0.5), at(-2.4, 0.2)])
    assert (r.direction, r.margin, r.undetermined) == (None, 0.05, False)
    # inside ±2.5 and containing 0, all three: a margin without a direction, not undetermined
    r = inf.combined_reading([at(-1, 1), at(-2, 0.5), at(-0.5, 2.4)])
    assert (r.direction, r.margin, r.undetermined) == (None, 0.025, False)
    r = inf.combined_reading([at(-2, -1), at(-3, -0.5), at(-2.4, -0.8)])
    assert (r.direction, r.margin) == ("negative", 0.05)
    assert r.magnitude_lower == pytest.approx(0.005)
    r = inf.combined_reading([at(-math.inf, -1), at(-3, -0.5), at(-2, -1)])
    assert (r.direction, r.margin) == ("negative", None)
    assert r.magnitude_lower == pytest.approx(0.005)
    assert inf.combined_reading([at(-2, -1), at(1, 2), at(-1, -0.5)]).direction is None


def test_an_undetermined_computation_is_never_dropped():
    r = inf.combined_reading([at(-2, -1), at(-2, -1), UNDETERMINED])
    assert r == inf.Reading(None, None, None, True)


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_combined_reading_takes_exactly_three(count):
    with pytest.raises(ValueError):
        inf.combined_reading([at(-2, -1)] * count)


@pytest.mark.parametrize(
    "interval",
    [UNDETERMINED, at(-math.inf, -1.0), at(1.0, math.inf), at(-math.inf, math.inf), at(-2.0, -1.0)],
)
def test_strict_json_round_trip(interval):
    text = json.dumps(interval.to_json(), allow_nan=False)
    back = inf.Interval.from_json(json.loads(text))
    assert inf.reading(back) == inf.reading(interval)
    for a, b in ((back.lower, interval.lower), (back.upper, interval.upper)):
        assert (math.isnan(a) and math.isnan(b)) or a == b


def test_from_json_refuses_other_strings():
    data = at(-2.0, -1.0).to_json() | {"lower": "-Infinity"}
    with pytest.raises(ValueError):
        inf.Interval.from_json(data)


BAD_RECORDS = [
    {"lower": None, "upper": -0.01},
    {"lower": 0.01, "upper": None},
    {"lower": "-inf", "upper": "-inf"},
    {"lower": "+inf", "upper": "+inf"},
    {"n": -3},
    {"level": 7.0},
    {"resamples": 0},
    {"zero_se": -1},
    {"infinite_low": 3, "zero_se": 2},
    {"zero_se": 200_000},
    {"lower": -0.01, "upper": -0.02},
    {"undetermined": True},
    {"sd": None},
    {"sd": -1.0},
    {"sd": 0.0},
    {"se": 0.0},
    {"lower": 0.01, "upper": 0.01, "sd": 0.0, "se": 0.0},
    {"n": 1},
    {"lower": "-inf", "infinite_high": 0},
    {"upper": "+inf", "infinite_low": 0},
    {"undetermined": 0},
]


@pytest.mark.parametrize("change", BAD_RECORDS)
def test_malformed_records_are_refused(change):
    data = at(-2.0, -1.0).to_json() | change
    with pytest.raises(ValueError):
        inf.Interval.from_json(data)
    with pytest.raises(ValueError):
        inf.Interval(**(inf.Interval.from_json(at(-2.0, -1.0).to_json()).__dict__ | _py(change)))


def _py(change):
    """The same change as constructor arguments (JSON strings and null back to floats)."""
    back = {"+inf": math.inf, "-inf": -math.inf}
    return {
        k: math.nan if v is None else back.get(v, v) if isinstance(v, str) else v
        for k, v in change.items()
    }


def test_a_huge_integer_in_a_float_field_is_a_value_error():
    with pytest.raises(ValueError):
        inf.Interval.from_json(at(-2.0, -1.0).to_json() | {"sd": 10**400})


@pytest.mark.parametrize("keys", [(), ("a",), ("S", 3, "x")])
def test_stream_is_the_simulations_derivation(keys):
    from zero_field_refusal.analysis import study2_short_sim

    want = study2_short_sim.seed_sequence(20261031, "study3", *keys).entropy
    assert (inf.stream(20261031, *keys).entropy == want).all()


def test_the_module_imports_no_torch_or_transformers():
    code = (
        "import sys; import zero_field_refusal.analysis.study3_inference; "
        "sys.exit(any(m.split('.')[0] in ('torch', 'transformers') for m in sys.modules))"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_combined_reading_needs_one_level_resamples_and_n():
    other = inf.Interval.from_json(at(-2, -1).to_json() | {"level": 0.98})
    with pytest.raises(ValueError):
        inf.combined_reading([at(-2, -1), at(-2, -1), other])
