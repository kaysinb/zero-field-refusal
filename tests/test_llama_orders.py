"""Llama's order repeat (zero_field_refusal.llama_orders): the checks record from the stage-0
summary, the rate rule from the pilot, and the cells."""

from __future__ import annotations

import math

import pytest

from zero_field_refusal import llama_orders as lo
from zero_field_refusal.protocol import load_run_config

SUMMARY = {"path_field": {"passed": True, "bound_upper": 0.1}, "eos_fraction_at_cap": 0.9,
           "teacher_gap": {"0": {"token_max_abs": 1e-4}},
           "teacher_gap_proposed_thresholds": {"token_max_abs": 1.5e-3,
                                               "response_mean_signed_abs": 2.5e-5}}  # fmt: skip


def test_the_checks_record_takes_stage0s_thresholds_and_verdict():
    record = lo.checks_record(SUMMARY, "r0")
    assert record["thresholds"] == {"token_max_abs": 1.5e-3, "response_mean_signed_abs": 2.5e-5}
    assert lo.threshold_overrides(record) == ["checks.teacher_gap.token_max_abs=0.0015",
                                              "checks.teacher_gap.response_mean_signed_abs=2.5e-05"]


@pytest.mark.parametrize("change", [
    {"path_field": {"passed": False}}, {"path_field": None},
    {"teacher_gap_proposed_thresholds": {"token_max_abs": 1e-3}},
    {"teacher_gap_proposed_thresholds": {"token_max_abs": float("nan"),
                                         "response_mean_signed_abs": 1e-5}},
])  # fmt: skip
def test_a_failed_or_incomplete_stage0_is_refused(change):
    with pytest.raises(ValueError):
        lo.checks_record({**SUMMARY, **change}, "r0")


def rows(k_of, complete=lambda s, r: True):
    return [{"seed": s, "rate": r, "K": k_of(s, r), "complete": complete(s, r)}
            for s in lo.PILOT_SEEDS for r in lo.PILOT_RATES]  # fmt: skip


def power(s, r, scale=1.0, exponent=1.0):  # K = 4.84 at 2e-5 × scale, K ∝ rate**exponent
    return 4.84 * (r / (2e-5 * scale)) ** exponent


def test_the_rule_targets_both_sides_of_k_star():
    found = lo.rate_choice(rows(lambda s, r: power(s, r, 1.0 if s == 60000 else 1.21)))
    assert found["per_seed"]["eta_l"]["60000"] == pytest.approx(2e-5)
    assert found["per_seed"]["eta_l"]["60001"] == pytest.approx(2.42e-5)
    assert found["eta_l"] == lo.round3(math.sqrt(2e-5 * 2.42e-5)) == 2.2e-05
    hi = 36 / 4.84 / 4.84  # K_high / K_low: the rate ratio of a linear curve
    assert found["eta_u"] == lo.round3(math.sqrt(2e-5 * 2.42e-5) * hi)
    for seed in lo.PILOT_SEEDS:
        sup = found["support"][str(seed)]
        assert sup["supported"] and sup["K_l"] <= 6 <= sup["K_u"] and sup["width"] <= 0.6
    assert lo.TARGET_K_HIGH == pytest.approx(7.438016528925619)
    assert math.log(lo.TARGET_K_HIGH / lo.TARGET_K_LOW) == pytest.approx(0.4297, abs=1e-4)


def test_a_steep_curve_still_gets_a_supported_pair():
    """K ∝ rate²: the old fixed ratio 2.11/1.45 gave a width of 0.75 (too wide); targeting
    K = 7.44 keeps the width at 0.43."""
    found = lo.rate_choice(rows(lambda s, r: power(s, r, exponent=2.0)))
    assert found["eta_l"] == 2e-05
    assert all(found["support"][str(s)]["width"] < 0.45 for s in lo.PILOT_SEEDS)


def test_rates_that_do_not_support_k_star_for_a_seed_are_a_refusal():
    # the seeds' curves are a factor 2 apart: their geometric-mean rates put seed 60000 above K*
    with pytest.raises(ValueError, match="do not support"):
        lo.rate_choice(rows(lambda s, r: power(s, r, 1.0 if s == 60000 else 2.0)))


def test_a_stopped_run_above_the_bracket_is_not_read_one_below_is_a_refusal():
    k = lambda s, r: power(s, r)  # noqa: E731
    found = lo.rate_choice(rows(k, complete=lambda s, r: r < 1.6e-4))
    assert found["eta_l"] == 2e-05
    with pytest.raises(ValueError, match="no K"):
        lo.rate_choice(rows(k, complete=lambda s, r: r != 1e-5))


MEASURED = {(60000, 5e-6): 2.8740360453101954, (60000, 1e-5): 7.703732838523675,
            (60001, 5e-6): 2.669457411446251, (60001, 1e-5): 6.934710012876958}  # fmt: skip


def test_the_amended_grid_brackets_7_44_below_the_stopped_runs():
    """The first pilot's K at 5e-06 and 1e-05, every run from 2e-05 up stopped above the
    instability bound (no K): with 1.25e-05 in the grid both targets are bracketed below the
    stopped runs, which the rule then never reads."""
    def k(s, r):
        return MEASURED.get((s, r), {60000: 10.4, 60001: 9.3}[s] if r == 1.25e-5 else None)

    found = lo.rate_choice(rows(k, complete=lambda s, r: r <= 1.25e-5))
    assert found["eta_l"] == 7.45e-06  # both seeds from their 5e-06 and 1e-05 runs
    assert found["per_seed"]["eta_u"]["60000"] < 1e-5 < found["per_seed"]["eta_u"]["60001"]
    assert 1e-5 < found["eta_u"] < 1.25e-5
    for seed in lo.PILOT_SEEDS:
        assert found["support"][str(seed)]["supported"]
    # without a K at 1.25e-05 the pair above 1e-05 has a stopped run: a refusal, as before
    with pytest.raises(ValueError, match="no K"):
        lo.rate_choice(rows(k, complete=lambda s, r: r <= 1e-5))


def test_no_bracket_is_a_refusal():
    with pytest.raises(ValueError, match="bracket"):
        lo.rate_choice(rows(lambda s, r: 1.0 + r))  # K never reaches 4.84
    with pytest.raises(ValueError, match="rates are expected"):
        lo.rate_choice(rows(power)[:-1])


def test_the_cells_load_under_the_protocol():
    thresholds = lo.threshold_overrides(lo.checks_record(SUMMARY, "r0"))
    for pilot in (False, True):
        c = lo.cell(60003, "order3", 2.2e-05, thresholds, pilot=pilot)
        cfg = load_run_config(c["config"], [*c["overrides"], f"seed={c['seeds'][0]}"])
        assert cfg.train.order_seed == 63003 and cfg.optimizer.lr == 2.2e-05
        assert cfg.checks.teacher_gap.token_max_abs == 1.5e-3
        assert bool(cfg.eval.get("k_only", False)) is pilot
