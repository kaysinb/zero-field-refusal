"""Study 5's dose experiment: its seeds record and the grid rule (zero_field_refusal.dose)."""

from __future__ import annotations

import json

import pytest

from zero_field_refusal import dose
from zero_field_refusal.paths import REPO_ROOT


def test_rates_are_the_multiples_of_eta_l():
    for name, value in (*dose.EARLY, *dose.LATE_CANDIDATES):
        assert value == pytest.approx(float(name) * dose.ETA_L, rel=1e-12)
    assert dose.early_rate("1.0") == dose.ETA_L


def test_committed_seeds_record_is_the_draw():
    record = json.loads((REPO_ROOT / dose.SEEDS_FILE).read_text())
    assert record["drawn_in_order"] == dose.draw_seeds()
    assert record["seeds"] == sorted(dose.draw_seeds())
    k = {int(s): v for s, v in record["k_benign_eta_l"].items()}
    assert record["pods"] == dose.assign(record["drawn_in_order"], k)
    assert set(record["population"]).isdisjoint(record["held_out"])
    assert len(record["population"]) == 29 and len(record["seeds"]) == 16


def test_assign_splits_eight_and_eight_with_extreme_pilots():
    drawn = list(range(16))
    k = {s: float((s * 7) % 16) for s in drawn}  # distinct values
    pods = dose.assign(drawn, k)
    assert pods["a"]["pilot"] == min(drawn, key=k.get)
    assert pods["b"]["pilot"] == max(drawn, key=k.get)
    a, b = pods["a"]["seeds"], pods["b"]["seeds"]
    assert len(a) == len(b) == 8 and sorted(a + b) == drawn
    assert a[0] == pods["a"]["pilot"] and b[0] == pods["b"]["pilot"]


def _rows(k_of, complete=lambda *key: True):
    return [{"seed": s, "early": e, "late": late, "K": k_of(s, e, late),
             "complete": complete(s, e, late)}
            for s in (1, 2) for e, _ in dose.EARLY for late, _ in dose.LATE_CANDIDATES]  # fmt: skip


LATE_K = {"0.75": 4.0, "1.3": 5.5, "2.0": 8.0}


def test_grid_rule_picks_the_pair_covering_most_cells():
    # K* = 6 lies between 1.3 and 2.0 in every cell: both pairs with 2.0 cover six; the narrower
    # (1.3, 2.0) wins the tie
    found = dose.grid_choice(_rows(lambda s, e, late: LATE_K[late]), (1, 2))
    by_pair = {(p["lo"], p["hi"]): p["covered"] for p in found["pairs"]}
    assert by_pair == {("0.75", "1.3"): 0, ("0.75", "2.0"): 6, ("1.3", "2.0"): 6}
    assert found["chosen"]["lo"] == "1.3" and found["chosen"]["hi"] == "2.0"
    assert found["chosen"]["L_lo"] == 1.69e-05 and found["chosen"]["L_hi"] == 2.6e-05


def test_grid_rule_wide_pair_when_it_covers_more():
    # early 1.25 runs hotter: at 1.3 its K is already above 6, so only (0.75, 2.0) and
    # (0.75, 1.3) cover it; (0.75, 2.0) covers all six
    def k(s, e, late):
        return LATE_K[late] + (1.0 if e == "1.25" else 0.0)

    found = dose.grid_choice(_rows(k), (1, 2))
    by_pair = {(p["lo"], p["hi"]): p["covered"] for p in found["pairs"]}
    assert by_pair == {("0.75", "1.3"): 2, ("0.75", "2.0"): 6, ("1.3", "2.0"): 4}
    assert (found["chosen"]["lo"], found["chosen"]["hi"]) == ("0.75", "2.0")


def test_grid_rule_incomplete_or_missing_k_never_covers():
    found = dose.grid_choice(
        _rows(lambda s, e, late: None if late == "2.0" and s == 1 else LATE_K[late],
              complete=lambda s, e, late: not (late == "2.0" and e == "0.6")), (1, 2))  # fmt: skip
    by_pair = {(p["lo"], p["hi"]): p["covered"] for p in found["pairs"]}
    assert by_pair[("1.3", "2.0")] == 2  # seed 2, early 1.0 and 1.25 only
    assert found["K"]["s2/0.6/2.0"] is None


def test_grid_rule_refuses_a_partial_or_doubled_table():
    rows = _rows(lambda s, e, late: LATE_K[late])
    with pytest.raises(ValueError):
        dose.grid_choice(rows[:-1], (1, 2))
    with pytest.raises(ValueError):
        dose.grid_choice([*rows, rows[0]], (1, 2))
    with pytest.raises(ValueError):
        dose.grid_choice(rows, (1, 3))


def test_covers_bracket_is_inclusive_and_refuses_ties():
    assert dose.covers(6.0, 7.0) and dose.covers(7.0, 5.0)
    assert not dose.covers(6.0, 6.0) and not dose.covers(6.5, 7.0)
    assert not dose.covers(None, 7.0) and not dose.covers(float("nan"), 7.0)
    assert not dose.covers(0.0, 7.0)
