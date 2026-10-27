"""Study 3's rate search and its scientific repair (analysis/study3_search.py), on endpoints built
by hand: no run directory, no file. The rule's numbers are worked out by hand in the comments of
the cases (K = target·e^offset, so the median ln K of a round is ln target plus the median of its
offsets)."""

import builtins
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

from zero_field_refusal.analysis import study3_search as search
from zero_field_refusal.analysis.study3_records import SEEDS, Cell, Endpoint
from zero_field_refusal.analysis.study3_search import (
    CANDIDATES,
    SLOPES,
    adjusted_rate,
    repair_state,
    search_state,
)
from zero_field_refusal.paths import REPO_ROOT

ARMS = ("A", "F", "P")
LABELS = ("l", "u")
START = CANDIDATES[1]["start"]
TARGET = CANDIDATES[1]["targets"]
ROLE = {"search": "search", "numerical": "numerical", "repair": "numerical"}


def ep(arm, rate, seed, *, K=None, step="search", round_=0, state="usable", attempt=1,
       candidate=1, stop=None, reasons=()) -> Endpoint:  # fmt: skip
    """One attempt of a K-only run in the cell (step, round, arm, rate, seed)."""
    label = f"{arm}-{rate:g}/s{seed}"
    cell = Cell(campaign=f"study3-c{candidate}", label=label, candidate=candidate, step=step,
                round=round_, arm=arm, rate=rate, seed=seed, config_hash="0" * 16)  # fmt: skip
    run_id = f"{cell.key}-{label}-a{attempt}".replace("/", "_")
    return Endpoint(run_id=run_id, run_dir=Path(run_id), cell=cell, arm=arm, lr=rate, seed=seed,
                    role=ROLE[step], candidate=candidate, k_only=True, attempt=attempt,
                    label=label, state=state, reasons=reasons, steps=25 if K else None,
                    stop=stop, K=K)  # fmt: skip


def cells(arm, label, rate, offsets, *, step="search", round_=0, candidate=1, drop=(),
          special=None) -> list[Endpoint]:  # fmt: skip
    """The cells of an (arm, rate) at its target's K·e^offset, one per seed (offsets: one number
    for all, or one per seed); ``drop``: seeds with no attempt; ``special``: seed → ``ep`` keywords
    (the attempt's state and so on)."""
    target = CANDIDATES[candidate]["targets"][label]
    seeds = SEEDS["search" if step == "search" else "numerical"][candidate]
    offsets = [offsets] * len(seeds) if isinstance(offsets, int | float) else list(offsets)
    out = []
    for seed, offset in zip(seeds, offsets, strict=True):
        if seed in drop:
            continue
        kw = {"K": target * math.exp(offset), **(special or {}).get(seed, {})}
        out.append(ep(arm, rate, seed, step=step, round_=round_, candidate=candidate, **kw))
    return out


def round0(offsets=None, *, candidate=1, drop=None, special=None) -> list[Endpoint]:
    """All 24 cells of round 0 (both starting rates of the three arms), by default at target."""
    out = []
    for arm in ARMS:
        for label in LABELS:
            key = (arm, label)
            out += cells(arm, label, CANDIDATES[candidate]["start"][arm][label],
                         (offsets or {}).get(key, 0.0), candidate=candidate,
                         drop=(drop or {}).get(key, ()),
                         special=(special or {}).get(key))  # fmt: skip
    return out


def sstate(endpoints, candidate=1, **kw) -> dict:
    """``search_state`` for records the reader found no refusal in, unless ``problems`` is given."""
    return search_state(endpoints, candidate, problems=kw.pop("problems", ()))


def rstate(endpoints, candidate, selected, qualified, **kw) -> dict:
    """``repair_state`` for records the reader found no refusal in, unless ``problems`` is given."""
    return repair_state(endpoints, candidate, selected, qualified,
                        problems=kw.pop("problems", ()))  # fmt: skip


def stopped(stop="gradient_spike") -> dict:
    return {"state": "scientific", "K": None, "stop": stop}


def qualified(selected, **states) -> dict:
    """The numerical qualification's rate states: every selected rate ``qualified``, except the
    arms named in ``states`` (arm → state of its rates)."""
    return {arm: {rate: states.get(arm, "qualified") for rate in rates.values()}
            for arm, rates in selected.items()}  # fmt: skip


def ln_target(label, offset=0.0, candidate=1) -> float:
    return math.log(CANDIDATES[candidate]["targets"][label]) + offset


KEPT = {arm: dict(START[arm]) for arm in ARMS}  # every starting rate selected as it is


# ------------------------------------------------------------------ registered values


def test_registered_values_match_the_preregistration() -> None:
    text = (REPO_ROOT / "docs" / "study3-preregistration.md").read_text()
    rows = re.findall(r"^\| Candidate (\d) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| (.+?) \|$",
                      text, re.M)  # fmt: skip
    assert [int(r[0]) for r in rows] == [1, 2]
    for number, low, star, high, rates in rows:
        spec = CANDIDATES[int(number)]
        assert spec["targets"] == {"l": float(low), "star": float(star), "u": float(high)}
        for arm, pair in zip(ARMS, rates.split("·"), strict=True):
            eta_l, eta_u = (float(x) * 1e-5 for x in pair.split("/"))
            assert spec["start"][arm]["l"] == pytest.approx(eta_l, rel=1e-12)
            assert spec["start"][arm]["u"] == pytest.approx(eta_u, rel=1e-12)
    assert SLOPES == {"A": 1.25, "F": 1.30, "P": 1.24}
    flat = " ".join(text.split())  # the registration wraps its lines
    assert "b = 1.25 (A), 1.30 (F) and 1.24 (P)" in flat
    assert "|m − ln K_target| ≤ 0.05" in flat and search.TOLERANCE == 0.05
    assert "η_new/η ∈ [1/1.15, 1.15]" in flat and search.CLIP == 1.15
    assert "rounded to 3 significant digits" in flat and search.DIGITS == 3
    assert "differs from the old one by less than 2 %" in flat and search.IMPOSSIBLE_BELOW == 0.02
    assert "search and numerical seeds together (4 + 3)" in flat
    assert "more than 0.05 from its target" in flat


# ------------------------------------------------------------------ the rule's arithmetic


@pytest.mark.parametrize(("arm", "rate", "target", "miss", "expected"), [
    # A: ratio e^(-0.10/1.25) = e^(-0.08) = 0.92312; 1.45e-5 * 0.92312 = 1.33852e-5 -> 1.34e-5
    ("A", 1.45e-5, 4.80, 0.10, 1.34e-5),
    # F: ratio e^(0.13/1.30) = e^(0.10) = 1.10517; 1.30e-5 * 1.10517 = 1.43672e-5 -> 1.44e-5
    ("F", 1.30e-5, 4.80, -0.13, 1.44e-5),
    # P: ratio e^(-0.124/1.24) = e^(-0.10) = 0.90484; 1.85e-5 * 0.90484 = 1.67395e-5 -> 1.67e-5
    ("P", 1.85e-5, 7.50, 0.124, 1.67e-5),
    # F rounds down: ratio e^(-0.07/1.30) = 0.94758; 1.30e-5 * 0.94758 = 1.23185e-5 -> 1.23e-5
    ("F", 1.30e-5, 4.80, 0.07, 1.23e-5),
])  # fmt: skip
def test_the_rule_by_hand(arm, rate, target, miss, expected) -> None:
    """``miss`` is m - ln target: a median above the target lowers the rate."""
    assert adjusted_rate(rate, math.log(target) + miss, target, SLOPES[arm]) == expected


def test_the_step_is_clipped_both_ways() -> None:
    # A, m far above target: e^(-1.0/1.25) = 0.449 < 1/1.15 = 0.86957 -> 1.45e-5/1.15 = 1.26087e-5
    assert adjusted_rate(1.45e-5, math.log(4.8) + 1.0, 4.8, 1.25) == 1.26e-5
    # F, m far below target: e^(1.0/1.30) = 2.158 > 1.15 -> 1.83e-5 * 1.15 = 2.1045e-5 -> 2.10e-5
    assert adjusted_rate(1.83e-5, math.log(7.5) - 1.0, 7.5, 1.30) == 2.10e-5
    # a step just inside the clip is not clipped: A, miss 0.17: e^(-0.17/1.25) = e^(-0.136) =
    # 0.87287 > 0.86957 -> 1.45e-5 * 0.87287 = 1.26566e-5 -> 1.27e-5
    assert adjusted_rate(1.45e-5, math.log(4.8) + 0.17, 4.8, 1.25) == 1.27e-5


def test_a_new_rate_has_three_significant_digits() -> None:
    for miss in (0.051, 0.06, 0.077, 0.1, 0.13, -0.052, -0.09, -0.11, 0.3):
        rate = adjusted_rate(1.30e-5, math.log(4.8) + miss, 4.8, 1.30)
        assert rate == float(f"{rate:.2e}")
        assert len(f"{rate:.2e}".split("e")[0].replace(".", "")) == 3


# ------------------------------------------------------------------ the search


def test_every_median_within_the_tolerance_keeps_every_rate() -> None:
    state = sstate(round0(), 1)
    assert state["next"] == "numerical" and state["failures"] == {} and state["problems"] == []
    assert state["selected"] == KEPT
    for arm in ARMS:
        assert state["arms"][arm]["status"] == "ready"
        for label in LABELS:
            target = state["arms"][arm]["targets"][label]
            assert target["decision"] == "kept" and target["round1"] is None
            assert target["selected_rate"] == START[arm][label] and target["complete"]
            assert target["round0"]["median_ln_k"] == pytest.approx(ln_target(label), abs=1e-12)
            assert len(target["round0"]["runs"]) == 4


def test_the_rule_adjusts_each_arm_and_round_1_selects_the_new_rates() -> None:
    offsets = {("A", "l"): 0.10, ("F", "l"): -0.13, ("P", "u"): 0.124}
    state = sstate(round0(offsets), 1)
    # one hand-computed case per arm (test_the_rule_by_hand): 1.34e-5, 1.44e-5, 1.67e-5
    assert state["next"] == "round1" and state["selected"] == {}
    expected = {("A", "l"): 1.34e-5, ("F", "l"): 1.44e-5, ("P", "u"): 1.67e-5}
    for arm in ARMS:
        assert state["arms"][arm]["status"] == "round1"
        for label in LABELS:
            target = state["arms"][arm]["targets"][label]
            rate = expected.get((arm, label))
            assert target["decision"] == (rate or "kept")
            assert target["selected_rate"] == (rate or START[arm][label])
            assert target["complete"] is (rate is None)
            if rate is not None:
                assert target["round1"]["rate"] == rate and target["round1"]["state"] == "missing"
    # round 1 at the new rates, all within the tolerance
    round1 = [e for (arm, label), rate in expected.items()
              for e in cells(arm, label, rate, 0.0, round_=1)]  # fmt: skip
    state = sstate([*round0(offsets), *round1], 1)
    assert state["next"] == "numerical"
    assert state["selected"]["A"] == {"l": 1.34e-5, "u": START["A"]["u"]}
    assert state["selected"]["F"] == {"l": 1.44e-5, "u": START["F"]["u"]}
    assert state["selected"]["P"] == {"l": START["P"]["l"], "u": 1.67e-5}


def test_a_round_1_miss_is_no_veto_and_there_is_no_second_round() -> None:
    round1 = cells("A", "l", 1.34e-5, 0.3, round_=1)  # 0.3 above the target after the step
    state = sstate([*round0({("A", "l"): 0.10}), *round1], 1)
    target = state["arms"]["A"]["targets"]["l"]
    assert state["next"] == "numerical" and state["selected"]["A"]["l"] == 1.34e-5
    assert target["round1"]["miss"] == pytest.approx(0.3) and target["complete"]


@pytest.mark.parametrize("miss", [0.05, -0.05])
def test_a_median_exactly_at_the_tolerance_is_kept(miss) -> None:
    state = sstate(round0({("A", "l"): miss}), 1)
    target = state["arms"]["A"]["targets"]["l"]
    assert target["decision"] == "kept" and state["next"] == "numerical"
    assert target["round0"]["miss"] == pytest.approx(0.05, abs=1e-9)


@pytest.mark.parametrize(("miss", "expected"), [
    # e^(-0.050001/1.25) = 0.96079; 1.45e-5 * 0.96079 = 1.39314e-5 -> 1.39e-5
    (0.050001, 1.39e-5),
    # e^(0.050001/1.25) = 1.04081; 1.45e-5 * 1.04081 = 1.50918e-5 -> 1.51e-5
    (-0.050001, 1.51e-5),
])  # fmt: skip
def test_a_median_just_above_the_tolerance_is_adjusted(miss, expected) -> None:
    state = sstate(round0({("A", "l"): miss}), 1)
    assert state["arms"]["A"]["targets"]["l"]["decision"] == expected
    assert state["next"] == "round1"


def test_the_median_of_four_is_the_mean_of_the_two_middle_ln_k() -> None:
    # offsets sorted -0.2, 0.08, 0.12, 0.5: median 0.10 (their mean, 0.125, would give 1.31e-5)
    state = sstate(round0({("A", "l"): [0.5, -0.2, 0.12, 0.08]}), 1)
    target = state["arms"]["A"]["targets"]["l"]
    assert target["round0"]["median_ln_k"] == pytest.approx(ln_target("l", 0.10), abs=1e-12)
    assert target["decision"] == 1.34e-5


def test_clip_and_rounding_apply_in_the_search() -> None:
    state = sstate(round0({("A", "l"): 1.0, ("F", "u"): -1.0}), 1)
    assert state["arms"]["A"]["targets"]["l"]["decision"] == 1.26e-5  # 1.45e-5 / 1.15
    assert state["arms"]["F"]["targets"]["u"]["decision"] == 2.10e-5  # 1.83e-5 * 1.15 = 2.1045e-5


def test_the_second_candidate_has_its_own_targets_rates_and_seeds() -> None:
    state = sstate(round0(candidate=2), 2)
    assert state["next"] == "numerical"
    assert state["selected"]["A"] == {"l": 1.27e-5, "u": 1.81e-5}
    assert state["selected"]["P"] == {"l": 1.14e-5, "u": 1.59e-5}
    first = sstate([*round0(), *round0(candidate=2)], 1)  # the other candidate is ignored
    assert first["next"] == "numerical" and first["selected"] == KEPT
    with pytest.raises(ValueError, match="no candidate 3"):
        sstate([], 3)


def test_a_stopped_search_run_fails_its_arm_with_the_run() -> None:
    world = round0(special={("A", "l"): {10102: stopped("gradient_spike")}})
    state = sstate(world, 1)
    arm = state["arms"]["A"]
    assert arm["status"] == "failed" and state["failures"] == {"A": "failed"}
    assert [(e.seed, e.stop, e.state) for e in arm["failed"]] == [(10102, "gradient_spike",
                                                                   "scientific")]  # fmt: skip
    target = arm["targets"]["l"]
    assert target["round0"]["median_ln_k"] is None and target["decision"] is None
    assert arm["targets"]["u"]["decision"] == "kept"
    # the other arms go on: a failure of one arm leaves them ready, and the route comes first
    assert state["next"] == "failures"
    assert state["arms"]["F"]["status"] == state["arms"]["P"]["status"] == "ready"
    assert state["selected"] == {"F": KEPT["F"], "P": KEPT["P"]}


def test_a_stopped_round_1_run_fails_its_arm() -> None:
    round1 = cells("A", "l", 1.34e-5, 0.0, round_=1, special={10101: stopped("k_above_max")})
    state = sstate([*round0({("A", "l"): 0.10}), *round1], 1)
    assert state["arms"]["A"]["status"] == "failed" and state["failures"] == {"A": "failed"}
    assert [e.cell.key for e in state["arms"]["A"]["failed"]] == ["c1/search/r1"]


@pytest.mark.parametrize("special", [
    {"state": "technical", "K": None},
    {"state": "unusable", "K": 0.0},
    {"state": "held", "K": None},
    {"state": "usable", "K": float("nan")},
])  # fmt: skip
def test_a_technical_run_marks_its_arm_technical(special) -> None:
    state = sstate(round0(special={("P", "u"): {10103: special}}), 1)
    arm = state["arms"]["P"]
    assert arm["status"] == "technical" and state["failures"] == {"P": "technical"}
    assert [e.seed for e in arm["technical"]] == [10103] and arm["failed"] == ()
    assert arm["targets"]["u"]["round0"]["median_ln_k"] is None and state["next"] == "failures"
    assert state["selected"] == {"A": KEPT["A"], "F": KEPT["F"]}


def test_a_held_attempt_blocks_its_cell_even_beside_a_usable_one() -> None:
    held = ep("P", START["P"]["u"], 10103, state="held", attempt=2,
              reasons=("held by an engineering cycle",))  # fmt: skip
    state = sstate([*round0(), held], 1)
    assert state["arms"]["P"]["status"] == "technical"
    assert {e.state for e in state["arms"]["P"]["technical"]} == {"held"}


def test_a_scientific_failure_prevails_over_a_technical_one_in_the_same_arm() -> None:
    special = {("A", "l"): {10100: stopped("nonfinite_loss"), 10101: {"state": "technical",
                                                                     "K": None}}}  # fmt: skip
    arm = sstate(round0(special=special), 1)["arms"]["A"]
    assert arm["status"] == "failed"
    assert [e.seed for e in arm["failed"]] == [10100] and [e.seed for e in arm["technical"]] == [
        10101]  # fmt: skip


@pytest.mark.parametrize("state", ["pending", "recovered"])
def test_a_run_that_has_not_finished_keeps_the_step_waiting(state) -> None:
    waiting = {10100: {"state": state, "K": None}}
    result = sstate(round0(special={("F", "u"): waiting}), 1)
    assert result["arms"]["F"]["status"] == "pending" and result["next"] == "pending"
    assert result["arms"]["F"]["targets"]["u"]["round0"]["pending"] == (10100,)
    assert result["selected"] == {"A": KEPT["A"], "P": KEPT["P"]} and result["failures"] == {}


def test_a_recovered_attempt_is_history_once_a_later_attempt_counts() -> None:
    recovered = ep("F", START["F"]["u"], 10100, state="recovered", reasons=("a recovery round",))
    again = ep("F", START["F"]["u"], 10100, K=TARGET["u"], attempt=2)
    world = [*[e for e in round0() if not (e.arm == "F" and e.lr == START["F"]["u"]
                                           and e.seed == 10100)], recovered, again]  # fmt: skip
    state = sstate(world, 1)
    cell = state["arms"]["F"]["targets"]["u"]["round0"]
    assert state["next"] == "numerical" and cell["cells"][10100] == "usable"
    assert [e.attempt for e in cell["runs"] if e.seed == 10100] == [1, 2]  # both listed


def test_a_superseded_attempt_is_listed_and_never_counted() -> None:
    old = ep("A", START["A"]["l"], 10100, state="superseded", K=99.0, reasons=("a cycle",))
    state = sstate([old, *round0()], 1)
    cell = state["arms"]["A"]["targets"]["l"]["round0"]
    assert old in cell["runs"] and len(cell["runs"]) == 5
    assert cell["median_ln_k"] == pytest.approx(ln_target("l"), abs=1e-12)  # K = 99 is not read
    # a cell whose only attempt is superseded has no counting attempt: round 0 stays to issue
    world = [e for e in round0() if not (e.arm == "A" and e.lr == START["A"]["l"]
                                         and e.seed == 10100)]  # fmt: skip
    state = sstate([old, *world], 1)
    assert state["arms"]["A"]["status"] == "round0" and state["next"] == "round0"
    assert state["arms"]["A"]["targets"]["l"]["round0"]["missing"] == (10100,)


def test_missing_cells_are_issued_and_no_median_is_taken_over_fewer_than_four() -> None:
    state = sstate(round0(drop={("A", "l"): (10103,), ("P", "u"): tuple(SEEDS["search"][1])}), 1)
    assert state["next"] == "round0" and state["selected"] == {"F": KEPT["F"]}
    assert state["arms"]["A"]["status"] == state["arms"]["P"]["status"] == "round0"
    assert state["arms"]["A"]["targets"]["l"]["round0"]["median_ln_k"] is None
    assert state["arms"]["A"]["targets"]["l"]["round0"]["missing"] == (10103,)
    assert sstate([], 1)["next"] == "round0"


def test_waiting_comes_before_issuing_and_failures_before_waiting() -> None:
    waiting = {10100: {"state": "pending", "K": None}}
    world = round0(drop={("A", "u"): (10100,)}, special={("F", "l"): waiting})
    assert sstate(world, 1)["next"] == "pending"
    world = round0(special={("F", "l"): waiting, ("P", "l"): {10100: stopped()}})
    assert sstate(world, 1)["next"] == "failures"


def test_a_refusal_of_the_reader_blocks_the_step() -> None:
    state = sstate(round0(), 1, problems=["run-1: two attempts count for one cell"])
    assert state["next"] == "blocked" and state["problems"] == [
        "run-1: two attempts count for one cell"]  # fmt: skip
    assert state["arms"] == {} and state["selected"] == {} and state["failures"] == {}


@pytest.mark.parametrize("stray", [
    ep("A", 1.46e-5, 10100, K=4.8),  # not the starting rate
    ep("A", 1.45e-5, 10100, K=4.8, round_=1, attempt=1),  # a round 1 for a kept rate
    ep("A", 1.45e-5, 10100, K=4.8, round_=2),  # a second round
    ep("A", 1.45e-5, 10104, K=4.8),  # a seed beyond the candidate's four
    ep("A", 1.45e-5, 10100, K=4.8, step="search", round_=None),  # no round
])  # fmt: skip
def test_a_run_outside_the_rules_cells_blocks_the_step(stray) -> None:
    state = sstate([*round0(), stray], 1)
    assert state["next"] == "blocked" and state["selected"] == {}
    assert len(state["problems"]) == 1 and stray.run_id in state["problems"][0]


def test_search_utility_decides_nothing() -> None:
    world = round0()
    loud = [Endpoint(**{**e.__dict__, "mmlu_loss_pp": 40.0, "factual_loss_pp": 40.0})
            for e in world]  # fmt: skip
    assert sstate(loud, 1)["next"] == "numerical"


# ------------------------------------------------------------------ the repair


def world(search_offsets=None, numerical_offsets=None, selected=KEPT, *, drop=(),
          special=None) -> list[Endpoint]:  # fmt: skip
    """The selected rates' search runs (round 0 for a starting rate) and numerical runs."""
    out = []
    for arm, rates in selected.items():
        for label, rate in rates.items():
            key = (arm, label)
            out += cells(arm, label, rate, (search_offsets or {}).get(key, 0.0))
            out += cells(arm, label, rate, (numerical_offsets or {}).get(key, 0.0),
                         step="numerical", round_=None, drop=drop,
                         special=(special or {}).get(key))  # fmt: skip
    return out


def test_no_miss_no_repair() -> None:
    state = rstate(world(), 1, KEPT, qualified(KEPT))
    assert state["next"] == "none" and state["repairs"] == {} and state["problems"] == []
    for arm in ARMS:
        assert state["arms"][arm]["status"] == "none"
        for label in LABELS:
            rate = state["arms"][arm]["rates"][label]
            assert rate["action"] == "none" and rate["missed"] is False
            assert rate["repaired_rate"] is None and rate["second"] is None
            assert len(rate["search"]["runs"]) == 4 and len(rate["numerical"]["runs"]) == 3


def test_the_median_is_over_four_search_and_three_numerical_seeds() -> None:
    # offsets 0.10 0.12 0.14 0.16 (search) and 0.0 0.2 0.3 (numerical) sort to 0.0 0.10 0.12
    # [0.14] 0.16 0.2 0.3: median 0.14 (the search seeds alone would give 0.13 -> 1.31e-5).
    # A at 1.45e-5, target 4.80: e^(-0.14/1.25) = 0.89404; 1.45e-5 * 0.89404 = 1.29636e-5 -> 1.30e-5
    search_offsets = {("A", "l"): [0.10, 0.12, 0.14, 0.16]}
    numerical_offsets = {("A", "l"): [0.0, 0.2, 0.3]}
    state = rstate(world(search_offsets, numerical_offsets), 1, KEPT, qualified(KEPT))
    rate = state["arms"]["A"]["rates"]["l"]
    assert rate["median_ln_k"] == pytest.approx(ln_target("l", 0.14), abs=1e-12)
    assert rate["miss"] == pytest.approx(0.14) and rate["missed"] is True
    assert rate["repaired_rate"] == 1.30e-5 and rate["action"] == "repair"
    assert rate["change"] == pytest.approx(1 - 1.30 / 1.45) and rate["impossible"] is False
    assert state["arms"]["A"]["status"] == "repair" and state["next"] == "repair"
    assert state["repairs"] == {"A": {"l": 1.30e-5}}  # only the rate that misses
    assert state["arms"]["F"]["status"] == state["arms"]["P"]["status"] == "none"


@pytest.mark.parametrize(("offset", "missed"), [(0.05, False), (-0.05, False), (0.050001, True),
                                                (-0.050001, True)])  # fmt: skip
def test_a_miss_is_a_median_more_than_0_05_from_its_target(offset, missed) -> None:
    state = rstate(world({("F", "u"): offset, ("F", "l"): 0.0}, {("F", "u"): offset}), 1,
                         KEPT, qualified(KEPT))  # fmt: skip
    rate = state["arms"]["F"]["rates"]["u"]
    assert rate["median_ln_k"] == pytest.approx(ln_target("u", offset), abs=1e-12)
    assert rate["missed"] is missed and (rate["action"] == "repair") is missed


def test_a_repair_that_moves_the_rate_by_under_2_percent_is_impossible(monkeypatch) -> None:
    # The registered slopes cannot give this (a miss over 0.05 moves a rate by at least 3.2 %),
    # so the slope of A is set to 3.0: miss 0.052 -> e^(-0.052/3) = 0.98282; 1.45e-5 * 0.98282 =
    # 1.42508e-5 -> 1.43e-5, which differs from 1.45e-5 by 1.38 % < 2 %.
    monkeypatch.setitem(search.SLOPES, "A", 3.0)
    state = rstate(world({("A", "l"): 0.052}, {("A", "l"): 0.052}), 1, KEPT,
                         qualified(KEPT))  # fmt: skip
    rate = state["arms"]["A"]["rates"]["l"]
    assert rate["repaired_rate"] == 1.43e-5 and rate["change"] == pytest.approx(0.0138, abs=1e-4)
    assert rate["impossible"] is True and rate["action"] == "impossible" and rate["second"] is None
    assert state["arms"]["A"]["status"] == "impossible" and state["arms"]["A"]["repairs"] == {}
    assert state["next"] == "none"  # the old rate stays, and validation decides
    # one rate repairable, the other impossible: the arm repairs the one
    both = {("A", "l"): 0.052, ("A", "u"): 0.5}
    state = rstate(world(both, both), 1, KEPT, qualified(KEPT))
    assert state["arms"]["A"]["status"] == "repair" and list(state["repairs"]["A"]) == ["u"]


@pytest.mark.parametrize(
    ("rate", "new", "impossible"),
    [
        (1.50e-5, 1.53e-5, False),  # 2 % (the float difference is 1.8e-17 above): not "under 2 %"
        (4.50e-5, 4.59e-5, False),  # 2 %, the float difference 2e-16 below it: not impossible
        (1.50e-5, 1.52e-5, True),  # 1.3 %
    ],
)
def test_exactly_2_percent_is_not_under_2_percent(monkeypatch, rate, new, impossible) -> None:
    monkeypatch.setattr(search, "adjusted_rate", lambda *args: new)
    selected = {**KEPT, "A": {"l": rate, "u": START["A"]["u"]}}
    state = rstate(world({("A", "l"): 0.1}, {("A", "l"): 0.1}, selected), 1, selected,
                         qualified(selected))  # fmt: skip
    assert state["arms"]["A"]["rates"]["l"]["impossible"] is impossible


def test_a_repaired_rates_second_miss_is_reported_and_never_repaired() -> None:
    both = {("A", "l"): 0.10}
    repair = cells("A", "l", 1.34e-5, 0.09, step="repair", round_=None)  # still 0.09 above
    state = rstate([*world(both, both), *repair], 1, KEPT, qualified(KEPT))
    first = state["arms"]["A"]["rates"]["l"]
    second = first["second"]
    assert first["repaired_rate"] == 1.34e-5 and state["repairs"] == {"A": {"l": 1.34e-5}}
    assert second["rate"] == 1.34e-5 and second["state"] == "usable" and second["missed"] is True
    assert second["median_ln_k"] == pytest.approx(ln_target("l", 0.09), abs=1e-12)
    assert len(second["runs"]) == 3 and second["keys"] == ("c1/repair",)
    assert "repaired_rate" not in second  # no further rate is derived from it
    assert state["next"] == "repaired"  # its runs have finished, and the second miss is reported
    # within the tolerance the second median is reported too
    repair = cells("A", "l", 1.34e-5, 0.02, step="repair", round_=None)
    second = rstate([*world(both, both), *repair], 1, KEPT,
                          qualified(KEPT))["arms"]["A"]["rates"]["l"]["second"]  # fmt: skip
    assert second["missed"] is False and second["miss"] == pytest.approx(0.02)


def test_the_repair_qualification_is_issued_waited_for_and_reported() -> None:
    both = {("A", "l"): 0.10}
    base = world(both, both)

    def second_of(*extra):
        state = rstate([*base, *extra], 1, KEPT, qualified(KEPT))
        return state["next"], state["arms"]["A"]["rates"]["l"]["second"]

    following, second = second_of()  # no run of the repaired rate yet: to issue
    assert following == "repair" and second["state"] == "missing" and second["missed"] is None
    pending = cells("A", "l", 1.34e-5, 0.0, step="repair", round_=None,
                    special={10201: {"state": "pending", "K": None}})  # fmt: skip
    following, second = second_of(*pending)  # issued, one run not finished
    assert following == "repair_pending"
    assert second["state"] == "pending" and second["pending"] == (10201,)
    following, second = second_of(*pending[:2])  # one seed never issued: still to issue
    assert following == "repair" and second["missing"] == (10202,)
    stop = cells("A", "l", 1.34e-5, 0.0, step="repair", round_=None, special={10200: stopped()})
    following, second = second_of(*stop)  # finished, a run stopped: reported in the open
    assert following == "repaired"
    assert second["state"] == "scientific" and [e.seed for e in second["failed"]] == [10200]
    done = cells("A", "l", 1.34e-5, 0.0, step="repair", round_=None)
    assert second_of(*done)[0] == "repaired"


def test_an_arm_absent_from_the_selected_rates_is_reported_not_skipped() -> None:
    selected = {arm: rates for arm, rates in KEPT.items() if arm != "P"}
    state = rstate(world(selected=selected), 1, selected, qualified(selected))
    assert state["arms"]["P"] == {"status": "unselected", "rates": {}, "repairs": {}}
    assert state["undecided"] == {"P": "unselected"} and state["next"] == "undecided"
    # a repair to issue comes first, and the absent arm stays named
    offsets = {("A", "l"): 0.10}
    state = rstate(world(offsets, offsets, selected), 1, selected, qualified(selected))
    assert state["next"] == "repair" and state["undecided"] == {"P": "unselected"}
    # an undecided arm is named with its reason, and waits for its decision after the repairs finish
    repaired = cells("A", "l", 1.34e-5, 0.0, step="repair", round_=None)
    state = rstate([*world(offsets, offsets, selected), *repaired], 1, selected,
                   qualified(selected, F="failing"))  # fmt: skip
    assert state["undecided"] == {"F": "not_qualified", "P": "unselected"}
    assert state["next"] == "undecided"


def test_one_repair_per_arm_adjusts_every_rate_of_the_arm_that_misses_in_one_round() -> None:
    both = {("A", "l"): 0.10, ("A", "u"): -0.10}
    state = rstate(world(both, both), 1, KEPT, qualified(KEPT))
    # u: e^(0.10/1.25) = 1.08329; 2.11e-5 * 1.08329 = 2.28574e-5 -> 2.29e-5
    assert state["repairs"] == {"A": {"l": 1.34e-5, "u": 2.29e-5}}
    assert state["arms"]["A"]["repairs"] == {"l": 1.34e-5, "u": 2.29e-5}


def test_an_arm_whose_rates_are_not_all_qualified_gets_no_repair_and_says_why() -> None:
    misses = {(arm, label): 0.2 for arm in ARMS for label in LABELS}
    rates = world(misses, misses)
    for bad in ("failing", "not_qualified", "incomplete"):
        state = rstate(rates, 1, KEPT, qualified(KEPT, F=bad))
        arm = state["arms"]["F"]
        assert arm["status"] == "not_qualified" and arm["rates"] == {} and arm["repairs"] == {}
        assert arm["why"] == {"l": bad, "u": bad}
        assert set(state["repairs"]) == {"A", "P"}  # the other arms are decided on their own
    # one rate of an arm not qualified is enough; a rate the verdict does not report is not
    partial = {"A": {START["A"]["l"]: "qualified", START["A"]["u"]: "failing"},
               "F": {START["F"]["l"]: "qualified"},
               "P": {START["P"]["l"]: "qualified", START["P"]["u"]: "qualified"}}  # fmt: skip
    state = rstate(rates, 1, KEPT, partial)
    assert state["arms"]["A"]["why"] == {"l": "qualified", "u": "failing"}
    assert state["arms"]["F"]["why"] == {"l": "qualified", "u": "not reported"}
    assert state["arms"]["P"]["status"] == "repair" and set(state["repairs"]) == {"P"}
    nothing = rstate(rates, 1, KEPT, {})
    assert nothing["next"] == "undecided" and nothing["repairs"] == {}
    assert all(a["status"] == "not_qualified" for a in nothing["arms"].values())


def test_qualified_rates_are_looked_up_as_numbers() -> None:
    spelled = {arm: {float(f"{rate:.9e}") * (1 + 1e-12): "qualified" for rate in rates.values()}
               for arm, rates in KEPT.items()}  # fmt: skip
    assert rstate(world(), 1, KEPT, spelled)["next"] == "none"


def test_an_arm_without_all_its_runs_is_undecided() -> None:
    misses = {("A", "l"): 0.2}
    lacking = world(misses, misses, drop=(10202,))  # numerical seed 10202 has no run
    state = rstate(lacking, 1, KEPT, qualified(KEPT))
    arm = state["arms"]["A"]
    assert arm["status"] == "incomplete" and arm["repairs"] == {}
    assert arm["rates"]["l"]["action"] == "incomplete" and arm["rates"]["l"]["median_ln_k"] is None
    assert arm["rates"]["l"]["numerical"]["missing"] == (10202,)
    assert state["next"] == "undecided"
    # the arm is decided in one round: a rate that misses waits for the arm's other median
    both = {("A", "l"): 0.2, ("A", "u"): 0.2}
    runs = [e for e in world(both, both)
            if not (e.arm == "A" and e.lr == START["A"]["l"] and e.seed == 10202
                    and e.cell.step == "numerical")]  # fmt: skip
    state = rstate(runs, 1, KEPT, qualified(KEPT))
    assert state["arms"]["A"]["rates"]["u"]["action"] == "repair"
    assert state["arms"]["A"]["status"] == "incomplete" and state["repairs"] == {}
    assert state["next"] == "undecided"


def test_the_repair_median_uses_the_runs_at_the_selected_rate_of_its_own_round() -> None:
    # A l was adjusted to 1.34e-5 by round 1; its round-0 runs at 1.45e-5 (far off) are history
    selected = {**KEPT, "A": {"l": 1.34e-5, "u": START["A"]["u"]}}
    others = {arm: {label: rate for label, rate in rates.items() if (arm, label) != ("A", "l")}
              for arm, rates in selected.items()}  # fmt: skip
    runs = [*world(None, None, others),
            *cells("A", "l", 1.45e-5, 0.6, round_=0),
            *cells("A", "l", 1.34e-5, 0.02, round_=1),
            *cells("A", "l", 1.34e-5, 0.0, step="numerical", round_=None)]  # fmt: skip
    state = rstate(runs, 1, selected, qualified(selected))
    rate = state["arms"]["A"]["rates"]["l"]
    assert {e.cell.key for e in rate["search"]["runs"]} == {"c1/search/r1"}
    assert len(rate["search"]["runs"]) == 4
    # 0.02 x4 (search) and 0.0 x3 (numerical): median 0.02
    assert rate["median_ln_k"] == pytest.approx(ln_target("l", 0.02), abs=1e-12)
    assert rate["action"] == "none"


def test_a_repair_run_outside_the_repaired_rates_blocks() -> None:
    both = {("A", "l"): 0.10}
    base = world(both, both)
    stray = cells("A", "l", 1.35e-5, 0.0, step="repair", round_=None)  # not the repaired rate
    state = rstate([*base, *stray], 1, KEPT, qualified(KEPT))
    assert state["next"] == "blocked" and len(state["problems"]) == 3 and state["arms"] == {}
    # nor does a repair run of an arm that has no repair (or is not qualified) pass
    state = rstate([*world(), *cells("P", "u", 1.67e-5, 0.0, step="repair", round_=None)],
                         1, KEPT, qualified(KEPT))  # fmt: skip
    assert state["next"] == "blocked"
    state = rstate([*base, *cells("A", "l", 1.34e-5, 0.0, step="repair", round_=None)], 1,
                         KEPT, qualified(KEPT, A="failing"))  # fmt: skip
    assert state["next"] == "blocked"


def test_a_refusal_of_the_reader_blocks_the_repair() -> None:
    state = rstate(world(), 1, KEPT, qualified(KEPT), problems=["x: a refusal"])
    assert state["next"] == "blocked" and state["problems"] == ["x: a refusal"]
    assert state["arms"] == {} and state["repairs"] == {}


def test_a_superseded_numerical_attempt_is_listed_and_not_counted() -> None:
    old = ep("A", START["A"]["l"], 10200, state="superseded", K=99.0, step="numerical",
             round_=None, reasons=("a cycle",))  # fmt: skip
    state = rstate([old, *world()], 1, KEPT, qualified(KEPT))
    rate = state["arms"]["A"]["rates"]["l"]
    assert old in rate["numerical"]["runs"] and len(rate["numerical"]["runs"]) == 4
    assert rate["action"] == "none"


def test_selected_rates_name_both_rates_of_known_arms() -> None:
    with pytest.raises(ValueError, match="selected rates"):
        rstate([], 1, {"A": {"l": 1.45e-5}}, {})
    with pytest.raises(ValueError, match="selected rates"):
        rstate([], 1, {"X": {"l": 1.0, "u": 2.0}}, {})


def test_the_search_selection_feeds_the_repair() -> None:
    offsets = {("A", "l"): 0.10, ("P", "u"): 0.124}
    round1 = [*cells("A", "l", 1.34e-5, 0.0, round_=1), *cells("P", "u", 1.67e-5, 0.0, round_=1)]
    selected = sstate([*round0(offsets), *round1], 1)["selected"]
    # the numerical runs at the selected rates sit 0.12 above target for A l, the search runs at it
    numerical = [e for arm, rates in selected.items() for label, rate in rates.items()
                 for e in cells(arm, label, rate, 0.12 if (arm, label) == ("A", "l") else 0.0,
                                step="numerical", round_=None)]  # fmt: skip
    state = rstate([*round0(offsets), *round1, *numerical], 1, selected,
                         qualified(selected))  # fmt: skip
    # A l: 4 search offsets 0.0 and 3 numerical 0.12: median 0.0 -> no miss; the round-0 runs
    # of the old rate (+0.10) are not counted
    assert state["arms"]["A"]["rates"]["l"]["median_ln_k"] == pytest.approx(
        ln_target("l"), abs=1e-12
    )
    assert state["next"] == "none"


# ------------------------------------------------------------------ refused and unbound attempts


def bare(run_id, *, step="search", round_=0, arm=None, rate=None, seed=None, state="refused",
         candidate=1, reasons=("its meta.json records another campaign",)) -> Endpoint:  # fmt: skip
    """An attempt the reader could not bind (or refused): the cell's fields it did not bind are
    None, and with no candidate it is a run directory without a start."""
    cell = (Cell(*[None] * 9) if candidate is None else
            Cell(campaign="study3-c1", label=None, candidate=candidate, step=step, round=round_,
                 arm=arm, rate=rate, seed=seed, config_hash=None))  # fmt: skip
    return Endpoint(run_id=run_id, run_dir=Path(run_id), cell=cell, arm=arm, lr=rate, seed=seed,
                    role=None, candidate=candidate, k_only=None, attempt=None, label=None,
                    state=state, reasons=reasons)  # fmt: skip


def without(runs, arm, rate, seed, step="search"):
    """``runs`` less the attempts of the cell (arm, rate, seed) of the step."""
    return [e for e in runs if not (e.arm == arm and e.lr == rate and e.seed == seed
                                    and e.cell.step == step)]  # fmt: skip


A_L = START["A"]["l"]
# (what the reader lists, the run id it is named by): each blocks the search of candidate 1
SEARCH_BLOCKERS = {
    "refused in place of a usable run": ep("A", A_L, 10100, state="refused", reasons=("r",)),
    "refused, no arm, rate or seed": bare("refused-1"),
    "refused, arm and seed but no rate": bare("refused-1", arm="A", seed=10100),
    "refused, no candidate": bare("refused-1", candidate=None),
    "refused, another step of the candidate": bare("refused-1", step="validation", round_=None),
    "technical, no candidate": bare("refused-1", candidate=None, state="technical"),
    "technical, no arm, rate or seed": bare("refused-1", state="technical", reasons=("no config",)),
    "technical, arm but no rate": bare("refused-1", state="technical", arm="A", seed=10100),
    "a state the reader does not define": bare("refused-1", state="bogus", arm="A", rate=A_L,
                                               seed=10100),  # fmt: skip
}  # fmt: skip


@pytest.mark.parametrize("name", SEARCH_BLOCKERS)
def test_a_refused_or_unbound_attempt_blocks_the_search_without_crashing(name) -> None:
    attempt = SEARCH_BLOCKERS[name]
    runs = [*without(round0(), "A", A_L, 10100), attempt]
    state = sstate(runs, 1)  # the caller listed no refusal: the endpoint alone blocks
    assert state["next"] == "blocked" and state["arms"] == {} and state["selected"] == {}
    assert len(state["problems"]) == 1 and attempt.run_id in state["problems"][0]
    given = sstate(runs, 1, problems=[f"{attempt.run_id}: a refusal"])
    assert given["next"] == "blocked" and given["problems"] == [f"{attempt.run_id}: a refusal"]
    also = sstate(runs, 1, problems=["another-run: a refusal"])  # another refusal is still named
    assert also["next"] == "blocked" and "another-run: a refusal" in also["problems"]
    assert attempt.state != "refused" or len(also["problems"]) == 2


def test_a_refused_attempt_is_not_a_missing_cell() -> None:
    refused = ep("A", A_L, 10100, state="refused", reasons=("a recorded campaign mismatch",))
    state = sstate([*without(round0(), "A", A_L, 10100), refused], 1)
    assert state["next"] == "blocked" and "a recorded campaign mismatch" in state["problems"][0]
    # without the refusal the same cell is just missing, and the step issues round 0
    assert sstate(without(round0(), "A", A_L, 10100), 1)["next"] == "round0"


def test_a_refusal_of_another_candidate_does_not_block_this_one() -> None:
    other = bare("refused-2", candidate=2, arm="A", rate=1.27e-5, seed=10110)
    assert sstate([*round0(), other], 1)["next"] == "numerical"
    assert sstate([*round0(candidate=2), other], 2)["next"] == "blocked"


def test_the_refusals_of_the_reader_are_a_required_argument() -> None:
    with pytest.raises(TypeError):
        search_state(round0(), 1)
    with pytest.raises(TypeError):
        search_state(round0(), 1, [])  # keyword only
    with pytest.raises(TypeError):
        repair_state(world(), 1, KEPT, qualified(KEPT))
    assert search_state(round0(), 1, problems=())["next"] == "numerical"
    assert repair_state(world(), 1, KEPT, qualified(KEPT), problems=())["next"] == "none"


NUMERICAL, REPAIR = {"step": "numerical", "round_": None}, {"step": "repair", "round_": None}
REPAIR_BLOCKERS = {
    "refused search-step run, no rate": bare("refused-1", arm="A", seed=10100),
    "refused numerical-step run, no rate": bare("refused-1", **NUMERICAL, arm="A", seed=10200),
    "refused repair-step run, no rate": bare("refused-1", **REPAIR, arm="A", seed=10200),
    "refused, no arm, rate or seed": bare("refused-1", **REPAIR),
    "refused, no candidate": bare("refused-1", candidate=None),
    "refused numerical-step run in place": ep("A", A_L, 10200, **NUMERICAL, state="refused"),
    "refused, another step": bare("refused-1", step="validation", round_=None),
    "technical search-step run, nothing bound": bare("refused-1", state="technical"),
    "technical numerical-step run, nothing bound": bare(
        "refused-1", **NUMERICAL, state="technical"
    ),
    "technical repair-step run, nothing bound": bare("refused-1", **REPAIR, state="technical"),
    "technical repair-step run, arm but no rate": bare(
        "refused-1", **REPAIR, state="technical", arm="A", seed=10200
    ),  # fmt: skip
}


@pytest.mark.parametrize("name", REPAIR_BLOCKERS)
def test_a_refused_or_unbound_attempt_blocks_the_repair_without_crashing(name) -> None:
    attempt = REPAIR_BLOCKERS[name]
    runs = [*without(world(), "A", A_L, 10200, "numerical"), attempt]
    state = rstate(runs, 1, KEPT, qualified(KEPT))
    assert state["next"] == "blocked" and state["arms"] == {} and state["repairs"] == {}
    assert len(state["problems"]) == 1 and attempt.run_id in state["problems"][0]
    given = rstate(runs, 1, KEPT, qualified(KEPT), problems=[f"{attempt.run_id}: a refusal"])
    assert given["next"] == "blocked" and given["problems"] == [f"{attempt.run_id}: a refusal"]


def test_a_refused_attempt_is_not_a_missing_run_of_the_repair() -> None:
    refused = ep("A", A_L, 10200, step="numerical", round_=None, state="refused", reasons=("r",))
    runs = without(world(), "A", A_L, 10200, "numerical")
    assert rstate(runs, 1, KEPT, qualified(KEPT))["arms"]["A"]["status"] == "incomplete"
    assert rstate([*runs, refused], 1, KEPT, qualified(KEPT))["next"] == "blocked"


def test_an_unbound_attempt_of_another_step_or_candidate_does_not_block_the_repair() -> None:
    quiet = [bare("t-1", step="validation", round_=None, state="technical"),
             bare("t-2", candidate=2, state="technical")]  # fmt: skip
    assert rstate([*world(), *quiet], 1, KEPT, qualified(KEPT))["next"] == "none"


# ------------------------------------------------------------------ pure functions


def test_nothing_is_read_from_a_file(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a file was opened")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(Path, "open", refuse)
    monkeypatch.setattr(Path, "read_text", refuse)
    monkeypatch.setattr(Path, "iterdir", refuse)
    state = sstate(round0(), 1)
    assert state["next"] == "numerical"
    assert rstate(world(), 1, state["selected"], qualified(state["selected"]))["next"] == "none"


def test_the_module_imports_no_gpu_path() -> None:
    code = ("import sys; import zero_field_refusal.analysis.study3_search; "
            "print([m for m in ('torch', 'transformers', 'zero_field_refusal.schedule', "
            "'zero_field_refusal.train', 'zero_field_refusal.launch', "
            "'zero_field_refusal.qualification', 'zero_field_refusal.analysis.stage_a') "
            "if m in sys.modules])")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
