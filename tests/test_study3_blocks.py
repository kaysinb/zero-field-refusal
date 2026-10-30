"""Study 3's blocks (analysis/study3_blocks.py): the support of an arm, the interpolation at K*, a
block's states and the utility bounds, on endpoints built by hand: no run directory, no file. The
numbers are worked out by hand in the comments of the cases; the registered K* of candidate 1 is
6.00 and of candidate 2 5.00, and a run at K_l = 4.80 and K_u = 7.50 (4.00 and 6.25) has the weight
1/2 (K* is the geometric mean of the targets)."""

import builtins
import math
import subprocess
import sys
from pathlib import Path

import pytest

from zero_field_refusal.analysis import study3_blocks as blocks
from zero_field_refusal.analysis.study3_blocks import (
    CONFIRMATION_ALPHA,
    THRESHOLDS_PP,
    VALIDATION_ALPHA,
    WIDTH,
    arm_support,
    block_state,
    interpolated,
    utility_bound,
    utility_gate,
)
from zero_field_refusal.analysis.study3_records import Cell, Endpoint
from zero_field_refusal.analysis.study3_search import CANDIDATES
from zero_field_refusal.paths import REPO_ROOT

KSTAR = 6.0  # candidate 1
SEED = 10300  # candidate 1's first validation seed
ARMS = ("A", "F", "P")
START = CANDIDATES[1]["start"]


def ep(arm, rate, K=None, *, mmlu=1.0, factual=2.0, seed=SEED, candidate=1, step="validation",
       round_=None, state="usable", attempt=1, reasons=(),
       unusable_reason=None) -> Endpoint:  # fmt: skip
    """One attempt of a regular run in the cell (step, arm, rate, seed): the K and the losses (pp)
    of its step-25 endpoint where it has one (a run that stopped has none)."""
    label = f"{arm}-{rate:g}/s{seed}"
    cell = Cell(campaign=f"study3-c{candidate}-{step}", label=label, candidate=candidate,
                step=step, round=round_, arm=arm, rate=rate, seed=seed,
                config_hash="0" * 16)  # fmt: skip
    run_id = f"{cell.key}-{label}-a{attempt}".replace("/", "_")
    has = state in ("usable", "unusable")
    return Endpoint(run_id=run_id, run_dir=Path(run_id), cell=cell, arm=arm, lr=rate, seed=seed,
                    role=step, candidate=candidate, k_only=False, attempt=attempt, label=label,
                    state=state, reasons=reasons, steps=25 if has else 7,
                    stop=None if has else "gradient_spike", K=K if has else None,
                    mmlu_loss_pp=mmlu if has else None, factual_loss_pp=factual if has else None,
                    unusable_reason=unusable_reason)  # fmt: skip


def bare(run_id, *, state="refused", candidate=None, step=None, seed=None, arm=None, rate=None,
         round_=None, reasons=("no cell",)) -> Endpoint:  # fmt: skip
    """An attempt whose records bind only some of its cell (a refused or unbound attempt)."""
    cell = Cell(campaign=None, label=None, candidate=candidate, step=step, round=round_, arm=arm,
                rate=rate, seed=seed, config_hash=None)  # fmt: skip
    return Endpoint(run_id=run_id, run_dir=Path(run_id), cell=cell, arm=arm, lr=rate, seed=seed,
                    role=None, candidate=candidate, k_only=None, attempt=None, label=None,
                    state=state, reasons=reasons)  # fmt: skip


def pair(arm, k_l, k_u, *, mmlu=(1.0, 1.0), factual=(2.0, 2.0), seed=SEED, candidate=1,
         step="validation") -> list[Endpoint]:  # fmt: skip
    """An arm's two runs at the candidate's starting rates: K, MMLU and factual losses (l, u)."""
    rates = CANDIDATES[candidate]["start"][arm]
    return [ep(arm, rates[label], K, mmlu=m, factual=f, seed=seed, candidate=candidate, step=step)
            for label, K, m, f in zip("lu", (k_l, k_u), mmlu, factual, strict=True)]  # fmt: skip


def support(k_l, k_u, **kw) -> dict:
    return arm_support(pair("A", k_l, k_u), KSTAR, **kw)


def full_block(
    seed=SEED, candidate=1, mmlu=None, factual=None, k=None, arms=ARMS
) -> list[Endpoint]:
    """Every run of a block. By default each arm sits at its targets (the weight is 1/2) and has
    the interpolated losses ``mmlu[arm]`` and ``factual[arm]`` (the endpoints m ∓ 1 and f ∓ 1);
    ``k``: arm → (K_lo, K_hi) to put an arm elsewhere."""
    targets = CANDIDATES[candidate]["targets"]
    step = "validation" if seed < 10400 else "confirmation"
    out = []
    for arm in arms:
        m, f = (mmlu or {}).get(arm, 1.0), (factual or {}).get(arm, 2.0)
        k_l, k_u = (k or {}).get(arm, (targets["l"], targets["u"]))
        out += pair(arm, k_l, k_u, mmlu=(m - 1, m + 1), factual=(f - 1, f + 1), seed=seed,
                    candidate=candidate, step=step)  # fmt: skip
    return out


def cohort(values, *, spread=1.0) -> list[dict]:
    """One block per entry of the lists in ``values`` (arm → MMLU losses, in seed order from the
    first validation seed), whose arm has that interpolated MMLU loss and, as factual loss, the
    same plus ``spread``. An arm not named keeps the default losses (1 and 2 pp)."""
    count = len(next(iter(values.values())))
    return [block_state(full_block(SEED + i, mmlu={a: v[i] for a, v in values.items()},
                                   factual={a: v[i] + spread for a, v in values.items()}),
                        SEED + i, 1, ARMS) for i in range(count)]  # fmt: skip


# ------------------------------------------------------------------ registered values


def test_registered_values_match_the_preregistration() -> None:
    text = (REPO_ROOT / "docs" / "study3-preregistration.md").read_text()
    flat = " ".join(text.split())  # the registration wraps its lines
    assert "ln K_hi − ln K_lo ≤ 0.60" in flat and WIDTH == 0.60
    assert "a mean MMLU loss ≤ 5 pp and a mean factual loss ≤ 10 pp" in flat
    assert THRESHOLDS_PP == {"mmlu": 5.0, "factual": 10.0}
    assert "the one-sided t upper bound at α = 0.05 per bound" in flat
    assert VALIDATION_ALPHA == 0.05
    assert "One-sided t upper bounds at α = 0.05/8 per bound" in flat
    assert CONFIRMATION_ALPHA == 0.05 / 8
    assert "K\\* = √(K_l·K_u)" in flat
    for spec in CANDIDATES.values():
        targets = spec["targets"]
        assert targets["star"] == pytest.approx(math.sqrt(targets["l"] * targets["u"]), rel=1e-12)


# ------------------------------------------------------------------ the weight


def test_the_helper_is_the_registered_weight() -> None:
    assert blocks.weight(2.0, 8.0, 4.0) == pytest.approx(0.5)


def test_the_weight_is_the_log_position_between_the_endpoints() -> None:
    # K_lo = 4.8, K_hi = 7.5: ln(6/4.8) = ln 1.25 and ln(7.5/4.8) = ln 1.5625 = 2 ln 1.25, so 1/2
    middle = support(4.8, 7.5)
    assert middle["supported"] and middle["weight"] == pytest.approx(0.5, abs=1e-12)
    # K_lo = 5, K_hi = 8: ln 1.2 / ln 1.6 = 0.1823216 / 0.4700036 = 0.3879152
    assert support(5.0, 8.0)["weight"] == pytest.approx(0.3879152, abs=1e-7)
    # candidate 2's K* = 5 between its targets 4.00 and 6.25: ln 1.25 / ln 1.5625 = 1/2 too
    second = arm_support(pair("A", 4.0, 6.25, candidate=2), 5.0)
    assert second["weight"] == pytest.approx(0.5, abs=1e-12)


def test_the_weight_is_0_and_1_at_the_endpoints() -> None:
    low = support(6.0, 7.5)  # K_lo = K*: ln K* − ln K_lo = 0
    assert low["supported"] and low["weight"] == 0.0 and low["k_lo"] == 6.0
    high = support(4.8, 6.0)  # K_hi = K*: the numerator is the denominator
    assert high["supported"] and high["weight"] == 1.0 and high["k_hi"] == 6.0


def test_the_endpoints_are_ordered_by_measured_k_not_by_rate() -> None:
    # the run at the higher rate measured the lower K
    swapped = arm_support(pair("A", 7.5, 4.8), KSTAR)
    assert swapped["supported"] and swapped["weight"] == pytest.approx(0.5, abs=1e-12)
    assert (swapped["k_lo"], swapped["k_hi"]) == (4.8, 7.5)
    assert swapped["lo"].lr == START["A"]["u"] and swapped["hi"].lr == START["A"]["l"]


def test_the_interpolated_value_is_the_weighted_mean_of_the_endpoints() -> None:
    assert interpolated(2.0, 8.0, 0.0) == 2.0 and interpolated(2.0, 8.0, 1.0) == 8.0
    assert interpolated(2.0, 8.0, 0.5) == 5.0
    assert interpolated(2.0, 8.0, 0.25) == pytest.approx(3.5)  # 0.75·2 + 0.25·8
    assert interpolated(-3.0, 5.0, 0.75) == pytest.approx(3.0)  # 0.25·(−3) + 0.75·5
    assert interpolated(1.0, 1.0, 0.3) == pytest.approx(1.0)


@pytest.mark.parametrize("weight", [-1e-12, 1.0 + 1e-12, 2.0, math.nan, math.inf])
def test_nothing_is_extrapolated(weight) -> None:
    with pytest.raises(ValueError, match="extrapolated"):
        interpolated(1.0, 2.0, weight)


# ------------------------------------------------------------------ the support's reasons


def test_a_bracket_inside_the_width_is_supported() -> None:
    state = support(4.8, 7.5)  # ln 7.5 − ln 4.8 = 0.4463 ≤ 0.60
    assert state["state"] == "supported" and state["supported"] is True and state["reason"] is None
    assert (state["k_lo"], state["k_hi"]) == (4.8, 7.5)
    assert state["failed"] == () and state["technical"] == () and state["pending"] == ()


def test_a_failed_endpoint_removes_the_support() -> None:
    stopped = ep("A", START["A"]["u"], state="scientific", reasons=("instability stop",))
    state = arm_support([pair("A", 4.8, 7.5)[0], stopped], KSTAR)
    assert state["state"] == "unsupported" and state["reason"] == "failed endpoint"
    assert state["failed"] == (stopped,) and state["weight"] is None and not state["supported"]


def test_a_non_positive_k_is_a_failed_endpoint_and_a_collapse_is_a_gross_collapse() -> None:
    lo = pair("A", 4.8, 7.5)[0]
    nonpositive = ep("A", START["A"]["u"], -0.1, state="unusable", unusable_reason="nonpositive_k")
    assert arm_support([lo, nonpositive], KSTAR)["reason"] == "failed endpoint"
    collapsed = ep("A", START["A"]["u"], 7.5, mmlu=15.1, state="unusable",
                   unusable_reason="gross_collapse")  # fmt: skip
    state = arm_support([lo, collapsed], KSTAR)
    assert state["state"] == "unsupported" and state["reason"] == "gross collapse"
    assert state["failed"] == (collapsed,)
    # an instability stop beside a collapse: the stop names the reason
    stopped = ep("A", START["A"]["l"], state="scientific")
    both = arm_support([stopped, collapsed], KSTAR)
    assert both["reason"] == "failed endpoint" and len(both["failed"]) == 2


def test_tied_endpoints_are_not_a_bracket() -> None:
    assert support(6.0, 6.0)["reason"] == "tied"  # both at K*: ln K_hi − ln K_lo = 0
    assert support(5.0, 5.0)["reason"] == "tied"  # tied before the bracket test
    assert support(0.0, 0.0)["reason"] == "tied"
    assert support(5.0, 5.0)["weight"] is None


@pytest.mark.parametrize(("k_l", "k_u"), [
    (7.0, 8.0),  # both above K* = 6: K_lo ≤ K* fails
    (4.0, 5.9),  # both below: K* ≤ K_hi fails
    (0.0, 7.5),  # K_lo = 0: 0 < K_lo fails (K* ≥ K_lo ≥ 0 would hold)
    (-0.2, 7.5),
    (4.8, math.inf),  # K_hi = ∞: K_hi < ∞ fails
])  # fmt: skip
def test_no_bracket(k_l, k_u) -> None:
    state = support(k_l, k_u)
    assert state["state"] == "unsupported" and state["reason"] == "no bracket"
    assert state["weight"] is None and (state["k_lo"], state["k_hi"]) == (k_l, k_u)


def test_a_bracket_wider_than_0_60_in_ln_k_is_too_wide() -> None:
    state = support(4.0, 8.0)  # ln 2 = 0.6931 > 0.60
    assert state["state"] == "unsupported" and state["reason"] == "too wide"
    assert state["weight"] is None
    assert support(5.0, 8.0)["supported"]  # ln 1.6 = 0.4700 ≤ 0.60


def test_the_width_is_inclusive() -> None:
    # K_hi = K_lo·e^0.60 exactly (up to the floating-point difference of two logarithms)
    on = 4.0 * math.exp(WIDTH)
    assert support(4.0, on)["supported"]
    assert support(4.0, 4.0 * math.exp(0.5999))["supported"]
    assert support(4.0, 4.0 * math.exp(0.6001))["reason"] == "too wide"
    assert support(4.0, 4.0 * math.exp(0.60 + 1e-6))["reason"] == "too wide"


def test_the_width_is_a_parameter() -> None:
    assert support(5.0, 8.0, width=0.50)["supported"]  # ln 1.6 = 0.4700
    assert support(5.0, 8.0, width=0.45)["reason"] == "too wide"


def test_the_reasons_follow_the_order_of_the_conditions() -> None:
    # a collapse or a stop comes before the bracket, a tie before the bracket, the bracket before
    # the width: K = 10 and 100 lie above K* = 6 (no bracket) and are far wider than 0.60
    assert support(10.0, 100.0)["reason"] == "no bracket"
    assert support(4.0, 8.0)["reason"] == "too wide"
    stopped = ep("A", START["A"]["u"], state="scientific")
    assert arm_support([pair("A", 6.0, 6.0)[0], stopped], KSTAR)["reason"] == "failed endpoint"


def test_k_star_is_inclusive_at_both_ends() -> None:
    assert support(6.0, 6.3)["supported"] and support(5.7, 6.0)["supported"]
    assert support(6.0 + 1e-9, 6.3)["reason"] == "no bracket"
    assert support(5.7, 6.0 - 1e-9)["reason"] == "no bracket"


# ------------------------------------------------------------------ an arm's other states


def test_a_technical_endpoint_leaves_the_arm_unresolved() -> None:
    lo, hi = pair("A", 4.8, 7.5)
    failed = ep("A", START["A"]["u"], state="technical", reasons=("no published terminal record",))
    state = arm_support([lo, failed], KSTAR)
    assert state["state"] == "unresolved" and state["reason"] == "technical"
    assert state["technical"] == (failed,) and not state["supported"] and state["weight"] is None
    # it waits, whatever its partner measured: not a failure, not a support
    assert arm_support([pair("A", 7.0, 8.0)[0], failed], KSTAR)["state"] == "unresolved"


@pytest.mark.parametrize("status", ["held", "refused"])
def test_a_held_or_refused_attempt_leaves_the_arm_unresolved(status) -> None:
    lo, hi = pair("A", 4.8, 7.5)
    odd = ep("A", START["A"]["u"], state=status, reasons=("why",))
    state = arm_support([lo, odd], KSTAR)
    assert state["state"] == "unresolved" and state["technical"] == (odd,)
    # even beside a counting attempt of its cell
    assert arm_support([lo, hi, odd], KSTAR)["state"] == "unresolved"


def test_a_failed_endpoint_beside_a_technical_one_is_unresolved_not_unsupported() -> None:
    # "Unresolved technical failures retain the technical route": the failure is listed, but the
    # arm is not counted until the technical run is diagnosed
    stopped = ep("A", START["A"]["l"], state="scientific")
    failed = ep("A", START["A"]["u"], state="technical")
    state = arm_support([stopped, failed], KSTAR)
    assert state["state"] == "unresolved"
    assert state["failed"] == (stopped,) and state["technical"] == (failed,)


def test_a_state_the_reader_does_not_define_is_not_ignored() -> None:
    lo = pair("A", 4.8, 7.5)[0]
    odd = ep("A", START["A"]["u"], 7.5, state="mystery")
    state = arm_support([lo, odd], KSTAR)
    assert state["state"] == "unresolved" and state["technical"] == (odd,)
    assert arm_support([lo, pair("A", 4.8, 7.5)[1], odd], KSTAR)["state"] == "unresolved"


def test_two_counting_attempts_of_a_cell_are_technical() -> None:
    lo, hi = pair("A", 4.8, 7.5)
    again = ep("A", START["A"]["u"], 7.4, attempt=2)
    assert arm_support([lo, hi, again], KSTAR)["state"] == "unresolved"


@pytest.mark.parametrize("broken", [
    {"K": None}, {"K": math.nan}, {"mmlu": None}, {"mmlu": math.nan},
    {"factual": None}, {"factual": math.inf},
])  # fmt: skip
def test_a_usable_run_without_its_numbers_is_technical(broken) -> None:
    good = ep("A", START["A"]["l"], 4.8)
    args = {"K": 7.5, "mmlu": 1.0, "factual": 2.0} | broken
    run = ep("A", START["A"]["u"], args["K"], mmlu=args["mmlu"], factual=args["factual"])
    assert arm_support([good, run], KSTAR)["state"] == "unresolved"


def test_a_run_that_has_not_ended_leaves_the_arm_open() -> None:
    lo = pair("A", 4.8, 7.5)[0]
    running = ep("A", START["A"]["u"], state="pending", reasons=("the launcher has not ended it",))
    state = arm_support([lo, running], KSTAR)
    assert state["state"] == "open" and state["reason"] == "pending"
    assert state["pending"] == (running,) and not state["supported"] and state["missing"] == 0


def test_a_recovered_attempt_with_no_later_one_leaves_the_arm_open() -> None:
    lo = pair("A", 4.8, 7.5)[0]
    interrupted = ep("A", START["A"]["u"], state="recovered", reasons=("pod interrupted",))
    waiting = arm_support([lo, interrupted], KSTAR)
    assert waiting["state"] == "open" and waiting["reason"] == "pending"
    assert waiting["pending"] == (interrupted,) and waiting["missing"] == 0
    rerun = ep("A", START["A"]["u"], 7.5, attempt=2)  # the next attempt decides
    state = arm_support([lo, interrupted, rerun], KSTAR)
    assert state["state"] == "supported" and len(state["attempts"]) == 3


def test_a_cell_without_an_attempt_leaves_the_arm_open() -> None:
    state = arm_support([pair("A", 4.8, 7.5)[0]], KSTAR)
    assert state["state"] == "open" and state["reason"] == "missing" and state["missing"] == 1
    nothing = arm_support([], KSTAR)
    assert nothing["state"] == "open" and nothing["missing"] == 2 and nothing["attempts"] == ()
    # a cell none of whose attempts counts is missing
    gone = ep("A", START["A"]["u"], 7.5, state="superseded", reasons=("pod",))
    assert arm_support([pair("A", 4.8, 7.5)[0], gone], KSTAR)["missing"] == 1
    assert arm_support([pair("A", 4.8, 7.5)[0], gone], KSTAR)["state"] == "open"


def test_a_failed_endpoint_beside_a_run_that_has_not_ended_waits_for_it() -> None:
    stopped = ep("A", START["A"]["l"], state="scientific")
    running = ep("A", START["A"]["u"], state="pending")
    state = arm_support([stopped, running], KSTAR)
    assert state["state"] == "open" and state["failed"] == (stopped,)
    ended = arm_support([stopped, pair("A", 4.8, 7.5)[1]], KSTAR)
    assert ended["state"] == "unsupported" and ended["reason"] == "failed endpoint"


def test_a_superseded_attempt_is_listed_and_never_counted() -> None:
    lo, hi = pair("A", 4.8, 7.5)
    old = ep("A", START["A"]["u"], 9.9, state="superseded", reasons=("case restart",))
    newer = ep("A", START["A"]["u"], 7.5, attempt=2)
    state = arm_support([old, lo, newer], KSTAR)
    assert state["state"] == "supported" and state["k_hi"] == 7.5
    assert old in state["attempts"] and state["hi"] is newer


def test_a_third_rate_leaves_the_arm_unresolved() -> None:
    lo, hi = pair("A", 4.8, 7.5)
    third = ep("A", 3.33e-5, 6.0)
    state = arm_support([lo, hi, third], KSTAR)
    assert state["state"] == "unresolved" and set(state["technical"]) == {lo, hi, third}
    assert state["weight"] is None


def test_rates_group_as_numbers() -> None:
    lo = pair("A", 4.8, 7.5)[0]
    spelled = ep("A", START["A"]["u"] * (1 + 1e-12), 7.5, attempt=2)  # the same rate
    old = ep("A", START["A"]["u"], state="recovered", attempt=1)  # its cell's earlier attempt
    assert arm_support([lo, old, spelled], KSTAR)["state"] == "supported"


def test_the_inputs_of_an_arm_are_checked() -> None:
    lo, hi = pair("A", 4.8, 7.5)
    with pytest.raises(ValueError, match="one arm and one seed"):
        arm_support([lo, pair("F", 4.8, 7.5)[1]], KSTAR)
    with pytest.raises(ValueError, match="one arm and one seed"):
        arm_support([lo, pair("A", 4.8, 7.5, seed=SEED + 1)[1]], KSTAR)
    with pytest.raises(ValueError, match="no rate"):
        arm_support([lo, bare("x", arm="A", seed=SEED, state="technical")], KSTAR)
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="kstar"):
            arm_support([lo, hi], bad)
        with pytest.raises(ValueError, match="width"):
            arm_support([lo, hi], KSTAR, width=bad)


# ------------------------------------------------------------------ a block


def test_a_block_interpolates_the_losses_of_every_supported_arm() -> None:
    # A at K 4.8 / 7.5 (w = 1/2): MMLU 2 and 6 -> 4, factual 4 and 8 -> 6
    # F at K 5 / 8 (w = 0.3879152): MMLU 1 and 3 -> 1 + 2w = 1.7758304, factual 0 and 10 -> 3.879152
    # P at K 6 / 7.5 (w = 0): MMLU 0 and 9 -> 0, factual 3 and 1 -> 3
    runs = (pair("A", 4.8, 7.5, mmlu=(2, 6), factual=(4, 8))
            + pair("F", 5.0, 8.0, mmlu=(1, 3), factual=(0, 10))
            + pair("P", 6.0, 7.5, mmlu=(0, 9), factual=(3, 1)))  # fmt: skip
    block = block_state(runs, SEED, 1, ARMS)
    assert block["step"] == "validation" and block["kstar"] == 6.0 and block["seed"] == SEED
    assert block["state"] == "resolved" and block["blockers"] == []
    a, f, p = (block["arms"][arm] for arm in ARMS)
    assert a["mmlu_loss_pp"] == pytest.approx(4.0) and a["factual_loss_pp"] == pytest.approx(6.0)
    assert f["mmlu_loss_pp"] == pytest.approx(1.7758304, abs=1e-6)
    assert f["factual_loss_pp"] == pytest.approx(3.879152, abs=1e-5)
    assert p["mmlu_loss_pp"] == 0.0 and p["factual_loss_pp"] == 3.0
    assert all(block["arms"][arm]["state"] == "supported" for arm in ARMS)
    assert block["AF"]["state"] == "supported" and block["AF"]["supported"] is True
    assert block["PF"]["state"] == "supported" and block["PF"]["reasons"] == {}
    assert block["AF"]["arms"] == ("A", "F") and block["PF"]["arms"] == ("P", "F")


def test_an_unsupported_arm_has_no_interpolated_loss_and_removes_its_comparisons() -> None:
    runs = full_block(k={"A": (4.0, 8.0)})  # A: too wide
    block = block_state(runs, SEED, 1, ARMS)
    assert block["arms"]["A"]["reason"] == "too wide"
    assert block["arms"]["A"]["mmlu_loss_pp"] is None
    assert block["arms"]["A"]["factual_loss_pp"] is None
    assert block["AF"]["state"] == "unsupported" and block["AF"]["reasons"] == {"A": "too wide"}
    assert block["PF"]["state"] == "supported"  # A–F and P–F are decided by their own arms
    assert block["state"] == "resolved"


def test_f_unsupported_removes_both_comparisons() -> None:
    block = block_state(full_block(k={"F": (7.0, 8.0)}), SEED, 1, ARMS)
    assert block["AF"]["reasons"] == {"F": "no bracket"}
    assert block["PF"]["reasons"] == {"F": "no bracket"}
    assert not block["AF"]["supported"] and not block["PF"]["supported"]


def test_p_unsupported_leaves_a_f_alone() -> None:
    block = block_state(full_block(k={"P": (6.5, 7.5)}), SEED, 1, ARMS)
    assert block["AF"]["supported"] and not block["PF"]["supported"]
    assert block["PF"]["reasons"] == {"P": "no bracket"}


def test_a_block_without_p_has_no_p_f_comparison() -> None:
    block = block_state(full_block(arms=("A", "F")), SEED, 1, ("A", "F"))
    assert block["PF"] is None and set(block["arms"]) == {"A", "F"}
    assert block["AF"]["supported"] and block["state"] == "resolved"
    # P's runs, if any are in the records, are not part of a block that does not list P
    stray = ep("P", START["P"]["l"], state="technical")
    quiet = block_state([*full_block(arms=("A", "F")), stray], SEED, 1, ("A", "F"))
    assert quiet["state"] == "resolved" and quiet["AF"]["supported"] and quiet["blockers"] == []


def test_a_technical_arm_leaves_its_comparisons_unresolved_and_the_others_alone() -> None:
    technical = ep("A", START["A"]["u"], state="technical", reasons=("no published terminal",))
    runs = [e for e in full_block() if e.run_id != ep("A", START["A"]["u"], 1.0).run_id]
    block = block_state([*runs, technical], SEED, 1, ARMS)
    assert block["arms"]["A"]["state"] == "unresolved"
    assert block["AF"]["state"] == "unresolved" and block["AF"]["reasons"] == {"A": "technical"}
    assert block["PF"]["state"] == "supported"  # P and F are resolved
    assert block["state"] == "unresolved"


def test_a_pending_run_leaves_the_block_open() -> None:
    running = ep("F", START["F"]["u"], state="pending")
    runs = [e for e in full_block() if e.run_id != ep("F", START["F"]["u"], 1.0).run_id]
    block = block_state([*runs, running], SEED, 1, ARMS)
    assert block["AF"]["state"] == "open" and block["PF"]["state"] == "open"
    assert block["AF"]["reasons"] == {"F": "pending"} and block["state"] == "open"
    # a block whose runs are not all issued is open too
    assert block_state(full_block()[:5], SEED, 1, ARMS)["state"] == "open"
    assert block_state([], SEED, 1, ARMS)["state"] == "open"


def test_unresolved_comes_before_open_before_unsupported() -> None:
    # A: too wide (decided). F: technical. P: pending
    runs = (pair("A", 4.0, 8.0)
            + [ep("F", START["F"]["l"], 4.8), ep("F", START["F"]["u"], state="technical")]
            + [ep("P", START["P"]["l"], 4.8),
               ep("P", START["P"]["u"], state="pending")])  # fmt: skip
    block = block_state(runs, SEED, 1, ARMS)
    assert block["AF"]["state"] == "unresolved" and block["PF"]["state"] == "unresolved"
    # F's run ends well: A unsupported, P open
    ended = runs[:3] + [ep("F", START["F"]["u"], 7.5)] + runs[4:]
    block = block_state(ended, SEED, 1, ARMS)
    assert block["AF"]["state"] == "unsupported" and block["PF"]["state"] == "open"
    assert block["state"] == "open"
    # a decided arm beside a partner that has not ended: the comparison waits for it
    waiting = [*pair("A", 4.0, 8.0), *pair("F", 4.8, 7.5)[:1],
               ep("F", START["F"]["u"], state="pending")]  # fmt: skip
    block = block_state(waiting, SEED, 1, ("A", "F"))
    assert block["arms"]["A"]["state"] == "unsupported" and block["AF"]["state"] == "open"


def test_a_refused_attempt_of_the_block_makes_it_unresolved() -> None:
    refused = ep("P", START["P"]["l"], state="refused", attempt=2, reasons=("two attempts count",))
    block = block_state([*full_block(), refused], SEED, 1, ARMS)
    assert block["arms"]["P"]["state"] == "unresolved" and block["PF"]["state"] == "unresolved"
    assert block["AF"]["state"] == "supported"


@pytest.mark.parametrize(("unbound", "why"), [
    (bare("no-start"), "no candidate, step or seed"),  # a run directory with no start
    (bare("no-seed", candidate=1, step="validation", state="technical"),
     "no candidate, step or seed"),  # its config can no longer be bound
    (bare("no-step", candidate=1, seed=SEED), "no candidate, step or seed"),
    (bare("no-candidate", step="validation", seed=SEED, arm="A", rate=1e-5),
     "no candidate, step or seed"),
    (bare("no-arm", candidate=1, step="validation", seed=SEED, state="held"),
     "no arm or rate"),
    (bare("no-rate", candidate=1, step="validation", seed=SEED, arm="F", state="technical"),
     "no arm or rate"),
    (bare("a-round", candidate=1, step="validation", seed=SEED, arm="F", rate=1.3e-5, round_=1,
          state="usable"), "under the key c1/validation/r1"),
])  # fmt: skip
def test_an_attempt_that_may_be_of_the_block_makes_every_comparison_unresolved(
    unbound, why
) -> None:
    block = block_state([*full_block(), unbound], SEED, 1, ARMS)
    assert why in block["blockers"][0] and "no cell" in block["blockers"][0]
    assert block["state"] == "unresolved" and block["AF"]["state"] == "unresolved"
    assert block["PF"]["state"] == "unresolved"
    assert [b.split(":")[0] for b in block["blockers"]] == [unbound.run_id]
    assert unbound.state in block["blockers"][0]
    assert all(block["arms"][arm]["state"] == "supported" for arm in ARMS)  # their own runs


def test_what_cannot_be_of_the_block_is_left_alone() -> None:
    elsewhere = [
        bare("other-candidate", candidate=2, step="validation", seed=SEED),
        bare("other-seed", candidate=1, step="validation", seed=SEED + 1, arm="A", rate=1e-5),
        bare("search", candidate=1, step="search", seed=None, state="technical"),
        bare("other-block", candidate=None, step="confirmation", seed=None),
        ep("A", START["A"]["l"], state="technical", seed=SEED + 1),
        bare("superseded", state="superseded", reasons=("r",)),
        bare("recovered", state="recovered", reasons=("r",)),
        bare("closed-arm", candidate=1, step="validation", seed=None, arm="P"),
    ]  # fmt: skip
    block = block_state([*full_block(arms=("A", "F")), *elsewhere], SEED, 1, ("A", "F"))
    assert block["state"] == "resolved" and block["blockers"] == []
    assert block["AF"]["supported"]


def test_the_step_comes_from_the_seed() -> None:
    assert block_state([], 10339, 1, ARMS)["step"] == "validation"
    assert block_state([], 10340, 2, ARMS)["step"] == "validation"
    assert block_state([], 10400, 1, ARMS)["step"] == "confirmation"
    assert block_state([], 10599, 2, ARMS)["step"] == "confirmation"
    for seed, candidate in ((10340, 1), (10339, 2), (10380, 2), (10399, 1), (10600, 1),
                            (10100, 1), (10000, 1)):  # fmt: skip
        with pytest.raises(ValueError, match="no validation or confirmation seed"):
            block_state([], seed, candidate, ARMS)


def test_a_confirmation_block_of_candidate_2_uses_its_k_star() -> None:
    # K* = 5; K 4.0 / 6.25 gives the weight 1/2: MMLU 0 and 2 -> 1
    runs = []
    for arm in ARMS:
        runs += pair(arm, 4.0, 6.25, mmlu=(0, 2), seed=10450, candidate=2, step="confirmation")
    block = block_state(runs, 10450, 2, ARMS)
    assert block["kstar"] == 5.0 and block["step"] == "confirmation"
    assert block["AF"]["supported"] and block["arms"]["A"]["mmlu_loss_pp"] == pytest.approx(1.0)
    # candidate 1's block of the same seed holds none of these runs
    assert block_state(runs, 10450, 1, ARMS)["state"] == "open"
    # and under candidate 1's K* = 6 the same K would not bracket it
    assert arm_support(pair("A", 4.0, 5.5), 6.0)["reason"] == "no bracket"


def test_the_inputs_of_a_block_are_checked() -> None:
    for arms in ((), ("A", "A"), ("A", "X"), ("a",)):
        with pytest.raises(ValueError, match="distinct arms"):
            block_state([], SEED, 1, arms)
    with pytest.raises(ValueError, match="no candidate"):
        block_state([], SEED, 3, ARMS)


def test_the_block_does_not_depend_on_the_order_of_its_records() -> None:
    runs = full_block(mmlu={"A": 3.0}, k={"F": (5.0, 8.0)})
    forward = block_state(runs, SEED, 1, ARMS)
    backward = block_state(runs[::-1], SEED, 1, ARMS)
    assert forward == backward


# ------------------------------------------------------------------ the t bound


def test_the_bound_of_three_values_matches_the_closed_form_of_t_with_2_degrees() -> None:
    # df = 2: t(p) = (2p − 1)/√(2p(1 − p)); p = 0.95: 0.9/√0.095 = 2.9199856
    # values 1, 2, 3: mean 2, s = 1, s/√3 = 0.5773503, bound = 2 + 2.9199856·0.5773503 = 3.685854
    expected = 2.0 + (0.9 / math.sqrt(0.095)) / math.sqrt(3)
    assert utility_bound([1.0, 2.0, 3.0], 0.05) == pytest.approx(expected, rel=1e-12)
    assert utility_bound([1.0, 2.0, 3.0], 0.05) == pytest.approx(3.685854, abs=1e-6)


def test_the_bound_of_four_values_uses_3_degrees_of_freedom() -> None:
    # values 1..4: mean 2.5, s = √(5/3) = 1.2909944, s/√4 = 0.6454972; t(0.95, 3) = 2.3533634 from
    # the table; bound = 2.5 + 2.3533634·0.6454972 = 4.0190896
    assert utility_bound([1.0, 2.0, 3.0, 4.0], 0.05) == pytest.approx(4.0190896, abs=1e-6)
    assert utility_bound([4.0, 1.0, 3.0, 2.0], 0.05) == pytest.approx(4.0190896, abs=1e-6)


def test_the_bound_of_two_values_is_the_cauchy_quantile() -> None:
    # df = 1: t(1 − α) = cot(πα); values 1 and 3: mean 2, s = √2, s/√2 = 1
    assert utility_bound([1.0, 3.0], 0.05) == pytest.approx(
        2.0 + 1.0 / math.tan(math.pi * 0.05), rel=1e-12
    )  # 2 + 6.3137515
    assert utility_bound([1.0, 3.0], 0.05) == pytest.approx(8.3137515, abs=1e-6)
    # confirmation's α = 0.05/8: cot(π·0.00625) = 50.9230366
    assert utility_bound([1.0, 3.0], CONFIRMATION_ALPHA) == pytest.approx(52.9230366, abs=1e-6)


def test_a_smaller_alpha_gives_a_larger_bound_and_identical_values_give_their_mean() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 9.0]
    assert utility_bound(values, 0.05 / 8) > utility_bound(values, 0.05) > sum(values) / 5
    assert utility_bound([5.0, 5.0, 5.0], 0.05) == 5.0  # s = 0


def test_the_bound_checks_its_inputs() -> None:
    with pytest.raises(ValueError, match="two values"):
        utility_bound([1.0], 0.05)
    with pytest.raises(ValueError, match="two values"):
        utility_bound([], 0.05)
    for bad in ([1.0, math.nan], [1.0, math.inf], [math.nan, math.nan]):
        with pytest.raises(ValueError, match="finite"):
            utility_bound(bad, 0.05)
    for alpha in (0.0, 1.0, -0.1, 1.5, math.nan):
        with pytest.raises(ValueError, match="alpha"):
            utility_bound([1.0, 2.0], alpha)


# ------------------------------------------------------------------ the gate


def test_the_gate_reports_each_bound_with_n_mean_and_upper_bound() -> None:
    # MMLU: A 1, 2, 3 (mean 2, bound 2 + 2.9199856/√3 = 3.685854); F 0, 1, 2 (mean 1, 2.685854)
    # factual = MMLU + 4: A 5, 6, 7 (bound 6 + 1.685854 = 7.685854), F 4, 5, 6 (6.685854)
    values = {"A": [1.0, 2.0, 3.0], "F": [0.0, 1.0, 2.0], "P": [0.0, 0.0, 0.0]}
    gate = utility_gate(cohort(values, spread=4.0), "AF", VALIDATION_ALPHA)
    assert gate["computed"] and gate["passes"] is True and gate["n"] == 3
    assert gate["excluded"] == () and gate["waiting"] == () and gate["alpha"] == 0.05
    assert set(gate["bounds"]) == {"A", "F"}
    t = 0.9 / math.sqrt(0.095) / math.sqrt(3)
    expected = {("A", "mmlu"): (2.0, 2.0 + t), ("A", "factual"): (6.0, 6.0 + t),
                ("F", "mmlu"): (1.0, 1.0 + t), ("F", "factual"): (5.0, 5.0 + t)}  # fmt: skip
    for (arm, metric), (mean, upper) in expected.items():
        bound = gate["bounds"][arm][metric]
        assert bound["n"] == 3 and bound["mean"] == pytest.approx(mean, abs=1e-9)
        assert bound["upper"] == pytest.approx(upper, abs=1e-9) and bound["passes"] is True
        assert bound["threshold"] == THRESHOLDS_PP[metric]


def test_the_gate_compares_the_upper_bound_not_the_mean() -> None:
    # MMLU 3, 4, 5: mean 4 ≤ 5 but the bound 4 + 1.685854 = 5.685854 > 5
    values = {"A": [3.0, 4.0, 5.0], "F": [0.0, 0.0, 0.0]}
    gate = utility_gate(cohort(values), "AF", VALIDATION_ALPHA)
    assert gate["bounds"]["A"]["mmlu"]["mean"] == pytest.approx(4.0)
    assert gate["bounds"]["A"]["mmlu"]["passes"] is False and gate["passes"] is False
    assert gate["bounds"]["A"]["factual"]["passes"] and gate["bounds"]["F"]["mmlu"]["passes"]


def test_the_mmlu_threshold_is_5_pp_on_either_side_of_the_bound() -> None:
    # MMLU m − 1, m, m + 1: bound m + 1.6858545; it is 5 at m = 3.3141455
    below = cohort({"A": [2.3141, 3.3141, 4.3141], "F": [0.0] * 3})  # bound 4.99995
    above = cohort({"A": [2.3142, 3.3142, 4.3142], "F": [0.0] * 3})  # bound 5.00005
    assert utility_gate(below, "AF", 0.05)["bounds"]["A"]["mmlu"]["passes"] is True
    assert utility_gate(above, "AF", 0.05)["bounds"]["A"]["mmlu"]["passes"] is False
    assert utility_gate(below, "AF", 0.05)["passes"] is True
    assert utility_gate(above, "AF", 0.05)["passes"] is False


def test_the_factual_threshold_is_10_pp_on_either_side_of_the_bound() -> None:
    # factual m − 1, m, m + 1: bound m + 1.6858545; it is 10 at m = 8.3141455
    def run(m) -> dict:  # F's factual losses over three blocks: m − 1, m, m + 1
        states = [block_state(full_block(SEED + i, factual={"F": m - 1 + i}), SEED + i, 1, ARMS)
                  for i in range(3)]  # fmt: skip
        return utility_gate(states, "AF", 0.05)

    assert run(8.3141)["bounds"]["F"]["factual"]["passes"] is True  # 8.3141 + 1.6858545 = 9.99995
    assert run(8.3142)["bounds"]["F"]["factual"]["passes"] is False  # 10.00005
    assert run(8.3141)["passes"] is True and run(8.3142)["passes"] is False


def test_a_mean_exactly_at_a_threshold_passes_and_a_hair_above_fails() -> None:
    # identical losses: s = 0, the bound is the mean
    at = cohort({"A": [5.0] * 3, "F": [5.0] * 3}, spread=5.0)  # MMLU 5, factual 10
    gate = utility_gate(at, "AF", 0.05)
    assert gate["passes"] is True
    assert [gate["bounds"]["A"][m]["upper"] for m in ("mmlu", "factual")] == pytest.approx([5, 10])
    over = cohort({"A": [5.001] * 3, "F": [0.0] * 3})
    assert utility_gate(over, "AF", 0.05)["bounds"]["A"]["mmlu"]["passes"] is False
    over = cohort({"A": [0.0] * 3, "F": [5.0] * 3}, spread=5.001)
    assert utility_gate(over, "AF", 0.05)["bounds"]["F"]["factual"]["passes"] is False
    assert utility_gate(over, "AF", 0.05)["bounds"]["F"]["mmlu"]["passes"] is True


@pytest.mark.parametrize(("arm", "metric"), [("A", "mmlu"), ("A", "factual"),
                                             ("F", "mmlu"), ("F", "factual")])  # fmt: skip
def test_every_one_of_the_four_bounds_can_fail_the_gate(arm, metric) -> None:
    bad = {"mmlu": 5.5, "factual": 10.5}[metric]  # a constant loss over its threshold: s = 0
    states = [block_state(full_block(SEED + i, mmlu={arm: bad} if metric == "mmlu" else None,
                                     factual={arm: bad} if metric == "factual" else None),
                          SEED + i, 1, ARMS) for i in range(3)]  # fmt: skip
    gate = utility_gate(states, "AF", 0.05)  # the other bounds sit at 1 and 2 pp
    assert gate["passes"] is False
    failing = [(a, m) for a, per in gate["bounds"].items() for m, b in per.items()
               if not b["passes"]]  # fmt: skip
    assert failing == [(arm, metric)]


def test_the_p_f_gate_reads_p_and_f_and_leaves_out_what_is_unsupported_for_it() -> None:
    # block 3: A too wide (A–F unsupported there), P and F fine: P–F's cohort is all 3 blocks
    runs = [full_block(SEED + i, mmlu={"P": 1.0 + i, "F": 0.0 + i}) for i in range(2)]
    runs.append(full_block(SEED + 2, mmlu={"P": 3.0, "F": 2.0}, k={"A": (4.0, 8.0)}))
    states = [block_state(r, SEED + i, 1, ARMS) for i, r in enumerate(runs)]
    pf = utility_gate(states, "PF", 0.05)
    assert set(pf["bounds"]) == {"P", "F"} and pf["n"] == 3 and pf["excluded"] == ()
    assert pf["bounds"]["P"]["mmlu"]["mean"] == pytest.approx(2.0)  # 1, 2, 3
    assert pf["bounds"]["F"]["mmlu"]["mean"] == pytest.approx(1.0)  # 0, 1, 2
    af = utility_gate(states, "AF", 0.05)
    assert af["n"] == 2 and af["excluded"] == (SEED + 2,) and set(af["bounds"]) == {"A", "F"}
    # A–F's cohort: F's MMLU over blocks 1 and 2 only: 0 and 1 (df = 1, s = 1/√2)
    assert af["bounds"]["F"]["mmlu"]["mean"] == pytest.approx(0.5)
    upper = 0.5 + (1.0 / math.tan(math.pi * 0.05)) * (1 / math.sqrt(2)) / math.sqrt(2)
    assert af["bounds"]["F"]["mmlu"]["upper"] == pytest.approx(upper, rel=1e-12)  # 0.5 + 3.1568757


def test_the_gate_waits_for_every_block_of_the_cohort() -> None:
    states = cohort({"A": [1.0, 2.0, 3.0], "F": [0.0, 1.0, 2.0], "P": [0.0, 0.0, 0.0]})
    running = [
        e
        for e in full_block(SEED + 3)
        if e.run_id != ep("F", START["F"]["u"], 1.0, seed=SEED + 3).run_id
    ]
    running.append(ep("F", START["F"]["u"], state="pending", seed=SEED + 3))
    technical = full_block(SEED + 4)[:-1] + [
        ep("P", START["P"]["u"], state="technical", seed=SEED + 4)
    ]
    states += [block_state(running, SEED + 3, 1, ARMS), block_state(technical, SEED + 4, 1, ARMS)]
    gate = utility_gate(states, "AF", 0.05)
    assert gate["computed"] is False and gate["passes"] is None and gate["bounds"] == {}
    # P's technical run in block 4 does not hold A–F
    assert gate["waiting"] == ({"seed": SEED + 3, "state": "open"},)
    pf = utility_gate(states, "PF", 0.05)
    assert pf["computed"] is False and pf["waiting"] == (
        {"seed": SEED + 3, "state": "open"},
        {"seed": SEED + 4, "state": "unresolved"},
    )
    # once the runs have ended the gate is computed
    assert utility_gate(states[:3], "AF", 0.05)["computed"] is True


def test_an_unresolved_record_of_the_block_holds_the_gate() -> None:
    states = cohort({"A": [1.0, 2.0, 3.0], "F": [0.0, 1.0, 2.0], "P": [0.0, 0.0, 0.0]})
    states.append(block_state([*full_block(SEED + 3), bare("lost")], SEED + 3, 1, ARMS))
    gate = utility_gate(states, "AF", 0.05)
    assert gate["computed"] is False and gate["waiting"][0]["state"] == "unresolved"


def test_a_cohort_of_fewer_than_two_supported_blocks_has_no_bound() -> None:
    one = cohort({"A": [1.0], "F": [1.0], "P": [1.0]})
    gate = utility_gate(one, "AF", 0.05)
    assert gate["computed"] is False and gate["passes"] is False and gate["n"] == 1
    assert "no t bound" in gate["reason"]
    empty = utility_gate([], "AF", 0.05)
    assert empty["n"] == 0 and empty["passes"] is False and empty["bounds"] == {}
    only_unsupported = [block_state(full_block(k={"A": (4.0, 8.0)}), SEED, 1, ARMS)]
    assert utility_gate(only_unsupported, "AF", 0.05)["excluded"] == (SEED,)


def test_the_gate_uses_the_alpha_it_is_given() -> None:
    states = cohort({"A": [1.0, 2.0, 3.0], "F": [0.0, 1.0, 2.0], "P": [0.0, 0.0, 0.0]})
    loose = utility_gate(states, "AF", VALIDATION_ALPHA)["bounds"]["A"]["mmlu"]["upper"]
    tight = utility_gate(states, "AF", CONFIRMATION_ALPHA)["bounds"]["A"]["mmlu"]["upper"]
    # df = 2: t = (2p − 1)/√(2p(1 − p)), p = 1 − 0.05/8 = 0.99375
    p = 1 - CONFIRMATION_ALPHA
    expected = 2.0 + (2 * p - 1) / math.sqrt(2 * p * (1 - p)) / math.sqrt(3)
    assert tight == pytest.approx(expected, rel=1e-12) and tight > loose
    assert utility_gate(states, "AF", CONFIRMATION_ALPHA)["alpha"] == CONFIRMATION_ALPHA


def test_the_gate_checks_its_inputs() -> None:
    states = cohort({"A": [1.0, 2.0], "F": [0.0, 1.0]})
    with pytest.raises(ValueError, match="no comparison"):
        utility_gate(states, "AP", 0.05)
    with pytest.raises(ValueError, match="distinct blocks of one candidate"):
        utility_gate([states[0], states[0]], "AF", 0.05)
    other = block_state(full_block(10340, candidate=2), 10340, 2, ARMS)
    with pytest.raises(ValueError, match="distinct blocks of one candidate"):
        utility_gate([states[0], other], "AF", 0.05)
    two_arms = block_state(full_block(SEED + 5, arms=("A", "F")), SEED + 5, 1, ("A", "F"))
    with pytest.raises(ValueError, match="does not carry the comparison PF"):
        utility_gate([*states, two_arms], "PF", 0.05)
    with pytest.raises(ValueError, match="alpha"):
        utility_gate(states, "AF", 0.0)


# ------------------------------------------------------------------ pure functions


def test_nothing_is_read_from_a_file(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a file was opened")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(Path, "open", refuse)
    monkeypatch.setattr(Path, "read_text", refuse)
    monkeypatch.setattr(Path, "iterdir", refuse)
    states = cohort({"A": [1.0, 2.0, 3.0], "F": [0.0, 1.0, 2.0], "P": [0.0, 0.0, 0.0]})
    assert utility_gate(states, "AF", 0.05)["passes"] is True
    assert blocks.utility_bound([1.0, 2.0], 0.05) > 1.5


def test_the_module_imports_no_gpu_path() -> None:
    code = ("import sys; import zero_field_refusal.analysis.study3_blocks; "
            "print([m for m in ('torch', 'transformers', 'zero_field_refusal.schedule', "
            "'zero_field_refusal.train', 'zero_field_refusal.launch', "
            "'zero_field_refusal.qualification', 'zero_field_refusal.analysis.stage_a') "
            "if m in sys.modules])")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
