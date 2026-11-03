"""Amendment 10 (docs/study3-preregistration.md): numerical eligibility after candidate 1's
qualification. A reference beyond the bound and a declared instability stop keep their records
(``failing``) and no longer reject a rate by themselves; technical reasons keep their route; a rate
with a stopped numerical run is not repaired.

The first tests read the real records under ``runs/`` (outside git: skipped where they are not);
the others pin the rule's edges on minimal verdicts."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import qualification_verdicts as verdicts
from zero_field_refusal.analysis.study3_progress import _condition, advance, read_records
from zero_field_refusal.analysis.study3_search import repair_state
from zero_field_refusal.qualification_constants import (
    ACCEPTED,
    FAILING,
    INCOMPLETE,
    NOT_QUALIFIED,
    QUALIFIED,
    SCIENTIFIC,
    TECHNICAL,
)

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
REAL = (RUNS / "qualification" / "c1" / "numerical").is_dir()
real = pytest.mark.skipif(not REAL, reason="the real records under runs/ are not here")

# candidate 1's qualification as recorded (docs/reports/study3-numerical-c1.md)
STATES = {"adamw-1.45e-05": QUALIFIED, "adamw-2.11e-05": QUALIFIED,
          "adam_frozen-1.3e-05": FAILING, "adam_frozen-1.83e-05": FAILING,
          "adamw_prewarmed-1.3e-05": QUALIFIED, "adamw_prewarmed-1.85e-05": FAILING}  # fmt: skip
VERDICT_SHA256 = "e6bf315e7b38c6b6d9073716e38ef957396703244642f13a7a4d5df42ec31b42"
SELECTED = {"A": {"l": 1.45e-05, "u": 2.11e-05}, "F": {"l": 1.3e-05, "u": 1.83e-05},
            "P": {"l": 1.3e-05, "u": 1.85e-05}}  # fmt: skip


@pytest.fixture(scope="module")
def records():
    expected = code_identity.gpu_code_identity(repo=REPO)["sha256"]
    return read_records(RUNS, repo=REPO, expected_code=expected), expected


@real
def test_the_real_verdicts_stay_as_recorded(records) -> None:
    """Every rate's state is the recorded one, the cache agrees and its bytes are unchanged; each
    of the six rates is eligible."""
    found, _ = records
    reading = found.qualification[1]
    assert reading["problems"] == []
    assert reading["states"] == STATES
    assert reading["cache"]["agrees"] is True
    cache = RUNS / "qualification" / "c1" / "verdict.json"
    assert hashlib.sha256(cache.read_bytes()).hexdigest() == VERDICT_SHA256
    assert {k: r["eligibility"] for k, r in reading["rates"].items()} == dict.fromkeys(
        STATES, QUALIFIED)  # fmt: skip
    for key, state in STATES.items():
        condition = _condition(reading["rates"][key], reading)[0]
        assert condition == ("qualified" if state == QUALIFIED else "admitted")


@real
def test_the_real_route_goes_to_candidate_1s_validation(records) -> None:
    """The route admits F's two rates and P's upper one, closes nothing, repairs nothing and
    issues the validation of candidate 1 at its six selected rates, A–F and P–F both active."""
    found, expected = records
    state = advance(found, expected_code=expected)
    assert state["problems"] == [] and state["outcome"] is None
    assert (state["stage"], state["candidate"]) == ("validation", 1)
    assert state["active"] == {"arms": ["A", "F", "P"], "comparisons": ["A-F", "P-F"]}
    assert state["rates"] == SELECTED
    assert [(e["candidate"], e["stage"], e["event"], e["arms"]) for e in state["route"]] == [
        (1, "numerical", "admitted", ["F"]), (1, "numerical", "admitted", ["P"])]  # fmt: skip
    assert "adam_frozen-1.3e-05: failing" in state["route"][0]["why"]
    assert "adam_frozen-1.83e-05: failing" in state["route"][0]["why"]
    assert "adamw_prewarmed-1.85e-05: failing" in state["route"][1]["why"]
    step = state["step"]
    assert (step["kind"], step["campaign"], step["candidate"]) == ("campaign", "validation", 1)
    cells = {(arm, lr) for arm, lr, _ in step["cells"]}
    assert cells == {(arm, lr) for arm, pair in SELECTED.items() for lr in pair.values()}
    assert len(step["cells"]) == 6 * 28


@real
def test_the_real_repair_has_no_k_for_the_stopped_rates(records) -> None:
    """F 1.83 and P 1.85 have a stopped run on seed 10201: no K, no repair. The other four have
    their seven values and miss by at most 0.05: none is repaired."""
    found, _ = records
    reading = found.qualification[1]
    eligible = {arm: {lr: next(r["eligibility"] for r in reading["rates"].values()
                               if r["arm"] == arm and math.isclose(r["lr"], lr))
                      for lr in pair.values()} for arm, pair in SELECTED.items()}  # fmt: skip
    state = repair_state(found.endpoints, 1, SELECTED, eligible, problems=())
    assert state["next"] == "none" and state["repairs"] == {}
    no_k = {(arm, r["rate"]) for arm, a in state["arms"].items() for r in a["rates"].values()
            if r["no_k"]}  # fmt: skip
    assert no_k == {("F", 1.83e-05), ("P", 1.85e-05)}
    for a in state["arms"].values():
        assert a["status"] == "none"
        for r in a["rates"].values():
            assert r["action"] == "none"
            assert r["no_k"] or (r["miss"] <= 0.05 and len(r["numerical"]["ln_k"]) == 3)


# ------------------------------------------------------------------ the rule's edges


def case(kind: str | None, technical: list[str], *, seed: int = 10200) -> dict:
    state = {SCIENTIFIC: FAILING, TECHNICAL: NOT_QUALIFIED, None: QUALIFIED}[kind]
    return {"case": {"seed": seed}, "state": state, "kind": kind, "technical_reasons": technical,
            "complete": kind is None, "launcher": {"checkpoint": {"sha256": "c" * 64}}}  # fmt: skip


SEEDS = (10200, 10201, 10202)


def eligibility(cases, *, checked=None, stale=False, not_counted=(), unlocated=()) -> str:
    return verdicts._eligibility(cases, list(not_counted), checked, seeds=SEEDS,
                                 unlocated=list(unlocated), stale_control=stale)  # fmt: skip


def kept(why: str = "other GPU code", technical=(), seed: int = 10200) -> dict:
    """A case of this rate not counted, scientific: one an engineering cycle keeps whole."""
    return {"kind": SCIENTIFIC, "why": why, "seed": seed, "technical_reasons": list(technical)}


def test_a_stop_or_a_reference_alone_is_admitted() -> None:
    cases = [case(SCIENTIFIC, []), case(None, [], seed=10201), case(None, [], seed=10202)]
    assert eligibility(cases) == QUALIFIED
    assert eligibility(cases, checked={"state": FAILING, "kind": SCIENTIFIC}) == QUALIFIED
    assert eligibility(cases, checked={"state": ACCEPTED, "kind": None}) == QUALIFIED


def test_a_scientific_case_a_cycle_keeps_stands_for_its_seed() -> None:
    """Not counted for its other GPU code alone, with no technical reason, it stands for its seed;
    for another reason, with a technical reason, or beside a counted case of its seed, not."""
    others = [case(None, [], seed=10201), case(None, [], seed=10202)]
    assert eligibility(others, not_counted=[kept()]) == QUALIFIED
    assert eligibility(others, not_counted=[kept("other settings")]) == NOT_QUALIFIED
    assert eligibility(others, not_counted=[kept(technical=["x"])]) == NOT_QUALIFIED
    assert eligibility([case(None, []), *others], not_counted=[kept()]) == INCOMPLETE


def test_technical_reasons_keep_their_route() -> None:
    """A technical reason beside a scientific one, a technical case, a record not counted, a
    control neither accepted nor stopped, or this rate's stopped control under other code: not
    qualified. A missing seed: incomplete."""
    others = [case(None, [], seed=10201), case(None, [], seed=10202)]
    beside = case(SCIENTIFIC, ["baseline~repeat: D₂₅ 0.1 beyond the bound 0.06"])
    assert eligibility([beside, *others]) == NOT_QUALIFIED
    assert eligibility([case(SCIENTIFIC, []), case(TECHNICAL, ["x"], seed=10201),
                        others[1]]) == NOT_QUALIFIED  # fmt: skip
    assert (
        eligibility(
            [case(SCIENTIFIC, []), *others],
            not_counted=[{"kind": TECHNICAL, "why": "other GPU code"}],
        )
        == NOT_QUALIFIED
    )
    assert (
        eligibility(
            [case(SCIENTIFIC, []), *others], checked={"state": NOT_QUALIFIED, "kind": TECHNICAL}
        )
        == NOT_QUALIFIED
    )
    assert (
        eligibility(
            [case(SCIENTIFIC, []), *others],
            checked={"state": FAILING, "kind": SCIENTIFIC},
            stale=True,
        )
        == NOT_QUALIFIED
    )
    assert eligibility([case(SCIENTIFIC, []), others[0]]) == INCOMPLETE
    assert eligibility([case(SCIENTIFIC, []), *others], unlocated=[10201]) == INCOMPLETE


def test_the_route_reads_eligibility() -> None:
    """A failing rate is never ``failing`` to the route: admitted, technical or waiting by its
    eligibility."""
    for eligible, condition in ((QUALIFIED, "admitted"), (NOT_QUALIFIED, "technical"),
                                (INCOMPLETE, "waiting")):  # fmt: skip
        assert _condition({"state": FAILING, "eligibility": eligible}, {})[0] == condition
