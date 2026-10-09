import math

import pytest

from zero_field_refusal.analysis.decisions import (
    h2_comparison,
    h2_mechanism,
    look_decision,
    sequential_decision,
)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (0.01, 0.09, "equivalence"),  # excludes 0, but equivalence takes precedence
        (-0.09, -0.01, "equivalence"),
        (-0.05, 0.05, "equivalence"),
        (-0.10, 0.10, "equivalence"),  # the margin itself is within ±10 pp
        (0.02, 0.15, "positive"),
        (-0.15, -0.02, "opposite"),
        (-0.05, 0.15, "inconclusive"),
        (0.0, 0.2, "inconclusive"),  # touches 0: does not exclude it
        (math.nan, math.nan, "inconclusive"),
    ],
)
def test_look_decision(low: float, high: float, expected: str) -> None:
    assert look_decision(low, high) == expected


def test_margin_is_a_parameter() -> None:
    assert look_decision(0.01, 0.09, margin=0.05) == "positive"


def test_look_2_only_after_an_inconclusive_look_1() -> None:
    assert sequential_decision("positive") == "positive"
    assert sequential_decision("equivalence", "positive") == "equivalence"
    assert sequential_decision("inconclusive", "opposite") == "opposite"
    assert sequential_decision("inconclusive", "inconclusive") == "inconclusive"
    with pytest.raises(ValueError, match="look 2"):
        sequential_decision("inconclusive")


def test_h2_comparison_labels() -> None:
    assert h2_comparison(0.01, 0.09) == "indistinguishable"
    assert h2_comparison(0.05, 0.30) == "different"
    assert h2_comparison(-0.30, -0.05) == "different"
    assert h2_comparison(-0.05, 0.30) == "inconclusive"


@pytest.mark.parametrize(
    ("vs_sgd", "vs_adamw", "between", "expected"),
    [
        ("indistinguishable", "different", True, "sign_step_rectification"),
        ("different", "indistinguishable", True, "flat_direction_heating"),
        ("different", "different", True, "both"),
        ("different", "different", False, "none"),  # outside the SGD-AdamW range
        ("indistinguishable", "indistinguishable", True, "none"),
        ("inconclusive", "different", True, "none"),
    ],
)
def test_h2_mechanism_table(vs_sgd: str, vs_adamw: str, between: bool, expected: str) -> None:
    mechanism = h2_mechanism(
        vs_sgd, vs_adamw, between, descriptive_only=False, primary_decision="positive"
    )
    assert mechanism == expected


def test_h2_draws_no_conclusion_without_overlap_or_without_h1() -> None:
    args = ("indistinguishable", "different", True)
    assert (
        h2_mechanism(*args, descriptive_only=True, primary_decision="positive")
        == "descriptive_only"
    )
    for decision in ("equivalence", "opposite", "inconclusive"):
        assert (
            h2_mechanism(*args, descriptive_only=False, primary_decision=decision)
            == "not_applicable"
        )
