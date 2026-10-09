"""Gate 2 decision rules and the H2 mechanism statements (docs/preregistration.md).

A look's result depends on the CI alone. Equivalence (the CI within ±margin, bounds included)
takes precedence over a directional result, even when the CI excludes 0. Otherwise a CI that
excludes 0 (strictly) is positive or opposite by its sign; anything else, including a CI with
non-finite bounds, is inconclusive. Look 1 stops on any of the first three results; look 2 runs
only after an inconclusive look 1 and its result is final.

H2 compares frozen Adam with SGD and with AdamW at K\\* through the CIs of the differences, at the
level of the look at which the decision is made: ``indistinguishable`` is the equivalence result,
``different`` a directional one. ``between`` means the estimate of h_FA(K\\*) lies between those of
SGD and AdamW (bounds included). A mechanism statement is made only when H1 holds (a positive
result) and frozen Adam's K range overlaps the common range. ``unavailable`` marks a look
without a frozen-Adam curve (no frozen-Adam run left after the exclusion of small K).
"""

from __future__ import annotations

import math

EQUIVALENCE_MARGIN = 0.10  # ±10 pp

POSITIVE = "positive"
OPPOSITE = "opposite"
EQUIVALENCE = "equivalence"
INCONCLUSIVE = "inconclusive"
DECISIONS = (POSITIVE, OPPOSITE, EQUIVALENCE, INCONCLUSIVE)

DIFFERENT = "different"
INDISTINGUISHABLE = "indistinguishable"

RECTIFICATION = "sign_step_rectification"
HEATING = "flat_direction_heating"
BOTH = "both"
NO_MECHANISM = "none"
DESCRIPTIVE_ONLY = "descriptive_only"
NOT_APPLICABLE = "not_applicable"
UNAVAILABLE = "unavailable"
MECHANISMS = (
    RECTIFICATION,
    HEATING,
    BOTH,
    NO_MECHANISM,
    DESCRIPTIVE_ONLY,
    NOT_APPLICABLE,
    UNAVAILABLE,
)


def look_decision(ci_low: float, ci_high: float, margin: float = EQUIVALENCE_MARGIN) -> str:
    """The result of one look from the CI of Δ."""
    if not (math.isfinite(ci_low) and math.isfinite(ci_high)):
        return INCONCLUSIVE
    if -margin <= ci_low and ci_high <= margin:
        return EQUIVALENCE
    if ci_low > 0:
        return POSITIVE
    if ci_high < 0:
        return OPPOSITE
    return INCONCLUSIVE


def sequential_decision(look1: str, look2: str | None = None) -> str:
    """The final result: look 1 unless it is inconclusive, then look 2."""
    if look1 != INCONCLUSIVE:
        return look1
    if look2 is None:
        raise ValueError("Look 1 is inconclusive: the decision needs look 2")
    return look2


def h2_comparison(ci_low: float, ci_high: float, margin: float = EQUIVALENCE_MARGIN) -> str:
    """Frozen Adam vs one optimizer: different, indistinguishable or inconclusive."""
    decision = look_decision(ci_low, ci_high, margin)
    if decision == EQUIVALENCE:
        return INDISTINGUISHABLE
    if decision in (POSITIVE, OPPOSITE):
        return DIFFERENT
    return INCONCLUSIVE


def h2_mechanism(
    vs_sgd: str,
    vs_adamw: str,
    between: bool,
    *,
    descriptive_only: bool,
    primary_decision: str,
) -> str:
    """The mechanism the result is consistent with, per the H2 statement."""
    if descriptive_only:
        return DESCRIPTIVE_ONLY
    if primary_decision != POSITIVE:
        return NOT_APPLICABLE
    if vs_sgd == INDISTINGUISHABLE and vs_adamw == DIFFERENT:
        return RECTIFICATION
    if vs_adamw == INDISTINGUISHABLE and vs_sgd == DIFFERENT:
        return HEATING
    if vs_sgd == DIFFERENT and vs_adamw == DIFFERENT and between:
        return BOTH
    return NO_MECHANISM
