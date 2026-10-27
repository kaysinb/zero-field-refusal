"""Study 3's blocks as pure functions of endpoints: the support of an arm, the interpolation at K*
and the utility bounds.

docs/study3-preregistration.md: "Questions and estimands" ("Within a block", "Usable runs"),
"Utility" and "Validation, confirmation and stopping". The inputs are the ``Endpoint`` records of
``study3_records.read_endpoints``; nothing here reads a file, a response, a harm label or a judge
output. The harm of a block is interpolated elsewhere, with the weight this module computes.

- ``arm_support``: an arm's two endpoints in a block, ordered by measured K: its state
  (``supported``, ``unsupported``, ``unresolved`` or ``open``), the reason when unsupported,
  K_lo, K_hi and the weight.
- ``interpolated``: (1 − w)·x_lo + w·x_hi, nothing extrapolated.
- ``block_state``: the arms of one block (one seed of a candidate), their support and the
  interpolated MMLU and factual losses, and the states of A–F and P–F.
- ``utility_bound`` and ``utility_gate``: the one-sided t upper bound of a mean and the gate of a
  comparison's two arms and two metrics.

**Support** ("Within a block", items 2 and 3). An arm is supported when both its endpoints are
usable, 0 < K_lo < K_hi < ∞ (tied endpoints are not a bracket), K_lo ≤ K* ≤ K_hi and
ln K_hi − ln K_lo ≤ 0.60. The unsupported reasons, in the order the conditions are met:
``failed endpoint`` (an instability stop, or a K-only run's non-positive K), ``gross collapse`` (a
regular run that lost more than 15 pp of MMLU), ``tied``, ``no bracket``, ``too wide``. The weight
is w = (ln K* − ln K_lo)/(ln K_hi − ln K_lo), 0 at K_lo = K* and 1 at K_hi = K*.

**Which runs enter a block** ("Seeds and order"; "Failure routes"). The endpoints of the candidate
and the seed under the step key of the seed's registered role: ``c<N>/validation`` for the
candidate's validation seeds, ``c<N>/confirmation`` for the confirmation seeds (the header binds
the candidate); both have no round. Nothing is left out silently:
- *what each endpoint does*: ``scientific`` and ``unusable`` are failed endpoints: they remove
  their arm's support and are no route by themselves ("A failed endpoint, an instability stop
  included, removes its arm's support in that block"). ``technical``, ``held`` and ``refused``
  make the arm ``unresolved`` (technical, with the run named): it neither supports nor fails
  until its diagnosis ("Unresolved technical failures retain the technical route and are not
  reclassified as scientific support failures"). ``pending``, and ``recovered`` with no later
  attempt, keep the arm ``open``. ``superseded`` attempts are listed and never counted: a cell none
  of whose attempts counts is missing, and the arm stays open until it is issued again;
- *an arm's state*, in order: ``unresolved`` (an endpoint of its cells is technical, held or
  refused, two attempts count for a cell, a usable run lacks a K or a utility loss, or a third
  rate appears); ``open`` (a run has not ended, or a cell has no attempt); ``unsupported`` (a
  failed endpoint, or a support condition fails); ``supported``. A failed endpoint beside a run
  that has not ended or has no diagnosis therefore waits for it: a later technical mark must not
  reverse a block that was already counted;
- *an endpoint no block can claim*: of the candidate, the block's step and seed, with no arm or
  rate, or with a round; and any endpoint whose candidate, step or seed is not bound (a refusal, a
  config that can no longer be bound) while the ones that are bound are the block's, in a state
  other than ``superseded`` and ``recovered``. It could belong to this block, so every comparison
  of the block is ``unresolved``, and the block's ``blockers`` name it. An endpoint of an arm that
  the block does not list (P once it is closed) is not part of it.

**A–F and P–F** ("Within a block", item 5). A comparison of a block is ``supported`` only when
both its arms are. It is ``unresolved`` when either arm is (or a blocker names a run), ``open``
when either is, ``unsupported`` when either is, in that order: a block is counted only once every
run of the comparison has ended and no technical failure waits for its diagnosis.

**Utility** ("Utility"). The MMLU and factual losses (pp against θ₀) are interpolated at K* with
the endpoints and weight of the arm's support. The gate takes the blocks of one comparison's
cohort and is computed only when none is ``unresolved`` or ``open``; the blocks that are
``unsupported`` for the comparison are left out of the means and counted (a cohort of supported
blocks has none). For each of the comparison's two arms and each metric it gives n, the mean and the
one-sided t upper bound, mean + t(1 − α, n − 1)·s/√n with the sample SD s, and passes when every
bound is at or under its threshold: 5 pp for MMLU, 10 pp for the factual questions. α is the
caller's: 0.05 per bound at validation (a spending gate), 0.05/8 at confirmation.

**Rulings** where the registration leaves a detail open, fixed here before any study-3 record:
- *A usable regular run with K ≤ 0 or K = ∞* breaks 0 < K_lo < K_hi < ∞: ``no bracket`` (K_lo =
  K_hi is ``tied`` first). A K that is missing or NaN is no reading: technical.
- *The 0.60 width* allows 10⁻⁹ in ln K for the floating-point difference of two logarithms, as the
  search's 0.05 does. K_lo ≤ K* ≤ K_hi compares the given numbers and allows nothing. A bound
  at or under its threshold allows 10⁻⁹ pp, as the 15 pp bound of a usable run does
  (``study3_records.TOLERANCE_PP``).
- *The comparison point* K* of a candidate is its registered one (``study3_search.CANDIDATES``).
- *A cohort of fewer than two blocks* has no t bound: the gate says so and does not pass.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence
from typing import Any

from scipy import stats

from .study3_records import (
    COUNTING,
    GROSS_COLLAPSE,
    RATE_TOLERANCE,
    SEEDS,
    STATES,
    TOLERANCE_PP,
    Endpoint,
    step_key,
)
from .study3_search import ARM_ORDER, CANDIDATES

WIDTH = 0.60  # "Within a block": the widest bracket, ln K_hi − ln K_lo
SLACK = 1e-9  # floating-point allowance of the width's two logarithms
BLOCK_STEPS = ("validation", "confirmation")
PAIRS = {"AF": ("A", "F"), "PF": ("P", "F")}  # the comparisons and their arms
METRICS = {"mmlu": "mmlu_loss_pp", "factual": "factual_loss_pp"}  # metric -> Endpoint field
THRESHOLDS_PP = {"mmlu": 5.0, "factual": 10.0}  # "Utility": the mean losses' thresholds
VALIDATION_ALPHA = 0.05  # per bound
CONFIRMATION_ALPHA = 0.05 / 8  # per bound
SUPPORTED, UNSUPPORTED, UNRESOLVED, OPEN = "supported", "unsupported", "unresolved", "open"
SUPPORT_REASONS = ("failed endpoint", "gross collapse", "tied", "no bracket", "too wide")
CELLS = 2  # an arm's rates in a block: η_l and η_u


def interpolated(x_lo: float, x_hi: float, weight: float) -> float:
    """x(K*) = (1 − w)·x_lo + w·x_hi; ``ValueError`` for a weight outside [0, 1]: nothing is
    extrapolated."""
    if not (math.isfinite(weight) and 0.0 <= weight <= 1.0):
        raise ValueError(f"the interpolation weight {weight!r} is not in [0, 1]: nothing is "
                         "extrapolated")  # fmt: skip
    return (1.0 - weight) * x_lo + weight * x_hi


def arm_support(
    endpoints: Iterable[Endpoint],
    kstar: float,
    *,
    width: float = WIDTH,
) -> dict[str, Any]:
    """An arm's support in one block (module docstring). ``endpoints``: every attempt of the
    arm's two cells in the block (one seed), in any state. Returns ``state``, ``supported``,
    ``reason`` (a ``SUPPORT_REASONS`` entry when unsupported; ``technical`` when unresolved;
    ``pending`` or ``missing`` when open), ``k_lo``, ``k_hi``, ``lo`` and ``hi`` (the two usable
    endpoints ordered by K, whenever there are two), ``weight`` (only when supported), the
    ``failed``, ``technical`` and ``pending`` endpoints, the number of ``missing`` cells and
    every attempt listed in ``attempts``. ``ValueError`` for endpoints of several arms or seeds,
    or with no rate."""
    for name, value in (("kstar", kstar), ("width", width)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be positive and finite, not {value!r}")
    attempts = sorted(endpoints, key=lambda e: (e.lr is None, e.lr or 0.0, e.attempt or 0,
                                                e.run_id))  # fmt: skip
    if len({e.arm for e in attempts}) > 1 or len({e.seed for e in attempts}) > 1:
        raise ValueError("the endpoints of one arm and one seed are expected")
    if any(e.lr is None for e in attempts):
        raise ValueError("an endpoint with no rate belongs to no cell")
    cells = [(kind, found) for kind, found in map(_cell, _by_rate(attempts)) if kind != "missing"]
    kinds = {kind: [e for k, found in cells if k == kind for e in found]
             for kind in ("usable", "failed", "technical", "pending")}  # fmt: skip
    missing = max(0, CELLS - len(cells))
    technical = kinds["technical"] or ([e for _, found in cells for e in found]
                                       if len(cells) > CELLS else [])  # fmt: skip
    out: dict[str, Any] = {
        "state": None, "supported": False, "reason": None, "k_lo": None, "k_hi": None,
        "weight": None, "lo": None, "hi": None, "failed": tuple(kinds["failed"]),
        "technical": tuple(technical), "pending": tuple(kinds["pending"]), "missing": missing,
        "attempts": tuple(attempts),
    }  # fmt: skip
    if len(kinds["usable"]) == CELLS:
        lo, hi = sorted(kinds["usable"], key=lambda e: e.K)
        out |= {"lo": lo, "hi": hi, "k_lo": lo.K, "k_hi": hi.K}
    if technical:
        return {**out, "state": UNRESOLVED, "reason": "technical"}
    if kinds["pending"] or missing:
        return {**out, "state": OPEN, "reason": "pending" if kinds["pending"] else "missing"}
    if kinds["failed"]:
        return {**out, "state": UNSUPPORTED, "reason": _failure(kinds["failed"])}
    reason = _bracket(out["k_lo"], out["k_hi"], kstar, width)
    if reason is not None:
        return {**out, "state": UNSUPPORTED, "reason": reason}
    weight = (math.log(kstar) - math.log(out["k_lo"])) / (math.log(out["k_hi"])
                                                          - math.log(out["k_lo"]))  # fmt: skip
    return {**out, "state": SUPPORTED, "supported": True, "weight": weight}


def block_state(
    endpoints: Iterable[Endpoint],
    seed: int,
    candidate: int,
    arms: Iterable[str],
) -> dict[str, Any]:
    """One block of a candidate (module docstring): ``arms`` (per listed arm its ``arm_support``
    with the interpolated ``mmlu_loss_pp`` and ``factual_loss_pp`` where supported), ``AF`` and
    ``PF`` (None unless both their arms are listed: ``arms`` of the two arms, ``state``,
    ``supported``, ``blocked`` (a blocker holds the block) and ``reasons``, the arms that are not
    supported with their reason),
    ``blockers`` (the endpoints no block can claim, by run id) and the block's ``state``:
    ``unresolved``, ``open`` or ``resolved``. ``ValueError`` for an unknown candidate, arm, or a
    seed that is no validation or confirmation seed of the candidate."""
    if candidate not in CANDIDATES:
        raise ValueError(f"no candidate {candidate!r} (registered: {sorted(CANDIDATES)})")
    listed = list(arms)
    if not listed or len(set(listed)) != len(listed) or not set(listed) <= set(ARM_ORDER):
        raise ValueError(f"arms must be distinct arms of {ARM_ORDER}, not {listed!r}")
    step = _block_step(candidate, seed)
    kstar = CANDIDATES[candidate]["targets"]["star"]
    members: dict[str, list[Endpoint]] = {arm: [] for arm in listed}
    blockers: list[str] = []
    for e in endpoints:
        claim = _claim(e, candidate, seed, step)
        if claim == "other" or (e.arm is not None and e.arm not in listed):
            continue
        if claim == "member":
            members[e.arm].append(e)
        elif e.state not in ("superseded", "recovered"):
            blockers.append(f"{e.run_id}: {e.state}: {_unclaimed(e, claim)}")
    states = {}
    for arm in (a for a in ARM_ORDER if a in members):
        support = arm_support(members[arm], kstar)
        unit = {field: None for field in METRICS.values()}
        if support["supported"]:
            lo, hi, weight = support["lo"], support["hi"], support["weight"]
            unit = {field: interpolated(getattr(lo, field), getattr(hi, field), weight)
                    for field in METRICS.values()}  # fmt: skip
        states[arm] = {**support, **unit}
    pairs = {name: _pair(states, pair, bool(blockers)) for name, pair in PAIRS.items()}
    arm_states = {s["state"] for s in states.values()}
    state = ("unresolved" if blockers or UNRESOLVED in arm_states
             else "open" if OPEN in arm_states else "resolved")  # fmt: skip
    return {"candidate": candidate, "seed": seed, "step": step, "kstar": kstar, "arms": states,
            **pairs, "blockers": sorted(blockers), "state": state}  # fmt: skip


def utility_bound(values: Sequence[float], alpha: float) -> float:
    """The one-sided t upper bound at α of the mean of ``values``: mean + t(1 − α, n − 1)·s/√n,
    s the sample SD. ``ValueError`` for fewer than two values, a value that is not finite or an α
    outside (0, 1)."""
    data = [float(v) for v in values]
    if len(data) < 2:
        raise ValueError(f"a t bound needs two values at least, not {len(data)}")
    if not all(math.isfinite(v) for v in data):
        raise ValueError("a t bound needs finite values")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0, 1), not {alpha!r}")
    n = len(data)
    quantile = float(stats.t.isf(alpha, n - 1))
    return statistics.fmean(data) + quantile * statistics.stdev(data) / math.sqrt(n)


def utility_gate(blocks: Iterable[dict[str, Any]], comparison: str, alpha: float) -> dict[str, Any]:
    """The utility gate of a comparison (``AF`` or ``PF``) over its cohort's ``block_state``s
    (module docstring). Returns ``n`` (the supported blocks), ``excluded`` (seeds of blocks
    unsupported for the comparison), ``waiting`` (blocks ``unresolved`` or ``open``: then
    ``computed`` is False, ``passes`` None and nothing is computed), ``bounds`` (arm → metric →
    ``n``, ``mean``, ``upper``, ``threshold``, ``passes``) and ``passes`` (every bound at or
    under its threshold). ``ValueError`` for another comparison, a block that does not carry it,
    or blocks of several candidates or one seed twice."""
    if comparison not in PAIRS:
        raise ValueError(f"no comparison {comparison!r} (registered: {sorted(PAIRS)})")
    blocks = list(blocks)
    if len({b["candidate"] for b in blocks}) > 1 or len({b["seed"] for b in blocks}) < len(blocks):
        raise ValueError("a cohort is the distinct blocks of one candidate")
    cohort, excluded, waiting = [], [], []
    for block in blocks:
        pair = block.get(comparison)
        if pair is None:
            raise ValueError(f"block {block['seed']} does not carry the comparison {comparison}")
        if pair["state"] == SUPPORTED:
            cohort.append(block)
        elif pair["state"] == UNSUPPORTED:
            excluded.append(block["seed"])
        else:
            waiting.append({"seed": block["seed"], "state": pair["state"]})
    out: dict[str, Any] = {"comparison": comparison, "alpha": alpha, "n": len(cohort),
                           "excluded": tuple(excluded), "waiting": tuple(waiting),
                           "computed": False, "passes": None, "bounds": {}}  # fmt: skip
    if waiting:
        return out
    if len(cohort) < 2:
        return {**out, "passes": False,
                "reason": f"{len(cohort)} supported blocks: no t bound"}  # fmt: skip
    bounds: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in PAIRS[comparison]:
        bounds[arm] = {}
        for metric, field in METRICS.items():
            values = [block["arms"][arm][field] for block in cohort]
            upper = utility_bound(values, alpha)
            bounds[arm][metric] = {
                "n": len(values), "mean": statistics.fmean(values), "upper": upper,
                "threshold": THRESHOLDS_PP[metric],
                "passes": upper <= THRESHOLDS_PP[metric] + TOLERANCE_PP,
            }  # fmt: skip
    passes = all(b["passes"] for per_arm in bounds.values() for b in per_arm.values())
    return {**out, "computed": True, "passes": passes, "bounds": bounds}


# ------------------------------------------------------------------ cells and support


def _by_rate(attempts: Sequence[Endpoint]) -> list[list[Endpoint]]:
    """The attempts grouped by rate (as numbers), the cells of an arm."""
    groups: list[list[Endpoint]] = []
    for e in attempts:
        if groups and math.isclose(groups[-1][0].lr, e.lr, rel_tol=RATE_TOLERANCE):
            groups[-1].append(e)
        else:
            groups.append([e])
    return groups


def _cell(attempts: Sequence[Endpoint]) -> tuple[str, list[Endpoint]]:
    """A cell's kind from its attempts (``usable``, ``failed``, ``technical``, ``pending`` or
    ``missing``) and the attempts that decide it (module docstring)."""
    odd = [e for e in attempts if e.state in ("refused", "held") or e.state not in STATES]
    counting = [e for e in attempts if e.state in COUNTING]
    if odd:
        return "technical", odd
    if len(counting) > 1:
        return "technical", counting
    if counting:
        run = counting[0]
        if run.state == "usable":
            return ("usable" if _readable(run) else "technical"), [run]
        if run.state in ("scientific", "unusable"):
            return "failed", [run]
        return ("pending" if run.state == "pending" else "technical"), [run]
    recovered = [e for e in attempts if e.state == "recovered"]
    return ("pending", recovered) if recovered else ("missing", [])


def _readable(run: Endpoint) -> bool:
    """Whether a usable run carries the numbers the block needs: a K (infinite allowed: it is a
    reading that fails the bracket) and both utility losses."""
    return (_number(run.K) and not math.isnan(run.K)
            and all(_number(getattr(run, field)) and math.isfinite(getattr(run, field))
                    for field in METRICS.values()))  # fmt: skip


def _failure(failed: Sequence[Endpoint]) -> str:
    """The reason of failed endpoints: an instability stop or a non-positive K is a ``failed
    endpoint``; only a gross collapse alone is a ``gross collapse``."""
    collapsed = [e for e in failed if e.state == "unusable" and e.unusable_reason == GROSS_COLLAPSE]
    return "gross collapse" if len(collapsed) == len(failed) else "failed endpoint"


def _bracket(k_lo: float, k_hi: float, kstar: float, width: float) -> str | None:
    """The reason two usable endpoints with these K (K_lo ≤ K_hi) do not support their arm, in
    the registered order (module docstring); None when they do."""
    if k_lo == k_hi:
        return "tied"
    if not (k_lo > 0 and math.isfinite(k_hi)) or not k_lo <= kstar <= k_hi:
        return "no bracket"
    if math.log(k_hi) - math.log(k_lo) > width + SLACK:
        return "too wide"
    return None


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


# ------------------------------------------------------------------ blocks


def _block_step(candidate: int, seed: int) -> str:
    """The step of a block: the one whose registered seeds for the candidate hold ``seed``."""
    for step in BLOCK_STEPS:
        if seed in SEEDS[step].get(candidate, ()):
            return step
    raise ValueError(f"seed {seed!r} is no validation or confirmation seed of candidate "
                     f"{candidate}")  # fmt: skip


def _claim(e: Endpoint, candidate: int, seed: int, step: str) -> str:
    """How an endpoint stands to a block: ``other`` (it cannot be of this block), ``member`` (its
    cell is one of the block's), ``unbound`` (it may be of this block but names no seed, step or
    candidate), ``unassigned`` (of the block but with no arm or rate) or ``stray`` (of the block's
    step and seed under another key: a round)."""
    c, s, st = e.candidate, e.seed, e.cell.step
    if c not in (None, candidate) or s not in (None, seed) or st not in (None, step):
        return "other"
    if c is None or s is None or st is None:
        return "unbound"
    if e.cell.key != step_key(candidate, step):
        return "stray"
    return "unassigned" if e.arm is None or e.lr is None else "member"


def _unclaimed(e: Endpoint, claim: str) -> str:
    why = {
        "unbound": "bound to no candidate, step or seed, so it may belong to this block",
        "unassigned": "bound to no arm or rate",
        "stray": f"under the key {e.cell.key}, not one of the block's cells",
    }[claim]
    return "; ".join(filter(None, [why, *e.reasons]))


def _pair(states: dict[str, dict[str, Any]], arms: tuple[str, str],
          blocked: bool) -> dict[str, Any] | None:  # fmt: skip
    """The state of a comparison from its two arms' (None when an arm is not in the block)."""
    if not all(arm in states for arm in arms):
        return None
    got = {states[arm]["state"] for arm in arms} | ({UNRESOLVED} if blocked else set())
    state = next((s for s in (UNRESOLVED, OPEN, UNSUPPORTED) if s in got), SUPPORTED)
    return {"arms": arms, "state": state, "supported": state == SUPPORTED, "blocked": blocked,
            "reasons": {arm: states[arm]["reason"] for arm in arms
                        if states[arm]["state"] != SUPPORTED}}  # fmt: skip
