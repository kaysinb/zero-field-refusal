"""Study 5's order repeat on Llama-3.2-1B-Instruct (docs/study5-llama-registration.md): its seeds,
orders and pilot, the rule that fixes its two rates from the pilot's K_benign, and the checks
record that fixes its teacher-gap thresholds from its stage-0 run.

**Runs.** ``experiments/study5_llama``: arm A (AdamW) on Llama's own θ₀ responses, 25 updates of
B = 20, each of ``SEEDS`` in the three orders ``ORDER_OFFSETS`` (``train.order_seed`` = s + 1000,
s + 2000, s + 3000) at the two rates of ``rate_choice``: 60 runs.

**The checks record** (``checks_record``) reads the stage-0 run's summary
(``experiments/study5_llama_stage0``, ``zero_field_refusal.checks``): the path-field verdict
(``path_field.passed``) and the teacher-gap thresholds stage 0 proposes (``checks.GAP_MARGIN`` ×
the largest gap over seeds 0-2, as for Qwen's runs). A failed path-field check, or a missing
threshold, is a refusal: no Llama run trains without both.

**The rate rule** (``rate_choice``), fixed before the pilot: the pilot is ``PILOT_SEEDS`` in order 1
at each of ``PILOT_RATES``, K-only (``experiments/study5_llama_k_only``). For each pilot seed, the
rates at which K_benign at step 25 reaches ``TARGET_K_LOW`` (4.84, the median K_benign of A's
study-3 runs at η_l^A) and ``TARGET_K_HIGH`` (K\\*² / 4.84 ≈ 7.44, so that the two straddle K\\* = 6
at ln-width 0.43) are interpolated linearly in ln K against ln η between the lowest adjacent pair
of pilot rates that brackets each (the rates walked upward: a run above that pair is not read).
η_l and η_u are the two seeds' geometric means, each rounded to three significant digits. Then
the support check: at the rounded rates, each pilot seed's K, interpolated the same way on its
pilot curve, must bracket K\\* (study 3's rule: K_l ≤ 6 ≤ K_u) at a width ln K_u − ln K_l of at
most 0.60. A pilot seed with a stopped run below a bracket, no bracket, or a failed support check
is a refusal: the rates are then not fixed and no main run starts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

SEEDS = tuple(range(60000, 60010))  # fresh seeds of Llama's own θ₀, none of Qwen's
ORDER_OFFSETS = (("order1", 1000), ("order2", 2000), ("order3", 3000))
PILOT_SEEDS = (60000, 60001)
PILOT_ORDER = "order1"
PILOT_RATES = (5.0e-06, 1.0e-05, 2.0e-05, 4.0e-05, 8.0e-05, 1.6e-04)
K_STAR = 6.0
TARGET_K_LOW = 4.84  # A's median K_benign at η_l^A in study 3 (docs/reports/study3-confirmation-c1.md)
TARGET_K_HIGH = K_STAR**2 / TARGET_K_LOW  # ≈ 7.44: ln-width 0.43 around K* = 6
WIDTH = 0.60  # study 3's widest bracket in ln K
CONFIG = "experiments/study5_llama"
PILOT_CONFIG = "experiments/study5_llama_k_only"
THETA0_CONFIG = "experiments/study5_llama_theta0"
STAGE0_CONFIG = "experiments/study5_llama_stage0"
GAP_KEYS = ("token_max_abs", "response_mean_signed_abs")


def round3(value: float) -> float:
    """``value`` to three significant digits, as the float of its decimal literal."""
    return float(f"{value:.3g}")


def order_seed(seed: int, order: str) -> int:
    return seed + dict(ORDER_OFFSETS)[order]


def checks_record(summary: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    """The Llama Pod's checks record from its stage-0 run's summary; ``ValueError`` when the
    path-field check did not pass or a threshold is missing."""
    path_field = summary.get("path_field")
    if not isinstance(path_field, Mapping) or path_field.get("passed") is not True:
        raise ValueError("the stage-0 path-field check did not pass")
    proposed = summary.get("teacher_gap_proposed_thresholds")
    if not isinstance(proposed, Mapping) or not all(
            isinstance(proposed.get(k), float) and math.isfinite(proposed[k]) and proposed[k] > 0
            for k in GAP_KEYS):  # fmt: skip
        raise ValueError("the stage-0 run proposes no finite positive teacher-gap thresholds")
    return {
        "kind": "study5-llama-checks",
        "stage0_run": run_id,
        "path_field_passed": True,
        "eos_fraction_at_cap": summary.get("eos_fraction_at_cap"),
        "teacher_gap": summary.get("teacher_gap"),
        "thresholds": {k: float(proposed[k]) for k in GAP_KEYS},
    }


def threshold_overrides(record: Mapping[str, Any]) -> list[str]:
    """The runs' overrides of ``checks.teacher_gap`` from a checks record."""
    return [f"checks.teacher_gap.{k}={float(record['thresholds'][k])!r}" for k in GAP_KEYS]


def pilot_curve(rows: Iterable[Mapping[str, Any]], seed: int) -> list[tuple[float, Any]]:
    """``seed``'s pilot points (ln η, ln K or the reason a run has none), the rates upward;
    ``ValueError`` without the pilot's rates."""
    found = {float(r["rate"]): r for r in rows if int(r["seed"]) == seed}
    if sorted(found) != sorted(PILOT_RATES):
        raise ValueError(f"seed {seed}: the pilot's {len(PILOT_RATES)} rates are expected")
    points: list[tuple[float, Any]] = []
    for rate in PILOT_RATES:
        row, k = found[rate], found[rate].get("K")
        ok = row.get("complete") is True and isinstance(k, float) and k > 0
        points.append((math.log(rate), math.log(k) if ok else f"the run at {rate!r} has no K"))
    return points


def _segments(points: list[tuple[float, Any]], seed: int):
    """Adjacent pilot pairs upward; a pair with a run that has no K is a refusal when reached."""
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        for y in (y0, y1):
            if isinstance(y, str):
                raise ValueError(f"seed {seed}: {y} at step 25")
        yield x0, y0, x1, y1


def seed_rate(rows: Iterable[Mapping[str, Any]], seed: int, target: float) -> float:
    """The rate at which ``seed``'s K reaches ``target`` (module docstring); ``ValueError`` when a
    run below the bracket has no K or no adjacent pair brackets it."""
    want = math.log(target)
    for x0, y0, x1, y1 in _segments(pilot_curve(rows, seed), seed):
        if y0 <= want <= y1 and y1 > y0:
            return math.exp(x0 + (want - y0) * (x1 - x0) / (y1 - y0))
    raise ValueError(f"seed {seed}: no adjacent pilot rates bracket K = {target:.4g}")


def predicted_k(rows: Iterable[Mapping[str, Any]], seed: int, rate: float) -> float:
    """``seed``'s K at ``rate`` by the same interpolation on its pilot curve; ``ValueError``
    outside the pilot's rates or across a run without K."""
    x = math.log(rate)
    for x0, y0, x1, y1 in _segments(pilot_curve(rows, seed), seed):
        if x0 <= x <= x1:
            return math.exp(y0 + (x - x0) * (y1 - y0) / (x1 - x0))
    raise ValueError(f"seed {seed}: the rate {rate!r} lies outside the pilot's rates")


def rate_choice(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The two rates from the pilot's rows (``seed``, ``rate``, ``K``, ``complete``) and the
    support check at them (module docstring); ``ValueError`` when the rule fixes none."""
    rows = list(rows)
    targets = {"eta_l": TARGET_K_LOW, "eta_u": TARGET_K_HIGH}
    per_seed = {name: {seed: seed_rate(rows, seed, k) for seed in PILOT_SEEDS}
                for name, k in targets.items()}  # fmt: skip
    rates = {name: round3(math.exp(sum(math.log(r) for r in found.values()) / len(found)))
             for name, found in per_seed.items()}  # fmt: skip
    support = {}
    for seed in PILOT_SEEDS:
        k_l = predicted_k(rows, seed, rates["eta_l"])
        k_u = predicted_k(rows, seed, rates["eta_u"])
        width = math.log(k_u) - math.log(k_l)
        supported = k_l <= K_STAR <= k_u and width <= WIDTH + 1e-9
        support[str(seed)] = {"K_l": k_l, "K_u": k_u, "width": width, "supported": supported}
        if not supported:
            raise ValueError(f"seed {seed}: at the rounded rates K {k_l:.4g} to {k_u:.4g} (width "
                             f"{width:.3f}) do not support K* = {K_STAR} within {WIDTH}")  # fmt: skip
    return {"targets": targets,
            "per_seed": {name: {str(s): r for s, r in found.items()}
                         for name, found in per_seed.items()},
            "eta_l": rates["eta_l"], "eta_u": rates["eta_u"], "support": support}  # fmt: skip


def cell(seed: int, order: str, lr: float, thresholds: list[str], *,
         pilot: bool = False) -> dict[str, Any]:  # fmt: skip
    """One Llama run: ``seed`` in ``order`` at ``lr``, the stage-0 thresholds as overrides."""
    return {"name": f"s5-llama-{'pilot-' if pilot else ''}{order}-{lr!r}-s{seed}",
            "config": PILOT_CONFIG if pilot else CONFIG,
            "overrides": [f"optimizer.lr={lr!r}", f"train.order_seed={order_seed(seed, order)}",
                          *thresholds],
            "seeds": [seed]}  # fmt: skip
