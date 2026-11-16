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
rate at which K_benign at step 25 reaches ``TARGET_K`` is interpolated linearly in ln K against
ln η between the two adjacent pilot rates that bracket it (the lowest such pair, the rates
walked upward: a run above that pair is not read); η_l is the
geometric mean of the two seeds' rates, rounded to three significant digits, and η_u = η_l ×
``RATE_RATIO`` (A's η_u / η_l in study 3), rounded the same way. ``TARGET_K`` is the median K_benign
of A's study-3 runs at η_l^A, so that Llama's η_l stands to K\\* as Qwen's A runs did. A pilot seed
with a stopped run below its bracket, or no bracket, is a refusal: the rates are then not fixed.
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
TARGET_K = 4.84  # A's median K_benign at η_l^A in study 3 (docs/reports/study3-confirmation-c1.md)
RATE_RATIO = 2.11e-05 / 1.45e-05  # A's η_u^A / η_l^A in study 3
K_STAR = 6.0
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


def seed_rate(rows: Iterable[Mapping[str, Any]], seed: int) -> float:
    """The rate at which ``seed``'s K reaches ``TARGET_K`` (module docstring); ``ValueError``
    when a pilot run of the seed did not end all 25 updates or no adjacent pair brackets it."""
    found = {float(r["rate"]): r for r in rows if int(r["seed"]) == seed}
    if sorted(found) != sorted(PILOT_RATES):
        raise ValueError(f"seed {seed}: the pilot's {len(PILOT_RATES)} rates are expected")
    target, previous = math.log(TARGET_K), None
    for rate in PILOT_RATES:  # upward; a run above the bracket found is not needed
        row = found[rate]
        k = row.get("K")
        if row.get("complete") is not True or not isinstance(k, float) or not k > 0:
            raise ValueError(f"seed {seed}: the run at {rate!r} has no K at step 25")
        x1, y1 = math.log(rate), math.log(k)
        if previous is not None:
            x0, y0 = previous
            if y0 <= target <= y1 and y1 > y0:
                return math.exp(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
        previous = (x1, y1)
    raise ValueError(f"seed {seed}: no adjacent pilot rates bracket K = {TARGET_K}")


def rate_choice(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The two rates from the pilot's rows (``seed``, ``rate``, ``K``, ``complete``)."""
    rows = list(rows)
    per_seed = {seed: seed_rate(rows, seed) for seed in PILOT_SEEDS}
    eta_l = round3(math.exp(sum(math.log(r) for r in per_seed.values()) / len(per_seed)))
    return {"target_k": TARGET_K, "per_seed": {str(s): r for s, r in per_seed.items()},
            "eta_l": eta_l, "eta_u": round3(eta_l * RATE_RATIO), "ratio": RATE_RATIO}  # fmt: skip


def cell(seed: int, order: str, lr: float, thresholds: list[str], *,
         pilot: bool = False) -> dict[str, Any]:  # fmt: skip
    """One Llama run: ``seed`` in ``order`` at ``lr``, the stage-0 thresholds as overrides."""
    return {"name": f"s5-llama-{'pilot-' if pilot else ''}{order}-{lr!r}-s{seed}",
            "config": PILOT_CONFIG if pilot else CONFIG,
            "overrides": [f"optimizer.lr={lr!r}", f"train.order_seed={order_seed(seed, order)}",
                          *thresholds],
            "seeds": [seed]}  # fmt: skip
