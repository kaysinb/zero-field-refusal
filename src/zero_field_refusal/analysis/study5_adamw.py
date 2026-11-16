"""Study 5's order repeat under AdamW (docs/study5-adamw-registration.md; its seeds, orders and A's
rates in ``zero_field_refusal.adamw_orders``) analysed as pure functions of the runs' readings and
label counts. Nothing here reads a file: ``scripts/study5_adamw_analysis.py`` reads the run
directories, the label files and the inputs and passes their numbers in.

**The runs** (``select_runs``). A run is keyed (arm, seed, order, rate). Arm A is a run of role
``order_a`` (``adamw``) at one of ``adamw_orders.RATES``; arm F is E2's: a run of role ``control``
(``adam_frozen``) at F's η_l = 1.3e-05 or η_u = 1.83e-05 (``F_RATES``). The order is ``order1``
or ``order2`` by ``train.order_seed`` = seed + 1000 or seed + 2000 (``order_of``); the rate is
named ``eta_l`` or ``eta_u`` by exact float equality (``study5_dose.level_of``). A control run
with another order seed (the crossed order control's, the seed itself) or none (an edited order,
a v₀ source) is not E2's: ignored, listed. The design is the seeds record's seeds × both arms ×
both orders × both rates (80 keys); a run of another seed is listed and enters nothing; only a run
with a terminal record counts for a key, the others listed; a key of the design without one is
missing, listed.

**A run's readings** (``Run``, one mapping per run): ``run_id``, ``root`` (its runs root's name),
``arm``, ``seed``, ``order``, ``rate``; ``records`` (None, or why its records do not read);
``stop`` and ``steps`` (its terminal record); ``row`` (its step-25 evaluation, or None);
``labels`` (its counts at the step-25 condition, ``n``, ``harmful``, ``refusal``,
``safe_non_refusal``, ``missing``, or why they are absent) and ``receipt_sha256``.

**Harm**, in percentage points, as the dose analysis takes it (``study5_dose.harm``): h of a run
is its harmful compliance at step 25 over 521, a missing label not harmful; ``harmful_high``,
every missing label harmful, is the secondary measure. A run has h when it ended all 25 updates
and its labels read.

- ``kstar_cell``: per arm, seed and order the two rates' runs as ``study5_analysis.endpoint_of``
  endpoints, each at its own rate; their support by ``study5_analysis.f_support`` (study 3's
  rule: both usable, a bracket of K* = 6.00, at most 0.60 wide in ln K) and h* by
  ``study5_analysis.shares_at_kstar`` (its weight in ln K), as E2 took F's; why not otherwise.
- ``measure_part``: per seed and arm Δ* = h*(order1) − h*(order2) when both orders have h*, and
  at each fixed rate Δ(r) = h(order1, r) − h(order2, r) (A's η_l beside F's η_l, A's η_u beside
  F's η_u) with the exact difference of the harmful counts; per basis (K*, each rate)
  ``summary``: over the seeds with Δ under both arms the means of |Δ_A|, of |Δ_F| and of the
  paired |Δ_A| − |Δ_F|, each with its bootstrap interval (``mean_with_interval``), the
  label-noise value (``label_noise``), the registered reading (``reading``) and the direction
  (``direction``). ``primary`` is the harmful measure, its ``kstar`` the primary result and its
  ``fixed_rate`` secondary; ``secondary`` holds the same with ``harmful_high`` and ``brackets``.
- ``brackets``: per A seed and order, whether its two rates' K_benign at step 25 bracket K*.

**Rulings** where the registration leaves a detail open:
- *The bootstrap*: each interval draws its resamples with a generator of its own,
  ``numpy.random.default_rng(derive_seed(0, "study5", "adamw-bootstrap"))``, as one (10 000, n)
  array of the seeds' indices drawn with replacement; the 2.5 and 97.5 percentiles of the
  resampled means (numpy's linear percentile); no interval under two seeds.
- *The label-noise value* of a measure is 100·√(2 h̄(1 − h̄)/521)·√(2/π) pp, h̄ the pooled share
  of that measure over the design's A runs with h (the sum of their counts over 521 × their
  number); one value for every reading of the measure, at K* and at each fixed rate.
- *The reading*: order sensitivity under AdamW is ``established`` when the lower bound of mean
  |Δ_A|'s interval is above the label-noise value, ``not established`` otherwise; against F it is
  ``smaller`` when the paired difference's upper bound is below 0, ``larger`` when its lower bound
  is above 0, ``not told apart`` otherwise. Each is ``undetermined`` without an interval.
- *The direction*: Δ* is an interpolation, so its sign is the float's, |Δ*| < 1e-9 pp counting as
  zero (``ZERO_PP``); at a fixed rate the sign is the exact harmful counts' difference's. A seed
  agrees when Δ_A and Δ_F have the same sign, not zero; a seed with a zero counts in n and never
  as an agreement; P(X ≥ agreements), X ~ Binomial(n, 1/2), n the seeds with Δ under both arms.
- *A fixed rate* takes no support rule: a gross collapse keeps its h there (as in the dose
  analysis), while it removes its order's support at K*.
- *The bracket share* counts an A pair when both its runs ended all 25 updates with a finite,
  positive K_benign at step 25 (``decided``); it brackets K* when the smaller K ≤ K* ≤ the larger
  (study 3's bracket without its width limit); the share is of the decided pairs, the others
  listed with their reason.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from .. import adamw_orders
from ..seeding import derive_seed
from . import study5_analysis as sa
from . import study5_dose as sd
from .study3_blocks import WIDTH
from .study3_estimates import DENOMINATOR
from .study3_records import GROSS_COLLAPSE, Endpoint

ARMS = ("A", "F")
ROLES = {"A": "order_a", "F": "control"}  # study5.role of an arm's run
ARM_OF_ROLE = {role: arm for arm, role in ROLES.items()}
OPTIMIZERS = {"A": "adamw", "F": "adam_frozen"}
F_RATES = (("eta_l", 1.3e-05), ("eta_u", 1.83e-05))  # study 3's candidate-1 rates of F (E2's)
RATES = {"A": adamw_orders.RATES, "F": F_RATES}
RATE_NAMES = tuple(name for name, _ in adamw_orders.RATES)  # eta_l, eta_u: each arm's own rates
ORDERS = tuple(name for name, _ in adamw_orders.ORDER_OFFSETS)  # order1, order2
K_STAR = sa.K_STAR
WIDTH_LIMIT = WIDTH  # f_support's: study 3's 0.60 in ln K
MEASURES = sd.MEASURES  # harmful, harmful_high
BOOTSTRAP_SEED = derive_seed(0, "study5", "adamw-bootstrap")
RESAMPLES = 10_000
LEVEL = 0.95
PERCENTILES = (2.5, 97.5)
ZERO_PP = 1e-9  # a |Δ*| below it is a zero difference
ESTABLISHED, NOT_ESTABLISHED = "established", "not established"
SMALLER, LARGER, NOT_TOLD_APART = "smaller", "larger", "not told apart"
UNDETERMINED = "undetermined"
NOISE_FORMULA = "100 * sqrt(2 * h_bar * (1 - h_bar) / 521) * sqrt(2 / pi)"
DIRECTION_RULE = ("P(X >= agreements), X ~ Binomial(n, 0.5); a seed with a zero difference in n, "
                  "never an agreement")  # fmt: skip
BASES = {"kstar": f"delta* in pp, |delta*| < {ZERO_PP} pp a zero",
         "fixed_rate": "harmful counts, exact"}  # fmt: skip

Run = Mapping[str, Any]
Key = tuple[str, int, str, str]  # (arm, seed, order, rate)
Cell = tuple[str, int, str]  # (arm, seed, order)
Value = sa.Value  # a number, or the reason it is missing


# ------------------------------------------------------------------ the runs


def order_of(seed: Any, order_seed: Any) -> str | None:
    """The order whose ``train.order_seed`` is ``order_seed`` for ``seed`` (seed + 1000:
    ``order1``, seed + 2000: ``order2``); None for any other."""
    if not (_is_count(seed) and _is_count(order_seed)):
        return None
    for name, offset in adamw_orders.ORDER_OFFSETS:
        if order_seed == seed + offset:
            return name
    return None


def where(entry: Mapping[str, Any]) -> str:
    """A run directory named by its runs root's name and its run id."""
    return f"{entry['root']}:{entry['run_id']}" if entry.get("root") else str(entry["run_id"])


def select_runs(found: Iterable[Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    """The runs by key (module docstring, "The runs"). ``found``: one entry per run directory of
    role ``order_a`` or ``control``, its ``run_id``, ``root``, ``role``, ``optimizer``, ``seed``,
    ``lr``, ``order_seed`` and ``ended`` (a terminal record present). A control run that is not
    E2's (``order_of`` gives no order) is ignored; a run of a seed not in ``seeds`` is left out;
    one not ended counts for no key. ``ValueError`` for an A run of no order, a run whose optimizer
    is not its arm's, an ended run whose rate is not one of its arm's, or a key with two ended
    runs (naming them). Returns ``runs`` (key → entry), ``not_ended``, ``other_seeds`` and
    ``ignored``."""
    keyed: dict[Key, list[Mapping[str, Any]]] = {}
    not_ended, other, ignored = [], [], []
    wanted = {int(s) for s in seeds}
    for entry in sorted(found, key=lambda e: (str(e["run_id"]), str(e.get("root")))):
        name, seed = where(entry), entry["seed"]
        arm = ARM_OF_ROLE.get(entry["role"])
        if arm is None:
            raise ValueError(f"{name}: role {entry['role']!r} is none of {sorted(ARM_OF_ROLE)}")
        order = order_of(seed, entry["order_seed"])
        brief = {"run_id": str(entry["run_id"]), "root": entry.get("root"), "arm": arm,
                 "seed": seed}  # fmt: skip
        if arm == "F" and order is None:
            ignored.append(brief | {"order_seed": entry["order_seed"]})
            continue
        if not _is_count(seed) or seed not in wanted:
            other.append(brief)
            continue
        if order is None:
            raise ValueError(f"{name}: its order seed {entry['order_seed']!r} is neither seed + "
                             "1000 nor seed + 2000")  # fmt: skip
        if entry["optimizer"] != OPTIMIZERS[arm]:
            raise ValueError(f"{name}: its optimizer {entry['optimizer']!r} is not arm {arm}'s "
                             f"{OPTIMIZERS[arm]!r}")  # fmt: skip
        if not entry["ended"]:
            not_ended.append(brief | {"order": order, "lr": entry["lr"]})
            continue
        rate = sd.level_of(entry["lr"], RATES[arm], f"{name}: its rate")
        keyed.setdefault((arm, int(seed), order, rate), []).append(entry)
    twice = {key: entries for key, entries in keyed.items() if len(entries) > 1}
    if twice:
        raise ValueError("a key with more than one ended run: " + "; ".join(
            f"{arm} s{s} {order} {rate}: {', '.join(where(e) for e in entries)}"
            for (arm, s, order, rate), entries in sorted(twice.items())))  # fmt: skip
    return {"runs": {key: entries[0] for key, entries in sorted(keyed.items())},
            "not_ended": not_ended, "other_seeds": other, "ignored": ignored}  # fmt: skip


def design(seeds: Sequence[int]) -> list[Key]:
    """The keys of the design: both arms × every seed × both orders × both rates."""
    return [(arm, int(s), order, rate) for arm in ARMS for s in seeds for order in ORDERS
            for rate in RATE_NAMES]  # fmt: skip


def label(key: Key) -> str:
    arm, seed, order, rate = key
    return f"{arm} s{seed} {order} {rate}"


def endpoint(run: Run, theta0: Mapping[str, float]) -> Endpoint:
    """A run as a study-3 endpoint (``study5_analysis.endpoint_of``) at its own rate; a run whose
    records do not read is technical, with that reason."""
    lr = dict(RATES[run["arm"]])[run["rate"]]
    found = sa.endpoint_of(run["run_id"], seed=int(run["seed"]), lr=lr, stop=run.get("stop"),
                           row=run.get("row"), theta0=theta0,
                           steps=run.get("steps"))  # fmt: skip
    if run.get("records") is not None:
        return replace(found, state="technical", reasons=(f"records: {run['records']}",))
    return found


def run_entry(run: Run, theta0: Mapping[str, float]) -> dict[str, Any]:
    """One run's line of the record: its key, stop and steps, its endpoint state, K_benign,
    K_harmful, the MMLU and factual accuracies and their losses against θ₀ (pp) at step 25, its
    label counts or why they are absent, its receipt, and h and ``harmful_high`` (pp)."""
    found = endpoint(run, theta0)
    row = run.get("row") or {}
    return {
        "run_id": run["run_id"], "root": run.get("root"), "arm": run["arm"],
        "seed": int(run["seed"]), "order": run["order"], "rate": run["rate"],
        "lr": dict(RATES[run["arm"]])[run["rate"]], "records": run.get("records"),
        "stop": run.get("stop"), "steps": run.get("steps"), "state": found.state,
        "reasons": list(found.reasons),
        "gross_collapse": found.unusable_reason == GROSS_COLLAPSE,
        "K_benign": _number_or_none(row.get("K_benign")),
        "K_harmful": _number_or_none(row.get("K_harmful")),
        "mmlu_accuracy": _number_or_none(row.get("benchmark_accuracy")),
        "factual_accuracy": _number_or_none(row.get("factual_accuracy")),
        "mmlu_loss_pp": _loss_pp(theta0["benchmark_accuracy"], row.get("benchmark_accuracy")),
        "factual_loss_pp": _loss_pp(theta0["factual_accuracy"], row.get("factual_accuracy")),
        "labels": (dict(run["labels"]) if isinstance(run.get("labels"), Mapping)
                   else {"missing": str(run.get("labels"))}),
        "receipt_sha256": run.get("receipt_sha256"),
        "h": _json(sd.harm(run)), "harmful_high": _json(sd.harm(run, "harmful_high")),
    }  # fmt: skip


# ------------------------------------------------------------------ h at K*


def kstar_cell(runs: Mapping[Key, Run], arm: str, seed: int, order: str,
               theta0: Mapping[str, float]) -> dict[str, Any]:  # fmt: skip
    """One arm, seed and order at K* (module docstring): its support state and reason, K_lo,
    K_hi, ln K_hi − ln K_lo, the weight, the two runs, and h* (the three classes and
    ``harmful_high``, pp), or ``missing``, why it has none."""
    pair = {rate: runs.get((arm, int(seed), order, rate)) for rate in RATE_NAMES}
    present = [run for run in pair.values() if run is not None]
    absent = [f"{arm} {order} {rate}: no run" for rate, run in pair.items() if run is None]
    endpoints = [endpoint(run, theta0) for run in present]
    support, block = sa.f_support(int(seed), endpoints)
    k_lo, k_hi = support["k_lo"], support["k_hi"]
    width = (math.log(k_hi) - math.log(k_lo) if _is_number(k_lo) and _is_number(k_hi)
             and k_lo > 0 and k_hi > 0 else None)  # fmt: skip
    cell: dict[str, Any] = {"arm": arm, "seed": int(seed), "order": order,
                            "state": support["state"], "reason": support["reason"], "k_lo": k_lo,
                            "k_hi": k_hi, "ln_width": width, "weight": support["weight"],
                            "lo": support["lo"], "hi": support["hi"], "h_star": None,
                            "missing": None}  # fmt: skip
    if block is None:
        detail = [f"{e.run_id}: {'; '.join(e.reasons)}" for e in endpoints if e.reasons]
        detail += absent
        cell["missing"] = f"{support['state']}: {support['reason']}" + (
            f" ({'; '.join(detail)})" if detail else "")  # fmt: skip
        return cell
    ends = [block["arms"][sa.ARM][end] for end in ("lo", "hi")]
    labels = {run["run_id"]: run.get("labels") for run in present}
    without = [f"{run_id}: {labels[run_id]}" for run_id in ends
               if not isinstance(labels[run_id], Mapping)]  # fmt: skip
    if without:
        cell["missing"] = "; ".join(without)
        return cell
    shares = sa.shares_at_kstar(block, {run_id: labels[run_id] for run_id in ends})
    cell["h_star"] = {name: 100.0 * value for name, value in shares.items()}
    return cell


def kstar_cells(runs: Mapping[Key, Run], seeds: Sequence[int],
                theta0: Mapping[str, float]) -> dict[Cell, dict[str, Any]]:  # fmt: skip
    """``kstar_cell`` of every arm, seed and order."""
    return {(arm, int(s), order): kstar_cell(runs, arm, int(s), order, theta0)
            for arm in ARMS for s in seeds for order in ORDERS}  # fmt: skip


def kstar_part(cells: Mapping[Cell, Mapping[str, Any]]) -> dict[str, Any]:
    """The cells at K*, those not supported and those supported without labels, each with its
    reason."""
    listed = [cells[key] for key in sorted(cells)]
    unsupported = [{"arm": c["arm"], "seed": c["seed"], "order": c["order"],
                    "state": c["state"], "reason": c["missing"]}
                   for c in listed if c["state"] != "supported"]  # fmt: skip
    unlabeled = [{"arm": c["arm"], "seed": c["seed"], "order": c["order"], "reason": c["missing"]}
                 for c in listed if c["state"] == "supported" and c["h_star"] is None]  # fmt: skip
    return {"k_star": K_STAR, "width_limit": WIDTH_LIMIT, "cells": listed,
            "unsupported": {"n": len(unsupported), "cells": unsupported},
            "without_labels": {"n": len(unlabeled), "cells": unlabeled}}  # fmt: skip


# ------------------------------------------------------------------ summaries over seeds


def bootstrap_mean(values: Sequence[float]) -> tuple[float | None, float | None]:
    """The percentile interval of the mean (module docstring, "The bootstrap"); (None, None)
    under two values."""
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return None, None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(x), size=(RESAMPLES, len(x)))
    lower, upper = np.percentile(x[idx].mean(axis=1), PERCENTILES)
    return float(lower), float(upper)


def mean_with_interval(values: Mapping[int, Value], seeds: Sequence[int]) -> dict[str, Any]:
    """A per-seed measure's mean over the seeds with a value and its bootstrap interval; every
    other seed listed with its reason."""
    used, excluded = [], []
    for seed in seeds:
        value = values.get(seed, "no value")
        if _is_number(value):
            used.append(int(seed))
        else:
            excluded.append({"seed": int(seed), "reason": str(value)})
    xs = [float(values[s]) for s in used]
    lower, upper = bootstrap_mean(xs)
    return {"n": len(used), "mean": statistics.fmean(xs) if xs else None, "lower": lower,
            "upper": upper, "level": LEVEL, "resamples": RESAMPLES, "rng_seed": BOOTSTRAP_SEED,
            "seeds": used, "excluded": excluded}  # fmt: skip


def label_noise(runs: Mapping[Key, Run], seeds: Sequence[int],
                measure: str = "harmful") -> dict[str, Any]:  # fmt: skip
    """The mean |Δ| that labels alone give two runs of equal harm (module docstring, "The
    label-noise value"), from the design's A runs with h; each other A run listed with its
    reason."""
    used, excluded = [], []
    for key in design(seeds):
        if key[0] != "A":
            continue
        count = sd.count_of(runs.get(key), measure)
        if _is_count(count):
            used.append(count)
        else:
            excluded.append({"seed": key[1], "order": key[2], "rate": key[3],
                             "reason": str(count)})  # fmt: skip
    h_bar = sum(used) / (DENOMINATOR * len(used)) if used else None
    value = (100.0 * math.sqrt(2.0 * h_bar * (1.0 - h_bar) / DENOMINATOR)
             * math.sqrt(2.0 / math.pi) if used else None)  # fmt: skip
    return {"measure": measure, "runs": len(used), "h_bar": h_bar, "value_pp": value,
            "formula": NOISE_FORMULA, "excluded": excluded}  # fmt: skip


def reading(abs_a: Mapping[str, Any], paired: Mapping[str, Any],
            noise: float | None) -> dict[str, Any]:  # fmt: skip
    """The registered reading (module docstring, "The reading"): ``order_sensitivity`` from mean
    |Δ_A|'s interval against the label-noise value, ``against_F`` from the paired difference's
    interval against 0."""
    lower = abs_a.get("lower")
    if lower is None or noise is None:
        sensitivity = UNDETERMINED
    else:
        sensitivity = ESTABLISHED if lower > noise else NOT_ESTABLISHED
    low, high = paired.get("lower"), paired.get("upper")
    if low is None or high is None:
        against = UNDETERMINED
    else:
        against = SMALLER if high < 0 else LARGER if low > 0 else NOT_TOLD_APART
    return {"order_sensitivity": sensitivity, "against_F": against, "label_noise_pp": noise,
            "abs_A_lower": lower, "paired_lower": low, "paired_upper": high}  # fmt: skip


def float_sign(value: Value) -> int | str:
    """Δ*'s sign (module docstring, "The direction"), or why it has none."""
    if not _is_number(value):
        return str(value)
    return 0 if abs(value) < ZERO_PP else (1 if value > 0 else -1)


def count_sign(value: int | str) -> int | str:
    """The sign of an exact count difference, or why it has none."""
    if not _is_count(value):
        return str(value)
    return (value > 0) - (value < 0)


def direction(sign_a: Mapping[int, int | str], sign_f: Mapping[int, int | str],
              seeds: Sequence[int], basis: str) -> dict[str, Any]:  # fmt: skip
    """The seeds where Δ_A and Δ_F have the same sign (module docstring, "The direction"), over
    the seeds with a sign under both arms; the others listed with their reasons."""
    both, excluded = [], []
    for s in seeds:
        a, f = sign_a.get(s, "no value"), sign_f.get(s, "no value")
        if _is_count(a) and _is_count(f):
            both.append(int(s))
        else:
            excluded.append({"seed": int(s), "reason": "; ".join(
                f"delta_{arm}: {v}" for arm, v in (("A", a), ("F", f))
                if not _is_count(v))})  # fmt: skip
    agreements = sum(1 for s in both if sign_a[s] == sign_f[s] != 0)
    opposite = sum(1 for s in both if sign_a[s] == -sign_f[s] != 0)
    return {"n": len(both), "agreements": agreements, "opposite": opposite,
            "with_zero": len(both) - agreements - opposite, "seeds": both,
            "excluded": excluded, "p_one_sided": sd.binomial_tail(agreements, len(both)),
            "basis": basis, "rule": DIRECTION_RULE}  # fmt: skip


def summary(delta_a: Mapping[int, Value], delta_f: Mapping[int, Value],
            sign_a: Mapping[int, int | str], sign_f: Mapping[int, int | str],
            seeds: Sequence[int], noise: float | None, basis: str) -> dict[str, Any]:  # fmt: skip
    """One basis (K*, or a fixed rate): over the seeds with Δ under both arms the means of
    |Δ_A|, |Δ_F| and |Δ_A| − |Δ_F| with their intervals, each other seed listed with its reason;
    the label-noise value; the reading; the direction."""
    abs_a: dict[int, Value] = {}
    abs_f: dict[int, Value] = {}
    paired: dict[int, Value] = {}
    for s in seeds:
        a, f = delta_a.get(s, "no value"), delta_f.get(s, "no value")
        if _is_number(a) and _is_number(f):
            abs_a[s], abs_f[s] = abs(float(a)), abs(float(f))
            paired[s] = abs_a[s] - abs_f[s]
        else:
            abs_a[s] = abs_f[s] = paired[s] = "; ".join(
                f"delta_{arm}: {v}" for arm, v in (("A", a), ("F", f))
                if not _is_number(v))  # fmt: skip
    means = {"abs_A": mean_with_interval(abs_a, seeds), "abs_F": mean_with_interval(abs_f, seeds),
             "paired": mean_with_interval(paired, seeds)}  # fmt: skip
    return {**means, "label_noise_pp": noise,
            "reading": reading(means["abs_A"], means["paired"], noise),
            "direction": direction(sign_a, sign_f, seeds, basis)}  # fmt: skip


# ------------------------------------------------------------------ one measure


def measure_part(runs: Mapping[Key, Run], cells: Mapping[Cell, Mapping[str, Any]],
                 seeds: Sequence[int], measure: str = "harmful") -> dict[str, Any]:  # fmt: skip
    """Everything of one measure (module docstring): per seed h*, Δ* and its sign under each arm,
    and at each fixed rate h, Δ(r), the count difference and its sign; the label-noise value;
    ``kstar`` (the summary over Δ*) and ``fixed_rate`` (the summary over Δ(r) per rate)."""
    if measure not in MEASURES:
        raise ValueError(f"no measure {measure!r}")
    hstar: dict[Cell, Value] = {}
    for key, cell in cells.items():
        hstar[key] = cell["h_star"][measure] if cell["h_star"] is not None else cell["missing"]
    keys = design(seeds)
    h = {key: sd.harm(runs.get(key), measure) for key in keys}
    counts = {key: sd.count_of(runs.get(key), measure) for key in keys}
    delta_star: dict[str, dict[int, Value]] = {arm: {} for arm in ARMS}
    delta: dict[str, dict[str, dict[int, Value]]] = {r: {arm: {} for arm in ARMS}
                                                    for r in RATE_NAMES}  # fmt: skip
    count_delta: dict[str, dict[str, dict[int, int | str]]] = {
        r: {arm: {} for arm in ARMS} for r in RATE_NAMES}  # fmt: skip
    for arm in ARMS:
        for s in seeds:
            s = int(s)
            delta_star[arm][s] = _difference(hstar[(arm, s, ORDERS[0])],
                                             hstar[(arm, s, ORDERS[1])], ORDERS)  # fmt: skip
            for r in RATE_NAMES:
                first, second = (arm, s, ORDERS[0], r), (arm, s, ORDERS[1], r)
                delta[r][arm][s] = _difference(h[first], h[second], ORDERS)
                a, b = counts[first], counts[second]
                count_delta[r][arm][s] = (a - b if _is_count(a) and _is_count(b) else "; ".join(
                    f"{name}: {v}" for name, v in zip(ORDERS, (a, b), strict=True)
                    if not _is_count(v)))  # fmt: skip
    signs_star = {arm: {s: float_sign(v) for s, v in delta_star[arm].items()} for arm in ARMS}
    signs = {r: {arm: {s: count_sign(v) for s, v in count_delta[r][arm].items()} for arm in ARMS}
             for r in RATE_NAMES}  # fmt: skip
    noise = label_noise(runs, seeds, measure)
    per_seed = []
    for s in (int(x) for x in seeds):
        entry: dict[str, Any] = {"seed": s, "kstar": {}, "fixed_rate": {r: {} for r in RATE_NAMES}}
        for arm in ARMS:
            entry["kstar"][arm] = {**{o: _json(hstar[(arm, s, o)]) for o in ORDERS},
                                   "delta": _json(delta_star[arm][s]),
                                   "sign": _json(signs_star[arm][s])}  # fmt: skip
            for r in RATE_NAMES:
                entry["fixed_rate"][r][arm] = {
                    **{o: _json(h[(arm, s, o, r)]) for o in ORDERS},
                    "delta": _json(delta[r][arm][s]),
                    "count_difference": _json(count_delta[r][arm][s]),
                    "sign": _json(signs[r][arm][s]),
                }  # fmt: skip
        per_seed.append(entry)
    value = noise["value_pp"]
    return {
        "measure": measure, "label_noise": noise, "per_seed": per_seed,
        "kstar": summary(delta_star["A"], delta_star["F"], signs_star["A"], signs_star["F"],
                         seeds, value, BASES["kstar"]),
        "fixed_rate": {r: summary(delta[r]["A"], delta[r]["F"], signs[r]["A"], signs[r]["F"],
                                  seeds, value, BASES["fixed_rate"]) for r in RATE_NAMES},
    }  # fmt: skip


# ------------------------------------------------------------------ the brackets


def brackets(runs: Mapping[Key, Run], seeds: Sequence[int]) -> dict[str, Any]:
    """Per A seed and order, its two rates' K_benign at step 25 and whether they bracket K*
    (module docstring, "The bracket share"); the share over the decided pairs."""
    pairs = []
    for s in seeds:
        for order in ORDERS:
            k: dict[str, Any] = {}
            why = []
            for rate in RATE_NAMES:
                run = runs.get(("A", int(s), order, rate))
                value = (run.get("row") or {}).get("K_benign") if run is not None else None
                if run is None:
                    why.append(f"{rate}: no run")
                elif not sd.complete(run):
                    end = run.get("records") or (
                        f"stop {run.get('stop')!r} after {run.get('steps')!r} updates")  # fmt: skip
                    why.append(f"{rate}: {run['run_id']}: {end}")
                elif not (_is_number(value) and value > 0):
                    why.append(f"{rate}: {run['run_id']}: K_benign {value!r}")
                else:
                    k[rate] = float(value)
                    continue
                k[rate] = {"missing": why[-1]}
            decided = not why
            low = min(k.values()) if decided else None
            high = max(k.values()) if decided else None
            pairs.append({"seed": int(s), "order": order, "K_benign": k,
                          "brackets": low <= K_STAR <= high if decided else None,
                          "reason": "; ".join(why) or None})  # fmt: skip
    decided = [p for p in pairs if p["brackets"] is not None]
    hits = sum(1 for p in decided if p["brackets"])
    return {"k_star": K_STAR, "pairs": len(pairs), "decided": len(decided), "bracketing": hits,
            "share": hits / len(decided) if decided else None, "per_pair": pairs}  # fmt: skip


# ------------------------------------------------------------------ the analysis


def analysis(runs: Mapping[Key, Run], seeds: Sequence[int],
             theta0: Mapping[str, float]) -> dict[str, Any]:  # fmt: skip
    """Every quantity of the analysis from the keyed runs' readings (module docstring): ``runs``
    (one line per keyed run), ``missing`` (the keys of the design without a run), ``kstar`` (the
    cells at K*), ``primary`` (the harmful measure) and ``secondary`` (``harmful_high`` and the
    bracket share)."""
    cells = kstar_cells(runs, seeds, theta0)
    return {
        "seeds": [int(s) for s in seeds],
        "rates": {arm: dict(RATES[arm]) for arm in ARMS},
        "orders": dict(adamw_orders.ORDER_OFFSETS),
        "runs": [run_entry(runs[key], theta0) for key in sorted(runs)],
        "missing": [{"arm": arm, "seed": s, "order": order, "rate": rate}
                    for arm, s, order, rate in design(seeds) if (arm, s, order, rate) not in runs],
        "kstar": kstar_part(cells),
        "primary": measure_part(runs, cells, seeds, "harmful"),
        "secondary": {"harmful_high": measure_part(runs, cells, seeds, "harmful_high"),
                      "brackets": brackets(runs, seeds)},
    }  # fmt: skip


# ------------------------------------------------------------------ helpers


def _difference(a: Value, b: Value, names: Sequence[str]) -> Value:
    """a − b, or the reasons either is missing, each by name."""
    if _is_number(a) and _is_number(b):
        return float(a) - float(b)
    return "; ".join(f"{name}: {v}" for name, v in zip(names, (a, b), strict=True)
                     if not _is_number(v))  # fmt: skip


def _loss_pp(theta0: float, accuracy: Any) -> float | None:
    return 100.0 * (theta0 - float(accuracy)) if _is_number(accuracy) else None


def _number_or_none(value: Any) -> float | None:
    return float(value) if _is_number(value) else None


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _json(value: Any) -> Any:
    return value if _is_number(value) else {"missing": str(value)}
