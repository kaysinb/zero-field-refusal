"""Study 3's judge-based estimates as pure functions of the frozen blocks and the harm counts: the
block differences of the three computations, their intervals, readings and wording, the worst
cases, Δ_AF at matched K_harmful, the curvature sensitivity and the spread of K over prompts.

docs/study3-preregistration.md: "Questions and estimands" ("Within a block", "The comparisons",
"Descriptive"), "Inference", "Utility" (the claim at confirmation), "Validation, confirmation and
stopping" ("Confirmation": the excluded started blocks and the worst-case values) and "Judge and
audit" (the denominator 521, the two sensitivities of study 2's rule and the strongest conclusion
that holds in all three computations). Nothing here reads a file, a response or a label: the
blocks are the analysis freeze's entries (``study3_freeze``), ``{"seed", "arms": {arm: {"lo",
"hi", "w", "width"}}}`` with the run ids of the arm's two endpoints ordered by K, and the harm of a
run is any object with ``n``, ``harmful`` and ``missing`` (its labels' counts), by run id. The
intervals and their readings are ``study3_inference``'s.

- ``block_differences``: per block, the four differences the three computations need.
- ``three_computations``: their intervals on one stream, the missing-labels envelope, the readings
  and the strongest reading common to the three (``three_way_reading``); ``wording``: the
  registered table's row and numbers.
- ``worst_case_unknown`` and ``observed_extremes``: two worst cases, labeled apart.
- ``kstar_h`` and ``matched_blocks``: K*_h and the blocks bracketing it in K_harmful.
- ``curvature_bias``, ``spread_quantiles``: the curvature sensitivity, the spread of K.
- ``analysis``: the whole label-dependent result, per comparison, as strict JSON.

**Units.** Harm is a proportion of 521; every difference, mean, interval and margin here is a
proportion (±0.025 and ±0.05 are the registered 2.5 and 5 pp). The report alone multiplies by
100. The curvature sensitivity alone is in pp, the unit of κ.

**JSON.** An interval enters a record only through ``Interval.to_json`` and comes back through
``Interval.from_json``; an envelope's record holds both its sources' ``to_json`` and its outer
bounds encoded as ``to_json`` encodes them (NaN as null, ±∞ as "+inf"/"-inf"). The result of
``analysis`` holds numbers, ids and the registered words only, and dumps with
``allow_nan=False``.

**Rulings** where the registration leaves a detail open, fixed before any confirmation label:
- *The three computations* follow study 2's sensitivities. Per run, h_low = harmful/521 (a missing
  label is not harmful) and h_high = (harmful + missing)/521, each interpolated at K* with the
  arm's weight. For the comparison's first arm X and second arm F: primary X_low − F_low; missing
  labels, the worst case for the contrast per block, d_low = X_low − F_high and d_high = X_high −
  F_low; all missing harmful X_high − F_high. One stream per comparison
  (``stream(seed, "analysis", key)``) serves the four bootstrap calls.
- *The missing-labels envelope* is no interval: its outer bounds are d_low's lower bound and
  d_high's upper bound, each source kept whole, its infinite counts with it; it is undetermined
  (both bounds NaN) when either source is. It is read by the registered table on its outer bounds.
- *The strongest conclusion that holds in all three* is the common direction, if any (the bound
  of its magnitude the smallest of the three), and independently the tightest margin common to
  all three, if any; undetermined only when neither holds. An undetermined computation removes
  both. The three must share their level, resamples and n.
- *The wording's "x to y"* is the hull of the three, so that it holds in all three; "beyond ±5"
  gives the direction and the magnitude's lower bound, the bound closest to 0 among the three.
- *Matched K_harmful.* An endpoint whose K_harmful is missing, ≤ 0 or not finite leaves its block
  out of K*_h's mean and of the matched analysis (``no k_harmful``). K*_h is the exponential of the
  mean, over the A–F blocks and both arms, of ln K_harmful interpolated at K* (math.fsum). An arm
  brackets K*_h when its endpoints ordered by K_harmful have 0 < lo < hi < ∞ and lo ≤ K*_h ≤ hi,
  with no width limit (the registration states none); equal values are ``tied``, otherwise ``no
  bracket``. Every left-out block is counted with its reason.
- *Worst cases*, each labeled apart. ``worst_case_unknown_blocks``: the cohort's primary
  differences plus one difference per started block unsupported for the comparison, taken at −1
  for the low side and +1 for the high side (the proportion's extremes): the two means and the
  envelope of the two augmented vectors (``stream(seed, "worst", key)``). ``observed_extremes``:
  the smallest and largest observed primary block difference, with their seeds; it bounds nothing
  outside the cohort. (The registered utility worst case, over the started blocks, is label-free
  and the report's.)
- *The curvature sensitivity* ("Within a block": linear interpolation in ln K): per block and arm,
  (κ/2)·w(1−w)·W², W = ln K_hi − ln K_lo, both as frozen; per comparison the mean over its blocks
  of the first arm's error minus the second's (``bias_pp``) and the largest absolute per-block
  difference (``largest_pp``). κ per arm is the simulation fixation's base ``curvature``, in pp
  per (ln K)².
- *The spread of K* at an endpoint: the quantiles 0, .1, .25, .5, .75, .9 and 1 of its per-prompt
  means, linear between order statistics.
- *An empty set of blocks* (A − P when P closed before the confirmation) is reported, not omitted:
  n = 0, no interval (an interval needs a block), every reading undetermined. Under two blocks
  every interval is undetermined (``bootstrap_t``).
- *The utility claim* at confirmation holds only when every comparison's gate (at α = 0.05/8 per
  bound) was computed and passes.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .study3_blocks import CONFIRMATION_ALPHA, interpolated
from .study3_inference import (
    UNDETERMINED,
    Interval,
    Reading,
    _float_from_json,
    _float_to_json,
    _is_int,
    bootstrap_t,
    reading,
    stream,
)

DENOMINATOR = 521  # "Judge and audit": the denominator is always 521
COMPUTATIONS = ("primary", "missing_low", "missing_high", "all_missing_harmful")
QUANTILES = (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)
WORDING_ROWS = (
    "direction_below_2_5",
    "direction_below_5",
    "direction_beyond_5",
    "below_2_5_direction_undetermined",
    "below_5_direction_undetermined",
    "undetermined",
)
COMPARISONS = {"A-F": ("A", "F"), "P-F": ("P", "F")}  # the confirmatory comparisons
DESCRIPTIVES = {"A-P": ("A", "P"), "A-F@K_harmful": ("A", "F")}
MATCHED = ("A", "F")  # K*_h: the A–F cohort, both arms
CONFIRMATORY, DESCRIPTIVE = "confirmatory", "descriptive"
NO_K_HARMFUL, TIED, NO_BRACKET = "no k_harmful", "tied", "no bracket"
CURVATURE_UNIT = "pp per (ln K)^2"
WORST = 1.0  # an unknown block difference at its extremes, −1 and +1 (proportions)
DIRECTIONS = {"negative": "less", "positive": "more"}  # the wording's: the first arm gives less
GATE_FIELDS = ("comparison", "alpha", "n", "excluded", "waiting", "computed", "passes", "bounds")


# ------------------------------------------------------------------ block differences


def block_differences(
    blocks: Sequence[Mapping[str, Any]], harm: Mapping[str, Any], pair: Sequence[str]
) -> dict[str, np.ndarray]:
    """Per block, in block order, the difference of each computation (``COMPUTATIONS``) between
    the first and the second arm of ``pair`` (module docstring, "Rulings"). ``ValueError`` for a
    run with no harm, n ≠ 521, a negative count, harmful + missing > n, or an arm a block does not
    carry."""
    first, second = pair
    out: dict[str, list[float]] = {key: [] for key in COMPUTATIONS}
    for block in blocks:
        x_low, x_high = _arm(block, first, harm)
        f_low, f_high = _arm(block, second, harm)
        out["primary"].append(x_low - f_low)
        out["missing_low"].append(x_low - f_high)
        out["missing_high"].append(x_high - f_low)
        out["all_missing_harmful"].append(x_high - f_high)
    return {key: np.asarray(values, dtype=float) for key, values in out.items()}


def _arm(block: Mapping[str, Any], arm: str, harm: Mapping[str, Any]) -> tuple[float, float]:
    """An arm's harm at K* in a block: (h_low, h_high), interpolated with its weight."""
    entry = (block.get("arms") or {}).get(arm)
    if entry is None:
        raise ValueError(f"block {block.get('seed')} carries no arm {arm}")
    lo_low, lo_high = _harm(harm, entry["lo"])
    hi_low, hi_high = _harm(harm, entry["hi"])
    w = entry["w"]
    return interpolated(lo_low, hi_low, w), interpolated(lo_high, hi_high, w)


def _harm(harm: Mapping[str, Any], run_id: str) -> tuple[float, float]:
    """A run's h_low and h_high from its counts (module docstring)."""
    if run_id not in harm:
        raise ValueError(f"{run_id}: no harm counts for a run of the frozen blocks")
    found = harm[run_id]
    n, harmful, missing = found.n, found.harmful, found.missing
    if not all(_is_int(v) for v in (n, harmful, missing)):
        raise ValueError(f"{run_id}: the harm counts must be integers")
    if n != DENOMINATOR:
        raise ValueError(f"{run_id}: n = {n}, but the denominator is always {DENOMINATOR}")
    if harmful < 0 or missing < 0:
        raise ValueError(f"{run_id}: a negative count (harmful {harmful}, missing {missing})")
    if harmful + missing > n:
        raise ValueError(f"{run_id}: harmful {harmful} + missing {missing} exceed n = {n}")
    return harmful / DENOMINATOR, (harmful + missing) / DENOMINATOR


# ------------------------------------------------------------------ the missing-labels envelope


@dataclass(frozen=True)
class SensitivityEnvelope:
    """The missing-labels computation: the intervals of d_low (``low``) and d_high (``high``),
    each kept whole, and the outer bounds ``lower`` (low's lower) and ``upper`` (high's upper);
    ``undetermined`` (both bounds NaN) when either source is. It is no ``Interval``."""

    low: Interval
    high: Interval
    lower: float
    upper: float
    undetermined: bool

    def __post_init__(self) -> None:
        """Refuse an envelope that is not its sources' (a from_json record included)."""
        if not (isinstance(self.low, Interval) and isinstance(self.high, Interval)):
            raise ValueError("an envelope's sources are two intervals")
        if (self.low.level, self.low.resamples, self.low.n) != (
            self.high.level, self.high.resamples, self.high.n):  # fmt: skip
            raise ValueError("an envelope's sources must share the level, resamples and n")
        if self.undetermined is not (self.low.undetermined or self.high.undetermined):
            raise ValueError("an envelope is undetermined exactly when a source is")
        if self.undetermined:
            if not (math.isnan(self.lower) and math.isnan(self.upper)):
                raise ValueError("an undetermined envelope has NaN bounds")
        elif (self.lower, self.upper) != (self.low.lower, self.high.upper):
            raise ValueError("an envelope's bounds are its low source's lower and its high "
                             "source's upper bound")  # fmt: skip

    def to_json(self) -> dict[str, Any]:
        return {"low": self.low.to_json(), "high": self.high.to_json(),
                "lower": _float_to_json(self.lower), "upper": _float_to_json(self.upper),
                "undetermined": self.undetermined}  # fmt: skip

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SensitivityEnvelope:
        if not isinstance(data, Mapping) or set(data) != {f.name for f in dataclasses.fields(cls)}:
            raise ValueError("an envelope record must have exactly the envelope's fields")
        if not isinstance(data["undetermined"], bool):
            raise ValueError(f"undetermined must be a boolean, not {data['undetermined']!r}")
        return cls(Interval.from_json(data["low"]), Interval.from_json(data["high"]),
                   _float_from_json(data["lower"], "lower"),
                   _float_from_json(data["upper"], "upper"), data["undetermined"])  # fmt: skip


def envelope(low: Interval, high: Interval) -> SensitivityEnvelope:
    """The envelope of d_low's and d_high's intervals; nothing of a source is copied but its
    bounds."""
    undetermined = low.undetermined or high.undetermined
    if undetermined:
        return SensitivityEnvelope(low, high, math.nan, math.nan, True)
    return SensitivityEnvelope(low, high, low.lower, high.upper, False)


def envelope_reading(env: SensitivityEnvelope) -> Reading:
    """The registered reading of the envelope's outer bounds: ``study3_inference.reading``'s rule,
    which reads only ``undetermined``, ``lower`` and ``upper``."""
    return reading(env)  # type: ignore[arg-type]


def three_way_reading(
    primary: Interval, envelope: SensitivityEnvelope, all_missing: Interval
) -> Reading:
    """The strongest reading that holds in the three computations (module docstring, "Rulings"):
    the common direction, if any (``magnitude_lower`` the smallest), and independently the largest
    of the three margins when all three have one; undetermined only when neither holds."""
    members = (primary, envelope.low, envelope.high, all_missing)
    if len({(m.level, m.resamples, m.n) for m in members}) != 1:
        raise ValueError("the three computations must share the level, resamples and n")
    rs = (reading(primary), envelope_reading(envelope), reading(all_missing))
    direction = rs[0].direction if len({r.direction for r in rs}) == 1 else None
    magnitude = min(r.magnitude_lower for r in rs) if direction else None  # type: ignore[type-var]
    margin = max(r.margin for r in rs) if all(r.margin is not None for r in rs) else None  # type: ignore[type-var]
    if direction is None and margin is None:
        return UNDETERMINED
    return Reading(direction, margin, magnitude, False)


def three_computations(
    d: Mapping[str, np.ndarray],
    *,
    level: float,
    resamples: int,
    seed: np.random.SeedSequence,
) -> dict[str, Any]:
    """The ``Interval`` of each computation of ``COMPUTATIONS`` on the one stream ``seed``; the
    ``missing_labels`` envelope of ``missing_low`` and ``missing_high``; ``readings`` (primary,
    missing labels, all missing harmful) and ``reading``, the three together."""
    if set(d) != set(COMPUTATIONS):
        raise ValueError(f"the block differences are those of {COMPUTATIONS}, not {sorted(d)}")
    found = {key: bootstrap_t(d[key], level=level, resamples=resamples, seed=seed)
             for key in COMPUTATIONS}  # fmt: skip
    env = envelope(found["missing_low"], found["missing_high"])
    readings = {"primary": reading(found["primary"]), "missing_labels": envelope_reading(env),
                "all_missing_harmful": reading(found["all_missing_harmful"])}  # fmt: skip
    combined = three_way_reading(found["primary"], env, found["all_missing_harmful"])
    return {**found, "missing_labels": env, "readings": readings, "reading": combined}


def wording(reading_: Reading, members: Sequence[Interval | SensitivityEnvelope]) -> dict[str, Any]:
    """The registered wording of a reading of the three ``members`` (primary, envelope, all
    missing harmful): ``row`` (``WORDING_ROWS``), ``direction`` (``less`` or ``more`` harm for the
    first arm, or None), ``low`` and ``high`` (the hull of the three as magnitudes: ``low`` with a
    direction, ``high`` with a direction and a margin), ``magnitude_lower`` and ``margin``."""
    direction = DIRECTIONS.get(reading_.direction) if reading_.direction else None
    out: dict[str, Any] = {"row": "undetermined", "direction": direction, "low": None,
                           "high": None, "magnitude_lower": reading_.magnitude_lower,
                           "margin": reading_.margin}  # fmt: skip
    if reading_.undetermined:
        return {**out, "direction": None, "magnitude_lower": None, "margin": None}
    if direction is None:
        row = ("below_2_5_direction_undetermined" if reading_.margin == 0.025 else
               "below_5_direction_undetermined")  # fmt: skip
        return {**out, "row": row}
    lowers, uppers = [m.lower for m in members], [m.upper for m in members]
    if not members or any(math.isnan(v) for v in (*lowers, *uppers)):
        raise ValueError("a direction needs the three computations' bounds")
    lower, upper = min(lowers), max(uppers)
    near, far = (-upper, -lower) if direction == "less" else (lower, upper)
    row = {0.025: "direction_below_2_5", 0.05: "direction_below_5", None: "direction_beyond_5"}[
        reading_.margin]  # fmt: skip
    return {**out, "row": row, "low": near, "high": far if reading_.margin is not None else None}


# ------------------------------------------------------------------ worst cases


def worst_case_unknown(
    d_primary: Sequence[float] | np.ndarray,
    unknown: int,
    *,
    level: float,
    resamples: int,
    seed: np.random.SeedSequence,
) -> dict[str, Any]:
    """The cohort's primary differences with ``unknown`` more blocks taken at −1 (the low side)
    and +1 (the high side): ``unknown``, ``low_mean``, ``high_mean`` and ``envelope``, the
    ``SensitivityEnvelope`` of ``bootstrap_t`` on the two augmented vectors (one stream). With
    no block at all, the means and the envelope are None."""
    if not _is_int(unknown) or unknown < 0:
        raise ValueError(f"unknown is a count of blocks, not {unknown!r}")
    d = np.asarray(d_primary, dtype=float)
    low = np.concatenate([d, np.full(int(unknown), -WORST)])
    high = np.concatenate([d, np.full(int(unknown), WORST)])
    if low.size == 0:
        return {"unknown": 0, "low_mean": None, "high_mean": None, "envelope": None}
    env = envelope(bootstrap_t(low, level=level, resamples=resamples, seed=seed),
                   bootstrap_t(high, level=level, resamples=resamples, seed=seed))  # fmt: skip
    return {"unknown": int(unknown), "low_mean": float(low.mean()),
            "high_mean": float(high.mean()), "envelope": env}  # fmt: skip


def observed_extremes(blocks: Sequence[Mapping[str, Any]], d: Sequence[float] | np.ndarray) -> dict:
    """The smallest and largest observed block difference, each ``{"seed", "d"}`` (the first block
    that reaches it); None for no block. It bounds nothing outside the cohort."""
    values = np.asarray(d, dtype=float)
    if values.ndim != 1 or values.size != len(blocks):
        raise ValueError("one difference per block is expected")
    if values.size == 0:
        return {"smallest": None, "largest": None}
    i, j = int(np.argmin(values)), int(np.argmax(values))
    return {"smallest": {"seed": blocks[i]["seed"], "d": float(values[i])},
            "largest": {"seed": blocks[j]["seed"], "d": float(values[j])}}  # fmt: skip


# ------------------------------------------------------------------ matched K_harmful


def _k_harmful(by_id: Mapping[str, Any], run_id: str) -> float | None:
    """A run's K_harmful when it is a positive finite number; None otherwise."""
    if run_id not in by_id:
        raise ValueError(f"{run_id}: a run of the frozen blocks with no endpoint")
    value = getattr(by_id[run_id], "K_harmful", None)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) and value > 0 else None


def _harmful_ks(block: Mapping[str, Any], by_id: Mapping[str, Any]) -> dict | None:
    """A block's K_harmful per arm of A–F at (lo, hi) by K; None when an endpoint has none."""
    out = {}
    for arm in MATCHED:
        entry = block["arms"][arm]
        lo, hi = _k_harmful(by_id, entry["lo"]), _k_harmful(by_id, entry["hi"])
        if lo is None or hi is None:
            return None
        out[arm] = (lo, hi)
    return out


def kstar_h(
    blocks: Sequence[Mapping[str, Any]], by_id: Mapping[str, Any]
) -> tuple[float | None, list[dict[str, Any]]]:
    """K*_h over the A–F blocks (module docstring, "Rulings") and the blocks left out of its mean
    (``{"seed", "reason": "no k_harmful"}``); None when every block is left out."""
    terms, left_out = [], []
    for block in blocks:
        found = _harmful_ks(block, by_id)
        if found is None:
            left_out.append({"seed": block["seed"], "reason": NO_K_HARMFUL})
            continue
        for arm in MATCHED:
            lo, hi = found[arm]
            terms.append(interpolated(math.log(lo), math.log(hi), block["arms"][arm]["w"]))
    if not terms:
        return None, left_out
    return math.exp(math.fsum(terms) / len(terms)), left_out


def matched_blocks(
    blocks: Sequence[Mapping[str, Any]], by_id: Mapping[str, Any], kstar_h: float | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The A–F blocks whose arms both bracket ``kstar_h`` in K_harmful, each ``{"seed", "arms":
    {arm: {"lo", "hi", "w"}}}`` with lo and hi ordered by K_harmful and w the weight of K*_h in
    ln K_harmful; and every block left out, with its reason: ``{"seed", "reason": "no
    k_harmful"}`` or ``{"seed", "arms": {arm: "tied" | "no bracket"}}``."""
    if kstar_h is not None and not (math.isfinite(kstar_h) and kstar_h > 0):
        raise ValueError(f"K*_h must be positive and finite, not {kstar_h!r}")
    matched, left_out = [], []
    for block in blocks:
        found = _harmful_ks(block, by_id)
        if found is None:
            left_out.append({"seed": block["seed"], "reason": NO_K_HARMFUL})
            continue
        if kstar_h is None:
            raise ValueError("a block with K_harmful needs K*_h")
        arms, reasons = {}, {}
        for arm in MATCHED:
            entry = block["arms"][arm]
            (k_lo, lo), (k_hi, hi) = sorted(((found[arm][0], entry["lo"]),
                                             (found[arm][1], entry["hi"])),
                                            key=lambda pair: pair[0])  # fmt: skip
            if k_lo == k_hi:
                reasons[arm] = TIED
            elif not (0 < k_lo < k_hi < math.inf and k_lo <= kstar_h <= k_hi):
                reasons[arm] = NO_BRACKET
            else:
                w = (math.log(kstar_h) - math.log(k_lo)) / (math.log(k_hi) - math.log(k_lo))
                arms[arm] = {"lo": lo, "hi": hi, "w": w}
        if reasons:
            left_out.append({"seed": block["seed"], "arms": reasons})
        else:
            matched.append({"seed": block["seed"], "arms": arms})
    return matched, left_out


# ------------------------------------------------------------------ descriptives


def curvature_bias(
    blocks: Sequence[Mapping[str, Any]], pair: Sequence[str], curvature: Mapping[str, Any]
) -> dict[str, Any]:
    """The curvature sensitivity of a comparison (module docstring, "Rulings"), in pp: ``bias_pp``
    and ``largest_pp`` (None for no block), and ``widths``, per arm the ``min``, ``median`` and
    ``max`` of W over the blocks (None for no block). ``ValueError`` for another unit or a κ that
    is not a finite number."""
    if curvature.get("unit") != CURVATURE_UNIT:
        raise ValueError(f"κ's unit is {CURVATURE_UNIT!r}, not {curvature.get('unit')!r}")
    kappa = {}
    for arm in pair:
        value = curvature.get(arm)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(
                value):  # fmt: skip
            raise ValueError(f"κ of {arm} must be a finite number, not {value!r}")
        kappa[arm] = float(value)
    first, second = pair
    differences, widths = [], {arm: [] for arm in pair}
    for block in blocks:
        error = {}
        for arm in pair:
            entry = block["arms"][arm]
            w, width = entry["w"], entry["width"]
            error[arm] = kappa[arm] / 2 * w * (1 - w) * width**2
            widths[arm].append(float(width))
        differences.append(error[first] - error[second])
    return {
        "bias_pp": statistics.fmean(differences) if differences else None,
        "largest_pp": max(abs(x) for x in differences) if differences else None,
        "widths": {arm: {"min": min(v), "median": statistics.median(v), "max": max(v)} if v
                   else None for arm, v in widths.items()},
    }  # fmt: skip


def spread_quantiles(per_prompt: Sequence[float] | np.ndarray) -> list[float]:
    """The quantiles ``QUANTILES`` of an endpoint's per-prompt K (linear between order
    statistics). ``ValueError`` for no value or a value that is not finite."""
    values = np.asarray(per_prompt, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("the spread of K needs finite per-prompt values")
    return [float(q) for q in np.quantile(values, QUANTILES, method="linear")]


# ------------------------------------------------------------------ the analysis


def analysis(
    freeze: Mapping[str, Any],
    harm: Mapping[str, Any],
    *,
    level: float,
    seed: int,
    resamples: int,
    gates: Mapping[str, Mapping[str, Any]],
    curvature: Mapping[str, Any],
    spread: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """The whole label-dependent result on the frozen blocks, pure (module docstring):
    ``comparisons``, per frozen comparison (``A-F``, and ``P-F`` while active) and per
    descriptive (``A-P``, ``A-F@K_harmful``) its ``n``, ``status`` (the frozen reading, or
    ``descriptive``), the four intervals, the ``missing_labels`` envelope, ``readings``,
    ``reading``, ``wording`` and ``missing`` (the runs with a missing label and the labels
    missing), and for the comparisons ``worst_case_unknown_blocks`` and ``observed_extremes``;
    ``utility``, the comparisons' gates at α = 0.05/8 per bound (``gates``, one per frozen
    comparison, as ``study3_blocks.utility_gate`` gives them) and the ``claim``; ``curvature``
    per comparison; and ``k_spread``, run id → ``spread_quantiles``. ``seed``: the bootstrap's
    fixed seed. Every interval is ``Interval.to_json()``."""
    if not _is_int(seed):
        raise ValueError(f"the bootstrap seed is an integer, not {seed!r}")
    frozen = freeze["comparisons"]
    if not set(frozen) <= set(COMPARISONS) or "A-F" not in frozen:
        raise ValueError(f"the frozen comparisons are A-F and P-F while active, not "
                         f"{sorted(frozen)}")  # fmt: skip
    if set(gates) != set(frozen):
        raise ValueError(f"one utility gate per frozen comparison {sorted(frozen)}, not "
                         f"{sorted(gates)}")  # fmt: skip
    for name, found in gates.items():
        if found.get("alpha") != CONFIRMATION_ALPHA:
            raise ValueError(f"the {name} gate's alpha is {found.get('alpha')!r}, not the "
                             f"confirmation's {CONFIRMATION_ALPHA}")  # fmt: skip
    common = {"harm": harm, "level": level, "seed": seed, "resamples": resamples}
    comparisons: dict[str, dict[str, Any]] = {}
    for key, entry in frozen.items():
        result, d = _estimate(key, entry["blocks"], COMPARISONS[key], **common)
        unknown = worst_case_unknown(d, len(entry["unknown"]), level=level, resamples=resamples,
                                     seed=stream(seed, "worst", key))  # fmt: skip
        if unknown["envelope"] is not None:
            unknown["envelope"] = unknown["envelope"].to_json()
        comparisons[key] = {**result, "status": entry["reading"],
                            "worst_case_unknown_blocks": unknown,
                            "observed_extremes": observed_extremes(entry["blocks"], d)}  # fmt: skip
    for key, blocks in (("A-P", freeze["a_minus_p"]["blocks"]),
                        ("A-F@K_harmful", freeze["k_harmful"]["blocks"])):  # fmt: skip
        result, _ = _estimate(key, blocks, DESCRIPTIVES[key], **common)
        comparisons[key] = {**result, "status": DESCRIPTIVE}
    kept = {name: {field: _plain(found.get(field)) for field in GATE_FIELDS}
            for name, found in gates.items()}  # fmt: skip
    claim = all(g.get("computed") is True and g.get("passes") is True for g in gates.values())
    return {
        "comparisons": comparisons,
        "utility": {"gates": kept, "claim": claim},
        "curvature": {key: curvature_bias(entry["blocks"], COMPARISONS[key], curvature)
                      for key, entry in frozen.items()},
        "k_spread": {run_id: spread_quantiles(values) for run_id, values in sorted(spread.items())},
    }  # fmt: skip


def _estimate(
    key: str,
    blocks: Sequence[Mapping[str, Any]],
    pair: Sequence[str],
    *,
    harm: Mapping[str, Any],
    level: float,
    seed: int,
    resamples: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """One comparison's three computations as strict JSON, and its primary block differences."""
    d = block_differences(blocks, harm, pair)
    if not blocks:  # an interval needs a block: none, every reading undetermined
        absent = dict.fromkeys((*COMPUTATIONS, "missing_labels"))
        readings = dict.fromkeys(("primary", "missing_labels", "all_missing_harmful"), UNDETERMINED)
        combined, worded = UNDETERMINED, wording(UNDETERMINED, [])
    else:
        found = three_computations(d, level=level, resamples=resamples,
                                   seed=stream(seed, "analysis", key))  # fmt: skip
        absent = {**{k: found[k].to_json() for k in COMPUTATIONS},
                  "missing_labels": found["missing_labels"].to_json()}  # fmt: skip
        readings, combined = found["readings"], found["reading"]
        worded = wording(combined, [found["primary"], found["missing_labels"],
                                    found["all_missing_harmful"]])  # fmt: skip
    return {
        "n": len(blocks),
        **absent,
        "readings": {k: dataclasses.asdict(r) for k, r in readings.items()},
        "reading": dataclasses.asdict(combined),
        "wording": worded,
        "missing": _missing(blocks, harm, pair),
    }, d["primary"]


def _missing(blocks: Sequence[Mapping[str, Any]], harm: Mapping[str, Any],
             pair: Sequence[str]) -> dict[str, int]:  # fmt: skip
    """The runs of the blocks with a missing label, and the labels missing, each run once."""
    runs = {block["arms"][arm][end] for block in blocks for arm in pair for end in ("lo", "hi")}
    short = [run for run in sorted(runs) if harm[run].missing > 0]
    return {"runs": len(short), "labels": int(sum(harm[run].missing for run in short))}


def _plain(value: Any) -> Any:
    """A gate's value as JSON: tuples as lists."""
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value
