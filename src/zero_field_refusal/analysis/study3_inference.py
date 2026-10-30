"""Study 3's inference as pure functions: the paired studentized bootstrap-t over blocks, the
readings of an interval and of the three computations together, and their strict-JSON form.

docs/study3-preregistration.md, "Inference". The coverage check of the simulation, its
verification and the final analysis all call this one implementation ("The simulation's coverage
check uses the same rule").

**The rule.** For block differences d_1..d_n (n ≥ 2), the mean d̄, the sample SD s (ddof 1) and
SE = s/√n, each resample draws n blocks with replacement and takes
t* = (d̄* − d̄)/(s*/√n). A resample with s* = 0 (all drawn values equal) takes t* = 0 when its
common value is *exactly* d̄ (floating-point equality, no tolerance) and t* = −∞ or +∞ by the sign
of the common value minus d̄ otherwise. Quantiles are taken over all resamples, infinite ones
included. The equal-tailed interval at level L is [d̄ − t*₍₁₊L₎/₂·SE, d̄ − t*₍₁₋L₎/₂·SE].

**The order statistics** are type 1 at exact ranks: the zero-based rank of probability p among M
resamples is ceil(M·p) − 1, with p = (1 ∓ L)/2 computed in ``fractions.Fraction`` from
``Fraction(str(level))`` (so 0.975 is 975/1000, not its binary neighbour). ``numpy.quantile`` is
never used. ``tail_ranks`` returns (k_lo, k_hi); lower = d̄ − t*[k_hi]·SE, upper = d̄ − t*[k_lo]·SE.

**Streams.** ``stream`` names a SeedSequence by the bootstrap seed and keys: the full SHA-256
digest of the JSON key list [seed, "study3", *keys], the same derivation as the simulation's
``seed_sequence`` (a test compares them), written out here so that this module imports only numpy
and the standard library (no torch, no transformers). The indices are drawn
chunk by chunk from one ``default_rng(seed)``, which gives the same draws as a single call, so the
result does not depend on ``chunk`` (only the memory does).

**Undetermined.** One block, or an observed s = 0, gives an undetermined interval (NaN bounds, no
resampling).

**Units.** The differences are in whatever unit the caller passes (study 3 passes harm as a
fraction, so ±0.025 and ±0.05 are the registered 2.5 and 5 pp margins); the margins in
``reading`` are in that unit.

**Readings** (the table of "Inference"). A direction when the interval excludes 0
(``magnitude_lower`` is the bound closest to 0); a margin when it lies inside ±0.025 (the tightest)
or ±0.05, ends included; an infinite end has no margin; undetermined when there is neither.
``combined_reading`` takes the three computations of "Judge and audit" and reports the common
direction and, independently, the tightest margin common to all; an undetermined constituent
removes both.

**JSON.** ``Interval.to_json`` keeps every field by name; a float that is NaN becomes ``null`` and
±∞ become ``"+inf"`` / ``"-inf"``, so ``json.dumps(..., allow_nan=False)`` works.
``from_json`` accepts only those forms and raises ``ValueError`` on anything else.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, fields
from fractions import Fraction

import numpy as np

MARGINS = (0.025, 0.05)
_FLOAT_FIELDS = ("mean", "sd", "se", "level", "lower", "upper")
_INT_FIELDS = ("n", "resamples", "infinite_low", "infinite_high", "zero_se")


def stream(bootstrap_seed: int, *keys: object) -> np.random.SeedSequence:
    """The stream of study 3 named by ``keys`` under the bootstrap seed."""
    blob = json.dumps([bootstrap_seed, "study3", *keys]).encode()
    return np.random.SeedSequence(np.frombuffer(hashlib.sha256(blob).digest(), dtype=np.uint32))


def _is_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _level(level: float | Fraction) -> Fraction:
    if isinstance(level, Fraction):
        value = level
    elif isinstance(level, (float, int, np.floating)) and not isinstance(level, bool):
        value = Fraction(str(level))
    else:
        raise ValueError(f"the level must be a number in (0, 1), not {level!r}")
    if not 0 < value < 1:
        raise ValueError(f"the level must lie strictly between 0 and 1, not {level!r}")
    return value


def tail_ranks(level: float | Fraction, resamples: int) -> tuple[int, int]:
    """The zero-based type-1 ranks (k_lo, k_hi) of the tail probabilities (1 − L)/2 and
    (1 + L)/2 among ``resamples`` values: ceil(M·p) − 1 in exact arithmetic."""
    lvl = _level(level)
    if not _is_int(resamples) or resamples < 1:
        raise ValueError(f"resamples must be a positive integer, not {resamples!r}")
    m = int(resamples)
    return (math.ceil(m * (1 - lvl) / 2) - 1, math.ceil(m * (1 + lvl) / 2) - 1)


@dataclass(frozen=True)
class Interval:
    """One bootstrap-t interval: the observed n, mean, SD and SE, the level and resamples, the
    bounds (NaN when undetermined, ±∞ when a tail is infinite), the counts of resamples with
    t* = −∞ (``infinite_low``) and t* = +∞ (``infinite_high``), of resamples with s* = 0
    (``zero_se``, those two included) and whether the interval is undetermined."""

    n: int
    mean: float
    sd: float
    se: float
    level: float
    resamples: int
    lower: float
    upper: float
    infinite_low: int
    infinite_high: int
    zero_se: int
    undetermined: bool

    def __post_init__(self) -> None:
        """Refuse a record no computation can produce (a from_json record included)."""
        if not isinstance(self.undetermined, bool):
            raise ValueError(f"undetermined must be a boolean, not {self.undetermined!r}")
        lo_nan, hi_nan = math.isnan(self.lower), math.isnan(self.upper)
        if self.undetermined != (lo_nan and hi_nan) or lo_nan != hi_nan:
            raise ValueError("an interval is undetermined exactly when both bounds are NaN")
        if not self.undetermined:
            if self.lower > self.upper:
                raise ValueError(f"lower {self.lower!r} lies above upper {self.upper!r}")
            if self.lower == self.upper and math.isinf(self.lower):
                raise ValueError("both ends are infinite on the same side")
            if self.n < 2 or not math.isfinite(self.mean):
                raise ValueError("a determined interval needs n >= 2 and a finite mean")
            if not (math.isfinite(self.sd) and math.isfinite(self.se)) or min(self.sd, self.se) < 0:
                raise ValueError("a determined interval needs a finite SD and SE of at least 0")
            if self.lower == -math.inf and self.infinite_high < 1:
                raise ValueError("a -inf lower bound needs resamples with t* = +inf")
            if self.upper == math.inf and self.infinite_low < 1:
                raise ValueError("a +inf upper bound needs resamples with t* = -inf")
        if not 0 < self.level < 1:
            raise ValueError(f"the level must lie strictly between 0 and 1, not {self.level!r}")
        if self.n < 1 or self.resamples < 1:
            raise ValueError("n and resamples must be at least 1")
        if not (
            self.infinite_low >= 0
            and self.infinite_high >= 0
            and self.infinite_low + self.infinite_high <= self.zero_se <= self.resamples
        ):
            raise ValueError(
                "the counts must satisfy 0 <= infinite_low + infinite_high <= zero_se <= resamples"
            )

    def to_json(self) -> dict:
        out: dict = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _FLOAT_FIELDS:
                value = _float_to_json(value)
            out[f.name] = value
        return out

    @classmethod
    def from_json(cls, data: dict) -> Interval:
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
            raise ValueError("an interval record must have exactly the interval's fields")
        kwargs: dict = {}
        for name in _FLOAT_FIELDS:
            kwargs[name] = _float_from_json(data[name], name)
        for name in _INT_FIELDS:
            if not _is_int(data[name]):
                raise ValueError(f"{name} must be an integer, not {data[name]!r}")
            kwargs[name] = int(data[name])
        if not isinstance(data["undetermined"], bool):
            raise ValueError(f"undetermined must be a boolean, not {data['undetermined']!r}")
        kwargs["undetermined"] = data["undetermined"]
        return cls(**kwargs)


def _float_to_json(value: float) -> float | str | None:
    if math.isnan(value):
        return None
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return float(value)


def _float_from_json(value: object, name: str) -> float:
    if value is None:
        return math.nan
    if value == "+inf" and isinstance(value, str):
        return math.inf
    if value == "-inf" and isinstance(value, str):
        return -math.inf
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            out = float(value)
        except OverflowError:
            out = math.inf
        if math.isfinite(out):
            return out
    raise ValueError(f"{name} must be a finite number, null, '+inf' or '-inf', not {value!r}")


def bootstrap_t(
    d: Sequence[float] | np.ndarray,
    *,
    level: float | Fraction,
    resamples: int,
    seed: np.random.SeedSequence,
    chunk: int = 10_000,
) -> Interval:
    """The paired studentized bootstrap-t interval of the mean of ``d`` (module docstring)."""
    arr = np.asarray(d, dtype=float) if not isinstance(d, np.ndarray) else d.astype(float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("d must be a non-empty one-dimensional sequence of numbers")
    if not np.all(np.isfinite(arr)):
        raise ValueError("d must hold finite numbers only")
    if not isinstance(seed, np.random.SeedSequence):
        raise ValueError(f"seed must be a numpy SeedSequence, not {type(seed).__name__}")
    k_lo, k_hi = tail_ranks(level, resamples)
    if not _is_int(chunk) or chunk < 1:
        raise ValueError(f"chunk must be a positive integer, not {chunk!r}")
    n = int(arr.size)
    mean = float(arr.mean())
    lvl = float(_level(level))
    if n < 2:
        return Interval(n, mean, math.nan, math.nan, lvl, int(resamples), math.nan, math.nan,
                        0, 0, 0, True)  # fmt: skip
    sd = float(arr.std(ddof=1))
    if float(arr.max()) == float(arr.min()):
        return Interval(n, mean, 0.0, 0.0, lvl, int(resamples), math.nan, math.nan,
                        0, 0, 0, True)  # fmt: skip
    se = sd / math.sqrt(n)
    rng = np.random.default_rng(seed)
    parts = []
    inf_low = inf_high = zero_se = 0
    done = 0
    while done < resamples:
        size = min(int(chunk), int(resamples) - done)
        sample = arr[rng.integers(0, n, size=(size, n))]
        row_mean = sample.mean(axis=1)
        row_sd = sample.std(axis=1, ddof=1)
        flat = sample.max(axis=1) == sample.min(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (row_mean - mean) / (row_sd / math.sqrt(n))
        if flat.any():
            common = sample[flat, 0]
            t[flat] = np.where(common == mean, 0.0, np.where(common < mean, -np.inf, np.inf))
            zero_se += int(flat.sum())
            inf_low += int((common < mean).sum())
            inf_high += int((common > mean).sum())
        parts.append(t)
        done += size
    t_all = np.concatenate(parts)
    if np.isnan(t_all).any():
        raise ValueError("a resample gave a NaN t*")
    t_part = np.partition(t_all, [k_lo, k_hi])
    lower = mean - float(t_part[k_hi]) * se
    upper = mean - float(t_part[k_lo]) * se
    return Interval(n, mean, sd, se, lvl, int(resamples), lower, upper, inf_low, inf_high,
                    zero_se, False)  # fmt: skip


@dataclass(frozen=True)
class Reading:
    """What an interval (or three together) supports: a ``direction`` ("negative" or "positive"),
    a ``margin`` (0.025 or 0.05), ``magnitude_lower`` (the bound closest to 0 when a direction
    holds) and ``undetermined`` (neither direction nor margin)."""

    direction: str | None
    margin: float | None
    magnitude_lower: float | None
    undetermined: bool


UNDETERMINED = Reading(None, None, None, True)


def reading(interval: Interval) -> Reading:
    """The reading of one interval by the registered table."""
    if interval.undetermined:
        return UNDETERMINED
    lo, hi = interval.lower, interval.upper
    direction = "negative" if hi < 0 else "positive" if lo > 0 else None
    magnitude = min(abs(lo), abs(hi)) if direction else None
    margin = next((m for m in MARGINS if -m <= lo and hi <= m), None)
    if direction is None and margin is None:
        return UNDETERMINED
    return Reading(direction, margin, magnitude, False)


def combined_reading(intervals: Sequence[Interval]) -> Reading:
    """The strongest reading that holds in all three computations: the common direction (with the
    smallest ``magnitude_lower``) and, independently, the largest of the margins when every
    computation has one. An undetermined computation removes both."""
    if len(intervals) != 3:
        raise ValueError(f"exactly three intervals are combined, not {len(intervals)}")
    if len({(i.level, i.resamples, i.n) for i in intervals}) != 1:
        raise ValueError("the three computations must share the level, resamples and n")
    rs = [reading(i) for i in intervals]
    direction = rs[0].direction if len({r.direction for r in rs}) == 1 else None
    magnitude = min(r.magnitude_lower for r in rs) if direction else None  # type: ignore[type-var]
    margin = max(r.margin for r in rs) if all(r.margin is not None for r in rs) else None  # type: ignore[type-var]
    if direction is None and margin is None:
        return UNDETERMINED
    return Reading(direction, margin, magnitude, False)
