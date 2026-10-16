"""Simulation of study 2's short-regime confirmation, as redirected by amendment 3.

Recalibrates and validates the sequential procedure of the primary contrast between AdamW-25 and
frozen Adam-25 at K\\*, and measures the bias of interpolating harm to K\\*. Pure numpy on the CPU;
it reads no run directory and no harm data. Spec: the amendment-3 simulation spec of 2026-10-16
with its revision 1 (after review fa3: the interpolation design only) and revision 2 (curves on
the population scale, look designs, the interpolation estimand and the bias allowance).

Notation. Arms: A = AdamW-25 and F = frozen Adam-25 (the primary contrast), P = pre-warmed
AdamW-25 (secondary). x = ln(K_benign/K\\*), K\\* = √40. Probabilities are fractions (1 pp =
0.01). h̄_a(x) is arm a's population curve: the mean of h over prompts and block effects.

Estimands. The primary estimand is the interpolation-based contrast: the expected interpolated
contrast E[Δ̂_b] through the fixed rates (estimated by the mean of Δ̂_b over every supported
block of a scenario). The pointwise contrast Δ(K\\*) = h̄_A(0) − h̄_F(0) is the scenario's Δ. Their
difference is the paired interpolation bias; the allowance b is its largest absolute value over
the curve families × selection misses at w_max 0.35 and 0.47 (``bias_allowance``). Pointwise
readings of an interval use b: a direction needs the interval to exclude [−b, b], equivalence
needs it inside ±(5 pp − b).

Design (``interp``). Each arm runs three fixed rates for exactly 25 steps. Their endpoints sit at
the nominal positions x_r = (−w, 0, +w), w = ln √(U/5) = 0.235 for U = 8, or, for
``positions="grid"`` (design ``interp_grid``), at the coarse B = 5 discovery brackets: A (−0.39,
0.17, 0.57), F (−0.23, 0.27, 0.73), with P at the nominal positions.

One replicate is one confirmation study:

- Prompts: α_i ~ N(0, σ_α²), i = 1..521, drawn once (the prompt set is fixed across blocks).
- Selection miss, drawn once from U(−τ, τ): ``common``, one shift for every arm and rate; ``arm``,
  one shift per arm shared by its three rates; ``rate``, one per arm and rate.
- Blocks b = 1, 2, … up to the cap of 80, independent given the above:
  - a stream effect β_b ~ N(0, σ_s²) shared by the arms;
  - speeds u_{b,a} ~ N(0, σ_u²), correlation ρ_u between any two arms;
  - an arm effect d_{b,a} whose shape is set by the scenario. Its driving variable w_{b,a} is
    standard normal with correlation ρ_uh with u_{b,a}, a Gaussian copula (the correlation of d
    with u is exactly ρ_uh only for ``normal``):
    - ``normal``: d = σ_d·w in every arm;
    - ``skewed``: d_A = σ_d·(G − 2)/√2, G the Gamma(2) quantile of Φ(w); d_F = d_P = 0. Only A
      carries it: independent skewed effects in both arms would make the paired difference
      symmetric;
    - ``rare_centered``: d_A = M·(B − π), B = 1[w > Φ⁻¹(1 − π)], π = 0.1; d_F = d_P = 0;
    - ``rare_only``: d = m·B, in A when Δ ≥ 0 and in F and P when Δ < 0 (from h ≈ 0.15 only a
      positive shift can carry a rare 5 pp difference); m solves the population Δ, so the mean
      difference sits in the rare blocks only.
  - endpoint positions x_{b,a,r} = x_r + τ_{a,r} + u_{b,a} + e_{b,a,r}, e ~ N(0, σ_e²);
  - technical failures, per run, with probability f, or (``failure_correlated``) with probability
    logistic(c_f + β_b/σ_s), so that high-harm streams fail more, c_f giving the marginal f.
- Curves: logit h_a(x) = c_a + s_a·x + q_a·x² + γ_a·max(x, 0) + α_i + β_b + d_{b,a}. A family
  (``FAMILIES``) gives each arm's target population curve, a logistic
  σ(logit(baseline) + s·x + q·x² + γ·max(x, 0)): ``study1`` (A s = 1.20, q = −0.13; F s = 1.39,
  q = −0.24: h̄′ 0.153 / 0.177 and h̄″ 0.096 / 0.112 at h̄(0) = 0.15), ``stress_slopes`` (A 4·x,
  F 2·x), ``kink`` (1.6·x + 1.5·max(x, 0) in A, 1.6·x in F) and ``high`` (``study1``'s logit
  coefficients around h̄(0) = 0.4). The per-prompt coefficients (c, s, q, γ) are fitted so that
  h̄_a, averaged over prompts and block effects (the arm's own effect distribution included),
  matches the target on x ∈ [−0.6, 0.6] by least squares on a grid (γ is free only where the
  target has a kink; ``rare_only`` fits without its rare shift). P follows F: same target, same
  effect distribution. On top, c_F makes h̄_F(0) the baseline exactly and c_A makes
  h̄_A(0) − h̄_F(0) = Δ; c_P = c_F. Under ``rare_only`` c_A = c_F = c_P make the unshifted
  population rate the baseline.
- Labels: y = 1[Z < Φ⁻¹(h)], Z = √ρ_b·Z_{b,i} + √(ρ_a − ρ_b)·Z_{b,a,i} + √(1 − ρ_a)·Z_{b,a,r,i}:
  a prompt tends to come out alike across an arm's rates (ρ_a) and across arms (ρ_b). An
  endpoint's harm is the mean of y over the prompts.

Estimator and support, per block and arm. The bracket is the nearest pair of non-failed
endpoints with x_lo < 0 ≤ x_hi, and ĥ(K\\*) is linear in x between them. The arm is unsupported
for one cause, taken in this order: ``bracketing`` (even all three endpoints fail to bracket 0),
``technical`` (the non-failed ones do not), ``width`` (the bracket is wider than w_max). A block
supports the primary contrast when A and F are supported; Δ̂_b = ĥ_A − ĥ_F. The secondary
contrasts P − F and A − P use the blocks supported in all three arms. Per block, the primary
contrast's cause is the first in that order among A's and F's.

Spread calibration. σ_d (M for ``rare_centered``) is set per scenario at Δ = 0 so that the
observed paired SD, the square root of the within-replicate variance of Δ̂_b over supported
blocks averaged over replicates, equals the scenario's target. Response noise is part of it. If
the noise alone exceeds the target, σ_d = 0 and the attained SD is reported. ``rare_only`` has no
spread to calibrate: its SD follows from Δ.

Sequential procedure, per look design (``LOOK_DESIGNS``): ``24_50`` (the planning one),
``36_60``, ``50`` (one terminal look) and ``24_50_eq_terminal`` (24/50 with equivalence
allowed only at the terminal look; a direction may still stop at look 1). Blocks start in order.
A look happens when the number of supported blocks reaches its size (look 1 of a two-look design
later by the scenario's ``postpone``); a look that lacks its minimum is postponed, within the cap
of 80 started blocks. At the cap the terminal look happens with whatever is supported (cap
exhaustion); if even look 1's size was not reached, that is the only look. The interval at a look
is mean ± t_{n−1, 1−α_k/2}·max(sd, s_min)/√n, and none (inconclusive) with fewer than 2 blocks.
It gives equivalence (inside ±5 pp, bounds included, as in Gate 2), a direction (excluding 0
strictly), both, or neither; the ±8 pp reading is reported and never stops. The study stops at
look 1 on equivalence (where the design allows it) or a direction; otherwise the terminal look
decides. Equivalence is not reported at a look-1 stop of ``24_50_eq_terminal``. If look 1 is
reached only at the cap, looks 1 and 2 use the same blocks. The secondary contrasts are read at
the look where the primary stopped, with two-sided 98.75 % intervals on the three-arm cohort and
the same floor. At the stopping look, the finite-cohort bounds of the started cohort are
[(S − m)/N, (S + m)/N]: N started blocks, m of them unsupported, S the sum of the supported Δ̂_b.

Levels. A two-look design uses the O'Brien–Fleming-shaped pair from Lan–DeMets spending of an
overall two-sided α at its information fraction n₁/n₂ (``obf_levels``); α = 0.05 at 24/50 gives
the planning levels 99.53 % and 95.19 %. A one-look design uses 1 − α.

Randomness. Replicate ``rep`` of a scenario draws from ``derive_seed(base_seed, stream, scenario
name, rep)``; the stream names the run (``calibration``, ``validation``, …), so a validation run
uses draws independent of the calibration's. Blocks are generated until 60 are supported (the
largest look) or 80 have started, so every look design, α and floor is evaluated on the same
replicates. The spread calibration uses ``derive_seed(base_seed, "spread", key name)`` in every
run, so a scenario's σ_d does not depend on the stream.
"""

from __future__ import annotations

import functools
import math
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares
from scipy.special import expit, gammaincinv, logit, ndtr, ndtri, roots_genlaguerre
from scipy.stats import norm
from scipy.stats import t as student_t

from ..seeding import derive_seed

ARMS = ("A", "F", "P")  # AdamW-25, frozen Adam-25, pre-warmed AdamW-25
SECONDARY_CONTRASTS = ("P-F", "A-P")
PROMPTS = 521
K_STAR = math.sqrt(40)
U = 8
HALF_SPAN = math.log(math.sqrt(U / 5))  # w = 0.235
NOMINAL = (-HALF_SPAN, 0.0, HALF_SPAN)
POSITIONS = {
    "nominal": (NOMINAL, NOMINAL, NOMINAL),
    "grid": ((-0.39, 0.17, 0.57), (-0.23, 0.27, 0.73), NOMINAL),  # B = 5 discovery medians
}
MISSES = ("common", "arm", "rate")
SHAPES = ("normal", "skewed", "rare_centered", "rare_only")
RARE = 0.1  # π
RARE_THRESHOLD = float(ndtri(1 - RARE))
GAMMA_SHAPE = 2.0  # k of the skewed shape
FAILURE_SLOPE = 1.0  # logit units per SD of β_b (technical failures)
FIT_GRID = np.linspace(-0.6, 0.6, 25)  # x where population curves are fitted to their targets


def _shape(x: np.ndarray, s: float, q: float, g: float) -> np.ndarray:
    return s * x + q * x**2 + g * np.maximum(x, 0.0)


@dataclass(frozen=True)
class Family:
    """Target population curves: h̄_a(x) = σ(logit(baseline) + s_a·x + q_a·x² + γ_a·max(x, 0)) per
    arm (A, F, P), with h̄_F(0) = baseline."""

    slope: tuple[float, float, float]
    curvature: tuple[float, float, float]
    kink: tuple[float, float, float]
    baseline: float

    def coefficients(self, arm: int) -> tuple[float, float, float]:
        return self.slope[arm], self.curvature[arm], self.kink[arm]

    def target(self, arm: int, x: np.ndarray) -> np.ndarray:
        return expit(logit(self.baseline) + _shape(np.asarray(x), *self.coefficients(arm)))


STUDY1 = dict(slope=(1.20, 1.39, 1.39), curvature=(-0.13, -0.24, -0.24), kink=(0.0, 0.0, 0.0))
FAMILIES = {
    "study1": Family(**STUDY1, baseline=0.15),
    "stress_slopes": Family((4.0, 2.0, 2.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.15),
    "kink": Family((1.6, 1.6, 1.6), (0.0, 0.0, 0.0), (1.5, 0.0, 0.0), 0.15),
    "high": Family(**STUDY1, baseline=0.40),
}


def obf_levels_uncached(alpha: float, fraction: float) -> tuple[float, float]:
    """Two-sided levels of two looks: Lan–DeMets O'Brien–Fleming spending of ``alpha``.

    Look 1 spends 2 − 2Φ(z_{1−α/2}/√t) at information fraction t; the look-2 bound then gives an
    overall two-sided crossing probability α for the canonical joint normal statistics
    (correlation √t).
    """
    spent = 2 * norm.sf(norm.isf(alpha / 2) / math.sqrt(fraction))
    c1 = norm.isf(spent / 2)
    rho = math.sqrt(fraction)
    scale = math.sqrt(1 - rho**2)

    def inside(c2: float) -> float:
        def density(z: float) -> float:
            return norm.pdf(z) * (
                norm.cdf((c2 - rho * z) / scale) - norm.cdf((-c2 - rho * z) / scale)
            )

        return quad(density, -c1, c1, epsabs=1e-13, epsrel=1e-12)[0]

    c2 = brentq(lambda c: inside(c) - (1 - alpha), 0.5, 10.0, xtol=1e-12)
    return float(1 - spent), float(1 - 2 * norm.sf(c2))


@functools.cache
def obf_levels(alpha: float, fraction: float = 24 / 50) -> tuple[float, float]:
    return obf_levels_uncached(alpha, fraction)


@dataclass(frozen=True)
class LookDesign:
    name: str
    sizes: tuple[int, ...]  # supported blocks at each look; the last look is terminal
    equivalence_at_first: bool = True  # a two-look design may stop on equivalence at look 1

    def levels(self, alpha: float) -> tuple[float, ...]:
        """OBF spending of α over two looks, 1 − α for one."""
        if len(self.sizes) == 1:
            return (1 - alpha,)
        return obf_levels(alpha, self.sizes[0] / self.sizes[1])

    def realized(self, postpone: int) -> tuple[int, ...]:
        """Look sizes with look 1 postponed by ``postpone`` supported blocks."""
        if len(self.sizes) == 1:
            return self.sizes
        return (self.sizes[0] + postpone, self.sizes[1])


LOOK_DESIGNS = (
    LookDesign("24_50", (24, 50)),
    LookDesign("36_60", (36, 60)),
    LookDesign("50", (50,)),
    LookDesign("24_50_eq_terminal", (24, 50), equivalence_at_first=False),
)
DESIGNS_BY_NAME = {d.name: d for d in LOOK_DESIGNS}
PLANNING_DESIGN = "24_50"
GENERATE_UNTIL = max(max(d.sizes) for d in LOOK_DESIGNS)  # supported blocks

CAP = 80
LOOKS = DESIGNS_BY_NAME[PLANNING_DESIGN].sizes
PLANNING_LEVELS = (0.9953, 0.9519)
PLANNING_ALPHA = 0.05
MARGIN = 0.05  # ±5 pp, the primary margin
WIDE_MARGIN = 0.08  # the ±8 pp reading
SECONDARY_LEVEL = 0.9875  # Bonferroni over 2 contrasts × 2 looks
SE_FLOORS = (0.0, 0.01, 0.02)
ALPHAS = tuple(round(0.02 + 0.0025 * i, 4) for i in range(25))  # 0.02 … 0.08
TARGET_SDS = (0.03, 0.05, 0.10, 0.15)
BINDING_SDS = (0.03, 0.05, 0.10)
DELTAS = (0.0, 0.025, -0.025, 0.05, -0.05, 0.055, -0.055, 0.08, -0.08)
W_MAXES = (0.30, 0.35, 0.47)
ALLOWANCE_W_MAXES = (0.35, 0.47)
K_SPREADS = ((0.07, 0.03), (0.10, 0.05), (0.12, 0.06))  # (σ_u, σ_e)
MAX_FALSE_DIRECTION = 0.025  # per side, Δ = 0
MAX_FALSE_EQUIVALENCE = 0.05  # |Δ| = 5 pp
POWER_TARGET = 0.8
BOUNDARY_REPS = 20_000
OTHER_REPS = 5_000
SMOKE_REPS = 2_000
CHUNK_BLOCKS = 10  # blocks generated per draw while a replicate runs
HERMITE_NODES = 80
LAGUERRE_NODES = 40

NOT_STARTED, SUPPORTED, BRACKETING, TECHNICAL, WIDTH = -1, 0, 1, 2, 3
CAUSES = {"bracketing": BRACKETING, "technical": TECHNICAL, "width": WIDTH}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

# Scenario fields that appear in its name when they differ from the default.
TAGS = {
    "sigma_alpha": "sa",
    "sigma_s": "ss",
    "miss": "miss",
    "tau": "tau",
    "sigma_u": "su",
    "sigma_e": "se",
    "rho_u": "ru",
    "rho_uh": "ruh",
    "rho_b": "rb",
    "rho_a": "ra",
    "failure": "f",
    "failure_correlated": "fcorr",
    "w_max": "w",
    "postpone": "postpone",
}


@dataclass(frozen=True)
class Scenario:
    positions: str = "nominal"  # "nominal" | "grid"
    delta: float = 0.0  # pointwise population Δ(K*)
    sd: float = 0.05  # target paired SD (unused by rare_only)
    shape: str = "normal"
    family: str = "study1"
    sigma_alpha: float = 1.5
    sigma_s: float = 0.5
    miss: str = "rate"
    tau: float = 0.05
    sigma_u: float = 0.07
    sigma_e: float = 0.03
    rho_u: float = 0.0
    rho_uh: float = 0.0
    rho_b: float = 0.3
    rho_a: float = 0.7
    failure: float = 0.0
    failure_correlated: bool = False
    w_max: float = 0.35
    postpone: int = 0  # look 1 of two-look designs happens this many supported blocks later
    reps: int = OTHER_REPS
    group: str = field(default="", compare=False)  # report grouping only

    def __post_init__(self) -> None:
        gaps = [d.sizes[1] - d.sizes[0] for d in LOOK_DESIGNS if len(d.sizes) == 2]
        checks = {
            "positions": self.positions in POSITIONS,
            "shape": self.shape in SHAPES,
            "family": self.family in FAMILIES,
            "miss": self.miss in MISSES,
            "rho_b, rho_a": 0 <= self.rho_b <= self.rho_a <= 1,
            "postpone": 0 <= self.postpone < min(gaps),
            "failure": 0 <= self.failure < 1,
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"Invalid scenario fields {bad}: {self}")

    @property
    def design(self) -> str:
        return "interp" if self.positions == "nominal" else "interp_grid"

    @property
    def name(self) -> str:
        sd = "na" if self.shape == "rare_only" else f"{round(100 * self.sd, 6):g}"
        parts = [self.design, self.shape, self.family, f"sd{sd}", f"d{_pp(self.delta)}"]
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name not in TAGS or value == f.default:
                continue
            tag = TAGS[f.name]
            if isinstance(value, bool):
                parts.append(tag)
            elif isinstance(value, str):
                parts.append(f"{tag}-{value}")
            else:
                parts.append(f"{tag}{value:g}")
        return "_".join(parts)


def _pp(value: float) -> str:
    return f"{round(100 * value, 6):+g}"


@dataclass(frozen=True)
class Parameters:
    """What a scenario's draws need besides the scenario itself."""

    offsets: tuple[float, float, float]  # c_A, c_F, c_P (logit)
    coefficients: tuple[tuple[float, float, float], ...]  # per-prompt (s, q, γ) per arm
    fit_rmse: tuple[float, float, float]  # of each arm's population curve against its target
    spread: float  # σ_d (normal, skewed), M (rare_centered) or m (rare_only)
    shifted: tuple[bool, bool, bool]  # the arms that carry the arm effect
    failure: tuple[float, float]  # intercept and slope of the failure logit in β_b/σ_s
    truth: dict[str, float]  # pointwise population contrasts at K*: "A-F", "P-F", "A-P"


def curve_offset(
    coefficients: Sequence[tuple[float, float, float]], x: np.ndarray, arms: Sequence[int]
) -> np.ndarray:
    """s·x + q·x² + γ·max(x, 0), with arm ``arms[j]`` on axis -2 of ``x`` (endpoints last)."""
    s, q, g = (np.array([coefficients[a][k] for a in arms])[:, None] for k in range(3))
    return _shape(x, s, q, g)


# Population quantities (quadrature)


@functools.cache
def _hermite() -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite_e.hermegauss(HERMITE_NODES)
    return nodes, weights / weights.sum()


@functools.cache
def _laguerre() -> tuple[np.ndarray, np.ndarray]:
    # Gamma(k) has density x^(k−1)·e^(−x)/Γ(k): generalized Gauss–Laguerre with α = k − 1.
    nodes, weights = roots_genlaguerre(LAGUERRE_NODES, GAMMA_SHAPE - 1)
    return nodes, weights / weights.sum()


NO_EFFECT = (np.zeros(1), np.ones(1))


def effect_distribution(shape: str, spread: float) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of a shifted arm's effect d (logit scale)."""
    if shape == "normal":
        nodes, weights = _hermite()
        return spread * nodes, weights
    if shape == "skewed":
        nodes, weights = _laguerre()
        return spread * (nodes - GAMMA_SHAPE) / math.sqrt(GAMMA_SHAPE), weights
    if shape == "rare_centered":
        return spread * (np.array([1.0, 0.0]) - RARE), np.array([RARE, 1 - RARE])
    if shape == "rare_only":
        return np.array([spread, 0.0]), np.array([RARE, 1 - RARE])
    raise ValueError(f"Unknown shape {shape!r}; expected one of {SHAPES}")


def population_values(
    sigma_v: float,
    offset: float,
    coefficients: tuple[float, float, float],
    effect: tuple[np.ndarray, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """h̄(x) = E[σ(c + f(x) + v + d)], v ~ N(0, σ_v²) (prompt + stream), d ~ ``effect``."""
    nodes, weights = _hermite()
    values, probabilities = effect
    eta = offset + _shape(np.asarray(x, dtype=float), *coefficients)
    inner = expit(eta[:, None, None] + sigma_v * nodes[None, :, None] + values[None, None, :])
    return inner @ probabilities @ weights


def _sigma_v(scenario: Scenario) -> float:
    return math.hypot(scenario.sigma_alpha, scenario.sigma_s)


def population_rate(
    scenario: Scenario, offset: float, effect: tuple[np.ndarray, np.ndarray]
) -> float:
    """h̄(0) for logit offset c."""
    return float(population_values(_sigma_v(scenario), offset, (0.0, 0.0, 0.0), effect, [0.0])[0])


@functools.cache
def fit_coefficients(
    family: str, arm: int, sigma_v: float, effect: tuple[str, float] | None
) -> tuple[tuple[float, float, float], float]:
    """Per-prompt (s, q, γ) whose population curve best matches the arm's target on
    ``FIT_GRID`` (least squares; the offset is fitted too), and the fit's RMSE.

    ``effect`` is the arm's effect (shape, spread), or None. γ is free only where the target has
    a kink.
    """
    target = FAMILIES[family]
    goal = target.target(arm, FIT_GRID)
    distribution = NO_EFFECT if effect is None else effect_distribution(*effect)
    kinked = target.kink[arm] != 0

    def unpack(p: np.ndarray) -> tuple[float, tuple[float, float, float]]:
        return p[0], (p[1], p[2], p[3] if kinked else 0.0)

    def residual(p: np.ndarray) -> np.ndarray:
        offset, coefficients = unpack(p)
        return population_values(sigma_v, offset, coefficients, distribution, FIT_GRID) - goal

    s, q, g = target.coefficients(arm)
    start = [logit(target.baseline), s, q] + ([g] if kinked else [])
    fit = least_squares(residual, start, method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _, coefficients = unpack(fit.x)
    return tuple(float(v) for v in coefficients), float(np.sqrt(np.mean(fit.fun**2)))


def _solve_offset(rate: Callable[[float], float], target: float) -> float:
    if not 0 < target < 1:
        raise ValueError(f"Population rate {target} outside (0, 1)")
    return float(brentq(lambda c: rate(c) - target, -40.0, 40.0, xtol=1e-12))


def _logistic_intercept(marginal: float, slope: float) -> float:
    """c with E[logistic(c + slope·z)] = marginal, z ~ N(0, 1); −∞ for a marginal of 0."""
    if marginal == 0:
        return -math.inf
    nodes, weights = _hermite()
    return float(
        brentq(lambda c: weights @ expit(c + slope * nodes) - marginal, -40.0, 40.0, xtol=1e-12)
    )


def _effects(scenario: Scenario, spread: float, shifted: Sequence[bool]) -> list:
    return [effect_distribution(scenario.shape, spread) if on else NO_EFFECT for on in shifted]


def solve_parameters(scenario: Scenario, spread: float | None = None) -> Parameters:
    """Curve coefficients, offsets, the arm effect and the failure law of a scenario.

    ``spread`` is the calibrated σ_d or M; ``rare_only`` ignores it and solves m from Δ.
    """
    s = scenario
    baseline = FAMILIES[s.family].baseline
    sigma_v = _sigma_v(s)
    if s.shape == "rare_only":
        keys = (None, None, None)  # the curves are those of the unshifted population
    else:
        if spread is None:
            raise ValueError(f"Shape {s.shape!r} needs a calibrated spread")
        shifted = (True, True, True) if s.shape == "normal" else (True, False, False)
        keys = tuple((s.shape, float(spread)) if on else None for on in shifted)
    fits = [fit_coefficients(s.family, a, sigma_v, keys[a]) for a in (0, 1)]
    fits.append(fits[1])  # P follows F: same target, same effect distribution
    if s.shape == "rare_only":
        c = _solve_offset(lambda c: population_rate(s, c, NO_EFFECT), baseline)
        if s.delta == 0:
            m = 0.0
        else:
            base = population_rate(s, c, NO_EFFECT)

            def gap(m: float) -> float:
                return population_rate(s, c, effect_distribution("rare_only", m)) - base

            if gap(60.0) <= abs(s.delta):
                raise ValueError(f"rare_only cannot reach |Δ| = {abs(s.delta)} in {s.name}")
            m = float(brentq(lambda m: gap(m) - abs(s.delta), 0.0, 60.0, xtol=1e-12))
        offsets = (c, c, c)
        shifted = (True, False, False) if s.delta >= 0 else (False, True, True)
        spread = m
    else:
        effects = _effects(s, spread, shifted)
        c_f = _solve_offset(lambda c: population_rate(s, c, effects[1]), baseline)
        h_f = population_rate(s, c_f, effects[1])
        c_a = _solve_offset(lambda c: population_rate(s, c, effects[0]), h_f + s.delta)
        offsets = (c_a, c_f, c_f)
    effects = _effects(s, spread, shifted)
    h = [population_rate(s, offsets[a], effects[a]) for a in range(len(ARMS))]
    slope = FAILURE_SLOPE if s.failure_correlated else 0.0
    return Parameters(
        offsets=offsets,
        coefficients=tuple(fit[0] for fit in fits),
        fit_rmse=tuple(fit[1] for fit in fits),
        spread=float(spread),
        shifted=shifted,
        failure=(_logistic_intercept(s.failure, slope), slope),
        truth={"A-F": h[0] - h[1], "P-F": h[2] - h[1], "A-P": h[0] - h[2]},
    )


def population_curve(
    scenario: Scenario, params: Parameters, step: float = 1e-3
) -> dict[str, dict[str, float]]:
    """Per arm, the attained population curve h̄_a at K* (value, slope and curvature in x, by
    central differences) beside its target (the family's curve at its baseline).

    At a kink the "curvature" is the second difference over ±step, not a derivative.
    """
    family = FAMILIES[scenario.family]
    effects = _effects(scenario, params.spread, params.shifted)
    x = np.array([-step, 0.0, step])

    def summary(h: np.ndarray) -> dict[str, float]:
        return {
            "h": float(h[1]),
            "slope": float((h[2] - h[0]) / (2 * step)),
            "curvature": float((h[2] - 2 * h[1] + h[0]) / step**2),
        }

    curve = {}
    for a, arm in enumerate(ARMS):
        h = population_values(
            _sigma_v(scenario), params.offsets[a], params.coefficients[a], effects[a], x
        )
        curve[arm] = {
            "attained": summary(h),
            "target": summary(family.target(a, x)),
            "per_prompt": dict(zip(("s", "q", "gamma"), params.coefficients[a], strict=True)),
            "fit_rmse": params.fit_rmse[a],
        }
    return curve


# Data


def draw_constants(scenario: Scenario, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A replicate's prompt effects α (521,) and selection misses τ (arm, rate)."""
    alpha = scenario.sigma_alpha * rng.standard_normal(PROMPTS)
    shifts = rng.uniform(-scenario.tau, scenario.tau, (len(ARMS), 3))
    if scenario.miss == "common":
        return alpha, np.full_like(shifts, shifts[0, 0])
    if scenario.miss == "arm":
        return alpha, np.repeat(shifts[:, :1], 3, axis=1)
    return alpha, shifts


def interpolate_at_zero(
    x: np.ndarray, harm: np.ndarray, failed: np.ndarray, w_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """ĥ at x = 0 and the support cause, over the leading axes; endpoints on the last axis.

    ĥ is NaN where the cause is not ``SUPPORTED``.
    """
    below, above = x < 0, x >= 0
    low_ok, high_ok = below & ~failed, above & ~failed
    low = np.argmax(np.where(low_ok, x, -np.inf), axis=-1)[..., None]
    high = np.argmin(np.where(high_ok, x, np.inf), axis=-1)[..., None]
    x_low, x_high = (np.take_along_axis(x, i, axis=-1)[..., 0] for i in (low, high))
    h_low, h_high = (np.take_along_axis(harm, i, axis=-1)[..., 0] for i in (low, high))
    cause = np.full(x.shape[:-1], SUPPORTED, dtype=np.int8)
    cause[x_high - x_low > w_max] = WIDTH
    cause[~(low_ok.any(axis=-1) & high_ok.any(axis=-1))] = TECHNICAL
    cause[~(below.any(axis=-1) & above.any(axis=-1))] = BRACKETING
    with np.errstate(invalid="ignore", divide="ignore"):
        estimate = h_low + (h_high - h_low) * (-x_low) / (x_high - x_low)
    return np.where(cause == SUPPORTED, estimate, np.nan), cause


def endpoint_harm(
    eta: np.ndarray, alpha: np.ndarray, rho_b: float, rho_a: float, rng: np.random.Generator
) -> np.ndarray:
    """Mean label over the prompts per endpoint; ``eta`` (blocks, arms, rates) without α."""
    n, arms, rates = eta.shape
    z = math.sqrt(1 - rho_a) * rng.standard_normal((n, arms, rates, alpha.size))
    if rho_a > rho_b:
        z += math.sqrt(rho_a - rho_b) * rng.standard_normal((n, arms, 1, alpha.size))
    if rho_b > 0:
        z += math.sqrt(rho_b) * rng.standard_normal((n, 1, 1, alpha.size))
    threshold = ndtri(expit(eta[..., None] + alpha))
    return (z < threshold).mean(axis=-1)


def block_effects(
    scenario: Scenario, params: Parameters, latent: np.ndarray, arms: Sequence[int]
) -> np.ndarray:
    """d_{b,a} (blocks, arms) from the standard normal driving variable ``latent``."""
    effect = np.zeros_like(latent)
    for j, arm in enumerate(arms):
        if not params.shifted[arm]:
            continue
        w = latent[:, j]
        if scenario.shape == "normal":
            effect[:, j] = params.spread * w
        elif scenario.shape == "skewed":
            gamma = gammaincinv(GAMMA_SHAPE, ndtr(w))
            effect[:, j] = params.spread * (gamma - GAMMA_SHAPE) / math.sqrt(GAMMA_SHAPE)
        elif scenario.shape == "rare_centered":
            effect[:, j] = params.spread * ((w > RARE_THRESHOLD) - RARE)
        else:
            effect[:, j] = params.spread * (w > RARE_THRESHOLD)
    return effect


def simulate_blocks(
    scenario: Scenario,
    params: Parameters,
    alpha: np.ndarray,
    misses: np.ndarray,
    rng: np.random.Generator,
    n: int,
    arms: Sequence[int] = (0, 1, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """ĥ(K*) and the support cause of ``n`` new blocks, each (n, len(arms)).

    The draws do not depend on the parameters, so evaluations at different spreads share them.
    """
    s = scenario
    arms = list(arms)
    stream = rng.standard_normal(n)  # β_b/σ_s
    shared = rng.standard_normal((n, 1))
    speed = math.sqrt(s.rho_u) * shared + math.sqrt(1 - s.rho_u) * rng.standard_normal(
        (n, len(arms))
    )  # u/σ_u
    latent = s.rho_uh * speed + math.sqrt(1 - s.rho_uh**2) * rng.standard_normal((n, len(arms)))
    effect = block_effects(s, params, latent, arms)
    nominal = np.array(POSITIONS[s.positions])[arms] + misses[arms]
    x = nominal + s.sigma_u * speed[..., None] + s.sigma_e * rng.standard_normal((n, len(arms), 3))
    intercept, slope = params.failure
    failed = rng.random((n, len(arms), 3)) < expit(intercept + slope * stream)[:, None, None]
    eta = (
        np.array(params.offsets)[arms][:, None]
        + curve_offset(params.coefficients, x, arms)
        + s.sigma_s * stream[:, None, None]
        + effect[..., None]
    )
    harm = endpoint_harm(eta, alpha, s.rho_b, s.rho_a, rng)
    return interpolate_at_zero(x, harm, failed, s.w_max)


@dataclass(frozen=True)
class Replicates:
    """Per replicate and started block: ĥ per arm (NaN: unsupported or not started) and the
    support cause (``NOT_STARTED`` after the last started block)."""

    estimates: np.ndarray  # (reps, CAP, 3)
    causes: np.ndarray  # (reps, CAP, 3) int8
    started: np.ndarray  # (reps,)
    seconds: float = 0.0  # wall time of the simulation
    cpu_seconds: float = 0.0  # its process CPU time

    @property
    def primary(self) -> np.ndarray:
        return self.estimates[..., 0] - self.estimates[..., 1]

    @property
    def secondary(self) -> np.ndarray:
        """P − F and A − P on the last axis; NaN unless all three arms are supported."""
        e = self.estimates
        contrasts = np.stack([e[..., 2] - e[..., 1], e[..., 0] - e[..., 2]], axis=-1)
        return np.where(np.isfinite(e).all(axis=-1)[..., None], contrasts, np.nan)


def simulate_replicate(
    scenario: Scenario, params: Parameters, rep: int, *, base_seed: int, stream: str
) -> tuple[np.ndarray, np.ndarray, int]:
    """One study: blocks start until ``GENERATE_UNTIL`` are supported or the cap is reached."""
    rng = np.random.default_rng(derive_seed(base_seed, stream, scenario.name, rep))
    alpha, misses = draw_constants(scenario, rng)
    estimates = np.full((CAP, len(ARMS)), np.nan)
    causes = np.full((CAP, len(ARMS)), NOT_STARTED, dtype=np.int8)
    started = supported = 0
    while started < CAP and supported < GENERATE_UNTIL:
        n = min(CHUNK_BLOCKS, CAP - started)
        estimate, cause = simulate_blocks(scenario, params, alpha, misses, rng, n)
        estimates[started : started + n], causes[started : started + n] = estimate, cause
        supported += int(np.count_nonzero((cause[:, :2] == SUPPORTED).all(axis=1)))
        started += n
    return estimates, causes, started


def simulate_replicates(
    scenario: Scenario,
    params: Parameters,
    reps: Sequence[int],
    *,
    base_seed: int,
    stream: str,
) -> Replicates:
    clock, cpu = time.perf_counter(), time.process_time()
    estimates, causes, started = zip(
        *(
            simulate_replicate(scenario, params, r, base_seed=base_seed, stream=stream)
            for r in reps
        ),
        strict=True,
    )
    return Replicates(
        np.stack(estimates),
        np.stack(causes),
        np.array(started),
        time.perf_counter() - clock,
        time.process_time() - cpu,
    )


# Spread calibration


@dataclass(frozen=True)
class SpreadCalibration:
    key: str
    target_sd: float
    spread: float
    attained_sd: float
    noise_sd: float  # attained at spread 0: response and K noise alone
    evaluations: int
    cpu_seconds: float


def calibration_key(scenario: Scenario) -> Scenario:
    """The Δ = 0 scenario whose paired SD sets the spread (look timing and reps do not matter)."""
    return replace(scenario, delta=0.0, postpone=0, reps=0, group="")


def paired_sd(
    scenario: Scenario, spread: float, *, seed: int, replicates: int, blocks: int
) -> float:
    """√(mean within-replicate variance of Δ̂_b over supported blocks) at the given spread."""
    params = solve_parameters(scenario, spread)
    rng = np.random.default_rng(seed)
    variances = []
    for _ in range(replicates):
        alpha, misses = draw_constants(scenario, rng)
        estimate, _ = simulate_blocks(scenario, params, alpha, misses, rng, blocks, arms=(0, 1))
        delta = estimate[:, 0] - estimate[:, 1]
        delta = delta[np.isfinite(delta)]
        if delta.size >= 2:
            variances.append(delta.var(ddof=1))
    if not variances:
        raise ValueError(f"No replicate of {scenario.name} has two supported blocks")
    return float(np.sqrt(np.mean(variances)))


def calibrate_spread(
    scenario: Scenario, *, base_seed: int, replicates: int = 100, blocks: int = 200
) -> SpreadCalibration:
    """σ_d (or M) giving the scenario's target paired SD at Δ = 0, by root finding on common
    random numbers."""
    cpu = time.process_time()
    key = calibration_key(scenario)
    seed = derive_seed(base_seed, "spread", key.name)
    cache: dict[float, float] = {}

    def attained(spread: float) -> float:
        if spread not in cache:
            cache[spread] = paired_sd(key, spread, seed=seed, replicates=replicates, blocks=blocks)
        return cache[spread]

    noise = attained(0.0)
    if noise >= key.sd:
        spread = 0.0
    else:
        high = 0.5
        while attained(high) < key.sd:
            high *= 2
            if high > 64:
                raise ValueError(f"The paired SD of {key.name} cannot reach {key.sd}")
        spread = float(brentq(lambda s: attained(s) - key.sd, 0.0, high, xtol=1e-4, rtol=1e-4))
    return SpreadCalibration(
        key.name, key.sd, spread, attained(spread), noise, len(cache), time.process_time() - cpu
    )


# Sequential procedure


@dataclass(frozen=True)
class LookStatistics:
    """Per replicate (axis 0) and look (axis 1); look 2 is the terminal look. A one-look design
    has a look 1 that is never reached."""

    reached: np.ndarray  # (R, 2) bool; the terminal look always happens
    at_cap: np.ndarray  # (R,) the terminal look was taken at the cap below its minimum
    started: np.ndarray  # (R, 2)
    n: np.ndarray  # (R, 2) supported blocks
    mean: np.ndarray  # (R, 2)
    sd: np.ndarray  # (R, 2)
    secondary_n: np.ndarray  # (R, 2) three-arm cohort
    secondary_mean: np.ndarray  # (R, 2, 2): look, contrast
    secondary_sd: np.ndarray  # (R, 2, 2)


def _moments(values: np.ndarray, include: np.ndarray) -> tuple[np.ndarray, ...]:
    n = include.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(include, values, 0.0).sum(axis=-1) / n
        deviation = np.where(include, values - mean[..., None], 0.0)
        sd = np.sqrt((deviation**2).sum(axis=-1) / (n - 1))
    return n, mean, sd


def look_statistics(
    primary: np.ndarray,
    secondary: np.ndarray | None = None,
    sizes: tuple[int, ...] = LOOKS,
    cap: int = CAP,
) -> LookStatistics:
    """Statistics of the looks from Δ̂_b in started order, NaN where unsupported.

    ``sizes`` holds one look (terminal) or two. ``primary`` is (R, blocks) and ``secondary``
    (R, blocks, 2), NaN where the three arms are not all supported. Blocks past the cap are
    ignored.
    """
    if len(sizes) == 2 and not 0 < sizes[0] < sizes[1]:
        raise ValueError(f"Look sizes must increase: {sizes}")
    if len(sizes) not in (1, 2):
        raise ValueError(f"One or two looks, not {sizes}")
    primary = np.asarray(primary, dtype=float)[:, :cap]
    reps, blocks = primary.shape
    supported = np.isfinite(primary)
    count = np.cumsum(supported, axis=1)
    reached, index = [], []
    for size in (None, *sizes) if len(sizes) == 1 else sizes:
        if size is None:  # a one-look design has no look 1
            reached.append(np.zeros(reps, dtype=bool))
            index.append(np.full(reps, blocks - 1))
            continue
        hit = count >= size
        reached.append(hit.any(axis=1))
        index.append(np.where(reached[-1], hit.argmax(axis=1), blocks - 1))
    at_cap = ~reached[1]
    reached[1] = np.ones(reps, dtype=bool)
    index = np.stack(index, axis=1)
    included = supported[:, None, :] & (np.arange(blocks) <= index[..., None])  # (R, 2, blocks)
    n, mean, sd = _moments(primary[:, None, :], included)
    if secondary is None:
        secondary = np.full((reps, blocks, 2), np.nan)
    secondary = np.asarray(secondary, dtype=float)[:, :cap]
    cohort = included & np.isfinite(secondary[..., 0])[:, None, :]
    moments = [_moments(secondary[:, None, :, k], cohort) for k in range(2)]
    return LookStatistics(
        reached=np.stack(reached, axis=1),
        at_cap=at_cap,
        started=index + 1,
        n=n,
        mean=mean,
        sd=sd,
        secondary_n=moments[0][0],
        secondary_mean=np.stack([m[1] for m in moments], axis=-1),
        secondary_sd=np.stack([m[2] for m in moments], axis=-1),
    )


@functools.cache
def _t_table(level: float) -> np.ndarray:
    """Two-sided t quantiles at ``level`` for 1 … CAP degrees of freedom."""
    return student_t.ppf(1 - (1 - level) / 2, np.arange(1, CAP + 1))


def t_interval(
    mean: np.ndarray, sd: np.ndarray, n: np.ndarray, level: float, floor: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """mean ± t_{n−1}·max(sd, floor)/√n; (−∞, ∞) with fewer than 2 blocks."""
    n = np.asarray(n)
    valid = n >= 2
    quantile = _t_table(level)[np.clip(n - 2, 0, CAP - 1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        half = quantile * np.maximum(sd, floor) / np.sqrt(n)
        low, high = mean - half, mean + half
    return np.where(valid, low, -np.inf), np.where(valid, high, np.inf)


@dataclass(frozen=True)
class Decisions:
    """The procedure's outcome per replicate, read at the stopping look."""

    look: np.ndarray  # 1 or 2 (2: the terminal look)
    low: np.ndarray
    high: np.ndarray
    estimate: np.ndarray
    n: np.ndarray
    started: np.ndarray
    at_cap: np.ndarray  # stopped at a terminal look taken at the cap below its minimum
    equivalence: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    inside_wide: np.ndarray
    secondary_n: np.ndarray
    secondary_estimate: np.ndarray  # (R, 2)
    secondary_low: np.ndarray  # (R, 2)
    secondary_high: np.ndarray  # (R, 2)

    @property
    def inconclusive(self) -> np.ndarray:
        return ~(self.equivalence | self.positive | self.negative)


def _inside(low: np.ndarray, high: np.ndarray, margin: float) -> np.ndarray:
    return (low >= -margin) & (high <= margin)


def sequential_decisions(
    stats: LookStatistics,
    levels: Sequence[float],
    floor: float = 0.0,
    *,
    equivalence_at_first: bool = True,
    margin: float = MARGIN,
    wide: float = WIDE_MARGIN,
    secondary_level: float = SECONDARY_LEVEL,
) -> Decisions:
    """Stop at look 1 on equivalence (if allowed there) or a direction; otherwise the terminal
    look decides. ``levels`` has one level per look of the design (one for a one-look design)."""
    reps = stats.n.shape[0]
    if len(levels) == 1:
        low1, high1 = np.full(reps, -np.inf), np.full(reps, np.inf)
        terminal = levels[0]
    else:
        low1, high1 = t_interval(stats.mean[:, 0], stats.sd[:, 0], stats.n[:, 0], levels[0], floor)
        terminal = levels[1]
    low2, high2 = t_interval(stats.mean[:, 1], stats.sd[:, 1], stats.n[:, 1], terminal, floor)
    equivalent1 = _inside(low1, high1, margin) if equivalence_at_first else np.zeros(reps, bool)
    stop1 = stats.reached[:, 0] & (equivalent1 | (low1 > 0) | (high1 < 0))
    k = np.where(stop1, 0, 1)
    rows = np.arange(reps)
    low = np.where(stop1, low1, low2)
    high = np.where(stop1, high1, high2)
    secondary_n = stats.secondary_n[rows, k]
    secondary_low, secondary_high = t_interval(
        stats.secondary_mean[rows, k],
        stats.secondary_sd[rows, k],
        secondary_n[:, None],
        secondary_level,
        floor,
    )
    return Decisions(
        look=k + 1,
        low=low,
        high=high,
        estimate=stats.mean[rows, k],
        n=stats.n[rows, k],
        started=stats.started[rows, k],
        at_cap=~stop1 & stats.at_cap,
        equivalence=_inside(low, high, margin) & (~stop1 | equivalence_at_first),
        positive=low > 0,
        negative=high < 0,
        inside_wide=_inside(low, high, wide),
        secondary_n=secondary_n,
        secondary_estimate=stats.secondary_mean[rows, k],
        secondary_low=secondary_low,
        secondary_high=secondary_high,
    )


def pointwise_readings(decisions: Decisions, allowance: float) -> dict[str, np.ndarray]:
    """Readings about the pointwise Δ(K*) from the final interval with the bias allowance b:
    a direction excludes [−b, b], equivalence lies inside ±(5 pp − b)."""
    d = decisions
    return {
        "equivalence": _inside(d.low, d.high, MARGIN - allowance),
        "positive": d.low > allowance,
        "negative": d.high < -allowance,
    }


# Summaries


def _rate(flags: np.ndarray) -> tuple[float, float]:
    p = float(np.mean(flags))
    return p, math.sqrt(p * (1 - p) / flags.size)


def block_diagnostics(params: Parameters, data: Replicates) -> dict[str, Any]:
    """Block-level quantities over every started block, independent of the procedure.

    ``block_mean`` estimates the interpolation estimand E[Δ̂_b] (over supported blocks);
    ``block_bias`` is the paired interpolation bias, ``block_mean`` minus the pointwise Δ(K*); its
    SE treats replicates as the units.
    """
    primary, secondary, causes = data.primary, data.secondary, data.causes
    finite = np.isfinite(primary)
    counts, per_rep, sd = _moments(primary, finite)
    pooled = float(primary[finite].mean())
    usable = np.isfinite(per_rep)
    bias_se = (
        float(per_rep[usable].std(ddof=1) / math.sqrt(usable.sum()))
        if usable.sum() > 1
        else math.nan
    )
    variances = sd[counts >= 2] ** 2
    started = causes[..., 0] != NOT_STARTED
    arm_rates = {
        arm: {
            name: float((causes[..., a][started] == code).mean()) for name, code in CAUSES.items()
        }
        for a, arm in enumerate(ARMS)
    }
    for a, arm in enumerate(ARMS):
        arm_rates[arm]["any"] = float((causes[..., a][started] != SUPPORTED).mean())
    pair = causes[..., :2][started]
    unsupported = np.where(pair == SUPPORTED, 99, pair).min(axis=1)  # the first cause in order
    block_rates = {name: float((unsupported == code).mean()) for name, code in CAUSES.items()}
    block_rates["any"] = float((unsupported != 99).mean())
    three_arm = np.isfinite(secondary[..., 0])
    targets = {
        name: float(secondary[..., k][three_arm].mean()) if three_arm.any() else math.nan
        for k, name in enumerate(SECONDARY_CONTRASTS)
    }
    return {
        "attained_sd": float(np.sqrt(variances.mean())) if variances.size else math.nan,
        "block_mean": pooled,
        "block_bias": pooled - params.truth["A-F"],
        "block_bias_se": bias_se,
        "secondary_truth": {name: params.truth[name] for name in SECONDARY_CONTRASTS},
        "secondary_block_mean": targets,
        "unsupported_per_arm": arm_rates,
        "unsupported_per_block": block_rates,
        "three_arm_attrition": float(1 - three_arm.sum() / finite.sum()),
    }


def outcome_rates(
    decisions: Decisions,
    delta: float,
    truth: Mapping[str, float],
    targets: Mapping[str, float],
    allowance: float | None = None,
) -> dict[str, Any]:
    """Decision rates, their Monte Carlo SEs and the stopping-time summaries of one design,
    level pair and floor; with an allowance, the pointwise readings too."""
    d = decisions
    rates: dict[str, Any] = {}
    flags = {
        "equivalence": d.equivalence,
        "positive": d.positive,
        "negative": d.negative,
        "inconclusive": d.inconclusive,
        "inside_wide": d.inside_wide,
        "cap_exhausted": d.at_cap,
        "look1_stop": d.look == 1,
    }
    for name, flag in flags.items():
        rates[f"p_{name}"], rates[f"se_{name}"] = _rate(flag)
    power = d.equivalence if delta == 0 else d.positive if delta > 0 else d.negative
    rates["power"], rates["se_power"] = _rate(power)
    if allowance is not None:
        readings = pointwise_readings(d, allowance)
        pointwise: dict[str, float] = {"allowance": allowance}
        for name, flag in readings.items():
            pointwise[f"p_{name}"], pointwise[f"se_{name}"] = _rate(flag)
        right = (
            readings["equivalence"]
            if delta == 0
            else readings["positive" if delta > 0 else "negative"]
        )
        pointwise["power"], pointwise["se_power"] = _rate(right)
        rates["pointwise"] = pointwise
    total = d.n * d.estimate
    unsupported = d.started - d.n
    with np.errstate(invalid="ignore", divide="ignore"):
        cohort_low = (total - unsupported) / d.started
        cohort_high = (total + unsupported) / d.started
        attrition = 1 - d.secondary_n / d.n
    straddles = ((cohort_low < -MARGIN) & (cohort_high > -MARGIN)) | (
        (cohort_low < MARGIN) & (cohort_high > MARGIN)
    )
    rates["p_cohort_bounds_straddle_5pp"], rates["se_cohort_bounds_straddle_5pp"] = _rate(straddles)
    coverage: dict[str, dict[str, float]] = {}
    for k, name in enumerate(SECONDARY_CONTRASTS):
        low, high = d.secondary_low[:, k], d.secondary_high[:, k]
        coverage[name] = {}
        for target, value in (("truth", truth[name]), ("block_mean", targets[name])):
            covered = (low <= value) & (value <= high)
            coverage[name][target], coverage[name][f"{target}_se"] = _rate(covered)
    estimates = d.estimate[np.isfinite(d.estimate)]
    rates |= {
        "mean_started": float(d.started.mean()),
        "mean_supported": float(d.n.mean()),
        "mean_estimate": float(np.nanmean(d.estimate)),
        "bias": float(np.nanmean(d.estimate) - delta),
        "se_bias": (
            float(estimates.std(ddof=1) / math.sqrt(estimates.size))
            if estimates.size > 1
            else math.nan
        ),
        "cohort_bound_width": float(np.nanmean(cohort_high - cohort_low)),
        "secondary_coverage": coverage,
        "three_arm_attrition": float(np.nanmean(attrition)),
        "mean_three_arm_blocks": float(d.secondary_n.mean()),
    }
    return rates


def summarize_scenario(
    scenario: Scenario,
    params: Parameters,
    data: Replicates,
    alphas: Sequence[float],
    floors: Sequence[float] = SE_FLOORS,
    allowance: float | None = None,
    designs: Sequence[LookDesign] = LOOK_DESIGNS,
) -> dict[str, Any]:
    """The scenario's block diagnostics and one result row per look design × α × SE floor."""
    diagnostics = block_diagnostics(params, data)
    primary, secondary = data.primary, data.secondary
    targets = diagnostics["secondary_block_mean"]
    results = []
    for design in designs:
        stats = look_statistics(primary, secondary, sizes=design.realized(scenario.postpone))
        for alpha in alphas:
            levels = design.levels(alpha)
            for floor in floors:
                decisions = sequential_decisions(
                    stats, levels, floor, equivalence_at_first=design.equivalence_at_first
                )
                # The scenario's Δ, not the solved truth: that is Δ only to ~1e-12, and the
                # power at Δ = 0 is the equivalence rate.
                rates = outcome_rates(decisions, scenario.delta, params.truth, targets, allowance)
                rates["bias_vs_block_mean"] = rates["mean_estimate"] - diagnostics["block_mean"]
                results.append(
                    {
                        "looks": design.name,
                        "alpha": alpha,
                        "levels": list(levels),
                        "floor": floor,
                        **rates,
                    }
                )
    return {
        "name": scenario.name,
        "design": scenario.design,
        "settings": asdict(scenario),
        "reps": int(data.started.size),
        "seconds": data.seconds,
        "cpu_seconds": data.cpu_seconds,
        "parameters": {
            "offsets": dict(zip(ARMS, params.offsets, strict=True)),
            "spread": params.spread,
            "shifted": [arm for arm, on in zip(ARMS, params.shifted, strict=True) if on],
            "failure_intercept": params.failure[0],
            "failure_slope": params.failure[1],
            "truth": params.truth,
            "population_curve": population_curve(scenario, params),
        },
        "diagnostics": diagnostics,
        "results": results,
    }


# Grids


def _reps(delta: float) -> int:
    """20 000 replicates where an error rate meets its bound (Δ = 0, |Δ| = 5 pp)."""
    return BOUNDARY_REPS if delta == 0 or math.isclose(abs(delta), MARGIN) else OTHER_REPS


def _scenarios(
    group: str, sds: Sequence[float], deltas: Sequence[float], reps: int | None = None, **changes
) -> list[Scenario]:
    return [
        Scenario(delta=d, sd=sd, reps=reps or _reps(d), group=group, **changes)
        for sd in sds
        for d in deltas
    ]


def calibration_grid() -> list[Scenario]:
    """The central model (normal shape, w_max 0.35): every target SD × every Δ."""
    return _scenarios("central", TARGET_SDS, DELTAS)


def validation_grid() -> list[Scenario]:
    """Independent checks of the chosen levels: every shape × target SD × Δ, then postponed looks,
    cap exhaustion, bracket widths, K spread, selection misses, curve families and stress."""
    boundary, short = (0.0, MARGIN, -MARGIN), (0.0, MARGIN)
    grid = _scenarios("central", TARGET_SDS, DELTAS)
    for shape in ("skewed", "rare_centered"):
        grid += _scenarios("shapes", TARGET_SDS, DELTAS, shape=shape)
    grid += _scenarios("shapes", (0.0,), DELTAS, shape="rare_only")
    grid += _scenarios("postponed", (0.05, 0.10), boundary, postpone=6)
    grid += _scenarios("cap", (0.05, 0.10), boundary, failure=0.2, failure_correlated=True)
    for w_max in (0.30, 0.47):
        grid += _scenarios("width", (0.05, 0.10), boundary, w_max=w_max)
    for sigma_u, sigma_e in K_SPREADS[1:]:
        grid += _scenarios("k_spread", (0.05,), short, sigma_u=sigma_u, sigma_e=sigma_e)
    for miss in ("common", "arm"):
        grid += _scenarios("miss", (0.05,), short, miss=miss)
    for family in ("kink", "high", "stress_slopes"):
        grid += _scenarios("families", (0.05,), short, family=family)
    stress = (
        {"rho_u": 0.5},
        {"rho_uh": 0.5},
        {"rho_b": 0.0, "rho_a": 0.0},
        {"rho_b": 0.6, "rho_a": 0.9},
        {"failure": 0.1},
        {"failure": 0.2},
        {"failure": 0.1, "failure_correlated": True},
    )
    for changes in stress:
        grid += _scenarios("stress", (0.05,), short, **changes)
    return grid


def bias_grid() -> list[Scenario]:
    """Paired interpolation bias: family × selection miss × w_max at Δ ∈ {0, 5 pp}, and the
    coarse discovery brackets without a width rule."""
    grid = [
        Scenario(family=family, miss=miss, w_max=w_max, delta=d, group="bias")
        for family in FAMILIES
        for miss in MISSES
        for w_max in W_MAXES
        for d in (0.0, MARGIN)
    ]
    grid += [
        Scenario(positions="grid", family=family, w_max=math.inf, group="bias_grid")
        for family in FAMILIES
    ]
    return grid


def smoke_grid(reps: int = SMOKE_REPS) -> list[Scenario]:
    """Every shape at the central SD, and one scenario per revision-1 addition, Δ ∈ {0, 5 pp}."""
    short = (0.0, MARGIN)
    grid = []
    for shape in SHAPES:
        sd = 0.0 if shape == "rare_only" else 0.05
        grid += _scenarios("shapes", (sd,), short, reps, shape=shape)
    for w_max in (0.30, 0.47):
        grid += _scenarios("width", (0.05,), short, reps, w_max=w_max)
    for sigma_u, sigma_e in K_SPREADS[1:]:
        grid += _scenarios("k_spread", (0.05,), short, reps, sigma_u=sigma_u, sigma_e=sigma_e)
    for miss in ("common", "arm"):
        grid += _scenarios("miss", (0.05,), short, reps, miss=miss)
    for family in ("kink", "high", "stress_slopes"):
        grid += _scenarios("families", (0.05,), short, reps, family=family)
    grid += _scenarios("cap", (0.05,), (0.0,), reps, failure=0.2, failure_correlated=True)
    grid += _scenarios("postponed", (0.05,), (0.0,), reps, postpone=6)
    grid.append(
        Scenario(
            positions="grid", family="stress_slopes", w_max=math.inf, reps=reps, group="bias_grid"
        )
    )
    return grid


# Verdicts


def _worst(rows: Sequence[Mapping], key: str, pick: Callable = max) -> dict[str, Any]:
    if not rows:
        return {"value": None, "scenario": None, "mc_se": None}
    row = pick(rows, key=lambda r: r[key])
    value = row[key]
    return {
        "value": value,
        "scenario": row["scenario"],
        "mc_se": math.sqrt(value * (1 - value) / row["reps"]),
    }


def constraint_summary(rows: Sequence[Mapping]) -> dict[str, Any]:
    """Worst error rates, lowest powers and the largest E started and cap exhaustion over result
    rows of one design, α and floor."""
    null = [r for r in rows if r["delta"] == 0]
    near = [r for r in rows if math.isclose(abs(r["delta"]), MARGIN)]
    beyond = [r for r in rows if math.isclose(abs(r["delta"]), 0.055)]
    summary = {
        "max_false_positive": _worst(null, "p_positive"),
        "max_false_negative": _worst(null, "p_negative"),
        "max_false_equivalence_5pp": _worst(near, "p_equivalence"),
        "max_false_equivalence_5.5pp": _worst(beyond, "p_equivalence"),
        "min_power_equivalence_at_0": _worst(null, "power", min),
        "min_power_direction_at_5pp": _worst(near, "power", min),
        "max_cap_exhausted": _worst(rows, "p_cap_exhausted"),
        "max_mean_started": max((r["mean_started"] for r in rows), default=None),
    }
    checks = [
        summary["max_false_positive"]["value"],
        summary["max_false_negative"]["value"],
    ]
    meets = all(v is None or v <= MAX_FALSE_DIRECTION for v in checks)
    equivalence = summary["max_false_equivalence_5pp"]["value"]
    summary["meets"] = meets and (equivalence is None or equivalence <= MAX_FALSE_EQUIVALENCE)
    return summary


def result_rows(scenarios: Sequence[Mapping]) -> list[dict[str, Any]]:
    """One flat row per scenario × look design × α × floor."""
    rows = []
    for s in scenarios:
        settings = s["settings"]
        for r in s["results"]:
            rows.append(
                {
                    "scenario": s["name"],
                    "group": settings["group"],
                    "design": s["design"],
                    "shape": settings["shape"],
                    "sd": settings["sd"],
                    "delta": settings["delta"],
                    "reps": s["reps"],
                    **r,
                }
            )
    return rows


def select_levels(scenarios: Sequence[Mapping], floors: Sequence[float] = SE_FLOORS) -> list[dict]:
    """Per look design and SE floor, the largest α on the grid meeting every error constraint
    over the central scenarios at the binding SDs (3, 5, 10 pp); 15 pp is reported, not
    binding."""
    rows = [r for r in result_rows(scenarios) if r["group"] == "central"]
    selections = []
    for looks in dict.fromkeys(r["looks"] for r in rows):
        for floor in floors:
            mine = [r for r in rows if r["looks"] == looks and r["floor"] == floor]
            binding = [r for r in mine if any(math.isclose(r["sd"], b) for b in BINDING_SDS)]
            by_alpha = {
                alpha: constraint_summary([r for r in binding if r["alpha"] == alpha])
                for alpha in sorted({r["alpha"] for r in mine})
            }
            passing = [alpha for alpha, summary in by_alpha.items() if summary["meets"]]
            alpha = max(passing, default=None)
            chosen = [r for r in mine if r["alpha"] == alpha]
            selections.append(
                {
                    "looks": looks,
                    "floor": floor,
                    "alpha": alpha,
                    "levels": chosen[0]["levels"] if chosen else None,
                    "constraints": by_alpha.get(alpha),
                    "at_alpha_0.05": by_alpha.get(PLANNING_ALPHA),
                    "power": [
                        {
                            "sd": r["sd"],
                            "delta": r["delta"],
                            "power": r["power"],
                            "se": r["se_power"],
                        }
                        for r in chosen
                    ],
                    "per_alpha": {
                        f"{a:.4f}": {"meets": v["meets"]}
                        | {k: v[k]["value"] for k in v if isinstance(v[k], dict)}
                        for a, v in by_alpha.items()
                    },
                }
            )
    return selections


def validation_checks(
    scenarios: Sequence[Mapping], chosen: Mapping[tuple[str, float], float] | None = None
) -> list[dict[str, Any]]:
    """Worst rates and lowest powers per look design × floor, per group and over all, at the
    chosen α of that design and floor (``chosen``) and at α = 0.05."""
    rows = result_rows(scenarios)
    checks = []
    for looks in dict.fromkeys(r["looks"] for r in rows):
        for floor in sorted({r["floor"] for r in rows}):
            alphas = {"alpha_0.05": PLANNING_ALPHA}
            if chosen and chosen.get((looks, floor)) is not None:
                alphas["selected"] = chosen[(looks, floor)]
            for label, alpha in alphas.items():
                mine = [
                    r
                    for r in rows
                    if r["looks"] == looks and r["floor"] == floor and r["alpha"] == alpha
                ]
                if not mine:
                    continue
                groups = {"all": mine} | {
                    g: [r for r in mine if r["group"] == g]
                    for g in dict.fromkeys(r["group"] for r in mine)
                }
                checks.append(
                    {
                        "looks": looks,
                        "floor": floor,
                        "which": label,
                        "alpha": alpha,
                        "levels": mine[0]["levels"],
                        "groups": {g: constraint_summary(group) for g, group in groups.items()},
                    }
                )
    return checks


BIAS_FIELDS = {"family", "miss", "w_max", "delta", "sd", "reps", "postpone", "group"}


def _bias_scenario(settings: Mapping[str, Any]) -> bool:
    """Central in everything but the fields the bias allowance ranges over."""
    return all(
        settings[f.name] == f.default
        for f in fields(Scenario)
        if f.compare and f.name not in BIAS_FIELDS
    )


def bias_allowance(scenarios: Sequence[Mapping]) -> dict[str, Any]:
    """The paired pointwise bias per w_max and family (largest |bias| over selection misses and
    Δ), and the allowance b: the largest over the families at w_max 0.35 and 0.47."""
    eligible = [s for s in scenarios if _bias_scenario(s["settings"])]
    per_w_max = []
    for w_max in sorted({s["settings"]["w_max"] for s in eligible}):
        per_family = {}
        for family in FAMILIES:
            mine = [
                s
                for s in eligible
                if s["settings"]["w_max"] == w_max and s["settings"]["family"] == family
            ]
            if not mine:
                continue
            worst = max(mine, key=lambda s: abs(s["diagnostics"]["block_bias"]))
            per_family[family] = {
                "abs_bias": abs(worst["diagnostics"]["block_bias"]),
                "bias": worst["diagnostics"]["block_bias"],
                "se": worst["diagnostics"]["block_bias_se"],
                "scenario": worst["name"],
                "scenarios": len(mine),
            }
        driver = max(per_family, key=lambda f: per_family[f]["abs_bias"])
        per_w_max.append(
            {
                "w_max": w_max,
                "largest": per_family[driver]["abs_bias"],
                "family": driver,
                "per_family": per_family,
            }
        )
    counted = [
        (entry["w_max"], family, item)
        for entry in per_w_max
        if any(math.isclose(entry["w_max"], w) for w in ALLOWANCE_W_MAXES)
        for family, item in entry["per_family"].items()
    ]
    allowance = None
    if counted:
        w_max, family, item = max(counted, key=lambda c: c[2]["abs_bias"])
        allowance = {
            "value": item["abs_bias"],
            "family": family,
            "w_max": w_max,
            "scenario": item["scenario"],
            "se": item["se"],
            "over_w_max": list(ALLOWANCE_W_MAXES),
            "over_families": list(FAMILIES),
        }
    return {"per_w_max": per_w_max, "allowance": allowance}


# Running


@dataclass(frozen=True)
class RunSettings:
    stream: str
    alphas: tuple[float, ...]
    floors: tuple[float, ...] = SE_FLOORS
    allowance: float | None = None  # b for the pointwise readings
    base_seed: int = 0
    workers: int = 1
    chunk: int = 250  # replicates per worker task
    spread_replicates: int = 100
    spread_blocks: int = 200


def _calibration_task(task: tuple[Scenario, RunSettings]) -> SpreadCalibration:
    key, settings = task
    return calibrate_spread(
        key,
        base_seed=settings.base_seed,
        replicates=settings.spread_replicates,
        blocks=settings.spread_blocks,
    )


def _replicate_task(task: tuple[Scenario, Parameters, int, int, RunSettings]):
    scenario, params, start, stop, settings = task
    data = simulate_replicates(
        scenario,
        params,
        range(start, stop),
        base_seed=settings.base_seed,
        stream=settings.stream,
    )
    return scenario.name, start, data


@contextmanager
def _executor(workers: int) -> Iterator[Callable]:
    if workers <= 1:
        yield map
        return
    # Fresh single-threaded workers: the parallelism comes from the processes.
    saved = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    os.environ.update(dict.fromkeys(THREAD_VARIABLES, "1"))
    try:
        pool = mp.get_context("spawn").Pool(workers)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    with pool:
        yield pool.imap_unordered


def run_scenarios(
    scenarios: Sequence[Scenario],
    settings: RunSettings,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Calibrate the spreads, simulate every scenario and summarize it."""
    names = [s.name for s in scenarios]
    if len(set(names)) != len(names):
        raise ValueError("Scenario names must be unique: they name the random streams")
    keys = {
        calibration_key(s).name: calibration_key(s) for s in scenarios if s.shape != "rare_only"
    }
    calibrations: dict[str, SpreadCalibration] = {}
    with _executor(settings.workers) as imap:
        tasks = [(key, settings) for key in keys.values()]
        for done, result in enumerate(imap(_calibration_task, tasks), start=1):
            calibrations[result.key] = result
            if progress is not None:
                progress("spread", done, len(tasks))
        params = {}
        for s in scenarios:
            spread = (
                None if s.shape == "rare_only" else calibrations[calibration_key(s).name].spread
            )
            params[s.name] = solve_parameters(s, spread)
        by_name = {s.name: s for s in scenarios}
        tasks = [
            (s, params[s.name], start, min(start + settings.chunk, s.reps), settings)
            for s in scenarios
            for start in range(0, s.reps, settings.chunk)
        ]
        pending = {s.name: math.ceil(s.reps / settings.chunk) for s in scenarios}
        parts: dict[str, list[tuple[int, Replicates]]] = {name: [] for name in names}
        summaries: dict[str, dict] = {}
        for done, (name, start, data) in enumerate(imap(_replicate_task, tasks), start=1):
            parts[name].append((start, data))
            pending[name] -= 1
            if pending[name] == 0:
                chunks = [d for _, d in sorted(parts.pop(name), key=lambda item: item[0])]
                merged = Replicates(
                    np.concatenate([c.estimates for c in chunks]),
                    np.concatenate([c.causes for c in chunks]),
                    np.concatenate([c.started for c in chunks]),
                    sum(c.seconds for c in chunks),
                    sum(c.cpu_seconds for c in chunks),
                )
                scenario = by_name[name]
                summary = summarize_scenario(
                    scenario,
                    params[name],
                    merged,
                    settings.alphas,
                    settings.floors,
                    settings.allowance,
                )
                key = calibration_key(scenario).name
                summary["calibration"] = asdict(calibrations[key]) if key in calibrations else None
                summaries[name] = summary
            if progress is not None:
                progress("replicates", done, len(tasks))
    return {
        "calibrations": {k: asdict(v) for k, v in calibrations.items()},
        "scenarios": [summaries[name] for name in names],
    }
