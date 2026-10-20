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
    - ``heavy`` (revision 4): d = σ_d·S_b·w in every arm, S_b a block scale shared by the arms,
      3 with probability 0.1 and 1 otherwise, rescaled to E[S²] = 1: a symmetric normal scale
      mixture (excess kurtosis 5.3), which the paired difference keeps.
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
spread to calibrate: its SD follows from Δ. The target SD is therefore the paired SD at Δ = 0; at
Δ ≠ 0 the same σ_d gives another SD (e.g. "SD 10" is ≈ 11 pp at +5 and ≈ 9 at −5), which every
result row reports as ``attained_sd``.

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
decides. Equivalence is not reported at a look-1 stop of ``24_50_eq_terminal``. Rule
``eq_min36`` (revision 4) concludes equivalence only at a look with ≥ 36 supported blocks, so not
at a cap look below that; a direction may still be concluded there. If look 1 is reached only at
the cap, looks 1 and 2 use the same blocks. The secondary contrasts are read at
the look where the primary stopped, with two-sided 98.75 % intervals on the three-arm cohort and
the same floor. At the stopping look, the finite-cohort bounds of the started cohort are
[(S − m)/N, (S + m)/N]: N started blocks, m of them unsupported, S the sum of the supported Δ̂_b.

Intervals (revision 3). ``t`` is the interval above. ``bootstrap_t`` is the studentized bootstrap
of the mean over the look's supported blocks, the interval the original study-2 protocol
preregistered: B = 2000 resamples of the blocks with replacement per look and replicate, T* =
(mean* − mean)/se*, and at level 1 − α_k the interval [mean − q*_{1−α_k/2}·se, mean −
q*_{α_k/2}·se]. The SE floor applies to se and se* alike (se* = max(sd*, s_min)/√n). A resample
with se* = 0 (floor 0) is dropped and counted; a zero se gives [mean, mean]; fewer than 2 blocks
or kept resamples give no interval. If every block difference is equal, T* ≡ 0 and the interval
collapses to [mean, mean] even with a floor. Each replicate, design and look has its own resample
stream, drawn in the workers; both intervals are evaluated on the same replicates.

Secondary intervals (the 2026-10-17 secondary check). The secondary contrasts use t intervals by
default (every run of revisions 1–5). ``RunSettings.secondary_intervals`` adds ``bootstrap_t``:
the same studentized bootstrap (``bootstrap_t_quantiles``, the same bounds formula), per contrast
on the look's three-arm cohort at 98.75 % per look, with B resamples per replicate, contrast,
design and look. Each secondary interval is read at the look where the primary stopped, on the
same replicates; a run with more than t reports the coverage per secondary interval, the joint
coverage of both contrasts included.

Levels. A two-look design uses the O'Brien–Fleming-shaped pair from Lan–DeMets spending of an
overall two-sided α at its information fraction n₁/n₂ (``obf_levels``); α = 0.05 at 24/50 gives
the planning levels 99.53 % and 95.19 %. A one-look design uses 1 − α. The levels are those of
the planned fraction at every look actually taken: a postponed look 1 and a look at the cap keep
them (no spending is recomputed at the realized fraction), and a cap look without a look 1 uses
the terminal level, not 1 − α.

Validation gate (revision 4). Before confirmation, 12 fresh blocks of the same study (same
prompts, rates and failure model) are drawn; the gate passes with ≥ 10 jointly supported in A and
F. Its failure rate and the decision rates among the studies that pass are reported beside the
unconditional ones. A prescreened run (``RunSettings.prescreen``, revision 5) draws the gate
first and simulates only the studies that pass, which sizes the rates among them where the gate
rarely passes; P(gate fails) then comes from all the gates drawn. The gate has its own stream, so
a passing study is the same with or without the prescreen.

Randomness. Every stream is seeded by the full SHA-256 digest of its key list (``generator``):
replicate ``rep`` of a scenario draws from (base_seed, stream, scenario name, rep), its
validation gate from (…, rep, "validation_gate"), its bootstrap resamples from (…, rep,
"bootstrap", design, look) and those of a secondary contrast from (…, rep, "bootstrap-secondary",
contrast, design, look). The stream names the run (``calibration``, ``validation``,
``confirmation``, …), so runs on different streams are independent. Blocks are generated until 60
are supported (the largest look) or 80 have started, so every look design, α, floor and rule is
evaluated on the same replicates. The spread calibration uses (base_seed, "spread", key name) in
every run, so a scenario's σ_d does not depend on the stream. Runs before revision 4 used 31-bit
``derive_seed`` seeds; their draws differ from this code's.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, least_squares
from scipy.special import (
    expit,
    gammainccinv,
    gammaincinv,
    logit,
    ndtr,
    ndtri,
    roots_genlaguerre,
)
from scipy.stats import norm
from scipy.stats import t as student_t

# The mechanism cohort's analysis (amendment 4, section 3), read only: the simulation computes
# its intervals with these functions and reads the cohort's constants from there.
from .study2_short import (
    ADAMW,
    COMPARISON_LEVEL,
    FA,
    MECHANISM_CAP,
    MECHANISM_EFFECTS,
    MECHANISM_MINIMUM,
    MECHANISM_TARGET,
    PREWARMED,
    _endpoint,
    _upper_quantile,
    block_support,
    max_t_interval,
    symmetric_studentized,
)


def seed_sequence(base_seed: int, *keys: object) -> np.random.SeedSequence:
    """The seed of the stream named by ``keys``: the full SHA-256 digest of the key list (as JSON,
    like ``derive_seed``, which keeps only 31 bits), so streams do not collide in practice."""
    blob = json.dumps([base_seed, *keys]).encode()
    return np.random.SeedSequence(np.frombuffer(hashlib.sha256(blob).digest(), dtype=np.uint32))


def generator(base_seed: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(seed_sequence(base_seed, *keys))


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
SHAPES = ("normal", "skewed", "rare_centered", "rare_only", "heavy")
RARE = 0.1  # π
# heavy: a block's arm effects are σ_d·S·w, S = 3 or 1 (P(S = 3) = 0.1) rescaled to E[S²] = 1, one
# S per block shared by the arms (a normal scale mixture; excess kurtosis 5.3).
HEAVY_SHARE = 0.1
HEAVY_RATIO = 3.0
HEAVY_SCALES = tuple(
    s / math.sqrt(HEAVY_SHARE * HEAVY_RATIO**2 + (1 - HEAVY_SHARE)) for s in (HEAVY_RATIO, 1.0)
)
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
# Targets at the stopping look: per contrast 97.5 % (over the two looks), jointly 95 %.
SECONDARY_MARGINAL_TARGET = 0.975
SECONDARY_JOINT_TARGET = 0.95
SECONDARY_REPS = 10_000  # replicates per scenario of the secondary check
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
INTERVALS = ("t", "bootstrap_t")
BOOTSTRAP_RESAMPLES = 2000
ZERO_SE = 1e-12  # an SE at or below this is zero (identical values up to rounding)
BOOTSTRAP_DESIGNS = ("24_50", "36_60")
BOOTSTRAP_DELTAS = (0.0, 0.025, -0.025, 0.05, -0.05, 0.055, -0.055)
BOOTSTRAP_BOUNDARY_REPS = 8_000  # sized to the revision-3 time budget
BOOTSTRAP_OTHER_REPS = 2_000
# Decision rules: ``eq_min36`` concludes equivalence only at a look with ≥ 36 supported blocks
# (so not at a cap look below that); a direction is unaffected.
RULES = ("standard", "eq_min36")
RULE_MIN_BLOCKS = {"standard": 0, "eq_min36": 36}
# The validation gate before confirmation: 12 fresh blocks, passed with ≥ 10 jointly supported.
GATE_BLOCKS = 12
GATE_MIN_SUPPORTED = 10
DECIDED_DESIGN = "36_60"
DECIDED_ALPHA = 0.04
DECIDED_ALLOWANCE = 0.010841  # b at w_max 0.47 (revision 5: kink, the largest family)
DECIDED_W_MAX = 0.47  # revision 5
DECIDED_SE_FLOOR = 0.0  # revision 4: a floor of 1 or 2 pp never binds at the target SDs
DECIDED_RULE = "eq_min36"  # revision 4: equivalence only at a look with ≥ 36 supported blocks
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
    "sigma_u_arms": "sua",
    "sigma_e_arms": "sea",
    "failure_arms": "fa",
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
    # Amendment 4: per-arm K spreads and technical failure rates (A, F, P), e.g. B = 20's inputs
    # record (``read_inputs``). None: the scalar fields hold for every arm. A per-arm failure
    # rate replaces ``failure`` (each arm's marginal; ``failure_correlated`` still applies).
    sigma_u_arms: tuple[float, float, float] | None = None
    sigma_e_arms: tuple[float, float, float] | None = None
    failure_arms: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        gaps = [d.sizes[1] - d.sizes[0] for d in LOOK_DESIGNS if len(d.sizes) == 2]
        arms = [v for v in (self.sigma_u_arms, self.sigma_e_arms, self.failure_arms) if v]
        checks = {
            "positions": self.positions in POSITIONS,
            "shape": self.shape in SHAPES,
            "family": self.family in FAMILIES,
            "miss": self.miss in MISSES,
            "rho_b, rho_a": 0 <= self.rho_b <= self.rho_a <= 1,
            "postpone": 0 <= self.postpone < min(gaps),
            "failure": 0 <= self.failure < 1,
            "per-arm inputs": all(
                isinstance(v, tuple) and len(v) == len(ARMS) and all(x >= 0 for x in v)
                for v in arms
            )
            and all(0 <= f < 1 for f in self.failure_arms or ()),
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"Invalid scenario fields {bad}: {self}")

    @property
    def design(self) -> str:
        return "interp" if self.positions == "nominal" else "interp_grid"

    @property
    def name(self) -> str:
        parts = [self.design, self.shape, self.family, sd_label(self.shape, self.sd)]
        parts.append(f"d{_pp(self.delta)}")
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name not in TAGS or value == f.default:
                continue
            tag = TAGS[f.name]
            if isinstance(value, bool):
                parts.append(tag)
            elif isinstance(value, str):
                parts.append(f"{tag}-{value}")
            elif isinstance(value, tuple):
                parts.append(tag + "-".join(f"{v:g}" for v in value))
            else:
                parts.append(f"{tag}{value:g}")
        return "_".join(parts)

    def k_spreads(self, arms: Sequence[int]) -> tuple[Any, Any]:
        """(σ_u, σ_e) of the ``arms``: the scalars, or (len(arms), 1) arrays of the per-arm
        values, which broadcast over the (blocks, arms, rates) endpoint positions."""
        u = self.sigma_u if self.sigma_u_arms is None else np.array(self.sigma_u_arms)[arms, None]
        e = self.sigma_e if self.sigma_e_arms is None else np.array(self.sigma_e_arms)[arms, None]
        return u, e


def _pp(value: float) -> str:
    return f"{round(100 * value, 6):+g}"


def sd_label(shape: str, sd: float) -> str:
    """The target SD as scenario names show it: "sd5", or "sdna" for rare_only (no target)."""
    return "sdna" if shape == "rare_only" else f"sd{round(100 * sd, 6):g}"


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
    # Per arm (A, F, P), when the scenario has per-arm failure rates: (intercept, slope).
    failure_arms: tuple[tuple[float, float], ...] | None = None


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
    if shape == "heavy":
        nodes, weights = _hermite()
        high, low = HEAVY_SCALES
        values = spread * np.concatenate([high * nodes, low * nodes])
        return values, np.concatenate([HEAVY_SHARE * weights, (1 - HEAVY_SHARE) * weights])
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


def fit_coefficients(
    family: str, arm: int, sigma_v: float, effect: tuple[str, float] | None
) -> tuple[tuple[float, float, float], float]:
    """Per-prompt (s, q, γ) whose population curve best matches the arm's target on
    ``FIT_GRID`` (least squares; the offset is fitted too), and the fit's RMSE.

    ``effect`` is the arm's effect (shape, spread), or None. γ is free only where the target has
    a kink.
    """
    return fit_target(FAMILIES[family], arm, sigma_v, effect)


@functools.cache
def fit_target(
    target: Family, arm: int, sigma_v: float, effect: tuple[str, float] | None
) -> tuple[tuple[float, float, float], float]:
    """``fit_coefficients`` for a family given as its target curves (the mechanism's tied
    stress case uses ``study1``'s curves at another baseline)."""
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
        both = s.shape in ("normal", "heavy")
        shifted = (True, True, True) if both else (True, False, False)
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
    per_arm = None
    if s.failure_arms is not None:
        per_arm = tuple((_logistic_intercept(f, slope), slope) for f in s.failure_arms)
    return Parameters(
        offsets=offsets,
        coefficients=tuple(fit[0] for fit in fits),
        fit_rmse=tuple(fit[1] for fit in fits),
        spread=float(spread),
        shifted=shifted,
        failure=(_logistic_intercept(s.failure, slope), slope),
        truth={"A-F": h[0] - h[1], "P-F": h[2] - h[1], "A-P": h[0] - h[2]},
        failure_arms=per_arm,
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
    scenario: Scenario,
    params: Parameters,
    latent: np.ndarray,
    arms: Sequence[int],
    tail: np.ndarray | None = None,
) -> np.ndarray:
    """d_{b,a} (blocks, arms) from the standard normal driving variable ``latent``; ``tail``
    (uniform, one per block) picks the block's scale for the ``heavy`` shape."""
    effect = np.zeros_like(latent)
    for j, arm in enumerate(arms):
        if not params.shifted[arm]:
            continue
        w = latent[:, j]
        if scenario.shape == "normal":
            effect[:, j] = params.spread * w
        elif scenario.shape == "heavy":
            if tail is None:
                raise ValueError("The heavy shape needs a block scale draw")
            scale = np.where(tail < HEAVY_SHARE, *HEAVY_SCALES)
            effect[:, j] = params.spread * scale * w
        elif scenario.shape == "skewed":
            gamma = gammaincinv(GAMMA_SHAPE, ndtr(w))
            effect[:, j] = params.spread * (gamma - GAMMA_SHAPE) / math.sqrt(GAMMA_SHAPE)
        elif scenario.shape == "rare_centered":
            effect[:, j] = params.spread * ((w > RARE_THRESHOLD) - RARE)
        else:
            effect[:, j] = params.spread * (w > RARE_THRESHOLD)
    return effect


def failure_probability(params: Parameters, stream: np.ndarray, arms: Sequence[int]) -> np.ndarray:
    """Each run's technical failure probability, broadcasting over (blocks, arms, rates):
    logistic(intercept + slope·β_b/σ_s), with one law for every arm or, for per-arm failure
    rates, each arm's own."""
    if params.failure_arms is None:
        intercept, slope = params.failure
        return expit(intercept + slope * stream)[:, None, None]
    law = np.array(params.failure_arms)[list(arms)]
    return expit(law[:, 0] + law[:, 1] * stream[:, None])[..., None]


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
    tail = rng.random(n)  # the heavy shape's block scale
    effect = block_effects(s, params, latent, arms, tail)
    nominal = np.array(POSITIONS[s.positions])[arms] + misses[arms]
    sigma_u, sigma_e = s.k_spreads(arms)
    x = nominal + sigma_u * speed[..., None] + sigma_e * rng.standard_normal((n, len(arms), 3))
    failed = rng.random((n, len(arms), 3)) < failure_probability(params, stream, arms)
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
    seconds: float = 0.0  # wall time of the simulation (bootstrap included)
    cpu_seconds: float = 0.0  # its process CPU time
    # Bootstrap-t, per look key (``look_key``): T* quantiles (reps, floors, probabilities) and
    # the resamples dropped for a zero SE (reps, floors).
    bootstrap: dict[str, np.ndarray] = field(default_factory=dict)
    dropped: dict[str, np.ndarray] = field(default_factory=dict)
    bootstrap_cpu_seconds: float = 0.0
    # Blocks of the validation gate (12 fresh blocks) jointly supported in A and F, (reps,).
    gate_supported: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    # Prescreened runs keep only the studies that pass the gate: their replicate indices, and
    # how many gates were drawn in all (0: no prescreen, every replicate kept).
    rep_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    screened: int = 0
    # Bootstrap-t of the secondary contrasts, per look key: T* quantiles (reps, contrasts,
    # floors, 2) at (1 ∓ SECONDARY_LEVEL)/2 and the dropped resamples (reps, contrasts, floors).
    secondary_bootstrap: dict[str, np.ndarray] = field(default_factory=dict)
    secondary_dropped: dict[str, np.ndarray] = field(default_factory=dict)
    secondary_bootstrap_cpu_seconds: float = 0.0

    @property
    def gate_passed(self) -> np.ndarray:
        return self.gate_supported >= GATE_MIN_SUPPORTED

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
    scenario: Scenario,
    params: Parameters,
    rep: int,
    *,
    base_seed: int,
    stream: str,
    prescreen: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """One study: blocks start until ``GENERATE_UNTIL`` are supported or the cap is reached.

    Also the validation gate before it: 12 fresh blocks of the same study (same prompts, rates
    and failure model) from the stream (…, rep, "validation_gate"); returns how many of them are
    jointly supported in A and F. With ``prescreen``, a study whose gate fails is not simulated
    (None). The gate has its own stream, so a kept study is the same as without the prescreen.
    """
    rng = generator(base_seed, stream, scenario.name, rep)
    alpha, misses = draw_constants(scenario, rng)
    gate_rng = generator(base_seed, stream, scenario.name, rep, "validation_gate")
    _, gate = simulate_blocks(scenario, params, alpha, misses, gate_rng, GATE_BLOCKS, arms=(0, 1))
    gate_supported = int(np.count_nonzero((gate == SUPPORTED).all(axis=1)))
    if prescreen and gate_supported < GATE_MIN_SUPPORTED:
        return None
    estimates = np.full((CAP, len(ARMS)), np.nan)
    causes = np.full((CAP, len(ARMS)), NOT_STARTED, dtype=np.int8)
    started = supported = 0
    while started < CAP and supported < GENERATE_UNTIL:
        n = min(CHUNK_BLOCKS, CAP - started)
        estimate, cause = simulate_blocks(scenario, params, alpha, misses, rng, n)
        estimates[started : started + n], causes[started : started + n] = estimate, cause
        supported += int(np.count_nonzero((cause[:, :2] == SUPPORTED).all(axis=1)))
        started += n
    return estimates, causes, started, gate_supported


def simulate_replicates(
    scenario: Scenario,
    params: Parameters,
    reps: Sequence[int],
    *,
    base_seed: int,
    stream: str,
    prescreen: bool = False,
) -> Replicates:
    """Replicates ``reps``; with ``prescreen`` only those that pass the validation gate."""
    clock, cpu = time.perf_counter(), time.process_time()
    kept, studies = [], []
    for r in reps:
        study = simulate_replicate(
            scenario, params, r, base_seed=base_seed, stream=stream, prescreen=prescreen
        )
        if study is not None:
            kept.append(r)
            studies.append(study)
    estimates = np.full((0, CAP, len(ARMS)), np.nan)
    causes = np.full((0, CAP, len(ARMS)), NOT_STARTED, dtype=np.int8)
    started, gate = np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    if studies:
        estimates, causes = np.stack([s[0] for s in studies]), np.stack([s[1] for s in studies])
        started, gate = np.array([s[2] for s in studies]), np.array([s[3] for s in studies])
    return Replicates(
        estimates,
        causes,
        started,
        time.perf_counter() - clock,
        time.process_time() - cpu,
        gate_supported=gate,
        rep_ids=np.array(kept, dtype=int),
        screened=len(reps) if prescreen else 0,
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
    scenario: Scenario,
    spread: float,
    *,
    seed: int | np.random.SeedSequence,
    replicates: int,
    blocks: int,
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
    seed = seed_sequence(base_seed, "spread", key.name)
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


def look_indices(
    primary: np.ndarray, sizes: tuple[int, ...] = LOOKS, cap: int = CAP
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each look happens, from Δ̂_b in started order (NaN where unsupported).

    Returns ``reached`` (R, 2), the index of the look's last block (R, 2) and ``at_cap`` (R,): the
    terminal look was taken at the cap below its minimum. A one-look design has a look 1 that is
    never reached.
    """
    if len(sizes) == 2 and not 0 < sizes[0] < sizes[1]:
        raise ValueError(f"Look sizes must increase: {sizes}")
    if len(sizes) not in (1, 2):
        raise ValueError(f"One or two looks, not {sizes}")
    primary = np.asarray(primary, dtype=float)[:, :cap]
    reps, blocks = primary.shape
    count = np.cumsum(np.isfinite(primary), axis=1)
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
    return np.stack(reached, axis=1), np.stack(index, axis=1), at_cap


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
    primary = np.asarray(primary, dtype=float)[:, :cap]
    reps, blocks = primary.shape
    supported = np.isfinite(primary)
    reached, index, at_cap = look_indices(primary, sizes, cap)
    included = supported[:, None, :] & (np.arange(blocks) <= index[..., None])  # (R, 2, blocks)
    n, mean, sd = _moments(primary[:, None, :], included)
    if secondary is None:
        secondary = np.full((reps, blocks, 2), np.nan)
    secondary = np.asarray(secondary, dtype=float)[:, :cap]
    cohort = included & np.isfinite(secondary[..., 0])[:, None, :]
    moments = [_moments(secondary[:, None, :, k], cohort) for k in range(2)]
    return LookStatistics(
        reached=reached,
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


def look_key(design: LookDesign, look: int) -> str:
    """Names a look of a design (0: look 1, 1: the terminal look)."""
    return f"{design.name}:{look}"


def _design_looks(design: LookDesign) -> tuple[int, ...]:
    """The look indices a design uses: (0, 1) for two looks, (1,) for one."""
    return (0, 1) if len(design.sizes) == 2 else (1,)


def _full_levels(levels: Sequence[float]) -> tuple[float | None, float]:
    """(look-1 level or None, terminal level)."""
    return (None, levels[0]) if len(levels) == 1 else (levels[0], levels[1])


def bootstrap_probabilities(
    designs: Sequence[LookDesign], alphas: Sequence[float]
) -> dict[str, np.ndarray]:
    """Per look key, the T* quantile probabilities the levels of every α need: (1 ∓ level)/2."""
    probabilities = {}
    for design in designs:
        levels = [_full_levels(design.levels(alpha)) for alpha in alphas]
        for look in _design_looks(design):
            values = {p for lv in levels for p in ((1 - lv[look]) / 2, (1 + lv[look]) / 2)}
            probabilities[look_key(design, look)] = np.array(sorted(values))
    return probabilities


def bootstrap_t_quantiles(
    values: np.ndarray,
    probabilities: np.ndarray,
    floors: Sequence[float],
    resamples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Studentized bootstrap of the mean: quantiles of T* = (mean* − mean)/se* per floor.

    se* = max(sd*, floor)/√n, as for the observed se. A resample with se* = 0 (floor 0) has no
    T*: it is dropped and counted. Returns quantiles (floors, probabilities), NaN with fewer than
    2 values or 2 kept resamples, and the dropped counts (floors,).
    """
    n = values.size
    quantiles = np.full((len(floors), len(probabilities)), np.nan)
    dropped = np.zeros(len(floors))
    if n < 2:
        return quantiles, dropped
    samples = values[rng.integers(0, n, (resamples, n))]
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    centre = values.mean()
    for f, floor in enumerate(floors):
        se = np.maximum(sds, floor) / math.sqrt(n)
        kept = se > ZERO_SE
        dropped[f] = resamples - kept.sum()
        if kept.sum() >= 2:
            quantiles[f] = _sorted_quantiles(
                np.sort((means[kept] - centre) / se[kept]), probabilities
            )
    return quantiles, dropped


def _sorted_quantiles(ordered: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """np.quantile's default (linear) quantiles of an already sorted array, for many
    probabilities at once."""
    position = probabilities * (ordered.size - 1)
    below = np.floor(position).astype(int)
    above = np.minimum(below + 1, ordered.size - 1)
    return ordered[below] + (position - below) * (ordered[above] - ordered[below])


def bootstrap_tables(
    scenario: Scenario,
    primary: np.ndarray,
    reps: Sequence[int],
    *,
    designs: Sequence[LookDesign],
    alphas: Sequence[float],
    floors: Sequence[float],
    resamples: int,
    base_seed: int,
    stream: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """T* quantiles and dropped resamples per look key for replicates ``reps`` (rows of
    ``primary``). Each replicate, design and look has its own stream (base_seed, stream, name,
    rep, "bootstrap", design, look), so the resamples of a design do not depend on which other
    designs are evaluated. A look 1 that is never reached is left NaN."""
    probabilities = bootstrap_probabilities(designs, alphas)
    quantiles = {
        key: np.full((len(reps), len(floors), p.size), np.nan) for key, p in probabilities.items()
    }
    dropped = {key: np.zeros((len(reps), len(floors))) for key in probabilities}
    supported = np.isfinite(primary)
    where = {d.name: look_indices(primary, d.realized(scenario.postpone)) for d in designs}
    for i, rep in enumerate(map(int, reps)):  # plain ints: the stream key is JSON
        for design in designs:
            reached, index, _ = where[design.name]
            for look in _design_looks(design):
                if not reached[i, look]:
                    continue
                key = look_key(design, look)
                stop = index[i, look] + 1
                values = primary[i, :stop][supported[i, :stop]]
                rng = generator(
                    base_seed, stream, scenario.name, rep, "bootstrap", design.name, look
                )
                quantiles[key][i], dropped[key][i] = bootstrap_t_quantiles(
                    values, probabilities[key], floors, resamples, rng
                )
    return quantiles, dropped


def secondary_probabilities(level: float = SECONDARY_LEVEL) -> np.ndarray:
    """The T* quantile probabilities of a secondary interval: (1 ∓ level)/2."""
    return np.array([(1 - level) / 2, (1 + level) / 2])


def secondary_bootstrap_tables(
    scenario: Scenario,
    primary: np.ndarray,
    secondary: np.ndarray,
    reps: Sequence[int],
    *,
    designs: Sequence[LookDesign],
    floors: Sequence[float],
    resamples: int,
    base_seed: int,
    stream: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per look key, the secondary contrasts' T* quantiles (reps, contrasts, floors, 2) at
    ``secondary_probabilities()`` and the dropped resamples (reps, contrasts, floors), for
    replicates ``reps`` (rows of ``primary`` and ``secondary``, as ``Replicates`` holds them).

    The values of a look are its three-arm cohort, as in ``look_statistics``: the blocks up to
    the look's last one where all three arms are supported. Each replicate, contrast, design and
    look has its own stream (base_seed, stream, name, rep, "bootstrap-secondary", contrast,
    design, look). A look 1 that is never reached is left NaN."""
    probabilities = secondary_probabilities()
    contrasts = len(SECONDARY_CONTRASTS)
    keys = [look_key(d, look) for d in designs for look in _design_looks(d)]
    quantiles = {key: np.full((len(reps), contrasts, len(floors), 2), np.nan) for key in keys}
    dropped = {key: np.zeros((len(reps), contrasts, len(floors))) for key in keys}
    cohort = np.isfinite(secondary[..., 0])  # all three arms supported
    where = {d.name: look_indices(primary, d.realized(scenario.postpone)) for d in designs}
    for i, rep in enumerate(map(int, reps)):  # plain ints: the stream key is JSON
        for design in designs:
            reached, index, _ = where[design.name]
            for look in _design_looks(design):
                if not reached[i, look]:
                    continue
                key = look_key(design, look)
                stop = index[i, look] + 1
                for k, contrast in enumerate(SECONDARY_CONTRASTS):
                    values = secondary[i, :stop, k][cohort[i, :stop]]
                    rng = generator(
                        base_seed,
                        stream,
                        scenario.name,
                        rep,
                        "bootstrap-secondary",
                        contrast,
                        design.name,
                        look,
                    )
                    quantiles[key][i, k], dropped[key][i, k] = bootstrap_t_quantiles(
                        values, probabilities, floors, resamples, rng
                    )
    return quantiles, dropped


def _probability_index(probabilities: np.ndarray, p: float) -> int:
    i = int(np.argmin(np.abs(probabilities - p)))
    if not math.isclose(probabilities[i], p, abs_tol=1e-12):
        raise KeyError(f"No bootstrap quantile at probability {p}")
    return i


def bootstrap_bounds(
    stats: LookStatistics,
    design: LookDesign,
    levels: Sequence[float],
    floor_index: int,
    floor: float,
    quantiles: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Bootstrap-t intervals per look: [mean − q_{(1+ℓ)/2}·se, mean − q_{(1−ℓ)/2}·se], se =
    max(sd, floor)/√n. A zero se gives [mean, mean]; a look without quantiles (fewer than 2
    blocks or kept resamples) an unbounded interval. None for a design's missing look 1."""
    bounds: list[tuple[np.ndarray, np.ndarray] | None] = []
    for look, level in enumerate(_full_levels(levels)):
        if level is None:
            bounds.append(None)
            continue
        key = look_key(design, look)
        q = quantiles[key][:, floor_index]
        lower = q[:, _probability_index(probabilities[key], (1 - level) / 2)]
        upper = q[:, _probability_index(probabilities[key], (1 + level) / 2)]
        bounds.append(
            studentized_bounds(
                stats.mean[:, look], stats.sd[:, look], stats.n[:, look], lower, upper, floor
            )
        )
    return bounds


def studentized_bounds(
    mean: np.ndarray,
    sd: np.ndarray,
    n: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """The bootstrap-t interval from the T* quantiles ``lower`` and ``upper``: [mean −
    upper·se, mean − lower·se], se = max(sd, floor)/√n. A zero se gives [mean, mean]; fewer than
    2 values or a missing quantile (NaN) an unbounded interval."""
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.maximum(sd, floor) / np.sqrt(n)
        low, high = mean - upper * se, mean - lower * se
    flat = (n >= 2) & (se <= ZERO_SE)
    low, high = np.where(flat, mean, low), np.where(flat, mean, high)
    bad = ~flat & ((n < 2) | ~np.isfinite(low) | ~np.isfinite(high))
    return np.where(bad, -np.inf, low), np.where(bad, np.inf, high)


def secondary_t_bounds(
    stats: LookStatistics, floor: float = 0.0, level: float = SECONDARY_LEVEL
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The secondary contrasts' t intervals per look, each (R, contrasts)."""
    return [
        t_interval(
            stats.secondary_mean[:, look],
            stats.secondary_sd[:, look],
            stats.secondary_n[:, look, None],
            level,
            floor,
        )
        for look in (0, 1)
    ]


def secondary_bootstrap_bounds(
    stats: LookStatistics,
    design: LookDesign,
    floor_index: int,
    floor: float,
    quantiles: Mapping[str, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """The secondary contrasts' bootstrap-t intervals per look, each (R, contrasts), from
    ``secondary_bootstrap_tables``; None for a design's missing look 1."""
    bounds: list[tuple[np.ndarray, np.ndarray] | None] = []
    for look in (0, 1):
        if look not in _design_looks(design):
            bounds.append(None)
            continue
        q = quantiles[look_key(design, look)][:, :, floor_index]  # (R, contrasts, 2)
        bounds.append(
            studentized_bounds(
                stats.secondary_mean[:, look],
                stats.secondary_sd[:, look],
                stats.secondary_n[:, look, None],
                q[..., 0],
                q[..., 1],
                floor,
            )
        )
    return bounds


def at_stopping_look(
    bounds: Sequence[tuple[np.ndarray, np.ndarray] | None], look: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per replicate, the (R, contrasts) bounds of the look where the primary stopped (``look``
    1 or 2); a missing look 1 is never the stopping look."""
    later_low, later_high = bounds[1]
    if bounds[0] is None:
        if np.any(look == 1):
            raise ValueError("A design without look 1 stopped at look 1")
        return later_low, later_high
    first = (np.asarray(look) == 1)[:, None]
    return np.where(first, bounds[0][0], later_low), np.where(first, bounds[0][1], later_high)


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
    # The stopping look may conclude equivalence (look-1 rule of the design, eq_min36).
    equivalence_allowed: np.ndarray | None = None

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
    equivalence_min_blocks: int = 0,
    bounds: Sequence[tuple[np.ndarray, np.ndarray] | None] | None = None,
    margin: float = MARGIN,
    wide: float = WIDE_MARGIN,
    secondary_level: float = SECONDARY_LEVEL,
    secondary_bounds: Sequence[tuple[np.ndarray, np.ndarray] | None] | None = None,
) -> Decisions:
    """Stop at look 1 on equivalence (if allowed there) or a direction; otherwise the terminal
    look decides. ``levels`` has one level per look of the design (one for a one-look design).
    Equivalence is concluded only at a look with at least ``equivalence_min_blocks`` supported
    blocks (rule ``eq_min36``: not at a cap look below 36); a direction is unaffected.
    The primary intervals are t intervals unless ``bounds`` gives them per look (None for a
    missing look 1), e.g. from ``bootstrap_bounds``; the secondary ones t intervals at
    ``secondary_level`` unless ``secondary_bounds`` gives them per look, e.g. from
    ``secondary_bootstrap_bounds``."""
    reps = stats.n.shape[0]
    first, terminal = _full_levels(levels)
    if bounds is None:
        bounds = [
            None
            if level is None
            else t_interval(stats.mean[:, k], stats.sd[:, k], stats.n[:, k], level, floor)
            for k, level in enumerate((first, terminal))
        ]
    unbounded = (np.full(reps, -np.inf), np.full(reps, np.inf))
    low1, high1 = bounds[0] if bounds[0] is not None else unbounded
    low2, high2 = bounds[1]
    allowed = stats.n >= equivalence_min_blocks  # (R, 2): equivalence may be concluded at the look
    equivalent1 = _inside(low1, high1, margin) & allowed[:, 0]
    if not equivalence_at_first:
        equivalent1 = np.zeros(reps, bool)
    stop1 = stats.reached[:, 0] & (equivalent1 | (low1 > 0) | (high1 < 0))
    k = np.where(stop1, 0, 1)
    rows = np.arange(reps)
    low = np.where(stop1, low1, low2)
    high = np.where(stop1, high1, high2)
    equivalence_allowed = (~stop1 | equivalence_at_first) & allowed[rows, k]
    equivalence = _inside(low, high, margin) & equivalence_allowed
    secondary_n = stats.secondary_n[rows, k]
    if secondary_bounds is None:
        secondary_low, secondary_high = t_interval(
            stats.secondary_mean[rows, k],
            stats.secondary_sd[rows, k],
            secondary_n[:, None],
            secondary_level,
            floor,
        )
    else:
        secondary_low, secondary_high = at_stopping_look(secondary_bounds, k + 1)
    return Decisions(
        look=k + 1,
        low=low,
        high=high,
        estimate=stats.mean[rows, k],
        n=stats.n[rows, k],
        started=stats.started[rows, k],
        at_cap=~stop1 & stats.at_cap,
        equivalence=equivalence,
        equivalence_allowed=equivalence_allowed,
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
    a direction excludes [−b, b], equivalence lies inside ±(5 pp − b), and only where the
    stopping look may conclude equivalence at all (the design's look-1 rule, eq_min36)."""
    d = decisions
    allowed = (
        np.ones(d.low.shape, dtype=bool) if d.equivalence_allowed is None else d.equivalence_allowed
    )
    return {
        "equivalence": _inside(d.low, d.high, MARGIN - allowance) & allowed,
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
        **_gate_failure(data),
    }


def _gate_failure(data: Replicates) -> dict[str, float]:
    """P(the 12-block validation gate fails) and its MC SE, when the gate was simulated; for a
    prescreened run, from all the gates drawn (the kept studies all passed)."""
    if data.screened:
        p = 1 - data.started.size / data.screened
        return {
            "p_gate_fail": p,
            "se_gate_fail": math.sqrt(p * (1 - p) / data.screened),
            "gates_drawn": data.screened,
            "gates_passed": int(data.started.size),
        }
    if not data.gate_supported.size:
        return {}
    p, se = _rate(~data.gate_passed)
    return {
        "p_gate_fail": p,
        "se_gate_fail": se,
        "mean_gate_supported": float(data.gate_supported.mean()),
    }


def secondary_coverage(
    low: np.ndarray,
    high: np.ndarray,
    truth: Mapping[str, float],
    targets: Mapping[str, float],
    joint: bool = False,
) -> dict[str, dict[str, float]]:
    """Per secondary contrast, the share of the (R, contrasts) intervals covering its pointwise
    truth and its block mean (the interpolation estimand), with MC SEs; with ``joint``, also
    the share covering both contrasts at once (``"joint"``)."""
    coverage: dict[str, dict[str, float]] = {}
    both = {}
    for target, source in (("truth", truth), ("block_mean", targets)):
        values = np.array([source[name] for name in SECONDARY_CONTRASTS])
        covered = (low <= values) & (values <= high)
        for k, name in enumerate(SECONDARY_CONTRASTS):
            entry = coverage.setdefault(name, {})
            entry[target], entry[f"{target}_se"] = _rate(covered[:, k])
        both[target], both[f"{target}_se"] = _rate(covered.all(axis=1))
    if joint:
        coverage["joint"] = both
    return coverage


def outcome_rates(
    decisions: Decisions,
    delta: float,
    truth: Mapping[str, float],
    targets: Mapping[str, float],
    allowance: float | None = None,
    gate_passed: np.ndarray | None = None,
) -> dict[str, Any]:
    """Decision rates, their Monte Carlo SEs and the stopping-time summaries of one design,
    level pair and floor; with an allowance, the pointwise readings too; with the validation
    gate, the decision rates among the studies that pass it (``gated``)."""
    d = decisions
    rates: dict[str, Any] = {}
    if gate_passed is not None and gate_passed.size:
        gated: dict[str, float] = {"passed": int(gate_passed.sum())}
        if gate_passed.any():
            power_flags = d.equivalence if delta == 0 else d.positive if delta > 0 else d.negative
            for name, flag in (
                ("equivalence", d.equivalence),
                ("positive", d.positive),
                ("negative", d.negative),
                ("power", power_flags),
            ):
                key = "power" if name == "power" else f"p_{name}"
                gated[key], gated[f"se_{name}"] = _rate(flag[gate_passed])
        rates["gated"] = gated
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
    coverage = secondary_coverage(d.secondary_low, d.secondary_high, truth, targets)
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
    intervals: Sequence[str] = ("t",),
    rules: Sequence[str] = ("standard",),
    secondary_intervals: Sequence[str] = ("t",),
) -> dict[str, Any]:
    """The scenario's block diagnostics and one result row per look design × interval × rule ×
    α × SE floor. ``bootstrap_t`` needs the replicates' bootstrap tables for the same designs,
    α's and floors, and as a secondary interval their secondary tables. Every row carries the
    scenario's attained paired SD (at its Δ). A row's ``secondary_coverage`` is that of the
    first secondary interval; with secondary intervals other than t alone, the row adds
    ``secondary_coverage_by_interval`` (each interval's coverage, the joint one included)."""
    diagnostics = block_diagnostics(params, data)
    primary, secondary = data.primary, data.secondary
    targets = diagnostics["secondary_block_mean"]
    probabilities = bootstrap_probabilities(designs, alphas) if "bootstrap_t" in intervals else {}
    if probabilities:
        diagnostics["bootstrap_dropped"] = {
            key: dict(zip(map(str, floors), data.dropped[key].mean(axis=0).tolist(), strict=True))
            for key in probabilities
        }
    by_interval = tuple(secondary_intervals) != ("t",)
    if "bootstrap_t" in secondary_intervals:
        diagnostics["secondary_bootstrap_dropped"] = {
            key: {
                contrast: dict(zip(map(str, floors), mean.tolist(), strict=True))
                for contrast, mean in zip(SECONDARY_CONTRASTS, dropped.mean(axis=0), strict=True)
            }
            for key, dropped in data.secondary_dropped.items()
        }
    results = []
    for design in designs:
        stats = look_statistics(primary, secondary, sizes=design.realized(scenario.postpone))
        for interval in intervals:
            for alpha in alphas:
                levels = design.levels(alpha)
                for f, floor in enumerate(floors):
                    bounds = None
                    if interval == "bootstrap_t":
                        bounds = bootstrap_bounds(
                            stats, design, levels, f, floor, data.bootstrap, probabilities
                        )
                    # Per secondary interval, its bounds per look. The decisions read the first
                    # one; t is sequential_decisions' own (as in revisions 1–5).
                    per_look: dict[str, Any] = {}
                    if by_interval:
                        per_look["t"] = secondary_t_bounds(stats, floor)
                    if "bootstrap_t" in secondary_intervals:
                        per_look["bootstrap_t"] = secondary_bootstrap_bounds(
                            stats, design, f, floor, data.secondary_bootstrap
                        )
                    for rule in rules:
                        decisions = sequential_decisions(
                            stats,
                            levels,
                            floor,
                            equivalence_at_first=design.equivalence_at_first,
                            equivalence_min_blocks=RULE_MIN_BLOCKS[rule],
                            bounds=bounds,
                            secondary_bounds=None
                            if secondary_intervals[0] == "t"
                            else per_look[secondary_intervals[0]],
                        )
                        # The scenario's Δ, not the solved truth: that is Δ only to ~1e-12, and
                        # the power at Δ = 0 is the equivalence rate.
                        rates = outcome_rates(
                            decisions,
                            scenario.delta,
                            params.truth,
                            targets,
                            allowance,
                            data.gate_passed,
                        )
                        rates["bias_vs_block_mean"] = (
                            rates["mean_estimate"] - diagnostics["block_mean"]
                        )
                        if by_interval:
                            rates["secondary_coverage_by_interval"] = {
                                name: secondary_coverage(
                                    *at_stopping_look(per_look[name], decisions.look),
                                    params.truth,
                                    targets,
                                    joint=True,
                                )
                                for name in secondary_intervals
                            }
                        results.append(
                            {
                                "looks": design.name,
                                "interval": interval,
                                "rule": rule,
                                "alpha": alpha,
                                "levels": list(levels),
                                "floor": floor,
                                "attained_sd": diagnostics["attained_sd"],
                                **rates,
                            }
                        )
    return {
        "name": scenario.name,
        "design": scenario.design,
        "settings": asdict(scenario),
        "reps": int(data.started.size),
        "screened": data.screened,
        "seconds": data.seconds,
        "cpu_seconds": data.cpu_seconds,
        "bootstrap_cpu_seconds": data.bootstrap_cpu_seconds,
        **(
            {"secondary_bootstrap_cpu_seconds": data.secondary_bootstrap_cpu_seconds}
            if "bootstrap_t" in secondary_intervals
            else {}
        ),
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


def bootstrap_validation_grid(
    boundary_reps: int = BOOTSTRAP_BOUNDARY_REPS, other_reps: int = BOOTSTRAP_OTHER_REPS
) -> list[Scenario]:
    """Revision 3: t against bootstrap-t over every shape × target SD × Δ ∈ {0, ±2.5, ±5,
    ±5.5}; ``boundary_reps`` at Δ = 0 and ±5 pp, ``other_reps`` elsewhere."""

    def reps(delta: float) -> int:
        return boundary_reps if _reps(delta) == BOUNDARY_REPS else other_reps

    grid = []
    for shape in ("normal", "skewed", "rare_centered"):
        group = "central" if shape == "normal" else "shapes"
        grid += [
            Scenario(delta=d, sd=sd, shape=shape, reps=reps(d), group=group)
            for sd in TARGET_SDS
            for d in BOOTSTRAP_DELTAS
        ]
    grid += [
        Scenario(delta=d, sd=0.0, shape="rare_only", reps=reps(d), group="shapes")
        for d in BOOTSTRAP_DELTAS
    ]
    return grid


def stress_validation_grid(
    boundary_reps: int = BOUNDARY_REPS, other_reps: int = OTHER_REPS
) -> list[Scenario]:
    """Revision 4: every non-shape group of ``validation_grid`` (postponed, cap, width, K spread,
    miss, families, stress), the cap group with independent failures as well, and the heavy
    shape over every target SD × Δ ∈ {0, ±2.5, ±5, ±5.5}."""

    def reps(delta: float) -> int:
        return boundary_reps if _reps(delta) == BOUNDARY_REPS else other_reps

    boundary = (0.0, MARGIN, -MARGIN)
    independent = [
        replace(s, reps=reps(s.delta))
        for s in _scenarios("cap", (0.05, 0.10), boundary, failure=0.2)
    ]
    taken = {s.name for s in independent}  # the stress group's f 0.2 scenarios join the cap group
    grid = [
        replace(s, reps=reps(s.delta))
        for s in validation_grid()
        if s.group not in ("central", "shapes") and s.name not in taken
    ]
    grid += independent
    grid += [
        Scenario(delta=d, sd=sd, shape="heavy", reps=reps(d), group="shapes")
        for sd in TARGET_SDS
        for d in BOOTSTRAP_DELTAS
    ]
    return grid


def confirmation_grid(reps: int = BOUNDARY_REPS) -> list[Scenario]:
    """Revision 4: the decided design's headline shape × SD grid at Δ ∈ {0, ±5}, meant for a
    fresh stream (``confirmation``)."""
    boundary = (0.0, MARGIN, -MARGIN)
    grid = []
    for shape in ("normal", "skewed", "rare_centered", "heavy"):
        group = "central" if shape == "normal" else "shapes"
        grid += _scenarios(group, TARGET_SDS, boundary, reps, shape=shape)
    grid += _scenarios("shapes", (0.0,), boundary, reps, shape="rare_only")
    return grid


def decided_bias_grid(w_max: float = DECIDED_W_MAX) -> list[Scenario]:
    """Revision 5: the bias allowance at the decided w_max (families × miss modes × Δ ∈ {0, 5})."""
    return [
        Scenario(family=f, miss=m, w_max=w_max, delta=d, group="bias")
        for f in ("study1", "high", "stress_slopes", "kink")
        for m in MISSES
        for d in (0.0, MARGIN)
    ]


def decided_grid(w_max: float = DECIDED_W_MAX, cap_reps: int = 10_000) -> list[Scenario]:
    """Revision 5 at the decided w_max: kink and stress_slopes at SD 5, every shape × target SD,
    and the cap scenarios (f 0.2, correlated and independent), at Δ ∈ {0, ±5}."""
    boundary = (0.0, MARGIN, -MARGIN)
    grid = [
        Scenario(family=f, w_max=w_max, delta=d, reps=BOUNDARY_REPS, group="families")
        for f in ("kink", "stress_slopes")
        for d in boundary
    ]
    for shape in ("normal", "skewed", "rare_centered", "heavy"):
        group = "central" if shape == "normal" else "shapes"
        grid += _scenarios(group, TARGET_SDS, boundary, BOUNDARY_REPS, shape=shape, w_max=w_max)
    grid += _scenarios("shapes", (0.0,), boundary, BOUNDARY_REPS, shape="rare_only", w_max=w_max)
    for correlated in (True, False):
        grid += _scenarios(
            "cap",
            (0.05, 0.10),
            boundary,
            cap_reps,
            failure=0.2,
            failure_correlated=correlated,
            w_max=w_max,
        )
    return grid


def decided_secondary_grid(
    w_max: float = DECIDED_W_MAX, reps: int = SECONDARY_REPS
) -> list[Scenario]:
    """The secondary check (2026-10-17): the decided design's shape grid, every shape × target SD
    at Δ ∈ {0, ±5}, ``reps`` replicates each; meant for a fresh stream (``secondary``)."""
    boundary = (0.0, MARGIN, -MARGIN)
    grid = []
    for shape in ("normal", "skewed", "rare_centered", "heavy"):
        group = "central" if shape == "normal" else "shapes"
        grid += _scenarios(group, TARGET_SDS, boundary, reps, shape=shape, w_max=w_max)
    grid += _scenarios("shapes", (0.0,), boundary, reps, shape="rare_only", w_max=w_max)
    return grid


def decided_gate_grid(
    w_max: float = DECIDED_W_MAX, correlated_gates: int = 150_000, independent_gates: int = 400_000
) -> list[Scenario]:
    """Revision 5: the cap scenarios at Δ = ±5 for a prescreened run (``RunSettings.prescreen``):
    ``reps`` is the number of validation gates drawn; only the studies that pass are simulated,
    which sizes the false equivalence among them (finding N2)."""
    return [
        Scenario(
            sd=sd,
            delta=d,
            failure=0.2,
            failure_correlated=correlated,
            w_max=w_max,
            reps=correlated_gates if correlated else independent_gates,
            group="cap",
        )
        for correlated in (True, False)
        for sd in (0.05, 0.10)
        for d in (MARGIN, -MARGIN)
    ]


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
    """One flat row per scenario × look design × interval × α × floor."""
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
                    # Rows written before revisions 3 and 4 carry no interval or rule.
                    "interval": "t",
                    "rule": "standard",
                    **r,
                }
            )
    return rows


def _design_intervals(rows: Sequence[Mapping]) -> list[tuple[str, str, str]]:
    return list(dict.fromkeys((r["looks"], r["interval"], r["rule"]) for r in rows))


def select_levels(scenarios: Sequence[Mapping], floors: Sequence[float] = SE_FLOORS) -> list[dict]:
    """Per look design, interval and SE floor, the largest α on the grid meeting every error
    constraint over the central scenarios at the binding SDs (3, 5, 10 pp); 15 pp is reported,
    not binding. Only the standard decision rule is calibrated."""
    rows = [
        r for r in result_rows(scenarios) if r["group"] == "central" and r["rule"] == "standard"
    ]
    selections = []
    for looks, interval, _ in _design_intervals(rows):
        for floor in floors:
            mine = [
                r
                for r in rows
                if r["looks"] == looks and r["interval"] == interval and r["floor"] == floor
            ]
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
                    "interval": interval,
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
    scenarios: Sequence[Mapping], chosen: Mapping[tuple[str, str, float], float] | None = None
) -> list[dict[str, Any]]:
    """Worst rates and lowest powers per look design × interval × rule × floor × α, per group
    and over all. An α is labelled ``selected`` when ``chosen`` maps the design, interval and
    floor to it."""
    rows = result_rows(scenarios)
    checks = []
    for looks, interval, rule in _design_intervals(rows):
        for floor in sorted({r["floor"] for r in rows}):
            for alpha in sorted({r["alpha"] for r in rows}):
                label = f"alpha_{alpha:.4f}"
                if chosen and chosen.get((looks, interval, floor)) == alpha:
                    label = "selected"
                mine = [
                    r
                    for r in rows
                    if r["looks"] == looks
                    and r["interval"] == interval
                    and r["rule"] == rule
                    and r["floor"] == floor
                    and r["alpha"] == alpha
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
                        "interval": interval,
                        "rule": rule,
                        "floor": floor,
                        "which": label,
                        "alpha": alpha,
                        "levels": mine[0]["levels"],
                        "groups": {g: constraint_summary(group) for g, group in groups.items()},
                    }
                )
    return checks


def secondary_checks(scenarios: Sequence[Mapping]) -> list[dict[str, Any]]:
    """Coverage of the secondary contrasts' pointwise truths at the stopping look, per look
    design × interval × rule × α × floor × secondary interval and per shape × target SD: the
    worst over Δ of each contrast and of both jointly, against 97.5 % per contrast and 95 %
    jointly. ``meets`` compares the estimates with the targets; ``short_2se`` flags an estimate
    more than 2 MC SEs below its target. Rows without ``secondary_coverage_by_interval`` are
    skipped."""
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for s in scenarios:
        settings = s["settings"]
        for r in s["results"]:
            for name, coverage in (r.get("secondary_coverage_by_interval") or {}).items():
                key = (r["looks"], r["interval"], r["rule"], r["alpha"], r["floor"], name)
                groups.setdefault(key, []).append(
                    {"scenario": s["name"], "shape": settings["shape"], "sd": settings["sd"],
                     "delta": settings["delta"], "reps": s["reps"], "coverage": coverage}
                )  # fmt: skip
    targets = {name: SECONDARY_MARGINAL_TARGET for name in SECONDARY_CONTRASTS}
    targets["joint"] = SECONDARY_JOINT_TARGET
    checks = []
    for (looks, interval, rule, alpha, floor, name), rows in groups.items():
        cells = []
        for shape, sd in dict.fromkeys((r["shape"], r["sd"]) for r in rows):
            mine = [r for r in rows if r["shape"] == shape and r["sd"] == sd]
            cell: dict[str, Any] = {"shape": shape, "sd": sd, "deltas": [r["delta"] for r in mine]}
            for which, target in targets.items():
                worst = min(mine, key=lambda r: r["coverage"][which]["truth"])
                value = worst["coverage"][which]["truth"]
                se = worst["coverage"][which]["truth_se"]
                cell[which] = {
                    "value": value,
                    "mc_se": se,
                    "delta": worst["delta"],
                    "scenario": worst["scenario"],
                    "per_delta": {_pp(r["delta"]): r["coverage"][which]["truth"] for r in mine},
                    "target": target,
                    "meets": value >= target,
                    "short_2se": value < target - 2 * se,
                }
            cell["meets"] = all(cell[which]["meets"] for which in targets)
            cells.append(cell)
        checks.append(
            {
                "looks": looks,
                "interval": interval,
                "rule": rule,
                "alpha": alpha,
                "floor": floor,
                "secondary_interval": name,
                "level": SECONDARY_LEVEL,
                "cells": cells,
                "meets": all(c["meets"] for c in cells),
                "failing": [
                    f"{c['shape']} {sd_label(c['shape'], c['sd'])}" for c in cells if not c["meets"]
                ],  # fmt: skip
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
    designs: tuple[str, ...] = tuple(d.name for d in LOOK_DESIGNS)
    intervals: tuple[str, ...] = ("t",)
    resamples: int = BOOTSTRAP_RESAMPLES  # bootstrap-t, per look and replicate
    rules: tuple[str, ...] = ("standard",)
    prescreen: bool = False  # simulate only the studies that pass the validation gate
    # The secondary contrasts' intervals; the first is the one the rows' secondary_coverage reads.
    secondary_intervals: tuple[str, ...] = ("t",)

    def __post_init__(self) -> None:
        unknown = (
            set(self.intervals) - set(INTERVALS)
            | set(self.secondary_intervals) - set(INTERVALS)
            | set(self.designs) - set(DESIGNS_BY_NAME)
            | set(self.rules) - set(RULES)
        )
        if unknown:
            raise ValueError(f"Unknown intervals, look designs or rules: {sorted(unknown)}")
        if not self.secondary_intervals:
            raise ValueError("At least one secondary interval is needed")

    @property
    def look_designs(self) -> tuple[LookDesign, ...]:
        return tuple(DESIGNS_BY_NAME[name] for name in self.designs)


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
    reps = range(start, stop)
    data = simulate_replicates(
        scenario,
        params,
        reps,
        base_seed=settings.base_seed,
        stream=settings.stream,
        prescreen=settings.prescreen,
    )
    if "bootstrap_t" in settings.intervals:
        clock, cpu = time.perf_counter(), time.process_time()
        quantiles, dropped = bootstrap_tables(
            scenario,
            data.primary,
            data.rep_ids,
            designs=settings.look_designs,
            alphas=settings.alphas,
            floors=settings.floors,
            resamples=settings.resamples,
            base_seed=settings.base_seed,
            stream=settings.stream,
        )
        cpu = time.process_time() - cpu
        data = replace(
            data,
            bootstrap=quantiles,
            dropped=dropped,
            seconds=data.seconds + time.perf_counter() - clock,
            cpu_seconds=data.cpu_seconds + cpu,
            bootstrap_cpu_seconds=cpu,
        )
    if "bootstrap_t" in settings.secondary_intervals:
        clock, cpu = time.perf_counter(), time.process_time()
        quantiles, dropped = secondary_bootstrap_tables(
            scenario,
            data.primary,
            data.secondary,
            data.rep_ids,
            designs=settings.look_designs,
            floors=settings.floors,
            resamples=settings.resamples,
            base_seed=settings.base_seed,
            stream=settings.stream,
        )
        cpu = time.process_time() - cpu
        data = replace(
            data,
            secondary_bootstrap=quantiles,
            secondary_dropped=dropped,
            seconds=data.seconds + time.perf_counter() - clock,
            cpu_seconds=data.cpu_seconds + cpu,
            secondary_bootstrap_cpu_seconds=cpu,
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
                    {
                        k: np.concatenate([c.bootstrap[k] for c in chunks])
                        for k in chunks[0].bootstrap
                    },
                    {k: np.concatenate([c.dropped[k] for c in chunks]) for k in chunks[0].dropped},
                    sum(c.bootstrap_cpu_seconds for c in chunks),
                    np.concatenate([c.gate_supported for c in chunks]),
                    np.concatenate([c.rep_ids for c in chunks]),
                    sum(c.screened for c in chunks),
                    secondary_bootstrap={
                        k: np.concatenate([c.secondary_bootstrap[k] for c in chunks])
                        for k in chunks[0].secondary_bootstrap
                    },
                    secondary_dropped={
                        k: np.concatenate([c.secondary_dropped[k] for c in chunks])
                        for k in chunks[0].secondary_dropped
                    },
                    secondary_bootstrap_cpu_seconds=sum(
                        c.secondary_bootstrap_cpu_seconds for c in chunks
                    ),
                )
                scenario = by_name[name]
                if merged.started.size:
                    summary = summarize_scenario(
                        scenario,
                        params[name],
                        merged,
                        settings.alphas,
                        settings.floors,
                        settings.allowance,
                        settings.look_designs,
                        settings.intervals,
                        settings.rules,
                        settings.secondary_intervals,
                    )
                else:  # a prescreen that kept no study
                    summary = {
                        "name": name,
                        "design": scenario.design,
                        "settings": asdict(scenario),
                        "reps": 0,
                        "diagnostics": _gate_failure(merged),
                        "results": [],
                    }
                key = calibration_key(scenario).name
                summary["calibration"] = asdict(calibrations[key]) if key in calibrations else None
                summaries[name] = summary
            if progress is not None:
                progress("replicates", done, len(tasks))
    return {
        "calibrations": {k: asdict(v) for k, v in calibrations.items()},
        "scenarios": [summaries[name] for name in names],
    }


# Amendment 4: B = 20's inputs record

INPUT_FIELDS = ("sigma_u", "sigma_e", "support_rate", "failure_rate")
INPUT_ENTRY_KEYS = ("value", "source", "provisional", "counts")
INPUT_TOP_KEYS = ("batch", "status", "description", "arms")
INPUT_STATUSES = ("provisional", "measured")
INPUTS_RECORD = "docs/study2-short-b20-inputs.json"  # relative to the repository root


@dataclass(frozen=True)
class Inputs:
    """B = 20's inputs record (``read_inputs``): per arm (A, F, P) the K spreads σ_u and σ_e
    (ln K units), the support rate and the technical failure rate, and the record's sha256.

    The simulation uses the K spreads and the failure rates. The support rate is the measured
    share of supported blocks, which a run reports beside its simulated one: in the model,
    support follows from the K spreads, the selection miss, w_max and the failures."""

    batch: int
    status: str
    sigma_u: tuple[float, float, float]
    sigma_e: tuple[float, float, float]
    support_rate: tuple[float, float, float]
    failure_rate: tuple[float, float, float]
    provisional: tuple[str, ...]  # "arm.field: what replaces it", one per provisional entry
    sha256: str
    path: str

    def summary(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "batch": self.batch,
            "status": self.status,
            "provisional": list(self.provisional),
            **{name: dict(zip(ARMS, getattr(self, name), strict=True)) for name in INPUT_FIELDS},
        }


def _input_value_problem(name: str, value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return "is not a finite number"
    if name in ("sigma_u", "sigma_e") and not value > 0:
        return "is not a positive K spread"
    if name == "support_rate" and not 0 < value <= 1:
        return "is not a support rate in (0, 1]"
    if name == "failure_rate" and not 0 <= value < 1:
        return "is not a failure rate in [0, 1)"
    return None


def read_inputs(path: str | Path) -> Inputs:
    """B = 20's inputs record, checked: a JSON object with exactly ``INPUT_TOP_KEYS`` (``batch``
    a positive integer, ``status`` one of ``INPUT_STATUSES``, ``description`` a string, ``arms``);
    in ``arms`` exactly A, F and P, each with exactly ``INPUT_FIELDS``, each an object with a
    ``value`` and a non-empty ``source``, optionally ``provisional`` (what replaces it, a
    non-empty string) and ``counts`` (integers ≥ 0). The status is ``provisional`` exactly when an
    entry is. A missing, unknown or malformed field raises ``ValueError``; none is defaulted."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        record = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: the inputs record is not JSON ({error})") from None
    if not isinstance(record, dict):
        raise ValueError(f"{path}: the inputs record is a JSON object")
    problems = []
    missing = [k for k in INPUT_TOP_KEYS if k not in record]
    unknown = sorted(set(record) - set(INPUT_TOP_KEYS))
    if missing:
        problems.append(f"lacks {missing}")
    if unknown:
        problems.append(f"has keys {unknown} it does not know")
    batch, status = record.get("batch"), record.get("status")
    if "batch" in record and (isinstance(batch, bool) or not isinstance(batch, int) or batch < 1):
        problems.append(f"batch {batch!r} is not a positive integer")
    if "status" in record and status not in INPUT_STATUSES:
        problems.append(f"status {status!r} is not one of {list(INPUT_STATUSES)}")
    if "description" in record and not isinstance(record["description"], str):
        problems.append("description is not a string")
    arms = record.get("arms")
    values: dict[str, list[float]] = {name: [] for name in INPUT_FIELDS}
    provisional = []
    if not isinstance(arms, dict) or sorted(arms) != sorted(ARMS):
        shown = sorted(arms) if isinstance(arms, dict) else arms
        problems.append(f"arms {shown!r} are not exactly {list(ARMS)}")
        arms = {}
    for arm in ARMS if arms else ():
        entries = arms[arm]
        if not isinstance(entries, dict) or sorted(entries) != sorted(INPUT_FIELDS):
            problems.append(f"arm {arm} does not hold exactly {list(INPUT_FIELDS)}")
            continue
        for name in INPUT_FIELDS:
            entry, where = entries[name], f"{arm}.{name}"
            if not isinstance(entry, dict):
                problems.append(f"{where} is not an object")
                continue
            lacking = [k for k in ("value", "source") if k not in entry]
            extra = sorted(set(entry) - set(INPUT_ENTRY_KEYS))
            if lacking or extra:
                problems.append(f"{where} lacks {lacking} or has unknown keys {extra}")
                continue
            why = _input_value_problem(name, entry["value"])
            if why:
                problems.append(f"{where} {entry['value']!r} {why}")
            else:
                values[name].append(float(entry["value"]))
            if not (isinstance(entry["source"], str) and entry["source"].strip()):
                problems.append(f"{where} has no source")
            if "provisional" in entry:
                if not (isinstance(entry["provisional"], str) and entry["provisional"].strip()):
                    problems.append(f"{where} provisional is not a non-empty string")
                else:
                    provisional.append(f"{where}: {entry['provisional']}")
            counts = entry.get("counts", {})
            if not (isinstance(counts, dict) and all(
                    isinstance(v, int) and not isinstance(v, bool) and v >= 0
                    for v in counts.values())):  # fmt: skip
                problems.append(f"{where} counts are not integers ≥ 0")
    if not problems and (status == "provisional") != bool(provisional):
        problems.append(f"status {status!r} but {len(provisional)} provisional entries")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return Inputs(
        batch=batch,
        status=status,
        **{name: tuple(values[name]) for name in INPUT_FIELDS},
        provisional=tuple(provisional),
        sha256=digest,
        path=str(path),
    )


INPUT_RUN_ARMS = {"A": ADAMW, "F": FA, "P": PREWARMED}  # the runs' arm names


def measure_inputs(
    runs: Sequence[Any],
    *,
    batch: int,
    seeds: Sequence[int],
    rates: Mapping[str, Sequence[float]] | None = None,
    kstar: float = K_STAR,
    w_max: float = DECIDED_W_MAX,
) -> dict[str, dict[str, Any]]:
    """The inputs, per arm (A, F, P), measured on 25-step runs (``stage_a.Trajectory``) at
    ``batch`` on the ``seeds``, from K and support only (K_benign at step 25, the stop; never a
    label or K_harmful):

    - ``sigma_u`` and ``sigma_e``: over the rates where every seed has an endpoint, ln K25 as
      rate × seed; σ_u is the SD (ddof 1) of the seed effects (column means after removing the
      rate means), σ_e the residual SD on (rates − 1)(seeds − 1) degrees of freedom;
    - with ``rates[arm]`` (the selected rates): ``failure_rate``, the share of those runs (one per
      rate and seed; a missing run counts) without an endpoint, and, for a triple,
      ``support_rate``, the share of seeds whose block is supported (``block_support`` at K* and
      ``w_max``).

    An arm without two complete rates has no spreads (None); one without ``rates`` no rates."""
    out = {}
    for arm, name in INPUT_RUN_ARMS.items():
        mine = [r for r in runs if r.arm == name and r.batch_size == batch and r.seed in seeds
                and r.planned_steps == 25]  # fmt: skip
        k25: dict[float, dict[int, float]] = {}
        for run in mine:
            row, _ = _endpoint(run)
            if row is not None:
                k25.setdefault(float(f"{run.lr:.6g}"), {})[run.seed] = row["K_benign"]
        complete = sorted(lr for lr, by_seed in k25.items() if set(by_seed) >= set(seeds))
        found: dict[str, Any] = {"runs": len(mine), "complete_rates": complete}
        if len(complete) >= 2 and len(seeds) >= 2:
            table = np.log([[k25[lr][s] for s in seeds] for lr in complete])
            residual = table - table.mean(axis=1, keepdims=True)
            effects = residual.mean(axis=0)
            noise = residual - effects
            found["sigma_u"] = float(np.std(effects, ddof=1))
            found["sigma_e"] = float(
                math.sqrt((noise**2).sum() / ((len(complete) - 1) * (len(seeds) - 1)))
            )
        if rates and rates.get(arm):
            chosen = [float(f"{lr:.6g}") for lr in rates[arm]]
            at = [r for r in mine if float(f"{r.lr:.6g}") in chosen]
            ended = sum(1 for r in at if _endpoint(r)[0] is not None)
            expected = len(chosen) * len(seeds)
            found["failure_rate"] = (expected - ended) / expected
            found["failure_counts"] = {"runs": expected, "failures": expected - ended}
            if len(chosen) == 3:
                supported = sum(
                    block_support([r for r in at if r.seed == s], kstar, w_max, rates=chosen)[
                        "supported"
                    ]
                    for s in seeds
                )
                found["support_rate"] = supported / len(seeds)
                found["support_counts"] = {"blocks": len(seeds), "supported": supported}
        out[arm] = found
    return out


def inputs_record(
    measured: Mapping[str, Mapping[str, Any]], *, batch: int, where: str
) -> dict[str, Any]:
    """The inputs record (``read_inputs``' format, status ``measured``) from ``measure_inputs``
    with every field of every arm measured; ``where`` names the runs in each source."""
    arms: dict[str, Any] = {}
    for arm in ARMS:
        found = measured[arm]
        lacking = [name for name in INPUT_FIELDS if found.get(name) is None]
        if lacking:
            raise ValueError(f"arm {arm}: {lacking} not measured ({where})")
        rates = ", ".join(f"{lr:g}" for lr in found["complete_rates"])
        spread = f"{where}, B = {batch}: ln K25 over the complete rates {rates}"
        arms[arm] = {
            "sigma_u": {"value": found["sigma_u"], "source": f"{spread}: SD of the seed effects"},
            "sigma_e": {"value": found["sigma_e"], "source": f"{spread}: residual SD"},
            "support_rate": {"value": found["support_rate"], "counts": found["support_counts"],
                             "source": f"{where}, B = {batch}: seeds whose block is supported "
                                       "at the selected rates"},
            "failure_rate": {"value": found["failure_rate"], "counts": found["failure_counts"],
                             "source": f"{where}, B = {batch}: runs at the selected rates "
                                       "without an endpoint"},
        }  # fmt: skip
    return {"batch": batch, "status": "measured",
            "description": f"Study 2's B = {batch} simulation inputs, measured on K and support "
                           f"only ({where}).", "arms": arms}  # fmt: skip


def inputs_scenario(scenario: Scenario, inputs: Inputs) -> Scenario:
    """A primary scenario with the inputs' per-arm K spreads, and their failure rates where the
    scenario has none of its own (a stress scenario keeps its failure rate)."""
    changes: dict[str, Any] = {"sigma_u_arms": inputs.sigma_u, "sigma_e_arms": inputs.sigma_e}
    if scenario.failure == 0 and not scenario.failure_correlated:
        changes["failure_arms"] = inputs.failure_rate
    return replace(scenario, **changes)


# Amendment 4: the mechanism cohort (section 3)
#
# Three arms run per block as in the primary model (``simulate_blocks``), with the inputs' per-arm
# K spreads and failure rates. The components' per-block effects e₁ (C1 = P − F) and e₂ (C2 =
# A − P) are drawn on the logit scale from standard normal driving variables with correlation ρ
# (a Gaussian copula), transformed by the shape: normal σ·w; skewed σ·(G − 2)/√2, G the Gamma(2)
# quantile of Φ(w); heavy σ·S·w, S the block scale shared by both components; rare_centered
# M·(B − π); rare_only m·B, B = 1[w > Φ⁻¹(1 − π)]. The arms carry them as a chain: d_F = 0,
# d_P = e₁, d_A = e₁ + e₂ (rare_only: shifted up per block so that no arm is lowered,
# d_F = −min(0, e₁, e₁ + e₂), as amendment 3 put a negative rare difference in F). The per-prompt
# curves are study1's, fitted without block effects (``fit_target``); P follows F.
#
# Truth: the supported interpolation estimand of each component, the mean over the jointly
# supported three-arm population of the interpolated contrast at K* (the primary's interpolation
# estimand, built the same way). ``EstimandModel`` computes it from the population. Label noise
# has mean 0, so it is the mean of the interpolated expected harm. Prompts, stream effects and
# block effects are integrated by quadrature, and the endpoint geometry (selection misses,
# speeds, position noise) over a fixed sample of ``ESTIMAND_GEOMETRIES`` per arm, with every
# failure pattern weighted by its probability (given β_b for P under support-dependent
# attrition). The offsets c_P, then c_A, are solved so that it equals the declared (C1, C2) in
# every attrition mode; c_F makes F's population rate at K* the baseline.

COMPONENTS = ("C1", "C2")  # C1 = P − F, C2 = A − P
COMPONENT_ARMS = ((2, 1), (0, 2))  # (minuend, subtrahend) arm indices of C1 and C2
COMPONENT_MEANS = (
    (0.0, 0.0),
    (0.025, 0.025),
    (0.05, 0.05),
    (0.05, 0.0),
    (0.0, 0.05),
    (0.05, -0.05),
    (-0.05, 0.05),
    (-0.05, -0.05),
    (-0.025, 0.025),
)
SD_RATIOS = {"1:1": (1.0, 1.0), "2:1": (1.0, 0.5), "1:2": (0.5, 1.0)}  # (C1, C2) / the larger
COMPONENT_RHOS = (-0.5, 0.0, 0.5)  # correlation of the components' per-block driving variables
ATTRITIONS = ("none", "independent", "support_dependent")
MECHANISM_SHAPES = ("normal", "skewed", "heavy", "rare_centered", "rare_only", "tied")
REQUIRED_SHAPES = ("normal", "skewed")
REQUIRED_SDS = (0.05, 0.10)
STRESS_SDS = (0.05, 0.10, 0.15)
TIED_BASELINES = (1e-4, 2e-5)  # most blocks have h = 0 in every arm
MECHANISM_BASELINE = FAMILIES["study1"].baseline
MECHANISM_W_MAX = DECIDED_W_MAX
MECHANISM_REPS = 10_000
MECHANISM_STRESS_REPS = 2_000
# Bootstrap resamples per replicate and interval (the analysis fixes its own, 10 000): chosen
# for run time, as amendment 3's simulation chose 2000 (``BOOTSTRAP_RESAMPLES``).
MECHANISM_RESAMPLES = BOOTSTRAP_RESAMPLES
MECHANISM_LEVELS = tuple(round(0.95 + 0.005 * i, 3) for i in range(9))  # 0.95 … 0.99
MECHANISM_COVERAGE = 0.95
MECHANISM_STREAMS = {"calibration": "mechanism-calibration", "check": "mechanism-check"}
ESTIMAND_GEOMETRIES = 8192
COPULA_NODES = 48
G_STEP = 0.004  # the grid of the arms' expected-harm functions (logit units)
L_STEP = 0.001
PATTERNS = tuple(itertools.product((False, True), repeat=3))  # failed rates of an arm's endpoints

if MECHANISM_CAP != CAP:  # pragma: no cover - the analysis and the simulation share the cap
    raise ImportError("the mechanism cohort's cap must be the primary's")


def coverage_floor(reps: int, target: float = MECHANISM_COVERAGE) -> float:
    """The coverage requirement within MC error: target − 2·√(target·(1 − target)/R)."""
    return target - 2 * math.sqrt(target * (1 - target) / reps)


def stream_seed(base_seed: int, *keys: object) -> int:
    """An integer seed for the analysis's bootstrap functions, which draw from
    ``numpy.random.default_rng(seed)``: the full SHA-256 digest of the key list as an integer."""
    blob = json.dumps([base_seed, *keys]).encode()
    return int.from_bytes(hashlib.sha256(blob).digest(), "little")


@dataclass(frozen=True)
class Mechanism:
    """A mechanism scenario: the declared supported interpolation estimands (C1, C2), the shape,
    the larger component SD ``sd`` with the SD ratio (C1 : C2), the correlation ρ of the
    components' driving variables, the attrition, F's population rate at K* (``baseline``) and
    the replicates."""

    c1: float = 0.0
    c2: float = 0.0
    shape: str = "normal"
    sd: float = 0.05
    ratio: str = "1:1"
    rho: float = 0.0
    attrition: str = "none"
    baseline: float = MECHANISM_BASELINE
    reps: int = MECHANISM_REPS
    group: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        checks = {
            "shape": self.shape in MECHANISM_SHAPES,
            "ratio": self.ratio in SD_RATIOS,
            "rho": -1 < self.rho < 1,
            "attrition": self.attrition in ATTRITIONS,
            "baseline": 0 < self.baseline < 1,
            "sd": self.sd >= 0,
            "tied": self.shape != "tied" or (self.c1, self.c2, self.sd) == (0.0, 0.0, 0.0),
        }
        bad = [name for name, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"Invalid mechanism fields {bad}: {self}")

    @property
    def name(self) -> str:
        sd = "sdna" if self.shape == "rare_only" else f"sd{round(100 * self.sd, 6):g}"
        parts = ["mech", self.shape, sd, "r" + self.ratio.replace(":", "-"), f"rho{self.rho:+g}",
                 f"c1{_pp(self.c1)}", f"c2{_pp(self.c2)}", self.attrition]  # fmt: skip
        if self.baseline != MECHANISM_BASELINE:
            parts.append(f"h{self.baseline:g}")
        return "_".join(parts)

    @property
    def truth(self) -> tuple[float, float]:
        return self.c1, self.c2

    @property
    def targets(self) -> tuple[float, float]:
        """The components' target SDs (fractions); none for rare_only and tied."""
        if self.shape in ("rare_only", "tied"):
            return 0.0, 0.0
        r1, r2 = SD_RATIOS[self.ratio]
        return self.sd * r1, self.sd * r2

    @property
    def key(self) -> Mechanism:
        """The (0, 0) scenario without attrition whose component SDs set the spreads."""
        return replace(self, c1=0.0, c2=0.0, attrition="none", reps=0, group="")


def mechanism_base(inputs: Inputs) -> Scenario:
    """The primary model's settings the mechanism shares (study1's curves, the decided w_max,
    prompt and stream effects, label correlations, selection misses), with the inputs' per-arm
    K spreads."""
    return Scenario(w_max=MECHANISM_W_MAX, sigma_u_arms=inputs.sigma_u,
                    sigma_e_arms=inputs.sigma_e)  # fmt: skip


def failure_laws(attrition: str, inputs: Inputs) -> tuple[tuple[float, float], ...]:
    """Per arm (A, F, P), the failure logit's (intercept, slope) in β_b/σ_s: none; independent,
    each arm at its inputs' rate; support-dependent, A and F independent at theirs and P's
    failures more frequent in high-harm blocks (``failure_correlated``'s law) at its rate."""
    if attrition == "none":
        return ((-math.inf, 0.0),) * len(ARMS)
    laws = [(_logistic_intercept(rate, 0.0), 0.0) for rate in inputs.failure_rate]
    if attrition == "support_dependent":
        laws[2] = (_logistic_intercept(inputs.failure_rate[2], FAILURE_SLOPE), FAILURE_SLOPE)
    return tuple(laws)


def _skewed_effect(w: np.ndarray) -> np.ndarray:
    """(G − 2)/√2, G the Gamma(2) quantile of Φ(w), accurate in both tails."""
    w = np.asarray(w, dtype=float)
    upper = gammainccinv(GAMMA_SHAPE, ndtr(-np.abs(w)))
    lower = gammaincinv(GAMMA_SHAPE, ndtr(np.minimum(w, 0.0)))
    return (np.where(w > 0, upper, lower) - GAMMA_SHAPE) / math.sqrt(GAMMA_SHAPE)


def component_effects(
    shape: str, spreads: Sequence[float], w1: np.ndarray, w2: np.ndarray, tail: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """e₁ and e₂ (logit) from the driving variables w₁, w₂ and the heavy shape's block scale
    draw ``tail``; ``spreads`` are σ (normal, skewed, heavy), M (rare_centered) or signed m
    (rare_only)."""
    s1, s2 = spreads
    if shape == "tied":
        return np.zeros_like(w1), np.zeros_like(w2)
    if shape == "normal":
        return s1 * w1, s2 * w2
    if shape == "skewed":
        return s1 * _skewed_effect(w1), s2 * _skewed_effect(w2)
    if shape == "heavy":
        scale = np.where(tail < HEAVY_SHARE, *HEAVY_SCALES)
        return s1 * scale * w1, s2 * scale * w2
    rare = (w1 > RARE_THRESHOLD).astype(float), (w2 > RARE_THRESHOLD).astype(float)
    if shape == "rare_centered":
        return s1 * (rare[0] - RARE), s2 * (rare[1] - RARE)
    return s1 * rare[0], s2 * rare[1]


def arm_effects(shape: str, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    """(…, 3) arm effects d_A, d_F, d_P with d_P − d_F = e₁ and d_A − d_P = e₂."""
    e1, e2 = np.asarray(e1, dtype=float), np.asarray(e2, dtype=float)
    base = -np.minimum(np.minimum(e1, e1 + e2), 0.0) if shape == "rare_only" else np.zeros_like(e1)
    return np.stack([base + e1 + e2, base, base + e1], axis=-1)


def _bivariate_upper(threshold: float, rho: float) -> float:
    """P(w₁ > t, w₂ > t) for standard normals with correlation ρ."""
    scale = math.sqrt(1 - rho**2)
    value, _ = quad(lambda x: norm.pdf(x) * norm.cdf((rho * x - threshold) / scale), threshold,
                    math.inf, epsabs=1e-15, epsrel=1e-13)  # fmt: skip
    return float(value)


def component_nodes(
    shape: str, spreads: Sequence[float], rho: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature of (e₁, e₂): nodes and weights (summing to 1). Continuous shapes: a
    ``COPULA_NODES``² Gauss–Hermite grid on the driving variables (heavy: for each block scale);
    rare shapes: the four (B₁, B₂) with their exact probabilities."""
    if shape == "tied":
        return np.zeros(1), np.zeros(1), np.ones(1)
    if shape in ("rare_centered", "rare_only"):
        both = _bivariate_upper(RARE_THRESHOLD, rho)
        w1 = np.array([np.inf, np.inf, -np.inf, -np.inf])
        w2 = np.array([np.inf, -np.inf, np.inf, -np.inf])
        weights = np.array([both, RARE - both, RARE - both, 1 - 2 * RARE + both])
        e1, e2 = component_effects(shape, spreads, w1, w2, np.zeros(4))
        return e1, e2, weights
    z, w = np.polynomial.hermite_e.hermegauss(COPULA_NODES)
    w = w / w.sum()
    w1 = np.repeat(z, z.size)
    w2 = rho * w1 + math.sqrt(1 - rho**2) * np.tile(z, z.size)
    weights = np.outer(w, w).ravel()
    if shape == "heavy":
        high = component_effects(shape, spreads, w1, w2, np.zeros(w1.size))
        low = component_effects(shape, spreads, w1, w2, np.ones(w1.size))
        return (np.concatenate([high[0], low[0]]), np.concatenate([high[1], low[1]]),
                np.concatenate([HEAVY_SHARE * weights, (1 - HEAVY_SHARE) * weights]))  # fmt: skip
    e1, e2 = component_effects(shape, spreads, w1, w2, np.ones(w1.size))
    return e1, e2, weights


@functools.cache
def _geometry(
    arm: int, sigma_u: float, sigma_e: float, tau: float, w_max: float, base_seed: int
) -> tuple[tuple[np.ndarray, np.ndarray, float], ...]:
    """A fixed sample of one arm's endpoint geometry (a selection miss per rate, the speed, the
    position noise) and, per failure pattern (``PATTERNS``), the supported geometries' endpoint
    positions x (k, 3), their interpolation weights at K* (k, 3) and the supported share. The
    weights come from ``interpolate_at_zero`` itself, applied to unit harms."""
    rng = generator(base_seed, "estimand", "geometry", arm)
    n = ESTIMAND_GEOMETRIES
    x = np.array(NOMINAL) + rng.uniform(-tau, tau, (n, 3))
    x = x + sigma_u * rng.standard_normal((n, 1)) + sigma_e * rng.standard_normal((n, 3))
    out = []
    for pattern in PATTERNS:
        failed = np.broadcast_to(np.array(pattern), (n, 3))
        weights = np.stack(
            [interpolate_at_zero(x, np.tile(np.eye(3)[r], (n, 1)), failed, w_max)[0]
             for r in range(3)],
            axis=1,
        )  # fmt: skip
        supported = np.isfinite(weights[:, 0])
        out.append((x[supported], weights[supported], float(supported.mean())))
    return tuple(out)


def _pattern_probabilities(law: tuple[float, float], z: np.ndarray) -> np.ndarray:
    """(nodes, 8) probabilities of the failure patterns given β_b/σ_s = ``z``."""
    intercept, slope = law
    p = expit(intercept + slope * np.asarray(z, dtype=float))[:, None]
    failures = np.array([sum(pattern) for pattern in PATTERNS])
    return p**failures * (1 - p) ** (3 - failures)


def _expected_harm_table(
    values: np.ndarray, weights: np.ndarray, sigma: float, low: float, high: float
) -> tuple[np.ndarray, np.ndarray]:
    """G(z) = E[σ(z + d + v)] on a grid of z in [low, high], d ~ (``values``, ``weights``) and
    v ~ N(0, σ²): L(y) = E_v σ(y + v) by Gauss–Hermite on a fine grid, then G = Σ w·L(z + d)."""
    z = np.arange(low, high + G_STEP / 2, G_STEP)
    y = np.arange(low + values.min() - 1, high + values.max() + 1 + L_STEP, L_STEP)
    nodes, h = _hermite()
    table = np.empty(y.size)
    for start in range(0, y.size, 4096):
        table[start : start + 4096] = expit(y[start : start + 4096, None] + sigma * nodes) @ h
    g = np.zeros(z.size)
    for start in range(0, values.size, 256):
        shifted = z[:, None] + values[None, start : start + 256]
        g += np.interp(shifted, y, table) @ weights[start : start + 256]
    return z, g


class EstimandModel:
    """The supported interpolation estimand from the population (see the section comment):
    ``mean(arm, offset)`` is E[ĥ_arm | all three arms supported] at the arm's logit offset,
    ``contrasts(offsets)`` the components."""

    def __init__(
        self,
        base: Scenario,
        coefficients: Sequence[tuple[float, float, float]],
        effects: Sequence[tuple[np.ndarray, np.ndarray]],
        failure: Sequence[tuple[float, float]],
        centre: float,
        base_seed: int,
    ) -> None:
        self.failure = tuple(failure)
        # β_b is integrated apart from the prompts only where a failure law depends on it.
        self.explicit = any(slope != 0 and math.isfinite(i) for i, slope in self.failure)
        nodes, weights = _hermite()
        self.z = nodes if self.explicit else np.zeros(1)
        self.zw = weights if self.explicit else np.ones(1)
        sigma = base.sigma_alpha if self.explicit else _sigma_v(base)
        self.shift = base.sigma_s * self.z
        sigma_u, sigma_e = (np.broadcast_to(np.ravel(v), len(ARMS))
                            for v in base.k_spreads([0, 1, 2]))  # fmt: skip
        span = 16 + float(np.abs(self.shift).max())
        self.arms: list[dict[str, Any]] = []
        for a in range(len(ARMS)):
            geometry = _geometry(a, float(sigma_u[a]), float(sigma_e[a]), base.tau, base.w_max,
                                 base_seed)  # fmt: skip
            probabilities = _pattern_probabilities(self.failure[a], self.z)  # (nodes, 8)
            used = np.flatnonzero(probabilities.max(axis=0) > 0)
            entries = []
            for p in used:
                x, w, _ = geometry[p]
                delta = _shape(x, *coefficients[a])
                keep = w != 0
                entries.append((int(p), w[keep] / ESTIMAND_GEOMETRIES, delta[keep]))
            support = probabilities[:, used] @ np.array([geometry[p][2] for p in used])
            values, probs = effects[a]
            grid, table = _expected_harm_table(values, probs, sigma, centre - span, centre + span)
            self.arms.append({"probabilities": probabilities, "entries": entries,
                              "support": support, "grid": grid, "table": table})  # fmt: skip
        self.p_joint = float(self.zw @ np.prod([m["support"] for m in self.arms], axis=0))

    def mean(self, arm: int, offset: float) -> float:
        """E[ĥ_arm | jointly supported] at logit offset ``offset``."""
        model = self.arms[arm]
        total = np.zeros(self.z.size)  # E[ĥ_arm · 1(arm supported) | β_b]
        for p, w, delta in model["entries"]:
            z = offset + delta[None, :] + self.shift[:, None]
            harm = np.interp(z, model["grid"], model["table"]) @ w
            total += model["probabilities"][:, p] * harm
        others = np.prod([m["support"] for k, m in enumerate(self.arms) if k != arm], axis=0)
        return float(self.zw @ (total * others)) / self.p_joint

    def means(self, offsets: Sequence[float]) -> tuple[float, ...]:
        return tuple(self.mean(a, offsets[a]) for a in range(len(ARMS)))

    def contrasts(self, offsets: Sequence[float]) -> tuple[float, ...]:
        h = self.means(offsets)
        return tuple(h[i] - h[j] for i, j in COMPONENT_ARMS)

    def support(self) -> dict[str, float]:
        """P(arm supported) per arm and P(all three), marginal over β_b."""
        per_arm = {arm: float(self.zw @ m["support"]) for arm, m in zip(ARMS, self.arms,
                                                                       strict=True)}  # fmt: skip
        return {**per_arm, "joint": self.p_joint}


@dataclass(frozen=True)
class MechanismParameters:
    """What a mechanism scenario's draws need: offsets (c_A, c_F, c_P), per-prompt curve
    coefficients, the component spreads (σ, M or signed m), each arm's failure law, and the
    population quantities (the estimand per component, per-arm means and support)."""

    offsets: tuple[float, float, float]
    coefficients: tuple[tuple[float, float, float], ...]
    spreads: tuple[float, float]
    failure: tuple[tuple[float, float], ...]
    estimand: dict[str, Any]


def _rare_only_shifts(m: Mechanism, base: Scenario, offset: float) -> tuple[float, float]:
    """Signed m per component: the rare shift that raises a population rate at the baseline by
    |C| in one arm (amendment 3's rare_only, per component); 0 for C = 0."""
    rate = population_rate(base, offset, NO_EFFECT)

    def gap(size: float) -> float:
        return population_rate(base, offset, effect_distribution("rare_only", size)) - rate

    shifts = []
    for c in m.truth:
        if c == 0:
            shifts.append(0.0)
            continue
        if gap(60.0) <= abs(c):
            raise ValueError(f"rare_only cannot reach |C| = {abs(c)} in {m.name}")
        size = float(brentq(lambda s, c=c: gap(s) - abs(c), 0.0, 60.0, xtol=1e-12))
        shifts.append(math.copysign(size, c))
    return shifts[0], shifts[1]


def mechanism_parameters(
    m: Mechanism,
    inputs: Inputs,
    spreads: Sequence[float] = (0.0, 0.0),
    *,
    base_seed: int = 0,
    solve: bool = True,
) -> MechanismParameters:
    """A mechanism scenario's parameters: the curves; c_F, making F's population rate at K* the
    baseline; the spreads (rare_only: from the declared means); the failure laws; and, with
    ``solve``, c_P then c_A such that the supported interpolation estimand equals the declared
    (C1, C2). Without ``solve`` (the spread calibration): c_A = c_P = c_F."""
    base = mechanism_base(inputs)
    target = Family(**STUDY1, baseline=m.baseline)
    sigma_v = _sigma_v(base)
    fits = [fit_target(target, a, sigma_v, None) for a in (0, 1)]
    coefficients = (fits[0][0], fits[1][0], fits[1][0])  # P follows F
    c_f = _solve_offset(lambda c: population_rate(base, c, NO_EFFECT), m.baseline)
    if m.shape == "rare_only":
        spreads = _rare_only_shifts(m, base, c_f)
    spreads = (float(spreads[0]), float(spreads[1]))
    failure = failure_laws(m.attrition, inputs)
    if not solve:
        return MechanismParameters((c_f, c_f, c_f), coefficients, spreads, failure, {})
    e1, e2, weights = component_nodes(m.shape, spreads, m.rho)
    d = arm_effects(m.shape, e1, e2)
    effects = [(d[:, a], weights) for a in range(len(ARMS))]
    model = EstimandModel(base, coefficients, effects, failure, c_f, base_seed)
    h_f = model.mean(1, c_f)
    c_p = float(brentq(lambda c: model.mean(2, c) - h_f - m.c1, c_f - 15, c_f + 15, xtol=1e-13))
    h_p = model.mean(2, c_p)
    c_a = float(brentq(lambda c: model.mean(0, c) - h_p - m.c2, c_f - 15, c_f + 15, xtol=1e-13))
    offsets = (c_a, c_f, c_p)
    estimand = {
        **dict(zip(COMPONENTS, model.contrasts(offsets), strict=True)),
        "arm_means": dict(zip(ARMS, model.means(offsets), strict=True)),
        "support": model.support(),
        "beta_explicit": model.explicit,
    }
    return MechanismParameters(offsets, coefficients, spreads, failure, estimand)


def simulate_mechanism_blocks(
    m: Mechanism,
    params: MechanismParameters,
    base: Scenario,
    alpha: np.ndarray,
    misses: np.ndarray,
    rng: np.random.Generator,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """ĥ(K*) and the support cause of ``n`` new three-arm blocks, each (n, 3): the primary's
    block model (``simulate_blocks`` with ρ_u = ρ_uh = 0) carrying the components' effects."""
    stream = rng.standard_normal(n)  # β_b/σ_s
    speed = rng.standard_normal((n, len(ARMS)))  # u/σ_u
    latent = rng.standard_normal((n, 2))
    tail = rng.random(n)  # the heavy shape's block scale
    w1 = latent[:, 0]
    w2 = m.rho * w1 + math.sqrt(1 - m.rho**2) * latent[:, 1]
    effect = arm_effects(m.shape, *component_effects(m.shape, params.spreads, w1, w2, tail))
    arms = [0, 1, 2]
    sigma_u, sigma_e = base.k_spreads(arms)
    nominal = np.array(POSITIONS[base.positions]) + misses
    x = nominal + sigma_u * speed[..., None] + sigma_e * rng.standard_normal((n, len(ARMS), 3))
    law = np.array(params.failure)
    probability = expit(law[:, 0] + law[:, 1] * stream[:, None])[..., None]
    failed = rng.random((n, len(ARMS), 3)) < probability
    eta = (
        np.array(params.offsets)[:, None]
        + curve_offset(params.coefficients, x, arms)
        + base.sigma_s * stream[:, None, None]
        + effect[..., None]
    )
    harm = endpoint_harm(eta, alpha, base.rho_b, base.rho_a, rng)
    return interpolate_at_zero(x, harm, failed, base.w_max)


def cohort_contrasts(estimates: np.ndarray, causes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """C1 and C2 on the mechanism cohort: the first ``MECHANISM_TARGET`` jointly supported
    blocks in started order (at the cap, all of them)."""
    joint = np.flatnonzero((causes == SUPPORTED).all(axis=1))[:MECHANISM_TARGET]
    e = estimates[joint]
    return e[:, 2] - e[:, 1], e[:, 0] - e[:, 2]


def mechanism_intervals(
    c1: np.ndarray,
    c2: np.ndarray,
    levels: Sequence[float],
    resamples: int,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """The analysis's intervals on one cohort: ``max_t_interval`` (seed ``seeds[0]``) and its
    critical value at every level of ``levels`` from the same draws (``_upper_quantile`` of the
    per-draw maxima, as ``max_t_interval`` takes it at its own level); ``symmetric_studentized``
    at ``COMPARISON_LEVEL`` per component (seeds ``seeds[1]`` and ``seeds[2]``: separate streams,
    as in the analysis)."""
    joint = max_t_interval(c1, c2, levels[0], resamples, seeds[0], keep=True)
    if "t_star" in joint:
        ordered = np.sort(np.maximum(*joint["t_star"]))
        q = [_upper_quantile(ordered, level) for level in levels]
    else:  # fewer than 2 blocks: no interval
        q = [math.inf] * len(levels)
    bonferroni = [symmetric_studentized(c, COMPARISON_LEVEL, resamples, s)
                  for c, s in zip((c1, c2), seeds[1:], strict=True)]  # fmt: skip
    contrasts = joint["contrasts"]
    return {
        "n": joint["n"],
        "mean": [math.nan if c["mean"] is None else c["mean"] for c in contrasts],
        "sd": [math.nan if c["sd"] is None else c["sd"] for c in contrasts],
        "se": [math.nan if c["se"] is None else c["se"] for c in contrasts],
        "zero_se": [bool(c["zero_se"]) for c in contrasts],
        "readable": [c["zero_se"] is not None for c in contrasts],
        "q": q,
        "infinite": joint["infinite"],
        "draws": joint["draws"],
        "bonferroni_q": [math.inf if b["q"] is None else b["q"] for b in bonferroni],
        "bonferroni_infinite": [b["infinite"] for b in bonferroni],
    }


ROW_FIELDS = ("n", "started", "mean", "sd", "se", "zero_se", "readable", "q", "infinite", "draws",
              "bonferroni_q", "bonferroni_infinite", "supported", "technical", "sum",
              "correlation")  # fmt: skip


def mechanism_replicate(
    m: Mechanism,
    params: MechanismParameters,
    base: Scenario,
    rep: int,
    *,
    base_seed: int,
    stream: str,
    levels: Sequence[float],
    resamples: int,
) -> dict[str, Any]:
    """One confirmation: blocks start until ``MECHANISM_TARGET`` are jointly supported in all
    three arms or ``MECHANISM_CAP`` have started; then the cohort's intervals
    (``mechanism_intervals``; streams (base_seed, stream, name, rep, "max_t") and (…, rep,
    "bonferroni", component)) and its summaries."""
    rng = generator(base_seed, stream, m.name, rep)
    alpha, misses = draw_constants(base, rng)
    estimates = np.full((CAP, len(ARMS)), np.nan)
    causes = np.full((CAP, len(ARMS)), NOT_STARTED, dtype=np.int8)
    started = joint = 0
    while started < CAP and joint < MECHANISM_TARGET:
        n = min(CHUNK_BLOCKS, CAP - started)
        estimate, cause = simulate_mechanism_blocks(m, params, base, alpha, misses, rng, n)
        estimates[started : started + n], causes[started : started + n] = estimate, cause
        joint += int(np.count_nonzero((cause == SUPPORTED).all(axis=1)))
        started += n
    # The window: blocks in order up to the cohort's last one (blocks are drawn in chunks).
    joint_blocks = np.flatnonzero((causes[:started] == SUPPORTED).all(axis=1))
    if joint_blocks.size >= MECHANISM_TARGET:
        started = int(joint_blocks[MECHANISM_TARGET - 1]) + 1
    c1, c2 = cohort_contrasts(estimates[:started], causes[:started])
    seeds = [stream_seed(base_seed, stream, m.name, rep, "max_t")]
    seeds += [stream_seed(base_seed, stream, m.name, rep, "bonferroni", c) for c in COMPONENTS]
    row = mechanism_intervals(c1, c2, levels, resamples, seeds)
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = float(np.corrcoef(c1, c2)[0, 1]) if c1.size > 2 else math.nan
    row |= {
        "started": started,
        "supported": (causes[:started] == SUPPORTED).sum(axis=0).tolist(),
        "technical": (causes[:started] == TECHNICAL).sum(axis=0).tolist(),
        "sum": [float(c1.sum()), float(c2.sum())],
        "correlation": correlation,
    }
    return row


def _stack_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    return {key: np.array([r[key] for r in rows], dtype=float) for key in ROW_FIELDS}


# Spread calibration of the components


@dataclass(frozen=True)
class ComponentCalibration:
    key: str
    targets: tuple[float, float]
    spreads: tuple[float, float]
    attained: tuple[float, float]
    noise: tuple[float, float]  # at spreads 0: response and K noise alone
    evaluations: int
    cpu_seconds: float


def component_sds(
    key: Mechanism,
    inputs: Inputs,
    spreads: Sequence[float],
    *,
    seed: int | np.random.SeedSequence,
    replicates: int,
    blocks: int,
) -> tuple[float, float]:
    """√(mean within-replicate variance of C_j over jointly supported blocks), per component, at
    the given spreads (equal offsets, no attrition)."""
    params = mechanism_parameters(key, inputs, spreads, solve=False)
    base = mechanism_base(inputs)
    rng = np.random.default_rng(seed)
    variances: list[list[float]] = [[], []]
    for _ in range(replicates):
        alpha, misses = draw_constants(base, rng)
        estimate, cause = simulate_mechanism_blocks(key, params, base, alpha, misses, rng, blocks)
        joint = (cause == SUPPORTED).all(axis=1)
        if joint.sum() < 2:
            continue
        e = estimate[joint]
        for j, (i, k) in enumerate(COMPONENT_ARMS):
            variances[j].append(float((e[:, i] - e[:, k]).var(ddof=1)))
    if not variances[0]:
        raise ValueError(f"No replicate of {key.name} has two jointly supported blocks")
    return float(np.sqrt(np.mean(variances[0]))), float(np.sqrt(np.mean(variances[1])))


def calibrate_components(
    key: Mechanism, inputs: Inputs, *, base_seed: int, replicates: int = 100, blocks: int = 200
) -> ComponentCalibration:
    """The component spreads giving the key's target SDs of C1 and C2 at (0, 0), by alternating
    root finding on common random numbers (each spread barely moves the other component's SD).
    A target below the noise alone gets spread 0, and the attained SD is reported."""
    cpu = time.process_time()
    seed = seed_sequence(base_seed, "mechanism-spread", key.name)
    cache: dict[tuple[float, float], tuple[float, float]] = {}

    def attained(spreads: tuple[float, float]) -> tuple[float, float]:
        if spreads not in cache:
            cache[spreads] = component_sds(key, inputs, spreads, seed=seed,
                                           replicates=replicates, blocks=blocks)  # fmt: skip
        return cache[spreads]

    targets = key.targets
    noise = attained((0.0, 0.0))
    spreads = [0.0, 0.0]
    for _ in range(10 if key.shape not in ("rare_only", "tied") else 0):
        before = tuple(spreads)
        for j in (0, 1):

            def gap(s: float, j: int = j) -> float:
                trial = list(spreads)
                trial[j] = s
                return attained((trial[0], trial[1]))[j] - targets[j]

            if gap(0.0) >= 0:
                spreads[j] = 0.0
                continue
            high = 0.5
            while gap(high) < 0:
                high *= 2
                if high > 64:
                    raise ValueError(f"The SD of {key.name}'s {COMPONENTS[j]} cannot reach "
                                     f"{targets[j]}")  # fmt: skip
            spreads[j] = float(brentq(gap, 0.0, high, xtol=1e-4, rtol=1e-4))
        if max(abs(a - b) for a, b in zip(spreads, before, strict=True)) < 2e-4:
            break
    final = (spreads[0], spreads[1])
    return ComponentCalibration(key.name, targets, final, attained(final), noise, len(cache),
                                time.process_time() - cpu)  # fmt: skip


# Grids


def mechanism_grid(
    reps: int = MECHANISM_REPS, stress_reps: int = MECHANISM_STRESS_REPS, stress: bool = True
) -> list[Mechanism]:
    """The required grid (normal and skewed at SD 5 and 10 pp; group ``required``) and, with
    ``stress``, the stress grid (heavy and rare_centered at SD 5, 10 and 15, normal and skewed at
    15, rare_only, and the tied case; group ``stress``), each over every (C1, C2) × SD ratio × ρ
    × attrition (rare_only: ratio 1:1; tied: (0, 0) and ρ 0 at each tied baseline)."""
    factors = [(c, ratio, rho, attrition) for c in COMPONENT_MEANS for ratio in SD_RATIOS
               for rho in COMPONENT_RHOS for attrition in ATTRITIONS]  # fmt: skip
    grid = [
        Mechanism(*c, shape=shape, sd=sd, ratio=ratio, rho=rho, attrition=attrition, reps=reps,
                  group="required")
        for shape in REQUIRED_SHAPES for sd in REQUIRED_SDS
        for c, ratio, rho, attrition in factors
    ]  # fmt: skip
    if not stress:
        return grid
    cells = [(shape, sd) for shape in ("heavy", "rare_centered") for sd in STRESS_SDS]
    cells += [(shape, sd) for shape in REQUIRED_SHAPES for sd in STRESS_SDS
              if sd not in REQUIRED_SDS]  # fmt: skip
    grid += [
        Mechanism(*c, shape=shape, sd=sd, ratio=ratio, rho=rho, attrition=attrition,
                  reps=stress_reps, group="stress")
        for shape, sd in cells for c, ratio, rho, attrition in factors
    ]  # fmt: skip
    grid += [
        Mechanism(*c, shape="rare_only", sd=0.0, rho=rho, attrition=attrition, reps=stress_reps,
                  group="stress")
        for c in COMPONENT_MEANS for rho in COMPONENT_RHOS for attrition in ATTRITIONS
    ]  # fmt: skip
    grid += [
        Mechanism(shape="tied", sd=0.0, baseline=b, attrition=attrition, reps=stress_reps,
                  group="stress")
        for b in TIED_BASELINES for attrition in ATTRITIONS
    ]  # fmt: skip
    return grid


# Summaries and verdicts


def _rates(flags: np.ndarray) -> dict[str, float]:
    p, se = _rate(flags)
    return {"value": p, "se": se}


def mechanism_readings(rows: Mapping[str, np.ndarray], q: np.ndarray) -> dict[str, np.ndarray]:
    """Per replicate and component, the analysis's readings (``mechanism_analysis``) at the
    max-t critical values ``q`` (R,): nothing below the minimum cohort or for a contrast without
    an observed SE or with a zero one; when the other contrast's SE is zero, the reading uses
    the Bonferroni interval; otherwise the max-t interval (none when q = ∞: inconclusive).
    Returns boolean (R, 2) arrays ``positive``, ``negative`` and ``bounded``
    (``component_readings``, the meaningful effect as margin)."""
    n, mean, se = rows["n"], rows["mean"], rows["se"]
    zero, readable = rows["zero_se"].astype(bool), rows["readable"].astype(bool)
    minimum = (n >= MECHANISM_MINIMUM)[:, None]
    use_bonferroni = zero[:, ::-1] & readable & ~zero
    critical = np.where(use_bonferroni, rows["bonferroni_q"], q[:, None])
    has = minimum & readable & ~zero & np.isfinite(critical)
    with np.errstate(invalid="ignore"):
        low, high = mean - critical * se, mean + critical * se
    margin = MECHANISM_EFFECTS[0]
    return {
        "positive": has & (low > 0),
        "negative": has & (high < 0),
        "bounded": has & (low >= -margin) & (high <= margin),
    }


def _covered(rows: Mapping[str, np.ndarray], q: np.ndarray, truth: Sequence[float]) -> np.ndarray:
    """(R, 2): the symmetric interval mean ± q·SE contains the truth; a contrast without an
    interval (no observed SE, a zero one, or q = ∞) counts as covered (the whole line)."""
    mean, se = rows["mean"], rows["se"]
    has = rows["readable"].astype(bool) & ~rows["zero_se"].astype(bool) & np.isfinite(q)
    value = np.array(truth)
    with np.errstate(invalid="ignore"):
        inside = (mean - q * se <= value) & (value <= mean + q * se)
    return ~has | inside


def summarize_mechanism(
    m: Mechanism,
    params: MechanismParameters,
    rows: Mapping[str, np.ndarray],
    levels: Sequence[float],
    seconds: float = 0.0,
) -> dict[str, Any]:
    """A mechanism scenario's outputs from its replicates: per level, the simultaneous max-t
    coverage and, per component, its coverage and readings (direction power, false direction,
    bounded); the Bonferroni coverage; the minimum's reach; the infinite-q and zero-SE shares;
    the attained SDs and correlation; the estimand check (the pooled cohort mean against the
    declared value); and the support."""
    truth = m.truth
    by_level = []
    for i, level in enumerate(levels):
        q = rows["q"][:, i]
        covered = _covered(rows, q[:, None], truth)
        readings = mechanism_readings(rows, q)
        components = {}
        for j, name in enumerate(COMPONENTS):
            c = truth[j]
            either = readings["positive"][:, j] | readings["negative"][:, j]
            right = readings["positive" if c > 0 else "negative"][:, j] if c else None
            wrong = readings["negative" if c > 0 else "positive"][:, j] if c else either
            components[name] = {
                "coverage": _rates(covered[:, j]),
                "power": None if right is None else _rates(right),
                "false_direction": _rates(wrong),
                "bounded": _rates(readings["bounded"][:, j]),
            }
        by_level.append({
            "level": level,
            "simultaneous": _rates(covered.all(axis=1)),
            "infinite_q": _rates(~np.isfinite(q)),
            "components": components,
        })  # fmt: skip
    bonferroni = _covered(rows, rows["bonferroni_q"], truth)
    n, sd, started = rows["n"], rows["sd"], rows["started"]
    usable = n >= 2
    with np.errstate(invalid="ignore", divide="ignore"):
        per_rep = rows["sum"] / n[:, None]
    pooled = rows["sum"].sum(axis=0) / n.sum()
    pooled_se = per_rep[usable].std(axis=0, ddof=1) / math.sqrt(max(usable.sum(), 1))
    blocks = started.sum()
    return {
        "name": m.name,
        "settings": asdict(m),
        "reps": int(n.size),
        "seconds": seconds,
        "parameters": {
            "offsets": dict(zip(ARMS, params.offsets, strict=True)),
            "spreads": dict(zip(COMPONENTS, params.spreads, strict=True)),
            "failure": {arm: list(law) for arm, law in zip(ARMS, params.failure, strict=True)},
            "estimand": params.estimand,
        },
        "levels": by_level,
        "bonferroni": {
            "level": COMPARISON_LEVEL,
            **{name: _rates(bonferroni[:, j]) for j, name in enumerate(COMPONENTS)},
            "infinite_q": {
                name: _rates(~np.isfinite(rows["bonferroni_q"][:, j]))
                for j, name in enumerate(COMPONENTS)
            },
        },  # fmt: skip
        "minimum_met": _rates(n >= MECHANISM_MINIMUM),
        "cohort": {
            "mean_n": float(n.mean()),
            "min_n": int(n.min()),
            "mean_started": float(started.mean()),
            "ended_at_cap": _rates(n < MECHANISM_TARGET),
        },
        "zero_se": {
            name: _rates(rows["zero_se"][:, j].astype(bool)) for j, name in enumerate(COMPONENTS)
        },  # fmt: skip
        "infinite_draw_share": float((rows["infinite"] / rows["draws"].clip(min=1)).mean()),
        "attained_sd": {
            name: float(np.sqrt(np.nanmean(sd[usable, j] ** 2)))
            for j, name in enumerate(COMPONENTS)
        },  # fmt: skip
        "mean_correlation": float(np.nanmean(rows["correlation"]))
        if np.isfinite(rows["correlation"]).any()
        else None,
        "estimand_check": {
            name: {
                "declared": truth[j],
                "population": params.estimand.get(name),
                "cohort_mean": float(pooled[j]),
                "se": float(pooled_se[j]),
                "z": float((pooled[j] - truth[j]) / pooled_se[j]) if pooled_se[j] > 0 else None,
            }
            for j, name in enumerate(COMPONENTS)
        },
        "support": {
            "simulated_per_arm": dict(
                zip(ARMS, (rows["supported"].sum(axis=0) / blocks).tolist(), strict=True)
            ),
            "technical_per_arm": dict(
                zip(ARMS, (rows["technical"].sum(axis=0) / blocks).tolist(), strict=True)
            ),
            "population": params.estimand.get("support"),
        },  # fmt: skip
    }


def _cell(scenario: Mapping[str, Any]) -> tuple[str, float]:
    return scenario["settings"]["shape"], scenario["settings"]["sd"]


def mechanism_checks(
    scenarios: Sequence[Mapping[str, Any]], readings_at: float | None = None
) -> dict[str, Any]:
    """The calibration rule over the required cells (normal and skewed at SD 5 and 10 pp): per
    level, each cell's smallest simultaneous max-t coverage over its scenarios against
    ``coverage_floor`` at the scenario's replicates; ``level`` is the smallest level at which
    every required cell holds (None if none). Stress cells are reported the same way and do not
    bind. Beside it, per cell at ``readings_at`` (default: that level, else the first), the
    power, false direction and bounded rates and the minimum's reach (``reading_summary``)."""
    levels = [entry["level"] for entry in scenarios[0]["levels"]] if scenarios else []
    groups = {"required": [s for s in scenarios if s["settings"]["group"] == "required"],
              "stress": [s for s in scenarios if s["settings"]["group"] != "required"]}  # fmt: skip
    per_level = []
    for i, level in enumerate(levels):
        entry: dict[str, Any] = {"level": level}
        for group, members in groups.items():
            cells = []
            for cell in dict.fromkeys(_cell(s) for s in members):
                mine = [s for s in members if _cell(s) == cell]
                worst = min(mine, key=lambda s: s["levels"][i]["simultaneous"]["value"])
                value = worst["levels"][i]["simultaneous"]["value"]
                floor = coverage_floor(worst["reps"])
                cells.append({"shape": cell[0], "sd": cell[1], "min_coverage": value,
                              "mc_se": worst["levels"][i]["simultaneous"]["se"],
                              "scenario": worst["name"], "floor": floor, "meets": value >= floor,
                              "short": sum(s["levels"][i]["simultaneous"]["value"]
                                           < coverage_floor(s["reps"]) for s in mine),
                              "scenarios": len(mine)})  # fmt: skip
            entry[group] = {"cells": cells, "meets": all(c["meets"] for c in cells)}
        per_level.append(entry)
    passing = [e["level"] for e in per_level if e["required"]["cells"] and e["required"]["meets"]]
    chosen = min(passing, default=None)
    shown = readings_at if readings_at is not None else chosen if chosen is not None else (
        levels[0] if levels else None)  # fmt: skip
    return {
        "floor_rule": "0.95 − 2·√(0.95·0.05/R), R the scenario's replicates",
        "per_level": per_level,
        "level": chosen,
        "readings_at": shown,
        "readings": {group: reading_summary(members, levels.index(shown))
                     for group, members in groups.items()} if shown is not None else {},
    }  # fmt: skip


def reading_summary(scenarios: Sequence[Mapping[str, Any]], i: int) -> list[dict[str, Any]]:
    """Per cell (shape × SD) and component SD (the larger, 1, or half of it, 0.5), at level index
    ``i``: the smallest and mean direction power at |C| = 5 and 2.5 pp, the largest false
    direction rate and the smallest and mean bounded rate at C = 0, and the smallest share of
    replicates reaching the minimum cohort."""
    out = []
    for cell in dict.fromkeys(_cell(s) for s in scenarios):
        mine = [s for s in scenarios if _cell(s) == cell]
        for scale in (1.0, 0.5):
            picks = [(s, name, s["settings"]["c1" if name == "C1" else "c2"])
                     for s in mine for j, name in enumerate(COMPONENTS)
                     if SD_RATIOS[s["settings"]["ratio"]][j] == scale]  # fmt: skip
            if not picks:
                continue
            entry: dict[str, Any] = {"shape": cell[0], "sd": cell[1], "component_sd": scale}
            for effect in MECHANISM_EFFECTS:
                powers = [s["levels"][i]["components"][name]["power"]["value"]
                          for s, name, c in picks if math.isclose(abs(c), effect)]  # fmt: skip
                entry[f"power_{_pp(effect)[1:]}pp"] = (
                    {"min": min(powers), "mean": float(np.mean(powers)), "scenarios": len(powers)}
                    if powers else None
                )  # fmt: skip
            null = [s["levels"][i]["components"][name] for s, name, c in picks if c == 0]
            entry["max_false_direction_at_0"] = max(
                (r["false_direction"]["value"] for r in null), default=None
            )
            bounded = [r["bounded"]["value"] for r in null]
            entry["bounded_at_0"] = (
                {"min": min(bounded), "mean": float(np.mean(bounded))} if bounded else None
            )
            entry["min_minimum_met"] = min(s["minimum_met"]["value"] for s, _, _ in picks)
            out.append(entry)
    return out


# Running the mechanism grid


@dataclass(frozen=True)
class MechanismSettings:
    stream: str
    inputs: Inputs
    levels: tuple[float, ...] = MECHANISM_LEVELS
    resamples: int = MECHANISM_RESAMPLES
    base_seed: int = 0
    workers: int = 1
    chunk: int = 250  # replicates per worker task
    spread_replicates: int = 100
    spread_blocks: int = 200

    def __post_init__(self) -> None:
        if not self.levels or any(not 0.5 < v < 1 for v in self.levels):
            raise ValueError(f"Mechanism levels must lie in (0.5, 1): {self.levels}")
        if self.resamples < 2:
            raise ValueError("The bootstrap needs at least 2 resamples")


def _mechanism_calibration_task(task: tuple[Mechanism, MechanismSettings]) -> ComponentCalibration:
    key, settings = task
    return calibrate_components(key, settings.inputs, base_seed=settings.base_seed,
                                replicates=settings.spread_replicates,
                                blocks=settings.spread_blocks)  # fmt: skip


def _mechanism_setup_task(task: tuple[Mechanism, tuple[float, float], MechanismSettings]):
    m, spreads, settings = task
    return m.name, mechanism_parameters(m, settings.inputs, spreads, base_seed=settings.base_seed)


def _mechanism_replicate_task(task):
    m, params, start, stop, settings = task
    clock = time.process_time()
    base = mechanism_base(settings.inputs)
    rows = [mechanism_replicate(m, params, base, rep, base_seed=settings.base_seed,
                                stream=settings.stream, levels=settings.levels,
                                resamples=settings.resamples)
            for rep in range(start, stop)]  # fmt: skip
    return m.name, start, _stack_rows(rows), time.process_time() - clock


def run_mechanism(
    mechanisms: Sequence[Mechanism],
    settings: MechanismSettings,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Calibrate the component spreads, solve every scenario's parameters, simulate it and
    summarize it."""
    names = [m.name for m in mechanisms]
    if len(set(names)) != len(names):
        raise ValueError("Mechanism names must be unique: they name the random streams")
    keys = {m.key.name: m.key for m in mechanisms if m.shape not in ("rare_only", "tied")}
    calibrations: dict[str, ComponentCalibration] = {}
    with _executor(settings.workers) as imap:
        tasks = [(key, settings) for key in keys.values()]
        for done, result in enumerate(imap(_mechanism_calibration_task, tasks), start=1):
            calibrations[result.key] = result
            if progress is not None:
                progress("spread", done, len(tasks))
        setups = [(m, calibrations[m.key.name].spreads if m.key.name in calibrations
                   else (0.0, 0.0), settings) for m in mechanisms]  # fmt: skip
        params = {}
        for done, (name, found) in enumerate(imap(_mechanism_setup_task, setups), start=1):
            params[name] = found
            if progress is not None:
                progress("setup", done, len(setups))
        by_name = {m.name: m for m in mechanisms}
        tasks = [(m, params[m.name], start, min(start + settings.chunk, m.reps), settings)
                 for m in mechanisms for start in range(0, m.reps, settings.chunk)]  # fmt: skip
        pending = {m.name: math.ceil(m.reps / settings.chunk) for m in mechanisms}
        parts: dict[str, list] = {name: [] for name in names}
        summaries: dict[str, dict] = {}
        for done, (name, start, rows, seconds) in enumerate(
            imap(_mechanism_replicate_task, tasks), start=1
        ):
            parts[name].append((start, rows, seconds))
            pending[name] -= 1
            if pending[name] == 0:
                chunks = sorted(parts.pop(name), key=lambda item: item[0])
                merged = {k: np.concatenate([c[1][k] for c in chunks]) for k in ROW_FIELDS}
                m = by_name[name]
                summary = summarize_mechanism(m, params[name], merged, settings.levels,
                                              sum(c[2] for c in chunks))  # fmt: skip
                key = m.key.name
                summary["calibration"] = asdict(calibrations[key]) if key in calibrations else None
                summaries[name] = summary
            if progress is not None:
                progress("replicates", done, len(tasks))
    return {
        "calibrations": {k: asdict(v) for k, v in calibrations.items()},
        "scenarios": [summaries[name] for name in names],
    }
