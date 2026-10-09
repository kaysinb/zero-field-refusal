"""Full-analysis simulation of Gate 2 (docs/preregistration.md#full-analysis-simulation).

Each Monte Carlo replicate runs the whole procedure on synthetic runs: sweep runs (seeds 0-4 in
every cell) → cell means → K\\* and decisive cells, fixed from then on → look 1 with seeds 0-19 in
the decisive cells → look 2 with seeds 0-39 if look 1 is inconclusive. The analysis is the
library's own (``select_k_star``, ``curve_draws``, the CI methods and the decision rules). Every CI
method and level is evaluated on the same data and the same bootstrap resamples; a ``Variant``
(resample count, exclusion factor) reruns the analysis of the same data.

Generative model (parameters in ``GenerativeModel``; per-scenario ones in ``Scenario``):

- Arms: sgd, adam_frozen, adamw; the H2a control arms are omitted.
- Cells: 4 learning rates per arm, each with a true mean log K (natural log of K in nats) given by
  a placement. ``standard``: common range [0.5, 2.5]. ``narrow``: AdamW shifted up, so the
  SGD-AdamW overlap is [2.0, 2.5]. ``frozen_apart``: frozen Adam below the SGD-AdamW overlap, so
  H2 is descriptive only and many frozen-Adam runs fall below C_lo/10.
- log K of a run = cell mean + seed effect + run noise. The seed effect, N(0, 0.20²), is shared by
  every cell of the seed (the shared teacher stream); the run noise is N(0, 0.25²); together they
  give a within-cell SD of log K of 0.32.
- Base curve, the SGD curve: ``logistic`` in log K, rising from 0.02 at log K = −1 to 0.60 at
  log K = 3.5. Two misspecified bases test the logistic fit under the null: ``saturating`` is
  0.7 · logistic, and ``floor`` is 0.05 + 0.95 · logistic (a judge false-positive floor).
- AdamW: ``shift`` curves are clip(base + Δ, 0, 1), so the true Δ is Δ at any unclipped K\\*; with
  Δ = 0 every arm shares the base curve. ``crossing`` curves (logistic base only): logit h_AdamW
  has ``crossing_slope_ratio`` times the SGD slope and passes through h_SGD(K\\*) + Δ at the K\\*
  realized in the replicate. K\\* depends on K values only, so outcomes are drawn after it is
  selected; the true Δ(K\\*) is then exactly Δ although the curves differ in shape elsewhere, and
  both arms are correctly specified. Frozen Adam follows SGD, AdamW, or the mean of the two,
  optionally shifted by a probability offset (clipped to [0, 1]); the H2 boundary scenarios put it
  beyond SGD, on the ±10 pp margin, or between but close to SGD.
- Seed variability: a run's rate is h_run ~ Beta(κ·h, κ·(1 − h)), h being the arm's curve at the
  run's K, so E[h_run | K] = h(K) exactly (beta-binomial overdispersion). κ is calibrated
  numerically (Gauss-Hermite quadrature over log K) so that the SD across seeds of the observed SGD
  rate in a cell at the middle of the curve (log K = 1.25; h = 0.15 for the logistic base),
  including the K spread and binomial noise, equals σ; it is calibrated per base curve. Runs are
  coupled through a Gaussian copula: h_run is the Beta quantile of Φ(z), z = √ρ·ζ_seed +
  √(1 − ρ)·ε_run, so ρ is the latent correlation between any two runs of one seed, across arms
  and learning rates. Rate effects are independent of the K seed effects.
- Outcomes: harmful ~ Binomial(300, h_run).

Randomness: the data of replicate ``rep`` come from ``derive_seed(base_seed, "simulation",
scenario name, rep)`` and its resamples from ``derive_seed(..., rep, "bootstrap", look)``. A
replicate is therefore reproducible on its own, and every CI method, level and variant sees the
same data.
"""

from __future__ import annotations

import dataclasses
import multiprocessing as mp
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import product
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.special import betaincinv, expit, logit
from scipy.stats import norm

from ..seeding import derive_seed
from .bootstrap import CI_METHODS, BootstrapDraws, confidence_intervals, dropped_replicates
from .decisions import (
    DECISIONS,
    EQUIVALENCE_MARGIN,
    INCONCLUSIVE,
    MECHANISMS,
    UNAVAILABLE,
    h2_comparison,
    h2_mechanism,
    look_decision,
)
from .kstar import EXCLUSION_FACTOR, CommonRangeUndefined, KStarSelection, select_k_star
from .primary import (
    FROZEN_VS_ADAMW,
    FROZEN_VS_SGD,
    PRIMARY_CONTRAST,
    curve_draws,
    frozen_adam_between,
)

SIM_ARMS = ("sgd", "adam_frozen", "adamw")
LEARNING_RATES = (1e-5, 2e-5, 4e-5, 8e-5)  # labels of the four cells of an arm
PLACEMENTS = {
    "standard": {
        "sgd": (-1.0, 0.5, 1.5, 2.5),
        "adam_frozen": (0.0, 1.25, 2.5, 3.0),
        "adamw": (0.5, 1.5, 2.5, 3.5),
    },
    "narrow": {
        "sgd": (-1.0, 0.5, 1.5, 2.5),
        "adam_frozen": (1.0, 2.0, 3.0, 3.75),
        "adamw": (2.0, 2.75, 3.5, 4.25),
    },
    "frozen_apart": {
        "sgd": (-1.0, 0.5, 1.5, 2.5),
        "adam_frozen": (-3.0, -2.25, -1.5, -0.75),
        "adamw": (0.5, 1.5, 2.5, 3.5),
    },
}
BASES = ("logistic", "saturating", "floor")
MISSPECIFIED_BASES = ("saturating", "floor")
SIGMAS = (0.10, 0.15, 0.22)
H2_BOUNDARY_OFFSETS = (-0.10, -0.05, 0.05, 0.10)
RHOS = (0.0, 0.5)

PLANNING_LEVEL = 0.9706
LEVELS = (
    *(0.95, 0.96, 0.965, 0.97, 0.9706, 0.9725, 0.975, 0.9775, 0.98),
    *(0.9825, 0.985, 0.9875, 0.99, 0.9925, 0.995),
)
# Acceptance criteria (preregistration, full-analysis simulation).
MAX_FALSE_POSITIVE = 0.025  # P(positive | Δ = 0)
MAX_DIRECTIONAL = 0.05  # P(positive or opposite | Δ = 0)
MAX_EQUIVALENCE = 0.05  # P(equivalence | |Δ| = 10 pp)
POWER_DELTAS = (0.10, 0.15, 0.20)
# The level calibration carries over between resample counts when no decision rate differs from
# that with the largest count by more than this.
MAX_RATE_DIFFERENCE = 0.01
RATE_COLUMNS = ("p_positive", "p_opposite", "p_equivalence")
EXCLUSION_FACTORS = (3.0, EXCLUSION_FACTOR, 30.0, float("inf"))
QUADRATURE_NODES = 64

UNDEFINED_DECISION = len(DECISIONS)  # K* undefined: SGD and AdamW do not overlap
UNDEFINED_MECHANISM = len(MECHANISMS)
INCONCLUSIVE_CODE = DECISIONS.index(INCONCLUSIVE)
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class GenerativeModel:
    judged: int = 300
    log_k_seed_sd: float = 0.20
    log_k_run_sd: float = 0.25
    curve_log_k: tuple[float, float] = (-1.0, 3.5)  # where the logistic base equals curve_rate
    curve_rate: tuple[float, float] = (0.02, 0.60)
    saturation: float = 0.7  # the saturating base: saturation · logistic
    judge_floor: float = 0.05  # the floor base: judge_floor + (1 − judge_floor) · logistic
    crossing_slope_ratio: float = 0.5
    sweep_seeds: int = 5
    look_seeds: tuple[int, int] = (20, 40)
    placements: dict[str, dict[str, tuple[float, ...]]] = field(default_factory=lambda: PLACEMENTS)

    @property
    def log_k_sd(self) -> float:
        """Within-cell SD of log K."""
        return float(np.hypot(self.log_k_seed_sd, self.log_k_run_sd))

    @property
    def sgd_coefficients(self) -> tuple[float, float]:
        """(a, b) of the logistic base, logit h = a + b·log K."""
        (x0, x1), (h0, h1) = self.curve_log_k, self.curve_rate
        slope = (logit(h1) - logit(h0)) / (x1 - x0)
        return float(logit(h0) - slope * x0), float(slope)

    def sgd_rate(self, log_k: np.ndarray) -> np.ndarray:
        """The logistic base curve."""
        a, b = self.sgd_coefficients
        return expit(a + b * np.asarray(log_k, dtype=float))

    def base_rate(self, log_k: np.ndarray, base: str) -> np.ndarray:
        """The SGD curve of a scenario: the logistic base or one of its misspecified variants."""
        logistic = self.sgd_rate(log_k)
        if base == "logistic":
            return logistic
        if base == "saturating":
            return self.saturation * logistic
        if base == "floor":
            return self.judge_floor + (1 - self.judge_floor) * logistic
        raise ValueError(f"Unknown base curve {base!r}; expected one of {BASES}")


@dataclass(frozen=True)
class Scenario:
    placement: str
    delta: float
    sigma: float
    rho: float
    frozen_adam: str = "between"  # "sgd" | "adamw" | "between"
    frozen_offset: float = 0.0  # added to the frozen-Adam curve
    curves: str = "shift"  # "shift" | "crossing"
    base: str = "logistic"  # "logistic" | "saturating" | "floor"
    reps: int = 1000

    @property
    def name(self) -> str:
        # Logistic-base names carry no base, so their data streams are those of earlier runs.
        curves = self.curves if self.base == "logistic" else f"{self.curves}-{self.base}"
        offset = f"{self.frozen_offset:+.2f}" if self.frozen_offset else ""
        return (
            f"{self.placement}_{curves}_d{self.delta:+.2f}_fa-{self.frozen_adam}{offset}"
            f"_s{self.sigma:.2f}_r{self.rho:.2f}"
        )


class Variant(NamedTuple):
    """How the same data are analyzed: resample count and small-K exclusion factor."""

    resamples: int
    exclusion_factor: float = EXCLUSION_FACTOR


@dataclass(frozen=True)
class Part:
    """Scenarios analyzed under several variants; written to ``<name>.csv``."""

    name: str
    scenarios: tuple[Scenario, ...]
    variants: tuple[Variant, ...]
    reference: Variant  # the other variants' decisions are compared with this one's


@dataclass(frozen=True)
class SimulationSettings:
    model: GenerativeModel
    grid: Part  # the acceptance grid, one variant
    resample_comparison: Part | None
    exclusion_sensitivity: Part | None
    h2_boundary: Part | None = None  # descriptive H2 label frequencies
    methods: tuple[str, ...] = CI_METHODS
    levels: tuple[float, ...] = LEVELS
    margin: float = EQUIVALENCE_MARGIN
    base_seed: int = 0
    chunk: int = 25  # replicates per worker task

    @property
    def parts(self) -> tuple[Part, ...]:
        extra = (self.resample_comparison, self.exclusion_sensitivity, self.h2_boundary)
        return (self.grid, *(part for part in extra if part is not None))


@dataclass(frozen=True)
class ReplicateOutcome:
    """One replicate under one variant. Codes index DECISIONS and MECHANISMS; the next code means
    K* was undefined."""

    look1: np.ndarray  # (methods, levels) decision code at look 1
    final: np.ndarray  # (methods, levels) decision code after the sequential rule
    mechanism: np.ndarray  # (methods, levels) H2 mechanism code at the deciding look
    diagnostics: dict[str, float]


def _nulls(sigma: float, rho: float, reps: int) -> list[Scenario]:
    """Δ = 0 scenarios: every placement, the exact shape-H0 and the misspecified bases."""
    nulls = [Scenario(placement, 0.0, sigma, rho, reps=reps) for placement in PLACEMENTS]
    nulls.append(
        Scenario("standard", 0.0, sigma, rho, frozen_adam="sgd", curves="crossing", reps=reps)
    )
    nulls += [
        Scenario(placement, 0.0, sigma, rho, base=base, reps=reps)
        for base, placement in product(MISSPECIFIED_BASES, ("standard", "narrow"))
    ]
    return nulls


def _h2_boundary(sigma: float, rho: float, reps: int) -> list[Scenario]:
    """Frozen Adam at SGD − 0.05 (beyond, within the margin), SGD ± 0.10 (on the margin) and
    SGD + 0.05 (between, near SGD), at Δ = 0.15 and 0.20 in the standard placement."""
    return [
        Scenario("standard", delta, sigma, rho, frozen_adam="sgd", frozen_offset=offset, reps=reps)
        for delta, offset in product((0.15, 0.20), H2_BOUNDARY_OFFSETS)
    ]


def full_settings(base_seed: int = 0) -> SimulationSettings:
    """The preregistered grid, σ ∈ {0.10, 0.15, 0.22} × ρ ∈ {0, 0.5}, and the side studies."""
    grid = []
    for sigma, rho in product(SIGMAS, RHOS):
        grid += _nulls(sigma, rho, reps=3000)
        for delta, placement in product(POWER_DELTAS, PLACEMENTS):
            grid.append(Scenario(placement, delta, sigma, rho))
        grid.append(Scenario("standard", -0.10, sigma, rho))
        for placement in PLACEMENTS:
            grid.append(Scenario(placement, 0.10, sigma, rho, curves="crossing"))
        grid.append(Scenario("standard", -0.10, sigma, rho, curves="crossing"))
        for frozen_adam in ("sgd", "adamw"):
            grid.append(Scenario("standard", 0.15, sigma, rho, frozen_adam=frozen_adam))
    resamples = tuple(Variant(r) for r in (1000, 2000, 4000))
    exclusion = tuple(Variant(2000, factor) for factor in EXCLUSION_FACTORS)
    return SimulationSettings(
        model=GenerativeModel(),
        grid=Part("results", tuple(grid), (Variant(2000),), Variant(2000)),
        resample_comparison=Part(
            "resamples",
            (
                Scenario("standard", 0.0, 0.15, 0.5),
                Scenario("standard", 0.10, 0.15, 0.5),
                Scenario("standard", 0.0, 0.22, 0.5),
            ),
            resamples,
            Variant(4000),
        ),
        exclusion_sensitivity=Part(
            "exclusion",
            tuple(
                Scenario(placement, 0.0, sigma, rho, base=base)
                for sigma, rho in product(SIGMAS, RHOS)
                for base, placement in product(BASES, ("standard", "narrow"))
            ),
            exclusion,
            Variant(2000),
        ),
        h2_boundary=Part(
            "h2_boundary",
            tuple(
                scenario
                for sigma, rho in product(SIGMAS, RHOS)
                for scenario in _h2_boundary(sigma, rho, reps=1000)
            ),
            (Variant(2000),),
            Variant(2000),
        ),
        base_seed=base_seed,
    )


def confirm_settings(base_seed: int) -> SimulationSettings:
    """The binding σ = 0.22 null scenarios of the grid, to rerun with a fresh base seed."""
    nulls = tuple(s for rho in RHOS for s in _nulls(0.22, rho, reps=3000))
    return SimulationSettings(
        model=GenerativeModel(),
        grid=Part("results", nulls, (Variant(2000),), Variant(2000)),
        resample_comparison=None,
        exclusion_sensitivity=None,
        base_seed=base_seed,
    )


def quick_settings(base_seed: int = 0) -> SimulationSettings:
    """A smoke run of every part: few replicates and resamples."""
    grid = (
        Scenario("standard", 0.0, 0.15, 0.5, reps=6),
        Scenario("standard", 0.0, 0.15, 0.5, frozen_adam="sgd", curves="crossing", reps=6),
        Scenario("standard", 0.0, 0.15, 0.5, base="saturating", reps=6),
        Scenario("narrow", 0.0, 0.15, 0.5, base="floor", reps=6),
        Scenario("standard", 0.10, 0.15, 0.5, reps=6),
        Scenario("standard", 0.10, 0.15, 0.5, curves="crossing", reps=6),
        Scenario("standard", 0.20, 0.15, 0.5, reps=6),
    )
    return SimulationSettings(
        model=GenerativeModel(),
        grid=Part("results", grid, (Variant(200),), Variant(200)),
        resample_comparison=Part(
            "resamples",
            (Scenario("standard", 0.0, 0.15, 0.5, reps=4),),
            (Variant(100), Variant(200)),
            Variant(200),
        ),
        exclusion_sensitivity=Part(
            "exclusion",
            (Scenario("narrow", 0.0, 0.15, 0.5, base="floor", reps=4),),
            tuple(Variant(200, factor) for factor in EXCLUSION_FACTORS),
            Variant(200),
        ),
        h2_boundary=Part(
            "h2_boundary",
            tuple(s for s in _h2_boundary(0.15, 0.5, reps=3) if s.delta == 0.15),
            (Variant(200),),
            Variant(200),
        ),
        base_seed=base_seed,
        chunk=6,
    )


def quick_confirm_settings(base_seed: int) -> SimulationSettings:
    """A smoke run of the confirmation."""
    nulls = (
        Scenario("standard", 0.0, 0.22, 0.5, reps=4),
        Scenario("narrow", 0.0, 0.22, 0.5, base="floor", reps=4),
    )
    return SimulationSettings(
        model=GenerativeModel(),
        grid=Part("results", nulls, (Variant(200),), Variant(200)),
        resample_comparison=None,
        exclusion_sensitivity=None,
        base_seed=base_seed,
        chunk=4,
    )


def beta_concentration(model: GenerativeModel, sigma: float, base: str = "logistic") -> float:
    """κ such that the SD across seeds of the observed SGD rate at the middle of the curve is σ."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(QUADRATURE_NODES)
    weights = weights / weights.sum()
    h = model.base_rate(np.mean(model.curve_log_k) + model.log_k_sd * nodes, base)
    curve_variance = weights @ (h - weights @ h) ** 2
    binomial_scale = weights @ (h * (1 - h))
    m = model.judged
    # Var(observed rate) = Var(h(K)) + E[h(1 − h)]·(q + (1 − q)/m), with q = 1/(κ + 1) the share
    # of h(1 − h) that the Beta draw contributes.
    q = ((sigma**2 - curve_variance) / binomial_scale - 1 / m) / (1 - 1 / m)
    if not 0 < q < 1:
        raise ValueError(f"σ = {sigma} cannot be reached with this model (q = {q:.4f})")
    return float(1 / q - 1)


def draw_run_rates(h: np.ndarray, kappa: float, latent: np.ndarray) -> np.ndarray:
    """Run rates: the Beta(κh, κ(1 − h)) quantile of Φ(latent); h of exactly 0 or 1 is kept."""
    h = np.asarray(h, dtype=float)
    rates = h.copy()
    inside = (h > 0) & (h < 1)
    rates[inside] = betaincinv(
        kappa * h[inside], kappa * (1 - h[inside]), norm.cdf(np.asarray(latent)[inside])
    )
    return rates


def simulate_runs(
    model: GenerativeModel, scenario: Scenario, kappa: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, KStarSelection]:
    """Synthetic runs of all seeds (0-39 in decisive cells) and the K* fixed from the sweep.

    Raises ``CommonRangeUndefined`` when the sweep's SGD and AdamW ranges do not overlap.
    """
    n_seeds = model.look_seeds[-1]
    seed_log_k = rng.normal(0.0, model.log_k_seed_sd, n_seeds)
    seed_latent = rng.standard_normal(n_seeds)
    means = model.placements[scenario.placement]

    def draw_runs(arm: str, lr: float, seeds: np.ndarray) -> pd.DataFrame:
        cell_mean = means[arm][LEARNING_RATES.index(lr)]
        log_k = cell_mean + seed_log_k[seeds] + rng.normal(0.0, model.log_k_run_sd, len(seeds))
        return pd.DataFrame({"seed": seeds, "arm": arm, "lr": lr, "log_k": log_k})

    sweep_seeds = np.arange(model.sweep_seeds)
    sweep = pd.concat([draw_runs(a, lr, sweep_seeds) for a in SIM_ARMS for lr in LEARNING_RATES])
    sweep["K"] = np.exp(sweep["log_k"])
    sweep["run_id"] = np.arange(len(sweep)).astype(str)
    selection = select_k_star(sweep)

    later_seeds = np.arange(model.sweep_seeds, n_seeds)
    extra = [draw_runs(arm, lr, later_seeds) for arm in SIM_ARMS for lr in selection.decisive[arm]]
    runs = pd.concat([sweep.drop(columns=["K", "run_id"]), *extra], ignore_index=True)
    log_k_star = float(np.log(selection.k_star))
    h = np.empty(len(runs))
    for arm in SIM_ARMS:
        rows = (runs["arm"] == arm).to_numpy()
        h[rows] = true_rate(model, scenario, arm, runs.loc[rows, "log_k"].to_numpy(), log_k_star)
    shared = np.sqrt(scenario.rho) * seed_latent[runs["seed"].to_numpy()]
    latent = shared + np.sqrt(1 - scenario.rho) * rng.standard_normal(len(runs))
    runs["harmful"] = rng.binomial(model.judged, draw_run_rates(h, kappa, latent))
    runs["judged"] = model.judged
    runs["K"] = np.exp(runs["log_k"])
    runs["run_id"] = np.arange(len(runs)).astype(str)
    return runs, selection


def simulate_replicate(
    model: GenerativeModel,
    scenario: Scenario,
    kappa: float,
    rep: int,
    *,
    base_seed: int,
    variants: Sequence[Variant],
    methods: Sequence[str],
    levels: Sequence[float],
    margin: float,
) -> dict[Variant, ReplicateOutcome]:
    """Run the sweep, the K* selection and both looks once; one outcome per variant."""
    rng = np.random.default_rng(derive_seed(base_seed, "simulation", scenario.name, rep))
    try:
        runs, selection = simulate_runs(model, scenario, kappa, rng)
    except CommonRangeUndefined:
        shape = (len(methods), len(levels))
        undefined = ReplicateOutcome(
            look1=np.full(shape, UNDEFINED_DECISION, dtype=np.int8),
            final=np.full(shape, UNDEFINED_DECISION, dtype=np.int8),
            mechanism=np.full(shape, UNDEFINED_MECHANISM, dtype=np.int8),
            diagnostics={"k_star_defined": 0.0},
        )
        return dict.fromkeys(variants, undefined)

    looks = [runs[runs["seed"] < n] for n in model.look_seeds]
    log_k_star = float(np.log(selection.k_star))
    common = {
        "k_star_defined": 1.0,
        "log_k_star": log_k_star,
        "true_delta": float(
            true_rate(model, scenario, "adamw", np.array([log_k_star]), log_k_star)[0]
            - true_rate(model, scenario, "sgd", np.array([log_k_star]), log_k_star)[0]
        ),
        "one_sided_arms": float(sum(len(selection.decisive[arm]) == 1 for arm in SIM_ARMS)),
        "descriptive_only": float(selection.h2_descriptive_only),
    }
    outcomes = {}
    for variant in variants:
        fixed = dataclasses.replace(selection, exclusion_factor=variant.exclusion_factor)
        bootstrap_seed = derive_seed(base_seed, "simulation", scenario.name, rep, "bootstrap", 1)
        look1, mechanism1, diagnostics = _look(
            looks[0], fixed, variant.resamples, bootstrap_seed, methods, levels, margin
        )
        final, mechanism = look1.copy(), mechanism1.copy()
        diagnostics = {f"{key}_look1": value for key, value in diagnostics.items()}
        pending = look1 == INCONCLUSIVE_CODE
        if pending.any():
            bootstrap_seed = derive_seed(
                base_seed, "simulation", scenario.name, rep, "bootstrap", 2
            )
            look2, mechanism2, diagnostics2 = _look(
                looks[1], fixed, variant.resamples, bootstrap_seed, methods, levels, margin
            )
            final[pending], mechanism[pending] = look2[pending], mechanism2[pending]
            diagnostics |= {f"{key}_look2": value for key, value in diagnostics2.items()}
        outcomes[variant] = ReplicateOutcome(look1, final, mechanism, common | diagnostics)
    return outcomes


def true_rate(
    model: GenerativeModel, scenario: Scenario, arm: str, log_k: np.ndarray, log_k_star: float
) -> np.ndarray:
    """The arm's true curve h(K) in a scenario, given the realized K*."""
    sgd = model.base_rate(log_k, scenario.base)
    if arm == "sgd":
        return sgd
    if scenario.curves == "shift":
        adamw = np.clip(sgd + scenario.delta, 0.0, 1.0)
    elif scenario.curves == "crossing":
        if scenario.base != "logistic":
            raise ValueError("Crossing curves need the logistic base")
        _, slope = model.sgd_coefficients
        anchor = logit(model.sgd_rate(log_k_star) + scenario.delta)
        adamw = expit(anchor + model.crossing_slope_ratio * slope * (log_k - log_k_star))
    else:
        raise ValueError(f"Unknown curves {scenario.curves!r}")
    if arm == "adamw":
        return adamw
    frozen = {"sgd": sgd, "adamw": adamw, "between": (sgd + adamw) / 2}[scenario.frozen_adam]
    return np.clip(frozen + scenario.frozen_offset, 0.0, 1.0) if scenario.frozen_offset else frozen


def _look(
    table: pd.DataFrame,
    selection: KStarSelection,
    resamples: int,
    seed: int,
    methods: Sequence[str],
    levels: Sequence[float],
    margin: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Decision and H2 mechanism codes, (methods, levels), and diagnostics of one look."""
    draws = curve_draws(table, selection, resamples=resamples, seed=seed)
    primary = draws.contrast(PRIMARY_CONTRAST)
    frozen: tuple[BootstrapDraws, ...] = ()
    if "adam_frozen" in draws.curves:
        frozen = (draws.contrast(FROZEN_VS_SGD), draws.contrast(FROZEN_VS_ADAMW))
        between = frozen_adam_between(draws)
    levels = np.asarray(levels)
    decisions = np.empty((len(methods), len(levels)), dtype=np.int8)
    mechanisms = np.empty_like(decisions)
    for i, method in enumerate(methods):
        primary_ci = confidence_intervals(primary, method, levels)
        frozen_ci = [confidence_intervals(d, method, levels) for d in frozen]
        for k in range(len(levels)):
            decision = look_decision(*primary_ci[k], margin)
            mechanism = UNAVAILABLE
            if frozen_ci:
                mechanism = h2_mechanism(
                    h2_comparison(*frozen_ci[0][k], margin),
                    h2_comparison(*frozen_ci[1][k], margin),
                    between,
                    descriptive_only=selection.h2_descriptive_only,
                    primary_decision=decision,
                )
            decisions[i, k] = DECISIONS.index(decision)
            mechanisms[i, k] = MECHANISMS.index(mechanism)
    curves = draws.curves.values()
    # bootstrap_t drops the most replicates (a non-finite value or SE): an upper bound for all.
    diagnostics = {
        "delta_hat": primary.estimate,
        "excluded_runs": float(sum(c.runs_excluded for c in curves) + sum(draws.unfitted.values())),
        "separated_fits": float(sum(c.fit.separated.sum() for c in curves)),
        "not_converged_fits": float(sum((~c.fit.converged).sum() for c in curves)),
        "dropped_primary": float(dropped_replicates(primary, "bootstrap_t")),
        "dropped_h2": float(sum(dropped_replicates(d, "bootstrap_t") for d in frozen)),
    }
    return decisions, mechanisms, diagnostics


# Running and summarizing


@dataclass(frozen=True)
class SimulationResult:
    tables: dict[str, pd.DataFrame]  # part name -> one row per scenario × variant × method × level
    kappa: dict[str, float]  # "σ/base" -> κ


def run_simulation(
    settings: SimulationSettings,
    *,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> SimulationResult:
    """Simulate every part; ``workers`` > 1 runs chunks of replicates in parallel."""
    calibrations = {(s.sigma, s.base) for part in settings.parts for s in part.scenarios}
    kappa = {key: beta_concentration(settings.model, *key) for key in sorted(calibrations)}
    # Parts with more variants are slower; they go first, so no worker is left with them at the end.
    parts = sorted(settings.parts, key=lambda part: -len(part.variants))
    tasks = [
        (settings, scenario, kappa[(scenario.sigma, scenario.base)], start, part.variants)
        for part in parts
        for scenario in part.scenarios
        for start in range(0, scenario.reps, settings.chunk)
    ]
    collected: dict[tuple[str, tuple[Variant, ...]], list] = {}
    with _executor(workers) as imap:
        for done, (key, start, chunk) in enumerate(imap(_run_chunk, tasks), start=1):
            collected.setdefault(key, []).append((start, chunk))
            if progress is not None:
                progress(done, len(tasks))

    tables = {}
    for part in settings.parts:
        rows = []
        for scenario in part.scenarios:
            chunks = sorted(collected[(scenario.name, part.variants)], key=lambda item: item[0])
            by_variant = {v: [o for _, chunk in chunks for o in chunk[v]] for v in part.variants}
            reference = np.stack([o.final for o in by_variant[part.reference]])
            for variant, outcomes in by_variant.items():
                disagreement = (np.stack([o.final for o in outcomes]) != reference).mean(0)
                summary = _summarize(scenario, variant, outcomes, settings)
                for row, share in zip(summary, disagreement.ravel(), strict=True):
                    row["disagree_with_reference"] = float(share)
                rows += summary
        tables[part.name] = pd.DataFrame(rows)
    labels = {f"{sigma:.2f}/{base}": value for (sigma, base), value in kappa.items()}
    return SimulationResult(tables, labels)


def _run_chunk(task):
    settings, scenario, kappa, start, variants = task
    stop = min(start + settings.chunk, scenario.reps)
    chunk: dict[Variant, list[ReplicateOutcome]] = {v: [] for v in variants}
    for rep in range(start, stop):
        outcomes = simulate_replicate(
            settings.model,
            scenario,
            kappa,
            rep,
            base_seed=settings.base_seed,
            variants=variants,
            methods=settings.methods,
            levels=settings.levels,
            margin=settings.margin,
        )
        for variant, outcome in outcomes.items():
            chunk[variant].append(outcome)
    return (scenario.name, variants), start, chunk


@contextmanager
def _executor(workers: int) -> Iterator[Callable]:
    if workers <= 1:
        yield map
        return
    # Fresh worker processes with single-threaded BLAS: parallelism comes from the workers.
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


def _summarize(
    scenario: Scenario,
    variant: Variant,
    outcomes: list[ReplicateOutcome],
    settings: SimulationSettings,
) -> list[dict]:
    """Rates per CI method × level, plus the scenario's diagnostics, over its replicates."""
    look1 = np.stack([o.look1 for o in outcomes])
    final = np.stack([o.final for o in outcomes])
    mechanism = np.stack([o.mechanism for o in outcomes])
    n = len(outcomes)
    diagnostics = pd.DataFrame([o.diagnostics for o in outcomes])

    def mean(column: str) -> float:
        return float(diagnostics[column].mean()) if column in diagnostics else float("nan")

    def sd(column: str) -> float:
        return float(diagnostics[column].std()) if column in diagnostics else float("nan")

    def per_replicate(name: str) -> pd.Series:
        """Sum over looks, NaN for replicates without a defined K*."""
        return diagnostics.filter(like=name).sum(axis=1, min_count=1)

    separated = per_replicate("separated_fits")
    shared = {
        "scenario": scenario.name,
        "placement": scenario.placement,
        "curves": scenario.curves,
        "base": scenario.base,
        "delta": scenario.delta,
        "frozen_adam": scenario.frozen_adam,
        "frozen_offset": scenario.frozen_offset,
        "sigma": scenario.sigma,
        "rho": scenario.rho,
        "reps": n,
        "resamples": variant.resamples,
        "exclusion_factor": variant.exclusion_factor,
        "mean_true_delta": mean("true_delta"),
        "mean_log_k_star": mean("log_k_star"),
        "sd_log_k_star": sd("log_k_star"),
        "mean_delta_hat_look1": mean("delta_hat_look1"),
        "sd_delta_hat_look1": sd("delta_hat_look1"),
        "mean_excluded_runs_look1": mean("excluded_runs_look1"),
        "mean_excluded_runs_look2": mean("excluded_runs_look2"),
        "mean_separated_fits": float(separated.mean()),
        "share_reps_with_separation": float((separated > 0).mean()),
        "mean_not_converged_fits": float(per_replicate("not_converged_fits").mean()),
        "mean_dropped_replicates_primary": float(per_replicate("dropped_primary").mean()),
        "mean_dropped_replicates_h2": float(per_replicate("dropped_h2").mean()),
        "share_one_sided_arms": mean("one_sided_arms") / len(SIM_ARMS),
        "share_descriptive_only": mean("descriptive_only"),
    }
    rows = []
    for i, method in enumerate(settings.methods):
        for k, level in enumerate(settings.levels):
            row = shared | {"method": method, "level": level}
            for code, decision in enumerate(DECISIONS):
                row[f"p_{decision}"] = float((final[:, i, k] == code).mean())
                row[f"look1_{decision}"] = float((look1[:, i, k] == code).mean())
            row["p_undefined"] = float((final[:, i, k] == UNDEFINED_DECISION).mean())
            row["stop_look1"] = float(
                (
                    (look1[:, i, k] != INCONCLUSIVE_CODE) & (look1[:, i, k] != UNDEFINED_DECISION)
                ).mean()
            )
            for code, label in enumerate(MECHANISMS):
                row[f"h2_{label}"] = float((mechanism[:, i, k] == code).mean())
            for decision in ("positive", "equivalence"):
                p = row[f"p_{decision}"]
                row[f"se_p_{decision}"] = float(np.sqrt(p * (1 - p) / n))
            rows.append(row)
    return rows


def null_worst_cases(table: pd.DataFrame, by: Sequence[str] = ("method", "level")) -> list[dict]:
    """Largest P(positive) and P(positive or opposite) over the Δ = 0 scenarios, per group."""
    null = table[table["delta"] == 0]
    worst = []
    for keys, group in null.groupby(list(by), sort=True):
        directional = group["p_positive"] + group["p_opposite"]
        worst.append(
            {
                **{name: _plain(key) for name, key in zip(by, keys, strict=True)},
                "max_p_positive_null": _worst(group["p_positive"], group),
                "max_p_directional_null": _worst(directional, group),
            }
        )
    return worst


def h2_label_frequencies(table: pd.DataFrame, method: str, level: float) -> list[dict]:
    """Per scenario, P(positive) and the share of each H2 label at one CI method and level."""
    rows = table[(table["method"] == method) & np.isclose(table["level"], level)]
    keys = ("scenario", "delta", "frozen_adam", "frozen_offset", "sigma", "rho", "p_positive")
    return [
        {key: _plain(row[key]) for key in keys}
        | {"h2": {label: float(row[f"h2_{label}"]) for label in MECHANISMS}}
        for _, row in rows.iterrows()
    ]


def acceptance(results: pd.DataFrame) -> list[dict]:
    """Acceptance criteria and power per CI method × level, over every scenario of the grid.

    The null criteria take the worst case over all Δ = 0 scenarios, misspecified ones included;
    the equivalence criterion over all |Δ| = 10 pp scenarios, shift and crossing. Power is reported
    per model of the alternative (shift or crossing curves, logistic base).
    """
    nulls = {(w["method"], w["level"]): w for w in null_worst_cases(results)}
    verdicts = []
    for (method, level), group in results.groupby(["method", "level"], sort=True):
        alternative = group[np.isclose(group["delta"].abs(), 0.10)]
        null = nulls.get((method, float(level)), {})
        criteria = {
            "max_p_positive_null": null.get("max_p_positive_null", _worst(pd.Series(), group)),
            "max_p_directional_null": null.get(
                "max_p_directional_null", _worst(pd.Series(), group)
            ),
            "max_p_equivalence_at_10pp": _worst(alternative["p_equivalence"], alternative),
        }
        passes = (
            criteria["max_p_positive_null"]["value"] <= MAX_FALSE_POSITIVE
            and criteria["max_p_directional_null"]["value"] <= MAX_DIRECTIONAL
            and criteria["max_p_equivalence_at_10pp"]["value"] <= MAX_EQUIVALENCE
        )
        power: dict[str, dict] = {}
        for model in ("shift", "crossing"):
            for delta in POWER_DELTAS:
                rows = group[
                    np.isclose(group["delta"], delta)
                    & (group["curves"] == model)
                    & (group["base"] == "logistic")
                ]
                if rows.empty:
                    continue
                power.setdefault(model, {})[f"{delta:.2f}"] = {
                    "min": float(rows["p_positive"].min()),
                    "mean": float(rows["p_positive"].mean()),
                    "mean_look1": float(rows["look1_positive"].mean()),
                    "mean_standard_placement": float(
                        rows.loc[rows["placement"] == "standard", "p_positive"].mean()
                    ),
                }
        verdicts.append(
            {
                "method": method,
                "level": float(level),
                "passes": bool(passes),
                **criteria,
                "power": power,
            }
        )
    return verdicts


def recommend(verdicts: list[dict], comparison: pd.DataFrame, settings: SimulationSettings) -> dict:
    """Per method, the planning level if it passes, else the smallest passing level above it.

    The recommended method has the highest mean power at Δ = 10 pp under the shift model.
    """
    per_method = {}
    for method in settings.methods:
        rows = [v for v in verdicts if v["method"] == method]
        passing = sorted(v["level"] for v in rows if v["passes"])
        planning = next((v for v in rows if np.isclose(v["level"], PLANNING_LEVEL)), None)
        if planning is not None and planning["passes"]:
            level = planning["level"]
        else:
            level = next((lv for lv in passing if lv > PLANNING_LEVEL), None)
        chosen = next((v for v in rows if level is not None and v["level"] == level), None)
        per_method[method] = {
            "planning_level_passes": None if planning is None else planning["passes"],
            "level": level,
            "smallest_passing_level": passing[0] if passing else None,
            "power": None if chosen is None else chosen["power"],
        }
    candidates = [m for m in settings.methods if per_method[m]["level"] is not None]
    best = max(
        candidates,
        key=lambda m: per_method[m]["power"]["shift"]["0.10"]["mean"],
        default=None,
    )
    recommendation = {"per_method": per_method, "method": best, "level": None, "resamples": None}
    if best is None:
        return recommendation
    level = per_method[best]["level"]
    recommendation["level"] = level
    # If the error rates at the grid's count match those at the largest count, the level calibrated
    # on the grid holds there too; more resamples then only reduce the bootstrap noise that flips
    # decisions near a CI boundary, at negligible cost in a real analysis.
    grid_count = settings.grid.variants[0].resamples
    chosen = comparison[(comparison["method"] == best) & np.isclose(comparison["level"], level)]
    most = int(chosen["resamples"].max())

    def rates(rows: pd.DataFrame) -> pd.DataFrame:
        # Under the null, P(equivalence) is the chance of a correct finding, not an error rate.
        rows = rows.set_index("scenario")
        return rows[list(RATE_COLUMNS)].assign(
            p_equivalence=rows["p_equivalence"].where(rows["delta"] != 0, 0.0)
        )

    reference = rates(chosen[chosen["resamples"] == most])
    rate_difference = {
        int(count): float((rates(group) - reference).abs().max().max())
        for count, group in chosen.groupby("resamples")
    }
    disagreement = chosen.groupby("resamples")["disagree_with_reference"].max()
    stable = rate_difference.get(grid_count, np.inf) <= MAX_RATE_DIFFERENCE
    recommendation["resamples"] = most if stable else None
    recommendation["comparison"] = {
        "calibration_carries_over": bool(stable),
        "max_rate_difference": rate_difference,
        "max_decision_disagreement": {int(r): float(s) for r, s in disagreement.items()},
        "rule": (
            f"the largest compared count ({most}) if no rate among {RATE_COLUMNS} with the "
            f"grid's {grid_count} resamples differs from that with {most} by more than "
            f"{MAX_RATE_DIFFERENCE} in any comparison scenario (P(equivalence) only where "
            "Δ ≠ 0); otherwise none (recalibrate)"
        ),
    }
    return recommendation


def _worst(values: pd.Series, rows: pd.DataFrame) -> dict:
    """The largest rate over scenarios, its scenario and its Monte Carlo SE."""
    if values.empty:
        return {"value": float("nan"), "scenario": None, "mc_se": float("nan")}
    at = values.idxmax()
    value = float(values[at])
    return {
        "value": value,
        "scenario": rows.loc[at, "scenario"],
        "mc_se": float(np.sqrt(value * (1 - value) / rows.loc[at, "reps"])),
    }


def _plain(value):
    return value.item() if hasattr(value, "item") else value
