"""Full-analysis simulation of Gate 2 (docs/preregistration.md#full-analysis-simulation).

Each Monte Carlo replicate runs the whole procedure on synthetic runs: sweep runs (seeds 0-4 in
every cell) → cell means → K\\* and decisive cells, fixed from then on → look 1 with seeds 0-19 in
the decisive cells → look 2 with seeds 0-39 if look 1 is inconclusive. The analysis is the
library's own (``select_k_star``, ``curve_draws``, the CI methods and the decision rules). Every CI
method and level is evaluated on the same data and the same bootstrap resamples.

Generative model (parameters in ``GenerativeModel``; per-scenario ones in ``Scenario``):

- Arms: sgd, adam_frozen, adamw; the H2a control arms are omitted.
- Cells: 4 learning rates per arm, each with a true mean log K (natural log of K in nats) given by
  a placement. ``standard``: common range [0.5, 2.5]. ``narrow``: AdamW shifted up, so the
  SGD-AdamW overlap is [2.0, 2.5]. ``frozen_apart``: frozen Adam below the SGD-AdamW overlap, so
  H2 is descriptive only and many frozen-Adam runs fall below C_lo/10.
- log K of a run = cell mean + seed effect + run noise. The seed effect, N(0, 0.20²), is shared by
  every cell of the seed (the shared teacher stream); the run noise is N(0, 0.25²); together they
  give a within-cell SD of log K of 0.32.
- True curves: h_SGD is logistic in log K, rising from 0.02 at log K = −1 to 0.60 at log K = 3.5.
  ``shift`` curves: h_AdamW = clip(h_SGD + Δ, 0, 1), so the true Δ is Δ at any unclipped K\\* and
  the logistic fit of AdamW is mildly misspecified. ``crossing`` curves: logit h_AdamW has
  ``crossing_slope_ratio`` times the SGD slope and passes through h_SGD(K\\*) + Δ at the K\\*
  realized in the replicate. K\\* depends on K values only, so outcomes are drawn after it is
  selected; the true Δ(K\\*) is then exactly Δ although the curves differ in shape elsewhere.
  Frozen Adam follows SGD, AdamW, or the mean of the two.
- Seed variability: a run's rate is h_run ~ Beta(κ·h, κ·(1 − h)), h being the arm's curve at the
  run's K, so E[h_run | K] = h(K) exactly (beta-binomial overdispersion). κ is calibrated
  numerically (Gauss-Hermite quadrature over log K) so that the SD across seeds of the observed SGD
  rate in a cell at the middle of the curve (log K = 1.25, h = 0.15), including the K spread and
  binomial noise, equals σ. Runs are coupled through a Gaussian copula: h_run is the Beta quantile
  of Φ(z), z = √ρ·ζ_seed + √(1 − ρ)·ε_run, so ρ is the latent correlation between any two runs of
  one seed, across arms and learning rates. Rate effects are independent of the K seed effects.
- Outcomes: harmful ~ Binomial(300, h_run).

Randomness: the data of replicate ``rep`` come from ``derive_seed(base_seed, "simulation",
scenario name, rep)`` and its resamples from ``derive_seed(..., rep, "bootstrap", look)``. A
replicate is therefore reproducible on its own, and every CI method, level and resample count sees
the same data.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from scipy.special import betaincinv, expit, logit
from scipy.stats import norm

from ..seeding import derive_seed
from .bootstrap import CI_METHODS, confidence_intervals
from .decisions import (
    DECISIONS,
    EQUIVALENCE_MARGIN,
    INCONCLUSIVE,
    MECHANISMS,
    h2_comparison,
    h2_mechanism,
    look_decision,
)
from .kstar import CommonRangeUndefined, KStarSelection, select_k_star
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
    curve_log_k: tuple[float, float] = (-1.0, 3.5)  # where h_SGD equals curve_rate
    curve_rate: tuple[float, float] = (0.02, 0.60)
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
        """(a, b) of logit h_SGD = a + b·log K."""
        (x0, x1), (h0, h1) = self.curve_log_k, self.curve_rate
        slope = (logit(h1) - logit(h0)) / (x1 - x0)
        return float(logit(h0) - slope * x0), float(slope)

    def sgd_rate(self, log_k: np.ndarray) -> np.ndarray:
        a, b = self.sgd_coefficients
        return expit(a + b * np.asarray(log_k, dtype=float))


@dataclass(frozen=True)
class Scenario:
    placement: str
    delta: float
    sigma: float
    rho: float
    frozen_adam: str = "between"  # "sgd" | "adamw" | "between"
    curves: str = "shift"  # "shift" | "crossing"
    reps: int = 1000

    @property
    def name(self) -> str:
        return (
            f"{self.placement}_{self.curves}_d{self.delta:+.2f}_fa-{self.frozen_adam}"
            f"_s{self.sigma:.2f}_r{self.rho:.2f}"
        )


@dataclass(frozen=True)
class SimulationSettings:
    model: GenerativeModel
    scenarios: tuple[Scenario, ...]
    resamples: int
    comparison: tuple[Scenario, ...]  # scenarios rerun with each comparison resample count
    comparison_resamples: tuple[int, ...]
    methods: tuple[str, ...] = CI_METHODS
    levels: tuple[float, ...] = LEVELS
    margin: float = EQUIVALENCE_MARGIN
    base_seed: int = 0
    chunk: int = 25  # replicates per worker task


@dataclass(frozen=True)
class ReplicateOutcome:
    """One replicate at one resample count. Codes index DECISIONS and MECHANISMS; the next code
    means K* was undefined."""

    look1: np.ndarray  # (methods, levels) decision code at look 1
    final: np.ndarray  # (methods, levels) decision code after the sequential rule
    mechanism: np.ndarray  # (methods, levels) H2 mechanism code at the deciding look
    diagnostics: dict[str, float]


def full_settings(base_seed: int = 0) -> SimulationSettings:
    """The preregistered grid: σ ∈ {0.10, 0.15, 0.22} × ρ ∈ {0, 0.5}."""
    scenarios = []
    for sigma, rho in product((0.10, 0.15, 0.22), (0.0, 0.5)):
        for placement in PLACEMENTS:
            scenarios.append(Scenario(placement, 0.0, sigma, rho, reps=3000))
        scenarios.append(
            Scenario("standard", 0.0, sigma, rho, frozen_adam="sgd", curves="crossing", reps=3000)
        )
        for delta, placement in product(POWER_DELTAS, PLACEMENTS):
            scenarios.append(Scenario(placement, delta, sigma, rho))
        scenarios.append(Scenario("standard", -0.10, sigma, rho))
        for frozen_adam in ("sgd", "adamw"):
            scenarios.append(Scenario("standard", 0.15, sigma, rho, frozen_adam=frozen_adam))
    comparison = (
        Scenario("standard", 0.0, 0.15, 0.5),
        Scenario("standard", 0.10, 0.15, 0.5),
        Scenario("standard", 0.0, 0.22, 0.5),
    )
    return SimulationSettings(
        model=GenerativeModel(),
        scenarios=tuple(scenarios),
        resamples=2000,
        comparison=comparison,
        comparison_resamples=(1000, 2000, 4000),
        base_seed=base_seed,
    )


def quick_settings(base_seed: int = 0) -> SimulationSettings:
    """A smoke run of the same pipeline: few replicates and resamples."""
    scenarios = tuple(
        Scenario("standard", delta, 0.15, 0.5, reps=6) for delta in (0.0, 0.10, 0.20)
    ) + (Scenario("standard", 0.0, 0.15, 0.5, frozen_adam="sgd", curves="crossing", reps=6),)
    return SimulationSettings(
        model=GenerativeModel(),
        scenarios=scenarios,
        resamples=200,
        comparison=(Scenario("standard", 0.0, 0.15, 0.5, reps=4),),
        comparison_resamples=(100, 200),
        base_seed=base_seed,
        chunk=6,
    )


def beta_concentration(model: GenerativeModel, sigma: float) -> float:
    """κ such that the SD across seeds of the observed SGD rate at the middle of the curve is σ."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(QUADRATURE_NODES)
    weights = weights / weights.sum()
    h = model.sgd_rate(np.mean(model.curve_log_k) + model.log_k_sd * nodes)
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
    resamples: Sequence[int],
    methods: Sequence[str],
    levels: Sequence[float],
    margin: float,
) -> dict[int, ReplicateOutcome]:
    """Run the sweep, the K* selection and both looks once; one outcome per resample count."""
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
        return dict.fromkeys(resamples, undefined)

    looks = [runs[runs["seed"] < n] for n in model.look_seeds]
    log_k_star = float(np.log(selection.k_star))
    common = {
        "k_star_defined": 1.0,
        "log_k_star": log_k_star,
        "true_delta": float(
            true_rate(model, scenario, "adamw", np.array([log_k_star]), log_k_star)[0]
            - model.sgd_rate(log_k_star)
        ),
        "one_sided_arms": float(sum(len(selection.decisive[arm]) == 1 for arm in SIM_ARMS)),
        "descriptive_only": float(selection.h2_descriptive_only),
    }
    outcomes = {}
    for count in resamples:
        bootstrap_seed = derive_seed(base_seed, "simulation", scenario.name, rep, "bootstrap", 1)
        look1, mechanism1, diagnostics = _look(
            looks[0], selection, count, bootstrap_seed, methods, levels, margin
        )
        final, mechanism = look1.copy(), mechanism1.copy()
        diagnostics = {f"{key}_look1": value for key, value in diagnostics.items()}
        pending = look1 == INCONCLUSIVE_CODE
        if pending.any():
            bootstrap_seed = derive_seed(
                base_seed, "simulation", scenario.name, rep, "bootstrap", 2
            )
            look2, mechanism2, diagnostics2 = _look(
                looks[1], selection, count, bootstrap_seed, methods, levels, margin
            )
            final[pending], mechanism[pending] = look2[pending], mechanism2[pending]
            diagnostics |= {f"{key}_look2": value for key, value in diagnostics2.items()}
        outcomes[count] = ReplicateOutcome(look1, final, mechanism, common | diagnostics)
    return outcomes


def true_rate(
    model: GenerativeModel, scenario: Scenario, arm: str, log_k: np.ndarray, log_k_star: float
) -> np.ndarray:
    """The arm's true curve h(K) in a scenario, given the realized K*."""
    sgd = model.sgd_rate(log_k)
    if arm == "sgd":
        return sgd
    if scenario.curves == "shift":
        adamw = np.clip(sgd + scenario.delta, 0.0, 1.0)
    elif scenario.curves == "crossing":
        _, slope = model.sgd_coefficients
        anchor = logit(model.sgd_rate(log_k_star) + scenario.delta)
        adamw = expit(anchor + model.crossing_slope_ratio * slope * (log_k - log_k_star))
    else:
        raise ValueError(f"Unknown curves {scenario.curves!r}")
    if arm == "adamw":
        return adamw
    frozen = {"sgd": sgd, "adamw": adamw, "between": (sgd + adamw) / 2}
    return frozen[scenario.frozen_adam]


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
    vs_sgd, vs_adamw = draws.contrast(FROZEN_VS_SGD), draws.contrast(FROZEN_VS_ADAMW)
    between = frozen_adam_between(draws)
    levels = np.asarray(levels)
    decisions = np.empty((len(methods), len(levels)), dtype=np.int8)
    mechanisms = np.empty_like(decisions)
    for i, method in enumerate(methods):
        intervals = [confidence_intervals(d, method, levels) for d in (primary, vs_sgd, vs_adamw)]
        for k in range(len(levels)):
            decision = look_decision(*intervals[0][k], margin)
            mechanism = h2_mechanism(
                h2_comparison(*intervals[1][k], margin),
                h2_comparison(*intervals[2][k], margin),
                between,
                descriptive_only=selection.h2_descriptive_only,
                primary_decision=decision,
            )
            decisions[i, k] = DECISIONS.index(decision)
            mechanisms[i, k] = MECHANISMS.index(mechanism)
    curves = draws.curves.values()
    diagnostics = {
        "delta_hat": primary.estimate,
        "excluded_runs": float(sum(c.runs_excluded for c in curves)),
        "separated_fits": float(sum(c.fit.separated.sum() for c in curves)),
        "not_converged_fits": float(sum((~c.fit.converged).sum() for c in curves)),
    }
    return decisions, mechanisms, diagnostics


# Running and summarizing


@dataclass(frozen=True)
class SimulationResult:
    results: pd.DataFrame  # grid: one row per scenario × method × level
    comparison: pd.DataFrame  # resample comparison: scenario × resamples × method × level
    kappa: dict[float, float]


def run_simulation(
    settings: SimulationSettings,
    *,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> SimulationResult:
    """Simulate every scenario; ``workers`` > 1 runs chunks of replicates in parallel."""
    sigmas = {s.sigma for s in settings.scenarios + settings.comparison}
    kappa = {sigma: beta_concentration(settings.model, sigma) for sigma in sorted(sigmas)}
    # The slower comparison tasks go first, so that no worker is left with them at the end.
    tasks = [
        (settings, scenario, kappa[scenario.sigma], start, resamples)
        for scenarios, resamples in (
            (settings.comparison, settings.comparison_resamples),
            (settings.scenarios, (settings.resamples,)),
        )
        for scenario in scenarios
        for start in range(0, scenario.reps, settings.chunk)
    ]
    collected: dict[tuple[str, tuple[int, ...]], list] = {}
    with _executor(workers) as imap:
        for done, (key, start, chunk) in enumerate(imap(_run_chunk, tasks), start=1):
            collected.setdefault(key, []).append((start, chunk))
            if progress is not None:
                progress(done, len(tasks))

    def outcomes(scenario: Scenario, resamples: tuple[int, ...]) -> dict[int, list]:
        chunks = sorted(collected[(scenario.name, resamples)], key=lambda item: item[0])
        return {r: [o for _, chunk in chunks for o in chunk[r]] for r in resamples}

    grid = [
        row
        for scenario in settings.scenarios
        for row in _summarize(
            scenario,
            settings.resamples,
            outcomes(scenario, (settings.resamples,))[settings.resamples],
            settings,
        )
    ]
    comparison = []
    for scenario in settings.comparison:
        by_count = outcomes(scenario, settings.comparison_resamples)
        reference = np.stack([o.final for o in by_count[max(by_count)]])
        for count, replicate_outcomes in by_count.items():
            disagreement = (np.stack([o.final for o in replicate_outcomes]) != reference).mean(0)
            rows = _summarize(scenario, count, replicate_outcomes, settings)
            for row, share in zip(rows, disagreement.ravel(), strict=True):
                row["disagree_with_most_resamples"] = float(share)
            comparison.extend(rows)
    return SimulationResult(pd.DataFrame(grid), pd.DataFrame(comparison), kappa)


def _run_chunk(task):
    settings, scenario, kappa, start, resamples = task
    stop = min(start + settings.chunk, scenario.reps)
    chunk: dict[int, list[ReplicateOutcome]] = {r: [] for r in resamples}
    for rep in range(start, stop):
        outcomes = simulate_replicate(
            settings.model,
            scenario,
            kappa,
            rep,
            base_seed=settings.base_seed,
            resamples=resamples,
            methods=settings.methods,
            levels=settings.levels,
            margin=settings.margin,
        )
        for count, outcome in outcomes.items():
            chunk[count].append(outcome)
    return (scenario.name, resamples), start, chunk


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
    resamples: int,
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

    separated = diagnostics.filter(like="separated_fits").sum(axis=1, min_count=1)
    shared = {
        "scenario": scenario.name,
        "placement": scenario.placement,
        "curves": scenario.curves,
        "delta": scenario.delta,
        "frozen_adam": scenario.frozen_adam,
        "sigma": scenario.sigma,
        "rho": scenario.rho,
        "reps": n,
        "resamples": resamples,
        "mean_true_delta": mean("true_delta"),
        "mean_log_k_star": mean("log_k_star"),
        "sd_log_k_star": float(diagnostics.get("log_k_star", pd.Series(dtype=float)).std()),
        "mean_delta_hat_look1": mean("delta_hat_look1"),
        "sd_delta_hat_look1": float(
            diagnostics.get("delta_hat_look1", pd.Series(dtype=float)).std()
        ),
        "mean_excluded_runs_look1": mean("excluded_runs_look1"),
        "mean_excluded_runs_look2": mean("excluded_runs_look2"),
        "mean_separated_fits": float(separated.mean()),
        "share_reps_with_separation": float((separated > 0).mean()),
        "mean_not_converged_fits": float(
            diagnostics.filter(like="not_converged_fits").sum(axis=1, min_count=1).mean()
        ),
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


def acceptance(results: pd.DataFrame) -> list[dict]:
    """Acceptance criteria and power per CI method × level, over every scenario of the grid."""
    verdicts = []
    for (method, level), group in results.groupby(["method", "level"], sort=True):
        null = group[group["delta"] == 0]
        alternative = group[np.isclose(group["delta"].abs(), 0.10)]
        directional = null["p_positive"] + null["p_opposite"]
        criteria = {
            "max_p_positive_null": _worst(null["p_positive"], null),
            "max_p_directional_null": _worst(directional, null),
            "max_p_equivalence_at_10pp": _worst(alternative["p_equivalence"], alternative),
        }
        passes = (
            criteria["max_p_positive_null"]["value"] <= MAX_FALSE_POSITIVE
            and criteria["max_p_directional_null"]["value"] <= MAX_DIRECTIONAL
            and criteria["max_p_equivalence_at_10pp"]["value"] <= MAX_EQUIVALENCE
        )
        power = {}
        for delta in POWER_DELTAS:
            shifted = group[np.isclose(group["delta"], delta) & (group["curves"] == "shift")]
            power[f"{delta:.2f}"] = {
                "min": float(shifted["p_positive"].min()),
                "mean": float(shifted["p_positive"].mean()),
                "mean_look1": float(shifted["look1_positive"].mean()),
                "mean_standard_placement": float(
                    shifted.loc[shifted["placement"] == "standard", "p_positive"].mean()
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
    """Per method, the planning level if it passes, else the smallest passing level above it."""
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
        key=lambda m: per_method[m]["power"]["0.10"]["mean"],
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
    chosen = comparison[(comparison["method"] == best) & np.isclose(comparison["level"], level)]
    most = max(settings.comparison_resamples)
    reference = chosen[chosen["resamples"] == most].set_index("scenario")[list(RATE_COLUMNS)]
    rate_difference = {
        int(count): float(
            (group.set_index("scenario")[list(RATE_COLUMNS)] - reference).abs().max().max()
        )
        for count, group in chosen.groupby("resamples")
    }
    disagreement = chosen.groupby("resamples")["disagree_with_most_resamples"].max()
    stable = rate_difference.get(settings.resamples, np.inf) <= MAX_RATE_DIFFERENCE
    recommendation["resamples"] = most if stable else None
    recommendation["comparison"] = {
        "calibration_carries_over": bool(stable),
        "max_rate_difference": rate_difference,
        "max_decision_disagreement": {int(r): float(s) for r, s in disagreement.items()},
        "rule": (
            f"the largest compared count ({most}) if no rate among {RATE_COLUMNS} with the "
            f"grid's {settings.resamples} resamples differs from that with {most} by more than "
            f"{MAX_RATE_DIFFERENCE} in any comparison scenario; otherwise none (recalibrate)"
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
