"""Paired bootstrap over seeds and its confidence intervals.

Runs with the same seed share the teacher stream across arms and learning rates. A resample
therefore draws seed ids with replacement from the seeds present at the look and keeps every run
of each drawn seed, as often as it was drawn: multinomial seed counts serve as case weights for all
runs of the seed.

Weight rows of a ``SeedResampling``: row 0 gives the point estimate (all ones), the next R rows
are the bootstrap resamples, and the last S rows are the jackknife over seeds (one seed left out),
which gives the BCa acceleration.

Interval types (two-sided, level 1 − α):
- ``percentile``: the α/2 and 1 − α/2 quantiles of the replicates;
- ``bca``: bias-corrected and accelerated (Efron 1987). The bias correction z0 comes from the
  share of replicates below the estimate (ties count half), the acceleration from the seed
  jackknife;
- ``bootstrap_t``: studentized, without a nested bootstrap. Each replicate's SE is its
  cluster-robust delta-method SE (clusters are seeds), and the interval is
  [θ̂ − q(1 − α/2)·se, θ̂ − q(α/2)·se], where q are quantiles of (θ* − θ̂)/se*.

Replicates with a non-finite value (for ``bootstrap_t`` also a non-finite or zero SE) are dropped.
A degenerate case (no usable replicate; for BCa all replicates on one side of the estimate) gives
NaN bounds. Quantiles use numpy's default linear interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

CI_METHODS = ("percentile", "bca", "bootstrap_t")


@dataclass(frozen=True)
class SeedResampling:
    """Seed weights for the point estimate, R bootstrap resamples and the seed jackknife."""

    seeds: np.ndarray  # (S,) seed ids present at the look, ascending
    resamples: int
    weights: np.ndarray  # (1 + R + S, S)

    @property
    def point(self) -> np.ndarray:
        return self.weights[:1]

    @property
    def bootstrap(self) -> np.ndarray:
        return self.weights[1 : 1 + self.resamples]

    @property
    def jackknife(self) -> np.ndarray:
        return self.weights[1 + self.resamples :]

    def seed_index(self, run_seeds: np.ndarray) -> np.ndarray:
        """Column of each run's seed in ``weights``."""
        run_seeds = np.asarray(run_seeds)
        index = np.searchsorted(self.seeds, run_seeds)
        if not (index < len(self.seeds)).all() or not (self.seeds[index] == run_seeds).all():
            raise ValueError("A run's seed is not among the resampled seeds")
        return index

    def run_weights(self, run_seeds: np.ndarray) -> np.ndarray:
        """Case weights of runs, (rows, runs): every run gets its seed's weight."""
        return self.weights[:, self.seed_index(run_seeds)]


@dataclass(frozen=True)
class BootstrapDraws:
    """A statistic at the point estimate, its bootstrap replicates and its seed jackknife."""

    estimate: float
    se: float  # cluster-robust delta-method SE at the estimate (for bootstrap_t)
    replicates: np.ndarray  # (R,)
    replicate_se: np.ndarray  # (R,)
    jackknife: np.ndarray  # (S,)


def paired_resampling(run_seeds: np.ndarray, *, resamples: int, seed: int) -> SeedResampling:
    """Resample the seeds present among ``run_seeds`` with replacement, ``resamples`` times."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    seeds = np.unique(np.asarray(run_seeds))
    n = len(seeds)
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1.0 / n), size=resamples)
    weights = np.vstack([np.ones((1, n)), counts, 1.0 - np.eye(n)])
    return SeedResampling(seeds=seeds, resamples=resamples, weights=weights)


def confidence_intervals(draws: BootstrapDraws, method: str, levels: np.ndarray) -> np.ndarray:
    """Two-sided intervals at each level: (len(levels), 2) array of (low, high)."""
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    if not ((levels > 0) & (levels < 1)).all():
        raise ValueError("levels must lie in (0, 1)")
    alpha = (1.0 - levels) / 2
    tails = np.column_stack([alpha, 1.0 - alpha])
    if method == "percentile":
        return _quantiles(draws.replicates[np.isfinite(draws.replicates)], tails)
    if method == "bca":
        return _bca(draws, tails)
    if method == "bootstrap_t":
        return _studentized(draws, tails)
    raise ValueError(f"Unknown CI method {method!r}; expected one of {CI_METHODS}")


def confidence_interval(draws: BootstrapDraws, method: str, level: float) -> tuple[float, float]:
    low, high = confidence_intervals(draws, method, np.array([level]))[0]
    return float(low), float(high)


def _quantiles(values: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    if values.size == 0 or not np.isfinite(probabilities).all():
        return np.full(probabilities.shape, np.nan)
    return np.quantile(values, probabilities)


def _bca(draws: BootstrapDraws, tails: np.ndarray) -> np.ndarray:
    replicates = draws.replicates[np.isfinite(draws.replicates)]
    if replicates.size == 0:
        return np.full(tails.shape, np.nan)
    below = np.sum(replicates < draws.estimate) + 0.5 * np.sum(replicates == draws.estimate)
    with np.errstate(divide="ignore"):
        z0 = norm.ppf(below / replicates.size)
    spread = draws.jackknife.mean() - draws.jackknife
    scale = np.sum(spread**2) ** 1.5
    acceleration = np.sum(spread**3) / (6.0 * scale) if scale > 0 else 0.0
    z = z0 + norm.ppf(tails)
    denominator = 1.0 - acceleration * z
    with np.errstate(invalid="ignore", divide="ignore"):
        adjusted = np.where(denominator > 0, norm.cdf(z0 + z / denominator), np.nan)
    return _quantiles(replicates, adjusted)


def _studentized(draws: BootstrapDraws, tails: np.ndarray) -> np.ndarray:
    se = draws.replicate_se
    usable = np.isfinite(draws.replicates) & np.isfinite(se) & (se > 0)
    pivots = (draws.replicates[usable] - draws.estimate) / se[usable]
    q = _quantiles(pivots, tails)
    return draws.estimate - q[:, ::-1] * draws.se
