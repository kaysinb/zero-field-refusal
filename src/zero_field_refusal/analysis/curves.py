"""Curves h(K): binomial logistic regression of harmful counts on log K.

logit h = a + b·log K is fitted to per-run counts (``harmful`` successes out of ``judged`` trials)
by Newton's method (IRLS) with step-halving. The fit is vectorized over the rows of a case-weight
matrix (rows × runs), so the point estimate and all bootstrap and jackknife replicates are fitted
at once. Weights act as frequency weights: a weight of 2 counts a run twice.

Separation. The unpenalized maximum-likelihood estimate does not exist, or is not unique, when the
counts allow a perfect fit along some direction: all runs harmless, all fully harmful, a K
threshold that splits harmless runs from fully harmful ones, or a single K value. ``separated``
flags such rows exactly, from the data rather than from the iterations. A ridge penalty
``RIDGE·b²/2`` on the slope keeps it finite under separation along K; on well-posed data it moves
the slope by about ``RIDGE·b/I_bb``, with I_bb ≳ 1e2 the information for b, far below 1e-6. When
all counts are 0 (or all equal ``judged``) the intercept diverges instead: the fit stops after
``MAX_ITER`` iterations with h numerically 0 (or 1), the limit of the maximum-likelihood fit, and
``converged`` is False.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logit

RIDGE = 1e-6
MAX_ITER = 100
TOL = 1e-9  # on the Newton step of the centered intercept and of the slope
MAX_HALVINGS = 30


@dataclass(frozen=True)
class LogisticFit:
    """logit h = intercept + slope·log K, one fit per weight row.

    Rows whose weights are all zero have NaN coefficients.
    """

    intercept: np.ndarray  # (rows,)
    slope: np.ndarray  # (rows,)
    converged: np.ndarray  # (rows,) bool
    separated: np.ndarray  # (rows,) bool: no unique finite unpenalized estimate
    ridge: float

    def predict(self, log_k: np.ndarray | float) -> np.ndarray:
        """h at each log K: (rows, len(log_k))."""
        log_k = np.atleast_1d(np.asarray(log_k, dtype=float))
        return expit(self.intercept[:, None] + self.slope[:, None] * log_k)


def fit_logistic(
    log_k: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    ridge: float = RIDGE,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
) -> LogisticFit:
    """Fit logit h = a + b·log K for each row of ``weights`` (rows × runs; None: one row of 1s)."""
    x, y, m = _arrays(log_k, successes, trials)
    w = _weight_rows(weights, len(x))
    rows = len(w)
    # Centering decorrelates intercept and slope; the slope and its penalty are unchanged.
    center = float(x.mean())
    xc = x - center
    wy, wm = w * y, w * m
    total = wm.sum(axis=1)
    filled = total > 0

    alpha = np.full(rows, np.nan)  # centered intercept
    beta = np.full(rows, np.nan)
    alpha[filled] = logit(np.clip(wy[filled].sum(axis=1) / total[filled], 1e-4, 1 - 1e-4))
    beta[filled] = 0.0
    converged = np.zeros(rows, dtype=bool)
    active = np.flatnonzero(filled)
    objective = np.full(rows, np.nan)
    objective[active] = _penalized_loglik(
        alpha[active], beta[active], xc, wy[active], wm[active], ridge
    )

    for _ in range(max_iter):
        if active.size == 0:
            break
        a, b, wy_a, wm_a = alpha[active], beta[active], wy[active], wm[active]
        p = expit(a[:, None] + b[:, None] * xc)
        residual = wy_a - wm_a * p
        g0 = residual.sum(axis=1)
        g1 = residual @ xc - ridge * b
        v = wm_a * p * (1 - p)
        h00, h01, h11 = v.sum(axis=1), v @ xc, v @ xc**2 + ridge
        det = h00 * h11 - h01**2
        ok = np.isfinite(det) & (det > 0)
        safe = np.where(ok, det, 1.0)
        step_a = np.where(ok, (h11 * g0 - h01 * g1) / safe, 0.0)
        step_b = np.where(ok, (h00 * g1 - h01 * g0) / safe, 0.0)

        old = objective[active]
        t = np.ones(active.size)
        for _ in range(MAX_HALVINGS):
            new_a, new_b = a + t * step_a, b + t * step_b
            new = _penalized_loglik(new_a, new_b, xc, wy_a, wm_a, ridge)
            worse = new < old - 1e-12 * np.abs(old)
            if not worse.any():
                break
            t = np.where(worse, t / 2, t)
        alpha[active], beta[active], objective[active] = new_a, new_b, new

        done = ok & (np.maximum(np.abs(step_a), np.abs(step_b)) < tol)
        converged[active[done]] = True
        active = active[~done & ok]

    return LogisticFit(
        intercept=alpha - beta * center,
        slope=beta,
        converged=converged,
        separated=_separated(x, y, m, w) & filled,
        ridge=ridge,
    )


def cluster_influence(
    fit: LogisticFit,
    log_k: np.ndarray,
    successes: np.ndarray,
    trials: np.ndarray,
    weights: np.ndarray | None,
    clusters: np.ndarray,
    n_clusters: int,
    at: np.ndarray,
) -> np.ndarray:
    """Influence of one copy of each cluster on h(at), per weight row: (rows, n_clusters, len(at)).

    phi[r, s, l] = ∇h_r(at_l)ᵀ J_r⁻¹ u_rs, with J_r the penalized information of row r and u_rs
    the score of the runs of cluster s at row r's estimate. Σ_s w_rs·phi[r, s, l]² is then the
    cluster-robust (sandwich) delta-method variance of h_r(at_l), without a small-sample
    correction, where w_rs is the weight that row r gives the runs of cluster s.
    """
    x, y, m = _arrays(log_k, successes, trials)
    w = _weight_rows(weights, len(x))
    at = np.atleast_1d(np.asarray(at, dtype=float))
    p = fit.predict(x)
    v = w * m * p * (1 - p)
    h00, h01, h11 = v.sum(axis=1), v @ x, v @ x**2 + fit.ridge
    det = h00 * h11 - h01**2

    h_at = fit.predict(at)
    slope_at = h_at * (1 - h_at)  # dh/dη: (rows, L); ∇h = dh/dη · (1, at)
    c0 = slope_at * (h11[:, None] - h01[:, None] * at) / det[:, None]
    c1 = slope_at * (h00[:, None] * at - h01[:, None]) / det[:, None]

    membership = np.zeros((len(x), n_clusters))
    membership[np.arange(len(x)), clusters] = 1.0
    residual = y - m * p  # one copy of each run
    u0 = residual @ membership
    u1 = (residual * x) @ membership
    return c0[:, None, :] * u0[:, :, None] + c1[:, None, :] * u1[:, :, None]


def _arrays(log_k, successes, trials) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(log_k, dtype=float)
    y = np.asarray(successes, dtype=float)
    m = np.asarray(trials, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or x.shape != m.shape or x.size == 0:
        raise ValueError("log_k, successes and trials must be non-empty 1-D arrays of one length")
    return x, y, m


def _weight_rows(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones((1, n))
    w = np.atleast_2d(np.asarray(weights, dtype=float))
    if w.ndim != 2 or w.shape[1] != n:
        raise ValueError(f"weights must have shape (rows, {n}), got {w.shape}")
    if not (np.isfinite(w).all() and (w >= 0).all()):
        raise ValueError("weights must be finite and non-negative")
    return w


def _penalized_loglik(a, b, xc, wy, wm, ridge) -> np.ndarray:
    eta = a[:, None] + b[:, None] * xc
    return (wy * eta - wm * np.logaddexp(0.0, eta)).sum(axis=1) - 0.5 * ridge * b**2


def _separated(x, y, m, w) -> np.ndarray:
    """Rows whose weighted runs admit a perfect fit (no unique finite estimate)."""
    used = w > 0
    harmful = used & (y > 0)
    harmless = used & (y < m)
    # η → +∞ above and −∞ below a threshold c: harmless labels only at x ≤ c, harmful at x ≥ c.
    rising = np.where(harmless, x, -np.inf).max(axis=1) <= np.where(harmful, x, np.inf).min(axis=1)
    falling = np.where(harmful, x, -np.inf).max(axis=1) <= np.where(harmless, x, np.inf).min(axis=1)
    return rising | falling
