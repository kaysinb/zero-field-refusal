"""Study 5's training answers along the refusal direction (docs/study5-refdir-registration.md), as
pure functions of the answers' projections and the recorded harm. Nothing here reads a file:
``scripts/study5_refdir_analysis.py`` passes the numbers in.

**Answers.** Per model and seed, the stream's 500 answers in stream order: each answer's prompt id,
its projection p (mean over its answer positions of ⟨h, r̂⟩) and its token-weighted projection
p_tok (the sum over the same positions divided by ``TOKEN_DENOMINATOR`` = 96).

**Centring** (``loso_baselines``): b_{−s}(x) is the mean of p over the answers to prompt x in the
other seeds of the same model; an answer whose prompt has no answer elsewhere has no b and is left
out of the centred scores (listed).

**Scores.** P^c(s) = mean_i [p_i − b_{−s}(x_i)] (primary), P(s) = mean_i p_i (secondary). For an
order o, the batches are the order's permutation of the stream's answers cut into
``BATCH_SIZE`` = 20; W(s, o) = Σ_t c_t · mean_{i ∈ batch t} [p_i − b_{−s}(x_i)] with Adam's
position weights c_t (``linear_response.momentum_weights``); W_tok the same with p_tok. A batch
holding an answer without b takes the mean of its other answers.

**Empty answers.** An answer whose only response token is its end token has no answer position:
its mean is undefined (NaN in the projection file), it is listed, and it enters neither a score
nor any baseline; P^c and each batch mean are means over the answers with a defined, centred
value. The token-weighted variant sums over positions, so an empty answer's sum is 0 and it
counts, in its scores and its baselines (``token_rows``).

**Tests** (one-sided, Holm over T1–T4): T1 and T2, Spearman ρ between P^c and the mean h* over a
stream's three orders, predicted negative, permutation p; T3, the sign test over streams of
whether the order with the lower W has the higher h* (ties |Δh*| < ``TIE_PP`` or ΔW = 0 excluded);
T4, the mean over streams of Spearman ρ between W and h* across the stream's three orders,
predicted negative, its null by permuting the three h* within each stream.

**Undefined statistics.** A test whose inputs hold a non-finite value, or whose statistic is
undefined (a constant predictor or outcome, fewer than three streams, no untied pair), is
``unavailable``: it has no p, never a small one. In T4 a stream whose three W or three h* are all
equal has no ρ; it is left out of T4 and listed, and T4 is unavailable when no stream remains.
Holm keeps the family of four: an unavailable test counts as p = 1 and reads "unavailable".
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

BATCH_SIZE = 20
STEPS = 25
TOKEN_DENOMINATOR = 96
TIE_PP = 1e-9
SEED = 20261118
PERMUTATIONS = 10_000
WITHIN_DRAWS = 100_000
BOOTSTRAP = 10_000
ALPHA = 0.05


# ------------------------------------------------------------------ centring and scores


def loso_baselines(
    prompts: Mapping[int, Sequence[str]], values: Mapping[int, Sequence[float]]
) -> dict[int, list[float | None]]:
    """Per seed and answer, b_{−s}(x): the mean of the other seeds' defined (finite) values for the
    same prompt; None when there is none."""
    total: dict[str, float] = {}
    count: dict[str, int] = {}
    per_seed: dict[int, dict[str, tuple[float, int]]] = {}
    for s in prompts:
        own: dict[str, list[float]] = {}
        for x, v in zip(prompts[s], values[s], strict=True):
            own.setdefault(x, [])
            if math.isfinite(float(v)):
                own[x].append(float(v))
        per_seed[s] = {x: (math.fsum(vs), len(vs)) for x, vs in own.items()}
        for x, (t, n) in per_seed[s].items():
            total[x] = total.get(x, 0.0) + t
            count[x] = count.get(x, 0) + n
    out: dict[int, list[float | None]] = {}
    for s in prompts:
        row: list[float | None] = []
        for x in prompts[s]:
            t, n = per_seed[s][x]
            rest = count[x] - n
            row.append(None if rest == 0 else (total[x] - t) / rest)
        out[s] = row
    return out


def centred(values: Sequence[float], baselines: Sequence[float | None]) -> list[float | None]:
    """v − b; None for an undefined value (an empty answer) or a missing baseline."""
    return [None if b is None or not math.isfinite(float(v)) else float(v) - b
            for v, b in zip(values, baselines, strict=True)]  # fmt: skip


def _mean(xs: Sequence[float | None]) -> float | None:
    kept = [x for x in xs if x is not None]
    return math.fsum(kept) / len(kept) if kept else None


def stream_score(values: Sequence[float | None]) -> float | None:
    return _mean(values)


def order_score(
    values: Sequence[float | None], order: Sequence[int] | None, weights: Sequence[float]
) -> float | None:
    """Σ_t c_t · mean over batch t of the values, the batches cut from ``order`` (stream order
    when None)."""
    index = list(range(len(values))) if order is None else list(order)
    if len(index) != STEPS * BATCH_SIZE or sorted(index) != list(range(len(values))):
        raise ValueError("An order must permute the stream's 500 answers")
    total = 0.0
    for t in range(STEPS):
        m = _mean([values[i] for i in index[t * BATCH_SIZE : (t + 1) * BATCH_SIZE]])
        if m is None:
            return None
        total += weights[t] * m
    return total


# ------------------------------------------------------------------ tests


def finite(values: Sequence[float]) -> bool:
    return bool(np.all(np.isfinite(np.asarray(values, dtype=float))))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman ρ; NaN when undefined (a non-finite value, a constant side, under three points)."""
    xx, yy = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if xx.size < 3 or not finite(xx) or not finite(yy) or np.ptp(xx) == 0 or np.ptp(yy) == 0:
        return float("nan")
    return float(stats.spearmanr(xx, yy).statistic)


def unavailable(why: str, **fields: Any) -> dict[str, Any]:
    return {**fields, "p": None, "unavailable": why}


def permutation_p(
    x: Sequence[float], y: Sequence[float], *, draws: int = PERMUTATIONS, seed: int = SEED
) -> dict[str, Any]:
    """ρ(x, y) and the one-sided p of ρ ≤ observed under permutations of y; unavailable when ρ is
    undefined (module docstring)."""
    if not finite(x) or not finite(y):
        return unavailable("a non-finite value", n=len(x), rho=None)
    rho = spearman(x, y)
    if not math.isfinite(rho):
        return unavailable("rho undefined", n=len(x), rho=None)
    rng = np.random.default_rng(seed)
    yy = np.asarray(y, dtype=float)
    count = sum(spearman(x, rng.permutation(yy)) <= rho for _ in range(draws))
    return {"n": len(x), "rho": rho, "p": (1 + count) / (1 + draws), "draws": draws,
            "unavailable": None}  # fmt: skip


def sign_test(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Pairs (ΔW, Δh) of the same two orders: agreement when the lower W has the higher h, i.e.
    ΔW and Δh of opposite signs; ties excluded; the exact one-sided binomial p; unavailable with a
    non-finite value or no untied pair."""
    if not all(math.isfinite(w) and math.isfinite(h) for w, h in pairs):
        return unavailable("a non-finite value", n=0, ties=None, agreements=None)
    used = [(w, h) for w, h in pairs if w != 0 and abs(h) >= TIE_PP]
    agree = sum(1 for w, h in used if (w < 0) == (h > 0))
    n = len(used)
    if n == 0:
        return unavailable("no untied pair", n=0, ties=len(pairs), agreements=0)
    p = float(stats.binomtest(agree, n, 0.5, alternative="greater").pvalue)
    return {"n": n, "ties": len(pairs) - n, "agreements": agree, "p": p, "unavailable": None}


def within_stream_test(
    w: Mapping[int, Sequence[float]],
    h: Mapping[int, Sequence[float]],
    *,
    draws: int = WITHIN_DRAWS,
    seed: int = SEED,
) -> dict[str, Any]:
    """The mean over streams of ρ(W, h*) across the stream's orders, and its one-sided p under
    within-stream permutations of h*; a stream without ρ left out and listed; unavailable with a
    non-finite value or no stream left (module docstring)."""
    if not all(finite(w[s]) and finite(h[s]) for s in w):
        return unavailable("a non-finite value", n=0, statistic=None, left_out=[])
    left_out = sorted(s for s in w if not math.isfinite(spearman(w[s], h[s])))
    streams = sorted(s for s in w if s not in left_out)
    if not streams:
        return unavailable("no stream with a defined rho", n=0, statistic=None, left_out=left_out)
    obs = float(np.mean([spearman(w[s], h[s]) for s in streams]))
    # each stream's ρ under every permutation of its h*, then uniform draws of one per stream
    table = []
    for s in streams:
        hs = list(h[s])
        table.append(
            [
                spearman(w[s], [hs[i] for i in perm])
                for perm in itertools.permutations(range(len(hs)))
            ]
        )
    table_arr = np.asarray(table, float)  # (streams, permutations)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, table_arr.shape[1], (draws, len(streams)))
    null = table_arr[np.arange(len(streams)), pick].mean(axis=1)
    count = int(np.sum(null <= obs))
    return {"n": len(streams), "statistic": obs, "p": (1 + count) / (1 + draws), "draws": draws,
            "left_out": left_out, "unavailable": None}  # fmt: skip


FAMILY = ("T1", "T2", "T3", "T4")


def holm(ps: Mapping[str, float | None], alpha: float = ALPHA) -> dict[str, Any]:
    """Holm-adjusted p-values over the registered family of four and the reading of each test; an
    unavailable test (p None or not finite) counts as p = 1 and reads "unavailable"."""
    if sorted(ps) != sorted(FAMILY):
        raise ValueError(f"the family is {FAMILY}, got {sorted(ps)}")
    available = {k: p is not None and math.isfinite(p) for k, p in ps.items()}
    named = sorted((ps[k] if available[k] else 1.0, k) for k in FAMILY)
    m = len(named)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (p, k) in enumerate(named):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[k] = running
    out = {}
    for k in FAMILY:
        if not available[k]:
            out[k] = {"p": None, "adjusted": None, "reading": "unavailable"}
            continue
        reading = ("retrospectively associated with harm" if adjusted[k] < alpha
                   else "not established")  # fmt: skip
        out[k] = {"p": ps[k], "adjusted": adjusted[k], "reading": reading}
    return out


def bootstrap_rho(
    x: Sequence[float], y: Sequence[float], *, draws: int = BOOTSTRAP, seed: int = SEED
) -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    xx, yy = np.asarray(x, float), np.asarray(y, float)
    if not finite(xx) or not finite(yy):
        return None, None
    vals = []
    for _ in range(draws):
        i = rng.integers(0, len(xx), len(xx))
        rho = spearman(xx[i], yy[i])
        if math.isfinite(rho):
            vals.append(rho)
    if not vals:
        return None, None
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def cluster_agreement(
    rows: Sequence[tuple[int, float, float]], *, draws: int = BOOTSTRAP, seed: int = SEED
) -> dict[str, Any]:
    """Rows (stream, ΔW, Δh): the agreement share with a 95 % interval from resampling streams;
    ties excluded; no p; unavailable with a non-finite value."""
    if not all(math.isfinite(w) and math.isfinite(h) for _, w, h in rows):
        return {"n": 0, "ties": None, "agreements": None, "share": None, "lower": None,
                "upper": None, "unavailable": "a non-finite value"}  # fmt: skip
    used = [(s, w, h) for s, w, h in rows if w != 0 and abs(h) >= TIE_PP]
    streams = sorted({s for s, _, _ in used})
    by = {s: [(w, h) for t, w, h in used if t == s] for s in streams}
    agree = sum(1 for _, w, h in used if (w < 0) == (h > 0))
    rng = np.random.default_rng(seed)
    shares = []
    for _ in range(draws):
        pick = [streams[int(i)] for i in rng.integers(0, len(streams), len(streams))]
        pool = [x for s in pick for x in by[s]]
        if pool:
            shares.append(sum(1 for w, h in pool if (w < 0) == (h > 0)) / len(pool))
    lo, hi = np.percentile(shares, [2.5, 97.5]) if shares else (None, None)
    return {
        "n": len(used),
        "ties": len(rows) - len(used),
        "agreements": agree,
        "share": agree / len(used) if used else None,
        "lower": None if lo is None else float(lo),
        "upper": None if hi is None else float(hi),
        "unavailable": None if used else "no untied pair",
    }


def bootstrap_mean(
    values: Sequence[float], *, draws: int = BOOTSTRAP, seed: int = SEED
) -> tuple[float | None, float | None]:
    """A 95 % percentile interval of the mean, resampling the values (one per stream)."""
    v = np.asarray(values, float)
    if v.size == 0 or not finite(v):
        return None, None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (draws, v.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def agreement_shares(dw: np.ndarray, dh: Sequence[float]) -> np.ndarray:
    """Per direction (column of ``dw``, rows = pairs), the share of the untied pairs whose lower W
    has the higher h; a pair is tied when ΔW = 0 or |Δh| < ``TIE_PP``; NaN for a direction with a
    non-finite ΔW or no untied pair."""
    dh_arr = np.asarray(dh, float)[:, None]
    used = (dw != 0) & (np.abs(dh_arr) >= TIE_PP)
    agree = used & ((dw < 0) == (dh_arr > 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = agree.sum(axis=0) / used.sum(axis=0)
    bad = ~np.all(np.isfinite(dw), axis=0) | ~np.all(np.isfinite(dh_arr))
    share[bad] = np.nan
    return share


def partial_spearman(x: Sequence[float], y: Sequence[float], z: Sequence[float]) -> float:
    """Spearman ρ of x and y given z: the correlation of the rank residuals after regressing the
    ranks of x and of y on the ranks of z."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))

    def resid(a: np.ndarray) -> np.ndarray:
        beta = np.polyfit(rz, a, 1)
        return a - np.polyval(beta, rz)

    return float(np.corrcoef(resid(rx), resid(ry))[0, 1])


def percentile(value: float, reference: Sequence[float]) -> dict[str, Any]:
    """The share of the finite reference values at or below ``value``, in percent, with how many
    were finite; None when ``value`` is not finite or no reference value is."""
    ref = np.asarray(reference, float)
    ok = ref[np.isfinite(ref)]
    pct = (float(100.0 * np.mean(ok <= value))
           if ok.size and value is not None and math.isfinite(value) else None)  # fmt: skip
    return {"percentile": pct, "finite": int(ok.size), "of": int(ref.size)}


# ------------------------------------------------------------------ vectorized scores


def centred_vectors(
    prompts: Mapping[int, Sequence[str]], vectors: Mapping[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """Per seed, (answers, d) float64: each answer's vector minus the mean of the other seeds'
    vectors for the same prompt (leave one stream out); a row is NaN when its own vector is
    undefined (an empty answer: NaN, never in a baseline) or no other seed has a defined vector for
    that prompt. Linear in the vectors, so a projection of the result is the centred projection."""
    total: dict[str, np.ndarray] = {}
    count: dict[str, int] = {}
    own: dict[int, dict[str, tuple[np.ndarray, int]]] = {}
    for s in prompts:
        defined = np.all(np.isfinite(vectors[s]), axis=1)
        groups: dict[str, list[int]] = {}
        for i, x in enumerate(prompts[s]):
            groups.setdefault(str(x), [])
            if defined[i]:
                groups[str(x)].append(i)
        own[s] = {
            x: (vectors[s][idx].astype(np.float64).sum(axis=0), len(idx))
            for x, idx in groups.items()
        }
        for x, (t, n) in own[s].items():
            total[x] = total.get(x, 0.0) + t
            count[x] = count.get(x, 0) + n
    out = {}
    for s in prompts:
        rows = np.empty_like(vectors[s], dtype=np.float64)
        defined = np.all(np.isfinite(vectors[s]), axis=1)
        for i, x in enumerate(prompts[s]):
            t, n = own[s][str(x)]
            rest = count[str(x)] - n
            if rest == 0 or not defined[i]:
                rows[i] = np.nan
            else:
                rows[i] = vectors[s][i] - (total[str(x)] - t) / rest
        out[s] = rows
    return out


def token_rows(means: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """(answers, d): each answer's sum over its positions divided by ``TOKEN_DENOMINATOR``, i.e.
    its mean times its number of positions / 96; an empty answer (no position, mean NaN) is 0."""
    n = np.asarray(counts, dtype=np.float64)
    safe = np.where(n[:, None] > 0, np.nan_to_num(means, nan=0.0), 0.0)
    return safe * (n[:, None] / TOKEN_DENOMINATOR)


def batch_means(rows: np.ndarray, order: Sequence[int] | None) -> np.ndarray:
    """(steps, d): the mean of the non-NaN rows of each batch of the order."""
    index = np.arange(rows.shape[0]) if order is None else np.asarray(order)
    if index.shape[0] != STEPS * BATCH_SIZE or sorted(index.tolist()) != list(range(rows.shape[0])):
        raise ValueError("An order must permute the stream's 500 answers")
    return np.stack(
        [
            np.nanmean(rows[index[t * BATCH_SIZE : (t + 1) * BATCH_SIZE]], axis=0)
            for t in range(STEPS)
        ]
    )


def stream_scores(rows: Mapping[int, np.ndarray], units: np.ndarray) -> dict[int, np.ndarray]:
    """Per seed, (directions,): the mean over its non-NaN answers of each projection."""
    return {s: np.nanmean(r, axis=0) @ units for s, r in rows.items()}


def order_scores(
    rows: Mapping[int, np.ndarray],
    orders: Mapping[tuple[int, str], Sequence[int] | None],
    weights: Sequence[float],
    units: np.ndarray,
) -> dict[tuple[int, str], np.ndarray]:
    """Per (seed, order), (directions,): W = Σ_t c_t · batch mean projection."""
    c = np.asarray(weights, float)
    return {(s, o): c @ (batch_means(rows[s], order) @ units) for (s, o), order in orders.items()}
