"""Study 5's extension 2 as pure functions of its records (docs/study5-preregistration.md,
"Extension 2 (2026-11-11)" and "Extension 2 amendment"): every quantity of M1–M6 per seed and
point, the numerical floors and the δ_lin validation criterion. Nothing here reads a file, a
response or a label: ``analysis.study5_ext2_cli`` reads pod 4's records, the run directories and
the registered analysis record and passes their numbers in. Nothing here imports torch.

**Values.** A per-seed value is a number or None; where it is None its row says why
(``reason``). ``spread`` gives the mean over the seeds where a quantity exists with the
seed-level standard error, sd (n − 1 degrees of freedom) over √n, and names every seed it left
out; no SE under two seeds.

**The δ_lin criterion** (``criterion``), fixed in the registration and implemented as fixed, over
the seeds where δ_lin did not diverge:

1. Spearman ρ(q, Δr) > 0 with one-sided p < 0.05. The primary p permutes q against Δr:
   ``numpy.random.default_rng(20261112)``, 100 000 permutations of q's ranks, p = (1 +
   #{ρ_perm ≥ ρ_obs}) / (1 + 100 000); a permuted ρ counts as at or above the observed one
   within 1e-12 (the registered analysis's ruling on ties, ``study5_analysis.TIE``), so that a
   permutation equal to the observation in exact arithmetic counts however its sum is rounded.
   scipy's ``spearmanr(q, Δr, alternative="greater")`` p stands beside it and decides nothing.
2. 0.5 ≤ mean(q) / mean(Δr) ≤ 2 over the same seeds; it fails when mean(Δr) is 0 or the ratio
   is not finite.
3. On the six order-control pairs (seeds 50000–50005, the registered control's order against
   the stream order, both at η_l): a pair agrees when sign(q_control − q_stream) =
   −sign(h_control − h_stream), both differences nonzero (lower q, more harm); a tie or a pair
   whose δ_lin diverged (either order) does not agree. It passes with at least 5 of 6. h is the
   registered per-run harm share at η_l (the registered analysis's
   ``descriptive.controls.order.per_seed.harmful``, control minus own η_l run).

``passed`` is all three, ``primary_predictor`` ``q`` when it passed and ``ln_K_harmful_at_K*``
otherwise. q is the δ_lin's (``q_lin``). An input the criterion needs that is missing (a seed's
linear record or Δr, a control's linear record, a pair's registered harm difference) withholds
the verdict: ``status`` ``withheld``, every part's numbers over what exists, ``passed`` and
``primary_predictor`` None, each missing input named.

**Reading rules** where the registration leaves a detail open:

- *r(θ₀).* θ₀ is common to every seed. Δr_i = r(θ₂₅,i) − r(θ₀) takes r(θ₂₅,i) from pod 2's
  surrogate endpoint record and r(θ₀) from the seed's own linear-replay record (pod 4, the q's
  baseline); M3's even parts take r(θ₀) from the seed's token record (the same panel code path
  as the small amplitudes' readout), the linear record's when it is absent; the twins take the
  pairs record's own r(θ₀). Every value of r(θ₀) read is reported with their spread.
- *M2's ε check:* per check step the largest of the two relative differences (ε/2 and 2ε
  against ε), and the largest over the check steps.
- *Near the divergence bound* (the amendment's "numerically ambiguous"): ``bound_ratio`` is the
  largest finite ‖δ_lin,t‖ over the bound 10‖Δθ_i‖; a case is ambiguous when |bound_ratio − 1|
  ≤ 10 × the linear repeat's relative δ_lin difference (‖δ_lin − δ′_lin‖/‖δ_lin‖ on seed 50000),
  the amendment's one order of magnitude; no floor, no flag. It is reported, never resolved.
- *Floors* (the amendment): beside a quantity with a floor, ratio = |effect| / floor and the flag
  ``resolved`` when the ratio is at least 10, else ``numerically_unresolved``; a zero floor
  resolves any nonzero effect (ratio None, ``floor_zero``). The linear repeat (seed 50000) floors
  q, K and the cosines with Δθ of every seed and order, and the pair and edit differences of q;
  and the relative difference ‖δ_lin − δ̂‖/‖δ̂‖ (by the repeat's ‖δ_lin − δ′_lin‖/‖δ_lin‖);
  the force repeat (panel 59990, step 5, ε = 10⁻²) floors every projection and ‖D f̂_t‖ at every
  step, ε and panel, and the panels' differences; the twins' floor (seed 50000) floors the
  distance and the r difference of every seed's twins at the same step. The criterion's verdict
  does not read the flags: they stand beside it.
- *M3 locality* (the analysis's reading rule): the odd part [r(a) − r(−a)]/2 is local where
  odd(a)/odd(a/2) lies in [1.5, 2.5] (it halves with a), the even part [r(a) + r(−a)]/2 − r(θ₀)
  where even(a)/even(a/2) lies in [3, 5] (it quarters); a ratio with a zero denominator is None,
  not local.
- *M5 agreement:* the panels' agreement per projection and for ‖D f̂_t‖ is their difference and
  its size relative to the mean of the two magnitudes; the halves' is ‖D(f̂^A − f̂^B)‖/‖D f̂_t‖
  and the halves' projection difference; ε-convergence is each value's relative difference
  against ε = 10⁻², the registered central step.
- *M6's excess over the prediction:* where the outcome is in q's units (r at the endpoint: the
  twins, and the stream orders' Δr), the excess is Δr − q of the run's own q, for δ̂ and δ_lin.
  The swaps and shuffles have no endpoint r: their outcomes (K, harm, utility) and their
  differences from the stream order are reported beside the predicted difference of q, with no
  fit.
- *Harm* is the registered reading: per condition the WildGuard counts of the registered label
  reader; harmful compliance over 521, a missing label not harmful, with ``harmful_high`` (every
  missing label harmful) beside.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

import numpy as np
from scipy import stats

from . import study5_analysis as sa

KIND, VERSION = "study5-extension-2-analysis", 1
ETA_L = 1.3e-05
SEEDS = tuple(range(50000, 50040))  # the 40 fresh seeds
M_SEEDS = tuple(range(50000, 50006))  # M3, M4's removals, δ_lin in full, the shuffles, the pairs
SWAP_SEEDS = M_SEEDS[:4]  # M6's swaps and twins
FORCE_SEED = 50000  # M5's seed and the amendment's floors
GROUP = 4  # pod 4's replay groups
E2_SEEDS = tuple(range(50006, 50016))  # extension 1's order repeats (sealed)
E2_OFFSETS = (1000, 2000)  # train.order_seed = s + offset
E2_NAME = re.compile(r"(?:^|-)s5-order[12]-")  # an E2 cell's name inside a run id
CHECK_STEPS = (5, 15, 25)
SMALL_AMPLITUDES = (0.1, 0.05, 0.025)  # M3's ±a
TOKEN_AMPLITUDES = (0.1, 1.0)  # M3's token points ±a
ODD_LOCAL, EVEN_LOCAL = (1.5, 2.5), (3.0, 5.0)  # the reading rule's bounds (module docstring)
TWIN_STEPS = (2, 3, 5, 8, 12, 18, 25)
FORCE_STEPS = (5, 10)
FORCE_EPSILONS = (5e-2, 1e-2, 5e-3)
CENTRAL_EPSILON = 1e-2
FORCE_REPEAT = {"panel": 59990, "step": 5, "epsilon": 1e-2}
SWAP_STEPS = (2, 12, 23)
SHUFFLES = ((1, 5), (21, 25))
DRAWS = (0, 1)
MIXTURES = (1, 2, 3)
RESOLVED_RATIO = 10.0  # the amendment's one order of magnitude
UTILITY = ("benchmark_accuracy", "factual_accuracy", "heldout_nll_per_token")
MEASURES = ("K_benign", "K_harmful", *UTILITY)
CRITERION = {"permutations": 100_000, "rng_seed": 20261112, "alpha": 0.05,
             "ratio_bounds": (0.5, 2.0), "pairs": len(M_SEEDS), "agreements_needed": 5}  # fmt: skip
PRIMARY = {True: "q", False: "ln_K_harmful_at_K*"}
P_FORMULA = "(1 + #{rho_perm >= rho_obs}) / (1 + 100000)"


# ------------------------------------------------------------------ values and spreads


def number(value: Any) -> float | None:
    """A finite real number as a float, else None (booleans are not numbers)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    found = float(value)
    return found if math.isfinite(found) else None


def difference(a: Any, b: Any) -> float | None:
    x, y = number(a), number(b)
    return None if x is None or y is None else x - y


def ratio(a: Any, b: Any) -> float | None:
    x, y = number(a), number(b)
    return None if x is None or y is None or y == 0 else number(x / y)


def spread(values: Mapping[Any, Any]) -> dict[str, Any]:
    """The mean over the keys with a value, sd (n − 1) and SE = sd/√n (none under two), the keys
    used and the keys left out."""
    used = [k for k in values if number(values[k]) is not None]
    data = [number(values[k]) for k in used]
    out: dict[str, Any] = {"n": len(data), "mean": statistics.fmean(data) if data else None,
                           "sd": None, "se": None, "used": used,
                           "without": [k for k in values if k not in used]}  # fmt: skip
    if len(data) >= 2:
        sd = statistics.stdev(data)
        out |= {"sd": sd, "se": sd / math.sqrt(len(data))}
    return out


def spreads(rows: Mapping[Any, Mapping[str, Any]], keys: Iterable[str]) -> dict[str, Any]:
    """``spread`` of each key over the rows (a missing key or row value: no value)."""
    return {key: spread({k: (row or {}).get(key) for k, row in rows.items()}) for key in keys}


def rank_rho(x: Mapping[Any, Any], y: Mapping[Any, Any]) -> dict[str, Any]:
    """Spearman ρ (``study5_analysis.spearman``) over the keys with both values (descriptive)."""
    used = [k for k in x if number(x[k]) is not None and number(y.get(k)) is not None]
    rho = sa.spearman([x[k] for k in used], [y[k] for k in used]) if len(used) >= 2 else math.nan
    return {"n": len(used), "rho": number(rho), "used": used}


def resolution(effect: Any, floor: Any) -> dict[str, Any]:
    """|effect| / floor and its flag (module docstring, "Floors")."""
    e, f = number(effect), number(floor)
    if e is None or f is None:
        why = "no effect" if e is None else "no floor"
        return {"ratio": None, "flag": None, "reason": why}
    e, f = abs(e), abs(f)
    if f == 0:
        return {"ratio": None, "floor_zero": True,
                "flag": "resolved" if e > 0 else "numerically_unresolved"}  # fmt: skip
    found = e / f
    return {"ratio": found, "flag": "resolved" if found >= RESOLVED_RATIO
            else "numerically_unresolved"}  # fmt: skip


def sign(x: float) -> int:
    return (x > 0) - (x < 0)


# ------------------------------------------------------------------ the sealed orders


def e2_repeat(config: Mapping[str, Any] | None, name: str) -> str | None:
    """Why a run directory is one of extension 1's order repeats (E2), or None: its name holds an
    E2 cell (``s5-order1-``, ``s5-order2-``), or its resolved config has an E2 seed with
    ``train.order_seed`` = seed + 1000 or + 2000, or that order seed itself."""
    if E2_NAME.search(name):
        return f"{name}: an extension-1 order repeat (E2) by its name"
    if not isinstance(config, Mapping):
        return None
    seed = config.get("seed")
    train = config.get("train") if isinstance(config.get("train"), Mapping) else {}
    order = train.get("order_seed")
    e2_orders = {s + o for s in E2_SEEDS for o in E2_OFFSETS}
    if isinstance(order, int) and not isinstance(order, bool) and (
            order in e2_orders or (seed in E2_SEEDS and order - seed in E2_OFFSETS)):  # fmt: skip
        return f"{name}: an extension-1 order repeat (E2), order seed {order}"
    return None


# ------------------------------------------------------------------ the criterion


def permutation_greater(q: Sequence[float], dr: Sequence[float], *, permutations: int,
                        rng_seed: int) -> dict[str, Any]:  # fmt: skip
    """ρ(q, Δr) and its one-sided permutation p for a positive association: q's average ranks
    permuted by ``numpy.random.default_rng(rng_seed)``, ``permutations`` times, against Δr's;
    p = (1 + #{ρ_perm ≥ ρ_obs}) / (1 + permutations), ties within ``study5_analysis.TIE``.
    ``ValueError`` when a side is constant."""
    rq = stats.rankdata(np.asarray(q, dtype=float), method="average")
    rd = stats.rankdata(np.asarray(dr, dtype=float), method="average")
    rq, rd = rq - rq.mean(), rd - rd.mean()
    denominator = math.sqrt(float(rq @ rq) * float(rd @ rd))
    if not denominator > 0:
        raise ValueError("ρ is undefined: a side is constant")
    observed = float(rd @ rq) / denominator
    rng = np.random.default_rng(rng_seed)
    count = 0
    for _ in range(int(permutations)):
        if float(rd @ rng.permutation(rq)) / denominator >= observed - sa.TIE:
            count += 1
    return {"rho": observed, "count": count, "p_value": (1 + count) / (1 + int(permutations))}


def criterion(own: Mapping[int, Mapping[str, Any]], pairs: Mapping[int, Mapping[str, Any]], *,
              seeds: Sequence[int] = SEEDS, pair_seeds: Sequence[int] = M_SEEDS,
              permutations: int = CRITERION["permutations"],
              rng_seed: int = CRITERION["rng_seed"]) -> dict[str, Any]:  # fmt: skip
    """The δ_lin validation criterion (module docstring).

    ``own[seed]``: ``q_lin``, ``diverged`` (bool), ``delta_r`` and ``missing`` (a reason that
    withholds the verdict, or None). ``pairs[seed]``: ``q_control``, ``q_stream``,
    ``diverged`` (either order's δ_lin), ``h_difference`` (control − stream) and ``missing``."""
    withheld = []
    for s in seeds:
        found = own.get(s)
        if found is None or found.get("missing"):
            withheld.append(f"s{s}: {found.get('missing') if found else 'no row'}")
    used = [s for s in seeds if s in own and not own[s].get("missing") and not own[s]["diverged"]]
    diverged = [s for s in seeds if s in own and not own[s].get("missing") and own[s]["diverged"]]
    q = [float(own[s]["q_lin"]) for s in used]
    dr = [float(own[s]["delta_r"]) for s in used]
    alpha, (low, high) = CRITERION["alpha"], CRITERION["ratio_bounds"]

    first: dict[str, Any] = {"n": len(used), "rho": None, "permutations": int(permutations),
                             "rng_seed": int(rng_seed), "count": None, "p_value": None,
                             "p_formula": P_FORMULA, "alpha": alpha, "scipy_p_greater": None,
                             "passed": False, "reason": None}  # fmt: skip
    try:
        if len(used) < 3:
            raise ValueError(f"{len(used)} seeds: at least 3 are needed")
        found = permutation_greater(q, dr, permutations=permutations, rng_seed=rng_seed)
        first |= {"rho": found["rho"], "count": found["count"], "p_value": found["p_value"],
                  "passed": found["rho"] > 0 and found["p_value"] < alpha}  # fmt: skip
        first["scipy_p_greater"] = number(stats.spearmanr(q, dr, alternative="greater").pvalue)
    except ValueError as exc:
        first["reason"] = str(exc)

    mean_q = statistics.fmean(q) if q else None
    mean_dr = statistics.fmean(dr) if dr else None
    value = ratio(mean_q, mean_dr)
    second = {"n": len(used), "mean_q": mean_q, "mean_delta_r": mean_dr, "ratio": value,
              "bounds": [low, high], "passed": value is not None and low <= value <= high,
              "reason": None if value is not None else
              "mean Δr is 0 or the ratio is not finite" if used else "no seed"}  # fmt: skip

    rows, agreements = [], 0
    for s in pair_seeds:
        found = pairs.get(s)
        if found is None or found.get("missing"):
            withheld.append(f"pair s{s}: {found.get('missing') if found else 'no row'}")
            rows.append(
                {
                    "seed": s,
                    "agrees": False,
                    "reason": f"missing: {found.get('missing') if found else 'no row'}",
                }
            )
            continue
        dq = difference(found.get("q_control"), found.get("q_stream"))
        dh = number(found.get("h_difference"))
        row = {"seed": s, "q_control": found.get("q_control"), "q_stream": found.get("q_stream"),
               "q_difference": dq, "h_difference": dh, "agrees": False,
               "reason": None}  # fmt: skip
        if found.get("diverged"):
            row["reason"] = "δ_lin diverged"
        elif dq is None or dh is None:
            row["reason"] = "a difference is not finite"
        elif dq == 0 or dh == 0:
            row["reason"] = "a tie"
        else:
            row["agrees"] = sign(dq) == -sign(dh)
            agreements += row["agrees"]
        rows.append(row)
    third = {"pairs": rows, "agreements": agreements,
             "needed": CRITERION["agreements_needed"], "of": len(pair_seeds),
             "passed": agreements >= CRITERION["agreements_needed"]}  # fmt: skip

    out: dict[str, Any] = {
        "q": "q_lin",
        "seeds": used,
        "diverged": diverged,
        "part1_rank": first,
        "part2_scale": second,
        "part3_pairs": third,
    }
    if withheld:
        for part in (first, second, third):
            part["verdict_withheld"] = True
            part["passed"] = None
        return out | {"status": "withheld", "withheld": withheld, "passed": None,
                      "primary_predictor": None}  # fmt: skip
    passed = bool(first["passed"] and second["passed"] and third["passed"])
    return out | {"status": "decided", "withheld": [], "passed": passed,
                  "primary_predictor": PRIMARY[passed]}  # fmt: skip


# ------------------------------------------------------------------ harm shares


def harm(counts: Mapping[str, Any] | str | None) -> dict[str, Any]:
    """A condition's classes over 521 by the registered reading (``study5_analysis.class_shares``:
    a missing label in no class) and ``harmful_high``, every missing label harmful; or the
    reason there are none."""
    if not isinstance(counts, Mapping):
        return {"available": False, "reason": str(counts) if counts else "no labels read"}
    shares = sa.class_shares(counts)
    return {"available": True, **shares,
            "harmful_high": (counts["harmful"] + counts["missing"]) / counts["n"],
            "missing_labels": counts["missing"], "n": counts["n"]}  # fmt: skip


# ------------------------------------------------------------------ pairs records


def pair_row(record: Mapping[str, Any] | str, index: int = 0, *, a_sha256: str | None = None,
             b_run: str | None = None, b_sha256: str | None = None) -> dict[str, Any]:  # fmt: skip
    """One row of a ``study5-pairs`` record: its dot, norms, cosine, distance and the norm ratio
    ‖a‖/‖b‖, after its items are checked against the expected vector (``a_sha256``) and run
    (``b_run``) or vector (``b_sha256``); or why it is not available."""
    if isinstance(record, str):
        return {"available": False, "reason": record}
    rows = record.get("pairs") if record.get("kind") == "study5-pairs" else None
    if not isinstance(rows, list) or len(rows) <= index:
        return {"available": False, "reason": "not a study5-pairs record with that row"}
    row = rows[index]
    a, b = row.get("a") or {}, row.get("b") or {}
    if a_sha256 is not None and a.get("sha256") != a_sha256:
        return {"available": False, "reason": "its first item is not the expected vector"}
    if b_run is not None and b.get("run") != b_run:
        return {"available": False, "reason": "its second item is not the expected run"}
    if b_sha256 is not None and b.get("sha256") != b_sha256:
        return {"available": False, "reason": "its second item is not the expected vector"}
    return {"available": True, "dot": number(row.get("dot")), "norm_a": number(row.get("norm_a")),
            "norm_b": number(row.get("norm_b")), "cosine": number(row.get("cosine")),
            "distance": number(row.get("distance")),
            "norm_ratio": ratio(row.get("norm_a"), row.get("norm_b"))}  # fmt: skip


def item_name(item: Mapping[str, Any]) -> str:
    return PurePosixPath(str(item.get("path", ""))).name


# ------------------------------------------------------------------ M1


def m1(vectors: Mapping[str, Any], h: Mapping[int, Any], delta_r: Mapping[int, Any],
       checks: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """M1 from the vectors record (``study5_vectors.py``): the Gram's summary, ‖μ‖² free of the
    noise, ‖μ̂‖, the halves' cosine, p_i with its coefficient of variation, ρ(p_i, h_i) over the
    supported seeds (``h``: the registered analysis's), ⟨Δθ_i, ∇r(θ₀)⟩ against Δr_i, and the
    replay checks (``checks``: the groups' summaries, ``replay_checks``)."""
    seeds = [int(s) for s in vectors["seeds"]]
    loo = vectors["leave_one_out"]
    p = {s: number(v) for s, v in zip(loo["seeds"], loo["p"], strict=True)}
    runs = {int(r["seed"]): r for r in vectors["runs"]}
    per_seed = {}
    grad = vectors.get("grad_r")
    dots = dict(zip(seeds, grad["per_run"], strict=True)) if grad else {}
    for j, s in enumerate(seeds):
        run = runs.get(s, {})
        per_seed[s] = {
            "run_id": run.get("run_id"),
            "delta_theta_norm": number(vectors["norms"][j]),
            "displacement_norm_recorded": number(run.get("displacement_norm_recorded")),
            "p": p.get(s), "cos_with_loo_mean": number(loo["cos"][j]),
            "loo_mean_norm": number(loo["norm"][j]),
            "h": number(h.get(s)),
            "dot_grad_r": number(dots.get(s)), "delta_r": number(delta_r.get(s)),
            "delta_r_minus_dot": difference(delta_r.get(s), dots.get(s)),
        }  # fmt: skip
    gram = np.asarray(vectors["gram"], dtype=float)
    off = gram[~np.eye(len(seeds), dtype=bool)] if len(seeds) > 1 else np.array([])
    return {
        "n": vectors["n"], "seeds": seeds, "skipped": vectors.get("skipped", []),
        "gram": {"diagonal_mean": float(np.diag(gram).mean()),
                 "off_diagonal_mean": float(off.mean()) if off.size else None,
                 "off_diagonal_min": float(off.min()) if off.size else None,
                 "off_diagonal_max": float(off.max()) if off.size else None},
        "mean": dict(vectors["mean"]),
        "norm_sq_noise_free": vectors["mean"]["norm_sq_noise_free"],
        "mu_hat_norm": vectors["mean"]["norm"],
        "halves": vectors["halves"],
        "split_half_cosine": vectors["halves"]["cosine"],
        "p": {"spread": loo["p_spread"], "cv": loo["p_spread"].get("cv"),
              "cos_spread": loo["cos_spread"]},
        "rho_p_h": rank_rho(p, {s: h.get(s) for s in p}),
        "grad_r": None if not grad else {
            "sha256": grad.get("sha256"), "norm": grad.get("norm"),
            "mean_vector_dot": grad.get("mean_vector"),
            "recorded": {k: grad.get(k) for k in ("mean", "sd", "se", "cv")}},
        "dot_against_delta_r": {
            "dot_grad_r": spread({s: r["dot_grad_r"] for s, r in per_seed.items()}),
            "delta_r": spread({s: r["delta_r"] for s, r in per_seed.items()}),
            "delta_r_minus_dot": spread({s: r["delta_r_minus_dot"] for s, r in per_seed.items()}),
            "rho": rank_rho({s: r["dot_grad_r"] for s, r in per_seed.items()},
                            {s: r["delta_r"] for s, r in per_seed.items()}),
        },
        "removals": vectors.get("removals", []),
        "mixtures": [{k: m.get(k) for k in ("k", "scale", "unscaled_norm", "target_norm")}
                     for m in vectors.get("mixtures", [])],
        "per_seed": {str(s): row for s, row in per_seed.items()},
        "replay_checks": checks,
    }  # fmt: skip


def check_summary(record: Mapping[str, Any] | str) -> dict[str, Any]:
    """One ``study5_engineering_check.py`` record: passed, exact, within tolerance, the answers
    differing and, per pair, its endpoint's tensors equal or their distance ratio."""
    if isinstance(record, str):
        return {"available": False, "reason": record}
    pairs = []
    for pair in record.get("pairs", []):
        tensors = pair.get("endpoint_tensors") or {}
        distance = tensors.get("distance") or {}
        pairs.append({"rerun": pair.get("rerun"), "config_hash": pair.get("config_hash"),
                      "exact": pair.get("exact"), "within_tolerance": pair.get("within_tolerance"),
                      "passed": pair.get("passed"),
                      "answers_differing": pair.get("answers_differing"),
                      "differing_files": sum(not f.get("equal") for f in pair.get("files", [])),
                      "endpoint_tensors_equal": tensors.get("equal"),
                      "endpoint_distance_ratio": number(distance.get("ratio"))})  # fmt: skip
    ratios = [p["endpoint_distance_ratio"] for p in pairs
              if p["endpoint_distance_ratio"] is not None]  # fmt: skip
    return {
        "available": True,
        "passed": record.get("passed"),
        "exact": record.get("exact"),
        "within_tolerance": record.get("within_tolerance"),
        "answers_differing": record.get("answers_differing"),
        "tolerance": record.get("tolerance"),
        "unpaired": record.get("unpaired"),
        "pairs": pairs,
        "largest_endpoint_distance_ratio": max(ratios) if ratios else None,
    }


# ------------------------------------------------------------------ M2


def linear_row(
    record: Mapping[str, Any] | str,
    *,
    pairs_hat: Mapping[str, Any] | str | None = None,
    pairs_lin: Mapping[str, Any] | str | None = None,
    replay_run: str | None = None,
    delta_r: float | None = None,
    lin_floor: float | None = None,
) -> dict[str, Any]:
    """One linear-replay record (``study5_linear_replay.py``) of one order: δ̂'s and δ_lin's q, r,
    K and norms, their cosines with Δθ_i and norm ratios from the pairs records (each checked to
    be of this record's vector and of the seed's replay), the ε checks, the divergence status
    and its distance from the bound (module docstring)."""
    if isinstance(record, str):
        return {"available": False, "reason": record}
    k = record.get("K")  # None: written with --no-k (predictions of q only)
    divergence = record.get("divergence") or {}
    vectors = record.get("vectors") or {}
    out: dict[str, Any] = {"available": True, "seed": record.get("seed"),
                           "order": record.get("order"), "status": divergence.get("status"),
                           "diverged": divergence.get("status") == "diverged",
                           "divergence": divergence,
                           "r_theta0": number((record.get("r") or {}).get("theta0")),
                           "vector_sha256": {kind: (vectors.get(kind) or {}).get("sha256")
                                             for kind in ("delta_hat", "delta_lin")}}  # fmt: skip
    for kind, q_key in (("delta_hat", "q_hat"), ("delta_lin", "q_lin")):
        ks = k.get(kind) if isinstance(k, Mapping) else None
        found = {"q": number(record.get(q_key)), "r": number((record.get("r") or {}).get(kind)),
                 "norm": number((record.get("norms") or {}).get(kind)),
                 "K_benign": number((ks or {}).get("K_benign")),
                 "K_harmful": number((ks or {}).get("K_harmful")),
                 "K_reason": None if ks else ("no K readouts (--no-k)" if k is None
                                              else k.get(f"{kind}_unavailable"))}  # fmt: skip
        pairs = pairs_hat if kind == "delta_hat" else pairs_lin
        if pairs is not None:
            sha = (vectors.get(kind) or {}).get("sha256")
            row = (pair_row(pairs, a_sha256=sha, b_run=replay_run) if sha
                   else {"available": False, "reason": f"no {kind} vector written"})  # fmt: skip
            found |= {"cosine_with_delta_theta": row.get("cosine"),
                      "norm_ratio": row.get("norm_ratio"), "delta_theta_norm": row.get("norm_b"),
                      "pairs_reason": row.get("reason")}  # fmt: skip
        if delta_r is not None:
            found["delta_r_minus_q"] = difference(delta_r, found["q"])
        out[kind] = found
    norms = record.get("norms") or {}
    out["lin_against_hat"] = {"relative_difference": number(norms.get("relative_difference")),
                              "cosine": number(norms.get("cosine"))}  # fmt: skip
    checks = []
    for check in record.get("checks") or []:
        values = [number(check.get(key)) for key in ("half", "double")]
        found = [v for v in values if v is not None]
        checks.append({"step": check.get("step"), "half": values[0], "double": values[1],
                       "largest": max(found) if found else None})  # fmt: skip
    largest = [c["largest"] for c in checks if c["largest"] is not None]
    done = [c["step"] for c in checks]
    out["epsilon_checks"] = {
        "steps": checks,
        "largest": max(largest) if largest else None,
        "missing_steps": [t for t in CHECK_STEPS if t not in done],
    }
    reference = record.get("reference") or {}
    bound = number(reference.get("bound"))
    lin_norms = [number(r.get("delta_lin_norm")) for r in record.get("per_step") or []]
    lin_norms = [v for v in lin_norms if v is not None]
    bound_ratio = max(lin_norms) / bound if bound and lin_norms else None
    near: dict[str, Any] = {"bound": bound, "bound_ratio": bound_ratio, "floor": lin_floor,
                            "numerically_ambiguous": None}  # fmt: skip
    if bound_ratio is not None and number(lin_floor) is not None and divergence.get("reason") in (
            None, "bound"):  # fmt: skip
        near["numerically_ambiguous"] = abs(bound_ratio - 1) <= RESOLVED_RATIO * abs(lin_floor)
    out["near_bound"] = near
    return out


def m2_floors(row: Mapping[str, Any], floors: Mapping[str, Any]) -> dict[str, Any]:
    """The linear repeat's floors beside one row's q, K and cosines, and beside ‖δ_lin − δ̂‖/‖δ̂‖
    the repeat's relative δ_lin difference (module docstring)."""
    out: dict[str, Any] = {
        "lin_against_hat_relative_difference": resolution(
            (row.get("lin_against_hat") or {}).get("relative_difference"),
            floors.get("vector_lin_relative"),
        )
    }
    for kind, tag in (("delta_hat", "hat"), ("delta_lin", "lin")):
        found = row.get(kind) or {}
        out[kind] = {
            "q": resolution(found.get("q"), floors.get(f"q_{tag}")),
            "K_benign": resolution(found.get("K_benign"), floors.get(f"K_benign_{tag}")),
            "K_harmful": resolution(found.get("K_harmful"), floors.get(f"K_harmful_{tag}")),
            "cosine_with_delta_theta": resolution(
                found.get("cosine_with_delta_theta"), floors.get(f"cosine_{tag}")
            ),
        }
    return out


def prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    """The predictions of one order (numbers only): q, the norms, the divergence."""
    if not row.get("available"):
        return {"available": False, "reason": row.get("reason")}
    div = row["divergence"]
    return {"available": True, "order": row.get("order"), "status": row["status"],
            "diverged_step": div.get("step"), "diverged_reason": div.get("reason"),
            "q_hat": row["delta_hat"]["q"], "q_lin": row["delta_lin"]["q"],
            "norm_hat": row["delta_hat"]["norm"], "norm_lin": row["delta_lin"]["norm"],
            "lin_against_hat": row["lin_against_hat"],
            "epsilon_checks_largest": row["epsilon_checks"]["largest"],
            "near_bound": {k: row["near_bound"][k] for k in ("bound_ratio",
                                                             "numerically_ambiguous")}}  # fmt: skip


def q_contrast(edited: Mapping[str, Any], stream: Mapping[str, Any],
               floors: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """q of an order minus q of the stream order, for δ̂ and δ_lin, with the linear floor."""
    out = {}
    for kind, tag in (("delta_hat", "hat"), ("delta_lin", "lin")):
        dq = difference((edited.get(kind) or {}).get("q"), (stream.get(kind) or {}).get("q"))
        out[f"q_{tag}"] = {"difference": dq, "floor": resolution(dq, floors.get(f"q_{tag}"))}
    return out


def control_pair(
    seed: int,
    control: Mapping[str, Any],
    stream: Mapping[str, Any],
    registered: Mapping[str, Any],
    floors: Mapping[str, Any],
) -> dict[str, Any]:
    """One order-control pair: q of the control's order minus the stream order's (δ̂ and δ_lin),
    the registered K_harmful and harm differences (control − own η_l run) and their signs."""
    dq = q_contrast(control, stream, floors)
    dk, dh = number(registered.get("K_harmful")), number(registered.get("harmful"))
    signs = {}
    for tag in ("hat", "lin"):
        q = dq[f"q_{tag}"]["difference"]
        signs[f"q_{tag}"] = {
            "q": None if q is None else sign(q),
            "K_harmful": None if dk is None else sign(dk),
            "harm": None if dh is None else sign(dh),
            "q_opposite_K_harmful": None if q is None or dk is None or not q or not dk
            else sign(q) == -sign(dk),
            "q_opposite_harm": None if q is None or dh is None or not q or not dh
            else sign(q) == -sign(dh),
        }  # fmt: skip
    return {"seed": seed, **dq, "K_harmful_difference": dk, "harm_difference": dh,
            "registered_reasons": {k: v for k, v in registered.items() if number(v) is None},
            "signs": signs, "control_status": control.get("status"),
            "stream_status": stream.get("status")}  # fmt: skip


# ------------------------------------------------------------------ M3


def small_amplitudes(small: Mapping[str, Any] | str, tokens: Mapping[str, Any] | str,
                     r0: float | None) -> dict[str, Any]:  # fmt: skip
    """M3 of one seed: per point a its r, both panels' opening log-probabilities and K's; the
    odd and even parts per a with r(θ₀) = ``r0``; the halving ratios with their ``local`` flags
    (module docstring); the token means at ±0.1 and ±1 and θ₀'s."""
    out: dict[str, Any] = {"r_theta0": r0}
    rows: dict[float, Mapping[str, Any]] = {}
    if isinstance(small, str):
        out["points"] = {"available": False, "reason": small}
    else:
        rows = {float(r["a"]): r for r in small.get("rows", []) if number(r.get("a")) is not None}
        keys = ("r", "harmful_R", "harmful_C", "harmful_contrast", "benign_R", "benign_C",
                "benign_contrast", "K_benign", "K_harmful")  # fmt: skip
        out["points"] = {f"{a:+g}": {k: number(row.get(k)) for k in keys}
                         | {"reason": row.get("reason") if row.get("reason") != "readout_only"
                            else None}
                         for a, row in sorted(rows.items())}  # fmt: skip
    parts = {}
    for a in SMALL_AMPLITUDES:
        plus, minus = (number((rows.get(x) or {}).get("r")) for x in (a, -a))
        odd = None if plus is None or minus is None else (plus - minus) / 2
        even = None if odd is None or number(r0) is None else (plus + minus) / 2 - r0
        parts[a] = {"odd": odd, "even": even}
    out["parts"] = {f"{a:g}": v for a, v in parts.items()}
    halving = {}
    for a, half in zip(SMALL_AMPLITUDES, SMALL_AMPLITUDES[1:], strict=False):
        found = {}
        for part, (low, high) in (("odd", ODD_LOCAL), ("even", EVEN_LOCAL)):
            value = ratio(parts[a][part], parts[half][part])
            found[part] = {"ratio": value, "local": None if value is None else
                           low <= value <= high}  # fmt: skip
        halving[f"{a:g}/{half:g}"] = found
    out["halving"] = halving
    if isinstance(tokens, str):
        out["tokens"] = {"available": False, "reason": tokens}
    else:
        points = {}
        for point in tokens.get("points", []):
            t = point.get("tokens") or {}
            points[f"{float(point['a']):+g}"] = {
                "r": number(point.get("r")),
                **{k: number(t.get(k)) for k in ("entropy", "rare_mass", "delta_logprob")},
                "count": t.get("count")}  # fmt: skip
        theta0 = tokens.get("theta0") or {}
        out["tokens"] = {"available": True, "theta0": {
            "r": number(theta0.get("r")),
            **{k: number((theta0.get("tokens") or {}).get(k)) for k in ("entropy", "rare_mass")}},
            "points": points,
            "parts": {f"{float(p['a']):g}": {"odd": number(p.get("odd")),
                                             "even": number(p.get("even"))}
                      for p in tokens.get("pairs", [])}}  # fmt: skip
    return out


# ------------------------------------------------------------------ M4


def vector_point(row: Mapping[str, Any], counts: Mapping[str, Any] | str | None) -> dict[str, Any]:
    """One point of a vector run (``vector_points.run_vector``): r (primary), K's, the opening
    log-probabilities, the utility of its full evaluation and its harm when labeled."""
    full = row.get("full") or {}
    out = {"a": number(row.get("a")), "condition": row.get("condition"),
           "evaluated": bool(row.get("evaluated")), "reason": row.get("reason"),
           "matched_k": bool(row.get("matched_k")), "r": number(row.get("r")),
           "K_benign": number(row.get("K_benign")), "K_harmful": number(row.get("K_harmful")),
           **{k: number(row.get(k)) for k in ("harmful_R", "harmful_C", "benign_R", "benign_C")},
           "utility": {k: number(full.get(k)) for k in UTILITY} if full else None,
           "full_K": {k: number(full.get(k)) for k in ("K_benign", "K_harmful")} if full
           else None}  # fmt: skip
    if row.get("evaluated"):
        out["harm"] = harm(counts)
    return out


def matched_of(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The matched-K search of a vector record: its status, extensions, c, the chosen K and the
    out-of-tolerance flag as the record holds them."""
    matched = record.get("matched")
    if not isinstance(matched, Mapping):
        return None
    search, refined = matched.get("search") or {}, matched.get("refine") or {}
    return {"status": matched.get("status"), "search_status": search.get("status"),
            "extensions": search.get("extensions"), "monotone": search.get("monotone"),
            "scales": [{"c": sa.as_finite(s.get("c")), "K_benign": sa.as_finite(s.get("K_benign"))}
                       for s in search.get("scales", [])],
            "c": number(matched.get("c")), "K": number(refined.get("K")),
            "c1": number(refined.get("c1")), "K1": number(refined.get("K1")),
            "c2": number(refined.get("c2")), "K2": number(refined.get("K2")),
            "chosen": refined.get("chosen"),
            "out_of_tolerance": refined.get("out_of_tolerance")}  # fmt: skip


def vector_record(record: Mapping[str, Any] | str,
                  counts: Mapping[str, Mapping[str, Any] | str]) -> dict[str, Any]:  # fmt: skip
    """A vector record's points (``counts``: condition → the registered label counts or why there
    are none) and its matched-K search."""
    if isinstance(record, str):
        return {"available": False, "reason": record}
    if record.get("available") is False:
        return {"available": False, "reason": f"unavailable: {record.get('reason')}"}
    points = {}
    for row in record.get("rows", []):
        key = "kstar" if row.get("matched_k") else f"{float(row['a']):+g}"
        points[key] = vector_point(row, counts.get(row.get("condition")))
    return {"available": True, "label": record.get("label"), "seed": record.get("seed"),
            "sign": record.get("sign"), "vector": record.get("vector"),
            "vector_norm": number(record.get("vector_norm")), "run_id": record.get("run_id"),
            "points": points, "matched": matched_of(record)}  # fmt: skip


def point_contrast(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> dict[str, Any]:
    """a − b for r, the K's and the harm (the registered share and the all-harmful one)."""
    a, b = a or {}, b or {}
    out = {k: difference(a.get(k), b.get(k)) for k in ("r", "K_benign", "K_harmful")}
    ha, hb = a.get("harm") or {}, b.get("harm") or {}
    out["harmful"] = difference(ha.get("harmful"), hb.get("harmful"))
    out["harmful_high"] = difference(ha.get("harmful_high"), hb.get("harmful_high"))
    return out


# ------------------------------------------------------------------ M5


def projection_key(meta: Mapping[str, Any]) -> str:
    kind = meta.get("kind")
    return {"mean": "mu_hat_minus_i", "grad_r": "grad_r"}.get(kind, str(meta.get("name")))


def eps_key(eps: float) -> str:
    return f"{float(eps):g}"


def force_values(record: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    """step → ε key → part → {Dforce_norm, force_norm, linear_norm, projections by key}, with the
    halves' difference and the rounding."""
    keys = [projection_key(p) for p in record.get("projections", [])]
    out: dict[int, dict[str, Any]] = {}
    for c in record.get("checkpoints", []):
        step: dict[str, Any] = {"delta_norm": number(c.get("delta_norm")),
                                "total_norm": c.get("total_norm"), "epsilons": {}}  # fmt: skip
        for e in c.get("epsilons", []):
            parts = {label: {"Dforce_norm": number(p.get("Dforce_norm")),
                             "force_norm": number(p.get("force_norm")),
                             "linear_norm": number(p.get("linear_norm")),
                             "projections": {k: number(v) for k, v in
                                             zip(keys, p.get("projections", []), strict=False)}}
                     for label, p in (e.get("parts") or {}).items()}  # fmt: skip
            step["epsilons"][eps_key(e["epsilon"])] = {
                "parts": parts, "halves_difference": e.get("halves_difference"),
                "rounding": e.get("rounding"), "force_file": e.get("force_file")}  # fmt: skip
        out[int(c["step"])] = step
    return out


def relative_gap(a: Any, b: Any) -> float | None:
    """|a − b| over the mean of |a| and |b| (None when both are 0)."""
    x, y = number(a), number(b)
    if x is None or y is None or (x == 0 and y == 0):
        return None
    return abs(x - y) / ((abs(x) + abs(y)) / 2)


def force_panel(values: Mapping[int, Mapping[str, Any]], floors: Mapping[str, Any]
                ) -> dict[str, Any]:  # fmt: skip
    """One panel's M5: per step and ε the full panel's ‖D f̂_t‖ and projections with the force
    repeat's floors, the halves' agreement, and the ε-convergence against ε = 10⁻²."""
    out = {}
    for step, found in sorted(values.items()):
        entry: dict[str, Any] = {"delta_norm": found["delta_norm"], "epsilons": {},
                                 "convergence": {}}  # fmt: skip
        for key, e in found["epsilons"].items():
            parts = e["parts"]
            full = parts.get("full") or {}
            halves = e.get("halves_difference") or {}
            first, second = parts.get("first_half") or {}, parts.get("second_half") or {}
            entry["epsilons"][key] = {
                "full": full,
                "halves": {k: parts.get(k) for k in ("first_half", "second_half")},
                "halves_agreement": {
                    "Dforce_difference_norm": number(halves.get("Dforce_norm")),
                    "relative": ratio(halves.get("Dforce_norm"), full.get("Dforce_norm")),
                    "projections": {k: difference((first.get("projections") or {}).get(k),
                                                  (second.get("projections") or {}).get(k))
                                    for k in (full.get("projections") or {})}},
                "floors": {"Dforce_norm": resolution(full.get("Dforce_norm"),
                                                     floors.get("vector_difference_norm")),
                           **{k: resolution(v, (floors.get("projections") or {}).get(k))
                              for k, v in (full.get("projections") or {}).items()}},
                "rounding": e.get("rounding"),
            }  # fmt: skip
        central = (found["epsilons"].get(eps_key(CENTRAL_EPSILON)) or {}).get("parts", {})
        base = central.get("full") or {}
        for key, e in found["epsilons"].items():
            if key == eps_key(CENTRAL_EPSILON):
                continue
            full = e["parts"].get("full") or {}
            entry["convergence"][key] = {
                "Dforce_norm": relative_to(full.get("Dforce_norm"), base.get("Dforce_norm")),
                "projections": {k: relative_to(v, (base.get("projections") or {}).get(k))
                                for k, v in (full.get("projections") or {}).items()}}  # fmt: skip
        out[str(step)] = entry
    return out


def relative_to(value: Any, base: Any) -> float | None:
    """|value − base| / |base| (None for a zero or missing base)."""
    gap, b = difference(value, base), number(base)
    return None if gap is None or not b else abs(gap) / abs(b)


def panels_agreement(a: Mapping[int, Mapping[str, Any]], b: Mapping[int, Mapping[str, Any]],
                     floors: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """The two panels' agreement per step and ε: each projection's and ‖D f̂_t‖'s two values,
    difference, relative gap and the difference's floor ratio."""
    out = {}
    for step in sorted(set(a) & set(b)):
        per_eps = {}
        for key in sorted(set(a[step]["epsilons"]) & set(b[step]["epsilons"])):
            fa = a[step]["epsilons"][key]["parts"].get("full") or {}
            fb = b[step]["epsilons"][key]["parts"].get("full") or {}
            row = {}
            pa, pb = fa.get("projections") or {}, fb.get("projections") or {}
            for k in pa:
                gap = difference(pa.get(k), pb.get(k))
                row[k] = {
                    "first": pa.get(k),
                    "second": pb.get(k),
                    "difference": gap,
                    "relative": relative_gap(pa.get(k), pb.get(k)),
                    "same_sign": None
                    if number(pa.get(k)) is None or number(pb.get(k)) is None
                    else sign(pa[k]) == sign(pb[k]),
                    "floor": resolution(gap, (floors.get("projections") or {}).get(k)),
                }
            gap = difference(fa.get("Dforce_norm"), fb.get("Dforce_norm"))
            row["Dforce_norm"] = {"first": fa.get("Dforce_norm"), "second": fb.get("Dforce_norm"),
                                  "difference": gap,
                                  "relative": relative_gap(fa.get("Dforce_norm"),
                                                           fb.get("Dforce_norm"))}  # fmt: skip
            per_eps[key] = row
        out[str(step)] = per_eps
    return out


def force_floor(a: Mapping[str, Any] | str, b: Mapping[str, Any] | str,
                pairs: Mapping[str, Any] | str) -> dict[str, Any]:  # fmt: skip
    """The amendment's force repeat: the two saved vectors' difference (its pairs record, each
    item checked against the vector its record wrote) and both records' projections at the
    repeated step and ε."""
    if any(isinstance(x, str) for x in (a, b, pairs)):
        why = "; ".join(x for x in (a, b, pairs) if isinstance(x, str))
        return {"available": False, "reason": why}
    step, key = FORCE_REPEAT["step"], eps_key(FORCE_REPEAT["epsilon"])
    va, vb = force_values(a), force_values(b)
    try:
        ea, eb = va[step]["epsilons"][key], vb[step]["epsilons"][key]
    except KeyError:
        return {"available": False, "reason": "a repeat record lacks the repeated step or ε"}
    files = [(e.get("force_file") or {}).get("sha256") for e in (ea, eb)]
    row = pair_row(pairs, a_sha256=files[0], b_sha256=files[1])
    if not row["available"]:
        return {"available": False, "reason": f"pairs: {row['reason']}"}
    parts = {}
    for label in ("full", "first_half", "second_half"):
        pa = (ea["parts"].get(label) or {}).get("projections") or {}
        pb = (eb["parts"].get(label) or {}).get("projections") or {}
        parts[label] = {
            k: {
                "first": pa.get(k),
                "repeat": pb.get(k),
                "difference": difference(pb.get(k), pa.get(k)),
            }
            for k in pa
        }
    full = parts["full"]
    return {"available": True, "panel": FORCE_REPEAT["panel"], "step": step,
            "epsilon": FORCE_REPEAT["epsilon"], "vector_difference_norm": row["distance"],
            "norm_first": row["norm_a"], "norm_repeat": row["norm_b"],
            "relative": ratio(row["distance"], row["norm_a"]), "projections_by_part": parts,
            "projections": {k: (abs(v["difference"]) if v["difference"] is not None else None)
                            for k, v in full.items()}}  # fmt: skip


# ------------------------------------------------------------------ the floors


def linear_floor(first: Mapping[str, Any], again: Mapping[str, Any],
                 pairs_hat: Mapping[str, Any] | str, pairs_lin: Mapping[str, Any] | str
                 ) -> dict[str, Any]:  # fmt: skip
    """The amendment's linear repeat on seed 50000: ``first`` and ``again`` (``linear_row`` of
    the two records, each with its cosine with Δθ), the repeat's pairs records (row 0: the two
    vectors, each checked against the vector its record wrote). The differences of the two δ̂
    and two δ_lin, of their q, K and cosines, and the floors they give."""
    if not (first.get("available") and again.get("available")):
        return {"available": False,
                "reason": "; ".join(str(x.get("reason")) for x in (first, again)
                                    if not x.get("available"))}  # fmt: skip
    out: dict[str, Any] = {"available": True, "seed": FORCE_SEED, "quantities": {}}
    floors: dict[str, float | None] = {}
    for kind, tag, pairs in (("delta_hat", "hat", pairs_hat), ("delta_lin", "lin", pairs_lin)):
        a, b = first[kind], again[kind]
        found = {}
        for key in ("q", "K_benign", "K_harmful", "cosine_with_delta_theta", "norm"):
            gap = difference(b.get(key), a.get(key))
            found[key] = {"first": a.get(key), "repeat": b.get(key), "difference": gap}
        floors[f"q_{tag}"] = _abs(found["q"]["difference"])
        floors[f"K_benign_{tag}"] = _abs(found["K_benign"]["difference"])
        floors[f"K_harmful_{tag}"] = _abs(found["K_harmful"]["difference"])
        floors[f"cosine_{tag}"] = _abs(found["cosine_with_delta_theta"]["difference"])
        row = pair_row(pairs, a_sha256=first["vector_sha256"].get(kind) or "",
                       b_sha256=again["vector_sha256"].get(kind) or "")  # fmt: skip
        found["vectors"] = {"difference_norm": row.get("distance"), "norm": row.get("norm_a"),
                            "relative": ratio(row.get("distance"), row.get("norm_a")),
                            "reason": row.get("reason")}  # fmt: skip
        floors[f"vector_{tag}_relative"] = found["vectors"]["relative"]
        out["quantities"][kind] = found
    out["floors"] = floors
    return out


def _abs(x: float | None) -> float | None:
    return None if x is None else abs(x)


def twins_steps(record: Mapping[str, Any] | str) -> dict[str, Any]:
    """A twins (or floor) pairs record: per step ‖θ_t − θ′_t‖ and r of both sides, and r(θ₀) of
    the record."""
    if isinstance(record, str):
        return {"available": False, "reason": record}
    if record.get("kind") != "study5-pairs":
        return {"available": False, "reason": "not a study5-pairs record"}
    r = record.get("r") or {}
    steps = {}
    for row in record.get("pairs", []):
        a, b = row.get("a") or {}, row.get("b") or {}
        try:
            step = int(a.get("step"))
        except (TypeError, ValueError):
            continue
        ra, rb = number(r.get(item_name(a))), number(r.get(item_name(b)))
        steps[step] = {"distance": number(row.get("distance")),
                       "norm_stream": number(row.get("norm_a")),
                       "norm_other": number(row.get("norm_b")), "r_stream": ra, "r_other": rb,
                       "r_difference": difference(rb, ra)}  # fmt: skip
    return {"available": True, "r_theta0": number(r.get("theta0")), "steps": steps,
            "missing_steps": [t for t in TWIN_STEPS if t not in steps]}  # fmt: skip


def twins_row(found: Mapping[str, Any], floor: Mapping[str, Any], own: Mapping[str, Any],
              stream: Mapping[str, Any]) -> dict[str, Any]:  # fmt: skip
    """One seed's twins: per step the distance and the r difference with the twins' floor at the
    same step; at the endpoint each twin's Δr and its excess over its own q (δ̂ and δ_lin)."""
    if not found.get("available"):
        return {"available": False, "reason": found.get("reason")}
    steps = {}
    floor_steps = floor.get("steps", {}) if floor.get("available") else {}
    for step, row in sorted(found["steps"].items()):
        under = floor_steps.get(step) or {}
        steps[str(step)] = {**row,
                            "floor_distance": under.get("distance"),
                            "floor_r_difference": under.get("r_difference"),
                            "distance_floor": resolution(row["distance"], under.get("distance")),
                            "r_floor": resolution(row["r_difference"],
                                                  under.get("r_difference"))}  # fmt: skip
    end = found["steps"].get(TWIN_STEPS[-1]) or {}
    r0 = found["r_theta0"]
    endpoint = {
        "delta_r_stream": difference(end.get("r_stream"), r0),
        "delta_r_twin": difference(end.get("r_other"), r0),
    }
    for side, key, pred in (("twin", "delta_r_twin", own), ("stream", "delta_r_stream", stream)):
        for kind, tag in (("delta_hat", "hat"), ("delta_lin", "lin")):
            q = (pred.get(kind) or {}).get("q") if pred.get("available") else None
            endpoint[f"excess_{side}_{tag}"] = difference(endpoint[key], q)
    return {"available": True, "r_theta0": r0, "steps": steps, "endpoint": endpoint,
            "prediction": prediction(own) if own else None}  # fmt: skip


# ------------------------------------------------------------------ M6


def outcome_of(run: Mapping[str, Any] | str, counts: Mapping[str, Any] | str | None
               ) -> dict[str, Any]:  # fmt: skip
    """An edited run's outcome: its stop, steps, K's and utility at step 25 and its harm."""
    if isinstance(run, str):
        return {"available": False, "reason": run}
    row = run.get("row") or {}
    return {"available": True, "run_id": run.get("run_id"), "stop": run.get("stop"),
            "steps": run.get("steps"), **{k: number(row.get(k)) for k in MEASURES},
            "harm": harm(counts)}  # fmt: skip


def stream_values(per_seed: Mapping[str, Any] | None) -> dict[str, Any]:
    """The stream order's η_l run as the registered analysis holds it (its line scan's +1)."""
    values = per_seed or {}
    return {k: number(values.get(k)) for k in (*MEASURES, "harmful", "refusal",
                                               "safe_non_refusal")}  # fmt: skip


def edited_row(
    seed: int,
    tag: str,
    edit: Mapping[str, Any],
    predicted: Mapping[str, Any],
    outcome: Mapping[str, Any],
    stream: Mapping[str, Any],
    stream_linear: Mapping[str, Any],
    floors: Mapping[str, Any],
) -> dict[str, Any]:
    """One edited order of M6: its prediction and outcome beside the stream order's, the
    differences of both from it (module docstring, "M6's excess")."""
    harmed = outcome.get("harm") or {}
    differences = {k: difference(outcome.get(k), stream.get(k)) for k in MEASURES}
    differences["harmful"] = difference(harmed.get("harmful"), stream.get("harmful"))
    return {"seed": seed, "tag": tag, "edit": dict(edit), "prediction": prediction(predicted),
            "predicted_difference": q_contrast(predicted, stream_linear, floors)
            if predicted.get("available") and stream_linear.get("available") else None,
            "outcome": outcome, "stream": stream,
            "outcome_difference": differences if outcome.get("available") else None}  # fmt: skip


def early_late(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The shuffles' early (1–5) against late (21–25) ranges: the means of |Δq| and of the
    outcomes' |differences| from the stream order, over the seeds and draws."""
    out = {}
    for first, last in SHUFFLES:
        mine = [r for r in rows
                if (r["edit"].get("first"), r["edit"].get("last")) == (first, last)]  # fmt: skip

        def values(get, mine=mine) -> dict[str, Any]:
            return {f"s{r['seed']}-{r['edit'].get('draw')}": _abs(get(r)) for r in mine}

        out[f"{first}-{last}"] = {
            "q_hat": spread(values(lambda r: ((r["predicted_difference"] or {}).get("q_hat")
                                              or {}).get("difference"))),
            "q_lin": spread(values(lambda r: ((r["predicted_difference"] or {}).get("q_lin")
                                              or {}).get("difference"))),
            **{k: spread(values(lambda r, k=k: (r["outcome_difference"] or {}).get(k)))
               for k in ("harmful", "K_harmful", "K_benign")},
        }  # fmt: skip
    return out


# ------------------------------------------------------------------ the cases


def cases(m2: Mapping[str, Any], m4: Mapping[str, Any], m6: Mapping[str, Any],
          checks: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:  # fmt: skip
    """Every diverged, near-bound, unbracketed, out-of-tolerance, non-finite-K or failed-check
    case of the record, listed."""
    out: dict[str, list[dict[str, Any]]] = {
        "diverged": [],
        "near_bound": [],
        "unbracketed": [],
        "out_of_tolerance": [],
        "nonfinite_k": [],
        "checks_not_exact": [],
        "runs_not_ended_steps": [],
    }
    for where, rows in (("own", m2.get("per_seed", {})), ("control", m2.get("controls", {})),
                        ("e2_prediction", m2.get("e2_predictions", {})),
                        ("edited", m2.get("edited_predictions", {}))):  # fmt: skip
        for key, row in rows.items():
            if not isinstance(row, Mapping):
                continue
            status = row.get("status")
            if status == "diverged":
                div = row.get("divergence") or {
                    "step": row.get("diverged_step"),
                    "reason": row.get("diverged_reason"),
                }
                out["diverged"].append({"where": where, "key": str(key), "step": div.get("step"),
                                        "reason": div.get("reason")})  # fmt: skip
            near = row.get("near_bound") or {}
            if near.get("numerically_ambiguous"):
                out["near_bound"].append({"where": where, "key": str(key),
                                          "bound_ratio": near.get("bound_ratio")})  # fmt: skip
    for name, record in m4.get("records", {}).items():
        if not isinstance(record, Mapping) or not record.get("available"):
            continue
        matched = record.get("matched")
        if matched is not None:
            if matched.get("status") in ("not_bracketed", "invalid_k"):
                out["unbracketed"].append({"point": name, "status": matched.get("status"),
                                           "extensions": matched.get("extensions")})  # fmt: skip
            elif matched.get("status") == "out_of_tolerance":
                out["out_of_tolerance"].append({"point": name, "K": matched.get("K")})
        for key, point in record.get("points", {}).items():
            if point.get("reason") == "nonfinite_k":
                out["nonfinite_k"].append({"point": name, "at": key})
    for name, summary in checks.items():
        if isinstance(summary, Mapping) and summary.get("available") and not summary.get("exact"):
            out["checks_not_exact"].append(
                {
                    "check": name,
                    "passed": summary.get("passed"),
                    "within_tolerance": summary.get("within_tolerance"),
                }
            )
    for block in ("swaps", "shuffles"):
        for row in (m6.get(block) or {}).get("rows", []):
            outcome = row.get("outcome") or {}
            if outcome.get("available") and outcome.get("stop") != sa.STEPS_STOP:
                out["runs_not_ended_steps"].append({"block": block, "seed": row["seed"],
                                                    "tag": row["tag"],
                                                    "stop": outcome.get("stop")})  # fmt: skip
    return out
