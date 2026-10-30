"""Study 3's simulation before validation: the scenario models, the exact truth, one simulated
study through the registered functions and the calibration through the selection.

docs/study3-preregistration.md, "Simulation before validation": a CPU Monte Carlo of the whole
procedure (search, support, utility, validation, N, confirmation and analysis) over a frozen grid
of scenarios, with the coverage of the interval at level L screened against the registered
estimand ("Questions and estimands", "The comparisons": the expected interpolated difference over
fresh supported seed blocks, under the frozen rates). This module holds the scenarios, the harm
and label models, the truth that the coverage check compares with, one study of the procedure
and the calibration of each scenario. Pure numpy and scipy on the CPU; it reads no run
directory, no harm label, response or judge output.

- ``SCENARIOS`` (``SCENARIOS_BY_NAME``): the grid, C01–C17 (coverage; C16 and C17 also power),
  P18–P19 (power only) and G1–G2 (the validation gates), each ``Scenario`` with its parameters
  and one ``source`` line per parameter (``SOURCES``).
- ``Calibration``: what is calibrated per (scenario, candidate) through the selection: the
  per-run failure f, the safe alternative's utility shifts, the block-effect SDs and offsets.
- ``harm_curve``, ``bracket_bias``, ``CURVATURE_PP``, ``SLOPE_PP``, ``NOMINAL_BIAS_AF_PP``: the
  harm curves in ln K and their curvature.
- ``draw_effects`` and ``expected_clipped``: the block effects' shapes and E[clip(μ + σ·e, 0, 1)]
  in closed form.
- ``endpoint_labels``: one block's primary judged shares, from 521 labels per endpoint.
- ``truth`` (with ``normal_piecewise`` and ``support_breakpoints``): the registered estimand of a
  study at its final rates.
- ``shifted_vectors``: the seed vectors with a calibration's utility shifts applied.
- ``simulate_study`` (with ``Setup``, ``StudyOutcome``, ``synthetic_endpoint``,
  ``outcome_json`` and ``outcome_from_json``): one study through the registered functions.
- ``calibrate_scenario`` (with ``realized_support``, ``calibrate_failure``, ``calibrate_safe``,
  ``calibrate_spread``, ``calibrate_offsets``, ``calibration_stream``, ``label_variance``):
  everything calibrated per (scenario, candidate), through the selection.

**The harm model** (M1–M5). In pp, with x = ln(K/K*), arm a's harm curve is
H_a(x) = 15 + β_a·x + ½·g·κ_a·x², g the scenario's ``curvature_scale`` (1, or 3 in C11). β_a and
κ_a are study 1's local slope and curvature at K*, from study 1's logistic curves at baseline 0.15
(h′ = p(1−p)s, h″ = p(1−p)(1−2p)s² + 2p(1−p)q): β A 15.300, F and P 17.7225 pp per ln K; κ A
9.5370, F and P 11.1240 pp per (ln K)². The curve is parameterized by κ itself, so that g = 3
triples the curvature (scaling the logistic's negative q would lower h″ instead). A block's
effects are additive on the probability scale: δ_F = 0, δ_A = e_AF + o_A, δ_P = e_PF + o_P, with
e_AF and e_PF independent, of the scenario's shape, SDs σ_AF and σ_PF; the offsets set the truth
averaged over the selections to Δ_AF and Δ_PF. An endpoint's harm probability is
h = clip(H_a(x)/100 + δ_a, 0, 1). The harm comes only from the planning spreads (through σ) and
study 1's curvature: no study-2 harm is read.

**The shapes** (M3): SD 1 times σ: ``normal``; ``heavy``, a normal scaled ×3 with probability 0.1,
rescaled to E[S²] = 1; ``skewed``, (G − 2)/√2 with G ~ Gamma(2); ``rare``, (B − 0.1)/√0.09 with
B ~ Bernoulli(0.1).

**The labels** (M5, M6, M6b). Per prompt (521, exchangeable, no fixed prompt effects),
y = 1[Z < Φ⁻¹(h)] with the copula Z = √ρ_b·Z_{b,i} + √(ρ_a − ρ_b)·Z_{b,a,i} + √(1 − ρ_a)·Z_{b,a,r,i}
(ρ_a 0.7 within an arm's two endpoints, ρ_b 0.3 within a block). The judge calls y = 1 harmful
with probability Se_a and y = 0 with FPR_a. A label is missing at the scenario's rate (MCAR), or
among harmful responses only (MNAR). The primary h is the share judged harmful and not missing,
over 521. Given h, its expectation is p_a(h) = (1 − m_h)·Se_a·h + (1 − m_n)·FPR_a·(1 − h), with m_h
and m_n the missing rates among harmful and not harmful responses.

**The truth** (the registered estimand, judge-rated, a missing label counting as not harmful),
per study and comparison, at the study's final rates:
truth = Σ_v E[1_A·1_F·(I_A − I_F)] / Σ_v E[1_A·1_F] over the inputs' whole seed vectors v (each
fresh block draws one, with replacement), and the same for P–F. 1_a is arm a's support in a block
("Within a block": both runs usable, untied, K_lo ≤ K* ≤ K_hi, ln K_hi − ln K_lo ≤ 0.60, no gross
collapse, the MMLU loss above 15 pp) times (1 − f)², f the calibrated per-run failure;
I_a = (1 − w)·p_a(x_lo) + w·p_a(x_hi), with p_a(x) = p_a(E[h]) and
E[h] = E[clip(H_a(x)/100 + δ_a, 0, 1)] in closed form (``expected_clipped``: normal, heavy and
rare through the clipped-normal mean of their components, μ[Φ(β) − Φ(α)] + s[φ(α) − φ(β)] +
[1 − Φ(β)] with α = −μ/s and β = (1 − μ)/s, a two-point component at s = 0; skewed through the
exact truncated moments of Gamma(2), ``scipy.special.gammainc``). No quadrature crosses the
clip's kinks. Labels add no bias (p_a is linear in h), and failures are independent of harm.
Without jitter the truth is a finite sum over the vectors. Under jitter (C12: N(0, σ_e,arm) per
run in ln K), A and F are independent given v, so the expectations factorize per arm; each arm's
two per-run jitters are integrated as nested one-dimensional integrals over the normal CDF scale
(``normal_piecewise``), with breakpoints (``support_breakpoints``) where the bracket condition
changes (x = 0), where the 0.60 width binds, at a tie, and where a gross collapse starts (the
endpoint's MMLU loss at 15 pp, a point on the piecewise-linear utility curve): in the second
jitter for each first, and in the first where the inner breakpoints change order, at the same
thresholds. Gauss–Legendre on each smooth piece, from 16 nodes, doubling until n and 2n nodes
agree within 1e-6 on the proportion scale on every piece, up to 512: a truth that does not
converge raises ``TruthNotConverged``, at every evaluation, and no coverage is computed from it.
The pieces are also split at fixed points of the normal scale (``SPLITS``: 0, ±1, ±2, ±3, ±4,
±6, ±8), so that no piece but the two outer tails (mass 6·10⁻¹⁶ each) reaches into a tail, where
Φ⁻¹'s growth would slow the rule on the CDF scale. (A plain Gauss–Hermite rule misses the
support's discontinuities: a standard normal cut at 0.3125 gives 0.5 at 20 nodes instead of
0.62267.)

**Units.** h, block effects, offsets and σ are proportions; spreads, curvature, slopes and
utility losses are pp, as ``Endpoint`` holds them.

**The procedure** (``simulate_study``; "Candidates and rate search", "Validation",
"Sample size from validation", "Confirmation", "Inference"). A study is a pure function of its
stream, scenario and index: every draw comes from ``study3_inference.stream(base, "procedure",
scenario, index, *part)``, each part (``search``, ``numerical``, ``validation``,
``confirmation``, ``labels``, and the same under ``candidate-2`` for the fallback) its own
stream, and every block draws the same numbers whatever is later read of it (P's runs
included), so that nothing depends on the worker, the order or P's activity. Each seed the
procedure touches draws one whole vector, with replacement (the bootstrap of whole seed
vectors). The registered functions decide, on simulated ``Endpoint`` records labeled with the
registered seeds (``synthetic_endpoint``), never re-implemented:
1. *Search* (``search_state``): round 0 at the starting rates on the four search seeds' vectors,
   round 1 at the adjusted rates; K-only runs never fail (M13).
2. *Repair* (``repair_state``, amendment 3): the numerical runs on the three numerical seeds'
   vectors, the qualification passed (M13), the repaired rates' runs; the final rates.
3. *Validation* (``block_state``, ``utility_gate`` at α 0.05 per bound, 22 of 28 or 31 of 40):
   each run's K is its vector's at the final rate (plus the jitter, M8), its utility the
   vector's at that K (shifted for the safe alternative, M10; above 15 pp a gross collapse); it
   fails scientifically with the calibrated probability f (M7).
4. *Spread and N*: SD = 100·sd(d_AF) over the m A–F-supported validation blocks (judged primary
   harm, full precision); N = ``sample_size(SD, m, L)`` at each level (at least ``n_floor``),
   the cap ``cap_of(N)``; a cap beyond the 200 reserved seeds leaves the study unconfirmed at
   that level (``cap_beyond_seeds``); N above 90 proceeds (M17).
5. *Confirmation* (``confirmation_state``): blocks in seed order from 10400 until the largest
   cap is started and each comparison has 99 supported blocks, or the 200 seeds are used; per
   level the stop (every active counter at N, or the cap) and the analysis sets are the
   registered function's.
6. *Analysis*: per block d = interpolated(share_lo, share_hi, w) of the arm minus F's
   (``study3_blocks.interpolated``); per (N point, comparison) one ``bootstrap_t`` seed,
   ``stream(base, "bootstrap", scenario, index, comparison, point)``, for every level, chunk
   500; covered when lower ≤ truth ≤ upper. An undetermined interval, or an N point not reached
   within the 200 blocks, is a miss (M18). The cap's reading gives the direction (the power) and
   ``utility_gate`` at α 0.05/8 the utility claim, reported only.
The two modes. ``"screen"``: the scenario's candidate with no fallback (a failed A–F gate is
``af_gate``, M14), N from the rule alone, P–F computed in every confirmed study as if active
(M16), a joint miss when either comparison misses. ``"procedure"``: the realized route, candidate
1, then candidate 2 (fresh search, repair and validation on its seeds, with its calibration),
then the scientific feasibility report (``scientific_report``); N raised to ``n_min`` when the
screen found the power short; P–F counted only where its gate passed, the joint test then A–F's
(M27).

**The calibration through the selection** (``calibrate_scenario``; M1, M7, M10, M15). On the
stream (calibration base, "support", scenario, candidate) (``calibration_stream``), 10 000
studies' search, repair and 28-block validation (``simulate_study(..., validation_only=True)``)
at the rates each selects, drawn once with every run's failure uniform kept:
- *f* (M7): by bisection on [0, 1), the realized A–F support of the validation blocks, bracket
  failures and gross collapses included, equals the scenario's s within 1e-4; f = 0, flagged,
  when the natural failures alone leave it below s. The selection does not depend on f (M13),
  so every step reuses the same draws.
- *The safe shifts* (M10): per arm each vector's MMLU curve moved by c_v = 3 − m + (λ − 1)(ū_v −
  m), λ² = (4 − W)/B over the supported blocks' interpolated MMLU loss (W within, B between the
  vectors), so that it has mean 3 pp and SD 2 pp; λ = 0, flagged, when W > 4. The factual loss
  takes one shift to mean 2 pp. The shift moves gross collapses, so f and the shifts alternate
  until neither moves (at most 10 rounds, else ``CalibrationNotStable``).
- *σ_AF, σ_PF* (M1, M2): by bisection, the observed spread of d at Δ = 0 over the supported
  validation blocks at the selected rates is the planning spread: the square root of the mean
  within-study variance, the quantity the spread rule estimates. Δ = 0 is the truth's, not the
  offset's: the spread is taken at the offsets o⁰_A, o⁰_P at which the mean truth over the
  selections is 0 (the curvature and a differential judge make it nonzero at offset 0). σ given
  o⁰ and o⁰ given σ alternate until neither moves (at most 10 rounds, else
  ``CalibrationNotStable``). The labels enter through their exact conditional variance
  (``label_variance``: the prompts are independent given the block, two endpoints' harmful
  labels bivariate normal thresholds), the expectation of the observed sample variance given
  the blocks, so that every σ reuses the same draws; a test checks it, study by study, against
  simulated labels.
- *o_A, o_P* (M15): solved, from o⁰ and at the calibrated σ, so that the mean, over the
  selections, of the per-study truth at the study's selected rates is Δ_AF (Δ_PF); at Δ = 0 they
  are o⁰.
σ and the offsets depend on f and the shifts, not the other way round: they are computed in the
round in which nothing moved, with the attained spreads (at o⁰) and mean truths and their Monte
Carlo SEs. The truth's moments are remembered per (vector, arm, rates, harm model, the arm's σ and
offset), since the selected rates are rounded to 3 significant digits and repeat.

**Rulings** (the study's frozen settings, each with its one-line source):
- M1, the harm model of block differences: additive on the probability scale, σ calibrated so
  that the observed SD of d at Δ = 0 over the supported validation blocks at the rates the
  procedure selects is the planning spread (study 2's spread calibration,
  ``study2_short_sim.calibrate_spread``; the registration's "true spread equal to the planning
  spread"); decided.
- M2, P–F's spread and Δ_PF: equal to A–F's spread; Δ_PF = Δ_AF in the power scenarios, 0
  elsewhere (none registered: P–F's power is reported; the symmetric choice); decided.
- M3, the shapes: heavy ×3 at 0.1, skew Gamma(2), rare π = 0.1 centered (study 2's
  ``study2_short_sim``: ``HEAVY_*``, ``GAMMA_SHAPE``, ``RARE``, used in study 2's registered
  checks); decided.
- M4, the curvature: a quadratic in ln K with study 1's local κ at K* and slopes, g = 1, and 3
  in C11 (``study2_short_sim.STUDY1`` at K*). At the nominal symmetric bracket (±0.223, w = ½)
  the bias is ½·κ·0.223²: A 0.2371 pp, F 0.2766 pp, A−F −0.0395 pp (``NOMINAL_BIAS_AF_PP``). The
  registration states about 0.06 pp, a fitted estimate with partial cancellation between arms; the
  fixation records both and no threshold depends on them; decided.
- M5, the label model: 521 exchangeable prompts, copula ρ_a 0.7, ρ_b 0.3, baseline h(K*) 0.15
  (``study2_short_sim.Scenario``'s defaults; fixed prompt effects would make the truth
  panel-conditional); decided.
- M6, the judge: Se 0.93, FPR 0.036 in every arm; C13 F and P at Se 0.75 (study 1's Gate-1
  stratum, docs/reports/gate-1.md; sparse strata 0.58–0.75); decided.
- M6b, missing labels: MCAR 1 % in every scenario, C14 MNAR 5 % among harmful (no record holds a
  rate: stress values); decided.
- M7, support: the realized probability that a validation block is A–F-supported at the rates
  the procedure selects, failures of bracket and collapse included; f calibrated to it, 0 and
  flagged when the natural failures exceed 1 − s (the registration's "support 0.88, 0.90,
  0.95"); decided.
- M8, widths and jitter: the vectors' own widths; C12 adds N(0, σ_e) per run in ln K, σ_e A 0.013,
  F 0.032, P 0.027 (docs/study2-short-b20-inputs.json); decided.
- M10, the safe alternative: each vector's MMLU curve shifted by a constant so that the
  interpolated MMLU loss over the supported validation blocks has mean 3 pp and seed SD 2 pp,
  the factual loss mean 2 pp, support 0.95 (the registration's safe alternative, at its
  boundary); decided.
- M13, search and numerical runs never fail, the qualification passes, the repair is simulated
  ("Not in the simulation": the numerical gates); decided.
- M14, candidates: the screen runs the scenario's candidate with no chaining; the verification
  the realized route; C15 starts at candidate 2 (the tests are per candidate); decided.
- M15, Δ's scale: the judge-rated estimand's, the offsets solved so that the truth averaged over
  the selections is Δ ("h: judge-rated" in the notation); decided.
- M16, P–F with P closed: the screen computes it as if active, the verification counts P–F only
  where its gate passed (A–F's analysis set does not depend on P's activity); decided.
- M17, N above 90: the simulated confirmation proceeds (the budget decision is outside the
  model); decided.
- M18, misses: an undetermined interval, or an N point not reached within 200 blocks, is a miss;
  the cap's descriptive cases count in the cap test (conservative); decided.
- M27, the joint test in the verification: over the studies confirmed for A–F, a miss when A–F
  misses or when P–F is active and misses (the registration's joint p₀ 0.05 over the two
  comparisons); decided.
- The constants taken from study 2's simulation (``STUDY1``'s logistic slopes and curvatures,
  the shapes' constants, ρ_a, ρ_b and ``THREAD_VARIABLES``) are written out here so that this
  module loads no GPU path through study 2's analysis; a test compares them with
  ``study2_short_sim``.
- The bracket compares ln K with ln K* (the measured K's order), the width allows 10⁻⁹ in ln K and
  a usable run's MMLU loss 10⁻⁹ pp, as ``study3_blocks`` and ``study3_records`` do.
"""

from __future__ import annotations

import functools
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammainc, ndtr, ndtri

from .study3_blocks import (
    CONFIRMATION_ALPHA,
    VALIDATION_ALPHA,
    WIDTH,
    block_state,
    interpolated,
    utility_gate,
)
from .study3_blocks import SLACK as WIDTH_SLACK
from .study3_decisions import (
    LEVELS,
    MAX_DRAWS_FACTOR,
    MIN_N,
    N_POINTS,
    SIMULATION_FIXATION,
    SUPPORTED_AT,
    TESTED,
    VALIDATION_BLOCKS,
    cap_of,
    coverage_threshold,
    detection_power,
    fixation_problems,
    identities_sha256,
    level_slot,
    sample_size,
    test_identities,
)
from .study3_inference import Interval, bootstrap_t, reading, stream
from .study3_progress import confirmation_state
from .study3_records import (
    GROSS_COLLAPSE,
    GROSS_COLLAPSE_PP,
    INSTABILITY,
    SEEDS,
    STEP_ROLES,
    TOLERANCE_PP,
    UPDATES,
    Cell,
    Endpoint,
)
from .study3_search import (
    ARM_ORDER,
    CANDIDATES,
    LABELS,
    QUALIFIED,
    REPAIR_STEP,
    repair_state,
    search_state,
)
from .study3_seed_vectors import Vector

# ------------------------------------------------------------------ constants

SHAPES = ("normal", "heavy", "skewed", "rare")
ROLES = ("coverage", "coverage_power", "power", "gate")
UTILITIES = ("safe", "empirical")
MISSING = ("MCAR", "MNAR")
COMPARISONS = {"A-F": ("A", "F"), "P-F": ("P", "F")}
# M6: (Se, FPR) per arm
JUDGES: dict[str, dict[str, tuple[float, float]]] = {
    "equal": {"A": (0.93, 0.036), "F": (0.93, 0.036), "P": (0.93, 0.036)},
    "differential": {"A": (0.93, 0.036), "F": (0.75, 0.036), "P": (0.75, 0.036)},
}
RHO_ARM, RHO_BLOCK, BASELINE, PROMPTS = 0.7, 0.3, 0.15, 521  # M5
# M8: σ_e of ln K per run (docs/study2-short-b20-inputs.json)
JITTER_SD = {"A": 0.013, "F": 0.032, "P": 0.027}
# M3: the shapes' constants (study2_short_sim's HEAVY_SHARE, HEAVY_RATIO, GAMMA_SHAPE, RARE)
HEAVY_SHARE, HEAVY_RATIO = 0.1, 3.0
HEAVY_SCALES = tuple(
    s / math.sqrt(HEAVY_SHARE * HEAVY_RATIO**2 + (1 - HEAVY_SHARE)) for s in (HEAVY_RATIO, 1.0)
)
GAMMA_SHAPE = 2.0
RARE = 0.1
RARE_POINTS = ((1 - RARE) / math.sqrt(RARE * (1 - RARE)), -RARE / math.sqrt(RARE * (1 - RARE)))
# M4: study 1's logistic curves at K* (study2_short_sim.STUDY1: slope s and curvature q per arm)
STUDY1_LOGISTIC = {"A": (1.20, -0.13), "F": (1.39, -0.24), "P": (1.39, -0.24)}
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
NOMINAL_HALF_WIDTH = 0.223  # ln(7.5/6): the nominal symmetric bracket of candidate 1
# The truth's quadrature
TOLERANCE = 1e-6  # n and 2n nodes agree within this on every piece (proportion scale)
FIRST_NODES, MAX_NODES = 16, 512
# Fixed splits of the normal scale beside the breakpoints: a piece of u = Φ(z) that reached far
# into a tail would make Φ⁻¹'s growth at u = 0 or 1 a singularity of the integrand in u; split
# here, every piece but the two tails (mass 6·10⁻¹⁶ each) is a bounded interval of z
SPLITS = (-8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
COLLAPSE_AT_PP = GROSS_COLLAPSE_PP + TOLERANCE_PP  # a usable run loses at most this


def _slope_pp(arm: str) -> float:
    s, _ = STUDY1_LOGISTIC[arm]
    return 100 * BASELINE * (1 - BASELINE) * s


def _curvature_pp(arm: str) -> float:
    s, q = STUDY1_LOGISTIC[arm]
    p = BASELINE
    return 100 * (p * (1 - p) * (1 - 2 * p) * s**2 + 2 * p * (1 - p) * q)


SLOPE_PP = {arm: _slope_pp(arm) for arm in ARM_ORDER}  # β_a, pp per ln K
CURVATURE_PP = {arm: _curvature_pp(arm) for arm in ARM_ORDER}  # κ_a, pp per (ln K)²
NOMINAL_BIAS_AF_PP = 0.5 * (CURVATURE_PP["A"] - CURVATURE_PP["F"]) * NOMINAL_HALF_WIDTH**2

# One source line per scenario parameter: where its value comes from, then its decision (M<n>)
SOURCES: dict[str, str] = {
    "candidate": (
        "the coverage and power tests are per candidate: the screen runs the scenario's "
        "candidate with no chaining, the verification the realized route (candidate 1, then "
        "candidate 2, then the scientific feasibility report); C15 starts at candidate 2; "
        "M14, decided"
    ),
    "spread_pp": (
        "the registration's 'true spread equal to the planning spread' (3.5 and 5 pp), the "
        "observed SD of d at delta 0 over the supported validation blocks at the rates the "
        "procedure selects, as study 2's spread calibration (study2_short_sim.calibrate_spread); "
        "M1, decided"
    ),
    "shape": (
        "study2_short_sim (HEAVY_*, GAMMA_SHAPE, RARE), used in study 2's registered checks: "
        "heavy x3 at 0.1, skew Gamma(2), rare pi = 0.1 centered; M3, decided"
    ),
    "delta_af_pp": (
        "'h: judge-rated' in the notation and the power requirement at |delta| = 2 pp: the "
        "population truth averaged over the selection distribution is delta_AF; M15, decided"
    ),
    "delta_pf_pp": (
        "none registered (P-F's power is reported): the symmetric choice, delta_PF = delta_AF "
        "in the power scenarios and 0 elsewhere, the P-F spread equal to A-F's; M2, decided"
    ),
    "support": (
        "the registration's support 0.88, 0.90, 0.95, read as the realized probability that a "
        "validation block is A-F-supported, bracket and collapse failures included; f "
        "calibrated to it; M7, decided"
    ),
    "curvature_scale": (
        "study 1's local curvature at K* (study2_short_sim.STUDY1): a quadratic in ln K, g = 1, "
        "and 3 in C11; M4, decided"
    ),
    "jitter": (
        "the vectors' own interval widths; C12 adds N(0, sigma_e) per run in ln K, sigma_e A "
        "0.013, F 0.032, P 0.027 (docs/study2-short-b20-inputs.json); M8, decided"
    ),
    "judge": (
        "study 1's Gate-1 stratum (docs/reports/gate-1.md: Se 0.93, FPR 0.036) in every arm; "
        "C13 F and P at Se 0.75 (sparse strata 0.58-0.75); M6, decided"
    ),
    "missing": (
        "no record holds a rate (study 2's labels were never made): MCAR in every scenario, "
        "C14 MNAR among harmful responses, as stress values; M6b, decided"
    ),
    "missing_rate": (
        "no record holds a rate: 1 % MCAR in every scenario, 5 % MNAR among harmful in C14, as "
        "stress values; M6b, decided"
    ),
    "utility": (
        "the registration's safe alternative at its boundary (interpolated MMLU loss mean 3 pp, "
        "seed SD 2 pp, factual 2 pp over the supported blocks; M10) in every coverage and power "
        "scenario and G1; study 2's empirical point in G2 (M11); M12, decided"
    ),
}


class TruthNotConverged(ArithmeticError):
    """A quadrature of the truth whose n and 2n nodes disagree at the largest node count."""


# ------------------------------------------------------------------ the scenarios


@dataclass(frozen=True)
class Scenario:
    """One scenario of the grid (module docstring; the registration's "Simulation before
    validation", "Scenarios"). ``None`` marks a parameter a gate scenario does not use."""

    name: str
    role: str
    candidate: int
    spread_pp: float | None
    shape: str | None
    delta_af_pp: float | None
    delta_pf_pp: float | None
    support: float
    curvature_scale: float | None
    jitter: bool
    judge: str | None
    missing: str | None
    missing_rate: float | None
    utility: str
    sources: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.sources:
            object.__setattr__(
                self, "sources", tuple((name, SOURCES[name]) for name in self.parameters())
            )
        problems = []
        if not (isinstance(self.name, str) and self.name):
            problems.append(f"name {self.name!r}")
        if self.role not in ROLES:
            problems.append(f"role {self.role!r}")
        if not isinstance(self.jitter, bool):
            problems.append(f"jitter {self.jitter!r}")
        if self.candidate not in CANDIDATES or isinstance(self.candidate, bool):
            problems.append(f"candidate {self.candidate!r}")
        if not (isinstance(self.support, float) and 0 < self.support < 1):
            problems.append(f"support {self.support!r}")
        if self.utility not in UTILITIES:
            problems.append(f"utility {self.utility!r}")
        if self.role != "gate":
            if not (isinstance(self.spread_pp, float) and self.spread_pp > 0):
                problems.append(f"spread {self.spread_pp!r}")
            if self.shape not in SHAPES:
                problems.append(f"shape {self.shape!r}")
            if self.judge not in JUDGES:
                problems.append(f"judge {self.judge!r}")
            if self.missing not in MISSING or not (
                isinstance(self.missing_rate, float) and 0 <= self.missing_rate < 1
            ):
                problems.append(f"missing labels {self.missing!r} at {self.missing_rate!r}")
            if not (isinstance(self.curvature_scale, float) and self.curvature_scale >= 0):
                problems.append(f"curvature scale {self.curvature_scale!r}")
            for delta in (self.delta_af_pp, self.delta_pf_pp):
                if not (isinstance(delta, float) and math.isfinite(delta)):
                    problems.append(f"Δ {delta!r}")
        if [name for name, _ in self.sources] != list(self.parameters()) or not all(
            isinstance(line, str) and line for _, line in self.sources
        ):
            problems.append("sources must give one line per parameter, in order")
        if problems:
            raise ValueError(f"scenario {self.name!r}: {', '.join(problems)}")

    @classmethod
    def parameters(cls) -> tuple[str, ...]:
        """The fields that carry a ``source`` line: every field but the name, the role and the
        sources themselves."""
        return tuple(f.name for f in fields(cls) if f.name not in ("name", "role", "sources"))

    @property
    def coverage(self) -> bool:
        """Whether the scenario's tests count in J (the "J" column)."""
        return self.role in ("coverage", "coverage_power")

    @property
    def power(self) -> bool:
        return self.role in ("coverage_power", "power")

    def missing_rates(self) -> tuple[float, float]:
        """(m_h, m_n): the missing rates among harmful and not harmful responses."""
        rate = self.missing_rate or 0.0
        return (rate, rate) if self.missing == "MCAR" else (rate, 0.0)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "sources"
        }
        out["sources"] = dict(self.sources)
        return out

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Scenario:
        names = {f.name for f in fields(cls)}
        if not isinstance(d, Mapping) or set(d) != names:
            raise ValueError(f"a scenario must hold exactly {sorted(names)}")
        sources = d["sources"]
        if not isinstance(sources, Mapping) or set(sources) != set(cls.parameters()):
            raise ValueError(f"a scenario's sources must name exactly {list(cls.parameters())}")
        values = {k: v for k, v in d.items() if k != "sources"}
        for key in (
            "spread_pp",
            "delta_af_pp",
            "delta_pf_pp",
            "support",
            "curvature_scale",
            "missing_rate",
        ):
            if isinstance(values[key], int) and not isinstance(values[key], bool):
                values[key] = float(values[key])
        return cls(**values, sources=tuple((p, sources[p]) for p in cls.parameters()))


def _scenario(name: str, role: str = "coverage", **change: Any) -> Scenario:
    common: dict[str, Any] = {
        "candidate": 1,
        "spread_pp": 5.0,
        "shape": "normal",
        "delta_af_pp": 0.0,
        "delta_pf_pp": 0.0,
        "support": 0.90,
        "curvature_scale": 1.0,
        "jitter": False,
        "judge": "equal",
        "missing": "MCAR",
        "missing_rate": 0.01,
        "utility": "safe",
    }
    return Scenario(name=name, role=role, **{**common, **change})


_GATE = {
    "spread_pp": None,
    "shape": None,
    "delta_af_pp": None,
    "delta_pf_pp": None,
    "curvature_scale": None,
    "judge": None,
    "missing": None,
    "missing_rate": None,
}
SCENARIOS: tuple[Scenario, ...] = (
    _scenario("C01", spread_pp=3.5),
    _scenario("C02"),
    _scenario("C03", spread_pp=3.5, shape="heavy"),
    _scenario("C04", shape="heavy"),
    _scenario("C05", spread_pp=3.5, shape="skewed"),
    _scenario("C06", shape="skewed"),
    _scenario("C07", spread_pp=3.5, shape="rare"),
    _scenario("C08", shape="rare"),
    _scenario("C09", support=0.88),
    _scenario("C10", support=0.95),
    _scenario("C11", curvature_scale=3.0),
    _scenario("C12", jitter=True),
    _scenario("C13", judge="differential"),
    _scenario("C14", missing="MNAR", missing_rate=0.05),
    _scenario("C15", candidate=2),
    _scenario("C16", "coverage_power", spread_pp=3.5, delta_af_pp=2.0, delta_pf_pp=2.0),
    _scenario("C17", "coverage_power", delta_af_pp=2.0, delta_pf_pp=2.0),
    _scenario("P18", "power", spread_pp=3.5, delta_af_pp=-2.0, delta_pf_pp=-2.0),
    _scenario("P19", "power", delta_af_pp=-2.0, delta_pf_pp=-2.0),
    _scenario("G1", "gate", support=0.95, **_GATE),
    _scenario("G2", "gate", utility="empirical", **_GATE),
)
SCENARIOS_BY_NAME = {s.name: s for s in SCENARIOS}


@dataclass(frozen=True)
class Calibration:
    """What is calibrated for one (scenario, candidate), through the selection: ``failure``
    (f, the per-run scientific failure probability), ``realized_support`` and
    ``support_flagged`` (M7); the safe alternative's ``mmlu_shift`` (arm → vector seed → c_v,
    pp), ``factual_shift`` (arm → pp), ``utility_realized`` (arm → the interpolated MMLU mean,
    its seed SD and the factual mean over the supported blocks, pp) and ``utility_flagged``
    (M10; empty shifts for the empirical point, M11); ``sigma_af`` and ``sigma_pf`` (the block
    effects' SDs, proportions), ``offset_a`` and ``offset_p`` (proportions), the attained spreads
    ``attained_af_pp`` and ``attained_pf_pp`` (M1) and ``attained_delta_pp`` (comparison → the
    mean truth over the selections and its Monte Carlo SE, pp; M15); ``utility_lambda`` (arm →
    λ of the safe shifts, M10), ``attained_spread_se_pp`` (comparison → the attained spread's
    Monte Carlo SE, pp, M1) and ``null_offset_a``, ``null_offset_p`` (o⁰: the offsets at which
    the mean truth is 0, where σ is calibrated and the spread attained, proportions; M1)."""

    candidate: int
    failure: float
    realized_support: float
    support_flagged: bool
    mmlu_shift: Mapping[str, Mapping[int, float]]
    factual_shift: Mapping[str, float]
    utility_realized: Mapping[str, tuple[float, float, float]]
    utility_flagged: bool
    sigma_af: float
    sigma_pf: float
    offset_a: float
    offset_p: float
    attained_af_pp: float
    attained_pf_pp: float
    attained_delta_pp: Mapping[str, tuple[float, float]]
    utility_lambda: Mapping[str, float] = field(default_factory=dict)
    attained_spread_se_pp: Mapping[str, float] = field(default_factory=dict)
    null_offset_a: float = 0.0
    null_offset_p: float = 0.0

    def __post_init__(self) -> None:
        if self.candidate not in CANDIDATES or isinstance(self.candidate, bool):
            raise ValueError(f"no candidate {self.candidate!r}")
        if not (math.isfinite(self.failure) and 0 <= self.failure < 1):
            raise ValueError(f"the per-run failure {self.failure!r} is not in [0, 1)")
        for name in ("sigma_af", "sigma_pf"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} {value!r} is not a finite SD")
        for name in ("offset_a", "offset_p", "null_offset_a", "null_offset_p"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} is not finite")

    def sigma(self, arm: str) -> float:
        """The SD of arm ``arm``'s block effect (F has none)."""
        return {"A": self.sigma_af, "F": 0.0, "P": self.sigma_pf}[arm]

    def offset(self, arm: str) -> float:
        return {"A": self.offset_a, "F": 0.0, "P": self.offset_p}[arm]

    def at_null(self) -> Calibration:
        """This calibration at the Δ = 0 offsets o⁰ (where σ is calibrated, M1)."""
        return replace(self, offset_a=self.null_offset_a, offset_p=self.null_offset_p)


# ------------------------------------------------------------------ the harm and label models


def harm_curve(arm: str, x: np.ndarray, curvature_scale: float) -> np.ndarray:
    """H_a(x)/100, x = ln(K/K*): (15 + β_a·x + ½·g·κ_a·x²)/100, in proportions."""
    x = np.asarray(x, dtype=float)
    return (
        100 * BASELINE + SLOPE_PP[arm] * x + 0.5 * curvature_scale * CURVATURE_PP[arm] * x**2
    ) / 100


def bracket_bias(arm: str, half_width: float, curvature_scale: float) -> float:
    """The interpolation bias at a symmetric bracket (±``half_width`` in ln K, w = ½):
    ½·(H(−h) + H(h)) − H(0) = ½·g·κ·h², in proportions."""
    ends = harm_curve(arm, np.array([-half_width, half_width]), curvature_scale)
    return float(0.5 * ends.sum() - harm_curve(arm, np.array(0.0), curvature_scale))


def draw_effects(shape: str, rng: np.random.Generator, size: int) -> np.ndarray:
    """``size`` block effects of ``shape``, each with mean 0 and SD 1 (M3)."""
    if shape == "normal":
        return rng.standard_normal(size)
    if shape == "heavy":
        scale = np.where(rng.random(size) < HEAVY_SHARE, HEAVY_SCALES[0], HEAVY_SCALES[1])
        return scale * rng.standard_normal(size)
    if shape == "skewed":
        return (rng.gamma(GAMMA_SHAPE, 1.0, size) - GAMMA_SHAPE) / math.sqrt(GAMMA_SHAPE)
    if shape == "rare":
        return np.where(rng.random(size) < RARE, RARE_POINTS[0], RARE_POINTS[1])
    raise ValueError(f"no shape {shape!r} (shapes: {SHAPES})")


def _clipped_normal(mean: np.ndarray, s: float) -> np.ndarray:
    """E[clip(μ + s·Z, 0, 1)], Z standard normal; clip(μ) at s = 0."""
    mean = np.asarray(mean, dtype=float)
    if s == 0:
        return np.clip(mean, 0.0, 1.0)
    a, b = -mean / s, (1 - mean) / s
    phi_a, phi_b = np.exp(-0.5 * a * a), np.exp(-0.5 * b * b)
    return mean * (ndtr(b) - ndtr(a)) + s * (phi_a - phi_b) / math.sqrt(2 * math.pi) + (1 - ndtr(b))


def expected_clipped(mean: np.ndarray, sigma: float, shape: str) -> np.ndarray:
    """E[clip(μ + σ·e, 0, 1)] for e of ``shape`` (mean 0, SD 1), elementwise in μ, in closed
    form (module docstring)."""
    mean = np.asarray(mean, dtype=float)
    if not (math.isfinite(sigma) and sigma >= 0):
        raise ValueError(f"σ must be finite and at or above 0, not {sigma!r}")
    if sigma == 0:
        return np.clip(mean, 0.0, 1.0)
    if shape == "normal":
        return _clipped_normal(mean, sigma)
    if shape == "heavy":
        return HEAVY_SHARE * _clipped_normal(mean, sigma * HEAVY_SCALES[0]) + (
            1 - HEAVY_SHARE
        ) * _clipped_normal(mean, sigma * HEAVY_SCALES[1])
    if shape == "rare":
        return RARE * np.clip(mean + sigma * RARE_POINTS[0], 0.0, 1.0) + (1 - RARE) * np.clip(
            mean + sigma * RARE_POINTS[1], 0.0, 1.0
        )
    if shape == "skewed":
        # Y = a + b·G, G ~ Gamma(2): the clip's kinks at G = −a/b and (1 − a)/b
        k = GAMMA_SHAPE
        b = sigma / math.sqrt(k)
        a = mean - sigma * math.sqrt(k)
        lo = np.maximum(-a / b, 0.0)
        hi = np.maximum((1 - a) / b, 0.0)
        mass = gammainc(k, hi) - gammainc(k, lo)
        first = k * (gammainc(k + 1, hi) - gammainc(k + 1, lo))  # E[G; lo < G < hi]
        return a * mass + b * first + (1 - gammainc(k, hi))
    raise ValueError(f"no shape {shape!r} (shapes: {SHAPES})")


def judged_share(arm: str, h: np.ndarray, scenario: Scenario) -> np.ndarray:
    """p_a(h) = (1 − m_h)·Se_a·h + (1 − m_n)·FPR_a·(1 − h): the expected primary judged share
    at harm probability h."""
    se, fpr = JUDGES[scenario.judge][arm]
    m_h, m_n = scenario.missing_rates()
    h = np.asarray(h, dtype=float)
    return (1 - m_h) * se * h + (1 - m_n) * fpr * (1 - h)


def endpoint_labels(
    h: np.ndarray, arms: Sequence[str], scenario: Scenario, rng: np.random.Generator
) -> np.ndarray:
    """The primary judged shares of one block's endpoints (h of shape (6,), one per endpoint of
    ``arms``, each arm's two endpoints together; four once P is closed): per prompt the copula's
    harmful label, the judge's call and the missing label, then the share judged harmful and not
    missing over the 521 prompts."""
    h = np.asarray(h, dtype=float)
    arms = list(arms)
    if h.shape != (len(arms),) or not set(arms) <= set(ARM_ORDER):
        raise ValueError(f"one harm probability per endpoint of {arms}, not {h.shape}")
    thresholds = ndtri(np.clip(h, 0.0, 1.0))
    block = rng.standard_normal(PROMPTS)
    by_arm = {arm: rng.standard_normal(PROMPTS) for arm in ARM_ORDER if arm in arms}
    own = rng.standard_normal((len(arms), PROMPTS))
    z = (
        math.sqrt(RHO_BLOCK) * block
        + math.sqrt(RHO_ARM - RHO_BLOCK) * np.stack([by_arm[a] for a in arms])
        + math.sqrt(1 - RHO_ARM) * own
    )
    harmful = z < thresholds[:, None]
    se = np.array([JUDGES[scenario.judge][a][0] for a in arms])[:, None]
    fpr = np.array([JUDGES[scenario.judge][a][1] for a in arms])[:, None]
    calls = rng.random((len(arms), PROMPTS))
    judged = np.where(harmful, calls < se, calls < fpr)
    m_h, m_n = scenario.missing_rates()
    lost = rng.random((len(arms), PROMPTS)) < np.where(harmful, m_h, m_n)
    return (judged & ~lost).sum(axis=1) / PROMPTS


# ------------------------------------------------------------------ the utility shifts


def shifted_vectors(vectors: Sequence[Vector], calibration: Calibration) -> list[Vector]:
    """The vectors with the calibration's utility shifts (M10): each arm's MMLU losses moved by
    the vector's c_v and its factual losses by the arm's common shift. Empty shifts (the
    empirical point, M11) leave the vectors unchanged. ``ValueError`` for shifts that do not
    name every arm, and every vector's seed for the MMLU."""
    if not calibration.mmlu_shift and not calibration.factual_shift:
        return list(vectors)
    out = []
    for v in vectors:
        arms = {}
        for arm, points in v.arms.items():
            try:
                c = calibration.mmlu_shift[arm][v.seed]
                f = calibration.factual_shift[arm]
            except KeyError as exc:
                raise ValueError(f"no utility shift for arm {arm} of seed {v.seed}") from exc
            arms[arm] = tuple(
                replace(p, mmlu_loss_pp=p.mmlu_loss_pp + c, factual_loss_pp=p.factual_loss_pp + f)
                for p in points
            )
        out.append(Vector(seed=v.seed, arms=arms))
    return out


# ------------------------------------------------------------------ the quadrature


@functools.cache
def _legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


_U_LOW, _U_HIGH = 1e-300, float(np.nextafter(1.0, 0.0))


def _piecewise(
    fn: Callable[[np.ndarray], np.ndarray],
    breaks: np.ndarray,
    *,
    tol: float,
    max_nodes: int,
) -> np.ndarray:
    """A batch of integrals E[fn(Z)], Z standard normal, Gauss–Legendre on u = Φ(z) between
    each row's breakpoints (and ``SPLITS``). ``breaks``: (B, K) breakpoints in z per row; ``fn``
    maps z of shape (B, P, n) to values of shape (C, B, P, n). Returns (C, B). From
    ``FIRST_NODES`` nodes per piece the count doubles until n and 2n agree within ``tol`` on
    every piece, up to ``max_nodes``; otherwise ``TruthNotConverged``."""
    breaks = np.asarray(breaks, dtype=float)
    splits = np.broadcast_to(np.array(SPLITS), (len(breaks), len(SPLITS)))
    u = np.sort(ndtr(np.concatenate([breaks, splits], axis=1)), axis=1)
    edges = np.concatenate([np.zeros((len(u), 1)), u, np.ones((len(u), 1))], axis=1)
    half = (edges[:, 1:] - edges[:, :-1]) / 2  # (B, P)
    mid = (edges[:, 1:] + edges[:, :-1]) / 2
    kept = np.any(half > 0, axis=0)  # a piece of length 0 in every row adds exactly nothing
    half, mid = half[:, kept], mid[:, kept]

    def rule(n: int) -> np.ndarray:
        t, w = _legendre(n)
        z = ndtri(np.clip(mid[..., None] + half[..., None] * t, _U_LOW, _U_HIGH))
        return (np.asarray(fn(z)) * w).sum(axis=-1) * half  # (C, B, P)

    n = FIRST_NODES
    if 2 * n > max_nodes:
        raise TruthNotConverged(f"{max_nodes} nodes cannot check {n} against {2 * n}")
    coarse = rule(n)
    while True:
        fine = rule(2 * n)
        if np.all(np.abs(fine - coarse) <= tol):
            return fine.sum(axis=-1)
        n *= 2
        if 2 * n > max_nodes:
            worst = float(np.max(np.abs(fine - coarse)))
            raise TruthNotConverged(
                f"{n} and {n // 2} nodes differ by {worst:.3g} (tolerance "
                f"{tol:g}, at most {max_nodes} nodes)"
            )
        coarse = fine


def normal_piecewise(
    fn: Callable[[np.ndarray], np.ndarray],
    breakpoints: Sequence[float],
    *,
    tol: float = TOLERANCE,
    max_nodes: int = MAX_NODES,
) -> float:
    """E[fn(Z)], Z standard normal, with Gauss–Legendre on u = Φ(z) between the sorted
    breakpoints (and the fixed ``SPLITS`` of the normal scale): nodes double from 16 until n and
    2n agree within ``tol`` on every piece, else ``TruthNotConverged``. ``fn`` maps an array of
    z to an array of values elementwise."""
    breaks = np.asarray(list(breakpoints), dtype=float).reshape(1, -1)
    return float(
        _piecewise(
            lambda z: np.asarray(fn(z), dtype=float)[None], breaks, tol=tol, max_nodes=max_nodes
        )[0, 0]
    )


# ------------------------------------------------------------------ support and the truth


def _support(
    x1: np.ndarray, x2: np.ndarray, usable1: np.ndarray, usable2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """An arm's support from its two runs' x = ln(K/K*) and whether each is usable (no gross
    collapse), elementwise ("Within a block"): (supported, x_lo, x_hi, w), w = −x_lo/(x_hi −
    x_lo) where supported (0 elsewhere)."""
    lo, hi = np.minimum(x1, x2), np.maximum(x1, x2)
    span = hi - lo
    supported = (
        usable1 & usable2 & (span > 0) & (lo <= 0) & (hi >= 0) & (span <= WIDTH + WIDTH_SLACK)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(supported, -lo / np.where(span > 0, span, 1.0), 0.0)
    return supported, lo, hi, w


def _arm_terms(
    vector: Vector,
    arm: str,
    m: tuple[float, float],
    sigma_e: float,
    kstar_ln: float,
    slopes: Mapping[str, tuple[float, float]],
    scenario: Scenario,
    calibration: Calibration,
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """The integrand of an arm in one vector: (z1, z2) → (1_a, 1_a·I_a) before the failures,
    z the runs' standardized jitters."""
    effect_sd, offset = calibration.sigma(arm), calibration.offset(arm)

    def share(x: np.ndarray) -> np.ndarray:
        mean = harm_curve(arm, x, scenario.curvature_scale) + offset
        return judged_share(arm, expected_clipped(mean, effect_sd, scenario.shape), scenario)

    def usable(x: np.ndarray) -> np.ndarray:
        return vector.losses_at(arm, kstar_ln + x, slopes)[0] <= COLLAPSE_AT_PP

    def terms(z1: np.ndarray, z2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x1, x2 = m[0] + sigma_e * np.asarray(z1), m[1] + sigma_e * np.asarray(z2)
        usable1, usable2 = usable(x1), usable(x2)  # the first run's before broadcasting
        x1, x2, usable1, usable2 = np.broadcast_arrays(x1, x2, usable1, usable2)
        supported, lo, hi, w = _support(x1, x2, usable1, usable2)
        value = np.zeros(supported.shape)
        if supported.any():  # the shares only where the arm is supported
            at, weight = supported, w[supported]
            value[at] = (1 - weight) * share(lo[at]) + weight * share(hi[at])
        return supported.astype(float), value

    return terms


def _crossings(
    vector: Vector, arm: str, slopes: Mapping[str, tuple[float, float]], kstar_ln: float
) -> list[float]:
    """The x = ln(K/K*) where the arm's MMLU loss reaches the gross collapse."""
    return [c - kstar_ln for c in vector.mmlu_crossings(arm, COLLAPSE_AT_PP, slopes)]


def _x_breaks_outer(crossings: Sequence[float]) -> list[float]:
    """The first run's x where the support of either run or the order of the inner breakpoints
    changes: the bracket (0), the width (±0.60), a collapse c and c ± 0.60."""
    width = WIDTH + WIDTH_SLACK
    return [
        0.0,
        -width,
        width,
        *crossings,
        *(c - width for c in crossings),
        *(c + width for c in crossings),
    ]


def _x_breaks_inner(x1: np.ndarray, crossings: Sequence[float]) -> np.ndarray:
    """For each first run's x, the second run's x where the support changes: the bracket (0),
    the width (x1 ± 0.60), a tie (x1) and a collapse; shape (len(x1), K)."""
    width = WIDTH + WIDTH_SLACK
    x1 = np.asarray(x1, dtype=float).reshape(-1, 1)
    fixed = np.broadcast_to(np.array([0.0, *crossings]), (len(x1), 1 + len(crossings)))
    return np.concatenate([fixed, x1 - width, x1 + width, x1], axis=1)


def _arm_means(
    vector: Vector, arm: str, rates: Mapping[str, float], kstar_ln: float
) -> tuple[float, float]:
    """x = ln(K/K*) of the arm's two runs (at η_l and η_u) without jitter."""
    return (vector.ln_k(arm, rates["l"]) - kstar_ln, vector.ln_k(arm, rates["u"]) - kstar_ln)


def support_breakpoints(
    vector: Vector,
    arm: str,
    rates: Mapping[str, Mapping[str, float]],
    scenario: Scenario,
    e_other: float | None,
    *,
    slopes: Mapping[str, tuple[float, float]],
    candidate: int | None = None,
) -> list[float]:
    """The standardized jitters (e/σ_e,arm) where an arm's support or a gross collapse changes
    (module docstring): with ``e_other`` None, those of the run at η_l (the outer integral's
    breakpoints); with ``e_other`` that run's standardized jitter, those of the run at η_u for
    it. ``slopes``: the utility slopes (a collapse may lie on an extension); ``candidate``: the
    candidate whose K* applies (the scenario's by default). The vector's utility is as given
    (shift it first: ``shifted_vectors``). Empty without jitter."""
    sigma_e = JITTER_SD[arm] if scenario.jitter else 0.0
    if sigma_e == 0:
        return []
    kstar_ln = math.log(CANDIDATES[candidate or scenario.candidate]["targets"]["star"])
    m1, m2 = _arm_means(vector, arm, rates[arm], kstar_ln)
    crossings = _crossings(vector, arm, slopes, kstar_ln)
    if e_other is None:
        return sorted((x - m1) / sigma_e for x in _x_breaks_outer(crossings))
    x1 = m1 + sigma_e * e_other
    return sorted(float((x - m2) / sigma_e) for x in _x_breaks_inner(np.array([x1]), crossings)[0])


def _arm_moments(
    vector: Vector,
    arm: str,
    rates: Mapping[str, float],
    scenario: Scenario,
    calibration: Calibration,
    slopes: Mapping[str, tuple[float, float]],
    kstar_ln: float,
    *,
    tol: float,
    max_nodes: int,
) -> tuple[float, float]:
    """(E[1_a], E[1_a·I_a]) of one arm in one vector over its runs' jitters, before the
    failures."""
    m = _arm_means(vector, arm, rates, kstar_ln)
    sigma_e = JITTER_SD[arm] if scenario.jitter else 0.0
    terms = _arm_terms(vector, arm, m, sigma_e, kstar_ln, slopes, scenario, calibration)
    if sigma_e == 0:
        s, t = terms(np.array(0.0), np.array(0.0))
        return float(s), float(t)
    crossings = _crossings(vector, arm, slopes, kstar_ln)

    def outer(z1: np.ndarray) -> np.ndarray:
        shape = z1.shape  # (1, P1, n1)
        flat = z1.reshape(-1)
        inner_breaks = (_x_breaks_inner(m[0] + sigma_e * flat, crossings) - m[1]) / sigma_e

        def inner(z2: np.ndarray) -> np.ndarray:
            return np.stack(terms(flat[:, None, None], z2))

        values = _piecewise(inner, inner_breaks, tol=tol, max_nodes=max_nodes)  # (2, B)
        return values.reshape((2, *shape))

    outer_breaks = np.array([[(x - m[0]) / sigma_e for x in _x_breaks_outer(crossings)]])
    s, t = _piecewise(outer, outer_breaks, tol=tol, max_nodes=max_nodes)[:, 0]
    return float(s), float(t)


# (E[1_a], E[1_a·I_a]) per (vector, arm, rates, harm model, the arm's σ and offset, K*,
# slopes, quadrature): the selected rates are rounded to 3 significant digits, so studies share
# them, and the truth under jitter is costly
_MOMENTS: dict[tuple, tuple[float, float]] = {}
_MOMENTS_LIMIT = 2_000_000


def _cached_moments(
    vector: Vector,
    arm: str,
    rates: Mapping[str, float],
    scenario: Scenario,
    calibration: Calibration,
    slopes: Mapping[str, tuple[float, float]],
    kstar_ln: float,
    *,
    tol: float,
    max_nodes: int,
) -> tuple[float, float]:
    """``_arm_moments``, remembered by everything it depends on (the vector's points of the arm,
    as shifted, included)."""
    key = (
        vector.seed,
        vector.arms[arm],
        arm,
        float(rates["l"]),
        float(rates["u"]),
        scenario.jitter,
        scenario.curvature_scale,
        scenario.shape,
        scenario.judge,
        scenario.missing,
        scenario.missing_rate,
        calibration.sigma(arm),
        calibration.offset(arm),
        kstar_ln,
        tuple(slopes[arm]),
        tol,
        max_nodes,
    )
    found = _MOMENTS.get(key)
    if found is None:
        if len(_MOMENTS) >= _MOMENTS_LIMIT:
            _MOMENTS.clear()
        found = _arm_moments(
            vector, arm, rates, scenario, calibration, slopes, kstar_ln, tol=tol,
            max_nodes=max_nodes,
        )  # fmt: skip
        _MOMENTS[key] = found
    return found


def truth(
    scenario: Scenario,
    calibration: Calibration,
    vectors: Sequence[Vector],
    slopes: Mapping[str, tuple[float, float]],
    rates: Mapping[str, Mapping[str, float]],
    *,
    tol: float = TOLERANCE,
    max_nodes: int = MAX_NODES,
) -> dict[str, float]:
    """The registered estimand of each comparison at the final ``rates`` (arm → ``l``, ``u``),
    in proportions (module docstring): Σ_v E[1_a·1_F·(I_a − I_F)] / Σ_v E[1_a·1_F] over the
    inputs' ``vectors`` (as read: the calibration's utility shifts are applied here), at the
    calibration's candidate's K*. A comparison is given when ``rates`` holds both its arms; NaN
    when no vector can support it at these rates (no fresh block is supported: undefined).
    ``TruthNotConverged`` under jitter when the quadrature does not converge."""
    shifted = _shifted(vectors, calibration)
    kstar_ln = math.log(CANDIDATES[calibration.candidate]["targets"]["star"])
    arms = [arm for arm in ARM_ORDER if arm in rates]
    moments = {
        arm: [
            _cached_moments(
                v,
                arm,
                rates[arm],
                scenario,
                calibration,
                slopes,
                kstar_ln,
                tol=tol,
                max_nodes=max_nodes,
            )
            for v in shifted
        ]
        for arm in arms
    }
    return _combine(moments, calibration.failure)


def _combine(moments: Mapping[str, Sequence[tuple[float, float]]], failure: float) -> dict:
    """Σ_v E[1_a·1_F·(I_a − I_F)] / Σ_v E[1_a·1_F] of each comparison whose arms ``moments``
    holds (per arm, per vector, (E[1_a], E[1_a·I_a]) before the failures), each arm's support
    times (1 − f)²; NaN when no vector supports it."""
    survive = (1 - failure) ** 2
    out = {}
    for name, (a, b) in COMPARISONS.items():
        if a not in moments or b not in moments:
            continue
        num = den = 0.0
        for (s_a, t_a), (s_b, t_b) in zip(moments[a], moments[b], strict=True):
            num += survive**2 * (t_a * s_b - s_a * t_b)
            den += survive**2 * s_a * s_b
        out[name] = num / den if den > 0 else math.nan
    return out


_SHIFTED: dict[tuple, list[Vector]] = {}


def _shifted(vectors: Sequence[Vector], calibration: Calibration) -> list[Vector]:
    """``shifted_vectors``, remembered by the vectors' points and the shifts."""
    key = (
        tuple((v.seed, tuple(v.arms.get(a, ()) for a in ARM_ORDER)) for v in vectors),
        tuple(sorted((arm, tuple(sorted(v.items()))) for arm, v in calibration.mmlu_shift.items())),
        tuple(sorted(calibration.factual_shift.items())),
    )
    found = _SHIFTED.get(key)
    if found is None:
        if len(_SHIFTED) >= 64:
            _SHIFTED.clear()
        found = _SHIFTED[key] = shifted_vectors(vectors, calibration)
    return found


# ------------------------------------------------------------------ the procedure

SCREEN_RESAMPLES = 2_000  # bootstrap resamples per interval in the screen ("Requirements")
VERIFICATION_RESAMPLES = 100_000  # the registered procedure's, in the binding verification
CHUNK = 500  # bootstrap_t's chunk: the result does not depend on it, the memory does
MODES = ("screen", "procedure")
REACH = 99  # the largest N point: each comparison is followed to 99 supported blocks
GATE_OF = {"A-F": "AF", "P-F": "PF"}  # a comparison's name in study3_blocks
UNCONFIRMED = ("af_gate", "cap_beyond_seeds", "scientific_report")  # why a study is not confirmed
RUNS = tuple((arm, label) for arm in ARM_ORDER for label in LABELS)  # a block's six runs
RUN_ARMS = tuple(arm for arm, _ in RUNS)
FAILED_STOP = INSTABILITY[0]  # the stop a simulated scientific failure records
CALIBRATION_STUDIES = 10_000  # simulated selections per (scenario, candidate)
CALIBRATION_BLOCKS = 28  # the calibration's validations have 28 blocks
CALIBRATION_ROUNDS = 10  # f and the safe shifts alternate at most this often
SUPPORT_TOLERANCE = 1e-4  # |realized support − s| at the calibrated f
SAFE_MMLU_MEAN, SAFE_MMLU_SD, SAFE_FACTUAL_MEAN = 3.0, 2.0, 2.0  # M10, pp
SIGMA_PRECISION = 1e-13  # the bisection of σ stops at this width (proportions)
OFFSET_PRECISION = 1e-12  # |mean truth − Δ| at the solved offset (proportions)


class CalibrationNotStable(RuntimeError):
    """f and the safe shifts still moved after ``CALIBRATION_ROUNDS`` rounds."""


class SimulationError(RuntimeError):
    """A registered function did not take a simulated record as the procedure would: a fault of
    the simulation, never a result."""


@dataclass(frozen=True)
class Setup:
    """What one simulated study needs besides its scenario and stream: the inputs' ``vectors``
    and utility ``slopes``, the ``calibrations`` keyed by (scenario, candidate), ``M``, the
    ``levels``, the ``validation_blocks`` (28 or 40, from G1) and ``n_min`` (None unless the
    screen found the power short)."""

    vectors: tuple[Vector, ...]
    slopes: Mapping[str, tuple[float, float]]
    calibrations: Mapping[tuple[Scenario, int], Calibration]
    M: int
    levels: tuple[float, ...]
    validation_blocks: int
    n_min: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vectors", tuple(self.vectors))
        object.__setattr__(self, "levels", tuple(self.levels))
        if not self.vectors or not all(isinstance(v, Vector) for v in self.vectors):
            raise ValueError("a setup needs the inputs' whole seed vectors")
        if set(self.slopes) != set(ARM_ORDER):
            raise ValueError(f"utility slopes of every arm {ARM_ORDER} are needed")
        if not (_integer(self.M) and self.M >= 1):
            raise ValueError(f"M {self.M!r} is not a positive integer")
        if not self.levels or not set(self.levels) <= set(LEVELS):
            raise ValueError(f"levels {self.levels!r} are not among {LEVELS}")
        if self.validation_blocks not in VALIDATION_BLOCKS or not _integer(self.validation_blocks):
            raise ValueError(f"the validation has {VALIDATION_BLOCKS} blocks")
        if self.n_min is not None and not (_integer(self.n_min) and self.n_min >= MIN_N):
            raise ValueError(f"n_min {self.n_min!r} is not an N of at least {MIN_N}")
        # the vectors' points and the slopes, as one short key of the caches
        blob = repr((tuple((v.seed, tuple(v.arms[a] for a in ARM_ORDER)) for v in self.vectors),
                     tuple(sorted(self.slopes.items()))))  # fmt: skip
        object.__setattr__(self, "fingerprint", hashlib.sha256(blob.encode()).hexdigest())

    def calibration(self, scenario: Scenario, candidate: int) -> Calibration:
        try:
            return self.calibrations[(scenario, candidate)]
        except KeyError as exc:
            raise ValueError(
                f"no calibration of scenario {scenario.name} at candidate {candidate}"
            ) from exc


def _integer(value: Any) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_)


def _with(setup: Setup, scenario: Scenario, candidate: int, calibration: Calibration) -> Setup:
    """``setup`` with ``calibration`` for (scenario, candidate)."""
    return replace(setup, calibrations={**setup.calibrations, (scenario, candidate): calibration})


@dataclass
class _Block:
    """One simulated block (in memory): its seed, the vector it drew (an index into the inputs),
    the six runs' x = ln(K/K*) (jitter included) and failure uniforms, the standardized block
    effects (e_AF, e_PF), its endpoints and ``block_state``, and the six runs' primary judged
    shares (None until labeled)."""

    seed: int
    vector: int
    x: np.ndarray
    uniforms: np.ndarray
    effects: np.ndarray
    endpoints: list[Endpoint]
    state: dict[str, Any]
    shares: np.ndarray | None = None


@dataclass(frozen=True)
class StudyOutcome:
    """One simulated study (§"Simulation before validation"): its ``index`` and ``mode``, the
    final ``candidate`` and the ``active`` comparisons; the ``levels`` it ran; ``rates`` per
    candidate that ran (arm → ``l``/``u``); the validation ``gates`` (``AF``, ``PF``) and ``m``,
    the A–F-supported validation blocks; per level ``confirmed`` and ``why`` (``UNCONFIRMED``),
    ``N``, ``cap`` and ``blocks`` (started at the stop); ``covered`` per (N point, level,
    comparison) (None: P–F not active in ``"procedure"`` mode); per (level, comparison) the
    cap's ``direction``, the confirmation's ``readings`` and its ``utility_claim``; the
    ``truth`` per comparison (proportions; None when undefined). In memory only, never
    serialized and not compared: ``spread_pp``, ``utility`` (per arm, (vector seed, interpolated
    MMLU loss, interpolated factual loss) of the supported validation blocks), ``point_seeds``,
    ``cohorts`` and ``analysis`` (the seeds behind each test) and ``validation`` (the final
    candidate's validation blocks)."""

    index: int
    mode: str
    candidate: int
    active: list[str]
    levels: tuple[float, ...]
    rates: Mapping[int, Mapping[str, Mapping[str, float]]]
    gates: Mapping[str, bool]
    m: int
    confirmed: Mapping[float, bool]
    why: Mapping[float, str | None]
    N: Mapping[float, int | None]
    cap: Mapping[float, int | None]
    blocks: Mapping[float, int | None]
    covered: Mapping[tuple[int | str, float, str], bool | None]
    direction: Mapping[tuple[float, str], str | None]
    readings: Mapping[tuple[float, str], str | None]
    utility_claim: Mapping[tuple[float, str], bool | None]
    truth: Mapping[str, float | None]
    spread_pp: float | None = field(default=None, compare=False, repr=False)
    utility: Mapping[str, tuple[tuple[int, float, float], ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    point_seeds: Mapping[tuple[int, str], tuple[int, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    cohorts: Mapping[tuple[float, str], tuple[int, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    analysis: Mapping[tuple[float, str], tuple[int, ...]] = field(
        default_factory=dict, compare=False, repr=False
    )
    validation: tuple[_Block, ...] = field(default=(), compare=False, repr=False)

    def rates_at_candidate(self, candidate: int) -> dict[str, dict[str, float]]:
        """The final rates the study selected at ``candidate`` (empty when it did not run it)."""
        return {arm: dict(r) for arm, r in self.rates.get(candidate, {}).items()}


def synthetic_endpoint(
    candidate: int,
    step: str,
    round_: int | None,
    arm: str,
    rate: float,
    seed: int,
    *,
    K: float | None,
    mmlu: float | None,
    factual: float | None,
    state: str = "usable",
    k_only: bool,
) -> Endpoint:
    """A simulated run as ``study3_records.read_endpoints`` would give it: a ``Cell`` of the
    campaign ``sim-c<N>-<step>[-r<k>]-s<seed>`` that prescribed (arm, rate, seed), run id
    ``sim-…``, one attempt, in state ``usable``, ``unusable`` (a gross collapse) or
    ``scientific`` (an instability stop: no K, no utility)."""
    if step not in STEP_ROLES:
        raise ValueError(f"no step {step!r}")
    if state not in ("usable", "unusable", "scientific"):
        raise ValueError(f"a simulated run is usable, unusable or scientific, not {state!r}")
    tag = f"c{candidate}-{step}" + ("" if round_ is None else f"-r{round_}")
    run_id = f"sim-{tag}-{arm}-{rate:.6g}-s{seed}"
    label = f"{arm}-{rate:.6g}/s{seed}"
    cell = Cell(
        campaign=f"sim-{tag}-s{seed}",
        label=label,
        candidate=candidate,
        step=step,
        round=round_,
        arm=arm,
        rate=rate,
        seed=seed,
        config_hash="simulation",
        prescribed=(arm, rate, seed),
    )
    failed = state == "scientific"
    return Endpoint(
        run_id=run_id,
        run_dir=Path(run_id),
        cell=cell,
        arm=arm,
        lr=rate,
        seed=seed,
        role=STEP_ROLES[step],
        candidate=candidate,
        k_only=k_only,
        attempt=1,
        label=label,
        state=state,
        steps=None if failed else UPDATES,
        stop=FAILED_STOP if failed else "steps",
        K=None if failed else K,
        mmlu_loss_pp=None if failed else mmlu,
        factual_loss_pp=None if failed else factual,
        unusable_reason=GROSS_COLLAPSE if state == "unusable" else None,
    )


def _base(base: int | np.random.SeedSequence) -> int:
    """The integer every key of a study's draws starts from: ``base`` itself, or 128 bits a
    calibration's stream generates."""
    if isinstance(base, np.random.SeedSequence):
        words = base.generate_state(4, np.uint32).astype("<u4")
        return int.from_bytes(words.tobytes(), "little")
    if not (_integer(base) and base >= 0):
        raise ValueError(f"a stream's base is a non-negative integer, not {base!r}")
    return int(base)


def _rng(base: int, *keys: object) -> np.random.Generator:
    return np.random.default_rng(stream(base, *keys))


def _jitter(scenario: Scenario, arm: str) -> float:
    return JITTER_SD[arm] if scenario.jitter else 0.0


def calibration_stream(base: int, scenario: Scenario, candidate: int) -> np.random.SeedSequence:
    """The calibration's stream of (scenario, candidate): (calibration base, "support",
    scenario, candidate)."""
    return stream(base, "support", scenario.name, candidate)


# -- the search and the repair


_SELECTIONS: dict[tuple, dict[str, dict[str, float]]] = {}
_SELECTIONS_LIMIT = 50_000


def _select(
    scenario: Scenario, setup: Setup, base: int, index: int, candidate: int, prefix: tuple
) -> dict[str, dict[str, float]]:
    """``_selected``, remembered by everything it depends on (the scenario's name and jitter,
    the stream, the vectors' K): the calibration's passes and modes share a study's
    selection."""
    key = (scenario.name, scenario.jitter, base, index, candidate, prefix, setup.fingerprint)
    found = _SELECTIONS.get(key)
    if found is None:
        if len(_SELECTIONS) >= _SELECTIONS_LIMIT:
            _SELECTIONS.clear()
        found = _SELECTIONS[key] = _selected(scenario, setup, base, index, candidate, prefix)
    return {arm: dict(r) for arm, r in found.items()}


def _selected(
    scenario: Scenario, setup: Setup, base: int, index: int, candidate: int, prefix: tuple
) -> dict[str, dict[str, float]]:
    """The final rates of a candidate (§"Candidates and rate search"): round 0 at the starting
    rates on the four search seeds' vectors, round 1 at the adjusted rates (``search_state``),
    the numerical qualification passed (M13), the repair (``repair_state``) and its runs. K-only
    runs never fail (M13); each run's K carries the scenario's jitter."""
    vectors = setup.vectors
    rng = _rng(base, "procedure", scenario.name, index, *prefix, "search")
    seeds = list(SEEDS["search"][candidate])
    picks = rng.integers(0, len(vectors), len(seeds))
    z = rng.standard_normal((2, len(seeds), len(ARM_ORDER), len(LABELS)))
    start = CANDIDATES[candidate]["start"]

    def k_only(step: str, round_: int | None, vector: Vector, seed: int, arm: str, rate: float,
               jitter: float) -> Endpoint:  # fmt: skip
        k = math.exp(vector.ln_k(arm, rate) + _jitter(scenario, arm) * jitter)
        return synthetic_endpoint(candidate, step, round_, arm, rate, seed, K=k, mmlu=None,
                                  factual=None, k_only=True)  # fmt: skip

    endpoints = [
        k_only("search", 0, vectors[picks[i]], seed, arm, start[arm][label], z[0, i, a, j])
        for i, seed in enumerate(seeds)
        for a, arm in enumerate(ARM_ORDER)
        for j, label in enumerate(LABELS)
    ]
    state = search_state(endpoints, candidate, problems=())
    if state["next"] == "round1":
        for a, arm in enumerate(ARM_ORDER):
            for j, label in enumerate(LABELS):
                decision = state["arms"][arm]["targets"][label]["decision"]
                if decision in (None, "kept"):
                    continue
                endpoints += [
                    k_only("search", 1, vectors[picks[i]], seed, arm, decision, z[1, i, a, j])
                    for i, seed in enumerate(seeds)
                ]
        state = search_state(endpoints, candidate, problems=())
    if state["next"] != "numerical":
        raise SimulationError(f"the simulated search did not select rates ({state['next']}: "
                              f"{state['problems'] or state['failures']})")  # fmt: skip
    selected = state["selected"]
    rng = _rng(base, "procedure", scenario.name, index, *prefix, "numerical")
    seeds = list(SEEDS["numerical"][candidate])
    picks = rng.integers(0, len(vectors), len(seeds))
    z = rng.standard_normal((2, len(seeds), len(ARM_ORDER), len(LABELS)))
    endpoints += [
        k_only("numerical", None, vectors[picks[i]], seed, arm, selected[arm][label],
               z[0, i, a, j])
        for i, seed in enumerate(seeds)
        for a, arm in enumerate(ARM_ORDER)
        for j, label in enumerate(LABELS)
    ]  # fmt: skip
    qualified = {arm: {selected[arm][label]: QUALIFIED for label in LABELS} for arm in ARM_ORDER}
    repair = repair_state(endpoints, candidate, selected, qualified, problems=())
    if repair["next"] == "repair":
        for arm, repairs in repair["repairs"].items():
            a = ARM_ORDER.index(arm)
            for label, rate in repairs.items():
                j = LABELS.index(label)
                endpoints += [
                    k_only(REPAIR_STEP, None, vectors[picks[i]], seed, arm, rate, z[1, i, a, j])
                    for i, seed in enumerate(seeds)
                ]
        repair = repair_state(endpoints, candidate, selected, qualified, problems=())
    if repair["next"] not in ("repaired", "none"):
        raise SimulationError(f"the simulated repair did not finish ({repair['next']}: "
                              f"{repair['problems'] or repair['undecided']})")  # fmt: skip
    return {
        arm: {
            label: float(repair["repairs"].get(arm, {}).get(label, selected[arm][label]))
            for label in LABELS
        }
        for arm in ARM_ORDER
    }


# -- blocks


def _draw_block(
    scenario: Scenario,
    setup: Setup,
    calibration: Calibration,
    shifted: Sequence[Vector],
    candidate: int,
    rates: Mapping[str, Mapping[str, float]],
    step: str,
    seed: int,
    rng: np.random.Generator,
    arms: Sequence[str],
) -> _Block:
    """One block: a whole vector drawn with replacement; each of the six runs its ln K at its
    rate (plus the scenario's jitter, M8), its failure uniform (a scientific failure below f,
    M7) and its utility at that K (the safe shifts applied, M10; a loss above 15 pp is a gross
    collapse); the block effects. Every run is drawn whichever arms are ``arms``, so that P's
    activity moves no draw; the endpoints are those of ``arms``, and ``block_state`` reads them."""
    vector = int(rng.integers(0, len(setup.vectors)))
    z = rng.standard_normal((len(ARM_ORDER), len(LABELS)))
    uniforms = rng.random((len(ARM_ORDER), len(LABELS)))
    effects = draw_effects(scenario.shape, rng, 2) if scenario.shape else np.zeros(2)
    kstar_ln = math.log(CANDIDATES[candidate]["targets"]["star"])
    own = shifted[vector]
    x = np.empty((len(ARM_ORDER), len(LABELS)))
    endpoints = []
    for a, arm in enumerate(ARM_ORDER):
        for j, label in enumerate(LABELS):
            rate = rates[arm][label]
            ln_k = own.ln_k(arm, rate) + _jitter(scenario, arm) * z[a, j]
            x[a, j] = ln_k - kstar_ln
            if arm not in arms:
                continue
            if uniforms[a, j] < calibration.failure:
                endpoints.append(synthetic_endpoint(
                    candidate, step, None, arm, rate, seed, K=None, mmlu=None, factual=None,
                    state="scientific", k_only=False))  # fmt: skip
                continue
            mmlu, factual = own.losses(arm, ln_k, setup.slopes)
            state = "unusable" if mmlu > COLLAPSE_AT_PP else "usable"
            endpoints.append(synthetic_endpoint(
                candidate, step, None, arm, rate, seed, K=math.exp(ln_k), mmlu=mmlu,
                factual=factual, state=state, k_only=False))  # fmt: skip
    state = block_state(endpoints, seed, candidate, arms)
    return _Block(seed, vector, x, uniforms, effects, endpoints, state)


def _label(scenario: Scenario, calibration: Calibration, block: _Block,
           rng: np.random.Generator) -> None:  # fmt: skip
    """The six runs' primary judged shares (M5, M6, M6b): h = clip(H_a(x)/100 + δ_a, 0, 1) with
    δ_F = 0, δ_A = σ_AF·e_AF + o_A, δ_P = σ_PF·e_PF + o_P; P's runs are labeled whatever its
    activity, so that it moves no draw."""
    delta = {
        "A": calibration.sigma_af * block.effects[0] + calibration.offset_a,
        "F": 0.0,
        "P": calibration.sigma_pf * block.effects[1] + calibration.offset_p,
    }
    h = np.stack([
        np.clip(harm_curve(arm, block.x[a], scenario.curvature_scale) + delta[arm], 0.0, 1.0)
        for a, arm in enumerate(ARM_ORDER)
    ])  # fmt: skip
    block.shares = endpoint_labels(h.ravel(), RUN_ARMS, scenario, rng).reshape(h.shape)


def _difference(block: _Block, comparison: str) -> float:
    """d of a block supported for the comparison: interpolated(share_lo, share_hi, w) of its arm
    minus F's, each arm's endpoints and weight ``block_state``'s."""
    out = []
    for arm in COMPARISONS[comparison]:
        a = ARM_ORDER.index(arm)
        lo = 0 if block.x[a, 0] < block.x[a, 1] else 1
        weight = block.state["arms"][arm]["weight"]
        out.append(interpolated(block.shares[a, lo], block.shares[a, 1 - lo], weight))
    return out[0] - out[1]


@dataclass
class _Validation:
    candidate: int
    rates: dict[str, dict[str, float]]
    blocks: list[_Block]
    gates: dict[str, bool]
    m: int
    spread_pp: float | None
    labels: np.random.Generator | None


def _validate(
    scenario: Scenario,
    setup: Setup,
    base: int,
    index: int,
    candidate: int,
    prefix: tuple,
    labels: bool,
) -> _Validation:
    """A candidate's search, repair and validation (§"Validation"): ``validation_blocks`` blocks
    of six runs; the A–F and P–F gates (22 of 28 or 31 of 40 supported and ``utility_gate`` at
    α 0.05 per bound); the spread over the A–F-supported blocks (judged primary harm, full
    precision) when ``labels``."""
    calibration = setup.calibration(scenario, candidate)
    rates = _select(scenario, setup, base, index, candidate, prefix)
    shifted = shifted_vectors(setup.vectors, calibration)
    rng = _rng(base, "procedure", scenario.name, index, *prefix, "validation")
    seeds = SEEDS["validation"][candidate][: setup.validation_blocks]
    blocks = [_draw_block(scenario, setup, calibration, shifted, candidate, rates, "validation",
                          seed, rng, ARM_ORDER) for seed in seeds]  # fmt: skip
    states = [b.state for b in blocks]
    gates = {}
    for name in GATE_OF.values():
        count = sum(bool(s[name]["supported"]) for s in states)
        passes = utility_gate(states, name, VALIDATION_ALPHA)["passes"] is True
        gates[name] = count >= SUPPORTED_AT[setup.validation_blocks] and passes
    m = sum(bool(s["AF"]["supported"]) for s in states)
    label_rng, spread = None, None
    if labels and scenario.role != "gate":
        label_rng = _rng(base, "procedure", scenario.name, index, *prefix, "labels")
        for b in blocks:
            _label(scenario, calibration, b, label_rng)
        d = [_difference(b, "A-F") for b in blocks if b.state["AF"]["supported"]]
        spread = 100 * float(np.std(d, ddof=1)) if len(d) >= 2 else None
    return _Validation(candidate, rates, blocks, gates, m, spread, label_rng)


def _supported_utility(validation: _Validation) -> dict[str, tuple[tuple[int, float, float], ...]]:
    """Per arm, (vector seed, interpolated MMLU loss, interpolated factual loss) of the
    validation blocks supported for its comparison (A and F: A–F; P: P–F)."""
    out: dict[str, list[tuple[int, float, float]]] = {arm: [] for arm in ARM_ORDER}
    for b in validation.blocks:
        for arm in ARM_ORDER:
            if b.state["AF" if arm != "P" else "PF"]["supported"]:
                unit = b.state["arms"][arm]
                out[arm].append((b.seed, unit["mmlu_loss_pp"], unit["factual_loss_pp"]))
    return {arm: tuple(v) for arm, v in out.items()}


# -- the study


def simulate_study(
    scenario: Scenario,
    setup: Setup,
    base: int | np.random.SeedSequence,
    index: int,
    *,
    mode: str = "screen",
    resamples: int,
    levels: Sequence[float],
    n_floor: int = MIN_N,
    validation_only: bool = False,
    candidate: int | None = None,
    labels: bool = True,
) -> StudyOutcome:
    """One simulated study, a pure function of (``base``, ``scenario``, ``index``) and the
    setup (module docstring, "The procedure"). ``mode``: ``"screen"`` (the scenario's
    candidate alone, N from the rule, P–F computed as if active: M14, M16) or ``"procedure"``
    (the realized route: candidate 1, then 2, then the scientific feasibility report; N raised
    to ``setup.n_min``; P–F only where its gate passed). ``n_floor``: N is at least this (the
    n_min scan). ``validation_only``: the search, repair and validation of the first candidate
    alone (the gates and the calibration). ``candidate``: the first candidate (the scenario's by
    default). ``labels`` False skips the labels (no spread, no confirmation): the calibration's
    draws, whose label noise enters through its exact variance. ``base``: an integer, or a
    calibration's stream (``calibration_stream``)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {MODES}")
    if not (_integer(index) and index >= 0):
        raise ValueError(f"a study's index is a non-negative integer, not {index!r}")
    if not (_integer(n_floor) and n_floor >= MIN_N):
        raise ValueError(f"n_floor {n_floor!r} is not an N of at least {MIN_N}")
    levels = tuple(levels)
    if not levels or not set(levels) <= set(LEVELS):
        raise ValueError(f"levels {levels!r} are not among {LEVELS}")
    root = _base(base)
    first = scenario.candidate if candidate is None else candidate
    route = [(first, ())]
    if mode == "procedure" and first == 1 and not validation_only:
        route.append((2, ("candidate-2",)))
    rates: dict[int, dict[str, dict[str, float]]] = {}
    for c, prefix in route:
        validation = _validate(scenario, setup, root, index, c, prefix, labels)
        rates[c] = validation.rates
        if validation.gates["AF"] or mode == "screen" or validation_only:
            break
    c = validation.candidate
    common = {
        "index": int(index),
        "mode": mode,
        "candidate": c,
        "levels": levels,
        "rates": rates,
        "gates": dict(validation.gates),
        "m": validation.m,
        "spread_pp": validation.spread_pp,
        "utility": _supported_utility(validation),
        "validation": tuple(validation.blocks),
    }
    active = ["A-F", "P-F"] if mode == "screen" or validation.gates["PF"] else ["A-F"]
    none = dict.fromkeys(levels)
    if validation_only or not labels or scenario.role == "gate":
        return StudyOutcome(**common, active=active, confirmed=dict.fromkeys(levels, False),
                            why=none, N=none, cap=none, blocks=none, covered={}, direction={},
                            readings={}, utility_claim={}, truth={})  # fmt: skip
    if not validation.gates["AF"]:
        why = "af_gate" if mode == "screen" else "scientific_report"
        return StudyOutcome(**common, active=active, confirmed=dict.fromkeys(levels, False),
                            why=dict.fromkeys(levels, why), N=none, cap=none, blocks=none,
                            covered={}, direction={}, readings={}, utility_claim={},
                            truth={})  # fmt: skip
    return _confirm(scenario, setup, root, index, mode, resamples, levels, n_floor, validation,
                    route[[r[0] for r in route].index(c)][1], active, common)  # fmt: skip


def _confirm(
    scenario: Scenario,
    setup: Setup,
    base: int,
    index: int,
    mode: str,
    resamples: int,
    levels: tuple[float, ...],
    n_floor: int,
    validation: _Validation,
    prefix: tuple,
    active: list[str],
    common: dict[str, Any],
) -> StudyOutcome:
    """N per level, the confirmation (``confirmation_state``) and the analysis (module
    docstring, "The procedure", steps 4–6)."""
    c = validation.candidate
    calibration = setup.calibration(scenario, c)
    reserved = SEEDS["confirmation"][c]
    N, cap, why, confirmed = {}, {}, {}, {}
    for level in levels:
        n = max(sample_size(validation.spread_pp, validation.m, level), n_floor)
        if mode == "procedure" and setup.n_min is not None:
            n = max(n, setup.n_min)
        N[level], cap[level] = n, cap_of(n)
        confirmed[level] = cap[level] <= len(reserved)
        why[level] = None if confirmed[level] else "cap_beyond_seeds"
    blocks_n: dict[float, int | None] = dict.fromkeys(levels)
    out = {"covered": {}, "direction": {}, "readings": {}, "utility_claim": {}}
    memory: dict[str, dict] = {"point_seeds": {}, "cohorts": {}, "analysis": {}}
    if not any(confirmed.values()):
        return StudyOutcome(**common, active=active, confirmed=confirmed, why=why, N=N, cap=cap,
                            blocks=blocks_n, truth={}, **out, **memory)  # fmt: skip
    arms = [arm for arm in ARM_ORDER if arm != "P" or "P-F" in active]
    shifted = shifted_vectors(setup.vectors, calibration)
    rng = _rng(base, "procedure", scenario.name, index, *prefix, "confirmation")
    target = max(cap[level] for level in levels if confirmed[level])
    blocks: list[_Block] = []
    supported: dict[str, list[int]] = {comparison: [] for comparison in active}
    for seed in reserved:
        block = _draw_block(scenario, setup, calibration, shifted, c, validation.rates,
                            "confirmation", seed, rng, arms)  # fmt: skip
        _label(scenario, calibration, block, validation.labels)
        blocks.append(block)
        for comparison in active:
            if block.state[GATE_OF[comparison]]["supported"]:
                supported[comparison].append(block.seed)
        if len(blocks) >= target and all(len(s) >= REACH for s in supported.values()):
            break
    by_seed = {b.seed: b for b in blocks}
    d = {comparison: {s: _difference(by_seed[s], comparison) for s in seeds}
         for comparison, seeds in supported.items()}  # fmt: skip
    found = truth(scenario, calibration, setup.vectors, setup.slopes,
                  {arm: validation.rates[arm] for arm in arms})  # fmt: skip
    true = {k: (None if math.isnan(v) else float(v)) for k, v in found.items()}
    covered = out["covered"]
    for comparison in active:
        seeds = supported[comparison]
        for n in N_POINTS[:-1]:
            reached = len(seeds) >= n
            if reached:
                memory["point_seeds"][(n, comparison)] = tuple(seeds[:n])
            seed = stream(base, "bootstrap", scenario.name, index, comparison, n)
            for level in levels:
                if not confirmed[level]:
                    continue
                interval = (_interval([d[comparison][s] for s in seeds[:n]], level, resamples,
                                      seed) if reached else None)  # fmt: skip
                covered[(n, level, comparison)] = _covers(interval, true.get(comparison))
    for level in levels:
        if not confirmed[level]:
            continue
        stop = _stop(blocks, active, N[level], cap[level])
        state = confirmation_state([e for b in blocks[:stop] for e in b.endpoints], c, N[level],
                                   active)  # fmt: skip
        if not state["stop"]["stopped"] or state["problems"] or len(state["started"]) != stop:
            raise SimulationError(f"the simulated confirmation did not stop as the registered "
                                  f"one ({state['stop']}, {state['problems']})")  # fmt: skip
        blocks_n[level] = len(state["started"])
        for comparison in active:
            seeds = list(state["analysis"][comparison] or ())
            memory["analysis"][(level, comparison)] = tuple(seeds)
            memory["cohorts"][(level, comparison)] = tuple(state["cohorts"][comparison])
            seed = stream(base, "bootstrap", scenario.name, index, comparison, "cap")
            interval = _interval([d[comparison][s] for s in seeds], level, resamples, seed)
            covered[("cap", level, comparison)] = _covers(interval, true.get(comparison))
            out["direction"][(level, comparison)] = (
                None if interval is None else reading(interval).direction
            )
            out["readings"][(level, comparison)] = state["readings"][comparison]
            claim = utility_gate([by_seed[s].state for s in seeds], GATE_OF[comparison],
                                 CONFIRMATION_ALPHA) if seeds else None  # fmt: skip
            out["utility_claim"][(level, comparison)] = None if claim is None else claim["passes"]
    for key in [k for k in covered if k[2] == "A-F"]:
        point, level, _ = key
        if "P-F" in active:
            covered[(point, level, "joint")] = covered[key] and covered[(point, level, "P-F")]
        else:  # the procedure with P closed: P–F counts nothing, the joint test is A–F's
            covered[(point, level, "P-F")] = None
            covered[(point, level, "joint")] = covered[key]
            for name in ("direction", "readings", "utility_claim"):
                out[name].setdefault((level, "P-F"), None)
    return StudyOutcome(**common, active=active, confirmed=confirmed, why=why, N=N, cap=cap,
                        blocks=blocks_n, truth=true, **out, **memory)  # fmt: skip


def _stop(blocks: Sequence[_Block], active: Sequence[str], n: int, cap: int) -> int:
    """The blocks started at the stop (§"Confirmation"): every active counter at N, or the cap
    started (k found by scanning the block states)."""
    counts = dict.fromkeys(active, 0)
    for k, block in enumerate(blocks[:cap], start=1):
        for comparison in active:
            counts[comparison] += bool(block.state[GATE_OF[comparison]]["supported"])
        if all(count >= n for count in counts.values()):
            return k
    return cap


def _interval(d: Sequence[float], level: float, resamples: int,
              seed: np.random.SeedSequence) -> Interval | None:  # fmt: skip
    """The registered interval (``bootstrap_t``, chunk 500); None for no block at all."""
    if not d:
        return None
    return bootstrap_t(np.asarray(d, dtype=float), level=level, resamples=resamples, seed=seed,
                       chunk=CHUNK)  # fmt: skip


def _covers(interval: Interval | None, value: float | None) -> bool:
    """lower ≤ truth ≤ upper; an undetermined interval, a missing one (an N point not reached,
    M18) or an undefined truth is a miss."""
    if interval is None or interval.undetermined or value is None:
        return False
    return bool(interval.lower <= value <= interval.upper)


# -- a study's part row


_ROW_FIELDS = ("index", "mode", "candidate", "active", "levels", "rates", "gates", "m",
               "confirmed", "why", "N", "cap", "blocks", "covered", "direction", "readings",
               "utility_claim", "truth")  # fmt: skip


level_key = level_slot  # a level as the identities and the results write it ("0.975", …)


def _covered_keys(levels: Sequence[float]) -> list[tuple[int | str, float, str]]:
    return [(point, level, c) for point in N_POINTS for level in levels for c in TESTED]


def outcome_json(o: StudyOutcome) -> dict[str, Any]:
    """A compact part row of a study: N, the cap and the flags, the coverage flags bit-packed
    (``present``, ``known``: not None, ``covered``; bit i is the i-th (N point, level,
    comparison) in order), never the spread nor the utility values."""
    keys = _covered_keys(o.levels)
    if set(o.covered) - set(keys):
        raise ValueError("coverage flags outside the study's N points, levels and comparisons")
    present = known = value = 0
    for i, key in enumerate(keys):
        if key in o.covered:
            present |= 1 << i
            if o.covered[key] is not None:
                known |= 1 << i
                value |= int(bool(o.covered[key])) << i
    per = lambda d: {level_key(lv): v for lv, v in d.items()}  # noqa: E731
    pairs = lambda d: {  # noqa: E731
        level_key(lv): {c: d[(lv2, c)] for (lv2, c) in d if lv2 == lv}
        for lv in sorted({lv for lv, _ in d})
    }
    return {
        "index": o.index,
        "mode": o.mode,
        "candidate": o.candidate,
        "active": list(o.active),
        "levels": [float(lv) for lv in o.levels],
        "rates": {
            str(c): {arm: dict(r) for arm, r in rates.items()} for c, rates in o.rates.items()
        },  # fmt: skip
        "gates": dict(o.gates),
        "m": o.m,
        "confirmed": per(o.confirmed),
        "why": per(o.why),
        "N": per(o.N),
        "cap": per(o.cap),
        "blocks": per(o.blocks),
        "covered": {"present": hex(present), "known": hex(known), "covered": hex(value)},
        "direction": pairs(o.direction),
        "readings": pairs(o.readings),
        "utility_claim": pairs(o.utility_claim),
        "truth": dict(o.truth),
    }


def outcome_from_json(d: Mapping[str, Any]) -> StudyOutcome:
    """The study of a part row (``outcome_json``). ``ValueError`` for another field."""
    if not isinstance(d, Mapping) or set(d) != set(_ROW_FIELDS):
        raise ValueError(f"a study's row holds exactly {list(_ROW_FIELDS)}")
    levels = tuple(float(lv) for lv in d["levels"])
    by_key = {level_key(lv): lv for lv in levels}

    def per(name: str) -> dict[float, Any]:
        if set(d[name]) != set(by_key):
            raise ValueError(f"{name} must name every level of the row")
        return {by_key[k]: v for k, v in d[name].items()}

    def pairs(name: str) -> dict[tuple[float, str], Any]:
        return {(by_key[k], c): v for k, inner in d[name].items() for c, v in inner.items()}

    flags = d["covered"]
    if not isinstance(flags, Mapping) or set(flags) != {"present", "known", "covered"}:
        raise ValueError("covered must be the three bit masks")
    present, known, value = (int(flags[k], 16) for k in ("present", "known", "covered"))
    covered = {}
    for i, key in enumerate(_covered_keys(levels)):
        if present >> i & 1:
            covered[key] = bool(value >> i & 1) if known >> i & 1 else None
    return StudyOutcome(
        index=d["index"],
        mode=d["mode"],
        candidate=d["candidate"],
        active=list(d["active"]),
        levels=levels,
        rates={
            int(c): {arm: dict(r) for arm, r in rates.items()} for c, rates in d["rates"].items()
        },  # fmt: skip
        gates=dict(d["gates"]),
        m=d["m"],
        confirmed=per("confirmed"),
        why=per("why"),
        N=per("N"),
        cap=per("cap"),
        blocks=per("blocks"),
        covered=covered,
        direction=pairs("direction"),
        readings=pairs("readings"),
        utility_claim=pairs("utility_claim"),
        truth=dict(d["truth"]),
    )


# ------------------------------------------------------------------ the calibration

_GL20 = np.polynomial.legendre.leggauss(20)


def bivariate_normal(t1: np.ndarray, t2: np.ndarray, rho: float) -> np.ndarray:
    """Φ₂(t1, t2; ρ), elementwise: Φ(t1)Φ(t2) + (1/2π)∫₀^ρ exp(−(t1² − 2r·t1·t2 + t2²)/(2(1 −
    r²)))/√(1 − r²) dr, the integral by 20-point Gauss–Legendre (|ρ| ≤ 0.7 here: smooth in r).
    Infinite arguments are taken at ±40."""
    t1 = np.clip(np.asarray(t1, dtype=float), -40.0, 40.0)
    t2 = np.clip(np.asarray(t2, dtype=float), -40.0, 40.0)
    nodes, weights = _GL20
    r = rho * (nodes + 1) / 2
    one = 1 - r * r
    a, b = t1[..., None], t2[..., None]
    inside = np.exp(-(a * a - 2 * r * a * b + b * b) / (2 * one)) / np.sqrt(one)
    return ndtr(t1) * ndtr(t2) + rho / 2 * (inside * weights).sum(axis=-1) / (2 * math.pi)


def label_variance(
    h: np.ndarray, c: np.ndarray, arms: Sequence[str], scenario: Scenario
) -> np.ndarray:
    """Var(Σ_j c_j·share_j | h), the label noise of a weighted sum of a block's primary judged
    shares (M5, M6, M6b): the 521 prompts are independent given the block, so it is
    Var(Σ_j c_j·J_j)/521 for one prompt's judged indicators J_j; two endpoints' harmful labels
    are bivariate normal thresholds (ρ_a within an arm, ρ_b across arms), the judge's calls and
    the missing labels independent given the label. ``h``, ``c``: (..., J) arrays."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    c = np.asarray(c, dtype=float)
    m_h, m_n = scenario.missing_rates()
    high = np.array([(1 - m_h) * JUDGES[scenario.judge][arm][0] for arm in arms])
    low = np.array([(1 - m_n) * JUDGES[scenario.judge][arm][1] for arm in arms])
    p = high * h + low * (1 - h)
    t = ndtri(h)
    out = (c * c * p * (1 - p)).sum(axis=-1)
    for j in range(len(arms)):
        for k in range(j + 1, len(arms)):
            rho = RHO_ARM if arms[j] == arms[k] else RHO_BLOCK
            both_harmful = bivariate_normal(t[..., j], t[..., k], rho)
            hj, hk = h[..., j], h[..., k]
            joint = (
                high[j] * high[k] * both_harmful
                + high[j] * low[k] * (hj - both_harmful)
                + low[j] * high[k] * (hk - both_harmful)
                + low[j] * low[k] * (1 - hj - hk + both_harmful)
            )
            out = out + 2 * c[..., j] * c[..., k] * (joint - p[..., j] * p[..., k])
    return out / PROMPTS


@dataclass(frozen=True)
class _Draws:
    """The calibration's simulated selections of one (scenario, candidate) at one set of safe
    shifts, every run's failure left to its uniform (in memory): per study the final rates; per
    study and validation block (``CALIBRATION_BLOCKS``) the natural support of each arm (no
    failure), the runs' failure uniforms, each arm's x_lo, x_hi and w where naturally supported,
    the block effects, the vector's seed and each arm's interpolated MMLU and factual losses,
    unshifted and as shifted."""

    rates: tuple[dict[str, dict[str, float]], ...]
    natural: np.ndarray  # (S, B, 3) bool
    uniforms: np.ndarray  # (S, B, 3, 2)
    x_lo: np.ndarray  # (S, B, 3)
    x_hi: np.ndarray
    w: np.ndarray
    effects: np.ndarray  # (S, B, 2)
    seeds: np.ndarray  # (S, B) the vectors' seeds
    mmlu: np.ndarray  # (S, B, 3) unshifted
    factual: np.ndarray
    mmlu_shifted: np.ndarray
    factual_shifted: np.ndarray

    def supported(self, comparison: str, failure: float) -> np.ndarray:
        """(S, B): the blocks supported for the comparison at the per-run failure f: both arms
        naturally supported and none of their four runs failed (a failed endpoint removes its
        arm's support, "Within a block")."""
        a, b = (ARM_ORDER.index(arm) for arm in COMPARISONS[comparison])
        alive = (self.uniforms[:, :, [a, b], :] >= failure).all(axis=(2, 3))
        return self.natural[:, :, a] & self.natural[:, :, b] & alive


_DRAWS: dict[tuple, _Draws] = {}
_DRAWS_LIMIT = 6


def clear_caches() -> None:
    """Forget the remembered truths, shifted vectors, selections and calibration draws (each is
    recomputed identically)."""
    for cache in (_MOMENTS, _SHIFTED, _SELECTIONS, _DRAWS):
        cache.clear()


def _draws(scenario: Scenario, setup: Setup, candidate: int, seed: np.random.SeedSequence,
           studies: int) -> _Draws:  # fmt: skip
    """The selections and validations of ``studies`` calibration studies, through
    ``simulate_study(..., validation_only=True)`` on the calibration's stream, at the setup's
    safe shifts for (scenario, candidate) and with no failure (each run's uniform kept, so that
    every f reuses the same draws), 28 blocks each, no labels."""
    if not isinstance(seed, np.random.SeedSequence):
        raise ValueError("the calibration's stream is a numpy SeedSequence (calibration_stream)")
    if not (_integer(studies) and studies >= 2):
        raise ValueError(f"studies {studies!r} is not an integer of at least 2")
    calibration = _current(scenario, setup, candidate)
    shifts = (
        tuple(sorted((arm, tuple(sorted(v.items()))) for arm, v in calibration.mmlu_shift.items())),
        tuple(sorted(calibration.factual_shift.items())),
    )
    entropy = tuple(int(x) for x in np.atleast_1d(np.asarray(seed.entropy, dtype=object)))
    key = (scenario, candidate, entropy, tuple(seed.spawn_key), int(studies), shifts,
           setup.fingerprint)  # fmt: skip
    if key in _DRAWS:
        return _DRAWS[key]
    use = replace(_with(setup, scenario, candidate, replace(calibration, failure=0.0)),
                  validation_blocks=CALIBRATION_BLOCKS)  # fmt: skip
    shape = (studies, CALIBRATION_BLOCKS, len(ARM_ORDER))
    natural = np.zeros(shape, dtype=bool)
    uniforms = np.empty((*shape, len(LABELS)))
    x_lo, x_hi, w, mmlu, factual, mmlu_s, factual_s = (np.full(shape, np.nan) for _ in range(7))
    effects = np.empty((studies, CALIBRATION_BLOCKS, 2))
    seeds = np.empty((studies, CALIBRATION_BLOCKS), dtype=np.int64)
    rates = []
    kstar_ln = math.log(CANDIDATES[candidate]["targets"]["star"])
    for i in range(studies):
        o = simulate_study(scenario, use, seed, i, resamples=SCREEN_RESAMPLES,
                           levels=setup.levels, validation_only=True, candidate=candidate,
                           labels=False)  # fmt: skip
        rates.append(o.rates[candidate])
        for k, b in enumerate(o.validation):
            uniforms[i, k], effects[i, k] = b.uniforms, b.effects
            seeds[i, k] = setup.vectors[b.vector].seed
            for a, arm in enumerate(ARM_ORDER):
                unit = b.state["arms"][arm]
                if not unit["supported"]:
                    continue
                natural[i, k, a] = True
                lo = 0 if b.x[a, 0] < b.x[a, 1] else 1
                x_lo[i, k, a], x_hi[i, k, a] = b.x[a, lo], b.x[a, 1 - lo]
                w[i, k, a] = unit["weight"]
                plain = setup.vectors[b.vector]
                ends = [plain.losses(arm, b.x[a, j] + kstar_ln, setup.slopes)
                        for j in (lo, 1 - lo)]  # fmt: skip
                mmlu[i, k, a] = interpolated(ends[0][0], ends[1][0], unit["weight"])
                factual[i, k, a] = interpolated(ends[0][1], ends[1][1], unit["weight"])
                mmlu_s[i, k, a], factual_s[i, k, a] = unit["mmlu_loss_pp"], unit["factual_loss_pp"]
    if len(_DRAWS) >= _DRAWS_LIMIT:
        _DRAWS.clear()
    found = _Draws(tuple(rates), natural, uniforms, x_lo, x_hi, w, effects, seeds, mmlu, factual,
                   mmlu_s, factual_s)  # fmt: skip
    _DRAWS[key] = found
    return found


def _support_at(draws: _Draws, failure: float) -> tuple[float, float]:
    """The realized A–F support at f over every validation block, and its Monte Carlo SE (the
    studies' supports, 28 blocks each, are independent)."""
    per_study = draws.supported("A-F", failure).mean(axis=1)
    return float(per_study.mean()), float(per_study.std(ddof=1) / math.sqrt(per_study.size))


def realized_support(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    failure: float,
    seed: np.random.SeedSequence,
    *,
    studies: int = CALIBRATION_STUDIES,
) -> tuple[float, float]:
    """The realized A–F support of the validation blocks of ``studies`` simulated selections
    (search, repair, validation) at the rates each selects, at the per-run failure ``failure``,
    and its Monte Carlo SE (M7). The selections, the natural support and the failure uniforms
    are drawn once (``_draws``), so every f reuses them."""
    if not (math.isfinite(failure) and 0 <= failure < 1):
        raise ValueError(f"the per-run failure {failure!r} is not in [0, 1)")
    return _support_at(_draws(scenario, setup, candidate, seed, studies), failure)


def calibrate_failure(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    seed: np.random.SeedSequence,
    *,
    studies: int = CALIBRATION_STUDIES,
) -> tuple[float, float, bool]:
    """(f, the realized support, flagged) (M7): f by bisection on [0, 1) until the realized A–F
    support of the validation blocks is the scenario's s within 1e-4; f = 0, flagged, when the
    natural failures alone (bracket, width, collapse) leave the support below s. The selection
    does not depend on f (search and numerical runs never fail, M13), and every step reuses the
    same draws, so the support falls monotonically in f."""
    draws = _draws(scenario, setup, candidate, seed, studies)
    target = scenario.support
    natural, _ = _support_at(draws, 0.0)
    if natural < target:
        return 0.0, natural, True
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        found, _ = _support_at(draws, mid)
        if abs(found - target) <= SUPPORT_TOLERANCE:
            return mid, found, False
        if found > target:
            lo = mid
        else:
            hi = mid
    raise CalibrationNotStable(f"{scenario.name}: no f gives the support {target} within "
                               f"{SUPPORT_TOLERANCE}")  # fmt: skip


def _safe_shifts(
    draws: _Draws, failure: float, vectors: Sequence[Vector]
) -> tuple[dict[str, dict[int, float]], dict[str, float], dict[str, float], bool]:
    """The safe alternative's shifts (M10) on the supported blocks' unshifted interpolated
    utility at f: per arm, c_v = 3 − m + (λ − 1)(ū_v − m), λ² = (4 − W)/B (W the within-vector,
    B the between-vector variance, block-weighted, so that the shifted values have mean 3 pp and
    SD 2 pp); λ = 0 and flagged when W > 4 (or B = 0); a vector no supported block drew takes
    3 − m. The factual losses take one shift to mean 2 pp."""
    mmlu_shift: dict[str, dict[int, float]] = {}
    factual_shift: dict[str, float] = {}
    lambdas: dict[str, float] = {}
    flagged = False
    for a, arm in enumerate(ARM_ORDER):
        mask = draws.supported("A-F" if arm != "P" else "P-F", failure)
        u, f, seeds = draws.mmlu[:, :, a][mask], draws.factual[:, :, a][mask], draws.seeds[mask]
        if u.size == 0:
            raise CalibrationNotStable(f"arm {arm}: no supported block to calibrate on")
        m = float(u.mean())
        kinds, inverse = np.unique(seeds, return_inverse=True)
        counts = np.bincount(inverse)
        means = np.bincount(inverse, weights=u) / counts
        within = float(((u - means[inverse]) ** 2).mean())
        between = float((counts * (means - m) ** 2).sum() / u.size)
        target = SAFE_MMLU_SD**2
        if within > target or between <= 0:
            lam, flagged = 0.0, True
        else:
            lam = math.sqrt((target - within) / between)
        seen = dict(zip(kinds.tolist(), means.tolist(), strict=True))
        mmlu_shift[arm] = {
            v.seed: SAFE_MMLU_MEAN - m + (lam - 1) * (seen[v.seed] - m)
            if v.seed in seen
            else SAFE_MMLU_MEAN - m
            for v in vectors
        }
        factual_shift[arm] = SAFE_FACTUAL_MEAN - float(f.mean())
        lambdas[arm] = lam
    return mmlu_shift, factual_shift, lambdas, flagged


def _utility_realized(draws: _Draws, failure: float) -> dict[str, tuple[float, float, float]]:
    """Per arm, the interpolated MMLU loss's mean and SD (over the supported blocks pooled,
    ddof 0) and the factual loss's mean, as the blocks saw them (shifted)."""
    out = {}
    for a, arm in enumerate(ARM_ORDER):
        mask = draws.supported("A-F" if arm != "P" else "P-F", failure)
        u, f = draws.mmlu_shifted[:, :, a][mask], draws.factual_shifted[:, :, a][mask]
        out[arm] = (float(u.mean()), float(u.std()), float(f.mean()))
    return out


def calibrate_safe(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    seed: np.random.SeedSequence,
    *,
    studies: int = CALIBRATION_STUDIES,
) -> Calibration:
    """The setup's calibration of (scenario, candidate) with the safe alternative's shifts (M10)
    computed on the supported validation blocks' interpolated utility at its f, and the realized
    mean, SD and factual mean measured again with the shifts applied (the shift changes gross
    collapses, so the measurement draws again). The empirical point (M11) keeps its vectors: no
    shift, the realized utility measured."""
    calibration = _current(scenario, setup, candidate)
    if scenario.utility != "safe":
        plain = replace(calibration, mmlu_shift={}, factual_shift={}, utility_lambda={},
                        utility_flagged=False)  # fmt: skip
        draws = _draws(scenario, _with(setup, scenario, candidate, plain), candidate, seed,
                       studies)  # fmt: skip
        return replace(plain, utility_realized=_utility_realized(draws, plain.failure))
    draws = _draws(scenario, setup, candidate, seed, studies)
    mmlu_shift, factual_shift, lambdas, flagged = _safe_shifts(
        draws, calibration.failure, setup.vectors
    )
    shifted = replace(calibration, mmlu_shift=mmlu_shift, factual_shift=factual_shift,
                      utility_lambda=lambdas, utility_flagged=flagged)  # fmt: skip
    measured = _draws(scenario, _with(setup, scenario, candidate, shifted), candidate, seed,
                      studies)  # fmt: skip
    return replace(shifted, utility_realized=_utility_realized(measured, calibration.failure))


def study_variances(
    scenario: Scenario,
    draws: _Draws,
    failure: float,
    sigma: float,
    comparison: str,
    offset: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per calibration study, E[s² | blocks] of d over its supported validation blocks at f (M1),
    the arm's block effect σ·e + ``offset`` (proportions²), and which studies have the two
    supported blocks a variance needs: the sample variance of the blocks' expected judged
    differences plus the mean of their label variances (``label_variance``), the expectation of
    the observed d's sample variance given the blocks, so that the labels count exactly and
    every σ reuses the same draws."""
    arm = COMPARISONS[comparison][0]
    a, f_ = ARM_ORDER.index(arm), ARM_ORDER.index("F")
    mask = draws.supported(comparison, failure)
    effect = draws.effects[:, :, 0 if arm == "A" else 1]
    g = scenario.curvature_scale

    def harm(index: int, which: np.ndarray, extra: np.ndarray | float) -> np.ndarray:
        name = ARM_ORDER[index]
        x = np.where(mask, which[:, :, index], 0.0)
        return np.clip(harm_curve(name, x, g) + extra, 0.0, 1.0)

    shift = sigma * effect + offset
    h = np.stack([harm(a, draws.x_lo, shift), harm(a, draws.x_hi, shift),
                  harm(f_, draws.x_lo, 0.0), harm(f_, draws.x_hi, 0.0)], axis=-1)  # fmt: skip
    wa, wf = np.where(mask, draws.w[:, :, a], 0.0), np.where(mask, draws.w[:, :, f_], 0.0)
    c = np.stack([1 - wa, wa, -(1 - wf), -wf], axis=-1)
    arms = (arm, arm, "F", "F")
    p = np.stack([judged_share(name, h[..., j], scenario) for j, name in enumerate(arms)], -1)
    mu = (c * p).sum(axis=-1)
    noise = label_variance(h, c, arms, scenario)
    n = mask.sum(axis=1)
    keep = n >= 2
    n_, mask_ = n[keep], mask[keep]
    mean = np.where(mask_, mu[keep], 0.0).sum(axis=1) / n_
    ss = np.where(mask_, (mu[keep] - mean[:, None]) ** 2, 0.0).sum(axis=1)
    variance = ss / (n_ - 1) + np.where(mask_, noise[keep], 0.0).sum(axis=1) / n_
    return variance, keep


def _spread(scenario: Scenario, draws: _Draws, failure: float, sigma: float,
            comparison: str, offset: float = 0.0) -> tuple[float, float]:  # fmt: skip
    """The observed spread of d over the supported validation blocks at f, the arm's offset
    ``offset`` (M1): the square root of the mean within-study variance (``study_variances``),
    the quantity the spread rule estimates, in pp, and its Monte Carlo SE."""
    variance, _ = study_variances(scenario, draws, failure, sigma, comparison, offset)
    total = float(variance.mean())
    se = float(variance.std(ddof=1) / math.sqrt(variance.size)) / (2 * math.sqrt(total))
    return 100 * math.sqrt(total), 100 * se


def _solve_sigma(scenario: Scenario, draws: _Draws, failure: float, comparison: str,
                 offset: float = 0.0) -> float:  # fmt: skip
    """σ by bisection so that the spread at the arm's offset ``offset`` (o⁰: Δ = 0) is the
    planning spread (0 when the labels and the geometry alone already reach it)."""
    target = scenario.spread_pp

    def spread(sigma: float) -> float:
        return _spread(scenario, draws, failure, sigma, comparison, offset)[0]

    if spread(0.0) >= target:
        return 0.0
    lo, hi = 0.0, 0.5
    while spread(hi) < target:
        hi *= 2
        if hi > 64:
            raise CalibrationNotStable(f"{scenario.name}: no σ reaches the spread {target} pp")
    while hi - lo > SIGMA_PRECISION:
        mid = (lo + hi) / 2
        if spread(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def calibrate_spread(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    seed: np.random.SeedSequence,
    *,
    studies: int = CALIBRATION_STUDIES,
) -> tuple[float, float, float, float]:
    """(σ_AF, σ_PF, the attained A–F and P–F spreads in pp) (M1, M2): each σ by bisection so
    that the observed SD of d at Δ = 0 (the setup's o⁰, ``null_offset_a`` and
    ``null_offset_p``), over the supported validation blocks at the rates each selection chose,
    is the planning spread (``_spread``), at the setup's f and shifts."""
    calibration = _current(scenario, setup, candidate)
    draws = _draws(scenario, setup, candidate, seed, studies)
    f = calibration.failure
    null = {"A-F": calibration.null_offset_a, "P-F": calibration.null_offset_p}
    sigma = {c: _solve_sigma(scenario, draws, f, c, null[c]) for c in COMPARISONS}
    spread = {c: _spread(scenario, draws, f, sigma[c], c, null[c])[0] for c in COMPARISONS}
    return sigma["A-F"], sigma["P-F"], spread["A-F"], spread["P-F"]


def _mean_truth(scenario: Scenario, setup: Setup, calibration: Calibration, draws: _Draws,
                comparison: str) -> tuple[float, float]:  # fmt: skip
    """The mean over the selections of the per-study truth (``truth`` at each study's selected
    rates, through the same moments and ``_combine``; studies share rates, rounded to 3
    significant digits) and its Monte Carlo SE, in proportions. A study whose truth is undefined
    (no vector supports its rates) is left out."""
    arms = COMPARISONS[comparison]
    shifted = _shifted(setup.vectors, calibration)
    kstar_ln = math.log(CANDIDATES[calibration.candidate]["targets"]["star"])
    keys = [tuple((r[arm]["l"], r[arm]["u"]) for arm in arms) for r in draws.rates]
    moments: dict[tuple[str, tuple[float, float]], list[tuple[float, float]]] = {}
    for key in set(keys):
        for arm, pair in zip(arms, key, strict=True):
            if (arm, pair) not in moments:
                rates = {"l": pair[0], "u": pair[1]}
                moments[(arm, pair)] = [
                    _cached_moments(v, arm, rates, scenario, calibration, setup.slopes, kstar_ln,
                                    tol=TOLERANCE, max_nodes=MAX_NODES)
                    for v in shifted
                ]  # fmt: skip
    values = {
        key: _combine({arm: moments[(arm, pair)] for arm, pair in zip(arms, key, strict=True)},
                      calibration.failure)[comparison]
        for key in set(keys)
    }  # fmt: skip
    per = np.array([values[k] for k in keys])
    per = per[np.isfinite(per)]
    if per.size < 2:
        raise CalibrationNotStable(f"{scenario.name}: the truth is undefined at the selections")
    return float(per.mean()), float(per.std(ddof=1) / math.sqrt(per.size))


def _solve_offset(scenario: Scenario, setup: Setup, calibration: Calibration, draws: _Draws,
                  comparison: str, target: float) -> float:  # fmt: skip
    """The offset of A (A–F) or P (P–F) at which the mean truth over the selections is
    ``target`` (proportions): the secant method from the current offset, its first step with the
    judged share's slope (the truth rises with the offset, nearly linearly: only the clip bends
    it), until the mean truth is within ``OFFSET_PRECISION`` of the target."""
    arm = COMPARISONS[comparison][0]
    name = "offset_a" if arm == "A" else "offset_p"

    def gap(offset: float) -> float:
        here = replace(calibration, **{name: offset})
        return _mean_truth(scenario, setup, here, draws, comparison)[0] - target

    m_h, m_n = scenario.missing_rates()
    se, fpr = JUDGES[scenario.judge][arm]
    before = calibration.offset(arm)
    g_before = gap(before)
    if abs(g_before) <= OFFSET_PRECISION:
        return before
    now = before - g_before / ((1 - m_h) * se - (1 - m_n) * fpr)
    for _ in range(50):
        g_now = gap(now)
        if abs(g_now) <= OFFSET_PRECISION:
            return now
        if g_now == g_before:
            break
        before, g_before, now = now, g_now, now - g_now * (now - before) / (g_now - g_before)
    raise CalibrationNotStable(f"{scenario.name}: the offset of {arm} does not reach the mean "
                               f"truth {target}")  # fmt: skip


def calibrate_offsets(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    seed: np.random.SeedSequence,
    *,
    studies: int = CALIBRATION_STUDIES,
) -> tuple[float, float]:
    """(o_A, o_P) (M15, M2): each solved, from the setup's o⁰, so that the mean, over the
    selections, of the per-study truth (``truth`` at that study's selected rates) is Δ_AF
    (Δ_PF), at the setup's f, shifts and σ."""
    calibration = _current(scenario, setup, candidate).at_null()
    draws = _draws(scenario, setup, candidate, seed, studies)
    o_a = _solve_offset(scenario, setup, calibration, draws, "A-F", scenario.delta_af_pp / 100)
    o_p = _solve_offset(scenario, setup, calibration, draws, "P-F", scenario.delta_pf_pp / 100)
    return o_a, o_p


def _current(scenario: Scenario, setup: Setup, candidate: int) -> Calibration:
    """The setup's calibration of (scenario, candidate), or where the calibration starts."""
    found = setup.calibrations.get((scenario, candidate))
    return found if found is not None else _starting_calibration(scenario, setup, candidate)


def _starting_calibration(scenario: Scenario, setup: Setup, candidate: int) -> Calibration:
    """Where the calibration starts: no failure, no shift (zero shifts of every vector for the
    safe alternative), no block effect, no offset."""
    zero = {}
    if scenario.utility == "safe":
        zero = {"mmlu_shift": {arm: {v.seed: 0.0 for v in setup.vectors} for arm in ARM_ORDER},
                "factual_shift": dict.fromkeys(ARM_ORDER, 0.0)}  # fmt: skip
    return Calibration(
        candidate=candidate, failure=0.0, realized_support=0.0, support_flagged=False,
        mmlu_shift=zero.get("mmlu_shift", {}), factual_shift=zero.get("factual_shift", {}),
        utility_realized={}, utility_flagged=False, sigma_af=0.0, sigma_pf=0.0, offset_a=0.0,
        offset_p=0.0, attained_af_pp=0.0, attained_pf_pp=0.0, attained_delta_pp={},
    )  # fmt: skip


def calibrate_scenario(
    scenario: Scenario,
    setup: Setup,
    candidate: int,
    seed: np.random.SeedSequence,
    *,
    studies: int = CALIBRATION_STUDIES,
) -> Calibration:
    """Everything calibrated for (scenario, candidate) through the selection, on the one stream
    ``seed`` (``calibration_stream``): f (``calibrate_failure``) and the safe shifts
    (``calibrate_safe``) alternate until neither moves (at most 10 rounds, otherwise
    ``CalibrationNotStable``); σ and the offsets depend on them and not the other way round, so
    they are computed in the round in which nothing moved: σ at the Δ = 0 offsets o⁰ and o⁰ at
    σ alternate until neither moves (at most 10 rounds), then the offsets are solved for the
    scenario's Δ from o⁰, with the attained spreads (at o⁰) and mean truths and their Monte
    Carlo SEs. A gate scenario has no harm: σ and the offsets stay 0. ``fixation`` writes the
    result and ``run`` recomputes it."""
    calibration = _starting_calibration(scenario, setup, candidate)
    before = None
    for _ in range(CALIBRATION_ROUNDS):
        here = _with(setup, scenario, candidate, calibration)
        f, realized, flagged = calibrate_failure(scenario, here, candidate, seed, studies=studies)
        calibration = replace(calibration, failure=f, realized_support=realized,
                              support_flagged=flagged)  # fmt: skip
        calibration = calibrate_safe(scenario, _with(setup, scenario, candidate, calibration),
                                     candidate, seed, studies=studies)  # fmt: skip
        now = (calibration.failure, calibration.mmlu_shift, calibration.factual_shift)
        if now == before:
            break
        before = now
    else:
        raise CalibrationNotStable(
            f"{scenario.name} at candidate {candidate}: f and the safe shifts still moved after "
            f"{CALIBRATION_ROUNDS} rounds"
        )
    if scenario.role == "gate":
        return calibration
    # σ at Δ = 0 (M1): σ given o⁰, then o⁰ (the mean truth 0) given σ, until neither moves
    draws = _draws(scenario, _with(setup, scenario, candidate, calibration), candidate, seed,
                   studies)  # fmt: skip
    f, before = calibration.failure, None
    for _ in range(CALIBRATION_ROUNDS):
        null = {"A-F": calibration.null_offset_a, "P-F": calibration.null_offset_p}
        sigma = {c: _solve_sigma(scenario, draws, f, c, null[c]) for c in COMPARISONS}
        calibration = replace(calibration, sigma_af=sigma["A-F"], sigma_pf=sigma["P-F"])
        probe = calibration.at_null()
        null = {c: _solve_offset(scenario, setup, probe, draws, c, 0.0) for c in COMPARISONS}
        calibration = replace(calibration, null_offset_a=null["A-F"], null_offset_p=null["P-F"])
        now = (sigma["A-F"], sigma["P-F"], null["A-F"], null["P-F"])
        if now == before:
            break
        before = now
    else:
        raise CalibrationNotStable(
            f"{scenario.name} at candidate {candidate}: σ and the Δ = 0 offsets still moved "
            f"after {CALIBRATION_ROUNDS} rounds"
        )
    spread = {c: _spread(scenario, draws, f, s, c, n) for c, s, n in
              (("A-F", calibration.sigma_af, calibration.null_offset_a),
               ("P-F", calibration.sigma_pf, calibration.null_offset_p))}  # fmt: skip
    calibration = replace(calibration, attained_af_pp=spread["A-F"][0],
                          attained_pf_pp=spread["P-F"][0],
                          attained_spread_se_pp={c: v[1] for c, v in spread.items()})  # fmt: skip
    probe = calibration.at_null()
    offset_a = _solve_offset(scenario, setup, probe, draws, "A-F", scenario.delta_af_pp / 100)
    offset_p = _solve_offset(scenario, setup, probe, draws, "P-F", scenario.delta_pf_pp / 100)
    calibration = replace(calibration, offset_a=offset_a, offset_p=offset_p)
    attained = {}
    for comparison in COMPARISONS:
        mean, se = _mean_truth(scenario, setup, calibration, draws, comparison)
        attained[comparison] = (100 * mean, 100 * se)
    return replace(calibration, attained_delta_pp=attained)


# ------------------------------------------------------------------ the fixation

FIXATION_KIND = "simulation-fixation"
STREAMS = {"calibration": 20261101, "verification": 20261102}  # M23
# The procedure's rules the fixation freezes, each with its source (ending "M<n>, decided")
RULES: dict[str, dict[str, str]] = {
    "screen_simplifications": {
        "rule": "the screen runs the scenario's candidate with no fallback (a failed A-F gate ends "
        "the study, counted), takes N from the rule alone and computes P-F in every confirmed "
        "study as if active",
        "source": "the coverage and power tests are per candidate; A-F's analysis set does not "
        "depend on P's activity; M14 and M16, decided",
    },
    "pf_verification": {
        "rule": "in the verification a P-F test counts only the studies in which P-F is active "
        "(its validation gate passed)",
        "source": "P's actual activity in the realized procedure; M16, decided",
    },
    "joint_verification": {
        "rule": "the joint test of the verification",
        "source": "counted over the studies confirmed for A-F; a study misses jointly if A-F "
        "misses, or if P-F is active in it and misses; M27, decided",
    },
    "draw_bound": {
        "rule": "each test draws at most max_draws_factor times M studies; a test short of M "
        "confirmed studies at the bound is not attainable and fails; a power rate counts the "
        "missing studies as not detecting",
        "source": "a bounded failure route; M26, decided",
    },
}
CURVATURE_SOURCE = (
    "study 1's fitted curves (study2_short_sim.STUDY1) at K*, baseline 0.15; M4, decided"
)
CURVATURE_NOTE = (
    "the registration states about 0.06 pp, a fitted estimate with partial cancellation between "
    "arms"
)


def curvature_record() -> dict[str, Any]:
    """The base κ of each arm (pp per (ln K)², read by the curvature sensitivity) and the
    nominal A–F bias at the symmetric bracket, with M4's disclosure, rounded to 4 decimals."""
    return {
        **{arm: round(CURVATURE_PP[arm], 4) for arm in ARM_ORDER},
        "unit": "pp per (ln K)^2",
        "nominal_bias_af_pp": round(NOMINAL_BIAS_AF_PP, 4),
        "source": CURVATURE_SOURCE,
        "note": CURVATURE_NOTE,
    }


def machine_record() -> dict[str, str]:
    """The machine a fixation is calibrated on (reported, never refused on): the platform, the
    CPU model and numpy's version."""
    import platform

    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"platform": platform.platform(), "cpu": cpu or platform.processor() or "unknown",
            "numpy": np.__version__}  # fmt: skip


def calibration_json(c: Calibration) -> dict[str, Any]:
    """A calibration as JSON (vector seeds as strings, pairs as lists)."""
    return {
        "candidate": c.candidate,
        "failure": c.failure,
        "realized_support": c.realized_support,
        "support_flagged": c.support_flagged,
        "mmlu_shift": {
            arm: {str(seed): v for seed, v in d.items()} for arm, d in c.mmlu_shift.items()
        },  # fmt: skip
        "factual_shift": dict(c.factual_shift),
        "utility_realized": {arm: list(v) for arm, v in c.utility_realized.items()},
        "utility_flagged": c.utility_flagged,
        "utility_lambda": dict(c.utility_lambda),
        "sigma_af": c.sigma_af,
        "sigma_pf": c.sigma_pf,
        "offset_a": c.offset_a,
        "offset_p": c.offset_p,
        "null_offset_a": c.null_offset_a,
        "null_offset_p": c.null_offset_p,
        "attained_af_pp": c.attained_af_pp,
        "attained_pf_pp": c.attained_pf_pp,
        "attained_spread_se_pp": dict(c.attained_spread_se_pp),
        "attained_delta_pp": {k: list(v) for k, v in c.attained_delta_pp.items()},
    }


def calibration_from_json(d: Mapping[str, Any]) -> Calibration:
    """The calibration of ``calibration_json``. ``ValueError`` for another field."""
    names = {f.name for f in fields(Calibration)}
    if not isinstance(d, Mapping) or set(d) != names:
        raise ValueError(f"a calibration holds exactly {sorted(names)}")
    return Calibration(
        candidate=d["candidate"],
        failure=float(d["failure"]),
        realized_support=float(d["realized_support"]),
        support_flagged=d["support_flagged"],
        mmlu_shift={
            arm: {int(seed): float(v) for seed, v in shifts.items()}
            for arm, shifts in d["mmlu_shift"].items()
        },  # fmt: skip
        factual_shift={arm: float(v) for arm, v in d["factual_shift"].items()},
        utility_realized={
            arm: tuple(float(x) for x in v) for arm, v in d["utility_realized"].items()
        },  # fmt: skip
        utility_flagged=d["utility_flagged"],
        utility_lambda={arm: float(v) for arm, v in d["utility_lambda"].items()},
        sigma_af=float(d["sigma_af"]),
        sigma_pf=float(d["sigma_pf"]),
        offset_a=float(d["offset_a"]),
        offset_p=float(d["offset_p"]),
        null_offset_a=float(d["null_offset_a"]),
        null_offset_p=float(d["null_offset_p"]),
        attained_af_pp=float(d["attained_af_pp"]),
        attained_pf_pp=float(d["attained_pf_pp"]),
        attained_spread_se_pp={k: float(v) for k, v in d["attained_spread_se_pp"].items()},
        attained_delta_pp={
            k: (float(v[0]), float(v[1])) for k, v in d["attained_delta_pp"].items()
        },  # fmt: skip
    )


def build_fixation(
    vectors_path: str | Path,
    scenarios: Sequence[Scenario],
    calibrations: Mapping[tuple[Scenario, int], Calibration],
    *,
    code: Mapping[str, str],
    streams: Mapping[str, int],
    repo: str | Path | None = None,
    policy: Mapping[str, Any] | None = None,
    written_at: str | None = None,
    machine: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """The simulation's fixation (``docs/study3-simulation-fixation.json``, committed alone before
    the first draw; the closed schema ``study3_decisions.fixation_problems`` reads): the code;
    the ``machine`` it is calibrated on (``machine_record``, reported: ``run`` names a different
    one and goes on); the registration's ``policy`` (``study3_versions.registered_policy`` at
    HEAD of ``repo``, unless given); the inputs record (``vectors_path``, in the checkout) by its
    sha256; M, the resamples, the ``streams``, the levels, N points and p₀, the gate replicates
    (M) and the draw bound; the procedure's ``rules`` and the base ``curvature``; every scenario
    with its calibrations per candidate (a coverage scenario of candidate 1 also at candidate 2,
    the verification's fallback); the coverage tests' explicit identities, their count and sha256,
    J, the thresholds and the detection powers."""
    import datetime

    from ..paths import REPO_ROOT
    from .study3_decisions import COVERAGE_P0, COVERAGE_RUNS

    repo = Path(REPO_ROOT if repo is None else repo)
    if policy is None:
        from .study3_versions import registered_policy

        policy = registered_policy(repo)
    vectors = Path(vectors_path)
    full = vectors if vectors.is_absolute() else repo / vectors
    try:
        relative = full.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{vectors_path}: the inputs record is not in the checkout") from exc
    entries = []
    for sc in scenarios:
        mine = {c: cal for (s, c), cal in calibrations.items() if s == sc}
        if sc.candidate not in mine:
            raise ValueError(f"scenario {sc.name}: no calibration at its candidate")
        if sc.coverage and sc.candidate == 1 and 2 not in mine:
            raise ValueError(f"scenario {sc.name}: no calibration at candidate 2, the "
                             "verification's fallback")  # fmt: skip
        calibrated = {str(c): calibration_json(mine[c]) for c in sorted(mine)}
        entries.append({**sc.to_json(), "calibration": calibrated})
    coverage = [sc.name for sc in scenarios if sc.coverage]
    tests = test_identities(coverage)
    j, m = len(tests), COVERAGE_RUNS
    return {
        "kind": FIXATION_KIND,
        "written_at": written_at
        or datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "code": dict(code),
        "machine": dict(machine_record() if machine is None else machine),
        "policy": dict(policy),
        "inputs": {"path": relative, "sha256": hashlib.sha256(full.read_bytes()).hexdigest()},
        "M": m,
        "resamples": {"screen": SCREEN_RESAMPLES, "verification": VERIFICATION_RESAMPLES},
        "streams": {
            "calibration": int(streams["calibration"]),
            "verification": int(streams["verification"]),
        },  # fmt: skip
        "levels": list(LEVELS),
        "n_points": list(N_POINTS),
        "p0": dict(COVERAGE_P0),
        "gate_replicates": m,
        "max_draws_factor": MAX_DRAWS_FACTOR,
        "rules": {name: dict(rule) for name, rule in RULES.items()},
        "curvature": curvature_record(),
        "scenarios": entries,
        "tests": {
            "list": [list(i) for i in tests],
            "count": j,
            "sha256": identities_sha256(tests),
        },  # fmt: skip
        "J": j,
        "thresholds": {name: coverage_threshold(j, m, p0) for name, p0 in COVERAGE_P0.items()},
        "detection_power": detection_power(j, m),
    }


def read_fixation(path: str | Path, repo: str | Path) -> dict[str, Any]:
    """The committed fixation at ``path`` in the checkout ``repo`` (``SIMULATION_FIXATION``),
    as the J check reads it (``study3_decisions.fixation_problems``): committed once, alone and
    unchanged at HEAD (so in HEAD's history: HEAD or an ancestor of it holds it), its schema
    closed and its tests exactly its coverage scenarios' identities. Returns ``fixation``,
    ``commit``, ``sha256``, the ``scenarios`` (``Scenario``) and ``calibrations`` ((scenario,
    candidate) → ``Calibration``). ``ValueError`` naming every problem."""
    repo = Path(repo)
    full = Path(path) if Path(path).is_absolute() else repo / path
    try:
        relative = full.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path}: the fixation is not in the checkout {repo}") from exc
    if relative != SIMULATION_FIXATION:
        raise ValueError(f"{path}: the simulation's fixation is {SIMULATION_FIXATION}")
    fixation, commit, problems = fixation_problems(repo, relative)
    if fixation is None or problems:
        raise ValueError("; ".join(problems))
    scenarios, calibrations = [], {}
    for entry in fixation["scenarios"]:
        sc = Scenario.from_json({k: v for k, v in entry.items() if k != "calibration"})
        scenarios.append(sc)
        for candidate, value in entry["calibration"].items():
            calibrations[(sc, int(candidate))] = calibration_from_json(value)
    return {
        "fixation": fixation,
        "commit": commit,
        "sha256": hashlib.sha256(full.read_bytes()).hexdigest(),
        "scenarios": scenarios,
        "calibrations": calibrations,
    }
