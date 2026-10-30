"""Study 3's simulation before validation: the scenario models and the exact truth.

docs/study3-preregistration.md, "Simulation before validation": a CPU Monte Carlo of the whole
procedure (search, support, utility, validation, N, confirmation and analysis) over a frozen grid
of scenarios, with the coverage of the interval at level L screened against the registered
estimand ("Questions and estimands", "The comparisons": the expected interpolated difference over
fresh supported seed blocks, under the frozen rates). This module holds the scenarios, the harm
and label models and the truth that the coverage check compares with. Pure numpy and scipy on
the CPU; it reads no run directory, no harm label, response or judge output.

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

**Rulings** (the study's frozen settings, each with its one-line source; M1, M7, M10 and M15
are calibrated through the selection by the procedure's part of this module):
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
- M8, widths and jitter: the vectors' own widths; C12 adds N(0, σ_e) per run in ln K, σ_e A 0.013,
  F 0.032, P 0.027 (docs/study2-short-b20-inputs.json); decided.
- The constants taken from study 2's simulation (``STUDY1``'s logistic slopes and curvatures,
  the shapes' constants, ρ_a, ρ_b and ``THREAD_VARIABLES``) are written out here so that this
  module loads no GPU path through study 2's analysis; a test compares them with
  ``study2_short_sim``.
- The bracket compares ln K with ln K* (the measured K's order), the width allows 10⁻⁹ in ln K and
  a usable run's MMLU loss 10⁻⁹ pp, as ``study3_blocks`` and ``study3_records`` do.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
from scipy.special import gammainc, ndtr, ndtri

from .study3_blocks import SLACK as WIDTH_SLACK
from .study3_blocks import WIDTH
from .study3_records import GROSS_COLLAPSE_PP, TOLERANCE_PP
from .study3_search import ARM_ORDER, CANDIDATES
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
    mean truth over the selections and its Monte Carlo SE, pp; M15)."""

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

    def __post_init__(self) -> None:
        if self.candidate not in CANDIDATES or isinstance(self.candidate, bool):
            raise ValueError(f"no candidate {self.candidate!r}")
        if not (math.isfinite(self.failure) and 0 <= self.failure < 1):
            raise ValueError(f"the per-run failure {self.failure!r} is not in [0, 1)")
        for name in ("sigma_af", "sigma_pf"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} {value!r} is not a finite SD")
        for name in ("offset_a", "offset_p"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} is not finite")

    def sigma(self, arm: str) -> float:
        """The SD of arm ``arm``'s block effect (F has none)."""
        return {"A": self.sigma_af, "F": 0.0, "P": self.sigma_pf}[arm]

    def offset(self, arm: str) -> float:
        return {"A": self.offset_a, "F": 0.0, "P": self.offset_p}[arm]


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
    shifted = shifted_vectors(vectors, calibration)
    kstar_ln = math.log(CANDIDATES[calibration.candidate]["targets"]["star"])
    arms = [arm for arm in ARM_ORDER if arm in rates]
    survive = (1 - calibration.failure) ** 2
    moments = {
        arm: [
            _arm_moments(
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
