"""Analysis of the decisive comparison (docs/preregistration.md#primary-analysis).

Typical use: ``select_k_star(sweep_runs)`` once before look 1, then ``analyze(look_runs,
selection, ...)`` at each look and ``sequential_decision`` across looks; ``fallback_contrast`` for
the unequal-displacement fallback. ``simulate`` calibrates the whole procedure by simulation.
"""

from .bootstrap import CI_METHODS, BootstrapDraws, confidence_interval, paired_resampling
from .curves import LogisticFit, fit_logistic
from .decisions import (
    DECISIONS,
    EQUIVALENCE_MARGIN,
    h2_comparison,
    h2_mechanism,
    look_decision,
    sequential_decision,
)
from .kstar import CommonRangeUndefined, KStarSelection, cell_means, select_k_star
from .primary import AnalysisResult, FallbackResult, analyze, curve_draws, fallback_contrast
from .runs import ARMS, CONTROL_ARMS, PRIMARY_ARMS, validate_runs

__all__ = [
    "ARMS",
    "CI_METHODS",
    "CONTROL_ARMS",
    "DECISIONS",
    "EQUIVALENCE_MARGIN",
    "PRIMARY_ARMS",
    "AnalysisResult",
    "BootstrapDraws",
    "CommonRangeUndefined",
    "FallbackResult",
    "KStarSelection",
    "LogisticFit",
    "analyze",
    "cell_means",
    "confidence_interval",
    "curve_draws",
    "fallback_contrast",
    "fit_logistic",
    "h2_comparison",
    "h2_mechanism",
    "look_decision",
    "paired_resampling",
    "select_k_star",
    "sequential_decision",
    "validate_runs",
]
