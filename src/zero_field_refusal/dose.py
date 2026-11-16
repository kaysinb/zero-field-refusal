"""Study 5's dose experiment (docs/study5-dose-registration.md): its rates, its seeds, the pods'
shares of them and the rule that fixes the common late grid from the pilot's K_benign.

Each run is arm F on its seed's own stream order and v₀ (``experiments/study5_dose``), 25
updates of B = 20, with the rate ``EARLY`` on updates 1 … ``EARLY_STEPS`` and the late rate
(``optimizer.lr``) on the rest (``train.rate_schedule``, ``study5_options``). The early levels
are 0.6, 1.0 and 1.25 × η_l; the late rate is one of two levels L_lo < L_hi common to every
early level, chosen among ``LATE_CANDIDATES`` (0.75, 1.3 and 2.0 × η_l) by ``grid_choice`` from
the pilot. The rates are written as decimal literals, so that a config, a run record and this
module name the same float.

**Seeds.** ``draw_seeds``: 16 of ``POPULATION`` (the 29 fresh seeds outside the ten held out of
the analysis of the existing runs, each with h at K\\*), drawn by
``numpy.random.default_rng(derive_seed(0, "study5", "dose-seeds")).choice(29, 16,
replace=False)`` over the population in seed order. ``assign``: pod A takes the drawn seed with
the smallest K_benign at η_l (pod 2's fresh run) as its pilot seed, pod B the largest; the other
14 alternate A, B in drawn order. The record is ``docs/study5-dose-seeds.json``
(``seeds_record``), committed before any dose run.

**The grid rule** (``grid_choice``), fixed before the pilot: the pilot is the two pilot seeds ×
the three early levels × the three late candidates (18 runs, 9 per pod). A pilot cell (seed,
early level) covers K\\* under a pair of candidates when both runs ended all 25 updates with a
finite, positive K_benign at step 25, the two K differ, and K\\* lies between them (inclusive),
the bracket of study 3's support rule without its width limit. Of the three pairs, the one that
covers K\\* in the most of the six cells is chosen; on a tie, the narrower pair (the smaller
ratio of its rates). The pilot's runs at the chosen levels are cells of the main experiment.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .seeding import derive_seed

ETA_L = 1.3e-05  # study 3's candidate-1 η_l of F
EARLY_STEPS = 5  # the updates at the early rate: 1 … 5
EARLY = (("0.6", 7.8e-06), ("1.0", 1.3e-05), ("1.25", 1.625e-05))  # × η_l, the rate
LATE_CANDIDATES = (("0.75", 9.75e-06), ("1.3", 1.69e-05), ("2.0", 2.6e-05))  # × η_l, the rate
K_STAR = 6.0  # study 3's candidate-1 comparison point in K_benign
STEPS, BATCH = 25, 20
POPULATION = (
    50000, 50002, 50003, 50004, 50006, 50007, 50008, 50009, 50010, 50012, 50013, 50014, 50015,
    50016, 50018, 50019, 50020, 50022, 50023, 50025, 50027, 50028, 50029, 50031, 50032, 50033,
    50035, 50036, 50039,
)  # fmt: skip
HELD_OUT = (50001, 50011, 50017, 50021, 50024, 50026, 50030, 50034, 50037, 50038)
DRAW_KEYS = ("study5", "dose-seeds")
N_SEEDS = 16
PODS = ("a", "b")
SEEDS_FILE = "docs/study5-dose-seeds.json"


def early_rate(level: str) -> float:
    return dict(EARLY)[level]


def late_rate(level: str) -> float:
    return dict(LATE_CANDIDATES)[level]


def draw_seeds() -> list[int]:
    """The 16 seeds in the order drawn."""
    rng = np.random.default_rng(derive_seed(0, *DRAW_KEYS))
    return [POPULATION[int(i)] for i in rng.choice(len(POPULATION), N_SEEDS, replace=False)]


def assign(drawn: Sequence[int], k_eta_l: Mapping[int, float]) -> dict[str, dict[str, Any]]:
    """Each pod's pilot seed and seeds (module docstring): A the smallest K_benign at η_l, B the
    largest, the other seeds alternating A, B in drawn order."""
    if len(set(drawn)) != len(drawn) or any(s not in k_eta_l for s in drawn):
        raise ValueError("distinct drawn seeds, each with K_benign at η_l, are expected")
    low = min(drawn, key=lambda s: (k_eta_l[s], s))
    high = max(drawn, key=lambda s: (k_eta_l[s], -s))
    if low == high:
        raise ValueError("the pilot needs two seeds")
    rest = [s for s in drawn if s not in (low, high)]
    return {
        "a": {"pilot": low, "seeds": [low, *rest[0::2]]},
        "b": {"pilot": high, "seeds": [high, *rest[1::2]]},
    }


def seeds_record(k_eta_l: Mapping[int, float], runs: Mapping[int, str]) -> dict[str, Any]:
    """``docs/study5-dose-seeds.json``: the population, the rule, the draw, each drawn seed's
    K_benign at η_l with its run, and the pods' shares."""
    drawn = draw_seeds()
    return {
        "what": "The seeds of study 5's dose experiment and each pod's share of them.",
        "population": list(POPULATION),
        "held_out": list(HELD_OUT),
        "rule": "numpy.random.default_rng(derive_seed(0, 'study5', 'dose-seeds'))"
        ".choice(29, 16, replace=False) over the population in seed order",
        "rng_seed": derive_seed(0, *DRAW_KEYS),
        "drawn_in_order": drawn,
        "seeds": sorted(drawn),
        "k_benign_eta_l": {str(s): float(k_eta_l[s]) for s in sorted(drawn)},
        "k_source_runs": {str(s): runs[s] for s in sorted(drawn)},
        "pods": assign(drawn, k_eta_l),
    }


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def covers(k_a: Any, k_b: Any, kstar: float = K_STAR) -> bool:
    """Whether two runs' K_benign bracket K* (module docstring, "The grid rule")."""
    if not (_number(k_a) and _number(k_b)) or k_a <= 0 or k_b <= 0 or k_a == k_b:
        return False
    return min(k_a, k_b) <= kstar <= max(k_a, k_b)


def candidate_pairs() -> list[tuple[str, str]]:
    """The three pairs of late candidates, each (lower, higher)."""
    names = [name for name, _ in LATE_CANDIDATES]
    return [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]


def grid_choice(rows: Iterable[Mapping[str, Any]], pilot_seeds: Sequence[int]) -> dict[str, Any]:
    """The common late grid from the pilot's runs. ``rows``: one per pilot run, ``seed``,
    ``early`` and ``late`` (level names), ``K`` (K_benign at step 25, or None) and ``complete``
    (all 25 updates, ended by ``steps``). Exactly the 2 × 3 × 3 runs are expected; ``ValueError``
    otherwise. Returns every pair's covered cells and the chosen pair."""
    table: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["early"]), str(row["late"]))
        if key in table:
            raise ValueError(f"two pilot runs of {key}")
        table[key] = row
    expected = {(int(s), e, late) for s in pilot_seeds for e, _ in EARLY
                for late, _ in LATE_CANDIDATES}  # fmt: skip
    if len(set(pilot_seeds)) != 2 or set(table) != expected:
        raise ValueError(f"the pilot's 18 runs of two seeds are expected, got {sorted(table)}")

    def k_of(seed: int, early: str, late: str) -> float | None:
        row = table[(seed, early, late)]
        return row["K"] if row.get("complete") is True and _number(row.get("K")) else None

    cells = [(int(s), e) for s in pilot_seeds for e, _ in EARLY]
    pairs = []
    for lo, hi in candidate_pairs():
        covered = [{"seed": s, "early": e} for s, e in cells
                   if covers(k_of(s, e, lo), k_of(s, e, hi))]  # fmt: skip
        ratio = late_rate(hi) / late_rate(lo)
        pairs.append({"lo": lo, "hi": hi, "covered": len(covered), "cells": covered,
                      "ratio": ratio})  # fmt: skip
    chosen = min(pairs, key=lambda p: (-p["covered"], p["ratio"]))
    return {
        "k_star": K_STAR,
        "cells": len(cells),
        "pairs": pairs,
        "chosen": {"lo": chosen["lo"], "hi": chosen["hi"], "L_lo": late_rate(chosen["lo"]),
                   "L_hi": late_rate(chosen["hi"]), "covered": chosen["covered"]},
        "K": {f"s{s}/{e}/{late}": k_of(s, e, late) for s, e, late in sorted(expected)},
    }  # fmt: skip
