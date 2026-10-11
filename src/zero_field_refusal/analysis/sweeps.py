"""The choice of the stage-1 sweep rates from the seed-0 pilots (preregistration, "Stage-1 sweeps").

Only K, divergence and utility are read (``meta.json``, ``config.yaml``, ``summary.json``), never
a judge label. Every run directory must hold the fixed condition of its experiment
(``gate1.committed_condition``: the committed config with the run's arm and rate), or reading
stops. Attempts of one logical run (arm, rate, seed, run type) are merged: a divergence and a
completion of the same run, or two completions, are contradictions and stop reading; otherwise the
run is *completed*, *diverged*, *pending* (an attempt still running, or fewer than three failed
attempts) or *missing* (three failed attempts).

A pilot is a *candidate* when its run completed with a positive, finite K and it is stable:
neither diverged nor collapsed, collapse being a factual accuracy more than 10 pp, or a benchmark
accuracy more than 5 pp, below θ₀'s (accuracies must be finite numbers in [0, 1]). The target range
[K_A,min, K_A,max] spans the cell-mean K (geometric mean over completed runs, each counted once) of
the model's AdamW cells under the main condition. Target points, in order: log K_A,min,
log K_A,max, then the evenly spaced points between them on the log scale (one for frozen Adam, two
for SGD). For each point the not yet chosen candidate with the nearest log K is chosen, a tie going
to the smaller rate; for SGD the largest stable candidate replaces the last choice if it was not
chosen. If no candidate lies within a factor 2 of K_A,min (or of K_A,max), the grid is extended by
one step in that direction (a factor 3 for SGD, 2 for frozen Adam), at most twice per end, and the
extension is piloted before the choice. While a needed pilot has no known outcome the result is
"wait", never a choice.

Refinement (amendment of 2026-10-11): once the outer extensions are settled, if no stable pilot
reaches K_A,max, η_L is the largest stable pilot rate and η_U the smallest collapsed or divergent
rate above it (technical failures and unfinished runs are no endpoints); with both, the rates
η_L (η_U/η_L)^(j/4), j = 1, 2, 3, are piloted at seed 0, the bracket is recomputed, and at most two
such rounds are made. Refinement stops early when a stable pilot reaches K_A,max or no ordered
bracket remains; the choice rule above is then applied to all pilots.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .evidence import arm_of, differing_keys, run_condition

FACTUAL_DROP, BENCHMARK_DROP = 0.10, 0.05  # collapse beyond these drops below θ₀
TOLERANCE = 1e-9  # accuracies are fractions of counts; equality at a threshold is no collapse
NEAR = math.log(2.0)  # a candidate within a factor 2 of a target end reaches it
MAX_EXTENSIONS = 2  # per end
MAX_ATTEMPTS = 3  # launch.MAX_ATTEMPTS: failed attempts after which a run is missing
REFINE_ROUNDS, REFINE_POINTS = 2, 3  # rounds of refinement, rates per round
PLACES = {"sgd": 4, "adam_frozen": 3}
STEP = {"sgd": 3.0, "adam_frozen": 2.0}
KNOWN = ("completed", "diverged", "missing")  # outcomes; "pending" is not one


@dataclass(frozen=True)
class Pilot:
    lr: float
    status: str  # completed | diverged | pending | missing
    k: float | None
    factual: float | None
    benchmark: float | None


@dataclass
class Choice:
    arm: str
    grid: list[float]  # the rates of the base grid
    candidates: list[dict[str, Any]]
    targets: list[float]  # target K values, in order
    status: str = "chosen"  # chosen | extend | refine | wait
    chosen: list[float] = field(default_factory=list)
    extend: list[float] = field(default_factory=list)  # outer extensions to pilot first
    refine: list[float] = field(default_factory=list)  # refinement rates to pilot first
    bracket: list[float] | None = None  # (η_L, η_U) of the last refinement considered
    pending: list[float] = field(default_factory=list)  # pilots without a known outcome yet
    notes: list[str] = field(default_factory=list)


def _accuracy(value: Any) -> bool:
    return (isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
            and 0.0 <= value <= 1.0)  # fmt: skip


def unstable_reason(pilot: Pilot, theta0: Mapping[str, float]) -> str | None:
    """Why a pilot is not a candidate (None: it is one)."""
    if pilot.status == "diverged":
        return "diverged"
    if pilot.status != "completed":
        return f"status {pilot.status!r}"
    if pilot.k is None or not math.isfinite(pilot.k) or pilot.k <= 0:
        return f"K {pilot.k!r}"
    if not (_accuracy(pilot.factual) and _accuracy(pilot.benchmark)):
        return "utility missing or invalid"
    if theta0["factual"] - pilot.factual > FACTUAL_DROP + TOLERANCE:
        return "collapse: factual accuracy"
    if theta0["benchmark"] - pilot.benchmark > BENCHMARK_DROP + TOLERANCE:
        return "collapse: benchmark accuracy"
    return None


def _bracket(pilots: Sequence[Pilot],
             candidates: Sequence[Mapping[str, Any]]) -> tuple[float, float] | None:  # fmt: skip
    """(η_L, η_U): the largest stable rate and the smallest collapsed or divergent rate above it."""
    stable = [p.lr for p, c in zip(pilots, candidates, strict=True) if c["candidate"]]
    if not stable:
        return None
    low = max(stable)

    def ended_badly(p: Pilot, c: Mapping[str, Any]) -> bool:
        # Collapse, divergence or a non-finite K; never a technical failure or an unfinished run.
        reason = c["reason"] or ""
        collapsed = reason.startswith("collapse") or reason.startswith("K")
        return p.status == "diverged" or (p.status == "completed" and collapsed)

    above = [p.lr for p, c in zip(pilots, candidates, strict=True)
             if p.lr > low * (1 + 1e-9) and ended_badly(p, c)]  # fmt: skip
    return (low, min(above)) if above else None


def target_points(k_min: float, k_max: float, places: int) -> list[float]:
    """log K_A,min, log K_A,max, then the evenly spaced interior points, as K values."""
    lo, hi = math.log(k_min), math.log(k_max)
    interior = [lo + (hi - lo) * j / (places - 1) for j in range(1, places - 1)]
    return [math.exp(x) for x in (lo, hi, *interior)]


def _same(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9)


def choose(arm: str, grid: Sequence[float], pilots: Iterable[Pilot],
           theta0: Mapping[str, float], k_min: float, k_max: float) -> Choice:  # fmt: skip
    """The rates for ``arm`` (sgd or adam_frozen) from its pilots, the extensions to pilot, or
    "wait" while a needed pilot has no known outcome."""
    for name in ("factual", "benchmark"):
        if not _accuracy(theta0.get(name)):
            raise ValueError(f"θ₀'s {name} accuracy {theta0.get(name)!r} is not in [0, 1]")
    pilots = sorted(pilots, key=lambda p: p.lr)
    candidates = []
    for p in pilots:
        reason = unstable_reason(p, theta0)
        candidates.append(asdict(p) | {"candidate": reason is None, "reason": reason})
    choice = Choice(arm, sorted(grid), candidates, target_points(k_min, k_max, PLACES[arm]))
    known = [p for p in pilots if p.status in KNOWN]
    unrun = [lr for lr in grid if not any(_same(lr, p.lr) for p in pilots)]
    choice.pending = sorted(unrun + [p.lr for p in pilots if p.status not in KNOWN])
    if choice.pending:
        choice.status = "wait"
        return choice
    usable = [p for p, c in zip(pilots, candidates, strict=True) if c["candidate"]]

    def reaches(target: float) -> bool:
        return any(abs(math.log(p.k) - math.log(target)) <= NEAR + 1e-12 for p in usable)

    rates = [p.lr for p in known]
    lows = [min(grid) / STEP[arm] ** n for n in range(1, MAX_EXTENSIONS + 1)]
    highs = [max(grid) * STEP[arm] ** n for n in range(1, MAX_EXTENSIONS + 1)]
    below = [lr for lr in rates if any(_same(lr, x) for x in lows)]
    above = [lr for lr in rates if any(_same(lr, x) for x in highs)]
    if not reaches(k_min) and len(below) < MAX_EXTENSIONS:
        choice.extend.append(lows[len(below)])
    if not reaches(k_max) and len(above) < MAX_EXTENSIONS:
        choice.extend.append(highs[len(above)])
    if choice.extend:
        choice.status = "extend"
        return choice
    outer = [*grid, *lows, *highs]
    refined = [lr for lr in rates if not any(_same(lr, x) for x in outer)]
    rounds = math.ceil(len(refined) / REFINE_POINTS)
    if not reaches(k_max) and rounds < REFINE_ROUNDS:
        bracket = _bracket(pilots, candidates)
        choice.bracket = list(bracket) if bracket else None
        if bracket is not None:
            low, high = bracket
            new = [low * (high / low) ** (j / 4) for j in range(1, REFINE_POINTS + 1)]
            new = [lr for lr in new if not any(_same(lr, x) for x in rates)]
            if new:
                choice.refine, choice.status = new, "refine"
                return choice
    if refined:
        choice.notes.append(f"{rounds} refinement round(s) made")
    for name, target in (("K_A,min", k_min), ("K_A,max", k_max)):
        if not reaches(target):
            choice.notes.append(f"no candidate within a factor 2 of {name} after the extensions")
    chosen: list[Pilot] = []
    for target in choice.targets:
        left = [p for p in usable if p not in chosen]
        if not left:
            break
        t = math.log(target)
        chosen.append(min(left, key=lambda p: (round(abs(math.log(p.k) - t), 12), p.lr)))
    if arm == "sgd" and usable:
        largest = max(usable, key=lambda p: p.lr)
        if largest not in chosen:
            chosen[-1] = largest
    choice.chosen = sorted(p.lr for p in chosen)
    if not usable:
        choice.notes.append("no stable candidate")
    return choice


# --- reading run directories --------------------------------------------------------------------


def read_logical_runs(root: Path, experiment: str) -> list[dict[str, Any]]:
    """The logical runs of ``root/experiment``, each once, with their outcome, K and utility.

    Raises ``ValueError`` if a run departs from the experiment's committed condition, or if the
    attempts of one run contradict each other.
    """
    from .gate1 import committed_condition  # gate1 imports this package's modules too

    attempts: dict[tuple[str, float, int, str], list[dict[str, Any]]] = {}
    departing = []
    for run_dir in sorted(p for p in (Path(root) / experiment).glob("*") if p.is_dir()):
        try:
            meta = json.loads((run_dir / "meta.json").read_text())
            cfg = OmegaConf.load(run_dir / "config.yaml")
        except (OSError, ValueError):
            continue  # killed before its metadata: a pending or failed attempt without a record
        expected, optimizer = committed_condition(experiment, cfg)
        actual = OmegaConf.to_container(cfg.optimizer, resolve=True)
        if differing_keys([run_condition(cfg) | {"optimizer": actual},
                           expected | {"optimizer": optimizer}]):  # fmt: skip
            departing.append(run_dir.name)
            continue
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        k = summary.get("K_harmful")
        key = (arm_of(cfg), float(cfg.optimizer.lr), int(cfg.seed), cfg.get("run_type", "train"))
        attempts.setdefault(key, []).append({
            "run_id": run_dir.name, "status": meta.get("status"),
            "k": float(k) if isinstance(k, int | float) and not isinstance(k, bool) else None,
            "factual": summary.get("factual_accuracy"),
            "benchmark": summary.get("benchmark_accuracy"),
        })  # fmt: skip
    if departing:
        raise ValueError(f"Runs of {experiment} depart from its committed config: {departing}")
    runs = []
    for (arm, lr, seed, run_type), group in sorted(attempts.items()):
        statuses = [a["status"] for a in group]
        completed = [a for a in group if a["status"] == "completed"]
        name = f"{experiment} {arm} {lr:g} seed {seed} {run_type}"
        if "diverged" in statuses and completed:
            raise ValueError(f"{name}: a diverged and a completed attempt contradict each other")
        if len(completed) > 1:
            raise ValueError(f"{name}: {len(completed)} completed attempts of one run")
        if "diverged" in statuses:
            outcome, values = "diverged", {"k": None, "factual": None, "benchmark": None}
        elif completed:
            outcome, values = "completed", completed[0]
        elif any(s not in ("failed",) for s in statuses) or len(group) < MAX_ATTEMPTS:
            outcome, values = "pending", {"k": None, "factual": None, "benchmark": None}
        else:
            outcome, values = "missing", {"k": None, "factual": None, "benchmark": None}
        runs.append({"arm": arm, "lr": lr, "seed": seed, "run_type": run_type, "status": outcome,
                     "k": values["k"], "factual": values["factual"],
                     "benchmark": values["benchmark"]})  # fmt: skip
    return runs


def adamw_range(runs: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Smallest and largest cell-mean K (geometric mean over completed runs) of the AdamW cells."""
    by_lr: dict[float, list[float]] = {}
    for r in runs:
        if (r["arm"] == "adamw" and r["run_type"] == "train" and r["status"] == "completed"
                and r["k"] is not None and math.isfinite(r["k"]) and r["k"] > 0):  # fmt: skip
            by_lr.setdefault(r["lr"], []).append(math.log(r["k"]))
    if not by_lr:
        raise ValueError("No completed AdamW run with a positive K")
    means = [math.exp(sum(v) / len(v)) for v in by_lr.values()]
    return min(means), max(means)


def theta0_utility(runs: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    found = [r for r in runs if r["run_type"] == "theta0" and r["status"] == "completed"]
    if len(found) != 1:
        raise ValueError(f"Expected one completed θ₀ run, found {len(found)}")
    utility = {"factual": found[0]["factual"], "benchmark": found[0]["benchmark"]}
    for name, value in utility.items():
        if not _accuracy(value):
            raise ValueError(f"θ₀'s {name} accuracy {value!r} is not in [0, 1]")
    return utility


def pilots_of(runs: Sequence[Mapping[str, Any]], arm: str) -> list[Pilot]:
    """The seed-0 training runs of ``arm``."""
    return [Pilot(r["lr"], r["status"], r["k"], r["factual"], r["benchmark"]) for r in runs
            if r["arm"] == arm and r["run_type"] == "train" and r["seed"] == 0]  # fmt: skip
