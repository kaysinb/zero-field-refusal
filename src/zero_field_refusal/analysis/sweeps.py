"""The choice of the stage-1 sweep rates from the seed-0 pilots (preregistration, "Stage-1 sweeps").

Only K, divergence and utility are read (``meta.json``, ``config.yaml``, ``summary.json``), never
a judge label. A pilot is a *candidate* when its run completed with a positive, finite K and it is
stable: neither diverged nor collapsed, collapse being a factual accuracy more than 10 pp, or a
benchmark accuracy more than 5 pp, below θ₀'s. The target range [K_A,min, K_A,max] spans the
cell-mean K (geometric mean) of the model's AdamW cells under the main condition. Target points,
in order: log K_A,min, log K_A,max, then the evenly spaced points between them on the log scale
(one for frozen Adam, two for SGD). For each point the not yet chosen candidate with the nearest
log K is chosen, a tie going to the smaller rate; for SGD the largest stable candidate replaces
the last choice if it was not chosen. If no candidate lies within a factor 2 of K_A,min (or of
K_A,max), the grid is extended by one step in that direction (a factor 3 for SGD, 2 for frozen
Adam), at most twice per end, and the extension is piloted before the choice.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .evidence import arm_of

FACTUAL_DROP, BENCHMARK_DROP = 0.10, 0.05  # collapse beyond these drops below θ₀
TOLERANCE = 1e-9  # accuracies are fractions of counts; equality at a threshold is no collapse
NEAR = math.log(2.0)  # a candidate within a factor 2 of a target end reaches it
MAX_EXTENSIONS = 2  # per end
PLACES = {"sgd": 4, "adam_frozen": 3}
STEP = {"sgd": 3.0, "adam_frozen": 2.0}


@dataclass(frozen=True)
class Pilot:
    lr: float
    status: str | None
    k: float | None
    factual: float | None
    benchmark: float | None


@dataclass
class Choice:
    arm: str
    grid: list[float]  # the rates of the base grid
    candidates: list[dict[str, Any]]
    targets: list[float]  # target K values, in order
    chosen: list[float] = field(default_factory=list)
    extend: list[float] = field(default_factory=list)  # rates to pilot before the choice
    notes: list[str] = field(default_factory=list)


def unstable_reason(pilot: Pilot, theta0: Mapping[str, float]) -> str | None:
    """Why a pilot is not a candidate (None: it is one)."""
    if pilot.status == "diverged":
        return "diverged"
    if pilot.status != "completed":
        return f"status {pilot.status!r}"
    if pilot.k is None or not math.isfinite(pilot.k) or pilot.k <= 0:
        return f"K {pilot.k!r}"
    if pilot.factual is None or pilot.benchmark is None:
        return "utility missing"
    if theta0["factual"] - pilot.factual > FACTUAL_DROP + TOLERANCE:
        return "collapse: factual accuracy"
    if theta0["benchmark"] - pilot.benchmark > BENCHMARK_DROP + TOLERANCE:
        return "collapse: benchmark accuracy"
    return None


def target_points(k_min: float, k_max: float, places: int) -> list[float]:
    """log K_A,min, log K_A,max, then the evenly spaced interior points, as K values."""
    lo, hi = math.log(k_min), math.log(k_max)
    interior = [lo + (hi - lo) * j / (places - 1) for j in range(1, places - 1)]
    return [math.exp(x) for x in (lo, hi, *interior)]


def choose(arm: str, grid: Sequence[float], pilots: Iterable[Pilot],
           theta0: Mapping[str, float], k_min: float, k_max: float) -> Choice:  # fmt: skip
    """The rates for ``arm`` (sgd or adam_frozen) from its pilots, or the extensions to pilot."""
    pilots = sorted(pilots, key=lambda p: p.lr)
    candidates = []
    for p in pilots:
        reason = unstable_reason(p, theta0)
        candidates.append(asdict(p) | {"candidate": reason is None, "reason": reason})
    usable = [p for p, c in zip(pilots, candidates, strict=True) if c["candidate"]]
    targets = target_points(k_min, k_max, PLACES[arm])
    choice = Choice(arm, sorted(grid), candidates, targets)

    def reaches(target: float) -> bool:
        return any(abs(math.log(p.k) - math.log(target)) <= NEAR + 1e-12 for p in usable)

    piloted = [p.lr for p in pilots]
    below = [lr for lr in piloted if lr < min(grid) * (1 - 1e-9)]
    above = [lr for lr in piloted if lr > max(grid) * (1 + 1e-9)]
    if not reaches(k_min) and len(below) < MAX_EXTENSIONS:
        choice.extend.append(min(piloted) / STEP[arm])
    if not reaches(k_max) and len(above) < MAX_EXTENSIONS:
        choice.extend.append(max(piloted) * STEP[arm])
    if choice.extend:
        return choice
    for name, target in (("K_A,min", k_min), ("K_A,max", k_max)):
        if not reaches(target):
            choice.notes.append(f"no candidate within a factor 2 of {name} after the extensions")
    chosen: list[Pilot] = []
    for target in targets:
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


def read_pilots(root: Path, experiment: str) -> list[dict[str, Any]]:
    """Every run directory of ``root/experiment``: arm, lr, seed, run_type, status, K, utility."""
    rows = []
    for run_dir in sorted(p for p in (Path(root) / experiment).glob("*") if p.is_dir()):
        try:
            meta = json.loads((run_dir / "meta.json").read_text())
            cfg = OmegaConf.load(run_dir / "config.yaml")
        except (OSError, ValueError):
            continue
        summary_path = run_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        k = summary.get("K_harmful")
        rows.append({
            "run_id": run_dir.name, "arm": arm_of(cfg), "lr": float(cfg.optimizer.lr),
            "seed": int(cfg.seed), "run_type": cfg.get("run_type", "train"),
            "status": meta.get("status"),
            "k": float(k) if isinstance(k, int | float) and not isinstance(k, bool) else None,
            "factual": summary.get("factual_accuracy"),
            "benchmark": summary.get("benchmark_accuracy"),
        })  # fmt: skip
    return rows


def adamw_range(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Smallest and largest cell-mean K (geometric mean over completed runs) of the AdamW cells."""
    by_lr: dict[float, list[float]] = {}
    for r in rows:
        if (r["arm"] == "adamw" and r["run_type"] == "train" and r["status"] == "completed"
                and r["k"] is not None and math.isfinite(r["k"]) and r["k"] > 0):  # fmt: skip
            by_lr.setdefault(r["lr"], []).append(math.log(r["k"]))
    if not by_lr:
        raise ValueError("No completed AdamW run with a positive K")
    means = [math.exp(sum(v) / len(v)) for v in by_lr.values()]
    return min(means), max(means)


def theta0_utility(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    runs = [r for r in rows if r["run_type"] == "theta0" and r["status"] == "completed"]
    if len(runs) != 1:
        raise ValueError(f"Expected one completed θ₀ run, found {len(runs)}")
    return {"factual": runs[0]["factual"], "benchmark": runs[0]["benchmark"]}


def pilots_of(rows: Sequence[Mapping[str, Any]], arm: str) -> list[Pilot]:
    """The seed-0 training runs of ``arm`` (the last attempt per rate when there are several)."""
    pilots: dict[float, Pilot] = {}
    for r in rows:
        if r["arm"] == arm and r["run_type"] == "train" and r["seed"] == 0:
            pilot = Pilot(r["lr"], r["status"], r["k"], r["factual"], r["benchmark"])
            if pilots.get(r["lr"]) is None or pilots[r["lr"]].status != "completed":
                pilots[r["lr"]] = pilot
    return list(pilots.values())
