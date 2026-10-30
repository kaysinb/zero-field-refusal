"""Study 2's whole seed vectors, K and utility only: the inputs of study 3's simulation.

docs/study3-preregistration.md, "Simulation before validation": the scenarios bootstrap "whole
seed vectors from study 2 (every arm and rate of a seed together, so that K and utility keep their
dependence), not runs as independent draws", with their realized interval widths, correlations
and asymmetry. This module reads those vectors from study 2's run records, writes them as one
record (``docs/study3-simulation-inputs.json``, which the simulation's fixation pins by its
sha256), and maps a vector to study 3's rates.

- ``study2_points``: every usable 25-update point of study 2's fixed-schedule runs at B = 20,
  under the given roots (by default ``RUNS_DIR/study2_short`` and ``RUNS_DIR/study2_stage_a``,
  ``STUDY2_ROOTS``), with the runs left out and counted, and the duplicates.
- ``seed_vectors``: the whole vectors among them: seeds whose A, F and P each have points at two
  or more distinct rates.
- ``Vector.ln_k`` and ``Vector.losses``: ln K at a rate, and the MMLU and factual losses at an
  ln K.
- ``utility_slopes``: per arm, the pooled within-seed slopes of the two losses in ln K.
- ``inputs_record`` and ``read_inputs``: the record, and its closed-schema reader.

**What is read, and nothing else.** Of a study-2 run directory only ``config.yaml``,
``terminal.json`` and ``evaluations.jsonl`` are opened (``RUN_FILES``); of θ₀'s evaluation only
its ``summary.json``, through ``study3_records.read_theta0``. From the config: the run type, the
control key, the schedule's policy, the steps, the batch, the optimizer (the arm,
``study3_records.ARMS``), its rate and the seed. From the terminal record: the stop's reason and
the steps. From the evaluations: the step-25 row's ``K_benign``, ``benchmark_accuracy`` and
``factual_accuracy`` (``ENDPOINT_KEYS``); of every row only these keys and ``step`` are taken, and
nothing else of a row is copied anywhere. No harm, label, response, judge output, keyword refusal
rate or K_harmful is read: the simulation's harm comes only from the planning spreads and study
1's curvature. Nothing under a run's ``responses/``, ``observations/`` or ``judge/`` folder is
opened: a record file, or a run folder, that resolves under one of them (or cannot be resolved,
a link loop) is refused unopened (``study3_records.under_sealed``, with ``judge`` added), and the
build fails with the path named.

**Which runs** (the study's frozen settings, M9: all whole seeds under the two study-2 roots).
A run directory is a direct child of a root with a ``config.yaml`` that has ``run_type: train``,
no control (``control`` null, as the zero-field protocol sets it, or absent),
``schedule.policy: fixed``, ``train.steps: 25``, ``train.batch_size: 20`` and
``optimizer.name`` among ``study3_records.ARMS`` (adamw → A, adam_frozen → F, adamw_prewarmed →
P); other runs are not points and are not counted. A usable point has 25 updates (its stop is
its steps) and a finite positive step-25 K with finite accuracies. The others are left out and
counted: ``instability`` (an instability stop, ``study3_records.INSTABILITY``), ``incomplete``
(a config that cannot be read, no terminal record, another stop or fewer steps, no single
step-25 row, a missing or non-finite value) and ``nonpositive_k``. Two runs of one (seed, arm,
rate): the earliest run id (its time stamp) is kept; the others are counted, with the largest
relative K difference |K − K_kept|/K_kept. The losses are in pp against θ₀ (``read_theta0``;
study 2's θ₀ evaluation ``20261014T192927Z-theta0-s2000-abb703a1-a1``): 100·(θ₀'s accuracy − the
run's).

**The map to study 3's rates** (M9). ln K of vector v and arm a is piecewise linear in ln η
through v's points, extended beyond them with the registered slope b_a
(``study3_search.SLOPES``: A 1.25, F 1.30, P 1.24). The MMLU and factual losses are piecewise
linear in ln K through the same points, extended beyond them with the arm's pooled within-seed
slope (least squares with one intercept per seed, ``utility_slopes``).

**The record** (``INPUTS_PATH``): ``kind`` (``study3-simulation-inputs``), ``description``,
``sources`` (the roots), ``theta0`` (``path`` and the two accuracies), ``vectors`` (per seed,
``arms``: per arm the points' ``rate``, ``K``, ``mmlu_loss_pp``, ``factual_loss_pp`` and
``run_id``, sorted by rate), ``excluded`` (reason → count, ``incomplete_seed`` included) and
``duplicates`` (``count`` and ``largest_relative_k_difference``, null without one). Its schema
is closed: ``read_inputs`` refuses any other field. The slopes are recomputed from the vectors
when the record is read.

**Rulings** (the study's frozen settings, each with its one-line source):
- M9, the seed vectors and the rate map: all whole seeds under the two study-2 roots; piecewise
  linear in ln η, extended with b; utility piecewise linear in ln K, extended with the pooled
  slope; duplicates keep the earliest run (study 2's validation, seeds 2100–2111 at three rates
  per arm, is the core); decided.
- Two points with the same K in one arm of a vector give one knot of the utility curve, the mean
  of their losses (a piecewise-linear function of ln K has one value per ln K).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from .study3_records import ARMS, INSTABILITY, read_theta0, sealed_why, under_sealed
from .study3_search import ARM_ORDER, SLOPES

INPUTS_PATH = "docs/study3-simulation-inputs.json"
STUDY2_ROOTS = ("study2_short", "study2_stage_a")
RUN_FILES = ("config.yaml", "terminal.json", "evaluations.jsonl")
ENDPOINT_KEYS = ("K_benign", "benchmark_accuracy", "factual_accuracy")
KIND = "study3-simulation-inputs"
THETA0_RUN = "20261014T192927Z-theta0-s2000-abb703a1-a1"  # study 2's θ₀ evaluation
UPDATES, BATCH = 25, 20
JUDGE = "judge"  # a run's judge outputs: never opened, as responses and observations
EXCLUSIONS = ("instability", "incomplete", "nonpositive_k")
INCOMPLETE_SEED = "incomplete_seed"
MIN_RATES = 2  # distinct rates per arm of a whole vector
DESCRIPTION = (
    "Study 2's whole seed vectors for study 3's simulation (docs/study3-preregistration.md, "
    "'Simulation before validation'): per seed and arm, the 25-update points at B = 20 with their "
    "rate, step-25 K_benign and MMLU and factual losses in pp against theta0; no harm, label, "
    "response or K on the harmful panel is read."
)
_FIELDS = ("kind", "description", "sources", "theta0", "vectors", "excluded", "duplicates")
_POINT_FIELDS = ("rate", "K", "mmlu_loss_pp", "factual_loss_pp", "run_id")


@dataclass(frozen=True)
class Point:
    """One usable 25-update run of a seed and arm: its rate, its step-25 K and its losses (pp
    against θ₀)."""

    rate: float
    K: float
    mmlu_loss_pp: float
    factual_loss_pp: float
    run_id: str


@dataclass(frozen=True)
class Vector:
    """One seed's points, per arm, sorted by rate (at least two distinct rates per arm)."""

    seed: int
    arms: Mapping[str, tuple[Point, ...]]
    _rate_map: dict[str, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _utility_map: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError(f"a vector's seed must be an integer, not {self.seed!r}")
        for arm, points in self.arms.items():
            if arm not in ARM_ORDER:
                raise ValueError(f"{arm!r} is not an arm of {ARM_ORDER}")
            rates = [p.rate for p in points]
            increasing = all(a < b for a, b in zip(rates, rates[1:], strict=False))
            if len(points) < MIN_RATES or not increasing:
                raise ValueError(
                    f"seed {self.seed}, arm {arm}: at least {MIN_RATES} points at "
                    f"strictly increasing rates are needed, not {rates}"
                )
            for p in points:
                if not (
                    _positive(p.rate)
                    and _positive(p.K)
                    and _finite(p.mmlu_loss_pp)
                    and _finite(p.factual_loss_pp)
                ):
                    raise ValueError(
                        f"seed {self.seed}, arm {arm}: a point needs a positive "
                        f"rate and K and finite losses ({p})"
                    )
            self._rate_map[arm] = (np.log([p.rate for p in points]), np.log([p.K for p in points]))
            knots: dict[float, list[tuple[float, float]]] = defaultdict(list)
            for p in points:
                knots[math.log(p.K)].append((p.mmlu_loss_pp, p.factual_loss_pp))
            xs = sorted(knots)
            self._utility_map[arm] = (
                np.array(xs),
                np.array([np.mean([m for m, _ in knots[x]]) for x in xs]),
                np.array([np.mean([f for _, f in knots[x]]) for x in xs]),
            )

    def ln_k(self, arm: str, rate: float) -> float:
        """ln K of ``arm`` at ``rate``: piecewise linear in ln η through the points, extended
        beyond them with the registered slope b of the arm (``study3_search.SLOPES``)."""
        x, y = self._rate_map[arm]
        at = math.log(rate)
        if at <= x[0]:
            return float(y[0] + SLOPES[arm] * (at - x[0]))
        if at >= x[-1]:
            return float(y[-1] + SLOPES[arm] * (at - x[-1]))
        return float(np.interp(at, x, y))

    def losses(
        self, arm: str, ln_k: float, slopes: Mapping[str, tuple[float, float]]
    ) -> tuple[float, float]:
        """The MMLU and factual losses (pp) of ``arm`` at ``ln_k``: piecewise linear in ln K
        through the points, extended beyond them with the arm's pooled slopes
        (``slopes[arm]``, as ``utility_slopes`` gives them)."""
        mmlu, factual = self.losses_at(arm, np.asarray(float(ln_k)), slopes)
        return float(mmlu), float(factual)

    def losses_at(
        self, arm: str, ln_k: np.ndarray, slopes: Mapping[str, tuple[float, float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """``losses`` at every ln K of an array, elementwise."""
        x, mmlu, factual = self._utility_map[arm]
        s_mmlu, s_factual = slopes[arm]
        ln_k = np.asarray(ln_k, dtype=float)
        below, above = ln_k < x[0], ln_k > x[-1]
        out = []
        for knots, slope in ((mmlu, s_mmlu), (factual, s_factual)):
            inside = np.interp(ln_k, x, knots)
            out.append(
                np.where(
                    below,
                    knots[0] + slope * (ln_k - x[0]),
                    np.where(above, knots[-1] + slope * (ln_k - x[-1]), inside),
                )
            )
        return out[0], out[1]

    def mmlu_crossings(
        self, arm: str, level_pp: float, slopes: Mapping[str, tuple[float, float]]
    ) -> list[float]:
        """The ln K values where the arm's MMLU loss (``losses``) equals ``level_pp``, in
        ascending order: one per segment of the piecewise-linear curve, the two extensions
        included, that crosses it (a flat segment at the level gives its ends)."""
        x, mmlu, _ = self._utility_map[arm]
        s = slopes[arm][0]
        out: list[float] = []
        if s != 0 and (level_pp - mmlu[0]) / s < 0:  # the lower extension
            out.append(float(x[0] + (level_pp - mmlu[0]) / s))
        for i in range(len(x) - 1):
            lo, hi = mmlu[i], mmlu[i + 1]
            if lo == hi:
                if lo == level_pp:
                    out += [float(x[i]), float(x[i + 1])]
                continue
            t = (level_pp - lo) / (hi - lo)
            if 0 <= t <= 1:
                out.append(float(x[i] + t * (x[i + 1] - x[i])))
        if len(x) == 1 and mmlu[0] == level_pp:
            out.append(float(x[0]))
        if s != 0 and (level_pp - mmlu[-1]) / s > 0:  # the upper extension
            out.append(float(x[-1] + (level_pp - mmlu[-1]) / s))
        return sorted(set(out))


# ------------------------------------------------------------------ study 2's points


def study2_points(
    roots: Sequence[Path], theta0: Path
) -> tuple[list[tuple[int, str, Point]], dict[str, Any]]:
    """The usable points under ``roots`` (module docstring), as (seed, arm, point), sorted by
    seed, arm and rate, duplicates removed (the earliest run kept); and what was left out:
    ``instability``, ``incomplete`` and ``nonpositive_k`` (counts) and ``duplicates``
    (``count`` and ``largest_relative_k_difference``, None without one). ``ValueError`` for a
    root that is not a folder, θ₀'s evaluation refused, or a run folder or record file that
    resolves under a sealed folder (never opened)."""
    baseline = read_theta0(theta0)
    excluded: dict[str, Any] = dict.fromkeys(EXCLUSIONS, 0)
    found: list[tuple[int, str, Point]] = []
    for root in map(Path, roots):
        if not root.is_dir():
            raise ValueError(f"{root}: a root of study 2's runs that is not a folder")
        for run in sorted(root.iterdir()):
            if not run.is_dir() and not run.is_symlink():
                continue
            _refuse_sealed(run, root)
            if not (run / "config.yaml").is_file():
                continue  # no run directory
            read = _run_point(run, baseline)
            if read is None:
                continue  # not a run of the three arms' fixed 25-update schedule at B = 20
            if isinstance(read, str):
                excluded[read] += 1
            else:
                found.append(read)
    points, duplicates = _without_duplicates(found)
    excluded["duplicates"] = duplicates
    return points, excluded


def _sealed(path: Path, base: Path) -> bool:
    """The sealed-path rule (``study3_records.under_sealed``), with a run's ``judge`` folder
    sealed as well: nothing is opened."""
    if under_sealed(path, base):
        return True
    try:
        resolved, root = Path(path).resolve(), Path(base).resolve()
    except (OSError, RuntimeError):  # a link loop
        return True
    below = resolved.relative_to(root) if resolved.is_relative_to(root) else resolved
    return JUDGE in below.parts


def _refuse_sealed(run: Path, root: Path) -> None:
    """``ValueError`` when the run's folder or one of its record files resolves under a sealed
    folder (or cannot be resolved), before anything of the run is opened."""
    if _sealed(run, root) or _sealed(run, run.parent):
        why = sealed_why(run, root, "resolves under a sealed folder")
        raise ValueError(f"{run}: the run folder {why}: never opened")
    for name in RUN_FILES:
        if _sealed(run / name, run):
            why = sealed_why(run / name, run, "resolves under a sealed folder")
            raise ValueError(f"{run / name} {why}: never opened")


def _run_point(run: Path, baseline: Mapping[str, float]) -> tuple[int, str, Point] | str | None:
    """A run's point, the reason it is left out, or None for a run that is not one of the
    three arms' fixed 25-update runs at B = 20. A config that cannot be read as a mapping is an
    incomplete record: counted, never silently dropped."""
    try:
        cfg = OmegaConf.create((run / "config.yaml").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - any parse failure of the YAML
        return "incomplete"
    if not OmegaConf.is_dict(cfg):
        return "incomplete"

    def get(key: str) -> Any:
        try:
            return OmegaConf.select(cfg, key, default=None)
        except Exception:  # noqa: BLE001 - an interpolation that cannot be resolved
            return None

    if (
        get("run_type") != "train"
        or get("control") is not None
        or get("schedule.policy") != "fixed"
        or get("train.steps") != UPDATES
        or get("train.batch_size") != BATCH
        or get("optimizer.name") not in ARMS
    ):
        return None
    arm, rate, seed = ARMS[get("optimizer.name")], get("optimizer.lr"), get("seed")
    if not (_positive(rate) and type(seed) is int):
        return "incomplete"
    try:
        terminal = json.loads((run / "terminal.json").read_text(encoding="utf-8"))
        stop, steps = terminal["stop"]["reason"], terminal["steps"]
    except (OSError, ValueError, KeyError, TypeError):
        return "incomplete"
    if stop in INSTABILITY:
        return "instability"
    if stop != "steps" or steps != UPDATES:
        return "incomplete"
    try:
        rows = [
            _endpoint_keys(line)
            for line in (run / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError):
        return "incomplete"
    at = [row for row in rows if row.get("step") == UPDATES]
    if len(at) != 1:
        return "incomplete"
    k, mmlu, factual = (at[0].get(key) for key in ENDPOINT_KEYS)
    if not (_finite(k) and _finite(mmlu) and _finite(factual)):
        return "incomplete"
    if k <= 0:
        return "nonpositive_k"
    point = Point(
        rate=float(rate),
        K=float(k),
        mmlu_loss_pp=100.0 * (baseline["benchmark_accuracy"] - mmlu),
        factual_loss_pp=100.0 * (baseline["factual_accuracy"] - factual),
        run_id=run.name,
    )
    return seed, arm, point


def _endpoint_keys(line: str) -> dict[str, Any]:
    """Of one evaluation row, ``step`` and ``ENDPOINT_KEYS`` only."""
    row = json.loads(line)
    if not isinstance(row, dict):
        raise ValueError("an evaluation row that is not an object")
    return {key: row[key] for key in ("step", *ENDPOINT_KEYS) if key in row}


def _without_duplicates(
    found: Sequence[tuple[int, str, Point]],
) -> tuple[list[tuple[int, str, Point]], dict[str, Any]]:
    """One point per (seed, arm, rate): the earliest run id; the others counted with their
    largest relative K difference."""
    groups: dict[tuple[int, str, float], list[Point]] = defaultdict(list)
    for seed, arm, point in found:
        groups[(seed, arm, _rate_key(point.rate))].append(point)
    kept: list[tuple[int, str, Point]] = []
    count, largest = 0, None
    for (seed, arm, _), points in groups.items():
        first, *others = sorted(points, key=lambda p: p.run_id)
        kept.append((seed, arm, first))
        for other in others:
            count += 1
            diff = abs(other.K - first.K) / first.K
            largest = diff if largest is None else max(largest, diff)
    kept.sort(key=lambda t: (t[0], ARM_ORDER.index(t[1]), t[2].rate))
    return kept, {"count": count, "largest_relative_k_difference": largest}


def _rate_key(rate: float) -> float:
    """A rate as written in a config, for matching (6 significant digits)."""
    return float(f"{rate:.6g}")


# ------------------------------------------------------------------ the vectors


def seed_vectors(points: Sequence[tuple[int, str, Point]]) -> tuple[list[Vector], dict[str, int]]:
    """The whole vectors (seeds whose A, F and P each have points at ``MIN_RATES`` or more
    distinct rates), sorted by seed, and ``incomplete_seed``, the seeds that are not whole.
    ``ValueError`` for two points of one seed, arm and rate."""
    by_seed: dict[int, dict[str, list[Point]]] = defaultdict(lambda: defaultdict(list))
    for seed, arm, point in points:
        by_seed[seed][arm].append(point)
    vectors, incomplete = [], 0
    for seed in sorted(by_seed):
        arms = by_seed[seed]
        for arm, mine in arms.items():
            if len({_rate_key(p.rate) for p in mine}) != len(mine):
                raise ValueError(f"seed {seed}, arm {arm}: two points at one rate")
        if all(len(arms.get(arm, ())) >= MIN_RATES for arm in ARM_ORDER):
            vectors.append(
                Vector(
                    seed=seed,
                    arms={arm: tuple(sorted(arms[arm], key=lambda p: p.rate)) for arm in ARM_ORDER},
                )
            )
        else:
            incomplete += 1
    return vectors, {INCOMPLETE_SEED: incomplete}


def utility_slopes(vectors: Sequence[Vector]) -> dict[str, tuple[float, float]]:
    """Per arm, the pooled within-seed slopes of the (MMLU, factual) losses in ln K: least
    squares with one intercept per seed. ``ValueError`` when an arm's ln K does not vary within
    any seed."""
    out = {}
    for arm in ARM_ORDER:
        sxx, sxm, sxf = 0.0, 0.0, 0.0
        for v in vectors:
            points = v.arms.get(arm, ())
            if not points:
                continue
            x = np.log([p.K for p in points])
            m = np.array([p.mmlu_loss_pp for p in points])
            f = np.array([p.factual_loss_pp for p in points])
            dx = x - x.mean()
            sxx += float(dx @ dx)
            sxm += float(dx @ (m - m.mean()))
            sxf += float(dx @ (f - f.mean()))
        if not sxx > 0:
            raise ValueError(f"arm {arm}: ln K does not vary within any seed; no utility slope")
        out[arm] = (sxm / sxx, sxf / sxx)
    return out


# ------------------------------------------------------------------ the record


def inputs_record(
    vectors: Sequence[Vector],
    excluded: Mapping[str, Any],
    *,
    theta0: Mapping[str, Any],
    sources: Sequence[str],
) -> dict[str, Any]:
    """The inputs record (module docstring). ``excluded``: ``study2_points``' and
    ``seed_vectors``' counts together; its ``duplicates`` entry becomes the record's own
    field."""
    counts = {k: v for k, v in excluded.items() if k != "duplicates"}
    duplicates = excluded.get("duplicates") or {"count": 0, "largest_relative_k_difference": None}
    return {
        "kind": KIND,
        "description": DESCRIPTION,
        "sources": [str(s) for s in sources],
        "theta0": dict(theta0),
        "vectors": [
            {
                "seed": v.seed,
                "arms": {
                    arm: [{name: getattr(p, name) for name in _POINT_FIELDS} for p in v.arms[arm]]
                    for arm in ARM_ORDER
                },
            }
            for v in vectors
        ],
        "excluded": dict(counts),
        "duplicates": dict(duplicates),
    }


def read_inputs(path: Path) -> tuple[list[Vector], dict[str, tuple[float, float]], str]:
    """The record's vectors, the utility slopes recomputed from them and the file's sha256.
    ``ValueError`` for a record that is unreadable or departs from the closed schema."""
    path = Path(path)
    try:
        data = path.read_bytes()
        record = json.loads(data)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: unreadable ({type(exc).__name__})") from exc
    problems = _record_problems(record)
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    vectors = []
    for entry in record["vectors"]:
        try:
            vectors.append(
                Vector(
                    seed=entry["seed"],
                    arms={arm: tuple(Point(**p) for p in entry["arms"][arm]) for arm in ARM_ORDER},
                )
            )
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    try:
        slopes = utility_slopes(vectors)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return vectors, slopes, hashlib.sha256(data).hexdigest()


def _record_problems(record: Any) -> list[str]:
    """What departs from the record's closed schema."""
    if not isinstance(record, dict):
        return ["not a JSON object"]
    problems = []
    if set(record) != set(_FIELDS):
        problems.append(f"fields {sorted(record)}, not {sorted(_FIELDS)}")
        return problems
    if record["kind"] != KIND:
        problems.append(f"kind {record['kind']!r}, not {KIND!r}")
    if not isinstance(record["description"], str):
        problems.append("a description that is not text")
    sources = record["sources"]
    if not (isinstance(sources, list) and sources and all(isinstance(s, str) for s in sources)):
        problems.append("sources must be a non-empty list of paths")
    theta0 = record["theta0"]
    if not (
        isinstance(theta0, dict)
        and set(theta0) == {"path", "benchmark_accuracy", "factual_accuracy"}
        and isinstance(theta0["path"], str)
        and all(
            _finite(theta0[k]) and 0 <= theta0[k] <= 1
            for k in ("benchmark_accuracy", "factual_accuracy")
        )
    ):
        problems.append("theta0 must be its path and its two accuracies, fractions in [0, 1]")
    vectors = record["vectors"]
    if not (isinstance(vectors, list) and vectors):
        problems.append("no vectors")
    else:
        seeds = [v.get("seed") if isinstance(v, dict) else None for v in vectors]
        if len(set(map(repr, seeds))) != len(seeds):
            problems.append("a seed listed twice")
        for v in vectors:
            problems += _vector_problems(v)
    excluded = record["excluded"]
    if not (
        isinstance(excluded, dict)
        and set(excluded) == {*EXCLUSIONS, INCOMPLETE_SEED}
        and all(type(n) is int and n >= 0 for n in excluded.values())
    ):
        problems.append(f"excluded must count {[*EXCLUSIONS, INCOMPLETE_SEED]}, each a count")
    duplicates = record["duplicates"]
    if not (
        isinstance(duplicates, dict)
        and set(duplicates) == {"count", "largest_relative_k_difference"}
        and type(duplicates["count"]) is int
        and duplicates["count"] >= 0
        and (
            (duplicates["count"] == 0 and duplicates["largest_relative_k_difference"] is None)
            or (
                duplicates["count"] > 0
                and _finite(duplicates["largest_relative_k_difference"])
                and duplicates["largest_relative_k_difference"] >= 0
            )
        )
    ):
        problems.append("duplicates must be a count and the largest relative K difference")
    return problems


def _vector_problems(entry: Any) -> list[str]:
    if not (isinstance(entry, dict) and set(entry) == {"seed", "arms"}):
        return ["a vector must hold its seed and its arms, nothing else"]
    seed = entry["seed"]
    if type(seed) is not int:
        return [f"a vector's seed {seed!r} is not an integer"]
    arms = entry["arms"]
    if not (isinstance(arms, dict) and set(arms) == set(ARM_ORDER)):
        return [f"seed {seed}: arms must be exactly {list(ARM_ORDER)}"]
    problems = []
    for arm, points in arms.items():
        if not isinstance(points, list):
            problems.append(f"seed {seed}, arm {arm}: the points are not a list")
            continue
        for p in points:
            if not (
                isinstance(p, dict)
                and set(p) == set(_POINT_FIELDS)
                and all(_finite(p[k]) for k in _POINT_FIELDS if k != "run_id")
                and isinstance(p["run_id"], str)
            ):
                problems.append(
                    f"seed {seed}, arm {arm}: a point must be exactly "
                    f"{list(_POINT_FIELDS)}, numbers and a run id"
                )
    return problems


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value: Any) -> bool:
    return _finite(value) and value > 0
