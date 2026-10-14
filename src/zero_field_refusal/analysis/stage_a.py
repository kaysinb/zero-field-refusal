"""Stage A of study 2: run records, their summaries and the selection rule, as one function.

docs/study2-preregistration.md ("Stage A — feasibility", "Selection rule", "Numerical
acceptance"). Everything here reads run directories written by ``schedule.run_scheduled`` and
never a harm label: the selection uses numerics, K, reachability, utility and measured time only.

- ``load_trajectory``: one run's records, checked for integrity (``TrajectoryError``).
- ``cell_summary``: one arm, batch and rate over its seeds: stops, reachability and median steps
  per level, utility losses at the levels, time.
- ``screen_candidates`` and ``numerical_acceptance``: soft-target controls, contamination and the
  accepted SGD rates.
- ``instability_summary``: the gradient rule's margin on stable runs, its firings and possible
  false alarms, and the revision of its factor 10 by the rule below, before validation.
- ``agreement``: the monitor against the scoring panel.
- ``select``: the selection rule: batch, U, K*, rates and T_max, with the reasons for every
  batch and level, or the Stage-A outcome that stops the design. It first checks that the
  prescribed search is whole: the screen of every SGD rate at both batches; discovery runs of
  exactly the two screen candidates per batch, the AdamW-long grid and the 25-step grids with
  their prescribed extensions, one run per arm, batch, rate and seed 2000-2002, nothing else;
  one full-length control per candidate on seed 2000, on the stream of the regular run of that
  seed. Anything missing, duplicated, unexpected or of another provenance makes the outcome
  ``incomplete``, never a scientific failure.
- A run is accepted on whole records agreeing with its published terminal record, whatever its
  status (the launcher's rule too): a failure of logging after the outcome does not undo it.

**Rulings** where the preregistration leaves a detail open, fixed here before any data:
- *Utility loss* is θ₀'s accuracy minus the run's at an evaluation, in percentage points;
  "in bounds" means the mean over the discovery seeds is at most 5 pp on MMLU and 10 pp on the
  factual questions at every evaluation used: for a long arm the evaluations at 5, K* and U, for
  a 25-step arm its evaluation at step 25.
- *U qualifies at a batch* when each long arm has a rate that meets its own selection rule at
  that U (rules 2 and 3), which includes reachability by every discovery seed and utility.
- *Time per seed block* is the median over seeds of the summed run times of the selected
  SGD-long, AdamW-long and AdamW-25 runs (frozen Adam-25 runs on 20 blocks only), excluding stream
  extension, plus generating the block's stream at the measured time per chunk.
- *AdamW-25 rates* are the injective assignment of qualifying rates to the targets (5, K*, U)
  with the least total |median K_benign − target|, ties to the smaller rates.
- *The gradient rule's factor* is judged by its margin, the largest ratio of a gradient norm to
  its reference on runs that ended stably, and by possible false alarms: gradient-rule stops in
  a cell (arm, batch, rate) where another seed ran stably. Such a stop is only possibly false,
  since breakdowns near the edge vary by seed and a stopped run cannot show what it would have
  done. The factor is raised only if the margin exceeds 7 or a possible false alarm occurred,
  to max(10, 1.5 × the margin), fixed before validation; it is never lowered (coordinator,
  2026-10-14).
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..config import load_config
from ..panels import monitor_agreement

LOWER = 5.0
UPPER = (8.0, 7.0)  # U, larger first
T_MAX_CAP = 4000
T_MAX_FACTOR = 1.5
MMLU_BOUND_PP = 5.0
FACTUAL_BOUND_PP = 10.0
CONTAMINATION_FRACTION = 0.01
DISCOVERY_K_STAR = math.sqrt(35)  # K* of the lowest candidate zone, until U is chosen
# The prescribed search (docs/study2-preregistration.md, "Numerical acceptance", "Discovery").
BATCHES = (5, 20)
DISCOVERY_SEEDS = (2000, 2001, 2002)
CONTROL_SEED = 2000
SCREEN_STEPS = 100
SGD_GRID = (0.0005, 0.001, 0.002, 0.003, 0.004, 0.006, 0.01, 0.015, 0.02, 0.028)
CANDIDATES_PER_BATCH = 2
ADAMW_LONG_GRID = (1e-6, 3e-6, 5e-6, 1e-5)
SHORT_GRID = (0.5e-5, 1e-5, 1.5e-5, 2e-5, 3e-5, 4e-5)
SHORT_ARMS = ("adamw_25", "frozen_adam_25")
ROLES = ("discovery", "screen", "full_control")  # stage_a.role in a run's config
GRAD_FACTOR = 10.0  # the gradient rule's factor in discovery
MARGIN_TRIGGER = 7.0  # a margin above this on stable runs raises the factor
REVISION_FACTOR = 1.5  # the raised factor: max(current, 1.5 × the margin); never lowered
INSTABILITY = (
    "nonfinite_loss",
    "nonfinite_gradient",
    "nonfinite_weight",
    "nonfinite_k",
    "k_above_max",
    "gradient_spike",
)


class TrajectoryError(ValueError):
    """A run's records are missing or inconsistent."""


@dataclass(frozen=True)
class Trajectory:
    run_dir: Path
    run_id: str
    arm: str  # sgd_long, adamw_long, adamw_25, frozen_adam_25, soft_control
    batch_size: int
    lr: float
    seed: int
    policy: str
    levels: tuple[float, ...]
    terminal: dict[str, Any]
    evaluations: list[dict[str, Any]]
    monitor: list[tuple[int, float]]
    grad_norms: list[float]
    seconds: float  # the run's time up to its published outcome
    chunk_size: int
    planned_steps: int  # train.steps: T_max, or the fixed length
    model: str  # name@revision
    stream: dict[str, Any]  # the training stream prefix it read: path, count, chunk hashes
    harmful_prompts: dict[str, Any]  # what every evaluation answered: set, count, ids hash
    status: str
    role: str | None  # stage_a.role: discovery, screen or full_control
    protocol_version: int
    spec: dict[str, Any]  # the schedule it ran under (``schedule_spec``)
    evaluate_at: tuple[int, ...] = field(default=())

    @property
    def steps(self) -> int:
        return self.terminal["steps"]

    @property
    def stop(self) -> str:
        return self.terminal["stop"]["reason"]

    @property
    def stable(self) -> bool:
        return self.stop not in INSTABILITY

    @property
    def training_seconds(self) -> float:
        return self.seconds - self.terminal.get("stream_extension", {}).get("seconds", 0.0)

    def evaluation_at(self, level: float) -> dict[str, Any] | None:
        """The evaluation that reached ``level``, if any."""
        for row in self.evaluations:
            if any(math.isclose(level, reached) for reached in row["levels"]):
                return row
        return None


def arm_of(cfg: Any) -> str:
    if cfg.get("control") == "soft_targets":
        return "soft_control"
    name, policy = cfg.optimizer.name, cfg.schedule.policy
    arms = {
        ("sgd", "k_stopped"): "sgd_long",
        ("adamw", "k_stopped"): "adamw_long",
        ("adamw", "fixed"): "adamw_25",
        ("adam_frozen", "fixed"): "frozen_adam_25",
    }
    if (name, policy) not in arms:
        raise TrajectoryError(f"No study-2 arm for optimizer {name!r} under {policy!r}")
    return arms[(name, policy)]


def load_trajectory(run_dir: Path) -> Trajectory:
    """A scheduled run's records, after checking that they agree with each other.

    The terminal record is the run's outcome: a run is accepted on whole records whatever its
    status says, since a failure of logging or finalization after the outcome was published
    must not turn an observed stop into a rerun (the launcher applies the same rule).
    """
    run_dir = Path(run_dir)
    problems: list[str] = []
    try:
        cfg = OmegaConf.load(run_dir / "config.yaml")
        meta = _json(run_dir / "meta.json")
        terminal = _json(run_dir / "terminal.json")
        summary = _json(run_dir / "summary.json") if (run_dir / "summary.json").is_file() else {}
        stop, steps = terminal["stop"]["reason"], terminal["steps"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TrajectoryError(f"{run_dir}: unreadable records ({exc!r})") from exc
    if OmegaConf.select(cfg, "schedule", default=None) is None:
        raise TrajectoryError(f"{run_dir}: not a scheduled run")
    if "terminal" in summary and summary["terminal"] != terminal:
        problems.append("summary.json's terminal record differs from terminal.json")
    evaluations = _jsonl(run_dir / "evaluations.jsonl")
    monitor = [(row["step"], row["K_monitor"]) for row in _jsonl(run_dir / "monitor.jsonl")]
    training = [row for row in _jsonl(run_dir / "metrics.jsonl") if "loss" in row]
    every = int(cfg.eval.monitor.every)
    policy, planned = cfg.schedule.policy, tuple(float(x) for x in cfg.schedule.levels)

    read_to = steps - 1 if stop == "nonfinite_weight" else steps  # no reading after a bad update
    if [s for s, _ in monitor] != list(range(every, read_to + 1, every)):
        problems.append("monitor readings are not every reading up to the last step")
    if [row["step"] for row in training] != list(range(steps)):
        problems.append("training metrics are not one row per step taken")
    eval_steps = [row["step"] for row in evaluations]
    if eval_steps != sorted(set(eval_steps)) or any(s > steps for s in eval_steps):
        problems.append("evaluation steps are not increasing within the run")
    if any(not math.isfinite(row.get("K_benign", math.nan)) for row in evaluations):
        problems.append("an evaluation lacks a finite K_benign")
    inputs = meta.get("inputs", {})
    prompts = inputs.get("harmful_prompts", {})
    problems += _response_problems(run_dir, cfg, evaluations, prompts)
    if not isinstance(terminal.get("seconds"), int | float):
        problems.append("the terminal record lacks the run's time")
    evaluate_at = tuple(int(t) for t in cfg.schedule.get("evaluate_at") or ())
    reason = stop
    if policy == "k_stopped":
        monitor_steps = {s for s, _ in monitor}
        if not set(eval_steps) <= monitor_steps:
            problems.append("an evaluation is not at a monitor reading")
        levels = terminal.get("levels", {})
        reached = [lvl for row in evaluations for lvl in row["levels"]]
        if [r["level"] for r in levels.get("reached", [])] != reached:
            problems.append("the terminal record's levels differ from the evaluations'")
        for record in levels.get("reached", []):
            row = next((r for r in evaluations if record["level"] in r["levels"]), None)
            if row is None or (row["step"], row["K_benign"]) != (
                record["step"],
                record["K_benign"],
            ):
                problems.append(f"level {record['level']}: step or K differs from its evaluation")
        missed = [lvl for lvl in planned if not any(math.isclose(lvl, r) for r in reached)]
        if levels.get("missed") != missed:
            problems.append("the terminal record's missed levels are wrong")
        failed = levels.get("failed_crossing")
        if failed is not None and (
            reason not in INSTABILITY or not all(lvl in missed for lvl in failed["levels"])
        ):
            problems.append("a failed crossing that is not a missed level of an unstable run")
        if reason == "target" and (missed or not eval_steps or eval_steps[-1] != steps):
            problems.append("a target stop without the top level evaluated at the last step")
        if reason == "plateau" and steps < cfg.schedule.plateau.from_step:
            problems.append("a plateau stop before its first step")
        if reason == "t_max" and steps != cfg.train.steps:
            problems.append("a T_max stop before T_max")
    else:
        if reason == "steps" and steps != cfg.train.steps:
            problems.append("a fixed run stopped by its steps before them")
        expected = [t for t in evaluate_at if t <= steps]
        unstable_at_last = reason in INSTABILITY and eval_steps == [
            t for t in expected if t < steps
        ]
        if eval_steps != expected and not unstable_at_last:
            problems.append("a fixed run's evaluations differ from evaluate_at")
    stream = inputs.get("streams", {}).get("training", {})
    used = stream.get("count")
    if used != steps * cfg.train.batch_size:
        problems.append(
            f"the recorded stream prefix holds {used} responses, the run used "
            f"{steps * cfg.train.batch_size}"
        )
    if problems:
        raise TrajectoryError(f"{run_dir}: " + "; ".join(problems))
    return Trajectory(
        run_dir=run_dir,
        run_id=run_dir.name,
        arm=arm_of(cfg),
        batch_size=int(cfg.train.batch_size),
        lr=float(cfg.optimizer.lr),
        seed=int(cfg.seed),
        policy=policy,
        levels=planned,
        terminal=terminal,
        evaluations=evaluations,
        monitor=monitor,
        grad_norms=[row["grad_norm"] for row in training],
        seconds=float(terminal["seconds"]),
        chunk_size=int(cfg.teacher.chunk_size),
        planned_steps=int(cfg.train.steps),
        model=f"{OmegaConf.select(cfg, 'model.name')}@{OmegaConf.select(cfg, 'model.revision')}",
        stream=stream,
        harmful_prompts=prompts,
        status=str(meta.get("status")),
        role=OmegaConf.select(cfg, "stage_a.role", default=None),
        protocol_version=int(cfg.get("protocol_version", 0)),
        spec=schedule_spec(cfg),
        evaluate_at=evaluate_at,
    )


def schedule_spec(cfg: Any) -> dict[str, Any]:
    """What a run's schedule prescribes: policy, levels, limits, stops, monitor, stream chunks."""
    schedule = cfg.schedule

    def plain(node: Any) -> Any:
        return OmegaConf.to_container(node, resolve=True) if node is not None else None

    return {
        "policy": schedule.policy,
        "levels": [float(x) for x in schedule.levels],
        "evaluate_at": plain(schedule.get("evaluate_at")),
        "train_steps": int(cfg.train.steps),
        "crossing_readings": schedule.get("crossing_readings"),
        "plateau": plain(schedule.get("plateau")),
        "instability": plain(schedule.get("instability")),
        "extend_stream": schedule.get("extend_stream"),
        "monitor_every": OmegaConf.select(cfg, "eval.monitor.every"),
        "chunk_size": OmegaConf.select(cfg, "teacher.chunk_size"),
        "harmful_set": OmegaConf.select(cfg, "eval.harmful.prompt_set"),
    }


@functools.cache
def prescribed() -> dict[str, Any]:
    """The model, protocol version and schedules the Stage-A campaigns prescribe, read from the
    study-2 configs themselves (scripts/stage_a_campaigns.py builds every run on them)."""
    long = load_config("experiments/study2_stage_a")
    short = load_config("experiments/study2_stage_a", ["schedule=short"])
    control = load_config("experiments/study2_controls")
    return {
        "model": f"{long.model.name}@{long.model.revision}",
        "protocol_version": int(long.protocol_version),
        "long": schedule_spec(long),
        "short": schedule_spec(short),
        "control": schedule_spec(control),
    }


def conformance(runs: Sequence[Trajectory], role: str) -> list[str]:
    """How runs depart from what the campaigns prescribe for their role: the role itself, the
    model and protocol version, the discovery seeds, and the schedule of their arm (a control's
    length and evaluation steps are checked against its regular runs, ``numerical_acceptance``)."""
    spec, problems = prescribed(), []
    free = ("train_steps", "evaluate_at")
    for run in runs:
        name = f"{role}: {run.run_id}"
        if run.role != role:
            problems.append(f"{name} has the role {run.role!r}")
        if run.model != spec["model"] or run.protocol_version != spec["protocol_version"]:
            problems.append(f"{name} ran {run.model} under protocol {run.protocol_version}")
        if run.seed not in DISCOVERY_SEEDS:
            problems.append(f"{name} has the seed {run.seed}, not a discovery seed")
        if run.arm == "soft_control":
            want = {k: v for k, v in spec["control"].items() if k not in free}
            have = {k: v for k, v in run.spec.items() if k not in free}
            if role == "screen":
                want |= {"train_steps": SCREEN_STEPS, "evaluate_at": [SCREEN_STEPS]}
                have |= {k: run.spec[k] for k in free}
        else:
            want = spec["long" if run.arm in ("sgd_long", "adamw_long") else "short"]
            have = run.spec
        differ = sorted(k for k in want if have.get(k) != want[k])
        if differ:
            problems.append(f"{name} departs from the prescribed schedule in {differ}")
    return problems


def stream_pairing(runs: Sequence[Trajectory]) -> list[str]:
    """Every run of a seed read that seed's one stream: one path, chunks equal where they meet."""
    problems = []
    by_seed: dict[int, list[Trajectory]] = defaultdict(list)
    for run in runs:
        by_seed[run.seed].append(run)
    for seed, group in sorted(by_seed.items()):
        paths = {run.stream.get("path") for run in group}
        if len(paths) != 1:
            problems.append(f"seed {seed}: runs read {len(paths)} different streams")
            continue
        longest = max((run.stream.get("chunks", []) for run in group), key=len)
        for run in group:
            chunks = run.stream.get("chunks", [])
            if chunks != longest[: len(chunks)]:
                problems.append(f"seed {seed}: {run.run_id} read other chunks of the stream")
    return problems


def _response_problems(
    run_dir: Path, cfg: Any, evaluations: Sequence[Mapping[str, Any]], prompts: Mapping[str, Any]
) -> list[str]:
    """Every evaluation answered exactly the recorded harmful prompts, once each, in order."""
    if not evaluations:
        return []
    if not {"set", "count", "ids_sha256"} <= set(prompts):
        return ["the run does not record its harmful prompts"]
    if prompts["set"] != cfg.eval.harmful.prompt_set:
        return [
            f"responses to {prompts['set']}, the config evaluates {cfg.eval.harmful.prompt_set}"
        ]
    rows = _jsonl(run_dir / "responses" / f"{prompts['set']}.jsonl")
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row.get("condition")].append(row)
    problems = []
    conditions = [row.get("condition") for row in evaluations]
    if set(by_condition) != set(conditions):
        problems.append("stored harmful responses for conditions other than the evaluations")
    for condition in conditions:
        answered = by_condition.get(condition, [])
        ids = [row.get("item_id") for row in answered]
        digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        whole = all(
            isinstance(row.get("response_ids"), list)
            and row["response_ids"]
            and isinstance(row.get("text"), str)
            for row in answered
        )
        if len(ids) != prompts["count"] or digest != prompts["ids_sha256"] or not whole:
            problems.append(
                f"the harmful responses of {condition} are not the recorded "
                f"{prompts['count']} prompts, once each, with responses"
            )
    return problems


def load_trajectories(*roots: Path) -> list[Trajectory]:
    """Every scheduled run directory under the ``roots``, each once even if they overlap."""
    dirs = sorted({p.parent.resolve() for root in roots for p in Path(root).rglob("terminal.json")})
    return [load_trajectory(d) for d in dirs]


Key = tuple[str, int, float, int]  # arm, batch size, rate, seed


def rate(lr: float) -> float:
    """A rate as written in a config, for exact matching against the grids."""
    return float(f"{lr:.6g}")


def index(runs: Iterable[Trajectory]) -> tuple[dict[Key, Trajectory], list[str]]:
    """One run per arm, batch, rate and seed; a second run of a key is a problem."""
    out: dict[Key, Trajectory] = {}
    problems = []
    for run in runs:
        key = (run.arm, run.batch_size, rate(run.lr), run.seed)
        if key in out and out[key].run_dir != run.run_dir:
            problems.append(f"two runs of {key}: {out[key].run_id} and {run.run_id}")
            continue
        out[key] = run
    return out, problems


# ------------------------------------------------------------------ summaries


def utility_loss(row: Mapping[str, Any], theta0: Mapping[str, float]) -> tuple[float, float]:
    """(MMLU, factual) losses of an evaluation against θ₀, in percentage points."""
    return (
        100 * (theta0["benchmark_accuracy"] - row["benchmark_accuracy"]),
        100 * (theta0["factual_accuracy"] - row["factual_accuracy"]),
    )


def cells(runs: Iterable[Trajectory]) -> dict[tuple[str, int, float], list[Trajectory]]:
    out: dict[tuple[str, int, float], list[Trajectory]] = defaultdict(list)
    for run in runs:
        out[(run.arm, run.batch_size, rate(run.lr))].append(run)
    return {key: sorted(group, key=lambda r: r.seed) for key, group in sorted(out.items())}


def level_summary(
    cell: Sequence[Trajectory], level: float, theta0: Mapping[str, float], seeds: Sequence[int]
) -> dict[str, Any]:
    """Reachability, median steps, scoring K and mean utility losses at one level."""
    rows = [(run, run.evaluation_at(level)) for run in cell if run.seed in seeds]
    reached = [(run, row) for run, row in rows if row is not None]
    losses = [utility_loss(row, theta0) for _, row in reached]
    return {
        "reached": sorted(run.seed for run, _ in reached),
        "all_seeds": {run.seed for run, _ in reached} == set(seeds),
        "median_steps": statistics.median(row["step"] for _, row in reached) if reached else None,
        "median_K_benign": (
            statistics.median(row["K_benign"] for _, row in reached) if reached else None
        ),
        "mmlu_loss_pp": statistics.mean(m for m, _ in losses) if losses else None,
        "factual_loss_pp": statistics.mean(f for _, f in losses) if losses else None,
    }


def in_bounds(summary: Mapping[str, Any]) -> bool:
    mmlu, factual = summary["mmlu_loss_pp"], summary["factual_loss_pp"]
    return mmlu is not None and mmlu <= MMLU_BOUND_PP and factual <= FACTUAL_BOUND_PP


def cell_summary(
    cell: Sequence[Trajectory], theta0: Mapping[str, float], seeds: Sequence[int]
) -> dict[str, Any]:
    first = cell[0]
    out: dict[str, Any] = {
        "arm": first.arm,
        "batch_size": first.batch_size,
        "lr": first.lr,
        "seeds": [run.seed for run in cell],
        "runs": [run.run_id for run in cell],
        "stops": dict(Counter(run.stop for run in cell)),
        "median_training_seconds": statistics.median(run.training_seconds for run in cell),
        "max_spike_ratio": {run.seed: spike_ratio(run) for run in cell},
    }
    if first.policy == "k_stopped":
        out["levels"] = {
            _name(level): level_summary(cell, level, theta0, seeds) for level in first.levels
        }
    else:
        out["final"] = final_summary(cell, theta0, seeds)
    return out


def final_summary(
    cell: Sequence[Trajectory], theta0: Mapping[str, float], seeds: Sequence[int]
) -> dict[str, Any]:
    """A fixed arm at its last scheduled evaluation: every seed must have it."""
    rows = [
        run.evaluations[-1]
        for run in cell
        if run.seed in seeds
        and run.evaluations
        and run.stable
        and run.evaluations[-1]["step"] == max(run.evaluate_at)
    ]
    losses = [utility_loss(row, theta0) for row in rows]
    evaluated = [
        run.seed
        for run in cell
        if run.seed in seeds
        and run.evaluations
        and run.stable
        and run.evaluations[-1]["step"] == max(run.evaluate_at)
    ]
    return {
        "complete": sorted(evaluated) == sorted(set(seeds)),
        "median_K_benign": statistics.median(r["K_benign"] for r in rows) if rows else None,
        "mmlu_loss_pp": statistics.mean(m for m, _ in losses) if losses else None,
        "factual_loss_pp": statistics.mean(f for _, f in losses) if losses else None,
    }


def spike_ratio(run: Trajectory, window: int = 20) -> float | None:
    """The largest gradient norm over the median of the preceding (up to ``window``) norms, the
    statistic of the gradient rule, including a firing step (not in the metrics)."""
    norms = list(run.grad_norms)
    detail = run.terminal["stop"].get("detail", {})
    if "grad_norm" in detail:
        norms.append(detail["grad_norm"])
    ratios = [
        norms[s] / statistics.median(norms[max(0, s - window) : s]) for s in range(1, len(norms))
    ]
    return max(ratios) if ratios else None


def instability_summary(
    runs: Sequence[Trajectory], *, current_factor: float = GRAD_FACTOR
) -> dict[str, Any]:
    """The gradient rule over the discovery runs and its revision, decided before validation.

    The *margin* is the largest ratio of a gradient norm to its reference on runs that ended
    stably: the main quantity. A gradient-rule stop in a cell (arm, batch, rate) where another
    seed ran stably is a *possible* false alarm: breakdowns near the edge vary by seed, and a
    stopped run cannot show whether it would have broken down. The factor is raised only if the
    margin exceeds ``MARGIN_TRIGGER`` or a possible false alarm occurred, to
    max(current, ``REVISION_FACTOR`` × margin); it is never lowered.
    """
    firings, possible, exposed = [], [], 0
    for cell in cells(runs).values():
        if cell[0].arm == "soft_control":
            continue
        stable = [run for run in cell if run.stable]
        fired = [run for run in cell if run.stop == "gradient_spike"]
        firings += [run.run_id for run in fired]
        if stable:
            exposed += len(cell)
            possible += [run.run_id for run in fired]
    margins = [
        r
        for run in runs
        if run.stable and run.arm != "soft_control"
        if (r := spike_ratio(run)) is not None
    ]
    margin = max(margins) if margins else None
    revise = bool(possible) or (margin is not None and margin > MARGIN_TRIGGER)
    new = max(current_factor, REVISION_FACTOR * (margin or 0.0)) if revise else current_factor
    return {
        "firings": firings,
        "possible_false_alarms": possible,
        "runs_in_stable_cells": exposed,
        "possible_false_alarm_rate": len(possible) / exposed if exposed else None,
        "margin_on_stable_runs": margin,
        "factor": {"current": current_factor, "revise": revise, "new": new},
    }


def agreement(runs: Sequence[Trajectory]) -> dict[str, Any]:
    """The monitor against the scoring panel at every evaluation that sits on a reading."""
    monitor, scoring = [], []
    for run in runs:
        readings = dict(run.monitor)
        for row in run.evaluations:
            if row["step"] in readings:
                monitor.append(readings[row["step"]])
                scoring.append(row["K_benign"])
    return monitor_agreement(monitor, scoring)


def short_grid_extension(
    runs: Sequence[Trajectory],
    arm: str,
    *,
    batch: int = 5,
    grid_minimum: float = 0.5e-5,
    halvings: int = 2,
) -> float | None:
    """The next rate of a 25-step arm's grid at ``batch``: half the smallest rate run so far if
    its median K_benign over its seeds already exceeds 5, at most ``halvings`` times below the
    grid; None otherwise."""
    rates = sorted(lr for a, b, lr in cells(runs) if (a, b) == (arm, batch))
    if not rates:
        return None
    smallest = rates[0]
    cell = cells(runs)[(arm, batch, smallest)]
    ks = [run.evaluations[-1]["K_benign"] for run in cell if run.evaluations]
    if not ks or statistics.median(ks) <= LOWER:
        return None
    if smallest / 2 < grid_minimum / 2**halvings * (1 - 1e-9):
        return None
    return smallest / 2


def control_schedule(regular: Sequence[Trajectory]) -> dict[str, Any]:
    """The schedule of a full-length soft-target control for one rate, batch and seed: as long
    as the longest regular discovery run of that rate and batch, evaluated at the evaluation
    steps of the regular run of the same seed and at its own last step."""
    if not regular or len({(r.arm, r.batch_size, r.lr) for r in regular}) != 1:
        raise ValueError("The regular runs of one arm, batch and rate are needed")
    steps = max(run.steps for run in regular)
    return {
        "steps": steps,
        "seeds": {
            run.seed: sorted({row["step"] for row in run.evaluations} | {steps}) for run in regular
        },
    }


# ------------------------------------------------------------------ numerics


def contamination(control: Trajectory) -> float | None:
    """max over the control's evaluations of its scoring K_benign: c(t) = K_b(θ₀ ‖ θ_soft(t))."""
    values = [row["K_benign"] for row in control.evaluations]
    return max(values) if values else None


def screen_candidates(
    screen: Sequence[Trajectory], *, k_star: float = DISCOVERY_K_STAR
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """The screen: one 100-step soft-target control of every SGD rate at both batches on seed
    2000. A rate passes when its control ran stably within the contamination threshold; the two
    largest passing rates of a batch are its candidates. Returns them and the problems of an
    incomplete screen."""
    threshold = CONTAMINATION_FRACTION * k_star
    found, problems = index(screen)
    expected = {("soft_control", b, rate(lr), CONTROL_SEED) for b in BATCHES for lr in SGD_GRID}
    problems += [f"screen: missing {key}" for key in sorted(expected - set(found))]
    problems += [f"screen: unexpected {key}" for key in sorted(set(found) - expected)]
    problems += [
        f"screen: {run.run_id} ran {run.planned_steps} steps, not {SCREEN_STEPS}"
        for run in found.values()
        if run.planned_steps != SCREEN_STEPS
    ]
    out: dict[int, dict[str, Any]] = {}
    for batch in BATCHES:
        rates = {}
        for lr in SGD_GRID:
            run = found.get(("soft_control", batch, rate(lr), CONTROL_SEED))
            if run is None:
                continue
            c = contamination(run)
            rates[rate(lr)] = {
                "run": run.run_id,
                "contamination": c,
                "stop": run.stop,
                "passes": run.stable and c is not None and c <= threshold,
            }
        passing = sorted((lr for lr, r in rates.items() if r["passes"]), reverse=True)
        out[batch] = {
            "rates": rates,
            "threshold": threshold,
            "candidates": passing[:CANDIDATES_PER_BATCH],
        }
    return out, problems


def expected_discovery(
    candidates: Mapping[int, Sequence[float]],
    present: Mapping[Key, Trajectory],
    seeds: Sequence[int] = DISCOVERY_SEEDS,
) -> set[Key]:
    """The discovery runs the preregistered search prescribes: SGD-long at the screen's
    candidates, AdamW-long on its grid, the 25-step arms on their grid and, at batch 5, the
    downward extensions the runs present call for."""
    expected: set[Key] = set()
    for batch in BATCHES:
        rates = {
            "sgd_long": candidates.get(batch, []),
            "adamw_long": ADAMW_LONG_GRID,
            **{arm: SHORT_GRID for arm in SHORT_ARMS},
        }
        for arm, grid in rates.items():
            expected |= {(arm, batch, rate(lr), seed) for lr in grid for seed in seeds}
    for arm in SHORT_ARMS:
        grid = list(SHORT_GRID)
        while True:  # each extension is prescribed by the runs of the smallest rate so far
            runs = [
                present[(arm, 5, rate(lr), s)]
                for lr in grid
                for s in seeds
                if (arm, 5, rate(lr), s) in present
            ]
            nxt = short_grid_extension(runs, arm)
            if nxt is None or rate(nxt) in map(rate, grid):
                break
            grid.append(nxt)
            expected |= {(arm, 5, rate(nxt), seed) for seed in seeds}
    return expected


def numerical_acceptance(
    controls: Sequence[Trajectory],
    regular: Mapping[Key, Trajectory],
    candidates: Mapping[int, Sequence[float]],
    *,
    k_star: float = DISCOVERY_K_STAR,
) -> tuple[dict[tuple[int, float], dict], list[str]]:
    """Full-length soft-target controls: exactly one per candidate rate, on seed 2000, of the
    same model and on the same stream as the regular run of that seed, planned as long as the
    longest regular discovery run of the rate and batch and evaluated at that seed's regular
    evaluation steps and its last step (otherwise a problem: no conclusion). Accepted when stable
    and with max c(t) ≤ 0.01 × K*."""
    threshold = CONTAMINATION_FRACTION * k_star
    found, problems = index(controls)
    wanted = {
        ("soft_control", b, rate(lr), CONTROL_SEED) for b, lrs in candidates.items() for lr in lrs
    }
    problems += [f"controls: missing {key}" for key in sorted(wanted - set(found))]
    problems += [f"controls: unexpected {key}" for key in sorted(set(found) - wanted)]
    out = {}
    for key in sorted(wanted & set(found)):
        control, (_, batch, lr, seed) = found[key], key
        twin = regular.get(("sgd_long", batch, lr, seed))
        if twin is None:
            problems.append(f"controls: no regular run of seed {seed} at {(batch, lr)}")
            continue
        common = min(len(control.stream.get("chunks", [])), len(twin.stream.get("chunks", [])))
        if (
            control.model != twin.model
            or control.stream.get("path") != twin.stream.get("path")
            or control.stream["chunks"][:common] != twin.stream["chunks"][:common]
        ):
            problems.append(
                f"controls: {control.run_id} is not on the model and stream of {twin.run_id}"
            )
            continue
        longest = max(run.steps for k, run in regular.items() if k[:3] == ("sgd_long", batch, lr))
        planned = set(control.evaluate_at)
        needed = {r["step"] for r in twin.evaluations} | {longest}
        if control.planned_steps != longest or not needed <= planned:
            # Not the prescribed control: no conclusion, whatever it showed.
            problems.append(
                f"controls: {control.run_id} planned {control.planned_steps} steps evaluated at "
                f"{sorted(planned)}; its rate needs {longest} steps evaluated at {sorted(needed)}"
            )
            continue
        reasons = []  # scientific: only instability or contamination under the prescribed plan
        if not control.stable:
            reasons.append(f"unstable ({control.stop})")
        c = contamination(control)
        if c is None or c > threshold:
            reasons.append(f"contamination {c} above {threshold:.4g}")
        out[(batch, lr)] = {
            "contamination": c,
            "threshold": threshold,
            "accepted": not reasons,
            "reasons": reasons,
            "run": control.run_id,
        }
    return out, problems


# ------------------------------------------------------------------ selection


def select(
    discovery: Sequence[Trajectory],
    screen: Sequence[Trajectory],
    controls: Sequence[Trajectory],
    theta0: Mapping[str, float],
) -> dict[str, Any]:
    """The Stage-A selection rule over the whole prescribed search; never reads harm.

    Outcomes: ``incomplete`` (the search or its records are not whole: no conclusion),
    ``numerics_fail`` (no SGD rate passes a whole screen, or no candidate its full control),
    ``barrier`` (no accepted SGD rate reaches U = 7 at either batch), ``infeasible`` (no batch
    eligible otherwise) or ``selected``.
    """
    seeds = DISCOVERY_SEEDS
    out: dict[str, Any] = {
        "seeds": list(seeds),
        "rulings": RULINGS,
        "runs": sorted(run.run_id for run in discovery),
    }
    screened, problems = screen_candidates(screen)
    problems += conformance(screen, "screen") + stream_pairing(screen)
    out["screen"] = screened
    if problems:
        out.update(outcome="incomplete", problems=problems)
        return out
    candidates = {batch: entry["candidates"] for batch, entry in screened.items()}
    if not any(candidates.values()):  # a whole screen with no passing rate at either batch
        out["outcome"] = "numerics_fail"
        return out
    present, duplicates = index(discovery)
    problems += duplicates + conformance(discovery, "discovery")
    problems += conformance(controls, "full_control")
    problems += stream_pairing([*screen, *discovery, *controls])
    expected = expected_discovery(candidates, present, seeds)
    problems += [f"discovery: missing {key}" for key in sorted(expected - set(present))]
    problems += [f"discovery: unexpected {key}" for key in sorted(set(present) - expected)]
    acceptance, control_problems = numerical_acceptance(controls, present, candidates)
    problems += control_problems
    prompts = {
        json.dumps(run.harmful_prompts, sort_keys=True) for run in discovery if run.evaluations
    }
    if len(prompts) > 1:
        problems.append("discovery runs answered different harmful prompt sets")
    out["acceptance"] = {f"B={b} lr={lr:g}": a for (b, lr), a in acceptance.items()}
    if problems:
        out.update(outcome="incomplete", problems=problems)
        return out
    if not any(a["accepted"] for a in acceptance.values()):
        out["outcome"] = "numerics_fail"
        return out

    by_cell = cells(present.values())
    chunk_seconds = _seconds_per_chunk(discovery)
    report: dict[int, dict[str, Any]] = {}
    for batch in BATCHES:
        entry: dict[str, Any] = {"levels": {}}
        for upper in UPPER:
            k_star = math.sqrt(LOWER * upper)
            sgd = _long_rates(by_cell, "sgd_long", batch, upper, k_star, theta0, seeds, acceptance)
            adamw = _long_rates(by_cell, "adamw_long", batch, upper, k_star, theta0, seeds, None)
            entry["levels"][_name(upper)] = {
                "K_star": k_star,
                "sgd_long": sgd,
                "adamw_long": adamw,
                "qualifies": bool(_ok(sgd) and _ok(adamw)),
            }
        qualifying = [u for u in UPPER if entry["levels"][_name(u)]["qualifies"]]
        entry["U"] = qualifying[0] if qualifying else None
        if entry["U"] is not None:
            entry.update(
                _choose(
                    by_cell,
                    batch,
                    entry["U"],
                    entry["levels"][_name(entry["U"])],
                    theta0,
                    seeds,
                    chunk_seconds,
                )
            )
        report[batch] = entry
    out["batches"] = report
    eligible = [b for b in BATCHES if report[b]["U"] is not None]
    if not eligible:
        out["outcome"] = _failure(by_cell, theta0, seeds, acceptance)
        return out
    best = max(report[b]["U"] for b in eligible)
    ties = [b for b in eligible if report[b]["U"] == best]
    batch = min(ties, key=lambda b: report[b]["seconds_per_block"])
    chosen = report[batch]
    eight = chosen["levels"][_name(8.0)]
    limited = chosen["U"] == 7.0 and not all(
        any(r["reaches"] for r in eight[arm].values()) for arm in ("sgd_long", "adamw_long")
    )
    keys = ("sgd_long", "adamw_long", "adamw_25", "frozen_adam_25", "t_max", "seconds_per_block")
    out.update(
        {
            "outcome": "selected",
            "batch_size": batch,
            "U": chosen["U"],
            "K_star": math.sqrt(LOWER * chosen["U"]),
            "levels": [LOWER, math.sqrt(LOWER * chosen["U"]), chosen["U"]],
            "u_limited_by_reachability": limited,
            **{k: chosen[k] for k in keys},
        }
    )
    return out


RULINGS = __doc__.split("**Rulings**")[1].strip() if __doc__ else ""


def _long_rates(by_cell, arm, batch, upper, k_star, theta0, seeds, acceptance) -> dict:
    """Each rate of a long arm at one batch and U: whether it meets its selection rule."""
    out = {}
    for (cell_arm, cell_batch, lr), cell in by_cell.items():
        if (cell_arm, cell_batch) != (arm, batch):
            continue
        levels = [level_summary(cell, level, theta0, seeds) for level in (LOWER, k_star, upper)]
        reasons = []
        reaches = levels[2]["all_seeds"]
        if not reaches:
            reasons.append(f"U = {upper:g} reached by seeds {levels[2]['reached']} only")
        if not all(in_bounds(level) for level in levels):
            reasons.append("utility out of bounds at 5, K* or U")
        if acceptance is not None and not acceptance.get((batch, lr), {}).get("accepted"):
            reasons.append("numerical acceptance not passed")
        if arm == "adamw_long":
            for run in cell:
                steps = [run.evaluation_at(level) for level in (LOWER, k_star, upper)]
                if None not in steps and len({row["step"] for row in steps}) < 3:
                    reasons.append(f"seed {run.seed}: 5, K* and U not three separate evaluations")
        out[lr] = {
            "ok": not reasons,
            "reaches": reaches,
            "reasons": reasons,
            "median_steps_to_U": levels[2]["median_steps"],
        }
    return out


def _ok(rates: Mapping[float, Mapping[str, Any]]) -> list[float]:
    return sorted(lr for lr, r in rates.items() if r["ok"])


def _choose(by_cell, batch, upper, level, theta0, seeds, chunk_seconds) -> dict[str, Any]:
    k_star = level["K_star"]
    sgd_lr, adamw_lr = max(_ok(level["sgd_long"])), max(_ok(level["adamw_long"]))
    targets = (LOWER, k_star, upper)
    short = {
        arm: _short_rates(by_cell, arm, batch, targets, theta0, seeds)
        for arm in ("adamw_25", "frozen_adam_25")
    }
    steps = [
        level["sgd_long"][sgd_lr]["median_steps_to_U"],
        level["adamw_long"][adamw_lr]["median_steps_to_U"],
    ]
    block = [
        by_cell[("sgd_long", batch, sgd_lr)],
        by_cell[("adamw_long", batch, adamw_lr)],
        *(by_cell[("adamw_25", batch, lr)] for lr in short["adamw_25"]["rates"]),
    ]
    return {
        "sgd_long": {"lr": sgd_lr, "median_steps_to_U": steps[0]},
        "adamw_long": {"lr": adamw_lr, "median_steps_to_U": steps[1]},
        **short,
        "t_max": min(T_MAX_CAP, math.ceil(T_MAX_FACTOR * max(steps))),
        "seconds_per_block": _block_seconds(block, seeds, chunk_seconds),
    }


def _short_rates(by_cell, arm, batch, targets, theta0, seeds) -> dict[str, Any]:
    qualifying = {}
    for (cell_arm, cell_batch, lr), cell in by_cell.items():
        if (cell_arm, cell_batch) == (arm, batch):
            final = final_summary(cell, theta0, seeds)
            if final["complete"] and in_bounds(final):
                qualifying[lr] = final["median_K_benign"]
    n = min(len(qualifying), len(targets))
    best: tuple[float, tuple[float, ...], tuple[tuple[int, float], ...]] | None = None
    for rates in itertools.permutations(sorted(qualifying), n):
        for slots in itertools.combinations(range(len(targets)), n):
            pairs = tuple(zip(slots, rates, strict=True))  # (target index, rate)
            cost = sum(abs(qualifying[lr] - targets[t]) for t, lr in pairs)
            key = (cost, tuple(sorted(rates)), pairs)  # ties: the smaller rates
            if best is None or key < best:
                best = key
    pairs = best[2] if best is not None else ()
    return {
        "rates": [lr for _, lr in pairs],
        "targets": [targets[t] for t, _ in pairs],
        "shortfall": len(targets) - n,
        "median_K_benign": {lr: qualifying[lr] for _, lr in pairs},
    }


def _block_seconds(block: Sequence[Sequence[Trajectory]], seeds, chunk_seconds) -> float | None:
    totals = []
    for seed in seeds:
        runs = [next((r for r in cell if r.seed == seed), None) for cell in block]
        if any(run is None for run in runs):
            continue
        responses = max(run.terminal["responses"] for run in runs)
        chunks = -(-responses // runs[0].chunk_size)
        totals.append(sum(run.training_seconds for run in runs) + chunks * (chunk_seconds or 0))
    return statistics.median(totals) if totals else None


def _seconds_per_chunk(runs: Sequence[Trajectory]) -> float | None:
    seconds = sum(run.terminal.get("stream_extension", {}).get("seconds", 0.0) for run in runs)
    chunks = sum(run.terminal.get("stream_extension", {}).get("chunks", 0) for run in runs)
    return seconds / chunks if chunks else None


def _failure(by_cell, theta0, seeds, acceptance) -> str:
    """Which Stage-A outcome stops the design once some SGD rate is accepted (the table
    "Outcomes of Stage A")."""
    reaches_seven = False
    for batch in BATCHES:
        rates = _long_rates(
            by_cell, "sgd_long", batch, 7.0, math.sqrt(35), theta0, seeds, acceptance
        )
        reaches_seven |= any(
            r["reaches"]
            for lr, r in rates.items()
            if acceptance.get((batch, lr), {}).get("accepted")
        )
    return "infeasible" if reaches_seven else "barrier"


def _name(level: float) -> str:
    return f"{level:.4f}"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]
