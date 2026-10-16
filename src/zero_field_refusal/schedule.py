"""Study-2 runs: lazy batches, monitor readings, evaluations at K levels, and stops.

The rules are those of docs/study2-preregistration.md ("Panels, monitoring and stopping"):

- **Batches** come from the seed's appendable stream in order, read lazily (``BatchFeed``). Each
  chunk is checked when first read, before any step trains on it: the teacher gap of its
  responses on the training path of the frozen θ₀ (invariant 1) and the run's own path-field
  bound for the run's maximum exposure, ``train.steps`` steps. A run may extend the stream with
  its frozen θ₀ (one writer per stream: see ``runtime.extend_stream``).
- **Monitor.** Every ``eval.monitor.every`` steps K is read on the monitor panel.
- **Levels** (policy ``k_stopped``). When the last two readings lie at or above a level not yet
  reached, the run is evaluated in full in place: the scoring panel's K, the greedy responses to
  the harmful prompts (stored for the judge, not scored) and utility. One evaluation can reach
  several levels. After the evaluation that reaches the top level the run stops (``target``).
- **Plateau.** From step ``from_step`` on, at every multiple of ``window`` steps: over the last
  ``windows`` windows the medians of the monitor readings span at most a factor ``span``, the
  median of the last half's medians is at most ``drift`` times the first half's, and all lie
  below ``below_next`` times the next level: the run stops (``plateau``).
- **Instability**, checked before every update from the first gradient: a non-finite loss,
  gradient, weight or K; a monitor K above ``k_factor`` times the top level; and the gradient
  rule ``grad_rule``: a gradient norm above ``grad_factor`` times the median of the preceding (up
  to ``grad_window``) step norms, the reference starting at the step-0 norm. Under ``spike`` (the
  default) it stops the run (``gradient_spike``); under ``log`` (the SGD-momentum pilot's
  soft-target controls, docs/study2-pilot-momentum.md) it is only recorded, in the terminal
  record's ``gradient_spikes``, and the update is taken: a soft-target control is judged by its
  scoring K. Either way the firing norm stays out of the reference, which is therefore the last
  ``grad_window`` norms that did not fire and may reach further back; under ``log`` a sustained
  rise can stay flagged at every step, which only lengthens the record. With
  ``contamination_k`` (the pilot's soft-target controls), a monitor reading or a scoring
  diagnostic above it is checked at once on the scoring panel without the cache
  (``contamination_checks.jsonl``), and a verified scoring K above it stops the control
  (``contamination``). One step suffices. An unstable run stops without that update; it is a
  result, recorded, not a failure.
- **Limit.** A ``k_stopped`` run stops at ``train.steps`` (T_max, ``t_max``); a ``fixed`` run
  takes exactly ``train.steps`` steps (``steps``), evaluated at the steps in ``evaluate_at``,
  and stops early only on instability.

During the run only K, utility and training statistics leave it; refusal and harm are not
computed. A later stop never removes an earlier evaluation. The run ends with a terminal record
(``terminal.json``): the stop and its reason, the steps taken, the levels reached (with the step
and the scoring panel's K) and those missed.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel

from .curvature import displacement_in_place, rayleigh_panel, record_endpoint_rayleigh
from .models import freeze
from .optim import ADAM_MODES, build_optimizer
from .panels import CacheMismatch, Monitor, ScoringDiagnostics, cache_check
from .prechecks import check_run_chi_square, check_teacher_gap, require_path_check
from .runtime import Environment, ReplayedSoftTargets, StreamPrefix, extend_stream
from .seeding import derive_seed, seed_everything
from .teacher import Sample
from .tracking import Diverged, Run
from .train import (
    Evaluator,
    _append_jsonl,
    _theta0_second_moment,
    accumulate_gradient,
    projection_directions,
    projection_stats,
    teacher_gap,
)

POLICIES = ("k_stopped", "fixed")
INSTABILITY = ("nonfinite_loss", "nonfinite_gradient", "nonfinite_weight", "nonfinite_k",
               "k_above_max", "gradient_spike", "contamination")  # fmt: skip
GRAD_RULES = ("spike", "log")


@dataclass(frozen=True)
class PlateauRule:
    from_step: int
    window: int
    windows: int
    span: float
    drift: float
    below_next: float

    def plateaued(
        self, readings: Sequence[tuple[int, float]], step: int, next_level: float
    ) -> bool:
        """Whether the monitor readings ``(step, K)`` up to ``step`` show a plateau below
        ``next_level``; decided only at multiples of ``window`` from ``from_step`` on."""
        if step < self.from_step or step % self.window:
            return False
        medians = []
        for w in range(self.windows):
            low = step - self.window * (self.windows - w)
            values = [k for s, k in readings if low < s <= low + self.window]
            if not values:
                return False
            medians.append(statistics.median(values))
        half = self.windows // 2
        return (
            max(medians) <= self.span * min(medians)
            and statistics.median(medians[-half:]) <= self.drift * statistics.median(medians[:half])
            and all(m < self.below_next * next_level for m in medians)
        )


class LevelTracker:
    """Levels crossed by the monitor: a level is crossed when the last ``readings`` readings lie
    at or above it, and reached only once an evaluation there succeeded (``reach``)."""

    def __init__(self, levels: Sequence[float], readings: int) -> None:
        self.levels, self.readings = list(levels), readings
        self.reached: list[float] = []
        self._recent: list[float] = []

    def update(self, k: float) -> list[float]:
        """Record a reading; returns the levels not yet reached that the last readings cross."""
        self._recent = [*self._recent, k][-self.readings :]
        if len(self._recent) < self.readings:
            return []
        return [
            level for level in self.levels
            if level not in self.reached and min(self._recent) >= level
        ]  # fmt: skip

    def reach(self, levels: Sequence[float]) -> None:
        self.reached += [level for level in levels if level not in self.reached]

    @property
    def next_level(self) -> float | None:
        missing = [level for level in self.levels if level not in self.reached]
        return missing[0] if missing else None


class InstabilityGuard:
    def __init__(
        self, grad_factor: float, grad_window: int, k_max: float, grad_rule: str = "spike"
    ) -> None:
        if grad_rule not in GRAD_RULES:
            raise ValueError(f"grad_rule must be one of {GRAD_RULES}, got {grad_rule!r}")
        self.grad_factor, self.grad_window, self.k_max = grad_factor, grad_window, k_max
        self.grad_rule = grad_rule
        self.norms: list[float] = []
        self.logged: list[dict[str, float]] = []  # under "log": every firing, with its step
        self.detail: dict[str, float] = {}  # what fired: the value and its bound

    def gradient(self, loss: float, grad_norm: float) -> str | None:
        """Checked before an update; the accepted step's norm joins the reference."""
        if not math.isfinite(loss):
            return "nonfinite_loss"
        if not math.isfinite(grad_norm):
            return "nonfinite_gradient"
        reference = self.norms[-self.grad_window :]
        if reference and grad_norm > self.grad_factor * statistics.median(reference):
            fired = {"grad_norm": grad_norm, "reference_median": statistics.median(reference)}
            if self.grad_rule == "log":
                self.logged.append({"step": len(self.norms) + len(self.logged), **fired})
                return None
            self.detail = fired
            return "gradient_spike"
        self.norms.append(grad_norm)
        return None

    def monitor(self, k: float) -> str | None:
        if not math.isfinite(k):
            return "nonfinite_k"
        if k > self.k_max:
            self.detail = {"K_monitor": k, "k_max": self.k_max}
            return "k_above_max"
        return None


@dataclass(frozen=True)
class Schedule:
    policy: str
    levels: tuple[float, ...]
    steps: int  # T_max, or the fixed length
    evaluate_at: tuple[int, ...]
    monitor_every: int
    crossing_readings: int
    plateau: PlateauRule
    grad_factor: float
    grad_window: int
    k_max: float
    extend_stream: bool
    grad_rule: str = "spike"
    contamination_k: float | None = None  # the pilot's soft-target controls only

    @classmethod
    def from_config(cls, cfg: DictConfig) -> Schedule:
        s = cfg.schedule
        levels = tuple(float(level) for level in s.levels)
        return cls(
            policy=s.policy, levels=levels, steps=int(cfg.train.steps),
            evaluate_at=tuple(int(t) for t in s.get("evaluate_at") or ()),
            monitor_every=int(cfg.eval.monitor.every), crossing_readings=int(s.crossing_readings),
            plateau=PlateauRule(**{key: s.plateau[key] for key in PlateauRule.__annotations__}),
            grad_factor=float(s.instability.grad_factor),
            grad_window=int(s.instability.grad_window),
            k_max=float(s.instability.k_factor) * levels[-1],
            extend_stream=bool(s.extend_stream),
            grad_rule=str(s.instability.get("grad_rule", "spike")),
            contamination_k=(None if s.instability.get("contamination_k") is None
                             else float(s.instability.contamination_k)),
        )  # fmt: skip


class BatchFeed:
    """Batch s of a run of batch size B: responses sB … (s+1)B − 1 of the seed's appendable
    stream, read lazily. Each chunk is checked when first read; the checks go to
    ``chunk_checks.jsonl``."""

    def __init__(
        self, cfg: DictConfig, env: Environment, run: Run, reference: PreTrainedModel, *,
        extend: bool,
    ) -> None:  # fmt: skip
        self.cfg, self.env, self.run, self.reference, self.extend = cfg, env, run, reference, extend
        self.prefix = StreamPrefix(cfg, env, cfg.seed)
        self.size = cfg.train.batch_size
        self.extension_seconds = 0.0  # spent extending the stream, not training
        self.extended_chunks = 0
        self._checked: set[int] = set()

    def batch(self, step: int) -> list[Sample]:
        start, stop = step * self.size, (step + 1) * self.size
        if stop > self.prefix.available and self.extend:
            self._extend(stop)
        for chunk in range(start // self.prefix.size, -(-stop // self.prefix.size)):
            if chunk not in self._checked:
                self._check(chunk)
        return self.prefix.take(start, stop)

    def identity(self, responses: int) -> dict[str, Any]:
        return self.prefix.identity(responses)

    def _extend(self, stop: int) -> None:
        device = next(self.reference.parameters()).device
        started = time.perf_counter()
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            created = extend_stream(
                self.cfg, self.env, self.cfg.seed, stop, model=self.reference,
                batch_size=self.cfg.execution.generation_batch_size,
            )  # fmt: skip
        self.extension_seconds += time.perf_counter() - started
        self.extended_chunks += len(created)

    def _check(self, chunk: int) -> None:
        samples = self.prefix.take(chunk * self.prefix.size, (chunk + 1) * self.prefix.size)
        gap, chi_square = teacher_gap(self.reference, samples, self.cfg, self.env)
        row = {"chunk": chunk, "teacher_gap": gap}
        try:
            check_teacher_gap(gap, self.cfg.checks.teacher_gap)
            row["path_field"] = check_run_chi_square(
                chi_square, batch_size=self.size, steps=self.cfg.train.steps
            )
        finally:
            _append_jsonl(self.run.path("chunk_checks.jsonl"), [row])
        self._checked.add(chunk)


def run_scheduled(cfg: DictConfig, env: Environment, *, run_id: str | None = None) -> Path:
    """One study-2 run under ``cfg.schedule``; returns the run directory."""
    if cfg.run_type != "train":
        raise ValueError("run_scheduled needs run_type=train")
    schedule = Schedule.from_config(cfg)
    seed_everything(derive_seed(cfg.seed, "torch"))
    started = time.perf_counter()
    with Run(cfg, run_id=run_id) as run:
        record = require_path_check(env.data_root, cfg, env.device)  # before anything costly
        run.record_inputs({"path_check": {"run_id": record["run_id"],
                                          "fingerprint": record["fingerprint"]},
                           "prompt_manifest_sha256": env.prompt_manifest_sha256})  # fmt: skip
        student, reference = env.new_model(), env.new_model()
        reference.requires_grad_(False)
        trainable = freeze(student, cfg.train.freeze)
        params = [p for _, p in trainable]
        theta0 = dict(reference.named_parameters())
        evaluate = Evaluator(cfg, env, run, reference, sealed=True)  # fails early on inputs
        panel = rayleigh_panel(cfg, env, run)  # None unless eval.rayleigh.enabled
        monitor = Monitor(cfg, env, run, reference)
        diagnostics = ScoringDiagnostics(cfg, env, run, reference) if monitor.version > 1 else None
        feed = BatchFeed(cfg, env, run, reference, extend=schedule.extend_stream)
        replayer = None
        if cfg.control == "soft_targets":
            replay_log = run.path("soft_targets_replay.jsonl")
            replayer = ReplayedSoftTargets(
                feed.prefix, reference, env, on_record=lambda row: _append_jsonl(replay_log, [row])
            )
        directions = projection_directions(trainable, cfg.eval.projections)
        second_moment = None
        if ADAM_MODES.get(cfg.optimizer.name, "adaptive") != "adaptive":
            second_moment = _theta0_second_moment(cfg, env, student, params, run)
        optimizer = build_optimizer(cfg.optimizer, params, second_moment=second_moment)
        del second_moment

        def release() -> None:  # the next step's zero_grad(set_to_none) would free them anyway
            for p in params:
                p.grad = None

        loop = _Loop(schedule, run, evaluate, monitor, diagnostics, release)
        reason = None
        if diagnostics is not None:  # the caches against uncached readings, at θ₀
            reason = loop.check_caches(student, scoring=diagnostics(student, cached=False))
        size = cfg.train.batch_size
        for step in range(schedule.steps if reason is None else 0):
            batch = feed.batch(step)
            optimizer.zero_grad(set_to_none=True)
            soft = None if replayer is None else replayer.take(step * size, (step + 1) * size)
            stats = accumulate_gradient(student, batch, cfg, env, soft)
            stats["grad_norm"] = _norm([p.grad for p in params])
            reason = loop.guard.gradient(stats["loss"], stats["grad_norm"])
            if reason:  # refused before the update: the batch is recorded, not counted
                loop.rejected = {"step": step, "loss": stats["loss"],
                                 "grad_norm": stats["grad_norm"]}  # fmt: skip
                break
            optimizer.step()
            loop.steps = step + 1  # the update is applied, whatever it did to the weights
            run.log(stats, step=step)
            if not math.isfinite(_norm(params)):
                reason = "nonfinite_weight"
                loop.rejected = {"step": step, "loss": stats["loss"],
                                 "grad_norm": stats["grad_norm"], "applied": True}  # fmt: skip
                break
            reason = loop.after_step(student, lambda: projection_stats(trainable, theta0,
                                                                       directions))  # fmt: skip
            if reason:
                break
        else:
            if reason is None:  # every step taken (a θ₀ verdict takes none)
                reason = "t_max" if schedule.policy == "k_stopped" else "steps"
        for p in params:
            p.grad = None
        terminal = loop.terminal(reason, size)
        terminal["stream_extension"] = {"seconds": feed.extension_seconds,
                                        "chunks": feed.extended_chunks}  # fmt: skip
        terminal["seconds"] = time.perf_counter() - started  # the run's time, up to its outcome
        run.record_inputs({"streams": {"training": feed.identity(loop.steps * size)}})
        _publish(run.path("terminal.json"), terminal)  # the run's outcome, whatever follows
        run.summarize({"terminal": terminal})
        if replayer is not None and replayer.records:
            run.summarize({"soft_targets_replay": {
                "batches": len(replayer.records),
                "max_abs": max(r["max_abs"] for r in replayer.records),
                "digest_equal": sum(r["digest_equal"] for r in replayer.records),
            }})  # fmt: skip
        if panel is not None:  # the endpoint Rayleigh quotient, once the run is over
            delta = displacement_in_place(trainable, theta0)  # the student now holds θ_end − θ₀
            # The optimizer state, the panels and their caches, and the student's other tensors
            # are not needed for the product: free them before it.
            del optimizer, loop, release, evaluate, monitor, diagnostics, feed, replayer
            del directions, student, trainable, params
            record_endpoint_rayleigh(cfg, env, run, reference, delta, *panel)
        return run.dir


def _peak_memory() -> dict[str, float]:
    """The GPU memory peak of the run so far (allocated, reserved), in GB; nothing on the CPU."""
    if not torch.cuda.is_available():
        return {}
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3)}  # fmt: skip


class _Loop:
    """The run's state after each step: monitor readings, evaluations, levels and stops."""

    def __init__(
        self,
        schedule: Schedule,
        run: Run,
        evaluate: Evaluator,
        monitor: Monitor,
        diagnostics: ScoringDiagnostics | None = None,
        release: Callable[[], None] | None = None,
    ):
        self.schedule, self.run, self.evaluate, self.monitor = schedule, run, evaluate, monitor
        self.diagnostics = diagnostics
        if schedule.contamination_k is not None and diagnostics is None:
            raise ValueError("contamination_k needs the scoring diagnostics of monitor version 2")
        self.release = release  # frees the gradients, which a full evaluation does not need
        self.read_at: int | None = None  # the step of the last operational monitor reading
        self.diagnosed = 0
        self.guard = InstabilityGuard(schedule.grad_factor, schedule.grad_window, schedule.k_max,
                                      schedule.grad_rule)  # fmt: skip
        self.levels = LevelTracker(schedule.levels, schedule.crossing_readings)
        self.readings: list[tuple[int, float]] = []
        self.evaluations: list[dict[str, Any]] = []
        self.steps = 0
        self.rejected: dict[str, Any] | None = None  # the batch or update that stopped the run
        self.failed_crossing: dict[str, Any] | None = None  # levels whose evaluation failed

    def after_step(self, model: PreTrainedModel, geometry: Any) -> str | None:
        """Monitor, evaluate and decide after step ``self.steps``; returns a stop reason."""
        schedule, done = self.schedule, self.steps
        if done % schedule.monitor_every == 0:
            k = self.monitor(model)
            self.read_at = done
            self.readings.append((done, k))
            reading = {"step": done, "K_monitor": k, **geometry()}
            cached = self.diagnostics is not None and self.monitor.last is not None
            if not math.isfinite(k) and cached:  # the state's, or the cache's?
                self._confirm(self.monitor.last, lambda: self.monitor.uncached(model, "primary"),
                              "monitor")  # fmt: skip
            reason = None
            diagnostic = self.diagnostics is not None and done % self.diagnostics.every == 0
            if diagnostic and math.isfinite(k):
                out, reason = self._diagnose(model, done, k)
                reading |= out
            if reason is None and schedule.contamination_k is not None and math.isfinite(k):
                reason = self._contamination(model, done, k, reading.get("K_scoring"))
            _append_jsonl(self.run.path("monitor.jsonl"), [reading])
            self.run.log({"K_monitor": k}, step=done - 1)
            reason = reason or self.guard.monitor(k)
            if reason:
                return reason
            if schedule.policy == "k_stopped":
                crossed = self.levels.update(k)
                if crossed:
                    reason = self._evaluate(model, crossed, k)
                    if reason:
                        self.failed_crossing = {"step": done, "levels": crossed}
                        return reason
                    self.levels.reach(crossed)
                    if self.levels.next_level is None:
                        return "target"
                next_level = self.levels.next_level
                if schedule.plateau.plateaued(self.readings, done, next_level):
                    return "plateau"
        if schedule.policy == "fixed" and done in schedule.evaluate_at:
            return self._evaluate(model, [], self.readings[-1][1] if self.readings else None)
        return None

    def _contamination(
        self, model: PreTrainedModel, step: int, k: float, scoring_k: float | None
    ) -> str | None:
        """A pilot soft-target control's reading above the contamination bound (the monitor's,
        or the scoring diagnostic's of this step), checked at once on the scoring panel without
        the cache and recorded; a verified scoring K above the bound stops the control."""
        bound = self.schedule.contamination_k
        if k <= bound and (scoring_k is None or scoring_k <= bound):
            return None
        value = self.diagnostics(model, cached=False).mean().item()
        _append_jsonl(self.run.path("contamination_checks.jsonl"),
                      [{"step": step, "K_monitor": k, "K_scoring_cached": scoring_k,
                        "K_scoring": value, "bound": bound}])  # fmt: skip
        if not math.isfinite(value):  # the state's own reading: an outcome, whatever the cache
            return "nonfinite_k"
        if scoring_k is not None and value != scoring_k:  # the cache's, not the state's
            raise CacheMismatch(f"step {step}: the scoring diagnostic read {scoring_k} with the "
                                f"cache and {value} without it")  # fmt: skip
        if value > bound:
            self.guard.detail = {"K_scoring": value, "bound": bound}
            return "contamination"
        return None

    def _shadow(self, model: PreTrainedModel) -> dict[str, Any]:
        """The monitor's full panel at this state, reusing its reading if one was just taken."""
        last = self.monitor.last if self.read_at == self.steps else None
        return self.monitor.shadow(model, last)

    def _confirm(self, cached: Any, uncached: Callable[[], torch.Tensor], what: str) -> bool:
        """A cached reading with a non-finite value, read again without the cache: True when
        the state reproduces it (an outcome: ``nonfinite_k``); a value that only the cache made
        non-finite stops the run as a technical failure. False when the reading is finite."""
        if all(math.isfinite(v) for v in torch.as_tensor(cached).tolist()):
            return False
        if bool(torch.isfinite(uncached()).all()):
            raise CacheMismatch(f"step {self.steps}: the cached {what} reading is not finite, "
                                "the uncached one is")  # fmt: skip
        return True

    def _diagnose(
        self, model: PreTrainedModel, step: int, k: float
    ) -> tuple[dict[str, float], str | None]:
        """The scoring panel's K alone and the monitor's full panel (amendment 1), recorded with
        the step's operational reading ``k``; ``nonfinite_k`` when the state makes either one
        non-finite."""
        scoring = self.diagnostics(model)
        shadow = self._shadow(model)
        row = {"step": step, "K_scoring": scoring.mean().item(),
               "K_monitor": k, "K_monitor_full": shadow["K_full"], **_peak_memory(),
               "scoring": scoring.tolist(), "monitor_primary": shadow["primary"],
               "monitor_secondary": shadow["secondary"]}  # fmt: skip
        _append_jsonl(self.run.path("diagnostics.jsonl"), [row])
        self.diagnosed += 1
        out = {"K_scoring": row["K_scoring"], "K_monitor_full": row["K_monitor_full"]}
        self.run.log(out, step=step - 1)
        nonfinite = self._confirm(scoring, lambda: self.diagnostics(model, cached=False),
                                  "scoring")  # fmt: skip
        nonfinite |= self._confirm(shadow["secondary"],
                                   lambda: self.monitor.uncached(model, "secondary"),
                                   "second-response")  # fmt: skip
        return out, "nonfinite_k" if nonfinite else None

    def check_caches(self, model: PreTrainedModel, *, scoring: torch.Tensor) -> str | None:
        """Every cached reading against an uncached one of this state (``scoring``: the scoring
        panel's K per response, computed without the cache). A non-finite uncached reading is
        the state's (``nonfinite_k``); any other difference stops the run as technical."""
        checks = self.monitor.check(model)
        checks["scoring"] = cache_check(self.diagnostics(model), scoring)
        _append_jsonl(self.run.path("cache_checks.jsonl"), [{"step": self.steps, **checks}])
        if not all(check["uncached_finite"] for check in checks.values()):
            return "nonfinite_k"
        unequal = sorted(part for part, check in checks.items() if not check["bitwise_equal"])
        if unequal:
            raise CacheMismatch(f"step {self.steps}: cached readings differ in {unequal}")
        return None

    def _evaluate(self, model: PreTrainedModel, levels: list[float], k: float | None) -> str | None:
        condition = f"step-{self.steps}"
        extra: dict[str, Any] = {}
        if self.release is not None:  # the step's reading took its geometry already
            self.release()

        def auxiliary() -> None:
            """The full monitor panel and the caches at this state, before the evaluation writes
            anything: a non-finite verdict leaves no record of the condition."""
            shadow = self._shadow(model)
            extra.update({"K_monitor_primary": shadow["K_primary"],
                          "K_monitor_full": shadow["K_full"]})  # fmt: skip
            if self.check_caches(model, scoring=self.evaluate.last_kl["kl_benign"]):
                raise Diverged(f"a reading without its cache is not finite at {condition}")

        check = auxiliary if self.diagnostics is not None else None
        try:
            out = self.evaluate(model, condition, check=check)
        except Diverged:
            return "nonfinite_k"
        out = {**out, **extra, **(_peak_memory() if check is not None else {})}
        row = {"step": self.steps, "condition": condition, "levels": levels, "K_monitor": k, **out}
        self.evaluations.append(row)
        _append_jsonl(self.run.path("evaluations.jsonl"), [row])
        self.run.log({f"eval/{key}": value for key, value in out.items()}, step=self.steps - 1)
        return None

    def terminal(self, reason: str, batch_size: int) -> dict[str, Any]:
        kind = ("instability" if reason in INSTABILITY else
                "limit" if reason in ("t_max", "steps") else reason)  # fmt: skip
        reached = [
            {"level": level, "step": row["step"], "K_benign": row["K_benign"]}
            for row in self.evaluations for level in row["levels"]
        ]  # fmt: skip
        stop: dict[str, Any] = {"reason": reason, "kind": kind}
        if reason in INSTABILITY and self.guard.detail:
            stop["detail"] = self.guard.detail
        if self.rejected is not None:
            stop["rejected"] = self.rejected
        out = {"stop": stop, "steps": self.steps,
               "responses": self.steps * batch_size, "policy": self.schedule.policy,
               "evaluations": [row["step"] for row in self.evaluations],
               "monitor_readings": len(self.readings)}  # fmt: skip
        if self.schedule.grad_rule == "log":  # the gradient rule's firings, recorded only
            out["gradient_spikes"] = self.guard.logged
        if self.diagnostics is not None:
            out["monitor"] = {"version": self.monitor.version, "drive": self.monitor.drive,
                              "samples_per_prompt": self.monitor.per_prompt}  # fmt: skip
            out["diagnostics"] = self.diagnosed
        if self.schedule.policy == "k_stopped":
            out["levels"] = {"planned": list(self.schedule.levels), "reached": reached,
                             "missed": [level for level in self.schedule.levels
                                        if level not in self.levels.reached]}  # fmt: skip
            if self.failed_crossing is not None:
                out["levels"]["failed_crossing"] = self.failed_crossing
        return out


def _publish(path: Path, record: dict[str, Any]) -> None:
    """Write a record atomically: a reader sees the whole record or none."""
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("w") as f:
        f.write(json.dumps(record, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, path)


def _norm(tensors: Sequence[torch.Tensor | None]) -> float:
    present = [t.detach() for t in tensors if t is not None]
    if not present:
        return 0.0
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(present))).item()
