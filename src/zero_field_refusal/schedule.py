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

**Update scalars** (amendment 4, section 4; ``tracking.update_scalars``, off in ``base`` and on for
the regular study-2-short runs): after every update of an Adam arm, a read-only hook
(``UpdateScalars``) writes a row to ``update_scalars.jsonl``. It changes no numerics, so it is
not part of the config hash; a test pins the parameters bit for bit with it on and off.

**Endpoint checkpoint and provenance hashes** (docs/study3-preregistration.md, "Numerical
qualification", "The execution path: harness and bridge"; ``tracking.endpoint_checkpoint`` and
``tracking.provenance_hashes``, both off in ``base``, neither in the config hash). With
``provenance_hashes`` the run records, before its optimizer is built,
``inputs.theta0_trainable_sha256`` (θ₀'s trainable tensors) and, for an arm that reads v₀,
``inputs.v0_sha256`` (``tensor_hash``, the harness's own function). With
``endpoint_checkpoint`` set to a directory, a run that takes all ``train.steps`` updates saves the
trainable parameters to ``<dir>/<run_id>.pt`` right after the last update and before
``_Loop.after_step`` (so before the step-25 evaluation), through a hidden ``.partial`` file and
``os.replace``; an interrupted save removes its partial file, leaves no file at the final path and
fails the run. The record ``{path, sha256, step, names_sha256}`` goes to ``summary.json`` and
``meta.json`` (``inputs.endpoint_checkpoint``) at once; a run that stops before its last update,
or on a non-finite weight, writes no file and records ``None``. The file's existence is not the
run's success: the later evaluation and the terminal record decide that. The save only reads the
parameters (host copies) and draws no random numbers. A relative directory is read against the
working directory; the strings "None" and "null" are refused (``_require_checkpoint_dir``).
"""

from __future__ import annotations

import hashlib
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
from .optim import ADAM_MODES, Adam, build_optimizer
from .panels import CacheMismatch, Monitor, ScoringDiagnostics, cache_check
from .prechecks import check_run_chi_square, check_teacher_gap, require_path_check
from .runtime import Environment, ReplayedSoftTargets, StreamPrefix, extend_stream
from .seeding import derive_seed, seed_everything
from .teacher import Sample
from .tensor_hash import file_sha256, tensors_sha256
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
DEPARTURE_STEPS = (5, 10, 25)  # the pre-warmed arm's denominator departure (amendment 4)
SCALAR_CHUNK = 1 << 20  # coordinates per piece of the update scalars' reductions


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


class UpdateScalars:
    """The update scalars of an Adam arm (amendment 4, section 4), read after each update from
    the optimizer state and from the step's gradient, which ``Adam.step`` leaves in ``.grad``.

    Per step t (the updates taken), of the algebraic step u_t = (lr / (1 − β₁ᵗ))·m_t / d_t, with
    d_t the denominator the update divided by (``Adam.denominator``: √v_t / √(1 − β₂ᵗ) + ε
    adaptive, √v₀ + ε frozen, √v_t + ε pre-warmed):

    - ``update_norm``: ‖u_t‖;
    - ``cos_update_grad``: the cosine between u_t and the step's gradient g_t.

    A parameter without a gradient is not updated (``Adam.step`` skips it) and adds nothing.

    Pre-warmed only, at the steps ``departure_at``: ``d_departure``, the departure of d_t from
    d₀ = √v₀ + ε (the denominator at v₀, before the first update) over all trainable coordinates:
    ``norm_ratio`` ‖d_t − d₀‖ / ‖d₀‖, and the ``median`` and ``p90`` of |d_t / d₀ − 1|, read on a
    log histogram (``_LogHistogram``: at most ≈ 1.16 % above the exact value, never below).

    It only reads. No tensor of the loop is written, and nothing persists on the GPU: v₀ is
    copied once, before the first update, into pinned host memory (plain host memory on the CPU)
    and read back piece by piece. Every reduction is in float64, per parameter and in pieces of
    ``chunk`` coordinates whose temporaries are freed before the next piece.
    """

    def __init__(
        self,
        optimizer: Adam,
        *,
        departure_at: Sequence[int] = DEPARTURE_STEPS,
        chunk: int = SCALAR_CHUNK,
    ) -> None:
        self.optimizer, self.chunk = optimizer, chunk
        self.params = [p for group in optimizer.param_groups for p in group["params"]]
        prewarmed = optimizer.second_moment == "prewarmed"
        self.departure_at = tuple(departure_at) if prewarmed else ()
        self.v0: list[torch.Tensor] = []  # pre-warmed only
        for p in self.params if prewarmed else []:
            state = optimizer.state[p]
            if "step" in state:
                raise ValueError("The update scalars copy v₀ before the first update")
            self._check(p)
            v = state["exp_avg_sq"]
            host = torch.empty(v.shape, dtype=v.dtype, pin_memory=v.is_cuda)
            self.v0.append(host.copy_(v))

    def _check(self, p: torch.Tensor) -> None:
        """What reading in pieces rests on: fp32 moments, as ``Adam.step`` makes them, and a
        contiguous state and gradient, so that a piece is a view and never a copy of the whole."""
        state = self.optimizer.state[p]
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and state[key].dtype != torch.float32:
                raise ValueError(
                    f"The update scalars read fp32 Adam moments; {key} of a "
                    f"parameter of shape {tuple(p.shape)} is {state[key].dtype}"
                )
            if key in state and not state[key].is_contiguous():
                raise ValueError(
                    f"The update scalars read a contiguous Adam state; {key} of a "
                    f"parameter of shape {tuple(p.shape)} is not contiguous"
                )
        if p.grad is not None and not p.grad.is_contiguous():
            raise ValueError(
                f"The update scalars read contiguous gradients; the gradient of a "
                f"parameter of shape {tuple(p.shape)} is not contiguous"
            )

    def _pieces(self, numel: int) -> list[slice]:
        return [slice(start, start + self.chunk) for start in range(0, numel, self.chunk)]

    @torch.no_grad()
    def __call__(self, step: int) -> dict[str, Any]:
        """The row of update ``step``, read after the update and before the gradients are freed."""
        optimizer, sums = self.optimizer, None  # Σ u², Σ u·g, Σ g²
        for p in self.params:
            if p.grad is None:
                continue
            self._check(p)
            if sums is None:
                sums = torch.zeros(3, dtype=torch.float64, device=p.device)
            scale = optimizer.step_size(p)
            m, g = optimizer.state[p]["exp_avg"].reshape(-1), p.grad.reshape(-1)
            for piece in self._pieces(m.numel()):
                # Always a copy: the in-place operations below never reach the state.
                u = m[piece].to(torch.float64, copy=True)
                u.div_(optimizer.denominator(p, piece)).mul_(scale)
                grad = g[piece].to(torch.float64, copy=True)
                sums[0] += u.square().sum()
                sums[1] += (u * grad).sum()
                sums[2] += grad.square().sum()
                del u, grad
        uu, ug, gg = (0.0, math.nan, 0.0) if sums is None else sums.tolist()
        row = {"step": step, "update_norm": math.sqrt(uu),
               "cos_update_grad": ug / math.sqrt(uu * gg) if uu * gg > 0 else math.nan}  # fmt: skip
        if step in self.departure_at:
            row["d_departure"] = self.departure()
        return row

    @torch.no_grad()
    def departure(self) -> dict[str, float]:
        """``d_departure`` at the current state; before the first update it is 0."""
        if not self.v0:
            raise ValueError("The departure is the pre-warmed arm's")
        optimizer, sums, histogram = self.optimizer, None, None  # sums: Σ (d_t − d₀)², Σ d₀²
        for p, v0 in zip(self.params, self.v0, strict=True):
            self._check(p)
            if sums is None:
                sums = torch.zeros(2, dtype=torch.float64, device=p.device)
                histogram = _LogHistogram(p.device)
            v0 = v0.reshape(-1)
            for piece in self._pieces(v0.numel()):
                start = v0[piece].to(p.device, non_blocking=True)  # from pinned memory
                d0 = optimizer.denominator(p, piece, second_moment=start)
                d0 = d0.to(torch.float64, copy=True)
                del start
                d = optimizer.denominator(p, piece).to(torch.float64, copy=True)
                d.sub_(d0)  # d_t − d₀
                sums[0] += d.square().sum()
                sums[1] += d0.square().sum()
                d.div_(d0).abs_().nan_to_num_(nan=math.inf, posinf=math.inf)  # |d_t / d₀ − 1|
                del d0
                histogram.add(d)
                del d
        diff, base = sums.tolist()
        return {"norm_ratio": math.sqrt(diff / base) if base > 0 else math.nan,
                **histogram.quantiles()}  # fmt: skip


class _LogHistogram:
    """Counts of log10 of non-negative values (+∞ allowed), on the values' device, in 0.005-decade
    bins over [1e-45, 1e10) (the grid of the FA diagnostic's ``LogHistogram``), zeros apart and
    the values beyond the grid in its end bins. A q-quantile, the value of rank ⌈qN⌉, is read at
    the upper edge of its bin, clamped to the extremes seen: for a value on the grid, at most
    10^0.005 − 1 ≈ 1.16 % above the exact one and not below it. The counts are integers, exact,
    and deterministic (``bincount``; ``histc`` has no deterministic CUDA kernel)."""

    LOW, HIGH, BINS = -45.0, 10.0, 11000
    QUANTILES = (("median", 0.5), ("p90", 0.9))

    def __init__(self, device: torch.device) -> None:
        self.counts = torch.zeros(self.BINS, dtype=torch.int64, device=device)
        self.total = self.zeros = 0
        self.low = torch.full((), math.inf, dtype=torch.float64, device=device)
        self.high = torch.zeros((), dtype=torch.float64, device=device)

    def add(self, values: torch.Tensor) -> None:
        values = values.reshape(-1)
        positive = values[values > 0]
        self.total += values.numel()
        self.zeros += values.numel() - positive.numel()
        if positive.numel():
            self.low = torch.minimum(self.low, positive.min())
            self.high = torch.maximum(self.high, positive.max())
            scale = self.BINS / (self.HIGH - self.LOW)
            bins = positive.log10_().clamp_(self.LOW, self.HIGH).sub_(self.LOW).mul_(scale)
            bins = bins.long().clamp_(0, self.BINS - 1)  # the floor: non-negative
            self.counts += torch.bincount(bins, minlength=self.BINS)

    def quantiles(self) -> dict[str, float]:
        cumulative = (self.zeros + torch.cumsum(self.counts.cpu(), 0)).double()
        low, high = self.low.item(), self.high.item()
        width = (self.HIGH - self.LOW) / self.BINS
        out = {}
        for name, q in self.QUANTILES:
            if q * self.total <= self.zeros:
                out[name] = 0.0
                continue
            rank = torch.tensor(q * self.total, dtype=torch.float64)
            index = min(int(torch.searchsorted(cumulative, rank)), self.BINS - 1)
            out[name] = min(max(10 ** (self.LOW + (index + 1) * width), low), high)
        return out


def run_scheduled(
    cfg: DictConfig,
    env: Environment,
    *,
    run_id: str | None = None,
    numerics_setup: dict[str, Any] | None = None,
) -> Path:
    """One study-2 run under ``cfg.schedule``; returns the run directory.

    ``numerics_setup`` is the record of ``runtime.configure_numerics`` from the entry point
    (scripts/train.py), kept in ``meta.json`` under ``inputs.numerics_setup``; callers that set
    no flags pass nothing and record nothing.
    """
    if cfg.run_type != "train":
        raise ValueError("run_scheduled needs run_type=train")
    schedule = Schedule.from_config(cfg)
    _require_checkpoint_dir(cfg.tracking.get("endpoint_checkpoint", None))
    seed_everything(derive_seed(cfg.seed, "torch"))
    started = time.perf_counter()
    with Run(cfg, run_id=run_id) as run:
        record = require_path_check(env.data_root, cfg, env.device)  # before anything costly
        run.record_inputs({"path_check": {"run_id": record["run_id"],
                                          "fingerprint": record["fingerprint"]},
                           "prompt_manifest_sha256": env.prompt_manifest_sha256})  # fmt: skip
        if numerics_setup is not None:
            run.record_inputs({"numerics_setup": numerics_setup})
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
        provenance = bool(cfg.tracking.get("provenance_hashes", False))
        if provenance:
            run.record_inputs({"theta0_trainable_sha256": tensors_sha256(
                [(name, theta0[name]) for name, _ in trainable])})  # fmt: skip
        second_moment = None
        if ADAM_MODES.get(cfg.optimizer.name, "adaptive") != "adaptive":
            second_moment = _theta0_second_moment(cfg, env, student, params, run)
            if provenance:  # before the optimizer takes it
                named = zip((name for name, _ in trainable), second_moment, strict=True)
                run.record_inputs({"v0_sha256": tensors_sha256(list(named))})
        optimizer = build_optimizer(cfg.optimizer, params, second_moment=second_moment)
        del second_moment
        scalars = None  # read-only; the pre-warmed arm's v₀ goes to host memory now
        if cfg.tracking.get("update_scalars", False) and isinstance(optimizer, Adam):
            scalars = UpdateScalars(optimizer)

        def release() -> None:  # the next step's zero_grad(set_to_none) would free them anyway
            for p in params:
                p.grad = None

        loop = _Loop(schedule, run, evaluate, monitor, diagnostics, release)
        reason = None
        if diagnostics is not None:  # the caches against uncached readings, at θ₀
            reason = loop.check_caches(student, scoring=diagnostics(student, cached=False))
        size = cfg.train.batch_size
        checkpoint_dir = cfg.tracking.get("endpoint_checkpoint", None)
        checkpoint = None if checkpoint_dir is None else (
            Path(checkpoint_dir).expanduser().resolve() / f"{run.run_id}.pt")  # fmt: skip
        recorded = False
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
            if scalars is not None:  # the step's gradient is still in .grad
                _append_jsonl(run.path("update_scalars.jsonl"), [scalars(loop.steps)])
            if not math.isfinite(_norm(params)):
                reason = "nonfinite_weight"
                loop.rejected = {"step": step, "loss": stats["loss"],
                                 "grad_norm": stats["grad_norm"], "applied": True}  # fmt: skip
                break
            if checkpoint is not None and loop.steps == cfg.train.steps:
                saved = {"endpoint_checkpoint": save_endpoint(trainable, checkpoint, loop.steps)}
                run.summarize(saved)  # the record, not the outcome: the terminal record decides
                run.record_inputs(saved)
                recorded = True
            reason = loop.after_step(student, lambda: projection_stats(trainable, theta0,
                                                                       directions))  # fmt: skip
            if reason:
                break
        else:
            if reason is None:  # every step taken (a θ₀ verdict takes none)
                reason = "t_max" if schedule.policy == "k_stopped" else "steps"
        for p in params:
            p.grad = None
        if checkpoint is not None and not recorded:  # stopped before its last update
            run.summarize({"endpoint_checkpoint": None})
            run.record_inputs({"endpoint_checkpoint": None})
        terminal = loop.terminal(reason, size)
        terminal["stream_extension"] = {"seconds": feed.extension_seconds,
                                        "chunks": feed.extended_chunks}  # fmt: skip
        terminal["seconds"] = time.perf_counter() - started  # the run's time, up to its outcome
        run.record_inputs({"streams": {"training": feed.identity(loop.steps * size)}})
        publish(run.path("terminal.json"), terminal)  # the run's outcome, whatever follows
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
            del optimizer, loop, release, evaluate, monitor, diagnostics, feed, replayer, scalars
            del directions, student, trainable, params
            record_endpoint_rayleigh(cfg, env, run, reference, delta, *panel)
        return run.dir


def _require_checkpoint_dir(directory: Any) -> None:
    """``tracking.endpoint_checkpoint`` is null (off) or a directory. A string that spells null
    (``key=None`` on a command line loads as the text "None") would save into a directory of that
    name under the working directory and leave the switch on: refused. A relative path is read
    against the working directory."""
    if directory is None:
        return
    if not isinstance(directory, str) or directory.strip().lower() in ("", "none", "null", "~"):
        raise ValueError(
            f"tracking.endpoint_checkpoint is {directory!r}: give a directory, or null (YAML) to "
            "switch the checkpoint off"
        )


def save_endpoint(
    trainable: Sequence[tuple[str, torch.nn.Parameter]], path: Path, step: int
) -> dict[str, Any]:
    """Save ``{name: tensor.detach().cpu()}`` as ``fa_control_diagnostic.save_trainable`` does,
    atomically; returns the checkpoint record. A failed write removes its partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        torch.save({name: p.detach().cpu() for name, p in trainable}, partial)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    names = json.dumps([name for name, _ in trainable])
    return {"path": str(path), "sha256": file_sha256(path), "step": step,
            "names_sha256": hashlib.sha256(names.encode()).hexdigest()}  # fmt: skip


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


def publish(path: Path, record: dict[str, Any]) -> None:
    """Write a record atomically: a reader sees the whole record or none."""
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("w") as f:
        f.write(json.dumps(record, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, path)


_publish = publish  # the former private name


def _norm(tensors: Sequence[torch.Tensor | None]) -> float:
    present = [t.detach() for t in tensors if t is not None]
    if not present:
        return 0.0
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(present))).item()
