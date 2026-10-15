"""The monitor panel of study 2 and its agreement with the scoring panel.

Two panels measure K_benign (docs/study2-preregistration.md, "Panels"):
- the **scoring panel** (``train.Evaluator``): 100 benign prompts × 4 θ₀ responses, with K_harmful
  and utility, at every full evaluation; the curves use its K;
- the **monitor panel** (``Monitor``), read every few steps during training to decide when to
  evaluate fully. Version 1: 25 further benign prompts × 2 θ₀ responses, disjoint from the
  scoring panel in prompts (``alpaca_monitor``) and in randomness. Version 2 (amendment 1): the
  scoring panel's own 100 prompts with fresh θ₀ responses of their own seeds (``kl_monitor``
  v2, ``samples_per_prompt`` per prompt); the first ``drive`` responses of each prompt are read
  every ``every`` steps, the others only at diagnostic states and full evaluations (the shadow
  panel), and θ₀'s logits are cached (``evaluation.ReferenceLogits``).

Stage A checks that the monitor tracks the scoring panel where it matters (``monitor_agreement``):
for K_benign ≥ 4, the median relative discrepancy must be at most 5 % and the 95th percentile at
most 15 %, and no reading may be non-finite. Under version 2 the scoring panel's K is also read
alone, every ``eval.diagnostics.every`` steps (``ScoringDiagnostics``), so that the agreement is
measured on many states of each run; every cached reading is checked against an uncached one at
θ₀ and at each full evaluation (``cache_check``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel

from .evaluation import ReferenceLogits, sequence_kl
from .runtime import Environment, load_stream_for
from .teacher import Sample, monitor_version
from .tracking import Run

AGREEMENT_FROM_K = 4.0
AGREEMENT_MEDIAN = 0.05
AGREEMENT_P95 = 0.15


class CacheMismatch(RuntimeError):
    """A cached reading differs from the uncached one: the cache would change K."""


def _cache(
    reference: PreTrainedModel,
    samples: Sequence[Sample],
    micro_batch_size: int,
    pad_id: int,
    where: str,
) -> ReferenceLogits | None:
    """θ₀'s logits for ``samples``, on the model's device (``device``) or in host memory
    (``host``); ``none`` caches nothing."""
    if where == "none" or not samples:
        return None
    device = next(reference.parameters()).device if where == "device" else torch.device("cpu")
    return ReferenceLogits(reference, samples, pad_id=pad_id, micro_batch_size=micro_batch_size,
                           device=device)  # fmt: skip


def cache_check(cached: torch.Tensor, uncached: torch.Tensor) -> dict[str, Any]:
    """Per-response K of the same state with and without the cache: equal bit for bit?"""
    return {
        "responses": int(cached.numel()),
        "bitwise_equal": bool(torch.equal(cached, uncached)),
        "max_abs_diff": float((cached - uncached).abs().max()) if cached.numel() else 0.0,
    }


class Monitor:
    """K_benign on the monitor panel: the mean over its responses of the summed per-position
    KL(θ₀ ‖ θ), as the scoring panel's K, against the run's frozen θ₀ copy.

    ``monitor(model)`` is the operational reading (version 1: all responses; version 2: the first
    ``drive`` of each prompt); ``shadow(model)`` adds the other responses of each prompt, so that a
    larger panel is measured on the same states; ``check(model)`` compares cached readings with
    uncached ones.
    """

    def __init__(self, cfg: DictConfig, env: Environment, run: Run, reference: PreTrainedModel):
        settings = cfg.eval.monitor
        self.env, self.reference = env, reference
        self.micro_batch_size = cfg.execution.micro_batch_size
        self.version = monitor_version(cfg)
        samples, manifest = load_stream_for(cfg, env, "kl_monitor", None)
        run.summarize({"stream/kl_monitor": manifest["sha256"]})
        identity = {key: manifest[key] for key in ("path", "sha256", "count")}
        run.record_inputs({"streams": {"kl_monitor": identity}})
        per_prompt = int(settings.samples_per_prompt)
        self.drive = int(settings.drive) if self.version > 1 else per_prompt
        self.per_prompt = per_prompt
        # The stream holds each prompt's responses together (teacher.stream_prompt_indices).
        self.primary = [s for i, s in enumerate(samples) if i % per_prompt < self.drive]
        self.secondary = [s for i, s in enumerate(samples) if i % per_prompt >= self.drive]
        self.caches: dict[str, ReferenceLogits | None] = {"primary": None, "secondary": None}
        if self.version > 1:
            for name, part in (("primary", self.primary), ("secondary", self.secondary)):
                self.caches[name] = _cache(reference, part, self.micro_batch_size, env.pad_id,
                                           settings.cache[name])  # fmt: skip
            run.summarize({"monitor": {
                "version": self.version, "drive": self.drive, "samples_per_prompt": per_prompt,
                "cache_bytes": {k: c.bytes if c else 0 for k, c in self.caches.items()},
            }})  # fmt: skip
        self.last: torch.Tensor | None = None  # the last operational reading, per response

    def _values(self, model: PreTrainedModel, part: str, *, cached: bool = True) -> torch.Tensor:
        samples = self.primary if part == "primary" else self.secondary
        return sequence_kl(model, self.reference, samples, pad_id=self.env.pad_id,
                           micro_batch_size=self.micro_batch_size,
                           cache=self.caches[part] if cached else None)  # fmt: skip

    def __call__(self, model: PreTrainedModel) -> float:
        self.last = self._values(model, "primary")
        return self.last.mean().item()

    def shadow(self, model: PreTrainedModel, primary: torch.Tensor | None) -> dict[str, Any]:
        """The full panel (every response of each prompt) at this state; ``primary`` is the
        operational reading of the same state, if one was just taken."""
        primary = self._values(model, "primary") if primary is None else primary
        secondary = self._values(model, "secondary") if self.secondary else primary[:0]
        both = torch.cat([primary, secondary])
        return {"K_primary": primary.mean().item(), "K_full": both.mean().item(),
                "primary": primary.tolist(), "secondary": secondary.tolist()}  # fmt: skip

    def check(self, model: PreTrainedModel) -> dict[str, Any]:
        """Each cached part read with and without its cache at this state."""
        out = {}
        for part, cache in self.caches.items():
            if cache is not None:
                out[part] = cache_check(self._values(model, part),
                                        self._values(model, part, cached=False))  # fmt: skip
        return out


class ScoringDiagnostics:
    """The scoring panel's K_benign alone (amendment 1): the Evaluator's samples and micro-batches,
    with θ₀'s logits cached, read every ``eval.diagnostics.every`` steps; no harm, no utility."""

    def __init__(self, cfg: DictConfig, env: Environment, run: Run, reference: PreTrainedModel):
        self.env, self.reference = env, reference
        self.micro_batch_size = cfg.execution.micro_batch_size
        self.every = int(cfg.eval.diagnostics.every)
        samples, _ = load_stream_for(cfg, env, "kl_benign", None)
        limit, per_prompt = cfg.eval.kl.limit, cfg.eval.kl.samples_per_prompt
        self.samples = samples[: limit * per_prompt] if limit else samples  # as the Evaluator
        self.cache = _cache(reference, self.samples, self.micro_batch_size, env.pad_id,
                            cfg.eval.diagnostics.cache)  # fmt: skip
        cached = self.cache.bytes if self.cache else 0
        run.summarize({"diagnostics": {"every": self.every, "cache_bytes": cached}})

    def __call__(self, model: PreTrainedModel, *, cached: bool = True) -> torch.Tensor:
        return sequence_kl(model, self.reference, self.samples, pad_id=self.env.pad_id,
                           micro_batch_size=self.micro_batch_size,
                           cache=self.cache if cached else None)  # fmt: skip


def monitor_agreement(
    monitor: Sequence[float], scoring: Sequence[float], *, from_k: float = AGREEMENT_FROM_K
) -> dict[str, Any]:
    """Relative discrepancy |K_monitor − K_scoring| / K_scoring of paired readings of the same
    states, over those with K_scoring ≥ ``from_k``: its median and 95th percentile (linear
    interpolation), and whether both meet the preregistered bounds.

    Every pair must be finite, whatever its K: a non-finite reading is a failed measurement,
    and dropping it, or letting it fall beyond the 95th percentile, could pass a bad monitor.
    Any invalid pair fails the check, with their number; no pair at or above ``from_k`` leaves
    it undetermined (``passes`` None).
    """
    invalid = sum(
        not (math.isfinite(m) and math.isfinite(s))
        for m, s in zip(monitor, scoring, strict=True)
    )  # fmt: skip
    pairs = [(m, s) for m, s in zip(monitor, scoring, strict=True)
             if math.isfinite(m) and math.isfinite(s) and s >= from_k]  # fmt: skip
    out: dict[str, Any] = {"pairs": len(pairs), "invalid": invalid, "median": None, "p95": None}
    if pairs:
        relative = np.array([abs(m - s) / s for m, s in pairs])
        out["median"] = float(np.median(relative))
        out["p95"] = float(np.percentile(relative, 95))
    if invalid:
        out["passes"] = False
    elif not pairs:
        out["passes"] = None
    else:
        out["passes"] = out["median"] <= AGREEMENT_MEDIAN and out["p95"] <= AGREEMENT_P95
    return out
