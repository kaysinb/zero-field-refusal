"""The monitor panel of study 2 and its agreement with the scoring panel.

Two panels measure K_benign (docs/study2-preregistration.md, "Panels"):
- the **scoring panel** (``train.Evaluator``): 100 benign prompts × 4 θ₀ responses, with K_harmful
  and utility, at every full evaluation; the curves use its K;
- the **monitor panel** (``Monitor``): 25 further benign prompts × 2 θ₀ responses, disjoint from
  the scoring panel in prompts (``alpaca_monitor``) and in randomness (the ``kl_monitor`` stream
  has its own seeds), read every few steps during training to decide when to evaluate fully.

Stage A checks that the monitor tracks the scoring panel where it matters (``monitor_agreement``):
for K_benign ≥ 4, the median relative discrepancy must be at most 5 % and the 95th percentile at
most 15 %; otherwise the monitor is enlarged before fixation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from omegaconf import DictConfig
from transformers import PreTrainedModel

from .evaluation import sequence_kl
from .runtime import Environment, load_stream_for
from .tracking import Run

AGREEMENT_FROM_K = 4.0
AGREEMENT_MEDIAN = 0.05
AGREEMENT_P95 = 0.15


class Monitor:
    """K_benign on the monitor panel: the mean over its responses of the summed per-position
    KL(θ₀ ‖ θ), as the scoring panel's K, against the run's frozen θ₀ copy."""

    def __init__(self, cfg: DictConfig, env: Environment, run: Run, reference: PreTrainedModel):
        self.env, self.reference = env, reference
        self.micro_batch_size = cfg.execution.micro_batch_size
        self.samples, manifest = load_stream_for(cfg, env, "kl_monitor", None)
        run.summarize({"stream/kl_monitor": manifest["sha256"]})
        identity = {key: manifest[key] for key in ("path", "sha256", "count")}
        run.record_inputs({"streams": {"kl_monitor": identity}})

    def __call__(self, model: PreTrainedModel) -> float:
        kl = sequence_kl(model, self.reference, self.samples, pad_id=self.env.pad_id,
                         micro_batch_size=self.micro_batch_size)  # fmt: skip
        return kl.mean().item()


def monitor_agreement(
    monitor: Sequence[float], scoring: Sequence[float], *, from_k: float = AGREEMENT_FROM_K
) -> dict[str, Any]:
    """Relative discrepancy |K_monitor − K_scoring| / K_scoring of paired readings of the same
    states, over those with K_scoring ≥ ``from_k``: its median and 95th percentile (linear
    interpolation), and whether both meet the preregistered bounds."""
    pairs = [(m, s) for m, s in zip(monitor, scoring, strict=True) if s >= from_k]
    if not pairs:
        return {"pairs": 0, "median": None, "p95": None, "passes": None}
    relative = np.array([abs(m - s) / s for m, s in pairs])
    median, p95 = float(np.median(relative)), float(np.percentile(relative, 95))
    return {"pairs": len(pairs), "median": median, "p95": p95,
            "passes": median <= AGREEMENT_MEDIAN and p95 <= AGREEMENT_P95}  # fmt: skip
