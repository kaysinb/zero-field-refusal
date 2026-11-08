"""Study 5's options of the scheduled run (docs/study5-preregistration.md, "Descriptive
measurements": the crossed controls). The line scans are runs of their own (``line_scan``).

Each key is absent from the shared configs, so that no existing config changes its hash, and an
absent key (or null) leaves the run exactly as it was. ``protocol`` lets only the study-5 roles
carry them (``study5.role``).

- ``train.order_seed``: the run trains on the same ``train.steps × train.batch_size`` responses
  of its seed's appendable stream in a fixed permutation, ``training_order``: the permutation of
  0 … N·B − 1 drawn by ``numpy.random.default_rng(derive_seed(order_seed, "study5", "order"))``.
  Batch s holds the responses at positions sB … (s+1)B − 1 of the permutation (``batch_samples``,
  which ``schedule.BatchFeed`` and ``linear_response.feed_batches`` both call). The run records
  ``inputs.training_order`` (``order_record``): the order seed, the number of responses and the
  sha256 of the permutation.
- ``optimizer.preconditioner.source_seed``: v₀ is estimated from that seed's preconditioner stream
  (``preconditioner_seed``, read by ``train._theta0_second_moment``); the stream must already be
  in the data cache. The run records ``inputs.preconditioner_source_seed`` beside the stream's
  identity (``inputs.streams.preconditioner``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .runtime import StreamPrefix
from .seeding import derive_seed
from .teacher import Sample

ORDER_KEYS = ("study5", "order")  # derive_seed(order_seed, *ORDER_KEYS)


def order_seed(cfg: DictConfig) -> int | None:
    return OmegaConf.select(cfg, "train.order_seed", default=None)


def training_order(cfg: DictConfig) -> list[int] | None:
    """The permutation of the run's N·B responses under ``train.order_seed``; None without one
    (stream order)."""
    seed = order_seed(cfg)
    if seed is None:
        return None
    responses = int(cfg.train.steps) * int(cfg.train.batch_size)
    rng = np.random.default_rng(derive_seed(seed, *ORDER_KEYS))
    return [int(i) for i in rng.permutation(responses)]


def order_record(cfg: DictConfig, order: Sequence[int]) -> dict[str, Any]:
    """``inputs.training_order``: the order seed, the responses and the permutation's sha256."""
    digest = hashlib.sha256(json.dumps(list(order)).encode()).hexdigest()
    return {"order_seed": order_seed(cfg), "responses": len(order), "permutation_sha256": digest}


def batch_samples(
    prefix: StreamPrefix, order: Sequence[int] | None, step: int, size: int
) -> list[Sample]:
    """Batch ``step`` of batch size ``size``: responses sB … (s+1)B − 1 of the stream, or, under
    ``order``, the responses at those positions of the permutation, each chunk read once."""
    if order is None:
        return prefix.take(step * size, (step + 1) * size)
    indices = list(order[step * size : (step + 1) * size])
    out: list[Sample | None] = [None] * len(indices)
    for chunk in sorted({i // prefix.size for i in indices}):
        base = chunk * prefix.size
        samples = prefix.take(base, base + prefix.size)
        for position, i in enumerate(indices):
            if i // prefix.size == chunk:
                out[position] = samples[i - base]
    return out  # type: ignore[return-value]


def preconditioner_source(cfg: DictConfig) -> int | None:
    return OmegaConf.select(cfg, "optimizer.preconditioner.source_seed", default=None)


def preconditioner_seed(cfg: DictConfig) -> int:
    """The seed whose preconditioner stream gives v₀: ``source_seed``, or the run's own."""
    source = preconditioner_source(cfg)
    return cfg.seed if source is None else source
