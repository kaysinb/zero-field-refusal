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
- ``train.order_edit`` (extension 2): a small edit of the stream order, a mapping with ``kind``:
  - ``swap_batches`` with ``step`` t: the batches at steps t and t + 1 exchanged whole;
  - ``swap_first`` with ``step`` t: the first responses of the batches at steps t and t + 1
    exchanged, everything else in place;
  - ``shuffle`` with ``first`` a, ``last`` b and ``draw`` k: the batches at steps a … b permuted
    as whole batches by ``default_rng(derive_seed(seed, "study5", "shuffle", a, b, k))``, the
    run's own seed.

  Steps count from 1 (t < ``train.steps``, 1 ≤ a < b ≤ ``train.steps``). The edit's permutation
  goes the same way as an order seed's (``batch_samples``), and ``order_record`` records the edit
  beside the permutation's sha256. ``protocol`` lets a control set exactly one of the three keys.
- ``train.rate_schedule`` (the dose experiment, docs/study5-dose-registration.md): a mapping of
  exactly ``early_steps`` k (an integer, 1 ≤ k < ``train.steps``) and ``early_lr`` (a positive
  finite float): updates 1 … k take the rate ``early_lr``, updates k + 1 … ``train.steps`` the
  rate ``optimizer.lr`` (``step_rates``). ``schedule.run_scheduled`` sets every parameter
  group's ``lr`` to the update's rate just before that update, so Adam's bias correction and
  the read accessors (``optim.Adam.step_size``) see the rate the update used; the first moment
  carries over unchanged. The run records ``inputs.rate_schedule`` (``schedule_record``).
  ``protocol`` lets only the ``dose`` role carry it, and that role carries none of the three
  keys above. With ``early_lr`` equal to ``optimizer.lr`` the run is the plain run.
"""

from __future__ import annotations

import hashlib
import json
import math
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


EDIT_KINDS = ("swap_batches", "swap_first", "shuffle")


def order_edit(cfg: DictConfig) -> dict[str, Any] | None:
    edit = OmegaConf.select(cfg, "train.order_edit", default=None)
    if edit is None:
        return None
    return OmegaConf.to_container(edit, resolve=True) if not isinstance(edit, dict) else dict(edit)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def edit_problems(edit: Any, steps: int) -> list[str]:
    """Why ``train.order_edit`` is not a valid edit for a run of ``steps`` steps (empty: valid)."""
    if not isinstance(edit, dict):
        return [f"train.order_edit: expected a mapping, got {edit!r}"]
    kind = edit.get("kind")
    expected = {"swap_batches": {"kind", "step"}, "swap_first": {"kind", "step"},
                "shuffle": {"kind", "first", "last", "draw"}}.get(kind)  # fmt: skip
    if expected is None:
        return [f"train.order_edit.kind: expected one of {EDIT_KINDS}, got {kind!r}"]
    if set(edit) != expected:
        return [f"train.order_edit: a {kind} edit takes exactly {sorted(expected)}, "
                f"got {sorted(edit)}"]  # fmt: skip
    if not all(_is_int(edit[key]) for key in expected - {"kind"}):
        return [f"train.order_edit: integer values expected, got {edit!r}"]
    if kind in ("swap_batches", "swap_first") and not 1 <= edit["step"] < steps:
        return [f"train.order_edit.step: expected 1 … {steps - 1}, got {edit['step']}"]
    if kind == "shuffle" and not (1 <= edit["first"] < edit["last"] <= steps and edit["draw"] >= 0):
        return [f"train.order_edit: expected 1 ≤ first < last ≤ {steps} and draw ≥ 0, got {edit}"]
    return []


def edited_order(cfg: DictConfig, edit: dict[str, Any]) -> list[int]:
    """The stream order of the run's N·B responses with ``edit`` applied."""
    steps, size = int(cfg.train.steps), int(cfg.train.batch_size)
    problems = edit_problems(edit, steps)
    if problems:
        raise ValueError("; ".join(problems))
    order = list(range(steps * size))
    batches = [order[s * size : (s + 1) * size] for s in range(steps)]
    kind = edit["kind"]
    if kind == "swap_batches":
        t = edit["step"] - 1
        batches[t], batches[t + 1] = batches[t + 1], batches[t]
    elif kind == "swap_first":
        t = edit["step"] - 1
        batches[t][0], batches[t + 1][0] = batches[t + 1][0], batches[t][0]
    else:
        a, b = edit["first"] - 1, edit["last"]
        rng = np.random.default_rng(derive_seed(cfg.seed, "study5", "shuffle", edit["first"],
                                                edit["last"], edit["draw"]))  # fmt: skip
        block = batches[a:b]
        batches[a:b] = [block[int(i)] for i in rng.permutation(len(block))]
    return [i for batch in batches for i in batch]


def training_order(cfg: DictConfig) -> list[int] | None:
    """The permutation of the run's N·B responses under ``train.order_seed`` or
    ``train.order_edit``; None without either (stream order)."""
    edit = order_edit(cfg)
    if edit is not None:
        return edited_order(cfg, edit)
    seed = order_seed(cfg)
    if seed is None:
        return None
    responses = int(cfg.train.steps) * int(cfg.train.batch_size)
    rng = np.random.default_rng(derive_seed(seed, *ORDER_KEYS))
    return [int(i) for i in rng.permutation(responses)]


def order_record(cfg: DictConfig, order: Sequence[int]) -> dict[str, Any]:
    """``inputs.training_order``: the order seed, the responses and the permutation's sha256, and
    the order edit when there is one (absent otherwise, so the registered records keep their
    fields)."""
    digest = hashlib.sha256(json.dumps(list(order)).encode()).hexdigest()
    record = {"order_seed": order_seed(cfg), "responses": len(order), "permutation_sha256": digest}
    edit = order_edit(cfg)
    if edit is not None:
        record["order_edit"] = edit
    return record


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


SCHEDULE_FIELDS = ("early_lr", "early_steps")


def rate_schedule(cfg: DictConfig) -> dict[str, Any] | None:
    value = OmegaConf.select(cfg, "train.rate_schedule", default=None)
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def schedule_problems(value: Any, steps: int) -> list[str]:
    """Why ``train.rate_schedule`` is not a valid schedule for a run of ``steps`` steps (empty:
    valid)."""
    if not isinstance(value, dict):
        return [f"train.rate_schedule: expected a mapping, got {value!r}"]
    if sorted(value) != list(SCHEDULE_FIELDS):
        return [f"train.rate_schedule: exactly {list(SCHEDULE_FIELDS)}, got {sorted(value)}"]
    problems = []
    k, lr = value["early_steps"], value["early_lr"]
    if not (_is_int(k) and 1 <= k < steps):
        problems.append(f"train.rate_schedule.early_steps: expected 1 … {steps - 1}, got {k!r}")
    if not (isinstance(lr, float) and math.isfinite(lr) and lr > 0):
        problems.append(f"train.rate_schedule.early_lr: expected a positive float, got {lr!r}")
    return problems


def step_rates(cfg: DictConfig) -> list[float] | None:
    """The rate of each update 1 … ``train.steps`` under ``train.rate_schedule`` (None without
    it)."""
    value = rate_schedule(cfg)
    if value is None:
        return None
    steps = int(cfg.train.steps)
    problems = schedule_problems(value, steps)
    if problems:
        raise ValueError("; ".join(problems))
    k, late = int(value["early_steps"]), float(cfg.optimizer.lr)
    return [float(value["early_lr"])] * k + [late] * (steps - k)


def schedule_record(cfg: DictConfig) -> dict[str, Any]:
    """``inputs.rate_schedule``: the early updates, the early and the late rate."""
    value = rate_schedule(cfg)
    if value is None:
        raise ValueError("the run has no train.rate_schedule")
    return {"early_steps": int(value["early_steps"]), "early_lr": float(value["early_lr"]),
            "late_lr": float(cfg.optimizer.lr)}  # fmt: skip


def preconditioner_source(cfg: DictConfig) -> int | None:
    return OmegaConf.select(cfg, "optimizer.preconditioner.source_seed", default=None)


def preconditioner_seed(cfg: DictConfig) -> int:
    """The seed whose preconditioner stream gives v₀: ``source_seed``, or the run's own."""
    source = preconditioner_source(cfg)
    return cfg.seed if source is None else source
