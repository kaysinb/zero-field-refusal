"""Study 5's linear-response predictor (docs/study5-preregistration.md, "The surrogate and the
predictor").

Arm F (``optimizer: adam_frozen``, ``optim.Adam`` in frozen mode, no weight decay) updates
θ_k = θ_{k−1} − η m_k ⊙ D / (1 − β₁ᵏ), with m_k = β₁ m_{k−1} + (1 − β₁) g_k, m_0 = 0 and
D = (√v₀ + ε)⁻¹, v₀ never updated and not bias-corrected. After N updates

    θ_N − θ₀ = −η Σ_t c_t D ⊙ g_t,   c_t = Σ_{k=t..N} (1 − β₁) β₁^{k−t} / (1 − β₁ᵏ),

exactly when g_t does not depend on θ. The predictor is a frozen-gradient approximation, not the
exact linearized training dynamics: it takes every g_t at θ₀ (the gradient of step t's batch,
evaluated at θ₀ and not along the trajectory):
δ̂ = −η Σ_t c_t D ⊙ g_t (``linear_displacement``) and p = ⟨δ̂, ∇r(θ₀)⟩ = −η Σ_t c_t ⟨g_t, u⟩ with
u = D ⊙ ∇r (``predictor``), which needs no stored g_t: one u, the model and one gradient buffer.
A larger p predicts less harm; reading it is the analysis's business.

The inputs come from the F run's own machinery, so that they are the run's:

- v₀ through ``train._theta0_second_moment``, the code path the scheduled run calls: the seed's
  preconditioner stream, its batches, the attention backends of ``runtime.V0_SDPA_BACKENDS``
  (``theta0_second_moment``); its hash as the run records it (``v0_sha256``);
- step t's batch as ``schedule.BatchFeed`` reads it: responses (t−1)B … tB − 1 of the seed's
  appendable training stream (``runtime.StreamPrefix``), or of its permutation under
  ``train.order_seed`` (``study5_options.batch_samples``, the function the run calls)
  (``feed_batches``), with the stream's identity as the run records it;
- g_t from ``train.accumulate_gradient``: the protocol's loss and denominator over the trainable
  set of ``models.freeze`` (``step_gradients``).

Inner products are reduced in float64; the tensors stay fp32.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel

from .optim import ADAM_MODES
from .runtime import Environment, StreamPrefix
from .study5_options import batch_samples, training_order
from .teacher import Sample, stream_layout
from .tensor_hash import tensors_sha256
from .tracking import _merge
from .train import _theta0_second_moment, accumulate_gradient

Named = Mapping[str, torch.Tensor]


def momentum_weights(beta1: float, steps: int) -> list[float]:
    """[c_1, …, c_N], N = ``steps``: c_t = Σ_{k=t..N} (1 − β₁) β₁^{k−t} / (1 − β₁ᵏ)."""
    if steps < 1 or not 0 <= beta1 < 1:
        raise ValueError(f"Need steps >= 1 and 0 <= beta1 < 1, got {steps}, {beta1}")
    return [math.fsum((1 - beta1) * beta1 ** (k - t) / (1 - beta1**k)
                      for k in range(t, steps + 1)) for t in range(1, steps + 1)]  # fmt: skip


def _paired(grads: Iterable[Named], weights: Sequence[float]) -> Iterator[tuple[float, Named]]:
    """(c_t, g_t) one step at a time; exactly one gradient set per weight."""
    it = iter(grads)
    for count, c in enumerate(weights):
        g = next(it, None)
        if g is None:
            raise ValueError(f"{count} gradients for {len(weights)} weights")
        yield c, g
    if next(it, None) is not None:
        extra = len(weights) + 1 + sum(1 for _ in it)
        raise ValueError(f"{extra} gradients for {len(weights)} weights")


def linear_displacement(
    grads: Iterable[Named], D: Named, weights: Sequence[float], lr: float
) -> dict[str, torch.Tensor]:
    """δ̂ = −η Σ_t c_t D ⊙ g_t, by parameter name, in D's dtype; ``grads`` yields one gradient
    set per step, each used before the next is read. Only the sum is taken without autograd, so
    that ``grads`` may compute each g_t by a backward pass (``step_gradients``)."""
    with torch.no_grad():
        delta = {name: torch.zeros_like(d) for name, d in D.items()}
    for c, g in _paired(grads, weights):
        with torch.no_grad():
            for name, d in D.items():
                delta[name].add_(d * g[name], alpha=-lr * c)
    return delta


@torch.no_grad()
def inner(a: Named, b: Named) -> float:
    """⟨a, b⟩ over the names of ``b``, reduced in float64."""
    return math.fsum(torch.dot(a[name].reshape(-1).double(), t.reshape(-1).double()).item()
                     for name, t in b.items())  # fmt: skip


def predictor(
    grads: Iterable[Named], u: Named, weights: Sequence[float], lr: float
) -> dict[str, Any]:
    """p = −η Σ_t c_t ⟨g_t, u⟩ with u = D ⊙ ∇r, and ``per_step``: [⟨g_t, u⟩ …]; ``grads``
    yields one gradient set per step, each used before the next is read."""
    per_step = [inner(g, u) for _, g in _paired(grads, weights)]
    return {"p": -lr * math.fsum(c * x for c, x in zip(weights, per_step, strict=True)),
            "per_step": per_step}  # fmt: skip


# ------------------------------------------------------------------------------- the F run's inputs


def require_frozen_adam(cfg: DictConfig) -> None:
    """The predictor is F's: frozen Adam without weight decay, trained on its seed's appendable
    stream with the plain loss (no control)."""
    problems = []
    if ADAM_MODES.get(cfg.optimizer.name) != "frozen":
        problems.append(f"optimizer {cfg.optimizer.name}, not adam_frozen")
    if cfg.optimizer.get("weight_decay", 0) != 0:
        problems.append(f"weight_decay {cfg.optimizer.weight_decay}, not 0")
    if cfg.control is not None:
        problems.append(f"control {cfg.control}, not the zero field")
    if stream_layout(cfg) != "appendable":
        problems.append("the training stream is not appendable")
    if problems:
        raise ValueError("Not an F run: " + "; ".join(problems))


class _Inputs:
    """Stands in for ``tracking.Run`` where the run's code records what it read: it keeps the
    records instead of writing them."""

    def __init__(self) -> None:
        self.inputs: dict[str, Any] = {}
        self.summary: dict[str, Any] = {}

    def record_inputs(self, inputs: Mapping[str, Any]) -> None:
        _merge(self.inputs, inputs)

    def summarize(self, values: Mapping[str, Any]) -> None:
        self.summary.update(values)


def theta0_second_moment(
    cfg: DictConfig,
    env: Environment,
    model: PreTrainedModel,
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """v₀ in ``trainable``'s order, computed by the run's own ``_theta0_second_moment`` at the
    model's current point, and the preconditioner stream's identity as the run records it. The
    parameters' ``.grad`` is left empty."""
    record = _Inputs()
    v0 = _theta0_second_moment(cfg, env, model, [p for _, p in trainable], record)
    return v0, record.inputs["streams"]["preconditioner"]


def v0_sha256(
    trainable: Sequence[tuple[str, torch.nn.Parameter]], v0: Sequence[torch.Tensor]
) -> str:
    """The hash the scheduled run records as ``inputs.v0_sha256``."""
    return tensors_sha256(list(zip((name for name, _ in trainable), v0, strict=True)))


def feed_batches(cfg: DictConfig, env: Environment) -> tuple[list[list[Sample]], dict[str, Any]]:
    """The run's ``train.steps`` batches as ``schedule.BatchFeed`` reads them (batch s: responses
    sB … (s+1)B − 1 of the seed's appendable stream, or of the permutation of
    ``train.order_seed``: ``study5_options.batch_samples``; the stream must hold them, nothing is
    generated), and the stream's identity as the run records it."""
    prefix, size = StreamPrefix(cfg, env, cfg.seed), int(cfg.train.batch_size)
    steps, order = int(cfg.train.steps), training_order(cfg)
    batches = [batch_samples(prefix, order, s, size) for s in range(steps)]
    return batches, prefix.identity(steps * size)


def step_gradients(
    model: PreTrainedModel,
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    batches: Sequence[Sequence[Sample]],
    cfg: DictConfig,
    env: Environment,
) -> Iterator[dict[str, torch.Tensor]]:
    """Per batch, its gradient at the model's current point by ``train.accumulate_gradient``, by
    parameter name: the parameters' own ``.grad``, valid until the next one is read. The model is
    not moved; the gradients are cleared at the end."""
    try:
        for step, batch in enumerate(batches):
            for _, p in trainable:
                p.grad = None
            stats = accumulate_gradient(model, batch, cfg, env)
            if not math.isfinite(stats["loss"]):
                raise ValueError(f"The training loss is not finite at θ₀ on batch {step}")
            yield {name: p.grad for name, p in trainable}
    finally:
        for _, p in trainable:
            p.grad = None
