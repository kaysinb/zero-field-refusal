"""Study 5's linearized replay of one seed's F run (docs/study5-preregistration.md, "Extension
2"): from θ₀ alone, the frozen-gradient displacement δ̂ and the linearized displacement δ_lin as
vector files, and one JSON record.

    python scripts/study5_linear_replay.py CONFIG [key=value ...] --out-dir DIR [--epsilon E]
        [--check-steps T ...] [--reference-norm X | --reference-run RUN_DIR]
        [--surrogate SPEC] [--panel-size N] [--grad-r-only]

CONFIG and the overrides are those of the seed's F run (``optimizer=adam_frozen``, its rate
η = ``optimizer.lr``, its ``seed`` and its order: the stream order or ``train.order_seed``), one
seed and one order per call; the run itself is not needed. The seed's training stream must hold
``train.steps`` batches and its preconditioner stream must exist; nothing is generated here.

**The two recursions.** Arm F's update (``optim.Adam``, frozen mode, no weight decay) with a
drive d_t in place of the gradient at the current point, t = 1 … N:

    m_t = β₁ m_{t−1} + (1 − β₁) d_t,    δ_t = δ_{t−1} − η/(1 − β₁ᵗ) · D ⊙ m_t,    m_0 = δ_0 = 0,

D = (√v₀ + ε_adam)⁻¹, v₀ by the run's own code path (``linear_response.theta0_second_moment``),
β₁ and ε_adam from the config. g_t(θ) is batch t's gradient (``linear_response.feed_batches``,
the run's order) by the pass of ``linear_response.step_gradients`` (``train.accumulate_gradient``,
the training loss over the trainable set of ``models.freeze``).

- δ̂: d_t = g_t(θ₀). Its endpoint is ``linear_response.linear_displacement``'s closed form.
- δ_lin: d_t = g_t(θ₀) + H_t δ_{t−1}, δ_{t−1} its own displacement before step t, H_t δ the
  central difference [g_t(θ₀ + εδ) − g_t(θ₀ − εδ)] / (2ε) (``--epsilon``, default 1e-2); at
  t = 1, δ_0 = 0 and no difference is taken. On a quadratic loss δ_lin is the run's trajectory.
- At the check steps (``--check-steps``, default 5 15 25; each in 2 … N) the difference is also
  taken at ε/2 and 2ε; the record holds ‖P_{ε/2} − P_ε‖/‖P_ε‖ and ‖P_{2ε} − P_ε‖/‖P_ε‖.
- **Divergence.** With a reference norm X (``--reference-norm``, or ``--reference-run``: the
  ``displacement_norm`` of the F run's step-N monitor reading, ``ladder.read_f_run``, which also
  requires the run to be this config's, and θ₀, v₀ and the streams to be the ones it recorded),
  δ_lin diverges when ‖δ_lin,t‖ > 10·X; with or without X, when a loss, the difference or δ_lin
  is not finite. Its recursion then stops; the record names the step, the reason and the norm,
  and no δ_lin file is written. δ̂ runs to N.
- **The exact product** (``--hvp exact``, extension 2's fourth amendment; the default
  ``central`` is the registered central difference above): H_t δ by reverse over reverse at θ₀
  (``exact_hvp``), micro-batch by micro-batch (``--hvp-micro-batch-size``, default
  ``execution.micro_batch_size``): g = ∇L_m with its graph, then the gradient of gᵀ·δ (δ
  detached) accumulates, each micro-batch's graph freed; the loss, its masks and its batch-wide
  denominator are ``train.accumulate_gradient``'s, and attention runs under the math kernel
  (double backward). D is a constant. At the check steps the central difference at ε/2, ε and
  2ε is compared with the exact product, for H_t δ and for D ⊙ H_t δ (descriptive, ``checks``
  entries with ``mode: exact``). g_t(θ₀) keeps the production pass, so δ̂ is unchanged.
- ``--no-r`` skips the surrogate readouts (r, q: none), for the derivative probe's replays.
- ``--probe FILE`` writes the derivative probe's record instead of a replay (``probe``): at the
  probe steps (``--probe-steps``, default 2 5), for δ̂_{t−1} and a random direction of its norm,
  the gradient's repeat, the exact product's repeat, linearity and symmetry, the new path's
  gradient against the production pass, the central difference at the probe's steps h with the
  rounded shift θ₀ ± hv actually set, and the gradient over the full step against the product.
  No r, q or K.

The points θ₀ ± εδ are set over the trainable set (``evaluation.set_point``) from a host copy of
θ₀, and θ₀ is copied back exactly before every pass at θ₀ and at the end: the record holds the
trainable hash after the replay (required equal to θ₀'s) and at the end. θ₀, D, both momenta,
both displacements and the difference are fp32 on the host, as the optimizer keeps its state in
fp32 and the vector files are fp32; norms and inner products are reduced in float64. Each pass's
gradient stays on the device and is read to the host once, tensor by tensor.

**q.** r(θ₀), r(θ₀ + δ̂) and r(θ₀ + δ_lin) by ``surrogate.prompt_values`` on the harmful panel
of SPEC (``--surrogate``, default ``configs/study5/surrogate-2.yaml``; it must hold
``--panel-size`` prompts, default 100), in micro-batches of ``execution.eval_batch_size``, r the
mean over prompts; q_hat = r(θ₀ + δ̂) − r(θ₀), q_lin = r(θ₀ + δ_lin) − r(θ₀).

**K.** K_benign and K_harmful at θ₀ + δ̂ and θ₀ + δ_lin, as the run's evaluation computes them
(``evaluation.sequence_kl`` against θ₀ over the scoring panel's ``kl_benign`` and ``kl_harmful``
samples, ``train.Evaluator``'s cut). A diverged δ_lin has none: the record says so
(``K.delta_lin_unavailable``) and δ̂'s are always there. ``--no-k`` skips them.

**Outputs** in DIR: ``delta_hat.pt`` and ``delta_lin.pt`` by ``vectors.save_vector`` (kinds
``delta_hat`` and ``delta_lin``; provenance: seed, rate, steps, β₁, ε_adam, ε, the order record
or ``stream``, the config hash, the streams' identity and v₀'s hash), then ``record.json``: the
per-step norms ‖δ̂_t‖, ‖δ_lin,t‖, ‖g_t(θ₀)‖, ‖H_t δ_{t−1}‖ and their ratio, the check ratios,
the divergence, the r's and q's, the hashes of θ₀, v₀, the config and the surrogate, the streams,
the time, the memory, the device, the numerics and the git state.

With ``--grad-r-only``: ∇r(θ₀) over the trainable set (``surrogate.surrogate_gradient``, in
micro-batches of ``execution.micro_batch_size``) as ``grad_r.pt`` (kind ``grad_r``) and its
``record.json``; nothing else.

stdout carries numbers, ids and hashes only, never a prompt or a continuation. A record or a
vector file is never overwritten: a DIR that holds one of this mode's files is refused before any
computation. Exits 0 when written, 1 on an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel

from zero_field_refusal import ladder, linear_response, surrogate, vectors
from zero_field_refusal.batches import chunks, collate
from zero_field_refusal.config import config_hash
from zero_field_refusal.evaluation import sequence_kl, set_point
from zero_field_refusal.losses import loss_denominator, summed_nll
from zero_field_refusal.models import freeze
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import (
    configure_numerics,
    device_name,
    environment,
    load_stream_for,
)
from zero_field_refusal.seeding import derive_seed, seed_everything
from zero_field_refusal.study5_options import order_record, training_order
from zero_field_refusal.tensor_hash import tensors_sha256
from zero_field_refusal.tracking import git_state
from zero_field_refusal.train import accumulate_gradient

EPSILON = 1e-2
CHECK_STEPS = (5, 15, 25)
CHECK_FACTORS = {"half": 0.5, "double": 2.0}  # the difference also at ε/2 and 2ε
DIVERGENCE_FACTOR = 10.0  # δ_lin diverges past this multiple of the reference norm
RECORD = "record.json"
FILES = {"delta_hat": "delta_hat.pt", "delta_lin": "delta_lin.pt", "grad_r": "grad_r.pt"}
KIND, GRAD_R_KIND = "study5-linear-replay", "study5-grad-r"
HVP_MODES = ("central", "exact")
EXACT_CHECK_FACTORS = {"half": 0.5, "unit": 1.0, "double": 2.0}  # the difference against H δ
PROBE_KIND = "study5-dlin-probe"
PROBE_STEPS = (2, 5)
PROBE_H = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
PROBE_RESIDUAL_H = (1e-2, 1e-3, 1e-4)  # where H is also applied to the shift's rounding residual

Named = Mapping[str, torch.Tensor]
Trainable = Sequence[tuple[str, torch.nn.Parameter]]
Gradient = Callable[[Any], float]  # a batch → its loss, its gradient added to the .grad
# (batch, direction, out) → the batch loss; out ← H v at the model's current point
Product = Callable[[Any, Sequence[torch.Tensor], dict[str, torch.Tensor]], float]


def _number(x: float) -> float | None:
    """A finite float, or None (JSON has no NaN or infinity)."""
    return x if math.isfinite(x) else None


def _distance(a: Named, b: Named) -> float:
    """‖a − b‖ over a's names, reduced in float64."""
    return math.sqrt(math.fsum((t - b[name]).double().square().sum().item()
                               for name, t in a.items()))  # fmt: skip


def _clear(trainable: Trainable) -> None:
    for _, p in trainable:
        p.grad = None


def _pass(trainable: Trainable, gradient: Gradient, batch: Any) -> float:
    """One gradient pass at the model's current point; ``.grad`` then holds the batch's
    gradient."""
    _clear(trainable)
    return float(gradient(batch))


def _product(
    trainable: Trainable,
    start: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    gradient: Gradient,
    batch: Any,
    eps: float,
    out: dict[str, torch.Tensor],
) -> tuple[float, float]:
    """``out`` ← [g(θ₀ + εδ) − g(θ₀ − εδ)] / (2ε) on the host (θ₀ = ``start``, δ =
    ``direction``); returns the two losses. The model is left at θ₀ − εδ, its ``.grad`` empty."""
    with torch.no_grad():
        set_point(trainable, start, direction, eps)
    plus = _pass(trainable, gradient, batch)
    with torch.no_grad():
        for name, p in trainable:
            out[name].copy_(p.grad)
        set_point(trainable, start, direction, -eps)
    minus = _pass(trainable, gradient, batch)
    with torch.no_grad():
        for name, p in trainable:
            out[name].sub_(p.grad.to(out[name].device)).div_(2 * eps)
    _clear(trainable)
    return plus, minus


def exact_hvp(
    model: torch.nn.Module,
    trainable: Trainable,
    batch: Any,
    cfg,
    env,
    direction: Sequence[torch.Tensor],
    out: dict[str, torch.Tensor],
    *,
    micro_batch_size: int | None = None,
    gradient_out: dict[str, torch.Tensor] | None = None,
) -> float:
    """``out`` ← H v at the model's current point: H the Hessian of the batch's training loss, v
    = ``direction`` (the trainable set's order, a constant). Reverse over reverse, micro-batch by
    micro-batch: g = ∇L_m with its graph, then ∇(gᵀv) accumulates into ``.grad`` and the
    micro-batch's graph is freed. The loss, its masks and its batch-wide denominator are
    ``train.accumulate_gradient``'s (``summed_nll``, no soft targets); attention runs under the
    math kernel, which supports double backward. ``gradient_out`` (the probe) also receives g.
    Returns the batch loss; ``.grad`` is left empty."""
    params = [p for _, p in trainable]
    cap = cfg.train.max_response_tokens
    denominator = loss_denominator(cfg.train.loss, batch, cap)
    micro = int(micro_batch_size or cfg.execution.micro_batch_size)
    v = [d.to(p.device, p.dtype) for d, p in zip(direction, params, strict=True)]
    if gradient_out is not None:
        for t in gradient_out.values():
            t.zero_()
    _clear(trainable)
    loss = 0.0
    with sdpa_kernel(SDPBackend.MATH):
        for chunk in chunks(batch, micro):
            rows = collate(chunk, pad_id=env.pad_id, max_response_tokens=cap, device=env.device)
            term = summed_nll(model, rows) / denominator
            grads = torch.autograd.grad(term, params, create_graph=True, allow_unused=True)
            used = [(p, g, u) for p, g, u in zip(params, grads, v, strict=True) if g is not None]
            dot = sum((g * u).sum() for _, g, u in used)
            torch.autograd.backward(dot, inputs=[p for p, _, _ in used])
            if gradient_out is not None:
                with torch.no_grad():
                    for (name, _), g in zip(trainable, grads, strict=True):
                        if g is not None:
                            gradient_out[name].add_(g.detach().to(gradient_out[name].device))
            loss += term.item()
            del grads, used, dot, term
    del v
    with torch.no_grad():
        for name, p in trainable:
            if p.grad is None:
                out[name].zero_()
            else:
                out[name].copy_(p.grad)
    _clear(trainable)
    return loss


def _weighted_distance(a: Named, b: Named | None, w: Named) -> float:
    """‖w ⊙ (a − b)‖ (‖w ⊙ a‖ without b) over a's names, reduced in float64."""
    return math.sqrt(math.fsum(
        (w[name].double() * (t.double() if b is None else t.double() - b[name].double()))
        .square().sum().item() for name, t in a.items()))  # fmt: skip


def _check_exact(
    trainable: Trainable,
    start: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    gradient: Gradient,
    batch: Any,
    epsilon: float,
    product: Named,
    D: Named,
    step: int,
    product_norm: float,
) -> dict[str, Any]:
    """The central difference P_h at h = ε/2, ε and 2ε against the exact product P (descriptive):
    ‖P_h − P‖/‖P‖, the same in D's weights (D ⊙ P drives the update), ‖P_h‖ and the losses."""
    other = {name: torch.empty_like(t) for name, t in product.items()}
    d_norm = _weighted_distance(product, None, D)
    out: dict[str, Any] = {"step": step, "mode": "exact", "epsilon": epsilon,
                           "norm": product_norm, "d_norm": _number(d_norm), "fd": {}}  # fmt: skip
    for key, factor in EXACT_CHECK_FACTORS.items():
        plus, minus = _product(trainable, start, direction, gradient, batch, factor * epsilon,
                               other)  # fmt: skip
        out["fd"][key] = {
            "h": factor * epsilon,
            "relative": _number(_distance(other, product) / product_norm)
            if product_norm > 0 else None,
            "relative_d": _number(_weighted_distance(other, product, D) / d_norm)
            if d_norm > 0 else None,
            "norm": _number(vectors.norm(other)),
            "losses": [_number(plus), _number(minus)],
        }  # fmt: skip
    return out


def replay(
    trainable: Trainable,
    theta0: Named,
    D: Named,
    batches: Sequence[Any],
    gradient: Gradient,
    *,
    lr: float,
    beta1: float,
    epsilon: float = EPSILON,
    check_steps: Sequence[int] = (),
    limit: float | None = None,
    feedback: bool = True,
    hvp: Product | None = None,
) -> dict[str, Any]:
    """δ̂ and δ_lin over ``batches`` (module docstring), from the host tensors ``theta0`` and
    ``D`` (by name, over the trainable set), the model's parameters ``trainable`` and
    ``gradient``, one pass of a batch at the model's current point. ``limit`` is δ_lin's norm
    bound (None: only a non-finite value diverges). Without ``feedback`` no difference is taken
    and δ_lin is δ̂ bit for bit. ``hvp`` (the exact product at θ₀, ``exact_hvp``) replaces the
    central difference; the check steps then compare the difference with it. The model is at θ₀
    exactly on return. Returns ``delta_hat``, ``delta_lin`` (None once diverged), ``per_step``,
    ``checks``, ``divergence`` (None or its step, reason and norms), ``passes``, the gradient
    passes taken, and ``hvp_passes``, the exact products."""
    names = [name for name, _ in trainable]
    start = [theta0[name] for name in names]

    def zeros() -> dict[str, torch.Tensor]:
        return {name: torch.zeros_like(theta0[name]) for name in names}

    m_hat, d_hat, m_lin, d_lin = zeros(), zeros(), zeros(), zeros()
    product = zeros() if feedback else None
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    divergence, passes, hvp_passes, lin_norm = None, 0, 0, 0.0
    with torch.no_grad():
        set_point(trainable, start, start, 0.0)  # θ₀ exactly
    for t, batch in enumerate(batches, start=1):
        began = time.perf_counter()
        scale = -lr / (1 - beta1**t)
        loss = _pass(trainable, gradient, batch)  # g_t(θ₀)
        passes += 1
        if not math.isfinite(loss):
            raise ValueError(f"The training loss is not finite at θ₀ on batch {t - 1}")
        squares = []
        with torch.no_grad():
            for name, p in trainable:
                g = p.grad.to(m_hat[name].device)
                squares.append(g.double().square().sum().item())
                m_hat[name].lerp_(g, 1 - beta1)
                d_hat[name].addcmul_(D[name], m_hat[name], value=scale)
                if d_lin is not None:
                    m_lin[name].lerp_(g, 1 - beta1)  # g_t(θ₀); H_t δ_{t−1} is added below
        _clear(trainable)
        grad_norm = math.sqrt(math.fsum(squares))
        if not math.isfinite(grad_norm):
            raise ValueError(f"The gradient is not finite at θ₀ on batch {t - 1}")
        row: dict[str, Any] = {"step": t, "loss": loss, "grad_norm": grad_norm,
                               "delta_hat_norm": vectors.norm(d_hat), "delta_lin_norm": None,
                               "hdelta_norm": None, "hdelta_ratio": None,
                               "loss_plus": None, "loss_minus": None}  # fmt: skip
        if d_lin is not None:
            previous, reason, hdelta = lin_norm, None, 0.0
            if feedback and t > 1:
                direction = [d_lin[name] for name in names]
                if hvp is None:
                    plus, minus = _product(trainable, start, direction, gradient, batch,
                                           epsilon, product)  # fmt: skip
                    passes += 2
                    losses: tuple[float, ...] = (plus, minus)
                    row.update(loss_plus=_number(plus), loss_minus=_number(minus))
                else:  # the exact product at θ₀, where the model is
                    at = hvp(batch, direction, product)
                    hvp_passes += 1
                    losses = (at,)
                    row["loss_hvp"] = _number(at)
                hdelta = vectors.norm(product)
                finite = all(math.isfinite(x) for x in (*losses, hdelta))
                if t in check_steps and finite:
                    if hvp is None:
                        checks.append(_check(trainable, start, direction, gradient, batch,
                                             epsilon, product, t, hdelta))  # fmt: skip
                        passes += 2 * len(CHECK_FACTORS)
                    else:
                        checks.append(_check_exact(trainable, start, direction, gradient, batch,
                                                   epsilon, product, D, t, hdelta))  # fmt: skip
                        passes += 2 * len(EXACT_CHECK_FACTORS)
                with torch.no_grad():
                    set_point(trainable, start, start, 0.0)  # θ₀ exactly
                if finite:
                    with torch.no_grad():
                        for name in names:
                            m_lin[name].add_(product[name], alpha=1 - beta1)
                else:
                    reason = "nonfinite_difference"
            row["hdelta_norm"] = _number(hdelta)
            row["hdelta_ratio"] = _number(hdelta / grad_norm) if grad_norm > 0 else None
            if reason is None:
                with torch.no_grad():
                    for name in names:
                        d_lin[name].addcmul_(D[name], m_lin[name], value=scale)
                lin_norm = vectors.norm(d_lin)
                row["delta_lin_norm"] = _number(lin_norm)
                if not math.isfinite(lin_norm):
                    reason = "nonfinite_displacement"
                elif limit is not None and lin_norm > limit:
                    reason = "bound"
            if reason is not None:
                divergence = {"step": t, "reason": reason,
                              "norm": None if reason == "nonfinite_difference"
                              else _number(lin_norm),
                              "previous_norm": previous}  # fmt: skip
                m_lin = d_lin = product = None  # freed: its recursion stops here
        row["seconds"] = time.perf_counter() - began
        rows.append(row)
    return {"delta_hat": d_hat, "delta_lin": d_lin, "per_step": rows, "checks": checks,
            "divergence": divergence, "passes": passes, "hvp_passes": hvp_passes}  # fmt: skip


def _check(
    trainable: Trainable,
    start: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    gradient: Gradient,
    batch: Any,
    epsilon: float,
    product: Named,
    step: int,
    product_norm: float,
) -> dict[str, Any]:
    """The difference at ε/2 and 2ε against P_ε (``product``): ‖P − P_ε‖/‖P_ε‖ and ‖P‖."""
    other = {name: torch.empty_like(t) for name, t in product.items()}
    out: dict[str, Any] = {"step": step, "epsilon": epsilon, "norm": product_norm}
    for key, factor in CHECK_FACTORS.items():
        plus, minus = _product(trainable, start, direction, gradient, batch, factor * epsilon,
                               other)  # fmt: skip
        ratio = _distance(other, product) / product_norm if product_norm > 0 else math.nan
        out[key] = _number(ratio)
        out[f"{key}_norm"] = _number(vectors.norm(other))
        out[f"{key}_losses"] = [_number(plus), _number(minus)]
    return out


# ------------------------------------------------------------------------------- the seed's inputs


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def panel_items(env, spec: surrogate.SurrogateSpec, panel_size: int) -> list[dict]:
    """The harmful panel of the surrogate; it must hold ``panel_size`` prompts."""
    items = env.prompt_set(spec.harmful_panel)
    if len(items) != panel_size:
        raise ValueError(f"panel {spec.harmful_panel} has {len(items)} prompts, not {panel_size}")
    return items


def panel_record(spec: surrogate.SurrogateSpec, items: list[dict]) -> dict[str, Any]:
    ids = [item["id"] for item in items]
    return {"name": spec.harmful_panel, "count": len(ids), "ids_sha256": ids_sha256(ids)}


def reference_of(args: argparse.Namespace, cfg) -> dict[str, Any] | None:
    """The reference norm X of the divergence rule: ``--reference-norm``, or the step-N
    ``displacement_norm`` of the F run at ``--reference-run`` (``ladder.read_f_run``: this
    config's run, all its updates) or at ``--reference-fresh-run`` (the seed's fresh F run at the
    same rate, for another order of the same responses), with the identities that run
    recorded."""
    if args.reference_norm is not None:
        x = float(args.reference_norm)
        if not (math.isfinite(x) and x > 0):
            raise ValueError(f"--reference-norm {x!r}: a finite positive norm is needed")
        return {"norm": x, "source": "--reference-norm"}
    if args.reference_fresh_run is not None:  # the seed's fresh F run at the same rate
        fresh = OmegaConf.load(Path(args.reference_fresh_run) / "config.yaml")  # its own config
        role = OmegaConf.select(fresh, "study5.role", default=None)
        if (role, int(fresh.seed), float(fresh.optimizer.lr)) != (
            "fresh",
            int(cfg.seed),
            float(cfg.optimizer.lr),
        ):
            raise ValueError(f"--reference-fresh-run {args.reference_fresh_run}: not this seed's "
                             "fresh F run at this rate")  # fmt: skip
        f_run = ladder.read_f_run(args.reference_fresh_run, fresh)
    elif args.reference_run is None:
        return None
    else:
        f_run = ladder.read_f_run(args.reference_run, cfg)
    return {"norm": f_run["displacement_norm"], "source": f_run["displacement_source"],
            "run_id": f_run["run_id"], "config_hash": f_run["config_hash"],
            "theta0_trainable_sha256": f_run["theta0_trainable_sha256"],
            "v0_sha256": f_run["v0_sha256"], "streams": f_run["streams"]}  # fmt: skip


def _same(what: str, ours: Any, theirs: Any) -> None:
    """An identity the reference run recorded, which the replay must share (none: no check)."""
    if theirs is not None and ours != theirs:
        raise ValueError(f"{what} differs from the reference run's: {ours!r} against {theirs!r}")


def _memory(device: torch.device) -> dict[str, float]:
    out = {"host_peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                     * 1024 / 1e9, 3)}  # fmt: skip
    if device.type == "cuda":
        out["peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        out["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    return out


def _common(cfg, env, spec: surrogate.SurrogateSpec, numerics: dict) -> dict[str, Any]:
    return {
        "seed": int(cfg.seed),
        "config_hash": config_hash(cfg),
        "optimizer": cfg.optimizer.name,
        "surrogate_sha256": spec.sha256,
        "surrogate_version": spec.version,
        "model": {"name": cfg.model.name, "revision": cfg.model.revision,
                  "dtype": cfg.model.dtype, "attn_implementation": cfg.model.attn_implementation},
        "prompt_manifest_sha256": env.prompt_manifest_sha256,
        "numerics": numerics,
        "device": device_name(torch.device(env.device)),
        "git": git_state(),
    }  # fmt: skip


# ------------------------------------------------------------------------------- the two modes


def k_panels(cfg, env) -> tuple[dict[str, list], dict[str, Any]]:
    """The scoring panel's K samples as ``train.Evaluator`` reads them (``kl_harmful`` and
    ``kl_benign``, each cut to ``eval.kl.limit`` prompts when a limit is set), and their streams'
    identities."""
    samples, identities = {}, {}
    limit, per_prompt = cfg.eval.kl.limit, cfg.eval.kl.samples_per_prompt
    for kind in ("kl_harmful", "kl_benign"):
        found, manifest = load_stream_for(cfg, env, kind, None)
        samples[kind] = found[: limit * per_prompt] if limit else found
        identities[kind] = {"sha256": manifest["sha256"], "count": len(samples[kind])}
    return samples, identities


def _setup(cfg, env, reference: Mapping[str, Any] | None) -> dict[str, Any]:
    """The seed's batches (its order), θ₀'s model and trainable set, v₀ and D on the host, θ₀
    on the host, each identity checked against the reference run's."""
    seed_everything(derive_seed(cfg.seed, "torch"))  # as the run seeds itself
    batches, training = linear_response.feed_batches(cfg, env)  # fails early on a short stream
    order = training_order(cfg)
    reference = dict(reference or {})
    streams = reference.get("streams", {})
    _same("the training stream", training, streams.get("training"))
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    names = [name for name, _ in trainable]
    theta0_sha = tensors_sha256(trainable)
    _same("θ₀'s trainable tensors", theta0_sha, reference.get("theta0_trainable_sha256"))
    v0, preconditioner = linear_response.theta0_second_moment(cfg, env, model, trainable)
    v0_sha = linear_response.v0_sha256(trainable, v0)
    _same("v₀", v0_sha, reference.get("v0_sha256"))
    _same("the preconditioner stream", preconditioner, streams.get("preconditioner"))
    eps, lr, beta1 = float(cfg.optimizer.eps), float(cfg.optimizer.lr), float(cfg.optimizer.beta1)
    D = {}
    for i, name in enumerate(names):  # each tensor to the host, its device copy freed
        D[name] = v0[i].sqrt_().add_(eps).reciprocal_().to("cpu", torch.float32)
        v0[i] = None
    del v0
    theta0 = {name: p.detach().to("cpu", torch.float32, copy=True) for name, p in trainable}
    return {"batches": batches, "training": training, "order": order, "reference": reference,
            "model": model, "trainable": trainable, "names": names, "theta0_sha": theta0_sha,
            "v0_sha": v0_sha, "preconditioner": preconditioner, "eps": eps, "lr": lr,
            "beta1": beta1, "D": D, "theta0": theta0}  # fmt: skip


def _grad_copy(trainable: Trainable) -> dict[str, torch.Tensor]:
    """The ``.grad`` of each trainable tensor on the host (fp32), then cleared."""
    out = {name: p.grad.detach().to("cpu", torch.float32, copy=True) for name, p in trainable}
    _clear(trainable)
    return out


def _ratio(a: float, b: float) -> float | None:
    return _number(a / b) if b > 0 else None


def _relative(a: Named, b: Named) -> float | None:
    """‖a − b‖ / ‖b‖, or None when b is zero."""
    return _ratio(_distance(a, b), vectors.norm(b))


def _relative_d(a: Named, b: Named, w: Named) -> float | None:
    """‖w ⊙ (a − b)‖ / ‖w ⊙ b‖, or None when w ⊙ b is zero."""
    return _ratio(_weighted_distance(a, b, w), _weighted_distance(b, None, w))


def probe(
    cfg,
    env,
    numerics: dict,
    *,
    steps: Sequence[int] = PROBE_STEPS,
    hs: Sequence[float] = PROBE_H,
    residual_hs: Sequence[float] = PROBE_RESIDUAL_H,
    direction_seed: int = 0,
    hvp_micro_batch_size: int | None = None,
    reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The derivative probe (extension 2's fourth amendment) of ``cfg``'s seed and order: at
    each probe step t, for v = δ̂_{t−1} (the frozen-gradient recursion's displacement before t)
    and for a random direction of its norm (``direction_seed``), on batch t at θ₀:

    - the production gradient pass twice (its repeat);
    - the exact product H v twice (its repeat), H(2v) against 2 H v (linearity), the new path's
      gradient and loss against the production pass; per step, uᵀ H w against wᵀ H u over the
      two directions (symmetry);
    - for each h of ``hs``: the shift actually set, s± = fl(θ₀ ± hv) − θ₀ (fp32): ‖s± ∓ hv‖/‖hv‖
      and the share of elements with v ≠ 0 whose shift is zero; the central difference P_h
      against H v, also in D's weights; for h in ``residual_hs``, H applied to the rounding
      residual (s₊ − s₋)/(2h) − v, relative to H v (rounding's share of P_h's error);
    - the gradient over the full step, g(θ₀ + v) − g(θ₀), against H v.

    Numbers only; no r, q or K. The model is at θ₀ exactly on return (checked)."""
    started = time.perf_counter()
    device = torch.device(env.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    s = _setup(cfg, env, reference)
    batches, model, trainable, names = s["batches"], s["model"], s["trainable"], s["names"]
    theta0, D, lr, beta1 = s["theta0"], s["D"], s["lr"], s["beta1"]
    micro = int(hvp_micro_batch_size or cfg.execution.micro_batch_size)
    steps = sorted(set(int(t) for t in steps))
    bad = [t for t in steps if not 2 <= t <= len(batches)]
    if bad:
        raise ValueError(f"probe steps {bad}: each must lie in 2 … {len(batches)}")
    start = [theta0[name] for name in names]

    def gradient(batch) -> float:
        return accumulate_gradient(model, batch, cfg, env)["loss"]

    def at_point(direction: Sequence[torch.Tensor] | None, a: float) -> None:
        with torch.no_grad():
            set_point(trainable, start, start if direction is None else direction, a)

    def product(batch, v: Named, *, gradient_out=None) -> tuple[dict[str, torch.Tensor], float]:
        at_point(None, 0.0)
        out = {name: torch.empty_like(theta0[name]) for name in names}
        loss = exact_hvp(model, trainable, batch, cfg, env, [v[name] for name in names], out,
                         micro_batch_size=micro, gradient_out=gradient_out)  # fmt: skip
        return out, loss

    entries, symmetry = [], []
    for t in steps:
        batch = batches[t - 1]
        # δ̂_{t−1}: the frozen-gradient recursion over the batches before t (as the replay's)
        prefix = replay(trainable, theta0, D, batches[: t - 1], gradient, lr=lr, beta1=beta1,
                        feedback=False)["delta_hat"]  # fmt: skip
        norm = vectors.norm(prefix)
        g = torch.Generator().manual_seed(int(direction_seed) * 1000 + t)
        noise = {name: torch.randn(theta0[name].shape, generator=g) for name in names}
        scale = norm / vectors.norm(noise)
        random = {name: noise[name].mul_(scale) for name in names}
        products = {}
        for kind, v in (("delta_hat", prefix), ("random", random)):
            direction = [v[name] for name in names]
            entry: dict[str, Any] = {
                "step": t,
                "direction": kind,
                "v_norm": _number(vectors.norm(v)),
            }
            at_point(None, 0.0)
            loss_a = _pass(trainable, gradient, batch)
            g_a = _grad_copy(trainable)
            loss_b = _pass(trainable, gradient, batch)
            g_b = _grad_copy(trainable)
            entry["gradient_repeat"] = {
                "relative": _relative(g_b, g_a), "relative_d": _relative_d(g_b, g_a, D),
                "losses": [_number(loss_a), _number(loss_b)],
                "equal": all(torch.equal(g_a[n], g_b[n]) for n in names)}  # fmt: skip
            del g_b
            new_g = {name: torch.zeros_like(theta0[name]) for name in names}
            hv, loss_new = product(batch, v, gradient_out=new_g)
            hv_b, _ = product(batch, v)
            twice = {name: 2 * v[name] for name in names}
            hv_2, _ = product(batch, twice)
            del twice
            double_hv = {name: 2 * hv[name] for name in names}
            hv_norm, hv_d = vectors.norm(hv), _weighted_distance(hv, None, D)
            entry["exact"] = {
                "norm": _number(hv_norm), "d_norm": _number(hv_d),
                "ratio_to_gradient": _number(hv_norm / vectors.norm(g_a)),
                "repeat_relative": _relative(hv_b, hv),
                "linearity_relative": _relative(hv_2, double_hv),
                "gradient_vs_production": _relative(new_g, g_a),
                # the same in D's weights: D ⊙ (g + H δ) drives the recursion
                "repeat_relative_d": _relative_d(hv_b, hv, D),
                "linearity_relative_d": _relative_d(hv_2, double_hv, D),
                "gradient_vs_production_d": _relative_d(new_g, g_a, D),
                "loss_vs_production": _number(abs(loss_new - loss_a)),
            }  # fmt: skip
            del hv_b, hv_2, double_hv, new_g
            rows = []
            for h in hs:
                shifts = {}
                for sign in (1.0, -1.0):
                    moved = {name: torch.add(theta0[name], v[name], alpha=sign * h) - theta0[name]
                             for name in names}  # fmt: skip
                    target = math.sqrt(math.fsum((h * v[name].double()).square().sum().item()
                                                 for name in names))  # fmt: skip
                    error = math.sqrt(math.fsum(
                        (moved[name].double() - sign * h * v[name].double()).square().sum().item()
                        for name in names))  # fmt: skip
                    nonzero = sum(int((v[name] != 0).sum()) for name in names)
                    lost = sum(int(((moved[name] == 0) & (v[name] != 0)).sum()) for name in names)
                    shifts[sign] = (moved, error / target if target > 0 else None,
                                    lost / nonzero if nonzero else None)  # fmt: skip
                p_h = {name: torch.empty_like(theta0[name]) for name in names}
                plus, minus = _product(trainable, start, direction, gradient, batch, h, p_h)
                at_point(None, 0.0)
                row = {
                    "h": h,
                    "shift_error": [_number(shifts[1.0][1]), _number(shifts[-1.0][1])],
                    "shift_lost_share": [shifts[1.0][2], shifts[-1.0][2]],
                    "norm": _number(vectors.norm(p_h)),
                    "relative": _relative(p_h, hv),
                    "relative_d": _ratio(_weighted_distance(p_h, hv, D), hv_d),
                    "losses": [_number(plus), _number(minus)],
                    "residual_relative": None,
                }
                if h in residual_hs:
                    residual = {
                        name: (
                            (shifts[1.0][0][name].double() - shifts[-1.0][0][name].double())
                            / (2 * h)
                            - v[name].double()
                        ).float()
                        for name in names
                    }
                    h_res, _ = product(batch, residual)
                    row["residual_relative"] = _ratio(vectors.norm(h_res), hv_norm)
                    del residual, h_res
                del shifts, p_h
                rows.append(row)
            entry["central"] = rows
            at_point(direction, 1.0)
            loss_step = _pass(trainable, gradient, batch)
            g_step = _grad_copy(trainable)
            at_point(None, 0.0)
            change = {name: g_step[name] - g_a[name] for name in names}
            entry["full_step"] = {
                "loss": _number(loss_step),
                "relative_to_product": _relative(change, hv),
                "ratio_to_product": _ratio(vectors.norm(change), hv_norm),
                "relative_d": _ratio(_weighted_distance(change, hv, D), hv_d),
            }
            del g_step, change, g_a
            products[kind] = (v, hv)
            entries.append(entry)
        (w, hw), (u, hu) = products["delta_hat"], products["random"]
        u_hw, w_hu = vectors.dot(u, hw), vectors.dot(w, hu)
        scale_sym = max(abs(u_hw), abs(w_hu))
        symmetry.append(
            {
                "step": t,
                "u_H_w": _number(u_hw),
                "w_H_u": _number(w_hu),
                "relative": _number(abs(u_hw - w_hu) / scale_sym) if scale_sym > 0 else None,
            }
        )
        del products, prefix, random, noise
    at_point(None, 0.0)
    restored = tensors_sha256(trainable)
    if restored != s["theta0_sha"]:
        raise ValueError(
            f"θ₀ is not restored after the probe: {restored} against {s['theta0_sha']}"
        )
    order = s["order"]
    return {
        "kind": PROBE_KIND,
        "point": "theta0",
        "lr": lr,
        "beta1": beta1,
        "eps": s["eps"],
        "steps": steps,
        "h": list(hs),
        "residual_h": list(residual_hs),
        "direction_seed": int(direction_seed),
        "order": "stream" if order is None else order_record(cfg, order),
        "hvp": {"mode": "exact", "backend": "math", "micro_batch_size": micro},
        "entries": entries,
        "symmetry": symmetry,
        "theta0_trainable_sha256": s["theta0_sha"],
        "restored_trainable_sha256": restored,
        "v0_sha256": s["v0_sha"],
        "streams": {"training": s["training"], "preconditioner": s["preconditioner"]},
        "micro_batch_size": int(cfg.execution.micro_batch_size),
        "seed": int(cfg.seed),
        "config_hash": config_hash(cfg),
        "optimizer": cfg.optimizer.name,
        "model": {"name": cfg.model.name, "revision": cfg.model.revision,
                  "dtype": cfg.model.dtype, "attn_implementation": cfg.model.attn_implementation},
        "numerics": numerics,
        "device": device_name(device),
        "git": git_state(),
        "memory": _memory(device),
        "seconds": time.perf_counter() - started,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip


def run(
    cfg,
    env,
    spec: surrogate.SurrogateSpec,
    numerics: dict,
    *,
    epsilon: float = EPSILON,
    check_steps: Sequence[int] = CHECK_STEPS,
    reference: Mapping[str, Any] | None = None,
    panel_size: int = 100,
    k_readouts: bool = True,
    hvp_mode: str = "central",
    hvp_micro_batch_size: int | None = None,
    r_readouts: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The replay of ``cfg``'s seed and order in ``env``: (record, vectors), vectors holding
    ``names``, ``provenance``, ``delta_hat`` and ``delta_lin`` (None when diverged)."""
    if hvp_mode not in HVP_MODES:
        raise ValueError(f"hvp mode {hvp_mode!r}: one of {HVP_MODES}")
    exact = hvp_mode == "exact"
    started = time.perf_counter()
    device = torch.device(env.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    items = panel_items(env, spec, panel_size) if r_readouts else []
    s = _setup(cfg, env, reference)
    batches, order, reference = s["batches"], s["order"], s["reference"]
    model, trainable, names = s["model"], s["trainable"], s["names"]
    theta0_sha, v0_sha, theta0, D = s["theta0_sha"], s["v0_sha"], s["theta0"], s["D"]
    training, preconditioner = s["training"], s["preconditioner"]
    eps, lr, beta1 = s["eps"], s["lr"], s["beta1"]
    hvp_micro = int(hvp_micro_batch_size or cfg.execution.micro_batch_size) if exact else None
    setup_seconds = time.perf_counter() - started  # the batches, θ₀, v₀ and D

    limit = None if not reference else DIVERGENCE_FACTOR * float(reference["norm"])
    began = time.perf_counter()
    result = replay(
        trainable, theta0, D, batches,
        lambda batch: accumulate_gradient(model, batch, cfg, env)["loss"],
        lr=lr, beta1=beta1, epsilon=epsilon, check_steps=check_steps, limit=limit,
        hvp=(lambda batch, direction, out: exact_hvp(
            model, trainable, batch, cfg, env, direction, out, micro_batch_size=hvp_micro))
        if exact else None,
    )  # fmt: skip
    replay_seconds = time.perf_counter() - began
    del D
    restored = tensors_sha256(trainable)
    if restored != theta0_sha:
        raise ValueError(f"θ₀ is not restored after the replay: {restored} against {theta0_sha}")
    delta_hat, delta_lin = result["delta_hat"], result["delta_lin"]

    began = time.perf_counter()
    start = [theta0[name] for name in names]
    prompts = [item["text"] for item in items]
    micro = int(cfg.execution.eval_batch_size)

    def r_at(delta: Named | None) -> float:
        with torch.no_grad():
            if delta is None:
                set_point(trainable, start, start, 0.0)
            else:
                set_point(trainable, start, [delta[name] for name in names], 1.0)
        values = surrogate.prompt_values(
            model, env.tokenizer, prompts, spec.refusals, spec.compliances, device=env.device,
            micro_batch_size=micro, max_tokens=spec.max_tokens,
        )  # fmt: skip
        return math.fsum(values) / len(values)

    r0 = r_hat = r_lin = None
    if r_readouts:
        r0, r_hat = r_at(None), r_at(delta_hat)
        r_lin = None if delta_lin is None else r_at(delta_lin)
    surrogate_seconds = time.perf_counter() - began

    # K_benign and K_harmful at θ₀ + δ̂ and θ₀ + δ_lin, as the run's evaluation computes them
    began = time.perf_counter()
    k_record: dict[str, Any] | None = None
    if k_readouts:
        samples, k_streams = k_panels(cfg, env)
        ref_model = env.new_model()
        ref_model.requires_grad_(False)
        k_micro = int(cfg.execution.micro_batch_size)

        def k_at(delta: Named) -> dict[str, float | None]:
            with torch.no_grad():
                set_point(trainable, start, [delta[name] for name in names], 1.0)
            return {f"K_{kind.removeprefix('kl_')}": _number(sequence_kl(
                model, ref_model, found, pad_id=env.pad_id, micro_batch_size=k_micro,
            ).mean().item()) for kind, found in samples.items()}  # fmt: skip

        k_record = {"delta_hat": k_at(delta_hat), "streams": k_streams,
                    "delta_lin": None if delta_lin is None else k_at(delta_lin),
                    "delta_lin_unavailable": None if delta_lin is not None else
                    "diverged: no delta_lin"}  # fmt: skip
        del ref_model
    with torch.no_grad():
        set_point(trainable, start, start, 0.0)
    final = tensors_sha256(trainable)
    k_seconds = time.perf_counter() - began

    hat_norm = vectors.norm(delta_hat)
    norms: dict[str, Any] = {"delta_hat": hat_norm, "delta_lin": None, "difference": None,
                             "relative_difference": None, "cosine": None}  # fmt: skip
    if delta_lin is not None:
        lin_norm, difference = vectors.norm(delta_lin), _distance(delta_lin, delta_hat)
        norms.update(delta_lin=lin_norm, difference=difference,
                     relative_difference=difference / hat_norm if hat_norm > 0 else None,
                     cosine=(vectors.dot(delta_lin, delta_hat) / (lin_norm * hat_norm)
                             if lin_norm * hat_norm > 0 else None))  # fmt: skip
    divergence = result["divergence"]
    order_rec: Any = "stream" if order is None else order_record(cfg, order)
    common = _common(cfg, env, spec, numerics)
    record = {
        "kind": KIND,
        "point": "theta0",
        "lr": lr,
        "beta1": beta1,
        "eps": eps,
        "steps": int(cfg.train.steps),
        "batch_size": int(cfg.train.batch_size),
        "epsilon": epsilon,
        "check_steps": list(check_steps),
        "order": order_rec,
        "reference": None if not reference else {
            "norm": float(reference["norm"]), "source": reference["source"],
            "run_id": reference.get("run_id"), "config_hash": reference.get("config_hash"),
            "factor": DIVERGENCE_FACTOR, "bound": limit},
        "divergence": {"status": "complete" if divergence is None else "diverged",
                       **(divergence or {"step": None, "reason": None, "norm": None,
                                         "previous_norm": None})},
        "hvp": {"mode": hvp_mode, "backend": "math" if exact else None,
                "micro_batch_size": hvp_micro},
        "r_readouts": r_readouts,
        "r": None if not r_readouts else {
            "theta0": _number(r0), "delta_hat": _number(r_hat),
            "delta_lin": None if r_lin is None else _number(r_lin)},
        "q_hat": None if not r_readouts else _number(r_hat - r0),
        "q_lin": None if r_lin is None else _number(r_lin - r0),
        "K": k_record,
        "norms": norms,
        "per_step": result["per_step"],
        "checks": result["checks"],
        "passes": result["passes"],
        "hvp_passes": result["hvp_passes"],
        "theta0_trainable_sha256": theta0_sha,
        "restored_trainable_sha256": restored,
        "final_trainable_sha256": final,
        "v0_sha256": v0_sha,
        "streams": {"training": training, "preconditioner": preconditioner},
        "panel": panel_record(spec, items) if r_readouts else None,
        "surrogate_micro_batch_size": micro,
        "micro_batch_size": int(cfg.execution.micro_batch_size),
        **common,
        "memory": _memory(device),
        "seconds": {"setup": setup_seconds, "replay": replay_seconds,
                    "surrogate": surrogate_seconds, "k": k_seconds,
                    "total": time.perf_counter() - started},
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    provenance = {"seed": int(cfg.seed), "lr": lr, "steps": int(cfg.train.steps), "beta1": beta1,
                  "eps": eps, "epsilon": epsilon, "order": order_rec,
                  "config_hash": common["config_hash"],
                  "streams": record["streams"], "v0_sha256": v0_sha,
                  **({"hvp": "exact"} if exact else {})}  # fmt: skip
    return record, {"names": names, "provenance": provenance, "delta_hat": delta_hat,
                    "delta_lin": delta_lin}  # fmt: skip


def run_grad_r(
    cfg, env, spec: surrogate.SurrogateSpec, numerics: dict, *, panel_size: int = 100
) -> tuple[dict[str, Any], dict[str, Any]]:
    """∇r(θ₀) over the trainable set: (record, vectors) as ``run``'s, with ``grad_r``."""
    started = time.perf_counter()
    device = torch.device(env.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    items = panel_items(env, spec, panel_size)
    seed_everything(derive_seed(cfg.seed, "torch"))
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    theta0_sha = tensors_sha256(trainable)
    micro = int(cfg.execution.micro_batch_size)
    grad = surrogate.surrogate_gradient(
        model, env.tokenizer, [item["text"] for item in items], spec.refusals, spec.compliances,
        device=env.device, trainable=trainable, micro_batch_size=micro, max_tokens=spec.max_tokens,
    )  # fmt: skip
    panel = panel_record(spec, items)
    common = _common(cfg, env, spec, numerics)
    record = {
        "kind": GRAD_R_KIND,
        "point": "theta0",
        "norm": vectors.norm(grad),
        "theta0_trainable_sha256": theta0_sha,
        "panel": panel,
        "micro_batch_size": micro,
        **common,
        "memory": _memory(device),
        "seconds": {"total": time.perf_counter() - started},
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    provenance = {"surrogate_sha256": spec.sha256, "surrogate_version": spec.version,
                  "panel": panel, "micro_batch_size": micro, "seed": int(cfg.seed),
                  "config_hash": common["config_hash"]}  # fmt: skip
    return record, {"names": [name for name, _ in trainable], "provenance": provenance,
                    "grad_r": grad}  # fmt: skip


# ------------------------------------------------------------------------------- files and stdout


def write_new(path: Path, record: dict) -> None:
    """Write ``record`` to ``path`` atomically; refuse an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; a record is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(partial, path)


def write(out_dir: Path, record: dict, vecs: dict) -> None:
    """The vector files (none for a diverged δ_lin), then the record with their hashes."""
    written = {}
    for kind in ("delta_hat", "delta_lin", "grad_r"):
        tensors = vecs.get(kind)
        if tensors is None:
            continue
        saved = vectors.save_vector(out_dir / FILES[kind], tensors, vecs["names"], kind=kind,
                                    theta0_sha256=record["theta0_trainable_sha256"],
                                    provenance=vecs["provenance"])  # fmt: skip
        written[kind] = {"file": FILES[kind], "sha256": saved["sha256"], "norm": saved["norm"],
                         "names_sha256": saved["names_sha256"]}  # fmt: skip
    record["vectors"] = written
    write_new(out_dir / RECORD, record)


def _fmt(x: float | None) -> str:
    return "none" if x is None else f"{x:.6e}"


def report(record: dict) -> None:
    """Numbers, ids and hashes only."""
    print(f"seed {record['seed']} config_hash {record['config_hash']}")
    if record["kind"] == GRAD_R_KIND:
        print(f"grad_r norm {record['norm']:.9e}")
    else:
        order = record["order"]
        print(f"lr {record['lr']:.6e} steps {record['steps']} epsilon {record['epsilon']:.3e} "
              f"order {order if order == 'stream' else order['order_seed']}")  # fmt: skip
        div = record["divergence"]
        print(f"delta_lin {div['status']} step {div['step']} reason {div['reason']} "
              f"norm {_fmt(div['norm'])}")  # fmt: skip
        print(f"hvp {record['hvp']['mode']} r_readouts {record['r_readouts']}")
        if record["r"] is not None:
            print(f"q_hat {_fmt(record['q_hat'])} q_lin {_fmt(record['q_lin'])} "
                  f"r_theta0 {_fmt(record['r']['theta0'])}")  # fmt: skip
        norms = record["norms"]
        print(f"norms delta_hat {_fmt(norms['delta_hat'])} delta_lin {_fmt(norms['delta_lin'])} "
              f"relative_difference {_fmt(norms['relative_difference'])}")  # fmt: skip
        for row in record["per_step"]:
            print(f"step {row['step']} delta_hat {_fmt(row['delta_hat_norm'])} "
                  f"delta_lin {_fmt(row['delta_lin_norm'])} "
                  f"hdelta_ratio {_fmt(row['hdelta_ratio'])}")  # fmt: skip
        for check in record["checks"]:
            if check.get("mode") == "exact":
                fd = check["fd"]
                print(f"check step {check['step']} exact: fd relative half "
                      f"{_fmt(fd['half']['relative'])} unit {_fmt(fd['unit']['relative'])} "
                      f"double {_fmt(fd['double']['relative'])}")  # fmt: skip
            else:
                print(f"check step {check['step']} half {_fmt(check['half'])} "
                      f"double {_fmt(check['double'])}")  # fmt: skip
        print(f"v0_sha256 {record['v0_sha256']}")
        print(f"seconds {record['seconds']['total']:.1f} passes {record['passes']}")
    for kind, vector in record["vectors"].items():
        print(f"vector {kind} sha256 {vector['sha256']} norm {vector['norm']:.9e}")
    print(f"theta0_trainable_sha256 {record['theta0_trainable_sha256']}")
    print(f"surrogate_sha256 {record['surrogate_sha256']}")


def check_steps_of(values: Sequence[int], steps: int) -> list[int]:
    """The check steps, sorted; each in 2 … N (at step 1 there is no difference)."""
    out = sorted(set(int(v) for v in values))
    bad = [v for v in out if not 2 <= v <= steps]
    if bad:
        raise ValueError(f"--check-steps {bad}: each must lie in 2 … {steps}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--check-steps", type=int, nargs="+", default=list(CHECK_STEPS))
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference-norm", type=float)
    reference.add_argument("--reference-run", type=Path)
    # an edited, sealed or control order: the norm of the seed's fresh F run at the same rate
    reference.add_argument("--reference-fresh-run", type=Path)
    parser.add_argument("--surrogate", type=Path, default=surrogate.SPEC_PATH)
    parser.add_argument("--panel-size", type=int, default=100)
    parser.add_argument("--grad-r-only", action="store_true")
    parser.add_argument("--no-k", action="store_true")  # skip the K readouts (tests)
    parser.add_argument("--hvp", choices=HVP_MODES, default="central")
    parser.add_argument("--hvp-micro-batch-size", type=int)
    parser.add_argument("--no-r", action="store_true")  # skip the surrogate readouts (r, q)
    # the derivative probe: its record at --out-dir/<FILE>, no replay
    parser.add_argument("--probe")
    parser.add_argument("--probe-steps", type=int, nargs="+", default=list(PROBE_STEPS))
    parser.add_argument("--direction-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.probe is not None:
        files = [args.probe]
    else:
        files = [RECORD, *([FILES["grad_r"]] if args.grad_r_only
                           else [FILES["delta_hat"], FILES["delta_lin"]])]  # fmt: skip
    existing = [name for name in files if (args.out_dir / name).exists()
                or vectors.record_path(args.out_dir / name).exists()]  # fmt: skip
    if existing:
        print(f"exists in {args.out_dir}: {' '.join(existing)}")
        return 1
    if args.hvp_micro_batch_size is not None and args.hvp_micro_batch_size < 1:
        print(f"ValueError: --hvp-micro-batch-size {args.hvp_micro_batch_size}: at least 1")
        return 1
    cfg = load_run_config(args.config, args.overrides)  # the protocol: fp32, frozen modules
    try:
        linear_response.require_frozen_adam(cfg)
        check_steps, ref = [], None
        if not args.grad_r_only:
            if args.probe is None:  # the probe takes no check steps and no ε
                check_steps = check_steps_of(args.check_steps, int(cfg.train.steps))
            if not (math.isfinite(args.epsilon) and args.epsilon > 0):
                raise ValueError(f"--epsilon {args.epsilon!r}: a finite positive step is needed")
            ref = reference_of(args, cfg)
    except (ValueError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 1
    numerics = configure_numerics(cfg, cfg.execution.device)  # first: before any CUDA work
    spec = surrogate.load_spec(args.surrogate)
    env = environment(cfg, cfg.execution.device)
    try:
        if args.probe is not None:
            found = probe(cfg, env, numerics, steps=args.probe_steps,
                          direction_seed=args.direction_seed,
                          hvp_micro_batch_size=args.hvp_micro_batch_size,
                          reference=ref)  # fmt: skip
            write_new(args.out_dir / args.probe, found)
            print(
                f"probe {found['seed']} steps {found['steps']} entries {len(found['entries'])} "
                f"restored {found['restored_trainable_sha256'] == found['theta0_trainable_sha256']}"
            )
            return 0
        if args.grad_r_only:
            record, vecs = run_grad_r(cfg, env, spec, numerics, panel_size=args.panel_size)
        else:
            record, vecs = run(cfg, env, spec, numerics, epsilon=args.epsilon,
                               check_steps=check_steps, reference=ref,
                               panel_size=args.panel_size, k_readouts=not args.no_k,
                               hvp_mode=args.hvp,
                               hvp_micro_batch_size=args.hvp_micro_batch_size,
                               r_readouts=not args.no_r)  # fmt: skip
        write(args.out_dir, record, vecs)
    except (ValueError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 1
    report(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
