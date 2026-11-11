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

from zero_field_refusal import ladder, linear_response, surrogate, vectors
from zero_field_refusal.config import config_hash
from zero_field_refusal.evaluation import set_point
from zero_field_refusal.models import freeze
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import configure_numerics, device_name, environment
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

Named = Mapping[str, torch.Tensor]
Trainable = Sequence[tuple[str, torch.nn.Parameter]]
Gradient = Callable[[Any], float]  # a batch → its loss, its gradient added to the .grad


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
) -> dict[str, Any]:
    """δ̂ and δ_lin over ``batches`` (module docstring), from the host tensors ``theta0`` and
    ``D`` (by name, over the trainable set), the model's parameters ``trainable`` and
    ``gradient``, one pass of a batch at the model's current point. ``limit`` is δ_lin's norm
    bound (None: only a non-finite value diverges). Without ``feedback`` no difference is taken
    and δ_lin is δ̂ bit for bit. The model is at θ₀ exactly on return. Returns ``delta_hat``,
    ``delta_lin`` (None once diverged), ``per_step``, ``checks``, ``divergence`` (None or its
    step, reason and norms) and ``passes``, the gradient passes taken."""
    names = [name for name, _ in trainable]
    start = [theta0[name] for name in names]

    def zeros() -> dict[str, torch.Tensor]:
        return {name: torch.zeros_like(theta0[name]) for name in names}

    m_hat, d_hat, m_lin, d_lin = zeros(), zeros(), zeros(), zeros()
    product = zeros() if feedback else None
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    divergence, passes, lin_norm = None, 0, 0.0
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
                plus, minus = _product(trainable, start, direction, gradient, batch, epsilon,
                                       product)  # fmt: skip
                passes += 2
                hdelta = vectors.norm(product)
                finite = all(math.isfinite(x) for x in (plus, minus, hdelta))
                row.update(loss_plus=_number(plus), loss_minus=_number(minus))
                if t in check_steps and finite:
                    checks.append(_check(trainable, start, direction, gradient, batch, epsilon,
                                         product, t, hdelta))  # fmt: skip
                    passes += 2 * len(CHECK_FACTORS)
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
            "divergence": divergence, "passes": passes}  # fmt: skip


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
    config's run, all its updates), with the identities that run recorded."""
    if args.reference_norm is not None:
        x = float(args.reference_norm)
        if not (math.isfinite(x) and x > 0):
            raise ValueError(f"--reference-norm {x!r}: a finite positive norm is needed")
        return {"norm": x, "source": "--reference-norm"}
    if args.reference_run is None:
        return None
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The replay of ``cfg``'s seed and order in ``env``: (record, vectors), vectors holding
    ``names``, ``provenance``, ``delta_hat`` and ``delta_lin`` (None when diverged)."""
    started = time.perf_counter()
    device = torch.device(env.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    items = panel_items(env, spec, panel_size)
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
    setup_seconds = time.perf_counter() - started  # the batches, θ₀, v₀ and D

    limit = None if not reference else DIVERGENCE_FACTOR * float(reference["norm"])
    began = time.perf_counter()
    result = replay(
        trainable, theta0, D, batches,
        lambda batch: accumulate_gradient(model, batch, cfg, env)["loss"],
        lr=lr, beta1=beta1, epsilon=epsilon, check_steps=check_steps, limit=limit,
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

    r0, r_hat = r_at(None), r_at(delta_hat)
    r_lin = None if delta_lin is None else r_at(delta_lin)
    with torch.no_grad():
        set_point(trainable, start, start, 0.0)
    final = tensors_sha256(trainable)
    surrogate_seconds = time.perf_counter() - began

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
        "r": {"theta0": _number(r0), "delta_hat": _number(r_hat),
              "delta_lin": None if r_lin is None else _number(r_lin)},
        "q_hat": _number(r_hat - r0),
        "q_lin": None if r_lin is None else _number(r_lin - r0),
        "norms": norms,
        "per_step": result["per_step"],
        "checks": result["checks"],
        "passes": result["passes"],
        "theta0_trainable_sha256": theta0_sha,
        "restored_trainable_sha256": restored,
        "final_trainable_sha256": final,
        "v0_sha256": v0_sha,
        "streams": {"training": training, "preconditioner": preconditioner},
        "panel": panel_record(spec, items),
        "surrogate_micro_batch_size": micro,
        "micro_batch_size": int(cfg.execution.micro_batch_size),
        **common,
        "memory": _memory(device),
        "seconds": {"setup": setup_seconds, "replay": replay_seconds,
                    "surrogate": surrogate_seconds, "total": time.perf_counter() - started},
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip
    provenance = {"seed": int(cfg.seed), "lr": lr, "steps": int(cfg.train.steps), "beta1": beta1,
                  "eps": eps, "epsilon": epsilon, "order": order_rec,
                  "config_hash": common["config_hash"],
                  "streams": record["streams"], "v0_sha256": v0_sha}  # fmt: skip
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
    parser.add_argument("--surrogate", type=Path, default=surrogate.SPEC_PATH)
    parser.add_argument("--panel-size", type=int, default=100)
    parser.add_argument("--grad-r-only", action="store_true")
    args = parser.parse_args(argv)
    files = [RECORD, *([FILES["grad_r"]] if args.grad_r_only
                       else [FILES["delta_hat"], FILES["delta_lin"]])]  # fmt: skip
    existing = [name for name in files if (args.out_dir / name).exists()
                or vectors.record_path(args.out_dir / name).exists()]  # fmt: skip
    if existing:
        print(f"exists in {args.out_dir}: {' '.join(existing)}")
        return 1
    cfg = load_run_config(args.config, args.overrides)  # the protocol: fp32, frozen modules
    try:
        linear_response.require_frozen_adam(cfg)
        check_steps, ref = [], None
        if not args.grad_r_only:
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
        if args.grad_r_only:
            record, vecs = run_grad_r(cfg, env, spec, numerics, panel_size=args.panel_size)
        else:
            record, vecs = run(cfg, env, spec, numerics, epsilon=args.epsilon,
                               check_steps=check_steps, reference=ref,
                               panel_size=args.panel_size)  # fmt: skip
        write(args.out_dir, record, vecs)
    except (ValueError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}")
        return 1
    report(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
