"""Top curvature of the training loss at θ₀, from Hessian–vector products (study 2, Stage A).

**The loss.** The Hessian is that of the loss SGD steps on: a batch's summed response-token NLL
divided by the constant B × cap, over the trainable parameters only.

**What 2/λ means.** Gradient descent on the expected loss is linearly stable along an
eigendirection of curvature λ only if η·λ < 2, so 2/λ_max of the pooled Hessian describes the mean
dynamics. It is not the mean-square stability threshold of SGD, which also depends on how the
batch Hessians spread around their mean, and the medians of single batches do not certify a rate
either. The curvature only guides the screen of SGD rates; the soft-target control accepts them.

**Batch and pooled Hessians.** With a constant denominator, a batch's Hessian is
H_b = Σ_{i∈b} H_i / (B·cap), and the mean of H_b over batches that partition N responses is
Σ_i H_i / (N·cap) whatever B is. So:
- a single batch's Hessian measures the curvature one SGD step meets, with its sampling spread
  (larger at B = 5 than at B = 20);
- the Hessian pooled over many responses estimates the expected curvature, the same for both batch
  sizes. At θ₀ the responses are θ₀'s own samples, so its expectation is the Fisher information
  divided by the cap.

**Products.** H·v is the gradient of ∇L·v (double backward), accumulated over micro-batches;
it is exact, since the loss is a sum over micro-batches. Attention runs on PyTorch's math kernel,
the only SDPA kernel with a second derivative.

**Eigenvalues.** Lanczos with the three-term recurrence and no reorthogonalization keeps two
parameter-sized vectors besides the product's own. Its extreme Ritz values converge to the
extreme eigenvalues first; lost orthogonality only adds spurious copies of converged values.
Each start iterates at least ``min_steps`` and stops once the top Ritz value's residual is
within a tolerance of it (or at ``max_steps``, unconverged); both traces are recorded, and a
measurement counts as converged only if its independent starts converged and agree. Start
vectors are drawn on the CPU from named seeds, identical on every GPU.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PreTrainedModel

from .batches import chunks, collate
from .losses import summed_nll
from .models import freeze
from .runtime import Environment, StreamPrefix
from .seeding import derive_seed
from .teacher import Sample
from .tracking import Run

Vector = list[torch.Tensor]


def hessian_vector_product(
    model: PreTrainedModel,
    params: Sequence[torch.nn.Parameter],
    batches: Sequence[Sequence[Sample]],
    vector: Sequence[torch.Tensor],
    *,
    cap: int,
    micro_batch_size: int,
    pad_id: int,
    device: torch.device,
    out: Vector | None = None,
) -> Vector:
    """H·v, where H is the Hessian of the mean over ``batches`` of each batch's training loss:
    its summed NLL over the constant B × cap of a run with that batch's size B. Added into
    ``out`` if given (saving a parameter-sized vector), else into zeros."""
    out = [torch.zeros_like(p) for p in params] if out is None else out
    with sdpa_kernel(SDPBackend.MATH):
        for batch in batches:
            denominator = len(batch) * cap * len(batches)
            for chunk in chunks(batch, micro_batch_size):
                rows = collate(chunk, pad_id=pad_id, max_response_tokens=cap, device=device)
                loss = summed_nll(model, rows) / denominator
                grads = torch.autograd.grad(loss, params, create_graph=True)
                dot = sum((g * v).sum() for g, v in zip(grads, vector, strict=True))
                for total, part in zip(out, torch.autograd.grad(dot, params), strict=True):
                    total.add_(part)
                del grads, dot
    return out


def start_vector(
    trainable: Sequence[tuple[str, torch.nn.Parameter]], seed: int, start: int
) -> Vector:
    """A unit vector with iid normal entries, drawn per parameter on the CPU from named seeds."""
    vector = []
    for name, param in trainable:
        generator = torch.Generator().manual_seed(
            derive_seed(seed, "curvature", start, name, list(param.shape))
        )
        vector.append(torch.randn(tuple(param.shape), generator=generator).to(param))
    norm = _norm(vector)
    return [v.div_(norm) for v in vector]


def lanczos(
    hvp: Callable[[Vector, Vector], Vector],
    start: Vector,
    *,
    min_steps: int,
    max_steps: int,
    tolerance: float,
) -> dict[str, Any]:
    """The top eigenvalue of a symmetric operator by Lanczos, until its residual is small.

    After step k, the top Ritz pair (θ, y) of the tridiagonal T_k has the residual
    ‖H·y − θ·y‖ = β_{k+1}·|s_k|, s the eigenvector of T_k for θ and β_{k+1} the next
    off-diagonal (the standard estimate, without another product). Iteration stops when it falls
    to ``tolerance``·|θ| (``converged``), but not before ``min_steps``, or after ``max_steps``.
    A small residual puts θ near an eigenvalue, not necessarily the top one: a random start in
    a space of n parameters overlaps the top eigenvector by ≈ 1/√n, and the Ritz value can rest
    near the second eigenvalue, residual small, before the top one emerges. The minimum number
    of steps and independent starts that agree guard against that; they do not prove it.
    ``hvp(v, out)`` adds H·v into ``out`` and returns it. Each step forms H·v − β·v_prev in the
    previous vector's memory, then takes α from it (Paige's ordering), so two vectors stay alive
    besides the product's own. ``start`` (a unit vector) is consumed: the list is emptied.
    Returns θ, its residual, the bottom Ritz value, both traces and the coefficients.
    """
    v, previous, beta = list(start), None, 0.0
    start.clear()
    alphas: list[float] = []
    betas: list[float] = []
    tops: list[float] = []
    residuals: list[float] = []
    converged = False
    for _ in range(max_steps):
        if previous is None:
            w = hvp(v, [torch.zeros_like(t) for t in v])
        else:
            w = hvp(v, [t.mul_(-beta) for t in previous])  # H·v − β·v_prev, in place
            previous = None
        alpha = _dot(w, v)
        for wi, vi in zip(w, v, strict=True):
            wi.sub_(vi, alpha=alpha)
        alphas.append(alpha)
        beta = _norm(w)
        values, vectors = _ritz(alphas, betas)
        tops.append(float(values[-1]))
        residuals.append(beta * abs(float(vectors[-1, -1])))
        converged = residuals[-1] <= tolerance * abs(tops[-1])
        invariant = beta <= 1e-12 * max(abs(a) for a in alphas)  # the Krylov space is complete
        if converged and (len(alphas) >= min_steps or invariant):
            break
        betas.append(beta)
        previous, v = v, [wi.div_(beta) for wi in w]
    values, _ = _ritz(alphas, betas[: len(alphas) - 1])
    return {"top": tops[-1], "residual": residuals[-1], "converged": converged,
            "bottom": float(values[0]), "steps": len(alphas), "top_trace": tops,
            "residual_trace": residuals, "alphas": alphas, "betas": betas}  # fmt: skip


def curvature_measurement(
    model: PreTrainedModel,
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    stream: Sequence[Sample],
    cfg: DictConfig,
    *,
    pad_id: int,
    device: torch.device,
    batch_sizes: Sequence[int],
    batches: int,
    starts: int,
    pooled_responses: int,
    pooled_starts: int,
    min_steps: int,
    max_steps: int,
    tolerance: float,
    seed: int,
) -> dict[str, Any]:
    """Top curvature at θ₀ of single batches at each batch size and of the pooled Hessian.

    Batch b of size B is responses bB … (b+1)B − 1 of ``stream``, the batch a run of that size
    takes at step b. Each single batch is measured from ``starts`` start vectors; the pooled
    Hessian over the first ``pooled_responses`` responses (in batches of the largest size, which
    gives the same Hessian) from ``pooled_starts``. A measurement's top is the largest over its
    starts; it counts as converged when every start converged and the starts agree within
    ``tolerance`` of that top.
    """
    if cfg.train.loss.normalization != "constant":
        raise ValueError("The curvature is measured for the constant denominator only")
    params = [p for _, p in trainable]

    def measure(selected: Sequence[Sequence[Sample]], first_start: int, count: int) -> dict:
        def hvp(v: Vector, out: Vector) -> Vector:
            return hessian_vector_product(
                model, params, selected, v, cap=cfg.train.max_response_tokens,
                micro_batch_size=cfg.execution.micro_batch_size, pad_id=pad_id, device=device,
                out=out,
            )  # fmt: skip

        steps = {"min_steps": min_steps, "max_steps": max_steps, "tolerance": tolerance}
        runs = [lanczos(hvp, start_vector(trainable, seed, first_start + s), **steps)
                for s in range(count)]  # fmt: skip
        tops = [r["top"] for r in runs]
        agree = max(tops) - min(tops) <= tolerance * abs(max(tops))
        converged = agree and all(r["converged"] for r in runs)
        return {"top": max(tops), "starts_agree": agree, "converged": converged, "starts": runs}

    single = []
    for size in batch_sizes:
        for b in range(batches):
            batch = list(stream[b * size : (b + 1) * size])
            if len(batch) != size:
                raise ValueError(f"The stream holds fewer than {(b + 1) * size} responses")
            single.append({"batch_size": size, "batch": b, **measure([batch], 0, starts)})
    size = max(batch_sizes)
    if pooled_responses % size or len(stream) < pooled_responses:
        raise ValueError(f"pooled_responses must be a multiple of {size} within the stream")
    pooled_batches = [list(stream[i : i + size]) for i in range(0, pooled_responses, size)]
    pooled = {"responses": pooled_responses, **measure(pooled_batches, starts, pooled_starts)}
    return {"single": single, "pooled": pooled, "summary": _summary(single, pooled, batch_sizes)}


def run_curvature(cfg: DictConfig, env: Environment, *, run_id: str | None = None) -> Path:
    """Measure the curvature at θ₀ on the appendable training stream of the config's seed, with
    the settings under ``curvature``; writes ``curvature.json`` and the summary to a run."""
    if cfg.run_type != "curvature":
        raise ValueError("run_curvature needs run_type=curvature")
    settings = cfg.curvature
    with Run(cfg, run_id=run_id) as run:
        prefix = StreamPrefix(cfg, env, cfg.seed)
        needed = max(settings.batches * max(settings.batch_sizes), settings.pooled_responses)
        stream = prefix.take(0, needed)
        run.record_inputs({"prompt_manifest_sha256": env.prompt_manifest_sha256,
                           "streams": {"training": prefix.identity(needed)}})  # fmt: skip
        model = env.new_model()
        trainable = freeze(model, cfg.train.freeze)
        result = curvature_measurement(
            model, trainable, stream, cfg, pad_id=env.pad_id, device=env.device,
            batch_sizes=list(settings.batch_sizes), batches=settings.batches,
            starts=settings.starts, pooled_responses=settings.pooled_responses,
            pooled_starts=settings.pooled_starts, min_steps=settings.min_steps,
            max_steps=settings.max_steps, tolerance=settings.tolerance, seed=cfg.seed,
        )  # fmt: skip
        run.path("curvature.json").write_text(json.dumps(result, indent=1) + "\n")
        run.summarize({"curvature": result["summary"]})
        return run.dir


def _summary(
    single: Sequence[dict[str, Any]], pooled: dict[str, Any], batch_sizes: Sequence[int]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for size in batch_sizes:
        measured = [r for r in single if r["batch_size"] == size]
        median = float(np.median([r["top"] for r in measured]))
        out[f"single_batch_{size}"] = {
            "top_median": median,
            "top_max": max(r["top"] for r in measured),
            "lr_at_2_over_median": 2 / median,
            "not_converged": sum(not r["converged"] for r in measured),
        }
    out["pooled"] = {"top": pooled["top"], "lr_at_2_over_top": 2 / pooled["top"],
                     "converged": pooled["converged"],
                     "bottom": min(r["bottom"] for r in pooled["starts"])}  # fmt: skip
    return out


def _ritz(alphas: Sequence[float], betas: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (ascending) and eigenvectors of the tridiagonal T with these coefficients."""
    k = len(alphas)
    t = np.diag(np.asarray(alphas, dtype=np.float64))
    off = np.asarray(betas[: k - 1], dtype=np.float64)
    t += np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigh(t)


def _dot(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> float:
    return sum((x.double() * y.double()).sum().item() for x, y in zip(xs, ys, strict=True))


def _norm(xs: Sequence[torch.Tensor]) -> float:
    return _dot(xs, xs) ** 0.5
