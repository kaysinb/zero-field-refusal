"""Top curvature of the training loss at θ₀, from Hessian–vector products (study 2, Stage A).

**The loss.** The Hessian is that of the loss SGD steps on: a batch's summed response-token NLL
divided by the constant B × cap, over the trainable parameters only. Plain SGD at rate η is
linearly stable along an eigendirection of curvature λ only if η·λ < 2, so 2/λ_max guides the
screen of SGD rates; it does not certify a rate, which the soft-target control does.

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

**Eigenvalues.** Lanczos with the three-term recurrence and no reorthogonalization keeps three
parameter-sized vectors. Its extreme Ritz values converge to the extreme eigenvalues first;
lost orthogonality only adds spurious copies of converged values. The top Ritz value after each
step is recorded, so convergence is visible. Start vectors are drawn on the CPU from named seeds,
identical on every GPU.
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
) -> Vector:
    """H·v, where H is the Hessian of the mean over ``batches`` of each batch's training loss:
    its summed NLL over the constant B × cap of a run with that batch's size B."""
    out = [torch.zeros_like(p) for p in params]
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


def lanczos(hvp: Callable[[Vector], Vector], start: Vector, steps: int) -> dict[str, Any]:
    """Extreme eigenvalues of a symmetric operator from ``steps`` Lanczos iterations.

    Returns the top and bottom Ritz values, the top Ritz value after each step, and the
    tridiagonal coefficients. Stops early if the Krylov space becomes invariant.
    """
    v, previous, beta = [t.clone() for t in start], None, 0.0
    alphas: list[float] = []
    betas: list[float] = []
    trace: list[float] = []
    for _ in range(steps):
        w = hvp(v)
        alpha = _dot(w, v)
        for i, wi in enumerate(w):
            wi.sub_(v[i], alpha=alpha)
            if previous is not None:
                wi.sub_(previous[i], alpha=beta)
        alphas.append(alpha)
        trace.append(float(_ritz(alphas, betas)[-1]))
        beta = _norm(w)
        if beta <= 1e-12 * max(abs(a) for a in alphas):
            break
        betas.append(beta)
        previous, v = v, [wi.div_(beta) for wi in w]
    ritz = _ritz(alphas, betas[: len(alphas) - 1])
    return {"top": float(ritz[-1]), "bottom": float(ritz[0]), "top_trace": trace,
            "steps": len(alphas), "alphas": alphas, "betas": betas}  # fmt: skip


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
    steps: int,
    seed: int,
) -> dict[str, Any]:
    """Top curvature at θ₀ of single batches at each batch size and of the pooled Hessian.

    Batch b of size B is responses bB … (b+1)B − 1 of ``stream``, the batch a run of that size
    takes at step b. Each single batch is measured from ``starts`` start vectors; the pooled
    Hessian over the first ``pooled_responses`` responses (in batches of the largest size, which
    gives the same Hessian) from ``pooled_starts``.
    """
    if cfg.train.loss.normalization != "constant":
        raise ValueError("The curvature is measured for the constant denominator only")
    params = [p for _, p in trainable]

    def operator(selected: Sequence[Sequence[Sample]]) -> Callable[[Vector], Vector]:
        return lambda v: hessian_vector_product(
            model, params, selected, v, cap=cfg.train.max_response_tokens,
            micro_batch_size=cfg.execution.micro_batch_size, pad_id=pad_id, device=device,
        )  # fmt: skip

    single = []
    for size in batch_sizes:
        for b in range(batches):
            batch = list(stream[b * size : (b + 1) * size])
            if len(batch) != size:
                raise ValueError(f"The stream holds fewer than {(b + 1) * size} responses")
            for s in range(starts):
                result = lanczos(operator([batch]), start_vector(trainable, seed, s), steps)
                single.append({"batch_size": size, "batch": b, "start": s, **result})
    size = max(batch_sizes)
    if pooled_responses % size or len(stream) < pooled_responses:
        raise ValueError(f"pooled_responses must be a multiple of {size} within the stream")
    pooled_batches = [list(stream[i : i + size]) for i in range(0, pooled_responses, size)]
    pooled = [
        {"responses": pooled_responses, "start": s,
         **lanczos(operator(pooled_batches), start_vector(trainable, seed, starts + s), steps)}
        for s in range(pooled_starts)
    ]  # fmt: skip
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
            pooled_starts=settings.pooled_starts, steps=settings.steps, seed=cfg.seed,
        )  # fmt: skip
        run.path("curvature.json").write_text(json.dumps(result, indent=1) + "\n")
        run.summarize({"curvature": result["summary"]})
        return run.dir


def _summary(
    single: Sequence[dict[str, Any]], pooled: Sequence[dict[str, Any]], batch_sizes: Sequence[int]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for size in batch_sizes:
        median = float(np.median([r["top"] for r in single if r["batch_size"] == size]))
        out[f"single_batch_{size}"] = {
            "top_median": median,
            "top_max": max(r["top"] for r in single if r["batch_size"] == size),
            "lr_at_2_over_median": 2 / median,
        }
    top = max(r["top"] for r in pooled)
    out["pooled"] = {"top": top, "lr_at_2_over_top": 2 / top,
                     "bottom": min(r["bottom"] for r in pooled)}  # fmt: skip
    return out


def _ritz(alphas: Sequence[float], betas: Sequence[float]) -> np.ndarray:
    k = len(alphas)
    t = np.diag(np.asarray(alphas, dtype=np.float64))
    off = np.asarray(betas[: k - 1], dtype=np.float64)
    t += np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.eigvalsh(t)


def _dot(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> float:
    return sum((x.double() * y.double()).sum().item() for x, y in zip(xs, ys, strict=True))


def _norm(xs: Sequence[torch.Tensor]) -> float:
    return _dot(xs, xs) ** 0.5
