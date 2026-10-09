"""Stage-0 measurements of the field at θ₀ and of the pipeline's numerical noise.

- **Mean gradient (invariant 9).** At zero field the mean of n independent batch gradients shrinks
  like 1/√n. A field f makes it level off at ‖f‖.
- **Token-count field (A1).** Dividing each batch by its own token count multiplies the
  constant-denominator gradient g_b by c_b = B·L / n_b, so both normalizations come from one
  backward pass. At zero field E[g_b] = 0, so the token-count field is f = E[(c_b − 1)·g_b], a
  control variate with far less noise than c_b·g_b. |f|² is estimated without bias from two
  independent halves of the batches, ⟨d̄_A, d̄_B⟩ = |d̄_A|·mean_B(d_b·u) with u = d̄_A / |d̄_A|,
  with the standard error |d̄_A|·sd_B(d_b·u)/√n. Over N steps the field moves the parameters by
  N·|f| along its direction, and the noise by √N·σ, where σ² is the variance of a training
  gradient along that direction, measured along u. The field is at the noise level if
  N·|f|² < σ² even at the upper end of its uncertainty (|f|² + 2 SE), detected if it exceeds σ²
  at the lower end, and undetermined otherwise (docs/preregistration.md, A1). The same estimate
  for the constant denominator checks that its field is zero.
- **Path discrepancy (invariant 8).** θ₀'s distributions recorded while sampling (generation path,
  KV cache) against the training path on the same samples: per-token KL and the step-0 gradient
  of the soft-target loss relative to the zero-field batch gradient.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel

from .batches import chunks, collate
from .losses import response_logprobs, soft_kl, summed_nll, targets
from .teacher import Sample


def _batch_gradient(
    model: PreTrainedModel,
    params: Sequence[torch.nn.Parameter],
    samples: Sequence[Sample],
    *,
    pad_id: int,
    cap: int,
    denominator: float,
    micro_batch_size: int,
    device: torch.device,
) -> int:
    """Leave the constant-denominator batch gradient in ``.grad``; return the batch token count."""
    for p in params:
        p.grad = None
    tokens = 0
    for chunk in chunks(samples, micro_batch_size):
        batch = collate(chunk, pad_id=pad_id, max_response_tokens=cap, device=device)
        (summed_nll(model, batch) / denominator).backward()
        tokens += batch.num_tokens
    return tokens


def field_decision(
    field_norm_sq: float, standard_error: float, noise_var: float, steps: int
) -> str:
    """``at_noise_level``, ``field_detected`` or ``undetermined`` for N·|f|² against σ²."""
    if steps * (field_norm_sq + 2 * standard_error) < noise_var:
        return "at_noise_level"
    if steps * (field_norm_sq - 2 * standard_error) > noise_var:
        return "field_detected"
    return "undetermined"


def field_measurement(
    model: PreTrainedModel,
    params: Sequence[torch.nn.Parameter],
    batches: Sequence[Sequence[Sample]],
    cfg: DictConfig,
    *,
    pad_id: int,
    device: torch.device,
    steps: int,
) -> dict:
    """Mean-gradient scaling and the fields of both normalizations at the current parameters.

    ``batches`` are fresh teacher batches, split into halves A (first) and B (second). ``steps`` is
    the number of training steps N over which drift and noise are compared.
    """
    cap = cfg.train.max_response_tokens
    denominator = float(cfg.train.batch_size * cap)
    grad = dict(pad_id=pad_id, cap=cap, denominator=denominator,
                micro_batch_size=cfg.execution.micro_batch_size, device=device)  # fmt: skip
    half = len(batches) // 2
    if half < 2:
        raise ValueError("The field measurement needs at least 4 batches")
    # Half A: fp32 sums of g_b (constant) and of d_b = (c_b - 1)·g_b (token count), in place.
    sums = {
        name: [torch.zeros_like(p, dtype=torch.float32) for p in params]
        for name in ("constant", "token_count")
    }
    checkpoints = {2**k for k in range(int(math.log2(half)) + 1)} | {half}
    mean_norm: dict[int, float] = {}
    sq_norm_total = 0.0
    for n, batch in enumerate(batches[:half], start=1):
        tokens = _batch_gradient(model, params, batch, **grad)
        excess = denominator / tokens - 1.0
        for total_c, total_t, p in zip(sums["constant"], sums["token_count"], params, strict=True):
            total_c.add_(p.grad)
            total_t.add_(p.grad, alpha=excess)
        sq_norm_total += sum(p.grad.double().pow(2).sum().item() for p in params)
        if n in checkpoints:
            mean_norm[n] = _norm(sums["constant"]) / n
    mean_a_norm = {}
    for name, totals in sums.items():  # unit directions in place; a zero sum stays zero
        norm = _norm(totals)
        mean_a_norm[name] = norm / half
        if norm > 0:
            for total in totals:
                total.div_(norm)

    # Half B, projected on the half-A directions: field terms and training-gradient noise.
    field_terms = {"constant": [], "token_count": []}
    noise_terms = {"constant": [], "token_count": []}
    for batch in batches[half : 2 * half]:
        tokens = _batch_gradient(model, params, batch, **grad)
        excess = denominator / tokens - 1.0
        for name in sums:
            dot = sum((p.grad * u).sum().item() for p, u in zip(params, sums[name], strict=True))
            noise_terms[name].append(dot)
            field_terms[name].append(dot if name == "constant" else excess * dot)
    for p in params:
        p.grad = None

    out = {
        "batches_per_half": half,
        "steps": steps,
        "mean_gradient_norm": mean_norm,  # invariant 9: ∝ 1/√n at zero field
        "gradient_noise_trace": sq_norm_total / half,  # E|g_b|² = tr Σ at zero field
    }
    for name in sums:
        if mean_a_norm[name] == 0:  # every batch had the same token count: no field at all
            out[name] = {"field_norm_sq": 0.0, "field_norm_sq_standard_error": 0.0,
                         "decision": "at_noise_level"}  # fmt: skip
            continue
        terms = torch.tensor(field_terms[name], dtype=torch.float64)
        noise = torch.tensor(noise_terms[name], dtype=torch.float64)
        field_sq = mean_a_norm[name] * terms.mean().item()
        se = mean_a_norm[name] * terms.std().item() / math.sqrt(half)
        noise_var = noise.var().item()
        out[name] = {
            "field_norm_sq": field_sq,
            "field_norm_sq_standard_error": se,
            "noise_variance_along_field": noise_var,
            "drift_to_noise_squared": steps * field_sq / noise_var,
            "share_of_mean_squared_displacement": steps * field_sq / out["gradient_noise_trace"],
            "decision": field_decision(field_sq, se, noise_var, steps),
        }
    return out


@torch.no_grad()
def path_discrepancy(
    model: PreTrainedModel,
    samples: Sequence[Sample],
    recorded: Sequence[torch.Tensor],
    *,
    pad_id: int,
    micro_batch_size: int,
    device: torch.device,
) -> dict:
    """Per-token KL(generation path ‖ training path) and log-prob differences of sampled tokens."""
    kl, diff = [], []
    for start in range(0, len(samples), micro_batch_size):
        chunk = samples[start : start + micro_batch_size]
        batch = collate(chunk, pad_id=pad_id, max_response_tokens=10**9, device=device)
        train = response_logprobs(model, batch).double()
        gen = torch.cat(list(recorded[start : start + micro_batch_size])).to(device).double()
        kl.append((gen.exp() * (gen - train)).sum(-1))
        picked = targets(batch)[:, None]
        diff.append(train.gather(1, picked)[:, 0] - gen.gather(1, picked)[:, 0])
    kl_all, diff_all = torch.cat(kl), torch.cat(diff)
    return {
        "tokens": kl_all.numel(),
        "kl_mean": kl_all.mean().item(),
        "kl_max": kl_all.max().item(),
        "kl_q99": torch.quantile(kl_all, 0.99).item(),
        "logprob_diff_mean": diff_all.mean().item(),
        "logprob_diff_max_abs": diff_all.abs().max().item(),
    }


def soft_target_gradient_ratio(
    model: PreTrainedModel,
    params: Sequence[torch.nn.Parameter],
    samples: Sequence[Sample],
    recorded: Sequence[torch.Tensor],
    cfg: DictConfig,
    *,
    pad_id: int,
    device: torch.device,
) -> dict:
    """‖∇ soft-target loss‖ / ‖∇ zero-field NLL loss‖ at θ₀ on one batch (invariant 8)."""
    cap = cfg.train.max_response_tokens
    denominator = float(cfg.train.batch_size * cap)
    micro = cfg.execution.micro_batch_size

    def gradient_norm(soft: bool) -> float:
        for p in params:
            p.grad = None
        for start in range(0, len(samples), micro):
            chunk = samples[start : start + micro]
            batch = collate(chunk, pad_id=pad_id, max_response_tokens=cap, device=device)
            if soft:
                target = torch.cat([t[:cap] for t in recorded[start : start + micro]]).to(device)
                loss = soft_kl(model, batch, target)
            else:
                loss = summed_nll(model, batch)
            (loss / denominator).backward()
        norm = math.sqrt(sum(p.grad.double().pow(2).sum().item() for p in params))
        for p in params:
            p.grad = None
        return norm

    soft, nll = gradient_norm(True), gradient_norm(False)
    return {"soft_target_grad_norm": soft, "nll_grad_norm": nll, "ratio": soft / nll}


def _norm(tensors: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(t.pow(2).sum().item() for t in tensors))
