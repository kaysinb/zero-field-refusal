"""Stage-0 measurements of the field at θ₀ and of the pipeline's numerical noise.

- **Mean gradient (invariant 9).** At zero field the mean of n independent batch gradients shrinks
  like 1/√n. A field f makes it level off at ‖f‖.
- **Token-count field (A1).** Dividing each batch by its own token count multiplies the
  constant-denominator gradient g_b by c_b = B·L / n_b, so both normalizations come from one
  backward pass. The field is estimated without the noise bias by splitting the batches into two
  halves: ⟨ḡ_A, ḡ_B⟩ estimates ‖f‖². Its size relative to the noise is measured along the
  direction u = ḡ_A / ‖ḡ_A‖ with the batches of half B, which are independent of u: drift per step
  m = mean(g_b·u), noise per step s = sd(g_b·u). Over N steps the drift is N·m and the noise √N·s;
  the field is at the noise level when √N·|m| < s (docs/preregistration.md, A1).
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
    """Mean-gradient scaling and the token-count field at the current parameters (θ₀).

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
    # fp32 sums of half A for both normalizations: two parameter-sized buffers in all.
    sums = {
        name: [torch.zeros_like(p, dtype=torch.float32) for p in params]
        for name in ("constant", "token_count")
    }
    checkpoints = {2**k for k in range(int(math.log2(half)) + 1)} | {half}
    mean_norm = {"constant": {}, "token_count": {}}
    for n, batch in enumerate(batches[:half], start=1):
        tokens = _batch_gradient(model, params, batch, **grad)
        scale = denominator / tokens  # token-count normalization of the same batch
        for total_c, total_t, p in zip(sums["constant"], sums["token_count"], params, strict=True):
            total_c.add_(p.grad)
            total_t.add_(p.grad, alpha=scale)
        if n in checkpoints:
            for name, totals in sums.items():
                mean_norm[name][n] = _norm(totals) / n
    mean_a_norm = {}
    for name, totals in sums.items():  # turn the sums into unit directions in place
        norm = _norm(totals)
        mean_a_norm[name] = norm / half
        for total in totals:
            total.div_(norm)

    projections = {"constant": [], "token_count": []}
    for batch in batches[half : 2 * half]:
        tokens = _batch_gradient(model, params, batch, **grad)
        dot = {
            name: sum((p.grad * u).sum().item() for p, u in zip(params, sums[name], strict=True))
            for name in sums
        }
        projections["constant"].append(dot["constant"])
        projections["token_count"].append(dot["token_count"] * denominator / tokens)
    for p in params:
        p.grad = None

    out = {"batches_per_half": half, "steps": steps, "mean_gradient_norm": mean_norm}
    for name in sums:
        values = torch.tensor(projections[name], dtype=torch.float64)
        drift, noise = values.mean().item(), values.std().item()
        out[name] = {
            # <mean_A, mean_B> = |mean_A| · (mean_B · u): unbiased for |f|², halves independent.
            "field_norm_sq_unbiased": mean_a_norm[name] * drift,
            "drift_per_step": drift,
            "drift_standard_error": noise / math.sqrt(half),
            "noise_per_step": noise,
            "drift_to_noise_over_steps": math.sqrt(steps) * abs(drift) / noise,
            "at_noise_level": math.sqrt(steps) * abs(drift) < noise,
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


def _unit(tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    norm = _norm(tensors)
    return [t / norm for t in tensors]
