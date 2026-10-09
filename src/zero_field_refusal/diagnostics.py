"""Stage-0 measurements of fields at θ₀ and of the pipeline's numerical noise.

**Estimating a field.** A field f is the mean of per-batch vectors x_b (batch gradients, or a
control variate). |f|² is estimated without bias from two independent halves A and B of the
batches as ⟨x̄_A, x̄_B⟩, in three passes: sum A; sum B while projecting each x_b on
u_A = x̄_A/|x̄_A|; project A again on u_B. Both projections give ⟨x̄_A, x̄_B⟩. Its standard
error counts the spread of both halves,
SE² = (|x̄_A|²·Var_B(x·u_A) + |x̄_B|²·Var_A(x·u_B)) / h,
which slightly overstates the variance (the noise-noise term enters twice), so decisions err
toward caution. Over N training steps a field moves the parameters by N·|f| along its direction
and the noise by √N·σ, where σ² is the variance of a training gradient along that direction,
measured along u_A and u_B. The comparison is N·|f|²/σ².

- **Mean gradient (invariant 9).** At zero field the mean of n batch gradients shrinks like 1/√n.
- **Token-count field (A1).** Dividing each batch by its own token count multiplies the
  constant-denominator gradient g_b by c_b = B·L / n_b, so both normalizations come from one
  backward pass. At zero field E[g_b] = 0, so the token-count field is f = E[(c_b − c̄)·g_b] for
  any constant c̄, a control variate with far less noise than c_b·g_b; c̄ is the mean of c_b. The
  field is at the noise level if N·|f|² < σ² even at |f|² + 2 SE, detected if it exceeds σ² at
  |f|² − 2 SE, and undetermined otherwise (docs/preregistration.md, A1). A sufficient criterion
  needs no gradients: N·Var(c_b) < 1 bounds N·|f|²/σ² below 1 (Cauchy–Schwarz with E[g_b] = 0).
  The same estimate for the constant denominator checks that its field is zero.
- **Path field (invariant 8).** The distributions θ₀ produced while sampling (generation path p,
  KV cache) differ slightly from the training path's (q). The student then trains toward p: a
  residual field f, the mean soft-target gradient. Write the training gradient as the soft-target
  gradient plus n_b = Σ_t (p_t − e_{y_t})·J_t, a sum of martingale differences. For any direction
  v, f·v = E[Σ_t E_p[(q/p − 1)·(J_t v)]], so Cauchy–Schwarz gives
  N·(f·v)²/Var(n_b·v) ≤ N·X with X = E[Σ_{t in a batch} χ²(q_t ‖ p_t)].
  The bound needs only the two paths' distributions, holds in every direction and in every
  linear metric, Adam's preconditioned one included, and does not depend on estimating the
  direction of a weak field. The check requires N·(X̂ + 2 SE) ≤ the preregistered bound over
  several batches. Gradient-based estimates of N·|f|²/σ² in the Euclidean and Adam metrics are
  reported alongside; their σ² is measured along an estimated direction, which a weak field leaves
  dominated by noise and biased toward noisy directions, so they do not decide.
- **Path discrepancy.** Per-token KL between the two paths' distributions on the same samples,
  and the one-batch soft-target gradient ratio, as descriptive numbers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel

from .batches import collate
from .losses import response_logprobs, soft_kl, summed_nll, targets
from .teacher import Sample

# A view of one batch: (field vector, its scalar factor, noise vector). The field contribution
# is factor × field; the noise vector is the training gradient whose spread is the noise.
View = tuple[Sequence[torch.Tensor], float, Sequence[torch.Tensor]]


@dataclass(frozen=True)
class FieldEstimate:
    field_norm_sq: float
    standard_error: float
    noise_variance: float  # of a training gradient along the field's direction
    batches_per_half: int

    def ratio(self, steps: int, shift: float = 0.0) -> float:
        """N·(|f|² + shift·SE)/σ²."""
        return steps * (self.field_norm_sq + shift * self.standard_error) / self.noise_variance

    def as_dict(self, steps: int) -> dict:
        return {**asdict(self), "drift_to_noise": self.ratio(steps),
                "drift_to_noise_upper": self.ratio(steps, 2.0),
                "drift_to_noise_lower": self.ratio(steps, -2.0)}  # fmt: skip


def split_half_fields(
    half: int,
    views: Callable[[int], Mapping[str, View]],
    names: Sequence[str],
    like: Sequence[torch.nn.Parameter],
    *,
    after_a: Callable[[int, Mapping[str, list[torch.Tensor]]], None] | None = None,
) -> dict[str, FieldEstimate | None]:
    """The three-pass estimate for each named view over batches 0..2·half−1.

    ``views(b)`` must return the same vectors whenever it is called for batch b (pass 1 and pass 3
    both visit half A). A view whose half sum is exactly zero has no field: None.
    Memory: two parameter-sized fp32 buffers per view.
    """
    sums = {name: [[torch.zeros_like(p, dtype=torch.float32) for p in like] for _ in range(2)]
            for name in names}  # fmt: skip
    for b in range(half):  # pass 1: sum A
        for name, (field, factor, _) in views(b).items():
            if name in sums:
                for total, x in zip(sums[name][0], field, strict=True):
                    total.add_(x, alpha=factor)
        if after_a is not None:
            after_a(b + 1, {name: s[0] for name, s in sums.items()})
    norm_a = {name: _unit_in_place(s[0]) for name, s in sums.items()}

    proj_b = {name: ([], []) for name in names}  # pass 2: sum B, project on u_A
    for b in range(half, 2 * half):
        for name, (field, factor, noise) in views(b).items():
            if name not in sums:
                continue
            u_a, total_b = sums[name][0], sums[name][1]
            proj_b[name][0].append(factor * _dot(field, u_a))
            proj_b[name][1].append(_dot(noise, u_a))
            for total, x in zip(total_b, field, strict=True):
                total.add_(x, alpha=factor)
    norm_b = {name: _unit_in_place(s[1]) for name, s in sums.items()}

    proj_a = {name: ([], []) for name in names}  # pass 3: project A on u_B
    for b in range(half):
        for name, (field, factor, noise) in views(b).items():
            if name in sums:
                proj_a[name][0].append(factor * _dot(field, sums[name][1]))
                proj_a[name][1].append(_dot(noise, sums[name][1]))

    out: dict[str, FieldEstimate | None] = {}
    for name in names:
        if norm_a[name] == 0 or norm_b[name] == 0:
            out[name] = None
            continue
        mean_a, mean_b = norm_a[name] / half, norm_b[name] / half
        pb = torch.tensor(proj_b[name][0], dtype=torch.float64)
        pa = torch.tensor(proj_a[name][0], dtype=torch.float64)
        noise = (torch.tensor(proj_b[name][1], dtype=torch.float64).var()
                 + torch.tensor(proj_a[name][1], dtype=torch.float64).var()) / 2  # fmt: skip
        field_sq = (mean_a * pb.mean() + mean_b * pa.mean()).item() / 2
        se = math.sqrt((mean_a**2 * pb.var().item() + mean_b**2 * pa.var().item()) / half)
        out[name] = FieldEstimate(field_sq, se, noise.item(), half)
    return out


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
    soft_targets: Sequence[torch.Tensor] | None = None,
) -> None:
    """Leave the batch gradient of Σ loss / denominator in ``.grad``: NLL, or the soft-target KL."""
    for p in params:
        p.grad = None
    for start in range(0, len(samples), micro_batch_size):
        chunk = samples[start : start + micro_batch_size]
        batch = collate(chunk, pad_id=pad_id, max_response_tokens=cap, device=device)
        if soft_targets is None:
            loss = summed_nll(model, batch)
        else:
            target = torch.cat([t[:cap] for t in soft_targets[start : start + micro_batch_size]])
            loss = soft_kl(model, batch, target.to(device))
        (loss / denominator).backward()


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
    """Mean-gradient scaling and the fields of both normalizations at the current parameters (θ₀).

    ``batches`` are fresh teacher batches; the first and second halves are A and B.
    """
    cap = cfg.train.max_response_tokens
    denominator = float(cfg.train.batch_size * cap)
    half = len(batches) // 2
    if half < 2:
        raise ValueError("The field measurement needs at least 4 batches")
    # c_b = B·L / n_b from the token counts alone, centered at their mean (a constant, which keeps
    # the control variate's expectation at the field; using all batches scales it by 1 − 1/2h).
    counts = [sum(min(len(s.response_ids), cap) for s in batch) for batch in batches[: 2 * half]]
    factors = torch.tensor([denominator / n for n in counts], dtype=torch.float64)
    excess = (factors - factors.mean()).tolist()
    count_bound = steps * factors.var().item()
    squared_norms: list[float] = []

    def views(b: int) -> dict[str, View]:
        _batch_gradient(model, params, batches[b], pad_id=pad_id, cap=cap,
                        denominator=denominator,
                        micro_batch_size=cfg.execution.micro_batch_size, device=device)  # fmt: skip
        g = [p.grad for p in params]
        if b < half and len(squared_norms) < half:
            squared_norms.append(sum(x.double().pow(2).sum().item() for x in g))
        return {"constant": (g, 1.0, g), "token_count": (g, excess[b], g)}

    checkpoints = {2**k for k in range(int(math.log2(half)) + 1)} | {half}
    mean_norm: dict[int, float] = {}

    def after_a(n: int, totals: Mapping[str, list[torch.Tensor]]) -> None:
        if n in checkpoints:
            mean_norm[n] = _norm(totals["constant"]) / n

    estimates = split_half_fields(half, views, ["constant", "token_count"], params,
                                  after_a=after_a)  # fmt: skip
    for p in params:
        p.grad = None
    trace = sum(squared_norms) / half  # E|g_b|² = tr Σ at zero field
    out = {"batches_per_half": half, "steps": steps, "mean_gradient_norm": mean_norm,
           "gradient_noise_trace": trace}  # fmt: skip
    for name, estimate in estimates.items():
        if estimate is None:  # every batch had the same token count: no field at all
            out[name] = {
                "field_norm_sq": 0.0,
                "standard_error": 0.0,
                "decision": "at_noise_level",
                "decision_basis": "equal_token_counts",
            }
            continue
        out[name] = estimate.as_dict(steps) | {
            "share_of_mean_squared_displacement": steps * estimate.field_norm_sq / trace,
            "decision": field_decision(
                estimate.field_norm_sq, estimate.standard_error, estimate.noise_variance, steps
            ),  # fmt: skip
            "decision_basis": "estimate",
        }
    token_count = out["token_count"]
    token_count["count_variance_bound"] = count_bound
    if count_bound < 1 and token_count["decision_basis"] == "estimate":
        token_count |= {"decision": "at_noise_level", "decision_basis": "count_variance_bound"}
    return out


@torch.no_grad()
def path_chi_square(
    model: PreTrainedModel,
    samples: Sequence[Sample],
    distributions: Sequence[torch.Tensor],
    cfg: DictConfig,
    *,
    pad_id: int,
    device: torch.device,
    steps: int,
) -> dict:
    """Per batch, Σ over response tokens of χ²(q_train ‖ p_generation); the bound N·(X̂ + 2 SE)."""
    size = cfg.train.batch_size
    cap = cfg.train.max_response_tokens
    totals = []
    for b in range(len(samples) // size):
        total = 0.0
        for start in range(b * size, (b + 1) * size, cfg.execution.micro_batch_size):
            stop = min(start + cfg.execution.micro_batch_size, (b + 1) * size)
            batch = collate(samples[start:stop], pad_id=pad_id, max_response_tokens=cap,
                            device=device)  # fmt: skip
            log_q = response_logprobs(model, batch).double()
            log_p = torch.cat([d[:cap] for d in distributions[start:stop]]).to(device).double()
            # χ²(q‖p) = Σ_y q²/p − 1, computed in log space.
            total += (torch.exp(2 * log_q - log_p).sum(-1) - 1.0).sum().item()
        totals.append(total)
    values = torch.tensor(totals, dtype=torch.float64)
    mean, se = values.mean().item(), values.std().item() / math.sqrt(len(totals))
    return {"batches": len(totals), "chi_square_per_batch": mean, "standard_error": se,
            "bound_upper": steps * (mean + 2 * se)}  # fmt: skip


def path_field_measurement(
    model: PreTrainedModel,
    params: Sequence[torch.nn.Parameter],
    samples: Sequence[Sample],
    distributions: Sequence[torch.Tensor],
    cfg: DictConfig,
    *,
    pad_id: int,
    device: torch.device,
    steps: int,
    preconditioner: Sequence[torch.Tensor] | None = None,
    eps: float = 1e-8,
) -> FieldEstimate | None:
    """The residual field of the path discrepancy: the mean soft-target gradient at θ₀.

    With ``preconditioner`` (the θ₀ second-moment estimate v̂), field and noise are measured in
    Adam's metric: every gradient is divided elementwise by √v̂ + ε.
    """
    cap = cfg.train.max_response_tokens
    denominator = float(cfg.train.batch_size * cap)
    size = cfg.train.batch_size
    half = len(samples) // size // 2
    if half < 2:
        raise ValueError("The path-field measurement needs at least 4 batches")
    scale = None if preconditioner is None else [1.0 / (v.sqrt() + eps) for v in preconditioner]
    soft = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    grad = dict(pad_id=pad_id, cap=cap, denominator=denominator,
                micro_batch_size=cfg.execution.micro_batch_size, device=device)  # fmt: skip

    def views(b: int) -> dict[str, View]:
        batch = samples[b * size : (b + 1) * size]
        _batch_gradient(model, params, batch, soft_targets=distributions[b * size : (b + 1) * size],
                        **grad)  # fmt: skip
        for buffer, p in zip(soft, params, strict=True):
            buffer.copy_(p.grad)
        _batch_gradient(model, params, batch, **grad)
        noise = [p.grad for p in params]
        if scale is not None:
            for s, n, k in zip(soft, noise, scale, strict=True):
                s.mul_(k)
                n.mul_(k)
        return {"path": (soft, 1.0, noise)}

    estimate = split_half_fields(half, views, ["path"], params)["path"]
    for p in params:
        p.grad = None
    return estimate


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


def _dot(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> float:
    return sum((x.float() * y).sum().item() for x, y in zip(xs, ys, strict=True))


def _norm(tensors: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(t.double().pow(2).sum().item() for t in tensors))


def _unit_in_place(tensors: Sequence[torch.Tensor]) -> float:
    """Divide by the joint norm in place (a zero vector stays zero); return the norm."""
    norm = _norm(tensors)
    if norm > 0:
        for t in tensors:
            t.div_(norm)
    return norm
