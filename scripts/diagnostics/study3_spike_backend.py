#!/usr/bin/env python3
"""Study 3, candidate 1's numerical qualification: do the production attention backend and MATH
agree at a recorded gradient jump, and does the jump repeat on another host?

One target is one recorded trajectory of one launcher run: its prefix is replayed from θ₀ as
``fa_control_diagnostic.run_twin`` trains (fp32, or the float64 reference's arithmetic), each step
held to the recorded row. A departure ends the target (``state: not_reproduced``, exit 3) with
nothing compared. With ``--no-gate``, as on another host, every row is taken whatever the record
says, a departure is only recorded (``first_departure``, ``largest_departure``, exit 3 at the end),
and inputs other than recorded do not stop it. At each step of ``--measure``, on the same
parameters, batch, v₀ and optimizer state, and in the trajectory's arithmetic, this records:

- the batch gradient under the production backend (as training runs it) and under MATH, each
  twice: their difference as a vector (relative to the production gradient, cosine, largest entry,
  the kinds of module holding it), and each backend against its own second pass;
- the update each gradient would make, by the optimizer's own step, then undone: its norm and the
  two updates' difference;
- the gradient vector of each micro-batch, as training composes them, under both backends, and how
  their sum stands to the batch gradient;
- the gradient norm of each response alone, under both backends (a different shape from training's:
  a description, compared by norm only).

The trajectory then goes on with the production gradient, so its later rows test that nothing here
disturbed it. Nothing is written but ``--out``; no run, campaign or qualification record is made.
The output holds numbers, ids and hashes, no text of a prompt or a response. These are comparisons
of two fp32 forward and backward computations: they do not establish which is right.

``--derivative STEPS`` checks, at those steps of the replay, whether the computed gradient G matches
the loss's slope: on the same parameters, batch, normalization, arithmetic and forward mode
(gradients enabled, as training runs it), along the unit direction of G and along a fixed Gaussian
direction, it evaluates central differences on the grid ``EPSILONS`` (displacement norms from 10⁻²
down by halving), with the actual fp32 displacements: L(θ₊) − L(θ₋) against Gᵀ(θ₊ − θ₋). The
unshifted loss is repeated for its noise; ``derivative_reading`` reads each direction by the
runbook's rule. The parameters are restored exactly (the hash in the record).

``--optimizer STEPS`` recomputes each of those steps' update in float64 from the saved gradient and
optimizer state by the registered formulas (m̂, v₀ or v, ε, the rate) and compares it with the
update the optimizer made, with where the update's mass lies. ``--anatomy STEPS`` records where the
gradient lies (by layer, by response, by token at the logits) and the extremes of intermediate
values (hidden states, query and key projections, logits) under the production backend.
``--v0-census RUN_DIR …`` estimates v₀ for each run's seed as its launcher did and records its
census, with no replay.

``--full LAUNCHER_DIR`` replays one case on another host instead: from a launcher run made there, it
runs the harness's baseline with its monitor to step 25 and, with ``--reference-rows``, the
reference. It checks that the launcher run, the record and the harness have the same inputs and
environment, and records the rows, K, the stop, the endpoint hashes and D₂₅ against the record;
``differences`` names what does not agree, and exit 3 follows from any.

Usage (through ``project-run``): ``study3_spike_backend.py --run-dir RUN --rows launcher|FILE:KEY
--measure 2,3,4 --end 5 --precision fp32|float64 --out FILE [--no-gate]``; ``--measure ""`` compares
nothing and records the rows alone.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import resource
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from zero_field_refusal import code_identity, runtime
from zero_field_refusal import fa_control_diagnostic as fcd
from zero_field_refusal import qualification as q
from zero_field_refusal.batches import chunks, collate
from zero_field_refusal.losses import loss_denominator, response_logits, summed_nll
from zero_field_refusal.losses import targets as target_ids
from zero_field_refusal.optim import ADAM_MODES, build_optimizer
from zero_field_refusal.schedule import _norm as grad_norm
from zero_field_refusal.seeding import derive_seed, seed_everything
from zero_field_refusal.tensor_hash import tensors_sha256
from zero_field_refusal.tracking import git_state, hardware, package_versions
from zero_field_refusal.train import accumulate_gradient

PRODUCTION, MATH = "production", "MATH"
ROW_TOLERANCE = 1e-3  # a replayed row against the recorded one, relative, loss and gradient norm
NOT_REPRODUCED = 3  # the exit of a target whose prefix departed from its record
TOP_GROUPS = 5
EPSILONS = tuple(1e-2 * 0.5**k for k in range(11))  # displacement norms, 1e-2 … 9.8e-6, halving
DIRECTION_SEED, DIRECTION_KEY = 20261103, "directional-derivative"  # the independent direction
NOISE_REPEATS = 3  # the unshifted loss, for its noise
RESOLVE, AGREE, MISMATCH, STABLE, BAND = 100.0, 0.05, 0.25, 0.20, 3  # the reading (runbook)
KEEP_K_STAR = 6.0  # ``twin_setup`` asks for it; only D_twin reads it, and nothing here measures D


def backend(name: str) -> contextlib.AbstractContextManager:
    """Training's own attention backends (no context), or MATH."""
    return sdpa_kernel(SDPBackend.MATH) if name == MATH else contextlib.nullcontext()


def _cpu(vector: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [v.detach().cpu().clone() for v in vector]


def batch_gradient(
    twins: fcd.Twins, batch: Sequence[Any], cfg: Any, precision: str
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    """The batch's row and gradient, as ``run_twin`` computes them; the parameters keep no
    gradient."""
    params = twins.params
    for p in params:
        p.grad = None
    if precision == "fp32":
        stats = accumulate_gradient(twins.student, batch, cfg, twins.env)
        stats["grad_norm"] = grad_norm([p.grad for p in params])
        return stats, fcd.take_gradients(params)
    with fcd.Float64Gradients(params) as sums:
        stats = fcd.nll_loss_float64(twins.student, batch, cfg, twins.env)
    grads = sums.take()
    stats["grad_norm"] = math.sqrt(sum(g.pow(2).sum().item() for g in grads))
    return stats, grads


def take_step(
    optimizer: Any, params: Sequence[torch.nn.Parameter], grads: Sequence[torch.Tensor],
    precision: str,
) -> None:  # fmt: skip
    """The optimizer's step on ``grads``; the parameters keep no gradient."""
    if precision == "fp32":
        for p, g in zip(params, grads, strict=True):
            p.grad = g
        optimizer.step()
        for p in params:
            p.grad = None
    else:
        optimizer.step(grads)


def _clone(value: Any) -> Any:
    return value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value


def optimizer_state(optimizer: Any, precision: str) -> Any:
    """A host copy of the optimizer's state."""
    if precision == "fp32":
        return [{k: _clone(v) for k, v in optimizer.state[p].items()}
                for g in optimizer.param_groups for p in g["params"]]  # fmt: skip
    return {"m": _cpu(optimizer.m), "v": _cpu(optimizer.v), "step_count": optimizer.step_count}


def restore_optimizer(optimizer: Any, state: Any, precision: str) -> None:
    if precision == "fp32":
        ordered = [p for g in optimizer.param_groups for p in g["params"]]
        for p, kept in zip(ordered, state, strict=True):
            current = optimizer.state[p]
            for key in [k for k in current if k not in kept]:  # a first step made them
                del current[key]
            for key, value in kept.items():
                if isinstance(value, torch.Tensor) and isinstance(current.get(key), torch.Tensor):
                    current[key].copy_(value)
                else:
                    current[key] = value.to(p.device) if isinstance(value, torch.Tensor) else value
    else:
        for target, value in zip(optimizer.m, state["m"], strict=True):
            target.copy_(value)
        for target, value in zip(optimizer.v, state["v"], strict=True):
            target.copy_(value)
        optimizer.step_count = state["step_count"]


def compare(
    names: Sequence[str], a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
) -> dict[str, Any]:
    """``a`` against ``b`` (host tensors), relative to ``a``: an exact count of differing entries,
    the norms, the difference, the cosine, and the kinds of module holding the difference."""
    a_sq = b_sq = dot = diff_sq = 0.0
    largest, differing = 0.0, 0
    groups: dict[str, float] = {}
    for name, x, y in zip(names, a, b, strict=True):
        x64, y64 = x.double(), y.double()
        d = x64 - y64
        part = d.pow(2).sum().item()
        a_sq += x64.pow(2).sum().item()
        b_sq += y64.pow(2).sum().item()
        dot += (x64 * y64).sum().item()
        diff_sq += part
        differing += int(torch.count_nonzero(d).item())
        largest = max(largest, d.abs().max().item())
        kind = fcd.group_keys(name)[0]
        groups[kind] = groups.get(kind, 0.0) + part
    top = sorted(groups.items(), key=lambda item: item[1], reverse=True)[:TOP_GROUPS]
    return {
        "norm_a": math.sqrt(a_sq),
        "norm_b": math.sqrt(b_sq),
        "difference_norm": math.sqrt(diff_sq),
        "relative": math.sqrt(diff_sq / a_sq) if a_sq > 0 else None,
        "cosine": dot / math.sqrt(a_sq * b_sq) if a_sq > 0 and b_sq > 0 else None,
        "largest_entry": largest,
        "differing_entries": differing,
        "difference_by_kind": {k: v / diff_sq for k, v in top} if diff_sq > 0 else {},
    }


def _term_gradient(
    twins: fcd.Twins, samples: Sequence[Any], denominator: float, cfg: Any, precision: str
) -> tuple[float, int, list[torch.Tensor]]:
    """(its loss term, its tokens, its gradient of term / denominator) on the device, in the
    trajectory's arithmetic, as ``accumulate_gradient`` or ``nll_loss_float64`` compute a
    micro-batch."""
    params = twins.params
    for p in params:
        p.grad = None
    cap = cfg.train.max_response_tokens
    rows = collate(samples, pad_id=twins.env.pad_id, max_response_tokens=cap,
                   device=twins.env.device)  # fmt: skip
    if precision == "fp32":
        term = summed_nll(twins.student, rows)
        (term / denominator).backward()
        return term.item(), rows.num_tokens, fcd.take_gradients(params)
    with fcd.Float64Gradients(params) as sums:
        term = F.cross_entropy(response_logits(twins.student, rows).double(), target_ids(rows),
                               reduction="sum")  # fmt: skip
        (term / denominator).backward()
    return term.item(), rows.num_tokens, sums.take()


def parts(
    twins: fcd.Twins, batch: Sequence[Any], cfg: Any, precision: str,
    production: Sequence[torch.Tensor],
) -> dict[str, Any]:  # fmt: skip
    """The micro-batches' gradient vectors under both backends, compared; their sum under
    production against the batch gradient ``production``; each response's gradient norm alone."""
    names = twins.names
    cap = cfg.train.max_response_tokens
    denominator = loss_denominator(cfg.train.loss, batch, cap)
    micro = int(cfg.execution.micro_batch_size)
    micro_batches = []
    total: list[torch.Tensor] | None = None
    for index, chunk in enumerate(chunks(batch, micro)):
        terms, vectors = {}, {}
        for name in (PRODUCTION, MATH):
            with backend(name):
                term, tokens, grads = _term_gradient(twins, chunk, denominator, cfg, precision)
            terms[name] = term
            vectors[name] = _cpu(grads)
            del grads
        if total is None:
            total = [g.double() for g in vectors[PRODUCTION]]
        else:
            for sum_, g in zip(total, vectors[PRODUCTION], strict=True):
                sum_.add_(g.double())
        micro_batches.append({"index": index, "tokens": tokens,
                              "term": terms[PRODUCTION], "math_term": terms[MATH],
                              **compare(names, vectors[PRODUCTION], vectors[MATH])})  # fmt: skip
        del vectors
    assert total is not None
    of_batch = compare(names, production, total)
    del total
    responses = []
    for index, sample in enumerate(batch):
        entry: dict[str, Any] = {"index": index}
        for name in (PRODUCTION, MATH):
            with backend(name):
                term, tokens, grads = _term_gradient(twins, [sample], denominator, cfg, precision)
            entry[name] = {"term": term, "grad_norm": grad_norm(grads)}
            entry["tokens"] = tokens
            del grads
        responses.append(entry)
    return {"micro_batches": micro_batches, "sum_against_batch": of_batch, "responses": responses}


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kilobytes on Linux


def measure(
    twins: fcd.Twins, batch: Sequence[Any], cfg: Any, optimizer: Any, precision: str,
    production: Sequence[torch.Tensor],
) -> dict[str, Any]:  # fmt: skip
    """What this file records at one step; ``production`` is the step's own gradient on the host,
    kept by the caller. The parameters and the optimizer are as they were afterwards (the hash in
    the record). What is compared is reduced to numbers and released before the next phase: the
    memory budget is the runbook's."""
    names, params = twins.names, twins.params
    before = tensors_sha256(twins.trainable)

    with backend(MATH):
        math_stats, grads = batch_gradient(twins, batch, cfg, precision)
    m_first = _cpu(grads)
    del grads
    gradient = compare(names, production, m_first)
    _, grads = batch_gradient(twins, batch, cfg, precision)
    production_again = compare(names, production, _cpu(grads))
    del grads
    with backend(MATH):
        _, grads = batch_gradient(twins, batch, cfg, precision)
    math_again = compare(names, m_first, _cpu(grads))
    del grads

    theta = _cpu(params)
    state = optimizer_state(optimizer, precision)
    after: dict[str, list[torch.Tensor]] = {}
    for name, vector in ((PRODUCTION, production), (MATH, m_first)):
        take_step(optimizer, params, [g.to(p.device) for g, p in zip(vector, params, strict=True)],
                  precision)  # fmt: skip
        after[name] = _cpu(params)
        for p, t in zip(params, theta, strict=True):
            p.data.copy_(t)
        restore_optimizer(optimizer, state, precision)
    del state, m_first
    update = compare(
        names,
        (a.double() - t.double() for a, t in zip(after[PRODUCTION], theta, strict=True)),
        (a.double() - t.double() for a, t in zip(after[MATH], theta, strict=True)),
    )  # fmt: skip
    del after, theta
    gc.collect()

    by_part = parts(twins, batch, cfg, precision, production)
    for p in params:
        p.grad = None
    return {
        "math_row": math_stats,
        "gradient": gradient,
        "production_again": production_again,
        "math_again": math_again,
        "update": update,
        "parts": by_part,
        "parameters_unchanged": tensors_sha256(twins.trainable) == before,
        "peak_rss_gb": _rss_gb(),
    }


def batch_loss(twins: fcd.Twins, batch: Sequence[Any], cfg: Any, precision: str) -> float:
    """The batch's loss as the trajectory's arithmetic computes it (``accumulate_gradient`` or
    ``nll_loss_float64``): Σ term/denominator over the micro-batches, the forward run with gradients
    enabled as in training, the graph then dropped."""
    cap = cfg.train.max_response_tokens
    denominator = loss_denominator(cfg.train.loss, batch, cap)
    total = 0.0
    with torch.enable_grad():
        for chunk in chunks(batch, int(cfg.execution.micro_batch_size)):
            rows = collate(chunk, pad_id=twins.env.pad_id, max_response_tokens=cap,
                           device=twins.env.device)  # fmt: skip
            if precision == "fp32":
                term = summed_nll(twins.student, rows)
            else:
                term = F.cross_entropy(response_logits(twins.student, rows).double(),
                                       target_ids(rows), reduction="sum")  # fmt: skip
            total += term.item() / denominator
            del term
    return total


def unit(vectors: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], float]:
    """The vectors scaled to unit global norm (float64 sums), as fp32 host tensors, and the norm."""
    norm = math.sqrt(sum(v.double().pow(2).sum().item() for v in vectors))
    return [(v.double() / norm).float() for v in vectors], norm


def independent_direction(twins: fcd.Twins) -> list[torch.Tensor]:
    """The fixed Gaussian direction, written before any run: seed ``DIRECTION_SEED``."""
    return [fcd.gaussian(name, tuple(p.shape), DIRECTION_SEED, DIRECTION_KEY)
            for name, p in twins.trainable]  # fmt: skip


def derivative_reading(points: Sequence[Mapping[str, Any]], noise: float) -> dict[str, Any]:
    """The runbook's reading of one direction's grid. A point is resolvable when |Gᵀ(θ₊ − θ₋)| is at
    least ``RESOLVE`` times the noise; it agrees when |ΔL − Gᵀ(θ₊ − θ₋)| ≤ ``AGREE``·|Gᵀ(θ₊ − θ₋)|,
    and mismatches at ``MISMATCH`` or more. A band is ``BAND`` neighboring ε, all resolvable.
    ``agreement``: a band that agrees throughout. Else ``mismatch``: a band that mismatches
    throughout with ΔL/Gᵀ(θ₊ − θ₋) within ``STABLE`` of its band's median. Else ``inconclusive``."""
    flags = []
    for point in points:
        predicted, measured = point["predicted"], point["delta_loss"]
        resolvable = abs(predicted) >= RESOLVE * noise and abs(predicted) > 0
        error = abs(measured - predicted) / abs(predicted) if predicted else math.inf
        flags.append({"resolvable": resolvable, "agrees": resolvable and error <= AGREE,
                      "mismatches": resolvable and error >= MISMATCH,
                      "ratio": measured / predicted if predicted else None})  # fmt: skip
    windows = [flags[i : i + BAND] for i in range(len(flags) - BAND + 1)]
    for window in windows:
        if all(f["agrees"] for f in window):
            return {"reading": "agreement", "flags": flags}
    for window in windows:
        if all(f["mismatches"] for f in window):
            ratios = sorted(f["ratio"] for f in window)
            middle = ratios[len(ratios) // 2]
            if all(abs(r - middle) <= STABLE * abs(middle) for r in ratios):
                return {"reading": "mismatch", "flags": flags}
    return {"reading": "inconclusive", "flags": flags}


def derivative(
    twins: fcd.Twins, batch: Sequence[Any], cfg: Any, precision: str,
    production: Sequence[torch.Tensor], row_loss: float,
) -> dict[str, Any]:  # fmt: skip
    """The directional-derivative check at one step; ``production`` is the step's gradient on the
    host. The parameters are as they were afterwards (the hash in the record)."""
    params = twins.params
    before = tensors_sha256(twins.trainable)
    theta = _cpu(params)
    base = [batch_loss(twins, batch, cfg, precision) for _ in range(NOISE_REPEATS)]
    noise = max(max(base) - min(base), abs(base[0]) * 2.0**-23)
    gradient = [g.double() for g in production]
    directions = {"gradient": unit(production), "independent": unit(independent_direction(twins))}
    out: dict[str, Any] = {"base_losses": base, "row_loss": row_loss, "noise": noise,
                           "epsilons": list(EPSILONS), "directions": {}}  # fmt: skip
    for name, (u, scale) in directions.items():
        slope = sum((g * v.double()).sum().item() for g, v in zip(gradient, u, strict=True))
        points = []
        for eps in EPSILONS:
            predicted = displacement = 0.0
            minus = []
            for p, t, v, g in zip(params, theta, u, gradient, strict=True):
                plus_t, minus_t = t + eps * v, t - eps * v  # fp32 on the host: the actual points
                d = plus_t.double() - minus_t.double()
                predicted += (g * d).sum().item()
                displacement += d.pow(2).sum().item()
                p.data.copy_(plus_t.to(p.device))
                minus.append(minus_t)
            plus_loss = batch_loss(twins, batch, cfg, precision)
            for p, m in zip(params, minus, strict=True):
                p.data.copy_(m.to(p.device))
            del minus
            minus_loss = batch_loss(twins, batch, cfg, precision)
            points.append({"epsilon": eps, "displacement_norm": math.sqrt(displacement),
                           "loss_plus": plus_loss, "loss_minus": minus_loss,
                           "delta_loss": plus_loss - minus_loss,
                           "predicted": predicted})  # fmt: skip
        for p, t in zip(params, theta, strict=True):
            p.data.copy_(t.to(p.device))
        out["directions"][name] = {"norm_before_unit": scale, "slope": slope, "points": points,
                                   **derivative_reading(points, noise)}  # fmt: skip
    for p in params:
        p.grad = None
    out["parameters_unchanged"] = tensors_sha256(twins.trainable) == before
    out["peak_rss_gb"] = _rss_gb()
    return out


def optimizer_state_of(optimizer: Any, params: Sequence[torch.nn.Parameter]) -> dict[str, Any]:
    """The optimizer's moments before a step, on the host, with its settings."""
    if isinstance(optimizer, fcd.Float64Adam):
        return {"mode": optimizer.mode, "lr": optimizer.lr, "eps": optimizer.eps,
                "beta1": optimizer.beta1, "beta2": optimizer.beta2, "step": optimizer.step_count,
                "m": _cpu(optimizer.m), "v": _cpu(optimizer.v)}  # fmt: skip
    group = optimizer.param_groups[0]
    beta1, beta2 = group["betas"]
    state = [optimizer.state[p] for p in params]
    return {"mode": optimizer.second_moment, "lr": group["lr"], "eps": group["eps"],
            "beta1": beta1, "beta2": beta2, "step": state[0].get("step", 0),
            "m": [_clone(s["exp_avg"]) if "exp_avg" in s else torch.zeros(p.shape)
                  for s, p in zip(state, params, strict=True)],
            "v": [_clone(s["exp_avg_sq"]) if "exp_avg_sq" in s else torch.zeros(p.shape)
                  for s, p in zip(state, params, strict=True)]}  # fmt: skip


def optimizer_check(
    names: Sequence[str], before: Mapping[str, Any], grads: Sequence[torch.Tensor],
    theta: Sequence[torch.Tensor], after: Sequence[torch.Tensor],
) -> dict[str, Any]:  # fmt: skip
    """The step's update recomputed in float64 by the registered formulas against the update made:
    m ← β₁m + (1 − β₁)g; v ← β₂v + (1 − β₂)g² (pre-warmed and adaptive; frozen keeps v₀);
    d = √v + ε (adaptive: √(v/(1 − β₂ᵗ)) + ε); Δθ = −lr/(1 − β₁ᵗ) · m/d. With where the update's
    mass lies: by layer, by the size of d, and its largest entries."""
    t = int(before["step"]) + 1
    b1, b2, lr, eps, mode = (before[k] for k in ("beta1", "beta2", "lr", "eps", "mode"))
    diff_sq = ref_sq = round_sq = 0.0
    largest = 0.0
    by_layer: dict[str, float] = {}
    by_denominator = {"below_10eps": 0.0, "below_100eps": 0.0, "below_1000eps": 0.0}
    top: list[tuple[float, str, int, float, float, float]] = []
    for name, g, m, v, t0, t1 in zip(names, grads, before["m"], before["v"], theta, after,
                                     strict=True):  # fmt: skip
        g64, m64, v64 = g.double(), m.double(), v.double()
        m_new = b1 * m64 + (1 - b1) * g64
        v_new = v64 if mode == "frozen" else b2 * v64 + (1 - b2) * g64 * g64
        root = v_new.sqrt() / ((1 - b2**t) ** 0.5) if mode == "adaptive" else v_new.sqrt()
        denominator = root + eps
        formula = -lr / (1 - b1**t) * m_new / denominator
        made = t1.double() - t0.double()
        d = made - formula
        rounded = (t0.double() + formula).float().double() - t0.double()  # the formula's fp32 point
        round_sq += (made - rounded).pow(2).sum().item()
        diff_sq += d.pow(2).sum().item()
        ref_sq += formula.pow(2).sum().item()
        largest = max(largest, d.abs().max().item())
        mass = made.pow(2)
        key = fcd.group_keys(name)[1]
        by_layer[key] = by_layer.get(key, 0.0) + mass.sum().item()
        for label, factor in (("below_10eps", 10), ("below_100eps", 100), ("below_1000eps", 1000)):
            by_denominator[label] += mass[root < factor * eps].sum().item()
        flat = made.abs().reshape(-1)
        k = min(5, flat.numel())
        values, indices = torch.topk(flat, k)
        for value, index in zip(values.tolist(), indices.tolist(), strict=True):
            top.append((value, name, index, g64.reshape(-1)[index].item(),
                        m_new.reshape(-1)[index].item(),
                        root.reshape(-1)[index].item()))  # fmt: skip
    total = sum(by_layer.values())
    top.sort(reverse=True)
    return {
        "mode": mode, "step": t, "update_norm": math.sqrt(total),
        "formula_norm": math.sqrt(ref_sq), "relative_difference": math.sqrt(diff_sq / ref_sq)
        if ref_sq else None, "largest_entry_difference": largest,
        "relative_to_rounded_formula": math.sqrt(round_sq / ref_sq) if ref_sq else None,
        "share_by_layer": dict(sorted(((k, v / total) for k, v in by_layer.items()),
                                      key=lambda kv: -kv[1])[:10]) if total else {},
        "share_by_root_v": {k: v / total for k, v in by_denominator.items()} if total else {},
        "largest_entries": [{"magnitude": a, "parameter": n, "index": i, "gradient": gi,
                             "first_moment": mi, "root_v": ri} for a, n, i, gi, mi, ri in top[:10]],
    }  # fmt: skip


def anatomy(twins: fcd.Twins, batch: Sequence[Any], cfg: Any, precision: str,
            production: Sequence[torch.Tensor]) -> dict[str, Any]:  # fmt: skip
    """Where the gradient lies and how large the intermediate values are, under the production
    backend: the gradient's share by layer; each response's gradient norm alone; by token, the
    logits' gradient ‖softmax − onehot‖/denominator with its NLL; the largest magnitudes, non-finite
    counts and values above 1e30 of every decoder layer's output and query and key projection,
    with the largest query and key head norms; the logits' extremes."""
    names = twins.names
    by_layer: dict[str, float] = {}
    for name, g in zip(names, production, strict=True):
        key = fcd.group_keys(name)[1]
        by_layer[key] = by_layer.get(key, 0.0) + g.double().pow(2).sum().item()
    total = sum(by_layer.values())
    model = twins.student
    config = model.config
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // heads))
    stats: dict[str, dict[str, float]] = {}

    def hook(label: str, n_heads: int | None):
        def record(module, inputs, output):
            x = (output[0] if isinstance(output, tuple) else output).detach()
            entry = stats.setdefault(label, {"max_abs": 0.0, "nonfinite": 0, "above_1e30": 0,
                                             "max_head_norm": 0.0})  # fmt: skip
            finite = torch.isfinite(x)
            entry["nonfinite"] += int((~finite).sum())
            values = x[finite].abs()
            if values.numel():
                entry["max_abs"] = max(entry["max_abs"], values.max().item())
                entry["above_1e30"] += int((values > 1e30).sum())
            if n_heads:
                norms = x.float().reshape(*x.shape[:-1], n_heads, head_dim).norm(dim=-1)
                entry["max_head_norm"] = max(entry["max_head_norm"], norms.max().item())

        return record

    handles = []
    for name, module in model.named_modules():
        if name.endswith("self_attn.q_proj"):
            handles.append(module.register_forward_hook(hook(name, heads)))
        elif name.endswith("self_attn.k_proj"):
            handles.append(module.register_forward_hook(hook(name, kv_heads)))
        elif name.split(".")[-2:-1] == ["layers"] or (
            name.count(".") >= 1 and name.rsplit(".", 1)[-1].isdigit()
        ):
            handles.append(module.register_forward_hook(hook(name, None)))
    cap = cfg.train.max_response_tokens
    denominator = loss_denominator(cfg.train.loss, batch, cap)
    tokens, logit_extremes = [], {"max": -math.inf, "min": math.inf}
    try:
        with torch.no_grad():
            for c, chunk in enumerate(chunks(batch, int(cfg.execution.micro_batch_size))):
                rows = collate(chunk, pad_id=twins.env.pad_id, max_response_tokens=cap,
                               device=twins.env.device)  # fmt: skip
                logits = response_logits(model, rows)
                logit_extremes["max"] = max(logit_extremes["max"], logits.max().item())
                logit_extremes["min"] = min(logit_extremes["min"], logits.min().item())
                y = target_ids(rows)
                logp = torch.log_softmax(logits.double() if precision != "fp32" else logits, -1)
                nll = -logp.gather(1, y[:, None])[:, 0]
                onehot = torch.zeros_like(logp).scatter_(1, y[:, None], 1.0)
                grad_norm_tok = (logp.exp() - onehot).norm(dim=-1) / denominator
                position = 0
                for r, length in enumerate(rows.lengths):
                    for j in range(length):
                        tokens.append((grad_norm_tok[position].item(), nll[position].item(),
                                       c * int(cfg.execution.micro_batch_size) + r, j,
                                       int(y[position])))  # fmt: skip
                        position += 1
                del logits, logp, onehot
    finally:
        for h in handles:
            h.remove()
    mass = sum(t[0] ** 2 for t in tokens)
    tokens.sort(reverse=True)
    responses = []
    for index, sample in enumerate(batch):
        _, n, grads = _term_gradient(twins, [sample], denominator, cfg, precision)
        responses.append({"index": index, "tokens": n, "grad_norm": grad_norm(grads)})
        del grads
    return {
        "gradient_share_by_layer": dict(sorted(((k, v / total) for k, v in by_layer.items()),
                                               key=lambda kv: -kv[1])[:10]) if total else {},
        "largest_layer_share": max(by_layer.values()) / total if total else None,
        "responses": responses,
        "tokens_logit_gradient_mass": mass,
        "largest_tokens": [{"logit_gradient": g, "nll": n, "response": r, "position": j,
                            "token_id": tok} for g, n, r, j, tok in tokens[:10]],
        "largest_token_share": tokens[0][0] ** 2 / mass if mass else None,
        "intermediates": stats,
        "nonfinite": sum(v["nonfinite"] for v in stats.values()),
        "above_1e30": sum(v["above_1e30"] for v in stats.values()),
        "logits": logit_extremes,
    }  # fmt: skip


def seed_census(env: Any, cfg: Any, run_dir: Path) -> dict[str, Any]:
    """v₀ for a run's seed, estimated as its launcher did, and its census."""
    student, trainable = fcd._model(env, cfg)
    names, params = [n for n, _ in trainable], [p for _, p in trainable]
    v0, _ = fcd.second_moment(cfg, env, student, params)
    meta = json.loads((run_dir / "meta.json").read_text())
    digest = tensors_sha256(list(zip(names, v0, strict=True)))
    census = fcd.v0_census(names, [v.cpu() for v in v0], float(cfg.optimizer.eps))
    census.pop("per_tensor", None)
    del student, trainable, params, v0
    gc.collect()
    return {"run_dir": str(run_dir), "v0_sha256": digest,
            "as_recorded": digest == (meta.get("inputs") or {}).get("v0_sha256"),
            "census": census}  # fmt: skip


def v0_census(args: argparse.Namespace) -> int:
    """``--v0-census``: v₀ estimated for each run's seed as its launcher did, and its census."""
    if args.out.exists():
        sys.exit(f"{args.out} exists")
    out: dict[str, Any] = {"provenance": provenance(args.v0_census[0], args), "seeds": {}}
    numerics = None
    for run_dir in args.v0_census:
        cfg = fcd.run_config(run_dir)
        if numerics is None:
            numerics = runtime.configure_numerics(cfg, args.device)
        env = runtime.environment(cfg, args.device)
        out["seeds"][str(cfg.seed)] = seed_census(env, cfg, run_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        write(args.out, {**out, "numerics_setup": numerics, "state": "running"})
        print(f"STEP v0 seed {cfg.seed} as recorded {out['seeds'][str(cfg.seed)]['as_recorded']}")
    write(args.out, {**out, "numerics_setup": numerics, "state": "measured"})
    return 0


def departure(row: Mapping[str, Any], recorded: Mapping[str, Any]) -> float:
    """The larger relative difference of the loss and the gradient norm."""
    return max(abs(row[k] - recorded[k]) / max(abs(recorded[k]), 1e-30)
               for k in ("loss", "grad_norm"))  # fmt: skip


def replay(
    cfg: Any, twins: fcd.Twins, recorded: Sequence[Mapping[str, Any]], *, measure_at: Sequence[int],
    end: int, precision: str, publish: Any = lambda record: None, gate: bool = True,
    derivative_at: Sequence[int] = (), optimizer_at: Sequence[int] = (),
    anatomy_at: Sequence[int] = (),
) -> dict[str, Any]:  # fmt: skip
    """Steps 0 … ``end`` − 1 from θ₀, as ``run_twin`` trains. See the module's docstring."""
    env, params, size = twins.env, twins.params, int(cfg.train.batch_size)
    twins.reset()
    seed_everything(derive_seed(cfg.seed, "torch"))
    fcd.reset_peak(env.device)
    v0 = twins.v0 if ADAM_MODES.get(cfg.optimizer.name, "adaptive") != "adaptive" else None
    optimizer: Any
    if precision == "float64":
        optimizer = fcd.Float64Adam(params, cfg.optimizer, v0)
    else:
        optimizer = build_optimizer(cfg.optimizer, params, second_moment=v0)
    record: dict[str, Any] = {"state": "running", "rows": [], "measured": {}, "derivatives": {},
                              "optimizer": {}, "anatomy": {},
                              "first_departure": None, "largest_departure": 0.0}  # fmt: skip
    by_step = {int(r["step"]): r for r in recorded}
    for step in range(end):
        batch = twins.prefix.take(step * size, (step + 1) * size)
        stats, grads = batch_gradient(twins, batch, cfg, precision)
        off = departure(stats, by_step[step])
        record["rows"].append({"step": step, **stats, "departure": off,
                               "recorded_grad_norm": by_step[step]["grad_norm"]})  # fmt: skip
        record["largest_departure"] = max(record["largest_departure"], off)
        if off > ROW_TOLERANCE and record["first_departure"] is None:
            record["first_departure"] = step
        if (gate and off > ROW_TOLERANCE) or not math.isfinite(stats["grad_norm"]):
            record.update(state="not_reproduced", at=step)
            publish(record)
            return record
        if step in measure_at or step in derivative_at or step in anatomy_at:
            host = _cpu(grads)  # room on the device while the checks run
            device = [g.device for g in grads]
            del grads
            if step in anatomy_at:
                started = time.perf_counter()
                record["anatomy"][str(step)] = anatomy(twins, batch, cfg, precision, host)
                record["anatomy"][str(step)]["seconds"] = time.perf_counter() - started
                for p in params:
                    p.grad = None
                publish(record)
            if step in derivative_at:
                started = time.perf_counter()
                record["derivatives"][str(step)] = derivative(twins, batch, cfg, precision, host,
                                                              stats["loss"])  # fmt: skip
                record["derivatives"][str(step)]["seconds"] = time.perf_counter() - started
                publish(record)
            if step in measure_at:
                started = time.perf_counter()
                record["measured"][str(step)] = measure(twins, batch, cfg, optimizer, precision,
                                                        host)  # fmt: skip
                record["measured"][str(step)]["seconds"] = time.perf_counter() - started
                publish(record)
            grads = [g.to(d) for g, d in zip(host, device, strict=True)]
            del host
        if step in optimizer_at:
            kept, theta, host_grads = (
                optimizer_state_of(optimizer, params),
                _cpu(params),
                _cpu(grads),
            )
        take_step(optimizer, params, grads, precision)
        del grads
        if step in optimizer_at:
            record["optimizer"][str(step)] = optimizer_check(twins.names, kept, host_grads, theta,
                                                             _cpu(params))  # fmt: skip
            del kept, theta, host_grads
            gc.collect()
            publish(record)
    record.update(state="measured", endpoint_sha256=tensors_sha256(twins.trainable),
                  **fcd.peak_memory(env.device))  # fmt: skip
    twins.reset()
    publish(record)
    return record


def _departures(rows: Sequence[Mapping[str, Any]], recorded: Sequence[Mapping[str, Any]]) -> dict:
    """Each row against the record: every recorded step must be there and finite as the record is.
    A row past the record's end is not compared (a baseline continues past a declared stop)."""
    by_step = {int(r["step"]): r for r in rows}
    off: dict[int, float] = {}
    missing, nonfinite = [], []
    for wanted in recorded:
        step = int(wanted["step"])
        row = by_step.get(step)
        if row is None:
            missing.append(step)
        elif not all(math.isfinite(row[k]) for k in ("loss", "grad_norm")):
            nonfinite.append(step)
        else:
            off[step] = departure(row, wanted)
    first = next((s for s in sorted(off) if off[s] > ROW_TOLERANCE), None)
    return {"first_departure": first, "largest_departure": max(off.values(), default=None),
            "steps_compared": len(off), "missing_steps": missing, "nonfinite_steps": nonfinite,
            "differs": first is not None or bool(missing) or bool(nonfinite)}  # fmt: skip


def _close(a: Any, b: Any) -> bool:
    """Equal, or both finite numbers within the rows' tolerance."""
    if a == b:
        return True
    if not all(isinstance(v, int | float) and math.isfinite(v) for v in (a, b)):
        return False
    return abs(a - b) <= ROW_TOLERANCE * max(abs(b), 1e-30)


def _lines(path: Path) -> list[dict[str, Any]]:
    return fcd.read_jsonl(path) if path.is_file() else []


def run_outcome(run_dir: Path) -> dict[str, Any]:
    """A launcher run's end: its reason and step, its K readings, K at step 25, its checkpoint."""
    terminal = json.loads((run_dir / "terminal.json").read_text())
    inputs = json.loads((run_dir / "meta.json").read_text()).get("inputs") or {}
    checkpoint = inputs.get("endpoint_checkpoint")
    evaluations = _lines(run_dir / "evaluations.jsonl")
    return {"reason": terminal["stop"]["reason"], "steps": terminal["steps"],
            "monitor": {int(r["step"]): r["K_monitor"] for r in _lines(run_dir / "monitor.jsonl")},
            "k_25": evaluations[-1].get("K_benign") if evaluations else None,
            "checkpoint_sha256": checkpoint.get("sha256") if isinstance(checkpoint, Mapping)
            else None}  # fmt: skip


def outcome_differences(here: Mapping[str, Any], recorded: Mapping[str, Any]) -> list[str]:
    """Where a launcher run's end differs from the record's."""
    out = [f"{key}: {here[key]} against {recorded[key]}" for key in ("reason", "steps")
           if here[key] != recorded[key]]  # fmt: skip
    if set(here["monitor"]) != set(recorded["monitor"]):
        out.append(f"K's steps: {sorted(here['monitor'])} against {sorted(recorded['monitor'])}")
    out += [f"K at step {s}: {here['monitor'][s]} against {v}"
            for s, v in recorded["monitor"].items()
            if s in here["monitor"] and not _close(here["monitor"][s], v)]  # fmt: skip
    if not _close(here["k_25"], recorded["k_25"]):
        out.append(f"K at step 25: {here['k_25']} against {recorded['k_25']}")
    return out


IDENTITY_INPUTS = (
    "prompt_manifest_sha256",
    "theta0_trainable_sha256",
    "v0_sha256",
    "numerics_setup",
)
ENVIRONMENT = ("hardware", "packages", "uv_lock_sha256", "python")


def identity_problems(
    meta: Mapping[str, Any], original: Mapping[str, Any], harness: Mapping[str, Any],
) -> list[str]:  # fmt: skip
    """Where this host's launcher run, the recorded one and this harness's setup are not the same
    computation or environment: the config, the inputs, the streams, the numerics, the code's
    identity, the driver, the libraries. ``harness``: its ``theta0_trainable_sha256``,
    ``v0_sha256``, ``numerics_setup`` and the ``training`` stream's identity recomputed for each
    run's count (``stream_here``, ``stream_recorded``)."""
    out = []
    if meta.get("config_hash") != original.get("config_hash"):
        out.append("config hash")
    ours, theirs = meta.get("inputs") or {}, original.get("inputs") or {}
    out += [f"launcher {key}" for key in IDENTITY_INPUTS if ours.get(key) != theirs.get(key)]
    out += [f"harness {key}" for key in ("theta0_trainable_sha256", "v0_sha256", "numerics_setup")
            if harness.get(key) != theirs.get(key)]  # fmt: skip
    a, b = ours.get("streams") or {}, theirs.get("streams") or {}
    for name in sorted(set(a) | set(b)):
        if name == "training":  # a stopped run's prefix: the same hash at its own count, below
            continue
        if (a.get(name) or {}).get("sha256") != (b.get(name) or {}).get("sha256"):
            out.append(f"stream {name}")
    if harness.get("stream_here") != (a.get("training") or {}).get("sha256"):
        out.append("harness training stream against this launcher's")
    if harness.get("stream_recorded") != (b.get("training") or {}).get("sha256"):
        out.append("harness training stream against the recorded launcher's")
    out += [f"environment {key}" for key in ENVIRONMENT if meta.get(key) != original.get(key)]
    if (meta.get("git") or {}).get("commit") != (original.get("git") or {}).get("commit"):
        out.append("environment commit")
    image = [(m.get("launch") or {}).get("container_image") for m in (meta, original)]
    if image[0] != image[1]:
        out.append("environment image")
    return out


def full_case(
    cfg: Any, twins: fcd.Twins, shared: Mapping[str, Any], *, recorded: Sequence[Mapping[str, Any]],
    checkpoints: Path, launcher_checkpoint: Path | None = None,
    reference_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:  # fmt: skip
    """One case: the baseline with its monitor, as ``qualification._baseline`` runs it; its declared
    instability, which ``run_twin`` does not stop at; and, given ``reference_rows``, the reference
    as ``qualification.run_trajectory`` runs it. D₂₅ among whichever ended at step 25."""
    twins.reset()
    run = fcd.run_twin(cfg, twins, eval_at=[], monitor=shared["monitor"])
    run["instability"] = q.harness_instability(cfg, run)
    run["against_record"] = _departures(run["rows"], recorded)
    out: dict[str, Any] = {"baseline": run, "comparisons": {}}
    targets: dict[str, Path] = {}
    if launcher_checkpoint is not None:
        targets["launcher"] = launcher_checkpoint
    if run["stop"] is None and int(run["steps"]) == int(cfg.train.steps):
        run["endpoint_sha256"] = tensors_sha256(twins.trainable)
        if launcher_checkpoint is not None:
            out["comparisons"]["launcher~baseline"] = twins.divergence(launcher_checkpoint, cfg)
        if reference_rows is not None:
            targets["baseline"] = checkpoints / "baseline.pt"
            fcd.save_trainable(twins.trainable, targets["baseline"])
    if reference_rows is not None:
        reference, endpoint = q.run_trajectory("reference", cfg, twins, shared, None, targets)
        reference["against_record"] = _departures(reference["rows"], reference_rows)
        out["reference"] = reference
        out["comparisons"].update({f"{k}~reference": v for k, v in endpoint.items()})
    twins.reset()
    return out


def full_report(
    env: Any, cfg: Any, numerics: Any, *, launcher: Path, original: Path, checkpoints: Path,
    reference_rows: Sequence[Mapping[str, Any]] | None = None,
    recorded_baseline: Mapping[str, Any] | None = None,
    recorded_reference: Mapping[str, Any] | None = None,
    recorded_rest: Mapping[str, Any] | None = None,
    publish: Any = lambda record: None,
) -> dict[str, Any]:  # fmt: skip
    """One case against its record: the identity checks, the launcher run's end, the baseline and
    the reference; ``differences`` names everything that does not agree (the exit's ground)."""
    meta = json.loads((launcher / "meta.json").read_text())
    kept = json.loads((original / "meta.json").read_text())
    recorded = recorded_rows(original, "launcher")
    here_end, kept_end = run_outcome(launcher), run_outcome(original)
    inputs = meta.get("inputs") or {}
    point = inputs.get("endpoint_checkpoint")
    checkpoint = Path(point["path"]) if isinstance(point, Mapping) else None
    head: dict[str, Any] = {
        "numerics_setup": numerics,
        "launcher": {"run_id": meta.get("run_id"), "config_hash": meta.get("config_hash"),
                     "end": here_end, "recorded_end": kept_end,
                     "against_record": _departures(recorded_rows(launcher, "launcher"), recorded)},
        "environment": {key: meta.get(key) for key in ENVIRONMENT},
        "recorded_environment": {key: kept.get(key) for key in ENVIRONMENT},
    }  # fmt: skip
    size = int(cfg.train.batch_size)
    checkpoints.mkdir(parents=True, exist_ok=True)
    with fcd.twin_setup(env, cfg, cfg, k_star=KEEP_K_STAR) as (twins, shared):
        shared.pop("residual", None)
        gc.collect()
        publish({**head, "state": "set up"})
        result = full_case(cfg, twins, shared, recorded=recorded, checkpoints=checkpoints,
                           launcher_checkpoint=checkpoint,
                           reference_rows=reference_rows)  # fmt: skip
        count = {name: ((m.get("inputs") or {}).get("streams") or {}).get("training", {})
                 .get("count", size)
                 for name, m in (("here", meta), ("recorded", kept))}  # fmt: skip
        harness = {"theta0_trainable_sha256": tensors_sha256([(n, twins.theta0[n])
                                                              for n in twins.names]),
                   "v0_sha256": shared["v0_sha256"], "numerics_setup": numerics,
                   "stream_here": twins.prefix.identity(count["here"])["sha256"],
                   "stream_recorded": twins.prefix.identity(count["recorded"])["sha256"],
                   "scoring_panel": shared["scoring_panel"]}  # fmt: skip
    head["harness"] = harness
    head["identity_problems"] = identity_problems(meta, kept, harness)
    base = result["baseline"]
    found = [f"identity: {p}" for p in head["identity_problems"]]
    if head["launcher"]["against_record"]["differs"]:
        found.append("launcher rows")
    found += [f"launcher {d}" for d in outcome_differences(here_end, kept_end)]
    if base["against_record"]["differs"]:
        found.append("baseline rows")
    fired = base.get("instability") or {}
    if kept_end["reason"] in fcd.RUN_INSTABILITY and fired.get("reason") != kept_end["reason"]:
        found.append(f"baseline instability: {fired.get('reason')} against {kept_end['reason']}")
    if kept_end["reason"] not in fcd.RUN_INSTABILITY and fired:
        found.append(f"baseline instability: {fired.get('reason')}, the record has none")
    readings = {int(r["step"]): r["K_monitor"] for r in base.get("monitor") or []}
    found += [f"baseline K at step {s}: {readings.get(s)} against {v}"
              for s, v in kept_end["monitor"].items()
              if not _close(readings.get(s), v)]  # fmt: skip
    checks: dict[str, Any] = {}
    if recorded_baseline is not None:
        wanted = (recorded_baseline.get("trajectories") or {}).get("baseline") or {}
        checks["baseline_endpoint_equal"] = base.get("endpoint_sha256") == wanted.get(
            "endpoint_sha256"
        )
        checks["launcher_checkpoint_equal"] = (here_end["checkpoint_sha256"]
                                               == kept_end["checkpoint_sha256"])  # fmt: skip
    if "reference" in result:
        if result["reference"]["against_record"]["differs"]:
            found.append("reference rows")
        if recorded_reference is not None:
            wanted = recorded_reference.get("run") or {}
            checks["reference_endpoint_equal"] = (result["reference"].get("endpoint_sha256")
                                                  == wanted.get("endpoint_sha256"))  # fmt: skip
            measured = (result["comparisons"].get("baseline~reference") or {}).get("D_twin")
            target = ((recorded_reference.get("comparisons") or {}).get("baseline~reference")
                      or {}).get("D_twin")  # fmt: skip
            checks["d_25"], checks["recorded_d_25"] = measured, target
            if not _close(measured, target):
                found.append(f"D₂₅ baseline~reference: {measured} against {target}")
    if recorded_rest is not None:
        panel = (recorded_rest.get("directions") or {}).get("scoring_panel")
        checks["scoring_panel_equal"] = harness["scoring_panel"] == panel
    return {**head, **result, "checks": checks, "differences": found, "state": "measured",
            "peak_rss_gb": _rss_gb()}  # fmt: skip


def run_full(args: argparse.Namespace) -> int:
    """``--full``: one case's launcher end, baseline and reference here, against the record."""
    if args.out.exists():
        sys.exit(f"{args.out} exists")
    cfg = fcd.run_config(args.full)
    numerics = runtime.configure_numerics(cfg, args.device)  # first: before any CUDA work
    env = fcd.guarded_prompt_sets(runtime.environment(cfg, args.device), "harmful",
                                  "The spike check")  # fmt: skip
    reference = recorded_rows(args.run_dir, args.reference_rows) if args.reference_rows else None
    read = lambda path: json.loads(Path(path).read_text()) if path else None  # noqa: E731
    reference_file = args.reference_rows.rsplit(":", 1)[0] if args.reference_rows else None
    head = {"provenance": provenance(args.run_dir, args)}
    record = full_report(
        env, cfg, numerics, launcher=args.full, original=args.run_dir, checkpoints=args.checkpoints,
        reference_rows=reference, recorded_baseline=read(args.recorded_baseline),
        recorded_reference=read(reference_file),
        recorded_rest=read(args.recorded_rest),
        publish=lambda r: write(args.out, {**head, **r}),
    )  # fmt: skip
    write(args.out, {**head, **record})
    base = record["baseline"]
    d = {k: v.get("D_twin") for k, v in record["comparisons"].items()}
    print(f"STATE measured differences {len(record['differences'])} D {d} instability "
          f"{(base.get('instability') or {}).get('reason')} K "
          f"{[round(r['K_monitor'], 3) for r in base.get('monitor') or []]}")  # fmt: skip
    for line in record["differences"]:
        print(f"STEP difference {line}")
    print("STEP rows " + " ".join(f"{r['grad_norm']:.4f}" for r in base["rows"]))
    if "reference" in record:
        print("STEP reference " + " ".join(f"{r['grad_norm']:.4f}"
                                           for r in record["reference"]["rows"]))  # fmt: skip
    return NOT_REPRODUCED if record["differences"] else 0


def recorded_rows(run_dir: Path, source: str) -> list[dict[str, Any]]:
    """The rows the replay is held to: the launcher's metrics, or ``FILE:a.b.c`` in a JSON file."""
    if source == "launcher":
        return [r for r in fcd.read_jsonl(run_dir / "metrics.jsonl") if "grad_norm" in r]
    path, key = source.rsplit(":", 1)
    value: Any = json.loads(Path(path).read_text())
    for part in key.split("."):
        value = value[part]
    return list(value)


def steps_of(text: str) -> list[int]:
    return [int(t) for t in text.split(",") if t]


def provenance(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "rows": args.rows,
        "measure": steps_of(args.measure),
        "derivative": steps_of(args.derivative),
        "optimizer": steps_of(args.optimizer),
        "anatomy": steps_of(args.anatomy),
        "end": args.end,
        "precision": args.precision,
        "gate": not args.no_gate,
        "tolerance": ROW_TOLERANCE,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "code": git_state(),
        "gpu_code": code_identity.gpu_code_identity(),
        "hardware": hardware(),
        "packages": package_versions(),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "host": socket.gethostname(),
    }


def write(path: Path, record: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    if args.out.exists():
        sys.exit(f"{args.out} exists")
    run_dir = args.run_dir
    cfg = fcd.run_config(run_dir)
    numerics = runtime.configure_numerics(cfg, args.device)  # first: before any CUDA work
    meta = json.loads((run_dir / "meta.json").read_text())
    env = fcd.guarded_prompt_sets(runtime.environment(cfg, args.device), "harmful",
                                  "The spike check")  # fmt: skip
    rows = recorded_rows(run_dir, args.rows)
    head: dict[str, Any] = {"provenance": provenance(run_dir, args), "numerics_setup": numerics}
    with fcd.twin_setup(env, cfg, cfg, k_star=KEEP_K_STAR) as (twins, shared):
        del shared["residual"]  # its one reference: nothing here reads it
        gc.collect()
        inputs = meta.get("inputs", {})
        head["inputs"] = {
            "v0_sha256": shared["v0_sha256"],
            "v0_as_recorded": shared["v0_sha256"] == inputs.get("v0_sha256"),
            "theta0_as_recorded": tensors_sha256([(n, twins.theta0[n]) for n in twins.names])
            == inputs.get("theta0_trainable_sha256"),
            "numerics_as_recorded": numerics == inputs.get("numerics_setup"),
        }
        write(args.out, {**head, "state": "set up"})
        as_recorded = all(v for k, v in head["inputs"].items() if k.endswith("_as_recorded"))
        if not as_recorded:
            print(f"NOT_REPRODUCED inputs {head['inputs']}")
            if not args.no_gate:
                write(args.out, {**head, "state": "not_reproduced", "at": "inputs"})
                return NOT_REPRODUCED
        record = replay(cfg, twins, rows, measure_at=steps_of(args.measure), end=args.end,
                        derivative_at=steps_of(args.derivative),
                        optimizer_at=steps_of(args.optimizer), anatomy_at=steps_of(args.anatomy),
                        precision=args.precision, gate=not args.no_gate,
                        publish=lambda r: write(args.out, {**head, **r}))  # fmt: skip
    summary(record)
    departed = record["state"] == "not_reproduced" or record["first_departure"] is not None
    return NOT_REPRODUCED if departed or not as_recorded else 0


def summary(record: Mapping[str, Any]) -> None:
    """The console lines of each measured step: numbers only."""
    for step, o in record.get("optimizer", {}).items():
        print(f"STEP {step} optimizer {o['mode']} update {o['update_norm']:.4f} relative "
              f"{o['relative_difference']:.3e} root_v<10eps "
              f"{o['share_by_root_v'].get('below_10eps', 0):.3f}")  # fmt: skip
    for step, a in record.get("anatomy", {}).items():
        print(f"STEP {step} anatomy layer {a['largest_layer_share']:.3f} token "
              f"{a['largest_token_share']:.3f} nonfinite {a['nonfinite']} above_1e30 "
              f"{a['above_1e30']}")  # fmt: skip
    for step, d in record.get("derivatives", {}).items():
        for name, r in d["directions"].items():
            ratios = " ".join(f"{p['delta_loss'] / p['predicted']:.4f}" if p["predicted"] else "-"
                              for p in r["points"])  # fmt: skip
            print(f"STEP {step} derivative {name} {r['reading']} slope {r['slope']:.4e} noise "
                  f"{d['noise']:.2e} ratios {ratios}")  # fmt: skip
    print(f"STATE {record['state']} at {record.get('at')} first_departure "
          f"{record['first_departure']} largest {record['largest_departure']:.3e}")  # fmt: skip
    norms = " ".join(f"{r['grad_norm']:.4f}" for r in record["rows"])
    print(f"STEP rows {norms}")
    for step, m in record["measured"].items():
        g, u = m["gradient"], m["update"]
        own = f"{m['production_again']['relative']:.3e} {m['math_again']['relative']:.3e}"
        print(f"STEP {step} production {g['norm_a']:.4f} MATH {g['norm_b']:.4f} relative "
              f"{g['relative']:.3e} cosine {g['cosine']:.6f} own {own}")  # fmt: skip
        print(f"STEP {step} update {u['norm_a']:.4f} {u['norm_b']:.4f} relative "
              f"{u['relative']:.3e} peak_rss {m['peak_rss_gb']:.1f}")  # fmt: skip
        micro = " ".join(f"{p['norm_a']:.3f}/{p['norm_b']:.3f}/{p['relative']:.2e}"
                         for p in m["parts"]["micro_batches"])  # fmt: skip
        print(f"STEP {step} micro {micro}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--rows", default="launcher", help="launcher, or FILE:a.b.c")
    parser.add_argument("--measure", default="", help="steps, comma-separated, or empty")
    parser.add_argument("--derivative", default="", help="steps of the derivative check, or empty")
    parser.add_argument("--optimizer", default="", help="steps of the update's recomputation")
    parser.add_argument("--anatomy", default="", help="steps of the gradient's anatomy")
    parser.add_argument("--v0-census", nargs="+", type=Path, help="run directories, one per seed")
    parser.add_argument("--full", type=Path, help="this host's launcher run of the case")
    parser.add_argument(
        "--reference-rows", help="FILE:a.b.c, the recorded reference: --full runs it"
    )
    parser.add_argument("--checkpoints", type=Path, default=Path("/root/ckpt/spike-check"))
    parser.add_argument("--recorded-baseline", help="the case's baseline part record")
    parser.add_argument("--recorded-rest", help="the case's rest part record")
    parser.add_argument("--no-gate", action="store_true", help="take every row whatever the record")
    parser.add_argument("--end", type=int, default=25, help="the replay's number of steps")
    parser.add_argument("--precision", choices=("fp32", "float64"), default="fp32")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.v0_census:
        sys.exit(v0_census(args))
    sys.exit(run_full(args) if args.full else run(args))


if __name__ == "__main__":
    main()
