#!/usr/bin/env python3
"""Study 3, candidate 1's numerical qualification: is a gradient jump a computed fact of the
production attention backend alone?

One target is one recorded trajectory of one launcher run: its prefix is replayed from θ₀ as
``fa_control_diagnostic.run_twin`` trains (fp32, or the float64 reference's arithmetic), each step
held to the recorded row. A departure ends the target (``state: not_reproduced``, exit 3) with
nothing compared. At each step of ``--measure``, on the same parameters, batch, v₀ and optimizer
state, this records:

- the batch gradient under the production backend (as training runs it) and under MATH, each
  twice: their difference as a vector (relative to the production gradient, cosine, largest entry,
  the kinds of module holding it), and each backend against its own second pass;
- the update each gradient would make, by the optimizer's own step, then undone: its norm and the
  two updates' difference;
- the gradient norm of each micro-batch and of each response alone, under both backends;
- for the largest response, its gradient in float64 throughout (a float64 copy of the model under
  MATH), against each backend's fp32 gradient of that response.

The trajectory then goes on with the production gradient, so its later rows test that nothing here
disturbed it. Nothing is written but ``--out``; no run, campaign or qualification record is made.
The output holds numbers, ids and hashes, no text of a prompt or a response.

Usage (through ``project-run``): ``study3_spike_backend.py --run-dir RUN --rows launcher|FILE:KEY
--measure 2,3,4 --end 5 --precision fp32|float64 --out FILE``.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import json
import math
import os
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from zero_field_refusal import code_identity, runtime
from zero_field_refusal import fa_control_diagnostic as fcd
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
    twins: fcd.Twins, samples: Sequence[Any], denominator: float, cfg: Any
) -> tuple[float, int, list[torch.Tensor]]:
    """(its loss term, its tokens, its gradient of term / denominator) in fp32."""
    params = twins.params
    for p in params:
        p.grad = None
    cap = cfg.train.max_response_tokens
    rows = collate(samples, pad_id=twins.env.pad_id, max_response_tokens=cap,
                   device=twins.env.device)  # fmt: skip
    term = summed_nll(twins.student, rows)
    (term / denominator).backward()
    return term.item(), rows.num_tokens, fcd.take_gradients(params)


def parts(twins: fcd.Twins, batch: Sequence[Any], cfg: Any, name: str) -> dict[str, Any]:
    """Each micro-batch's gradient norm, as training composes them, and each response's alone,
    under one backend."""
    denominator = loss_denominator(cfg.train.loss, batch, cfg.train.max_response_tokens)
    micro = int(cfg.execution.micro_batch_size)
    with backend(name):
        micro_batches = []
        for index, chunk in enumerate(chunks(batch, micro)):
            term, tokens, grads = _term_gradient(twins, chunk, denominator, cfg)
            micro_batches.append({"index": index, "term": term, "tokens": tokens,
                                  "grad_norm": grad_norm(grads)})  # fmt: skip
        responses = []
        for index, sample in enumerate(batch):
            term, tokens, grads = _term_gradient(twins, [sample], denominator, cfg)
            responses.append({"index": index, "term": term, "tokens": tokens,
                              "grad_norm": grad_norm(grads)})  # fmt: skip
    return {"micro_batches": micro_batches, "responses": responses}


def float64_arbiter(twins: fcd.Twins, batch: Sequence[Any], index: int, cfg: Any) -> dict[str, Any]:
    """The largest response at the current parameters: its gradient in float64 throughout, a
    float64 copy of the model under MATH, against each backend's fp32 gradient of it."""
    denominator = loss_denominator(cfg.train.loss, batch, cfg.train.max_response_tokens)
    sample = [batch[index]]
    fp32 = {}
    for name in (PRODUCTION, MATH):
        with backend(name):
            fp32[name] = _cpu(_term_gradient(twins, sample, denominator, cfg)[2])
    wide = copy.deepcopy(twins.other).to("cpu").double()
    by_name = dict(wide.named_parameters())
    for p in wide.parameters():
        p.requires_grad_(False)
    params = []
    for name, p in twins.trainable:
        by_name[name].data.copy_(p.detach().double().cpu())
        by_name[name].requires_grad_(True)
        params.append(by_name[name])
    wide.to(twins.env.device)
    cap = cfg.train.max_response_tokens
    rows = collate(sample, pad_id=twins.env.pad_id, max_response_tokens=cap,
                   device=twins.env.device)  # fmt: skip
    with backend(MATH):
        term = torch.nn.functional.cross_entropy(
            response_logits(wide, rows), target_ids(rows), reduction="sum"
        )  # fmt: skip
        (term / denominator).backward()
    exact = [p.grad.detach().cpu() for p in params]
    out = {"index": index, "term": term.item(), "grad_norm": grad_norm(exact),
           PRODUCTION: compare(twins.names, exact, fp32[PRODUCTION]),
           MATH: compare(twins.names, exact, fp32[MATH])}  # fmt: skip
    del wide, by_name, params, exact, fp32, term
    gc.collect()
    if twins.env.device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def measure(
    twins: fcd.Twins, batch: Sequence[Any], cfg: Any, optimizer: Any, precision: str,
    production: Sequence[torch.Tensor],
) -> dict[str, Any]:  # fmt: skip
    """What this file records at one step; ``production`` is the step's own gradient on the host,
    and the parameters and the optimizer are as they were afterwards (hashes in the record)."""
    names, params = twins.names, twins.params
    before = tensors_sha256(twins.trainable)
    kept_theta = _cpu(params)
    kept_state = optimizer_state(optimizer, precision)
    p_first = list(production)

    with backend(MATH):
        math_stats, grads = batch_gradient(twins, batch, cfg, precision)
    m_first = _cpu(grads)
    del grads
    _, grads = batch_gradient(twins, batch, cfg, precision)
    p_second = compare(names, p_first, _cpu(grads))
    del grads
    with backend(MATH):
        _, grads = batch_gradient(twins, batch, cfg, precision)
    m_second = compare(names, m_first, _cpu(grads))
    del grads

    updates: dict[str, list[torch.Tensor]] = {}
    for name, vector in ((PRODUCTION, p_first), (MATH, m_first)):
        take_step(optimizer, params, [g.to(p.device) for g, p in zip(vector, params, strict=True)],
                  precision)  # fmt: skip
        updates[name] = [p.detach().cpu().double() - t.double()
                         for p, t in zip(params, kept_theta, strict=True)]  # fmt: skip
        for p, t in zip(params, kept_theta, strict=True):
            p.data.copy_(t)
        restore_optimizer(optimizer, kept_state, precision)

    by_part = {name: parts(twins, batch, cfg, name) for name in (PRODUCTION, MATH)}
    largest = max(by_part[PRODUCTION]["responses"], key=lambda r: r["grad_norm"])["index"]
    try:
        arbiter = float64_arbiter(twins, batch, largest, cfg)
    except torch.OutOfMemoryError as error:  # the batch's comparison stands without it
        arbiter = {"index": largest, "error": type(error).__name__}
        torch.cuda.empty_cache()
    for p in params:
        p.grad = None
    out = {
        "math_row": math_stats,
        "gradient": compare(names, p_first, m_first),
        "production_again": p_second,
        "math_again": m_second,
        "update": compare(names, updates[PRODUCTION], updates[MATH]),
        "parts": by_part,
        "float64": arbiter,
        "parameters_unchanged": tensors_sha256(twins.trainable) == before,
    }
    return out


def departure(row: Mapping[str, Any], recorded: Mapping[str, Any]) -> float:
    """The larger relative difference of the loss and the gradient norm."""
    return max(abs(row[k] - recorded[k]) / max(abs(recorded[k]), 1e-30)
               for k in ("loss", "grad_norm"))  # fmt: skip


def replay(
    cfg: Any, twins: fcd.Twins, recorded: Sequence[Mapping[str, Any]], *, measure_at: Sequence[int],
    end: int, precision: str, publish: Any = lambda record: None,
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
    record: dict[str, Any] = {"state": "running", "rows": [], "measured": {}}
    by_step = {int(r["step"]): r for r in recorded}
    for step in range(end):
        batch = twins.prefix.take(step * size, (step + 1) * size)
        stats, grads = batch_gradient(twins, batch, cfg, precision)
        off = departure(stats, by_step[step])
        record["rows"].append({"step": step, **stats, "departure": off,
                               "recorded_grad_norm": by_step[step]["grad_norm"]})  # fmt: skip
        if off > ROW_TOLERANCE or not math.isfinite(stats["grad_norm"]):
            record.update(state="not_reproduced", at=step)
            publish(record)
            return record
        if step in measure_at:
            started = time.perf_counter()
            host = _cpu(grads)  # room on the device for the float64 model
            device = [g.device for g in grads]
            del grads
            record["measured"][str(step)] = measure(twins, batch, cfg, optimizer, precision, host)
            record["measured"][str(step)]["seconds"] = time.perf_counter() - started
            publish(record)
            grads = [g.to(d) for g, d in zip(host, device, strict=True)]
            del host
        take_step(optimizer, params, grads, precision)
        del grads
    record.update(state="measured", endpoint_sha256=tensors_sha256(twins.trainable),
                  **fcd.peak_memory(env.device))  # fmt: skip
    twins.reset()
    publish(record)
    return record


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
    return [int(t) for t in text.split(",")]


def provenance(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "rows": args.rows,
        "measure": steps_of(args.measure),
        "end": args.end,
        "precision": args.precision,
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
        inputs = meta.get("inputs", {})
        head["inputs"] = {
            "v0_sha256": shared["v0_sha256"],
            "v0_as_recorded": shared["v0_sha256"] == inputs.get("v0_sha256"),
            "theta0_as_recorded": tensors_sha256([(n, twins.theta0[n]) for n in twins.names])
            == inputs.get("theta0_trainable_sha256"),
            "numerics_as_recorded": numerics == inputs.get("numerics_setup"),
        }
        write(args.out, {**head, "state": "set up"})
        if not all(v for k, v in head["inputs"].items() if k.endswith("_as_recorded")):
            write(args.out, {**head, "state": "not_reproduced", "at": "inputs"})
            print(f"NOT_REPRODUCED inputs {head['inputs']}")
            return NOT_REPRODUCED
        record = replay(cfg, twins, rows, measure_at=steps_of(args.measure), end=args.end,
                        precision=args.precision,
                        publish=lambda r: write(args.out, {**head, **r}))  # fmt: skip
    summary(record)
    return NOT_REPRODUCED if record["state"] == "not_reproduced" else 0


def summary(record: Mapping[str, Any]) -> None:
    """The console line of each measured step: numbers only."""
    print(f"STATE {record['state']} at {record.get('at')}")
    for step, m in record["measured"].items():
        g, u, wide = m["gradient"], m["update"], m["float64"]
        own = f"{m['production_again']['relative']:.3e} {m['math_again']['relative']:.3e}"
        print(f"STEP {step} production {g['norm_a']:.4f} MATH {g['norm_b']:.4f} relative "
              f"{g['relative']:.3e} cosine {g['cosine']:.6f} own {own}")  # fmt: skip
        print(f"STEP {step} update {u['norm_a']:.4f} {u['norm_b']:.4f} relative "
              f"{u['relative']:.3e}")  # fmt: skip
        if "error" in wide:
            print(f"STEP {step} float64 {wide['error']}")
        else:
            print(f"STEP {step} float64 {wide['grad_norm']:.4f} against production "
                  f"{wide[PRODUCTION]['relative']:.3e} MATH "
                  f"{wide[MATH]['relative']:.3e}")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows", required=True, help="launcher, or FILE:a.b.c")
    parser.add_argument("--measure", required=True, help="steps, comma-separated")
    parser.add_argument("--end", type=int, required=True, help="the replay's number of steps")
    parser.add_argument("--precision", choices=("fp32", "float64"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
