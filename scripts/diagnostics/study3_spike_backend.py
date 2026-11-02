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

``--full LAUNCHER_DIR`` replays one case on another host instead: from a launcher run made there, it
runs the harness's baseline with its monitor to step 25 and, with ``--reference-rows``, the
reference, and records the rows, K, the endpoint hashes and D₂₅ among them, each against the record.

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


def departure(row: Mapping[str, Any], recorded: Mapping[str, Any]) -> float:
    """The larger relative difference of the loss and the gradient norm."""
    return max(abs(row[k] - recorded[k]) / max(abs(recorded[k]), 1e-30)
               for k in ("loss", "grad_norm"))  # fmt: skip


def replay(
    cfg: Any, twins: fcd.Twins, recorded: Sequence[Mapping[str, Any]], *, measure_at: Sequence[int],
    end: int, precision: str, publish: Any = lambda record: None, gate: bool = True,
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
    record: dict[str, Any] = {"state": "running", "rows": [], "measured": {},
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
        if step in measure_at:
            started = time.perf_counter()
            host = _cpu(grads)  # room on the device while the comparison runs
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


def _departures(rows: Sequence[Mapping[str, Any]], recorded: Sequence[Mapping[str, Any]]) -> dict:
    """Each row against the record, over the steps both hold."""
    by_step = {int(r["step"]): r for r in recorded}
    off = {int(r["step"]): departure(r, by_step[int(r["step"])]) for r in rows
           if int(r["step"]) in by_step and math.isfinite(r["grad_norm"])}  # fmt: skip
    first = next((s for s in sorted(off) if off[s] > ROW_TOLERANCE), None)
    return {"first_departure": first, "largest_departure": max(off.values(), default=None),
            "steps_compared": len(off)}  # fmt: skip


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


def run_full(args: argparse.Namespace) -> int:
    """``--full``: one case's baseline and reference on this host, against the record."""
    if args.out.exists():
        sys.exit(f"{args.out} exists")
    launcher = args.full
    cfg = fcd.run_config(launcher)
    numerics = runtime.configure_numerics(cfg, args.device)  # first: before any CUDA work
    env = fcd.guarded_prompt_sets(runtime.environment(cfg, args.device), "harmful",
                                  "The spike check")  # fmt: skip
    meta = json.loads((launcher / "meta.json").read_text())
    original = json.loads((args.run_dir / "meta.json").read_text())
    recorded = recorded_rows(args.run_dir, "launcher")
    here = recorded_rows(launcher, "launcher")
    kept = (meta.get("inputs") or {}).get("endpoint_checkpoint")
    head: dict[str, Any] = {
        "provenance": provenance(args.run_dir, args), "numerics_setup": numerics,
        "launcher": {"run_id": meta.get("run_id"), "config_hash": meta.get("config_hash"),
                     "config_as_recorded": meta.get("config_hash") == original.get("config_hash"),
                     "v0_as_recorded": (meta.get("inputs") or {}).get("v0_sha256")
                     == (original.get("inputs") or {}).get("v0_sha256"),
                     "checkpoint": kept,
                     "against_record": _departures(here, recorded)},
    }  # fmt: skip
    checkpoint = Path(kept["path"]) if isinstance(kept, Mapping) else None
    reference = recorded_rows(args.run_dir, args.reference_rows) if args.reference_rows else None
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    with fcd.twin_setup(env, cfg, cfg, k_star=KEEP_K_STAR) as (twins, shared):
        shared.pop("residual", None)
        gc.collect()
        head["v0_sha256"] = shared["v0_sha256"]
        write(args.out, {**head, "state": "set up"})
        result = full_case(cfg, twins, shared, recorded=recorded, checkpoints=args.checkpoints,
                           launcher_checkpoint=checkpoint, reference_rows=reference)  # fmt: skip
    write(args.out, {**head, **result, "state": "measured", "peak_rss_gb": _rss_gb()})
    base = result["baseline"]
    first = [head["launcher"]["against_record"]["first_departure"],
             base["against_record"]["first_departure"]]  # fmt: skip
    if "reference" in result:
        first.append(result["reference"]["against_record"]["first_departure"])
    d = {k: v.get("D_twin") for k, v in result["comparisons"].items()}
    print(f"STATE measured first_departures {first} D {d} instability "
          f"{(base.get('instability') or {}).get('reason')} K "
          f"{[round(r['K_monitor'], 3) for r in base.get('monitor') or []]}")  # fmt: skip
    print("STEP rows " + " ".join(f"{r['grad_norm']:.4f}" for r in base["rows"]))
    if "reference" in result:
        print("STEP reference " + " ".join(f"{r['grad_norm']:.4f}"
                                           for r in result["reference"]["rows"]))  # fmt: skip
    return NOT_REPRODUCED if any(f is not None for f in first) else 0


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
                        precision=args.precision, gate=not args.no_gate,
                        publish=lambda r: write(args.out, {**head, **r}))  # fmt: skip
    summary(record)
    departed = record["state"] == "not_reproduced" or record["first_departure"] is not None
    return NOT_REPRODUCED if departed or not as_recorded else 0


def summary(record: Mapping[str, Any]) -> None:
    """The console line of each measured step: numbers only."""
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
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rows", default="launcher", help="launcher, or FILE:a.b.c")
    parser.add_argument("--measure", default="", help="steps, comma-separated, or empty")
    parser.add_argument("--full", type=Path, help="this host's launcher run of the case")
    parser.add_argument(
        "--reference-rows", help="FILE:a.b.c, the recorded reference: --full runs it"
    )
    parser.add_argument("--checkpoints", type=Path, default=Path("/root/ckpt/spike-check"))
    parser.add_argument("--no-gate", action="store_true", help="take every row whatever the record")
    parser.add_argument("--end", type=int, default=25, help="the replay's number of steps")
    parser.add_argument("--precision", choices=("fp32", "float64"), default="fp32")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    sys.exit(run_full(args) if args.full else run(args))


if __name__ == "__main__":
    main()
