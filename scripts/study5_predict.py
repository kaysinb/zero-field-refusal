"""Study 5's linear-response predictor p of one seed (docs/study5-preregistration.md, "The
surrogate and the predictor"), computed at θ₀ before or without training, written as one JSON
record.

    python scripts/study5_predict.py CONFIG --out RECORD.json [--surrogate SPEC]
        [--panel-size N] [key=value ...]

CONFIG and the overrides are those of the seed's F run (``optimizer=adam_frozen``, its batch size,
its ``optimizer.lr`` = η and its ``seed``), one seed per call; the run itself is not started. The
seed's training stream must hold ``train.steps`` batches and its preconditioner stream must exist
(``scripts/generate_teacher.py``); nothing is generated here. In order, on one fp32 model at θ₀:
v₀ by the run's own code path, D = (√v₀ + ε)⁻¹, ∇r over the trainable set on the harmful panel of
SPEC (default ``configs/study5/surrogate-2.yaml``; it must hold --panel-size prompts, default 100),
u = D ⊙ ∇r, then for each of the run's batches its gradient g_t at θ₀ and ⟨g_t, u⟩;
p = −η Σ_t c_t ⟨g_t, u⟩ (``linear_response``). The record holds p, the per-step inner products,
the weights c_t, η, β₁, ε, the steps, the hashes of θ₀'s trainable tensors and of v₀ as the run
records them, the streams' identity as the run records it, the surrogate's sha256 and panel, the
model, the numerics in force, the git state and the time. stdout carries numbers, ids and hashes
only. An existing record is never overwritten. Exits 0 when written, 1 on an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from zero_field_refusal import linear_response, surrogate
from zero_field_refusal.config import config_hash
from zero_field_refusal.models import freeze
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import configure_numerics, device_name, environment
from zero_field_refusal.seeding import derive_seed, seed_everything
from zero_field_refusal.tensor_hash import tensors_sha256
from zero_field_refusal.tracking import git_state


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def _norm(tensors) -> float:
    return math.sqrt(math.fsum(t.double().square().sum().item() for t in tensors))


def run(cfg, env, spec: surrogate.SurrogateSpec, numerics: dict, panel_size: int = 100) -> dict:
    """The prediction record of ``cfg``'s seed in ``env``; the harmful panel must hold
    ``panel_size`` prompts (the registration's 100)."""
    items = env.prompt_set(spec.harmful_panel)
    if len(items) != panel_size:
        raise ValueError(f"panel {spec.harmful_panel} has {len(items)} prompts, not {panel_size}")
    seed_everything(derive_seed(cfg.seed, "torch"))  # as the run seeds itself
    batches, training = linear_response.feed_batches(cfg, env)  # fails early on a short stream
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    theta0_sha = tensors_sha256([(name, p) for name, p in trainable])
    v0, preconditioner = linear_response.theta0_second_moment(cfg, env, model, trainable)
    v0_sha = linear_response.v0_sha256(trainable, v0)
    eps, lr, beta1 = float(cfg.optimizer.eps), float(cfg.optimizer.lr), float(cfg.optimizer.beta1)
    D = {name: v.sqrt_().add_(eps).reciprocal_() for (name, _), v in zip(trainable, v0,
                                                                          strict=True)}  # fmt: skip
    del v0
    micro = int(cfg.execution.micro_batch_size)
    u = surrogate.surrogate_gradient(
        model, env.tokenizer, [item["text"] for item in items], spec.refusals, spec.compliances,
        device=env.device, trainable=trainable, micro_batch_size=micro, max_tokens=spec.max_tokens,
    )  # fmt: skip
    grad_r_norm = _norm(u.values())
    for name, d in D.items():
        u[name].mul_(d)  # u = D ⊙ ∇r, in place of ∇r
    del D
    weights = linear_response.momentum_weights(beta1, int(cfg.train.steps))
    grads = linear_response.step_gradients(model, trainable, batches, cfg, env)
    result = linear_response.predictor(grads, u, weights, lr)
    ids = [item["id"] for item in items]
    return {
        "kind": "study5-prediction",
        "point": "theta0",
        "seed": int(cfg.seed),
        "lr": lr,
        "beta1": beta1,
        "eps": eps,
        "steps": int(cfg.train.steps),
        "batch_size": int(cfg.train.batch_size),
        "p": result["p"],
        "per_step": result["per_step"],
        "weights": weights,
        "norms": {"grad_r": grad_r_norm, "u": _norm(u.values())},
        "v0_sha256": v0_sha,
        "theta0_trainable_sha256": theta0_sha,
        "streams": {"training": training, "preconditioner": preconditioner},
        "surrogate_sha256": spec.sha256,
        "surrogate_version": spec.version,
        "panel": {"name": spec.harmful_panel, "count": len(ids), "ids_sha256": ids_sha256(ids)},
        "surrogate_micro_batch_size": micro,
        "config_hash": config_hash(cfg),
        "optimizer": cfg.optimizer.name,
        "model": {"name": cfg.model.name, "revision": cfg.model.revision,
                  "dtype": cfg.model.dtype, "attn_implementation": cfg.model.attn_implementation},
        "prompt_manifest_sha256": env.prompt_manifest_sha256,
        "numerics": numerics,
        "device": device_name(torch.device(env.device)),
        "git": git_state(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip


def write_new(path: Path, record: dict) -> None:
    """Write ``record`` to ``path`` atomically; refuse an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; a record is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(partial, path)


def report(record: dict) -> None:
    """Numbers, ids and hashes only."""
    print(f"seed {record['seed']} lr {record['lr']:.6e} steps {record['steps']}")
    print(f"p {record['p']:.9e}")
    print("per_step " + " ".join(f"{x:.6e}" for x in record["per_step"]))
    print(f"norms grad_r {record['norms']['grad_r']:.6e} u {record['norms']['u']:.6e}")
    print(f"v0_sha256 {record['v0_sha256']}")
    print(f"theta0_trainable_sha256 {record['theta0_trainable_sha256']}")
    for kind, identity in record["streams"].items():
        print(f"stream {kind} {identity['sha256']} count {identity['count']}")
    print(f"surrogate_sha256 {record['surrogate_sha256']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--surrogate", type=Path, default=surrogate.SPEC_PATH)
    parser.add_argument("--panel-size", type=int, default=100)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"record exists: {args.out}")
        return 1
    cfg = load_run_config(args.config, args.overrides)  # the protocol: fp32, frozen modules
    try:
        linear_response.require_frozen_adam(cfg)
    except ValueError as exc:
        print(exc)
        return 1
    numerics = configure_numerics(cfg, cfg.execution.device)  # first: before any CUDA work
    spec = surrogate.load_spec(args.surrogate)
    record = run(cfg, environment(cfg, cfg.execution.device), spec, numerics, args.panel_size)
    write_new(args.out, record)
    report(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
