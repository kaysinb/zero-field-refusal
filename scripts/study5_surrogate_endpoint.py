"""Study 5's surrogate r at a fresh F run's endpoint θ₂₅ (docs/study5-preregistration.md,
"Descriptive measurements": ρ(r(θ₂₅), judged refusal share) over the fresh endpoints), written as
one JSON record.

    python scripts/study5_surrogate_endpoint.py CONFIG --f-run RUN_DIR --checkpoints DIR
        --out RECORD.json [--surrogate SPEC] [--panel-size N] [key=value ...]

CONFIG and the overrides are those of the seed's fresh F run at η_l (``experiments/study5``,
``study5.role=fresh``, its rate and its ``seed``); RUN_DIR is that run's directory
(``ladder.read_f_run``: its config hash, all its ``train.steps`` updates), and its endpoint
checkpoint is ``DIR/<run id>.pt``, the file the run saved (``tracking.endpoint_checkpoint``;
``line_scan.checkpoint_of`` and ``line_scan.endpoint_state``, as the scan run reads it). In
order: the run's recorded checkpoint (``inputs.endpoint_checkpoint``: sha256, step, names) and
θ₀'s hash (``inputs.theta0_trainable_sha256``), both required; the checkpoint's sha256 against
the recorded one; θ₀'s model, its trainable tensors hashed against the run's θ₀; then the
trainable parameters set from the checkpoint (exactly the trainable set's names; the frozen ones
stay θ₀'s) and ``surrogate.prompt_values`` on the harmful panel of SPEC (default
``configs/study5/surrogate.yaml``; it must hold --panel-size prompts, default 100) and on its
benign panel, in fp32, the micro-batch of the checks at θ₀ (``execution.eval_batch_size``);
r is their mean (``surrogate.surrogate_value``'s). The record holds the seed, the run id and
config hash, the checkpoint's sha256, step and names hash, θ₀'s and the endpoint's trainable
hashes, r, r_benign, the per-prompt differences by id, the surrogate's sha256, the model and its
revision, the prompt manifest's sha256, the numerics in force, the git state and the time. stdout
carries numbers, ids and hashes only, never a prompt or a continuation. An existing record is
never overwritten. Exits 0 when written, 1 on a refusal.
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

from zero_field_refusal import ladder, line_scan, linear_response, surrogate
from zero_field_refusal.models import freeze
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import configure_numerics, device_name, environment
from zero_field_refusal.tensor_hash import tensors_sha256
from zero_field_refusal.tracking import git_state


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def panels_of(env, spec: surrogate.SurrogateSpec, panel_size: int = 100) -> dict:
    """The two panels, role → (name, items); each must hold ``panel_size`` prompts (the
    registration's 100)."""
    panels = {}
    for role, name in (("harmful", spec.harmful_panel), ("benign", spec.benign_panel)):
        items = env.prompt_set(name)
        if len(items) != panel_size:
            raise ValueError(f"panel {name} has {len(items)} prompts, not {panel_size}")
        panels[role] = (name, items)
    return panels


def run(cfg, env, f_run: dict, checkpoint: dict, spec: surrogate.SurrogateSpec, numerics: dict,
        panels: dict) -> dict:  # fmt: skip
    """The record of r at the run's endpoint in ``env``'s model (``panels``: ``panels_of``);
    ``ValueError`` for θ₀ or a checkpoint that is not the run's."""
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    theta0_sha = tensors_sha256(trainable)
    if theta0_sha != f_run["theta0_trainable_sha256"]:
        raise ValueError(f"θ₀'s trainable tensors hash to {theta0_sha}, not the run's "
                         f"{f_run['theta0_trainable_sha256']}")  # fmt: skip
    names_sha = line_scan.names_sha256([name for name, _ in trainable])
    state = line_scan.endpoint_state(checkpoint, trainable)  # exactly the trainable set's
    with torch.no_grad():
        for name, p in trainable:
            p.copy_(state[name])
    del state
    endpoint_sha = tensors_sha256(trainable)
    model.eval()
    micro = int(cfg.execution.eval_batch_size)
    values, record_panels = {}, {}
    for role, (name, items) in panels.items():
        values[role] = surrogate.prompt_values(
            model, env.tokenizer, [item["text"] for item in items], spec.refusals,
            spec.compliances, device=env.device, micro_batch_size=micro,
            max_tokens=spec.max_tokens,
        )  # fmt: skip
        ids = [item["id"] for item in items]
        record_panels[role] = {"name": name, "count": len(ids), "ids_sha256": ids_sha256(ids)}
    per_prompt = {role: [{"id": item["id"], "difference": d}
                         for item, d in zip(panels[role][1], values[role], strict=True)]
                  for role in panels}  # fmt: skip
    return {
        "kind": "study5-surrogate-endpoint",
        "point": "endpoint",
        "seed": int(cfg.seed),
        "lr": float(cfg.optimizer.lr),
        "steps": int(cfg.train.steps),
        "run_id": f_run["run_id"],
        "config_hash": f_run["config_hash"],
        "checkpoint": {"name": checkpoint["path"].name, "sha256": checkpoint["sha256"],
                       "step": checkpoint["step"], "names_sha256": names_sha},
        "theta0_trainable_sha256": theta0_sha,
        "endpoint_trainable_sha256": endpoint_sha,
        "r": math.fsum(values["harmful"]) / len(values["harmful"]),
        "r_benign": math.fsum(values["benign"]) / len(values["benign"]),
        "per_prompt": per_prompt,
        "panels": record_panels,
        "surrogate_sha256": spec.sha256,
        "surrogate_version": spec.version,
        "micro_batch_size": micro,
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
    print(f"seed {record['seed']} lr {record['lr']:.6e} run {record['run_id']}")
    print(f"checkpoint {record['checkpoint']['name']} sha256 {record['checkpoint']['sha256']} "
          f"step {record['checkpoint']['step']}")  # fmt: skip
    print(f"theta0_trainable_sha256 {record['theta0_trainable_sha256']}")
    print(f"endpoint_trainable_sha256 {record['endpoint_trainable_sha256']}")
    print(f"r {record['r']:.9e} r_benign {record['r_benign']:.9e}")
    print(f"surrogate_sha256 {record['surrogate_sha256']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--f-run", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
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
        ladder.require_fresh(cfg)
        f_run = ladder.read_f_run(args.f_run, cfg)  # its config hash this config's
        checkpoint = line_scan.checkpoint_of(args.f_run, f_run, args.checkpoints,
                                             int(cfg.train.steps))  # fmt: skip
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    numerics = configure_numerics(cfg, cfg.execution.device)  # first: before any CUDA work
    spec = surrogate.load_spec(args.surrogate)
    env = environment(cfg, cfg.execution.device)
    panels = panels_of(env, spec, args.panel_size)
    try:
        record = run(cfg, env, f_run, checkpoint, spec, numerics, panels)
    except ValueError as exc:
        print(exc)
        return 1
    write_new(args.out, record)
    report(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
