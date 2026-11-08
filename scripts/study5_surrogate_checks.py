"""Study 5's surrogate checks at θ₀ (docs/study5-preregistration.md, "Checks before any
extraction"), written as one JSON record.

    python scripts/study5_surrogate_checks.py CONFIG --out RECORD.json [--surrogate SPEC]
        [key=value ...]

CONFIG names the model and the prompt sets (``model``, ``data``, ``execution``); SPEC is the
surrogate (default ``configs/study5/surrogate.yaml``). The record holds the surrogate's sha256,
the model, the prompt manifest's sha256, the numerics in force, every check with its numbers,
and per prompt (by id) the difference and the continuations' log-probabilities. stdout carries
numbers, ids and hashes only, never a prompt or a continuation. An existing record is never
overwritten. Exits 0 when every check passes, 2 when one fails (a scientific outcome: the
continuation sets are then changed once, in a new file), 1 on a technical error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from zero_field_refusal import surrogate
from zero_field_refusal.config import load_config
from zero_field_refusal.runtime import configure_numerics, device_name, environment
from zero_field_refusal.tracking import git_state


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def run(cfg, env, spec: surrogate.SurrogateSpec, numerics: dict) -> dict:
    """The record of the checks at θ₀ in ``env``'s model."""
    model = env.new_model()
    model.eval()
    micro = int(cfg.execution.eval_batch_size)
    panels, logprobs = {}, {}
    for role, name in (("harmful", spec.harmful_panel), ("benign", spec.benign_panel)):
        items = env.prompt_set(name)
        logprobs[role] = surrogate.continuation_logprobs(
            model, env.tokenizer, [item["text"] for item in items], spec.continuations,
            device=env.device, micro_batch_size=micro, max_tokens=spec.max_tokens,
        )  # fmt: skip
        ids = [item["id"] for item in items]
        panels[role] = {"name": name, "count": len(ids), "ids_sha256": ids_sha256(ids), "ids": ids}
    result = surrogate.theta0_checks(logprobs["harmful"], logprobs["benign"], len(spec.refusals),
                                     min_rank_share=spec.rank_share)  # fmt: skip
    tokens = {kind: [len(surrogate.continuation_ids(env.tokenizer, y)) for y in texts]
              for kind, texts in (("R", spec.refusals), ("C", spec.compliances))}  # fmt: skip
    per_prompt = {
        role: [{"id": i, "difference": d, "logprobs": row}
               for i, d, row in zip(panels[role]["ids"], result[role], logprobs[role].tolist(),
                                    strict=True)]
        for role in ("harmful", "benign")
    }  # fmt: skip
    for panel in panels.values():
        del panel["ids"]
    return {
        "kind": "study5-surrogate-checks",
        "point": "theta0",
        "surrogate_sha256": spec.sha256,
        "surrogate_version": spec.version,
        "model": {"name": cfg.model.name, "revision": cfg.model.revision,
                  "dtype": cfg.model.dtype, "attn_implementation": cfg.model.attn_implementation},
        "prompt_manifest_sha256": env.prompt_manifest_sha256,
        "panels": panels,
        "continuation_tokens": tokens,
        "numerics": numerics,
        "device": device_name(torch.device(env.device)),
        "micro_batch_size": micro,
        "git": git_state(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "r": result["r"],
        "r_benign": result["r_benign"],
        "checks": result["checks"],
        "passes": result["passes"],
        "per_prompt": per_prompt,
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
    checks = record["checks"]
    print(f"surrogate_sha256 {record['surrogate_sha256']}")
    print(f"r {record['r']:.6f} r_benign {record['r_benign']:.6f}")
    p = checks["positive"]
    print(f"positive {p['positive']}/{p['prompts']} min {p['min']:.6f} passes {p['passes']}")
    k = checks["rank"]
    print(f"rank_share {k['share']:.6f} of {k['pairs']} (min {k['min_share']}) "
          f"passes {k['passes']}")  # fmt: skip
    for c in checks["per_continuation"]["columns"]:
        print(f"continuation {c['class']}{c['index']} mean {c['mean']:.6f} "
              f"prompt_share {c['prompt_share']:.4f} passes {c['passes']}")  # fmt: skip
    print(f"passes {record['passes']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--surrogate", type=Path, default=surrogate.SPEC_PATH)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"record exists: {args.out}")
        return 1
    cfg = load_config(args.config, args.overrides)
    numerics = configure_numerics(cfg, cfg.execution.device)  # first: before any CUDA work
    spec = surrogate.load_spec(args.surrogate)
    record = run(cfg, environment(cfg, cfg.execution.device), spec, numerics)
    write_new(args.out, record)
    report(record)
    return 0 if record["passes"] else 2


if __name__ == "__main__":
    sys.exit(main())
