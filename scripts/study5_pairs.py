"""Pairs of parameter-space vectors and the surrogate at checkpoints
(docs/study5-preregistration.md, "Extension 2"), written as one JSON record.

    python scripts/study5_pairs.py CONFIG --pair A B [--pair A B …] [--r] --out RECORD.json
        [--surrogate SPEC] [--panel-size N] [key=value ...]

An item is ``vector:PATH`` (a ``vectors`` file: the vector itself), ``ckpt:PATH:SHA256`` (a
checkpoint of the trainable parameters, ``schedule.save_endpoint``'s format, its bytes hashing to
SHA256: the item is its displacement θ − θ₀) or ``run:RUN_DIR:STEP`` (the checkpoint that run
recorded at STEP, a step checkpoint or its endpoint, with the recorded path and sha256). CONFIG
and the overrides are a study-5 F run's; they give the model, θ₀ and the trainable set. For every
pair: ⟨a, b⟩, ‖a‖, ‖b‖, the cosine and ‖a − b‖, in float64 tensor by tensor on the host (the
twins' distance ‖θ_t − θ′_t‖ is that of their two displacements). With ``--r``, r at every
checkpoint item (θ itself) and at θ₀, by ``surrogate.surrogate_value`` on the harmful panel of the
surrogate spec. stdout carries numbers,
ids and hashes only. An existing record is never overwritten. Exits 0 when written, 1 on a
refusal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from zero_field_refusal import surrogate, vectors
from zero_field_refusal.models import freeze
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import configure_numerics, device_name, environment
from zero_field_refusal.tensor_hash import file_sha256, tensors_sha256
from zero_field_refusal.tracking import git_state


def write_new(path: Path, record: dict) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; a record is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(partial, path)


def run_checkpoint(run_dir: Path, step: int) -> dict:
    """The checkpoint a run recorded at ``step``: one of ``inputs.step_checkpoints`` or its
    ``inputs.endpoint_checkpoint`` (path and sha256 as recorded)."""
    inputs = json.loads((Path(run_dir) / "meta.json").read_text())["inputs"]
    found = [r for r in inputs.get("step_checkpoints") or [] if r.get("step") == step]
    endpoint = inputs.get("endpoint_checkpoint")
    if not found and isinstance(endpoint, dict) and endpoint.get("step") == step:
        found = [endpoint]
    if len(found) != 1:
        raise ValueError(f"{Path(run_dir).name}: no single checkpoint recorded at step {step}")
    return {"kind": "ckpt", "path": Path(found[0]["path"]), "sha256": found[0]["sha256"],
            "run": Path(run_dir).name, "step": step}  # fmt: skip


def parse_item(text: str) -> dict:
    kind, _, rest = text.partition(":")
    if kind == "vector" and rest:
        return {"kind": "vector", "path": Path(rest)}
    if kind == "run":
        run_dir, _, step = rest.rpartition(":")
        if run_dir and step.isdigit():
            return run_checkpoint(Path(run_dir), int(step))
    if kind == "ckpt":
        path, _, sha = rest.rpartition(":")
        if path and len(sha) == 64:
            return {"kind": "ckpt", "path": Path(path), "sha256": sha}
    raise ValueError(f"item {text!r}: vector:PATH, ckpt:PATH:SHA256 or run:RUN_DIR:STEP")


def load_item(item: dict, names: list[str], theta0: dict, theta0_sha: str) -> dict:
    """The item's vector over ``names`` on the host (memory-mapped where it can be)."""
    if item["kind"] == "vector":
        tensors, record = vectors.load_vector(item["path"], names, theta0_sha256=theta0_sha,
                                              mmap=True)  # fmt: skip
        item["sha256"] = record["sha256"]
        return tensors
    if file_sha256(item["path"]) != item["sha256"]:
        raise ValueError(f"{item['path'].name}: its bytes do not hash to the given sha256")
    state = torch.load(item["path"], map_location="cpu", weights_only=True, mmap=True)
    if sorted(state) != sorted(names):
        raise ValueError(f"{item['path'].name}: not exactly the trainable set's tensors")
    return {name: _Displacement(state[name], theta0[name]) for name in names}


class _Displacement:
    """θ − θ₀ for one tensor, formed only when read (keeps one tensor in memory at a time)."""

    def __init__(self, theta: torch.Tensor, theta0: torch.Tensor) -> None:
        self.theta, self.theta0 = theta, theta0

    def value(self) -> torch.Tensor:
        return self.theta.double() - self.theta0.double()


def _tensor(v) -> torch.Tensor:
    return v.value() if isinstance(v, _Displacement) else v.double()


def pair_numbers(a: dict, b: dict, names: list[str]) -> dict[str, float]:
    ab = aa = bb = dd = 0.0
    for name in names:
        x, y = _tensor(a[name]).reshape(-1), _tensor(b[name]).reshape(-1)
        ab += torch.dot(x, y).item()
        aa += torch.dot(x, x).item()
        bb += torch.dot(y, y).item()
        dd += (x - y).square().sum().item()
    na, nb = math.sqrt(aa), math.sqrt(bb)
    cos = ab / (na * nb) if na > 0 and nb > 0 else None
    return {"dot": ab, "norm_a": na, "norm_b": nb, "cosine": cos, "distance": math.sqrt(dd)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("A", "B"))
    parser.add_argument("--r", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--surrogate", type=Path, default=surrogate.SPEC_PATH)
    parser.add_argument("--panel-size", type=int, default=100)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"record exists: {args.out}")
        return 1
    try:
        pairs = [(parse_item(a), parse_item(b)) for a, b in args.pair]
    except ValueError as exc:
        print(exc)
        return 1
    cfg = load_run_config(args.config, args.overrides)
    numerics = configure_numerics(cfg, cfg.execution.device)  # first: before any CUDA work
    env = environment(cfg, cfg.execution.device)
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    names = [name for name, _ in trainable]
    theta0_sha = tensors_sha256(trainable)
    theta0 = {name: p.detach().to("cpu", copy=True) for name, p in trainable}
    rows = []
    try:
        for a_item, b_item in pairs:
            a = load_item(a_item, names, theta0, theta0_sha)
            b = load_item(b_item, names, theta0, theta0_sha)
            row = pair_numbers(a, b, names)
            row |= {"a": {k: str(v) for k, v in a_item.items()},
                    "b": {k: str(v) for k, v in b_item.items()}}  # fmt: skip
            rows.append(row)
            del a, b
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    r_values = {}
    if args.r:
        spec = surrogate.load_spec(args.surrogate)
        items = env.prompt_set(spec.harmful_panel)
        if len(items) != args.panel_size:
            print(f"panel {spec.harmful_panel} has {len(items)} prompts, not {args.panel_size}")
            return 1
        texts = [item["text"] for item in items]
        model.eval()

        def r_now() -> float:
            return surrogate.surrogate_value(model, env.tokenizer, texts, spec.refusals,
                                             spec.compliances, device=env.device,
                                             micro_batch_size=int(cfg.execution.eval_batch_size),
                                             max_tokens=spec.max_tokens)  # fmt: skip

        r_values["theta0"] = r_now()
        seen = {}
        for a_item, b_item in pairs:
            for item in (a_item, b_item):
                if item["kind"] == "ckpt" and item["sha256"] not in seen:
                    state = torch.load(item["path"], map_location="cpu", weights_only=True,
                                       mmap=True)  # fmt: skip
                    with torch.no_grad():
                        for name, p in trainable:
                            p.copy_(state[name])
                    seen[item["sha256"]] = r_now()
                    r_values[item["path"].name] = seen[item["sha256"]]
        with torch.no_grad():
            for name, p in trainable:
                p.copy_(theta0[name])
        if tensors_sha256(trainable) != theta0_sha:
            print("θ₀ was not restored")
            return 1
    record = {
        "kind": "study5-pairs",
        "pairs": rows,
        "r": r_values,
        "theta0_trainable_sha256": theta0_sha,
        "surrogate_sha256": surrogate.load_spec(args.surrogate).sha256 if args.r else None,
        "numerics": numerics,
        "device": device_name(torch.device(env.device)),
        "git": git_state(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_new(args.out, record)
    for row in rows:
        ids = f"{row['a'].get('sha256', '')[:12]} {row['b'].get('sha256', '')[:12]}"
        print(f"pair {ids} dot {row['dot']:.6e} cosine {row['cosine']} "
              f"distance {row['distance']:.6e}")  # fmt: skip
    for key, value in r_values.items():
        print(f"r {key} {value:.9e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
