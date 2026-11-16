"""Study 5's line scan of one fresh seed's η_l run (docs/study5-preregistration.md, "Descriptive
measurements", "Line scans"), in a run of its own from that run's endpoint checkpoint, written as
one JSON record.

    python scripts/study5_scan.py CONFIG --f-run RUN_DIR --checkpoints DIR --out RECORD.json
        [--run-id ID] [--points A ...] [key=value ...]

CONFIG and the overrides are those of the seed's fresh F run at η_l (``experiments/study5``,
``study5.role=fresh``, its rate and its ``seed``), one seed per call; RUN_DIR is that run's
directory (``ladder.read_f_run``: its config hash, all its ``train.steps`` updates), and its
endpoint checkpoint is ``DIR/<run id>.pt``, the file the run saved, its sha256 the recorded one
(``line_scan.checkpoint_of``). In order (``line_scan.run_scan``): θ₀'s model, its trainable
tensors hashed against the run's θ₀; the checkpoint's tensors, exactly the trainable set's; then
for each registered point a ∈ {−1, −0.5, −0.25, +0.25, +0.5} the trainable parameters set to
θ₀ + aΔθ and evaluated in full by the F run's evaluator (its panels, prompts and greedy
decoding) under the condition ``line_scan_{a:+g}``, in a run directory of its own (experiment
``study5_scan``, run type ``study5_scan``, id ID when given), one row per point as each ends.
The record holds the seed, η, the points, which were evaluated, each point's K and utilities, the
F run's identities, the checkpoint's sha256, θ₀'s and the endpoint's trainable hashes, the
streams' identities, the device memory peak, the numerics in force, the git state and the time.
CONFIG may also be a dose run's (``experiments/study5_dose``, its rates and seed); ``--points``
scans a subset of the registered points only (the dose experiment: ``--points -0.5``,
``line_scan.scan_points``), and the record then holds the run's ``rate_schedule``.
stdout carries numbers, ids and hashes only. An existing record is never overwritten. Exits 0
when written, 1 on a refusal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from zero_field_refusal import ladder, line_scan, linear_response
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import configure_numerics, device_name, environment
from zero_field_refusal.tracking import git_state


def write_new(path: Path, record: dict) -> None:
    """Write ``record`` to ``path`` atomically; refuse an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; a record is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(partial, path)


def _number(value) -> str:
    return "None" if value is None else f"{value:.9e}"


def report(record: dict) -> None:
    """Numbers, ids and hashes only."""
    print(f"seed {record['seed']} lr {record['lr']:.6e} f_run {record['f_run']['run_id']} "
          f"run {record['experiment']}/{record['run_id']}")  # fmt: skip
    print(f"checkpoint {record['checkpoint']['name']} sha256 {record['checkpoint']['sha256']}")
    print(f"theta0_trainable_sha256 {record['theta0_trainable_sha256']}")
    print(f"endpoint_trainable_sha256 {record['endpoint_trainable_sha256']}")
    for row in record["rows"]:
        values = " ".join(f"{key} {_number(row.get(key))}" for key in line_scan.KEYS)
        print(f"point {row['a']:+g} {row['condition']} evaluated {row['evaluated']} {values}")
    for key, value in record["memory"].items():
        print(f"memory {key} {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--f-run", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--points", type=float, nargs="+", default=None)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"record exists: {args.out}")
        return 1
    cfg = load_run_config(args.config, args.overrides)  # the protocol: fp32, frozen modules
    try:
        linear_response.require_frozen_adam(cfg)
        line_scan.require_scannable(cfg)
        line_scan.scan_points(args.points)
        f_run = ladder.read_f_run(args.f_run, cfg)  # its config hash this config's
        checkpoint = line_scan.checkpoint_of(args.f_run, f_run, args.checkpoints,
                                             int(cfg.train.steps))  # fmt: skip
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    numerics = configure_numerics(cfg, cfg.execution.device)  # first: before any CUDA work
    env = environment(cfg, cfg.execution.device)
    try:
        record = line_scan.run_scan(cfg, env, f_run, checkpoint, run_id=args.run_id,
                                    points=args.points)
    except ValueError as exc:
        print(exc)
        return 1
    record |= {
        "numerics": numerics,
        "device": device_name(torch.device(env.device)),
        "git": git_state(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    write_new(args.out, record)
    report(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
