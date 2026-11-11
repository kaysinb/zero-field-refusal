"""Points θ₀ + a·v along one parameter-space vector (docs/study5-preregistration.md,
"Extension 2"), written as one JSON record.

    python scripts/study5_vector_points.py CONFIG --label LABEL
        (--vector FILE | --f-run RUN_DIR --checkpoints DIR) [--sign -1] [--points A …]
        [--full A …] [--match] [--extensions N] --out RECORD.json [--run-id ID]
        [--surrogate SPEC] [--panel-size N] [key=value ...]

CONFIG and the overrides are a fresh F run's (``study5.role=fresh``, its rate and its ``seed``):
its evaluator's panels and seeds serve every point. The vector is a ``vectors`` file or, with
``--f-run`` and ``--checkpoints``, that F run's own displacement θ_N − θ₀ from its endpoint
checkpoint (``ladder.read_f_run`` checks the run against CONFIG; ``line_scan.checkpoint_of`` the
file against the run's record). In order (``vector_points.run_vector``): at every a of --points
(default 1) K_benign, K_harmful and r with its refusal and compliance pieces on both surrogate
panels; at the points of --full also the full evaluation under ``vector_<label>_<a>``; with
--match the ladder's scale search and matched-K step from the vector's own scale (at most N
extensions, default 8) and the readout and full evaluation at the chosen scale under
``vector_<label>_kstar``; all in a run directory of its own (experiment ``study5_vector``, id ID
when given). stdout carries numbers, ids and hashes only. An existing record is never
overwritten. Exits 0 when written, 1 on a refusal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from zero_field_refusal import ladder, line_scan, linear_response, surrogate, vector_points
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


def report(record: dict) -> None:
    """Numbers, ids and hashes only."""
    print(f"label {record['label']} sign {record['sign']} run {record['experiment']}/"
          f"{record['run_id']} vector_norm {record['vector_norm']:.9e}")  # fmt: skip
    for row in record["rows"]:
        numbers = " ".join(f"{key} {row[key]:.6e}" for key in
                           ("a", "K_benign", "K_harmful", "r", "harmful_R", "harmful_C",
                            "benign_R", "benign_C") if isinstance(row.get(key), float))  # fmt: skip
        print(f"point {row['condition']} evaluated {row['evaluated']} {numbers}")
    matched = record["matched"]
    if matched is not None:
        print(f"match status {matched['status']} c {matched.get('c')}")
    print(f"theta0_trainable_sha256 {record['theta0_trainable_sha256']}")
    for key, value in record["vector"].items():
        print(f"vector {key} {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--label", required=True)
    parser.add_argument("--vector", type=Path, default=None)
    parser.add_argument("--f-run", type=Path, default=None)
    parser.add_argument("--checkpoints", type=Path, default=None)
    parser.add_argument("--sign", type=int, choices=(1, -1), default=1)
    parser.add_argument("--points", type=float, nargs="+", default=[1.0])
    parser.add_argument("--full", type=float, nargs="*", default=[])
    parser.add_argument("--match", action="store_true")
    parser.add_argument("--extensions", type=int, default=vector_points.EXTENSIONS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--surrogate", type=Path, default=surrogate.SPEC_PATH)
    parser.add_argument("--panel-size", type=int, default=100)
    args = parser.parse_args(argv)
    if (args.vector is None) == (args.f_run is None) or (args.f_run is None) != (
        args.checkpoints is None
    ):
        print("give --vector, or --f-run with --checkpoints")
        return 1
    if args.extensions < 0 or len(set(args.points)) != len(args.points):
        print("--extensions must be non-negative and --points distinct")
        return 1
    if args.out.exists():
        print(f"record exists: {args.out}")
        return 1
    cfg = load_run_config(args.config, args.overrides)  # the protocol: fp32, frozen modules
    f_run = checkpoint = None
    try:
        linear_response.require_frozen_adam(cfg)
        ladder.require_fresh(cfg)
        vector_points.vector_condition_check(args.label, args.points)
        if args.f_run is not None:
            f_run = ladder.read_f_run(args.f_run, cfg)
            checkpoint = line_scan.checkpoint_of(args.f_run, f_run, args.checkpoints,
                                                 int(cfg.train.steps))  # fmt: skip
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    numerics = configure_numerics(cfg, cfg.execution.device)  # first: before any CUDA work
    env = environment(cfg, cfg.execution.device)
    try:
        record = vector_points.run_vector(
            cfg, env, label=args.label, vector=args.vector, f_run=f_run, checkpoint=checkpoint,
            sign=args.sign, points=args.points, full=args.full, match=args.match,
            extensions=args.extensions, surrogate_spec=args.surrogate,
            panel_size=args.panel_size, run_id=args.run_id,
        )  # fmt: skip
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
