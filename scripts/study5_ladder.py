"""One rung of study 5's ladder of synthetic displacements for one seed
(docs/study5-preregistration.md, "Descriptive measurements", "Ladder"), written as one JSON record.

    python scripts/study5_ladder.py CONFIG --rung RUNG --f-run RUN_DIR --out RECORD.json
        [--run-id ID] [key=value ...]

CONFIG and the overrides are those of the seed's fresh F run (``study5.role=fresh``, its rate
``optimizer.lr`` = η and its ``seed``), one seed and one rung per call; RUN_DIR is that run's
directory, which must hold the config hash of CONFIG and all its ``train.steps`` updates. RUNG is
one of ``gaussian_d``, ``linear_no_d``, ``linear``, ``linear_shuffled_d``. The seed's streams must
be in the data cache (nothing is generated). In order (``ladder.run_rung``): δ of the rung at θ₀,
c₀ = ‖Δθ‖/‖δ‖ with ‖Δθ‖ the F run's recorded step-N displacement norm, K_benign at c₀/2, c₀, 2c₀
(extended by ×2 or ÷2 at most three times), ln K interpolated in ln c to K* = 6.00 (study 3's), K
measured at that c₁ and, outside ±10 % of K*, one refinement step c₂ (``ladder.refine``), and the
full evaluation at the one of c₁, c₂ nearer K* in ln K under the condition ``ladder_<rung>``, in a
run directory of its own (experiment ``study5_ladder``, id ID when given); a rung still outside
the tolerance is evaluated there and flagged ``out_of_tolerance``. The record holds the rung, the
seed, η, c₀, ‖δ‖, every (c, K_benign), the pair, c₁, c₂ and their K, the chosen c and the flag,
the full evaluation, the F run's endpoint, the
hashes of v₀ and θ₀, the streams' identities, the seeds of the random draws, the device memory
peak, the numerics in force, the git state and the time. stdout carries numbers, ids and hashes
only. An existing record is never overwritten. Exits 0 when written (also for a rung that is not
bracketed), 1 on a refusal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

from zero_field_refusal import ladder, linear_response
from zero_field_refusal.evaluation import LADDER_RUNGS
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
    print(f"seed {record['seed']} rung {record['rung']} lr {record['lr']:.6e} "
          f"status {record['status']}")  # fmt: skip
    print(f"f_run {record['f_run']['run_id']} run {record['experiment']}/{record['run_id']} "
          f"condition {record['condition']}")  # fmt: skip
    print(f"displacement_norm {record['displacement_norm']:.9e} delta_norm "
          f"{record['delta_norm']:.9e} c0 {record['c0']:.9e}")  # fmt: skip
    search = record["search"]
    for scale in search["scales"]:
        print(f"scale {scale['c']:.9e} K_benign {scale['K_benign']:.9e}")
    print(f"extensions {search['extensions']} monotone {search['monotone']} "
          f"choice {search['choice']}")  # fmt: skip
    if search["pair"] is not None:
        print("pair " + " ".join(f"{c:.9e}" for c in search["pair"]["c"]))
    matched = record["matched_k"] or {}
    for key in ("c1", "K1", "c2", "K2"):
        if key in matched:
            print(f"{key} {_number(matched[key])}")
    print(f"c {_number(record['c'])} k_star {record['k_star']} "
          f"out_of_tolerance {record['out_of_tolerance']}")  # fmt: skip
    for key, value in (record["evaluation"] or {}).items():
        print(f"evaluation {key} {_number(value)}")
    for key, value in record["f_run"]["endpoint"].items():
        if key != "condition":
            print(f"endpoint {key} {_number(value)}")
    print(f"v0_sha256 {record['v0_sha256']}")
    print(f"theta0_trainable_sha256 {record['theta0_trainable_sha256']}")
    for kind, identity in record["streams"].items():
        if identity is not None:
            print(f"stream {kind} {identity['sha256']} count {identity['count']}")
    for key, value in record["memory"].items():
        print(f"memory {key} {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--rung", required=True, choices=LADDER_RUNGS)
    parser.add_argument("--f-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"record exists: {args.out}")
        return 1
    cfg = load_run_config(args.config, args.overrides)  # the protocol: fp32, frozen modules
    try:
        linear_response.require_frozen_adam(cfg)
        ladder.require_fresh(cfg)
        f_run = ladder.read_f_run(args.f_run, cfg)
    except ValueError as exc:
        print(exc)
        return 1
    numerics = configure_numerics(cfg, cfg.execution.device)  # first: before any CUDA work
    env = environment(cfg, cfg.execution.device)
    record = ladder.run_rung(cfg, env, args.rung, f_run, run_id=args.run_id)
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
