"""Measure the top curvature of the training loss at θ₀ (study 2, Stage A); GPU required.

    python scripts/curvature.py CONFIG [--batch-sizes 5 20] [--batches 3] [--starts 2]
        [--pooled-responses 100] [--pooled-starts 2] [--min-steps 40] [--max-steps 60]
        [--tolerance 0.01]
        [key=value ...]

Reads the appendable training stream of the config's seed (extend it first with
generate_teacher.py --responses N), measures the top curvature of single batches at each batch
size and of the Hessian pooled over the first responses by Lanczos on Hessian–vector products,
and writes curvature.json and the summary to a run directory (run_type=curvature). The settings
are stored under ``curvature`` in the run's config.

Cost of the defaults on one L40S at 1.5B, from an HVP of ≈ 3.2 training steps (measured at 0.5B
on an RTX 5090, 2026-10-14) and the pilot's step times: ≈ 3 s per product at batch 5, ≈ 6 s at
batch 20, ≈ 29 s pooled over 100 responses; 40–60 products per start, ≈ 1.4–1.9 h in all.
"""

import argparse

from omegaconf import OmegaConf

from zero_field_refusal.config import load_config
from zero_field_refusal.curvature import run_curvature
from zero_field_refusal.protocol import check_run_config
from zero_field_refusal.runtime import environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--batches", type=int, default=3, help="single batches per batch size")
    parser.add_argument("--starts", type=int, default=2, help="start vectors per single batch")
    parser.add_argument("--pooled-responses", type=int, default=100)
    parser.add_argument("--pooled-starts", type=int, default=2)
    parser.add_argument("--min-steps", type=int, default=40, help="Lanczos steps at least")
    parser.add_argument("--max-steps", type=int, default=60, help="Lanczos steps at most")
    parser.add_argument(
        "--tolerance", type=float, default=0.01, help="relative residual and agreement of starts"
    )
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, [*overrides, "run_type=curvature"])
    settings = {key: value for key, value in vars(args).items() if key != "config"}
    cfg = OmegaConf.merge(cfg, {"curvature": settings})
    check_run_config(cfg)
    print(run_curvature(cfg, environment(cfg, cfg.execution.device)))


if __name__ == "__main__":
    main()
