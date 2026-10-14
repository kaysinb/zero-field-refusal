"""Measure the top curvature of the training loss at θ₀ (study 2, Stage A); GPU required.

    python scripts/curvature.py CONFIG [--batch-sizes 5 20] [--batches 4] [--starts 3]
        [--pooled-responses 160] [--pooled-starts 2] [--steps 20] [key=value ...]

Reads the appendable training stream of the config's seed (extend it first with
generate_teacher.py --responses N), measures the top curvature of single batches at each batch
size and of the Hessian pooled over the first responses by Lanczos on Hessian–vector products,
and writes curvature.json and the summary to a run directory (run_type=curvature). The settings
are stored under ``curvature`` in the run's config.
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
    parser.add_argument("--batches", type=int, default=4, help="single batches per batch size")
    parser.add_argument("--starts", type=int, default=3, help="start vectors per single batch")
    parser.add_argument("--pooled-responses", type=int, default=160)
    parser.add_argument("--pooled-starts", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20, help="Lanczos steps")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, [*overrides, "run_type=curvature"])
    settings = {key: value for key, value in vars(args).items() if key != "config"}
    cfg = OmegaConf.merge(cfg, {"curvature": settings})
    check_run_config(cfg)
    print(run_curvature(cfg, environment(cfg, cfg.execution.device)))


if __name__ == "__main__":
    main()
