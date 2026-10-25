"""Evaluate the starting model θ₀ (no training): the baseline of refusal, harm and utility.

    python scripts/evaluate_theta0.py CONFIG [key=value ...]

Sets run_type=theta0, so the run's config hash differs from training runs of the same config.
The precision and backend flags are set first, before any CUDA work (runtime.configure_numerics),
as in every entry point that computes: study 3's engineering stage checks θ₀'s utility through
this script.
"""

import sys

from zero_field_refusal.config import from_cli
from zero_field_refusal.protocol import check_run_config
from zero_field_refusal.runtime import configure_numerics, environment
from zero_field_refusal.train import evaluate_theta0


def main() -> None:
    cfg = from_cli([*sys.argv[1:], "run_type=theta0"])
    check_run_config(cfg)
    configure_numerics(cfg, cfg.execution.device)  # first: before any CUDA work
    print(evaluate_theta0(cfg, environment(cfg, cfg.execution.device)))


if __name__ == "__main__":
    main()
