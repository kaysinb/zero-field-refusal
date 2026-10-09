"""Evaluate the starting model θ₀ (no training): the baseline of refusal, harm and utility.

    python scripts/evaluate_theta0.py CONFIG [key=value ...]

Sets run_type=theta0, so the run's config hash differs from training runs of the same config.
"""

import sys

from zero_field_refusal.config import from_cli
from zero_field_refusal.protocol import check_run_config
from zero_field_refusal.runtime import environment
from zero_field_refusal.train import evaluate_theta0


def main() -> None:
    cfg = from_cli([*sys.argv[1:], "run_type=theta0"])
    check_run_config(cfg)
    print(evaluate_theta0(cfg, environment(cfg, cfg.execution.device)))


if __name__ == "__main__":
    main()
