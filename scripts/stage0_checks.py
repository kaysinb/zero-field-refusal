"""Run the stage-0 checks of the real pipeline on one model; GPU required.

    python scripts/stage0_checks.py experiments/stage0_checks [model=qwen2.5-1.5b-instruct ...]

Generates the streams the checks need (training seeds 0-2, 200 fresh batches for the field, the
path-field batches with recorded distributions) unless they exist, then writes all results to a
run directory and the environment's path-field record under ``DATA_DIR/checks/``.
"""

from zero_field_refusal.checks import run_stage0_checks
from zero_field_refusal.config import from_cli
from zero_field_refusal.protocol import check_run_config
from zero_field_refusal.runtime import environment


def main() -> None:
    cfg = from_cli()
    check_run_config(cfg)
    run = run_stage0_checks(cfg, environment(cfg, cfg.execution.device))
    print(run.dir)


if __name__ == "__main__":
    main()
