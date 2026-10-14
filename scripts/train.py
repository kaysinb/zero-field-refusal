"""Train and evaluate one run; its teacher streams must already exist.

python scripts/train.py CONFIG [key=value ...]

A config with a ``schedule`` (study 2) runs under it (schedule.run_scheduled): lazy batches,
monitor readings, evaluations at K levels and stops; its training stream may be extended by the
run itself if ``schedule.extend_stream``.
"""

from zero_field_refusal.config import from_cli
from zero_field_refusal.protocol import check_run_config
from zero_field_refusal.runtime import environment
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.train import run_training


def main() -> None:
    cfg = from_cli()
    check_run_config(cfg)
    env = environment(cfg, cfg.execution.device)
    run_dir = run_scheduled(cfg, env) if cfg.get("schedule") else run_training(cfg, env)
    print(run_dir)


if __name__ == "__main__":
    main()
