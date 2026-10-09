"""Train and evaluate one run; its teacher streams must already exist.

python scripts/train.py CONFIG [key=value ...]
"""

from zero_field_refusal.config import from_cli
from zero_field_refusal.protocol import check_run_config
from zero_field_refusal.runtime import environment
from zero_field_refusal.train import run_training


def main() -> None:
    cfg = from_cli()
    check_run_config(cfg)
    run_dir = run_training(cfg, environment(cfg, cfg.execution.device))
    print(run_dir)


if __name__ == "__main__":
    main()
