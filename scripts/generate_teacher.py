"""Sample teacher streams from θ₀ and store them under DATA_DIR/teacher/.

    python scripts/generate_teacher.py CONFIG --stream KIND [--seeds 0 1 ...] [key=value ...]

KIND is one of training, preconditioner, kl_harmful, kl_benign, soft_targets, alpaca_reference.
K's reference streams (kl_*) take no seed: they are sampled once per model. Existing streams are
never overwritten; they are the record of the experiment. After an interrupted generation, rerun
with --remove-partial (once nothing is generating) to delete its partial directory.
"""

import argparse

from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import environment, generate_stream, remove_partial_streams
from zero_field_refusal.teacher import STREAM_KINDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--stream", required=True, choices=STREAM_KINDS)
    parser.add_argument("--seeds", type=int, nargs="*", default=[])
    parser.add_argument(
        "--remove-partial",
        action="store_true",
        help="delete partial directories of interrupted generations, then exit",
    )
    args, overrides = parser.parse_known_args()
    cfg = load_run_config(args.config, overrides)
    env = environment(cfg, cfg.execution.device)
    seeds = [None] if args.stream.startswith("kl_") else args.seeds
    if not seeds:
        parser.error(f"--seeds is required for {args.stream} streams")
    if args.remove_partial:
        for seed in seeds:
            for directory in remove_partial_streams(cfg, env, args.stream, seed):
                print(f"removed {directory}")
        return
    model = None if args.stream == "alpaca_reference" else env.new_model()
    for seed in seeds:
        path = generate_stream(cfg, env, args.stream, seed, model=model,
                               batch_size=cfg.execution.generation_batch_size)  # fmt: skip
        print(path)


if __name__ == "__main__":
    main()
