"""Draw the blind sample of the judge check across optimizers (zero_field_refusal.judge_check).

    python scripts/make_judge_check_sample.py RUNS_ROOT --out DIR [--round 1] [--seed 0]
        [--with-100x5]

Reads the 1.5B main-condition runs (RUNS_ROOT/stage1_main_1.5b; with --with-100x5 also
RUNS_ROOT/stage1_sgd100x5_1.5b), chooses each stratum's cell from K alone and writes into DIR,
which must not exist: sample.jsonl (what the labelers see), key.json (origin of every item; keep it
unread until the reference labels are done) and selection.json (strata, cells, K*). Round 2 is the
pre-specified addition of 100 further prompts per stratum. Prints counts and cells, never texts.
The files hold model responses to harmful prompts: keep them out of git.
"""

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from zero_field_refusal.data import load_prompt_set
from zero_field_refusal.judge_check import (
    draw_sample,
    runs_table,
    select_strata,
    selection_record,
    trained_responses,
)
from zero_field_refusal.labeling import write_calibration_files

MAIN, SGD_100X5 = "stage1_main_1.5b", "stage1_sgd100x5_1.5b"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", type=int, default=1, dest="round_")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--with-100x5", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: a drawn sample is never overwritten")
    runs = {MAIN: runs_table(args.runs_root, MAIN)}
    if args.with_100x5:
        runs[SGD_100X5] = runs_table(args.runs_root, SGD_100X5)
    strata = select_strata(runs[MAIN], runs.get(SGD_100X5), main_experiment=MAIN,
                           experiment_100x5=SGD_100X5)  # fmt: skip
    first = args.runs_root / MAIN / runs[MAIN]["run_id"].iloc[0]
    cfg = OmegaConf.load(first / "config.yaml")
    prompts = {p["id"]: p["text"] for p in load_prompt_set(cfg.eval.harmful.prompt_set, cfg.data)}

    def responses(experiment: str, run_id: str) -> dict[str, str]:
        return trained_responses(args.runs_root / experiment / run_id)

    blinded, key = draw_sample(strata, runs, responses, prompts, round_=args.round_,
                               seed=args.seed)  # fmt: skip
    args.out.mkdir(parents=True)
    write_calibration_files(blinded, key, args.out / "sample.jsonl", args.out / "key.json")
    record = selection_record(strata, args.round_, args.seed)
    (args.out / "selection.json").write_text(json.dumps(record, indent=1) + "\n")
    for stratum in record["strata"]:
        print(json.dumps(stratum))
    print(json.dumps({"items": len(blinded), "out": str(args.out)}))


if __name__ == "__main__":
    main()
