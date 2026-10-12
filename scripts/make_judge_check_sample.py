"""Draw the blind sample of the judge check across optimizers (zero_field_refusal.judge_check).

    python scripts/make_judge_check_sample.py RUNS_ROOT --out DIR [--round 1] [--seed 0]

Reads the 1.5B runs of RUNS_ROOT/stage1_main_1.5b and RUNS_ROOT/stage1_sgd100x5_1.5b (K,
divergence and utility, never a judge label), finds the eligible cells and freezes the audit cells
of every panel (``judge_check.select_panels``): equal K with K* from all runs and from eligible
cells, the two fallback comparisons and the Gate 1 cell, each cell drawn once. Writes into DIR,
which must not exist: sample.jsonl (the items), unique.jsonl (each distinct prompt and response
once, what the labelers see), key.json (origin and unique id of every item; keep it unread until
the reference labels are done) and selection.json (cells, panels, strata, families and the hashes
of the two sample files). Round 2 is the pre-specified addition of 100 further prompts per
stratum. Prints counts and cells, never texts. The files hold model responses to harmful prompts:
keep them out of git.
"""

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

from zero_field_refusal.analysis.sweeps import read_logical_runs, theta0_utility
from zero_field_refusal.data import load_prompt_set
from zero_field_refusal.judge_check import (
    FAMILY,
    cell_table,
    draw_sample,
    runs_table,
    select_panels,
    selection_record,
    trained_responses,
    unique_texts,
)
from zero_field_refusal.labeling import write_calibration_files

MAIN, SGD_100X5 = "stage1_main_1.5b", "stage1_sgd100x5_1.5b"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", type=int, default=1, dest="round_")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: a drawn sample is never overwritten")
    runs = {name: runs_table(args.runs_root, name) for name in (MAIN, SGD_100X5)}
    theta0 = theta0_utility(read_logical_runs(args.runs_root, MAIN))
    cells = {name: cell_table(args.runs_root, name, theta0) for name in runs}
    strata, panels = select_panels(runs[MAIN], runs[SGD_100X5], cells[MAIN], cells[SGD_100X5],
                                   main_experiment=MAIN, experiment_100x5=SGD_100X5)  # fmt: skip
    first = args.runs_root / MAIN / runs[MAIN]["run_id"].iloc[0]
    cfg = OmegaConf.load(first / "config.yaml")
    prompts = {p["id"]: p["text"] for p in load_prompt_set(cfg.eval.harmful.prompt_set, cfg.data)}

    def responses(experiment: str, run_id: str) -> dict[str, str]:
        return trained_responses(args.runs_root / experiment / run_id)

    blinded, key = draw_sample(strata, runs, responses, prompts, round_=args.round_,
                               seed=args.seed)  # fmt: skip
    unique, unique_of = unique_texts(blinded, args.round_)
    for item_id, unique_id in unique_of.items():
        key[item_id]["unique_id"] = unique_id
    args.out.mkdir(parents=True)
    write_calibration_files(blinded, key, args.out / "sample.jsonl", args.out / "key.json")
    lines = [json.dumps(u, ensure_ascii=False) + "\n" for u in unique]
    (args.out / "unique.jsonl").write_text("".join(lines))
    record = selection_record(strata, args.round_, args.seed)
    record |= {
        "theta0": theta0,
        "cells": {name: json.loads(t.to_json(orient="records")) for name, t in cells.items()},
        "panels": panels,
        "family": FAMILY,
        "files": {name: {"items": len((args.out / name).read_text().splitlines()),
                         "sha256": hashlib.sha256((args.out / name).read_bytes()).hexdigest()}
                  for name in ("sample.jsonl", "unique.jsonl")},
    }  # fmt: skip
    (args.out / "selection.json").write_text(json.dumps(record, indent=1) + "\n")
    for stratum in record["strata"]:
        print(json.dumps({k: stratum[k] for k in ("name", "experiment", "lrs", "cell_k")}))
    for name, panel in panels.items():
        print(json.dumps({"panel": name, "criteria": panel["criteria"]}))
    print(json.dumps({"out": str(args.out), **record["files"]}))


if __name__ == "__main__":
    main()
