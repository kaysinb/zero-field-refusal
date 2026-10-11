"""Rebuild the analysis tables, results and figure of one experiment from its run directories.

    python scripts/build_analysis.py RUNS_ROOT --experiment NAME --judge NAME --out DIR [--seed N]
        [--k-star K_STAR_JSON]

Reads every run directory RUNS_ROOT/<experiment>/<run id>/, failed runs included, with the labels
of the judge NAME (configs/judge/NAME.yaml; label files <run dir>/judge/NAME.jsonl), and runs the
Gate 2 analysis with the stage-1 interval settings (percentile CI at 99.5 %, 4000 resamples,
bootstrap seed --seed). Writes runs and exclusions, cell, arm and estimate tables, k_star.json,
results.json, figure.svg and manifest.json into DIR, which must not exist yet (see
zero_field_refusal.analysis.build). No model, judge or GPU is used.

Without --k-star, K* is provisional: it is selected from the analyzed runs themselves, which suits
the stage-0 fixture check and the stage-1 sweep, and results.json says so (k_star_status). It is
not the Gate 2 K*, which is fixed from the sweep's KL values before look 1 and never re-estimated,
so the build refuses when any run of the experiment is a Gate 2 run (tracking.gate: 2). For a
Gate 2 look, pass --k-star with the k_star.json of the sweep build that fixed K*; the build uses
it unchanged and refuses one computed from Gate 2 runs. Look membership is not an input yet: build
the look-1 analysis before any look-2 run joins the experiment.

Put DIR outside the checkout or under an ignored path such as RUNS_DIR: the manifest records
whether the checkout is dirty, and an untracked DIR inside it would make it so.
"""

import argparse
from pathlib import Path

from zero_field_refusal.analysis.build import build_analysis
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import judge_identity


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_root", type=Path, help="the runs root, e.g. $RUNS_DIR")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--judge", required=True, help="a judge config in configs/judge/")
    parser.add_argument("--out", type=Path, required=True, help="a new output directory")
    parser.add_argument("--seed", type=int, default=0, help="seed of the paired bootstrap")
    parser.add_argument("--k-star", type=Path, help="frozen K*: an earlier build's k_star.json")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: results are versioned, never overwritten")
    identity = judge_identity(load_config(f"judge/{args.judge}"))
    try:
        out = build_analysis(args.runs_root, args.experiment, args.judge, identity, args.out,
                             seed=args.seed, k_star=args.k_star)  # fmt: skip
    except ValueError as exc:
        raise SystemExit(f"Not built: {exc}") from exc
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
