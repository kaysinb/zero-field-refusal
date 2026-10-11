"""Write the Gate 1 report of stage 1 from the run directories of one model.

    python scripts/gate1_report.py RUNS_ROOT --judge NAME --out DIR [--model 1.5b]
        [--calibration FILE]

Reads RUNS_ROOT/stage1_main_<model>/ (training runs and the θ₀ evaluation),
RUNS_ROOT/stage1_token_count_<model>/ and, on 1.5B, RUNS_ROOT/stage1_sgd100x5_1.5b/ when present,
with the labels of the judge NAME (configs/judge/NAME.yaml; label files <run dir>/judge/NAME.jsonl),
and writes gate1.json, cells.csv and manifest.json into DIR, which must not exist yet: the effect
criterion, collapse, stability and completeness per cell, K_lo and K_hi with their case, SGD
coverage and its route, the "effect disappears" rule, and the Gate 1 verdict (gate1: passes, next
= stage2-equal-KL, stage2-equal-KL-100x5-needs-amendment, fallback, stop, undetermined (a
diverged seed of AdamW 5e-5) or pending, and what blocks it; see
zero_field_refusal.analysis.gate1). --calibration takes the judge calibration result (the JSON of
scripts/judge.py agreement) whose accuracy is recorded against the threshold; without it Gate 1
is pending. On 0.5B only the effect, collapse and the cells are computed. No model, judge or GPU
is used.

Put DIR outside the checkout or under an ignored path such as RUNS_DIR: the manifest records
whether the checkout is dirty, and an untracked DIR inside it would make it so.
"""

import argparse
from pathlib import Path

from zero_field_refusal.analysis.gate1 import GATE_MODEL, MODELS, gate1_report
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import judge_identity


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_root", type=Path, help="the runs root, e.g. $RUNS_DIR")
    parser.add_argument("--judge", required=True, help="a judge config in configs/judge/")
    parser.add_argument("--out", type=Path, required=True, help="a new output directory")
    parser.add_argument("--model", choices=MODELS, default=GATE_MODEL)
    parser.add_argument("--calibration", type=Path, help="a judge-calibration result (JSON)")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: results are versioned, never overwritten")
    identity = judge_identity(load_config(f"judge/{args.judge}"))
    try:
        out = gate1_report(args.runs_root, args.judge, identity, args.out, model=args.model,
                           calibration=args.calibration)  # fmt: skip
    except ValueError as exc:
        raise SystemExit(f"Not written: {exc}") from exc
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
