"""Qualify the judge on the frozen judge-check sample (zero_field_refusal.judge_check).

    python scripts/judge_check_qualify.py SAMPLE_DIR --reference LABELS.jsonl --runs-root RUNS
        --judge NAME --out OUT.json

Reads the frozen sample's key.json and selection.json, the reference labels of its distinct
texts (label records keyed by the ids of unique.jsonl; the latest record per id counts) and the
judge's labels in the run directories, held to the judge's identity (configs/judge/NAME.yaml).
Writes OUT.json, which must not exist: for every panel, the qualification of its criteria strata
(and with its 100 x 5 stratum where it has one), the report-only three-class tables of the
not-harmful side (amendment of 2026-10-12), the settings and the sha256 of every input. Which
panel decides follows from the route Gate 1 selects. Prints aggregates only, never texts.
"""

import argparse
import hashlib
import json
from pathlib import Path

from zero_field_refusal.config import load_config
from zero_field_refusal.judge import judge_identity
from zero_field_refusal.judge_check import check_records, not_harmful_tables, qualify_panels
from zero_field_refusal.labeling import load_labels

SETTINGS = {"permutations": 10_000, "posterior_draws": 100_000, "seed": 0}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists: results are never overwritten")
    identity = judge_identity(load_config(f"judge/{args.judge}"))
    reference = {item: record["label"] for item, record in load_labels(args.reference).items()}
    records = check_records(args.sample_dir, reference, args.runs_root, args.judge, identity)
    selection = json.loads((args.sample_dir / "selection.json").read_text())
    panels = qualify_panels(records, selection["panels"], **SETTINGS)
    result = {
        "inputs": {
            "key.json": sha256(args.sample_dir / "key.json"),
            "selection.json": sha256(args.sample_dir / "selection.json"),
            "reference": {"file": args.reference.name, "sha256": sha256(args.reference)},
        },
        "judge": {"name": args.judge, "identity": identity},
        "settings": SETTINGS,
        "items": len(records),
        "panels": panels,
        "not_harmful_tables": not_harmful_tables(records),
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for name, variants in panels.items():
        for variant, outcome in variants.items():
            q = outcome["result"]
            print(json.dumps({"panel": name, "variant": variant, "status": q["status"],
                              "reasons": q["reasons"]}))  # fmt: skip
    print(json.dumps({"out": str(args.out), "items": len(records)}))


if __name__ == "__main__":
    main()
