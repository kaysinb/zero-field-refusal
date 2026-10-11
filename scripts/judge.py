"""Label stored responses with the configured judge, or compare labels with reference labels.

    judge.py label CONFIG ITEMS.jsonl OUT.jsonl [key=value ...]
    judge.py agreement REFERENCE.jsonl OTHER.jsonl

``label`` reads items ``{"item_id", "prompt", "response"}`` and appends label records to OUT.jsonl,
which doubles as the cache: finished items are skipped on a rerun. ``agreement`` prints, as JSON,
the agreement of OTHER with the reference labels (manual or imported): OTHER is a judge's label
file or one labeler's labels ``{item_id, label}``, whose agreement is the ceiling a judge can
reach. In OTHER, ``unsure`` counts as unlabeled.
"""

import argparse
import json
import os
import sys

from zero_field_refusal.config import load_config
from zero_field_refusal.judge import (
    UNSURE,
    JudgeClient,
    agreement,
    check_wildguard_tokens,
    count_unparseable,
    read_jsonl,
)
from zero_field_refusal.labeling import load_labels
from zero_field_refusal.paths import load_env


def label(args: argparse.Namespace) -> None:
    load_env()
    cfg = load_config(args.config, args.overrides)
    client = JudgeClient.from_config(cfg, cache_path=args.out, max_workers=args.workers)
    tokens = None
    if client.backend == "wildguard":  # before any item: a doubled BOS would change every label
        tokens = check_wildguard_tokens(
            os.environ["JUDGE_API_BASE"], client.model, os.environ.get("JUDGE_API_KEY")
        )
    records = client.label_items(read_jsonl(args.items))
    print(
        json.dumps(
            {
                "items": len(records),
                "unparseable": count_unparseable(records),
                "backend": client.backend,
                "prompt_version": client.prompt_version,
                "judge_model": client.model,
                "judge_revision": client.revision,
                "token_check": tokens,
            }
        )
    )


def compare(args: argparse.Namespace) -> None:
    reference = {i: record["label"] for i, record in load_labels(args.reference).items()}
    records = read_jsonl(args.other)
    if records and all("backend" in r for r in records):  # a judge's label file
        identities = {
            (r["backend"], r["prompt_version"], r["judge_model"], r["judge_revision"])
            for r in records
        }
        if len(identities) > 1:
            sys.exit(f"OTHER holds labels of several judges; split it: {sorted(identities)}")
        other = {r["item_id"]: r["label"] for r in records}
    else:  # one labeler's labels; the latest record per item wins
        other = {i: r["label"] for i, r in load_labels(args.other).items()}
    other = {i: None if label == UNSURE else label for i, label in other.items()}
    result = agreement(reference, other)
    if result["missing_from_judged"]:
        print(
            f"warning: {result['missing_from_judged']} reference items have no label in OTHER "
            "and count as unlabeled",
            file=sys.stderr,
        )
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("label", help="label items with the configured judge")
    p.add_argument("config")
    p.add_argument("items")
    p.add_argument("out")
    p.add_argument("overrides", nargs="*", help="key=value config overrides")
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(run=label)
    p = sub.add_parser("agreement", help="agreement of labels with the reference labels")
    p.add_argument("reference")
    p.add_argument("other")
    p.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
