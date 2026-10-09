"""Label stored responses with the configured judge, or compare judge and manual labels.

    judge.py label CONFIG ITEMS.jsonl OUT.jsonl [key=value ...]
    judge.py agreement MANUAL.jsonl JUDGED.jsonl

``label`` reads items ``{"item_id", "prompt", "response"}`` and appends label records to OUT.jsonl,
which doubles as the cache: finished items are skipped on a rerun. ``agreement`` prints the
agreement metrics as JSON.
"""

import argparse
import json
import sys

from zero_field_refusal.config import load_config
from zero_field_refusal.judge import JudgeClient, agreement, count_unparseable, read_jsonl
from zero_field_refusal.labeling import load_labels
from zero_field_refusal.paths import load_env


def label(args: argparse.Namespace) -> None:
    load_env()
    cfg = load_config(args.config, args.overrides)
    client = JudgeClient.from_config(cfg, cache_path=args.out, max_workers=args.workers)
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
            }
        )
    )


def compare(args: argparse.Namespace) -> None:
    manual = {item_id: record["label"] for item_id, record in load_labels(args.manual).items()}
    records = read_jsonl(args.judged)
    identities = {
        (r["backend"], r["prompt_version"], r["judge_model"], r["judge_revision"]) for r in records
    }
    if len(identities) > 1:
        sys.exit(f"JUDGED holds labels from several judges; split the file: {sorted(identities)}")
    judged = {record["item_id"]: record["label"] for record in records}
    result = agreement(manual, judged)
    if result["missing_from_judged"]:
        print(
            f"warning: {result['missing_from_judged']} manually labeled items have no judge "
            "record and count as unlabeled",
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
    p = sub.add_parser("agreement", help="agreement of judge labels with manual labels")
    p.add_argument("manual")
    p.add_argument("judged")
    p.set_defaults(run=compare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
