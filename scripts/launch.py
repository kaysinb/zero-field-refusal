"""Run the missing runs of a campaign one after another (zero_field_refusal.launch).

    python scripts/launch.py CAMPAIGN.yaml [--dry-run]

Checks every run's config first; skips runs whose config hash already has a completed run in
RUNS_DIR/<experiment>/; runs the rest in order, each in its own process, and keeps going after a
failure. --dry-run lists what would run. Exits non-zero if any run failed.
"""

import argparse
import json
import sys
from pathlib import Path

from zero_field_refusal.launch import load_campaign, run_campaign
from zero_field_refusal.paths import load_env, runs_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="list the runs that would start")
    args = parser.parse_args(argv)
    load_env()
    name, runs = load_campaign(args.campaign)
    result = run_campaign(name, runs, runs_root(), dry_run=args.dry_run)
    print(json.dumps({key: labels for key, labels in result.items() if labels}, indent=1))
    if result["failed"]:
        sys.exit(f"{len(result['failed'])} runs failed; see RUNS_DIR/launch/{name}.jsonl")


if __name__ == "__main__":
    main()
