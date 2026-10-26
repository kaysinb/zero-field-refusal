"""Run the missing runs of a campaign one after another (zero_field_refusal.launch).

    python scripts/launch.py CAMPAIGN.yaml [--dry-run] [--recovery REASON] [--only LABEL]
        [--max-attempts N]

Checks every run's config first, and refuses to start if a completed run with a campaign run's
config cannot stand for it (wrong gate, unclean checkout, missing artifacts). Skips runs that
already have a qualifying completed run, never reruns a diverged one, retries technical failures
up to three attempts in all, and runs the rest in order, each in its own process with its own
RUN_ID. --dry-run lists what would run. --recovery REASON, once the technical cause of exhausted
runs is fixed, opens the one recovery round of their cells (logged with the reason). Exits non-zero
if any run used up its attempts.

--only LABEL (a run's label, ``<cell>/s<seed>``) assesses and runs that run alone: every other
run of the campaign is neither read nor started, and the journal records the label; a recovery
round under it is the run's own. --max-attempts N bounds the attempts of a run (default 3, and N
more in a recovery round); study 3's qualification driver passes 1 (zero_field_refusal.launch).

SIGTERM to this process (not to its process group) stops the campaign at a run boundary: the
running attempt, a child process the signal does not reach, finishes and is recorded; then the
launcher exits non-zero without starting another. Started again, the campaign carries on.
"""

import argparse
import json
import sys
from pathlib import Path

from zero_field_refusal.launch import (
    MAX_ATTEMPTS,
    CampaignError,
    StopRequest,
    campaign_header,
    load_campaign,
    run_campaign,
)
from zero_field_refusal.paths import load_env, runs_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="list the runs that would start")
    parser.add_argument("--recovery", metavar="REASON", help="the fixed technical cause")
    parser.add_argument("--only", metavar="LABEL", help="assess and run this run alone")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS, metavar="N",
                        help=f"attempts per run (default {MAX_ATTEMPTS})")  # fmt: skip
    args = parser.parse_args(argv)
    load_env()
    name, runs = load_campaign(args.campaign)
    header = campaign_header(args.campaign)  # logged with every start (study 2's w_max)
    stop = StopRequest.on_sigterm()
    try:
        result = run_campaign(name, runs, runs_root(), dry_run=args.dry_run,
                              recovery=args.recovery, stop=stop, header=header,
                              only=args.only, max_attempts=args.max_attempts)  # fmt: skip
    except CampaignError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({key: labels for key, labels in result.items() if labels}, indent=1))
    if result["exhausted"]:
        sys.exit(f"{len(result['exhausted'])} runs used up their attempts; see "
                 f"RUNS_DIR/launch/{name}.jsonl")  # fmt: skip
    if result["stopped"]:
        sys.exit(f"Stopped on request after the running attempt; {len(result['stopped'])} runs "
                 f"left, started again by the same command")  # fmt: skip


if __name__ == "__main__":
    main()
