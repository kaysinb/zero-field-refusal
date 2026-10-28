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

    python scripts/launch.py --resolve-torn LOG --offset N --sha256 HEX --reason TEXT

While the launch journal is not whole, the launcher starts nothing, settles nothing and opens
no recovery round: it prints what is wrong and exits 12 (TORN_JOURNAL). The journal is not whole
while a log under RUNS_DIR/launch/ holds a torn line that no torn_resolved record names, or a
published qualification record under RUNS_DIR/qualification/ (a cycle's supersede-<n>.json, a
case's restart-<n>.json) requires a held or superseded line that no log holds
(zero_field_refusal.launch, "The journal"):
- a torn line is any line of a log that is not a JSON object, a blank line included; the tail a
  crash leaves in the middle of an append is one case of it. Each is printed with its log,
  offset, length, kind and sha256. Once what the line's kind lost is restored (docs/operations.md,
  "Launching runs"), --resolve-torn, with no campaign file, records the resolution in that log
  (LOG: its file name, as printed): it first writes the line's torn record if none names it, then
  the torn_resolved record. It refuses an offset and sha256 that name no torn line of the log, a
  line already resolved and an empty reason;
- a missing line is printed with its record's path, its event and its run id. It is lifted by
  the record's own repair alone: scripts/qualify.py supersede run again with the cycle's record,
  or scripts/qualify.py restart-case for the restart's case. An unreadable record, or an entry
  that names no run, blocks too and is restored by hand.
"""

import argparse
import json
import sys
from pathlib import Path

from zero_field_refusal.launch import (
    MAX_ATTEMPTS,
    TORN_JOURNAL,
    CampaignError,
    StopRequest,
    TornJournalError,
    campaign_header,
    load_campaign,
    resolve_torn,
    run_campaign,
)
from zero_field_refusal.paths import load_env, runs_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("campaign", type=Path, nargs="?")
    parser.add_argument("--dry-run", action="store_true", help="list the runs that would start")
    parser.add_argument("--recovery", metavar="REASON", help="the fixed technical cause")
    parser.add_argument("--only", metavar="LABEL", help="assess and run this run alone")
    parser.add_argument("--max-attempts", type=int, metavar="N",
                        help=f"attempts per run (default {MAX_ATTEMPTS})")  # fmt: skip
    parser.add_argument("--resolve-torn", metavar="LOG", help="resolve a torn line of this log")
    parser.add_argument("--offset", type=int, metavar="N", help="the torn line's offset")
    parser.add_argument("--sha256", metavar="HEX", help="the torn line's sha256")
    parser.add_argument("--reason", metavar="TEXT", help="what was restored, and from what")
    args = parser.parse_args(argv)
    resolution = (args.offset, args.sha256, args.reason)
    if args.resolve_torn is not None:
        if (args.campaign is not None or args.dry_run or args.recovery is not None
                or args.only is not None or args.max_attempts is not None):  # fmt: skip
            parser.error("--resolve-torn takes no campaign file and no campaign option")
        if None in resolution:
            parser.error("--resolve-torn needs --offset N, --sha256 HEX and --reason TEXT")
        load_env()
        try:
            record = resolve_torn(runs_root(), args.resolve_torn, offset=args.offset,
                                  sha256=args.sha256, reason=args.reason)  # fmt: skip
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps(record))
        return
    if args.campaign is None:
        parser.error("a campaign file is needed (or --resolve-torn LOG)")
    if any(value is not None for value in resolution):
        parser.error("--offset, --sha256 and --reason go with --resolve-torn")
    max_attempts = MAX_ATTEMPTS if args.max_attempts is None else args.max_attempts
    load_env()
    name, runs = load_campaign(args.campaign)
    header = campaign_header(args.campaign)  # logged with every start (study 2's w_max)
    stop = StopRequest.on_sigterm()
    try:
        result = run_campaign(name, runs, runs_root(), dry_run=args.dry_run,
                              recovery=args.recovery, stop=stop, header=header,
                              only=args.only, max_attempts=max_attempts)  # fmt: skip
    except TornJournalError as exc:
        print(exc, file=sys.stderr)
        sys.exit(TORN_JOURNAL)
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
