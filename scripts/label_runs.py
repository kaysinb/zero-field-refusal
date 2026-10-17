"""Label the responses of run directories with a judge, for the analysis build.

    python scripts/label_runs.py --judge NAME RUN_DIR [RUN_DIR ...] [--workers N] [--rounds N]

For each completed run, labels every response of its harmful prompt set (all conditions: the
trained model, θ₀, line scans, random directions) with the judge configs/judge/NAME.yaml into
<run dir>/judge/NAME.jsonl, keyed <condition>/<prompt id> (zero_field_refusal.run_judging). The
label file is the cache: a rerun labels only what is missing, including outputs that did not
parse. Needs JUDGE_API_BASE (and JUDGE_API_KEY for a keyed server); a WildGuard server first passes
the tokenizer check. A run whose responses file is not the complete set its config implies is
reported with its error and the other runs go on. Prints counts per run, never texts; exits
non-zero while any run has an error or an unparseable output. A run with unparseable outputs left
is labeled again, at most --rounds rounds in all (1 to 3, default 3) counted across every
invocation: <run dir>/judge/NAME.rounds.json counts each round before its requests are sent, so
an interrupted round counts too, and a rerun resumes with the rounds left (none once they are
used: it then sends nothing). A reply already in the label file is reused and is no round. The
summary shows `rounds` (the rounds used in all) when not 1, and its `unparseable` is the number of
items that stay without a label. Study-2 runs (training runs with a `schedule` in their config)
are labeled at every evaluation they recorded, whatever their stop reason, once their terminal
record was published with whole records, as the launcher decides it, whatever their status (a
Rayleigh measurement that failed afterwards leaves it `failed`); one that stopped before its first
evaluation is skipped.

Once a run's rounds are over, its judging receipt <run dir>/judge/NAME.receipt.json records the
items expected, the judge's identity, the rounds used in all, the labels still missing and the
label file's sha256. A round deletes the receipt before it starts, and an interrupted one leaves
none: study 2's analysis counts no label of a run without its receipt (the run stays pending), and
counts a missing label only after all three rounds. A run with a completed receipt (no label
missing, or its rounds used) is left as it is, nothing sent (`kept` in its summary). A label file
without its rounds record (labeled before the record existed) is refused: its rounds are unknown.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from zero_field_refusal.config import load_config
from zero_field_refusal.data import load_prompt_set
from zero_field_refusal.judge import JudgeClient, check_wildguard_tokens
from zero_field_refusal.paths import load_env
from zero_field_refusal.run_judging import label_runs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--judge", required=True, help="a judge config in configs/judge/")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        choices=range(1, 4),
        help="labeling rounds in all per run and judge, across reruns",
    )
    args = parser.parse_args(argv)
    load_env()
    cfg = load_config(f"judge/{args.judge}")
    probe = JudgeClient.from_config(cfg)
    tokens = None
    if probe.backend == "wildguard":  # before any item: a doubled BOS would change every label
        tokens = check_wildguard_tokens(
            os.environ["JUDGE_API_BASE"], probe.model, os.environ.get("JUDGE_API_KEY")
        )

    def prompts(run_cfg) -> dict[str, str]:
        texts = load_prompt_set(run_cfg.eval.harmful.prompt_set, run_cfg.data)
        return {item["id"]: item["text"] for item in texts}

    summaries = label_runs(
        args.run_dirs,
        args.judge,
        prompts,
        lambda path: JudgeClient.from_config(cfg, cache_path=path, max_workers=args.workers),
        rounds=args.rounds,
    )
    for summary in summaries:
        print(json.dumps(summary))
    print(json.dumps({"runs": len(summaries), "token_check": tokens}))
    if any(s.get("error") for s in summaries):
        sys.exit("Some runs could not be labeled; see their errors above")
    if any(s.get("unparseable") for s in summaries):
        sys.exit(
            "Some outputs did not parse. A rerun resumes each such run within the rounds it "
            "has left (--rounds in all, across reruns) and sends nothing for a run whose rounds "
            "are used"
        )


if __name__ == "__main__":
    main()
