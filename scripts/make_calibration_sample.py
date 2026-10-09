"""Draw the blinded calibration sample: make_calibration_sample.py POOL.jsonl SAMPLE.jsonl KEY.json

POOL.jsonl holds items with ``prompt``, ``response`` and the hidden condition fields (e.g. model
and whether the response is from theta_0 or a trained model). SAMPLE.jsonl is what the labeler
sees; KEY.json maps its opaque item ids to everything else and must stay unread until the labels
are done. Both contain model responses to harmful prompts: keep them out of git.
"""

import argparse

from zero_field_refusal.labeling import make_calibration_sample, read_jsonl, write_calibration_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pool")
    parser.add_argument("sample")
    parser.add_argument("key")
    parser.add_argument("--strata", required=True, help="comma-separated pool fields")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--min-per-stratum", type=int, default=10)
    args = parser.parse_args()
    blinded, key = make_calibration_sample(
        read_jsonl(args.pool),
        args.n,
        strata_key=args.strata.split(","),
        seed=args.seed,
        min_per_stratum=args.min_per_stratum,
    )
    write_calibration_files(blinded, key, args.sample, args.key)
    print(f"wrote {len(blinded)} items to {args.sample} and the key to {args.key}")


if __name__ == "__main__":
    main()
