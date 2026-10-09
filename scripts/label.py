"""Blind manual labeling: label.py SAMPLE.jsonl LABELS.jsonl

Shows each item's prompt and response; the key file of the sample is never read. Keys: r refusal,
s safe non-refusal, h harmful compliance, u unsure, b back, q quit. Answers are appended to
LABELS.jsonl at once, and a restart resumes at the first unlabeled item.
"""

import argparse

from zero_field_refusal.labeling import read_jsonl, run_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sample")
    parser.add_argument("labels")
    parser.add_argument("--width", type=int, default=100)
    args = parser.parse_args()
    run_session(read_jsonl(args.sample), args.labels, width=args.width)


if __name__ == "__main__":
    main()
