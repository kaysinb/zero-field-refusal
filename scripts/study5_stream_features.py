"""Study 5's predefined stream-content features of one seed (docs/study5-preregistration.md,
"Descriptive measurements", "Stream-content features"), written as one JSON record.

    python scripts/study5_stream_features.py CONFIG --out RECORD.json [--features SPEC]
        [key=value ...]

CONFIG and the overrides are those of the seed's F run, one seed per call. The responses are the
``train.steps × train.batch_size`` training responses the run trains on, read as the run reads
them (``linear_response.feed_batches``); under ``train.order_seed`` the set is the same, so the
features do not depend on the order. Nothing is generated, and no model is loaded.

A stream stores a response as its token ids up to and including the first end id, or truncated at
``teacher.sampling.max_new_tokens`` (96) without one. With n the position of the first id of the
run's ``env.end_ids`` in the first ``max_tokens`` (96) ids (none: n = the number of those ids):

- ``mean_tokens``: the mean of n, the tokens before the end token (the end token is not counted);
- ``ended_share``: the share of responses with an end id in the first ``max_tokens`` ids;
- ``phrase_share``: the share of responses whose text contains any phrase of SPEC (default
  ``configs/study5/stream_features.yaml``);
- ``phrase_share_by_phrase``: that share for each phrase alone, in the list's order.

A response's text is its ids decoded in memory by the run's tokenizer with
``skip_special_tokens=True``. Matching is a case-insensitive substring match after the typographic
apostrophes ’ and ‘ are replaced by ' in the text and in the phrases. A decoded text is never
stored, printed or logged. The record holds the features, the training stream's identity as the
run records it, the sha256 of SPEC's mapping, the tokenizer and model revision, the git state and
the time; stdout carries numbers only. An existing record is never overwritten. Exits 0 when
written, 1 on an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf

from zero_field_refusal import linear_response
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import environment
from zero_field_refusal.tracking import git_state

SPEC_PATH = REPO_ROOT / "configs" / "study5" / "stream_features.yaml"
APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})


def normalize(text: str) -> str:
    """Typographic apostrophes to ', lower case."""
    return text.translate(APOSTROPHES).lower()


def load_spec(path: str | Path = SPEC_PATH) -> dict:
    """The feature spec: ``phrases``, ``max_tokens``, ``version`` and ``sha256`` of the canonical
    JSON of the file's ``stream_features`` mapping."""
    node = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)["stream_features"]
    phrases = [str(p) for p in node["phrases"]]
    if not phrases or any(not normalize(p).strip() for p in phrases):
        raise ValueError("The feature list needs at least one non-empty phrase")
    blob = json.dumps(node, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return {"version": int(node["version"]), "max_tokens": int(node["max_tokens"]),
            "phrases": phrases, "sha256": digest}  # fmt: skip


def tokens_before_end(
    ids: Sequence[int], end_ids: Sequence[int], max_tokens: int
) -> tuple[int, bool]:
    """(n, ended): n ids before the first end id within the first ``max_tokens`` ids, or all of
    those ids if there is none; ``ended`` whether one was found."""
    ends = set(end_ids)
    window = list(ids[:max_tokens])
    for i, token in enumerate(window):
        if token in ends:
            return i, True
    return len(window), False


def features(
    responses: Sequence[Sequence[int]], decode, end_ids: Sequence[int], spec: dict
) -> dict:
    """The features of ``responses`` (id sequences); ``decode`` maps ids to text."""
    if not responses:
        raise ValueError("No responses")
    phrases = [normalize(p) for p in spec["phrases"]]
    hits = [0] * len(phrases)
    any_hit = ended = tokens = 0
    for ids in responses:
        n, was_ended = tokens_before_end(ids, end_ids, spec["max_tokens"])
        tokens += n
        ended += was_ended
        text = normalize(decode(list(ids)))
        found = [phrase in text for phrase in phrases]
        any_hit += any(found)
        hits = [h + f for h, f in zip(hits, found, strict=True)]
    count = len(responses)
    return {"responses": count, "phrase_share": any_hit / count,
            "phrase_share_by_phrase": [h / count for h in hits],
            "mean_tokens": tokens / count, "ended_share": ended / count}  # fmt: skip


def run(cfg, env, spec: dict) -> dict:
    """The record of ``cfg``'s seed in ``env``."""
    batches, training = linear_response.feed_batches(cfg, env)  # fails on a short stream
    responses = [s.response_ids for batch in batches for s in batch]
    return {
        "kind": "study5-stream-features",
        "seed": int(cfg.seed),
        "features": features(responses, env.decode, env.end_ids, spec),
        "streams": {"training": training},
        "features_sha256": spec["sha256"],
        "features_version": spec["version"],
        "max_tokens": spec["max_tokens"],
        "end_ids": [int(i) for i in env.end_ids],
        "model": {"name": cfg.model.name, "revision": cfg.model.revision},
        "git": git_state(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip


def write_new(path: Path, record: dict) -> None:
    """Write ``record`` to ``path`` atomically; refuse an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; a record is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(partial, path)


def report(record: dict) -> None:
    """Numbers and hashes only."""
    f = record["features"]
    print(f"seed {record['seed']} responses {f['responses']}")
    print(f"phrase_share {f['phrase_share']:.6f}")
    print("phrase_share_by_phrase " + " ".join(f"{x:.6f}" for x in f["phrase_share_by_phrase"]))
    print(f"mean_tokens {f['mean_tokens']:.6f}")
    print(f"ended_share {f['ended_share']:.6f}")
    print(f"stream {record['streams']['training']['sha256']}")
    print(f"features_sha256 {record['features_sha256']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=SPEC_PATH)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"record exists: {args.out}")
        return 1
    cfg = load_run_config(args.config, args.overrides)
    spec = load_spec(args.features)
    record = run(cfg, environment(cfg, cfg.execution.device), spec)
    write_new(args.out, record)
    report(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
