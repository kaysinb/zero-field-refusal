"""The cleaning of study 2's harmful evaluation set: pool sizes and pairs for a blind labeling.

    python scripts/harmful_pool_cleaning.py counts PROMPT_SETS_DIR
    python scripts/harmful_pool_cleaning.py pairs PROMPT_SETS_DIR --out PAIRS.jsonl

PROMPT_SETS_DIR is a study-1 build (DATA_DIR/prompts/<config hash>/). ``counts`` prints only
numbers: the pool outside the 100 K_harmful prompts (judged and calibration sets) after greedy
deduplication in source order at word-set Jaccard thresholds 0.8 to 0.4, without and with the
prompts near a K prompt. ``pairs`` writes, for a blind labeling of whether two prompts ask for the
same thing, up to 40 pairs per Jaccard band in [0.4, 0.8) (fixed seed), within the pool and
against the K prompts. PAIRS.jsonl holds harmful prompt texts: write it outside Git (scratchpad/
or DATA_DIR) and do not print it. The labels of 2026-10-14 are in configs/data/study2.yaml
(``harmful.evaluation.basis``) and docs/study2-preregistration.md ("Panels").
"""

import argparse
import json
import random
from pathlib import Path

from zero_field_refusal.data import _normalize

BANDS = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]


def load(root: Path, name: str) -> list[dict]:
    return [json.loads(line) for line in (root / f"{name}.jsonl").read_text().splitlines()]


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("counts", "pairs"))
    parser.add_argument("sets", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    kl = load(args.sets, "harmful_kl")
    pool = load(args.sets, "harmful_judged") + load(args.sets, "harmful_calibration")
    pool.sort(key=lambda i: (not i["id"].startswith("harmbench"), int(i["id"].split(":")[1])))
    words = {i["id"]: frozenset(_normalize(i["text"]).split()) for i in kl + pool}
    kl_sets = [words[i["id"]] for i in kl]
    if args.command == "counts":
        near = {i["id"]: max(jaccard(words[i["id"]], k) for k in kl_sets) for i in pool}
        print("pool", len(pool))
        for tau in (0.8, 0.7, 0.6, 0.5, 0.4):
            kept: list[dict] = []
            for item in pool:
                if not any(jaccard(words[item["id"]], words[k["id"]]) >= tau for k in kept):
                    kept.append(item)
            clean = [i for i in kept if near[i["id"]] < tau]
            harmbench = sum(i["id"].startswith("harmbench") for i in clean)
            print(
                f"jaccard {tau}: deduplicated {len(kept)}, and not near a K prompt "
                f"{len(clean)} (HarmBench {harmbench})"
            )
        return
    if args.out is None:
        parser.error("pairs needs --out")
    text = {i["id"]: i["text"] for i in kl + pool}
    found: dict = {(kind, band): [] for kind in ("pool", "kl") for band in BANDS}
    for n, a in enumerate(pool):
        for kind, others in (("pool", pool[n + 1 :]), ("kl", kl)):
            for b in others:
                j = jaccard(words[a["id"]], words[b["id"]])
                for band in BANDS:
                    if band[0] <= j < band[1]:
                        found[(kind, band)].append((a["id"], b["id"], j))
    rng = random.Random(2026)
    with args.out.open("w") as f:
        for (kind, band), pairs in found.items():
            chosen = rng.sample(pairs, min(40, len(pairs)))
            print(kind, band, "pairs", len(pairs), "sampled", len(chosen))
            for a, b, j in chosen:
                f.write(
                    json.dumps(
                        {
                            "kind": kind,
                            "band": list(band),
                            "jaccard": round(j, 3),
                            "a": text[a],
                            "b": text[b],
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
