"""Blind manual labeling for judge calibration.

``make_calibration_sample`` draws a stratified sample from a pool of responses and splits it into
a blinded file (opaque item ids, prompt and response only) and a key (everything else, including
the hidden condition). The labeling session reads only the blinded file and never the key.
"""

from __future__ import annotations

import json
import random
import textwrap
from collections.abc import Callable, Hashable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .judge import LABEL_DEFINITIONS, LABELS, UNSURE, read_jsonl, repair_jsonl

StrataKey = str | Sequence[str] | Callable[[Mapping[str, Any]], Hashable]

ANSWERS = {"r": "refusal", "s": "safe_non_refusal", "h": "harmful_compliance", "u": UNSURE}
assert tuple(ANSWERS.values()) == (*LABELS, UNSURE)


def _stratum_fn(strata_key: StrataKey) -> Callable[[Mapping[str, Any]], Hashable]:
    if callable(strata_key):
        return strata_key
    names = [strata_key] if isinstance(strata_key, str) else list(strata_key)
    return lambda item: tuple(item[name] for name in names)


def _allocate(sizes: Mapping[Hashable, int], n: int, minimum: int) -> dict[Hashable, int]:
    """Proportional allocation of ``n`` over strata, each getting ``minimum`` if it has that many.

    If the minimums do not fit in ``n``, the minimum shrinks to an even share. Remaining slots go
    one at a time to the stratum furthest below its proportional share that still has items; ties
    go to the stratum that sorts first.
    """
    total = sum(sizes.values())
    minimum = min(minimum, n // len(sizes))  # "if possible": never more than an even share
    alloc = {key: min(size, minimum) for key, size in sizes.items()}
    order = sorted(sizes, key=repr)
    for _ in range(n - sum(alloc.values())):
        open_strata = [key for key in order if alloc[key] < sizes[key]]
        best = max(open_strata, key=lambda key: n * sizes[key] / total - alloc[key])
        alloc[best] += 1
    return alloc


def make_calibration_sample(
    pool: Sequence[Mapping[str, Any]],
    n: int = 200,
    *,
    strata_key: StrataKey,
    seed: int,
    min_per_stratum: int = 10,
) -> tuple[list[dict[str, str]], dict[str, dict[str, Any]]]:
    """A stratified, shuffled sample of ``n`` pool items as ``(blinded, key)``.

    ``strata_key`` is a field name, a sequence of field names, or a function of a pool item that
    returns the hidden condition. Pool items need ``prompt`` and ``response``; every other field
    goes to ``key[item_id]``, with the pool's own ``item_id`` renamed ``original_item_id``.
    """
    if n > len(pool):
        raise ValueError(f"Cannot sample {n} items from a pool of {len(pool)}")
    stratum_of = _stratum_fn(strata_key)
    groups: dict[Hashable, list[Mapping[str, Any]]] = {}
    for item in pool:
        groups.setdefault(stratum_of(item), []).append(item)
    alloc = _allocate({key: len(group) for key, group in groups.items()}, n, min_per_stratum)

    rng = random.Random(seed)
    chosen: list[tuple[Hashable, Mapping[str, Any]]] = []
    for stratum in sorted(groups, key=repr):
        chosen += [(stratum, item) for item in rng.sample(groups[stratum], alloc[stratum])]
    rng.shuffle(chosen)

    blinded: list[dict[str, str]] = []
    key: dict[str, dict[str, Any]] = {}
    for stratum, item in chosen:
        item_id = f"c{rng.getrandbits(48):012x}"
        while item_id in key:
            item_id = f"c{rng.getrandbits(48):012x}"
        blinded.append({"item_id": item_id, "prompt": item["prompt"], "response": item["response"]})
        rest = {k: v for k, v in item.items() if k not in ("prompt", "response", "item_id")}
        if "item_id" in item:
            rest["original_item_id"] = item["item_id"]
        key[item_id] = {**rest, "stratum": repr(stratum)}
    return blinded, key


def write_calibration_files(
    blinded: Sequence[Mapping[str, str]],
    key: Mapping[str, Any],
    sample_path: str | Path,
    key_path: str | Path,
) -> None:
    """Write the blinded items as JSONL and the key as one JSON object."""
    Path(sample_path).write_text("".join(json.dumps(b, ensure_ascii=False) + "\n" for b in blinded))
    Path(key_path).write_text(json.dumps(key, ensure_ascii=False, indent=1) + "\n")


# --- labeling session ---------------------------------------------------------------------------


def load_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    """The latest record per item id; a missing file means no labels yet."""
    path = Path(path)
    if not path.exists():
        return {}
    return {record["item_id"]: record for record in read_jsonl(path)}


def _wrap(text: str, width: int) -> str:
    """Wrap each line to ``width`` and keep the text whole (no pager, nothing cut)."""
    lines = [
        textwrap.fill(line, width, replace_whitespace=False) or "" for line in text.split("\n")
    ]
    return "\n".join(lines)


def run_session(
    items: Sequence[Mapping[str, str]],
    labels_path: str | Path,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    width: int = 100,
) -> None:
    """Label ``items`` one by one, appending each answer to ``labels_path`` at once.

    Keys: r refusal, s safe non-refusal, h harmful compliance, u unsure, b back, q quit. The
    session resumes at the first unlabeled item; the latest record per item wins.
    """
    labels_path = Path(labels_path)
    repair_jsonl(labels_path)
    labeled = set(load_labels(labels_path))
    total = len(items)
    position = next((i for i, item in enumerate(items) if item["item_id"] not in labeled), total)
    if position == total:
        output_fn(f"All {total} items are labeled.")
        return
    output_fn("Labels:")
    for definition in LABEL_DEFINITIONS.values():
        output_fn(_wrap("  " + definition, width))
    output_fn(
        _wrap(
            "Keys: r refusal, s safe non-refusal, h harmful compliance, u unsure, b back, q quit",
            width,
        )
    )

    while position < total:
        item = items[position]
        output_fn("")
        output_fn(f"===== item {position + 1}/{total} =====")
        output_fn("PROMPT:")
        output_fn(_wrap(item["prompt"], width))
        output_fn("")
        output_fn("RESPONSE:")
        output_fn(_wrap(item["response"], width))
        output_fn("")
        previous = load_labels(labels_path).get(item["item_id"])
        if previous:
            output_fn(f"(current label: {previous['label']})")
        while True:
            try:
                answer = input_fn("[r/s/h/u/b/q] > ").strip().lower()
            except EOFError:
                answer = "q"
            if answer in ANSWERS or answer in ("b", "q"):
                break
            output_fn("Use r, s, h, u, b or q.")
        if answer == "q":
            output_fn(f"Stopped at item {position + 1}/{total}; rerun to resume.")
            return
        if answer == "b":
            position = max(position - 1, 0)
            continue
        record = {
            "item_id": item["item_id"],
            "label": ANSWERS[answer],
            "labeled_at": datetime.now(UTC).isoformat(),
        }
        with labels_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        labeled.add(item["item_id"])
        position = next(
            (i for i in range(position + 1, total) if items[i]["item_id"] not in labeled), total
        )
    output_fn(f"All {total} items are labeled.")
