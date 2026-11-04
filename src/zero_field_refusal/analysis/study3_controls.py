"""Study 3's judge control items: 40 archived prompt–response pairs of study 1's judge calibration,
drawn once, pinned by their hashes, and checked whenever they are used.

docs/study3-preregistration.md, "Judge and audit" ("Control items"): every labeling batch includes
40 fixed control items, immutable archived pairs from study 1's judge calibration, drawn once by a
recorded seed, 20 that WildGuard labeled harmful there and 20 not. Each is pinned by its id, the
sha256 of its prompt and of its response, and its expected label; an item whose content hash
differs is refused, so that no regenerated or changed input can stand in for it.

- ``select_controls``: the draw, from the calibration archive (``ARCHIVE`` under RUNS_DIR: the
  WildGuard labels ``LABELS_FILE`` and the item file they label, ``ITEMS_FILE``), as the record.
- ``load_controls``: the 40 items of a record, read from the archive and each verified against
  its two sha256, for a labeling; ``ControlsRefused`` when any is not.
- ``python -m zero_field_refusal.analysis.study3_controls select [--archive DIR] [--out PATH]``
  writes the record once (``RECORD``, by default) and prints its counts and sha256 only.

**The draw.** The item file is the one the labels label: every label record names an item of it
by id, with the judge's content hash (``judge._content_hash``) of that item's prompt and response;
a label of anything else refuses the draw. Only parsed labels count (a null label, an unparsed
output, is left out), all of one judge identity. Harmful is WildGuard's ``harmful_compliance``;
not harmful is ``refusal`` or ``safe_non_refusal``. With one ``random.Random(SEED)``, 20 ids are
drawn from the harmful ids in sorted order, then 20 from the not-harmful ids in sorted order
(``random.Random.sample``); the items keep that order.

**The record** (``docs/study3-judge-controls.json``, ``KIND``): ``seed``, ``method``,
``source`` (``root`` ``runs``, the archive's ``directory`` relative to RUNS_DIR, and the
``labels`` and ``items`` files by ``name`` and ``sha256``), ``identity`` (the judge's, as its label
records carry it), ``counts`` and ``items``, each ``item_id``, ``prompt_sha256`` and
``response_sha256`` (the sha256 of the UTF-8 text) and ``expected_label``, WildGuard's label
there (one of ``judge.LABELS``). No text, and no time stamp: the same archive gives the same bytes.

**Rulings** where the registration leaves a detail open:
- *The expected label is the label itself*, not only its two classes: a control whose label
  moves between ``refusal`` and ``safe_non_refusal`` has changed too (the judge is frozen with
  its template, parser and decoding, greedy).
- *What is verified at use* is each item's two sha256, as registered; the files' sha256 are
  recorded for provenance and not required, so that only a changed control item refuses.

Nothing here prints a prompt, a response or a judge output: only ids, hashes, labels and counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..judge import IDENTITY_FIELDS, LABELS, _content_hash
from ..paths import REPO_ROOT, load_env, runs_root
from .evidence import HARMFUL

KIND = "study3-judge-controls"
SEED = 20261104
PER_CLASS = 20  # harmful, and not harmful
ARCHIVE = "judge_calibration/calibration"  # under RUNS_DIR
LABELS_FILE = "judge-wildguard.jsonl"
ITEMS_FILE = "sample.jsonl"
RECORD = "docs/study3-judge-controls.json"  # in the checkout
NOT_HARMFUL = tuple(label for label in LABELS if label != HARMFUL)
METHOD = (f"random.Random(seed).sample: {PER_CLASS} of the ids WildGuard labeled "
          f"{HARMFUL}, in sorted order, then {PER_CLASS} of the ids it labeled "
          f"{' or '.join(NOT_HARMFUL)}, in sorted order, with one generator; parsed labels "
          "only")  # fmt: skip
FIELDS = ("kind", "seed", "method", "source", "identity", "counts", "items")
ITEM_FIELDS = ("item_id", "prompt_sha256", "response_sha256", "expected_label")


class ControlsRefused(ValueError):
    """A control record or archive that cannot be used: the item named, never its text."""


def text_sha256(text: str) -> str:
    """The sha256 of a text's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _jsonl(data: bytes, name: str) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise ControlsRefused(f"{name}, line {number}, is not JSON") from None
        if not isinstance(row, dict) or not isinstance(row.get("item_id"), str):
            raise ControlsRefused(f"{name}, line {number}, names no item")
        rows.append(row)
    return rows


def _items(data: bytes, name: str) -> dict[str, dict[str, Any]]:
    """The item file's rows by id; ``ControlsRefused`` for an id twice or a row without texts."""
    found: dict[str, dict[str, Any]] = {}
    for row in _jsonl(data, name):
        if not (isinstance(row.get("prompt"), str) and isinstance(row.get("response"), str)):
            raise ControlsRefused(f"{name}: {row['item_id']} has no prompt or response text")
        if found.setdefault(row["item_id"], row) is not row:
            raise ControlsRefused(f"{name}: {row['item_id']} appears twice")
    return found


def select_controls(archive: str | Path, *, directory: str = ARCHIVE) -> dict[str, Any]:
    """The control record drawn from the calibration archive ``archive`` (the folder holding
    ``LABELS_FILE`` and ``ITEMS_FILE``; ``directory``: that folder relative to RUNS_DIR, as the
    record names it). See the module docstring."""
    archive = Path(archive)
    labels_data = (archive / LABELS_FILE).read_bytes()
    items_data = (archive / ITEMS_FILE).read_bytes()
    items = _items(items_data, ITEMS_FILE)
    labels: dict[str, str] = {}
    identities = set()
    for row in _jsonl(labels_data, LABELS_FILE):
        item = items.get(row["item_id"])
        if item is None:
            raise ControlsRefused(f"{LABELS_FILE}: {row['item_id']} is no item of {ITEMS_FILE}")
        if row.get("content_sha256") != _content_hash(item["prompt"], item["response"]):
            raise ControlsRefused(f"{LABELS_FILE}: the label of {row['item_id']} is not of the "
                                  f"item {ITEMS_FILE} holds")  # fmt: skip
        identities.add(tuple(row.get(key) for key in IDENTITY_FIELDS))
        label = row.get("label")
        if label is None:
            continue  # unparsed: no label
        if label not in LABELS or labels.setdefault(row["item_id"], label) != label:
            raise ControlsRefused(f"{LABELS_FILE}: {row['item_id']} has the label {label!r} "
                                  "twice differently or outside the labels")  # fmt: skip
    if len(identities) != 1:
        raise ControlsRefused(f"{LABELS_FILE}: the labels are of {len(identities)} judge "
                              "identities, not one")  # fmt: skip
    harmful = sorted(i for i, label in labels.items() if label == HARMFUL)
    not_harmful = sorted(i for i, label in labels.items() if label in NOT_HARMFUL)
    if min(len(harmful), len(not_harmful)) < PER_CLASS:
        raise ControlsRefused(f"{LABELS_FILE}: fewer than {PER_CLASS} parsed labels of a class")
    rng = random.Random(SEED)
    drawn = rng.sample(harmful, PER_CLASS) + rng.sample(not_harmful, PER_CLASS)
    return {
        "kind": KIND,
        "seed": SEED,
        "method": METHOD,
        "source": {"root": "runs", "directory": directory,
                   "labels": {"name": LABELS_FILE,
                              "sha256": hashlib.sha256(labels_data).hexdigest()},
                   "items": {"name": ITEMS_FILE,
                             "sha256": hashlib.sha256(items_data).hexdigest()}},
        "identity": dict(zip(IDENTITY_FIELDS, identities.pop(), strict=True)),
        "counts": {"harmful": PER_CLASS, "not_harmful": PER_CLASS},
        "items": [{"item_id": i, "prompt_sha256": text_sha256(items[i]["prompt"]),
                   "response_sha256": text_sha256(items[i]["response"]),
                   "expected_label": labels[i]} for i in drawn],
    }  # fmt: skip


def _checked(record: Any) -> list[dict[str, str]]:
    """The record's items, its form checked: ``ControlsRefused`` otherwise."""
    if not isinstance(record, dict) or set(record) != set(FIELDS) or record["kind"] != KIND:
        raise ControlsRefused(f"not a {KIND} record (its fields are {list(FIELDS)})")
    items = record["items"]
    hexes = ("prompt_sha256", "response_sha256")
    if not (isinstance(items, list) and all(
            isinstance(e, dict) and set(e) == set(ITEM_FIELDS) and isinstance(e["item_id"], str)
            and all(isinstance(e[k], str) and len(e[k]) == 64 for k in hexes)
            and e["expected_label"] in LABELS for e in items)):  # fmt: skip
        raise ControlsRefused(f"its items are not {list(ITEM_FIELDS)} each")
    harmful = sum(e["expected_label"] == HARMFUL for e in items)
    if (len(items) != 2 * PER_CLASS or harmful != PER_CLASS
            or len({e["item_id"] for e in items}) != len(items)):  # fmt: skip
        raise ControlsRefused(f"it does not hold {PER_CLASS} harmful and {PER_CLASS} not harmful "
                              "items, each once")  # fmt: skip
    return items


def load_controls(record: Mapping[str, Any], archive: str | Path) -> list[dict[str, str]]:
    """The control items of ``record`` (the record as read) from the archive folder ``archive``
    (by default ``RUNS_DIR/<record's source directory>``), in the record's order: ``item_id``
    (``control/<id>``, apart from any run's items), ``prompt``, ``response`` and
    ``expected_label``. Each is verified against its two sha256; one that differs, or is not in
    the item file, refuses the whole call (``ControlsRefused``, naming the item id)."""
    entries = _checked(record)
    name = record["source"]["items"]["name"]
    items = _items((Path(archive) / name).read_bytes(), name)
    out = []
    for entry in entries:
        item = items.get(entry["item_id"])
        if item is None:
            raise ControlsRefused(f"the control item {entry['item_id']} is not in {name}")
        if (text_sha256(item["prompt"]) != entry["prompt_sha256"]
                or text_sha256(item["response"]) != entry["response_sha256"]):  # fmt: skip
            raise ControlsRefused(f"the control item {entry['item_id']}: its content hash differs "
                                  "from the record's: refused")  # fmt: skip
        out.append(
            {
                "item_id": f"control/{entry['item_id']}",
                "prompt": item["prompt"],
                "response": item["response"],
                "expected_label": entry["expected_label"],
            }
        )
    return out


def record_bytes(record: Mapping[str, Any]) -> bytes:
    """The record's bytes as written."""
    return (json.dumps(record, indent=1, ensure_ascii=False) + "\n").encode()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m zero_field_refusal.analysis.study3_controls",
                                     description=__doc__.splitlines()[0])  # fmt: skip
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="draw the control items and write the record once")
    select.add_argument("--archive", type=Path,
                        help=f"the calibration archive (default $RUNS_DIR/{ARCHIVE})")  # fmt: skip
    select.add_argument("--out", type=Path, default=REPO_ROOT / RECORD)
    args = parser.parse_args(argv)
    load_env()
    archive = args.archive if args.archive is not None else runs_root() / ARCHIVE
    try:
        record = select_controls(archive)
    except ControlsRefused as exc:
        parser.exit(1, f"refused: {exc}\n")
    data = record_bytes(record)
    try:  # written once
        handle = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        parser.exit(1, f"refused: {args.out} exists; the controls are drawn once\n")
    with os.fdopen(handle, "wb") as out:
        out.write(data)
    print(json.dumps({**record["counts"], "record": str(args.out),
                      "sha256": hashlib.sha256(data).hexdigest()}))  # fmt: skip


if __name__ == "__main__":
    main()
