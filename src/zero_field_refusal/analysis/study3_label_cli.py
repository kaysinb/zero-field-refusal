"""Study 3's labeling command: the control list's one-time draw, and the control items' fetch and
check.

    python -m zero_field_refusal.analysis.study3_label_cli controls-draw --calibration DIR \
        --seed S --archive URI [--runs-root DIR]

docs/study3-preregistration.md: "Judge and audit" (every labeling batch includes 40 fixed control
items: immutable archived prompt–response pairs from study 1's judge calibration, drawn once by a
recorded seed, 20 that WildGuard labeled harmful there and 20 not, each pinned by its id, the
sha256 of its prompt and of its response, and its expected label; an item whose content hash
differs is refused; a changed control label leads to relabeling the batch) and "Seeds and order".

**Where study 1's calibration set lives.** Locally, ``RUNS_DIR/judge_calibration/calibration/``
(study 1's Gate-1 command; docs/reports/gate-1.md): ``sample.jsonl``, the blind sample
``{item_id, prompt, response}`` with ids ``c`` and 12 hexadecimal digits, whose sha256 is fixed
in code (``analysis.gate1.CALIBRATION_SAMPLE_SHA256``); ``judge-wildguard.jsonl``, WildGuard's
labels, the latest record per id, as ``gate1._agreement`` reads them; and the reference labels.
It is archived in the create-only bucket; at labeling time the folder is restored from there,
read only, and given as ``--controls-source``. Every item is checked by id and both hashes.

**The draw** (``draw_controls``; ``controls-draw`` publishes it once at
``docs/study3-judge-controls.json``, ``study3_pins.CONTROLS``, refused when it exists; the
researcher commits it alone, before the validation fixation, which pins it):
1. the sample's sha256 must be the fixed one;
2. the latest label record per id, each carrying the identity of ``configs/judge/wildguard.yaml``;
3. the sample's items with a non-null label split into harmful (``harmful_compliance``) and not
   harmful, at least 20 of each;
4. ``random.Random(seed)``: 20 of each, each drawn from the sorted ids, listed by id;
5. each item with the sha256 of its prompt's and of its response's UTF-8 bytes, its study-1 label
   and its expected label; the record checked by ``study3_pins.control_list_problems``.
The record holds ids, hashes and labels, never a text; the command prints the counts, the path
and the sha256.

**The fetch** (``fetch_controls``) refuses, before any request: a source sample whose sha256 is
not the list's, and (``ControlMismatch``, naming the ids, never a text) an id that is missing or
doubled, or whose prompt or response hash differs. **The check**: ``label_controls`` labels all
40 on a cache file of its own, which must not exist (no control label is reused), in up to three
rounds while a label is null; ``changed_controls`` lists the ids whose two-class outcome
(``harmful_compliance`` or not) differs from the expected one, a null counting as changed, and
those whose label parsed (a parsed mismatch: a control mismatch).

**Rulings** where the registration leaves a detail open:
- *The two-class outcome.* The registration's judge has a two-class outcome; a control changes
  when its outcome differs from the expected one. The three-class label is kept as information.
- *A null control label* (an output that did not parse within the rounds) counts as changed, as
  a parsed mismatch does; only the parsed ones are a control mismatch in the sense that rules out
  an infrastructure cause.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import load_config
from ..judge import IDENTITY_FIELDS, LABELS, JudgeClient, count_unparseable, judge_identity
from ..paths import REPO_ROOT, load_env, runs_root
from ..tracking import git_state
from .study3_decisions import _publish
from .study3_labels import JUDGE_CONFIG, ROUNDS
from .study3_pins import (
    CONTROL_ITEMS,
    CONTROLS,
    CONTROLS_KIND,
    CONTROLS_WHAT,
    EXPECTED,
    HARMFUL,
    PER_CLASS,
    SOURCE_FILES,
    control_list_problems,
)
from .study3_records import under_sealed

PROG = "python -m zero_field_refusal.analysis.study3_label_cli"
CONTROL_PREFIX = "control/"  # a control item's id as the judge client sees it


class ControlMismatch(ValueError):
    """Control items of the list that the source does not hold once, or holds with another
    prompt or response: refused before any request, by id."""

    def __init__(self, ids: Sequence[str], why: str) -> None:
        self.ids = list(ids)
        super().__init__(f"control items {self.ids} {why}: refused before any request")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jsonl(path: Path, data: bytes) -> list[dict[str, Any]]:
    """Every line of a JSONL file as an object; ``ValueError`` naming the line's number."""
    rows = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}, line {number}, is not JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}, line {number}, is no object")
        rows.append(row)
    return rows


def _source_file(folder: Path, name: str, root: Path) -> tuple[Path, bytes]:
    path = folder / name
    if under_sealed(path, root):
        raise ValueError(f"{path} resolves under a sealed folder: never opened")
    try:
        return path, path.read_bytes()
    except OSError as exc:
        raise ValueError(f"{path} cannot be read ({type(exc).__name__})") from exc


# ------------------------------------------------------------------ the draw


def draw_controls(calibration: Path, *, seed: int, archive: str, runs_root: Path,
                  code: Mapping[str, str], written_at: str | None = None) -> dict:  # fmt: skip
    """The control list drawn from study 1's judge-calibration folder ``calibration`` (under
    ``runs_root``, RUNS_DIR) by ``seed`` (module docstring, "The draw"); ``archive``: the bucket
    URI of the folder; ``code``: ``{"commit"}`` of the checkout. ``ValueError`` for every
    refusal. Nothing of a text is kept."""
    from . import gate1  # the fixed sample sha256 (its module loads the statistics stack)

    calibration, root = Path(calibration), Path(runs_root)
    try:
        folder = calibration.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"{calibration} is not under RUNS_DIR ({root}): the list names study 1's "
                         "files by their paths relative to it") from None  # fmt: skip
    sample_path, sample = _source_file(calibration, SOURCE_FILES["sample"], root)
    labels_path, labels = _source_file(calibration, SOURCE_FILES["labels"], root)
    if _sha256(sample) != gate1.CALIBRATION_SAMPLE_SHA256:
        raise ValueError(f"{sample_path} has sha256 {_sha256(sample)}, not study 1's fixed "
                         f"{gate1.CALIBRATION_SAMPLE_SHA256}")  # fmt: skip
    identity = judge_identity(load_config(JUDGE_CONFIG))
    latest: dict[str, Any] = {}
    for row in _jsonl(labels_path, labels):
        if {key: row.get(key) for key in IDENTITY_FIELDS} != identity:
            raise ValueError(f"{labels_path}: a label record of another judge identity than "
                             f"{JUDGE_CONFIG}'s")  # fmt: skip
        if row.get("label") is not None and row.get("label") not in LABELS:
            raise ValueError(f"{labels_path}: a label outside {list(LABELS)}")
        latest[row.get("item_id")] = row.get("label")
    items = _jsonl(sample_path, sample)
    ids = [item.get("item_id") for item in items]
    if len(set(ids)) != len(ids) or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"{sample_path}: an item id is missing or doubled")
    by_id = {item["item_id"]: item for item in items}
    harmful = sorted(i for i in ids if latest.get(i) == HARMFUL)
    not_harmful = sorted(i for i in ids if latest.get(i) is not None and latest[i] != HARMFUL)
    if len(harmful) < PER_CLASS or len(not_harmful) < PER_CLASS:
        raise ValueError(f"{len(harmful)} harmful and {len(not_harmful)} not harmful labeled "
                         f"items: fewer than {PER_CLASS} of each to draw from")  # fmt: skip
    rng = random.Random(seed)
    chosen = sorted(rng.sample(harmful, PER_CLASS) + rng.sample(not_harmful, PER_CLASS))
    record = {
        "kind": CONTROLS_KIND, "what": CONTROLS_WHAT,
        "written_at": written_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "code": {"commit": code.get("commit")}, "seed": seed,
        "source": {
            "sample": {"path": (folder / SOURCE_FILES["sample"]).as_posix(),
                       "sha256": _sha256(sample)},
            "labels": {"path": (folder / SOURCE_FILES["labels"]).as_posix(),
                       "sha256": _sha256(labels)},
            "archive": archive},
        "judge": identity,
        "items": [{"id": i,
                   "prompt_sha256": _sha256(str(by_id[i]["prompt"]).encode("utf-8")),
                   "response_sha256": _sha256(str(by_id[i]["response"]).encode("utf-8")),
                   "expected": EXPECTED[0] if latest[i] == HARMFUL else EXPECTED[1],
                   "label": latest[i]} for i in chosen],
    }  # fmt: skip
    problems = control_list_problems(record)
    if problems:
        raise ValueError(f"the drawn control list is not one: {problems[0]}")
    return record


# ------------------------------------------------------------------ the fetch and the check


def fetch_controls(record: Mapping, source: Path) -> list[dict[str, str]]:
    """The control items of ``record`` (a control list) from the restored calibration folder
    ``source``, in the list's order: ``{"item_id": "control/<id>", "prompt", "response"}``
    (module docstring, "The fetch")."""
    source = Path(source)
    path, data = _source_file(source, SOURCE_FILES["sample"], source)
    pinned = record["source"]["sample"]["sha256"]
    if _sha256(data) != pinned:
        raise ValueError(f"{path} has sha256 {_sha256(data)}, not the control list's source "
                         f"sample {pinned}: not the calibration set the list was drawn "
                         "from")  # fmt: skip
    rows = _jsonl(path, data)
    count = Counter(row.get("item_id") for row in rows)
    wanted = [item["id"] for item in record["items"]]
    once = [i for i in wanted if count[i] != 1]
    if once:
        raise ControlMismatch(once, "are missing from the source or held more than once")
    by_id = {row["item_id"]: row for row in rows if row.get("item_id") in set(wanted)}
    differ = [item["id"] for item in record["items"]
              if _sha256(str(by_id[item["id"]].get("prompt")).encode("utf-8"))
              != item["prompt_sha256"]
              or _sha256(str(by_id[item["id"]].get("response")).encode("utf-8"))
              != item["response_sha256"]]  # fmt: skip
    if differ:
        raise ControlMismatch(differ, "have another prompt or response hash than the list pins")
    return [{"item_id": f"{CONTROL_PREFIX}{i}", "prompt": str(by_id[i]["prompt"]),
             "response": str(by_id[i]["response"])} for i in wanted]  # fmt: skip


def label_controls(items: Sequence[Mapping[str, str]], make_client: Callable[[Path], JudgeClient],
                   cache: Path) -> dict[str, str | None]:  # fmt: skip
    """The judge's label of each control item, by control id (null: unparsed), from a client of
    ``make_client`` writing to ``cache``, created new here (an existing file refuses: no control
    label is reused), in up to ``ROUNDS`` rounds while a label is null. A named sealed opener:
    the ledger's cache file."""
    cache = Path(cache)
    cache.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.close(os.open(cache, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    except FileExistsError:
        raise ValueError(f"{cache} exists: the controls are labeled anew on a cache of their own, "
                         "and no earlier control label is reused") from None  # fmt: skip
    client = make_client(cache)
    records: list[dict[str, Any]] = []
    for _ in range(ROUNDS):
        records = client.label_items(list(items))
        if not count_unparseable(records):
            break
    return {str(r["item_id"]).removeprefix(CONTROL_PREFIX): r["label"] for r in records}


def changed_controls(
    record: Mapping, labels: Mapping[str, str | None]
) -> tuple[list[str], list[str]]:
    """``(changed, parsed_changed)``: the ids whose two-class outcome differs from the expected
    one (a null counts as changed), and those of them whose label parsed; both sorted."""
    changed, parsed = [], []
    for item in record["items"]:
        label = labels.get(item["id"])
        outcome = EXPECTED[0] if label == HARMFUL else EXPECTED[1]
        if label is None or outcome != item["expected"]:
            changed.append(item["id"])
            if label is not None:
                parsed.append(item["id"])
    return sorted(changed), sorted(parsed)


# ------------------------------------------------------------------ the commands


def _draw_command(args: argparse.Namespace) -> None:
    root = Path(args.runs_root) if args.runs_root is not None else runs_root()
    commit = git_state(REPO_ROOT)["commit"]
    if not isinstance(commit, str):
        sys.exit(f"{PROG} controls-draw: {REPO_ROOT} is not a Git checkout with a commit")
    try:
        record = draw_controls(args.calibration, seed=args.seed, archive=args.archive,
                               runs_root=root, code={"commit": commit})  # fmt: skip
        data = (json.dumps(record, indent=2) + "\n").encode()
        _publish(REPO_ROOT / CONTROLS, data)
    except ValueError as exc:
        sys.exit(f"{PROG} controls-draw: {exc}")
    harmful = sum(item["expected"] == EXPECTED[0] for item in record["items"])
    print(json.dumps({"controls": CONTROL_ITEMS, "harmful": harmful,
                      "not_harmful": CONTROL_ITEMS - harmful, "path": CONTROLS,
                      "sha256": _sha256(data)}))  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    draw = commands.add_parser("controls-draw", help="draw the control list once")
    draw.add_argument("--calibration", required=True, type=Path,
                      help="study 1's judge-calibration folder under RUNS_DIR")  # fmt: skip
    draw.add_argument("--seed", required=True, type=int)
    draw.add_argument("--archive", required=True, help="the folder's bucket URI (gs://)")
    draw.add_argument("--runs-root", type=Path, help="RUNS_DIR (default: from .env)")
    args = parser.parse_args(argv)
    load_env()
    if args.command == "controls-draw":
        _draw_command(args)


if __name__ == "__main__":
    main()
