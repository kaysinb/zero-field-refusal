"""Study 5's engineering check (docs/study5-preregistration.md, "Protocol"): study 3's F runs on
seed 10403, rerun with the study-5 code, against study 3's archived records of the same runs,
written as one JSON record.

    python scripts/study5_engineering_check.py --archived DIR [DIR ...] --rerun DIR [DIR ...]
        --out RECORD.json

Each DIR is a run directory. The runs pair by the config hash their ``meta.json`` records: every
archived run needs a rerun of its config, and the reverse. Within a pair every record file is
compared: every regular file below the run directory except the folders ``wandb/`` (W&B's own
files), ``judge/`` and ``judge-superseded/`` (the labeling's outputs, written after the run;
never opened). A file present on one side only differs. A file is compared after the volatile
fields are removed exactly as the golden single-process fixture removes them
(``analysis.record_normalize.normalize``): a ``.jsonl`` file line by line, each line's JSON
normalized; a ``.json`` file as one normalized JSON value; ``config.yaml`` as its normalized
YAML mapping; any other file byte for byte. ``meta.json`` is compared without ``launch`` and
``campaign`` besides the volatile fields: they name the machine and the campaign file that
started the run (the Pod, its image and data center; the campaign's name, label and header), not
what the run computed. The endpoint state is compared where a run records one: its checkpoint's
sha256 is in ``summary.json`` and ``meta.json`` (``endpoint_checkpoint``; the path is
volatile), so an archived checkpoint record must be matched by the rerun's.

The record holds, per pair, the config hash, both run ids and, per file, its name, whether it is
equal and, for a differing file, its line counts and the number of lines that differ; for a
differing record file outside ``responses/`` and ``observations/``, also the names of the
top-level keys that differ (names only, never values). A file under ``responses/`` or
``observations/`` is compared in memory and reported by its name and those counts alone: no text
of it is stored or printed. stdout carries file names, counts and hashes only.

Exits 0 when every pair is equal, 2 when a file differs, a file is on one side only or a run has
no partner, 1 on an error (a missing run directory or ``meta.json``, an unreadable record, an
existing --out), which writes no record. An existing record is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from zero_field_refusal.analysis.record_normalize import normalize
from zero_field_refusal.tracking import git_state

SKIPPED = ("wandb", "judge", "judge-superseded")  # top-level folders never compared or opened
SEALED = ("responses", "observations")  # compared in memory; reported by name and counts only
PLACEMENT = ("launch", "campaign")  # meta.json: where and from which file the run was started
EQUAL, DIFFERENT, ERROR = 0, 2, 1


class CheckError(ValueError):
    """A run or a record that cannot be read: the check has no answer."""


def record_files(run: Path) -> set[str]:
    """Every regular file below ``run`` but those under ``SKIPPED``, as POSIX relative paths."""
    out = set()
    for dirpath, dirnames, filenames in os.walk(run):
        here = Path(dirpath)
        if here == run:
            dirnames[:] = [d for d in dirnames if d not in SKIPPED]
        for name in filenames:
            path = here / name
            if path.is_file() and not path.is_symlink():
                out.add(path.relative_to(run).as_posix())
    return out


def _json(text: str, where: str) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise CheckError(f"{where}: not JSON ({type(exc).__name__})") from None


def comparable(run: Path, name: str) -> list[Any]:
    """The file's content as a list of normalized values (one per line of a ``.jsonl``)."""
    path = run / name
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CheckError(f"{name}: unreadable ({type(exc).__name__})") from None
    if name.endswith(".jsonl"):
        lines = data.decode("utf-8").splitlines()
        return [normalize(_json(line, f"{name} line {i}")) for i, line in enumerate(lines, 1)]
    if name.endswith(".json"):
        value = normalize(_json(data.decode("utf-8"), name))
        if name == "meta.json" and isinstance(value, dict):
            value = {k: v for k, v in value.items() if k not in PLACEMENT}
        return [value]
    if name == "config.yaml":
        try:
            return [normalize(yaml.safe_load(data.decode("utf-8")))]
        except yaml.YAMLError as exc:
            raise CheckError(f"{name}: not YAML ({type(exc).__name__})") from None
    return [hashlib.sha256(data).hexdigest()]


def compare_file(archived: Path, rerun: Path, name: str) -> dict[str, Any]:
    left, right = comparable(archived, name), comparable(rerun, name)
    if left == right:
        return {"name": name, "equal": True}
    differing = sum(a != b for a, b in zip(left, right, strict=False)) + abs(len(left) - len(right))
    entry: dict[str, Any] = {"name": name, "equal": False, "lines": [len(left), len(right)],
                             "differing_lines": differing}  # fmt: skip
    if name.split("/", 1)[0] not in SEALED:
        keys = set()
        for a, b in zip(left, right, strict=False):
            if a != b and isinstance(a, dict) and isinstance(b, dict):
                keys |= {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
        entry["keys"] = sorted(keys)
    return entry


def compare_runs(archived: Path, rerun: Path) -> list[dict[str, Any]]:
    left, right = record_files(archived), record_files(rerun)
    files = [compare_file(archived, rerun, name) for name in sorted(left & right)]
    files += [{"name": n, "equal": False, "only_in": "archived"} for n in sorted(left - right)]
    files += [{"name": n, "equal": False, "only_in": "rerun"} for n in sorted(right - left)]
    return files


def run_hash(run: Path) -> str:
    if not run.is_dir():
        raise CheckError(f"{run}: no run directory")
    meta = run / "meta.json"
    if not meta.is_file():
        raise CheckError(f"{run}: no meta.json")
    value = _json(meta.read_text(encoding="utf-8"), f"{run.name}/meta.json")
    found = value.get("config_hash") if isinstance(value, dict) else None
    if not isinstance(found, str) or not found:
        raise CheckError(f"{run.name}/meta.json: no config_hash")
    return found


def by_hash(runs: list[Path], side: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for run in runs:
        found = run_hash(run)
        if found in out:
            raise CheckError(f"--{side}: two runs of config hash {found}")
        out[found] = run
    return out


def check(archived: list[Path], rerun: list[Path]) -> dict[str, Any]:
    left, right = by_hash(archived, "archived"), by_hash(rerun, "rerun")
    pairs = []
    for found in sorted(left.keys() & right.keys()):
        files = compare_runs(left[found], right[found])
        pairs.append({"config_hash": found, "archived": left[found].name,
                      "rerun": right[found].name, "equal": all(f["equal"] for f in files),
                      "files": files})  # fmt: skip
    unpaired = {"archived": sorted(left.keys() - right.keys()),
                "rerun": sorted(right.keys() - left.keys())}  # fmt: skip
    equal = all(p["equal"] for p in pairs) and not unpaired["archived"] and not unpaired["rerun"]
    return {
        "kind": "study5-engineering-check",
        "equal": equal,
        "pairs": pairs,
        "unpaired": unpaired,
        "rule": "analysis.record_normalize.normalize; meta.json without " + ", ".join(PLACEMENT),
        "skipped_folders": list(SKIPPED),
        "git": git_state(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_new(path: Path, record: dict[str, Any]) -> None:
    """Write ``record`` to ``path`` atomically; refuse an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; a record is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(partial, path)


def report(record: dict[str, Any]) -> None:
    """Names, counts and hashes only."""
    for pair in record["pairs"]:
        differing = [f for f in pair["files"] if not f["equal"]]
        print(f"pair {pair['config_hash']} archived {pair['archived']} rerun {pair['rerun']} "
              f"files {len(pair['files'])} differing {len(differing)}")  # fmt: skip
        for entry in differing:
            where = entry.get("only_in")
            detail = (f"only in {where}" if where else
                      f"lines {entry['lines']} differing {entry['differing_lines']}")  # fmt: skip
            print(f"  differs {entry['name']}: {detail}")
    for side, hashes in record["unpaired"].items():
        for found in hashes:
            print(f"unpaired {side} {found}")
    print("equal" if record["equal"] else "different")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archived", type=Path, nargs="+", required=True)
    parser.add_argument("--rerun", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"record exists: {args.out}")
        return ERROR
    try:
        record = check(args.archived, args.rerun)
    except (CheckError, UnicodeDecodeError) as exc:
        print(f"error: {exc}")
        return ERROR
    write_new(args.out, record)
    report(record)
    return EQUAL if record["equal"] else DIFFERENT


if __name__ == "__main__":
    sys.exit(main())
