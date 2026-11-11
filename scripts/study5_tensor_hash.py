"""The endpoint tensors of runs, hashed (docs/study5-preregistration.md, "Extension 2", M1),
written as one JSON record for the engineering check's tensor comparison.

    python scripts/study5_tensor_hash.py RUN_DIR [RUN_DIR ...] --out RECORD.json

For each run directory: its recorded endpoint checkpoint (``inputs.endpoint_checkpoint`` in
``meta.json``: path and sha256), the file's sha256 against that record, and, when they match, the
checkpoint's tensors hash (``tensor_hash.tensors_sha256`` over the saved tensors in their order,
the hash a study-5 surrogate endpoint record holds as ``endpoint_trainable_sha256``). The
checkpoint is memory-mapped on the host; nothing is moved to a device. The record maps each run
directory's name to ``{"file_ok", "tensors_sha256"}`` (null tensors when the file does not match
or there is no checkpoint). stdout carries names and hashes only. An existing record is never
overwritten. Exits 0 when written, 1 on an error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from zero_field_refusal.tensor_hash import file_sha256, tensors_sha256


def endpoint_hash(run: Path) -> dict:
    inputs = json.loads((run / "meta.json").read_text(encoding="utf-8")).get("inputs") or {}
    record = inputs.get("endpoint_checkpoint")
    if not isinstance(record, dict):
        return {"file_ok": None, "tensors_sha256": None, "reason": "no endpoint checkpoint"}
    path = Path(record["path"])
    if not path.is_file():
        return {"file_ok": False, "tensors_sha256": None, "reason": "the file is missing"}
    if file_sha256(path) != record.get("sha256"):
        return {"file_ok": False, "tensors_sha256": None,
                "reason": "the file does not hash to its record"}  # fmt: skip
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    return {"file_ok": True, "tensors_sha256": tensors_sha256(list(state.items()))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"record exists: {args.out}")
        return 1
    record = {}
    for run in args.runs:
        if not (run / "meta.json").is_file():
            print(f"{run.name}: no meta.json")
            return 1
        record[run.name] = endpoint_hash(run)
        print(f"{run.name} file_ok {record[run.name]['file_ok']} "
              f"tensors {record[run.name]['tensors_sha256']}")  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(f".{args.out.name}.partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(partial, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
