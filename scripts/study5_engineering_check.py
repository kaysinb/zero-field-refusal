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

Extension 2 (pod 4) reruns runs that save their checkpoints elsewhere and may save step
checkpoints. Three storage details are canonicalized before the comparison, nothing else:
``tracking.step_checkpoints`` absent or null is the same; the endpoint directory
(``tracking.endpoint_checkpoint``) is compared as set or not set; and a checkpoint record's file
``sha256`` (inside ``endpoint_checkpoint`` and ``step_checkpoints``) is not compared, because
``torch.save`` writes the file's own name into its archive. The endpoint state is instead compared
by its tensors: with ``--endpoint-tensors CONFIG_HASH=SHA256`` (the archived run's endpoint
tensors hash, ``tensor_hash.tensors_sha256``, as its surrogate endpoint record holds it) and
``--rerun-endpoint-tensors RECORD`` (``scripts/study5_tensor_hash.py``'s record of the reruns:
each checkpoint's file against its recorded sha256 and its tensors hash), the rerun's file must
match its record and its tensors SHA256. A pair whose archived run recorded an endpoint checkpoint
and has no such value, or whose rerun has none hashed, cannot be checked (an error). This script
loads no checkpoint itself.

The record holds the checkout's commit and GPU code identity (``code``) and, per pair, the config
hash, both run ids, the rerun's recorded commit and cleanness (``rerun_commit``,
``rerun_dirty``) and, per file, its name, whether it is
equal and, for a differing file, its line counts and the number of lines that differ; for a
differing record file outside ``responses/`` and ``observations/``, also the names of the
top-level keys that differ (names only, never values). A file under ``responses/`` or
``observations/`` is compared in memory and reported by its name and those counts alone: no text
of it is stored or printed. stdout carries file names, counts and hashes only.

**Numerical equivalence** (``--tolerance REL``; docs/study5-preregistration.md, "Extension 2
amendment"). The canonicalization above stays; every comparison respects the JSON type
(``true`` is not ``1``, ``1`` is not ``1.0``). A differing file is compared by type, line by line:
the structure, the keys and every leaf that is not a finite float are exact (identities and
hashes, steps and counts, booleans, termination states). A finite float x of the rerun passes
against the archived y when |x − y| ≤ 10⁻⁹ + REL·|y|, wherever it stands (summaries and
``observations/`` too). ``config.yaml`` and ``meta.json`` (the config and the run's identity),
and a differing file that is not JSON, stay exact. The greedy answers (``responses/``) are
compared answer by answer: every answer complete on both sides (its identity, ``text``,
``response_ids`` and ``ended``, each of its type) with the same keys, the same items in the same
order, only those fields different, and at most two answers so per file. An incomplete answer,
or one that differs elsewhere, fails the file without counting against the two. Each different
answer is recorded by its item id, the sha256 of both texts and its first differing token, never
by its text. Endpoint tensors that differ pass only through ``--endpoint-distance
CONFIG_HASH=RECORD``, ``scripts/study5_pairs.py``'s record of the archived run's checkpoint
(fetched back from its archive, by the sha256 its run recorded) against the rerun's endpoint:
‖θ_rerun − θ_archived‖ / ‖θ_archived − θ₀‖ ≤ END, the endpoint bound: ``--endpoint-tolerance
END`` (extension 2's second amendment: a bound of its own, beside the measurements' REL), REL
when it is not given. For a differing open line file whose lines
record a step, the largest relative drift of each differing line is kept (``by_step``). Several
reruns of one config (``--rerun``) are each compared, for a run without an endpoint checkpoint.

The record states ``exact`` (every file equal after the canonicalization, every endpoint's tensors
equal) and, under the tolerance, ``within_tolerance`` apart, per file, per pair and in all, with
the number of different answers; ``passed`` is ``exact`` without a tolerance and ``exact`` or
``within_tolerance`` with one. Its ``tolerance`` names REL (``relative``), the absolute part, the
answers allowed and END (``endpoint_distance``).

**Acceptance for endpoint analysis** (``--endpoint-analysis``, with ``--tolerance`` and
``--endpoint-tolerance``; docs/study5-preregistration.md, "Extension 2 amendment 3"), for the
replays, whose endpoint distance is checked. Everything above is computed and recorded as before
(``exact``, ``within_tolerance``, ``passed``, the drift by file and by step); beside it, per pair
and in all, ``accepted_for_endpoint_analysis``. A pair is accepted when it has an endpoint
checkpoint whose tensors are equal or within the endpoint bound by their distance (the rerun's
file matching its record, the distance record present); no file is on one side only;
``config.yaml``, ``meta.json`` and every file that is not JSON are equal; in every other differing
file the line count, the structure, the keys and every leaf that is not a finite float are exact,
a finite float of the archived run finite in the rerun, whatever its drift; the greedy answers
complete (equal or not) with the same items in the same order, however many differ; the pair
holds the greedy answers, the factual answers and the benchmark letters (``REQUIRED_OUTPUTS``).
Generated outputs may differ: the greedy answers (``ANSWER_PAYLOAD``), and in ``observations/``
alone the factual answers' text and the benchmark's predicted letter with their correctness flag
``value`` (``OUTPUTS``), each of its type on both sides (equal or not); the rows' identities
(condition, measure, item id) stay exact.
Per file and per pair, ``outputs`` records for each measure and condition the number of items, how
many outputs differ, the correctness flips and both accuracies (counts and means only, never a
text). The exit code then follows ``accepted_for_endpoint_analysis``.

Exits 0 when the check passes, 2 when it does not (a file differs beyond the rule, a file is on
one side only, a run has no partner), 1 on an error (a missing run directory or ``meta.json``, an
unreadable record, an endpoint distance record of other checkpoints, an existing --out), which
writes no record. An existing record is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from zero_field_refusal import code_identity
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


IGNORED: frozenset[str] = frozenset()  # --ignore: keys dropped at every depth besides normalize's


def _drop(value: Any) -> Any:
    """``value`` without the keys in ``IGNORED``, at every depth."""
    if not IGNORED:
        return value
    if isinstance(value, dict):
        return {k: _drop(v) for k, v in value.items() if k not in IGNORED}
    if isinstance(value, list):
        return [_drop(v) for v in value]
    return value


CHECKPOINT_RECORDS = ("endpoint_checkpoint", "step_checkpoints")


def _canonical(value: Any) -> Any:
    """Storage details canonicalized at every depth (module docstring): a null
    ``step_checkpoints`` dropped, a checkpoint record's file ``sha256`` dropped."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == "step_checkpoints" and item is None:
                continue
            if key in CHECKPOINT_RECORDS:
                records = item if isinstance(item, list) else [item]
                cleaned = [{k: v for k, v in r.items() if k != "sha256"} if isinstance(r, dict)
                           else r for r in records]  # fmt: skip
                out[key] = cleaned if isinstance(item, list) else cleaned[0]
                continue
            out[key] = _canonical(item)
        return out
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return value


def _canonical_config(value: Any) -> Any:
    """config.yaml: the endpoint directory compared as set or not set; a null step list dropped."""
    if isinstance(value, dict) and isinstance(value.get("tracking"), dict):
        tracking = dict(value["tracking"])
        if tracking.get("endpoint_checkpoint") is not None:
            tracking["endpoint_checkpoint"] = "<set>"
        if "step_checkpoints" in tracking and tracking["step_checkpoints"] is None:
            del tracking["step_checkpoints"]
        value = {**value, "tracking": tracking}
    return value


def comparable(run: Path, name: str) -> list[Any]:
    """The file's content as a list of normalized values (one per line of a ``.jsonl``)."""
    values = _comparable(run, name)
    if name == "config.yaml":
        values = [_canonical_config(v) for v in values]
    elif name.endswith((".json", ".jsonl")):
        values = [_canonical(v) for v in values]
    return [_drop(value) for value in values]


def _comparable(run: Path, name: str) -> list[Any]:
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


TOLERANCE: float | None = None  # --tolerance: extension 2's numerical equivalence (relative)
# --endpoint-tolerance: the endpoint distance's bound under the tolerance (extension 2's second
# amendment), TOLERANCE when not given
ENDPOINT_TOLERANCE: float | None = None
ABSOLUTE = 1e-9  # its absolute part: |x − y| ≤ ABSOLUTE + TOLERANCE·|y|, y the archived value
GREEDY = "responses"  # the greedy answers, compared answer by answer under the tolerance
ANSWERS_ALLOWED = 2  # differing greedy answers a file may hold under the tolerance
ANSWER_KEYS = ("condition", "item_id")  # an answer's identity, always exact
# --endpoint-analysis: extension 2's third amendment, the replays accepted for endpoint analysis
ENDPOINT_ANALYSIS = False
# observations/: the generated outputs and their correctness flag, by measure, which may differ
# under --endpoint-analysis (each of its type on both sides); a row's other keys stay exact
OUTPUTS = {"factual_qa": {"value": float, "text": str},
           "benchmark": {"value": float, "predicted": int}}  # fmt: skip
HARMFUL = "harmful_answers"  # the greedy answers' measure in ``outputs``
OBSERVED = "observations"  # the folder whose rows may carry ``OUTPUTS``
REQUIRED_OUTPUTS = (HARMFUL, *OUTPUTS)  # the answers a replay accepted for analysis must hold


def _leaves(a: Any, b: Any) -> tuple[bool, float, float]:
    """``a`` archived, ``b`` rerun: (the structure and every leaf but the finite floats equal,
    the largest |b − a| / (ABSOLUTE + TOLERANCE·|a|) over the finite float leaves, the largest
    |b − a| / |a| over those with a ≠ 0). A float is a JSON number written with a fraction or an
    exponent; integers (steps, counts), booleans, strings (identities, hashes, reasons) and null
    are exact."""
    if isinstance(a, float) and isinstance(b, float) and math.isfinite(a) and math.isfinite(b):
        gap = abs(b - a)
        ratio = gap / (ABSOLUTE + (TOLERANCE or 0.0) * abs(a))
        relative = gap / abs(a) if a else 0.0  # at y = 0 only the ratio (the absolute part)
        return True, ratio, relative
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False, 0.0, 0.0
        parts = [_leaves(a[k], b[k]) for k in a]
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False, 0.0, 0.0
        parts = [_leaves(x, y) for x, y in zip(a, b, strict=True)]
    else:
        same = type(a) is type(b) and (a == b or (isinstance(a, float) and a != a and b != b))
        return same, 0.0, 0.0
    return (all(p[0] for p in parts), max((p[1] for p in parts), default=0.0),
            max((p[2] for p in parts), default=0.0))  # fmt: skip


def _numerical(left: list[Any], right: list[Any], sealed: bool) -> dict[str, Any]:
    """A differing file under the tolerance: line by line by ``_leaves``; for an open file also
    the largest relative drift of each differing line that records a step (scalars only)."""
    if len(left) != len(right):
        return {"within_tolerance": False, "max_ratio": None, "max_relative": None}
    found = [_leaves(a, b) for a, b in zip(left, right, strict=True)]
    ratio = max((f[1] for f in found), default=0.0)
    entry: dict[str, Any] = {
        "within_tolerance": all(f[0] for f in found) and ratio <= 1.0,
        "non_float_differences": sum(not f[0] for f in found),
        "max_ratio": ratio,
        "max_relative": max((f[2] for f in found), default=0.0),
    }
    if not sealed:
        by_step = [
            [a["step"], f[2]]
            for a, b, f in zip(left, right, found, strict=True)
            if not same(a, b) and isinstance(a, dict) and isinstance(a.get("step"), int)
        ]
        if by_step:
            entry["by_step"] = by_step
    return entry


def _answer_sha(line: Any) -> str:
    text = line.get("text") if isinstance(line, dict) else None
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _first_token(a: Any, b: Any) -> int | None:
    """The position of the first differing generated token, from ``response_ids``."""
    x = a.get("response_ids") if isinstance(a, dict) else None
    y = b.get("response_ids") if isinstance(b, dict) else None
    if not isinstance(x, list) or not isinstance(y, list):
        return None
    for i, (p, q) in enumerate(zip(x, y, strict=False)):
        if p != q:
            return i
    return min(len(x), len(y)) if len(x) != len(y) else None


ANSWER_PAYLOAD = {"text": str, "response_ids": list, "ended": bool}  # what may differ, typed
ANSWER_DERIVED = ("keyword_refusal",)  # read from the text where it is recorded


def same(a: Any, b: Any) -> bool:
    """Equality that respects the JSON type: ``true`` is not ``1`` and ``1`` is not ``1.0``
    (Python's own ``==`` treats them as equal)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return a == b or (isinstance(a, float) and a != a and b != b)


def _of_type(x: Any, kind: type) -> bool:
    """``x`` of the JSON type ``kind``; a boolean is never an integer."""
    return isinstance(x, bool) if kind is bool else isinstance(x, kind) and not isinstance(x, bool)


def _answer_shape(line: Any) -> bool:
    """A complete answer: its identity and every payload field, each of its type."""
    return (
        isinstance(line, dict)
        and all(k in line for k in ANSWER_KEYS)
        and all(k in line and _of_type(line[k], t) for k, t in ANSWER_PAYLOAD.items())
        and all(_of_type(x, int) for x in line["response_ids"])
    )


def _answers(left: list[Any], right: list[Any]) -> dict[str, Any]:
    """A differing greedy-answer file under the tolerance: every answer complete on both sides
    (``_answer_shape``) with the same keys, the same answers in the same order (``ANSWER_KEYS``
    exact), only the payload (``ANSWER_PAYLOAD``, and what is derived from it) different, and at
    most ``ANSWERS_ALLOWED`` answers so; each different one recorded by its item id, both texts'
    sha256 and its first differing token (no text is kept). An answer that is not complete, or
    differs elsewhere, fails the file without counting against the allowance."""
    if len(left) != len(right):
        return {"within_tolerance": False, "answers_complete": False}
    may_differ = {*ANSWER_PAYLOAD, *ANSWER_DERIVED}
    for a, b in zip(left, right, strict=True):
        if not (_answer_shape(a) and _answer_shape(b)) or a.keys() != b.keys():
            return {"within_tolerance": False, "answers_complete": False}
        if not all(same(a[k], b[k]) for k in a if k not in may_differ):
            return {"within_tolerance": False, "answers_complete": True,
                    "answers_same_items": False}  # fmt: skip
    changed = [{"item_id": a.get("item_id"), "archived_sha256": _answer_sha(a),
                "rerun_sha256": _answer_sha(b), "first_token": _first_token(a, b)}
               for a, b in zip(left, right, strict=True) if not same(a, b)]  # fmt: skip
    return {"within_tolerance": len(changed) <= ANSWERS_ALLOWED, "answers_complete": True,
            "answers_same_items": True, "answers_differing": len(changed),
            "answers": changed}  # fmt: skip


EXACT_FILES = ("config.yaml", "meta.json")  # the config and the run's identity: never a tolerance


def _without_outputs(line: Any) -> Any:
    """An ``observations/`` row without its generated output and correctness (``OUTPUTS``)."""
    fields = OUTPUTS.get(line.get("measure")) if isinstance(line, dict) else None
    return line if fields is None else {k: v for k, v in line.items() if k not in fields}


def _outputs_shaped(a: Any, b: Any) -> bool:
    """Both rows' outputs present, each of its type, a correctness flag finite on both sides."""
    fields = OUTPUTS.get(a.get("measure")) if isinstance(a, dict) else None
    if fields is None:
        return True
    return isinstance(b, dict) and all(
        k in a and k in b and _of_type(a[k], t) and _of_type(b[k], t)
        and (t is not float or (math.isfinite(a[k]) and math.isfinite(b[k])))
        for k, t in fields.items()
    )  # fmt: skip


def outputs(left: list[Any], right: list[Any], greedy: bool) -> list[dict[str, Any]]:
    """Per measure and condition: the items, the outputs that differ, and for ``OUTPUTS`` the
    correctness flips and both accuracies (the mean of ``value``). Counts and means only."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for a, b in zip(left, right, strict=True):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            continue
        measure = HARMFUL if greedy else a.get("measure")
        if measure != HARMFUL and measure not in OUTPUTS:
            continue
        entry = found.setdefault((measure, str(a.get("condition"))), {"items": 0, "differing": 0})
        entry["items"] += 1
        if greedy:
            entry["differing"] += not same(a, b)
            continue
        flag = "value"
        entry["differing"] += any(not same(a.get(k), b.get(k)) for k in OUTPUTS[measure]
                                  if k != flag)  # fmt: skip
        entry["flips"] = entry.get("flips", 0) + (not same(a.get(flag), b.get(flag)))
        for side, row in (("archived_sum", a), ("rerun_sum", b)):
            value = row.get(flag)
            entry[side] = entry.get(side, 0.0) + (value if _of_type(value, float) else 0.0)
    out = []
    for (measure, condition), entry in sorted(found.items()):
        row = {"measure": measure, "condition": condition, "items": entry["items"],
               "differing": entry["differing"]}  # fmt: skip
        if "flips" in entry:
            archived = entry["archived_sum"] / entry["items"]
            rerun = entry["rerun_sum"] / entry["items"]
            row |= {"flips": entry["flips"], "archived_accuracy": archived,
                    "rerun_accuracy": rerun, "accuracy_difference": rerun - archived}  # fmt: skip
        out.append(row)
    return out


def accepted_file(entry: dict[str, Any], left: list[Any], right: list[Any]) -> bool:
    """Extension 2's third amendment, one file (module docstring): every greedy answer complete
    and every output row of ``observations/`` of its shape, equal or not; then equal; or, for
    the greedy answers, the same items; or, for any other JSON file but ``EXACT_FILES``, the
    same lines, structure, keys and every leaf but the finite floats exact, the outputs of
    ``observations/`` aside."""
    folder = entry["name"].split("/", 1)[0]
    if folder == GREEDY and not all(_answer_shape(x) for x in (*left, *right)):
        return False
    if folder == OBSERVED and not all(
        _outputs_shaped(a, b) for a, b in zip(left, right, strict=False)
    ):
        return False
    if entry["equal"]:
        return True
    if folder == GREEDY:
        return entry.get("answers_complete") is True and entry.get("answers_same_items") is True
    if entry["name"] in EXACT_FILES or not entry["name"].endswith((".json", ".jsonl")):
        return False
    strip = _without_outputs if folder == OBSERVED else (lambda line: line)
    return len(left) == len(right) and all(
        _leaves(strip(a), strip(b))[0] for a, b in zip(left, right, strict=True)
    )  # fmt: skip


def compare_file(archived: Path, rerun: Path, name: str) -> dict[str, Any]:
    left, right = comparable(archived, name), comparable(rerun, name)
    if ENDPOINT_ANALYSIS:
        entry = _compare_file(left, right, name)
        entry["accepted"] = accepted_file(entry, left, right)
        folder = name.split("/", 1)[0]
        if len(left) == len(right) and folder in (GREEDY, OBSERVED):
            found = outputs(left, right, folder == GREEDY)
            if found:
                entry["outputs"] = found
        return entry
    return _compare_file(left, right, name)


def _compare_file(left: list[Any], right: list[Any], name: str) -> dict[str, Any]:
    if same(left, right):
        return {"name": name, "equal": True}
    differing = sum(not same(a, b) for a, b in zip(left, right, strict=False))
    differing += abs(len(left) - len(right))
    entry: dict[str, Any] = {"name": name, "equal": False, "lines": [len(left), len(right)],
                             "differing_lines": differing}  # fmt: skip
    folder = name.split("/", 1)[0]
    if TOLERANCE is not None:
        if folder == GREEDY:
            entry |= _answers(left, right)
        elif name in EXACT_FILES or not name.endswith((".json", ".jsonl")):
            entry["within_tolerance"] = False  # exact: no tolerance
        else:
            entry |= _numerical(left, right, folder in SEALED)
    if folder not in SEALED:
        keys = set()
        for a, b in zip(left, right, strict=False):
            if not same(a, b) and isinstance(a, dict) and isinstance(b, dict):
                keys |= {k for k in a.keys() | b.keys() if not same(a.get(k), b.get(k))}
        entry["keys"] = sorted(keys)
    return entry


def compare_runs(archived: Path, rerun: Path) -> list[dict[str, Any]]:
    left, right = record_files(archived), record_files(rerun)
    files = [compare_file(archived, rerun, name) for name in sorted(left & right)]
    files += [{"name": n, "equal": False, "only_in": "archived"} for n in sorted(left - right)]
    files += [{"name": n, "equal": False, "only_in": "rerun"} for n in sorted(right - left)]
    if ENDPOINT_ANALYSIS:
        for entry in files:
            entry.setdefault("accepted", False)  # a file on one side only
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


def reruns_by_hash(runs: list[Path]) -> dict[str, list[Path]]:
    """The reruns by config hash: several of one config are each compared with its archived run
    (extension 2 keeps every rerun of the engineering check in its record)."""
    out: dict[str, list[Path]] = {}
    for run in runs:
        found = run_hash(run)
        if run in out.get(found, []):
            raise CheckError(f"--rerun: {run.name} given twice")
        out.setdefault(found, []).append(run)
    return out


def rerun_git(run: Path) -> dict[str, Any]:
    """The commit and cleanness a rerun's ``meta.json`` records (volatile in the comparison,
    kept in the record so that the check binds to the code its reruns ran)."""
    git = json.loads((run / "meta.json").read_text(encoding="utf-8")).get("git") or {}
    return {"rerun_commit": git.get("commit"), "rerun_dirty": git.get("dirty")}


def _endpoint(run: Path) -> dict[str, Any] | None:
    inputs = json.loads((run / "meta.json").read_text(encoding="utf-8")).get("inputs") or {}
    found = inputs.get("endpoint_checkpoint")
    return found if isinstance(found, dict) else None


def endpoint_distance(archived: Path, rerun: Path, record: dict[str, Any] | None) -> dict[str, Any]:
    """Under the tolerance, for endpoint tensors that differ: ``scripts/study5_pairs.py``'s
    record of the pair (the archived run's checkpoint, fetched back from its archive, against the
    rerun's endpoint). Its first item must be the archived checkpoint by the sha256 its run
    recorded and its second the rerun's recorded endpoint; the pair passes when
    ‖θ_rerun − θ_archived‖ / ‖θ_archived − θ₀‖ ≤ the endpoint bound (``ENDPOINT_TOLERANCE``).
    A record of an absent checkpoint (``available: false``) or none fails."""
    if record is None or record.get("available") is False:
        return {"available": False, "within_tolerance": False}
    rows = record.get("pairs") if record.get("kind") == "study5-pairs" else None
    if not isinstance(rows, list) or len(rows) != 1:
        raise CheckError(f"{rerun.name}: the endpoint distance record is not one study5-pairs row")
    row, left, right = rows[0], _endpoint(archived) or {}, _endpoint(rerun) or {}
    a, b = row.get("a") or {}, row.get("b") or {}
    if a.get("kind") != "ckpt" or a.get("sha256") != left.get("sha256"):
        raise CheckError(f"{archived.name}: the distance record's first item is not its "
                         "recorded checkpoint")  # fmt: skip
    if (b.get("kind") != "ckpt" or b.get("run") != rerun.name
            or b.get("sha256") != right.get("sha256")):  # fmt: skip
        raise CheckError(f"{rerun.name}: the distance record's second item is not its endpoint")
    ratio = float(row["distance"]) / float(row["norm_a"])
    return {"available": True, "distance": row["distance"], "archived_norm": row["norm_a"],
            "ratio": ratio, "within_tolerance": ratio <= (ENDPOINT_TOLERANCE or 0.0)}  # fmt: skip


def endpoint_tensors(archived: Path, rerun: Path, expected: str | None,
                     hashed: dict[str, Any] | None,
                     distance: dict[str, Any] | None = None) -> dict[str, Any] | None:  # fmt: skip
    """The endpoint state's comparison by tensors (module docstring); None when neither run
    recorded an endpoint checkpoint. ``hashed`` is ``scripts/study5_tensor_hash.py``'s record of
    the reruns (this script loads no checkpoint); under the tolerance, tensors that differ are
    compared by their distance (``endpoint_distance``)."""
    left, right = _endpoint(archived), _endpoint(rerun)
    if left is None and right is None:
        return None
    if left is not None and expected is None:
        raise CheckError(f"{archived.name}: an endpoint checkpoint and no --endpoint-tensors for "
                         "its config hash")  # fmt: skip
    if right is None:
        return {"expected": expected, "rerun": None, "equal": False, "reason": "no checkpoint",
                "within_tolerance": False}  # fmt: skip
    found = (hashed or {}).get(rerun.name)
    if found is None:
        raise CheckError(f"{rerun.name}: an endpoint checkpoint and no hash of its tensors "
                         "(--rerun-endpoint-tensors)")  # fmt: skip
    if found.get("file_ok") is not True:
        return {"expected": expected, "rerun": None, "equal": False,
                "reason": found.get("reason", "the file does not match its record"),
                "within_tolerance": False}  # fmt: skip
    out: dict[str, Any] = {
        "expected": expected,
        "rerun": found["tensors_sha256"],
        "equal": found["tensors_sha256"] == expected,
    }
    if TOLERANCE is not None and not out["equal"]:
        out["distance"] = endpoint_distance(archived, rerun, distance)
    out["within_tolerance"] = out["equal"] or bool(out.get("distance", {}).get("within_tolerance"))
    return out


def check(archived: list[Path], rerun: list[Path],
          expected_tensors: dict[str, str] | None = None,
          hashed: dict[str, Any] | None = None,
          distances: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:  # fmt: skip
    left, right = by_hash(archived, "archived"), reruns_by_hash(rerun)
    pairs = []
    for found, again in sorted((h, r) for h in left.keys() & right.keys() for r in right[h]):
        if len(right[found]) > 1 and _endpoint(left[found]) is not None:
            raise CheckError(f"--rerun: several reruns of {found}, whose run has an endpoint")
        files = compare_runs(left[found], again)
        tensors = endpoint_tensors(left[found], again, (expected_tensors or {}).get(found),
                                   hashed, (distances or {}).get(found))  # fmt: skip
        exact = all(f["equal"] for f in files) and (tensors is None or tensors["equal"])
        pair = {"config_hash": found, "archived": left[found].name, "rerun": again.name,
                **rerun_git(again), "exact": exact}  # fmt: skip
        if TOLERANCE is not None:
            pair["within_tolerance"] = all(
                f["equal"] or f.get("within_tolerance") for f in files
            ) and (tensors is None or tensors["within_tolerance"])
            pair["answers_differing"] = sum(f.get("answers_differing", 0) for f in files)
        pair["passed"] = exact or bool(pair.get("within_tolerance"))
        if ENDPOINT_ANALYSIS:
            pair["outputs"] = [row for f in files for row in f.get("outputs", [])]
            held = {row["measure"] for row in pair["outputs"] if row["items"]}
            pair["accepted_for_endpoint_analysis"] = (
                tensors is not None and tensors["within_tolerance"]
                and all(f["accepted"] for f in files) and held >= set(REQUIRED_OUTPUTS)
            )  # fmt: skip
        pairs.append({**pair, "files": files, "endpoint_tensors": tensors})
    unpaired = {"archived": sorted(left.keys() - right.keys()),
                "rerun": sorted(right.keys() - left.keys())}  # fmt: skip
    complete = not unpaired["archived"] and not unpaired["rerun"]
    identity = code_identity.gpu_code_identity()
    record: dict[str, Any] = {
        "kind": "study5-engineering-check",
        "exact": complete and all(p["exact"] for p in pairs),
        "passed": complete and all(p["passed"] for p in pairs),
        "pairs": pairs,
        "unpaired": unpaired,
        "rule": "analysis.record_normalize.normalize; meta.json without "
        + ", ".join(PLACEMENT)
        + "; storage canonicalized (module docstring); the endpoint compared by its tensors"
        + ("" if TOLERANCE is None else "; numerical equivalence (module docstring)"),
        "ignored_keys": sorted(IGNORED),
        "tolerance": None,
        "skipped_folders": list(SKIPPED),
        "code": {"commit": identity["commit"], "gpu_code_sha256": identity["sha256"]},
        "git": git_state(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if TOLERANCE is not None:
        record["within_tolerance"] = complete and all(p["within_tolerance"] for p in pairs)
        record["answers_differing"] = sum(p["answers_differing"] for p in pairs)
        record["tolerance"] = {"relative": TOLERANCE, "absolute": ABSOLUTE,
                               "answers_allowed": ANSWERS_ALLOWED,
                               "endpoint_distance": ENDPOINT_TOLERANCE}  # fmt: skip
    if ENDPOINT_ANALYSIS:
        record["accepted_for_endpoint_analysis"] = (
            complete and bool(pairs) and all(p["accepted_for_endpoint_analysis"] for p in pairs)
        )  # fmt: skip
        record["endpoint_analysis"] = ENDPOINT_ANALYSIS_RULE
    return record


ENDPOINT_ANALYSIS_RULE = (
    "docs/study5-preregistration.md, Extension 2 amendment 3: a replay is accepted for endpoint "
    "analysis by its identities, completeness, finiteness, evidence and endpoint distance; the "
    "measurements' drift and the generated outputs that differ are recorded, not gated"
)


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
            if "max_ratio" in entry and entry["max_ratio"] is not None:
                detail += (f" max relative {entry['max_relative']:.3e} ratio "
                           f"{entry['max_ratio']:.3e}")  # fmt: skip
            if "answers_differing" in entry:
                detail += f" answers differing {entry['answers_differing']}"
            if "within_tolerance" in entry:
                detail += f" within tolerance {entry['within_tolerance']}"
            print(f"  differs {entry['name']}: {detail}")
        tensors = pair.get("endpoint_tensors")
        if tensors is not None:
            print(f"  endpoint tensors equal {tensors['equal']} rerun {tensors['rerun']}")
            distance = tensors.get("distance")
            if distance is not None:
                ratio = distance.get("ratio")
                print(f"  endpoint distance available {distance['available']} ratio "
                      f"{'-' if ratio is None else f'{ratio:.3e}'}")  # fmt: skip
        if "within_tolerance" in pair:
            print(f"  exact {pair['exact']} within tolerance {pair['within_tolerance']} "
                  f"answers differing {pair['answers_differing']}")  # fmt: skip
        if "accepted_for_endpoint_analysis" in pair:
            for row in pair["outputs"]:
                print(f"  outputs {row['measure']} {row['condition']}: differing "
                      f"{row['differing']} of {row['items']}"
                      + (f" flips {row['flips']}" if "flips" in row else ""))  # fmt: skip
            print(f"  accepted for endpoint analysis {pair['accepted_for_endpoint_analysis']}")
    for side, hashes in record["unpaired"].items():
        for found in hashes:
            print(f"unpaired {side} {found}")
    print(f"exact {record['exact']} passed {record['passed']}"
          + ("" if record["tolerance"] is None else
             f" within tolerance {record['within_tolerance']}")
          + ("" if "accepted_for_endpoint_analysis" not in record else " accepted for endpoint "
             f"analysis {record['accepted_for_endpoint_analysis']}"))  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archived", type=Path, nargs="+", required=True)
    parser.add_argument("--rerun", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    # extension 2: a key a rerun records beside the training (its step checkpoints' list), not
    # compared; the record names it
    parser.add_argument("--ignore", action="append", default=[], choices=("step_checkpoints",))
    parser.add_argument("--endpoint-tensors", action="append", default=[],
                        metavar="CONFIG_HASH=SHA256")  # fmt: skip
    parser.add_argument("--rerun-endpoint-tensors", type=Path)  # study5_tensor_hash.py's record
    parser.add_argument("--tolerance", type=float)  # extension 2: numerical equivalence
    # extension 2's second amendment: the endpoint distance's bound, with --tolerance only
    parser.add_argument("--endpoint-tolerance", type=float)
    parser.add_argument(
        "--endpoint-distance", action="append", default=[], metavar="CONFIG_HASH=RECORD"
    )  # study5_pairs.py's record, or its absence
    # extension 2's third amendment: the replays accepted for endpoint analysis
    parser.add_argument("--endpoint-analysis", action="store_true")
    args = parser.parse_args(argv)
    expected = {}
    for item in args.endpoint_tensors:
        key, _, value = item.partition("=")
        if not key or len(value) != 64:
            print(f"--endpoint-tensors {item!r}: CONFIG_HASH=SHA256")
            return ERROR
        expected[key] = value
    distances: dict[str, dict[str, Any]] = {}
    for item in args.endpoint_distance:
        key, _, value = item.partition("=")
        if not key or not value:
            print(f"--endpoint-distance {item!r}: CONFIG_HASH=RECORD")
            return ERROR
        path = Path(value)
        try:
            distances[key] = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.is_file()
                else {"available": False}
            )
        except ValueError:
            print(f"--endpoint-distance {item!r}: not JSON")
            return ERROR
    global IGNORED, TOLERANCE, ENDPOINT_TOLERANCE, ENDPOINT_ANALYSIS
    if args.endpoint_analysis and args.endpoint_tolerance is None:
        print("--endpoint-analysis needs --tolerance and --endpoint-tolerance")
        return ERROR
    IGNORED = frozenset(args.ignore)
    if args.tolerance is not None and not 0 < args.tolerance < 1:
        print(f"--tolerance {args.tolerance}: a relative bound in (0, 1)")
        return ERROR
    if args.endpoint_tolerance is not None:
        if args.tolerance is None:
            print("--endpoint-tolerance needs --tolerance")
            return ERROR
        if not 0 < args.endpoint_tolerance < 1:
            print(f"--endpoint-tolerance {args.endpoint_tolerance}: a relative bound in (0, 1)")
            return ERROR
    TOLERANCE = args.tolerance
    ENDPOINT_TOLERANCE = (args.tolerance if args.endpoint_tolerance is None
                          else args.endpoint_tolerance)  # fmt: skip
    ENDPOINT_ANALYSIS = args.endpoint_analysis
    if args.out.exists():
        print(f"record exists: {args.out}")
        return ERROR
    try:
        hashed = (json.loads(args.rerun_endpoint_tensors.read_text(encoding="utf-8"))
                  if args.rerun_endpoint_tensors else None)  # fmt: skip
        record = check(args.archived, args.rerun, expected, hashed, distances)
    except (CheckError, UnicodeDecodeError) as exc:
        print(f"error: {exc}")
        return ERROR
    write_new(args.out, record)
    report(record)
    verdict = "accepted_for_endpoint_analysis" if ENDPOINT_ANALYSIS else "passed"
    return EQUAL if record[verdict] else DIFFERENT


if __name__ == "__main__":
    sys.exit(main())
