"""Study 3's analysis freeze: the cohorts and K*_h, frozen once the confirmation is done and
before any confirmation label is opened, and the one gate every label reader of the final
analysis passes first.

docs/study3-preregistration.md: "Seeds and order" (order steps 8–11: the confirmation; the cohorts
and K*_h frozen, before any confirmation label is opened; the labeling of the confirmation; the
judge-based analysis), "Questions and estimands" ("Descriptive": K*_h fixed after the confirmation
ends and before any confirmation label is opened, A − P on blocks where both are supported,
validation's harm after the final analysis and apart from it), "Confirmation" (the analysis
cohorts, the stop and the cap's readings; the unsupported started blocks reported), "Judge and
audit" ("Sealing": harm leaves sealed storage at the final analysis only after the cohorts and
K*_h are frozen) and "Voiding and supersession" (a run under another commit does not count).
Nothing here opens a response, a label or a judge output: the record is built from the state
(``study3_progress.read_state``), the endpoints' K and K_harmful, the in-force confirmation
supplement and, for a trace of a label, folder names alone (``os.path.lexists``, nothing opened).

- ``freeze_record``: the record, pure, from candidate N's state at ``done`` and its endpoints.
- ``labels_untouched``: every trace of a confirmation label before the freeze.
- ``analysis_freeze``: the writer's builder over the records under RUNS_DIR; ``FreezeRefused``
  names every problem. It writes nothing (``scripts/study3_analysis.py freeze`` does).
- ``freeze_open``: the gate, with the signature the labeling calls: ``()`` when the final point is
  open, otherwise its problems. ``read_freeze``: the record with the same problems, and no check
  of its own, so that the analysis and the gate cannot diverge.
- ``in_force_supplement``: the in-force confirmation supplement and its link, from one reading.

**The record**, ``docs/study3-analysis-freeze-c<N>.json`` (``freeze_name``), closed schema
(``FIELDS``): ``kind`` (``analysis-freeze``), ``candidate``, ``written_at`` (ISO 8601, UTC),
``code`` (``commit``, ``gpu_code_sha256``), ``supplement`` (the in-force supplement's ``path``
and ``sha256``, ``root`` ``repo``); ``confirmation``: ``N``, ``cap``, ``started`` (the started
seeds), ``stop`` (``counters`` or ``cap``), ``outcome`` and ``active`` (the comparisons);
``comparisons``: per active comparison its ``reading`` (``confirmatory`` or ``descriptive``),
``rule`` (``n_reached`` when its counter reached N, else ``cap``), ``blocks`` (its analysis
cohort, ``[{seed, arms: {arm: {lo, hi, w, width}}}]``: the run ids ordered by K, the weight and
W = ln K_hi − ln K_lo) and ``unknown`` (the started seeds unsupported for it); ``a_minus_p``:
``{blocks}``; ``k_harmful``: ``kstar_h`` (null when no block has K_harmful), ``blocks`` (the
matched blocks, ``{seed, arms: {arm: {lo, hi, w}}}`` ordered by K_harmful) and ``left_out``
(``{seed, reason}`` or ``{seed, arms: {arm: reason}}``); ``validation``: ``A-F`` and ``P-F``
(null once P–F is closed), each ``{blocks}``; ``labels``: the counts checked,
``confirmation_runs`` and ``ledger_runs``.

**The gate** (``freeze_open``) checks, needing no label: the record's schema; Git (below); the
in-force supplement valid, at the record's path and sha256; and the record rebuilt from the
state now (``read_state`` under the record's GPU code, θ₀ at ``THETA0_RUN``) equal to the stored
one, but for ``written_at``, ``code`` and ``labels``: any change of a cohort run's K or state, a
block added, or a stage that is no longer ``done`` makes the current records "not the frozen
ones". It reads run records as ``progress`` does, label-free, and opens nothing sealed. The
labeling refuses a confirmation run while the gate names a problem.

**Rulings** where the registration leaves a detail open, fixed before any confirmation label:
- *Written once, committed alone.* The record is written from a clean checkout and committed
  alone: the commit that adds it (once, ``study3_versions._added``) changes nothing but the record
  from its ``code.commit`` (``committed_changes``), and HEAD and the checkout hold it unchanged.
- *No confirmation label before the freeze.* For every confirmation run of the candidate (any
  state; a run whose candidate is not bound included), no ``judge`` or ``judge-superseded`` entry
  exists in its folder (a dangling link included), and no batch file of the labeling ledger
  (``study3_label_cli``'s, <sealed dir>/study3-ledger/) names any of them.
- *The cohorts* are the state's: per comparison its analysis blocks, at N its first N supported
  blocks by seed order, at the cap all its supported blocks. A comparison's unknown blocks are its
  started blocks unsupported for it.
- *A − P* takes every started block in which A and P are both supported, frozen here, whether or
  not P–F is still active: none once P closed before the confirmation (a block without P cells
  leaves P open).
- *K*_h and the matched blocks* are ``study3_estimates.kstar_h`` and ``matched_blocks`` over A–F's
  cohort, by K_harmful; an endpoint whose K_harmful is missing, ≤ 0 or not finite leaves its block
  out, counted.
- *Validation's blocks*: for A–F, the validation blocks supported for A and F (the n record's
  cohort: the route accepts the n record only when its cohort is exactly these blocks); for P–F,
  those supported for P and F, while P–F is active.
- *A run under another commit* (the state's ``not_counted``) enters no block here, as it enters no
  count.
- *The in-force supplement* is the last link of the supplement's chain (the first record, or
  its successor after a cycle: ``study3_versions.fixation_chain``), valid by its full reader
  (``study3_versions.read_confirmation_fixation``), read through ``in_force_supplement`` alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..paths import REPO_ROOT
from .study3_blocks import OPEN, PAIRS, SUPPORTED, UNRESOLVED, block_state
from .study3_decisions import _sealed_problem, _under_sealed, decisions
from .study3_estimates import kstar_h, matched_blocks
from .study3_progress import COMPARISONS, read_state
from .study3_records import SEEDS, Endpoint
from .study3_search import ARM_ORDER
from .study3_versions import (
    _added,
    _blob,
    committed_changes,
    fixation_chain,
    fixation_name,
    read_confirmation_fixation,
)

KIND = "analysis-freeze"
FIELDS = ("kind", "candidate", "written_at", "code", "supplement", "confirmation", "comparisons",
          "a_minus_p", "k_harmful", "validation", "labels")  # fmt: skip
UNFROZEN = ("written_at", "code", "labels")  # what the gate does not rebuild
TRACES = ("judge", "judge-superseded")  # a run's folders of judge outputs: a label's trace
LEDGER = "<sealed dir>/study3-ledger/"  # the labeling's ledger (study3_label_cli)
_HEX = frozenset("0123456789abcdef")


class FreezeRefused(ValueError):
    """An analysis freeze that cannot be built: its problems named."""


def freeze_name(candidate: int) -> str:
    """``docs/study3-analysis-freeze-c<N>.json``, relative to the checkout."""
    if type(candidate) is not int or candidate not in (1, 2):
        raise ValueError(f"candidate {candidate!r} is not 1 or 2")
    return f"docs/study3-analysis-freeze-c{candidate}.json"


# ------------------------------------------------------------------ the record


def freeze_record(
    state: Mapping[str, Any],
    endpoints: Iterable[Endpoint],
    *,
    candidate: int,
    supplement: Mapping[str, Any],
    code: Mapping[str, Any],
    written_at: str,
    labels: Mapping[str, int],
) -> dict[str, Any]:
    """The analysis freeze of candidate N (module docstring), pure: from ``state`` (``advance``'s
    or ``read_state``'s) and the endpoints it read. ``FreezeRefused`` unless the state is
    candidate N's ``done`` with no problem and a reading for every active comparison."""
    problems = _not_done(state, candidate)
    if problems:
        raise FreezeRefused(_refusal(candidate, problems))
    confirmation = state["confirmation"]
    active = list(confirmation["active"])
    N = confirmation["N"]  # the registration's N
    uncounted = {entry.get("run_id") for entry in state.get("not_counted") or ()}
    kept = [e for e in endpoints if e.run_id not in uncounted]
    by_seed: dict[int | None, list[Endpoint]] = defaultdict(list)
    for e in kept:
        by_seed[e.seed].append(e)

    def of(seed: int, arms: Sequence[str]) -> dict[str, Any]:  # as confirmation_state does
        return block_state([*by_seed.get(seed, ()), *by_seed.get(None, ())], seed, candidate, arms)

    arms = [arm for arm in ARM_ORDER if arm != "P" or "P-F" in active]
    started = list(confirmation["started"])
    blocks = {seed: of(seed, arms) for seed in started}
    comparisons = {}
    found = [f"{c}: block {seed} is started and still {blocks[seed][COMPARISONS[c]]['state']}: "
             "the cohorts are frozen once every started block is resolved"
             for c in active for seed in started
             if blocks[seed][COMPARISONS[c]]["state"] in (OPEN, UNRESOLVED)]  # fmt: skip
    if found:
        raise FreezeRefused(_refusal(candidate, found))
    for comparison in active:
        name = COMPARISONS[comparison]
        cohort = list(confirmation["analysis"][comparison])
        # The worst case ("Confirmation": "Reported: unsupported and excluded started blocks, and
        # the worst-case values"): every started block outside the cohort is unknown, the
        # unsupported ones and the supported ones beyond N alike.
        outside = [s for s in started if s not in set(cohort)]
        if sorted([*cohort, *outside]) != sorted(started) or len(set(cohort)) != len(cohort):
            found.append(f"{comparison}: the cohort and the unknown blocks do not cover each "
                         "started block exactly once")  # fmt: skip
        found += [f"{comparison}: block {seed} of its cohort is not supported as the state reads it"
                  for seed in cohort if blocks[seed][name]["state"] != SUPPORTED]  # fmt: skip
        comparisons[comparison] = {
            "reading": confirmation["readings"][comparison],
            "rule": "n_reached" if confirmation["counters"][comparison] >= N else "cap",
            "blocks": [_block(blocks[seed], PAIRS[name]) for seed in cohort],
            "unknown": outside,
        }
    if found:
        raise FreezeRefused(_refusal(candidate, found))
    a_minus_p = []
    for seed in started:
        block = of(seed, ["A", "P"])
        if not block["blockers"] and all(block["arms"][arm]["state"] == SUPPORTED
                                         for arm in ("A", "P")):  # fmt: skip
            a_minus_p.append(_block(block, ("A", "P")))
    by_id = {e.run_id: e for e in kept}
    kstar, _ = kstar_h(comparisons["A-F"]["blocks"], by_id)
    matched, left_out = matched_blocks(comparisons["A-F"]["blocks"], by_id, kstar)
    seeds = SEEDS["validation"][candidate][: state["validation"]["blocks"]]
    validation: dict[str, Any] = {}
    for comparison, name in COMPARISONS.items():
        if comparison not in active:
            validation[comparison] = None
            continue
        pair = PAIRS[name]
        found_blocks = [of(seed, list(pair)) for seed in seeds]
        validation[comparison] = {"blocks": [_block(b, pair) for b in found_blocks
                                             if b[name]["state"] == SUPPORTED]}  # fmt: skip
    return {
        "kind": KIND,
        "candidate": candidate,
        "written_at": written_at,
        "code": {"commit": code.get("commit"), "gpu_code_sha256": code.get("gpu_code_sha256")},
        "supplement": {"path": supplement.get("path"), "sha256": supplement.get("sha256"),
                       "root": "repo"},
        "confirmation": {"N": N, "cap": confirmation["cap"], "started": started,
                         "stop": confirmation["stop"]["why"], "outcome": state.get("outcome"),
                         "active": active},
        "comparisons": comparisons,
        "a_minus_p": {"blocks": a_minus_p},
        "k_harmful": {"kstar_h": kstar, "blocks": matched, "left_out": left_out},
        "validation": validation,
        "labels": {"confirmation_runs": labels["confirmation_runs"],
                   "ledger_runs": labels["ledger_runs"]},
    }  # fmt: skip


def _not_done(state: Mapping[str, Any], candidate: int) -> list[str]:
    """Why the state is not candidate N's ``done`` with no problem and every reading."""
    where = (state.get("stage"), state.get("candidate"))
    if where != ("done", candidate):
        return [f"the study is at {where[0]!r} of candidate {where[1]}, not at candidate "
                f"{candidate}'s done: the cohorts are frozen once the confirmation is "
                "done"]  # fmt: skip
    found = [f"the state's problem: {p}" for p in state.get("problems") or ()]
    confirmation = state.get("confirmation") or {}
    readings = confirmation.get("readings") or {}
    found += [f"{c}: no reading yet" for c in confirmation.get("active") or ()
              if readings.get(c) is None]  # fmt: skip
    return found


def _block(block: Mapping[str, Any], pair: Sequence[str]) -> dict[str, Any]:
    """A supported block of a comparison as frozen: per arm its runs by K, weight and width."""
    arms = {}
    for arm in pair:
        support = block["arms"][arm]
        arms[arm] = {"lo": support["lo"].run_id, "hi": support["hi"].run_id,
                     "w": support["weight"],
                     "width": math.log(support["k_hi"]) - math.log(support["k_lo"])}  # fmt: skip
    return {"seed": block["seed"], "arms": arms}


def _refusal(candidate: int, problems: Sequence[str]) -> str:
    return f"analysis-freeze-c{candidate}: " + "; ".join(problems)


# ------------------------------------------------------------------ no label before the freeze


def labels_untouched(
    endpoints: Iterable[Endpoint], *, candidate: int, ledger_runs: Collection[str]
) -> list[str]:
    """Every trace of a confirmation label of candidate N before the freeze (module docstring,
    "Rulings"): a ``judge`` or ``judge-superseded`` entry in a confirmation run's folder (names
    alone: nothing is opened), or the labeling ledger naming a confirmation run."""
    found = []
    ledger = frozenset(ledger_runs)
    for e in _confirmation_runs(endpoints, candidate):
        for name in TRACES:
            path = Path(e.run_dir) / name
            if os.path.lexists(path):
                found.append(f"{e.run_id}: its folder holds {name}/ ({path}): a trace of a "
                             "confirmation label before the freeze")  # fmt: skip
        if e.run_id in ledger:
            found.append(f"{e.run_id}: the labeling ledger ({LEDGER}) names it: a confirmation "
                         "label before the freeze")  # fmt: skip
    return found


def _confirmation_runs(endpoints: Iterable[Endpoint], candidate: int) -> list[Endpoint]:
    """The candidate's confirmation runs, in any state, a run whose candidate is not bound
    included, each once, by run id."""
    found = {e.run_id: e for e in endpoints if e.candidate in (None, candidate)
             and "confirmation" in (e.cell.step, e.role)}  # fmt: skip
    return [found[run_id] for run_id in sorted(found)]


# ------------------------------------------------------------------ the builder


def in_force_supplement(
    candidate: int, *, runs_root: str | Path, repo: str | Path = REPO_ROOT
) -> tuple[dict[str, Any] | None, dict[str, str] | None, list[str]]:
    """The in-force confirmation supplement of candidate N, valid by its full reader
    (``read_confirmation_fixation``), its chain's last link (``fixation_chain``: the first record,
    or its successor after a cycle), its link
    ``{"path", "sha256", "root": "repo"}`` and its problems; absent: (None, None, []). The record
    and the link are one reading's: the bytes hashed must parse to the record the reader returned
    (otherwise the file changed while it was read, a problem), so that what a caller takes from
    the record (the bootstrap's seed, the audit's seeds) and the link it records always agree."""
    record, problems = read_confirmation_fixation(candidate, runs_root=runs_root, repo=repo)
    if record is None:
        return None, None, list(problems)
    name = fixation_chain("confirmation", candidate, repo=repo)[-1]
    try:
        data = (Path(repo) / name).read_bytes()
        same = json.loads(data) == record
    except (OSError, ValueError):
        same = False
    if not same:
        return None, None, [f"{name}: it changed while it was read, or cannot be read again"]
    return record, {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "root": "repo"}, []


def analysis_freeze(
    runs_root: str | Path,
    candidate: int,
    *,
    code: Mapping[str, str],
    repo: str | Path = REPO_ROOT,
    theta0: str | Path | None = None,
    ledger_runs: Collection[str],
) -> dict[str, Any]:
    """Candidate N's analysis freeze from the records under ``runs_root`` and the checkout
    ``repo`` (module docstring): the state and its endpoints (``read_state`` under ``code``'s GPU
    code), the in-force supplement (valid, pinned by its sha256) and no trace of a confirmation
    label (``ledger_runs``: the runs the labeling ledger names). ``code``: the commit it is
    written at and that commit's GPU code. ``FreezeRefused`` names every problem."""
    freeze_name(candidate)
    state, endpoints = read_state(runs_root, repo=repo, expected_code=code["gpu_code_sha256"],
                                  theta0=theta0)  # fmt: skip
    link, problems = _supplement(candidate, runs_root, repo)
    problems += labels_untouched(endpoints, candidate=candidate, ledger_runs=ledger_runs)
    counts = {"confirmation_runs": len(_confirmation_runs(endpoints, candidate)),
              "ledger_runs": len(frozenset(ledger_runs))}  # fmt: skip
    try:
        record = freeze_record(state, endpoints, candidate=candidate, supplement=link or {},
                               code=code, written_at=_now(), labels=counts)  # fmt: skip
    except FreezeRefused as exc:
        problems.insert(0, str(exc).removeprefix(f"analysis-freeze-c{candidate}: "))
    else:
        problems += _validation_cohort(record, candidate, runs_root, repo)
    if problems:
        raise FreezeRefused(_refusal(candidate, problems))
    return record


def _validation_cohort(record: Mapping[str, Any], candidate: int, runs_root: str | Path,
                       repo: str | Path) -> list[str]:  # fmt: skip
    """The frozen A-F validation blocks against the in-force n record's cohort (registration,
    "Sample size from validation"): the same seeds."""
    entry = decisions(runs_root, candidate, repo=repo).get(f"n-c{candidate}") or {}
    n = entry.get("record")
    if n is None:
        return [
            f"validation: the in-force n-c{candidate} record is {entry.get('state', 'missing')}"
        ]
    frozen = [b["seed"] for b in record["validation"]["A-F"]["blocks"]]
    cohort = list(n["cohort"]["seeds"])
    if frozen != cohort:
        return [
            f"validation: the A-F validation blocks {frozen} differ from the n-c{candidate} "
            f"record's cohort {cohort}"
        ]
    return []


def _supplement(candidate: int, runs_root: str | Path,
                repo: str | Path) -> tuple[dict[str, str] | None, list[str]]:  # fmt: skip
    """The in-force supplement's link, and its problems named as the supplement's."""
    _, link, found = in_force_supplement(candidate, runs_root=runs_root, repo=repo)
    if link is not None:
        return link, []
    name = fixation_name("confirmation", candidate)
    return None, [f"supplement: {p}" for p in found] or [
        f"supplement: the confirmation supplement {name} is missing"]  # fmt: skip


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ------------------------------------------------------------------ the gate


def freeze_open(candidate: int, runs_root: str | Path, repo_root: str | Path) -> tuple[str, ...]:
    """The gate before any confirmation label opens (module docstring, "The gate"): ``()`` when
    candidate N's final point is open, otherwise every problem, each naming the record."""
    return tuple(_open(candidate, Path(runs_root), Path(repo_root))[1])


def read_freeze(
    candidate: int, *, runs_root: str | Path, repo: str | Path = REPO_ROOT
) -> tuple[dict[str, Any] | None, list[str]]:
    """Candidate N's analysis freeze and ``freeze_open``'s problems: the record only when there
    is no problem. No check of its own."""
    return _open(candidate, Path(runs_root), Path(repo))


def _open(candidate: int, root: Path, repo: Path) -> tuple[dict[str, Any] | None, list[str]]:
    name = freeze_name(candidate)
    record, data, problems = _stored(repo, name, candidate)
    if record is not None:
        problems += _git_problems(repo, name, record, data)
        link, found = _supplement(candidate, root, repo)
        problems += found
        if link is not None and link != record["supplement"]:
            problems.append(f"supplement: the in-force supplement {link} is not the frozen "
                            f"{record['supplement']}")  # fmt: skip
        problems += _against_state(record, candidate, root, repo)
    if problems:
        return None, [f"{name}: {p}" for p in problems]
    return record, []


def _stored(repo: Path, name: str,
            candidate: int) -> tuple[dict[str, Any] | None, bytes, list[str]]:  # fmt: skip
    """The record as stored, its bytes, and the problems of its schema."""
    path = repo / name
    if not os.path.lexists(path):
        return None, b"", ["no analysis freeze: the record is not written, and no confirmation "
                           "label opens before it"]  # fmt: skip
    if _under_sealed(path, repo):
        return None, b"", [_sealed_problem(path)]
    try:
        data = path.read_bytes()
        record = json.loads(data)
    except (OSError, ValueError) as exc:
        return None, b"", [f"unreadable ({type(exc).__name__})"]
    problems = _schema(record, candidate)
    return (None if problems else record), data, problems


def _schema(record: Any, candidate: int) -> list[str]:
    """The closed schema's problems; the frozen part is checked against its rebuild."""
    if not isinstance(record, dict) or tuple(sorted(record)) != tuple(sorted(FIELDS)):
        found = sorted(record) if isinstance(record, dict) else type(record).__name__
        return [f"its fields {found} are not the record's {list(FIELDS)}"]
    found = []
    if record["kind"] != KIND:
        found.append(f"kind is {record['kind']!r}, not {KIND!r}")
    if type(record["candidate"]) is not int or record["candidate"] != candidate:
        found.append(f"candidate is {record['candidate']!r}, not {candidate}")
    if not _utc(record["written_at"]):
        found.append(f"written_at {record['written_at']!r} is no ISO 8601 time in UTC")
    code = record["code"]
    if not (isinstance(code, dict) and set(code) == {"commit", "gpu_code_sha256"}
            and _hex(code["commit"], 40) and _hex(code["gpu_code_sha256"], 64)):  # fmt: skip
        found.append("code is not {commit: 40 hexadecimal digits, gpu_code_sha256: 64}")
    supplement = record["supplement"]
    if not (isinstance(supplement, dict) and set(supplement) == {"path", "sha256", "root"}
            and isinstance(supplement["path"], str) and _hex(supplement["sha256"], 64)
            and supplement["root"] == "repo"):  # fmt: skip
        found.append("supplement is not {path, sha256: 64 hexadecimal digits, root: repo}")
    labels = record["labels"]
    if not (isinstance(labels, dict) and set(labels) == {"confirmation_runs", "ledger_runs"}
            and all(type(v) is int and v >= 0 for v in labels.values())):  # fmt: skip
        found.append("labels is not {confirmation_runs, ledger_runs}, counts")
    found += [f"{key} is no object" for key in FIELDS if key not in (*UNFROZEN, "kind", "candidate",
              "supplement") and not isinstance(record[key], dict)]  # fmt: skip
    return found


def _utc(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        offset = datetime.fromisoformat(value).utcoffset()
    except ValueError:
        return False
    return offset == timedelta(0)


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and set(value) <= _HEX


def _git_problems(repo: Path, name: str, record: Mapping[str, Any], data: bytes) -> list[str]:
    """Written once and committed alone as a record-only descendant of its ``code.commit``,
    unchanged since (module docstring, "Rulings")."""
    added, why = _added(repo, name)
    if added is None:
        return [str(why)]
    found = []
    commit = record["code"]["commit"]
    changes = committed_changes(commit, added, repo)
    if changes is None:
        found.append(f"the commit that added it, {added}, does not descend from its code.commit "
                     f"{commit}")  # fmt: skip
    elif changes != [name]:
        found.append(f"the commit that added it, {added}, is no record-only descendant of its "
                     f"code.commit {commit}: it changes {changes}")  # fmt: skip
    at_head = _blob(repo, "HEAD", name)
    if at_head != _blob(repo, added, name):
        found.append("it was rewritten by a later commit: a record is written once")
    if data != at_head:
        found.append("the checkout holds an uncommitted change to it")
    return found


def _against_state(record: Mapping[str, Any], candidate: int, root: Path, repo: Path) -> list[str]:
    """The record rebuilt from the state now, against the stored one (module docstring, "The
    gate")."""
    try:
        state, endpoints = read_state(root, repo=repo,
                                      expected_code=record["code"]["gpu_code_sha256"])  # fmt: skip
        rebuilt = freeze_record(state, endpoints, candidate=candidate,
                                supplement=record["supplement"], code=record["code"],
                                written_at=record["written_at"],
                                labels=record["labels"])  # fmt: skip
    except FreezeRefused as exc:
        why = str(exc).removeprefix(f"analysis-freeze-c{candidate}: ")
        return [f"the current records are not the frozen ones: {why}"]
    except (OSError, ValueError) as exc:
        return [f"the study's state cannot be read ({type(exc).__name__}: {exc})"]
    rebuilt = json.loads(json.dumps(rebuilt, allow_nan=False))
    differ = [key for key in FIELDS if key not in UNFROZEN and rebuilt[key] != record[key]]
    if differ:
        return [f"the current records are not the frozen ones: its {', '.join(differ)} differ "
                "from the records now"]  # fmt: skip
    return []
