"""Study 3's rate search and its scientific repair, as pure functions of endpoints.

docs/study3-preregistration.md: "Candidates and rate search" ("Search", "Scientific repair"),
"Failure routes", amendment 1 (what K-only runs measure) and amendment 3 (one repair round per
arm, after a passed numerical qualification). The inputs are the ``Endpoint`` records
of ``study3_records.read_endpoints`` and its refusals (``problems``, a required argument of both
functions); nothing here reads a file, a response, a harm label or a judge output.

- ``CANDIDATES``, ``SLOPES``: the registered targets, starting rates and slopes.
- ``adjusted_rate``: the rule's step, clipped and rounded.
- ``search_state``: round 0 and round 1 of a candidate's search, per arm and target, with the
  decision, the selected rates and what the step needs next.
- ``repair_state``: the scientific repair of an arm's selected rates, once their numerical
  qualification has passed.

**The rule** ("Search", item 2). For an (arm, target), m is the median over the four search seeds
of ln K at step 25. If |m − ln K_target| ≤ 0.05 the rate is kept; otherwise ln η_new = ln η +
(ln K_target − m)/b, with b = 1.25 (A), 1.30 (F), 1.24 (P), the ratio η_new/η clipped to
[1/1.15, 1.15] and the rate then rounded to 3 significant digits. Round 1 runs the adjusted rates
once on the same four seeds, and they are the selected rates; a kept rate is selected as it is,
with its round-0 runs as its search runs. There is no second round. The median of four is the mean
of the two middle values. The comparison with 0.05 allows 10⁻⁹ in ln K for the floating-point
difference of two logarithms, so a miss of exactly 0.05 is kept.

**Which runs enter a median** ("Search", item 2; "Failure routes"). Only the endpoints of the
current prescribed cells: the step keys ``c<N>/search/r0`` (the starting rate) and
``c<N>/search/r1`` (the adjusted rate), the arm, the rate and the candidate's search seeds
(``study3_records.step_key``, ``Cell.key``). A median needs all four endpoints in state
``usable`` (a K-only run: 25 updates, finite records, positive finite K). No run is silently left
out:
- ``scientific`` (an instability stop): the arm fails for the candidate (``failed``), with the
  stopped run; no rerun passes it;
- ``technical`` or ``held`` (an unexplained failure, a record that does not conform, an outcome
  the engineering cycle could not resolve): the arm is ``technical``, blocked for diagnosis;
- ``pending``, or ``recovered`` with no later attempt: the step waits (``pending``);
- ``superseded`` and every attempt a later one replaces: listed under the cell's runs, never
  counted. A cell none of whose attempts counts is missing, and the round has to be issued again.
A failed arm is ``failed`` even when another of its runs is technical. Two counting attempts of a
cell cannot arise (``read_endpoints`` refuses them), and are ``technical`` here if they do.

**Refusals block.** The state is ``blocked``, with the problems named and nothing computed, for:
- any refusal of ``read_endpoints`` (the ``problems`` argument, however it is passed);
- an endpoint of the candidate in state ``refused`` (its run id and reasons are named, whether or
  not ``problems`` lists it), an endpoint with no candidate (a run directory with no start), and
  an endpoint of the candidate in a state the reader does not define;
- an endpoint of the candidate's search, numerical or repair step that cannot be assigned to an
  arm, a rate and a seed (an unbound attempt: its fields are None), and one that matches none of
  the rule's cells (a rate that is not the starting or the adjusted rate, a round 1 for a kept
  rate, a seed outside the cell's seeds, a round 2, a repair run at another rate).
Ambiguous records are never read as if they were not: a refused attempt is no missing cell.

**What the search state says.** ``next`` is, in order: ``blocked`` (a refusal); ``failures`` (an
arm is ``failed`` or ``technical``: the failure route, "Failure routes", comes before anything
else is issued; each arm's own ``status`` still says what that arm needs, so that a failure of P
alone leaves A and F going); ``pending`` (a run has not finished); ``round0`` (cells of round 0 to
issue); ``round1`` (cells of round 1 to issue); ``numerical`` (every arm is ``ready``: both its
selected rates have their search runs). ``selected`` holds the selected rates of the ready arms
only. Search utility (amendment 1) is recorded in the endpoints and decides nothing here.

**The scientific repair** ("Scientific repair"). At most one per arm and candidate, before
validation, and only for a K miss: the median ln K of a selected rate over its search and
numerical seeds together (4 + 3) lies more than 0.05 from its target (the same 10⁻⁹ allowance). The
rule's step applied to that median gives the repaired rate, which gets its own numerical
qualification (the step key ``c<N>/repair``, the numerical seeds); if it differs from the old
rate by less than 2 % the repair is impossible and the old rate stays, for validation to decide.
- *Gate.* An arm's repair is computed only when every one of its selected rates is ``qualified``
  by the numerical qualification (``qualification_verdicts``' rate state, the FA arm's same-path
  control included, passed in as ``qualified``: arm → rate → state). An arm with any rate
  ``failing``, ``not_qualified``, ``incomplete`` or not reported gets no repair, and says why: a
  numerical failure has its failure route and is never repaired this way.
- *One repair per arm.* The repair of an arm is one round that adjusts every one of its selected
  rates that misses (an arm has two rates, and the registration allows one repair of the arm,
  not of a rate). It is decided only when the medians of both rates exist (``incomplete``
  otherwise). Nothing is repaired again: the repaired rate's own median over its repair
  qualification's numerical seeds (the only seeds it runs on) is reported under ``second`` with
  ``missed``, and no further rate is derived from it.
- *What the repair state says.* ``next`` is, in order: ``blocked``; ``repair`` (the runs of a
  repaired rate's qualification are to be issued: some cell of the rate's ``c<N>/repair`` step has
  no attempt); ``repair_pending`` (they are issued, and a run has not finished); ``undecided`` (an
  arm has no decision: not qualified, a median missing, or no selected rates, ``unselected``;
  ``undecided`` names each arm and why); ``repaired`` (every repaired rate's runs have finished:
  ``second`` reports each one's median and miss, or the failure of a run, whose route is the
  qualification's); ``none`` (no repair). An arm absent from ``selected`` is reported as
  ``unselected``, never skipped.
- *Where the rule is idle.* A miss above 0.05 moves the rate by at least 1 − e^(−0.05/1.30) ≈ 3.8 %
  before rounding and by at least 3.2 % after it, so the registered slopes never make a repair
  impossible; the 2 % test is kept as registered and is tested with a slope that would.

**Rulings** where the registration leaves a detail open, fixed here before any study-3 record:
- *A K-only run whose K is not positive* (``unusable``) is neither an instability stop nor an
  incomplete record. The median needs positive finite K ("Search", item 2), so it is ``technical``:
  blocked for diagnosis rather than a scientific failure that would move the study to candidate 2.
- *A kept rate has no round 1*: its round-0 runs are its search runs, in the repair's median too.
- *Rates compare as numbers* (relative tolerance ``study3_records.RATE_TOLERANCE``): the spelling
  of a rate in a campaign or a qualification verdict does not matter.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .study3_records import COUNTING, RATE_TOLERANCE, SEEDS, STATES, Endpoint, step_key

ARM_ORDER = ("A", "F", "P")
LABELS = ("l", "u")  # η_l targets K_l, η_u targets K_u
# "Candidates and rate search": K_l, K*, K_u and the starting rates η_l / η_u per arm.
CANDIDATES: dict[int, dict[str, Any]] = {
    1: {"targets": {"l": 4.80, "star": 6.00, "u": 7.50},
        "start": {"A": {"l": 1.45e-5, "u": 2.11e-5},
                  "F": {"l": 1.30e-5, "u": 1.83e-5},
                  "P": {"l": 1.30e-5, "u": 1.85e-5}}},
    2: {"targets": {"l": 4.00, "star": 5.00, "u": 6.25},
        "start": {"A": {"l": 1.27e-5, "u": 1.81e-5},
                  "F": {"l": 1.14e-5, "u": 1.58e-5},
                  "P": {"l": 1.14e-5, "u": 1.59e-5}}},
}  # fmt: skip
SLOPES = {"A": 1.25, "F": 1.30, "P": 1.24}  # the slope b of ln K in ln η
TOLERANCE = 0.05  # in ln K: the search's keep and the repair's miss
CLIP = 1.15  # the step's ratio η_new/η lies in [1/CLIP, CLIP]
DIGITS = 3  # significant digits of a new rate
IMPOSSIBLE_BELOW = 0.02  # a repair that moves the rate by less than this is impossible
SLACK = 1e-9  # floating-point allowance of the comparisons with TOLERANCE and IMPOSSIBLE_BELOW
QUALIFIED = "qualified"  # the numerical qualification's rate state that passes the repair's gate
REPAIR_STEP = "repair"
# a round's or a cell's state, worst first
ROUND_STATES = ("scientific", "technical", "pending", "missing", "usable")
REPAIRED_STEPS = ("search", "numerical")  # the steps whose runs enter a repaired rate's median


def adjusted_rate(rate: float, median_ln_k: float, target: float, slope: float) -> float:
    """The rule's new rate ("Search", item 2): ln η_new = ln η + (ln target − m)/b, the ratio
    η_new/η clipped to [1/1.15, 1.15], then rounded to 3 significant digits."""
    ratio = math.exp((math.log(target) - median_ln_k) / slope)
    ratio = min(max(ratio, 1.0 / CLIP), CLIP)
    return float(f"{rate * ratio:.{DIGITS - 1}e}")


def search_state(
    endpoints: Iterable[Endpoint],
    candidate: int,
    *,
    problems: Sequence[str],
) -> dict[str, Any]:
    """The candidate's search (module docstring): ``arms`` (per arm its ``status``: ``failed``,
    ``technical``, ``pending``, ``round0``, ``round1`` or ``ready``; its ``targets`` ``l`` and
    ``u``, each with its starting rate, ``round0`` (rate, runs, cells, ``median_ln_k``, ``miss``),
    ``decision`` (``kept``, or the round-1 rate), ``round1``, ``selected_rate`` and ``complete``;
    the ``failed`` and ``technical`` runs), the ``selected`` rates of the ready arms, the
    ``failures`` (arm → status) and ``next``. ``problems``: the refusals of ``read_endpoints``
    (required: pass ``()`` only for records that had none)."""
    spec = _spec(candidate)
    seeds = SEEDS["search"][candidate]
    endpoints = list(endpoints)
    if blockers := _blockers(endpoints, candidate, problems):
        return _blocked(candidate, blockers)
    mine = [e for e in endpoints if e.candidate == candidate and e.cell.step == "search"]
    arms: dict[str, Any] = {}
    claimed: set[str] = set()
    for arm in ARM_ORDER:
        targets = {label: _search_target(mine, candidate, spec, arm, label, seeds)
                   for label in LABELS}  # fmt: skip
        rounds = [r for t in targets.values() for r in (t["round0"], t["round1"]) if r is not None]
        claimed |= {e.run_id for r in rounds for e in r["runs"]}
        failed = tuple(e for r in rounds for e in r["failed"])
        technical = tuple(e for r in rounds for e in r["technical"])
        arms[arm] = {"status": _search_status(targets, failed, technical), "targets": targets,
                     "failed": failed, "technical": technical}  # fmt: skip
    stray = [f"{e.run_id}: a search run {_why_stray(e)} ({_cell_name(e)})"
             for e in mine if e.run_id not in claimed]  # fmt: skip
    if stray:
        return _blocked(candidate, stray)
    failures = {
        arm: a["status"] for arm, a in arms.items() if a["status"] in ("failed", "technical")
    }
    statuses = {a["status"] for a in arms.values()}
    if failures:
        following = "failures"
    else:
        following = next(s for s in ("pending", "round0", "round1", "numerical")
                         if s in statuses or s == "numerical")  # fmt: skip
    selected = {arm: {label: a["targets"][label]["selected_rate"] for label in LABELS}
                for arm, a in arms.items() if a["status"] == "ready"}  # fmt: skip
    return {"candidate": candidate, "problems": [], "arms": arms, "selected": selected,
            "failures": failures, "next": following}  # fmt: skip


def repair_state(
    endpoints: Iterable[Endpoint],
    candidate: int,
    selected: Mapping[str, Mapping[str, float]],
    qualified: Mapping[str, Mapping[float, str]],
    *,
    problems: Sequence[str],
) -> dict[str, Any]:
    """The scientific repair of a candidate's arms (module docstring). ``selected``: arm → ``l`` /
    ``u`` → the search's selected rate (``search_state``'s ``selected``); ``qualified``: arm →
    rate → the numerical qualification's rate state. Per arm: ``status`` (``not_qualified``, with
    ``why`` the states; ``incomplete``, a median is missing (the rates' cells say which);
    ``none``; ``impossible``; ``repair``), its rates (``median_ln_k`` over the search and
    numerical runs, ``miss``, ``missed``, ``repaired_rate``, ``change``, ``impossible``,
    ``action``: ``none``, ``repair``, ``impossible`` or ``incomplete``, and for a repair
    ``second``: the repaired rate's own median and miss, reported and never repaired) and
    ``repairs`` (label → repaired rate); an arm absent from ``selected`` is ``unselected``.
    ``undecided``: arm → status of each arm without a decision. ``next``: ``blocked``,
    ``repair``, ``repair_pending``, ``undecided``, ``repaired`` or ``none`` (module docstring).
    ``problems``: the refusals of ``read_endpoints`` (required)."""
    spec = _spec(candidate)
    for arm, rates in selected.items():
        if arm not in ARM_ORDER or set(rates) != set(LABELS):
            raise ValueError(f"selected rates of {arm!r} must name arms {ARM_ORDER} and rates "
                             f"{LABELS}")  # fmt: skip
    endpoints = list(endpoints)
    if blockers := _blockers(endpoints, candidate, problems):
        return _blocked(candidate, blockers)
    mine = [e for e in endpoints if e.candidate == candidate]
    arms: dict[str, Any] = {}
    claimed: set[str] = set()
    for arm in ARM_ORDER:
        if arm not in selected:
            arms[arm] = {"status": "unselected", "rates": {}, "repairs": {}}
            continue
        states = {label: _lookup(qualified.get(arm, {}), selected[arm][label]) or "not reported"
                  for label in LABELS}  # fmt: skip
        if any(state != QUALIFIED for state in states.values()):
            arms[arm] = {"status": "not_qualified", "why": states, "rates": {}, "repairs": {}}
            continue
        rates = {label: _repair_rate(mine, candidate, spec, arm, label, selected[arm][label])
                 for label in LABELS}  # fmt: skip
        claimed |= {e.run_id for r in rates.values() if r["second"] for e in r["second"]["runs"]}
        actions = {r["action"] for r in rates.values()}
        status = next(a for a in ("incomplete", "repair", "impossible", "none") if a in actions)
        repairs = {label: r["repaired_rate"] for label, r in rates.items()
                   if r["action"] == "repair" and status != "incomplete"}  # fmt: skip
        arms[arm] = {"status": status, "rates": rates, "repairs": repairs}
    stray = [f"{e.run_id}: a {e.cell.step} run {_why_stray(e)} ({_cell_name(e)})"
             for e in mine if _repair_stray(e, claimed)]  # fmt: skip
    if stray:
        return _blocked(candidate, stray)
    repairs = {arm: a["repairs"] for arm, a in arms.items() if a["repairs"]}
    undecided = {arm: a["status"] for arm, a in arms.items()
                 if a["status"] in ("not_qualified", "incomplete", "unselected")}  # fmt: skip
    seconds = [r["second"] for a in arms.values() for r in a["rates"].values()
               if r["action"] == "repair" and a["repairs"]]  # fmt: skip
    if any(second["missing"] for second in seconds):
        following = "repair"
    elif any(second["pending"] for second in seconds):
        following = "repair_pending"
    elif undecided:
        following = "undecided"
    else:
        following = "repaired" if seconds else "none"
    return {"candidate": candidate, "problems": [], "arms": arms, "repairs": repairs,
            "undecided": undecided, "next": following}  # fmt: skip


# ------------------------------------------------------------------ the search


def _spec(candidate: int) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError(f"no candidate {candidate!r} (registered: {sorted(CANDIDATES)})")
    return CANDIDATES[candidate]


def _blocked(candidate: int, problems: Sequence[str]) -> dict[str, Any]:
    """The state of records with a refusal: nothing is computed from them."""
    return {"candidate": candidate, "problems": list(problems), "arms": {}, "selected": {},
            "failures": {}, "repairs": {}, "undecided": {}, "next": "blocked"}  # fmt: skip


def _blockers(endpoints: Sequence[Endpoint], candidate: int, problems: Sequence[str]) -> list[str]:
    """Why the records cannot be read (module docstring, "Refusals block"): the reader's refusals,
    and the endpoints of this candidate that are refused, have no candidate or are in a state the
    reader does not define, each named by its run id (once, when ``problems`` names it too)."""
    out = [*problems]
    named = {problem.split(":", 1)[0] for problem in problems}
    for e in endpoints:
        if e.candidate not in (None, candidate) or e.run_id in named:
            continue
        if e.state == "refused":
            out.append(f"{e.run_id}: refused: {'; '.join(e.reasons) or 'no reason recorded'}")
        elif e.candidate is None:
            out.append(f"{e.run_id}: an attempt that no candidate owns")
        elif e.state not in STATES:
            out.append(f"{e.run_id}: a state the reader does not define ({e.state!r})")
    return out


def _unassigned(e: Endpoint) -> bool:
    """Whether an endpoint cannot be assigned to an arm, a rate and a seed (an unbound attempt)."""
    return e.arm is None or e.lr is None or e.seed is None


def _repair_stray(e: Endpoint, claimed: set[str]) -> bool:
    """Whether an endpoint of the repair's inputs has no cell: it is bound to no arm, rate and
    seed, or it is a repair run no repaired rate's cells claim."""
    if e.cell.step not in (*REPAIRED_STEPS, REPAIR_STEP):
        return False
    return _unassigned(e) or (e.cell.step == REPAIR_STEP and e.run_id not in claimed)


def _why_stray(e: Endpoint) -> str:
    return "bound to no arm, rate and seed" if _unassigned(e) else "outside the rule's cells"


def _cell_name(e: Endpoint) -> str:
    rate = "no rate" if e.lr is None else f"{e.lr:g}"
    return f"{e.cell.key}, arm {e.arm}, rate {rate}, seed {e.seed}"


def _search_target(mine: Sequence[Endpoint], candidate: int, spec: Mapping[str, Any], arm: str,
                   label: str, seeds: range) -> dict[str, Any]:  # fmt: skip
    """One (arm, target) of the search: round 0 at the starting rate, the decision, and for an
    adjusted rate round 1 at it."""
    target, start = spec["targets"][label], spec["start"][arm][label]
    round0 = _round(mine, (step_key(candidate, "search", 0),), arm, start, seeds, target)
    out: dict[str, Any] = {"target": target, "start_rate": start, "slope": SLOPES[arm],
                           "round0": round0, "decision": None, "round1": None,
                           "selected_rate": None, "complete": False}  # fmt: skip
    median = round0["median_ln_k"]
    if median is None:
        return out
    if round0["miss"] <= TOLERANCE + SLACK:
        return {**out, "decision": "kept", "selected_rate": start, "complete": True}
    rate = adjusted_rate(start, median, target, SLOPES[arm])
    round1 = _round(mine, (step_key(candidate, "search", 1),), arm, rate, seeds, target)
    return {**out, "decision": rate, "round1": round1, "selected_rate": rate,
            "complete": round1["state"] == "usable"}  # fmt: skip


def _search_status(targets: Mapping[str, Mapping[str, Any]], failed: Sequence[Endpoint],
                   technical: Sequence[Endpoint]) -> str:  # fmt: skip
    """An arm's status, from its rounds (module docstring)."""
    if failed:
        return "failed"
    if technical:
        return "technical"
    rounds = [r for t in targets.values() for r in (t["round0"], t["round1"]) if r is not None]
    if any(r["state"] == "pending" for r in rounds):
        return "pending"
    if any(t["round0"]["state"] == "missing" for t in targets.values()):
        return "round0"
    if any(t["round1"] is not None and t["round1"]["state"] == "missing"
           for t in targets.values()):  # fmt: skip
        return "round1"
    return "ready"


# ------------------------------------------------------------------ the repair


def _repair_rate(mine: Sequence[Endpoint], candidate: int, spec: Mapping[str, Any], arm: str,
                 label: str, rate: float) -> dict[str, Any]:  # fmt: skip
    """One selected rate of a qualified arm: its search and numerical runs, the median over them
    and the repair the rule gives."""
    target, slope = spec["targets"][label], SLOPES[arm]
    search_keys = tuple(step_key(candidate, "search", r) for r in (0, 1))
    search = _round(mine, search_keys, arm, rate, SEEDS["search"][candidate], target)
    numerical = _round(mine, (step_key(candidate, "numerical"),), arm, rate,
                       SEEDS["numerical"][candidate], target)  # fmt: skip
    out: dict[str, Any] = {"rate": rate, "target": target, "slope": slope, "search": search,
                           "numerical": numerical, "median_ln_k": None, "miss": None,
                           "missed": None, "repaired_rate": None, "change": None,
                           "impossible": None, "action": "incomplete", "second": None}  # fmt: skip
    if search["state"] != "usable" or numerical["state"] != "usable":
        return out
    median = statistics.median([*search["ln_k"], *numerical["ln_k"]])
    miss = abs(median - math.log(target))
    out |= {"median_ln_k": median, "miss": miss, "missed": miss > TOLERANCE + SLACK}
    if not out["missed"]:
        return {**out, "action": "none"}
    repaired = adjusted_rate(rate, median, target, slope)
    change = abs(repaired / rate - 1.0)
    impossible = change < IMPOSSIBLE_BELOW and not math.isclose(change, IMPOSSIBLE_BELOW,
                                                                 rel_tol=SLACK)  # fmt: skip
    out |= {"repaired_rate": repaired, "change": change, "impossible": impossible}
    if impossible:
        return {**out, "action": "impossible"}
    # the repaired rate's own qualification runs on the numerical seeds; its median is reported
    second = _round(mine, (step_key(candidate, REPAIR_STEP),), arm, repaired,
                    SEEDS["numerical"][candidate], target)  # fmt: skip
    second["missed"] = None if second["miss"] is None else second["miss"] > TOLERANCE + SLACK
    return {**out, "action": "repair", "second": second}


def _lookup(states: Mapping[float, str], rate: float) -> str | None:
    """The state a mapping keyed by rates holds for ``rate`` (rates compare as numbers)."""
    for key, state in states.items():
        if _same(key, rate):
            return state
    return None


# ------------------------------------------------------------------ rounds and cells


def _same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and math.isclose(a, b, rel_tol=RATE_TOLERANCE)


def _round(endpoints: Sequence[Endpoint], keys: tuple[str, ...], arm: str, rate: float,
           seeds: range, target: float) -> dict[str, Any]:  # fmt: skip
    """The cells of an (arm, rate) under the step keys ``keys``, one per seed: every attempt
    listed in ``runs``, each cell's state and the deciding runs, the round's ``state`` (the worst
    cell's, ``ROUND_STATES``), and for four usable cells the median ln K and its miss from the
    target (the median of a round that has a technical, scientific, pending or missing cell is
    never computed)."""
    runs = sorted((e for e in endpoints if e.cell.key in keys and e.arm == arm
                   and _same(e.lr, rate) and e.seed in seeds),
                  key=lambda e: (e.seed, e.attempt or 0, e.run_id))  # fmt: skip
    cells: dict[int, str] = {}
    deciding: dict[str, list[Endpoint]] = {"scientific": [], "technical": []}
    waiting: list[int] = []
    missing: list[int] = []
    ln_k: list[float] = []
    for seed in seeds:
        state, found = _cell([e for e in runs if e.seed == seed])
        cells[seed] = state
        if state in deciding:
            deciding[state].extend(found)
        elif state == "pending":
            waiting.append(seed)
        elif state == "missing":
            missing.append(seed)
        else:
            ln_k.append(math.log(found[0].K))
    state = next(s for s in ROUND_STATES if s == "usable" or s in set(cells.values()))
    median = statistics.median(ln_k) if state == "usable" else None
    return {"keys": keys, "rate": rate, "target": target, "runs": tuple(runs), "cells": cells,
            "state": state, "failed": tuple(deciding["scientific"]),
            "technical": tuple(deciding["technical"]), "pending": tuple(waiting),
            "missing": tuple(missing), "ln_k": tuple(ln_k), "median_ln_k": median,
            "miss": None if median is None else abs(median - math.log(target))}  # fmt: skip


def _cell(attempts: Sequence[Endpoint]) -> tuple[str, list[Endpoint]]:
    """A cell's state from its attempts, and the attempts that decide it (module docstring,
    "Which runs enter a median")."""
    counting = [e for e in attempts if e.state in COUNTING]
    stopped = [e for e in counting if e.state == "scientific"]
    held = [e for e in attempts if e.state == "held"]
    if stopped:
        return "scientific", stopped
    if held:
        return "technical", held
    if len(counting) > 1:
        return "technical", counting
    if counting:
        run = counting[0]
        if run.state == "usable" and _positive(run.K):
            return "usable", [run]
        if run.state == "pending":
            return "pending", [run]
        return "technical", [run]  # technical, unusable, or a usable run without a positive K
    recovered = [e for e in attempts if e.state == "recovered"]
    return ("pending", recovered) if recovered else ("missing", [])


def _positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0
