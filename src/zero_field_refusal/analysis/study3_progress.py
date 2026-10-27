"""Study 3's state machine: from its records to where the study is and what comes next.

docs/study3-preregistration.md: "Seeds and order", "Failure routes" (and "If a requirement is not
met"), "Candidates and rate search" ("Search", "Scientific repair", "Choice of candidate"),
"Numerical qualification", "Validation, confirmation and stopping", "Simulation before
validation" and amendments 1–3. The inputs are what the readers give: the endpoints of
``study3_records.read_endpoints`` and its refusals, the holds of ``read_holds``, the numerical
qualification of ``qualification_records.read_qualification``, and the decision records of
``study3_decisions`` (``decisions`` and ``closures``). Nothing here reads a response, a harm label
or a judge output: the result holds numbers, ids, paths and hashes only.

- ``advance``: the pure core, over records already read (``Records``): the stage, the next step,
  the active arms and comparisons, the closures, the problems and, for a terminal stage, the
  outcome; and ``prescribed``, the cells the route has issued so far, per step key.
- ``progress``: the thin reader. It reads the records under RUNS_DIR and calls ``advance``.
- ``confirmation_state``: the confirmation's blocks, counters, cohorts, stop and readings; the
  campaign driver's between-block guard calls it.

**The result.** ``stage``: ``engineering``, ``simulation``, ``search``, ``numerical``, ``repair``,
``validation``, ``choice`` (validation passed, the ``n-c<N>`` record is to be written), ``n`` (an
``n-c<N>`` record that cannot be accepted), ``budget``, ``confirmation``, ``done``, or a terminal
outcome (``TERMINAL``, below). ``candidate``.
``step``, one of:
- ``campaign``: the campaign kind (``search-round0``, ``search-round1``, ``numerical``,
  ``repair``, ``validation``, ``confirmation``), its candidate, the header's step and round, and
  its cells (arm, rate, seed);
- ``decision``: the record needed (its kind and its path under RUNS_DIR), why, and for an
  ``n-c<N>``, ``budget-c<N>`` or closure record the fields the route fixes;
- ``pending``: what the step waits for (runs that have not ended, rates whose qualification has
  no record yet; at the confirmation, the blocks of those runs and each run's way out);
- ``blocked``: the reasons; ``technical`` when it waits for a diagnosis, with ``ways_out`` for a
  technical run outside the qualification (below: for such a mark the step also carries the
  ``reason`` id ``engineering_route_pending`` and its ``why``);
- ``report`` or ``done``: the outcome and its reason.
``outcome`` (a terminal stage), each with its ``reason`` id and the ``reasons`` that name why:
- ``scientific_feasibility_report``: A or F failed scientifically at candidate 2
  (``scientific_failure``);
- ``technical_feasibility_report``: a dead end in the code: a failed engineering record
  (``engineering_failed``), an unresolved technical mark in A or F: a held run closed
  (``held_run``), a run under another commit closed (its uncounted reason id,
  ``study3_versions.UNCOUNTED_REASONS``), a qualification technical after the candidate's
  engineering cycle (``unresolved_after_cycle``);
- ``design_stop``: the confirmation cannot start as designed: the simulation record failed (no
  level passed, or its verification failed: ``simulation_failed``), or the n record failed, its
  cap beyond the reserved confirmation seeds (``cap_beyond_reserved_seeds``);
- ``budget_stop``: the budget record's decision is stop (``budget_stop``);
- at ``done``, ``confirmed`` (A–F's reading is confirmatory) or ``descriptive``, with the
  readings of every active comparison (``readings``), its reason the stop's (``n_reached``,
  ``cap_reached``).
A chain of engineering cycles that is broken is a problem, never an outcome.
``active``: the arms and comparisons still open. ``rates``: the candidate's selected (or repaired)
rates. ``route``: the decisions taken on the way (a candidate left, P–F closed, the candidate
chosen). ``closures``: the closure records written. ``validation`` and ``confirmation``: the
stage's summary once reached. ``problems``: every problem found. ``not_counted``: every validation
or confirmation run that does not count for the commit it ran (below), wherever the walk stops.

**Order** ("Seeds and order"). The engineering record passed (the cells of the engineering stage,
every arm at candidate 1's starting rates on 10000–10003, are issued from the start), then the
simulation record passed (its validation size), then per candidate: the search, round 0 and,
where a median missed, round 1; the numerical qualification of the selected rates (per rate,
from ``read_qualification``); the scientific repair (amendment 3: one round per arm, after the
arm's qualification passed, its FA control included; a repaired rate's qualification as ``repair``
runs); the validation (28 or 40 blocks, as the simulation record says); the choice of the
candidate; the ``n-c<N>`` record; the ``budget-c<N>`` record when ``budget_reason`` asks for one
(or one is written); the confirmation; done. A missing decision record stops the route at the step
that needs it (a ``decision`` step, not a problem).

**A stage's decision**, the same at the search, the qualification, the repair, the validation
and (for its technical marks) the confirmation (the arms are the active ones; a terminal or route
decision is taken only on settled evidence, so that no later record of the stage can reverse it):
1. A or F with an unresolved technical mark ("After a technical mark"): a held run or a run under
   another commit with its closure record, or a qualification technical after the candidate's
   engineering cycle: the technical feasibility report, and candidate 2 is not taken ("the
   failure lies in the code"). A held run without its closure: the step names it.
2. P with such a mark: P–F closes (P with it, for the rest of the study), and A–F goes on.
3. A run technical, awaiting its diagnosis, in any active arm: the stage stays at its step,
   ``technical``, with the case named (in the qualification, the registered engineering cycle;
   outside it, the stop for the researcher's decision, below). A failed arm is failed whatever
   else, but the route waits: the diagnosis may end in the technical report.
4. A run that has not ended (``pending``, or ``recovered`` with no later attempt), or a rate whose
   qualification has no record yet: the step waits.
5. A or F failed scientifically: candidate 1 moves to candidate 2, candidate 2 ends the study
   with the scientific feasibility report. P failed at the same stage closes with it.
6. P failed scientifically: P–F closes, and A–F goes on.
7. Cells to issue: the campaign that issues them (round 0 before round 1).

**Endpoint states** (``study3_records``): pending stays pending; recovered waits for its next
attempt; technical goes to diagnosis; scientific is the arm's failure at the search (an
instability stop); a K-only run that is not usable is technical at the search (``study3_search``),
a failed endpoint in a block removes only its arm's support (``study3_blocks``).

**The numerical qualification** ("Numerical qualification", "Failure routes"). A rate's state is
``read_qualification``'s, recomputed with ``rate_verdict`` under the configs of this checkout (a
changed hashed key locates nothing) and under the GPU code in force for the candidate's runs
(below, "Code agreement": the chain's code after the candidate's own cycle, or before it), so
that a later candidate's cycle leaves an earlier candidate's reading as it was. ``qualified``
passes; ``failing`` is a scientific failure; ``incomplete`` waits; ``not_qualified`` is
technical, except an FA rate whose same-path control has no record yet and that has no other
technical mark (it waits for its control). Technical before the candidate's supersession record
(``c<N>/supersede-<n>.json``, the engineering cycle; one per candidate) is an engineering cycle in
progress: the stage stays at ``numerical`` (or ``repair``), technical. Technical after it is
unresolved after its cycle. A launcher run that has ended and that the reading does not locate is
a problem (loud, never a wait). A cell with no attempt is issued; a cell whose attempts were all
superseded waits for its next attempt (the launcher's recovery round).

**The repair** (amendment 3). ``repair_state`` gets the active arms only, and the route reads each
arm's status and each repaired rate's ``second`` round, never ``next`` alone: a repair run that
ended technical or held is a technical blocker (held: the hold route), never a finished repair;
the repaired rate's own qualification decides it (``qualified``: it replaces the selected rate for
validation). An impossible repair keeps the old rate.

**Validation** ("Validation"). Every cell of a block (six, or four once P is closed) is issued
with the block, and a block with a never-issued cell is not resolved. A failed endpoint
(scientific, gross collapse) removes only its arm's support in that block; a technical endpoint
blocks the aggregate until its diagnosis; the candidate moves on only after every block of the 28
or 40 is resolved. Then the gates: A–F passes with at least 22 of 28 (31 of 40) blocks supported
for A and F and its utility bounds passed (α = 0.05 per bound); P–F likewise for P and F. A failed
A–F gate is the candidate's failure; a failed P–F gate closes P–F, whether or not A–F passed.

**The n record** ("Sample size from validation"). It is accepted only after the engineering
record passed, for the chosen candidate, with its rates the validated ones, its comparisons the
active ones, its cohort's seeds the validation blocks supported for A and F, and its cohort's run
ids exactly the A and F endpoints of those blocks (A and F at ``lr_l`` and ``lr_u``: four per
block), the counting runs the route itself read for their support: as a set, none missing, none
extra (a P run, a run of another block or of another candidate), none twice; a problem names the
counts, never every id. A failed n record (its cap beyond the 200 reserved confirmation seeds)
means the confirmation cannot start: the design stop, with the reason; N is never cut. A power
shortfall needs the budget record (``budget_reason``); a budget decision to stop ends the study
before the confirmation (the budget stop). Whether the simulation's J is the one its fixation
froze is checked by the simulation package (package 3), which writes both; it is not checked
here.

**The confirmation** ("Confirmation"). One block per campaign, in seed order from 10400, up to the
cap N + max(10, ⌈0.12·N⌉) (the cells of every seed up to the cap are prescribed, no further). The
next block is issued while ``confirmation_state`` allows it; a started block with a never-issued
cell is completed first. A run of a started block that has not ended (``pending``, or
``recovered`` with no later attempt) holds the confirmation: the stop is checked between blocks,
and the counters wait for that run, so the step waits (``pending``), naming the blocks, the runs
and each run's way out once its launcher stopped without ending it (its recovery round, from its
own block file), and ``confirmation_state`` lets no block start. A technical mark in a started
block holds the confirmation (the step is blocked, and ``confirmation_state`` lets no block start)
until its way out: a recovery round where the launcher would open one, or the researcher's
decision (below, "Technical marks outside the qualification"); a run under another commit's
closure record routes as any technical mark's (A or F: the technical feasibility report; P: P–F
closes and the blocks go on with A and F). Once stopped, every comparison's reading needs its
blocks resolved. The confirmation never moves the candidate: its failed endpoints (a scientific
failure) remove support only, and P is closed in it only by a technical mark.

**A run under another commit** ("Voiding and supersession", Versions: a validation or confirmation
run under a commit other than the pinned one does not count; it is kept and reported). The reader
compares each such run's own commit (its ``meta.json``'s ``git``) with the commit the versions
record its header names pins (``study3_versions.uncounted_runs``: the pinned commit or a
record-only descendant of it, from a clean checkout). A run that does not count is kept, excluded
from every count and listed in ``not_counted`` with its reason id and the commit it ran, wherever
the walk stops. It is a technical mark outside the qualification of its arm, read before anything
else of its stage (the candidate's validation or confirmation), but no case for an engineering
cycle: a run launched from the wrong checkout needs no code fix. Its ways out:
- a recovery round of its label, only where the launcher would open one (no published outcome:
  the launcher skips a completed run), launched after ``study3_summary.py check`` passes, from a
  checkout of the pinned commit: its next attempt decides;
- its closure record (``closure-c<N>-<arm>``), citing the run's last ``end`` line, or its start
  line when the launcher never ended it, with the run's uncounted reason id as its reason.
Until one is taken, the step is ``blocked``, ``technical``, naming each run, its reason id and its
commit, with the closure record's fields exactly as the closure schema takes them. Once the
closure is valid (it names the run, with its reason), the route is "After a technical mark": in
A or F, the technical feasibility report, its reason the run's; in P alone, P–F closes and the
stage goes on with A and F. A published result is never rerun: no cell is issued again for it.

**The hold route** (a held launcher run: an outcome the candidate's engineering cycle could not
resolve, kept as it is and never attempted again). It is unresolved technical evidence, neither a
scientific failure nor a permission to rerun, and the route shows it as a terminal technical
outcome, never as a pending step: held in A or F, the stage is the technical feasibility report
(candidate 2 is not taken); held in P alone, P–F closes and A–F goes on. A cell with only a held
attempt is unresolved technical. Until the closure record exists (``closure-c<N>-<arm>``,
``study3_decisions``), the step names the closure record to write, with the candidate, the arm,
rate and seed, the run id, its held line (``line``, kind ``held``) and the cycle record with their
hashes (from ``read_holds``), exactly as the closure schema takes them, and beside them
(``hold_reasons``) the ids of every reason the held line states. Once a closure
record exists and is valid, the outcome follows its decision; the hold and its evidence stay as
they are. A hold that no supersession record or no journal line names cannot be closed, and
blocks.

**Technical marks outside the qualification** ("Failure routes", "Technical"): every run of the
search, the validation or the confirmation that the readers count as unresolved and technical, in
any arm: an endpoint ``technical``; at the search, a K-only run that is not usable or has no
positive K; in a block, a usable run that lacks its K or a utility loss, or a run the block
cannot place; in the confirmation, every run its blocks list as unresolved. Its registered route
is one engineering cycle at once (diagnosis, fix, re-qualification of every affected case, one
cycle per candidate); the engineering cycle of ``qualify.py`` is the qualification's, and the
route for a run outside it is not implemented in this package. Until it is, nothing cancels it
silently: such a mark stops the route for the researcher's decision. The step is ``blocked``,
``technical``, with the reason id ``engineering_route_pending`` (``reason``) and its ``why`` ("the
registered engineering cycle for a technical mark outside the qualification is not implemented
yet: the researcher decides"), naming each run (its id, label, arm and step). Its ways out
(``ways_out``): the researcher's decision (``{"decision": "engineering_route_pending"}``), beside a
recovery round of its label with its documented cause (the registered infrastructure route: the
attempt is ``recovered`` and its next attempt decides), only where the launcher would open one
(``launch.run_campaign``: no published terminal record, a ``meta.json`` that records the run
neither diverged nor completed, and its cell's round, or its own, not yet spent), for a validation
or confirmation run launched after ``study3_summary.py check`` passes. No closure is offered for
such a mark, nothing is voided, no outcome is reached and no cell is issued (the confirmation's
guard and its writer refuse a new block). A closure record that names such a mark is not accepted:
a problem that blocks ("a closure of a technical mark outside the qualification is not accepted
until its engineering route exists"), and it decides nothing.

A validation or confirmation run under another commit (above) is a technical mark of its arm too,
read before the rest of its stage, but no cycle case: launched from the wrong checkout, it needs no
code fix. Its ways out are a recovery round where the launcher would open one and its closure
record, citing the run's last ``end`` line whatever its outcome, or its start line when the
launcher never ended it (kind ``end`` or ``start``, no cycle; the step gives its fields exactly as
the closure schema takes them, and the closure must carry the reason they give, the run's
uncounted reason). Its route is "After a technical mark". The hold route is unchanged.

Two attempts that count for one cell, a refused run or a state no reader defines are no technical
mark but a problem of the records' integrity: they block, with no way out. A closure record that
names no mark the route met, and no held run of its route, is a problem, unless the walk stopped
before the stage of the run it names (it is then judged once the walk reaches it). Run records the
reader cannot read at all (a run id longer than a file name, a run directory it may not search)
block the step: "the run records cannot be read".

**Problems.** Nothing is decided on problems. While ``read_endpoints`` or ``read_holds`` reports a
problem, the step is ``blocked`` with them, and the stage is the last one the decision records
establish (engineering, simulation, or the search, where the route starts to rest on the run
records); the qualification's problems block from the numerical stage on; an invalid decision
record blocks the stage that rests on it, and a record rests on the ones before it (an invalid
simulation record invalidates the n records, an invalid n record its budget record). A search
start whose campaign header does not name the engineering record by its path and its sha256 now
(``engineering_record`` and ``engineering_record_sha256``, the flat form ``immutable`` protects:
that shows the record existed before the search started), an n or budget record the route has not
reached, and a closure record that names no mark of its route are problems too.

**Code agreement** (the chain of engineering cycles, ``code_chain``). A GPU code is admissible
when it lies on the chain: it starts at the engineering record's ``code.gpu_code_sha256``, and
each supersession record (``qualification/c<N>/supersede-<n>.json`` as
``qualification_records.supersessions`` lists it, read alone: it does not depend on the expected
code; candidate 1's before candidate 2's) moves it from its old code to its fix's. A supersession
whose old code is not the chain's is a break, and so is a second record for a candidate (one
cycle per candidate; ``qualification_records`` voids nothing by either, nor by a record that is not
the candidate's or does not cite its technical record), and ``expected_code`` must be the chain's
end: each is a problem. The reader builds the chain first, then reads each candidate's
qualification under the code in force for its runs. Per record: the engineering record is the
chain's start (it stays valid across cycles: a cycle's re-qualification certifies its fix); the
simulation record and the closure records may carry any code on the chain (the simulation runs
no GPU code; a closure is a record about a run, bound to its line's and its cycle's hashes, so
one written before its candidate's cycle stays valid); ``n-c<N>`` and ``budget-c<N>`` carry the
code in force for candidate N's runs. A record of another code is invalid. The records' commits
may differ (a change of analysis code moves no run).

**The cells** (``prescribed``). ``read_endpoints`` binds every run to a cell the route issued, so
the reader first derives the cells: it reads the runs under every cell they bind, runs the core
leniently (problems set aside, refused runs left out: only the rule decides which cells it issues
next), then reads the runs under exactly those cells and runs the core strictly, all three
readings from one snapshot (``study3_records.EndpointReader``: each run file is read once). A
run of a cell the rule never issued is then refused by name, and blocks.

**Rulings** where the registration leaves a detail open, fixed here before any study-3 record:
- *A failed engineering or simulation record* is final (a record is written once): the technical
  feasibility report, or the design stop. A failed simulation record means the confirmation cannot
  start ("If no level passes, or the verification fails"), so nothing after it is run.
- *P closed* is closed for the rest of the study: P–F ends ("only P–F ends"), and candidate 2
  issues no P cell.
- *Waiting before routing*: a stage's route decision waits for its runs to end and for its
  technical marks' diagnoses, in every active arm (rule 3 and 4 above), so that a decision once
  shown cannot be reversed by a record of that stage written later.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ..config import load_config
from ..paths import REPO_ROOT
from ..qualification_constants import FAILING, INCOMPLETE, NOT_QUALIFIED, QUALIFIED, TECHNICAL
from .study3_blocks import (
    OPEN,
    PAIRS,
    SUPPORTED,
    UNRESOLVED,
    VALIDATION_ALPHA,
    block_state,
    utility_gate,
)
from .study3_decisions import (
    DIRECTORY,
    START_REASON,
    SUPPORTED_AT,
    budget_reason,
    cap_of,
    closures,
    decision_path,
    decisions,
    end_lines,
    end_reason,
    hold_reasons,
    invalidate,
    kind_of,
    start_lines,
)
from .study3_records import (
    COUNTING,
    EXPERIMENT,
    RATE_TOLERANCE,
    SEEDS,
    STATES,
    Endpoint,
    EndpointReader,
    _read_journal,
    journal_lines,
    plain_name,
    read_holds,
    read_theta0,
    step_key,
    under_sealed,
)
from .study3_search import ARM_ORDER, CANDIDATES, LABELS, repair_state, search_state

# study 2's θ₀ evaluation, the utility losses' baseline, under RUNS_DIR
THETA0_RUN = Path("study2_stage_a") / "20261014T192927Z-theta0-s2000-abb703a1-a1"
STAGES = ("engineering", "simulation", "search", "numerical", "repair", "validation", "choice",
          "n", "budget", "confirmation", "done")  # fmt: skip
SCIENTIFIC_REPORT = "scientific_feasibility_report"  # no K* where A and F hold, at candidate 2
TECHNICAL_REPORT = "technical_feasibility_report"  # a dead end in the code
DESIGN_STOP = "design_stop"  # the confirmation cannot start: simulation failed, cap beyond seeds
BUDGET_STOP = "budget_stop"  # the researcher's budget decision is stop
TERMINAL = (SCIENTIFIC_REPORT, TECHNICAL_REPORT, DESIGN_STOP, BUDGET_STOP)
CONFIRMED, DESCRIPTIVE = "confirmed", "descriptive"  # the outcome at done: A–F's reading
CONFIRMATORY = "confirmatory"
CONFIRMATORY_AT = 36  # "Confirmation": at the cap, confirmatory with at least 36 supported blocks
COMPARISONS = {"A-F": "AF", "P-F": "PF"}  # the n record's names of the comparisons, study3_blocks'
ACTIVE = (["A-F"], ["A-F", "P-F"])
CAMPAIGNS = ("search-round0", "search-round1", "numerical", "repair", "validation",
             "confirmation")  # the campaign kinds, in the order a step issues them  # fmt: skip
ROUTED = ("search", "numerical", "repair", "validation", "confirmation")  # a mark routes there
OUTSIDE = ("search", "validation", "confirmation")  # the steps outside the qualification
ENDED = ("usable", "unusable", "scientific")  # a run whose records are published and read
# A technical mark outside the qualification: its registered engineering cycle is not implemented
# here, so the route stops for the researcher's decision (module docstring)
ENGINEERING_ROUTE_PENDING = "engineering_route_pending"
ENGINEERING_ROUTE_WHY = (
    "the registered engineering cycle for a technical mark outside the "
    "qualification is not implemented yet: the researcher decides"
)
NOT_ACCEPTED = ("a closure of a technical mark outside the qualification is not accepted until "
                "its engineering route exists")  # fmt: skip


@dataclass(frozen=True)
class Records:
    """What the readers give, the input of ``advance``: the endpoints and refusals of
    ``read_endpoints``, the holds and problems of ``read_holds``, per candidate the entries of
    ``study3_decisions.decisions`` and the reading of ``read_qualification``, the entries of
    ``study3_decisions.closures`` by kind, each candidate's engineering-cycle records
    (``cycles``: ``qualification_records.supersessions``), each run's campaign header from its
    start in the launch journal (``headers``), the sha256 of each valid decision record's file
    now (``digests``), each run's last end line and its start line
    (``study3_decisions.end_lines``, ``start_lines``), the runs the launcher would open a
    recovery round for (``recoverable``) and the validation and confirmation runs that do not
    count for the commit they ran (``uncounted``: ``study3_versions.uncounted_runs``)."""

    endpoints: Sequence[Endpoint] = ()
    problems: Sequence[str] = ()
    holds: Sequence[Mapping[str, Any]] = ()
    hold_problems: Sequence[str] = ()
    decisions: Mapping[int, Mapping[str, Mapping[str, Any]]] = field(default_factory=dict)
    closures: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    qualification: Mapping[int, Mapping[str, Any]] = field(default_factory=dict)
    headers: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    digests: Mapping[str, str] = field(default_factory=dict)
    ends: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    starts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    cycles: Mapping[int, Sequence[Mapping[str, Any]]] = field(default_factory=dict)
    recoverable: Collection[str] = frozenset()
    uncounted: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


# ------------------------------------------------------------------ the reader


def progress(
    runs_root: str | Path,
    *,
    repo: str | Path = REPO_ROOT,
    expected_code: str,
    theta0: str | Path | None = None,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The state of study 3 under ``runs_root`` (module docstring). ``expected_code``: the GPU
    code's sha256 the qualification and every decision record must carry; ``theta0``: θ₀'s
    evaluation (study 2's, ``THETA0_RUN`` under ``runs_root``, by default); ``settings``:
    configs/experiments/qualification.yaml as loaded (by default, this checkout's); ``repo``: the
    checkout the decision records' ``repo`` evidence is relative to."""
    if not (isinstance(expected_code, str) and expected_code):
        raise ValueError(f"expected_code is the GPU code's sha256, not {expected_code!r}")
    from . import qualification_records  # it loads torch (qualification_verdicts): here only

    root, repo = Path(runs_root), Path(repo)
    theta0 = root / THETA0_RUN if theta0 is None else Path(theta0)
    settings = load_config("experiments/qualification") if settings is None else settings
    holds, hold_problems = read_holds(root)
    entries = {c: decisions(root, c, repo=repo) for c in (1, 2)}
    found = {k: v for c in (1, 2) for k, v in closures(root, c, repo=repo).items()}
    valid = {k for found_ in (*entries.values(), found) for k, e in found_.items()
             if e["state"] in ("passed", "failed")}  # read already: no sealed path  # fmt: skip
    cycles = {c: qualification_records.supersessions(root, c) for c in (1, 2)}
    # each candidate's qualification under the code in force for its runs (the chain's)
    in_force = code_chain(_engineering_code(entries[1].get("engineering") or {}), cycles)[1]
    codes = {c: code if isinstance(code, str) and code else expected_code
             for c, code in in_force.items()}  # fmt: skip
    logs, _ = journal_lines(root)
    fixed = {
        "holds": tuple(holds), "hold_problems": tuple(hold_problems),
        "decisions": entries, "closures": found, "cycles": cycles,
        "qualification": {c: qualification_records.read_qualification(
            root, c, expected_code=codes.get(c, expected_code), settings=settings)
            for c in (1, 2)},
        "headers": _headers(root),
        "digests": {k: _record_digest(root, k) for k in valid},
        "ends": end_lines(root, logs), "starts": start_lines(root, logs),
    }  # fmt: skip
    try:
        read_theta0(theta0)
    except ValueError as exc:
        why = (f"{exc}: θ₀'s evaluation is the utility losses' baseline, and no run record is "
               "read without it")  # fmt: skip
        return advance(Records(problems=(why,), **fixed), expected_code=expected_code)
    try:  # one snapshot of the run records, read once whatever cells it is read under
        reader = EndpointReader(root, theta0=theta0)
    except (OSError, ValueError) as exc:
        why = f"{root}: the run records cannot be read ({type(exc).__name__})"
        return advance(Records(problems=(why,), **fixed), expected_code=expected_code)
    try:
        every = _bound(reader.read({})[0])  # every cell a run binds: each outcome is read
        endpoints, problems = reader.read(every)
        derived = advance(Records(tuple(endpoints), tuple(problems), **fixed),
                          expected_code=expected_code, lenient=True)["prescribed"]  # fmt: skip
        endpoints, problems = reader.read(derived)
        recoverable = _recoverable(reader, endpoints)
    except (OSError, ValueError) as exc:
        why = f"{root}: the run records cannot be read ({type(exc).__name__})"
        return advance(Records(problems=(why,), **fixed), expected_code=expected_code)
    from .study3_versions import uncounted_runs  # each run's own commit, against the checkout's

    uncounted = uncounted_runs(endpoints, repo=repo)
    return advance(Records(tuple(endpoints), tuple(problems), recoverable=recoverable,
                           uncounted=uncounted, **fixed), expected_code=expected_code)  # fmt: skip


def _recoverable(reader: EndpointReader, endpoints: Iterable[Endpoint]) -> frozenset[str]:
    """The runs the launcher would open a recovery round for (``launch.run_campaign``): a run
    with no published terminal record whose ``meta.json`` records it neither diverged nor
    completed (the launcher reads those as outcomes, never reruns), its cell's round not yet
    spent (once per cell, ``launch.recovered_cells``) nor its own (``recovered_runs``). A refused
    attempt, or one whose run id is no plain name, binds no run: it is skipped before anything of
    it is read."""
    cells, runs = reader.recovery_rounds()
    found = set()
    for e in endpoints:
        if e.state == "refused" or not plain_name(e.run_id):
            continue
        if e.label is None or "/s" not in e.label:
            continue
        cell = e.label.rsplit("/s", 1)[0]
        if ((EXPERIMENT, cell) in cells or (EXPERIMENT, e.label) in runs
                or (e.run_dir / "terminal.json").is_file()
                or reader.meta_status(e.run_dir) in ("diverged", "completed")):  # fmt: skip
            continue
        found.add(e.run_id)
    return frozenset(found)


def _engineering_code(entry: Mapping[str, Any]) -> str | None:
    """The GPU code of a valid engineering record: the chain's start."""
    record = entry.get("record") if entry.get("state") in ("passed", "failed") else None
    return ((record or {}).get("code") or {}).get("gpu_code_sha256")


def code_chain(
    start: str | None, cycles: Mapping[int, Sequence[Mapping[str, Any]]]
) -> tuple[list[str | None], dict[int, str | None], list[str]]:
    """The chain of engineering cycles (module docstring, "Code agreement"): from ``start`` (the
    engineering record's GPU code), each candidate's supersession records (candidate 1's, then
    candidate 2's, as ``qualification_records.supersessions`` lists them) move it from their old
    code to their fix's. Returns the codes on the chain, the code in force for each candidate's
    runs (the chain's after its own cycle, or before it), and the breaks: a record whose old
    code is not the chain's, and more than one record for a candidate (one engineering cycle per
    candidate: ``qualify.py supersede`` writes ``c<N>/supersede-1.json`` and refuses a second).
    Without a start there is no chain: no code, and none in force."""
    if start is None:
        return [], {}, []
    codes, in_force, breaks, current = [start], {}, [], start
    for candidate in (1, 2):
        found = list(cycles.get(candidate) or [])
        if len(found) > 1:
            breaks.append(
                f"candidate {candidate} has {len(found)} supersession records "
                f"({', '.join(str(c.get('path')) for c in found)}): one engineering cycle per "
                "candidate, a break in the chain of engineering cycles"
            )
        for cycle in found:
            old, fix = cycle.get("old_gpu_code"), cycle.get("fix_gpu_code")
            if old != current:
                breaks.append(
                    f"{cycle.get('path')}: its old GPU code {old} is not the chain's "
                    f"{current}: a break in the chain of engineering cycles"
                )
            current = fix
            codes.append(fix)
        in_force[candidate] = current
    return codes, in_force, breaks


def _bound(endpoints: Iterable[Endpoint]) -> dict[str, set[tuple[str, float, int]]]:
    """The cells the runs bind (step key, arm, rate, seed), whether or not they were issued."""
    out: dict[str, set[tuple[str, float, int]]] = defaultdict(set)
    for e in endpoints:
        if e.cell.key is not None and None not in (e.arm, e.lr, e.seed):
            out[e.cell.key].add((e.arm, e.lr, e.seed))
    return out


def _record_digest(root: Path, kind: str) -> str | None:
    """The sha256 of a valid decision record, the sealed-path rule applied again where it is read:
    None for one that resolves under a sealed folder now (never opened; a header that names it then
    matches nothing)."""
    path = decision_path(root, kind)
    return None if under_sealed(path, root) else hashlib.sha256(path.read_bytes()).hexdigest()


def _headers(root: Path) -> dict[str, Mapping[str, Any]]:
    """Each start's campaign header in the launch journal, by run id (``study3_records``'
    reading of it); none when the journal cannot be read (``read_endpoints`` and ``read_holds``
    say why)."""
    try:
        journal = _read_journal(root, {})
    except (OSError, ValueError):
        return {}
    return {start["run_id"]: start["header"] for _, start in journal.starts
            if isinstance(start.get("run_id"), str)
            and isinstance(start.get("header"), Mapping)}  # fmt: skip


# ------------------------------------------------------------------ the core


def advance(records: Records, *, expected_code: str, lenient: bool = False) -> dict[str, Any]:
    """The route over records already read (module docstring). ``lenient``: problems are set
    aside and refused runs left out, so that ``prescribed`` holds every cell the rule issues from
    the records (the reader's derivation of the cells); its stage and step decide nothing."""
    return _Walk(records, expected_code, lenient).walk()


class _Stop(Exception):
    """The route's end: a step, a report or done."""

    def __init__(self, result: dict[str, Any]) -> None:
        super().__init__("stop")
        self.result = result


class _Move(Exception):
    """A or F failed scientifically: the next candidate."""

    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__("move")
        self.event = event


@dataclass
class _Arm:
    """What an arm's records say at a stage (module docstring, "A stage's decision")."""

    hold: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)  # technical after the cycle
    technical: list[str] = field(default_factory=list)  # awaiting its diagnosis
    pending: list[str] = field(default_factory=list)
    scientific: list[str] = field(default_factory=list)
    issue: list[tuple[str, str, int | None, tuple[str, float, int]]] = field(default_factory=list)
    closable: list[dict[str, Any]] = field(default_factory=list)  # technical outside the qual.


class _Walk:
    """One walk of the route over the records."""

    def __init__(self, records: Records, expected_code: str, lenient: bool) -> None:
        self.records, self.expected, self.lenient = records, expected_code, lenient
        self.endpoints = [e for e in records.endpoints if not (lenient and e.state == "refused")]
        self.by_key: dict[str | None, list[Endpoint]] = defaultdict(list)
        self.by_seed: dict[int | None, list[Endpoint]] = defaultdict(list)
        self.by_id: dict[str, Endpoint] = {}
        for e in self.endpoints:
            self.by_key[e.cell.key].append(e)
            self.by_seed[e.seed].append(e)
            self.by_id[e.run_id] = e
        self.prescribed: dict[str, set[tuple[str, float, int]]] = defaultdict(set)
        self.route: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self.arms: list[str] = list(ARM_ORDER)
        self.candidate = 1
        self.rates: dict[str, dict[str, float]] = {}
        self.extra: dict[str, Any] = {}
        self.consumed: set[str] = set()
        self.marks: set[str] = set()  # the technical marks outside the qualification met
        self.pending_route: set[str] = set()  # those of them whose route is pending (no closure)
        self.reached: set[tuple[int, str]] = set()  # (candidate, stage) whose marks were read
        self.blocks = 0
        self.codes: list[str] = []  # the chain of GPU codes, from the engineering record's
        self.in_force: dict[int, str] = {}  # candidate -> the code its runs are under
        self.chain_problems: list[str] = []
        self.entries = self._entries()
        self.placed, self.unplaced = self._place_holds()

    # -- the walk

    def walk(self) -> dict[str, Any]:
        try:
            self._walk()
        except _Stop as stop:
            return self._final(stop.result)
        raise AssertionError("every route ends with a step, a report or done")  # pragma: no cover

    def _walk(self) -> None:
        start = CANDIDATES[1]["start"]
        self._prescribe(1, "engineering", None, [
            (arm, start[arm][label], seed) for arm in ARM_ORDER for label in LABELS
            for seed in SEEDS["engineering"][1]])  # fmt: skip
        engineering, simulation = self.entry("engineering"), self.entry("simulation")
        found = [
            *self.records.problems,
            *self.records.hold_problems,
            *self.unplaced,
            *self.chain_problems,
        ]
        if found:  # the stage the decision records establish, where the run records come in
            stage = ("engineering" if engineering["state"] != "passed" else
                     "simulation" if simulation["state"] != "passed" else "search")  # fmt: skip
            # an invalid engineering or simulation record (a cause they may have) is named too
            gates = [p for e in (engineering, simulation) if e["state"] == "invalid"
                     for p in e["problems"]]  # fmt: skip
            self._block_on(stage, [*found, *gates])
        self._gate("engineering", engineering, "the engineering stage (its dry run, the bridge, "
                   "θ₀'s utility and the stage-0 check) passed before any selection")  # fmt: skip
        self._gate("simulation", simulation, "the simulation's level L and validation size, "
                   "before the search")  # fmt: skip
        self.blocks = simulation["record"]["validation_blocks"]
        self._search_after_engineering()
        for candidate in (1, 2):
            self.candidate = candidate
            self.rates, self.extra = {}, {}
            try:
                self._candidate(candidate)
            except _Move as move:
                self.route.append(move.event)
                if candidate == 2:
                    raise _Stop(self._report(SCIENTIFIC_REPORT, "scientific_failure", [
                        f"candidate 1 and candidate 2 failed: candidate 2 at its "
                        f"{move.event['stage']}: {move.event['why']}"])) from None  # fmt: skip

    def _candidate(self, candidate: int) -> None:
        selected = self._search(candidate)
        self._numerical(candidate, selected)
        final = self._repair(candidate, selected)
        validated = self._validation(candidate, final)
        n = self._choice(candidate, final, validated)
        self._confirmation(candidate, final, n)

    # -- engineering and the simulation

    def _gate(self, kind: str, entry: Mapping[str, Any], why: str) -> None:
        """A record that gates the route: invalid blocks, missing is the step, failed is
        final."""
        if entry["state"] == "invalid":
            raise _Stop(self._blocked(kind, entry["problems"]))
        if entry["state"] == "missing":
            raise _Stop(self._result(kind, self._decision(kind, why)))
        if entry["state"] == "failed" and kind == "simulation":
            raise _Stop(self._report(DESIGN_STOP, "simulation_failed", [
                "simulation: the record failed: no level passed the coverage screen or its "
                "verification failed, and the confirmation cannot start"]))  # fmt: skip
        if entry["state"] == "failed":
            raise _Stop(
                self._report(
                    TECHNICAL_REPORT,
                    "engineering_failed",
                    ["engineering: the record failed: the engineering stage did not pass"],
                )
            )

    def _search_after_engineering(self) -> None:
        """The engineering record existed before the search started: every search start's
        campaign header names it by its path and its sha256 now (``engineering_record``,
        ``engineering_record_sha256``; ``immutable`` keeps the record as named)."""
        digest = self.records.digests.get("engineering")
        wanted = (*DIRECTORY, "engineering.json")
        found = []
        for e in self.endpoints:
            if e.cell.step != "search" or e.cell.campaign is None:
                continue
            header = self.records.headers.get(e.run_id) or {}
            named = header.get("engineering_record")
            if not (isinstance(named, str)
                    and PurePosixPath(named.replace("\\", "/")).parts[-3:] == wanted
                    and header.get("engineering_record_sha256") == digest):  # fmt: skip
                found.append(f"{e.run_id}: its search campaign's header does not name the "
                             f"engineering record with its sha256 now ({digest}): the search "
                             "starts once the engineering record exists")  # fmt: skip
        self._block_on("search", found)

    # -- the search

    def _search(self, candidate: int) -> dict[str, dict[str, float]]:
        stage, spec = "search", CANDIDATES[candidate]
        seeds = SEEDS["search"][candidate]
        self._prescribe(candidate, "search", 0, [
            (arm, spec["start"][arm][label], seed) for arm in self.arms for label in LABELS
            for seed in seeds])  # fmt: skip
        state = search_state(self.endpoints, candidate, problems=())
        if state["next"] == "blocked":
            raise _Stop(self._blocked(stage, state["problems"]))
        for arm in self.arms:
            for label in LABELS:
                decision = state["arms"][arm]["targets"][label]["decision"]
                if decision not in (None, "kept"):
                    self._prescribe(candidate, "search", 1, [(arm, decision, s) for s in seeds])
        conditions = {}
        for arm in self.arms:
            found = state["arms"][arm]
            c = _Arm(hold=self._holds_at(candidate, arm, ("search",)))
            c.scientific = [e.run_id for e in found["failed"]]
            for target in found["targets"].values():
                for round_ in (0, 1):
                    rounds = target[f"round{round_}"]
                    if rounds is None:
                        continue
                    c.pending += [e.run_id for e in rounds["runs"] if e.seed in rounds["pending"]
                                  and e.state in ("pending", "recovered")]  # fmt: skip
                    c.issue += [
                        (f"search-round{round_}", "search", round_, (arm, rounds["rate"], seed))
                        for seed in rounds["missing"]
                    ]
            conditions[arm] = c
        self._block_on(stage, self._unresolved(stage, conditions, [
            e for arm in self.arms for e in state["arms"][arm]["technical"]]))  # fmt: skip
        self._decide(stage, conditions)
        return {arm: {label: state["arms"][arm]["targets"][label]["selected_rate"]
                      for label in LABELS} for arm in self.arms}  # fmt: skip

    # -- the numerical qualification and the repair

    def _numerical(self, candidate: int, selected: Mapping[str, Mapping[str, float]]) -> None:
        stage = "numerical"
        self.rates = {arm: dict(selected[arm]) for arm in self.arms}
        self._prescribe(candidate, "numerical", None, [
            (arm, selected[arm][label], seed) for arm in self.arms for label in LABELS
            for seed in SEEDS["numerical"][candidate]])  # fmt: skip
        reading = self._reading(stage, candidate)
        conditions, loud = {}, []
        for arm in self.arms:
            c = _Arm(hold=self._holds_at(candidate, arm, ("numerical",)))
            for label in LABELS:
                loud += self._qualified(c, candidate, "numerical", arm, selected[arm][label],
                                        reading)  # fmt: skip
            conditions[arm] = c
        self._block_on(stage, loud)
        self._decide(stage, conditions)

    def _repair(
        self, candidate: int, selected: Mapping[str, Mapping[str, float]]
    ) -> dict[str, dict[str, float]]:
        stage = "repair"
        seeds = SEEDS["numerical"][candidate]
        reading = self._reading(stage, candidate)
        qualified = {arm: {selected[arm][label]: (_rate(reading, arm, selected[arm][label]) or {})
                           .get("state") or "not reported" for label in LABELS}
                     for arm in self.arms}  # fmt: skip
        state = repair_state(self.endpoints, candidate, {a: selected[a] for a in self.arms},
                             qualified, problems=())  # fmt: skip
        if state["next"] == "blocked":
            raise _Stop(self._blocked(stage, state["problems"]))
        final = {arm: dict(selected[arm]) for arm in self.arms}
        conditions, loud = {}, []
        for arm in self.arms:
            found = state["arms"][arm]
            c = _Arm(hold=self._holds_at(candidate, arm, ("repair",)))
            if found["status"] in ("not_qualified", "unselected"):
                c.technical.append(f"{arm}: the repair's gate is not met ({found.get('why')})")
            elif found["status"] == "incomplete":
                runs = sorted({e.run_id for r in found["rates"].values()
                               for part in (r["search"], r["numerical"])
                               for e in (*part["technical"], *part["failed"])})  # fmt: skip
                listed = ", ".join(runs) or "a run with no usable endpoint"
                c.technical.append(f"{arm}: the median of a selected rate over its search and "
                                   f"numerical runs is missing ({listed})")  # fmt: skip
            for label, rate in found["rates"].items():
                if rate["action"] != "repair" or label not in found["repairs"]:
                    continue
                repaired = final[arm][label] = rate["repaired_rate"]
                self._prescribe(candidate, "repair", None, [(arm, repaired, s) for s in seeds])
                second = rate["second"]  # its qualification runs: technical is never finished
                c.technical += [f"{e.run_id}: {e.state}" for e in second["technical"]
                                if e.state != "held"]  # fmt: skip
                c.scientific += [e.run_id for e in second["failed"]]
                loud += self._qualified(c, candidate, "repair", arm, repaired, reading)
            conditions[arm] = c
        self._block_on(stage, loud)
        self._decide(stage, conditions)
        self.rates = {arm: final[arm] for arm in self.arms}
        return {arm: dict(final[arm]) for arm in self.arms}

    def _reading(self, stage: str, candidate: int) -> Mapping[str, Any]:
        """The candidate's qualification reading; its problems block."""
        reading = self.records.qualification.get(candidate) or {}
        self._block_on(stage, list(reading.get("problems") or []))
        return reading

    def _qualified(self, c: _Arm, candidate: int, step: str, arm: str, rate: float,
                   reading: Mapping[str, Any]) -> list[str]:  # fmt: skip
        """A rate's qualification into ``c``: its launcher runs (issued, waiting) and its state
        (module docstring, "The numerical qualification"). Returns the problems: a launcher run
        that ended and is not located."""
        key = step_key(candidate, step)
        name = f"{key} {arm} {rate:g}"
        issued = True
        for seed in SEEDS["numerical"][candidate]:
            attempts = self._attempts(key, arm, rate, seed)
            counting = [e for e in attempts if e.state in COUNTING]
            if not attempts:
                c.issue.append((step, step, None, (arm, rate, seed)))
                issued = False
            elif any(e.state == "held" for e in attempts):
                continue  # the hold route
            elif any(e.state == "pending" for e in counting):
                c.pending += [e.run_id for e in counting if e.state == "pending"]
            elif not counting:  # recovered or superseded: the launcher's next attempt
                c.pending.append(f"{name} s{seed}: its next attempt")
        entry = _rate(reading, arm, rate)
        if entry is None:
            if issued:
                c.pending.append(f"{name}: its qualification has no record yet")
            return []
        condition, why = _condition(entry, reading)
        label = f"{entry.get('key') or name}: {entry.get('state')}{why}"
        if condition == "failing":
            c.scientific.append(label)
        elif condition == "technical":
            (c.unresolved if self.records.cycles.get(candidate) else c.technical).append(label)
        elif condition == "waiting":
            c.pending.append(label)
        return self._unlocated(entry, name)

    def _unlocated(self, entry: Mapping[str, Any], name: str) -> list[str]:
        """An issued launcher run whose latest attempt has ended and that the qualification's
        reading does not locate: under this checkout's configs its cell has another config hash
        (a hashed key changed), and that is loud, never a wait."""
        out = []
        for run in entry.get("runs") or []:
            ids = run.get("run_ids") or []
            latest = self.by_id.get(ids[-1]) if ids else None
            if run.get("located") is None and latest is not None and latest.state in ENDED:
                out.append(f"{name}: the launcher run {latest.run_id} has ended, and the "
                           "qualification read under this checkout's configs does not locate "
                           "it: the cell's config hash is not the one it ran under")  # fmt: skip
        return out

    # -- validation

    def _validation(self, candidate: int,
                    final: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:  # fmt: skip
        stage, count = "validation", self.blocks
        seeds = list(SEEDS["validation"][candidate][:count])
        self._prescribe(candidate, "validation", None, [
            (arm, final[arm][label], seed) for arm in self.arms for label in LABELS
            for seed in seeds])  # fmt: skip
        self._other_commits(stage, step_key(candidate, "validation"))
        # holds first: a closed P leaves the blocks
        self._decide(stage, {arm: _Arm(hold=self._holds_at(candidate, arm, ("validation",)))
                             for arm in self.arms})  # fmt: skip
        key = step_key(candidate, "validation")
        blocks = {seed: block_state(self._of_seed(seed), seed, candidate, self.arms)
                  for seed in seeds}  # fmt: skip
        conditions = {arm: _Arm() for arm in self.arms}
        unresolved: list[Endpoint] = []
        for seed, block in blocks.items():
            unresolved += self._placed(block, stage)
            for arm in self.arms:
                support, c = block["arms"][arm], conditions[arm]
                if support["state"] == UNRESOLVED:
                    unresolved += support["technical"]
                elif support["state"] == OPEN:
                    c.pending += [e.run_id for e in support["pending"]]
                    c.issue += [
                        ("validation", "validation", None, cell)
                        for cell in self._missing(key, arm, final[arm], seed)
                    ]
        self._block_on(stage, self._unresolved(stage, conditions, unresolved))
        self._decide(stage, conditions)
        minimum = SUPPORTED_AT[count]
        gates, short = {}, {}
        for name, pair in PAIRS.items():
            if not all(arm in self.arms for arm in pair):
                continue
            supported = [s for s, b in blocks.items() if b[name]["state"] == SUPPORTED]
            utility = utility_gate(list(blocks.values()), name, VALIDATION_ALPHA)
            gates[name] = {
                "supported": len(supported),
                "minimum": minimum,
                "utility": utility["passes"],
                "passes": len(supported) >= minimum and utility["passes"] is True,
            }
            short[name] = [
                arm
                for arm in pair
                if sum(b["arms"][arm]["state"] == SUPPORTED for b in blocks.values()) < minimum
                or not all(m["passes"] for m in utility["bounds"].get(arm, {}).values())
            ]
        self.extra["validation"] = {"blocks": count, "gates": gates}
        if "PF" in gates and not gates["PF"]["passes"]:
            self._close_p(
                stage,
                f"the P–F validation gate: {gates['PF']['supported']} of {count} "
                f"blocks supported ({minimum} needed), utility {gates['PF']['utility']}",
            )
        if not gates["AF"]["passes"]:
            raise _Move(self._event(stage, "candidate", short["AF"] or ["A", "F"],
                                    f"the A–F validation gate: {gates['AF']['supported']} of "
                                    f"{count} blocks supported ({minimum} needed), utility "
                                    f"{gates['AF']['utility']}"))  # fmt: skip
        supported = [s for s, b in blocks.items() if b["AF"]["state"] == SUPPORTED]
        run_ids = sorted(
            run.run_id
            for s in supported
            for arm in ("A", "F")
            for run in (blocks[s]["arms"][arm]["lo"], blocks[s]["arms"][arm]["hi"])
        )
        return {"supported": supported, "run_ids": run_ids}

    def _placed(self, block: Mapping[str, Any], stage: str) -> list[Endpoint]:
        """The runs a block cannot place (its ``blockers``): unresolved runs of the stage."""
        found = []
        for blocker in block["blockers"]:
            run_id = blocker.split(":", 1)[0]
            if run_id in self.by_id:
                found.append(self.by_id[run_id])
            else:
                self._block_on(stage, [f"{run_id}: a run block {block['seed']} cannot place and "
                                       "the readers do not list"])  # fmt: skip
        return found

    def _missing(self, key: str, arm: str, rates: Mapping[str, float],
                 seed: int) -> list[tuple[str, float, int]]:  # fmt: skip
        """The cells of an arm in a block that have no attempt that counts or waits (never
        issued, or every attempt superseded): to issue."""
        return [
            (arm, rates[label], seed)
            for label in LABELS
            if not any(
                e.state in COUNTING or e.state in ("recovered", "held", "refused")
                for e in self._attempts(key, arm, rates[label], seed)
            )
        ]

    # -- the choice, N and the budget

    def _choice(self, candidate: int, final: Mapping[str, Mapping[str, float]],
                validated: Mapping[str, Any]) -> Mapping[str, Any]:  # fmt: skip
        kind, budget_kind = f"n-c{candidate}", f"budget-c{candidate}"
        self.route.append(
            self._event(
                "choice",
                "chosen",
                list(self.arms),
                "the first candidate whose A and F passed their qualification and validation",
            )
        )
        fields = {"candidate": candidate,
                  "rates": {arm: {"lr_l": final[arm]["l"], "lr_u": final[arm]["u"]}
                            for arm in self.arms},
                  "active": self._comparisons(),
                  "cohort": {"seeds": list(validated["supported"]),
                             "run_ids": list(validated["run_ids"])}}  # fmt: skip
        entry = self.entry(kind)
        if entry["state"] == "missing":
            raise _Stop(self._result("choice", self._decision(
                kind, "validation's spread and N, the candidate, its rates and the judge sealed "
                "and validation's labels opened for the spread alone", fields)))  # fmt: skip
        self.consumed.add(kind)
        if entry["state"] == "invalid":
            raise _Stop(self._blocked("n", entry["problems"]))
        record = entry["record"]
        self._block_on("n", self._against_n(kind, record, fields))
        if record["state"] == "failed":
            raise _Stop(self._report(DESIGN_STOP, "cap_beyond_reserved_seeds", [
                f"{kind}: N = {record['N']} needs the cap {record['cap']}, beyond the "
                f"{len(SEEDS['confirmation'][candidate])} confirmation seeds reserved: the "
                "confirmation cannot start, and N is never cut"]))  # fmt: skip
        reason = budget_reason(record["N"], self.entry("simulation")["record"])
        budget = self.entry(budget_kind)
        if reason is None and budget["state"] == "missing":
            return record
        self.consumed.add(budget_kind)
        if budget["state"] == "invalid":
            raise _Stop(self._blocked("budget", budget["problems"]))
        if budget["state"] == "missing":
            raise _Stop(self._result("budget", self._decision(
                budget_kind, f"the researcher's decision on the extra blocks and their price: "
                f"{reason}", {"candidate": candidate, "N": record["N"], "cap": record["cap"],
                              "why": reason})))  # fmt: skip
        if budget["state"] == "failed":
            raise _Stop(self._report(BUDGET_STOP, "budget_stop", [
                f"{budget_kind}: the decision on N = {record['N']} and its cap {record['cap']} "
                "is stop: the confirmation does not start"]))  # fmt: skip
        return record

    def _against_n(self, kind: str, record: Mapping[str, Any],
                   fields: Mapping[str, Any]) -> list[str]:  # fmt: skip
        """An n record against the route (module docstring, "The n record")."""
        found = []
        if self.entry("engineering")["state"] != "passed":
            found.append(f"{kind}: an n record is accepted only once the engineering record "
                         "passed")  # fmt: skip
        if record.get("active") != fields["active"]:
            found.append(f"{kind}: its comparisons {record.get('active')} are not the active "
                         f"ones {fields['active']}")  # fmt: skip
        rates = record.get("rates") or {}
        if set(rates) != set(fields["rates"]) or not all(
                math.isclose(rates[arm][k], fields["rates"][arm][k], rel_tol=RATE_TOLERANCE)
                for arm in fields["rates"] for k in ("lr_l", "lr_u")):  # fmt: skip
            found.append(f"{kind}: its rates {rates} are not the validated ones "
                         f"{fields['rates']}")  # fmt: skip
        cohort = record.get("cohort") or {}
        seeds = list(cohort.get("seeds") or [])
        if seeds != fields["cohort"]["seeds"]:
            found.append(f"{kind}: its cohort's seeds are not the "
                         f"{len(fields['cohort']['seeds'])} validation blocks where A and F are "
                         "supported")  # fmt: skip
        given = cohort.get("run_ids")
        given = list(given) if isinstance(given, list) else []
        named = [r for r in given if isinstance(r, str)]
        expected = set(fields["cohort"]["run_ids"])
        missing, extra = expected - set(named), set(named) - expected
        twice = len(named) - len(set(named))
        if missing or extra or twice or len(named) != len(given):
            blocks = len(fields["cohort"]["seeds"])
            found.append(f"{kind}: its cohort's run ids are not exactly the A and F endpoints (A "
                         f"and F at lr_l and lr_u) of its {blocks} supported validation blocks, "
                         f"{len(expected)} runs: {len(missing)} missing, "
                         f"{len(extra) + len(given) - len(named)} extra, {twice} "
                         "duplicated")  # fmt: skip
        return found

    # -- the confirmation

    def _confirmation(self, candidate: int, final: Mapping[str, Mapping[str, float]],
                      n: Mapping[str, Any]) -> None:  # fmt: skip
        stage = "confirmation"
        seeds = list(SEEDS["confirmation"][candidate][: n["cap"]])
        self._prescribe(candidate, "confirmation", None, [
            (arm, final[arm][label], seed) for arm in self.arms for label in LABELS
            for seed in seeds])  # fmt: skip
        self._other_commits(stage, step_key(candidate, "confirmation"))
        while True:  # once more when a closure closed P: the blocks are A–F's alone
            arms = list(self.arms)
            state = confirmation_state(self.endpoints, candidate, n["N"], self._comparisons())
            self.extra["confirmation"] = state
            self._block_on(stage, state["problems"])
            conditions = {arm: _Arm(hold=self._holds_at(candidate, arm, ("confirmation",)))
                          for arm in self.arms}  # fmt: skip
            listed = sorted({run for block in state["technical"] for run in block["runs"]})
            unknown = [f"{run_id}: a run the confirmation's blocks list and the readers do not"
                       for run_id in listed if run_id not in self.by_id]  # fmt: skip
            found = [self.by_id[r] for r in listed if r in self.by_id]
            self._block_on(stage, [*unknown, *self._unresolved(stage, conditions, found)])
            self._decide(stage, conditions)
            if self.arms == arms:
                break
        key = step_key(candidate, "confirmation")
        missing = [
            ("confirmation", "confirmation", None, cell)
            for seed in state["started"]
            for arm in self.arms
            for cell in self._missing(key, arm, final[arm], seed)
        ]
        if missing:  # a started block's cells are issued with it
            raise _Stop(self._campaign(stage, missing))
        if not state["stop"]["stopped"]:
            unended = [block for block in state["open"] if block["runs"]]
            if unended:  # the counters wait for them: no new block
                raise _Stop(self._result(stage, self._unended(unended)))
            if not state["may_start_next"]:
                raise _Stop(self._blocked(stage, ["the reserved confirmation seeds are used up"]))
            seed = state["next_seed"]
            raise _Stop(self._campaign(stage, [
                ("confirmation", "confirmation", None, (arm, final[arm][label], seed))
                for arm in self.arms for label in LABELS]))  # fmt: skip
        readings = state["readings"]
        if any(reading is None for reading in readings.values()):
            waiting = sorted(run for block in state["open"] for run in block["runs"])
            raise _Stop(self._result(stage, {"kind": "pending", "waiting": waiting}))
        outcome = CONFIRMED if readings["A-F"] == CONFIRMATORY else DESCRIPTIVE
        reason = "n_reached" if state["stop"]["why"] == "counters" else "cap_reached"
        raise _Stop(self._result("done", {"kind": "done", "outcome": outcome, "reason": reason},
                                 outcome=outcome, reason=reason, readings=readings))  # fmt: skip

    def _unended(self, blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """The wait for the runs of started blocks that have not ended (module docstring, "The
        confirmation"): the blocks, their runs and each run's way out once the launcher that
        started it has stopped without ending it, its recovery round from its own block file."""
        ways = []
        for run_id in sorted({run for block in blocks for run in block["runs"]}):
            e = self.by_id.get(run_id)
            label = None if e is None else e.label
            campaign = None if e is None else e.cell.campaign
            ways.append({"run_id": run_id, "label": label, "campaign": campaign, "recovery": (
                f"once the launcher that started it has stopped without ending it (a Pod stopped "
                f"mid-run), its recovery round from its own block file ({campaign}.yaml), with "
                f"the interruption as its reason, launched after study3_summary.py check passes "
                f"on that file: launch.py <block file> --only {label} "
                "--recovery REASON --max-attempts 1")})  # fmt: skip
        return {"kind": "pending", "waiting": [way["run_id"] for way in ways],
                "why": "runs of started blocks have not ended: the counters wait for them, and "
                       "no new block starts",
                "blocks": [{"seed": block["seed"], "runs": list(block["runs"])}
                           for block in blocks],
                "ways_out": ways}  # fmt: skip

    # -- a run under another commit

    def _other_commits(self, stage: str, key: str) -> None:
        """The runs of the stage's cells (``key``) that do not count for the commit they ran
        (module docstring, "A run under another commit"), read before anything else of the
        stage: technical marks outside the qualification of their active arms, but no cycle
        cases (a run launched from the wrong checkout needs no code fix), so their closure is
        accepted. A valid closure decides as any technical mark's (A or F: the technical
        feasibility report, its reason the run's; P: P–F closes and the stage goes on); otherwise
        the step is blocked, technical, naming each run, its reason, its commit and its ways out
        (the lenient walk goes on)."""
        found = sorted((e for e in self.by_key.get(key, ()) if e.run_id in self.records.uncounted
                        and e.arm in self.arms), key=lambda e: e.run_id)  # fmt: skip
        if not found:
            return
        conditions: dict[str, _Arm] = {}
        for e in found:
            entry = self.records.uncounted[e.run_id]
            mark = self._mark(e, uncounted=entry)
            c = conditions.setdefault(str(e.arm), _Arm())
            c.closable.append(mark)
            ways = ([f"a recovery round of {e.label} with its documented cause, launched after "
                     "study3_summary.py check passes, reruns it"] if mark["recovery"]
                    else [])  # fmt: skip
            if mark["closure"] is not None:
                ways.append(f"a closure record ({mark['closure']['record']}) closes it")
            c.technical.append(
                f"{e.run_id}: a {e.cell.step} run under a commit other than the pinned one does "
                f"not count ({entry.get('reason')}, commit {entry.get('commit')}: "
                f"{entry.get('why')}); it is kept, excluded from every count and listed, a "
                "technical mark outside the qualification until its way out: "
                + (" or ".join(ways) or "no way out the records give"))  # fmt: skip
        for arm in ("A", "F"):
            if arm in conditions:
                self._close(stage, arm, conditions[arm])
        if "P" in conditions and self._close(stage, "P", conditions["P"]):
            del conditions["P"]
        if not conditions or self.lenient:
            return
        reasons = [x for c in conditions.values() for x in c.technical]
        ways = [self._way_out(m) for c in conditions.values() for m in c.closable]
        raise _Stop(self._result(stage, {"kind": "blocked", "technical": True,
                                         "reasons": reasons, "ways_out": ways}))  # fmt: skip

    # -- a stage's decision

    def _decide(self, stage: str, conditions: dict[str, _Arm]) -> None:
        """Module docstring, "A stage's decision": raises the stage's step, a move to the next
        candidate or a report; returns when every active arm is through the stage."""
        primary = [arm for arm in ("A", "F") if arm in conditions]
        for arm in primary:
            self._close(stage, arm, conditions[arm])
        for arm in primary:
            if conditions[arm].unresolved:
                why = "; ".join(conditions[arm].unresolved)
                self.route.append(self._event(stage, "report", [arm], why))
                raise _Stop(self._report(TECHNICAL_REPORT, "unresolved_after_cycle", [
                    f"candidate {self.candidate}, {arm}: {why}: unresolved after the candidate's "
                    "engineering cycle"]))  # fmt: skip
        p = conditions.get("P")
        if p is not None and self._close(stage, "P", p):
            del conditions["P"]
        elif p is not None and p.unresolved:
            self._close_p(stage, "unresolved after the candidate's engineering cycle: "
                          + "; ".join(p.unresolved))  # fmt: skip
            del conditions["P"]
        technical = [x for c in conditions.values() for x in c.technical]
        if technical:
            step = {"kind": "blocked", "technical": True, "reasons": technical}
            ways = [self._way_out(run) for c in conditions.values() for run in c.closable]
            if ways:  # a technical mark outside the qualification: the researcher decides
                step = {**step, "reason": ENGINEERING_ROUTE_PENDING, "why": ENGINEERING_ROUTE_WHY,
                        "ways_out": ways}  # fmt: skip
            raise _Stop(self._result(stage, step))
        pending = [x for c in conditions.values() for x in c.pending]
        if pending:
            raise _Stop(self._result(stage, {"kind": "pending", "waiting": pending}))
        failed = [arm for arm in primary if conditions[arm].scientific]
        if "P" in conditions and conditions["P"].scientific:
            self._close_p(stage, "a scientific failure: "
                          + "; ".join(conditions["P"].scientific))  # fmt: skip
            del conditions["P"]
        if failed:
            why = "; ".join(x for arm in failed for x in conditions[arm].scientific)
            raise _Move(self._event(stage, "candidate", failed, f"a scientific failure: {why}"))
        issue = [x for c in conditions.values() for x in c.issue]
        if issue:
            raise _Stop(self._campaign(stage, issue))

    # -- technical marks: holds, technical runs outside the qualification, their closures

    def _unresolved(
        self, stage: str, conditions: dict[str, _Arm], runs: Iterable[Endpoint]
    ) -> list[str]:
        """The unresolved runs of a stage outside the qualification, sorted into ``conditions``
        (module docstring, "Technical marks outside the qualification"): a held run is the hold
        route's; a run that has not ended waits; every other is a technical mark with its ways
        out. Returns the records-integrity problems, which block with no way out: two attempts
        that count for one cell, a refused run, a state no reader defines."""
        self.reached.add((self.candidate, stage))
        runs = list({e.run_id: e for e in runs}.values())
        cells: dict[tuple, list[Endpoint]] = defaultdict(list)
        for e in runs:
            if e.state in COUNTING:
                cells[(e.cell.key, e.arm, e.lr, e.seed)].append(e)
        twice = {e.run_id: group for group in cells.values() if len(group) > 1 for e in group}
        problems = []
        for e in runs:
            c = conditions.get(e.arm) or next(iter(conditions.values()))
            if e.state == "held":
                continue
            if e.run_id in twice:
                problems.append(f"{e.run_id}: one of {len(twice[e.run_id])} attempts that count "
                                f"for one cell ({e.cell.key}, {e.arm}, {e.lr}, {e.seed}): the "
                                "records' integrity, no technical mark")  # fmt: skip
            elif e.state == "refused" or e.state not in STATES:
                problems.append(f"{e.run_id}: {e.state}: no run the readers list")
            elif e.state in ("pending", "recovered"):
                c.pending.append(e.run_id)
            else:
                self._technical_run(c, e)
        return problems

    def _mark(self, e: Endpoint, uncounted: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """A technical mark outside the qualification: its run, cell and ways out (a recovery
        round where the launcher would open one, its closure record). ``uncounted``: a run under
        another commit (``uncounted_runs``' entry): its reason and commit, and no recovery round
        for a run whose outcome is published."""
        recovery = e.run_id in self.records.recoverable
        mark = {"run_id": e.run_id, "label": e.label, "candidate": e.candidate,
                "step": e.cell.step, "arm": e.arm, "rate": e.lr, "seed": e.seed,
                "recovery": recovery and (uncounted is None or e.state not in ENDED)}  # fmt: skip
        if uncounted is not None:
            mark |= {"uncounted": uncounted.get("reason"), "commit": uncounted.get("commit")}
        # a closure only for a run under another commit: any other mark's route is its cycle's
        mark["closure"] = self._closure_of(mark) if uncounted is not None else None
        self.marks.add(e.run_id)
        if uncounted is None:
            self.pending_route.add(e.run_id)
        return mark

    def _technical_run(self, c: _Arm, e: Endpoint) -> None:
        """A run outside the qualification that the readers count as a technical mark: its
        registered route, one engineering cycle, is not implemented here, so the step stops for
        the researcher's decision, beside the recovery round where the launcher would open one;
        no closure is offered (module docstring, "Technical marks outside the qualification")."""
        mark = self._mark(e)
        c.closable.append(mark)
        way = (f"; a recovery round of {e.label} with its documented cause reruns it"
               if mark["recovery"] else "")  # fmt: skip
        c.technical.append(f"{e.run_id} ({e.label}, {e.arm}, {e.cell.step}): {e.state}, a "
                           f"technical mark outside the qualification: {ENGINEERING_ROUTE_WHY}"
                           f"{way}")  # fmt: skip

    def _closure_of(self, mark: Mapping[str, Any]) -> dict[str, Any] | None:
        """The closure record of a technical mark, with the fields the closure schema accepts:
        its last end line, or its start line when the launcher never ended it, the reason the
        line gives (a run under another commit: its uncounted reason); None when the run's cell
        or line is unknown."""
        if None in (mark["candidate"], mark["arm"], mark["rate"], mark["seed"]):
            return None
        end, start = self.records.ends.get(mark["run_id"]), self.records.starts.get(mark["run_id"])
        uncounted = mark.get("uncounted")
        if end is not None and (uncounted or end_reason(end.get("outcome"))) is not None:
            line, reason = {"kind": "end", **end}, uncounted or end_reason(end["outcome"])
        elif end is None and start is not None:
            line, reason = {"kind": "start", **start}, uncounted or START_REASON
        else:
            return None
        kind = f"closure-c{mark['candidate']}-{mark['arm']}"
        return {"record": kind, "path": _path(kind), "fields": {
            "candidate": mark["candidate"], "arm": mark["arm"], "rate": mark["rate"],
            "seed": mark["seed"], "run_id": mark["run_id"],
            "line": {k: line[k] for k in ("kind", "path", "line", "line_sha256")},
            "cycle": None, "reason": reason,
            "decision": "close_pf" if mark["arm"] == "P" else TECHNICAL_REPORT}}  # fmt: skip

    def _way_out(self, mark: Mapping[str, Any]) -> dict[str, Any]:
        """A technical mark's ways out, only those that apply: a recovery round of its label
        when no terminal record was published (the launcher opens one only then), for a
        validation or confirmation run after ``study3_summary.py check`` passes; for a run under
        another commit, its closure record; for any other mark, the researcher's decision
        (``ENGINEERING_ROUTE_PENDING``), with its arm and step."""
        out: dict[str, Any] = {"run_id": mark["run_id"], "label": mark["label"]}
        if "uncounted" in mark:  # a run under another commit: its reason and commit
            out |= {"reason": mark["uncounted"], "commit": mark["commit"]}
        else:
            out |= {"arm": mark["arm"], "step": mark["step"],
                    "decision": {"decision": ENGINEERING_ROUTE_PENDING}}  # fmt: skip
        if mark["recovery"]:
            checked = (", launched after study3_summary.py check passes on its campaign file"
                       if mark["step"] in ("validation", "confirmation") else "")  # fmt: skip
            out["recovery"] = (f"a recovery round of {mark['label']} with its documented "
                               f"cause{checked}: its next attempt decides")  # fmt: skip
        if mark["closure"] is not None:
            out["closure"] = mark["closure"]
        return out

    def _close(self, stage: str, arm: str, c: _Arm) -> bool:
        """The closure of an arm's unresolved technical mark at this stage (module docstring,
        "The hold route", "Technical marks outside the qualification"): True once P's is valid
        (P closed); raises once A's or F's is (the technical feasibility report), for a held run
        until its closure is written, and for a closure that names none of the marks; False when
        there is no mark, or technical runs wait for their way out."""
        if not (c.hold or c.closable):
            return False
        kind = f"closure-c{self.candidate}-{arm}"
        shown = TECHNICAL_REPORT if arm in ("A", "F") else stage
        entry = self.entry(kind)
        if entry["state"] == "invalid":
            raise _Stop(self._blocked(shown, entry["problems"]))
        if entry["state"] == "missing":
            if not c.hold:
                return False
            hold = c.hold[0]
            lacking = [what for what, value in (
                ("a held line in the launch journal", hold["hold"]),
                (f"candidate {self.candidate}'s supersession record (its engineering cycle)",
                 hold["cycle"]), ("a reason the closure can name", hold["reasons"]))
                if not value]  # fmt: skip
            if lacking:
                raise _Stop(self._blocked(shown, [
                    f"{hold['run_id']}: a held run with no {' and no '.join(lacking)}: a closure "
                    "record cannot cite it"]))  # fmt: skip
            fields = {"candidate": self.candidate, "arm": arm, "rate": hold["rate"],
                      "seed": hold["seed"], "run_id": hold["run_id"],
                      "line": {"kind": "held", **hold["hold"]}, "cycle": hold["cycle"],
                      "reason": hold["reasons"][0],
                      "decision": "close_pf" if arm == "P" else TECHNICAL_REPORT}  # fmt: skip
            step = self._decision(
                kind,
                f"{hold['run_id']} is held, unresolved after the candidate's engineering "
                "cycle: a closure record states the terminal technical outcome",
                fields,
            )
            raise _Stop(self._result(shown, {**step, "hold_reasons": hold["reasons"]}))
        record = entry["record"]
        line = record.get("line") if isinstance(record.get("line"), Mapping) else {}
        marks = [*(("held", h) for h in c.hold), *(("mark", r) for r in c.closable)]
        wanted = "held" if line.get("kind") == "held" else "mark"  # an end or a start line
        match = next((m for kind_, m in marks if kind_ == wanted
                      and m["run_id"] == record.get("run_id")), None)  # fmt: skip
        if match is not None and wanted == "mark" and "uncounted" not in match:
            raise _Stop(self._blocked(stage, [_not_accepted(entry["path"], match["run_id"])]))
        reason = None  # a mark's closure carries the reason its line gives, or the run's own
        if match is not None and wanted == "mark" and match["closure"] is not None:
            reason = match["closure"]["fields"]["reason"]
        if match is None or not (
            record.get("seed") == match["seed"]
            and record.get("arm") == match["arm"]
            and record.get("candidate") == match["candidate"]
            and isinstance(record.get("rate"), int | float)
            and math.isclose(record["rate"], match["rate"], rel_tol=RATE_TOLERANCE)
            and record.get("reason") == (reason or record.get("reason"))
        ):
            raise _Stop(
                self._blocked(
                    shown,
                    [
                        f"{entry['path']}: it names {record.get('run_id')} ({record.get('arm')}, "
                        f"{record.get('rate')}, {record.get('seed')}, its {line.get('kind')} "
                        f"line, reason {record.get('reason')}), not a mark of candidate "
                        f"{self.candidate}'s {arm} at its {stage}: "
                        f"{[(kind_, m['run_id'], m['rate'], m['seed']) for kind_, m in marks]}"
                    ],
                )
            )
        self.consumed.add(kind)
        if arm == "P":
            self._close_p(stage, kind)
            return True
        reason = match.get("uncounted") or "held_run"
        what = "a run under another commit, " if match.get("uncounted") else ""
        self.route.append(self._event(stage, "report", [arm], kind))
        raise _Stop(self._report(TECHNICAL_REPORT, reason, [
            f"{kind}: {record['run_id']} ({line.get('kind')} line): {what}an unresolved technical "
            f"mark of {arm} at candidate {self.candidate}'s {stage}"]))  # fmt: skip

    def _place_holds(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Every held run placed on its cell, by its run id (an attempt ``read_endpoints`` lists
        as held) or, for a hold of a label alone, by its label; each with the hold line and the
        candidate's supersession record that cite it (``read_holds``). A study-3 hold that no
        start places is a problem."""
        infos = {h.get("run_id"): h for h in self.records.holds if isinstance(h, Mapping)}
        placed, problems, known = [], [], set(self.by_id)
        for e in self.endpoints:
            if e.state == "held":
                placed.append(_hold(e, infos.get(e.run_id), e.run_id))
        for run_id, info in infos.items():
            if run_id in known or info.get("experiment") not in (None, EXPERIMENT):
                continue
            owners = [
                e for e in self.endpoints if e.label is not None and e.label == info.get("run")
            ]
            if owners:
                placed.append(_hold(owners[0], info, run_id))
            else:
                problems.append(f"{run_id}: a held run (label {info.get('run')!r}) that no "
                                "study-3 start places")  # fmt: skip
        return placed, problems

    def _holds_at(self, candidate: int, arm: str, steps: Sequence[str]) -> list[dict[str, Any]]:
        return [
            h
            for h in self.placed
            if h["candidate"] == candidate and h["arm"] == arm and h["step"] in steps
        ]

    def _close_p(self, stage: str, why: str) -> None:
        if "P" in self.arms:
            self.arms.remove("P")
            self.route.append(self._event(stage, "close_pf", ["P"], why))

    # -- the decision records

    def _entries(self) -> dict[str, dict[str, Any]]:
        """The decision records by kind, each checked against the chain of GPU codes (module
        docstring, "Code agreement"), and a record invalid when one it rests on is."""
        entries: dict[str, dict[str, Any]] = {}
        for found in (*self.records.decisions.values(), self.records.closures):
            for kind, value in found.items():
                entries[kind] = {**value, "problems": list(value.get("problems") or [])}
        self._chain(entries.get("engineering") or {})
        for kind, entry in entries.items():
            record = entry.get("record")
            if entry.get("state") not in ("passed", "failed") or not isinstance(record, Mapping):
                continue
            family, candidate = kind_of(kind)
            code = (record.get("code") or {}).get("gpu_code_sha256")
            if family in ("simulation", "closure") and self.codes and code not in self.codes:
                invalidate(entry, f"its GPU code {code} is not on the chain of engineering "
                           f"cycles {self.codes}")  # fmt: skip
            elif (family in ("n", "budget") and candidate in self.in_force
                  and code != self.in_force[candidate]):  # fmt: skip
                invalidate(entry, f"its GPU code {code} is not {self.in_force[candidate]}, the "
                            f"code in force for candidate {candidate}'s runs")  # fmt: skip
        simulation = entries.get("simulation", {})
        for candidate in (1, 2):
            n, budget = entries.get(f"n-c{candidate}"), entries.get(f"budget-c{candidate}")
            if n and simulation.get("state") == "invalid" and n["state"] in ("passed", "failed"):
                invalidate(n, "the simulation record it rests on is invalid")
            if budget and n and n["state"] == "invalid" and budget["state"] in ("passed", "failed"):
                invalidate(budget, f"the n-c{candidate} record it answers is invalid")
        return entries

    def _chain(self, engineering: Mapping[str, Any]) -> None:
        """The chain of engineering cycles (``code_chain``) and ``expected_code`` at its end."""
        self.codes, self.in_force, breaks = code_chain(_engineering_code(engineering),
                                                       self.records.cycles)  # fmt: skip
        self.chain_problems += breaks
        end = self.codes[-1] if self.codes else None
        if self.codes and end != self.expected:
            self.chain_problems.append(f"expected_code {self.expected} is not the end of the "
                                       f"chain of engineering cycles, {end}")  # fmt: skip

    def entry(self, kind: str) -> dict[str, Any]:
        return self.entries.get(kind) or {"kind": kind, "path": _path(kind), "state": "missing",
                                          "record": None, "problems": []}  # fmt: skip

    def _final(self, result: dict[str, Any]) -> dict[str, Any]:
        """The records the route does not rest on: an n or budget record it has not reached
        (unless problems keep the route from showing where it is), and a closure record that
        names neither a held run of its route nor a mark the walk met (unless the walk stopped
        before the stage of the run it names). Each is a problem, and blocks."""
        if self.lenient:
            return result
        step = result["step"]
        placed = not (step["kind"] == "blocked" and not step["technical"])
        found = []
        for kind, entry in sorted(self.entries.items()):
            family, candidate = kind_of(kind)
            if entry["state"] == "missing":
                continue
            if family in ("n", "budget") and kind not in self.consumed and placed:
                found.append(f"{entry['path']}: a record the route has not reached (it is at "
                             f"candidate {self.candidate}'s {result['stage']}): a decision "
                             "record answers the step that needs it")  # fmt: skip
            elif family == "closure" and entry["state"] == "invalid":
                found += entry["problems"]
            elif family == "closure":
                record, arm = entry["record"], kind.rsplit("-", 1)[1]
                line = (record.get("line") or {}).get("kind")
                run = self.by_id.get(record.get("run_id"))
                held = any(
                    h["run_id"] == record.get("run_id")
                    and h["candidate"] == candidate
                    and h["arm"] == arm
                    and h["step"] in ROUTED
                    for h in self.placed
                )
                ended = (run is not None and run.run_id in self.marks and run.arm == arm
                         and run.candidate == candidate)  # fmt: skip
                if line != "held" and ended and run.run_id in self.pending_route:
                    found.append(_not_accepted(entry["path"], run.run_id))
                    continue
                unreached = (line != "held" and run is not None
                             and (run.candidate, run.cell.step) not in self.reached)  # fmt: skip
                if not (held if line == "held" else ended) and not unreached:
                    found.append(f"{entry['path']}: it names {record.get('run_id')}, which is "
                                 f"neither a held run of candidate {candidate}'s {arm} on the "
                                 "route nor a technical run of it outside the qualification "
                                 "(its end line)")  # fmt: skip
        found = [p for p in found if p not in self.problems]
        if not found:
            return result
        self.problems += found
        reasons = [*step["reasons"], *found] if step["kind"] == "blocked" else found
        return {**result, "step": {"kind": "blocked", "technical": False, "reasons": reasons},
                "outcome": None, "problems": list(dict.fromkeys(self.problems))}  # fmt: skip

    # -- the result

    def _block_on(self, stage: str, problems: Sequence[str], *, technical: bool = False) -> None:
        """Problems block the step (strictly); the lenient walk sets them aside."""
        problems = list(dict.fromkeys(problems))
        if problems:
            self.problems += [p for p in problems if p not in self.problems]
            if not self.lenient:
                raise _Stop(self._result(stage, {"kind": "blocked", "technical": technical,
                                                 "reasons": problems}))  # fmt: skip

    def _blocked(self, stage: str, problems: Sequence[str]) -> dict[str, Any]:
        problems = list(dict.fromkeys(problems))
        self.problems += [p for p in problems if p not in self.problems]
        return self._result(stage, {"kind": "blocked", "technical": False, "reasons": problems})

    def _decision(self, kind: str, why: str, fields: Mapping[str, Any] | None = None) -> dict:
        step = {"kind": "decision", "record": kind, "path": _path(kind), "why": why}
        return step if fields is None else {**step, "fields": dict(fields)}

    def _campaign(self, stage: str, issue: Sequence[tuple], **extra: Any) -> dict[str, Any]:
        """The campaign of the first kind among the cells to issue, with its cells."""
        campaign, step, round_ = min((x[:3] for x in issue), key=lambda x: CAMPAIGNS.index(x[0]))
        cells = sorted({x[3] for x in issue if x[:3] == (campaign, step, round_)},
                       key=lambda c: (c[2], ARM_ORDER.index(c[0]), c[1]))  # fmt: skip
        return self._result(stage, {"kind": "campaign", "campaign": campaign,
                                    "candidate": self.candidate, "step": step, "round": round_,
                                    "cells": cells, **extra})  # fmt: skip

    def _report(self, outcome: str, reason: str, reasons: Sequence[str]) -> dict[str, Any]:
        """A terminal outcome (``TERMINAL``) with its reason id and why."""
        return self._result(outcome, {"kind": "report", "outcome": outcome, "reason": reason},
                            outcome=outcome, reason=reason, reasons=reasons)  # fmt: skip

    def _event(self, stage: str, event: str, arms: Sequence[str], why: str) -> dict[str, Any]:
        return {"candidate": self.candidate, "stage": stage, "event": event, "arms": list(arms),
                "why": why}  # fmt: skip

    def _result(self, stage: str, step: dict[str, Any], *, outcome: str | None = None,
                reason: str | None = None, reasons: Sequence[str] = (),
                **extra: Any) -> dict[str, Any]:  # fmt: skip
        return {
            "stage": stage, "candidate": self.candidate, "step": step, "outcome": outcome,
            "reason": reason, "reasons": list(reasons),
            "active": {"arms": list(self.arms), "comparisons": self._comparisons()},
            "rates": {arm: dict(r) for arm, r in self.rates.items() if arm in self.arms},
            "route": list(self.route),
            "closures": {k: {key: e[key] for key in ("state", "path", "record", "problems")}
                         for k, e in sorted(self.entries.items())
                         if k.startswith("closure-") and e["state"] != "missing"},
            "problems": list(dict.fromkeys(self.problems)),
            "not_counted": [dict(self.records.uncounted[k])
                            for k in sorted(self.records.uncounted)],
            "prescribed": {k: sorted(v) for k, v in sorted(self.prescribed.items()) if v},
            **self.extra, **extra,
        }  # fmt: skip

    # -- helpers

    def _comparisons(self) -> list[str]:
        return ["A-F", "P-F"] if "P" in self.arms else ["A-F"]

    def _prescribe(self, candidate: int, step: str, round_: int | None,
                   cells: Iterable[tuple[str, float, int]]) -> None:  # fmt: skip
        self.prescribed[step_key(candidate, step, round_)].update(cells)

    def _attempts(self, key: str, arm: str, rate: float, seed: int) -> list[Endpoint]:
        return [
            e
            for e in self.by_key.get(key, ())
            if e.arm == arm
            and e.seed == seed
            and e.lr is not None
            and math.isclose(e.lr, rate, rel_tol=RATE_TOLERANCE)
        ]

    def _of_seed(self, seed: int) -> list[Endpoint]:
        """The endpoints of a block's seed, and those bound to no seed (they may be the
        block's)."""
        return [*self.by_seed.get(seed, ()), *self.by_seed.get(None, ())]


def _hold(e: Endpoint, info: Mapping[str, Any] | None, run_id: str) -> dict[str, Any]:
    """A held run on its cell, with the hold line and the candidate's cycle record that a closure
    record cites (``read_holds``' records: the line by its own sha256, the cycle by its file's)."""
    records = (info or {}).get("records") or []
    journal = next((r for r in records if r.get("kind") == "journal"), None)
    cycle = next((r for r in records if r.get("kind") == "supersession" and str(
        r.get("path")).startswith(f"qualification/c{e.candidate}/supersede-")), None)  # fmt: skip
    return {"run_id": run_id, "candidate": e.candidate, "step": e.cell.step, "arm": e.arm,
            "rate": e.lr, "seed": e.seed,
            "hold": None if journal is None else {k: journal[k] for k in ("path", "line",
                                                                          "line_sha256")},
            "cycle": None if cycle is None else {k: cycle[k] for k in ("path", "sha256")},
            "reasons": hold_reasons((info or {}).get("reason"))}  # fmt: skip


def _not_accepted(path: Any, run_id: Any) -> str:
    """The problem of a closure that names a technical mark outside the qualification other than
    a run under another commit: it decides nothing."""
    return f"{path}: it names {run_id}: {NOT_ACCEPTED}"


def _path(kind: str) -> str:
    """A decision record's path under RUNS_DIR (``decision_path``)."""
    return decision_path(".", kind).as_posix()


def _rate(reading: Mapping[str, Any], arm: str, lr: float) -> Mapping[str, Any] | None:
    """The rate of a qualification reading for ``arm`` and ``lr`` (relative tolerance 10⁻⁹)."""
    for rate in (reading.get("rates") or {}).values():
        if (rate.get("arm") == arm and isinstance(rate.get("lr"), int | float)
                and math.isclose(rate["lr"], lr, rel_tol=RATE_TOLERANCE)):  # fmt: skip
            return rate
    return None


def _condition(entry: Mapping[str, Any], reading: Mapping[str, Any]) -> tuple[str, str]:
    """A rate's condition (``qualified``, ``failing``, ``technical`` or ``waiting``) and why
    (module docstring, "The numerical qualification")."""
    state = entry.get("state")
    if state == QUALIFIED:
        return "qualified", ""
    if state == FAILING:
        return "failing", ""
    if state == INCOMPLETE:
        return "waiting", ""
    if state != NOT_QUALIFIED:
        return "technical", " (no state recomputed)"
    verdict = entry.get("verdict") or {}
    control = verdict.get("control")
    awaiting = isinstance(control, Mapping) and control.get("record_sha256") is None
    if awaiting and not verdict.get("not_counted") and not _technical_case(entry, reading):
        return "waiting", " (its same-path control has no record yet)"
    return "technical", ""


def _technical_case(entry: Mapping[str, Any], reading: Mapping[str, Any]) -> bool:
    """Whether a case of the rate that counts (neither voided nor left out by the verdict) is
    technical."""
    verdict = entry.get("verdict") or {}
    excluded = {*(reading.get("voided") or {}),
                *(e.get("record_sha256") for e in verdict.get("not_counted") or [])}  # fmt: skip
    for case in reading.get("cases") or []:
        named = case.get("case") or {}
        lr = named.get("lr")
        if (named.get("optimizer") == entry.get("optimizer") and isinstance(lr, int | float)
                and math.isclose(lr, entry["lr"], rel_tol=RATE_TOLERANCE)
                and case.get("record_sha256") not in excluded
                and case.get("kind") == TECHNICAL):  # fmt: skip
            return True
    return False


# ------------------------------------------------------------------ the confirmation's state


def confirmation_state(
    endpoints: Iterable[Endpoint], candidate: int, N: int, active: Sequence[str]
) -> dict[str, Any]:
    """The confirmation of a candidate ("Confirmation"; module docstring): ``started``, the
    blocks started, in seed order, as the distinct seeds of its starts in the launch journal
    (a start with no directory yet included; a retry of a label adds no block); ``blocks`` (per
    started seed, its state and each active comparison's); per active comparison (``active``:
    ``["A-F"]`` or ``["A-F", "P-F"]``) its ``counters`` (the supported blocks counted in seed
    order, up to the first unfinished one: an unfinished earlier block holds the later ones),
    ``held_up`` (that block), ``cohorts`` (the first N supported blocks by seed order),
    ``supported`` (every supported block started, in any order); the ``stop`` (every active
    counter at N, or the cap N + max(10, ⌈0.12·N⌉) blocks started); the ``readings`` once stopped
    (``confirmatory`` for a comparison at N, and at the cap with at least 36 supported blocks,
    ``descriptive`` with fewer; None while a block it needs is unfinished) and ``analysis``, the
    blocks each is read on (its cohort, or at the cap all its supported blocks); ``technical``
    and ``open`` (the unfinished blocks and their runs: an open block's runs are those that have
    not ended, none when it only lacks cells); ``problems`` (blocks started out of seed order or
    beyond the cap); ``may_start_next`` (False once stopped, with a problem, while a started block
    holds a technical mark awaiting its way out or a run that has not ended, ``pending`` or
    ``recovered`` with no later attempt, since the stop is checked between blocks and such a run
    leaves the counters unknown; or once the reserved seeds are used up) and ``next_seed``.
    ``ValueError`` for another candidate, an N that is not a positive integer, or other
    comparisons."""
    if candidate not in CANDIDATES or isinstance(candidate, bool):
        raise ValueError(f"no candidate {candidate!r} (registered: {sorted(CANDIDATES)})")
    if type(N) is not int or N < 1:
        raise ValueError(f"N is a positive integer, not {N!r}")
    if isinstance(active, str) or list(active) not in ACTIVE:
        raise ValueError(f"active comparisons are one of {ACTIVE}, not {active!r}")
    comparisons = list(active)
    arms = [arm for arm in ARM_ORDER if arm != "P" or "P-F" in comparisons]
    cap = cap_of(N)
    reserved = list(SEEDS["confirmation"][candidate])
    key = step_key(candidate, "confirmation")
    endpoints = list(endpoints)
    started = sorted({e.seed for e in endpoints if e.cell.key == key
                      and e.cell.campaign is not None and e.seed is not None})  # fmt: skip
    problems = []
    skipped = [s for s in reserved[: len(started)] if s not in started]
    if skipped:
        problems.append(
            f"confirmation blocks started out of seed order: {skipped[:5]} not started "
            f"before later seeds ({reserved[0]}, {reserved[1]}, … in that order)"
        )
    beyond = [s for s in started if s not in reserved[:cap]]
    if beyond:
        problems.append(f"confirmation blocks started beyond the cap of {cap} blocks or the "
                        f"reserved seeds: {beyond[:5]}")  # fmt: skip
    by_seed: dict[int | None, list[Endpoint]] = defaultdict(list)
    for e in endpoints:
        by_seed[e.seed].append(e)
    blocks = {s: block_state([*by_seed[s], *by_seed[None]], s, candidate, arms)
              for s in started if s in reserved}  # fmt: skip
    counters, cohorts, held_up, supported = {}, {}, {}, {}
    for comparison in comparisons:
        name = COMPARISONS[comparison]
        count, cohort, first = 0, [], None
        for s in blocks:
            state = blocks[s][name]["state"]
            if state in (OPEN, UNRESOLVED):
                first = s
                break
            if state == SUPPORTED:
                count += 1
                if len(cohort) < N:
                    cohort.append(s)
        counters[comparison], cohorts[comparison], held_up[comparison] = count, cohort, first
        supported[comparison] = [s for s, b in blocks.items() if b[name]["state"] == SUPPORTED]
    why = ("counters" if all(counters[c] >= N for c in comparisons) else
           "cap" if len(started) >= cap else None)  # fmt: skip
    readings: dict[str, str | None] = {}
    analysis: dict[str, list[int] | None] = {}
    for comparison in comparisons:
        unfinished = any(b[COMPARISONS[comparison]]["state"] in (OPEN, UNRESOLVED)
                         for b in blocks.values())  # fmt: skip
        if counters[comparison] >= N:
            readings[comparison], analysis[comparison] = CONFIRMATORY, cohorts[comparison]
        elif why == "cap" and not unfinished:
            count = len(supported[comparison])
            readings[comparison] = CONFIRMATORY if count >= CONFIRMATORY_AT else DESCRIPTIVE
            analysis[comparison] = supported[comparison]
        else:
            readings[comparison], analysis[comparison] = None, None
    technical, waiting = [], []
    for s, b in blocks.items():
        if b["state"] == UNRESOLVED:
            runs = {e.run_id for arm in arms for e in b["arms"][arm]["technical"]}
            runs |= {blocker.split(":", 1)[0] for blocker in b["blockers"]}
            technical.append({"seed": s, "runs": sorted(runs)})
        elif b["state"] == OPEN:
            runs = {e.run_id for arm in arms for e in b["arms"][arm]["pending"]}
            waiting.append({"seed": s, "runs": sorted(runs)})
    room = len(started) < min(cap, len(reserved))
    unended = [block for block in waiting if block["runs"]]  # a run not ended holds the counters
    may = why is None and not problems and not technical and not unended and room
    return {
        "candidate": candidate, "N": N, "cap": cap, "active": comparisons, "started": started,
        "blocks": {s: {"state": b["state"], **{c: b[COMPARISONS[c]]["state"]
                                                for c in comparisons}}
                   for s, b in blocks.items()},
        "counters": counters, "held_up": held_up, "cohorts": cohorts,
        "supported": {c: len(v) for c, v in supported.items()},
        "stop": {"stopped": why is not None, "why": why}, "readings": readings,
        "analysis": analysis, "technical": technical, "open": waiting, "problems": problems,
        "may_start_next": may, "next_seed": reserved[len(started)] if may else None,
    }  # fmt: skip
