"""Print the state of study 3, guard its confirmation between blocks, and write the records the
researcher writes: the versions record, the staged fixation, the decision records and a closure.

    python scripts/study3_summary.py progress --runs-root DIR [--theta0 RUN_DIR]
    python scripts/study3_summary.py guard --runs-root DIR --candidate N --seed S --campaign FILE
        [--theta0 RUN_DIR]
    python scripts/study3_summary.py check --runs-root DIR --campaign FILE
    python scripts/study3_summary.py versions --out PATH --reason ID [--fixation PATH
        --runs-root DIR] [--supersedes PATH --supersedes-reason ID]
    python scripts/study3_summary.py fixation validation --runs-root DIR --candidate N
        [--theta0 RUN_DIR]
    python scripts/study3_summary.py fixation confirmation --runs-root DIR --candidate N
        --bootstrap-seed S --audit-seed S --agreement-seed S
    python scripts/study3_summary.py record KIND --runs-root DIR --from FILE
    python scripts/study3_summary.py closure --runs-root DIR --candidate N --arm ARM
        [--run-id RUN_ID] [--theta0 RUN_DIR]
    python scripts/study3_summary.py cycle --runs-root DIR --candidate N --diagnosis TEXT
        --fix TEXT --stage STAGE=reuse|rerun[:ARMS] ... --justification STAGE=TEXT ...
        [--theta0 RUN_DIR] [--dry-run]

The rules are docs/study3-preregistration.md's: "Validation, confirmation and stopping", "Failure
routes", "Seeds and order", "Voiding and supersession" and amendments 1-3.

**The state** is ``analysis.study3_progress.progress`` of --runs-root (θ₀'s evaluation from
--theta0, study 2's under RUNS_DIR by default), its expected code this checkout's GPU code
(``code_identity.gpu_code_identity``). ``progress`` prints it whole as JSON: numbers, ids, paths
and hashes only (a spread appears only as an n record's ``spread_pp``, rounded to 0.5 pp), the
launch journal's resolved torn lines included (``resolved_torn_lines``: each line's log, number
and kind).

**The guard** is the between-block guard of the confirmation's driver (``run.sh``, written by
scripts/study3_campaigns.py confirmation, which runs it before each block and stops at its first
non-zero exit). It exits 0 only when block S (--seed) of candidate N may run now, from the block's
campaign file FILE (--campaign), printing the block; otherwise it exits non-zero with the reason.
All of these must hold:
- the state's step is candidate N's confirmation campaign, and S is the seed of cells in it. A
  block not yet started must be the next block the confirmation authorizes (``confirmation``'s
  ``may_start_next`` and ``next_seed``): not after the stop (every active counter at N, or the
  cap N + max(10, ⌈0.12·N⌉) of blocks started), not while a technical mark in a started block
  waits for its way out, not while a run of a started block has not ended (pending, or recovered
  with no later attempt: the counters wait for it, and the refusal names its recovery round, the
  run's own at --max-attempts 1 with its incident), not
  out of seed order, and not while a started block's missing cells are still to be completed. A
  started block only completes the cells the step names for it,
  whatever ``may_start_next`` says: ``progress`` issues them even after the stop;
- FILE is the block as the writer wrote it for the step: its header names study 3, candidate N,
  the step ``confirmation`` and no round, with the confirmation's keys only; its cells are
  exactly the step's cells at S, one launcher cell per run as the writer issues them
  (``c<N>-confirmation-<optimizer>-<rate>-s<S>``, config experiments/study3, the arm, B = 20, the
  rate and the role, the one seed S);
- the records its header names are as they were when it was written: the n record (always) and
  the budget record (when named), each the candidate's under RUNS_DIR (its chain's link in force:
  a successor after a cycle) with the sha256 the header gives, and the versions record, in the
  checkout, of its named sha256;
- ``study3_versions.versions_problems`` finds no problem in that versions record for the
  confirmation of candidate N, the running checkout's HEAD included: it prevents a block from
  running under a commit other than the pinned one. It does not stand for comparing each run's
  own commit with the pinned one, which ``progress`` does (``study3_versions.uncounted_runs``):
  a run under another commit does not count, and the state blocks on it, technical, until its
  way out (a recovery round where the launcher would open one, the run's own at --max-attempts 1
  with a reason no earlier round of its label gave and its incident, or its closure record): no
  block starts meanwhile.

**check** is the same check of the versions record before any other launch of a validation or
confirmation campaign FILE (--campaign): the validation's launch and every resumption of it, and
every recovery round of a validation or confirmation run, the first or a further one (a confirmation
block's own launch is guarded by ``guard``). It exits 0 only when FILE's header is a validation or
confirmation campaign of study 3 (study 3, the step, candidate 1 or 2), it names a versions record
of the checkout with that record's sha256 now, and ``versions_problems`` finds no problem in that
record for the step and the candidate, the running checkout's HEAD included; otherwise it exits
non-zero with the reason. Run it from the checkout (``$SRC``) the launch runs from. The launcher
never reruns a completed run, so a run launched under another commit stays, does not count, and
blocks the state until its closure record is written.

**versions** writes a versions record (``study3_versions.versions_record``) at --out, a name the
readers accept: docs/study3-versions.json, or docs/study3-versions-<k>.json (k = 2, 3, ...) for a
later record (a relative path is read from the checkout). It needs a clean checkout of a known
commit, whose HEAD it records, and is written once; it is then committed alone. --reason says why
it is written, --supersedes-reason why the record before it no longer serves: ids (a lowercase
letter, then lowercase letters, digits or underscores), never free text. --fixation names the
committed fixation record its runs are under, read with its evidence under --runs-root; without
it the record names none and stands for no validation or confirmation run. A later record
supersedes the one before it (--supersedes, which must be that record, with
--supersedes-reason); the first supersedes none.

**fixation** writes candidate N's validation fixation (``study3_versions.validation_fixation``:
the stage read as ``progress`` reads it, θ₀ from --theta0) or its confirmation supplement
(``confirmation_fixation``: the bootstrap's seed, the audit draw's seed and the seed of the draw
of agreements, integers in [0, 2³²), as "Inference" and "The audit" require), once, at its fixed
path docs/study3-fixation-<step>-c<N>.json. It needs a clean checkout of a known commit: the
record's code is HEAD's commit and that commit's GPU code, which the builders take as given. The
record is then committed alone, before the next one is written. After candidate N's cycle
(module docstring, "After a cycle") a record of it is written only when the state's step names
it: the first supplement once the state is at N's confirmation, a stale fixation's successor
(``docs/study3-fixation-<step>-c<N>-<k>.json``, naming the record before it and the cycle) when
the step names that successor.

**record** writes a decision record (engineering, simulation, n-c<N>, budget-c<N>) from a JSON
file through ``study3_decisions.write_decision``, which validates it against the evidence it
names and the records it answers: written once, never over another; a refusal prints its reasons.
After candidate N's cycle an n or budget record is written only when the state's step names it:
the first record, or, the record in force stale through the cycle, its next successor
(``n-c<N>-<k>``, ``budget-c<N>-<k>``; the file carries ``supersedes``), its content checked
first (after the confirmation has started, N, the cap and the cohort stand). Once candidate N has
a cycle record, its n and budget records, its fixations and a versions record naming its fixation
are written only under the cycle's fix (the code in force for N's runs), and no record under the
old code of any candidate's cycle (``study3_decisions.after_cycle_code``): what a cycle lists as
stale then changes after it only when a file is tampered with.

**After a cycle** ("Voiding and supersession", "Sample size from validation", "Seeds and order").
The write-once records of the old GPU code are stale; their successors come in the registered
order, each committed alone: (1) the validation fixation's successor and its versions record;
(2) the validation's reissued cells, issued under them; (3) the n record's successor (spread, N,
cohort, on the validation that counts) and the budget record's when N asks for one; (4) the
supplement's successor and its versions record. A successor written out of turn is refused with
the step the state names. Once the candidate's confirmation has started, the validation is not
reopened (``cycle`` refuses validation=rerun and search=rerun) and the successors keep N, the cap,
the cohort and the seeds.

**closure** writes the closure record closure-c<N>-<ARM> with exactly the fields the state's step
gives: a held run's decision step, the way out of a run under another commit (its last end
line, or its start line, with its uncounted reason), or, once candidate N's engineering cycle is
spent, the way out of a technical mark outside the qualification left unresolved after it (its
last end line, or its start line, with the reason ``unresolved_after_cycle`` and the candidate's
supersession record, ``qualification/c<N>/supersede-<n>.json``, by its sha256: "Failure routes").
Before the cycle such a mark takes its engineering cycle (``engineering_cycle``, below): the step
names no closure for it, and none is written. It refuses when the step names no closure of
that candidate and arm, and, when the step names several (two runs of the arm), unless --run-id
chooses the run. The record's state is ``failed`` (a terminal technical
outcome) and its code HEAD's commit and that commit's GPU code.

**cycle** writes candidate N's engineering cycle for its technical marks outside the
qualification ("Failure routes": one cycle per candidate, a diagnosis, a fix, re-qualification of
the affected cases and reruns of the affected runs; "Voiding and supersession"): the cycle
record ``RUNS_DIR/study3/decisions/cycle-c<N>.json`` and its bridge,
``qualification/c<N>/supersede-1.json`` in ``scripts/qualify.py``'s own format
(``analysis.study3_cycle``). The right to a cycle is judged
on the existing chain of engineering cycles: the records are read with no expected code
(``read_records``), the code in force is the chain's end (``chain_end``; N's own supersession
records aside, so that a cycle recorded already is refused as such), and the state under it must
be candidate N's ``engineering_cycle`` step; the fix's own code never enters that judgement
(the chain does not hold it before the cycle). The fix is checked on its own: a clean checkout of
a known commit whose GPU code is another than the code in force. The ordinary readers
(``progress``, ``check``, ``guard``, the campaign writer) keep this checkout's code as their
expected code: from the fix's checkout, before the cycle, they report the chain problem and refuse
to launch or issue anything. The marks, their stage and the closed arms are the state's; each
claim (``--stage STAGE=reuse|rerun[:ARMS]``, every arm by default; the qualification's all or
nothing) carries its justification (``--justification STAGE=TEXT``), one line each, as the
diagnosis and the fix. The journal prefix is pinned first, then the records before the cycle are
read under it (``pre_cycle_records``) and the cycle derived from them (``derive_cycle``):
``written_at`` now, ``host`` this machine. --dry-run prints the record and the bridge and writes
nothing; otherwise both are published, the record first, and the two paths are printed, then the
journal repair to run where RUNS_DIR is (the bridge's ``old_record.path`` names this machine's
path): ``project-run "$SRC" python scripts/qualify.py supersede --campaign <engineering campaign
file> --candidate N --engineering --case <its first case> --record <the cycle record> --diagnosis
D --fix F``, the campaign file ``RUNS_DIR/campaigns/<name>.yaml``, ``<name>`` the campaign of the
journal's first engineering start. Run again with its own arguments over a published record
whose bridge is missing, it writes only the bridge (from the record: ``rebuild_bridge``).
Refusals, in this order: the fix (a checkout that is not clean or of no known commit); a launch
journal that cannot be read whole (a log unreadable or under a sealed folder, an unresolved torn
line: ``journal_unread``, named before anything that reads the journal short of it); the code in
force ("a fix that leaves the GPU code unchanged is no cycle"); a supersession record of N
there already, or a cycle record whose arguments are not these ("recorded already"); no
engineering campaign file where it is looked for; on the existing chain, a step that is not N's
engineering cycle, a state with problems, or a journal that is not whole (an unresolved torn
line, a line a published record requires and no log holds: ``journal_problems``); a run of N in a
stage and arm the claims rerun that has not ended (pending, or recovered with no later attempt);
a claim or justification missing or malformed, a marked arm not rerun, the search rerun once a
validation fixation exists, or the validation or the search rerun once the confirmation has
started; and what ``derive_cycle`` refuses (a run of N in flight, in any stage and arm: its
start in the pinned journal prefix, no end line in it and nothing settling it, named, since its
records would change after the cycle is written; a mark under a known other GPU code, a record
under a sealed folder or there and unreadable, a qualification launcher run the readers cannot
settle, and the rest it names).

Every record is JSON, written whole and only where none is (``study3_decisions._publish``).
Nothing here opens a response, a harm label or a judge output: a campaign file, a record file or a
record that resolves under a ``responses/`` or ``observations/`` folder
(``study3_decisions._under_sealed``) is refused before it is read or hashed.
"""

import argparse
import hashlib
import json
import math
import re
import shlex
import socket
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

import yaml

from zero_field_refusal import code_identity
from zero_field_refusal.analysis.study3_cycle import (
    STAGE_STEPS,
    CycleRefused,
    derive_cycle,
    journal_problems,
    journal_seen,
    journal_unread,
    parse_claims,
    pre_cycle_records,
    publish_cycle,
    rebuild_bridge,
)
from zero_field_refusal.analysis.study3_decisions import (  # its sealed-path rule, its
    DecisionRefused,  # write-once publication
    _problems,
    _publish,
    _under_sealed,
    after_cycle_code,
    chain,
    confirmation_started,
    cycle_path,
    decision_path,
    kind_of,
    next_successor,
    read_decision,
    write_decision,
)
from zero_field_refusal.analysis.study3_progress import (
    ENGINEERING_CYCLE,
    THETA0_RUN,
    advance,
    chain_end,
    progress,
    read_records,
)
from zero_field_refusal.analysis.study3_records import (
    ARMS,
    BATCH,
    RATE_TOLERANCE,
    SEEDS,
    STEP_ROLES,
    journal_lines,
    sealed_why,
)
from zero_field_refusal.analysis.study3_search import ARM_ORDER
from zero_field_refusal.analysis.study3_versions import (
    _FIXATION_NAME,
    RecordRefused,
    confirmation_fixation,
    fixation_chain,
    fixation_name,
    fixation_number,
    validation_fixation,
    versions_name,
    versions_problems,
    versions_record,
)
from zero_field_refusal.config import load_config
from zero_field_refusal.paths import REPO_ROOT, runs_root
from zero_field_refusal.tracking import git_state

REPO = REPO_ROOT  # the checkout: its versions records, the decision records' repo evidence
REGULAR = "experiments/study3"  # the confirmation runs' config
OPTIMIZERS = {arm: name for name, arm in ARMS.items()}
# A confirmation block's header, as the writer gives it: these keys and no other
HEADER = {"study": 3, "step": "confirmation", "round": None}
NAMED = ("versions_record", "versions_record_sha256", "n_record", "n_record_sha256",
         "budget_record", "budget_record_sha256")  # fmt: skip
OVERRIDES = ("optimizer", "train.batch_size", "optimizer.lr", "study3.role")
MISSING = object()
# The decision records ``record`` writes from a file (a closure is written from the step)
RECORDS = ("engineering", "simulation", "n-c1", "n-c2", "budget-c1", "budget-c2")
STEPS = ("validation", "confirmation")  # the fixation's stages
NO_CYCLE = ("a fix that leaves the GPU code unchanged is no cycle: commit it and read the state "
            "again")  # fmt: skip


def refuse(command: str, why: str) -> NoReturn:
    sys.exit(f"study3_summary.py {command}: {why}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unopened(path: Path, base: Path) -> str:
    """Why the sealed-path rule refuses ``path`` (``study3_decisions._under_sealed``): it resolves
    under a ``responses/`` or ``observations/`` folder, or it cannot be resolved (a link loop)."""
    return sealed_why(path, base, "resolves under a responses/ or observations/ folder")


def state_of(args: argparse.Namespace) -> dict[str, Any]:
    """The state of the records under --runs-root, read under this checkout's GPU code."""
    expected = code_identity.gpu_code_identity(repo=REPO)["sha256"]
    return progress(args.runs_root, repo=REPO, expected_code=expected,
                    theta0=getattr(args, "theta0", None))  # fmt: skip


def after_cycle(args: argparse.Namespace, candidate: int) -> dict[str, Any] | None:
    """The state, once candidate N has a valid cycle record outside the qualification (module
    docstring, "After a cycle"); None before (no state is read when no cycle record is
    there)."""
    if not (cycle_path(args.runs_root, candidate).exists()
            or cycle_path(args.runs_root, candidate).is_symlink()):  # fmt: skip
        return None
    state = state_of(args)
    found = (state.get("cycle") or {}).get(candidate) or {}
    return state if found.get("state") == "valid" else None


def named_step(command: str, state: dict[str, Any], kind: str) -> None:
    """Refused unless the state's step names the record ``kind`` (module docstring, "After a
    cycle": each successor in the registered order)."""
    step = state["step"]
    if not (step.get("kind") == "decision" and step.get("record") == kind):
        refuse(command, f"{kind}: the state's step does not name it: after the cycle the records "
               "are written in the registered order, each when the step names it (the state names "
               f"{shown(state)})")  # fmt: skip


def shown(state: dict[str, Any]) -> str:
    """The step the state names, as JSON (a campaign's cells counted, with their seeds)."""
    step = dict(state["step"])
    if isinstance(step.get("cells"), list):
        cells = step["cells"]
        step["cells"] = f"{len(cells)} cells on seeds {sorted({c[2] for c in cells})}"
    out = {"stage": state["stage"], "candidate": state["candidate"], "step": step}
    if state.get("outcome") is not None:
        out.update(outcome=state["outcome"], reason=state.get("reason"),
                   reasons=state.get("reasons"))  # fmt: skip
    return json.dumps(out, default=str, ensure_ascii=False)


# ------------------------------------------------------------------ progress


def print_progress(args: argparse.Namespace) -> None:
    print(json.dumps(state_of(args), indent=1, default=str, ensure_ascii=False))


# ------------------------------------------------------------------ the guard


def stopped(confirmation: dict[str, Any]) -> str | None:
    """Why the confirmation has stopped, or None."""
    stop = confirmation.get("stop") or {}
    if not stop.get("stopped"):
        return None
    if stop.get("why") == "counters":
        return (f"the confirmation has stopped: every active counter has reached N = "
                f"{confirmation.get('N')} ({confirmation.get('counters')})")  # fmt: skip
    return f"the confirmation has stopped: the cap of {confirmation.get('cap')} blocks has started"


def not_next(confirmation: dict[str, Any]) -> str:
    """Why the confirmation lets no new block start (``may_start_next`` false)."""
    if stopped(confirmation):
        return stopped(confirmation)
    if confirmation.get("technical"):
        return (f"a technical mark in a started block waits for its way out "
                f"({confirmation['technical']})")  # fmt: skip
    unended = [block for block in confirmation.get("open") or [] if block.get("runs")]
    if unended:
        return f"runs of started blocks have not ended ({unended})"
    if confirmation.get("problems"):
        return "; ".join(confirmation["problems"])
    return "the reserved confirmation seeds are used up"


def the_block(state: dict[str, Any], candidate: int, seed: int) -> tuple[list, bool]:
    """The step's cells at ``seed`` as (arm, rate), and whether the block has started, when the
    block may run now (module docstring, "The guard"); refused otherwise."""
    if seed not in SEEDS["confirmation"][candidate]:
        seeds = SEEDS["confirmation"][candidate]
        refuse("guard", f"block {seed} is not one of candidate {candidate}'s confirmation seeds "
               f"({seeds[0]}-{seeds[-1]})")  # fmt: skip
    step = state["step"]
    confirmation = state.get("confirmation") or {}
    if confirmation.get("candidate") != candidate:
        confirmation = {}
    started = set(confirmation.get("started") or [])
    named = (step.get("kind") == "campaign" and step.get("campaign") == "confirmation"
             and state.get("candidate") == candidate)  # fmt: skip
    if not named:
        if stopped(confirmation) and seed not in started:
            refuse("guard", f"block {seed} may not start: {stopped(confirmation)}")
        if step.get("kind") == "blocked":
            what = ("a technical mark holds the confirmation" if step.get("technical")
                    else "the step is blocked")  # fmt: skip
            refuse("guard", f"block {seed} may not start: {what}: "
                   + "; ".join(step.get("reasons") or []))  # fmt: skip
        if step.get("kind") == "pending" and state.get("stage") == "confirmation":
            ways = "; ".join(f"{way.get('run_id')}: {way.get('recovery')}"
                             for way in step.get("ways_out") or [])  # fmt: skip
            refuse("guard", f"block {seed} may not start: runs of started blocks have not ended "
                   f"({', '.join(map(str, step.get('waiting') or []))}), and the counters wait "
                   f"for them" + (f"; {ways}" if ways else ""))  # fmt: skip
        refuse("guard", f"the step is not candidate {candidate}'s confirmation: the state names "
               f"{shown(state)}")  # fmt: skip
    seeds = sorted({cell[2] for cell in step["cells"]})
    if seed in started:
        if seed not in seeds:
            refuse("guard", f"block {seed} has started, and the step names none of its cells (it "
                   f"names block(s) {seeds}): another seed")  # fmt: skip
    elif seed not in seeds:
        if stopped(confirmation):
            refuse("guard", f"block {seed} may not start: {stopped(confirmation)}")
        completing = [s for s in seeds if s in started]
        if completing:
            refuse("guard", f"block {seed} may not start: the step completes the started "
                   f"block(s) {completing} first")  # fmt: skip
        refuse("guard", f"block {seed} is out of seed order: the next block is "
               f"{confirmation.get('next_seed')}")  # fmt: skip
    elif not confirmation.get("may_start_next"):
        refuse("guard", f"block {seed} may not start: {not_next(confirmation)}")
    elif seed != confirmation.get("next_seed"):
        refuse("guard", f"block {seed} is out of seed order: the next block is "
               f"{confirmation.get('next_seed')}")  # fmt: skip
    cells = [(arm, float(rate)) for arm, rate, s in step["cells"] if s == seed]
    return cells, seed in started


def launcher_cell(cell: Any, candidate: int, seed: int) -> tuple[str, float] | str:
    """A launcher cell of block ``seed`` as the writer issues it: its arm and rate; or why it is
    not one."""
    if not isinstance(cell, dict):
        return f"a cell that is not a mapping ({cell!r})"
    name, overrides = cell.get("name"), cell.get("overrides")
    if not (isinstance(overrides, list) and all(isinstance(o, str) and "=" in o
                                                for o in overrides)):  # fmt: skip
        return f"cell {name!r}: its overrides {overrides!r} are not key=value strings"
    pairs = dict(o.split("=", 1) for o in overrides)
    if len(pairs) != len(overrides) or sorted(pairs) != sorted(OVERRIDES):
        return f"cell {name!r}: its overrides {overrides} are not the confirmation's {OVERRIDES}"
    arm = ARMS.get(pairs["optimizer"])
    try:
        rate = float(pairs["optimizer.lr"])
    except ValueError:
        rate = math.nan
    problems = []
    if arm is None or not (math.isfinite(rate) and rate > 0):
        problems.append(f"its overrides {overrides} name no arm and rate")
    if (pairs["train.batch_size"], pairs["study3.role"]) != (
        str(BATCH),
        STEP_ROLES["confirmation"],
    ):
        problems.append(f"its overrides {overrides} are not B = {BATCH} and the role "
                        f"{STEP_ROLES['confirmation']}")  # fmt: skip
    if cell.get("config") != REGULAR:
        problems.append(f"its config {cell.get('config')!r} is not {REGULAR}")
    seeds = cell.get("seeds")
    if not (isinstance(seeds, list) and len(seeds) == 1 and type(seeds[0]) is int
            and seeds[0] == seed):  # fmt: skip
        problems.append(f"its seeds {seeds!r} are not [{seed}]")
    if not problems and name != f"c{candidate}-confirmation-{pairs['optimizer']}-{rate:g}-s{seed}":
        problems.append(f"its name is not c{candidate}-confirmation-{pairs['optimizer']}-"
                        f"{rate:g}-s{seed}, the run's")  # fmt: skip
    if problems:
        return f"cell {name!r}: " + "; ".join(problems)
    return arm, rate


def same_cells(found: list[tuple[str, float]], expected: list[tuple[str, float]]) -> bool:
    """The same (arm, rate) cells, each once, rates compared as the readers compare them."""
    order = sorted(found, key=lambda c: (ARM_ORDER.index(c[0]), c[1]))
    wanted = sorted(expected, key=lambda c: (ARM_ORDER.index(c[0]), c[1]))
    return len(order) == len(wanted) and all(
        a == b and math.isclose(x, y, rel_tol=RATE_TOLERANCE)
        for (a, x), (b, y) in zip(order, wanted, strict=True)
    )


def block_problems(data: Any, candidate: int, seed: int, cells: list,
                   root: Path) -> list[str]:  # fmt: skip
    """How a block's campaign file departs from the step's block at ``seed`` (module docstring,
    "The guard"): its header, its cells and the records its header names."""
    if not (
        isinstance(data, dict)
        and isinstance(data.get("name"), str)
        and isinstance(data.get("header"), dict)
        and isinstance(data.get("cells"), list)
    ):
        return ["not a launcher campaign (a name, a header and cells)"]
    header = data["header"]
    problems = []
    for key, value in {**HEADER, "candidate": candidate}.items():
        found = header.get(key, MISSING)
        if found is MISSING or found != value or type(found) is not type(value):
            what = "missing" if found is MISSING else repr(found)
            problems.append(f"its header's {key} is {what}, not {value!r}")
    other = sorted(set(header) - {*HEADER, "candidate", *NAMED})
    if other:
        problems.append(f"its header has keys no confirmation header has: {other}")
    found_cells = []
    for cell in data["cells"]:
        found = launcher_cell(cell, candidate, seed)
        if isinstance(found, str):
            problems.append(found)
        else:
            found_cells.append(found)
    if len(found_cells) == len(data["cells"]) and not same_cells(found_cells, cells):
        problems.append(f"its cells {sorted(found_cells)} are not the step's cells at {seed} "
                        f"{sorted(cells)}")  # fmt: skip
    for key, kind in (("n_record", f"n-c{candidate}"), ("budget_record", f"budget-c{candidate}")):
        if key not in header:
            if key == "n_record":
                problems.append("its header names no n record (n_record): the confirmation "
                                "rests on it")  # fmt: skip
            continue
        path = in_force(root, kind)  # the chain's link in force (a successor after a cycle)
        named = decision_path(".", path.name[: -len(".json")]).as_posix()
        if header[key] != named:
            problems.append(f"its header's {key} is {header[key]!r}, not {named}")
        elif _under_sealed(path, root):
            problems.append(f"{path} {unopened(path, root)}: not opened")
        elif not path.is_file():
            problems.append(f"its header's {key} {named} is no longer there")
        elif sha256(path) != header.get(f"{key}_sha256"):
            problems.append(
                f"its header names {key} {named} with sha256 "
                f"{header.get(f'{key}_sha256')}, and it is now {sha256(path)}"
            )
    named = header.get("versions_record")
    if not isinstance(named, str):
        problems.append("its header names no versions record (versions_record)")
    else:
        path = REPO / named
        if _under_sealed(path, REPO):
            problems.append(f"{path} {unopened(path, REPO)}: not opened")
        elif path.is_file() and sha256(path) != header.get("versions_record_sha256"):
            problems.append(f"its header names versions_record {named} with sha256 "
                            f"{header.get('versions_record_sha256')}, and it is now "
                            f"{sha256(path)}")  # fmt: skip
    return problems


def in_force(root: Path, kind: str) -> Path:
    """The link in force of an n or budget record's chain (``study3_decisions.chain``: the first
    record, or its last successor); the first record's path when there is none."""
    links = chain(root, kind)
    return Path(links[-1]["path"]) if links else decision_path(root, kind)


def check(args: argparse.Namespace) -> None:
    """Exit 0 only when the versions record a validation or confirmation campaign names holds
    for its step and candidate, the running checkout's HEAD included (module docstring,
    "check")."""
    path, root = args.campaign, args.runs_root
    if _under_sealed(path, root):
        refuse("check", f"{path} {unopened(path, root)}: no campaign is one, and it is not "
               "opened")  # fmt: skip
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        refuse("check", f"{path} cannot be read ({type(exc).__name__})")
    header = data.get("header") if isinstance(data, dict) else None
    header = header if isinstance(header, dict) else {}
    step, candidate = header.get("step"), header.get("candidate")
    if not (header.get("study") == 3 and type(header.get("study")) is int
            and step in STEPS and type(candidate) is int and candidate in (1, 2)):  # fmt: skip
        refuse("check", f"{path} is not a validation or confirmation campaign of study 3 (its "
               f"header: study {header.get('study')!r}, step {step!r}, candidate "
               f"{candidate!r}): no versions record stands for it")  # fmt: skip
    named, sha = header.get("versions_record"), header.get("versions_record_sha256")
    record = REPO / named if isinstance(named, str) and named else None
    if record is None or _under_sealed(record, REPO):
        refuse("check", f"{path}: its header names no versions record of the checkout "
               f"({named!r})")  # fmt: skip
    if not record.is_file() or sha256(record) != sha:
        now = sha256(record) if record.is_file() else "missing"
        refuse("check", f"{path}: its header names {named} with sha256 {sha}, and the checkout's "
               f"is {now}: the campaign was not written under the record that stands")  # fmt: skip
    problems = versions_problems(record, runs_root=root, repo=REPO, step=step,
                                 candidate=candidate)  # fmt: skip
    if problems:
        refuse("check", "the versions record does not hold: " + "; ".join(problems))
    print(f"{path}: the versions record {named} holds for candidate {candidate}'s {step}, the "
          "running checkout included: its launch may run")  # fmt: skip


def guard(args: argparse.Namespace) -> None:
    """Exit 0 only when the block may run now (module docstring, "The guard")."""
    candidate, seed, path, root = args.candidate, args.seed, args.campaign, args.runs_root
    cells, started = the_block(state_of(args), candidate, seed)
    if _under_sealed(path, root):
        refuse("guard", f"{path} {unopened(path, root)}: no campaign is one, and it is not "
               "opened")  # fmt: skip
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        refuse("guard", f"{path} cannot be read ({type(exc).__name__})")
    problems = block_problems(data, candidate, seed, cells, root)
    if problems:
        refuse("guard", f"{path} does not match the step's block {seed}: " + "; ".join(problems))
    problems = versions_problems(REPO / data["header"]["versions_record"], runs_root=root,
                                 repo=REPO, step="confirmation", candidate=candidate)  # fmt: skip
    if problems:
        refuse("guard", "the versions record does not hold: " + "; ".join(problems))
    what = "completes its missing cells" if started else "may start"
    print(f"block {seed} of candidate {candidate}'s confirmation {what}: {path}")


# ------------------------------------------------------------------ the records


def in_checkout(command: str, path: Path) -> str:
    """``path`` relative to the checkout (a relative path is read from it); refused outside it."""
    resolved = (path if path.is_absolute() else REPO / path).resolve()
    if not resolved.is_relative_to(REPO.resolve()):
        refuse(command, f"{path} is not in the checkout {REPO}")
    return resolved.relative_to(REPO.resolve()).as_posix()


def publish(command: str, path: Path, record: dict[str, Any]) -> None:
    """The record at ``path``, whole or not at all, and only if there is none."""
    data = (json.dumps(record, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        _publish(path, data)
    except DecisionRefused:
        refuse(command, f"{path} exists: a record is written once")
    print(path)


def head_code(command: str, *, clean: bool) -> dict[str, str]:
    """HEAD's commit and its GPU code (``code_identity.gpu_code_identity`` of that commit), as a
    record's ``code`` names them; with ``clean``, refused unless the checkout is clean."""
    state = git_state(REPO)
    if not state.get("commit") or (clean and state.get("dirty") is not False):
        what = "a clean checkout of a known commit" if clean else "a known commit"
        refuse(command, f"the record's code needs {what}, not {state}")
    identity = code_identity.gpu_code_identity(state["commit"], REPO)
    return {"commit": state["commit"], "gpu_code_sha256": identity["sha256"]}


def write_versions(args: argparse.Namespace) -> None:
    """The versions record (module docstring, "versions")."""
    if args.fixation is not None and args.runs_root is None:
        refuse("versions", "--fixation needs --runs-root: the fixation's evidence lies under "
               "RUNS_DIR")  # fmt: skip
    if (args.supersedes is None) != (args.supersedes_reason is None):
        refuse("versions", "--supersedes and --supersedes-reason go together: both or neither")
    own = in_checkout("versions", args.out)
    number = re.fullmatch(r"docs/study3-versions(?:-(\d+))?\.json", own)
    k = int(number.group(1) or 1) if number else 0
    if k < 1 or versions_name(k) != own:
        refuse("versions", f"{args.out} is not a versions record of the checkout "
               f"({versions_name(1)}, docs/study3-versions-<k>.json, k = 2, 3, ...)")  # fmt: skip
    if args.supersedes is not None:
        if k == 1:
            refuse("versions", f"{own} is the first versions record: it supersedes none")
        if in_checkout("versions", args.supersedes) != versions_name(k - 1):
            refuse("versions", f"{own} supersedes {versions_name(k - 1)}, the record before "
                   f"it, not {args.supersedes}")  # fmt: skip
    root = args.runs_root if args.runs_root is not None else runs_root()  # read with a fixation
    code = head_code("versions", clean=False)
    stale = after_cycle_code(root, code["gpu_code_sha256"],
                             candidate=fixation_candidate(args.fixation), repo=REPO)  # fmt: skip
    if stale:
        refuse("versions", "; ".join(stale))
    try:
        record = versions_record(REPO / own, runs_root=root, fixation=args.fixation,
                                 reason=args.reason, supersedes_reason=args.supersedes_reason,
                                 repo=REPO)  # fmt: skip
    except RecordRefused as exc:
        refuse("versions", str(exc))
    publish("versions", REPO / own, record)


def fixation_candidate(path: Path | None) -> int | None:
    """The candidate of the fixation record a versions record names (``--fixation``), or None
    (none named, or no fixation record's name in the checkout)."""
    if path is None:
        return None
    try:
        name = (REPO / path).resolve().relative_to(REPO.resolve()).as_posix()
    except (OSError, ValueError):
        return None
    found = _FIXATION_NAME.fullmatch(name)
    return int(found.group(2)) if found else None


def write_fixation(args: argparse.Namespace) -> None:
    """The validation fixation or the confirmation supplement (module docstring,
    "fixation")."""
    step, candidate = args.step, args.candidate
    names = fixation_chain(step, candidate, repo=REPO)
    state = after_cycle(args, candidate)
    supersedes = None
    path = REPO / fixation_name(step, candidate)
    if names and state is None:
        refuse("fixation", f"{REPO / names[-1]} exists: a fixation record is written once")
    if names:  # a stale record's successor, when the state's step names it
        k = fixation_number(names[-1]) + 1
        named_step("fixation", state, f"fixation-{step}-c{candidate}-{k}")
        path = REPO / fixation_name(step, candidate, k)
        cycle = cycle_path(args.runs_root, candidate)
        supersedes = {"path": names[-1], "sha256": sha256(REPO / names[-1]),
                      "reason": f"cycle_c{candidate}",
                      "cycle": {"path": decision_path(".", f"cycle-c{candidate}").as_posix(),
                                "sha256": sha256(cycle)}}  # fmt: skip
    elif state is not None and step == "confirmation" and (
            state["stage"], state["candidate"]) != ("confirmation", candidate):  # fmt: skip
        refuse("fixation", f"the confirmation supplement of candidate {candidate} is written once "
               "the state is at its confirmation: after the cycle, the records come in the "
               f"registered order (the state names {shown(state)})")  # fmt: skip
    if path.exists() or path.is_symlink():
        refuse("fixation", f"{path} exists: a fixation record is written once")
    code = head_code("fixation", clean=True)
    stale = after_cycle_code(args.runs_root, code["gpu_code_sha256"], candidate=candidate,
                             repo=REPO)  # fmt: skip
    if stale:
        refuse("fixation", "; ".join(stale))
    try:
        if step == "validation":
            record = validation_fixation(args.runs_root, candidate, code=code, repo=REPO,
                                         theta0=args.theta0, supersedes=supersedes)  # fmt: skip
        else:
            record = confirmation_fixation(args.runs_root, candidate, code=code,
                                           bootstrap_seed=args.bootstrap_seed,
                                           audit_seed=args.audit_seed,
                                           agreement_seed=args.agreement_seed,
                                           repo=REPO, supersedes=supersedes)  # fmt: skip
    except RecordRefused as exc:
        refuse("fixation", str(exc))
    publish("fixation", path, record)


def write_record(args: argparse.Namespace) -> None:
    """A decision record from a JSON file, through ``study3_decisions.write_decision``."""
    source = args.source
    if _under_sealed(source, args.runs_root):
        refuse("record", f"{source} {unopened(source, args.runs_root)}: no record is one, and it "
               "is not opened")  # fmt: skip
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        refuse("record", f"{source} cannot be read ({type(exc).__name__})")
    kind = args.kind
    family, candidate = kind_of(kind)
    state = after_cycle(args, candidate) if family in ("n", "budget") else None
    if state is not None:  # after the cycle: the first record, or the next successor
        if chain(args.runs_root, kind):
            kind = next_successor(args.runs_root, kind)
            try:
                found = json.loads(json.dumps(record, allow_nan=False))
            except (TypeError, ValueError) as exc:
                refuse("record", f"{kind}: not a JSON record ({exc})")
            problems = _problems(args.runs_root, kind, found, REPO)
            if problems:  # its content first: after the confirmation started, N stands
                refuse("record", f"{kind}: not written, {len(problems)} problem(s): "
                       + "; ".join(problems))  # fmt: skip
        named_step("record", state, kind)
    try:
        path = write_decision(args.runs_root, kind, record, repo=REPO)
    except DecisionRefused as exc:
        refuse("record", str(exc))
    print(path)


def closures_named(step: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """The fields of every closure record ``kind`` the step names: a held run's decision step,
    or the ways out of a technical step."""
    if step.get("kind") == "decision" and step.get("record") == kind:
        return [dict(step["fields"])]
    return [dict(way["closure"]["fields"]) for way in step.get("ways_out") or []
            if (way.get("closure") or {}).get("record") == kind]  # fmt: skip


def write_closure(args: argparse.Namespace) -> None:
    """The closure record whose fields the step gives (module docstring, "closure")."""
    kind = f"closure-c{args.candidate}-{args.arm}"
    state = state_of(args)
    named = closures_named(state["step"], kind)
    if args.run_id is not None:
        named = [fields for fields in named if fields.get("run_id") == args.run_id]
    if not named:
        run = "" if args.run_id is None else f" for run {args.run_id}"
        refuse("closure", f"the step names no closure record {kind}{run}: the state names "
               f"{shown(state)}")  # fmt: skip
    if len(named) > 1:
        refuse(
            "closure",
            f"the step names {len(named)} closures of {kind} (runs "
            f"{[fields['run_id'] for fields in named]}): name the run with --run-id",
        )
    record = {"kind": kind, "state": "failed",
              "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "code": head_code("closure", clean=False), **named[0]}  # fmt: skip
    try:
        path = write_decision(args.runs_root, kind, record, repo=REPO)
    except DecisionRefused as exc:
        refuse("closure", str(exc))
    print(path)


# ------------------------------------------------------------------ the engineering cycle


def write_cycle(args: argparse.Namespace) -> None:
    """Candidate N's engineering cycle outside the qualification (module docstring, "cycle")."""
    root, candidate = args.runs_root, args.candidate
    state = git_state(REPO)
    if not state.get("commit") or state.get("dirty") is not False:
        refuse("cycle", f"the fix is not a clean checkout of a known commit ({state}): {NO_CYCLE}")
    identity = code_identity.gpu_code_identity(state["commit"], REPO)
    code = {"commit": state["commit"], "gpu_code_sha256": identity["sha256"]}
    theta0 = args.theta0 if args.theta0 is not None else root / THETA0_RUN
    settings = load_config("experiments/qualification")
    try:
        seen = journal_seen(root)  # pinned before the records are read
    except CycleRefused as exc:
        refuse("cycle", "; ".join(exc.reasons))
    unread = journal_unread(root)  # what every reader below would read short of: named first
    if unread:
        refuse("cycle", "the launch journal is not whole: " + "; ".join(unread))
    records = read_records(root, repo=REPO, expected_code=None, theta0=theta0, settings=settings)
    # the code in force before N's cycle: the existing chain's end, N's own supersession records
    # aside (with none of them, the chain's end; with one, the cycle is recorded already)
    old = chain_end(replace(records, cycles={c: found for c, found in records.cycles.items()
                                             if c < candidate}))  # fmt: skip
    if old is None:
        refuse("cycle", "the chain of engineering cycles has no start (no valid engineering "
               "record): no code is in force for a fix to leave")  # fmt: skip
    if code["gpu_code_sha256"] == old:
        refuse("cycle", f"the fix's GPU code is the code in force ({old}): {NO_CYCLE}")
    recorded = sorted((root / "qualification" / f"c{candidate}").glob("supersede-*.json"))
    if recorded:
        refuse("cycle", f"candidate {candidate}'s engineering cycle is recorded already "
               f"({recorded[0]}): one engineering cycle per candidate")  # fmt: skip
    campaign, case = engineering_campaign(root)
    path = cycle_path(root, candidate)
    if path.exists() or path.is_symlink():  # its bridge missing: written from the record alone
        record, problems = read_decision(root, f"cycle-c{candidate}", repo=REPO)
        if record is None:
            refuse("cycle", f"candidate {candidate}'s engineering cycle is recorded already "
                   f"({path}), and its record is not valid: " + "; ".join(problems))  # fmt: skip
        claims = claims_of(args)
        asked = {"diagnosis": args.diagnosis, "fix": args.fix, "code": code,
                 "stages": {name: {"action": c.action, "arms": list(c.arms),
                                   "justification": c.justification}
                            for name, c in claims.items()}}  # fmt: skip
        other = [key for key, value in asked.items() if record.get(key) != value]
        if other:
            refuse("cycle", f"candidate {candidate}'s engineering cycle is recorded already "
                   f"({path}), with another {', '.join(other)}: a cycle record is written once, "
                   "and run again with its own arguments the command writes only its missing "
                   "bridge")  # fmt: skip
        try:
            bridge = rebuild_bridge(root, record, fix_identity=identity,
                                    run_codes=records.run_codes, settings=settings,
                                    host=socket.gethostname())  # fmt: skip
        except CycleRefused as exc:
            refuse("cycle", "; ".join(exc.reasons))
        return finish_cycle(args, record, bridge, campaign, case)
    found = advance(records, expected_code=old)  # the right to a cycle, on the existing chain
    step = found["step"]
    ways = [way["cycle"] for way in step.get("ways_out") or []
            if isinstance(way, dict) and "cycle" in way]  # fmt: skip
    if (found["problems"] or found.get("candidate") != candidate
            or step.get("reason") != ENGINEERING_CYCLE or not ways):  # fmt: skip
        why = "; ".join(found["problems"]) or shown(found)
        refuse("cycle", f"on the existing chain (the code in force, {old}) the state is not "
               f"candidate {candidate}'s engineering cycle: {why}")  # fmt: skip
    broken = journal_problems(root)
    if broken:
        refuse("cycle", "the launch journal is not whole: " + "; ".join(broken))
    fields, fixation = ways[0]["fields"], ways[0].get("fixation")
    claims = claims_of(args)
    reruns = {step_: claim.arms for name, claim in claims.items() if claim.action == "rerun"
              for step_ in STAGE_STEPS[name]}  # fmt: skip
    latest: dict[Any, int] = {}
    for e in records.endpoints:
        latest[e.label] = max(latest.get(e.label, 0), e.attempt or 0)
    waiting = sorted(e.run_id for e in records.endpoints if e.cell.candidate == candidate
                     and e.cell.step in reruns and e.arm in reruns[e.cell.step]
                     and e.state in ("pending", "recovered")
                     and (e.attempt or 0) >= latest.get(e.label, 0))  # fmt: skip
    if waiting:
        refuse("cycle", f"runs of a stage the claims rerun have not ended (pending, or "
               f"recovered with no later attempt): {waiting[:10]}: the cycle waits for "
               "them")  # fmt: skip
    if fixation is not None and "search" in claims and claims["search"].action == "rerun":
        refuse("cycle", f"the validation fixation {fixation} exists: the search is never rerun "
               "after it: search=rerun is refused (search=reuse, with its "
               "justification)")  # fmt: skip
    if confirmation_started(root, candidate) and any(
            name in claims and claims[name].action == "rerun"
            for name in ("search", "validation")):  # fmt: skip
        refuse("cycle", "after the confirmation has started, validation is not reopened without "
               "an explicit prospective amendment: validation=rerun and search=rerun are "
               "refused")  # fmt: skip
    ref = None if fixation is None else {"path": fixation, "sha256": sha256(REPO / fixation),
                                         "root": "repo"}  # fmt: skip
    try:
        before = pre_cycle_records(root, candidate, journal=seen, without=None, theta0=theta0,
                                   repo=REPO, settings=settings)  # fmt: skip
        record, bridge = derive_cycle(
            root, candidate, records=before, marks=fields["marks"], stage=fields["stage"],
            closed_arms=[arm for arm in ARM_ORDER if arm not in found["active"]["arms"]],
            old_code=old, fixation=ref, claims=claims, diagnosis=args.diagnosis, fix=args.fix,
            code=code, fix_identity=identity, settings=settings, repo=REPO,
            written_at=datetime.now(UTC).isoformat(timespec="seconds"),
            host=socket.gethostname(), journal=seen)  # fmt: skip
    except CycleRefused as exc:
        refuse("cycle", "; ".join(exc.reasons))
    finish_cycle(args, record, bridge, campaign, case)


def claims_of(args: argparse.Namespace) -> dict[str, Any]:
    """The claims and their justifications (``study3_cycle.parse_claims``), or refused."""
    try:
        return parse_claims(args.stage or [], args.justification or [])
    except CycleRefused as exc:
        refuse("cycle", "; ".join(exc.reasons))


def finish_cycle(args: argparse.Namespace, record: dict, bridge: dict, campaign: Path,
                 case: str) -> None:  # fmt: skip
    """The cycle printed (--dry-run), or published, its paths and its journal repair printed."""
    if args.dry_run:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        print(json.dumps(bridge, indent=1, ensure_ascii=False))
        return
    try:
        path, bridge_path = publish_cycle(args.runs_root, args.candidate, record, bridge,
                                          repo=REPO)  # fmt: skip
    except CycleRefused as exc:
        refuse("cycle", "; ".join(exc.reasons))
    print(path)
    print(bridge_path)
    print(repair_command(campaign, args.candidate, case, bridge["old_record"]["path"],
                         record["diagnosis"], record["fix"]))  # fmt: skip


def repair_command(campaign: Path, candidate: int, case: str, record: str, diagnosis: str,
                   fix: str) -> str:  # fmt: skip
    """qualify.py's journal repair of the cycle (``supersede`` with the cycle record), to run
    where RUNS_DIR is: it writes the bridge's journal lines."""
    words = ["supersede", "--campaign", str(campaign), "--candidate", str(candidate),
             "--engineering", "--case", case, "--record", record, "--diagnosis", diagnosis,
             "--fix", fix]  # fmt: skip
    return 'project-run "$SRC" python scripts/qualify.py ' + " ".join(map(shlex.quote, words))


def engineering_campaign(root: Path) -> tuple[Path, str]:
    """The engineering campaign file the journal repair takes and its first case
    (``s<seed>/<optimizer>-<rate>``): ``RUNS_DIR/campaigns/<name>.yaml``, ``<name>`` the
    campaign of the journal's first start of the engineering stage. Refused, naming where it
    looked, when there is none."""
    first = None
    for log in journal_lines(root)[0]:
        for line in log.lines:
            found = line.record or {}
            header = found.get("header") if isinstance(found.get("header"), dict) else {}
            if (found.get("event") == "start" and header.get("study") == 3
                    and header.get("step") == "engineering"):  # fmt: skip
                key = (str(found.get("time") or ""), log.path, line.line)
                if first is None or key < first[0]:
                    first = (key, PurePosixPath(log.path).stem)
    where = root / "campaigns"
    if first is None:
        refuse("cycle", f"the launch journal holds no start of the engineering stage: no "
               f"engineering campaign file under {where}/ can be named for the journal repair "
               "(qualify.py supersede --campaign)")  # fmt: skip
    path = where / f"{first[1]}.yaml"
    if _under_sealed(path, root):
        refuse("cycle", f"{path} {unopened(path, root)}: not opened")
    if not path.is_file():
        refuse("cycle", f"the engineering campaign file {path} is not there (the campaign of the "
               f"journal's first engineering start, under {where}/): the journal repair takes "
               "it (qualify.py supersede --campaign)")  # fmt: skip
    try:
        cell = yaml.safe_load(path.read_text(encoding="utf-8"))["cells"][0]
        pairs = dict(o.split("=", 1) for o in cell["overrides"])
        case = f"s{int(cell['seeds'][0])}/{pairs['optimizer']}-{float(pairs['optimizer.lr']):g}"
    except (OSError, ValueError, yaml.YAMLError, KeyError, IndexError, TypeError) as exc:
        refuse("cycle", f"{path}: its first cell names no optimizer, rate and seed "
               f"({type(exc).__name__}): no engineering campaign")  # fmt: skip
    return path, case


# ------------------------------------------------------------------ the command line


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    shown_ = commands.add_parser("progress", help="print the state as JSON")
    checked = commands.add_parser("guard", help="the confirmation's between-block guard")
    held = commands.add_parser("check", help="the versions record a validation or confirmation "
                               "campaign names, before its launch")  # fmt: skip
    versions = commands.add_parser("versions", help="write a versions record")
    fixed = commands.add_parser("fixation", help="write a validation fixation or a confirmation "
                                "supplement")  # fmt: skip
    decided = commands.add_parser("record", help="write a decision record from a JSON file")
    closed = commands.add_parser("closure", help="write the closure record the step names")
    cycled = commands.add_parser("cycle", help="write the engineering cycle of technical marks "
                                 "outside the qualification")  # fmt: skip
    runs = "RUNS_DIR: the run records, the launch journal and the decision records"
    for command in (shown_, checked, held, fixed, decided, closed, cycled):
        command.add_argument("--runs-root", type=Path, required=True, help=runs)
    for command in (shown_, checked, fixed, closed, cycled):
        command.add_argument("--theta0", type=Path,
                             help="θ₀'s evaluation (study 2's under RUNS_DIR by default; "
                             "fixation: validation only)")  # fmt: skip
    for command in (checked, fixed, closed, cycled):
        command.add_argument("--candidate", type=int, choices=(1, 2), required=True)
    cycled.add_argument("--diagnosis", required=True, help="the diagnosis, one line")
    cycled.add_argument("--fix", required=True, help="the fix, one line")
    cycled.add_argument("--stage", action="append", metavar="STAGE=reuse|rerun[:ARMS]",
                        help="a stage's claim, one per stage started")  # fmt: skip
    cycled.add_argument("--justification", action="append", metavar="STAGE=TEXT",
                        help="a claim's justification, one line")  # fmt: skip
    cycled.add_argument("--dry-run", action="store_true", help="print the records, write none")
    checked.add_argument("--seed", type=int, required=True, help="the block's seed")
    checked.add_argument("--campaign", type=Path, required=True, help="the block's campaign file")
    held.add_argument("--campaign", type=Path, required=True,
                      help="the validation or confirmation campaign file")  # fmt: skip
    versions.add_argument("--out", type=Path, required=True,
                          help="docs/study3-versions.json, or docs/study3-versions-<k>.json for "
                          "a later record")  # fmt: skip
    versions.add_argument("--reason", required=True, help="why the record is written (an id)")
    versions.add_argument("--fixation", type=Path,
                          help="the fixation record its runs are under (committed)")  # fmt: skip
    versions.add_argument("--runs-root", type=Path, help=f"{runs} (needed with --fixation)")
    versions.add_argument("--supersedes", type=Path, help="a later record: the one before it")
    versions.add_argument("--supersedes-reason",
                          help="a later record: why the one before it no longer serves (an "
                          "id)")  # fmt: skip
    fixed.add_argument("step", choices=STEPS)
    for name in ("--bootstrap-seed", "--audit-seed", "--agreement-seed"):
        fixed.add_argument(name, type=int, help="confirmation: an integer in [0, 2³²)")
    decided.add_argument("kind", choices=RECORDS)
    decided.add_argument("--from", dest="source", type=Path, required=True,
                         help="the record (JSON)")  # fmt: skip
    closed.add_argument("--arm", choices=tuple(OPTIMIZERS), required=True)
    closed.add_argument("--run-id", help="the closed run, when the step names closures of several")
    args = parser.parse_args(argv)
    if args.command == "fixation":
        seeds = (args.bootstrap_seed, args.audit_seed, args.agreement_seed)
        if args.step == "validation" and any(seed is not None for seed in seeds):
            fixed.error("--bootstrap-seed, --audit-seed and --agreement-seed go with the "
                        "confirmation supplement")  # fmt: skip
        if args.step == "confirmation" and args.theta0 is not None:
            fixed.error("--theta0 goes with the validation fixation")
        if args.step == "confirmation" and None in seeds:
            fixed.error("the confirmation supplement needs --bootstrap-seed, --audit-seed and "
                        "--agreement-seed")  # fmt: skip
    commands_ = {
        "progress": print_progress,
        "guard": guard,
        "check": check,
        "versions": write_versions,
        "fixation": write_fixation,
        "record": write_record,
        "closure": write_closure,
        "cycle": write_cycle,
    }
    commands_[args.command](args)


if __name__ == "__main__":
    main()
