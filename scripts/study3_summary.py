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
  the budget record (when named), each the candidate's under RUNS_DIR with the sha256 the header
  gives, and the versions record, in the checkout, of its named sha256;
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
record is then committed alone, before the next one is written. The two files the validation
fixation pins are committed before it: the spread's decision script scripts/study3_spread.py
(with every file of its import closure under src/zero_field_refusal/) and the control list
docs/study3-judge-controls.json; the fixation pins both as they are at its commit.

**record** writes a decision record (engineering, simulation, n-c<N>, budget-c<N>) from a JSON
file through ``study3_decisions.write_decision``, which validates it against the evidence it
names and the records it answers: written once, never over another; a refusal prints its reasons.

**closure** writes the closure record closure-c<N>-<ARM> with exactly the fields the state's step
gives: a held run's decision step, or the way out of a run under another commit (its last end
line, or its start line, with its uncounted reason). Any other technical mark outside the
qualification stops the route for the researcher's decision (``engineering_route_pending``): the
step names no closure for it, and none is written. It refuses when the step names no closure of
that candidate and arm, and, when the step names several (two runs of the arm), unless --run-id
chooses the run. The record's state is ``failed`` (a terminal technical
outcome) and its code HEAD's commit and that commit's GPU code.

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
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import yaml

from zero_field_refusal import code_identity
from zero_field_refusal.analysis.study3_decisions import (  # its sealed-path rule, its
    DecisionRefused,  # write-once publication
    _publish,
    _under_sealed,
    decision_path,
    write_decision,
)
from zero_field_refusal.analysis.study3_progress import progress
from zero_field_refusal.analysis.study3_records import (
    ARMS,
    BATCH,
    RATE_TOLERANCE,
    SEEDS,
    STEP_ROLES,
    sealed_why,
)
from zero_field_refusal.analysis.study3_search import ARM_ORDER
from zero_field_refusal.analysis.study3_versions import (
    RecordRefused,
    confirmation_fixation,
    fixation_name,
    validation_fixation,
    versions_name,
    versions_problems,
    versions_record,
)
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
    return progress(args.runs_root, repo=REPO, expected_code=expected, theta0=args.theta0)


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
        named, path = decision_path(".", kind).as_posix(), decision_path(root, kind)
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
    try:
        record = versions_record(REPO / own, runs_root=root, fixation=args.fixation,
                                 reason=args.reason, supersedes_reason=args.supersedes_reason,
                                 repo=REPO)  # fmt: skip
    except RecordRefused as exc:
        refuse("versions", str(exc))
    publish("versions", REPO / own, record)


def write_fixation(args: argparse.Namespace) -> None:
    """The validation fixation or the confirmation supplement (module docstring,
    "fixation")."""
    step, candidate = args.step, args.candidate
    path = REPO / fixation_name(step, candidate)
    if path.exists() or path.is_symlink():
        refuse("fixation", f"{path} exists: a fixation record is written once")
    code = head_code("fixation", clean=True)
    try:
        if step == "validation":
            record = validation_fixation(args.runs_root, candidate, code=code, repo=REPO,
                                         theta0=args.theta0)  # fmt: skip
        else:
            record = confirmation_fixation(args.runs_root, candidate, code=code,
                                           bootstrap_seed=args.bootstrap_seed,
                                           audit_seed=args.audit_seed,
                                           agreement_seed=args.agreement_seed,
                                           repo=REPO)  # fmt: skip
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
    try:
        path = write_decision(args.runs_root, args.kind, record, repo=REPO)
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
    runs = "RUNS_DIR: the run records, the launch journal and the decision records"
    for command in (shown_, checked, held, fixed, decided, closed):
        command.add_argument("--runs-root", type=Path, required=True, help=runs)
    for command in (shown_, checked, fixed, closed):
        command.add_argument("--theta0", type=Path,
                             help="θ₀'s evaluation (study 2's under RUNS_DIR by default; "
                             "fixation: validation only)")  # fmt: skip
    for command in (checked, fixed, closed):
        command.add_argument("--candidate", type=int, choices=(1, 2), required=True)
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
    }
    commands_[args.command](args)


if __name__ == "__main__":
    main()
