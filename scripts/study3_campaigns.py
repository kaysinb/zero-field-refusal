"""Write study 3's launcher campaigns from the state of its records, and the driver of fresh seeds'
streams.

    python scripts/study3_campaigns.py KIND --runs-root DIR --candidate N --out FILE
        [--seeds S [S ...]] [--theta0 RUN_DIR] [--checkpoints DIR]
    python scripts/study3_campaigns.py confirmation --runs-root DIR --candidate N --out DIR
        [--seeds S [S ...] | --blocks FIRST LAST] [--theta0 RUN_DIR]
    python scripts/study3_campaigns.py prep --seeds S [S ...] --commit SHA --out FILE

KIND is ``engineering``, ``search-round0``, ``search-round1``, ``numerical``, ``repair`` or
``validation``. The rules are docs/study3-preregistration.md's: "Candidates and rate search",
"Numerical qualification", "Validation, confirmation and stopping", "Seeds and order" and
amendments 1-3. A campaign is YAML as ``launch.load_campaign`` reads it.

**The step decides.** The state is ``analysis.study3_progress.progress`` of --runs-root (θ₀'s
evaluation from --theta0, study 2's under RUNS_DIR by default), its expected code this checkout's
GPU code (``code_identity.gpu_code_identity``). A kind is written only when the state's step is
that campaign of candidate N (--candidate): ``search-round0`` and ``search-round1`` the search's
rounds 0 and 1, ``numerical``, ``repair``, ``validation`` and ``confirmation`` as ``progress``
names them; ``engineering`` while the step is the engineering record, which rests on that
campaign's runs. Anything else is refused, non-zero, with the step the state names printed:
another kind or candidate, a blocked, pending or decision step, a terminal stage.

**Cells.** Exactly the step's cells (the engineering stage's: those the route prescribes, every arm
at candidate 1's two starting rates on 10000-10003), one launcher cell per run, block by block
in seed order. --seeds takes the step's first blocks (a block is a seed), in order: it never adds
a block, skips one or reorders them, so a block of the validation or the confirmation is issued
whole (six cells, four once P is closed; for a block already started, the cells it still lacks,
as the step names them). The qualification's campaigns (engineering, numerical, repair) are
written whole, one file per step, and --seeds is refused for them: ``scripts/qualify.py`` reads a
campaign as its whole step (each FA rate's same-path control lies on the first numerical seed, so
a later part alone is no qualification campaign), and a stopped qualification goes on by running
its driver again on the same file, which skips the complete cases. A label that a ``held`` line
of the launch journal or a supersession record holds (``study3_records.read_holds``), or a cell
whose run a closure record closed (``study3_decisions.closures``), is never issued again: a step
naming one is refused.

**Names.** The launcher keeps one log per campaign name (``RUNS_DIR/launch/<name>.jsonl``), and
the validation fixation pins the numerical and repair logs by sha256, so that a later line breaks
it for good. Each file therefore gets a name no campaign used, ``study3-c<N>-<kind>-<UTC stamp>``
(a confirmation block's ``study3-c<N>-confirmation-s<seed>-<UTC stamp>``), and a name whose log
exists is refused. A launcher cell holds one run, and its name is fixed by that run,
``c<N>-<step>[-r<round>]-<optimizer>-<rate>-s<seed>``: no two runs of the study share a cell name,
and a run gets the same name, and the same label (``<cell>/s<seed>``), whatever file issues it.
The launcher reads a cell's recovery round by its cell name, and a run's own rounds by its label,
from every log, so a round opened for one run never gives another run, in a later file of the
step, a second attempt at --max-attempts 1 (which the readers would refuse as two attempts of one
cell); and a hold by label keeps its meaning across files.

**The header** (flat and plain, as the launcher logs it with every start, and as
``study3_records`` binds a run by it): ``study`` 3, ``candidate``, ``step`` and ``round`` (0 or 1
for the search, null otherwise); the decision records the campaign rests on, each by its path
under RUNS_DIR and the sha256 of its file now (``<key>`` and ``<key>_sha256``, which
``study3_decisions.immutable`` protects): the search names the engineering record
(``engineering_record``), the validation the simulation record (``simulation_record``), the
confirmation the n record (``n_record``) and the budget record (``budget_record``) whenever one
is written, as it must be when ``budget_reason`` says N needs one; each must be valid and
passed. The validation and the confirmation also name the versions record that stands (the last
of docs/study3-versions.json, docs/study3-versions-2.json, ...: ``versions_record`` and
``versions_record_sha256``), once ``study3_versions.versions_problems`` with the step and the
candidate finds no problem in it; a problem refuses, printed.

**Configs.** Every run at B = 20. Search: experiments/study3_k_only, ``study3.role=search``.
Engineering, numerical and repair: experiments/study3_k_only, ``study3.role=engineering`` or
``numerical``, and ``tracking.endpoint_checkpoint`` the one absolute directory where
``scripts/qualify.py`` reads the launcher's step-25 checkpoint for the bridge, on the Pod's
container disk (--checkpoints, ``CHECKPOINTS`` by default). Validation and confirmation:
experiments/study3 with their role.

**Running.** Nothing is run here; the command is printed. Every campaign runs with ``--max-attempts
1`` (one technical attempt uses the launcher's allowance; a rerun is a recovery round of the run's
own, one attempt per documented incident: ``python scripts/launch.py FILE --only LABEL
--max-attempts 1 --recovery REASON --incident-time UTC --cause TEXT --evidence REF``, its reason one
that no earlier round of the label gave, zero_field_refusal.launch, "Recovery rounds"): search and
validation as ``python scripts/launch.py FILE --max-attempts 1``; engineering, numerical and repair
through ``python scripts/qualify.py driver --campaign FILE --candidate N [--engineering] --commit
SHA --out DRIVER --kept /workspace/kept-checkpoints`` (``--kept``: the folder on the Pod's volume
a technical case's checkpoints are exported to), which passes it itself. The validation's launch
is preceded by its check, printed first: ``python scripts/study3_summary.py check --runs-root DIR
--campaign FILE``, which exits 0 only while the versions record its header names holds for the
running checkout (a run under another commit does not count, and the launcher never reruns it);
it is run again, from the same checkout, before every resumption of the file and every recovery
round of one of its runs. Campaign files belong outside the checkout (under $RUNS_DIR/campaigns/,
say): the versions record holds only for a clean checkout.

**The confirmation** is written into --out DIR: one campaign file per block, in seed order, and
the driver ``run.sh``. Before each block, the driver checks that the block's file is the one
written (its sha256), then runs the between-block guard, ``python scripts/study3_summary.py guard
--runs-root DIR --candidate N --seed S --campaign FILE`` (the guard reads the versions record the
block's header names; with --theta0 given to the writer, every guard call passes the same
``--theta0``, so that the guard reads the θ₀ the writer read), and stops at its first non-zero
exit; otherwise it runs ``python
scripts/launch.py FILE --max-attempts 1``, stopping when the launcher fails. It never runs two
blocks at once. Its blocks: the step's (the next block, or the cells a started block
still lacks: those are completed first, alone); with --blocks FIRST LAST (seeds) or --seeds, the
step's next block and those after it in seed order up to the cap N + max(10, ⌈0.12·N⌉), each
whole, every one started only when its guard allows it. The driver sources the Pod's
environment (``/etc/rp_environment``, ``/root/zfr-env.sh``) where those files exist, as the other
Pod drivers do, pins ``RUNS_DIR`` to the runs directory the guard reads, so that the launcher
writes where the guard looks, and runs ``python`` from the environment it is started in (on a
Pod: ``project-run "$SRC" bash DIR/run.sh``).

**prep** writes the bash driver that generates the training and preconditioner streams of fresh
study-3 seeds at B = 20 (``analysis.campaign_prep.streams_driver``) from the source checkout of
commit SHA on a Pod, each seed's under the config its runs locate them with (its role's:
experiments/study3_k_only for engineering, search and numerical seeds, experiments/study3 for
validation and confirmation seeds). The seeds are registered study-3 seeds, each named once.

FILE and DIR must not exist; every file is written in UTF-8. Nothing here opens a response, a
harm label or a judge output: the state holds numbers, ids, paths and hashes, and only those are
printed; a record that resolves under a sealed folder (``study3_records.SEALED``) is refused
before it is hashed.
"""

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

import yaml

from zero_field_refusal import code_identity
from zero_field_refusal.analysis.campaign_prep import comment, revision, streams_driver
from zero_field_refusal.analysis.study3_decisions import (  # its one sealed-path rule
    _under_sealed,
    budget_reason,
    closures,
    decision_path,
    read_decision,
    start_lines,
)
from zero_field_refusal.analysis.study3_progress import TERMINAL, progress
from zero_field_refusal.analysis.study3_records import (
    ARMS,
    BATCH,
    EXPERIMENT,
    K_ONLY_ROLES,
    RATE_TOLERANCE,
    SEALED_WHY,
    SEEDS,
    STEP_ROLES,
    UPDATES,
    read_holds,
    sealed_why,
    step_key,
)
from zero_field_refusal.analysis.study3_search import ARM_ORDER
from zero_field_refusal.analysis.study3_versions import versions_name, versions_problems
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.tracking import git_state

REPO = REPO_ROOT  # the checkout: its versions records, the decision records' repo evidence
# The launcher's step-25 checkpoint for the bridge: one absolute directory on the Pod's container
# disk (scripts/qualify.py's cases; docs/operations.md, "Study 3: numerical qualification")
CHECKPOINTS = Path("/root/ckpt")
# A kind's step and round, as a campaign header names them
KINDS = {"engineering": ("engineering", None), "search-round0": ("search", 0),
         "search-round1": ("search", 1), "numerical": ("numerical", None),
         "repair": ("repair", None), "validation": ("validation", None),
         "confirmation": ("confirmation", None)}  # fmt: skip
QUALIFICATION = ("engineering", "numerical", "repair")  # run through scripts/qualify.py driver
KEPT = "/workspace/kept-checkpoints"  # the driver's --kept: the Pod's volume
VERSIONED = ("validation", "confirmation")
# The decision records a step's campaign rests on, by header key
RECORDS = {"search": {"engineering_record": "engineering"},
           "validation": {"simulation_record": "simulation"}}  # fmt: skip
REGULAR, K_ONLY = "experiments/study3", "experiments/study3_k_only"
OPTIMIZERS = {arm: name for name, arm in ARMS.items()}
STAMP = "$(date -u +%FT%TZ)"
WRITER = "scripts/study3_campaigns.py"

Cell = tuple[str, float]  # an arm and its rate
Block = tuple[int, list[Cell]]  # a seed and its cells


def stamp() -> str:
    """The UTC time stamp that makes a campaign's name new."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def refuse(kind: str, why: str) -> NoReturn:
    sys.exit(f"study3_campaigns.py {kind}: {why}")


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


# ------------------------------------------------------------------ the step and its blocks


def the_step(kind: str, candidate: int, state: dict[str, Any]) -> list[Block]:
    """The step's blocks, when the state names ``kind``'s campaign of ``candidate`` (module
    docstring, "The step decides"); refused otherwise."""
    stage, step = state["stage"], state["step"]
    if stage in TERMINAL or stage == "done":
        refuse(kind, f"the stage is terminal ({stage}: {state.get('outcome')}): no campaign "
               f"follows; the state names {shown(state)}")  # fmt: skip
    if step["kind"] == "blocked":
        refuse(kind, f"the step is blocked: no campaign is written; the state names "
               f"{shown(state)}")  # fmt: skip
    if kind == "engineering":
        named = (
            stage == "engineering"
            and step["kind"] == "decision"
            and step.get("record") == "engineering"
        )
    else:
        named = (step["kind"] == "campaign" and step.get("campaign") == kind
                 and (step.get("step"), step.get("round")) == KINDS[kind])  # fmt: skip
    if not named:
        refuse(kind, f"the step is not {kind}: the state names {shown(state)}")
    if state["candidate"] != candidate:
        refuse(kind, f"the step is candidate {state['candidate']}'s, not candidate {candidate}'s: "
               f"the state names {shown(state)}")  # fmt: skip
    if kind == "engineering":
        cells = state["prescribed"].get(step_key(candidate, "engineering")) or []
    else:
        cells = step["cells"]
    return blocks_of(cells)


def blocks_of(cells) -> list[Block]:
    """(arm, rate, seed) cells as blocks in seed order, each block's cells in the arms' order,
    then by rate."""
    by_seed: dict[int, list[Cell]] = defaultdict(list)
    for arm, rate, seed in sorted(cells, key=lambda c: (c[2], ARM_ORDER.index(c[0]), c[1])):
        by_seed[seed].append((arm, float(rate)))
    return list(by_seed.items())


def authorized(kind: str, candidate: int, state: dict[str, Any],
               own: list[Block]) -> tuple[list[Block], bool]:  # fmt: skip
    """The blocks a selection may take, in order, and whether the step completes started
    blocks. For the confirmation's next block: it and the blocks after it in seed order up to the
    cap, each whole; otherwise the step's own blocks."""
    if kind != "confirmation":
        return own, False
    confirmation = state.get("confirmation") or {}
    started = set(confirmation.get("started") or [])
    if any(seed in started for seed, _ in own) or len(own) != 1:
        return own, True
    (seed, cells), arms = own[0], state["active"]["arms"]
    if sorted({arm for arm, _ in cells}) != sorted(arms) or len(cells) != 2 * len(arms):
        refuse(kind, f"the next block {seed} is not whole ({cells}): every active arm "
               f"{arms} at its two rates")  # fmt: skip
    reserved = list(SEEDS["confirmation"][candidate])
    cap = confirmation.get("cap")
    if not isinstance(cap, int) or seed not in reserved[:cap]:
        refuse(kind, f"block {seed} is not within the cap of {cap} blocks: the state names "
               f"{shown(state)}")  # fmt: skip
    return [(s, list(cells)) for s in reserved[reserved.index(seed) : cap]], False


def select(kind: str, blocks: list[Block], count: int, seeds: list[int] | None,
           span: list[int] | None, completing: bool) -> list[Block]:  # fmt: skip
    """The first blocks of ``blocks`` that --seeds or --blocks name (``count`` of them by
    default): a selection never adds a block, skips one or reorders them."""
    order = [seed for seed, _ in blocks]
    if not order:
        refuse(kind, "the step names no cells")
    later = (" (a started block's missing cells are completed first, alone; the next blocks are "
             "written once they are issued)" if completing else "")  # fmt: skip
    if span is not None:
        first, last = span
        if first != order[0]:
            refuse(kind, f"--blocks {first} {last} skips the step's first block {order[0]}: a "
                   "selection is the step's first blocks, in seed order")  # fmt: skip
        if last not in order:
            refuse(kind, f"--blocks {first} {last}: {last} is not among the blocks this step "
                   f"authorizes ({order[0]}-{order[-1]}): a selection never adds a "
                   f"block{later}")  # fmt: skip
        wanted = order[: order.index(last) + 1]
    elif seeds is not None:
        unknown = [s for s in seeds if s not in order]
        if unknown:
            refuse(kind, f"--seeds {unknown}: not among the blocks this step authorizes "
                   f"({order[0]}-{order[-1]}): a selection never adds a block{later}")  # fmt: skip
        places = [order.index(s) for s in seeds]
        if places != sorted(places):
            refuse(kind, f"--seeds {seeds} is not in the step's order {order[: len(seeds)]}: a "
                   "selection never reorders the blocks")  # fmt: skip
        if places != list(range(len(places))):
            skipped = [s for s in order[: places[-1]] if s not in seeds]
            refuse(kind, f"--seeds {seeds} skips {skipped}: a selection is the step's first "
                   "blocks, in order")  # fmt: skip
        wanted = list(seeds)
    else:
        wanted = order[:count]
    return [block for block in blocks if block[0] in wanted]


def forbidden(root: Path, candidate: int, step: str, round_: int | None,
              blocks: list[Block]) -> list[str]:  # fmt: skip
    """Why a cell of ``blocks`` may not be issued: its label held in the launch journal, or its
    run (or label) closed by a closure record."""
    holds, _ = read_holds(root)  # their problems block the state already
    held = {h["run"]: h["run_id"] for h in holds if isinstance(h.get("run"), str)
            and h.get("experiment") in (None, EXPERIMENT)}  # fmt: skip
    starts = start_lines(root)
    closed = []
    for c in (1, 2):
        for kind, entry in closures(root, c, repo=REPO).items():
            if entry["record"] is not None:
                record = entry["record"]
                label = (starts.get(record["run_id"]) or {}).get("run")
                closed.append((kind, record, label))
    found = []
    for seed, cells in blocks:
        for arm, rate in cells:
            label = f"{cell_name(candidate, step, round_, arm, rate, seed)}/s{seed}"
            if label in held:
                found.append(f"{label}: held in the launch journal (run {held[label]}, an "
                             "outcome the engineering cycle could not resolve): never issued "
                             "again")  # fmt: skip
            for kind, record, closed_label in closed:
                same = (record["candidate"] == candidate and record["arm"] == arm
                        and record["seed"] == seed
                        and math.isclose(record["rate"], rate, rel_tol=RATE_TOLERANCE))  # fmt: skip
                if same or closed_label == label:
                    found.append(f"({arm}, {rate:g}, {seed}): its run {record['run_id']} is "
                                 f"closed by the closure record {kind}: never issued "
                                 "again")  # fmt: skip
    return found


# ------------------------------------------------------------------ the campaign


def cell_name(candidate: int, step: str, round_: int | None, arm: str, rate: float,
              seed: int) -> str:  # fmt: skip
    """A launcher cell's name, fixed by its one run (module docstring, "Names")."""
    tag = "" if round_ is None else f"-r{round_}"
    return f"c{candidate}-{step}{tag}-{OPTIMIZERS[arm]}-{rate:g}-s{seed}"


def launcher_cells(candidate: int, step: str, round_: int | None, blocks: list[Block],
                   checkpoints: Path | None) -> list[dict[str, Any]]:  # fmt: skip
    """One launcher cell per run, block by block in seed order (module docstring, "Names",
    "Configs")."""
    role = STEP_ROLES[step]
    config = K_ONLY if role in K_ONLY_ROLES else REGULAR
    out = []
    for seed, cells in blocks:
        for arm, rate in cells:
            overrides = [f"optimizer={OPTIMIZERS[arm]}", f"train.batch_size={BATCH}",
                         f"optimizer.lr={float(rate)!r}", f"study3.role={role}"]  # fmt: skip
            if checkpoints is not None:
                overrides.append(f"tracking.endpoint_checkpoint={checkpoints}")
            out.append({"name": cell_name(candidate, step, round_, arm, rate, seed),
                        "config": config, "overrides": overrides, "seeds": [seed]})  # fmt: skip
    return out


def sha256(kind: str, path: Path, base: Path) -> str:
    """The sha256 of a record file the header names, below ``base`` (the runs directory or the
    checkout); refused, unopened, when it resolves under a sealed folder
    (``study3_records.SEALED``) below ``base`` (or anywhere, when it leads outside it), as
    ``study3_decisions._under_sealed`` decides: a folder of that name above ``base`` does not
    count."""
    if _under_sealed(path, base):
        why = sealed_why(path, base, f"resolves {SEALED_WHY}")
        refuse(kind, f"{path} {why}: no record a campaign rests on is one, and it is not opened")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def standing_versions(repo: Path) -> Path:
    """The versions record that stands: the last of docs/study3-versions.json,
    docs/study3-versions-2.json, ... in the checkout (the first's path when there is none)."""
    found = {}
    for path in (repo / "docs").glob("study3-versions*.json"):
        number = re.fullmatch(r"study3-versions(?:-(\d+))?\.json", path.name)
        if number is None:
            continue
        k = int(number.group(1) or 1)
        if k >= 1 and versions_name(k) == f"docs/{path.name}":
            found[k] = path
    return found[max(found)] if found else repo / versions_name(1)


def decision(kind: str, root: Path, record_kind: str, key: str) -> tuple[dict, dict[str, str]]:
    """A decision record the campaign rests on, valid and passed, and its header entries: its
    path under RUNS_DIR and the sha256 of its file now."""
    record, problems = read_decision(root, record_kind, repo=REPO)
    if record is None or record["state"] != "passed":
        why = "; ".join(problems) or ("not written" if record is None else "it did not pass")
        refuse(kind, f"the {record_kind} record the campaign rests on is not valid and passed: "
               f"{why}")  # fmt: skip
    path = decision_path(root, record_kind)
    named = {
        key: decision_path(".", record_kind).as_posix(),
        f"{key}_sha256": sha256(kind, path, root),
    }
    return record, named


def header_of(kind: str, candidate: int, root: Path) -> dict[str, Any]:
    """The campaign's header (module docstring, "The header")."""
    step, round_ = KINDS[kind]
    header: dict[str, Any] = {"study": 3, "candidate": candidate, "step": step, "round": round_}
    if step in VERSIONED:
        path = standing_versions(REPO)
        problems = versions_problems(path, runs_root=root, repo=REPO, step=step,
                                     candidate=candidate)  # fmt: skip
        if problems:
            refuse(kind, "the versions record does not hold: " + "; ".join(problems))
        header.update(versions_record=path.relative_to(REPO).as_posix(),
                      versions_record_sha256=sha256(kind, path, REPO))  # fmt: skip
    for key, record_kind in RECORDS.get(step, {}).items():
        header.update(decision(kind, root, record_kind, key)[1])
    if step == "confirmation":
        simulation, _ = read_decision(root, "simulation", repo=REPO)
        n, named = decision(kind, root, f"n-c{candidate}", "n_record")
        header.update(named)
        budget = f"budget-c{candidate}"
        if (budget_reason(n["N"], simulation) is not None
                or decision_path(root, budget).exists()):  # fmt: skip
            header.update(decision(kind, root, budget, "budget_record")[1])
    return header


def campaign_text(name: str, header: dict[str, Any], cells: list[dict[str, Any]],
                  notes: list[str]) -> str:  # fmt: skip
    lines = [line for note in notes for line in comment(note)]
    data = {"name": name, "header": header, "cells": cells}
    return "\n".join(lines) + "\n" + yaml.safe_dump(data, sort_keys=False)


def run_command(kind: str, candidate: int, out: Path) -> str:
    """The command that runs the campaign (module docstring, "Running")."""
    if kind not in QUALIFICATION:
        return f"python scripts/launch.py {shlex.quote(str(out))} --max-attempts 1"
    state = git_state(REPO)
    commit = state.get("commit") if state.get("dirty") is False else None
    engineering = " --engineering" if kind == "engineering" else ""
    driver = out.with_name(f"{out.stem}-driver.sh")
    return (f"python scripts/qualify.py driver --campaign {shlex.quote(str(out))} --candidate "
            f"{candidate}{engineering} --commit {commit or 'SHA'} --out "
            f"{shlex.quote(str(driver))} --kept {KEPT}")  # fmt: skip


def check_command(root: Path, out: Path) -> str:
    """The check of the versions record a validation campaign names, run before its launch, every
    resumption and every recovery round of one of its runs (``study3_summary.py check``)."""
    return (f"python scripts/study3_summary.py check --runs-root {shlex.quote(str(root))} "
            f"--campaign {shlex.quote(str(out))}")  # fmt: skip


def driver_text(candidate: int, root: Path, files: list[tuple[int, Path, str]],
                written_at: str, theta0: Path | None = None) -> str:  # fmt: skip
    """The confirmation's driver (module docstring, "The confirmation"); ``theta0``: the
    writer's --theta0, which every guard call passes on."""
    read = "" if theta0 is None else f" --theta0 {shlex.quote(str(theta0.absolute()))}"
    seeds = [seed for seed, _, _ in files]
    listed = f"{seeds[0]}-{seeds[-1]}" if len(seeds) > 1 else str(seeds[0])
    lines = [
        "#!/usr/bin/env bash",
        *comment(f"Written by {WRITER} confirmation at {written_at}; do not edit."),
        *comment(f"Study 3's confirmation of candidate {candidate}: blocks {listed} ({len(seeds)} "
                 "in all), one campaign file each, one block at a time in seed order. Before "
                 "each block the between-block guard (scripts/study3_summary.py guard) reads the "
                 "records again and refuses a block that may not start (every active counter at "
                 "N, the cap used up, a technical mark, another step or seed, a problem in the "
                 "versions record the block's header names): the driver stops at its first "
                 "non-zero exit, and when the launcher fails. Each block's file is checked "
                 "against its sha256 before its guard, and runs with --max-attempts 1. The "
                 "driver pins RUNS_DIR to the runs directory the guard reads."),
        *comment("It sources the Pod's environment where there is one, and runs python from "
                 'the environment it is started in: on a Pod, project-run "$SRC" bash run.sh. '
                 f"After a stop, the blocks left are written again ({WRITER} confirmation), "
                 "under new names."),
        "set -euo pipefail",
        "# The Pod's environment, where there is one (an SSH session does not see it)",
        "if [ -f /etc/rp_environment ]; then . /etc/rp_environment; fi",
        "if [ -f /root/zfr-env.sh ]; then . /root/zfr-env.sh; fi",
        f"cd {shlex.quote(str(REPO.absolute()))}",
        f"RUNS_ROOT={shlex.quote(str(root.absolute()))}",
        "# The launcher writes where the guard reads",
        'export RUNS_DIR="$RUNS_ROOT"',
        "block() {",
        '  local seed="$1" campaign="$2" sha="$3" rc=0',
        '  if [ "$(sha256sum "$campaign" | cut -d " " -f 1)" != "$sha" ]; then',
        f'    echo "== {STAMP} $campaign is not the file written for block $seed: the driver '
        'stops" >&2',
        "    exit 1",
        "  fi",
        '  python scripts/study3_summary.py guard --runs-root "$RUNS_ROOT" '
        f'--candidate {candidate} --seed "$seed" --campaign "$campaign"{read} || rc=$?',
        '  if [ "$rc" -ne 0 ]; then',
        f'    echo "== {STAMP} the guard refused block $seed (exit $rc): the driver stops" >&2',
        '    exit "$rc"',
        "  fi",
        f'  echo "== {STAMP} block $seed: $campaign"',
        '  python scripts/launch.py "$campaign" --max-attempts 1',
        "}",
        *(f"block {seed} {shlex.quote(str(path))} {digest}" for seed, path, digest in files),
        f'echo "== {STAMP} blocks {listed} ran"',
    ]  # fmt: skip
    return "\n".join(lines) + "\n"


def write_new(path: Path, text: str) -> str:
    """Write ``path`` whole or not at all, in UTF-8; the sha256 of the bytes written."""
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_bytes(data)
    partial.replace(path)
    return hashlib.sha256(data).hexdigest()


def campaign(args: argparse.Namespace) -> None:
    """Write the campaign ``args.kind`` names, from the state; print where and how it runs."""
    kind, candidate, root = args.kind, args.candidate, args.runs_root
    step, round_ = KINDS[kind]
    state = progress(root, repo=REPO, expected_code=code_identity.gpu_code_identity()["sha256"],
                     theta0=args.theta0)  # fmt: skip
    own = the_step(kind, candidate, state)
    allowed, completing = authorized(kind, candidate, state, own)
    blocks = select(kind, allowed, len(own), args.seeds, args.blocks, completing)
    found = forbidden(root, candidate, step, round_, blocks)
    if found:
        refuse(kind, "; ".join(found))
    header = header_of(kind, candidate, root)
    checkpoints = (args.checkpoints or CHECKPOINTS) if kind in QUALIFICATION else None
    when, written_at = stamp(), revision()
    if kind == "confirmation":
        files = [(seed, args.out / f"study3-c{candidate}-confirmation-s{seed}-{when}.yaml",
                  [(seed, cells)]) for seed, cells in blocks]  # fmt: skip
    else:
        files = [(None, args.out, blocks)]
    names = [path.stem if kind == "confirmation" else f"study3-c{candidate}-{kind}-{when}"
             for _, path, _ in files]  # fmt: skip
    for name in names:
        if (root / "launch" / f"{name}.jsonl").exists():
            refuse(kind, f"the campaign name {name} is already used (its launch log exists): a "
                   "campaign's name is new")  # fmt: skip
    written = []
    for name, (seed, path, chosen) in zip(names, files, strict=True):
        runs = sum(len(cells) for _, cells in chosen)
        what = (f"{kind} of candidate {candidate}" if seed is None else
                f"block {seed} of the confirmation of candidate {candidate}")  # fmt: skip
        notes = [f"Written by {WRITER} {kind} at {written_at}; do not edit.",
                 f"Study 3: the {what}, {runs} runs on seeds "
                 f"{', '.join(str(s) for s, _ in chosen)}, from the state of the records under "
                 f"{root}."]  # fmt: skip
        if seed is None:
            if kind == "validation":
                notes.append(f"Check: {check_command(root.absolute(), path.absolute())}")
            notes.append(f"Run: {run_command(kind, candidate, path.absolute())}")
        else:
            notes.append(f"Run by {args.out.absolute() / 'run.sh'}, after the between-block "
                         "guard.")  # fmt: skip
        text = campaign_text(name, header, launcher_cells(candidate, step, round_, chosen,
                                                          checkpoints), notes)  # fmt: skip
        written.append((seed, path.absolute(), write_new(path, text)))
        print(path)
    if kind == "confirmation":
        driver = args.out / "run.sh"
        write_new(driver, driver_text(candidate, root, written, written_at, args.theta0))
        driver.chmod(0o755)
        print(driver)
        print(f"Run: bash {shlex.quote(str(driver.absolute()))}")
    else:
        if kind == "validation":
            print(f"Check: {check_command(root.absolute(), args.out.absolute())}")
        print(f"Run: {run_command(kind, candidate, args.out.absolute())}")


# ------------------------------------------------------------------ prep


def role_of(seed: int) -> str | None:
    """The role whose registered seeds hold ``seed`` (any candidate's)."""
    return next((role for role, ranges in SEEDS.items()
                 if any(seed in seeds for seeds in ranges.values())), None)  # fmt: skip


def prep(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    """The streams' driver of fresh study-3 seeds (module docstring, "prep")."""
    if not args.seeds or not args.commit:
        parser.error("prep needs --seeds and --commit")
    if args.candidate is not None or args.blocks or args.checkpoints is not None:
        parser.error("prep takes --seeds, --commit and --out")
    if len(args.commit) != 40 or set(args.commit) - set("0123456789abcdef"):
        parser.error("--commit takes the full 40-character commit hash")
    unknown = [seed for seed in args.seeds if role_of(seed) is None]
    if unknown:
        parser.error(f"--seeds {unknown}: not registered study-3 seeds (docs/study3-"
                     "preregistration.md, \"Seeds and order\")")  # fmt: skip
    configs = {}
    for seed in args.seeds:
        role = role_of(seed)
        configs[seed] = f"{K_ONLY if role in K_ONLY_ROLES else REGULAR} study3.role={role}"
    return streams_driver(args.seeds, args.commit, configs=configs, batch=BATCH, steps=UPDATES,
                          writer=f"{WRITER} prep", title="Study 3",
                          written_at=revision())  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=(*KINDS, "prep"))
    parser.add_argument("--runs-root", type=Path,
                        help="RUNS_DIR: the run records, the launch journal and the decision "
                        "records")  # fmt: skip
    parser.add_argument("--candidate", type=int, choices=(1, 2),
                        help="the candidate whose step the campaign is")  # fmt: skip
    parser.add_argument("--theta0", type=Path,
                        help="θ₀'s evaluation (study 2's under RUNS_DIR by default)")  # fmt: skip
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="the step's first blocks (seeds), in order; prep: its "
                        "seeds")  # fmt: skip
    parser.add_argument("--blocks", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="confirmation: the blocks (seeds) FIRST to LAST")  # fmt: skip
    parser.add_argument("--checkpoints", type=Path,
                        help="engineering, numerical, repair: the launcher's step-25 checkpoint "
                        f"directory (default {CHECKPOINTS})")  # fmt: skip
    parser.add_argument("--commit", help="prep: the commit whose source checkout the Pod runs")
    parser.add_argument("--out", type=Path, required=True,
                        help="the campaign file (confirmation: its folder; prep: the "
                        "driver)")  # fmt: skip
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error(f"{args.out} exists")
    seeds = args.seeds or []
    if len(set(seeds)) != len(seeds):
        parser.error(f"--seeds names a seed twice: {seeds}")
    if args.kind == "prep":
        text = prep(parser, args)
        write_new(args.out, text)
        print(args.out)
        return
    if args.runs_root is None or args.candidate is None:
        parser.error(f"{args.kind} needs --runs-root and --candidate")
    if args.commit is not None:
        parser.error("--commit goes with prep")
    if args.blocks is not None and args.kind != "confirmation":
        parser.error("--blocks goes with confirmation")
    if args.blocks is not None and args.seeds:
        parser.error("--seeds or --blocks, not both")
    if args.seeds and args.kind in QUALIFICATION:
        parser.error(f"--seeds goes with search, validation and confirmation: {args.kind} is a "
                     "qualification campaign, written whole, one file per step, as "
                     "scripts/qualify.py reads it (each FA rate's same-path control lies on the "
                     "first numerical seed)")  # fmt: skip
    if args.checkpoints is not None and args.kind not in QUALIFICATION:
        parser.error("--checkpoints goes with engineering, numerical and repair")
    if args.checkpoints is not None and not args.checkpoints.is_absolute():
        parser.error(f"--checkpoints {args.checkpoints}: an absolute directory "
                     "(scripts/qualify.py reads every checkpoint of a campaign from one absolute "
                     "directory)")  # fmt: skip
    campaign(args)


if __name__ == "__main__":
    main()
