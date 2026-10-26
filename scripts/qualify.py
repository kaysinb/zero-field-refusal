"""Study 3's numerical qualification on a Pod: the plan, a case's parts, the driver, the verdict.

    python scripts/qualify.py plan --campaign FILE --candidate N [--engineering]
        [--diagnostic-steps T1,T2,...]
    python scripts/qualify.py driver --campaign FILE --candidate N [--engineering]
        [--diagnostic-steps T1,T2,...] --commit SHA --out FILE [--dry-run]
    python scripts/qualify.py part --campaign FILE --candidate N [--engineering] --case NAME
        --part baseline|rest|control
        (--resume | --attempt N [--infrastructure REASON | --engineering-cycle])
        --run-dir DIR --checkpoints DIR [--diagnostic-steps T1,T2,...] [--device D]
    python scripts/qualify.py complete|decided|clear-stale|locate|finish [--release]
        --campaign FILE --candidate N [--engineering] --case NAME
    python scripts/qualify.py interrupted --campaign FILE --candidate N [--engineering]
        --case NAME --part baseline|rest|control --attempt N --technical REASON
    python scripts/qualify.py restart-case --campaign FILE --candidate N [--engineering]
        --case NAME --infrastructure REASON
    python scripts/qualify.py supersede --campaign FILE --candidate N [--engineering]
        --case NAME --record PATH --diagnosis TEXT --fix TEXT
    python scripts/qualify.py verdict --campaign FILE --candidate N [--engineering] [--e-eng X]

Every command also takes ``--runs-root DIR`` (RUNS_DIR) and ``--out DIR`` (the part records,
default RUNS_DIR/qualification/). The rules are docs/study3-preregistration.md, "Numerical
qualification", "Failure routes", "Voiding and supersession" and "Amendment 2"; the harness, the
records and the verdicts are ``zero_field_refusal.qualification``'s, and the settings the tracked
configs/experiments/qualification.yaml (never another file).

**The cases** (``campaign_cases``) are the runs of a launcher campaign (``launch.load_campaign``),
one case per run: its seed, arm and rate as its config has them, for ``--candidate`` (numerical,
or with ``--engineering`` the engineering stage). Each run must be a K-only training run
(``eval.k_only``) of the role (``study3.role``: numerical, or engineering), at B 20 and 25 steps,
on the candidate's numerical seeds (or the engineering seeds), with its endpoint checkpoint
(``tracking.endpoint_checkpoint``) in one absolute directory for all; an FA rate (the control
arm) of a numerical campaign needs its case on the candidate's first numerical seed, which holds
the rate's same-path control (amendment 2). ``plan`` lists the cases, the expected GPU time
(``MINUTES``: a launcher run ≈ 7 min, a baseline part ≈ 7 min, a rest part ≈ 25 min, a
same-path control ≈ 11 min), the disk need and the host memory, and exits 1 on a campaign
outside this.

**A case, in order** (one at a time, on one Pod; the driver):

1. ``complete`` exits 0 when the case is done (below): then nothing of it runs again; 3 when it
   is not. ``complete`` and ``decided`` say no with exit 3 alone: any other non-zero exit is a
   failure, and stops the driver. First, both check the case's recorded parts (the latest
   attempt of each) against this checkout's GPU code (``code_identity.gpu_code_identity``) and
   the tracked settings: a record of other code or settings makes the case ``stale``, exit 10
   (``STALE``), and the driver stops there: a stale case is never rerun by it, and never skipped
   (its re-qualification is a decision of its own, "Voiding and supersession").
2. ``clear-stale`` removes ``<dir>/<run_id>.pt`` (and its partial file) of every attempt of the
   case's launcher run that the launch journal shows superseded (``superseded_run_ids``: ended
   technical, or not ended, without a settled directory); the checkpoint of a settled run (done,
   diverged, or completed but in conflict, kept for the diagnosis) stays.
3. The free disk of the checkpoint directory against the need (``disk_need_gb``: two endpoints,
   the launcher's and the harness baseline's, ``checkpoint_gb`` each, one state per diagnostic
   step, one checkpoint of headroom for an atomic save, and ``disk_margin_gb``); below it the
   driver exits 1.
4. ``python scripts/launch.py CAMPAIGN --only LABEL --max-attempts 1``: the launcher run alone,
   one attempt; a failed one stops the driver, and is attempted again only with
   ``--recovery REASON`` on that command (a recorded reason).
5. ``locate`` prints the launcher run's directory (its settled run: done, or diverged).
6. ``part --part baseline --resume``; then, unless ``decided`` (exit 0: the baseline record is
   already a scientific failure, a declared instability rule fired on the unshifted baseline or
   the launcher run was one; ``qualification.case_verdict`` of the baseline alone), ``part
   --part rest --resume``: a decided case is not measured further, its rate fails whatever the
   rest shows. For an FA rate on the first numerical seed, ``part --part control --resume``
   either way (the control reads no checkpoint).
7. ``finish`` deletes the case's checkpoints (every ``<dir>/<run_id>.pt`` of its launcher run's
   attempts and the harness's directory ``<dir>/harness/<case>/``), only once its records are
   published and valid; it exits 1 and keeps them for a case that is not complete. A case whose
   verdict is technical (its own, or its same-path control's: ``technical_reasons``) keeps them
   for the diagnosis, "Failure routes": ``finish`` exits 9, and the driver stops there, before
   the next case. ``finish --release`` deletes them once the diagnosis is done.

A case is **complete** when the latest attempt of each of its parts is recorded, is that part's
of that case, and the verdicts read it without error (``case_verdict``, and ``control_verdict``
for its control): the baseline, the due control and, unless decided, the rest, which counts only
when it names the latest baseline record's sha256 (``baseline_record``; a rest that read no
baseline, having run nothing or failed before, excepted). Records are published atomically and
durably (``qualification.publish``).

**Attempts of a part** (``part``; "Failure routes": an infrastructure interruption is rerun under a
new attempt id, a technical or scientific record never). Before ``run_part`` the part publishes
``<record>.started``; an attempt that has it and no record was interrupted. ``--resume`` (the
driver's call) runs attempt 1 of a part without any attempt, does nothing for a part whose latest
attempt is recorded (whatever its kind), and exits 8 for one whose latest attempt was interrupted.
``--attempt N`` runs attempt 1 of a part without any attempt, or the attempt after an interrupted
one, which needs ``--infrastructure REASON``, the documented interruption, and its record carries it
from its one publish (``infrastructure``: the reason, the earlier attempts and the case's restart
records); a part whose latest attempt is recorded, whatever its kind, is refused. ``interrupted
--technical REASON`` records an interrupted attempt, unexplained, as a technical part record
carrying the reason ("an unexplained incomplete record is technical"); it refuses an attempt with a
record. A rest attempt publishes each trajectory's result as soon as it is done (``qualification``,
"Trajectory results"): when its attempt is interrupted, the verdict reads them as its record
(``rest_from_trajectories``), ``interrupted`` records them in the part's record, and ``part
--attempt N`` refuses to rerun an attempt whose results hold a failure, scientific or technical
(they keep counting). The part needs a clean checkout of a commit, sets the numerics up
(``runtime.configure_numerics``) before any CUDA work and builds the environment from the run's
campaign config; the survival retry and the float64 fallback stay inside the part.

**A case restart** (``restart-case --infrastructure REASON``; "Voiding and supersession": an
infrastructure-interrupted attempt is void and kept, its rerun a new attempt id). A Pod stop during
a case loses the container disk, with the launcher run's checkpoint and the harness's: the case
cannot go on from its recorded parts. ``restart-case`` publishes (atomically, from a clean checkout)
``restart-<n>.json`` in the case's folder, naming every attempt of the case's parts (the baseline,
the rest and, for the case that holds it, the same-path control): each recorded one with its
record's sha256, status and kind, each interrupted one by its number; the reason; and the case's
launcher run. Then, the record published, it logs that run superseded in the launch journal
(``launch.supersede``): the run is kept, no longer counts for its label (``assess``, ``located``),
and the launcher's recovery round with the same reason starts a new run; run again once every
attempt is named, the command only completes that journal line. Only an interrupted case is
restarted: it is refused without a reason, when every attempt is named already, for a complete case,
without evidence of an interruption (an interrupted attempt, or the launcher run's checkpoint gone
from disk), when the launcher run itself is a completed failure (its declared instability stop,
scientific, or another stop or a departing run, technical: ``launcher_problems`` without the
comparison with this process's environment), and when any attempt is a completed failure
(``attempt_failures``): a scientific one, which keeps counting ("no reruns to pass"), a technical
one (the engineering cycle lifts it: below), or an unreadable record; an interrupted rest attempt
counts by its trajectory results. A recorded attempt that passes so far (the baseline, a control)
depends on the lost run or checkpoint and is named with the others. The attempts it names are void:
a part whose recorded attempts are all named is ``voided``, so the case is not complete; ``part
--resume`` exits 8 for it, and ``part --attempt N`` takes its next attempt with ``--infrastructure
REASON`` (its record names the restart records); a rest waits for the baseline's new attempt. The
named records are kept, listed, and voided in the verdict (with the reason, under ``restart``).

**Supersession** (``supersede --record PATH --diagnosis TEXT --fix TEXT``; "Failure routes": one
engineering cycle per candidate; "Voiding and supersession"). A completed technical failure stops
counting only through the candidate's supersession record, ``c<N>/supersede-1.json``: it names the
technical record that started the cycle (its path and sha256), the diagnosis and the fix, the GPU
code of the fix (this clean checkout's) and the old one, every record of the candidate made under
the old GPU code in all its stage folders, passed ones included, with the cases re-qualified, and
every old-code launcher run of the candidate in the launch journal (the runs its records name and
those of the campaign's cases, a case with no part attempt yet included), which it then logs
superseded. A case that holds a scientific failure is kept whole: none of its records (its parts,
its control, its trajectory results) nor its launcher run is voided, it is listed as kept with why,
and the driver's stale check passes it; the failure keeps counting (no reruns to pass). The cycle is
refused a second time for the candidate (a technical failure left unresolved after it routes to the
technical feasibility report), for a record that is not a completed technical failure of the case,
and when the fix's GPU code is the old one (the same path is not simply rerun); run again with the
same record, it only writes the journal's lines that are missing. Each case it names re-qualifies
under the new code from a new launcher run (the recovery round) and new attempts of its parts: the
attempt after one the cycle named carries ``--engineering-cycle`` (its record:
``engineering_cycle``, the supersession's path and sha256) instead of ``--infrastructure``; a later
interruption of a new-code attempt is rerun with ``--infrastructure REASON``, as any other. The
verdict lists the records it voids under ``supersession``.

``verdict`` writes RUNS_DIR/qualification/c{N}/verdict.json (with ``--engineering``,
``c{N}/engineering/verdict.json`` with ``engineering_summary``): per rate of the campaign its
``rate_verdict`` under this checkout's GPU code and the tracked settings, with its cases' and its
same-path control's verdicts (states, kinds, reasons, the 13 comparisons, warnings with
``--e-eng``), every case record of the candidate on disk shown (one of no campaign rate under
``unassigned``), every attempt of every part (``attempts``: recorded or interrupted, with each
record's sha256, status, kind and the reason of the restart that names it, as ``restart``, or of the
supersession, as ``supersession``), and prints the states. A record a case restart or supersession
names is listed under ``superseded`` with that reason and passed to ``rate_verdict`` as voided (its
record sha256 → the reason): it neither counts nor blocks. A rest whose latest attempt was
interrupted is read from its trajectory results. A record under a later attempt that no restart
names (which these commands never write) is not hidden by it: it is listed under ``superseded`` and
counted against its rate as a case of its own (a control's as a superseded control of its rate). The
launcher runs located now (one settled directory per run) are the expected ones; a rate one of whose
launcher runs is not located is at most ``incomplete``, with the reason and a warning.

``driver`` writes the bash driver for a Pod from the source checkout of SHA: it checks the checkout
and the campaign file's sha256, that nothing else uses the GPU, and the free host memory
(``host_memory_gb`` of the settings, ``drift.host_memory``: θ₀, v₀, the second model and the p1
direction on the host and a checkpoint save's transient), then runs each case in order as above, and
writes the verdict on every exit. Exits: 0 when every case is complete; 1 a precondition (the host
memory included), the disk (``df``'s answer must be a number) or a ``finish`` that refuses a case
not complete; 8 an interrupted attempt (or a part a case restart or supersession names); 9 a
technical case; 10 a stale case; the launcher's own when it stops; any other, the failed command's.
The driver never reruns a part or the launcher run: a rerun is a command of its own, with its
reason."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import shutil
import socket
import sys
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf

from zero_field_refusal import code_identity, runtime
from zero_field_refusal import qualification as q
from zero_field_refusal.analysis.qualification_verdicts import REST_MISSING
from zero_field_refusal.config import load_config
from zero_field_refusal.launch import (
    Run,
    assess,
    launch_records,
    load_campaign,
    supersede,
    superseded_runs,
)
from zero_field_refusal.paths import CONFIG_ROOT, load_env, runs_root
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.tracking import CLEAN_CHECKOUT, git_state

SETTINGS = CONFIG_ROOT / "experiments" / "qualification.yaml"  # the tracked settings, by path
RUN = 'project-run "$SRC" python scripts/qualify.py'
# A case restart or an engineering cycle's supersession record, in the case's folder
VOIDING_FILE = re.compile(r"(restart|supersede)-(\d+)\.json")
LAUNCH = 'project-run "$SRC" python scripts/launch.py'
NO = 3  # ``complete`` and ``decided``: no (any other non-zero exit is a failure)
INTERRUPTED = 8  # ``part --resume``: the latest attempt was started and left no record
TECHNICAL_CASE = 9  # ``finish``: the case's verdict is technical; its checkpoints are kept
STALE = 10  # ``complete`` and ``decided``: records of other GPU code or settings
# The Pod's environment and the source checkouts (the driver's preamble)
POD_ENVIRONMENT = (". /etc/rp_environment", ". /root/zfr-env.sh")
SOURCES = "/workspace/code/zero-field-refusal"
GPU_USED_MIB = 1000  # the driver starts only below this much GPU memory in use
# Whether the free host memory (``drift.host_memory``: MemAvailable, or the cgroup's free memory
# when smaller) reaches the settings' ``host_memory_gb`` (exit 0); run by the driver through
# project-run. No $, backtick, backslash or double quote inside.
HOST_FREE = ("import sys; from zero_field_refusal.drift import host_memory; m = host_memory(); "
             "print('free host memory (GB):', m); "
             "sys.exit(0 if (m['available_gb'] or 0) >= float(sys.argv[1]) else 1)")  # fmt: skip
# The expected GPU minutes on one L40S at 1.5B: a launcher run (≈ 7.1 min, "Budget and timing"),
# the baseline part (one trajectory with its setup), the rest (six) and a same-path control
MINUTES = {"launcher": 7, "baseline": 7, "rest": 25, "control": 11}
ROLES = {False: "numerical", True: "engineering"}


def comment(text: str) -> list[str]:
    return textwrap.wrap(text, 100, initial_indent="# ", subsequent_indent="# ",
                         break_long_words=False, break_on_hyphens=False)  # fmt: skip


def load_settings():
    return load_config(SETTINGS)


@dataclass(frozen=True)
class Planned:
    """A case of the campaign: its launcher run, the directory of that run's endpoint
    checkpoint, and whether it holds its rate's same-path control."""

    run: Run
    case: q.QualificationCase
    endpoint: Path
    control: bool

    @property
    def harness(self) -> Path:
        """The harness's checkpoints of the case (``run_part``'s ``checkpoints``)."""
        return self.endpoint / "harness" / self.case.name.replace("/", "-")


def campaign_cases(
    path: Path, scope: q.QualificationScope, candidate: int, engineering: bool
) -> tuple[str, list[Planned], list[str]]:
    """The campaign's name, its cases in order and why it is not a qualification campaign of the
    candidate (module docstring, "The cases"; empty when it is)."""
    name, runs = load_campaign(path)
    problems: list[str] = []
    try:
        seeds = scope.engineering_seeds if engineering else scope.seeds(candidate)
        first = None if engineering else scope.seeds(candidate)[0]
    except q.QualificationError as exc:
        return name, [], [str(exc)]
    planned, directories = [], set()
    for run in runs:
        cfg = load_run_config(run.config, run.overrides)
        case = q.QualificationCase(int(cfg.seed), str(cfg.optimizer.name), float(cfg.optimizer.lr),
                                   candidate, engineering, batch=int(cfg.train.batch_size),
                                   steps=int(cfg.train.steps))  # fmt: skip
        where = OmegaConf.select(cfg, "tracking.endpoint_checkpoint", default=None)
        role = OmegaConf.select(cfg, "study3.role", default=None)
        mine = []
        if run.entry != "train" or OmegaConf.select(cfg, "eval.k_only", default=None) is not True:
            mine.append("not a K-only training run (eval.k_only)")
        if role != ROLES[engineering]:
            mine.append(f"study3.role is {role!r}, not {ROLES[engineering]!r}")
        if (case.batch, case.steps) != (q.BATCH, q.STEPS):
            mine.append(f"B {case.batch} and {case.steps} steps, not B {q.BATCH} and {q.STEPS}")
        if case.seed not in seeds:
            mine.append(f"seed {case.seed} is not among the {ROLES[engineering]} seeds "
                        f"{list(seeds)}")  # fmt: skip
        if not (isinstance(where, str) and Path(where).is_absolute()):
            mine.append(f"tracking.endpoint_checkpoint is {where!r}, not an absolute directory")
        else:
            directories.add(Path(where))
        problems += [f"{run.label}: {p}" for p in mine]
        control = case.optimizer == scope.control_arm and case.seed == first
        planned.append(Planned(run, case, Path(where or "."), control))
    if len(directories) > 1:
        problems.append(f"the endpoint checkpoints go to several directories: "
                        f"{sorted(map(str, directories))}")  # fmt: skip
    names = [p.case.name for p in planned]
    problems += [f"two runs are case {n}" for n in sorted({n for n in names if names.count(n) > 1})]
    if not engineering:
        held = {p.case.lr for p in planned if p.control}
        for lr in sorted({p.case.lr for p in planned if p.case.optimizer == scope.control_arm}):
            if lr not in held:
                problems.append(f"{scope.control_arm} at {lr:g} has no case on seed {first}, "
                                "which holds its same-path control")  # fmt: skip
    return name, planned, problems


def disk_need_gb(settings, diagnostic_steps) -> float:
    """The free space a case needs (module docstring, step 3), in GB of 1e9 bytes."""
    checkpoints = 2 + len(list(diagnostic_steps)) + 1
    return checkpoints * float(settings.checkpoint_gb) + float(settings.disk_margin_gb)


def need_kb(gb: float) -> int:
    """``gb`` (1e9 bytes) in the 1024-byte blocks of ``df -Pk``, rounded up."""
    return math.ceil(gb * 1e9 / 1024)


def minutes(planned: list[Planned]) -> int:
    per_case = MINUTES["launcher"] + MINUTES["baseline"] + MINUTES["rest"]
    return per_case * len(planned) + MINUTES["control"] * sum(p.control for p in planned)


def parse_steps(text: str | None) -> list[int]:
    return sorted({int(t) for t in text.split(",")}) if text else []


# ------------------------------------------------------------------ the records of a case


@dataclass(frozen=True)
class PartState:
    """A part's latest attempt on disk: ``due`` (none), ``recorded`` (with its record),
    ``interrupted`` (started, no record), ``voided`` (every recorded attempt named by the case's
    restart or supersession records) or ``invalid`` (a record that is unreadable, or not that
    part's of that case). ``attempts`` maps every attempt to recorded or interrupted."""

    state: str
    attempt: int
    record: dict | None
    attempts: dict[int, str]


def attempt_files(folder: Path, part: str) -> dict[int, str]:
    """Every attempt of a part in a case's folder: recorded (its record exists) or interrupted
    (only its ``.started`` file does), by attempt number."""
    pattern = re.compile(rf"{re.escape(part)}-attempt-(\d+)\.json(\.started)?")
    seen: dict[int, set[str]] = {}
    for file in folder.glob(f"{part}-attempt-*") if folder.is_dir() else []:
        match = pattern.fullmatch(file.name)
        if match:
            seen.setdefault(int(match[1]), set()).add("started" if match[2] else "record")
    return {n: "recorded" if "record" in kinds else "interrupted"
            for n, kinds in sorted(seen.items())}  # fmt: skip


def case_folder(out: Path, case: q.QualificationCase) -> Path:
    """The folder of a case's attempt records and restart records."""
    return (out / case.attempt_path("baseline", 1)).parent


def file_sha256(path: Path) -> str | None:
    """The sha256 of a record file's bytes (``record_sha256`` of a published record), or None."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def case_name_of(folder: Path) -> str:
    """The case a folder holds (``s<seed>/<arm>-<rate>``)."""
    return f"{folder.parent.name}/{folder.name}"


def stage_folder(folder: Path) -> Path:
    """The candidate's stage folder (``c<N>/<stage>/``) above a case's folder."""
    return folder.parent.parent


def candidate_folder(folder: Path) -> Path:
    """The candidate's folder (``c<N>/``) above a case's folder: it holds the supersession."""
    return folder.parent.parent.parent


def voiding_records(folder: Path, kind: str | None = None) -> list[tuple[Path, dict | None]]:
    """The records that void attempts of the case in ``folder``: its restart records
    (``restart-<n>.json``, in the folder) and the candidate's supersession record
    (``supersede-<n>.json``, in the candidate's folder, ``c<N>/``); ``kind``: one of them alone.
    In order, each with its record."""
    found = []
    for where, wanted in ((folder, "restart"), (candidate_folder(folder), "supersede")):
        if kind not in (None, wanted) or not where.is_dir():
            continue
        found += [(wanted, int(m[2]), f) for f in where.glob(f"{wanted}-*.json")
                  if (m := VOIDING_FILE.fullmatch(f.name))]  # fmt: skip
    return [(f, _read(f)) for _, _, f in sorted(found)]


def restart_records(folder: Path) -> list[tuple[Path, dict | None]]:
    return voiding_records(folder, "restart")


def void_reason(path: Path, record: dict | None) -> str:
    """Why the attempts a voiding record names are void: a restart's reason, or the
    supersession's diagnosis and fix."""
    record = record or {}
    if path.name.startswith("supersede-"):
        return (f"superseded by an engineering cycle ({path.name}): {record.get('diagnosis')}; "
                f"fix: {record.get('fix')}")  # fmt: skip
    return str(record.get("reason"))


def _entries(folder: Path, kind: str | None = None):
    """(voiding record path, its record, an attempt entry of this case) for every attempt the
    voiding records name (a supersession's entries name their case)."""
    case, stage = case_name_of(folder), stage_folder(folder).name
    for path, record in voiding_records(folder, kind):
        for entry in (record or {}).get("attempts") or []:
            if (isinstance(entry, dict) and entry.get("case", case) == case
                    and entry.get("stage", stage) == stage):  # fmt: skip
                yield path, record, entry


def named_records(folder: Path, kind: str | None = None) -> dict[str, str]:
    """The attempt records the case's restart and supersession records name: each record
    file's sha256 → why it is void (module docstring, "A case restart", "Supersession")."""
    named: dict[str, str] = {}
    for path, record, entry in _entries(folder, kind):
        if isinstance(entry.get("sha256"), str):
            named.setdefault(entry["sha256"], void_reason(path, record))
    return named


def named_attempts(folder: Path, kind: str | None = None) -> set[tuple[str, int]]:
    """Every (part, attempt) a restart or supersession record names, recorded or interrupted."""
    return {(e.get("part"), e.get("attempt")) for _, _, e in _entries(folder, kind)}


def cycle_of(folder: Path, part: str, attempt: int) -> tuple[Path, dict | None] | None:
    """The supersession record that names this attempt of this part of the case (its latest),
    or None: the rerun after it carries ``--engineering-cycle``; a later interruption of a
    new-code attempt is rerun with ``--infrastructure`` as any other."""
    for path, record, entry in _entries(folder, "supersede"):
        if entry.get("part") == part and entry.get("attempt") == attempt:
            return path, record
    return None


def trajectory_results(folder: Path, attempt: int) -> list[dict]:
    """The trajectory results a rest attempt published (``qualification``, "Trajectory
    results")."""
    found = [_read(f) for f in sorted(folder.glob(f"rest-attempt-{attempt}.trajectory-*.json"))]
    return [r for r in found if r is not None]


def interrupted_rest(folder: Path, attempt: int) -> dict | None:
    """An interrupted rest attempt as a record, from its trajectory results
    (``rest_from_trajectories``), or None without any."""
    return q.rest_from_trajectories(trajectory_results(folder, attempt))


def part_state(out: Path, case: q.QualificationCase, part: str) -> PartState:
    folder = case_folder(out, case)
    attempts = attempt_files(folder, part)
    if not attempts:
        return PartState("due", 1, None, attempts)
    latest = max(attempts)
    if attempts[latest] == "interrupted":
        return PartState("interrupted", latest, None, attempts)
    named = named_records(folder)
    if all(file_sha256(folder / f"{part}-attempt-{n}.json") in named
           for n, s in attempts.items() if s == "recorded"):  # fmt: skip
        return PartState("voided", latest, None, attempts)
    record = _read(out / case.attempt_path(part, latest))
    if record is None or record.get("part") != part or record.get("case") != case.record():
        return PartState("invalid", latest, None, attempts)
    return PartState("recorded", latest, record, attempts)


def _read(path: Path) -> dict | None:
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def case_state(out: Path, p: Planned, scope: q.QualificationScope) -> tuple[bool, bool, str]:
    """Whether the case is complete, whether its baseline decided it (scientific), and why
    (module docstring, "complete")."""
    baseline = part_state(out, p.case, "baseline")
    if baseline.state != "recorded":
        return False, False, f"baseline: {baseline.state} (attempt {baseline.attempt})"
    try:
        decided = q.case_verdict(baseline.record, None, scope)["kind"] == q.SCIENTIFIC
    except Exception as exc:  # a record the verdict cannot read is not a valid one
        return False, False, f"the baseline record cannot be read: {type(exc).__name__}: {exc}"
    needed = ([] if decided else ["rest"]) + (["control"] if p.control else [])
    states = {part: part_state(out, p.case, part) for part in needed}
    for part, state in states.items():
        if state.state != "recorded":
            return False, decided, f"{part}: {state.state} (attempt {state.attempt})"
    rest = states["rest"].record if "rest" in states else None
    # A rest that read no baseline (it ran nothing, or failed before reading one) binds none:
    # the verdict holds it technical
    read_none = rest is not None and (rest.get("status") == "not_run" or (
        rest.get("status") == "failed" and rest.get("baseline_record") is None))  # fmt: skip
    if (rest is not None and not read_none
            and (rest.get("baseline_record") or {}).get("sha256")
            != q.record_sha256(baseline.record)):  # fmt: skip
        return False, decided, ("the rest record is not bound to the latest baseline record: "
                                "the case is not complete")  # fmt: skip
    try:
        q.case_verdict(baseline.record, rest, scope)
        if p.control:
            q.control_verdict(states["control"].record, scope, p.case.candidate)
    except Exception as exc:
        return False, decided, f"the records cannot be read: {type(exc).__name__}: {exc}"
    return True, decided, "decided by the baseline: a scientific failure" if decided else "complete"


def stale_records(
    out: Path, p: Planned, code: dict[str, Any] | None, policy: dict[str, Any] | None
) -> list[str]:
    """The case's recorded parts (the latest attempt of each) written under other GPU code or
    settings than this checkout's (``code``: ``code_identity.gpu_code_identity``; ``policy``:
    the tracked settings' record): such a case is ``stale`` (module docstring, step 1). A
    record the candidate's engineering cycle kept (a scientific failure) is not."""
    stale = []
    kept = kept_records(case_folder(out, p.case))  # the engineering cycle's: they keep counting
    for part in ["baseline", "rest", *(["control"] if p.control else [])]:
        state = part_state(out, p.case, part)
        if state.state != "recorded" or file_sha256(
                out / p.case.attempt_path(part, state.attempt)) in kept:  # fmt: skip
            continue
        provenance = state.record.get("provenance") or {}
        if code is None or (provenance.get("gpu_code") or {}).get("sha256") != code.get("sha256"):
            stale.append(f"{part} attempt {state.attempt}: other GPU code")
        elif provenance.get("settings") != policy:
            stale.append(f"{part} attempt {state.attempt}: other settings")
    return stale


def technical_reasons(out: Path, p: Planned, scope: q.QualificationScope) -> list[str]:
    """Why a complete case's verdict is technical (its own, or its same-path control's), or
    nothing: such a case keeps its checkpoints for the diagnosis ("Failure routes")."""
    baseline = part_state(out, p.case, "baseline").record
    rest = part_state(out, p.case, "rest").record
    verdict = q.case_verdict(baseline, rest, scope)
    reasons = list(verdict["reasons"]) if verdict["kind"] == q.TECHNICAL else []
    if p.control:
        control = q.control_verdict(part_state(out, p.case, "control").record, scope,
                                    p.case.candidate)  # fmt: skip
        if control["kind"] == q.TECHNICAL:
            reasons += [f"control: {r}" for r in control["reasons"]]
    return reasons


# --------------------------------------------------------------- the launcher run's files


def journal_run_ids(run: Run, root: Path) -> list[str]:
    """The run ids of every logged attempt of the run (launch journal, start records)."""
    return list(dict.fromkeys(
        r["run_id"] for r in launch_records(root)
        if r.get("event") == "start" and r.get("run") == run.label
        and r.get("experiment") == run.experiment and r.get("config_hash") == run.config_hash
        and isinstance(r.get("run_id"), str)))  # fmt: skip


def settled(run: Run, root: Path) -> Path | None:
    """The run's settled launcher directory (``launch.assess``: done, or diverged), or None."""
    state = assess(run, root, launch_records(root))
    return state.done or state.diverged


def remove_checkpoints(p: Planned, run_ids: list[str]) -> list[str]:
    removed = []
    for run_id in run_ids:
        for path in (p.endpoint / f"{run_id}.pt", p.endpoint / f".{run_id}.pt.partial"):
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed


def located(run: Run, root: Path) -> Path | None:
    """For the verdict, after the checkpoints are gone (``assess`` would call such a run a
    conflict): the one directory of the run's config whose outcome is settled, or None. A run a
    case restart or supersession named superseded in the journal is not one."""
    found = []
    superseded = superseded_runs(run, launch_records(root))
    for run_dir in sorted((root / run.experiment).glob("*")):
        meta = _read(run_dir / "meta.json")
        if meta is None or meta.get("config_hash") != run.config_hash:
            continue
        if run_dir.name in superseded:
            continue
        if meta.get("status") in ("completed", "diverged") or (run_dir / "terminal.json").is_file():
            found.append(run_dir)
    return found[0] if len(found) == 1 else None


# ------------------------------------------------------------------------------- commands


def require_clean() -> None:
    """A part or a verdict writes records only from a clean checkout of a commit."""
    state = git_state()
    if state.get("commit") is None or state.get("dirty") is not False:
        sys.exit("The checkout is not clean (tracking.git_state): records made from it never "
                 "count, and none is written from it; commit first")  # fmt: skip


def plan_text(name: str, planned: list[Planned], problems: list[str], settings, steps) -> str:
    need = disk_need_gb(settings, steps)
    lines = [f"campaign {name}: {len(planned)} cases, in this order"]
    for p in planned:
        lines.append(f"  {p.case.name}: launcher run {p.run.label}"
                     + ("; with its rate's same-path control" if p.control else ""))  # fmt: skip
    if planned:
        lines.append(f"checkpoints: {planned[0].endpoint} (the harness's under harness/)")
    lines.append(f"≈ {minutes(planned)} GPU min (a launcher run ≈ {MINUTES['launcher']}, a "
                 f"baseline ≈ {MINUTES['baseline']}, a rest ≈ {MINUTES['rest']}, a same-path "
                 f"control ≈ {MINUTES['control']} min)")  # fmt: skip
    lines.append(f"free disk per case: {need:g} GB ({2 + len(steps) + 1} checkpoints of "
                 f"{float(settings.checkpoint_gb):g} GB and {float(settings.disk_margin_gb):g} "
                 "GB of margin)")  # fmt: skip
    lines.append(f"free host memory: {float(settings.host_memory_gb):g} GB")
    lines += [f"PROBLEM: {p}" for p in problems]
    return "\n".join(lines)


def part(args, planned: Planned, out: Path) -> None:
    case, name = planned.case, args.part
    state = part_state(out, case, name)
    reason = args.infrastructure
    # The engineering cycle named its latest attempt: the rerun after it carries the mark
    cycle = None if state.state == "due" else cycle_of(case_folder(out, case), name, state.attempt)
    if args.resume:
        if state.state == "recorded":
            print(f"{case.name} {name}: attempt {state.attempt} is recorded "
                  f"({state.record.get('status')}); nothing runs")  # fmt: skip
            return
        if state.state in ("interrupted", "voided"):
            mark = "--engineering-cycle" if cycle is not None else "--infrastructure REASON"
            why = ("was started and left no record" if state.state == "interrupted"
                   else "is named by the case's restart or supersession record")  # fmt: skip
            print(f"{case.name} {name}: attempt {state.attempt} {why}: its next attempt is run "
                  f"by hand, --attempt {state.attempt + 1} {mark}", file=sys.stderr)  # fmt: skip
            sys.exit(INTERRUPTED)
        if state.state == "invalid":
            sys.exit(f"{case.name} {name}: attempt {state.attempt}'s record is unreadable or "
                     "not this part's: nothing runs")  # fmt: skip
        attempt = 1
    else:
        attempt = args.attempt
        if state.state in ("recorded", "invalid"):
            sys.exit(f"{case.name} {name}: attempt {state.attempt} is recorded (or its "
                     "record unreadable): a recorded attempt, whatever its kind, is never "
                     "rerun")  # fmt: skip
        allowed = 1 if state.state == "due" else state.attempt + 1
        if attempt != allowed:
            sys.exit(f"{case.name} {name}: the next attempt is {allowed}, not {attempt}")
        if cycle is not None:  # a re-qualification after the engineering cycle
            if attempt > 1 and (not args.engineering_cycle or reason):
                sys.exit(f"{case.name} {name}: its attempts are superseded by the engineering "
                         f"cycle ({cycle[0].name}): its rerun carries --engineering-cycle, not "
                         "--infrastructure")  # fmt: skip
        elif args.engineering_cycle:
            sys.exit(f"{case.name} {name}: no engineering cycle supersedes this part")
        elif (attempt > 1) != bool(reason):
            sys.exit("an attempt above 1 reruns an interrupted one, or one a case restart "
                     "names, and needs --infrastructure REASON; attempt 1 takes none")  # fmt: skip
        folder = case_folder(out, case)
        if (name == "rest" and state.state == "interrupted"
                and ("rest", state.attempt) not in named_attempts(folder)):  # fmt: skip
            results = interrupted_rest(folder, state.attempt)
            kind = (
                None
                if results is None
                else failure_kind(
                    "rest", results, recorded_baselines(folder), scope_of(), case.candidate
                )
            )
            if kind is not None:
                sys.exit(f"{case.name} rest: attempt {state.attempt}'s trajectory results hold a "
                         f"{kind} failure, which keeps counting: it is never rerun; record the "
                         "attempt with interrupted --technical REASON")  # fmt: skip
    if name == "rest" and part_state(out, case, "baseline").state == "voided":
        sys.exit(f"{case.name} rest: the baseline's attempts are named by the case's restart or "
                 "supersession record: its next baseline attempt runs first")  # fmt: skip
    if args.run_dir is None or args.checkpoints is None:
        sys.exit("part needs --run-dir and --checkpoints")
    require_clean()
    settings = load_settings()
    cfg = load_run_config(planned.run.config, planned.run.overrides)
    runtime.configure_numerics(cfg, args.device)  # first: before any CUDA work
    env = runtime.environment(cfg, args.device)
    path = out / case.attempt_path(name, attempt)
    q.publish(Path(f"{path}.started"), {
        "part": name, "attempt": attempt, "case": case.record(), "run_dir": str(args.run_dir),
        "infrastructure": reason, "engineering_cycle": None if cycle is None else str(cycle[0]),
        "host": socket.gethostname(), "pod_id": q.pod_id(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds")})  # fmt: skip
    infrastructure = engineering_cycle = None
    previous = [{"attempt": n, "state": s} for n, s in state.attempts.items() if n < attempt]
    if reason:  # the record carries its reason from its one publish, before any part reads it
        restarts = [{"path": str(f), "sha256": file_sha256(f)}
                    for f, _ in restart_records(case_folder(out, case))]  # fmt: skip
        infrastructure = {"reason": reason, "previous": previous, "restarts": restarts}
    if cycle is not None and attempt > 1:  # the supersession it re-qualifies after
        engineering_cycle = {"path": str(cycle[0]), "sha256": file_sha256(cycle[0]),
                             "previous": previous}  # fmt: skip
    record = q.run_part(settings, env, case, args.run_dir, out, part=name, attempt=attempt,
                        checkpoints=args.checkpoints,
                        diagnostic_steps=parse_steps(args.diagnostic_steps),
                        infrastructure=infrastructure,
                        engineering_cycle=engineering_cycle)  # fmt: skip
    print(json.dumps({key: record.get(key) for key in ("part", "attempt", "status", "kind",
                                                       "reasons")}))  # fmt: skip


def finish(
    planned: Planned, out: Path, root: Path, scope: q.QualificationScope, *, release: bool
) -> None:
    complete, _, why = case_state(out, planned, scope)
    if not complete:
        sys.exit(f"{planned.case.name}: {why}; its checkpoints are kept")
    technical = technical_reasons(out, planned, scope)
    if technical and not release:
        print(f"{planned.case.name}: a technical verdict; its checkpoints are kept for the "
              f"diagnosis: {technical}", file=sys.stderr)  # fmt: skip
        sys.exit(TECHNICAL_CASE)
    removed = remove_checkpoints(planned, journal_run_ids(planned.run, root))
    if planned.harness.exists():
        shutil.rmtree(planned.harness)
        removed.append(str(planned.harness))
    print(json.dumps({"case": planned.case.name, "state": why, "removed": removed}))


def interrupted(args, planned: Planned, out: Path, scope: q.QualificationScope) -> None:
    """An interrupted attempt recorded as a technical part record with ``--technical REASON``
    ("Failure routes": an unexplained incomplete record is technical): the case, its launcher
    run as the ``.started`` file names it, the policy, and for a rest the baseline record it was
    built on (the latest baseline then); never over a record. A rest attempt that published
    trajectory results carries them, bound as they were (their launcher run, baseline record,
    bridge checks, code and settings): what they show keeps counting, a scientific failure
    included."""
    case, name, attempt = planned.case, args.part, args.attempt
    state = attempt_files((out / case.attempt_path(name, 1)).parent, name)
    if state.get(attempt) != "interrupted" or attempt != max(state):
        sys.exit(f"{case.name} {name}: attempt {attempt} is not an interrupted attempt (the "
                 f"latest, started and without a record): {state}")  # fmt: skip
    require_clean()
    path = out / case.attempt_path(name, attempt)
    started = _read(Path(f"{path}.started")) or {}
    run_dir = started.get("run_dir")
    reason = f"interrupted, recorded technical: {args.technical}"
    record: dict[str, Any] = {
        "part": name, "attempt": attempt, "case": case.record(),
        "K_star": scope.k_star(case.candidate), "bound": scope.bound(case.candidate),
        "launcher_run": run_dir, "status": "failed", "kind": q.TECHNICAL, "reasons": [reason],
        "launcher": {"kind": None, "reasons": [],
                     **(q.launcher_record(Path(run_dir)) if run_dir
                        else {"identity": None, "pod_id": None})},
        "interrupted": {"reason": args.technical, "started": started},
    }  # fmt: skip
    # The Pod is the one the attempt started on (its ``.started`` file), not this process's
    record["provenance"] = q.qualification_provenance(load_settings(),
                                                      pod_id=started.get("pod_id"))  # fmt: skip
    results = interrupted_rest(case_folder(out, case), attempt) if name == "rest" else None
    if results is not None:  # what the attempt published keeps counting, bound as it was
        record.update({key: results.get(key) for key in (
            "launcher", "baseline_record", "baseline_checkpoint_sha256", "bridge",
            "v0_unchanged", "trajectories", "comparisons", "from_trajectories",
            "provenance")})  # fmt: skip
        record["reasons"] += results.get("reasons") or []  # results bound to different inputs
        record["provenance"] = {**(results.get("provenance") or {}),
                                "pod_id": started.get("pod_id")}  # fmt: skip
    elif name == "rest":
        base = (out / case.attempt_path("baseline", 1)).parent
        found = max((n for n, s in attempt_files(base, "baseline").items() if s == "recorded"),
                    default=None)  # fmt: skip
        if found is not None:
            source = out / case.attempt_path("baseline", found)
            baseline = _read(source) or {}
            record["baseline_record"] = {"path": str(source),
                                         "sha256": q.record_sha256(baseline)}  # fmt: skip
            record["baseline_checkpoint_sha256"] = (baseline.get("checkpoint") or {}).get("sha256")
    q.publish(path, record)
    print(json.dumps({key: record[key] for key in ("part", "attempt", "status", "kind",
                                                   "reasons")}))  # fmt: skip


def scope_of() -> q.QualificationScope:
    return q.QualificationScope.of(load_settings())


def recorded_baselines(folder: Path) -> dict[str | None, dict | None]:
    """The case's recorded baseline attempts, by their record's sha256."""
    out: dict[str | None, dict | None] = {}
    for n, state in attempt_files(folder, "baseline").items():
        if state == "recorded":
            path = folder / f"baseline-attempt-{n}.json"
            out[file_sha256(path)] = _read(path)
    return out


def failure_kind(
    part: str, record: dict, baselines: dict, scope: q.QualificationScope, candidate: int
) -> str | None:
    """Whether a part's record (or an interrupted rest's trajectory results) is a completed
    failure, and of which kind: a baseline its verdict alone fails (scientific: a declared
    instability rule, the launcher run's instability stop; technical: any reason but the rest
    still to come); a rest failing with the baseline record it names, or technical when it names
    none recorded; a control not accepted. None for a record that passes so far."""
    if part == "baseline":
        verdict = q.case_verdict(record, None, scope)
        if verdict["kind"] == q.SCIENTIFIC:
            return q.SCIENTIFIC
        return q.TECHNICAL if [r for r in verdict["reasons"] if r != REST_MISSING] else None
    if part == "rest":
        bound = baselines.get((record.get("baseline_record") or {}).get("sha256"))
        if bound is None:
            return q.SCIENTIFIC if record.get("kind") == q.SCIENTIFIC else q.TECHNICAL
        return q.case_verdict(bound, record, scope)["kind"]
    verdict = q.control_verdict(record, scope, candidate)
    return None if verdict["state"] == q.ACCEPTED else verdict["kind"]


def attempt_failures(folder: Path, p: Planned, scope: q.QualificationScope) -> list[list[str]]:
    """The case's attempts no restart may name (module docstring, "A case restart"), as
    [label, kind]: every recorded attempt that is a completed failure, scientific or technical
    (``failure_kind``), or that the verdicts cannot read, and every interrupted rest attempt
    whose trajectory results are one; attempts a restart or supersession names already are
    void and left out."""
    named, listed = named_records(folder), named_attempts(folder)
    baselines = recorded_baselines(folder)
    found: list[list[str]] = []
    for name in ["baseline", "rest", *(["control"] if p.control else [])]:
        for n, state in attempt_files(folder, name).items():
            path = folder / f"{name}-attempt-{n}.json"
            label = f"{name} attempt {n}"
            if state == "recorded":
                if file_sha256(path) in named:
                    continue
                record = _read(path)
            elif name == "rest" and (name, n) not in listed:
                record = interrupted_rest(folder, n)
                if record is None:
                    continue
                label += " (its trajectory results)"
            else:
                continue
            if record is None:
                found.append([label, "unreadable"])
                continue
            try:
                kind = failure_kind(name, record, baselines, scope, p.case.candidate)
            except Exception as exc:  # a record the verdicts cannot read is never voided
                found.append([label, f"unreadable ({type(exc).__name__})"])
                continue
            if kind is not None:
                found.append([label, kind])
    return found


def case_attempts(folder: Path, p: Planned) -> list[dict[str, Any]]:
    """Every attempt of the case's parts as a voiding record names it."""
    attempts = []
    for name in ["baseline", "rest", *(["control"] if p.control else [])]:
        for n, state in attempt_files(folder, name).items():
            path = folder / f"{name}-attempt-{n}.json"
            record = _read(path) if state == "recorded" else None
            attempts.append({"part": name, "attempt": n, "state": state, "path": str(path),
                             "sha256": file_sha256(path) if state == "recorded" else None,
                             "status": (record or {}).get("status"),
                             "kind": (record or {}).get("kind")})  # fmt: skip
    return attempts


def launcher_to_supersede(planned: Planned, root: Path) -> tuple[Path | None, dict | None]:
    """The case's settled launcher run (``located``) and how a voiding record names it."""
    try:
        run_dir = located(planned.run, root)
        if run_dir is None:
            return None, None
        return run_dir, {"run_id": run_dir.name, "identity": q.launcher_identity(run_dir)}
    except Exception as exc:  # the record names what it can
        return None, {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}


def log_superseded(
    planned: Planned, root: Path, campaign: str, run_dir: Path | None, *, reason: str, record: Path
) -> None:
    """The launcher run, once the voiding record that names it is published, logged superseded in
    the launch journal (``launch.supersede``): kept, it no longer counts for its label, and the
    launcher's recovery round starts a new attempt. Done once."""
    if run_dir is None:
        return
    if run_dir.name not in superseded_runs(planned.run, launch_records(root)):
        supersede(root, campaign, planned.run, run_dir.name, reason=reason, record=str(record))


def checkpoint_missing(run_dir: Path | None) -> bool:
    """Whether the launcher run's recorded step-25 checkpoint is gone from disk."""
    if run_dir is None:
        return False
    record = (_read(run_dir / "summary.json") or {}).get("endpoint_checkpoint")
    return isinstance(record, dict) and not Path(str(record.get("path"))).is_file()


def next_number(folder: Path, kind: str) -> int:
    return max((int(VOIDING_FILE.fullmatch(f.name)[2]) for f, _ in voiding_records(folder, kind)),
               default=0) + 1  # fmt: skip


def restart_case(
    args, planned: Planned, out: Path, root: Path, scope: q.QualificationScope, campaign: str
) -> None:
    """A case restart (module docstring, "A case restart"): a record, published atomically, that
    names every attempt of the case's parts (the baseline, the rest and its control if it holds
    one), each recorded one with its record's sha256, and the infrastructure reason; the
    attempts it names are void, and the case's launcher run is then logged superseded. Refused
    without a reason, for a complete case, with no attempt it does not name already, without
    evidence of an interruption (an interrupted attempt, or the launcher run's checkpoint gone
    from disk), when the launcher run itself is a completed failure (``launcher_problems``:
    its declared instability stop, scientific; another stop or a departing run, technical), and
    when any attempt is a completed failure, scientific or technical, or unreadable
    (``attempt_failures``): only an interrupted case is restarted. A launcher run whose
    checkpoint is lost is evidence enough, and restarts the case even before any attempt (the
    Pod stopped between the launch and the baseline). Run again after a stop between the record
    and the journal's line, it only writes that line."""
    case, reason = planned.case, args.infrastructure
    if not reason:
        sys.exit("restart-case needs --infrastructure REASON: the documented interruption")
    folder = case_folder(out, case)
    attempts, listed = case_attempts(folder, planned), named_attempts(folder)
    fresh = [a for a in attempts if (a["part"], a["attempt"]) not in listed]
    earlier = restart_records(folder)
    superseded = superseded_runs(planned.run, launch_records(root))
    named_run = ((earlier[-1][1] or {}).get("launcher") or {}).get("run_id") if earlier else None
    if named_run and named_run not in superseded:  # stopped before the journal's line
        path, record = earlier[-1]
        log_superseded(planned, root, campaign, root / planned.run.experiment / named_run,
                       reason=str((record or {}).get("reason")), record=path)  # fmt: skip
        sys.exit(f"{case.name}: the restart is recorded ({path.name}); the launch journal now "
                 "names its launcher run")  # fmt: skip
    run_dir, launcher = launcher_to_supersede(planned, root)
    missing = checkpoint_missing(run_dir)
    if not fresh and not missing:  # a lost checkpoint alone is evidence, before any attempt
        sys.exit(f"{case.name}: no attempt to restart (none recorded or started, or every one "
                 "named by an earlier restart or supersession record), and the launcher run's "
                 "checkpoint is not lost")  # fmt: skip
    if case_state(out, planned, scope)[0]:
        sys.exit(f"{case.name}: the case is complete; a restart voids only an interrupted case")
    failures = attempt_failures(folder, planned, scope)
    if failures:
        sys.exit(f"{case.name}: no restart: a completed failure keeps counting: {failures}. A "
                 "scientific one is never rerun (no reruns to pass); a technical one is lifted "
                 "only by an engineering cycle's supersession record (supersede)")  # fmt: skip
    if run_dir is not None:
        kind, why = q.launcher_problems(run_dir, case, checkpoint=False, environment=False)
        if kind is not None:
            sys.exit(f"{case.name}: no restart: the launcher run {run_dir.name} is itself a "
                     f"completed {kind} failure, which keeps counting: {why}")  # fmt: skip
    if not any(a["state"] == "interrupted" for a in fresh) and not missing:
        sys.exit(f"{case.name}: no restart: no evidence of an interruption (no interrupted "
                 "attempt, and the launcher run's checkpoint is on disk)")  # fmt: skip
    require_clean()
    number = next_number(folder, "restart")
    path = folder / f"restart-{number}.json"
    record = {"case": case.record(), "restart": number, "reason": reason, "attempts": attempts,
              "launcher": launcher, "checkpoint_missing": missing,
              "host": socket.gethostname(),
              "restarted_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "provenance": q.qualification_provenance(load_settings())}  # fmt: skip
    q.publish(path, record)  # first the record, then the journal's line that points to it
    log_superseded(planned, root, campaign, run_dir, reason=reason, record=path)
    print(json.dumps({"case": case.name, "restart": str(path), "reason": reason,
                      "attempts": [(a["part"], a["attempt"]) for a in attempts],
                      "launcher": launcher}))  # fmt: skip


def scientific_attempts(
    folder: Path, p: Planned | None, scope: q.QualificationScope, *, candidate: int
) -> set[tuple[str, int]]:
    """The case's attempts an engineering cycle keeps: every scientific failure (a record, or an
    interrupted rest's trajectory results; ``attempt_failures``), and the baseline record a kept
    rest is bound to, without which it could not be read. Only a record's own kind is read for a
    case of no campaign run."""
    keep: set[tuple[str, int]] = set()
    if p is None:
        for part_name in q.PARTS:
            for n, state in attempt_files(folder, part_name).items():
                record = _read(folder / f"{part_name}-attempt-{n}.json") or {}
                if state == "recorded" and record.get("kind") == q.SCIENTIFIC:
                    keep.add((part_name, n))
    else:
        for label, kind in attempt_failures(folder, p, scope):
            if kind == q.SCIENTIFIC:
                part_name, _, n = label.split(" (")[0].split(" ")
                keep.add((part_name, int(n)))
    for part_name, n in list(keep):
        if part_name != "rest":
            continue
        record = _read(folder / f"rest-attempt-{n}.json")
        if record is None:  # an interrupted attempt: the baseline its results name
            record = interrupted_rest(folder, n) or {}
        named = (record.get("baseline_record") or {}).get("sha256")
        for m, state in attempt_files(folder, "baseline").items():
            if state == "recorded" and file_sha256(folder / f"baseline-attempt-{m}.json") == named:
                keep.add(("baseline", m))
    return keep


def kept_records(folder: Path) -> set[str]:
    """The sha256 of the case's records the candidate's engineering cycle kept: every record of
    a case that holds a scientific failure, which it keeps whole."""
    case, stage = case_name_of(folder), stage_folder(folder).name
    return {e["sha256"] for _, r in voiding_records(folder, "supersede")
            for e in (r or {}).get("kept") or []
            if isinstance(e, dict) and e.get("case") == case and e.get("stage", stage) == stage
            and e.get("sha256")}  # fmt: skip


def journal_starts(root: Path) -> dict[str, tuple[str, dict]]:
    """Every launcher attempt in the launch journals, by run id: (campaign, its start record)."""
    starts: dict[str, tuple[str, dict]] = {}
    for log in sorted((root / "launch").glob("*.jsonl")):
        for line in log.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("event") == "start" and isinstance(
                    record.get("run_id"), str):  # fmt: skip
                starts.setdefault(record["run_id"], (log.stem, record))
    return starts


def run_gpu_code(run_dir: Path) -> str | None:
    """The GPU-code identity (sha256) of the commit a launcher run recorded, from a clean
    checkout; None when it cannot be told."""
    git = (_read(run_dir / "meta.json") or {}).get("git") or {}
    if not git.get("commit") or git.get("dirty") is not False:
        return None
    try:
        return code_identity.gpu_code_identity(git["commit"]).get("sha256")
    except Exception:  # an unknown commit: its code cannot be told
        return None


def supersede_case(
    args, chosen: Planned, planned: list[Planned], out: Path, root: Path,
    scope: q.QualificationScope, campaign: str,
) -> None:  # fmt: skip
    """The candidate's engineering cycle, recorded (module docstring, "Supersession"; "Failure
    routes", "Voiding and supersession"): ``c<N>/supersede-1.json`` names the technical record
    that started it (its path and sha256), the diagnosis and the fix, the GPU code of the fix
    (this clean checkout's) and the old one, and every record of the candidate made under the
    old GPU code, in all its stage folders, passed ones included, with the cases re-qualified,
    and every old-code launcher run of the candidate in the launch journal (the runs its
    records name and those of the campaign's cases, a case without any part attempt included),
    which it then logs superseded. A case that holds a scientific failure is kept whole (none of
    its records, nor its launcher run, is voided: the failure keeps counting, no reruns to pass)
    and listed as kept, with why. Refused: a second time for the candidate (one cycle per
    candidate: an unresolved technical failure then routes to the technical feasibility report);
    for a record that is not a completed technical failure of the case; when the fix's GPU code
    is the old one (the same path is not simply rerun); from a checkout that is not clean. Run
    again with the same record, it only writes the journal's lines that are missing."""
    case = chosen.case
    if args.record is None or not args.diagnosis or not args.fix:
        sys.exit("supersede needs --record PATH, --diagnosis TEXT and --fix TEXT")
    folder = case_folder(out, case)
    candidate_dir = candidate_folder(folder)
    path = args.record.resolve()
    sha = file_sha256(path)
    cycles = sorted(candidate_dir.glob("supersede-*.json"))
    if cycles:
        published = _read(cycles[0]) or {}
        old_record = published.get("old_record") or {}
        if old_record.get("path") == str(path) and old_record.get("sha256") == sha:
            repaired = repair_cycle_journal(cycles[0], published, root)
            print(json.dumps({"supersession": str(cycles[0]), "journal_lines_written": repaired}))
            return
        sys.exit(f"candidate {case.candidate}: its engineering cycle is recorded already "
                 f"({cycles[0]}): one cycle per candidate, and a technical failure left "
                 "unresolved after it routes to the technical feasibility report")  # fmt: skip
    match = re.fullmatch(r"(baseline|rest|control)-attempt-(\d+)\.json", path.name)
    if (not match or path.parent != folder.resolve()
            or attempt_files(folder, match[1]).get(int(match[2])) != "recorded"):  # fmt: skip
        sys.exit(f"{case.name}: {args.record} is not a recorded attempt of this case")
    if sha in named_records(folder):
        sys.exit(f"{case.name}: {path.name} is void already")
    failures = attempt_failures(folder, chosen, scope)
    label = f"{match[1]} attempt {match[2]}"
    if dict(map(tuple, failures)).get(label) != q.TECHNICAL:
        sys.exit(f"{case.name}: {path.name} is not a completed technical failure: an "
                 "engineering cycle supersedes only one")  # fmt: skip
    old = ((_read(path) or {}).get("provenance") or {}).get("gpu_code") or {}
    require_clean()
    fix = code_identity.gpu_code_identity()
    if not old.get("sha256") or fix.get("sha256") == old.get("sha256"):
        sys.exit(f"{case.name}: the fix's GPU code is the superseded record's "
                 f"({old.get('sha256')}): the same path is not simply rerun")  # fmt: skip
    by_name = {(p.case.name, ROLES[p.case.engineering]): p for p in planned}
    attempts, requalified, kept, kept_cases, run_ids, kept_runs = [], [], [], [], set(), set()
    for stage in sorted(d for d in candidate_dir.iterdir() if d.is_dir()):
        for case_dir in sorted({f.parent for f in stage.glob("s*/*/*-attempt-*")}):
            name = case_name_of(case_dir)
            p = by_name.get((name, stage.name))
            entries, mine = [], []
            named = named_records(case_dir)
            for part_name in q.PARTS:
                for n, state in attempt_files(case_dir, part_name).items():
                    file = case_dir / f"{part_name}-attempt-{n}.json"
                    record = _read(file) if state == "recorded" else None
                    identity = ((record or {}).get("launcher") or {}).get("identity") or {}
                    entry = {"case": name, "stage": stage.name, "part": part_name,
                             "attempt": n, "state": state, "path": str(file),
                             "sha256": file_sha256(file) if state == "recorded" else None,
                             "status": (record or {}).get("status"),
                             "kind": (record or {}).get("kind"),
                             "launcher_run_id": identity.get("run_id")}  # fmt: skip
                    entries.append(entry)
                    if state == "recorded" and (entry["sha256"] in named or (
                            ((record or {}).get("provenance") or {}).get("gpu_code") or {}
                    ).get("sha256") != old["sha256"]):  # fmt: skip
                        continue  # void already, or not of the old code
                    mine.append(entry)
            keep = scientific_attempts(case_dir, p, scope, candidate=case.candidate)
            if keep:  # a scientific failure keeps the case whole, its launcher run included
                kept += entries
                why = sorted(f"{part_name} attempt {n}" for part_name, n in keep)
                kept_cases.append(
                    {
                        "case": name,
                        "stage": stage.name,
                        "why": f"a scientific failure keeps counting: {why}",
                    }
                )
                kept_runs |= {e["launcher_run_id"] for e in entries if e["launcher_run_id"]}
                if p is not None:
                    kept_runs |= set(journal_run_ids(p.run, root))
                continue
            if any(e["state"] == "recorded" for e in mine):
                attempts += mine
                requalified.append({"case": name, "stage": stage.name})
                run_ids |= {e["launcher_run_id"] for e in mine if e["launcher_run_id"]}
    kept_names = {(k["case"], k["stage"]) for k in kept_cases}
    for (name, stage_name), p in by_name.items():  # every campaign case: its launcher runs
        if (name, stage_name) not in kept_names:
            run_ids |= set(journal_run_ids(p.run, root))
    starts = journal_starts(root)
    launchers = []
    for run_id in sorted(run_ids - kept_runs):
        if run_id not in starts:
            continue
        campaign_name, start = starts[run_id]
        run_dir = root / str(start.get("experiment")) / run_id
        if run_dir.is_dir() and run_gpu_code(run_dir) == old["sha256"]:
            launchers.append({"run_id": run_id, "campaign": campaign_name,
                              "run": start.get("run"), "experiment": start.get("experiment"),
                              "config_hash": start.get("config_hash"),
                              "identity": q.launcher_identity(run_dir)})  # fmt: skip
    record_path = candidate_dir / "supersede-1.json"
    record = {"candidate": case.candidate, "supersession": 1,
              "old_record": {"case": case.name, "stage": stage_folder(folder).name,
                             "path": str(path), "sha256": sha, "part": match[1],
                             "attempt": int(match[2])},
              "diagnosis": args.diagnosis, "fix": args.fix, "old_gpu_code": old,
              "fix_gpu_code": fix, "cases_requalified": requalified, "attempts": attempts,
              "kept_cases": kept_cases, "kept": kept, "launchers": launchers,
              "host": socket.gethostname(),
              "superseded_at": datetime.now(UTC).isoformat(timespec="seconds"),
              "provenance": q.qualification_provenance(load_settings())}  # fmt: skip
    q.publish(record_path, record)  # first the record, then the journal's lines
    repair_cycle_journal(record_path, record, root)
    print(json.dumps({"candidate": case.candidate, "supersession": str(record_path),
                      "old_record": record["old_record"], "cases_requalified": requalified,
                      "kept_cases": kept_cases,
                      "launchers": [r["run_id"] for r in launchers]}))  # fmt: skip


def repair_cycle_journal(path: Path, record: dict, root: Path) -> list[str]:
    """The journal's ``superseded`` line of every launcher run the supersession names, where it
    is missing; returns the run ids written."""
    reason = (
        f"superseded by an engineering cycle: {record.get('diagnosis')}; fix: {record.get('fix')}"
    )
    written = []
    for entry in record.get("launchers") or []:
        run = SimpleNamespace(label=entry.get("run"), experiment=entry.get("experiment"),
                              config_hash=entry.get("config_hash"))  # fmt: skip
        if entry["run_id"] not in superseded_runs(run, launch_records(root)):
            supersede(root, entry["campaign"], run, entry["run_id"], reason=reason,
                      record=str(path))  # fmt: skip
            written.append(entry["run_id"])
    return written


def superseded_run_ids(run: Run, root: Path) -> list[str]:
    """The attempts of the run that the launch journal shows superseded: named so by a case
    restart or supersession (``launch.supersede``), or started, and ended technical or not ended
    at all, with no settled directory (completed, whether it stands or is in conflict, or
    diverged). A completed run that cannot stand keeps its checkpoint for the diagnosis."""
    settled_outcomes = {r.get("run_id") for r in launch_records(root)
                        if r.get("event") == "end" and r.get("run") == run.label
                        and r.get("outcome") in ("done", "diverged", "conflict")}  # fmt: skip
    superseded = []
    named = superseded_runs(run, launch_records(root))  # a restart or supersession names them
    for run_id in journal_run_ids(run, root):
        if run_id in named:
            superseded.append(run_id)
            continue
        run_dir = root / run.experiment / run_id
        meta = _read(run_dir / "meta.json") or {}
        if (run_id in settled_outcomes or meta.get("status") in ("completed", "diverged")
                or (run_dir / "terminal.json").is_file()):  # fmt: skip
            continue
        superseded.append(run_id)
    return superseded


def clear_stale(planned: Planned, root: Path) -> None:
    stale = superseded_run_ids(planned.run, root)
    print(json.dumps({"case": planned.case.name, "removed": remove_checkpoints(planned, stale)}))


def latest(folder: Path, part: str, named: dict[str, str] | None = None) -> dict | None:
    """The record of a part's latest attempt in a case's folder, or None (none, interrupted,
    unreadable, or named by a restart record: ``named``)."""
    attempts = attempt_files(folder, part)
    if not attempts or attempts[max(attempts)] != "recorded":
        return None
    path = folder / f"{part}-attempt-{max(attempts)}.json"
    if file_sha256(path) in (named or {}):
        return None
    return _read(path)


def attempt_listing(
    folder: Path, named: dict[str, str]
) -> tuple[dict[str, list], list[tuple[str, int, dict | None, str | None]]]:
    """Every attempt of every part in a case's folder (number, state, path, the record's sha256,
    status, kind and, when a restart record names it, the restart's reason), and the superseded
    records with that reason or None: a record a restart names (void), and one under a later
    attempt that no restart names, which the commands here never write and which is not hidden
    by it."""
    listing, superseded = {}, []
    cycles = named_records(folder, "supersede")
    for part in q.PARTS:
        attempts = attempt_files(folder, part)
        rows = []
        for n, state in attempts.items():
            path = folder / f"{part}-attempt-{n}.json"
            record = _read(path) if state == "recorded" else None
            sha = file_sha256(path) if state == "recorded" else None
            void = named.get(sha) if sha is not None else None
            cycle = void is not None and sha in cycles
            restart, supersession = (None, void) if cycle else (void, None)
            rows.append({"attempt": n, "state": state, "path": str(path), "sha256": sha,
                         "status": (record or {}).get("status"),
                         "kind": (record or {}).get("kind"), "restart": restart,
                         "supersession": supersession})  # fmt: skip
            if state == "recorded" and (void is not None or n < max(attempts)):
                superseded.append((part, n, record, restart, supersession))
        if rows:
            listing[part] = rows
    return listing, superseded


def verdict(args, settings, scope, planned: list[Planned], out: Path, root: Path) -> Path:
    candidate, stage = args.candidate, ROLES[args.engineering]
    base = out / f"c{candidate}" / stage
    folders = sorted({f.parent for f in base.glob("s*/*/*-attempt-*")})
    cases, records, attempts, superseded = [], [], {}, []
    old_controls: list[dict[str, Any]] = []  # the superseded controls' verdicts
    voided: dict[str, str] = {}  # a void record's verdict (record_sha256) → the restart's reason
    warnings: list[str] = []
    for folder in folders:
        name = str(folder.relative_to(base))
        named = named_records(folder)
        attempts[name], stale = attempt_listing(folder, named)
        baseline, rest = latest(folder, "baseline", named), latest(folder, "rest", named)
        rests = attempt_files(folder, "rest")
        if (rests and rests[max(rests)] == "interrupted"
                and ("rest", max(rests)) not in named_attempts(folder)):  # fmt: skip
            rest = interrupted_rest(folder, max(rests))  # its trajectory results count
        if baseline is not None or rest is not None:
            records += [r for r in (baseline, rest) if r is not None]
            cases.append(q.case_verdict(baseline, rest, scope, e_eng=args.e_eng))
        for part, n, record, restart, supersession in stale:  # shown; void, or counted
            void = restart or supersession
            mark = {"part": part, "attempt": n, "restart": restart, "supersession": supersession}
            superseded.append({"case": name, "sha256": q.record_sha256(record), **mark})
            if part == "control":  # counted against its rate as a superseded case is
                checked = {**q.control_verdict(record, scope, candidate), "superseded": mark}
                old_controls.append(checked)
                if void is not None and checked["record_sha256"] is not None:
                    voided[checked["record_sha256"]] = void
                continue
            pair = (record, None) if part == "baseline" else (None, record)
            verdict_ = {**q.case_verdict(*pair, scope, e_eng=args.e_eng), "superseded": mark}
            cases.append(verdict_)
            if void is not None:
                voided[verdict_["record_sha256"]] = void
    campaign = {"path": str(args.campaign.resolve()),
                "sha256": hashlib.sha256(args.campaign.read_bytes()).hexdigest()}  # fmt: skip
    body: dict[str, Any] = {"candidate": candidate, "stage": stage, "campaign": campaign,
                            "e_eng": args.e_eng, "cases": cases, "attempts": attempts,
                            "superseded": superseded}  # fmt: skip
    if args.engineering:
        body["summary"] = q.engineering_summary(records)
        target = out / f"c{candidate}" / "engineering" / "verdict.json"
    else:
        code, policy = code_identity.gpu_code_identity(), q.settings_record(settings)
        rates, states, assigned = {}, {}, set()
        for optimizer, lr in dict.fromkeys((p.case.optimizer, p.case.lr) for p in planned):
            key = f"{optimizer}-{lr:g}"
            mine = [
                v
                for v in cases
                if (v["case"] or {}).get("optimizer") == optimizer
                and math.isclose((v["case"] or {}).get("lr") or math.nan, lr, rel_tol=1e-9)
            ]
            assigned |= {id(v) for v in mine}
            control = None
            holder = next((p for p in planned if p.control and p.case.lr == lr
                           and p.case.optimizer == optimizer), None)  # fmt: skip
            if holder is not None:
                where = case_folder(out, holder.case)  # a restart-named control is not current
                record = latest(where, "control", named_records(where))
                control = None if record is None else q.control_verdict(record, scope, candidate)
            runs = {p.case.seed: located(p.run, root) for p in planned
                    if (p.case.optimizer, p.case.lr) == (optimizer, lr)}  # fmt: skip
            launchers = {seed: None if d is None else q.launcher_identity(d)
                         for seed, d in runs.items()}  # fmt: skip
            if any(identity is None for identity in launchers.values()):
                warnings.append(f"{key}: launcher runs not located "
                                f"({sorted(s for s, i in launchers.items() if i is None)}): "
                                "the rate is at most incomplete")  # fmt: skip
            olds = [c for c in old_controls if (c["case"] or {}).get("optimizer") == optimizer
                    and math.isclose((c["case"] or {}).get("lr") or math.nan, lr,
                                     rel_tol=1e-9)]  # fmt: skip
            assigned |= {id(c) for c in olds}
            rates[key] = q.rate_verdict(mine, scope=scope, candidate=candidate,
                                        optimizer=optimizer, lr=lr, code=code, policy=policy,
                                        control=control, superseded_controls=olds,
                                        launchers=launchers, voided=voided)  # fmt: skip
            states[key] = rates[key]["state"]
            warnings += [f"{key}: τ exceeded by {name}" for name in rates[key]["warnings"]]
        body.update(
            states=states,
            rates=rates,
            unassigned=[
                (v["case"] or {}).get("name")
                for v in [*cases, *old_controls]
                if id(v) not in assigned
            ],
        )
        target = out / f"c{candidate}" / "verdict.json"
    body.update(warnings=warnings, provenance=q.qualification_provenance(settings))
    q.publish(target, body)
    for key, state in (body.get("states") or {}).items():
        print(f"{key}: {state}")
    return target


def driver_script(
    commit: str, written_at: str, campaign: Path, args, planned: list[Planned], settings,
    steps: list[int],
) -> str:  # fmt: skip
    """The bash driver of the qualification on a Pod (module docstring)."""
    stamp = "$(date -u +%FT%TZ)"
    sha = hashlib.sha256(campaign.read_bytes()).hexdigest()
    need = disk_need_gb(settings, steps)
    host = float(settings.host_memory_gb)
    stage = ROLES[args.engineering]
    common = f"--campaign \"$CAMPAIGN\" --candidate {args.candidate}" + (
        " --engineering" if args.engineering else "")  # fmt: skip
    diagnostic = f" --diagnostic-steps {','.join(map(str, steps))}" if steps else ""
    endpoint = planned[0].endpoint
    lines = [
        "#!/usr/bin/env bash",
        *comment(f"Written by scripts/qualify.py driver at {written_at}; do not edit."),
        *comment(f"Study 3's numerical qualification ({stage}) of candidate {args.candidate} "
                 f"from commit {commit}: {len(planned)} cases of the campaign {campaign}, one at "
                 "a time: the launcher run alone (one attempt), the baseline part, then, unless "
                 "the baseline is already a scientific failure, the rest, and the FA rate's "
                 "same-path control, each in a fresh process; then the case's checkpoints are "
                 "deleted, unless its verdict is technical: then the driver stops (exit "
                 f"{TECHNICAL_CASE}) with them kept for the diagnosis. K-only and blind to "
                 "outcomes."),
        *comment(f"A case needs {need:g} GB free under {endpoint} (the container disk). A "
                 "complete case is skipped, so the driver can be run again after a stop. It "
                 "never reruns a part or a launcher run: an interrupted part (exit "
                 f"{INTERRUPTED}) is rerun by hand as its next attempt with --infrastructure "
                 "REASON, a failed launcher run with scripts/launch.py --only LABEL "
                 "--max-attempts 1 --recovery REASON. The verdict is written on every exit."),
        *comment(f"Exits: 0 every case complete; 1 a precondition (the host memory "
                 f"included), the disk or a case finish refuses (not complete); {INTERRUPTED} an "
                 f"interrupted attempt (or a part a case restart names); {TECHNICAL_CASE} a "
                 f"technical case; {STALE} a stale case (records of other GPU code or settings); "
                 "the launcher's own when it stops; another a failed command."),
        "set -euo pipefail",
        *POD_ENVIRONMENT,
        f"SRC={SOURCES}/{commit}/source",
        'cd "$SRC"',
        f'test "$(git rev-parse HEAD)" = {commit}',
        f'project-run "$SRC" python -c "{CLEAN_CHECKOUT}"',
        f"CAMPAIGN={shlex.quote(str(campaign))}",
        f'test "$(sha256sum "$CAMPAIGN" | cut -d " " -f 1)" = {sha}',
        f"CKPT={shlex.quote(str(endpoint))}",
        'OUT="${RUNS_DIR:?}/qualification"',
        f'mkdir -p "$OUT/c{args.candidate}" "$CKPT"',
        f'exec > >(tee -a "$OUT/c{args.candidate}/driver-{stage}-$(date -u +%Y%m%dT%H%M%SZ).log")'
        " 2>&1",
        f'step() {{ echo "== {stamp} start: $*"; "$@"; echo "== {stamp} done: $*"; }}',
        f'stop() {{ echo "== {stamp} $1" >&2; exit "$2"; }}',
        "# The verdict on every exit, whatever stopped the driver",
        f"trap 'rc=$?; {RUN} verdict {common} || true; exit $rc' EXIT",
        "# No GPU job beside the qualification: no compute process, and (as a container may not",
        "# see other processes) no GPU memory in use",
        'if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then',
        '  stop "a process holds the GPU; the qualification does not start" 1',
        "fi",
        "used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)",
        f"if echo \"$used\" | awk '$1 + 0 >= {GPU_USED_MIB} {{busy = 1}} END {{exit !busy}}'; then",
        '  stop "GPU memory in use ($used MiB); the qualification does not start" 1',
        "fi",
        *comment("The host holds θ₀, v₀, the second model and the p1 direction, and a "
                 "checkpoint save's transient (host_memory_gb; the free memory: MemAvailable, "
                 "or the cgroup's when smaller)"),
        f'if ! project-run "$SRC" python -c "{HOST_FREE}" {host:g}; then',
        f'  stop "{host:g} GB of free host memory are needed; the qualification does not '
        'start" 1',
        "fi",
    ]  # fmt: skip
    for index, p in enumerate(planned, 1):
        case = f"--case {shlex.quote(p.case.name)}"
        parts = f'{common} {case} --run-dir "$RUN_DIR" --checkpoints {shlex.quote(str(p.harness))}'
        lines += [
            f"# ---- case {index}/{len(planned)}: {p.case.name}, launcher run {p.run.label}",
            "rc=0",
            f"{RUN} complete {common} {case} || rc=$?",
            'case "$rc" in',
            f'  0) echo "== {stamp} complete: {p.case.name} is skipped" ;;',
            f'  {STALE}) stop "{p.case.name}: records of other GPU code or settings (stale); '
            f'nothing is rerun or skipped" {STALE} ;;',
            f"  {NO})",
            f"    {RUN} clear-stale {common} {case}",
            "    free_kb=$(df -Pk \"$CKPT\" | awk 'NR == 2 {print $4}')",
            "    case \"$free_kb\" in",
            "      ''|*[!0-9]*) stop \"df gave no free space for $CKPT: '$free_kb'\" 1 ;;",
            "    esac",
            f'    if [ "$free_kb" -lt {need_kb(need)} ]; then',
            f'      stop "{need:g} GB free are needed under $CKPT, $free_kb KB are" 1',
            "    fi",
            f'    step {LAUNCH} "$CAMPAIGN" --only {shlex.quote(p.run.label)} --max-attempts 1',
            f"    RUN_DIR=$({RUN} locate {common} {case})",
            f"    step {RUN} part {parts} --part baseline --resume{diagnostic}",
            "    rc=0",
            f"    {RUN} decided {common} {case} || rc=$?",
            '    case "$rc" in',
            f'      0) echo "== {stamp} {p.case.name}: the baseline is a scientific failure; '
            'no rest" ;;',
            f"      {NO}) step {RUN} part {parts} --part rest --resume{diagnostic} ;;",
            '      *) stop "decided failed (exit $rc)" "$rc" ;;',
            "    esac",
            *([f"    step {RUN} part {parts} --part control --resume"] if p.control else []),
            "    ;;",
            '  *) stop "complete failed (exit $rc)" "$rc" ;;',
            "esac",
            f"# A technical case stops the driver here (exit {TECHNICAL_CASE}), its checkpoints "
            "kept",
            f"step {RUN} finish {common} {case}",
        ]  # fmt: skip
    lines.append(f'echo "== {stamp} every case is complete"')
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        choices=[
            "plan",
            "driver",
            "part",
            "complete",
            "decided",
            "clear-stale",
            "locate",
            "finish",
            "verdict",
            "interrupted",
            "restart-case",
            "supersede",
        ],
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--engineering", action="store_true", help="the engineering stage")
    parser.add_argument("--diagnostic-steps", metavar="T1,T2,...",
                        help="engineering: the steps of the intermediate D")  # fmt: skip
    parser.add_argument("--case", metavar="NAME", help="s<seed>/<optimizer>-<lr>")
    parser.add_argument("--part", choices=list(q.PARTS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="part: the driver's call")
    mode.add_argument("--attempt", type=int, help="part: the next attempt")
    parser.add_argument(
        "--infrastructure",
        metavar="REASON",
        help="part: the documented interruption an attempt above 1 reruns; restart-case: why "
        "the case starts again",
    )
    parser.add_argument("--technical", metavar="REASON",
                        help="interrupted: why the attempt is recorded technical")  # fmt: skip
    parser.add_argument(
        "--record",
        type=Path,
        metavar="PATH",
        help="supersede: the technical record the engineering cycle supersedes",
    )
    parser.add_argument("--diagnosis", metavar="TEXT", help="supersede: the cycle's diagnosis")
    parser.add_argument(
        "--engineering-cycle",
        action="store_true",
        help="part: a rerun after the candidate's engineering cycle",
    )
    parser.add_argument("--fix", metavar="TEXT", help="supersede: the cycle's fix")
    parser.add_argument(
        "--release",
        action="store_true",
        help="finish: delete a technical case's checkpoints once diagnosed",
    )
    parser.add_argument("--run-dir", type=Path, help="part: the launcher run")
    parser.add_argument("--checkpoints", type=Path, help="part: the harness's checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--e-eng", type=float, help="verdict: e_eng, for the warning τ")
    parser.add_argument("--runs-root", type=Path, help="RUNS_DIR")
    parser.add_argument("--out", type=Path, help="the part records (driver: the driver file)")
    parser.add_argument("--commit", help="driver: the source checkout's commit")
    parser.add_argument("--dry-run", action="store_true", help="driver: print it")
    args = parser.parse_args(argv)
    load_env()
    settings = load_settings()
    scope = q.QualificationScope.of(settings)
    steps = parse_steps(args.diagnostic_steps)
    if steps and not args.engineering:
        sys.exit("--diagnostic-steps belong to the engineering stage (--engineering)")
    name, planned, problems = campaign_cases(args.campaign, scope, args.candidate,
                                             args.engineering)  # fmt: skip
    root = args.runs_root or runs_root()
    if args.command == "plan":
        print(plan_text(name, planned, problems, settings, steps))
        if problems or not planned:
            sys.exit(1)
        return
    if args.command == "driver":
        if problems or not planned:
            sys.exit("\n".join(["The campaign is not a qualification campaign:", *problems]))
        script = driver_script(args.commit or "<SHA>",
                               datetime.now(UTC).isoformat(timespec="seconds"),
                               args.campaign.resolve(), args, planned, settings,
                               steps)  # fmt: skip
        if args.dry_run:
            print(script, end="")
            return
        if not args.commit or not args.out or args.out.exists():
            sys.exit("driver needs --commit SHA and an --out FILE that does not exist")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(script)
        print(args.out)
        return
    out = args.out or root / "qualification"
    if args.command in ("part", "verdict", "interrupted", "restart-case", "supersede") and problems:
        sys.exit("\n".join(["The campaign is not a qualification campaign:", *problems]))
    if args.command == "verdict":
        require_clean()
        print(verdict(args, settings, scope, planned, out, root))
        return
    found = [p for p in planned if p.case.name == args.case]
    if len(found) != 1:
        sys.exit(f"The campaign has no case {args.case!r}: {[p.case.name for p in planned]}")
    chosen = found[0]
    if args.command == "part":
        if args.part is None or not (args.resume or args.attempt is not None):
            sys.exit("part needs --part and --resume or --attempt N")
        if args.part == "control" and not chosen.control:
            sys.exit(f"{chosen.case.name} holds no same-path control")
        part(args, chosen, out)
    elif args.command in ("complete", "decided"):
        stale = stale_records(out, chosen, code_identity.gpu_code_identity(),
                              q.settings_record(settings))  # fmt: skip
        if stale:
            print(f"{chosen.case.name}: stale, records of other GPU code or settings than this "
                  f"checkout's: {stale}", file=sys.stderr)  # fmt: skip
            sys.exit(STALE)
        complete, decided, why = case_state(out, chosen, scope)
        print(f"{chosen.case.name}: {why}")
        if not (complete if args.command == "complete" else decided):
            sys.exit(NO)
    elif args.command == "finish":
        finish(chosen, out, root, scope, release=args.release)
    elif args.command == "interrupted":
        if args.part is None or args.attempt is None or not args.technical:
            sys.exit("interrupted needs --part, --attempt N and --technical REASON")
        interrupted(args, chosen, out, scope)
    elif args.command == "restart-case":
        restart_case(args, chosen, out, root, scope, name)
    elif args.command == "supersede":
        supersede_case(args, chosen, planned, out, root, scope, name)
    elif args.command == "clear-stale":
        clear_stale(chosen, root)
    else:  # locate
        run_dir = settled(chosen.run, root)
        if run_dir is None:
            sys.exit(f"{chosen.run.label}: no settled launcher run (done or diverged)")
        print(run_dir)


if __name__ == "__main__":
    main()
