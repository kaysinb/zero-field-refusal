"""Study 3's numerical qualification on a Pod: the plan, a case's parts, the driver, the verdict.

    python scripts/qualify.py plan --campaign FILE --candidate N [--engineering]
        [--diagnostic-steps T1,T2,...]
    python scripts/qualify.py driver --campaign FILE --candidate N [--engineering]
        [--diagnostic-steps T1,T2,...] --commit SHA --out FILE [--dry-run]
    python scripts/qualify.py part --campaign FILE --candidate N [--engineering] --case NAME
        --part baseline|rest|control (--resume | --attempt N [--infrastructure REASON])
        --run-dir DIR --checkpoints DIR [--diagnostic-steps T1,T2,...] [--device D]
    python scripts/qualify.py complete|decided|clear-stale|locate|finish [--release]
        --campaign FILE --candidate N [--engineering] --case NAME
    python scripts/qualify.py interrupted --campaign FILE --candidate N [--engineering]
        --case NAME --part baseline|rest|control --attempt N --technical REASON
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
same-path control ≈ 11 min) and the disk need, and exits 1 on a campaign outside this.

**A case, in order** (one at a time, on one Pod; the driver):

1. ``complete`` exits 0 when the case is done (below): then nothing of it runs again; 3 when it
   is not. ``complete`` and ``decided`` say no with exit 3 alone: any other non-zero exit is a
   failure, and stops the driver.
2. ``clear-stale`` removes ``<dir>/<run_id>.pt`` (and its partial file) of every earlier
   attempt of the case's launcher run in the launch journal but its settled one.
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

**Attempts of a part** (``part``; "Failure routes": an infrastructure interruption is rerun under
a new attempt id, a technical or scientific record never). Before ``run_part`` the part publishes
``<record>.started``; an attempt that has it and no record was interrupted. ``--resume`` (the
driver's call) runs attempt 1 of a part without any attempt, does nothing for a part whose latest
attempt is recorded (whatever its kind), and exits 8 for one whose latest attempt was interrupted.
``--attempt N`` runs attempt 1 of a part without any attempt, or the attempt after an
interrupted one, which needs ``--infrastructure REASON``, the documented interruption, and its
record carries it (``infrastructure``: the reason and the earlier attempts); a part whose latest
attempt is recorded, whatever its kind, is refused. ``interrupted --technical REASON`` records an
interrupted attempt, unexplained, as a technical part record carrying the reason ("an
unexplained incomplete record is technical"); it refuses an attempt with a record. The part needs a
clean checkout of a commit, sets the numerics up (``runtime.configure_numerics``) before any
CUDA work and builds the environment from the run's campaign config; the survival retry and the
float64 fallback stay inside the part.

``verdict`` writes RUNS_DIR/qualification/c{N}/verdict.json (with ``--engineering``,
``c{N}/engineering/verdict.json`` with ``engineering_summary``): per rate of the campaign its
``rate_verdict`` under this checkout's GPU code and the tracked settings, with its cases' and its
same-path control's verdicts (states, kinds, reasons, the 13 comparisons, warnings with
``--e-eng``), every case record of the candidate on disk shown (one of no campaign rate under
``unassigned``), every attempt of every part (``attempts``: recorded or interrupted, with each
record's sha256, status and kind), and prints the states. A record under a later attempt (which
these commands never write) is not hidden by it: it is listed under ``superseded`` and counted
against its rate as a case of its own (a control's, as a warning). The launcher runs located now
(one settled directory per run) are the expected ones; when a rate's are not all located, none
is checked, with a warning.

``driver`` writes the bash driver for a Pod from the source checkout of SHA: it checks the
checkout and the campaign file's sha256, that nothing else uses the GPU, then runs each case in
order as above, and writes the verdict on every exit. Exits: 0 when every case is complete; 1
a precondition, the disk (``df``'s answer must be a number) or a ``finish`` that refuses a case
not complete; 8 an interrupted attempt; 9 a technical case; the launcher's own when it stops;
any other, the failed command's. The
driver never reruns a part or the launcher run: a rerun is a command of its own, with its reason.
"""

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
from typing import Any

from omegaconf import OmegaConf

from zero_field_refusal import code_identity, runtime
from zero_field_refusal import qualification as q
from zero_field_refusal.config import load_config
from zero_field_refusal.launch import Run, assess, launch_records, load_campaign
from zero_field_refusal.paths import CONFIG_ROOT, load_env, runs_root
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.tracking import CLEAN_CHECKOUT, git_state

SETTINGS = CONFIG_ROOT / "experiments" / "qualification.yaml"  # the tracked settings, by path
RUN = 'project-run "$SRC" python scripts/qualify.py'
LAUNCH = 'project-run "$SRC" python scripts/launch.py'
NO = 3  # ``complete`` and ``decided``: no (any other non-zero exit is a failure)
INTERRUPTED = 8  # ``part --resume``: the latest attempt was started and left no record
TECHNICAL_CASE = 9  # ``finish``: the case's verdict is technical; its checkpoints are kept
# The Pod's environment and the source checkouts (the driver's preamble)
POD_ENVIRONMENT = (". /etc/rp_environment", ". /root/zfr-env.sh")
SOURCES = "/workspace/code/zero-field-refusal"
GPU_USED_MIB = 1000  # the driver starts only below this much GPU memory in use
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
    ``interrupted`` (started, no record) or ``invalid`` (a record that is unreadable, or not
    that part's of that case). ``attempts`` maps every attempt to recorded or interrupted."""

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


def part_state(out: Path, case: q.QualificationCase, part: str) -> PartState:
    attempts = attempt_files((out / case.attempt_path(part, 1)).parent, part)
    if not attempts:
        return PartState("due", 1, None, attempts)
    latest = max(attempts)
    if attempts[latest] == "interrupted":
        return PartState("interrupted", latest, None, attempts)
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
    conflict): the one directory of the run's config whose outcome is settled, or None."""
    found = []
    for run_dir in sorted((root / run.experiment).glob("*")):
        meta = _read(run_dir / "meta.json")
        if meta is None or meta.get("config_hash") != run.config_hash:
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
    lines += [f"PROBLEM: {p}" for p in problems]
    return "\n".join(lines)


def part(args, planned: Planned, out: Path) -> None:
    case, name = planned.case, args.part
    state = part_state(out, case, name)
    reason = args.infrastructure
    if args.resume:
        if state.state == "recorded":
            print(f"{case.name} {name}: attempt {state.attempt} is recorded "
                  f"({state.record.get('status')}); nothing runs")  # fmt: skip
            return
        if state.state == "interrupted":
            print(f"{case.name} {name}: attempt {state.attempt} was started and left no record: "
                  f"an interrupted attempt, rerun only as attempt {state.attempt + 1} with "
                  "--infrastructure REASON", file=sys.stderr)  # fmt: skip
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
        if (attempt > 1) != bool(reason):
            sys.exit("an attempt above 1 reruns an interrupted one and needs --infrastructure "
                     "REASON; attempt 1 takes none")  # fmt: skip
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
        "infrastructure": reason, "host": socket.gethostname(),
        "started_at": datetime.now(UTC).isoformat(timespec="seconds")})  # fmt: skip
    record = q.run_part(settings, env, case, args.run_dir, out, part=name, attempt=attempt,
                        checkpoints=args.checkpoints,
                        diagnostic_steps=parse_steps(args.diagnostic_steps))  # fmt: skip
    if reason:  # the record carries its reason, before any later part reads it
        previous = [{"attempt": n, "state": s} for n, s in state.attempts.items() if n < attempt]
        record = {**record, "infrastructure": {"reason": reason, "previous": previous}}
        q.publish(path, record)
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
    built on (the latest baseline then); never over a record."""
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
                     "identity": q.launcher_identity(Path(run_dir)) if run_dir else None},
        "interrupted": {"reason": args.technical, "started": started},
    }  # fmt: skip
    if name == "rest":
        base = (out / case.attempt_path("baseline", 1)).parent
        found = max((n for n, s in attempt_files(base, "baseline").items() if s == "recorded"),
                    default=None)  # fmt: skip
        if found is not None:
            source = out / case.attempt_path("baseline", found)
            baseline = _read(source) or {}
            record["baseline_record"] = {"path": str(source),
                                         "sha256": q.record_sha256(baseline)}  # fmt: skip
            record["baseline_checkpoint_sha256"] = (baseline.get("checkpoint") or {}).get("sha256")
    record["provenance"] = q.qualification_provenance(load_settings())
    q.publish(path, record)
    print(json.dumps({key: record[key] for key in ("part", "attempt", "status", "kind",
                                                   "reasons")}))  # fmt: skip


def clear_stale(planned: Planned, root: Path) -> None:
    keep = settled(planned.run, root)
    stale = [i for i in journal_run_ids(planned.run, root) if keep is None or i != keep.name]
    print(json.dumps({"case": planned.case.name, "removed": remove_checkpoints(planned, stale)}))


def latest(folder: Path, part: str) -> dict | None:
    """The record of a part's latest attempt in a case's folder, or None (none, interrupted,
    or unreadable)."""
    attempts = attempt_files(folder, part)
    if not attempts or attempts[max(attempts)] != "recorded":
        return None
    return _read(folder / f"{part}-attempt-{max(attempts)}.json")


def attempt_listing(folder: Path) -> tuple[dict[str, list], list[tuple[str, int, dict | None]]]:
    """Every attempt of every part in a case's folder (number, state, path, the record's sha256,
    status and kind), and the superseded records: a record under a later attempt, which the
    commands here never write, is not hidden by it."""
    listing, superseded = {}, []
    for part in q.PARTS:
        attempts = attempt_files(folder, part)
        rows = []
        for n, state in attempts.items():
            path = folder / f"{part}-attempt-{n}.json"
            record = _read(path) if state == "recorded" else None
            rows.append({"attempt": n, "state": state, "path": str(path),
                         "sha256": q.record_sha256(record),
                         "status": (record or {}).get("status"),
                         "kind": (record or {}).get("kind")})  # fmt: skip
            if state == "recorded" and n < max(attempts):
                superseded.append((part, n, record))
        if rows:
            listing[part] = rows
    return listing, superseded


def verdict(args, settings, scope, planned: list[Planned], out: Path, root: Path) -> Path:
    candidate, stage = args.candidate, ROLES[args.engineering]
    base = out / f"c{candidate}" / stage
    folders = sorted({f.parent for f in base.glob("s*/*/*-attempt-*")})
    cases, records, attempts, superseded = [], [], {}, []
    warnings: list[str] = []
    for folder in folders:
        name = str(folder.relative_to(base))
        attempts[name], stale = attempt_listing(folder)
        baseline, rest = latest(folder, "baseline"), latest(folder, "rest")
        if baseline is not None or rest is not None:
            records += [r for r in (baseline, rest) if r is not None]
            cases.append(q.case_verdict(baseline, rest, scope, e_eng=args.e_eng))
        for part, n, record in stale:  # shown, and counted against its rate as a case
            superseded.append({"case": name, "part": part, "attempt": n,
                               "sha256": q.record_sha256(record)})  # fmt: skip
            if part == "control":
                warnings.append(f"{name}: control attempt {n} is superseded by a later attempt")
                continue
            pair = (record, None) if part == "baseline" else (None, record)
            cases.append({**q.case_verdict(*pair, scope, e_eng=args.e_eng),
                          "superseded": {"part": part, "attempt": n}})  # fmt: skip
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
                record = latest((out / holder.case.attempt_path("control", 1)).parent, "control")
                control = None if record is None else q.control_verdict(record, scope, candidate)
            runs = {p.case.seed: located(p.run, root) for p in planned
                    if (p.case.optimizer, p.case.lr) == (optimizer, lr)}  # fmt: skip
            launchers = None
            if all(d is not None for d in runs.values()):
                launchers = {seed: q.launcher_identity(d) for seed, d in runs.items()}
            else:
                warnings.append(f"{key}: launcher runs not located "
                                f"({sorted(s for s, d in runs.items() if d is None)}): the "
                                "records' launcher runs are not checked against them")  # fmt: skip
            rates[key] = q.rate_verdict(mine, scope=scope, candidate=candidate,
                                        optimizer=optimizer, lr=lr, code=code, policy=policy,
                                        control=control, launchers=launchers)  # fmt: skip
            states[key] = rates[key]["state"]
            warnings += [f"{key}: τ exceeded by {name}" for name in rates[key]["warnings"]]
        body.update(
            states=states,
            rates=rates,
            unassigned=[v["case"].get("name") for v in cases if id(v) not in assigned],
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
        *comment(f"Exits: 0 every case complete; 1 a precondition, the disk or a case finish "
                 f"refuses (not complete); {INTERRUPTED} an interrupted attempt; "
                 f"{TECHNICAL_CASE} a technical case; the launcher's own when it stops; another "
                 "a failed command."),
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
        help="part: the documented interruption an attempt above 1 reruns",
    )
    parser.add_argument("--technical", metavar="REASON",
                        help="interrupted: why the attempt is recorded technical")  # fmt: skip
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
    if args.command in ("part", "verdict", "interrupted") and problems:  # in scope only
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
    elif args.command == "clear-stale":
        clear_stale(chosen, root)
    else:  # locate
        run_dir = settled(chosen.run, root)
        if run_dir is None:
            sys.exit(f"{chosen.run.label}: no settled launcher run (done or diverged)")
        print(run_dir)


if __name__ == "__main__":
    main()
