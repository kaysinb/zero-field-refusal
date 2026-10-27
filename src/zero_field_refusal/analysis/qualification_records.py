"""Study 3's numerical qualification read from its records, each rate's state recomputed.

docs/study3-preregistration.md: "Numerical qualification" ("Qualification"), "Failure routes",
"Voiding and supersession" and amendment 2 (the same-path control). The records lie under
``RUNS_DIR/qualification/c<N>/``, in the formats ``scripts/qualify.py`` writes and reads; that
script, ``qualification`` and ``launch`` run on a GPU and are not imported: their formats are
read here, and the verdict rules are called where they live, in ``qualification_verdicts``
(``case_verdict``, ``control_verdict``, ``rate_verdict``, ``rest_from_trajectories``).

- ``read_qualification``: for each rate (arm, rate) of the candidate's numerical qualification,
  its cases (each part's attempts, the one that counts and the earlier ones), the records the
  case restarts and the engineering cycle void, with their reasons, the launcher runs issued,
  located, superseded and held (each hold with the records that hold it and their sha256), the
  same-path control and the rate's state, recomputed with ``rate_verdict`` from inputs
  assembled as ``qualify.py verdict`` assembles them; ``verdict.json`` read as a cache and
  compared with it; the record files it took as inputs (``inputs``); and the ``problems``.
- ``rate_of``: one rate of a reading, by arm (or optimizer) and rate.
- ``supersessions``: the candidate's engineering-cycle records as ``read_qualification`` lists
  them (old and fix GPU codes included), read alone: they do not depend on the expected code.
- ``Scope``, ``settings_record``, ``launcher_identity`` and ``resolved_verdict``: what
  ``qualification.QualificationScope``, ``qualification.settings_record``,
  ``qualification.launcher_identity`` and ``qualify.py``'s ``resolved_verdict`` give, from the
  same inputs.

**The inputs of a rate**, as the verdict assembles them:
- *the cases*: every case folder of ``c<N>/numerical/`` (``s<seed>/<optimizer>-<rate>/``), read
  with the case's restart records (``restart-<n>.json``, in the folder) and the candidate's
  engineering cycle (its supersession record, ``c<N>/supersede-1.json``, once it passes the checks
  below), which void the attempts they name, by their record file's sha256. One case verdict per
  folder, of the latest recorded baseline and rest attempts that neither names; a rest whose
  latest attempt was interrupted is read from its trajectory results (``rest_from_trajectories``),
  unless a restart or supersession names that attempt. Every other recorded attempt is a
  superseded record: void when named, with the restart's reason or the cycle's diagnosis and fix,
  and otherwise counted against its rate as a case of its own (a control as a superseded
  control);
- *the cases the engineering cycle kept on their launcher run's resolved outcome* (``kept_cases``
  with a ``resolution``: no part record classified that run, ``launcher.classified``; none names
  it, or each that does raised before its launcher check): that outcome as a case verdict, beside
  the verdicts of the case's own records, which stay;
- *the control* of an FA rate (the control arm): the latest control attempt of the rate's case on
  the candidate's first numerical seed, unless named;
- *the launcher runs* located now, one per seed of the rate (``located``: the one settled
  directory of the cell's config hash now that no ``superseded`` line of the journal names for
  that hash, a line written from a supersession record that voids nothing aside), as the records
  name them (``launcher_identity``). The cell's config hash now is the
  campaign's, which the verdict is given: its label's latest start's command (its config and
  overrides) composed under this checkout's configs, as ``launch.load_campaign`` composes a
  cell. It is that start's own hash unless a fix changed a hashed config key since; then it is
  the fix's, and a run of the earlier hash (kept or held, never attempted again) is not
  located, as the verdict has it;
- the expected GPU code (``expected_code``, from the fixation record) and the settings as a record
  holds them (``settings_record``).

**The engineering cycle** ("Failure routes": one engineering cycle per candidate; "Voiding and
supersession"). A supersession record voids something (attempts and their records, launcher runs,
kept cases) or holds a run only once the one check has accepted it (``cycle_checks``, ``_Cycles``:
``qualify.py supersede`` mirrored, so that a cycle the script would refuse is a problem naming the
record, and one it accepts, followed by the registered steps after it, stays valid):
- *its name*: only ``supersede-<n>.json`` (``VOIDING_FILE``) is a supersession record; any other
  ``supersede-*.json`` is a problem;
- *the candidate's*: it lies in ``c<N>/``, and every candidate it names is N: its ``candidate``,
  its old record's (the folder it lies in and the case it holds), its kept cases', and the starts
  of the launcher runs it supersedes or holds (a record of ``c<N>/`` applies to candidate N's runs
  alone);
- *its only cycle*: ``qualify.py supersede`` writes ``c<N>/supersede-1.json`` and refuses a
  second, in either stage; with a second record for the candidate, under any number, neither
  voids anything;
- *from its technical record*: it names its diagnosis, fix and cases re-qualified, and its old
  record (path and sha256) is found under RUNS_DIR by the path's last components
  (``qualification/c<N>/<stage>/s<seed>/<case>/<part>-attempt-<n>.json``, the stage the case's
  own, ``numerical`` or, with ``supersede --engineering``, ``engineering``; whatever root the
  machine that wrote it had), resolves under no sealed folder, has the cited sha256, agrees with
  the case, stage, part and attempt the record names, is of a case an issued start of its stage
  names, and is what ``supersede`` requires: not void at the time of the cycle (no case restart
  whose ``restarted_at`` is strictly earlier than the record's ``superseded_at`` names it; a
  restart without a readable time counts as earlier; a restart after the cycle, which names
  every attempt of the case, the old record included, changes nothing), a completed technical
  failure (``qualify.py``'s ``failure_kind``, mirrored: ``_failure_kind``), and of a case the
  cycle does not keep whole: none its kept cases list, and none that holds a scientific failure
  under the old GPU code (the case's attempts of that code re-read, as ``scientific_attempts``
  reads them at the time of the cycle);
- *a new code*: its old GPU code (``old_gpu_code.sha256``) is its old record's
  (``provenance.gpu_code.sha256``), and its fix's (``fix_gpu_code.sha256``) is named and another
  (the same path is not simply rerun).
A record that fails any check voids and holds nothing: the failed attempts keep counting, its
launcher runs stay located and its kept cases add nothing. A ``superseded`` or ``held`` line that
cites a supersession record counts only through a record the check accepted, and only for a run
that record names under ``launchers`` or ``held``; any other (a record not there, in no
candidate's folder, not accepted, or not naming the run) is a problem and counts for nothing.
A record in a folder whose name is not exactly ``c<N>`` (``c01/``, say) belongs to no
candidate. Every reader of what a record voids or holds applies the same check
(``study3_records``: ``read_endpoints``, ``read_holds``), and every problem of it is one of the
run records' problems, so that it blocks at every step. The chain of GPU codes
(``study3_progress.code_chain``) reads the records alone (``supersessions``) and counts a second
record for a candidate as a break.

**The issued runs.** The campaign file the verdict was given is not read: the launch journal
stands for it. A start of a campaign whose header names study 3, the candidate and the step
``numerical`` or ``repair`` (a repaired rate is qualified as numerical runs) issued a case: its
config (its run's ``config.yaml``, or its start's command loaded again, bound by the start's
config hash as ``study3_records`` binds a start) gives the case's arm, rate and seed. The starts
of one label and experiment are the attempts of one launcher run whatever their config hash (a
fix may change a hashed config key: ``qualify.py``'s ``label_run_ids``), in attempt order: the
attempt number, then the start's time stamp, never the journal file that holds it
(``launch.attempt_key``). Every start is bound by its own config hash, one refused (its
``config.yaml`` of another config hash) is a problem whatever the others, and starts of one label
that bind different cases are a problem. Every case record (a part record, an interrupted rest's
trajectory results, a kept case's resolution) must name a numerical case of the candidate that an
issued start names: the run bound to an issued cell.

**Held runs** (``held``; ``study3_records.read_holds``): every hold of an issued label, whatever
the config hash it was written with (``launch.held_runs``), or of an issued run id, or named by
the candidate's supersession records; each with ``sources`` (``journal <campaign>`` for each
``held`` line, the supersession record's path) and ``records``, the same records as evidence
references with their sha256 (a journal line with its log's path and the line's number), and the
issued ``case`` it holds (arm, optimizer, rate, seed; None for a label no issued start binds). A
rate lists the holds of its labels, an issued run those of its label.

**The rates** are those the issued runs, the case records and ``verdict.json`` name, one rate per
optimizer and rate (relative tolerance 10⁻⁹); each is recomputed from the records, whoever names
it. A rate named by the cache alone has no case and no located run; one named by case records
alone is a rate of its own, and each of its records a problem (no issued start). The key of a
rate is ``<optimizer>-<rate:g>``, the verdict's.

**The cache** (``c<N>/verdict.json``, when it exists): its rate states, each compared with the
recomputation. A rate it does not name is listed under ``uncached``: each qualification campaign
of the candidate (the repair's included) rewrites the file with its own rates.

**Problems** (each blocks every decision that rests on the qualification): an unreadable record
(a part attempt, a trajectory result, a restart, supersession or release record, the journal, the
cache, or a cache without its rates); a malformed record, never an exception: one the verdicts
cannot read (a case, an interrupted rest's trajectory results, a kept case, a rate; where the
script would raise, the record is named and left out); a supersession record that fails a check
of the engineering cycle (above: it voids nothing); a start of the qualification refused or
that cannot be bound to its case, starts of one label bound to different cases, two issued runs
(labels) of one case, a cell whose config this checkout's configs cannot compose, a case record
no issued start names, and a record that may hold a run (a launch log, the candidate's
supersession record) that ``read_holds`` cannot read; and every disagreement with the cache: a
rate's state, the cache
written for another candidate or stage, under another GPU code or other settings than expected.
The states are always the recomputation's.

**Pure reading.** No file is written. Of a launcher run, ``meta.json``, ``summary.json`` and the
files of its identity (``config.yaml``, ``metrics.jsonl``, ``monitor.jsonl``, ``terminal.json``)
are read; nothing under a run's ``responses/`` or ``observations/`` is opened. Every file is
checked by ``study3_records``' one sealed-path rule (``under_sealed``) before it is opened, read
or hashed: a run's files with the run's folder as the base (``_sealed_records``: a run whose
folder or record files resolve under a sealed folder is read not at all, a problem, and no
launcher run is located there); the launch logs and the qualification's records (part attempts,
trajectory results, restart, supersession and release records, the verdict cache) with RUNS_DIR
as the base. One that resolves under a sealed folder is never opened: it is a problem named by its
path. This checkout's
``configs/`` are read to compose a cell's config. The decision records' folder in the
experiment's folder (``study3_records.DECISIONS``) is no run and is never walked. The GPU code
identity is not computed here: it is an argument. No GPU path is imported here;
``qualification_verdicts`` itself imports ``fa_control_diagnostic`` (and with it torch).

**Rulings** where the verdict's command line leaves an input open:
- *τ* (the warning, ``--e-eng``) is not recomputed: it is a prompt for diagnosis, never a state,
  and e_eng is not a record of the numerical stage. The rates' ``warnings`` list none.
- *An FA rate's control* is read from its first seed's case whether or not that case was issued:
  a qualification campaign of an FA rate must hold that case (``qualify.py``, "The cases").
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from ..config import config_hash, load_config
from ..qualification_constants import (
    ACCEPTED,
    FAILING,
    INCOMPLETE,
    PARTS,
    SCIENTIFIC,
    SHIFTS,
    TECHNICAL,
    QualificationError,
    record_sha256,
)
from . import qualification_verdicts as verdicts
from .study3_records import (
    ARMS,
    DECISIONS,
    RATE_TOLERANCE,
    _command_config,
    _sealed_records,
    _unbacked,
    plain_name,
    read_holds,
    sealed_record,
    under_sealed,
)
from .study3_records import _config as _start_config

STAGE = "numerical"  # the stage folder of the numerical qualification, c<N>/numerical/
STEPS = ("numerical", "repair")  # the campaign steps whose runs are qualification cases
ENGINEERING = "engineering"  # the engineering stage's folder and its campaign step
VERDICT = "verdict.json"
VOIDING_FILE = re.compile(r"(restart|supersede)-(\d+)\.json")
PART_FILE = re.compile(r"(baseline|rest|control)-attempt-(\d+)\.json")  # a part attempt record
RELEASED = "checkpoints-removed"  # ``finish``'s record of the checkpoints it removed
RELEASE_RECORD = "release record"  # what a release record is, among the records read
IDENTITY_FILES = ("config.yaml", "metrics.jsonl", "monitor.jsonl", "terminal.json")
SETTLED = ("completed", "diverged")  # a launcher run's settled statuses (``located``)


# ------------------------------------------------------------------ the scope and the policy


@dataclass(frozen=True)
class Scope:
    """The registered scope as ``qualification.QualificationScope.of`` reads it from
    configs/experiments/qualification.yaml (that module runs on a GPU and is not imported): the
    same fields, checks and methods, which the verdicts call."""

    spec: str
    bound_fraction: float
    k_stars: tuple[tuple[int, float], ...]  # (candidate, K*)
    norm: float
    gaussian: tuple[tuple[str, int], ...]  # (("s3", 3), ("s4", 4))
    survival_share: float
    rounding_scale: float
    warning_floor: float
    warning_factor: float
    engineering_seeds: tuple[int, ...]
    numerical_seeds: tuple[tuple[int, tuple[int, ...]], ...]  # (candidate, seeds)
    control_arm: str
    control_steps: int
    control_bound_fraction: float

    @classmethod
    def of(cls, settings: Mapping[str, Any]) -> Scope:
        plain = settings_record(settings)
        if plain.get("experiment") != "qualification":
            raise QualificationError(f"Not the qualification's settings: "
                                     f"{plain.get('experiment')!r}")  # fmt: skip
        try:
            seeds = plain["seeds"]
            scope = cls(
                spec=str(plain["spec"]), bound_fraction=float(plain["bound_fraction"]),
                k_stars=tuple(sorted((int(c), float(k)) for c, k in plain["k_star"].items())),
                norm=float(plain["norm"]),
                gaussian=tuple((str(k), int(v)) for k, v in plain["gaussian"].items()),
                survival_share=float(plain["survival_share"]),
                rounding_scale=float(plain["rounding_scale"]),
                warning_floor=float(plain["warning"]["floor"]),
                warning_factor=float(plain["warning"]["factor"]),
                engineering_seeds=tuple(int(s) for s in seeds["engineering"]),
                numerical_seeds=tuple(sorted((int(c), tuple(int(s) for s in v))
                                             for c, v in seeds["numerical"].items())),
                control_arm=str(plain["control"]["arm"]),
                control_steps=int(plain["control"]["steps"]),
                control_bound_fraction=float(plain["control"]["bound_fraction"]),
            )  # fmt: skip
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise QualificationError(f"Not a qualification scope: {exc!r}") from exc
        every = [*scope.engineering_seeds, *(s for _, v in scope.numerical_seeds for s in v)]
        if (
            [k for k, _ in scope.gaussian] != list(SHIFTS)[2:]
            or {c for c, _ in scope.k_stars} != {c for c, _ in scope.numerical_seeds}
            or len(set(every)) != len(every)
            or not all(_number(k) and k > 0 for _, k in scope.k_stars)
            or not (scope.norm > 0 and 0 < scope.survival_share <= 1 and scope.rounding_scale > 1)
        ):
            raise QualificationError(f"Inconsistent qualification scope: {scope}")
        return scope

    def k_star(self, candidate: int) -> float:
        try:
            return dict(self.k_stars)[candidate]
        except KeyError:
            raise QualificationError(f"No candidate {candidate!r}") from None

    def bound(self, candidate: int) -> float:
        """0.01·K*: every comparison's bound."""
        return self.bound_fraction * self.k_star(candidate)

    def warning(self, e_eng: float, candidate: int) -> float:
        """τ = K*·max(0.001, 3·e_eng): a prompt for diagnosis, never a bound."""
        return self.k_star(candidate) * max(self.warning_floor, self.warning_factor * e_eng)

    def seeds(self, candidate: int) -> tuple[int, ...]:
        """The candidate's numerical seeds; the first holds the same-path controls."""
        try:
            return dict(self.numerical_seeds)[candidate]
        except KeyError:
            raise QualificationError(f"No candidate {candidate!r}") from None

    def control_bound(self, candidate: int) -> float:
        """Amendment 2: the same-path control's endpoint K_benign ≤ 0.01·K*."""
        return self.control_bound_fraction * self.k_star(candidate)


def settings_record(settings: Mapping[str, Any]) -> dict[str, Any]:
    """The settings as a part record holds them (``qualification.settings_record``: JSON, the
    candidates' keys as strings), from the loaded config or a plain mapping."""
    return json.loads(json.dumps(_container(settings)))


def _container(value: Any) -> Any:
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return {k: _container(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_container(v) for v in value]
    return value


# ------------------------------------------------------------------ records on disk


def _object(path: Path, base: Path) -> dict[str, Any] | None:
    """A JSON object on disk, or None (missing, unreadable, not an object, or resolving under a
    sealed folder below ``base``: never opened)."""
    if under_sealed(path, base):
        return None
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _file_sha256(path: Path, base: Path) -> str | None:
    """The sha256 of a record file's bytes (``record_sha256`` of a published record); None when it
    cannot be read or resolves under a sealed folder below ``base`` (never opened)."""
    if under_sealed(path, base):
        return None
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _number(value: Any) -> bool:
    """A finite real number: not None, not a bool, not ±∞ or NaN."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _given(value: Any) -> bool:
    """A non-empty string."""
    return isinstance(value, str) and bool(value)


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


def launcher_identity(run_dir: Path) -> dict[str, Any] | None:
    """How a part record names its launcher run (``qualification.launcher_identity``): its place
    under the runs root, run id, config hash and the sha256 of its config, metrics, monitor and
    terminal records, with its endpoint-checkpoint record; None when it cannot be read, and when
    its folder or a record file of it resolves under a sealed folder (none is opened)."""
    if _sealed_records(run_dir) is not None:
        return None
    try:
        summary = _object(run_dir / "summary.json", run_dir) or {}
        digest = hashlib.sha256()
        for name in IDENTITY_FILES:
            path = run_dir / name
            digest.update(f"{name}\0".encode())
            digest.update(path.read_bytes() if path.exists() else b"\0missing\0")
        meta = run_dir / "meta.json"
        config_hash = json.loads(meta.read_text()).get("config_hash") if meta.exists() else None
    except (OSError, ValueError, AttributeError):
        return None
    return {"run": f"{run_dir.parent.name}/{run_dir.name}", "run_id": run_dir.name,
            "config_hash": config_hash, "sha256": digest.hexdigest(),
            "checkpoint": summary.get("endpoint_checkpoint")}  # fmt: skip


def resolved_verdict(resolution: Mapping[str, Any]) -> dict[str, Any]:
    """A case verdict of a launcher run's outcome the engineering cycle resolved, for a case
    with no part record (``qualify.py``'s ``resolved_verdict``): ``failing`` and scientific for a
    conforming declared stop, an ``incomplete`` case of no kind for an unresolved one. Its GPU
    code is the run's, so ``rate_verdict`` never counts it as a qualified case."""
    scientific = resolution.get("outcome") == SCIENTIFIC
    return {"case": resolution.get("case_record"),
            "state": FAILING if scientific else INCOMPLETE,
            "kind": SCIENTIFIC if scientific else None,
            "reasons": [("" if scientific else "unresolved: ") + r
                        for r in resolution.get("reasons") or []],
            "complete": False, "comparisons": {}, "bound": None, "tau": None, "warning": None,
            "reference_mode": None, "fallback": False,
            "launcher": resolution.get("identity"), "gpu_code": resolution.get("gpu_code"),
            "settings": None, "records": {}, "resolution": dict(resolution),
            "record_sha256": hashlib.sha256(json.dumps(resolution, sort_keys=True).encode())
            .hexdigest()}  # fmt: skip


def _void_reason(path: Path, record: Mapping[str, Any] | None) -> str:
    """Why the attempts a voiding record names are void: a restart's reason, or the
    supersession's diagnosis and fix."""
    record = record or {}
    if path.name.startswith("supersede-"):
        return (f"superseded by an engineering cycle ({path.name}): {record.get('diagnosis')}; "
                f"fix: {record.get('fix')}")  # fmt: skip
    return str(record.get("reason"))


# ------------------------------------------------------------------ the launch journal


@dataclass
class _Issued:
    """A launcher run the candidate's qualification campaigns issued: one label and experiment,
    its attempts (their run ids and starts, in attempt order, under any config hash), the case
    its config binds (None: unbound) and the cell's config hash now (``located``'s; None until
    composed)."""

    label: str
    experiment: str
    run_ids: list[str] = field(default_factory=list)
    campaigns: list[str] = field(default_factory=list)
    starts: list[dict[str, Any]] = field(default_factory=list)
    config_hash: str | None = None
    optimizer: str | None = None
    lr: float | None = None
    seed: int | None = None
    ambiguous: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return self.label, self.experiment

    @property
    def config_hashes(self) -> list[str]:
        """The config hashes of its starts, in attempt order, each once."""
        return list(dict.fromkeys(start["config_hash"] for start in self.starts))

    def record(self) -> dict[str, Any]:
        return {"label": self.label, "experiment": self.experiment,
                "config_hash": self.config_hash, "config_hashes": self.config_hashes,
                "campaigns": sorted(set(self.campaigns)), "run_ids": list(self.run_ids),
                "optimizer": self.optimizer, "lr": self.lr, "seed": self.seed}  # fmt: skip


def _journal(root: Path, problems: list[str]) -> list[tuple[str, dict[str, Any]]]:
    """Every record of every launch log (UTF-8, as ``study3_records`` reads it), with its
    campaign; a line cut short by a crash is skipped (the launcher's rule). A log that resolves
    under a sealed folder is a problem, never opened."""
    records = []
    for log in sorted((root / "launch").glob("*.jsonl")):
        if under_sealed(log, root):
            problems.append(f"{log}: {sealed_record('a launch log', log, root)}")
            continue
        try:
            lines = log.read_text(encoding="utf-8").splitlines()
        except (OSError, ValueError):
            problems.append(f"{log}: an unreadable launch journal")
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                records.append((log.stem, record))
    return records


def _issues_case(record: Mapping[str, Any], candidate: int, steps: Sequence[str] = STEPS) -> bool:
    """Whether a journal record is a start of the candidate's qualification (module docstring,
    "The issued runs"), of one of ``steps`` (the engineering stage's: ``(ENGINEERING,)``)."""
    head = record.get("header")
    return (record.get("event") == "start" and isinstance(head, Mapping) and head.get("study") == 3
            and type(head.get("candidate")) is int and head.get("candidate") == candidate
            and head.get("step") in steps)  # fmt: skip


def _attempt_key(record: Mapping[str, Any]) -> tuple[int, str]:
    """How a start sorts among its label's attempts (``launch.attempt_key``): by its attempt
    number (a start without one first), then by its time stamp, never by the journal file that
    holds it."""
    attempt = record.get("attempt")
    number = attempt if isinstance(attempt, int) and not isinstance(attempt, bool) else -1
    return number, str(record.get("time") or "")


def _named(run: _Issued) -> str:
    """A run as a problem names it: its label and its starts' config hashes."""
    return f"{run.label} ({', '.join(run.config_hashes)})"


def _issued(
    root: Path, journal: Sequence[tuple[str, Mapping[str, Any]]], candidate: int,
    problems: list[str], steps: Sequence[str] = STEPS,
) -> list[_Issued]:  # fmt: skip
    """The launcher runs the candidate's qualification campaigns issued, one per label whatever
    its starts' config hashes, its attempts in attempt order, each bound to its case, with the
    cell's config hash now (module docstring, "The issued runs"); an unbound one, starts of one
    label bound to different cases, two runs of one case and a cell that does not compose are
    problems."""
    runs: dict[tuple[str, str], _Issued] = {}
    mine = [(campaign, record) for campaign, record in journal
            if _issues_case(record, candidate, steps)]  # fmt: skip
    # attempt order; a stable sort, so equal keys keep the journal's order, as ``label_run_ids``
    for campaign, record in sorted(mine, key=lambda item: _attempt_key(item[1])):
        run_id, label = record.get("run_id"), record.get("run")
        experiment, chash = record.get("experiment"), record.get("config_hash")
        if not (plain_name(run_id) and plain_name(experiment) and isinstance(label, str)
                and label and isinstance(chash, str) and chash):  # fmt: skip
            problems.append(f"{campaign}: a start of candidate {candidate}'s qualification "
                            f"without a run id, label, experiment or config hash: "
                            f"{run_id!r}, {label!r}, {experiment!r}, {chash!r}")  # fmt: skip
            continue
        run = runs.setdefault((label, experiment), _Issued(label, experiment))
        run.run_ids.append(run_id)
        run.campaigns.append(campaign)
        run.starts.append(dict(record))
    for run in runs.values():
        cases: dict[str, tuple[tuple[Any, ...], list[str]]] = {}  # by repr: a value may be a list
        why = []
        for start in run.starts:  # every start, by its own config hash
            found, unbound, refusal = _start_config(root / run.experiment / start["run_id"],
                                                    start, start["config_hash"], root)  # fmt: skip
            if refusal is not None:
                problems.append(f"{start['run_id']} ({run.label}): a start of candidate "
                                f"{candidate}'s qualification refused: {refusal}")  # fmt: skip
            elif found is None:
                why.append(f"{start['run_id']}: {unbound}")
            else:
                case = tuple(OmegaConf.select(found, key, default=None)
                             for key in ("optimizer.name", "optimizer.lr", "seed"))  # fmt: skip
                cases.setdefault(repr(case), (case, []))[1].append(start["run_id"])
        if not cases:
            if why:
                problems.append(f"{_named(run)}: an issued run of candidate {candidate}'s "
                                f"qualification whose case cannot be bound: "
                                f"{'; '.join(why)}")  # fmt: skip
            continue
        if len(cases) > 1:
            listed = "; ".join(f"{ids} {key}" for key, (_, ids) in cases.items())
            problems.append(f"{_named(run)}: the starts of one label bind different cases: "
                            f"{listed}")  # fmt: skip
            continue
        (optimizer, lr, seed), _ = next(iter(cases.values()))
        if not (isinstance(optimizer, str) and _number(lr) and type(seed) is int):
            problems.append(f"{_named(run)}: its config names no arm, rate and seed "
                            f"({optimizer!r}, {lr!r}, {seed!r})")  # fmt: skip
            continue
        run.optimizer, run.lr, run.seed = optimizer, float(lr), seed
        try:  # the campaign's cell now: its latest start's command under this checkout
            run.config_hash = config_hash(_command_config(run.starts[-1].get("command")))
        except Exception as exc:  # nothing to locate under: the verdict could not run either
            problems.append(f"{_named(run)}: its latest start's command does not compose under "
                            f"this checkout's configs ({type(exc).__name__}: {exc}): its launcher "
                            "run cannot be located")  # fmt: skip
    bound = [r for r in runs.values() if r.optimizer is not None]
    for run in bound:
        same = [o for o in bound if (o.optimizer, o.seed) == (run.optimizer, run.seed)
                and math.isclose(o.lr, run.lr, rel_tol=RATE_TOLERANCE)]  # fmt: skip
        if len(same) > 1:
            run.ambiguous = True
            names = ", ".join(sorted(_named(o) for o in same))
            problems.append(f"two issued runs are case s{run.seed}/{run.optimizer}-{run.lr:g} "
                            f"of candidate {candidate}: {names}")  # fmt: skip
    return list(runs.values())


def _named_runs(
    journal: Sequence[tuple[str, Mapping[str, Any]]], event: str
) -> list[dict[str, Any]]:
    """The journal's ``superseded`` or ``held`` lines (``launch.supersede``, ``launch.hold``)."""
    return [{"campaign": campaign, **record} for campaign, record in journal
            if record.get("event") == event and isinstance(record.get("run_id"), str)]  # fmt: skip


# ------------------------------------------------------------------ the engineering cycle


def _when(value: Any) -> datetime | None:
    """An ISO time stamp with its zone, or None."""
    try:
        found = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None
    return found if found is not None and found.tzinfo is not None else None


def _code(record: Mapping[str, Any] | None, key: str = "provenance") -> Any:
    """The GPU code's sha256 a record names: a part record's provenance, or a supersession
    record's ``old_gpu_code`` or ``fix_gpu_code``."""
    found = (record or {}).get(key)
    if key == "provenance":
        found = found.get("gpu_code") if isinstance(found, Mapping) else None
    return found.get("sha256") if isinstance(found, Mapping) else None


class _Cycles:
    """The one check of a candidate's supersession records (module docstring, "The engineering
    cycle"): ``qualify.py supersede`` mirrored, each record accepted or rejected with the
    problems that name why. Nothing is opened under a sealed folder."""

    def __init__(self, root: Path, candidate: int, scope: Scope,
                 journal: Sequence[tuple[str, Mapping[str, Any]]],
                 issued: Sequence[tuple[str, float, int]] | None = None) -> None:  # fmt: skip
        self.root, self.candidate, self.scope, self.journal = root, candidate, scope, journal
        self.folder = root / "qualification" / f"c{candidate}"
        self.cases: dict[str, list[tuple[Any, float, int]] | None] = dict.fromkeys(
            (STAGE, ENGINEERING)
        )
        if issued is not None:
            self.cases[STAGE] = list(issued)
        self.starts: dict[str, Any] = {}  # each start's run id -> the candidate its header names
        for _, record in journal:
            head = record.get("header")
            if record.get("event") == "start" and isinstance(record.get("run_id"), str):
                self.starts.setdefault(record["run_id"], head.get("candidate")
                                       if isinstance(head, Mapping) and head.get("study") == 3
                                       else None)  # fmt: skip

    def issued(self, stage: str) -> list[tuple[Any, float, int]]:
        """The (optimizer, rate, seed) the stage's issued starts name (``_issued``: the numerical
        and repair campaigns, or the engineering campaign)."""
        if self.cases[stage] is None:
            steps = STEPS if stage == STAGE else (ENGINEERING,)
            runs = _issued(self.root, self.journal, self.candidate, [], steps)
            self.cases[stage] = [(r.optimizer, r.lr, r.seed) for r in runs
                                 if r.optimizer is not None]  # fmt: skip
        return self.cases[stage]  # type: ignore[return-value]

    def check(self) -> dict[Path, dict[str, Any]]:
        """Each ``supersede-*.json`` of the candidate's folder: ``record`` (None when it cannot
        be read: its reader names that), ``accepted`` and ``problems``. Only
        ``supersede-<n>.json`` is a supersession record, and one of two or more voids nothing."""
        files = sorted(self.folder.glob("supersede-*.json"))
        named = [p for p in files if VOIDING_FILE.fullmatch(p.name)]
        out: dict[Path, dict[str, Any]] = {}
        for path in files:
            found: list[str] = []
            record = None
            if path not in named:
                found.append("not a supersession record's name (supersede-<n>.json, as qualify.py "
                             "names it): it voids nothing")  # fmt: skip
            else:
                record = None if under_sealed(path, self.root) else _object(path, self.root)
                if len(named) > 1:
                    found.append(f"one of {len(named)} supersession records of candidate "
                                 f"{self.candidate} ({', '.join(p.name for p in named)}): one "
                                 "engineering cycle per candidate, in both stages, and no record "
                                 "of a second voids anything")  # fmt: skip
                if record is not None:
                    found += self._record(record)
            out[path] = {"candidate": self.candidate, "record": record,
                         "readable": path not in named or record is not None,
                         "accepted": record is not None and not found,
                         **_named_entries(record),
                         "problems": [f"{path}: {why}" for why in found]}  # fmt: skip
        return out

    def _record(self, record: Mapping[str, Any]) -> list[str]:
        """Why a supersession record voids nothing, or none: it names the old record (its path
        and sha256), the diagnosis, the fix and the cases re-qualified ("Voiding and
        supersession"); every candidate it names is this one (its ``candidate``, its kept cases',
        and the starts of the launcher runs it supersedes or holds); and its old record is the
        technical failure ``qualify.py supersede`` starts a cycle from (``_old_record``)."""
        found = []
        old = record.get("old_record")
        requalified = record.get("cases_requalified")
        given = {"diagnosis": _given(record.get("diagnosis")), "fix": _given(record.get("fix")),
                 "old_record": isinstance(old, Mapping) and _given(old.get("path"))
                 and _given(old.get("sha256")),
                 "cases_requalified": isinstance(requalified, list)}  # fmt: skip
        missing = [key for key, ok in given.items() if not ok]
        if missing:
            found.append(f"a supersession record without its {', '.join(missing)}")
        named = record.get("candidate")
        if type(named) is not int or named != self.candidate:
            found.append(f"it names candidate {named!r}, not {self.candidate}: a "
                         f"supersession record in c{self.candidate}/ is candidate "
                         f"{self.candidate}'s")  # fmt: skip
        kept_cases = record.get("kept_cases")
        for kept in kept_cases if isinstance(kept_cases, list) else []:
            resolution = kept.get("resolution") if isinstance(kept, Mapping) else None
            case = resolution.get("case_record") if isinstance(resolution, Mapping) else None
            if isinstance(case, Mapping) and case.get("candidate") != self.candidate:
                found.append(f"a kept case of candidate {case.get('candidate')!r}, not "
                             f"{self.candidate}")  # fmt: skip
        for key in ("launchers", "held"):
            entries = record.get(key)
            for entry in entries if isinstance(entries, list) else []:
                run_id = entry.get("run_id") if isinstance(entry, Mapping) else None
                if isinstance(run_id, str) and self.starts.get(run_id) != self.candidate:
                    found.append(f"it names the launcher run {run_id} ({key}), which no start of "
                                 f"candidate {self.candidate}'s study 3 issued: a record in "
                                 f"c{self.candidate}/ applies to candidate {self.candidate}'s "
                                 "runs alone")  # fmt: skip
        if given["old_record"]:
            found += self._old_record(old, record)
        return found

    def _old_record(self, old: Mapping[str, Any], record: Mapping[str, Any]) -> list[str]:
        """Why the old record a supersession cites is not the technical failure that started the
        candidate's cycle, or none, as ``qualify.py supersede`` requires it: a part attempt record
        of a case of the candidate in its own stage folder (``qualification/c<N>/numerical/`` or,
        with ``--engineering``, ``qualification/c<N>/engineering/``, then ``s<seed>/<case>/``,
        found under RUNS_DIR whatever root the machine that wrote the cycle had), resolving under
        no sealed folder, of the cited sha256, the case, stage, part and attempt the record names,
        of an issued case of its stage; the record's old GPU code its old record's and its fix's
        another; not void at the time of the cycle (no case restart earlier than the record names
        it); a completed technical failure (``_failure_kind``); and of a case the cycle does not
        keep whole: none its kept cases list, and none that holds a scientific failure under the
        old GPU code (``_scientific``)."""
        cited = str(old["path"])
        parts = PurePosixPath(cited.replace("\\", "/")).parts[-6:]
        if (len(parts) < 6 or parts[0] != "qualification"
                or not re.fullmatch(r"c\d+", parts[1])):  # fmt: skip
            form = "qualification/c<N>/<stage>/s<seed>/<case>/<part>-attempt-<n>.json"
            return [f"its old record {cited} is no record of the qualification ({form})"]
        if parts[1] != f"c{self.candidate}":
            return [f"its old record {cited} lies under qualification/{parts[1]}/, candidate "
                    f"{parts[1][1:]}'s, not candidate {self.candidate}'s"]  # fmt: skip
        stage, match = parts[2], PART_FILE.fullmatch(parts[5])
        if stage not in (STAGE, ENGINEERING) or match is None:
            return [f"its old record {cited} is no part attempt record of candidate "
                    f"{self.candidate}'s {STAGE} or {ENGINEERING} stage"]  # fmt: skip
        name, (part, attempt) = f"{parts[3]}/{parts[4]}", (match[1], int(match[2]))
        path = self.folder / stage / parts[3] / parts[4] / parts[5]
        if under_sealed(path, self.root):
            return [f"its old record {path}: {sealed_record('a record', path, self.root)}"]
        found = [f"its old record names {key} {old.get(key)!r}, not its file's {value!r}"
                 for key, value in (("case", name), ("stage", stage), ("part", part),
                                    ("attempt", attempt))
                 if old.get(key, value) != value]  # fmt: skip
        sha = _file_sha256(path, self.root)
        if sha is None:
            return [*found, f"its old record {path} cannot be read"]
        if sha != old["sha256"]:
            return [*found, f"its old record {path} has sha256 {sha} now, not the cited "
                    f"{old['sha256']}"]  # fmt: skip
        data = _object(path, self.root)
        case = data.get("case") if isinstance(data, Mapping) else None
        if not isinstance(case, Mapping) or case.get("candidate") != self.candidate:
            named = case.get("candidate") if isinstance(case, Mapping) else None
            return [*found, f"its old record {path} is a record of candidate {named!r}, not "
                    f"{self.candidate}"]  # fmt: skip
        optimizer, lr, seed = case.get("optimizer"), case.get("lr"), case.get("seed")
        if case.get("engineering") is not (stage == ENGINEERING) or not any(
                o == optimizer and s == seed and _number(lr)
                and math.isclose(r, lr, rel_tol=RATE_TOLERANCE)
                for o, r, s in self.issued(stage)):  # fmt: skip
            found.append(f"its old record {path} is of case s{seed}/{optimizer}-{lr!r}, which "
                         f"no issued start of candidate {self.candidate}'s {stage} stage "
                         "names")  # fmt: skip
        old_code, fix_code, cited_code = _code(record, "old_gpu_code"), _code(
            record, "fix_gpu_code"), _code(data)  # fmt: skip
        if not (isinstance(old_code, str) and old_code) or old_code != cited_code:
            found.append(f"its old GPU code {old_code!r} is not its old record's {cited_code!r}: "
                         "the fix's GPU code is checked against the superseded record's, and "
                         "the same path is not simply rerun")  # fmt: skip
        elif not (isinstance(fix_code, str) and fix_code) or fix_code == old_code:
            found.append(f"the fix's GPU code is the superseded record's ({fix_code!r}, old "
                         f"{old_code!r}): the same path is not simply rerun")  # fmt: skip
        named_before, listed_before = self._restarted_before(path.parent, record)
        if sha in named_before:
            found.append(f"its old record {path} is void already, by a case restart earlier than "
                         "the cycle")  # fmt: skip
        try:
            kind = self._failure_kind(part, data, path.parent)
        except Exception as exc:  # a record the verdicts cannot read is never superseded
            kind = f"unreadable ({type(exc).__name__})"
        if kind != TECHNICAL:
            found.append(f"its old record {path} is not a completed technical failure "
                         f"({kind or 'it passes so far'}): an engineering cycle supersedes only "
                         "one")  # fmt: skip
        kept_cases = record.get("kept_cases")
        if any(isinstance(k, Mapping) and k.get("case") == name and k.get("stage") == stage
               for k in (kept_cases if isinstance(kept_cases, list) else [])):  # fmt: skip
            found.append(f"its old record's case {name} is one the cycle keeps whole")
        science = self._scientific(path.parent, old_code, named_before, listed_before)
        if science:
            found.append(f"its old record's case {name} holds a scientific failure under the old "
                         f"GPU code ({', '.join(science)}), which keeps counting: the cycle keeps "
                         "such a case whole and cannot start from one of its records")  # fmt: skip
        return found

    def _restarted_before(
        self, folder: Path, record: Mapping[str, Any]
    ) -> tuple[set[str], set[tuple[Any, Any]]]:
        """The attempts the case's restart records named at the time of the cycle (``supersede``
        checks "void already" then): those of every restart whose ``restarted_at`` is strictly
        earlier than the record's ``superseded_at``, a restart without a readable time counting
        as earlier (and every restart, when the record's time cannot be read). Their records'
        sha256 and their (part, attempt). An unreadable restart record is named by its reader."""
        cycle = _when(record.get("superseded_at"))
        named: set[str] = set()
        listed: set[tuple[Any, Any]] = set()
        for path in sorted(folder.glob("restart-*.json")):
            restart = _object(path, self.root) if VOIDING_FILE.fullmatch(path.name) else None
            if restart is None:
                continue
            when = _when(restart.get("restarted_at"))
            if cycle is not None and when is not None and when >= cycle:
                continue
            entries = restart.get("attempts")
            for entry in entries if isinstance(entries, list) else []:
                if isinstance(entry, Mapping):
                    listed.add((entry.get("part"), entry.get("attempt")))
                    if isinstance(entry.get("sha256"), str):
                        named.add(entry["sha256"])
        return named, listed

    def _scientific(self, folder: Path, old_code: Any, named: Collection[str],
                    listed: Collection[tuple[Any, Any]]) -> list[str]:  # fmt: skip
        """The case's attempts that are scientific failures under the old GPU code, as
        ``qualify.py``'s ``scientific_attempts`` finds them at the time of the cycle: every
        recorded attempt, and every interrupted rest's trajectory results, of that code, that no
        earlier case restart names (``_restarted_before``)."""
        found = []
        for part in PARTS:
            for n, state in attempt_files(folder, part).items():
                file = folder / f"{part}-attempt-{n}.json"
                if state == "recorded":
                    if _file_sha256(file, self.root) in named:
                        continue
                    data = _object(file, self.root)
                elif part == "rest" and (part, n) not in listed:
                    results = [
                        _object(f, self.root)
                        for f in sorted(folder.glob(f"rest-attempt-{n}.trajectory-*.json"))
                    ]
                    try:
                        data = verdicts.rest_from_trajectories([r for r in results if r])
                    except Exception:  # results the verdicts cannot read decide nothing here
                        data = None
                else:
                    continue
                if data is None or _code(data) != old_code:
                    continue
                try:
                    kind = self._failure_kind(part, data, folder)
                except Exception:  # a record the verdicts cannot read is no scientific failure
                    continue
                if kind == SCIENTIFIC:
                    found.append(f"{part} attempt {n}")
        return found

    def _failure_kind(self, part: str, record: Mapping[str, Any], folder: Path) -> str | None:
        """Whether a part's record is a completed failure, and of which kind, as
        ``qualify.py``'s ``failure_kind`` reads it: a baseline its verdict alone fails
        (scientific: a declared instability rule; technical: any reason but the rest still to
        come); a rest failing with the baseline record it names (of the case's recorded
        baselines, by sha256), or technical when it names none recorded; a control not accepted.
        None for a record that passes so far."""
        if part == "baseline":
            verdict = verdicts.case_verdict(record, None, self.scope)
            if verdict["kind"] == SCIENTIFIC:
                return SCIENTIFIC
            reasons = [r for r in verdict["reasons"] if r != verdicts.REST_MISSING]
            return TECHNICAL if reasons else None
        if part == "rest":
            named = (record.get("baseline_record") or {}).get("sha256")
            bound = None
            for n, state in attempt_files(folder, "baseline").items():
                file = folder / f"baseline-attempt-{n}.json"
                if state == "recorded" and _file_sha256(file, self.root) == named:
                    bound = _object(file, self.root)
            if bound is None:
                return SCIENTIFIC if record.get("kind") == SCIENTIFIC else TECHNICAL
            return verdicts.case_verdict(bound, record, self.scope)["kind"]
        verdict = verdicts.control_verdict(record, self.scope, self.candidate)
        return None if verdict["state"] == ACCEPTED else verdict["kind"]


def _named_entries(record: Mapping[str, Any] | None) -> dict[str, list[str]]:
    """The run ids a supersession record names under ``launchers`` and ``held``: the runs a
    journal line written from it may supersede or hold (``study3_records._unbacked``)."""
    out: dict[str, list[str]] = {}
    for key in ("launchers", "held"):
        entries = (record or {}).get(key)
        out[key] = [e["run_id"] for e in (entries if isinstance(entries, list) else [])
                    if isinstance(e, Mapping) and isinstance(e.get("run_id"), str)]  # fmt: skip
    return out


def _checks(
    root: Path, scope: Scope, journal: Sequence[tuple[str, Mapping[str, Any]]],
    issued: Mapping[int, Sequence[tuple[str, float, int]]] | None = None,
) -> dict[Path, dict[str, Any]]:  # fmt: skip
    """The one check (``_Cycles``) of every ``supersede-*.json`` under ``qualification/c*/``,
    by its path; ``issued``: a candidate's issued numerical cases, when read already. One in a
    folder of no candidate of the scope voids nothing."""
    out: dict[Path, dict[str, Any]] = {}
    folders = sorted({p.parent for p in (root / "qualification").glob("c*/supersede-*.json")})
    for folder in folders:
        number = re.fullmatch(r"c([1-9]\d*)", folder.name)  # exactly c<N>: c01 is no folder
        candidate = int(number[1]) if number else None
        if candidate is None or candidate not in dict(scope.numerical_seeds):
            for path in sorted(folder.glob("supersede-*.json")):
                out[path] = {"candidate": candidate, "record": None, "readable": True,
                             "accepted": False, "launchers": [], "held": [],
                             "problems": [f"{path}: it lies in {folder.name}/, the folder of no "
                                          "candidate of the qualification: it voids "
                                          "nothing"]}  # fmt: skip
            continue
        known = (issued or {}).get(candidate)
        out |= _Cycles(root, candidate, scope, journal, known).check()
    return out


def cycle_checks(
    runs_root: str | Path, *, settings: Mapping[str, Any] | None = None
) -> dict[str, dict[str, Any]]:
    """The one check of every supersession record under RUNS_DIR (module docstring, "The
    engineering cycle"), by its path under RUNS_DIR (``qualification/c<N>/<name>``): its
    ``candidate``, whether it could be ``readable`` (its readers name one that cannot), whether
    it is ``accepted`` and the ``problems`` that name why not, and the run ids it names under
    ``launchers`` and ``held`` (a journal line counts only for those, and only when the record
    is accepted). A record in a folder whose name is not exactly ``c<N>`` belongs to no
    candidate: a problem. Every
    reader of what a record voids or holds applies an accepted record alone
    (``study3_records``). ``settings``: configs/experiments/qualification.yaml as loaded (this
    checkout's, by default); when they are not the qualification's, no record is accepted."""
    root = Path(runs_root)
    files = sorted((root / "qualification").glob("c*/supersede-*.json"))
    if not files:
        return {}
    try:
        scope = Scope.of(load_config("experiments/qualification") if settings is None
                         else settings)  # fmt: skip
    except (QualificationError, OSError, ValueError) as exc:
        return {path.relative_to(root).as_posix(): {
            "candidate": None, "readable": True, "accepted": False, "launchers": [], "held": [],
            "problems": [f"{path}: the qualification's settings cannot be read ({exc}): no "
                         "supersession record is checked, and none voids anything"]}
            for path in files}  # fmt: skip
    return {path.relative_to(root).as_posix(): {
        "candidate": check["candidate"], "readable": check["readable"],
        "accepted": check["accepted"], "problems": check["problems"],
        "launchers": check["launchers"], "held": check["held"]}
        for path, check in _checks(root, scope, _journal(root, [])).items()}  # fmt: skip


# ------------------------------------------------------------------ the reading


class _Reading:
    """The candidate's numerical stage read as ``qualify.py verdict`` reads it (module
    docstring, "The inputs of a rate"), every unreadable record a problem."""

    def __init__(self, root: Path, candidate: int, scope: Scope) -> None:
        self.root, self.candidate, self.scope = root, candidate, scope
        self.candidate_dir = root / "qualification" / f"c{candidate}"
        self.base = self.candidate_dir / STAGE
        self.problems: list[str] = []
        self.inputs: dict[Path, str] = {}  # every record file read, and what it is
        self.cases: list[dict[str, Any]] = []  # the case verdicts, superseded ones included
        self.old_controls: list[dict[str, Any]] = []  # the superseded controls' verdicts
        self.voided: dict[str, str] = {}  # a void record's verdict (record_sha256) -> why
        self.superseded: list[dict[str, Any]] = []
        self.folders: list[dict[str, Any]] = []
        self.issued: list[tuple[str, float, int]] = []  # the issued cases (``read``)
        self._read: dict[Path, dict[str, Any] | None] = {}  # every record ``_flagged`` read
        found = sorted(self.candidate_dir.glob("supersede-*.json"))
        self.supersessions = [(p, self._flagged(p, "supersession record")) for p in found]
        self.voiding: list[tuple[Path, dict[str, Any]]] = []  # the accepted cycle (``read``)

    def _flagged(self, path: Path, what: str) -> dict[str, Any] | None:
        """A record read once, an unreadable one named once."""
        if path in self.inputs:
            return self._read.get(path)
        self.inputs[path] = what
        if under_sealed(path, self.root):
            self.problems.append(f"{path}: {sealed_record(f'a {what}', path, self.root)}")
            self._read[path] = None
            return None
        record = _object(path, self.root)
        if record is None:
            self.problems.append(f"{path}: an unreadable {what}")
        self._read[path] = record
        return record

    def read(self, issued: Sequence[tuple[str, float, int]] = (),
             checks: Mapping[Path, Mapping[str, Any]] | None = None) -> None:  # fmt: skip
        """Every case folder, then the cases the engineering cycle kept on a resolved launcher
        run, beside their records' own verdicts (no record classified that run);
        ``issued``: the (optimizer, rate, seed) the issued starts name; ``checks``: the one check
        of every supersession record (``_checks``), of which the candidate's: an accepted record
        voids, any other is a problem and voids nothing."""
        self.issued = list(issued)
        for path, check in (checks or {}).items():
            if path.parent != self.candidate_dir:
                continue
            self.problems += [p for p in check["problems"] if p not in self.problems]
            if check["accepted"]:
                self.voiding.append((path, check["record"]))
        for folder in sorted({f.parent for f in self.base.glob("s*/*/*-attempt-*")}):
            try:
                self._folder(folder)
            except Exception as exc:  # a malformed record no rule below names
                self.problems.append(f"{folder}: its records cannot be read: "
                                     f"{type(exc).__name__}: {exc}")  # fmt: skip
        for path, record in self.voiding:  # cases kept on a resolved launcher run
            kept_cases = record.get("kept_cases") or []
            if not isinstance(kept_cases, list):
                self.problems.append(f"{path}: its kept cases are not a list")
                continue
            for kept in kept_cases:
                try:
                    resolution = kept.get("resolution") if isinstance(kept, Mapping) else None
                    if (isinstance(resolution, Mapping) and resolution
                            and kept.get("stage") == STAGE):  # fmt: skip
                        verdict = resolved_verdict(resolution)
                        self._issued_case(path, verdict["case"])
                        self.cases.append(verdict)
                except Exception as exc:  # as a case record the verdicts cannot read
                    self.problems.append(f"{path}: a kept case the verdicts cannot read: "
                                         f"{type(exc).__name__}: {exc}")  # fmt: skip

    def _issued_case(self, where: Path, case: Any) -> None:
        """A record's case must be one an issued start of the candidate's qualification names
        (every run bound to an issued cell): a problem otherwise."""
        if not (isinstance(case, Mapping) and case.get("candidate") == self.candidate
                and case.get("engineering") is False and isinstance(case.get("optimizer"), str)
                and _number(case.get("lr")) and type(case.get("seed")) is int):  # fmt: skip
            self.problems.append(f"{where}: a record of no numerical case of candidate "
                                 f"{self.candidate}: {case!r}")  # fmt: skip
            return
        optimizer, lr, seed = case["optimizer"], case["lr"], case["seed"]
        if not any(o == optimizer and s == seed and math.isclose(r, lr, rel_tol=RATE_TOLERANCE)
                   for o, r, s in self.issued):  # fmt: skip
            self.problems.append(f"{where}: case s{seed}/{optimizer}-{lr:g}, which no issued "
                                 f"start of candidate {self.candidate}'s qualification "
                                 "names")  # fmt: skip

    def _entries(self, folder: Path, name: str) -> Iterator[tuple[Path, dict | None, dict]]:
        """(voiding record, its record, an attempt entry of this case) for every attempt the
        case's restart records and the candidate's supersession records name."""
        restarts = sorted(
            (int(m[2]), p) for p in folder.glob("restart-*.json")
            if (m := VOIDING_FILE.fullmatch(p.name))
        )  # fmt: skip
        voiders = [(p, self._flagged(p, "case-restart record")) for _, p in restarts]
        voiders += list(self.voiding)  # the valid cycle alone
        for path, record in voiders:
            entries = (record or {}).get("attempts") or []
            if not isinstance(entries, list):
                self.problems.append(f"{path}: its attempts are not a list")
                continue
            for entry in entries:
                if (isinstance(entry, dict) and entry.get("case", name) == name
                        and entry.get("stage", STAGE) == STAGE):  # fmt: skip
                    yield path, record, entry

    def _folder(self, folder: Path) -> None:
        """A case folder (``qualify.py verdict``'s loop): its attempts, its case verdict and its
        superseded records."""
        name = str(folder.relative_to(self.base))
        entries = list(self._entries(folder, name))
        named: dict[str, str] = {}
        cycle_named: set[str] = set()
        for path, record, entry in entries:
            if isinstance(entry.get("sha256"), str):
                named.setdefault(entry["sha256"], _void_reason(path, record))
                if path.name.startswith("supersede-"):
                    cycle_named.add(entry["sha256"])
        named_attempts = {(e.get("part"), e.get("attempt")) for _, _, e in entries}
        listing: dict[str, list[dict[str, Any]]] = {}
        records: dict[tuple[str, int], dict[str, Any] | None] = {}
        stale: list[tuple[str, int, dict | None, str | None, str | None]] = []
        for part in PARTS:
            attempts = attempt_files(folder, part)
            for n, state in attempts.items():
                path = folder / f"{part}-attempt-{n}.json"
                recorded = state == "recorded"
                record = self._flagged(path, f"{part} attempt record") if recorded else None
                sha = _file_sha256(path, self.root) if recorded else None
                void = named.get(sha) if sha is not None else None
                restart, supersession = (None, void) if sha in cycle_named else (void, None)
                listing.setdefault(part, []).append(
                    {"attempt": n, "state": state, "path": str(path), "sha256": sha,
                     "status": (record or {}).get("status"), "kind": (record or {}).get("kind"),
                     "restart": restart, "supersession": supersession})  # fmt: skip
                records[(part, n)] = record
                if recorded and (void is not None or n < max(attempts)):
                    stale.append((part, n, record, restart, supersession))

        def latest(part: str) -> tuple[int | None, dict[str, Any] | None]:
            """The part's latest attempt when it is recorded and named by no voiding record."""
            rows = listing.get(part) or []
            if not rows or rows[-1]["state"] != "recorded" or rows[-1]["sha256"] in named:
                return None, None
            return rows[-1]["attempt"], records[(part, rows[-1]["attempt"])]

        counts: dict[str, int | None] = {}
        counts["baseline"], baseline = latest("baseline")
        counts["rest"], rest = latest("rest")
        counts["control"], control = latest("control")
        rests = listing.get("rest") or []
        interrupted = None
        if (rests and rests[-1]["state"] == "interrupted"
                and ("rest", rests[-1]["attempt"]) not in named_attempts):  # fmt: skip
            interrupted = rests[-1]["attempt"]
            try:
                rest = verdicts.rest_from_trajectories(self._trajectories(folder, interrupted))
            except Exception as exc:  # as a record the verdicts cannot read
                self.problems.append(f"{folder}: the verdicts cannot read its trajectory results: "
                                     f"{type(exc).__name__}: {exc}")  # fmt: skip
                rest = None
            counts["rest"] = None if rest is None else interrupted
        named_cases = [r.get("case") for r in records.values() if r is not None]
        if interrupted is not None and rest is not None:
            named_cases.append(rest.get("case"))
        for case in {json.dumps(c, sort_keys=True, default=str): c for c in named_cases}.values():
            self._issued_case(folder, case)
        verdict = None
        if baseline is not None or rest is not None:
            verdict = self._case_verdict(name, baseline, rest)
        mine = []
        for part, n, record, restart, supersession in stale:  # shown; void, or counted
            void = restart or supersession
            mark = {"part": part, "attempt": n, "restart": restart, "supersession": supersession}
            entry = {"case": name, "sha256": record_sha256(record), **mark}
            self.superseded.append(entry)
            mine.append(entry)
            if part == "control":  # counted against its rate as a superseded case is
                checked = self._control_verdict(name, record)
                if checked is None:
                    continue
                checked = {**checked, "superseded": mark}
                self.old_controls.append(checked)
                if void is not None and checked["record_sha256"] is not None:
                    self.voided[checked["record_sha256"]] = void
                continue
            pair = (record, None) if part == "baseline" else (None, record)
            old = self._case_verdict(name, *pair, superseded=mark)
            if old is not None and void is not None:
                self.voided[old["record_sha256"]] = void
        seed = folder.parent.name[1:]
        self.folders.append({
            "case": name, "folder": str(folder), "key": folder.name,
            "seed": int(seed) if seed.isdigit() else None,
            "parts": {part: {"attempts": listing.get(part, []), "counts": counts[part],
                             "from_trajectories": part == "rest" and interrupted is not None
                             and counts["rest"] is not None}
                      for part in PARTS},
            "verdict": verdict, "superseded": mine, "control": control,
            "control_attempt": counts["control"], "released": self._released(folder),
        })  # fmt: skip

    def _case_verdict(self, name: str, baseline: Any, rest: Any, superseded: Any = None) -> Any:
        try:
            verdict = verdicts.case_verdict(baseline, rest, self.scope)
        except Exception as exc:  # a record the verdicts cannot read
            self.problems.append(f"{self.base / name}: the verdicts cannot read its records: "
                                 f"{type(exc).__name__}: {exc}")  # fmt: skip
            return None
        if superseded is not None:
            verdict = {**verdict, "superseded": superseded}
        self.cases.append(verdict)
        return verdict

    def _control_verdict(self, name: str, record: Any) -> dict[str, Any] | None:
        try:
            return verdicts.control_verdict(record, self.scope, self.candidate)
        except Exception as exc:
            self.problems.append(f"{self.base / name}: the verdicts cannot read its control: "
                                 f"{type(exc).__name__}: {exc}")  # fmt: skip
            return None

    def _trajectories(self, folder: Path, attempt: int) -> list[dict[str, Any]]:
        """The trajectory results an interrupted rest attempt published."""
        files = sorted(folder.glob(f"rest-attempt-{attempt}.trajectory-*.json"))
        found = [self._flagged(f, "trajectory result") for f in files]
        return [r for r in found if r is not None]

    def _released(self, folder: Path) -> list[dict[str, Any]]:
        """The case's records of the checkpoints ``finish`` removed on purpose."""
        out = []
        for path in sorted(folder.glob(f"{RELEASED}-*.json")):
            record = self._flagged(path, RELEASE_RECORD) or {}
            out.append({"path": str(path), "removed": record.get("removed"),
                        "release": record.get("release"),
                        "technical": record.get("technical")})  # fmt: skip
        return out


# ------------------------------------------------------------------ the launcher runs located


class _Located:
    """``qualify.py``'s ``located`` for the issued runs: the one directory of the cell's config
    hash now, settled (completed or diverged, or with a terminal record), that no ``superseded``
    line of the journal names for its label and that hash; each experiment's ``meta.json`` files
    read once. A run directory whose folder or record files resolve under a sealed folder is read
    not at all: a problem, into ``problems``, and no launcher run is located there."""

    def __init__(self, root: Path, superseded: Sequence[Mapping[str, Any]],
                 problems: list[str]) -> None:  # fmt: skip
        self.root, self.superseded, self.problems = root, superseded, problems
        self.metas: dict[str, list[tuple[Path, dict[str, Any] | None]]] = {}

    def _meta(self, folder: Path) -> dict[str, Any] | None:
        why = _sealed_records(folder, self.root)
        if why is not None:
            self.problems.append(f"{folder}: {why}; no launcher run is located there")
            return None
        return _object(folder / "meta.json", folder)

    def __call__(self, run: _Issued) -> Path | None:
        if run.config_hash is None:  # its cell does not compose: nothing to locate under
            return None
        if run.experiment not in self.metas:
            directory = self.root / run.experiment
            # the decision records' folder is no run (study3_records.DECISIONS)
            self.metas[run.experiment] = [(d, self._meta(d)) for d in sorted(directory.glob("*"))
                                          if d.name != DECISIONS]  # fmt: skip
        named = {r["run_id"] for r in self.superseded if r.get("run") == run.label
                 and r.get("experiment") == run.experiment
                 and r.get("config_hash") == run.config_hash}  # fmt: skip
        found = [d for d, meta in self.metas[run.experiment]
                 if meta is not None and meta.get("config_hash") == run.config_hash
                 and d.name not in named
                 and (meta.get("status") in SETTLED
                      or (d / "terminal.json").is_file())]  # fmt: skip
        return found[0] if len(found) == 1 else None


# ------------------------------------------------------------------ the rates


def _of_rate(verdict: Mapping[str, Any], optimizer: str, lr: float) -> bool:
    """Whether a case or control verdict names this optimizer and rate (the verdict's rule)."""
    case = verdict.get("case")
    if not isinstance(case, Mapping) or case.get("optimizer") != optimizer:
        return False
    rate = case.get("lr") or math.nan
    return _number(rate) and math.isclose(rate, lr, rel_tol=RATE_TOLERANCE)


def _add_rate(rates: list[tuple[str, float]], optimizer: Any, lr: Any) -> None:
    if not (isinstance(optimizer, str) and _number(lr)):
        return
    if not any(o == optimizer and math.isclose(r, lr, rel_tol=RATE_TOLERANCE) for o, r in rates):
        rates.append((optimizer, float(lr)))


def _cache(path: Path, problems: list[str], root: Path) -> dict[str, Any] | None:
    """``verdict.json`` as a cache: its rates (optimizer and rate) and states, its identity;
    ``valid`` unless it is unreadable (or resolves under a sealed folder, never opened) or holds no
    rate as the verdict writes one."""
    if not path.exists():
        return None
    before = len(problems)
    sealed = under_sealed(path, root)
    record = None if sealed else _object(path, root)
    if record is None:
        problems.append(f"{path}: " + (sealed_record("a verdict cache", path, root) if sealed
                                       else "an unreadable verdict cache"))  # fmt: skip
        return {"path": str(path), "sha256": _file_sha256(path, root), "states": {}, "rates": {},
                "readable": False, "valid": False}  # fmt: skip
    states = record.get("states")
    rates = record.get("rates")
    if not (isinstance(states, Mapping) and isinstance(rates, Mapping)):
        problems.append(f"{path}: the verdict cache holds no rate states")
        states, rates = {}, {}
    named: dict[str, tuple[Any, Any]] = {}
    for key in states:
        entry = rates.get(key)
        named[key] = ((entry.get("optimizer"), entry.get("lr")) if isinstance(entry, Mapping)
                      else (None, None))  # fmt: skip
        if not (isinstance(named[key][0], str) and _number(named[key][1])):
            problems.append(f"{path}: the cache names rate {key!r} without its arm and rate")
    provenance = record.get("provenance") if isinstance(record.get("provenance"), Mapping) else {}
    gpu_code = provenance.get("gpu_code")
    return {"path": str(path), "sha256": _file_sha256(path, root), "readable": True,
            "valid": len(problems) == before,
            "candidate": record.get("candidate"), "stage": record.get("stage"),
            "campaign": record.get("campaign"), "states": dict(states), "rates": named,
            "gpu_code": gpu_code.get("sha256") if isinstance(gpu_code, Mapping) else None,
            "settings": provenance.get("settings")}  # fmt: skip


def _compare(
    cache: dict[str, Any], rates: Mapping[str, Mapping[str, Any]], *, candidate: int,
    expected_code: str, policy: Mapping[str, Any], problems: list[str],
) -> list[str]:  # fmt: skip
    """Every disagreement of the cache with the recomputation, into ``problems``; returns the
    recomputed rates the cache does not name."""
    path, before = cache["path"], len(problems)
    if cache["readable"]:
        if cache["candidate"] != candidate or cache["stage"] != STAGE:
            problems.append(f"{path}: the cache is of candidate {cache['candidate']!r}, stage "
                            f"{cache['stage']!r}, not candidate {candidate}'s "
                            f"{STAGE} stage")  # fmt: skip
        if cache["gpu_code"] != expected_code:
            problems.append(f"{path}: the cache was written under GPU code {cache['gpu_code']!r}, "
                            f"not the expected {expected_code!r}")  # fmt: skip
        if cache["settings"] != policy:
            problems.append(f"{path}: the cache was written under other settings than the "
                            "registered ones")  # fmt: skip
    seen = set()
    for key, state in cache["states"].items():
        optimizer, lr = cache["rates"].get(key, (None, None))
        mine = next((k for k, r in rates.items() if r["optimizer"] == optimizer and _number(lr)
                     and math.isclose(r["lr"], lr, rel_tol=RATE_TOLERANCE)), None)  # fmt: skip
        if mine is None:
            problems.append(f"{path}: the cache's rate {key!r} is not recomputed")
            continue
        seen.add(mine)
        if rates[mine]["state"] != state:
            problems.append(f"{mine}: the records give {rates[mine]['state']!r}, the cache "
                            f"{path} {state!r}")  # fmt: skip
    cache["agrees"] = cache["valid"] and len(problems) == before
    return [key for key in rates if key not in seen]


def read_qualification(
    runs_root: str | Path, candidate: int, *, expected_code: str, settings: Mapping[str, Any]
) -> dict[str, Any]:
    """The candidate's numerical qualification from its records (module docstring): per rate
    (``rates``, keyed ``<optimizer>-<rate:g>``) its ``arm``, optimizer, ``lr``, ``state`` and
    ``verdict`` (``rate_verdict``'s), its ``cases`` (each folder's ``parts`` with every attempt
    and the one that ``counts``, its case ``verdict`` and its ``superseded`` records), its
    ``control`` (an FA rate's: the folder, the attempt and the verdict the rate read), its issued
    ``runs`` with the one ``located``, the ``superseded`` and the ``held`` ones, and its
    ``launchers``; the rates' ``states``; every case verdict (``cases``), the ``superseded``
    records, the ``restarts`` and ``supersessions``, the ``held`` runs, the cases of no rate
    (``unassigned``); the ``cache`` and the rates it does not name (``uncached``); the
    ``inputs``, the record files of the candidate's qualification the reading took as inputs,
    sorted (the part attempt records, the interrupted rests' trajectory results, the case
    restart and supersession records; not the cache, nor ``finish``'s release records, which
    decide no state); and the ``problems``, sorted. ``expected_code`` is the GPU code's sha256
    the qualification must carry (the fixation record's); ``settings``
    configs/experiments/qualification.yaml as loaded. ``QualificationError`` (a ``ValueError``)
    for settings that are not the qualification's or a candidate outside them."""
    if not (isinstance(expected_code, str) and expected_code):
        raise ValueError(f"expected_code is the GPU code's sha256, not {expected_code!r}")
    scope = Scope.of(settings)
    scope.seeds(candidate)  # a candidate outside the scope: QualificationError
    policy = settings_record(settings)
    root = Path(runs_root)
    reading = _Reading(root, candidate, scope)
    problems = reading.problems
    journal = _journal(root, problems)
    issued = _issued(root, journal, candidate, problems)
    cases = [(r.optimizer, r.lr, r.seed) for r in issued if r.optimizer is not None]
    checks = _checks(root, scope, journal, {candidate: cases})  # the one check of every cycle
    cycles = {path.relative_to(root).as_posix(): check for path, check in checks.items()}
    holds, unreadable = read_holds(root, cycles=cycles)
    problems += _hold_problems(root, candidate, unreadable)
    reading.read(cases, checks)
    # a superseded line counts only through an accepted record that names its run
    superseded_lines = []
    for line in _named_runs(journal, "superseded"):
        why = _unbacked(line, cycles, root)
        if why is None:
            superseded_lines.append(line)
        else:
            problems.append(f"{root / 'launch' / (line['campaign'] + '.jsonl')}: a superseded "
                            f"line of {line['run_id']} counts for nothing: {why}")  # fmt: skip
    cache = _cache(reading.candidate_dir / VERDICT, problems, root)
    order: list[tuple[str, float]] = []
    for run in issued:
        _add_rate(order, run.optimizer, run.lr)
    for verdict in [*reading.cases, *reading.old_controls]:
        case = verdict.get("case") if isinstance(verdict.get("case"), Mapping) else {}
        _add_rate(order, case.get("optimizer"), case.get("lr"))
    for optimizer, lr in (cache or {}).get("rates", {}).values():
        _add_rate(order, optimizer, lr)
    located = _Located(root, superseded_lines, problems)
    held = _held(root, candidate, holds, issued)
    inputs = {"scope": scope, "candidate": candidate, "code": {"sha256": expected_code},
              "policy": policy}  # fmt: skip
    rates: dict[str, dict[str, Any]] = {}
    for optimizer, lr in order:
        rate = _rate(reading, optimizer, lr, issued, located, superseded_lines, held, **inputs)
        if rate["key"] in rates:
            problems.append(f"two rates have the key {rate['key']!r}: "
                            f"{rates[rate['key']]['lr']!r} and {lr!r}")  # fmt: skip
        rates[rate["key"]] = rate
    assigned = {id(v) for rate in rates.values() for v in rate.pop("_inputs")}
    uncached = [] if cache is None else _compare(cache, rates, candidate=candidate,
                                                 expected_code=expected_code, policy=policy,
                                                 problems=problems)  # fmt: skip
    return {
        "candidate": candidate, "stage": STAGE, "expected_code": expected_code,
        "rates": rates, "states": {key: rate["state"] for key, rate in rates.items()},
        "cases": reading.cases, "superseded": reading.superseded,
        "voided": dict(reading.voided), "restarts": _restarts(reading),
        "supersessions": [_supersession(p, r, root) for p, r in reading.supersessions
                          if VOIDING_FILE.fullmatch(p.name)],
        "held": held, "issued": [r.record() for r in issued],
        "unassigned": [_case_name(v) for v in [*reading.cases, *reading.old_controls]
                       if id(v) not in assigned],
        "cache": cache, "uncached": uncached,
        "inputs": sorted(str(p) for p, what in reading.inputs.items() if what != RELEASE_RECORD),
        "problems": sorted(set(problems)),
    }  # fmt: skip


def _rate(
    reading: _Reading, optimizer: str, lr: float, issued: Sequence[_Issued], located: _Located,
    superseded_lines: Sequence[Mapping[str, Any]], held: Sequence[Mapping[str, Any]], *,
    scope: Scope, candidate: int, code: Mapping[str, Any], policy: Mapping[str, Any],
) -> dict[str, Any]:  # fmt: skip
    """One rate, its inputs assembled as ``qualify.py verdict`` assembles them (module
    docstring, "The inputs of a rate"), and its ``rate_verdict``. ``_inputs`` holds the case and
    control verdicts it read, for the caller to pop."""
    key = f"{optimizer}-{lr:g}"
    mine = [v for v in reading.cases if _of_rate(v, optimizer, lr)]
    olds = [c for c in reading.old_controls if _of_rate(c, optimizer, lr)]
    control, read_control = None, None
    if optimizer == scope.control_arm:  # the case of the first seed holds the control
        where = reading.base / f"s{scope.seeds(candidate)[0]}" / key
        holder = next((f for f in reading.folders if f["folder"] == str(where)), None)
        record = None if holder is None else holder["control"]
        if record is not None:
            control = reading._control_verdict(str(where.relative_to(reading.base)), record)
        attempt = None if holder is None else holder["control_attempt"]
        read_control = {"folder": str(where), "attempt": attempt, "verdict": control}
    runs = [r for r in issued if r.optimizer == optimizer
            and math.isclose(r.lr, lr, rel_tol=RATE_TOLERANCE)]  # fmt: skip
    launchers: dict[int, dict[str, Any] | None] = {}
    listed = []
    for run in runs:  # one per seed, unless two issued runs are one case (none located then)
        run_dir = None if run.ambiguous else located(run)
        launchers[run.seed] = None if run_dir is None else launcher_identity(run_dir)
        own = set(run.run_ids)
        listed.append({**run.record(), "located": None if run_dir is None else run_dir.name,
                       "superseded": {r["run_id"]: r.get("reason") for r in superseded_lines
                                      if r["run_id"] in own},
                       "held": {h["run_id"]: h["reason"] for h in held
                                if _holds(h, run)}})  # fmt: skip
    try:
        verdict = verdicts.rate_verdict(
            mine, scope=scope, candidate=candidate, optimizer=optimizer, lr=lr, code=code,
            policy=policy, control=control, superseded_controls=olds, launchers=launchers,
            voided=reading.voided,
        )  # fmt: skip
    except Exception as exc:  # a record the verdicts cannot read
        reading.problems.append(f"{key}: the rate's verdict cannot be computed: "
                                f"{type(exc).__name__}: {exc}")  # fmt: skip
        verdict = None
    return {"key": key, "arm": ARMS.get(optimizer), "optimizer": optimizer, "lr": lr,
            "state": None if verdict is None else verdict["state"], "verdict": verdict,
            "cases": [f for f in reading.folders if f["key"] == key],
            "control": read_control, "runs": listed, "launchers": launchers,
            "held": [h for h in held if any(_holds(h, run) for run in runs)],
            "_inputs": [*mine, *olds]}  # fmt: skip


def _case_name(verdict: Mapping[str, Any]) -> Any:
    case = verdict.get("case")
    return case.get("name") if isinstance(case, Mapping) else None


def _held(
    root: Path, candidate: int, holds: Sequence[Mapping[str, Any]], issued: Sequence[_Issued]
) -> list[dict[str, Any]]:
    """The held launcher runs of the candidate's qualification (module docstring, "Held
    runs"), from ``study3_records.read_holds``: a hold of an issued label (whatever the config
    hash it was written with) or of an issued run id, or one the candidate's supersession
    records name; each with its ``sources``, its ``records`` (evidence with their sha256) and
    the issued ``case`` it holds."""
    cycles = f"qualification/c{candidate}/"
    held = []
    for hold in holds:
        run = next((r for r in issued if _holds(hold, r)), None)
        records = [dict(r) for r in hold["records"]]
        if run is None and not any(r["kind"] == "supersession" and r["path"].startswith(cycles)
                                   for r in records):  # fmt: skip
            continue
        sources = [f"journal {Path(r['path']).stem}" if r["kind"] == "journal"
                   else str(root / r["path"]) for r in records]  # fmt: skip
        case = None if run is None or run.optimizer is None else {
            "arm": ARMS.get(run.optimizer), "optimizer": run.optimizer, "lr": run.lr,
            "seed": run.seed}  # fmt: skip
        held.append({"run_id": hold["run_id"], "run": hold["run"],
                     "experiment": hold["experiment"], "config_hash": hold["config_hash"],
                     "reason": hold["reason"], "case": case, "sources": sources,
                     "records": records})  # fmt: skip
    return held


def _hold_problems(root: Path, candidate: int, unreadable: Sequence[str]) -> list[str]:
    """The records ``read_holds`` could not read that may hold a run of the candidate's
    qualification (any launch log, the candidate's supersession records), named as the reading
    names its own unreadable records, so that one record is one problem: a hold it holds would
    otherwise be missing from ``held`` without a word."""
    out = []
    for problem in unreadable:
        where, _, what = problem.partition(": ")
        if where.startswith(("launch/", f"qualification/c{candidate}/")):
            out.append(f"{root / where}: {what}")
    return out


def _holds(hold: Mapping[str, Any], run: _Issued) -> bool:
    """Whether a hold applies to an issued run: it names the run's label in its experiment,
    under any config hash (``launch.held_runs``), or one of its attempts."""
    return (hold.get("run"), hold.get("experiment")) == run.key or hold["run_id"] in run.run_ids


def _restarts(reading: _Reading) -> list[dict[str, Any]]:
    """The case restart records of the stage, as listed: the case, the reason, the attempts it
    names and the launcher run it supersedes."""
    out = []
    for path in sorted(reading.base.glob("s*/*/restart-*.json")):
        record = reading._flagged(path, "case-restart record") or {}
        launcher = record.get("launcher") if isinstance(record.get("launcher"), Mapping) else {}
        attempts = record.get("attempts") if isinstance(record.get("attempts"), list) else []
        out.append({"path": str(path), "sha256": _file_sha256(path, reading.root),
                    "case": str(path.parent.relative_to(reading.base)),
                    "reason": record.get("reason"), "launcher_run_id": launcher.get("run_id"),
                    "attempts": [(a.get("part"), a.get("attempt"), a.get("state"))
                                 for a in attempts if isinstance(a, Mapping)]})  # fmt: skip
    return out


def _supersession(path: Path, record: Mapping[str, Any] | None, root: Path) -> dict[str, Any]:
    """A supersession record as listed: what it names and keeps (``root``: RUNS_DIR, the base of
    the sealed-path rule for its sha256)."""
    record = record or {}
    fix = record.get("fix_gpu_code") if isinstance(record.get("fix_gpu_code"), Mapping) else {}
    old = record.get("old_gpu_code") if isinstance(record.get("old_gpu_code"), Mapping) else {}
    return {"path": str(path), "sha256": _file_sha256(path, root),
            "old_record": record.get("old_record"), "diagnosis": record.get("diagnosis"),
            "fix": record.get("fix"), "old_gpu_code": old.get("sha256"),
            "fix_gpu_code": fix.get("sha256"),
            "cases_requalified": record.get("cases_requalified"),
            "kept_cases": record.get("kept_cases"),
            "launchers": [e.get("run_id") for e in record.get("launchers") or []
                          if isinstance(e, Mapping)]}  # fmt: skip


def supersessions(runs_root: str | Path, candidate: int) -> list[dict[str, Any]]:
    """The candidate's supersession records (``c<N>/supersede-<n>.json``) as
    ``read_qualification`` lists them, in their order; an unreadable one, or one that resolves
    under a sealed folder (never opened), is listed with its fields None (``read_qualification``
    names it a problem)."""
    root = Path(runs_root)
    folder = root / "qualification" / f"c{candidate}"
    return [
        _supersession(path, _object(path, root), root)
        for path in sorted(folder.glob("supersede-*.json"))
        if VOIDING_FILE.fullmatch(path.name)
    ]


def rate_of(reading: Mapping[str, Any], arm: str, lr: float) -> dict[str, Any] | None:
    """The rate of a ``read_qualification`` reading for ``arm`` (its letter, A, F or P, or its
    optimizer) and ``lr`` (relative tolerance 10⁻⁹), or None when the records name none."""
    for rate in (reading.get("rates") or {}).values():
        if arm in (rate["arm"], rate["optimizer"]) and math.isclose(
                rate["lr"], lr, rel_tol=RATE_TOLERANCE):  # fmt: skip
            return rate
    return None
