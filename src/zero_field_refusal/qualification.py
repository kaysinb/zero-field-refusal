"""Study 3: the qualification harness and bridge at step 25, with failure kinds and verdicts.

The rules are docs/study3-preregistration.md, "Numerical qualification" (all its subsections),
"Failure routes" and "Amendment 2" (the same-path control); the registered values are
configs/experiments/qualification.yaml (``QualificationScope``: the bound 0.01·K*, K* 6.0 and 5.0
for candidates 1 and 2, the norm 5e-5, the Gaussian seeds 3 and 4, the survival share 0.5 with one
retry ×10, the warning τ = K*·max(0.001, 3·e_eng), the seeds). The study-2 harness is reused, not
copied (``fa_control_diagnostic``: ``twin_setup``, ``Twins``, ``run_twin``, ``reference_twin``,
``seed_directions``, ``apply_perturbation``, ``same_path_run``).

**A case** (``QualificationCase``: one arm, one rate, one seed of a candidate, numerical or
engineering) is a K-only launcher run that saved its trainable parameters right after update 25
(``schedule``, ``tracking.endpoint_checkpoint``), then two parts in fresh processes
(``run_part``), each written per attempt to ``QualificationCase.attempt_path``:

- **baseline**: the harness's fp32 run of the launcher run's config from θ₀, with its monitor;
  its step-25 trainable parameters saved to ``harness-baseline.pt`` (and, in the engineering stage
  only, its states at ``diagnostic_steps``); D₂₅ against the launcher checkpoint; the bridge's
  checks (``bridge_checks``); and, as a diagnostic, its rows and monitor against the launcher
  run's (``bitwise_rows``, scientific fields only: ``fa_control_diagnostic.compare_archive``);
- **rest**: the repeat (fp32), the float64 reference (``reference_twin``: float64 loss,
  accumulation and optimizer arithmetic; on ``torch.OutOfMemoryError`` alone the
  FLOAT64_ACCUMULATION fallback, the state reset and the failed attempt kept) and the shifts
  s1–s4 at the norm 5e-5 (``seed_directions``: s1 and s2 are ± the F-preconditioned first update
  −P·g of the seed's generation-path soft residual, P from the seed's v₀; s3 and s4 the fixed
  Gaussian directions of seeds 3 and 4), each applied to θ₀ before the first update with its
  realized norm, survival and initial functional difference D(θ₀ + s, θ₀) recorded
  (``apply_perturbation``: a realized norm below half the intended one is retried once at ×10).
  Each trajectory's D₂₅ against the launcher checkpoint and against ``harness-baseline.pt``; with
  ``diagnostic_steps`` (engineering), D against the baseline's states at those steps too. The
  record names the baseline record it builds on (path and sha256) and that checkpoint's sha256;
- **control** (amendment 2): the same-path FA control (``same_path_run``: θ₀'s training-path
  distributions as targets, float64 loss and accumulation) for 25 steps on the candidate's first
  numerical seed at a selected FA rate, from that launcher run's config, with K_scoring at step 25
  and the streams and v₀ checked against the run's records (``control_verdict``).

Thirteen comparisons at step 25 (``COMPARISONS``): the launcher checkpoint against the baseline,
the repeat, the reference and the four shifts (7), and the harness baseline against the repeat,
the reference and the shifts (6); each passes with D₂₅ ≤ 0.01·K*. D is ``Twins.divergence``:
max(KL(A‖B), KL(B‖A)) on the scoring panel's responses, full vocabulary; no distribution is
stored. Intermediate D belongs to the engineering stage's diagnostics alone.

**Order and state** (``run_part``). ``runtime.configure_numerics`` is called first, before any
CUDA work, and its record is kept; then the launcher run is checked (``launcher_problems``, file
reads only), then ``prechecks.require_path_check``. The launcher run's identity comes first:
seed, arm, rate, batch, steps, ``eval.k_only``; its ``train``, ``optimizer``, ``execution``,
``data`` and ``seed`` equal to the confirmation config's ("Config agreement",
``confirmation_config``); its recorded numerics setup equal to this process's; the hashes of θ₀
and, for an arm that reads it, v₀; the run's commit (clean) with this checkout's GPU-code
identity, and its recorded GPU, driver, CUDA, Python, packages and container image this
process's (``code_environment_problems``: study 2's ``code_check`` and ``environment_check``); a
completed run with a whole terminal record. A run that departs from it is technical, whatever
stopped it. Then a conforming run stopped by an
instability rule (``schedule.INSTABILITY``) is a scientific failure; one stopped otherwise,
without its step-25 evaluation, or without a checkpoint whose file's sha256 is the recorded one,
is technical; either way nothing runs, and the record still names the run and its Pod
(``launcher_record``; when they cannot be read, as missing with the error, and the record is
published all the same). Every trajectory starts from θ₀ with no gradient, a fresh optimizer and
the RNG seeded as ``run_twin`` seeds it; v₀ is never written (``v0_unchanged``). An AdamW case,
whose run reads no v₀, sets the seed's F v₀ up for s1 and s2 alone, from the F optimizer group
(``setup_config``). An exception is the part's record, ``failed``, with the trajectories done
before it.

**Trajectory results** (``QualificationCase.trajectory_path``). Each rest trajectory's result
(its run, its comparisons, the bridge checks as of then, v₀ unchanged) is published as soon as it
is done, before the next one starts, atomically and durably (``publish``), bound to its inputs:
the case, the attempt, the launcher run (identity and Pod), the baseline record's sha256, the
baseline and launcher checkpoints' sha256, the numerics setup, the GPU code and the settings. An
interruption keeps what was done. When the bridge checks hold and a result is a scientific
failure (``scientific_result``: the declared rules on the repeat, a non-finite reference or
shift, the baseline's D₂₅ against the reference or a shift beyond the bound), the part publishes
its record at once, ``status: "decided"`` and kind scientific, and runs no further trajectory.

**Instability of the unshifted trajectories** (``harness_instability``). The baseline and the
repeat read the monitor at the launcher's cadence, and the launcher's declared rules are applied
to their rows and readings after the run: a non-finite loss, gradient or weight, the ×10 gradient
rule (``schedule.InstabilityGuard``), a monitor K above its maximum; the rule that fired, its
step and values are recorded. The reference and the shifts are held to non-finite values alone.

**The bridge** (``bridge_checks``, each false one technical): each training row's ``tokens`` and
``denominator`` and the step indices exactly equal to the launcher run's (a trajectory a non-finite
value stopped, a scientific failure, against the launcher's rows of its length); the training
stream's identity; v₀'s, θ₀'s and the trainable names' hashes; the optimizer section; and the
numerics setup against the launcher run's ``inputs.numerics_setup``. Bitwise agreement of rows
across processes is a diagnostic, not an acceptance condition.

**Verdicts** are functions of the records alone, in ``analysis.qualification_verdicts`` (its
docstring states the failure kinds; outside the GPU-code identity) and re-exported here:
``case_verdict``, ``control_verdict``, ``rate_verdict``, ``engineering_summary``,
``rest_from_trajectories``. Every constant a
part executes or records (``SHIFTS``, ``REST``, ``STEPS``, the kinds, ``record_sha256``) is
``qualification_constants``', inside the identity, and re-exported here too.

**Blind to outcomes.** Only K, D, norms and training statistics leave this module; every prompt
set whose name holds "harmful" is refused (``fa_control_diagnostic.guarded_prompt_sets``), and
nothing generates. Records carry their own policy identity and Pod (``qualification_provenance``)
and are published atomically and durably (``publish``).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from omegaconf import DictConfig, OmegaConf

from . import code_identity, prechecks, runtime
from . import fa_control_diagnostic as fcd
from . import schedule as schedule_module
from .analysis.qualification_verdicts import case_verdict as case_verdict
from .analysis.qualification_verdicts import control_verdict as control_verdict
from .analysis.qualification_verdicts import engineering_summary as engineering_summary
from .analysis.qualification_verdicts import rate_verdict as rate_verdict
from .analysis.qualification_verdicts import rest_from_trajectories as rest_from_trajectories
from .config import apply_override, load_config
from .optim import ADAM_MODES
from .qualification_constants import (
    ACCEPTED,
    BATCH,
    BRIDGE_KEYS,
    COMPARISONS,
    FAILING,
    FAMILIES,
    INCOMPLETE,
    NOT_QUALIFIED,
    PARTS,
    QUALIFIED,
    REST,
    SCIENTIFIC,
    SHIFTS,
    STEPS,
    TECHNICAL,
    UNSHIFTED,
    QualificationError,
    record_sha256,
)
from .runtime import Environment
from .schedule import INSTABILITY, InstabilityGuard, Schedule
from .tensor_hash import file_sha256, tensors_sha256
from .tracking import git_state, hardware, package_versions

__all__ = ["ACCEPTED", "BATCH", "COMPARISONS", "FAILING", "FAMILIES", "INCOMPLETE",
           "NOT_QUALIFIED", "QUALIFIED", "REST", "SCIENTIFIC", "SHIFTS", "STEPS", "TECHNICAL",
           "UNSHIFTED", "QualificationCase", "QualificationError", "QualificationScope",
           "bridge_checks", "case_verdict", "control_verdict", "engineering_summary",
           "harness_instability", "launcher_problems", "rate_verdict", "record_sha256",
           "run_part", "settings_record"]  # fmt: skip

BASELINE_CHECKPOINT = "harness-baseline.pt"
BASELINE_STATES = "baseline-states"  # the baseline's diagnostic states, A-step-<t>.pt
CHECKPOINT_KEYS = ("path", "sha256", "step", "names_sha256")
# "Config agreement": the launcher run's keys that must equal the confirmation config's
CONFIG_AGREEMENT = ("train", "optimizer", "execution", "data", "seed")


@dataclass(frozen=True)
class QualificationScope:
    """The registered scope (configs/experiments/qualification.yaml). ``norm`` and ``gaussian``
    are what ``fa_control_diagnostic.seed_directions`` reads."""

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
    def of(cls, settings: DictConfig) -> QualificationScope:
        if settings.get("experiment") != "qualification":
            raise QualificationError(f"Not the qualification's settings: "
                                     f"{settings.get('experiment')!r}")  # fmt: skip
        seeds = settings.seeds
        scope = cls(
            spec=str(settings.spec), bound_fraction=float(settings.bound_fraction),
            k_stars=tuple(sorted((int(c), float(k)) for c, k in settings.k_star.items())),
            norm=float(settings.norm),
            gaussian=tuple((str(k), int(v)) for k, v in settings.gaussian.items()),
            survival_share=float(settings.survival_share),
            rounding_scale=float(settings.rounding_scale),
            warning_floor=float(settings.warning.floor),
            warning_factor=float(settings.warning.factor),
            engineering_seeds=tuple(int(s) for s in seeds.engineering),
            numerical_seeds=tuple(sorted((int(c), tuple(int(s) for s in v))
                                         for c, v in seeds.numerical.items())),
            control_arm=str(settings.control.arm), control_steps=int(settings.control.steps),
            control_bound_fraction=float(settings.control.bound_fraction),
        )  # fmt: skip
        every = [*scope.engineering_seeds, *(s for _, v in scope.numerical_seeds for s in v)]
        if (
            [k for k, _ in scope.gaussian] != list(SHIFTS)[2:]
            or {c for c, _ in scope.k_stars} != {c for c, _ in scope.numerical_seeds}
            or len(set(every)) != len(every)
            or not all(fcd._number(k) and k > 0 for _, k in scope.k_stars)
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


@dataclass(frozen=True)
class QualificationCase:
    """One case: ``optimizer`` at ``lr`` on ``seed`` for ``candidate``, a numerical case or (with
    ``engineering``) one of the engineering stage, at the protocol's batch and steps."""

    seed: int
    optimizer: str
    lr: float
    candidate: int
    engineering: bool
    batch: int = BATCH
    steps: int = STEPS

    @property
    def name(self) -> str:
        return f"s{self.seed}/{self.optimizer}-{self.lr:g}"

    def attempt_path(self, part: str, attempt: int) -> str:
        """A part's attempt record, relative to the output directory."""
        stage = "engineering" if self.engineering else "numerical"
        return f"c{self.candidate}/{stage}/{self.name}/{part}-attempt-{attempt}.json"

    def trajectory_path(self, attempt: int, trajectory: str) -> str:
        """A rest trajectory's result, published as soon as the trajectory is done (module
        docstring, "Trajectory results"), relative to the output directory."""
        return self.attempt_path("rest", attempt).replace(".json", f".trajectory-{trajectory}.json")

    def record(self) -> dict[str, Any]:
        return {"name": self.name, **dataclasses.asdict(self)}


def reads_v0(optimizer: str) -> bool:
    """Whether an arm's run reads v₀ (FA and pre-warmed AdamW)."""
    return ADAM_MODES.get(optimizer, "adaptive") != "adaptive"


def names_sha256(names: Sequence[str]) -> str:
    """The trainable names' hash as the launcher's checkpoint record has it."""
    return hashlib.sha256(json.dumps(list(names)).encode()).hexdigest()


def publish(path: Path, record: Mapping[str, Any]) -> None:
    """Write a record atomically and durably (temporary file, fsync, ``os.replace``, then an
    fsync of its directory, so that the rename itself survives a crash)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    schedule_module.publish(path, dict(record))
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def settings_record(settings: DictConfig) -> dict[str, Any]:
    """The settings as a record holds them (JSON: the candidates' keys as strings)."""
    return json.loads(json.dumps(OmegaConf.to_container(settings, resolve=True)))


def qualification_provenance(settings: DictConfig, **extra: Any) -> dict[str, Any]:
    """A record's own policy identity: the registered spec and settings, the GPU code
    (``code_identity.gpu_code_identity``), the checkout, the hardware, the packages, the host
    memory and the Pod (``pod_id``). A value that cannot be read is recorded as its error: such a
    record then matches no expected policy."""

    def read(what: Callable[[], Any]) -> Any:
        try:
            return what()
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}

    return {"spec": str(settings.spec), "settings": settings_record(settings),
            "gpu_code": read(code_identity.gpu_code_identity), "code": read(git_state),
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "hardware": read(hardware), "packages": read(package_versions),
            "host_memory": read(fcd.host_record), "pod_id": pod_id(), **extra}  # fmt: skip


def pod_id() -> str | None:
    """This process's Pod (``RUNPOD_POD_ID``, as ``tracking`` records a run's), or None."""
    return os.environ.get("RUNPOD_POD_ID") or None


# ------------------------------------------------------------------------- the launcher run


def confirmation_config(case: QualificationCase) -> DictConfig:
    """The confirmation config of the case's arm, rate, seed, batch and steps
    (experiments/study3, role confirmation): what "Config agreement" holds the launcher run to."""
    overrides = ["study3.role=confirmation", f"optimizer={case.optimizer}",
                 f"optimizer.lr={case.lr!r}", f"seed={case.seed}",
                 f"train.batch_size={case.batch}", f"train.steps={case.steps}"]  # fmt: skip
    return load_config("experiments/study3", overrides)


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value


def _agreement_problems(cfg: DictConfig, case: QualificationCase) -> list[str]:
    """The keys of ``CONFIG_AGREEMENT`` in which the launcher run's config departs from the
    confirmation config (``confirmation_config``)."""
    try:
        reference = confirmation_config(case)
    except Exception as exc:  # no confirmation config to agree with: not a conforming run
        return [f"no confirmation config: {type(exc).__name__}: {str(exc)[:300]}"]
    problems = []
    for key in CONFIG_AGREEMENT:
        mine, theirs = (_plain(OmegaConf.select(c, key)) for c in (cfg, reference))
        if mine != theirs:
            problems.append(f"{key} departs from the confirmation config's")
    return problems


def code_environment_problems(run_dir: Path) -> list[str]:
    """The launcher run against this process's code and run environment, with study 2's checks
    (``fa_control_diagnostic.code_check``, ``environment_check``): the run was made from a clean
    checkout of a commit whose GPU-code identity is this checkout's (a clean one), and its
    recorded GPU, driver, CUDA, Python, packages and container image are this process's. A
    launcher run of an earlier GPU implementation or another environment is not the run the
    harness can bridge, however small its D. Git and ``nvidia-smi`` alone: no CUDA work."""
    try:
        here = git_state()
        tool = here.get("commit") if here.get("dirty") is False else None
        code = fcd.code_check([run_dir], tool)
        environment = fcd.environment_check([run_dir])
    except Exception as exc:  # unreadable: the run cannot be shown to conform
        return [f"its code and environment cannot be checked: {type(exc).__name__}: "
                f"{str(exc)[:300]}"]  # fmt: skip
    return ([f"code: {p}" for p in code["problems"]]
            + [f"environment: {p}" for p in environment["problems"]])  # fmt: skip


def agreement_problems(cfg: DictConfig, case: QualificationCase) -> list[str]:
    """The keys of ``CONFIG_AGREEMENT`` in which a launcher run's config departs from the case's
    confirmation config ("Config agreement")."""
    return _agreement_problems(cfg, case)


def _identity_problems(
    cfg: DictConfig, meta: Mapping[str, Any], case: QualificationCase, numerics: Any
) -> list[str]:
    expected = {
        "seed": (cfg.get("seed"), case.seed),
        "optimizer.name": (OmegaConf.select(cfg, "optimizer.name"), case.optimizer),
        "train.batch_size": (OmegaConf.select(cfg, "train.batch_size"), case.batch),
        "train.steps": (OmegaConf.select(cfg, "train.steps"), case.steps),
        "eval.k_only": (OmegaConf.select(cfg, "eval.k_only", default=None), True),
    }
    problems = [f"{key} is {have!r}, not {want!r}" for key, (have, want) in expected.items()
                if have != want]  # fmt: skip
    lr = OmegaConf.select(cfg, "optimizer.lr")
    if not (fcd._number(lr) and math.isclose(float(lr), case.lr, rel_tol=1e-9)):
        problems.append(f"optimizer.lr is {lr!r}, not {case.lr:g}")
    problems += _agreement_problems(cfg, case)
    inputs = meta.get("inputs") or {}
    if not isinstance(inputs.get("numerics_setup"), dict):
        problems.append("no numerics setup (inputs.numerics_setup)")
    elif numerics is not None and inputs["numerics_setup"] != numerics:
        problems.append(f"its numerics setup {inputs['numerics_setup']} is not this process's "
                        f"{numerics}")  # fmt: skip
    if not isinstance(inputs.get("theta0_trainable_sha256"), str):
        problems.append("no hash of θ₀'s trainable parameters (inputs.theta0_trainable_sha256)")
    if reads_v0(case.optimizer) and not isinstance(inputs.get("v0_sha256"), str):
        problems.append("no hash of v₀ (inputs.v0_sha256)")
    if meta.get("status") != "completed":
        problems.append(f"the run's status is {meta.get('status')!r}, not 'completed'")
    return problems


def _checkpoint_problems(meta: Mapping, summary: Mapping, case: QualificationCase) -> list[str]:
    record = summary.get("endpoint_checkpoint")
    if not isinstance(record, dict) or set(record) != set(CHECKPOINT_KEYS):
        return [f"no endpoint checkpoint record: {record!r}"]
    problems = []
    if (meta.get("inputs") or {}).get("endpoint_checkpoint") != record:
        problems.append("the checkpoint records of summary.json and meta.json differ")
    if record["step"] != case.steps:
        problems.append(f"the checkpoint is of step {record['step']!r}, not {case.steps}")
    path = Path(str(record["path"]))
    if not path.is_file():
        problems.append(f"the checkpoint {path} is missing")
    elif file_sha256(path) != record["sha256"]:
        problems.append(f"the checkpoint {path}'s sha256 is not the recorded one")
    return problems


def launcher_problems(
    run_dir: Path, case: QualificationCase, *, checkpoint: bool = True, numerics: Any = None,
    environment: bool = True,
) -> tuple[str | None, list[str]]:  # fmt: skip
    """The launcher run against the case (module docstring, "Order and state"): ``("technical",
    reasons)`` for a run that departs from the case (its config against the confirmation
    config's ``CONFIG_AGREEMENT`` keys; with ``numerics``, this process's
    ``runtime.configure_numerics`` record, its numerics setup; its GPU code and run environment
    against this process's, ``code_environment_problems``, unless ``environment`` is false) or
    is not a whole record, checked first;
    ``("scientific", [reason])`` for a conforming run stopped by an instability rule; technical
    again for any other stop, a missing step-25 evaluation, or (with ``checkpoint``) a missing
    checkpoint record or file, or one whose sha256 is not the recorded one; ``(None, [])`` when it
    conforms. Only the config, meta, summary and terminal records and the checkpoint are read."""
    try:
        cfg = fcd.run_config(run_dir)
        meta = json.loads((run_dir / "meta.json").read_text())
    except (OSError, ValueError) as exc:
        return TECHNICAL, [f"the launcher run's records are unreadable: "
                           f"{type(exc).__name__}: {exc}"]  # fmt: skip
    problems = _identity_problems(cfg, meta, case, numerics)
    if environment:  # against this process's code and environment (not a fact of the run's)
        problems += code_environment_problems(run_dir)
    terminal = fcd._read_object(run_dir / "terminal.json")
    summary = fcd._read_object(run_dir / "summary.json") or {}
    stop = (terminal or {}).get("stop")
    if not (isinstance(stop, dict) and isinstance(stop.get("reason"), str)
            and isinstance(terminal.get("steps"), int)):  # fmt: skip
        problems.append("its terminal record is missing or incomplete")
    elif summary.get("terminal") != terminal:
        problems.append("its summary does not hold its terminal record")
    if problems:
        return TECHNICAL, ["the launcher run does not conform: " + p for p in problems]
    reason, steps = stop["reason"], terminal["steps"]
    if reason in INSTABILITY:
        detail = f", {json.dumps(stop['detail'])}" if stop.get("detail") else ""
        return SCIENTIFIC, [f"instability: the launcher run stopped by {reason!r} after {steps} "
                            f"steps{detail}"]  # fmt: skip
    if reason != "steps" or steps != case.steps:
        problems.append(f"the launcher run stopped by {reason!r} after {steps} steps, not after "
                        f"its {case.steps}")  # fmt: skip
    elif case.steps not in (terminal.get("evaluations") or []):
        problems.append(f"the launcher run has no step-{case.steps} evaluation")
    elif checkpoint:
        problems += _checkpoint_problems(meta, summary, case)
    return (TECHNICAL, problems) if problems else (None, [])


def launcher_identity(run_dir: Path) -> dict[str, Any] | None:
    """How a record names the launcher run, whatever its state: ``fa_control_diagnostic.
    run_identity`` and its checkpoint record (None without one); None when it cannot be read."""
    try:
        summary = fcd._read_object(run_dir / "summary.json") or {}
        return {**fcd.run_identity(run_dir), "checkpoint": summary.get("endpoint_checkpoint")}
    except (OSError, ValueError):
        return None


def launcher_pod_id(run_dir: Path) -> str | None:
    """The Pod the launcher run ran on (``meta.json``: ``launch.pod_id``), or None."""
    meta = fcd._read_object(run_dir / "meta.json") or {}
    pod = (meta.get("launch") or {}).get("pod_id")
    return pod if isinstance(pod, str) and pod else None


def launcher_record(run_dir: Path) -> dict[str, Any]:
    """How a part record names its launcher run before anything is checked: ``identity``
    (``launcher_identity``) and ``pod_id`` (``launcher_pod_id``). Whatever reading them raises,
    the record keeps the run as missing (None) with ``identity_error``, and is still published."""
    try:
        return {"identity": launcher_identity(run_dir), "pod_id": launcher_pod_id(run_dir)}
    except Exception as exc:  # the record is written whatever the launcher run's state
        return {"identity": None, "pod_id": None,
                "identity_error": f"{type(exc).__name__}: {str(exc)[:500]}"}  # fmt: skip


def launcher_facts(run_dir: Path) -> dict[str, Any]:
    """What the bridge compares of the launcher run (``bridge_checks``) and how a record names it
    (``identity``: ``fa_control_diagnostic.run_identity``, and the checkpoint record)."""
    cfg = fcd.run_config(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    summary = fcd._read_object(run_dir / "summary.json") or {}
    inputs = meta.get("inputs") or {}
    streams = inputs.get("streams") or {}
    checkpoint = summary.get("endpoint_checkpoint")
    rows = [{k: v for k, v in r.items() if k != "time_s"}
            for r in fcd.read_jsonl(run_dir / "metrics.jsonl") if "loss" in r]  # fmt: skip
    return {"identity": launcher_identity(run_dir), "rows": rows,
            "training_stream": streams.get("training"),
            "preconditioner_stream": streams.get("preconditioner"),
            "v0_sha256": inputs.get("v0_sha256"),
            "theta0_trainable_sha256": inputs.get("theta0_trainable_sha256"),
            "names_sha256": (checkpoint or {}).get("names_sha256"),
            "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
            "numerics_setup": inputs.get("numerics_setup")}  # fmt: skip


def bridge_checks(
    launcher: Mapping[str, Any], harness_rows: Sequence[Mapping[str, Any]],
    setup: Mapping[str, Any], *, stopped: bool = False,
) -> dict[str, bool]:  # fmt: skip
    """The bridge's conditions, exactly (module docstring): the step indices, each row's
    ``tokens`` and ``denominator``, the training stream's identity, v₀'s hash (None on both sides
    for an arm that reads none), θ₀'s and the names' hashes, the optimizer section and the
    numerics setup. ``launcher`` is ``launcher_facts``; ``setup`` the harness's side
    (``_setup_facts``). The checks are exactly ``BRIDGE_KEYS``, in that order. A trajectory that
    ``stopped`` (a non-finite loss, gradient or weight:
    ``run_twin`` ends there) is held to the launcher run's rows of the same length: the steps,
    tokens and denominators of that prefix, exactly, and the stream chunks it read the launcher
    run's first ones."""
    mine, theirs = list(harness_rows), list(launcher.get("rows") or [])
    if stopped and len(mine) <= len(theirs):
        theirs = theirs[: len(mine)]
    steps = [r.get("step") for r in mine] == [r.get("step") for r in theirs] and bool(mine)

    def rows_equal(key: str) -> bool:
        return steps and all(key in a and key in b and a[key] == b[key]
                             for a, b in zip(mine, theirs, strict=True))  # fmt: skip

    def present_and_equal(key: str) -> bool:
        return launcher.get(key) is not None and launcher.get(key) == setup.get(key)

    def stream_prefix() -> bool:  # the stopped trajectory's chunks: the launcher's first ones
        ours, full = setup.get("training_stream"), launcher.get("training_stream")
        if not (isinstance(ours, Mapping) and isinstance(full, Mapping)):
            return False
        chunks, all_chunks = list(ours.get("chunks") or []), list(full.get("chunks") or [])
        return (ours.get("path") == full.get("path") and bool(chunks)
                and chunks == all_chunks[: len(chunks)]
                and (ours.get("count") or 0) <= (full.get("count") or 0))  # fmt: skip

    training = stream_prefix() if stopped else present_and_equal("training_stream")
    checks = {"steps": steps, "tokens": rows_equal("tokens"),
            "denominator": rows_equal("denominator"),
            "training_stream": training,
            "v0_sha256": launcher.get("v0_sha256") == setup.get("v0_sha256"),
            "theta0_trainable_sha256": present_and_equal("theta0_trainable_sha256"),
            "names_sha256": present_and_equal("names_sha256"),
            "optimizer": present_and_equal("optimizer"),
            "numerics_setup": present_and_equal("numerics_setup")}  # fmt: skip
    if set(checks) != set(BRIDGE_KEYS):  # the registered set, and no other
        raise QualificationError(f"The bridge's checks {sorted(checks)} are not the registered "
                                 f"{list(BRIDGE_KEYS)}")  # fmt: skip
    return {key: checks[key] for key in BRIDGE_KEYS}


# ------------------------------------------------------------------------------ the harness


def setup_config(cfg: DictConfig, control_arm: str) -> DictConfig:
    """The config ``twin_setup`` estimates v₀ and the residual under: the run's own for an arm
    that reads v₀; for another (AdamW), the run's with the F optimizer group (``control_arm``) at
    the run's rate, so that s1 and s2 are F-preconditioned at every arm. Its trajectories still
    run the run's own config."""
    if reads_v0(str(cfg.optimizer.name)):
        return cfg
    out = apply_override(copy.deepcopy(cfg), f"optimizer={control_arm}")
    out.optimizer.lr = cfg.optimizer.lr
    OmegaConf.resolve(out)
    return out


def _setup_facts(
    cfg: DictConfig, twins: fcd.Twins, shared: Mapping[str, Any], numerics_setup: Any, steps: int
) -> dict[str, Any]:
    """The harness's side of the bridge, for a trajectory of ``steps`` steps."""
    return {"training_stream": twins.prefix.identity(steps * int(cfg.train.batch_size)),
            "v0_sha256": shared["v0_sha256"] if reads_v0(str(cfg.optimizer.name)) else None,
            "theta0_trainable_sha256": tensors_sha256([(n, twins.theta0[n]) for n in twins.names]),
            "names_sha256": names_sha256(twins.names),
            "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
            "numerics_setup": numerics_setup}  # fmt: skip


def _v0_sha256(twins: fcd.Twins) -> str:
    return tensors_sha256(list(zip(twins.names, twins.v0, strict=True)))


def harness_instability(cfg: DictConfig, run: Mapping[str, Any]) -> dict[str, Any] | None:
    """The launcher's declared instability rules applied after the run to an unshifted harness
    trajectory (the baseline, the repeat; "The execution path": a declared instability stop of an
    unshifted harness trajectory), from its per-step rows and monitor readings, in the order the
    launcher checks them (``schedule.run_scheduled``): before each update, a non-finite loss or
    gradient and the gradient rule (``schedule.InstabilityGuard``: above ``grad_factor`` times
    the median of up to ``grad_window`` preceding norms, from the step-0 norm on, a firing norm
    kept out of the reference; under ``grad_rule: log`` recorded, not a stop); after the update,
    a non-finite weight (``run_twin``'s stop), and at each monitor reading (every
    ``eval.monitor.every`` steps) a non-finite K or one above ``k_max``. Returns the first rule
    that fires, ``{"reason", "step", "detail"}`` (the step whose update the launcher would have
    refused, or after which it would have stopped), or None."""
    rules = Schedule.from_config(cfg)
    guard = InstabilityGuard(rules.grad_factor, rules.grad_window, rules.k_max, rules.grad_rule)
    readings = {r.get("step"): r.get("K_monitor") for r in run.get("monitor") or []}
    for row in run.get("rows") or []:
        step = row.get("step")
        loss, norm = row.get("loss"), row.get("grad_norm")
        values = {"loss": loss, "grad_norm": norm}
        loss, norm = (v if fcd._number(v) else math.nan for v in (loss, norm))
        reason = guard.gradient(loss, norm)
        if reason is not None:
            return {"reason": reason, "step": step, "detail": guard.detail or values}
        if step + 1 in readings:
            k = readings[step + 1]
            reason = guard.monitor(k if fcd._number(k) else math.nan)
            if reason is not None:
                return {"reason": reason, "step": step + 1,
                        "detail": guard.detail or {"K_monitor": k}}  # fmt: skip
    if run.get("stop") is not None:  # the weights after an update (run_twin's other stop)
        return {"reason": "nonfinite_weight", "step": run.get("steps"),
                "detail": {"stop": run["stop"]}}  # fmt: skip
    return None


@torch.no_grad()
def initial_difference(twins: fcd.Twins, cfg: DictConfig) -> dict[str, Any]:
    """D(θ₀ + s, θ₀) on the scoring panel: the shifted student against the second model set back
    to θ₀ (only its trainable parameters ever change)."""
    fcd.set_trainable(twins.other, twins.theta0, twins.names)
    twins.other.to(twins.env.device)
    try:
        return fcd.twin_divergence(twins.other, twins.student, twins.scoring,
                                   pad_id=twins.env.pad_id,
                                   micro_batch_size=cfg.execution.micro_batch_size)  # fmt: skip
    finally:
        twins.other.to("cpu")


def run_trajectory(
    name: str,
    cfg: DictConfig,
    twins: fcd.Twins,
    shared: Mapping[str, Any],
    scope: QualificationScope,
    targets: Mapping[str, Path],
    *,
    diagnostic_steps: Sequence[int] = (),
    states: Path | None = None,
    monitor: Any = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """One trajectory of the rest part (``REST``), from θ₀ reset (``Twins.reset``: θ₀, no
    gradient; ``run_twin`` builds a fresh optimizer and seeds the RNG): ``repeat`` in fp32,
    ``reference`` by ``reference_twin`` (its fallback on out-of-memory alone), a shift after its
    perturbation of θ₀. With ``diagnostic_steps``, D against the baseline's ``states`` at those
    steps. The repeat, unshifted, reads ``monitor`` at the launcher's cadence and is held to the
    declared instability rules (``harness_instability``); the reference and the shifts are not
    (a non-finite value alone counts for them, as their stop). Returns the trajectory's record
    and, when it took every step, its D₂₅ against each of ``targets`` (label → checkpoint). The
    student is at θ₀ afterwards."""
    twins.reset()
    extra: dict[str, Any] = {}
    steps = list(diagnostic_steps)
    if name in SHIFTS:
        key = SHIFTS[name]
        if key not in shared["scales"]:  # its direction has no finite, positive norm
            twins.reset()
            return {"steps": 0, "stop": f"no shift: {name}'s direction has no finite, positive "
                    "norm", "diverged": False, "rows": [], "not_run": True}, {}  # fmt: skip
        extra["rounding"] = fcd.apply_perturbation(
            twins.trainable, twins.theta0, shared["directions"][key], shared["scales"][key],
            survival_share=scope.survival_share, rounding_scale=scope.rounding_scale,
        )  # fmt: skip
        extra["survived"] = fcd.survived(extra["rounding"], scope.survival_share)
        extra["initial_difference"] = initial_difference(twins, cfg)
    if name == "reference":
        run = fcd.reference_twin(cfg, twins, steps, states if states is not None else Path())
    elif name in SHIFTS:
        run = fcd.run_twin(cfg, twins, eval_at=steps, compare_dir=states if steps else None)
    elif name == "repeat":
        run = fcd.run_twin(cfg, twins, eval_at=steps, monitor=monitor,
                           compare_dir=states if steps else None)  # fmt: skip
        run["instability"] = harness_instability(cfg, run)
    else:
        raise QualificationError(f"No trajectory {name!r} in the rest part")
    run = {**run, **extra}
    endpoint = {}
    if run["stop"] is None:
        run["endpoint_sha256"] = tensors_sha256(twins.trainable)
        endpoint = {label: twins.divergence(path, cfg) for label, path in targets.items()}
    twins.reset()
    return run, endpoint


def _baseline(
    scope: QualificationScope, env: Environment, cfg: DictConfig, case: QualificationCase,
    run_dir: Path, facts: Mapping[str, Any], *, checkpoints: Path, steps: Sequence[int],
    numerics_setup: Any, progress: dict[str, Any],
) -> dict[str, Any]:  # fmt: skip
    trajectories = progress.setdefault("trajectories", {})
    comparisons = progress.setdefault("comparisons", {})
    setup_cfg = setup_config(cfg, scope.control_arm)
    k_star = scope.k_star(case.candidate)
    with fcd.twin_setup(env, setup_cfg, setup_cfg, k_star=k_star) as (twins, shared):
        v0_before = _v0_sha256(twins)
        twins.reset()
        states = checkpoints / BASELINE_STATES
        run = trajectories["baseline"] = fcd.run_twin(
            cfg, twins, eval_at=list(steps), monitor=shared["monitor"],
            save_dir=states if steps else None,
        )  # fmt: skip
        run["instability"] = harness_instability(cfg, run)
        # The stream the trajectory read: a stopped one, the batches of its rows
        setup = _setup_facts(cfg, twins, shared, numerics_setup, len(run["rows"]))
        # The bridge as soon as the trajectory is done: a failure after it keeps its checks, so
        # the verdict may read the trajectory (and reads nothing of a part without them)
        stopped = run["stop"] is not None
        progress.update(
            setup=setup, bridge=bridge_checks(facts, run["rows"], setup, stopped=stopped)
        )
        v0_stream = shared["v0_stream"] if reads_v0(case.optimizer) else None
        streams = {"training": setup["training_stream"], "preconditioner": v0_stream}
        archive = fcd.compare_archive(run["rows"], run["monitor"], run_dir, end=run,
                                      streams=streams)  # fmt: skip
        run.update(archive=archive, bitwise_rows=archive["bitwise_equal"])  # a diagnostic
        checkpoint = None
        if run["stop"] is None:
            path = checkpoints / BASELINE_CHECKPOINT
            fcd.save_trainable(twins.trainable, path)
            checkpoint = {"path": str(path), "sha256": file_sha256(path), "step": run["steps"],
                          "names_sha256": setup["names_sha256"]}  # fmt: skip
            run["endpoint_sha256"] = tensors_sha256(twins.trainable)
            comparisons["launcher~baseline"] = twins.divergence(
                Path(facts["identity"]["checkpoint"]["path"]), cfg)  # fmt: skip
        twins.reset()
        body = {"status": "complete", "checkpoint": checkpoint, "setup": setup,
                "bridge": progress["bridge"],
                "v0_sha256": shared["v0_sha256"], "v0_unchanged": _v0_sha256(twins) == v0_before,
                "trajectories": trajectories, "comparisons": comparisons}  # fmt: skip
    return body


def latest_record(out: Path, case: QualificationCase, part: str) -> tuple[Path | None, dict | None]:
    """The highest attempt of a case's part on disk, with its record (None when unreadable)."""
    found = sorted(out.glob(case.attempt_path(part, "*")),
                   key=lambda p: int(p.stem.rsplit("-", 1)[-1]))  # fmt: skip
    if not found:
        return None, None
    return found[-1], fcd._read_object(found[-1])


def _baseline_problems(
    baseline: Mapping[str, Any] | None, case: QualificationCase, facts: Mapping[str, Any],
    steps: Sequence[int],
) -> list[str]:  # fmt: skip
    if baseline is None:
        return ["no readable baseline record"]
    if baseline.get("part") != "baseline" or baseline.get("case") != case.record():
        return ["the baseline record is not this case's"]
    if baseline.get("status") != "complete" or not isinstance(baseline.get("checkpoint"), dict):
        return [f"the baseline part did not end with its checkpoint ({baseline.get('status')})"]
    problems = []
    if baseline.get("diagnostic_steps") != list(steps):
        problems.append(f"the baseline's diagnostic steps are {baseline.get('diagnostic_steps')}, "
                        f"not {list(steps)}")  # fmt: skip
    if baseline.get("launcher", {}).get("identity") != facts["identity"]:
        problems.append("the baseline was run against another launcher run or checkpoint")
    path = Path(baseline["checkpoint"]["path"])
    if not path.is_file():
        problems.append(f"the baseline checkpoint {path} is missing")
    elif file_sha256(path) != baseline["checkpoint"]["sha256"]:
        problems.append(f"the baseline checkpoint {path}'s sha256 is not the recorded one")
    return problems


def scientific_result(
    name: str, run: Mapping[str, Any], endpoint: Mapping[str, Any], bound: float
) -> str | None:
    """The rest part's stopping rule on a completed trajectory (module docstring, "Trajectory
    results"): the declared instability rules on the repeat (``harness_instability``, or its
    stop), a non-finite loss, gradient or model in the reference or a shift, or D₂₅ of the
    harness baseline against the reference or a shift beyond 0.01·K*; why, or None. It applies
    only once the bridge checks hold (``_rest``)."""
    if run.get("not_run"):
        return None
    if name == "repeat":
        fired = run.get("instability")
        if fired is not None:
            return (f"instability: the harness repeat: {fired.get('reason')!r} at step "
                    f"{fired.get('step')}")  # fmt: skip
        stop = run.get("stop")
        return None if stop is None else f"instability: the harness repeat stopped: {stop}"
    if run.get("stop") is not None:
        return f"a non-finite loss, gradient or model in the {name}: {run['stop']}"
    d = (endpoint.get("baseline") or {}).get("D_twin")
    if fcd._number(d) and d > bound:
        return f"baseline~{name}: D₂₅ {d:.6g} beyond the bound {bound:.6g}"
    return None


def _rest(
    scope: QualificationScope, env: Environment, cfg: DictConfig, case: QualificationCase,
    facts: Mapping[str, Any], out: Path, *, checkpoints: Path, steps: Sequence[int],
    numerics_setup: Any, progress: dict[str, Any],
    keep: Callable[[str, Mapping[str, Any]], None] = lambda name, result: None,
) -> dict[str, Any]:  # fmt: skip
    path, baseline = latest_record(out, case, "baseline")
    reference = {"path": None if path is None else str(path),
                 "sha256": None if path is None else file_sha256(path)}  # fmt: skip
    problems = _baseline_problems(baseline, case, facts, steps)
    if problems:
        return {"status": "not_run", "kind": TECHNICAL, "reasons": problems,
                "baseline_record": reference}  # fmt: skip
    head = {"baseline_record": reference,
            "baseline_checkpoint_sha256": baseline["checkpoint"]["sha256"]}  # fmt: skip
    progress.update(head)
    trajectories = progress.setdefault("trajectories", {})
    comparisons = progress.setdefault("comparisons", {})
    targets = {"launcher": Path(facts["identity"]["checkpoint"]["path"]),
               "baseline": Path(baseline["checkpoint"]["path"])}  # fmt: skip
    setup_cfg = setup_config(cfg, scope.control_arm)
    k_star = scope.k_star(case.candidate)
    with fcd.twin_setup(env, setup_cfg, setup_cfg, k_star=k_star) as (twins, shared):
        v0_before = _v0_sha256(twins)
        directions = fcd.seed_directions(scope, twins, shared, float(setup_cfg.optimizer.eps))
        progress["directions"] = directions
        decided = None
        for name in REST:
            trajectories[name], endpoint = run_trajectory(
                name, cfg, twins, shared, scope, targets, diagnostic_steps=steps,
                states=checkpoints / BASELINE_STATES, monitor=shared["monitor"],
            )  # fmt: skip
            for label, measured in endpoint.items():
                comparisons[f"{label}~{name}"] = measured
            if name == "repeat":  # the bridge, from the repeat's rows, before anything else
                setup = _setup_facts(cfg, twins, shared, numerics_setup,
                                     len(trajectories["repeat"]["rows"]))  # fmt: skip
                progress.update(
                    setup=setup,
                    bridge=bridge_checks(
                        facts,
                        trajectories["repeat"]["rows"],
                        setup,
                        stopped=trajectories["repeat"]["stop"] is not None,
                    ),
                )
            progress["v0_unchanged"] = _v0_sha256(twins) == v0_before
            bridge = progress.get("bridge")  # none before the repeat has run
            keep(name, {"run": trajectories[name],
                        "comparisons": {f"{label}~{name}": m for label, m in endpoint.items()},
                        "bridge": bridge, "v0_unchanged": progress["v0_unchanged"]})  # fmt: skip
            bridged = (bool(bridge) and all(v is True for v in bridge.values())
                       and progress["v0_unchanged"])  # fmt: skip
            decided = bridged and scientific_result(name, trajectories[name], endpoint,
                                                    scope.bound(case.candidate))  # fmt: skip
            if decided:  # a valid scientific failure: published at once, the part stops here
                break
        body = {"status": "decided" if decided else "complete", **head,
                "setup": progress["setup"], "directions": directions,
                "bridge": progress["bridge"], "v0_sha256": shared["v0_sha256"],
                "v0_unchanged": progress["v0_unchanged"],
                "reference_mode": (trajectories.get("reference") or {}).get("mode"),
                "fallback_reason": (trajectories.get("reference") or {}).get("fallback_reason"),
                "trajectories": trajectories, "comparisons": comparisons}  # fmt: skip
        if decided:
            body.update(kind=SCIENTIFIC, reasons=[decided])
    return body


def _control(
    scope: QualificationScope, env: Environment, cfg: DictConfig, case: QualificationCase,
    run_dir: Path, progress: dict[str, Any],
) -> dict[str, Any]:  # fmt: skip
    setup = fcd._control_setup(env, cfg)
    try:
        result = progress["control"] = fcd.same_path_run(
            env, cfg, setup["student"], setup["trainable"], setup["v0"],
            k_at=[scope.control_steps], bound=scope.control_bound(case.candidate),
            k_star=scope.k_star(case.candidate),
        )  # fmt: skip
        inputs = json.loads((run_dir / "meta.json").read_text()).get("inputs") or {}
        streams = inputs.get("streams") or {}
        shared_state = {
            "v0_stream": {"control": setup["v0_stream"], "launcher": streams.get("preconditioner")},
            "training_stream": {"control": result["training_stream"],
                                "launcher": streams.get("training")},
            "v0_sha256": {"control": setup["v0_sha256"], "launcher": inputs.get("v0_sha256")},
        }  # fmt: skip
        for check in shared_state.values():
            check["equal"] = check["launcher"] is not None and check["control"] == check["launcher"]
    finally:
        setup.clear()
    return {"status": "complete", "control": result, "shared_state": shared_state}


def run_part(
    settings: DictConfig,
    env: Environment,
    case: QualificationCase,
    run_dir: Path,
    out: Path,
    *,
    part: Literal["baseline", "rest", "control"],
    attempt: int,
    checkpoints: Path,
    diagnostic_steps: Sequence[int] = (),
    infrastructure: Mapping[str, Any] | None = None,
    engineering_cycle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One part of a case in this process (module docstring), written to
    ``out / case.attempt_path(part, attempt)`` whatever happened, in one publish, and returned.
    ``run_dir`` is the case's launcher run; ``checkpoints`` holds the harness's checkpoints
    (``harness-baseline.pt``, the baseline's diagnostic states). ``diagnostic_steps``
    (engineering cases, baseline and rest alone, steps before the last) are the steps of the
    intermediate D. ``infrastructure`` (an attempt above 1: the documented interruption it
    reruns) and ``engineering_cycle`` (a rerun after the candidate's engineering cycle: its
    supersession record's path and sha256) go into the record as given. Nothing runs when the
    launcher run does not conform (``launcher_problems``): the record says why, and of what
    kind. The control part's case is the candidate's first numerical seed at an FA rate."""
    scope = QualificationScope.of(settings)
    steps = sorted({int(t) for t in diagnostic_steps})
    if part not in PARTS or not (isinstance(attempt, int) and attempt >= 1):
        raise QualificationError(f"No part {part!r} or attempt {attempt!r}")
    if steps and (not case.engineering or part == "control"
                  or not all(1 <= t < case.steps for t in steps)):  # fmt: skip
        raise QualificationError(f"Diagnostic steps {steps} belong to an engineering case's "
                                 f"baseline and rest, before step {case.steps}")  # fmt: skip
    if part == "control" and (case.optimizer != scope.control_arm or case.engineering
                              or case.seed != scope.seeds(case.candidate)[0]
                              or case.steps != scope.control_steps):  # fmt: skip
        raise QualificationError(f"The same-path control is {scope.control_arm}'s, on the "
                                 f"first numerical seed of candidate {case.candidate}: "
                                 f"not {case}")  # fmt: skip
    started = time.perf_counter()
    record: dict[str, Any] = {"part": part, "attempt": attempt, "case": case.record(),
                              "K_star": scope.k_star(case.candidate),
                              "bound": scope.bound(case.candidate), "launcher_run": str(run_dir),
                              "diagnostic_steps": steps}  # fmt: skip
    if infrastructure is not None:
        record["infrastructure"] = dict(infrastructure)
    if engineering_cycle is not None:
        record["engineering_cycle"] = dict(engineering_cycle)
    # The launcher run's name, whatever its state, so that a verdict counts every record of it
    record["launcher"] = {"kind": None, "reasons": [], **launcher_record(run_dir)}
    progress: dict[str, Any] = {}
    failed = False
    try:
        cfg = fcd.run_config(run_dir)
        # First: no CUDA work before it, and the path check fingerprints its flags
        numerics_setup = record["numerics_setup"] = runtime.configure_numerics(cfg, env.device)
        kind, reasons = launcher_problems(run_dir, case, checkpoint=part != "control",
                                          numerics=numerics_setup)  # file reads only  # fmt: skip
        record["launcher"].update(kind=kind, reasons=reasons)
        if kind is None:
            prechecks.require_path_check(env.data_root, cfg, env.device)
        facts = launcher_facts(run_dir) if kind is None else None
        guarded = fcd.guarded_prompt_sets(env, "harmful", "The qualification harness")
        if kind is not None:
            record.update(status="not_run", kind=kind, reasons=reasons)
        elif part == "baseline":
            record.update(_baseline(scope, guarded, cfg, case, run_dir, facts,
                                    checkpoints=checkpoints, steps=steps,
                                    numerics_setup=numerics_setup, progress=progress))  # fmt: skip
        elif part == "rest":

            def keep(name: str, result: Mapping[str, Any]) -> None:
                """A trajectory's result, durable and bound to its inputs, before the next."""
                publish(out / case.trajectory_path(attempt, name), {
                    "part": "rest", "attempt": attempt, "case": case.record(),
                    "trajectory": name, "K_star": record["K_star"], "bound": record["bound"],
                    "launcher": record["launcher"], "numerics_setup": numerics_setup,
                    "baseline_record": progress.get("baseline_record"),
                    "baseline_checkpoint_sha256": progress.get("baseline_checkpoint_sha256"),
                    "launcher_checkpoint_sha256": facts["identity"]["checkpoint"]["sha256"],
                    **result, "provenance": qualification_provenance(settings)})  # fmt: skip

            record.update(_rest(scope, guarded, cfg, case, facts, out, checkpoints=checkpoints,
                                steps=steps, numerics_setup=numerics_setup,
                                progress=progress, keep=keep))  # fmt: skip
        else:
            record.update(_control(scope, guarded, cfg, case, run_dir, progress))
    except Exception as exc:  # the part's record: technical, with what was done before it
        error = fcd._error(exc)
        record.update(status="failed", kind=TECHNICAL, reasons=[error["reason"]], **error,
                      **progress)  # fmt: skip
        failed = True
    if failed:  # the exception, and the frames it held, are dropped by now
        fcd._release_device(env.device)
    record.update(seconds=time.perf_counter() - started, numerics=fcd.numerics(env.device),
                  provenance=qualification_provenance(settings))  # fmt: skip
    publish(out / case.attempt_path(part, attempt), record)
    return record
