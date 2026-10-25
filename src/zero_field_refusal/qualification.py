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
CUDA work and before ``prechecks.require_path_check``, and its record is kept. The launcher run is
checked next (``launcher_problems``): its identity first (seed, arm, rate, batch, steps,
``eval.k_only``, the numerics setup and the hashes of θ₀ and, for an arm that reads it, v₀; a
completed run with a whole terminal record); a run that departs from it is technical, whatever
stopped it. Then a conforming run stopped by an instability rule (``schedule.INSTABILITY``) is a
scientific failure; one stopped otherwise, without its step-25 evaluation, or without a checkpoint
whose file's sha256 is the recorded one, is technical; either way nothing runs. Every trajectory
starts from θ₀ with no gradient, a fresh optimizer and the RNG seeded as ``run_twin`` seeds it;
v₀ is never written (``v0_unchanged``). An AdamW case, whose run reads no v₀, sets the seed's F
v₀ up for s1 and s2 alone, from the F optimizer group (``setup_config``). An exception is the
part's record, ``failed``, with the trajectories done before it.

**The bridge** (``bridge_checks``, each false one technical): each training row's ``tokens`` and
``denominator`` and the step indices exactly equal to the launcher run's; the training stream's
identity; v₀'s, θ₀'s and the trainable names' hashes; the optimizer section; and the numerics
setup against the launcher run's ``inputs.numerics_setup``. Bitwise agreement of rows across
processes is a diagnostic, not an acceptance condition.

**Kinds** (``case_verdict``; the failure routes). Scientific: an instability stop of the launcher
run, of the baseline or of the repeat; a non-finite loss, gradient or model in the reference or a
shift; ``baseline~reference`` or ``baseline~sK`` beyond the bound. Technical: ``launcher~baseline``,
``launcher~repeat`` or ``baseline~repeat`` beyond the bound; a launcher comparison beyond the bound
whose baseline counterpart is not; a shift lost to fp32 rounding after its retry; a non-finite D
between finite models; a failed bridge check; a ``rest`` that does not build on the baseline
given; parts under other settings or GPU code; any exception or missing part. A case with both is
scientific (no rerun passes it), its technical reasons kept. States: ``failing`` (scientific),
``not_qualified`` (technical), ``qualified`` (all 13 within), else ``incomplete``. ``warning``
when ``baseline~repeat`` exceeds τ: a prompt for diagnosis only. ``rate_verdict`` counts only the
cases of the candidate's numerical seeds, arm and rate under the expected GPU code and settings:
``failing`` if one is scientific, else ``not_qualified`` if one is technical (or an FA rate's
same-path control is not accepted), else ``qualified`` when every prescribed seed has exactly one
qualified case, else ``incomplete``. ``engineering_summary``: e_eng, the path maxima, the margins.

**Blind to outcomes.** Only K, D, norms and training statistics leave this module; every prompt
set whose name holds "harmful" is refused (``fa_control_diagnostic.guarded_prompt_sets``), and
nothing generates. Records carry their own policy identity (``qualification_provenance``).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from omegaconf import DictConfig, OmegaConf

from . import code_identity, prechecks, runtime
from . import config as config_module
from . import fa_control_diagnostic as fcd
from .optim import ADAM_MODES
from .paths import CONFIG_ROOT
from .runtime import Environment
from .schedule import INSTABILITY, _publish
from .tensor_hash import file_sha256, tensors_sha256
from .tracking import git_state, hardware, package_versions

BATCH, STEPS = 20, 25  # "Protocol": B = 20 and exactly 25 updates
PARTS = ("baseline", "rest", "control")
SCIENTIFIC, TECHNICAL = "scientific", "technical"
QUALIFIED, FAILING, INCOMPLETE = "qualified", "failing", "incomplete"  # a case, a rate
NOT_QUALIFIED = "not_qualified"  # technical
ACCEPTED = "accepted"  # a same-path control
SHIFTS = {"s1": "p1", "s2": "p2", "s3": "s3", "s4": "s4"}  # a shift and its direction's key
REST = ("repeat", "reference", *SHIFTS)  # the rest part's trajectories, in order
COMPARISONS = ("launcher~baseline",  # the 13 comparisons at step 25
               *(f"{t}~{name}" for name in REST for t in ("launcher", "baseline")))  # fmt: skip
BASELINE_CHECKPOINT = "harness-baseline.pt"
BASELINE_STATES = "baseline-states"  # the baseline's diagnostic states, A-step-<t>.pt
CHECKPOINT_KEYS = ("path", "sha256", "step", "names_sha256")


class QualificationError(ValueError):
    """The qualification cannot run as registered (a scope or a request outside it)."""


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

    def record(self) -> dict[str, Any]:
        return {"name": self.name, **dataclasses.asdict(self)}


def reads_v0(optimizer: str) -> bool:
    """Whether an arm's run reads v₀ (FA and pre-warmed AdamW)."""
    return ADAM_MODES.get(optimizer, "adaptive") != "adaptive"


def names_sha256(names: Sequence[str]) -> str:
    """The trainable names' hash as the launcher's checkpoint record has it."""
    return hashlib.sha256(json.dumps(list(names)).encode()).hexdigest()


def record_sha256(record: Mapping[str, Any]) -> str:
    """The sha256 of a record file as ``publish`` writes it (so of the file read back, too)."""
    return hashlib.sha256((json.dumps(record, indent=1) + "\n").encode()).hexdigest()


def publish(path: Path, record: Mapping[str, Any]) -> None:
    """Write a record atomically and durably (temporary file, fsync, ``os.replace``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish(path, dict(record))


def settings_record(settings: DictConfig) -> dict[str, Any]:
    """The settings as a record holds them (JSON: the candidates' keys as strings)."""
    return json.loads(json.dumps(OmegaConf.to_container(settings, resolve=True)))


def qualification_provenance(settings: DictConfig, **extra: Any) -> dict[str, Any]:
    """A record's own policy identity: the registered spec and settings, the GPU code
    (``code_identity.gpu_code_identity``), the checkout, the hardware, the packages and the host
    memory. A value that cannot be read is recorded as its error: such a record then matches no
    expected policy."""

    def read(what: Callable[[], Any]) -> Any:
        try:
            return what()
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}

    return {"spec": str(settings.spec), "settings": settings_record(settings),
            "gpu_code": read(code_identity.gpu_code_identity), "code": read(git_state),
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "hardware": read(hardware), "packages": read(package_versions),
            "host_memory": read(fcd.host_record), **extra}  # fmt: skip


# ------------------------------------------------------------------------- the launcher run


def _identity_problems(cfg: DictConfig, meta: Mapping[str, Any], case: QualificationCase) -> list:
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
    inputs = meta.get("inputs") or {}
    if not isinstance(inputs.get("numerics_setup"), dict):
        problems.append("no numerics setup (inputs.numerics_setup)")
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
    run_dir: Path, case: QualificationCase, *, checkpoint: bool = True
) -> tuple[str | None, list[str]]:
    """The launcher run against the case (module docstring, "Order and state"): ``("technical",
    reasons)`` for a run that departs from the case or is not a whole record, checked first;
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
    problems = _identity_problems(cfg, meta, case)
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
    return {"identity": {**fcd.run_identity(run_dir), "checkpoint": checkpoint}, "rows": rows,
            "training_stream": streams.get("training"),
            "preconditioner_stream": streams.get("preconditioner"),
            "v0_sha256": inputs.get("v0_sha256"),
            "theta0_trainable_sha256": inputs.get("theta0_trainable_sha256"),
            "names_sha256": (checkpoint or {}).get("names_sha256"),
            "optimizer": OmegaConf.to_container(cfg.optimizer, resolve=True),
            "numerics_setup": inputs.get("numerics_setup")}  # fmt: skip


def bridge_checks(
    launcher: Mapping[str, Any], harness_rows: Sequence[Mapping[str, Any]], setup: Mapping[str, Any]
) -> dict[str, bool]:
    """The bridge's conditions, exactly (module docstring): the step indices, each row's
    ``tokens`` and ``denominator``, the training stream's identity, v₀'s hash (None on both sides
    for an arm that reads none), θ₀'s and the names' hashes, the optimizer section and the
    numerics setup. ``launcher`` is ``launcher_facts``; ``setup`` the harness's side
    (``_setup_facts``)."""
    mine, theirs = list(harness_rows), list(launcher.get("rows") or [])
    steps = [r.get("step") for r in mine] == [r.get("step") for r in theirs] and bool(mine)

    def rows_equal(key: str) -> bool:
        return steps and all(key in a and key in b and a[key] == b[key]
                             for a, b in zip(mine, theirs, strict=True))  # fmt: skip

    def present_and_equal(key: str) -> bool:
        return launcher.get(key) is not None and launcher.get(key) == setup.get(key)

    return {"steps": steps, "tokens": rows_equal("tokens"),
            "denominator": rows_equal("denominator"),
            "training_stream": present_and_equal("training_stream"),
            "v0_sha256": launcher.get("v0_sha256") == setup.get("v0_sha256"),
            "theta0_trainable_sha256": present_and_equal("theta0_trainable_sha256"),
            "names_sha256": present_and_equal("names_sha256"),
            "optimizer": present_and_equal("optimizer"),
            "numerics_setup": present_and_equal("numerics_setup")}  # fmt: skip


# ------------------------------------------------------------------------------ the harness


def setup_config(cfg: DictConfig, control_arm: str) -> DictConfig:
    """The config ``twin_setup`` estimates v₀ and the residual under: the run's own for an arm
    that reads v₀; for another (AdamW), the run's with the F optimizer group (``control_arm``) at
    the run's rate, so that s1 and s2 are F-preconditioned at every arm. Its trajectories still
    run the run's own config."""
    if reads_v0(str(cfg.optimizer.name)):
        return cfg
    out = config_module._apply_override(copy.deepcopy(cfg), f"optimizer={control_arm}", CONFIG_ROOT)
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
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """One trajectory of the rest part (``REST``), from θ₀ reset (``Twins.reset``: θ₀, no
    gradient; ``run_twin`` builds a fresh optimizer and seeds the RNG): ``repeat`` in fp32,
    ``reference`` by ``reference_twin`` (its fallback on out-of-memory alone), a shift after its
    perturbation of θ₀. With ``diagnostic_steps``, D against the baseline's ``states`` at those
    steps. Returns the trajectory's record and, when it took every step, its D₂₅ against each of
    ``targets`` (label → checkpoint). The student is at θ₀ afterwards."""
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
    elif name in ("repeat", *SHIFTS):
        run = fcd.run_twin(cfg, twins, eval_at=steps, compare_dir=states if steps else None)
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
        setup = _setup_facts(cfg, twins, shared, numerics_setup, run["steps"])
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
                "bridge": bridge_checks(facts, run["rows"], setup),
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


def _rest(
    scope: QualificationScope, env: Environment, cfg: DictConfig, case: QualificationCase,
    facts: Mapping[str, Any], out: Path, *, checkpoints: Path, steps: Sequence[int],
    numerics_setup: Any, progress: dict[str, Any],
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
        for name in REST:
            trajectories[name], endpoint = run_trajectory(
                name, cfg, twins, shared, scope, targets, diagnostic_steps=steps,
                states=checkpoints / BASELINE_STATES,
            )  # fmt: skip
            for label, measured in endpoint.items():
                comparisons[f"{label}~{name}"] = measured
        repeat = trajectories["repeat"]
        setup = _setup_facts(cfg, twins, shared, numerics_setup, repeat["steps"])
        body = {"status": "complete", **head, "setup": setup, "directions": directions,
                "bridge": bridge_checks(facts, repeat["rows"], setup),
                "v0_sha256": shared["v0_sha256"], "v0_unchanged": _v0_sha256(twins) == v0_before,
                "reference_mode": trajectories["reference"].get("mode"),
                "fallback_reason": trajectories["reference"].get("fallback_reason"),
                "trajectories": trajectories, "comparisons": comparisons}  # fmt: skip
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
) -> dict[str, Any]:
    """One part of a case in this process (module docstring), written to
    ``out / case.attempt_path(part, attempt)`` whatever happened, and returned. ``run_dir`` is the
    case's launcher run; ``checkpoints`` holds the harness's checkpoints (``harness-baseline.pt``,
    the baseline's diagnostic states). ``diagnostic_steps`` (engineering cases, baseline and rest
    alone, steps before the last) are the steps of the intermediate D. Nothing runs when the
    launcher run does not conform (``launcher_problems``): the record says why, and of what kind.
    The control part's case is the candidate's first numerical seed at an FA rate."""
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
    progress: dict[str, Any] = {}
    failed = False
    try:
        cfg = fcd.run_config(run_dir)
        # First: no CUDA work before it, and the path check fingerprints its flags
        numerics_setup = record["numerics_setup"] = runtime.configure_numerics(cfg, env.device)
        prechecks.require_path_check(env.data_root, cfg, env.device)
        kind, reasons = launcher_problems(run_dir, case, checkpoint=part != "control")
        facts = launcher_facts(run_dir) if kind is None else None
        record["launcher"] = {"kind": kind, "reasons": reasons,
                              "identity": None if facts is None else facts["identity"]}  # fmt: skip
        guarded = fcd.guarded_prompt_sets(env, "harmful", "The qualification harness")
        if kind is not None:
            record.update(status="not_run", kind=kind, reasons=reasons)
        elif part == "baseline":
            record.update(_baseline(scope, guarded, cfg, case, run_dir, facts,
                                    checkpoints=checkpoints, steps=steps,
                                    numerics_setup=numerics_setup, progress=progress))  # fmt: skip
        elif part == "rest":
            record.update(_rest(scope, guarded, cfg, case, facts, out, checkpoints=checkpoints,
                                steps=steps, numerics_setup=numerics_setup,
                                progress=progress))  # fmt: skip
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


# ------------------------------------------------------------------------------- verdicts


def _within(d: Any, bound: float | None) -> bool | None:
    return d <= bound if fcd._number(d) and bound is not None else None


def _policy(record: Mapping[str, Any]) -> tuple[Any, Any]:
    """A part record's settings and GPU code (its sha256)."""
    provenance = record.get("provenance") or {}
    return provenance.get("settings"), (provenance.get("gpu_code") or {}).get("sha256")


def _part_reasons(record: Mapping[str, Any], technical: list[str], science: list[str]) -> None:
    """A part that ran nothing or failed: its kind and reasons."""
    reasons = [f"{record.get('part')}: {r}" for r in record.get("reasons") or []]
    (science if record.get("kind") == SCIENTIFIC else technical).extend(reasons)


def case_verdict(
    baseline: Mapping[str, Any] | None,
    rest: Mapping[str, Any] | None,
    scope: QualificationScope,
    *,
    e_eng: float | None = None,
) -> dict[str, Any]:
    """A case's 13 comparisons at step 25 (``COMPARISONS``, each ``{"D", "within"}``), its state,
    kind and reasons, by the registration's rules (module docstring, "Kinds"), from its baseline
    and rest records (None: no record). With ``e_eng``, ``warning`` when ``baseline~repeat``
    exceeds τ (``QualificationScope.warning``). The case's identity, launcher run, GPU code and
    settings are kept for ``rate_verdict``."""
    science: list[str] = []
    technical: list[str] = []
    measured: dict[str, Any] = dict.fromkeys(COMPARISONS)
    head = baseline or rest or {}
    case = head.get("case") or {}
    try:
        bound = scope.bound(case.get("candidate"))
    except QualificationError as exc:
        bound = None
        technical.append(str(exc))
    ended = False  # the baseline decided the case: the rest is not needed
    if baseline is None:
        technical.append("the baseline part has no record")
    elif baseline.get("part") != "baseline":
        technical.append("the baseline record is another part's")
    elif baseline.get("status") != "complete":  # a failed baseline's trajectory still counts
        _part_reasons(baseline, technical, science)
        ended = True
    run = ((baseline or {}).get("trajectories") or {}).get("baseline") or {}
    if (baseline or {}).get("part") == "baseline" and run.get("stop") is not None:
        science.append(f"instability: the harness baseline stopped: {run['stop']}")
        ended = True
    if (baseline or {}).get("part") == "baseline" and baseline.get("status") == "complete":
        failed = [k for k, v in (baseline.get("bridge") or {}).items() if v is not True]
        if failed or not baseline.get("bridge"):
            technical.append(f"the bridge failed: {failed or 'no checks'}")
        if baseline.get("v0_unchanged") is not True:
            technical.append("v₀ changed during the baseline part")
        measured["launcher~baseline"] = ((baseline.get("comparisons") or {})
                                         .get("launcher~baseline") or {}).get("D_twin")  # fmt: skip
    if not ended:
        if rest is None:
            technical.append("the rest part has no record")
        else:
            _rest_verdict(baseline, rest, measured, technical, science)
    comparisons = {name: {"D": measured[name], "within": _within(measured[name], bound)}
                   for name in COMPARISONS}  # fmt: skip
    for name, c in comparisons.items():
        target, trajectory = name.split("~")
        if c["D"] is not None and not fcd._number(c["D"]):
            technical.append(f"{name}: a non-finite D between finite models ({c['D']})")
        elif c["within"] is not False:
            continue
        elif trajectory in ("baseline", "repeat"):
            technical.append(f"{name}: D₂₅ {c['D']:.6g} beyond the bound {bound:.6g}")
        elif target == "baseline":
            science.append(f"{name}: D₂₅ {c['D']:.6g} beyond the bound {bound:.6g}")
        elif comparisons[f"baseline~{trajectory}"]["within"] is not False:
            technical.append(f"{name}: D₂₅ {c['D']:.6g} beyond the bound {bound:.6g}, its "
                             "baseline counterpart not")  # fmt: skip
    complete = all(c["within"] is not None for c in comparisons.values())
    kind = SCIENTIFIC if science else TECHNICAL if technical else None
    within = all(c["within"] for c in comparisons.values())
    state = (FAILING if science else NOT_QUALIFIED if technical
             else QUALIFIED if within else INCOMPLETE)  # fmt: skip
    tau = warning = None
    repeat = comparisons["baseline~repeat"]["D"]
    if e_eng is not None and bound is not None:
        tau = scope.warning(e_eng, case["candidate"])
        warning = repeat > tau if fcd._number(repeat) else None
    settings, code = _policy(head)
    reference = ((rest or {}).get("trajectories") or {}).get("reference") or {}
    return {"case": case, "state": state, "kind": kind, "reasons": science + technical,
            "complete": complete, "comparisons": comparisons, "bound": bound, "tau": tau,
            "warning": warning, "reference_mode": reference.get("mode"),
            "fallback": reference.get("mode") == fcd.FLOAT64_ACCUMULATION,
            "launcher": (head.get("launcher") or {}).get("identity"), "gpu_code": code,
            "settings": settings}  # fmt: skip


def _rest_verdict(
    baseline: Mapping[str, Any] | None, rest: Mapping[str, Any], measured: dict[str, Any],
    technical: list[str], science: list[str],
) -> None:  # fmt: skip
    """The rest record's part of ``case_verdict``."""
    if rest.get("part") != "rest":
        technical.append("the rest record is another part's")
        return
    base = baseline or {}
    if rest.get("case") != base.get("case"):
        technical.append("the rest record is another case's")
        return
    if (rest.get("launcher") or {}).get("identity") != (base.get("launcher") or {}).get("identity"):
        technical.append("the rest part ran against another launcher run or checkpoint")
    if _policy(rest) != _policy(base):
        technical.append("the parts were written under other settings or GPU code")
    if rest.get("status") == "not_run":
        _part_reasons(rest, technical, science)
        return
    if (rest.get("baseline_record") or {}).get("sha256") != record_sha256(base) or rest.get(
        "baseline_checkpoint_sha256"
    ) != (base.get("checkpoint") or {}).get("sha256"):
        technical.append("the rest part does not build on this baseline record and checkpoint")
    if rest.get("status") != "complete":  # its trajectories before the failure still count
        _part_reasons(rest, technical, science)
    else:
        failed = [k for k, v in (rest.get("bridge") or {}).items() if v is not True]
        if failed or not rest.get("bridge"):
            technical.append(f"the bridge failed in the rest part: {failed or 'no checks'}")
        if rest.get("v0_unchanged") is not True:
            technical.append("v₀ changed during the rest part")
    trajectories = rest.get("trajectories") or {}
    for name in REST:
        run = trajectories.get(name)
        if run is None:
            if rest.get("status") == "complete":
                technical.append(f"the rest part has no {name}")
            continue
        if run.get("not_run"):
            technical.append(f"{name}: {run.get('stop')}")
        elif run.get("stop") is not None:
            what = "instability: the harness repeat stopped" if name == "repeat" else (
                f"a non-finite loss, gradient or model in the {name}")  # fmt: skip
            science.append(f"{what}: {run['stop']}")
        if name in SHIFTS and "survived" in run and run["survived"] is not True:
            technical.append(f"{name} was lost to fp32 rounding after its retry")
    for name, value in (rest.get("comparisons") or {}).items():
        if name in measured and name != "launcher~baseline":
            measured[name] = (value or {}).get("D_twin")


def control_verdict(
    record: Mapping[str, Any] | None, scope: QualificationScope, candidate: int
) -> dict[str, Any]:
    """Amendment 2: the same-path control is ``accepted`` when its records are finite (no
    divergence, every loss, gradient norm and K finite) and its K_scoring at step 25 is at most
    0.01·K*; a missing or failed control (exceeding, non-finite, incomplete, its streams or v₀ not
    the launcher run's, an error) is ``not_qualified`` (technical); one whose launcher run an
    instability rule stopped is ``failing`` (scientific)."""
    bound = scope.control_bound(candidate)
    out: dict[str, Any] = {"state": NOT_QUALIFIED, "kind": TECHNICAL, "reasons": [],
                           "K_scoring": None, "bound": bound, "candidate": candidate,
                           "case": None if record is None else record.get("case")}  # fmt: skip
    reasons = out["reasons"]
    case = (record or {}).get("case") or {}
    if record is None:
        reasons.append("no same-path control record")
    elif record.get("part") != "control" or case.get("candidate") != candidate:
        reasons.append("the record is not this candidate's same-path control")
    elif record.get("status") != "complete":
        reasons += record.get("reasons") or [f"the control did not run ({record.get('status')})"]
        if record.get("kind") == SCIENTIFIC:
            out.update(state=FAILING, kind=SCIENTIFIC)
    else:
        control = record.get("control") or {}
        state = fcd.control_state(control, k_at=[scope.control_steps], bound=bound)
        k = {r.get("step"): r.get("K_scoring") for r in control.get("scoring") or []}.get(
            scope.control_steps
        )
        out["K_scoring"] = k
        unequal = [key for key, c in (record.get("shared_state") or {}).items() if not c["equal"]]
        if unequal or not record.get("shared_state"):
            reasons.append(f"the control's {unequal or 'shared state'} differ from the "
                           "launcher run's")  # fmt: skip
        if state["diverged"]:
            reasons.append("a non-finite record in the control")
        elif not state["complete"] or not fcd._number(k):
            reasons.append(f"the control is incomplete: {control.get('stop')}")
        elif k > bound:
            reasons.append(f"K_scoring {k:.6g} at step {scope.control_steps} beyond {bound:.6g}")
        if not reasons:
            out.update(state=ACCEPTED, kind=None)
    return out


def _not_counted(
    verdict: Mapping[str, Any], *, candidate: int, optimizer: str, lr: float, seeds: Sequence[int],
    code: Mapping[str, Any] | None, policy: Mapping[str, Any] | None,
    launchers: Mapping[int, Mapping[str, Any]] | None,
) -> str | None:  # fmt: skip
    """Why a case verdict is not one of this rate's cases under the expected policy, or None."""
    case = verdict.get("case") or {}
    lr_ = case.get("lr")
    if case.get("candidate") != candidate or case.get("optimizer") != optimizer:
        return "another candidate or arm"
    if not (fcd._number(lr_) and math.isclose(lr_, lr, rel_tol=1e-9)):
        return "another rate"
    if case.get("engineering") is not False or case.get("seed") not in seeds:
        return "not a numerical seed of the candidate"
    if case.get("batch") != BATCH or case.get("steps") != STEPS:
        return "not the protocol's batch and steps"
    if code is None or verdict.get("gpu_code") != code.get("sha256"):
        return "other GPU code"
    if policy is None or verdict.get("settings") != policy:
        return "other settings"
    launcher = verdict.get("launcher") or {}
    if not (launcher.get("run_id") and (launcher.get("checkpoint") or {}).get("sha256")):
        return "no launcher run identity or checkpoint"
    if launchers is not None and launchers.get(case.get("seed")) != launcher:
        return "another launcher run or checkpoint"
    return None


def rate_verdict(
    cases: Sequence[Mapping[str, Any]],
    *,
    scope: QualificationScope,
    candidate: int,
    optimizer: str,
    lr: float,
    code: Mapping[str, Any] | None,
    policy: Mapping[str, Any] | None,
    control: Mapping[str, Any] | None = None,
    launchers: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """An (arm, rate)'s verdict from its case verdicts (``case_verdict``) and, for an FA rate, its
    same-path control's (``control_verdict``; None: missing). A case counts only when it is this
    candidate's, arm's and rate's, on one of its numerical seeds at the protocol's batch and
    steps, under the expected GPU code (``code``: its ``sha256``) and settings (``policy``:
    ``settings_record``), with a launcher run and checkpoint (equal to ``launchers[seed]`` when
    given). ``failing`` if a counted case is scientific (or the control's launcher run stopped by
    instability); else ``not_qualified`` if one is technical or an FA rate's control is not
    accepted; else ``qualified`` when the counted cases are exactly the prescribed seeds, one
    each, every one qualified with its 13 comparisons; else ``incomplete`` (an empty or partial
    list included)."""
    seeds = list(scope.seeds(candidate))
    counted, reasons = [], []
    for verdict in cases:
        why = _not_counted(verdict, candidate=candidate, optimizer=optimizer, lr=lr, seeds=seeds,
                           code=code, policy=policy, launchers=launchers)  # fmt: skip
        if why is None:
            counted.append(verdict)
        else:
            reasons.append(f"{(verdict.get('case') or {}).get('name')}: not counted: {why}")
    for verdict in counted:
        reasons += [f"{verdict['case']['name']}: {r}" for r in verdict.get("reasons") or []]
    found = [verdict["case"]["seed"] for verdict in counted]
    kinds = {verdict.get("kind") for verdict in counted}
    checked = None
    if optimizer == scope.control_arm:
        checked = dict(control or control_verdict(None, scope, candidate))
        rate = (checked.get("case") or {}).get("lr")
        if checked.get("state") == ACCEPTED and not (
            checked.get("candidate") == candidate and fcd._number(rate)
            and math.isclose(rate, lr, rel_tol=1e-9)
            and (checked.get("case") or {}).get("seed") == seeds[0]
        ):  # fmt: skip
            checked.update(state=NOT_QUALIFIED, kind=TECHNICAL,
                           reasons=["the control is not this rate's"])  # fmt: skip
        reasons += [f"control: {r}" for r in checked.get("reasons") or []]
    missing = [s for s in seeds if s not in found]
    duplicated = sorted({s for s in found if found.count(s) > 1})
    if SCIENTIFIC in kinds or (checked is not None and checked.get("kind") == SCIENTIFIC):
        state = FAILING
    elif TECHNICAL in kinds or (checked is not None and checked.get("state") != ACCEPTED):
        state = NOT_QUALIFIED
    elif (not missing and not duplicated and len(found) == len(seeds)
          and all(v.get("state") == QUALIFIED and v.get("complete") for v in counted)):  # fmt: skip
        state = QUALIFIED
    else:
        state = INCOMPLETE
        reasons += [f"no counted case of seed {s}" for s in missing]
        reasons += [f"more than one case of seed {s}" for s in duplicated]
    return {"candidate": candidate, "optimizer": optimizer, "lr": lr, "state": state,
            "seeds": seeds, "counted": sorted(found), "reasons": reasons, "control": checked,
            "warnings": [v["case"]["name"] for v in counted if v.get("warning")],
            "fallbacks": [v["case"]["name"] for v in counted if v.get("fallback")]}  # fmt: skip


FAMILIES = ("launcher~baseline", "launcher~repeat", "baseline~repeat", "launcher~reference",
            "baseline~reference", "launcher~shift", "baseline~shift")  # fmt: skip


def engineering_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The engineering stage's report from part records (baseline and rest): e_eng, the largest
    D₂₅/K* over the unshifted repeats (``baseline~repeat``); the path maxima, the largest
    D_t/K* at the diagnostic steps per trajectory (shifts pooled); and per comparison family the
    margin, the smallest bound/D₂₅ over the records (∞ for a D of 0). Diagnostics: no verdict."""
    repeats: list[float] = []
    paths: dict[str, list[float]] = {}
    ratios: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for record in records:
        k_star, bound = record.get("K_star"), record.get("bound")
        if not (fcd._number(k_star) and fcd._number(bound)):
            continue
        for name, value in (record.get("comparisons") or {}).items():
            d = (value or {}).get("D_twin")
            target, trajectory = name.split("~")
            family = f"{target}~{'shift' if trajectory in SHIFTS else trajectory}"
            if family not in ratios or d is None:
                continue
            ratios[family].append(bound / d if fcd._number(d) and d > 0 else
                                  math.inf if d == 0 else 0.0)  # fmt: skip
            if name == "baseline~repeat":
                repeats.append(d / k_star if fcd._number(d) else math.inf)
        if record.get("part") == "rest":
            for name, run in (record.get("trajectories") or {}).items():
                values = [r.get("D_twin") for r in run.get("D_twin") or []]
                if values:
                    family = "shift" if name in SHIFTS else name
                    paths.setdefault(family, []).append(fcd._largest(values) / k_star)
    return {"e_eng": max(repeats) if repeats else None, "repeats": len(repeats),
            "path_maxima": {family: max(v) for family, v in paths.items()},
            "margins": {family: min(v) if v else None for family, v in ratios.items()}}  # fmt: skip
