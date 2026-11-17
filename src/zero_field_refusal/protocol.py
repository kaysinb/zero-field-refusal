"""Run configs checked against the zero-field protocol at load time.

A setting that would make the expected gradient at θ₀ non-zero, or would make runs incomparable,
still yields a plausible-looking run. Entry points therefore load configs through
``load_run_config``, which rejects every departure from the protocol unless the declared
``control`` explains it. Each control (``configs/control/``) permits exactly its own deviation.
The rules mirror docs/architecture.md ("Zero-field invariants") and docs/preregistration.md.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from .config import load_config
from .paths import CONFIG_ROOT

ZERO_FIELD_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
# The sharpened-sampling control of docs/design.md ("Controls").
SHARPENED_SAMPLING = {**ZERO_FIELD_SAMPLING, "temperature": 0.7, "top_p": 0.9}
CONTROLS = (
    None,
    "token_count_normalization",
    "sharpened_sampling",
    "alpaca_reference",
    "soft_targets",
)
FROZEN_MODULES = ("input_embeddings", "output_head")
# Study-2 run schedules (schedule.Schedule).
SCHEDULE_POLICIES = ("k_stopped", "fixed")
# curvature: the Stage-A curvature measurement of study 2 (scripts/curvature.py).
RUN_TYPES = ("train", "theta0", "checks", "curvature")
# fixed: one stream of a run's exact length (study 1); appendable: chunks read in order (study 2).
STREAM_LAYOUTS = ("fixed", "appendable")
# The preregistered bound on N·|f|²/σ² of the path field (docs/preregistration.md).
PATH_FIELD_BOUND = 0.01
# One-sided confidence of the upper bound on the path discrepancy's χ² checked against it.
PATH_FIELD_LEVEL = 0.975
ATTENTION = ("sdpa", "eager")
# Study 3's roles whose runs may save the step-25 checkpoint (tracking.endpoint_checkpoint)
STUDY3_CHECKPOINT_ROLES = ("numerical", "engineering")
# Study 5's roles (study5.role): the fresh seeds' F runs and the crossed controls; both may save
# the endpoint checkpoint. The keys only the controls carry:
STUDY5_ROLES = ("fresh", "control", "dose", "order_a", "beta0")
STUDY5_KEYS = ("train.order_seed", "optimizer.preconditioner.source_seed", "train.order_edit")
RATE_SCHEDULE = "train.rate_schedule"  # the dose role's key (study5_options)
_COMMIT = re.compile(r"[0-9a-f]{40}")

REQUIRED = (
    "control",
    "teacher.source",
    "teacher.prompt_assignment",
    "teacher.responses",
    *(f"teacher.sampling.{key}" for key in ZERO_FIELD_SAMPLING),
    "teacher.sampling.max_new_tokens",
    "train.steps",
    "train.batch_size",
    "train.max_response_tokens",
    "train.loss.kind",
    "train.loss.normalization",
    "train.loss.include_end_token",
    "train.loss.denominator",
    "train.freeze",
    "optimizer.name",
    "optimizer.weight_decay",
    "model.revision",
    "model.dtype",
    "model.attn_implementation",
    "model.tf32",
    "model.end_tokens",
)

_ABSENT = object()


class ProtocolError(ValueError):
    """A run config departs from the zero-field protocol."""


def load_run_config(
    path: str | Path, overrides: Sequence[str] = (), *, root: Path = CONFIG_ROOT
) -> DictConfig:
    """``config.load_config`` followed by ``check_run_config``."""
    cfg = load_config(path, overrides, root=root)
    check_run_config(cfg)
    return cfg


def check_run_config(cfg: DictConfig) -> None:
    """Raise ``ProtocolError`` listing every departure from the protocol."""
    problems = protocol_problems(cfg)
    if problems:
        raise ProtocolError(
            "The run config departs from the zero-field protocol:\n- " + "\n- ".join(problems)
        )


def protocol_problems(cfg: DictConfig) -> list[str]:
    """Every departure from the protocol that the declared control does not explain."""
    v = {key: OmegaConf.select(cfg, key, default=_ABSENT) for key in REQUIRED}
    missing = [f"{key}: missing" for key, value in v.items() if value is _ABSENT]
    if missing:
        return missing

    problems: list[str] = []
    control = v["control"]

    def expect(key: str, wanted: Any, actual: Any) -> None:
        if actual != wanted:
            problems.append(f"{key}: expected {wanted!r}, got {actual!r} (control={control!r})")

    run_type = OmegaConf.select(cfg, "run_type", default="train")
    if run_type not in RUN_TYPES:
        problems.append(f"run_type: expected one of {RUN_TYPES}, got {run_type!r}")
    # Blocking-check thresholds: numbers for training runs; a missing value is ??? (unset).
    gap = OmegaConf.select(cfg, "checks.teacher_gap", default=None)
    for key in ("checks.teacher_gap.token_max_abs", "checks.teacher_gap.response_mean_signed_abs"):
        leaf = key.rsplit(".", 1)[1]
        if not isinstance(gap, DictConfig) or gap._get_node(leaf) is None:
            problems.append(f"{key}: missing")
            continue
        if OmegaConf.is_missing(gap, leaf):  # ???: set later; computing the config hash fails
            continue
        value = gap[leaf]
        if run_type == "train" and value is not None and not value > 0:
            problems.append(f"{key}: expected a positive threshold, got {value!r}")
        elif run_type == "train" and value is None:
            problems.append(f"{key}: a training run needs the threshold set")
    batches = OmegaConf.select(cfg, "checks.path_field.batches", default=_ABSENT)
    if batches is _ABSENT or not (isinstance(batches, int) and batches >= 4 and batches % 2 == 0):
        problems.append(f"checks.path_field.batches: expected an even number >= 4, got {batches!r}")
    bound = OmegaConf.select(cfg, "checks.path_field.max_drift_to_noise", default=_ABSENT)
    if bound != PATH_FIELD_BOUND:
        problems.append(
            f"checks.path_field.max_drift_to_noise: the preregistered bound is {PATH_FIELD_BOUND}, "
            f"got {bound!r}"
        )
    if control not in CONTROLS:
        problems.append(f"control: unknown control {control!r}; known: {CONTROLS}")

    # Teacher: fresh θ₀ samples at the model's own distribution (invariants 2 and 3).
    expect(
        "teacher.source",
        "alpaca_reference" if control == "alpaca_reference" else "theta0",
        v["teacher.source"],
    )
    expect("teacher.prompt_assignment", "iid_with_replacement", v["teacher.prompt_assignment"])
    sampling = SHARPENED_SAMPLING if control == "sharpened_sampling" else ZERO_FIELD_SAMPLING
    for key, wanted in sampling.items():
        expect(f"teacher.sampling.{key}", wanted, v[f"teacher.sampling.{key}"])

    # Training stream: each response once (invariant 6); truncation only, never selection.
    steps, batch = v["train.steps"], v["train.batch_size"]
    expect("teacher.responses", steps * batch, v["teacher.responses"])
    layout = OmegaConf.select(cfg, "teacher.stream", default="fixed")
    if layout not in STREAM_LAYOUTS:
        problems.append(f"teacher.stream: expected one of {STREAM_LAYOUTS}, got {layout!r}")
    elif layout == "appendable":
        chunk = OmegaConf.select(cfg, "teacher.chunk_size", default=None)
        if not (isinstance(chunk, int) and not isinstance(chunk, bool) and chunk > 0):
            problems.append(f"teacher.chunk_size: expected a positive integer, got {chunk!r}")
        if control not in (None, "soft_targets"):  # other controls keep their own streams
            problems.append(f"teacher.stream: appendable streams serve zero-field runs and the "
                            f"soft-target control only (control={control!r})")  # fmt: skip
    cap, sampled = v["train.max_response_tokens"], v["teacher.sampling.max_new_tokens"]
    if not 1 <= cap <= sampled:
        problems.append(
            f"train.max_response_tokens: {cap} must lie in [1, {sampled}], the cap of "
            "the teacher samples it truncates"
        )

    problems += _schedule_problems(cfg, layout, steps)
    problems += _rayleigh_problems(cfg)

    # Loss: summed NLL over a constant (invariant 5), including the sampled end token.
    expect(
        "train.loss.kind", "soft_kl" if control == "soft_targets" else "nll", v["train.loss.kind"]
    )
    expect(
        "train.loss.normalization",
        "token_count" if control == "token_count_normalization" else "constant",
        v["train.loss.normalization"],
    )
    expect("train.loss.include_end_token", True, v["train.loss.include_end_token"])
    expect("train.loss.denominator", batch * cap, v["train.loss.denominator"])

    # Frozen modules and no weight decay (invariant 7); fp32 optimizer state.
    frozen = v["train.freeze"]
    frozen = list(frozen) if isinstance(frozen, ListConfig | list) else [frozen]
    lacking = [name for name in FROZEN_MODULES if name not in frozen]
    if lacking:
        problems.append(f"train.freeze: must include {lacking}")
    expect("optimizer.weight_decay", 0, v["optimizer.weight_decay"])
    for key in ("first_moment_dtype", "second_moment_dtype"):
        value = OmegaConf.select(cfg, f"optimizer.{key}", default=_ABSENT)
        if value is not _ABSENT:
            expect(f"optimizer.{key}", "float32", value)
    if control == "soft_targets":
        # Under Adam, tiny residual gradients would become full-size sign steps. Heavy-ball
        # momentum is linear in the gradient and keeps them tiny: from protocol version 2 the
        # SGD-momentum pilot's controls use it (docs/study2-pilot-momentum.md). Frozen Adam
        # qualifies from protocol version 3 for the same reason: its denominator is the fixed
        # v₀ estimated from NLL gradients (never from the soft loss), and its first moment is
        # linear in the gradient with a bias correction, so the update is linear in the gradient
        # and tiny residual gradients stay tiny. Plain Adam, AdamW and the prewarmed AdamW
        # renormalize by the running gradient scale and stay rejected at every version.
        version = OmegaConf.select(cfg, "protocol_version", default=1)
        name = v["optimizer.name"]
        if name == "adam_frozen" and version >= 3:
            # The control is judged by its scoring K against the bound, which lives in the
            # schedule: the schedule, its fixed policy (the arm's name depends on it) and the
            # campaign's positive bound are all required; the config's placeholder is missing.
            if OmegaConf.select(cfg, "schedule", default=None) is None:
                problems.append("schedule: a frozen-Adam soft control is a scheduled run")
            else:
                policy = OmegaConf.select(cfg, "schedule.policy", default=None)
                if policy != "fixed":
                    problems.append(
                        f"schedule.policy: expected 'fixed' for a frozen-Adam soft control, "
                        f"got {policy!r}"
                    )
                key = "schedule.instability.contamination_k"
                bound = None
                if (
                    not OmegaConf.is_missing(cfg.schedule, "instability")
                    and (OmegaConf.select(cfg, "schedule.instability", default=None) is not None)
                    and not OmegaConf.is_missing(cfg.schedule.instability, "contamination_k")
                ):
                    bound = OmegaConf.select(cfg, key, default=None)
                if not (isinstance(bound, int | float) and not isinstance(bound, bool)
                        and bound > 0):  # fmt: skip
                    problems.append(
                        f"{key}: a frozen-Adam soft control needs the positive bound "
                        f"(0.01 x K*) from its campaign, got {bound!r}"
                    )
        else:
            expect("optimizer.name", "sgd", name)
            momentum = OmegaConf.select(cfg, "optimizer.momentum", default=0)
            if version < 2:
                expect("optimizer.momentum", 0, momentum)
            elif not (isinstance(momentum, int | float) and 0 <= momentum < 1):
                problems.append(f"optimizer.momentum: expected a value in [0, 1), got {momentum!r}")

    # Model: pinned revision, the same fp32 numerics everywhere (invariant 1).
    if not (isinstance(v["model.revision"], str) and _COMMIT.fullmatch(v["model.revision"])):
        problems.append(
            f"model.revision: expected a 40-character commit hash, got {v['model.revision']!r}"
        )
    expect("model.dtype", "float32", v["model.dtype"])
    expect("model.tf32", False, v["model.tf32"])
    if v["model.attn_implementation"] not in ATTENTION:
        problems.append(
            f"model.attn_implementation: expected one of {ATTENTION}, got "
            f"{v['model.attn_implementation']!r}"
        )
    end_tokens = v["model.end_tokens"]
    if not (
        isinstance(end_tokens, ListConfig | list)
        and end_tokens
        and all(isinstance(token, str) and token for token in end_tokens)
    ):
        problems.append(
            f"model.end_tokens: expected a non-empty list of token strings, got {end_tokens!r}"
        )

    # Study 3: the step-25 checkpoint belongs to the numerical and engineering runs alone; a
    # search, validation or confirmation run never saves it (an unset role neither)
    if OmegaConf.select(cfg, "study3", default=None) is not None:
        role = OmegaConf.select(cfg, "study3.role", default=None)
        checkpoint = OmegaConf.select(cfg, "tracking.endpoint_checkpoint", default=None)
        if role not in STUDY3_CHECKPOINT_ROLES and checkpoint is not None:
            problems.append(
                f"tracking.endpoint_checkpoint: a study-3 run of role {role!r} saves no "
                f"endpoint checkpoint (only {STUDY3_CHECKPOINT_ROLES} do), got {checkpoint!r}"
            )
    return problems + _study5_problems(cfg)


def _study5_problems(cfg: DictConfig) -> list[str]:
    """Study 5's roles (``study5.role``, docs/study5-preregistration.md, "Descriptive
    measurements") and the keys only they carry (``study5_options``): a ``fresh`` F run is a plain
    F run (its line scan is a run of its own, ``line_scan``), a ``control`` run sets exactly one
    of ``train.order_seed``, ``optimizer.preconditioner.source_seed`` and ``train.order_edit``
    (extension 2; ``study5_options.edit_problems`` checks it); both may save the endpoint
    checkpoint. A ``dose`` run (docs/study5-dose-registration.md) carries a valid
    ``train.rate_schedule`` (``study5_options.schedule_problems``) and none of the three
    control keys; no other role carries the schedule. An ``order_a`` run
    (docs/study5-adamw-registration.md) is arm A (``adamw``) of a fresh seed in a new order:
    exactly ``train.order_seed`` among the three control keys, no schedule. A ``beta0`` run
    (docs/study5-beta0-registration.md) is arm F without momentum (``optimizer.beta1`` 0) on a
    seed's own responses, with at most one of ``train.order_seed`` and ``train.order_edit``, no
    other control key and no schedule; no other arm-F role takes ``optimizer.beta1`` 0. Every
    other study-5 run is arm F (``adam_frozen``); every study-5 run is a fixed-policy scheduled
    run; no other run carries a study-5 key."""
    role = OmegaConf.select(cfg, "study5.role", default=None)
    present = {key: OmegaConf.select(cfg, key, default=None)
               for key in (*STUDY5_KEYS, RATE_SCHEDULE)}  # fmt: skip
    present = {key: value for key, value in present.items() if value is not None}
    if OmegaConf.select(cfg, "study5", default=None) is None:
        return [f"{key}: carried by a study-5 role only (study5.role), got {value!r}"
                for key, value in present.items()]  # fmt: skip
    if OmegaConf.select(cfg, "study3", default=None) is not None:
        return ["study5: a study-3 run (study3.role) takes no study-5 role"]
    if role not in STUDY5_ROLES:
        return [f"study5.role: expected one of {STUDY5_ROLES}, got {role!r}"]
    problems = []
    arm = "adamw" if role == "order_a" else "adam_frozen"
    if OmegaConf.select(cfg, "optimizer.name", default=None) != arm:
        problems.append(f"optimizer.name: a study-5 run of role {role!r} is {arm}, got "
                        f"{OmegaConf.select(cfg, 'optimizer.name', default=None)!r}")  # fmt: skip
    if OmegaConf.select(cfg, "schedule.policy", default=None) != "fixed":
        problems.append("schedule.policy: a study-5 run is a fixed-policy scheduled run")
    if arm == "adam_frozen":
        beta1 = OmegaConf.select(cfg, "optimizer.beta1", default=None)
        if (role == "beta0") != (beta1 == 0):
            problems.append(f"optimizer.beta1: 0 in a run of role 'beta0' and in no other arm-F "
                            f"role, got {beta1!r} with role {role!r}")  # fmt: skip
    for key in ("train.order_seed", "optimizer.preconditioner.source_seed"):
        value = present.get(key)
        if value is not None and not (isinstance(value, int) and not isinstance(value, bool)
                                      and value >= 0):  # fmt: skip
            problems.append(f"{key}: expected a non-negative integer seed, got {value!r}")
    edit = present.get("train.order_edit")
    if edit is not None:
        from .study5_options import edit_problems  # a late import: study5_options imports runtime

        container = (
            OmegaConf.to_container(edit, resolve=True) if OmegaConf.is_config(edit) else edit
        )
        problems += edit_problems(container, int(cfg.train.steps))
    controls = [key for key in STUDY5_KEYS if key in present]
    schedule = present.get(RATE_SCHEDULE)
    if role != "dose" and schedule is not None:
        problems.append(f"{RATE_SCHEDULE}: carried by the dose role only, got role {role!r}")
    if role == "dose":
        from .study5_options import schedule_problems  # late, as above

        problems += [f"{key}: a control's key; a study-5 run of role 'dose' takes none"
                     for key in controls]  # fmt: skip
        if schedule is None:
            problems.append(f"{RATE_SCHEDULE}: a dose run needs one")
        else:
            container = (OmegaConf.to_container(schedule, resolve=True)
                         if OmegaConf.is_config(schedule) else schedule)  # fmt: skip
            problems += schedule_problems(container, int(cfg.train.steps))
    elif role == "fresh":
        problems += [f"{key}: a control's key; a study-5 run of role 'fresh' takes none"
                     for key in controls]  # fmt: skip
    elif role == "beta0":
        if len(controls) > 1 or "optimizer.preconditioner.source_seed" in controls:
            problems.append(
                "study5.role: a beta0 run sets at most one of train.order_seed "
                f"and train.order_edit and no other control key, got {controls}"
            )
    elif role == "order_a":
        if controls != ["train.order_seed"]:
            problems.append("study5.role: an order_a run sets train.order_seed and no other "
                            f"control key, got {controls}")  # fmt: skip
    elif len(controls) != 1:  # control
        problems.append("study5.role: a control sets exactly one of train.order_seed, "
                        "optimizer.preconditioner.source_seed and train.order_edit, "
                        f"got {controls}")  # fmt: skip
    return problems


def _schedule_problems(cfg: DictConfig, layout: Any, steps: Any) -> list[str]:
    """A study-2 schedule (schedule.Schedule): appendable stream, levels, limits, the monitor."""
    schedule = OmegaConf.select(cfg, "schedule", default=None)
    if schedule is None:
        return []
    problems = []
    if layout != "appendable":
        problems.append("schedule: a scheduled run reads an appendable stream (teacher.stream)")
    policy = schedule.get("policy")
    if policy not in SCHEDULE_POLICIES:
        problems.append(f"schedule.policy: expected one of {SCHEDULE_POLICIES}, got {policy!r}")
    levels = list(schedule.get("levels") or [])
    increasing = all(a < b for a, b in zip(levels, levels[1:], strict=False))
    if not levels or not increasing or levels[0] <= 0:
        problems.append(f"schedule.levels: expected increasing positive K, got {levels!r}")
    if policy == "fixed":
        at = list(schedule.get("evaluate_at") or [])
        if not at or any(not (isinstance(t, int) and 1 <= t <= steps) for t in at):
            problems.append(f"schedule.evaluate_at: expected steps in [1, {steps}], got {at!r}")
        elif max(at) != steps:  # a fixed run ends with its evaluation
            problems.append(f"schedule.evaluate_at: must include the last step {steps}, got {at!r}")
    every = OmegaConf.select(cfg, "eval.monitor.every", default=None)
    if not (isinstance(every, int) and every > 0):
        problems.append(f"eval.monitor.every: a scheduled run reads the monitor, got {every!r}")
    return problems + _monitor_problems(cfg, every)


def _rayleigh_problems(cfg: DictConfig) -> list[str]:
    """The endpoint Rayleigh quotient (``eval.rayleigh``, curvature.endpoint_rayleigh): measured
    by study-2 runs on a curvature panel of a positive size, for the constant denominator."""
    rayleigh = OmegaConf.select(cfg, "eval.rayleigh", default=None)
    if rayleigh is None:
        return []
    if not isinstance(rayleigh, DictConfig):
        return [f"eval.rayleigh: expected a mapping, got {rayleigh!r}"]
    enabled = rayleigh.get("enabled")
    if not isinstance(enabled, bool):
        return [f"eval.rayleigh.enabled: expected true or false, got {enabled!r}"]
    if not enabled:
        return []
    problems = []
    if OmegaConf.select(cfg, "schedule", default=None) is None:
        problems.append("eval.rayleigh: measured by scheduled (study-2) runs only")
    responses = rayleigh.get("responses")
    if not (isinstance(responses, int) and not isinstance(responses, bool) and responses > 0):
        problems.append(f"eval.rayleigh.responses: expected a positive integer, got {responses!r}")
    if OmegaConf.select(cfg, "train.loss.normalization") != "constant":
        problems.append("eval.rayleigh: the Hessian is that of the constant denominator")
    return problems


MONITOR_CACHES = ("device", "host", "none")


def _monitor_problems(cfg: DictConfig, every: Any) -> list[str]:
    """Monitor version 2 (amendment 1): the scoring prompts, nested panels, caches and the
    scoring panel's diagnostics at a fixed multiple of the reading interval."""
    monitor = OmegaConf.select(cfg, "eval.monitor", default=None)
    version = monitor.get("version", 1) if monitor is not None else 1
    if version == 1:
        return []
    problems = []
    if version != 2:
        return [f"eval.monitor.version: expected 1 or 2, got {version!r}"]
    if monitor.get("prompt_set") != "alpaca_benign_kl":
        problems.append("eval.monitor.prompt_set: version 2 reads the scoring panel's prompts "
                        f"(alpaca_benign_kl), got {monitor.get('prompt_set')!r}")  # fmt: skip
    per_prompt, drive = monitor.get("samples_per_prompt"), monitor.get("drive")
    if not (isinstance(drive, int) and isinstance(per_prompt, int) and 1 <= drive <= per_prompt):
        problems.append(f"eval.monitor.drive: expected 1 <= drive <= samples_per_prompt, got "
                        f"{drive!r} of {per_prompt!r}")  # fmt: skip
    caches = monitor.get("cache") or {}
    for part in ("primary", "secondary"):
        if caches.get(part) not in MONITOR_CACHES:
            problems.append(f"eval.monitor.cache.{part}: expected one of {MONITOR_CACHES}")
    diagnostics = OmegaConf.select(cfg, "eval.diagnostics", default=None) or {}
    step = diagnostics.get("every")
    if not (isinstance(step, int) and isinstance(every, int) and every > 0 and step > 0
            and step % every == 0):  # fmt: skip
        problems.append(f"eval.diagnostics.every: a positive multiple of eval.monitor.every "
                        f"({every!r}), got {step!r}")  # fmt: skip
    if diagnostics.get("cache") not in MONITOR_CACHES:
        problems.append(f"eval.diagnostics.cache: expected one of {MONITOR_CACHES}")
    return problems
