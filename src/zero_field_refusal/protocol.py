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
        # Under Adam, tiny residual gradients would become full-size sign steps.
        expect("optimizer.name", "sgd", v["optimizer.name"])
        expect("optimizer.momentum", 0, OmegaConf.select(cfg, "optimizer.momentum", default=0))

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
    return problems
