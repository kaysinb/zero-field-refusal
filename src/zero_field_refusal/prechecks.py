"""Checks that must pass before a run takes its first optimizer step.

A stream sampled in a numerically different environment, or a pipeline whose generation and
training paths disagree, would put a real field into a zero-field run that still looks plausible.
Two checks therefore block training:

- **Path field (invariant 8).** The stage-0 checks measure, per execution environment, the bound
  on the residual field that the path discrepancy creates and write a record under
  ``DATA_DIR/checks/``. A training run needs a passed record for its own environment (model
  revision and numerics, determinism and the cuBLAS workspace setting, device, library versions,
  execution batch sizes, and the check, sampler and protocol versions) that covers its steps,
  batch size and response cap. A missing, failed or out-of-scope record stops it.
- **Teacher gap (invariant 1).** At step 0 the training path must reproduce the stored teacher
  log-probabilities of the run's whole stream within the preregistered thresholds.
- **The run's own path field.** From the same step-0 differences Δ_t of the sampled tokens,
  Σ_t expm1(Δ_t)² per response estimates the response's χ²(training path ‖ generation path)
  without bias (E_{y~p}[(q/p − 1)²] = χ²(q‖p)). The run requires the upper confidence bound on
  N·X to stay within the preregistered bound for the pair of its stream and its environment,
  whichever environment generated the stream.

Both χ² checks use the same upper bound (``chi_square_bound``): responses are independent, so the
per-response sums are the units, and the bound is the one-sided studentized-bootstrap (bootstrap-t)
bound at ``PATH_FIELD_LEVEL``, which keeps its coverage much better than a Student quantile when
the sums are right-skewed (tests/test_prechecks.py).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import secrets
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .protocol import PATH_FIELD_BOUND, PATH_FIELD_LEVEL
from .teacher import SAMPLER_VERSION, fsync_directory
from .tracking import package_versions


class CheckFailed(RuntimeError):
    """A blocking check failed or is missing; the run stops before any optimizer step."""


# Part of every record's fingerprint: bump on any change to how the path field is measured, so
# records written by older code never unlock training.
CHECK_VERSION = 3
# Resamples of the bootstrap-t bound on the χ² of the path discrepancy, and their seed.
BOOTSTRAP_RESAMPLES = 9999
BOOTSTRAP_SEED = 0


def environment_fingerprint(cfg: DictConfig, device: torch.device) -> dict[str, Any]:
    """What determines the numerics of the generation and training paths, and the check's code.

    Determinism (``execution.deterministic``) and, on a GPU, the process's
    ``CUBLAS_WORKSPACE_CONFIG`` (as ``runtime.configure_numerics`` left it) are numerics too
    (docs/study3-preregistration.md, "The bridge's conditions"). Each enters the fingerprint only
    when set, so the records of studies 1 and 2 (deterministic false, no cuBLAS setting) keep
    their digests and keep matching their configs, and a record without them never matches a run
    that sets them.
    """
    packages = package_versions()
    fingerprint = {
        "check_version": CHECK_VERSION,
        "sampler_version": SAMPLER_VERSION,
        "protocol_version": cfg.protocol_version,
        "micro_batch_size": cfg.execution.micro_batch_size,  # kernel shapes
        "generation_batch_size": cfg.execution.generation_batch_size,
        "model": cfg.model.name,
        "revision": cfg.model.revision,
        "dtype": cfg.model.dtype,
        "attn_implementation": cfg.model.attn_implementation,
        "tf32": bool(cfg.model.tf32),
        "device": device.type,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()
        ),
        "torch": packages.get("torch"),
        "transformers": packages.get("transformers"),
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
    }
    if OmegaConf.select(cfg, "execution.deterministic", default=False):
        fingerprint["deterministic"] = True
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG") if device.type == "cuda" else None
    if workspace is not None:
        fingerprint["cublas_workspace_config"] = workspace
    return fingerprint


def record_path(root: Path, fingerprint: dict[str, Any]) -> Path:
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
    return root / "checks" / fingerprint["model"].replace("/", "--") / f"path-field-{digest}.json"


def write_check_record(
    root: Path, fingerprint: dict[str, Any], results: dict[str, Any], *, passed: bool, run_id: str
) -> Path:
    """Write (atomically) the path-field record of one environment; an older one moves to
    ``history/``, so a later pass never silently replaces a failure.

    ``results`` must hold what the record certifies: ``steps``, ``batch_size``, ``cap``,
    ``criterion`` and ``chi_square_bound`` (as returned by ``chi_square_bound``).
    """
    path = record_path(root, fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        history = path.parent / "history"
        history.mkdir(exist_ok=True)
        path.rename(history / f"{path.stem}.{secrets.token_hex(4)}.json")
    record = {"fingerprint": fingerprint, "passed": passed, "results": results, "run_id": run_id,
              "created_at": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fsync_directory(path.parent)  # make the replacement itself durable
    return path


def require_path_check(root: Path, cfg: DictConfig, device: torch.device) -> dict[str, Any]:
    """The passed path-field record of this environment, or ``CheckFailed``."""
    fingerprint = environment_fingerprint(cfg, device)
    path = record_path(root, fingerprint)
    if not path.is_file():
        raise CheckFailed(
            f"No stage-0 path-field check for this environment ({fingerprint}); "
            "run scripts/stage0_checks.py here first"
        )
    record = json.loads(path.read_text())
    if not record["passed"]:
        raise CheckFailed(f"The stage-0 path-field check failed for this environment: {path}")
    certified = record["results"]
    if certified["criterion"] != PATH_FIELD_BOUND:
        raise CheckFailed(f"{path} certifies the bound {certified['criterion']}, not the "
                          f"preregistered {PATH_FIELD_BOUND}")  # fmt: skip
    if cfg.train.max_response_tokens > certified["cap"]:
        raise CheckFailed(f"{path} certifies responses up to {certified['cap']} tokens, the run "
                          f"uses {cfg.train.max_response_tokens}")  # fmt: skip
    bound = certified["chi_square_bound"]
    if bound["level"] != PATH_FIELD_LEVEL:
        raise CheckFailed(f"{path} bounds χ² at confidence {bound['level']}, not the "
                          f"preregistered {PATH_FIELD_LEVEL}")  # fmt: skip
    # X scales with the responses of a batch; the bound with the number of steps.
    value = _upper(bound, cfg.train.batch_size, cfg.train.steps)
    if not value <= PATH_FIELD_BOUND:
        raise CheckFailed(f"{path}: for this run's steps and batch size the bound is {value:.3g} "
                          f"> {PATH_FIELD_BOUND}")  # fmt: skip
    return record


def check_teacher_gap(gap: dict[str, float], thresholds: DictConfig) -> None:
    """Raise ``CheckFailed`` if the step-0 gap exceeds the thresholds, or they are not set."""
    keys = ("token_max_abs", "response_mean_signed_abs")
    if any(OmegaConf.is_missing(thresholds, k) or thresholds[k] is None for k in keys):
        raise CheckFailed("The teacher-gap thresholds are not set (checks.teacher_gap)")
    token, response = thresholds.token_max_abs, thresholds.response_mean_signed_abs
    # Written so that NaN fails: NaN <= x is False.
    if not (gap["token_max_abs"] <= token and abs(gap["response_mean_signed"]) <= response):
        raise CheckFailed(
            f"Step-0 teacher gap {gap} exceeds the thresholds (token {token}, response {response})"
        )


def chi_square_bound(
    per_response: Sequence[float],
    *,
    batch_size: int,
    steps: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, float]:
    """Upper confidence bound on N·X from per-response sums of χ² over the response's tokens.

    Instructions are drawn with replacement and responses sampled independently, so the
    per-response sums are i.i.d.: X̂ = B·mean and SE = B·sd/√n per batch of B responses. The
    bound N·(X̂ + c·SE) takes c from the bootstrap distribution of the studentized mean
    (bootstrap-t) at the one-sided level ``PATH_FIELD_LEVEL``; it adapts to the right skew of the
    sums, where a Student quantile undercovers. ``skewness`` and ``largest_share`` (the largest
    response's share of the total) show how heavy the tail is: no bound keeps its level for very
    heavy tails.
    """
    values = np.asarray(list(per_response), dtype=np.float64)
    n = len(values)
    if n < 2:
        raise ValueError(f"The χ² bound needs at least 2 responses, got {n}")
    mean, sd = values.mean(), values.std(ddof=1)
    se = sd / math.sqrt(n)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    t = np.empty(resamples)
    for start in range(0, resamples, 1000):  # bounded memory for long streams
        draws = values[rng.integers(0, n, (min(1000, resamples - start), n))]
        spread = draws.std(axis=1, ddof=1) / math.sqrt(n)
        t[start : start + len(draws)] = np.where(
            spread > 0, (draws.mean(axis=1) - mean) / np.where(spread > 0, spread, 1.0), 0.0
        )
    quantile = float(-np.quantile(t, 1 - PATH_FIELD_LEVEL)) if se > 0 else 0.0
    total = values.sum()
    out = {"responses": n, "level": PATH_FIELD_LEVEL, "method": "bootstrap-t",
           "resamples": resamples, "quantile": quantile,
           "chi_square_per_response": float(mean), "standard_error_per_response": float(se),
           "chi_square_per_batch": float(batch_size * mean),
           "standard_error": float(batch_size * se),
           "skewness": float(((values - mean) ** 3).mean() / sd**3) if sd > 0 else 0.0,
           "largest_share": float(values.max() / total) if total > 0 else 0.0}  # fmt: skip
    out["bound_upper"] = _upper(out, batch_size, steps)
    return out


def _upper(bound: dict[str, float], batch_size: int, steps: int) -> float:
    mean, se = bound["chi_square_per_response"], bound["standard_error_per_response"]
    return steps * batch_size * (mean + bound["quantile"] * se)


def check_run_chi_square(
    per_response: Sequence[float], *, batch_size: int, steps: int
) -> dict[str, float]:
    """The run's own path-field bound from per-response Σ_t expm1(Δ_t)², or ``CheckFailed``."""
    out = chi_square_bound(per_response, batch_size=batch_size, steps=steps)
    if not out["bound_upper"] <= PATH_FIELD_BOUND:
        raise CheckFailed(f"The run's stream fails the path-field bound: {out}")
    return out
