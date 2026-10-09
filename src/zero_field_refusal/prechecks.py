"""Checks that must pass before a run takes its first optimizer step.

A stream sampled in a numerically different environment, or a pipeline whose generation and
training paths disagree, would put a real field into a zero-field run that still looks plausible.
Two checks therefore block training:

- **Path field (invariant 8).** The stage-0 checks measure, per execution environment, the bound
  on the residual field that the path discrepancy creates and write a record under
  ``DATA_DIR/checks/``. A training run needs a passed record for its own environment: model
  revision and numerics, device and library versions. A missing or failed record stops it.
- **Teacher gap (invariant 1).** At step 0 the training path must reproduce the stored teacher
  log-probabilities of the run's whole stream within the preregistered thresholds.
- **The run's own path field.** From the same step-0 differences Δ_t of the sampled tokens,
  Σ_t expm1(Δ_t)² per batch estimates the batch's χ²(training path ‖ generation path) without
  bias (E_{y~p}[(q/p − 1)²] = χ²(q‖p)). The run requires N·(X̂ + 2 SE) ≤ the preregistered bound
  for the pair of its stream and its environment, whichever environment generated the stream.
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

import torch
from omegaconf import DictConfig, OmegaConf

from .protocol import PATH_FIELD_BOUND
from .teacher import SAMPLER_VERSION
from .tracking import package_versions


class CheckFailed(RuntimeError):
    """A blocking check failed or is missing; the run stops before any optimizer step."""


# Part of every record's fingerprint: bump on any change to how the path field is measured, so
# records written by older code never unlock training.
CHECK_VERSION = 2


def environment_fingerprint(cfg: DictConfig, device: torch.device) -> dict[str, Any]:
    """What determines the numerics of the generation and training paths, and the check's code."""
    packages = package_versions()
    return {
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


def record_path(root: Path, fingerprint: dict[str, Any]) -> Path:
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12]
    return root / "checks" / fingerprint["model"].replace("/", "--") / f"path-field-{digest}.json"


def write_check_record(
    root: Path, fingerprint: dict[str, Any], results: dict[str, Any], *, passed: bool, run_id: str
) -> Path:
    """Write (atomically) the path-field record of one environment; an older one moves to
    ``history/``, so a later pass never silently replaces a failure.

    ``results`` must hold what the record certifies: ``steps``, ``batch_size``, ``cap``,
    ``criterion`` and ``chi_square_bound`` (``chi_square_per_batch``, ``standard_error``).
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
    # X scales with the tokens of a batch; the bound with the number of steps.
    bound = certified["chi_square_bound"]
    value = (cfg.train.steps * cfg.train.batch_size / certified["batch_size"]
             * (bound["chi_square_per_batch"] + 2 * bound["standard_error"]))  # fmt: skip
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


def check_run_chi_square(batch_sums: Sequence[float], steps: int) -> dict[str, float]:
    """The run's own path-field bound from per-batch Σ_t expm1(Δ_t)², or ``CheckFailed``."""
    values = torch.tensor(list(batch_sums), dtype=torch.float64)
    mean = values.mean().item()
    se = values.std().item() / math.sqrt(len(values)) if len(values) > 1 else float("inf")
    out = {"chi_square_per_batch": mean, "standard_error": se,
           "bound_upper": steps * (mean + 2 * se)}  # fmt: skip
    if not out["bound_upper"] <= PATH_FIELD_BOUND:
        raise CheckFailed(f"The run's stream fails the path-field bound: {out}")
    return out
