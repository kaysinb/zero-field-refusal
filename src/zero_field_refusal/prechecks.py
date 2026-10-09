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
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig

from .tracking import package_versions


class CheckFailed(RuntimeError):
    """A blocking check failed or is missing; the run stops before any optimizer step."""


def environment_fingerprint(cfg: DictConfig, device: torch.device) -> dict[str, Any]:
    """What determines the numerics of the generation and training paths."""
    packages = package_versions()
    return {
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
    """Write (atomically) the path-field record of one environment."""
    path = record_path(root, fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"fingerprint": fingerprint, "passed": passed, "results": results, "run_id": run_id,
              "created_at": datetime.now(UTC).isoformat(timespec="seconds")}  # fmt: skip
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
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
    return record


def check_teacher_gap(gap: dict[str, float], thresholds: DictConfig) -> None:
    """Raise ``CheckFailed`` if the step-0 gap exceeds the thresholds, or they are not set."""
    token, response = thresholds.token_max_abs, thresholds.response_mean_signed_abs
    if token is None or response is None:
        raise CheckFailed("The teacher-gap thresholds are not set (checks.teacher_gap)")
    if gap["token_max_abs"] > token or abs(gap["response_mean_signed"]) > response:
        raise CheckFailed(
            f"Step-0 teacher gap {gap} exceeds the thresholds (token {token}, response {response})"
        )
