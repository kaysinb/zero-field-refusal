"""Study 2: the diagnostic of the FA soft-control failure, outside the confirmation.

The specification is docs/study2-fa-control-diagnostic.md, frozen before any run; its Revision 1
supersedes the base text where they differ, its Revision 2 fixes the operationalizations (the
path criterion, the same-path control, the cases of Branch F, survival of fp32 rounding, the
AdamW and FA 1e-5 scales, D2's map, the float64 reference, "incomplete"), and its Revision 3
chooses or excludes a branch only on complete evidence (shared state gates every branch,
survival is required, three states per measurement, the branch order). The settings are
configs/experiments/fa_control_diagnostic.yaml. Nothing here changes the protocol, the
selection, the analysis or a run record.

**Blind to outcomes.** Only K, KL between twins, norms, eigenvalues and v₀ statistics leave this
module. Every evaluation is K-only, on benign panels: the scoring panel's responses
(``kl_benign``, as ``train.Evaluator`` limits them) and the monitor panel. No harmful prompt set,
stream or generation is touched, and no refusal, harm or judge quantity is computed
(tests/test_fa_control_diagnostic.py checks both). Outputs go to
``RUNS_DIR/diagnostics/fa_control/`` (``output_root``), a namespace no analysis of the stage
reads; no ``meta.json`` or ``terminal.json`` is written there.

**The runs re-examined** are found by their archived configs (``locate_runs``): the regular runs
of the targeted round (``study2_short``, seed 2000, B 5) and the FA soft controls
(``study2_short_fa_controls``). Every computation uses the archived run's own resolved config.

Parts, in the specification's order (Revision 1, "Order"):

- **D4 + D1** (``run_d1``): the control and the regular run share v₀ (sha256), the loss
  normalization, the trainable parameters and the initial optimizer state; the census of v₀
  (``v0_census``); the separation of the initial residual gradient on the control's first batch:
  (i) soft loss with training-path targets in fp32, (ii) the same with loss and accumulation in
  float64, (iii) the control's own gradient (generation-path targets, replayed), recomputed and
  checked against the recorded norm; the share of ‖P·g‖² from ε-dominated coordinates; and the
  same-path float64 soft control at the failing rate, 25 steps, with K read on the scoring panel
  (Branch F's condition).
- **D2** (``run_d2``): per batch of the control's first k batches, the top (and bottom) Ritz
  values of P^½ H P^½ by Lanczos (``curvature.lanczos``) for (a) the soft-target loss with its
  replayed targets and (b) the sampled NLL, with the top eigenvector's module localization from a
  second Lanczos pass; the code's bias-corrected recurrence linearized over the run's steps with
  the k batches cycled, started from the measured residual direction, for each rate; reference
  thresholds; and the controls' observed growth.
- **D3** (``run_d3``): determinism first, then per case the twins: the unperturbed regular run
  (twin A, its trainable parameters saved at the evaluation steps), the perturbed runs p1–p4 and
  the float64 reference, with D_twin(t) = max(KL(A_t‖B_t), KL(B_t‖A_t)) on the scoring panel.
- **Branches** (``decide_branch``): F, B, A, ambiguous or incomplete, from the parts' outputs,
  in Revision 3's order, on each measurement's state: a D3 pair within, exceeding or
  incomplete (``pair_state``), the same-path control below, exceeding, at the bound or
  incomplete (``control_state``); divergence counts as exceeding.

**Memory** (1.5B, fp32): a trainable-sized vector is ≈ 5.2 GB. The base D3's rule, "46 GB
reserved", is enforced as a cap on the process's CUDA memory (``apply_memory_rule``, in the GB of
the run records' peaks, 1e9 bytes); every peak is recorded with ``within_memory_rule``. The
second model of D3 stays in host memory except while D_twin is measured; float64 gradients are
summed as each parameter's gradient arrives (``Float64Gradients``), so no fp32 gradient set
coexists with them; D2's Ritz vector stays beside the Lanczos vectors only if it fits under the
cap, and otherwise goes to host memory through one pinned buffer.

**Validity.** Outputs written at another commit are moved aside and redone (``kept``); a twin A
that differs from its record, or a failed D1 shared-state check (the trainable set, v₀'s hash,
the loss normalization, the initial optimizer state, the recomputed first gradient against
every control's record) makes the outcome ``incomplete`` (``decide_branch``), as does an
undecided path criterion, a failed determinism check, a missing output or a case without its
five pairs.

**The numerical check** (study 2, amendment 4, section 1; scripts/numerical_check.py, settings
configs/experiments/numerical_check.yaml) generalizes D1's same-path control and D3's twins to
fresh seeds, all three arms and the selected B and U, with Revision 3's states. Its outputs go
to ``RUNS_DIR/numerical/b{B}-u{U}/`` (``numerical_root``), and the same rules hold: K-only,
benign panels, nothing written to a run record.

- **Cases** (``numerical_cases``): every selected (arm, rate) on the first numerical seed, and
  each arm's U-rate on the others. Twin A is the regular run of that seed, arm, rate and B
  (experiment ``study2_short``, ``study2_short.role=numerical``), which must conform
  (``conformance``) and is re-run bit for bit (``compare_archive``). Per seed (``twin_setup``,
  ``seed_directions``), p1 and p2 are ±(−P·g), FA's first-update direction for the soft-target
  residual g at θ₀ on batch 0 with generation-path targets, the same for every rate and arm;
  p3 and p4 the fixed Gaussian directions. Every perturbation has the scope's fixed norm
  (5e-5), whatever any control did; survival and the ×10 retry are Revision 3, 2's. The
  reference is the float64 run (pre-warmed Adam included, ``Float64Adam``), with its
  float64-memory fallback, recorded per case. D_twin is read at the scope's steps against
  0.01·K*(U), K*(U) = √(5U). The norm one fp32 rounding of θ₀ would leave is recorded as a
  heuristic, not a bound on accumulated error (``rounding_heuristic``). The regular runs are
  K-only: nothing of an evaluation of harm is read from them.
- **The same-path control** (``run_numerical_controls``) is D1's (``same_path_run``) at each
  selected FA rate on the first seed, from its regular run's config, checked to read that run's
  v₀ stream and training prefix.
- **Attempts.** Each case and control is written per attempt; an incomplete one runs once
  more, in a fresh process. A failure is the attempt's record, never lost.
- **The verdict** (``numerical_verdict``), with its policy (``NumericalScope.policy``) and the
  seeds each rate was tested on: an (arm, rate) is qualified when every pair of every case is
  within, failing when a pair exceeds (evidence of amplification) or an instability rule
  stopped a case's regular run, not_qualified (technical, never instability) when a case is
  incomplete after its rerun, else incomplete (``qualification``); a control is accepted when
  "below", rejected when exceeding, at the bound or its run stopped by instability,
  not_qualified when incomplete after its rerun, else incomplete (``control_verdict``). Each
  selection keeps its own verdict (``verdicts/<selection_digest>.json``, ``write_verdict``). A
  replacement rate's cases are ``replacement_cases``.
- **Which records count** (``RecordReader``): those written from a clean checkout at a commit
  whose GPU code is HEAD's (``code_identity.reuse_check``), under the frozen settings, with
  the policy's perturbations, of the regular run located now. Reading moves nothing; only a
  real run of a part, from a clean checkout and after the code check, sets a record aside.
- **GPU code** (review fa13): the qualification attaches to the GPU code the regular runs and the
  check executed (``code_identity.gpu_code_identity``), not to a commit. Nothing runs, and no
  record counts, unless the commit each regular run recorded has the check's GPU identity
  (``code_check``, ``require_code``).
- **The run environment** (the final review of gate B, I3): twin A reproduces its run bit for
  bit only where the run was made; ``environment_check`` compares each regular run's recorded
  GPU names, driver, CUDA, Python, packages and container image with the check's
  (``environment_departures``), and the script runs nothing on a mismatch. Each record holds
  the host memory when it was written (``host_record``, M7).
- **Voiding** (the final review of gate B, I2): after a GPU-path change the numerical runs made
  under the old GPU code can never be checked again; ``void_plan`` and ``void_numerical`` move
  them, the check's directory and the launcher's record of their campaigns to
  ``RUNS_DIR/void/<old GPU code>/`` with an inventory, so that the selection calls for the runs
  and the check anew. Not once S3 has started at that batch (the coordinator's ruling).
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import platform
import re
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from omegaconf import DictConfig, OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PreTrainedModel

from .batches import chunks, collate
from .code_identity import gpu_code_identity, reuse_check
from .curvature import _dot, _norm, _ritz, lanczos, start_vector
from .evaluation import NO_CAP, sequence_kl
from .losses import loss_denominator, response_logits, response_logprobs, soft_kl, summed_nll
from .losses import targets as target_ids
from .models import freeze
from .optim import ADAM_MODES, build_optimizer
from .panels import Monitor
from .paths import runs_root
from .runtime import (
    Environment,
    ReplayedSoftTargets,
    StreamPrefix,
    load_stream_for,
    locate_stream,
)
from .schedule import INSTABILITY as RUN_INSTABILITY
from .schedule import _norm as grad_norm
from .seeding import derive_seed, seed_everything
from .teacher import MANIFEST_FILE, Sample, published_chunks
from .tracking import LAUNCH_ENV, git_state, hardware, package_versions
from .train import _theta0_second_moment, accumulate_gradient

Vector = list[torch.Tensor]
Term = Callable[[], torch.Tensor]
QUANTILES = (0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
PERTURBATIONS = ("p1", "p2", "p3", "p4")
REFERENCE = "reference"
PAIRS = (*PERTURBATIONS, REFERENCE)  # every D3 case holds exactly these pairs
BRANCHES = ("F", "B", "A", "ambiguous", "incomplete")
HESSIANS = {  # D2's two Hessians, as their labels say (review M9)
    "soft": "(a) exact soft-loss Hessian, with the control's replayed targets (≈ Gauss–Newton at "
            "θ₀: the two differ by the residual term)",
    "nll": "(b) exact Hessian of the sampled NLL (may be indefinite)",
    "F": "the 4 batches' exact soft-loss Hessian, pooled (≈ Gauss–Newton at θ₀)",
}  # fmt: skip


class DiagnosticError(RuntimeError):
    """The diagnostic cannot run as specified (a missing or ambiguous record, say)."""


class DeterminismFailure(DiagnosticError):
    """The re-run of the regular run differs from its archive: D3 stops (the specification)."""


# ------------------------------------------------------------------------------- settings, runs


@dataclass(frozen=True)
class Case:
    """A D3 case: the regular run of ``optimizer`` at ``lr`` (as archived), perturbed at the norm
    of the FA soft control's first-step displacement at ``control_lr``."""

    name: str
    optimizer: str
    lr: float
    control_lr: float


def cases_of(settings: DictConfig) -> dict[str, Case]:
    return {
        name: Case(name, str(c.optimizer), float(c.lr), float(c.control_lr))
        for name, c in settings.diagnostic.d3.cases.items()
    }


@dataclass
class Runs:
    """The archived run directories the diagnostic reads: the regular run of each case and the
    FA soft control at each rate."""

    regular: dict[str, Path]
    controls: dict[float, Path]

    def control(self, rate: float) -> Path:
        for lr, path in self.controls.items():
            if math.isclose(lr, rate, rel_tol=1e-9):
                return path
        raise DiagnosticError(f"No FA soft control at {rate:g} among {sorted(self.controls)}")


def _matches(cfg: DictConfig, *, seed: int, batch: int, optimizer: str, lr: float,
             control: str | None, role: str | None = None) -> bool:  # fmt: skip
    return (
        cfg.get("seed") == seed
        and cfg.train.batch_size == batch
        and cfg.optimizer.name == optimizer
        and math.isclose(float(cfg.optimizer.lr), lr, rel_tol=1e-9)
        and cfg.get("control") == control
        and (role is None or OmegaConf.select(cfg, "study2_short.role", default=None) == role)
    )


def find_run(root: Path, experiment: str, **match: Any) -> Path:
    """The one finished run (a ``terminal.json``) under ``root/experiment`` whose archived config
    matches (with ``role``, its ``study2_short.role`` too); anything else is an error naming what
    was found."""
    found = []
    for directory in sorted((root / experiment).glob("*")):
        finished = (directory / "config.yaml").is_file() and (directory / "terminal.json").is_file()
        if finished and _matches(OmegaConf.load(directory / "config.yaml"), **match):
            found.append(directory)
    if len(found) != 1:
        raise DiagnosticError(
            f"{len(found)} finished runs under {root / experiment} match {match}: "
            f"{[p.name for p in found]}; exactly one is needed"
        )
    return found[0]


def locate_runs(
    settings: DictConfig, root: Path, *, case_names: Sequence[str] | None = None
) -> Runs:
    d = settings.diagnostic
    seed, batch = int(d.seed), int(d.batch_size)
    cases = cases_of(settings)
    names = list(cases) if case_names is None else list(case_names)
    regular = {
        name: find_run(root, d.experiments.regular, seed=seed, batch=batch,
                       optimizer=cases[name].optimizer, lr=cases[name].lr, control=None)
        for name in names
    }  # fmt: skip
    controls = {
        float(rate): find_run(root, d.experiments.control, seed=seed, batch=batch,
                              optimizer="adam_frozen", lr=float(rate), control="soft_targets")
        for rate in d.controls
    }  # fmt: skip
    return Runs(regular=regular, controls=controls)


def base_control_rate(settings: DictConfig) -> float:
    """The rate of the FA soft control whose archived config serves where a control config is
    needed at another rate (the first step at a case's rate): the pair's of D1."""
    return cases_of(settings)[settings.diagnostic.d1.pair].control_lr


def run_config(directory: Path) -> DictConfig:
    """A run's archived, resolved config."""
    return OmegaConf.load(directory / "config.yaml")


def with_lr(cfg: DictConfig, lr: float) -> DictConfig:
    out = copy.deepcopy(cfg)
    out.optimizer.lr = lr
    return out


def output_root(root: Path | None = None) -> Path:
    return (runs_root() if root is None else root) / "diagnostics" / "fa_control"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    """Write a record atomically: a reader sees all of it or nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=1) + "\n")
    os.replace(partial, path)


def provenance(settings: DictConfig, runs: Runs) -> dict[str, Any]:
    return {
        "spec": "docs/study2-fa-control-diagnostic.md",
        "code": git_state(),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": OmegaConf.to_container(settings.diagnostic, resolve=True),
        "runs": {"regular": {k: str(v) for k, v in runs.regular.items()},
                 "controls": {f"{k:g}": str(v) for k, v in runs.controls.items()}},
        "hardware": hardware(),
        "packages": package_versions(),
    }  # fmt: skip


class Recorder:
    """Stands in for ``tracking.Run`` where the training code records its inputs (v₀'s stream,
    the monitor's panel), so that their identities land in the diagnostic's own output."""

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.inputs: dict[str, Any] = {}

    def summarize(self, values: Mapping[str, Any]) -> None:
        self.summary.update(json.loads(json.dumps(values, default=str)))

    def record_inputs(self, inputs: Mapping[str, Any]) -> None:
        for key, value in json.loads(json.dumps(inputs, default=str)).items():
            if isinstance(value, dict) and isinstance(self.inputs.get(key), dict):
                self.inputs[key].update(value)
            else:
                self.inputs[key] = value


# The memory rule in force (``apply_memory_rule``): the base D3's "46 GB reserved", in the GB of
# the study-2 run records (1e9 bytes, as ``schedule._peak_memory`` writes their peaks).
_MEMORY_RULE: dict[str, Any] = {}


def cuda_index(device: torch.device | str) -> int:
    """The index of a CUDA device: its own, or the current device's when it has none (as
    ``runtime.environment`` gives it, ``torch.device("cuda")``). Every per-device CUDA call here
    takes this index, since some (``set_per_process_memory_fraction``) refuse an unindexed
    device."""
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def apply_memory_rule(device: torch.device, limit_gb: float) -> dict[str, Any]:
    """Cap this process's CUDA memory at ``limit_gb`` (1e9 bytes) reserved, so that the rule
    itself raises the out-of-memory error: the float64 reference then falls back (recorded) and
    every other part fails. Nothing on the CPU. A limit above the device's memory caps nothing
    (the fraction is clamped to 1)."""
    device = torch.device(device)
    _MEMORY_RULE.clear()
    _MEMORY_RULE.update({"limit_gb": float(limit_gb), "fraction": None, "total_gb": None})
    if device.type == "cuda":
        index = cuda_index(device)
        total = torch.cuda.get_device_properties(index).total_memory
        fraction = min(1.0, float(limit_gb) * 1e9 / total)
        torch.cuda.set_per_process_memory_fraction(fraction, index)
        _MEMORY_RULE.update({"fraction": fraction, "total_gb": round(total / 1e9, 3),
                             "device_index": index})  # fmt: skip
    return dict(_MEMORY_RULE)


def memory_limit_bytes(device: torch.device) -> float | None:
    """The bytes the rule allows this process, or None without a rule or a GPU."""
    device = torch.device(device)
    if device.type != "cuda" or not _MEMORY_RULE.get("fraction"):
        return None
    total = torch.cuda.get_device_properties(cuda_index(device)).total_memory
    return _MEMORY_RULE["fraction"] * total


def peak_memory(device: torch.device) -> dict[str, Any]:
    """The GPU peaks since the last reset (allocated, reserved; GB of 1e9 bytes), with
    ``within_memory_rule``: the reserved peak against the rule's limit."""
    device = torch.device(device)
    if device.type != "cuda":
        return {}
    index = cuda_index(device)
    reserved = round(torch.cuda.max_memory_reserved(index) / 1e9, 3)
    out = {"peak_allocated_gb": round(torch.cuda.max_memory_allocated(index) / 1e9, 3),
           "peak_reserved_gb": reserved}  # fmt: skip
    if _MEMORY_RULE.get("limit_gb") is not None:
        out["memory_rule_gb"] = _MEMORY_RULE["limit_gb"]
        out["within_memory_rule"] = reserved <= _MEMORY_RULE["limit_gb"]
    return out


def reset_peak(device: torch.device) -> None:
    gc.collect()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_index(device))


def numerics(device: torch.device) -> dict[str, Any]:
    return {"device": str(device), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "memory_rule": dict(_MEMORY_RULE) or None}  # fmt: skip


def kept(out: Path, relative: str, commit: str | None) -> dict[str, Any] | None:
    """An output already written under ``out``, if it was written at ``commit``; one written at
    another commit is moved aside to ``stale/<its commit>/`` (kept, never read) and None is
    returned, so the part runs again at this commit."""
    path = out / relative
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    written = output_commit(record)
    if written == commit:
        return record
    set_aside(out, relative, str(written))
    return None


def set_aside(out: Path, relative: str, label: str) -> None:
    """Move the output ``out/relative`` to ``out/stale/<label>/relative``: kept, never read."""
    target = out / "stale" / label / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(out / relative, target)


def output_commit(record: Mapping[str, Any] | None) -> str | None:
    """The code commit that wrote an output (``provenance.code.commit``)."""
    return (((record or {}).get("provenance") or {}).get("code") or {}).get("commit")


def output_dirty(record: Mapping[str, Any] | None) -> bool | None:
    """Whether the checkout that wrote an output was dirty (``provenance.code.dirty``)."""
    return (((record or {}).get("provenance") or {}).get("code") or {}).get("dirty")


# ------------------------------------------------------------------------- tensors and groups

_GROUPS = (
    ("embed_tokens", "embedding"), ("lm_head", "lm_head"),
    ("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"), ("self_attn.o_proj", "attn_o"),
    ("mlp.gate_proj", "mlp_gate"), ("mlp.up_proj", "mlp_up"), ("mlp.down_proj", "mlp_down"),
    ("layernorm", "norm"), ("model.norm.", "norm"),
)  # fmt: skip


def module_group(name: str) -> tuple[str, int | None]:
    """(kind, layer) of a parameter name: embedding, attn_q/k/v/o, mlp_gate/up/down, norm,
    lm_head or other, and its layer index if it has one."""
    layer = re.search(r"layers\.(\d+)\.", name)
    kind = next((kind for key, kind in _GROUPS if key in name), "other")
    return kind, int(layer.group(1)) if layer else None


def group_keys(name: str) -> tuple[str, str]:
    """The aggregate group (kind) and the per-layer group (kind.layer) of a parameter."""
    kind, layer = module_group(name)
    return kind, kind if layer is None else f"{kind}.{layer}"


def tensors_sha256(named: Sequence[tuple[str, torch.Tensor]]) -> str:
    """sha256 over names, shapes, dtypes and bytes, in order."""
    digest = hashlib.sha256()
    for name, tensor in named:
        t = tensor.detach().contiguous().cpu()
        digest.update(f"{name}|{tuple(t.shape)}|{t.dtype}|".encode())
        digest.update(t.view(torch.uint8).numpy() if t.numel() else b"")
    return digest.hexdigest()


def preconditioner(v: torch.Tensor, eps: float) -> torch.Tensor:
    """P = 1/(√v₀ + ε), with the optimizer's own denominator (``optim.Adam``, frozen mode)."""
    return v.sqrt().add_(eps).reciprocal_()


def sqrt_preconditioner(v: torch.Tensor, eps: float) -> torch.Tensor:
    """P^½ = (√v₀ + ε)^(−½)."""
    return v.sqrt().add_(eps).rsqrt_()


class LogHistogram:
    """Counts of log10 of non-negative values in 0.005-decade bins over [1e-45, 1e10), with the
    zeros apart; quantiles are read at the upper edge of their bin (≈ 1.2 % relative), the
    extremes exactly."""

    LOW, HIGH, BINS = -45.0, 10.0, 11000

    def __init__(self) -> None:
        self.counts = torch.zeros(self.BINS, dtype=torch.float64)
        self.zeros = self.total = 0
        self.min, self.max = math.inf, -math.inf

    def add(self, values: torch.Tensor) -> None:
        flat = values.detach().reshape(-1)
        positive = flat[flat > 0]
        self.total += flat.numel()
        self.zeros += flat.numel() - positive.numel()
        if positive.numel():
            logs = positive.float().log10().clamp_(self.LOW, self.HIGH - 1e-3)
            counts = torch.histc(logs, bins=self.BINS, min=self.LOW, max=self.HIGH)
            self.counts += counts.double().cpu()
            self.min = min(self.min, positive.min().item())
            self.max = max(self.max, positive.max().item())
        if self.zeros:
            self.min = 0.0

    def merge(self, other: LogHistogram) -> None:
        self.counts += other.counts
        self.zeros += other.zeros
        self.total += other.total
        self.min, self.max = min(self.min, other.min), max(self.max, other.max)

    def quantiles(self, levels: Sequence[float] = QUANTILES) -> dict[str, float | None]:
        if not self.total:
            return {f"{q:g}": None for q in levels}
        cumulative = self.zeros + torch.cumsum(self.counts, 0)
        width = (self.HIGH - self.LOW) / self.BINS
        out = {}
        for q in levels:
            if q <= 0:
                out[f"{q:g}"] = self.min
            elif q >= 1:
                out[f"{q:g}"] = self.max
            elif q * self.total <= self.zeros:
                out[f"{q:g}"] = 0.0
            else:
                index = int(torch.searchsorted(cumulative, torch.tensor(q * self.total)).item())
                edge = 10 ** (self.LOW + (min(index, self.BINS - 1) + 1) * width)
                out[f"{q:g}"] = min(max(edge, self.min), self.max)
        return out


def _index_stats(mask: torch.Tensor, top: int = 5) -> dict[str, Any]:
    """Where the coordinates of ``mask`` lie: rows and columns of a matrix, indices of a vector."""
    count = int(mask.sum())
    if not count:
        return {"count": 0}
    if mask.dim() == 2:
        rows, cols = mask.sum(1), mask.sum(0)

        def largest(counts: torch.Tensor) -> list[list[int]]:
            values, indices = torch.topk(counts, min(top, counts.numel()))
            return [[int(i), int(c)] for c, i in zip(values, indices, strict=True) if c > 0]

        return {"count": count, "rows_with_any": int((rows > 0).sum()),
                "cols_with_any": int((cols > 0).sum()), "top_rows": largest(rows),
                "top_cols": largest(cols)}  # fmt: skip
    indices = torch.nonzero(mask.reshape(-1))[:, 0]
    return {"count": count, "first_indices": indices[:20].tolist()}


@torch.no_grad()
def v0_census(names: Sequence[str], v0: Sequence[torch.Tensor], eps: float) -> dict[str, Any]:
    """D1's census of √v₀ over the trainable parameters: per tensor, per kind and per layer
    group, and overall — quantiles of √v₀, the fractions with √v₀ < ε and < 10ε (and = 0), the
    largest P = 1/(√v₀ + ε) and where it is."""
    groups: dict[str, dict[str, Any]] = {}

    def group(key: str) -> dict[str, Any]:
        return groups.setdefault(key, {"hist": LogHistogram(), "below_eps": 0,
                                       "below_10eps": 0, "numel": 0, "max_P": 0.0,
                                       "max_P_at": None})  # fmt: skip

    per_tensor = {}
    for name, v in zip(names, v0, strict=True):
        root = v.sqrt()
        p = preconditioner(v, eps)
        below, below10 = root < eps, root < 10 * eps
        flat_index = int(torch.argmax(p.reshape(-1)))
        where = list(np.unravel_index(flat_index, tuple(v.shape))) if v.dim() else []
        record = {
            "shape": list(v.shape), "numel": v.numel(), "below_eps": int(below.sum()),
            "below_10eps": int(below10.sum()), "zeros": int((v == 0).sum()),
            "min_sqrt_v0": root.min().item(), "max_sqrt_v0": root.max().item(),
            "max_P": p.reshape(-1)[flat_index].item(), "max_P_index": [int(i) for i in where],
            "below_10eps_where": _index_stats(below10),
        }  # fmt: skip
        per_tensor[name] = record
        hist = LogHistogram()
        hist.add(root)
        for key in dict.fromkeys((*group_keys(name), "all")):
            g = group(key)
            g["hist"].merge(hist)
            g["numel"] += record["numel"]
            g["below_eps"] += record["below_eps"]
            g["below_10eps"] += record["below_10eps"]
            if record["max_P"] > g["max_P"]:
                g["max_P"], g["max_P_at"] = record["max_P"], [name, record["max_P_index"]]
    summary = {}
    for key, g in sorted(groups.items()):
        summary[key] = {
            "numel": g["numel"], "below_eps": g["below_eps"], "below_10eps": g["below_10eps"],
            "fraction_below_eps": g["below_eps"] / g["numel"],
            "fraction_below_10eps": g["below_10eps"] / g["numel"],
            "zeros": g["hist"].zeros, "sqrt_v0_quantiles": g["hist"].quantiles(),
            "max_P": g["max_P"], "max_P_at": g["max_P_at"],
        }  # fmt: skip
    return {"epsilon": eps, "groups": summary, "per_tensor": per_tensor,
            "quantile_note": "0.005-decade log bins; extremes exact"}  # fmt: skip


@torch.no_grad()
def preconditioned_shares(
    names: Sequence[str], v0: Sequence[torch.Tensor], g: Sequence[torch.Tensor], eps: float
) -> dict[str, Any]:
    """How ‖P·g‖² splits: over √v₀ < ε, ε ≤ √v₀ < 10ε and √v₀ ≥ 10ε, and over module kinds;
    quantiles of P·|g| and its largest entry."""
    classes = {"below_eps": 0.0, "eps_to_10eps": 0.0, "above_10eps": 0.0}
    kinds: dict[str, float] = {}
    hist, top = LogHistogram(), (0.0, None)
    for name, v, grad in zip(names, v0, g, strict=True):
        root = v.sqrt()
        u = (grad.double() * preconditioner(v, eps).double()).abs()
        sq = u.pow(2)
        below, below10 = root < eps, root < 10 * eps
        classes["below_eps"] += sq[below].sum().item()
        classes["eps_to_10eps"] += sq[below10 & ~below].sum().item()
        classes["above_10eps"] += sq[~below10].sum().item()
        kind = module_group(name)[0]
        kinds[kind] = kinds.get(kind, 0.0) + sq.sum().item()
        hist.add(u)
        largest = u.max().item()
        if largest > top[0]:
            index = int(torch.argmax(u.reshape(-1)))
            where = list(np.unravel_index(index, tuple(u.shape))) if u.dim() else []
            top = (largest, [name, [int(i) for i in where]])
    total = sum(classes.values())
    share = (lambda x: x / total) if total > 0 else (lambda x: None)
    return {"norm": total**0.5, "share_by_sqrt_v0": {k: share(v) for k, v in classes.items()},
            "share_by_kind": {k: share(v) for k, v in sorted(kinds.items())},
            "P_abs_g_quantiles": hist.quantiles(), "max_P_abs_g": top[0],
            "max_P_abs_g_at": top[1]}  # fmt: skip


def vector_shares(
    names: Sequence[str],
    x: Sequence[torch.Tensor],
    v0: Sequence[torch.Tensor],
    eps: float,
    *,
    in_theta: bool = False,
) -> dict[str, Any]:
    """The share of ‖x‖² per module kind, per layer group and on ε-dominated coordinates; with
    ``in_theta``, of ‖P^½x‖² (a vector of P^½-coordinates taken back to θ's), tensor by tensor."""
    kinds: dict[str, float] = {}
    layers: dict[str, float] = {}
    eps_class = {"below_eps": 0.0, "below_10eps": 0.0}
    total = 0.0
    for name, xi, v in zip(names, x, v0, strict=True):
        xi = xi.to(v.device).double()
        if in_theta:
            xi = xi * sqrt_preconditioner(v, eps).double()
        sq = xi.pow(2)
        value = sq.sum().item()
        total += value
        kind, layer = group_keys(name)
        kinds[kind] = kinds.get(kind, 0.0) + value
        layers[layer] = layers.get(layer, 0.0) + value
        root = v.sqrt()
        eps_class["below_eps"] += sq[root < eps].sum().item()
        eps_class["below_10eps"] += sq[root < 10 * eps].sum().item()
    if total <= 0:
        return {"total": total}
    top_layers = sorted(layers.items(), key=lambda kv: -kv[1])[:10]
    return {"total": total, "share_by_kind": {k: v / total for k, v in sorted(kinds.items())},
            "top_layer_groups": [[k, v / total] for k, v in top_layers],
            "share_below_eps": eps_class["below_eps"] / total,
            "share_below_10eps": eps_class["below_10eps"] / total}  # fmt: skip


def cosine(xs: Sequence[torch.Tensor], ys: Sequence[torch.Tensor]) -> float | None:
    nx, ny = _norm(xs), _norm(ys)
    return None if nx == 0 or ny == 0 else _dot(xs, ys) / (nx * ny)


# ------------------------------------------------------------------------- gradients and losses


def take_gradients(params: Sequence[torch.nn.Parameter]) -> Vector:
    """The parameters' gradients, released from the parameters."""
    out = []
    for p in params:
        out.append(p.grad.detach())
        p.grad = None
    return out


class Float64Gradients:
    """Within the context, each parameter's gradient of every backward pass is added into a
    float64 sum as soon as it is accumulated, and the fp32 gradient is released: the gradient
    accumulation across micro-batches in float64, without an fp32 gradient set beside it."""

    def __init__(self, params: Sequence[torch.nn.Parameter]) -> None:
        self.params = list(params)
        self.sums: list[torch.Tensor | None] = [None] * len(self.params)
        self._handles: list[Any] = []

    def __enter__(self) -> Float64Gradients:
        for i, p in enumerate(self.params):
            self._handles.append(p.register_post_accumulate_grad_hook(self._hook(i)))
        return self

    def _hook(self, i: int) -> Callable[[torch.Tensor], None]:
        def hook(param: torch.Tensor) -> None:
            grad = param.grad
            if self.sums[i] is None:
                self.sums[i] = grad.double()
            else:
                self.sums[i].add_(grad)
            param.grad = None

        return hook

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def take(self) -> Vector:
        out = [s if s is not None else torch.zeros_like(p, dtype=torch.float64)
               for s, p in zip(self.sums, self.params, strict=True)]  # fmt: skip
        self.sums = [None] * len(self.params)
        return out

    def to_grad(self) -> None:
        """Round each float64 sum once into the parameter's fp32 ``.grad``, freeing the sums one
        at a time."""
        for i, p in enumerate(self.params):
            total = self.sums[i]
            p.grad = torch.zeros_like(p) if total is None else total.to(p.dtype)
            self.sums[i] = None


def training_path_targets(
    model: PreTrainedModel, samples: Sequence[Sample], cfg: DictConfig, env: Environment
) -> list[torch.Tensor]:
    """θ₀'s fp32 log-probabilities of each sample through the training path (``model`` at θ₀,
    no gradient), micro-batch by micro-batch as ``accumulate_gradient`` batches them: the
    same-path soft targets of Revision 1, (i)."""
    cap, micro = cfg.train.max_response_tokens, cfg.execution.micro_batch_size
    out = []
    with torch.no_grad():
        for chunk in chunks(samples, micro):
            rows = collate(chunk, pad_id=env.pad_id, max_response_tokens=cap, device=env.device)
            out += list(response_logprobs(model, rows).split(rows.lengths))
    return out


def soft_loss_float64(
    student: PreTrainedModel,
    reference: PreTrainedModel,
    samples: Sequence[Sample],
    cfg: DictConfig,
    env: Environment,
) -> dict[str, float]:
    """The soft-target loss with training-path targets (``reference`` at θ₀) and its gradient,
    with the loss in float64 (the forward and backward in fp32): Σ KL(target ‖ student) over
    the response positions / the training denominator, added to ``.grad`` (or to the sums of a
    ``Float64Gradients`` context)."""
    cap, micro = cfg.train.max_response_tokens, cfg.execution.micro_batch_size
    denominator = loss_denominator(cfg.train.loss, samples, cap)
    loss, tokens = 0.0, 0
    for chunk in chunks(samples, micro):
        rows = collate(chunk, pad_id=env.pad_id, max_response_tokens=cap, device=env.device)
        with torch.no_grad():
            target = torch.log_softmax(response_logits(reference, rows).double(), dim=-1)
        student_lp = torch.log_softmax(response_logits(student, rows).double(), dim=-1)
        term = F.kl_div(student_lp, target, log_target=True, reduction="sum")
        (term / denominator).backward()
        loss += term.item() / denominator
        tokens += rows.num_tokens
        del target, student_lp, term
    return {"loss": loss, "denominator": denominator, "tokens": tokens}


def nll_loss_float64(
    student: PreTrainedModel, samples: Sequence[Sample], cfg: DictConfig, env: Environment
) -> dict[str, float]:
    """``accumulate_gradient``'s NLL with the loss in float64 (forward and backward in fp32)."""
    cap, micro = cfg.train.max_response_tokens, cfg.execution.micro_batch_size
    denominator = loss_denominator(cfg.train.loss, samples, cap)
    loss = nll = 0.0
    tokens = 0
    for chunk in chunks(samples, micro):
        rows = collate(chunk, pad_id=env.pad_id, max_response_tokens=cap, device=env.device)
        term = F.cross_entropy(response_logits(student, rows).double(), target_ids(rows),
                               reduction="sum")  # fmt: skip
        (term / denominator).backward()
        loss += term.item() / denominator
        nll += term.item()
        tokens += rows.num_tokens
    return {"loss": loss, "denominator": denominator, "tokens": tokens,
            "nll_per_token": nll / tokens}  # fmt: skip


class Float64Adam:
    """``optim.Adam``'s arithmetic in float64 on fp32 parameters: the first moment (and, adaptive
    or pre-warmed, the second) in float64, the update formed in float64 and rounded once into the
    fp32 parameter. Frozen mode reads v₀ as stored (fp32) and widens it exactly. Pre-warmed mode
    starts its second moment at v₀ widened exactly (float64) and updates it as
    ``optim.Adam(second_moment="prewarmed")`` does: v ← β₂v + (1−β₂)g², with no bias correction
    of v (a full estimate from the start), the first moment bias-corrected as in every mode. The
    parameters stay fp32, since the forward is fp32: no float64 master copy."""

    def __init__(self, params: Sequence[torch.nn.Parameter], opt_cfg: DictConfig,
                 second_moment: Sequence[torch.Tensor] | None = None) -> None:  # fmt: skip
        self.params = list(params)
        self.mode = ADAM_MODES[opt_cfg.name]
        self.lr, self.eps = float(opt_cfg.lr), float(opt_cfg.eps)
        if self.mode == "frozen":
            self.beta1, self.beta2 = float(opt_cfg.beta1), None
            if second_moment is None:
                raise DiagnosticError("Frozen Adam needs v₀")
            self.v = [v.to(p.device) for v, p in zip(second_moment, self.params, strict=True)]
        elif self.mode == "prewarmed":
            self.beta1, self.beta2 = (float(b) for b in opt_cfg.betas)
            if second_moment is None:
                raise DiagnosticError("Pre-warmed Adam needs v₀")
            # a copy: v is updated in place, and v₀ is shared by every twin of the seed
            self.v = [v.to(p.device, torch.float64).clone()
                      for v, p in zip(second_moment, self.params, strict=True)]  # fmt: skip
        else:
            self.beta1, self.beta2 = (float(b) for b in opt_cfg.betas)
            self.v = [torch.zeros_like(p, dtype=torch.float64) for p in self.params]
        self.m = [torch.zeros_like(p, dtype=torch.float64) for p in self.params]
        self.step_count = 0

    @torch.no_grad()
    def step(self, grads: Sequence[torch.Tensor]) -> None:
        self.step_count += 1
        t, b1 = self.step_count, self.beta1
        for p, g, m, v in zip(self.params, grads, self.m, self.v, strict=True):
            m.lerp_(g, 1 - b1)
            if self.mode == "adaptive":
                v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                denom = (v.sqrt() / (1 - self.beta2**t) ** 0.5).add_(self.eps)
            elif self.mode == "prewarmed":
                v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
                denom = v.sqrt().add_(self.eps)
            else:
                denom = v.double().sqrt().add_(self.eps)
            p.copy_(p.double().addcdiv_(m, denom, value=-self.lr / (1 - b1**t)))


# ------------------------------------------------------------------------ Hessian–vector products


def loss_products(
    model: PreTrainedModel,
    params: Sequence[torch.nn.Parameter],
    terms: Sequence[Term],
    vector: Sequence[torch.Tensor],
    consume: Callable[[Sequence[torch.Tensor]], None],
) -> None:
    """Pass H_term·v of each loss term in turn to ``consume``; H·v of the sum of the terms is
    their sum. The terms are micro-batch losses (``batch_terms``), so the product is exact. As
    ``curvature``'s products: double backward on PyTorch's math attention kernel, the
    first-order gradients released before the second backward."""
    with sdpa_kernel(SDPBackend.MATH):
        for term in terms:
            loss = term()
            grads = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((g * v).sum() for g, v in zip(grads, vector, strict=True))
            del grads
            consume(torch.autograd.grad(dot, params))
            del dot, loss


def batch_terms(
    model: PreTrainedModel,
    batches: Sequence[Sequence[Sample]],
    cfg: DictConfig,
    env: Environment,
    *,
    kind: str,
    soft_targets: Sequence[Sequence[torch.Tensor]] | None = None,
) -> list[Term]:
    """The micro-batch loss terms of the mean over ``batches`` of each batch's training loss,
    normalized as training (``losses.loss_denominator`` of the whole batch): ``nll`` the summed
    NLL, ``soft`` the summed KL to the batch's soft targets (one (length, vocab) row per
    sample, as ``accumulate_gradient`` takes them)."""
    if kind not in ("nll", "soft"):
        raise ValueError(f"Unknown loss kind {kind!r}")
    cap, micro = cfg.train.max_response_tokens, cfg.execution.micro_batch_size
    out: list[Term] = []
    for b, batch in enumerate(batches):
        denominator = loss_denominator(cfg.train.loss, batch, cap) * len(batches)
        for start in range(0, len(batch), micro):
            chunk = batch[start : start + micro]

            def term(chunk=chunk, start=start, b=b, denominator=denominator) -> torch.Tensor:
                rows = collate(chunk, pad_id=env.pad_id, max_response_tokens=cap,
                               device=env.device)  # fmt: skip
                if kind == "nll":
                    return summed_nll(model, rows) / denominator
                target = torch.cat([t[:cap] for t in soft_targets[b][start : start + micro]])
                return soft_kl(model, rows, target.to(env.device)) / denominator

            out.append(term)
    return out


def preconditioned_hvp(
    model: PreTrainedModel,
    params: Sequence[torch.nn.Parameter],
    terms: Sequence[Term],
    v0: Sequence[torch.Tensor],
    eps: float,
) -> Callable[[Vector, Vector], Vector]:
    """``hvp(v, out)`` for ``curvature.lanczos``: adds P^½ H P^½ v into ``out``, P = 1/(√v₀+ε).
    P^½ is formed per tensor as needed from v₀, and v is scaled by it in place for the products
    and scaled back afterwards (exact up to one rounding of v's entries), so that no vector
    beyond Lanczos's own is needed: at 1.5B every trainable-sized vector is ≈ 5.2 GB."""

    def consume(out: Vector) -> Callable[[Sequence[torch.Tensor]], None]:
        def add(part: Sequence[torch.Tensor]) -> None:
            for o, w, h in zip(out, v0, part, strict=True):
                o.addcmul_(sqrt_preconditioner(w, eps), h)

        return add

    def hvp(v: Vector, out: Vector) -> Vector:
        with torch.no_grad():
            for x, w in zip(v, v0, strict=True):
                x.mul_(sqrt_preconditioner(w, eps))
        try:
            loss_products(model, params, terms, v, consume(out))
        finally:
            with torch.no_grad():
                for x, w in zip(v, v0, strict=True):
                    x.div_(sqrt_preconditioner(w, eps))
        return out

    return hvp


def quadratic_form(
    model: PreTrainedModel,
    params: Sequence[torch.nn.Parameter],
    terms: Sequence[Term],
    vector: Sequence[torch.Tensor],
) -> float:
    """vᵀHv in float64, summed over the terms as each product is formed."""
    total = 0.0

    def consume(part: Sequence[torch.Tensor]) -> None:
        nonlocal total
        total += _dot(vector, part)

    loss_products(model, params, terms, vector, consume)
    return total


def ritz_vector(
    hvp: Callable[[Vector, Vector], Vector],
    start: Vector,
    result: Mapping[str, Any],
    *,
    host: bool = True,
) -> Vector:
    """The top Ritz vector of a ``lanczos`` result, by a second pass of the same recurrence from
    the same start with the first pass's coefficients: y = Σ_j s_j v_j, s the top eigenvector of
    T_k. ``start`` is consumed. y is accumulated beside the vectors, or with ``host`` (on a GPU)
    in host memory through one pinned staging buffer the size of the largest tensor."""
    alphas, betas = list(result["alphas"]), list(result["betas"])
    k = len(alphas)
    _, vectors = _ritz(alphas, betas[: k - 1])
    s = vectors[:, -1]
    device = start[0].device
    staging = None
    if host and device.type == "cuda":
        staging = torch.empty(max(t.numel() for t in start), dtype=torch.float32, pin_memory=True)
    where = torch.device("cpu") if staging is not None else device
    y = [torch.zeros(t.shape, dtype=torch.float32, device=where) for t in start]
    v, previous = list(start), None
    start.clear()
    for j in range(k):
        for yi, vi in zip(y, v, strict=True):
            if staging is None:
                yi.add_(vi, alpha=float(s[j]))
            else:
                buffer = staging[: vi.numel()].view(vi.shape)
                buffer.copy_(vi)  # device to pinned host memory, synchronous
                yi.add_(buffer, alpha=float(s[j]))
        if j == k - 1:
            break
        if previous is None:
            w = hvp(v, [torch.zeros_like(t) for t in v])
        else:
            w = hvp(v, [t.mul_(-betas[j - 1]) for t in previous])
        for wi, vi in zip(w, v, strict=True):
            wi.sub_(vi, alpha=alphas[j]).div_(betas[j])
        previous, v = v, w
    del v, previous
    return y


def check_vector(hvp: Callable[[Vector, Vector], Vector], y: Vector, device: torch.device) -> dict:
    """The Rayleigh quotient of y and its relative residual ‖Ay − θy‖ / (|θ|‖y‖)."""
    y = [t.to(device) for t in y]
    ay = hvp(y, [torch.zeros_like(t) for t in y])
    norm_sq = _dot(y, y)
    theta = _dot(y, ay) / norm_sq
    for a, t in zip(ay, y, strict=True):
        a.sub_(t, alpha=theta)
    residual = _norm(ay) / (abs(theta) * norm_sq**0.5) if theta else math.inf
    return {"rayleigh": theta, "relative_residual": residual}


# --------------------------------------------------------------------- the linearized recurrence


def heavy_ball_threshold(eta: float, beta: float) -> float:
    """2(1+β)/((1−β)η): the asymptotic stability bound on λ of the code's EMA recurrence."""
    return 2 * (1 + beta) / ((1 - beta) * eta)


def stationary_radius(eta_lambda: float, beta: float) -> float:
    """Spectral radius of the stationary map (bias correction → 1) of one eigenmode:
    μ² − (1 + β − (1−β)ηλ) μ + β = 0."""
    roots = np.roots([1.0, -(1 + beta - (1 - beta) * eta_lambda), beta])
    return float(np.max(np.abs(roots)))


def scalar_mode(eta: float, lam: float, beta: float, steps: int) -> list[float]:
    """|z_t|, t = 1 … steps, of one eigenmode (eigenvalue λ of P^½ H P^½) under the code's
    bias-corrected recurrence, started as the vector map is: after the first step z₁ = 1 and
    n₁ = −(1−β)/η (m₁ = (1−β)g₁, δ₁ = −ηPg₁); then n_t = βn_{t−1} + (1−β)λz_{t−1} and
    z_t = z_{t−1} − η n_t / (1−β^t)."""
    z, n = 1.0, -(1 - beta) / eta
    out = [1.0]
    for t in range(2, steps + 1):
        n = beta * n + (1 - beta) * lam * z
        z = z - eta / (1 - beta**t) * n
        out.append(abs(z))
    return out


def per_step(values: Mapping[int, float], *, power: float = 1.0) -> list[dict[str, Any]]:
    """Per-step growth between consecutive recorded steps: (x₂/x₁)^(power/Δ)."""
    steps = sorted(values)
    out = []
    for a, b in zip(steps, steps[1:], strict=False):
        x1, x2 = values[a], values[b]
        ok = x1 > 0 and x2 > 0 and math.isfinite(x1) and math.isfinite(x2)
        out.append({"from": a, "to": b,
                    "per_step": (x2 / x1) ** (power / (b - a)) if ok else None})  # fmt: skip
    return out


def observed_growth(monitor_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A control's observed per-step growth of δ between monitor readings: from K
    (√ of K's per-step growth, K ∝ ‖δ‖²) and from the recorded displacement norm."""
    k = {int(r["step"]): float(r["K_monitor"]) for r in monitor_rows}
    d = {int(r["step"]): float(r["displacement_norm"]) for r in monitor_rows
         if r.get("displacement_norm") is not None}  # fmt: skip
    return {"K_monitor": k, "displacement_norm": d,
            "delta_per_step_from_K": per_step(k, power=0.5),
            "delta_per_step_from_displacement": per_step(d)}  # fmt: skip


def linearized_run(
    model: PreTrainedModel,
    params: Sequence[torch.nn.Parameter],
    v0: Sequence[torch.Tensor],
    residual: Sequence[torch.Tensor],
    *,
    eta: float,
    beta: float,
    eps: float,
    steps: int,
    step_terms: Sequence[Sequence[Term]],
    fisher_terms: Sequence[Term],
    fisher_at: Sequence[int],
) -> dict[str, Any]:
    """The code's recurrence (optim.Adam, frozen mode) linearized at θ₀ along a run of ``steps``
    steps whose batch at step t is ``step_terms[(t−1) mod k]``: g_t = H_{b(t)} δ_{t−1},
    m_t = β m_{t−1} + (1−β) g_t, δ_t = δ_{t−1} − η P m_t / (1−β^t). It starts from the measured
    residual r as the code's first step does: m₁ = (1−β) r, δ₁ = −η P r. The map is linear and
    homogeneous, so it runs on δ₁ scaled to unit norm; ‖δ_t‖ is recorded in units of the true
    ‖δ₁‖ = η‖P r‖ too, and δ_tᵀ F δ_t at ``fisher_at``."""
    scale = None
    m, delta = [], []
    with torch.no_grad():
        for r, v in zip(residual, v0, strict=True):
            m.append(r.to(v.device).mul(1 - beta))
            delta.append(m[-1] * preconditioner(v, eps) * (-eta / (1 - beta)))
        scale = _norm(delta)
        if not scale > 0:
            raise DiagnosticError("The first step's displacement is zero: nothing to propagate")
        for x in (*m, *delta):
            x.div_(scale)
    norms = {1: 1.0}
    fisher: dict[int, float] = {}
    if 1 in fisher_at:
        fisher[1] = quadratic_form(model, params, fisher_terms, delta)
    for t in range(2, steps + 1):
        with torch.no_grad():
            for x in m:
                x.mul_(beta)

        def consume(part: Sequence[torch.Tensor]) -> None:
            with torch.no_grad():
                for x, h in zip(m, part, strict=True):
                    x.add_(h, alpha=1 - beta)

        loss_products(model, params, step_terms[(t - 1) % len(step_terms)], delta, consume)
        with torch.no_grad():
            for x, mi, v in zip(delta, m, v0, strict=True):
                x.addcmul_(mi, preconditioner(v, eps), value=-eta / (1 - beta**t))
        norms[t] = _norm(delta)
        if not math.isfinite(norms[t]):
            break
        if t in fisher_at:
            fisher[t] = quadratic_form(model, params, fisher_terms, delta)
    return {"eta": eta, "first_step_norm": scale, "delta_norm_relative": norms,
            "delta_norm": {t: n * scale for t, n in norms.items()},
            "delta_F_delta_relative": {t: q / fisher[1] for t, q in fisher.items()}
            if fisher.get(1) else None,
            "delta_per_step": per_step(norms),
            "delta_per_step_from_F": per_step(fisher, power=0.5) if fisher else None}  # fmt: skip


# ------------------------------------------------------------------------------- D3: twins


@torch.no_grad()
def twin_divergence(
    model_a: PreTrainedModel,
    model_b: PreTrainedModel,
    samples: Sequence[Sample],
    *,
    pad_id: int,
    micro_batch_size: int,
) -> dict[str, Any]:
    """D_twin = max(KL(A‖B), KL(B‖A)): per response, the summed per-position KL between the full
    next-token distributions along the fixed continuations, averaged over the responses as
    K_benign is (``evaluation.sequence_kl``'s arithmetic: float64 log-softmax of fp32 logits;
    KL(A‖B) is ``sequence_kl(B, A)``). One forward pass of each model gives both directions."""
    device = next(model_b.parameters()).device
    ab, ba = [], []
    for chunk in chunks(samples, micro_batch_size):
        batch = collate(chunk, pad_id=pad_id, max_response_tokens=NO_CAP, device=device)
        log_a = torch.log_softmax(response_logits(model_a, batch).double(), dim=-1)
        log_b = torch.log_softmax(response_logits(model_b, batch).double(), dim=-1)
        per_ab = (log_a.exp() * (log_a - log_b)).sum(-1)
        per_ba = (log_b.exp() * (log_b - log_a)).sum(-1)
        ab += [part.sum() for part in per_ab.split(batch.lengths)]
        ba += [part.sum() for part in per_ba.split(batch.lengths)]
    kl_ab, kl_ba = torch.stack(ab).cpu(), torch.stack(ba).cpu()
    mean_ab, mean_ba = kl_ab.mean().item(), kl_ba.mean().item()
    finite = math.isfinite(mean_ab) and math.isfinite(mean_ba)
    return {"D_twin": max(mean_ab, mean_ba) if finite else math.inf, "KL_A_B": mean_ab,
            "KL_B_A": mean_ba, "max_response_KL_A_B": kl_ab.max().item(),
            "max_response_KL_B_A": kl_ba.max().item(), "responses": len(ab)}  # fmt: skip


def operational_monitor(cfg: DictConfig) -> DictConfig:
    """The run's config with the monitor's second responses uncached: only the operational
    reading (the first response of each prompt, cached as the runs cached it) is taken. The
    primary reading must be cached: the monitor's reference model is later given other states
    (D3 loads twin A's checkpoints into it), so an uncached reading would read against them."""
    out = copy.deepcopy(cfg)
    caches = out.eval.monitor.get("cache") or {}
    if caches.get("primary", "none") == "none":
        raise DiagnosticError(
            "The monitor's primary reading must be cached (eval.monitor.cache.primary is "
            "none): its reference model does not stay at θ₀"
        )
    out.eval.monitor.cache.secondary = "none"
    return out


def scoring_samples(cfg: DictConfig, env: Environment) -> tuple[list[Sample], dict[str, Any]]:
    """The scoring panel's benign continuations, limited as ``train.Evaluator`` limits them."""
    samples, manifest = load_stream_for(cfg, env, "kl_benign", None)
    limit, per_prompt = cfg.eval.kl.limit, cfg.eval.kl.samples_per_prompt
    chosen = samples[: limit * per_prompt] if limit else samples
    return chosen, {key: manifest[key] for key in ("path", "sha256", "count")}


@torch.no_grad()
def set_trainable(
    model: PreTrainedModel, values: Mapping[str, torch.Tensor], names: Sequence[str]
) -> None:
    parameters = dict(model.named_parameters())
    for name in names:
        parameters[name].copy_(values[name])


def save_trainable(trainable: Sequence[tuple[str, torch.nn.Parameter]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    torch.save({name: p.detach().cpu() for name, p in trainable}, partial)
    os.replace(partial, path)


def load_trainable(model: PreTrainedModel, path: Path, names: Sequence[str]) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    set_trainable(model, state, names)
    del state


def gaussian(name: str, shape: Sequence[int], seed: int, key: str) -> torch.Tensor:
    """A fixed N(0, 1) tensor for one parameter, drawn on the CPU from a named seed."""
    generator = torch.Generator().manual_seed(
        derive_seed(seed, "fa_control_diagnostic", key, name, list(shape))
    )
    return torch.randn(tuple(shape), generator=generator)


def gaussian_norm(trainable: Sequence[tuple[str, torch.Tensor]], seed: int, key: str) -> float:
    """‖·‖ of the fixed Gaussian direction ``key`` over the trainable parameters (float64 sums)."""
    return math.sqrt(sum(gaussian(n, p.shape, seed, key).double().pow(2).sum().item()
                         for n, p in trainable))  # fmt: skip


@torch.no_grad()
def perturb(
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    theta0: Mapping[str, torch.Tensor],
    direction: Callable[[str, Sequence[int]], torch.Tensor],
    scale: float,
    *,
    survival_share: float = 0.5,
) -> dict[str, Any]:
    """Set the trainable parameters to θ₀ + scale·direction in fp32 and record what survived
    the rounding: the intended norm, the realized norm ‖fp32(θ₀ + p) − θ₀‖ and the coordinates
    that differ from θ₀. The perturbation survives when the realized norm is at least
    ``survival_share`` of the intended one (Revision 2, 4; Revision 3, 2)."""
    changed = 0
    realized = intended = 0.0
    numel = 0
    for name, p in trainable:
        start = theta0[name].to(p.device)
        d = direction(name, tuple(p.shape)).to(p.device, torch.float32) * scale
        p.copy_(start + d)
        changed += int((p != start).sum())
        numel += p.numel()
        realized += (p.double() - start.double()).pow(2).sum().item()
        intended += d.double().pow(2).sum().item()
    realized, intended = realized**0.5, intended**0.5
    return {"scale": scale, "changed": changed, "fraction_changed": changed / numel,
            "intended_norm": intended, "realized_norm": realized,
            "survived": realized >= survival_share * intended}  # fmt: skip


def apply_perturbation(
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    theta0: Mapping[str, torch.Tensor],
    direction: Callable[[str, Sequence[int]], torch.Tensor],
    scale: float,
    *,
    survival_share: float,
    rounding_scale: float,
) -> list[dict[str, Any]]:
    """Revision 3, 2 (correcting Revision 2, 4): perturb θ₀; if the perturbation does not survive
    fp32 rounding, scale it by ``rounding_scale`` once. Each attempt is judged against its own
    intended norm. Returns every attempt's record; the parameters hold the last attempt. If that
    one did not survive either, the pair can never be within the bound (``pair_state``)."""
    attempts = [perturb(trainable, theta0, direction, scale, survival_share=survival_share)]
    if not attempts[-1]["survived"]:
        attempts.append(perturb(trainable, theta0, direction, scale * rounding_scale,
                                survival_share=survival_share))  # fmt: skip
    return attempts


def compare_archive(
    rows: Sequence[Mapping[str, Any]], readings: Sequence[Mapping[str, Any]], run_dir: Path
) -> dict[str, Any]:
    """The re-run's per-step metrics and monitor K against the archived run's, bit for bit."""
    archived = [r for r in read_jsonl(run_dir / "metrics.jsonl") if "loss" in r]
    monitor = read_jsonl(run_dir / "monitor.jsonl")
    keys = ("loss", "grad_norm", "nll_per_token", "tokens", "denominator")
    mismatches = []
    if len(archived) != len(rows):
        mismatches.append({"what": "training steps", "rerun": len(rows), "archived": len(archived)})
    for mine, theirs in zip(rows, archived, strict=False):
        for key in keys:
            if key in theirs and mine.get(key) != theirs[key]:
                mismatches.append({"step": mine["step"], "key": key, "rerun": mine.get(key),
                                   "archived": theirs[key]})  # fmt: skip
    archived_k = {int(r["step"]): r["K_monitor"] for r in monitor}
    rerun_k = {int(r["step"]): r["K_monitor"] for r in readings}
    if set(archived_k) != set(rerun_k):
        mismatches.append({"what": "monitor steps", "rerun": sorted(rerun_k),
                           "archived": sorted(archived_k)})  # fmt: skip
    for step in sorted(set(archived_k) & set(rerun_k)):
        if archived_k[step] != rerun_k[step]:
            mismatches.append({"step": step, "key": "K_monitor", "rerun": rerun_k[step],
                               "archived": archived_k[step]})  # fmt: skip
    return {"run": str(run_dir), "equal": not mismatches, "steps_compared": len(archived),
            "monitor_compared": len(archived_k), "mismatches": mismatches[:50]}  # fmt: skip


@dataclass
class Twins:
    """What every twin of one D3 process shares: the student, the second model (θ₀ until a
    checkpoint is loaded into it; in host memory between measurements), θ₀'s trainable
    parameters (host), v₀ (host), the stream and the scoring panel."""

    env: Environment
    student: PreTrainedModel
    other: PreTrainedModel
    trainable: list[tuple[str, torch.nn.Parameter]]
    theta0: dict[str, torch.Tensor]
    v0: list[torch.Tensor]
    prefix: StreamPrefix
    scoring: list[Sample]
    k_star: float

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.trainable]

    @property
    def params(self) -> list[torch.nn.Parameter]:
        return [p for _, p in self.trainable]

    def reset(self) -> None:
        for p in self.params:
            p.grad = None
        set_trainable(self.student, self.theta0, self.names)

    def divergence(self, checkpoint: Path, cfg: DictConfig) -> dict[str, Any]:
        """D_twin between the checkpoint (A_t) and the student (B_t), and D_twin/K*."""
        load_trainable(self.other, checkpoint, self.names)
        self.other.to(self.env.device)
        try:
            out = twin_divergence(self.other, self.student, self.scoring, pad_id=self.env.pad_id,
                                  micro_batch_size=cfg.execution.micro_batch_size)  # fmt: skip
        finally:
            self.other.to("cpu")
        return {**out, "D_twin_over_K_star": out["D_twin"] / self.k_star}


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def run_twin(
    cfg: DictConfig,
    twins: Twins,
    *,
    eval_at: Sequence[int],
    precision: str = "fp32",
    monitor: Monitor | None = None,
    save_dir: Path | None = None,
    compare_dir: Path | None = None,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One twin: the regular run of ``cfg`` from the student's current parameters, step for
    step as ``schedule.run_scheduled`` trains (no evaluation, no stop rule), for the archived
    run's ``train.steps``. ``precision``: ``fp32`` is the run's own arithmetic; ``float64`` and
    ``float64_accumulation`` are the reference (loss and gradient accumulation in float64, and
    the optimizer arithmetic too in ``float64``). At the steps of ``eval_at`` the trainable
    parameters go to ``save_dir`` (twin A) and/or D_twin is measured against ``compare_dir``'s
    checkpoints; the monitor is read every ``eval.monitor.every`` steps if given. ``progress``,
    if given, holds the rows and readings as they are taken, so that a failed attempt's record
    survives its exception."""
    env, params, device = twins.env, twins.params, twins.env.device
    steps, size = int(cfg.train.steps), int(cfg.train.batch_size)
    seed_everything(derive_seed(cfg.seed, "torch"))
    reset_peak(device)
    started = time.perf_counter()
    # v₀ for frozen and pre-warmed Adam, as schedule.run_scheduled passes it
    v0 = twins.v0 if ADAM_MODES.get(cfg.optimizer.name, "adaptive") != "adaptive" else None
    if precision == "float64":
        optimizer: Any = Float64Adam(params, cfg.optimizer, v0)
    else:
        optimizer = build_optimizer(cfg.optimizer, params, second_moment=v0)
    rows, readings, divergence = [], [], []
    if progress is not None:
        progress.update({"rows": rows, "monitor": readings, "D_twin": divergence})
    stop, taken, diverged = None, 0, False
    every = int(cfg.eval.monitor.every)
    for step in range(steps):
        batch = twins.prefix.take(step * size, (step + 1) * size)
        for p in params:
            p.grad = None
        if precision == "fp32":
            stats = accumulate_gradient(twins.student, batch, cfg, env)
            stats["grad_norm"] = grad_norm([p.grad for p in params])
        else:
            with Float64Gradients(params) as sums:
                stats = nll_loss_float64(twins.student, batch, cfg, env)
            if precision == "float64":
                grads = sums.take()
                stats["grad_norm"] = math.sqrt(sum(g.pow(2).sum().item() for g in grads))
            else:
                sums.to_grad()
                stats["grad_norm"] = grad_norm([p.grad for p in params])
        rows.append({"step": step, **stats})
        if not _finite(stats["loss"], stats["grad_norm"]):
            stop, diverged = f"non-finite loss or gradient at step {step}", True
            break
        if precision == "float64":
            optimizer.step(grads)
            del grads
        else:
            optimizer.step()
        taken = t = step + 1
        if not math.isfinite(grad_norm(params)):
            stop, diverged = f"non-finite weights after step {step}", True
            break
        if monitor is not None and t % every == 0:
            readings.append({"step": t, "K_monitor": monitor(twins.student)})
        if t in eval_at:
            if save_dir is not None:
                save_trainable(twins.trainable, save_dir / f"A-step-{t}.pt")
            if compare_dir is not None:
                for p in params:
                    p.grad = None
                divergence.append({"step": t, **twins.divergence(
                    compare_dir / f"A-step-{t}.pt", cfg)})  # fmt: skip
    for p in params:
        p.grad = None
    del optimizer
    # Both stops above are a divergence (Revision 3, 3): a non-finite loss, gradient or weight.
    out = {"steps": taken, "stop": stop, "diverged": diverged, "rows": rows,
           "seconds": time.perf_counter() - started, **peak_memory(device)}  # fmt: skip
    if monitor is not None:
        out["monitor"] = readings
    if compare_dir is not None:
        out["D_twin"] = divergence
        out.update(_pair_summary(divergence, eval_at, stop))
        out["max_D_twin_over_K_star"] = _over(out["max_D_twin"], twins.k_star)
    return out


# ----------------------------------------------- the states of the measurements (Revision 3)

EXCEEDING, WITHIN, INCOMPLETE = "exceeding", "within", "incomplete"  # a D3 pair
BELOW, AT_BOUND = "below", "at_bound"  # the same-path control: also exceeding or incomplete
HOLDS, FALSE, UNDECIDED = "holds", "false", "undecided"  # the path criterion
# Revision 3, 1: the D1 checks that gate every branch, and what a failed one means
SHARED_STATE_CHECKS = {
    "trainable_parameters": "the trainable parameter sets differ",
    "v0": "the v₀ hashes under the two configs differ",
    "loss_normalization": "the loss normalizations differ",
    "initial_optimizer_state": "the initial optimizer states differ",
    "iii_equals_record": "the recomputed (iii) differs from a control's recorded first gradient",
}


def _number(value: Any) -> bool:
    """A finite real number: not None, not a bool, not ±∞ or NaN."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _diverging(values: Iterable[Any]) -> bool:
    """A non-finite number among ``values`` (a divergence, Revision 3, 3). A missing value is
    no evidence of divergence."""
    return any(isinstance(v, int | float) and not isinstance(v, bool) and not math.isfinite(v)
               for v in values)  # fmt: skip


def _largest(values: Sequence[Any]) -> float | None:
    """The largest measured value: ∞ when one is non-finite (a NaN never hides under ``max``),
    None when nothing was measured."""
    if not values:
        return None
    return max(values) if all(_number(v) for v in values) else math.inf


def _over(value: float | None, k_star: float) -> float | None:
    return None if value is None else value / k_star


def _row_values(rows: Sequence[Mapping[str, Any]]) -> Iterator[Any]:
    """The losses and gradient norms of a run's per-step rows."""
    for row in rows:
        yield row.get("loss")
        yield row.get("grad_norm")


def _pair_summary(
    divergence: Sequence[Mapping[str, Any]], eval_at: Sequence[int], stop: str | None
) -> dict[str, Any]:
    """The pair's D_twin series: ``complete`` when every measurement step is present and
    finite and the twin did not stop; ``max_D_twin``, the largest measured value (``_largest``).
    An incomplete series is not read as an unbounded one: the pair's state (``pair_state``)
    tells exceeding from incomplete (Revision 3, 3)."""
    values = [d["D_twin"] for d in divergence]
    complete = (stop is None and [d["step"] for d in divergence] == list(eval_at)
                and all(math.isfinite(v) for v in values))  # fmt: skip
    return {"max_D_twin": _largest(values), "complete": complete}


def survived(attempts: Sequence[Mapping[str, Any]] | None, survival_share: float) -> bool:
    """Revision 3, 2: the attempt the twin ran with (the last one: the first, or its single ×10
    retry) realized at least ``survival_share`` of its own intended norm after fp32 rounding."""
    if not attempts or len(attempts) > 2:
        return False
    realized, intended = attempts[-1].get("realized_norm"), attempts[-1].get("intended_norm")
    return _number(realized) and _number(intended) and realized >= survival_share * intended


def pair_state(
    pair: Mapping[str, Any],
    *,
    perturbed: bool,
    eval_at: Sequence[int],
    bound: float,
    survival_share: float,
) -> dict[str, Any]:
    """A D3 pair's state, Revision 3, 3, from its record:
    - **exceeding** when a measured D_twin is above the bound, or the twin diverged (a non-finite
      loss, gradient, weight or D_twin); survival is not required for this state;
    - **within** when its perturbation survived fp32 rounding (``survived``; the reference has no
      perturbation), D_twin was measured at every step of ``eval_at`` and every value is ≤ the
      bound;
    - **incomplete** otherwise: a perturbation lost to rounding after its ×10 retry, a missing
      step, or a stop that is not a divergence."""
    readings = list(pair.get("D_twin") or [])
    values = [r.get("D_twin") for r in readings]
    diverged = (pair.get("diverged") is True or _diverging(values)
                or _diverging(_row_values(pair.get("rows") or [])))  # fmt: skip
    kept_norm = survived(pair.get("rounding"), survival_share) if perturbed else None
    measured = [r.get("step") for r in readings] == list(eval_at)
    if diverged or any(_number(v) and v > bound for v in values):
        state = EXCEEDING
    elif (kept_norm is not False and measured and pair.get("stop") is None
          and all(_number(v) and v <= bound for v in values)):  # fmt: skip
        state = WITHIN
    else:
        state = INCOMPLETE
    return {"state": state, "survived": kept_norm, "diverged": diverged,
            "measured_every_step": measured}  # fmt: skip


def control_state(control: Mapping[str, Any], *, k_at: Sequence[int], bound: float) -> dict:
    """The same-path float64 control's state, Revision 3, 3, from its record:
    - **exceeding** when the run diverged (a non-finite loss, gradient, weight, K_scoring or
      K_monitor), or a K_scoring of the complete series is above the bound;
    - **below** when K_scoring was measured at every step of ``k_at`` and every value is strictly
      below the bound;
    - **at the bound** when it is neither, on a complete series;
    - **incomplete** otherwise: a missing step, or a stop that is not a divergence."""
    scoring = list(control.get("scoring") or [])
    values = [r.get("K_scoring") for r in scoring]
    steps = [r.get("step") for r in scoring]
    diverged = (control.get("diverged") is True or _diverging(values)
                or _diverging(r.get("K_monitor") for r in control.get("monitor") or [])
                or _diverging(_row_values(control.get("rows") or [])))  # fmt: skip
    stopped = control.get("stop") is not None and not diverged
    complete = steps == list(k_at) and all(_number(v) for v in values) and not stopped
    if diverged or (complete and any(v > bound for v in values)):
        state = EXCEEDING
    elif complete and all(v < bound for v in values):
        state = BELOW
    elif complete:
        state = AT_BOUND
    else:
        state = INCOMPLETE
    return {"state": state, "complete": complete, "diverged": diverged,
            "max_K_scoring": _largest(values),
            "nonfinite_steps": [s for s, v in zip(steps, values, strict=True)
                                if not _number(v)]}  # fmt: skip


def path_criterion(residual: Mapping[str, Any], share_max: float) -> dict[str, Any]:
    """Revision 2, 1: the residual comes from the path discrepancy when ‖(i)‖ ≤ share_max·‖(iii)‖.
    Revision 3, 1: when ‖(i)‖ or ‖(iii)‖ is non-finite, or ‖(iii)‖ is zero, it is undecided,
    never false."""
    norms = residual.get("norms") or {}
    norm_i, norm_iii = norms.get("i_training_path_fp32"), norms.get("iii_control_recorded")
    out: dict[str, Any] = {"norm_i": norm_i, "norm_iii": norm_iii, "share_max": share_max}
    if not (_number(norm_i) and _number(norm_iii)) or norm_iii == 0:
        return {**out, "state": UNDECIDED, "ratio_i_to_iii": None}
    ratio = norm_i / norm_iii
    return {**out, "state": HOLDS if ratio <= share_max else FALSE, "ratio_i_to_iii": ratio}


def shared_state_checks(d1: Mapping[str, Any], controls: Sequence[str]) -> dict[str, bool]:
    """Revision 3, 1: each D1 check that gates every branch, True only when it passed: the
    trainable set, v₀'s hash under both configs, the loss normalization, the initial optimizer
    state (each built from its own config's v₀) and the recomputed (iii), equal to the recorded
    first gradient of every control in ``controls`` (their rates, as D1 keys them)."""
    shared = d1.get("shared_state") or {}
    equals = (d1.get("residual") or {}).get("iii_equals_record") or {}

    def passed(check: str, key: str = "equal") -> bool:
        return (shared.get(check) or {}).get(key) is True

    return {
        "trainable_parameters": passed("trainable_parameters"),
        "v0": passed("v0"),
        "loss_normalization": passed("loss_normalization"),
        "initial_optimizer_state": passed("initial_optimizer_state", "equal_except_lr"),
        "iii_equals_record": (set(equals) == set(controls)
                              and all(v is True for v in equals.values())),
    }  # fmt: skip


def control_keys(settings: DictConfig) -> list[str]:
    """The FA soft controls' rates as D1 keys their records."""
    return [f"{float(rate):g}" for rate in settings.diagnostic.controls]


def first_control_step(
    control_cfg: DictConfig, twins: Twins, residual: Sequence[torch.Tensor]
) -> list[torch.Tensor]:
    """δ₁ = θ₁ − θ₀ of the FA soft control's first step at its config's rate: frozen Adam built
    as the control builds it, stepped once on the measured residual gradient (host tensors).
    The student is back at θ₀ afterwards."""
    twins.reset()
    params = twins.params
    optimizer = build_optimizer(control_cfg.optimizer, params, second_moment=twins.v0)
    for p, r in zip(params, residual, strict=True):
        p.grad = r.to(p.device)
    optimizer.step()
    delta = [(p.detach() - twins.theta0[name].to(p.device)).cpu()
             for name, p in twins.trainable]  # fmt: skip
    del optimizer
    twins.reset()
    return delta


def control_residual(
    control_cfg: DictConfig, env: Environment, model: PreTrainedModel,
    params: Sequence[torch.nn.Parameter], prefix: StreamPrefix,
) -> tuple[Vector, dict[str, Any], list[torch.Tensor]]:  # fmt: skip
    """The control's first gradient at θ₀ as the control computed it: batch 0 of the stream,
    generation-path targets replayed by ``model`` (at θ₀), ``accumulate_gradient``. Returns the
    gradient (released from the parameters), its statistics and the batch's targets."""
    size = int(control_cfg.train.batch_size)
    batch = prefix.take(0, size)
    replay = ReplayedSoftTargets(prefix, model, env).take(0, size)
    for p in params:
        p.grad = None
    stats = accumulate_gradient(model, batch, control_cfg, env, replay)
    stats["grad_norm"] = grad_norm([p.grad for p in params])
    return take_gradients(params), stats, replay


# ------------------------------------------------------------------------------- the parts


def _model(env: Environment, cfg: DictConfig) -> tuple[PreTrainedModel, list]:
    model = env.new_model()
    return model, freeze(model, cfg.train.freeze)


def second_moment(
    cfg: DictConfig, env: Environment, model: PreTrainedModel, params: Sequence
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """v₀ exactly as the run estimates it (``train._theta0_second_moment``), with the
    preconditioner stream's identity."""
    recorder = Recorder()
    v0 = _theta0_second_moment(cfg, env, model, list(params), recorder)
    return v0, recorder.inputs.get("streams", {}).get("preconditioner", {})


def first_recorded(run_dir: Path) -> dict[str, Any]:
    """The archived run's first training record (step 0)."""
    rows = [r for r in read_jsonl(run_dir / "metrics.jsonl") if "loss" in r]
    return {key: rows[0].get(key) for key in ("loss", "grad_norm", "tokens", "denominator")}


def run_d1(settings: DictConfig, env: Environment, runs: Runs, out: Path) -> dict[str, Any]:
    """D4's shared-state checks and D1 (module docstring); writes ``d1/result.json``. Its
    ``status`` is ``incomplete``, with its ``reasons``, when a shared-state check fails or the
    path criterion is undecided (Revision 3, 1)."""
    d = settings.diagnostic
    eps = float(d.epsilon)
    case = cases_of(settings)[d.d1.pair]
    regular_dir, control_dir = runs.regular[case.name], runs.control(case.control_lr)
    regular_cfg, control_cfg = run_config(regular_dir), run_config(control_dir)
    device = env.device
    reset_peak(device)
    started = time.perf_counter()
    student = env.new_model()
    control_set = [(n, tuple(p.shape)) for n, p in freeze(student, control_cfg.train.freeze)]
    trainable = freeze(student, regular_cfg.train.freeze)
    names, params = [n for n, _ in trainable], [p for _, p in trainable]
    checks: dict[str, Any] = {
        "trainable_parameters": {
            "equal": [(n, tuple(p.shape)) for n, p in trainable] == control_set,
            "count": len(names), "numel": sum(p.numel() for p in params),
            "freeze": {"regular": list(regular_cfg.train.freeze),
                       "control": list(control_cfg.train.freeze)},
        },
    }  # fmt: skip

    # v₀ under each config, on the same θ₀: the same code path, hashed
    v0, regular_stream = second_moment(regular_cfg, env, student, params)
    v0_hash = tensors_sha256(list(zip(names, v0, strict=True)))
    v0_control, control_stream = second_moment(control_cfg, env, student, params)
    checks["v0"] = {
        "regular_sha256": v0_hash,
        "control_sha256": tensors_sha256(list(zip(names, v0_control, strict=True))),
        "regular_stream": regular_stream, "control_stream": control_stream,
    }  # fmt: skip
    checks["v0"]["equal"] = checks["v0"]["regular_sha256"] == checks["v0"]["control_sha256"]
    # Each run's optimizer from its own config's v₀ estimate
    checks["initial_optimizer_state"] = _initial_state(
        (regular_cfg, v0), (control_cfg, v0_control), params
    )  # fmt: skip
    del v0_control
    gc.collect()

    prefix = StreamPrefix(control_cfg, env, int(control_cfg.seed))
    batch = prefix.take(0, int(control_cfg.train.batch_size))
    cap = int(regular_cfg.train.max_response_tokens)
    loss_keys = ("normalization", "denominator", "include_end_token")
    checks["loss_normalization"] = {
        "regular": {**{k: regular_cfg.train.loss[k] for k in loss_keys}, "cap": cap,
                    "batch_denominator": loss_denominator(regular_cfg.train.loss, batch, cap)},
        "control": {**{k: control_cfg.train.loss[k] for k in loss_keys},
                    "cap": int(control_cfg.train.max_response_tokens),
                    "batch_denominator": loss_denominator(control_cfg.train.loss, batch, cap)},
        "loss_kind": {"regular": regular_cfg.train.loss.kind,
                      "control": control_cfg.train.loss.kind},
    }  # fmt: skip
    norm = checks["loss_normalization"]
    norm["equal"] = norm["regular"] == norm["control"]

    # The residual gradient at θ₀ on the control's first batch: (iii), (i), (ii), and the NLL
    g_iii, stats_iii, _ = control_residual(control_cfg, env, student, params, prefix)
    recorded = {f"{lr:g}": first_recorded(path) for lr, path in runs.controls.items()}
    same_path = training_path_targets(student, batch, control_cfg, env)
    accumulate_gradient(student, batch, control_cfg, env, same_path)
    del same_path
    g_i = take_gradients(params)
    with Float64Gradients(params) as sums:
        stats_ii = soft_loss_float64(student, student, batch, control_cfg, env)
    g_ii64 = sums.take()
    norm_ii = math.sqrt(sum(g.pow(2).sum().item() for g in g_ii64))
    g_ii = [g.float() for g in g_ii64]
    del g_ii64
    accumulate_gradient(student, batch, regular_cfg, env)
    g_nll = take_gradients(params)
    vectors = {"i_training_path_fp32": g_i, "ii_training_path_float64": g_ii,
               "iii_control_recorded": g_iii}  # fmt: skip
    norms = {key: _norm(g) for key, g in vectors.items()}
    norms["ii_training_path_float64"] = norm_ii
    keys = list(vectors)
    residual = {
        "norms": norms,
        "cosines": {f"{a}|{b}": cosine(vectors[a], vectors[b])
                    for i, a in enumerate(keys) for b in keys[i + 1 :]},
        "iii_recomputed": stats_iii, "iii_recorded": recorded,
        "iii_equals_record": {lr: r.get("grad_norm") == stats_iii["grad_norm"]
                              and r.get("loss") == stats_iii["loss"]
                              for lr, r in recorded.items()},
        "ii_loss": stats_ii["loss"],
        "nll_first_gradient_norm": _norm(g_nll),
        "path_share_max": float(d.d1.path_share_max),
    }  # fmt: skip
    # The path criterion (Revision 2, 1): holds, false, or undecided on a non-finite or zero
    # norm (Revision 3, 1); no ratio is taken over an undecided pair of norms.
    residual["path_criterion"] = path_criterion(residual, float(d.d1.path_share_max))
    residual["ratio_i_to_iii"] = residual["path_criterion"]["ratio_i_to_iii"]
    decided = residual["path_criterion"]["state"] != UNDECIDED
    residual["ratio_ii_to_iii"] = norm_ii / norms["iii_control_recorded"] if decided else None
    shares = {"iii_control_recorded": preconditioned_shares(names, v0, g_iii, eps),
              "i_training_path_fp32": preconditioned_shares(names, v0, g_i, eps),
              "nll_first_step": preconditioned_shares(names, v0, g_nll, eps)}  # fmt: skip
    del g_i, g_ii, g_iii, g_nll, vectors
    census = v0_census(names, v0, eps)
    v0_host = [v.cpu() for v in v0]
    del v0
    reset_peak_keep = peak_memory(device)

    # The same-path float64 soft control at the failing rate (Branch F's condition)
    same_path_cfg = with_lr(control_cfg, float(d.d1.same_path_control.rate))
    control = same_path_control(settings, env, same_path_cfg, student, trainable, v0_host)
    # Revision 3, 1: every shared-state check gates every branch, (iii) equal to every control's
    # recorded first gradient bit for bit among them, and an undecided path criterion too
    gates = shared_state_checks({"shared_state": checks, "residual": residual},
                                control_keys(settings))  # fmt: skip
    unequal = sorted(lr for lr, equal in residual["iii_equals_record"].items() if not equal)
    detail = {"iii_equals_record": f" (the controls {unequal})"}
    reasons = [SHARED_STATE_CHECKS[name] + detail.get(name, "")
               for name, passed in gates.items() if not passed]  # fmt: skip
    if not decided:
        reasons.append(f"the path criterion is undecided: ‖(i)‖ {norms['i_training_path_fp32']}, "
                       f"‖(iii)‖ {norms['iii_control_recorded']}")  # fmt: skip
    status = {"status": "incomplete", "reasons": reasons} if reasons else {"status": "complete"}
    result = {
        "part": "d1", **status, "shared_state": checks, "v0_census": census, "residual": residual,
        "preconditioned_first_step": shares, "same_path_control": control,
        "numerics": numerics(device), "seconds": time.perf_counter() - started,
        "memory": {"before_same_path_control": reset_peak_keep},
        "provenance": provenance(settings, runs),
    }  # fmt: skip
    write_json(out / "d1" / "result.json", result)
    return result


def _initial_state(
    regular: tuple[DictConfig, Sequence[torch.Tensor]],
    control: tuple[DictConfig, Sequence[torch.Tensor]],
    params: Sequence,
) -> dict[str, Any]:
    """Both runs' optimizers as built before their first step, each from its own config and its
    own v₀ estimate: mode, hyperparameters (the rate aside), and the state, which holds only v₀
    (m₀ and the step counter start at zeros and 0 in the first step, optim.Adam)."""
    out = {}
    for label, (cfg, v0) in (("regular", regular), ("control", control)):
        optimizer = build_optimizer(cfg.optimizer, list(params), second_moment=list(v0))
        group = optimizer.param_groups[0]
        state = [optimizer.state[p] for p in params]
        out[label] = {
            "class": type(optimizer).__name__,
            "mode": getattr(optimizer, "second_moment", None),
            "betas": list(group["betas"]), "eps": group["eps"], "lr": group["lr"],
            "state_keys": sorted({k for s in state for k in s}),
            "has_step_or_m0": any("step" in s or "exp_avg" in s for s in state),
            "v_sha256": tensors_sha256([(str(i), s["exp_avg_sq"]) for i, s in enumerate(state)]),
        }  # fmt: skip
        del optimizer, state
        gc.collect()
    keys = ("class", "mode", "betas", "eps", "state_keys", "has_step_or_m0", "v_sha256")
    out["equal_except_lr"] = all(out["regular"][k] == out["control"][k] for k in keys)
    return out


def same_path_control(
    settings: DictConfig,
    env: Environment,
    cfg: DictConfig,
    student: PreTrainedModel,
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    v0: Sequence[torch.Tensor],
) -> dict[str, Any]:
    """D1's same-path float64 soft control (``same_path_run``) at the diagnostic's steps, bound
    and K*."""
    d = settings.diagnostic
    return same_path_run(env, cfg, student, trainable, v0,
                         k_at=[int(t) for t in d.d1.same_path_control.k_at],
                         bound=float(d.bound), k_star=float(d.k_star))  # fmt: skip


def same_path_run(
    env: Environment,
    cfg: DictConfig,
    student: PreTrainedModel,
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    v0: Sequence[torch.Tensor],
    *,
    k_at: Sequence[int],
    bound: float,
    k_star: float,
) -> dict[str, Any]:
    """The FA soft control with same-path targets: θ₀'s training-path distributions (the frozen
    θ₀ copy, recomputed per micro-batch), the loss and the gradient accumulation in float64,
    the forward and backward in fp32, frozen Adam as the control runs it, for the run's steps.
    K on the scoring panel at ``k_at`` (uncached, against the θ₀ copy) and on the monitor panel
    every ``eval.monitor.every`` steps, with ‖δ‖. ``student`` must be at θ₀. The record carries
    the control's state (``control_state``: below, exceeding, at the bound or incomplete) and
    the identity of the training-stream prefix it read (``training_stream``)."""
    device = env.device
    params = [p for _, p in trainable]
    reference = env.new_model()
    reference.requires_grad_(False)
    theta0 = dict(reference.named_parameters())
    reset_peak(device)
    started = time.perf_counter()
    scoring, identity = scoring_samples(cfg, env)
    monitor = Monitor(operational_monitor(cfg), env, Recorder(), reference)
    prefix = StreamPrefix(cfg, env, int(cfg.seed))
    optimizer = build_optimizer(cfg.optimizer, params, second_moment=list(v0))
    k_at = [int(t) for t in k_at]
    every, size = int(cfg.eval.monitor.every), int(cfg.train.batch_size)
    rows, readings, scoring_k = [], [], []
    stop, taken, diverged = None, 0, False
    for step in range(int(cfg.train.steps)):
        batch = prefix.take(step * size, (step + 1) * size)
        for p in params:
            p.grad = None
        with Float64Gradients(params) as sums:
            stats = soft_loss_float64(student, reference, batch, cfg, env)
        sums.to_grad()
        stats["grad_norm"] = grad_norm([p.grad for p in params])
        rows.append({"step": step, **stats})
        if not _finite(stats["loss"], stats["grad_norm"]):
            stop, diverged = f"non-finite loss or gradient at step {step}", True
            break
        optimizer.step()
        taken = t = step + 1
        with torch.no_grad():
            displacement = math.sqrt(sum((p.double() - theta0[n].double()).pow(2).sum().item()
                                         for n, p in trainable))  # fmt: skip
        if not math.isfinite(displacement):
            stop, diverged = f"non-finite weights after step {step}", True
            break
        if t % every == 0:
            readings.append({"step": t, "K_monitor": monitor(student),
                             "displacement_norm": displacement})  # fmt: skip
        if t in k_at:
            for p in params:
                p.grad = None
            per_response = sequence_kl(student, reference, scoring, pad_id=env.pad_id,
                                       micro_batch_size=cfg.execution.micro_batch_size)  # fmt: skip
            scoring_k.append({"step": t, "K_scoring": per_response.mean().item(),
                              "displacement_norm": displacement})  # fmt: skip
    for p in params:
        p.grad = None
    bound = float(bound)
    # Both stops above are a divergence (Revision 3, 3): a non-finite loss, gradient or weight.
    out = {"rate": float(cfg.optimizer.lr), "steps": taken, "stop": stop, "diverged": diverged,
           "rows": rows, "monitor": readings, "scoring": scoring_k, "scoring_panel": identity,
           "bound": bound, "K_star": float(k_star), "k_at": k_at,
           "training_stream": prefix.identity(taken * size),
           "seconds": time.perf_counter() - started, **peak_memory(device)}  # fmt: skip
    out.update(control_state(out, k_at=k_at, bound=bound))  # its state, Revision 3, 3
    out["max_K_scoring_over_K_star"] = _over(out["max_K_scoring"], float(k_star))
    del optimizer, monitor, reference, theta0
    gc.collect()
    return out


def run_d2(settings: DictConfig, env: Environment, runs: Runs, out: Path) -> dict[str, Any]:
    """D2 (module docstring); writes ``d2/result.json``."""
    d, s = settings.diagnostic, settings.diagnostic.d2
    eps = float(d.epsilon)
    case = cases_of(settings)[d.d1.pair]
    control_cfg = run_config(runs.control(case.control_lr))
    device = env.device
    reset_peak(device)
    started = time.perf_counter()
    model, trainable = _model(env, control_cfg)
    names, params = [n for n, _ in trainable], [p for _, p in trainable]
    v0, stream = second_moment(control_cfg, env, model, params)
    v0_hash = tensors_sha256(list(zip(names, v0, strict=True)))
    beta = float(control_cfg.optimizer.beta1)
    size, k = int(control_cfg.train.batch_size), int(s.batches)
    prefix = StreamPrefix(control_cfg, env, int(control_cfg.seed))
    batches = [prefix.take(b * size, (b + 1) * size) for b in range(k)]
    replay = ReplayedSoftTargets(prefix, model, env).take(0, k * size)
    targets = [replay[b * size : (b + 1) * size] for b in range(k)]
    residual, residual_stats, _ = control_residual(control_cfg, env, model, params, prefix)
    residual = [r.cpu() for r in residual]
    gc.collect()

    terms = {kind: [batch_terms(model, [batch], control_cfg, env, kind=kind,
                                soft_targets=None if kind == "nll" else [targets[b]])
                    for b, batch in enumerate(batches)]
             for kind in ("soft", "nll")}  # fmt: skip
    fisher_terms = batch_terms(model, batches, control_cfg, env, kind="soft", soft_targets=targets)
    lanczos_steps, tolerance = int(s.lanczos.steps), float(s.lanczos.tolerance)
    vector_bytes = sum(p.numel() for p in params) * 4
    spectra: dict[str, list[dict[str, Any]]] = {"soft": [], "nll": []}
    for kind_index, kind in enumerate(("soft", "nll")):
        for b in range(k):
            hvp = preconditioned_hvp(model, params, terms[kind][b], v0, eps)
            index = 1000 * (kind_index + 1) + b
            result = lanczos(hvp, start_vector(trainable, int(d.seed), index), steps=lanczos_steps,
                             tolerance=tolerance)  # fmt: skip
            # The Ritz vector beside the Lanczos vectors when the first pass's peak leaves room
            # for it under the memory rule (1 GB margin), else in host memory, staged
            limit = memory_limit_bytes(device)
            needed = vector_bytes + 1e9
            reserved = 0 if limit is None else torch.cuda.max_memory_reserved(cuda_index(device))
            on_gpu = limit is not None and reserved + needed <= limit
            y = ritz_vector(hvp, start_vector(trainable, int(d.seed), index), result,
                            host=not on_gpu)  # fmt: skip
            check = check_vector(hvp, y, device)
            spectra[kind].append({
                "batch": b, **{key: result[key] for key in ("top", "bottom", "residual",
                                                           "ritz_converged", "steps",
                                                           "top_trace", "alphas", "betas")},
                "top_vector_check": check,
                "ritz_vector_on": "gpu" if on_gpu else "host",
                "localization": vector_shares(names, y, v0, eps),
                "localization_in_theta": vector_shares(names, y, v0, eps, in_theta=True),
            })  # fmt: skip
            del y
            gc.collect()

    rates = [float(r) for r in s.rates]
    steps = int(control_cfg.train.steps)
    fisher_at = [int(t) for t in s.fisher_at]
    thresholds = {}
    for eta in rates:
        thresholds[f"{eta:g}"] = {
            "heavy_ball_lambda": heavy_ball_threshold(eta, beta),
            "no_momentum_lambda": 2 / eta,
            "per_batch": {kind: [{
                "batch": r["batch"], "top": r["top"], "bottom": r["bottom"],
                "eta_top": eta * r["top"], "first_step_multiplier_top": 1 - eta * r["top"],
                "first_step_multiplier_bottom": 1 - eta * r["bottom"],
                "stationary_radius_top": stationary_radius(eta * r["top"], beta),
                "stationary_radius_bottom": stationary_radius(eta * r["bottom"], beta),
                "scalar_mode_top": scalar_mode(eta, r["top"], beta, steps),
                "scalar_mode_bottom": scalar_mode(eta, r["bottom"], beta, steps),
            } for r in spectra[kind]] for kind in ("soft", "nll")},
        }  # fmt: skip
    maps = {}
    for eta in rates:
        for kind in ("soft", "nll"):
            maps[f"{kind}|{eta:g}"] = linearized_run(
                model, params, v0, residual, eta=eta, beta=beta, eps=eps, steps=steps,
                step_terms=terms[kind], fisher_terms=fisher_terms, fisher_at=fisher_at,
            )  # fmt: skip
            gc.collect()
    observed = {f"{lr:g}": observed_growth(read_jsonl(path / "monitor.jsonl"))
                for lr, path in runs.controls.items()}  # fmt: skip
    result = {
        "part": "d2", "v0_sha256": v0_hash, "v0_stream": stream, "beta1": beta, "epsilon": eps,
        "batches": k, "hessians": HESSIANS, "residual": {"norm": _norm(residual), **residual_stats},
        "spectra": spectra, "thresholds": thresholds, "linearized": maps, "observed": observed,
        "numerics": numerics(device), "seconds": time.perf_counter() - started,
        **peak_memory(device), "provenance": provenance(settings, runs),
    }  # fmt: skip
    write_json(out / "d2" / "result.json", result)
    return result


@contextmanager
def _twins(settings: DictConfig, env: Environment, runs: Runs) -> Iterator[tuple[Twins, Any]]:
    """The models and inputs every D3 case shares, and the control's residual (host)."""
    d = settings.diagnostic
    determinism = cases_of(settings)[d.d3.determinism]
    cfg = run_config(runs.regular[determinism.name])
    control_cfg = run_config(runs.control(determinism.control_lr))
    with twin_setup(env, cfg, control_cfg, k_star=float(d.k_star)) as out:
        yield out


@contextmanager
def twin_setup(
    env: Environment, cfg: DictConfig, residual_cfg: DictConfig, *, k_star: float
) -> Iterator[tuple[Twins, Any]]:
    """What every twin of one seed shares (``Twins``), built from the run config ``cfg`` (its
    trainable set, v₀ as the run estimates it, its training stream, scoring panel and monitor),
    and in ``shared``: v₀'s hash and stream, the monitor (its reference is the second model) and
    the soft-target residual at θ₀ on batch 0 with generation-path targets (``control_residual``
    under ``residual_cfg``; host tensors), with its statistics."""
    student, trainable = _model(env, cfg)
    other = env.new_model()
    other.requires_grad_(False)
    names, params = [n for n, _ in trainable], [p for _, p in trainable]
    theta0 = {name: p.detach().cpu().clone() for name, p in trainable}
    v0, stream = second_moment(cfg, env, student, params)
    v0_hash = tensors_sha256(list(zip(names, v0, strict=True)))
    v0 = [v.cpu() for v in v0]
    prefix = StreamPrefix(cfg, env, int(cfg.seed))
    residual, stats, _ = control_residual(residual_cfg, env, student, params, prefix)
    residual = [r.cpu() for r in residual]
    gc.collect()
    scoring, identity = scoring_samples(cfg, env)
    monitor = Monitor(operational_monitor(cfg), env, Recorder(), other)
    other.to("cpu")
    twins = Twins(env=env, student=student, other=other, trainable=trainable, theta0=theta0,
                  v0=v0, prefix=prefix, scoring=scoring, k_star=float(k_star))  # fmt: skip
    shared = {"v0_sha256": v0_hash, "v0_stream": stream, "control_residual": stats,
              "control_residual_norm": _norm(residual), "scoring_panel": identity,
              "residual": residual, "monitor": monitor}  # fmt: skip
    try:
        yield twins, shared
    finally:
        del twins, monitor, other, student
        gc.collect()


def run_d3(
    settings: DictConfig,
    env: Environment,
    runs: Runs,
    out: Path,
    *,
    cases: Sequence[str],
    determinism: bool = True,
    keep_checkpoints: bool = False,
) -> dict[str, Any]:
    """D3 (module docstring): the determinism check unless done, then each case not yet written;
    writes ``d3/determinism.json`` and ``d3/<case>.json``. Records written at another commit
    are moved aside (``kept``) and redone. Raises ``DeterminismFailure`` (after writing its
    record) when the re-run differs from the archive."""
    d = settings.diagnostic
    all_cases = cases_of(settings)
    eval_at = [int(t) for t in d.d3.eval_at]
    directory = out / "d3"
    record_path = directory / "determinism.json"
    commit = git_state().get("commit")
    for path in sorted(directory.glob("*.json")):  # any record of another commit goes aside
        kept(out, f"d3/{path.name}", commit)
    done = {}
    with _twins(settings, env, runs) as (twins, shared):
        record = kept(out, "d3/determinism.json", commit)
        if record is not None:
            pass
        elif determinism:
            case = all_cases[d.d3.determinism]
            cfg = run_config(runs.regular[case.name])
            twins.reset()
            rerun = run_twin(cfg, twins, eval_at=[], monitor=shared["monitor"])
            record = {"part": "d3-determinism", "case": case.name,
                      **compare_archive(rerun["rows"], rerun["monitor"], runs.regular[case.name]),
                      "rerun": rerun, "numerics": numerics(env.device),
                      "provenance": provenance(settings, runs)}  # fmt: skip
            write_json(record_path, record)
        else:
            raise DiagnosticError("D3 needs its determinism check first")
        if not record["equal"]:
            raise DeterminismFailure(f"The re-run of {record['run']} differs from its archive: "
                                     f"{record['mismatches'][:3]}; D3 stops")  # fmt: skip
        for name in cases:
            record = kept(out, f"d3/{name}.json", commit)
            if record is not None:
                done[name] = record
                continue
            done[name] = run_case(settings, all_cases[name], twins, shared, runs, directory,
                                  eval_at=eval_at, keep_checkpoints=keep_checkpoints)  # fmt: skip
            write_json(directory / f"{name}.json", done[name])
    return done


def _remove_checkpoints(directory: Path) -> None:
    """Twin A's checkpoints of a case, and any partial one an interrupted save left."""
    for path in [*directory.glob("A-step-*.pt"), *directory.glob(".A-step-*.pt.partial")]:
        path.unlink()


def run_twin_a(
    cfg: DictConfig,
    twins: Twins,
    monitor: Monitor,
    run_dir: Path,
    checkpoints: Path,
    eval_at: Sequence[int],
) -> dict[str, Any]:
    """Twin A: the regular run of ``cfg`` re-run from θ₀ with its monitor, its trainable
    parameters saved to ``checkpoints`` at ``eval_at``, and compared with its archive
    (``run_dir``) bit for bit (``archive``)."""
    twins.reset()
    twin_a = run_twin(cfg, twins, eval_at=eval_at, monitor=monitor, save_dir=checkpoints)
    twin_a["archive"] = compare_archive(twin_a["rows"], twin_a["monitor"], run_dir)
    return twin_a


def run_pairs(
    cfg: DictConfig,
    twins: Twins,
    directions: Mapping[str, Callable[[str, Sequence[int]], torch.Tensor]],
    scales: Mapping[str, float],
    checkpoints: Path,
    *,
    eval_at: Sequence[int],
    bound: float,
    survival_share: float,
    rounding_scale: float,
    pairs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The pairs of a case against twin A's checkpoints: p1–p4 (θ₀ + scale·direction, with the
    retry of ``apply_perturbation``) and the float64 reference, each with its state
    (``pair_state``). A perturbation without a scale (its direction has no finite, positive
    norm) is not run and its pair is incomplete. ``pairs``, if given, receives each pair as it
    is done, so that the pairs before a failure survive it."""
    pairs = {} if pairs is None else pairs
    for key in PERTURBATIONS:
        if key not in scales:
            pairs[key] = {"steps": 0, "stop": f"no perturbation: {key}'s direction has no finite, "
                          "positive norm", "diverged": False, "rows": [], "D_twin": [],
                          "max_D_twin": None, "complete": False}  # fmt: skip
            pairs[key].update(pair_state(pairs[key], perturbed=True, eval_at=eval_at,
                                         bound=bound, survival_share=survival_share))  # fmt: skip
            continue
        twins.reset()
        rounding = apply_perturbation(
            twins.trainable, twins.theta0, directions[key], scales[key],
            survival_share=survival_share, rounding_scale=rounding_scale,
        )  # fmt: skip
        # A perturbation lost to rounding after its retry still runs: it can be exceeding, never
        # within (Revision 3, 2 and 3)
        twin = run_twin(cfg, twins, eval_at=eval_at, compare_dir=checkpoints)
        twin["rounding"] = rounding
        twin.update(pair_state(twin, perturbed=True, eval_at=eval_at, bound=bound,
                               survival_share=survival_share))  # fmt: skip
        pairs[key] = twin
    twins.reset()
    pairs[REFERENCE] = reference_twin(cfg, twins, eval_at, checkpoints)
    pairs[REFERENCE].update(pair_state(pairs[REFERENCE], perturbed=False, eval_at=eval_at,
                                       bound=bound, survival_share=survival_share))  # fmt: skip
    return pairs


def run_case(
    settings: DictConfig,
    case: Case,
    twins: Twins,
    shared: Mapping[str, Any],
    runs: Runs,
    directory: Path,
    *,
    eval_at: Sequence[int],
    keep_checkpoints: bool,
) -> dict[str, Any]:
    """One D3 case: twin A (checked against its archive), p1–p4 and the reference, each against
    twin A by D_twin at ``eval_at``, each with its state (``pair_state``: within, exceeding or
    incomplete). A twin A that differs from its archived run, bit for bit, makes the case
    ``incomplete``: no pair is run against a twin that is not the archived run (review I1)."""
    d = settings.diagnostic
    bound = float(d.bound)
    env = twins.env
    run_dir = runs.regular[case.name]
    cfg = run_config(run_dir)
    if not math.isclose(float(cfg.optimizer.lr), case.lr, rel_tol=1e-9):
        raise DiagnosticError(f"{run_dir} ran at {cfg.optimizer.lr}, the case names {case.lr}")
    started = time.perf_counter()
    checkpoints = directory / "checkpoints" / case.name
    twin_a = run_twin_a(cfg, twins, shared["monitor"], run_dir, checkpoints, eval_at)
    record = {"part": "d3", "case": case.name, "optimizer": case.optimizer, "lr": case.lr,
              "control_lr": case.control_lr, "regular_run": str(run_dir), "bound": bound,
              "K_star": twins.k_star, "eval_at": list(eval_at)}  # fmt: skip
    if twin_a["stop"] is not None or not twin_a["archive"]["equal"]:
        _remove_checkpoints(checkpoints)
        twins.reset()
        reason = (f"twin A stopped: {twin_a['stop']}" if twin_a["stop"] is not None
                  else "twin A differs from its archived run")  # fmt: skip
        return {**record, "status": "incomplete", "reason": reason, "twin_A": twin_a,
                "pairs": {}, "numerics": numerics(env.device),
                "seconds": time.perf_counter() - started,
                "provenance": provenance(settings, runs)}  # fmt: skip

    control_cfg = with_lr(run_config(runs.control(base_control_rate(settings))), case.control_lr)
    delta = first_control_step(control_cfg, twins, shared["residual"])
    delta_map = dict(zip(twins.names, delta, strict=True))
    delta_norm = _norm(delta)  # the realized displacement, after the fp32 rounding of θ₁
    eps = float(control_cfg.optimizer.eps)
    intended = case.control_lr * math.sqrt(sum(
        (r.double() * preconditioner(v, eps).double()).pow(2).sum().item()
        for r, v in zip(shared["residual"], twins.v0, strict=True)))  # fmt: skip
    directions: dict[str, Callable[[str, Sequence[int]], torch.Tensor]] = {
        "p1": lambda name, shape: delta_map[name],
        "p2": lambda name, shape: -delta_map[name],
    }
    norms = {"p1": delta_norm, "p2": delta_norm}
    for key in ("p3", "p4"):
        directions[key] = (lambda name, shape, s=int(d.d3.gaussian[key]), k=key:
                           gaussian(name, shape, s, k))  # fmt: skip
        norms[key] = gaussian_norm(twins.trainable, int(d.d3.gaussian[key]), key)
    pairs = run_pairs(
        cfg, twins, directions, {key: delta_norm / norms[key] for key in PERTURBATIONS},
        checkpoints, eval_at=eval_at, bound=bound, survival_share=float(d.d3.survival_share),
        rounding_scale=float(d.d3.rounding_scale),
    )  # fmt: skip
    if not keep_checkpoints:
        _remove_checkpoints(checkpoints)
    twins.reset()
    measured = [p["max_D_twin"] for p in pairs.values() if p["max_D_twin"] is not None]
    largest = max(measured) if measured else None
    # "complete": every pair ran against the archived twin A; each pair's own state says
    # whether it is within, exceeding or incomplete
    return {**record, "status": "complete", "first_step_displacement_norm": delta_norm,
            "first_step_update_norm": intended,
            "first_step_changed": sum(int((t != 0).sum()) for t in delta),
            "twin_A": twin_a, "pairs": pairs, "max_D_twin": largest,
            "max_D_twin_over_K_star": _over(largest, twins.k_star),
            "pair_states": {key: pair["state"] for key, pair in pairs.items()},
            "shared": {k: v for k, v in shared.items() if k not in ("residual", "monitor")},
            "numerics": numerics(env.device), "seconds": time.perf_counter() - started,
            "provenance": provenance(settings, runs)}  # fmt: skip


def reference_twin(
    cfg: DictConfig, twins: Twins, eval_at: Sequence[int], checkpoints: Path
) -> dict[str, Any]:
    """The float64 reference (Revisions 1 and 2, 8): float64 optimizer arithmetic if it fits
    under the memory rule, else float64 loss and accumulation only. A failed attempt is kept
    under ``failed_attempt``: its error, its rows and its D_twin readings so far, and its peaks
    (all failures are kept and reported)."""
    progress: dict[str, Any] = {}
    try:
        out = run_twin(cfg, twins, eval_at=eval_at, precision="float64", compare_dir=checkpoints,
                       progress=progress)  # fmt: skip
        out["mode"] = FLOAT64_FULL
        return out
    except torch.OutOfMemoryError as exc:
        failed = {"precision": "float64", "error": f"{type(exc).__name__}: {str(exc)[:4000]}",
                  "rows": list(progress.get("rows", [])),
                  "D_twin": list(progress.get("D_twin", [])),
                  **peak_memory(twins.env.device)}  # fmt: skip
    gc.collect()
    if twins.env.device.type == "cuda":
        torch.cuda.empty_cache()
    twins.reset()
    out = run_twin(cfg, twins, eval_at=eval_at, precision="float64_accumulation",
                   compare_dir=checkpoints)  # fmt: skip
    out["mode"] = FLOAT64_ACCUMULATION
    out["fallback_reason"] = failed["error"].splitlines()[0]
    out["failed_attempt"] = failed
    return out


FLOAT64_FULL = "float64 loss, accumulation and optimizer arithmetic"
FLOAT64_ACCUMULATION = "float64 loss and accumulation only (fp32 optimizer state)"


def check_inputs(settings: DictConfig, env: Environment, runs: Runs) -> tuple[list[str], list[str]]:
    """The DATA_DIR streams D1–D3 read, located as the runs locate them: seed 2000's training
    stream (enough chunks for the archived runs' steps), v₀'s preconditioner stream, and the
    scoring and monitor panels. Returns what was found and what is missing."""
    return stream_inputs(run_config(runs.regular[str(settings.diagnostic.d3.determinism)]), env)


def stream_inputs(cfg: DictConfig, env: Environment) -> tuple[list[str], list[str]]:
    """The DATA_DIR streams a run config's twins read: its seed's training stream (enough chunks
    for its steps), v₀'s preconditioner stream, and the scoring and monitor panels. Returns what
    was found and what is missing."""
    found, missing = [], []
    responses = int(cfg.train.steps) * int(cfg.train.batch_size)
    _, path, _ = locate_stream(cfg, env, "training", int(cfg.seed))
    need, have = -(-responses // int(cfg.teacher.chunk_size)), published_chunks(path)
    entry = f"training stream {path}: {have} chunks, {need} needed"
    (found if have >= need else missing).append(entry)
    for kind, seed in (("preconditioner", int(cfg.seed)), ("kl_benign", None),
                       ("kl_monitor", None)):  # fmt: skip
        _, path, _ = locate_stream(cfg, env, kind, seed)
        (found if (path / MANIFEST_FILE).is_file() else missing).append(f"{kind} stream {path}")
    return found, missing


# ------------------------------------------------------------------------------------ branches


def load_outputs(out: Path) -> tuple[dict | None, dict | None, dict[str, dict]]:
    """D1's result, D3's determinism record and D3's case records, as written."""

    def read(path: Path) -> dict | None:
        return json.loads(path.read_text()) if path.exists() else None

    cases = {p.stem: json.loads(p.read_text()) for p in sorted((out / "d3").glob("*.json"))
             if p.name != "determinism.json"}  # fmt: skip
    return read(out / "d1" / "result.json"), read(out / "d3" / "determinism.json"), cases


def decide_branch(
    settings: DictConfig,
    d1: Mapping[str, Any] | None,
    determinism: Mapping[str, Any] | None,
    cases: Mapping[str, Mapping[str, Any]],
    *,
    commit: str | None = None,
) -> dict[str, Any]:
    """The frozen decision branches (docs/study2-fa-control-diagnostic.md, "Decision
    branches", Revisions 2 and 3), read from the diagnostic's outputs. A branch is chosen, or
    excluded, only on complete evidence (Revision 3).

    **Validity first.** The outcome is ``incomplete``, listing why under ``missing`` or in
    ``facts.incomplete``, when:
    - an output is missing: D1, the determinism record, or a prescribed case;
    - an output was written at another commit than ``commit`` (HEAD; without it, D1's), or at
      a commit other than the other parts' (review I4);
    - the determinism check failed (Revision 2, 9), whatever any case says (it is checked
      before Branch A);
    - a prescribed case does not hold exactly the pairs p1–p4 and the reference;
    - a prescribed case is itself ``incomplete``: its twin A differs from its archived run;
    - a D1 shared-state check failed (Revision 3, 1; ``shared_state_checks``): the trainable
      set, v₀'s hash, the loss normalization, the initial optimizer state, or the recomputed
      (iii) against every control's recorded first gradient;
    - the path criterion is undecided: ‖(i)‖ or ‖(iii)‖ non-finite, or ‖(iii)‖ zero
      (``path_criterion``);
    - D1 is not ``complete``.

    **The states** (Revision 3, 3), recomputed here from the records: each D3 pair is within,
    exceeding or incomplete (``pair_state``; survival of fp32 rounding is required for within,
    Revision 3, 2), and the same-path float64 control is below, exceeding, at the bound or
    incomplete (``control_state``). Divergence counts as exceeding.

    **Then the branches,** in Revision 3, 4's order, with FA 1.21e-5 the failing case:
    1. **A** when any failing-case pair is exceeding;
    2. **incomplete** when any failing-case pair is incomplete;
    3. **incomplete** when the same-path control is incomplete (every failing-case pair is
       within from here on);
    4. **B** when the same-path control is exceeding;
    5. **F** when the same-path control is below, the path criterion holds, and every pair of
       every case in ``branches.regular_cases`` (every D3 case, AdamW 1.32e-5 included) is
       within;
    6. **incomplete** when the same-path control is below, the path criterion holds, no pair of
       those cases is exceeding, and some pair of FA 1e-5 or AdamW is incomplete;
    7. **ambiguous** otherwise (the path criterion false on finite norms, a pair of FA 1e-5 or
       AdamW exceeding, or the control at the bound): the expansion case (FA 1.46e-5) is added
       to D3 once (``expand``), then the outcome is reported without choosing.

    An incomplete outcome has no branch and no expansion. The expansion never changes the
    branch. The per-pair and control states, the shared-state checks, the path criterion,
    D_twin/K*, and each reference's mode and fallback reason are reported in ``facts``."""
    d = settings.diagnostic
    bound, k_star = float(d.bound), float(d.k_star)
    failing = str(d.branches.failing)
    regular_cases = [str(c) for c in d.branches.regular_cases]
    prescribed = [str(c) for c in d.d3.order]
    expansion = str(d.d3.expansion)
    eval_at = [int(t) for t in d.d3.eval_at]
    k_at = [int(t) for t in d.d1.same_path_control.k_at]
    survival_share = float(d.d3.survival_share)
    facts: dict[str, Any] = {"bound": bound, "K_star": k_star, "failing_case": failing}
    missing: list[str] = []
    incomplete: list[str] = []

    if d1 is None:
        missing.append("d1/result.json")
    if determinism is None:
        missing.append("d3/determinism.json")
    missing += [f"d3/{name}.json" for name in prescribed if name not in cases]

    # One commit for every part, HEAD's if given
    present = {"d1/result.json": d1, "d3/determinism.json": determinism,
               **{f"d3/{name}.json": record for name, record in cases.items()}}  # fmt: skip
    commits = {name: output_commit(record) for name, record in present.items()
               if record is not None}  # fmt: skip
    reference = commit if commit is not None else commits.get("d1/result.json")
    facts.update({"commit": reference, "commits": commits})
    stale = {name for name, written in commits.items() if written != reference}
    missing += [f"{name} (written at {commits[name]}, not {reference})" for name in sorted(stale)]
    usable = {name: record for name, record in cases.items() if f"d3/{name}.json" not in stale}

    current = determinism is not None and "d3/determinism.json" not in stale
    if current and not determinism.get("equal"):
        missing.append("D3 (stopped: the determinism check failed)")
    for name in prescribed:  # the expansion case is only reported (facts.expansion_status)
        record = usable.get(name)
        if record is None:
            continue
        if record.get("status") == "incomplete":
            incomplete.append(f"d3/{name}.json: {record.get('reason', 'incomplete')}")
        elif set(record.get("pairs") or {}) != set(PAIRS):
            missing.append(f"d3/{name}.json holds the pairs {sorted(record.get('pairs') or {})},"
                           f" not {list(PAIRS)}")  # fmt: skip
        elif not ((record.get("twin_A") or {}).get("archive") or {}).get("equal"):
            incomplete.append(f"d3/{name}.json: twin A differs from its archived run")
    if expansion in usable:
        facts["expansion_status"] = usable[expansion].get("status")
    if d1 is not None and "d1/result.json" not in stale:  # Revision 3, 1
        residual = d1.get("residual") or {}
        facts["iii_equals_record"] = residual.get("iii_equals_record") or {}
        facts["shared_state"] = shared_state_checks(d1, control_keys(settings))
        reasons = [f"d1/result.json: {SHARED_STATE_CHECKS[name]}"
                   for name, passed in facts["shared_state"].items() if not passed]  # fmt: skip
        path = facts["path_criterion"] = path_criterion(residual, float(d.d1.path_share_max))
        if path["state"] == UNDECIDED:
            reasons.append(f"d1/result.json: the path criterion is undecided (‖(i)‖ "
                           f"{path['norm_i']}, ‖(iii)‖ {path['norm_iii']})")  # fmt: skip
        if not reasons and d1.get("status") != "complete":
            reasons.append(f"d1/result.json: status {d1.get('status')!r}")
        incomplete += reasons
        control = control_state(d1.get("same_path_control") or {}, k_at=k_at, bound=bound)
        facts["same_path_control"] = control
        facts["same_path_control_state"] = control["state"]
    facts["twin_A_equal"] = {name: ((record.get("twin_A") or {}).get("archive") or {})
                             .get("equal") for name, record in usable.items()}  # fmt: skip
    facts["incomplete"] = incomplete

    def readable(record: Mapping[str, Any]) -> bool:
        twin_a = ((record.get("twin_A") or {}).get("archive") or {}).get("equal")
        complete = record.get("status") != "incomplete"
        return bool(complete and twin_a and set(record.get("pairs") or {}) == set(PAIRS))

    valid = {name for name, record in usable.items() if readable(record)}
    reported = [name for name in [*prescribed, expansion] if name in valid]
    details = {name: {key: pair_state(pair, perturbed=key != REFERENCE, eval_at=eval_at,
                                      bound=bound, survival_share=survival_share)
                      for key, pair in usable[name]["pairs"].items()}
               for name in reported}  # fmt: skip
    facts["pair_states"] = {name: {key: s["state"] for key, s in pairs.items()}
                            for name, pairs in details.items()}  # fmt: skip
    facts["pairs_survived"] = {name: {key: pairs[key]["survived"] for key in PERTURBATIONS}
                               for name, pairs in details.items()}  # fmt: skip
    facts["max_D_twin_over_K_star"] = {
        name: {key: _over(p.get("max_D_twin"), k_star) for key, p in usable[name]["pairs"].items()}
        for name in reported
    }  # fmt: skip
    facts["references"] = {
        name: {key: usable[name]["pairs"][REFERENCE].get(key)
               for key in ("mode", "fallback_reason")}
        for name in reported
    }  # fmt: skip
    out: dict[str, Any] = {"facts": facts, "missing": missing}
    if missing or incomplete:
        return {**out, "branch": "incomplete", "expand": None}

    # Revision 3, 4: the branches in order; facts.step names the step that decided
    states, control = facts["pair_states"], facts["same_path_control_state"]

    def outcome(step: int, branch: str, reason: str | None = None) -> dict[str, Any]:
        facts["step"] = step
        if reason is not None:
            incomplete.append(f"step {step}: {reason}")
        expand = expansion if branch == "ambiguous" and expansion not in usable else None
        if branch == "ambiguous" and expansion in valid:
            facts["expansion"] = {"case": expansion, "pair_states": states[expansion],
                                  "note": "reported without choosing"}  # fmt: skip
        return {**out, "branch": branch, "expand": expand}

    failing_states = states[failing]
    if EXCEEDING in failing_states.values():
        return outcome(1, "A")
    if INCOMPLETE in failing_states.values():
        unread = sorted(key for key, state in failing_states.items() if state == INCOMPLETE)
        return outcome(2, "incomplete", f"the pairs {unread} of {failing} are incomplete")
    if control == INCOMPLETE:
        return outcome(3, "incomplete", "the same-path control is incomplete")
    if control == EXCEEDING:
        return outcome(4, "B")
    every = [state for name in regular_cases for state in states[name].values()]
    if control == BELOW and facts["path_criterion"]["state"] == HOLDS:
        if all(state == WITHIN for state in every):
            return outcome(5, "F")
        if EXCEEDING not in every:  # some pair of FA 1e-5 or AdamW is incomplete
            unread = {name: sorted(key for key, state in states[name].items()
                                   if state == INCOMPLETE)
                      for name in regular_cases if INCOMPLETE in states[name].values()}  # fmt: skip
            return outcome(6, "incomplete", f"the pairs {unread} are incomplete")
    return outcome(7, "ambiguous")


# ----------------------------------------- the numerical check (amendment 4, section 1, fa12)

QUALIFIED, FAILING = "qualified", "failing"  # an (arm, rate)
NOT_QUALIFIED = "not_qualified"  # technical: incomplete after its one rerun; never instability
ACCEPTED, REJECTED = "accepted", "rejected"  # a same-path control
# An attempt or a case whose regular run an instability rule stopped (the ×10 gradient rule, say):
# evidence against the rate, which is then failing, and its FA control rejected
INSTABILITY = "instability"
ATTEMPTS = (1, 2)  # an incomplete case or control runs once more, in a fresh process
NUMERICAL_SPEC = ("study 2, amendment 4, section 1 (numerical qualification); "
                  "configs/experiments/numerical_check.yaml")  # fmt: skip
NUMERICAL_REFERENCE = ("float64 loss, accumulation and optimizer arithmetic; fallback float64 "
                       "loss and accumulation only")  # fmt: skip
ROUNDING_LABEL = "heuristic, not a bound on accumulated error"


# The settings' keys that NumericalScope reads; every other key of ``numerical`` is declarative
SCOPE_KEYS = frozenset({"seeds", "arms", "primary_arms", "rates_per_arm", "control_arm", "regular",
                        "eval_at", "perturbation", "k_star_factor", "bound_fraction",
                        "memory_rule_gb"})  # fmt: skip


class NumericalError(DiagnosticError):
    """The numerical check cannot run as specified (a rate set outside the scope, say)."""


@dataclass(frozen=True)
class NumericalScope:
    """The numerical check's frozen scope (configs/experiments/numerical_check.yaml): seeds,
    arms, the regular runs' experiment and role, the measurement steps, the perturbations'
    fixed norm and survival rule, the bound's definition and the memory rule."""

    seeds: tuple[int, ...]  # the first holds every selected rate and the controls
    arms: tuple[str, ...]  # optimizer names
    primary_arms: tuple[str, ...]  # a selection may hold these alone (pre-warmed dropped)
    rates_per_arm: int
    control_arm: str  # FA: one same-path control per selected rate of this arm
    experiment: str
    role: str
    steps: int
    protocol_version: int
    eval_at: tuple[int, ...]
    norm: float
    gaussian: tuple[tuple[str, int], ...]  # (p3, seed), (p4, seed)
    survival_share: float
    rounding_scale: float
    k_star_factor: float  # K*(U) = √(factor·U)
    bound_fraction: float  # the bound is this fraction of K*(U)
    memory_rule_gb: float
    # The settings' declarative keys (``evaluation: k_only`` and the generation flag beside it),
    # as JSON by key: compared like the rest, so that settings declaring another evaluation
    # are not this scope's (review nit)
    declared: tuple[tuple[str, str], ...]

    @classmethod
    def of(cls, settings: DictConfig) -> NumericalScope:
        n = settings.numerical
        scope = cls(
            seeds=tuple(int(s) for s in n.seeds), arms=tuple(str(a) for a in n.arms),
            primary_arms=tuple(str(a) for a in n.primary_arms),
            rates_per_arm=int(n.rates_per_arm), control_arm=str(n.control_arm),
            experiment=str(n.regular.experiment), role=str(n.regular.role),
            steps=int(n.regular.steps), protocol_version=int(n.regular.protocol_version),
            eval_at=tuple(int(t) for t in n.eval_at), norm=float(n.perturbation.norm),
            gaussian=tuple((str(k), int(v)) for k, v in n.perturbation.gaussian.items()),
            survival_share=float(n.perturbation.survival_share),
            rounding_scale=float(n.perturbation.rounding_scale),
            k_star_factor=float(n.k_star_factor), bound_fraction=float(n.bound_fraction),
            memory_rule_gb=float(n.memory_rule_gb),
            declared=tuple(sorted(
                (str(key), json.dumps(value, sort_keys=True))
                for key, value in OmegaConf.to_container(n, resolve=True).items()
                if key not in SCOPE_KEYS)),
        )  # fmt: skip
        if dict(scope.declared).get("evaluation") != json.dumps("k_only"):
            raise NumericalError(f"The numerical check is K-only: its settings declare "
                                 f"{dict(scope.declared)}")  # fmt: skip
        if (scope.control_arm not in scope.primary_arms
                or not set(scope.primary_arms) < set(scope.arms)
                or len(set(scope.seeds)) != len(scope.seeds)):  # fmt: skip
            raise NumericalError(f"Inconsistent scope: {scope}")
        if [k for k, _ in scope.gaussian] != list(PERTURBATIONS[2:]):
            raise NumericalError(f"The Gaussian directions must be {PERTURBATIONS[2:]}")
        if not scope.eval_at or max(scope.eval_at) > scope.steps or not scope.norm > 0:
            raise NumericalError(f"Inconsistent scope: {scope}")
        return scope

    @property
    def control_seed(self) -> int:
        return self.seeds[0]

    def k_star(self, upper: float) -> float:
        """K*(U) = √(5U): the geometric mean of the lower target 5 and U."""
        return math.sqrt(self.k_star_factor * upper)

    def bound(self, upper: float) -> float:
        """0.01·K*(U): the twins' bound, and the same-path control's."""
        return self.bound_fraction * self.k_star(upper)

    def policy(
        self, upper: float, gpu_code: Mapping[str, Any] | None, writers: Sequence[str]
    ) -> dict[str, Any]:
        """The numerical policy a qualification is attached to (with its batch and seeds): the
        GPU code the regular runs and the check executed (``code_identity.gpu_code_identity``:
        commit, file count, sha256; review fa13), the norm, the Gaussian directions' seeds, the
        survival rule and its retry, the bound, the steps and the reference's modes; beside it,
        for provenance, the commits that wrote the records read (``tool_commit``). Any change of
        it voids the qualification."""
        return {"gpu_code": None if gpu_code is None else dict(gpu_code),
                "tool_commit": sorted(writers), "norm": self.norm, "gaussian": dict(self.gaussian),
                "survival_share": self.survival_share, "retry_scale": self.rounding_scale,
                "bound": self.bound(upper), "steps": list(self.eval_at),
                "reference": NUMERICAL_REFERENCE}  # fmt: skip

    def same_policy(self, settings: Mapping[str, Any] | None) -> bool:
        """Whether a record's ``provenance.settings`` are this scope's: every value but the
        memory rule (which only decides whether the reference falls back, and that is recorded
        per case), the declarative keys included (``declared``: settings that declare a
        generation the check never makes are not this scope's)."""
        try:
            other = NumericalScope.of(OmegaConf.create({"numerical": dict(settings or {})}))
        except Exception:  # missing or malformed settings: not this scope's
            return False
        return replace(other, memory_rule_gb=self.memory_rule_gb) == self


def rate_key(optimizer: str, lr: float) -> str:
    """An (arm, rate)'s key in the verdict: ``<optimizer>|<lr:g>``."""
    return f"{optimizer}|{lr:g}"


@dataclass(frozen=True)
class NumericalCase:
    """A twin case: the regular run of ``optimizer`` at ``lr`` on ``seed`` at batch ``batch``
    (experiment and role of the scope), checked against the bound of ``upper`` (U)."""

    seed: int
    optimizer: str
    lr: float
    batch: int
    upper: float

    @property
    def name(self) -> str:
        return f"s{self.seed}-{self.optimizer}-{self.lr:g}"

    @property
    def key(self) -> str:
        return rate_key(self.optimizer, self.lr)

    def attempt_path(self, attempt: int) -> str:
        """An attempt's record, relative to the output directory."""
        return f"twins/s{self.seed}/{self.optimizer}-{self.lr:g}/attempt-{attempt}.json"


def control_path(lr: float, attempt: int) -> str:
    """A same-path control's attempt record, relative to the output directory."""
    return f"controls/{lr:g}/attempt-{attempt}.json"


def selected_rates(
    scope: NumericalScope, rates: Mapping[str, Sequence[float]]
) -> dict[str, list[float]]:
    """The selected rates, ascending, by arm in the scope's order: of every arm of the scope, or
    of the primary arms alone (a selection without the pre-warmed arm, the candidate's
    fallback); any other set of arms is refused. Each arm needs exactly its ``rates_per_arm``
    finite, positive rates, distinct as their keys print them. The rates come from the caller
    (a selection record or the command line), never from the code."""
    if set(rates) not in (set(scope.arms), set(scope.primary_arms)):
        raise NumericalError(f"Rates are needed for exactly the arms {list(scope.arms)}, or the "
                             f"primary arms {list(scope.primary_arms)} alone; got "
                             f"{sorted(rates)}")  # fmt: skip
    out = {}
    for arm in (arm for arm in scope.arms if arm in rates):
        values = sorted(float(r) for r in rates[arm])
        if (len(values) != scope.rates_per_arm or not all(_number(r) and r > 0 for r in values)
                or len({f"{r:g}" for r in values}) != len(values)):  # fmt: skip
            raise NumericalError(f"{arm} needs {scope.rates_per_arm} distinct positive rates, "
                                 f"got {list(rates[arm])}")  # fmt: skip
        out[arm] = values
    return out


def _upper(upper: float) -> None:
    if not (_number(upper) and upper > 0):
        raise NumericalError(f"U must be positive, got {upper!r}")


def numerical_cases(
    settings: DictConfig, rates: Mapping[str, Sequence[float]], upper: float, *, batch: int
) -> list[NumericalCase]:
    """The frozen scope's cases: every selected (arm, rate) on the first numerical seed, then
    each selected arm's U-rate (its largest selected rate) on each other seed; 15 with three
    arms of three rates on three seeds, 10 with the primary arms alone."""
    scope = NumericalScope.of(settings)
    _upper(upper)
    selected = selected_rates(scope, rates)
    first, *others = scope.seeds
    cases = [NumericalCase(first, arm, lr, int(batch), upper)
             for arm in selected for lr in selected[arm]]  # fmt: skip
    cases += [NumericalCase(seed, arm, selected[arm][-1], int(batch), upper)
              for seed in others for arm in selected]  # fmt: skip
    return cases


def replacement_cases(
    settings: DictConfig, arm: str, rate: float, is_u_rate: bool, *, batch: int, upper: float
) -> list[NumericalCase]:
    """The cases a replacement rate brings: one on the first numerical seed, and one on each
    other seed when it is its arm's new U-rate. A replacement FA rate also gets its same-path
    control on the first seed. The check over the re-solved selection (``numerical_cases``)
    holds these, keeps every record already written at this commit, and runs the rest."""
    scope = NumericalScope.of(settings)
    _upper(upper)
    if arm not in scope.arms or not (_number(rate) and rate > 0):
        raise NumericalError(f"No replacement case for {arm} at {rate!r}")
    seeds = scope.seeds if is_u_rate else scope.seeds[:1]
    return [NumericalCase(seed, arm, float(rate), int(batch), upper) for seed in seeds]


def numerical_root(batch: int, upper: float, root: Path | None = None) -> Path:
    """``RUNS_DIR/numerical/b{B}-u{U}/``: the check's outputs and its verdict."""
    if float(upper) != int(upper):
        raise NumericalError(f"U must be an integer, got {upper!r}")
    return (runs_root() if root is None else root) / "numerical" / f"b{int(batch)}-u{int(upper)}"


def locate_case(root: Path, scope: NumericalScope, case: NumericalCase) -> Path:
    """The case's one finished regular run, by its archived config (``find_run``)."""
    return find_run(root, scope.experiment, seed=case.seed, batch=case.batch,
                    optimizer=case.optimizer, lr=case.lr, control=None,
                    role=scope.role)  # fmt: skip


def conformance(
    cfg: DictConfig, run_dir: Path, scope: NumericalScope, case: NumericalCase, *, stop: bool = True
) -> list[str]:
    """How a case's regular run departs from the prescribed one: its experiment, role, seed,
    batch, arm, rate, protocol version, planned steps, and (with ``stop``) a terminal record of
    every step taken (a run that stopped early cannot be twin A). Empty when it conforms. Only
    the config and the terminal record are read: nothing of an evaluation of harm, which K-only
    runs lack. The parts check it without ``stop`` first, then the instability stop
    (``instability_stop``), then the stop: a run that departs otherwise is a technical
    departure, even when an instability rule stopped it."""
    expected = {
        "experiment": (cfg.get("experiment"), scope.experiment),
        "study2_short.role": (OmegaConf.select(cfg, "study2_short.role", default=None),
                              scope.role),
        "seed": (cfg.get("seed"), case.seed),
        "train.batch_size": (cfg.train.batch_size, case.batch),
        "optimizer.name": (cfg.optimizer.name, case.optimizer),
        "control": (cfg.get("control"), None),
        "protocol_version": (cfg.get("protocol_version"), scope.protocol_version),
        "train.steps": (cfg.train.steps, scope.steps),
    }  # fmt: skip
    problems = [f"{key} is {have!r}, not {want!r}" for key, (have, want) in expected.items()
                if have != want]  # fmt: skip
    if not math.isclose(float(cfg.optimizer.lr), case.lr, rel_tol=1e-9):
        problems.append(f"optimizer.lr is {cfg.optimizer.lr}, not {case.lr:g}")
    if not stop:
        return problems
    terminal = json.loads((run_dir / "terminal.json").read_text())
    reason = (terminal.get("stop") or {}).get("reason")
    if reason != "steps" or terminal.get("steps") != scope.steps:
        problems.append(f"the run stopped by {reason!r} after {terminal.get('steps')} steps, "
                        f"not after its {scope.steps}")  # fmt: skip
    return problems


def instability_stop(run_dir: Path) -> str | None:
    """Why the regular run stopped, when an instability rule stopped it (its terminal record's
    ``stop.kind`` is ``instability``, or its reason one of ``schedule.INSTABILITY``: the ×10
    gradient rule, a non-finite loss, gradient, weight or K, K above its maximum): evidence
    against the rate, not a technical failure. None otherwise."""
    terminal = json.loads((run_dir / "terminal.json").read_text())
    stop = terminal.get("stop") or {}
    if stop.get("kind") != "instability" and stop.get("reason") not in RUN_INSTABILITY:
        return None
    detail = f", {json.dumps(stop['detail'])}" if stop.get("detail") else ""
    return (f"instability: the regular run stopped by {stop.get('reason')!r} after "
            f"{terminal.get('steps')} steps{detail}")  # fmt: skip


def run_identity(run_dir: Path) -> dict[str, Any]:
    """A regular run as a record names it: its place under the runs root
    (``<experiment>/<run id>``, so that RUNS_DIR may be restored elsewhere), run id, config hash
    (``meta.json``) and the sha256 of the records the check compares or reads (config, metrics,
    monitor, terminal). The verdict counts a record only for the run located now (review of
    Task 1, M3 and N3)."""
    digest = hashlib.sha256()
    for name in ("config.yaml", "metrics.jsonl", "monitor.jsonl", "terminal.json"):
        path = run_dir / name
        digest.update(f"{name}\0".encode())
        digest.update(path.read_bytes() if path.exists() else b"\0missing\0")
    meta = run_dir / "meta.json"
    config_hash = json.loads(meta.read_text()).get("config_hash") if meta.exists() else None
    return {"run": f"{run_dir.parent.name}/{run_dir.name}", "run_id": run_dir.name,
            "config_hash": config_hash, "sha256": digest.hexdigest()}  # fmt: skip


def code_check(run_dirs: Iterable[Path], tool_commit: str | None) -> dict[str, Any]:
    """Review fa13: the GPU-code identity (``code_identity.gpu_code_identity``) of the check's
    own commit and of the commit each regular run recorded (``meta.json``, its ``git`` entry).
    ``consistent`` when every run was made from a clean checkout of a commit whose GPU identity
    equals the check's; the check refuses to run otherwise (``require_code``). ``gpu_code`` is
    the runs' identity (their commit's when they share one, else the check's: equal to it)."""

    def identity(commit: str) -> dict[str, Any]:
        try:
            return gpu_code_identity(commit)
        except Exception as exc:  # an unknown commit, say: no identity, so no run
            return {"commit": commit, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}

    if tool_commit:
        tool = identity(tool_commit)
    else:
        tool = {"commit": None, "error": "the check has no commit"}
    problems = [f"the check's GPU code: {tool['error']}"] if "error" in tool else []
    runs: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    for run_dir in run_dirs:
        recorded = json.loads((Path(run_dir) / "meta.json").read_text()).get("git") or {}
        commit, dirty = recorded.get("commit"), recorded.get("dirty")
        runs[str(run_dir)] = {"commit": commit, "dirty": dirty}
        if commit is None or dirty is not False:
            problems.append(f"{Path(run_dir).name} was made from {commit} (dirty: {dirty}), not "
                            "a clean checkout of a commit")  # fmt: skip
            continue
        if commit not in identities:
            identities[commit] = identity(commit)
        found = identities[commit]
        if "error" in found:
            problems.append(f"{Path(run_dir).name}: the GPU code of {commit}: {found['error']}")
        elif "error" not in tool and found["sha256"] != tool["sha256"]:
            problems.append(f"{Path(run_dir).name} was made at {commit}, whose GPU code differs "
                            f"from the check's at {tool['commit']}")  # fmt: skip
    if not runs:
        problems.append("no regular run to compare with")
    commits = {run["commit"] for run in runs.values()}
    gpu_code = None
    if not problems:
        gpu_code = identities[commits.pop()] if len(commits) == 1 else tool
    return {"consistent": not problems, "gpu_code": gpu_code, "tool": tool, "runs": runs,
            "identities": identities, "problems": problems}  # fmt: skip


def require_code(code: Mapping[str, Any] | None) -> None:
    """Refuse to run unless the regular runs' GPU code is the check's (``code_check``)."""
    if code is None or not code.get("consistent"):
        problems = (code or {}).get("problems") or ["no code check"]
        raise NumericalError("The regular runs' GPU code is not the check's: "
                             + "; ".join(problems))  # fmt: skip


# The run environment a regular run's meta.json records (``tracking.Run``), compared field by
# field with the check's own (the final review of gate B, I3): twin A reproduces its run bit for
# bit only on the same GPU, driver, CUDA, Python and packages, so another environment would turn
# a technical difference into not_qualified for every case.
ENVIRONMENT_FIELDS = ("hardware.gpus", "hardware.driver", "hardware.torch_cuda", "python")
NOT_RECORDED = "not recorded"


def current_environment() -> dict[str, Any]:
    """This process's run environment as a run's meta.json records it (``tracking.Run``): the
    GPU names (as a set: the machine may hold several of one kind), the driver and torch's CUDA
    (``tracking.hardware``), the Python version, the tracked package versions and the container
    image (``CONTAINER_IMAGE``, None when unset)."""
    found = hardware()
    image = os.environ.get(LAUNCH_ENV["container_image"]) or None
    return {"hardware.gpus": sorted(set(found["gpus"])), "hardware.driver": found["driver"],
            "hardware.torch_cuda": found["torch_cuda"], "python": platform.python_version(),
            "packages": package_versions(), "launch.container_image": image}  # fmt: skip


def _recorded(meta: Mapping[str, Any], field: str) -> tuple[bool, Any]:
    """Whether ``meta.json`` holds the dotted ``field``, and its value."""
    node: Any = meta
    for part in field.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _absent_here(value: Any) -> bool:
    return value is None or value == [] or value == {}


def environment_departures(
    meta: Mapping[str, Any], here: Mapping[str, Any]
) -> list[tuple[str, Any, Any]]:
    """How a run's recorded environment (its ``meta.json``) departs from ``here``
    (``current_environment``), as (field, recorded, here). A field the run records is compared
    with this environment's value; a field it does not record (an older tracker's meta.json) is
    a departure when this environment records a value for it, since the run cannot show it ran
    in this one, and none when this environment records none either. The packages are compared
    one by one, the same way. The container image is compared only when the run recorded one:
    the operator sets ``CONTAINER_IMAGE``, and a run without it recorded none."""
    out = []
    for field in ENVIRONMENT_FIELDS:
        present, value = _recorded(meta, field)
        want = here[field]
        if present and field == "hardware.gpus" and isinstance(value, list):
            value = sorted(set(value))
        if present and value != want:
            out.append((field, value, want))
        elif not present and not _absent_here(want):
            out.append((field, NOT_RECORDED, want))
    present, packages = _recorded(meta, "packages")
    packages = packages if present and isinstance(packages, Mapping) else {}
    for name in sorted({*packages, *here["packages"]}):
        if name not in packages:
            out.append((f"packages.{name}", NOT_RECORDED, here["packages"][name]))
        elif packages[name] != here["packages"].get(name):
            out.append((f"packages.{name}", packages[name], here["packages"].get(name)))
    present, image = _recorded(meta, "launch.container_image")
    if present and image is not None and image != here["launch.container_image"]:
        out.append(("launch.container_image", image, here["launch.container_image"]))
    return out


def environment_check(
    run_dirs: Iterable[Path], here: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The final review of gate B, I3: each regular run's recorded environment against the
    check's (``environment_departures``; ``here``: ``current_environment`` by default).
    ``consistent`` when no run departs; ``problems`` name each departing field, with the runs
    that recorded each value. The parts that run the check refuse otherwise, recording
    nothing (scripts/numerical_check.py)."""
    here = current_environment() if here is None else dict(here)
    departing: dict[tuple[str, str, str], list[str]] = {}
    count = 0
    for run_dir in run_dirs:
        count += 1
        meta_path = Path(run_dir) / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        for field, have, want in environment_departures(meta, here):
            key = (field, json.dumps(have), json.dumps(want))
            departing.setdefault(key, []).append(Path(run_dir).name)
    problems = [f"{field}: {have} recorded by {len(names)} regular run(s) ({names[0]}"
                f"{', …' if len(names) > 1 else ''}), {want} here"
                for (field, have, want), names in sorted(departing.items())]  # fmt: skip
    if not count:
        problems.append("no regular run to compare with")
    return {"consistent": not problems, "here": here, "problems": problems}


# --------------------------------- voiding a numerical check after a GPU-path change (gate B, I2)

VOID_DIR = "void"  # RUNS_DIR/void/<the old GPU code's sha256, first 12>/: what ``void`` moved
# The roles of S3 (validation, confirmation, the time course): a run of one at the batch refuses
# the void (the coordinator's ruling on concern C1)
VOID_REFUSED_ROLES = ("validation", "confirmation", "timecourse")
CAMPAIGNS_DIR = "campaigns"  # RUNS_DIR/campaigns/: where the operator keeps the campaign files


def _journal_starts(log: Path) -> list[dict[str, Any]]:
    """The start records of one launch log (a line cut short by a crash is skipped, as
    ``launch.launch_records`` skips it)."""
    starts = []
    for line in log.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("event") == "start":
            starts.append(record)
    return starts


def _void_journals(root: Path, moved: set[str]) -> tuple[list[str], list[str]]:
    """The launcher's record of the campaigns whose start records name a run in ``moved``
    (``<experiment>/<run id>`` under ``root``): each campaign's log ``launch/<name>.jsonl``, its
    attempts' logs ``launch/<name>/``, and its file under ``campaigns/`` (a YAML whose ``name``
    is the campaign's) with its ``-driver.sh`` beside it when present; and the problems that
    stop the void: such a log that also names a run directory left in place."""
    paths, blocking = [], []
    for log in sorted((root / "launch").glob("*.jsonl")):
        named = {f"{r.get('experiment')}/{r.get('run_id')}" for r in _journal_starts(log)}
        if not named & moved:
            continue
        left = sorted(n for n in named - moved if (root / n).exists())
        if left:
            blocking.append(f"launch/{log.name} names runs that void moves and {left}, which it "
                            "does not: void them by hand, with a record")  # fmt: skip
        paths.append(log.relative_to(root).as_posix())
        attempts = log.with_suffix("")
        if attempts.is_dir():
            paths.append(attempts.relative_to(root).as_posix())
        for file in sorted((root / CAMPAIGNS_DIR).glob("*.y*ml")):
            try:
                name = (yaml.safe_load(file.read_text()) or {}).get("name")
            except (OSError, ValueError, AttributeError, yaml.YAMLError):
                continue  # not a campaign file
            if name == log.stem:
                paths.append(file.relative_to(root).as_posix())
                driver = file.with_name(f"{file.stem}-driver.sh")
                if driver.is_file():
                    paths.append(driver.relative_to(root).as_posix())
    return paths, blocking


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def void_plan(
    root: Path, scope: NumericalScope, batch: int, upper: float, state: Mapping[str, Any]
) -> dict[str, Any]:
    """What ``void_numerical`` would move, read only (the final review of gate B, I2): after a
    GPU-path change the check cannot run on the numerical runs made under the old GPU code (the
    code check refuses them), and the selection would wait at a stale check forever. Under
    ``root`` (RUNS_DIR): every run of the scope's experiment at ``batch`` whose role is the
    scope's (numerical), or that runs on a numerical seed or evaluates K only, and whose recorded
    commit's GPU code (``code_identity.gpu_code_identity``) differs from HEAD's (``state``, a
    clean checkout); the check's directory ``numerical/b{B}-u{U}/``, if there; and the launcher's
    record of the campaigns that made those runs (the coordinator's ruling on concern C2), so
    that a fresh campaign at the same rates, with the same cell names and config hashes, starts
    with every attempt and can be given up: each ``launch/<campaign>.jsonl`` whose start records
    name a run moved, its attempts' logs ``launch/<campaign>/``, and the campaign's own file
    under ``campaigns/`` (a YAML whose ``name`` is the campaign's), with its ``-driver.sh`` beside
    it when present (neither the launcher nor the selection reads those files). They go to
    ``void/<old GPU code's sha256, first 12>/``, at their paths under ``root``. ``problems``
    (each a refusal): a checkout that is not clean; a run of the role validation, confirmation
    or timecourse at ``batch`` (S3 has started: a GPU-path change after that is outside this
    procedure, the coordinator's ruling on concern C1); a check directory ``numerical/b{B}-u*/``
    of another U at ``batch`` (the numerical runs at B serve that check too: the coordinator's
    ruling on the re-review's N1); no numerical run whose GPU code differs; such a run of
    another role; a run that records no clean commit or whose commit has no readable GPU code;
    old runs of several GPU codes; a journal to move that also names a run left in place; a
    destination that exists. A missing ``numerical/b{B}-u{U}/`` is allowed (a GPU-path change
    can fall between the K-only runs and their check): ``check_directory`` then says "no check
    directory", and so does the record."""
    problems: list[str] = []
    commit, dirty = state.get("commit"), state.get("dirty")
    head: dict[str, Any] | None = None
    if commit is None or dirty is not False:
        problems.append(f"the checkout is not clean ({commit}, dirty: {dirty}): commit first")
    else:
        try:
            head = gpu_code_identity(commit)
        except Exception as exc:
            problems.append(f"HEAD's GPU code: {type(exc).__name__}: {str(exc)[:300]}")
    identities: dict[str, dict[str, Any]] = {}
    moved, kept, later = [], [], []
    for run_dir in sorted(p for p in (root / scope.experiment).glob("*") if p.is_dir()):
        try:
            cfg = OmegaConf.load(run_dir / "config.yaml")
        except (OSError, ValueError):
            continue  # an attempt that left no config: not a run the check reads
        if OmegaConf.select(cfg, "train.batch_size", default=None) != batch:
            continue
        role = OmegaConf.select(cfg, "study2_short.role", default=None)
        if role in VOID_REFUSED_ROLES:
            later.append(f"{scope.experiment}/{run_dir.name}")
            continue
        if not (role == scope.role or cfg.get("seed") in scope.seeds
                or OmegaConf.select(cfg, "eval.k_only", default=False) is True):  # fmt: skip
            continue
        name = f"{scope.experiment}/{run_dir.name}"
        meta_path = run_dir / "meta.json"
        git = (json.loads(meta_path.read_text()).get("git") or {}) if meta_path.is_file() else {}
        made, made_dirty = git.get("commit"), git.get("dirty")
        if made is None or made_dirty is not False:
            problems.append(f"{name} records no clean commit ({made}, dirty: {made_dirty}): its "
                            "GPU code is unknown; move it by hand, with a record")  # fmt: skip
            continue
        if made not in identities:
            try:
                identities[made] = gpu_code_identity(made)
            except Exception as exc:  # an unknown commit (a shallow clone, say)
                identities[made] = {"commit": made, "error": f"{type(exc).__name__}: "
                                                            f"{str(exc)[:300]}"}  # fmt: skip
        identity = identities[made]
        if "error" in identity:
            problems.append(f"{name}: the GPU code of {made}: {identity['error']}")
        elif head is not None and identity["sha256"] == head["sha256"]:
            kept.append(name)
        elif head is not None:
            if role != scope.role:
                problems.append(f"{name} has the role {role!r}, not {scope.role!r}: void moves "
                                "the numerical runs only")  # fmt: skip
            moved.append({"run": name, "commit": made, "gpu_code": identity})
    if later:
        problems.insert(0, f"{len(later)} run(s) of the role validation, confirmation or "
                           f"timecourse at B {batch} ({later[0]}{', …' if len(later) > 1 else ''})"
                           ": S3 has started. A GPU-path change after S3 has started is outside "
                           "this procedure and goes to the coordinator: voiding those runs means "
                           "spending on S3 again, which is a budget decision")  # fmt: skip
    old = {run["gpu_code"]["sha256"]: run["gpu_code"] for run in moved}
    if head is not None and not moved:
        problems.append(f"no numerical run at B {batch} has GPU code other than HEAD's "
                        f"({head['sha256'][:12]}): nothing to void")  # fmt: skip
    if len(old) > 1:
        problems.append(f"the runs to move have {len(old)} GPU codes "
                        f"({sorted(s[:12] for s in old)}): void them by hand, with a "
                        "record")  # fmt: skip
    directory = numerical_root(batch, upper, root)
    # The numerical runs at B serve every U's check: another U's check directory at B holds
    # verdicts (and marks) the replay must still read, which voiding the runs would strand (the
    # coordinator's ruling on the re-review's N1)
    others = sorted(p.relative_to(root).as_posix()
                    for p in directory.parent.glob(f"b{int(batch)}-u*")
                    if p.is_dir() and p != directory)  # fmt: skip
    if others:
        problems.append(f"the check directories {others} of another U at B {batch} exist: the "
                        "numerical runs at B serve them too, and voiding them would leave those "
                        "verdicts unread; this case goes to the coordinator")  # fmt: skip
    paths = [run["run"] for run in moved]
    check = directory.relative_to(root).as_posix() if directory.is_dir() else None
    if check is not None:
        paths.append(check)
    journals, blocking = _void_journals(root, {run["run"] for run in moved})
    problems += blocking
    paths += journals
    old_code = next(iter(old.values())) if len(old) == 1 else None
    destination = None if old_code is None else root / VOID_DIR / old_code["sha256"][:12]
    if destination is not None:
        problems += [f"{destination / path} exists: nothing is overwritten" for path in paths
                     if (destination / path).exists()]  # fmt: skip
    return {"batch": int(batch), "U": upper, "head": head, "old_gpu_code": old_code,
            "runs": moved, "kept": kept, "paths": paths,
            "check_directory": check or "no check directory",
            "destination": None if destination is None else str(destination),
            "problems": problems}  # fmt: skip


def void_numerical(root: Path, plan: Mapping[str, Any], state: Mapping[str, Any]) -> Path:
    """Move what ``void_plan`` found (``plan``, without problems) into its destination, never
    deleting: first the inventory ``inventory-<time>.tsv`` there (every file moved, its path
    under RUNS_DIR, size and sha256), then the moves, then the record ``void-<time>.json`` (what
    was moved, from and to which GPU code, by which checkout). Returns the record's path."""
    if plan["problems"] or plan["destination"] is None:
        raise NumericalError("void refuses: " + "; ".join(plan["problems"] or ["no destination"]))
    destination = Path(plan["destination"])
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    lines = ["path\tsize\tsha256\n"]
    for path in plan["paths"]:
        source = root / path
        files = [source] if source.is_file() else sorted(p for p in source.rglob("*")
                                                         if p.is_file())  # fmt: skip
        lines += [f"{p.relative_to(root).as_posix()}\t{p.stat().st_size}\t{_file_sha256(p)}\n"
                  for p in files]  # fmt: skip
    destination.mkdir(parents=True, exist_ok=True)
    inventory = destination / f"inventory-{stamp}.tsv"
    inventory.write_text("".join(lines))
    for path in plan["paths"]:
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(root / path, target)
    record = destination / f"void-{stamp}.json"
    write_json(record, {
        "what": "the numerical runs and the check made under GPU code other than HEAD's, with "
                "the launcher's logs and the files of the campaigns that made them, moved out of "
                "RUNS_DIR's experiment, numerical/, launch/ and campaigns/ by "
                "scripts/numerical_check.py void (the final review of gate B, I2; the "
                "coordinator's ruling on concern C2); nothing deleted",
        "batch": plan["batch"], "U": plan["U"], "moved": plan["paths"],
        "check_directory": plan["check_directory"],
        "runs": plan["runs"], "kept": plan["kept"], "old_gpu_code": plan["old_gpu_code"],
        "head_gpu_code": plan["head"], "inventory": inventory.name, "code": dict(state),
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })  # fmt: skip
    return record


def guarded_prompt_sets(env: Environment, word: str, owner: str) -> Environment:
    """The environment with every prompt set whose name contains ``word`` refused: a K-only part
    never reads such a set, and this makes any path to one raise."""
    original = env.prompt_set

    def prompt_set(name: str) -> list[dict]:
        if word in name:
            raise RuntimeError(f"{owner} must not read the prompt set {name!r}")
        return original(name)

    return replace(env, prompt_set=prompt_set)


def rounding_heuristic(theta: Iterable[torch.Tensor]) -> dict[str, Any]:
    """Report-only: the norm one fp32 rounding of θ would leave, sqrt(Σ (ulp(θ)/√12)²) over the
    coordinates (each rounding error uniform over ± half an ulp, the ulp the spacing above
    |θ|). A heuristic, not a bound on the numerical error accumulated over a run."""
    total, count = 0.0, 0
    for t in theta:
        a = t.detach().float().abs()
        ulp = torch.nextafter(a, torch.full_like(a, math.inf)).sub_(a).double()
        total += ulp.pow(2).sum().item() / 12
        count += a.numel()
    return {"norm": math.sqrt(total), "coordinates": count, "label": ROUNDING_LABEL,
            "estimate": "sqrt(sum over the trainable coordinates of (ulp(theta)/sqrt(12))^2), "
                        "fp32 ulps of theta_0"}  # fmt: skip


@torch.no_grad()
def first_update_direction(
    residual: Sequence[torch.Tensor], v0: Sequence[torch.Tensor], eps: float
) -> tuple[Vector, float]:
    """−P·g, P = 1/(√v₀ + ε): the direction of FA's first update for the gradient g (its
    bias-corrected first moment is g, so δ₁ = −η·P·g at any rate η). v₀ and g are widened to
    float64 first, so that P and the product are formed in float64; the result is kept in fp32
    per tensor. Returns it and its norm (float64 sums of the fp32 tensors)."""
    out = [(g.double() * preconditioner(v.double(), eps)).neg_().float()
           for g, v in zip(residual, v0, strict=True)]  # fmt: skip
    return out, _norm(out)


def seed_directions(
    scope: NumericalScope, twins: Twins, shared: dict[str, Any], eps: float
) -> dict[str, Any]:
    """The seed's perturbation directions and their scales to the scope's fixed norm, put in
    ``shared`` (``directions``, ``scales``): p1 = −P·g and p2 = +P·g, g the soft-target residual
    of ``twin_setup`` (generation-path targets, batch 0, θ₀), the same at every rate and arm of
    the seed; p3 and p4 the fixed Gaussian directions. The residual is released. Returns the
    seed's facts: the residual, the norms, the scales and the rounding heuristic."""
    residual = shared.pop("residual")
    direction, norm = first_update_direction(residual, twins.v0, eps)
    del residual
    by_name = dict(zip(twins.names, direction, strict=True))
    directions: dict[str, Callable[[str, Sequence[int]], torch.Tensor]] = {
        "p1": lambda name, shape: by_name[name],
        "p2": lambda name, shape: -by_name[name],
    }
    norms = {"p1": norm, "p2": norm}
    for key, seed in scope.gaussian:
        directions[key] = lambda name, shape, s=seed, k=key: gaussian(name, shape, s, k)
        norms[key] = gaussian_norm(twins.trainable, seed, key)
    # The fixed norm, whatever any control's displacement: no scale for a direction without a
    # finite, positive norm, whose pair is then incomplete
    scales = {key: scope.norm / n for key, n in norms.items() if _number(n) and n > 0}
    shared.update(directions=directions, scales=scales)
    theta0 = (twins.theta0[n] for n in twins.names)
    return {"v0_sha256": shared["v0_sha256"], "v0_stream": shared["v0_stream"],
            "residual": shared["control_residual"],
            "residual_norm": shared["control_residual_norm"], "epsilon": eps,
            "intended_norm": scope.norm, "direction_norms": norms, "scales": scales,
            "scoring_panel": shared["scoring_panel"],
            "rounding_heuristic": rounding_heuristic(theta0)}  # fmt: skip


def run_numerical_case(
    scope: NumericalScope,
    case: NumericalCase,
    twins: Twins,
    shared: Mapping[str, Any],
    run_dir: Path,
    checkpoints: Path,
    *,
    keep_checkpoints: bool,
    progress: dict[str, Any],
) -> dict[str, Any]:
    """One attempt of one case: twin A (the archived regular run, re-run and compared bit for
    bit), then p1–p4 at the scope's fixed norm and the float64 reference, each against twin A
    by D_twin at the scope's steps, each with its state (``pair_state``). A regular run stopped
    by an instability rule gives the attempt ``instability`` (``instability_stop``: evidence
    against the rate); one that does not conform otherwise, or a twin A that stopped or differs
    from its archive, makes the attempt incomplete; either way no pair is run. ``progress``
    holds twin A and the pairs as they are done."""
    eval_at, bound = list(scope.eval_at), scope.bound(case.upper)
    cfg = run_config(run_dir)
    problems = conformance(cfg, run_dir, scope, case, stop=False)
    if not problems:  # review N4: an instability stop is evidence only for a conforming run
        unstable = instability_stop(run_dir)
        if unstable is not None:
            return {"status": INSTABILITY, "reason": unstable, "pairs": {}}
        problems = conformance(cfg, run_dir, scope, case)
    if problems:
        return {
            "status": "incomplete",
            "pairs": {},
            "reason": "the regular run does not conform: " + "; ".join(problems),
        }
    try:
        twin_a = progress["twin_A"] = run_twin_a(cfg, twins, shared["monitor"], run_dir,
                                                 checkpoints, eval_at)  # fmt: skip
        if twin_a["stop"] is not None or not twin_a["archive"]["equal"]:
            reason = (f"twin A stopped: {twin_a['stop']}" if twin_a["stop"] is not None
                      else "twin A differs from its archived run")  # fmt: skip
            return {"status": "incomplete", "reason": reason, "twin_A": twin_a, "pairs": {}}
        pairs = run_pairs(cfg, twins, shared["directions"], shared["scales"], checkpoints,
                          eval_at=eval_at, bound=bound, survival_share=scope.survival_share,
                          rounding_scale=scope.rounding_scale,
                          pairs=progress.setdefault("pairs", {}))  # fmt: skip
    finally:
        if not keep_checkpoints:
            _remove_checkpoints(checkpoints)
        twins.reset()
    measured = [p["max_D_twin"] for p in pairs.values() if p.get("max_D_twin") is not None]
    largest = max(measured) if measured else None
    reference = pairs[REFERENCE]
    return {
        "status": "complete",
        "twin_A": twin_a,
        "pairs": pairs,
        "pair_states": {key: pair["state"] for key, pair in pairs.items()},
        "reference_mode": reference.get("mode"),
        "fallback_reason": reference.get("fallback_reason"),
        "max_D_twin": largest,
        "max_D_twin_over_K_star": _over(largest, twins.k_star),
    }


def host_record() -> dict[str, Any]:
    """The host memory when a record is written (the final review of gate B, M7), as the drift
    records it: this process's peak resident memory so far (``ru_maxrss``) and the free host
    memory (MemAvailable, the cgroup's limit less its working set, and the smaller), GB of 1e9
    bytes. The check holds θ₀, v₀, the second model and the p1 direction on the host (≈ 25-30 GB
    at 1.5B), and ≈ 5 GB more while a checkpoint is saved."""
    from .drift import host_memory, host_peak_rss_gb  # drift imports this module

    return {"peak_rss_gb": host_peak_rss_gb(), **host_memory()}


def numerical_provenance(settings: DictConfig, **extra: Any) -> dict[str, Any]:
    return {"spec": NUMERICAL_SPEC, "code": git_state(),
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": OmegaConf.to_container(settings.numerical, resolve=True),
            "hardware": hardware(), "packages": package_versions(), "host_memory": host_record(),
            **extra}  # fmt: skip


def _error(exc: BaseException) -> dict[str, Any]:
    return {"reason": f"error: {type(exc).__name__}: {str(exc)[:2000]}",
            "traceback": traceback.format_exc()[-8000:]}  # fmt: skip


def _release_device(device: torch.device) -> None:
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def _attempt_number(attempt: int) -> None:
    if attempt not in ATTEMPTS:
        raise NumericalError(f"Attempt {attempt}: a case or control has the attempts {ATTEMPTS}")


def run_numerical_twins(
    settings: DictConfig,
    env: Environment,
    root: Path,
    out: Path,
    cases: Sequence[NumericalCase],
    *,
    attempt: int,
    code: Mapping[str, Any] | None,
    keep_checkpoints: bool = False,
    setups: Sequence[NumericalCase] | None = None,
) -> dict[str, dict[str, Any]]:
    """Attempt ``attempt`` of each case, seed by seed: one ``twin_setup`` per seed, from the
    regular run of its first control-arm case among ``setups`` (the selection's cases; by
    default ``cases``): v₀ and the residual for p1 and p2, which no rate changes; then the
    seed's cases (``run_numerical_case``). Each attempt is written to
    ``NumericalCase.attempt_path`` under ``out``, whatever happened: a case that failed is its
    record, incomplete, with its error and the pairs done before it; a seed whose setup failed
    gives each of its cases that record; a case whose regular run is not found is recorded
    incomplete with why, and a seed none of whose cases has its run (a rerun due only for a run
    the launcher gave up on, say) is set up for nothing. Nothing runs unless ``code``
    (``code_check``) shows the regular runs' GPU code to be the check's (``require_code``).
    Returns the records by case name."""
    _attempt_number(attempt)
    require_code(code)
    scope = NumericalScope.of(settings)
    done: dict[str, dict[str, Any]] = {}

    def finish(case: NumericalCase, body: Mapping[str, Any], facts: Any = None) -> None:
        record = {"part": "twins", "case": case.name, "seed": case.seed,
                  "optimizer": case.optimizer, "lr": case.lr, "batch": case.batch,
                  "U": case.upper, "attempt": attempt, "bound": scope.bound(case.upper),
                  "K_star": scope.k_star(case.upper), "eval_at": list(scope.eval_at),
                  "intended_norm": scope.norm, **body, "seed_facts": facts,
                  "numerics": numerics(env.device),
                  "provenance": numerical_provenance(settings,
                                                     gpu_code=code["gpu_code"])}  # fmt: skip
        write_json(out / case.attempt_path(attempt), record)
        done[case.name] = record

    for seed in dict.fromkeys(case.seed for case in cases):
        mine = [case for case in cases if case.seed == seed]
        located: dict[str, Path] = {}
        unlocated: dict[str, str] = {}
        for case in mine:
            try:
                located[case.name] = locate_case(root, scope, case)
            except DiagnosticError as exc:
                unlocated[case.name] = f"the regular run is not found: {exc}"
        if not located:  # no run to set up for (the rerun of a run given up, say): recorded
            for case in mine:
                finish(case, {"status": "incomplete", "reason": unlocated[case.name],
                              "pairs": {}})  # fmt: skip
            continue
        setup = None
        for candidate in setups or cases:
            if candidate.seed == seed and candidate.optimizer == scope.control_arm:
                try:
                    setup = located.get(candidate.name) or locate_case(root, scope, candidate)
                    break
                except DiagnosticError:
                    continue
        if setup is None:
            for case in mine:
                reason = unlocated.get(case.name) or (
                    f"no {scope.control_arm} regular run of seed {seed} to set v₀ and p1, p2 up"
                )
                finish(case, {"status": "incomplete", "reason": reason, "pairs": {}})
            continue
        try:
            cfg = run_config(setup)
            with twin_setup(env, cfg, cfg, k_star=scope.k_star(mine[0].upper)) as (twins, shared):
                facts = {
                    "setup_run": str(setup),
                    **seed_directions(scope, twins, shared, float(cfg.optimizer.eps)),
                }
                for case in mine:
                    if case.name in unlocated:
                        finish(case, {"status": "incomplete", "reason": unlocated[case.name],
                                      "pairs": {}}, facts)  # fmt: skip
                        continue
                    progress: dict[str, Any] = {}
                    started = time.perf_counter()
                    checkpoints = out / "twins" / "checkpoints" / case.name
                    try:
                        body = run_numerical_case(scope, case, twins, shared, located[case.name],
                                                  checkpoints, keep_checkpoints=keep_checkpoints,
                                                  progress=progress)  # fmt: skip
                    except Exception as exc:  # recorded; the next case goes on
                        body = {"status": "incomplete", **_error(exc),
                                "twin_A": progress.get("twin_A"),
                                "pairs": progress.get("pairs", {})}  # fmt: skip
                        _release_device(env.device)
                    finish(case, {"regular_run": str(located[case.name]),
                                  "regular_run_identity": run_identity(located[case.name]),
                                  **body, "seconds": time.perf_counter() - started},
                           facts)  # fmt: skip
        except Exception as exc:  # the seed's setup: every case not yet written records it
            error = _error(exc)
            error["reason"] = f"the setup of seed {seed} failed: {error['reason']}"
            for case in mine:
                if case.name not in done:
                    finish(case, {"status": "incomplete", "pairs": {}, **error})
            _release_device(env.device)
    return done


def run_numerical_controls(
    settings: DictConfig,
    env: Environment,
    root: Path,
    out: Path,
    *,
    rates: Sequence[float],
    batch: int,
    upper: float,
    attempt: int,
    code: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Attempt ``attempt`` of the same-path float64 soft control (``same_path_run``, D1's) at
    each of ``rates`` (selected rates of the control arm), on the first numerical seed at
    ``batch``, for the regular run's steps, K on the scoring panel at the scope's steps against
    the bound of ``upper``. Each control uses its rate's regular run config, which must conform,
    and must read v₀'s stream and the training prefix that run read (``shared_state``, from the
    run's ``meta.json``); otherwise, or after an error, the attempt is incomplete. A regular run
    stopped by an instability rule gives ``instability`` instead (``instability_stop``), and no
    control runs. Each attempt names its regular run (``run_identity``) and is written to
    ``control_path``. Nothing runs unless ``code`` shows the regular runs' GPU code
    to be the check's (``require_code``). Returns the records by rate."""
    _attempt_number(attempt)
    require_code(code)
    scope = NumericalScope.of(settings)
    k_at, bound, k_star = list(scope.eval_at), scope.bound(upper), scope.k_star(upper)
    done: dict[str, dict[str, Any]] = {}
    setup: dict[str, Any] = {}
    for lr in sorted(float(r) for r in rates):
        case = NumericalCase(scope.control_seed, scope.control_arm, lr, int(batch), upper)
        record = {"part": "control", "case": case.name, "rate": lr, "seed": case.seed,
                  "batch": case.batch, "U": upper, "attempt": attempt, "bound": bound,
                  "K_star": k_star, "k_at": k_at}  # fmt: skip
        try:
            run_dir = locate_case(root, scope, case)
            cfg = run_config(run_dir)
            record.update(regular_run=str(run_dir), regular_run_identity=run_identity(run_dir))
            problems = conformance(cfg, run_dir, scope, case, stop=False)
            unstable = None if problems else instability_stop(run_dir)  # review N4
            problems = problems or conformance(cfg, run_dir, scope, case)
            if unstable is not None:  # evidence against the rate: the control is rejected
                record.update(status=INSTABILITY, state=INSTABILITY, reason=unstable)
            elif problems:
                record.update(
                    status="incomplete",
                    state=INCOMPLETE,
                    reason="the regular run does not conform: " + "; ".join(problems),
                )
            else:
                if not setup:  # v₀ once, from the first control's run; θ₀ kept on the host
                    student, trainable = _model(env, cfg)
                    names = [n for n, _ in trainable]
                    theta0 = {n: p.detach().cpu().clone() for n, p in trainable}
                    v0, stream = second_moment(cfg, env, student, [p for _, p in trainable])
                    setup.update(student=student, trainable=trainable, names=names,
                                 theta0=theta0, v0_stream=stream,
                                 v0_sha256=tensors_sha256(list(zip(names, v0, strict=True))),
                                 v0=[v.cpu() for v in v0])  # fmt: skip
                    del v0
                    gc.collect()
                set_trainable(setup["student"], setup["theta0"], setup["names"])
                result = same_path_run(env, cfg, setup["student"], setup["trainable"],
                                       setup["v0"], k_at=k_at, bound=bound,
                                       k_star=k_star)  # fmt: skip
                set_trainable(setup["student"], setup["theta0"], setup["names"])
                streams = ((json.loads((run_dir / "meta.json").read_text()).get("inputs") or {})
                           .get("streams") or {})  # fmt: skip
                shared_state = {
                    "v0_stream": {"control": setup["v0_stream"],
                                  "regular": streams.get("preconditioner")},
                    "training_stream": {"control": result["training_stream"],
                                        "regular": streams.get("training")},
                }  # fmt: skip
                for check in shared_state.values():
                    check["equal"] = check["control"] == check["regular"]
                unequal = [key for key, check in shared_state.items() if not check["equal"]]
                record.update(result, shared_state=shared_state, v0_sha256=setup["v0_sha256"],
                              first_gradient_norm=(result["rows"][0]["grad_norm"]
                                                   if result["rows"] else None))  # fmt: skip
                if unequal:
                    record.update(status="incomplete", reason=f"the control's {unequal} differ "
                                  "from its regular run's")  # fmt: skip
                else:
                    record["status"] = "complete"
        except Exception as exc:  # recorded: the attempt is incomplete
            record.update(status="incomplete", state=INCOMPLETE, **_error(exc))
            _release_device(env.device)
        record.update(
            numerics=numerics(env.device),
            provenance=numerical_provenance(settings, gpu_code=code["gpu_code"]),
        )
        write_json(out / control_path(lr, attempt), record)
        done[f"{lr:g}"] = record
    setup.clear()
    _release_device(env.device)
    return done


# ------------------------------------------------------------------ the numerical verdict


def _value(x: Any) -> str:
    return f"{x:.3g}" if _number(x) else str(x)


def _close(value: Any, target: float) -> bool:
    return _number(value) and math.isclose(value, target, rel_tol=1e-6)


def _same(record: Mapping[str, Any], identity: Mapping[str, Any], key: str, lr: float) -> bool:
    value = record.get(key)
    return all(record.get(k) == v for k, v in identity.items()) and (
        _number(value) and math.isclose(value, lr, rel_tol=1e-9)
    )


class RecordReader:
    """Which records the verdict and the parts read (review of Task 1, I1, I3 and M3). A record
    counts when:

    - this checkout is a clean checkout of a commit (HEAD);
    - it was written from a clean checkout of a commit whose GPU code is HEAD's
      (``code_identity.reuse_check(writer, HEAD)`` reusable): records stay valid across
      analysis-only commits, and are stale after a change of a GPU path;
    - it was written under the scope's frozen settings (``NumericalScope.same_policy``);
    - it names the regular run located now (``run_identity``: path, run id, config hash and the
      sha256 of the records compared); a complete record, or an instability one, must name one.

    Reading moves nothing. ``checks`` holds the reuse check of every writer commit met."""

    def __init__(self, scope: NumericalScope, state: Mapping[str, Any], root: Path) -> None:
        self.scope, self.state, self.root = scope, dict(state), Path(root)
        self.checks: dict[str, dict[str, Any]] = {}
        self._runs: dict[str, dict[str, Any]] = {}

    def reuse(self, writer: str) -> dict[str, Any]:
        if writer not in self.checks:
            try:
                self.checks[writer] = reuse_check(writer, self.state["commit"])
            except Exception as exc:  # an unknown commit, say: not reusable
                self.checks[writer] = {
                    "reusable": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
        return self.checks[writer]

    def run(self, case: NumericalCase) -> dict[str, Any]:
        """The case's regular run located now (``run_identity``), or why there is none."""
        if case.name not in self._runs:
            try:
                self._runs[case.name] = run_identity(locate_case(self.root, self.scope, case))
            except (DiagnosticError, OSError) as exc:
                self._runs[case.name] = {"error": str(exc)}
        return self._runs[case.name]

    def why_not(self, record: Mapping[str, Any] | None, case: NumericalCase) -> str | None:
        """None when ``record`` counts, else why not."""
        if record is None:
            return "no record"
        commit, dirty = self.state.get("commit"), self.state.get("dirty")
        if commit is None or dirty is not False:
            return f"read from {commit} (dirty: {dirty}), not a clean checkout of a commit"
        writer, writer_dirty = output_commit(record), output_dirty(record)
        if writer is None or writer_dirty is not False:
            return f"written from {writer} (dirty: {writer_dirty}), not a clean checkout"
        check = self.reuse(writer)
        if not check.get("reusable"):
            detail = check.get("error") or f"GPU paths changed: {check.get('changed')}"
            return f"written at {writer}, whose GPU code is not HEAD's ({detail})"
        if not self.scope.same_policy((record.get("provenance") or {}).get("settings")):
            return "written under other settings than the frozen scope's"
        named = record.get("regular_run_identity")
        if named is None and record.get("status") in ("complete", INSTABILITY):
            return "it names no regular run"
        if named is not None and named != self.run(case):
            return "its regular run is not the one located now (path, run id, config or records)"
        return None

    def writers(self) -> list[str]:
        return sorted(c for c, check in self.checks.items() if check.get("reusable"))


def _policy_problems(record: Mapping[str, Any], scope: NumericalScope) -> list[str]:
    """How an attempt's perturbations depart from the policy (review of Task 1, I1): each
    perturbed pair's first attempt at the fixed norm (1e-6 relative), a second attempt only as
    the single retry, exactly ``rounding_scale`` times the first."""
    problems = []
    if not _close(record.get("intended_norm"), scope.norm):
        problems.append(f"the intended norm {record.get('intended_norm')!r}, not {scope.norm:g}")
    for key, pair in (record.get("pairs") or {}).items():
        if key not in PERTURBATIONS:
            continue
        attempts = list(pair.get("rounding") or [])
        if not attempts:
            if pair.get("D_twin"):
                problems.append(f"{key} was measured without a recorded perturbation")
            continue
        first = attempts[0]
        if not _close(first.get("intended_norm"), scope.norm):
            problems.append(f"{key}'s perturbation has the norm {first.get('intended_norm')!r}, "
                            f"not {scope.norm:g}")  # fmt: skip
        if len(attempts) > 2:
            problems.append(f"{key} has {len(attempts)} rounding attempts, not at most 2")
        elif len(attempts) == 2:
            second = attempts[1]
            scaled = (_number(first.get("scale")) and _number(second.get("scale"))
                      and second["scale"] == first["scale"] * scope.rounding_scale)  # fmt: skip
            if not scaled or not _close(second.get("intended_norm"),
                                        scope.norm * scope.rounding_scale):  # fmt: skip
                problems.append(f"{key}'s retry is not {scope.rounding_scale:g} times its first "
                                "attempt")  # fmt: skip
    return problems


def attempt_state(
    record: Mapping[str, Any] | None, case: NumericalCase, *, reader: RecordReader
) -> dict[str, Any]:
    """One attempt of a case. Only a record that counts (``RecordReader``) and is of this case
    is ``present``. Then:

    - **instability** when its regular run was stopped by an instability rule;
    - **incomplete** when its perturbations depart from the policy (``_policy_problems``);
    - **exceeding** when any of its pairs is (``pair_state``, recomputed at the bound of U;
      divergence counts), whatever else the attempt shows;
    - **within** when it is complete (twin A equal to its archive, every pair of p1–p4 and the
      reference present, the reference in one of the policy's modes) and every pair is within;
    - **incomplete** otherwise."""
    scope = reader.scope
    why = reader.why_not(record, case)
    if why is not None:
        return {"state": INCOMPLETE, "present": False, "reasons": [why]}
    identity = {"seed": case.seed, "optimizer": case.optimizer, "batch": case.batch,
                "U": case.upper}  # fmt: skip
    if not _same(record, identity, "lr", case.lr):
        return {"state": INCOMPLETE, "present": False, "reasons": ["the record is of another case"]}
    if record.get("status") == INSTABILITY:
        return {"present": True, "state": INSTABILITY, "pair_states": {},
                "reference_mode": None, "fallback_reason": None,
                "reasons": [str(record.get("reason"))]}  # fmt: skip
    pairs = {key: pair for key, pair in (record.get("pairs") or {}).items() if key in PAIRS}
    states = {key: pair_state(pair, perturbed=key != REFERENCE, eval_at=list(scope.eval_at),
                              bound=scope.bound(case.upper), survival_share=scope.survival_share)
              for key, pair in pairs.items()}  # fmt: skip
    reference = pairs.get(REFERENCE) or {}
    out: dict[str, Any] = {"present": True,
                           "pair_states": {key: s["state"] for key, s in states.items()},
                           "reference_mode": reference.get("mode"),
                           "fallback_reason": reference.get("fallback_reason")}  # fmt: skip
    policy = _policy_problems(record, scope)
    if policy:
        return {**out, "state": INCOMPLETE, "reasons": [f"not the policy's: {p}" for p in policy]}
    exceeding = [key for key in PAIRS if states.get(key, {}).get("state") == EXCEEDING]
    if exceeding:
        return {
            **out,
            "state": EXCEEDING,
            "reasons": [
                f"{key} exceeding (max D_twin {_value(pairs[key].get('max_D_twin'))}"
                + (", diverged)" if states[key]["diverged"] else ")")
                for key in exceeding
            ],
        }
    reasons = []
    if record.get("status") != "complete":
        reasons.append(str(record.get("reason") or "incomplete"))
    elif not ((record.get("twin_A") or {}).get("archive") or {}).get("equal"):
        reasons.append("twin A differs from its archived run")
    if set(pairs) != set(PAIRS):
        reasons.append(f"the pairs {sorted(pairs)}, not {list(PAIRS)}")
    unread = [key for key in PAIRS if states.get(key, {}).get("state") == INCOMPLETE]
    if unread:
        reasons.append(f"the pairs {unread} are incomplete")
    if reference and reference.get("mode") not in (FLOAT64_FULL, FLOAT64_ACCUMULATION):
        reasons.append(f"the reference's mode {reference.get('mode')!r} is not the policy's")
    if reasons:
        return {**out, "state": INCOMPLETE, "reasons": reasons}
    return {**out, "state": WITHIN, "reasons": []}


EVIDENCE = (EXCEEDING, INSTABILITY)  # a case state that makes its rate failing


def case_state(
    attempts: Mapping[int, Mapping[str, Any] | None],
    case: NumericalCase,
    *,
    reader: RecordReader,
) -> dict[str, Any]:
    """A case over its attempts (``attempt_state``): **instability** or **exceeding** when any
    attempt is (evidence against the rate); then, without a first attempt, **incomplete**; with
    a rerun, the rerun's state, ``not_qualified`` (technical) when it is incomplete too; else
    the first attempt's (incomplete: its rerun is due). The reference's mode and fallback
    reason are the deciding attempt's."""
    states = {n: attempt_state(attempts.get(n), case, reader=reader) for n in ATTEMPTS}
    present = [n for n in ATTEMPTS if states[n]["present"]]
    first, rerun = states[ATTEMPTS[0]], states[ATTEMPTS[1]]
    evidence = [n for n in present if states[n]["state"] in EVIDENCE]
    deciding = present[-1] if present else None
    if evidence:
        deciding = evidence[0]
        verdict, technical = states[deciding]["state"], False
        if verdict == INSTABILITY:  # its reason starts with "instability:"
            reasons = list(states[deciding]["reasons"])
        else:
            reasons = [f"attempt {n}: {r}" for n in evidence for r in states[n]["reasons"]]
    elif not first["present"]:
        verdict, technical, deciding = INCOMPLETE, False, None
        reasons = [f"no first attempt: {'; '.join(first['reasons'])}"]
    elif rerun["present"]:
        verdict, technical = rerun["state"], rerun["state"] == INCOMPLETE
        reasons = ([f"not qualified (technical), incomplete after its rerun: attempt 1: "
                    f"{'; '.join(first['reasons'])}; attempt 2: {'; '.join(rerun['reasons'])}"]
                   if technical else [])  # fmt: skip
    else:
        verdict, technical = first["state"], False
        reasons = ([f"incomplete, its rerun not yet done: {'; '.join(first['reasons'])}"]
                   if verdict == INCOMPLETE else [])  # fmt: skip
    decided = states[deciding] if deciding is not None else {}
    return {"seed": case.seed, "arm": case.optimizer, "lr": case.lr, "state": verdict,
            "attempts": len(present), "not_qualified": technical,
            "attempt_states": [states[n]["state"] if states[n]["present"] else None
                               for n in ATTEMPTS],
            "pair_states": [states[n].get("pair_states") for n in ATTEMPTS],
            "reference_mode": decided.get("reference_mode"),
            "fallback_reason": decided.get("fallback_reason"), "reasons": reasons}  # fmt: skip


def control_case(scope: NumericalScope, rate: float, batch: int, upper: float) -> NumericalCase:
    """The case whose regular run a same-path control at ``rate`` uses."""
    return NumericalCase(scope.control_seed, scope.control_arm, float(rate), int(batch), upper)


def control_attempt_state(
    record: Mapping[str, Any] | None,
    *,
    rate: float,
    batch: int,
    upper: float,
    reader: RecordReader,
) -> dict[str, Any]:
    """One attempt of a same-path control. Only a record that counts (``RecordReader``) and is
    of this control is ``present``. Then **rejected** when its regular run was stopped by an
    instability rule (reason ``instability: …``), or it was measured exceeding (or diverged) or
    at the bound, D1's states (``control_state``) at 0.01·K*(U); **accepted** when its record is
    complete (its run conforms and it read its regular run's v₀ stream and training prefix) and
    K_scoring was measured at every step of the scope, every reading finite and strictly below
    the bound; **incomplete** otherwise."""
    scope = reader.scope
    why = reader.why_not(record, control_case(scope, rate, batch, upper))
    if why is not None:
        return {"state": INCOMPLETE, "present": False, "reasons": [why]}
    identity = {"seed": scope.control_seed, "batch": int(batch), "U": upper}
    if not _same(record, identity, "rate", rate):
        return {"state": INCOMPLETE, "present": False,
                "reasons": ["the record is of another control"]}  # fmt: skip
    if record.get("status") == INSTABILITY:
        return {"present": True, "state": REJECTED, "instability": True,
                "reasons": [str(record.get("reason"))]}  # fmt: skip
    measured = control_state(record, k_at=list(scope.eval_at), bound=scope.bound(upper))
    out = {"present": True, "measured": measured["state"],
           "max_K_scoring": measured["max_K_scoring"],
           "first_gradient_norm": record.get("first_gradient_norm")}  # fmt: skip
    if measured["state"] in (EXCEEDING, AT_BOUND):
        return {
            **out,
            "state": REJECTED,
            "reasons": [
                f"{measured['state']} (max K_scoring {_value(measured['max_K_scoring'])}, bound "
                f"{scope.bound(upper):.4g}" + (", diverged)" if measured["diverged"] else ")")
            ],
        }
    reasons = []
    if record.get("status") != "complete":
        reasons.append(str(record.get("reason") or "incomplete"))
    if measured["state"] != BELOW:
        steps = [r.get("step") for r in record.get("scoring") or []]
        reasons.append(f"K_scoring measured at {steps}, not at every step of "
                       f"{list(scope.eval_at)}")  # fmt: skip
    if reasons:
        return {**out, "state": INCOMPLETE, "reasons": reasons}
    return {**out, "state": ACCEPTED, "reasons": []}


def control_verdict(
    attempts: Mapping[int, Mapping[str, Any] | None],
    *,
    rate: float,
    batch: int,
    upper: float,
    reader: RecordReader,
) -> dict[str, Any]:
    """A same-path control over its attempts (``control_attempt_state``): **rejected** when any
    attempt is (exceeding, at the bound, or its run stopped by instability); then, without a
    first attempt, **incomplete**; with a rerun, **accepted** when it is accepted and
    **not_qualified** (technical) when it is incomplete too; else the first attempt's:
    **accepted**, or **incomplete** (its rerun is due)."""
    states = {n: control_attempt_state(attempts.get(n), rate=rate, batch=batch, upper=upper,
                                       reader=reader) for n in ATTEMPTS}  # fmt: skip
    present = [n for n in ATTEMPTS if states[n]["present"]]
    first, rerun = states[ATTEMPTS[0]], states[ATTEMPTS[1]]
    rejected = [n for n in present if states[n]["state"] == REJECTED]
    if rejected:
        verdict, reasons = REJECTED, []
        for n in rejected:  # an instability reason keeps its prefix
            reasons += (states[n]["reasons"] if states[n].get("instability")
                        else [f"attempt {n}: {r}" for r in states[n]["reasons"]])  # fmt: skip
    elif not first["present"]:
        verdict, reasons = INCOMPLETE, [f"no first attempt: {'; '.join(first['reasons'])}"]
    elif rerun["present"]:
        verdict = ACCEPTED if rerun["state"] == ACCEPTED else NOT_QUALIFIED
        reasons = (
            []
            if verdict == ACCEPTED
            else [
                f"not qualified (technical), incomplete after its rerun: attempt 1: "
                f"{'; '.join(first['reasons'])}; attempt 2: {'; '.join(rerun['reasons'])}"
            ]
        )
    else:
        verdict = first["state"]
        reasons = ([f"incomplete, its rerun not yet done: {'; '.join(first['reasons'])}"]
                   if verdict == INCOMPLETE else [])  # fmt: skip
    decided = states[present[-1]] if present else {}
    return {"verdict": verdict, "attempts": len(present), "reasons": reasons,
            "attempt_states": [states[n]["state"] if states[n]["present"] else None
                               for n in ATTEMPTS],
            **{key: decided.get(key) for key in ("measured", "max_K_scoring",
                                                 "first_gradient_norm")}}  # fmt: skip


def control_verdicts(
    records: Mapping[float, Mapping[int, Mapping[str, Any] | None]],
    *,
    rates: Sequence[float],
    batch: int,
    upper: float,
    reader: RecordReader,
) -> dict[float, str]:
    """Each FA rate's same-path control (``control_verdict``; ``records`` maps a rate to its
    attempts' records by number): accepted, rejected, not_qualified or incomplete."""
    return {lr: control_verdict(records.get(lr) or {}, rate=lr, batch=batch, upper=upper,
                                reader=reader)["verdict"] for lr in rates}  # fmt: skip


def retry_due(
    attempts: Mapping[str, Mapping[int, Mapping[str, Any] | None]],
    cases: Sequence[NumericalCase],
    *,
    reader: RecordReader,
) -> list[NumericalCase]:
    """The cases whose counted first attempt is incomplete (not exceeding, not instability),
    with no counted rerun yet."""
    due = []
    for case in cases:
        mine = attempts.get(case.name) or {}
        first, rerun = (attempt_state(mine.get(n), case, reader=reader) for n in ATTEMPTS)
        if first["present"] and first["state"] == INCOMPLETE and not rerun["present"]:
            due.append(case)
    return due


def control_retry_due(
    controls: Mapping[float, Mapping[int, Mapping[str, Any] | None]],
    rates: Sequence[float],
    *,
    batch: int,
    upper: float,
    reader: RecordReader,
) -> list[float]:
    """The control rates whose counted first attempt is incomplete, with no counted rerun yet."""
    due = []
    for lr in rates:
        mine = controls.get(lr) or {}
        first, rerun = (control_attempt_state(mine.get(n), rate=lr, batch=batch, upper=upper,
                                              reader=reader) for n in ATTEMPTS)  # fmt: skip
        if first["present"] and first["state"] == INCOMPLETE and not rerun["present"]:
            due.append(lr)
    return due


def qualification(
    records: Mapping[str, Mapping[int, Mapping[str, Any] | None]],
    cases: Sequence[NumericalCase],
    *,
    reader: RecordReader,
) -> dict[str, Any]:
    """Each selected (arm, rate) over its cases (``case_state``; ``records`` maps a case's name
    to its attempts' records by number): **failing** when a pair of any case is exceeding
    (evidence of amplification) or an instability rule stopped a case's regular run (reason
    ``instability: …``); else **not_qualified** (technical) when a case is incomplete after its
    rerun; else **incomplete** when a case is incomplete with its rerun not yet done; else
    **qualified**: every pair of every case within. Returns ``rates`` (by ``rate_key``),
    ``seeds`` (the seeds each was tested on), ``reasons`` for every rate not qualified, and
    ``cases``."""
    entries = {case.name: case_state(records.get(case.name) or {}, case, reader=reader)
               for case in cases}  # fmt: skip
    rates: dict[str, str] = {}
    seeds: dict[str, list[int]] = {}
    reasons: dict[str, list[str]] = {}
    for key in dict.fromkeys(case.key for case in cases):
        mine = [case for case in cases if case.key == key]
        states = [entries[case.name] for case in mine]
        if any(s["state"] in EVIDENCE for s in states):
            rates[key] = FAILING
        elif any(s["not_qualified"] for s in states):
            rates[key] = NOT_QUALIFIED
        elif any(s["state"] != WITHIN for s in states):
            rates[key] = INCOMPLETE
        else:
            rates[key] = QUALIFIED
        seeds[key] = sorted(case.seed for case in mine)
        found = []
        for case in mine:
            entry = entries[case.name]
            for reason in entry["reasons"]:  # an instability reason leads with its prefix
                found.append(
                    f"instability: case s{case.seed}: {reason.removeprefix('instability: ')}"
                    if entry["state"] == INSTABILITY
                    else f"case s{case.seed}: {reason}"
                )
        if found:
            reasons[key] = found
    return {"rates": rates, "seeds": seeds, "reasons": reasons,
            "cases": [entries[c.name] for c in cases]}  # fmt: skip


def load_numerical(
    out: Path, cases: Sequence[NumericalCase], control_rates: Sequence[float]
) -> tuple[dict[str, dict[int, dict | None]], dict[float, dict[int, dict | None]]]:
    """The attempts of every case (by name) and of every control (by rate), each by attempt
    number, as written under ``out``; None where nothing is. Nothing is moved."""

    def read(relative: str) -> dict | None:
        path = out / relative
        return json.loads(path.read_text()) if path.exists() else None

    attempts = {case.name: {n: read(case.attempt_path(n)) for n in ATTEMPTS} for case in cases}
    controls = {lr: {n: read(control_path(lr, n)) for n in ATTEMPTS} for lr in control_rates}
    return attempts, controls


def selection_keys(rates: Mapping[str, Sequence[float]]) -> list[str]:
    """The sorted ``<optimizer>|<lr:g>`` keys of a selection."""
    return sorted(rate_key(arm, float(lr)) for arm, values in rates.items() for lr in values)


def selection_digest(rates: Mapping[str, Sequence[float]]) -> str:
    """The first 12 hex digits of the sha256 of the selection's sorted keys, one per line
    (``selection_keys`` joined by newlines, UTF-8): its verdict's name under ``verdicts/``."""
    return hashlib.sha256("\n".join(selection_keys(rates)).encode()).hexdigest()[:12]


def verdict_path(out: Path, rates: Mapping[str, Sequence[float]]) -> Path:
    """``<out>/verdicts/<selection_digest>.json``: every selection keeps its own verdict."""
    return out / "verdicts" / f"{selection_digest(rates)}.json"


def write_verdict(out: Path, verdict: Mapping[str, Any], rates: Mapping[str, Sequence[float]]):
    """Write a selection's verdict to ``verdict_path`` and copy it to ``<out>/verdict.json``
    (the latest, for people to read). An earlier verdict of the same selection moves to
    ``verdicts/history/<digest>/``; a file at that path holding another selection is refused
    (never overwritten). Returns the path."""
    path = verdict_path(out, rates)
    if path.exists():
        earlier = json.loads(path.read_text())
        held = ((earlier.get("provenance") or {}).get("selection") or {}).get("keys")
        if held != selection_keys(rates):
            raise NumericalError(f"{path} holds the verdict of another selection ({held}); "
                                 "nothing is overwritten")  # fmt: skip
        stamp = str((earlier.get("provenance") or {}).get("written_at")).replace(":", "")
        history = out / "verdicts" / "history" / selection_digest(rates)
        history.mkdir(parents=True, exist_ok=True)
        os.replace(path, history / f"{stamp}-{len(list(history.iterdir()))}.json")
    write_json(path, verdict)
    write_json(out / "verdict.json", verdict)
    return path


def numerical_verdict(
    settings: DictConfig,
    out: Path,
    *,
    rates: Mapping[str, Sequence[float]],
    batch: int,
    upper: float,
    code: Mapping[str, Any] | None,
    root: Path,
    state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The verdict record of one selection, which the selection's replay reads
    (``write_verdict`` keeps one per selection): ``batch``, ``U``, ``arms`` (the selected arms:
    all three, or the primary two alone), ``bound``, ``commit`` (the reading checkout's);
    ``policy`` (``NumericalScope.policy``, with ``code``'s GPU identity of
    the regular runs and the records' writer commits: a qualification holds only for this
    batch and this policy); ``rates`` (``qualification``: qualified, failing, not_qualified or
    incomplete, by ``rate_key``) with the ``seeds`` each was tested on; ``controls``
    (``control_verdict``, by ``<lr:g>``: accepted, rejected, not_qualified or incomplete);
    ``qualified``, true iff every rate is qualified and every control accepted; ``reasons`` (by
    rate key: its cases' and, for the control arm, ``control: …``); ``cases``, each with its
    reference's mode and fallback reason; ``reference_fallback``, the cases whose deciding
    attempt's reference fell back to float64 loss and accumulation; and ``provenance``, with the
    selection, each writer commit's reuse check, the code check and the reruns due. Only
    records that count (``RecordReader``, on the runs under ``root``) are read, and none unless
    ``code`` (``code_check``) shows the regular runs' GPU code to be the check's, each rate then
    giving that reason."""
    scope = NumericalScope.of(settings)
    state = dict(git_state() if state is None else state)
    reader = RecordReader(scope, state, root)
    cases = numerical_cases(settings, rates, upper, batch=batch)
    control_rates = selected_rates(scope, rates)[scope.control_arm]
    attempts, controls = load_numerical(out, cases, control_rates)
    consistent = code is not None and bool(code.get("consistent"))
    if not consistent:  # no record counts against other GPU code
        attempts = {name: dict.fromkeys(ATTEMPTS) for name in attempts}
        controls = {lr: dict.fromkeys(ATTEMPTS) for lr in controls}
    q = qualification(attempts, cases, reader=reader)
    details = {lr: control_verdict(controls[lr], rate=lr, batch=batch, upper=upper,
                                   reader=reader) for lr in control_rates}  # fmt: skip
    reasons = {key: list(found) for key, found in q["reasons"].items()}
    if not consistent:
        problems = (code or {}).get("problems") or ["no code check"]
        for key in q["rates"]:
            reasons[key] = [f"code: the regular runs' GPU code is not the check's: {p}"
                            for p in problems] + reasons.get(key, [])  # fmt: skip
    for lr, detail in details.items():
        if detail["reasons"]:
            reasons.setdefault(rate_key(scope.control_arm, lr), []).extend(
                f"control: {reason}" for reason in detail["reasons"])  # fmt: skip
    verdicts = {f"{lr:g}": detail["verdict"] for lr, detail in details.items()}
    qualified = (all(v == QUALIFIED for v in q["rates"].values())
                 and all(v == ACCEPTED for v in verdicts.values()))  # fmt: skip
    fallback = [{"case": case.name, "rate": case.key, "seed": case.seed,
                 "fallback_reason": entry["fallback_reason"]}
                for case, entry in zip(cases, q["cases"], strict=True)
                if entry["reference_mode"] == FLOAT64_ACCUMULATION]  # fmt: skip
    seeds: dict[int, Any] = {}
    for case in cases:  # the setup facts of each seed, from its latest counted attempt
        for n in reversed(ATTEMPTS):
            record = attempts[case.name][n]
            if (record is not None and reader.why_not(record, case) is None
                    and record.get("seed_facts") and case.seed not in seeds):  # fmt: skip
                seeds[case.seed] = record["seed_facts"]
    due = {
        "cases": [case.name for case in retry_due(attempts, cases, reader=reader)],
        "controls": [f"{lr:g}" for lr in control_retry_due(
            controls, control_rates, batch=batch, upper=upper, reader=reader)],
    }  # fmt: skip
    gpu_code = (code or {}).get("gpu_code") if consistent else None
    return {
        "batch": int(batch), "U": int(upper) if float(upper) == int(upper) else upper,
        "arms": list(selected_rates(scope, rates)),
        "bound": scope.bound(upper), "commit": state.get("commit"),
        "policy": scope.policy(upper, gpu_code, reader.writers()),
        "rates": q["rates"], "seeds": q["seeds"], "controls": verdicts, "qualified": qualified,
        "reasons": reasons, "cases": q["cases"], "reference_fallback": fallback,
        "provenance": {
            "spec": NUMERICAL_SPEC, "code": state,
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": OmegaConf.to_container(settings.numerical, resolve=True),
            "selection": {"digest": selection_digest(rates), "keys": selection_keys(rates)},
            "selected_rates": selected_rates(scope, rates), "K_star": scope.k_star(upper),
            "writers": dict(sorted(reader.checks.items())),
            "controls": {f"{lr:g}": detail for lr, detail in details.items()},
            "seeds": {str(seed): facts for seed, facts in sorted(seeds.items())},
            "retry_due": due, "code_check": code,
        },
    }  # fmt: skip
