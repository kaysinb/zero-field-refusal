"""Study 2: the diagnostic of the FA soft-control failure, outside the confirmation.

The specification is docs/study2-fa-control-diagnostic.md, frozen before any run; its Revision 1
supersedes the base text where they differ, and its Revision 2 fixes the operationalizations
(the path criterion, the same-path control, the cases of Branch F, survival of fp32 rounding,
the AdamW and FA 1e-5 scales, D2's map, the float64 reference, "incomplete"). The settings are
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
- **Branches** (``decide_branch``): F, B, A or ambiguous, from the parts' outputs.

**Memory** (1.5B, fp32, 46 GB): a trainable-sized vector is ≈ 5.2 GB. The second model of D3
stays in host memory except while D_twin is measured; float64 gradients are summed as each
parameter's gradient arrives (``Float64Gradients``), so no fp32 gradient set coexists with them.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PreTrainedModel

from .batches import chunks, collate
from .curvature import _dot, _norm, _ritz, lanczos, start_vector
from .evaluation import NO_CAP, sequence_kl
from .losses import loss_denominator, response_logits, response_logprobs, soft_kl, summed_nll
from .losses import targets as target_ids
from .models import freeze
from .optim import ADAM_MODES, build_optimizer
from .panels import Monitor
from .paths import runs_root
from .runtime import Environment, ReplayedSoftTargets, StreamPrefix, load_stream_for
from .schedule import _norm as grad_norm
from .seeding import derive_seed, seed_everything
from .teacher import Sample
from .tracking import git_state, hardware, package_versions
from .train import _theta0_second_moment, accumulate_gradient

Vector = list[torch.Tensor]
Term = Callable[[], torch.Tensor]
QUANTILES = (0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
PERTURBATIONS = ("p1", "p2", "p3", "p4")
REFERENCE = "reference"
BRANCHES = ("F", "B", "A", "ambiguous", "incomplete")


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
             control: str | None) -> bool:  # fmt: skip
    return (
        cfg.get("seed") == seed
        and cfg.train.batch_size == batch
        and cfg.optimizer.name == optimizer
        and math.isclose(float(cfg.optimizer.lr), lr, rel_tol=1e-9)
        and cfg.get("control") == control
    )


def find_run(root: Path, experiment: str, **match: Any) -> Path:
    """The one finished run (a ``terminal.json``) under ``root/experiment`` whose archived config
    matches; anything else is an error naming what was found."""
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


def peak_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1e9, 3)}  # fmt: skip


def reset_peak(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def numerics(device: torch.device) -> dict[str, Any]:
    return {"device": str(device), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision()}  # fmt: skip


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
    names: Sequence[str], x: Sequence[torch.Tensor], v0: Sequence[torch.Tensor], eps: float
) -> dict[str, Any]:
    """The share of ‖x‖² per module kind, per layer group and on ε-dominated coordinates."""
    kinds: dict[str, float] = {}
    layers: dict[str, float] = {}
    eps_class = {"below_eps": 0.0, "below_10eps": 0.0}
    total = 0.0
    for name, xi, v in zip(names, x, v0, strict=True):
        xi = xi.to(v.device).double()
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
    """``optim.Adam``'s arithmetic in float64 on fp32 parameters: the first (and, adaptive, the
    second) moment in float64, the update formed in float64 and rounded once into the fp32
    parameter. Frozen mode reads v₀ as stored (fp32) and widens it exactly. The parameters
    stay fp32, since the forward is fp32: no float64 master copy."""

    def __init__(self, params: Sequence[torch.nn.Parameter], opt_cfg: DictConfig,
                 second_moment: Sequence[torch.Tensor] | None = None) -> None:  # fmt: skip
        self.params = list(params)
        self.mode = ADAM_MODES[opt_cfg.name]
        if self.mode == "prewarmed":
            raise DiagnosticError("The float64 reference covers frozen and adaptive Adam only")
        self.lr, self.eps = float(opt_cfg.lr), float(opt_cfg.eps)
        if self.mode == "frozen":
            self.beta1, self.beta2 = float(opt_cfg.beta1), None
            if second_moment is None:
                raise DiagnosticError("Frozen Adam needs v₀")
            self.v = [v.to(p.device) for v, p in zip(second_moment, self.params, strict=True)]
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
    T_k. ``start`` is consumed. y is accumulated in host memory with ``host``."""
    alphas, betas = list(result["alphas"]), list(result["betas"])
    k = len(alphas)
    _, vectors = _ritz(alphas, betas[: k - 1])
    s = vectors[:, -1]
    device = torch.device("cpu") if host else start[0].device
    y = [torch.zeros(t.shape, dtype=torch.float32, device=device) for t in start]
    v, previous = list(start), None
    start.clear()
    for j in range(k):
        for yi, vi in zip(y, v, strict=True):
            yi.add_(vi.to(device), alpha=float(s[j]))
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
    reading (the first response of each prompt, cached as the runs cached it) is taken."""
    out = copy.deepcopy(cfg)
    if "cache" in out.eval.monitor:
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
    ``survival_share`` of the intended one (Revision 2, 4)."""
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
    """Revision 2, 4: perturb θ₀; if the perturbation does not survive fp32 rounding, scale it by
    ``rounding_scale`` once, and keep that second attempt whatever it realizes. Returns every
    attempt's record; the parameters hold the last attempt."""
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
        """D_twin between the checkpoint (A_t) and the student (B_t)."""
        load_trainable(self.other, checkpoint, self.names)
        self.other.to(self.env.device)
        try:
            return twin_divergence(self.other, self.student, self.scoring, pad_id=self.env.pad_id,
                                   micro_batch_size=cfg.execution.micro_batch_size)  # fmt: skip
        finally:
            self.other.to("cpu")


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
) -> dict[str, Any]:
    """One twin: the regular run of ``cfg`` from the student's current parameters, step for
    step as ``schedule.run_scheduled`` trains (no evaluation, no stop rule), for the archived
    run's ``train.steps``. ``precision``: ``fp32`` is the run's own arithmetic; ``float64`` and
    ``float64_accumulation`` are the reference (loss and gradient accumulation in float64, and
    the optimizer arithmetic too in ``float64``). At the steps of ``eval_at`` the trainable
    parameters go to ``save_dir`` (twin A) and/or D_twin is measured against ``compare_dir``'s
    checkpoints; the monitor is read every ``eval.monitor.every`` steps if given."""
    env, params, device = twins.env, twins.params, twins.env.device
    steps, size = int(cfg.train.steps), int(cfg.train.batch_size)
    seed_everything(derive_seed(cfg.seed, "torch"))
    reset_peak(device)
    started = time.perf_counter()
    frozen = ADAM_MODES.get(cfg.optimizer.name, "adaptive") == "frozen"
    if precision == "float64":
        optimizer: Any = Float64Adam(params, cfg.optimizer, twins.v0 if frozen else None)
    else:
        optimizer = build_optimizer(cfg.optimizer, params,
                                    second_moment=twins.v0 if frozen else None)  # fmt: skip
    rows, readings, divergence = [], [], []
    stop, taken = None, 0
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
            stop = f"non-finite loss or gradient at step {step}"
            break
        if precision == "float64":
            optimizer.step(grads)
            del grads
        else:
            optimizer.step()
        taken = t = step + 1
        if not math.isfinite(grad_norm(params)):
            stop = f"non-finite weights after step {step}"
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
    out = {"steps": taken, "stop": stop, "rows": rows, "seconds": time.perf_counter() - started,
           **peak_memory(device)}  # fmt: skip
    if monitor is not None:
        out["monitor"] = readings
    if compare_dir is not None:
        out["D_twin"] = divergence
        out.update(_pair_summary(divergence, eval_at, stop))
    return out


def _pair_summary(
    divergence: Sequence[Mapping[str, Any]], eval_at: Sequence[int], stop: str | None
) -> dict[str, Any]:
    """The pair's largest D_twin; a twin that stopped before the last measurement (a non-finite
    loss, gradient or weight) counts as unbounded."""
    values = [d["D_twin"] for d in divergence]
    complete = len(values) == len(eval_at) and stop is None
    largest = max(values) if values else math.inf
    if not complete:
        largest = math.inf
    return {"max_D_twin": largest, "complete": complete}


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
    """D4's shared-state checks and D1 (module docstring); writes ``d1/result.json``."""
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
    checks["initial_optimizer_state"] = _initial_state(regular_cfg, control_cfg, params, v0)

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
    recorded_norm = norms["iii_control_recorded"]
    ratio = norms["i_training_path_fp32"] / recorded_norm if recorded_norm else math.inf
    residual["ratio_i_to_iii"] = ratio
    residual["ratio_ii_to_iii"] = norm_ii / recorded_norm if recorded_norm else math.inf
    residual["path_dominates"] = ratio <= float(d.d1.path_share_max)
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
    result = {
        "part": "d1", "shared_state": checks, "v0_census": census, "residual": residual,
        "preconditioned_first_step": shares, "same_path_control": control,
        "numerics": numerics(device), "seconds": time.perf_counter() - started,
        "memory": {"before_same_path_control": reset_peak_keep},
        "provenance": provenance(settings, runs),
    }  # fmt: skip
    write_json(out / "d1" / "result.json", result)
    return result


def _initial_state(
    regular_cfg: DictConfig, control_cfg: DictConfig, params: Sequence, v0: Sequence
) -> dict[str, Any]:
    """Both runs' optimizers as built before their first step: mode, hyperparameters (the rate
    aside), and the state, which holds only v₀ (m₀ and the step counter start at zeros and 0 in
    the first step, optim.Adam)."""
    out = {}
    for label, cfg in (("regular", regular_cfg), ("control", control_cfg)):
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
    """The FA soft control with same-path targets: θ₀'s training-path distributions (the frozen
    θ₀ copy, recomputed per micro-batch), the loss and the gradient accumulation in float64,
    the forward and backward in fp32, frozen Adam as the control runs it, for the run's steps.
    K on the scoring panel at ``k_at`` (uncached, against the θ₀ copy) and on the monitor panel
    every ``eval.monitor.every`` steps, with ‖δ‖. ``student`` must be at θ₀."""
    d = settings.diagnostic
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
    k_at = [int(t) for t in d.d1.same_path_control.k_at]
    every, size = int(cfg.eval.monitor.every), int(cfg.train.batch_size)
    rows, readings, scoring_k = [], [], []
    stop, taken = None, 0
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
            stop = f"non-finite loss or gradient at step {step}"
            break
        optimizer.step()
        taken = t = step + 1
        with torch.no_grad():
            displacement = math.sqrt(sum((p.double() - theta0[n].double()).pow(2).sum().item()
                                         for n, p in trainable))  # fmt: skip
        if not math.isfinite(displacement):
            stop = f"non-finite weights after step {step}"
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
    bound = float(d.bound)
    values = [r["K_scoring"] for r in scoring_k]
    complete = stop is None and len(values) == len(k_at)
    largest = max(values) if values else math.inf
    out = {"rate": float(cfg.optimizer.lr), "steps": taken, "stop": stop, "rows": rows,
           "monitor": readings, "scoring": scoring_k, "scoring_panel": identity,
           "max_K_scoring": largest if complete and math.isfinite(largest) else math.inf,
           "bound": bound, "complete": complete,
           "below_bound": complete and math.isfinite(largest) and largest < bound,
           "seconds": time.perf_counter() - started, **peak_memory(device)}  # fmt: skip
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
    spectra: dict[str, list[dict[str, Any]]] = {"soft": [], "nll": []}
    for kind_index, kind in enumerate(("soft", "nll")):
        for b in range(k):
            hvp = preconditioned_hvp(model, params, terms[kind][b], v0, eps)
            index = 1000 * (kind_index + 1) + b
            result = lanczos(hvp, start_vector(trainable, int(d.seed), index), steps=lanczos_steps,
                             tolerance=tolerance)  # fmt: skip
            y = ritz_vector(hvp, start_vector(trainable, int(d.seed), index), result)
            check = check_vector(hvp, y, device)
            in_theta = [yi * sqrt_preconditioner(v, eps).cpu() for yi, v in zip(y, v0, strict=True)]
            spectra[kind].append({
                "batch": b, **{key: result[key] for key in ("top", "bottom", "residual",
                                                           "ritz_converged", "steps",
                                                           "top_trace", "alphas", "betas")},
                "top_vector_check": check,
                "localization": vector_shares(names, y, v0, eps),
                "localization_in_theta": vector_shares(names, in_theta, v0, eps),
            })  # fmt: skip
            del y, in_theta
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
        "batches": k, "residual": {"norm": _norm(residual), **residual_stats},
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
    student, trainable = _model(env, cfg)
    other = env.new_model()
    other.requires_grad_(False)
    names, params = [n for n, _ in trainable], [p for _, p in trainable]
    theta0 = {name: p.detach().cpu().clone() for name, p in trainable}
    v0, stream = second_moment(cfg, env, student, params)
    v0_hash = tensors_sha256(list(zip(names, v0, strict=True)))
    v0 = [v.cpu() for v in v0]
    prefix = StreamPrefix(cfg, env, int(cfg.seed))
    residual, stats, _ = control_residual(control_cfg, env, student, params, prefix)
    residual = [r.cpu() for r in residual]
    gc.collect()
    scoring, identity = scoring_samples(cfg, env)
    monitor = Monitor(operational_monitor(cfg), env, Recorder(), other)
    other.to("cpu")
    twins = Twins(env=env, student=student, other=other, trainable=trainable, theta0=theta0,
                  v0=v0, prefix=prefix, scoring=scoring)  # fmt: skip
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
    writes ``d3/determinism.json`` and ``d3/<case>.json``. Raises ``DeterminismFailure`` (after
    writing its record) when the re-run differs from the archive."""
    d = settings.diagnostic
    all_cases = cases_of(settings)
    eval_at = [int(t) for t in d.d3.eval_at]
    directory = out / "d3"
    record_path = directory / "determinism.json"
    done = {}
    with _twins(settings, env, runs) as (twins, shared):
        if record_path.exists():
            record = json.loads(record_path.read_text())
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
            path = directory / f"{name}.json"
            if path.exists():
                done[name] = json.loads(path.read_text())
                continue
            done[name] = run_case(settings, all_cases[name], twins, shared, runs, directory,
                                  eval_at=eval_at, keep_checkpoints=keep_checkpoints)  # fmt: skip
            write_json(path, done[name])
    return done


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
    twin A by D_twin at ``eval_at``."""
    d = settings.diagnostic
    bound = float(d.bound)
    env = twins.env
    run_dir = runs.regular[case.name]
    cfg = run_config(run_dir)
    if not math.isclose(float(cfg.optimizer.lr), case.lr, rel_tol=1e-9):
        raise DiagnosticError(f"{run_dir} ran at {cfg.optimizer.lr}, the case names {case.lr}")
    started = time.perf_counter()
    checkpoints = directory / "checkpoints" / case.name
    twins.reset()
    twin_a = run_twin(cfg, twins, eval_at=eval_at, monitor=shared["monitor"], save_dir=checkpoints)
    twin_a["archive"] = compare_archive(twin_a["rows"], twin_a["monitor"], run_dir)
    if twin_a["stop"] is not None:
        raise DiagnosticError(f"Twin A of {case.name} stopped: {twin_a['stop']}")

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
        norms[key] = math.sqrt(sum(directions[key](n, p.shape).double().pow(2).sum().item()
                                   for n, p in twins.trainable))  # fmt: skip
    pairs: dict[str, Any] = {}
    for key in PERTURBATIONS:
        twins.reset()
        rounding = apply_perturbation(
            twins.trainable, twins.theta0, directions[key], delta_norm / norms[key],
            survival_share=float(d.d3.survival_share),
            rounding_scale=float(d.d3.rounding_scale),
        )  # fmt: skip
        twin = run_twin(cfg, twins, eval_at=eval_at, compare_dir=checkpoints)
        twin["rounding"] = rounding
        twin["within_bound"] = twin["max_D_twin"] <= bound
        pairs[key] = twin
    twins.reset()
    pairs[REFERENCE] = reference_twin(cfg, twins, eval_at, checkpoints)
    pairs[REFERENCE]["within_bound"] = pairs[REFERENCE]["max_D_twin"] <= bound
    if not keep_checkpoints:
        for path in checkpoints.glob("A-step-*.pt"):
            path.unlink()
    twins.reset()
    largest = max(p["max_D_twin"] for p in pairs.values())
    return {"part": "d3", "case": case.name, "optimizer": case.optimizer, "lr": case.lr,
            "control_lr": case.control_lr, "regular_run": str(run_dir), "bound": bound,
            "eval_at": list(eval_at), "first_step_displacement_norm": delta_norm,
            "first_step_update_norm": intended,
            "first_step_changed": sum(int((t != 0).sum()) for t in delta),
            "twin_A": twin_a, "pairs": pairs, "max_D_twin": largest,
            "within_bound": all(p["within_bound"] for p in pairs.values()),
            "shared": {k: v for k, v in shared.items() if k not in ("residual", "monitor")},
            "numerics": numerics(env.device), "seconds": time.perf_counter() - started,
            "provenance": provenance(settings, runs)}  # fmt: skip


def reference_twin(
    cfg: DictConfig, twins: Twins, eval_at: Sequence[int], checkpoints: Path
) -> dict[str, Any]:
    """The float64 reference (Revision 1): float64 optimizer arithmetic if memory allows, else
    float64 loss and accumulation only, with the reason recorded."""
    try:
        out = run_twin(cfg, twins, eval_at=eval_at, precision="float64", compare_dir=checkpoints)
        out["mode"] = "float64 loss, accumulation and optimizer arithmetic"
        return out
    except torch.OutOfMemoryError as exc:
        reason = f"out of memory: {str(exc).splitlines()[0] if str(exc) else ''}"
    gc.collect()
    if twins.env.device.type == "cuda":
        torch.cuda.empty_cache()
    twins.reset()
    out = run_twin(cfg, twins, eval_at=eval_at, precision="float64_accumulation",
                   compare_dir=checkpoints)  # fmt: skip
    out["mode"] = "float64 loss and accumulation only (fp32 optimizer state)"
    out["fallback_reason"] = reason
    return out


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
) -> dict[str, Any]:
    """The frozen decision branches (docs/study2-fa-control-diagnostic.md, "Decision
    branches" and Revision 2), read from the diagnostic's outputs:

    - **A** when any D3 pair at the failing case (FA 1.21e-5) exceeds the bound (a pair that
      stopped on a non-finite value counts as exceeding);
    - **F** when D1 shows that the residual comes from the path discrepancy (``path_dominates``:
      the fp32 same-path residual (i) is at most ``path_share_max`` of the recorded one (iii)),
      the same-path float64 soft control keeps K_scoring strictly below the bound at every
      measured step, and every pair of every case in ``branches.regular_cases`` (every D3 case,
      AdamW 1.32e-5 included) stays within the bound;
    - **B** when the same-path control exceeds the bound but every pair at the failing case
      stays within it;
    - **ambiguous** otherwise (an AdamW pair above the bound with F's other conditions met,
      say): the expansion case (FA 1.46e-5) is added to D3 once (``expand``), then the outcome
      is reported without choosing.

    The branch is decided on the three prescribed cases only; the expansion never changes it.
    The AdamW pairs are reported under every branch (``facts``). Missing inputs (D3 stopped by
    its determinism check, say) give ``incomplete``."""
    d = settings.diagnostic
    bound = float(d.bound)
    failing = str(d.branches.failing)
    regular_cases = [str(c) for c in d.branches.regular_cases]
    prescribed = [str(c) for c in d.d3.order]
    expansion = str(d.d3.expansion)
    missing = []
    if d1 is None:
        missing.append("d1/result.json")
    if determinism is None:
        missing.append("d3/determinism.json")
    elif not determinism.get("equal"):
        missing.append("D3 (stopped: the determinism check failed)")
    missing += [f"d3/{name}.json" for name in prescribed if name not in cases]
    facts: dict[str, Any] = {"bound": bound, "failing_case": failing}

    def pairs_within(name: str) -> dict[str, bool]:
        return {key: bool(p["max_D_twin"] <= bound) for key, p in cases[name]["pairs"].items()}

    out: dict[str, Any] = {"facts": facts, "missing": missing}
    # Every prescribed case's pairs, AdamW's included, reported under every branch
    facts["pairs_within"] = {name: pairs_within(name) for name in prescribed if name in cases}
    if failing in cases:
        facts["failing_pairs_within"] = pairs_within(failing)
        if not all(facts["failing_pairs_within"].values()):
            return {**out, "branch": "A", "expand": None}
    if missing:
        return {**out, "branch": "incomplete", "expand": None}
    facts["path_dominates"] = bool(d1["residual"]["path_dominates"])
    facts["same_path_control_below"] = bool(d1["same_path_control"]["below_bound"])
    facts["regular_cases_within"] = {name: all(pairs_within(name).values())
                                     for name in regular_cases}  # fmt: skip
    failing_ok = all(facts["failing_pairs_within"].values())
    if (facts["path_dominates"] and facts["same_path_control_below"]
            and all(facts["regular_cases_within"].values())):  # fmt: skip
        branch = "F"
    elif not facts["same_path_control_below"] and failing_ok:
        branch = "B"
    else:
        branch = "ambiguous"
    expand = expansion if branch == "ambiguous" and expansion not in cases else None
    if branch == "ambiguous" and expansion in cases:
        facts["expansion"] = {"case": expansion, "pairs_within": pairs_within(expansion),
                              "note": "reported without choosing"}  # fmt: skip
    return {**out, "branch": branch, "expand": expand}
