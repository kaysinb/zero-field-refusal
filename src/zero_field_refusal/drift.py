"""Study 2, amendment 4, section 4: the first-step drift diagnostic, descriptive, outside the
confirmation. It enters no decision.

**Setup.** θ₀ of the study model at the selected batch B, and n = 64 independent batches of a
dedicated stream: seed 2300's appendable training stream, batch i holding its responses
iB … (i+1)B − 1, in two fixed halves (batches 0–31 and 32–63). g_i is the gradient of the training
NLL with the training normalization (``train.accumulate_gradient`` under the regular runs' config
at B: the same loss and denominator), at θ₀. v₀ is the seed's own preconditioner stream, 50
batches at B, estimated as the FA arm estimates it (``train._theta0_second_moment``).

**First updates** (``first_update``): each g_i is transformed by the first update of four
optimizers, the direction and scale without the learning rate (ε = 1e-8, β₂ = 0.999):

- ``sgd``: u = g;
- ``adamw``: u = m̂₁ / (√v̂₁ + ε) with m̂₁ = g and v̂₁ = g², Adam's bias-corrected first step
  (approximately sign-normalized);
- ``fa``: u = g / (√v₀ + ε), frozen Adam;
- ``prewarmed``: u = g / (√(β₂·v₀ + (1 − β₂)·g²) + ε), pre-warmed Adam (not bias-corrected).

Each is computed with ``optim.Adam``'s own operations in its mode, so it is that optimizer's first
step at lr = 1 and β₁ = 0 (m̂₁ = g), bit for bit (tests/test_drift.py).

**Statistics** (``drift_statistics``), per optimizer, with S = Σ u_i, Q = Σ ‖u_i‖² and n = 64:
``mean_norm2`` = (‖S‖² − Q) / (n(n − 1)), the unbiased estimate of ‖E u‖²; ``trace_cov`` =
(Q − ‖S‖²/n) / (n − 1), the unbiased estimate of tr Cov(u); their ``ratio``; and ``half_cosine``,
the cosine between the sums S_a and S_b of the two halves. Negative estimates are kept as they
are: no null distribution, sign flip or p-value. Across optimizers, ``cross_cosines``: the cosine
of the sums S of every pair.

**Precision and memory.** Each u_i is formed in fp32, per parameter, from the gradient the
backward pass left in ``.grad``; it is added to its half's sum (fp32, on the device) and freed
with the gradient, parameter by parameter. Every norm, dot product and Q is reduced in float64,
per parameter in pieces of ``REDUCTION_CHUNK`` coordinates. One optimizer is processed at a time,
with the 64 gradients recomputed for each (4 × 64 backward passes; ``gradients_identical``
records whether the four passes saw the same losses and gradient norms, and
``gradient_differences`` the largest relative difference of each and where, so that a
rounding-level difference is told from a real one). A non-finite loss or ‖g_i‖² stops the run
(``DriftError``): it is a failure, not a statistic. After its half statistics, only its full sum
S = S_a + S_b (fp32) goes to host memory, for the cross cosines, which are reduced there:
≈ 5.24 GB per optimizer at 1.5B, 21 GB for four. v₀ stays in pinned host memory, ≈ 6.7 GB at 1.5B
rather than 5.24 (PyTorch's pinned allocator rounds each block up to a power of two; one flat
buffer would round to 8.6 GB), and is copied to the device one parameter at a time, so no
parameter-sized tensor persists on the device beyond the two half-sums and the gradient.

**Blind to outcomes.** Only norms, dot products, losses and v₀'s hash leave this module. It reads
the teacher prompts' training stream and v₀'s stream alone: no harmful prompt set, stream or
generation, no judge, no refusal (tests/test_drift.py). The script wraps the environment so that
a prompt set named "harmful" raises.

**Streams** (``prepare_streams``): the drift stream is the seed's appendable training stream,
grown by ``runtime.extend_stream`` to 64·B responses, and v₀'s stream is its preconditioner
stream, made by ``runtime.generate_stream``: exactly as the regular runs' streams are made
(scripts/generate_teacher.py). A stream that exists is located, never regenerated.

The settings are configs/experiments/drift_diagnostic.yaml (``DriftScope``); the script is
scripts/drift_diagnostic.py; the result goes to ``RUNS_DIR/diagnostics/drift/b{B}/result.json``.
"""

from __future__ import annotations

import gc
import math
import resource
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel

from .code_identity import gpu_code_identity
from .fa_control_diagnostic import numerics, peak_memory, reset_peak, second_moment, tensors_sha256
from .models import freeze
from .paths import CONFIG_ROOT, runs_root
from .protocol import load_run_config
from .runtime import Environment, StreamPrefix, extend_stream, generate_stream, locate_stream
from .seeding import derive_seed, seed_everything
from .teacher import MANIFEST_FILE, Sample, published_chunks, stream_layout
from .tracking import git_state, hardware, package_versions
from .train import accumulate_gradient

SPEC = "study 2, amendment 4, section 4 (first-step drift against diffusion)"
OPTIMIZERS = ("sgd", "adamw", "fa", "prewarmed")
USES_V0 = ("fa", "prewarmed")
REDUCTION_CHUNK = 1 << 20  # coordinates per piece of a float64 reduction
RESULT = "result.json"
Vector = Sequence[torch.Tensor]  # a parameter-sized vector, one tensor per trainable parameter


class DriftError(ValueError):
    """The settings, the run config or a gradient departs from the diagnostic's definitions."""


@dataclass(frozen=True)
class DriftScope:
    """The frozen settings (configs/experiments/drift_diagnostic.yaml, ``drift``)."""

    seed: int
    batches: int
    optimizers: tuple[str, ...]
    eps: float
    beta2: float
    v0_batches: int
    experiment: str
    overrides: tuple[str, ...]
    memory_rule_gb: float
    host_sums_gb: float
    host_factor: float
    host_margin_gb: float

    @classmethod
    def of(cls, settings: DictConfig) -> DriftScope:
        d = settings.drift
        batches = int(d.batches)
        if batches < 4 or batches % 2:
            raise DriftError(f"drift.batches must be even and at least 4, got {batches}")
        half = batches // 2
        halves = OmegaConf.to_container(d.halves)
        if halves != [[0, half - 1], [half, batches - 1]]:
            raise DriftError(f"drift.halves must be [[0, {half - 1}], [{half}, {batches - 1}]], "
                             f"the two halves of the batches in order, got {halves}")  # fmt: skip
        optimizers = tuple(str(name) for name in d.optimizers)
        if (
            not optimizers
            or len(set(optimizers)) != len(optimizers)
            or set(optimizers) - set(OPTIMIZERS)
        ):
            raise DriftError(f"drift.optimizers: distinct names among {OPTIMIZERS}, got "
                             f"{list(optimizers)}")  # fmt: skip
        if not (float(d.eps) > 0 and 0 < float(d.beta2) < 1):
            raise DriftError(f"drift.eps must be positive and drift.beta2 in (0, 1), got "
                             f"{d.eps} and {d.beta2}")  # fmt: skip
        host = d.host_memory
        return cls(
            seed=int(d.seed), batches=batches, optimizers=optimizers, eps=float(d.eps),
            beta2=float(d.beta2), v0_batches=int(d.v0_batches), experiment=str(d.run.experiment),
            overrides=tuple(str(o) for o in d.run.overrides),
            memory_rule_gb=float(d.memory_rule_gb), host_sums_gb=float(host.sums_gb),
            host_factor=float(host.factor), host_margin_gb=float(host.margin_gb),
        )  # fmt: skip

    @property
    def half(self) -> int:
        return self.batches // 2

    @property
    def host_required_gb(self) -> float:
        """The free host memory a run needs: factor × the four sums + the margin (47.5 GB)."""
        return self.host_factor * self.host_sums_gb + self.host_margin_gb


def output_root(batch: int, root: Path | None = None) -> Path:
    return (runs_root() if root is None else root) / "diagnostics" / "drift" / f"b{batch}"


def run_config(scope: DriftScope, batch: int) -> DictConfig:
    """The gradients' config: the regular runs' experiment at B and the drift seed, with the
    optimizer whose preconditioner settings locate v₀'s stream (50 batches at B)."""
    if batch < 1:
        raise DriftError(f"B must be positive, got {batch}")
    cfg = load_run_config(scope.experiment, [*scope.overrides, f"train.batch_size={batch}",
                                             f"seed={scope.seed}"])  # fmt: skip
    check_run_config(scope, cfg)
    return cfg


# The Adam arms whose β₂ the drift's ``beta2`` restates (FA has none), by optimizer config
BETA2_ARMS = ("adamw", "adamw_prewarmed")


def optimizer_settings(name: str) -> DictConfig:
    """The tracked optimizer config configs/optimizer/<name>.yaml, as written (unresolved: its
    preconditioner interpolates the run's batch)."""
    return OmegaConf.load(CONFIG_ROOT / "optimizer" / f"{name}.yaml").optimizer


def check_run_config(scope: DriftScope, cfg: DictConfig) -> None:
    """The config computes what the definitions say: the drift seed, an appendable training
    stream, and v₀ from ``v0_batches`` batches at the run's B; and the settings' ε and β₂ are the
    optimizers' (review M5): ε that of the run config's optimizer and of the tracked AdamW and
    pre-warmed AdamW configs, β₂ the second of the latter two's ``betas``."""
    problems = []
    if int(cfg.seed) != scope.seed:
        problems.append(f"seed {cfg.seed}, not {scope.seed}")
    if float(cfg.optimizer.eps) != scope.eps:
        problems.append(f"drift.eps {scope.eps:g}, but the run's optimizer "
                        f"({cfg.optimizer.name}) has ε {cfg.optimizer.eps}")  # fmt: skip
    for name in BETA2_ARMS:
        tracked = optimizer_settings(name)
        if float(tracked.eps) != scope.eps:
            problems.append(f"drift.eps {scope.eps:g}, but configs/optimizer/{name}.yaml has ε "
                            f"{tracked.eps}")  # fmt: skip
        if float(tracked.betas[1]) != scope.beta2:
            problems.append(f"drift.beta2 {scope.beta2:g}, but configs/optimizer/{name}.yaml has "
                            f"β₂ {tracked.betas[1]}")  # fmt: skip
    if stream_layout(cfg) != "appendable":
        problems.append("the training stream must be appendable (teacher.stream)")
    pre = OmegaConf.select(cfg, "optimizer.preconditioner", default=None)
    if pre is None:
        problems.append("no optimizer.preconditioner: v₀'s stream is located by it "
                        "(optimizer=adam_frozen)")  # fmt: skip
    elif (int(pre.batches), int(pre.batch_size)) != (scope.v0_batches, int(cfg.train.batch_size)):
        problems.append(f"v₀ from {pre.batches} batches of {pre.batch_size}, not "
                        f"{scope.v0_batches} of B {cfg.train.batch_size}")  # fmt: skip
    if problems:
        raise DriftError("The drift diagnostic's run config: " + "; ".join(problems))


def config_summary(scope: DriftScope, cfg: DictConfig) -> dict[str, Any]:
    """What the gradients' config fixes, for the provenance."""
    return {
        "experiment": scope.experiment,
        "overrides": [*scope.overrides, f"train.batch_size={cfg.train.batch_size}",
                      f"seed={cfg.seed}"],
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "loss": OmegaConf.to_container(cfg.train.loss, resolve=True),
        "max_response_tokens": int(cfg.train.max_response_tokens),
        "freeze": list(cfg.train.freeze),
        "micro_batch_size": int(cfg.execution.micro_batch_size),
        "preconditioner": OmegaConf.to_container(cfg.optimizer.preconditioner, resolve=True),
    }  # fmt: skip


# ----------------------------------------------------------------------------------- streams


def stream_state(cfg: DictConfig, env: Environment, responses: int) -> dict[str, Any]:
    """Where the drift stream (the seed's appendable training stream, its first ``responses``)
    and v₀'s stream are, located as every run locates them, and whether they are there."""
    seed = int(cfg.seed)
    spec, path, _ = locate_stream(cfg, env, "training", seed)
    need, have = -(-responses // int(spec["chunk_size"])), published_chunks(path)
    _, pre_path, _ = locate_stream(cfg, env, "preconditioner", seed)
    return {
        "drift": {"path": str(path), "responses": responses, "chunks": have, "needed": need,
                  "ready": have >= need},
        "preconditioner": {"path": str(pre_path),
                           "responses": int(cfg.optimizer.preconditioner.responses),
                           "ready": (pre_path / MANIFEST_FILE).is_file()},
    }  # fmt: skip


def prepare_streams(cfg: DictConfig, env: Environment, responses: int) -> dict[str, Any]:
    """Generate the streams that are missing, as scripts/generate_teacher.py does: the training
    stream extended by chunks (``runtime.extend_stream``), the preconditioner stream sampled once
    (``runtime.generate_stream``), one θ₀ model for both, at the config's generation batch size.
    Streams that exist are located and kept. Returns their state and what was generated."""
    state = stream_state(cfg, env, responses)
    generated: list[str] = []
    if state["drift"]["ready"] and state["preconditioner"]["ready"]:
        return {**state, "generated": generated}
    seed, size = int(cfg.seed), int(cfg.execution.generation_batch_size)
    model = env.new_model()
    if not state["drift"]["ready"]:
        generated += [str(p) for p in extend_stream(cfg, env, seed, responses, model=model,
                                                    batch_size=size)]  # fmt: skip
    if not state["preconditioner"]["ready"]:
        generated.append(str(generate_stream(cfg, env, "preconditioner", seed, model=model,
                                             batch_size=size)))  # fmt: skip
    del model
    gc.collect()
    return {**stream_state(cfg, env, responses), "generated": generated}


# ---------------------------------------------------------------------------- first updates


def first_update(
    name: str, g: torch.Tensor, v0: torch.Tensor | None = None, *, eps: float, beta2: float
) -> torch.Tensor:
    """The first update u of optimizer ``name`` for the gradient ``g`` (module docstring): the
    step ``optim.Adam`` takes at lr = 1 and β₁ = 0, with its operations in its mode, negated; for
    ``sgd`` the gradient (a copy). ``v0`` (``fa``, ``prewarmed``) has ``g``'s shape and device.
    Nothing is written to ``g`` or ``v0``."""
    if name == "sgd":
        return g.clone()
    if name == "adamw":  # adaptive Adam's state from zero: v₁ = (1 − β₂)g², bias-corrected
        v = torch.zeros_like(g).addcmul_(g, g, value=1 - beta2)
        denominator = (v.sqrt() / (1 - beta2) ** 0.5).add_(eps)
    elif name in USES_V0:
        if v0 is None:
            raise DriftError(f"The {name} update needs v₀")
        # frozen: v₀ itself; pre-warmed: v₁ = β₂v₀ + (1 − β₂)g², not bias-corrected
        v = v0 if name == "fa" else v0.mul(beta2).addcmul_(g, g, value=1 - beta2)
        denominator = v.sqrt().add_(eps)
    else:
        raise DriftError(f"Unknown optimizer {name!r}; known: {OPTIMIZERS}")
    return g / denominator


def first_updates(
    g: torch.Tensor, v0: torch.Tensor, eps: float, beta2: float
) -> dict[str, torch.Tensor]:
    """The first update of each optimizer (``first_update``) for one gradient."""
    return {name: first_update(name, g, v0, eps=eps, beta2=beta2) for name in OPTIMIZERS}


# ------------------------------------------------------------------------------- statistics


def squared_norm(x: torch.Tensor, chunk: int = REDUCTION_CHUNK) -> torch.Tensor:
    """‖x‖² reduced in float64 in pieces of ``chunk`` coordinates: a 0-dim float64 tensor on
    ``x``'s device (no synchronization)."""
    flat = x.reshape(-1)
    total = torch.zeros((), dtype=torch.float64, device=x.device)
    for start in range(0, flat.numel(), chunk):
        piece = flat[start : start + chunk].to(torch.float64)
        total += torch.dot(piece, piece)
        del piece
    return total


def gram(vectors: Sequence[Vector], chunk: int = REDUCTION_CHUNK) -> list[list[float]]:
    """The Gram matrix of parameter-sized vectors (each a sequence of same-shaped tensors, one
    per parameter, on one device), reduced in float64 per parameter in pieces of ``chunk``
    coordinates."""
    m = len(vectors)
    device = vectors[0][0].device if m and len(vectors[0]) else torch.device("cpu")
    total = torch.zeros(m, m, dtype=torch.float64, device=device)
    for parts in zip(*vectors, strict=True):
        flats = [t.reshape(-1) for t in parts]
        numel = flats[0].numel()
        if any(f.numel() != numel for f in flats):
            raise DriftError("The vectors' parameters differ in size")
        for start in range(0, numel, chunk):
            x = torch.stack([f[start : start + chunk].to(torch.float64) for f in flats])
            total += x @ x.T
            del x
    return total.tolist()


def estimates(sum_norm2: float, q: float, n: int) -> dict[str, float | None]:
    """The unbiased estimates of ‖E u‖² and tr Cov(u) from ‖S‖², Q and n, and their ratio
    (None when tr Cov is estimated as 0). Negative estimates are kept as they are."""
    mean_norm2 = (sum_norm2 - q) / (n * (n - 1))
    trace_cov = (q - sum_norm2 / n) / (n - 1)
    return {"mean_norm2": mean_norm2, "trace_cov": trace_cov,
            "ratio": mean_norm2 / trace_cov if trace_cov != 0 else None}  # fmt: skip


def drift_statistics(
    S_half: tuple[Vector, Vector],
    Q_half: tuple[float, float],
    n_half: int,
    chunk: int = REDUCTION_CHUNK,
) -> dict[str, Any]:
    """The statistics of one optimizer from the sums S_a, S_b and the Q_a, Q_b of its two halves
    of ``n_half`` batches each (n = 2·n_half): ``mean_norm2``, ``trace_cov``, ``ratio`` and
    ``half_cosine`` (None when a half-sum is 0), with ‖S‖² = ‖S_a‖² + 2 S_a·S_b + ‖S_b‖² and
    Q = Q_a + Q_b, all in float64."""
    (aa, ab), (_, bb) = gram(S_half, chunk)
    n, q = 2 * n_half, float(Q_half[0]) + float(Q_half[1])
    sum_norm2 = aa + 2 * ab + bb
    return {
        "n": n,
        **estimates(sum_norm2, q, n),
        "half_cosine": ab / math.sqrt(aa * bb) if aa > 0 and bb > 0 else None,
        "sum_norm2": sum_norm2,
        "Q": q,
        "halves": [{"sum_norm2": aa, "Q": float(Q_half[0])},
                   {"sum_norm2": bb, "Q": float(Q_half[1])}],
    }  # fmt: skip


def cross_cosines(sums: Mapping[str, Vector], chunk: int = REDUCTION_CHUNK) -> dict[str, Any]:
    """cos(S_x, S_y) for every pair of optimizers, in the order given ("x|y"; None when a sum is
    0), and each ‖S‖², from one float64 Gram matrix of the sums."""
    names = list(sums)
    g = gram([sums[name] for name in names], chunk)
    cosines = {}
    for i, x in enumerate(names):
        for j in range(i + 1, len(names)):
            norms = g[i][i] * g[j][j]
            cosines[f"{x}|{names[j]}"] = g[i][j] / math.sqrt(norms) if norms > 0 else None
    return {"cosines": cosines, "sum_norm2": {name: g[i][i] for i, name in enumerate(names)}}


# -------------------------------------------------------------------------------- the run


def _host_copy(t: torch.Tensor) -> torch.Tensor:
    """A copy in host memory, pinned when ``t`` is on a GPU (fast copies back, one parameter at
    a time)."""
    host = torch.empty(t.shape, dtype=t.dtype, pin_memory=t.is_cuda)
    return host.copy_(t)


def optimizer_pass(
    name: str,
    model: PreTrainedModel,
    params: Sequence[torch.nn.Parameter],
    batches: Sequence[Sequence[Sample]],
    cfg: DictConfig,
    env: Environment,
    scope: DriftScope,
    v0: Sequence[torch.Tensor] | None,
    chunk: int = REDUCTION_CHUNK,
) -> tuple[dict[str, Any], list[torch.Tensor], list[dict[str, Any]]]:
    """One optimizer's statistics (``drift_statistics``), its full sum S in host memory (fp32),
    and per batch the loss and ‖g_i‖² (float64). Each batch's gradient is computed anew at θ₀;
    each u_i is formed per parameter, added to its half's sum and freed with the gradient."""
    device = params[0].device
    sums = [[torch.zeros_like(p) for p in params] for _ in range(2)]  # S_a, S_b: fp32
    q = torch.zeros(2, dtype=torch.float64, device=device)
    rows = []
    for i, batch in enumerate(batches):
        half = i // scope.half
        for p in params:
            p.grad = None
        stats = accumulate_gradient(model, batch, cfg, env)
        g2 = torch.zeros((), dtype=torch.float64, device=device)
        with torch.no_grad():
            for k, p in enumerate(params):
                g = p.grad
                if g is None or g.dtype != torch.float32 or not g.is_contiguous():
                    raise DriftError(f"Parameter {k} has no contiguous fp32 gradient")
                start = None if v0 is None else v0[k].to(device, non_blocking=True)
                u = first_update(name, g, start, eps=scope.eps, beta2=scope.beta2)
                sums[half][k].add_(u)
                q[half] += squared_norm(u, chunk)
                g2 += squared_norm(g, chunk)
                del u, start, g
                p.grad = None  # the gradient is freed parameter by parameter
        row = {"batch": i, "half": half, "loss": stats["loss"], "grad_norm2": g2.item()}
        if not (math.isfinite(row["loss"]) and math.isfinite(row["grad_norm2"])):
            raise DriftError(f"{name}, batch {i}: the loss ({row['loss']}) or ‖g‖² "
                             f"({row['grad_norm2']}) is not finite")  # fmt: skip
        rows.append(row)
    q_a, q_b = q.tolist()
    statistics = drift_statistics((sums[0], sums[1]), (q_a, q_b), scope.half, chunk)
    total = []
    for k in range(len(params)):  # S = S_a + S_b in fp32, to host memory parameter by parameter
        sums[0][k].add_(sums[1][k])
        sums[1][k] = None
        total.append(sums[0][k].cpu())
        sums[0][k] = None
    return statistics, total, rows


def gradient_agreement(rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Whether every pass saw the first pass's losses and ‖g_i‖² (``identical``), and for each
    the largest relative difference from the first pass, with the batch and optimizer where it
    occurs (``at``: None when all are equal)."""
    names = list(rows)
    first = rows[names[0]]
    out: dict[str, Any] = {"identical": all(rows[name] == first for name in names)}
    for key in ("loss", "grad_norm2"):
        largest, at = 0.0, None
        for name in names[1:]:
            for a, b in zip(first, rows[name], strict=True):
                if a[key] == b[key]:
                    continue
                scale = max(abs(a[key]), abs(b[key]))
                difference = abs(a[key] - b[key]) / scale
                if at is None or difference > largest:
                    largest, at = difference, {"batch": a["batch"], "optimizer": name}
        out[key] = {"max_relative_difference": largest, "at": at}
    return out


def host_peak_rss_gb() -> float:
    """This process's peak resident host memory so far (GB of 1e9 bytes; Linux reports KiB)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e9, 3)


def _all_peaks(peaks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The largest GPU peaks over the run's phases (each phase resets them)."""
    phases = [p for p in peaks.values() if p]
    if not phases:
        return {}
    out = {key: max(p[key] for p in phases) for key in ("peak_allocated_gb", "peak_reserved_gb")}
    if all("within_memory_rule" in p for p in phases):
        out["memory_rule_gb"] = phases[0]["memory_rule_gb"]
        out["within_memory_rule"] = all(p["within_memory_rule"] for p in phases)
    return out


def code_identity() -> dict[str, Any]:
    """The GPU-code identity of HEAD (``code_identity.gpu_code_identity``), or why there is
    none."""
    try:
        return gpu_code_identity()
    except Exception as exc:  # no git checkout, say: recorded, never a crash
        return {"commit": None, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}


def provenance(settings: DictConfig, scope: DriftScope, cfg: DictConfig) -> dict[str, Any]:
    return {"spec": SPEC, "code": git_state(), "gpu_code": code_identity(),
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "settings": OmegaConf.to_container(settings.drift, resolve=True),
            "run_config": config_summary(scope, cfg), "hardware": hardware(),
            "packages": package_versions()}  # fmt: skip


def run_drift(
    settings: DictConfig, cfg: DictConfig, env: Environment, *, chunk: int = REDUCTION_CHUNK
) -> dict[str, Any]:
    """The whole diagnostic at the config's B (module docstring); returns the result record,
    which the script writes to ``result.json``. The streams must exist (``prepare_streams``)."""
    scope = DriftScope.of(settings)
    check_run_config(scope, cfg)
    batch = int(cfg.train.batch_size)
    started = time.perf_counter()
    seed_everything(derive_seed(scope.seed, "torch"))
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    names, params = [n for n, _ in trainable], [p for _, p in trainable]
    prefix = StreamPrefix(cfg, env, scope.seed)
    responses = scope.batches * batch
    samples = prefix.take(0, responses)
    batches = [samples[i * batch : (i + 1) * batch] for i in range(scope.batches)]
    streams = {"drift": prefix.identity(responses)}

    peaks: dict[str, dict[str, Any]] = {}
    reset_peak(env.device)
    on_device, streams["preconditioner"] = second_moment(cfg, env, model, params)
    v0 = [_host_copy(v) for v in on_device]  # pinned host memory, read back per parameter
    del on_device
    peaks["v0"] = peak_memory(env.device)
    v0_sha256 = tensors_sha256(list(zip(names, v0, strict=True)))

    results, sums, rows = {}, {}, {}
    for name in scope.optimizers:
        reset_peak(env.device)
        begun = time.perf_counter()
        results[name], sums[name], rows[name] = optimizer_pass(
            name, model, params, batches, cfg, env, scope, v0 if name in USES_V0 else None, chunk
        )
        results[name]["seconds"] = round(time.perf_counter() - begun, 3)
        peaks[name] = peak_memory(env.device)
    del model, trainable, params
    reset_peak(env.device)
    cross = cross_cosines(sums, chunk)  # in host memory
    del sums
    gc.collect()

    first = rows[scope.optimizers[0]]
    agreement = gradient_agreement(rows)
    rounding = {name: (cross["sum_norm2"][name] - results[name]["sum_norm2"])
                / results[name]["sum_norm2"] if results[name]["sum_norm2"] > 0 else None
                for name in scope.optimizers}  # fmt: skip
    return {
        "batch": batch,
        "seed": scope.seed,
        "n": scope.batches,
        "halves": [[0, scope.half - 1], [scope.half, scope.batches - 1]],
        "eps": scope.eps,
        "beta2": scope.beta2,
        "optimizers": results,
        "cross_cosines": cross["cosines"],
        "batches": first,
        "parameters": {"tensors": len(names), "coordinates": sum(v.numel() for v in v0)},
        "v0_sha256": v0_sha256,
        "streams": streams,
        "peaks": {**peaks, "all": _all_peaks(peaks), "host_peak_rss_gb": host_peak_rss_gb()},
        "numerics": {
            **numerics(env.device),
            "updates": "fp32, per parameter",
            "half_sums": "fp32, on the device",
            "reductions": "float64",
            "reduction_chunk": chunk,
            # the four passes recompute the same 64 gradients: equal losses and ‖g_i‖², and if
            # not, how far apart and where
            "gradients_identical": agreement["identical"],
            "gradient_differences": {key: agreement[key] for key in ("loss", "grad_norm2")},
            # ‖S‖² of the fp32 sum in host memory against the float64 one from the halves
            "sum_rounding": rounding,
        },  # fmt: skip
        "seconds": round(time.perf_counter() - started, 3),
        "provenance": provenance(settings, scope, cfg),
    }


# ----------------------------------------------------------------------------- host memory


def _read_int(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def _stat(path: Path, key: str) -> int:
    try:
        for line in path.read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == key:
                return int(value)
    except (OSError, ValueError):
        pass
    return 0


UNLIMITED = 1 << 60  # a cgroup limit at or above this is no limit


def host_memory(
    meminfo: Path = Path("/proc/meminfo"), cgroup: Path = Path("/sys/fs/cgroup")
) -> dict[str, float | None]:
    """Free host memory for this process (GB of 1e9 bytes): ``meminfo_gb`` (MemAvailable),
    ``cgroup_gb`` (the container's limit less its working set, usage without inactive file
    cache, under cgroup v2 or v1; None without a limit) and ``available_gb``, the smaller."""
    meminfo_gb = None
    try:
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                meminfo_gb = int(line.split()[1]) * 1024 / 1e9
    except (OSError, ValueError, IndexError):
        pass
    cgroup_gb = None
    for limit, usage, stat, inactive in (
        ("memory.max", "memory.current", "memory.stat", "inactive_file"),
        ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes", "memory/memory.stat",
         "total_inactive_file"),
    ):  # fmt: skip
        cap, used = _read_int(cgroup / limit), _read_int(cgroup / usage)
        if cap is not None and cap < UNLIMITED and used is not None:
            working = max(used - _stat(cgroup / stat, inactive), 0)
            cgroup_gb = max(cap - working, 0) / 1e9
            break
    known = [x for x in (meminfo_gb, cgroup_gb) if x is not None]
    return {"available_gb": min(known) if known else None, "meminfo_gb": meminfo_gb,
            "cgroup_gb": cgroup_gb}  # fmt: skip


def host_memory_check(scope: DriftScope, **paths: Path) -> dict[str, Any]:
    """Whether the free host memory reaches the requirement (``host_required_gb``); an unknown
    amount does not."""
    found = host_memory(**paths)
    available = found["available_gb"]
    return {
        **found,
        "required_gb": scope.host_required_gb,
        "sufficient": available is not None and available >= scope.host_required_gb,
    }
