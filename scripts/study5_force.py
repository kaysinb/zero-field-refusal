"""Study 5's nonlinear population force along an F run's trajectory, on a panel of fresh responses.

A pilot of docs/study5-preregistration.md, "Extension 2": the force estimated on an independent
panel of fresh θ₀ responses, written as one JSON record.

    python scripts/study5_force.py CONFIG [key=value ...] --panel-config KEY=VALUE [KEY=VALUE ...]
        (--run RUN_DIR [--checkpoints DIR] [--step T ...]
         | --checkpoint FILE --step T --sha256 HEX [--checkpoint FILE --step T --sha256 HEX ...])
        --out RECORD.json [--epsilon E ...] [--project VECTOR ...] [--split-halves]
        [--save-force DIR]

CONFIG and the overrides are those of the studied seed's F run (``optimizer=adam_frozen``, its
rate and its ``seed``); the overrides come before every option. For fresh batches
E[g_t | θ_{t−1}] = ∇L(θ_{t−1}), L the population loss, whose minimum is θ₀. With δ_t = θ_t − θ₀,
θ_t the run's parameters after step t, the force's nonlinear part along the trajectory is

    f̂_t = ĝ(θ₀ + δ_t) − ĝ(θ₀) − [ĝ(θ₀ + εδ_t) − ĝ(θ₀ − εδ_t)] / (2ε),

the last term the central difference of the linear part Hδ_t. ĝ(θ) is the panel's mean gradient at
θ: (1/K) Σ_b g_b(θ) over the panel's K batches, every g_b by ``train.accumulate_gradient``, the
training loss path that ``linear_response.step_gradients`` calls per batch, under the studied
config. With the protocol's constant denominator (``losses.loss_denominator``: B × response cap,
``train.loss.normalization`` constant, required here) a batch's gradient is
∇ Σ_{i∈b} nll_i / (B·cap) whatever its token count, so that the mean over K equal batches is
∇ Σ_i nll_i / (K·B·cap), the gradient of the panel's mean loss in the run's normalization, with the
expectation of a training batch's gradient at the same point. ``accumulate_gradient`` adds each
batch's gradient to the parameters' ``.grad``; the panel's sum is formed there, batch after batch,
as each batch's gradient is formed from its micro-batches, and is read to the host once per point
(``summed_gradient``).

**The panel.** ``--panel-config`` holds overrides applied after the studied ones (e.g.
``seed=59990 train.steps=50 schedule.evaluate_at=[50]``: that seed's appendable training stream,
K = 50 batches of the run's B = 20, 1000 responses; a fixed-policy config evaluates at its last
step, so the steps come with it), read by ``linear_response.feed_batches`` (the stream must hold
them; nothing is generated). It must not be a stream that a studied run trains on, which is the
caller's responsibility; refused here: the studied seed's own stream, and another batch size or
response cap (the denominator is the studied run's). With ``--split-halves`` every quantity is
also computed on each half of the panel (the first K/2 batches and the last K/2), so that the
noise of f̂_t can be read from the halves' disagreement; the full panel's mean is then the mean of
the halves' means (equal halves), computed from the halves.

**The trajectory.** The θ_t files are torch.save of ``{name: tensor}`` over the trainable set,
``schedule.save_endpoint``'s format. With ``--run`` they are the run's step checkpoints as its
``meta.json`` records them (``inputs.step_checkpoints``: ``{path, sha256, step, names_sha256}``;
``--step`` selects some, by default all), each file at its recorded path or, with
``--checkpoints``, under that directory by its recorded name; the run's recorded hashes of θ₀'s
trainable tensors and of v₀ must equal those computed here. Without ``--run``, each
``--checkpoint`` FILE comes with its ``--step`` and ``--sha256``. A file whose sha256 is not the
recorded (or given) one, or whose tensors are not exactly the trainable set's names, shapes and
dtypes, is refused before anything is computed.

**What is computed.** On one fp32 model at θ₀, in order: v₀ by the run's own code path
(``linear_response.theta0_second_moment``) and D = (√v₀ + ε_Adam)⁻¹ as ``study5_predict.py``
computes it, kept on the host; ĝ(θ₀) once, cached on the host; then for each θ_t the points
θ₀ + δ_t and, for each ε of ``--epsilon`` (default 1e-2 5e-3), θ₀ ± εδ_t. A point is
θ₀ + sδ_t in float64 per tensor rounded to the parameters' fp32 (θ₀ + δ_t is the file's θ_t
exactly); the rounding ‖θ − (θ₀ + sδ_t)‖ of each point set is recorded, absolute and relative to
|s|·‖δ_t‖. Per θ_t, ε and part (``full`` and, with halves, ``first_half``, ``second_half``), in
float64 per tensor: ‖ĝ(θ₀ + δ_t) − ĝ(θ₀)‖ (``total_norm``, per θ_t), the linear part's norm, ‖f̂_t‖,
‖D ⊙ f̂_t‖ and ⟨D ⊙ f̂_t, w⟩/‖w‖ for each ``--project`` vector w (``vectors.load_vector`` files on
this θ₀, e.g. μ̂_{−i} and ∇r); with halves also ‖f̂^A − f̂^B‖ and ‖D ⊙ (f̂^A − f̂^B)‖; and ‖ĝ(θ₀)‖
per part, the panel's own noise floor at θ₀. With ``--save-force DIR`` the full panel's D ⊙ f̂_t
is written per θ_t and ε as a vector file (``vectors.save_vector``, kind ``force``) named
``force-step-<t>-eps-<ε>.pt``; an existing one is refused before anything is computed.

**Memory.** Every per-parameter accumulator is on the host in fp32, one tensor at a time moved
from the device: θ₀, D, and per part the sum Σ_b g_b(θ₀), ĝ(θ₀ + δ_t) − ĝ(θ₀) and the linear part
(the sum at θ₀ + εδ_t, then replaced by the difference with the sum at θ₀ − εδ_t): 2 + 3P vectors
for P parts, ≈ 26 GB at 1.5B without halves and ≈ 42 GB with them, plus D ⊙ f̂_t's ≈ 5 GB while a
force file is written. The θ_t files and the projection vectors are memory-mapped. The sums are
read from the device in fp32 without loss, and differences are taken from them in float64 per
tensor before they are kept in fp32 (a relative rounding of ~6e-8 of each difference itself), so
subtracting nearly equal gradients adds no rounding of the gradients' size. On the device: the
model and its gradient, as at a training step without Adam's state.

**The record** (``--out``, written atomically and never overwritten) holds every number above, the
checkpoints' steps, names, sha256 and names hashes, ‖δ_t‖, the run's id and config hash (with
``--run``), the hashes of θ₀'s trainable tensors and of v₀, the preconditioner stream's identity,
the panel's seed, overrides, batches, batch size, order seed, config hash and stream identity as
the run records one, the epsilons, the projection vectors' names, kinds, sha256 and norms, the
force files, the timings, the model, the numerics in force, the git state and the time. θ₀ is
restored exactly at the end (its hash is checked again). stdout carries numbers, ids and hashes
only. Exits 0 when written, 1 on a refusal or an error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from zero_field_refusal import linear_response, vectors
from zero_field_refusal.config import config_hash
from zero_field_refusal.models import freeze
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import configure_numerics, device_name, environment
from zero_field_refusal.seeding import derive_seed, seed_everything
from zero_field_refusal.study5_options import order_seed
from zero_field_refusal.tensor_hash import file_sha256, tensors_sha256
from zero_field_refusal.tracking import git_state
from zero_field_refusal.train import accumulate_gradient

HALVES = ("first_half", "second_half")
DEFAULT_EPSILONS = (1e-2, 5e-3)


# ------------------------------------------------------------------------------------- the inputs


def panel_config(cfg, config: str, overrides: Sequence[str], panel: Sequence[str]):
    """The panel's config: the studied one's overrides, then ``panel``'s; ``ValueError`` for the
    studied seed's own stream, another batch size or response cap (the denominator is the studied
    run's), or a loss without the constant denominator."""
    out = load_run_config(config, [*overrides, *panel])
    problems = []
    if cfg.train.loss.normalization != "constant":
        problems.append(f"loss normalization {cfg.train.loss.normalization}, not constant")
    if int(out.seed) == int(cfg.seed):
        problems.append(f"panel seed {out.seed} is the studied seed: its own training stream")
    for key in ("train.batch_size", "train.max_response_tokens"):
        if OmegaConf.select(out, key) != OmegaConf.select(cfg, key):
            problems.append(f"panel {key} {OmegaConf.select(out, key)}, not the studied "
                            f"{OmegaConf.select(cfg, key)}")  # fmt: skip
    if problems:
        raise ValueError("Not a panel for this run: " + "; ".join(problems))
    return out


def run_checkpoints(
    run_dir: Path, checkpoints: Path | None, steps: Sequence[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The run's identity and its recorded step checkpoints (``inputs.step_checkpoints``), those
    of ``steps`` (all when empty), each file at its recorded path or under ``checkpoints`` by its
    recorded name; ``ValueError`` when the run records none of them or no hash of θ₀ or v₀."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    inputs = meta.get("inputs", {})
    for key in ("theta0_trainable_sha256", "v0_sha256"):
        if not isinstance(inputs.get(key), str):
            raise ValueError(f"{run_dir.name}: the run records no {key}")
    recorded = inputs.get("step_checkpoints")
    if not isinstance(recorded, list) or not recorded:
        raise ValueError(f"{run_dir.name}: the run records no step checkpoints")
    by_step: dict[int, dict[str, Any]] = {}
    for item in recorded:
        step = item.get("step") if isinstance(item, dict) else None
        if (
            not isinstance(step, int)
            or step in by_step
            or not all(isinstance(item.get(key), str) for key in ("path", "sha256"))
        ):
            raise ValueError(f"{run_dir.name}: a step checkpoint record without a path or a "
                             f"sha256, or with a missing or repeated step ({step!r})")  # fmt: skip
        by_step[step] = item
    chosen = list(steps) or list(by_step)
    missing = [t for t in chosen if t not in by_step]
    if missing:
        raise ValueError(f"{run_dir.name}: no step checkpoint recorded for steps {missing}")
    out = []
    for t in chosen:
        item = by_step[t]
        path = Path(item["path"])
        if checkpoints is not None:
            path = Path(checkpoints) / path.name
        out.append({"path": path, "sha256": item["sha256"], "step": t,
                    "names_sha256": item.get("names_sha256")})  # fmt: skip
    run = {"run_id": meta.get("run_id"), "config_hash": meta.get("config_hash"),
           "theta0_trainable_sha256": inputs["theta0_trainable_sha256"],
           "v0_sha256": inputs["v0_sha256"]}  # fmt: skip
    return run, out


def given_checkpoints(
    files: Sequence[Path], steps: Sequence[int], sha256s: Sequence[str]
) -> list[dict[str, Any]]:
    """Plain files, each with its step and sha256."""
    if not files or not (len(files) == len(steps) == len(sha256s)):
        raise ValueError(f"{len(files)} --checkpoint, {len(steps)} --step and {len(sha256s)} "
                         "--sha256: one of each per file")  # fmt: skip
    return [{"path": Path(f), "sha256": s, "step": t, "names_sha256": None}
            for f, t, s in zip(files, steps, sha256s, strict=True)]  # fmt: skip


def check_files(checkpoints: Sequence[Mapping[str, Any]]) -> None:
    """Each file a regular file with the expected sha256, the steps positive and distinct;
    ``ValueError`` otherwise."""
    steps = [c["step"] for c in checkpoints]
    if len(set(steps)) != len(steps) or any(t < 1 for t in steps):
        raise ValueError(f"steps {steps}: positive and distinct")
    for c in checkpoints:
        path = Path(c["path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{path}: no checkpoint file")
        digest = file_sha256(path)
        if digest != c["sha256"]:
            raise ValueError(f"{path.name}: sha256 {digest}, not the expected {c['sha256']}")


def checkpoint_state(
    checkpoint: Mapping[str, Any], trainable: Sequence[tuple[str, torch.nn.Parameter]]
) -> dict[str, torch.Tensor]:
    """θ_t memory-mapped on the host: exactly the trainable set's names, shapes and dtypes (and
    the recorded names' hash, where recorded); ``ValueError`` otherwise."""
    names = [name for name, _ in trainable]
    recorded = checkpoint.get("names_sha256")
    if recorded is not None and recorded != vectors.names_sha256(names):
        raise ValueError(f"{Path(checkpoint['path']).name}: its recorded names are not the "
                         "trainable set's")  # fmt: skip
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=True, mmap=True)
    if sorted(state) != sorted(names):
        raise ValueError(f"{Path(checkpoint['path']).name}: {len(state)} tensors, not exactly "
                         f"the {len(names)} trainable ones")  # fmt: skip
    for name, p in trainable:
        if state[name].shape != p.shape or state[name].dtype != p.dtype:
            raise ValueError(f"{name}: the checkpoint's tensor is not the parameter's shape and "
                             "dtype")  # fmt: skip
    return state


def force_path(directory: Path, step: int, epsilon: float) -> Path:
    return Path(directory) / f"force-step-{step}-eps-{epsilon:g}.pt"


# ------------------------------------------------------------------------------- the computation


def panel_parts(batches: Sequence[Sequence[Any]], split: bool) -> list[list[Sequence[Any]]]:
    """The whole panel, or its two halves: the first K/2 batches and the last K/2."""
    if not split:
        return [list(batches)]
    if not batches or len(batches) % 2:
        raise ValueError(f"{len(batches)} panel batches do not split into two equal halves")
    half = len(batches) // 2
    return [list(batches[:half]), list(batches[half:])]


@contextmanager
def summed_gradient(
    model, trainable: Sequence[tuple[str, torch.nn.Parameter]], batches, cfg, env
) -> Iterator[dict[str, torch.Tensor]]:
    """Σ_b g_b over ``batches`` at the model's current point, each batch's gradient added to the
    parameters' own ``.grad`` by ``train.accumulate_gradient`` (in fp32, on the model's device);
    yields ``{name: .grad}``, valid inside the block. The gradients are cleared on exit."""
    try:
        for _, p in trainable:
            p.grad = None
        for index, batch in enumerate(batches):
            stats = accumulate_gradient(model, batch, cfg, env)
            if not math.isfinite(stats["loss"]):
                raise ValueError(f"the panel's loss is not finite on its batch {index}")
        missing = [name for name, p in trainable if p.grad is None]
        if missing:
            raise ValueError(f"no gradient for {len(missing)} trainable tensors")
        yield {name: p.grad for name, p in trainable}
    finally:
        for _, p in trainable:
            p.grad = None


def host_sum(model, trainable, batches, cfg, env) -> dict[str, torch.Tensor]:
    """``summed_gradient`` copied to the host, fp32 (lossless), one tensor at a time."""
    with summed_gradient(model, trainable, batches, cfg, env) as grads:
        return {name: g.detach().to("cpu", copy=True) for name, g in grads.items()}


@torch.no_grad()
def set_point(
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    theta0: Mapping[str, torch.Tensor],
    theta_t: Mapping[str, torch.Tensor],
    s: float,
) -> float:
    """Set the trainable parameters to θ₀ + s(θ_t − θ₀), in float64 per tensor on the host rounded
    to the parameters' dtype (s = 0: θ₀ and s = 1: θ_t, exactly); returns ‖θ − (θ₀ + s(θ_t −
    θ₀))‖, the rounding of the point set, in float64."""
    squares = []
    for name, p in trainable:
        if s == 0:
            value = theta0[name]
        elif s == 1:
            value = theta_t[name]
        else:
            t0 = theta0[name].double()
            exact = t0 + s * (theta_t[name].double() - t0)
            value = exact.to(p.dtype)
            squares.append((value.double() - exact).square().sum().item())
        p.copy_(value)
    return math.sqrt(math.fsum(squares))


def _labels(parts: int) -> list[str]:
    return ["full", *HALVES] if parts == 2 else ["full"]


def labelled_norms(
    names: Sequence[str],
    parts: Sequence[Mapping[str, torch.Tensor]],
    scales: Sequence[float],
    weights: Sequence[float],
) -> dict[str, float]:
    """‖v‖ per part (v_h = scale_h · parts[h]) and of the full panel's Σ_h weight_h · v_h, in
    float64 per tensor."""
    labels = _labels(len(parts))
    squares: dict[str, list[float]] = {label: [] for label in labels}
    for name in names:
        values = [scale * part[name].double() for scale, part in zip(scales, parts, strict=True)]
        full = sum(w * v for w, v in zip(weights, values, strict=True))
        squares["full"].append(full.square().sum().item())
        if len(parts) == 2:
            for label, v in zip(HALVES, values, strict=True):
                squares[label].append(v.square().sum().item())
    return {label: math.sqrt(math.fsum(s)) for label, s in squares.items()}


def force_terms(
    names: Sequence[str],
    total: Sequence[Mapping[str, torch.Tensor]],
    linear: Sequence[Mapping[str, torch.Tensor]],
    D: Mapping[str, torch.Tensor],
    weights: Sequence[float],
    projections: Sequence[tuple[Mapping[str, torch.Tensor], float]],
    keep: bool = False,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    """Per part and for the full panel (Σ_h weight_h · part), in float64 per tensor: the linear
    part's norm, ‖f̂‖, ‖D ⊙ f̂‖ and ⟨D ⊙ f̂, w⟩/‖w‖ per projection (w, ‖w‖), f̂ = total − linear;
    with two parts also ‖f̂^A − f̂^B‖ and ‖D ⊙ (f̂^A − f̂^B)‖. With ``keep``, also the full panel's
    D ⊙ f̂ (fp32, host)."""
    labels = _labels(len(total))
    keys = ("linear", "force", "Dforce")
    squares = {label: {key: [] for key in keys} for label in labels}
    dots = {label: [[] for _ in projections] for label in labels}
    difference: dict[str, list[float]] = {"force": [], "Dforce": []}
    kept: dict[str, torch.Tensor] | None = {} if keep else None
    for name in names:
        d = D[name].double()
        lin = [part[name].double() for part in linear]
        force = [t[name].double() - x for t, x in zip(total, lin, strict=True)]
        by_label = {"full": (sum(w * x for w, x in zip(weights, lin, strict=True)),
                             sum(w * f for w, f in zip(weights, force, strict=True)))}  # fmt: skip
        if len(total) == 2:
            by_label.update(zip(HALVES, zip(lin, force, strict=True), strict=True))
        for label, (lin_value, f) in by_label.items():
            df = d * f
            squares[label]["linear"].append(lin_value.square().sum().item())
            squares[label]["force"].append(f.square().sum().item())
            squares[label]["Dforce"].append(df.square().sum().item())
            for k, (w, _) in enumerate(projections):
                dots[label][k].append(torch.dot(df.reshape(-1),
                                                w[name].reshape(-1).double()).item())  # fmt: skip
            if label == "full" and kept is not None:
                kept[name] = df.float()
        if len(total) == 2:
            gap = force[0] - force[1]
            difference["force"].append(gap.square().sum().item())
            difference["Dforce"].append((d * gap).square().sum().item())
    out: dict[str, Any] = {
        "parts": {label: {"linear_norm": math.sqrt(math.fsum(squares[label]["linear"])),
                          "force_norm": math.sqrt(math.fsum(squares[label]["force"])),
                          "Dforce_norm": math.sqrt(math.fsum(squares[label]["Dforce"])),
                          "projections": [math.fsum(dots[label][k]) / norm
                                          for k, (_, norm) in enumerate(projections)]}
                  for label in labels},
        "halves_difference": None,
    }  # fmt: skip
    if len(total) == 2:
        out["halves_difference"] = {
            key + "_norm": math.sqrt(math.fsum(values)) for key, values in difference.items()
        }
    return out, kept


def measure(
    model,
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    theta0: Mapping[str, torch.Tensor],
    D: Mapping[str, torch.Tensor],
    parts: Sequence[Sequence[Any]],
    cfg,
    env,
    checkpoints: Sequence[Mapping[str, Any]],
    epsilons: Sequence[float],
    projections: Sequence[tuple[Mapping[str, torch.Tensor], float]] = (),
    save_force=None,
) -> dict[str, Any]:
    """ĝ(θ₀) per part, then per checkpoint and ε the force's terms (``force_terms``); the model's
    trainable parameters are θ₀ (``theta0``, host) on entry and are restored to it exactly on
    exit, also after an error. ``save_force(step, epsilon, tensors)`` receives the full panel's
    D ⊙ f̂ and returns what the record keeps of its file."""
    names = [name for name, _ in trainable]
    counts = [len(part) for part in parts]
    weights = [k / sum(counts) for k in counts]  # the full panel: Σ_h (K_h/K) · part mean
    out: dict[str, Any] = {"checkpoints": [], "seconds": {}}
    try:
        start = time.monotonic()
        g0 = [host_sum(model, trainable, part, cfg, env) for part in parts]  # Σ_b g_b(θ₀)
        out["g0_norm"] = labelled_norms(names, g0, [1 / k for k in counts], weights)
        out["seconds"]["g0"] = time.monotonic() - start
        for checkpoint in checkpoints:
            start = time.monotonic()
            theta_t = checkpoint_state(checkpoint, trainable)
            delta_norm = math.sqrt(math.fsum(
                (theta_t[name].double() - theta0[name].double()).square().sum().item()
                for name in names))  # fmt: skip
            entry: dict[str, Any] = {"step": checkpoint["step"],
                                     "file": Path(checkpoint["path"]).name,
                                     "sha256": checkpoint["sha256"],
                                     "names_sha256": vectors.names_sha256(names),
                                     "delta_norm": delta_norm, "epsilons": []}  # fmt: skip
            set_point(trainable, theta0, theta_t, 1.0)
            total = []  # ĝ(θ₀ + δ) − ĝ(θ₀) per part, fp32 on the host
            for k, (part, base) in enumerate(zip(parts, g0, strict=True)):
                with summed_gradient(model, trainable, part, cfg, env) as grads:
                    total.append({name: ((grads[name].detach().to("cpu").double()
                                          - base[name].double()) / counts[k]).float()
                                  for name in names})  # fmt: skip
            entry["total_norm"] = labelled_norms(names, total, [1.0] * len(parts), weights)
            for eps in epsilons:
                rounding = {"plus": set_point(trainable, theta0, theta_t, eps)}
                linear = [host_sum(model, trainable, part, cfg, env) for part in parts]
                rounding["minus"] = set_point(trainable, theta0, theta_t, -eps)
                for k, part in enumerate(parts):  # in place: (Σ⁺ − Σ⁻) / (2εK_h)
                    with summed_gradient(model, trainable, part, cfg, env) as grads:
                        for name in names:
                            minus = grads[name].detach().to("cpu").double()
                            linear[k][name] = ((linear[k][name].double() - minus)
                                               / (2 * eps * counts[k])).float()  # fmt: skip
                terms, kept = force_terms(names, total, linear, D, weights, projections,
                                          keep=save_force is not None)  # fmt: skip
                del linear
                scale = eps * delta_norm
                terms = {"epsilon": eps, **terms,
                         "rounding": {**rounding,
                                      "plus_relative": rounding["plus"] / scale if scale else None,
                                      "minus_relative": rounding["minus"] / scale if scale
                                      else None},
                         "force_file": None}  # fmt: skip
                if kept is not None:
                    terms["force_file"] = save_force(checkpoint["step"], eps, kept)
                    del kept
                entry["epsilons"].append(terms)
            set_point(trainable, theta0, theta0, 0.0)
            del total, theta_t
            entry["seconds"] = time.monotonic() - start
            out["checkpoints"].append(entry)
    finally:
        set_point(trainable, theta0, theta0, 0.0)
    return out


# ------------------------------------------------------------------------------------- the record


def run(cfg, panel_cfg, env, checkpoints, numerics: dict, *, epsilons, projections=(),
        split: bool = False, save_dir: Path | None = None, run_record: dict | None = None,
        panel_overrides: Sequence[str] = ()) -> dict:  # fmt: skip
    """The force record of ``cfg``'s seed in ``env`` on ``panel_cfg``'s panel; ``ValueError`` when
    θ₀ or v₀ are not the run's (``run_record``), or a checkpoint or projection vector is not on
    this trainable set and θ₀."""
    began = time.monotonic()
    seed_everything(derive_seed(cfg.seed, "torch"))  # as the run seeds itself
    batches, panel_stream = linear_response.feed_batches(panel_cfg, env)
    parts = panel_parts(batches, split)
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    names = [name for name, _ in trainable]
    theta0_sha = tensors_sha256(trainable)
    if run_record is not None and theta0_sha != run_record["theta0_trainable_sha256"]:
        raise ValueError(f"θ₀'s trainable tensors hash to {theta0_sha}, not the run's "
                         f"{run_record['theta0_trainable_sha256']}")  # fmt: skip
    for checkpoint in checkpoints:  # names, shapes and dtypes, before anything is computed
        checkpoint_state(checkpoint, trainable)
    loaded = []
    for path in projections:
        tensors, record = vectors.load_vector(path, names, theta0_sha256=theta0_sha, mmap=True)
        if not (math.isfinite(record["norm"]) and record["norm"] > 0):
            raise ValueError(f"{Path(path).name}: norm {record['norm']}")
        loaded.append((tensors, record["norm"], {"name": Path(path).name, "kind": record["kind"],
                                                 "sha256": record["sha256"],
                                                 "norm": record["norm"]}))  # fmt: skip
    start = time.monotonic()
    v0, preconditioner = linear_response.theta0_second_moment(cfg, env, model, trainable)
    v0_sha = linear_response.v0_sha256(trainable, v0)
    if run_record is not None and v0_sha != run_record["v0_sha256"]:
        raise ValueError(f"v₀ hashes to {v0_sha}, not the run's {run_record['v0_sha256']}")
    eps_adam = float(cfg.optimizer.eps)
    D = {}
    for k, (name, _) in enumerate(trainable):  # as study5_predict.py, then to the host
        D[name] = v0[k].sqrt_().add_(eps_adam).reciprocal_().to("cpu")
        v0[k] = None
    del v0
    v0_seconds = time.monotonic() - start
    theta0 = {name: p.detach().to("cpu", copy=True) for name, p in trainable}

    def save_force(step: int, epsilon: float, tensors: dict) -> dict:
        path = force_path(save_dir, step, epsilon)
        provenance = {"quantity": "D ⊙ f̂_t, the full panel", "step": step, "epsilon": epsilon,
                      "seed": int(cfg.seed), "lr": float(cfg.optimizer.lr),
                      "run_id": None if run_record is None else run_record["run_id"],
                      "checkpoint_sha256": next(c["sha256"] for c in checkpoints
                                                if c["step"] == step),
                      "v0_sha256": v0_sha, "panel_seed": int(panel_cfg.seed),
                      "panel_stream_sha256": panel_stream["sha256"],
                      "panel_batches": len(batches)}  # fmt: skip
        record = vectors.save_vector(path, tensors, names, kind="force",
                                     theta0_sha256=theta0_sha, provenance=provenance)  # fmt: skip
        return {"name": path.name, "sha256": record["sha256"], "norm": record["norm"]}

    measured = measure(model, trainable, theta0, D, parts, cfg, env, checkpoints, epsilons,
                       [(t, norm) for t, norm, _ in loaded],
                       save_force if save_dir is not None else None)  # fmt: skip
    if tensors_sha256(trainable) != theta0_sha:
        raise ValueError("θ₀ was not restored exactly")
    return {
        "kind": "study5-force",
        "seed": int(cfg.seed),
        "lr": float(cfg.optimizer.lr),
        "eps_adam": eps_adam,
        "steps": int(cfg.train.steps),
        "batch_size": int(cfg.train.batch_size),
        "run": None if run_record is None else {"run_id": run_record["run_id"],
                                                "config_hash": run_record["config_hash"]},
        "config_hash": config_hash(cfg),
        "theta0_trainable_sha256": theta0_sha,
        "v0_sha256": v0_sha,
        "theta0_restored": True,
        "streams": {"preconditioner": preconditioner},
        "panel": {"seed": int(panel_cfg.seed), "overrides": list(panel_overrides),
                  "batches": len(batches), "batch_size": int(panel_cfg.train.batch_size),
                  "responses": sum(len(b) for b in batches), "order_seed": order_seed(panel_cfg),
                  "split_halves": split, "config_hash": config_hash(panel_cfg),
                  "stream": panel_stream},
        "loss": {"normalization": cfg.train.loss.normalization,
                 "denominator": float(cfg.train.loss.denominator),
                 "max_response_tokens": int(cfg.train.max_response_tokens)},
        "epsilons": list(epsilons),
        "projections": [meta for _, _, meta in loaded],
        "g0_norm": measured["g0_norm"],
        "checkpoints": measured["checkpoints"],
        "seconds": {"v0": v0_seconds, "g0": measured["seconds"]["g0"],
                    "total": time.monotonic() - began},
        "micro_batch_size": int(cfg.execution.micro_batch_size),
        "optimizer": cfg.optimizer.name,
        "model": {"name": cfg.model.name, "revision": cfg.model.revision,
                  "dtype": cfg.model.dtype, "attn_implementation": cfg.model.attn_implementation},
        "prompt_manifest_sha256": env.prompt_manifest_sha256,
        "numerics": numerics,
        "device": device_name(torch.device(env.device)),
        "git": git_state(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }  # fmt: skip


def write_new(path: Path, record: dict) -> None:
    """Write ``record`` to ``path`` atomically; refuse an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; a record is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(partial, path)


def _parts_line(values: Mapping[str, float]) -> str:
    return " ".join(f"{label} {value:.6e}" for label, value in values.items())


def report(record: dict) -> None:
    """Numbers, ids and hashes only."""
    panel = record["panel"]
    print(f"seed {record['seed']} lr {record['lr']:.6e} panel_seed {panel['seed']} panel_batches "
          f"{panel['batches']} split_halves {int(panel['split_halves'])}")  # fmt: skip
    print(f"theta0_trainable_sha256 {record['theta0_trainable_sha256']}")
    print(f"v0_sha256 {record['v0_sha256']}")
    print(f"stream panel {panel['stream']['sha256']} count {panel['stream']['count']}")
    print(f"g0_norm {_parts_line(record['g0_norm'])}")
    for c in record["checkpoints"]:
        print(f"step {c['step']} sha256 {c['sha256']} delta_norm {c['delta_norm']:.6e} "
              f"total_norm {_parts_line(c['total_norm'])}")  # fmt: skip
        for e in c["epsilons"]:
            for label, part in e["parts"].items():
                projections = " ".join(f"{x:.6e}" for x in part["projections"])
                print(
                    f"step {c['step']} eps {e['epsilon']:g} {label} "
                    f"linear {part['linear_norm']:.6e} force {part['force_norm']:.6e} "
                    f"Dforce {part['Dforce_norm']:.6e} projections {projections}".rstrip()
                )
            if e["halves_difference"] is not None:
                gap = e["halves_difference"]
                print(
                    f"step {c['step']} eps {e['epsilon']:g} halves_difference "
                    f"force {gap['force_norm']:.6e} Dforce {gap['Dforce_norm']:.6e}"
                )
    print(
        f"theta0_restored {int(record['theta0_restored'])} seconds {record['seconds']['total']:.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--panel-config", nargs="+", required=True, metavar="KEY=VALUE")
    parser.add_argument("--run", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--step", type=int, action="append", default=[])
    parser.add_argument("--sha256", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    parser.add_argument("--project", type=Path, nargs="+", default=[])
    parser.add_argument("--split-halves", action="store_true")
    parser.add_argument("--save-force", type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"record exists: {args.out}")
        return 1
    try:
        epsilons = args.epsilon
        if any(not 0 < e < 1 for e in epsilons) or len(set(epsilons)) != len(epsilons):
            raise ValueError(f"epsilons {epsilons}: distinct, in (0, 1)")
        if args.run is not None:
            if args.checkpoint or args.sha256:
                raise ValueError("--run takes its checkpoints from the run's record")
            run_record, checkpoints = run_checkpoints(args.run, args.checkpoints, args.step)
        else:
            if args.checkpoints is not None:
                raise ValueError("--checkpoints needs --run")
            run_record, checkpoints = None, given_checkpoints(args.checkpoint, args.step,
                                                              args.sha256)  # fmt: skip
        check_files(checkpoints)
        if args.save_force is not None:
            for c in checkpoints:
                for eps in args.epsilon:
                    path = force_path(args.save_force, c["step"], eps)
                    if path.exists() or vectors.record_path(path).exists():
                        raise ValueError(f"{path} exists; a vector file is never overwritten")
        cfg = load_run_config(args.config, args.overrides)  # the protocol: fp32, frozen modules
        linear_response.require_frozen_adam(cfg)
        panel = panel_config(cfg, args.config, args.overrides, args.panel_config)
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    numerics = configure_numerics(cfg, cfg.execution.device)  # first: before any CUDA work
    env = environment(cfg, cfg.execution.device)
    try:
        record = run(cfg, panel, env, checkpoints, numerics, epsilons=args.epsilon,
                     projections=args.project, split=args.split_halves,
                     save_dir=args.save_force, run_record=run_record,
                     panel_overrides=args.panel_config)  # fmt: skip
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    write_new(args.out, record)
    report(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
