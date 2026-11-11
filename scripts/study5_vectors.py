"""Study 5's trained displacements over the fresh seeds (docs/study5-preregistration.md,
"Extension 2", M1 and the directions of M4): the Gram matrix, ‖μ‖² free of the noise, the mean μ̂
and the leave-one-out means, the balanced-sign mixtures and the removal vectors, from the runs'
endpoint checkpoints on the host, written as vector files (``zero_field_refusal.vectors``) and
one JSON record.

    python scripts/study5_vectors.py CONFIG [key=value ...] (--runs RUN_DIR ... | --runs-from FILE)
        --checkpoints DIR --out-dir OUT [--grad-r FILE] [--loo-seeds SEED ...]
        [--removal-seeds SEED ...] [--mixtures K] [--skip-bad]

**The runs.** CONFIG and the overrides are the fresh seeds' F runs at η_l (``experiments/study5``,
``study5.role=fresh``, ``optimizer.lr``, the ``seed`` of any of them); ``--runs-from`` names one
run directory per line. Each run's seed is read from its ``config.yaml``, and the run must be
CONFIG at that seed (``ladder.read_f_run``: its config hash, all its ``train.steps`` updates). Its
endpoint checkpoint ``DIR/<run id>.pt`` must have the sha256 the run recorded
(``line_scan.checkpoint_of``) and hold exactly the trainable set's names, shapes and dtypes (the
checks of ``line_scan.endpoint_state``, made here on the memory-mapped file). A run whose
checkpoint is missing or is not the recorded one is refused (exit 1, nothing written) or, with
``--skip-bad``, left out and recorded. θ₀ is ``env.new_model()`` on the host; its trainable
tensors (``freeze``) must hash to every run's θ₀. Seeds and run ids must be distinct. The runs are
ordered by seed: position j is the j-th smallest seed.

**What is computed.** Δθ_i = θ₂₅,i − θ₀ over the trainable set, the fp32 differences (as the run's
own ``displacement_norm``), every sum over them in float64.

1. G_ij = ⟨Δθ_i, Δθ_j⟩ and the norms ‖Δθ_i‖ = √G_ii.
2. ‖μ‖² free of the noise, the mean of G_ij over i ≠ j; beside it ‖μ̂‖² = Σ_ij G_ij / n², which
   exceeds it by (mean_i ‖Δθ_i‖² − that mean) / n; the mean pairwise cosine.
3. μ̂ = Σ_i Δθ_i / n (``mean.pt``, kind ``mean``).
4. μ̂_{−i} = (S − Δθ_i)/(n − 1) for each seed of ``--loo-seeds`` (default: all;
   ``mean_minus_<seed>.pt``, kind ``mean``); for every i, from G: p_i = ⟨Δθ_i, μ̂_{−i}⟩/‖μ̂_{−i}‖
   and cos(Δθ_i, μ̂_{−i}), with their mean, SD (n − 1 degrees of freedom) and coefficient of
   variation SD/|mean|.
5. The split halves: the seeds at even positions (0, 2, …) and at odd positions (1, 3, …); the
   norms of the two half-means and their cosine.
6. With ``--grad-r`` (a vector file of kind ``grad_r`` on the same θ₀): ⟨Δθ_i, ∇r⟩ per run, its
   mean, SD and seed-level SE (SD/√n), and ⟨v, ∇r⟩ of every vector v written, as written (fp32);
   that of ``mean.pt`` is ⟨μ̂, ∇r⟩.
7. Mixtures k = 1 … K (``--mixtures``, default 3): s_kj = +1 at the first n/2 entries of
   ``numpy.random.default_rng(derive_seed(k, "study5", "mixture")).permutation(n)``, −1 at the
   others; m_k = Σ_j s_kj Δθ_j / n, scaled to ‖μ̂‖ (``mixture_<k>.pt``, kind ``mixture``). The
   signs sum to zero, so a common component of every Δθ_j cancels exactly.
8. Removals, for each seed i of ``--removal-seeds`` (default: those of 50000–50005 present), kind
   ``removal``: ``removal_mean_<i>.pt`` = Δθ_i − μ̂_{−i}; ``removal_mixture_<i>.pt`` = Δθ_i − m_{−i},
   m_{−i} the balanced-sign mixture of the other seeds but the largest of them,
   Σ_j s_j Δθ_j / (n − 2) with s_j = +1 at the first (n − 2)/2 entries of
   ``default_rng(derive_seed(i, "study5", "removal")).permutation(n − 2)``, scaled so that
   ‖m_{−i}‖ = ‖μ̂_{−i}‖. Recorded: ‖Δθ_i‖, both removed vectors' norms, both results' norms and
   their cosines with Δθ_i.

A balanced-sign mixture needs an even n: with any requested, an odd n is refused. Each vector
written is a fixed combination Σ_j w_j Δθ_j, its weights set from G, so its norm and its inner
products with each Δθ_i are exact from G (√(wᵀGw), Gw); its file is the fp32 rounding of the
float64 combination, and records its own norm.

**The record** ``OUT/record.json``, written last (a directory without it is incomplete) and never
overwritten: the runs (seed, run id, config hash, checkpoint name, sha256, step and names hash,
‖Δθ‖ as the run recorded it and as computed here), the runs left out and why, θ₀'s hash, the
names' hash and the element count, G in full, every quantity above, each vector's weights and
provenance with its file's record from ``vectors.save_vector`` (sha256, norm, count), the model,
the numerics, the git state and the time. stdout carries numbers, ids and hashes only; no prompt
or response is read. Exits 0 when written, 1 on a refusal: an existing record or vector file, or
too little free disk, refuses the call before anything is written, and a call that fails while
writing removes the files it wrote.

**Memory.** Two passes over the checkpoints, memory-mapped (``torch.load(mmap=True)``), each a
slice of at most ``CHUNK`` elements of one tensor at a time. Pass 1 forms the n × CHUNK float64
differences X and adds XXᵀ to G (and X∇r). Pass 2, once the weights W are set, forms WX for every
vector at once and appends each row, in fp32, to that vector's scratch file in OUT; each vector
file is then written by ``vectors.save_vector`` from its scratch file memory-mapped (no copy in
memory) and the scratch file removed. Host memory at the peak: θ₀'s trainable tensors, X, WX in
float64 and fp32, and one tensor of the vector being written in float64; the pages of the
checkpoints and of the scratch files are file-backed. Disk: the vector files and, at the peak, one
vector more. No GPU is used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from zero_field_refusal import ladder, line_scan, linear_response, vectors
from zero_field_refusal.models import freeze
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import configure_numerics, environment
from zero_field_refusal.seeding import derive_seed
from zero_field_refusal.tensor_hash import tensors_sha256
from zero_field_refusal.tracking import git_state

CHUNK = 1 << 20  # elements of one tensor per step of a pass
MIXTURES = 3  # m_1 … m_3
REMOVAL_SEEDS = tuple(range(50000, 50006))  # M4's removal seeds, those present
MIXTURE_KEYS = ("study5", "mixture")  # derive_seed(k, *MIXTURE_KEYS)
REMOVAL_KEYS = ("study5", "removal")  # derive_seed(seed, *REMOVAL_KEYS)
RECORD = "record.json"
DEVICE = "cpu"
FLOAT32 = np.dtype(np.float32).itemsize


# ------------------------------------------------------------------------------------ the runs


def run_seed(run_dir: Path) -> int:
    """The seed of the run's own resolved config."""
    return int(OmegaConf.load(Path(run_dir) / "config.yaml").seed)


def read_runs(
    config: str, overrides: list[str], run_dirs: list[Path], checkpoints: Path, *, skip_bad: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Each run's ``read_f_run`` under CONFIG at the run's seed and its ``checkpoint_of``, in seed
    order, and the runs left out: with ``skip_bad`` a checkpoint that is missing or not the
    recorded one leaves its run out instead of refusing. ``ValueError`` otherwise."""
    runs, skipped = [], []
    for run_dir in run_dirs:
        seed = run_seed(run_dir)
        cfg = load_run_config(config, [*overrides, f"seed={seed}"])
        f_run = ladder.read_f_run(run_dir, cfg)
        try:
            checkpoint = line_scan.checkpoint_of(run_dir, f_run, checkpoints, int(cfg.train.steps))
        except ValueError as exc:
            if not skip_bad:
                raise
            skipped.append({"seed": seed, "run_id": f_run["run_id"], "reason": str(exc)})
            continue
        runs.append({"seed": seed, "f_run": f_run, "checkpoint": checkpoint})
    runs.sort(key=lambda run: run["seed"])
    for what, values in (
        ("seed", [run["seed"] for run in runs]),
        ("run id", [run["f_run"]["run_id"] for run in runs]),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"a {what} is given twice: {sorted(values)}")
    return runs, skipped


def open_endpoint(
    checkpoint: dict[str, Any], trainable: list[tuple[str, torch.nn.Parameter]]
) -> dict[str, torch.Tensor]:
    """The checkpoint's tensors memory-mapped, with ``line_scan.endpoint_state``'s checks: exactly
    the trainable set's names, shapes and dtypes, and the recorded names' hash where recorded."""
    names = [name for name, _ in trainable]
    recorded = checkpoint.get("names_sha256")
    if recorded is not None and recorded != line_scan.names_sha256(names):
        raise ValueError(f"{checkpoint['path']}: its recorded names are not the trainable set's")
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=True, mmap=True)
    if sorted(state) != sorted(names):
        raise ValueError(
            f"{checkpoint['path']}: {len(state)} tensors, not exactly the "
            f"{len(names)} trainable ones"
        )
    for name, p in trainable:
        if state[name].shape != p.shape or state[name].dtype != p.dtype:
            raise ValueError(f"{checkpoint['path']}: {name} is not the parameter's shape and dtype")
    return state


# ---------------------------------------------------------------------------------- the weights


def balanced_signs(count: int, rng_seed: int) -> list[int]:
    """+1 at the first count/2 entries of ``numpy.random.default_rng(rng_seed).permutation(count)``,
    −1 at the others."""
    if count < 2 or count % 2:
        raise ValueError(f"a balanced-sign mixture needs an even count of at least 2, not {count}")
    signs = [-1] * count
    for position in np.random.default_rng(rng_seed).permutation(count)[: count // 2]:
        signs[int(position)] = 1
    return signs


def quadratic(G: np.ndarray, w: np.ndarray) -> float:
    """‖Σ_j w_j Δθ_j‖² = wᵀGw."""
    return float(w @ G @ w)


def leave_one_out(n: int, i: int) -> np.ndarray:
    w = np.full(n, 1.0 / (n - 1))
    w[i] = 0.0
    return w


def plan(
    G: np.ndarray,
    seeds: list[int],
    run_ids: list[str],
    loo_seeds: list[int],
    removal_seeds: list[int],
    mixtures: int,
) -> list[dict[str, Any]]:
    """Every vector to write, in order: its file, kind, weights over the runs (seed order) and
    provenance (module docstring, 3, 4, 7 and 8); the scales from G."""
    n = len(seeds)
    at = {seed: j for j, seed in enumerate(seeds)}
    mean = np.full(n, 1.0 / n)
    mean_norm = math.sqrt(quadratic(G, mean))
    out = [
        {
            "file": "mean.pt",
            "kind": "mean",
            "weights": mean,
            "provenance": {"seeds": seeds, "run_ids": run_ids, "divisor": n},
        }
    ]
    for seed in loo_seeds:
        kept = [j for j in range(n) if j != at[seed]]
        out.append(
            {
                "file": f"mean_minus_{seed}.pt",
                "kind": "mean",
                "weights": leave_one_out(n, at[seed]),
                "provenance": {
                    "left_out": seed,
                    "seeds": [seeds[j] for j in kept],
                    "run_ids": [run_ids[j] for j in kept],
                    "divisor": n - 1,
                },
            }
        )
    for k in range(1, mixtures + 1):
        rng_seed = derive_seed(k, *MIXTURE_KEYS)
        signs = balanced_signs(n, rng_seed)
        unscaled = np.array(signs, dtype=np.float64) / n
        unscaled_norm = math.sqrt(quadratic(G, unscaled))
        scale = mean_norm / unscaled_norm
        out.append(
            {
                "file": f"mixture_{k}.pt",
                "kind": "mixture",
                "weights": scale * unscaled,
                "provenance": {
                    "k": k,
                    "seeds": seeds,
                    "signs": signs,
                    "rng_seed": rng_seed,
                    "divisor": n,
                    "scale": scale,
                    "unscaled_norm": unscaled_norm,
                    "target_norm": mean_norm,
                    "target": "mean.pt",
                },
            }
        )
    for seed in removal_seeds:
        i = at[seed]
        own = np.zeros(n)
        own[i] = 1.0
        loo = leave_one_out(n, i)
        loo_norm = math.sqrt(quadratic(G, loo))
        out.append(
            {
                "file": f"removal_mean_{seed}.pt",
                "kind": "removal",
                "weights": own - loo,
                "provenance": {
                    "seed": seed,
                    "removed": f"mean_minus_{seed}",
                    "removed_norm": loo_norm,
                },
            }
        )
        others = [s for s in seeds if s != seed]
        left_out = max(others)
        used = [s for s in others if s != left_out]
        rng_seed = derive_seed(seed, *REMOVAL_KEYS)
        signs = balanced_signs(n - 2, rng_seed)
        mixture = np.zeros(n)
        for s, sign in zip(used, signs, strict=True):
            mixture[at[s]] = sign / (n - 2)
        unscaled_norm = math.sqrt(quadratic(G, mixture))
        scale = loo_norm / unscaled_norm
        out.append(
            {
                "file": f"removal_mixture_{seed}.pt",
                "kind": "removal",
                "weights": own - scale * mixture,
                "provenance": {
                    "seed": seed,
                    "removed": "mixture",
                    "left_out": left_out,
                    "seeds": used,
                    "signs": signs,
                    "rng_seed": rng_seed,
                    "divisor": n - 2,
                    "scale": scale,
                    "unscaled_norm": unscaled_norm,
                    "target_norm": loo_norm,
                    "target": f"mean_minus_{seed}",
                },
            }
        )
    return out


# ------------------------------------------------------------------------------ the Gram's numbers


def spread(values: np.ndarray) -> dict[str, float | None]:
    """Mean, SD (n − 1), seed-level SE (SD/√n) and coefficient of variation SD/|mean|."""
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    return {
        "mean": mean,
        "sd": sd,
        "se": sd / math.sqrt(values.size),
        "cv": sd / abs(mean) if mean else None,
    }


def gram_numbers(G: np.ndarray, seeds: list[int]) -> dict[str, Any]:
    """Items 1, 2, 4 (every i) and 5 of the module docstring, from G alone."""
    n = len(seeds)
    diagonal = np.diag(G).copy()
    norms = np.sqrt(diagonal)
    total, rows = float(G.sum()), G.sum(axis=1)
    pairs = n * (n - 1)
    cosines = G / np.outer(norms, norms)
    loo_norms = np.sqrt(total - 2 * rows + diagonal) / (n - 1)
    p = (rows - diagonal) / (n - 1) / loo_norms
    halves = {}
    for name, positions in (("even", list(range(0, n, 2))), ("odd", list(range(1, n, 2)))):
        w = np.zeros(n)
        w[positions] = 1.0 / len(positions)
        halves[name] = {"seeds": [seeds[j] for j in positions], "weights": w}
    a, b = halves["even"]["weights"], halves["odd"]["weights"]
    a_norm, b_norm = math.sqrt(quadratic(G, a)), math.sqrt(quadratic(G, b))
    noise_free = (total - float(diagonal.sum())) / pairs
    return {
        "norms": norms.tolist(),
        "mean": {
            "norm_sq_noise_free": noise_free,
            "norm_sq": total / n**2,
            "norm": math.sqrt(total) / n,
            "mean_norm_sq": float(diagonal.mean()),
            "excess": total / n**2 - noise_free,
            "mean_pairwise_cosine": float((cosines.sum() - np.trace(cosines)) / pairs),
        },
        "leave_one_out": {
            "seeds": seeds,
            "p": p.tolist(),
            "cos": (p / norms).tolist(),
            "norm": loo_norms.tolist(),
            "p_spread": spread(p),
            "cos_spread": spread(p / norms),
        },
        "halves": {
            "even": {"seeds": halves["even"]["seeds"], "norm": a_norm},
            "odd": {"seeds": halves["odd"]["seeds"], "norm": b_norm},
            "cosine": float(a @ G @ b) / (a_norm * b_norm),
        },
    }


# ------------------------------------------------------------------------------------ the passes


def slices(theta0: dict[str, torch.Tensor], chunk: int):
    """(name, start, stop) over the trainable tensors in order, at most ``chunk`` elements each."""
    for name, t in theta0.items():
        for start in range(0, t.numel(), chunk):
            yield name, start, min(start + chunk, t.numel())


def differences(
    states: list[dict[str, torch.Tensor]],
    theta0: dict[str, torch.Tensor],
    name: str,
    start: int,
    stop: int,
) -> torch.Tensor:
    """The n × (stop − start) float64 slice of the Δθ_i: the fp32 differences."""
    base = theta0[name].reshape(-1)[start:stop]
    out = torch.empty(len(states), stop - start, dtype=torch.float64)
    for row, state in zip(out, states, strict=True):
        row.copy_(state[name].reshape(-1)[start:stop] - base)
    return out


def gram_pass(
    states: list[dict[str, torch.Tensor]],
    theta0: dict[str, torch.Tensor],
    grad: dict[str, torch.Tensor] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Pass 1: G (symmetrized) and, with ∇r, the ⟨Δθ_i, ∇r⟩."""
    n = len(states)
    G = torch.zeros(n, n, dtype=torch.float64)
    dots = None if grad is None else torch.zeros(n, dtype=torch.float64)
    for name, start, stop in slices(theta0, CHUNK):
        X = differences(states, theta0, name, start, stop)
        G.addmm_(X, X.T)
        if dots is not None:
            dots.addmv_(X, grad[name].reshape(-1)[start:stop].double())
    G = (G + G.T) / 2
    return G.numpy(), None if dots is None else dots.numpy()


def write_pass(
    states: list[dict[str, torch.Tensor]],
    theta0: dict[str, torch.Tensor],
    W: np.ndarray,
    paths: list[Path],
    grad: dict[str, torch.Tensor] | None,
) -> np.ndarray | None:
    """Pass 2: each vector Σ_j W_oj Δθ_j in fp32, appended slice by slice to its scratch file
    ``paths[o]``; with ∇r, each written vector's ⟨v, ∇r⟩."""
    weights = torch.from_numpy(W)
    dots = None if grad is None else torch.zeros(len(paths), dtype=torch.float64)
    with ExitStack() as stack:
        files = [stack.enter_context(path.open("wb")) for path in paths]
        for name, start, stop in slices(theta0, CHUNK):
            Y = (weights @ differences(states, theta0, name, start, stop)).float()
            for row, f in zip(Y, files, strict=True):
                f.write(row.numpy())
            if dots is not None:
                dots.addmv_(Y.double(), grad[name].reshape(-1)[start:stop].double())
    return None if dots is None else dots.numpy()


def scratch_vector(path: Path, theta0: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The vector in a scratch file, one memory-mapped fp32 tensor per name (copy-on-write: the
    file is never changed through it)."""
    out, offset = {}, 0
    for name, t in theta0.items():
        array = np.memmap(
            path, dtype=np.float32, mode="c", offset=offset * FLOAT32, shape=tuple(t.shape)
        )
        out[name] = torch.from_numpy(array)
        offset += t.numel()
    return out


def require_space(out_dir: Path, vectors_count: int, count: int) -> None:
    """Free disk for every vector file and one scratch file more (each with 1 MiB of slack)."""
    need = (vectors_count + 1) * (count * FLOAT32 + (1 << 20))
    where = out_dir
    while not where.exists():
        where = where.parent
    free = shutil.disk_usage(where).free
    if free < need:
        raise ValueError(f"{where}: {free} bytes free, {need} needed for {vectors_count} vectors")


# ---------------------------------------------------------------------------------------- the run


def run(
    cfg,
    env,
    runs: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    *,
    out_dir: Path,
    loo_seeds: list[int],
    removal_seeds: list[int],
    mixtures: int,
    grad_r: Path | None,
    numerics: dict,
) -> dict[str, Any]:
    """The vector files and the record (module docstring); ``ValueError`` before any write for θ₀,
    a checkpoint or ∇r that is not the runs', or an existing file."""
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    theta0_sha = tensors_sha256(trainable)
    for item in runs:
        recorded = item["f_run"]["theta0_trainable_sha256"]
        if recorded != theta0_sha:
            raise ValueError(
                f"θ₀'s trainable tensors hash to {theta0_sha}, not run "
                f"{item['f_run']['run_id']}'s {recorded}"
            )
    names = [name for name, _ in trainable]
    theta0 = {name: p.detach() for name, p in trainable}
    del model  # the frozen modules go; the trainable tensors stay
    count = sum(t.numel() for t in theta0.values())
    states = [open_endpoint(item["checkpoint"], trainable) for item in runs]
    grad, grad_record = None, None
    if grad_r is not None:
        grad, grad_record = vectors.load_vector(grad_r, names, theta0_sha256=theta0_sha, mmap=True)
        if grad_record["kind"] != "grad_r":
            raise ValueError(f"{grad_r.name}: a vector of kind {grad_record['kind']!r}, not grad_r")
    seeds = [item["seed"] for item in runs]
    run_ids = [item["f_run"]["run_id"] for item in runs]

    G, run_dots = gram_pass(states, theta0, grad)
    numbers = gram_numbers(G, seeds)
    planned = plan(G, seeds, run_ids, loo_seeds, removal_seeds, mixtures)
    for item in planned:
        path = out_dir / item["file"]
        if path.exists() or vectors.record_path(path).exists():
            raise ValueError(f"{path} exists; a vector file is never overwritten")
    require_space(out_dir, len(planned), count)

    out_dir.mkdir(parents=True, exist_ok=True)
    W = np.stack([item["weights"] for item in planned])
    written: list[Path] = []
    scratch = Path(tempfile.mkdtemp(prefix=".scratch-", dir=out_dir))
    try:
        paths = [scratch / f"{o}.f32" for o in range(len(planned))]
        vector_dots = write_pass(states, theta0, W, paths, grad)
        for item, path in zip(planned, paths, strict=True):
            if path.stat().st_size != count * FLOAT32:
                raise OSError(f"{path.name}: {path.stat().st_size} bytes, not {count * FLOAT32}")
            target = out_dir / item["file"]
            written.append(target)
            item["record"] = vectors.save_vector(
                target,
                scratch_vector(path, theta0),
                names,
                kind=item["kind"],
                theta0_sha256=theta0_sha,
                provenance=item["provenance"],
            )
            path.unlink()
    except BaseException:
        for target in written:
            target.unlink(missing_ok=True)
            vectors.record_path(target).unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    norms = np.sqrt(np.diag(G))
    files = []
    for item in planned:
        w = item["weights"]
        files.append(
            {
                "file": item["file"],
                "kind": item["kind"],
                "weights": w.tolist(),
                "norm_gram": math.sqrt(quadratic(G, w)),
                "provenance": item["provenance"],
                **{key: item["record"][key] for key in ("sha256", "norm", "count", "names_sha256")},
            }
        )
    by_file = {item["file"]: item for item in planned}
    mixture_rows = [
        {
            "k": k,
            "file": f"mixture_{k}.pt",
            **{
                key: by_file[f"mixture_{k}.pt"]["provenance"][key]
                for key in ("signs", "rng_seed", "scale", "unscaled_norm", "target_norm")
            },
        }
        for k in range(1, mixtures + 1)
    ]
    removals = []
    for seed in removal_seeds:
        i = seeds.index(seed)
        row: dict[str, Any] = {"seed": seed, "delta_norm": float(norms[i])}
        for part in ("mean", "mixture"):
            item = by_file[f"removal_{part}_{seed}.pt"]
            w = item["weights"]
            removed = w.copy()
            removed[i] -= 1.0  # w − e_i: the removed vector, negated
            result_norm = math.sqrt(quadratic(G, w))
            row[part] = {
                "file": item["file"],
                "removed_norm": math.sqrt(quadratic(G, removed)),
                "result_norm": result_norm,
                "cos_with_delta": float((G @ w)[i]) / (result_norm * float(norms[i])),
                **{
                    key: item["provenance"][key]
                    for key in ("left_out", "seeds", "signs", "rng_seed", "scale", "unscaled_norm")
                    if key in item["provenance"]
                },
            }
        removals.append(row)
    grad_numbers = None
    if grad is not None:
        grad_numbers = {
            "file": grad_r.name,
            "sha256": grad_record["sha256"],
            "norm": grad_record["norm"],
            "provenance": grad_record["provenance"],
            "per_run": run_dots.tolist(),
            **spread(run_dots),
            "mean_vector": float(vector_dots[0]),
            "vectors": {
                item["file"]: float(d) for item, d in zip(planned, vector_dots, strict=True)
            },
        }
    return {
        "kind": "study5-vectors",
        "n": len(seeds),
        "seeds": seeds,
        "lr": float(cfg.optimizer.lr),
        "steps": int(cfg.train.steps),
        "runs": [
            {
                "seed": item["seed"],
                "run_id": item["f_run"]["run_id"],
                "config_hash": item["f_run"]["config_hash"],
                "checkpoint": {
                    "name": Path(item["checkpoint"]["path"]).name,
                    "sha256": item["checkpoint"]["sha256"],
                    "step": item["checkpoint"]["step"],
                    "names_sha256": item["checkpoint"]["names_sha256"],
                },
                "displacement_norm_recorded": item["f_run"]["displacement_norm"],
                "displacement_norm": float(norms[j]),
            }
            for j, item in enumerate(runs)
        ],
        "skipped": skipped,
        "theta0_trainable_sha256": theta0_sha,
        "names_sha256": vectors.names_sha256(names),
        "count": count,
        "gram": G.tolist(),
        **numbers,
        "grad_r": grad_numbers,
        "loo_seeds": loo_seeds,
        "removal_seeds": removal_seeds,
        "mixtures": mixture_rows,
        "removals": removals,
        "vectors": files,
        "chunk_elements": CHUNK,
        "model": {
            "name": cfg.model.name,
            "revision": cfg.model.revision,
            "dtype": cfg.model.dtype,
            "attn_implementation": cfg.model.attn_implementation,
        },
        "prompt_manifest_sha256": env.prompt_manifest_sha256,
        "numerics": numerics,
        "device": DEVICE,
        "git": git_state(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def write_new(path: Path, record: dict) -> None:
    """Write ``record`` to ``path`` atomically; refuse an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} exists; a record is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    os.replace(partial, path)


def report(record: dict) -> None:
    """Numbers, ids and hashes only."""
    print(f"n {record['n']} lr {record['lr']:.6e} steps {record['steps']}")
    for item in record["runs"]:
        print(
            f"run {item['seed']} {item['run_id']} checkpoint {item['checkpoint']['sha256']} "
            f"norm {item['displacement_norm']:.9e} recorded "
            f"{item['displacement_norm_recorded']:.9e}"
        )
    for item in record["skipped"]:
        print(f"skipped {item['seed']} {item['run_id']}")
    print(f"theta0_trainable_sha256 {record['theta0_trainable_sha256']}")
    mean = record["mean"]
    print(
        f"norm_sq_noise_free {mean['norm_sq_noise_free']:.9e} norm_sq_mean {mean['norm_sq']:.9e} "
        f"mean_pairwise_cosine {mean['mean_pairwise_cosine']:.6f}"
    )
    for key in ("p", "cos"):
        s = record["leave_one_out"][f"{key}_spread"]
        cv = "none" if s["cv"] is None else f"{s['cv']:.6f}"
        print(f"leave_one_out {key} mean {s['mean']:.9e} sd {s['sd']:.9e} cv {cv}")
    halves = record["halves"]
    print(
        f"halves cosine {halves['cosine']:.6f} norms {halves['even']['norm']:.9e} "
        f"{halves['odd']['norm']:.9e}"
    )
    if record["grad_r"] is not None:
        g = record["grad_r"]
        print(
            f"grad_r {g['sha256']} per_run mean {g['mean']:.9e} se {g['se']:.9e} "
            f"mean_vector {g['mean_vector']:.9e}"
        )
    for item in record["vectors"]:
        print(f"vector {item['file']} {item['sha256']} norm {item['norm']:.9e}")


def chosen(given: list[int] | None, default: list[int], seeds: list[int], what: str) -> list[int]:
    """The seeds named (default ``default``), each one of the runs', in increasing order."""
    picked = sorted(set(default if given is None else given))
    missing = [seed for seed in picked if seed not in seeds]
    if missing:
        raise ValueError(f"{what}: seeds {missing} are not among the runs'")
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    given = parser.add_mutually_exclusive_group(required=True)
    given.add_argument("--runs", type=Path, nargs="+")
    given.add_argument("--runs-from", type=Path)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grad-r", type=Path)
    parser.add_argument("--loo-seeds", type=int, nargs="*")
    parser.add_argument("--removal-seeds", type=int, nargs="*")
    parser.add_argument("--mixtures", type=int, default=MIXTURES)
    parser.add_argument("--skip-bad", action="store_true")
    args = parser.parse_args(argv)
    record_path = args.out_dir / RECORD
    if record_path.exists():
        print(f"record exists: {record_path}")
        return 1
    try:
        if args.mixtures < 0:
            raise ValueError(f"--mixtures {args.mixtures}: not a count")
        run_dirs = args.runs or [
            Path(line.strip()) for line in args.runs_from.read_text().splitlines() if line.strip()
        ]
        cfg = load_run_config(args.config, args.overrides)  # the protocol: fp32, frozen modules
        linear_response.require_frozen_adam(cfg)
        ladder.require_fresh(cfg)
        runs, skipped = read_runs(
            args.config, args.overrides, run_dirs, args.checkpoints, skip_bad=args.skip_bad
        )
        seeds = [item["seed"] for item in runs]
        loo_seeds = chosen(args.loo_seeds, seeds, seeds, "--loo-seeds")
        removal_seeds = chosen(
            args.removal_seeds, [s for s in REMOVAL_SEEDS if s in seeds], seeds, "--removal-seeds"
        )
        if len(seeds) < 4:
            raise ValueError(f"{len(seeds)} runs: at least 4 are needed")
        if (args.mixtures or removal_seeds) and len(seeds) % 2:
            raise ValueError(f"{len(seeds)} runs: a balanced-sign mixture needs an even number")
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    numerics = configure_numerics(cfg, DEVICE)  # first: before any CUDA work (none here)
    env = environment(cfg, DEVICE)
    try:
        record = run(
            cfg,
            env,
            runs,
            skipped,
            out_dir=args.out_dir,
            loo_seeds=loo_seeds,
            removal_seeds=removal_seeds,
            mixtures=args.mixtures,
            grad_r=args.grad_r,
            numerics=numerics,
        )
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    write_new(record_path, record)
    report(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
