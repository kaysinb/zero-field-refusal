"""Study 5's small-amplitude surrogate and token-level readouts along one vector v of the trainable
set (docs/study5-preregistration.md, "Extension 2", M3), written as one JSON record.

    python scripts/study5_tokens.py CONFIG [key=value ...] (--vector FILE | --f-run RUN_DIR
        --checkpoints DIR) --scales A [A ...] --out RECORD.json [--surrogate SPEC] [--rare P]
        [--panel-size N] [--no-tokens]

CONFIG and the overrides (written before the options) are those of the seed's F run: they name the
model, the trainable set (``train.freeze``), the K panel's stream and the micro-batches. The
scales are plain decimals (``-0.1``, not ``-1e-1``) and distinct. v is either

- a vector file (``vectors.load_vector``: the trainable set's names in order, its bytes against its
  recorded sha256, built on this θ₀), or
- an F run's displacement Δθ = θ_N − θ₀: the run of this config (``ladder.read_f_run``: its config
  hash, all its updates; arm F, ``linear_response.require_frozen_adam``), its endpoint checkpoint
  ``DIR/<run id>.pt`` against the recorded sha256 and step (``line_scan.checkpoint_of``) with
  exactly the trainable set's tensors (``line_scan.endpoint_state``), θ₀'s trainable hash the
  run's, and Δθ by ``vectors.displacement``, as a displacement file holds it: both inputs give the
  same v bit for bit.

The point θ(a) = θ₀ + a·v of each scale is set in fp32 by ``evaluation.set_point`` from θ₀ (a
frozen reference model, never written); v stays on the host and goes to the device one tensor at a
time; the frozen parameters stay θ₀'s. At θ₀ and at every point:

- **The surrogate's pieces.** ``surrogate.continuation_logprobs`` on the harmful and on the benign
  panel of SPEC (default ``configs/study5/surrogate-2.yaml``; each must hold --panel-size prompts,
  default 100), in fp32 at the micro-batch of the checks (``execution.eval_batch_size``). Per
  panel, the mean over prompts of logmeanexp over R (``R``), of logmeanexp over C (``C``) and of
  their difference (``difference``). The harmful panel's ``difference`` is r, by the arithmetic of
  ``surrogate.surrogate_value``.
- **Token level** (unless --no-tokens), on the K_benign scoring panel's θ₀ responses as
  ``train.Evaluator`` reads them (the ``kl_benign`` stream, cut to ``eval.kl.limit`` prompts),
  batched, padded and masked as ``evaluation.sequence_kl`` (``execution.micro_batch_size``; every
  response token, a sampled end token included). At each response position, with p₀ and p the
  next-token distributions of θ₀ and θ(a), float64 log-softmaxes of the fp32 logits: the entropy
  −Σ_v p(v) ln p(v); the rare mass Σ_{v: p₀(v) < P} p(v) (--rare P, default 10⁻³); and
  ln p(y) − ln p₀(y) for the response's own token y. Each is a mean over all the panel's response
  tokens. θ₀'s entropy and rare mass (p = p₀) are recorded once, from the first point's pass.

**Memory.** Two models stay on the device: θ₀ (the reference) and θ(a). Each point's parameters are
set once; then every micro-batch goes through both. θ₀'s distribution gives the rare mask, its own
entropy and rare mass and ln p₀(y), and is freed before θ(a)'s is computed. Only per-position
scalars leave a micro-batch: no (positions × vocabulary) array of the panel is kept, and θ₀'s
forward pass is repeated at each point instead of keeping its masks. At the peak: the two models
and one micro-batch's logits with a few float64 copies, as in ``sequence_kl``.

The record holds v's identity (the file's name, sha256, kind, norm, count and provenance, or the
run's id and config hash, its checkpoint's sha256 and step and the endpoint's trainable hash, with
v's norm and count), θ₀'s trainable hash, the surrogate's sha256, the panels' names, counts and id
hashes, the K panel's stream identity and size, θ₀'s values, each point's a, r, pieces and token
means and count, and for each pair ±a (a > 0) the odd part [r(a) − r(−a)]/2 and the even part
[r(a) + r(−a)]/2 − r(θ₀). After the last point θ(a) is set back to θ₀ and its trainable hash
recorded (``theta0_restored_sha256``: equal to θ₀'s when the reference stayed unwritten). Then the
model, the numerics in force, the device's peak memory, the git state and the time. stdout carries
numbers, ids and hashes only, never a prompt, a response or a continuation. An existing record is
never overwritten. Exits 0 when written, 1 on a refusal or an error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel

from zero_field_refusal import ladder, line_scan, linear_response, surrogate, vectors
from zero_field_refusal.batches import chunks, collate
from zero_field_refusal.config import config_hash
from zero_field_refusal.evaluation import NO_CAP, set_point
from zero_field_refusal.losses import response_logits, targets
from zero_field_refusal.models import freeze
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import (
    configure_numerics,
    device_name,
    environment,
    load_stream_for,
)
from zero_field_refusal.teacher import Sample, StreamError
from zero_field_refusal.tensor_hash import tensors_sha256
from zero_field_refusal.tracking import git_state

RARE = 1e-3  # the registered threshold of θ₀'s probability below which a token is rare


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def _mean(values: torch.Tensor) -> float:
    return math.fsum(values.tolist()) / values.numel()


def checked_scales(scales: Sequence[float]) -> list[float]:
    """The scales as given; ``ValueError`` unless finite and distinct."""
    scales = [float(a) for a in scales]
    if not all(math.isfinite(a) for a in scales) or len(set(scales)) != len(scales):
        raise ValueError(f"the scales must be finite and distinct, got {scales}")
    return scales


# ------------------------------------------------------------------------------------ inputs


def panels_of(env, spec: surrogate.SurrogateSpec, panel_size: int = 100) -> dict:
    """The surrogate's two panels, role → (name, items); each must hold ``panel_size`` prompts
    (the registration's 100)."""
    panels = {}
    for role, name in (("harmful", spec.harmful_panel), ("benign", spec.benign_panel)):
        items = env.prompt_set(name)
        if len(items) != panel_size:
            raise ValueError(f"panel {name} has {len(items)} prompts, not {panel_size}")
        panels[role] = (name, items)
    return panels


def k_panel(cfg, env) -> tuple[list[Sample], dict[str, Any]]:
    """The K_benign scoring panel's θ₀ responses as ``train.Evaluator`` reads them (the
    ``kl_benign`` stream, its first ``eval.kl.limit`` prompts' responses when a limit is set),
    and the stream's identity as a run records it."""
    samples, manifest = load_stream_for(cfg, env, "kl_benign", None)
    limit, per_prompt = cfg.eval.kl.limit, cfg.eval.kl.samples_per_prompt
    chosen = samples[: limit * per_prompt] if limit else samples
    return chosen, {key: manifest[key] for key in ("path", "sha256", "count")}


def direction(
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    theta0: Mapping[str, torch.Tensor],
    theta0_sha: str,
    *,
    vector: Path | None = None,
    f_run: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    """v on the host, by name, and its identity: the vector file, or the run's Δθ; ``ValueError``
    for one of another θ₀ or not of the trainable set."""
    names = [name for name, _ in trainable]
    if vector is not None:
        v, record = vectors.load_vector(vector, names, theta0_sha256=theta0_sha, mmap=True)
        return v, {"source": "vector", "file": Path(vector).name,
                   **{key: record[key] for key in ("sha256", "kind", "norm", "count",
                                                   "names_sha256", "theta0_sha256",
                                                   "provenance")}}  # fmt: skip
    if theta0_sha != f_run["theta0_trainable_sha256"]:
        raise ValueError(f"θ₀'s trainable tensors hash to {theta0_sha}, not the run's "
                         f"{f_run['theta0_trainable_sha256']}")  # fmt: skip
    state = line_scan.endpoint_state(checkpoint, list(trainable))
    endpoint_sha = tensors_sha256([(name, state[name]) for name in names])
    v = vectors.displacement(state, theta0, names)
    del state
    return v, {"source": "f_run", "run_id": f_run["run_id"], "config_hash": f_run["config_hash"],
               "checkpoint": {"name": Path(checkpoint["path"]).name,
                              "sha256": checkpoint["sha256"], "step": checkpoint["step"]},
               "endpoint_trainable_sha256": endpoint_sha, "norm": vectors.norm(v),
               "count": sum(t.numel() for t in v.values())}  # fmt: skip


@torch.no_grad()
def place(
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    start: Sequence[torch.Tensor],
    v: Mapping[str, torch.Tensor],
    a: float,
) -> None:
    """θ₀ + a·v by ``evaluation.set_point``, one tensor at a time: v's tensor goes to the device
    only while its parameter is set (a = 0 copies θ₀ exactly)."""
    for (name, p), s in zip(trainable, start, strict=True):
        set_point([(name, p)], [s], [v[name].to(s.device)], a)


# ---------------------------------------------------------------------------- the surrogate


def surrogate_pieces(
    model: PreTrainedModel, env, spec: surrogate.SurrogateSpec, items: Sequence[dict], micro: int
) -> dict[str, float]:
    """One panel's means over prompts: logmeanexp over R (``R``), over C (``C``) and their
    difference (``difference``; on the harmful panel, r as ``surrogate.surrogate_value``)."""
    logp = surrogate.continuation_logprobs(
        model, env.tokenizer, [item["text"] for item in items], spec.continuations,
        device=env.device, micro_batch_size=micro, max_tokens=spec.max_tokens,
    )  # fmt: skip
    k = len(spec.refusals)
    return {"R": _mean(surrogate.logmeanexp(logp[:, :k])),
            "C": _mean(surrogate.logmeanexp(logp[:, k:])),
            "difference": _mean(surrogate.contrasts(logp, k))}  # fmt: skip


# --------------------------------------------------------------------------------- the tokens


def entropy_and_mass(log_p: torch.Tensor, rare: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per position (rows of ``log_p``, float64 log-probabilities over the vocabulary): the
    entropy −Σ p ln p and the mass on the vocabulary entries ``rare`` marks."""
    p = log_p.exp()
    entropy = -(p * log_p).sum(-1)
    mass = torch.where(rare, p, 0.0).sum(-1)
    return entropy, mass


def reference_readouts(
    log_p0: torch.Tensor, y: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """From θ₀'s float64 log-probabilities at the positions: the rare mask (p₀ < ``threshold``),
    θ₀'s entropy and rare mass, and ln p₀ of each position's own token ``y``."""
    rare = log_p0.exp() < threshold
    entropy, mass = entropy_and_mass(log_p0, rare)
    return rare, entropy, mass, log_p0.gather(1, y[:, None])[:, 0]


@torch.no_grad()
def token_statistics(
    student: PreTrainedModel,
    reference: PreTrainedModel,
    samples: Sequence[Sample],
    *,
    pad_id: int,
    micro_batch_size: int,
    rare: float = RARE,
) -> dict[str, Any]:
    """Means over the samples' response tokens: θ(a)'s entropy, its rare mass and the change of
    the own token's log-probability, and θ₀'s entropy and rare mass (module docstring); micro-batch
    by micro-batch as ``evaluation.sequence_kl`` batches them."""
    device = next(student.parameters()).device
    keys = ("entropy", "rare_mass", "delta_logprob", "theta0_entropy", "theta0_rare_mass")
    parts: dict[str, list[torch.Tensor]] = {key: [] for key in keys}
    for chunk in chunks(samples, micro_batch_size):
        batch = collate(chunk, pad_id=pad_id, max_response_tokens=NO_CAP, device=device)
        y = targets(batch)
        log_p0 = torch.log_softmax(response_logits(reference, batch).double(), dim=-1)
        mask, entropy0, mass0, own0 = reference_readouts(log_p0, y, rare)
        del log_p0
        log_p = torch.log_softmax(response_logits(student, batch).double(), dim=-1)
        entropy, mass = entropy_and_mass(log_p, mask)
        delta = log_p.gather(1, y[:, None])[:, 0] - own0
        del log_p, mask
        for key, t in zip(keys, (entropy, mass, delta, entropy0, mass0), strict=True):
            parts[key].append(t.cpu())
    values = {key: torch.cat(t) for key, t in parts.items()}
    return {"count": values["entropy"].numel(), **{key: _mean(t) for key, t in values.items()}}


# ------------------------------------------------------------------------------------- the run


def odd_even(points: Sequence[Mapping[str, Any]], r0: float) -> list[dict[str, float]]:
    """For each a > 0 whose −a is a point too, in the points' order: the odd part
    [r(a) − r(−a)]/2 and the even part [r(a) + r(−a)]/2 − r(θ₀)."""
    r = {point["a"]: point["r"] for point in points}
    return [{"a": a, "odd": (r[a] - r[-a]) / 2, "even": (r[a] + r[-a]) / 2 - r0}
            for a in r if a > 0 and -a in r]  # fmt: skip


def _peak_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3)}  # fmt: skip


def run(
    cfg,
    env,
    spec: surrogate.SurrogateSpec,
    numerics: dict,
    panels: dict,
    scales: Sequence[float],
    *,
    k_samples: tuple[list[Sample], dict[str, Any]] | None,
    rare: float = RARE,
    vector: Path | None = None,
    f_run: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The record (module docstring) in ``env``'s model: v from ``vector`` or from ``f_run`` and
    its ``checkpoint``; ``panels`` from ``panels_of``; ``k_samples`` from ``k_panel``, or None
    without token readouts. ``ValueError`` for a v of another θ₀ or trainable set."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    reference = env.new_model()
    reference.requires_grad_(False)
    reference.eval()
    student = env.new_model()
    student.eval()
    trainable = freeze(student, cfg.train.freeze)
    names = [name for name, _ in trainable]
    theta0 = dict(reference.named_parameters())
    start = [theta0[name] for name in names]
    theta0_sha = tensors_sha256(list(zip(names, start, strict=True)))
    v, identity = direction(trainable, theta0, theta0_sha, vector=vector, f_run=f_run,
                            checkpoint=checkpoint)  # fmt: skip
    surrogate_micro = int(cfg.execution.eval_batch_size)
    token_micro = int(cfg.execution.micro_batch_size)

    def surrogate_at(a: float) -> dict[str, Any]:
        place(trainable, start, v, a)
        pieces = {role: surrogate_pieces(student, env, spec, items, surrogate_micro)
                  for role, (_, items) in panels.items()}  # fmt: skip
        return {"r": pieces["harmful"]["difference"], "surrogate": pieces}

    at_theta0: dict[str, Any] = {**surrogate_at(0.0), "tokens": None}
    points = []
    for a in scales:
        point: dict[str, Any] = {"a": a, **surrogate_at(a), "tokens": None}
        if k_samples is not None:  # θ(a) is in place
            stats = token_statistics(student, reference, k_samples[0], pad_id=env.pad_id,
                                     micro_batch_size=token_micro, rare=rare)  # fmt: skip
            point["tokens"] = {
                key: stats[key] for key in ("entropy", "rare_mass", "delta_logprob", "count")
            }
            if at_theta0["tokens"] is None:
                at_theta0["tokens"] = {"entropy": stats["theta0_entropy"],
                                       "rare_mass": stats["theta0_rare_mass"],
                                       "count": stats["count"]}  # fmt: skip
        points.append(point)
    place(trainable, start, v, 0.0)
    restored = tensors_sha256(trainable)
    k_record = None
    if k_samples is not None:
        samples, stream = k_samples
        k_record = {"stream": stream, "samples": len(samples),
                    "prompts": len({s.prompt_id for s in samples}),
                    "tokens": sum(len(s.response_ids) for s in samples),
                    "micro_batch_size": token_micro, "rare": rare}  # fmt: skip
    record_panels = {}
    for role, (name, items) in panels.items():
        ids = [item["id"] for item in items]
        record_panels[role] = {"name": name, "count": len(ids), "ids_sha256": ids_sha256(ids)}
    return {
        "kind": "study5-tokens",
        "seed": int(cfg.seed),
        "lr": float(cfg.optimizer.lr),
        "steps": int(cfg.train.steps),
        "config_hash": config_hash(cfg),
        "vector": identity,
        "theta0_trainable_sha256": theta0_sha,
        "names_sha256": vectors.names_sha256(names),
        "scales": list(scales),
        "theta0": at_theta0,
        "points": points,
        "pairs": odd_even(points, at_theta0["r"]),
        "theta0_restored_sha256": restored,
        "surrogate_sha256": spec.sha256,
        "surrogate_version": spec.version,
        "surrogate_micro_batch_size": surrogate_micro,
        "panels": record_panels,
        "k_panel": k_record,
        "model": {"name": cfg.model.name, "revision": cfg.model.revision,
                  "dtype": cfg.model.dtype, "attn_implementation": cfg.model.attn_implementation},
        "prompt_manifest_sha256": env.prompt_manifest_sha256,
        "numerics": numerics,
        "device": device_name(torch.device(env.device)),
        "memory": _peak_memory(),
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


def _tokens(tokens: Mapping[str, Any] | None) -> str:
    if tokens is None:
        return ""
    out = f" entropy {tokens['entropy']:.9e} rare_mass {tokens['rare_mass']:.9e}"
    if "delta_logprob" in tokens:
        out += f" delta_logprob {tokens['delta_logprob']:.9e}"
    return out + f" tokens {tokens['count']}"


def _surrogate(values: Mapping[str, Any]) -> str:
    h, b = values["surrogate"]["harmful"], values["surrogate"]["benign"]
    return (f"r {values['r']:.9e} R {h['R']:.9e} C {h['C']:.9e} r_benign {b['difference']:.9e} "
            f"R_benign {b['R']:.9e} C_benign {b['C']:.9e}")  # fmt: skip


def report(record: dict) -> None:
    """Numbers, ids and hashes only."""
    vector = record["vector"]
    if vector["source"] == "vector":
        print(f"vector {vector['file']} sha256 {vector['sha256']} kind {vector['kind']}")
    else:
        print(f"run {vector['run_id']} checkpoint {vector['checkpoint']['sha256']} "
              f"step {vector['checkpoint']['step']}")  # fmt: skip
    print(f"norm {vector['norm']:.9e} count {vector['count']}")
    print(f"theta0_trainable_sha256 {record['theta0_trainable_sha256']}")
    print(f"surrogate_sha256 {record['surrogate_sha256']}")
    print(f"theta0 {_surrogate(record['theta0'])}{_tokens(record['theta0']['tokens'])}")
    for point in record["points"]:
        print(f"a {point['a']:+.6g} {_surrogate(point)}{_tokens(point['tokens'])}")
    for pair in record["pairs"]:
        print(f"pair {pair['a']:.6g} odd {pair['odd']:.9e} even {pair['even']:.9e}")
    print(f"theta0_restored_sha256 {record['theta0_restored_sha256']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--vector", type=Path)
    source.add_argument("--f-run", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--scales", type=float, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--surrogate", type=Path, default=surrogate.SPEC_PATH)
    parser.add_argument("--rare", type=float, default=RARE)
    parser.add_argument("--panel-size", type=int, default=100)
    parser.add_argument("--no-tokens", action="store_true")
    args = parser.parse_args(argv)
    if (args.f_run is None) != (args.checkpoints is None):
        parser.error("--checkpoints DIR goes with --f-run, and only with it")
    if args.out.exists():
        print(f"record exists: {args.out}")
        return 1
    try:
        scales = checked_scales(args.scales)
        if not 0 < args.rare < 1:
            raise ValueError(f"--rare {args.rare}: not a probability in (0, 1)")
    except ValueError as exc:
        print(exc)
        return 1
    cfg = load_run_config(args.config, args.overrides)  # the protocol: fp32, frozen modules
    f_run = checkpoint = None
    try:
        if args.f_run is not None:
            linear_response.require_frozen_adam(cfg)
            f_run = ladder.read_f_run(args.f_run, cfg)  # its config hash this config's
            checkpoint = line_scan.checkpoint_of(args.f_run, f_run, args.checkpoints,
                                                 int(cfg.train.steps))  # fmt: skip
        else:
            vectors.read_record(args.vector)  # the file's record: read before any model
            if not args.vector.is_file():
                raise ValueError(f"{args.vector}: no vector file")
    except (ValueError, OSError) as exc:
        print(exc)
        return 1
    numerics = configure_numerics(cfg, cfg.execution.device)  # first: before any CUDA work
    spec = surrogate.load_spec(args.surrogate)
    env = environment(cfg, cfg.execution.device)
    try:
        panels = panels_of(env, spec, args.panel_size)
        k_samples = None if args.no_tokens else k_panel(cfg, env)
        record = run(cfg, env, spec, numerics, panels, scales, k_samples=k_samples,
                     rare=args.rare, vector=args.vector, f_run=f_run,
                     checkpoint=checkpoint)  # fmt: skip
    except (ValueError, OSError, StreamError) as exc:
        print(exc)
        return 1
    write_new(args.out, record)
    report(record)
    return 0


if __name__ == "__main__":
    sys.exit(main())
