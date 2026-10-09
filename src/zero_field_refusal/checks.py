"""Stage-0 checks of the real pipeline on the pinned checkpoints (docs/plan.md, stage 0).

One run directory collects:
- the fraction of θ₀ responses that end within the 96-token cap;
- the teacher–student gap at step 0 on a full training stream (invariant 1);
- the mean-gradient scaling and the token-count field at θ₀ (invariant 9, A1);
- the generation-path vs training-path discrepancy and the soft-target gradient ratio
  (invariant 8);
- the custom Adam against ``torch.optim.AdamW`` on the real parameters, stepping on identical
  gradients so that only optimizer arithmetic is compared;
- peak memory and time of a training step.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from omegaconf import DictConfig, OmegaConf

from .batches import chunks
from .diagnostics import field_measurement, path_discrepancy, soft_target_gradient_ratio
from .models import freeze
from .optim import Adam
from .runtime import (
    Environment,
    generate_stream,
    load_distributions,
    load_stream_for,
    locate_stream,
)
from .teacher import StreamError
from .tracking import Run
from .train import accumulate_gradient, teacher_gap, training_batches

# Seed of the fresh θ₀ batches of the field measurement, outside the run seeds 0–39.
FIELD_SEED = 1000


def run_stage0_checks(
    cfg: DictConfig, env: Environment, *, log: Callable[[str], None] = print
) -> Run:
    with Run(cfg) as run:
        streams = _ensure_streams(cfg, env, log)
        model = env.new_model()
        trainable = freeze(model, cfg.train.freeze)
        params = [p for _, p in trainable]

        stream, manifest = load_stream_for(cfg, env, "training", 0)
        run.summarize({"eos_fraction_at_cap": manifest["ended_fraction"],
                       "mean_response_tokens": manifest["mean_response_tokens"]})  # fmt: skip
        log("teacher-student gap on the training stream")
        run.summarize({"teacher_gap": teacher_gap(model, stream, cfg, env)})

        log("training-step memory and time")
        run.summarize({"step_profile": _step_profile(model, training_batches(stream, cfg)[0],
                                                     cfg, env)})  # fmt: skip

        log("field at theta_0")
        field_stream, _ = load_stream_for(cfg, env, "preconditioner", FIELD_SEED)
        field = field_measurement(
            model, params, chunks(field_stream, cfg.train.batch_size), cfg, pad_id=env.pad_id,
            device=env.device, steps=cfg.train.steps,
        )  # fmt: skip
        run.summarize({"field": field})

        log("generation path vs training path")
        one_batch = streams["soft_targets"]
        samples, _ = load_stream_for(one_batch, env, "soft_targets", 0)
        _, path, _ = locate_stream(one_batch, env, "soft_targets", 0)
        recorded = load_distributions(path)
        run.summarize({
            "path_discrepancy": path_discrepancy(
                model, samples, recorded, pad_id=env.pad_id,
                micro_batch_size=cfg.execution.micro_batch_size, device=env.device),
            "soft_target_gradient": soft_target_gradient_ratio(
                model, params, samples, recorded, one_batch, pad_id=env.pad_id,
                device=env.device),
        })  # fmt: skip

        log("custom Adam against torch.optim.AdamW")
        run.summarize({"adam_vs_torch": _compare_adam(model, params, stream, cfg, env)})
        return run


def _ensure_streams(cfg: DictConfig, env: Environment, log) -> dict[str, DictConfig]:
    """Generate the streams the checks need, unless they exist."""
    one_batch = OmegaConf.merge(cfg, {"train": {"steps": 1}})  # 20 responses with distributions
    needed = [(cfg, "training", 0), (cfg, "preconditioner", FIELD_SEED),
              (one_batch, "soft_targets", 0)]  # fmt: skip
    model = None
    for config, kind, seed in needed:
        _, path, _ = locate_stream(config, env, kind, seed)
        if path.exists():
            continue
        log(f"generating {kind} stream (seed {seed})")
        model = model if model is not None else env.new_model()
        generate_stream(config, env, kind, seed, model=model,
                        batch_size=cfg.execution.generation_batch_size)  # fmt: skip
    del model
    return {"soft_targets": one_batch}


def _step_profile(model, batch, cfg: DictConfig, env: Environment) -> dict:
    cuda = env.device.type == "cuda"
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    accumulate_gradient(model, batch, cfg, env)
    if cuda:
        torch.cuda.synchronize()
    out = {
        "seconds": time.perf_counter() - start,
        "micro_batch_size": cfg.execution.micro_batch_size,
    }
    if cuda:
        out["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
    for p in model.parameters():
        p.grad = None
    return out


def _compare_adam(model, params, stream, cfg: DictConfig, env: Environment, steps: int = 5) -> dict:
    """The custom Adam on the model, torch AdamW on a shadow copy; both on the same gradients."""
    lr, betas, eps = 5e-5, (0.9, 0.999), 1e-8
    start = [p.detach().clone() for p in params]
    shadow = [torch.nn.Parameter(p.detach().clone()) for p in params]
    ours = Adam(params, lr=lr, betas=betas, eps=eps)
    reference = torch.optim.AdamW(shadow, lr=lr, betas=betas, eps=eps, weight_decay=0.0,
                                  foreach=False)  # fmt: skip
    for batch in training_batches(stream, cfg)[:steps]:
        for p in params:
            p.grad = None
        accumulate_gradient(model, batch, cfg, env)
        for p, s in zip(params, shadow, strict=True):
            s.grad = p.grad  # shared, read-only for both optimizers
        ours.step()
        reference.step()
    diff = disp = 0.0
    max_abs = 0.0
    for p, s, p0 in zip(params, shadow, start, strict=True):
        d = (p.detach() - s.detach()).double()
        diff += d.pow(2).sum().item()
        disp += (p.detach().double() - p0.double()).pow(2).sum().item()
        max_abs = max(max_abs, d.abs().max().item())
    with torch.no_grad():  # leave the model at θ₀
        for p, p0 in zip(params, start, strict=True):
            p.copy_(p0)
            p.grad = None
    if disp == 0:
        raise StreamError("Adam comparison: the parameters did not move")
    return {"steps": steps, "max_abs_diff": max_abs, "relative_diff": (diff / disp) ** 0.5}
