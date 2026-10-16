"""Stage-0 checks of the real pipeline on the pinned checkpoints (docs/plan.md, stage 0).

One run directory collects:
- the fraction of θ₀ responses that end within the 96-token cap;
- the teacher–student gap at step 0 on a full training stream (invariant 1);
- the mean-gradient scaling and the token-count field at θ₀ (invariant 9, A1);
- the residual field of the generation-vs-training path discrepancy over several batches: the
  χ² bound that decides (invariant 8), gradient estimates in the Euclidean and Adam metrics, the
  per-token KL and a one-batch gradient ratio; the verdict is written as the environment's
  path-field record, which training runs require;
- the custom Adam against ``torch.optim.AdamW`` on the real parameters, stepping on identical
  gradients so that only optimizer arithmetic is compared;
- peak memory and time of the run configuration (student, θ₀ copy, Adam, a greedy batch).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import torch
from omegaconf import DictConfig, OmegaConf

from .batches import chunks
from .diagnostics import (
    field_measurement,
    path_chi_square,
    path_discrepancy,
    path_field_measurement,
    soft_target_gradient_ratio,
)
from .evaluation import greedy_responses
from .models import freeze
from .optim import Adam, estimate_second_moment
from .prechecks import environment_fingerprint, write_check_record
from .runtime import (
    Environment,
    generate_stream,
    load_stream_for,
    locate_stream,
)
from .teacher import load_distributions
from .tracking import Run
from .train import accumulate_gradient, teacher_gap, training_batches

# Seed of the fresh θ₀ batches of the field measurement, outside the run seeds 0–39.
FIELD_SEED = 1000
# Batches of that stream for the θ₀ second-moment estimate of Adam's metric (as in H2a).
ADAM_ESTIMATE_BATCHES = 50
# The teacher-gap thresholds: GAP_MARGIN × the largest gap over the training streams of these
# seeds (docs/preregistration.md, "Teacher-gap thresholds").
GAP_SEEDS = (0, 1, 2)
GAP_MARGIN = 10.0


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
        log("teacher-student gap on the training streams of seeds 0-2")
        gaps = {}
        for seed in GAP_SEEDS:
            seed_stream, _ = load_stream_for(cfg, env, "training", seed)
            gaps[seed] = teacher_gap(model, seed_stream, cfg, env)[0]
        run.summarize({"teacher_gap": gaps, "teacher_gap_proposed_thresholds": {
            "token_max_abs": GAP_MARGIN * max(g["token_max_abs"] for g in gaps.values()),
            "response_mean_signed_abs": GAP_MARGIN * max(abs(g["response_mean_signed"])
                                                         for g in gaps.values()),
        }})  # fmt: skip

        log("memory and time of the run configuration")
        run.summarize({"run_profile": _run_profile(model, stream, cfg, env)})

        log("field at theta_0")
        field_stream, _ = load_stream_for(cfg, env, "preconditioner", FIELD_SEED)
        field = field_measurement(
            model, params, chunks(field_stream, cfg.train.batch_size), cfg, pad_id=env.pad_id,
            device=env.device, steps=cfg.train.steps,
        )  # fmt: skip
        run.summarize({"field": field})

        log("custom Adam against torch.optim.AdamW")
        run.summarize({"adam_vs_torch": _compare_adam(model, params, stream, cfg, env)})

        log("path field: generation path vs training path over several batches")
        path_cfg = streams["soft_targets"]
        samples, _ = load_stream_for(path_cfg, env, "soft_targets", 0)
        _, path, _ = locate_stream(path_cfg, env, "soft_targets", 0)
        recorded = load_distributions(path)
        steps = cfg.train.steps
        common = dict(pad_id=env.pad_id, device=env.device, steps=steps)
        bound = path_chi_square(model, samples, recorded, path_cfg, **common)
        run.path("path_field_chi_square.json").write_text(
            json.dumps({"per_response": bound.pop("per_response")}) + "\n"
        )
        passed = bound["bound_upper"] <= cfg.checks.path_field.max_drift_to_noise
        certified = {"criterion": cfg.checks.path_field.max_drift_to_noise, "steps": steps,
                     "batch_size": cfg.train.batch_size, "cap": cfg.train.max_response_tokens,
                     "chi_square_bound": bound}  # fmt: skip
        # The verdict first: the diagnostics below must not be able to leave training locked.
        record = write_check_record(env.data_root, environment_fingerprint(cfg, env.device),
                                    certified, passed=passed, run_id=run.run_id)  # fmt: skip
        run.summarize({"path_field": certified | {"passed": passed}})
        run.record_inputs({"path_check_record": str(record.relative_to(env.data_root))})

        size = cfg.train.batch_size
        euclid = path_field_measurement(model, params, samples, recorded, path_cfg, **common)
        v_hat = [v.cpu() for v in estimate_second_moment(  # on the CPU: see path_field_measurement
            params, lambda i: accumulate_gradient(model, field_stream[i * size : (i + 1) * size],
                                                  cfg, env),
            min(ADAM_ESTIMATE_BATCHES, len(field_stream) // size),
        )]  # fmt: skip
        adam = path_field_measurement(model, params, samples, recorded, path_cfg,
                                      preconditioner=v_hat, **common)  # fmt: skip
        del v_hat
        run.summarize({"path_field": certified | {"passed": passed} | {
            "euclidean": None if euclid is None else euclid.as_dict(steps),
            "adam": None if adam is None else adam.as_dict(steps),
            "path_discrepancy": path_discrepancy(
                model, samples, recorded, pad_id=env.pad_id,
                micro_batch_size=cfg.execution.micro_batch_size, device=env.device),
            "soft_target_gradient_one_batch": soft_target_gradient_ratio(
                model, params, samples[:size], recorded[:size], path_cfg, pad_id=env.pad_id,
                device=env.device),
        }})  # fmt: skip
        return run


def path_field_config(cfg: DictConfig) -> DictConfig:
    """The config of the path-field check's soft-target stream (seed 0): the run's, with the
    check's batches as its steps and their responses."""
    # The loaded config is resolved: set the response count with the steps, or it stays at the
    # training stream's 500 (≈ 29 GB of distributions).
    batches = cfg.checks.path_field.batches
    responses = batches * cfg.train.batch_size
    return OmegaConf.merge(cfg, {"train": {"steps": batches}, "teacher": {"responses": responses}})


def _ensure_streams(cfg: DictConfig, env: Environment, log) -> dict[str, DictConfig]:
    """Generate the streams the checks need, unless they exist."""
    path_cfg = path_field_config(cfg)
    needed = [*((cfg, "training", seed) for seed in GAP_SEEDS),
              (cfg, "preconditioner", FIELD_SEED), (path_cfg, "soft_targets", 0)]  # fmt: skip
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
    return {"soft_targets": path_cfg}


def _run_profile(model, stream, cfg: DictConfig, env: Environment) -> dict:
    """Peak memory and times of a run's configuration: student, frozen θ₀ copy, Adam state, one
    training step and one greedy batch of the evaluation length. Leaves the model at θ₀."""
    cuda = env.device.type == "cuda"
    reference = env.new_model()
    reference.requires_grad_(False)
    params = [p for _, p in freeze(model, cfg.train.freeze)]
    optimizer = Adam(params, lr=5e-5)
    batch = training_batches(stream, cfg)[0]
    accumulate_gradient(model, batch, cfg, env)  # warm-up, not timed
    for p in params:
        p.grad = None
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    accumulate_gradient(model, batch, cfg, env)
    optimizer.step()
    if cuda:
        torch.cuda.synchronize()
    step_seconds = time.perf_counter() - start
    del optimizer
    for p in params:
        p.grad = None
    items = env.prompt_set(cfg.eval.harmful.prompt_set)[: cfg.execution.generation_batch_size]
    prompts = [env.encode_prompt(item["text"]) for item in items]
    start = time.perf_counter()
    responses = greedy_responses(model, prompts, end_ids=env.end_ids, pad_id=env.pad_id,
                                 max_new_tokens=cfg.eval.harmful.max_new_tokens,
                                 batch_size=cfg.execution.generation_batch_size)  # fmt: skip
    if cuda:
        torch.cuda.synchronize()
    out = {
        "training_step_seconds": step_seconds,
        "greedy_batch_seconds": time.perf_counter() - start,
        "greedy_batch_longest_response": max(len(ids) for ids, _ in responses),
        "micro_batch_size": cfg.execution.micro_batch_size,
        "generation_batch_size": cfg.execution.generation_batch_size,
    }
    if cuda:
        out["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 2**30
        out["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
    with torch.no_grad():  # back to θ₀
        theta0 = dict(reference.named_parameters())
        for name, p in model.named_parameters():
            p.copy_(theta0[name])
    return out


def _compare_adam(model, params, stream, cfg: DictConfig, env: Environment, steps: int = 5) -> dict:
    """The custom Adam against torch AdamW on identical gradients, on every fourth tensor.

    Adam is elementwise and has no shape-dependent path, so a subset of tensors checks the
    arithmetic while keeping the extra state small enough for 48 GB. Leaves the model at θ₀.
    """
    lr, betas, eps = 5e-5, (0.9, 0.999), 1e-8
    subset = params[::4]
    start = [p.detach().clone() for p in subset]
    shadow = [torch.nn.Parameter(p.detach().clone()) for p in subset]
    ours = Adam(subset, lr=lr, betas=betas, eps=eps)
    reference = torch.optim.AdamW(shadow, lr=lr, betas=betas, eps=eps, weight_decay=0.0,
                                  foreach=False)  # fmt: skip
    for batch in training_batches(stream, cfg)[:steps]:
        for p in params:
            p.grad = None
        accumulate_gradient(model, batch, cfg, env)
        for p, s in zip(subset, shadow, strict=True):
            s.grad = p.grad  # shared, read-only for both optimizers
        ours.step()
        reference.step()
    diff = disp = max_abs = 0.0
    for p, s, p0 in zip(subset, shadow, start, strict=True):
        d = (p.detach() - s.detach()).double()
        diff += d.pow(2).sum().item()
        disp += (p.detach().double() - p0.double()).pow(2).sum().item()
        max_abs = max(max_abs, d.abs().max().item())
    with torch.no_grad():
        for p, p0 in zip(subset, start, strict=True):
            p.copy_(p0)
    for p in params:
        p.grad = None
    if disp == 0:
        raise RuntimeError("Adam comparison: the parameters did not move")
    return {"steps": steps, "tensors": len(subset), "elements": sum(p.numel() for p in subset),
            "max_abs_diff": max_abs, "relative_diff": (diff / disp) ** 0.5}  # fmt: skip
