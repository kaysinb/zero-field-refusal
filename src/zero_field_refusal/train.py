"""One run: the stored teacher stream, N optimizer steps, then in-process evaluation.

The student starts from θ₀ and takes one step per batch of the stream in the order given by
``derive_seed(seed, "data_order")``, so every optimizer sees the same responses in the same order
and each response once (invariant 6). A frozen θ₀ copy stays on the device for K, the weight
geometry and the weight-space controls; weights are not saved by default.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel

from .batches import chunks, collate
from .data import factual_answer_correct, mmlu_prompt
from .evaluation import (
    choice_predictions,
    displacement,
    greedy_responses,
    random_direction,
    sequence_kl,
    sequence_nll,
    set_point,
)
from .losses import loss_denominator, response_logprobs, soft_kl, summed_nll, targets
from .models import freeze
from .optim import ADAM_MODES, build_optimizer, estimate_second_moment
from .prechecks import check_run_chi_square, check_teacher_gap, require_path_check
from .refusal import is_keyword_refusal
from .runtime import Environment, load_stream_for, locate_stream
from .seeding import derive_seed, seed_everything
from .teacher import Sample, load_distributions, reference_samples
from .tracking import Run

FACTUAL_MAX_NEW_TOKENS = 32
CHOICE_LETTERS = ("A", "B", "C", "D")


def stream_kind(cfg: DictConfig) -> str:
    """The stream a run trains on: teacher samples, or the stream its control requires."""
    return {"alpaca_reference": "alpaca_reference", "soft_targets": "soft_targets"}.get(
        cfg.control, "training"
    )


def training_batches(stream: Sequence[Sample], cfg: DictConfig) -> list[list[Sample]]:
    steps, size = cfg.train.steps, cfg.train.batch_size
    if len(stream) != steps * size:
        raise ValueError(f"The stream has {len(stream)} responses; the run needs {steps * size}")
    order = np.random.default_rng(derive_seed(cfg.seed, "data_order")).permutation(len(stream))
    return [[stream[i] for i in order[s * size : (s + 1) * size]] for s in range(steps)]


def accumulate_gradient(
    model: PreTrainedModel,
    samples: Sequence[Sample],
    cfg: DictConfig,
    env: Environment,
    soft_targets: Sequence[torch.Tensor] | None = None,
) -> dict[str, float]:
    """Add the batch gradient of loss = Σ loss terms / denominator to ``.grad``.

    With the constant denominator, micro-batch gradients add up exactly to the batch gradient.
    """
    cap = cfg.train.max_response_tokens
    denominator = loss_denominator(cfg.train.loss, samples, cap)
    loss = nll = 0.0
    tokens = 0
    micro = cfg.execution.micro_batch_size
    for start in range(0, len(samples), micro):
        chunk = samples[start : start + micro]
        batch = collate(chunk, pad_id=env.pad_id, max_response_tokens=cap, device=env.device)
        if soft_targets is None:
            term = summed_nll(model, batch)
        else:
            target = torch.cat([t[:cap] for t in soft_targets[start : start + micro]])
            term = soft_kl(model, batch, target.to(env.device))
        (term / denominator).backward()
        loss += term.item() / denominator
        tokens += batch.num_tokens
        if soft_targets is None:
            nll += term.item()
    stats = {"loss": loss, "denominator": denominator, "tokens": tokens}
    if soft_targets is None:
        stats["nll_per_token"] = nll / tokens
    return stats


@torch.no_grad()
def teacher_gap(
    model: PreTrainedModel, samples: Sequence[Sample], cfg: DictConfig, env: Environment
) -> tuple[dict[str, float], list[float]]:
    """Training-path log-probs at θ₀ against the stored teacher log-probs (invariant 1).

    Returns the gap statistics and, per sample, Σ_t expm1(Δ_t)² over its response tokens: an
    unbiased estimate of Σ_t χ²(training path ‖ generation path), with rounding entering squared.
    """
    cap = cfg.train.max_response_tokens
    per_token, per_response, chi_square = [], [], []
    for chunk in chunks(samples, cfg.execution.micro_batch_size):
        batch = collate(chunk, pad_id=env.pad_id, max_response_tokens=cap, device=env.device)
        logp = response_logprobs(model, batch).gather(1, targets(batch)[:, None])[:, 0]
        stored = torch.tensor(
            [lp for s in chunk for lp in s.logprobs[:cap]], dtype=torch.float64, device=env.device
        )
        diff = logp.double() - stored
        per_token.append(diff)
        per_response += [part.sum() for part in diff.split(batch.lengths)]
        chi_square += [torch.expm1(part).pow(2).sum().item() for part in diff.split(batch.lengths)]
    token = torch.cat(per_token)
    response = torch.stack(per_response)
    stats = {
        "token_max_abs": token.abs().max().item(),
        "response_mean_signed": response.mean().item(),
        "response_mean_abs": response.abs().mean().item(),
        "response_max_abs": response.abs().max().item(),
    }
    return stats, chi_square


def projection_directions(
    trainable: Sequence[tuple[str, torch.nn.Parameter]], count: int
) -> dict[str, list[torch.Tensor]]:
    """Fixed random directions with independent N(0, 1) entries, identical in every run.

    Each derives from the parameter name and full shape and is generated on the CPU, because CUDA
    random numbers of large tensors depend on the GPU model. They stay in pinned CPU memory
    (count × the trainable size) and are moved to the device one tensor at a time.
    """
    directions = {}
    for name, param in trainable:
        directions[name] = []
        for i in range(count):
            generator = torch.Generator().manual_seed(
                derive_seed(0, "projection", i, name, list(param.shape))
            )
            r = torch.randn(tuple(param.shape), generator=generator)
            directions[name].append(r.pin_memory() if torch.cuda.is_available() else r)
    return directions


@torch.no_grad()
def projection_stats(
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    theta0: Mapping[str, torch.Tensor],
    directions: Mapping[str, Sequence[torch.Tensor]],
) -> dict[str, Any]:
    """Gradient and displacement norms, and their projections on the fixed random directions.

    Differences of consecutive displacement projections are the projections of the updates.
    """
    count = len(next(iter(directions.values())))
    grad_sq = disp_sq = 0.0
    grad_proj, disp_proj = [0.0] * count, [0.0] * count
    for name, param in trainable:
        grad, delta = param.grad, param - theta0[name]
        grad_sq += grad.double().pow(2).sum().item()
        disp_sq += delta.double().pow(2).sum().item()
        for i, r in enumerate(directions[name]):
            r = r.to(param.device, non_blocking=True)
            grad_proj[i] += (grad * r).sum().item()
            disp_proj[i] += (delta * r).sum().item()
    return {
        "grad_norm": grad_sq**0.5,
        "displacement_norm": disp_sq**0.5,
        "grad_projection": grad_proj,
        "displacement_projection": disp_proj,
    }


class Evaluator:
    """Measurements of one model state ("condition"), written as per-item observations."""

    def __init__(self, cfg: DictConfig, env: Environment, run: Run, reference: PreTrainedModel):
        self.cfg, self.env, self.run, self.reference = cfg, env, run, reference
        kl_limit = cfg.eval.kl.limit
        per_prompt = cfg.eval.kl.samples_per_prompt
        self.kl_streams = {}
        for kind in ("kl_harmful", "kl_benign"):
            samples, manifest = load_stream_for(cfg, env, kind, None)
            self.kl_streams[kind] = samples[: kl_limit * per_prompt] if kl_limit else samples
            run.summarize({f"stream/{kind}": manifest["sha256"]})
            run.record_inputs({"streams": {kind: _identity(manifest)}})
        self.harmful = self._limited(cfg.eval.harmful.prompt_set, cfg.eval.harmful.limit)
        self.heldout = reference_samples(
            env.prompt_set("alpaca_heldout"), env.encode_prompt, env.encode_text,
            end_id=env.end_ids[0], cap=cfg.teacher.sampling.max_new_tokens,
        )  # fmt: skip
        self.factual = env.prompt_set("factual_qa")
        self.mmlu = env.prompt_set("mmlu")

    def _limited(self, name: str, limit: int | None) -> list[dict]:
        items = self.env.prompt_set(name)
        return items[:limit] if limit else items

    def __call__(self, model: PreTrainedModel, condition: str) -> dict[str, float]:
        cfg, env, execution = self.cfg, self.env, self.cfg.execution
        out: dict[str, float] = {}
        rows = []
        for kind, samples in self.kl_streams.items():
            kl = sequence_kl(model, self.reference, samples, pad_id=env.pad_id,
                             micro_batch_size=execution.micro_batch_size)  # fmt: skip
            out[f"K_{kind.removeprefix('kl_')}"] = kl.mean().item()
            rows += [
                {"condition": condition, "measure": kind, "item_id": s.prompt_id, "sample": i,
                 "value": v, "tokens": len(s.response_ids)}
                for i, (s, v) in enumerate(zip(samples, kl.tolist(), strict=True))
            ]  # fmt: skip

        responses = greedy_responses(
            model, [env.encode_prompt(item["text"]) for item in self.harmful],
            end_ids=env.end_ids, pad_id=env.pad_id,
            max_new_tokens=cfg.eval.harmful.max_new_tokens,
            batch_size=execution.generation_batch_size,
        )  # fmt: skip
        records = []
        for item, (ids, ended) in zip(self.harmful, responses, strict=True):
            text = env.decode(ids)
            refusal = is_keyword_refusal(text, cfg.eval.refusal_keywords)
            records.append({"condition": condition, "item_id": item["id"], "response_ids": ids,
                            "text": text, "ended": ended, "keyword_refusal": refusal})  # fmt: skip
        out["keyword_refusal_rate"] = float(np.mean([r["keyword_refusal"] for r in records]))
        _append_jsonl(self.run.path("responses", f"{cfg.eval.harmful.prompt_set}.jsonl"), records)

        utility = cfg.eval.utility
        if utility.heldout_reference_nll:
            nll, tokens = sequence_nll(model, self.heldout, pad_id=env.pad_id,
                                       micro_batch_size=execution.micro_batch_size)  # fmt: skip
            out["heldout_nll_per_token"] = (nll.sum() / tokens.sum()).item()
            rows += [
                {"condition": condition, "measure": "heldout_nll", "item_id": s.prompt_id,
                 "value": v, "tokens": n}
                for s, v, n in zip(self.heldout, nll.tolist(), tokens.tolist(), strict=True)
            ]  # fmt: skip
        if utility.factual_qa:
            answers = greedy_responses(
                model, [env.encode_prompt(item["text"]) for item in self.factual],
                end_ids=env.end_ids, pad_id=env.pad_id, max_new_tokens=FACTUAL_MAX_NEW_TOKENS,
                batch_size=execution.generation_batch_size,
            )  # fmt: skip
            correct = [
                factual_answer_correct(env.decode(ids), item["answers"])
                for item, (ids, _) in zip(self.factual, answers, strict=True)
            ]
            out["factual_accuracy"] = float(np.mean(correct))
            rows += [
                {"condition": condition, "measure": "factual_qa", "item_id": item["id"],
                 "value": float(ok), "text": env.decode(ids)}
                for item, (ids, _), ok in zip(self.factual, answers, correct, strict=True)
            ]  # fmt: skip
        if utility.benchmark:
            letters = [env.tokenizer.convert_tokens_to_ids(letter) for letter in CHOICE_LETTERS]
            if len(set(letters)) != 4 or env.tokenizer.unk_token_id in letters or None in letters:
                raise ValueError(f"Answer letters {CHOICE_LETTERS} are not distinct tokens")
            predicted = choice_predictions(
                model, [env.encode_prompt(mmlu_prompt(item)) for item in self.mmlu], letters,
                pad_id=env.pad_id, batch_size=execution.eval_batch_size,
            )  # fmt: skip
            correct = [p == item["answer"] for p, item in zip(predicted, self.mmlu, strict=True)]
            out["benchmark_accuracy"] = float(np.mean(correct))
            rows += [
                {"condition": condition, "measure": "benchmark", "item_id": item["id"],
                 "value": float(ok), "predicted": p}
                for item, p, ok in zip(self.mmlu, predicted, correct, strict=True)
            ]  # fmt: skip
        _append_jsonl(self.run.path("observations", "measurements.jsonl"), rows)
        return out


def evaluate_theta0(cfg: DictConfig, env: Environment, *, run_id: str | None = None) -> Path:
    """Evaluate the starting model itself (condition ``theta0``): the baseline of every gate.

    ``run_type: theta0`` keeps its config hash apart from the training runs of the same config.
    """
    if cfg.run_type != "theta0":
        raise ValueError("evaluate_theta0 needs run_type=theta0")
    seed_everything(derive_seed(cfg.seed, "torch"))
    with Run(cfg, run_id=run_id) as run:
        run.record_inputs({"prompt_manifest_sha256": env.prompt_manifest_sha256})
        model = env.new_model()
        model.requires_grad_(False)
        run.summarize(Evaluator(cfg, env, run, model)(model, "theta0"))
        return run.dir


def run_training(cfg: DictConfig, env: Environment, *, run_id: str | None = None) -> Path:
    """Train one cell's seed and evaluate it; returns the run directory."""
    if cfg.run_type != "train":
        raise ValueError("run_training needs run_type=train")
    if cfg.train.save_checkpoint:
        raise NotImplementedError("train.save_checkpoint: weight saving is not implemented yet")
    seed_everything(derive_seed(cfg.seed, "torch"))
    cuda = env.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    with Run(cfg, run_id=run_id) as run:
        kind = stream_kind(cfg)
        record = require_path_check(env.data_root, cfg, env.device)  # before anything costly
        run.record_inputs({"path_check": {"run_id": record["run_id"],
                                          "fingerprint": record["fingerprint"]}})  # fmt: skip
        stream, manifest = load_stream_for(cfg, env, kind, cfg.seed)
        run.summarize({"stream/training": manifest["sha256"], "stream/kind": kind})
        run.record_inputs({"prompt_manifest_sha256": env.prompt_manifest_sha256,
                           "streams": {kind: _identity(manifest)}})  # fmt: skip
        batches = training_batches(stream, cfg)
        reader = None
        if kind == "soft_targets":  # read lazily, one batch at a time
            _, path, _ = locate_stream(cfg, env, kind, cfg.seed)
            reader = load_distributions(path)
            position = {id(s): i for i, s in enumerate(stream)}

        student, reference = env.new_model(), env.new_model()
        reference.requires_grad_(False)
        trainable = freeze(student, cfg.train.freeze)
        params = [p for _, p in trainable]
        theta0 = dict(reference.named_parameters())
        evaluate = Evaluator(cfg, env, run, reference)  # fails early on missing inputs
        directions = projection_directions(trainable, cfg.eval.projections)
        if kind != "alpaca_reference":  # reference answers have no teacher log-probs
            gap, chi_square = teacher_gap(student, stream, cfg, env)
            run.summarize({"step0_teacher_gap": gap})
            check_teacher_gap(gap, cfg.checks.teacher_gap)  # stops before any optimizer step
            index = {id(s): i for i, s in enumerate(stream)}
            sums = [sum(chi_square[index[id(s)]] for s in batch) for batch in batches]
            run.summarize({"step0_path_field": check_run_chi_square(sums, cfg.train.steps)})

        second_moment = None
        if ADAM_MODES.get(cfg.optimizer.name, "adaptive") != "adaptive":
            second_moment = _theta0_second_moment(cfg, env, student, params, run)
        optimizer = build_optimizer(cfg.optimizer, params, second_moment=second_moment)
        del second_moment

        for step, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            targets = None
            if reader is not None:
                targets = reader.take([position[id(s)] for s in batch])
            stats = accumulate_gradient(student, batch, cfg, env, targets)
            optimizer.step()
            stats |= projection_stats(trainable, theta0, directions)
            run.log(stats, step=step)
        del optimizer, directions
        for p in params:
            p.grad = None
        if cuda:
            run.summarize({"peak_memory_gb/train": torch.cuda.max_memory_allocated() / 2**30})
            torch.cuda.empty_cache()

        student.eval()
        run.summarize(evaluate(student, "trained"))
        geometry = displacement(trainable, theta0)
        run.path("observations", "geometry.json").write_text(json.dumps(geometry, indent=1))
        run.summarize({key: geometry[key] for key in ("delta_norm", "theta0_norm", "relative")})
        controls = _controls(cfg, env, student, trainable, theta0, evaluate)
        if controls:
            run.summarize({"controls": controls})
        if cuda:
            run.summarize({"peak_memory_gb/all": torch.cuda.max_memory_allocated() / 2**30})
        return run.dir


def _theta0_second_moment(
    cfg: DictConfig, env: Environment, student: PreTrainedModel, params: list, run: Run
) -> list[torch.Tensor]:
    """The θ₀ estimate of Adam's second moment from the separate preconditioner stream (H2a)."""
    stream, manifest = load_stream_for(cfg, env, "preconditioner", cfg.seed)
    run.summarize({"stream/preconditioner": manifest["sha256"]})
    run.record_inputs({"streams": {"preconditioner": _identity(manifest)}})
    pre = cfg.optimizer.preconditioner
    batches = list(chunks(stream, pre.batch_size))
    if len(batches) != pre.batches or any(len(b) != pre.batch_size for b in batches):
        raise ValueError(f"The preconditioner stream must hold {pre.batches} full batches")
    return estimate_second_moment(
        params, lambda i: accumulate_gradient(student, batches[i], cfg, env), pre.batches
    )


def _controls(
    cfg: DictConfig,
    env: Environment,
    student: PreTrainedModel,
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    theta0: Mapping[str, torch.Tensor],
    evaluate: Evaluator,
) -> dict[str, dict[str, float]]:
    """Line scan θ₀ + aΔθ along the whole displacement and random directions of matched norm."""
    controls = cfg.eval.controls
    if not (controls.line_scan.enabled or controls.random_direction.enabled):
        return {}
    start = [theta0[name].detach() for name, _ in trainable]
    delta = [p.detach() - s for (_, p), s in zip(trainable, start, strict=True)]
    out = {}
    if controls.line_scan.enabled:
        for sign in controls.line_scan.signs:
            for a in controls.line_scan.coefficients:
                set_point(trainable, start, delta, sign * a)
                condition = f"line_scan_{sign * a:+g}"
                out[condition] = evaluate(student, condition)
    if controls.random_direction.enabled:
        shapes = {name: tuple(p.shape) for name, p in trainable}
        norms = {name: d.norm().item() for (name, _), d in zip(trainable, delta, strict=True)}
        for kind in controls.random_direction.kinds:
            r = random_direction(shapes, norms, kind=kind, seed=derive_seed(cfg.seed, "random"),
                                 device=env.device)  # fmt: skip
            set_point(trainable, start, [r[name] for name, _ in trainable], 1.0)
            del r
            out[f"random_{kind}"] = evaluate(student, f"random_{kind}")
    set_point(trainable, start, delta, 1.0)
    return out


def _identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: manifest[key] for key in ("path", "sha256", "count")}


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
