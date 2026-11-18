"""Study 5's refusal direction at θ₀ (docs/study5-refdir-registration.md): the candidate directions,
their interventions, the registered selection rule and the projections of the training answers.

Everything here is a forward pass of θ₀ in fp32 under the run's own numerics, with no padding:
rows run in batches of equal length (``equal_length_batches``), so no row ever holds a pad token.

**Layers.** ``hidden_states[ℓ]`` is the input of block ℓ (ℓ ≥ 1), read by a forward pre-hook on
``model.model.layers[ℓ]`` (``capture_inputs``); the embedding output (ℓ = 0) is not a candidate.

**Interventions.**
- ``ablate(model, unit)``: the unit vector is projected out of every write to the residual stream,
  at every position: the embedding output and the output of every attention and every MLP sublayer
  (forward hooks).
- ``add(model, layer, vector)``: the vector is added to the input of block ``layer`` at every
  position (a forward pre-hook).

**The refusal score** is the study's surrogate r per prompt (``surrogate.prompt_values``'
definition: logmeanexp over the refusal continuations of the teacher-forced log-probability minus
the same over the compliance continuations), computed here row by row on equal-length batches of
the training path's rows (``surrogate.continuation_rows``, ``batches.collate``,
``surrogate._row_logprobs``), with whatever hooks are active.

**Selection** (``select``): per candidate, bypass (mean r on the harmful selection prompts under
ablation), induce (mean r on the harmless ones under addition) and KL (mean KL(p_θ₀ ‖ p_ablated)
of the next-token distribution at the last prompt position on the harmless ones). A candidate is
validated when bypass < the harmful baseline, induce > the harmless baseline and > 0, and KL < 0.1;
the chosen one has the lowest bypass, ties by the lower layer, then the earlier position; none
validated: no direction (``None``).
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from . import surrogate
from .batches import collate
from .models import encode_chat
from .teacher import Sample

KL_MAX = 0.1
MAX_LAYER_SHARE = 0.8  # candidate layers: 1 <= l < 0.8 L
BATCH = 16  # rows per equal-length batch


# ------------------------------------------------------------------ rows and batches


def equal_length_batches(lengths: Sequence[int], size: int = BATCH) -> Iterator[list[int]]:
    """Row indices grouped by equal length (longest first), each group cut into batches of at
    most ``size``; every row appears once."""
    groups: dict[int, list[int]] = {}
    for i, n in enumerate(lengths):
        groups.setdefault(int(n), []).append(i)
    for n in sorted(groups, reverse=True):
        rows = groups[n]
        for a in range(0, len(rows), size):
            yield rows[a : a + size]


def common_suffix(sequences: Sequence[Sequence[int]]) -> list[int]:
    """The longest suffix shared by every sequence: the template's post-instruction tokens."""
    if not sequences:
        raise ValueError("No sequence")
    n = min(len(s) for s in sequences)
    k = 0
    while k < n and len({int(s[len(s) - 1 - k]) for s in sequences}) == 1:
        k += 1
    if k == 0:
        raise ValueError("The prompts share no suffix: no post-instruction token")
    return [int(t) for t in sequences[0][len(sequences[0]) - k :]]


def post_instruction_ids(
    tokenizer: PreTrainedTokenizerBase, prompts: Sequence[Sequence[int]]
) -> list[int]:
    """The template's post-instruction tokens: the suffix that two one-token instructions share
    once rendered (the tokens after the instruction up to the start of the answer). Every
    encoded prompt must end with it; a prompt that does not raises ``ValueError``."""
    tail = common_suffix([encode_chat(tokenizer, "A"), encode_chat(tokenizer, "B")])
    for ids in prompts:
        if [int(t) for t in ids[len(ids) - len(tail) :]] != tail:
            raise ValueError("A prompt does not end with the template's post-instruction tokens")
    return tail


# ------------------------------------------------------------------ hooks


def _blocks(model: PreTrainedModel):
    return model.model.layers


def num_layers(model: PreTrainedModel) -> int:
    return len(_blocks(model))


def candidate_layers(model: PreTrainedModel) -> list[int]:
    """1 <= l < 0.8 L."""
    limit = MAX_LAYER_SHARE * num_layers(model)
    return [layer for layer in range(1, num_layers(model)) if layer < limit]


def _hidden(args: tuple, kwargs: dict) -> torch.Tensor:
    return args[0] if args else kwargs["hidden_states"]


@contextmanager
def capture_inputs(model: PreTrainedModel, layers: Iterable[int]) -> Iterator[dict[int, Any]]:
    """The input of each block in ``layers`` at the last forward pass, by layer."""
    store: dict[int, Any] = {}
    handles = []
    for layer in layers:

        def hook(module, args, kwargs, layer=layer):  # noqa: ANN001
            store[layer] = _hidden(args, kwargs).detach()

        handles.append(_blocks(model)[layer].register_forward_pre_hook(hook, with_kwargs=True))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def _remove(t: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    return t - (t @ unit).unsqueeze(-1) * unit


@contextmanager
def ablate(model: PreTrainedModel, unit: torch.Tensor) -> Iterator[None]:
    """``unit`` projected out of the embedding output and every attention and MLP output."""
    if not math.isclose(float(unit.norm()), 1.0, rel_tol=1e-5):
        raise ValueError("ablate needs a unit vector")

    def out(module, args, output):  # noqa: ANN001
        if isinstance(output, tuple):
            return (_remove(output[0], unit), *output[1:])
        return _remove(output, unit)

    handles = [model.model.embed_tokens.register_forward_hook(out)]
    for block in _blocks(model):
        handles.append(block.self_attn.register_forward_hook(out))
        handles.append(block.mlp.register_forward_hook(out))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def add(model: PreTrainedModel, layer: int, vector: torch.Tensor) -> Iterator[None]:
    """``vector`` added to the input of block ``layer`` at every position."""

    def pre(module, args, kwargs):  # noqa: ANN001
        if args:
            return (args[0] + vector, *args[1:]), kwargs
        return args, {**kwargs, "hidden_states": kwargs["hidden_states"] + vector}

    handle = _blocks(model)[layer].register_forward_pre_hook(pre, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


# ------------------------------------------------------------------ forward passes


@torch.no_grad()
def post_instruction_means(
    model: PreTrainedModel,
    prompt_ids: Sequence[Sequence[int]],
    layers: Sequence[int],
    positions: int,
    device: torch.device,
) -> torch.Tensor:
    """(layers, positions, d) float64: the mean over prompts of each block input at the last
    ``positions`` tokens of each prompt."""
    total = None
    for rows in equal_length_batches([len(p) for p in prompt_ids]):
        ids = torch.tensor([list(prompt_ids[i]) for i in rows], device=device)
        with capture_inputs(model, layers) as store:
            model(input_ids=ids)
        stacked = torch.stack([store[layer][:, -positions:, :] for layer in layers], dim=1)
        part = stacked.double().sum(dim=0).cpu()
        total = part if total is None else total + part
    return total / len(prompt_ids)


@torch.no_grad()
def refusal_scores(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    spec: surrogate.SurrogateSpec,
    device: torch.device,
) -> list[float]:
    """Per prompt, the surrogate r under whatever hooks are active (module docstring)."""
    rows = surrogate.continuation_rows(tokenizer, texts, spec.continuations, spec.max_tokens)
    out = torch.empty(len(rows), dtype=torch.float64)
    lengths = [len(r.prompt_ids) + len(r.response_ids) for r in rows]
    for group in equal_length_batches(lengths):
        batch = collate(
            [rows[i] for i in group],
            pad_id=tokenizer.pad_token_id,
            max_response_tokens=surrogate.NO_CAP,
            device=device,
        )
        out[group] = surrogate._row_logprobs(model, batch).double().cpu()
    logp = out.view(len(texts), len(spec.continuations))
    return surrogate.contrasts(logp, len(spec.refusals)).tolist()


@torch.no_grad()
def last_logprobs(
    model: PreTrainedModel, prompt_ids: Sequence[Sequence[int]], device: torch.device
) -> torch.Tensor:
    """(prompts, vocab) float64 log-probabilities of the next token after each prompt."""
    out = [None] * len(prompt_ids)
    for rows in equal_length_batches([len(p) for p in prompt_ids]):
        ids = torch.tensor([list(prompt_ids[i]) for i in rows], device=device)
        logits = model(input_ids=ids).logits[:, -1, :].double()
        for j, i in enumerate(rows):
            out[i] = F.log_softmax(logits[j], dim=-1).cpu()
    return torch.stack(out)


def kl_mean(base: torch.Tensor, other: torch.Tensor) -> float:
    """Mean over prompts of KL(base ‖ other), both log-probabilities."""
    return float((base.exp() * (base - other)).sum(dim=-1).mean())


# ------------------------------------------------------------------ selection


def candidates(
    means_harmful: torch.Tensor, means_harmless: torch.Tensor, layers: Sequence[int]
) -> tuple[dict[tuple[int, int], torch.Tensor], list[dict]]:
    """r(l, i) = harmful mean − harmless mean; a zero or non-finite vector is excluded, listed."""
    diff = means_harmful - means_harmless
    kept, excluded = {}, []
    for a, layer in enumerate(layers):
        for i in range(diff.shape[1]):
            v = diff[a, i]
            norm = float(v.norm())
            if not torch.isfinite(v).all() or not math.isfinite(norm) or norm == 0.0:
                excluded.append({"layer": layer, "position": i, "norm": norm})
                continue
            kept[(layer, i)] = v
    return kept, excluded


def validated(row: Mapping[str, float], base_harmful: float, base_harmless: float) -> bool:
    return (
        row["bypass"] < base_harmful
        and row["induce"] > base_harmless
        and row["induce"] > 0
        and row["kl"] < KL_MAX
    )


def choose(
    rows: Sequence[Mapping[str, Any]], base_harmful: float, base_harmless: float
) -> Mapping[str, Any] | None:
    """The validated candidate with the lowest bypass; ties by layer, then position; or None."""
    ok = [r for r in rows if validated(r, base_harmful, base_harmless)]
    if not ok:
        return None
    return min(ok, key=lambda r: (r["bypass"], r["layer"], r["position"]))


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    spec: surrogate.SurrogateSpec,
    vector: torch.Tensor,
    layer: int,
    harmful: Sequence[str],
    harmless: Sequence[str],
    harmless_ids: Sequence[Sequence[int]],
    base_logp: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    """One candidate's bypass, induce and KL (module docstring)."""
    v = vector.to(device=device, dtype=torch.float32)
    unit = v / v.norm()
    with ablate(model, unit):
        bypass = refusal_scores(model, tokenizer, harmful, spec, device)
        ablated = last_logprobs(model, harmless_ids, device)
    with add(model, layer, v):
        induce = refusal_scores(model, tokenizer, harmless, spec, device)
    return {
        "bypass": math.fsum(bypass) / len(bypass),
        "induce": math.fsum(induce) / len(induce),
        "kl": kl_mean(base_logp, ablated),
    }


def select(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    spec: surrogate.SurrogateSpec,
    vectors: Mapping[tuple[int, int], torch.Tensor],
    harmful: Sequence[str],
    harmless: Sequence[str],
    device: torch.device,
    layers: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Baselines, every candidate's scores (or only those of ``layers``, for the timing) and the
    chosen candidate by the registered rule."""
    t0 = time.monotonic()
    base_h = refusal_scores(model, tokenizer, harmful, spec, device)
    base_s = refusal_scores(model, tokenizer, harmless, spec, device)
    harmless_ids = [encode_chat(tokenizer, t) for t in harmless]
    base_logp = last_logprobs(model, harmless_ids, device)
    baseline_s = time.monotonic() - t0
    base_harmful, base_harmless = math.fsum(base_h) / len(base_h), math.fsum(base_s) / len(base_s)
    rows = []
    for (layer, i), v in sorted(vectors.items()):
        if layers is not None and layer not in layers:
            continue
        t = time.monotonic()
        found = evaluate(
            model, tokenizer, spec, v, layer, harmful, harmless, harmless_ids, base_logp, device
        )
        rows.append({"layer": layer, "position": i, **found, "seconds": time.monotonic() - t})
    for r in rows:
        r["validated"] = validated(r, base_harmful, base_harmless)
    chosen = choose(rows, base_harmful, base_harmless)
    return {
        "base_harmful": base_harmful,
        "base_harmless": base_harmless,
        "baseline_seconds": baseline_s,
        "rows": rows,
        "chosen": None
        if chosen is None
        else {"layer": chosen["layer"], "position": chosen["position"]},
    }


def confirm(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    spec: surrogate.SurrogateSpec,
    vector: torch.Tensor,
    layer: int,
    harmful: Sequence[str],
    harmless: Sequence[str],
    device: torch.device,
) -> dict[str, float]:
    """On the confirmation prompts: mean r with and without ablation (harmful) and with and
    without addition (harmless)."""
    v = vector.to(device=device, dtype=torch.float32)
    unit = v / v.norm()
    base_h = refusal_scores(model, tokenizer, harmful, spec, device)
    base_s = refusal_scores(model, tokenizer, harmless, spec, device)
    with ablate(model, unit):
        abl_h = refusal_scores(model, tokenizer, harmful, spec, device)
    with add(model, layer, v):
        add_s = refusal_scores(model, tokenizer, harmless, spec, device)

    def mean(xs: Sequence[float]) -> float:
        return math.fsum(xs) / len(xs)

    return {
        "harmful_base": mean(base_h),
        "harmful_ablated": mean(abl_h),
        "harmless_base": mean(base_s),
        "harmless_added": mean(add_s),
    }


# ------------------------------------------------------------------ projections


def answer_positions(sample: Sample) -> list[int]:
    """The positions of an answer's response tokens in prompt + response, the end token
    excluded when the answer ended with one."""
    start = len(sample.prompt_ids)
    n = len(sample.response_ids) - (1 if sample.ended else 0)
    return list(range(start, start + n))


@torch.no_grad()
def answer_means(
    model: PreTrainedModel, samples: Sequence[Sample], layers: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, list[int]]:
    """(samples, layers, d) float32: per answer, the mean of each block input over its answer
    positions; and each answer's number of positions. An answer without a position (its only
    response token is the end token) has no mean: its rows are NaN, never 0."""
    d = model.config.hidden_size
    out = torch.full((len(samples), len(layers), d), float("nan"), dtype=torch.float32)
    counts = [len(answer_positions(s)) for s in samples]
    lengths = [len(s.prompt_ids) + len(s.response_ids) for s in samples]
    for rows in equal_length_batches(lengths):
        ids = torch.tensor(
            [list(samples[i].prompt_ids) + list(samples[i].response_ids) for i in rows],
            device=device,
        )
        with capture_inputs(model, layers) as store:
            model(input_ids=ids)
        for j, i in enumerate(rows):
            pos = answer_positions(samples[i])
            if not pos:
                continue
            for a, layer in enumerate(layers):
                out[i, a] = store[layer][j, pos, :].double().mean(dim=0).float().cpu()
    return out, counts
