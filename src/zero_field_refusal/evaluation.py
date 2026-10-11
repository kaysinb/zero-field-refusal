"""In-process measurements of a trained student: displacement K, greedy responses, utility,
weight geometry and the weight-space controls.

K is the exact KL(p_θ₀ ‖ p_θ) between full next-token distributions, summed over the positions of
θ₀'s stored responses (docs/preregistration.md, "Primary analysis"). θ₀'s distributions come from a
frozen copy evaluated in the same forward path as the student, so K is exactly zero at θ₀ and every
term is non-negative up to rounding. The arithmetic is in float64 to keep small KL values accurate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from .batches import chunks, collate
from .losses import response_logits, targets
from .seeding import derive_seed
from .teacher import Sample, capture_step_logits, check_unprocessed, split_response

# No truncation: evaluation sequences are scored in full.
NO_CAP = 10**9


@torch.no_grad()
def sequence_kl(
    student: PreTrainedModel,
    reference: PreTrainedModel,
    samples: Sequence[Sample],
    *,
    pad_id: int,
    micro_batch_size: int,
) -> torch.Tensor:
    """(n,) float64: per sample, Σ over response positions of KL(p_reference ‖ p_student)."""
    device = next(student.parameters()).device
    per_sample = []
    for chunk in chunks(samples, micro_batch_size):
        batch = collate(chunk, pad_id=pad_id, max_response_tokens=NO_CAP, device=device)
        log_p0 = torch.log_softmax(response_logits(reference, batch).double(), dim=-1)
        log_p = torch.log_softmax(response_logits(student, batch).double(), dim=-1)
        per_token = (log_p0.exp() * (log_p0 - log_p)).sum(-1)
        per_sample += [part.sum() for part in per_token.split(batch.lengths)]
    return torch.stack(per_sample).cpu()


@torch.no_grad()
def sequence_nll(
    model: PreTrainedModel, samples: Sequence[Sample], *, pad_id: int, micro_batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(n,) float64 summed NLL of each sample's response tokens, and (n,) token counts."""
    device = next(model.parameters()).device
    nll, tokens = [], []
    for chunk in chunks(samples, micro_batch_size):
        batch = collate(chunk, pad_id=pad_id, max_response_tokens=NO_CAP, device=device)
        per_token = F.cross_entropy(response_logits(model, batch), targets(batch), reduction="none")
        nll += [part.double().sum() for part in per_token.split(batch.lengths)]
        tokens += batch.lengths
    return torch.stack(nll).cpu(), torch.tensor(tokens)


@torch.no_grad()
def greedy_responses(
    model: PreTrainedModel,
    prompts: Sequence[Sequence[int]],
    *,
    end_ids: Sequence[int],
    pad_id: int,
    max_new_tokens: int,
    batch_size: int,
) -> list[tuple[list[int], bool]]:
    """Greedy responses (up to and including the first end token) and whether each ended.

    As in sampling, no logits processor may change the scores: a repetition penalty would change
    greedy outputs too (docs/architecture.md, "Generation settings").
    """
    device = next(model.parameters()).device
    out: list[tuple[list[int], bool]] = []
    for chunk in chunks(prompts, batch_size):
        width = max(len(p) for p in chunk)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for b, prompt in enumerate(chunk):
            input_ids[b, width - len(prompt) :] = torch.tensor(prompt)
            attention_mask[b, width - len(prompt) :] = 1
        with capture_step_logits(model) as raw:
            generated = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens,
                eos_token_id=list(end_ids),
                pad_token_id=pad_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
        check_unprocessed(raw, generated.scores, "greedy decoding")
        for row in generated.sequences[:, width:].tolist():
            out.append(split_response(row, end_ids))
    return out


@torch.no_grad()
def choice_predictions(
    model: PreTrainedModel,
    prompts: Sequence[Sequence[int]],
    choice_ids: Sequence[int],
    *,
    pad_id: int,
    batch_size: int,
) -> list[int]:
    """Index of the most likely choice token at the position after each prompt.

    Prompts are right-padded, so the last prompt position does not attend to padding.
    """
    device = next(model.parameters()).device
    choice_index = torch.tensor(list(choice_ids), device=device)
    predictions: list[int] = []
    for chunk in chunks(prompts, batch_size):
        width = max(len(p) for p in chunk)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for b, prompt in enumerate(chunk):
            input_ids[b, : len(prompt)] = torch.tensor(prompt)
            attention_mask[b, : len(prompt)] = 1
        hidden = model.base_model(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
        ).last_hidden_state
        last = torch.tensor([len(p) - 1 for p in chunk], device=device)
        logits = model.get_output_embeddings()(hidden[torch.arange(len(chunk)), last])
        predictions += logits[:, choice_index].argmax(-1).tolist()
    return predictions


@torch.no_grad()
def displacement(
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    theta0: Mapping[str, torch.Tensor],
) -> dict:
    """Norms of Δθ = θ − θ₀ and of θ₀ over the trainable parameters, in total and per tensor."""
    per_tensor = {}
    delta_sq = theta0_sq = 0.0
    for name, param in trainable:
        start = theta0[name]
        d = (param.detach().double() - start.double()).pow(2).sum().item()
        s = start.double().pow(2).sum().item()
        per_tensor[name] = {"delta_norm": d**0.5, "theta0_norm": s**0.5}
        delta_sq += d
        theta0_sq += s
    return {
        "delta_norm": delta_sq**0.5,
        "theta0_norm": theta0_sq**0.5,
        "relative": delta_sq**0.5 / theta0_sq**0.5,
        "per_tensor": per_tensor,
    }


def line_scan_condition(a: float) -> str:
    """The response condition of the line-scan point θ₀ + aΔθ, e.g. ``line_scan_-0.25``."""
    return f"line_scan_{a:+g}"


def random_condition(kind: str) -> str:
    """The response condition of a random direction of the displacement's norm."""
    return f"random_{kind}"


@torch.no_grad()
def set_point(
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    start: Sequence[torch.Tensor],
    direction: Sequence[torch.Tensor],
    a: float,
) -> None:
    """Set the trainable parameters to start + a·direction (a = 0 restores start exactly)."""
    for (_, param), s, d in zip(trainable, start, direction, strict=True):
        param.copy_(s if a == 0 else torch.add(s, d, alpha=a))


def random_direction(
    shapes: Mapping[str, Sequence[int]],
    norms: Mapping[str, float],
    *,
    kind: str,
    seed: int,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Gaussian directions with the norm of the displacement, globally or per tensor.

    Each tensor's direction derives from its parameter name and full shape, never from parameter
    order or a shard. It is generated on the CPU and then moved to ``device``: CUDA random numbers
    of large tensors depend on the GPU model, CPU ones do not.
    """
    if kind not in ("global", "per_tensor"):
        raise ValueError(f"Unknown random-direction kind {kind!r}")
    directions = {}
    for name, shape in shapes.items():
        generator = torch.Generator().manual_seed(
            derive_seed(seed, "random_direction", name, list(shape))
        )
        r = torch.randn(tuple(shape), generator=generator).to(device)
        directions[name] = r * (norms[name] / r.norm()) if kind == "per_tensor" else r
    if kind == "global":
        target = sum(norms[name] ** 2 for name in shapes) ** 0.5
        scale = target / torch.sqrt(sum(r.double().pow(2).sum() for r in directions.values()))
        directions = {name: r * scale.float() for name, r in directions.items()}
    return directions
