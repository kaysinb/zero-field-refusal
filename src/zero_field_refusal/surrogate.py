"""Study 5's refusal-propensity surrogate r(θ) and its gradient (docs/study5-preregistration.md,
"The surrogate and the predictor").

On the harmful prompts x of the K_harmful scoring panel, with fixed refusal continuations R and
compliance continuations C (``configs/study5/surrogate-2.yaml``, the second set):

    r(θ) = mean_x [ logmeanexp_{y∈R} log p_θ(y|x) − logmeanexp_{y∈C} log p_θ(y|x) ],

where log p_θ(y|x) is the token-summed log-probability of y, teacher-forced right after the
assistant-turn start of the chat template, without length normalization.

The numbers come from the training path itself: a row is the prompt's chat encoding
(``models.encode_chat``, as every teacher stream's prompt) followed by the continuation's ids
(the tokenizer without special tokens: no BOS, no end token), collated as a training batch
(``batches.collate``) and scored by ``losses.response_logits`` in fp32, the forward pass and
attention implementation of training. Only the arithmetic after the per-row sums (logmeanexp, the
mean over prompts) is in float64.

The gradient is taken over the parameters training updates (``models.freeze``'s trainable set)
in two passes: the per-row log-probabilities first, without a graph; then each micro-batch's
rows weighted by ∂r/∂log p(y|x), which is the softmax of each class's log-probabilities within
its prompt divided by the number of prompts (positive for R, negative for C), and
backpropagated. The micro-batch gradients add up to ∇r exactly, as training's micro-batches add
up to the batch gradient.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .batches import ResponseBatch, chunks, collate
from .losses import response_logits, targets
from .models import encode_chat
from .paths import CONFIG_ROOT
from .teacher import Sample

# The second set (surrogate.yaml without R4); surrogate.yaml, the first, stays as the record
SPEC_PATH = CONFIG_ROOT / "study5" / "surrogate-2.yaml"
NO_CAP = 10**9  # continuations are scored in full
DEFAULT_MICRO_BATCH = 32


@dataclass(frozen=True)
class SurrogateSpec:
    """The surrogate's definition, and the sha256 of its canonical JSON (``sha256``)."""

    version: int
    harmful_panel: str
    benign_panel: str
    max_tokens: int
    refusals: tuple[str, ...]
    compliances: tuple[str, ...]
    rank_share: float
    sha256: str

    @property
    def continuations(self) -> tuple[str, ...]:
        """R, then C: the column order of ``continuation_logprobs``."""
        return self.refusals + self.compliances


def spec_sha256(node: Any) -> str:
    """sha256 of the surrogate mapping as canonical JSON (sorted keys, no spaces, UTF-8)."""
    blob = json.dumps(node, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_spec(path: str | Path = SPEC_PATH) -> SurrogateSpec:
    node = OmegaConf.to_container(OmegaConf.load(Path(path)), resolve=True)["surrogate"]
    refusals, compliances = tuple(node["refusals"]), tuple(node["compliances"])
    if not refusals or not compliances:
        raise ValueError("The surrogate needs at least one refusal and one compliance")
    return SurrogateSpec(
        version=int(node["version"]),
        harmful_panel=str(node["panels"]["harmful"]),
        benign_panel=str(node["panels"]["benign"]),
        max_tokens=int(node["max_tokens"]),
        refusals=refusals,
        compliances=compliances,
        rank_share=float(node["checks"]["rank_share"]),
        sha256=spec_sha256(node),
    )


# ------------------------------------------------------------------------------- token rows


def continuation_ids(
    tokenizer: PreTrainedTokenizerBase, text: str, max_tokens: int | None = None
) -> list[int]:
    """The ids of a continuation as it follows the assistant-turn start: no special token (no
    BOS, no end-of-turn), at most ``max_tokens``."""
    ids = [int(i) for i in tokenizer(text, add_special_tokens=False)["input_ids"]]
    special = set(tokenizer.all_special_ids)
    if not ids or special.intersection(ids):
        raise ValueError("A continuation encodes to no tokens or contains a special token")
    if max_tokens is not None and len(ids) > max_tokens:
        raise ValueError(f"A continuation has {len(ids)} tokens, more than {max_tokens}")
    return ids


def continuation_rows(
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    continuations: Sequence[str],
    max_tokens: int | None = None,
) -> list[Sample]:
    """One training-path row per (prompt, continuation), prompt-major: the chat encoding of the
    prompt with the generation prompt, then the continuation as the response."""
    encoded = [tuple(continuation_ids(tokenizer, y, max_tokens)) for y in continuations]
    rows = []
    for i, text in enumerate(prompts):
        prefix = tuple(encode_chat(tokenizer, text))
        rows += [Sample(prompt_id=f"{i}:{j}", prompt_ids=prefix, response_ids=ids, logprobs=(),
                        ended=False) for j, ids in enumerate(encoded)]  # fmt: skip
    return rows


def _batch(rows: Sequence[Sample], tokenizer: PreTrainedTokenizerBase, device) -> ResponseBatch:
    return collate(rows, pad_id=tokenizer.pad_token_id, max_response_tokens=NO_CAP, device=device)


def _row_logprobs(model: PreTrainedModel, batch: ResponseBatch) -> torch.Tensor:
    """(rows,) fp32: per row, the summed log-probability of its continuation tokens."""
    per_token = -F.cross_entropy(response_logits(model, batch), targets(batch), reduction="none")
    return torch.stack([part.sum() for part in per_token.split(batch.lengths)])


# ------------------------------------------------------------------------------- values


def logmeanexp(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.logsumexp(values, dim=dim) - math.log(values.shape[dim])


@torch.no_grad()
def continuation_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    continuations: Sequence[str],
    *,
    device: str | torch.device,
    micro_batch_size: int = DEFAULT_MICRO_BATCH,
    max_tokens: int | None = None,
) -> torch.Tensor:
    """(prompts, continuations) float64 log p(y|x), each summed over y's tokens in fp32."""
    rows = continuation_rows(tokenizer, prompts, continuations, max_tokens)
    return _score(model, tokenizer, rows, device, micro_batch_size).view(
        len(prompts), len(continuations)
    )


@torch.no_grad()
def _score(model, tokenizer, rows, device, micro_batch_size) -> torch.Tensor:
    """(rows,) float64 continuation log-probabilities, micro-batch by micro-batch."""
    out = [_row_logprobs(model, _batch(chunk, tokenizer, device)).cpu()
           for chunk in chunks(rows, micro_batch_size)]  # fmt: skip
    return torch.cat(out).double()


def contrasts(logprobs: torch.Tensor, refusals: int) -> torch.Tensor:
    """(prompts,): logmeanexp over the first ``refusals`` columns (R) minus over the rest (C)."""
    return logmeanexp(logprobs[:, :refusals]) - logmeanexp(logprobs[:, refusals:])


def prompt_values(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    refusals: Sequence[str],
    compliances: Sequence[str],
    *,
    device: str | torch.device,
    micro_batch_size: int = DEFAULT_MICRO_BATCH,
    max_tokens: int | None = None,
) -> list[float]:
    """Per prompt, logmeanexp_R log p(y|x) − logmeanexp_C log p(y|x)."""
    logp = continuation_logprobs(model, tokenizer, prompts, [*refusals, *compliances],
                                 device=device, micro_batch_size=micro_batch_size,
                                 max_tokens=max_tokens)  # fmt: skip
    return contrasts(logp, len(refusals)).tolist()


def surrogate_value(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    refusals: Sequence[str],
    compliances: Sequence[str],
    *,
    device: str | torch.device,
    micro_batch_size: int = DEFAULT_MICRO_BATCH,
    max_tokens: int | None = None,
) -> float:
    """r(θ): the mean over prompts of ``prompt_values``."""
    values = prompt_values(model, tokenizer, prompts, refusals, compliances, device=device,
                           micro_batch_size=micro_batch_size, max_tokens=max_tokens)  # fmt: skip
    return math.fsum(values) / len(values)


def row_weights(logprobs: torch.Tensor, refusals: int) -> torch.Tensor:
    """(prompts, continuations) float64 ∂r/∂log p(y|x): within each prompt, the softmax over R
    (positive) and minus the softmax over C, divided by the number of prompts."""
    n = logprobs.shape[0]
    return torch.cat([torch.softmax(logprobs[:, :refusals], dim=-1),
                      -torch.softmax(logprobs[:, refusals:], dim=-1)], dim=-1) / n  # fmt: skip


def surrogate_gradient(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    refusals: Sequence[str],
    compliances: Sequence[str],
    *,
    device: str | torch.device,
    trainable: Sequence[tuple[str, torch.nn.Parameter]],
    micro_batch_size: int = DEFAULT_MICRO_BATCH,
    max_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """∇r(θ) over ``trainable`` (``models.freeze``'s list), fp32, by parameter name.

    The parameters' ``.grad`` must be empty on entry (a leftover gradient would be added in) and
    is left empty; the gradient is returned detached, on the parameters' devices.
    """
    if any(p.grad is not None for _, p in trainable):
        raise ValueError("The trainable parameters hold a gradient; clear it first")
    if not all(p.requires_grad for _, p in trainable):
        raise ValueError("A parameter of the trainable set does not require a gradient")
    continuations = [*refusals, *compliances]
    rows = continuation_rows(tokenizer, prompts, continuations, max_tokens)
    logp = _score(model, tokenizer, rows, device, micro_batch_size).view(len(prompts), -1)
    weights = row_weights(logp, len(refusals)).flatten().float()
    with torch.enable_grad():
        for start in range(0, len(rows), micro_batch_size):
            chunk = rows[start : start + micro_batch_size]
            lp = _row_logprobs(model, _batch(chunk, tokenizer, device))
            w = weights[start : start + len(chunk)].to(lp.device)
            (lp * w).sum().backward()
    grads = {}
    for name, p in trainable:
        # The accumulated tensor itself, handed over: no copy of the trainable size.
        grads[name] = torch.zeros_like(p) if p.grad is None else p.grad.detach()
        p.grad = None
    return grads


# ------------------------------------------------------------------------------- checks at θ₀


def rank_share(harmful: torch.Tensor, benign: torch.Tensor) -> float:
    """The share of (harmful, benign) prompt pairs whose harmful difference is the larger; a tie
    counts one half."""
    diff = harmful[:, None] - benign[None, :]
    return ((diff > 0).double().sum() + 0.5 * (diff == 0).double().sum()).item() / diff.numel()


def theta0_checks(
    harmful_logprobs: torch.Tensor,
    benign_logprobs: torch.Tensor,
    refusals: int,
    *,
    min_rank_share: float,
) -> dict[str, Any]:
    """The surrogate's checks before any extraction, from the (prompts, R + C) log-probability
    matrices of the two panels:

    - ``positive``: the difference is > 0 on every harmful prompt;
    - ``rank``: harmful > benign in at least ``min_rank_share`` of the prompt pairs;
    - ``per_continuation``: each continuation alone, in place of its class's set, has its class's
      sign in the harmful panel's mean: for y ∈ R, mean_x [log p(y|x) − logmeanexp_C] > 0; for
      y ∈ C, mean_x [log p(y|x) − logmeanexp_R] < 0. The share of prompts with that sign is
      recorded beside it.
    """
    harmful = contrasts(harmful_logprobs, refusals)
    benign = contrasts(benign_logprobs, refusals)
    positive = int((harmful > 0).sum())
    share = rank_share(harmful, benign)
    lme_r = logmeanexp(harmful_logprobs[:, :refusals])
    lme_c = logmeanexp(harmful_logprobs[:, refusals:])
    columns = []
    for j in range(harmful_logprobs.shape[1]):
        is_refusal = j < refusals
        alone = harmful_logprobs[:, j] - (lme_c if is_refusal else lme_r)
        sign = 1.0 if is_refusal else -1.0
        mean = alone.mean().item()
        columns.append({"class": "R" if is_refusal else "C",
                        "index": j if is_refusal else j - refusals, "mean": mean,
                        "prompt_share": (sign * alone > 0).double().mean().item(),
                        "passes": sign * mean > 0})  # fmt: skip
    checks = {
        "positive": {"prompts": harmful.numel(), "positive": positive,
                     "min": harmful.min().item(), "passes": positive == harmful.numel()},
        "rank": {"pairs": harmful.numel() * benign.numel(), "share": share,
                 "min_share": min_rank_share, "passes": share >= min_rank_share},
        "per_continuation": {"columns": columns, "passes": all(c["passes"] for c in columns)},
    }  # fmt: skip
    return {
        "r": harmful.mean().item(),
        "r_benign": benign.mean().item(),
        "checks": checks,
        "passes": all(c["passes"] for c in checks.values()),
        "harmful": harmful.tolist(),
        "benign": benign.tolist(),
    }
