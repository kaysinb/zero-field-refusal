"""Training-path log-probabilities and losses.

This is the training path that the soft-target check compares with the generation path: one full
forward pass over right-padded prompt + response rows. Only response positions reach the output
head, which keeps the (tokens × vocabulary) logits small.

The loss of a batch is its summed token NLL divided by a constant, ``batch size × response cap``
(invariant 5). Dividing by the batch's own token count would weight short responses more and push
the model toward shorter answers; it exists only as the token-count control. With a constant
denominator, micro-batch gradients add up exactly to the batch gradient.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from transformers import PreTrainedModel

from .batches import ResponseBatch

if TYPE_CHECKING:
    from .teacher import Sample


def response_logits(model: PreTrainedModel, batch: ResponseBatch) -> torch.Tensor:
    """(num_tokens, vocab) logits at the positions that predict response tokens."""
    hidden = model.base_model(
        input_ids=batch.input_ids, attention_mask=batch.attention_mask
    ).last_hidden_state
    return model.get_output_embeddings()(hidden[:, :-1][batch.target_mask]).float()


def response_logprobs(model: PreTrainedModel, batch: ResponseBatch) -> torch.Tensor:
    """(num_tokens, vocab) fp32 log-softmax at the response positions."""
    return torch.log_softmax(response_logits(model, batch), dim=-1)


def targets(batch: ResponseBatch) -> torch.Tensor:
    """(num_tokens,) response token ids, in the order of ``response_logits``."""
    return batch.input_ids[:, 1:][batch.target_mask]


def summed_nll(model: PreTrainedModel, batch: ResponseBatch) -> torch.Tensor:
    """Sum over response tokens, including a sampled end token, of −log p(token | prefix)."""
    return F.cross_entropy(response_logits(model, batch), targets(batch), reduction="sum")


def soft_kl(
    model: PreTrainedModel, batch: ResponseBatch, target_logprobs: torch.Tensor
) -> torch.Tensor:
    """Σ over response positions of KL(target ‖ student) for the soft-target control.

    ``target_logprobs`` are θ₀'s (num_tokens, vocab) log-probabilities recorded on the generation
    path. The gradient at θ₀ vanishes only as far as both paths agree.
    """
    student = response_logprobs(model, batch)
    return F.kl_div(student, target_logprobs, log_target=True, reduction="sum")


def loss_denominator(
    loss_cfg: DictConfig, samples: Sequence[Sample], max_response_tokens: int
) -> float:
    """The constant of the protocol, or the batch's token count under the token-count control."""
    if loss_cfg.normalization == "constant":
        return float(loss_cfg.denominator)
    if loss_cfg.normalization == "token_count":
        return float(sum(min(len(s.response_ids), max_response_tokens) for s in samples))
    raise ValueError(f"Unknown loss normalization {loss_cfg.normalization!r}")
