"""Training batches of stored token ids.

A row is the prompt ids followed by the response ids, truncated to ``max_response_tokens``
(the bridge cell truncates the 96-token stream to 48, which is identical in distribution to
sampling with a 48-token cap). Rows are right-padded. Which positions carry a response target is
recorded explicitly from the lengths, never derived from ``!= pad``: Qwen2.5's pad id equals one
of its end ids, so such a mask would drop a sampled end token (invariant 4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .teacher import Sample


@dataclass
class ResponseBatch:
    input_ids: torch.Tensor  # (B, L) long, right-padded
    attention_mask: torch.Tensor  # (B, L) long
    target_mask: torch.Tensor  # (B, L - 1) bool: input_ids[:, 1:] is a response token
    lengths: list[int]  # response tokens per row, after truncation
    num_tokens: int  # sum of lengths


def collate(
    samples: Sequence[Sample],
    *,
    pad_id: int,
    max_response_tokens: int,
    device: str | torch.device | None = None,
) -> ResponseBatch:
    rows, prompt_lengths, lengths = [], [], []
    for s in samples:
        response = list(s.response_ids[:max_response_tokens])
        rows.append(list(s.prompt_ids) + response)
        prompt_lengths.append(len(s.prompt_ids))
        lengths.append(len(response))
    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    target_mask = torch.zeros((len(rows), width - 1), dtype=torch.bool)
    for b, (row, prompt_length, length) in enumerate(
        zip(rows, prompt_lengths, lengths, strict=True)
    ):
        input_ids[b, : len(row)] = torch.tensor(row)
        attention_mask[b, : len(row)] = 1
        # The token at position t is predicted from position t - 1.
        target_mask[b, prompt_length - 1 : prompt_length - 1 + length] = True
    return ResponseBatch(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        target_mask=target_mask.to(device),
        lengths=lengths,
        num_tokens=sum(lengths),
    )


def chunks[T](items: Sequence[T], size: int) -> list[Sequence[T]]:
    """Consecutive slices of at most ``size`` items, in order."""
    return [items[i : i + size] for i in range(0, len(items), size)]
