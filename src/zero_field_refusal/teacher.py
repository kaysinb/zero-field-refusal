"""Zero-field sampling from θ₀ and stored teacher streams.

The field is zero only if the student trains on responses drawn from its own distribution at θ₀
(docs/architecture.md, "Zero-field invariants"):
- sampling at temperature 1 without top-k, top-p, repetition penalty or minimum length; every
  sampling field is passed explicitly, and for zero-field streams the raw logits must equal the
  scores the sampler used at every step, so no logits processor reshaped the distribution;
- a response ends at the first end token (kept) or is truncated at ``max_new_tokens``; nothing is
  filtered or resampled;
- responses are stored as token ids with their log-probabilities under the raw logits.

GPU sampling is not bit-reproducible, so streams are stored once and treated as the record of the
experiment, each with a manifest and a content hash.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedModel

from .seeding import derive_seed

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens")


class StreamError(RuntimeError):
    """A teacher stream is invalid or does not match its manifest."""


@dataclass(frozen=True)
class Sample:
    prompt_id: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]  # up to and including the first end token, or the cap
    logprobs: tuple[float, ...]  # log-probabilities of response_ids under the raw logits
    ended: bool  # False if truncated at the cap

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "logprobs": list(self.logprobs),
            "ended": self.ended,
        }

    @classmethod
    def from_json(cls, record: Mapping[str, Any]) -> Sample:
        return cls(
            prompt_id=record["prompt_id"],
            prompt_ids=tuple(record["prompt_ids"]),
            response_ids=tuple(record["response_ids"]),
            logprobs=tuple(record["logprobs"]),
            ended=record["ended"],
        )


def draw_prompt_indices(n: int, num_prompts: int, seed: int) -> list[int]:
    """Instruction of each response, drawn uniformly with replacement."""
    return np.random.default_rng(seed).integers(0, num_prompts, size=n).tolist()


def split_response(generated: Sequence[int], end_ids: Sequence[int]) -> tuple[list[int], bool]:
    """The response up to and including the first end token, and whether one was sampled.

    Generation pads finished rows with the pad id, which is itself an end token for Qwen2.5, so
    everything after the first end token is discarded.
    """
    ends = set(end_ids)
    for i, token in enumerate(generated):
        if token in ends:
            return list(generated[: i + 1]), True
    return list(generated), False


@torch.no_grad()
def sample_responses(
    model: PreTrainedModel,
    prompts: Sequence[Sequence[int]],
    prompt_ids: Sequence[str],
    *,
    sampling: Mapping[str, Any],
    end_ids: Sequence[int],
    pad_id: int,
    seed: int,
    batch_size: int,
    zero_field: bool = True,
    record_distributions: bool = False,
) -> tuple[list[Sample], list[torch.Tensor] | None]:
    """Sample one response per prompt with the generation path (``generate``, KV cache).

    ``seed`` seeds each chunk of ``batch_size`` prompts separately (``derive_seed(seed, i)``).
    With ``record_distributions``, also returns per sample the (length, vocab) fp32 log-softmax of
    the raw logits at every response position, on the CPU: the soft targets of invariant 8.
    """
    if len(prompts) != len(prompt_ids):
        raise ValueError("prompts and prompt_ids differ in length")
    device = next(model.parameters()).device
    samples: list[Sample] = []
    distributions: list[torch.Tensor] | None = [] if record_distributions else None
    for index, start in enumerate(range(0, len(prompts), batch_size)):
        chunk = prompts[start : start + batch_size]
        input_ids, attention_mask = _left_pad(chunk, pad_id, device)
        torch.manual_seed(derive_seed(seed, index))
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            num_beams=1,
            **{key: sampling[key] for key in SAMPLING_KEYS},
            eos_token_id=list(end_ids),
            pad_token_id=pad_id,
            output_logits=True,
            output_scores=zero_field,
            return_dict_in_generate=True,
        )
        if zero_field:
            for step, (raw, used) in enumerate(zip(out.logits, out.scores, strict=True)):
                if not torch.equal(raw, used):
                    raise StreamError(
                        f"A logits processor changed the sampling distribution at step {step}; "
                        "zero-field sampling must use the raw distribution"
                    )
        generated = out.sequences[:, input_ids.shape[1] :].tolist()
        # (chunk, steps, vocab) would be large; gather per step instead.
        step_logprobs = [torch.log_softmax(raw.float(), dim=-1) for raw in out.logits]
        for row, tokens in enumerate(generated):
            response, ended = split_response(tokens, end_ids)
            logp = torch.stack([step_logprobs[t][row] for t in range(len(response))])
            picked = logp.gather(1, torch.tensor(response, device=logp.device)[:, None])[:, 0]
            samples.append(
                Sample(
                    prompt_id=prompt_ids[start + row],
                    prompt_ids=tuple(int(i) for i in chunk[row]),
                    response_ids=tuple(response),
                    logprobs=tuple(picked.tolist()),
                    ended=ended,
                )
            )
            if distributions is not None:
                distributions.append(logp.cpu())
    return samples, distributions


def _left_pad(
    chunk: Sequence[Sequence[int]], pad_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(p) for p in chunk)
    input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(chunk), width), dtype=torch.long)
    for b, prompt in enumerate(chunk):
        input_ids[b, width - len(prompt) :] = torch.tensor(prompt)
        attention_mask[b, width - len(prompt) :] = 1
    return input_ids.to(device), attention_mask.to(device)


def save_stream(path: Path, samples: Sequence[Sample], manifest: Mapping[str, Any]) -> None:
    """Write samples as JSONL and ``<name>.manifest.json`` with their hash and summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(s.to_json(), sort_keys=True) + "\n" for s in samples)
    record = {
        **manifest,
        "count": len(samples),
        "ended_fraction": sum(s.ended for s in samples) / len(samples) if samples else None,
        "mean_response_tokens": (
            sum(len(s.response_ids) for s in samples) / len(samples) if samples else None
        ),
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    _manifest_path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def load_stream(path: Path) -> list[Sample]:
    """Samples of a stream, after checking the file against its manifest."""
    manifest_path = _manifest_path(path)
    if not path.is_file() or not manifest_path.is_file():
        raise StreamError(f"Missing stream or manifest: {path}; generate the teacher stream first")
    manifest = json.loads(manifest_path.read_text())
    text = path.read_text()
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != manifest["sha256"]:
        raise StreamError(f"{path}: sha256 {digest} does not match the manifest")
    samples = [Sample.from_json(json.loads(line)) for line in text.splitlines()]
    if len(samples) != manifest["count"]:
        raise StreamError(f"{path}: {len(samples)} samples, manifest says {manifest['count']}")
    return samples


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(_manifest_path(path).read_text())


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.stem + ".manifest.json")
