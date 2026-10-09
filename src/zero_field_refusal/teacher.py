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
# Part of every stream's identity: bump on any change to how samples are drawn or stored, so old
# streams are never reused by new code.
SAMPLER_VERSION = 1


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


def validate_samples(samples: Sequence[Sample], end_ids: Sequence[int], cap: int) -> None:
    """Check the record invariants of a sampled stream: one log-prob per token, the first end
    token last, ``ended`` exactly when it is there, and no response longer than the cap."""
    ends = set(end_ids)
    for i, s in enumerate(samples):
        problems = []
        if not s.response_ids or len(s.response_ids) > cap:
            problems.append(f"length {len(s.response_ids)} outside [1, {cap}]")
        if len(s.logprobs) != len(s.response_ids):
            problems.append("log-probs do not match the response tokens")
        if any(token in ends for token in s.response_ids[:-1]):
            problems.append("an end token before the last position")
        if s.ended != (s.response_ids[-1] in ends if s.response_ids else False):
            problems.append("ended flag disagrees with the last token")
        if not s.ended and len(s.response_ids) != cap:
            problems.append(
                f"no end token but {len(s.response_ids)} < {cap} tokens: not truncation"
            )
        if problems:
            raise StreamError(f"Sample {i} ({s.prompt_id}): " + "; ".join(problems))


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(_manifest_path(path).read_text())


def _manifest_path(path: Path) -> Path:
    return path.with_name(path.stem + ".manifest.json")


# Streams. A stream's directory is named by the hash of everything that determines its samples, so
# a changed model, prompt set or sampling setting can never reuse old samples.
STREAM_KINDS = ("training", "preconditioner", "kl_harmful", "kl_benign", "soft_targets",
                "alpaca_reference")  # fmt: skip


def prompt_set_digest(items: Sequence[Mapping[str, Any]]) -> str:
    """sha256 of the ids, texts and reference answers (if any) of a prompt set, in order."""
    blob = json.dumps(
        [[item["id"], item["text"], item.get("reference")] for item in items],
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def stream_spec(cfg: Any, kind: str, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Everything that determines a stream's samples, given its prompt set ``items``.

    K's reference streams always use zero-field sampling, whatever control a run declares: they are
    the measuring instrument shared by all runs of a model.
    """
    if kind not in STREAM_KINDS:
        raise ValueError(f"Unknown stream kind {kind!r}; known: {STREAM_KINDS}")
    model = cfg.model
    spec: dict[str, Any] = {
        "kind": kind,
        "sampler_version": SAMPLER_VERSION,
        "model": model.name,
        "revision": model.revision,
        "dtype": model.dtype,
        "attn_implementation": model.attn_implementation,
        "end_tokens": list(model.end_tokens),
        "prompts": prompt_set_digest(items),
    }
    zero_field = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "repetition_penalty": 1.0}
    if kind in ("training", "soft_targets"):
        spec["sampling"] = {key: cfg.teacher.sampling[key] for key in SAMPLING_KEYS}
        spec["responses"] = int(cfg.teacher.responses)
    elif kind == "preconditioner":
        spec["sampling"] = {key: cfg.teacher.sampling[key] for key in SAMPLING_KEYS}
        spec["responses"] = int(cfg.optimizer.preconditioner.responses)
    elif kind in ("kl_harmful", "kl_benign"):
        spec["sampling"] = {**zero_field, "max_new_tokens": int(cfg.eval.kl.max_new_tokens)}
        spec["samples_per_prompt"] = int(cfg.eval.kl.samples_per_prompt)
    else:  # alpaca_reference: reference answers, truncated to the sampling cap
        spec["max_new_tokens"] = int(cfg.teacher.sampling.max_new_tokens)
        spec["responses"] = int(cfg.teacher.responses)
    return spec


def stream_path(spec: Mapping[str, Any], seed: int | None, root: Path) -> Path:
    """``<root>/teacher/<model>/<kind>/<spec hash>/seed-<seed>.jsonl`` (K: ``reference.jsonl``)."""
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    model = spec["model"].replace("/", "--")
    name = "reference.jsonl" if seed is None else f"seed-{seed}.jsonl"
    return root / "teacher" / model / spec["kind"] / digest / name


def stream_seeds(kind: str, seed: int | None) -> tuple[int, int]:
    """Seeds of the prompt draws and of sampling for a stream.

    K's reference streams are sampled once per model, independently of any run seed.
    """
    if kind in ("kl_harmful", "kl_benign"):
        return derive_seed(0, "kl_reference", kind, "prompts"), derive_seed(0, "kl_reference", kind)
    if seed is None:
        raise ValueError(f"A {kind} stream needs a run seed")
    name = "teacher" if kind in ("training", "soft_targets") else kind
    return derive_seed(seed, name, "prompts"), derive_seed(seed, name, "sampling")


def stream_prompt_indices(spec: Mapping[str, Any], num_prompts: int, prompt_seed: int) -> list[int]:
    """Which prompt each response answers: each prompt k times for K, iid draws otherwise."""
    if "samples_per_prompt" in spec:
        return [i for i in range(num_prompts) for _ in range(spec["samples_per_prompt"])]
    return draw_prompt_indices(spec["responses"], num_prompts, prompt_seed)


def reference_samples(
    items: Sequence[Mapping[str, Any]],
    encode_prompt: Any,
    encode_text: Any,
    *,
    end_id: int,
    cap: int,
) -> list[Sample]:
    """Reference answers as samples: truncated to ``cap`` tokens, or closed by ``end_id``.

    Used for the held-out reference NLL and for the Alpaca-reference control, which is a real
    field by design. These texts are tokenized once; nothing here comes from teacher samples.
    """
    samples = []
    for item in items:
        tokens = list(encode_text(item["reference"]))
        ended = len(tokens) < cap
        response = tokens + [end_id] if ended else tokens[:cap]
        samples.append(
            Sample(
                prompt_id=item["id"],
                prompt_ids=tuple(encode_prompt(item["text"])),
                response_ids=tuple(response),
                logprobs=(),
                ended=ended,
            )
        )
    return samples
