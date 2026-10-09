"""What a run needs besides its config: tokenizer, models, prompt sets, and teacher streams.

``Environment`` collects the loaders so that the training and generation code can run unchanged
on a tiny CPU model in tests and on the pinned checkpoints on a GPU.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from safetensors.torch import load_file, save_file
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from . import models
from .paths import data_root
from .teacher import (
    Sample,
    StreamError,
    load_manifest,
    load_stream,
    reference_samples,
    sample_responses,
    save_stream,
    stream_path,
    stream_prompt_indices,
    stream_seeds,
    stream_spec,
    validate_samples,
)
from .tracking import hardware, package_versions

# Prompt set of each stream kind; training-like streams answer the teacher prompts.
STREAM_PROMPT_SETS = {
    "kl_harmful": "harmful_kl",
    "kl_benign": "alpaca_benign_kl",
    "alpaca_reference": "alpaca_reference_pool",
}


@dataclass
class Environment:
    tokenizer: PreTrainedTokenizerBase
    end_ids: list[int]
    pad_id: int
    device: torch.device
    new_model: Callable[[], PreTrainedModel]  # θ₀ in eval mode with a neutral generation config
    prompt_set: Callable[[str], list[dict]]
    data_root: Path

    def encode_prompt(self, text: str) -> list[int]:
        return models.encode_chat(self.tokenizer, text)

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)


def environment(cfg: DictConfig, device: str | torch.device) -> Environment:
    """The production environment: pinned checkpoint and tokenizer, prompt sets from DATA_DIR."""
    from . import data  # imported here: building prompt sets needs the datasets stack

    tokenizer = models.load_tokenizer(cfg.model)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the NVIDIA driver (CUDA 13 needs >= 580)")
    return Environment(
        tokenizer=tokenizer,
        end_ids=models.end_token_ids(tokenizer, cfg.model),
        pad_id=tokenizer.pad_token_id,
        device=device,
        new_model=lambda: models.load_model(cfg.model, device, tokenizer),
        prompt_set=lambda name: data.load_prompt_set(name, cfg.data),
        data_root=data_root(),
    )


def prompt_set_of(cfg: DictConfig, kind: str) -> str:
    return STREAM_PROMPT_SETS.get(kind, cfg.teacher.prompts)


def locate_stream(cfg: DictConfig, env: Environment, kind: str, seed: int | None):
    """(spec, path, prompt items) of a stream; K's reference streams take no seed."""
    items = env.prompt_set(prompt_set_of(cfg, kind))
    spec = stream_spec(cfg, kind, items)
    seed = None if kind in ("kl_harmful", "kl_benign") else seed
    return spec, stream_path(spec, seed, env.data_root), items


def generate_stream(
    cfg: DictConfig,
    env: Environment,
    kind: str,
    seed: int | None,
    *,
    model: PreTrainedModel | None = None,
    batch_size: int,
) -> Path:
    """Sample a stream from θ₀ and store it with its manifest; never overwrites a stream."""
    spec, path, items = locate_stream(cfg, env, kind, seed)
    if path.exists():
        raise StreamError(f"{path} exists; streams are the record and are never regenerated")
    prompt_seed, sampling_seed = stream_seeds(kind, seed)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    distributions = None
    if kind == "alpaca_reference":
        chosen = _distinct(items, spec["responses"], prompt_seed)
        samples = reference_samples(
            chosen, env.encode_prompt, env.encode_text, end_id=env.end_ids[0],
            cap=spec["max_new_tokens"],
        )  # fmt: skip
    else:
        model = model if model is not None else env.new_model()
        indices = stream_prompt_indices(spec, len(items), prompt_seed)
        prompts = [env.encode_prompt(items[i]["text"]) for i in indices]
        sampling = spec["sampling"]
        samples, distributions = sample_responses(
            model, prompts, [items[i]["id"] for i in indices], sampling=sampling,
            end_ids=env.end_ids, pad_id=env.pad_id, seed=sampling_seed, batch_size=batch_size,
            zero_field=cfg.control != "sharpened_sampling" or kind.startswith("kl_"),
            record_distributions=kind == "soft_targets",
        )  # fmt: skip
    manifest = {
        "spec": spec,
        "seed": seed,
        "prompt_seed": prompt_seed,
        "sampling_seed": sampling_seed,
        "generation": {
            "do_sample": True,
            "num_beams": 1,
            "eos_token_id": env.end_ids,
            "pad_token_id": env.pad_id,
            "batch_size": batch_size,
            **spec.get("sampling", {}),
        },
        # The model's own generation config after neutralization; explicit fields override it.
        "model_generation_config": (
            None if model is None else _diff_dict(model.generation_config.to_diff_dict())
        ),
        "zero_field_checked": kind != "alpaca_reference"
        and (cfg.control != "sharpened_sampling" or kind.startswith("kl_")),
        "started_at": started,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": package_versions(),
        "hardware": hardware(),
    }
    if distributions is not None:
        manifest["distributions"] = save_distributions(path, distributions)
    save_stream(path, samples, manifest)
    return path


def load_stream_for(
    cfg: DictConfig, env: Environment, kind: str, seed: int | None
) -> tuple[list[Sample], dict[str, Any]]:
    """A stored stream and its manifest, checked against the spec the config implies."""
    spec, path, _ = locate_stream(cfg, env, kind, seed)
    samples = load_stream(path)
    manifest = load_manifest(path)
    if manifest["spec"] != json.loads(json.dumps(spec)):
        raise StreamError(f"{path}: manifest spec differs from the config's spec")
    if kind != "alpaca_reference":
        validate_samples(samples, env.end_ids, spec["sampling"]["max_new_tokens"])
    return samples, manifest


def save_distributions(path: Path, distributions: Sequence[torch.Tensor]) -> dict[str, Any]:
    """Store per-sample (length, vocab) fp32 log-probabilities next to a stream."""
    target = path.with_name(path.stem + ".distributions.safetensors")
    target.parent.mkdir(parents=True, exist_ok=True)
    lengths = torch.tensor([d.shape[0] for d in distributions])
    save_file({"logprobs": torch.cat(list(distributions)).contiguous(), "lengths": lengths},
              str(target))  # fmt: skip
    return {"file": target.name, "samples": len(distributions), "dtype": "float32"}


def load_distributions(path: Path) -> list[torch.Tensor]:
    target = path.with_name(path.stem + ".distributions.safetensors")
    stored = load_file(str(target))
    return list(stored["logprobs"].split(stored["lengths"].tolist()))


def _distinct(items: Sequence[dict], n: int, seed: int) -> list[dict]:
    import numpy as np

    if n > len(items):
        raise StreamError(f"Need {n} distinct reference examples, the pool has {len(items)}")
    order = np.random.default_rng(seed).permutation(len(items))[:n]
    return [items[i] for i in order]


def _diff_dict(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "transformers_version"}
