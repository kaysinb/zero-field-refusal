"""What a run needs besides its config: tokenizer, models, prompt sets, and teacher streams.

``Environment`` collects the loaders so that the training and generation code can run unchanged
on a tiny CPU model in tests and on the pinned checkpoints on a GPU.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from . import models
from .paths import data_root
from .teacher import (
    Sample,
    StreamError,
    load_manifest,
    load_stream,
    partial_directory,
    publish,
    reference_samples,
    sample_responses,
    stale_partials,
    stream_path,
    stream_prompt_indices,
    stream_seeds,
    stream_spec,
    validate_samples,
    write_distributions,
    write_stream,
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
    prompt_manifest_sha256: str | None = None  # the manifest of the prompt sets used

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
    manifest = data.prompt_sets_dir(cfg.data) / data.MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(f"{manifest} is missing; run scripts/prepare_data.py first")
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
        prompt_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
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
    """Sample a stream from θ₀ and publish it atomically with its manifest.

    The stream is written into a partial directory and renamed when complete, so an interrupted
    generation never leaves a stream that looks complete. Existing streams are never overwritten.
    """
    spec, path, items = locate_stream(cfg, env, kind, seed)
    if path.exists():
        raise StreamError(f"{path} exists; streams are the record and are never regenerated")
    stale = stale_partials(path)
    if stale:
        raise StreamError(
            f"Partial directories of an interrupted generation exist: {[str(s) for s in stale]}; "
            "remove them (remove_partial_streams) once no generation of this stream is running"
        )
    partial = partial_directory(path)
    partial.mkdir(parents=True)
    prompt_seed, sampling_seed = stream_seeds(kind, seed)
    started = datetime.now(UTC).isoformat(timespec="seconds")
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
        samples, distributions = sample_responses(
            model, prompts, [items[i]["id"] for i in indices], sampling=spec["sampling"],
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
    if kind == "soft_targets":
        manifest["distributions"] = write_distributions(partial, distributions)
        del distributions
    write_stream(partial, samples, manifest)
    publish(partial, path)
    return path


def remove_partial_streams(cfg: DictConfig, env: Environment, kind: str, seed: int | None) -> list:
    """Delete partial directories of interrupted generations of one stream; returns them."""
    import shutil

    _, path, _ = locate_stream(cfg, env, kind, seed)
    stale = stale_partials(path)
    for directory in stale:
        shutil.rmtree(directory)
    return stale


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
        zero_field = cfg.control != "sharpened_sampling" or kind.startswith("kl_")
        if zero_field and not manifest.get("zero_field_checked"):
            raise StreamError(f"{path}: not checked as a zero-field stream")
        neutral = {"eos_token_id", "pad_token_id"}
        if set(manifest.get("model_generation_config") or {}) - neutral:
            raise StreamError(f"{path}: the model's generation config was not neutral")
    return samples, {**manifest, "path": str(path.relative_to(env.data_root))}


def _distinct(items: Sequence[dict], n: int, seed: int) -> list[dict]:
    import numpy as np

    if n > len(items):
        raise StreamError(f"Need {n} distinct reference examples, the pool has {len(items)}")
    order = np.random.default_rng(seed).permutation(len(items))[:n]
    return [items[i] for i in order]


def _diff_dict(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "transformers_version"}
