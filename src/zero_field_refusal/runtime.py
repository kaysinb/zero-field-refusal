"""What a run needs besides its config: tokenizer, models, prompt sets, and teacher streams.

``Environment`` collects the loaders so that the training and generation code can run unchanged
on a tiny CPU model in tests and on the pinned checkpoints on a GPU.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
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
    CHUNK_PREFIX,
    REFERENCE_KINDS,
    DistributionBuffer,
    Sample,
    StreamError,
    chunk_directory,
    chunk_seeds,
    load_fingerprints,
    load_manifest,
    load_stream,
    partial_directory,
    publish,
    published_chunks,
    reference_samples,
    replay_distributions,
    sample_responses,
    stale_partials,
    stream_layout,
    stream_path,
    stream_prompt_indices,
    stream_seeds,
    stream_spec,
    validate_samples,
    write_fingerprints,
    write_stream,
)
from .tracking import hardware, package_versions

# Prompt set of each stream kind; training-like streams answer the teacher prompts.
STREAM_PROMPT_SETS = {
    "kl_harmful": "harmful_kl",
    "kl_benign": "alpaca_benign_kl",
    "kl_monitor": "alpaca_monitor",
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
    seed = None if kind in REFERENCE_KINDS else seed
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
    if kind == "soft_targets" and stream_layout(cfg) == "appendable":
        raise StreamError("With appendable streams the soft-target control replays its targets; "
                          "a stored soft-target stream is not used")  # fmt: skip
    spec, path, items = locate_stream(cfg, env, kind, seed)
    if spec.get("layout") == "appendable":
        raise StreamError("An appendable stream grows by chunks (extend_stream)")
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
        if kind == "soft_targets":
            vocab = model.get_output_embeddings().weight.shape[0]
            buffer = DistributionBuffer(len(indices) * spec["sampling"]["max_new_tokens"], vocab)
        samples, _ = sample_responses(
            model, prompts, [items[i]["id"] for i in indices], sampling=spec["sampling"],
            end_ids=env.end_ids, pad_id=env.pad_id, seed=sampling_seed, batch_size=batch_size,
            zero_field=cfg.control != "sharpened_sampling" or kind.startswith("kl_"),
            on_chunk_distributions=buffer.append if kind == "soft_targets" else None,
        )  # fmt: skip
    manifest = _generation_manifest(
        cfg, env, kind, spec, seed, (prompt_seed, sampling_seed), model, batch_size, started
    )
    if kind == "soft_targets":
        manifest["distributions"] = buffer.save(partial)
        del buffer
    write_stream(partial, samples, manifest)
    publish(partial, path)
    return path


def _generation_manifest(
    cfg: DictConfig,
    env: Environment,
    kind: str,
    spec: dict[str, Any],
    seed: int | None,
    seeds: tuple[int, int],
    model: PreTrainedModel | None,
    batch_size: int,
    started: str,
) -> dict[str, Any]:
    prompt_seed, sampling_seed = seeds
    return {
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


def extend_stream(
    cfg: DictConfig,
    env: Environment,
    seed: int,
    responses: int,
    *,
    model: PreTrainedModel | None = None,
    batch_size: int,
) -> list[Path]:
    """Append chunks to a seed's appendable training stream until it holds ``responses``.

    Each chunk is sampled from θ₀ with the seeds of its own index and published atomically, so
    the chunks already there never change. A stream has one writer: every run that may extend a
    seed's stream runs on the machine that holds it, or the stream is generated before the runs.
    On that machine, writers take an exclusive lock on the stream's directory, and count the
    published chunks only once they hold it. Returns the new chunks' directories.
    """
    spec, path, items = locate_stream(cfg, env, "training", seed)
    if spec.get("layout") != "appendable":
        raise StreamError("extend_stream needs teacher.stream: appendable")
    with _locked(path):
        return _extend_locked(cfg, env, seed, responses, spec, path, items, model, batch_size)


@contextmanager
def _locked(stream: Path) -> Iterator[None]:
    """An exclusive lock on a stream's directory, held by one process at a time (flock)."""
    import fcntl

    stream.mkdir(parents=True, exist_ok=True)
    with (stream / ".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _extend_locked(cfg, env, seed, responses, spec, path, items, model, batch_size) -> list[Path]:
    have, need = published_chunks(path), -(-responses // spec["chunk_size"])
    created = []
    for index in range(have, need):
        final = chunk_directory(path, index)
        stale = stale_partials(final)
        if stale:
            raise StreamError(
                f"Partial directories of an interrupted generation exist: "
                f"{[str(s) for s in stale]}; remove them (remove_partial_streams) once no "
                "generation of this stream is running"
            )
        model = model if model is not None else env.new_model()
        prompt_seed, sampling_seed = chunk_seeds(seed, index)
        started = datetime.now(UTC).isoformat(timespec="seconds")
        indices = stream_prompt_indices(spec, len(items), prompt_seed)
        fingerprints: list[tuple[str, torch.Tensor]] = []  # for the soft-target replay
        samples, _ = sample_responses(
            model, [env.encode_prompt(items[i]["text"]) for i in indices],
            [items[i]["id"] for i in indices], sampling=spec["sampling"], end_ids=env.end_ids,
            pad_id=env.pad_id, seed=sampling_seed, batch_size=batch_size,
            fingerprints=fingerprints,
        )  # fmt: skip
        manifest = _generation_manifest(
            cfg, env, "training", spec, seed, (prompt_seed, sampling_seed), model, batch_size,
            started,
        )  # fmt: skip
        partial = partial_directory(final)
        partial.mkdir(parents=True)
        sizes = [min(batch_size, len(samples) - i) for i in range(0, len(samples), batch_size)]
        manifest["fingerprints"] = write_fingerprints(partial, fingerprints, sizes)
        manifest["device"] = device_name(next(model.parameters()).device)
        write_stream(partial, samples, {**manifest, "chunk": index})
        publish(partial, final)
        created.append(final)
    return created


class StreamPrefix:
    """Read access to the published chunks of a seed's appendable training stream.

    ``take(a, b)`` returns responses a … b−1 in stream order. Chunks are loaded and checked on
    first use (hash, spec, seed, index, count and the sample invariants), and only the two most
    recent stay in memory.
    """

    def __init__(self, cfg: DictConfig, env: Environment, seed: int) -> None:
        self.spec, self.path, _ = locate_stream(cfg, env, "training", seed)
        if self.spec.get("layout") != "appendable":
            raise StreamError("StreamPrefix needs teacher.stream: appendable")
        self.cfg, self.env, self.seed = cfg, env, seed
        self.size = self.spec["chunk_size"]
        self.chunks = published_chunks(self.path)
        self._loaded: dict[int, list[Sample]] = {}
        self._manifests: dict[int, dict[str, Any]] = {}  # of every chunk read and checked

    @property
    def available(self) -> int:
        return self.chunks * self.size

    def take(self, start: int, stop: int) -> list[Sample]:
        if not 0 <= start <= stop:
            raise ValueError(f"Invalid range [{start}, {stop})")
        if stop > self.available:  # chunks may have been appended since
            self.chunks = published_chunks(self.path)
        if stop > self.available:
            raise StreamError(
                f"{self.path}: responses up to {stop} needed, {self.available} published; "
                "extend the stream first (extend_stream)"
            )
        out: list[Sample] = []
        for index in range(start // self.size, -(-stop // self.size)):
            samples = self._chunk(index)
            base = index * self.size
            out += samples[max(start - base, 0) : min(stop - base, self.size)]
        return out

    def identity(self, responses: int) -> dict[str, Any]:
        """The record of the prefix of ``responses`` a run read: the hashes of its chunks, as
        checked when they were read, and a hash of them with the count."""
        needed = range(-(-responses // self.size))
        unread = [i for i in needed if i not in self._manifests]
        if unread:
            raise StreamError(f"{self.path}: chunks {unread} of the prefix were never read")
        chunks = [self._manifests[i]["sha256"] for i in needed]
        digest = hashlib.sha256(json.dumps({"chunks": chunks, "count": responses}).encode())
        return {"path": str(self.path.relative_to(self.env.data_root)), "count": responses,
                "sha256": digest.hexdigest(), "chunks": chunks}  # fmt: skip

    def fingerprints(self, index: int) -> list[tuple[str, torch.Tensor]]:
        """The distribution fingerprints of chunk ``index``, one per generation batch."""
        self.manifest(index)  # the chunk itself is checked first
        return load_fingerprints(chunk_directory(self.path, index))

    def manifest(self, index: int) -> dict[str, Any]:
        """The manifest of chunk ``index``, read and checked with its samples."""
        if index not in self._manifests:
            self._chunk(index)
        return self._manifests[index]

    def _chunk(self, index: int) -> list[Sample]:
        if index not in self._loaded:
            directory = chunk_directory(self.path, index)
            samples, manifest = load_stream(directory), load_manifest(directory)
            _check_manifest(self.cfg, self.env, "training", self.spec, directory, samples,
                            manifest)  # fmt: skip
            if manifest.get("chunk") != index or manifest["seed"] != self.seed:
                raise StreamError(f"{directory}: manifest names chunk {manifest.get('chunk')} "
                                  f"of seed {manifest['seed']}")  # fmt: skip
            seeds = manifest["prompt_seed"], manifest["sampling_seed"]
            if seeds != chunk_seeds(self.seed, index):
                raise StreamError(f"{directory}: seeds differ from the chunk's derived seeds")
            if len(samples) != self.size:
                raise StreamError(f"{directory}: {len(samples)} responses, chunks hold {self.size}")
            while len(self._loaded) >= 2:
                del self._loaded[min(self._loaded)]
            self._loaded[index] = samples
            self._manifests[index] = manifest
        return self._loaded[index]


# The largest accepted |replayed − stored| log-probability of a sampled token. A replay repeats the
# sampling's forward calls exactly, so it should be 0. 1e-5 is a margin for fp32 rounding alone:
# an order of magnitude below the largest token difference between two computation orders of the
# same weights in stage 0 (training path against generation path, 1.97e-4 on L40S at 1.5B).
REPLAY_MAX_ABS = 1e-5


def device_name(device: torch.device) -> str:
    """The GPU model of a CUDA device, or ``cpu``."""
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


class ReplayedSoftTargets:
    """Soft targets of an appendable stream (invariant 8), recomputed instead of stored.

    θ₀'s generation-path distributions are recomputed by replaying each generation batch of a
    chunk, as the batch it was sampled in (``replay_distributions``), and checked against the
    stored log-probabilities of the sampled tokens and against the fingerprint of the whole
    distributions recorded at sampling (``fingerprint``). Only an equal digest accepts a batch:
    a bound on a few projections bounds the error vector's norm, not each entry. A stream is
    therefore replayed on the GPU model that generated it, which is checked first. At most two
    generation batches stay in host memory, whatever the run's length (stored, 4000 steps of
    batch 5 would need ≈ 1 TB). Each replayed batch is recorded in ``records`` and passed to
    ``on_record`` before its checks, so a failing batch is recorded too: chunk, batch,
    responses, tokens, the sampled tokens' largest and mean deviation, whether the digest
    matched, and the projections' largest deviation (a diagnostic).
    """

    def __init__(
        self,
        prefix: StreamPrefix,
        model: PreTrainedModel,
        env: Environment,
        on_record: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.prefix, self.model, self.env, self.on_record = prefix, model, env, on_record
        self.records: list[dict[str, Any]] = []
        self._batches: dict[tuple[int, int], list[torch.Tensor]] = {}

    def take(self, start: int, stop: int) -> list[torch.Tensor]:
        """(length, vocab) fp32 log-probabilities of responses start … stop−1."""
        out = []
        for i in range(start, stop):
            chunk, offset = divmod(i, self.prefix.size)
            width = self.prefix.manifest(chunk)["generation"]["batch_size"]
            key = (chunk, offset // width)
            if key not in self._batches:
                self._replay(*key, width)
            out.append(self._batches[key][offset % width])
        return out

    def _replay(self, chunk: int, batch: int, width: int) -> None:
        generated_on = self.prefix.manifest(chunk).get("device")
        replayed_on = device_name(next(self.model.parameters()).device)
        if generated_on != replayed_on:
            raise StreamError(
                f"Chunk {chunk} was generated on {generated_on}, the replay runs on "
                f"{replayed_on}: a soft-target control replays its stream on the GPU model that "
                "generated it"
            )
        first = chunk * self.prefix.size + batch * width
        samples = self.prefix.take(first, min(first + width, (chunk + 1) * self.prefix.size))
        stored_digest, stored_projections = self.prefix.fingerprints(chunk)[batch]
        distributions, errors, (digest, projections) = replay_distributions(
            self.model, samples, sampling=self.prefix.spec["sampling"], end_ids=self.env.end_ids,
            pad_id=self.env.pad_id,
        )  # fmt: skip
        deviation = errors.max().item()
        if projections.shape != stored_projections.shape:
            raise StreamError(
                f"Chunk {chunk}, batch {batch}: the replay has {projections.shape[0]} positions, "
                f"the fingerprint {stored_projections.shape[0]}"
            )
        spread = (projections - stored_projections).abs().max().item()
        exact = digest == stored_digest
        self.records.append({"chunk": chunk, "batch": batch, "responses": len(samples),
                             "tokens": errors.numel(), "max_abs": deviation,
                             "mean_abs": errors.mean().item(), "digest_equal": exact,
                             "projection_max_abs": spread})  # fmt: skip
        if self.on_record is not None:
            self.on_record(self.records[-1])
        if not deviation <= REPLAY_MAX_ABS:
            raise StreamError(
                f"Replayed soft targets of chunk {chunk}, batch {batch} differ from the stored "
                f"log-probabilities by {deviation:.3g} > {REPLAY_MAX_ABS}"
            )
        if not exact:
            raise StreamError(
                f"Replayed distributions of chunk {chunk}, batch {batch} differ from the "
                f"sampling's: the digest differs (projections by {spread:.3g})"
            )
        while len(self._batches) >= 2:
            del self._batches[min(self._batches)]
        self._batches[(chunk, batch)] = distributions


def remove_partial_streams(cfg: DictConfig, env: Environment, kind: str, seed: int | None) -> list:
    """Delete partial directories of interrupted generations of one stream (of every chunk of an
    appendable stream); returns them."""
    import shutil

    spec, path, _ = locate_stream(cfg, env, kind, seed)
    if spec.get("layout") == "appendable":
        stale = sorted(path.glob(f".{CHUNK_PREFIX}*.partial-*")) if path.is_dir() else []
    else:
        stale = stale_partials(path)
    for directory in stale:
        shutil.rmtree(directory)
    return stale


def load_stream_for(
    cfg: DictConfig, env: Environment, kind: str, seed: int | None
) -> tuple[list[Sample], dict[str, Any]]:
    """A stored stream and its manifest, checked against the spec the config implies."""
    spec, path, _ = locate_stream(cfg, env, kind, seed)
    if spec.get("layout") == "appendable":
        raise StreamError("An appendable stream is read through StreamPrefix")
    samples = load_stream(path)
    manifest = load_manifest(path)
    _check_manifest(cfg, env, kind, spec, path, samples, manifest)
    return samples, {**manifest, "path": str(path.relative_to(env.data_root))}


def _check_manifest(
    cfg: DictConfig,
    env: Environment,
    kind: str,
    spec: dict[str, Any],
    path: Path,
    samples: Sequence[Sample],
    manifest: dict[str, Any],
) -> None:
    """The stored stream (or chunk) matches the config's spec and was sampled at the zero field."""
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


def _distinct(items: Sequence[dict], n: int, seed: int) -> list[dict]:
    import numpy as np

    if n > len(items):
        raise StreamError(f"Need {n} distinct reference examples, the pool has {len(items)}")
    order = np.random.default_rng(seed).permutation(len(items))[:n]
    return [items[i] for i in order]


def _diff_dict(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "transformers_version"}
