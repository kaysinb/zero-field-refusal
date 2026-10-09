"""Pinned prompt sets.

``build_prompt_sets`` selects every set from datasets pinned by revision, writes one JSONL file per
set and a manifest (config, hashes, revisions, original row indices, dedup statistics) into
``DATA_DIR/prompts/<config hash>/``. ``load_prompt_set`` reads a set back after checking its hash
against the manifest, so a run always sees exactly the prompts that were selected for its config.

Items are dicts with at least ``id`` (stable, e.g. ``"advbench:123"``, the original row index) and
``text``. All randomness comes from ``derive_seed(data.selection_seed, <name>, ...)``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from .paths import data_root
from .seeding import derive_seed

SET_NAMES = (
    "alpaca_teacher",
    "alpaca_benign_kl",
    "alpaca_heldout",
    "alpaca_reference_pool",
    "harmful_kl",
    "harmful_judged",
    "harmful_calibration",
    "factual_qa",
    "mmlu",
)
MANIFEST = "manifest.json"
JACCARD_THRESHOLD = 0.8
ANSWER_SUFFIX = " Answer briefly."

# Columns of the pinned revisions that the selection reads. HarmBench `standard` has `prompt` and
# `category`; AdvBench has `prompt` and `target`; Alpaca has `instruction`, `input`, `output` and
# `text`; MMLU has `question`, `subject`, `choices` and `answer`.
HARMFUL_TEXT_COLUMN = {"walledai/HarmBench": "prompt", "walledai/AdvBench": "prompt"}

LoadRows = Callable[[Mapping[str, Any]], list[dict]]


def _container(data_cfg: Any) -> dict:
    if isinstance(data_cfg, DictConfig):
        container = OmegaConf.to_container(data_cfg, resolve=True, throw_on_missing=True)
    else:
        container = json.loads(json.dumps(data_cfg))
    if not isinstance(container, dict):
        raise TypeError("The data config must be a mapping")
    return container


def data_config_hash(data_cfg: Any) -> str:
    """First 12 hex digits of the sha256 of the canonical JSON of the resolved data config."""
    blob = json.dumps(_container(data_cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def prompt_sets_dir(data_cfg: Any, data_dir: Path | None = None) -> Path:
    """``DATA_DIR/prompts/<config hash>/``: where the sets of this data config live."""
    return (data_dir or data_root()) / "prompts" / data_config_hash(data_cfg)


# ---------------------------------------------------------------------------------- selection


def allocate_largest_remainder(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Split ``total`` across the keys in proportion to ``sizes`` (largest remainder method).

    Ties go to the earlier key. No key gets more than its size.
    """
    population = sum(sizes.values())
    if total > population:
        raise ValueError(f"Cannot allocate {total} items from only {population}")
    if total == 0:
        return dict.fromkeys(sizes, 0)
    alloc = {key: total * size // population for key, size in sizes.items()}
    # Stable sort: equal remainders keep the key order.
    by_remainder = sorted(sizes, key=lambda key: -(total * sizes[key] % population))
    for key in by_remainder[: total - sum(alloc.values())]:
        alloc[key] += 1
    return alloc


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))
    return " ".join(text.split())


def _deduplicate(candidates: Sequence[dict], enabled: bool) -> tuple[list[dict], dict[str, Any]]:
    """Keep the first of each group of duplicate prompts; ``candidates`` are in source order."""
    kept: list[dict] = []
    seen: set[str] = set()
    word_sets: list[frozenset[str]] = []
    stats: dict[str, Any] = {
        "enabled": enabled,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "raw": len(candidates),
        "dropped_empty": 0,
        "dropped_exact": 0,
        "dropped_near_duplicate": 0,
        "by_source": {},
    }
    for item in candidates:
        counts = stats["by_source"].setdefault(item["source"], {"raw": 0, "kept": 0})
        counts["raw"] += 1
        norm = _normalize(item["text"])
        words = frozenset(norm.split())
        if not words:
            stats["dropped_empty"] += 1
            continue
        if enabled:
            if norm in seen:
                stats["dropped_exact"] += 1
                continue
            if any(len(words & w) / len(words | w) >= JACCARD_THRESHOLD for w in word_sets):
                stats["dropped_near_duplicate"] += 1
                continue
        seen.add(norm)
        word_sets.append(words)
        kept.append(item)
        counts["kept"] += 1
    stats["kept"] = len(kept)
    return kept, stats


def _row_index(row: Mapping[str, Any]) -> int:
    return int(row["row_index"])


def _select_alpaca(cfg: Mapping, rows: Sequence[dict], seed: int) -> dict[str, list[dict]]:
    if cfg.get("filter") != "no_input":
        raise ValueError(f"Unsupported alpaca filter {cfg.get('filter')!r}; only 'no_input'")
    eligible = [
        row
        for row in rows
        if not row["input"].strip() and row["instruction"].strip() and row["output"].strip()
    ]
    sizes = {
        "alpaca_teacher": cfg["teacher_prompts"],
        "alpaca_benign_kl": cfg["benign_kl_prompts"],
        "alpaca_heldout": cfg["heldout_references"],
    }
    if sum(sizes.values()) > len(eligible):
        raise ValueError(
            f"alpaca: {len(eligible)} eligible rows, but the config needs {sum(sizes.values())}"
        )
    order = np.random.default_rng(derive_seed(seed, "alpaca")).permutation(len(eligible))
    chosen = [eligible[i] for i in order]
    out: dict[str, list[dict]] = {}
    start = 0
    for name, count in [
        *sizes.items(),
        ("alpaca_reference_pool", len(chosen) - sum(sizes.values())),
    ]:
        part = chosen[start : start + count]
        start += count
        items = []
        for row in part:
            item = {"id": f"alpaca:{_row_index(row)}", "text": row["instruction"].strip()}
            if name in ("alpaca_heldout", "alpaca_reference_pool"):
                item["reference"] = row["output"].strip()
            items.append(item)
        out[name] = items
    return out


def _harmful_candidates(source: Mapping, rows: Sequence[dict]) -> list[dict]:
    dataset = source["dataset"]
    if dataset not in HARMFUL_TEXT_COLUMN:
        raise ValueError(f"No known text column for harmful source {dataset!r}")
    label = dataset.split("/")[-1].lower()
    items = []
    for row in sorted(rows, key=_row_index):
        item = {
            "id": f"{label}:{_row_index(row)}",
            "text": row[HARMFUL_TEXT_COLUMN[dataset]].strip(),
            "source": label,
        }
        if "category" in row:
            item["category"] = row["category"]
        items.append(item)
    return items


def _select_harmful(
    cfg: Mapping, rows_by_source: Sequence[Sequence[dict]], seed: int
) -> tuple[dict[str, list[dict]], dict[str, Any]]:
    candidates = [
        item
        for source, rows in zip(cfg["sources"], rows_by_source, strict=True)
        for item in _harmful_candidates(source, rows)
    ]
    kept, stats = _deduplicate(candidates, bool(cfg.get("deduplicate", True)))
    by_source: dict[str, list[dict]] = {}
    for item in kept:
        by_source.setdefault(item["source"], []).append(item)
    sizes = {label: len(items) for label, items in by_source.items()}
    kl = allocate_largest_remainder(sizes, cfg["kl_prompts"])
    judged = allocate_largest_remainder(sizes, cfg["judged_prompts"])
    parts: dict[str, list[dict]] = {
        "harmful_kl": [],
        "harmful_judged": [],
        "harmful_calibration": [],
    }
    for label, items in by_source.items():
        if kl[label] + judged[label] > len(items):
            raise ValueError(f"harmful: not enough {label} prompts for the KL and judged sets")
        order = np.random.default_rng(derive_seed(seed, "harmful", label)).permutation(len(items))
        shuffled = [items[i] for i in order]
        parts["harmful_kl"] += shuffled[: kl[label]]
        parts["harmful_judged"] += shuffled[kl[label] : kl[label] + judged[label]]
        parts["harmful_calibration"] += shuffled[kl[label] + judged[label] :]
    # Report in source order, then original row order.
    position = {item["id"]: n for n, item in enumerate(kept)}
    for items in parts.values():
        items.sort(key=lambda item: position[item["id"]])
    stats["allocation"] = {"harmful_kl": kl, "harmful_judged": judged}
    return parts, stats


def _select_mmlu(cfg: Mapping, rows: Sequence[dict], seed: int) -> list[dict]:
    if cfg.get("stratify") != "subject":
        raise ValueError(f"Unsupported mmlu stratification {cfg.get('stratify')!r}")
    by_subject: dict[str, list[dict]] = {}
    for row in sorted(rows, key=_row_index):
        by_subject.setdefault(row["subject"], []).append(row)
    subjects = sorted(by_subject)
    alloc = allocate_largest_remainder({s: len(by_subject[s]) for s in subjects}, cfg["questions"])
    chosen: list[dict] = []
    for subject in subjects:
        pool = by_subject[subject]
        order = np.random.default_rng(derive_seed(seed, "mmlu", subject)).permutation(len(pool))
        chosen += [pool[i] for i in order[: alloc[subject]]]
    items = []
    for row in sorted(chosen, key=_row_index):
        item = {
            "id": f"mmlu:{_row_index(row)}",
            "question": row["question"],
            "choices": [str(c) for c in row["choices"]],
            "answer": int(row["answer"]),
            "subject": row["subject"],
        }
        if len(item["choices"]) != 4 or item["answer"] not in range(4):
            raise ValueError(f"{item['id']}: expected 4 choices and an answer in 0..3")
        item["text"] = mmlu_prompt(item)
        items.append(item)
    return items


def mmlu_prompt(item: Mapping[str, Any]) -> str:
    a, b, c, d = item["choices"]
    return (
        f"{item['question']}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
        "Answer with the letter of the correct option."
    )


# ---------------------------------------------------------------------------- factual questions


def _read_facts(name: str) -> bytes:
    return (resources.files("zero_field_refusal") / "resources" / f"{name}.jsonl").read_bytes()


def load_facts(name: str) -> list[dict]:
    """The factual-question resource ``resources/<name>.jsonl`` shipped with the package."""
    return [json.loads(line) for line in _read_facts(name).decode("utf-8").splitlines() if line]


def _apostrophes(text: str) -> str:
    return text.replace("’", "'").replace("‘", "'")


def factual_answer_correct(text: str, answers: Sequence[str]) -> bool:
    """Whether an alias occurs in ``text`` as a whole word or number (case-insensitive).

    An alias must not follow a word character or ``.`` (so ``9`` is not in ``19`` or ``3.9``) and
    must not be followed by a word character.
    """
    haystack = _apostrophes(text).lower()
    return any(
        re.search(r"(?<![\w.])" + re.escape(_apostrophes(alias).lower()) + r"(?!\w)", haystack)
        for alias in answers
    )


# ----------------------------------------------------------------------------------- building


def _load_hf_rows(source: Mapping[str, Any]) -> list[dict]:
    from datasets import load_dataset
    from datasets.exceptions import DatasetNotFoundError

    try:
        dataset = load_dataset(
            source["dataset"],
            source.get("config"),
            split=source.get("split", "train"),
            revision=source["revision"],
        )
    except DatasetNotFoundError as exc:
        raise RuntimeError(
            f"Cannot load {source['dataset']}: {exc} Gated datasets need an accepted license on "
            "the Hub and HF_TOKEN in the environment or .env."
        ) from exc
    return [{**row, "row_index": index} for index, row in enumerate(dataset)]


def _jsonl(items: Sequence[Mapping[str, Any]]) -> bytes:
    lines = (json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items)
    return "".join(line + "\n" for line in lines).encode("utf-8")


def _source_entry(source: Mapping[str, Any]) -> dict[str, Any]:
    entry = {"dataset": source["dataset"], "revision": source["revision"]}
    if source.get("config"):
        entry["config"] = source["config"]
    entry["split"] = source.get("split", "train")
    return entry


def build_prompt_sets(
    data_cfg: Any, out_dir: Path | None = None, *, load_rows: LoadRows | None = None
) -> Path:
    """Select every prompt set, write ``<set>.jsonl`` files and the manifest; return its path.

    ``load_rows(source)`` returns the source's rows as dicts, each with its original ``row_index``;
    it defaults to loading the pinned revision from the Hugging Face Hub.
    """
    cfg = _container(data_cfg)
    load = load_rows or _load_hf_rows
    out_dir = out_dir or prompt_sets_dir(cfg)
    seed = cfg["selection_seed"]

    alpaca_source = {"split": "train", **cfg["alpaca"]}
    harmful_sources = [{"split": "train", **s} for s in cfg["harmful"]["sources"]]
    mmlu_source = dict(cfg["benchmark"])

    sets: dict[str, list[dict]] = _select_alpaca(cfg["alpaca"], load(alpaca_source), seed)
    harmful, dedup = _select_harmful(
        cfg["harmful"], [load(source) for source in harmful_sources], seed
    )
    sets.update(harmful)

    facts_name = cfg["factual_qa"]["set"]
    facts = load_facts(facts_name)
    minimum = cfg["factual_qa"]["min_questions"]
    if len(facts) < minimum:
        raise ValueError(
            f"{facts_name} has {len(facts)} questions; the config needs at least {minimum}"
        )
    sets["factual_qa"] = [
        {
            "id": fact["id"],
            "text": fact["question"] + ANSWER_SUFFIX,
            "answers": fact["answers"],
            "category": fact["category"],
        }
        for fact in facts
    ]
    sets["mmlu"] = _select_mmlu(cfg["benchmark"], load(mmlu_source), seed)

    sources = {
        "alpaca": [_source_entry(alpaca_source)],
        "harmful": [_source_entry(s) for s in harmful_sources],
        "factual_qa": [
            {
                "resource": f"{facts_name}.jsonl",
                "sha256": hashlib.sha256(_read_facts(facts_name)).hexdigest(),
            }
        ],
        "mmlu": [_source_entry(mmlu_source)],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / MANIFEST).unlink(missing_ok=True)  # the manifest marks a complete build
    entries: dict[str, Any] = {}
    for name in SET_NAMES:
        items = sets[name]
        raw = _jsonl(items)
        (out_dir / f"{name}.jsonl").write_bytes(raw)
        family = name.split("_")[0] if name.startswith(("alpaca", "harmful")) else name
        row_indices: dict[str, list[int]] = {}
        for item in items:
            label, _, index = item["id"].partition(":")
            row_indices.setdefault(label, []).append(int(index))
        entries[name] = {
            "count": len(items),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "sources": sources[family],
            "row_indices": row_indices,
        }
        if family == "harmful":
            entries[name]["dedup"] = dedup
    manifest = {
        "config": cfg,
        "config_hash": data_config_hash(cfg),
        "versions": {
            "datasets": importlib.metadata.version("datasets"),
            "huggingface_hub": importlib.metadata.version("huggingface_hub"),
        },
        "created_utc": datetime.now(UTC).isoformat(),
        "sets": entries,
    }
    path = out_dir / MANIFEST
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------ loading


def load_prompt_set(name: str, data_cfg: Any, data_dir: Path | None = None) -> list[dict]:
    """Read a prompt set after verifying the manifest against the config and the file's hash."""
    if name not in SET_NAMES:
        raise ValueError(f"Unknown prompt set {name!r}; choose from {SET_NAMES}")
    directory = prompt_sets_dir(data_cfg, data_dir)
    hint = "Run scripts/prepare_data.py with this config to (re)build the prompt sets."
    manifest_path = directory / MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No prompt-set manifest at {manifest_path}. {hint}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = data_config_hash(data_cfg)
    if manifest.get("config_hash") != expected:
        raise ValueError(
            f"{manifest_path} was built for config hash {manifest.get('config_hash')}, "
            f"not {expected}. {hint}"
        )
    entry = manifest.get("sets", {}).get(name)
    path = directory / f"{name}.jsonl"
    if entry is None:
        raise ValueError(f"{manifest_path} has no entry for {name}. {hint}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. {hint}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"{path} does not match the sha256 in its manifest. {hint}")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()]
