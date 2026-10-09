"""Experiment configuration.

A config is a YAML file. It may list other config files under ``defaults``; they are merged in
order, then the file's own keys, then command-line overrides. Entries in ``defaults`` are paths
relative to the repository's ``configs/`` directory, with or without the ``.yaml`` suffix. A file
reached twice through ``defaults`` is merged once, at its first occurrence.

Overrides are applied in order:

- ``optimizer.lr=3e-5`` sets an existing key, so that a typo in a sweep fails loudly instead of
  silently adding a key;
- ``+eval.extra=1`` adds a new key;
- ``optimizer=sgd`` loads ``configs/optimizer/sgd.yaml`` (with its defaults) and replaces each
  top-level section it defines. A top-level section can be replaced only this way, from a file in
  the folder named after it.

``???`` marks a value that must be set before a run: the config loads, but ``config_hash`` fails
until the value is set. ``${mul:a,b}`` is the product of its arguments.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import InterpolationToMissingValueError, MissingMandatoryValue

from .paths import CONFIG_ROOT

# Keys that label a run without changing what it computes; excluded from the config hash.
NON_IDENTITY_KEYS = ("experiment", "name", "tracking")


def _mul(*factors: float) -> float:
    """``${mul:a,b,...}``: the product of the arguments, an int when all of them are ints."""
    if not all(isinstance(f, int | float) and not isinstance(f, bool) for f in factors):
        raise TypeError(f"mul expects numbers, got {factors}")
    return math.prod(factors)


OmegaConf.register_new_resolver("mul", _mul, replace=True)


def load_config(
    path: str | Path, overrides: Sequence[str] = (), *, root: Path = CONFIG_ROOT
) -> DictConfig:
    """Load a config (a file path or a name under ``root``) with its defaults and overrides."""
    cfg = _load(_locate(path, root), root)
    for item in overrides:
        cfg = _apply_override(cfg, item, root)
    OmegaConf.resolve(cfg)
    return cfg


def from_cli(argv: Sequence[str] | None = None) -> DictConfig:
    """Load the config named by the first argument; later ``key=value`` arguments override it."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or "=" in args[0]:
        raise SystemExit("usage: <script> CONFIG [key=value ...]")
    return load_config(args[0], args[1:])


def config_hash(cfg: DictConfig) -> str:
    """Short stable hash of everything that determines what a run computes.

    Raises ``ValueError`` while any value is still ``???``.
    """
    try:
        data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except (MissingMandatoryValue, InterpolationToMissingValueError) as exc:
        raise ValueError(f"Set these config values before a run: {_missing_keys(cfg)}") from exc
    for key in NON_IDENTITY_KEYS:
        data.pop(key, None)
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _missing_keys(cfg: DictConfig) -> str:
    resolved = copy.deepcopy(cfg)
    OmegaConf.resolve(resolved)  # an interpolation of a ??? value becomes ??? itself
    return ", ".join(sorted(OmegaConf.missing_keys(resolved)))


def _with_suffix(name: str) -> str:
    # Not Path.with_suffix: names such as "qwen2.5-1.5b" contain dots.
    return name if name.endswith((".yaml", ".yml")) else f"{name}.yaml"


def _in_root(name: str, root: Path) -> Path:
    path = (root / _with_suffix(name)).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {name!r} under {root}")
    return path


def _locate(path: str | Path, root: Path) -> Path:
    candidate = Path(_with_suffix(str(path)))
    if candidate.is_file():
        return candidate.resolve()
    return _in_root(str(path), root)


def _load(
    path: Path, root: Path, chain: tuple[Path, ...] = (), seen: set[Path] | None = None
) -> DictConfig:
    """``path`` merged over its defaults; files in ``seen`` were merged earlier and are skipped."""
    if path in chain:
        cycle = " -> ".join(p.name for p in (*chain, path))
        raise ValueError(f"Cyclic defaults: {cycle}")
    seen = set() if seen is None else seen
    if path in seen:
        return OmegaConf.create()
    seen.add(path)
    node = OmegaConf.load(path)
    if not isinstance(node, DictConfig):
        raise TypeError(f"{path.name} must contain a mapping")
    defaults = node.pop("defaults", [])
    if not isinstance(defaults, ListConfig | list):
        raise TypeError(f"{path.name}: defaults must be a list of config names, got {defaults!r}")
    merged = OmegaConf.create()
    for entry in defaults:
        child = _in_root(str(entry), root)
        merged = OmegaConf.merge(merged, _load(child, root, (*chain, path), seen))
    return OmegaConf.merge(merged, node)


def _apply_override(cfg: DictConfig, item: str, root: Path) -> DictConfig:
    key, sep, value = item.partition("=")
    if not sep or key.startswith("++") or not key.lstrip("+"):
        raise ValueError(f"Override must look like key=value or +key=value: {item!r}")
    if key.startswith("+"):
        return OmegaConf.merge(cfg, OmegaConf.from_dotlist([item[1:]]))
    if "." not in key and isinstance(cfg._get_node(key), DictConfig):
        return _replace_sections(cfg, _load(_in_root(f"{key}/{value}", root), root))
    if not _exists(cfg, key):
        raise KeyError(f"Unknown config key {key!r}; use +{key}=... to add a new key")
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist([item]))


def _exists(cfg: DictConfig, key: str) -> bool:
    """Whether ``key`` is in the config, including keys whose value is ``???``."""
    parent_key, _, leaf = key.rpartition(".")
    parent = (
        OmegaConf.select(cfg, parent_key, throw_on_resolution_failure=False) if parent_key else cfg
    )
    return isinstance(parent, DictConfig) and parent._get_node(leaf) is not None


def _replace_sections(cfg: DictConfig, group: DictConfig) -> DictConfig:
    """Group override: each top-level section of ``group`` replaces the one in ``cfg``."""
    for key in group:
        cfg[key] = group._get_node(key)  # the node itself: ??? and ${...} stay unresolved
    return cfg
