"""Run directories and metric logging.

Each run owns ``<runs root>/<experiment>/<run id>/``:

- ``config.yaml``: the resolved config;
- ``meta.json``: identity and provenance (config hash, command-line arguments, git commit, lockfile
  hash, package versions, GPUs, source snapshot and container image, input hashes, W&B run,
  status: ``running``, ``completed``, ``failed``, or ``diverged`` when training stopped on a
  non-finite loss);
- ``git.diff``: uncommitted changes and untracked files, when the working tree is dirty;
- ``metrics.jsonl``: one JSON object per ``log`` call;
- ``summary.json``: final values;
- any outputs the run saves under ``run.path(...)``.

These files are the source of truth for analysis. NumPy and PyTorch scalars and arrays are stored as
numbers and lists; any other value JSON cannot represent raises an error instead of being
stringified. NaN and infinities are written as Python's ``json`` writes them (``NaN``,
``Infinity``), which Python and pandas read back but strict JSON parsers reject.

Weights & Biases, when enabled in the config, mirrors config, metrics and summary for browsing; its
entity, API key and mode (``WANDB_MODE``) come from the environment. Nothing else is sent: no
artifacts, weights or response texts. It records no hostname, paths,
git state, code or console output. A failure to start it is recorded in ``meta.json`` and the run
goes on, unless ``tracking.wandb.required`` is set.

In a distributed job only rank 0 writes (``RANK`` decides unless ``is_main`` is given). With
``WORLD_SIZE`` > 1 and no ``RUN_ID``, rank 0 generates the run id and broadcasts it, so the process
group must be initialized before ``Run`` is created on every rank.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import subprocess
import sys
import time
import warnings
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .config import config_hash
from .paths import REPO_ROOT, runs_root

TRACKED_PACKAGES = ("torch", "transformers", "peft", "datasets", "accelerate", "numpy", "wandb")

# Launcher-provided identities of the executed source and environment. A development run on a Pod
# executes a source snapshot without Git metadata; its snapshot ID is the code provenance.
LAUNCH_ENV = {"source_snapshot_id": "SOURCE_SNAPSHOT_ID", "container_image": "CONTAINER_IMAGE"}

# Keys every metrics.jsonl record has; a metric must not shadow them.
RESERVED_KEYS = ("step", "time_s")

# A run id is also a directory name and the W&B run id, which is limited to 64 characters.
_RUN_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")


class Diverged(RuntimeError):
    """Training stopped on a non-finite loss: an outcome of the learning rate, not a failure.

    The run ends with status ``diverged``; it is not rerun (docs/operations.md, "Launching runs").
    """

    run_status = "diverged"


class Run:
    """One run directory; use as a context manager so the status is recorded on exit."""

    def __init__(
        self, cfg: DictConfig, *, run_id: str | None = None, is_main: bool | None = None
    ) -> None:
        self.cfg = cfg
        self.is_main = _rank() == 0 if is_main is None else is_main
        self.experiment = _slug(str(cfg.get("experiment") or "")) or "scratch"
        self.config_hash = config_hash(cfg)
        self.run_id = run_id or os.environ.get("RUN_ID") or self._shared_id()
        if not _RUN_ID.fullmatch(self.run_id):
            raise ValueError(f"Run id {self.run_id!r} does not match {_RUN_ID.pattern}")
        self.dir = runs_root() / self.experiment / self.run_id
        self._start = time.monotonic()
        self._summary: dict[str, Any] = {}
        self._wandb = None
        self._finished = False

        tracking = cfg.get("tracking") or {}
        git = git_state()
        if tracking.get("require_clean_git") and (git["commit"] is None or git["dirty"]):
            raise RuntimeError("tracking.require_clean_git is set: commit all changes first")
        if not self.is_main:
            return

        self.dir.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(cfg, self.dir / "config.yaml", resolve=True)
        if git["dirty"]:
            (self.dir / "git.diff").write_text(git_diff())
        self._meta: dict[str, Any] = {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "name": cfg.get("name"),
            "config_hash": self.config_hash,
            "status": "running",
            "started_at": _now(),
            "finished_at": None,
            "argv": sys.argv[1:],
            "git": git,
            "launch": {key: os.environ.get(var) or None for key, var in LAUNCH_ENV.items()},
            "uv_lock_sha256": _sha256(REPO_ROOT / "uv.lock"),
            "python": platform.python_version(),
            "packages": package_versions(),
            "hardware": hardware(),
            "wandb": None,
        }
        _write_json(self.dir / "meta.json", self._meta)

        wandb_cfg = tracking.get("wandb") or {}
        if wandb_cfg.get("enabled"):
            self._start_wandb(wandb_cfg)

    def path(self, *parts: str) -> Path:
        """A path inside the run directory; the main process creates its parent directories."""
        target = self.dir.joinpath(*parts)
        if self.is_main:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def log(self, metrics: Mapping[str, Any], step: int) -> None:
        """Append one record; steps may repeat or go back (e.g. an evaluation pass)."""
        if not self.is_main:
            return
        reserved = sorted(set(metrics) & set(RESERVED_KEYS))
        if reserved:
            raise ValueError(f"Reserved metric names {reserved}: pass step= and leave time_s out")
        values = _plain(metrics)
        step = int(step)
        record = {"step": step, "time_s": round(time.monotonic() - self._start, 3), **values}
        line = json.dumps(record)  # before opening the file: a bad value leaves no partial line
        with (self.dir / "metrics.jsonl").open("a") as f:
            f.write(line + "\n")
        if self._wandb is not None:
            self._wandb.log({**_indexed(values), "step": step})

    def summarize(self, values: Mapping[str, Any]) -> None:
        """Add final values to ``summary.json`` (later calls update earlier keys)."""
        if not self.is_main:
            return
        summary = {**self._summary, **_plain(values)}
        _write_json(self.dir / "summary.json", summary)
        self._summary = summary
        if self._wandb is not None:
            self._wandb.summary.update(summary)

    def record_inputs(self, inputs: Mapping[str, Any]) -> None:
        """Add input identities (stream and prompt-set hashes, paths) to ``meta.json``."""
        if not self.is_main:
            return
        _merge(self._meta.setdefault("inputs", {}), _plain(inputs))
        _write_json(self.dir / "meta.json", self._meta)

    def finish(self, status: str = "completed") -> None:
        if not self.is_main or self._finished:
            return
        self._finished = True
        self._meta.update(status=status, finished_at=_now())
        _write_json(self.dir / "meta.json", self._meta)
        if self._wandb is not None:
            self._wandb.finish(exit_code=0 if status == "completed" else 1)

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        # An exception may name the run's outcome (Diverged); any other one is a failure.
        self.finish("completed" if exc_type is None else getattr(exc, "run_status", "failed"))

    def _shared_id(self) -> str:
        """A new run id, the same on every process of a distributed job."""
        if _world_size() <= 1:
            return self._new_id()
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError(
                "WORLD_SIZE > 1: initialize the process group before creating Run, so that rank 0 "
                "can broadcast the run id (or set RUN_ID for every process)"
            )
        box = [self._new_id() if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def _new_id(self) -> str:
        """``<UTC time>-[<name>-]<config hash>-<random>``, at most 64 characters."""
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        name = _slug(str(self.cfg.get("name") or "")[:34])
        label = [name] if name else []
        return "-".join([stamp, *label, self.config_hash[:8], secrets.token_hex(2)])

    def _start_wandb(self, wandb_cfg: Mapping[str, Any]) -> None:
        """Start the W&B mirror; a failure is recorded and only raised if W&B is required."""
        try:
            self._wandb = self._init_wandb(wandb_cfg)
        except Exception as exc:
            self._meta["wandb"] = {"error": repr(exc)}
            if wandb_cfg.get("required"):
                self.finish("failed")
                raise
            warnings.warn(f"W&B mirror disabled, the run goes on: {exc!r}", stacklevel=3)
        else:
            self._meta["wandb"] = {"id": self._wandb.id, "project": self._wandb.project}
        _write_json(self.dir / "meta.json", self._meta)

    def _init_wandb(self, wandb_cfg: Mapping[str, Any]) -> Any:
        import wandb

        config = OmegaConf.to_container(self.cfg, resolve=True)
        # No hostname, username, paths, git state, code, installed packages or console output.
        # The x_ settings are private W&B API, hence the wandb version cap in pyproject.toml.
        settings = wandb.Settings(
            host="worker",
            x_disable_meta=True,
            x_disable_machine_info=True,
            disable_git=True,
            disable_code=True,
            save_code=False,
            x_save_requirements=False,
            console="off",
        )
        run = wandb.init(
            project=wandb_cfg.get("project"),
            group=self.experiment,
            name=self.run_id,
            id=self.run_id,
            tags=list(wandb_cfg.get("tags") or []),
            config={**config, "config_hash": self.config_hash},
            dir=str(self.dir),
            settings=settings,
        )
        # Our step is a metric, not W&B's own step counter, so out-of-order steps are kept.
        run.define_metric("step")
        run.define_metric("*", step_metric="step")
        return run


def git_state(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Current commit and whether the working tree is dirty: any uncommitted change or untracked
    file (``worktree_changes``, whatever ``status.showUntrackedFiles``, ``.git/info/exclude`` and
    ``core.excludesFile`` say).

    Both are None unless ``repo`` is the top level of a git checkout (not a folder inside another
    repository).
    """
    try:
        top = _git(repo, "rev-parse", "--show-toplevel").strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    if Path(top).resolve() != repo.resolve():
        return {"commit": None, "dirty": None}
    try:
        commit: str | None = _git(repo, "rev-parse", "HEAD").strip()
    except subprocess.CalledProcessError:
        commit = None
    changes = worktree_changes(repo)
    return {"commit": commit, "dirty": bool(changes["uncommitted"] or changes["untracked"])}


# A Pod driver's check, as a ``python -c`` program run through ``project-run``, that its source
# checkout is clean as the runs read it (``git_state``: no uncommitted change, no untracked file
# that its .gitignore does not ignore, whatever the Git configuration says). Shared by the drivers
# of scripts/study2_short_campaigns.py and scripts/fa_control_diagnostic.py.
CLEAN_CHECKOUT = (
    "import sys; from zero_field_refusal.tracking import git_state; "
    "sys.exit(None if git_state()['dirty'] is False else 'the source checkout is not clean "
    "(tracking.git_state): an uncommitted change or an untracked file')"
)


def worktree_changes(repo: Path = REPO_ROOT) -> dict[str, list[str]]:
    """How the checkout at ``repo`` (its top level) departs from its HEAD: ``uncommitted``, the
    tracked paths whose staged or working-tree content differs from HEAD (every tracked path
    before the first commit); ``untracked``, the files Git does not track and no ``.gitignore``
    of the checkout ignores. The one reading of a dirty tree, for the runs (``git_state``) and the
    versions record (``code_changes``).

    What does not change it: ``status.showUntrackedFiles`` (``git ls-files`` lists untracked
    files, which ``git status`` hides under ``no``), ``.git/info/exclude`` and the user's
    ``core.excludesFile`` (only the checkout's ``.gitignore`` files ignore), ``core.quotePath``
    (paths are read unquoted) and ``diff.renames`` (a rename is both paths). What still hides a
    change, as from ``git status``: index flags set on purpose (``--assume-unchanged``,
    ``--skip-worktree``, a sparse checkout) and a mode change under ``core.fileMode`` false.
    Raises ``subprocess.CalledProcessError`` for no checkout."""
    try:
        _git(repo, "rev-parse", "--verify", "--quiet", "HEAD")
    except subprocess.CalledProcessError:  # no commit yet: everything tracked is new
        changed = _git(repo, "ls-files", "-z", "--cached").split("\0")
    else:
        changed = [*_git(repo, "diff", "--name-only", "-z", "--no-renames", "HEAD").split("\0"),
                   *_git(repo, "diff", "--name-only", "-z", "--no-renames", "--cached",
                         "HEAD").split("\0")]  # fmt: skip
    untracked = _git(repo, "ls-files", "-z", "--others",
                     "--exclude-per-directory=.gitignore").split("\0")  # fmt: skip
    return {"uncommitted": sorted(set(changed) - {""}), "untracked": sorted(set(untracked) - {""})}


def git_diff(repo: Path = REPO_ROOT) -> str:
    """``git diff HEAD`` (a patch ``git apply`` accepts) followed by the list of untracked files
    (``worktree_changes``)."""
    try:
        diff = _git(repo, "diff", "HEAD")
    except subprocess.CalledProcessError:  # no commit yet
        diff = ""
    untracked = "".join(f"{name}\n" for name in worktree_changes(repo)["untracked"])
    return f"{diff}# Untracked files:\n{untracked}"


def code_changes(commit: str, repo: Path = REPO_ROOT) -> dict[str, list[str]] | None:
    """How the checkout differs from ``commit``: ``committed``, the paths changed between it and
    HEAD; ``uncommitted`` and ``untracked`` as ``worktree_changes`` reads them (the tracked paths
    with uncommitted changes, the files Git does not track and no ``.gitignore`` ignores). None
    when ``commit`` is neither HEAD nor an ancestor of it (or ``repo`` is no checkout)."""
    try:
        _git(repo, "merge-base", "--is-ancestor", commit, "HEAD")
        committed = _git(repo, "diff", "--name-only", "-z", "--no-renames", commit, "HEAD")
        changes = worktree_changes(repo)
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"committed": sorted(set(committed.split("\0")) - {""}), **changes}


def tracked(path: str, repo: Path = REPO_ROOT) -> bool:
    """Whether ``path`` (relative to ``repo``) is a file Git tracks."""
    try:
        _git(repo, "ls-files", "--error-unmatch", "--", path)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def package_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def hardware() -> dict[str, Any]:
    """GPUs, driver and CUDA versions, read without initializing CUDA in this process.

    Inference engines such as vLLM start workers that fail if the parent has initialized CUDA.
    ``gpus`` lists every GPU on the machine; ``cuda_visible_devices`` says which ones the run sees.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    rows = [line.rsplit(",", 1) for line in out.splitlines() if "," in line]
    try:
        import torch
    except ImportError:
        torch_cuda = None
    else:
        torch_cuda = torch.version.cuda
    return {
        "gpus": [name.strip() for name, _ in rows],
        "driver": rows[0][1].strip() if rows else None,
        "torch_cuda": torch_cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "world_size": _world_size(),
    }


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, errors="replace", check=True
    )
    return out.stdout


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _slug(text: str) -> str:
    """``text`` as a safe file name: other characters become ``-``; no ``.`` or ``-`` at ends."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip(".-")


def _plain(value: Any) -> Any:
    """NumPy and PyTorch scalars and arrays, also inside mappings and lists, as plain values;
    mapping keys as strings, as JSON writes them and as W&B requires (e.g. seeds as keys)."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _indexed(values: Mapping[str, Any]) -> dict[str, Any]:
    """Lists of numbers as ``key/0``, ``key/1``, ... so that W&B charts them as series."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, list) and all(isinstance(v, int | float) for v in value):
            out.update({f"{key}/{i}": v for i, v in enumerate(value)})
        else:
            out[key] = value
    return out


def _merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    """Merge nested mappings into ``target`` in place; other values replace."""
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)
