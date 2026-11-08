"""Study 5's line scan of a fresh seed's η_l run (docs/study5-preregistration.md, "Descriptive
measurements", "Line scans"), evaluated in a run of its own from that run's endpoint checkpoint;
the η_l run itself is a plain F run.

**The endpoint.** The F run (``ladder.read_f_run``: its config hash this config's, all its
``train.steps`` updates) records its checkpoint (``inputs.endpoint_checkpoint``: sha256, step,
names) and θ₀'s trainable hash (``inputs.theta0_trainable_sha256``); both are required. The file
``<checkpoints>/<run id>.pt`` must have the recorded sha256 (``checkpoint_of``), and its tensors
must be exactly the trainable set's names, shapes and dtypes (``endpoint_state``). The same two
functions give ``scripts/study5_surrogate_endpoint.py`` its endpoint.

**The points.** For each a of ``POINTS`` (the registered −1, −0.5, −0.25, +0.25, +0.5; a = 0 is
θ₀'s own run, a = +1 the F run's step-N evaluation) the trainable parameters are set to
θ₀ + aΔθ, Δθ = θ_N − θ₀ in fp32 from θ₀ (the frozen reference model, never written) and θ_N (the
checkpoint's tensors, held unchanged on the host), by ``evaluation.set_point``; the frozen
parameters stay θ₀'s. Each point is evaluated in full by an ``train.Evaluator`` of the F run's
config (sealed, as the scheduled run's: K on both panels, utility, the greedy responses to the
harmful evaluation set) under the condition ``evaluation.line_scan_condition(a)``
(``line_scan_-0.25``). The evaluator draws no random number; the points differ by θ alone. A
non-finite K at a point is that point's outcome (``evaluated: false``, ``nonfinite_k``, nothing of
its condition written) and the scan goes on.

**The run.** One run directory per seed (``run_scan``): run type ``evaluation.SCAN_RUN_TYPE``,
experiment ``EXPERIMENT``, the config otherwise the F run's with ``study5.scan`` (the F run's id
and the points) added. ``line_scan.jsonl`` gets one row per point as each ends, so a run cut off
keeps the rows of the points it finished; the summary's ``line_scan`` (points, conditions,
evaluated) is written after the last. ``run_judging.response_conditions`` reads both, so the
points are labeled like any condition; a scan run that did not finish is not labeled, and its
points are missing by name in the analysis. Nothing of the F run is written.

**Identities.** The scan refuses an F run whose θ₀ or K-panel streams are not its own (where the
F run recorded them) and a checkpoint that is not the run's.

Device memory at the peak: the student and the reference model, Δθ and one evaluation; host
memory: the checkpoint's trainable tensors.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, open_dict

from . import linear_response
from .config import config_hash
from .evaluation import SCAN_RUN_TYPE, line_scan_condition, set_point
from .ladder import _check, _peak_memory, require_fresh
from .models import freeze
from .runtime import Environment
from .seeding import derive_seed, seed_everything
from .tensor_hash import file_sha256, tensors_sha256
from .tracking import Diverged, Run
from .train import Evaluator

POINTS = (-1.0, -0.5, -0.25, 0.25, 0.5)  # the registered points besides a = 0 and a = +1
EXPERIMENT = "study5_scan"  # the experiment directory of the scan runs
KEYS = ("K_benign", "K_harmful", "heldout_nll_per_token", "factual_accuracy",
        "benchmark_accuracy")  # fmt: skip


def names_sha256(names: list[str]) -> str:
    return hashlib.sha256(json.dumps(names).encode()).hexdigest()


def checkpoint_of(f_run_dir: Path, f_run: Mapping[str, Any], checkpoints: Path,
                  steps: int) -> dict[str, Any]:  # fmt: skip
    """The run's endpoint checkpoint as it recorded it, its file ``checkpoints/<run id>.pt``
    checked against the recorded sha256 and step; ``ValueError`` otherwise."""
    f_run_dir = Path(f_run_dir)
    recorded = f_run.get("endpoint_checkpoint")
    if not isinstance(recorded, dict) or not isinstance(recorded.get("sha256"), str):
        raise ValueError(f"{f_run_dir.name}: the run records no endpoint checkpoint")
    if recorded.get("step") != steps:
        raise ValueError(f"{f_run_dir.name}: its checkpoint is of step {recorded.get('step')}, "
                         f"not {steps}")  # fmt: skip
    if not isinstance(f_run.get("theta0_trainable_sha256"), str):
        raise ValueError(f"{f_run_dir.name}: the run records no hash of θ₀'s trainable tensors")
    path = Path(checkpoints) / f"{f_run['run_id']}.pt"
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path}: no checkpoint file of the run")
    digest = file_sha256(path)
    if digest != recorded["sha256"]:
        raise ValueError(f"{path.name}: sha256 {digest}, not the run's recorded "
                         f"{recorded['sha256']}")  # fmt: skip
    return {"path": path, "sha256": digest, "step": recorded["step"],
            "names_sha256": recorded.get("names_sha256")}  # fmt: skip


def endpoint_state(
    checkpoint: Mapping[str, Any], trainable: list[tuple[str, torch.nn.Parameter]]
) -> dict[str, torch.Tensor]:
    """The checkpoint's tensors on the host, exactly the trainable set's names, shapes and dtypes
    (and the recorded names' hash, where recorded); ``ValueError`` otherwise."""
    names = [name for name, _ in trainable]
    if checkpoint.get("names_sha256") is not None and checkpoint["names_sha256"] != names_sha256(
        names
    ):
        raise ValueError("the checkpoint's recorded names are not the trainable set's")
    state = torch.load(checkpoint["path"], map_location="cpu", weights_only=True)
    if sorted(state) != sorted(names):
        raise ValueError(f"the checkpoint holds {len(state)} tensors, not exactly the "
                         f"{len(names)} trainable ones")  # fmt: skip
    for name, p in trainable:
        if state[name].shape != p.shape or state[name].dtype != p.dtype:
            raise ValueError(f"{name}: the checkpoint's tensor is not the parameter's shape and "
                             "dtype")  # fmt: skip
    return state


def scan_config(cfg: DictConfig, f_run_id: str) -> DictConfig:
    """The scan run's config: the F run's, run type ``SCAN_RUN_TYPE``, experiment
    ``EXPERIMENT`` and ``study5.scan`` = {f_run, points}."""
    out = copy.deepcopy(cfg)
    with open_dict(out):
        out.run_type = SCAN_RUN_TYPE
        out.experiment = EXPERIMENT
        out.study5.scan = {"f_run": f_run_id, "points": list(POINTS)}
    return out


def run_scan(
    cfg: DictConfig,
    env: Environment,
    f_run: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """The scan of one seed's η_l run in a run directory of its own (module docstring). ``cfg``
    is the F run's config, ``f_run`` its ``ladder.read_f_run`` and ``checkpoint`` its
    ``checkpoint_of``. Returns the record (no git state or time: the caller's)."""
    linear_response.require_frozen_adam(cfg)
    require_fresh(cfg)
    seed = int(cfg.seed)
    torch_seed = derive_seed(seed, "torch")
    seed_everything(torch_seed)  # as the run seeds itself
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    scfg = scan_config(cfg, f_run["run_id"])
    rows: list[dict[str, Any]] = []
    with Run(scfg, run_id=run_id) as run:
        run.record_inputs({"prompt_manifest_sha256": env.prompt_manifest_sha256,
                           "f_run": {"run_id": f_run["run_id"],
                                     "config_hash": f_run["config_hash"],
                                     "endpoint_checkpoint": {
                                         "name": Path(checkpoint["path"]).name,
                                         "sha256": checkpoint["sha256"],
                                         "step": checkpoint["step"]}}})  # fmt: skip
        student, reference = env.new_model(), env.new_model()
        reference.requires_grad_(False)
        trainable = freeze(student, cfg.train.freeze)
        theta0 = dict(reference.named_parameters())
        theta0_sha = tensors_sha256([(name, theta0[name]) for name, _ in trainable])
        _check("θ₀'s trainable tensors", theta0_sha, f_run["theta0_trainable_sha256"])
        state = endpoint_state(checkpoint, trainable)
        endpoint_sha = tensors_sha256([(name, state[name]) for name, _ in trainable])
        evaluate = Evaluator(scfg, env, run, reference, sealed=True)  # fails early on inputs
        streams = json.loads((run.dir / "meta.json").read_text())["inputs"]["streams"]
        for kind, identity in streams.items():  # the panels the F run was measured on
            _check(f"the {kind} stream", identity, f_run["streams"].get(kind))
        run.record_inputs({"theta0_trainable_sha256": theta0_sha,
                           "endpoint_trainable_sha256": endpoint_sha})  # fmt: skip
        start = [theta0[name] for name, _ in trainable]
        with torch.no_grad():
            direction = [state[name].to(theta0[name].device) - theta0[name]
                         for name, _ in trainable]  # fmt: skip
        del state
        record: dict[str, Any] = {"points": list(POINTS), "conditions": [], "evaluated": []}
        for a in POINTS:
            condition = line_scan_condition(a)
            record["conditions"].append(condition)
            set_point(trainable, start, direction, a)
            row: dict[str, Any] = {"step": int(cfg.train.steps), "a": a, "condition": condition}
            try:
                row |= {"evaluated": True, **evaluate(student, condition), **_peak_memory()}
                record["evaluated"].append(condition)
            except Diverged:  # the point's outcome, recorded; nothing of its condition written
                row |= {"evaluated": False, "reason": "nonfinite_k"}
            with run.path("line_scan.jsonl").open("a") as f:  # each point as it ends
                f.write(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)
        peak = _peak_memory()
        run.summarize({"line_scan": {**record, **peak}})
        run_dir = run.dir
    return {
        "kind": "study5-scan",
        "seed": seed,
        "lr": float(cfg.optimizer.lr),
        "steps": int(cfg.train.steps),
        "points": list(POINTS),
        "conditions": record["conditions"],
        "evaluated": record["evaluated"],
        "rows": [{key: row.get(key) for key in ("a", "condition", "evaluated", "reason", *KEYS)}
                 for row in rows],  # fmt: skip
        "f_run": dict(f_run),
        "checkpoint": {"name": Path(checkpoint["path"]).name, "sha256": checkpoint["sha256"],
                       "step": checkpoint["step"]},
        "experiment": EXPERIMENT,
        "run_id": run_dir.name,
        "config_hash": config_hash(scfg),
        "f_config_hash": config_hash(cfg),
        "theta0_trainable_sha256": theta0_sha,
        "endpoint_trainable_sha256": endpoint_sha,
        "streams": streams,
        "rng": {"torch": torch_seed},
        "memory": peak,
        "model": {"name": cfg.model.name, "revision": cfg.model.revision,
                  "dtype": cfg.model.dtype, "attn_implementation": cfg.model.attn_implementation},
        "prompt_manifest_sha256": env.prompt_manifest_sha256,
    }  # fmt: skip
