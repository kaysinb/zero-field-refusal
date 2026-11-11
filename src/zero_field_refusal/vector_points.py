"""Study 5's points along a parameter-space vector (docs/study5-preregistration.md,
"Extension 2"): θ₀ + a·v for a vector v kept as a file (``vectors``) or a fresh F run's own
displacement Δθ = θ_N − θ₀ (its endpoint checkpoint), read in a run of its own.

**The vector.** Either a ``vectors`` file, its names the trainable set's and its θ₀ this
model's (``vectors.load_vector``), or a fresh F run's endpoint (``ladder.read_f_run``,
``line_scan.checkpoint_of`` and ``endpoint_state``, as the line scan reads it). ``sign`` −1 takes
−v. The vector stays fp32; the trainable parameters are set to θ₀ + a·v by
``evaluation.set_point``, θ₀ the frozen reference model, never written; the frozen parameters stay
θ₀'s.

**The readout at every point** (``points``, in order):
- K_benign and K_harmful as the run's evaluation computes them (``evaluation.sequence_kl`` over
  the scoring panel's K samples, the evaluator's own);
- r and its pieces on both surrogate panels (``surrogate.continuation_logprobs``): per panel, the
  mean over prompts of logmeanexp over the refusal openings (R), over the compliance openings (C)
  and of their difference; r is the harmful panel's difference (``surrogate.surrogate_value``);
- for a point in ``full``, the full evaluation as well (``train.Evaluator``, sealed: K on both
  panels, utility, the greedy responses to the harmful evaluation set) under the condition
  ``evaluation.vector_condition(label, a)``, so the responses are labeled like any condition.

A non-finite K is the point's outcome (``nonfinite_k``) and the points go on.

**Matched K** (``match``): the registered scale search and matched-K step of the ladder
(``ladder.scale_search`` from c₀ = 1, the vector's own scale, with at most ``extensions``
extensions, then ``ladder.refine``) on K_benign along the signed vector; at the chosen scale the
readout and the full evaluation, under ``evaluation.vector_condition(label, "kstar")``.

**The run.** One run directory per vector (``run_vector``): run type
``evaluation.VECTOR_RUN_TYPE``, experiment ``EXPERIMENT``, the config otherwise the given fresh F
run config's with ``study5.vector`` (label, the vector's identity, sign, points, full, match)
added. ``vector_points.jsonl`` gets one row per point as it ends; the summary's ``vector``
(points, conditions, evaluated, the matched-K record) is written after the last.
``run_judging.vector_conditions`` reads both.

Device memory at the peak: the student and the reference model, v and one evaluation.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, open_dict

from . import ladder, linear_response, surrogate, vectors
from .config import config_hash
from .evaluation import VECTOR_RUN_TYPE, sequence_kl, set_point, vector_condition
from .ladder import K_STAR, _peak_memory, refine, require_fresh, scale_search
from .line_scan import endpoint_state
from .models import freeze
from .runtime import Environment
from .seeding import derive_seed, seed_everything
from .tensor_hash import tensors_sha256
from .tracking import Diverged, Run
from .train import Evaluator

EXPERIMENT = "study5_vector"
EXTENSIONS = 8  # the matched-K search's most ×2 (÷2) extensions, as extension 1's
KEYS = ("K_benign", "K_harmful", "heldout_nll_per_token", "factual_accuracy",
        "benchmark_accuracy")  # fmt: skip


def vector_condition_check(label: str, points: Sequence[float]) -> None:
    """A valid label whose points give distinct conditions; ``ValueError`` otherwise."""
    conditions = [vector_condition(label, a) for a in points]
    if len(set(conditions)) != len(conditions):
        raise ValueError(f"the points {list(points)} do not give distinct conditions")


def vector_config(cfg: DictConfig, spec: Mapping[str, Any]) -> DictConfig:
    """The run's config: ``cfg``'s, run type ``VECTOR_RUN_TYPE``, experiment ``EXPERIMENT`` and
    ``study5.vector`` = ``spec``."""
    out = copy.deepcopy(cfg)
    with open_dict(out):
        out.run_type = VECTOR_RUN_TYPE
        out.experiment = EXPERIMENT
        out.study5.vector = dict(spec)
    return out


def surrogate_readout(
    model: torch.nn.Module,
    env: Environment,
    spec: surrogate.SurrogateSpec,
    panels: Mapping[str, Sequence[str]],
    micro_batch_size: int,
) -> dict[str, float]:
    """Per panel (``harmful``, ``benign``): the means over prompts of logmeanexp_R, logmeanexp_C
    and their difference; ``r`` is the harmful panel's difference."""
    out: dict[str, float] = {}
    continuations = [*spec.refusals, *spec.compliances]
    n_r = len(spec.refusals)
    for role, prompts in panels.items():
        logp = surrogate.continuation_logprobs(
            model, env.tokenizer, list(prompts), continuations, device=env.device,
            micro_batch_size=micro_batch_size, max_tokens=spec.max_tokens,
        )  # fmt: skip
        r_part = surrogate.logmeanexp(logp[:, :n_r])
        c_part = surrogate.logmeanexp(logp[:, n_r:])
        out[f"{role}_R"] = math.fsum(r_part.tolist()) / len(prompts)
        out[f"{role}_C"] = math.fsum(c_part.tolist()) / len(prompts)
        out[f"{role}_contrast"] = math.fsum((r_part - c_part).tolist()) / len(prompts)
    out["r"] = out["harmful_contrast"]
    return out


def run_vector(
    cfg: DictConfig,
    env: Environment,
    *,
    label: str,
    vector: Path | None = None,
    f_run: Mapping[str, Any] | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    sign: int = 1,
    points: Sequence[float] = (1.0,),
    full: Sequence[float] = (),
    match: bool = False,
    extensions: int = EXTENSIONS,
    surrogate_spec: Path = surrogate.SPEC_PATH,
    panel_size: int = 100,
    run_id: str | None = None,
) -> dict[str, Any]:
    """The points of one signed vector in a run directory of its own (module docstring).
    ``cfg`` is a fresh F run's config (its evaluator's panels and seeds); the vector is the file
    ``vector`` or, with ``f_run`` and ``checkpoint``, that run's displacement. Returns the record
    (no git state or time: the caller's)."""
    if (vector is None) == (checkpoint is None):
        raise ValueError("give exactly one of a vector file and an F run's checkpoint")
    if sign not in (1, -1):
        raise ValueError(f"sign is {sign!r}, not ±1")
    if not set(full) <= set(points):
        raise ValueError(f"the full evaluations {list(full)} are not among the points")
    vector_condition_check(label, points)
    linear_response.require_frozen_adam(cfg)
    require_fresh(cfg)
    seed = int(cfg.seed)
    torch_seed = derive_seed(seed, "torch")
    seed_everything(torch_seed)  # as the run seeds itself
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    spec = surrogate.load_spec(surrogate_spec)
    panels = {}
    for role, name in (("harmful", spec.harmful_panel), ("benign", spec.benign_panel)):
        items = env.prompt_set(name)
        if len(items) != panel_size:
            raise ValueError(f"panel {name} has {len(items)} prompts, not {panel_size}")
        panels[role] = [item["text"] for item in items]
    if vector is not None:
        identity: dict[str, Any] = {"file": Path(vector).name,
                                    "sha256": vectors.read_record(vector)["sha256"]}  # fmt: skip
    else:
        identity = {"f_run": f_run["run_id"], "checkpoint": Path(checkpoint["path"]).name,
                    "checkpoint_sha256": checkpoint["sha256"]}  # fmt: skip
    vspec = {"label": label, "vector": identity, "sign": sign, "points": list(points),
             "full": list(full), "match": match, "extensions": extensions}  # fmt: skip
    vcfg = vector_config(cfg, vspec)
    rows: list[dict[str, Any]] = []
    with Run(vcfg, run_id=run_id) as run:
        run.record_inputs({"prompt_manifest_sha256": env.prompt_manifest_sha256,
                           "vector": identity})  # fmt: skip
        student, reference = env.new_model(), env.new_model()
        reference.requires_grad_(False)
        trainable = freeze(student, cfg.train.freeze)
        names = [name for name, _ in trainable]
        theta0 = dict(reference.named_parameters())
        theta0_sha = tensors_sha256([(name, theta0[name]) for name in names])
        if vector is not None:
            loaded, vrecord = vectors.load_vector(vector, names, theta0_sha256=theta0_sha)
            direction = [loaded[name].to(theta0[name].device) * sign for name in names]
            del loaded
            vnorm = vrecord["norm"]
        else:
            if f_run["theta0_trainable_sha256"] != theta0_sha:
                raise ValueError("θ₀'s trainable tensors differ from the F run's")
            state = endpoint_state(checkpoint, trainable)
            with torch.no_grad():
                direction = [(state[name].to(theta0[name].device) - theta0[name]) * sign
                             for name in names]  # fmt: skip
            del state
            vnorm = vectors.norm(dict(zip(names, direction, strict=True)))
        evaluate = Evaluator(vcfg, env, run, reference, sealed=True)  # fails early on inputs
        streams = json.loads((run.dir / "meta.json").read_text())["inputs"]["streams"]
        run.record_inputs({"theta0_trainable_sha256": theta0_sha})
        start = [theta0[name] for name in names]
        micro = int(cfg.execution.micro_batch_size)
        eval_micro = int(cfg.execution.eval_batch_size)

        def k_of(c: float) -> float:  # the ladder's K_benign, the run's evaluation's
            set_point(trainable, start, direction, c)
            return ladder.k_benign(student, reference, evaluate.kl_streams["kl_benign"],
                                   pad_id=env.pad_id, micro_batch_size=micro)  # fmt: skip

        def point(a: float, condition: str, in_full: bool) -> dict[str, Any]:
            set_point(trainable, start, direction, a)
            row: dict[str, Any] = {"a": a, "condition": condition}
            for kind, samples in evaluate.kl_streams.items():
                row[f"K_{kind.removeprefix('kl_')}"] = (
                    sequence_kl(
                        student, reference, samples, pad_id=env.pad_id, micro_batch_size=micro
                    )
                    .mean()
                    .item()
                )
            row |= surrogate_readout(student, env, spec, panels, eval_micro)
            if not all(math.isfinite(row[k]) for k in ("K_benign", "K_harmful")):
                return row | {"evaluated": False, "reason": "nonfinite_k"}
            if not in_full:
                return row | {"evaluated": False, "reason": "readout_only"}
            try:
                full_row = evaluate(student, condition)
            except Diverged:  # the point's outcome, recorded; nothing of its condition written
                return row | {"evaluated": False, "reason": "nonfinite_k"}
            return row | {"evaluated": True, "full": {k: full_row.get(k) for k in KEYS}}

        def keep(row: dict[str, Any]) -> None:
            with run.path("vector_points.jsonl").open("a") as f:  # each point as it ends
                f.write(json.dumps(row, sort_keys=True) + "\n")
            rows.append(row)

        record: dict[str, Any] = {"points": list(points), "conditions": [], "evaluated": []}
        for a in points:
            condition = vector_condition(label, a)
            row = point(a, condition, a in full)
            keep(row)
            record["conditions"].append(condition)
            if row["evaluated"]:
                record["evaluated"].append(condition)
        matched = None
        if match:
            search = scale_search(k_of, 1.0, K_STAR, extensions=extensions)
            matched = {"search": search, "status": search["status"]}
            if search["status"] == "bracketed":
                refined = refine(k_of, search, K_STAR)
                matched |= {"refine": refined, "status": refined["status"]}
                if refined["status"] != "invalid_k":
                    condition = vector_condition(label, "kstar")
                    row = point(refined["c"], condition, True) | {"matched_k": True}
                    keep(row)
                    record["conditions"].append(condition)
                    if row["evaluated"]:
                        record["evaluated"].append(condition)
                    matched["c"] = refined["c"]
        peak = _peak_memory()
        run.summarize({"vector": {**record, "label": label, "matched": matched,
                                  "vector_norm": vnorm, **peak}})  # fmt: skip
        run_dir = run.dir
    return {
        "kind": "study5-vector",
        "label": label,
        "seed": seed,
        "vector": identity,
        "vector_norm": vnorm,
        "sign": sign,
        "points": list(points),
        "full": list(full),
        "conditions": record["conditions"],
        "evaluated": record["evaluated"],
        "matched": matched,
        "rows": rows,
        "k_star": K_STAR,
        "experiment": EXPERIMENT,
        "run_id": run_dir.name,
        "config_hash": config_hash(vcfg),
        "f_config_hash": config_hash(cfg),
        "theta0_trainable_sha256": theta0_sha,
        "surrogate_sha256": spec.sha256,
        "streams": streams,
        "rng": {"torch": torch_seed},
        "memory": peak,
        "model": {"name": cfg.model.name, "revision": cfg.model.revision,
                  "dtype": cfg.model.dtype, "attn_implementation": cfg.model.attn_implementation},
        "prompt_manifest_sha256": env.prompt_manifest_sha256,
    }  # fmt: skip
