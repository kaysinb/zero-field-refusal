"""Study 5's ladder of synthetic displacements (docs/study5-preregistration.md, "Descriptive
measurements", "Ladder"): for one seed of the crossed-control subset, a displacement δ of the
trainable parameters, scaled to the benign drift of study 3's comparison point and evaluated in
full at θ₀ + cδ, beside the seed's real endpoint.

**Rungs** (``evaluation.LADDER_RUNGS``), with D = (√v₀ + ε)⁻¹ elementwise (v₀ as arm F uses it,
from the seed's own preconditioner stream by ``linear_response.theta0_second_moment``), c_t F's
momentum weights, g_t the run's ``train.steps`` batch gradients at θ₀ and η = ``optimizer.lr``:

- ``gaussian_d``: δ = D^{1/2} ⊙ z, z ~ N(0, I): Gaussian with covariance D, not isotropic in
  parameter coordinates (``gaussian_d_direction``);
- ``linear_no_d``: δ = −η Σ_t c_t g_t (``linear_response.linear_displacement`` with D = 1);
- ``linear``: δ = δ̂ = −η Σ_t c_t D ⊙ g_t (``linear_response.linear_displacement``);
- ``linear_shuffled_d``: δ = −η Σ_t c_t π(D) ⊙ g_t, π permuting D's entries within each tensor
  (``shuffle_within_tensors``). Whether this rung is run is decided elsewhere.

The random draws are on the CPU, fp32, from one dedicated generator per rung seeded by
``derive_seed(seed, "study5", "gaussian_d")`` (z) or ``derive_seed(seed, "study5", "shuffle")`` (π),
over the trainable tensors by name and full shape in sorted name order: they do not depend on the
device, the parameter order or the mapping's order.

**Scale search** (``scale_search``). c₀ = ‖Δθ‖/‖δ‖, Δθ = θ_N − θ₀ of the seed's F run at η (role
``fresh``): its norm is the one the run records at its step-N monitor reading
(``monitor.jsonl``, ``displacement_norm`` of ``train.projection_stats``: ‖θ_N − θ₀‖ over the
trainable set, the fp32 differences reduced in float64), ``read_f_run``. K_benign is measured at
c ∈ {c₀/2, c₀, 2c₀} exactly as the run's evaluation computes it (``k_benign``: the mean over the
scoring panel's benign K samples of ``evaluation.sequence_kl``, the run's evaluator's samples,
reference and micro-batches). ln K is interpolated linearly in ln c between the bracketing
neighbours to K\\* (``K_STAR`` = 6.00, study 3's candidate-1 comparison point, in the units of
``K_STAR_UNIT``). If K\\* is not bracketed, the grid is extended by ×2 above the largest scale
(every K below K\\*) or by ÷2 below the smallest (every K above), at most ``EXTENSIONS`` times;
still not bracketed, the rung is ``not_bracketed`` with every value and is not evaluated. A K that
is not finite and positive stops the rung (``invalid_k``). A K that does not increase strictly
with c is recorded (``monotone``). When more than one pair brackets K\\* (a non-monotone K, or a K
exactly equal to K\\* at a grid point), the pair is the bracketing pair nearest to K\\*: the one
holding the evaluated K nearest to K\\* in ln K (ties: the other end's distance, then the smaller
c).

**Matched K** (``refine``). The interpolated scale c₁ is measured (K-only). A rung is matched
when that K lies within ±``TOLERANCE`` (10 %) of K\\*: 5.40 ≤ K ≤ 6.60. Outside it, one
refinement step: c₂ by the same interpolation of ln K in ln c between c₁ and the nearest measured
scale (in ln c) whose K lies on the other side of K\\*, and K measured at c₂. The full evaluation
runs at whichever of c₁ and c₂ has the smaller |ln K − ln K\\*| (c₁ on a tie, or when K(c₂) is not
finite and positive); a rung still outside the tolerance is evaluated there, at its measured K, and
flagged ``out_of_tolerance``. A K at c₁ that is not finite and positive stops the rung
(``invalid_k``). c₁, c₂, both K and the flag are recorded.

**Full evaluation.** At the chosen c the student is set to θ₀ + cδ (``evaluation.set_point``, fp32,
θ₀ the frozen reference copy) and evaluated once by an ``train.Evaluator`` of the F run's config
(sealed, as the scheduled run's: K on both panels, utility, the greedy responses to the harmful
evaluation set) under the condition ``evaluation.ladder_condition(rung)`` (``ladder_linear``), in
a run directory of its own (``run_rung``): run type ``LADDER_RUN_TYPE``, experiment
``EXPERIMENT``, the config otherwise the F run's with ``study5.ladder`` (the rung and the F run's
id) added. Its summary's ``ladder`` names the rung, the condition and whether it was evaluated;
``run_judging.response_conditions`` reads it, so the responses are judged like the scan points'.
A non-finite K in the full evaluation is the rung's outcome (``nonfinite_k``), nothing of its
condition written.

**The F run's identities.** The rung refuses an F run whose recorded config hash is not its
config's or that did not take all its updates (``read_f_run``), and, where the F run recorded
them, a θ₀, v₀, training stream or K-panel stream other than its own.

Device memory at the peak (building δ of a gradient rung): the student and the reference model,
D, δ and one gradient set; afterwards the two models and δ.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import PreTrainedModel

from . import linear_response
from .config import config_hash
from .evaluation import (
    LADDER_RUN_TYPE,
    LADDER_RUNGS,
    ladder_condition,
    sequence_kl,
    set_point,
)
from .models import freeze
from .runtime import Environment
from .seeding import derive_seed, seed_everything
from .teacher import Sample
from .tensor_hash import tensors_sha256
from .tracking import Diverged, Run
from .train import Evaluator

Named = Mapping[str, torch.Tensor]

# Study 3's K* (candidate 1), pinned here in the GPU code: the analysis package lies outside the
# GPU code identity, so the GPU code does not read it from there. tests/test_ladder.py checks it
# against analysis.study3_search's value.
K_STAR = 6.00
K_STAR_UNIT = (
    "K_benign as study 3 matches it (docs/study3-preregistration.md, Notation): the mean over the "
    "scoring panel's benign K samples (100 held-out benign prompts x 4 theta0 responses) of the "
    "sum over response positions of KL(p_theta0 || p_theta), full vocabulary, in nats per "
    "response"
)
EXTENSIONS = 3  # at most this many ×2 (÷2) extensions of the grid
TOLERANCE = 0.10  # a rung is matched when its measured K is within ±10 % of K*: 5.40 ≤ K ≤ 6.60
EXPERIMENT = "study5_ladder"  # the experiment directory of the rungs' evaluation runs
GAUSSIAN_D_KEYS = ("study5", "gaussian_d")  # derive_seed(seed, *GAUSSIAN_D_KEYS)
SHUFFLE_KEYS = ("study5", "shuffle")
D_RUNGS = ("gaussian_d", "linear", "linear_shuffled_d")  # the rungs that read v₀
GRADIENT_RUNGS = ("linear_no_d", "linear", "linear_shuffled_d")  # the rungs that read the g_t


# --------------------------------------------------------------------------------- the directions


@torch.no_grad()
def gaussian_d_direction(D: Mapping[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """δ = D^{1/2} ⊙ z (Gaussian with covariance D) in ``D``'s order; z drawn on the CPU in fp32
    by one generator seeded by ``derive_seed(seed, "study5", "gaussian_d")``, tensor by tensor in
    sorted name order at its full shape. ``D``'s tensors are consumed: each becomes δ's (no
    second set is held)."""
    generator = torch.Generator().manual_seed(derive_seed(seed, *GAUSSIAN_D_KEYS))
    for name in sorted(D):
        d = D[name]
        z = torch.randn(tuple(d.shape), generator=generator, dtype=torch.float32)
        d.sqrt_().mul_(z.to(d.device))
        del z
    return dict(D)


@torch.no_grad()
def shuffle_within_tensors(D: Mapping[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """π(D): each tensor's entries permuted within the tensor by ``torch.randperm`` of one CPU
    generator seeded by ``derive_seed(seed, "study5", "shuffle")``, tensor by tensor in sorted name
    order. In place: ``D``'s tensors are returned permuted."""
    generator = torch.Generator().manual_seed(derive_seed(seed, *SHUFFLE_KEYS))
    for name in sorted(D):
        flat = D[name].view(-1)
        perm = torch.randperm(flat.numel(), generator=generator)
        flat.copy_(flat[perm.to(flat.device)])
        del perm
    return dict(D)


def _unit(like: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """D = 1 without a set of ones: a one-element tensor expanded to each shape."""
    return {name: torch.ones((), dtype=t.dtype, device=t.device).expand(t.shape)
            for name, t in like.items()}  # fmt: skip


def rung_direction(
    rung: str,
    *,
    seed: int,
    weights: Sequence[float],
    lr: float,
    grads: Iterable[Mapping[str, torch.Tensor]] | None = None,
    D: Mapping[str, torch.Tensor] | None = None,
    like: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """The rung's δ before scaling. ``D`` (consumed by ``gaussian_d`` and ``linear_shuffled_d``) for
    the rungs that read it; ``grads`` yields one gradient set per weight for the gradient rungs;
    ``like`` gives ``linear_no_d`` the names, shapes, dtype and device."""
    if rung not in LADDER_RUNGS:
        raise ValueError(f"Unknown ladder rung {rung!r}; known: {LADDER_RUNGS}")
    if rung == "gaussian_d":
        return gaussian_d_direction(D, seed)
    if rung == "linear":
        return linear_response.linear_displacement(grads, D, weights, lr)
    if rung == "linear_shuffled_d":
        return linear_response.linear_displacement(grads, shuffle_within_tensors(D, seed),
                                                   weights, lr)  # fmt: skip
    return linear_response.linear_displacement(grads, _unit(like), weights, lr)


def norm(tensors: Iterable[torch.Tensor]) -> float:
    """The Euclidean norm, each tensor's square sum in float64."""
    return math.sqrt(math.fsum(t.double().square().sum().item() for t in tensors))


# ------------------------------------------------------------------------------- the scale search


def _valid(k: float) -> bool:
    return math.isfinite(k) and k > 0


def scale_search(
    k_of: Callable[[float], float], c0: float, target: float = K_STAR, *,
    extensions: int = EXTENSIONS,
) -> dict[str, Any]:  # fmt: skip
    """The scale c with K(c) = ``target`` by linear interpolation of ln K in ln c (module
    docstring). ``k_of(c)`` measures K at θ₀ + cδ. Returns ``status`` (``bracketed``,
    ``not_bracketed`` or ``invalid_k``), ``scales`` (every {c, K_benign}, in evaluation order),
    ``extensions``, ``monotone`` (K strictly increasing in c over the scales), ``bracketing_pairs``,
    ``pair`` ({c, K_benign} of the pair interpolated in, or None), ``choice`` (``only`` or
    ``nearest``) and ``c`` (None unless bracketed)."""
    if not (math.isfinite(c0) and c0 > 0):
        raise ValueError(f"c0 must be finite and positive, got {c0!r}")
    scales: list[dict[str, float]] = []
    out: dict[str, Any] = {"target": target, "c0": c0, "scales": scales, "extensions": 0,
                           "monotone": None, "bracketing_pairs": [], "pair": None,
                           "choice": None, "c": None}  # fmt: skip

    def measure(c: float) -> bool:
        k = float(k_of(c))
        scales.append({"c": c, "K_benign": k})
        return _valid(k)

    for c in (c0 / 2, c0, 2 * c0):
        if not measure(c):
            return out | {"status": "invalid_k"}
    while True:
        ordered = sorted((s["c"], s["K_benign"]) for s in scales)
        pairs = [(a, b) for a, b in pairwise(ordered)
                 if (a[1] - target) * (b[1] - target) <= 0]  # fmt: skip
        if pairs or out["extensions"] == extensions:
            break
        up = all(k < target for _, k in ordered)  # else every K is above it
        out["extensions"] += 1
        if not measure(ordered[-1][0] * 2 if up else ordered[0][0] / 2):
            return out | {"status": "invalid_k"}
    out["monotone"] = all(a[1] < b[1] for a, b in pairwise(ordered))
    if not pairs:
        return out | {"status": "not_bracketed"}
    out["bracketing_pairs"] = [[a[0], b[0]] for a, b in pairs]
    ln_t = math.log(target)

    def distance(pair: tuple[tuple[float, float], tuple[float, float]]) -> tuple[float, ...]:
        d = sorted(abs(math.log(k) - ln_t) for _, k in pair)
        return d[0], d[1], pair[0][0]

    (c_a, k_a), (c_b, k_b) = min(pairs, key=distance)
    return out | {"status": "bracketed", "pair": {"c": [c_a, c_b], "K_benign": [k_a, k_b]},
                  "choice": "only" if len(pairs) == 1 else "nearest",
                  "c": _interpolated(c_a, k_a, c_b, k_b, target)}  # fmt: skip


def _interpolated(c_a: float, k_a: float, c_b: float, k_b: float, target: float) -> float:
    """The c of K = ``target`` on the line through (ln c_a, ln k_a) and (ln c_b, ln k_b)."""
    ln_t = math.log(target)
    w = 0.0 if k_a == k_b else (ln_t - math.log(k_a)) / (math.log(k_b) - math.log(k_a))
    return c_a if w == 0 else c_b if w == 1 else math.exp(
        math.log(c_a) + w * (math.log(c_b) - math.log(c_a)))  # fmt: skip


def within(k: float, target: float = K_STAR, tolerance: float = TOLERANCE) -> bool:
    """Whether a measured K matches ``target`` within the relative ``tolerance``."""
    return _valid(k) and (1 - tolerance) * target <= k <= (1 + tolerance) * target


def refine(
    k_of: Callable[[float], float], search: Mapping[str, Any], target: float = K_STAR, *,
    tolerance: float = TOLERANCE,
) -> dict[str, Any]:  # fmt: skip
    """The matched-K step after a bracketed ``scale_search`` (module docstring, "Matched K"):
    K measured at c₁ = ``search["c"]``; outside the tolerance, one refinement c₂ with K measured
    there. Returns ``c1``, ``K1``, ``c2``, ``K2`` (None without a refinement), ``other`` (the
    measured {c, K_benign} c₂ was interpolated with), ``chosen`` (``c1`` or ``c2``), ``c`` and
    ``K`` (the chosen point), ``tolerance`` (relative, lower, upper), ``out_of_tolerance`` and
    ``status`` (``matched``, ``out_of_tolerance`` or ``invalid_k``)."""
    c1 = float(search["c"])
    out: dict[str, Any] = {
        "c1": c1, "K1": None, "c2": None, "K2": None, "other": None, "chosen": None, "c": None,
        "K": None, "out_of_tolerance": None,
        "tolerance": {"relative": tolerance, "lower": (1 - tolerance) * target,
                      "upper": (1 + tolerance) * target},
    }  # fmt: skip
    k1 = float(k_of(c1))
    out["K1"] = k1
    if not _valid(k1):
        return out | {"status": "invalid_k"}
    chosen = ("c1", c1, k1)
    above = k1 > target
    if not within(k1, target, tolerance) and any(  # a scale on the other side to refine with
            _valid(s["K_benign"]) and (s["K_benign"] < target if above else s["K_benign"] > target)
            for s in search["scales"]):  # fmt: skip
        other = [s for s in search["scales"] if _valid(s["K_benign"])
                 and (s["K_benign"] < target if above else s["K_benign"] > target)]  # fmt: skip
        nearest = min(other, key=lambda s: (abs(math.log(s["c"]) - math.log(c1)), s["c"]))
        c2 = _interpolated(c1, k1, nearest["c"], nearest["K_benign"], target)
        k2 = float(k_of(c2))
        out |= {"c2": c2, "K2": k2, "other": {"c": nearest["c"], "K_benign": nearest["K_benign"]}}
        ln_t = math.log(target)
        if _valid(k2) and abs(math.log(k2) - ln_t) < abs(math.log(k1) - ln_t):
            chosen = ("c2", c2, k2)
    name, c, k = chosen
    flagged = not within(k, target, tolerance)
    return out | {"chosen": name, "c": c, "K": k, "out_of_tolerance": flagged,
                  "status": "out_of_tolerance" if flagged else "matched"}  # fmt: skip


def k_benign(
    student: PreTrainedModel,
    reference: PreTrainedModel,
    samples: Sequence[Sample],
    *,
    pad_id: int,
    micro_batch_size: int,
) -> float:
    """K_benign as the run's evaluation computes it (``train.Evaluator``): the mean of
    ``sequence_kl`` over the scoring panel's benign K samples."""
    return sequence_kl(student, reference, samples, pad_id=pad_id,
                       micro_batch_size=micro_batch_size).mean().item()  # fmt: skip


# ---------------------------------------------------------------------------------- the F run


def require_fresh(cfg: DictConfig) -> None:
    """The ladder scales to a fresh seed's F run (``study5.role`` fresh)."""
    role = OmegaConf.select(cfg, "study5.role", default=None)
    if role != "fresh":
        raise ValueError(f"study5.role is {role!r}: the ladder scales to a fresh F run")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_f_run(run_dir: Path, cfg: DictConfig) -> dict[str, Any]:
    """The seed's F run of ``cfg`` (its recorded config hash equals ``cfg``'s), with all its
    ``train.steps`` updates: ‖Δθ‖ from its step-N monitor reading, its step-N evaluation, and the
    inputs it recorded (v₀ and θ₀ hashes, streams). ``ValueError`` when the run is not that."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("config_hash") != config_hash(cfg):
        raise ValueError(f"{run_dir}: config hash {meta.get('config_hash')}, not this config's "
                         f"{config_hash(cfg)}: not this seed's F run")  # fmt: skip
    steps = int(cfg.train.steps)
    terminal = json.loads((run_dir / "terminal.json").read_text())
    if terminal.get("steps") != steps or terminal.get("stop", {}).get("reason") != "steps":
        stop = terminal.get("stop")
        raise ValueError(f"{run_dir}: the run did not take its {steps} updates "
                         f"(steps {terminal.get('steps')}, stop {stop})")  # fmt: skip
    readings = [r for r in _rows(run_dir / "monitor.jsonl") if r.get("step") == steps]
    evaluations = [r for r in _rows(run_dir / "evaluations.jsonl") if r.get("step") == steps]
    if len(readings) != 1 or len(evaluations) != 1:
        raise ValueError(f"{run_dir}: {len(readings)} monitor readings and {len(evaluations)} "
                         f"evaluations at step {steps}, not one each")  # fmt: skip
    displacement = readings[0].get("displacement_norm")
    if not (isinstance(displacement, float) and math.isfinite(displacement) and displacement > 0):
        raise ValueError(f"{run_dir}: displacement norm {displacement!r} at step {steps}")
    keys = ("K_benign", "K_harmful", "heldout_nll_per_token", "factual_accuracy",
            "benchmark_accuracy")  # fmt: skip
    inputs = meta.get("inputs", {})
    return {
        "run_id": meta["run_id"],
        "config_hash": meta["config_hash"],
        "displacement_norm": displacement,
        "displacement_source": f"monitor.jsonl step {steps} displacement_norm",
        "endpoint": {"condition": evaluations[0]["condition"],
                     **{key: evaluations[0].get(key) for key in keys}},
        "v0_sha256": inputs.get("v0_sha256"),
        "theta0_trainable_sha256": inputs.get("theta0_trainable_sha256"),
        "streams": inputs.get("streams", {}),
        "endpoint_checkpoint": inputs.get("endpoint_checkpoint"),
    }  # fmt: skip


def ladder_config(cfg: DictConfig, rung: str, f_run_id: str) -> DictConfig:
    """The rung's evaluation run: the F run's config, run type ``LADDER_RUN_TYPE``, experiment
    ``EXPERIMENT`` and ``study5.ladder`` = {rung, f_run}."""
    out = copy.deepcopy(cfg)
    with open_dict(out):
        out.run_type = LADDER_RUN_TYPE
        out.experiment = EXPERIMENT
        out.study5.ladder = {"rung": rung, "f_run": f_run_id}
    return out


def _peak_memory() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3)}  # fmt: skip


def _check(what: str, ours: Any, theirs: Any) -> None:
    """An identity the F run recorded, which the rung must share (none recorded: no check)."""
    if theirs is not None and ours != theirs:
        raise ValueError(f"{what} differs from the F run's: {ours!r} against {theirs!r}")


# ---------------------------------------------------------------------------------- one rung


def run_rung(
    cfg: DictConfig,
    env: Environment,
    rung: str,
    f_run: Mapping[str, Any],
    *,
    run_id: str | None = None,
    extensions: int = EXTENSIONS,
) -> dict[str, Any]:
    """One rung of one seed: δ, the scale search, the matched-K step (``refine``) and, when
    bracketed, the full evaluation at the chosen scale, in a
    run directory of its own (module docstring). ``cfg`` is the seed's F run's config and
    ``f_run`` its ``read_f_run``. ``extensions``: the scale search's most ×2 (÷2) extensions,
    the registered ``EXTENSIONS`` unless an extension of the study sets another
    (docs/study5-preregistration.md, "Extension 1"); the record names it. Returns the record (no
    git state or time: the caller's)."""
    condition = ladder_condition(rung)
    linear_response.require_frozen_adam(cfg)
    require_fresh(cfg)
    seed = int(cfg.seed)
    torch_seed = derive_seed(seed, "torch")
    seed_everything(torch_seed)  # as the run seeds itself
    batches, training = None, None
    if rung in GRADIENT_RUNGS:
        batches, training = linear_response.feed_batches(cfg, env)  # fails early on a short stream
        _check("the training stream", training, f_run["streams"].get("training"))
    lr, beta1, eps = float(cfg.optimizer.lr), float(cfg.optimizer.beta1), float(cfg.optimizer.eps)
    weights = linear_response.momentum_weights(beta1, int(cfg.train.steps))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    lcfg = ladder_config(cfg, rung, f_run["run_id"])
    with Run(lcfg, run_id=run_id) as run:
        run.record_inputs({"prompt_manifest_sha256": env.prompt_manifest_sha256,
                           "f_run": {"run_id": f_run["run_id"],
                                     "config_hash": f_run["config_hash"]}})  # fmt: skip
        student, reference = env.new_model(), env.new_model()
        reference.requires_grad_(False)
        trainable = freeze(student, cfg.train.freeze)
        theta0 = dict(reference.named_parameters())
        theta0_sha = tensors_sha256([(name, theta0[name]) for name, _ in trainable])
        _check("θ₀'s trainable tensors", theta0_sha, f_run["theta0_trainable_sha256"])
        evaluate = Evaluator(lcfg, env, run, reference, sealed=True)  # fails early on inputs
        streams = json.loads((run.dir / "meta.json").read_text())["inputs"]["streams"]
        for kind, identity in streams.items():  # the panels the F run was measured on
            _check(f"the {kind} stream", identity, f_run["streams"].get(kind))

        D, v0_sha, preconditioner = None, None, None
        if rung in D_RUNGS:
            v0, preconditioner = linear_response.theta0_second_moment(cfg, env, student, trainable)
            v0_sha = linear_response.v0_sha256(trainable, v0)
            _check("v₀", v0_sha, f_run["v0_sha256"])
            run.record_inputs({"v0_sha256": v0_sha, "streams": {"preconditioner": preconditioner}})
            D = {name: v.sqrt_().add_(eps).reciprocal_()
                 for (name, _), v in zip(trainable, v0, strict=True)}  # fmt: skip
            del v0
        grads = None
        if batches is not None:
            run.record_inputs({"streams": {"training": training}})
            grads = linear_response.step_gradients(student, trainable, batches, cfg, env)
        delta = rung_direction(rung, seed=seed, weights=weights, lr=lr, grads=grads, D=D,
                               like=dict(trainable))  # fmt: skip
        del D, grads
        delta_norm = norm(delta.values())
        if not (math.isfinite(delta_norm) and delta_norm > 0):
            raise ValueError(f"‖δ‖ = {delta_norm!r}: the rung's direction cannot be scaled")
        c0 = f_run["displacement_norm"] / delta_norm
        start = [theta0[name] for name, _ in trainable]
        direction = [delta[name] for name, _ in trainable]
        samples = evaluate.kl_streams["kl_benign"]
        micro = int(cfg.execution.micro_batch_size)

        def k_of(c: float) -> float:
            set_point(trainable, start, direction, c)
            return k_benign(student, reference, samples, pad_id=env.pad_id,
                            micro_batch_size=micro)  # fmt: skip

        search = scale_search(k_of, c0, K_STAR, extensions=extensions)
        status, evaluation, matched, c = search["status"], None, None, None
        if status == "bracketed":
            matched = refine(k_of, search, K_STAR)
            status, c = ("invalid_k", None) if matched["status"] == "invalid_k" else (
                status, matched["c"])  # fmt: skip
        if status == "bracketed":
            set_point(trainable, start, direction, c)
            try:
                evaluation = evaluate(student, condition)
                status = "evaluated"
            except Diverged:  # the point's outcome: recorded, nothing of its condition written
                status = "nonfinite_k"
        peak = _peak_memory()
        flagged = matched["out_of_tolerance"] if matched else None
        run.summarize({"ladder": {"rung": rung, "condition": condition, "status": status,
                                  "evaluated": evaluation is not None, "c": c,
                                  "c0": c0, "delta_norm": delta_norm,
                                  "extensions_allowed": extensions,
                                  "scales": search["scales"], "matched_k": matched,
                                  "out_of_tolerance": flagged, "evaluation": evaluation,
                                  **peak}})  # fmt: skip
        run_dir = run.dir
    return {
        "kind": "study5-ladder",
        "rung": rung,
        "seed": seed,
        "condition": condition,
        "status": status,
        "lr": lr,
        "beta1": beta1,
        "eps": eps,
        "steps": int(cfg.train.steps),
        "batch_size": int(cfg.train.batch_size),
        "weights": weights,
        "k_star": K_STAR,
        "k_star_unit": K_STAR_UNIT,
        "displacement_norm": f_run["displacement_norm"],
        "displacement_source": f_run["displacement_source"],
        "delta_norm": delta_norm,
        "c0": c0,
        "extensions_allowed": extensions,
        "search": search,
        "matched_k": matched,
        "out_of_tolerance": flagged,
        "c": c,
        "evaluation": evaluation,
        "f_run": dict(f_run),
        "experiment": EXPERIMENT,
        "run_id": run_dir.name,
        "config_hash": config_hash(lcfg),
        "f_config_hash": config_hash(cfg),
        "v0_sha256": v0_sha,
        "theta0_trainable_sha256": theta0_sha,
        "streams": {"training": training, "preconditioner": preconditioner, **streams},
        "rng": {"torch": torch_seed,
                "gaussian_d": (derive_seed(seed, *GAUSSIAN_D_KEYS) if rung == "gaussian_d"
                               else None),
                "shuffle": (derive_seed(seed, *SHUFFLE_KEYS) if rung == "linear_shuffled_d"
                            else None),
                "name_order": "sorted"},
        "memory": peak,
        "model": {"name": cfg.model.name, "revision": cfg.model.revision,
                  "dtype": cfg.model.dtype, "attn_implementation": cfg.model.attn_implementation},
        "prompt_manifest_sha256": env.prompt_manifest_sha256,
    }  # fmt: skip
