"""The numerical check of study 2's selected rates (amendment 4, section 1, with reviews fa12 and
fa13, and the review of Task 1): its frozen scope and replacement cases, the fixed perturbation
norm, the float64 reference against a float64 oracle, which records count (the policy, the GPU
code, the regular run located now), the states of cases, rates and controls, and the verdict
record kept per selection; then the whole check on tiny K-only CPU runs with harmful generation
forbidden, a twin A unlike its archive incomplete twice, an instability stop, the reference's
fallback, and (skipped without a GPU) the check through the script on an unindexed CUDA device
under a 1 GB memory rule."""

import ast
import dataclasses
import functools
import hashlib
import json
import math
import platform
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from conftest import tiny_model, tiny_tokenizer
from omegaconf import OmegaConf
from test_fa_control_diagnostic import forbid_outcomes
from test_train import APPENDABLE, SMALL, pass_path_check, prepare, prompt_sets

from zero_field_refusal import code_identity
from zero_field_refusal import fa_control_diagnostic as fcd
from zero_field_refusal.config import load_config
from zero_field_refusal.models import freeze, install_neutral_generation_config, set_numerics
from zero_field_refusal.optim import Adam
from zero_field_refusal.runtime import Environment
from zero_field_refusal.schedule import run_scheduled

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "numerical_check.py"
CPU = torch.device("cpu")
CUDA = torch.device("cuda")  # unindexed, as runtime.environment gives it on a Pod
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
COMMIT = "c" * 40  # HEAD in these tests
LATER = "e" * 40  # an analysis-only commit after it: the same GPU code
OTHER = "f" * 40  # a commit with other GPU code
CLEAN = {"commit": COMMIT, "dirty": False}
BOUND = 0.0632455532033676  # 0.01·√(5·8)
EVAL_AT = [1, 5, 10, 15, 20, 25]
ARMS = ("adamw", "adam_frozen", "adamw_prewarmed")
STATES = {"qualified", "failing", "not_qualified", "incomplete"}
CONTROL_STATES = {"accepted", "rejected", "not_qualified", "incomplete"}
# Illustrative selections, as a selection record would give them: never in the code
RATES = {"adamw": [1.32e-05, 1.12e-05, 1.6e-05], "adam_frozen": [9.98e-06, 1.21e-05, 1.46e-05],
         "adamw_prewarmed": [1.1e-05, 1.3e-05, 1.5e-05]}  # fmt: skip
VERDICT_KEYS = ["batch", "U", "arms", "bound", "commit", "policy", "rates", "seeds", "controls",
                "qualified", "reasons", "cases", "reference_fallback", "provenance"]  # fmt: skip


def settings(**changes):
    return OmegaConf.merge(load_config("experiments/numerical_check"), changes)


FROZEN = OmegaConf.to_container(settings().numerical, resolve=True)


def scope():
    return fcd.NumericalScope.of(settings())


def cases(rates=RATES, upper=8, batch=20):
    return fcd.numerical_cases(settings(), rates, upper, batch=batch)


RUN_COMMIT = "r" * 40  # the commit the regular runs recorded
GPU_SHA = {COMMIT: "same", RUN_COMMIT: "same", LATER: "same", OTHER: "other"}


def fake_identity(commit="HEAD", repo=None):
    """``code_identity.gpu_code_identity`` over ``GPU_SHA``; an unknown commit fails as git
    would."""
    if commit not in GPU_SHA:
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", commit])
    return {"commit": commit, "files": 7, "sha256": GPU_SHA[commit]}


def fake_reuse(base, head="HEAD", repo=None):
    """``code_identity.reuse_check`` over ``GPU_SHA``."""
    a, b = fake_identity(base), fake_identity(head)
    same = a["sha256"] == b["sha256"]
    return {"base": a, "head": b, "changed": [] if same else ["src/zero_field_refusal/optim.py"],
            "reusable": same}  # fmt: skip


GPU_CODE = fake_identity(RUN_COMMIT)


@pytest.fixture
def clean_head(monkeypatch):
    """HEAD at ``COMMIT`` from a clean checkout, for every record and verdict the test makes,
    with the GPU identities of ``GPU_SHA``."""
    monkeypatch.setattr(fcd, "git_state", lambda *args, **kwargs: dict(CLEAN))
    monkeypatch.setattr(fcd, "gpu_code_identity", fake_identity)
    monkeypatch.setattr(fcd, "reuse_check", fake_reuse)


def consistent_code():
    return {"consistent": True, "gpu_code": GPU_CODE, "problems": []}


@functools.cache
def recorded_environment() -> dict:
    """This machine's run environment as ``tracking.Run`` records it in meta.json."""
    from zero_field_refusal import tracking

    return {"python": platform.python_version(), "packages": tracking.package_versions(),
            "hardware": tracking.hardware(),
            "launch": {"source_snapshot_id": None, "container_image": None}}  # fmt: skip


def fake_runs(root, found, commit=RUN_COMMIT):
    """A finished, conforming regular run's records for each case, made at ``commit`` in this
    machine's run environment."""
    for case in found:
        run = root / "study2_short" / case.name
        if run.exists():
            continue
        run.mkdir(parents=True)
        OmegaConf.save(
            OmegaConf.create(
                {
                    "experiment": "study2_short",
                    "seed": case.seed,
                    "control": None,
                    "protocol_version": 2,
                    "study2_short": {"role": "numerical"},
                    "train": {"batch_size": case.batch, "steps": 25},
                    "optimizer": {"name": case.optimizer, "lr": case.lr},
                }
            ),
            run / "config.yaml",
        )
        (run / "terminal.json").write_text(
            json.dumps({"stop": {"reason": "steps", "kind": "limit"}, "steps": 25})
        )
        (run / "metrics.jsonl").write_text(json.dumps({"step": 0, "loss": 1.0}) + "\n")
        meta = {"git": {"commit": commit, "dirty": False}, "config_hash": f"hash-{case.name}"}
        (run / "meta.json").write_text(json.dumps({**meta, **recorded_environment()}))
    # fmt: skip


WORLD: dict = {}


@pytest.fixture
def world(tmp_path, clean_head):
    """The regular runs of every case of the original and a re-solved selection, under a RUNS_DIR
    of the test's own; synthetic records name them."""
    root = tmp_path / "runs"
    fake_runs(root, cases() + cases(RESOLVED))
    WORLD["root"] = root
    yield root
    WORLD.clear()


RESOLVED = {**RATES, "adam_frozen": [9.98e-06, 1.05e-05, 1.21e-05]}  # FA 1.46e-05 dropped


def identity_of(case):
    return fcd.run_identity(fcd.locate_case(WORLD["root"], scope(), case))


def reader(state=CLEAN, the_scope=None):
    return fcd.RecordReader(the_scope or scope(), state, WORLD["root"])


# --------------------------------------------------------------------------------- the scope


def test_numerical_cases_are_the_frozen_scope() -> None:
    s = scope()
    assert s.seeds == (2200, 2201, 2202) and s.arms == ARMS and s.control_arm == "adam_frozen"
    assert (s.experiment, s.role, s.steps, s.protocol_version) == ("study2_short", "numerical",
                                                                   25, 2)  # fmt: skip
    assert s.eval_at == tuple(EVAL_AT) and s.norm == 5e-05
    assert dict(s.gaussian) == {"p3": 3, "p4": 4}
    assert (s.survival_share, s.rounding_scale, s.memory_rule_gb) == (0.5, 10, 46)
    assert s.bound(8) == BOUND and s.k_star(8) == math.sqrt(40) and s.k_star(7) == math.sqrt(35)
    assert s.policy(8, GPU_CODE, [LATER, COMMIT]) == {
        "gpu_code": {"commit": RUN_COMMIT, "files": 7, "sha256": "same"},
        "tool_commit": [COMMIT, LATER], "norm": 5e-05, "gaussian": {"p3": 3, "p4": 4},
        "survival_share": 0.5, "retry_scale": 10, "bound": BOUND, "steps": EVAL_AT,
        "reference": "float64 loss, accumulation and optimizer arithmetic; fallback float64 "
                     "loss and accumulation only"}  # fmt: skip
    n = settings().numerical
    assert n.evaluation == "k_only" and n.harmful_generation is False
    assert "rates" not in n  # the rates come from the selection, never from the scope

    found = cases()
    assert len(found) == 15 and len({c.name for c in found}) == 15
    first = [c for c in found if c.seed == 2200]
    assert {(c.optimizer, c.lr) for c in first} == {(a, r) for a in ARMS for r in RATES[a]}
    assert len(first) == 9 and found[:9] == first  # the nine pairs first, on seed 2200
    for seed in (2201, 2202):
        assert {(c.optimizer, c.lr) for c in found if c.seed == seed} == {
            (arm, max(RATES[arm])) for arm in ARMS
        }  # each arm's U-rate
    assert {c.batch for c in found} == {20} and {c.upper for c in found} == {8}
    assert {c.key for c in found} == {f"{a}|{r:g}" for a in ARMS for r in RATES[a]}
    assert fcd.numerical_root(20, 8, Path("/r")) == Path("/r/numerical/b20-u8")

    for bad in (
        {**RATES, "sgd": [1e-3, 2e-3, 3e-3]},
        {k: RATES[k] for k in ARMS[1:]},
        {**RATES, "adamw": [1e-5, 2e-5]},
        {**RATES, "adamw": [1e-5, 1e-5, 2e-5]},
        {**RATES, "adamw": [1e-5, 2e-5, math.nan]},
        {**RATES, "adamw": [-1, 1, 2]},
    ):
        with pytest.raises(fcd.NumericalError):
            cases(bad)
    with pytest.raises(fcd.NumericalError):
        cases(upper=0)
    with pytest.raises(fcd.NumericalError):
        fcd.numerical_root(20, 7.5)
    # The scope compares records' settings: every value but the memory rule
    assert s.same_policy(FROZEN) and s.same_policy({**FROZEN, "memory_rule_gb": 1})
    for changed in ({"perturbation": {**FROZEN["perturbation"], "norm": 5e-9}},
                    {"perturbation": {**FROZEN["perturbation"], "gaussian": {"p3": 7, "p4": 8}}},
                    {"perturbation": {**FROZEN["perturbation"], "rounding_scale": 1000}},
                    {"perturbation": {**FROZEN["perturbation"], "survival_share": 0.1}},
                    {"eval_at": [1, 25]}, {"seeds": [1, 2, 3]}):  # fmt: skip
        assert not s.same_policy({**FROZEN, **changed}), changed
    assert not s.same_policy(None) and not s.same_policy({"norm": 5e-5})
    # The declarative keys too: settings claiming harmful generation are no scope (review nit)
    assert dict(s.declared) == {"evaluation": '"k_only"', "harmful_generation": "false"}
    for changed in ({"harmful_generation": True}, {"evaluation": "full"}, {"extra": 1}):
        assert not s.same_policy({**FROZEN, **changed}), changed
    assert not s.same_policy({k: v for k, v in FROZEN.items() if k != "harmful_generation"})
    with pytest.raises(fcd.NumericalError, match="K-only"):
        fcd.NumericalScope.of(settings(numerical={"evaluation": "full"}))


def test_replacement_cases() -> None:
    """A replacement rate gets a case on 2200, and on 2201 and 2202 too when it is its arm's new
    U-rate; the cases are those the re-solved selection's scope holds for it."""
    s = settings()
    low = fcd.replacement_cases(s, "adam_frozen", 1.05e-05, False, batch=20, upper=8)
    assert [(c.seed, c.optimizer, c.lr, c.batch, c.upper) for c in low] == [
        (2200, "adam_frozen", 1.05e-05, 20, 8)
    ]
    top = fcd.replacement_cases(s, "adamw", 1.7e-05, True, batch=20, upper=8)
    assert [c.seed for c in top] == [2200, 2201, 2202] and {c.lr for c in top} == {1.7e-05}
    assert set(top) <= set(cases({**RATES, "adamw": [1.12e-05, 1.32e-05, 1.7e-05]}))
    assert set(low) <= set(cases(RESOLVED))
    for arm, rate in (("sgd", 1e-3), ("adamw", 0.0), ("adamw", math.nan)):
        with pytest.raises(fcd.NumericalError):
            fcd.replacement_cases(s, arm, rate, True, batch=20, upper=8)


def tiny_twins(v0_scale=1e-6):
    model = tiny_model(tied=False, head_std=0.02)
    trainable = freeze(model, ["input_embeddings", "output_head"])
    g = torch.Generator().manual_seed(5)
    v0 = [torch.rand(p.shape, generator=g) * v0_scale for _, p in trainable]
    v0[0].reshape(-1)[:4] = 0.0  # ε-dominated coordinates
    theta0 = {n: p.detach().clone() for n, p in trainable}
    return fcd.Twins(env=None, student=model, other=None, trainable=trainable, theta0=theta0,
                     v0=v0, prefix=None, scoring=[], k_star=math.sqrt(40))  # fmt: skip


def shared_with(residual):
    return {"residual": residual, "v0_sha256": "v", "v0_stream": {}, "control_residual": {},
            "control_residual_norm": fcd._norm(residual), "scoring_panel": {}}  # fmt: skip


def test_perturbation_norm_is_fixed() -> None:
    """p1-p4 are scaled to the scope's 5e-5 whatever the residual's size (the old control's
    first-step displacement scaled them; nothing of a control does now): their intended norm is
    5e-5, p1 points along FA's first update −P·g (P formed in float64), p2 opposite."""
    s, eps = scope(), 1e-8
    twins = tiny_twins()
    g = torch.Generator().manual_seed(6)
    residual = [torch.randn(p.shape, generator=g) * 1e-4 for _, p in twins.trainable]
    records = {}
    for size in (1.0, 1e-3, 1e3):  # the residual's size, as an old control's rate would scale it
        shared = shared_with([r * size for r in residual])
        facts = fcd.seed_directions(s, twins, shared, eps)
        assert "residual" not in shared and set(shared["scales"]) == set(fcd.PERTURBATIONS)
        assert facts["intended_norm"] == 5e-05
        for key in fcd.PERTURBATIONS:
            assert shared["scales"][key] * facts["direction_norms"][key] == pytest.approx(5e-05)
            attempts = fcd.apply_perturbation(
                twins.trainable, twins.theta0, shared["directions"][key], shared["scales"][key],
                survival_share=s.survival_share, rounding_scale=s.rounding_scale,
            )  # fmt: skip
            assert attempts[0]["intended_norm"] == pytest.approx(5e-05, rel=1e-6), key
            assert attempts[0]["survived"] and len(attempts) == 1, key
            records[(size, key)] = [p.detach().clone() - twins.theta0[n]
                                    for n, p in twins.trainable]  # fmt: skip
            fcd.set_trainable(twins.student, twins.theta0, twins.names)
        heuristic = facts["rounding_heuristic"]
        assert heuristic["norm"] > 0 and heuristic["label"] == (
            "heuristic, not a bound on accumulated error"
        )
        assert heuristic["coordinates"] == sum(p.numel() for _, p in twins.trainable)
    for key in fcd.PERTURBATIONS:  # the same perturbation at every size of the residual
        for size in (1e-3, 1e3):
            for a, b in zip(records[(1.0, key)], records[(size, key)], strict=True):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-12)
    direction, norm = fcd.first_update_direction(residual, twins.v0, eps)
    exact = [-(r.double() / (v.double().sqrt() + eps)) for r, v in zip(residual, twins.v0,
                                                                      strict=True)]  # fmt: skip
    for d, e in zip(direction, exact, strict=True):  # P in float64, then one fp32 rounding
        assert torch.equal(d, e.float())
    assert norm == pytest.approx(fcd._norm(exact), rel=1e-6)
    flat = torch.cat([u.reshape(-1) for u in exact])
    p1 = torch.cat([d.reshape(-1).double() for d in records[(1.0, "p1")]])
    p2 = torch.cat([d.reshape(-1).double() for d in records[(1.0, "p2")]])
    assert torch.nn.functional.cosine_similarity(p1, flat, dim=0) > 0.9999
    assert torch.nn.functional.cosine_similarity(p2, -flat, dim=0) > 0.9999
    # No positive, finite norm, no scale: that pair is then not run (incomplete)
    shared = shared_with([torch.zeros_like(r) for r in residual])
    fcd.seed_directions(s, twins, shared, eps)
    assert set(shared["scales"]) == {"p3", "p4"}


def test_the_rounding_heuristic() -> None:
    """sqrt(Σ (ulp/√12)²): for θ = ±1 (ulp 2⁻²³) over n coordinates, √n·2⁻²³/√12."""
    out = fcd.rounding_heuristic([torch.ones(300), torch.full((100,), -1.0)])
    assert out["norm"] == pytest.approx(math.sqrt(400) * 2.0**-23 / math.sqrt(12))
    assert out["coordinates"] == 400 and "not a bound" in out["label"]


MODES = {"adaptive": "adamw", "frozen": "adam_frozen", "prewarmed": "adamw_prewarmed"}
LR, BETA1, BETA2, EPS = 1e-2, 0.9, 0.999, 1e-8


def oracle(mode, start, curvature, v0, steps):
    """Adam's rule written out in float64 (optim.py's module docstring), independently of the
    code: m ← β₁m + (1−β₁)g; v ← β₂v + (1−β₂)g² (adaptive, from 0; pre-warmed, from v₀), fixed
    at v₀ (frozen); the denominator √(v/(1−β₂ᵗ)) + ε (adaptive), √v₀ + ε (frozen), √v + ε (pre-
    warmed, not bias-corrected); θ ← θ − lr/(1−β₁ᵗ)·m/denominator."""
    theta = start.double().clone()
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta) if mode == "adaptive" else v0.double().clone()
    for t in range(1, steps + 1):
        g = curvature * theta
        m = BETA1 * m + (1 - BETA1) * g
        if mode != "frozen":
            v = BETA2 * v + (1 - BETA2) * g * g
        denominator = torch.sqrt(v / (1 - BETA2**t)) if mode == "adaptive" else torch.sqrt(v)
        theta = theta - LR / (1 - BETA1**t) * m / (denominator + EPS)
    return theta


def quadratic():
    g = torch.Generator().manual_seed(3)
    start = torch.randn(64, generator=g, dtype=torch.float64)
    curvature = torch.rand(64, generator=g, dtype=torch.float64) * 4 + 0.1
    v0 = torch.rand(64, generator=g) * 9e-3 + 1e-3  # fp32, as v₀ is estimated
    return start, curvature, v0


def config_of(name):
    return OmegaConf.create({"name": name, "lr": LR, "eps": EPS, "beta1": BETA1,
                             "betas": [BETA1, BETA2]})  # fmt: skip


@pytest.mark.parametrize("mode", list(MODES))
def test_float64_adam_matches_a_float64_oracle(mode) -> None:
    """The float64 reference's arithmetic (``Float64Adam``, on float64 parameters here) against
    the rule written out in float64, on a quadratic: agreement to about 1e-14 relative in every
    mode (review fa12), in norm (a coordinate passing near zero has no relative error of its
    own to speak of), with every coordinate within 1e-14 of the largest."""
    start, curvature, v0 = quadratic()
    b = torch.nn.Parameter(start.clone())
    wide = fcd.Float64Adam([b], config_of(MODES[mode]), None if mode == "adaptive" else [v0])
    assert wide.mode == mode
    for _ in range(30):
        wide.step([curvature * b.detach()])
    expected = oracle(mode, start, curvature, v0, 30)
    assert not torch.equal(expected, start)
    difference = b.detach() - expected
    relative = (difference.norm() / expected.norm()).item()
    assert relative <= 1e-14, relative
    assert difference.abs().max() <= 1e-14 * expected.abs().max()
    if mode == "prewarmed":
        assert not torch.equal(wide.v[0], v0.double())  # the second moment moved from v₀
    else:
        with pytest.raises(fcd.DiagnosticError, match="needs v₀"):
            fcd.Float64Adam([b], config_of(MODES["prewarmed"]))


def test_the_prewarmed_reference_leaves_v0_alone() -> None:
    """Review M5: a float64 v₀ on the parameters' device is copied, not updated in place."""
    start, curvature, v0 = quadratic()
    v0 = v0.double()
    kept = v0.clone()
    b = torch.nn.Parameter(start.clone())
    wide = fcd.Float64Adam([b], config_of("adamw_prewarmed"), [v0])
    for _ in range(3):
        wide.step([curvature * b.detach()])
    assert torch.equal(v0, kept) and not torch.equal(wide.v[0], kept)


@pytest.mark.parametrize("mode", list(MODES))
def test_prewarmed_float64_reference_matches_optim(mode) -> None:
    """A sanity check at fp32 tolerance: on fp32 parameters, the float64 reference follows the
    code's fp32 ``optim.Adam`` in each mode, pre-warmed included."""
    start, curvature, v0 = quadratic()
    a, b = torch.nn.Parameter(start.float()), torch.nn.Parameter(start.float())
    betas = (BETA1, None) if mode == "frozen" else (BETA1, BETA2)
    code = Adam([a], lr=LR, betas=betas, eps=EPS, second_moment=mode,
                initial_second_moment=None if mode == "adaptive" else [v0])  # fmt: skip
    wide = fcd.Float64Adam([b], config_of(MODES[mode]), None if mode == "adaptive" else [v0])
    for _ in range(30):
        a.grad = (curvature * a.detach().double()).float()
        code.step()
        wide.step([curvature * b.detach().double()])
    assert b.dtype == torch.float32
    torch.testing.assert_close(b.data, a.data, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------- records and their states

ROUNDING = [{"scale": 1.0, "intended_norm": 5e-5, "realized_norm": 5e-5, "changed": 9,
             "survived": True}]  # fmt: skip


def stamped(record, commit=COMMIT, dirty=False, settings_=None):
    provenance = {"code": {"commit": commit, "dirty": dirty},
                  "settings": FROZEN if settings_ is None else settings_}  # fmt: skip
    return {**record, "provenance": provenance}


def pair(last=0.001, *, values=None, steps=EVAL_AT, rounding=ROUNDING, stop=None,
         diverged=False):  # fmt: skip
    """A pair's record as ``run_pairs`` writes it: D_twin 0.001 at every step but the last."""
    values = [0.001] * (len(steps) - 1) + [last] if values is None else values
    return {"D_twin": [{"step": s, "D_twin": v} for s, v in zip(steps, values, strict=True)],
            "rows": [], "stop": stop, "diverged": diverged, "rounding": rounding,
            "max_D_twin": fcd._largest(values)}  # fmt: skip


def attempt(case, n=1, *, pairs=None, status="complete", equal=True, reason=None,
            mode=fcd.FLOAT64_FULL, rounding=ROUNDING, settings_=None, named=None,
            intended=5e-05, **stamp):  # fmt: skip
    """A case's attempt as ``run_numerical_twins`` writes it; ``pairs`` maps a pair to its last
    D_twin or its record, or None to leave it out (every other pair within)."""
    built = {}
    for key in fcd.PAIRS:
        spec = (pairs or {}).get(key, 0.001)
        if spec is None:
            continue
        record = pair(spec, rounding=rounding) if isinstance(spec, int | float) else dict(spec)
        if key == fcd.REFERENCE:
            record.pop("rounding", None)
            record["mode"] = mode
            if mode == fcd.FLOAT64_ACCUMULATION:
                record["fallback_reason"] = "OutOfMemoryError: CUDA out of memory"
        built[key] = record
    body = {"part": "twins", "case": case.name, "seed": case.seed, "optimizer": case.optimizer,
            "lr": case.lr, "batch": case.batch, "U": case.upper, "attempt": n,
            "intended_norm": intended, "status": status, "reason": reason,
            "regular_run_identity": identity_of(case) if named is None else named,
            "twin_A": {"archive": {"equal": equal}}, "pairs": built}  # fmt: skip
    return stamped(body, settings_=settings_, **stamp)


def control(lr, last=1e-12, *, steps=EVAL_AT, status="complete", diverged=False, stop=None,
            reason=None, n=1, **stamp):  # fmt: skip
    """A same-path control's attempt as ``run_numerical_controls`` writes it."""
    values = [1e-12] * (len(steps) - 1) + [last] if steps else []
    case = fcd.control_case(scope(), lr, 20, 8)
    return stamped({"part": "control", "rate": lr, "seed": 2200, "batch": 20, "U": 8,
                    "attempt": n, "status": status, "reason": reason,
                    "regular_run_identity": identity_of(case),
                    "scoring": [{"step": s, "K_scoring": v} for s, v in zip(steps, values,
                                                                           strict=True)],
                    "monitor": [], "rows": [{"step": 0, "loss": 1e-12, "grad_norm": 2e-14}],
                    "stop": stop, "diverged": diverged, "first_gradient_norm": 2e-14},
                   **stamp)  # fmt: skip


def within_all(found):
    return {c.name: {1: attempt(c), 2: None} for c in found}


def qualify(records, found, state=CLEAN):
    return fcd.qualification(records, found, reader=reader(state))


def by_name(found, seed, arm, lr):
    return next(c for c in found if (c.seed, c.optimizer, c.lr) == (seed, arm, lr))


def test_failing_pair_makes_rate_failing(world) -> None:
    found = cases()
    upper_fa = by_name(found, 2201, "adam_frozen", 1.46e-05)
    key = "adam_frozen|1.46e-05"
    out = qualify(within_all(found), found)
    assert set(out["rates"].values()) == {"qualified"} and out["reasons"] == {}
    assert out["seeds"][key] == [2200, 2201, 2202]
    assert out["seeds"]["adam_frozen|1.21e-05"] == [2200]
    for pair_key in fcd.PAIRS:
        diverged = pair(values=[0.001, math.inf], steps=[1, 5], diverged=True,
                        stop="non-finite weights after step 7")  # fmt: skip
        other = "p4" if pair_key != "p4" else "p3"
        for record in (attempt(upper_fa, pairs={pair_key: 0.081}),
                       attempt(upper_fa, pairs={pair_key: diverged}),
                       # exceeding wins over an attempt otherwise incomplete
                       attempt(upper_fa, pairs={pair_key: 0.2, other: None}, status="incomplete",
                               reason="error: OutOfMemoryError")):  # fmt: skip
            records = within_all(found)
            records[upper_fa.name][1] = record
            out = qualify(records, found)
            assert out["rates"][key] == "failing", pair_key
            assert {v for k, v in out["rates"].items() if k != key} == {"qualified"}
            assert any(r.startswith(f"case s2201: attempt 1: {pair_key} exceeding")
                       for r in out["reasons"][key])  # fmt: skip
            entry = next(e for e in out["cases"]
                         if (e["seed"], e["arm"]) == (2201, "adam_frozen"))  # fmt: skip
            assert entry["state"] == "exceeding" and not entry["not_qualified"]
    records = within_all(found)
    records[upper_fa.name][1] = attempt(upper_fa, pairs={"p1": 0.081})
    assert qualify(records, found)["reasons"][key] == [
        "case s2201: attempt 1: p1 exceeding (max D_twin 0.081)"
    ]
    # At the bound is within; an exceeding first attempt fails even when a rerun is within
    records[upper_fa.name][1] = attempt(upper_fa, pairs={"p3": BOUND})
    assert qualify(records, found)["rates"][key] == "qualified"
    records[upper_fa.name] = {1: attempt(upper_fa, pairs={"p2": 0.1}), 2: attempt(upper_fa, 2)}
    assert qualify(records, found)["rates"][key] == "failing"
    # Failing (evidence) wins over not_qualified (technical) in another case of the rate
    other = by_name(found, 2200, "adam_frozen", 1.46e-05)
    records[other.name] = {1: attempt(other, equal=False), 2: attempt(other, 2, equal=False)}
    assert qualify(records, found)["rates"][key] == "failing"
    # A lost perturbation never counts as within; it can still exceed
    lost = [{"scale": 1.0, "intended_norm": 5e-5, "realized_norm": 0.0, "changed": 0,
             "survived": False},
            {"scale": 10.0, "intended_norm": 5e-4, "realized_norm": 0.0, "changed": 0,
             "survived": False}]  # fmt: skip
    records = within_all(found)
    records[upper_fa.name][1] = attempt(upper_fa, pairs={"p1": pair(rounding=lost)})
    assert qualify(records, found)["rates"][key] == "incomplete"
    records[upper_fa.name][1] = attempt(upper_fa, pairs={"p1": pair(0.2, rounding=lost)})
    assert qualify(records, found)["rates"][key] == "failing"


def test_records_are_bound_to_the_policy(world) -> None:
    """Review I1: a record counts only under the frozen settings and with the policy's
    perturbations: the first attempt at the norm 5e-5, a second only ×10 the first."""
    found = cases()
    c = found[0]

    def rate_of(record):
        records = within_all(found)
        records[c.name][1] = record
        out = qualify(records, found)
        return out["rates"][c.key], out["reasons"].get(c.key, [])

    tiny = [{**ROUNDING[0], "intended_norm": 1e-12, "realized_norm": 1e-12}]
    rate, why = rate_of(attempt(c, rounding=tiny))
    assert rate == "incomplete" and "not the policy's" in why[0] and "1e-12" in why[0]
    rate, why = rate_of(attempt(c, intended=5e-9))
    assert rate == "incomplete" and "the intended norm" in why[0]
    for changed in ({"norm": 5e-9}, {"gaussian": {"p3": 7, "p4": 8}}, {"rounding_scale": 1000},
                    {"survival_share": 0.1}):  # fmt: skip
        other = {**FROZEN, "perturbation": {**FROZEN["perturbation"], **changed}}
        rate, why = rate_of(attempt(c, settings_=other))
        assert rate == "incomplete" and "other settings" in why[0], changed
    rate, _ = rate_of(attempt(c, settings_={**FROZEN, "eval_at": [1, 25]}))
    assert rate == "incomplete"
    first = {**ROUNDING[0], "realized_norm": 1e-6, "survived": False}
    for second, ok in (({"scale": 10.0, "intended_norm": 5e-4}, True),
                       ({"scale": 1000.0, "intended_norm": 5e-2}, False),
                       ({"scale": 10.0, "intended_norm": 5e-2}, False),
                       ({"scale": 10.0000001, "intended_norm": 5e-4}, False)):  # fmt: skip
        retry = [first, {**second, "realized_norm": second["intended_norm"], "changed": 9,
                         "survived": True}]  # fmt: skip
        rate, why = rate_of(attempt(c, rounding=retry))
        assert (rate == "qualified") == ok, (second, why)
    three = [first, {"scale": 10.0, "intended_norm": 5e-4, "realized_norm": 0.0}] * 2
    assert rate_of(attempt(c, rounding=three))[0] == "incomplete"
    # The memory rule is not part of the policy: a record made under another still counts
    assert rate_of(attempt(c, settings_={**FROZEN, "memory_rule_gb": 1}))[0] == "qualified"


def test_incomplete_twice_is_not_qualified(world) -> None:
    """Review fa12: a case still incomplete after its one rerun makes its rate not_qualified, a
    technical verdict (unusable at that batch, never called instability), not failing."""
    found = cases()
    case = by_name(found, 2200, "adamw_prewarmed", 1.1e-05)
    key = "adamw_prewarmed|1.1e-05"
    for first in (
        attempt(case, equal=False),
        attempt(case, status="incomplete", reason="error: CUDA error"),
        attempt(case, pairs={"reference": None}),  # a missing pair
        attempt(case, pairs={"p2": pair(values=[0.001] * 5, steps=EVAL_AT[:5])}),
        attempt(case, mode="float32"),
    ):  # a reference outside the policy
        records = within_all(found)
        records[case.name] = {1: first, 2: None}
        out = qualify(records, found)
        assert out["rates"][key] == "incomplete"  # its rerun is due
        assert out["reasons"][key][0].startswith("case s2200: incomplete, its rerun not yet done")
        assert fcd.retry_due(records, found, reader=reader()) == [case]
        records[case.name][2] = attempt(case, 2, equal=False)
        out = qualify(records, found)
        assert out["rates"][key] == "not_qualified"
        reason = out["reasons"][key][0]
        assert reason.startswith("case s2200: not qualified (technical), incomplete after its "
                                 "rerun")  # fmt: skip
        assert "instab" not in reason and "exceeding" not in reason
        entry = next(e for e in out["cases"]
                     if e["arm"] == "adamw_prewarmed" and e["lr"] == 1.1e-5)  # fmt: skip
        assert (entry["state"], entry["not_qualified"], entry["attempts"]) == ("incomplete", True,
                                                                               2)  # fmt: skip
        assert entry["attempt_states"] == ["incomplete", "incomplete"]
        assert not fcd.retry_due(records, found, reader=reader())
        records[case.name][2] = attempt(case, 2)  # a complete rerun within the bound
        assert qualify(records, found)["rates"][key] == "qualified"
    # No first attempt, or one that does not count, is incomplete and not a rerun's due
    for first in (
        None,
        attempt(case, commit=OTHER),
        attempt(case, commit="u" * 40),
        attempt(case, dirty=True),
        attempt(by_name(found, 2200, "adamw", 1.12e-05)),
    ):  # another case's record
        records = within_all(found)
        records[case.name] = {1: first, 2: None}
        assert qualify(records, found)["rates"][key] == "incomplete"
        assert not fcd.retry_due(records, found, reader=reader())
    # A checkout that is not clean reads nothing
    out = qualify(within_all(found), found, state={**CLEAN, "dirty": True})
    assert set(out["rates"].values()) == {"incomplete"}


def test_an_instability_stop_makes_the_rate_failing(world) -> None:
    """Review M6: a regular run stopped by an instability rule is evidence against its rate:
    failing, with the reason prefix ``instability:``; its FA control is rejected."""
    found = cases()
    case = by_name(found, 2201, "adam_frozen", 1.46e-05)
    stopped = ("instability: the regular run stopped by 'gradient_spike' after 4 steps, "
               '{"grad_norm": 9.0, "reference_median": 0.5}')  # fmt: skip
    records = within_all(found)
    records[case.name][1] = attempt(case, status="instability", reason=stopped, pairs={
        key: None for key in fcd.PAIRS})  # fmt: skip
    out = qualify(records, found)
    assert out["rates"][case.key] == "failing"
    assert out["reasons"][case.key] == [
        "instability: case s2201: the regular run stopped by 'gradient_spike' after 4 steps, "
        '{"grad_norm": 9.0, "reference_median": 0.5}'
    ]
    entry = next(e for e in out["cases"] if (e["seed"], e["arm"]) == (2201, "adam_frozen"))
    assert entry["state"] == "instability" and not entry["not_qualified"]
    assert not fcd.retry_due(records, found, reader=reader())
    verdict = control_of(control(1.46e-05, status="instability", reason=stopped), rate=1.46e-05)
    assert verdict["verdict"] == "rejected" and verdict["reasons"] == [stopped]


def test_the_reference_fallback_is_recorded_per_case(world) -> None:
    found = cases()
    case = by_name(found, 2202, "adamw", 1.6e-05)
    records = within_all(found)
    records[case.name][1] = attempt(case, mode=fcd.FLOAT64_ACCUMULATION)
    out = qualify(records, found)
    assert out["rates"]["adamw|1.6e-05"] == "qualified"  # the fallback is part of the policy
    entry = next(e for e in out["cases"] if (e["seed"], e["arm"]) == (2202, "adamw"))
    assert entry["reference_mode"] == fcd.FLOAT64_ACCUMULATION
    assert entry["fallback_reason"].startswith("OutOfMemoryError")
    assert {e["reference_mode"] for e in out["cases"] if e is not entry} == {fcd.FLOAT64_FULL}


def test_a_record_counts_only_for_the_run_located_now(world) -> None:
    """Review M3: a regular run relaunched, or whose records changed, after the twins leaves
    the old record uncounted."""
    found = cases()
    case = found[4]
    records = within_all(found)
    assert qualify(records, found)["rates"][case.key] == "qualified"
    run = fcd.locate_case(world, scope(), case)
    metrics = run / "metrics.jsonl"
    metrics.write_text(metrics.read_text() + json.dumps({"step": 1, "loss": 0.5}) + "\n")
    out = qualify(records, found)
    assert out["rates"][case.key] == "incomplete"
    assert "not the one located now" in out["reasons"][case.key][0]
    relaunched = run.with_name(run.name + "-relaunch")
    run.rename(relaunched)  # a new run id, the old directory gone
    records[case.name][1] = attempt(case)  # a record of the relaunched run counts
    assert qualify(records, found)["rates"][case.key] == "qualified"
    relaunched.rename(run)
    assert qualify(records, found)["rates"][case.key] == "incomplete"
    # A complete record that names no run never counts
    nameless = attempt(case)
    nameless["regular_run_identity"] = None
    records[case.name][1] = nameless
    assert "names no regular run" in qualify(records, found)["reasons"][case.key][0]


def control_of(first, rerun=None, rate=1.21e-05, **changes):
    return fcd.control_verdict({1: first, 2: rerun}, rate=rate, batch=20, upper=8,
                               reader=reader({**CLEAN, **changes}))  # fmt: skip


def test_same_path_control_accept_and_reject(world) -> None:
    good = control_of(control(1.21e-05, 0.01))
    assert good["verdict"] == "accepted" and good["measured"] == "below" and not good["reasons"]
    assert good["first_gradient_norm"] == 2e-14 and good["attempts"] == 1
    rejected = {  # measured: exceeding or at the bound
        "at the bound": control(1.21e-05, BOUND),
        "above": control(1.21e-05, BOUND * 1.0001),
        "non-finite": control(1.21e-05, math.nan),
        "diverged": control(1.21e-05, 0.01, steps=[1], diverged=True,
                            stop="non-finite loss or gradient at step 3"),
    }  # fmt: skip
    for name, record in rejected.items():
        out = control_of(record)
        assert out["verdict"] == "rejected" and out["reasons"], name
        assert control_of(control(1.21e-05, 0.01), record)["verdict"] == "rejected", name
    assert control_of(control(1.21e-05, BOUND))["measured"] == "at_bound"
    incomplete = {  # not evidence: a rerun is due
        "one point": control(1.21e-05, 0.01, steps=[1]),
        "five steps": control(1.21e-05, 0.01, steps=EVAL_AT[:5]),
        "stopped": control(1.21e-05, 0.01, stop="interrupted"),
        "shared state": control(1.21e-05, 0.01, status="incomplete",
                                reason="the control's ['training_stream'] differ"),
        "an error": control(1.21e-05, 0.01, steps=[], status="incomplete",
                            reason="error: OutOfMemoryError"),
    }  # fmt: skip
    for name, record in incomplete.items():
        out = control_of(record)
        assert out["verdict"] == "incomplete" and out["reasons"], name
        assert out["reasons"][0].startswith("incomplete, its rerun not yet done"), name
        assert control_of(record, control(1.21e-05, 0.01, n=2))["verdict"] == "accepted", name
        twice = control_of(record, record)
        assert twice["verdict"] == "not_qualified", name
        assert twice["reasons"][0].startswith("not qualified (technical)"), name
    for name, record in {"no record": None, "another GPU code": control(1.21e-05, commit=OTHER),
                         "dirty": control(1.21e-05, dirty=True),
                         "other settings": control(1.21e-05, settings_={**FROZEN, "seeds": [1]}),
                         "another rate": control(1.46e-05)}.items():  # fmt: skip
        assert control_of(record)["verdict"] == "incomplete", name
        assert control_of(record, control(1.21e-05, n=2))["verdict"] == "incomplete", name
    assert control_of(control(1.21e-05, 0.01), dirty=True)["verdict"] == "incomplete"
    # Written at an analysis-only later commit (the same GPU code), it counts
    assert control_of(control(1.21e-05, commit=LATER))["verdict"] == "accepted"
    records = {1.21e-05: {1: control(1.21e-05), 2: None},
               1.46e-05: {1: control(1.46e-05, steps=[1]), 2: None}}  # fmt: skip
    rates = [1.21e-05, 1.46e-05, 9.98e-06]
    assert fcd.control_verdicts(records, rates=rates, batch=20, upper=8, reader=reader()) == {
        1.21e-05: "accepted", 1.46e-05: "incomplete", 9.98e-06: "incomplete"}  # fmt: skip
    assert fcd.control_retry_due(records, rates, batch=20, upper=8,
                                 reader=reader()) == [1.46e-05]  # fmt: skip


def write_records(out, records, controls, found=None):
    by = {c.name: c for c in (found or cases() + cases(RESOLVED))}
    for name, attempts in records.items():
        for n, record in attempts.items():
            if record is not None:
                fcd.write_json(out / by[name].attempt_path(n), record)
    for lr, attempts in controls.items():
        for n, record in attempts.items():
            if record is not None:
                fcd.write_json(out / fcd.control_path(lr, n), record)


def script_argv(part, root, rates=RATES, *extra):
    argv = [str(SCRIPT), part, "--batch", "20", "--upper", "8", "--runs-root", str(root)]
    for arm, values in rates.items():
        argv += ["--rates", f"{arm}={','.join(repr(r) for r in values)}"]
    return [*argv, *extra]


def call(monkeypatch, argv, *, untracked=False):
    """The script's ``main``, as on the command line; ``untracked`` is the tests' own override
    of the tracked settings."""
    monkeypatch.setattr("sys.argv", argv)
    namespace = runpy.run_path(str(SCRIPT))
    try:
        namespace["main"](untracked_config_for_tests=untracked)
    except SystemExit as exc:
        return exc.code
    return 0


def test_verdict_schema(world, monkeypatch, capsys) -> None:
    """The record the selection reads, at RUNS_DIR/numerical/b{B}-u{U}/verdicts/<digest>.json
    (and its copy verdict.json), through the script; ``qualified`` iff every rate is qualified
    and every control accepted; the policy with the regular runs' GPU code (review fa13)."""
    found = cases()
    root = world
    out = fcd.numerical_root(20, 8, root)
    records = within_all(found)
    controls = {lr: {1: control(lr), 2: None} for lr in RATES["adam_frozen"]}
    write_records(out, records, controls)

    def script(*extra):
        return call(monkeypatch, script_argv("verdict", root, RATES, *extra))

    assert script() == 0
    digest = fcd.selection_digest(RATES)
    keys = sorted(f"{a}|{r:g}" for a in ARMS for r in RATES[a])
    assert digest == hashlib.sha256("\n".join(keys).encode()).hexdigest()[:12]
    verdict = json.loads((out / "verdicts" / f"{digest}.json").read_text())
    assert json.loads((out / "verdict.json").read_text()) == verdict
    assert list(verdict) == VERDICT_KEYS
    assert (verdict["batch"], verdict["U"], verdict["bound"]) == (20, 8, BOUND)
    assert isinstance(verdict["batch"], int) and isinstance(verdict["U"], int)
    assert verdict["commit"] == COMMIT and fcd.output_commit(verdict) == COMMIT
    assert verdict["policy"] == scope().policy(8, GPU_CODE, [COMMIT])
    assert verdict["policy"]["tool_commit"] == [COMMIT]
    assert verdict["provenance"]["writers"] == {COMMIT: fake_reuse(COMMIT, COMMIT)}
    assert verdict["provenance"]["selection"] == {"digest": digest, "keys": keys}
    assert verdict["provenance"]["code_check"]["consistent"] is True
    assert verdict["rates"] == {f"{a}|{r:g}": "qualified" for a in ARMS for r in sorted(RATES[a])}
    assert verdict["seeds"]["adamw|1.6e-05"] == [2200, 2201, 2202]
    assert verdict["seeds"]["adamw|1.12e-05"] == [2200]
    assert set(verdict["seeds"]) == set(verdict["rates"])
    assert verdict["controls"] == {"9.98e-06": "accepted", "1.21e-05": "accepted",
                                   "1.46e-05": "accepted"}  # fmt: skip
    assert verdict["qualified"] is True and verdict["reasons"] == {}
    assert verdict["reference_fallback"] == []
    assert len(verdict["cases"]) == 15
    for entry, case in zip(verdict["cases"], found, strict=True):
        assert {"seed", "arm", "lr", "state", "attempts", "reference_mode",
                "fallback_reason"} <= set(entry)  # fmt: skip
        assert (entry["seed"], entry["arm"], entry["lr"]) == (case.seed, case.optimizer, case.lr)
        assert (entry["state"], entry["attempts"]) == ("within", 1)
    assert verdict["provenance"]["retry_due"] == {"cases": [], "controls": []}
    assert script("--retry-needed") == 3

    def outcome(changed_records=None, changed_controls=None):
        shutil.rmtree(out)
        write_records(out, {**records, **(changed_records or {})},
                      {**controls, **(changed_controls or {})})  # fmt: skip
        assert script() == 0
        v = json.loads((out / "verdict.json").read_text())
        assert set(v["rates"].values()) <= STATES and set(v["controls"].values()) <= CONTROL_STATES
        assert v["qualified"] == (
            set(v["rates"].values()) == {"qualified"}
            and set(v["controls"].values()) == {"accepted"}
        )
        return v

    # A control rejected, one not qualified, one with its rerun due: not qualified
    v = outcome(changed_controls={
        1.21e-05: {1: control(1.21e-05, BOUND), 2: None},
        1.46e-05: {1: control(1.46e-05, steps=[1]), 2: control(1.46e-05, steps=[1], n=2)},
        9.98e-06: {1: control(9.98e-06, steps=[1]), 2: None}})  # fmt: skip
    assert v["qualified"] is False and set(v["rates"].values()) == {"qualified"}
    assert v["controls"] == {"9.98e-06": "incomplete", "1.21e-05": "rejected",
                             "1.46e-05": "not_qualified"}  # fmt: skip
    assert v["reasons"]["adam_frozen|1.21e-05"][0].startswith("control: attempt 1: at_bound")
    assert v["reasons"]["adam_frozen|1.46e-05"][0].startswith("control: not qualified "
                                                              "(technical)")  # fmt: skip
    assert v["provenance"]["retry_due"] == {"cases": [], "controls": ["9.98e-06"]}
    assert script("--retry-needed") == 0
    v = outcome(changed_controls={1.46e-05: {1: None, 2: None}})
    assert v["qualified"] is False and v["controls"]["1.46e-05"] == "incomplete"
    assert v["reasons"]["adam_frozen|1.46e-05"] == ["control: no first attempt: no record"]
    # One rate failing, one not qualified, one incomplete with its rerun due, one fallback
    failing, technical, due, fallback = found[0], found[9], found[13], found[14]
    v = outcome({failing.name: {1: attempt(failing, pairs={"p1": 0.081}), 2: None},
                 technical.name: {1: attempt(technical, equal=False),
                                  2: attempt(technical, 2, equal=False)},
                 due.name: {1: attempt(due, equal=False), 2: None},
                 fallback.name: {1: attempt(fallback, mode=fcd.FLOAT64_ACCUMULATION),
                                 2: None}})  # fmt: skip
    assert v["qualified"] is False
    assert v["rates"][failing.key] == "failing" and v["rates"][technical.key] == "not_qualified"
    assert v["rates"][due.key] == "incomplete" and v["rates"][fallback.key] == "qualified"
    assert v["reasons"][failing.key] == [f"case s{failing.seed}: attempt 1: p1 exceeding (max "
                                         "D_twin 0.081)"]  # fmt: skip
    assert v["reference_fallback"] == [{"case": fallback.name, "rate": fallback.key,
                                        "seed": fallback.seed,
                                        "fallback_reason": "OutOfMemoryError: CUDA out of "
                                                           "memory"}]  # fmt: skip
    assert v["provenance"]["retry_due"] == {"cases": [due.name], "controls": []}
    # Records of other GPU code are stale: nothing qualifies, and none is a writer
    v = outcome({c.name: {1: attempt(c, commit=OTHER), 2: None} for c in found})
    assert v["qualified"] is False and set(v["rates"].values()) == {"incomplete"}
    assert v["policy"]["tool_commit"] == [COMMIT]  # the controls' records still count
    assert v["provenance"]["writers"][OTHER]["reusable"] is False
    # --dry-run and --retry-needed write nothing
    shutil.rmtree(out)
    write_records(out, records, controls)
    assert script("--dry-run") == 0 and script("--retry-needed") == 3
    assert not (out / "verdicts").exists() and not (out / "verdict.json").exists()
    # Review fa13: the regular runs' GPU code unlike the check's, no record counts
    monkeypatch.setitem(GPU_SHA, RUN_COMMIT, "other")
    v = outcome()
    assert v["qualified"] is False and set(v["rates"].values()) == {"incomplete"}
    assert set(v["controls"].values()) == {"incomplete"} and v["policy"]["gpu_code"] is None
    for key in v["rates"]:
        assert v["reasons"][key][0].startswith("code: the regular runs' GPU code is not the "
                                               "check's")  # fmt: skip
    assert v["provenance"]["code_check"]["consistent"] is False
    capsys.readouterr()


def test_every_selection_keeps_its_verdict(world, monkeypatch, capsys) -> None:
    """Review I4: the verdict that made a rate failing stays when the re-solved selection's is
    written; the same selection's earlier verdict moves to the history; a digest file holding
    another selection is never overwritten."""
    root = world
    out = fcd.numerical_root(20, 8, root)
    found = cases() + [c for c in cases(RESOLVED) if c not in cases()]
    records = {c.name: {1: attempt(c), 2: None} for c in found}
    failing = by_name(cases(), 2200, "adam_frozen", 1.46e-05)
    records[failing.name][1] = attempt(failing, pairs={"p1": 0.081})
    controls = {lr: {1: control(lr), 2: None} for lr in (9.98e-06, 1.05e-05, 1.21e-05, 1.46e-05)}
    write_records(out, records, controls, found)
    assert call(monkeypatch, script_argv("verdict", root, RATES)) == 0
    assert call(monkeypatch, script_argv("verdict", root, RESOLVED)) == 0
    first = json.loads((out / "verdicts" / f"{fcd.selection_digest(RATES)}.json").read_text())
    second = json.loads((out / "verdicts" / f"{fcd.selection_digest(RESOLVED)}.json").read_text())
    assert first["rates"]["adam_frozen|1.46e-05"] == "failing" and not first["qualified"]
    assert second["qualified"] is True and "adam_frozen|1.46e-05" not in second["rates"]
    assert json.loads((out / "verdict.json").read_text()) == second  # the latest
    assert call(monkeypatch, script_argv("verdict", root, RATES)) == 0
    history = list((out / "verdicts" / "history" / fcd.selection_digest(RATES)).iterdir())
    assert len(history) == 1 and json.loads(history[0].read_text())["rates"] == first["rates"]
    assert json.loads((out / "verdict.json").read_text())["rates"] == first["rates"]
    # A file at a selection's path that holds another selection is refused, untouched
    path = fcd.verdict_path(out, RESOLVED)
    path.write_text(json.dumps(first))
    with pytest.raises(fcd.NumericalError, match="another selection"):
        fcd.write_verdict(out, second, RESOLVED)
    assert json.loads(path.read_text()) == first
    capsys.readouterr()


PRIMARY = {arm: RATES[arm] for arm in ("adamw", "adam_frozen")}


def test_a_primary_only_selection(world, monkeypatch, capsys) -> None:
    """A selection of the primary arms alone (the candidate's fallback without the pre-warmed
    arm): 10 cases, one control per FA rate, a verdict over what was selected with its arms
    recorded, under the digest of its own keys. Any other subset is refused."""
    s = scope()
    assert s.primary_arms == ("adamw", "adam_frozen")
    found = cases(PRIMARY)
    assert len(found) == 10 and {c.optimizer for c in found} == set(PRIMARY)
    assert {(c.optimizer, c.lr) for c in found if c.seed == 2200} == {
        (a, r) for a in PRIMARY for r in RATES[a]}  # fmt: skip
    for seed in (2201, 2202):
        assert {(c.optimizer, c.lr) for c in found if c.seed == seed} == {
            (arm, max(RATES[arm])) for arm in PRIMARY}  # fmt: skip
    for arms in (("adamw_prewarmed",), ("adamw",), ("adam_frozen",),
                 ("adamw", "adamw_prewarmed"), ("adam_frozen", "adamw_prewarmed")):  # fmt: skip
        with pytest.raises(fcd.NumericalError, match="primary arms"):
            cases({arm: RATES[arm] for arm in arms})
    root = world
    out = fcd.numerical_root(20, 8, root)
    records = {c.name: {1: attempt(c), 2: None} for c in found}
    controls = {lr: {1: control(lr), 2: None} for lr in RATES["adam_frozen"]}
    write_records(out, records, controls)
    assert call(monkeypatch, script_argv("verdict", root, PRIMARY)) == 0
    verdict = json.loads(fcd.verdict_path(out, PRIMARY).read_text())
    keys = sorted(f"{a}|{r:g}" for a in PRIMARY for r in RATES[a])
    assert fcd.selection_digest(PRIMARY) == hashlib.sha256(
        "\n".join(keys).encode()).hexdigest()[:12] != fcd.selection_digest(RATES)  # fmt: skip
    assert list(verdict) == VERDICT_KEYS and verdict["arms"] == ["adamw", "adam_frozen"]
    assert sorted(verdict["rates"]) == keys and set(verdict["rates"].values()) == {"qualified"}
    assert verdict["controls"] == dict.fromkeys(["9.98e-06", "1.21e-05", "1.46e-05"], "accepted")
    assert verdict["qualified"] is True and len(verdict["cases"]) == 10
    assert verdict["provenance"]["selection"]["keys"] == keys
    # A pre-warmed record that is there anyway is not read by this verdict
    assert not any("prewarmed" in entry["arm"] for entry in verdict["cases"])
    # The full selection's verdict is another file, the primary one kept
    write_records(out, within_all(cases()), {})
    assert call(monkeypatch, script_argv("verdict", root, RATES)) == 0
    assert json.loads(fcd.verdict_path(out, RATES).read_text())["arms"] == list(ARMS)
    assert json.loads(fcd.verdict_path(out, PRIMARY).read_text()) == verdict
    capsys.readouterr()


def snapshot(directory: Path) -> dict[str, bytes]:
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob("*"))
            if p.is_file()}  # fmt: skip


def test_a_dry_run_moves_nothing(world, monkeypatch, capsys) -> None:
    """Review I2: every --dry-run (and --retry-needed) only reads: at a later commit with the
    same GPU code, at a commit with other GPU code, and from a dirty checkout, ``out/`` stays
    byte-identical. A verdict from a dirty checkout is refused; a later analysis-only commit
    keeps every record (review I3: nothing is due)."""
    root = world
    out = fcd.numerical_root(20, 8, root)
    records = within_all(cases())
    records[cases()[2].name][1] = attempt(cases()[2], equal=False)  # a rerun is due
    controls = {lr: {1: control(lr), 2: None} for lr in RATES["adam_frozen"]}
    write_records(out, records, controls)
    assert call(monkeypatch, script_argv("verdict", root)) == 0
    before = snapshot(out)
    for state in (
        {"commit": LATER, "dirty": False},
        {"commit": OTHER, "dirty": False},
        {"commit": COMMIT, "dirty": True},
        {"commit": None, "dirty": None},
    ):
        monkeypatch.setattr(fcd, "git_state", lambda *a, state=state, **k: dict(state))
        for argv in (script_argv("control", root, RATES, "--dry-run"),
                     script_argv("twins", root, RATES, "--dry-run"),
                     script_argv("control", root, RATES, "--dry-run", "--retry"),
                     script_argv("twins", root, RATES, "--dry-run", "--retry"),
                     script_argv("verdict", root, RATES, "--dry-run"),
                     script_argv("verdict", root, RATES, "--retry-needed"),
                     script_argv("plan", root, RATES, "--dry-run")):  # fmt: skip
            assert call(monkeypatch, argv) in (0, 3), (state, argv[1:3])
        assert snapshot(out) == before, state
        printed = capsys.readouterr().out
        dues = [json.loads(line)["due"] for line in printed.splitlines()
                if line.startswith('{"part"')]  # fmt: skip
        if state["commit"] == LATER:  # the same GPU code: every record counts
            assert dues == [[], [], [], [cases()[2].name]]
        else:  # nothing counts: everything would be redone, nothing is moved now
            assert dues[:2] == [[f"{lr:g}" for lr in sorted(RATES["adam_frozen"])],
                                [c.name for c in cases()]]  # fmt: skip
    monkeypatch.setattr(fcd, "git_state", lambda *a, **k: {**CLEAN, "dirty": True})
    for part in ("verdict", "control", "twins"):
        assert "not clean" in str(call(monkeypatch, script_argv(part, root))), part
    assert snapshot(out) == before


def test_a_shadowing_config_is_refused(tmp_path, world, monkeypatch, capsys) -> None:
    """Review N2: a file ``experiments/numerical_check.yaml`` in the working directory (which
    ``load_config`` reads before the tracked one) is not the tracked settings: the writing
    parts refuse it; a dry run only prints."""
    shadow = tmp_path / "cwd"
    (shadow / "experiments").mkdir(parents=True)
    OmegaConf.save(settings(numerical={"perturbation": {"norm": 5e-9}}),
                   shadow / "experiments" / "numerical_check.yaml")  # fmt: skip
    monkeypatch.chdir(shadow)
    namespace = runpy.run_path(str(SCRIPT))
    assert not namespace["tracked_config"]("experiments/numerical_check")
    assert namespace["config_path"]("experiments/numerical_check") == (
        shadow / "experiments" / "numerical_check.yaml").resolve()  # fmt: skip
    out = fcd.numerical_root(20, 8, world)
    write_records(out, within_all(cases()), {lr: {1: control(lr), 2: None}
                                             for lr in RATES["adam_frozen"]})  # fmt: skip
    before = snapshot(out)
    for part in ("verdict", "twins", "control"):
        refused = str(call(monkeypatch, script_argv(part, world)))
        assert "tracked settings" in refused and str(shadow.resolve()) in refused, part
    assert call(monkeypatch, script_argv("verdict", world, RATES, "--dry-run")) == 0
    driver = [str(SCRIPT), "driver", "--batch", "20", "--upper", "8", "--commit", "a" * 40,
              "--out", str(tmp_path / "driver.sh")]  # fmt: skip
    for arm, values in RATES.items():
        driver += ["--rates", f"{arm}={','.join(repr(r) for r in values)}"]
    assert "tracked settings" in str(call(monkeypatch, driver))
    assert snapshot(out) == before and not (tmp_path / "driver.sh").exists()
    monkeypatch.chdir(ROOT)  # the same name resolves to the tracked file again
    assert namespace["tracked_config"]("experiments/numerical_check")
    assert call(monkeypatch, script_argv("verdict", world)) == 0
    capsys.readouterr()


def test_a_verdict_counts_its_records_at_another_runs_root(tmp_path, world, monkeypatch,
                                                           capsys) -> None:  # fmt: skip
    """Review N3: a record names its run relative to the runs root, so a verdict recomputed with
    RUNS_DIR restored elsewhere counts the same records."""
    out = fcd.numerical_root(20, 8, world)
    write_records(out, within_all(cases()), {lr: {1: control(lr), 2: None}
                                             for lr in RATES["adam_frozen"]})  # fmt: skip
    assert identity_of(cases()[0])["run"] == f"study2_short/{cases()[0].name}"
    assert "path" not in identity_of(cases()[0])
    assert call(monkeypatch, script_argv("verdict", world)) == 0
    first = json.loads((out / "verdict.json").read_text())
    restored = tmp_path / "restored" / "runs"
    restored.parent.mkdir()
    shutil.move(world, restored)
    assert call(monkeypatch, script_argv("verdict", restored)) == 0
    again = json.loads((fcd.numerical_root(20, 8, restored) / "verdict.json").read_text())
    assert first["qualified"] is again["qualified"] is True
    assert again["rates"] == first["rates"] and again["controls"] == first["controls"]
    assert [e["attempts"] for e in again["cases"]] == [1] * 15
    capsys.readouterr()


def test_the_plan_notes_an_instability_stop(world, monkeypatch, capsys) -> None:
    """Review N1 and N4: a regular run that an instability rule stopped, conforming otherwise,
    is found with a note, and is no problem for the plan; one that also departs from the
    prescribed run (here its protocol version) stays a problem."""
    namespace = runpy.run_path(str(SCRIPT))
    s = scope()
    found = cases()
    stopped, departing = found[11], found[14]
    for case in (stopped, departing):
        run = fcd.locate_case(world, s, case)
        (run / "terminal.json").write_text(
            json.dumps({"stop": {"reason": "gradient_spike", "kind": "instability"}, "steps": 3})
        )
    run = fcd.locate_case(world, s, departing)
    cfg = OmegaConf.load(run / "config.yaml")
    cfg.protocol_version = 3
    OmegaConf.save(cfg, run / "config.yaml")
    located, problems, notes = namespace["find_runs"](world, s, found)
    assert len(located) == 15 and set(problems) == {departing.name}
    assert "protocol_version is 3, not 2" in problems[departing.name]
    assert set(notes) == {stopped.name}
    assert notes[stopped.name].startswith(
        "instability: the parts will record the rate as "
        "failing (the regular run stopped by 'gradient_spike'"
    )
    assert call(monkeypatch, script_argv("plan", world, RATES, "--no-stream-check")) == 1
    printed = capsys.readouterr().out
    assert f"{stopped.name}: " in printed and "NOTE: instability" in printed
    assert f"{departing.name}: PROBLEM: does not conform" in printed


def test_the_plan_notes_runs_given_up(tmp_path, world, monkeypatch, capsys) -> None:
    """The final review of gate B, I1: a case whose regular run the launcher gave up on, named
    in the selection record's ``given_up``, is no problem for the plan but a note: the parts
    record it incomplete, then not qualified. A seed whose only FA run is given up has its other
    cases noted too (that run sets the seed's v₀ and p1, p2 up). Unnamed, a missing run stays a
    problem; a name that is not a case's is refused; the driver passes the names on."""
    namespace = runpy.run_path(str(SCRIPT))
    s, found = scope(), cases()
    adamw = by_name(found, 2200, "adamw", 1.32e-05)
    fa_2200 = by_name(found, 2200, "adam_frozen", 9.98e-06)
    fa_2201 = by_name(found, 2201, "adam_frozen", 1.46e-05)
    given = [adamw.name, fa_2200.name, fa_2201.name]
    for case in (adamw, fa_2200, fa_2201):
        shutil.rmtree(fcd.locate_case(world, s, case))
    assert set(namespace["find_runs"](world, s, found)[1]) == set(given)
    located, problems, notes = namespace["find_runs"](world, s, found, given)
    assert problems == {} and len(located) == 12
    note = "given up by the launcher: the parts will record it not qualified"
    assert notes[adamw.name] == namespace["GIVEN_UP"] == note
    assert notes[fa_2200.name] == namespace["GIVEN_UP"] + " (and its same-path control)"
    seed_2201 = {c.name for c in found if c.seed == 2201} - {fa_2201.name}
    assert set(notes) == {*given, *seed_2201}
    for name in seed_2201:
        assert notes[name].startswith("the seed's adam_frozen runs are given up by the launcher")
    # The plan through the script, the selection record naming the cases given up
    record = tmp_path / "selection.json"
    record.write_text(json.dumps({"rates": RATES, "batch": 20, "U": 8, "given_up": given}))
    plan = [str(SCRIPT), "plan", "--selection", str(record), "--batch", "20", "--upper", "8",
            "--runs-root", str(world), "--no-stream-check"]  # fmt: skip
    assert call(monkeypatch, plan) == 0
    printed = capsys.readouterr().out
    assert f"{adamw.name}: no regular run; NOTE: given up by the launcher" in printed
    assert "PROBLEM" not in printed and "run environment: the regular runs' equals" in printed
    record.write_text(json.dumps({"rates": RATES, "batch": 20, "U": 8, "given_up": given[:2]}))
    assert call(monkeypatch, plan) == 1  # the FA run of 2201 is missing and not given up
    assert f"{fa_2201.name}: PROBLEM:" in capsys.readouterr().out
    record.write_text(json.dumps({"rates": RATES, "batch": 20, "U": 8,
                                  "given_up": ["s2200-adamw-9e-06"]}))  # fmt: skip
    assert "name no case of this check" in str(call(monkeypatch, plan))
    both = script_argv("plan", world, RATES, "--given-up", adamw.name, "--selection", str(record))
    assert "Give the rates by --rates" in str(call(monkeypatch, both))
    # The streams: a seed whose FA runs are all given up has none to check, noted
    script_globals = namespace["check_streams"].__globals__
    monkeypatch.setitem(script_globals, "guard_prompt_sets", lambda env: env)
    monkeypatch.setitem(script_globals, "stream_inputs", lambda cfg, env: (["ok"], []))
    monkeypatch.setattr("zero_field_refusal.runtime.environment", lambda cfg, device: None)
    settings_ = settings()
    have, missing = namespace["check_streams"](settings_, found, world, "cpu", given)
    assert missing == [] and "seed 2201: no stream checked: its adam_frozen runs are given up by " \
        "the launcher" in have  # fmt: skip
    assert namespace["check_streams"](settings_, found, world, "cpu")[1] == [
        "seed 2201: no adam_frozen run to locate its streams by"]  # fmt: skip
    # The driver written from the selection record passes them to every part
    record.write_text(json.dumps({"rates": RATES, "batch": 20, "U": 8, "given_up": given}))
    driver = tmp_path / "driver.sh"
    assert call(monkeypatch, [str(SCRIPT), "driver", "--selection", str(record), "--batch", "20",
                              "--upper", "8", "--commit", "a" * 40, "--out",
                              str(driver)]) == 0  # fmt: skip
    args = next(line for line in driver.read_text().splitlines() if line.startswith("ARGS=("))
    assert args.endswith(" ".join(f"--given-up {name}" for name in given) + ")")
    capsys.readouterr()


def test_run_environment_departures() -> None:
    """The final review of gate B, I3: the fields a run records are compared (the GPU names as a
    set); a field it does not record departs when this environment records one; the container
    image only when the run recorded one."""
    here = {"hardware.gpus": ["NVIDIA L40S"], "hardware.driver": "580.95",
            "hardware.torch_cuda": "13.0", "python": "3.12.3",
            "packages": {"torch": "2.14.0", "numpy": "2.5.0"},
            "launch.container_image": None}  # fmt: skip
    meta = {"hardware": {"gpus": ["NVIDIA L40S", "NVIDIA L40S"], "driver": "580.95",
                         "torch_cuda": "13.0", "world_size": 1},
            "python": "3.12.3", "packages": {"torch": "2.14.0", "numpy": "2.5.0"},
            "launch": {"container_image": None}}  # fmt: skip
    assert fcd.environment_departures(meta, here) == []

    def departures(**changes):
        return fcd.environment_departures({**meta, **changes}, here)

    assert departures(hardware={**meta["hardware"], "gpus": ["NVIDIA A100"]}) == [
        ("hardware.gpus", ["NVIDIA A100"], ["NVIDIA L40S"])]  # fmt: skip
    assert departures(hardware={**meta["hardware"], "driver": "575.1"}) == [
        ("hardware.driver", "575.1", "580.95")]  # fmt: skip
    assert departures(python="3.12.4") == [("python", "3.12.4", "3.12.3")]
    assert departures(packages={"torch": "2.14.1", "numpy": "2.5.0"}) == [
        ("packages.torch", "2.14.1", "2.14.0")]  # fmt: skip
    assert departures(packages={"torch": "2.14.0"}) == [("packages.numpy", "not recorded",
                                                         "2.5.0")]  # fmt: skip
    old = {key: value for key, value in meta.items() if key != "hardware"}
    assert [d[0] for d in fcd.environment_departures(old, here)] == [
        "hardware.gpus", "hardware.driver", "hardware.torch_cuda"]  # fmt: skip
    cpu = {**here, "hardware.gpus": [], "hardware.driver": None, "hardware.torch_cuda": None}
    assert fcd.environment_departures(old, cpu) == []  # nothing recorded on either side
    image = "registry/zfr@sha256:" + "a" * 64
    assert departures(launch={"container_image": image}) == [
        ("launch.container_image", image, None)]  # fmt: skip
    assert fcd.environment_departures({**meta, "launch": {"container_image": image}},
                                      {**here, "launch.container_image": image}) == []  # fmt: skip
    assert departures(launch={"container_image": None}) == []  # none recorded: not compared
    check = fcd.environment_check([], here)
    assert not check["consistent"] and check["problems"] == ["no regular run to compare with"]


def test_records_follow_the_gpu_code(tmp_path, world) -> None:
    """Review I3, option (a), with code_identity on a repository of its own: records written at
    a commit X count at a later analysis-only commit Y, and are stale after a change of a GPU
    path (Z)."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip()  # fmt: skip

    def commit(files, message):
        for name, text in files.items():
            (repo / name).parent.mkdir(parents=True, exist_ok=True)
            (repo / name).write_text(text)
        git("add", "-A")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message)
        return git("rev-parse", "HEAD")

    git("init", "-q")
    x = commit({"src/zero_field_refusal/optim.py": "a\n", "docs/a.md": "a\n"}, "x")
    y = commit({"docs/a.md": "b\n", "src/zero_field_refusal/analysis/s.py": "b\n"}, "y")
    z = commit({"src/zero_field_refusal/optim.py": "c\n"}, "z")
    real = functools.partial(code_identity.reuse_check, repo=repo)
    found = cases()
    records = {c.name: {1: attempt(c, commit=x), 2: None} for c in found}
    for head, counted in ((x, True), (y, True), (z, False)):
        the_reader = fcd.RecordReader(scope(), {"commit": head, "dirty": False}, world)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(fcd, "reuse_check", real)
            out = fcd.qualification(records, found, reader=the_reader)
        assert (set(out["rates"].values()) == {"qualified"}) is counted, head
        assert the_reader.checks[x]["reusable"] is counted
        if not counted:
            assert the_reader.checks[x]["changed"] == ["src/zero_field_refusal/optim.py"]
            assert "whose GPU code is not HEAD's" in out["reasons"][found[0].key][0]
        assert the_reader.writers() == ([x] if counted else [])


def test_the_selection_record(tmp_path) -> None:
    """Review M4: ``rates`` is required, ``batch`` and ``U`` are the only other keys, anything
    else is refused with a clear message."""
    namespace = runpy.run_path(str(SCRIPT))
    parsed = namespace["parse_rates"](["adamw=1e-05,2e-05,3e-05", "adam_frozen=1e-05"])
    assert parsed == {"adamw": [1e-05, 2e-05, 3e-05], "adam_frozen": [1e-05]}
    with pytest.raises(fcd.NumericalError):
        namespace["parse_rates"](["adamw=1e-05", "adamw=2e-05"])
    record = tmp_path / "selection.json"
    arms = {"adamw_25": RATES["adamw"], "frozen_adam_25": RATES["adam_frozen"],
            "prewarmed_adamw_25": RATES["adamw_prewarmed"]}  # fmt: skip
    record.write_text(json.dumps({"batch": 20, "U": 8, "rates": arms}))
    assert namespace["read_selection"](record, 20, 8.0) == (RATES, [])
    # The final review of gate B, I1: the cases given up by the launcher, the one optional key
    assert namespace["SELECTION_KEYS"] == {"rates", "batch", "U", "given_up"}
    record.write_text(json.dumps({"batch": 20, "U": 8, "rates": arms,
                                  "given_up": ["s2200-adamw-1.32e-05"]}))  # fmt: skip
    assert namespace["read_selection"](record, 20, 8.0) == (RATES, ["s2200-adamw-1.32e-05"])
    for content, message in (({"batch": 20, "U": 8, **arms}, "needs 'rates'"),
                             ({"rates": arms, "next": {}}, "unexpected keys"),
                             ({"rates": arms, "batch": 5}, "batch 5"),
                             ({"rates": arms, "U": 7}, "U 7"),
                             ({"rates": {"adamw_25": 1e-05}}, "list of numbers"),
                             ({"rates": {"adamw_25": ["1e-05"]}}, "list of numbers"),
                             ({"rates": arms, "given_up": "s2200-adamw-1.32e-05"}, "given_up"),
                             ({"rates": arms, "given_up": ["a", "a"]}, "distinct case names"),
                             ([1, 2], "needs 'rates'")):  # fmt: skip
        record.write_text(json.dumps(content))
        with pytest.raises(fcd.NumericalError, match=message):
            namespace["read_selection"](record, 20, 8.0)
    record.write_text("{not json")
    with pytest.raises(fcd.NumericalError, match="not a readable JSON"):
        namespace["read_selection"](record, 20, 8.0)


def test_the_cost_model() -> None:
    """Review M1: 6.5 twin-equivalents per case; the plan prints the cost with and without an
    allowance for a full-float64 reference lost before an AdamW or pre-warmed fallback."""
    namespace = runpy.run_path(str(SCRIPT))
    s = scope()
    cost = namespace["costs"](20, s)
    twin = 25 * 0.17 * 20
    assert cost["case"] == pytest.approx((6.5 * twin + 5 * 6 * 30 + 6 * 5) / 60)
    assert cost["fallback"] == pytest.approx((1.5 * twin + 6 * 30) / 60)
    minutes, allowed = namespace["total_cost"](20, s, cases(), 3)
    assert minutes == pytest.approx(4 * cost["setup"] + 15 * cost["case"] + 3 * cost["control"])
    assert allowed == pytest.approx(minutes + 10 * cost["fallback"])  # 10 AdamW and pre-warmed
    assert 6.6 < minutes / 60 < 7.0 and 7.4 < allowed / 60 < 7.8


def test_the_check_refuses_other_gpu_code(tmp_path, clean_head, monkeypatch) -> None:
    """Review fa13: every regular run's recorded commit must have the check's GPU identity, from
    a clean checkout; otherwise neither part runs anything."""
    found = cases()
    root = tmp_path / "runs"
    fake_runs(root, found[:3])
    runs = sorted((root / "study2_short").iterdir())
    code = fcd.code_check(runs, COMMIT)
    assert code["consistent"] and code["gpu_code"] == GPU_CODE and not code["problems"]
    assert code["tool"] == fake_identity(COMMIT)
    assert {r["commit"] for r in code["runs"].values()} == {RUN_COMMIT}
    fcd.require_code(code)
    # Two commits of one GPU identity: the check's identity stands for both
    meta = runs[0] / "meta.json"
    meta.write_text(json.dumps({"git": {"commit": LATER, "dirty": False}}))
    assert fcd.code_check(runs, COMMIT)["gpu_code"] == fake_identity(COMMIT)
    for recorded, tool, problem in (
        ({"commit": RUN_COMMIT, "dirty": True}, COMMIT, "not a clean checkout"),
        ({"commit": None, "dirty": False}, COMMIT, "not a clean checkout"),
        ({"commit": "u" * 40, "dirty": False}, COMMIT, "CalledProcessError"),  # unknown commit
        ({"commit": RUN_COMMIT, "dirty": False}, None, "the check has no commit"),
        ({"commit": RUN_COMMIT, "dirty": False}, "u" * 40, "the check's GPU code"),
        ({"commit": OTHER, "dirty": False}, COMMIT, "whose GPU code differs"),
    ):  # fmt: skip
        meta.write_text(json.dumps({"git": recorded}))
        code = fcd.code_check(runs, tool)
        assert not code["consistent"] and code["gpu_code"] is None
        assert any(problem in p for p in code["problems"]), (recorded, tool, code["problems"])
        with pytest.raises(fcd.NumericalError, match="GPU code is not the check's"):
            fcd.require_code(code)
    assert fcd.code_check([], COMMIT)["problems"] == ["no regular run to compare with"]
    # Neither part writes anything against other GPU code
    s, out = settings(), tmp_path / "out"
    for refused in (None, code):
        with pytest.raises(fcd.NumericalError):
            fcd.run_numerical_twins(s, None, root, out, found[:3], attempt=1, code=refused)
        with pytest.raises(fcd.NumericalError):
            fcd.run_numerical_controls(s, None, root, out, rates=[9.98e-06], batch=20, upper=8,
                                       attempt=1, code=refused)  # fmt: skip
    assert not out.exists()


def test_the_code_check_on_this_checkout(tmp_path) -> None:
    """The real ``code_identity.gpu_code_identity``: a run recorded at HEAD from a clean checkout
    has HEAD's GPU code."""
    head = code_identity.gpu_code_identity("HEAD")
    root = tmp_path / "runs"
    fake_runs(root, cases()[:1], commit=head["commit"])
    code = fcd.code_check(sorted((root / "study2_short").iterdir()), head["commit"])
    assert code["consistent"] and code["gpu_code"] == head and head["files"] > 0


# ---------------------------------------------------------------- outcome blindness, the driver


def test_the_script_names_no_outcome_and_guards_every_harmful_prompt_set() -> None:
    namespace = runpy.run_path(str(SCRIPT))
    env = Environment(tokenizer=None, end_ids=[5, 6], pad_id=6, device=CPU, new_model=None,
                      prompt_set=lambda name: [name], data_root=Path("."))  # fmt: skip
    guarded = namespace["guard_prompt_sets"](env)
    assert guarded.prompt_set("alpaca_benign_kl") == ["alpaca_benign_kl"]
    for name in ("harmful_kl", "harmful_judged", "harmful_evaluation", "harmful_calibration"):
        with pytest.raises(RuntimeError, match="must not read"):
            guarded.prompt_set(name)
    tree = ast.parse(SCRIPT.read_text())
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
              for alias in node.names}  # fmt: skip
    for banned in ("harm", "refusal", "judge", "greedy", "Evaluator", "generate"):
        assert not [n for n in names if banned in n], banned


def test_the_driver(tmp_path, monkeypatch) -> None:
    commit = "a" * 40
    out = tmp_path / "driver.sh"
    argv = [str(SCRIPT), "driver", "--batch", "20", "--upper", "8", "--commit", commit]
    for arm, values in RATES.items():
        argv += ["--rates", f"{arm}={','.join(repr(r) for r in values)}"]
    assert call(monkeypatch, [*argv, "--out", str(out)]) == 0
    subprocess.run(["bash", "-n", str(out)], check=True)
    text = out.read_text()
    lines = text.splitlines()
    assert f"SRC=/workspace/code/zero-field-refusal/{commit}/source" in text
    assert 'OUT="${RUNS_DIR:?}/numerical/b20-u8"' in text
    assert all(len(line) <= 100 for line in text.splitlines() if line.startswith("#"))
    assert "harmful_" not in text and "instab" not in text and "--config" not in text
    args = next(line for line in lines if line.startswith("ARGS=("))
    assert args == ("ARGS=(--batch 20 --upper 8 --rates adamw=1.12e-05,1.32e-05,1.6e-05 "
                    "--rates adam_frozen=9.98e-06,1.21e-05,1.46e-05 "
                    "--rates adamw_prewarmed=1.1e-05,1.3e-05,1.5e-05)")  # fmt: skip
    run = 'project-run "$SRC" python scripts/numerical_check.py'
    order = [f'{run} plan "${{ARGS[@]}}"', f'{run} control "${{ARGS[@]}}"',
             f'{run} twins "${{ARGS[@]}}"', f'step {run} verdict "${{ARGS[@]}}"',
             f'{run} verdict "${{ARGS[@]}}" --retry-needed',
             f'    {run} control "${{ARGS[@]}}" --retry', f'    {run} twins "${{ARGS[@]}}" --retry',
             f'    step {run} verdict "${{ARGS[@]}}"']  # fmt: skip
    positions = [lines.index(line) for line in order]
    assert positions == sorted(positions)
    # The checkout, the log, a free GPU and the disk come first; every stop records the verdict
    head = lines.index(f'test "$(git rev-parse HEAD)" = {commit}')
    log = lines.index('exec > >(tee -a "$OUT/driver-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1')
    gpu = lines.index('if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; '
                      "then")  # fmt: skip
    disk = next(i for i, line in enumerate(lines) if line.startswith('if [ "$free_kb" -lt 40'))
    namespace = runpy.run_path(str(SCRIPT))
    # The final review of gate B: the free host memory (M7), then the checkpoints an interrupted
    # case left cleared (M8), before the disk check and the plan
    host = lines.index(f'if ! project-run "$SRC" python -c "{namespace["HOST_FREE"]}" 52.5; then')
    assert lines[host + 1].startswith('  stop "52.5 GB of free host memory are needed')
    cleared = lines.index('rm -rf -- "$OUT/twins/checkpoints"')
    assert [line for line in lines if line.startswith("rm ")] == [lines[cleared]]
    assert head < log < gpu < host < cleared < disk < positions[0]
    assert namespace["HOST_MEMORY_GB"] == 1.5 * (30 + 5)
    stop = next(line for line in lines if line.startswith("stop() {"))
    assert f'{run} verdict "${{ARGS[@]}}" || true; exit "$2"; }}' in stop
    exits = [line for line in lines if "exit" in line and not line.startswith("#")]
    assert [line for line in exits if "exit 1" in line or 'exit "$' in line] == [stop]
    for program in (namespace["READ_FIELD"], namespace["HAS_INCOMPLETE"], namespace["HOST_FREE"]):
        assert not {"$", "`", "\\", '"'} & set(program)
    # The host-memory program: exit 0 with enough free memory, 1 without
    for need, code in (("0", 0), ("1e9", 1)):
        done = subprocess.run([sys.executable, "-c", namespace["HOST_FREE"], need], check=False,
                              capture_output=True, text=True, cwd=ROOT)  # fmt: skip
        assert done.returncode == code and "free host memory (GB)" in done.stdout, done.stderr
    # The programs on a verdict: whether a rate or a control is incomplete
    verdict = tmp_path / "verdict.json"
    for rates, controls, code in (({"a|1": "not_qualified"}, {"1": "rejected"}, 1),
                                  ({"a|1": "incomplete"}, {"1": "accepted"}, 0),
                                  ({"a|1": "qualified"}, {"1": "incomplete"}, 0)):  # fmt: skip
        verdict.write_text(json.dumps({"qualified": False, "rates": rates, "controls": controls}))
        done = subprocess.run(["python3", "-c", namespace["HAS_INCOMPLETE"], str(verdict)],
                              check=False)  # fmt: skip
        assert done.returncode == code
    # The stop function, run: it records the verdict, then exits with its code
    stand_ins = ['project-run() { echo "verdict recorded"; }', "ARGS=(x)", "SRC=.", stop]
    done = subprocess.run(["bash", "-c", "\n".join(["set -euo pipefail", *stand_ins,
                                                    'stop "busy" 1', "echo after"])],
                          capture_output=True, text=True, check=False)  # fmt: skip
    assert done.returncode == 1 and "verdict recorded" in done.stdout and "after" not in done.stdout
    # --dry-run prints the driver and writes nothing
    assert call(monkeypatch, [*argv, "--out", str(tmp_path / "other.sh"), "--dry-run"]) == 0
    assert not (tmp_path / "other.sh").exists()


# ----------------------------------------------------------------- the whole check, tiny runs

STEPS = 6
BATCH = 4
UPPER = 8
TINY_SEEDS = [0, 1, 2]
TINY_RATES = {arm: [8e-4, 1e-3, 1.2e-3] for arm in ARMS}
OPTIMIZER = {"adamw": [], "adam_frozen": ["optimizer=adam_frozen"],
             "adamw_prewarmed": ["optimizer=adamw_prewarmed"]}  # fmt: skip


def tiny_settings(**changes):
    """The scope on tiny runs: three seeds of their own, 6 steps read at 1, 3 and 6, the smoke
    configs' protocol version; every other value frozen."""
    tiny = settings(numerical={"seeds": TINY_SEEDS, "eval_at": [1, 3, 6],
                               "regular": {"steps": STEPS, "protocol_version": 1}})  # fmt: skip
    return OmegaConf.merge(tiny, changes)


def tiny_cases():
    return fcd.numerical_cases(tiny_settings(), TINY_RATES, UPPER, batch=BATCH)


def run_config_of(base, case, device):
    """A tiny numerical run as the stage makes them (experiments/study2_short_k_only): the K-only
    evaluation (``eval.k_only``) and the update scalars on (``tracking.update_scalars``), so
    that the archive composes the K-only path, the scalars and twin A (the final review of gate
    B, M3)."""
    experiment = base / "numerical.yaml"
    experiment.write_text("defaults: [experiments/smoke, eval/study2, schedule/short]\n")
    overrides = [*SMALL, *APPENDABLE, "teacher.chunk_size=12", f"train.steps={STEPS}",
                 f"schedule.evaluate_at=[{STEPS}]", "+schedule.instability.grad_rule=log",
                 "experiment=study2_short", "+study2_short.role=numerical", "+eval.k_only=true",
                 "tracking.update_scalars=true", f"seed={case.seed}", *OPTIMIZER[case.optimizer],
                 f"optimizer.lr={case.lr}"]  # fmt: skip
    if case.optimizer != "adamw":
        overrides.append("optimizer.preconditioner.batches=2")
    if device.type != "cpu":
        overrides.append(f"execution.device={device.type}")
    return load_config(experiment, overrides)


def build_archive(base: Path, device: torch.device):
    """The tiny regular numerical runs of every case, on ``device``, archived as the stage's
    are, with their seeds' streams (made where the check re-examines them): K-only runs with
    the update scalars, as the stage's numerical runs are, recorded at ``RUN_COMMIT`` from a
    clean checkout. No harmful prompt set is read, so no harmful stream exists."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RUNS_DIR", str(base / "runs"))
        patch.setattr("zero_field_refusal.tracking.git_state",
                      lambda *args, **kwargs: {"commit": RUN_COMMIT, "dirty": False})  # fmt: skip
        tokenizer = tiny_tokenizer()
        end_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
        sets = prompt_sets()

        def new_model():
            model = tiny_model(tied=False, head_std=0.02).to(device)
            install_neutral_generation_config(model, end_ids, tokenizer.pad_token_id)
            return model

        env = Environment(tokenizer=tokenizer, end_ids=end_ids, pad_id=tokenizer.pad_token_id,
                          device=device, new_model=new_model, prompt_set=lambda name: sets[name],
                          data_root=base / "data")  # fmt: skip
        pass_path_check(env)
        found = tiny_cases()
        first = run_config_of(base, found[3], device)  # FA
        if device.type == "cuda":
            set_numerics(first.model)  # TF32 off, as in every run
        prepare(env, first, "kl_monitor")
        for seed in TINY_SEEDS:
            prepare(env, first, "preconditioner", seed=seed)
        guarded = fcd.guarded_prompt_sets(env, "harmful", "A K-only numerical run")
        for case in found:
            run_scheduled(run_config_of(base, case, device), guarded,
                          run_id=f"numerical-{case.name}")  # fmt: skip
    return env, base / "runs"


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    return build_archive(tmp_path_factory.mktemp("numerical"), CPU)


@pytest.fixture(scope="module")
def cuda_archive(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    return build_archive(tmp_path_factory.mktemp("numerical-cuda"), CUDA)


def copy_world(archive, tmp_path, monkeypatch):
    env, source = archive
    root = tmp_path / "runs"
    shutil.copytree(source, root)
    monkeypatch.setenv("RUNS_DIR", str(root))
    return env, root


def run_script(monkeypatch, config, root, part, *extra):
    """The script on the tiny world, with the tests' override of the tracked settings."""
    argv = [str(SCRIPT), part, "--batch", str(BATCH), "--upper", str(UPPER), "--config",
            str(config), "--runs-root", str(root)]  # fmt: skip
    for arm, values in TINY_RATES.items():
        argv += ["--rates", f"{arm}={','.join(repr(r) for r in values)}"]
    return call(monkeypatch, [*argv, *extra], untracked=True)


def tiny_reader(root, state=CLEAN):
    return fcd.RecordReader(fcd.NumericalScope.of(tiny_settings()), state, root)


def check_the_records(out, root, *, device, rule_gb=None):
    """What every check of the tiny world asserts of its records."""
    found = tiny_cases()
    norm = tiny_settings().numerical.perturbation.norm
    the_reader = tiny_reader(root)
    # The final review of gate B, M3: twin A reproduced K-only runs that recorded the scalars
    for run in sorted((root / "study2_short").iterdir()):
        meta = json.loads((run / "meta.json").read_text())
        assert meta["inputs"]["k_only"] is True and not (run / "responses").exists(), run.name
        assert "kl_harmful" not in meta["inputs"]["streams"], run.name
        assert len(fcd.read_jsonl(run / "update_scalars.jsonl")) == STEPS, run.name
    host = []
    for lr in TINY_RATES["adam_frozen"]:
        record = json.loads((out / fcd.control_path(lr, 1)).read_text())
        assert record["status"] == "complete", record.get("reason")
        assert record["state"] in ("below", "exceeding", "at_bound") and record["complete"]
        assert record["steps"] == STEPS and [r["step"] for r in record["scoring"]] == [1, 3, 6]
        assert all(check["equal"] for check in record["shared_state"].values())
        assert math.isfinite(record["first_gradient_norm"]) and record["seed"] == 0
        assert record["numerics"]["device"] == str(device) and record["attempt"] == 1
        assert the_reader.why_not(record, fcd.control_case(the_reader.scope, lr, BATCH,
                                                           UPPER)) is None  # fmt: skip
        if rule_gb is not None:
            assert record["memory_rule_gb"] == rule_gb and record["within_memory_rule"] is True
        host.append(record["provenance"]["host_memory"])
    facts = {}
    for case in found:
        record = json.loads((out / case.attempt_path(1)).read_text())
        assert record["status"] == "complete", (case.name, record.get("reason"))
        host.append(record["provenance"]["host_memory"])
        assert the_reader.why_not(record, case) is None, case.name
        assert record["twin_A"]["archive"]["equal"], case.name  # twin A is its run, bitwise
        assert set(record["pairs"]) == set(fcd.PAIRS)
        for key, p in record["pairs"].items():
            assert [r["step"] for r in p["D_twin"]] == [1, 3, 6], (case.name, key)
            assert p["complete"] and math.isfinite(p["max_D_twin"]) and p["state"] != "incomplete"
            if rule_gb is not None:
                assert p["within_memory_rule"] is True
        for key in fcd.PERTURBATIONS:
            first = record["pairs"][key]["rounding"][0]
            assert first["intended_norm"] == pytest.approx(norm, rel=1e-6), (case.name, key)
        assert record["reference_mode"] == record["pairs"]["reference"]["mode"]
        assert record["reference_mode"] == fcd.FLOAT64_FULL
        assert record["numerics"]["device"] == str(device)
        assert not list((out / "twins" / "checkpoints" / case.name).glob("*.pt"))
        facts.setdefault(case.seed, record["seed_facts"])
        assert record["seed_facts"] == facts[case.seed]  # one setup per seed, every arm and rate
    for f in facts.values():
        assert f["direction_norms"]["p1"] == f["direction_norms"]["p2"] > 0
        assert f["scales"]["p1"] * f["direction_norms"]["p1"] == pytest.approx(norm)
        assert f["rounding_heuristic"]["norm"] > 0 and f["residual"]["grad_norm"] > 0
    assert len({f["v0_sha256"] for f in facts.values()}) == 3  # each seed its v₀
    # The final review of gate B, M7: each record holds the host's peak resident memory and the
    # free host memory when it was written
    for entry in host:
        assert set(entry) == {"peak_rss_gb", "available_gb", "meminfo_gb", "cgroup_gb"}
        assert entry["peak_rss_gb"] > 0 and entry["meminfo_gb"] > 0
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict == json.loads(fcd.verdict_path(out, TINY_RATES).read_text())
    assert list(verdict) == VERDICT_KEYS
    assert verdict["commit"] == COMMIT and (verdict["batch"], verdict["U"]) == (BATCH, UPPER)
    assert verdict["policy"]["tool_commit"] == [COMMIT]
    assert set(verdict["rates"]) == {f"{a}|{r:g}" for a in ARMS for r in TINY_RATES[a]}
    assert set(verdict["controls"]) == {f"{r:g}" for r in TINY_RATES["adam_frozen"]}
    assert all(entry["attempts"] == 1 for entry in verdict["cases"])
    assert set(verdict["rates"].values()) <= STATES - {"incomplete", "not_qualified"}
    assert set(verdict["controls"].values()) <= {"accepted", "rejected"}
    assert verdict["qualified"] == (
        set(verdict["rates"].values()) == {"qualified"}
        and set(verdict["controls"].values()) == {"accepted"}
    )
    assert verdict["reference_fallback"] == []
    assert set(verdict["provenance"]["seeds"]) == {"0", "1", "2"}
    return verdict


def test_no_harm_paths(archive, tmp_path, monkeypatch, clean_head, capsys) -> None:
    """The plan, the controls, the twins and the verdict through the script on tiny K-only CPU
    runs, with every harmful prompt set, the evaluator, greedy generation and refusal scoring
    made to raise: they succeed, and no harmful set is asked for."""
    env, root = copy_world(archive, tmp_path, monkeypatch)
    guarded, requested = forbid_outcomes(monkeypatch, env)
    devices = []

    def environment(cfg, device):
        devices.append(device)
        return guarded

    monkeypatch.setattr("zero_field_refusal.runtime.environment", environment)
    config = tmp_path / "settings.yaml"
    OmegaConf.save(tiny_settings(), config)
    for part in ("plan", "control", "twins", "verdict"):
        assert run_script(monkeypatch, config, root, part, "--device", "cpu") == 0, part
    printed = capsys.readouterr().out
    assert "cost ≈" in printed and "MISSING" not in printed and "PROBLEM" not in printed
    out = fcd.numerical_root(BATCH, UPPER, root)
    check_the_records(out, root, device=CPU)
    assert requested and not [name for name in requested if "harmful" in name]
    # Run again, here and at a later analysis-only commit: everything is kept, nothing is due
    for state in (CLEAN, {"commit": LATER, "dirty": False}):
        monkeypatch.setattr(fcd, "git_state", lambda *a, state=state, **k: dict(state))
        for part in ("control", "twins"):
            assert run_script(monkeypatch, config, root, part, "--device", "cpu") == 0
            assert '"due": []' in capsys.readouterr().out
        assert run_script(monkeypatch, config, root, "verdict", "--retry-needed") == 3
        for part in ("control", "twins"):
            assert run_script(monkeypatch, config, root, part, "--retry") == 0
            assert '"due": []' in capsys.readouterr().out
    monkeypatch.setattr(fcd, "git_state", lambda *a, **k: dict(CLEAN))
    # The tracked settings are required: the tiny settings without the tests' override
    argv = [str(SCRIPT), "twins", "--batch", str(BATCH), "--upper", str(UPPER), "--config",
            str(config), "--runs-root", str(root)]  # fmt: skip
    for arm, values in TINY_RATES.items():
        argv += ["--rates", f"{arm}={','.join(repr(r) for r in values)}"]
    before = snapshot(out)
    shutil.rmtree(out / "twins" / "s2")
    assert "tracked settings" in str(call(monkeypatch, argv))
    assert call(monkeypatch, [*argv, "--dry-run"]) == 0  # only printing is allowed
    # Review fa13: GPU code unlike the regular runs' runs nothing; the plan fails on it
    shutil.rmtree(out / "controls")
    monkeypatch.setitem(GPU_SHA, RUN_COMMIT, "other")
    assert run_script(monkeypatch, config, root, "plan", "--device", "cpu") == 1
    assert "GPU CODE MISMATCH" in capsys.readouterr().out
    after = snapshot(out)
    for part in ("control", "twins"):
        assert run_script(monkeypatch, config, root, part, "--device", "cpu") == 7, part
    assert snapshot(out) == after and len(after) < len(before)
    monkeypatch.setitem(GPU_SHA, RUN_COMMIT, "same")
    # A checkout that is not clean runs and moves nothing
    monkeypatch.setattr(fcd, "git_state", lambda *args, **kwargs: {**CLEAN, "dirty": True})
    assert "not clean" in str(run_script(monkeypatch, config, root, "control"))
    assert snapshot(out) == after


def test_the_parts_record_an_instability_stop(archive, tmp_path, monkeypatch, clean_head,
                                              capsys) -> None:  # fmt: skip
    """Review N1, at the driver's level: an FA regular run that the ×10 gradient rule stopped
    lets the plan pass (with a note); ``control`` and ``twins`` record it; the verdict reads its
    rate failing (``instability:``) and its FA control rejected."""
    env, root = copy_world(archive, tmp_path, monkeypatch)
    guarded, _ = forbid_outcomes(monkeypatch, env)
    monkeypatch.setattr("zero_field_refusal.runtime.environment", lambda cfg, device: guarded)
    scope_ = fcd.NumericalScope.of(tiny_settings())
    case = next(c for c in tiny_cases() if (c.seed, c.optimizer, c.lr) == (0, "adam_frozen",
                                                                          1.2e-3))  # fmt: skip
    path = fcd.locate_case(root, scope_, case) / "terminal.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "steps": 4, "stop": {
        "reason": "gradient_spike", "kind": "instability",
        "detail": {"grad_norm": 9.0, "reference_median": 0.5}}}))  # fmt: skip
    config = tmp_path / "settings.yaml"
    OmegaConf.save(tiny_settings(), config)
    assert run_script(monkeypatch, config, root, "plan", "--device", "cpu") == 0
    printed = capsys.readouterr().out
    assert "NOTE: instability: the parts will record the rate as failing" in printed
    assert "PROBLEM" not in printed and "MISSING" not in printed
    for part in ("control", "twins", "verdict"):
        assert run_script(monkeypatch, config, root, part, "--device", "cpu") == 0, part
    out = fcd.numerical_root(BATCH, UPPER, root)
    assert json.loads((out / case.attempt_path(1)).read_text())["status"] == "instability"
    assert json.loads((out / fcd.control_path(1.2e-3, 1)).read_text())["status"] == "instability"
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["rates"]["adam_frozen|0.0012"] == "failing" and verdict["qualified"] is False
    assert verdict["controls"]["0.0012"] == "rejected"
    reasons = verdict["reasons"]["adam_frozen|0.0012"]
    assert reasons[0].startswith("instability: case s0: the regular run stopped by "
                                 "'gradient_spike' after 4 steps")  # fmt: skip
    assert any(r.startswith("control: instability: ") for r in reasons)
    assert verdict["provenance"]["retry_due"] == {"cases": [], "controls": []}
    others = {k: v for k, v in verdict["rates"].items() if k != "adam_frozen|0.0012"}
    assert "incomplete" not in others.values() and "not_qualified" not in others.values()
    capsys.readouterr()


def test_a_twin_a_unlike_its_archive_is_incomplete_twice(
    archive, tmp_path, monkeypatch, clean_head, capsys
) -> None:
    """A doctored archive: twin A differs, no pair is run, the case is incomplete; its rerun in
    a fresh process (``twins --retry``) differs again, and the rate is not_qualified. The
    record names the doctored run, as located when it ran."""
    env, root = copy_world(archive, tmp_path, monkeypatch)
    s = tiny_settings()
    found = tiny_cases()
    case = next(c for c in found if c.seed == 1 and c.optimizer == "adamw_prewarmed")
    run_dir = fcd.locate_case(root, fcd.NumericalScope.of(s), case)
    metrics = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    next(r for r in rows if "loss" in r)["grad_norm"] *= 1 + 1e-12
    metrics.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = fcd.numerical_root(BATCH, UPPER, root)
    done = fcd.run_numerical_twins(s, env, root, out, [case], attempt=1, setups=found,
                                   code=consistent_code())  # fmt: skip
    first = done[case.name]
    assert first["status"] == "incomplete" and first["pairs"] == {}
    assert first["reason"] == "twin A differs from its archived run"
    assert first["twin_A"]["archive"]["mismatches"][0]["key"] == "grad_norm"
    assert first["regular_run_identity"] == fcd.run_identity(run_dir)
    config = tmp_path / "settings.yaml"
    OmegaConf.save(s, config)
    monkeypatch.setattr("zero_field_refusal.runtime.environment", lambda cfg, device: env)
    assert run_script(monkeypatch, config, root, "verdict", "--retry-needed") == 0
    capsys.readouterr()
    assert run_script(monkeypatch, config, root, "twins", "--retry", "--device", "cpu") == 0
    assert json.loads(capsys.readouterr().out.splitlines()[0])["due"] == [case.name]
    rerun = json.loads((out / case.attempt_path(2)).read_text())
    assert rerun["attempt"] == 2 and rerun["status"] == "incomplete"
    assert run_script(monkeypatch, config, root, "verdict") == 0
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["rates"][case.key] == "not_qualified" and verdict["qualified"] is False
    assert any(r.startswith("case s1: not qualified (technical), incomplete after its rerun")
               for r in verdict["reasons"][case.key])  # fmt: skip
    entry = next(e for e in verdict["cases"] if (e["seed"], e["arm"]) == (1, "adamw_prewarmed"))
    assert (entry["attempts"], entry["attempt_states"]) == (2, ["incomplete", "incomplete"])
    # The cases without a first attempt are incomplete, their rerun not due
    assert verdict["rates"]["adamw|0.0008"] == "incomplete"
    assert verdict["provenance"]["retry_due"] == {"cases": [], "controls": []}


def test_the_reference_falls_back_in_a_case(archive, tmp_path, monkeypatch, clean_head) -> None:
    """Review M2: the full-float64 reference of one case meets the memory cap
    (``torch.OutOfMemoryError``); its fallback is recorded in the attempt and listed by the
    verdict."""
    env, root = copy_world(archive, tmp_path, monkeypatch)
    s = tiny_settings()
    found = tiny_cases()
    case = next(c for c in found if c.seed == 2 and c.optimizer == "adamw")
    run_twin = fcd.run_twin

    def capped(cfg, twins, *, precision="fp32", **kwargs):
        if precision == "float64" and cfg.optimizer.name == "adamw" and cfg.seed == 2:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 5.20 GiB")
        return run_twin(cfg, twins, precision=precision, **kwargs)

    monkeypatch.setattr(fcd, "run_twin", capped)
    out = fcd.numerical_root(BATCH, UPPER, root)
    done = fcd.run_numerical_twins(s, env, root, out, [case], attempt=1, setups=found,
                                   code=consistent_code())  # fmt: skip
    record = done[case.name]
    assert record["status"] == "complete"
    assert record["reference_mode"] == fcd.FLOAT64_ACCUMULATION
    assert record["fallback_reason"] == ("OutOfMemoryError: CUDA out of memory. Tried to "
                                         "allocate 5.20 GiB")  # fmt: skip
    reference = record["pairs"]["reference"]
    assert reference["failed_attempt"]["precision"] == "float64"
    assert reference["state"] != "incomplete"
    verdict = fcd.numerical_verdict(s, out, rates=TINY_RATES, batch=BATCH, upper=UPPER,
                                    code=consistent_code(), root=root)  # fmt: skip
    assert verdict["reference_fallback"] == [{
        "case": case.name, "rate": case.key, "seed": 2,
        "fallback_reason": record["fallback_reason"]}]  # fmt: skip
    entry = next(e for e in verdict["cases"] if (e["seed"], e["arm"]) == (2, "adamw"))
    assert entry["reference_mode"] == fcd.FLOAT64_ACCUMULATION and entry["state"] != "incomplete"


def test_a_missing_or_unstable_run_is_recorded(archive, tmp_path, monkeypatch,
                                               clean_head) -> None:  # fmt: skip
    """A case without its regular run is recorded incomplete with why, and a seed without its FA
    run records every case of the seed. A regular run stopped by an instability rule is
    recorded as such (review M6), its rate failing and its FA control rejected; one stopped
    otherwise is technical. A control without its run is recorded incomplete."""
    env, root = copy_world(archive, tmp_path, monkeypatch)
    s = tiny_settings()
    scope_ = fcd.NumericalScope.of(s)
    found = tiny_cases()
    out = fcd.numerical_root(BATCH, UPPER, root)
    seed2 = [c for c in found if c.seed == 2]
    fa2 = next(c for c in seed2 if c.optimizer == "adam_frozen")
    shutil.rmtree(fcd.locate_case(root, scope_, fa2))
    done = fcd.run_numerical_twins(s, env, root, out, seed2, attempt=1, code=consistent_code())
    assert [done[c.name]["status"] for c in seed2] == ["incomplete"] * 3
    assert "not found" in done[fa2.name]["reason"]
    adamw2 = next(c for c in seed2 if c.optimizer == "adamw")
    assert done[adamw2.name]["reason"].startswith("no adam_frozen regular run of seed 2")
    seed1 = [c for c in found if c.seed == 1]
    adamw1, fa1 = (next(c for c in seed1 if c.optimizer == arm) for arm in ("adamw", "adam_frozen"))
    unstable = {
        "stop": {
            "reason": "gradient_spike",
            "kind": "instability",
            "detail": {"grad_norm": 9.0, "reference_median": 0.5},
        },
        "steps": 4,
    }
    interrupted = {"stop": {"reason": "t_max", "kind": "limit"}, "steps": 4}
    prewarmed1 = next(c for c in seed1 if c.optimizer == "adamw_prewarmed")
    for case, terminal in ((adamw1, unstable), (fa1, interrupted), (prewarmed1, unstable)):
        path = fcd.locate_case(root, scope_, case) / "terminal.json"
        path.write_text(json.dumps({**json.loads(path.read_text()), **terminal}))
    # Review N4: a run that departs otherwise (its protocol version) is technical, even when an
    # instability rule stopped it
    config = fcd.locate_case(root, scope_, prewarmed1) / "config.yaml"
    config.write_text(config.read_text().replace("protocol_version: 1", "protocol_version: 2"))
    done = fcd.run_numerical_twins(s, env, root, out, seed1, attempt=1, code=consistent_code())
    assert done[adamw1.name]["status"] == "instability" and done[adamw1.name]["pairs"] == {}
    assert done[adamw1.name]["reason"].startswith("instability: the regular run stopped by "
                                                  "'gradient_spike' after 4 steps")  # fmt: skip
    assert done[fa1.name]["status"] == "incomplete"
    assert "stopped by 't_max' after 4 steps" in done[fa1.name]["reason"]
    assert done[prewarmed1.name]["status"] == "incomplete"
    assert "protocol_version is 2, not 1" in done[prewarmed1.name]["reason"]
    assert "instability" not in done[prewarmed1.name]["reason"]
    the_reader = tiny_reader(root)
    out_q = fcd.qualification({c.name: {1: done[c.name], 2: None} for c in seed1}, seed1,
                              reader=the_reader)  # fmt: skip
    assert out_q["rates"][adamw1.key] == "failing"
    assert out_q["reasons"][adamw1.key][0].startswith("instability: case s1: the regular run")
    assert out_q["rates"][fa1.key] == "incomplete"  # technical: its rerun is due
    with pytest.raises(fcd.NumericalError):
        fcd.run_numerical_twins(s, env, root, out, seed1, attempt=3, code=consistent_code())
    # The FA control of a run stopped by instability is rejected, without running
    fa0 = next(c for c in found if c.seed == 0 and c.optimizer == "adam_frozen")
    path = fcd.locate_case(root, scope_, fa0) / "terminal.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **unstable}))
    controls = fcd.run_numerical_controls(s, env, root, out, rates=[fa0.lr], batch=BATCH,
                                          upper=UPPER, attempt=1,
                                          code=consistent_code())  # fmt: skip
    (record,) = controls.values()
    assert record["status"] == "instability" and "scoring" not in record
    verdict = fcd.control_verdict({1: record}, rate=fa0.lr, batch=BATCH, upper=UPPER,
                                  reader=tiny_reader(root))  # fmt: skip
    assert verdict["verdict"] == "rejected" and verdict["reasons"][0].startswith("instability:")
    shutil.rmtree(fcd.locate_case(root, scope_, fa0))
    controls = fcd.run_numerical_controls(s, env, root, out, rates=[fa0.lr], batch=BATCH,
                                          upper=UPPER, attempt=1,
                                          code=consistent_code())  # fmt: skip
    (record,) = controls.values()
    assert record["status"] == "incomplete" and record["state"] == "incomplete"
    assert "0 finished runs" in record["reason"] and "traceback" in record


def driver_parts(monkeypatch, config, root, args, capsys) -> dict[str, int]:
    """The parts in the order and with the exit handling of the Pod's driver
    (``driver_script``), each through the script with the driver's ``ARGS`` and the tests'
    override of the tracked settings: the plan must pass; the controls and the twins' first
    attempts; the verdict; one rerun of everything incomplete when ``--retry-needed`` says one is
    due; the verdict again. Returns each step's exit code; the plan's print is under ``plan``."""

    def part(*extra):
        argv = [str(SCRIPT), *extra, "--config", str(config), "--runs-root", str(root), *args]
        return call(monkeypatch, argv, untracked=True)

    codes = {"plan": part("plan")}
    codes["plan_text"] = capsys.readouterr().out
    assert codes["plan"] == 0, codes["plan_text"]
    codes["control"] = part("control", "--device", "cpu")
    codes["twins"] = part("twins", "--device", "cpu")
    assert part("verdict") == 0
    codes["due"] = part("verdict", "--retry-needed")
    if codes["due"] == 0:
        codes["rerun_control"] = part("control", "--retry", "--device", "cpu")
        codes["rerun_twins"] = part("twins", "--retry", "--device", "cpu")
        assert part("verdict") == 0
    codes["after"] = part("verdict", "--retry-needed")
    capsys.readouterr()
    return codes


def test_a_given_up_run_lets_the_driver_finish(archive, tmp_path, monkeypatch, clean_head,
                                               capsys) -> None:  # fmt: skip
    """The final review of gate B, I1, end to end: a K-only run the launcher gave up on, named
    in the selection record (``given_up``), which the driver passes to every part. The plan
    notes it, the parts record its case incomplete twice, and the verdict holds its rate
    not_qualified (technical) with nothing left incomplete: the driver ends."""
    env, root = copy_world(archive, tmp_path, monkeypatch)
    guarded, requested = forbid_outcomes(monkeypatch, env)
    monkeypatch.setattr("zero_field_refusal.runtime.environment", lambda cfg, device: guarded)
    s = tiny_settings()
    case = next(c for c in tiny_cases() if (c.seed, c.optimizer, c.lr) == (0, "adamw", 1e-3))
    shutil.rmtree(fcd.locate_case(root, fcd.NumericalScope.of(s), case))
    config = tmp_path / "settings.yaml"
    OmegaConf.save(s, config)
    namespace = runpy.run_path(str(SCRIPT))
    text = namespace["driver_script"]("a" * 40, "now", BATCH, UPPER, TINY_RATES, [case.name])
    line = next(line for line in text.splitlines() if line.startswith("ARGS=("))
    args = shlex.split(line.removeprefix("ARGS=(").removesuffix(")"))
    assert args[-2:] == ["--given-up", case.name]
    codes = driver_parts(monkeypatch, config, root, args, capsys)
    assert f"{case.name}: no regular run; NOTE: given up by the launcher" in codes["plan_text"]
    assert "PROBLEM" not in codes["plan_text"] and "MISSING" not in codes["plan_text"]
    assert {k: v for k, v in codes.items() if k != "plan_text"} == {
        "plan": 0, "control": 0, "twins": 0, "due": 0, "rerun_control": 0, "rerun_twins": 0,
        "after": 3}  # fmt: skip
    out = fcd.numerical_root(BATCH, UPPER, root)
    for n in fcd.ATTEMPTS:
        record = json.loads((out / case.attempt_path(n)).read_text())
        assert record["status"] == "incomplete" and "traceback" not in record
        assert record["reason"].startswith("the regular run is not found")
    verdict = json.loads((out / "verdict.json").read_text())
    assert verdict["rates"][case.key] == "not_qualified" and verdict["qualified"] is False
    assert verdict["reasons"][case.key][0].startswith(
        "case s0: not qualified (technical), incomplete after its rerun: attempt 1: the regular "
        "run is not found")  # fmt: skip
    others = {k: v for k, v in verdict["rates"].items() if k != case.key}
    assert set(others.values()) <= {"qualified", "failing"}
    assert "incomplete" not in verdict["controls"].values()
    # The driver's last check: nothing incomplete, so it ends without exit 5
    done = subprocess.run([sys.executable, "-c", namespace["HAS_INCOMPLETE"],
                           str(out / "verdict.json")], check=False)  # fmt: skip
    assert done.returncode == 1
    assert requested and not [name for name in requested if "harmful" in name]


def test_another_run_environment_is_refused(archive, tmp_path, monkeypatch, clean_head,
                                            capsys) -> None:  # fmt: skip
    """The final review of gate B, I3: a regular run whose meta.json records another driver,
    another package version, or a container image this process does not run in: the plan
    exits 1 naming the field, and ``control`` and ``twins`` exit 7 recording nothing."""
    env, root = copy_world(archive, tmp_path, monkeypatch)
    guarded, _ = forbid_outcomes(monkeypatch, env)
    monkeypatch.setattr("zero_field_refusal.runtime.environment", lambda cfg, device: guarded)
    monkeypatch.delenv("CONTAINER_IMAGE", raising=False)
    s = tiny_settings()
    config = tmp_path / "settings.yaml"
    OmegaConf.save(s, config)
    case = tiny_cases()[4]
    meta_path = fcd.locate_case(root, fcd.NumericalScope.of(s), case) / "meta.json"
    original = json.loads(meta_path.read_text())
    out = fcd.numerical_root(BATCH, UPPER, root)
    image = "registry.example/zfr@sha256:" + "b" * 64
    for change, field in (
        ({"hardware": {**original["hardware"], "driver": "0.0-other"}}, "hardware.driver"),
        ({"packages": {**original["packages"], "torch": "0.0"}}, "packages.torch"),
        ({"launch": {**original["launch"], "container_image": image}}, "launch.container_image"),
    ):  # fmt: skip
        meta_path.write_text(json.dumps({**original, **change}))
        assert run_script(monkeypatch, config, root, "plan", "--no-stream-check") == 1
        printed = capsys.readouterr().out
        assert f"RUN ENVIRONMENT MISMATCH: {field}: " in printed, printed
        assert f"recorded by 1 regular run(s) ({meta_path.parent.name})" in printed
        for part in ("control", "twins"):
            assert run_script(monkeypatch, config, root, part, "--device", "cpu") == 7, part
            assert "RUN ENVIRONMENT MISMATCH" in capsys.readouterr().err
        assert not out.exists()  # nothing recorded
    # The image the run recorded is this process's: no mismatch
    monkeypatch.setenv("CONTAINER_IMAGE", image)
    assert run_script(monkeypatch, config, root, "plan", "--no-stream-check") == 0
    assert "run environment: the regular runs' equals this one" in capsys.readouterr().out


def test_void_moves_what_another_gpu_code_made(world, monkeypatch, capsys) -> None:
    """The final review of gate B, I2: after a GPU-path change (HEAD at ``OTHER``), ``void``
    moves, never deletes, every numerical run at B whose recorded commit's GPU code is not
    HEAD's and the check's directory to RUNS_DIR/void/<old GPU code>/, with an inventory of
    every file and its sha256 and a record; a run made at HEAD's GPU code stays. With them go
    the launcher's log of the campaign that made them, its attempts' logs and the campaign's
    file under campaigns/ with its driver (the coordinator's ruling on concern C2). It refuses
    on a dirty checkout, with nothing to void, before moving a run of another role, and when
    the campaign's log also names a run it leaves in place."""
    s = scope()
    out = fcd.numerical_root(20, 8, world)
    write_records(out, within_all(cases()), {lr: {1: control(lr), 2: None}
                                             for lr in RATES["adam_frozen"]})  # fmt: skip
    other_b = fcd.NumericalCase(2200, "adamw", 1.32e-05, 5, 8)
    fake_runs(world / "b5", [other_b])  # at another batch: not touched
    shutil.move(world / "b5" / "study2_short" / other_b.name, world / "study2_short" / "b5-run")
    # The launcher's record of the campaign that made the first three runs, and its file
    campaign = "study2-short-numerical-b20-u8-test"

    def start(run_id: str) -> str:
        return json.dumps({"event": "start", "run": "cell/s0", "attempt": 1, "run_id": run_id,
                           "experiment": "study2_short", "config_hash": "h"}) + "\n"  # fmt: skip

    def journal(*names):
        (world / "launch").mkdir(exist_ok=True)
        text = "".join(start(name) for name in names) + "{cut short\n"
        (world / "launch" / f"{campaign}.jsonl").write_text(text)

    journal(*(c.name for c in cases()[:3]))
    (world / "launch" / campaign).mkdir()
    (world / "launch" / campaign / "attempt.log").write_text("output\n")
    (world / "launch" / "other.jsonl").write_text(start("x"))
    (world / "campaigns").mkdir()
    (world / "campaigns" / "numerical.yaml").write_text(f"# header\nname: {campaign}\ncells: []\n")
    (world / "campaigns" / "numerical-driver.sh").write_text("#!/usr/bin/env bash\n")
    (world / "campaigns" / "discovery.yaml").write_text("name: study2-short-discovery\n")
    (world / "campaigns" / "broken.yaml").write_text("name: [\n")

    def void(*extra):
        argv = [str(SCRIPT), "void", "--batch", "20", "--upper", "8", "--runs-root", str(world),
                *extra]  # fmt: skip
        return call(monkeypatch, argv)

    assert "nothing to void" in str(void("--dry-run"))  # HEAD at COMMIT: the runs' GPU code
    kept = fcd.NumericalCase(2200, "adamw", 1.4e-05, 20, 8)
    fake_runs(world, [kept], commit=OTHER)  # a run made after the change
    monkeypatch.setattr(fcd, "git_state", lambda *a, **k: {"commit": OTHER, "dirty": True})
    assert "the checkout is not clean" in str(void())
    monkeypatch.setattr(fcd, "git_state", lambda *a, **k: {"commit": OTHER, "dirty": False})
    discovery = fcd.locate_case(world, s, cases()[0]) / "config.yaml"
    text = discovery.read_text()
    discovery.write_text(text.replace("role: numerical", "role: discovery"))
    assert "has the role 'discovery', not 'numerical'" in str(void())
    discovery.write_text(text)
    assert "takes --batch and --upper only" in str(void("--rates", "adamw=1e-05"))
    journal(*(c.name for c in cases()[:3]), kept.name)  # a log naming a run left in place
    assert (f"launch/{campaign}.jsonl names runs that void moves and "
            f"['study2_short/{kept.name}'], which it does not") in str(void())  # fmt: skip
    journal(*(c.name for c in cases()[:3]))
    before = snapshot(world)
    capsys.readouterr()
    assert void("--dry-run") == 0
    shown = json.loads(capsys.readouterr().out)
    assert snapshot(world) == before  # a dry run moves nothing
    moved = sorted({f"study2_short/{c.name}" for c in cases() + cases(RESOLVED)})
    assert shown["paths"] == [*moved, "numerical/b20-u8", f"launch/{campaign}.jsonl",
                              f"launch/{campaign}", "campaigns/numerical.yaml",
                              "campaigns/numerical-driver.sh"]  # fmt: skip
    assert shown["kept"] == [f"study2_short/{kept.name}"]
    assert shown["destination"] == str(world / "void" / "same")  # the old GPU code, sha256[:12]
    assert void() == 0
    capsys.readouterr()
    destination = world / "void" / "same"
    (inventory,) = destination.glob("inventory-*.tsv")
    rows = [line.split("\t") for line in inventory.read_text().splitlines()[1:]]
    left = {"launch/other.jsonl", "campaigns/discovery.yaml", "campaigns/broken.yaml"}
    assert {row[0] for row in rows} == set(before) - left - {
        f"study2_short/{kept.name}/{name}" for name in ("config.yaml", "terminal.json",
                                                        "metrics.jsonl", "meta.json")} - {
        f"study2_short/b5-run/{name}" for name in ("config.yaml", "terminal.json",
                                                   "metrics.jsonl", "meta.json")}  # fmt: skip
    assert f"launch/{campaign}/attempt.log" in {row[0] for row in rows}
    assert all((world / path).is_file() for path in left)
    for path, size, digest in rows:
        assert (destination / path).stat().st_size == int(size)
        assert hashlib.sha256((destination / path).read_bytes()).hexdigest() == digest
        assert not (world / path).exists()
    (record,) = destination.glob("void-*.json")
    found = json.loads(record.read_text())
    assert found["moved"] == shown["paths"] and found["old_gpu_code"]["sha256"] == "same"
    assert found["head_gpu_code"]["sha256"] == "other" and found["inventory"] == inventory.name
    assert (world / "study2_short" / kept.name).is_dir()
    assert (world / "study2_short" / "b5-run").is_dir() and not out.exists()
    assert "nothing to void" in str(void())  # done once: nothing left to move


@pytest.mark.parametrize("role", ["validation", "confirmation", "timecourse"])
def test_void_refuses_once_s3_has_started(world, monkeypatch, capsys, role) -> None:
    """The coordinator's ruling on concern C1: one run of the role validation, confirmation or
    timecourse at B, whatever its GPU code, refuses the void, and nothing moves: a GPU-path
    change after S3 has started goes to the coordinator (a budget decision)."""
    monkeypatch.setattr(fcd, "git_state", lambda *a, **k: {"commit": OTHER, "dirty": False})
    started = fcd.NumericalCase(2100, "adamw", 1.32e-05, 20, 8)
    fake_runs(world, [started], commit=OTHER)
    config = world / "study2_short" / started.name / "config.yaml"
    config.write_text(config.read_text().replace("role: numerical", f"role: {role}"))
    before = snapshot(world)
    for extra in ((), ("--dry-run",)):
        argv = [str(SCRIPT), "void", "--batch", "20", "--upper", "8", "--runs-root", str(world),
                *extra]  # fmt: skip
        refused = call(monkeypatch, argv)
        assert refused not in (0, None) and "S3 has started" in str(refused)
        assert (f"1 run(s) of the role validation, confirmation or timecourse at B 20 "
                f"(study2_short/{started.name})") in str(refused)  # fmt: skip
        assert "goes to the coordinator" in str(refused) and "budget decision" in str(refused)
    assert snapshot(world) == before and not (world / "void").exists()
    capsys.readouterr()


# ---------------------------------------------------------------------------------- on a GPU


@pytest.fixture
def cuda_rule():
    """Lifts the memory rule after a test: its CUDA fraction holds for the whole process."""
    yield
    torch.cuda.set_per_process_memory_fraction(1.0, fcd.cuda_index(CUDA))
    fcd._MEMORY_RULE.clear()


@needs_cuda
def test_cuda_pipeline(cuda_archive, cuda_rule, tmp_path, monkeypatch, clean_head,
                       capsys) -> None:  # fmt: skip
    """The Pod's path on a GPU: control, twins and verdict through the script on
    ``torch.device("cuda")`` without an index, under a 1 GB memory rule. The tiny K-only runs
    were archived on this GPU, so the replays and every twin A compare with their records bit
    for bit."""
    env, root = copy_world(cuda_archive, tmp_path, monkeypatch)
    guarded, requested = forbid_outcomes(monkeypatch, env)
    devices = []

    def environment(cfg, device):  # runtime.environment's device, as the script passes it
        devices.append(device)
        return dataclasses.replace(guarded, device=torch.device(device))

    monkeypatch.setattr("zero_field_refusal.runtime.environment", environment)
    config = tmp_path / "settings.yaml"
    OmegaConf.save(tiny_settings(numerical={"memory_rule_gb": 1}), config)
    for part in ("control", "twins", "verdict"):
        assert run_script(monkeypatch, config, root, part) == 0, part
    assert devices == ["cuda", "cuda"]
    index = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    rules = [json.loads(line)["memory_rule"] for line in capsys.readouterr().out.splitlines()
             if line.startswith('{"memory_rule"')]  # fmt: skip
    assert rules == [{"limit_gb": 1.0, "fraction": 1e9 / total, "total_gb": round(total / 1e9, 3),
                      "device_index": index}] * 2  # fmt: skip
    out = fcd.numerical_root(BATCH, UPPER, root)
    check_the_records(out, root, device="cuda", rule_gb=1)
    assert requested and not [name for name in requested if "harmful" in name]
