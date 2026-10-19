"""The numerical check of study 2's selected rates (amendment 4, section 1, with review fa12): its
frozen scope and replacement cases, the fixed perturbation norm, the float64 reference against a
float64 oracle, the states of cases, rates and controls, and the verdict record with its policy;
then the whole check on tiny K-only CPU runs with harmful generation forbidden, a twin A unlike
its archive incomplete twice, and (skipped without a GPU) the check through the script on an
unindexed CUDA device under a 1 GB memory rule."""

import ast
import dataclasses
import json
import math
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from conftest import tiny_model, tiny_tokenizer
from omegaconf import OmegaConf
from test_fa_control_diagnostic import SURVIVED, forbid_outcomes
from test_train import APPENDABLE, SMALL, pass_path_check, prepare, prompt_sets

from zero_field_refusal import fa_control_diagnostic as fcd
from zero_field_refusal.config import load_config
from zero_field_refusal.models import freeze, install_neutral_generation_config, set_numerics
from zero_field_refusal.optim import Adam
from zero_field_refusal.runtime import Environment, locate_stream
from zero_field_refusal.schedule import run_scheduled

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "numerical_check.py"
CPU = torch.device("cpu")
CUDA = torch.device("cuda")  # unindexed, as runtime.environment gives it on a Pod
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
COMMIT = "c" * 40
CLEAN = {"commit": COMMIT, "dirty": False}
BOUND = 0.0632455532033676  # 0.01·√(5·8)
EVAL_AT = [1, 5, 10, 15, 20, 25]
ARMS = ("adamw", "adam_frozen", "adamw_prewarmed")
STATES = {"qualified", "failing", "not_qualified", "incomplete"}
CONTROL_STATES = {"accepted", "rejected", "not_qualified", "incomplete"}
# Illustrative selections, as a selection record would give them: never in the code
RATES = {"adamw": [1.32e-05, 1.12e-05, 1.6e-05], "adam_frozen": [9.98e-06, 1.21e-05, 1.46e-05],
         "adamw_prewarmed": [1.1e-05, 1.3e-05, 1.5e-05]}  # fmt: skip


def settings(**changes):
    return OmegaConf.merge(load_config("experiments/numerical_check"), changes)


def scope():
    return fcd.NumericalScope.of(settings())


def cases(rates=RATES, upper=8, batch=20):
    return fcd.numerical_cases(settings(), rates, upper, batch=batch)


RUN_COMMIT = "r" * 40  # the commit the regular runs recorded
GPU_SHA = {COMMIT: "same", RUN_COMMIT: "same"}  # the GPU identities of the known commits


def fake_identity(commit="HEAD", repo=None):
    """``code_identity.gpu_code_identity`` over ``GPU_SHA``; an unknown commit fails as git
    would."""
    if commit not in GPU_SHA:
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", commit])
    return {"commit": commit, "files": 7, "sha256": GPU_SHA[commit]}


GPU_CODE = fake_identity(RUN_COMMIT)


@pytest.fixture
def clean_head(monkeypatch):
    """HEAD at ``COMMIT`` from a clean checkout, for every record and verdict the test makes,
    with the GPU identities of ``GPU_SHA`` (the regular runs' and HEAD's equal)."""
    monkeypatch.setattr(fcd, "git_state", lambda *args, **kwargs: dict(CLEAN))
    monkeypatch.setattr(fcd, "gpu_code_identity", fake_identity)


def consistent_code():
    return {"consistent": True, "gpu_code": GPU_CODE, "problems": []}


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
    assert s.policy(8, GPU_CODE, COMMIT) == {
        "gpu_code": {"commit": RUN_COMMIT, "files": 7, "sha256": "same"}, "tool_commit": COMMIT,
        "norm": 5e-05, "survival_share": 0.5, "retry_scale": 10, "bound": BOUND, "steps": EVAL_AT,
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
        {k: RATES[k] for k in ARMS[:2]},
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
    resolved = {**RATES, "adamw": [1.12e-05, 1.32e-05, 1.7e-05]}
    assert set(top) <= set(cases(resolved))
    assert set(low) <= set(cases({**RATES, "adam_frozen": [1.05e-05, 1.21e-05, 1.46e-05]}))
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
    5e-5, p1 points along FA's first update −P·g, p2 opposite."""
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
    update = [-(r.double() / (v.double().sqrt() + eps)) for r, v in zip(residual, twins.v0,
                                                                       strict=True)]  # fmt: skip
    flat = torch.cat([u.reshape(-1) for u in update])
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


def stamped(record, commit=COMMIT, dirty=False):
    return {**record, "provenance": {"code": {"commit": commit, "dirty": dirty}}}


def pair(last=0.001, *, values=None, steps=EVAL_AT, rounding=SURVIVED, stop=None,
         diverged=False):  # fmt: skip
    """A pair's record as ``run_pairs`` writes it: D_twin 0.001 at every step but the last."""
    values = [0.001] * (len(steps) - 1) + [last] if values is None else values
    return {"D_twin": [{"step": s, "D_twin": v} for s, v in zip(steps, values, strict=True)],
            "rows": [], "stop": stop, "diverged": diverged, "rounding": rounding,
            "max_D_twin": fcd._largest(values)}  # fmt: skip


def attempt(case, n=1, *, pairs=None, status="complete", equal=True, reason=None,
            mode=fcd.FLOAT64_FULL, **stamp):  # fmt: skip
    """A case's attempt as ``run_numerical_twins`` writes it; ``pairs`` maps a pair to its last
    D_twin or its record, or None to leave it out (every other pair within)."""
    built = {}
    for key in fcd.PAIRS:
        spec = (pairs or {}).get(key, 0.001)
        if spec is None:
            continue
        record = pair(spec) if isinstance(spec, int | float) else dict(spec)
        if key == fcd.REFERENCE:
            record.pop("rounding", None)
            record["mode"] = mode
            if mode == fcd.FLOAT64_ACCUMULATION:
                record["fallback_reason"] = "OutOfMemoryError: CUDA out of memory"
        built[key] = record
    body = {"part": "twins", "case": case.name, "seed": case.seed, "optimizer": case.optimizer,
            "lr": case.lr, "batch": case.batch, "U": case.upper, "attempt": n,
            "status": status, "reason": reason, "twin_A": {"archive": {"equal": equal}},
            "pairs": built}  # fmt: skip
    return stamped(body, **stamp)


def control(lr, last=1e-12, *, steps=EVAL_AT, status="complete", diverged=False, stop=None,
            reason=None, n=1, **stamp):  # fmt: skip
    """A same-path control's attempt as ``run_numerical_controls`` writes it."""
    values = [1e-12] * (len(steps) - 1) + [last] if steps else []
    return stamped({"part": "control", "rate": lr, "seed": 2200, "batch": 20, "U": 8,
                    "attempt": n, "status": status, "reason": reason,
                    "scoring": [{"step": s, "K_scoring": v} for s, v in zip(steps, values,
                                                                           strict=True)],
                    "monitor": [], "rows": [{"step": 0, "loss": 1e-12, "grad_norm": 2e-14}],
                    "stop": stop, "diverged": diverged, "first_gradient_norm": 2e-14},
                   **stamp)  # fmt: skip


def within_all(found):
    return {c.name: {1: attempt(c), 2: None} for c in found}


def qualify(records, found, state=CLEAN):
    return fcd.qualification(records, found, scope=scope(), state=state)


def by_name(found, seed, arm, lr):
    return next(c for c in found if (c.seed, c.optimizer, c.lr) == (seed, arm, lr))


def test_failing_pair_makes_rate_failing() -> None:
    found = cases()
    upper_fa = by_name(found, 2201, "adam_frozen", 1.46e-05)
    key = "adam_frozen|1.46e-05"
    out = qualify(within_all(found), found)
    assert set(out["rates"].values()) == {"qualified"} and out["reasons"] == {}
    assert out["seeds"][key] == [2200, 2201, 2202] and out["seeds"]["adam_frozen|1.21e-05"] == [
        2200
    ]
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
            entry = next(e for e in out["cases"] if (e["seed"], e["arm"]) == (2201, "adam_frozen"))
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
             "survived": False}] * 2  # fmt: skip
    records = within_all(found)
    records[upper_fa.name][1] = attempt(upper_fa, pairs={"p1": pair(rounding=lost)})
    assert qualify(records, found)["rates"][key] == "incomplete"
    records[upper_fa.name][1] = attempt(upper_fa, pairs={"p1": pair(0.2, rounding=lost)})
    assert qualify(records, found)["rates"][key] == "failing"


def test_incomplete_twice_is_failing() -> None:
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
        assert fcd.retry_due(records, found, scope=scope(), state=CLEAN) == [case]
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
        assert not fcd.retry_due(records, found, scope=scope(), state=CLEAN)
        records[case.name][2] = attempt(case, 2)  # a complete rerun within the bound
        assert qualify(records, found)["rates"][key] == "qualified"
    # No first attempt, or one not current, is incomplete and not a rerun's due
    for first in (
        None,
        attempt(case, commit="d" * 40),
        attempt(case, dirty=True),
        attempt(by_name(found, 2200, "adamw", 1.12e-05)),
    ):  # another case's record
        records = within_all(found)
        records[case.name] = {1: first, 2: None}
        assert qualify(records, found)["rates"][key] == "incomplete"
        assert not fcd.retry_due(records, found, scope=scope(), state=CLEAN)
    # A verdict from a checkout that is not clean reads nothing
    out = qualify(within_all(found), found, state={**CLEAN, "dirty": True})
    assert set(out["rates"].values()) == {"incomplete"}


def test_the_reference_fallback_is_recorded_per_case() -> None:
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


def control_of(first, rerun=None, **changes):
    return fcd.control_verdict({1: first, 2: rerun}, rate=1.21e-05, batch=20, upper=8,
                               scope=scope(), state={**CLEAN, **changes})  # fmt: skip


def test_same_path_control_accept_and_reject() -> None:
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
    for name, record in {"no record": None, "another commit": control(1.21e-05, commit="d" * 40),
                         "dirty": control(1.21e-05, dirty=True),
                         "another rate": control(1.46e-05)}.items():  # fmt: skip
        assert control_of(record)["verdict"] == "incomplete", name
        assert control_of(record, control(1.21e-05, n=2))["verdict"] == "incomplete", name
    assert control_of(control(1.21e-05, 0.01), dirty=True)["verdict"] == "incomplete"
    records = {1.21e-05: {1: control(1.21e-05), 2: None},
               1.46e-05: {1: control(1.46e-05, steps=[1]), 2: None}}  # fmt: skip
    assert fcd.control_verdicts(records, rates=[1.21e-05, 1.46e-05, 9.98e-06], batch=20, upper=8,
                                scope=scope(), state=CLEAN) == {
        1.21e-05: "accepted", 1.46e-05: "incomplete", 9.98e-06: "incomplete"}  # fmt: skip
    assert fcd.control_retry_due(records, [1.21e-05, 1.46e-05, 9.98e-06], batch=20, upper=8,
                                 scope=scope(), state=CLEAN) == [1.46e-05]  # fmt: skip


def write_records(out, records, controls):
    by = {c.name: c for c in cases()}
    for name, attempts in records.items():
        for n, record in attempts.items():
            if record is not None:
                fcd.write_json(out / by[name].attempt_path(n), record)
    for lr, attempts in controls.items():
        for n, record in attempts.items():
            if record is not None:
                fcd.write_json(out / fcd.control_path(lr, n), record)


VERDICT_KEYS = [
    "batch",
    "U",
    "bound",
    "commit",
    "policy",
    "rates",
    "seeds",
    "controls",
    "qualified",
    "reasons",
    "cases",
    "reference_fallback",
    "provenance",
]


def fake_runs(root, found, commit=RUN_COMMIT):
    """A finished, conforming regular run's records for each case, made at ``commit``."""
    for case in found:
        run = root / "study2_short" / case.name
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
        (run / "terminal.json").write_text(json.dumps({"stop": {"reason": "steps"}, "steps": 25}))
        (run / "meta.json").write_text(json.dumps({"git": {"commit": commit, "dirty": False}}))


def test_verdict_schema(tmp_path, clean_head, monkeypatch, capsys) -> None:
    """The record the selection reads, at RUNS_DIR/numerical/b{B}-u{U}/verdict.json, through the
    script; ``qualified`` iff every rate is qualified and every control accepted; the policy
    with the regular runs' GPU code (review fa13)."""
    found = cases()
    root = tmp_path / "runs"
    fake_runs(root, found)
    out = fcd.numerical_root(20, 8, root)
    records = within_all(found)
    controls = {lr: {1: control(lr), 2: None} for lr in RATES["adam_frozen"]}
    write_records(out, records, controls)
    argv = ["--batch", "20", "--upper", "8", "--runs-root", str(root)]
    for arm, values in RATES.items():
        argv += ["--rates", f"{arm}={','.join(repr(r) for r in values)}"]

    def script(*extra):
        monkeypatch.setattr("sys.argv", [str(SCRIPT), "verdict", *argv, *extra])
        try:
            runpy.run_path(str(SCRIPT), run_name="__main__")
        except SystemExit as exc:
            return exc.code
        return 0

    assert script() == 0
    verdict = json.loads((root / "numerical" / "b20-u8" / "verdict.json").read_text())
    assert list(verdict) == VERDICT_KEYS
    assert (verdict["batch"], verdict["U"], verdict["bound"]) == (20, 8, BOUND)
    assert isinstance(verdict["batch"], int) and isinstance(verdict["U"], int)
    assert verdict["commit"] == COMMIT and fcd.output_commit(verdict) == COMMIT
    assert verdict["policy"] == scope().policy(8, GPU_CODE, COMMIT)
    assert verdict["policy"]["gpu_code"] == {"commit": RUN_COMMIT, "files": 7, "sha256": "same"}
    assert verdict["policy"]["tool_commit"] == COMMIT
    assert verdict["provenance"]["code_check"]["consistent"] is True
    assert verdict["rates"] == {f"{a}|{r:g}": "qualified" for a in ARMS for r in sorted(RATES[a])}
    assert "adam_frozen|9.98e-06" in verdict["rates"] and "adamw|1.6e-05" in verdict["rates"]
    assert verdict["seeds"]["adamw|1.6e-05"] == [2200, 2201, 2202]
    assert verdict["seeds"]["adamw|1.12e-05"] == [2200] and set(verdict["seeds"]) == set(
        verdict["rates"]
    )
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
    # Records of another commit are stale: nothing qualifies
    v = outcome({c.name: {1: attempt(c, commit="d" * 40), 2: None} for c in found})
    assert v["qualified"] is False and set(v["rates"].values()) == {"incomplete"}
    # A verdict of another commit is set aside when a new one is written
    old = stamped({"qualified": True}, commit="e" * 40)
    fcd.write_json(out / "verdict.json", old)
    assert script() == 0
    assert json.loads((out / "stale" / ("e" * 40) / "verdict.json").read_text()) == old
    # --dry-run and --retry-needed write nothing
    (out / "verdict.json").unlink()
    assert script("--dry-run") == 0 and script("--retry-needed") == 3
    assert not (out / "verdict.json").exists()
    # Review fa13: the regular runs' GPU code unlike the check's, no record counts
    monkeypatch.setitem(GPU_SHA, RUN_COMMIT, "other")
    v = outcome()
    assert v["qualified"] is False and set(v["rates"].values()) == {"incomplete"}
    assert set(v["controls"].values()) == {"incomplete"} and v["policy"]["gpu_code"] is None
    assert v["policy"]["tool_commit"] == COMMIT
    for key in v["rates"]:
        assert v["reasons"][key][0].startswith("code: the regular runs' GPU code is not the "
                                               "check's")  # fmt: skip
    assert v["provenance"]["code_check"]["consistent"] is False
    capsys.readouterr()


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
    monkeypatch.setitem(GPU_SHA, "s" * 40, "same")
    meta = runs[0] / "meta.json"
    meta.write_text(json.dumps({"git": {"commit": "s" * 40, "dirty": False}}))
    assert fcd.code_check(runs, COMMIT)["gpu_code"] == fake_identity(COMMIT)
    for recorded, tool, problem in (
        ({"commit": RUN_COMMIT, "dirty": True}, COMMIT, "not a clean checkout"),
        ({"commit": None, "dirty": False}, COMMIT, "not a clean checkout"),
        ({"commit": "u" * 40, "dirty": False}, COMMIT, "CalledProcessError"),  # unknown commit
        ({"commit": RUN_COMMIT, "dirty": False}, None, "the check has no commit"),
        ({"commit": RUN_COMMIT, "dirty": False}, "u" * 40, "the check's GPU code"),
    ):  # fmt: skip
        meta.write_text(json.dumps({"git": recorded}))
        code = fcd.code_check(runs, tool)
        assert not code["consistent"] and code["gpu_code"] is None
        assert any(problem in p for p in code["problems"]), (recorded, tool, code["problems"])
        with pytest.raises(fcd.NumericalError, match="GPU code is not the check's"):
            fcd.require_code(code)
    meta.write_text(json.dumps({"git": {"commit": RUN_COMMIT, "dirty": False}}))
    monkeypatch.setitem(GPU_SHA, RUN_COMMIT, "other")
    code = fcd.code_check(runs, COMMIT)
    assert not code["consistent"] and "whose GPU code differs" in code["problems"][0]
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
    from zero_field_refusal.code_identity import gpu_code_identity

    head = gpu_code_identity("HEAD")
    root = tmp_path / "runs"
    fake_runs(root, cases()[:1], commit=head["commit"])
    code = fcd.code_check(sorted((root / "study2_short").iterdir()), head["commit"])
    assert code["consistent"] and code["gpu_code"] == head and head["files"] > 0


def test_the_rates_come_from_the_command_line_or_a_selection_record(tmp_path) -> None:
    namespace = runpy.run_path(str(SCRIPT))
    parsed = namespace["parse_rates"](["adamw=1e-05,2e-05,3e-05", "adam_frozen=1e-05"])
    assert parsed == {"adamw": [1e-05, 2e-05, 3e-05], "adam_frozen": [1e-05]}
    with pytest.raises(fcd.NumericalError):
        namespace["parse_rates"](["adamw=1e-05", "adamw=2e-05"])
    record = tmp_path / "selection.json"
    record.write_text(json.dumps({"batch": 20, "U": 8, "rates": {
        "adamw_25": RATES["adamw"], "frozen_adam_25": RATES["adam_frozen"],
        "prewarmed_adamw_25": RATES["adamw_prewarmed"]}}))  # fmt: skip
    assert namespace["read_selection"](record, 20, 8.0) == RATES
    with pytest.raises(fcd.NumericalError, match="batch"):
        namespace["read_selection"](record, 5, 8.0)


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
    monkeypatch.setattr("sys.argv", [*argv, "--out", str(out)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    subprocess.run(["bash", "-n", str(out)], check=True)
    text = out.read_text()
    lines = text.splitlines()
    assert f"SRC=/workspace/code/zero-field-refusal/{commit}/source" in text
    assert 'OUT="${RUNS_DIR:?}/numerical/b20-u8"' in text
    assert all(len(line) <= 100 for line in text.splitlines() if line.startswith("#"))
    assert "harmful_" not in text and "instab" not in text
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
    assert head < log < gpu < disk < positions[0]
    stop = next(line for line in lines if line.startswith("stop() {"))
    assert f'{run} verdict "${{ARGS[@]}}" || true; exit "$2"; }}' in stop
    exits = [line for line in lines if "exit" in line and not line.startswith("#")]
    assert [line for line in exits if "exit 1" in line or 'exit "$' in line] == [stop]
    namespace = runpy.run_path(str(SCRIPT))
    for program in (namespace["READ_FIELD"], namespace["HAS_INCOMPLETE"]):
        assert not {"$", "`", "\\", '"'} & set(program)
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
    monkeypatch.setattr("sys.argv", [*argv, "--out", str(tmp_path / "other.sh"), "--dry-run"])
    runpy.run_path(str(SCRIPT), run_name="__main__")
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
    experiment = base / "numerical.yaml"
    experiment.write_text("defaults: [experiments/smoke, eval/study2, schedule/short]\n")
    overrides = [*SMALL, *APPENDABLE, "teacher.chunk_size=12", f"train.steps={STEPS}",
                 f"schedule.evaluate_at=[{STEPS}]", "+schedule.instability.grad_rule=log",
                 "experiment=study2_short", "+study2_short.role=numerical", f"seed={case.seed}",
                 *OPTIMIZER[case.optimizer], f"optimizer.lr={case.lr}"]  # fmt: skip
    if case.optimizer != "adamw":
        overrides.append("optimizer.preconditioner.batches=2")
    if device.type != "cpu":
        overrides.append(f"execution.device={device.type}")
    return load_config(experiment, overrides)


def k_only(env, runs: Path, base: Path):
    """The archive as K-only numerical runs leave it (review fa12): no response to a harmful
    prompt, no harmful K stream, no harm-related summary, input or measurement. The data are
    copied first; returns the environment on the copy."""
    data = base / "data-k-only"
    shutil.copytree(env.data_root, data)
    cfg = run_config_of(base, tiny_cases()[3], env.device)
    _, stream, _ = locate_stream(cfg, env, "kl_harmful", None)
    shutil.rmtree(data / stream.relative_to(env.data_root))
    for run in sorted(runs.glob("study2_short/*")):
        shutil.rmtree(run / "responses", ignore_errors=True)
        summary = json.loads((run / "summary.json").read_text())
        (run / "summary.json").write_text(
            json.dumps({k: v for k, v in summary.items() if "harm" not in k and "refusal" not in k})
        )
        meta = json.loads((run / "meta.json").read_text())
        streams = meta["inputs"]["streams"]
        streams.pop("kl_harmful", None)
        meta["inputs"].pop("harmful_prompts", None)
        meta["git"] = {"commit": RUN_COMMIT, "dirty": False}  # as a clean checkout records it
        (run / "meta.json").write_text(json.dumps(meta))
        measurements = run / "observations" / "measurements.jsonl"
        if measurements.exists():
            rows = [json.loads(line) for line in measurements.read_text().splitlines()]
            measurements.write_text("".join(json.dumps(r) + "\n" for r in rows
                                            if r.get("measure") != "kl_harmful"))  # fmt: skip
    return dataclasses.replace(env, data_root=data)


def build_archive(base: Path, device: torch.device):
    """The tiny regular numerical runs of every case, on ``device``, archived as the stage's
    are, with their seeds' streams (made where the check re-examines them), then made K-only."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RUNS_DIR", str(base / "runs"))
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
        for case in found:
            run_scheduled(run_config_of(base, case, device), env, run_id=f"numerical-{case.name}")
    return k_only(env, base / "runs", base), base / "runs"


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
    argv = [str(SCRIPT), part, "--batch", str(BATCH), "--upper", str(UPPER), "--config",
            str(config), "--runs-root", str(root)]  # fmt: skip
    for arm, values in TINY_RATES.items():
        argv += ["--rates", f"{arm}={','.join(repr(r) for r in values)}"]
    monkeypatch.setattr("sys.argv", [*argv, *extra])
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as exc:
        return exc.code
    return 0


def check_the_records(out, *, device, rule_gb=None):
    """What every check of the tiny world asserts of its records."""
    found = tiny_cases()
    norm = tiny_settings().numerical.perturbation.norm
    for lr in TINY_RATES["adam_frozen"]:
        record = json.loads((out / fcd.control_path(lr, 1)).read_text())
        assert record["status"] == "complete", record.get("reason")
        assert record["state"] in ("below", "exceeding", "at_bound") and record["complete"]
        assert record["steps"] == STEPS and [r["step"] for r in record["scoring"]] == [1, 3, 6]
        assert all(check["equal"] for check in record["shared_state"].values())
        assert math.isfinite(record["first_gradient_norm"]) and record["seed"] == 0
        assert record["numerics"]["device"] == str(device) and record["attempt"] == 1
        if rule_gb is not None:
            assert record["memory_rule_gb"] == rule_gb and record["within_memory_rule"] is True
    facts = {}
    for case in found:
        record = json.loads((out / case.attempt_path(1)).read_text())
        assert record["status"] == "complete", (case.name, record.get("reason"))
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
        assert (
            record["reference_mode"] == record["pairs"]["reference"]["mode"] == (fcd.FLOAT64_FULL)
        )
        assert record["numerics"]["device"] == str(device)
        assert not list((out / "twins" / "checkpoints" / case.name).glob("*.pt"))
        facts.setdefault(case.seed, record["seed_facts"])
        assert record["seed_facts"] == facts[case.seed]  # one setup per seed, every arm and rate
    for f in facts.values():
        assert f["direction_norms"]["p1"] == f["direction_norms"]["p2"] > 0
        assert f["scales"]["p1"] * f["direction_norms"]["p1"] == pytest.approx(norm)
        assert f["rounding_heuristic"]["norm"] > 0 and f["residual"]["grad_norm"] > 0
    assert len({f["v0_sha256"] for f in facts.values()}) == 3  # each seed its v₀
    verdict = json.loads((out / "verdict.json").read_text())
    assert list(verdict) == VERDICT_KEYS
    assert verdict["commit"] == COMMIT and (verdict["batch"], verdict["U"]) == (BATCH, UPPER)
    assert set(verdict["rates"]) == {f"{a}|{r:g}" for a in ARMS for r in TINY_RATES[a]}
    assert set(verdict["controls"]) == {f"{r:g}" for r in TINY_RATES["adam_frozen"]}
    assert all(entry["attempts"] == 1 for entry in verdict["cases"])
    assert set(verdict["rates"].values()) <= STATES - {"incomplete", "not_qualified"}
    assert set(verdict["controls"].values()) <= {"accepted", "rejected"}
    assert verdict["qualified"] == (
        set(verdict["rates"].values()) == {"qualified"}
        and set(verdict["controls"].values()) == {"accepted"}
    )
    assert verdict["reference_fallback"] == [] and set(verdict["provenance"]["seeds"]) == {
        "0",
        "1",
        "2",
    }
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
    check_the_records(out, device=CPU)
    assert requested and not [name for name in requested if "harmful" in name]
    # Run again: everything is kept, and no rerun is due
    for part in ("control", "twins"):
        assert run_script(monkeypatch, config, root, part, "--device", "cpu") == 0
        assert '"due": []' in capsys.readouterr().out
    assert run_script(monkeypatch, config, root, "verdict", "--retry-needed") == 3
    for part in ("control", "twins"):
        assert run_script(monkeypatch, config, root, part, "--retry") == 0
        assert '"due": []' in capsys.readouterr().out
    # Review fa13: GPU code unlike the regular runs' runs nothing; the plan fails on it
    shutil.rmtree(out / "controls")
    shutil.rmtree(out / "twins")
    monkeypatch.setitem(GPU_SHA, RUN_COMMIT, "other")
    assert run_script(monkeypatch, config, root, "plan", "--device", "cpu") == 1
    assert "GPU CODE MISMATCH" in capsys.readouterr().out
    for part in ("control", "twins"):
        assert run_script(monkeypatch, config, root, part, "--device", "cpu") == 7, part
    assert not (out / "controls").exists() and not (out / "twins").exists()
    monkeypatch.setitem(GPU_SHA, RUN_COMMIT, "same")
    # A checkout that is not clean runs nothing
    monkeypatch.setattr(fcd, "git_state", lambda *args, **kwargs: {**CLEAN, "dirty": True})
    assert "not clean" in str(run_script(monkeypatch, config, root, "control"))


def test_a_twin_a_unlike_its_archive_is_incomplete_twice(
    archive, tmp_path, monkeypatch, clean_head, capsys
) -> None:
    """A doctored archive: twin A differs, no pair is run, the case is incomplete; its rerun in
    a fresh process (``twins --retry``) differs again, and the rate is not_qualified."""
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
    done = fcd.run_numerical_twins(
        s, env, root, out, [case], attempt=1, setups=found, code=consistent_code()
    )
    first = done[case.name]
    assert first["status"] == "incomplete" and first["pairs"] == {}
    assert first["reason"] == "twin A differs from its archived run"
    assert first["twin_A"]["archive"]["mismatches"][0]["key"] == "grad_norm"
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


def test_a_missing_or_nonconforming_run_is_recorded(archive, tmp_path, monkeypatch,
                                                    clean_head) -> None:  # fmt: skip
    """A case without its regular run, or with one that stopped early, is recorded incomplete
    with why; a seed without its FA run records every case of the seed; a control without its
    run is recorded incomplete."""
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
    adamw1 = next(c for c in found if c.seed == 1 and c.optimizer == "adamw")
    terminal = fcd.locate_case(root, scope_, adamw1) / "terminal.json"
    record = json.loads(terminal.read_text())
    terminal.write_text(json.dumps({**record, "stop": {"reason": "gradient_spike"}, "steps": 4}))
    seed1 = [c for c in found if c.seed == 1]
    done = fcd.run_numerical_twins(s, env, root, out, seed1, attempt=1, code=consistent_code())
    assert done[adamw1.name]["status"] == "incomplete"
    assert "stopped by 'gradient_spike' after 4 steps" in done[adamw1.name]["reason"]
    assert all(done[c.name]["status"] == "complete" for c in seed1 if c != adamw1)
    with pytest.raises(fcd.NumericalError):
        fcd.run_numerical_twins(s, env, root, out, seed1, attempt=3, code=consistent_code())
    fa0 = next(c for c in found if c.seed == 0 and c.optimizer == "adam_frozen")
    shutil.rmtree(fcd.locate_case(root, scope_, fa0))
    controls = fcd.run_numerical_controls(s, env, root, out, rates=[fa0.lr], batch=BATCH,
                                          upper=UPPER, attempt=1,
                                          code=consistent_code())  # fmt: skip
    (record,) = controls.values()
    assert record["status"] == "incomplete" and record["state"] == "incomplete"
    assert "0 finished runs" in record["reason"] and "traceback" in record


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
    check_the_records(out, device="cuda", rule_gb=1)
    assert requested and not [name for name in requested if "harmful" in name]
