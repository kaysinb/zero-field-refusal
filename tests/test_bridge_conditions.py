"""The bridge's process-independence conditions on the tiny model, and θ₀'s utility check
(docs/study3-preregistration.md, "Numerical qualification", "The bridge's conditions" and
"Between-step work").

Between-step work (the monitor, the diagnostics, the generation switch ``eval.k_only``, an
evaluation) changes no parameter, gradient, buffer, optimizer state, train/eval mode or RNG state
that a later step uses; and the harness's baseline of a launcher run agrees with it. Rows are
compared on their scientific fields only (not ``time_s``). Bitwise agreement on the CPU is a test
of the code here, not an acceptance gate across processes on the L40S: there the registered
tolerances decide.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_endpoint_checkpoint import final_models, rng_state, same_rng, scientific
from test_qualification import FA, SCOPE, part, spiked
from test_qualification import fa_parts as fa_parts  # noqa: F401  (the fixtures)
from test_qualification import harness_code_environment as harness_code_environment  # noqa: F401
from test_qualification import tiny_confirmation as tiny_confirmation  # noqa: F401
from test_qualification import world as world  # noqa: F401
from test_schedule import arm_config, built_optimizers, read_jsonl
from test_schedule import env as env  # noqa: F401  (the fixture)

from zero_field_refusal import fa_control_diagnostic as fcd
from zero_field_refusal import qualification as q
from zero_field_refusal import schedule, train
from zero_field_refusal.schedule import run_scheduled

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_theta0_utility.py"
ROW_FIELDS = ("tokens", "denominator", "loss", "grad_norm")


def trained(env, tmp_path, monkeypatch, name, *overrides) -> dict:
    """A fixed 25-step run and what a later step would use of it: rows, parameters, gradients,
    buffers, optimizer state and RNG state."""
    built, models = built_optimizers(monkeypatch), final_models(monkeypatch)
    cfg = arm_config(env, tmp_path, "adamw", *overrides)
    run_dir = run_scheduled(cfg, env, run_id=f"bridge-{name}")
    optimizer = built[-1]
    params = [p for g in optimizer.param_groups for p in g["params"]]
    return {
        "monitor": [m["step"] for m in read_jsonl(run_dir / "monitor.jsonl")],
        "metrics": scientific([r for r in read_jsonl(run_dir / "metrics.jsonl") if "loss" in r]),
        "evaluation": read_jsonl(run_dir / "evaluations.jsonl"),
        "params": [p.detach().clone() for p in params],
        "grads": [p.grad for p in params],
        "buffers": {n: b.clone() for n, b in models[-1].named_buffers()},
        "training": models[-1].training,
        "state": [optimizer.state[p] for p in params],
        "rng": rng_state(),
    }  # fmt: skip


def assert_same_training(a: dict, b: dict) -> None:
    assert a["metrics"] == b["metrics"]  # exact, field by field
    assert all(torch.equal(x, y) for x, y in zip(a["params"], b["params"], strict=True))
    assert [g is None for g in a["grads"]] == [g is None for g in b["grads"]]
    assert all(torch.equal(x, y) for x, y in zip(a["grads"], b["grads"], strict=True)
               if x is not None)  # fmt: skip
    assert a["buffers"].keys() == b["buffers"].keys()
    assert all(torch.equal(a["buffers"][n], b["buffers"][n]) for n in a["buffers"])
    assert a["training"] == b["training"]
    for x, y in zip(a["state"], b["state"], strict=True):
        assert x.keys() == y.keys()
        assert all(torch.equal(x[k], y[k]) if torch.is_tensor(x[k]) else x[k] == y[k] for k in x)
    assert same_rng(a["rng"], b["rng"])


def test_the_monitor_cadence_changes_no_later_step(env, tmp_path, monkeypatch) -> None:
    every_5 = trained(env, tmp_path, monkeypatch, "every-5", "eval.monitor.every=5")
    at_25 = trained(env, tmp_path, monkeypatch, "at-25", "eval.monitor.every=25")
    assert every_5["monitor"] == [5, 10, 15, 20, 25] and at_25["monitor"] == [25]
    assert len(every_5["metrics"]) == 25
    assert_same_training(every_5, at_25)


def test_the_generation_switch_changes_no_later_step(env, tmp_path, monkeypatch) -> None:
    full = trained(env, tmp_path, monkeypatch, "full", "+eval.k_only=false")
    k_only = trained(env, tmp_path, monkeypatch, "k-only", "+eval.k_only=true")
    assert_same_training(full, k_only)
    (a,), (b,) = full["evaluation"], k_only["evaluation"]  # what both measure is equal
    shared = {"K_benign", "heldout_nll_per_token", "factual_accuracy", "benchmark_accuracy"}
    assert shared <= a.keys() and "K_harmful" in a and "K_harmful" not in b
    assert {k: a[k] for k in shared} == {k: b[k] for k in shared}


def test_an_evaluation_changes_no_state_a_following_step_uses(env, tmp_path, monkeypatch) -> None:
    """Around the loop's between-step work and the evaluator's call alike: the parameters, the
    buffers, the train/eval mode and the RNG state are as they were."""
    seen = []

    def watch(kind, function):
        def wrapped(self, model, *args, **kwargs):
            def state():  # copies: the run goes on to change the live tensors
                return (model.training, [p.detach().clone() for p in model.parameters()],
                        {n: b.clone() for n, b in model.named_buffers()}, rng_state())  # fmt: skip

            before = state()
            out = function(self, model, *args, **kwargs)
            seen.append((kind, before, state()))
            return out

        return wrapped

    monkeypatch.setattr(train.Evaluator, "__call__", watch("evaluation", train.Evaluator.__call__))
    monkeypatch.setattr(schedule._Loop, "after_step", watch("between", schedule._Loop.after_step))
    cfg = arm_config(env, tmp_path, "adamw", "+eval.k_only=false")
    run_scheduled(cfg, env, run_id="isolated")
    assert {kind for kind, *_ in seen} == {"evaluation", "between"}
    assert sum(kind == "evaluation" for kind, *_ in seen) >= 1
    assert sum(kind == "between" for kind, *_ in seen) == 25  # every step
    for kind, before, after in seen:
        assert before[0] == after[0], kind  # model.training
        assert all(torch.equal(a, b) for a, b in zip(before[1], after[1], strict=True)), kind
        assert before[2].keys() == after[2].keys()
        assert all(torch.equal(before[2][n], after[2][n]) for n in before[2]), kind
        assert same_rng(before[3], after[3]), kind


def test_the_harness_baseline_agrees_with_the_launcher_run(world, fa_parts) -> None:  # noqa: F811
    _, runs = world
    _, _, baseline, _ = fa_parts
    assert baseline["comparisons"]["launcher~baseline"]["D_twin"] == 0.0
    run = baseline["trajectories"]["baseline"]
    assert run["bitwise_rows"] is True  # a diagnostic flag, not a gate
    launcher = [r for r in read_jsonl(runs["fa"] / "metrics.jsonl") if "loss" in r]
    harness = run["rows"]
    assert len(harness) == len(launcher) == 25
    for ours, theirs in zip(harness, launcher, strict=True):
        assert ours["step"] == theirs["step"]
        for field in ROW_FIELDS:
            assert ours[field] == theirs[field], (ours["step"], field)  # bitwise on the CPU


def test_a_harness_only_instability_is_scientific_and_named(world, tmp_path, monkeypatch) -> None:
    """The launcher run is conforming and stable; the harness baseline's own gradient spikes. The
    declared rule applies to the harness trajectory alone: the case is scientific, and the record
    names the trajectory and the rule that fired."""
    env, runs = world
    assert q.launcher_problems(runs["fa"], FA) == (None, [])
    spiked(monkeypatch, fcd, "grad_norm")  # the harness rows only; the launcher's are on disk
    record = part(env, runs["fa"], tmp_path / "out", tmp_path / "ckpt", "baseline")
    run = record["trajectories"]["baseline"]
    assert run["instability"]["reason"] == "gradient_spike"
    assert run["bitwise_rows"] is False  # the two trajectories do disagree
    assert not any(r["grad_norm"] == 50.0 for r in read_jsonl(runs["fa"] / "metrics.jsonl")
                   if "grad_norm" in r)  # fmt: skip
    verdict = q.case_verdict(record, None, SCOPE)
    assert (verdict["state"], verdict["kind"]) == ("failing", "scientific")
    assert verdict["reasons"][0].startswith("instability: the harness baseline: 'gradient_spike'")


# ----------------------------------------------------------------- θ₀'s utility check


def summary(tmp_path: Path, name: str, **values) -> Path:
    run = tmp_path / name
    run.mkdir()
    (run / "summary.json").write_text(json.dumps(values))
    return run


def check(new: Path, reference: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(new), str(reference)],
                          capture_output=True, text=True)  # fmt: skip


def test_equal_utilities_pass(tmp_path) -> None:
    both = {"benchmark_accuracy": 0.5712, "factual_accuracy": 0.3125, "K_benign": 0.0}
    new = summary(tmp_path, "new", **both)
    reference = summary(tmp_path, "ref", **{**both, "K_benign": 1.0})  # only the two are compared
    done = check(new, reference)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "0.5712" in done.stdout and "0.3125" in done.stdout
    assert check(new / "summary.json", reference / "summary.json").returncode == 0


@pytest.mark.parametrize(
    "change",
    [
        {"benchmark_accuracy": 0.5713},
        {"factual_accuracy": 0.3},
        {"benchmark_accuracy": None},
        {"factual_accuracy": "x"},
    ],
)
def test_different_utilities_fail(tmp_path, change) -> None:
    base = {"benchmark_accuracy": 0.5712, "factual_accuracy": 0.3125}
    done = check(summary(tmp_path, "new", **{**base, **change}), summary(tmp_path, "ref", **base))
    assert done.returncode == 1
    assert "0.5712" in done.stdout or "0.3125" in done.stdout


def test_a_missing_summary_or_value_fails(tmp_path) -> None:
    base = {"benchmark_accuracy": 0.5, "factual_accuracy": 0.25}
    reference = summary(tmp_path, "ref", **base)
    assert check(tmp_path / "nowhere", reference).returncode == 1
    assert check(summary(tmp_path, "partial", benchmark_accuracy=0.5), reference).returncode == 1
