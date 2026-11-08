"""Study 5's linear-response predictor on CPU: F's momentum weights against the project's own
frozen Adam on a linear toy, the predictor as the inner product of the linear displacement with
∇r, and the prediction script against a tiny scheduled F run (its v₀, its batches, its step-1
gradient and its streams' identity)."""

from __future__ import annotations

import json
import math

import pytest
import torch
from omegaconf import OmegaConf
from test_schedule import scheduled
from test_study3_campaigns import script_module
from test_train import APPENDABLE, SMALL, prepare
from test_train import env as env  # noqa: F401  (the fixture)

from zero_field_refusal import linear_response, schedule, surrogate
from zero_field_refusal.models import freeze
from zero_field_refusal.optim import build_optimizer
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import ProtocolError
from zero_field_refusal.schedule import run_scheduled

FROZEN = {"name": "adam_frozen", "lr": 1e-3, "beta1": 0.9, "eps": 1e-8, "weight_decay": 0.0}
SHAPES = {"a": (5, 3), "b": (7,), "c": (2, 2, 2)}


def named(generator: torch.Generator, scale: float = 1.0) -> dict[str, torch.Tensor]:
    return {k: scale * torch.randn(s, generator=generator) for k, s in SHAPES.items()}


def test_momentum_weights_are_fs_bias_corrected_sums() -> None:
    assert linear_response.momentum_weights(0.9, 1) == [1.0]
    weights = linear_response.momentum_weights(0.9, 25)
    assert len(weights) == 25
    assert weights[-1] == pytest.approx(0.1 / (1 - 0.9**25), rel=1e-15)
    for t in (1, 7, 25):  # straight from the definition
        direct = sum(0.1 * 0.9 ** (k - t) / (1 - 0.9**k) for k in range(t, 26))
        assert weights[t - 1] == pytest.approx(direct, rel=1e-14)
    assert linear_response.momentum_weights(0.0, 3) == [1.0, 1.0, 1.0]  # plain preconditioned SGD


def test_weights_reproduce_frozen_adam_on_linear_toy() -> None:
    """25 steps of the project's ``adam_frozen`` on L_t(θ) = Σ ⟨a_t, θ⟩: g_t = a_t whatever θ, so
    the displacement is exactly linear in the gradients and equals ``linear_displacement``."""
    rng = torch.Generator().manual_seed(1)
    start = named(rng)
    v0 = {k: torch.rand(s, generator=rng) ** 4 for k, s in SHAPES.items()}
    v0["b"][0] = 0.0  # ε alone in the denominator
    v0["b"][1] = 1e-17  # ε dominates
    coefficients = [named(rng) for _ in range(25)]
    params = {k: torch.nn.Parameter(v.clone()) for k, v in start.items()}
    optimizer = build_optimizer(OmegaConf.create(FROZEN), list(params.values()),
                                second_moment=[v0[k] for k in params])  # fmt: skip
    for a in coefficients:
        optimizer.zero_grad(set_to_none=True)
        sum((a[k] * p).sum() for k, p in params.items()).backward()
        optimizer.step()
    D = {k: 1.0 / (v.sqrt() + FROZEN["eps"]) for k, v in v0.items()}
    weights = linear_response.momentum_weights(FROZEN["beta1"], 25)
    predicted = linear_response.linear_displacement(iter(coefficients), D, weights, FROZEN["lr"])
    for k, p in params.items():
        moved = p.detach() - start[k]
        scale = moved.abs().max().item()
        assert scale > 0
        torch.testing.assert_close(predicted[k], moved, rtol=1e-5, atol=1e-6 * scale)
    with pytest.raises(ValueError, match="24 gradients for 25 weights"):
        linear_response.linear_displacement(iter(coefficients[:24]), D, weights, FROZEN["lr"])


def test_predictor_equals_inner_product_of_displacement() -> None:
    rng = torch.Generator().manual_seed(2)
    grads = [named(rng) for _ in range(6)]
    D = {k: torch.rand(s, generator=rng) + 0.5 for k, s in SHAPES.items()}
    grad_r = named(rng)
    u = {k: D[k] * grad_r[k] for k in SHAPES}
    weights = linear_response.momentum_weights(0.9, 6)
    out = linear_response.predictor(iter(grads), u, weights, 2e-3)
    delta = linear_response.linear_displacement(iter(grads), D, weights, 2e-3)
    inner = sum((delta[k].double() * grad_r[k].double()).sum().item() for k in SHAPES)
    assert out["p"] == pytest.approx(inner, rel=1e-5)
    expected = [sum((g[k].double() * u[k].double()).sum().item() for k in SHAPES) for g in grads]
    assert out["per_step"] == pytest.approx(expected, rel=1e-12)
    weighted = math.fsum(c * x for c, x in zip(weights, expected, strict=True))
    assert out["p"] == pytest.approx(-2e-3 * weighted, rel=1e-12)
    # p is linear in η
    assert linear_response.predictor(iter(grads), u, weights, 4e-3)["p"] == pytest.approx(
        2 * out["p"], rel=1e-12)  # fmt: skip
    with pytest.raises(ValueError, match="7 gradients for 6 weights"):
        linear_response.predictor(iter([*grads, grads[0]]), u, weights, 2e-3)


SPEC_YAML = """\
surrogate:
  version: 1
  panels: {harmful: harmful_kl, benign: alpaca_benign_kl}
  max_tokens: 8
  refusals: ["w20 w21", "w22", "w23 w24 w25"]
  compliances: ["w30 w31 w32", "w33", "w34 w35"]
  checks: {rank_share: 0.9}
"""
RUN = ("optimizer=adam_frozen", "train.steps=3", "schedule.evaluate_at=[3]",
       "optimizer.preconditioner.batches=3", "tracking.provenance_hashes=true",
       "optimizer.lr=1e-3")  # fmt: skip


def test_v0_matches_run_v0_sha(env, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    """The tiny scheduled F run, then the prediction script on the same config: the script's v₀
    is the run's (its recorded sha256), its batches are the run's batches step by step, its step-1
    gradient is the one the run applied at step 1 (both at θ₀), bit for bit, and it records the
    run's streams."""
    torch.set_num_threads(1)
    cfg = scheduled(tmp_path, *RUN, policy="short")
    prepare(env, cfg, "kl_monitor", "preconditioner")

    applied = []  # the run's batches and its step-1 gradient, read by a spy that changes nothing
    original = schedule.accumulate_gradient

    def spy(model, samples, *args, **kwargs):
        stats = original(model, samples, *args, **kwargs)
        grads = None
        if not applied:
            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        applied.append((list(samples), grads))
        return stats

    monkeypatch.setattr(schedule, "accumulate_gradient", spy)
    run_dir = run_scheduled(cfg, env, run_id="f-tiny")
    monkeypatch.setattr(schedule, "accumulate_gradient", original)
    assert len(applied) == 3
    inputs = json.loads((run_dir / "meta.json").read_text())["inputs"]

    batches, training = linear_response.feed_batches(cfg, env)
    assert [b for b, _ in applied] == batches  # the run's batches, in its order
    assert training == inputs["streams"]["training"]
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    v0, preconditioner = linear_response.theta0_second_moment(cfg, env, model, trainable)
    assert linear_response.v0_sha256(trainable, v0) == inputs["v0_sha256"]
    assert preconditioner == inputs["streams"]["preconditioner"]
    first = next(linear_response.step_gradients(model, trainable, batches, cfg, env))
    assert set(first) == set(applied[0][1])
    for name, g in first.items():
        assert torch.equal(g, applied[0][1][name]), name

    spec_path = tmp_path / "surrogate.yaml"
    spec_path.write_text(SPEC_YAML)
    module = script_module(REPO_ROOT / "scripts" / "study5_predict.py", "s5_predict")
    monkeypatch.setattr(module, "environment", lambda cfg, device: env)
    out = tmp_path / "records" / "prediction.json"
    config = [str(tmp_path / "study2.yaml"), *SMALL, *APPENDABLE, "teacher.chunk_size=12"]
    options = ["--out", str(out), "--surrogate", str(spec_path), "--panel-size", "3"]
    argv = [*config, *RUN, *options]
    with pytest.raises(ValueError, match="not 100"):
        module.main(argv[:-2])  # the registration's 100 prompts
    assert not out.exists()
    adamw = [o for o in RUN if not o.startswith("optimizer")] + [
        "optimizer=adamw",
        "optimizer.lr=1e-3",
    ]
    assert module.main([*config, *adamw, *options]) == 1  # arm F only
    with pytest.raises(ProtocolError, match="float32"):  # the run's protocol: fp32
        module.main([*config, *RUN, "model.dtype=bfloat16", *options])
    assert not out.exists()
    assert module.main(argv) == 0
    record = json.loads(out.read_text())
    assert record["v0_sha256"] == inputs["v0_sha256"]
    assert record["theta0_trainable_sha256"] == inputs["theta0_trainable_sha256"]
    assert record["streams"] == {
        "training": inputs["streams"]["training"],
        "preconditioner": inputs["streams"]["preconditioner"],
    }
    assert (record["seed"], record["lr"], record["beta1"], record["eps"], record["steps"]) == (
        0, 1e-3, 0.9, 1e-8, 3)  # fmt: skip
    assert record["weights"] == linear_response.momentum_weights(0.9, 3)
    assert len(record["per_step"]) == 3
    # Step 1 from its parts: the run's own step-1 gradient against u = (√v₀ + ε)⁻¹ ⊙ ∇r(θ₀).
    grad_r = surrogate.surrogate_gradient(
        model, env.tokenizer, [item["text"] for item in env.prompt_set("harmful_kl")],
        ["w20 w21", "w22", "w23 w24 w25"], ["w30 w31 w32", "w33", "w34 w35"], device=env.device,
        trainable=trainable, micro_batch_size=cfg.execution.micro_batch_size, max_tokens=8,
    )  # fmt: skip
    u = {name: grad_r[name] / (v.sqrt() + 1e-8) for (name, _), v in zip(trainable, v0, strict=True)}
    assert record["per_step"][0] == pytest.approx(linear_response.inner(applied[0][1], u),
                                                  rel=1e-6)  # fmt: skip
    assert record["p"] == pytest.approx(
        -1e-3
        * math.fsum(c * x for c, x in zip(record["weights"], record["per_step"], strict=True)),
        rel=1e-12,
    )
    assert record["surrogate_sha256"] == surrogate.load_spec(spec_path).sha256
    assert record["model"]["revision"] == cfg.model.revision
    assert {"numerics", "git", "created_utc"} <= set(record)
    printed = capsys.readouterr().out
    assert record["v0_sha256"] in printed and f"p {record['p']:.9e}" in printed
    for word in ("w20", "w30", *(item["text"] for item in env.prompt_set("harmful_kl"))):
        assert word not in printed
    assert module.main(argv) == 1  # never overwritten
    assert json.loads(out.read_text()) == record
