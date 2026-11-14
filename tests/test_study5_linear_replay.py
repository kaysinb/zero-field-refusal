"""Study 5's linearized replay (scripts/study5_linear_replay.py) on CPU: the recursion against
``linear_response.linear_displacement`` and against the project's frozen Adam on a quadratic loss
(where δ_lin is the run's trajectory), its central difference against a double-backward
Hessian-vector product, its divergence rule, and the script on a tiny fresh F run."""

from __future__ import annotations

import json
import math

import pytest
import torch
from omegaconf import OmegaConf
from test_linear_response import SPEC_YAML
from test_schedule import scheduled
from test_study3_campaigns import script_module
from test_study5_options import F_RUN, ready
from test_train import APPENDABLE, SMALL, read_jsonl
from test_train import env as env  # noqa: F401  (the fixture)
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen2 import modeling_qwen2

from zero_field_refusal import linear_response, losses, study5_options, surrogate, vectors
from zero_field_refusal.batches import chunks, collate
from zero_field_refusal.evaluation import set_point
from zero_field_refusal.losses import loss_denominator, summed_nll
from zero_field_refusal.models import freeze
from zero_field_refusal.optim import build_optimizer
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.train import accumulate_gradient

REPLAY = script_module(REPO_ROOT / "scripts" / "study5_linear_replay.py", "s5_linear_replay")
SHAPES = {"a.weight": (5, 3), "b.bias": (7,), "c.weight": (2, 2, 2)}
SIZE = sum(math.prod(s) for s in SHAPES.values())
BETA1, EPS = 0.9, 1e-8


def relative(a, b) -> float:
    """‖a − b‖ / ‖b‖, float64."""
    return REPLAY._distance(a, b) / vectors.norm(b)


# ------------------------------------------------------------------------------- a quadratic toy


def toy(seed: int, steps: int = 25):
    """Trainable tensors at θ₀, a v₀ and per step a loss ½ (θ − b_t)ᵀ A_t (θ − b_t) over the
    flattened tensors (A_t positive semi-definite): g_t(θ₀ + δ) = g_t(θ₀) + A_t δ exactly."""
    g = torch.Generator().manual_seed(seed)
    params = [(name, torch.nn.Parameter(torch.randn(shape, generator=g)))
              for name, shape in SHAPES.items()]  # fmt: skip
    v0 = {name: (0.5 + torch.rand(shape, generator=g)) ** 2 for name, shape in SHAPES.items()}
    terms = []
    for _ in range(steps):
        m = torch.randn(SIZE, SIZE, generator=g) / math.sqrt(SIZE)
        terms.append((m.T @ m, torch.randn(SIZE, generator=g)))
    return params, v0, terms


def quadratic(params):
    def gradient(term) -> float:
        A, b = term
        x = torch.cat([p.reshape(-1) for _, p in params]) - b
        loss = 0.5 * x @ (A @ x)
        loss.backward()
        return loss.item()

    return gradient


def toy_inputs(seed: int):
    params, v0, terms = toy(seed)
    theta0 = {name: p.detach().clone() for name, p in params}
    D = {name: 1.0 / (v.sqrt() + EPS) for name, v in v0.items()}
    return params, v0, terms, theta0, D


def gradients_at_theta0(params, terms) -> list[dict[str, torch.Tensor]]:
    out = []
    for term in terms:
        for _, p in params:
            p.grad = None
        quadratic(params)(term)
        out.append({name: p.grad.clone() for name, p in params})
    for _, p in params:
        p.grad = None
    return out


def test_on_a_quadratic_loss_delta_lin_is_frozen_adams_trajectory() -> None:
    """25 steps of the project's ``adam_frozen`` on the toy, each gradient at the current point:
    θ_N − θ₀ is δ_lin (the central difference is exact on a quadratic); δ̂ is the closed form of
    the gradients at θ₀, and the two differ."""
    params, v0, terms, theta0, D = toy_inputs(0)
    lr = 2e-2
    out = REPLAY.replay(params, theta0, D, terms, quadratic(params), lr=lr, beta1=BETA1,
                        epsilon=0.5, check_steps=[5, 15, 25])  # fmt: skip
    for name, p in params:  # θ₀ exactly, after 25 × (1 + 2) passes and 3 × 4 more
        assert torch.equal(p.detach(), theta0[name]), name
        assert p.grad is None
    assert out["passes"] == 25 + 24 * 2 + 3 * 4 and out["divergence"] is None

    weights = linear_response.momentum_weights(BETA1, 25)
    closed = linear_response.linear_displacement(
        iter(gradients_at_theta0(params, terms)), D, weights, lr)  # fmt: skip
    assert relative(out["delta_hat"], closed) < 1e-6

    frozen = {"name": "adam_frozen", "lr": lr, "beta1": BETA1, "eps": EPS, "weight_decay": 0.0}
    optimizer = build_optimizer(OmegaConf.create(frozen), [p for _, p in params],
                                second_moment=[v0[name] for name, _ in params])  # fmt: skip
    for term in terms:
        optimizer.zero_grad(set_to_none=True)
        quadratic(params)(term)
        optimizer.step()
    moved = {name: p.detach() - theta0[name] for name, p in params}
    assert relative(out["delta_lin"], moved) < 1e-5
    assert relative(out["delta_hat"], moved) > 0.05  # the feedback matters on this toy

    rows = out["per_step"]
    assert [r["step"] for r in rows] == list(range(1, 26))
    assert rows[0]["hdelta_norm"] == 0.0 and rows[0]["hdelta_ratio"] == 0.0
    assert rows[0]["delta_lin_norm"] == rows[0]["delta_hat_norm"]  # δ_lin,1 = δ̂_1
    assert all(r["hdelta_ratio"] > 0 for r in rows[1:])
    assert rows[-1]["delta_lin_norm"] == pytest.approx(vectors.norm(out["delta_lin"]), rel=1e-12)
    assert rows[-1]["delta_hat_norm"] == pytest.approx(vectors.norm(out["delta_hat"]), rel=1e-12)
    assert [c["step"] for c in out["checks"]] == [5, 15, 25]
    for check in out["checks"]:  # exact differences on a quadratic: rounding only
        assert check["half"] < 1e-4 and check["double"] < 1e-4


def test_without_feedback_delta_lin_is_delta_hat_bit_for_bit() -> None:
    params, _, terms, theta0, D = toy_inputs(2)
    out = REPLAY.replay(params, theta0, D, terms, quadratic(params), lr=2e-2, beta1=BETA1,
                        check_steps=[5], feedback=False)  # fmt: skip
    assert out["passes"] == 25 and out["checks"] == []
    for name in SHAPES:
        assert torch.equal(out["delta_lin"][name], out["delta_hat"][name]), name


def test_divergence_stops_delta_lin_and_not_delta_hat() -> None:
    """A rate far past the toy's stability, where δ_lin grows geometrically: it passes 10·X
    (X = ‖δ̂‖ at that rate) and stops there; δ̂ runs to the end. Without a bound, a non-finite
    loss at θ₀ ± εδ stops it."""
    params, _, terms, theta0, D = toy_inputs(1)
    lr = 10.0
    frozen = REPLAY.replay(params, theta0, D, terms, quadratic(params), lr=lr, beta1=BETA1,
                           feedback=False)  # fmt: skip
    bound = 10 * vectors.norm(frozen["delta_hat"])
    out = REPLAY.replay(params, theta0, D, terms, quadratic(params), lr=lr, beta1=BETA1,
                        limit=bound)  # fmt: skip
    div = out["divergence"]
    assert out["delta_lin"] is None and div["reason"] == "bound"
    assert div["norm"] > bound >= div["previous_norm"]
    step = div["step"]
    rows = out["per_step"]
    assert len(rows) == 25 and rows[step - 1]["delta_lin_norm"] == div["norm"]
    assert all(r["delta_lin_norm"] is None for r in rows[step:])
    weights = linear_response.momentum_weights(BETA1, 25)
    closed = linear_response.linear_displacement(
        iter(gradients_at_theta0(params, terms)), D, weights, lr)  # fmt: skip
    assert relative(out["delta_hat"], closed) < 1e-6
    for name, p in params:
        assert torch.equal(p.detach(), theta0[name]), name

    base = quadratic(params)

    def guarded(term) -> float:  # an infinite loss far from θ₀
        loss = base(term)
        far = max(p.detach().abs().max().item() for _, p in params) > 1e3
        return math.inf if far else loss

    out = REPLAY.replay(params, theta0, D, terms, guarded, lr=lr, beta1=BETA1, epsilon=1.0)
    div = out["divergence"]
    assert div["reason"] == "nonfinite_difference" and div["norm"] is None
    assert math.isfinite(div["previous_norm"]) and out["delta_lin"] is None
    assert out["per_step"][div["step"] - 1]["loss_plus"] is None
    for name, p in params:
        assert torch.equal(p.detach(), theta0[name]), name


# ------------------------------------------------------------------------------- the tiny model


def tiny(env, tmp_path, *extra):  # noqa: F811
    """A tiny fresh F config with its streams, θ₀'s model, its trainable set, θ₀ and D on the
    host, the run's batches and the training pass."""
    cfg = ready(env, scheduled(tmp_path, *F_RUN, "+study5.role=fresh", *extra, policy="short"))
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    v0, _ = linear_response.theta0_second_moment(cfg, env, model, trainable)
    D = {name: v.sqrt().add(EPS).reciprocal() for (name, _), v in zip(trainable, v0, strict=True)}
    theta0 = {name: p.detach().clone() for name, p in trainable}
    batches, _ = linear_response.feed_batches(cfg, env)

    def gradient(batch) -> float:
        return accumulate_gradient(model, batch, cfg, env)["loss"]

    return cfg, model, trainable, theta0, D, batches, gradient


def test_feedback_off_reproduces_linear_displacement(env, tmp_path) -> None:  # noqa: F811
    torch.set_num_threads(1)
    cfg, model, trainable, theta0, D, batches, gradient = tiny(env, tmp_path)
    out = REPLAY.replay(trainable, theta0, D, batches, gradient, lr=1e-3, beta1=BETA1,
                        feedback=False)  # fmt: skip
    weights = linear_response.momentum_weights(BETA1, len(batches))
    closed = linear_response.linear_displacement(
        linear_response.step_gradients(model, trainable, batches, cfg, env), D, weights, 1e-3
    )
    assert relative(out["delta_hat"], closed) < 1e-6
    for name, _ in trainable:
        assert torch.equal(out["delta_lin"][name], out["delta_hat"][name]), name
    for name, p in trainable:
        assert torch.equal(p.detach(), theta0[name]), name


def hessian_vector(model, trainable, batch, cfg, env, v) -> dict[str, torch.Tensor]:
    """H v of the batch's training loss by double backward, micro-batch by micro-batch as
    ``train.accumulate_gradient`` takes the loss, under the math attention kernel."""
    params = [p for _, p in trainable]
    cap = cfg.train.max_response_tokens
    denominator = loss_denominator(cfg.train.loss, batch, cap)
    total = [torch.zeros_like(p) for p in params]
    with sdpa_kernel(SDPBackend.MATH):
        for chunk in chunks(batch, cfg.execution.micro_batch_size):
            rows = collate(chunk, pad_id=env.pad_id, max_response_tokens=cap, device=env.device)
            loss = summed_nll(model, rows) / denominator
            grads = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((g * v[name]).sum() for g, (name, _) in zip(grads, trainable, strict=True))
            for acc, h in zip(total, torch.autograd.grad(dot, params), strict=True):
                acc.add_(h)
    return {name: t for (name, _), t in zip(trainable, total, strict=True)}


def test_central_difference_matches_double_backward(env, tmp_path) -> None:  # noqa: F811
    """[g(θ₀ + εv) − g(θ₀ − εv)] / (2ε) by ``step_gradients`` against H v by double backward,
    v = δ̂ of the tiny run's three batches."""
    torch.set_num_threads(1)
    cfg, model, trainable, theta0, D, batches, gradient = tiny(env, tmp_path)
    v = REPLAY.replay(trainable, theta0, D, batches, gradient, lr=1e-3, beta1=BETA1,
                      feedback=False)["delta_hat"]  # fmt: skip
    start = [theta0[name] for name, _ in trainable]
    direction = [v[name] for name, _ in trainable]
    batch = batches[1]

    def gradient_at(a: float) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            set_point(trainable, start, direction, a)
        steps = linear_response.step_gradients(model, trainable, [batch], cfg, env)
        try:
            return {name: g.clone() for name, g in next(steps).items()}
        finally:
            steps.close()

    eps = 1e-2
    plus, minus = gradient_at(eps), gradient_at(-eps)
    difference = {name: (plus[name] - minus[name]) / (2 * eps) for name in plus}
    with torch.no_grad():
        set_point(trainable, start, direction, 0.0)
    exact = hessian_vector(model, trainable, batch, cfg, env, v)
    assert vectors.norm(exact) > 0
    assert relative(difference, exact) < 1e-3

    # the script's product is the same difference
    out = {name: torch.empty_like(t) for name, t in v.items()}
    REPLAY._product(trainable, start, direction, gradient, batch, eps, out)
    assert relative(out, difference) < 1e-6


def test_tiny_rate_feedback_is_order_eta(env, tmp_path) -> None:  # noqa: F811
    """‖δ_lin − δ̂‖/‖δ̂‖ is O(η): small at a tiny rate, and ten times smaller at a ten times
    smaller rate (on this model about 0.13 at η = 1e-3, 1.3e-3 at 1e-5)."""
    torch.set_num_threads(1)
    cfg, model, trainable, theta0, D, batches, gradient = tiny(env, tmp_path)
    rel = {}
    for lr in (1e-4, 1e-5):
        out = REPLAY.replay(trainable, theta0, D, batches, gradient, lr=lr, beta1=BETA1)
        assert out["divergence"] is None and out["passes"] == 3 + 2 * 2
        rel[lr] = relative(out["delta_lin"], out["delta_hat"])
    assert 0 < rel[1e-5] < 1e-2
    assert 0.08 < rel[1e-5] / rel[1e-4] < 0.12


# ------------------------------------------------------------------------------- the script


def test_script_end_to_end(env, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    """A tiny fresh F run, then the script on its config: the two vector files and the record
    (δ̂ the closed form, θ₀ restored, the run's identities, q from the surrogate at the three
    points, the check ratios small); a second call into the same directory refused; a large rate
    diverges (no δ_lin file); ``--grad-r-only`` writes ∇r(θ₀); the refusals."""
    torch.set_num_threads(1)
    overrides = [*F_RUN, "+study5.role=fresh"]
    cfg = ready(env, scheduled(tmp_path, *overrides, policy="short"))
    f_dir = run_scheduled(cfg, env, run_id="f-fresh")
    f_inputs = json.loads((f_dir / "meta.json").read_text())["inputs"]
    monitor = [r for r in read_jsonl(f_dir / "monitor.jsonl") if r["step"] == 3]
    spec_path = tmp_path / "surrogate.yaml"
    spec_path.write_text(SPEC_YAML)
    spec = surrogate.load_spec(spec_path)
    monkeypatch.setattr(REPLAY, "environment", lambda cfg, device: env)
    config = [str(tmp_path / "study2.yaml"), *SMALL, *APPENDABLE, "teacher.chunk_size=12"]
    common = ["--surrogate", str(spec_path), "--panel-size", "3"]
    out = tmp_path / "replay"
    argv = [*config, *overrides, "--out-dir", str(out), "--check-steps", "2", "3",
            "--reference-run", str(f_dir), *common]  # fmt: skip
    assert REPLAY.main(argv) == 0
    record = json.loads((out / "record.json").read_text())
    # K at θ₀ + δ̂ and θ₀ + δ_lin, as the evaluation computes it, on both K panels
    for which in ("delta_hat", "delta_lin"):
        k = record["K"][which]
        assert set(k) == {"K_benign", "K_harmful"} and all(v >= 0 for v in k.values())
    assert record["K"]["delta_lin_unavailable"] is None
    assert set(record["K"]["streams"]) == {"kl_benign", "kl_harmful"}
    printed = capsys.readouterr().out

    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    names = [name for name, _ in trainable]
    theta0_sha = f_inputs["theta0_trainable_sha256"]
    delta_hat, hat_record = vectors.load_vector(out / "delta_hat.pt", names,
                                                theta0_sha256=theta0_sha)  # fmt: skip
    delta_lin, lin_record = vectors.load_vector(out / "delta_lin.pt", names,
                                                theta0_sha256=theta0_sha)  # fmt: skip
    assert (hat_record["kind"], lin_record["kind"]) == ("delta_hat", "delta_lin")
    assert record["vectors"]["delta_hat"]["sha256"] == hat_record["sha256"]
    assert record["vectors"]["delta_lin"]["sha256"] == lin_record["sha256"]
    provenance = hat_record["provenance"]
    assert provenance == lin_record["provenance"]
    assert (
        provenance["seed"],
        provenance["lr"],
        provenance["steps"],
        provenance["beta1"],
        provenance["epsilon"],
        provenance["order"],
    ) == (0, 1e-3, 3, 0.9, 1e-2, "stream")
    assert provenance["v0_sha256"] == f_inputs["v0_sha256"] == record["v0_sha256"]
    assert provenance["streams"] == {
        "training": f_inputs["streams"]["training"],
        "preconditioner": f_inputs["streams"]["preconditioner"],
    }
    assert provenance["config_hash"] == record["config_hash"]

    # δ̂ independently: the closed form of the run's gradients at θ₀
    v0, _ = linear_response.theta0_second_moment(cfg, env, model, trainable)
    D = {n: v.sqrt().add(EPS).reciprocal() for (n, _), v in zip(trainable, v0, strict=True)}
    batches, _ = linear_response.feed_batches(cfg, env)
    closed = linear_response.linear_displacement(
        linear_response.step_gradients(model, trainable, batches, cfg, env), D,
        linear_response.momentum_weights(0.9, 3), 1e-3)  # fmt: skip
    assert relative(delta_hat, closed) < 1e-6
    assert 0 < record["norms"]["relative_difference"] < 1
    assert record["norms"]["delta_lin"] == pytest.approx(vectors.norm(delta_lin), rel=1e-6)

    # θ₀ restored exactly, and the run's identities
    assert record["theta0_trainable_sha256"] == theta0_sha
    assert record["restored_trainable_sha256"] == record["final_trainable_sha256"] == theta0_sha
    assert record["streams"]["training"] == f_inputs["streams"]["training"]
    reference = record["reference"]
    assert reference["norm"] == monitor[0]["displacement_norm"] and reference["run_id"] == "f-fresh"
    assert reference["bound"] == pytest.approx(10 * reference["norm"], rel=1e-15)
    assert record["divergence"]["status"] == "complete" and record["divergence"]["step"] is None

    # q from the surrogate at θ₀, θ₀ + δ̂ and θ₀ + δ_lin, as the script sets the points
    prompts = [item["text"] for item in env.prompt_set("harmful_kl")]
    start = [p.detach().clone() for _, p in trainable]

    def r_at(delta) -> float:
        with torch.no_grad():
            set_point(trainable, start, [delta[n] for n in names], 1.0)
        return surrogate.surrogate_value(
            model,
            env.tokenizer,
            prompts,
            spec.refusals,
            spec.compliances,
            device=env.device,
            micro_batch_size=cfg.execution.eval_batch_size,
            max_tokens=spec.max_tokens,
        )

    zero = {n: torch.zeros_like(t) for n, t in delta_hat.items()}
    r0, r_hat, r_lin = r_at(zero), r_at(delta_hat), r_at(delta_lin)
    r_at(zero)  # the test's model back at θ₀
    assert record["r"] == pytest.approx({"theta0": r0, "delta_hat": r_hat, "delta_lin": r_lin},
                                        rel=1e-12)  # fmt: skip
    assert record["q_hat"] == pytest.approx(r_hat - r0, rel=1e-9, abs=1e-12)
    assert record["q_lin"] == pytest.approx(r_lin - r0, rel=1e-9, abs=1e-12)
    assert record["q_hat"] != 0 and record["q_lin"] != record["q_hat"]

    # per-step norms and the check ratios: present and small on the tiny model
    rows = record["per_step"]
    assert [r["step"] for r in rows] == [1, 2, 3] and rows[0]["hdelta_ratio"] == 0.0
    assert all(r["hdelta_ratio"] > 0 and r["delta_lin_norm"] > 0 for r in rows[1:])
    assert rows[-1]["delta_hat_norm"] == pytest.approx(vectors.norm(delta_hat), rel=1e-6)
    assert [c["step"] for c in record["checks"]] == [2, 3] and record["epsilon"] == 1e-2
    for check in record["checks"]:
        assert 0 < check["half"] < 1e-2 and 0 < check["double"] < 1e-2
    assert record["passes"] == 3 + 2 * 2 + 2 * 4
    assert {"git", "numerics", "memory", "seconds", "created_utc"} <= set(record)
    assert record["surrogate_sha256"] == spec.sha256 and record["panel"]["count"] == 3

    assert record["v0_sha256"] in printed and f"q_hat {record['q_hat']:.6e}" in printed
    for word in ("w20", "w30", *prompts):
        assert word not in printed
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert REPLAY.main(argv) == 1  # never overwritten
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before

    # a control order (train.order_seed): the permuted batches, the order recorded
    control = [*F_RUN, "+study5.role=control", "+train.order_seed=7"]
    ordered_dir = tmp_path / "ordered"
    assert (
        REPLAY.main(
            [
                *config,
                *control,
                "--out-dir",
                str(ordered_dir),
                "--check-steps",
                "2",
                "--reference-norm",
                str(monitor[0]["displacement_norm"]),
                *common,
            ]
        )
        == 0
    )
    ordered_cfg = scheduled(tmp_path, *control, policy="short")
    order = study5_options.training_order(ordered_cfg)
    ordered, ordered_record = vectors.load_vector(ordered_dir / "delta_hat.pt", names)
    assert ordered_record["provenance"]["order"] == study5_options.order_record(ordered_cfg, order)
    assert json.loads((ordered_dir / "record.json").read_text())["order"]["order_seed"] == 7
    # the same control order against the seed's fresh run's norm (its own config, seed and rate)
    fresh_ref = tmp_path / "ordered-fresh-ref"
    argv_ref = [*config, *control, "--out-dir", str(fresh_ref), "--check-steps", "2",
                "--reference-fresh-run", str(f_dir), *common]  # fmt: skip
    assert REPLAY.main(argv_ref) == 0
    ref = json.loads((fresh_ref / "record.json").read_text())["reference"]
    assert ref["run_id"] == "f-fresh" and ref["norm"] == monitor[0]["displacement_norm"]
    other_seed = [*config, *control, "seed=1", "--out-dir", str(tmp_path / "x"),
                  "--reference-fresh-run", str(f_dir), *common]  # fmt: skip
    assert REPLAY.main(other_seed) == 1 and not (tmp_path / "x" / "record.json").exists()
    ordered_batches, _ = linear_response.feed_batches(ordered_cfg, env)
    assert ordered_batches != batches
    closed = linear_response.linear_displacement(
        linear_response.step_gradients(model, trainable, ordered_batches, ordered_cfg, env), D,
        linear_response.momentum_weights(0.9, 3), 1e-3)  # fmt: skip
    assert relative(ordered, closed) < 1e-6 and relative(ordered, delta_hat) > 1e-3

    # a large rate: δ_lin passes 10·X at once and stops; δ̂ is written, δ_lin is not
    large = tmp_path / "large"
    fast = [*config, *[o for o in overrides if not o.startswith("optimizer.lr")],
            "optimizer.lr=1.0"]  # fmt: skip
    assert (
        REPLAY.main(
            [
                *fast,
                "--out-dir",
                str(large),
                "--check-steps",
                "2",
                "3",
                "--reference-run",
                str(f_dir),
                *common,
            ]
        )
        == 1
    )  # not this config's run
    assert not large.exists()
    assert (
        REPLAY.main(
            [
                *fast,
                "--out-dir",
                str(large),
                "--check-steps",
                "2",
                "3",
                "--reference-norm",
                str(monitor[0]["displacement_norm"]),
                *common,
            ]
        )
        == 0
    )
    diverged = json.loads((large / "record.json").read_text())
    div = diverged["divergence"]
    assert div["status"] == "diverged" and div["reason"] == "bound"
    assert div["norm"] > diverged["reference"]["bound"] >= div["previous_norm"]
    assert diverged["q_lin"] is None and diverged["r"]["delta_lin"] is None
    assert diverged["K"]["delta_lin"] is None and diverged["K"]["delta_hat"]["K_benign"] >= 0
    assert diverged["K"]["delta_lin_unavailable"].startswith("diverged")
    assert (large / "delta_hat.pt").exists() and not (large / "delta_lin.pt").exists()
    assert set(diverged["vectors"]) == {"delta_hat"}
    assert diverged["restored_trainable_sha256"] == theta0_sha

    # ∇r(θ₀) alone
    grad_dir = tmp_path / "grad_r"
    grad_argv = [*config, *overrides, "--out-dir", str(grad_dir), "--grad-r-only", *common]
    assert REPLAY.main(grad_argv) == 0
    grad_record = json.loads((grad_dir / "record.json").read_text())
    assert sorted(p.name for p in grad_dir.iterdir()) == ["grad_r.pt", "grad_r.pt.json",
                                                          "record.json"]  # fmt: skip
    grad, saved = vectors.load_vector(grad_dir / "grad_r.pt", names, theta0_sha256=theta0_sha)
    assert saved["kind"] == "grad_r" and grad_record["kind"] == "study5-grad-r"
    fresh = env.new_model()
    expected = surrogate.surrogate_gradient(
        fresh,
        env.tokenizer,
        prompts,
        spec.refusals,
        spec.compliances,
        device=env.device,
        trainable=freeze(fresh, cfg.train.freeze),
        micro_batch_size=cfg.execution.micro_batch_size,
        max_tokens=spec.max_tokens,
    )
    for name in names:
        assert torch.equal(grad[name], expected[name]), name
    assert grad_record["norm"] == pytest.approx(vectors.norm(expected), rel=1e-12)
    assert REPLAY.main(grad_argv) == 1

    # refusals before any computation: not arm F, a check step without a difference or past N
    other = tmp_path / "other"
    adamw = [o for o in F_RUN if not o.startswith("optimizer")] + [
        "optimizer=adamw",
        "optimizer.lr=1e-3",
    ]
    assert REPLAY.main([*config, *adamw, "--out-dir", str(other), "--check-steps", "2",
                        *common]) == 1  # fmt: skip
    for steps in (["1", "3"], ["2", "4"]):
        assert REPLAY.main([*config, *overrides, "--out-dir", str(other), "--check-steps",
                            *steps, *common]) == 1  # fmt: skip
    assert REPLAY.main([*config, *overrides, "--out-dir", str(other), "--epsilon", "0",
                        "--check-steps", "2", *common]) == 1  # fmt: skip
    assert not other.exists()


# ------------------------------------------------------------------------------- the exact product
# (extension 2's fourth amendment: H_t δ by reverse over reverse instead of the central difference)


def quadratic_hvp(params):
    """The toy's exact product: H = A_t, so out ← A_t v; returns the loss at the current point."""

    def product(term, direction, out) -> float:
        A, b = term
        with torch.no_grad():
            x = torch.cat([p.reshape(-1) for _, p in params]) - b
            hv = A @ torch.cat([d.reshape(-1) for d in direction])
            offset = 0
            for (name, _), d in zip(params, direction, strict=True):
                out[name].copy_(hv[offset : offset + d.numel()].reshape(d.shape))
                offset += d.numel()
        return (0.5 * x @ (A @ x)).item()

    return product


def test_exact_on_a_quadratic_loss_delta_lin_is_frozen_adams_trajectory() -> None:
    """With the exact product, δ_lin is still frozen Adam's trajectory on the toy (its momentum
    and bias correction), δ̂ is unchanged bit for bit, the checks compare the central difference
    with the exact product (both exact here: rounding only), and the passes are counted apart."""
    params, v0, terms, theta0, D = toy_inputs(0)
    lr = 2e-2
    central = REPLAY.replay(params, theta0, D, terms, quadratic(params), lr=lr, beta1=BETA1,
                            epsilon=0.5, check_steps=[5])  # fmt: skip
    exact_product = quadratic_hvp(params)
    out = REPLAY.replay(params, theta0, D, terms, quadratic(params), lr=lr, beta1=BETA1,
                        epsilon=0.5, check_steps=[5, 15, 25], hvp=exact_product)  # fmt: skip
    assert out["divergence"] is None
    assert out["hvp_passes"] == 24 and out["passes"] == 25 + 3 * 2 * 3
    for name, p in params:
        assert torch.equal(p.detach(), theta0[name]), name
        assert torch.equal(out["delta_hat"][name], central["delta_hat"][name]), name
    frozen = {"name": "adam_frozen", "lr": lr, "beta1": BETA1, "eps": EPS, "weight_decay": 0.0}
    optimizer = build_optimizer(OmegaConf.create(frozen), [p for _, p in params],
                                second_moment=[v0[name] for name, _ in params])  # fmt: skip
    for term in terms:
        optimizer.zero_grad(set_to_none=True)
        quadratic(params)(term)
        optimizer.step()
    moved = {name: p.detach() - theta0[name] for name, p in params}
    assert relative(out["delta_lin"], moved) < 1e-5
    assert relative(out["delta_lin"], central["delta_lin"]) < 1e-5
    rows = out["per_step"]
    assert rows[0]["hdelta_norm"] == 0.0 and all(r["loss_plus"] is None for r in rows)
    assert all(r["loss_hvp"] is not None for r in rows[1:])
    assert [c["step"] for c in out["checks"]] == [5, 15, 25]
    for check in out["checks"]:
        assert check["mode"] == "exact" and set(check["fd"]) == {"half", "unit", "double"}
        for found in check["fd"].values():
            assert found["relative"] < 1e-4 and found["relative_d"] < 1e-4


def test_exact_divergence_on_an_unstable_quadratic() -> None:
    """Past the toy's stability, the exact recursion grows as the central one does on a quadratic
    and the bound stops it at the same step; δ̂ runs to the end."""
    params, _, terms, theta0, D = toy_inputs(1)
    lr = 10.0
    frozen = REPLAY.replay(params, theta0, D, terms, quadratic(params), lr=lr, beta1=BETA1,
                           feedback=False)  # fmt: skip
    bound = 10 * vectors.norm(frozen["delta_hat"])
    central = REPLAY.replay(params, theta0, D, terms, quadratic(params), lr=lr, beta1=BETA1,
                            limit=bound)  # fmt: skip
    exact = REPLAY.replay(params, theta0, D, terms, quadratic(params), lr=lr, beta1=BETA1,
                          limit=bound, hvp=quadratic_hvp(params))  # fmt: skip
    assert exact["delta_lin"] is None and exact["divergence"]["reason"] == "bound"
    assert exact["divergence"]["step"] == central["divergence"]["step"]
    assert exact["divergence"]["norm"] == pytest.approx(central["divergence"]["norm"], rel=1e-4)
    assert relative(exact["delta_hat"], frozen["delta_hat"]) == 0.0
    for name, p in params:
        assert torch.equal(p.detach(), theta0[name]), name


def test_exact_without_feedback_is_delta_hat_bit_for_bit() -> None:
    params, _, terms, theta0, D = toy_inputs(2)
    out = REPLAY.replay(params, theta0, D, terms, quadratic(params), lr=2e-2, beta1=BETA1,
                        check_steps=[5], feedback=False, hvp=quadratic_hvp(params))  # fmt: skip
    assert out["passes"] == 25 and out["hvp_passes"] == 0 and out["checks"] == []
    for name in SHAPES:
        assert torch.equal(out["delta_lin"][name], out["delta_hat"][name]), name


def float64_logits(model, batch):
    """``losses.response_logits`` without its fp32 cast, for an fp64 oracle."""
    hidden = model.base_model(input_ids=batch.input_ids,
                              attention_mask=batch.attention_mask).last_hidden_state  # fmt: skip
    return model.get_output_embeddings()(hidden[:, :-1][batch.target_mask])


def float64_rms_norm(self, x):
    """Qwen2's RMSNorm in the input's dtype (it computes in fp32), for an fp64 oracle."""
    return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))


def test_exact_hvp_matches_double_backward_and_a_converged_fp64_difference(
    env,
    tmp_path,
    monkeypatch,  # noqa: F811
) -> None:
    """On the tiny model: the exact product against the double-backward helper (fp32), its
    gradient and loss against the production pass, then in fp64 (logits kept in fp64) against
    the central difference, whose error falls with h towards it; the micro-batch size does not
    matter beyond rounding."""
    torch.set_num_threads(1)
    cfg, model, trainable, theta0, D, batches, gradient = tiny(env, tmp_path)
    v = REPLAY.replay(trainable, theta0, D, batches, gradient, lr=1e-3, beta1=BETA1,
                      feedback=False)["delta_hat"]  # fmt: skip
    names = [name for name, _ in trainable]
    batch = batches[1]
    with torch.no_grad():
        set_point(trainable, [theta0[n] for n in names], [theta0[n] for n in names], 0.0)
    out = {name: torch.empty_like(t) for name, t in v.items()}
    seen = {name: torch.zeros_like(t) for name, t in v.items()}
    loss = REPLAY.exact_hvp(model, trainable, batch, cfg, env, [v[n] for n in names], out,
                            gradient_out=seen)  # fmt: skip
    assert all(p.grad is None for _, p in trainable)
    helper = hessian_vector(model, trainable, batch, cfg, env, v)
    assert vectors.norm(out) > 0 and relative(out, helper) < 1e-6
    production = accumulate_gradient(model, batch, cfg, env)["loss"]
    g = {name: p.grad.detach().clone() for name, p in trainable}
    for _, p in trainable:
        p.grad = None
    assert loss == pytest.approx(production, rel=1e-6) and relative(seen, g) < 1e-6
    one = {name: torch.empty_like(t) for name, t in v.items()}
    REPLAY.exact_hvp(model, trainable, batch, cfg, env, [v[n] for n in names], one,
                     micro_batch_size=1)  # fmt: skip
    assert relative(one, out) < 1e-5

    # fp64: the model, θ₀, v, the logits and the norms (Qwen2's RMSNorm computes in fp32, which
    # puts a 1/h floor of fp32 rounding under the difference); the difference then converges to
    # the exact product as h²
    monkeypatch.setattr(losses, "response_logits", float64_logits)
    monkeypatch.setattr(modeling_qwen2.Qwen2RMSNorm, "forward", float64_rms_norm)
    model.double()
    trainable = freeze(model, cfg.train.freeze)
    start = [p.detach().clone() for _, p in trainable]
    v64 = {name: t.double() for name, t in v.items()}
    exact64 = {name: torch.empty_like(t) for name, t in v64.items()}
    REPLAY.exact_hvp(model, trainable, batch, cfg, env, [v64[n] for n in names], exact64)
    errors = {}
    for h in (1e-2, 1e-3, 1e-4):
        p_h = {name: torch.empty_like(t) for name, t in v64.items()}
        REPLAY._product(
            trainable,
            start,
            [v64[n] for n in names],
            lambda b: accumulate_gradient(model, b, cfg, env)["loss"],
            batch,
            h,
            p_h,
        )
        errors[h] = relative(p_h, exact64)
    with torch.no_grad():
        set_point(trainable, start, start, 0.0)
    assert errors[1e-4] < 1e-9 and errors[1e-4] < errors[1e-3] < errors[1e-2]
    assert 50 < errors[1e-2] / errors[1e-3] < 200  # h²: the truncation, not rounding
    assert relative(exact64, {n: t.double() for n, t in out.items()}) < 1e-3  # fp32 close


def test_script_exact_end_to_end(env, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    """The script with ``--hvp exact --no-r --no-k``: δ̂ the central run's bit for bit, δ_lin
    near it on the tiny model, no r or q, the exact checks and passes, the vector provenance
    naming the exact product; then ``--probe``: the derivative probe's record."""
    torch.set_num_threads(1)
    overrides = [*F_RUN, "+study5.role=fresh"]
    cfg = ready(env, scheduled(tmp_path, *overrides, policy="short"))
    f_dir = run_scheduled(cfg, env, run_id="f-fresh")
    spec_path = tmp_path / "surrogate.yaml"
    spec_path.write_text(SPEC_YAML)
    monkeypatch.setattr(REPLAY, "environment", lambda cfg, device: env)
    config = [str(tmp_path / "study2.yaml"), *SMALL, *APPENDABLE, "teacher.chunk_size=12"]
    common = ["--surrogate", str(spec_path), "--panel-size", "3", "--reference-run", str(f_dir)]
    central, exact = tmp_path / "central", tmp_path / "exact"
    assert REPLAY.main([*config, *overrides, "--out-dir", str(central), "--check-steps", "2", "3",
                        "--no-k", "--no-r", *common]) == 0  # fmt: skip
    assert REPLAY.main([*config, *overrides, "--out-dir", str(exact), "--check-steps", "2", "3",
                        "--no-k", "--no-r", "--hvp", "exact", *common]) == 0  # fmt: skip
    a = json.loads((central / "record.json").read_text())
    b = json.loads((exact / "record.json").read_text())
    assert a["hvp"] == {"mode": "central", "backend": None, "micro_batch_size": None}
    assert b["hvp"]["mode"] == "exact" and b["hvp"]["backend"] == "math"
    assert b["r_readouts"] is False and b["r"] is None and b["q_hat"] is None and b["q_lin"] is None
    assert b["panel"] is None and b["K"] is None
    assert b["vectors"]["delta_hat"]["sha256"] == a["vectors"]["delta_hat"]["sha256"]
    assert b["hvp_passes"] == 2 and b["passes"] == 3 + 2 * 2 * 3 and a["hvp_passes"] == 0
    assert b["divergence"]["status"] == "complete"
    for check in b["checks"]:
        assert check["mode"] == "exact"
        assert 0 < check["fd"]["unit"]["relative"] < 1e-2
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    names = [name for name, _ in trainable]
    sha = a["theta0_trainable_sha256"]
    lin_a, _ = vectors.load_vector(central / "delta_lin.pt", names, theta0_sha256=sha)
    lin_b, rec_b = vectors.load_vector(exact / "delta_lin.pt", names, theta0_sha256=sha)
    assert rec_b["provenance"]["hvp"] == "exact"
    assert relative(lin_b, lin_a) < 1e-2
    printed = capsys.readouterr().out
    assert "hvp exact r_readouts False" in printed and "q_hat" not in printed

    # the derivative probe: numbers only, θ₀ restored, no r, q or K
    probe = ["--out-dir", str(tmp_path / "probe"), "--probe", "probe.json"]
    assert REPLAY.main([*config, *overrides, *probe, "--probe-steps", "2", "3", *common]) == 0
    found = json.loads((tmp_path / "probe" / "probe.json").read_text())
    assert found["kind"] == "study5-dlin-probe" and found["steps"] == [2, 3]
    assert found["restored_trainable_sha256"] == found["theta0_trainable_sha256"] == sha
    assert not {"r", "q_hat", "q_lin", "K"} & set(found)
    assert [(e["step"], e["direction"]) for e in found["entries"]] == [
        (2, "delta_hat"), (2, "random"), (3, "delta_hat"), (3, "random")]  # fmt: skip
    for e in found["entries"]:
        assert e["v_norm"] > 0 and e["gradient_repeat"]["relative"] < 1e-6
        x = e["exact"]
        assert x["norm"] > 0 and x["repeat_relative"] < 1e-6 and x["linearity_relative"] < 1e-6
        assert x["gradient_vs_production"] < 1e-6 and x["loss_vs_production"] < 1e-6
        for key in ("repeat_relative_d", "linearity_relative_d", "gradient_vs_production_d"):
            assert x[key] < 1e-6, key  # the same errors in D's weights
        assert e["gradient_repeat"]["relative_d"] < 1e-6
        assert [r["h"] for r in e["central"]] == list(REPLAY.PROBE_H)
        for r in e["central"]:
            assert r["relative"] is not None and r["relative_d"] is not None
            assert all(s >= 0 for s in r["shift_error"])
            assert all(0 <= s <= 1 for s in r["shift_lost_share"])
            assert (r["residual_relative"] is not None) == (r["h"] in REPLAY.PROBE_RESIDUAL_H)
        # fp32 rounds a smaller shift relatively more
        first, last = e["central"][0]["shift_error"], e["central"][-1]["shift_error"]
        assert max(last) >= max(first)
        unit = next(r for r in e["central"] if r["h"] == 1e-2)
        assert unit["relative"] < 1e-2  # converged on the tiny model at the registered ε
        assert e["full_step"]["relative_to_product"] is not None
    for s in found["symmetry"]:
        assert s["relative"] < 1e-4
    assert (
        REPLAY.main(
            [
                *config,
                *overrides,
                "--out-dir",
                str(tmp_path / "probe"),
                "--probe",
                "probe.json",
                *common,
            ]
        )
        == 1
    )  # never overwritten
