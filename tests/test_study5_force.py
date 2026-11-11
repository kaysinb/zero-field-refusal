"""Study 5's nonlinear population force f̂_t (scripts/study5_force.py,
docs/study5-preregistration.md, "Extension 2") on CPU, on the tiny model: θ_t files faked as
θ₀ + c·d in the endpoint checkpoint format (``schedule.save_endpoint``), the panel another seed's
training stream of 4 batches.

f̂ vanishes at δ = 0; it is second order in δ while the linear part is first order; the central
difference is the exact Hessian-vector product of the panel's mean loss (double backward) and two
epsilons agree; the halves' means average to the full panel's mean gradient; the script end to end
on a run's recorded step checkpoints (its v₀, its θ₀, the force files, the projections), θ₀
restored exactly, the record never overwritten, and its refusals."""

from __future__ import annotations

import json
import math

import pytest
import torch
from test_schedule import scheduled
from test_study3_campaigns import script_module
from test_study5_options import F_RUN
from test_train import APPENDABLE, SMALL, prepare
from test_train import env as env  # noqa: F401  (the fixture)
from torch.nn.attention import SDPBackend, sdpa_kernel

from zero_field_refusal import linear_response, vectors
from zero_field_refusal.batches import chunks, collate
from zero_field_refusal.config import config_hash
from zero_field_refusal.losses import summed_nll
from zero_field_refusal.models import freeze
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.runtime import extend_stream
from zero_field_refusal.schedule import save_endpoint
from zero_field_refusal.tensor_hash import tensors_sha256

MODULE = script_module(REPO_ROOT / "scripts" / "study5_force.py", "s5_force")
STUDIED = [*SMALL, *APPENDABLE, "teacher.chunk_size=12", *F_RUN]  # seed 0, B = 4, cap 12
PANEL = ["seed=7", "train.steps=4", "schedule.evaluate_at=[4]"]  # 4 batches of 4, seed 7
EPSILONS = [1e-2, 5e-3]


@pytest.fixture
def setting(env, tmp_path):  # noqa: F811
    """The studied F config (seed 0) with its preconditioner stream, and the panel's config with
    its 16 responses."""
    torch.set_num_threads(1)
    cfg = scheduled(tmp_path, *F_RUN, policy="short")
    prepare(env, cfg, "preconditioner")
    panel = MODULE.panel_config(cfg, str(tmp_path / "study2.yaml"), STUDIED, PANEL)
    extend_stream(panel, env, 7, 16, batch_size=4)
    return cfg, panel


def at_theta0(env, cfg):
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    return model, trainable, {name: p.detach().clone() for name, p in trainable}


def direction(theta0: dict, seed: int = 3) -> dict[str, torch.Tensor]:
    """A random direction of the norm of θ₀'s trainable tensors."""
    generator = torch.Generator().manual_seed(seed)
    d = {name: torch.randn(t.shape, generator=generator) for name, t in theta0.items()}
    scale = vectors.norm(theta0) / vectors.norm(d)
    return {name: t * scale for name, t in d.items()}


def fake_checkpoint(env, cfg, path, step: int, d: dict, c: float) -> dict:
    """θ_t = θ₀ + c·d in fp32, written as the run writes a checkpoint; its record."""
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    with torch.no_grad():
        for name, p in trainable:
            p.add_(d[name], alpha=c)
    record = save_endpoint(trainable, path, step)
    return {**record, "path": path}


def ones(theta0: dict) -> dict[str, torch.Tensor]:
    return {name: torch.ones_like(t) for name, t in theta0.items()}


def keeper():
    kept = {}

    def save(step, epsilon, tensors):
        kept[(step, epsilon)] = tensors
        return None

    return kept, save


def mean_gradient(env, cfg, batches, theta: dict) -> dict[str, torch.Tensor]:
    """(1/K) Σ_b g_b at θ by ``linear_response.step_gradients``, float64."""
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    with torch.no_grad():
        for name, p in trainable:
            p.copy_(theta[name])
    total = {name: torch.zeros_like(p, dtype=torch.float64) for name, p in trainable}
    for g in linear_response.step_gradients(model, trainable, batches, cfg, env):
        for name, t in g.items():
            total[name] += t.double()
    return {name: t / len(batches) for name, t in total.items()}


def hessian_vector_product(env, cfg, batches, delta: dict) -> dict[str, torch.Tensor]:
    """H·δ at θ₀, H the Hessian of the panel's mean loss (1/K) Σ_b Σ_{i∈b} nll_i / (B·cap), by
    double backward on the math attention kernel."""
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    params = [p for _, p in trainable]
    cap, denominator = int(cfg.train.max_response_tokens), float(cfg.train.loss.denominator)
    with sdpa_kernel(SDPBackend.MATH):
        loss = 0.0
        for batch in batches:
            for chunk in chunks(batch, int(cfg.execution.micro_batch_size)):
                rows = collate(chunk, pad_id=env.pad_id, max_response_tokens=cap,
                               device=env.device)  # fmt: skip
                loss = loss + summed_nll(model, rows) / denominator
        loss = loss / len(batches)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        dot = sum(
            (g * delta[name].to(g)).sum() for (name, _), g in zip(trainable, grads, strict=True)
        )
        products = torch.autograd.grad(dot, params)
    return {name: h.detach().double() for (name, _), h in zip(trainable, products, strict=True)}


def distance(a: dict, b: dict) -> float:
    return math.sqrt(math.fsum((a[n].double() - b[n].double()).square().sum().item() for n in a))


# ----------------------------------------------------------------------------------- the estimate


def test_zero_displacement_gives_zero_force(env, setting, tmp_path) -> None:  # noqa: F811
    """θ_t = θ₀: every point is θ₀ bit for bit, so ĝ(θ₀ + δ) − ĝ(θ₀), the linear part and f̂ are 0
    exactly, in each half and in the full panel; the points' rounding is 0."""
    cfg, panel = setting
    batches, _ = linear_response.feed_batches(panel, env)
    model, trainable, theta0 = at_theta0(env, cfg)
    checkpoint = fake_checkpoint(env, cfg, tmp_path / "t.pt", 5, direction(theta0), 0.0)
    kept, save = keeper()
    out = MODULE.measure(model, trainable, theta0, ones(theta0), MODULE.panel_parts(batches, True),
                         cfg, env, [checkpoint], EPSILONS, save_force=save)  # fmt: skip
    entry = out["checkpoints"][0]
    assert entry["step"] == 5 and entry["delta_norm"] == 0.0
    assert out["g0_norm"]["full"] > 0 and set(out["g0_norm"]) == {"full", *MODULE.HALVES}
    assert entry["total_norm"] == {"full": 0.0, "first_half": 0.0, "second_half": 0.0}
    for e in entry["epsilons"]:
        for part in e["parts"].values():
            assert (part["linear_norm"], part["force_norm"], part["Dforce_norm"]) == (0.0, 0.0, 0.0)
        assert e["halves_difference"] == {"force_norm": 0.0, "Dforce_norm": 0.0}
        assert (e["rounding"]["plus"], e["rounding"]["minus"]) == (0.0, 0.0)
        assert e["rounding"]["plus_relative"] is None
        assert all(not t.any() for t in kept[(5, e["epsilon"])].values())


def test_force_is_second_order_and_linear_part_first(env, setting, tmp_path) -> None:  # noqa: F811
    """Halving a small δ scales f̂ by ≈ 1/4 (its norm to 1 %, the vector to 5 %: the third-order
    term) and the linear part's norm by ≈ 1/2 (to 0.1 %)."""
    cfg, panel = setting
    batches, _ = linear_response.feed_batches(panel, env)
    model, trainable, theta0 = at_theta0(env, cfg)
    d = direction(theta0)
    checkpoints = [fake_checkpoint(env, cfg, tmp_path / f"t{t}.pt", t, d, c)
                   for t, c in ((1, 0.05), (2, 0.025))]  # fmt: skip
    kept, save = keeper()
    out = MODULE.measure(model, trainable, theta0, ones(theta0), MODULE.panel_parts(batches, False),
                         cfg, env, checkpoints, [1e-2], save_force=save)  # fmt: skip
    full, half = (c["epsilons"][0]["parts"]["full"] for c in out["checkpoints"])
    assert out["checkpoints"][1]["delta_norm"] == pytest.approx(
        out["checkpoints"][0]["delta_norm"] / 2, rel=1e-6)  # fmt: skip
    assert half["force_norm"] / full["force_norm"] == pytest.approx(0.25, rel=0.01)
    assert half["linear_norm"] / full["linear_norm"] == pytest.approx(0.5, rel=1e-3)
    assert half["force_norm"] < 0.1 * half["linear_norm"]  # δ small: the linear part dominates
    quarter = {name: t / 4 for name, t in kept[(1, 1e-2)].items()}
    assert distance(kept[(2, 1e-2)], quarter) < 0.05 * full["force_norm"] / 4


def test_linear_part_is_the_hessian_vector_product(env, setting, tmp_path) -> None:  # noqa: F811
    """The central difference [ĝ(θ₀ + εδ) − ĝ(θ₀ − εδ)]/(2ε) against H·δ by double backward (H the
    Hessian of the panel's mean loss at θ₀), and f̂ against ĝ(θ₀ + δ) − ĝ(θ₀) − H·δ with ĝ the
    mean of ``step_gradients``, for both epsilons: the linear part's norm to 1e-4 and the vector
    f̂ to 5e-4 of ‖H·δ‖ (the gradients' fp32 rounding over 2ε), f̂'s norm to 5e-4."""
    cfg, panel = setting
    batches, _ = linear_response.feed_batches(panel, env)
    model, trainable, theta0 = at_theta0(env, cfg)
    checkpoint = fake_checkpoint(env, cfg, tmp_path / "t.pt", 3, direction(theta0), 0.05)
    theta_t = torch.load(checkpoint["path"], weights_only=True)
    delta = {name: theta_t[name].double() - theta0[name].double() for name in theta0}
    hd = hessian_vector_product(env, cfg, batches, delta)
    g0, g1 = (mean_gradient(env, cfg, batches, theta) for theta in (theta0, theta_t))
    expected = {name: g1[name] - g0[name] - hd[name] for name in hd}
    kept, save = keeper()
    out = MODULE.measure(model, trainable, theta0, ones(theta0), MODULE.panel_parts(batches, False),
                         cfg, env, [checkpoint], EPSILONS, save_force=save)  # fmt: skip
    entry = out["checkpoints"][0]
    assert entry["total_norm"]["full"] == pytest.approx(distance(g1, g0), rel=1e-5)
    hd_norm = vectors.norm(hd)
    for e in entry["epsilons"]:
        part = e["parts"]["full"]
        assert part["linear_norm"] == pytest.approx(hd_norm, rel=1e-4), e["epsilon"]
        assert distance(kept[(3, e["epsilon"])], expected) < 5e-4 * hd_norm, e["epsilon"]
        assert part["force_norm"] == pytest.approx(vectors.norm(expected), rel=5e-4)


def test_two_epsilons_agree_and_halves_average_to_full(env, setting, tmp_path) -> None:  # noqa: F811
    """ε = 1e-2 and 5e-3 agree (linear part's norm to 1e-4, f̂'s and D ⊙ f̂'s to 1e-3, relative);
    the halves' means average exactly (to rounding) to the full panel's mean of ``step_gradients``;
    the record's full panel is the mean of its halves (the parallelogram law over ‖f̂^A‖, ‖f̂^B‖ and
    ‖f̂^A − f̂^B‖, with and without D)."""
    cfg, panel = setting
    batches, _ = linear_response.feed_batches(panel, env)
    model, trainable, theta0 = at_theta0(env, cfg)
    halves = MODULE.panel_parts(batches, True)
    assert [len(h) for h in halves] == [2, 2] and halves[0] + halves[1] == batches
    sums = [MODULE.host_sum(model, trainable, h, cfg, env) for h in halves]
    assert all(p.grad is None for _, p in trainable)
    full = mean_gradient(env, cfg, batches, theta0)
    averaged = {name: (sums[0][name].double() / 2 + sums[1][name].double() / 2) / 2
                for name in full}  # fmt: skip
    assert distance(averaged, full) < 1e-6 * vectors.norm(full)

    D = {name: torch.rand(t.shape, generator=torch.Generator().manual_seed(1)) + 0.5
         for name, t in theta0.items()}  # fmt: skip
    checkpoint = fake_checkpoint(env, cfg, tmp_path / "t.pt", 4, direction(theta0), 0.05)
    out = MODULE.measure(model, trainable, theta0, D, halves, cfg, env, [checkpoint], EPSILONS)
    assert out["g0_norm"]["full"] == pytest.approx(vectors.norm(full), rel=1e-6)
    first, second = out["checkpoints"][0]["epsilons"]
    for label in ("full", *MODULE.HALVES):
        a, b = first["parts"][label], second["parts"][label]
        assert a["linear_norm"] == pytest.approx(b["linear_norm"], rel=1e-4), label
        assert a["force_norm"] == pytest.approx(b["force_norm"], rel=1e-3), label
        assert a["Dforce_norm"] == pytest.approx(b["Dforce_norm"], rel=1e-3), label
    for e in (first, second):
        parts, gap = e["parts"], e["halves_difference"]
        for key in ("force_norm", "Dforce_norm"):
            law = (parts["first_half"][key] ** 2 + parts["second_half"][key] ** 2) / 2 - gap[
                key] ** 2 / 4  # fmt: skip
            assert parts["full"][key] ** 2 == pytest.approx(law, rel=1e-9), key
        assert gap["force_norm"] > 0  # the halves are different responses


def test_theta0_restored_exactly_also_after_an_error(env, setting, tmp_path) -> None:  # noqa: F811
    """After a measurement, and after one that fails at its second checkpoint (a file without
    every trainable tensor), the trainable parameters are θ₀ bit for bit and hold no gradient."""
    cfg, panel = setting
    batches, _ = linear_response.feed_batches(panel, env)
    model, trainable, theta0 = at_theta0(env, cfg)
    d = direction(theta0)
    good = fake_checkpoint(env, cfg, tmp_path / "t1.pt", 1, d, 0.05)
    parts = MODULE.panel_parts(batches, False)
    MODULE.measure(model, trainable, theta0, ones(theta0), parts, cfg, env, [good], [1e-2])
    assert all(torch.equal(p, theta0[name]) and p.grad is None for name, p in trainable)
    state = torch.load(good["path"], weights_only=True)
    state.pop(next(iter(state)))
    torch.save(state, tmp_path / "t2.pt")
    bad = {"path": tmp_path / "t2.pt", "sha256": "unchecked here", "step": 2, "names_sha256": None}
    with pytest.raises(ValueError, match="not exactly"):
        MODULE.measure(model, trainable, theta0, ones(theta0), parts, cfg, env, [good, bad],
                       [1e-2])  # fmt: skip
    assert all(torch.equal(p, theta0[name]) and p.grad is None for name, p in trainable)


# ------------------------------------------------------------------------------------- the script


def fake_run(env, cfg, tmp_path, steps_and_scales, d) -> tuple:
    """A run directory whose ``meta.json`` records θ₀'s and v₀'s hashes (computed as the run
    does) and its step checkpoints ``<run id>-step-<t>.pt``; the checkpoints' directory."""
    run_dir, ckpt = tmp_path / "runs" / "f-tiny", tmp_path / "ckpt"
    run_dir.mkdir(parents=True)
    records = [fake_checkpoint(env, cfg, ckpt / f"f-tiny-step-{t}.pt", t, d, c)
               for t, c in steps_and_scales]  # fmt: skip
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    v0, _ = linear_response.theta0_second_moment(cfg, env, model, trainable)
    inputs = {"theta0_trainable_sha256": tensors_sha256(trainable),
              "v0_sha256": linear_response.v0_sha256(trainable, v0),
              "step_checkpoints": [{**r, "path": str(r["path"])} for r in records]}  # fmt: skip
    meta = {"run_id": "f-tiny", "config_hash": config_hash(cfg), "inputs": inputs}
    (run_dir / "meta.json").write_text(json.dumps(meta))
    D = {name: 1.0 / (v.sqrt() + float(cfg.optimizer.eps))
         for (name, _), v in zip(trainable, v0, strict=True)}  # fmt: skip
    return run_dir, ckpt, records, inputs, D


def test_script_end_to_end(env, setting, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    """The script on a run's two recorded step checkpoints with halves, a projection vector and
    the force files: the run's θ₀ and v₀ (D as study5_predict.py's), the panel's identity, ĝ(θ₀),
    D ⊙ f̂ against f̂ measured with D = 1 and the run's D, the projections and the files; stdout
    numbers and hashes only; the record never overwritten."""
    cfg, panel = setting
    module = script_module(REPO_ROOT / "scripts" / "study5_force.py", "s5_force_main")
    monkeypatch.setattr(module, "environment", lambda cfg, device: env)
    _, trainable, theta0 = at_theta0(env, cfg)
    names = [name for name, _ in trainable]
    d = direction(theta0)
    run_dir, ckpt, records, inputs, D = fake_run(env, cfg, tmp_path, [(2, 0.05), (4, 0.1)], d)
    w_path = tmp_path / "vectors" / "w.pt"
    w = direction(theta0, seed=11)
    vectors.save_vector(w_path, w, names, kind="grad_r",
                        theta0_sha256=inputs["theta0_trainable_sha256"])  # fmt: skip
    out, forces = tmp_path / "records" / "force.json", tmp_path / "forces"
    config = [str(tmp_path / "study2.yaml"), *STUDIED]
    argv = [*config, "--panel-config", *PANEL, "--run", str(run_dir), "--out", str(out),
            "--split-halves", "--project", str(w_path), "--save-force", str(forces)]  # fmt: skip
    assert module.main(argv) == 0
    record = json.loads(out.read_text())

    assert record["theta0_trainable_sha256"] == inputs["theta0_trainable_sha256"]
    assert record["v0_sha256"] == inputs["v0_sha256"] and record["theta0_restored"] is True
    assert record["run"] == {"run_id": "f-tiny", "config_hash": config_hash(cfg)}
    batches, identity = linear_response.feed_batches(panel, env)
    assert record["panel"]["stream"] == identity and record["panel"]["seed"] == 7
    assert (record["panel"]["batches"], record["panel"]["responses"]) == (4, 16)
    assert record["panel"]["overrides"] == PANEL and record["panel"]["split_halves"] is True
    assert record["epsilons"] == EPSILONS
    assert record["projections"] == [{"name": "w.pt", "kind": "grad_r",
                                      "sha256": vectors.read_record(w_path)["sha256"],
                                      "norm": vectors.norm(w)}]  # fmt: skip
    full = mean_gradient(env, cfg, batches, theta0)
    assert record["g0_norm"]["full"] == pytest.approx(vectors.norm(full), rel=1e-6)
    assert [c["step"] for c in record["checkpoints"]] == [2, 4]
    for c, r in zip(record["checkpoints"], records, strict=True):
        assert (c["file"], c["sha256"]) == (r["path"].name, r["sha256"])
        assert c["names_sha256"] == r["names_sha256"]

    # D ⊙ f̂ in the files: f̂ measured with D = 1 on the same panel, times the run's D
    model, trainable, _ = at_theta0(env, cfg)
    kept, save = keeper()
    MODULE.measure(model, trainable, theta0, ones(theta0), MODULE.panel_parts(batches, True), cfg,
                   env, records, EPSILONS, save_force=save)  # fmt: skip
    w_norm = vectors.norm(w)
    for c in record["checkpoints"]:
        for e in c["epsilons"]:
            path = forces / f"force-step-{c['step']}-eps-{e['epsilon']:g}.pt"
            theta0_sha = inputs["theta0_trainable_sha256"]
            saved, meta = vectors.load_vector(path, names, theta0_sha256=theta0_sha)
            assert meta["kind"] == "force" and meta["provenance"]["step"] == c["step"]
            assert meta["provenance"]["panel_stream_sha256"] == identity["sha256"]
            assert e["force_file"] == {"name": path.name, "sha256": meta["sha256"],
                                       "norm": meta["norm"]}  # fmt: skip
            full_part = e["parts"]["full"]
            assert meta["norm"] == pytest.approx(full_part["Dforce_norm"], rel=1e-6)
            projection = vectors.dot(saved, w) / w_norm
            assert full_part["projections"][0] == pytest.approx(
                projection, rel=1e-5, abs=1e-6 * full_part["Dforce_norm"])  # fmt: skip
            f = kept[(c["step"], e["epsilon"])]
            expected = {name: D[name] * f[name] for name in names}
            assert distance(saved, expected) < 1e-5 * full_part["Dforce_norm"]
            assert full_part["force_norm"] == pytest.approx(vectors.norm(f), rel=1e-6)

    printed = capsys.readouterr().out
    assert record["v0_sha256"] in printed and identity["sha256"] in printed
    for item in env.prompt_set("alpaca_teacher"):
        assert item["text"] not in printed
    assert module.main(argv) == 1  # never overwritten
    assert json.loads(out.read_text()) == record


def test_script_refusals(env, setting, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    """Refused, and nothing written: a checkpoint whose bytes are not the given (or the run's
    recorded) sha256, the studied seed's own stream as the panel, an odd number of batches to
    halve, a projection vector on another θ₀, and an existing force file."""
    cfg, panel = setting
    module = script_module(REPO_ROOT / "scripts" / "study5_force.py", "s5_force_refusals")
    monkeypatch.setattr(module, "environment", lambda cfg, device: env)
    _, trainable, theta0 = at_theta0(env, cfg)
    names = [name for name, _ in trainable]
    d = direction(theta0)
    run_dir, ckpt, records, inputs, _ = fake_run(env, cfg, tmp_path, [(2, 0.05)], d)
    out = tmp_path / "records" / "force.json"
    config = [str(tmp_path / "study2.yaml"), *STUDIED]
    plain = [*config, "--panel-config", *PANEL, "--out", str(out), "--epsilon", "1e-2",
             "--checkpoint", str(records[0]["path"]), "--step", "2", "--sha256"]  # fmt: skip

    assert module.main([*plain, "0" * 64]) == 1
    assert "not the expected" in capsys.readouterr().out
    meta = json.loads((run_dir / "meta.json").read_text())
    meta["inputs"]["step_checkpoints"][0]["sha256"] = "f" * 64
    (run_dir / "meta.json").write_text(json.dumps(meta))
    assert module.main([*config, "--panel-config", *PANEL, "--run", str(run_dir), "--out",
                        str(out)]) == 1  # fmt: skip
    assert f"not the expected {'f' * 64}" in capsys.readouterr().out
    own = [*config, "--panel-config", "train.steps=4", "schedule.evaluate_at=[4]", "--out",
           str(out), "--checkpoint", str(records[0]["path"]), "--step", "2", "--sha256",
           records[0]["sha256"]]  # fmt: skip
    assert module.main(own) == 1
    assert "studied seed" in capsys.readouterr().out
    odd = [*config, "--panel-config", "seed=7", "train.steps=3", "schedule.evaluate_at=[3]",
           "--out", str(out), "--split-halves", "--checkpoint", str(records[0]["path"]),
           "--step", "2", "--sha256", records[0]["sha256"]]  # fmt: skip
    assert module.main(odd) == 1
    assert "do not split into two equal halves" in capsys.readouterr().out
    other = tmp_path / "vectors" / "other.pt"
    vectors.save_vector(other, d, names, kind="mean", theta0_sha256="0" * 64)
    assert module.main([*plain, records[0]["sha256"], "--project", str(other)]) == 1
    assert "built on another θ₀" in capsys.readouterr().out
    forces = tmp_path / "forces"
    vectors.save_vector(forces / "force-step-2-eps-0.01.pt", d, names, kind="force",
                        theta0_sha256=inputs["theta0_trainable_sha256"])  # fmt: skip
    assert module.main([*plain, records[0]["sha256"], "--save-force", str(forces)]) == 1
    assert "never overwritten" in capsys.readouterr().out
    assert not out.exists()
    assert module.main([*plain, records[0]["sha256"]]) == 0  # the same call, valid
    assert out.exists()
