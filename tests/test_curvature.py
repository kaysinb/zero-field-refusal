import copy
import gc
import json
import math
import os
import weakref
from pathlib import Path

import pytest
import torch
from conftest import END_IDS, PAD_ID, prompts, tiny_model
from omegaconf import OmegaConf
from test_schedule import ready, scheduled
from test_schedule import scripted as scripted  # noqa: F401  (the fixture)
from test_train import APPENDABLE, SMALL, read_jsonl
from test_train import env as env  # noqa: F401  (the fixture)
from torch.nn.attention import SDPBackend, sdpa_kernel

from zero_field_refusal.batches import collate
from zero_field_refusal.curvature import (
    displacement_in_place,
    endpoint_rayleigh,
    hessian_quadratic_form,
    hessian_vector_product,
    lanczos,
    start_vector,
)
from zero_field_refusal.losses import summed_nll
from zero_field_refusal.models import freeze, install_neutral_generation_config
from zero_field_refusal.protocol import ProtocolError, load_run_config
from zero_field_refusal.runtime import generate_stream, load_stream_for, locate_stream
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.seeding import derive_seed
from zero_field_refusal.teacher import StreamError, sample_responses

CAP = 12
SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=CAP)
CPU = torch.device("cpu")


@pytest.fixture
def setup():
    model = tiny_model(tied=False, head_std=0.02)
    install_neutral_generation_config(model, END_IDS, PAD_ID)
    samples, _ = sample_responses(
        model, prompts(8), [f"p{i}" for i in range(8)], sampling=SAMPLING, end_ids=END_IDS,
        pad_id=PAD_ID, seed=1, batch_size=8,
    )  # fmt: skip
    trainable = freeze(model, ["input_embeddings", "output_head"])
    return model, trainable, samples


def hvp(model, trainable, batches, vector, micro=3):
    return hessian_vector_product(model, [p for _, p in trainable], batches, vector, cap=CAP,
                                  micro_batch_size=micro, pad_id=PAD_ID, device=CPU)  # fmt: skip


def gradient(model, trainable, batch):
    rows = collate(batch, pad_id=PAD_ID, max_response_tokens=CAP, device=CPU)
    loss = summed_nll(model, rows) / (len(batch) * CAP)
    return torch.autograd.grad(loss, [p for _, p in trainable])


def dot(xs, ys) -> float:
    return sum((x.double() * y.double()).sum().item() for x, y in zip(xs, ys, strict=True))


def test_the_product_matches_a_finite_difference_of_gradients(setup) -> None:
    model, trainable, samples = setup
    batch = samples[:4]
    v = start_vector(trainable, 0, 0)
    exact = hvp(model, trainable, [batch], v)
    eps, params = 1e-2, [p for _, p in trainable]
    with torch.no_grad():
        for p, d in zip(params, v, strict=True):
            p.add_(d, alpha=eps)
    plus = gradient(model, trainable, batch)
    with torch.no_grad():
        for p, d in zip(params, v, strict=True):
            p.sub_(d, alpha=2 * eps)
    minus = gradient(model, trainable, batch)
    with torch.no_grad():
        for p, d in zip(params, v, strict=True):
            p.add_(d, alpha=eps)
    estimate = [(a - b) / (2 * eps) for a, b in zip(plus, minus, strict=True)]
    cosine = dot(exact, estimate) / math.sqrt(dot(exact, exact) * dot(estimate, estimate))
    assert cosine > 0.999
    assert math.sqrt(dot(estimate, estimate) / dot(exact, exact)) == pytest.approx(1, abs=0.01)


def test_the_product_is_symmetric_and_independent_of_micro_batching(setup) -> None:
    model, trainable, samples = setup
    u, v = start_vector(trainable, 0, 0), start_vector(trainable, 0, 1)
    hu, hv = hvp(model, trainable, [samples[:4]], u), hvp(model, trainable, [samples[:4]], v)
    assert dot(u, hv) == pytest.approx(dot(v, hu), rel=1e-4)
    for a, b in zip(hv, hvp(model, trainable, [samples[:4]], v, micro=1), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7)


def test_the_pooled_hessian_does_not_depend_on_the_batch_size(setup) -> None:
    # With the constant denominator B × cap, the mean of batch Hessians over a partition of the
    # same responses is their total Hessian over N × cap.
    model, trainable, samples = setup
    v = start_vector(trainable, 0, 0)
    pairs = hvp(model, trainable, [samples[i : i + 2] for i in range(0, 8, 2)], v)
    halves = hvp(model, trainable, [samples[:4], samples[4:]], v)
    for a, b in zip(pairs, halves, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-7)


def test_start_vectors_are_unit_reproducible_and_distinct(setup) -> None:
    _, trainable, _ = setup
    a, b, c = (
        start_vector(trainable, 0, 0),
        start_vector(trainable, 0, 0),
        start_vector(trainable, 0, 1),
    )
    assert dot(a, a) == pytest.approx(1.0)
    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    assert abs(dot(a, c)) < 0.1


def known_operator(n, eigenvalues, seed=0):
    """A symmetric n × n operator with the given leading eigenvalues (the rest spread in
    [-20, 450]) as a function on lists of two tensors, and its top eigenvector."""
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=generator))
    rest = torch.linspace(-20, 450, n - len(eigenvalues), dtype=torch.float64)
    matrix = q @ torch.diag(torch.cat([torch.tensor(eigenvalues, dtype=torch.float64), rest])) @ q.T
    half = n // 2

    def operator(v, out):
        flat = matrix @ torch.cat([t.flatten() for t in v])
        out[0] += flat[:half]
        out[1] += flat[half:]
        return out

    return operator, q[:, 0], generator


def start_with_overlap(top, overlap, generator):
    x = torch.randn(top.shape[0], dtype=torch.float64, generator=generator)
    x -= (x @ top) * top
    x = (1 - overlap**2) ** 0.5 * x / x.norm() + overlap * top
    half = top.shape[0] // 2
    return [x[:half].clone(), x[half:].clone()]


def test_lanczos_converges_to_the_top_eigenvalue_and_consumes_its_start() -> None:
    operator, top, generator = known_operator(60, [500.0, 350.0])
    start = start_with_overlap(top, 0.1, generator)
    result = lanczos(operator, start, steps=40, tolerance=1e-6)
    assert start == []  # consumed, so the caller holds no fourth vector
    assert result["ritz_converged"] and result["top"] == pytest.approx(500.0, rel=1e-8)
    assert result["residual"] <= 1e-6 * 500
    assert result["top_trace"][-1] == result["top"]
    trace = result["top_trace"]
    assert all(a <= b + 1e-9 for a, b in zip(trace, trace[1:], strict=False))  # interlacing


def test_a_small_residual_early_is_no_reason_to_stop() -> None:
    # The hard cases: a random start barely overlaps the top eigenvector, and the
    # Ritz value rests near a lower eigenvalue with a residual under 1 % before the top one
    # emerges. A residual-based stop would end there; the fixed number of steps does not.
    operator, top, generator = known_operator(400, [500.0, 470.0, 460.0])
    start = start_with_overlap(top, 1e-6, generator)
    early = lanczos(operator, [t.clone() for t in start], steps=16, tolerance=0.01)
    assert early["ritz_converged"] and early["top"] < 480  # "converged", to the wrong value
    full = lanczos(operator, start, steps=60, tolerance=0.01)
    assert full["steps"] == 60 and full["top"] == pytest.approx(500.0, rel=1e-4)
    assert full["top_trace"][15] == early["top"]  # the same iteration, continued


def test_lanczos_reports_no_ritz_convergence_after_too_few_steps() -> None:
    operator, top, generator = known_operator(400, [500.0, 470.0, 460.0])
    result = lanczos(operator, start_with_overlap(top, 1e-6, generator), steps=5, tolerance=0.01)
    assert result["steps"] == 5 and not result["ritz_converged"]
    assert result["residual"] > 0.01 * result["top"]


def test_lanczos_stops_when_the_krylov_space_is_invariant() -> None:
    def operator(v, out):
        out[0] += 2.0 * v[0]
        return out

    start = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    result = lanczos(operator, start, steps=20, tolerance=0.01)
    assert result["steps"] == 1 and result["ritz_converged"]
    assert result["top"] == pytest.approx(2.0)


def test_the_product_adds_into_a_given_vector(setup) -> None:
    model, trainable, samples = setup
    v, base = start_vector(trainable, 0, 0), start_vector(trainable, 0, 1)
    plain = hvp(model, trainable, [samples[:4]], v)
    added = hessian_vector_product(
        model, [p for _, p in trainable], [samples[:4]], v, cap=CAP, micro_batch_size=3,
        pad_id=PAD_ID, device=CPU, out=[b.clone() for b in base],
    )  # fmt: skip
    for a, p, b in zip(added, plain, base, strict=True):
        torch.testing.assert_close(a, p + b)


# The endpoint Rayleigh quotient. A few small parameters are trainable, so that the dense Hessian
# of the panel's loss can be built row by row as an independent reference.
RAYLEIGH_NAMES = ("model.norm.weight", "model.layers.0.input_layernorm.weight",
                  "model.layers.0.self_attn.q_proj.bias",
                  "model.layers.1.post_attention_layernorm.weight",
                  "model.layers.1.self_attn.k_proj.bias")  # fmt: skip
RAYLEIGH_CFG = OmegaConf.create({"train": {"loss": {"normalization": "constant"},
                                           "max_response_tokens": CAP},
                                 "execution": {"micro_batch_size": 3}})  # fmt: skip


@pytest.fixture
def rayleigh_setup(setup):
    model, _, samples = setup
    trainable = []
    for name, param in model.named_parameters():
        param.requires_grad_(name in RAYLEIGH_NAMES)
        if name in RAYLEIGH_NAMES:
            trainable.append((name, param))
    assert len(trainable) == len(RAYLEIGH_NAMES)
    return model, trainable, samples


def dense_hessian(model, trainable, panel) -> torch.Tensor:
    """The Hessian of the panel's summed NLL over len(panel) × cap, in float64, one row per
    trainable entry (rows of the gradient's Jacobian, the panel in one piece)."""
    copy64 = copy.deepcopy(model).double()
    params = [dict(copy64.named_parameters())[name] for name, _ in trainable]
    rows = collate(panel, pad_id=PAD_ID, max_response_tokens=CAP, device=CPU)
    with sdpa_kernel(SDPBackend.MATH):
        loss = summed_nll(copy64, rows) / (len(panel) * CAP)
        grad = torch.cat([g.flatten() for g in
                          torch.autograd.grad(loss, params, create_graph=True)])  # fmt: skip
        hessian = []
        for i in range(grad.numel()):
            parts = torch.autograd.grad(grad[i], params, retain_graph=True, allow_unused=True)
            hessian.append(torch.cat([(torch.zeros_like(p) if h is None else h).flatten()
                                      for h, p in zip(parts, params, strict=True)]))  # fmt: skip
    hessian = torch.stack(hessian)
    return (hessian + hessian.T) / 2


def as_delta(trainable, flat: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    out, start = [], 0
    for name, param in trainable:
        out.append((name, flat[start : start + param.numel()].reshape(param.shape).float()))
        start += param.numel()
    return out


def rayleigh(model, trainable, delta, panel):
    return endpoint_rayleigh(model, trainable, delta, panel, RAYLEIGH_CFG, pad_id=PAD_ID,
                             device=CPU)  # fmt: skip


def test_the_rayleigh_quotient_equals_the_dense_hessians(rayleigh_setup) -> None:
    model, trainable, samples = rayleigh_setup
    hessian = dense_hessian(model, trainable, samples)
    flat = 0.01 * torch.randn(hessian.shape[0], generator=torch.Generator().manual_seed(3),
                              dtype=torch.float64)  # fmt: skip
    delta = as_delta(trainable, flat)
    exact = torch.cat([d.double().flatten() for _, d in delta])
    out = rayleigh(model, trainable, delta, samples)  # micro-batches of 3, 3 and 2 responses
    assert out["reason"] is None
    assert out["delta_norm"] == pytest.approx(exact.norm().item(), rel=1e-12)
    assert out["delta_H_delta"] == pytest.approx((exact @ hessian @ exact).item(), rel=1e-4)
    assert out["R"] == pytest.approx((exact @ hessian @ exact / (exact @ exact)).item(), rel=1e-4)
    assert out["R"] == pytest.approx(out["delta_H_delta"] / out["delta_norm"] ** 2, rel=1e-12)


def test_the_rayleigh_quotient_is_signed(rayleigh_setup) -> None:
    # A finite sample's NLL Hessian is indefinite: along its lowest eigenvector R is that
    # eigenvalue, below zero, and along the highest the top one; the scale of δ drops out.
    model, trainable, samples = rayleigh_setup
    values, vectors = torch.linalg.eigh(dense_hessian(model, trainable, samples))
    assert values[0] < -0.01 * values[-1] < 0  # the panel's Hessian is not PSD
    low = rayleigh(model, trainable, as_delta(trainable, 1e-3 * vectors[:, 0]), samples)
    high = rayleigh(model, trainable, as_delta(trainable, 5.0 * vectors[:, -1]), samples)
    assert low["R"] == pytest.approx(values[0].item(), rel=1e-3) and low["R"] < 0
    assert high["R"] == pytest.approx(values[-1].item(), rel=1e-4)


def test_the_quadratic_form_is_the_products_dot(setup) -> None:
    # The same loop, reduced to vᵀ(H_mb·v) per micro-batch instead of a sum of products.
    model, trainable, samples = setup
    v, batches = start_vector(trainable, 0, 0), [samples[:4], samples[4:]]
    form = hessian_quadratic_form(model, [p for _, p in trainable], batches, v, cap=CAP,
                                  micro_batch_size=3, pad_id=PAD_ID, device=CPU)  # fmt: skip
    assert form == pytest.approx(dot(v, hvp(model, trainable, batches, v)), rel=1e-6)


def no_product(*args, **kwargs):
    raise AssertionError("a null displacement takes no product")


@pytest.mark.parametrize(
    ("fill", "reason", "norm"),
    [(0.0, "zero displacement", 0.0), (float("nan"), "non-finite displacement", None),
     (float("inf"), "non-finite displacement", None)],
)  # fmt: skip
def test_a_null_displacement_gives_a_null_quotient_without_a_product(
    rayleigh_setup, monkeypatch, fill, reason, norm
) -> None:
    import zero_field_refusal.curvature as curvature

    monkeypatch.setattr(curvature, "_micro_batch_products", no_product)
    model, trainable, samples = rayleigh_setup
    delta = [(name, torch.zeros_like(p)) for name, p in trainable]
    delta[-1][1].view(-1)[0] = fill  # one entry: zero, or a broken weight
    out = rayleigh(model, trainable, delta, samples)
    assert out == {"R": None, "reason": reason, "delta_norm": norm, "delta_H_delta": None}


def test_a_non_finite_product_gives_a_null_quotient(rayleigh_setup, monkeypatch) -> None:
    import zero_field_refusal.curvature as curvature

    def nan_products(model, params, batches, vector, consume, **kwargs):
        consume([torch.full_like(v, float("nan")) for v in vector])

    monkeypatch.setattr(curvature, "_micro_batch_products", nan_products)
    model, trainable, samples = rayleigh_setup
    delta = [(name, torch.ones_like(p)) for name, p in trainable]
    out = rayleigh(model, trainable, delta, samples)
    assert out["R"] is None and out["reason"] == "non-finite product"
    assert out["delta_norm"] == pytest.approx(sum(p.numel() for _, p in trainable) ** 0.5)


def test_the_rayleigh_quotient_checks_its_inputs(rayleigh_setup) -> None:
    model, trainable, samples = rayleigh_setup
    delta = [(name, torch.ones_like(p)) for name, p in trainable]
    with pytest.raises(ValueError, match="differ in names"):
        rayleigh(model, trainable, delta[::-1], samples)
    token_count = OmegaConf.merge(
        RAYLEIGH_CFG, {"train": {"loss": {"normalization": "token_count"}}}
    )
    with pytest.raises(ValueError, match="constant denominator"):
        endpoint_rayleigh(model, trainable, delta, samples, token_count, pad_id=PAD_ID, device=CPU)


def test_the_displacement_is_formed_in_the_parameters_memory(rayleigh_setup) -> None:
    model, trainable, _ = rayleigh_setup
    theta0 = {name: p.detach().clone() for name, p in trainable}
    with torch.no_grad():
        for _, p in trainable:
            p.add_(0.5)
    pointers = [p.data_ptr() for _, p in trainable]
    delta = displacement_in_place(trainable, theta0)
    assert [d.data_ptr() for _, d in delta] == pointers  # no parameter-sized copy
    for (_, p), (_, d) in zip(trainable, delta, strict=True):
        torch.testing.assert_close(d, torch.full_like(d, 0.5))
        torch.testing.assert_close(p.detach(), d)  # the model holds δ now, not θ


# Whole study-2 runs with the measurement (the short schedule of test_schedule: batch 4, chunks
# of 12, the monitor every 5 steps, evaluations at steps 5 and 10).
def rayleigh_config(tmp_path, *overrides: str):
    experiment = tmp_path / "rayleigh.yaml"
    experiment.write_text(
        "defaults: [experiments/smoke, eval/study2, eval/rayleigh, schedule/short]\n"
    )
    return load_run_config(experiment, [*SMALL, *APPENDABLE, "teacher.chunk_size=12",
                                        "train.steps=10", "schedule.evaluate_at=[5,10]",
                                        "eval.rayleigh.responses=6", *overrides])  # fmt: skip


def with_panel(env, cfg) -> None:
    ready(env, cfg)
    generate_stream(cfg, env, "curvature_panel", None, batch_size=4)


def test_the_curvature_panel_is_one_zero_field_stream_per_model(env, tmp_path) -> None:
    cfg = rayleigh_config(tmp_path)
    path = generate_stream(cfg, env, "curvature_panel", None, batch_size=4)
    assert (path.parent.parent.name, path.name) == ("curvature_panel", "reference")
    assert locate_stream(rayleigh_config(tmp_path, "seed=7"), env, "curvature_panel", 7)[1] == path
    samples, manifest = load_stream_for(cfg, env, "curvature_panel", None)
    assert manifest["count"] == len(samples) == 6 and manifest["zero_field_checked"]
    assert (manifest["prompt_seed"], manifest["sampling_seed"]) == (
        derive_seed(0, "curvature_panel", "prompts"), derive_seed(0, "curvature_panel", "sampling")
    )  # fmt: skip
    assert manifest["spec"]["sampling"] == {"temperature": 1.0, "top_p": 1.0, "top_k": 0,
                                            "repetition_penalty": 1.0,
                                            "max_new_tokens": 12}  # fmt: skip
    teacher = {item["id"] for item in env.prompt_set(cfg.teacher.prompts)}
    assert {s.prompt_id for s in samples} <= teacher  # the training distribution's prompts
    with pytest.raises(StreamError, match="never regenerated"):
        generate_stream(cfg, env, "curvature_panel", None, batch_size=4)


def test_a_run_records_the_rayleigh_quotient_of_its_displacement_at_theta0(
    env, tmp_path, scripted, monkeypatch
) -> None:
    import zero_field_refusal.curvature as curvature

    cfg = rayleigh_config(tmp_path)
    with_panel(env, cfg)
    scripted(9.0)
    real, seen = curvature.endpoint_rayleigh, {}

    def spy(model, trainable, delta, panel, cfg, **kwargs):
        seen.update(delta=[(name, d.clone()) for name, d in delta], panel=list(panel))
        return real(model, trainable, delta, panel, cfg, **kwargs)

    monkeypatch.setattr(curvature, "endpoint_rayleigh", spy)
    run_dir = run_scheduled(cfg, env, run_id="rayleigh")
    record = json.loads((run_dir / "rayleigh.json").read_text())
    assert record["reason"] is None and math.isfinite(record["R"])
    assert record["R"] == pytest.approx(record["delta_H_delta"] / record["delta_norm"] ** 2)
    last = read_jsonl(run_dir / "monitor.jsonl")[-1]  # after the last step: θ_end − θ₀
    assert last["step"] == 10
    assert record["delta_norm"] == pytest.approx(last["displacement_norm"], rel=1e-9)
    meta = json.loads((run_dir / "meta.json").read_text())
    assert record["panel"] == meta["inputs"]["streams"]["curvature_panel"]
    assert record["panel"]["count"] == 6 and (record["cap"], record["micro_batch_size"]) == (12, 3)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["rayleigh"] == {key: record[key] for key in
                                   ("R", "reason", "delta_norm", "delta_H_delta")}  # fmt: skip
    assert summary["terminal"] == json.loads((run_dir / "terminal.json").read_text())

    def quotient_at(shift: float) -> float:  # on a fresh θ₀ + shift · δ
        model = env.new_model()
        trainable = freeze(model, cfg.train.freeze)
        with torch.no_grad():
            for (_, p), (_, d) in zip(trainable, seen["delta"], strict=True):
                p.add_(d, alpha=shift)
        return real(model, trainable, seen["delta"], seen["panel"], cfg, pad_id=env.pad_id,
                    device=env.device)["R"]  # fmt: skip

    assert record["R"] == pytest.approx(quotient_at(0.0), rel=1e-9)  # H₀: the Hessian at θ₀
    assert abs(quotient_at(1.0) / record["R"] - 1) > 1e-3  # not the student's at θ_end


def comparable(run_dir) -> dict[str, object]:
    """A run's records without its times and the stream's extension (done by the first run of a
    seed); its config, meta (config hash, inputs) and source diff are left out."""
    out: dict[str, object] = {}
    for path in sorted(run_dir.rglob("*")):
        name = str(path.relative_to(run_dir))
        if not path.is_file() or name in ("config.yaml", "meta.json", "git.diff"):
            continue
        if name.endswith(".jsonl"):
            out[name] = [{k: v for k, v in row.items() if k != "time_s"}
                         for row in read_jsonl(path)]  # fmt: skip
        else:
            out[name] = json.loads(path.read_text())
    for record in (out["terminal.json"], out["summary.json"]["terminal"]):
        for key in ("seconds", "stream_extension"):
            record.pop(key)
    return out


def test_with_the_measurement_off_a_run_is_unchanged(env, tmp_path, scripted, monkeypatch):
    import zero_field_refusal.curvature as curvature

    on = rayleigh_config(tmp_path)
    off = scheduled(tmp_path, "train.steps=10", "schedule.evaluate_at=[5,10]", policy="short")
    disabled = rayleigh_config(tmp_path, "eval.rayleigh.enabled=false")
    assert "rayleigh" not in off.eval
    with_panel(env, on)
    scripted(9.0)
    loaded = curvature.load_stream_for
    runs = {}
    for name, cfg in (("off", off), ("disabled", disabled)):
        monkeypatch.setattr(curvature, "load_stream_for", None)  # no panel is read
        runs[name] = run_scheduled(cfg, env, run_id=name)
    monkeypatch.setattr(curvature, "load_stream_for", loaded)
    runs["on"] = run_scheduled(on, env, run_id="on")
    records = {name: comparable(run_dir) for name, run_dir in runs.items()}
    assert records["off"] == records["disabled"]
    rayleigh = records["on"].pop("rayleigh.json")
    assert rayleigh["R"] is not None
    assert records["on"]["summary.json"].pop("rayleigh")["R"] == rayleigh["R"]
    assert records["on"] == records["off"]  # the measurement changed nothing else
    for name in ("off", "disabled"):
        meta = json.loads((runs[name] / "meta.json").read_text())
        assert "curvature_panel" not in meta["inputs"]["streams"]


def test_the_run_releases_its_training_state_before_the_product(
    env, tmp_path, scripted, monkeypatch
) -> None:
    # The fit under the GPU's memory rests on it: no optimizer state (AdamW's moments), no
    # student and no panel caches may outlive the run into the product.
    import zero_field_refusal.curvature as curvature
    import zero_field_refusal.schedule as schedule

    cfg = rayleigh_config(tmp_path)
    with_panel(env, cfg)
    scripted(9.0)
    made: dict[str, list] = {"models": [], "optimizers": [], "panels": []}

    def tracking(factory, kind):
        def make(*args, **kwargs):
            made[kind].append(weakref.ref(thing := factory(*args, **kwargs)))
            return thing

        return make

    monkeypatch.setattr(env, "new_model", tracking(env.new_model, "models"))
    monkeypatch.setattr(schedule, "build_optimizer", tracking(schedule.build_optimizer,
                                                              "optimizers"))  # fmt: skip
    for name in ("Monitor", "ScoringDiagnostics"):
        monkeypatch.setattr(schedule, name, tracking(getattr(schedule, name), "panels"))
    real, alive = curvature.endpoint_rayleigh, {}

    def probe(model, *args, **kwargs):
        gc.collect()
        alive.update({kind: [ref() for ref in refs if ref() is not None]
                      for kind, refs in made.items()})  # fmt: skip
        alive["product_on"] = model
        return real(model, *args, **kwargs)

    monkeypatch.setattr(curvature, "endpoint_rayleigh", probe)
    run_scheduled(cfg, env, run_id="released")
    assert [len(made[kind]) for kind in ("models", "optimizers", "panels")] == [2, 1, 2]
    assert alive["optimizers"] == [] and alive["panels"] == []
    assert alive["models"] == [alive["product_on"]]  # the student is gone; the θ₀ copy remains
    assert alive["product_on"] is made["models"][1]()  # made second: the run's reference


def test_a_failed_measurement_is_recorded_and_raised(env, tmp_path, scripted, monkeypatch):
    import zero_field_refusal.curvature as curvature

    cfg = rayleigh_config(tmp_path)
    with_panel(env, cfg)
    scripted(9.0)

    def out_of_memory(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB\nmore detail")

    monkeypatch.setattr(curvature, "hessian_quadratic_form", out_of_memory)
    with pytest.raises(torch.OutOfMemoryError):
        run_scheduled(cfg, env, run_id="oom")
    run_dir = Path(os.environ["RUNS_DIR"]) / cfg.experiment / "oom"
    record = json.loads((run_dir / "rayleigh.json").read_text())
    reason = "failed: OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
    assert (record["R"], record["reason"]) == (None, reason)
    assert record["panel"]["count"] == 6 and record["seconds"] >= 0
    assert json.loads((run_dir / "summary.json").read_text())["rayleigh"]["reason"] == reason
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "failed"
    assert (run_dir / "terminal.json").is_file()  # the outcome, published before


def test_a_run_stopped_before_its_first_update_records_a_null_quotient(
    env, tmp_path, scripted, monkeypatch
) -> None:
    import zero_field_refusal.curvature as curvature
    import zero_field_refusal.schedule as schedule

    cfg = rayleigh_config(tmp_path)
    with_panel(env, cfg)
    scripted(9.0)
    real = schedule.accumulate_gradient

    def nan_loss(*args, **kwargs):
        return real(*args, **kwargs) | {"loss": float("nan")}

    monkeypatch.setattr(schedule, "accumulate_gradient", nan_loss)
    monkeypatch.setattr(curvature, "_micro_batch_products", no_product)
    run_dir = run_scheduled(cfg, env, run_id="no-step")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert (terminal["stop"]["reason"], terminal["steps"]) == ("nonfinite_loss", 0)
    record = json.loads((run_dir / "rayleigh.json").read_text())
    assert (record["R"], record["reason"], record["delta_norm"]) == (None, "zero displacement", 0.0)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["rayleigh"]["R"] is None


def test_a_run_whose_update_broke_the_weights_records_a_null_quotient(
    env, tmp_path, scripted, monkeypatch
) -> None:
    import zero_field_refusal.curvature as curvature
    import zero_field_refusal.schedule as schedule

    cfg = rayleigh_config(tmp_path)
    with_panel(env, cfg)
    scripted(9.0)
    real, calls = schedule._norm, {"weights": 0}

    def norm(tensors):
        first = next(t for t in tensors if t is not None)
        if first.requires_grad:  # the weights after an update: the fifth one breaks them
            calls["weights"] += 1
            if calls["weights"] == 5:
                with torch.no_grad():
                    first.view(-1)[0] = float("nan")
        return real(tensors)

    monkeypatch.setattr(schedule, "_norm", norm)
    monkeypatch.setattr(curvature, "_micro_batch_products", no_product)
    run_dir = run_scheduled(cfg, env, run_id="broken")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert (terminal["stop"]["reason"], terminal["steps"]) == ("nonfinite_weight", 5)
    record = json.loads((run_dir / "rayleigh.json").read_text())
    assert (record["R"], record["reason"], record["delta_norm"]) == (
        None, "non-finite displacement", None
    )  # fmt: skip


def test_a_run_without_its_panel_stops_before_it_trains(env, tmp_path, scripted) -> None:
    cfg = rayleigh_config(tmp_path)
    ready(env, cfg)  # no curvature panel
    scripted(9.0)
    with pytest.raises(StreamError, match="Missing stream .*curvature_panel"):
        run_scheduled(cfg, env, run_id="no-panel")
    run_dir = Path(os.environ["RUNS_DIR"]) / cfg.experiment / "no-panel"
    assert not (run_dir / "metrics.jsonl").exists() and not (run_dir / "terminal.json").exists()


@pytest.mark.parametrize(
    ("override", "message"),
    [("eval.rayleigh.responses=0", "eval.rayleigh.responses"),
     ("eval.rayleigh.enabled=1", "eval.rayleigh.enabled")],
)  # fmt: skip
def test_the_rayleigh_settings_are_checked_at_load(tmp_path, override, message) -> None:
    with pytest.raises(ProtocolError, match=message.replace(".", r"\.")):
        rayleigh_config(tmp_path, override)


def test_only_scheduled_runs_measure_the_rayleigh_quotient(tmp_path) -> None:
    experiment = tmp_path / "fixed.yaml"
    experiment.write_text("defaults: [experiments/smoke, eval/rayleigh]\n")
    with pytest.raises(ProtocolError, match="scheduled"):
        load_run_config(experiment, SMALL)
