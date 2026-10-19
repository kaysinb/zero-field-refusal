"""The first-step drift diagnostic (study 2, amendment 4, section 4): its frozen settings, the
four first updates against ``optim.Adam``'s first step bit for bit, the unbiased estimators on
synthetic draws, the cosines on hand values, the sequential per-optimizer computation on a tiny
model against a direct one that keeps all 64 updates, the host-memory rule, the script's records
with every harmful path forbidden, its driver, and (skipped without a GPU) the run through the
script on an unindexed CUDA device under a 1 GB memory rule."""

import ast
import dataclasses
import json
import math
import runpy
import subprocess
from pathlib import Path

import pytest
import torch
from conftest import tiny_model, tiny_tokenizer
from omegaconf import OmegaConf
from test_fa_control_diagnostic import forbid_outcomes
from test_train import APPENDABLE, SMALL, prompt_sets

from zero_field_refusal import drift
from zero_field_refusal import fa_control_diagnostic as fcd
from zero_field_refusal.config import load_config
from zero_field_refusal.models import freeze, install_neutral_generation_config, set_numerics
from zero_field_refusal.optim import Adam, estimate_second_moment
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import Environment, StreamPrefix, load_stream_for
from zero_field_refusal.teacher import stream_spec
from zero_field_refusal.tracking import CLEAN_CHECKOUT
from zero_field_refusal.train import accumulate_gradient

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "drift_diagnostic.py"
MODULE = ROOT / "src" / "zero_field_refusal" / "drift.py"
CPU = torch.device("cpu")
CUDA = torch.device("cuda")  # unindexed, as runtime.environment gives it on a Pod
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
EPS, BETA2 = 1e-8, 0.999
BATCH = 2  # the tiny runs' B
CHUNK = 100  # small pieces, so that the tiny model's reductions cross piece boundaries
COMMIT = "c" * 40
CLEAN = {"commit": COMMIT, "dirty": False}


def tracked():
    return load_config("experiments/drift_diagnostic")


def tiny_settings(experiment: Path, **changes):
    """The tracked settings on a tiny run config (tests/test_train.py's), with v₀ from 2 batches
    and no host-memory requirement."""
    run = {"experiment": str(experiment),
           "overrides": [*SMALL, *APPENDABLE, "teacher.chunk_size=12", "schedule.evaluate_at=[2]",
                         "optimizer=adam_frozen",
                         "optimizer.preconditioner.batches=2"]}  # fmt: skip
    return OmegaConf.merge(tracked(), {"drift": {"v0_batches": 2, "run": run,
                                                 "host_memory": {"sums_gb": 0, "margin_gb": 0}}},
                           {"drift": changes})  # fmt: skip


def tiny_experiment(base: Path) -> Path:
    path = base / "tiny_drift_run.yaml"
    path.write_text("defaults: [experiments/smoke, eval/study2, schedule/short]\n")
    return path


def make_env(base: Path, device: torch.device) -> Environment:
    """tests/test_train.py's environment on ``device``, with its data under ``base``."""
    tokenizer = tiny_tokenizer()
    end_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    sets = prompt_sets()

    def new_model():
        model = tiny_model(tied=False, head_std=0.02).to(device)
        install_neutral_generation_config(model, end_ids, tokenizer.pad_token_id)
        return model

    return Environment(tokenizer=tokenizer, end_ids=end_ids, pad_id=tokenizer.pad_token_id,
                       device=device, new_model=new_model, prompt_set=lambda name: sets[name],
                       data_root=base / "data")  # fmt: skip


# ------------------------------------------------------------------------------- settings


def test_the_settings_are_frozen() -> None:
    scope = drift.DriftScope.of(tracked())
    assert (scope.seed, scope.batches, scope.half, scope.optimizers) == (
        2300, 64, 32, ("sgd", "adamw", "fa", "prewarmed"))  # fmt: skip
    assert (scope.eps, scope.beta2, scope.v0_batches, scope.memory_rule_gb) == (1e-8, 0.999, 50, 46)
    assert scope.host_required_gb == 1.5 * 21 + 16 == 47.5
    cfg = drift.run_config(scope, 20)
    assert (cfg.seed, cfg.train.batch_size, cfg.train.loss.denominator) == (2300, 20, 20 * 96)
    assert (cfg.train.loss.kind, cfg.train.loss.normalization) == ("nll", "constant")
    pre = cfg.optimizer.preconditioner
    assert (pre.batches, pre.batch_size, pre.responses) == (50, 20, 1000)
    # The gradients are the regular runs' (any arm's, at B), and the streams are made as theirs:
    # the specs equal those of the regular runs and of the campaigns' stream config
    items = [{"id": "x", "text": "y"}]
    for arm in ([], ["optimizer=adam_frozen"], ["optimizer=adamw_prewarmed"]):
        regular = load_run_config("experiments/study2_short", [*arm, "train.batch_size=20",
                                                               "seed=2300"])  # fmt: skip
        assert OmegaConf.to_container(regular.train) == OmegaConf.to_container(cfg.train)
        assert regular.model == cfg.model and regular.execution == cfg.execution
        assert stream_spec(regular, "training", items) == stream_spec(cfg, "training", items)
        # ε and β₂ are each arm's own (FA has no β₂)
        assert regular.optimizer.eps == scope.eps
        if regular.optimizer.name != "adam_frozen":
            assert regular.optimizer.betas[1] == scope.beta2
    stage_a = load_run_config("experiments/study2_stage_a", ["optimizer=adam_frozen",
                                                             "train.batch_size=20"])  # fmt: skip
    for kind in ("training", "preconditioner"):
        assert stream_spec(stage_a, kind, items) == stream_spec(cfg, kind, items)
    # Departures are refused
    for changes in ({"halves": [[0, 30], [31, 63]]}, {"optimizers": ["sgd", "adam"]},
                    {"batches": 63}, {"beta2": 1.0}):  # fmt: skip
        with pytest.raises(drift.DriftError):
            drift.DriftScope.of(OmegaConf.merge(tracked(), {"drift": changes}))
    adamw = drift.DriftScope.of(OmegaConf.merge(tracked(), {"drift": {"run": {"overrides": []}}}))
    with pytest.raises(drift.DriftError, match="preconditioner"):
        drift.run_config(adamw, 20)


# --------------------------------------------------------------------------- first updates


@pytest.mark.parametrize("device", [CPU, pytest.param(CUDA, marks=needs_cuda)])
def test_first_updates_match_optim_first_step(device) -> None:
    """Each first update is ``optim.Adam``'s first step at lr = 1 and β₁ = 0 (m̂₁ = g) in its
    mode, bit for bit, negated; SGD's is the gradient. On a toy parameter whose gradient spans 30
    decades (g² underflows at the small end) with exact zeros, and a v₀ with a zero."""
    rng = torch.Generator().manual_seed(1)
    shape = (6, 5)
    scales = torch.logspace(-30, 0, 30).reshape(shape)
    grad = torch.randn(shape, generator=rng) * scales
    grad[0, 0], grad[1, 1] = 0.0, -0.0
    v0 = torch.rand(shape, generator=rng) * torch.logspace(-20, -2, 30).reshape(shape)
    v0[2, 2] = 0.0
    grad, v0 = grad.to(device), v0.to(device)
    before = (grad.clone(), v0.clone())
    updates = drift.first_updates(grad, v0, EPS, BETA2)
    assert list(updates) == list(drift.OPTIMIZERS)
    assert torch.equal(updates["sgd"], grad) and updates["sgd"] is not grad
    for name, mode in (("adamw", "adaptive"), ("fa", "frozen"), ("prewarmed", "prewarmed")):
        p = torch.nn.Parameter(torch.zeros(shape, device=device))
        optimizer = Adam([p], lr=1.0, betas=(0.0, BETA2), eps=EPS, second_moment=mode,
                         initial_second_moment=None if mode == "adaptive" else [v0])  # fmt: skip
        p.grad = grad.clone()
        optimizer.step()
        assert torch.equal(optimizer.state[p]["exp_avg"], grad), name  # m̂₁ = m₁ = g
        assert torch.equal(p.detach(), -updates[name]), name
    assert torch.equal(grad, before[0]) and torch.equal(v0, before[1])  # inputs untouched
    # The definitions, in float64: equal to fp32 rounding
    g, v = grad.double(), v0.double()
    literal = {"adamw": g / (g.square().sqrt() + EPS), "fa": g / (v.sqrt() + EPS),
               "prewarmed": g / ((BETA2 * v + (1 - BETA2) * g.square()).sqrt() + EPS)}  # fmt: skip
    for name, expected in literal.items():
        torch.testing.assert_close(updates[name].double(), expected, rtol=1e-6, atol=0)
    assert (updates["adamw"].abs() <= 1 + 1e-6).all()  # approximately sign-normalized
    with pytest.raises(drift.DriftError, match="needs v₀"):
        drift.first_update("fa", grad, None, eps=EPS, beta2=BETA2)


# ------------------------------------------------------------------------------ statistics


def stats_of(u: torch.Tensor, n_half: int) -> dict:
    """``drift_statistics`` of the draws ``u`` (n × d), from their halves' sums and Q."""
    a, b = u[:n_half], u[n_half:]
    return drift.drift_statistics(([a.sum(0)], [b.sum(0)]),
                                  (a.square().sum().item(), b.square().sum().item()),
                                  n_half)  # fmt: skip


def test_unbiased_estimators() -> None:
    """On draws with a known mean μ and covariance Σ (n = 64, 4000 repetitions), the means of
    ``mean_norm2`` and ``trace_cov`` match ‖μ‖² and tr Σ within the Monte-Carlo error; with
    μ = 0, negative ``mean_norm2`` estimates occur and are kept as the formula gives them."""
    rng = torch.Generator().manual_seed(0)
    n, reps = 64, 4000
    mu = torch.tensor([0.3, -0.2, 0.1, 0.0, 0.5, -0.4], dtype=torch.float64)
    root = torch.tril(torch.randn(6, 6, generator=rng, dtype=torch.float64)) * 0.5  # Σ = LLᵀ
    trace = root.square().sum().item()  # tr Σ
    noise = torch.randn(reps, n, 6, generator=rng, dtype=torch.float64) @ root.T
    for mean, truth in ((mu, mu.square().sum().item()), (torch.zeros(6, dtype=torch.float64), 0.0)):
        draws = [stats_of(mean + noise[r], n // 2) for r in range(reps)]
        assert all(s["n"] == n for s in draws)
        for key, expected in (("mean_norm2", truth), ("trace_cov", trace)):
            values = torch.tensor([s[key] for s in draws], dtype=torch.float64)
            error = values.std().item() / math.sqrt(reps)
            assert abs(values.mean().item() - expected) < 4 * error, (key, truth)
    negative = [(r, s) for r, s in enumerate(draws) if s["mean_norm2"] < 0]
    assert len(negative) > reps // 10  # with μ = 0 about half are negative
    r, s = negative[0]
    u = noise[r]
    by_formula = (u.sum(0).square().sum().item() - u.square().sum().item()) / (n * (n - 1))
    assert s["mean_norm2"] == pytest.approx(by_formula, rel=1e-12) and by_formula < 0
    assert s["ratio"] == s["mean_norm2"] / s["trace_cov"] < 0  # kept, not clipped


def test_half_cosine_and_cross_cosines() -> None:
    """Hand values on tiny vectors split into two parameters, with one-coordinate pieces."""
    t = torch.tensor
    s_a, s_b = [t([1.0, 0.0]), t([0.0])], [t([1.0, 1.0]), t([0.0])]
    for chunk in (drift.REDUCTION_CHUNK, 1):
        stats = drift.drift_statistics((s_a, s_b), (1.0, 2.0), 2, chunk)
        # S = (2, 1, 0): ‖S‖² = 5, Q = 3, n = 4
        assert (stats["n"], stats["sum_norm2"], stats["Q"]) == (4, 5.0, 3.0)
        assert stats["mean_norm2"] == pytest.approx((5 - 3) / 12)
        assert stats["trace_cov"] == pytest.approx((3 - 5 / 4) / 3)
        assert stats["ratio"] == pytest.approx(2 / 7)
        assert stats["half_cosine"] == pytest.approx(1 / math.sqrt(2))
        assert stats["halves"] == [{"sum_norm2": 1.0, "Q": 1.0}, {"sum_norm2": 2.0, "Q": 2.0}]
    # Opposite halves: a negative estimate of ‖E u‖², kept
    stats = drift.drift_statistics(([t([1.0, 0.0])], [t([-1.0, 0.0])]), (1.0, 1.0), 1)
    assert (stats["mean_norm2"], stats["trace_cov"], stats["half_cosine"]) == (-1.0, 2.0, -1.0)
    zero = drift.drift_statistics(([t([0.0])], [t([1.0])]), (0.0, 1.0), 1)
    assert zero["half_cosine"] is None
    sums = {"sgd": [t([1.0, 0.0]), t([0.0])], "adamw": [t([1.0, 1.0]), t([0.0])],
            "fa": [t([0.0, 0.0]), t([-2.0])], "prewarmed": [t([-1.0, 0.0]), t([0.0])]}  # fmt: skip
    for chunk in (drift.REDUCTION_CHUNK, 1):
        cross = drift.cross_cosines(sums, chunk)
        expected = {"sgd|adamw": 1 / math.sqrt(2), "sgd|fa": 0.0, "sgd|prewarmed": -1.0,
                    "adamw|fa": 0.0, "adamw|prewarmed": -1 / math.sqrt(2),
                    "fa|prewarmed": 0.0}  # fmt: skip
        assert list(cross["cosines"]) == list(expected)
        for pair, value in expected.items():
            assert cross["cosines"][pair] == pytest.approx(value, abs=1e-15), pair
        assert cross["sum_norm2"] == {"sgd": 1.0, "adamw": 2.0, "fa": 4.0, "prewarmed": 1.0}
    assert drift.cross_cosines({"a": [t([0.0])], "b": [t([1.0])]})["cosines"] == {"a|b": None}


# ------------------------------------------------------------------- the run on a tiny model


@pytest.fixture(scope="module")
def tiny_world(tmp_path_factory):
    """The streams of the drift seed at B 2 on the CPU, and the diagnostic computed from them."""
    base = tmp_path_factory.mktemp("drift")
    env = make_env(base, CPU)
    settings = tiny_settings(tiny_experiment(base))
    scope = drift.DriftScope.of(settings)
    cfg = drift.run_config(scope, BATCH)
    state = drift.prepare_streams(cfg, env, scope.batches * BATCH)
    assert state["drift"]["ready"] and state["preconditioner"]["ready"] and state["generated"]
    again = drift.prepare_streams(cfg, env, scope.batches * BATCH)  # located, kept
    assert again["generated"] == [] and again["drift"] == state["drift"]
    result = drift.run_drift(settings, cfg, env, chunk=CHUNK)
    return env, settings, cfg, result


def direct(env, cfg, scope) -> tuple[dict, list, str]:
    """All 64 updates of every optimizer kept in float64 (flattened over the trainable
    parameters), v₀ by ``optim.estimate_second_moment`` on the v₀ stream's batches, and the
    statistics from their definitions."""
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    names, params = [n for n, _ in trainable], [p for _, p in trainable]
    stream, _ = load_stream_for(cfg, env, "preconditioner", scope.seed)
    pre = [stream[i * BATCH : (i + 1) * BATCH] for i in range(scope.v0_batches)]
    v0 = estimate_second_moment(params, lambda i: accumulate_gradient(model, pre[i], cfg, env),
                                scope.v0_batches)  # fmt: skip
    v0_sha256 = fcd.tensors_sha256(list(zip(names, v0, strict=True)))
    flat_v0 = torch.cat([v.reshape(-1) for v in v0])
    samples = StreamPrefix(cfg, env, scope.seed).take(0, scope.batches * BATCH)
    updates = {name: [] for name in drift.OPTIMIZERS}
    rows = []
    for i in range(scope.batches):
        for p in params:
            p.grad = None
        stats = accumulate_gradient(model, samples[i * BATCH : (i + 1) * BATCH], cfg, env)
        g = torch.cat([p.grad.reshape(-1) for p in params])
        rows.append({"loss": stats["loss"], "grad_norm2": g.double().square().sum().item()})
        for name, u in drift.first_updates(g, flat_v0, EPS, BETA2).items():
            updates[name].append(u.double())
    out, n, half = {}, scope.batches, scope.half
    for name, us in updates.items():
        u = torch.stack(us)  # n × d
        mean = u.mean(0)
        trace_cov = (u - mean).square().sum().item() / (n - 1)
        a, b = u[:half].sum(0), u[half:].sum(0)
        out[name] = {"mean_norm2": mean.square().sum().item() - trace_cov / n,
                     "trace_cov": trace_cov,
                     "half_cosine": (a @ b / (a.norm() * b.norm())).item(),
                     "S": u.sum(0), "Q": u.square().sum().item()}  # fmt: skip
    return out, rows, v0_sha256


def test_sequential_processing_matches_all_at_once(tiny_world) -> None:
    """The per-optimizer sequential computation (fp32 half-sums, float64 reductions in pieces,
    one optimizer at a time, gradients recomputed) equals the direct one that keeps all 64
    updates, to fp32 accumulation tolerance; v₀ is the same tensor bit for bit."""
    env, settings, cfg, result = tiny_world
    scope = drift.DriftScope.of(settings)
    expected, rows, v0_sha256 = direct(env, cfg, scope)
    assert result["v0_sha256"] == v0_sha256
    assert (result["batch"], result["seed"], result["n"]) == (BATCH, 2300, 64)
    assert result["halves"] == [[0, 31], [32, 63]]
    assert [r["batch"] for r in result["batches"]] == list(range(64))
    assert [r["half"] for r in result["batches"]] == [0] * 32 + [1] * 32
    assert [r["loss"] for r in result["batches"]] == [r["loss"] for r in rows]
    for got, want in zip(result["batches"], rows, strict=True):
        assert got["grad_norm2"] == pytest.approx(want["grad_norm2"], rel=1e-12)
    assert result["numerics"]["gradients_identical"] is True  # four passes, the same gradients
    assert result["numerics"]["gradient_differences"] == {
        key: {"max_relative_difference": 0.0, "at": None} for key in ("loss", "grad_norm2")
    }
    n = 64
    for name, want in expected.items():
        got = result["optimizers"][name]
        assert got["n"] == n
        assert got["Q"] == pytest.approx(want["Q"], rel=1e-12)  # float64 throughout
        s2 = want["S"].square().sum().item()
        assert got["sum_norm2"] == pytest.approx(s2, rel=1e-5)
        mean_scale = (s2 + want["Q"]) / (n * (n - 1))  # the scale of the cancellation
        assert abs(got["mean_norm2"] - want["mean_norm2"]) <= 1e-5 * mean_scale, name
        trace_scale = (want["Q"] + s2 / n) / (n - 1)
        assert abs(got["trace_cov"] - want["trace_cov"]) <= 1e-5 * trace_scale, name
        assert got["half_cosine"] == pytest.approx(want["half_cosine"], abs=1e-5), name
        assert got["ratio"] == got["mean_norm2"] / got["trace_cov"]
        assert abs(result["numerics"]["sum_rounding"][name]) < 1e-5  # the fp32 sum's ‖S‖²
    for pair, value in result["cross_cosines"].items():
        x, y = (expected[name]["S"] for name in pair.split("|"))
        assert value == pytest.approx((x @ y / (x.norm() * y.norm())).item(), abs=1e-5), pair
    assert len(result["cross_cosines"]) == 6
    # SGD's update is the gradient: its Q is Σ ‖g_i‖²
    sgd_q = sum(r["grad_norm2"] for r in result["batches"])
    assert result["optimizers"]["sgd"]["Q"] == pytest.approx(sgd_q, rel=1e-12)
    # The record: statistics, peaks, numerics, provenance with the GPU code, stream identities
    assert set(result["streams"]) == {"drift", "preconditioner"}
    assert result["streams"]["drift"]["count"] == 64 * BATCH
    assert result["streams"]["preconditioner"]["count"] == 2 * BATCH
    assert set(result["peaks"]) == {"v0", *drift.OPTIMIZERS, "all", "host_peak_rss_gb"}
    assert result["numerics"]["reductions"] == "float64"
    assert {"spec", "code", "gpu_code", "settings", "run_config"} <= set(result["provenance"])
    assert result["parameters"]["coordinates"] == sum(
        p.numel() for _, p in freeze(env.new_model(), cfg.train.freeze)
    )
    json.dumps(result)  # writable as JSON


def test_gradient_agreement() -> None:
    """The largest relative difference of the losses and ‖g_i‖² from the first pass's, and
    where it occurs."""
    rows = [{"batch": i, "half": 0, "loss": 1.0 + i, "grad_norm2": 4.0} for i in range(3)]
    same = drift.gradient_agreement({"sgd": rows, "adamw": [dict(r) for r in rows]})
    assert same == {"identical": True,
                    "loss": {"max_relative_difference": 0.0, "at": None},
                    "grad_norm2": {"max_relative_difference": 0.0, "at": None}}  # fmt: skip
    other = [dict(r) for r in rows]
    other[2]["grad_norm2"] = 4.0 * (1 + 1e-7)
    worse = [dict(r) for r in rows]
    worse[1]["grad_norm2"], worse[0]["loss"] = 5.0, 1.5
    found = drift.gradient_agreement({"sgd": rows, "adamw": other, "fa": worse})
    assert found["identical"] is False
    assert found["grad_norm2"]["max_relative_difference"] == pytest.approx(0.2)
    assert found["grad_norm2"]["at"] == {"batch": 1, "optimizer": "fa"}
    assert found["loss"] == {"max_relative_difference": pytest.approx(1 / 3),
                             "at": {"batch": 0, "optimizer": "fa"}}  # fmt: skip


def test_a_non_finite_gradient_is_a_failure(tiny_world, monkeypatch) -> None:
    """A non-finite ‖g_i‖² or loss stops the run with ``DriftError`` (the script records it as
    a failure), rather than writing NaN statistics."""
    env, settings, cfg, _ = tiny_world
    original, seen = drift.accumulate_gradient, []

    def broken(kind, model, batch, cfg, env, *args):
        stats = original(model, batch, cfg, env, *args)
        seen.append(1)
        if len(seen) == 3 and kind == "grad":  # the third batch of the first pass
            grad = next(p.grad for p in model.parameters() if p.grad is not None)
            grad.view(-1)[0] = math.inf
        elif len(seen) == 3:
            stats["loss"] = math.nan
        return stats

    for kind in ("grad", "loss"):
        seen.clear()
        monkeypatch.setattr(drift, "accumulate_gradient",
                            lambda *args, kind=kind: broken(kind, *args))  # fmt: skip
        with pytest.raises(drift.DriftError, match="sgd, batch 2: .* is not finite"):
            drift.run_drift(settings, cfg, env, chunk=CHUNK)


# -------------------------------------------------------------------------- host memory


def test_host_memory_rule(tmp_path) -> None:
    """The free host memory is MemAvailable, or the container's limit less its working set when
    a cgroup (v2 or v1) limits it, whichever is smaller; unknown is not sufficient."""
    scope = drift.DriftScope.of(tracked())
    meminfo, cgroup = tmp_path / "meminfo", tmp_path / "cgroup"
    cgroup.mkdir()
    meminfo.write_text("MemTotal:       100000000 kB\nMemAvailable:    60000000 kB\n")

    def check():
        return drift.host_memory_check(scope, meminfo=meminfo, cgroup=cgroup)

    found = check()
    assert found["available_gb"] == pytest.approx(61.44) and found["cgroup_gb"] is None
    assert found["sufficient"] and found["required_gb"] == 47.5
    (cgroup / "memory.max").write_text("50000000000\n")  # v2: 50 GB, 10 used, 4 of it cache
    (cgroup / "memory.current").write_text("10000000000\n")
    (cgroup / "memory.stat").write_text("anon 6000000000\ninactive_file 4000000000\n")
    found = check()
    assert found["available_gb"] == pytest.approx(44.0) and not found["sufficient"]
    (cgroup / "memory.max").write_text("max\n")  # no limit
    assert check()["cgroup_gb"] is None and check()["sufficient"]
    for name in ("memory.max", "memory.current", "memory.stat"):
        (cgroup / name).unlink()
    (cgroup / "memory").mkdir()  # v1: 64 GB, 20 used, 2 of it cache
    (cgroup / "memory" / "memory.limit_in_bytes").write_text("64000000000\n")
    (cgroup / "memory" / "memory.usage_in_bytes").write_text("20000000000\n")
    (cgroup / "memory" / "memory.stat").write_text("total_inactive_file 2000000000\n")
    found = check()
    assert found["available_gb"] == pytest.approx(46.0) and not found["sufficient"]
    meminfo.unlink()
    (cgroup / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")  # none
    assert check()["available_gb"] is None and not check()["sufficient"]


# ------------------------------------------------------------------------------- the script


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


def script_world(base: Path, monkeypatch, device: torch.device, **changes):
    """A tiny environment on ``device`` with every outcome path forbidden, given to the script
    as ``runtime.environment``; the settings file; and the argv of a part. HEAD is a clean
    checkout of ``COMMIT``."""
    env = make_env(base, device)
    guarded, requested = forbid_outcomes(monkeypatch, env)
    devices = []

    def environment(cfg, device):
        devices.append(device)
        return dataclasses.replace(guarded, device=torch.device(device))

    monkeypatch.setattr("zero_field_refusal.runtime.environment", environment)
    monkeypatch.setattr(drift, "git_state", lambda *a, **k: dict(CLEAN))
    monkeypatch.setattr(
        drift, "gpu_code_identity", lambda *a, **k: {"commit": COMMIT, "files": 7, "sha256": "same"}
    )
    config = base / "settings.yaml"
    OmegaConf.save(tiny_settings(tiny_experiment(base), **changes), config)
    root = base / "runs"

    def argv(part, *extra):
        return [str(SCRIPT), part, "--batch", str(BATCH), "--config", str(config),
                "--runs-root", str(root), "--device", device.type, *extra]  # fmt: skip

    return argv, requested, devices, drift.output_root(BATCH, root)


def test_the_module_names_no_outcome() -> None:
    """Outside docstrings, nothing in the module refers to harmful prompts, refusal, judges,
    greedy generation or the evaluator."""
    tree = ast.parse(MODULE.read_text())
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef)
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant)}  # fmt: skip
    words = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            words.append(node.id)
        elif isinstance(node, ast.Attribute):
            words.append(node.attr)
        elif isinstance(node, ast.alias):
            words.append(node.name)
        elif isinstance(node, ast.ImportFrom):
            words.append(node.module or "")
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and id(node) not in docstrings):  # fmt: skip
            words.append(node.value)
    for banned in ("harm", "refusal", "judge", "greedy", "Evaluator"):
        assert not [w for w in words if banned in w], banned


def test_no_harm_paths(tmp_path, monkeypatch, capsys) -> None:
    """prepare, plan and run through the script on a tiny CPU model, with every harmful prompt
    set, the evaluator, greedy generation and refusal scoring made to raise: they succeed, no
    harmful set is asked for, and the script's own guard refuses one."""
    argv, requested, devices, out = script_world(tmp_path, monkeypatch, CPU)
    assert call(monkeypatch, argv("prepare", "--dry-run")) == 0
    assert "MISSING: drift stream" in capsys.readouterr().out
    assert "tracked settings" in str(call(monkeypatch, argv("prepare")))  # it writes DATA_DIR
    assert call(monkeypatch, argv("prepare"), untracked=True) == 0
    assert "generated:" in capsys.readouterr().out
    assert call(monkeypatch, argv("plan")) == 0
    printed = capsys.readouterr().out
    assert "MISSING" not in printed and "sufficient" in printed and "cost ≈" in printed
    assert call(monkeypatch, argv("run", "--dry-run")) == 0
    assert not out.exists()  # a dry run writes nothing
    assert call(monkeypatch, argv("run"), untracked=True) == 0
    result = json.loads((out / drift.RESULT).read_text())
    assert set(result["optimizers"]) == set(drift.OPTIMIZERS)
    assert result["numerics"]["gradients_identical"] is True
    assert result["provenance"]["gpu_code"]["sha256"] == "same"
    assert result["numerics"]["memory_rule"]["limit_gb"] == 46.0  # recorded only on the CPU
    assert requested and not [name for name in requested if "harmful" in name]
    namespace = runpy.run_path(str(SCRIPT))
    with pytest.raises(RuntimeError, match="must not read the prompt set 'harmful_kl'"):
        namespace["guard_prompt_sets"](make_env(tmp_path, CPU)).prompt_set("harmful_kl")


def test_the_run_keeps_counted_results_and_records_failures(tmp_path, monkeypatch,
                                                            capsys) -> None:  # fmt: skip
    """``run`` through the script, with the computation stubbed: a result of this B written with
    this GPU code is kept; one of another B or other GPU code is set aside and redone; gradients
    that differ between the passes give a WARNING; a dirty checkout, untracked settings and too
    little host memory run nothing; a failure, an out-of-memory error with exit 4 included, is
    recorded before the exit and renamed by a later success."""
    argv, _, _, out = script_world(tmp_path, monkeypatch, CPU)
    outcomes = [
        "ok",
        "differing",
        RuntimeError("a failure"),
        torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB"),
        "ok",
    ]
    calls = []

    def fake_run(settings, cfg, env, **kwargs):
        calls.append(outcome := outcomes[len(calls)])
        if isinstance(outcome, BaseException):
            raise outcome
        differences = {
            "loss": {"max_relative_difference": 0.0, "at": None},
            "grad_norm2": {"max_relative_difference": 1e-7, "at": {"batch": 3, "optimizer": "fa"}},
        }
        return {"batch": int(cfg.train.batch_size),
                "optimizers": {"sgd": {"mean_norm2": 1.0, "trace_cov": 2.0, "ratio": 0.5,
                                       "half_cosine": 0.1}},
                "cross_cosines": {},
                "numerics": {"gradients_identical": outcome == "ok",
                             "gradient_differences": differences},
                "peaks": {"all": {}},
                "provenance": {"code": drift.git_state(),
                               "gpu_code": drift.gpu_code_identity()}}  # fmt: skip

    def stale():
        return [json.loads(p.read_text()) for p in (out / "stale").glob("*/result.json")]

    monkeypatch.setattr(drift, "run_drift", fake_run)
    assert call(monkeypatch, argv("run"), untracked=True) == 0
    assert call(monkeypatch, argv("run"), untracked=True) == 0  # kept
    assert "kept:" in capsys.readouterr().out and len(calls) == 1
    assert "tracked settings" in str(call(monkeypatch, argv("run")))
    # A result of another B (an --out shared between batches, say) does not count
    record = json.loads((out / drift.RESULT).read_text())
    (out / drift.RESULT).write_text(json.dumps({**record, "batch": BATCH + 1}))
    assert call(monkeypatch, argv("run"), untracked=True) == 0
    assert [r["batch"] for r in stale()] == [BATCH + 1] and len(calls) == 2
    printed = capsys.readouterr()
    assert "WARNING: the four passes did not see identical gradients" in printed.err
    assert '"batch": 3' in printed.err and "1e-07" in printed.err
    assert call(monkeypatch, argv("run"), untracked=True) == 0  # kept: this B, this GPU code
    assert len(calls) == 2
    monkeypatch.setattr(
        drift,
        "gpu_code_identity",
        lambda *a, **k: {"commit": COMMIT, "files": 7, "sha256": "other"},
    )
    with pytest.raises(RuntimeError, match="a failure"):  # other GPU code: redone, and it fails
        call(monkeypatch, argv("run"), untracked=True)
    failure = json.loads((out / "failed.json").read_text())
    assert failure["part"] == "run" and "a failure" in failure["error"]
    assert failure["provenance"]["gpu_code"]["sha256"] == "other"
    assert stale() and not (out / drift.RESULT).exists()
    capsys.readouterr()
    assert call(monkeypatch, argv("run"), untracked=True) == 4  # out of memory under the rule
    assert "out of memory under the 46 GB rule" in capsys.readouterr().err
    failure = json.loads((out / "failed.json").read_text())
    assert failure["error"].startswith("OutOfMemoryError: CUDA out of memory")
    assert "peaks" in failure and "traceback" in failure
    assert not (out / drift.RESULT).exists()
    assert call(monkeypatch, argv("run"), untracked=True) == 0
    assert (out / drift.RESULT).exists() and not (out / "failed.json").exists()
    assert len(list(out.glob("failed-before-*.json"))) == 1
    (out / drift.RESULT).unlink()
    monkeypatch.setattr(drift, "git_state", lambda *a, **k: {**CLEAN, "dirty": True})
    assert "not clean" in str(call(monkeypatch, argv("run"), untracked=True))
    monkeypatch.setattr(drift, "git_state", lambda *a, **k: dict(CLEAN))
    short = {"available_gb": None, "meminfo_gb": None, "cgroup_gb": None}  # unknown
    monkeypatch.setattr(drift, "host_memory", lambda **paths: dict(short))
    assert call(monkeypatch, argv("run"), untracked=True) == 5
    assert call(monkeypatch, argv("plan")) == 1
    assert "INSUFFICIENT" in capsys.readouterr().out
    failure = json.loads((out / "failed.json").read_text())
    assert failure["host_memory"]["sufficient"] is False and len(calls) == 5


def test_the_driver(tmp_path, monkeypatch) -> None:
    """The Pod's driver: written once from the tracked settings, valid bash, with the
    operational checks in order before the plan, the streams and the run."""
    path = tmp_path / "drift-b20.sh"
    base = [str(SCRIPT), "driver", "--batch", "20"]
    assert call(monkeypatch, [*base, "--commit", COMMIT, "--out", str(path)]) == 0
    text = path.read_text()
    subprocess.run(["bash", "-n", str(path)], check=True)
    assert f"SRC=/workspace/code/zero-field-refusal/{COMMIT}/source" in text
    assert 'OUT="${RUNS_DIR:?}/diagnostics/drift/b20"' in text and "tee -a" in text
    order = [CLEAN_CHECKOUT, "--query-compute-apps", "memory.used", "df -Pk",
             "step plan", "step prepare", "step run"]  # fmt: skip
    positions = [text.index(item) for item in order]
    assert positions == sorted(positions)
    assert "trap 'stop" in text and "failed.json" in text
    assert "drift_diagnostic.py run --batch 20" in text
    assert call(monkeypatch, [*base, "--commit", COMMIT, "--out", str(path)]) != 0  # it exists
    # stop() with a broken project-run (no checkout at $SRC): bash writes a minimal record, valid
    # JSON whatever the reason holds; a record already there is left as it is
    begin = text.index("stop() {")
    stop = tmp_path / "stop.sh"
    stop.write_text(text[begin : text.index("\n}\n", begin) + 3])
    out = tmp_path / "out"
    out.mkdir()
    program = ('set -euo pipefail; project-run() { return 127; }; OUT="$1"; SRC=/nonexistent; '
               '. "$2"; stop "$3" 3')  # fmt: skip
    reason = 'run failed: "quoted" \\ and a backslash (exit 3)'
    done = subprocess.run(["bash", "-c", program, "bash", str(out), str(stop), reason],
                          capture_output=True, text=True)  # fmt: skip
    assert done.returncode == 3 and reason in done.stderr
    record = json.loads((out / "failed.json").read_text())
    assert record == {"part": "driver", "reason": "run failed: quoted  and a backslash (exit 3)",
                      "exit_code": 3}  # fmt: skip
    (out / "failed.json").write_text('{"part": "run"}\n')  # the failing step's own record
    done = subprocess.run(["bash", "-c", program, "bash", str(out), str(stop), "other"])
    assert done.returncode == 3 and (out / "failed.json").read_text() == '{"part": "run"}\n'


# ------------------------------------------------------------------------------------- CUDA


@pytest.fixture
def cuda_rule():
    """Lifts the memory rule after a test: its CUDA fraction holds for the whole process."""
    yield
    torch.cuda.set_per_process_memory_fraction(1.0, fcd.cuda_index(CUDA))
    fcd._MEMORY_RULE.clear()


@needs_cuda
def test_cuda_run(tmp_path, monkeypatch, cuda_rule, capsys) -> None:
    """The Pod's path on a GPU: prepare and run through the script on ``torch.device("cuda")``
    without an index, under a 1 GB memory rule; the peaks are recorded within it."""
    argv, requested, devices, out = script_world(tmp_path, monkeypatch, CUDA, memory_rule_gb=1)
    set_numerics(OmegaConf.create({"tf32": False}))  # TF32 off, as in every run
    assert call(monkeypatch, argv("prepare"), untracked=True) == 0
    assert call(monkeypatch, argv("run"), untracked=True) == 0
    assert devices == ["cuda", "cuda"]
    index = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    rules = [json.loads(line)["memory_rule"] for line in capsys.readouterr().out.splitlines()
             if line.startswith('{"memory_rule"')]  # fmt: skip
    assert rules == [{"limit_gb": 1.0, "fraction": 1e9 / total, "total_gb": round(total / 1e9, 3),
                      "device_index": index}]  # fmt: skip
    result = json.loads((out / drift.RESULT).read_text())
    assert result["numerics"]["device"] == "cuda"
    assert result["numerics"]["memory_rule"]["limit_gb"] == 1.0
    assert result["peaks"]["all"]["within_memory_rule"] is True
    for phase in ("v0", *drift.OPTIMIZERS):
        assert 0 < result["peaks"][phase]["peak_reserved_gb"] <= 1.0
    for stats in result["optimizers"].values():
        assert all(math.isfinite(stats[key]) for key in ("mean_norm2", "trace_cov", "Q"))
        assert stats["trace_cov"] > 0 and -1 <= stats["half_cosine"] <= 1
    assert all(-1 <= c <= 1 for c in result["cross_cosines"].values())
    assert requested and not [name for name in requested if "harmful" in name]
