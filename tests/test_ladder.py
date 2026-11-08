"""Study 5's ladder of synthetic displacements on CPU (docs/study5-preregistration.md,
"Descriptive measurements"): the rungs' directions, the scale search on synthetic K(c), and the
ladder script end to end on a tiny scheduled F run with a stubbed K."""

from __future__ import annotations

import json
import math

import pytest
import torch
from omegaconf import OmegaConf
from test_schedule import scheduled
from test_study3_campaigns import script_module
from test_study5_options import F_RUN, ready
from test_train import APPENDABLE, SMALL, read_jsonl
from test_train import env as env  # noqa: F401  (the fixture)

from zero_field_refusal import ladder, linear_response
from zero_field_refusal.analysis.study3_search import CANDIDATES
from zero_field_refusal.evaluation import LADDER_RUN_TYPE, ladder_condition
from zero_field_refusal.models import freeze
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.run_judging import check_responses, response_conditions, unlabelable
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.seeding import derive_seed
from zero_field_refusal.train import Evaluator

SHAPES = {"b.weight": (4, 6), "a.bias": (10,), "c.weight": (2, 3, 2)}


def named(generator: torch.Generator, positive: bool = False) -> dict[str, torch.Tensor]:
    return {k: (torch.rand(s, generator=generator) + 0.5 if positive
                else torch.randn(s, generator=generator)) for k, s in SHAPES.items()}  # fmt: skip


def clone(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in tensors.items()}


# --------------------------------------------------------------------------------- the directions


def test_gaussian_d_covariance_is_D() -> None:
    """δ = D^{1/2} ⊙ z: per entry mean 0 and variance D (empirical variance ratio ≈ 1 at every
    level of D), z the N(0, I) draws of one CPU generator seeded by
    derive_seed(seed, "study5", "gaussian_d"), tensors in sorted name order, fp32."""
    levels = [0.25, 1.0, 4.0, 9.0]
    big = torch.tensor(levels).repeat_interleave(5000).reshape(40, 500)  # 5000 entries a level
    D = {"z.big": big, "a.small": torch.full((3000,), 2.0)}
    delta = ladder.gaussian_d_direction(clone(D), 50000)
    assert list(delta) == list(D) and all(t.dtype == torch.float32 for t in delta.values())
    for level in levels:
        values = delta["z.big"][big == level].double()
        assert (values.square().mean() / level).item() == pytest.approx(1.0, abs=0.08), level
        assert abs(values.mean().item()) < 4 * math.sqrt(level / values.numel())
    small = delta["a.small"].double()
    assert (small.square().mean() / 2.0).item() == pytest.approx(1.0, abs=0.1)
    # exactly D^{1/2} ⊙ z from the named stream
    generator = torch.Generator().manual_seed(derive_seed(50000, "study5", "gaussian_d"))
    z = {name: torch.randn(tuple(D[name].shape), generator=generator, dtype=torch.float32)
         for name in sorted(D)}  # fmt: skip
    for name in D:
        assert torch.equal(delta[name], D[name].sqrt() * z[name]), name
    # independent of the mapping's order; another seed draws another z
    again = ladder.gaussian_d_direction({k: clone(D)[k] for k in reversed(list(D))}, 50000)
    assert all(torch.equal(again[k], delta[k]) for k in D)
    other = ladder.gaussian_d_direction(clone(D), 50001)
    assert not torch.equal(other["z.big"], delta["z.big"])


def test_shuffle_keeps_tensor_multiset() -> None:
    """π(D): each tensor's entries permuted within the tensor (same shape, same multiset of
    values, the values of no other tensor), by randperm of one CPU generator seeded by
    derive_seed(seed, "study5", "shuffle"), tensors in sorted name order."""
    D = {"b.weight": torch.arange(24.0).reshape(4, 6) + 1,
         "a.bias": torch.arange(10.0) + 100}  # fmt: skip
    shuffled = ladder.shuffle_within_tensors(clone(D), 50000)
    for name, d in D.items():
        s = shuffled[name]
        assert s.shape == d.shape and s.dtype == d.dtype
        assert torch.equal(s.reshape(-1).sort().values, d.reshape(-1).sort().values), name
        assert not torch.equal(s, d), name
    generator = torch.Generator().manual_seed(derive_seed(50000, "study5", "shuffle"))
    for name in sorted(D):
        perm = torch.randperm(D[name].numel(), generator=generator)
        assert torch.equal(shuffled[name].reshape(-1), D[name].reshape(-1)[perm]), name
    again = ladder.shuffle_within_tensors(clone(D), 50000)
    assert all(torch.equal(again[k], shuffled[k]) for k in D)
    other = ladder.shuffle_within_tensors(clone(D), 50001)
    assert not torch.equal(other["b.weight"], shuffled["b.weight"])


def test_rungs_match_linear_response() -> None:
    """``linear`` is ``linear_response.linear_displacement``; ``linear_no_d`` is it with D = 1
    (−η Σ_t c_t g_t); ``linear_shuffled_d`` is it with π(D); ``gaussian_d`` is D^{1/2} ⊙ z."""
    rng = torch.Generator().manual_seed(3)
    grads = [named(rng) for _ in range(5)]
    D = named(rng, positive=True)
    weights = linear_response.momentum_weights(0.9, 5)
    lr = 1.3e-5

    def rung(name, d=None):
        return ladder.rung_direction(name, seed=50000, weights=weights, lr=lr, grads=iter(grads),
                                     D=d, like=D)  # fmt: skip

    expected = linear_response.linear_displacement(iter(grads), D, weights, lr)
    got = rung("linear", clone(D))
    assert all(torch.equal(got[k], expected[k]) for k in SHAPES)
    ones = {k: torch.ones_like(v) for k, v in D.items()}
    no_d = rung("linear_no_d")
    unit = linear_response.linear_displacement(iter(grads), ones, weights, lr)
    for k in SHAPES:
        assert torch.equal(no_d[k], unit[k]), k
        direct = -lr * sum(c * g[k].double() for c, g in zip(weights, grads, strict=True))
        torch.testing.assert_close(
            no_d[k].double(), direct, rtol=1e-5, atol=1e-6 * direct.abs().max().item()
        )  # fp32 sums
        assert no_d[k].shape == D[k].shape and no_d[k].is_contiguous()
    shuffled = rung("linear_shuffled_d", clone(D))
    permuted = linear_response.linear_displacement(
        iter(grads), ladder.shuffle_within_tensors(clone(D), 50000), weights, lr)  # fmt: skip
    assert all(torch.equal(shuffled[k], permuted[k]) for k in SHAPES)
    assert not torch.equal(shuffled["b.weight"], expected["b.weight"])
    iso = rung("gaussian_d", clone(D))
    reference = ladder.gaussian_d_direction(clone(D), 50000)
    assert all(torch.equal(iso[k], reference[k]) for k in SHAPES)
    with pytest.raises(ValueError, match="Unknown ladder rung"):
        rung("random", clone(D))


# ------------------------------------------------------------------------------- the scale search


def power(c_star: float, slope: float = 1.3, target: float = 6.0):
    """K(c) = K*·(c/c*)^b, exactly linear in ln K against ln c, with K(c*) = K*."""
    calls = []

    def k_of(c: float) -> float:
        calls.append(c)
        return target * (c / c_star) ** slope

    return k_of, calls


def test_scale_search_interpolates_and_extends() -> None:
    """ln K interpolated linearly in ln c between the bracketing neighbours (exact on a power
    law), the grid extended by ×2 or ÷2 on the needed side at most 3 times, then refused as
    ``not_bracketed`` with every value; a non-monotone K recorded and the bracketing pair nearest
    to K* chosen."""
    assert ladder.K_STAR == CANDIDATES[1]["targets"]["star"] == 6.00
    # pinned in the GPU code, not read from the analysis package (outside the GPU identity)
    assert "analysis" not in "".join(
        line
        for line in (REPO_ROOT / "src/zero_field_refusal/ladder.py").read_text().splitlines()
        if line.startswith(("from ", "import "))
    )

    k_of, calls = power(1.3)
    out = ladder.scale_search(k_of, 1.0, 6.0)
    assert calls == [0.5, 1.0, 2.0]
    assert out["status"] == "bracketed" and out["extensions"] == 0 and out["monotone"] is True
    assert [s["c"] for s in out["scales"]] == [0.5, 1.0, 2.0]
    assert out["pair"]["c"] == [1.0, 2.0] and out["choice"] == "only"
    assert out["c"] == pytest.approx(1.3, rel=1e-12)

    k_of, calls = power(5.0, slope=1.7)  # above 2c₀: extended upward twice
    out = ladder.scale_search(k_of, 1.0, 6.0)
    assert calls == [0.5, 1.0, 2.0, 4.0, 8.0] and out["extensions"] == 2
    assert out["pair"]["c"] == [4.0, 8.0] and out["c"] == pytest.approx(5.0, rel=1e-12)

    k_of, calls = power(0.1)  # below c₀/2: extended downward three times
    out = ladder.scale_search(k_of, 1.0, 6.0)
    assert calls == [0.5, 1.0, 2.0, 0.25, 0.125, 0.0625] and out["extensions"] == 3
    assert out["status"] == "bracketed" and out["pair"]["c"] == [0.0625, 0.125]
    assert out["c"] == pytest.approx(0.1, rel=1e-12)

    k_of, calls = power(100.0)  # not reached after 3 extensions
    out = ladder.scale_search(k_of, 1.0, 6.0)
    assert calls == [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    assert out["status"] == "not_bracketed" and out["extensions"] == 3
    assert [s["c"] for s in out["scales"]] == calls
    assert [s["K_benign"] for s in out["scales"]] == [6.0 * (c / 100.0) ** 1.3 for c in calls]
    assert out["c"] is None and out["pair"] is None

    k_of, calls = power(0.001)  # every K above K*: downward, then refused
    out = ladder.scale_search(k_of, 1.0, 6.0)
    assert calls == [0.5, 1.0, 2.0, 0.25, 0.125, 0.0625] and out["status"] == "not_bracketed"

    table = {0.5: 5.0, 1.0: 7.0, 2.0: 5.5}  # not monotone: two bracketing pairs
    out = ladder.scale_search(table.__getitem__, 1.0, 6.0)
    assert out["monotone"] is False and out["status"] == "bracketed"
    assert out["bracketing_pairs"] == [[0.5, 1.0], [1.0, 2.0]]
    assert out["choice"] == "nearest" and out["pair"]["c"] == [1.0, 2.0]  # 5.5 is nearest K*
    w = (math.log(6.0) - math.log(7.0)) / (math.log(5.5) - math.log(7.0))
    assert out["c"] == pytest.approx(math.exp(w * math.log(2.0)), rel=1e-12)

    out = ladder.scale_search({0.5: 3.0, 1.0: 6.0, 2.0: 12.0}.__getitem__, 1.0, 6.0)
    assert out["c"] == 1.0  # K* met at a grid scale

    for bad in (math.nan, math.inf, 0.0, -1e-9):
        out = ladder.scale_search({0.5: 3.0, 1.0: 5.0, 2.0: bad}.__getitem__, 1.0, 6.0)
        assert out["status"] == "invalid_k" and out["c"] is None
        assert len(out["scales"]) == 3
    with pytest.raises(ValueError, match="c0"):
        ladder.scale_search(power(1.0)[0], 0.0, 6.0)


def test_the_matched_k_step_refines_once_and_flags_a_rung_still_outside() -> None:
    """K measured at the interpolated c₁; within ±10 % of K* (5.40–6.60) the rung is matched at
    c₁. Outside, one refinement c₂ by the ln K–ln c interpolation between c₁ and the nearest
    measured scale (in ln c) on the other side of K*; the evaluation at whichever is nearer K* in
    ln K; still outside, flagged ``out_of_tolerance``."""
    assert ladder.TOLERANCE == 0.10
    assert ladder.within(5.40) and ladder.within(6.60) and ladder.within(6.0)
    assert not ladder.within(5.39) and not ladder.within(6.61) and not ladder.within(math.nan)
    k_of, calls = power(1.3)
    search = ladder.scale_search(k_of, 1.0, 6.0)
    calls.clear()
    out = ladder.refine(k_of, search)
    assert calls == [search["c"]] and out["status"] == "matched" and out["chosen"] == "c1"
    assert out["K1"] == pytest.approx(6.0) and out["c2"] is None and not out["out_of_tolerance"]
    assert out["tolerance"] == {"relative": 0.1, "lower": pytest.approx(5.4),
                                "upper": pytest.approx(6.6)}  # fmt: skip

    # the scales say K* lies at 1.3 (K = 6·c^1.3 between 1 and 2), but K at c₁ is 7.5: refined
    # once between c₁ and the nearest scale below K* in ln c (c = 1.0, K = 6/1.3^1.3)
    scales = {0.5: 6.0 * (0.5 / 1.3) ** 1.3, 1.0: 6.0 * (1.0 / 1.3) ** 1.3,
              2.0: 6.0 * (2.0 / 1.3) ** 1.3}  # fmt: skip
    search = ladder.scale_search(scales.__getitem__, 1.0, 6.0)
    c1 = search["c"]
    table = {c1: 7.5}
    c2_expected = math.exp(math.log(c1) + (math.log(6.0) - math.log(7.5)) / (
        math.log(scales[1.0]) - math.log(7.5)) * (math.log(1.0) - math.log(c1)))  # fmt: skip

    def measured(c):
        return table.get(c, 6.2)  # K at c₂: 6.2, within the tolerance

    out = ladder.refine(measured, search)
    assert out["other"] == {"c": 1.0, "K_benign": scales[1.0]}
    assert out["c2"] == pytest.approx(c2_expected, rel=1e-12) and 1.0 < out["c2"] < c1
    assert out["K2"] == 6.2 and out["chosen"] == "c2" and out["c"] == out["c2"]
    assert out["status"] == "matched" and out["out_of_tolerance"] is False

    # still outside after the one step: the nearer of the two, flagged
    out = ladder.refine(lambda c: 7.5 if c == c1 else 7.0, search)
    assert out["chosen"] == "c2" and out["K"] == 7.0 and out["out_of_tolerance"] is True
    assert out["status"] == "out_of_tolerance"
    out = ladder.refine(lambda c: 7.5 if c == c1 else 9.0, search)
    assert out["chosen"] == "c1" and out["K"] == 7.5 and out["out_of_tolerance"] is True
    # K below K* at c₁: refined toward the nearest scale above K* (c = 2.0)
    out = ladder.refine(lambda c: 4.0 if c == c1 else 5.9, search)
    assert out["other"]["c"] == 2.0 and c1 < out["c2"] < 2.0 and out["chosen"] == "c2"
    # a K at c₂ that is not finite: evaluated at c₁; at c₁: the rung stops
    out = ladder.refine(lambda c: 7.5 if c == c1 else math.nan, search)
    assert out["chosen"] == "c1" and out["out_of_tolerance"] is True
    out = ladder.refine(lambda c: math.inf, search)
    assert out["status"] == "invalid_k" and out["c"] is None and out["c2"] is None


# ------------------------------------------------------------------------------ the script


def stub_k(target: float, ratio: float):
    """K = K*·(‖θ − θ₀‖ / (ratio·‖Δθ‖))², so that K* lies at c = ratio·c₀; ``norm`` is ‖Δθ‖."""
    box = {}

    def k_benign(student, reference, samples, *, pad_id, micro_batch_size):
        sq = sum(
            (p.detach().double() - q.detach().double()).square().sum().item()
            for p, q in zip(student.parameters(), reference.parameters(), strict=True)
        )
        return target * (math.sqrt(sq) / (ratio * box["norm"])) ** 2

    return k_benign, box


def test_ladder_script_end_to_end(env, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    """A tiny fresh F run, then the ladder script on it: ``linear`` with a stubbed K placing K*
    at 1.3c₀ (the point θ₀ + cδ̂ exactly, evaluated in full by the run's evaluator under
    ``ladder_linear``, judgeable), ``gaussian_d`` with a K that never reaches K* (refused, nothing
    evaluated), and the refusals."""
    torch.set_num_threads(1)
    ckpt = tmp_path / "ckpt"
    overrides = [*F_RUN, "+study5.role=fresh"]
    cfg = ready(env, scheduled(tmp_path, *overrides, f"tracking.endpoint_checkpoint={ckpt}",
                               policy="short"))  # fmt: skip
    f_dir = run_scheduled(cfg, env, run_id="f-fresh")
    f_inputs = json.loads((f_dir / "meta.json").read_text())["inputs"]
    end = torch.load(ckpt / "f-fresh.pt", weights_only=True)
    theta0 = {n: p.detach() for n, p in freeze(env.new_model(), cfg.train.freeze)}
    from_ckpt = math.sqrt(math.fsum((end[n] - theta0[n]).double().square().sum().item()
                                    for n in theta0))  # fmt: skip
    monitor = [r for r in read_jsonl(f_dir / "monitor.jsonl") if r["step"] == 3]
    assert monitor[0]["displacement_norm"] == pytest.approx(from_ckpt, rel=1e-12)

    module = script_module(REPO_ROOT / "scripts" / "study5_ladder.py", "s5_ladder")
    monkeypatch.setattr(module, "environment", lambda cfg, device: env)
    real_k = ladder.k_benign
    k_benign, box = stub_k(ladder.K_STAR, 1.3)
    box["norm"] = monitor[0]["displacement_norm"]
    monkeypatch.setattr(ladder, "k_benign", k_benign)
    at_full = []
    original = Evaluator.__call__

    def spy(self, model, condition, **kwargs):
        at_full.append((condition, {n: p.detach().clone()
                                    for n, p in freeze(model, self.cfg.train.freeze)}))  # fmt: skip
        return original(self, model, condition, **kwargs)

    monkeypatch.setattr(Evaluator, "__call__", spy)
    config = [str(tmp_path / "study2.yaml"), *SMALL, *APPENDABLE, "teacher.chunk_size=12"]
    out = tmp_path / "records" / "ladder-linear.json"
    argv = [*config, *overrides, "--rung", "linear", "--f-run", str(f_dir), "--out", str(out),
            "--run-id", "ladder-linear"]  # fmt: skip
    assert module.main(argv) == 0
    record = json.loads(out.read_text())

    assert (record["rung"], record["seed"], record["lr"]) == ("linear", 0, 1e-3)
    assert record["condition"] == ladder_condition("linear") == "ladder_linear"
    assert record["k_star"] == 6.0 and "nats" in record["k_star_unit"]
    assert record["f_run"]["run_id"] == "f-fresh"
    assert record["displacement_norm"] == monitor[0]["displacement_norm"]
    assert record["displacement_source"].startswith("monitor.jsonl")
    assert record["v0_sha256"] == f_inputs["v0_sha256"]
    assert record["theta0_trainable_sha256"] == f_inputs["theta0_trainable_sha256"]
    assert record["streams"]["training"] == f_inputs["streams"]["training"]
    assert record["streams"]["preconditioner"] == f_inputs["streams"]["preconditioner"]
    assert record["rng"]["torch"] == derive_seed(0, "torch")

    # δ̂ independently from the predictor's code path, and the point evaluated in full
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    v0, _ = linear_response.theta0_second_moment(cfg, env, model, trainable)
    D = {n: 1.0 / (v.sqrt() + 1e-8) for (n, _), v in zip(trainable, v0, strict=True)}
    batches, _ = linear_response.feed_batches(cfg, env)
    weights = linear_response.momentum_weights(0.9, 3)
    delta = linear_response.linear_displacement(
        linear_response.step_gradients(model, trainable, batches, cfg, env), D, weights, 1e-3
    )
    norm = math.sqrt(math.fsum(t.double().square().sum().item() for t in delta.values()))
    assert record["delta_norm"] == pytest.approx(norm, rel=1e-12)
    assert record["c0"] == pytest.approx(box["norm"] / norm, rel=1e-12)
    assert [s["c"] for s in record["search"]["scales"]] == pytest.approx(
        [record["c0"] / 2, record["c0"], 2 * record["c0"]], rel=1e-15)  # fmt: skip
    assert record["search"]["pair"]["c"] == [record["c0"], 2 * record["c0"]]
    assert record["status"] == "evaluated"
    c = record["c"]
    assert c == pytest.approx(1.3 * record["c0"], rel=1e-6)  # θ₀ + cδ̂ rounded to fp32
    matched = record["matched_k"]  # K measured at c₁: K* there, matched, no refinement
    assert matched["c1"] == record["search"]["c"] == c and matched["chosen"] == "c1"
    assert matched["K1"] == pytest.approx(6.0, rel=1e-5) and matched["c2"] is None
    assert record["out_of_tolerance"] is False and matched["status"] == "matched"
    assert [name for name, _ in at_full] == ["ladder_linear"]
    for n, p in at_full[0][1].items():
        assert torch.equal(p, torch.add(theta0[n], delta[n], alpha=c)), n
    evaluation = record["evaluation"]
    for key in ("K_benign", "K_harmful", "heldout_nll_per_token", "factual_accuracy",
                "benchmark_accuracy"):  # fmt: skip
        assert math.isfinite(evaluation[key]), key
    # the scale search's K is the evaluator's K_benign at the same point
    point, reference = env.new_model(), env.new_model()
    for n, p in freeze(point, cfg.train.freeze):
        p.data.copy_(at_full[0][1][n])
    samples = Evaluator(cfg, env, _NoRun(), reference).kl_streams["kl_benign"]
    assert (
        real_k(
            point,
            reference,
            samples,
            pad_id=env.pad_id,
            micro_batch_size=cfg.execution.micro_batch_size,
        )
        == evaluation["K_benign"]
    )

    # the rung's evaluation run: judgeable under its one condition, like the scan points
    run_dir = tmp_path / "runs" / ladder.EXPERIMENT / "ladder-linear"
    assert record["run_id"] == "ladder-linear"
    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    assert run_cfg.run_type == LADDER_RUN_TYPE and run_cfg.study5.ladder.rung == "linear"
    assert response_conditions(run_cfg, run_dir) == ["ladder_linear"]
    assert unlabelable(run_dir, run_cfg) is None
    responses = read_jsonl(run_dir / "responses" / f"{cfg.eval.harmful.prompt_set}.jsonl")
    ids = [item["id"] for item in env.prompt_set(cfg.eval.harmful.prompt_set)]
    check_responses(run_dir, responses, ["ladder_linear"], ids)
    with pytest.raises(ValueError, match="run directory"):
        response_conditions(run_cfg)
    printed = capsys.readouterr().out
    assert f"c {c:.9e}" in printed and record["v0_sha256"] in printed
    for item in env.prompt_set(cfg.eval.harmful.prompt_set):
        assert item["text"] not in printed
    assert module.main(argv) == 1  # never overwritten
    assert json.loads(out.read_text()) == record

    # gaussian_d: K never reaches K* within c₀/2 … 16c₀: refused, nothing evaluated
    monkeypatch.setattr(ladder, "k_benign", lambda *args, **kwargs: 1e-3)
    out_iso = tmp_path / "records" / "ladder-gaussian_d.json"
    argv_iso = [*config, *overrides, "--rung", "gaussian_d", "--f-run", str(f_dir),
                "--out", str(out_iso), "--run-id", "ladder-gaussian_d"]  # fmt: skip
    assert module.main(argv_iso) == 0
    iso = json.loads(out_iso.read_text())
    assert iso["status"] == "not_bracketed" and iso["evaluation"] is None and iso["c"] is None
    assert len(iso["search"]["scales"]) == 6 and iso["search"]["extensions"] == 3
    assert iso["v0_sha256"] == f_inputs["v0_sha256"]
    assert iso["rng"]["gaussian_d"] == derive_seed(0, "study5", "gaussian_d")
    assert iso["streams"]["training"] is None  # no gradient is read
    assert [name for name, _ in at_full] == ["ladder_linear"]
    iso_dir = tmp_path / "runs" / ladder.EXPERIMENT / "ladder-gaussian_d"
    assert response_conditions(OmegaConf.load(iso_dir / "config.yaml"), iso_dir) == []
    assert not (iso_dir / "responses").exists()

    # refusals: another config than the F run's, an unknown rung, a run that is not F
    other = tmp_path / "records" / "other.json"
    lr = [o for o in overrides if not o.startswith("optimizer.lr")] + ["optimizer.lr=2e-3"]
    assert module.main([*config, *lr, "--rung", "linear", "--f-run", str(f_dir),
                        "--out", str(other)]) == 1  # fmt: skip
    with pytest.raises(SystemExit):
        module.main([*config, *overrides, "--rung", "random", "--f-run", str(f_dir),
                     "--out", str(other)])  # fmt: skip
    assert not other.exists()


class _NoRun:
    """Takes an evaluator's input records and drops them."""

    def summarize(self, values) -> None:
        pass

    def record_inputs(self, inputs) -> None:
        pass
