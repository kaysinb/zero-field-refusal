"""The diagnostic of the FA soft-control failure (docs/study2-fa-control-diagnostic.md) on tiny
CPU models: its products, Lanczos, recurrence, twin KL and branches, and the whole diagnostic on
tiny archived runs with harmful generation forbidden."""

import ast
import dataclasses
import json
import math
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import END_IDS, PAD_ID, prompts, tiny_model, tiny_tokenizer
from omegaconf import OmegaConf
from test_train import APPENDABLE, SMALL, pass_path_check, prepare, prompt_sets
from torch.nn.attention import SDPBackend, sdpa_kernel

from zero_field_refusal import fa_control_diagnostic as fcd
from zero_field_refusal.config import load_config
from zero_field_refusal.curvature import hessian_vector_product, lanczos, start_vector
from zero_field_refusal.evaluation import sequence_kl
from zero_field_refusal.models import freeze, install_neutral_generation_config
from zero_field_refusal.optim import Adam
from zero_field_refusal.runtime import Environment
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.teacher import sample_responses
from zero_field_refusal.tracking import CLEAN_CHECKOUT

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fa_control_diagnostic.py"
MODULE = ROOT / "src" / "zero_field_refusal" / "fa_control_diagnostic.py"
CAP = 12
SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=CAP)
CPU = torch.device("cpu")
SUBSET = ("model.layers.1.self_attn.v_proj.bias", "model.norm.weight")  # 48 parameters


def settings(**changes):
    return OmegaConf.merge(load_config("experiments/fa_control_diagnostic"), changes)


# ------------------------------------------------------------------------- outcome blindness


def test_the_settings_are_k_only_without_harmful_generation() -> None:
    d = settings().diagnostic
    assert d.evaluation == "k_only" and d.harmful_generation is False
    assert d.bound == pytest.approx(0.01 * math.sqrt(40)) == pytest.approx(0.01 * d.k_star)
    assert d.k_star == pytest.approx(math.sqrt(40)) and d.memory_rule_gb == 46
    assert list(d.d3.order) == ["fa-1.21e-05", "fa-1e-05", "adamw-1.32e-05"]
    assert d.d3.expansion == "fa-1.46e-05" and d.d3.expansion not in d.d3.order
    assert d.branches.failing == "fa-1.21e-05"
    assert list(d.branches.regular_cases) == list(d.d3.order)  # Revision 2, 3: AdamW included
    assert (d.d3.survival_share, d.d3.rounding_scale, d.d1.path_share_max) == (0.5, 10, 0.1)
    assert list(d.d3.eval_at) == [1, 5, 10, 15, 20, 25] and d.d2.batches == 4


def test_the_module_names_no_outcome_and_no_generation() -> None:
    """Outside docstrings, nothing in the module refers to harmful prompts, refusal, judges or
    greedy generation, and it imports neither the evaluator nor the generation helpers."""
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
    for banned in ("harm", "refusal", "judge", "greedy", "Evaluator", "generate"):
        assert not [w for w in words if banned in w], banned


# ------------------------------------------------------------------------------ pieces


def test_module_groups() -> None:
    assert fcd.module_group("model.layers.3.self_attn.q_proj.weight") == ("attn_q", 3)
    assert fcd.module_group("model.layers.0.self_attn.o_proj.weight") == ("attn_o", 0)
    assert fcd.module_group("model.layers.27.mlp.down_proj.weight") == ("mlp_down", 27)
    assert fcd.module_group("model.layers.2.post_attention_layernorm.weight") == ("norm", 2)
    assert fcd.module_group("model.norm.weight") == ("norm", None)
    assert fcd.module_group("model.embed_tokens.weight") == ("embedding", None)
    assert fcd.module_group("lm_head.weight") == ("lm_head", None)


def test_the_log_histogram_reads_quantiles_within_its_bin() -> None:
    g = torch.Generator().manual_seed(0)
    values = torch.exp(torch.randn(200_000, generator=g) * 8)
    values[:1000] = 0.0
    hist = fcd.LogHistogram()
    for part in values.split(30_000):
        hist.add(part)
    q = hist.quantiles((0.0, 0.001, 0.25, 0.5, 0.9, 1.0))
    exact = np.quantile(values.numpy(), [0.25, 0.5, 0.9])
    assert q["0"] == 0.0 and q["0.001"] == 0.0 and q["1"] == values.max().item()
    for key, value in zip(("0.25", "0.5", "0.9"), exact, strict=True):
        assert q[key] == pytest.approx(value, rel=0.03)


def test_the_census_counts_epsilon_dominated_coordinates() -> None:
    eps = 1e-8
    v = torch.full((4, 5), 1e-6)
    v[1, 2] = 0.0
    v[3, 0] = (0.5 * eps) ** 2
    v[3, 4] = (5 * eps) ** 2
    bias = torch.full((5,), 1e-4)
    names = ["model.layers.0.mlp.up_proj.weight", "model.layers.0.self_attn.q_proj.bias"]
    census = fcd.v0_census(names, [v, bias], eps)
    up = census["per_tensor"]["model.layers.0.mlp.up_proj.weight"]
    assert (up["below_eps"], up["below_10eps"], up["zeros"]) == (2, 3, 1)
    assert up["max_P"] == pytest.approx(1 / eps) and up["max_P_index"] == [1, 2]
    assert up["below_10eps_where"]["rows_with_any"] == 2
    groups = census["groups"]
    assert groups["all"]["below_eps"] == 2 and groups["all"]["numel"] == 25
    assert groups["mlp_up"]["fraction_below_10eps"] == pytest.approx(3 / 20)
    assert groups["mlp_up.0"]["max_P_at"][0] == "model.layers.0.mlp.up_proj.weight"
    assert groups["attn_q"]["below_eps"] == 0
    shares = fcd.preconditioned_shares(["model.layers.0.mlp.up_proj.weight"], [v],
                                       [torch.ones(4, 5)], eps)  # fmt: skip
    p = 1 / (v.double().sqrt() + eps)
    assert shares["share_by_sqrt_v0"]["below_eps"] == pytest.approx(
        (p[1, 2] ** 2 + p[3, 0] ** 2).item() / p.pow(2).sum().item(), rel=1e-6
    )
    assert shares["max_P_abs_g_at"] == ["model.layers.0.mlp.up_proj.weight", [1, 2]]


def test_float64_gradients_sum_each_backward_and_release_the_fp32_ones() -> None:
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    with fcd.Float64Gradients([w]) as sums:
        (w * torch.tensor([3.0, 4.0])).sum().backward()
        assert w.grad is None
        (w.pow(2)).sum().backward()
    (total,) = sums.take()
    assert total.dtype == torch.float64 and torch.equal(total, torch.tensor([5.0, 8.0]).double())
    with fcd.Float64Gradients([w]) as sums:
        (w * 2).sum().backward()
    sums.to_grad()
    assert w.grad.dtype == torch.float32 and torch.equal(w.grad, torch.tensor([2.0, 2.0]))


@pytest.mark.parametrize("name", ["adam_frozen", "adamw"])
def test_the_float64_adam_follows_the_code_s_adam(name) -> None:
    g = torch.Generator().manual_seed(1)
    start = torch.randn(50, generator=g)
    grads = [torch.randn(50, generator=g) for _ in range(5)]
    v0 = torch.rand(50, generator=g) * 1e-2
    cfg = OmegaConf.create({"name": name, "lr": 1e-2, "eps": 1e-8, "beta1": 0.9,
                            "betas": [0.9, 0.999]})  # fmt: skip
    a, b = torch.nn.Parameter(start.clone()), torch.nn.Parameter(start.clone())
    mode = "frozen" if name == "adam_frozen" else "adaptive"
    betas = (0.9, None) if mode == "frozen" else (0.9, 0.999)
    reference = Adam([a], lr=1e-2, betas=betas, eps=1e-8, second_moment=mode,
                     initial_second_moment=[v0] if mode == "frozen" else None)  # fmt: skip
    wide = fcd.Float64Adam([b], cfg, [v0] if mode == "frozen" else None)
    for grad in grads:
        a.grad = grad.clone()
        reference.step()
        wide.step([grad.double()])
    torch.testing.assert_close(b.data, a.data, rtol=1e-5, atol=1e-6)
    assert b.dtype == torch.float32


@pytest.mark.parametrize("eta_lambda", [1.5, 30.0, 45.0])
def test_the_scalar_mode_is_the_code_s_recurrence_on_a_quadratic(eta_lambda) -> None:
    """One coordinate under frozen Adam (optim.Adam) on ½hx² after a first step on a residual r:
    its displacement, relative to the first step's, is ``scalar_mode`` at λ = P·h."""
    v0, eps, eta, beta = torch.tensor([4e-6]), 1e-8, 1e-3, 0.9
    p = 1 / (math.sqrt(4e-6) + eps)
    h = eta_lambda / (eta * p)
    x = torch.nn.Parameter(torch.zeros(1))
    optimizer = Adam([x], lr=eta, betas=(beta, None), eps=eps, second_moment="frozen",
                     initial_second_moment=[v0])  # fmt: skip
    x.grad = torch.tensor([1e-4])
    optimizer.step()
    first, path = x.item(), [1.0]
    for _ in range(24):
        x.grad = h * x.detach().clone()
        optimizer.step()
        path.append(abs(x.item() / first))
    predicted = fcd.scalar_mode(eta, p * h, beta, 25)
    np.testing.assert_allclose(path, predicted, rtol=1e-4)
    # The stationary map's radius: below 1 under the heavy-ball bound, above it beyond.
    bound = fcd.heavy_ball_threshold(eta, beta)
    assert bound == pytest.approx(38 / eta)
    assert (
        fcd.stationary_radius(eta * 0.9 * bound, beta)
        < 1
        < fcd.stationary_radius(eta * 1.1 * bound, beta)
    )
    assert fcd.stationary_radius(-eta * 100, beta) > 1  # negative curvature grows


def test_the_per_step_growth_and_the_observed_controls() -> None:
    rows = [{"step": 5, "K_monitor": 1e-4, "displacement_norm": 1e-3},
            {"step": 10, "K_monitor": 1.6e-3, "displacement_norm": 4e-3}]  # fmt: skip
    observed = fcd.observed_growth(rows)
    (window,) = observed["delta_per_step_from_K"]
    assert window["per_step"] == pytest.approx(16 ** (1 / 10))
    assert observed["delta_per_step_from_displacement"][0]["per_step"] == pytest.approx(4**0.2)


def ones(name, shape):
    return torch.ones(shape)


def test_perturbations_survive_rounding_by_their_realized_norm() -> None:
    """Revision 2, 4: survival is ‖fp32(θ₀ + p) − θ₀‖ ≥ ½‖p‖, not a changed coordinate."""
    p = torch.nn.Parameter(torch.ones(1000))
    theta0 = {"w": torch.ones(1000)}
    lost = fcd.perturb([("w", p)], theta0, ones, 1e-12)
    assert lost["changed"] == 0 and not lost["survived"] and lost["realized_norm"] == 0
    kept = fcd.perturb([("w", p)], theta0, ones, 1e-6)
    assert kept["changed"] == 1000 and kept["survived"]
    assert kept["realized_norm"] == pytest.approx(kept["intended_norm"], rel=0.1)
    # One small coordinate of θ₀ takes the perturbation, the 999 others round it away: some
    # coordinates change, but the realized norm is far under half the intended one.
    small = torch.ones(1000)
    small[7] = 1e-6
    partial = fcd.perturb([("w", p)], {"w": small}, ones, 1e-12)
    assert partial["changed"] == 1 and not partial["survived"]
    assert partial["realized_norm"] < 0.5 * partial["intended_norm"]
    assert partial["intended_norm"] == pytest.approx(1e-12 * 1000**0.5)


def test_a_perturbation_lost_to_rounding_is_scaled_once_and_the_second_attempt_kept() -> None:
    p = torch.nn.Parameter(torch.ones(1000))
    small = torch.ones(1000)
    small[7] = 1e-6
    attempts = fcd.apply_perturbation([("w", p)], {"w": small}, ones, 1e-12,
                                      survival_share=0.5, rounding_scale=10)  # fmt: skip
    assert [a["scale"] for a in attempts] == [1e-12, 1e-11]
    assert not attempts[0]["survived"] and not attempts[1]["survived"]  # kept all the same
    assert all({"intended_norm", "realized_norm", "changed"} <= set(a) for a in attempts)
    torch.testing.assert_close(p.data, small + torch.ones(1000) * 1e-11, rtol=0, atol=0)
    survives = fcd.apply_perturbation([("w", p)], {"w": torch.ones(1000)}, ones, 1e-6,
                                      survival_share=0.5, rounding_scale=10)  # fmt: skip
    assert len(survives) == 1 and survives[0]["survived"]


# --------------------------------------------------------------- products, Lanczos and the map


@pytest.fixture
def tiny():
    model = tiny_model(tied=False, head_std=0.02)
    install_neutral_generation_config(model, END_IDS, PAD_ID)
    samples, distributions = sample_responses(
        model, prompts(8), [f"p{i}" for i in range(8)], sampling=SAMPLING, end_ids=END_IDS,
        pad_id=PAD_ID, seed=1, batch_size=8, record_distributions=True,
    )  # fmt: skip
    trainable = freeze(model, ["input_embeddings", "output_head"])
    cfg = OmegaConf.create({
        "train": {"max_response_tokens": CAP, "loss": {"normalization": "constant",
                                                        "denominator": 4 * CAP}},
        "execution": {"micro_batch_size": 3},
    })  # fmt: skip
    env = fcd.Environment(tokenizer=None, end_ids=END_IDS, pad_id=PAD_ID, device=CPU,
                          new_model=None, prompt_set=None, data_root=Path("."))  # fmt: skip
    batches = [samples[:4], samples[4:]]
    targets = [distributions[:4], distributions[4:]]
    return model, trainable, cfg, env, batches, targets


class _Loss(torch.nn.Module):
    def __init__(self, model, terms):
        super().__init__()
        self.model, self.terms = model, terms

    def forward(self):
        return sum(term() for term in self.terms)


def dense_hessian(model, terms, names):
    """The Hessian of the terms' sum over the named parameters, by autograd on a functional
    call (no product code of the module)."""
    params = dict(model.named_parameters())
    flat = torch.cat([params[n].detach().reshape(-1) for n in names]).double()
    shapes = [params[n].shape for n in names]
    loss = _Loss(model, terms)

    def f(x):
        pieces, i = {}, 0
        for n, shape in zip(names, shapes, strict=True):
            size = math.prod(shape)
            pieces[f"model.{n}"] = x[i : i + size].reshape(shape).float()
            i += size
        return torch.func.functional_call(loss, pieces, ()).double()

    with sdpa_kernel(SDPBackend.MATH):
        return torch.autograd.functional.hessian(f, flat)


def subset(trainable):
    chosen = [(n, p) for n, p in trainable if n in SUBSET]
    assert [n for n, _ in chosen] == list(SUBSET)
    return chosen


def flat(xs):
    return torch.cat([x.reshape(-1) for x in xs]).double()


@pytest.mark.parametrize("kind", ["soft", "nll"])
def test_the_loss_products_match_a_dense_hessian(tiny, kind) -> None:
    model, trainable, cfg, env, batches, targets = tiny
    terms = fcd.batch_terms(model, batches, cfg, env, kind=kind,
                            soft_targets=targets if kind == "soft" else None)  # fmt: skip
    chosen = subset(trainable)
    params = [p for _, p in chosen]
    dense = dense_hessian(model, terms, list(SUBSET))
    torch.testing.assert_close(dense, dense.T, rtol=1e-4, atol=1e-6)
    v = start_vector(chosen, 0, 0)
    out = [torch.zeros_like(p) for p in params]

    def consume(part):
        for o, h in zip(out, part, strict=True):
            o.add_(h)

    fcd.loss_products(model, params, terms, v, consume)
    torch.testing.assert_close(flat(out), dense @ flat(v), rtol=1e-3, atol=1e-6)
    assert fcd.quadratic_form(model, params, terms, v) == pytest.approx(
        (flat(v) @ dense @ flat(v)).item(), rel=1e-3
    )


def test_the_nll_products_equal_the_curvature_module_s(tiny) -> None:
    model, trainable, cfg, env, batches, _ = tiny
    params = [p for _, p in trainable]
    terms = fcd.batch_terms(model, batches, cfg, env, kind="nll")
    v = start_vector(trainable, 0, 3)
    out = [torch.zeros_like(p) for p in params]

    def consume(part):
        for o, h in zip(out, part, strict=True):
            o.add_(h)

    fcd.loss_products(model, params, terms, v, consume)
    expected = hessian_vector_product(model, params, batches, v, cap=CAP, micro_batch_size=3,
                                      pad_id=PAD_ID, device=CPU)  # fmt: skip
    for a, b in zip(out, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-9)


def test_preconditioned_lanczos_finds_the_dense_spectrum(tiny) -> None:
    model, trainable, cfg, env, batches, targets = tiny
    terms = fcd.batch_terms(model, batches, cfg, env, kind="soft", soft_targets=targets)
    chosen = subset(trainable)
    params = [p for _, p in chosen]
    g = torch.Generator().manual_seed(2)
    v0 = [torch.rand(p.shape, generator=g) * 1e-4 for p in params]
    v0[0].reshape(-1)[:3] = 1e-20  # ε-dominated coordinates, P ≈ 1e8
    eps = 1e-8
    scale = flat([fcd.sqrt_preconditioner(v, eps) for v in v0])
    dense = scale[:, None] * dense_hessian(model, terms, list(SUBSET)) * scale[None, :]
    values, vectors = torch.linalg.eigh(dense)
    hvp = fcd.preconditioned_hvp(model, params, terms, v0, eps)
    n = dense.shape[0]
    result = lanczos(hvp, start_vector(chosen, 0, 0), steps=n // 2, tolerance=1e-3)
    assert result["top"] == pytest.approx(values[-1].item(), rel=1e-3)
    assert result["bottom"] >= values[0].item() - 1e-6 * values[-1].item()  # inside the spectrum
    y = fcd.ritz_vector(hvp, start_vector(chosen, 0, 0), result)
    check = fcd.check_vector(hvp, y, CPU)
    assert check["rayleigh"] == pytest.approx(values[-1].item(), rel=1e-3)
    assert abs(torch.nn.functional.cosine_similarity(flat(y), vectors[:, -1], dim=0)) > 0.999
    shares = fcd.vector_shares(list(SUBSET), y, v0, eps)
    assert sum(shares["share_by_kind"].values()) == pytest.approx(1.0)
    assert 0 <= shares["share_below_eps"] <= shares["share_below_10eps"] <= 1


def test_the_linearized_run_is_the_recurrence_on_the_dense_hessians(tiny) -> None:
    """``linearized_run`` against the code's recurrence written with dense batch Hessians, the
    two batches cycled, started from a residual as the first step starts."""
    model, trainable, cfg, env, batches, targets = tiny
    chosen = subset(trainable)
    params = [p for _, p in chosen]
    step_terms = [fcd.batch_terms(model, [b], cfg, env, kind="soft", soft_targets=[t])
                  for b, t in zip(batches, targets, strict=True)]  # fmt: skip
    fisher = fcd.batch_terms(model, batches, cfg, env, kind="soft", soft_targets=targets)
    g = torch.Generator().manual_seed(3)
    v0 = [torch.rand(p.shape, generator=g) * 1e-6 + 1e-7 for p in params]
    residual = [torch.randn(p.shape, generator=g) * 1e-4 for p in params]
    eta, beta, eps, steps = 3e-3, 0.9, 1e-8, 7
    result = fcd.linearized_run(model, params, v0, residual, eta=eta, beta=beta, eps=eps,
                                steps=steps, step_terms=step_terms, fisher_terms=fisher,
                                fisher_at=[1, 4, 7])  # fmt: skip
    hessians = [dense_hessian(model, t, list(SUBSET)) for t in step_terms]
    f = dense_hessian(model, fisher, list(SUBSET))
    p = 1 / (flat(v0).sqrt() + eps)
    m = (1 - beta) * flat(residual)
    delta = -eta * p * flat(residual)
    norms, quadratic = {1: delta.norm().item()}, {1: (delta @ f @ delta).item()}
    for t in range(2, steps + 1):
        m = beta * m + (1 - beta) * hessians[(t - 1) % 2] @ delta
        delta = delta - eta / (1 - beta**t) * p * m
        norms[t] = delta.norm().item()
        quadratic[t] = (delta @ f @ delta).item()
    assert result["first_step_norm"] == pytest.approx(norms[1], rel=1e-5)
    for t in range(1, steps + 1):
        assert result["delta_norm"][t] == pytest.approx(norms[t], rel=1e-3)
    for t in (4, 7):
        assert result["delta_F_delta_relative"][t] == pytest.approx(
            quadratic[t] / quadratic[1], rel=1e-2
        )


def test_twin_divergence_is_k_in_both_directions(tiny) -> None:
    model, _, _, env, batches, _ = tiny
    other = tiny_model(tied=False, head_std=0.02)
    with torch.no_grad():
        for p, q in zip(other.parameters(), model.parameters(), strict=True):
            p.copy_(q + 1e-3 * torch.randn_like(q))
    samples = [s for b in batches for s in b]
    out = fcd.twin_divergence(model, other, samples, pad_id=PAD_ID, micro_batch_size=3)
    a_b = sequence_kl(other, model, samples, pad_id=PAD_ID, micro_batch_size=3).mean().item()
    b_a = sequence_kl(model, other, samples, pad_id=PAD_ID, micro_batch_size=3).mean().item()
    assert (out["KL_A_B"], out["KL_B_A"]) == (a_b, b_a)
    assert out["D_twin"] == max(a_b, b_a) > 0 and out["responses"] == len(samples)
    same = fcd.twin_divergence(model, model, samples, pad_id=PAD_ID, micro_batch_size=3)
    assert same["D_twin"] == 0.0


# ------------------------------------------------------------------------------- branches

BOUND = 0.0632455532033676
EVAL_AT = [1, 5, 10, 15, 20, 25]  # D3's steps and the same-path control's (the settings)
COMMIT = "c" * 40
SURVIVED = [{"scale": 1.0, "intended_norm": 1e-3, "realized_norm": 1e-3, "changed": 9,
             "survived": True}]  # fmt: skip
SHARED = {"trainable_parameters": "equal", "v0": "equal", "loss_normalization": "equal",
          "initial_optimizer_state": "equal_except_lr"}  # fmt: skip


def written(record, commit=COMMIT):
    return {**record, "provenance": {"code": {"commit": commit}}}


def series(last, steps=EVAL_AT):
    """A measured series: 0.001 at every step but the last, which is ``last``."""
    return [0.001] * (len(steps) - 1) + [last]


def pair(last=0.001, *, steps=EVAL_AT, values=None, rounding=SURVIVED, stop=None,
         diverged=False, rows=()):  # fmt: skip
    """A D3 pair's record as ``run_case`` writes it (D_twin readings, rounding, stop)."""
    values = series(last, steps) if values is None else values
    return {"D_twin": [{"step": s, "D_twin": v} for s, v in zip(steps, values, strict=True)],
            "rows": list(rows), "stop": stop, "diverged": diverged, "rounding": rounding,
            "max_D_twin": fcd._largest(values)}  # fmt: skip


def control(last=0.01, *, steps=EVAL_AT, stop=None, diverged=False, monitor=()):
    """The same-path control's record as ``same_path_control`` writes it."""
    return {"scoring": [{"step": s, "K_scoring": v}
                        for s, v in zip(steps, series(last, steps), strict=True)],
            "monitor": list(monitor), "rows": [], "stop": stop, "diverged": diverged}  # fmt: skip


def outputs(*, path=True, same_path=0.01, pairs=None, determinism=True, expansion=False):
    """Synthetic D1 and D3 outputs at one commit, every validity check passed: ``pairs`` maps a
    case to its pairs, each a number (its D_twin at the last step, 0.001 at the others) or a
    record; ``same_path`` is the control's K_scoring at its last step, or its record; ``path``
    puts ‖(i)‖ at about 1 % or 88 % of ‖(iii)‖."""
    pairs = pairs or {}
    names = ["fa-1.21e-05", "fa-1e-05", "adamw-1.32e-05"] + (["fa-1.46e-05"] if expansion else [])

    def record(key, spec):
        out = pair(spec) if isinstance(spec, int | float) else dict(spec)
        if key == "reference":  # no perturbation, a mode
            out.pop("rounding", None)
            out["mode"] = fcd.FLOAT64_FULL
        return out

    cases = {name: written({
        "status": "complete", "twin_A": {"archive": {"equal": True}},
        "pairs": {key: record(key, pairs.get(name, {}).get(key, 0.001)) for key in fcd.PAIRS},
    }) for name in names}  # fmt: skip
    d1 = written({
        "status": "complete",
        "shared_state": {check: {key: True} for check, key in SHARED.items()},
        "residual": {"norms": {"i_training_path_fp32": 1e-6 if path else 1e-4,
                               "iii_control_recorded": 1.1356e-4},
                     "iii_equals_record": {"1e-05": True, "1.21e-05": True, "1.46e-05": True}},
        "same_path_control": (control(same_path) if isinstance(same_path, int | float)
                              else same_path),
    })  # fmt: skip
    return d1, written({"equal": determinism}), cases


def decide(d1, determinism, cases, **kwargs):
    return fcd.decide_branch(settings(), d1, determinism, cases, **kwargs)


def branch(**changes):
    out = decide(*outputs(**changes))
    return out["branch"], out["expand"]


def test_branch_f_needs_the_path_the_same_path_control_and_the_regular_runs() -> None:
    out = decide(*outputs())
    assert (out["branch"], out["facts"]["step"]) == ("F", 5)
    facts = out["facts"]
    assert facts["path_criterion"]["state"] == "holds"
    assert facts["same_path_control_state"] == "below"
    assert facts["pair_states"]["adamw-1.32e-05"] == dict.fromkeys(fcd.PAIRS, "within")
    assert facts["shared_state"] == dict.fromkeys(fcd.SHARED_STATE_CHECKS, True)
    assert branch(pairs={"fa-1e-05": {"p3": 0.2}}) == ("ambiguous", "fa-1.46e-05")
    assert branch(path=False) == ("ambiguous", "fa-1.46e-05")  # false on finite norms


@pytest.mark.parametrize("key", ["p1", "p2", "p3", "p4", "reference"])
def test_an_adamw_pair_above_the_bound_excludes_f(key) -> None:
    """Revision 2, 3: Branch F's regular runs are every D3 case, AdamW 1.32e-5 included; with
    every other F condition met, the outcome is ambiguous (A and B read FA 1.21e-5 only)."""
    out = decide(*outputs(pairs={"adamw-1.32e-05": {key: BOUND * 1.01}}))
    assert (out["branch"], out["expand"], out["facts"]["step"]) == ("ambiguous", "fa-1.46e-05", 7)
    states = out["facts"]["pair_states"]
    assert set(states["fa-1.21e-05"].values()) == {"within"}
    assert states["adamw-1.32e-05"][key] == "exceeding"  # reported
    # The same AdamW pair under Branch B's conditions leaves B, and under A's leaves A.
    assert branch(same_path=0.2, pairs={"adamw-1.32e-05": {key: 1.0}}) == ("B", None)
    a = decide(*outputs(pairs={"adamw-1.32e-05": {key: 1.0}, "fa-1.21e-05": {"p1": 1.0}}))
    assert a["branch"] == "A" and a["facts"]["pair_states"]["adamw-1.32e-05"][key] == "exceeding"


def test_branch_b_and_a() -> None:
    assert branch(same_path=0.2) == ("B", None)
    assert branch(same_path=0.2, path=False) == ("B", None)
    for key in ("p1", "p2", "p3", "p4", "reference"):
        out = decide(*outputs(pairs={"fa-1.21e-05": {key: BOUND * 1.01}}))
        assert out["branch"] == "A"
        assert out["facts"]["pair_states"]["fa-1.21e-05"][key] == "exceeding"
    assert branch(pairs={"fa-1.21e-05": {"p1": BOUND}}) == ("F", None)  # ≤ the bound is within


def test_a_diverged_pair_is_exceeding() -> None:
    """Revision 3, 3: a non-finite loss, gradient, weight or D_twin is a divergence, which counts
    as exceeding, without survival and on an incomplete series."""
    stopped = pair(values=[0.001, 0.002], steps=[1, 5], stop="non-finite weights after step 6",
                   diverged=True)  # fmt: skip
    nan = pair(values=[0.001, math.nan, *[0.001] * 4])
    infinite_loss = pair(rows=[{"step": 0, "loss": 1.0, "grad_norm": 2.0},
                               {"step": 1, "loss": math.inf, "grad_norm": 2.0}])  # fmt: skip
    lost = [{"scale": 1.0, "intended_norm": 1e-3, "realized_norm": 0.0, "changed": 0,
             "survived": False}] * 2  # fmt: skip
    for record in (stopped, nan, infinite_loss, {**stopped, "rounding": lost}):
        out = decide(*outputs(pairs={"fa-1.21e-05": {"p3": record}}))
        assert (out["branch"], out["facts"]["step"]) == ("A", 1)
        assert out["facts"]["pair_states"]["fa-1.21e-05"]["p3"] == "exceeding"
    assert decide(*outputs(pairs={"fa-1.21e-05": {"reference": stopped}}))["branch"] == "A"
    assert branch(pairs={"fa-1.21e-05": {"p2": math.inf}}) == ("A", None)
    # A diverged pair of FA 1e-5 is a regular-run pair exceeding: ambiguous, not F
    assert branch(pairs={"fa-1e-05": {"p1": stopped}}) == ("ambiguous", "fa-1.46e-05")


def test_an_ambiguous_branch_expands_once_then_reports_without_choosing() -> None:
    out = decide(*outputs(path=False, expansion=True, pairs={"fa-1.46e-05": {"p1": 1.0}}))
    assert (out["branch"], out["expand"]) == ("ambiguous", None)
    assert out["facts"]["expansion"]["pair_states"]["p1"] == "exceeding"
    # The expansion never changes a decided branch.
    assert branch(expansion=True, pairs={"fa-1.46e-05": {"p1": 1.0}}) == ("F", None)


@pytest.mark.parametrize("check", [*SHARED, "iii_equals_record", "iii_without_a_control"])
def test_each_failed_shared_state_check_makes_the_outcome_incomplete(check) -> None:
    """Revision 3, 1 (review fa8 reached F with each of these checks false): every D1
    shared-state check gates every branch, A, B and F alike."""
    for changes, valid in (({}, "F"), ({"same_path": 0.2}, "B"),
                           ({"pairs": {"fa-1.21e-05": {"p1": 1.0}}}, "A")):  # fmt: skip
        d1, determinism, cases = outputs(**changes)
        assert decide(d1, determinism, cases)["branch"] == valid
        if check == "iii_equals_record":
            d1["residual"]["iii_equals_record"]["1.46e-05"] = False
        elif check == "iii_without_a_control":  # (iii) checked against two controls of three
            del d1["residual"]["iii_equals_record"]["1e-05"]
        else:
            d1["shared_state"][check][SHARED[check]] = False
        out = decide(d1, determinism, cases)
        assert (out["branch"], out["expand"]) == ("incomplete", None)
        name = "iii_equals_record" if check.startswith("iii") else check
        assert out["facts"]["shared_state"][name] is False
        assert out["facts"]["incomplete"] == [f"d1/result.json: {fcd.SHARED_STATE_CHECKS[name]}"]
    d1, determinism, cases = outputs()
    d1["status"] = "incomplete"  # D1's own verdict is read too
    assert decide(d1, determinism, cases)["branch"] == "incomplete"


UNDECIDED = [(math.nan, 1e-4), (math.inf, 1e-4), (1e-6, math.nan), (1e-6, math.inf), (1e-6, 0.0),
             (0.0, 0.0), (None, 1e-4)]  # fmt: skip


@pytest.mark.parametrize("norms", UNDECIDED)
def test_an_undecided_path_criterion_makes_the_outcome_incomplete(norms) -> None:
    """Revision 3, 1: a non-finite ‖(i)‖ or ‖(iii)‖, or a zero ‖(iii)‖, leaves the path
    criterion undecided, never false; the outcome is incomplete, whatever else holds."""
    residual = {"norms": {"i_training_path_fp32": norms[0], "iii_control_recorded": norms[1]}}
    criterion = fcd.path_criterion(residual, 0.1)
    assert criterion["state"] == "undecided" and criterion["ratio_i_to_iii"] is None
    for changes in ({}, {"same_path": 0.2}, {"pairs": {"fa-1.21e-05": {"p1": 1.0}}},
                    {"pairs": {"fa-1e-05": {"p1": 1.0}}}):  # fmt: skip
        d1, determinism, cases = outputs(**changes)
        d1["residual"]["norms"].update(residual["norms"])
        out = decide(d1, determinism, cases)
        assert (out["branch"], out["expand"]) == ("incomplete", None)
        assert out["facts"]["path_criterion"]["state"] == "undecided"
    finite = {"norms": {"i_training_path_fp32": 2e-5, "iii_control_recorded": 1e-4}}
    assert fcd.path_criterion(finite, 0.1)["state"] == "false"
    assert fcd.path_criterion(finite, 0.2)["state"] == "holds"  # ≤: the bound itself holds


def lost_to_rounding():
    """Review fa8's reproduction: both attempts of a perturbation realize no displacement."""
    p = torch.nn.Parameter(torch.ones(1000))
    attempts = fcd.apply_perturbation([("w", p)], {"w": torch.ones(1000)}, ones, 1e-12,
                                      survival_share=0.5, rounding_scale=10)  # fmt: skip
    assert [a["realized_norm"] for a in attempts] == [0.0, 0.0]
    assert all(a["intended_norm"] > 0 for a in attempts)
    return attempts


@pytest.mark.parametrize("key", ["p1", "p2", "p3", "p4"])
def test_a_perturbation_lost_to_rounding_is_never_within(key) -> None:
    """Revision 3, 2 (review fa8 reached F with two attempts of zero realized displacement):
    after its single ×10 retry, a perturbation that does not survive leaves its pair
    incomplete, never within; it can still be exceeding."""
    attempts = lost_to_rounding()
    lost = pair(0.0, values=[0.0] * 6, rounding=attempts)
    state = fcd.pair_state(lost, perturbed=True, eval_at=EVAL_AT, bound=BOUND, survival_share=0.5)
    assert (state["state"], state["survived"]) == ("incomplete", False)
    out = decide(*outputs(pairs={"fa-1.21e-05": {key: lost}}))
    assert (out["branch"], out["expand"], out["facts"]["step"]) == ("incomplete", None, 2)
    assert out["facts"]["pairs_survived"]["fa-1.21e-05"][key] is False
    assert branch(same_path=0.2, pairs={"fa-1.21e-05": {key: lost}}) == ("incomplete", None)
    assert branch(pairs={"fa-1e-05": {key: lost}}) == ("incomplete", None)  # step 6
    # A lost perturbation that still exceeds shows at least that sensitivity
    assert branch(pairs={"fa-1.21e-05": {key: pair(0.2, rounding=attempts)}}) == ("A", None)
    # Each attempt is judged against its own intended norm: a lost first attempt, then a retry
    # realizing 0.6, or 0.4, of its own norm (6 and 4 times the first attempt's)
    retry = {"scale": 10.0, "intended_norm": 1e-2, "realized_norm": 0.6e-2, "changed": 9,
             "survived": True}  # fmt: skip
    kept = pair(rounding=[attempts[0], retry])
    assert branch(pairs={"fa-1.21e-05": {key: kept}}) == ("F", None)
    short = pair(rounding=[attempts[0], {**retry, "realized_norm": 0.4e-2}])
    assert branch(pairs={"fa-1.21e-05": {key: short}}) == ("incomplete", None)
    assert branch(pairs={"fa-1.21e-05": {key: pair(rounding=[])}}) == ("incomplete", None)


def test_an_incomplete_pair_at_the_failing_rate_is_not_a() -> None:
    """Revision 3, 3: an incomplete series is not an unbounded one (``_pair_summary`` mapped it
    to ∞, which gave A); the outcome is incomplete."""
    summary = fcd._pair_summary([{"step": 1, "D_twin": 1.21e-5}, {"step": 5, "D_twin": 1.21e-5}],
                                EVAL_AT, None)  # fmt: skip
    assert summary == {"max_D_twin": 1.21e-5, "complete": False}
    short = pair(values=[1.21e-5] * 5, steps=EVAL_AT[:5])  # step 25 never measured
    stopped = pair(1.21e-5, stop="interrupted")  # a stop that is not a divergence
    for key, record in (("p1", short), ("reference", short), ("p1", stopped)):
        out = decide(*outputs(pairs={"fa-1.21e-05": {key: record}}))
        assert (out["branch"], out["expand"], out["facts"]["step"]) == ("incomplete", None, 2)
        assert out["facts"]["pair_states"]["fa-1.21e-05"][key] == "incomplete"
        assert out["facts"]["incomplete"] == [
            f"step 2: the pairs {[key]} of fa-1.21e-05 are incomplete"
        ]


def test_a_one_point_control_series_is_incomplete_not_b() -> None:
    """Review fa8 reproduced B from a one-point scoring series: "not below" is no evidence.
    Revision 3, 3: the control is exceeding only on its complete series, or when it diverged."""
    out = decide(*outputs(same_path=control(0.2, steps=[1])))
    assert (out["branch"], out["expand"], out["facts"]["step"]) == ("incomplete", None, 3)
    assert out["facts"]["same_path_control_state"] == "incomplete"
    assert out["facts"]["incomplete"] == ["step 3: the same-path control is incomplete"]
    stopped = control(0.01, steps=[1, 5], stop="interrupted")  # not a divergence
    assert branch(same_path=stopped) == ("incomplete", None)
    # A run that diverged is exceeding, whatever it measured; so is a complete series above
    diverged = control(0.01, steps=[1], stop="non-finite loss or gradient at step 3",
                       diverged=True)  # fmt: skip
    assert branch(same_path=diverged) == ("B", None)
    assert branch(same_path=control(math.nan)) == ("B", None)
    monitor = [{"step": 5, "K_monitor": math.inf, "displacement_norm": 1.0}]
    assert branch(same_path=control(0.01, monitor=monitor)) == ("B", None)  # a non-finite K
    assert branch(same_path=control(BOUND * 1.0001)) == ("B", None)


def test_the_control_exactly_at_the_bound_is_ambiguous() -> None:
    """Revision 3, 3 and 4: neither below nor exceeding on a complete series is at the bound,
    and the outcome is ambiguous (step 7), no longer B."""
    out = decide(*outputs(same_path=BOUND))
    assert (out["branch"], out["expand"], out["facts"]["step"]) == ("ambiguous", "fa-1.46e-05", 7)
    assert out["facts"]["same_path_control_state"] == "at_bound"
    assert out["facts"]["same_path_control"]["max_K_scoring"] == BOUND


@pytest.mark.parametrize("case", ["fa-1e-05", "adamw-1.32e-05"])
def test_step_6_is_incomplete(case) -> None:
    """Revision 3, 4, step 6: the control below, the path criterion holding, no pair exceeding,
    and a pair of FA 1e-5 or AdamW incomplete: no branch, and no expansion."""
    short = pair(values=[0.001] * 5, steps=EVAL_AT[:5])
    out = decide(*outputs(pairs={case: {"p2": short}}))
    assert (out["branch"], out["expand"], out["facts"]["step"]) == ("incomplete", None, 6)
    assert out["facts"]["pair_states"][case]["p2"] == "incomplete"
    unread = {case: ["p2"]}
    assert out["facts"]["incomplete"] == [f"step 6: the pairs {unread} are incomplete"]
    # Step 7 instead when a pair exceeds, the path criterion is false or the control is at the
    # bound; step 4 when the control exceeds
    other = "adamw-1.32e-05" if case == "fa-1e-05" else "fa-1e-05"
    assert branch(pairs={case: {"p2": short}, other: {"p4": 0.2}}) == ("ambiguous", "fa-1.46e-05")
    assert branch(path=False, pairs={case: {"p2": short}}) == ("ambiguous", "fa-1.46e-05")
    assert branch(same_path=BOUND, pairs={case: {"p2": short}}) == ("ambiguous", "fa-1.46e-05")
    assert branch(same_path=0.2, pairs={case: {"p2": short}}) == ("B", None)


def test_missing_outputs_leave_the_branch_incomplete() -> None:
    d1, determinism, cases = outputs()
    assert decide(None, determinism, cases)["branch"] == "incomplete"
    out = decide(d1, written({"equal": False}), {})
    assert out["branch"] == "incomplete" and any("determinism" in m for m in out["missing"])
    del cases["adamw-1.32e-05"]
    assert decide(d1, determinism, cases)["missing"] == ["d3/adamw-1.32e-05.json"]


def test_a_failed_determinism_check_wins_over_branch_a() -> None:
    """Review I4, probe 4: the determinism check comes before A (Revision 2, 9)."""
    d1, _, cases = outputs(pairs={"fa-1.21e-05": {"p1": 1.0}})
    out = decide(d1, written({"equal": False}), cases)
    assert out["branch"] == "incomplete"
    assert decide(d1, written({"equal": True}), cases)["branch"] == "A"


@pytest.mark.parametrize("case", ["fa-1.21e-05", "fa-1e-05", "adamw-1.32e-05"])
def test_a_case_needs_every_pair(case) -> None:
    """Review I4, probes 1 and 2: an empty pair set, or one without the reference, is no
    evidence that the pairs stay within the bound."""
    d1, determinism, cases = outputs()
    cases[case]["pairs"] = {}
    out = decide(d1, determinism, cases)
    assert out["branch"] == "incomplete" and any(case in m for m in out["missing"])
    d1, determinism, cases = outputs()
    del cases[case]["pairs"]["reference"]
    assert decide(d1, determinism, cases)["branch"] == "incomplete"


@pytest.mark.parametrize("case", ["fa-1.21e-05", "fa-1e-05", "adamw-1.32e-05"])
def test_a_twin_a_unlike_its_archive_makes_the_branch_incomplete(case) -> None:
    """Review I1, probe 3: no pair is read against a twin A that is not the archived run."""
    d1, determinism, cases = outputs()
    cases[case]["twin_A"]["archive"]["equal"] = False
    out = decide(d1, determinism, cases)
    assert out["branch"] == "incomplete" and out["facts"]["twin_A_equal"][case] is False
    assert any(case in reason for reason in out["facts"]["incomplete"])
    d1, determinism, cases = outputs()
    reason = "twin A differs from its archived run"
    cases[case] = written({"status": "incomplete", "reason": reason,
                           "twin_A": {"archive": {"equal": False}}, "pairs": {}})  # fmt: skip
    assert decide(d1, determinism, cases)["branch"] == "incomplete"


def test_a_recomputed_iii_unlike_its_record_makes_the_branch_incomplete() -> None:
    d1, determinism, cases = outputs()
    d1["residual"]["iii_equals_record"]["1.21e-05"] = False
    out = decide(d1, determinism, cases)
    assert out["branch"] == "incomplete"
    assert out["facts"]["iii_equals_record"]["1.21e-05"] is False


def test_outputs_of_another_commit_are_not_read() -> None:
    d1, determinism, cases = outputs()
    assert decide(d1, determinism, cases, commit=COMMIT)["branch"] == "F"
    out = decide(d1, determinism, cases, commit="d" * 40)  # HEAD moved on: nothing is current
    assert out["branch"] == "incomplete" and len(out["missing"]) == 5
    cases["fa-1e-05"] = written(cases["fa-1e-05"], "e" * 40)  # one part of another commit
    out = decide(d1, determinism, cases)
    assert out["branch"] == "incomplete"
    assert out["missing"] == [f"d3/fa-1e-05.json (written at {'e' * 40}, not {COMMIT})"]


def test_the_references_mode_is_reported() -> None:
    d1, determinism, cases = outputs()
    cases["adamw-1.32e-05"]["pairs"]["reference"].update(
        mode=fcd.FLOAT64_ACCUMULATION, fallback_reason="OutOfMemoryError: …"
    )
    references = decide(d1, determinism, cases)["facts"]["references"]
    assert references["fa-1.21e-05"] == {"mode": fcd.FLOAT64_FULL, "fallback_reason": None}
    assert references["adamw-1.32e-05"]["mode"] == fcd.FLOAT64_ACCUMULATION


def test_kept_outputs_must_come_from_this_commit(tmp_path) -> None:
    (tmp_path / "d3").mkdir()
    (tmp_path / "d3" / "fa-1e-05.json").write_text(json.dumps(written({"x": 1}, "a" * 40)))
    assert fcd.kept(tmp_path, "d3/fa-1e-05.json", "a" * 40) == written({"x": 1}, "a" * 40)
    assert fcd.kept(tmp_path, "d3/fa-1e-05.json", "b" * 40) is None
    assert not (tmp_path / "d3" / "fa-1e-05.json").exists()  # moved aside, kept
    assert json.loads((tmp_path / "stale" / ("a" * 40) / "d3" / "fa-1e-05.json").read_text())
    assert fcd.kept(tmp_path, "d3/fa-1e-05.json", "b" * 40) is None  # nothing left to read


def test_the_control_s_state_and_a_pair_s_series() -> None:
    """Review I2 and Revision 3, 3: a NaN never hides under ``max`` (it is a divergence); a
    missing step is incomplete, not unbounded."""
    steps = [1, 5, 10]

    def state(values, measured=steps, **record):
        scoring = [{"step": s, "K_scoring": v} for s, v in zip(measured, values, strict=True)]
        return fcd.control_state({"scoring": scoring, **record}, k_at=steps, bound=BOUND)

    for values in ([0.01, math.nan, 0.02], [math.nan, 0.01, 0.02], [0.01, 0.02, math.inf]):
        out = state(values)
        assert (out["state"], out["diverged"], out["complete"]) == ("exceeding", True, False)
        assert out["max_K_scoring"] == math.inf and out["nonfinite_steps"]
    good = state([0.01, 0.02, 0.03])
    assert (good["state"], good["max_K_scoring"], good["complete"]) == ("below", 0.03, True)
    assert state([0.01, 0.02], [1, 5])["state"] == "incomplete"
    assert state([0.01, 0.2], [1, 5])["state"] == "incomplete"  # above, on an incomplete series
    assert state([BOUND] * 3)["state"] == "at_bound"
    assert state([0.01, BOUND, 0.2])["state"] == "exceeding"
    assert state([0.01, 0.02, 0.03], stop="interrupted")["state"] == "incomplete"
    summary = fcd._pair_summary([{"step": 1, "D_twin": 0.01}, {"step": 5, "D_twin": math.nan},
                                 {"step": 10, "D_twin": 0.02}], steps, None)  # fmt: skip
    assert summary == {"max_D_twin": math.inf, "complete": False}
    assert fcd._pair_summary([], steps, "non-finite loss or gradient at step 0") == {
        "max_D_twin": None, "complete": False}  # fmt: skip


def test_a_failed_float64_reference_is_kept(monkeypatch) -> None:
    """Review M3: the out-of-memory attempt's rows and D_twin readings stay in the record."""

    def run_twin(cfg, twins, *, eval_at, precision, compare_dir, progress=None):
        if precision == "float64":
            progress.update(rows=[{"step": 0}], D_twin=[{"step": 1, "D_twin": 0.001}])
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 5 GiB\nmore")
        return {"steps": 25, "max_D_twin": 0.002}

    monkeypatch.setattr(fcd, "run_twin", run_twin)
    twins = type("T", (), {"env": type("E", (), {"device": CPU})(), "reset": lambda self: None})()
    out = fcd.reference_twin(None, twins, [1, 5], Path("."))
    assert out["mode"] == fcd.FLOAT64_ACCUMULATION and out["max_D_twin"] == 0.002
    failed = out["failed_attempt"]
    assert failed["rows"] == [{"step": 0}] and failed["D_twin"][0]["D_twin"] == 0.001
    assert out["fallback_reason"].startswith("OutOfMemoryError: CUDA out of memory")


def test_the_initial_states_are_built_from_each_run_s_own_v0() -> None:
    """Review M5: the optimizer state's v hash compares two estimates, not one twice."""
    params = [torch.nn.Parameter(torch.zeros(3))]
    cfg = OmegaConf.create({"optimizer": {"name": "adam_frozen", "lr": 1e-5, "beta1": 0.9,
                                          "eps": 1e-8, "weight_decay": 0.0}})  # fmt: skip
    same = fcd._initial_state((cfg, [torch.ones(3)]), (cfg, [torch.ones(3)]), params)
    assert same["equal_except_lr"] and same["regular"]["state_keys"] == ["exp_avg_sq"]
    other = fcd._initial_state((cfg, [torch.ones(3)]), (cfg, [torch.full((3,), 2.0)]), params)
    assert not other["equal_except_lr"]


def test_the_memory_rule_and_the_monitor_cache() -> None:
    rule = fcd.apply_memory_rule(CPU, 46)
    assert rule == {"limit_gb": 46.0, "fraction": None, "total_gb": None}
    assert fcd.numerics(CPU)["memory_rule"]["limit_gb"] == 46.0
    assert fcd.peak_memory(CPU) == {}
    cfg = OmegaConf.create({"eval": {"monitor": {"cache": {"primary": "none",
                                                           "secondary": "host"}}}})  # fmt: skip
    with pytest.raises(fcd.DiagnosticError, match="primary"):  # review M2
        fcd.operational_monitor(cfg)
    cfg.eval.monitor.cache.primary = "host"
    assert fcd.operational_monitor(cfg).eval.monitor.cache.secondary == "none"


def test_the_script_refuses_every_harmful_prompt_set() -> None:
    """Review M7: a runtime guard on the Pod, not only in the tests."""
    namespace = runpy.run_path(str(SCRIPT))
    env = Environment(tokenizer=None, end_ids=END_IDS, pad_id=PAD_ID, device=CPU, new_model=None,
                      prompt_set=lambda name: [name], data_root=Path("."))  # fmt: skip
    guarded = namespace["guard_prompt_sets"](env)
    assert guarded.prompt_set("alpaca_benign_kl") == ["alpaca_benign_kl"]
    for name in ("harmful_kl", "harmful_judged", "harmful_evaluation", "harmful_calibration"):
        with pytest.raises(RuntimeError, match="must not read"):
            guarded.prompt_set(name)


# ----------------------------------------------------------------------- the driver and plan


def namespace_of_script() -> dict:
    return runpy.run_path(str(SCRIPT))


def test_the_driver_programs(tmp_path, monkeypatch, capsys) -> None:
    """The driver's python -c programs: a field read back (the branch, D1's status), and D2's
    fallback failure record."""
    namespace = namespace_of_script()
    for program in (namespace["READ_FIELD"], namespace["D2_FAILURE_RECORD"]):
        assert not {"$", "`", "\\", '"'} & set(program)  # safe inside the driver's double quotes
    (tmp_path / "branch.json").write_text(json.dumps({"branch": "incomplete"}))
    (tmp_path / "result.json").write_text(json.dumps({"status": "complete", "reasons": []}))
    for name, key, value in (("branch.json", "branch", "incomplete"),
                             ("result.json", "status", "complete")):  # fmt: skip
        monkeypatch.setattr("sys.argv", ["-c", str(tmp_path / name), key])
        exec(namespace["READ_FIELD"], {})
        assert capsys.readouterr().out == f"{value}\n"
    monkeypatch.setattr("sys.argv", ["-c", str(tmp_path / "failed.json"), "137"])
    exec(namespace["D2_FAILURE_RECORD"], {})
    record = json.loads((tmp_path / "failed.json").read_text())
    assert record["part"] == "d2" and record["error"].startswith("exit 137")
    assert set(record["provenance"]["code"]) == {"commit", "dirty"}


def test_a_failed_expansion_is_recorded_beside_the_three_case_outcome(tmp_path, monkeypatch):
    from zero_field_refusal.tracking import git_state

    head = git_state()["commit"]
    d1, determinism, cases = outputs(path=False)  # ambiguous: the expansion is due
    records = {"d1/result.json": d1, "d3/determinism.json": determinism,
               **{f"d3/{name}.json": record for name, record in cases.items()}}  # fmt: skip
    for path, record in records.items():
        fcd.write_json(tmp_path / path, written(record, head))
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "branch", "--out", str(tmp_path),
                                     "--expansion-failed", "4"])  # fmt: skip
    runpy.run_path(str(SCRIPT), run_name="__main__")
    decision = json.loads((tmp_path / "branch" / "branch.json").read_text())
    assert (decision["branch"], decision["expand"]) == ("ambiguous", "fa-1.46e-05")
    assert decision["facts"]["expansion_failure"]["exit_code"] == 4


def test_the_driver(tmp_path, monkeypatch) -> None:
    commit = "a" * 40
    out = tmp_path / "driver.sh"
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "driver", "--commit", commit, "--out", str(out)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    subprocess.run(["bash", "-n", str(out)], check=True)
    text = out.read_text()
    assert f"SRC=/workspace/code/zero-field-refusal/{commit}/source" in text
    assert all(len(line) <= 100 for line in text.splitlines() if line.startswith("#"))
    run = 'project-run "$SRC" python scripts/fa_control_diagnostic.py'
    lines = text.splitlines()
    order = [f"step {run} plan", f"step {run} d1", f"{run} d2", f"{run} d3",
             f"{run} branch --expansion-needed", f"    {run} d3 --expansion",
             f"step {run} branch"]  # fmt: skip
    positions = [lines.index(line) for line in order]
    assert positions == sorted(positions)
    # Review M1: a failed determinism check (exit 2) still records the branch, then stops.
    d3 = lines.index(f"{run} d3")
    assert lines[d3 + 1 : d3 + 6] == ["rc=$?", "set -e", 'if [ "$rc" -ne 0 ]; then',
                                      f'  if [ "$rc" -eq 2 ]; then step {run} branch; fi',
                                      '  exit "$rc"']  # fmt: skip
    # Review r3 m1: an incomplete D1 records the branch (incomplete, with why) and stops with
    # exit 5 before D2; a complete one goes on.
    d1 = lines.index(f"step {run} d1")
    read = namespace_of_script()["READ_FIELD"]
    block = lines[d1 + 1 : lines.index("# D2: a failure is recorded and the driver goes on; no "
                                       "branch reads D2")]  # fmt: skip
    assert [line for line in block if not line.startswith("#")] == [
        f'd1_status=$(project-run "$SRC" python -c "{read}" "$OUT/d1/result.json" status)',
        'if [ "$d1_status" != complete ]; then',
        '  echo "== $(date -u +%FT%TZ) D1 is $d1_status: D2 and D3 do not run" >&2',
        f"  step {run} branch", "  exit 5", "fi"]  # fmt: skip
    stand_ins = ['project-run() { shift 2; "$PYTHON" "$@"; }', 'step() { echo "step: $*"; }']
    for status, code in (("incomplete", 5), ("complete", 0)):
        (tmp_path / "d1").mkdir(exist_ok=True)
        (tmp_path / "d1" / "result.json").write_text(json.dumps({"status": status}))
        program = "\n".join(["set -euo pipefail", *stand_ins, *block, "echo 'on to D2'"])
        done = subprocess.run(["bash", "-c", program], capture_output=True, text=True,
                              env={"PYTHON": sys.executable, "OUT": str(tmp_path), "SRC": "."},
                              check=False)  # fmt: skip
        assert done.returncode == code, done.stderr
        assert ("step: " in done.stdout and "branch" in done.stdout) == (status == "incomplete")
        assert ("on to D2" in done.stdout) == (status == "complete")
    # Re-review N2: a D2 failure is recorded (by D2, else here) and D3 goes on.
    d2 = lines.index(f"{run} d2")
    assert lines[d2 + 1 : d2 + 4] == ["d2=$?", "set -e", 'if [ "$d2" -ne 0 ]; then']
    record = namespace_of_script()["D2_FAILURE_RECORD"]
    assert lines[d2 + 7] == (f'    project-run "$SRC" python -c "{record}" '
                             '"$OUT/d2/failed.json" "$d2"')  # fmt: skip
    stops = [line for line in lines[d2 + 1 : d3]
             if line.strip().startswith("exit") or "; exit" in line]  # fmt: skip
    assert not stops  # nothing stops between D2 and D3
    # Re-review N1: a failed expansion records the branch (the three cases'), then stops; the
    # final branch is printed, and an incomplete one, or a D2 failure, ends non-zero.
    expansion = lines.index(f"    {run} d3 --expansion")
    assert lines[expansion + 1 : expansion + 6] == [
        "    rc=$?", "    set -e", '    if [ "$rc" -ne 0 ]; then',
        f'      step {run} branch --expansion-failed "$rc"', '      exit "$rc"']  # fmt: skip
    final = positions[-1]
    read = namespace_of_script()["READ_FIELD"]
    assert lines[final + 1 : final + 4] == [
        f'outcome=$(project-run "$SRC" python -c "{read}" "$OUT/branch/branch.json" branch)',
        'echo "== $(date -u +%FT%TZ) the branch: $outcome"',
        'if [ "$outcome" = incomplete ]; then exit 5; fi']  # fmt: skip
    assert lines[final + 4].startswith('if [ "$d2" -ne 0 ]; then') and "exit 6" in lines[final + 4]
    assert lines[-1].endswith('the diagnostic is complete: $OUT"')
    # Review M8 and re-review N3: a free GPU (no compute process, under 1000 MiB in use), the
    # disk for D3's checkpoints, and a log in the output directory.
    gpu = lines.index('if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; '
                      "then")  # fmt: skip
    used = lines.index("used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)")
    assert lines[used + 1] == ("if echo \"$used\" | awk '$1 + 0 >= 1000 {busy = 1} END "
                               "{exit !busy}'; then")  # fmt: skip
    assert lines[used + 3] == "  exit 1"
    disk = next(i for i, line in enumerate(lines) if line.startswith('if [ "$free_kb" -lt 40'))
    log = lines.index('exec > >(tee -a "$OUT/driver-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1')
    assert log < gpu < used < disk < positions[0]
    for gpu_memory, busy in (("0", False), ("999", False), ("1000", True), ("12\n46000", True)):
        check = subprocess.run(["awk", "$1 + 0 >= 1000 {busy = 1} END {exit !busy}"],
                               input=gpu_memory, text=True, check=False)  # fmt: skip
        assert (check.returncode == 0) == busy
    assert 'OUT="${RUNS_DIR:?}/diagnostics/fa_control"' in text
    assert "harmful_" not in text
    # The checkout is clean as tracking.git_state reads it, whatever the Git configuration (the
    # check the Pod prep driver uses, shared), not as `git status` shows it.
    head = lines.index(f'test "$(git rev-parse HEAD)" = {commit}')
    assert lines[head + 1] == f'project-run "$SRC" python -c "{CLEAN_CHECKOUT}"'
    assert head + 1 < log and "git status" not in text
    for dirty in (False, True, None):  # None: no checkout
        monkeypatch.setattr("zero_field_refusal.tracking.git_state",
                            lambda dirty=dirty: {"commit": commit, "dirty": dirty})  # fmt: skip
        with pytest.raises(SystemExit) as stopped:
            exec(CLEAN_CHECKOUT, {})
        assert (stopped.value.code is None) == (dirty is False)


# ----------------------------------------------------------------- the whole diagnostic, tiny

STEPS = 6
RATES = {"fa-1.21e-05": 1e-3, "fa-1e-05": 8e-4, "fa-1.46e-05": 1.2e-3, "adamw-1.32e-05": 1e-3}
CONTROLS = [9e-4, 1e-3, 1.2e-3]


def tiny_settings():
    cases = {
        "fa-1.21e-05": {"optimizer": "adam_frozen", "lr": 1e-3, "control_lr": 1e-3},
        "fa-1e-05": {"optimizer": "adam_frozen", "lr": 8e-4, "control_lr": 9e-4},
        "adamw-1.32e-05": {"optimizer": "adamw", "lr": 1e-3, "control_lr": 1.1e-3},
        "fa-1.46e-05": {"optimizer": "adam_frozen", "lr": 1.2e-3, "control_lr": 1.2e-3},
    }
    return settings(diagnostic={
        "seed": 0, "batch_size": 4, "controls": CONTROLS,
        "d1": {"same_path_control": {"rate": 1e-3, "k_at": [1, 3, 6]}},
        "d2": {"batches": 2, "rates": [9e-4, 1e-3], "lanczos": {"steps": 4},
               "fisher_at": [1, 3, 6]},
        "d3": {"eval_at": [1, 3, 6], "cases": cases},
    })  # fmt: skip


def tiny_run_config(tmp_path, *, optimizer, lr, control):
    experiment = tmp_path / f"{optimizer}-{control}.yaml"
    defaults = ["experiments/smoke", "eval/study2", "schedule/short"]
    if control:
        defaults.append("control/soft_targets")
    experiment.write_text(f"defaults: [{', '.join(defaults)}]\n")
    name = "study2_short_fa_controls" if control else "study2_short"
    overrides = [*SMALL, *APPENDABLE, "teacher.chunk_size=12", f"train.steps={STEPS}",
                 f"schedule.evaluate_at=[{STEPS}]", "+schedule.instability.grad_rule=log",
                 f"experiment={name}"]  # fmt: skip
    if optimizer == "adam_frozen":
        overrides += ["optimizer=adam_frozen", "optimizer.preconditioner.batches=2"]
    return load_config(experiment, [*overrides, f"optimizer.lr={lr}"])


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """Tiny regular runs of every case and FA soft controls, archived as the stage's are, with
    the environment of tests/test_train.py's ``env`` fixture; built once for the module."""
    base = tmp_path_factory.mktemp("fa-diagnostic")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RUNS_DIR", str(base / "runs"))
        tokenizer = tiny_tokenizer()
        end_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
        sets = prompt_sets()

        def new_model():
            model = tiny_model(tied=False, head_std=0.02)
            install_neutral_generation_config(model, end_ids, tokenizer.pad_token_id)
            return model

        env = Environment(tokenizer=tokenizer, end_ids=end_ids, pad_id=tokenizer.pad_token_id,
                          device=CPU, new_model=new_model, prompt_set=lambda name: sets[name],
                          data_root=base / "data")  # fmt: skip
        pass_path_check(env)
        first = tiny_run_config(base, optimizer="adam_frozen", lr=1e-3, control=False)
        prepare(env, first, "kl_monitor", "preconditioner")
        for name, lr in RATES.items():
            optimizer = "adamw" if name.startswith("adamw") else "adam_frozen"
            run_scheduled(tiny_run_config(base, optimizer=optimizer, lr=lr, control=False), env,
                          run_id=f"regular-{name}")  # fmt: skip
        for lr in CONTROLS:
            run_scheduled(tiny_run_config(base, optimizer="adam_frozen", lr=lr, control=True),
                          env, run_id=f"control-{lr:g}")  # fmt: skip
    return env, base / "runs"


@pytest.fixture
def world(archive, tmp_path, monkeypatch):
    """A copy of the archived runs for one test, which may write or alter it."""
    env, source = archive
    root = tmp_path / "runs"
    shutil.copytree(source, root)
    monkeypatch.setenv("RUNS_DIR", str(root))
    return env, root


def forbid_outcomes(monkeypatch, env):
    """Harmful generation, the evaluator and refusal scoring raise; the prompt sets requested
    are recorded."""
    from zero_field_refusal import evaluation, refusal, train

    def forbidden(*args, **kwargs):
        raise AssertionError("the diagnostic must not generate on harmful prompts or score them")

    targets = ((evaluation, "greedy_responses"), (train, "greedy_responses"),
               (refusal, "is_keyword_refusal"), (train, "is_keyword_refusal"))  # fmt: skip
    for module, name in targets:
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(train.Evaluator, "__init__", forbidden)
    requested = []
    sets = env.prompt_set

    def prompt_set(name):
        requested.append(name)
        if "harmful" in name:
            raise AssertionError(f"the diagnostic requested the harmful prompt set {name}")
        return sets(name)

    return dataclasses.replace(env, prompt_set=prompt_set), requested


def test_the_whole_diagnostic_on_tiny_runs_without_harmful_generation(
    world, tmp_path, monkeypatch, capsys
) -> None:
    env, root = world
    s = tiny_settings()
    runs = fcd.locate_runs(s, root)
    assert sorted(runs.regular) == sorted(RATES) and len(runs.controls) == 3
    guarded, requested = forbid_outcomes(monkeypatch, env)
    out = fcd.output_root(root)

    found, missing = fcd.check_inputs(s, guarded, runs)  # the streams D1-D3 read
    assert len(found) == 4 and not missing
    fcd.apply_memory_rule(CPU, s.diagnostic.memory_rule_gb)  # recorded only, on the CPU

    d1 = fcd.run_d1(s, guarded, runs, out)
    assert d1["status"] == "complete" and d1["numerics"]["memory_rule"] is not None
    assert "reasons" not in d1
    shared = d1["shared_state"]
    assert shared["v0"]["equal"] and shared["trainable_parameters"]["equal"]
    assert shared["loss_normalization"]["equal"]
    assert shared["initial_optimizer_state"]["equal_except_lr"]
    assert shared["initial_optimizer_state"]["regular"]["state_keys"] == ["exp_avg_sq"]
    residual = d1["residual"]
    assert all(residual["iii_equals_record"].values())  # the recorded first gradient, bit for bit
    assert residual["path_criterion"]["state"] in ("holds", "false")  # decided: finite norms
    assert residual["ratio_i_to_iii"] == residual["path_criterion"]["ratio_i_to_iii"]
    assert residual["norms"]["ii_training_path_float64"] < residual["norms"]["iii_control_recorded"]
    assert set(residual["cosines"]) == {
        "i_training_path_fp32|ii_training_path_float64",
        "i_training_path_fp32|iii_control_recorded",
        "ii_training_path_float64|iii_control_recorded",
    }
    control = d1["same_path_control"]
    assert control["steps"] == STEPS and [r["step"] for r in control["scoring"]] == [1, 3, 6]
    assert [r["step"] for r in control["monitor"]] == [5]
    assert control["K_star"] == pytest.approx(math.sqrt(40)) and not control["nonfinite_steps"]
    assert control["state"] in ("below", "exceeding", "at_bound") and control["complete"]
    assert control["diverged"] is False and control["stop"] is None
    assert "all" in d1["v0_census"]["groups"] and "attn_q" in d1["v0_census"]["groups"]

    d2 = fcd.run_d2(s, guarded, runs, out)
    assert d2["v0_sha256"] == shared["v0"]["regular_sha256"]
    assert d2["hessians"]["soft"].startswith("(a) exact soft-loss Hessian")
    assert {r["ritz_vector_on"] for k in ("soft", "nll") for r in d2["spectra"][k]} == {"host"}
    assert [len(d2["spectra"][k]) for k in ("soft", "nll")] == [2, 2]
    assert set(d2["linearized"]) == {"soft|0.0009", "soft|0.001", "nll|0.0009", "nll|0.001"}
    first = d2["linearized"]["soft|0.001"]
    assert len(first["delta_norm"]) == STEPS and set(first["delta_F_delta_relative"]) == {1, 3, 6}
    assert d2["thresholds"]["0.001"]["heavy_ball_lambda"] == pytest.approx(38 / 1e-3)
    assert set(d2["observed"]) == {"0.0009", "0.001", "0.0012"}

    order = list(s.diagnostic.d3.order)
    done = fcd.run_d3(s, guarded, runs, out, cases=order)
    determinism = json.loads((out / "d3" / "determinism.json").read_text())
    assert determinism["equal"] and determinism["steps_compared"] == STEPS
    for name in order:
        case = done[name]
        assert case["status"] == "complete" and case["K_star"] == pytest.approx(math.sqrt(40))
        assert case["twin_A"]["archive"]["equal"]  # each twin A is its archived run, bitwise
        assert set(case["pairs"]) == {"p1", "p2", "p3", "p4", "reference"}
        for key, pair in case["pairs"].items():
            assert [r["step"] for r in pair["D_twin"]] == [1, 3, 6], key
            assert pair["complete"] and math.isfinite(pair["max_D_twin"])
            assert pair["diverged"] is False and pair["measured_every_step"]
            assert pair["state"] == case["pair_states"][key] != "incomplete", key
            reading = pair["D_twin"][-1]
            assert reading["D_twin_over_K_star"] == pytest.approx(reading["D_twin"] / math.sqrt(40))
        assert case["pairs"]["p1"]["rounding"][0]["survived"]
        for key in ("p1", "p2", "p3", "p4"):
            attempts = case["pairs"][key]["rounding"]
            assert len(attempts) == (1 if attempts[0]["survived"] else 2), key
            for a in attempts:
                assert a["survived"] == (a["realized_norm"] >= 0.5 * a["intended_norm"])
            assert case["pairs"][key]["survived"] is attempts[-1]["survived"] is True, key
        assert case["pairs"]["reference"]["survived"] is None  # no perturbation
        assert case["pairs"]["reference"]["mode"].startswith("float64 loss, accumulation and")
        assert not list((out / "d3" / "checkpoints" / name).glob("*.pt"))  # deleted
        p1 = case["pairs"]["p1"]["rounding"][0]  # θ₀ + δ₁ is the control's θ₁ exactly
        assert p1["realized_norm"] == pytest.approx(case["first_step_displacement_norm"], rel=1e-6)
        assert p1["changed"] == case["first_step_changed"]
    updates = [done[name]["first_step_update_norm"] for name in order]
    assert updates[0] / updates[1] == pytest.approx(1e-3 / 9e-4)  # η‖P r‖, ∝ the control's rate

    decision = fcd.decide_branch(s, *fcd.load_outputs(out))
    assert decision["branch"] in fcd.BRANCHES and not decision["missing"]
    assert decision["branch"] != "incomplete" and not decision["facts"]["incomplete"]
    assert set(decision["facts"]["references"]) == set(order)
    facts = decision["facts"]
    assert facts["shared_state"] == dict.fromkeys(fcd.SHARED_STATE_CHECKS, True)
    assert facts["pair_states"] == {name: done[name]["pair_states"] for name in order}
    assert facts["same_path_control_state"] == control["state"]
    if decision["expand"]:
        fcd.run_d3(s, guarded, runs, out, cases=[decision["expand"]])
        again = fcd.decide_branch(s, *fcd.load_outputs(out))
        assert (again["branch"], again["expand"]) == ("ambiguous", None)
    assert requested and not [name for name in requested if "harmful" in name]

    # A part already written is kept: the script skips it, and the plan finds the runs.
    config = tmp_path / "settings.yaml"
    OmegaConf.save(s, config)
    for argv in (["d1"], ["d3"], ["plan", "--no-stream-check"], ["d3", "--dry-run"], ["branch"]):
        monkeypatch.setattr("sys.argv", [str(SCRIPT), *argv, "--config", str(config),
                                         "--runs-root", str(root)])  # fmt: skip
        runpy.run_path(str(SCRIPT), run_name="__main__")
    printed = capsys.readouterr().out
    assert printed.count("kept:") == 2 and "total ≈" in printed
    assert json.loads((out / "branch" / "branch.json").read_text())["branch"] in fcd.BRANCHES


def test_a_run_that_differs_from_its_archive_stops_d3(world) -> None:
    env, root = world
    s = tiny_settings()
    runs = fcd.locate_runs(s, root)
    metrics = runs.regular["fa-1e-05"] / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    next(r for r in rows if "loss" in r)["grad_norm"] *= 1 + 1e-12
    metrics.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = fcd.output_root(root)
    with pytest.raises(fcd.DeterminismFailure, match="differs from its archive"):
        fcd.run_d3(s, env, runs, out, cases=list(s.diagnostic.d3.order))
    record = json.loads((out / "d3" / "determinism.json").read_text())
    assert not record["equal"] and record["mismatches"][0]["key"] == "grad_norm"
    assert not list((out / "d3").glob("fa-*.json"))
    decision = fcd.decide_branch(s, *fcd.load_outputs(out))
    assert decision["branch"] == "incomplete"


def test_a_doctored_archive_or_record_makes_the_outcome_incomplete(world, monkeypatch) -> None:
    """Review I1: twin A of FA 1.21e-5 against a doctored archive makes that case incomplete,
    with no pair run; a doctored control record makes D1's (iii) check fail."""
    env, root = world
    s = tiny_settings()
    runs = fcd.locate_runs(s, root)
    for path in (runs.regular["fa-1.21e-05"] / "metrics.jsonl",
                 runs.control(1.2e-3) / "metrics.jsonl"):  # fmt: skip
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        next(r for r in rows if "loss" in r)["grad_norm"] *= 1 + 1e-12
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = fcd.output_root(root)
    done = fcd.run_d3(s, env, runs, out, cases=["fa-1.21e-05"])
    case = done["fa-1.21e-05"]
    assert case["status"] == "incomplete" and case["pairs"] == {}
    assert not case["twin_A"]["archive"]["equal"]
    assert not list((out / "d3" / "checkpoints" / "fa-1.21e-05").glob("*"))
    monkeypatch.setattr(fcd, "same_path_control", lambda *args: {"below_bound": True})
    d1 = fcd.run_d1(s, env, runs, out)
    assert d1["status"] == "incomplete" and d1["residual"]["iii_equals_record"] == {
        "0.0009": True,
        "0.001": True,
        "0.0012": False,
    }
    assert d1["reasons"] == [f"{fcd.SHARED_STATE_CHECKS['iii_equals_record']} (the controls "
                             "['0.0012'])"]  # fmt: skip
    decision = fcd.decide_branch(s, *fcd.load_outputs(out))
    assert decision["branch"] == "incomplete"
    reasons = " ".join(decision["facts"]["incomplete"])
    assert "fa-1.21e-05" in reasons and "(iii)" in reasons


def test_d1_is_incomplete_when_a_shared_state_check_fails(world, monkeypatch) -> None:
    """Revision 3, 1, end to end: D1 with unequal initial optimizer states (as when the two
    configs' v₀ estimates differ) records itself incomplete, and so is the outcome."""
    env, root = world
    s = tiny_settings()
    runs = fcd.locate_runs(s, root)
    initial_state = fcd._initial_state

    def unequal(*args, **kwargs):
        return {**initial_state(*args, **kwargs), "equal_except_lr": False}

    monkeypatch.setattr(fcd, "_initial_state", unequal)
    monkeypatch.setattr(fcd, "same_path_control", lambda *args: fcd.control_state(
        {"scoring": [{"step": t, "K_scoring": 0.0} for t in (1, 3, 6)]}, k_at=[1, 3, 6],
        bound=BOUND))  # fmt: skip
    out = fcd.output_root(root)
    d1 = fcd.run_d1(s, env, runs, out)
    assert d1["status"] == "incomplete" and all(d1["residual"]["iii_equals_record"].values())
    assert d1["reasons"] == [fcd.SHARED_STATE_CHECKS["initial_optimizer_state"]]
    assert d1["residual"]["path_criterion"]["state"] in ("holds", "false")
    decision = fcd.decide_branch(s, d1, {"equal": True, "provenance": d1["provenance"]}, {})
    assert decision["branch"] == "incomplete"
    assert decision["facts"]["shared_state"]["initial_optimizer_state"] is False


def test_a_d2_failure_is_recorded_and_kept(world, tmp_path, monkeypatch) -> None:
    """Re-review N2: the script records D2's failure before it exits; a later success keeps
    that record under another name."""
    env, root = world
    config = tmp_path / "settings.yaml"
    OmegaConf.save(tiny_settings(), config)
    monkeypatch.setattr("zero_field_refusal.runtime.environment", lambda cfg, device: env)
    argv = [str(SCRIPT), "d2", "--config", str(config), "--runs-root", str(root)]
    monkeypatch.setattr("sys.argv", argv)

    def failing(*args, **kwargs):
        raise RuntimeError("the products failed")

    monkeypatch.setattr(fcd, "run_d2", failing)
    with pytest.raises(RuntimeError, match="the products failed"):
        runpy.run_path(str(SCRIPT), run_name="__main__")
    d2 = fcd.output_root(root) / "d2"
    record = json.loads((d2 / "failed.json").read_text())
    assert record["part"] == "d2" and record["error"] == "RuntimeError: the products failed"
    assert "Traceback" in record["traceback"] and record["provenance"]["code"]["commit"]
    monkeypatch.setattr(fcd, "run_d2", lambda *args, **kwargs: {"spectra": {}})
    runpy.run_path(str(SCRIPT), run_name="__main__")
    assert not (d2 / "failed.json").exists() and len(list(d2.glob("failed-before-*.json"))) == 1


def test_the_plan_checks_the_streams(world) -> None:
    env, root = world
    s = tiny_settings()
    runs = fcd.locate_runs(s, root)
    found, missing = fcd.check_inputs(s, env, runs)
    assert not missing
    monitor = next(f for f in found if f.startswith("kl_monitor")).split(" stream ")[1]
    (Path(monitor) / "manifest.json").rename(Path(monitor) / "manifest.json.aside")
    try:
        _, missing = fcd.check_inputs(s, env, runs)
        assert [m.split(" ")[0] for m in missing] == ["kl_monitor"]
    finally:
        (Path(monitor) / "manifest.json.aside").rename(Path(monitor) / "manifest.json")


def test_runs_are_found_by_their_archived_configs(world) -> None:
    _, root = world
    s = tiny_settings()
    with pytest.raises(fcd.DiagnosticError, match="0 finished runs"):
        fcd.locate_runs(settings(diagnostic={"seed": 0, "batch_size": 4}), root)
    copy = root / "study2_short" / "regular-copy"
    source = fcd.locate_runs(s, root).regular["fa-1e-05"]
    copy.mkdir()
    for name in ("config.yaml", "terminal.json"):
        (copy / name).write_text((source / name).read_text())
    with pytest.raises(fcd.DiagnosticError, match="2 finished runs"):
        fcd.locate_runs(s, root)
