import pytest
import torch
from omegaconf import OmegaConf

from zero_field_refusal.config import load_config
from zero_field_refusal.optim import (
    SECOND_MOMENT_MODES,
    Adam,
    build_optimizer,
    estimate_second_moment,
)
from zero_field_refusal.schedule import UpdateScalars

LR, B1, B2, EPS = 1e-2, 0.9, 0.999, 1e-8


def params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(5, 3, generator=g)),
        torch.nn.Parameter(torch.randn(4, generator=g)),
    ]


def gradients(steps, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [[torch.randn(5, 3, generator=g), torch.randn(4, generator=g)] for _ in range(steps)]


def run(optimizer, ps, grads):
    for step in grads:
        for p, grad in zip(ps, step, strict=True):
            p.grad = grad.clone()
        optimizer.step()


def test_adaptive_mode_reproduces_torch_adamw() -> None:
    ours, reference = params(), params()
    grads = gradients(30)
    run(Adam(ours, lr=LR, betas=(B1, B2), eps=EPS), ours, grads)
    torch_adamw = torch.optim.AdamW(
        reference, lr=LR, betas=(B1, B2), eps=EPS, weight_decay=0.0, foreach=False
    )
    run(torch_adamw, reference, grads)
    for a, b in zip(ours, reference, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)


def test_frozen_mode_divides_by_the_fixed_estimate_without_bias_correction() -> None:
    ps = params()
    start = [p.detach().clone() for p in ps]
    v_hat = [torch.rand_like(p) + 0.5 for p in ps]
    grads = gradients(2)
    optimizer = Adam(ps, lr=LR, betas=(B1, None), eps=EPS, second_moment="frozen",
                     initial_second_moment=v_hat)  # fmt: skip
    run(optimizer, ps, grads)
    for i, p in enumerate(ps):
        m1 = (1 - B1) * grads[0][i]
        p1 = start[i] - LR / (1 - B1) * m1 / (v_hat[i].sqrt() + EPS)
        m2 = B1 * m1 + (1 - B1) * grads[1][i]
        p2 = p1 - LR / (1 - B1**2) * m2 / (v_hat[i].sqrt() + EPS)
        torch.testing.assert_close(p.detach(), p2, rtol=1e-6, atol=1e-7)
        assert torch.equal(optimizer.state[p]["exp_avg_sq"], v_hat[i])  # never updated


def test_prewarmed_mode_updates_the_estimate_without_bias_correction() -> None:
    ps = params()
    start = [p.detach().clone() for p in ps]
    v_hat = [torch.rand_like(p) + 0.5 for p in ps]
    grads = gradients(2)
    optimizer = Adam(ps, lr=LR, betas=(B1, B2), eps=EPS, second_moment="prewarmed",
                     initial_second_moment=v_hat)  # fmt: skip
    run(optimizer, ps, grads)
    for i, p in enumerate(ps):
        m1 = (1 - B1) * grads[0][i]
        v1 = B2 * v_hat[i] + (1 - B2) * grads[0][i] ** 2
        p1 = start[i] - LR / (1 - B1) * m1 / (v1.sqrt() + EPS)
        m2 = B1 * m1 + (1 - B1) * grads[1][i]
        v2 = B2 * v1 + (1 - B2) * grads[1][i] ** 2
        p2 = p1 - LR / (1 - B1**2) * m2 / (v2.sqrt() + EPS)
        torch.testing.assert_close(p.detach(), p2, rtol=1e-6, atol=1e-7)
    assert optimizer.state[ps[0]]["step"] == 2


def test_state_is_fp32_and_initial_estimate_is_copied() -> None:
    ps = params()
    v_hat = [torch.ones_like(p) for p in ps]
    optimizer = Adam(ps, lr=LR, betas=(B1, B2), second_moment="prewarmed",
                     initial_second_moment=v_hat)  # fmt: skip
    run(optimizer, ps, gradients(1))
    state = optimizer.state[ps[0]]
    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    assert torch.equal(v_hat[0], torch.ones_like(v_hat[0]))  # the caller's estimate is untouched


@pytest.mark.parametrize("mode", ["frozen", "prewarmed"])
def test_frozen_and_prewarmed_modes_need_an_estimate(mode: str) -> None:
    with pytest.raises(ValueError, match="initial_second_moment"):
        Adam(params(), lr=LR, betas=(B1, B2), second_moment=mode)


def test_estimate_is_the_mean_of_squared_batch_gradients() -> None:
    ps = params()
    grads = gradients(5)

    def batch_gradient(index: int) -> None:
        for p, grad in zip(ps, grads[index], strict=True):
            p.grad = grad.clone()

    estimate = estimate_second_moment(ps, batch_gradient, num_batches=5)
    for i, v in enumerate(estimate):
        expected = torch.stack([step[i] ** 2 for step in grads]).mean(0)
        torch.testing.assert_close(v, expected)


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("sgd", torch.optim.SGD),
        ("sgd_momentum", torch.optim.SGD),
        ("adamw", Adam),
        ("adam_frozen", Adam),
        ("adamw_prewarmed", Adam),
    ],
)
def test_build_optimizer_from_repository_configs(name: str, kind: type) -> None:
    cfg = load_config("experiments/smoke", [f"optimizer={name}", "optimizer.lr=0.5"])
    ps = params()
    estimate = [torch.ones_like(p) for p in ps]
    optimizer = build_optimizer(cfg.optimizer, ps, second_moment=estimate)
    assert isinstance(optimizer, kind)
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.5 and group.get("weight_decay", 0.0) == 0.0
    if name.startswith("sgd"):
        assert group["momentum"] == cfg.optimizer.momentum
    else:
        assert optimizer.second_moment == {
            "adamw": "adaptive", "adam_frozen": "frozen", "adamw_prewarmed": "prewarmed"
        }[name]  # fmt: skip


def test_build_optimizer_rejects_weight_decay_and_unknown_names() -> None:
    with pytest.raises(ValueError, match="weight_decay"):
        build_optimizer(OmegaConf.create({"name": "sgd", "lr": 1.0, "momentum": 0.0,
                                          "weight_decay": 0.1}), params())  # fmt: skip
    with pytest.raises(ValueError, match="lion"):
        build_optimizer(OmegaConf.create({"name": "lion", "lr": 1.0, "weight_decay": 0.0}),
                        params())  # fmt: skip


def test_build_optimizer_rejects_non_fp32_state() -> None:
    cfg = load_config("experiments/smoke", ["optimizer.first_moment_dtype=bfloat16"])
    with pytest.raises(ValueError, match="first_moment_dtype"):
        build_optimizer(cfg.optimizer, params())


def adam(ps, mode):
    betas = (B1, None) if mode == "frozen" else (B1, B2)
    v_hat = None if mode == "adaptive" else [torch.rand_like(p) + 0.5 for p in ps]
    return Adam(ps, lr=LR, betas=betas, eps=EPS, second_moment=mode,
                initial_second_moment=v_hat)  # fmt: skip


@pytest.mark.parametrize("mode", SECOND_MOMENT_MODES)
def test_the_denominator_and_step_size_are_those_the_step_applied(mode: str) -> None:
    # The read accessors behind the update scalars: from the state after each update they give
    # back the update itself, bit for bit, whole or by coordinates of the flattened parameter.
    ps = params()
    optimizer = adam(ps, mode)
    for step in gradients(3):
        before = [p.detach().clone() for p in ps]
        run(optimizer, ps, [step])
        for p, start in zip(ps, before, strict=True):
            d = optimizer.denominator(p)
            m = optimizer.state[p]["exp_avg"]
            assert torch.equal(start.addcdiv(m, d, value=-optimizer.step_size(p)), p.detach())
            assert torch.equal(optimizer.denominator(p, slice(1, 3)), d.reshape(-1)[1:3])
            assert d.data_ptr() != optimizer.state[p]["exp_avg_sq"].data_ptr()  # a new tensor


# Hand-computed (fp64, from the formulas of the three modes): one parameter of three coordinates,
# lr 0.1, β₁ 0.5, β₂ 0.75, ε 0.5, gradients g₁ = (2, −1, 0.5) and g₂ = (−2, 3, 1), and for the
# frozen and pre-warmed modes v₀ = (1, 4, 0.25). In every mode m₁ = 0.5·g₁ = (1, −0.5, 0.25),
# m₂ = 0.5·m₁ + 0.5·g₂ = (−0.5, 1.25, 0.625), and u_t = (0.1 / (1 − 0.5ᵗ))·m_t / d_t.
# - adaptive: v₁ = 0.25·g₁² = (1, 0.25, 0.0625), d₁ = √(v₁ / 0.25) + 0.5 = (2.5, 1.5, 1),
#   u₁ = 0.2·m₁ / d₁ = (0.08, −1/15, 0.05); v₂ = 0.75·v₁ + 0.25·g₂² = (1.75, 2.4375, 0.296875),
#   d₂ = √(v₂ / 0.4375) + 0.5 = (2.5, 2.86039, 1.32375), u₂ = (2/15)·m₂ / d₂.
# - frozen: d = √v₀ + 0.5 = (1.5, 2.5, 1) at both steps, u₁ = (2/15, −0.04, 0.05),
#   u₂ = (2/15)·m₂ / d = (−2/45, 1/15, 1/12).
# - pre-warmed: v₁ = 0.75·v₀ + 0.25·g₁² = (1.75, 3.25, 0.25), d₁ = √v₁ + 0.5 =
#   (1.82288, 2.30278, 1); v₂ = 0.75·v₁ + 0.25·g₂² = (2.3125, 4.6875, 0.4375), d₂ = √v₂ + 0.5.
#   Against d₀ = (1.5, 2.5, 1): |d₁/d₀ − 1| = (0.215250, 0.078890, 0) and |d₂/d₀ − 1| =
#   (0.347127, 0.066025, 0.161438); the median is the second smallest of the three, the 90th
#   percentile the largest.
# Each row: ‖u_t‖, cos(u_t, g_t), and for the pre-warmed mode ‖d_t − d₀‖/‖d₀‖, median, p90.
HAND = {
    "adaptive": [(0.11551815634108971, 0.9508145001997473),
                 (0.08982850653347269, 0.8660535526858691)],
    "frozen": [(0.14791138488222527, 0.9786346104356113),
               (0.11560362248158229, 0.8605316273230326)],
    "prewarmed": [(0.1281544808847305, 0.9803178536812223,
                   0.1227518412370991, 0.07888974490720213, 0.21525043702153024),
                  (0.10073483908669542, 0.8631833608349878,
                   0.18479771849820742, 0.16143782776614768, 0.3471270883830364)],
}  # fmt: skip
HISTOGRAM_BIN = 10**0.005  # a quantile is read at the upper edge of its 0.005-decade bin


@pytest.mark.parametrize("mode", SECOND_MOMENT_MODES)
def test_update_scalars_values(mode: str) -> None:
    p = torch.nn.Parameter(torch.zeros(3))
    betas = (0.5, None) if mode == "frozen" else (0.5, 0.75)
    v0 = None if mode == "adaptive" else [torch.tensor([1.0, 4.0, 0.25])]
    optimizer = Adam([p], lr=0.1, betas=betas, eps=0.5, second_moment=mode,
                     initial_second_moment=v0)  # fmt: skip
    scalars = UpdateScalars(optimizer, departure_at=(1, 2))  # the departure: pre-warmed only
    grads = [[2.0, -1.0, 0.5], [-2.0, 3.0, 1.0]]
    for t, (g, hand) in enumerate(zip(grads, HAND[mode], strict=True), 1):
        p.grad = torch.tensor(g)
        optimizer.step()
        row = scalars(t)
        assert row["step"] == t
        assert row["update_norm"] == pytest.approx(hand[0], rel=1e-6)  # fp32 state
        assert row["cos_update_grad"] == pytest.approx(hand[1], rel=1e-6)
        if mode != "prewarmed":
            assert set(row) == {"step", "update_norm", "cos_update_grad"}
            continue
        departure = row["d_departure"]
        assert set(departure) == {"norm_ratio", "median", "p90"}
        assert departure["norm_ratio"] == pytest.approx(hand[2], rel=1e-6)
        assert hand[3] * (1 - 1e-6) <= departure["median"] <= hand[3] * HISTOGRAM_BIN
        assert departure["p90"] == pytest.approx(hand[4], rel=1e-6)  # the largest: exact


def test_departure_zero_at_start() -> None:
    # d₀ = √v₀ + ε from the host copy of v₀ is the pre-warmed denominator before the first
    # update, bit for bit, so a departure read then is 0.
    ps = params()
    optimizer = adam(ps, "prewarmed")
    scalars = UpdateScalars(optimizer)
    assert len(scalars.v0) == len(ps)
    for p, v0 in zip(ps, scalars.v0, strict=True):
        assert v0.device.type == "cpu" and v0.dtype == torch.float32
        assert v0.data_ptr() != optimizer.state[p]["exp_avg_sq"].data_ptr()
        assert torch.equal(optimizer.denominator(p, second_moment=v0), optimizer.denominator(p))
    assert scalars.departure() == {"norm_ratio": 0.0, "median": 0.0, "p90": 0.0}
    copies = [v0.clone() for v0 in scalars.v0]
    run(optimizer, ps, gradients(1))
    assert all(torch.equal(a, b) for a, b in zip(copies, scalars.v0, strict=True))  # v₀ still
    departure = scalars.departure()
    assert departure["norm_ratio"] > 0 and 0 < departure["median"] <= departure["p90"]
    with pytest.raises(ValueError, match="before the first update"):
        UpdateScalars(optimizer)


def test_update_scalars_do_not_depend_on_the_chunk() -> None:
    # Per-parameter reductions in pieces of the flattened parameter: the pieces change only the
    # float64 summation order, and the histogram's counts not at all.
    ps = params()
    optimizer = adam(ps, "prewarmed")
    pieces, whole = UpdateScalars(optimizer, chunk=4), UpdateScalars(optimizer)
    for t, step in enumerate(gradients(3), 1):
        run(optimizer, ps, [step])
        a, b = pieces(t), whole(t)
        assert a["update_norm"] == pytest.approx(b["update_norm"], rel=1e-12)
        assert a["cos_update_grad"] == pytest.approx(b["cos_update_grad"], rel=1e-12)
        assert "d_departure" not in a  # steps 5, 10 and 25 only
    a, b = pieces.departure(), whole.departure()
    assert a["norm_ratio"] == pytest.approx(b["norm_ratio"], rel=1e-12)
    assert (a["median"], a["p90"]) == (b["median"], b["p90"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_update_scalars_on_cuda_agree_with_the_cpu_under_deterministic_algorithms() -> None:
    # Every kernel of the hook has a deterministic CUDA implementation (no histc), and the copy
    # of v₀ is pinned host memory.
    rows = {}
    for device in ("cpu", "cuda"):
        ps = [torch.nn.Parameter(p.detach().to(device)) for p in params()]
        g = torch.Generator().manual_seed(2)
        v_hat = [(torch.rand(p.shape, generator=g) + 0.5).to(device) for p in ps]
        optimizer = Adam(ps, lr=LR, betas=(B1, B2), eps=EPS, second_moment="prewarmed",
                         initial_second_moment=v_hat)  # fmt: skip
        scalars = UpdateScalars(optimizer, departure_at=(2,), chunk=4)
        assert all(v.is_pinned() is (device == "cuda") for v in scalars.v0)
        torch.use_deterministic_algorithms(True)
        try:
            for t, step in enumerate(gradients(2), 1):
                run(optimizer, ps, [[grad.to(device) for grad in step]])
                rows[device] = scalars(t)
        finally:
            torch.use_deterministic_algorithms(False)
    cpu, cuda = rows["cpu"], rows["cuda"]
    for key in ("update_norm", "cos_update_grad"):
        assert cuda[key] == pytest.approx(cpu[key], rel=1e-6)
    departure, expected = cuda["d_departure"], cpu["d_departure"]
    assert departure["norm_ratio"] == pytest.approx(expected["norm_ratio"], rel=1e-5)
    for key in ("median", "p90"):  # the same bin, or a neighbour for a value on its edge
        assert expected[key] / HISTOGRAM_BIN <= departure[key] <= expected[key] * HISTOGRAM_BIN


def test_a_parameter_without_a_gradient_is_not_updated_and_adds_nothing() -> None:
    # Adam.step skips a parameter whose gradient is None: its u_t is 0 at that step.
    ps = params()
    optimizer = adam(ps, "adaptive")
    run(optimizer, ps, gradients(1))
    ps[1].grad = None
    optimizer.step()
    row = UpdateScalars(optimizer)(2)
    m, d = optimizer.state[ps[0]]["exp_avg"], optimizer.denominator(ps[0])
    u = optimizer.step_size(ps[0]) * m.double() / d.double()
    g = ps[0].grad.double()
    assert row["update_norm"] == pytest.approx(u.norm().item(), rel=1e-12)
    cos = (u * g).sum() / (u.norm() * g.norm())
    assert row["cos_update_grad"] == pytest.approx(cos.item(), rel=1e-12)
