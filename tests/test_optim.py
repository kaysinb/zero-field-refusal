import pytest
import torch
from omegaconf import OmegaConf

from zero_field_refusal.config import load_config
from zero_field_refusal.optim import Adam, build_optimizer, estimate_second_moment

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
