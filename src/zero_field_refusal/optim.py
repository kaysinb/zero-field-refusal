"""Optimizers of the five arms.

The optimizer is the object of study, so it stays a plain per-parameter implementation with fp32
state; no fused, foreach or 8-bit variant replaces it. ``Adam`` has three second-moment modes
(docs/preregistration.md, H2a):

- ``adaptive``: standard Adam without weight decay. Its operations follow PyTorch's
  single-tensor AdamW, and a test checks that both agree;
- ``frozen``: the second moment is the θ₀ estimate, never updated and not bias-corrected;
- ``prewarmed``: the second moment starts from the θ₀ estimate, is updated as usual and is not
  bias-corrected, since it is a full estimate from the start.

In every mode the first moment starts at zero with the usual bias correction and the step counter
starts at 0. SGD with and without momentum is ``torch.optim.SGD``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch
from omegaconf import DictConfig

SECOND_MOMENT_MODES = ("adaptive", "frozen", "prewarmed")
ADAM_MODES = {"adamw": "adaptive", "adam_frozen": "frozen", "adamw_prewarmed": "prewarmed"}


class Adam(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float,
        betas: tuple[float, float | None] = (0.9, 0.999),
        eps: float = 1e-8,
        *,
        second_moment: str = "adaptive",
        initial_second_moment: Sequence[torch.Tensor] | None = None,
    ) -> None:
        if second_moment not in SECOND_MOMENT_MODES:
            raise ValueError(f"second_moment must be one of {SECOND_MOMENT_MODES}")
        if second_moment != "frozen" and betas[1] is None:
            raise ValueError(f"{second_moment} Adam needs beta2")
        super().__init__(list(params), {"lr": lr, "betas": betas, "eps": eps})
        self.second_moment = second_moment
        if second_moment == "adaptive":
            return
        ordered = [p for group in self.param_groups for p in group["params"]]
        if initial_second_moment is None or len(initial_second_moment) != len(ordered):
            raise ValueError(
                f"{second_moment} Adam needs initial_second_moment for every parameter"
            )
        for p, v in zip(ordered, initial_second_moment, strict=True):
            if v.shape != p.shape:
                raise ValueError(f"Second-moment estimate of shape {v.shape} for {p.shape}")
            self.state[p]["exp_avg_sq"] = (
                v.detach().to(device=p.device, dtype=torch.float32).clone()
            )

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> None:
        if closure is not None:
            raise ValueError("closures are not supported")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    if self.second_moment == "adaptive":
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1
                step = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                exp_avg.lerp_(grad, 1 - beta1)
                if self.second_moment != "frozen":
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if self.second_moment == "adaptive":
                    bias_correction2_sqrt = (1 - beta2**step) ** 0.5
                    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
                else:
                    denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-lr / (1 - beta1**step))

    # Read accessors (the update scalars, schedule.UpdateScalars). They read the state and
    # allocate only their result; ``step`` does not use them, and a test checks that they give
    # back its update bit for bit.

    def _group(self, p: torch.Tensor) -> dict:
        for group in self.param_groups:
            if any(q is p for q in group["params"]):
                return group
        raise ValueError("Not a parameter of this optimizer")

    def step_size(self, p: torch.Tensor) -> float:
        """lr / (1 − β₁ᵗ) of ``p``'s last update: the factor ``step`` applied to m_t / d_t."""
        group = self._group(p)
        return group["lr"] / (1 - group["betas"][0] ** self.state[p]["step"])

    @torch.no_grad()
    def denominator(
        self,
        p: torch.Tensor,
        coordinates: slice | None = None,
        *,
        second_moment: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The denominator d_t that ``step`` divided by at ``p``'s last update, computed with
        its operations into a new fp32 tensor: √v_t / √(1 − β₂ᵗ) + ε (adaptive), √v + ε (frozen:
        v₀; pre-warmed: v_t, before the first update v₀). With ``coordinates``, only those of the
        flattened parameter; with ``second_moment`` (on those coordinates), from that v instead
        of the state's."""
        group, state = self._group(p), self.state[p]
        v = second_moment
        if v is None:
            if "exp_avg_sq" not in state:
                raise ValueError("Adaptive Adam has no denominator before its first update")
            v = state["exp_avg_sq"]
            if coordinates is not None:
                v = v.reshape(-1)[coordinates]  # a view: UpdateScalars checks contiguity
        if self.second_moment == "adaptive":
            bias_correction2_sqrt = (1 - group["betas"][1] ** state["step"]) ** 0.5
            return (v.sqrt() / bias_correction2_sqrt).add_(group["eps"])
        return v.sqrt().add_(group["eps"])


def estimate_second_moment(
    params: Sequence[torch.nn.Parameter],
    batch_gradient: Callable[[int], None],
    num_batches: int,
) -> list[torch.Tensor]:
    """Mean over ``num_batches`` of the squared batch gradient at the current parameters.

    ``batch_gradient(i)`` must leave the gradient of batch ``i`` of the preconditioner stream in
    ``p.grad``, with the same loss and denominator as training, so the estimate has the units of
    Adam's second moment at the training batch size.
    """
    sums = [torch.zeros_like(p, dtype=torch.float32) for p in params]
    for i in range(num_batches):
        for p in params:
            p.grad = None
        batch_gradient(i)
        for total, p in zip(sums, params, strict=True):
            total.addcmul_(p.grad, p.grad)
    for p in params:
        p.grad = None
    return [total.div_(num_batches) for total in sums]


def build_optimizer(
    opt_cfg: DictConfig,
    params: Sequence[torch.nn.Parameter],
    *,
    second_moment: Sequence[torch.Tensor] | None = None,
) -> torch.optim.Optimizer:
    """The optimizer of an ``optimizer`` config section; weight decay must be 0."""
    if opt_cfg.weight_decay != 0:
        raise ValueError(f"weight_decay must be 0 at zero field, got {opt_cfg.weight_decay}")
    for key in ("first_moment_dtype", "second_moment_dtype"):
        if opt_cfg.get(key, "float32") != "float32":
            raise ValueError(f"{key} must be float32: Adam state is kept in fp32")
    name = opt_cfg.name
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=opt_cfg.lr, momentum=opt_cfg.momentum, weight_decay=0.0, foreach=False
        )
    if name not in ADAM_MODES:
        raise ValueError(f"Unknown optimizer {name!r}")
    mode = ADAM_MODES[name]
    betas = (opt_cfg.beta1, None) if mode == "frozen" else tuple(opt_cfg.betas)
    return Adam(
        params,
        lr=opt_cfg.lr,
        betas=betas,
        eps=opt_cfg.eps,
        second_moment=mode,
        initial_second_moment=None if mode == "adaptive" else second_moment,
    )
