import math

import pytest
import torch
from conftest import END_IDS, PAD_ID, prompts, tiny_model

from zero_field_refusal.batches import collate
from zero_field_refusal.curvature import hessian_vector_product, lanczos, start_vector
from zero_field_refusal.losses import summed_nll
from zero_field_refusal.models import freeze, install_neutral_generation_config
from zero_field_refusal.teacher import sample_responses

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


def test_lanczos_finds_the_extreme_eigenvalues_of_a_known_matrix() -> None:
    generator = torch.Generator().manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(60, 60, dtype=torch.float64, generator=generator))
    eigenvalues = torch.cat([torch.tensor([500.0, 350.0, -40.0], dtype=torch.float64),
                             torch.linspace(-5, 20, 57, dtype=torch.float64)])  # fmt: skip
    matrix = q @ torch.diag(eigenvalues) @ q.T

    # Two "parameters" of 20 and 40 entries, as lists of tensors.
    def operator(v):
        flat = matrix @ torch.cat([t.flatten() for t in v])
        return [flat[:20].clone(), flat[20:].clone()]

    start = torch.randn(60, dtype=torch.float64, generator=generator)
    start /= start.norm()
    result = lanczos(operator, [start[:20], start[20:]], steps=30)
    assert result["top"] == pytest.approx(500.0, rel=1e-8)
    assert result["bottom"] == pytest.approx(-40.0, rel=1e-6)
    assert result["top_trace"][-1] == result["top"]
    trace = result["top_trace"]
    assert all(a <= b + 1e-9 for a, b in zip(trace, trace[1:], strict=False))  # interlacing


def test_lanczos_stops_when_the_krylov_space_is_invariant() -> None:
    def operator(v):
        return [2.0 * v[0]]

    result = lanczos(operator, [torch.tensor([1.0, 0.0], dtype=torch.float64)], steps=10)
    assert result["steps"] == 1 and result["top"] == pytest.approx(2.0)
