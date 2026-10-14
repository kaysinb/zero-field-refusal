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
    result = lanczos(operator, start, min_steps=10, max_steps=60, tolerance=1e-6)
    assert start == []  # consumed, so the caller holds no fourth vector
    assert result["converged"] and result["top"] == pytest.approx(500.0, rel=1e-8)
    assert result["residual"] <= 1e-6 * 500
    assert result["top_trace"][-1] == result["top"]
    trace = result["top_trace"]
    assert all(a <= b + 1e-9 for a, b in zip(trace, trace[1:], strict=False))  # interlacing


def test_a_minimum_of_steps_keeps_a_small_overlap_from_stopping_early() -> None:
    # The reviewer's case: a random start barely overlaps the top eigenvector, and the Ritz value
    # rests near the second eigenvalue with a residual under 1 % before the top one emerges.
    operator, top, generator = known_operator(400, [500.0, 470.0, 460.0])
    early = lanczos(operator, start_with_overlap(top, 1e-6, generator), min_steps=1,
                    max_steps=60, tolerance=0.01)  # fmt: skip
    assert early["converged"] and early["top"] < 480  # converged, to the wrong eigenvalue
    late = lanczos(operator, start_with_overlap(top, 1e-6, generator), min_steps=40,
                   max_steps=60, tolerance=0.01)  # fmt: skip
    assert late["converged"] and late["top"] == pytest.approx(500.0, rel=1e-4)


def test_lanczos_reports_no_convergence_at_its_step_limit() -> None:
    operator, top, generator = known_operator(400, [500.0, 470.0, 460.0])
    result = lanczos(operator, start_with_overlap(top, 1e-6, generator), min_steps=1,
                     max_steps=5, tolerance=0.01)  # fmt: skip
    assert result["steps"] == 5 and not result["converged"]
    assert result["residual"] > 0.01 * result["top"]


def test_lanczos_stops_when_the_krylov_space_is_invariant() -> None:
    def operator(v, out):
        out[0] += 2.0 * v[0]
        return out

    start = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    result = lanczos(operator, start, min_steps=10, max_steps=20, tolerance=0.01)
    assert result["steps"] == 1 and result["converged"] and result["top"] == pytest.approx(2.0)


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
