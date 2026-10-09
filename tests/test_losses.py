import pytest
import torch
from conftest import END_IDS, PAD_ID, prompts, tiny_model
from omegaconf import OmegaConf

from zero_field_refusal.batches import collate
from zero_field_refusal.losses import loss_denominator, response_logprobs, soft_kl, summed_nll
from zero_field_refusal.models import freeze, install_neutral_generation_config
from zero_field_refusal.teacher import Sample, sample_responses

SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=40)


@pytest.fixture(scope="module")
def teacher_and_samples():
    model = tiny_model(tied=False, head_std=0.02)
    install_neutral_generation_config(model, END_IDS, PAD_ID)
    samples, distributions = sample_responses(
        model, prompts(8), [f"p{i}" for i in range(8)], sampling=SAMPLING, end_ids=END_IDS,
        pad_id=PAD_ID, seed=0, batch_size=4, record_distributions=True,
    )  # fmt: skip
    return model, samples, distributions


def nll(model, samples, cap=40):
    with torch.no_grad():
        return summed_nll(model, collate(samples, pad_id=PAD_ID, max_response_tokens=cap)).item()


def test_a_response_contributes_the_same_whatever_else_is_in_the_batch(teacher_and_samples):
    # Invariant 5: no term of the loss depends on the lengths of the other responses.
    model, samples, _ = teacher_and_samples
    lengths = [len(s.response_ids) for s in samples]
    assert len(set(lengths)) > 1
    together = nll(model, samples)
    alone = sum(nll(model, [s]) for s in samples)
    assert together == pytest.approx(alone, rel=1e-6)


def test_nll_sums_the_stored_log_probs_including_the_end_token(teacher_and_samples):
    model, samples, _ = teacher_and_samples
    ended = [s for s in samples if s.ended]
    assert ended
    expected = -sum(sum(s.logprobs) for s in ended)
    assert nll(model, ended) == pytest.approx(expected, abs=1e-4)


def test_truncation_drops_tokens_after_the_cap(teacher_and_samples):
    model, samples, _ = teacher_and_samples
    s = max(samples, key=lambda s: len(s.response_ids))
    expected = -sum(s.logprobs[:3])
    assert nll(model, [s], cap=3) == pytest.approx(expected, abs=1e-4)


def test_denominators() -> None:
    samples = [Sample("p", (1,), tuple(range(n)), (0.0,) * n, False) for n in (2, 5, 9)]
    constant = OmegaConf.create({"normalization": "constant", "denominator": 1920})
    assert loss_denominator(constant, samples, 96) == 1920.0
    token_count = OmegaConf.create({"normalization": "token_count", "denominator": 1920})
    assert loss_denominator(token_count, samples, 96) == 16.0
    assert loss_denominator(token_count, samples, 4) == 2 + 4 + 4  # after truncation


def gradient_norm(model, loss) -> float:
    model.zero_grad(set_to_none=True)
    loss.backward()
    return torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None))


def test_soft_targets_from_the_training_path_give_a_vanishing_gradient(teacher_and_samples):
    model, samples, _ = teacher_and_samples
    freeze(model, ["input_embeddings", "output_head"])
    batch = collate(samples, pad_id=PAD_ID, max_response_tokens=40)
    with torch.no_grad():
        same_function = response_logprobs(model, batch)
    kl_grad = gradient_norm(model, soft_kl(model, batch, same_function))
    nll_grad = gradient_norm(model, summed_nll(model, batch))
    assert kl_grad < 1e-5 * nll_grad


def test_soft_targets_from_the_generation_path_measure_the_path_discrepancy(teacher_and_samples):
    model, samples, distributions = teacher_and_samples
    freeze(model, ["input_embeddings", "output_head"])
    batch = collate(samples, pad_id=PAD_ID, max_response_tokens=40)
    recorded = torch.cat(distributions)
    with torch.no_grad():
        assert soft_kl(model, batch, recorded).item() >= -1e-6
    kl_grad = gradient_norm(model, soft_kl(model, batch, recorded))
    nll_grad = gradient_norm(model, summed_nll(model, batch))
    assert kl_grad < 1e-3 * nll_grad  # the proposed criterion of invariant 8, on CPU fp32


def test_micro_batch_gradients_add_up_to_the_batch_gradient(teacher_and_samples):
    model, samples, _ = teacher_and_samples
    params = [p for _, p in freeze(model, ["input_embeddings", "output_head"])]

    def gradient(micro: int) -> list[torch.Tensor]:
        model.zero_grad(set_to_none=True)
        for start in range(0, len(samples), micro):
            batch = collate(samples[start : start + micro], pad_id=PAD_ID, max_response_tokens=40)
            (summed_nll(model, batch) / 1920).backward()
        return [p.grad.clone() for p in params]

    for whole, parts in zip(gradient(len(samples)), gradient(3), strict=True):
        torch.testing.assert_close(parts, whole, rtol=1e-5, atol=1e-8)
