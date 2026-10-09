import math

import pytest
import torch
from conftest import END_IDS, PAD_ID, prompts, tiny_model
from omegaconf import OmegaConf

from zero_field_refusal.batches import chunks, collate
from zero_field_refusal.diagnostics import (
    field_decision,
    field_measurement,
    path_discrepancy,
    soft_target_gradient_ratio,
)
from zero_field_refusal.losses import summed_nll
from zero_field_refusal.models import freeze, install_neutral_generation_config
from zero_field_refusal.teacher import Sample, sample_responses

BATCH, HALF, CAP = 4, 16, 8
CFG = OmegaConf.create(
    {
        "train": {"batch_size": BATCH, "max_response_tokens": CAP},
        "execution": {"micro_batch_size": 4},
    }
)


def stream(model, temperature: float, n: int, *, record=False):
    sampling = dict(temperature=temperature, top_p=1.0, top_k=0, repetition_penalty=1.0,
                    max_new_tokens=CAP)  # fmt: skip
    return sample_responses(
        model, prompts(n, seed=5), [f"p{i}" for i in range(n)], sampling=sampling, end_ids=END_IDS,
        pad_id=PAD_ID, seed=11, batch_size=16, zero_field=temperature == 1.0,
        record_distributions=record,
    )  # fmt: skip


@pytest.fixture(scope="module")
def teacher():
    model = tiny_model(tied=False, head_std=0.3)
    install_neutral_generation_config(model, END_IDS, PAD_ID)
    params = [p for _, p in freeze(model, ["input_embeddings", "output_head"])]
    return model, params


def measure(teacher, temperature):
    model, params = teacher
    samples, _ = stream(model, temperature, 2 * HALF * BATCH)
    batches = chunks(samples, BATCH)
    return field_measurement(model, params, batches, CFG, pad_id=PAD_ID,
                             device=torch.device("cpu"), steps=25)  # fmt: skip


def test_zero_field_mean_gradient_shrinks_like_one_over_sqrt_n(teacher):
    # Invariant 9: no field, so the mean of n batch gradients is pure noise of size ~1/sqrt(n).
    out = measure(teacher, 1.0)
    norms = out["mean_gradient_norm"]
    assert math.sqrt(HALF) / 2 < norms[1] / norms[HALF] < 2 * math.sqrt(HALF)
    constant = out["constant"]
    assert abs(constant["field_norm_sq"]) < 3 * constant["field_norm_sq_standard_error"]
    assert constant["decision"] != "field_detected"


def test_a_sharpened_teacher_creates_a_detectable_field(teacher):
    out = measure(teacher, 0.3)
    constant = out["constant"]
    assert constant["field_norm_sq"] > 3 * constant["field_norm_sq_standard_error"]
    assert constant["decision"] == "field_detected"
    norms = out["mean_gradient_norm"]
    assert norms[1] / norms[HALF] < math.sqrt(HALF) / 2  # levels off at the field


def test_equal_token_counts_leave_no_token_count_field(teacher):
    # All responses truncated at the cap: c_b = 1 for every batch, so the control variate is zero.
    model, params = teacher
    g = torch.Generator().manual_seed(0)
    samples = [
        Sample(f"p{i}", tuple(prompt), tuple(torch.randint(7, 64, (CAP,), generator=g).tolist()),
               (0.0,) * CAP, False)
        for i, prompt in enumerate(prompts(2 * HALF * BATCH, seed=6))
    ]  # fmt: skip
    out = field_measurement(model, params, chunks(samples, BATCH), CFG, pad_id=PAD_ID,
                            device=torch.device("cpu"), steps=25)  # fmt: skip
    assert out["token_count"] == {
        "field_norm_sq": 0.0, "field_norm_sq_standard_error": 0.0, "count_variance_bound": 0.0,
        "decision": "at_noise_level", "decision_basis": "equal_token_counts",
    }  # fmt: skip


def test_field_estimates_equal_the_split_half_inner_products(teacher):
    # Brute force: per-batch gradients as vectors, c_b from token counts, centered at their mean.
    model, params = teacher
    samples, _ = stream(model, 1.0, 2 * 4 * BATCH)
    batches = chunks(samples, BATCH)
    out = field_measurement(model, params, batches, CFG, pad_id=PAD_ID,
                            device=torch.device("cpu"), steps=25)  # fmt: skip
    vectors, factors = [], []
    for batch in batches:
        model.zero_grad(set_to_none=True)
        (summed_nll(model, collate(batch, pad_id=PAD_ID, max_response_tokens=CAP)) / (BATCH * CAP)
         ).backward()  # fmt: skip
        vectors.append(torch.cat([p.grad.flatten().double() for p in params]))
        factors.append(BATCH * CAP / sum(min(len(s.response_ids), CAP) for s in batch))
    g = torch.stack(vectors)
    c = torch.tensor(factors, dtype=torch.float64)
    assert c.std() > 0  # responses of different lengths
    d = (c - c.mean())[:, None] * g
    for name, x in (("constant", g), ("token_count", d)):
        expected = (x[:4].mean(0) @ x[4:].mean(0)).item()
        assert out[name]["field_norm_sq"] == pytest.approx(expected, rel=1e-4, abs=1e-12)
    assert out["token_count"]["count_variance_bound"] == pytest.approx(25 * c.var().item())


@pytest.mark.parametrize(
    ("field", "se", "expected"),
    [
        (0.001, 0.0005, "at_noise_level"),
        (0.1, 0.01, "field_detected"),
        (0.04, 0.02, "undetermined"),
    ],
)
def test_field_decision_accounts_for_uncertainty(field, se, expected):
    # N = 25 and sigma^2 = 1: the field is at the noise level when 25·|f|^2 < 1.
    assert field_decision(field, se, 1.0, 25) == expected


def test_generation_and_training_paths_agree_on_cpu(teacher):
    model, params = teacher
    samples, recorded = stream(model, 1.0, BATCH, record=True)
    out = path_discrepancy(model, samples, recorded, pad_id=PAD_ID, micro_batch_size=2,
                           device=torch.device("cpu"))  # fmt: skip
    assert out["tokens"] == sum(len(s.response_ids) for s in samples)
    assert 0 <= out["kl_max"] < 1e-6 and out["logprob_diff_max_abs"] < 1e-5
    ratio = soft_target_gradient_ratio(model, params, samples, recorded, CFG, pad_id=PAD_ID,
                                       device=torch.device("cpu"))  # fmt: skip
    assert ratio["ratio"] < 1e-3 and ratio["nll_grad_norm"] > 0
