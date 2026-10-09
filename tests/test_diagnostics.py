import math

import pytest
import torch
from conftest import END_IDS, PAD_ID, prompts, tiny_model
from omegaconf import OmegaConf

from zero_field_refusal.batches import chunks
from zero_field_refusal.diagnostics import (
    field_measurement,
    path_discrepancy,
    soft_target_gradient_ratio,
)
from zero_field_refusal.models import freeze, install_neutral_generation_config
from zero_field_refusal.teacher import sample_responses

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
    norms = out["mean_gradient_norm"]["constant"]
    ratio = norms[1] / norms[HALF]
    assert math.sqrt(HALF) / 2 < ratio < 2 * math.sqrt(HALF)
    constant = out["constant"]
    assert abs(constant["drift_per_step"]) < 4 * constant["drift_standard_error"]


def test_a_sharpened_teacher_creates_a_detectable_field(teacher):
    out = measure(teacher, 0.3)
    constant = out["constant"]
    assert constant["drift_per_step"] > 4 * constant["drift_standard_error"]
    assert constant["field_norm_sq_unbiased"] > 0
    assert not constant["at_noise_level"]
    norms = out["mean_gradient_norm"]["constant"]
    assert norms[1] / norms[HALF] < math.sqrt(HALF) / 2  # levels off at the field


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
