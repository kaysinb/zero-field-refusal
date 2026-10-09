"""CUDA paths on a tiny model: sampling, training with pinned projection directions, controls,
peak-memory records and the stage-0 checks. Skipped without a GPU; no checkpoint is downloaded."""

import json
from pathlib import Path

import pytest
import torch
from conftest import END_IDS, PAD_ID, prompts, tiny_model, tiny_tokenizer
from test_train import config, pass_path_check, prepare, prompt_sets

from zero_field_refusal.batches import collate
from zero_field_refusal.losses import response_logprobs
from zero_field_refusal.models import install_neutral_generation_config, set_numerics
from zero_field_refusal.runtime import Environment
from zero_field_refusal.teacher import sample_responses
from zero_field_refusal.train import run_training

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
CUDA = ["execution.device=cuda"]


@pytest.fixture
def gpu_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Environment:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    set_numerics(config().model)  # TF32 off, as in every run
    tokenizer = tiny_tokenizer()
    end_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    sets = prompt_sets()

    def new_model():
        model = tiny_model(tied=False, head_std=0.02).to("cuda")
        install_neutral_generation_config(model, end_ids, tokenizer.pad_token_id)
        return model

    env = Environment(
        tokenizer=tokenizer, end_ids=end_ids, pad_id=tokenizer.pad_token_id,
        device=torch.device("cuda"), new_model=new_model, prompt_set=lambda name: sets[name],
        data_root=tmp_path / "data",
    )  # fmt: skip
    pass_path_check(env)
    return env


def test_cuda_sampling_log_probs_match_the_training_path() -> None:
    set_numerics(config().model)
    model = tiny_model(tied=False, head_std=0.02).to("cuda")
    install_neutral_generation_config(model, END_IDS, PAD_ID)
    sampling = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=40)
    samples, distributions = sample_responses(
        model, prompts(12), [f"p{i}" for i in range(12)], sampling=sampling, end_ids=END_IDS,
        pad_id=PAD_ID, seed=0, batch_size=5, record_distributions=True,
    )  # fmt: skip
    batch = collate(samples, pad_id=PAD_ID, max_response_tokens=40, device="cuda")
    with torch.no_grad():
        logp = response_logprobs(model, batch)
    rescored = logp.gather(1, batch.input_ids[:, 1:][batch.target_mask][:, None])[:, 0]
    stored = torch.tensor([lp for s in samples for lp in s.logprobs], device="cuda")
    torch.testing.assert_close(rescored, stored, rtol=0, atol=1e-4)
    assert all(d.device.type == "cpu" for d in distributions)


def test_cuda_training_run_with_controls_and_memory_records(gpu_env) -> None:
    cfg = config(*CUDA, "eval.controls.line_scan.enabled=true",
                 "eval.controls.line_scan.coefficients=[1.0]",
                 "eval.controls.random_direction.enabled=true")  # fmt: skip
    prepare(gpu_env, cfg, "training")
    run_dir = run_training(cfg, gpu_env, run_id="gpu")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["peak_memory_gb/train"] > 0 and summary["peak_memory_gb/all"] > 0
    assert summary["step0_teacher_gap"]["token_max_abs"] < 1e-4
    assert summary["controls"]["line_scan_+1"]["K_harmful"] == pytest.approx(
        summary["K_harmful"], rel=1e-3
    )


def test_projection_directions_are_the_same_on_cpu_and_cuda(gpu_env) -> None:
    from zero_field_refusal.evaluation import random_direction
    from zero_field_refusal.models import freeze
    from zero_field_refusal.train import projection_directions

    trainable = freeze(gpu_env.new_model(), ["input_embeddings", "output_head"])
    on_cuda = projection_directions(trainable, 2)
    assert all(r.device.type == "cpu" for rs in on_cuda.values() for r in rs)
    shapes = {name: tuple(p.shape) for name, p in trainable}
    norms = {name: 1.0 for name in shapes}
    gpu = random_direction(shapes, norms, kind="global", seed=3, device="cuda")
    cpu = random_direction(shapes, norms, kind="global", seed=3, device="cpu")
    for name in shapes:
        torch.testing.assert_close(gpu[name].cpu(), cpu[name])


def test_cuda_stage0_checks(gpu_env) -> None:
    from zero_field_refusal.checks import run_stage0_checks

    cfg = config(*CUDA, "run_type=checks", "optimizer=adam_frozen", "optimizer.lr=1e-3",
                 "optimizer.preconditioner.batches=8")  # fmt: skip
    run = run_stage0_checks(cfg, gpu_env, log=lambda _: None)
    summary = json.loads((run.dir / "summary.json").read_text())
    assert summary["run_profile"]["peak_memory_gb"] > 0
    assert summary["adam_vs_torch"]["relative_diff"] < 1e-5
    assert summary["path_field"]["passed"]
    assert summary["field"]["token_count"]["decision"] in (
        "at_noise_level", "field_detected", "undetermined"
    )  # fmt: skip


def wide_model() -> torch.nn.Module:
    """About 50M parameters over a tiny vocabulary: parameter-sized buffers dominate the memory,
    as in the real checkpoints, so their count can be read off the peak."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=64, hidden_size=1024, intermediate_size=2816,
                         num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=2,
                         max_position_embeddings=256, tie_word_embeddings=False)  # fmt: skip
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).to(torch.float32).eval().to("cuda")
    install_neutral_generation_config(model, END_IDS, PAD_ID)
    return model


def test_field_measurements_hold_only_their_documented_buffers() -> None:
    # 1.5B in fp32 on 48 GB: the field at θ₀ may hold the gradient and two buffers, the path field
    # in Adam's metric the gradient, a soft-target copy and two buffers, with v̂ on the CPU.
    from zero_field_refusal.batches import chunks
    from zero_field_refusal.diagnostics import field_measurement, path_field_measurement
    from zero_field_refusal.models import freeze

    set_numerics(config().model)
    model = wide_model()
    params = [p for _, p in freeze(model, ["input_embeddings", "output_head"])]
    size = sum(p.numel() for p in params) * 4
    sampling = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=24)
    samples, distributions = sample_responses(
        model, prompts(16), [f"p{i}" for i in range(16)], sampling=sampling, end_ids=END_IDS,
        pad_id=PAD_ID, seed=0, batch_size=8, record_distributions=True,
    )  # fmt: skip
    cfg = config(*CUDA, "train.batch_size=4", "train.max_response_tokens=24",
                 "teacher.sampling.max_new_tokens=24")  # fmt: skip
    device = torch.device("cuda")

    def extra_buffers(measure) -> float:
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        measure()
        return (torch.cuda.max_memory_allocated() - base) / size

    field = extra_buffers(
        lambda: field_measurement(
            model, params, list(chunks(samples, 4)), cfg, pad_id=PAD_ID, device=device, steps=25
        )
    )
    v_hat = [torch.ones_like(p, device="cpu") for p in params]
    path = extra_buffers(lambda: path_field_measurement(
        model, params, samples, distributions, cfg, pad_id=PAD_ID, device=device, steps=25,
        preconditioner=v_hat))  # fmt: skip
    # Measured ≈ 3.65 and ≈ 4.5 (with one layer's temporaries); one more buffer adds 1.
    assert field < 4.0 and path < 5.0, (field, path)
