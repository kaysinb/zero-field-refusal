"""Llama's configs (docs/study5-llama-registration.md): the runs' config is the AdamW order
repeat's but for the model and the experiment's name; the K-only pilot and θ₀'s evaluation are the
runs' config; the stage-0 checks write the path-field record the runs require."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.prechecks import environment_fingerprint
from zero_field_refusal.protocol import load_run_config

RUN = ["optimizer.lr=2.0e-05", "seed=60000", "train.order_seed=61000",
       "checks.teacher_gap.token_max_abs=1.0", "checks.teacher_gap.response_mean_signed_abs=1.0"]  # fmt: skip


def as_dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


def test_the_llama_runs_are_the_adamw_repeats_but_for_the_model():
    llama = load_run_config("experiments/study5_llama", RUN)
    qwen = load_run_config("experiments/study5_order_a", RUN)
    assert config_hash(llama) and llama.model.name == "meta-llama/Llama-3.2-1B-Instruct"
    a, b = as_dict(llama), as_dict(qwen)
    for d in (a, b):
        d.pop("model"), d.pop("experiment"), d.pop("checks")
    assert a == b


def test_the_pilot_and_theta0_are_the_runs_config():
    run = as_dict(load_run_config("experiments/study5_llama", RUN))
    pilot = as_dict(load_run_config("experiments/study5_llama_k_only", RUN))
    assert pilot.pop("eval") == {**run.pop("eval"), "k_only": True}
    assert pilot == run
    theta0 = as_dict(load_config("experiments/study5_llama_theta0", ["seed=60000"]))
    assert theta0["model"]["name"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert theta0["train"]["order_seed"] == 60000


def test_the_stage0_record_is_the_one_the_runs_require():
    stage0 = load_config("experiments/study5_llama_stage0")
    run = load_run_config("experiments/study5_llama", RUN)
    assert stage0.model.name == run.model.name and stage0.run_type == "checks"
    assert stage0.model.chat_date == "26 Jul 2024"
    cpu = torch.device("cpu")
    assert environment_fingerprint(stage0, cpu) == environment_fingerprint(run, cpu)
    assert stage0.checks.teacher_gap.token_max_abs is None  # measured there


def test_the_stage0_cap_covers_the_runs_responses():
    stage0 = load_config("experiments/study5_llama_stage0")
    run = load_run_config("experiments/study5_llama", RUN)
    assert stage0.train.max_response_tokens >= run.train.max_response_tokens
    assert stage0.teacher.sampling == run.teacher.sampling  # the same θ₀ sampling
