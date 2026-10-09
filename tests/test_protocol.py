from pathlib import Path

import pytest

from zero_field_refusal.config import config_hash
from zero_field_refusal.paths import CONFIG_ROOT
from zero_field_refusal.protocol import ProtocolError, load_run_config

OPTIMIZERS = ["sgd", "sgd_momentum", "adamw", "adam_frozen", "adamw_prewarmed"]


def experiment(tmp_path: Path, *defaults: str) -> Path:
    """An experiment file outside the repository composing repository configs."""
    path = tmp_path / "exp.yaml"
    names = ["experiments/smoke", *defaults]
    path.write_text("defaults: [" + ", ".join(names) + "]\n")
    return path


THRESHOLDS = [
    "checks.teacher_gap.token_max_abs=1e-4",
    "checks.teacher_gap.response_mean_signed_abs=1e-3",
]


def test_smoke_experiment_follows_the_protocol() -> None:
    cfg = load_run_config("experiments/smoke")
    assert cfg.control is None
    assert cfg.train.loss.denominator == cfg.train.batch_size * 96
    with pytest.raises(ValueError, match="checks.teacher_gap"):  # set from stage-0 measurements
        config_hash(cfg)
    config_hash(load_run_config("experiments/smoke", THRESHOLDS))  # every other value is set


def test_training_needs_numeric_thresholds_but_checks_and_theta0_do_not() -> None:
    null = [
        "checks.teacher_gap.token_max_abs=null",
        "checks.teacher_gap.response_mean_signed_abs=null",
    ]
    with pytest.raises(ProtocolError, match="needs the threshold set"):
        load_run_config("experiments/smoke", null)
    for run_type in ("checks", "theta0"):
        load_run_config("experiments/smoke", [*null, f"run_type={run_type}"])
    with pytest.raises(ProtocolError, match="positive threshold"):
        load_run_config(
            "experiments/smoke", [*THRESHOLDS[1:], "checks.teacher_gap.token_max_abs=0"]
        )
    with pytest.raises(ProtocolError, match="checks.path_field.batches"):
        load_run_config("experiments/smoke", ["checks.path_field.batches=3"])


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_every_optimizer_follows_the_protocol(name: str) -> None:
    cfg = load_run_config("experiments/smoke", [f"optimizer={name}", "optimizer.lr=1e-5"])
    assert cfg.optimizer.weight_decay == 0


def test_bridge_cell_truncates_the_stream_and_rescales_the_denominator() -> None:
    cfg = load_run_config("experiments/smoke", ["train.max_response_tokens=48"])
    assert cfg.teacher.sampling.max_new_tokens == 96  # same teacher stream
    assert cfg.train.loss.denominator == cfg.train.batch_size * 48


@pytest.mark.parametrize(
    ("control", "extra"),
    [
        ("token_count", []),
        ("sharpened_sampling", []),
        ("alpaca_reference", []),
        ("soft_targets", ["optimizer/sgd"]),
    ],
)
def test_each_control_permits_its_own_deviation(
    tmp_path: Path, control: str, extra: list[str]
) -> None:
    cfg = load_run_config(experiment(tmp_path, *extra, f"control/{control}"))
    assert cfg.control


def test_control_files_exist_for_every_named_control() -> None:
    assert {p.stem for p in (CONFIG_ROOT / "control").glob("*.yaml")} == {
        "token_count",
        "sharpened_sampling",
        "alpaca_reference",
        "soft_targets",
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("teacher.sampling.temperature=0.7", "teacher.sampling.temperature"),
        ("teacher.sampling.top_k=20", "teacher.sampling.top_k"),
        ("teacher.sampling.repetition_penalty=1.1", "teacher.sampling.repetition_penalty"),
        ("teacher.prompt_assignment=balanced", "teacher.prompt_assignment"),
        ("teacher.source=alpaca_reference", "teacher.source"),
        ("teacher.responses=600", "teacher.responses"),
        ("train.loss.normalization=token_count", "train.loss.normalization"),
        ("train.loss.denominator=1000", "train.loss.denominator"),
        ("train.loss.include_end_token=false", "train.loss.include_end_token"),
        ("train.loss.kind=soft_kl", "train.loss.kind"),
        ("train.max_response_tokens=128", "train.max_response_tokens"),
        ("train.freeze=[input_embeddings]", "train.freeze"),
        ("optimizer.weight_decay=0.01", "optimizer.weight_decay"),
        ("optimizer.first_moment_dtype=bfloat16", "optimizer.first_moment_dtype"),
        ("model.dtype=bfloat16", "model.dtype"),
        ("model.tf32=true", "model.tf32"),
        ("model.revision=main", "model.revision"),
        ("model.attn_implementation=flash_attention_2", "model.attn_implementation"),
        ("model.end_tokens=[]", "model.end_tokens"),
        ("control=token_count_normalization", "train.loss.normalization"),
        ("control=unknown", "control"),
    ],
)
def test_departures_from_the_protocol_fail_at_load(override: str, message: str) -> None:
    with pytest.raises(ProtocolError, match=message.replace(".", r"\.")):
        load_run_config("experiments/smoke", [override])


def test_soft_targets_require_plain_sgd(tmp_path: Path) -> None:
    with pytest.raises(ProtocolError, match="optimizer.name"):
        load_run_config(experiment(tmp_path, "control/soft_targets"))
    with pytest.raises(ProtocolError, match="optimizer.momentum"):
        load_run_config(experiment(tmp_path, "optimizer/sgd_momentum", "control/soft_targets"))


def test_all_problems_are_reported_together() -> None:
    with pytest.raises(ProtocolError) as info:
        load_run_config("experiments/smoke", ["model.tf32=true", "optimizer.weight_decay=0.1"])
    assert "model.tf32" in str(info.value) and "optimizer.weight_decay" in str(info.value)


def test_missing_protocol_keys_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "bare.yaml"
    path.write_text("defaults: [base, model/qwen2.5-0.5b-instruct]\n")
    with pytest.raises(ProtocolError, match=r"teacher\.sampling\.temperature: missing"):
        load_run_config(path)
