"""Amendment 3: the frozen-Adam soft-target control (experiments/study2_short_fa_controls)."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from test_protocol import THRESHOLDS, experiment

from zero_field_refusal.analysis.stage_a import arm_of
from zero_field_refusal.protocol import ProtocolError, load_run_config

EXPERIMENT = "experiments/study2_short_fa_controls"
CAMPAIGN = [*THRESHOLDS, "optimizer.lr=1e-5", "schedule.instability.contamination_k=0.01",
            "train.batch_size=8"]  # fmt: skip


def test_the_control_composes_and_follows_the_protocol() -> None:
    cfg = load_run_config(EXPERIMENT, CAMPAIGN)
    assert cfg.experiment == "study2_short_fa_controls"
    assert cfg.control == "soft_targets" and cfg.train.loss.kind == "soft_kl"
    assert cfg.protocol_version == 3 and cfg.train.steps == 25
    assert list(cfg.schedule.evaluate_at) == [25] and cfg.schedule.policy == "fixed"
    assert cfg.schedule.instability.grad_rule == "log"
    opt = cfg.optimizer
    assert (opt.name, opt.beta1, opt.eps, opt.weight_decay) == ("adam_frozen", 0.9, 1e-8, 0.0)
    assert opt.preconditioner.batches == 50 and opt.preconditioner.batch_size == 8
    assert "momentum" not in opt
    assert arm_of(cfg) == "soft_control_frozen_adam"


@pytest.mark.parametrize("bound", [None, 0, -0.01])
def test_the_campaign_must_set_a_positive_contamination_bound(bound: float | None) -> None:
    overrides = [o for o in CAMPAIGN if "contamination_k" not in o]
    with pytest.raises(ProtocolError, match="contamination_k"):  # the placeholder stays missing
        load_run_config(EXPERIMENT, overrides)
    with pytest.raises(ProtocolError, match="contamination_k"):
        load_run_config(EXPERIMENT, [*overrides, f"schedule.instability.contamination_k={bound}"])


@pytest.mark.parametrize("name", ["adamw", "adamw_prewarmed"])
@pytest.mark.parametrize("version", [1, 2, 3])
def test_other_adam_variants_stay_rejected_under_soft_targets(tmp_path: Path, name: str, version: int
) -> None:
    path = experiment(tmp_path, "control/soft_targets")
    overrides = [*THRESHOLDS, f"optimizer={name}", "optimizer.lr=1e-5",
                 f"+protocol_version={version}"]  # fmt: skip
    with pytest.raises(ProtocolError, match=r"optimizer\.name"):
        load_run_config(path, overrides)


@pytest.mark.parametrize("version", [1, 2])
def test_frozen_adam_soft_control_needs_protocol_version_3(version: int) -> None:
    with pytest.raises(ProtocolError, match=r"optimizer\.name"):
        load_run_config(EXPERIMENT, [*CAMPAIGN, f"protocol_version={version}"])
    load_run_config(EXPERIMENT, CAMPAIGN)  # version 3


def test_plain_adam_is_rejected_at_version_3() -> None:
    with pytest.raises(ProtocolError, match=r"optimizer\.name"):
        load_run_config(EXPERIMENT, [*CAMPAIGN, "optimizer.name=adam"])


def test_arm_names() -> None:
    cfg = OmegaConf.create({"control": "soft_targets", "optimizer": {"name": "adam_frozen"},
                            "schedule": {"policy": "fixed"}})  # fmt: skip
    assert arm_of(cfg) == "soft_control_frozen_adam"
    cfg.optimizer.name = "sgd"
    assert arm_of(cfg) == "soft_control"
    cfg.control = None
    cfg.optimizer.name = "adam_frozen"
    assert arm_of(cfg) == "frozen_adam_25"


def test_a_frozen_adam_soft_control_trajectory_loads_as_its_own_arm(tmp_path: Path) -> None:
    from test_pilot_momentum import with_momentum
    from test_stage_a import write_run

    from zero_field_refusal.analysis.stage_a import load_trajectory

    run = with_momentum(write_run(tmp_path, "fa", arm="soft_control", batch=5, lr=1e-5,
                                  seed=2000, final_k=0.001, evaluate_at=(25,)))  # fmt: skip
    cfg = OmegaConf.load(run / "config.yaml")
    del cfg.optimizer["momentum"]
    cfg.optimizer.name, cfg.optimizer.beta1 = "adam_frozen", 0.9
    cfg.schedule.policy, cfg.protocol_version = "fixed", 3
    OmegaConf.save(cfg, run / "config.yaml")
    trajectory = load_trajectory(run)
    assert trajectory.arm == "soft_control_frozen_adam" and trajectory.policy == "fixed"
    assert [e["step"] for e in trajectory.evaluations] == [25]


def test_the_schedule_policy_and_bound_are_required_unconditionally(tmp_path: Path) -> None:
    from zero_field_refusal.protocol import protocol_problems

    path = experiment(tmp_path, "optimizer/adam_frozen", "control/soft_targets")
    base = [*THRESHOLDS, "optimizer.lr=1e-5", "+protocol_version=3"]
    with pytest.raises(ProtocolError, match="scheduled run"):  # no schedule node
        load_run_config(path, base)
    with pytest.raises(ProtocolError, match="policy"):
        load_run_config(EXPERIMENT, [*CAMPAIGN, "schedule.policy=k_stopped"])
    cfg = load_run_config(EXPERIMENT, CAMPAIGN)
    del cfg.schedule["instability"]  # a schedule without an instability block
    assert any("contamination_k" in p for p in protocol_problems(cfg))
