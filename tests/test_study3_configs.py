"""Study 3's configs: the regular and K-only runs, the qualification scope and the bridge's
config agreement (docs/study3-preregistration.md, "Numerical qualification")."""

import pytest
from omegaconf import OmegaConf

from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.protocol import check_run_config

ARMS = ("adamw", "adam_frozen", "adamw_prewarmed")
ROLES = ("engineering", "search", "numerical", "validation", "confirmation")
# config_hash of experiments/study2_short (role confirmation, seed 3000, lr 1e-5) before the
# execution and tracking keys were added to base.yaml: those keys must not move any hash.
STUDY2_HASHES = {
    "adamw": "a8051b290199",
    "adam_frozen": "ed2c869f647a",
    "adamw_prewarmed": "7332304b8449",
}


def study2(arm):
    return load_config(
        "experiments/study2_short",
        ["study2_short.role=confirmation", f"optimizer={arm}", "optimizer.lr=1e-5", "seed=3000"],
    )


def study3(name, role, arm, *extra):
    return load_config(
        f"experiments/{name}",
        [f"study3.role={role}", f"optimizer={arm}", "optimizer.lr=1e-5", "seed=3000", *extra],
    )


def plain(cfg, *drop):
    data = OmegaConf.to_container(cfg, resolve=True)
    for key in drop:
        node = data
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.get(part, {})
        node.pop(leaf, None)
    return data


@pytest.mark.parametrize("arm", ARMS)
def test_study2_short_hash_is_unchanged(arm):
    assert config_hash(study2(arm)) == STUDY2_HASHES[arm]


@pytest.mark.parametrize("arm", ARMS)
def test_k_only_agrees_with_regular_run(arm):
    k_only = study3("study3_k_only", "numerical", arm, "tracking.endpoint_checkpoint=/x")
    regular = study3("study3", "confirmation", arm)
    assert k_only.eval.k_only is True
    for key in ("train", "optimizer", "execution", "data"):
        assert OmegaConf.to_container(k_only[key]) == OmegaConf.to_container(regular[key]), key
    assert k_only.seed == regular.seed


@pytest.mark.parametrize("arm", ARMS)
def test_study3_equals_study2_short_but_for_its_identity(arm):
    drop = (
        "experiment",
        "study2_short",
        "study3",
        "execution.memory_limit_gb",
        "tracking.provenance_hashes",
    )
    assert plain(study3("study3", "confirmation", arm), *drop) == plain(study2(arm), *drop)


def test_study3_keys():
    cfg = study3("study3", "search", "adamw")
    assert cfg.execution.memory_limit_gb == 46
    assert cfg.tracking.provenance_hashes is True
    assert cfg.protocol_version == 2
    assert cfg.experiment == "study3"
    base = load_config("base")
    assert base.execution.deterministic is False
    assert base.execution.memory_limit_gb is None
    assert base.tracking.provenance_hashes is False
    assert base.tracking.endpoint_checkpoint is None


def test_role_is_required_and_no_study2_section_remains():
    cfg = load_config("experiments/study3", ["optimizer.lr=1e-5"])
    with pytest.raises(ValueError, match="study3.role"):
        config_hash(cfg)
    for name in ("study3", "study3_k_only"):
        loaded = study3(name, "search", "adamw")
        assert "study2_short" not in loaded
        assert config_hash(loaded)


@pytest.mark.parametrize("name", ("study3", "study3_k_only"))
@pytest.mark.parametrize("role", ROLES)
def test_run_configs_pass_the_protocol_check(name, role):
    check_run_config(study3(name, role, "adamw"))


def test_qualification_scope():
    q = load_config("experiments/qualification")
    assert q.bound_fraction == 0.01
    assert dict(q.k_star) == {1: 6.0, 2: 5.0}
    assert q.norm == 5.0e-05
    assert dict(q.gaussian) == {"s3": 3, "s4": 4}
    assert q.survival_share == 0.5
    assert q.rounding_scale == 10
    assert dict(q.warning) == {"floor": 0.001, "factor": 3}
    assert list(q.seeds.engineering) == [10000, 10001, 10002, 10003]
    assert list(q.seeds.numerical[1]) == [10200, 10201, 10202]
    assert list(q.seeds.numerical[2]) == [10210, 10211, 10212]
    assert q.checkpoint_gb == 6.5
    assert q.disk_margin_gb == 4
    assert dict(q.control) == {"arm": "adam_frozen", "steps": 25, "bound_fraction": 0.01}
    assert q.spec == "study 3, numerical qualification (docs/study3-preregistration.md)"
