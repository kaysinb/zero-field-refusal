"""The campaign, its header and the host in every run's meta.json."""

import json
import platform
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from zero_field_refusal.tracking import Run

HEADER = {"versions_record": "docs/x.json", "versions_record_sha256": "ab"}
CAMPAIGN = {"name": "c", "label": "l", "attempt": 1, "header": HEADER}


@pytest.fixture(autouse=True)
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    for key in ("RUN_ID", "RANK", "WORLD_SIZE", "ZFR_CAMPAIGN", "RUNPOD_POD_ID", "RUNPOD_DC_ID"):
        monkeypatch.delenv(key, raising=False)


def meta_of(run_id: str, tmp_path: Path) -> dict:
    return json.loads((tmp_path / "unit" / run_id / "meta.json").read_text())


def make_cfg():
    return OmegaConf.create(
        {"experiment": "unit", "seed": 0, "tracking": {"wandb": {"enabled": False}}}
    )


def test_meta_records_the_campaign_its_header_the_pod_and_the_host(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ZFR_CAMPAIGN", json.dumps(CAMPAIGN))
    monkeypatch.setenv("RUNPOD_POD_ID", "p1")
    monkeypatch.setenv("RUNPOD_DC_ID", "EU-RO-1")
    with Run(make_cfg(), run_id="r1"):
        pass
    meta = meta_of("r1", tmp_path)
    assert meta["campaign"] == CAMPAIGN
    assert meta["campaign"]["header"]["versions_record_sha256"] == "ab"
    assert meta["launch"]["pod_id"] == "p1" and meta["launch"]["data_center"] == "EU-RO-1"
    assert meta["host"] == {"hostname": platform.node()}


def test_without_the_variables_the_fields_are_none(tmp_path) -> None:
    with Run(make_cfg(), run_id="r1"):
        pass
    meta = meta_of("r1", tmp_path)
    assert meta["campaign"] is None
    assert meta["launch"]["pod_id"] is None and meta["launch"]["data_center"] is None
    assert meta["host"]["hostname"] == platform.node()


@pytest.mark.parametrize("value", ["{", "[1]", '"c"'])
def test_an_unusable_campaign_raises_and_leaves_no_run_directory(tmp_path, monkeypatch,
                                                                 value) -> None:  # fmt: skip
    monkeypatch.setenv("ZFR_CAMPAIGN", value)
    with pytest.raises(ValueError, match="ZFR_CAMPAIGN"):
        Run(make_cfg(), run_id="r1")
    assert not (tmp_path / "unit").exists()
