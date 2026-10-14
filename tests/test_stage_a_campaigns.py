"""The campaigns of Stage A: written from the rules and the run records, loadable by the launcher
with every run's config checked against the protocol."""

import runpy
import shutil
from pathlib import Path

import pytest
from test_stage_a import SHORT, search

from zero_field_refusal.analysis.stage_a import control_schedule, load_trajectories
from zero_field_refusal.launch import load_campaign

SCRIPT = Path(__file__).parents[1] / "scripts" / "stage_a_campaigns.py"


def write(monkeypatch, tmp_path: Path, *args: str) -> list:
    out = tmp_path / f"{args[0]}-{len(list(tmp_path.glob('*.yaml')))}.yaml"
    monkeypatch.setattr("sys.argv", [str(SCRIPT), *args, "--out", str(out)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return load_campaign(out)[1]


def labels(runs) -> list[str]:
    return [run.label for run in runs]


def test_the_fixed_campaigns(monkeypatch, tmp_path) -> None:
    assert labels(write(monkeypatch, tmp_path, "setup")) == ["theta0/s2000"]
    screen = write(monkeypatch, tmp_path, "screen")
    assert len(screen) == 20 and all(run.seed == 2000 for run in screen)
    grids = write(monkeypatch, tmp_path, "grids", "--seed", "2001")
    assert len(grids) == 8 + 24 and all(run.seed == 2001 for run in grids)
    assert "frozen_adam_25-b5-5e-06/s2001" in labels(grids)


def test_sgd_discovery_takes_the_screen_candidates(monkeypatch, tmp_path) -> None:
    search(tmp_path / "runs")
    runs = write(monkeypatch, tmp_path, "sgd", "--runs", str(tmp_path / "runs"), "--seed", "2002")
    assert labels(runs) == [
        "sgd_long-b5-0.004/s2002",
        "sgd_long-b5-0.003/s2002",
        "sgd_long-b20-0.004/s2002",
        "sgd_long-b20-0.003/s2002",
    ]
    shutil.rmtree(next((tmp_path / "runs").rglob("screen-20-0.01")))
    with pytest.raises(SystemExit):  # an incomplete screen names no candidate
        write(monkeypatch, tmp_path, "sgd", "--runs", str(tmp_path / "runs"), "--seed", "2002")


def test_extensions_are_the_rates_the_runs_call_for(monkeypatch, tmp_path) -> None:
    search(tmp_path / "runs", short={5: SHORT | {0.5e-5: 5.5}, 20: SHORT})
    runs = write(
        monkeypatch, tmp_path, "extension", "--runs", str(tmp_path / "runs"), "--seed", "2000"
    )
    assert labels(runs) == ["adamw_25-b5-2.5e-06/s2000", "frozen_adam_25-b5-2.5e-06/s2000"]


def test_controls_follow_the_regular_runs(monkeypatch, tmp_path) -> None:
    search(tmp_path / "runs")
    shutil.rmtree(tmp_path / "runs" / "controls")
    runs = write(monkeypatch, tmp_path, "controls", "--runs", str(tmp_path / "runs" / "discovery"))
    assert sorted(labels(runs)) == [
        "control-b20-0.003/s2000",
        "control-b20-0.004/s2000",
        "control-b5-0.003/s2000",
        "control-b5-0.004/s2000",
    ]
    regular = [
        r
        for r in load_trajectories(tmp_path / "runs" / "discovery")
        if (r.arm, r.batch_size, r.lr) == ("sgd_long", 20, 0.004)
    ]
    plan = control_schedule(regular)
    control = next(run for run in runs if run.label == "control-b20-0.004/s2000")
    assert f"train.steps={plan['steps']}" in control.overrides
    assert f"schedule.evaluate_at=[{','.join(map(str, plan['seeds'][2000]))}]" in control.overrides
