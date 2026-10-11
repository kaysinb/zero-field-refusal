"""The campaign launcher: checked configs, skipping completed runs, logging, carrying on."""

import json
from pathlib import Path

import pytest

from zero_field_refusal.launch import completed_hashes, load_campaign, run_campaign
from zero_field_refusal.protocol import ProtocolError

CELLS = """
name: test-campaign
cells:
  - name: theta0
    config: experiments/smoke
    entry: theta0
    seeds: [0]
  - name: adamw-5e-5
    config: experiments/smoke
    overrides: [optimizer.lr=5.0e-5]
    seeds: [0, 1]
  - name: sgd-0.1
    config: experiments/smoke
    overrides: [optimizer=sgd, optimizer.lr=0.1]
    seeds: [0]
"""


def campaign(tmp_path: Path, text: str = CELLS) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(text)
    return path


def write_meta(root: Path, experiment: str, run: str, config_hash: str, status: str) -> None:
    directory = root / experiment / run
    directory.mkdir(parents=True)
    (directory / "meta.json").write_text(json.dumps({"status": status,
                                                     "config_hash": config_hash}))  # fmt: skip


def test_a_campaign_expands_into_checked_runs(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    assert name == "test-campaign"
    assert [run.label for run in runs] == ["theta0/s0", "adamw-5e-5/s0", "adamw-5e-5/s1",
                                           "sgd-0.1/s0"]  # fmt: skip
    assert {run.experiment for run in runs} == {"smoke"}
    assert len({run.config_hash for run in runs}) == 4  # seed and run_type are in the hash
    theta0, adamw = runs[0], runs[1]
    assert theta0.command()[1].endswith("scripts/evaluate_theta0.py")
    assert "run_type=theta0" not in theta0.command()  # the script sets it
    assert adamw.command()[1].endswith("scripts/train.py")
    assert adamw.command()[2:] == ["experiments/smoke", "optimizer.lr=5.0e-5", "seed=0"]


@pytest.mark.parametrize(
    ("cells", "error", "match"),
    [("[{name: a, config: experiments/smoke, seeds: [0], lr: 1}]", ValueError, "unknown keys"),
     ("[{name: a, config: experiments/smoke, seeds: [0]}, "
      "{name: a, config: experiments/smoke, seeds: [1]}]", ValueError, "Two cells"),
     ("[{name: a, config: experiments/smoke, seeds: [0], overrides: [seed=3]}]", ValueError,
      "seed and run_type"),
     ("[{name: a, config: experiments/smoke, seeds: []}]", ValueError, "non-empty list"),
     ("[{name: a, config: experiments/smoke, seeds: [0, 0]}]", ValueError, "seed twice"),
     ("[{name: a, config: experiments/smoke, seeds: [0], entry: eval}]", ValueError, "entry"),
     ("[{name: a, config: experiments/smoke, seeds: [0]}, "
      "{name: b, config: experiments/smoke, seeds: [0]}]", ValueError, "same config"),
     ("[{name: a, config: experiments/smoke, seeds: [0], "
      "overrides: [train.loss.normalization=token_count]}]", ProtocolError, "departs"),
     ("[{name: a, config: experiments/smoke, seeds: [0], overrides: [optimizer=sgd]}]",
      ValueError, "optimizer.lr")],
)  # fmt: skip
def test_a_wrong_cell_stops_the_campaign_before_anything_runs(tmp_path, cells, error,
                                                               match) -> None:  # fmt: skip
    with pytest.raises(error, match=match):
        load_campaign(campaign(tmp_path, f"name: c\ncells: {cells}\n"))


def test_completed_runs_are_skipped_and_failures_do_not_stop_the_rest(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    write_meta(root, "smoke", "done", runs[1].config_hash, "completed")
    write_meta(root, "smoke", "crashed", runs[2].config_hash, "failed")  # does not count
    assert completed_hashes(root, "smoke") == {runs[1].config_hash}

    started = []

    def execute(run, output: Path) -> int:
        started.append(run.label)
        output.write_text("run output\n")
        return 1 if run.cell == "theta0" else 0

    result = run_campaign(name, runs, root, execute=execute)
    assert started == ["theta0/s0", "adamw-5e-5/s1", "sgd-0.1/s0"]
    assert result == {"skipped": ["adamw-5e-5/s0"], "done": ["adamw-5e-5/s1", "sgd-0.1/s0"],
                      "failed": ["theta0/s0"], "pending": []}  # fmt: skip
    log = [json.loads(line) for line in (root / "launch" / f"{name}.jsonl").read_text().split("\n")
           if line]  # fmt: skip
    assert [(r["event"], r["run"]) for r in log] == [
        ("start", "theta0/s0"), ("end", "theta0/s0"), ("start", "adamw-5e-5/s1"),
        ("end", "adamw-5e-5/s1"), ("start", "sgd-0.1/s0"), ("end", "sgd-0.1/s0")]  # fmt: skip
    assert log[1]["returncode"] == 1
    outputs = sorted(p.name for p in (root / "launch" / name).iterdir())
    assert len(outputs) == 3 and outputs[0].startswith("adamw-5e-5-s1-")


def test_a_dry_run_starts_nothing(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    write_meta(root, "smoke", "done", runs[0].config_hash, "completed")

    def execute(run, output):
        raise AssertionError("a dry run must not start a run")

    result = run_campaign(name, runs, root, execute=execute, dry_run=True)
    assert result["skipped"] == ["theta0/s0"]
    assert result["pending"] == ["adamw-5e-5/s0", "adamw-5e-5/s1", "sgd-0.1/s0"]
    assert not (root / "launch").exists()


CAMPAIGNS = sorted((Path(__file__).resolve().parents[1] / "configs" / "campaigns").glob("*.yaml"))


@pytest.mark.parametrize("path", CAMPAIGNS, ids=lambda p: p.name)
def test_the_committed_campaigns_load_and_pass_the_protocol(path) -> None:
    name, runs = load_campaign(path)
    assert name == path.stem and runs
