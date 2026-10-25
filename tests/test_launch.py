"""The campaign launcher: checked configs, skipping completed runs, logging, carrying on."""

import json
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from zero_field_refusal.launch import (
    CampaignError,
    StopRequest,
    campaign_header,
    load_campaign,
    run_campaign,
)
from zero_field_refusal.protocol import ProtocolError
from zero_field_refusal.tracking import _RUN_ID as RUN_ID

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


def make_run_dir(root: Path, run, run_id: str, status: str = "completed", *, dirty: bool = False,
                 gate=None, k: float | None = 5.0, responses: bool = True) -> Path:  # fmt: skip
    """A run directory as the tracker would leave it for ``run``."""
    from omegaconf import OmegaConf

    from zero_field_refusal.protocol import load_run_config

    cfg = load_run_config(run.config, run.overrides)
    if gate is not None:
        OmegaConf.update(cfg, "tracking.gate", gate)
    run_dir = root / run.experiment / run_id
    (run_dir / "responses").mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    meta = {"status": status, "config_hash": run.config_hash,
            "git": {"commit": "abc123", "dirty": dirty}}  # fmt: skip
    (run_dir / "meta.json").write_text(json.dumps(meta))
    if k is not None:
        (run_dir / "summary.json").write_text(json.dumps({"K_harmful": k}))
    if responses:
        (run_dir / "responses" / f"{run.prompt_set}.jsonl").write_text("{}\n")
    return run_dir


class Executor:
    """Plays one scripted outcome per attempt: completed, diverged, failed or none (no run dir)."""

    def __init__(self, root: Path, outcomes: dict[str, list[str]]) -> None:
        self.root, self.outcomes, self.calls = root, outcomes, []

    def __call__(self, run, run_id: str, output: Path, campaign=None) -> int:
        self.calls.append((run.label, run_id))
        output.write_text("run output\n")
        outcome = self.outcomes.get(run.label, ["completed"]).pop(0)
        if outcome != "none":
            k = (0.0 if run.entry == "theta0" else 5.0) if outcome == "completed" else None
            make_run_dir(self.root, run, run_id, outcome, k=k)
        return 0 if outcome == "completed" else 1


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


def test_the_campaigns_header_goes_into_every_start(tmp_path) -> None:
    """The final review of gate B, I4: a campaign's optional ``header`` (study 2's records the
    w_max it was written under) is logged with every start; without one, nothing is added; a
    header that is not a mapping of plain values is refused."""
    path = campaign(tmp_path, "header: {w_max: 0.47, source: fixation}\n" + CELLS)
    assert campaign_header(path) == {"w_max": 0.47, "source": "fixation"}
    name, runs = load_campaign(path)
    root = tmp_path / "runs"
    run_campaign(name, runs[1:2], root, execute=Executor(root, {}),
                 header=campaign_header(path))  # fmt: skip
    starts = [json.loads(line) for line in (root / "launch" / f"{name}.jsonl").read_text()
              .splitlines() if json.loads(line)["event"] == "start"]  # fmt: skip
    assert len(starts) == 1 and starts[0]["header"] == {"w_max": 0.47, "source": "fixation"}
    plain = campaign(tmp_path, CELLS)
    assert campaign_header(plain) is None
    run_campaign(name, runs[2:3], root, execute=Executor(root, {}), header=None)
    last = json.loads((root / "launch" / f"{name}.jsonl").read_text().splitlines()[-2])
    assert last["event"] == "start" and "header" not in last
    for header in ("header: [0.47]\n", "header: {w_max: [0.47]}\n"):
        with pytest.raises(ValueError, match="plain values"):
            campaign_header(campaign(tmp_path, header + CELLS))


def test_runs_are_skipped_retried_or_left_by_their_outcome(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    theta0, adamw0, adamw1, sgd = runs
    root = tmp_path / "runs"
    make_run_dir(root, adamw0, "earlier-qualifying")  # done: skipped
    make_run_dir(root, sgd, f"x-{sgd.config_hash[:8]}-a1", "failed")  # one technical attempt
    make_run_dir(root, sgd, f"x-{sgd.config_hash[:8]}-a2", "running")  # killed: another
    execute = Executor(root, {"theta0/s0": ["failed", "none", "completed"],
                              "adamw-5e-5/s1": ["diverged"],
                              "sgd-0.1/s0": ["failed"]})  # fmt: skip
    result = run_campaign(name, runs, root, execute=execute)
    assert result == {"skipped": ["adamw-5e-5/s0"], "done": ["theta0/s0"],
                      "diverged": ["adamw-5e-5/s1"], "exhausted": ["sgd-0.1/s0"],
                      "stopped": [], "pending": []}  # fmt: skip
    assert [label for label, _ in execute.calls] == [
        "theta0/s0", "theta0/s0", "theta0/s0", "adamw-5e-5/s1", "sgd-0.1/s0"]  # fmt: skip
    ids = [run_id for _, run_id in execute.calls]
    assert len(set(ids)) == len(ids) and all(re.fullmatch(RUN_ID, i) for i in ids)
    log = [
        json.loads(line) for line in (root / "launch" / f"{name}.jsonl").read_text().splitlines()
    ]
    ends = [(r["run"], r["attempt"], r["outcome"]) for r in log if r["event"] == "end"]
    assert ends == [("theta0/s0", 1, "technical"), ("theta0/s0", 2, "technical"),
                    ("theta0/s0", 3, "done"), ("adamw-5e-5/s1", 1, "diverged"),
                    ("sgd-0.1/s0", 3, "technical")]  # fmt: skip

    # A second invocation finds everything settled: nothing runs again.
    again = Executor(root, {})
    result = run_campaign(name, runs, root, execute=again)
    assert again.calls == [] and result["exhausted"] == ["sgd-0.1/s0"]
    assert result["skipped"] == ["theta0/s0", "adamw-5e-5/s0"]


GATED = """
name: gated
cells:
  - name: adamw
    config: experiments/smoke
    overrides: [tracking.gate=1, tracking.require_clean_git=true]
    seeds: [0]
"""


@pytest.mark.parametrize(
    ("damage", "problem"),
    [({"dirty": True}, "clean checkout"), ({"gate": 2}, "tracking.gate is 2"),
     ({"k": float("nan")}, "K_harmful"), ({"k": None}, "K_harmful"),
     ({"responses": False}, "responses/harmful_judged.jsonl is missing")],
)  # fmt: skip
def test_a_completed_run_that_cannot_stand_stops_the_campaign(tmp_path, damage, problem) -> None:
    name, runs = load_campaign(campaign(tmp_path, GATED))
    root = tmp_path / "runs"
    make_run_dir(root, runs[0], "old", gate=damage.pop("gate", 1), **damage)
    execute = Executor(root, {})
    with pytest.raises(CampaignError, match=problem):
        run_campaign(name, runs, root, execute=execute)
    assert execute.calls == []


def test_a_qualifying_gate_run_is_skipped(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path, GATED))
    root = tmp_path / "runs"
    make_run_dir(root, runs[0], "old", gate=1)
    assert run_campaign(name, runs, root, execute=Executor(root, {}))["skipped"] == ["adamw/s0"]


@pytest.mark.parametrize("second", [{"dirty": True}, {}])
def test_a_second_completed_run_of_the_config_is_a_conflict(tmp_path, second) -> None:
    name, runs = load_campaign(campaign(tmp_path, GATED))
    root = tmp_path / "runs"
    make_run_dir(root, runs[0], "first", gate=1)
    make_run_dir(root, runs[0], "second", gate=1, **second)
    with pytest.raises(CampaignError, match="second"):
        run_campaign(name, runs, root, execute=Executor(root, {}))


def test_an_attempt_that_completes_without_qualifying_stops_the_campaign(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"

    def execute(run, run_id, output, campaign=None):
        make_run_dir(root, run, run_id, responses=False)
        return 0

    with pytest.raises(CampaignError, match="cannot stand for theta0/s0"):
        run_campaign(name, runs, root, execute=execute)
    log = [json.loads(line) for line in (root / "launch" / f"{name}.jsonl").read_text().split("\n")
           if line]  # fmt: skip
    assert log[-1]["outcome"] == "conflict"


def test_a_completed_run_with_zero_k_qualifies(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    make_run_dir(root, runs[3], "soft-targets-like", k=0.0)
    assert "sgd-0.1/s0" in run_campaign(name, runs, root, execute=Executor(root, {}))["skipped"]


def test_attempts_that_left_no_directory_count_across_invocations(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    crash = Executor(root, {r.label: ["none"] * 9 for r in runs})
    first = run_campaign(name, runs[:1], root, execute=crash)
    assert first["exhausted"] == ["theta0/s0"] and len(crash.calls) == 3
    second = run_campaign(name, runs[:1], root, execute=crash)
    assert second["exhausted"] == ["theta0/s0"] and len(crash.calls) == 3  # no new attempt


def test_one_recovery_round_per_cell_after_the_cause_is_fixed(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    cell = [r for r in runs if r.cell == "adamw-5e-5"]
    broken = Executor(root, {r.label: ["failed"] * 3 for r in cell})
    assert run_campaign(name, cell, root, execute=broken)["exhausted"] == [r.label for r in cell]
    assert run_campaign(name, cell, root, execute=broken)["exhausted"] == [r.label for r in cell]
    assert len(broken.calls) == 6  # three each, then nothing without a recovery round

    fixed = Executor(root, {cell[0].label: ["failed"] * 3, cell[1].label: ["failed", "diverged"]})
    result = run_campaign(name, cell, root, execute=fixed, recovery="disk quota raised")
    assert result["exhausted"] == [cell[0].label] and result["diverged"] == [cell[1].label]
    assert len(fixed.calls) == 5  # three more for one run, two for the other

    again = Executor(root, {})
    result = run_campaign(name, cell, root, execute=again, recovery="another try")
    assert again.calls == [] and result["exhausted"] == [cell[0].label]
    log = [json.loads(line) for line in (root / "launch" / f"{name}.jsonl").read_text().split("\n")
           if line]  # fmt: skip
    recoveries = [r for r in log if r["event"] == "recovery"]
    assert [(r["cell"], r["reason"]) for r in recoveries] == [("adamw-5e-5", "disk quota raised")]


def test_a_dry_run_starts_nothing(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    make_run_dir(root, runs[0], "done", k=0.0)

    def execute(run, run_id, output, campaign=None):
        raise AssertionError("a dry run must not start a run")

    result = run_campaign(name, runs, root, execute=execute, dry_run=True)
    assert result["skipped"] == ["theta0/s0"]
    assert result["pending"] == ["adamw-5e-5/s0", "adamw-5e-5/s1", "sgd-0.1/s0"]
    assert not (root / "launch").exists()


def test_each_attempt_runs_under_its_own_run_id(tmp_path, monkeypatch) -> None:
    from zero_field_refusal import launch

    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("RUN_ID", "the-launcher")
    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    _, runs = load_campaign(campaign(tmp_path))
    launch._subprocess(runs[1], "attempt-id", tmp_path / "out.log")
    assert seen["RUN_ID"] == "attempt-id"


def test_the_child_gets_its_campaign_in_the_environment(tmp_path, monkeypatch) -> None:
    """``execute`` receives the campaign's name, the run's label, the attempt and the header; the
    default executor passes them to the child as JSON in ``ZFR_CAMPAIGN``."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, "header: {versions_record: docs/x.json}\n" + CELLS)
    name, runs = load_campaign(path)
    root, received = tmp_path / "runs", []

    def spy(run, run_id, output, campaign):
        received.append(campaign)
        seen = {}

        def fake_run(command, **kwargs):
            seen.update(kwargs["env"])
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(launch.subprocess, "run", fake_run)
        launch._subprocess(run, run_id, output, campaign)
        received.append(json.loads(seen["ZFR_CAMPAIGN"]))
        return Executor(root, {})(run, run_id, output)

    run_campaign(name, runs[1:2], root, execute=spy, header=campaign_header(path))
    expected = {"name": name, "label": runs[1].label, "attempt": 1,
                "header": {"versions_record": "docs/x.json"}}  # fmt: skip
    assert received == [expected, expected]
    run_campaign(name, runs[2:3], root, execute=lambda *a: received.append(a[3]) or 1, header=None)
    assert received[-1]["header"] is None


CAMPAIGNS = sorted((Path(__file__).resolve().parents[1] / "configs" / "campaigns").glob("*.yaml"))


@pytest.mark.parametrize("path", CAMPAIGNS, ids=lambda p: p.name)
def test_the_committed_campaigns_load_and_pass_the_protocol(path) -> None:
    name, runs = load_campaign(path)
    assert name == path.stem and runs


def scheduled_run_dir(root: Path, run, name: str, *, status: str = "completed") -> Path:
    """Whole records of a scheduled run (as test_stage_a builds them) standing for ``run``."""
    from test_stage_a import FULL, write_run

    run_dir = write_run(root / run.experiment, name, arm="sgd_long", batch=5, lr=0.004,
                        seed=2000, reach=FULL)  # fmt: skip
    meta = json.loads((run_dir / "meta.json").read_text())
    meta.update(status=status, config_hash=run.config_hash, git={"commit": "abc", "dirty": False})
    (run_dir / "meta.json").write_text(json.dumps(meta))
    return run_dir


def test_a_scheduled_run_qualifies_by_its_whole_records(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    scheduled_run_dir(root, runs[1], "scheduled")
    result = run_campaign(name, runs, root, execute=Executor(root, {}))
    assert runs[1].label in result["skipped"]


@pytest.mark.parametrize(
    "damage",
    [
        lambda d: (d / "terminal.json").write_text("{"),
        lambda d: (d / "evaluations.jsonl").unlink(),
        lambda d: (d / "responses" / "harmful_evaluation.jsonl").unlink(),
    ],
)
def test_incomplete_scheduled_records_stop_the_campaign(tmp_path, damage) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    damage(scheduled_run_dir(root, runs[1], "scheduled"))
    with pytest.raises(CampaignError, match="records are incomplete|unreadable"):
        run_campaign(name, runs, root, execute=Executor(root, {}))


@pytest.mark.parametrize("status", ["failed", "running"])
def test_a_published_outcome_is_not_rerun_after_a_later_failure(tmp_path, status) -> None:
    # W&B or finalization failed after the terminal record was published.
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    scheduled_run_dir(root, runs[1], "scheduled", status=status)
    executor = Executor(root, {})
    result = run_campaign(name, runs, root, execute=executor)
    assert runs[1].label in result["skipped"]
    assert runs[1].label not in [label for label, _ in executor.calls]


def rayleigh_run_dir(root: Path, run) -> Path:
    """A scheduled run of a config that measures the endpoint Rayleigh quotient; its status is
    ``failed``, as after a measurement that failed once the outcome was published."""
    from omegaconf import OmegaConf

    run_dir = scheduled_run_dir(root, run, "scheduled", status="failed")
    cfg = OmegaConf.load(run_dir / "config.yaml")
    OmegaConf.update(cfg, "eval.rayleigh", {"enabled": True, "responses": 100}, force_add=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    return run_dir


def checkpointed_run_dir(root: Path, run, tmp_path: Path, *, record: str = "ok") -> Path:
    """A scheduled run of a config that sets ``tracking.endpoint_checkpoint``, with its file and
    the record in its summary as the run leaves them (damaged as ``record`` says)."""
    import hashlib

    from omegaconf import OmegaConf

    run_dir = scheduled_run_dir(root, run, "scheduled")
    cfg = OmegaConf.load(run_dir / "config.yaml")
    OmegaConf.update(cfg, "tracking.endpoint_checkpoint", str(tmp_path / "ckpt"), force_add=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    file = tmp_path / "ckpt" / "scheduled.pt"
    file.parent.mkdir()
    file.write_bytes(b"weights")
    entry = {"path": str(file), "sha256": hashlib.sha256(b"weights").hexdigest(),
             "step": 25, "names_sha256": "0" * 64}  # fmt: skip
    summary = (
        json.loads((run_dir / "summary.json").read_text())
        if (run_dir / "summary.json").exists()
        else {}
    )
    if record != "no_record":
        summary["endpoint_checkpoint"] = entry
    (run_dir / "summary.json").write_text(json.dumps(summary))
    if record == "missing_file":
        file.unlink()
    elif record == "other_bytes":
        file.write_bytes(b"other weights")
    return run_dir


def test_a_run_with_its_checkpoint_counts_as_done(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    checkpointed_run_dir(root, runs[1], tmp_path)
    result = run_campaign(name, runs, root, execute=Executor(root, {}))
    assert runs[1].label in result["skipped"]


@pytest.mark.parametrize(
    ("record", "problem"),
    [("no_record", "holds no endpoint-checkpoint record"),
     ("missing_file", "is missing"),
     ("other_bytes", "does not match its recorded sha256")],
)  # fmt: skip
def test_a_run_without_a_sound_checkpoint_is_a_conflict(tmp_path, record, problem) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    checkpointed_run_dir(root, runs[1], tmp_path, record=record)
    with pytest.raises(CampaignError, match=problem):
        run_campaign(name, runs, root, execute=Executor(root, {}))


@pytest.mark.parametrize(
    "record",
    [{"R": 12.5, "reason": None}, {"R": None, "reason": "zero displacement"},
     {"R": None, "reason": "failed: OutOfMemoryError: CUDA out of memory."}],
)  # fmt: skip
def test_a_rayleigh_record_with_r_or_its_reason_counts_as_done(tmp_path, record) -> None:
    # R is descriptive: a recorded failure is done, as a null R for a zero displacement is.
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    (rayleigh_run_dir(root, runs[1]) / "rayleigh.json").write_text(json.dumps(record))
    result = run_campaign(name, runs, root, execute=Executor(root, {}))
    assert runs[1].label in result["skipped"]


@pytest.mark.parametrize(
    ("text", "problem"),
    [(None, "rayleigh.json is missing or unreadable"),
     ("{", "rayleigh.json is missing or unreadable"),
     (json.dumps({"R": None, "reason": None}), "neither a finite R nor the reason"),
     (json.dumps({"R": float("nan"), "reason": None}), "neither a finite R nor the reason")],
)  # fmt: skip
def test_a_rayleigh_record_without_r_or_its_reason_stops_the_campaign(
    tmp_path, text, problem
) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    run_dir = rayleigh_run_dir(root, runs[1])
    if text is not None:
        (run_dir / "rayleigh.json").write_text(text)
    with pytest.raises(CampaignError, match=problem):
        run_campaign(name, runs, root, execute=Executor(root, {}))


STUDY2_THETA0 = """
name: study2-setup
cells:
  - name: theta0
    config: experiments/study2_stage_a
    entry: theta0
    seeds: [2000]
"""


def test_a_theta0_evaluation_of_a_scheduled_config_is_done_without_a_trajectory(tmp_path):
    # The θ₀ run inherits the schedule but trains nothing: its K and responses make it done.
    name, runs = load_campaign(campaign(tmp_path, STUDY2_THETA0))
    root = tmp_path / "runs"
    run_dir = make_run_dir(root, runs[0], "theta0-run", k=0.0)
    assert (run_dir / "responses" / "harmful_evaluation.jsonl").is_file()
    assert not (run_dir / "terminal.json").exists()
    result = run_campaign(name, runs, root, execute=Executor(root, {}))
    assert result["skipped"] == ["theta0/s2000"]


def launch_log(root: Path, name: str) -> list[dict]:
    return [
        json.loads(line) for line in (root / "launch" / f"{name}.jsonl").read_text().splitlines()
    ]


def test_a_stop_request_lets_the_running_attempt_finish_and_starts_no_other(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    execute = Executor(root, {"theta0/s0": ["failed", "completed"]})
    stop = lambda: len(execute.calls) >= 1  # noqa: E731 - requested during the first attempt
    result = run_campaign(name, runs, root, execute=execute, stop=stop)
    assert [label for label, _ in execute.calls] == ["theta0/s0"]  # no retry, no next run
    assert result["stopped"] == ["theta0/s0", "adamw-5e-5/s0", "adamw-5e-5/s1", "sgd-0.1/s0"]
    events = [(r["event"], r["run"]) for r in launch_log(root, name)]
    assert events == [("start", "theta0/s0"), ("end", "theta0/s0"), ("stop", "theta0/s0")]

    # Started again, the campaign carries on: the failed attempt counts, the rest run.
    again = Executor(root, {})
    result = run_campaign(name, runs, root, execute=again)
    assert result["done"] == ["theta0/s0", "adamw-5e-5/s0", "adamw-5e-5/s1", "sgd-0.1/s0"]
    assert [r["attempt"] for r in launch_log(root, name) if r["event"] == "start"] == [
        1,
        2,
        1,
        1,
        1,
    ]


def test_sigterm_reaches_only_the_launcher_and_the_child_finishes(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    finished = tmp_path / "finished"
    calls = []

    def execute(run, run_id, output, campaign=None):  # a child process that outlives the signal
        calls.append(run.label)
        child = "import sys, time; time.sleep(1.0); open(sys.argv[1], 'a').write('x')"
        code = subprocess.run([sys.executable, "-c", child, str(finished)], check=False).returncode
        make_run_dir(root, run, run_id, "completed", k=0.0 if run.entry == "theta0" else 5.0)
        return code

    stop = StopRequest.on_sigterm()
    try:
        threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGTERM)).start()
        result = run_campaign(name, runs, root, execute=execute, stop=stop)
    finally:
        signal.signal(signal.SIGTERM, stop.previous)
    assert stop() and calls == ["theta0/s0"] and finished.read_text() == "x"
    assert result["done"] == ["theta0/s0"] and len(result["stopped"]) == 3
    ends = [r for r in launch_log(root, name) if r["event"] == "end"]
    assert [(r["run"], r["returncode"], r["outcome"]) for r in ends] == [("theta0/s0", 0, "done")]
