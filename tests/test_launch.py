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
    _numerics_problem,
    campaign_header,
    expected_numerics,
    hold,
    label_starts,
    launch_records,
    load_campaign,
    open_start,
    run_campaign,
    supersede,
)
from zero_field_refusal.protocol import ProtocolError, load_run_config
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


@pytest.mark.parametrize("name", ["7", "[a, b]", "{a: 1}", "null"])
def test_a_cell_name_that_is_not_a_string_is_a_campaign_error(tmp_path, monkeypatch,
                                                                name) -> None:  # fmt: skip
    """A cell whose name parses to a number, a list, a mapping or null: the campaign is refused
    as a campaign error naming the cell by its position, before anything is launched or
    written; the script exits with that message, no traceback."""
    import importlib.util

    path = campaign(tmp_path, "name: c\ncells:\n  - {name: ok, config: experiments/smoke, "
                              f"seeds: [0]}}\n  - {{name: {name}, config: experiments/smoke, "
                              "seeds: [1]}\n")  # fmt: skip
    with pytest.raises(CampaignError, match="cell 2 of the campaign: its name must be a string"):
        load_campaign(path)
    where = Path(__file__).resolve().parents[1] / "scripts" / "launch.py"
    spec = importlib.util.spec_from_file_location("launch_script", where)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    with pytest.raises(SystemExit, match="cell 2 of the campaign: its name must be a string"):
        script.main([str(path)])
    assert not (tmp_path / "runs").exists()


def test_the_campaigns_header_goes_into_every_start(tmp_path) -> None:
    """A campaign's optional ``header`` (study 2's records the
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
    meta = json.loads((run_dir / "meta.json").read_text())  # the numerics its campaign asks for
    meta.setdefault("inputs", {})["numerics_setup"] = {
        **expected_numerics(load_run_config(run.config, run.overrides)),
        "cublas_workspace_config": None, "memory_fraction": None, "device_index": 0}  # fmt: skip
    (run_dir / "meta.json").write_text(json.dumps(meta))
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


def test_a_missing_checkpoint_conflict_points_to_the_lost_checkpoint_procedure(tmp_path) -> None:
    """A missing endpoint checkpoint is a possible loss, not a run to move or rerun: the message
    holds the conflict line and a note naming the procedure. Another conflict has no note."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    checkpointed_run_dir(root, runs[1], tmp_path, record="missing_file")
    with pytest.raises(CampaignError, match="cannot stand") as lost:
        run_campaign(name, runs, root, execute=Executor(root, {}))
    text = str(lost.value)
    assert re.search(r"^- \S+/\S+: the endpoint checkpoint \S+ is missing$", text, re.M)
    assert text.count("a possible loss requiring diagnosis") == 1
    assert '(docs/operations.md, "A lost checkpoint")' in " ".join(text.split("\n"))
    assert "unless finish removed it" in text and "lost container disk" not in text
    note = " ".join(text.splitlines()[2:])
    assert note.endswith(".") and ". " not in note  # one sentence
    assert all(len(line) <= 100 for line in text.splitlines()[2:])

    other = tmp_path / "other"
    (other / "runs").mkdir(parents=True)
    name, runs = load_campaign(campaign(other))
    checkpointed_run_dir(other / "runs", runs[1], other, record="other_bytes")
    with pytest.raises(CampaignError, match="does not match its recorded sha256") as mismatch:
        run_campaign(name, runs, other / "runs", execute=Executor(other / "runs", {}))
    assert "possible loss" not in str(mismatch.value) and "A lost checkpoint" not in str(
        mismatch.value
    )


def test_expected_numerics_is_unchanged_by_the_v0_policy(tmp_path) -> None:
    """A run of code older than the v₀ policy does not depart from what it was issued: judged
    against the key, its declared stop, a scientific failure, would be superseded instead of kept
    whole (``qualify.py``, ``resolve_launcher_run``). The harness's own check compares the key."""
    from omegaconf import OmegaConf

    _, runs = load_campaign(campaign(tmp_path))
    run = runs[1]
    run_dir = checkpointed_run_dir(tmp_path / "runs", run, tmp_path)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    assert set(expected_numerics(cfg)) == {"deterministic", "tf32_matmul", "tf32_cudnn",
                                           "matmul_precision", "memory_limit_gb"}  # fmt: skip
    meta = json.loads((run_dir / "meta.json").read_text())
    assert "v0_sdpa_backends" not in meta["inputs"]["numerics_setup"]
    assert _numerics_problem(run, cfg, meta) is None
    meta["inputs"]["numerics_setup"]["v0_sdpa_backends"] = ["MATH"]  # new code's record too
    assert _numerics_problem(run, cfg, meta) is None


@pytest.mark.parametrize(
    "departure",
    [{"deterministic": True}, {"tf32_matmul": True}, {"matmul_precision": "high"},
     {"memory_limit_gb": 46.0}, None],
)  # fmt: skip
def test_a_checkpoint_run_under_other_numerics_is_a_conflict(tmp_path, departure) -> None:
    """The recorded numerics setup against what the campaign's config asks for, read without
    CUDA: determinism, TF32 and its matmul precision, the memory limit; none recorded at all is a
    conflict too. The device-dependent values (fraction, index) are not compared here."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    run_dir = checkpointed_run_dir(root, runs[1], tmp_path)
    meta = json.loads((run_dir / "meta.json").read_text())
    if departure is None:
        del meta["inputs"]["numerics_setup"]
    else:
        meta["inputs"]["numerics_setup"].update(departure)
    (run_dir / "meta.json").write_text(json.dumps(meta))
    match = "records no numerics setup" if departure is None else "numerics setup departs"
    with pytest.raises(CampaignError, match=match):
        run_campaign(name, runs, root, execute=Executor(root, {}))


@pytest.mark.parametrize(
    "overrides",
    [[], ["model.tf32=true"], ["execution.deterministic=true"], ["execution.memory_limit_gb=3"]],
)
def test_the_expected_numerics_are_what_the_setup_records(overrides) -> None:
    """``expected_numerics`` reads the config as ``runtime.configure_numerics`` sets it (on the
    CPU here: the device-independent fields). The setup runs in a fresh process: with
    determinism on it needs the cuBLAS workspace set before any CUDA work, which earlier tests of
    this process may have done, and it leaves this process's flags alone."""
    from zero_field_refusal.config import load_config
    from zero_field_refusal.paths import REPO_ROOT

    program = (
        "import json, sys\n"
        "from zero_field_refusal import runtime\n"
        "from zero_field_refusal.config import load_config\n"
        "cfg = load_config('experiments/smoke', json.loads(sys.argv[1]))\n"
        "print(json.dumps(runtime.configure_numerics(cfg, 'cpu')))\n"
    )
    env = {key: value for key, value in os.environ.items() if key != "CUBLAS_WORKSPACE_CONFIG"}
    done = subprocess.run([sys.executable, "-c", program, json.dumps(overrides)], cwd=REPO_ROOT,
                          capture_output=True, text=True, env=env)  # fmt: skip
    assert done.returncode == 0, done.stderr
    setup = json.loads(done.stdout.splitlines()[-1])
    cfg = load_config("experiments/smoke", overrides)  # TF32 on departs from the protocol
    wanted = expected_numerics(cfg)
    assert {key: setup[key] for key in wanted} == wanted


def test_a_settled_run_that_lost_only_its_checkpoint_records_exactly_the_missing_checkpoint(
    tmp_path,
) -> None:
    """The settle of a start without an end, over whole scheduled records of which only the
    checkpoint file is gone, judges the run with the missing-checkpoint problem alone (no check
    before it reads the container disk): the end it writes is a settled ``conflict`` whose problem
    satisfies ``is_missing_checkpoint``, which is what lets ``restart-case`` tell it from any other
    conflict. It pins the problem text and the predicate together, not the order of the checks."""
    from zero_field_refusal.launch import is_missing_checkpoint

    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    run_dir = checkpointed_run_dir(root, runs[1], tmp_path, record="missing_file")
    log_start(root, name, runs[1], run_dir.name)  # the launcher died with the container
    with pytest.raises(CampaignError, match="is missing"):
        run_campaign(name, runs, root, execute=Executor(root, {}))
    (end,) = [r for r in launch_records(root) if r.get("event") == "end"]
    assert end["settled"] is True and end["returncode"] is None and end["outcome"] == "conflict"
    assert is_missing_checkpoint(end["problem"])


def test_a_superseded_run_no_longer_counts_and_its_recovery_starts_anew(tmp_path) -> None:
    """A completed run named superseded in the journal (a case restart after its checkpoint was
    lost) stays on disk and in the journal, is a used attempt, and is neither done nor a
    conflict: the run's recovery round starts a new attempt."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    run_dir = checkpointed_run_dir(root, runs[1], tmp_path, record="missing_file")
    log = root / "launch" / f"{name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps({"event": "start", "run": runs[1].label, "run_id": run_dir.name,
                               "experiment": runs[1].experiment,
                               "config_hash": runs[1].config_hash}) + "\n")  # fmt: skip
    only = {"only": runs[1].label, "max_attempts": 1}
    with pytest.raises(CampaignError, match="is missing"):  # the lost checkpoint: a conflict
        run_campaign(name, runs, root, execute=Executor(root, {}), **only)
    supersede(root, name, runs[1], run_dir.name, reason="the Pod stopped", record="restart-1")
    executor = Executor(root, {})
    result = run_campaign(name, runs, root, execute=executor, **only)
    assert result["exhausted"] == [runs[1].label] and executor.calls == []  # one attempt used
    result = run_campaign(name, runs, root, execute=executor, recovery="the Pod stopped",
                          incident=INCIDENT, **only)  # fmt: skip
    assert result["done"] == [runs[1].label] and len(executor.calls) == 1
    assert run_dir.is_dir()  # kept
    events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
    assert events[:4] == ["start", "end", "superseded", "recovery"]  # the conflict was ended


def stopped_by(run_dir: Path, reason: str, *, record: str) -> None:
    """Give a checkpointed run dir the terminal stop ``reason`` and, per ``record``, a null
    checkpoint record (``null``) or a record whose file is gone (``missing_file``)."""
    terminal = json.loads((run_dir / "terminal.json").read_text())
    terminal["stop"] = {"reason": reason, "kind": "other"}
    (run_dir / "terminal.json").write_text(json.dumps(terminal))
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["terminal"] = terminal
    if record == "null":
        summary["endpoint_checkpoint"] = None
    else:
        Path(summary["endpoint_checkpoint"]["path"]).unlink()
    (run_dir / "summary.json").write_text(json.dumps(summary))


@pytest.mark.parametrize(
    ("reason", "record"),
    [("gradient_spike", "null"), ("nonfinite_weight", "null"), ("k_above_max", "missing_file")],
)
def test_an_instability_stop_is_the_outcome_not_a_conflict(
    tmp_path, monkeypatch, reason, record
) -> None:
    monkeypatch.setattr("zero_field_refusal.launch.load_trajectory", lambda d: None)  # the rule
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    stopped_by(checkpointed_run_dir(root, runs[1], tmp_path), reason, record=record)
    executor = Executor(root, {})
    result = run_campaign(name, runs, root, execute=executor)
    assert runs[1].label in result["skipped"]
    assert runs[1].label not in [label for label, _ in executor.calls]


@pytest.mark.parametrize("reason", ["steps", "t_max"])
def test_a_null_record_after_an_ordinary_stop_is_a_conflict(tmp_path, monkeypatch, reason) -> None:
    monkeypatch.setattr("zero_field_refusal.launch.load_trajectory", lambda d: None)  # the rule
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    stopped_by(checkpointed_run_dir(root, runs[1], tmp_path), reason, record="null")
    with pytest.raises(CampaignError, match="holds no endpoint-checkpoint record"):
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


# ------------------------------------------------------- one label at a time, and the attempts


def test_only_runs_and_assesses_exactly_one_label(tmp_path, monkeypatch) -> None:
    """``only``: that label alone is assessed and run; another run's conflict (a completed run
    whose checkpoint was deleted, say) is neither read nor raised, and the journal records the
    label with every start and end."""
    from zero_field_refusal import launch

    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    make_run_dir(root, runs[3], "conflicting", responses=False)  # sgd-0.1/s0 cannot stand
    with pytest.raises(CampaignError, match="conflicting"):
        run_campaign(name, runs, root, execute=Executor(root, {}))
    assessed = []
    real = launch.assess
    monkeypatch.setattr(launch, "assess",
                        lambda run, *a: assessed.append(run.label) or real(run, *a))  # fmt: skip
    execute = Executor(root, {})
    result = run_campaign(name, runs, root, execute=execute, only="adamw-5e-5/s1")
    assert assessed == ["adamw-5e-5/s1"]
    assert [label for label, _ in execute.calls] == ["adamw-5e-5/s1"]
    assert result["done"] == ["adamw-5e-5/s1"] and not result["skipped"]
    assert sum(result.values(), []) == ["adamw-5e-5/s1"]
    logged = [r for r in launch_log(root, name) if r["event"] in ("start", "end")]
    assert [(r["event"], r["run"], r["only"]) for r in logged] == [
        ("start", "adamw-5e-5/s1", "adamw-5e-5/s1"),
        ("end", "adamw-5e-5/s1", "adamw-5e-5/s1"),
    ]
    with pytest.raises(CampaignError, match="no run labelled"):
        run_campaign(name, runs, root, execute=execute, only="adamw-5e-5/s9")


def test_max_attempts_one_stops_after_one_failed_attempt(tmp_path) -> None:
    """``max_attempts=1`` (the qualification's launcher run): a failed attempt is not retried,
    not in this invocation nor in the next; a recovery round with its reason gives that run
    alone one more, and under ``only`` the round is the run's own, not its cell's."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    first, second = runs[1], runs[2]  # one cell, two seeds
    broken = Executor(root, {first.label: ["failed"] * 3, second.label: ["failed"] * 3})
    result = run_campaign(name, runs, root, execute=broken, only=first.label, max_attempts=1)
    assert result["exhausted"] == [first.label] and len(broken.calls) == 1
    again = run_campaign(name, runs, root, execute=broken, only=first.label, max_attempts=1)
    assert again["exhausted"] == [first.label] and len(broken.calls) == 1
    run_campaign(name, runs, root, execute=broken, only=first.label, max_attempts=1,
                 recovery="the Pod's deadline stopped attempt 1", incident=INCIDENT)  # fmt: skip
    assert [label for label, _ in broken.calls] == [first.label] * 2
    # The other seed of the cell has no round of its own: one attempt, no retry
    run_campaign(name, runs, root, execute=broken, only=second.label, max_attempts=1)
    assert [label for label, _ in broken.calls] == [first.label] * 2 + [second.label]
    recoveries = [r for r in launch_log(root, name) if r["event"] == "recovery"]
    assert [(r["runs"], r["only"]) for r in recoveries] == [([first.label], first.label)]
    starts = [r for r in launch_log(root, name) if r["event"] == "start"]
    assert {r["max_attempts"] for r in starts} == {1}
    with pytest.raises(CampaignError, match="max_attempts"):
        run_campaign(name, runs, root, execute=broken, max_attempts=0)


def test_the_default_attempts_and_records_are_unchanged(tmp_path) -> None:
    """Without ``only`` and ``max_attempts``, the start and end records carry neither key."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    run_campaign(name, runs[1:2], root, execute=Executor(root, {}))
    for record in launch_log(root, name):
        assert "only" not in record and "max_attempts" not in record


def test_the_launch_script_passes_only_and_max_attempts(tmp_path, monkeypatch) -> None:
    import importlib.util

    from zero_field_refusal import launch

    where = Path(__file__).resolve().parents[1] / "scripts" / "launch.py"
    spec = importlib.util.spec_from_file_location("launch_script", where)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    seen = {}

    def fake(name, runs, root, **kwargs):
        seen.update(kwargs)
        return {key: [] for key in ("skipped", "done", "diverged", "exhausted", "stopped",
                                    "pending")}  # fmt: skip

    monkeypatch.setattr(script, "run_campaign", fake)
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    path = campaign(tmp_path)
    script.main([str(path), "--only", "adamw-5e-5/s0", "--max-attempts", "1"])
    assert seen["only"] == "adamw-5e-5/s0" and seen["max_attempts"] == 1
    script.main([str(path)])
    assert seen["only"] is None and seen["max_attempts"] == launch.MAX_ATTEMPTS


def test_the_child_never_inherits_a_stale_campaign(tmp_path, monkeypatch) -> None:
    """Without a campaign, a ``ZFR_CAMPAIGN`` of the launcher's own environment is removed from
    the child's: the child would record a campaign it is not part of."""
    from zero_field_refusal import launch

    seen = {}

    def fake_run(command, **kwargs):
        seen.update(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("ZFR_CAMPAIGN", json.dumps({"name": "stale"}))
    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    _, runs = load_campaign(campaign(tmp_path))
    launch._subprocess(runs[1], "attempt-id", tmp_path / "out.log")
    assert "ZFR_CAMPAIGN" not in seen and seen["RUN_ID"] == "attempt-id"


def test_an_attempt_needs_its_campaigns_name() -> None:
    import inspect

    from zero_field_refusal import launch

    parameter = inspect.signature(launch._attempt).parameters["name"]
    assert parameter.default is inspect.Parameter.empty


# ------------------------------------------- a start found without its end gets the launcher's end


def log_start(root: Path, name: str, run, run_id: str, attempt: int = 1, **extra) -> dict:
    """Log a start of ``run`` as the launcher writes it, with nothing after it (it died)."""
    record = {"event": "start", "run": run.label, "attempt": attempt, "run_id": run_id,
              "experiment": run.experiment, "config_hash": run.config_hash,
              "command": run.command()[1:], **extra}  # fmt: skip
    log = root / "launch" / f"{name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def ends(root: Path, name: str) -> list[dict]:
    return [r for r in launch_log(root, name) if r["event"] == "end"]


def test_a_published_run_found_without_its_end_gets_a_settled_done_and_is_skipped(
    tmp_path,
) -> None:
    """The launcher died after its child published: the relaunch writes the attempt's end
    (same run id and attempt, no return code, ``settled``, the outcome ``done``) and skips the
    run; a third invocation finds the end and writes nothing."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    run_id = f"x-{run.config_hash[:8]}-a1"
    make_run_dir(root, run, run_id)
    log_start(root, name, run, run_id)
    executor = Executor(root, {})
    result = run_campaign(name, [run], root, execute=executor)
    assert result["skipped"] == [run.label] and executor.calls == []
    assert [(r["event"], r["run_id"]) for r in launch_log(root, name)] == [
        ("start", run_id), ("end", run_id)]  # fmt: skip
    (end,) = ends(root, name)
    assert (end["run"], end["attempt"], end["returncode"], end["outcome"], end["problem"],
            end["settled"]) == (run.label, 1, None, "done", None, True)  # fmt: skip
    assert "only" not in end
    run_campaign(name, [run], root, execute=executor)
    assert len(launch_log(root, name)) == 2


def test_a_diverged_run_found_without_its_end_gets_a_settled_diverged(tmp_path) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    run_id = f"x-{run.config_hash[:8]}-a1"
    make_run_dir(root, run, run_id, "diverged", k=None)
    log_start(root, name, run, run_id)
    result = run_campaign(name, [run], root, execute=Executor(root, {}))
    assert result["diverged"] == [run.label]
    assert [(r["outcome"], r["settled"]) for r in ends(root, name)] == [("diverged", True)]


def test_a_non_conforming_run_found_without_its_end_gets_a_settled_conflict_first(tmp_path) -> None:
    """The end, with the problem that stops the campaign, is written before ``CampaignError`` is
    raised, and nothing starts."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    run_id = f"x-{run.config_hash[:8]}-a1"
    make_run_dir(root, run, run_id, responses=False)
    log_start(root, name, run, run_id)
    executor = Executor(root, {})
    with pytest.raises(CampaignError, match="responses/harmful_judged.jsonl is missing"):
        run_campaign(name, [run], root, execute=executor)
    assert executor.calls == []
    (end,) = ends(root, name)
    assert (end["run_id"], end["attempt"], end["outcome"], end["settled"]) == (
        run_id, 1, "conflict", True)  # fmt: skip
    assert "responses/harmful_judged.jsonl is missing" in end["problem"]
    assert launch_log(root, name)[-1] == end


def test_every_assessed_start_is_ended_before_a_conflict_is_raised(tmp_path) -> None:
    """The ends of all the assessed runs are written before the conflict of one is raised."""
    name, runs = load_campaign(campaign(tmp_path))
    broken, fine, root = runs[1], runs[2], tmp_path / "runs"
    for run, damage in ((broken, {"responses": False}), (fine, {})):
        run_id = f"x-{run.config_hash[:8]}-a1"
        make_run_dir(root, run, run_id, **damage)
        log_start(root, name, run, run_id)
    with pytest.raises(CampaignError, match="cannot stand"):
        run_campaign(name, [broken, fine], root, execute=Executor(root, {}))
    assert [(r["run"], r["outcome"]) for r in ends(root, name)] == [
        (broken.label, "conflict"), (fine.label, "done")]  # fmt: skip


@pytest.mark.parametrize("directory", ["none", "failed", "running"])
def test_a_start_without_a_directory_or_a_finished_run_gets_a_settled_technical_end(
    tmp_path, directory
) -> None:
    """No run directory, or one whose status says it did not finish: a technical end, then a new
    attempt (the second), which the technical attempt had already used."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    run_id = f"x-{run.config_hash[:8]}-a1"
    if directory != "none":
        make_run_dir(root, run, run_id, directory)
    log_start(root, name, run, run_id)
    executor = Executor(root, {})
    result = run_campaign(name, [run], root, execute=executor)
    assert result["done"] == [run.label] and len(executor.calls) == 1
    first, second = ends(root, name)
    assert (first["run_id"], first["attempt"], first["outcome"], first["settled"]) == (
        run_id, 1, "technical", True)  # fmt: skip
    assert first["returncode"] is None and first["problem"]
    assert (second["attempt"], second["outcome"], "settled" in second) == (2, "done", False)
    assert [(r["event"], r["attempt"]) for r in launch_log(root, name)] == [
        ("start", 1), ("end", 1), ("start", 2), ("end", 2)]  # fmt: skip


def test_an_existing_end_is_never_rewritten(tmp_path) -> None:
    """An attempt with its end is left as it is, whatever its directory says now: the journal is
    the same, byte for byte, after the run is skipped."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    run_id = f"x-{run.config_hash[:8]}-a1"
    make_run_dir(root, run, run_id)  # it qualifies now; its end says it failed
    log_start(root, name, run, run_id)
    end = {"event": "end", "run": run.label, "attempt": 1, "run_id": run_id, "returncode": 1,
           "outcome": "technical", "problem": "status 'failed'"}  # fmt: skip
    log = root / "launch" / f"{name}.jsonl"
    with log.open("a") as f:
        f.write(json.dumps(end) + "\n")
    before = log.read_text()
    result = run_campaign(name, [run], root, execute=Executor(root, {}))
    assert result["skipped"] == [run.label] and log.read_text() == before


def test_a_start_that_a_later_record_of_its_label_follows_is_not_ended(tmp_path) -> None:
    """A start that its attempt's ``superseded`` record names (by its run id) gets no settled
    end; the run's next attempt is a start and an end of its own."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    run_id = f"x-{run.config_hash[:8]}-a1"
    log_start(root, name, run, run_id)
    supersede(root, name, run, run_id, reason="a restart", record="restart-1")
    result = run_campaign(name, [run], root, execute=Executor(root, {}))
    assert result["done"] == [run.label]
    assert [(r["event"], r.get("settled")) for r in launch_log(root, name)] == [
        ("start", None), ("superseded", None), ("start", None), ("end", None)]  # fmt: skip


@pytest.mark.parametrize(("first", "second"), [("z-original", "a-recovery"),
                                               ("a-original", "z-recovery")])  # fmt: skip
def test_a_start_in_another_journal_is_settled_whatever_the_file_names(
    tmp_path, first, second
) -> None:
    """Attempt 1's start and end are in one journal, attempt 2's start, without its end, in
    another (its recovery, logged under another campaign name). Whichever name sorts first, the
    end is matched to attempt 1 by its run id and attempt 2 is the open start: the launcher
    writes its settled end (``done``: it completed) and skips the run."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    one, two = (f"x-{run.config_hash[:8]}-a{n}" for n in (1, 2))
    make_run_dir(root, run, one, "failed")
    log_start(root, first, run, one)
    end = {"event": "end", "run": run.label, "attempt": 1, "run_id": one, "returncode": 1,
           "outcome": "technical", "problem": "status 'failed'"}  # fmt: skip
    with (root / "launch" / f"{first}.jsonl").open("a") as f:
        f.write(json.dumps(end) + "\n")
    make_run_dir(root, run, two)  # completed, and it qualifies
    log_start(root, second, run, two, attempt=2)
    assert open_start(run, launch_records(root))["run_id"] == two
    executor = Executor(root, {})
    result = run_campaign(second, [run], root, execute=executor)
    assert result["skipped"] == [run.label] and executor.calls == []
    (settled,) = [r for r in launch_records(root) if r.get("settled")]
    assert (settled["run_id"], settled["attempt"], settled["outcome"]) == (two, 2, "done")
    assert open_start(run, launch_records(root)) is None


def test_a_labels_attempts_are_ordered_by_number_then_time(tmp_path) -> None:
    """``label_starts`` orders a label's starts by attempt number, then time, whatever their
    order in the records; ``open_start`` is the latest one that no end, ``superseded`` or
    ``held`` record names (an earlier attempt left open included)."""
    name, runs = load_campaign(campaign(tmp_path))
    run = runs[1]

    def start(attempt, time, run_id):
        return {"event": "start", "run": run.label, "attempt": attempt, "run_id": run_id,
                "experiment": run.experiment, "config_hash": run.config_hash,
                "time": time}  # fmt: skip

    records = [start(2, "2026-10-26T10:00:00+00:00", "b"), start(1, "2026-10-26T09:00:00+00:00",
               "a2"), start(1, "2026-10-26T08:00:00+00:00", "a1")]  # fmt: skip
    assert [r["run_id"] for r in label_starts(run, records)] == ["a1", "a2", "b"]
    assert open_start(run, records)["run_id"] == "b"
    closed = [*records, {"event": "end", "run_id": "b"}, {"event": "held", "run_id": "a2"}]
    assert open_start(run, closed)["run_id"] == "a1"
    assert open_start(run, [*closed, {"event": "superseded", "run_id": "a1"}]) is None


def test_a_hold_under_another_config_hash_refuses_its_label(tmp_path) -> None:
    """A ``held`` record of the label written under another config hash (the cell's config has
    changed since): the label is a conflict naming the held run and the reason, with or without
    ``only``, its recovery round included and in a dry run; no attempt starts and nothing is
    written."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    earlier = SimpleNamespace(label=run.label, experiment=run.experiment, config_hash="0" * 12)
    hold(root, name, earlier, "r-old", reason="no recorded commit", record="supersede-1.json")
    log = root / "launch" / f"{name}.jsonl"
    before, executor = log.read_text(), Executor(root, {})
    for kwargs in ({}, {"only": run.label}, {"only": run.label, "max_attempts": 1,
                                             "recovery": "the cause is fixed"},
                   {"dry_run": True}, {"only": run.label, "dry_run": True}):  # fmt: skip
        with pytest.raises(CampaignError, match="r-old: held, its outcome unresolved: no "
                                                "recorded commit"):  # fmt: skip
            run_campaign(name, runs, root, execute=executor, **kwargs)
    assert executor.calls == [] and log.read_text() == before
    other = runs[2]  # another label of the campaign is not held
    assert run_campaign(name, [other], root, execute=executor)["done"] == [other.label]


def test_under_only_the_open_starts_of_other_labels_are_untouched(tmp_path) -> None:
    """The label alone is assessed and ended, and its settled end carries ``only``; a dry run
    assesses and writes nothing."""
    name, runs = load_campaign(campaign(tmp_path))
    mine, other, root = runs[1], runs[2], tmp_path / "runs"
    for run in (mine, other):
        run_id = f"x-{run.config_hash[:8]}-a1"
        make_run_dir(root, run, run_id)
        log_start(root, name, run, run_id)
    log = root / "launch" / f"{name}.jsonl"
    before = log.read_text()
    run_campaign(name, runs, root, execute=Executor(root, {}), only=mine.label, dry_run=True)
    assert log.read_text() == before
    result = run_campaign(name, runs, root, execute=Executor(root, {}), only=mine.label)
    assert result["skipped"] == [mine.label]
    (end,) = ends(root, name)
    assert (end["run"], end["outcome"], end["only"], end["settled"]) == (
        mine.label, "done", mine.label, True)  # fmt: skip
    opened = {r["run"] for r in launch_log(root, name) if r["event"] == "start"} - {mine.label}
    assert opened == {other.label} and len(launch_log(root, name)) == 3
    run_campaign(name, [mine], root, execute=Executor(root, {}))  # a run left out is not assessed
    assert len(launch_log(root, name)) == 3


# ------------------------------------------- further recovery rounds: one per documented incident

R1 = "attempt 1: the Pod stopped at 03:12Z and its container disk was lost"
R2 = "attempt 2: the Pod stopped again at 05:40Z and its container disk was lost"
INCIDENT = {"incident_time": "2026-10-27T03:12:00+00:00",
            "cause": "the Pod stopped: a host maintenance event",
            "evidence": "the Pod's event log, pod-1, 2026-10-27"}  # fmt: skip


def own_round(run, reason: str, *, max_attempts: int = 1, **incident) -> dict:
    """``run_campaign``'s keywords for a round of the run's own: ``only``, the attempts, the
    reason and the incident it documents (``INCIDENT`` changed by ``incident``; None drops a
    field)."""
    documented = {key: value for key, value in {**INCIDENT, **incident}.items()
                  if value is not None}  # fmt: skip
    return {"only": run.label, "max_attempts": max_attempts, "recovery": reason,
            "incident": documented}  # fmt: skip


def rounds_of(root: Path, name: str) -> list[dict]:
    return [r for r in launch_log(root, name) if r["event"] == "recovery"]


def run_ids(execute: Executor) -> list[str]:
    return [run_id for _, run_id in execute.calls]


@pytest.mark.parametrize("second", ["failed", "superseded"])
def test_a_second_loss_after_the_runs_round_opens_a_further_round(tmp_path, second) -> None:
    """Under ``only`` at one attempt: attempt 1 fails; round 1 allows attempt 2, which fails, or
    completes and is then superseded (a case restart after a second lost disk). A round with a
    new reason is round 2: it names the attempts it follows and allows attempt 3 alone. The
    label, the run ids' form and every earlier byte of the journal stay."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    then = "failed" if second == "failed" else "completed"
    execute = Executor(root, {run.label: ["failed", then, "completed"]})
    only = {"only": run.label, "max_attempts": 1}
    assert run_campaign(name, runs, root, execute=execute, **only)["exhausted"] == [run.label]
    result = run_campaign(name, runs, root, execute=execute, **own_round(run, R1))
    if second == "superseded":
        assert result["done"] == [run.label]
        supersede(root, name, run, run_ids(execute)[1], reason=R2, record="restart-2.json")
    else:
        assert result["exhausted"] == [run.label]
    assert run_campaign(name, runs, root, execute=execute, **only)["exhausted"] == [run.label]
    log = root / "launch" / f"{name}.jsonl"
    before = log.read_bytes()
    result = run_campaign(name, runs, root, execute=execute, **own_round(run, R2))
    assert result["done"] == [run.label] and len(execute.calls) == 3
    assert log.read_bytes().startswith(before)
    ids = run_ids(execute)
    assert [(r["round"], r["after"], r["reason"], r["granted"], r["max_attempts"])
            for r in rounds_of(root, name)] == [(1, ids[:1], R1, 1, 1),
                                                (2, ids[:2], R2, 1, 1)]  # fmt: skip
    for r in rounds_of(root, name):
        assert (r["only"], r["runs"], r["cell"], r["experiment"], r["config_hash"]) == (
            run.label, [run.label], run.cell, run.experiment, run.config_hash)  # fmt: skip
        assert {key: r[key] for key in INCIDENT} == INCIDENT
    starts = [r for r in launch_log(root, name) if r["event"] == "start"]
    assert [(r["run"], r["attempt"], r["run_id"]) for r in starts] == [
        (run.label, n, ids[n - 1]) for n in (1, 2, 3)]  # fmt: skip
    assert [run_id.rsplit("-", 1)[1] for run_id in ids] == ["a1", "a2", "a3"]
    assert run_campaign(name, runs, root, execute=execute, **only)["skipped"] == [run.label]


def test_a_further_round_needs_a_reason_no_earlier_round_gave(tmp_path) -> None:
    """Round 1's reason again (spaced otherwise too), or a blank reason: refused, naming the
    label and round 1 by its log and line; nothing is written and nothing runs."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    execute = Executor(root, {run.label: ["failed", "failed", "completed"]})
    run_campaign(name, runs, root, execute=execute, only=run.label, max_attempts=1)
    run_campaign(name, runs, root, execute=execute, **own_round(run, R1))
    log = root / "launch" / f"{name}.jsonl"
    before = log.read_bytes()
    for reason in (R1, f"  {R1}\t"):
        with pytest.raises(CampaignError, match=rf"round 1 \({name}\.jsonl line 3\)") as refusal:
            run_campaign(name, runs, root, execute=execute, **own_round(run, reason))
        assert run.label in str(refusal.value) and "--recovery" in str(refusal.value)
    for reason in ("", "  \n"):
        with pytest.raises(CampaignError, match="needs its reason") as refusal:
            run_campaign(name, runs, root, execute=execute, **own_round(run, reason))
        assert run.label in str(refusal.value)
    assert log.read_bytes() == before and len(execute.calls) == 2
    result = run_campaign(name, runs, root, execute=execute, **own_round(run, R2))
    assert result["done"] == [run.label] and len(execute.calls) == 3


def test_a_further_round_waits_for_the_previous_rounds_attempt(tmp_path) -> None:
    """Round 1 is logged and its attempt not yet started (a stop came first): a new reason
    writes no round, and the attempt runs under round 1's allowance: rounds never stack."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    execute = Executor(root, {run.label: ["failed", "failed"]})
    run_campaign(name, runs, root, execute=execute, only=run.label, max_attempts=1)
    result = run_campaign(name, runs, root, execute=execute, stop=lambda: True,
                          **own_round(run, R1))  # fmt: skip
    assert result["stopped"] == [run.label] and len(execute.calls) == 1
    result = run_campaign(name, runs, root, execute=execute, **own_round(run, R2))
    assert result["exhausted"] == [run.label] and len(execute.calls) == 2
    assert [r["reason"] for r in rounds_of(root, name)] == [R1]
    again = run_campaign(name, runs, root, execute=execute, only=run.label, max_attempts=1)
    assert again["exhausted"] == [run.label] and len(execute.calls) == 2


@pytest.mark.parametrize("first", ["written", "before numbering"])
def test_a_further_round_after_six_attempts_allows_exactly_one_more(tmp_path, first) -> None:
    """Six attempts under the three-attempt policy (three, then round 1's three; round 1
    written now, with ``granted: 3``, or before rounds were numbered): a further round at three
    attempts is refused and writes nothing; round 2 at one attempt allows exactly one more, and
    no later ``max_attempts`` recalculates what the rounds granted."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    log = root / "launch" / f"{name}.jsonl"
    execute = Executor(root, {run.label: ["failed"] * 8})
    run_campaign(name, runs, root, execute=execute, only=run.label)  # three attempts
    if first == "written":
        run_campaign(name, runs, root, execute=execute, **own_round(run, R1, max_attempts=3))
        (written,) = rounds_of(root, name)
        assert (written["round"], written["granted"], written["after"]) == (
            1, 3, run_ids(execute)[:3])  # fmt: skip
    else:
        from zero_field_refusal import launch

        launch._log(log, {"event": "recovery", "experiment": run.experiment, "cell": run.cell,
                          "reason": R1, "runs": [run.label], "only": run.label})  # fmt: skip
        run_campaign(name, runs, root, execute=execute, only=run.label)
    assert len(execute.calls) == 6
    before = log.read_bytes()
    with pytest.raises(CampaignError, match="--max-attempts 1") as refusal:
        run_campaign(name, runs, root, execute=execute, **own_round(run, R2, max_attempts=3))
    assert run.label in str(refusal.value) and "round 2" in str(refusal.value)
    assert log.read_bytes() == before and len(execute.calls) == 6
    result = run_campaign(name, runs, root, execute=execute, **own_round(run, R2))
    assert result["exhausted"] == [run.label] and len(execute.calls) == 7
    last = rounds_of(root, name)[-1]
    assert (last["round"], last["granted"], last["after"]) == (2, 1, run_ids(execute)[:6])
    for attempts in (1, 3, 9):
        result = run_campaign(name, runs, root, execute=execute, only=run.label,
                              max_attempts=attempts)  # fmt: skip
        assert result["exhausted"] == [run.label] and len(execute.calls) == 7
    assert [r["attempt"] for r in launch_log(root, name) if r["event"] == "start"] == [
        1, 2, 3, 4, 5, 6, 7]  # fmt: skip


def lost_checkpoint(tmp_path: Path, record: str):
    """An attempt whose checkpoint run (``checkpointed_run_dir``) is damaged as ``record``."""

    def execute(run, run_id, output, campaign=None) -> int:
        checkpointed_run_dir(tmp_path / "runs", run, tmp_path, record=record)
        return 0

    return execute


def published_but_failed(tmp_path: Path):
    """An attempt that published its terminal record, then failed with its records incomplete:
    no outcome the launcher reads, and still a published record."""

    def execute(run, run_id, output, campaign=None) -> int:
        run_dir = scheduled_run_dir(tmp_path / "runs", run, run_id, status="failed")
        (run_dir / "evaluations.jsonl").unlink()
        return 1

    return execute


def moved_conflict(tmp_path: Path):
    """An attempt that completed without qualifying (a conflict, ended so), whose directory was
    then moved out of the experiment."""

    def execute(run, run_id, output, campaign=None) -> int:
        make_run_dir(tmp_path / "runs", run, run_id, "completed", responses=False)
        return 0

    return execute


OUTCOMES = ["done", "diverged", "held", "missing checkpoint", "other sha256", "published record",
            "moved conflict"]  # fmt: skip


@pytest.mark.parametrize("case", OUTCOMES)
def test_a_further_round_never_reruns_an_outcome(tmp_path, case) -> None:
    """After round 1, its attempt done, diverged, held, a provenance conflict (a missing
    checkpoint, one of another sha256), a published terminal record with incomplete records, or a
    conflict moved out of the experiment: a round with a new reason is skipped, diverged or
    refused as each already is, and no ``recovery`` record is written."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    run_campaign(name, runs, root, execute=Executor(root, {run.label: ["failed"]}),
                 only=run.label, max_attempts=1)  # fmt: skip
    then = {"done": Executor(root, {run.label: ["completed"]}),
            "diverged": Executor(root, {run.label: ["diverged"]}),
            "held": Executor(root, {run.label: ["failed"]}),
            "missing checkpoint": lost_checkpoint(tmp_path, "missing_file"),
            "other sha256": lost_checkpoint(tmp_path, "other_bytes"),
            "published record": published_but_failed(tmp_path),
            "moved conflict": moved_conflict(tmp_path)}[case]  # fmt: skip
    try:
        run_campaign(name, runs, root, execute=then, **own_round(run, R1))
    except CampaignError:
        assert case == "moved conflict"
    if case == "held":
        (second,) = run_ids(then)
        hold(root, name, run, second, reason="no recorded commit", record="supersede-1.json")
    if case == "moved conflict":
        (moved,) = [d for d in (root / run.experiment).iterdir() if d.name.endswith("-a2")]
        moved.rename(tmp_path / moved.name)
    log = root / "launch" / f"{name}.jsonl"
    before, execute = log.read_bytes(), Executor(root, {})
    expected = {"done": "skipped", "diverged": "diverged"}
    if case in expected:
        result = run_campaign(name, runs, root, execute=execute, **own_round(run, R2))
        assert result[expected[case]] == [run.label]
    else:
        match = {"held": "held, its outcome unresolved", "missing checkpoint": "is missing",
                 "other sha256": "does not match its recorded sha256",
                 "published record": "its terminal record is published",
                 "moved conflict": "its end records the outcome conflict"}[case]  # fmt: skip
        with pytest.raises(CampaignError, match=match):
            run_campaign(name, runs, root, execute=execute, **own_round(run, R2))
    assert execute.calls == [] and [r["reason"] for r in rounds_of(root, name)] == [R1]
    written = [json.loads(line)["event"] for line in log.read_bytes()[len(before) :].splitlines()]
    assert written in ([], ["end"])  # nothing but the settled end of an attempt found without one


def test_rounds_under_another_config_hash_give_this_run_nothing(tmp_path) -> None:
    """A label's round under its config hash X; the campaign now has hash Y (a fix changed a
    hashed key). Y's first attempt fails and a relaunch runs nothing, at any ``max_attempts``;
    a round with a new reason is the label's round 2, after every start of the label, and Y's
    attempts are numbered on from the label's highest."""
    from zero_field_refusal import launch

    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    log = root / "launch" / f"{name}.jsonl"
    old = "0" * 12

    def start(attempt: int, run_id: str) -> None:
        launch._log(log, {"event": "start", "run": run.label, "attempt": attempt,
                          "run_id": run_id, "experiment": run.experiment, "config_hash": old,
                          "only": run.label, "max_attempts": 1})  # fmt: skip
        launch._log(log, {"event": "end", "run": run.label, "attempt": attempt,
                          "run_id": run_id, "returncode": 1, "outcome": "technical",
                          "problem": "status 'failed'", "only": run.label})  # fmt: skip

    start(1, "x-a1")
    launch._log(log, {"event": "recovery", "experiment": run.experiment, "cell": run.cell,
                      "reason": R1, "runs": [run.label], "only": run.label, "round": 1,
                      "config_hash": old, "after": ["x-a1"], "max_attempts": 1, "granted": 1,
                      **INCIDENT})  # fmt: skip
    start(2, "x-a2")
    execute = Executor(root, {run.label: ["failed", "completed"]})
    only = {"only": run.label, "max_attempts": 1}
    assert run_campaign(name, runs, root, execute=execute, **only)["exhausted"] == [run.label]
    for attempts in ({}, {"max_attempts": 3}):
        result = run_campaign(name, runs, root, execute=execute, only=run.label, **attempts)
        assert result["exhausted"] == [run.label] and len(execute.calls) == 1
    with pytest.raises(CampaignError, match=r"round 1 \("):  # its reason, under any hash
        run_campaign(name, runs, root, execute=execute, **own_round(run, R1))
    result = run_campaign(name, runs, root, execute=execute, **own_round(run, R2))
    assert result["done"] == [run.label]
    ids = run_ids(execute)
    assert [i.rsplit("-", 1)[1] for i in ids] == ["a3", "a4"]
    last = rounds_of(root, name)[-1]
    assert (last["round"], last["after"], last["config_hash"], last["granted"]) == (
        2, ["x-a1", "x-a2", ids[0]], run.config_hash, 1)  # fmt: skip


def test_attempt_numbers_rise_per_label_across_config_hashes(tmp_path) -> None:
    """A new start is numbered after every start of its label, whatever its config hash; under
    one config hash the numbers are those of before (one more than the attempts used)."""
    name, runs = load_campaign(campaign(tmp_path))
    mine, other, root = runs[1], runs[2], tmp_path / "runs"
    earlier = SimpleNamespace(label=mine.label, experiment=mine.experiment,
                              config_hash="0" * 12, command=mine.command)  # fmt: skip
    log_start(root, name, earlier, "x-a1", attempt=1)
    log_start(root, name, earlier, "x-a2", attempt=2)
    execute = Executor(root, {mine.label: ["failed"] * 3, other.label: ["failed"] * 3})
    run_campaign(name, [mine, other], root, execute=execute)
    starts = [(r["run"], r["attempt"]) for r in launch_log(root, name)
              if r["event"] == "start" and r["run_id"] not in ("x-a1", "x-a2")]  # fmt: skip
    assert starts == [(mine.label, 3), (mine.label, 4), (mine.label, 5),
                      (other.label, 1), (other.label, 2), (other.label, 3)]  # fmt: skip
    assert [i.rsplit("-", 1)[1] for i in run_ids(execute)] == ["a3", "a4", "a5", "a1", "a2", "a3"]


def test_a_cell_round_after_a_labels_own_round_grants_it_nothing(tmp_path) -> None:
    """A label that had a round of its own: a cell-wide recovery call whose round would cover it
    is refused, naming the label, its round and the form of a further round, and writes
    nothing; a cell round in the journal after the label's round (as an older launcher opened
    it) gives that label no attempt, and the cell's other run its round's attempts."""
    from zero_field_refusal import launch

    name, runs = load_campaign(campaign(tmp_path))
    mine, other, root = runs[1], runs[2], tmp_path / "runs"  # one cell, two seeds
    log = root / "launch" / f"{name}.jsonl"
    execute = Executor(root, {mine.label: ["failed"] * 4, other.label: ["failed"] * 6})
    run_campaign(name, runs, root, execute=execute, only=mine.label, max_attempts=1)
    run_campaign(name, runs, root, execute=execute, **own_round(mine, R1))
    run_campaign(name, [other], root, execute=execute)  # its three attempts
    assert len(execute.calls) == 5
    before = log.read_bytes()
    reason = "the disk quota is raised"
    with pytest.raises(CampaignError, match=rf"round 1 \({name}\.jsonl line 3\)") as refusal:
        run_campaign(name, [mine, other], root, execute=execute, recovery=reason)
    assert mine.label in str(refusal.value) and f"--only {mine.label}" in str(refusal.value)
    assert log.read_bytes() == before and len(execute.calls) == 5
    launch._log(log, {"event": "recovery", "experiment": mine.experiment, "cell": mine.cell,
                      "reason": reason, "runs": [mine.label, other.label]})  # fmt: skip
    result = run_campaign(name, [mine, other], root, execute=execute)
    assert result["exhausted"] == [mine.label, other.label]
    assert [label for label, _ in execute.calls[5:]] == [other.label] * 3


@pytest.mark.parametrize("change", [
    {"incident_time": None}, {"cause": None}, {"evidence": None}, {"cause": "  "},
    {"evidence": ""}, {"incident_time": "yesterday"}, {"incident_time": "2026-10-27T03:12:00"},
    {"incident_time": "2026-10-27T05:12:00+02:00"}, {"incident_time": "2999-01-01T00:00:00Z"},
    {"cause": "the Pod stopped\nand restarted"}, {"incident_time": 3},
])  # fmt: skip
def test_a_round_of_a_label_records_its_incident(tmp_path, change) -> None:
    """A run's own round records its incident: the UTC time (ISO 8601, not later than the
    round), the external cause on one line and the evidence. Any of them missing, blank or
    malformed: refused, naming the field; nothing is written and nothing runs. A time with
    ``Z`` is UTC."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    execute = Executor(root, {run.label: ["failed", "completed"]})
    run_campaign(name, runs, root, execute=execute, only=run.label, max_attempts=1)
    log = root / "launch" / f"{name}.jsonl"
    before = log.read_bytes()
    (field,) = change
    with pytest.raises(CampaignError, match=field) as refusal:
        run_campaign(name, runs, root, execute=execute, **own_round(run, R1, **change))
    assert run.label in str(refusal.value)
    with pytest.raises(CampaignError, match="incident_time"):
        run_campaign(name, runs, root, execute=execute,
                     **{**own_round(run, R1), "incident": None})  # fmt: skip
    assert log.read_bytes() == before and len(execute.calls) == 1
    zulu = own_round(run, R1, incident_time="2026-10-27T03:12:00Z")
    assert run_campaign(name, runs, root, execute=execute, **zulu)["done"] == [run.label]
    assert rounds_of(root, name)[0]["incident_time"] == "2026-10-27T03:12:00Z"


def test_an_incident_goes_with_a_runs_own_round(tmp_path) -> None:
    """The incident documents a run's own round (``only``): given without it, refused."""
    name, runs = load_campaign(campaign(tmp_path))
    with pytest.raises(CampaignError, match="only"):
        run_campaign(name, runs, tmp_path / "runs", execute=Executor(tmp_path, {}),
                     recovery=R1, incident=INCIDENT)  # fmt: skip


def test_the_launch_script_passes_the_incident(tmp_path, monkeypatch) -> None:
    import importlib.util

    where = Path(__file__).resolve().parents[1] / "scripts" / "launch.py"
    spec = importlib.util.spec_from_file_location("launch_script", where)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    seen = {}

    def fake(name, runs, root, **kwargs):
        seen.update(kwargs)
        return {key: [] for key in ("skipped", "done", "diverged", "exhausted", "stopped",
                                    "pending")}  # fmt: skip

    monkeypatch.setattr(script, "run_campaign", fake)
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    path = campaign(tmp_path)
    only = ["--only", "adamw-5e-5/s0", "--max-attempts", "1", "--recovery", R1]
    incident = ["--incident-time", INCIDENT["incident_time"], "--cause", INCIDENT["cause"],
                "--evidence", INCIDENT["evidence"]]  # fmt: skip
    script.main([str(path), *only, *incident])
    assert seen["recovery"] == R1 and seen["incident"] == INCIDENT
    script.main([str(path), *only])
    assert seen["incident"] is None
    script.main([str(path), *only, "--cause", "x"])  # the launcher names what is missing
    assert seen["incident"] == {"cause": "x"}
    for argv in ([str(path), *incident], [str(path), "--recovery", R1, *incident]):
        with pytest.raises(SystemExit) as error:  # without --recovery, or without --only
            script.main(argv)
        assert error.value.code == 2


def test_a_numbered_round_that_names_no_attempts_grants_nothing(tmp_path) -> None:
    """A round of the run's own written since rounds are numbered, but without a list of the
    attempts it follows (not as the launcher writes it): it grants no attempt, and the run is
    exhausted until a round of its own is opened as the launcher writes it."""
    from zero_field_refusal import launch

    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    execute = Executor(root, {run.label: ["failed", "completed"]})
    only = {"only": run.label, "max_attempts": 1}
    run_campaign(name, runs, root, execute=execute, **only)
    launch._log(root / "launch" / f"{name}.jsonl",
                {"event": "recovery", "experiment": run.experiment, "cell": run.cell,
                 "reason": R1, "runs": [run.label], "only": run.label, "round": 1,
                 "config_hash": run.config_hash, "after": "all of them", "max_attempts": 1,
                 "granted": 1, **INCIDENT})  # fmt: skip
    assert run_campaign(name, runs, root, execute=execute, **only)["exhausted"] == [run.label]
    assert len(execute.calls) == 1
    result = run_campaign(name, runs, root, execute=execute, **own_round(run, R2))
    assert result["done"] == [run.label] and rounds_of(root, name)[-1]["round"] == 2


# ------------------------- grants by rule; no call runs more attempts than its own max_attempts


def test_a_first_round_at_three_attempts_grants_at_most_the_labels_base(tmp_path) -> None:
    """Attempt 1 at one attempt fails; the label's first round, asked at ``--max-attempts 3``,
    grants the smaller of 3 and the label's base (its first start's 1): exactly one attempt runs
    for the one documented incident, and the round records ``granted: 1``."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    execute = Executor(root, {run.label: ["failed"] * 4})
    run_campaign(name, runs, root, execute=execute, only=run.label, max_attempts=1)
    result = run_campaign(name, runs, root, execute=execute, **own_round(run, R1, max_attempts=3))
    assert result["exhausted"] == [run.label] and len(execute.calls) == 2
    (written,) = rounds_of(root, name)
    assert (written["round"], written["max_attempts"], written["granted"]) == (1, 3, 1)


def test_a_stopped_round_then_the_drivers_call_runs_one_attempt(tmp_path) -> None:
    """The same round, stopped before its attempt, then the driver's own call (``--only L
    --max-attempts 1``, no ``--recovery``): one attempt runs, and no later call runs another."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    execute = Executor(root, {run.label: ["failed"] * 4})
    only = {"only": run.label, "max_attempts": 1}
    run_campaign(name, runs, root, execute=execute, **only)
    result = run_campaign(name, runs, root, execute=execute, stop=lambda: True,
                          **own_round(run, R1, max_attempts=3))  # fmt: skip
    assert result["stopped"] == [run.label] and len(execute.calls) == 1
    for _ in range(3):
        assert run_campaign(name, runs, root, execute=execute, **only)["exhausted"] == [run.label]
    assert len(execute.calls) == 2


def test_a_hand_made_further_round_granting_five_allows_one_attempt(tmp_path) -> None:
    """A round 2 written by hand with ``granted: 5``: grants come from the rule, never from the
    journal, so it allows exactly one attempt, at any ``max_attempts``."""
    from zero_field_refusal import launch

    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    execute = Executor(root, {run.label: ["failed"] * 8})
    run_campaign(name, runs, root, execute=execute, only=run.label, max_attempts=1)
    run_campaign(name, runs, root, execute=execute, **own_round(run, R1))
    launch._log(root / "launch" / f"{name}.jsonl",
                {"event": "recovery", "experiment": run.experiment, "cell": run.cell,
                 "reason": R2, "runs": [run.label], "only": run.label, "round": 2,
                 "config_hash": run.config_hash, "after": run_ids(execute), "max_attempts": 1,
                 "granted": 5, **INCIDENT})  # fmt: skip
    for attempts in (3, 9, 1):
        result = run_campaign(name, runs, root, execute=execute, only=run.label,
                              max_attempts=attempts)  # fmt: skip
        assert result["exhausted"] == [run.label]
    assert len(execute.calls) == 3


def test_no_call_runs_more_attempts_than_its_own_max_attempts(tmp_path) -> None:
    """Three attempts at the default, then round 1 at three attempts (``granted: 3``), stopped
    before its attempts: a call at one attempt runs one, one at two runs two, and the next runs
    nothing. The attempts left stay for later calls; no grant is recalculated."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    execute = Executor(root, {run.label: ["failed"] * 7})
    run_campaign(name, runs, root, execute=execute, only=run.label)
    run_campaign(name, runs, root, execute=execute, stop=lambda: True,
                 **own_round(run, R1, max_attempts=3))  # fmt: skip
    assert rounds_of(root, name)[0]["granted"] == 3 and len(execute.calls) == 3
    for attempts, total in ((1, 4), (2, 6), (3, 6)):
        run_campaign(name, runs, root, execute=execute, only=run.label, max_attempts=attempts)
        assert len(execute.calls) == total


STUDY3_CELLS = CELLS.replace("name: test-campaign\n", "name: test-campaign\nheader: {study: 3, "
                             "candidate: 1, step: search, round: 0}\n")  # fmt: skip


def test_a_study3_campaign_takes_no_cell_round(tmp_path, monkeypatch) -> None:
    """A campaign whose header names study 3: ``--recovery`` without ``--only`` is refused, exit
    1, with the command form of a run's own round (``--only LABEL --max-attempts 1 --recovery``
    and the incident), and writes nothing; study 2's cell round (a header without ``study``)
    opens as before."""
    import importlib.util

    path = campaign(tmp_path, STUDY3_CELLS)
    name, runs = load_campaign(path)
    root = tmp_path / "runs"
    cell = [r for r in runs if r.cell == "adamw-5e-5"]
    header = campaign_header(path)
    broken = Executor(root, {r.label: ["failed"] * 6 for r in cell})
    run_campaign(name, cell, root, execute=broken, header=header)
    log = root / "launch" / f"{name}.jsonl"
    before = log.read_bytes()
    with pytest.raises(CampaignError, match="--only LABEL --max-attempts 1 --recovery") as refusal:
        run_campaign(name, cell, root, execute=broken, header=header, recovery="disk replaced")
    assert "--incident-time" in str(refusal.value) and "--evidence" in str(refusal.value)
    where = Path(__file__).resolve().parents[1] / "scripts" / "launch.py"
    spec = importlib.util.spec_from_file_location("launch_script", where)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setenv("RUNS_DIR", str(root))
    with pytest.raises(SystemExit) as stop:
        script.main([str(path), "--recovery", "disk replaced"])
    assert isinstance(stop.value.code, str) and "--only LABEL" in stop.value.code  # exit 1
    assert log.read_bytes() == before and len(broken.calls) == 6
    study2 = tmp_path / "study2"
    study2.mkdir()
    study2_cells = CELLS.replace("name: test-campaign\n",
                                 "name: test-campaign\nheader: {w_max: 0.47}\n")  # fmt: skip
    path = campaign(study2, study2_cells)
    name, runs = load_campaign(path)
    cell, root = [r for r in runs if r.cell == "adamw-5e-5"], study2 / "runs"
    broken = Executor(root, {cell[0].label: ["failed"] * 6,
                             cell[1].label: ["failed"] * 3 + ["completed"]})  # fmt: skip
    run_campaign(name, cell, root, execute=broken, header=campaign_header(path))
    result = run_campaign(name, cell, root, execute=broken, header=campaign_header(path),
                          recovery="disk replaced")  # fmt: skip
    assert result["exhausted"] == [cell[0].label] and result["done"] == [cell[1].label]
    assert len(broken.calls) == 10
    assert [r["cell"] for r in rounds_of(root, name)] == ["adamw-5e-5"]


def test_a_study3_recovery_over_a_run_not_used_up_is_refused(tmp_path, monkeypatch) -> None:
    """A study-3 campaign: the label's attempt under an earlier config hash (an engineering
    cycle's fix changed a hashed key, and superseded it); under the new hash the run is new, not
    used up. ``--recovery`` over it is refused, exit 1, naming the command without
    ``--recovery``; nothing is written and nothing runs. The same call without ``--recovery``
    runs its base attempt. A done run is refused a round too. Study 2's call over a run with
    attempts left still runs them, writing no round."""
    import importlib.util

    path = campaign(tmp_path, STUDY3_CELLS)
    name, runs = load_campaign(path)
    run, root, header = runs[1], tmp_path / "runs", campaign_header(path)
    earlier = SimpleNamespace(label=run.label, experiment=run.experiment, config_hash="0" * 12,
                              command=run.command)  # fmt: skip
    log_start(root, name, earlier, "x-a1", only=run.label, max_attempts=1)
    supersede(root, name, earlier, "x-a1", reason="the cycle's fix", record="c1/supersede-1.json")
    log = root / "launch" / f"{name}.jsonl"
    before, execute = log.read_bytes(), Executor(root, {run.label: ["completed"]})
    only = {"only": run.label, "max_attempts": 1}
    with pytest.raises(CampaignError, match="not used up") as refusal:
        run_campaign(name, runs, root, execute=execute, header=header, recovery=R1,
                     incident=INCIDENT, **only)  # fmt: skip
    form = f"scripts/launch.py CAMPAIGN --only {run.label} --max-attempts 1"
    assert form in str(refusal.value) and log.read_bytes() == before and execute.calls == []
    where = Path(__file__).resolve().parents[1] / "scripts" / "launch.py"
    spec = importlib.util.spec_from_file_location("launch_script", where)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setenv("RUNS_DIR", str(root))
    incident = ["--incident-time", INCIDENT["incident_time"], "--cause", INCIDENT["cause"],
                "--evidence", INCIDENT["evidence"]]  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        script.main([str(path), "--only", run.label, "--max-attempts", "1", "--recovery", R1,
                     *incident])  # fmt: skip
    assert isinstance(stop.value.code, str) and "not used up" in stop.value.code  # exit 1
    assert log.read_bytes() == before and execute.calls == []
    result = run_campaign(name, runs, root, execute=execute, header=header, **only)
    assert result["done"] == [run.label] and run_ids(execute)[0].endswith("-a2")
    with pytest.raises(CampaignError, match="done"):
        run_campaign(name, runs, root, execute=execute, header=header, recovery=R1,
                     incident=INCIDENT, **only)  # fmt: skip
    assert rounds_of(root, name) == [] and len(execute.calls) == 1
    study2 = tmp_path / "study2"
    study2.mkdir()
    study2_cells = CELLS.replace("name: test-campaign\n",
                                 "name: test-campaign\nheader: {w_max: 0.47}\n")  # fmt: skip
    path = campaign(study2, study2_cells)
    name, runs = load_campaign(path)
    run, root = runs[1], study2 / "runs"
    execute = Executor(root, {run.label: ["completed"]})
    result = run_campaign(name, runs, root, execute=execute, header=campaign_header(path),
                          recovery=R1, incident=INCIDENT, **only)  # fmt: skip
    assert result["done"] == [run.label] and rounds_of(root, name) == []


def test_a_study3_dry_run_reports_the_recovery_refusal_of_the_real_call(tmp_path, monkeypatch):
    """A study-3 ``--dry-run --recovery`` over a run that is not used up (a new run, and a done
    one) fails exactly as the real call does: the same message, exit 1 through the script, and
    nothing runs or is written."""
    import importlib.util

    path = campaign(tmp_path, STUDY3_CELLS)
    name, runs = load_campaign(path)
    run, root, header = runs[1], tmp_path / "runs", campaign_header(path)
    only = {"only": run.label, "max_attempts": 1, "recovery": R1, "incident": INCIDENT}
    where = Path(__file__).resolve().parents[1] / "scripts" / "launch.py"
    spec = importlib.util.spec_from_file_location("launch_script", where)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setenv("RUNS_DIR", str(root))
    argv = [str(path), "--only", run.label, "--max-attempts", "1", "--recovery", R1,
            "--incident-time", INCIDENT["incident_time"], "--cause", INCIDENT["cause"],
            "--evidence", INCIDENT["evidence"]]  # fmt: skip
    for state in ("new", "done"):
        if state == "done":
            run_campaign(name, runs, root, execute=Executor(root, {}), header=header,
                         only=run.label, max_attempts=1)  # fmt: skip
        execute = Executor(root, {})
        messages = []
        for dry_run in (True, False):
            with pytest.raises(CampaignError) as refusal:
                run_campaign(name, runs, root, execute=execute, header=header, dry_run=dry_run,
                             **only)  # fmt: skip
            messages.append(str(refusal.value))
            with pytest.raises(SystemExit) as stop:
                script.main(argv + (["--dry-run"] if dry_run else []))
            assert stop.value.code == str(refusal.value)  # exit 1, the same message
        assert messages[0] == messages[1] and execute.calls == []
        assert [r for r in launch_records(root) if r.get("event") == "recovery"] == []
    assert "done" in messages[0] and "diverged" not in messages[0]  # the state the run has
    assert "no recovery round is written" in messages[0] and "nothing is written" not in messages[0]
