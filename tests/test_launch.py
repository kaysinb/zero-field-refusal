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
    result = run_campaign(name, runs, root, execute=executor, recovery="the Pod stopped", **only)
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
                 recovery="the Pod's deadline stopped attempt 1")  # fmt: skip
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
