"""scripts/study5_campaigns.py: study 5's two Pod campaigns as plans of steps, built without running
anything, and the helpers its steps call (docs/study5-preregistration.md, "Seeds and order", "The
confirmatory test"; docs/operations.md, "Study 5 campaigns")."""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.launch import load_campaign
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import load_run_config


def script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


camp = script_module(REPO_ROOT / "scripts" / "study5_campaigns.py", "study5_campaigns")

COMMIT = "c" * 40
GPU = "e" * 64
# Study 3's F runs on seed 10403 (docs/study3-analysis-freeze-c1.json): their config hashes
STUDY3_F_HASHES = {"d1ce7a92e45b", "3b4601aa846f"}


@pytest.fixture
def where(tmp_path):
    return {
        "out": tmp_path / "campaign",
        "runs": tmp_path / "runs",
        "data": tmp_path / "data",
        "checkpoints": Path("/root/ckpt/study5"),
    }


def pod1(where):
    return camp.pod1_plan(archive_id="t1", commit=COMMIT, gpu_code=GPU, **where)


def pod2(where, n=40):
    return camp.pod2_plan(archive_id="t2", commit=COMMIT, gpu_code=GPU, n=n, **where)


def steps(plan, block=None):
    return [s for b in plan.blocks if block in (None, b.name) for s in b.steps]


def campaign_runs(plan, tmp_path, name):
    path = tmp_path / f"{name}.yaml"
    path.write_text(plan.campaigns[name])
    return load_campaign(path)[1]


# ------------------------------------------------------------------ seeds and configs


def test_the_retrospective_seeds_are_study3s_a_f_cohort_from_its_freeze():
    seeds, runs = camp.study3_cohort()
    freeze = json.loads((REPO_ROOT / "docs" / "study3-analysis-freeze-c1.json").read_text())
    assert seeds == [b["seed"] for b in freeze["comparisons"]["A-F"]["blocks"]]
    assert len(seeds) == 60 and seeds[0] == 10400 and 10428 not in seeds and 10403 in seeds
    assert sorted(runs) == sorted(
        {
            freeze_run
            for b in freeze["comparisons"]["A-F"]["blocks"]
            if b["seed"] == 10403
            for freeze_run in b["arms"]["F"].values()
            if isinstance(freeze_run, str)
        }
    )
    assert all("adam_frozen" in r and "s10403" in r for r in runs)


def test_fresh_seeds_and_source_seeds():
    assert camp.fresh_seeds(40) == list(range(50000, 50040))
    assert camp.fresh_seeds(24) == list(range(50000, 50024))
    for n in (0, 23, 25, 39, 41, 60):
        with pytest.raises(ValueError):
            camp.fresh_seeds(n)
    assert [camp.source_seed(s) for s in range(50000, 50006)] == [
        50001,
        50002,
        50003,
        50004,
        50005,
        50000,
    ]


def test_study5_equals_study3s_f_configuration_but_for_its_identity():
    def plain(cfg, *drop):
        data = OmegaConf.to_container(cfg, resolve=True)
        for key in drop:
            node = data
            *parents, leaf = key.split(".")
            for part in parents:
                node = node.get(part, {})
            node.pop(leaf, None)
        return data

    for rate in (camp.ETA_L, camp.ETA_U):
        study3 = load_config(
            "experiments/study3",
            [
                "study3.role=confirmation",
                "optimizer=adam_frozen",
                f"optimizer.lr={rate!r}",
                "seed=50000",
            ],
        )
        study5 = load_config(
            "experiments/study5", ["study5.role=fresh", f"optimizer.lr={rate!r}", "seed=50000"]
        )
        drop = ("experiment", "study3", "study5")
        assert plain(study5, *drop) == plain(study3, *drop)
    assert study5.study5.role == "fresh" and study5.optimizer.name == "adam_frozen"
    assert study5.train.batch_size == 20 and study5.train.steps == 25
    assert study5.eval.controls.line_scan.get("points") is None  # the scan is a run of its own
    assert not (REPO_ROOT / "configs" / "experiments" / "study5_scan.yaml").exists()
    assert camp.ETA_L == 1.3e-05 and camp.ETA_U == 1.83e-05


def test_every_study5_config_passes_the_protocol():
    cases = [
        (
            "experiments/study5",
            [
                "study5.role=fresh",
                "optimizer.lr=1.3e-05",
                "seed=50000",
                "tracking.endpoint_checkpoint=/root/ckpt/study5",
            ],
        ),
        ("experiments/study5", ["study5.role=fresh", "optimizer.lr=1.83e-05", "seed=50000"]),
        ("experiments/study5_control_order", ["optimizer.lr=1.3e-05", "seed=50003"]),
        (
            "experiments/study5_control_v0",
            ["optimizer.lr=1.3e-05", "seed=50003", "optimizer.preconditioner.source_seed=50004"],
        ),
        ("experiments/study5_theta0", ["seed=50000", "run_type=theta0"]),
    ]
    for name, overrides in cases:
        cfg = load_run_config(name, overrides)
        assert cfg.experiment == "study5" and cfg.optimizer.name == "adam_frozen", name
        assert config_hash(cfg)
    order = load_run_config(*cases[2])
    assert order.train.order_seed == 50003 and order.study5.role == "control"
    with pytest.raises(Exception, match="source_seed|Missing"):
        config_hash(load_config("experiments/study5_control_v0", ["optimizer.lr=1.3e-05"]))
    with pytest.raises(Exception, match="role"):
        load_run_config("experiments/study5", ["optimizer.lr=1.3e-05"])


# ------------------------------------------------------------------ pod 1


def test_pod1_blocks_in_order_each_set_deadline_first_and_archived_last(where):
    plan = pod1(where)
    assert [b.name for b in plan.blocks] == [
        "engineering",
        "theta0",
        "surrogate",
        "retrospective",
        "close",
    ]
    for block in plan.blocks[:-1]:
        assert block.steps[0].kind == "deadline" and block.steps[0].argv[1] == "deadline"
        assert block.steps[-1].kind == "receipt"
        assert block.steps[-2].kind == "archive" and block.steps[-2].argv[1] == "snapshot"
    close = plan.blocks[-1]
    assert [s.kind for s in close.steps] == ["archive", "deadline"]
    assert all(s.always for s in close.steps)


def test_pod1_engineering_reruns_study3s_f_runs_and_stops_on_any_failure(where, tmp_path):
    plan = pod1(where)
    block = steps(plan, "engineering")
    assert all(s.on_fail == "stop" for s in block)
    kinds = [s.argv[1] if s.argv[0].endswith("study5_campaigns.py") else s.argv[0] for s in block]
    assert kinds == [
        "deadline",
        "fetch-streams",
        "fetch-run",
        "fetch-run",
        "eng-launch",
        "with-runs",
        "archive",
        "snapshot",
        "receipt",
    ]
    fetch = block[1]
    assert fetch.argv[fetch.argv.index("--seeds") + 1 :][:1] == ("10403",)
    launch_step = block[4]
    assert launch_step.kind == "launch" and not launch_step.kept_if_runs
    assert launch_step.argv[launch_step.argv.index("--gpu-code") + 1] == GPU
    campaign_file = launch_step.argv[launch_step.argv.index("--campaign-file") + 1]
    runs = campaign_runs(plan, tmp_path, Path(campaign_file).stem)
    assert {run.config_hash for run in runs} == STUDY3_F_HASHES
    assert {run.seed for run in runs} == {10403} and all(r.experiment == "study3" for r in runs)
    check = block[5]
    assert "scripts/study5_engineering_check.py" in check.argv
    fetched = [a for a in check.argv if "/fetched/study3/" in a]
    _, freeze_runs = camp.study3_cohort()
    assert sorted(Path(a).name for a in fetched) == sorted(freeze_runs)


def test_pod1_theta0_surrogate_and_retrospective(where, tmp_path):
    plan = pod1(where)
    theta0 = [s for s in steps(plan, "theta0") if s.kind == "launch"]
    assert len(theta0) == 1 and theta0[0].on_fail == "stop"  # the analysis needs θ₀
    (run,) = campaign_runs(plan, tmp_path, Path(theta0[0].argv[1]).stem)
    assert run.entry == "theta0" and run.config == "experiments/study5_theta0"
    surrogate = [s for s in steps(plan, "surrogate") if s.kind == "run"]
    assert [s.argv[0] for s in surrogate] == ["scripts/study5_surrogate_checks.py"]
    assert surrogate[0].on_fail == "stop"
    assert surrogate[0].argv[-2:] == ("--surrogate", "configs/study5/surrogate-2.yaml")
    retro = steps(plan, "retrospective")
    seeds, _ = camp.study3_cohort()
    fetch = retro[1]
    assert fetch.argv[1] == "fetch-streams" and fetch.on_fail == "stop"
    assert [int(x) for x in fetch.argv[fetch.argv.index("--seeds") + 1 :]] == seeds
    predicts = [s for s in retro if s.argv[0] == "scripts/study5_predict.py"]
    features = [s for s in retro if s.argv[0] == "scripts/study5_stream_features.py"]
    assert [int(s.argv[s.argv.index("--out") - 1].split("=")[1]) for s in predicts] == seeds
    assert len(features) == 60
    for step in predicts + features:
        assert step.argv[1:6] == (
            "experiments/study3",
            "optimizer=adam_frozen",
            "train.batch_size=20",
            "optimizer.lr=1.3e-05",
            "study3.role=confirmation",
        )
        assert step.on_fail == "record"


def surrogate_steps(plan):
    """The steps that read the surrogate: the checks, the predictor, the endpoint's r."""
    names = ("study5_surrogate_checks.py", "study5_predict.py", "study5_surrogate_endpoint.py")
    return [s for s in steps(plan) if any(n in a for a in s.argv for n in names)]


def test_both_plans_name_one_surrogate_spec_and_pass_it_on(where, tmp_path, monkeypatch):
    """One surrogate spec per plan (--surrogate, a file of the checkout), its sha256 in the plan,
    given to the checks, every predictor and every endpoint's r; a second set is another file."""
    from zero_field_refusal.surrogate import load_spec

    spec_path = "configs/study5/surrogate-2.yaml"
    default = load_spec(REPO_ROOT / spec_path).sha256
    for plan, count in ((pod1(where), 1 + 60), (pod2(where, 24), 2 * 24)):
        found = surrogate_steps(plan)
        assert len(found) == count
        for step in found:
            assert step.argv[step.argv.index("--surrogate") + 1] == spec_path
        assert (plan.surrogate, plan.surrogate_sha256) == (spec_path, default)
        assert plan.to_json()["surrogate_sha256"] == default
    # a second set: a new file in the checkout, named by --surrogate of both plans
    repo = tmp_path / "repo"
    second = repo / "configs" / "study5" / "surrogate-3.yaml"
    second.parent.mkdir(parents=True)
    text = (REPO_ROOT / "configs" / "study5" / "surrogate-2.yaml").read_text()
    second.write_text(text.replace("version: 1", "version: 2"))
    monkeypatch.setattr(camp, "REPO", repo)
    plan = camp.pod2_plan(archive_id="t2", commit=COMMIT, gpu_code=GPU, n=24,
                          surrogate="configs/study5/surrogate-3.yaml", **where)  # fmt: skip
    assert plan.surrogate_sha256 == load_spec(second).sha256 != default
    for step in surrogate_steps(plan):
        assert step.argv[step.argv.index("--surrogate") + 1] == "configs/study5/surrogate-3.yaml"
    for bad in ("/abs/surrogate.yaml", "../surrogate.yaml", "configs/study5/none.yaml"):
        with pytest.raises(ValueError, match="--surrogate"):
            camp.pod2_plan(archive_id="t2", commit=COMMIT, gpu_code=GPU, n=24, surrogate=bad,
                           **where)  # fmt: skip


def registered(
    tmp_path, *, status="registered", name_fixation=True, name_retro=True, retro_fixation=None
):
    """A checkout with a registration, a fixation record and a retrospective record."""
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True, exist_ok=True)
    fixation = repo / "docs" / "study5-fixation.json"
    fixation.write_text('{"kind": "study5-fixation"}\n')
    fixation_sha = hashlib.sha256(fixation.read_bytes()).hexdigest()
    retro = tmp_path / "retrospective.json"
    fixed = {"sha256": retro_fixation or fixation_sha}
    retro.write_text(json.dumps({"kind": "study5-retrospective", "fixation": fixed}))
    retro_sha = hashlib.sha256(retro.read_bytes()).hexdigest()
    lines = [f"Status: `{status}`. It binds from this commit.", "", "## Question"]
    if name_fixation:
        lines.append(f"The fixation record's sha256: {fixation_sha}.")
    if name_retro:
        lines.append(f"The retrospective record's sha256: {retro_sha}.")
    (repo / "docs" / "study5-preregistration.md").write_text("\n".join(lines) + "\n")
    return repo, retro


@pytest.mark.parametrize(
    "case, problem",
    [
        ({}, None),
        ({"status": "draft"}, "not registered"),
        ({"name_fixation": False}, "sha256 of docs/study5-fixation.json"),
        ({"name_retro": False}, "retrospective record's sha256"),
        ({"retro_fixation": "0" * 64}, "not of the committed fixation record"),
    ],
)
def test_pod2_starts_only_after_the_registration(tmp_path, case, problem):
    repo, retro = registered(tmp_path, **case)
    found = camp.registration_problems(retro, repo=repo)
    assert found == [] if problem is None else any(problem in f for f in found), found
    assert any("--retrospective" in f for f in camp.registration_problems(None, repo=repo))
    assert camp.registration_problems(retro, repo=REPO_ROOT)  # this checkout: not registered yet


def test_pod2_starts_only_on_pod1s_gates(where, tmp_path):
    from zero_field_refusal.surrogate import load_spec

    plan = pod2(where, 24)
    plan.gpu_code = HEAD["sha256"]
    spec = load_spec(REPO_ROOT / "configs" / "study5" / "surrogate-2.yaml").sha256
    eng, checks = tmp_path / "eng.json", tmp_path / "checks.json"

    def checks_record(**extra):
        return {"kind": "study5-surrogate-checks", "passes": True, "surrogate_sha256": spec,
                "git": {"commit": HEAD["commit"], "dirty": False}, **extra}  # fmt: skip

    eng.write_text(json.dumps(engineering_record()))
    checks.write_text(json.dumps(checks_record()))
    assert camp.gate_problems(plan, eng, checks) == []
    for record, problem in (
        (engineering_record(equal=False), "difference"),
        (engineering_record(gpu="f" * 64), "GPU code"),
        (engineering_record(dirty=True), "clean checkout"),
    ):
        eng.write_text(json.dumps(record))
        assert any(problem in f for f in camp.gate_problems(plan, eng, checks)), problem
    eng.write_text(json.dumps(engineering_record()))
    for record, problem in (
        (checks_record(passes=False), "did not pass"),
        (checks_record(surrogate_sha256="0" * 64), "not of configs/study5/surrogate-2.yaml"),
        (checks_record(git={"commit": HEAD["commit"], "dirty": True}), "not clean"),
    ):
        checks.write_text(json.dumps(record))
        assert any(problem in f for f in camp.gate_problems(plan, eng, checks)), problem
    found = camp.gate_problems(plan, None, None)
    assert any("--engineering-check" in f for f in found)
    assert any("--surrogate-checks" in f for f in found)


def test_the_preflight_of_pod2_alone_reads_the_registration_and_the_gates(where, monkeypatch):
    seen = []
    monkeypatch.setattr(camp, "registration_problems", lambda retro: seen.append(retro) or ["r"])
    monkeypatch.setattr(camp, "gate_problems",
                        lambda plan, eng, checks: seen.append((eng, checks)) or ["g"])  # fmt: skip
    monkeypatch.setattr(camp, "seeds_problems", lambda plan: seen.append(plan.n) or ["s"])
    args = camp.parser_of().parse_args(
        ["pod2", "--out", "o", "--archive-id", "t2", "--engineering-check", "e.json",
         "--surrogate-checks", "c.json", "--retrospective", "x.json"])  # fmt: skip
    found = camp.preflight(pod2(where, 24), args)
    assert found[-3:] == ["r", "s", "g"]
    assert seen == [Path("x.json"), 24, (Path("e.json"), Path("c.json"))]
    seen.clear()
    assert "r" not in camp.preflight(pod1(where), None) and seen == []


# ------------------------------------------------------------------ pod 2


@pytest.mark.parametrize("n", [40, 24])
def test_pod2_seeds_and_the_predictions_before_any_f_run(where, n):
    plan = pod2(where, n)
    assert [b.name for b in plan.blocks] == ["streams", "predictions", "fresh", "controls", "close"]
    seeds = list(range(50000, 50000 + n))
    gen = [s for s in steps(plan, "streams") if s.argv[0] == "scripts/generate_teacher.py"]
    assert [int(s.argv[s.argv.index("--seeds") + 1]) for s in gen] == [
        x for x in seeds for _ in (0, 1)
    ]
    every = steps(plan)
    publish = next(i for i, s in enumerate(every) if s.argv[1:2] == ("publish",))
    record = next(i for i, s in enumerate(every) if s.argv[1:2] == ("predictions",))
    first_launch = next(i for i, s in enumerate(every) if s.kind == "launch")
    assert record < publish < first_launch
    assert every[record].on_fail == every[publish].on_fail == "stop"
    predictions = every[record]
    assert [int(x) for x in predictions.argv[predictions.argv.index("--seeds") + 1 :]] == seeds
    fresh = steps(plan, "fresh")
    check = fresh[1]
    assert check.argv[1] == "check-published" and check.on_fail == "stop"
    assert [int(x) for x in check.argv[check.argv.index("--seeds") + 1 :]] == seeds
    assert "--commit" not in check.argv  # the record's commit is information only
    assert check.argv[check.argv.index("--gpu-code") + 1] == GPU
    assert every[record].gate == "predictions"
    launches = [s for s in fresh if s.kind == "launch"]
    assert len(launches) == 2 * n and all(s.on_fail == "record" for s in launches)


def test_pod2_rejects_another_n(where):
    for n in (10, 30, 50):
        with pytest.raises(ValueError):
            pod2(where, n)


def test_pod2_fresh_runs_per_seed(where, tmp_path):
    plan = pod2(where, 24)
    fresh = steps(plan, "fresh")
    for seed in (50000, 50023):
        group = [s for s in fresh if s.group == f"s{seed}"]
        assert [s.argv[1] if "study5_campaigns" in s.argv[0] else s.argv[0] for s in group] == [
            "deadline",
            "check-checkpoints",
            "scripts/launch.py",
            "checkpoint",
            "with-runs",
            "with-runs",
            "checkpoint",
            "scripts/launch.py",
            "archive",
        ]
        assert [s.name.rsplit("/", 1)[-1] for s in group[2:7]] == [
            "launch-fresh-lower", "checkpoint-archive", "surrogate-endpoint", "scan",
            "checkpoint"]  # fmt: skip
        assert group[1].on_fail == "stop" and group[8].on_fail == "stop"
        assert group[1].argv[-2:] == ("--max", "1")
        # η_l alone; its checkpoint archived and verified at once (kept), r at its endpoint and
        # its scan run, each restoring the checkpoint from the archive when it is gone, the
        # local file removed, then η_u: the archive covers both runs, the scan and two records
        (low,) = campaign_runs(plan, tmp_path, Path(group[2].argv[1]).stem)
        (high,) = campaign_runs(plan, tmp_path, Path(group[7].argv[1]).stem)
        assert (low.config, low.seed) == ("experiments/study5", seed)
        assert (high.config, high.seed) == ("experiments/study5", seed)
        keep, remove = group[3], group[6]
        assert keep.argv[-1] == "--keep" and "--keep" not in remove.argv
        assert keep.argv[:-1] == remove.argv  # the same checkpoint, archive and inventory
        assert remove.argv[remove.argv.index("--label") + 1] == low.label
        inventory = remove.argv[remove.argv.index("--inventory") + 1]
        for step in group[4:6]:
            argv = step.argv
            assert argv[argv.index("--restore-label") + 1] == low.label
            assert argv[argv.index("--restore-dir") + 1] == str(where["checkpoints"])
            assert argv[argv.index("--restore-inventory") + 1] == inventory
            assert argv[argv.index("--restore-remote") + 1] == "study5/t2/checkpoints"
            assert argv.index("--restore-label") < argv.index("--")
        archived = [group[8].argv[i + 1] for i, a in enumerate(group[8].argv) if a == "--campaign"]
        assert archived == [Path(group[2].argv[1]).stem, Path(group[7].argv[1]).stem]
        assert group[2].kept_if_runs == (f"{Path(group[2].argv[1]).stem}\t{low.label}",)
        lower, upper = (load_run_config(r.config, list(r.overrides)) for r in (low, high))
        assert lower.optimizer.lr == 1.3e-05 and upper.optimizer.lr == 1.83e-05
        assert lower.tracking.endpoint_checkpoint == "/root/ckpt/study5"
        assert upper.tracking.endpoint_checkpoint is None
        assert lower.study5.role == upper.study5.role == "fresh"
        assert config_hash(lower) != config_hash(upper)  # the same config but for the rate


@pytest.mark.parametrize("n", [40, 24])
def test_pod2_scores_the_surrogate_at_every_endpoint_before_its_checkpoint_goes(where, n):
    """In every fresh group: r at the η_l run's endpoint (scripts/study5_surrogate_endpoint.py on
    that run and its checkpoint), after the run and before the checkpoint step archives and
    removes the file; a failure is recorded, not a stop; its record is archived with the runs."""
    plan = pod2(where, n)
    fresh = steps(plan, "fresh")
    for seed in camp.fresh_seeds(n):
        group = [s for s in fresh if s.group == f"s{seed}"]
        names = [s.name for s in group]
        launch = names.index(f"fresh/s{seed}/launch-fresh-lower")
        endpoint = names.index(f"fresh/s{seed}/surrogate-endpoint")
        checkpoint = names.index(f"fresh/s{seed}/checkpoint")
        assert launch < endpoint < checkpoint
        step = group[endpoint]
        lower = Path(group[launch].argv[1]).stem
        record = where["runs"] / "study5_records" / "t2" / "surrogate-endpoint" / f"s{seed}.json"
        assert step.argv[:5] == (camp.SELF, "with-runs", "--runs", str(where["runs"]),
                                 "--campaign") and step.argv[5] == lower  # fmt: skip
        command = step.argv[step.argv.index("--") + 1 :]
        assert command[0] == "scripts/study5_surrogate_endpoint.py"
        assert tuple(command[1:5]) == camp.fresh_args(seed)
        label = group[checkpoint].argv[group[checkpoint].argv.index("--label") + 1]
        assert command[command.index("--f-run") + 1] == f"{{run:{label}}}"
        assert command[command.index("--checkpoints") + 1] == str(where["checkpoints"])
        assert command[command.index("--out") + 1] == str(record)
        assert step.on_fail == "record" and step.outputs == (str(record),)
        assert step.minutes == camp.MINUTES["surrogate-endpoint"] == 1.0
        assert step.timeout == camp.timeout_of(1.0)
        archive = group[-1]
        paths = [archive.argv[i + 1] for i, a in enumerate(archive.argv) if a == "--path"]
        assert paths[0] == record.relative_to(where["runs"]).as_posix()


@pytest.mark.parametrize("n", [40, 24])
def test_pod2_scans_every_endpoint_in_a_run_of_its_own_before_its_checkpoint_goes(where, n):
    """In every fresh group: the line scan of the η_l run's endpoint (scripts/study5_scan.py on
    that run and its checkpoint), after its r and before the checkpoint step archives and removes
    the file; a partial scan run is moved aside before a rerun; a failure is recorded, not a
    stop; its record and its run directory are archived with the runs."""
    plan = pod2(where, n)
    fresh = steps(plan, "fresh")
    for seed in camp.fresh_seeds(n):
        group = [s for s in fresh if s.group == f"s{seed}"]
        names = [s.name for s in group]
        endpoint = names.index(f"fresh/s{seed}/surrogate-endpoint")
        scan = names.index(f"fresh/s{seed}/scan")
        checkpoint = names.index(f"fresh/s{seed}/checkpoint")
        assert endpoint < scan < checkpoint
        step = group[scan]
        lower = Path(group[names.index(f"fresh/s{seed}/launch-fresh-lower")].argv[1]).stem
        record = where["runs"] / "study5_records" / "t2" / "scan" / f"s{seed}.json"
        run_id = f"s5-t2-scan-s{seed}"
        assert step.argv[:6] == (camp.SELF, "with-runs", "--runs", str(where["runs"]),
                                 "--campaign", lower)  # fmt: skip
        assert step.argv[step.argv.index("--aside") + 1] == f"study5_scan/{run_id}"
        assert step.argv[step.argv.index("--unless") + 1] == str(record)
        command = step.argv[step.argv.index("--") + 1 :]
        assert command[0] == "scripts/study5_scan.py"
        assert tuple(command[1:5]) == camp.fresh_args(seed)
        label = group[checkpoint].argv[group[checkpoint].argv.index("--label") + 1]
        assert command[command.index("--f-run") + 1] == f"{{run:{label}}}"
        assert command[command.index("--checkpoints") + 1] == str(where["checkpoints"])
        assert command[command.index("--out") + 1] == str(record)
        assert command[command.index("--run-id") + 1] == run_id
        assert step.on_fail == "record" and step.outputs == (str(record),)
        archive = group[-1]
        paths = [archive.argv[i + 1] for i, a in enumerate(archive.argv) if a == "--path"]
        assert paths[1:] == [record.relative_to(where["runs"]).as_posix()]
        # the scan run's directory: written once when its record exists, else the attempt's
        run_dirs = [archive.argv[i + 1] for i, a in enumerate(archive.argv) if a == "--run-dir"]
        assert run_dirs == [f"study5_scan/{run_id}={record.relative_to(where['runs'])}"]


def test_pod2_controls_and_ladder_on_the_first_six_seeds(where, tmp_path):
    plan = pod2(where)
    controls = steps(plan, "controls")
    groups = sorted({s.group for s in controls if s.group})
    assert groups == [f"s{s}" for s in range(50000, 50006)]
    for seed in range(50000, 50006):
        group = [s for s in controls if s.group == f"s{seed}"]
        assert group[0].kind == "deadline"
        launch_step = group[1]
        assert launch_step.kind == "launch" and launch_step.on_fail == "record"
        runs = campaign_runs(plan, tmp_path, Path(launch_step.argv[1]).stem)
        order, v0 = (load_run_config(r.config, list(r.overrides)) for r in runs)
        assert order.train.order_seed == seed and order.optimizer.lr == 1.3e-05
        assert v0.optimizer.preconditioner.source_seed == camp.source_seed(seed)
        ladders = [s for s in group if "scripts/study5_ladder.py" in s.argv]
        rungs = [s.argv[s.argv.index("--rung") + 1] for s in ladders]
        assert rungs == ["gaussian_d", "linear_no_d", "linear"]
        assert all(s.on_fail == "record" for s in ladders)
        assert group[-1].kind == "archive" and group[-1].on_fail == "stop"
    assert not any("linear_shuffled_d" in " ".join(s.argv) for s in steps(plan))


def pod3(where):
    return camp.pod3_plan(archive_id="t3", commit=COMMIT, gpu_code=GPU, **where)


def test_pod3_extension_1(where, tmp_path):
    """Extension 1 (docs/study5-preregistration.md): the engineering rerun of pod 2's order
    control of 50000 first, every failure there a stop; the fetches; the two unbracketed rungs on
    50000-50005 with --extensions 8 on pod 2's F runs; two new orders at both rates on
    50006-50015; no checkpoint, no labeling."""
    plan = pod3(where)
    assert [b.name for b in plan.blocks] == ["engineering", "fetch", "ladder", "orders", "close"]
    engineering = steps(plan, "engineering")
    assert all(s.on_fail == "stop" for s in engineering)
    launch_step = next(s for s in engineering if s.kind == "launch")
    (rerun,) = campaign_runs(plan, tmp_path, Path(launch_step.argv[launch_step.argv.index(
        "--campaign-file") + 1]).stem)  # fmt: skip
    cfg = load_run_config(rerun.config, list(rerun.overrides))
    # pod 2's run: its id carries its config hash
    assert camp.POD2_ORDER_RUN.split("-")[-2] == config_hash(cfg)[:8]
    assert cfg.train.order_seed == 50000 and cfg.optimizer.lr == 1.3e-05
    check = next(s for s in engineering if s.gate == "engineering")
    assert "scripts/study5_engineering_check.py" in check.argv
    assert str(where["out"] / "fetched" / "study5" / camp.POD2_ORDER_RUN) in check.argv
    fetches = [s for s in steps(plan) if "fetch-run" in s.argv]
    assert all(s.argv[s.argv.index("--prefix") + 1] == "study5/pod2-20261109/runs/study5"
               for s in fetches)  # fmt: skip
    assert {s.argv[s.argv.index("--run-id") + 1] for s in fetches} == {
        camp.POD2_ORDER_RUN, *camp.POD2_F_RUNS.values()}  # fmt: skip
    streams = [int(s.argv[s.argv.index("--seeds") + 1]) for s in steps(plan)
               if "fetch-streams" in s.argv]  # fmt: skip
    assert sorted(streams) == list(range(50000, 50016))
    ladders = [s for s in steps(plan, "ladder") if "scripts/study5_ladder.py" in s.argv]
    assert len(ladders) == 12 and all(s.on_fail == "record" for s in ladders)
    for s in ladders:
        assert s.argv[s.argv.index("--rung") + 1] in ("gaussian_d", "linear_no_d")
        assert s.argv[s.argv.index("--extensions") + 1] == "8"
        seed = int(s.argv[s.argv.index("--f-run") + 1].split("-s")[-1].split("-")[0])
        assert s.argv[s.argv.index("--f-run") + 1].endswith(camp.POD2_F_RUNS[seed])
        assert f"seed={seed}" in s.argv and "optimizer.lr=1.3e-05" in s.argv
    orders = [s for s in steps(plan, "orders") if s.kind == "launch"]
    assert [s.group for s in orders] == [f"s{s}" for s in range(50006, 50016)]
    hashes = set()
    for s in orders:
        runs = campaign_runs(plan, tmp_path, Path(s.argv[1]).stem)
        seed = int(s.group[1:])
        found = []
        for r in runs:
            cfg = load_run_config(r.config, list(r.overrides))
            assert cfg.study5.role == "control" and r.seed == seed
            found.append((cfg.train.order_seed - seed, cfg.optimizer.lr))
            hashes.add(config_hash(cfg))
        assert sorted(found) == [(1000, 1.3e-05), (1000, 1.83e-05), (2000, 1.3e-05),
                                 (2000, 1.83e-05)]  # fmt: skip
    assert len(hashes) == 40
    text = " ".join(" ".join(s.argv) for s in steps(plan))
    assert "linear_shuffled_d" not in text and "scripts/label_runs.py" not in text
    commands = {s.argv[1] for s in steps(plan) if s.argv[0] == "scripts/study5_campaigns.py"}
    assert not commands & {"checkpoint", "check-checkpoints", "export-checkpoints"}
    assert not any("endpoint_checkpoint" in text for text in plan.campaigns.values())


def test_pod3_needs_extension_1s_gpu_code_and_its_registration(where, monkeypatch):
    plan = pod3(where)
    problems = camp.preflight(plan)
    assert any("study5-extension-1-code-identity.json" in p for p in problems)
    recorded = json.loads((REPO_ROOT / camp.EXTENSION_IDENTITY).read_text())
    plan.gpu_code = recorded["gpu_code_sha256"]
    assert not any("GPU code" in p for p in camp.preflight(plan))
    assert camp.EXTENSION_HEADING in (REPO_ROOT / camp.REGISTRATION).read_text()
    assert not any("extension 1" in p for p in camp.preflight(plan))
    monkeypatch.setattr(camp, "EXTENSION_HEADING", "### Extension 9")
    assert any("does not hold extension 1" in p for p in camp.preflight(plan))


def _meta(runs, rel, archive_id):
    path = runs / rel / "meta.json"
    path.parent.mkdir(parents=True)
    header = {"study": 5, "pod": archive_id.split("-")[0], "archive_id": archive_id}
    path.write_text(json.dumps({"campaign": {"name": "x", "header": header}}))


def _pod3_root(tmp_path, name):
    """A runs root holding pod 3's own entries only (as at a restart) and its plan."""
    where = {"out": tmp_path / name / "runs" / "campaigns" / "study5-pod3",
             "runs": tmp_path / name / "runs", "data": tmp_path / name / "data",
             "checkpoints": Path("/root/ckpt/study5")}  # fmt: skip
    plan = camp.pod3_plan(archive_id="pod3-20261111", commit=COMMIT, gpu_code=GPU, **where)
    runs = where["runs"]
    _meta(runs, "study5/20261111T000000Z-s5-order1-1.3e-05-s50006-s50006-aaaaaaaa-a1",
          "pod3-20261111")  # fmt: skip
    (runs / "study5_records" / "pod3-20261111").mkdir(parents=True)
    (runs / "launch").mkdir()
    (runs / "launch" / "study5-pod3-20261111-orders-s50006.jsonl").write_text("")
    (runs / "study5_ladder" / "s5-pod3-20261111-ladder-linear_no_d-s50000").mkdir(parents=True)
    plan.out.mkdir(parents=True)
    return plan, runs


@pytest.mark.parametrize("entry", [
    "pod2-order-run", "pod2-records", "pod2-journal", "run-without-meta", "pod2-ladder-run",
    "pod2-scan-run", "pod2-campaign-folder",
])  # fmt: skip
def test_pod3_needs_a_runs_root_of_its_own(tmp_path, entry):
    """Pod 3 never writes into a root that holds another archive's runs, records, journals or
    campaign folders: refused before anything is written; its own entries pass (a restart)."""
    plan, runs = _pod3_root(tmp_path, entry)
    assert camp.own_root_problems(plan) == []
    {
        "pod2-order-run": lambda: _meta(runs, f"study5/{camp.POD2_ORDER_RUN}", "pod2-20261109"),
        "pod2-records": lambda: (runs / "study5_records" / "pod2-20261109").mkdir(),
        "pod2-journal": lambda: (
            runs / "launch" / "study5-pod2-20261109-controls-s50000.jsonl").write_text(""),
        "run-without-meta": lambda: (runs / "study5" / "20261109T000000Z-s5-fresh").mkdir(),
        "pod2-ladder-run": lambda: (
            runs / "study5_ladder" / "s5-pod2-20261109-ladder-linear-s50000").mkdir(),
        "pod2-scan-run": lambda: (
            runs / "study5_scan" / "s5-pod2-20261109-scan-s50000").mkdir(parents=True),
        "pod2-campaign-folder": lambda: (runs / "campaigns" / "study5-pod2").mkdir(),
    }[entry]()  # fmt: skip
    (found,) = camp.own_root_problems(plan)
    assert "is not pod3's own" in found


def test_pod3_needs_an_archive_id_of_its_own(where):
    for archive_id in ("pod2-20261109", "pod1-20261109", "t3"):
        plan = camp.pod3_plan(archive_id=archive_id, commit=COMMIT, gpu_code=GPU, **where)
        (found,) = camp.own_root_problems(plan)
        assert "pod3's archive id starts with 'pod3-'" in found
    plan = camp.pod3_plan(archive_id="pod3-20261111", commit=COMMIT, gpu_code=GPU, **where)
    assert camp.own_root_problems(plan) == []


def test_pod3_admits_a_group_only_when_its_work_and_the_close_fit_before_the_cap(
    where, tmp_path, monkeypatch
):
    """Pod 3's blocks and seed groups are admitted only when 1.5 × their estimated minutes and
    the close's 10 minutes end before the cap; the watchdog window stays 2 × the minutes + 60,
    cut at the cap, so an admitted group is never stopped early by its own window."""
    plan = pod3(where)
    deadlines = [s for b in plan.blocks if b.name != "close" for s in b.steps
                 if s.kind == "deadline"]  # fmt: skip
    assert deadlines and not any("--reserve" in s.argv for s in deadlines)
    assert all(s.on_fail == "stop" for s in deadlines)
    group = next(s for s in deadlines if s.group == "s50006")
    work = sum(s.minutes for s in steps(plan, "orders")
               if s.group == "s50006" and s.kind != "deadline")  # fmt: skip
    assert work == pytest.approx(4 * camp.F_RUN + camp.MINUTES["archive"])
    admit = int(group.argv[group.argv.index("--admit") + 1])
    assert admit == math.ceil(1.5 * work + 10)
    window = minutes_of(group)
    assert window == camp.deadline_minutes(work) > admit
    now = int(time.time())
    target = deadline_env(monkeypatch, tmp_path, now + 60 * admit - 120)
    record = tmp_path / "rec.json"
    assert camp.main([*group.argv[1:], "--record", str(record)]) == camp.NOT_ADMITTED
    assert not target.exists() and json.loads(record.read_text())["admitted"] is False
    cap = now + 60 * admit + 300  # the work and the close fit, the full window does not
    deadline_env(monkeypatch, tmp_path, cap)
    assert camp.main([*group.argv[1:], "--record", str(record)]) == 0
    assert int(target.read_text()) == cap and json.loads(record.read_text())["admitted"]
    assert not any("--admit" in s.argv for s in steps(pod2(where, 24)))


def pod4(where):
    return camp.pod4_plan(archive_id="pod4-20261112", commit=COMMIT, gpu_code=GPU, **where)


def test_pod4_extension_2(where, tmp_path):
    """Extension 2 (docs/study5-preregistration.md): the engineering rerun first; the 40 replays
    are pod 2's fresh η_l cells (the same config hashes), checked against pod 2's runs; step
    checkpoints only on the twins' and the force pilot's seeds; the twins' edited order; the
    sealed orders only as predictions from θ₀; the swaps and shuffles as edited orders."""
    plan = pod4(where)
    assert [b.name for b in plan.blocks] == ["engineering", "fetch", "replays", "vectors", "force",
                                             "linear", "sealed", "points", "swaps", "shuffles",
                                             "close"]  # fmt: skip
    assert all(s.on_fail == "stop" for s in steps(plan, "engineering"))
    replays = [s for s in steps(plan, "replays") if s.kind == "launch"]
    assert [s.group for s in replays] == ["g50000", *[f"g{s}" for s in range(50000, 50040, 4)]]
    seen = {}
    for launch_step in replays:
        stem = Path(launch_step.argv[1]).stem
        for r in campaign_runs(plan, tmp_path, stem):
            cfg = load_run_config(r.config, list(r.overrides))
            steps_ = cfg.tracking.step_checkpoints
            if "-repeat-" in stem:  # the twins' floor: FORCE_SEED's replay again, its order kept
                assert r.seed == camp.FORCE_SEED and cfg.study5.role == "fresh"
                assert config_hash(cfg)[:8] == camp.POD2_FRESH_L[r.seed].split("-")[-2]
                assert list(steps_) == list(camp.TWIN_STEPS)
                assert cfg.tracking.endpoint_checkpoint == "/root/ckpt/study5"
                assert cfg.experiment == r.experiment == camp.FLOOR_EXPERIMENT
                continue
            if cfg.study5.role == "fresh":
                assert config_hash(cfg)[:8] == camp.POD2_FRESH_L[r.seed].split("-")[-2]
                assert cfg.tracking.endpoint_checkpoint == "/root/ckpt/study5"
                seen[r.seed] = None if steps_ is None else list(steps_)
            else:  # a twin: the first responses of the batches at steps 2 and 3 exchanged
                assert r.seed in camp.SWAP_SEEDS
                assert dict(cfg.train.order_edit) == {"kind": "swap_first", "step": 2}
                assert list(steps_) == list(camp.TWIN_STEPS)
    assert sorted(seen) == list(range(50000, 50040))
    assert seen[50000] == list(camp.FORCE_SEED_STEPS)
    assert all(seen[s] == list(camp.TWIN_STEPS) for s in (50001, 50002, 50003))
    assert all(seen[s] is None for s in range(50004, 50040))
    checks = [s for s in steps(plan, "replays") if s.gate == "replay"]
    assert len(checks) == 10 and all(s.on_fail == "stop" for s in checks)
    hashes = [s for s in steps(plan, "replays") if "scripts/study5_tensor_hash.py" in s.argv]
    assert len(hashes) == 10 and all(s.on_fail == "stop" for s in hashes)
    for h, c in zip(hashes, checks, strict=True):  # each check reads its group's tensor hashes
        assert c.argv[c.argv.index("--rerun-endpoint-tensors") + 1] == h.outputs[0]
        expected = [c.argv[i + 1] for i, a in enumerate(c.argv) if a == "--endpoint-tensors"]
        assert len(expected) == 4 and all("=" in e for e in expected)
    assert ["--ignore" in s.argv for s in checks] == [True] + [False] * 9
    fetches = {s.argv[s.argv.index("--run-id") + 1] for s in steps(plan) if "fetch-run" in s.argv}
    assert fetches == {camp.POD2_ORDER_RUN, *camp.POD2_FRESH_L.values()}
    # the sealed orders: twenty predictions from θ₀, nothing of any E2 run
    sealed = [s for s in steps(plan, "sealed") if "scripts/study5_linear_replay.py" in s.argv]
    found = sorted((int(next(a for a in s.argv if a.startswith("seed=")).split("=")[1]),
                    int(next(a for a in s.argv if a.startswith("train.order_seed="))
                        .split("=")[1])) for s in sealed)  # fmt: skip
    assert found == sorted((s, s + o) for s in range(50006, 50016) for o in (1000, 2000))
    assert all("--reference-fresh-run" in s.argv for s in sealed)
    text = " ".join(" ".join(s.argv) for s in steps(plan))
    assert "s5-order1-" not in text and "s5-order2-" not in text and "pod3-" not in text
    assert "scripts/label_runs.py" not in text
    sealed_text = " ".join(" ".join(s.argv) for s in steps(plan, "sealed"))
    for script in ("study5_vector_points", "study5_pairs", "study5_tokens", "launch.py"):
        assert script not in sealed_text
    # the edited orders: swaps on four seeds, shuffles on six; each with its predictions
    hashes = set()
    for block, count in (("swaps", 3), ("shuffles", 4)):
        for launch_step in [s for s in steps(plan, block) if s.kind == "launch"]:
            runs = campaign_runs(plan, tmp_path, Path(launch_step.argv[1]).stem)
            assert len(runs) == count
            for r in runs:
                cfg = load_run_config(r.config, list(r.overrides))
                assert cfg.study5.role == "control" and cfg.optimizer.lr == 1.3e-05
                hashes.add(config_hash(cfg))
    assert len(hashes) == 4 * 3 + 6 * 4
    predictions = [s for s in steps(plan, "swaps") if "scripts/study5_linear_replay.py" in s.argv]
    assert len(predictions) == 4 * 4  # three swaps and the twin edit per seed
    points = [s for s in steps(plan, "points") if "scripts/study5_vector_points.py" in s.argv]
    labels = sorted(s.argv[s.argv.index("--label") + 1] for s in points)
    assert labels == sorted(["mean", "mean_k", "mean_neg_k", "mix1", "mix2", "mix3",
                             *[f"small_s{s}" for s in range(50000, 50006)],
                             *[f"rmean_s{s}" for s in range(50000, 50006)],
                             *[f"rmixture_s{s}" for s in range(50000, 50006)]])  # fmt: skip
    for s in points:
        if s.argv[s.argv.index("--label") + 1] in ("mean_k", "mean_neg_k", "mix1"):
            assert "--match" in s.argv
    assert "--sign" in next(s for s in points if "mean_neg_k" in s.argv).argv
    commands = {s.argv[1] for s in steps(plan) if s.argv[0] == "scripts/study5_campaigns.py"}
    assert not commands & {"checkpoint", "check-checkpoints", "export-checkpoints"}
    assert {"settle", "drop-checkpoints", "purge"} <= commands


def test_pod4_amendment_numerical_equivalence(where):
    """The amendment (docs/study5-preregistration.md): every rerun check under the tolerance; the
    replays' endpoints whose tensors differ measured against pod 2's checkpoint, fetched back by
    its archived sha256 and removed once read; the noise repeats on FORCE_SEED."""
    plan = pod4(where)
    tol = str(camp.EXT2_TOLERANCE["relative"])
    (eng,) = [s for s in steps(plan, "engineering") if s.gate == "engineering"]
    assert eng.argv[eng.argv.index("--tolerance") + 1] == tol
    assert eng.outputs[0].endswith("engineering-check-numerical.json")
    all_steps = steps(plan, "replays")
    checks = [s for s in all_steps if s.gate == "replay"]
    fetched = [s for s in all_steps if s.argv[:2] == ("scripts/study5_campaigns.py",
                                                       "fetch-endpoint")]  # fmt: skip
    assert len(fetched) == 40
    names = [s.name for s in all_steps]
    for f in fetched:
        seed = int(f.name.rsplit("-s", 1)[1])
        assert f.argv[f.argv.index("--sha256") + 1] == camp.POD2_CHECKPOINT_SHA256[seed]
        assert f.argv[f.argv.index("--expected") + 1] == camp.POD2_ENDPOINTS[seed][1]
        assert f.argv[f.argv.index("--remote") + 1] == f"{camp.POD2_ARCHIVE}/checkpoints"
        group = f.name.split("/")[1]
        distance = next(s for s in all_steps if s.name == f"replays/{group}/distance-s{seed}")
        remove = next(s for s in all_steps if s.name == f"replays/{group}/remove-s{seed}")
        check = next(s for s in checks if s.group == group)
        assert names.index(f.name) < names.index(distance.name) < names.index(remove.name)
        assert names.index(distance.name) < names.index(check.name)
        target = f"{f.argv[f.argv.index('--dest') + 1]}/{camp.POD2_FRESH_L[seed]}.pt"
        assert target.startswith("/root/ckpt/study5/pod2/")
        assert distance.argv[distance.argv.index("--only-if") + 1] == target
        assert f"ckpt:{target}:{camp.POD2_CHECKPOINT_SHA256[seed]}" in distance.argv
        assert remove.argv[remove.argv.index("--path") + 1] == target
        records = [check.argv[i + 1] for i, a in enumerate(check.argv)
                   if a == "--endpoint-distance"]  # fmt: skip
        assert f"{camp.POD2_ENDPOINTS[seed][0]}={distance.outputs[0]}" in records
    assert len(checks) == 10
    assert all(c.argv[c.argv.index("--tolerance") + 1] == tol for c in checks)
    floor = [
        s for s in all_steps if s.name == f"replays/g{camp.FORCE_SEED}/floor-s{camp.FORCE_SEED}"
    ]
    assert len(floor) == 1 and "--r" in floor[0].argv
    for f in fetched:  # a fetched checkpoint goes only once its distance record read it
        group, seed = f.name.split("/")[1], f.name.rsplit("-s", 1)[1]
        remove = next(s for s in all_steps if s.name == f"replays/{group}/remove-s{seed}")
        distance = next(s for s in all_steps if s.name == f"replays/{group}/distance-s{seed}")
        assert remove.argv[remove.argv.index("--record") + 1] == distance.outputs[0]
    (drop_steps,) = [s for s in steps(plan, "force") if s.name == "force/drop-steps"]
    consumers = [drop_steps.argv[i + 1] for i, a in enumerate(drop_steps.argv) if a == "--consumer"]
    assert any(floor[0].outputs[0].endswith(c) for c in consumers)  # the floor read them too
    floor_archive = next(s for s in all_steps if s.name == f"replays/g{camp.FORCE_SEED}/archive")
    named = [drop_steps.argv[i + 1] for i, a in enumerate(drop_steps.argv) if a == "--inventory"]
    assert floor_archive.outputs[0] in named
    for k in ("a", "b"):  # accumulated as the primary measurement is
        (repeat,) = [s for s in steps(plan, "force") if s.name == f"force/repeat-{k}"]
        assert "--split-halves" in repeat.argv
    linear = steps(plan, "linear")
    assert any(s.name == f"linear/g{camp.FORCE_SEED}/s{camp.FORCE_SEED}-repeat" for s in linear)
    force = [s.name for s in steps(plan, "force")]
    assert {"force/repeat-a", "force/repeat-b", "force/repeat-pairs"} <= set(force)
    good = {
        "kind": "study5-engineering-check",
        "exact": False,
        "passed": True,
        "within_tolerance": True,
        "tolerance": dict(camp.EXT2_TOLERANCE),
        "pairs": [],
    }
    assert camp.gate_problem("engineering", _write(where, "g.json", good), plan) == (
        "the engineering check holds no pair")  # fmt: skip
    other = {**good, "tolerance": {**camp.EXT2_TOLERANCE, "relative": 1e-3}}
    assert "another tolerance" in camp.gate_problem("engineering", _write(where, "o.json", other),
                                                    plan)  # fmt: skip
    failed = {**good, "passed": False}
    assert "found a difference" in camp.gate_problem(
        "engineering", _write(where, "f.json", failed), plan
    )
    earlier = {"kind": "study5-engineering-check", "equal": False, "pairs": []}
    assert "found a difference" in camp.gate_problem(
        "engineering", _write(where, "e.json", earlier), plan
    )


def test_pod4_amendment_2_endpoint_bound(where):
    """Extension 2's second amendment: the endpoint bound 1e-3 of its own beside the measurements'
    1e-4, passed to every rerun check; each replay group checked into a record named by the
    bound, never the earlier ``check-<group>.json``; a check recorded under a stricter endpoint
    bound stands, a looser one or another measurement rule does not."""
    assert camp.EXT2_TOLERANCE == {"relative": 1e-4, "absolute": 1e-9, "answers_allowed": 2,
                                   "endpoint_distance": 1e-3}  # fmt: skip
    plan = pod4(where)
    (eng,) = [s for s in steps(plan, "engineering") if s.gate == "engineering"]
    assert eng.argv[eng.argv.index("--endpoint-tolerance") + 1] == "0.001"
    checks = [s for s in steps(plan, "replays") if s.gate == "replay"]
    assert len(checks) == 10
    for c in checks:
        assert c.argv[c.argv.index("--tolerance") + 1] == "0.0001"
        assert c.argv[c.argv.index("--endpoint-tolerance") + 1] == "0.001"
        assert c.outputs[0].endswith(f"/replays/check-{c.group}-endpoint-analysis-0.001.json")
        assert c.argv[c.argv.index("--out") + 1] == c.outputs[0]
    stands = camp.tolerance_stands
    assert stands(dict(camp.EXT2_TOLERANCE))
    assert stands({**camp.EXT2_TOLERANCE, "endpoint_distance": 1e-4})  # the first amendment's
    for bad in ({**camp.EXT2_TOLERANCE, "endpoint_distance": 1e-2},
                {**camp.EXT2_TOLERANCE, "endpoint_distance": 0.0},
                {**camp.EXT2_TOLERANCE, "endpoint_distance": True},
                {**camp.EXT2_TOLERANCE, "endpoint_distance": None},
                {**camp.EXT2_TOLERANCE, "relative": 1e-3},
                {**camp.EXT2_TOLERANCE, "answers_allowed": 3},
                {k: v for k, v in camp.EXT2_TOLERANCE.items() if k != "absolute"},
                {**camp.EXT2_TOLERANCE, "extra": 1},
                None, [1e-4]):  # fmt: skip
        assert not stands(bad), bad
    good = {"kind": "study5-engineering-check", "exact": False, "passed": True,
            "within_tolerance": True, "pairs": []}  # fmt: skip
    stricter = {**good, "tolerance": {**camp.EXT2_TOLERANCE, "endpoint_distance": 1e-4}}
    assert camp.gate_problem("engineering", _write(where, "s.json", stricter), plan) == (
        "the engineering check holds no pair")  # past the tolerance  # fmt: skip
    looser = {**good, "tolerance": {**camp.EXT2_TOLERANCE, "endpoint_distance": 1e-2}}
    assert "another tolerance" in camp.gate_problem(
        "engineering", _write(where, "l.json", looser), plan
    )


def test_pod4_amendment_3_replays_accepted_for_endpoint_analysis(where):
    """Extension 2's third amendment: every replay group's check runs with --endpoint-analysis
    into a record of its own and is kept by ``accepted_for_endpoint_analysis`` (the gate
    ``replay``), its ``passed`` aside; the engineering gate still reads ``passed``; a record
    without the acceptance (the second amendment's) does not stand for a replay group."""
    plan = pod4(where)
    checks = [s for s in steps(plan, "replays") if s.name.endswith("/check")]
    assert len(checks) == 10 and {c.gate for c in checks} == {"replay"}
    assert all("--endpoint-analysis" in c.argv for c in checks)
    (eng,) = [s for s in steps(plan, "engineering") if s.gate == "engineering"]
    assert "--endpoint-analysis" not in eng.argv
    accepted = {"kind": "study5-engineering-check", "exact": False, "passed": False,
                "within_tolerance": False, "accepted_for_endpoint_analysis": True,
                "tolerance": dict(camp.EXT2_TOLERANCE), "pairs": []}  # fmt: skip
    assert camp.gate_problem("replay", _write(where, "a3.json", accepted), plan) == (
        "the engineering check holds no pair")  # past the verdict and the tolerance  # fmt: skip
    assert "found a difference" in camp.gate_problem(
        "engineering", _write(where, "a3e.json", accepted), plan
    )
    second = {k: v for k, v in accepted.items() if k != "accepted_for_endpoint_analysis"}
    for name, record in (("no.json", {**accepted, "accepted_for_endpoint_analysis": False}),
                         ("a2.json", {**second, "passed": True})):  # fmt: skip
        assert "did not accept" in camp.gate_problem("replay", _write(where, name, record), plan)


def test_pod4_floor_repeat_runs_anew_beside_the_completed_replay(where, tmp_path):
    """The launcher reuses a completed run of the same experiment and config hash: the floor
    repeat has the replay's config hash, in an experiment folder of its own, so with the replay
    completed the replay is done and the repeat is not."""
    from zero_field_refusal.launch import assess

    plan = pod4(where)
    launches = [s for s in steps(plan, "replays") if s.kind == "launch"]
    first, repeat = (campaign_runs(plan, tmp_path, Path(s.argv[1]).stem) for s in launches[:2])
    replay = next(r for r in first if r.seed == camp.FORCE_SEED and "-fresh-" in r.cell)
    (floor,) = repeat
    assert floor.config_hash == replay.config_hash and floor.experiment != replay.experiment
    runs = tmp_path / "runs"
    done = runs / replay.experiment / f"20261111T000000Z-x-{replay.config_hash[:8]}-a1"
    done.mkdir(parents=True)
    (done / "meta.json").write_text(json.dumps({"config_hash": replay.config_hash,
                                                "status": "completed"}))  # fmt: skip
    found = assess(replay, runs)
    assert found.done == done or done.name in found.conflicts  # seen: reused or refused
    found = assess(floor, runs)
    assert found.done is None and not found.conflicts and not found.technical


def _write(where, name, record) -> Path:
    path = where["runs"] / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def test_pod4_numerical_check_keeps_the_earlier_reruns_and_records(where):
    """A restart on the volume of the first attempt: the bitwise record stays and is archived,
    and the diagnostic rerun beside it enters the numerical check with the campaign's own."""
    records = where["runs"] / "study5_records" / "pod4-20261112"
    records.mkdir(parents=True)
    (records / "engineering-check.json").write_text("{}")
    diag = where["runs"] / camp.POD4_DIAG / camp.POD4_DIAG_RERUN
    diag.mkdir(parents=True)
    plan = pod4(where)
    (eng,) = [s for s in steps(plan, "engineering") if s.gate == "engineering"]
    reruns = eng.argv[eng.argv.index("--rerun") + 1 : eng.argv.index("--out")]
    assert len(reruns) == 2 and reruns[1] == str(diag)
    (archive,) = [s for s in steps(plan, "engineering") if s.name == "engineering/archive"]
    paths = [archive.argv[i + 1] for i, a in enumerate(archive.argv) if a == "--path"]
    assert paths == ["study5_records/pod4-20261112/engineering-check.json", camp.POD4_DIAG]


def test_fetch_endpoint_fetches_only_differing_tensors_by_the_archived_sha(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "replay-a1"
    run.mkdir(parents=True)
    monkeypatch.setattr(camp, "completed_run", lambda runs, campaign, run_label: run)
    hashed = tmp_path / "hashed.json"
    hashed.write_text(json.dumps({"replay-a1": {"file_ok": True, "tensors_sha256": "a" * 64}}))
    content = b"pod2 checkpoint"
    sha = hashlib.sha256(content).hexdigest()
    calls = []
    monkeypatch.setattr(camp, "archived_earlier", lambda args, name, recorded: recorded == sha)
    monkeypatch.setattr(camp, "artifact_remote", lambda: "gs://bucket")

    def fake_gcloud(argv):
        calls.append(argv)
        Path(argv[-1]).write_bytes(content)
        return ""

    monkeypatch.setattr(camp, "gcloud", fake_gcloud)
    dest = tmp_path / "pod2"
    argv = ["fetch-endpoint", "--runs", str(tmp_path / "runs"), "--campaign", "c", "--label", "x",
            "--hashed", str(hashed), "--expected", "a" * 64, "--run-id", "pod2-run-a1",
            "--sha256", sha, "--remote", "study5/pod2/checkpoints", "--inventories-remote",
            "study5/pod2/inventories", "--dest", str(dest)]  # fmt: skip
    assert camp.main(argv) == 0 and not calls and not dest.exists()  # equal tensors
    argv[argv.index("--expected") + 1] = "b" * 64
    assert camp.main(argv) == 0 and len(calls) == 1
    assert (dest / "pod2-run-a1.pt").read_bytes() == content
    assert camp.main(argv) == 0 and len(calls) == 1  # already there with its sha256
    (dest / "pod2-run-a1.pt").unlink()
    argv[argv.index("--sha256") + 1] = "0" * 64  # not shown verified in its archive
    assert camp.main(argv) == 1 and not (dest / "pod2-run-a1.pt").exists()
    record = tmp_path / "distance.json"
    remove = ["remove-fetched", "--path", str(dest / "pod2-run-a1.pt"), "--sha256", sha,
              "--record", str(record)]  # fmt: skip
    assert camp.main(remove) == 0  # nothing there
    (dest / "pod2-run-a1.pt").write_bytes(content)
    assert camp.main(remove) == 1 and (dest / "pod2-run-a1.pt").exists()  # no record yet
    record.write_text(json.dumps({"available": False, "missing": "pod2-run-a1.pt"}))
    assert camp.main(remove) == 1 and (dest / "pod2-run-a1.pt").exists()  # not computed
    row = {"a": {"kind": "ckpt", "sha256": sha}, "b": {"kind": "ckpt"}, "distance": 1.0}
    record.write_text(json.dumps({"kind": "study5-pairs", "pairs": [row]}))
    (dest / "pod2-run-a1.pt").write_bytes(b"something else")
    assert camp.main(remove) == 1 and (dest / "pod2-run-a1.pt").exists()
    (dest / "pod2-run-a1.pt").write_bytes(content)
    assert camp.main(remove) == 0 and not (dest / "pod2-run-a1.pt").exists()


def test_pod4_needs_extension_2s_gpu_code_registration_and_root(where, monkeypatch, tmp_path):
    plan = pod4(where)
    problems = camp.preflight(plan)
    assert any("study5-extension-2-code-identity.json" in p for p in problems) or not (
        REPO_ROOT / camp.EXT2_IDENTITY).exists()  # fmt: skip
    monkeypatch.setattr(camp, "EXT2_HEADING", "### Extension 9")
    assert any("does not hold extension 2" in p for p in camp.preflight(plan))
    for archive_id in ("pod3-20261111", "pod2-20261109", "t4"):
        other = camp.pod4_plan(archive_id=archive_id, commit=COMMIT, gpu_code=GPU, **where)
        (found,) = camp.own_root_problems(other)
        assert "pod4's archive id starts with 'pod4-'" in found
    assert camp.own_root_problems(plan) == []
    earlier = where["runs"] / "campaigns" / "study5-pod4-first"  # an earlier attempt's --out
    earlier.mkdir(parents=True)
    (earlier / "plan.json").write_text(json.dumps({"pod": "pod4", "archive_id": plan.archive_id}))
    assert camp.own_root_problems(plan) == []
    (earlier / "plan.json").write_text(json.dumps({"pod": "pod4", "archive_id": "pod4-x"}))
    assert camp.own_root_problems(plan) != []
    (earlier / "plan.json").unlink()
    assert camp.own_root_problems(plan) != []
    earlier.rmdir()
    (where["runs"] / "study5_vector" / "s5-pod3-20261111-x").mkdir(parents=True)
    (found,) = camp.own_root_problems(plan)
    assert "is not pod4's own" in found


def test_run_references_inside_arguments_and_across_campaigns(tmp_path, monkeypatch):
    calls = []

    def completed(runs, campaign, run_label):
        calls.append((campaign, run_label))
        return runs / campaign / run_label

    monkeypatch.setattr(camp, "completed_run", completed)
    argv = ["x", "{run:a}", "run:{run:b@other}:5", "--pair", "{run:c@third}"]
    out = camp.substituted(argv, tmp_path, "own")
    assert out == ["x", str(tmp_path / "own" / "a"), f"run:{tmp_path / 'other' / 'b'}:5",
                   "--pair", str(tmp_path / "third" / "c")]  # fmt: skip
    assert calls == [("own", "a"), ("other", "b"), ("third", "c")]


def _inventory(path, rows) -> None:
    lines = ["path\tsize\tsha256\tmd5_b64\turl\tstatus"]
    lines += [f"{rel}\t1\t{sha}\tx\tgs://b/{rel}\t{status}" for rel, sha, status in rows]
    path.write_text("\n".join(lines) + "\n")


def test_settle_copies_records_and_removes_nothing(tmp_path):
    folder, records = tmp_path / "vec", tmp_path / "records"
    folder.mkdir()
    (folder / "a.pt").write_bytes(b"tensor")
    (folder / "a.pt.json").write_text("{}")
    (folder / "record.json").write_text('{"x": 1}')
    assert camp.main(["settle", "--dir", str(folder), "--records", str(records)]) == 0
    assert camp.main(["settle", "--dir", str(folder), "--records", str(records)]) == 0  # again
    assert sorted(p.name for p in folder.iterdir()) == ["a.pt", "a.pt.json", "record.json"]
    assert sorted(p.name for p in records.iterdir()) == ["a.pt.json", "record.json"]
    (folder / "record.json").write_text('{"x": 2}')
    assert camp.main(["settle", "--dir", str(folder), "--records", str(records)]) == 1


def test_purge_needs_every_consumer_in_a_verified_archive(tmp_path):
    """Vector files go only once every record that read them is written and in an archive's
    inventory with status ok and the local bytes; otherwise nothing is removed."""
    runs, folder = tmp_path / "runs", tmp_path / "vec"
    (runs / "rec").mkdir(parents=True)
    folder.mkdir()
    (folder / "v.pt").write_bytes(b"tensor")
    (folder / "v.pt.json").write_text("{}")
    consumer = runs / "rec" / "points.json"
    inventory = tmp_path / "inv.tsv"
    argv = ["purge", "--dir", str(folder), "--runs", str(runs), "--inventory", str(inventory),
            "--consumer", "rec/points.json"]  # fmt: skip
    assert camp.main(argv) == 1  # no inventory, no record
    consumer.write_text('{"r": 1}')
    assert camp.main(argv) == 1  # written, not archived
    sha = hashlib.sha256(consumer.read_bytes()).hexdigest()
    _inventory(inventory, [("rec/points.json", sha, "MISMATCH remote=1,x")])
    assert camp.main(argv) == 1  # archived wrongly
    _inventory(inventory, [("rec/points.json", "0" * 64, "ok")])
    assert camp.main(argv) == 1  # another file's bytes
    assert (folder / "v.pt").exists()
    _inventory(inventory, [("rec/points.json", sha, "ok")])
    assert camp.main(argv) == 0
    assert sorted(p.name for p in folder.iterdir()) == ["v.pt.json"]  # the record stays


def test_drop_checkpoints_needs_archived_consumers_and_the_recorded_bytes(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    files = {}
    for step in (2, 3, 25):
        path = tmp_path / f"r-step-{step}.pt"
        path.write_bytes(f"theta{step}".encode())
        files[step] = path
    sha = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}
    inputs = {
        "step_checkpoints": [{"path": str(files[t]), "sha256": sha[t], "step": t} for t in (2, 3)],
        "endpoint_checkpoint": {"path": str(files[25]), "sha256": sha[25], "step": 25},
    }
    (run / "meta.json").write_text(json.dumps({"inputs": inputs}))  # fmt: skip
    monkeypatch.setattr(camp, "completed_run", lambda runs, campaign, run_label: run)
    record, inventory = tmp_path / "pairs.json", tmp_path / "inv.tsv"
    argv = ["drop-checkpoints", "--runs", str(tmp_path), "--campaign", "c", "--run", "x",
            "--inventory", str(inventory), "--consumer", "pairs.json"]  # fmt: skip
    assert camp.main(argv) == 1 and all(p.exists() for p in files.values())  # nothing archived
    record.write_text("{}")
    _inventory(inventory, [("pairs.json", hashlib.sha256(b"{}").hexdigest(), "ok")])
    files[3].write_bytes(b"other")
    assert camp.main(argv) == 1  # a file that is not the recorded one is kept
    assert not files[2].exists() and files[3].exists() and files[25].exists()
    files[3].write_bytes(b"theta3")
    assert camp.main(argv) == 0 and files[25].exists()  # the endpoint only with --endpoint
    assert camp.main([*argv, "--endpoint"]) == 0
    assert not any(p.exists() for p in files.values())


def test_pod4_removes_only_after_a_verified_archive_and_never_a_replays_endpoint(where):
    """Every purge and drop names inventories of archive steps that come before it; the twins'
    endpoints may go, the replays' never; the δ_lin consumers run only when δ_lin was written."""
    plan = pod4(where)
    archived = set()
    for s in steps(plan):
        if s.kind == "archive":
            archived.update(s.outputs)
        if s.argv[:2] in (("scripts/study5_campaigns.py", "purge"),
                          ("scripts/study5_campaigns.py", "drop-checkpoints")):  # fmt: skip
            named = [s.argv[i + 1] for i, a in enumerate(s.argv) if a == "--inventory"]
            assert named and set(named) <= archived, s.name
            assert "--consumer" in s.argv
        if s.argv[:2] == ("scripts/study5_campaigns.py", "drop-checkpoints"):
            runs_ = [s.argv[i + 1] for i, a in enumerate(s.argv) if a == "--run"]
            if "--endpoint" in s.argv:  # the twins' and the floor repeat's, never a replay's
                campaign = s.argv[s.argv.index("--campaign") + 1]
                assert all("-edit-twin-" in r for r in runs_) or "-repeat-" in campaign, s.name
        if "scripts/study5_pairs.py" in s.argv and "delta_lin.pt" in " ".join(s.argv):
            assert "--only-if" in s.argv
    close = steps(plan, "close")
    records = next(s for s in close if s.name == "close/records")
    assert records.always and "--mutable-path" in records.argv
    for s in steps(plan):  # the deletions and settles only copy or remove inside the volume
        if s.argv[:2] == ("scripts/study5_campaigns.py", "purge"):
            folder = s.argv[s.argv.index("--dir") + 1]
            assert folder.startswith("/root/ckpt/study5/vec/")


def test_pod4_windows_cover_their_groups(where):
    """After the first seed group of a block only the block's snapshot and receipt run outside a
    group: every other step is inside a group with a window of its own."""
    plan = pod4(where)
    for block in plan.blocks:
        seen, tail = False, []
        for s in block.steps:
            if s.group:
                seen, tail = True, []
            elif seen and s.kind != "deadline":
                tail.append(s.name)
        assert all(n.endswith(("/snapshot", "/receipt")) for n in tail), (block.name, tail)


def test_no_plan_holds_a_bucket_path_or_a_key(where, monkeypatch):
    monkeypatch.setenv("ARTIFACT_REMOTE", "gs://secret-bucket/prefix")
    for plan in (pod1(where), pod2(where), pod3(where), pod4(where), topup(where)):
        text = json.dumps(plan.to_json()) + "".join(plan.campaigns.values())
        assert "gs://" not in text and "secret-bucket" not in text and "KEY" not in text


def test_the_plan_is_data_and_deterministic(where):
    assert pod1(where).to_json() == pod1(where).to_json()
    assert pod2(where).to_json() == pod2(where).to_json()
    assert pod3(where).to_json() == pod3(where).to_json()
    assert pod4(where).to_json() == pod4(where).to_json()
    assert topup(where).to_json() == topup(where).to_json()
    for plan in (pod2(where), pod3(where), pod4(where), topup(where)):
        names = [s.name for s in steps(plan)]
        assert len(names) == len(set(names))


# ------------------------------------------------------------------ the executor


def py(code: str) -> tuple[str, ...]:
    return ("-c", code)


def fake_plan(tmp_path, blocks):
    return camp.Plan(
        pod="test",
        archive_id="t",
        out=tmp_path / "c",
        runs=tmp_path / "r",
        data=tmp_path / "d",
        checkpoints=tmp_path / "k",
        commit=COMMIT,
        gpu_code=GPU,
        n=None,
        blocks=blocks,
        campaigns={},
    )


def run_plan(plan, tmp_path):
    plan.out.mkdir(parents=True, exist_ok=True)
    marks = tmp_path / "marks"
    marks.mkdir(exist_ok=True)
    return camp.execute(plan, python=[sys.executable], cwd=tmp_path, env={})


def mark(tmp_path, name, code=0):
    return py(
        f"import pathlib,sys; pathlib.Path(r'{tmp_path}/marks/{name}').touch(); sys.exit({code})"
    )


def marks(tmp_path):
    return sorted(p.name for p in (tmp_path / "marks").iterdir())


def test_execute_records_a_failure_and_goes_on_and_stops_at_a_stop(tmp_path):
    S = camp.Step
    blocks = [
        camp.Block(
            "one",
            (
                S("one/a", mark(tmp_path, "a", 1), on_fail="record"),
                S("one/b", mark(tmp_path, "b")),
                S("one/c", mark(tmp_path, "c", 2), on_fail="stop"),
                S("one/d", mark(tmp_path, "d")),
                S("one/e", mark(tmp_path, "e"), kind="archive", on_fail="stop"),
                S("one/f", mark(tmp_path, "f"), kind="receipt", on_fail="stop"),
            ),
        ),
        camp.Block("two", (S("two/a", mark(tmp_path, "two")),)),
        camp.Block("close", (S("close/a", mark(tmp_path, "close"), always=True),)),
    ]
    plan = fake_plan(tmp_path, blocks)
    assert run_plan(plan, tmp_path) == camp.STOPPED
    assert marks(tmp_path) == ["a", "b", "c", "close", "e"]
    status = camp.read_journal(plan.out / "journal.jsonl")
    assert status["one/a"] == "failed" and status["one/c"] == "failed"
    assert status["one/d"] == "skipped" and "one/f" not in status or status["one/f"] == "skipped"
    # a restart stops again at the recorded stop: nothing done is run again, close always runs
    for p in (tmp_path / "marks").iterdir():
        p.unlink()
    assert run_plan(plan, tmp_path) == camp.STOPPED
    assert marks(tmp_path) == ["close"]


def test_execute_skips_a_block_with_its_receipt_and_reruns_an_interrupted_step(tmp_path):
    S = camp.Step
    receipt = tmp_path / "c" / "receipts" / "one.json"
    blocks = [
        camp.Block("one", (S("one/a", mark(tmp_path, "a")),)),
        camp.Block(
            "two", (S("two/a", mark(tmp_path, "two-a")), S("two/b", mark(tmp_path, "two-b")))
        ),
        camp.Block("close", (S("close/a", mark(tmp_path, "close"), always=True),)),
    ]
    plan = fake_plan(tmp_path, blocks)
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}")
    journal = plan.out / "journal.jsonl"
    journal.write_text(
        json.dumps({"event": "end", "step": "two/a", "status": "done", "exit": 0})
        + "\n"
        + json.dumps({"event": "start", "step": "two/b"})
        + "\n"
    )
    assert run_plan(plan, tmp_path) == camp.COMPLETED
    assert marks(tmp_path) == ["close", "two-b"]


def test_execute_skip_group_and_outputs_and_an_existing_stream(tmp_path):
    S = camp.Step
    out = tmp_path / "exists.json"
    out.write_text("{}")
    exists = py(f"import sys; print('x: {camp.EXISTS}'); sys.exit(1)")
    blocks = [
        camp.Block(
            "one",
            (
                S("one/s1/gen", exists, group="s1", exists_ok=True, on_fail="skip-group"),
                S("one/s1/pre", mark(tmp_path, "pre", 1), group="s1", on_fail="skip-group"),
                S("one/s1/predict", mark(tmp_path, "predict"), group="s1"),
                S("one/s1/archive", mark(tmp_path, "archive"), group="s1", kind="archive"),
                S("one/s2/predict", mark(tmp_path, "kept"), group="s2", outputs=(str(out),)),
                S("one/s2/next", mark(tmp_path, "s2-next"), group="s2"),
            ),
        ),
        camp.Block("close", ()),
    ]
    plan = fake_plan(tmp_path, blocks)
    assert run_plan(plan, tmp_path) == camp.COMPLETED
    assert marks(tmp_path) == ["archive", "pre", "s2-next"]
    status = camp.read_journal(plan.out / "journal.jsonl")
    assert status["one/s1/gen"] == "kept" and status["one/s1/predict"] == "skipped"
    assert status["one/s2/predict"] == "kept"


# ------------------------------------------------------------------ helpers of the steps


def journal_of(runs: Path, name: str, lines: list[dict]) -> None:
    path = runs / "launch" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))


def test_a_label_resolves_to_its_completed_run_only(tmp_path):
    runs = tmp_path / "runs"
    journal_of(
        runs,
        "c",
        [
            {"event": "start", "run": "cell/s1", "run_id": "r-a1", "experiment": "study5"},
            {"event": "end", "run": "cell/s1", "run_id": "r-a1", "outcome": "done"},
            {"event": "start", "run": "bad/s1", "run_id": "b-a1", "experiment": "study5"},
            {"event": "end", "run": "bad/s1", "run_id": "b-a1", "outcome": "technical"},
        ],
    )
    (runs / "study5" / "r-a1").mkdir(parents=True)
    assert camp.completed_run(runs, "c", "cell/s1") == runs / "study5" / "r-a1"
    for label in ("bad/s1", "none/s1"):
        with pytest.raises(camp.StepError):
            camp.completed_run(runs, "c", label)
    assert camp.campaign_paths(runs, "c") == ["launch/c.jsonl", "study5/r-a1"]


def test_with_runs_substitutes_and_moves_a_partial_directory_aside(tmp_path):
    runs = tmp_path / "runs"
    journal_of(
        runs,
        "c",
        [
            {"event": "start", "run": "cell/s1", "run_id": "r-a1", "experiment": "study5"},
            {"event": "end", "run": "cell/s1", "run_id": "r-a1", "outcome": "done"},
        ],
    )
    (runs / "study5" / "r-a1").mkdir(parents=True)
    partial = runs / "study5_ladder" / "lad"
    partial.mkdir(parents=True)
    argv = camp.substituted(["x", "--f-run", "{run:cell/s1}"], runs, "c")
    assert argv == ["x", "--f-run", str(runs / "study5" / "r-a1")]
    moved = camp.move_aside(runs, "study5_ladder/lad")
    assert not partial.exists() and moved.exists() and moved.parent.name == "aside"
    assert camp.move_aside(runs, "study5_ladder/lad") is None
    with pytest.raises(camp.StepError):
        camp.move_aside(runs, "../outside")


def test_the_stream_inventory_selects_each_seeds_two_streams(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_REMOTE", "gs://b/p")
    rows = []
    for seed in (1, 2):
        for kind, files in (
            ("training", ["chunk-00000/samples.jsonl", ".lock"]),
            ("preconditioner", ["samples.jsonl"]),
        ):
            for name in files:
                path = f"teacher/M/{kind}/h/seed-{seed}/{name}"
                rows.append(
                    {
                        "path": path,
                        "size": "3",
                        "sha256": "a" * 64,
                        "md5_b64": "x",
                        "url": f"gs://b/p/data/{path}",
                        "status": "ok",
                    }
                )
    chosen = camp.stream_rows(rows, [2])
    assert sorted(r["path"] for r in chosen) == sorted(r["path"] for r in rows[3:])
    with pytest.raises(camp.StepError, match="seed 3"):
        camp.stream_rows(rows, [3])
    rows[3]["url"] = "gs://other/data/x"
    with pytest.raises(camp.StepError, match="url"):
        camp.stream_rows(rows, [2])
    rows[3]["url"] = f"gs://b/p/data/{rows[3]['path']}"
    rows[3]["path"] = "teacher/M/training/h/seed-2/../../x"
    with pytest.raises(camp.StepError):
        camp.stream_rows(rows, [2])


def test_the_predictions_record_holds_every_seed_in_order(tmp_path):
    records = tmp_path / "predictions"
    records.mkdir()
    for seed in (50001, 50000):
        (records / f"s{seed}.json").write_text(
            json.dumps({"kind": "study5-prediction", "seed": seed, "lr": 1.3e-05, "p": seed * 1e-9})
        )
    out = tmp_path / "record.json"
    record = camp.predictions_record(records, [50000, 50001], identity={"sha256": "e" * 64})
    assert [e["seed"] for e in record["predictions"]] == [50000, 50001]
    for entry in record["predictions"]:
        data = (records / f"s{entry['seed']}.json").read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    digest = camp.write_record(out, record)
    assert (tmp_path / "record.json.sha256").read_text() == f"{digest}  record.json\n"
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()
    with pytest.raises(camp.StepError, match="50002"):
        camp.predictions_record(records, [50000, 50001, 50002], identity={})
    (records / "s50001.json").write_text(
        json.dumps({"kind": "study5-prediction", "seed": 50001, "lr": 1.83e-05, "p": 0.0})
    )
    with pytest.raises(camp.StepError, match="lr"):
        camp.predictions_record(records, [50000, 50001], identity={})


def test_a_checkpoint_is_removed_only_inside_its_directory_once_verified(tmp_path):
    root = tmp_path / "ckpt"
    root.mkdir()
    ckpt = root / "run-a1.pt"
    ckpt.write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    inventory = [{"path": "run-a1.pt", "sha256": digest, "status": "ok"}]
    outside = tmp_path / "run-a1.pt"
    outside.write_bytes(b"weights")
    with pytest.raises(camp.StepError):
        camp.remove_checkpoint(outside, root, digest, inventory)
    link = root / "link.pt"
    link.symlink_to(outside)
    with pytest.raises(camp.StepError):
        camp.remove_checkpoint(link, root, digest, [{**inventory[0], "path": "link.pt"}])
    with pytest.raises(camp.StepError):
        camp.remove_checkpoint(ckpt, root, "0" * 64, inventory)
    with pytest.raises(camp.StepError):
        camp.remove_checkpoint(ckpt, root, digest, [{**inventory[0], "status": "MISMATCH"}])
    with pytest.raises(camp.StepError):
        camp.remove_checkpoint(root, root, digest, inventory)
    camp.remove_checkpoint(ckpt, root, digest, inventory)
    assert not ckpt.exists() and outside.exists()


def test_check_checkpoints_counts_the_files_on_the_disk(tmp_path):
    root = tmp_path / "ckpt"
    assert camp.main(["check-checkpoints", "--dir", str(root), "--max", "1"]) == 0
    root.mkdir()
    (root / "a.pt").write_bytes(b"1")
    assert camp.main(["check-checkpoints", "--dir", str(root), "--max", "1"]) == 0
    (root / "b.pt").write_bytes(b"1")
    assert camp.main(["check-checkpoints", "--dir", str(root), "--max", "1"]) == 1


# ------------------------------------------------------------------ the bucket steps, on a fake


FAKE_GCLOUD = r"""#!PYTHON
import base64, hashlib, json, os, shutil, sys
from pathlib import Path
assert os.environ["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"]
store = Path(os.environ["FAKE_BUCKETS"])
args = sys.argv[1:]
assert args[0] == "storage"
def local(url):
    return store / url.removeprefix("gs://")
if args[1] == "cp":
    if os.environ.get("FAKE_GCLOUD_FAIL_UPLOAD") and not args[-2].startswith("gs://"):
        sys.exit(1)  # an upload that fails (a test's forced failure)
    flags = {a for a in args if a.startswith("--")}
    src, dst = [a for a in args[2:] if not a.startswith("--")]
    def put(source, target):
        if target.exists() and "--no-clobber" in flags:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if src.startswith("gs://"):
        if not local(src).is_file():
            sys.exit(1)
        put(local(src), Path(dst))
    elif Path(src).is_dir():
        for f in Path(src).rglob("*"):
            if f.is_file():
                put(f, local(dst) / Path(src).name / f.relative_to(src))
    else:
        put(Path(src), local(dst))
elif args[1:3] == ["objects", "list"]:
    url = args[3]
    base = local(url.removesuffix("/**"))
    found = [p for p in base.rglob("*") if p.is_file()] if url.endswith("/**") else (
        [base] if base.is_file() else [])
    out = []
    for p in found:
        rel = p.relative_to(store).as_posix()
        bucket, name = rel.split("/", 1)
        data = p.read_bytes()
        out.append({"bucket": bucket, "name": name, "size": str(len(data)),
                    "md5_hash": base64.b64encode(hashlib.md5(data).digest()).decode()})
    print(json.dumps(out))
elif args[1:3] == ["objects", "describe"]:
    assert local(args[3]).is_file()
    print(json.dumps({"creation_time": "2026-11-09T00:00:00+0000"}))
else:
    sys.exit(f"unsupported: {args}")
"""


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud = bin_dir / "gcloud"
    gcloud.write_text(FAKE_GCLOUD.replace("PYTHON", sys.executable, 1))
    gcloud.chmod(0o755)
    store = tmp_path / "buckets"
    (store / "bkt" / "proj").mkdir(parents=True)
    key = tmp_path / "key.json"
    key.write_text("{}")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_BUCKETS", str(store))
    monkeypatch.setenv("ARTIFACT_GCS_KEY", str(key))
    monkeypatch.setenv("ARTIFACT_REMOTE", "gs://bkt/proj")
    return store / "bkt" / "proj"


def test_archive_puts_the_campaigns_runs_and_its_inventory_in_the_bucket(tmp_path, bucket):
    """Written-once files under the archive's runs: the records and an ended run's directory;
    under the attempt's place the mutable ones: the launcher journal, a run without an end, a
    gate's record and a scan run whose record does not exist yet."""
    runs = tmp_path / "runs"
    journal_of(
        runs, "c", [{"event": "start", "run": "cell/s1", "run_id": "r-a1", "experiment": "study5"},
                    {"event": "start", "run": "cell/s2", "run_id": "r-b1", "experiment": "study5"},
                    {"event": "end", "run": "cell/s2", "run_id": "r-b1", "outcome": "done"}]
    )  # fmt: skip
    for run_id in ("r-a1", "r-b1"):
        (runs / "study5" / run_id).mkdir(parents=True)
        (runs / "study5" / run_id / "meta.json").write_text("{}")
    (runs / "rec.json").write_text("{}")
    (runs / "gate.json").write_text("{}")
    (runs / "scan" / "partial").mkdir(parents=True)
    (runs / "scan" / "partial" / "rows.jsonl").write_text("{}")
    inventory = tmp_path / "c" / "inventories" / "x.tsv"
    argv = [
        "archive",
        "--inventory",
        str(inventory),
        "--remote",
        "study5/t/runs",
        "--root",
        str(runs),
        "--inventories-remote",
        "study5/t/inventories",
        "--path",
        "rec.json",
        "--path",
        "missing.json",
        "--mutable-path",
        "gate.json",
        "--run-dir",
        "scan/partial=scan-record.json",
        "--attempt-remote",
        "study5/t/attempts/a1/runs",
        "--campaign",
        "c",
        "--runs",
        str(runs),
    ]
    assert camp.main(argv) == 0
    rows = camp.read_inventory(inventory)
    assert sorted(r["path"] for r in rows) == [
        "gate.json",
        "launch/c.jsonl",
        "rec.json",
        "scan/partial/rows.jsonl",
        "study5/r-a1/meta.json",
        "study5/r-b1/meta.json",
    ]
    assert all(r["status"] == "ok" for r in rows)
    stable, attempt = bucket / "study5/t/runs", bucket / "study5/t/attempts/a1/runs"
    for rel in ("rec.json", "study5/r-b1/meta.json"):
        assert (stable / rel).is_file() and not (attempt / rel).exists(), rel
    for rel in ("gate.json", "launch/c.jsonl", "scan/partial/rows.jsonl", "study5/r-a1/meta.json"):
        assert (attempt / rel).is_file() and not (stable / rel).exists(), rel
    assert (bucket / "study5/t/inventories/x.tsv").read_bytes() == inventory.read_bytes()
    assert camp.main(argv) == 0  # done: kept
    # mutable paths without the attempt's place: refused
    inventory.unlink()
    assert camp.main([a for a in argv if a not in ("--attempt-remote",
                                                    "study5/t/attempts/a1/runs")]) == 1  # fmt: skip


def test_checkpoint_is_archived_verified_then_removed(tmp_path, bucket):
    runs, ckpt = tmp_path / "runs", tmp_path / "ckpt"
    journal_of(
        runs,
        "f",
        [
            {"event": "start", "run": "low/s1", "run_id": "r-a1", "experiment": "study5"},
            {"event": "end", "run": "low/s1", "run_id": "r-a1", "outcome": "done"},
        ],
    )
    (runs / "study5" / "r-a1").mkdir(parents=True)
    ckpt.mkdir()
    (ckpt / "r-a1.pt").write_bytes(b"weights")
    (ckpt / "other.pt").write_bytes(b"other")
    digest = hashlib.sha256(b"weights").hexdigest()
    (runs / "study5" / "r-a1" / "summary.json").write_text(
        json.dumps({"endpoint_checkpoint": {"path": str(ckpt / "r-a1.pt"), "sha256": digest}})
    )
    argv = [
        "checkpoint",
        "--runs",
        str(runs),
        "--campaign",
        "f",
        "--label",
        "low/s1",
        "--dir",
        str(ckpt),
        "--inventory",
        str(tmp_path / "inv" / "ck.tsv"),
        "--remote",
        "study5/t/checkpoints",
        "--inventories-remote",
        "study5/t/inventories",
    ]
    assert camp.main(argv) == 0
    assert not (ckpt / "r-a1.pt").exists() and (ckpt / "other.pt").exists()
    assert (bucket / "study5/t/checkpoints/r-a1.pt").read_bytes() == b"weights"
    assert camp.main(argv) == 0  # nothing left to do


def test_a_checkpoint_kept_after_its_archive_is_restored_after_a_pod_stop(tmp_path, bucket):
    """The group's order: the checkpoint archived and verified right after the η_l run (kept);
    the Pod stops and the container disk is wiped; the scan's step fetches it back from the
    archive, its sha256 checked, then runs; the last checkpoint step removes it. A fetched copy
    of another sha256 is refused and never put in place."""
    runs, ckpt = tmp_path / "runs", tmp_path / "ckpt"
    journal_of(
        runs,
        "f",
        [
            {"event": "start", "run": "low/s1", "run_id": "r-a1", "experiment": "study5"},
            {"event": "end", "run": "low/s1", "run_id": "r-a1", "outcome": "done"},
        ],
    )
    (runs / "study5" / "r-a1").mkdir(parents=True)
    ckpt.mkdir()
    (ckpt / "r-a1.pt").write_bytes(b"weights")
    digest = hashlib.sha256(b"weights").hexdigest()
    (runs / "study5" / "r-a1" / "summary.json").write_text(
        json.dumps({"endpoint_checkpoint": {"path": str(ckpt / "r-a1.pt"), "sha256": digest}})
    )
    inventory = tmp_path / "inv" / "ck.tsv"
    store = ["--remote", "study5/t/checkpoints", "--inventories-remote", "study5/t/inventories"]
    checkpoint = ["checkpoint", "--runs", str(runs), "--campaign", "f", "--label", "low/s1",
                  "--dir", str(ckpt), "--inventory", str(inventory), *store]  # fmt: skip
    assert camp.main([*checkpoint, "--keep"]) == 0
    assert (ckpt / "r-a1.pt").read_bytes() == b"weights"  # kept on the disk
    assert (bucket / "study5/t/checkpoints/r-a1.pt").read_bytes() == b"weights"

    (ckpt / "r-a1.pt").unlink()  # the Pod stopped: the container disk is wiped
    seen = tmp_path / "seen"
    script = tmp_path / "scan.py"
    script.write_text(f"import pathlib, sys; pathlib.Path(r'{seen}').write_bytes("
                      f"pathlib.Path(r'{ckpt}/r-a1.pt').read_bytes())\n")  # fmt: skip
    restore = ["--restore-label", "low/s1", "--restore-dir", str(ckpt), "--restore-remote",
               "study5/t/checkpoints", "--restore-inventory", str(inventory),
               "--restore-inventories-remote", "study5/t/inventories"]  # fmt: skip
    with_runs = ["with-runs", "--runs", str(runs), "--campaign", "f", *restore, "--",
                 str(script)]  # fmt: skip
    assert camp.main(with_runs) == 0
    assert seen.read_bytes() == b"weights" and (ckpt / "r-a1.pt").read_bytes() == b"weights"
    assert camp.main(checkpoint) == 0  # removed after the scan, guarded as before
    assert not (ckpt / "r-a1.pt").exists()

    # an archived object that is not the run's sha256: refused, nothing put in place
    (bucket / "study5/t/checkpoints/r-a1.pt").write_bytes(b"other weights")
    seen.unlink()
    assert camp.main(with_runs) == 1
    assert not (ckpt / "r-a1.pt").exists() and not seen.exists()
    assert not list(ckpt.glob(".*restore"))
    # nothing in the bucket at all
    (bucket / "study5/t/checkpoints/r-a1.pt").unlink()
    assert camp.main(with_runs) == 1 and not seen.exists()


def test_publish_then_check_published(tmp_path, bucket):
    record = tmp_path / "r" / "predictions-record.json"
    camp.write_record(record, predictions_of([50000]))
    receipt = tmp_path / "r" / "predictions-receipt.json"
    check = [
        "check-published",
        "--record",
        str(record),
        "--receipt",
        str(receipt),
        "--gpu-code",
        GPU,
        "--seeds",
        "50000",
    ]
    assert camp.main(check) == 1  # not yet published: no fresh run starts
    argv = [
        "publish",
        "--record",
        str(record),
        "--receipt",
        str(receipt),
        "--remote",
        "study5/t/predictions",
        "--inventory",
        str(tmp_path / "inv" / "p.tsv"),
    ]
    assert camp.main(argv) == 0
    found = json.loads(receipt.read_text())
    assert found["sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    assert found["created"] == "2026-11-09T00:00:00+0000"
    assert "gs://" not in receipt.read_text()
    assert (bucket / "study5/t/predictions/predictions-record.json").is_file()
    assert camp.main(check) == 0
    record.write_text(record.read_text() + " ")
    assert camp.main(check) == 1


def test_fetch_streams_and_a_study3_run_from_the_bucket(tmp_path, bucket):
    rows, data = [], tmp_path / "data"
    for seed in (10403, 10404):
        for rel, body in (
            (f"teacher/M/training/h/seed-{seed}/chunk-00000/samples.jsonl", b"t"),
            (f"teacher/M/preconditioner/g/seed-{seed}/samples.jsonl", b"p"),
        ):
            (bucket / "data" / rel).parent.mkdir(parents=True, exist_ok=True)
            (bucket / "data" / rel).write_bytes(body)
            rows.append(
                f"{rel}\t1\t{hashlib.sha256(body).hexdigest()}\tx\tgs://bkt/proj/data/{rel}\tok\n"
            )
    inventory = bucket / camp.STUDY3_STREAMS_INVENTORY
    inventory.parent.mkdir(parents=True)
    inventory.write_text("path\tsize\tsha256\tmd5_b64\turl\tstatus\n" + "".join(rows))
    argv = [
        "fetch-streams",
        "--inventory",
        camp.STUDY3_STREAMS_INVENTORY,
        "--data",
        str(data),
        "--work",
        str(tmp_path / "work"),
        "--seeds",
        "10403",
    ]
    assert camp.main(argv) == 0
    assert sorted(p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file()) == [
        "teacher/M/preconditioner/g/seed-10403/samples.jsonl",
        "teacher/M/training/h/seed-10403/chunk-00000/samples.jsonl",
    ]
    assert camp.main(argv) == 0  # kept, checked by sha256
    (data / "teacher/M/preconditioner/g/seed-10403/samples.jsonl").write_bytes(b"q")
    assert camp.main(argv) == 1  # never replaced

    run = bucket / camp.STUDY3_RUNS_REMOTE / "study3" / "r-a1"
    for rel in (
        "meta.json",
        "responses/harmful_evaluation.jsonl",
        "judge/wildguard.jsonl",
        "wandb/x.log",
    ):
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_text("{}")
    dest = tmp_path / "fetched"
    assert camp.main(["fetch-run", "--run-id", "r-a1", "--dest", str(dest)]) == 0
    assert sorted(
        p.relative_to(dest / "r-a1").as_posix() for p in (dest / "r-a1").rglob("*") if p.is_file()
    ) == ["meta.json", "responses/harmful_evaluation.jsonl"]


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    out = tmp_path / "c"
    assert (
        camp.main(["pod2", "--out", str(out), "--archive-id", "t", "--n", "24", "--dry-run"]) == 0
    )
    printed = capsys.readouterr().out
    assert "block fresh" in printed and "s5-fresh-1.3e-05-s50024" not in printed
    assert "study5-t-fresh-lower-s50023.yaml" in printed and not out.exists()
    with pytest.raises(SystemExit):
        camp.main(["pod2", "--out", str(out), "--archive-id", "t", "--n", "30", "--dry-run"])


def test_the_labels_a_step_reads_are_its_campaigns(where, tmp_path):
    plan = pod2(where)
    labels = {}
    for name in plan.campaigns:
        labels[name] = {r.label for r in campaign_runs(plan, tmp_path, name)}
    for step in steps(plan):
        if "--campaign" in step.argv:
            name = step.argv[step.argv.index("--campaign") + 1]
            wanted = [a[5:-1] for a in step.argv if a.startswith("{run:")]
            if "--label" in step.argv:
                wanted.append(step.argv[step.argv.index("--label") + 1])
            assert set(wanted) <= labels[name], step.name
    plan1 = pod1(where)
    check = next(s for s in steps(plan1) if s.name == "engineering/check")
    name = check.argv[check.argv.index("--campaign") + 1]
    runs = campaign_runs(plan1, tmp_path, name)
    assert {a[5:-1] for a in check.argv if a.startswith("{run:")} == {r.label for r in runs}


def test_after_a_stop_only_the_begun_groups_and_the_block_are_archived(tmp_path):
    S = camp.Step
    blocks = [
        camp.Block(
            "one",
            (
                S("one/s1/run", mark(tmp_path, "s1"), group="s1"),
                S("one/s1/archive", mark(tmp_path, "s1-archive"), group="s1", kind="archive"),
                S("one/s2/check", mark(tmp_path, "s2", 1), group="s2", on_fail="stop"),
                S("one/s2/archive", mark(tmp_path, "s2-archive"), group="s2", kind="archive"),
                S("one/s3/run", mark(tmp_path, "s3"), group="s3"),
                S("one/s3/archive", mark(tmp_path, "s3-archive"), group="s3", kind="archive"),
                S("one/snapshot", mark(tmp_path, "snapshot"), kind="archive"),
            ),
        ),
        camp.Block("close", ()),
    ]
    assert run_plan(fake_plan(tmp_path, blocks), tmp_path) == camp.STOPPED
    assert marks(tmp_path) == ["s1", "s1-archive", "s2", "s2-archive", "snapshot"]


# ------------------------------------------------------------------ gates, deadlines, timeouts


def predictions_of(seeds, commit=COMMIT, gpu=GPU):
    return {
        "kind": "study5-predictions",
        "n": len(seeds),
        "seeds": list(seeds),
        "predictions": [{"seed": s, "p": 0.0} for s in seeds],
        "code": {"commit": commit, "gpu_code_sha256": gpu},
    }


def writes(tmp_path, name, path, record):
    """A step that marks itself and writes ``record`` at ``path``: a gate run again."""
    return py(
        f"import json, pathlib; pathlib.Path(r'{tmp_path}/marks/{name}').touch(); "
        f"pathlib.Path(r'{path}').write_text(json.dumps({record!r}))"
    )


def gate_plan(tmp_path, step, n=None):
    plan = fake_plan(tmp_path, [camp.Block("one", (step,)), camp.Block("close", ())])
    plan.n = n
    return plan


HEAD = camp.code_identity.gpu_code_identity()
# The commit of HEAD's GPU code's record (extension 2's): an earlier commit of HEAD's GPU code
SAME = json.loads((REPO_ROOT / "docs" / "study5-extension-2-code-identity.json").read_text())[
    "commit"
]
OTHER = "fd54945"  # study 3's qualified GPU code: another identity


def in_history(commit):
    found = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO_ROOT, capture_output=True
    )
    return found.returncode == 0


def engineering_record(equal=True, commit=None, rerun_commit=None, gpu=None, dirty=False):
    commit = commit or HEAD["commit"]
    return {
        "kind": "study5-engineering-check",
        "equal": equal,
        "pairs": [
            {"rerun": "r", "rerun_commit": rerun_commit or HEAD["commit"], "rerun_dirty": dirty}
        ],
        "code": {"commit": commit, "gpu_code_sha256": gpu or HEAD["sha256"]},
        "git": {"commit": commit, "dirty": False},
    }


@pytest.mark.parametrize(
    "case, kept",
    [
        ("good", True),
        ("later-commit-same-gpu", True),  # a commit is compared by its GPU code only
        ("different", False),  # a difference: never kept
        ("record-gpu-disagrees", False),
        ("unknown-commit", False),
        ("reruns-other-gpu", False),
        ("reruns-dirty", False),
    ],
)
def test_an_engineering_record_is_kept_by_its_verdict_and_gpu_code_only(tmp_path, case, kept):
    if case in ("later-commit-same-gpu", "reruns-other-gpu") and not (
        in_history(SAME) and in_history(OTHER)
    ):
        pytest.skip("the commits compared are not in this history")
    record = {
        "good": engineering_record(),
        "later-commit-same-gpu": engineering_record(commit=SAME, rerun_commit=SAME),
        "different": engineering_record(equal=False),
        "record-gpu-disagrees": engineering_record(gpu="f" * 64),
        "unknown-commit": engineering_record(commit="d" * 40),
        "reruns-other-gpu": engineering_record(rerun_commit=OTHER),
        "reruns-dirty": engineering_record(dirty=True),
    }[case]
    path = tmp_path / "check.json"
    path.write_text(json.dumps(record))
    step = camp.Step(
        "one/check",
        writes(tmp_path, "check", path, engineering_record()),
        on_fail="stop",
        outputs=(str(path),),
        gate="engineering",
    )
    plan = gate_plan(tmp_path, step)
    plan.gpu_code = HEAD["sha256"]
    assert run_plan(plan, tmp_path) == camp.COMPLETED
    assert marks(tmp_path) == ([] if kept else ["check"])
    aside = sorted(p.name for p in tmp_path.glob("check.json.aside-*"))
    assert (aside == []) is kept
    if not kept:
        assert json.loads((tmp_path / aside[0]).read_text()) == record  # moved, never deleted


@pytest.mark.parametrize(
    "passes, commit, dirty, kept",
    [
        (True, "head", False, True),
        (True, "same", False, True),  # a later commit of the same GPU code
        (False, "head", False, False),
        (True, "head", True, False),
        (True, "other", False, False),
        (True, "d" * 40, False, False),
    ],
)
def test_a_surrogate_record_is_kept_by_its_verdict_and_its_gpu_code(
    tmp_path, passes, commit, dirty, kept
):
    commit = {"head": HEAD["commit"], "same": SAME, "other": OTHER}.get(commit, commit)
    if commit in (SAME, OTHER) and not in_history(commit):
        pytest.skip("the commit compared is not in this history")
    path = tmp_path / "checks.json"
    path.write_text(
        json.dumps(
            {
                "kind": "study5-surrogate-checks",
                "passes": passes,
                "git": {"commit": commit, "dirty": dirty},
            }
        )
    )
    good = {
        "kind": "study5-surrogate-checks",
        "passes": True,
        "git": {"commit": HEAD["commit"], "dirty": False},
    }
    step = camp.Step(
        "one/checks",
        writes(tmp_path, "checks", path, good),
        on_fail="stop",
        outputs=(str(path),),
        gate="surrogate",
    )
    plan = gate_plan(tmp_path, step)
    plan.gpu_code = HEAD["sha256"]
    assert run_plan(plan, tmp_path) == camp.COMPLETED
    assert marks(tmp_path) == ([] if kept else ["checks"])


def published(tmp_path, record):
    """The record written with its sidecar, and its receipt: published."""
    path = tmp_path / "pub" / "predictions-record.json"
    digest = camp.write_record(path, record)
    receipt = path.with_name(camp.RECEIPT_NAME)
    receipt.write_text(json.dumps({"sha256": digest, "created": "t"}))
    return path, receipt


def check_published(path, receipt, seeds, gpu=GPU):
    return camp.main(
        [
            "check-published",
            "--record",
            str(path),
            "--receipt",
            str(receipt),
            "--gpu-code",
            gpu,
            "--seeds",
            *map(str, seeds),
        ]
    )


def predictions_plan(tmp_path, path, plan_n, gpu=GPU):
    step = camp.Step(
        "one/record",
        mark(tmp_path, "record"),
        on_fail="stop",
        outputs=(str(path),),
        gate="predictions",
    )
    after = camp.Step("one/launch", mark(tmp_path, "launch"), kind="launch")
    plan = fake_plan(tmp_path, [camp.Block("one", (step, after)), camp.Block("close", ())])
    plan.n, plan.gpu_code = plan_n, gpu
    return plan


@pytest.mark.parametrize("record_n, plan_n", [(24, 40), (40, 24), (40, 40)])
def test_a_predictions_record_of_other_seeds_stops_before_any_f_run(tmp_path, record_n, plan_n):
    path, receipt = published(tmp_path, predictions_of(camp.fresh_seeds(record_n)))
    plan = predictions_plan(tmp_path, path, plan_n)
    expected = camp.COMPLETED if record_n == plan_n else camp.STOPPED
    assert run_plan(plan, tmp_path) == expected
    assert marks(tmp_path) == (["launch"] if record_n == plan_n else [])
    # check-published, before the first fresh run, refuses it the same way
    assert check_published(path, receipt, camp.fresh_seeds(plan_n)) == (
        0 if record_n == plan_n else 1
    )


def test_a_predictions_record_is_bound_to_its_gpu_code_and_its_published_sha(tmp_path):
    seeds = camp.fresh_seeds(24)
    # a resume from a later commit of the same GPU code: the record's commit is information
    path, receipt = published(tmp_path, predictions_of(seeds, commit="d" * 40))
    plan = predictions_plan(tmp_path, path, 24)
    plan.commit = "a" * 40
    assert run_plan(plan, tmp_path) == camp.COMPLETED
    assert marks(tmp_path) == ["launch"]
    assert check_published(path, receipt, seeds) == 0
    # another GPU code stops
    shutil.rmtree(tmp_path / "c")
    (tmp_path / "marks" / "launch").unlink()
    assert run_plan(predictions_plan(tmp_path, path, 24, gpu="f" * 64), tmp_path) == camp.STOPPED
    assert marks(tmp_path) == []
    assert check_published(path, receipt, seeds, gpu="f" * 64) == 1
    # a published copy whose sha256 is not the local record's stops
    receipt.write_text(json.dumps({"sha256": "0" * 64, "created": "t"}))
    shutil.rmtree(tmp_path / "c")
    assert run_plan(predictions_plan(tmp_path, path, 24), tmp_path) == camp.STOPPED
    assert marks(tmp_path) == []
    assert check_published(path, receipt, seeds) == 1


def minutes_of(step) -> int:
    return int(step.argv[step.argv.index("--minutes") + 1])


def test_every_planned_step_has_a_timeout_and_every_group_sets_its_deadline_first(where):
    for plan in (pod1(where), pod2(where), pod2(where, 24), topup(where)):
        for block in plan.blocks:
            assert all(isinstance(s.timeout, int) and s.timeout > 0 for s in block.steps)
            if block.name == "close":
                assert block.steps[-1].kind == "deadline"
                continue
            assert block.steps[0].kind == "deadline" and not block.steps[0].group
            window = minutes_of(block.steps[0])
            seen = set()
            for step in block.steps[1:]:
                if step.group and step.group not in seen:
                    seen.add(step.group)
                    assert step.kind == "deadline" and step.group, (block.name, step.name)
                    window = minutes_of(step)
                    members = [s for s in block.steps if s.group == step.group and s is not step]
                    assert window == camp.deadline_minutes(sum(s.minutes for s in members))
                assert step.timeout <= 60 * window, step.name  # a hung step ends in its window
    lower = next(s for s in steps(pod2(where)) if s.name == "fresh/s50000/launch-fresh-lower")
    assert lower.minutes == pytest.approx(6.9 + 0.5)
    assert lower.timeout == camp.timeout_of(lower.minutes)
    scan = next(s for s in steps(pod2(where)) if s.name == "fresh/s50000/scan")
    assert scan.minutes == pytest.approx(15.5 + 2.0)  # five evaluations and the loading
    assert scan.timeout == camp.timeout_of(scan.minutes) == 60 * (3 * 17.5 + 20)


def test_a_step_past_its_timeout_is_killed_and_fails(tmp_path):
    """A step that outlives its timeout: its process is gone afterwards (killed, not waited
    out), the journal records the timeout and the campaign goes on. No wall-clock bound beyond
    a generous one: the child would sleep 600 s."""
    pid_file = tmp_path / "pid"
    child = f"import os, time; open(r'{pid_file}', 'w').write(str(os.getpid())); time.sleep(600)"
    slow = camp.Step("one/slow", py(child), on_fail="record", timeout=10)
    after = camp.Step("one/after", mark(tmp_path, "after"))
    plan = fake_plan(tmp_path, [camp.Block("one", (slow, after)), camp.Block("close", ())])
    started = time.monotonic()
    assert run_plan(plan, tmp_path) == camp.COMPLETED
    assert time.monotonic() - started < 300
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # killed with its process group
    assert camp.read_journal(plan.out / "journal.jsonl")["one/slow"] == "failed"
    assert '"exit": "timeout"' in (plan.out / "journal.jsonl").read_text()
    assert marks(tmp_path) == ["after"]


def test_the_close_deadline_runs_on_an_unexpected_exception(tmp_path, monkeypatch):
    close = camp.Block(
        "close",
        (camp.Step("close/deadline", mark(tmp_path, "deadline"), kind="deadline", always=True),),
    )
    plan = fake_plan(tmp_path, [camp.Block("one", ()), close])

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(camp, "_blocks", broken)
    with pytest.raises(OSError):
        run_plan(plan, tmp_path)
    assert marks(tmp_path) == ["deadline"]


def test_a_launch_whose_runs_completed_is_not_run_again(tmp_path):
    journal_of(
        tmp_path / "r",
        "camp-a",
        [
            {"event": "start", "run": "cell/s1", "run_id": "r-a1", "experiment": "study5"},
            {"event": "end", "run": "cell/s1", "run_id": "r-a1", "outcome": "done"},
        ],
    )
    (tmp_path / "r" / "study5" / "r-a1").mkdir(parents=True)
    done = camp.Step(
        "one/launch-a", mark(tmp_path, "a"), kind="launch", kept_if_runs=("camp-a\tcell/s1",)
    )
    todo = camp.Step(
        "one/launch-b", mark(tmp_path, "b"), kind="launch", kept_if_runs=("camp-b\tcell/s1",)
    )
    plan = fake_plan(tmp_path, [camp.Block("one", (done, todo)), camp.Block("close", ())])
    assert run_plan(plan, tmp_path) == camp.COMPLETED
    assert marks(tmp_path) == ["b"]


def test_the_checkpoint_step_refuses_a_checkpoint_its_run_did_not_record(tmp_path, bucket):
    runs, ckpt = tmp_path / "runs", tmp_path / "ckpt"
    journal_of(
        runs, "f", [{"event": "start", "run": "low/s1", "run_id": "r-a1", "experiment": "study5"}]
    )
    (runs / "study5" / "r-a1").mkdir(parents=True)
    (runs / "study5" / "r-a1" / "summary.json").write_text("{}")
    ckpt.mkdir()
    (ckpt / "r-a1.pt").write_bytes(b"weights")
    argv = [
        "checkpoint",
        "--runs",
        str(runs),
        "--campaign",
        "f",
        "--label",
        "low/s1",
        "--dir",
        str(ckpt),
        "--inventory",
        str(tmp_path / "inv" / "ck.tsv"),
        "--remote",
        "study5/t/checkpoints",
        "--inventories-remote",
        "study5/t/inventories",
    ]
    assert camp.main(argv) == 1
    assert (ckpt / "r-a1.pt").exists() and not (bucket / "study5/t/checkpoints").exists()
    # recorded but neither on the disk nor in the bucket: lost, named
    digest = hashlib.sha256(b"weights").hexdigest()
    (runs / "study5" / "r-a1" / "summary.json").write_text(
        json.dumps({"endpoint_checkpoint": {"sha256": digest}})
    )
    (ckpt / "r-a1.pt").unlink()
    assert camp.main(argv) == 1


def test_the_study5_driver_scripts_are_analysis_scripts_with_no_gpu_use():
    from test_code_identity import gpu_uses

    from zero_field_refusal.code_identity import ANALYSIS_SCRIPTS, is_gpu_path

    for name in ("study5_campaigns.py", "study5_engineering_check.py"):
        assert f"scripts/{name}" in ANALYSIS_SCRIPTS and not is_gpu_path(f"scripts/{name}")
        assert not gpu_uses(REPO_ROOT / "scripts" / name), name


def test_a_gate_that_writes_a_record_that_does_not_stand_fails(tmp_path):
    path = tmp_path / "check.json"
    writes = py(
        f"import json, pathlib; pathlib.Path(r'{path}').write_text(json.dumps("
        f"{engineering_record(rerun_commit='d' * 40)!r}))"
    )
    step = camp.Step("one/check", writes, on_fail="stop", outputs=(str(path),), gate="engineering")
    after = camp.Step("one/after", mark(tmp_path, "after"))
    plan = fake_plan(tmp_path, [camp.Block("one", (step, after)), camp.Block("close", ())])
    assert run_plan(plan, tmp_path) == camp.STOPPED
    assert marks(tmp_path) == []
    assert camp.read_journal(plan.out / "journal.jsonl")["one/check"] == "failed"


def rerun_dir(runs, name, config_hash, commit, dirty=False):
    run = runs / "study3" / name
    run.mkdir(parents=True)
    run.joinpath("meta.json").write_text(
        json.dumps(
            {
                "config_hash": config_hash,
                "status": "completed",
                "git": {"commit": commit, "dirty": dirty},
            }
        )
    )
    return run


def test_engineering_reruns_of_another_gpu_code_are_moved_aside_then_recovered(tmp_path):
    if not in_history(OTHER):
        pytest.skip("study 3's qualified commit is not in this history")
    runs = tmp_path / "runs"
    keep = rerun_dir(runs, "keep-a1", "d1ce7a92e45b", HEAD["commit"])
    old = rerun_dir(runs, "old-a1", "d1ce7a92e45b", OTHER)
    dirty = rerun_dir(runs, "dirty-a1", "d1ce7a92e45b", HEAD["commit"], dirty=True)
    rerun_dir(runs, "other-config-a1", "3b4601aa846f", OTHER)
    stale = camp.stale_reruns(runs, "study3", "d1ce7a92e45b", HEAD["sha256"])
    assert stale == sorted([old, dirty])
    assert keep not in stale
    plain = camp.engineering_launch(Path("e.yaml"), False, HEAD["sha256"])
    assert plain == ["scripts/launch.py", "e.yaml", "--max-attempts", "1"]
    moved = camp.engineering_launch(Path("e.yaml"), True, HEAD["sha256"])
    assert moved[:4] == plain and moved[4] == "--recovery"
    for run_dir in stale:  # as eng-launch moves them: aside, never deleted
        camp.move_aside(runs, run_dir.relative_to(runs).as_posix())
    assert sorted(p.name.rsplit("-a1-", 1)[0] for p in (runs / "aside").iterdir()) == [
        "study3--dirty",
        "study3--old",
    ]
    assert keep.exists()


def test_the_engineering_campaign_hashes_are_the_launchers(where, tmp_path):
    plan = pod1(where)
    step = next(s for s in steps(plan) if s.name == "engineering/launch")
    name = Path(step.argv[step.argv.index("--campaign-file") + 1]).stem
    path = tmp_path / f"{name}.yaml"
    path.write_text(plan.campaigns[name])
    assert camp.campaign_hashes(path) == [
        (r.experiment, r.config_hash) for r in load_campaign(path)[1]
    ]
    assert {h for _, h in camp.campaign_hashes(path)} == STUDY3_F_HASHES


DRIVER = r"""
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("c", sys.argv[1])
camp = importlib.util.module_from_spec(spec); sys.modules["c"] = camp; spec.loader.exec_module(camp)
tmp = Path(sys.argv[2])
slow = camp.Step("one/slow", ("-c", "import time; time.sleep(120)"), timeout=300)
touch = f"import pathlib; pathlib.Path(r'{tmp}/closed').touch()"
close = camp.Step("close/deadline", ("-c", touch), kind="deadline", always=True)
plan = camp.Plan(pod="t", archive_id="t", out=tmp / "c", runs=tmp / "r", data=tmp / "d",
                 checkpoints=tmp / "k", commit="c" * 40, gpu_code="e" * 64, n=None,
                 blocks=[camp.Block("one", (slow,)), camp.Block("close", (close,))])
(tmp / "c").mkdir()
camp.install_sigterm()
camp.execute(plan, python=[sys.executable], cwd=tmp, env={})
"""


def test_sigterm_to_the_driver_runs_the_close(tmp_path):
    script = tmp_path / "driver.py"
    script.write_text(DRIVER)
    process = subprocess.Popen(
        [sys.executable, str(script), str(REPO_ROOT / camp.SELF), str(tmp_path)], env={**os.environ}
    )
    journal = tmp_path / "c" / "journal.jsonl"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not (
        journal.exists() and '"step": "one/slow"' in journal.read_text()
    ):
        time.sleep(0.2)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=60) == camp.STOPPED
    assert (tmp_path / "closed").exists()
    assert camp.read_journal(journal)["one/slow"] == "interrupted"  # a restart runs it again


def test_a_checkpoint_archived_earlier_counts_only_with_its_recorded_sha(tmp_path, bucket):
    runs, ckpt = tmp_path / "runs", tmp_path / "ckpt"
    journal_of(
        runs, "f", [{"event": "start", "run": "low/s1", "run_id": "r-a1", "experiment": "study5"}]
    )
    (runs / "study5" / "r-a1").mkdir(parents=True)
    ckpt.mkdir()
    digest = hashlib.sha256(b"weights").hexdigest()
    (runs / "study5" / "r-a1" / "summary.json").write_text(
        json.dumps({"endpoint_checkpoint": {"sha256": digest}})
    )
    (bucket / "study5/t/checkpoints").mkdir(parents=True)
    (bucket / "study5/t/checkpoints/r-a1.pt").write_bytes(b"weights")
    inventories = bucket / "study5/t/inventories"
    inventories.mkdir(parents=True)
    header = "path\tsize\tsha256\tmd5_b64\turl\tstatus\n"
    argv = [
        "checkpoint",
        "--runs",
        str(runs),
        "--campaign",
        "f",
        "--label",
        "low/s1",
        "--dir",
        str(ckpt),
        "--inventory",
        str(tmp_path / "new-out" / "ck.tsv"),
        "--remote",
        "study5/t/checkpoints",
        "--inventories-remote",
        "study5/t/inventories",
    ]
    # the inventory's bucket copy is named by the run: one checkpoint, written once
    copy = inventories / "checkpoint-r-a1.tsv"
    copy.write_text(header + f"r-a1.pt\t7\t{'0' * 64}\tx\tu\tok\n")
    assert camp.main(argv) == 1  # the archive shows another sha256
    copy.write_text(header + f"r-a1.pt\t7\t{digest}\tx\tu\tok\n")
    for p in (tmp_path / "new-out").rglob("*"):
        p.is_file() and p.unlink()
    assert camp.main(argv) == 0


# ------------------------------------------------------------------ the committed seeds


def seeds_repo(tmp_path, chosen: dict, named: bool = True) -> Path:
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    data = (json.dumps(chosen) + "\n").encode()
    (repo / camp.SEEDS_FILE).write_bytes(data)
    digest = hashlib.sha256(data).hexdigest() if named else "0" * 64
    (repo / camp.REGISTRATION).write_text(f"Status: `registered`.\nSeeds file: {digest}\n")
    return repo


@pytest.mark.parametrize(
    "chosen, n, named, problem",
    [
        ({"n": 40, "seeds": list(range(50000, 50040))}, 40, True, None),
        ({"n": 24, "seeds": list(range(50000, 50024))}, 24, True, None),
        ({"n": 24, "seeds": list(range(50000, 50024))}, 40, True, "not the committed choice"),
        ({"n": 40, "seeds": list(range(50000, 50040))}, 24, True, "not the committed choice"),
        ({"n": 40, "seeds": list(range(50000, 50040))}, 40, False, "does not name the sha256"),
        ({"n": 40, "seeds": list(range(50001, 50041))}, 40, True, "is not n = 40"),
        ({"n": 30, "seeds": list(range(50000, 50030))}, 40, True, "is not n = 40"),
    ],
)  # fmt: skip
def test_pod2s_seeds_are_the_committed_choice_named_by_the_registration(
    where, tmp_path, chosen, n, named, problem
):
    """Every pod 2 execution (a restart into a new --out too) checks the n chosen before
    generation: docs/study5-confirmatory-seeds.json at the plan's commit, its sha256 named in the
    registration there, the plan's fresh seeds exactly its seeds."""
    found = camp.seeds_problems(pod2(where, n), seeds_repo(tmp_path, chosen, named))
    if problem is None:
        assert found == []
    else:
        assert any(problem in f for f in found), found
    assert any("does not read" in f for f in camp.seeds_problems(pod2(where), tmp_path / "none"))


def test_the_committed_seeds_file_is_the_default_choice():
    chosen = json.loads((REPO_ROOT / camp.SEEDS_FILE).read_text())
    assert chosen == {"n": 40, "seeds": list(range(50000, 50040))}
    assert camp.committed_n() == 40
    assert set(camp.CONTROL_SEEDS) <= set(chosen["seeds"][:24])  # in both choices
    args = camp.parser_of().parse_args(["pod2", "--out", "o", "--archive-id", "t2"])
    assert args.n is None  # pod 2's n is the committed one unless given (and then checked)


# ------------------------------------------------------------------ the fresh runs' completion


def test_the_completion_record_lists_every_started_fresh_run_with_its_end(tmp_path, where):
    plan = pod2(where, 24)
    fresh = steps(plan, "fresh")
    names = [s.name for s in fresh]
    step = fresh[names.index("fresh/completion")]
    # after the last fresh group, before the block's snapshot; every fresh launcher campaign
    assert names.index("fresh/completion") > names.index("fresh/s50023/archive")
    assert names[names.index("fresh/completion") + 1] == "fresh/completion-archive"
    campaigns = [step.argv[i + 1] for i, a in enumerate(step.argv) if a == "--campaign"]
    launched = {Path(s.argv[1]).stem for s in fresh if s.kind == "launch"}
    assert set(campaigns) == launched and len(campaigns) == 48
    assert step.gate == "completion" and step.on_fail == "record"
    archive = fresh[names.index("fresh/completion-archive")]
    assert archive.argv[archive.argv.index("--mutable-path") + 1] == (
        "study5_records/t2/fresh-completion.json")  # fmt: skip

    runs, out = tmp_path / "runs", tmp_path / "rec" / "completion.json"
    journal_of(runs, "lo", [{"event": "start", "run": "lo/s1", "run_id": "a", "time": "t1"},
                            {"event": "end", "run": "lo/s1", "run_id": "a", "outcome": "done",
                             "time": "t2"}])  # fmt: skip
    journal_of(runs, "hi", [{"event": "start", "run": "hi/s1", "run_id": "b", "time": "t3"}])
    argv = ["completion", "--runs", str(runs), "--out", str(out), "--campaign", "lo",
            "--campaign", "hi"]  # fmt: skip
    assert camp.main(argv) == 1 and not out.exists()  # a started run without an end: no record
    journal_of(runs, "hi", [{"event": "start", "run": "hi/s1", "run_id": "b", "time": "t3"},
                            {"event": "end", "run": "hi/s1", "run_id": "b", "outcome": "diverged",
                             "time": "t4"}])  # fmt: skip
    assert camp.main(argv) == 0
    record = json.loads(out.read_text())
    assert record["kind"] == "study5-fresh-completion" and record["started"] == 2
    assert [(r["campaign"], r["run"], r["run_id"], r["outcome"]) for r in record["runs"]] == [
        ("lo", "lo/s1", "a", "done"), ("hi", "hi/s1", "b", "diverged")]  # fmt: skip
    assert camp.main(argv) == 1  # written once
    # kept while the journals say the same; a recovery round since: set aside, written again
    gate = fake_plan(tmp_path, [])
    gate.runs = runs
    assert camp.gate_problem("completion", out, gate) is None
    journal_of(runs, "hi", [{"event": "start", "run": "hi/s1", "run_id": "b", "time": "t3"},
                            {"event": "end", "run": "hi/s1", "run_id": "b", "outcome": "diverged",
                             "time": "t4"},
                            {"event": "start", "run": "hi/s1", "run_id": "c", "time": "t5"},
                            {"event": "end", "run": "hi/s1", "run_id": "c", "outcome": "done",
                             "time": "t6"}])  # fmt: skip
    assert "changed" in camp.gate_problem("completion", out, gate)


# ------------------------------------------------------------------ the close's export


def test_the_close_of_pod2_exports_every_fresh_checkpoint_before_its_deadline(where):
    plan = pod2(where, 24)
    close = steps(plan, "close")
    names = [s.name for s in close]
    assert names == ["close/export-deadline", "close/checkpoints", "close/archive",
                     "close/deadline"]  # fmt: skip
    assert all(s.always for s in close) and close[0].kind == "run"  # in order, before the export
    export = close[1]
    runs = [export.argv[i + 1] for i, a in enumerate(export.argv) if a == "--run"]
    lowers = [s for s in steps(plan, "fresh") if s.name.endswith("/launch-fresh-lower")]
    assert runs == [f"{Path(s.argv[1]).stem}={s.kept_if_runs[0].split(chr(9))[1]}"
                    for s in lowers]  # fmt: skip
    assert export.on_fail == "record"
    assert "close" in close[2].argv  # its record goes with the plan and the receipts
    assert [s.name for s in steps(pod1(where), "close")] == ["close/archive", "close/deadline"]


def test_the_close_exports_an_unexported_checkpoint_and_reports_a_failure(tmp_path, bucket,
                                                                          monkeypatch):  # fmt: skip
    """A planned shutdown with one endpoint checkpoint on the disk and not in the archive: the
    close uploads it, verifies its sha256 against the run's, and records it; with the upload
    failing, the failure is in the record and printed with the path, the sha256 and the error,
    and the close goes on."""
    runs, ckpt, inventories = tmp_path / "runs", tmp_path / "ckpt", tmp_path / "c" / "inventories"
    digest = hashlib.sha256(b"weights").hexdigest()
    for run_id, campaign in (("r-a1", "f1"), ("r-b1", "f2")):
        journal_of(runs, campaign, [
            {"event": "start", "run": "low/s1", "run_id": run_id, "experiment": "study5"},
            {"event": "end", "run": "low/s1", "run_id": run_id, "outcome": "done"}])  # fmt: skip
        (runs / "study5" / run_id).mkdir(parents=True)
        (runs / "study5" / run_id / "summary.json").write_text(
            json.dumps({"endpoint_checkpoint": {"sha256": digest}}))  # fmt: skip
    ckpt.mkdir()
    (ckpt / "r-a1.pt").write_bytes(b"weights")
    # r-b1's checkpoint: archived and verified earlier, gone from the disk
    (bucket / "study5/t/checkpoints").mkdir(parents=True)
    (bucket / "study5/t/checkpoints/r-b1.pt").write_bytes(b"weights")
    (bucket / "study5/t/inventories").mkdir(parents=True)
    (bucket / "study5/t/inventories/checkpoint-r-b1.tsv").write_text(
        "path\tsize\tsha256\tmd5_b64\turl\tstatus\n" + f"r-b1.pt\t7\t{digest}\tx\tu\tok\n"
    )

    def close(record, *extra):
        return camp.main(["export-checkpoints", "--runs", str(runs), "--dir", str(ckpt),
                          "--remote", "study5/t/checkpoints", "--inventories-remote",
                          "study5/t/inventories", "--inventories", str(inventories),
                          "--record", str(record), "--run", "f1=low/s1", "--run", "f2=low/s1",
                          *extra])  # fmt: skip

    monkeypatch.setenv("FAKE_GCLOUD_FAIL_UPLOAD", "1")
    record = tmp_path / "c" / "close" / "checkpoints-1.json"
    assert close(record) == 1
    found = json.loads(record.read_text())
    failed = [c for c in found["checkpoints"] if c["status"] == "failed"]
    assert found["failed"] == 1 and failed[0]["run_id"] == "r-a1"
    assert failed[0]["path"] == str(ckpt / "r-a1.pt") and failed[0]["sha256"] == digest
    assert failed[0]["error"]
    assert {c["run_id"]: c["status"] for c in found["checkpoints"]}["r-b1"] == "verified"
    assert not (bucket / "study5/t/checkpoints/r-a1.pt").exists()

    monkeypatch.delenv("FAKE_GCLOUD_FAIL_UPLOAD")
    record = tmp_path / "c" / "close" / "checkpoints-2.json"
    assert close(record) == 0
    found = json.loads(record.read_text())
    assert {c["run_id"]: c["status"] for c in found["checkpoints"]} == {
        "r-a1": "exported", "r-b1": "verified"}  # fmt: skip
    assert (bucket / "study5/t/checkpoints/r-a1.pt").read_bytes() == b"weights"
    copy = camp.read_inventory(bucket / "study5/t/inventories/checkpoint-r-a1.tsv")
    assert [(r["path"], r["sha256"], r["status"]) for r in copy] == [("r-a1.pt", digest, "ok")]
    assert (ckpt / "r-a1.pt").exists()  # exported, never removed by the close
    # a file on the disk that is not the run's: not exported, reported
    (ckpt / "r-a1.pt").write_bytes(b"other")
    (bucket / "study5/t/checkpoints/r-a1.pt").unlink()
    for path in inventories.glob("*"):
        path.is_file() and path.unlink()
    record = tmp_path / "c" / "close" / "checkpoints-3.json"
    assert close(record) == 1
    failed = [c for c in json.loads(record.read_text())["checkpoints"] if c["status"] == "failed"]
    assert "not the one its run recorded" in failed[0]["error"]


# ------------------------------------------------------------------ attempts


LAUNCH = r"""
import json, pathlib, sys
runs, attempt = pathlib.Path(sys.argv[1]), int(sys.argv[2])
journal = runs / "launch" / "study5-t-one.jsonl"
journal.parent.mkdir(parents=True, exist_ok=True)
(runs / "launch" / "study5-t-one").mkdir(exist_ok=True)
(runs / "launch" / "study5-t-one" / f"log-{attempt}.txt").write_text(f"attempt {attempt}\n")
def run(run_id, label, end=None, text="x"):
    d = runs / "study5" / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(text)
    lines = [{"event": "start", "run": label, "run_id": run_id, "experiment": "study5"}]
    if end:
        lines.append({"event": "end", "run": label, "run_id": run_id, "outcome": end})
    with journal.open("a") as out:
        out.write("".join(json.dumps(x) + "\n" for x in lines))
records = runs / "study5_records" / "t"
records.mkdir(parents=True, exist_ok=True)
scan = runs / "study5_scan" / "s5-t-scan"
if attempt == 1:
    run("r-1", "cell/s1", "done")
    run("r-2", "cell/s2", None, text="cut off")   # the Pod stopped it: no end
    (records / "gate.json").write_text("first verdict")
    scan.mkdir(parents=True)
    (scan / "line_scan.jsonl").write_text("partial")  # cut off: no record
    (scan / "partial.tmp").write_text("a file the completed rerun does not have")
else:
    with journal.open("a") as out:   # the launcher settles r-2, its recovery round runs r-3
        out.write(json.dumps({"event": "end", "run": "cell/s2", "run_id": "r-2",
                              "outcome": "interrupted"}) + "\n")
    run("r-3", "cell/s2", "done")
    (records / "gate.json").write_text("second verdict")  # the gate ran again
    scan.rename(runs / "aside-scan")                       # moved aside, run again
    scan.mkdir(parents=True)
    (scan / "line_scan.jsonl").write_text("whole")
    (records / "scan.json").write_text("{}")
"""


def attempt_plan(tmp_path, out: Path, attempt: int):
    """One block as pod 2 builds it (a launch, its archive with a gate's record and a scan run,
    the snapshot and the receipt) and its close; the deadlines left out."""
    b = camp.Builder("pod2", "t", out, tmp_path / "runs", tmp_path / "data", tmp_path / "ckpt")
    script = tmp_path / "launch.py"
    script.write_text(LAUNCH)
    b.begin()
    b.step("one/launch", (str(script), str(b.runs), str(attempt)), kind="launch", on_fail="stop")
    b.runs_archive("one/archive", paths=("study5_records/t/scan.json",),
                   mutable=("study5_records/t/gate.json",),
                   run_dirs=(("study5_scan/s5-t-scan", "study5_records/t/scan.json"),),
                   campaign="study5-t-one")  # fmt: skip
    b.end("one")
    b.close()
    plan = b.plan(COMMIT, GPU, None, (camp.SURROGATE, "s" * 64))
    plan.blocks = [camp.Block(x.name, tuple(s for s in x.steps if s.kind != "deadline"))
                   for x in plan.blocks]  # fmt: skip
    return plan


def objects(store: Path) -> dict[str, bytes]:
    return {p.relative_to(store).as_posix(): p.read_bytes() for p in store.rglob("*")
            if p.is_file()}  # fmt: skip


def test_the_documented_restart_against_an_archive_an_earlier_attempt_wrote(tmp_path, bucket):
    """A stop, then the restart of docs/operations.md (a new --out, the same archive id) against
    the archive the first attempt populated, archive.py --no-clobber throughout: no mismatch, and
    every object the first attempt wrote is byte-identical afterwards. The launcher journal and
    its logs, the gate's record rewritten, the scan run cut off then run again, the snapshots, the
    plan and the inventories all change between the attempts: each attempt's go under its own id;
    the ended runs' directories and the records written once stay at their stable paths."""
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    first = attempt_plan(tmp_path, tmp_path / "out-1", 1)
    camp.prepare(first)  # the plan and its campaigns into --out, as the driver does
    assert camp.execute(first, python=[sys.executable], cwd=REPO_ROOT, env=env) == camp.COMPLETED
    assert (first.out / "receipts" / "one.json").is_file()
    before = objects(bucket)
    attempts = {k.split("/")[3] for k in before if k.startswith("study5/t/attempts/")}
    assert len(attempts) == 1
    assert "study5/t/runs/study5/r-1/meta.json" in before  # ended: written once, stable
    assert not any(k.startswith("study5/t/runs/study5/r-2") for k in before)  # no end yet

    second = attempt_plan(tmp_path, tmp_path / "out-2", 2)
    camp.prepare(second)  # the plan and its campaigns into --out, as the driver does
    assert camp.execute(second, python=[sys.executable], cwd=REPO_ROOT, env=env) == camp.COMPLETED
    status = camp.read_journal(second.out / "journal.jsonl")
    assert status["one/archive"] == "done" and status["close/archive"] == "done", status
    after = objects(bucket)
    assert {k: after.get(k) for k in before} == before  # the earlier attempt's objects unchanged
    new = {k.split("/")[3] for k in after if k.startswith("study5/t/attempts/")} - attempts
    assert len(new) == 1
    (second_id,) = new
    for rel in ("runs/launch/study5-t-one.jsonl", "runs/study5_records/t/gate.json",
                "campaign/snapshots/one/journal.jsonl", "campaign/plan.json"):  # fmt: skip
        assert f"study5/t/attempts/{second_id}/{rel}" in after, rel
    assert after["study5/t/runs/study5_scan/s5-t-scan/line_scan.jsonl"] == b"whole"
    assert after["study5/t/runs/study5/r-3/meta.json"] == b"x"
    assert after["study5/t/runs/study5/r-2/meta.json"] == b"cut off"  # ended now: stable
    for inventory in (second.out / "inventories").glob("*.tsv"):
        assert all(r["status"] == "ok" for r in camp.read_inventory(inventory)), inventory.name
    drivers = [json.loads(line) for out in (first.out, second.out)
               for line in (out / "journal.jsonl").read_text().splitlines()
               if json.loads(line).get("event") == "driver"]  # fmt: skip
    assert [d["attempt"] for d in drivers] == [*attempts, second_id]

    # the documented restore (restore-runs) into an empty root: every attempt's files in
    # their order, then each completed stable run directory whole; it equals the final records
    restored = tmp_path / "restored"
    assert camp.main(["restore-runs", "--archive-id", "t", "--dest", str(restored)]) == 0
    runs = tmp_path / "runs"
    found = {p.relative_to(restored).as_posix() for p in restored.rglob("*") if p.is_file()}
    assert found and not any(".partial" in rel for rel in found)
    for rel in found:
        assert (restored / rel).read_bytes() == (runs / rel).read_bytes(), rel
    scan = "study5_scan/s5-t-scan"
    assert sorted(p.name for p in (restored / scan).iterdir()) == ["line_scan.jsonl"]
    assert (restored / scan / "line_scan.jsonl").read_bytes() == b"whole"  # not the partial
    for rel in (
        "launch/study5-t-one.jsonl",
        "study5_records/t/gate.json",
        "study5_records/t/scan.json",
        "study5/r-1/meta.json",
        "study5/r-3/meta.json",
    ):
        assert rel in found, rel  # fmt: skip
    # the order matters: the partial scan (an earlier attempt's) restored last would survive
    assert "attempts/" in "".join(after) and any(
        k.endswith("study5_scan/s5-t-scan/partial.tmp") for k in after
    )


def test_an_attempt_id_is_the_utc_time_and_a_random_suffix_and_fills_the_argv():
    one, two = camp.new_attempt(), camp.new_attempt()
    assert one != two and len(one.split("-")) == 2 and one.split("-")[0].endswith("z")
    assert camp.attempt_argv(("--remote", "study5/x/attempts/{attempt}/runs"), one) == [
        "--remote", f"study5/x/attempts/{one}/runs"]  # fmt: skip


def test_a_ladder_rungs_minutes_cover_its_matched_k_step(where):
    rung = next(s for s in steps(pod2(where)) if s.name == "controls/s50000/ladder-linear")
    assert rung.minutes == pytest.approx(5 * camp.K_POINT + camp.EVALUATION) == 10.6
    assert rung.timeout == camp.timeout_of(10.6)
    rung = next(s for s in steps(pod3(where)) if s.name == "ladder/s50000/gaussian_d")
    # extension 1, at most: 3 grid points, 8 extensions, c₁ and c₂, then one evaluation
    assert rung.minutes == pytest.approx(13 * camp.K_POINT + camp.EVALUATION) == 22.6


# ------------------------------------------------------------------ the deadlines and the cap


def deadline_env(monkeypatch, tmp_path, cap):
    target = tmp_path / "zfr-deadline"
    monkeypatch.setenv("ZFR_DEADLINE_FILE", str(target))
    monkeypatch.setenv("ZFR_DEADLINE_EPOCH", str(cap))
    return target


def test_pod2_admits_a_group_only_while_its_window_and_the_close_fit_before_the_cap(
    where, tmp_path, monkeypatch
):
    """Every block and seed group of pod 2 reserves the close's export window (76 minutes)
    against the hard cap ZFR_DEADLINE_EPOCH: a group whose window and the close's do not both
    fit is not admitted (the step fails: a stop, the close follows); the cap is never raised."""
    assert camp.EXPORT_WINDOW == 76
    plan = pod2(where, 24)
    deadlines = [s for b in plan.blocks if b.name != "close" for s in b.steps
                 if s.kind == "deadline"]  # fmt: skip
    assert deadlines and all(s.argv[s.argv.index("--reserve") + 1] == "76" for s in deadlines)
    assert all(s.on_fail == "stop" for s in deadlines)
    assert not any("--reserve" in s.argv for s in steps(pod1(where)))
    group = next(s for s in deadlines if s.group == "s50003" and s.name.startswith("fresh/"))
    window = minutes_of(group)
    now = int(time.time())
    target = deadline_env(monkeypatch, tmp_path, now + 60 * (window + 76) - 120)
    record = tmp_path / "rec.json"
    assert camp.main([*group.argv[1:], "--record", str(record)]) == camp.NOT_ADMITTED
    assert not target.exists()  # nothing written, the cap stands
    found = json.loads(record.read_text())
    assert found["admitted"] is False and found["effective"] is None
    deadline_env(monkeypatch, tmp_path, now + 60 * (window + 76) + 600)
    assert camp.main([*group.argv[1:], "--record", str(record)]) == 0
    written = int(target.read_text())
    assert abs(written - (now + 60 * window)) <= 5 and json.loads(record.read_text())["admitted"]


def test_the_close_deadline_is_never_above_the_cap_and_a_short_window_is_said(
    where, tmp_path, monkeypatch
):
    close = {s.name: s for s in steps(pod2(where, 24), "close")}
    export = close["close/export-deadline"]
    assert "--reserve" not in export.argv and minutes_of(export) == 76
    assert export.argv[export.argv.index("--needs") + 1] == "8.0"
    now = int(time.time())
    cap = now + 30 * 60  # 30 minutes left: the request (76) exceeds it
    target = deadline_env(monkeypatch, tmp_path, cap)
    record = tmp_path / "export-deadline.json"
    argv = [a.replace("{attempt}", "a1") for a in export.argv[1:]]
    argv[argv.index("--record") + 1] = str(record)
    assert camp.main(argv) == 0
    found = json.loads(record.read_text())
    assert int(target.read_text()) == cap  # min(requested, cap)
    assert found["effective"] == found["cap"] == camp._iso(cap)
    assert found["requested"] > found["cap"] and found["short"] is False  # 30 ≥ 8 minutes
    deadline_env(monkeypatch, tmp_path, now + 5 * 60)  # 5 minutes left: shorter than the export
    assert camp.main(argv) == 0
    found = json.loads(record.read_text())
    assert found["short"] is True and found["effective_minutes"] < 8
    final = close["close/deadline"]
    assert "--record" in final.argv and "--reserve" not in final.argv


def test_with_runs_records_an_absent_input_and_keeps_an_incomplete_folder(tmp_path):
    record = tmp_path / "rec" / "points.json"
    argv = ["with-runs", "--runs", str(tmp_path), "--campaign", "c", "--only-if",
            str(tmp_path / "delta_lin.pt"), "--else-record", str(record), "--",
            "scripts/does_not_exist.py"]  # fmt: skip
    assert camp.main(argv) == 0  # the command never runs
    assert json.loads(record.read_text()) == {
        "available": False, "missing": "delta_lin.pt",
        "reason": "its producer did not write it (its record says why)"}  # fmt: skip
    folder = tmp_path / "vec" / "s1"
    folder.mkdir(parents=True)
    (folder / "delta_hat.pt").write_bytes(b"partial")
    argv = ["with-runs", "--runs", str(tmp_path), "--campaign", "c", "--aside-dir", str(folder),
            "--", "scripts/does_not_exist.py"]  # fmt: skip
    camp.main(argv)  # the command fails; the incomplete folder was moved aside first
    kept = [p for p in (tmp_path / "vec").iterdir() if p.name.startswith("s1.aside-")]
    assert len(kept) == 1 and (kept[0] / "delta_hat.pt").read_bytes() == b"partial"
    assert not folder.exists()


def test_json_snapshot_copies_records_only_with_their_paths(tmp_path):
    src, dest = tmp_path / "vec", tmp_path / "records" / "vec-json" / "a1"
    (src / "linear" / "s1").mkdir(parents=True)
    (src / "linear" / "s1" / "record.json").write_text('{"q": 1}')
    (src / "linear" / "s1" / "delta_hat.pt").write_bytes(b"tensor")
    (src / "linear" / "s1" / "delta_hat.pt.json").write_text("{}")
    assert camp.main(["json-snapshot", "--src", str(src), "--dest", str(dest)]) == 0
    found = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert found == ["linear/s1/delta_hat.pt.json", "linear/s1/record.json"]
    assert camp.main(["json-snapshot", "--src", str(tmp_path / "none"), "--dest", str(dest)]) == 0


def test_pod4_k_readouts_only_where_k_is_declared_and_the_m1_purge_names_force(where):
    plan = pod4(where)
    replays_ = [s for s in steps(plan) if "scripts/study5_linear_replay.py" in s.argv
                and "--grad-r-only" not in s.argv]  # fmt: skip
    for s in replays_:
        own_order = s.name.startswith("linear/g")
        assert ("--no-k" in s.argv) is not own_order, s.name
    sealed = [s for s in replays_ if s.name.startswith("sealed/")]
    assert len(sealed) == 20 and all("--no-k" in s.argv for s in sealed)
    force_archive = next(s for s in steps(plan) if s.name == "force/archive")
    purge_m1 = next(s for s in steps(plan) if s.name == "points/global/purge-m1")
    assert force_archive.outputs[0] in purge_m1.argv
    close = [s.name for s in steps(plan, "close")]
    assert close.index("close/json-snapshot") < close.index("close/records")


# ------------------------------------------------------------------ the top-up of E2's orders

TOPUP_ID = "topup-20261113"
E2_SEEDS = tuple(range(50006, 50016))
# How the E2 test finds a seed's order runs and its predictions, copied from
# zero_field_refusal.analysis.study5_e2_test (``campaign_of``: the seed's one launcher journal;
# ``predictions_of``: ``*/record.json`` under the sealed folder) and
# zero_field_refusal.analysis.study5_ext1 (``order_parts``: the cell's label, its config of
# train.order_seed = s + offset and study5.role control); the analysis is not imported here
E2_JOURNAL = "study5-*-orders-s{seed}.jsonl"
E2_CELL = "s5-{order}-{eta!r}-s{seed}/s{seed}"
E2_ORDERS = (("order1", 1000), ("order2", 2000))
E2_RATES = (1.3e-05, 1.83e-05)
E2_PREDICTION = "*/record.json"


def topup(where, archive_id=TOPUP_ID):
    return camp.topup_plan(archive_id=archive_id, commit=COMMIT, gpu_code=GPU, **where)


def replay_steps(plan, block):
    return [s for s in steps(plan, block) if "scripts/study5_linear_replay.py" in s.argv]


def test_topup_repeats_e2s_design_on_the_thirty_seeds_outside_it(where, tmp_path):
    """Every fresh seed outside E2, in order; per seed one launcher campaign of four F runs built
    as pod 3 builds E2's (two new orders at both rates, role control); no checkpoint, no
    labeling."""
    plan = topup(where)
    assert [b.name for b in plan.blocks] == ["engineering", "fetch", "predictions", "orders",
                                             "close"]  # fmt: skip
    assert (*range(50000, 50006), *range(50016, 50040)) == camp.TOPUP_SEEDS
    assert not set(camp.TOPUP_SEEDS) & set(camp.ORDER_SEEDS)
    assert sorted({*camp.TOPUP_SEEDS, *camp.ORDER_SEEDS}) == list(camp.EXT2_SEEDS)
    orders = [s for s in steps(plan, "orders") if s.kind == "launch"]
    assert [s.group for s in orders] == [f"s{s}" for s in camp.TOPUP_SEEDS]
    hashes = set()
    for s in orders:
        name, seed = Path(s.argv[1]).stem, int(s.group[1:])
        assert name == f"study5-{TOPUP_ID}-orders-s{seed}"
        found = []
        for r in campaign_runs(plan, tmp_path, name):
            cfg = load_run_config(r.config, list(r.overrides))
            assert (
                r.config == camp.CONTROL_ORDER and cfg.study5.role == "control" and r.seed == seed
            )
            found.append((cfg.train.order_seed - seed, cfg.optimizer.lr))
            hashes.add(config_hash(cfg))
        assert sorted(found) == [(1000, 1.3e-05), (1000, 1.83e-05), (2000, 1.3e-05),
                                 (2000, 1.83e-05)]  # fmt: skip
    assert len(hashes) == 120
    # pod 3's E2 group of 50006, renamed, is the top-up's group of 50016
    three = camp.pod3_plan(archive_id="pod3-20261111", commit=COMMIT, gpu_code=GPU, **where)

    def renamed(text: str) -> str:
        text = text.replace("pod3-20261111", TOPUP_ID).replace("pod3", "topup")
        for offset in (0, 1000, 2000):
            text = text.replace(str(50006 + offset), str(50016 + offset))
        return text

    group = [s for s in steps(three, "orders") if s.group == "s50006"]
    mine = [s for s in steps(plan, "orders") if s.group == "s50016"]
    assert [renamed(json.dumps(camp.asdict(s))) for s in group] == [
        json.dumps(camp.asdict(s)) for s in mine]  # fmt: skip
    theirs = three.campaigns["study5-pod3-20261111-orders-s50006"]
    assert OmegaConf.create(renamed(theirs)) == OmegaConf.create(
        plan.campaigns[f"study5-{TOPUP_ID}-orders-s50016"]
    )  # the cells and the header
    text = " ".join(" ".join(s.argv) for s in steps(plan))
    assert "scripts/label_runs.py" not in text and "linear_shuffled_d" not in text
    commands = {s.argv[1] for s in steps(plan) if s.argv[0] == "scripts/study5_campaigns.py"}
    assert not commands & {"checkpoint", "check-checkpoints", "export-checkpoints",
                           "drop-checkpoints"}  # fmt: skip
    assert not any("endpoint_checkpoint" in text for text in plan.campaigns.values())


def test_topup_names_are_those_the_e2_test_reads(where, tmp_path):
    """The E2 test, given the top-up's seeds, finds each seed's one journal and its four cells
    by pod 3's names, and each order's prediction as ``*/record.json`` of the sealed folder; the
    prediction's config is that of the order's η_l run (the same config hash)."""
    plan = topup(where)
    journals = [f"{name}.jsonl" for name in plan.campaigns]
    eta_l = {}
    for seed in (*camp.TOPUP_SEEDS, *E2_SEEDS):
        found = [j for j in journals if fnmatch.fnmatch(j, E2_JOURNAL.format(seed=seed))]
        if seed in E2_SEEDS:
            assert found == [], seed
            continue
        (journal,) = found
        expected = {E2_CELL.format(order=order, eta=eta, seed=seed): (seed + offset, eta)
                    for order, offset in E2_ORDERS for eta in E2_RATES}  # fmt: skip
        runs = campaign_runs(plan, tmp_path, journal.removesuffix(".jsonl"))
        assert {r.label for r in runs} == set(expected)
        for r in runs:
            cfg = load_run_config(r.config, [*r.overrides, f"seed={r.seed}"])
            assert (cfg.train.order_seed, cfg.optimizer.lr) == expected[r.label]
            assert cfg.study5.role == "control" and cfg.seed == seed
            if cfg.optimizer.lr == 1.3e-05:
                eta_l[(seed, cfg.train.order_seed)] = r.config_hash
    sealed = where["runs"] / "study5_records" / TOPUP_ID / "sealed"
    settles = {s.argv[s.argv.index("--dir") + 1]: Path(s.argv[s.argv.index("--records") + 1])
               for s in steps(plan, "predictions") if s.name.endswith("-settle")}  # fmt: skip
    assert sorted(settles.values()) == sorted(
        sealed / f"s{s}-order{k}" for s in camp.TOPUP_SEEDS for k in (1, 2))  # fmt: skip
    assert all(fnmatch.fnmatch(f"{t.name}/record.json", E2_PREDICTION) for t in settles.values())
    replays = replay_steps(plan, "predictions")
    assert len(replays) == 60
    for s in replays:
        argv = list(s.argv)
        folder = argv[argv.index("--out-dir") + 1]
        assert s.outputs == (f"{folder}/record.json",) and folder in settles  # settled from there
        script = argv.index("scripts/study5_linear_replay.py")
        config, overrides = argv[script + 1], argv[script + 2 : argv.index("--out-dir")]
        cfg = load_run_config(config, overrides)
        assert config_hash(cfg) == eta_l.pop((cfg.seed, cfg.train.order_seed))
    assert not eta_l  # every order has its prediction


def test_topup_predicts_every_order_before_any_order_run_as_pod4_sealed(where):
    """The 60 orders' δ̂, δ_lin and q from θ₀ with pod 4's sealed options (--no-k, pod 2's fresh
    η_l run as the reference); a failed replay or settle stops the campaign, each seed's records
    are archived before its vector files are purged, and the whole block ends before the first
    order run."""
    plan = topup(where)
    names = [s.name for s in steps(plan)]
    predictions = steps(plan, "predictions")
    first_run = min(names.index(s.name) for s in steps(plan) if s.kind == "launch"
                    and s.name.startswith("orders/"))  # fmt: skip
    assert max(names.index(s.name) for s in predictions) < first_run
    replays = replay_steps(plan, "predictions")
    for s in predictions:
        if s in replays or s.name.endswith("-settle") or s.kind in ("archive", "receipt"):
            assert s.on_fail == "stop", s.name
    vec = where["checkpoints"] / "vec" / TOPUP_ID / "sealed"
    for seed in camp.TOPUP_SEEDS:
        group = [s for s in predictions if s.group == f"s{seed}"]
        (archive,) = [s for s in group if s.name.endswith("/archive")]
        paths = [archive.argv[i + 1] for i, a in enumerate(archive.argv) if a == "--path"]
        assert paths == [f"study5_records/{TOPUP_ID}/sealed/s{seed}-order{k}" for k in (1, 2)]
        purges = [s for s in group if s.name.endswith("-purge")]
        assert len(purges) == 2 and all(archive.outputs[0] in p.argv for p in purges)
        assert all(names.index(archive.name) < names.index(p.name) for p in purges)
        assert {p.argv[p.argv.index("--dir") + 1] for p in purges} == {
            str(vec / f"s{seed}-order{k}") for k in (1, 2)}  # fmt: skip
    (snapshot,) = [s for s in steps(plan, "close") if s.name == "close/json-snapshot"]
    assert snapshot.argv[snapshot.argv.index("--src") + 1] == str(vec.parent)

    def shape(step) -> tuple[list[str], list[str]]:
        """The step's with-runs flags and its replay's argv, its seed, run and folder named."""
        argv = list(step.argv)
        cut = argv.index("--")
        command = argv[cut + 1 :]
        seed = int(next(a for a in command if a.startswith("seed=")).split("=")[1])
        folder = command[command.index("--out-dir") + 1]
        run = command[command.index("--reference-fresh-run") + 1]
        assert run.endswith(f"/fetched/study5/{camp.POD2_FRESH_L[seed]}")
        out = []
        for a in command:
            a = a.replace(folder, "<folder>").replace(run, "<run>")
            for offset, tag in ((1000, "<s1>"), (2000, "<s2>"), (0, "<s>")):
                a = a.replace(str(seed + offset), tag)
            out.append(a)
        return [a for a in argv[:cut] if a.startswith("--")], out

    mine = {json.dumps(shape(s)) for s in replays}
    theirs = {json.dumps(shape(s)) for s in replay_steps(pod4(where), "sealed")}
    assert mine == theirs and len(mine) == 2  # order 1 and order 2
    assert all("--no-k" in s.argv and camp.SURROGATE in s.argv for s in replays)


def test_topup_never_names_an_e2_run(where):
    """No planned argv, step, output or campaign names an E2 seed, its new orders' seeds, an E2
    cell or pod 3's archive; every seed named is a top-up seed, every run fetched pod 2's."""
    plan = topup(where)
    numbers = "|".join(str(s + offset) for s in E2_SEEDS for offset in (0, 1000, 2000))
    e2 = re.compile(rf"(?<!\d)(?:{numbers})(?!\d)")
    texts = [x for s in steps(plan) for x in (s.name, *s.argv, *s.outputs, *s.kept_if_runs)]
    texts += [*plan.campaigns, *plan.campaigns.values()]
    assert [t for t in texts if e2.search(t)] == []
    assert [t for t in texts if "pod3" in t] == []
    argv = [s.argv for s in steps(plan)]
    seeds = {int(a.split("=", 1)[1]) for args in argv for a in args if a.startswith("seed=")}
    seeds |= {int(args[args.index("--seeds") + 1]) for args in argv if "--seeds" in args}
    assert seeds == set(camp.TOPUP_SEEDS)
    fetched = {args[args.index("--run-id") + 1] for args in argv if "fetch-run" in args}
    assert fetched == {camp.POD2_ORDER_RUN, *(camp.POD2_FRESH_L[s] for s in camp.TOPUP_SEEDS)}
    assert all(args[args.index("--prefix") + 1] == camp.POD2_RUNS for args in argv
               if "fetch-run" in args)  # fmt: skip
    streams = [args for args in argv if "fetch-streams" in args]
    assert sorted(int(a[a.index("--seeds") + 1]) for a in streams) == sorted(camp.TOPUP_SEEDS)


def test_topup_engineering_check_reads_pod4s_completed_rerun(where, tmp_path, monkeypatch):
    """The engineering check as on pod 4, under the numerical rule: pod 2's order control of
    50000 at η_l, the same cell. On pod 4's volume the launcher reuses pod 4's completed rerun of
    that config and journals nothing in the top-up's campaign, so the check names that rerun
    through pod 4's engineering campaign, and pod 4's diagnostic rerun beside it when present."""
    from zero_field_refusal import launch

    plan = topup(where)
    eng = steps(plan, "engineering")
    assert all(s.on_fail == "stop" for s in eng)
    launch_step = next(s for s in eng if s.kind == "launch")
    name = Path(launch_step.argv[launch_step.argv.index("--campaign-file") + 1]).stem
    assert name == f"study5-{TOPUP_ID}-engineering"
    (rerun,) = campaign_runs(plan, tmp_path, name)
    four = pod4(where)
    (theirs,) = campaign_runs(four, tmp_path, "study5-pod4-20261112-engineering")
    assert (rerun.label, rerun.config_hash) == (theirs.label, theirs.config_hash)
    assert camp.POD2_ORDER_RUN.split("-")[-2] == rerun.config_hash[:8]
    (check,) = [s for s in eng if s.gate == "engineering"]
    assert check.argv[check.argv.index("--tolerance") + 1] == str(camp.EXT2_TOLERANCE["relative"])
    assert str(where["out"] / "fetched" / "study5" / camp.POD2_ORDER_RUN) in check.argv
    pod4_eng = f"study5-{camp.POD4_ARCHIVE_ID}-engineering"
    reruns = check.argv[check.argv.index("--rerun") + 1 : check.argv.index("--out")]
    assert list(reruns) == [f"{{run:{rerun.label}@{pod4_eng}}}"]
    assert check.outputs[0].endswith(f"study5_records/{TOPUP_ID}/engineering-check-numerical.json")
    # pod 4's volume: its rerun completed under its own engineering campaign
    runs = where["runs"]
    done = runs / rerun.experiment / f"20261111T000000Z-x-{rerun.config_hash[:8]}-a1"
    done.mkdir(parents=True)
    (done / "meta.json").write_text(json.dumps({"config_hash": rerun.config_hash,
                                                "status": "completed"}))  # fmt: skip
    journal_of(runs, pod4_eng, [
        {"event": "start", "run": rerun.label, "run_id": done.name, "attempt": 1,
         "experiment": rerun.experiment, "config_hash": rerun.config_hash},
        {"event": "end", "run": rerun.label, "run_id": done.name, "attempt": 1,
         "outcome": "done"}])  # fmt: skip
    monkeypatch.setattr(launch, "_disqualification", lambda run, run_dir, meta: None)

    def never(*args):
        raise AssertionError("the completed rerun is not run again")

    result = launch.run_campaign(name, [rerun], runs, execute=never)
    assert result["skipped"] == [rerun.label]
    assert not (runs / "launch" / f"{name}.jsonl").exists()  # nothing journaled here
    assert camp.substituted(list(reruns), runs, name) == [str(done)]
    with pytest.raises(camp.StepError):  # the top-up's own campaign has no end of it
        camp.substituted([f"{{run:{rerun.label}}}"], runs, name)
    diag = runs / camp.POD4_DIAG / camp.POD4_DIAG_RERUN
    diag.mkdir(parents=True)
    (check,) = [s for s in steps(topup(where), "engineering") if s.gate == "engineering"]
    reruns = check.argv[check.argv.index("--rerun") + 1 : check.argv.index("--out")]
    assert list(reruns) == [f"{{run:{rerun.label}@{pod4_eng}}}", str(diag)]


def test_topup_fetches_only_what_is_missing(where, tmp_path, bucket):
    """The top-up's fetch steps on pod 4's volume: a stream already in DATA_DIR is checked by its
    sha256 and kept, a missing one fetched; a run already fetched is kept by its output."""
    plan = topup(where)
    seed = 50016
    (step,) = [s for s in steps(plan, "fetch") if s.name == f"fetch/s{seed}/streams"]
    data, rows = where["data"], []
    for rel, body in ((f"teacher/M/training/h/seed-{seed}/chunk-00000/samples.jsonl", b"t"),
                      (f"teacher/M/preconditioner/g/seed-{seed}/samples.jsonl", b"p")):  # fmt: skip
        (bucket / "data" / rel).parent.mkdir(parents=True, exist_ok=True)
        (bucket / "data" / rel).write_bytes(body)
        rows.append(
            f"{rel}\t1\t{hashlib.sha256(body).hexdigest()}\tx\tgs://bkt/proj/data/{rel}\tok\n"
        )
    inventory = bucket / camp.POD2_STREAMS.format(seed=seed)
    inventory.parent.mkdir(parents=True)
    inventory.write_text("path\tsize\tsha256\tmd5_b64\turl\tstatus\n" + "".join(rows))
    present = data / rows[0].split("\t")[0]  # fetched by pod 4
    present.parent.mkdir(parents=True)
    present.write_bytes(b"t")
    assert camp.main(list(step.argv[1:])) == 0
    assert (data / rows[1].split("\t")[0]).read_bytes() == b"p"
    assert camp.main(list(step.argv[1:])) == 0  # both kept now
    present.write_bytes(b"other")
    assert camp.main(list(step.argv[1:])) == 1  # never replaced
    (run,) = [s for s in steps(plan, "fetch") if s.name == f"fetch/s{seed}/f-run"]
    assert run.on_fail == "stop" and run.outputs == (
        str(where["out"] / "fetched" / "study5" / camp.POD2_FRESH_L[seed] / "meta.json"),
    )


def _pod4_volume(tmp_path):
    """Pod 4's runs root after its campaign, with the top-up's plan on it."""
    runs = tmp_path / "runs"
    where = {"out": runs / "campaigns" / "study5-topup", "runs": runs, "data": tmp_path / "data",
             "checkpoints": Path("/workspace/ckpt/study5-pod4")}  # fmt: skip
    plan = topup(where)
    own = camp.POD4_ARCHIVE_ID
    _meta(runs, "study5/20261111T000000Z-s5-fresh-1.3e-05-s50000-s50000-f3e969f1-a1", own)
    _meta(runs, "study5_floor/20261111T000000Z-s5-floor-1.3e-05-s50000-s50000-f3e969f1-a1", own)
    (runs / "study5_records" / own).mkdir(parents=True)
    (runs / "launch").mkdir()
    (runs / "launch" / f"study5-{own}-engineering.jsonl").write_text("")
    (runs / "launch" / f"study5-{own}-replays-g50000").mkdir()
    (runs / "study5_vector" / f"s5-{own}-mean").mkdir(parents=True)
    for folder in ("study5-pod4", "study5-pod4-r2"):  # pod 4's attempts
        (runs / "campaigns" / folder).mkdir(parents=True)
        (runs / "campaigns" / folder / "plan.json").write_text(
            json.dumps({"pod": "pod4", "archive_id": own}))  # fmt: skip
    (runs / camp.POD4_DIAG / "study5").mkdir(parents=True)
    # the top-up's own, as at a restart
    _meta(runs, "study5/20261113T000000Z-s5-order1-1.3e-05-s50016-s50016-aaaaaaaa-a1", TOPUP_ID)
    (runs / "study5_records" / TOPUP_ID).mkdir()
    (runs / "launch" / f"study5-{TOPUP_ID}-orders-s50016.jsonl").write_text("")
    plan.out.mkdir(parents=True)
    return plan, runs


@pytest.mark.parametrize("entry", [
    "pod3-journal", "pod3-records", "pod3-order-run", "other-pod4-records", "pod3-campaign-folder",
    "other-vector-run", "run-without-meta",
])  # fmt: skip
def test_topup_needs_pod4s_root_holding_only_pod4s_archive_and_its_own(tmp_path, entry):
    """The top-up writes into pod 4's runs root, which may hold pod 4's archive and the top-up's
    own and nothing else: pod 3's E2 journals, records or runs, another pod 4 id, another
    campaign folder or a run without a readable meta.json are refused before anything is
    written."""
    plan, runs = _pod4_volume(tmp_path)
    assert camp.own_root_problems(plan) == []
    {
        "pod3-journal": lambda: (
            runs / "launch" / "study5-pod3-20261111-orders-s50006.jsonl").write_text(""),
        "pod3-records": lambda: (runs / "study5_records" / "pod3-20261111").mkdir(),
        "pod3-order-run": lambda: _meta(
            runs, "study5/20261111T000000Z-s5-order1-1.3e-05-s50006-s50006-bbbbbbbb-a1",
            "pod3-20261111"),
        "other-pod4-records": lambda: (runs / "study5_records" / "pod4-20261112").mkdir(),
        "pod3-campaign-folder": lambda: (runs / "campaigns" / "study5-pod3").mkdir(),
        "other-vector-run": lambda: (runs / "study5_vector" / "s5-pod4-20261112-mean").mkdir(),
        "run-without-meta": lambda: (runs / "study5" / "20261113T000000Z-s5-order2").mkdir(),
    }[entry]()  # fmt: skip
    (found,) = camp.own_root_problems(plan)
    assert f"is not topup's own or {camp.POD4_ARCHIVE_ID}'s" in found


def test_topup_needs_an_archive_id_of_its_own_and_pod4_refuses_the_topups(where):
    for archive_id in (camp.POD4_ARCHIVE_ID, "pod3-20261111", "t5"):
        (found,) = camp.own_root_problems(topup(where, archive_id))
        assert "topup's archive id starts with 'topup-'" in found
    assert camp.own_root_problems(topup(where)) == []
    (found,) = camp.own_root_problems(camp.pod4_plan(archive_id="topup-20261113", commit=COMMIT,
                                                     gpu_code=GPU, **where))  # fmt: skip
    assert "pod4's archive id starts with 'pod4-'" in found
    plan = pod4(where)
    (where["runs"] / "study5_records" / TOPUP_ID).mkdir(parents=True)
    (found,) = camp.own_root_problems(plan)
    assert "is not pod4's own" in found


def test_topup_needs_pod4s_gpu_code_and_both_extensions(where, monkeypatch):
    plan = topup(where)
    problems = camp.preflight(plan)
    assert any("study5-extension-2-code-identity.json" in p for p in problems)
    plan.gpu_code = json.loads((REPO_ROOT / camp.EXT2_IDENTITY).read_text())["gpu_code_sha256"]
    problems = camp.preflight(plan)
    assert not any("GPU code" in p or "does not hold" in p for p in problems)
    monkeypatch.setattr(camp, "EXTENSION_HEADING", "### Extension 9")
    assert any("does not hold extension 1" in p for p in camp.preflight(plan))
    monkeypatch.setattr(camp, "EXT2_HEADING", "### Extension 10")
    assert any("does not hold extension 2" in p for p in camp.preflight(plan))


def test_topup_admits_each_group_by_its_minutes_and_the_dry_run_prints_its_hours(
    where, tmp_path, capsys, monkeypatch
):
    """Minutes from the unit costs the code holds (an F run 6.9, a q-only replay 4.5, a settle or
    purge 0.5, an archive 1.0); each block and seed group admitted only when 1.5 × its minutes
    and the close's 10 fit before the cap; the dry run prints the hours, planned and × 1.5, and
    writes nothing."""
    plan = topup(where)
    for block, seed, work in (("predictions", 50016, 2 * (4.5 + 0.5 + 0.5) + 1.0),
                              ("orders", 50016, 4 * 6.9 + 1.0)):  # fmt: skip
        group = [s for s in steps(plan, block) if s.group == f"s{seed}"]
        assert group[0].kind == "deadline" and group[0].on_fail == "stop"
        assert sum(s.minutes for s in group[1:]) == pytest.approx(work)
        admit = int(group[0].argv[group[0].argv.index("--admit") + 1])
        assert admit == math.ceil(1.5 * work + 10)
        assert minutes_of(group[0]) == camp.deadline_minutes(work)
    deadlines = [s for b in plan.blocks if b.name != "close" for s in b.steps
                 if s.kind == "deadline"]  # fmt: skip
    assert all("--admit" in s.argv and "--reserve" not in s.argv for s in deadlines)
    minutes = camp.planned_minutes(plan)
    assert minutes["predictions"] == pytest.approx(30 * 12.0 + 1.5)
    assert minutes["orders"] == pytest.approx(30 * 28.6 + 1.5)
    total = sum(minutes.values())
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    out = tmp_path / "c"
    assert camp.main(["topup", "--out", str(out), "--archive-id", TOPUP_ID, "--checkpoints",
                      "/workspace/ckpt/study5-pod4", "--dry-run"]) == 0  # fmt: skip
    printed = capsys.readouterr().out
    assert f"planned {total / 60:.2f} h, x1.5 {1.5 * total / 60:.2f} h" in printed
    assert "block predictions: 243 steps, 361.5 min" in printed
    assert not out.exists() and not (tmp_path / "runs").exists()


def test_topup_preflight_refuses_a_short_cap_a_lost_prediction_and_a_missing_pod4_rerun(
    where, tmp_path, monkeypatch
):
    """Before anything runs, the top-up refuses: a cap ZFR_DEADLINE_EPOCH not set or under 33 h
    away (1.5 × the planned minutes and the close fit in 33 h); a prediction settled under
    sealed/ whose vector folder's record.json is gone, which would be computed again; pod 4's
    engineering rerun, which the check names, without its completed run in pod 4's journal."""
    plan = topup({**where, "checkpoints": tmp_path / "ckpt"})
    runs, now = where["runs"], 1_800_000_000
    pod4_eng = f"study5-{camp.POD4_ARCHIVE_ID}-engineering"
    label = "s5-order-1.3e-05-s50000/s50000"
    start = {"event": "start", "run": label, "run_id": "r-a1", "experiment": "study5"}
    done = {"event": "end", "run": label, "run_id": "r-a1", "outcome": "done"}
    (runs / "study5" / "r-a1").mkdir(parents=True)
    journal_of(runs, pod4_eng, [start, done])
    monkeypatch.setenv("ZFR_DEADLINE_EPOCH", str(now + 33 * 3600))
    assert camp.topup_problems(plan, now) == []
    # the cap
    need = camp.ADMIT_FACTOR * sum(camp.planned_minutes(plan).values()) + camp.CLOSE_BUDGET
    assert need <= 60 * camp.TOPUP_CAP_HOURS == 60 * 33
    monkeypatch.setenv("ZFR_DEADLINE_EPOCH", str(now + 33 * 3600 - 60))
    (found,) = camp.topup_problems(plan, now)
    assert "is 32.98 h away, under the 33 h the top-up needs" in found
    monkeypatch.setenv("ZFR_DEADLINE_EPOCH", str(int(time.time()) + 32 * 3600))
    assert any("under the 33 h" in p for p in camp.preflight(plan))
    monkeypatch.delenv("ZFR_DEADLINE_EPOCH")
    (found,) = camp.topup_problems(plan, now)
    assert "ZFR_DEADLINE_EPOCH is not set" in found
    monkeypatch.setenv("ZFR_DEADLINE_EPOCH", str(now + 34 * 3600))
    # a settled prediction whose vector folder lost its record
    settle = next(
        s for s in steps(plan, "predictions") if s.name == "predictions/s50016/order1-settle"
    )
    folder = Path(settle.argv[settle.argv.index("--dir") + 1])
    target = Path(settle.argv[settle.argv.index("--records") + 1])
    target.mkdir(parents=True)
    (target / "record.json").write_text("{}")
    (found,) = camp.topup_problems(plan, now)
    assert "1 predictions are settled" in found and "s50016-order1" in found
    folder.mkdir(parents=True)
    (folder / "record.json").write_text("{}")  # as the replay left it: kept, never recomputed
    assert camp.topup_problems(plan, now) == []
    (target / "record.json").unlink()  # written but not settled yet: settle copies it
    assert camp.topup_problems(plan, now) == []
    # pod 4's rerun: not ended, not completed, its directory gone, its journal gone
    for lines, gone in (([start], False), ([start, {**done, "outcome": "technical"}], False),
                        ([start, done], True)):  # fmt: skip
        journal_of(runs, pod4_eng, lines)
        if gone:
            (runs / "study5" / "r-a1").rmdir()
        (found,) = camp.topup_problems(plan, now)
        assert found.startswith("the run a step names is not there") and pod4_eng in found
    (runs / "launch" / f"{pod4_eng}.jsonl").unlink()
    (found,) = camp.topup_problems(plan, now)
    assert "has no completed run (no end)" in found
