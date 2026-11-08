"""scripts/study5_campaigns.py: study 5's two Pod campaigns as plans of steps, built without running
anything, and the helpers its steps call (docs/study5-preregistration.md, "Seeds and order", "The
confirmatory test"; docs/operations.md, "Study 5 campaigns")."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
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
    assert surrogate[0].argv[-2:] == ("--surrogate", "configs/study5/surrogate.yaml")
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

    default = load_spec(REPO_ROOT / "configs" / "study5" / "surrogate.yaml").sha256
    for plan, count in ((pod1(where), 1 + 60), (pod2(where, 24), 2 * 24)):
        found = surrogate_steps(plan)
        assert len(found) == count
        for step in found:
            assert step.argv[step.argv.index("--surrogate") + 1] == "configs/study5/surrogate.yaml"
        assert (plan.surrogate, plan.surrogate_sha256) == ("configs/study5/surrogate.yaml", default)
        assert plan.to_json()["surrogate_sha256"] == default
    # a second set: a new file in the checkout, named by --surrogate of both plans
    repo = tmp_path / "repo"
    second = repo / "configs" / "study5" / "surrogate-2.yaml"
    second.parent.mkdir(parents=True)
    text = (REPO_ROOT / "configs" / "study5" / "surrogate.yaml").read_text()
    second.write_text(text.replace("version: 1", "version: 2"))
    monkeypatch.setattr(camp, "REPO", repo)
    plan = camp.pod2_plan(archive_id="t2", commit=COMMIT, gpu_code=GPU, n=24,
                          surrogate="configs/study5/surrogate-2.yaml", **where)  # fmt: skip
    assert plan.surrogate_sha256 == load_spec(second).sha256 != default
    for step in surrogate_steps(plan):
        assert step.argv[step.argv.index("--surrogate") + 1] == "configs/study5/surrogate-2.yaml"
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
    spec = load_spec(REPO_ROOT / "configs" / "study5" / "surrogate.yaml").sha256
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
        (checks_record(surrogate_sha256="0" * 64), "not of configs/study5/surrogate.yaml"),
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
    args = camp.parser_of().parse_args(
        ["pod2", "--out", "o", "--archive-id", "t2", "--engineering-check", "e.json",
         "--surrogate-checks", "c.json", "--retrospective", "x.json"])  # fmt: skip
    found = camp.preflight(pod2(where, 24), args)
    assert found[-2:] == ["r", "g"]
    assert seen == [Path("x.json"), (Path("e.json"), Path("c.json"))]
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
        assert paths[1:] == [record.relative_to(where["runs"]).as_posix(), f"study5_scan/{run_id}"]


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
        assert rungs == ["isotropic", "linear_no_d", "linear"]
        assert all(s.on_fail == "record" for s in ladders)
        assert group[-1].kind == "archive" and group[-1].on_fail == "stop"
    assert not any("linear_shuffled_d" in " ".join(s.argv) for s in steps(plan))


def test_no_plan_holds_a_bucket_path_or_a_key(where, monkeypatch):
    monkeypatch.setenv("ARTIFACT_REMOTE", "gs://secret-bucket/prefix")
    for plan in (pod1(where), pod2(where)):
        text = json.dumps(plan.to_json()) + "".join(plan.campaigns.values())
        assert "gs://" not in text and "secret-bucket" not in text and "KEY" not in text


def test_the_plan_is_data_and_deterministic(where):
    assert pod1(where).to_json() == pod1(where).to_json()
    assert pod2(where).to_json() == pod2(where).to_json()
    names = [s.name for s in steps(pod2(where))]
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
    runs = tmp_path / "runs"
    journal_of(
        runs, "c", [{"event": "start", "run": "cell/s1", "run_id": "r-a1", "experiment": "study5"}]
    )
    (runs / "study5" / "r-a1").mkdir(parents=True)
    (runs / "study5" / "r-a1" / "meta.json").write_text("{}")
    (runs / "rec.json").write_text("{}")
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
        "--campaign",
        "c",
        "--runs",
        str(runs),
    ]
    assert camp.main(argv) == 0
    rows = camp.read_inventory(inventory)
    assert sorted(r["path"] for r in rows) == [
        "launch/c.jsonl",
        "rec.json",
        "study5/r-a1/meta.json",
    ]
    assert all(r["status"] == "ok" for r in rows)
    assert (bucket / "study5/t/runs/study5/r-a1/meta.json").is_file()
    assert (bucket / "study5/t/inventories/x.tsv").read_bytes() == inventory.read_bytes()
    assert camp.main(argv) == 0  # done: kept


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
# The commit docs/study5-code-identity.json records: an earlier commit of HEAD's GPU code
SAME = json.loads((REPO_ROOT / "docs" / "study5-code-identity.json").read_text())["commit"]
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


def test_every_planned_step_has_a_timeout_and_every_group_sets_its_deadline_first(where):
    for plan in (pod1(where), pod2(where), pod2(where, 24)):
        for block in plan.blocks:
            assert all(isinstance(s.timeout, int) and s.timeout > 0 for s in block.steps)
            if block.name == "close":
                assert block.steps[-1].kind == "deadline"
                continue
            assert block.steps[0].kind == "deadline" and not block.steps[0].group
            window = int(block.steps[0].argv[-1])
            seen = set()
            for step in block.steps[1:]:
                if step.group and step.group not in seen:
                    seen.add(step.group)
                    assert step.kind == "deadline" and step.group, (block.name, step.name)
                    window = int(step.argv[-1])
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
    (inventories / "ck.tsv").write_text(header + f"r-a1.pt\t7\t{'0' * 64}\tx\tu\tok\n")
    assert camp.main(argv) == 1  # the archive shows another sha256
    (inventories / "ck.tsv").write_text(header + f"r-a1.pt\t7\t{digest}\tx\tu\tok\n")
    for p in (tmp_path / "new-out").rglob("*"):
        p.is_file() and p.unlink()
    assert camp.main(argv) == 0
