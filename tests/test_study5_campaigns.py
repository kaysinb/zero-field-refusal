"""scripts/study5_campaigns.py: study 5's two Pod campaigns as plans of steps, built without running
anything, and the helpers its steps call (docs/study5-preregistration.md, "Seeds and order", "The
confirmatory test"; docs/operations.md, "Study 5 campaigns")."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
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
    return camp.pod1_plan(archive_id="t1", commit=COMMIT, **where)


def pod2(where, n=40):
    return camp.pod2_plan(archive_id="t2", commit=COMMIT, n=n, **where)


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
    scan = load_config("experiments/study5_scan", ["optimizer.lr=1.83e-05", "seed=50000"])
    drop = ("experiment", "study5", "eval.controls.line_scan.points")
    assert plain(scan, *drop) == plain(study5, *drop)  # study5 above: the last rate, η_u
    assert scan.eval.controls.line_scan.points == [-1, -0.5, -0.25, 0.25, 0.5]
    assert scan.study5.role == "fresh" and scan.optimizer.name == "adam_frozen"
    assert scan.train.batch_size == 20 and scan.train.steps == 25
    assert camp.ETA_L == 1.3e-05 and camp.ETA_U == 1.83e-05


def test_every_study5_config_passes_the_protocol():
    cases = [
        (
            "experiments/study5_scan",
            [
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
        "scripts/launch.py",
        "with-runs",
        "archive",
        "snapshot",
        "receipt",
    ]
    fetch = block[1]
    assert fetch.argv[fetch.argv.index("--seeds") + 1 :][:1] == ("10403",)
    launch_step = block[4]
    assert launch_step.argv[-2:] == ("--max-attempts", "1")
    runs = campaign_runs(plan, tmp_path, Path(launch_step.argv[1]).stem)
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
    assert len(theta0) == 1 and theta0[0].on_fail == "record"
    (run,) = campaign_runs(plan, tmp_path, Path(theta0[0].argv[1]).stem)
    assert run.entry == "theta0" and run.config == "experiments/study5_theta0"
    surrogate = [s for s in steps(plan, "surrogate") if s.kind == "run"]
    assert [s.argv[0] for s in surrogate] == ["scripts/study5_surrogate_checks.py"]
    assert surrogate[0].on_fail == "stop"
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
    assert fresh[1].argv[1] == "check-published" and fresh[1].on_fail == "stop"
    launches = [s for s in fresh if s.kind == "launch"]
    assert len(launches) == n and all(s.on_fail == "record" for s in launches)


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
            "check-checkpoints",
            "scripts/launch.py",
            "checkpoint",
            "archive",
        ]
        assert group[0].on_fail == "stop" and group[3].on_fail == "stop"
        assert group[0].argv[-2:] == ("--max", "1")
        runs = campaign_runs(plan, tmp_path, Path(group[1].argv[1]).stem)
        assert [(r.config, r.seed) for r in runs] == [
            ("experiments/study5_scan", seed),
            ("experiments/study5", seed),
        ]
        lower, upper = (load_run_config(r.config, list(r.overrides)) for r in runs)
        assert lower.optimizer.lr == 1.3e-05 and upper.optimizer.lr == 1.83e-05
        assert lower.tracking.endpoint_checkpoint == "/root/ckpt/study5"
        assert upper.tracking.endpoint_checkpoint is None
        assert upper.eval.controls.line_scan.get("points") is None
        assert lower.study5.role == upper.study5.role == "fresh"


def test_pod2_controls_and_ladder_on_the_first_six_seeds(where, tmp_path):
    plan = pod2(where)
    controls = steps(plan, "controls")
    groups = sorted({s.group for s in controls if s.group})
    assert groups == [f"s{s}" for s in range(50000, 50006)]
    for seed in range(50000, 50006):
        group = [s for s in controls if s.group == f"s{seed}"]
        launch_step = group[0]
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


def test_publish_then_check_published(tmp_path, bucket):
    record = tmp_path / "r" / "predictions-record.json"
    camp.write_record(record, {"kind": "study5-predictions", "seeds": [50000]})
    receipt = tmp_path / "r" / "predictions-receipt.json"
    check = ["check-published", "--record", str(record), "--receipt", str(receipt)]
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
    assert "block fresh" in printed and "s5-fresh-scan-1.3e-05-s50024" not in printed
    assert "study5-t-fresh-s50023.yaml" in printed and not out.exists()
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
