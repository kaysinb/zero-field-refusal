"""Study 5's dose experiment on two Pods (scripts/study5_dose_campaigns.py): the two plans as
data, the pilot table, the grid step and its record, the main plan built from that record."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_study5_campaigns import COMMIT, GPU, campaign_runs, script_module, steps

from zero_field_refusal import dose
from zero_field_refusal.config import config_hash
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import load_run_config

camp = script_module(REPO_ROOT / "scripts" / "study5_campaigns.py", "study5_campaigns")
m = script_module(REPO_ROOT / "scripts" / "study5_dose_campaigns.py", "study5_dose_campaigns")
SEEDS = json.loads((REPO_ROOT / dose.SEEDS_FILE).read_text())
K = {"0.75": 4.0, "1.3": 5.5, "2.0": 8.0}  # the late candidates' K_benign in every cell


@pytest.fixture
def where(tmp_path):
    return {"runs": tmp_path / "runs", "data": tmp_path / "data",
            "checkpoints": Path("/workspace/ckpt/dose"), "commit": COMMIT, "gpu_code": GPU}  # fmt: skip


def pilot(where, pod="a"):
    out = where["runs"] / "campaigns" / f"study5-dose-{pod}-pilot"
    return m.pilot_plan(pod=pod, out=out, archive_id=f"dose-{pod}-t",
                        other_archive_id=f"dose-{m.other(pod)}-t", **where)  # fmt: skip


def table(pod, seed, k=K, gpu=GPU, complete=lambda e, late: True):
    rows = [{"seed": seed, "early": e, "late": late, "label": "x", "run_id": f"r-{e}-{late}",
             "outcome": "done", "steps": 25, "stop": "steps", "complete": complete(e, late),
             "K": k[late]} for e, _ in dose.EARLY for late, _ in dose.LATE_CANDIDATES]  # fmt: skip
    return {"kind": m.TABLE_KIND, "pod": pod, "seed": seed, "gpu_code": gpu, "rows": rows}


def grid_record(path: Path, k=K) -> Path:
    tables = {pod: table(pod, SEEDS["pods"][pod]["pilot"], k) for pod in dose.PODS}
    rows = [r for pod in dose.PODS for r in tables[pod]["rows"]]
    seeds = [SEEDS["pods"][pod]["pilot"] for pod in dose.PODS]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": m.GRID_KIND, "tables": tables,
                                "grid": dose.grid_choice(rows, seeds)}))  # fmt: skip
    return path


def main(where, pod="a", k=K):
    grid = grid_record(where["runs"] / "study5_records" / f"dose-{pod}-t" / m.GRID, k)
    out = where["runs"] / "campaigns" / f"study5-dose-{pod}-main"
    return m.main_plan(pod=pod, out=out, archive_id=f"dose-{pod}-t", grid_path=grid,
                       pilot_out=out.with_name(f"study5-dose-{pod}-pilot"), **where)  # fmt: skip


def test_the_pods_share_the_sixteen_seeds_with_the_extreme_pilots():
    a, b = SEEDS["pods"]["a"], SEEDS["pods"]["b"]
    k = {int(s): v for s, v in SEEDS["k_benign_eta_l"].items()}
    assert sorted(a["seeds"] + b["seeds"]) == SEEDS["seeds"] and len(a["seeds"]) == 8
    assert k[a["pilot"]] == min(k.values()) and k[b["pilot"]] == max(k.values())
    assert m.seeds_record()["pods"] == SEEDS["pods"]


def test_the_seeds_record_must_be_the_draw(tmp_path):
    (tmp_path / "docs").mkdir()
    edited = dict(SEEDS, drawn_in_order=list(reversed(SEEDS["drawn_in_order"])))
    (tmp_path / dose.SEEDS_FILE).write_text(json.dumps(edited))
    with pytest.raises(camp.StepError, match="draw"):
        m.seeds_record(tmp_path)


def test_a_dose_runs_config_passes_the_protocol_and_the_scan_reads_its_hash(where, tmp_path):
    plan = main(where)
    check = steps(plan, "pilot-scans")[1]
    lo = json.loads(Path(check.argv[3]).read_text())["grid"]["chosen"]["lo"]
    assert lo == "1.3"
    for name in plan.campaigns:
        for run in campaign_runs(plan, tmp_path, name):
            early, late = run.cell.split("-")[2][1:], run.cell.split("-")[3][1:]
            cfg = load_run_config(*m.run_args(run.seed, early, late)[:1],
                                  list(m.run_args(run.seed, early, late)[1:]))  # fmt: skip
            assert config_hash(cfg) == run.config_hash, run.label
            assert cfg.study5.role == "dose" and cfg.optimizer.lr == dose.late_rate(late)
            assert cfg.train.rate_schedule.early_lr == dose.early_rate(early)
            assert ("tracking.endpoint_checkpoint=/workspace/ckpt/dose" in run.overrides) == (
                late == lo)  # fmt: skip


def test_the_pilot_plan(where, tmp_path):
    plan = pilot(where)
    assert [b.name for b in plan.blocks] == ["engineering", "fetch", "pilot", "grid", "close"]
    for block in plan.blocks[:-1]:
        assert block.steps[0].kind == "deadline" and block.steps[-1].kind == "receipt"
    check = next(s for s in steps(plan) if s.name == "engineering/check")
    assert check.on_fail == "stop" and check.gate == "engineering"
    assert check.argv[check.argv.index("--tolerance") + 1] == str(camp.EXT2_TOLERANCE["relative"])
    fetched = [s for s in steps(plan, "fetch") if s.name.endswith("/streams")]
    assert [int(s.argv[s.argv.index("--seeds") + 1]) for s in fetched] == SEEDS["pods"]["a"]["seeds"]
    seed = SEEDS["pods"]["a"]["pilot"]
    runs = campaign_runs(plan, tmp_path, f"study5-dose-a-t-pilot-s{seed}")
    assert len(runs) == 9 and {r.seed for r in runs} == {seed}
    assert all("tracking.endpoint_checkpoint=/workspace/ckpt/dose" in r.overrides for r in runs)
    names = [s.name for s in steps(plan, "pilot")]
    assert names.index("pilot/launch") < names.index("pilot/table") < names.index("pilot/publish")
    tab = next(s for s in steps(plan) if s.name == "pilot/table")
    assert tab.on_fail == "stop" and tab.argv.count("--run") == 9
    choice = next(s for s in steps(plan) if s.name == "grid/choice")
    assert choice.on_fail == "stop"
    argv = list(choice.argv)
    assert argv[argv.index("--other-remote") + 1] == "study5/dose-b-t/pilot/pilot-table.json"
    assert argv[argv.index("--wait") + 1] == str(m.WAIT)
    pilots = [argv[i + 1] for i, x in enumerate(argv) if x == "--pilot"]
    assert pilots == [f"a={seed}", f"b={SEEDS['pods']['b']['pilot']}"]
    assert m.main(["run", "--pod", "a", "--out", str(tmp_path / "o"), "--archive-id", "dose-b-x",
                   "--other-archive-id", "dose-a-x", "--checkpoints", "/x", "--dry-run"]) == 1


def test_the_main_plan_from_its_grid(where, tmp_path):
    plan = main(where)
    assert [b.name for b in plan.blocks] == ["pilot-scans", "main", "close"]
    first = steps(plan, "pilot-scans")[1]
    assert first.name == "pilot-scans/check-grid" and first.on_fail == "stop"
    seeds = [s for s in SEEDS["pods"]["a"]["seeds"] if s != SEEDS["pods"]["a"]["pilot"]]
    assert len(plan.campaigns) == len(seeds) == 7
    for seed in seeds:
        runs = campaign_runs(plan, tmp_path, f"study5-dose-a-t-main-s{seed}")
        cells = {(r.cell.split("-")[2], r.cell.split("-")[3]) for r in runs}
        assert cells == {(f"e{e}", f"l{late}") for e, _ in dose.EARLY for late in ("1.3", "2.0")}
        group = [s for s in steps(plan, "main") if s.group == f"s{seed}"]
        names = [s.name.split("/")[-1] for s in group]
        assert names[:3] == ["deadline", "check-checkpoints", "launch-main"]
        scans = [s for s in group if "/scan-" in s.name]
        assert len(scans) == 3
        for s in scans:
            argv = list(s.argv)
            assert argv[argv.index("--points") + 1] == "-0.5"
            assert "optimizer.lr=1.69e-05" in argv  # the scans are of the L_lo runs
        drops = [s for s in group if "/drop-" in s.name]
        assert len(drops) == 3 and names.index("archive") < names.index(drops[0].name.split("/")[-1])
        for d in drops:
            argv = list(d.argv)
            assert argv[argv.index("--consumer") + 1].startswith("study5_records/dose-a-t/scan/")
    pilot_drops = [s for s in steps(plan, "pilot-scans") if "/drop-" in s.name]
    assert len(pilot_drops) == 9
    for d in pilot_drops:
        consumer = d.argv[list(d.argv).index("--consumer") + 1]
        if d.name.endswith("-l1.3"):
            assert consumer.startswith("study5_records/dose-a-t/scan/")
        else:
            assert consumer == "study5_records/dose-a-t/grid.json"


def test_the_main_plan_refuses_a_grid_that_is_not_the_rules(where):
    path = grid_record(where["runs"] / "study5_records" / "dose-a-t" / m.GRID)
    record = json.loads(path.read_text())
    record["grid"]["chosen"]["lo"] = "0.75"
    path.write_text(json.dumps(record))
    with pytest.raises(camp.StepError, match="rule"):
        m.read_grid(path)


def _run_dir(runs: Path, campaign: str, label: str, run_id: str, *, outcome="done", steps=25,
             stop="steps", k=5.5) -> None:  # fmt: skip
    launch = runs / "launch"
    launch.mkdir(parents=True, exist_ok=True)
    with (launch / f"{campaign}.jsonl").open("a") as f:
        f.write(json.dumps({"event": "start", "run": label, "run_id": run_id,
                            "experiment": "study5"}) + "\n")  # fmt: skip
        f.write(json.dumps({"event": "end", "run": label, "run_id": run_id,
                            "outcome": outcome}) + "\n")  # fmt: skip
    run_dir = runs / "study5" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "terminal.json").write_text(json.dumps({"steps": steps, "stop": {"reason": stop}}))
    rows = [{"step": steps, "condition": f"step-{steps}", "K_benign": k}] if k is not None else []
    (run_dir / "evaluations.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_pilot_table_holds_each_runs_end_and_k(tmp_path, capsys):
    runs = tmp_path / "runs"
    _run_dir(runs, "c", "done/s1", "r1", k=5.25)
    _run_dir(runs, "c", "stopped/s1", "r2", outcome="diverged", steps=7, stop="grad_norm", k=None)
    out = tmp_path / "t.json"
    assert m.main(["table", "--runs", str(runs), "--campaign", "c", "--pod", "a", "--seed", "1",
                   "--gpu-code", GPU, "--run", "0.6", "1.3", "done/s1", "--run", "1.0", "2.0",
                   "stopped/s1", "--out", str(out)]) == 0  # fmt: skip
    rows = json.loads(out.read_text())["rows"]
    assert rows[0]["complete"] is True and rows[0]["K"] == 5.25
    assert rows[1]["complete"] is False and rows[1]["K"] is None and rows[1]["stop"] == "grad_norm"
    assert m.main(["table", "--runs", str(runs), "--campaign", "c", "--pod", "a", "--seed", "1",
                   "--gpu-code", GPU, "--run", "0.6", "1.3", "absent/s1", "--out",
                   str(tmp_path / "u.json")]) == 1  # fmt: skip


def grid_args(tmp_path, own, theirs_target):
    a, b = SEEDS["pods"]["a"]["pilot"], SEEDS["pods"]["b"]["pilot"]
    return ["grid", "--own", str(own), "--other-remote", "study5/dose-b-t/pilot/x.json",
            "--fetched", str(theirs_target), "--wait", "0.01", "--pilot", f"a={a}",
            "--pilot", f"b={b}", "--gpu-code", GPU, "--out", str(tmp_path / "grid.json")]  # fmt: skip


def test_the_grid_step_applies_the_rule_to_both_tables(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_REMOTE", "gs://bucket")
    own, theirs = tmp_path / "own.json", tmp_path / "theirs.json"
    own.write_text(json.dumps(table("a", SEEDS["pods"]["a"]["pilot"])))
    other = table("b", SEEDS["pods"]["b"]["pilot"])

    def fake_cp(args):  # the other Pod's table appears in the bucket
        assert args[0] == "cp" and args[2] == "gs://bucket/study5/dose-b-t/pilot/x.json"
        Path(args[3]).write_text(json.dumps(other))
        return ""

    monkeypatch.setattr(camp, "gcloud", fake_cp)
    assert m.main(grid_args(tmp_path, own, theirs)) == 0
    record = m.read_grid(tmp_path / "grid.json")
    assert record["grid"]["chosen"]["lo"] == "1.3" and record["grid"]["chosen"]["hi"] == "2.0"
    assert sorted(record["tables_sha256"]) == ["a", "b"]


def test_the_grid_step_refuses_another_gpu_code_or_no_table(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_REMOTE", "gs://bucket")
    own = tmp_path / "own.json"
    own.write_text(json.dumps(table("a", SEEDS["pods"]["a"]["pilot"])))
    theirs = tmp_path / "theirs.json"
    theirs.write_text(json.dumps(table("b", SEEDS["pods"]["b"]["pilot"], gpu="f" * 64)))
    assert m.main(grid_args(tmp_path, own, theirs)) == 1  # kept: no fetch, then refused
    assert not (tmp_path / "grid.json").exists()

    def missing(args):
        raise camp.StepError("gcloud storage cp failed (exit 1)")

    monkeypatch.setattr(camp, "gcloud", missing)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    absent = tmp_path / "absent.json"
    assert m.main(grid_args(tmp_path, own, absent)) == 1
    assert not absent.exists() and not (tmp_path / "grid.json").exists()


def test_the_adamw_plan(where, tmp_path):
    """The order repeat under AdamW: the dose Pods' engineering check first, the ten seeds'
    streams, then per seed one campaign of four A runs (E2's two orders × A's two rates), each
    passing the protocol under ``order_a``."""
    from zero_field_refusal import adamw_orders

    out = where["runs"] / "campaigns" / "study5-adamw"
    plan = m.adamw_plan(out=out, archive_id="adamw-t", **where)
    assert plan.pod == "adamw"
    assert [b.name for b in plan.blocks] == ["engineering", "fetch", "orders", "close"]
    check = next(s for s in steps(plan) if s.name == "engineering/check")
    assert check.on_fail == "stop" and check.gate == "engineering"
    seeds = json.loads((REPO_ROOT / adamw_orders.SEEDS_FILE).read_text())["seeds"]
    fetched = [s for s in steps(plan, "fetch") if s.name.endswith("/streams")]
    assert [int(s.argv[s.argv.index("--seeds") + 1]) for s in fetched] == seeds
    assert len(plan.campaigns) == 11  # the engineering rerun and ten seeds
    for seed in seeds:
        runs = campaign_runs(plan, tmp_path, f"study5-adamw-t-orders-s{seed}")
        found = {(r.overrides[0], r.overrides[1]) for r in runs}
        assert found == {(f"optimizer.lr={lr!r}", f"train.order_seed={seed + offset}")
                         for _, offset in adamw_orders.ORDER_OFFSETS
                         for _, lr in adamw_orders.RATES}  # fmt: skip
        for run in runs:
            cfg = load_run_config(adamw_orders.CONFIG, list(run.overrides))
            assert config_hash(cfg) == run.config_hash and cfg.optimizer.name == "adamw"
            assert not any("endpoint_checkpoint" in o for o in run.overrides)
    assert m.main(["adamw", "--out", str(tmp_path / "o"), "--archive-id", "dose-a-x",
                   "--checkpoints", "/x", "--dry-run"]) == 1  # not the AdamW Pod's archive id


def test_the_adamw_plan_refuses_a_seeds_record_that_is_not_the_draw(where, tmp_path):
    from zero_field_refusal import adamw_orders

    (tmp_path / "docs").mkdir()
    record = json.loads((REPO_ROOT / adamw_orders.SEEDS_FILE).read_text())
    record["seeds"] = record["seeds"][:-1]
    (tmp_path / adamw_orders.SEEDS_FILE).write_text(json.dumps(record))
    with pytest.raises(camp.StepError, match="draw"):
        m.adamw_plan(out=tmp_path / "o", archive_id="adamw-t", repo=tmp_path, **where)
