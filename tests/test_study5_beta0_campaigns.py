"""Study 5's run without momentum on one Pod (scripts/study5_beta0_campaigns.py): the three plans as
data, the pilot table, the rate and gate steps and their records, the main plan built from the
rate record, the orders plan built from the amended gate record (the addendum of 2026-11-18) and
the restart after the registered gate failed on a scientific stop."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_study5_campaigns import COMMIT, GPU, campaign_runs, script_module, steps

from zero_field_refusal import beta0
from zero_field_refusal.analysis import study5_beta0 as sb
from zero_field_refusal.config import config_hash
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import load_run_config

camp = script_module(REPO_ROOT / "scripts" / "study5_campaigns.py", "study5_campaigns")
m = script_module(REPO_ROOT / "scripts" / "study5_beta0_campaigns.py", "study5_beta0_campaigns")
TARGETS = m.targets()


@pytest.fixture
def where(tmp_path):
    return {
        "runs": tmp_path / "runs",
        "data": tmp_path / "data",
        "checkpoints": Path("/workspace/ckpt/beta0"),
        "commit": COMMIT,
        "gpu_code": GPU,
    }


def pilot(where):
    out = where["runs"] / "campaigns" / "study5-beta0-pilot"
    return m.pilot_plan(out=out, archive_id="beta0-t", **where)


def table_rows(k_of=lambda s, lr: 3.0 * (lr / 1.3e-05) ** 2):
    return [{"seed": s, "level": level, "label": "x", "run_id": f"r{s}-{level}", "outcome": "done",
             "steps": 25, "stop": "steps", "complete": True, "K": k_of(s, lr)}
            for s in beta0.PILOT_SEEDS for level, lr in beta0.GRID]  # fmt: skip


def rate_record(path: Path, rows=None) -> Path:
    rows = table_rows() if rows is None else rows
    table = {"kind": m.TABLE_KIND, "gpu_code": GPU, "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": m.RATE_KIND, "table": table,
                                "choice": beta0.rate_choice(rows, TARGETS)}))  # fmt: skip
    return path


def main(where, rows=None):
    rate = rate_record(where["runs"] / "study5_records" / "beta0-t" / m.RATE, rows)
    out = where["runs"] / "campaigns" / "study5-beta0-main"
    return m.main_plan(out=out, archive_id="beta0-t", rate_path=rate, **where)


def gate_rows(stopped=(), factor=1.0) -> list[dict]:
    rows = []
    for s in beta0.SEEDS:
        if s in stopped:
            rows.append({"seed": s, "outcome": "done", "steps": 8, "stop": "gradient_spike",
                         "complete": False, "K": None})  # fmt: skip
        else:
            rows.append({"seed": s, "outcome": "done", "steps": 25, "stop": "steps",
                         "complete": True, "K": TARGETS[s] * factor})  # fmt: skip
    return rows


def amended_record(where, stopped=(), factor=1.0) -> Path:
    rows = gate_rows(stopped, factor)
    path = where["runs"] / "study5_records" / "beta0-t" / m.AMENDED
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": m.AMENDED_KIND, "gate": sb.amended_gate(rows, TARGETS),
                                "registered": beta0.drift_gate(rows, TARGETS)}))  # fmt: skip
    return path


def orders(where, stopped=(50005,), rows=None):
    main(where, rows)  # writes the rate record
    rate = where["runs"] / "study5_records" / "beta0-t" / m.RATE
    out = where["runs"] / "campaigns" / "study5-beta0-orders"
    return m.orders_plan(out=out, archive_id="beta0-t", rate_path=rate,
                         gate_path=amended_record(where, stopped), **where)  # fmt: skip


def test_the_pilot_plan(where, tmp_path):
    plan = pilot(where)
    assert [b.name for b in plan.blocks] == ["engineering", "fetch", "pilot", "close"]
    check = next(s for s in steps(plan) if s.name == "engineering/check")
    assert check.on_fail == "stop" and check.gate == "engineering"
    fetched = [s for s in steps(plan, "fetch") if s.name.endswith("/streams")]
    assert [int(s.argv[s.argv.index("--seeds") + 1]) for s in fetched] == list(beta0.SEEDS)
    (name,) = [n for n in plan.campaigns if n.endswith("-pilot")]
    runs = campaign_runs(plan, tmp_path, name)
    assert len(runs) == 8 and {r.seed for r in runs} == set(beta0.PILOT_SEEDS)
    for run in runs:
        cfg = load_run_config(beta0.PILOT_CONFIG, [*run.overrides, f"seed={run.seed}"])
        assert config_hash(cfg) == run.config_hash and cfg.eval.k_only is True
        assert cfg.optimizer.beta1 == 0.0 and cfg.study5.role == "beta0"
    names = [s.name for s in steps(plan, "pilot")]
    assert names.index("pilot/launch") < names.index("pilot/table") < names.index("pilot/rate")
    for step in ("pilot/table", "pilot/rate"):
        assert next(s for s in steps(plan) if s.name == step).on_fail == "stop"


def _checked_runs(plan, tmp_path) -> list:
    eta0 = beta0.rate_choice(table_rows(), TARGETS)["eta0"]
    out = []
    for name in plan.campaigns:
        for run in campaign_runs(plan, tmp_path, name):
            cfg = load_run_config(beta0.CONFIG, [*run.overrides, f"seed={run.seed}"])
            assert config_hash(cfg) == run.config_hash and cfg.optimizer.lr == eta0
            assert cfg.optimizer.beta1 == 0.0 and not cfg.eval.get("k_only", False)
            out.append(run)
    return out


def test_the_main_plan_from_its_rate(where, tmp_path):
    plan = main(where)
    assert [b.name for b in plan.blocks] == ["stream", "close"]
    names = [s.name for s in steps(plan, "stream")]
    assert names[1] == "stream/check-rate"
    assert (names.index("stream/launch") < names.index("stream/gate")
            < names.index("stream/gate-amended") < names.index("stream/archive"))  # fmt: skip
    for name in ("stream/gate", "stream/gate-amended"):
        assert next(s for s in steps(plan) if s.name == name).on_fail == "stop"
    archive = next(s for s in steps(plan) if s.name == "stream/archive")
    assert all(any(f.endswith(n) for f in archive.argv) for n in (m.GATE, m.AMENDED))
    runs = _checked_runs(plan, tmp_path)
    assert len(runs) == 6 and {r.seed for r in runs} == set(beta0.SEEDS)


@pytest.mark.parametrize(("stopped", "total"), [((), 29), ((50005,), 24), ((50003,), 25)])
def test_the_orders_plan_runs_the_seeds_with_a_value(where, tmp_path, stopped, total):
    plan = orders(where, stopped)
    assert [b.name for b in plan.blocks] == ["orders", "close"]
    runs = _checked_runs(plan, tmp_path)
    assert len(runs) == total and not {r.seed for r in runs} & set(stopped)
    groups = [s.name for s in steps(plan, "orders") if s.name.endswith("/archive")]
    assert groups == [f"orders/s{s}/archive" for s in beta0.SEEDS if s not in stopped]
    check = next(s for s in steps(plan) if s.name == "orders/check-gate")
    seeds = [int(check.argv[i + 1]) for i, a in enumerate(check.argv) if a == "--seed"]
    assert seeds == [s for s in beta0.SEEDS if s not in stopped]


def test_the_orders_plan_needs_a_passed_amended_gate_of_its_archive(where):
    main(where)
    rate = where["runs"] / "study5_records" / "beta0-t" / m.RATE
    out = where["runs"] / "campaigns" / "o"
    failed = amended_record(where, stopped=(50004, 50005))
    with pytest.raises(camp.StepError, match="did not pass"):
        m.orders_plan(out=out, archive_id="beta0-t", rate_path=rate, gate_path=failed, **where)
    other = where["runs"] / "elsewhere" / m.AMENDED
    other.parent.mkdir(parents=True)
    other.write_text(amended_record(where).read_text())
    with pytest.raises(camp.StepError, match="archive"):
        m.orders_plan(out=out, archive_id="beta0-t", rate_path=rate, gate_path=other, **where)


def test_the_main_plan_refuses_a_rate_that_is_not_the_rules(where):
    path = rate_record(where["runs"] / "study5_records" / "beta0-t" / m.RATE)
    record = json.loads(path.read_text())
    record["choice"]["eta0"] = 2.0e-05
    path.write_text(json.dumps(record))
    with pytest.raises(camp.StepError, match="rule"):
        m.read_rate(path)
    with pytest.raises(camp.StepError, match="no"):
        main(where, table_rows(lambda s, lr: 1.0))


def _run_dir(runs: Path, campaign: str, label: str, run_id: str, *, outcome="done", steps=25,
             stop="steps", k=5.0) -> None:  # fmt: skip
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


def test_the_table_and_rate_steps(tmp_path):
    runs = tmp_path / "runs"
    argv = ["table", "--runs", str(runs), "--campaign", "c", "--gpu-code", GPU]
    for seed in beta0.PILOT_SEEDS:
        for level, lr in beta0.GRID:
            label = f"{level}/s{seed}"
            _run_dir(runs, "c", label, f"r{seed}-{level}", k=3.0 * (lr / 1.3e-05) ** 2)
            argv += ["--run", str(seed), level, label]
    table = tmp_path / "t.json"
    assert m.main([*argv, "--out", str(table)]) == 0
    rate = tmp_path / "rate.json"
    assert m.main(["rate", "--table", str(table), "--out", str(rate)]) == 0
    record = m.read_rate(rate)
    assert record["choice"]["eta0"] == beta0.rate_choice(table_rows(), TARGETS)["eta0"]
    digest = camp.sha256_of(rate)
    assert m.main(["check-rate", "--rate", str(rate), "--sha256", digest]) == 0
    assert m.main(["check-rate", "--rate", str(rate), "--sha256", "0" * 64]) == 1


def test_the_rate_step_fails_without_a_bracket(tmp_path):
    runs = tmp_path / "runs"
    argv = ["table", "--runs", str(runs), "--campaign", "c", "--gpu-code", GPU]
    for seed in beta0.PILOT_SEEDS:
        for level, _ in beta0.GRID:
            _run_dir(runs, "c", f"{level}/s{seed}", f"r{seed}-{level}", k=1.0)
            argv += ["--run", str(seed), level, f"{level}/s{seed}"]
    table = tmp_path / "t.json"
    assert m.main([*argv, "--out", str(table)]) == 0
    assert m.main(["rate", "--table", str(table), "--out", str(tmp_path / "r.json")]) == 1


def _stream_runs(runs: Path, campaign: str, labels, *, stopped=(), factor=1.0,
                 stop="gradient_spike", outcome="done") -> None:  # fmt: skip
    for seed, label in zip(beta0.SEEDS, labels, strict=True):
        if seed in stopped:
            _run_dir(runs, campaign, label, f"r{seed}", steps=8, stop=stop, k=None,
                     outcome=outcome)  # fmt: skip
        else:
            _run_dir(runs, campaign, label, f"r{seed}", k=TARGETS[seed] * factor)


@pytest.mark.parametrize(("factor", "stopped", "registered", "amended"), [
    (1.0, (), True, True),
    (1.3, (), False, False),
    (0.75, (), False, False),
    (1.0, (50005,), False, True),
    (1.0, (50004, 50005), False, False),
])  # fmt: skip
def test_the_gate_steps(tmp_path, factor, stopped, registered, amended):
    """``gate`` writes the registered verdict and no longer stops; ``gate-amended`` decides."""
    runs = tmp_path / "runs"
    labels = [f"s/{seed}" for seed in beta0.SEEDS]
    _stream_runs(runs, "c", labels, stopped=stopped, factor=factor)
    argv = ["--runs", str(runs), "--campaign", "c"]
    for seed, label in zip(beta0.SEEDS, labels, strict=True):
        argv += ["--run", str(seed), label]
    assert m.main(["gate", *argv, "--out", str(tmp_path / "g.json")]) == 0
    gate = json.loads((tmp_path / "g.json").read_text())
    assert gate["kind"] == m.GATE_KIND and gate["gate"]["passed"] is registered
    code = m.main(["gate-amended", *argv, "--out", str(tmp_path / "a.json")])
    assert code == (0 if amended else 1)
    record = json.loads((tmp_path / "a.json").read_text())
    assert record["kind"] == m.AMENDED_KIND and record["gate"]["passed"] is amended
    assert record["registered"] == gate["gate"] and "addendum" in record["rule"]


def test_run_refuses_another_archive_id(tmp_path):
    assert m.main(["run", "--out", str(tmp_path / "o"), "--archive-id", "dose-a-x",
                   "--checkpoints", "/x", "--dry-run"]) == 1  # fmt: skip


def test_the_orders_start_with_a_gate_check_run_at_every_start(where):
    plan = orders(where)
    orders_ = steps(plan, "orders")
    check = orders_[1]
    assert check.name == "orders/check-gate" and check.always and check.on_fail == "stop"
    assert check.outputs == () and orders_[0].kind == "deadline"
    assert all("launch" not in s.name for s in orders_[:2])


def test_the_restart_after_the_registered_gate_failed_on_a_scientific_stop(where, tmp_path):
    """The registered gate failed on 50005's gradient spike and its gate.json is restored into a
    new attempt: the gate step keeps it as written (its output exists), the amended gate step
    writes gate-amended.json and passes, and the orders plan built from it runs five seeds; its
    check, run at every start, recomputes both verdicts from the runs. A forged record, other
    seeds or a stop turned technical fail it."""
    import os
    import sys

    plan = main(where)
    eta0 = beta0.rate_choice(table_rows(), TARGETS)["eta0"]
    runs = where["runs"]
    (stream,) = [n for n in plan.campaigns if n.endswith("-stream")]
    assert stream == m.stream_campaign("beta0-t")
    labels = [camp.label(beta0.cell(seed, "stream", eta0)) for seed in beta0.SEEDS]
    _stream_runs(runs, stream, labels, stopped=(50005,))
    gate_step = next(s for s in steps(plan) if s.name == "stream/gate")
    amended_step = next(s for s in steps(plan) if s.name == "stream/gate-amended")
    gate_path = Path(gate_step.outputs[0])
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    original = {"kind": m.GATE_KIND, "rows": gate_rows((50005,)),
                "gate": beta0.drift_gate(gate_rows((50005,)), TARGETS)}  # fmt: skip
    assert original["gate"]["passed"] is False
    gate_path.write_text(json.dumps(original))
    written = gate_path.read_bytes()
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}

    def run(step, status, plan_=plan):
        return camp.run_step(step, status, plan=plan_, journal=tmp_path / "journal.jsonl",
                             logs=tmp_path / "logs", python=[sys.executable], cwd=REPO_ROOT,
                             env=env)  # fmt: skip

    (tmp_path / "logs").mkdir()
    assert run(gate_step, {}) == "kept" and gate_path.read_bytes() == written
    assert run(amended_step, {}) == "done"
    amended_path = Path(amended_step.outputs[0])
    amended = json.loads(amended_path.read_text())["gate"]
    assert amended["passed"] is True and amended["stopped"] == [50005]
    rate = runs / "study5_records" / "beta0-t" / m.RATE
    later = m.orders_plan(out=runs / "campaigns" / "study5-beta0-orders", archive_id="beta0-t",
                          rate_path=rate, gate_path=amended_path, **where)  # fmt: skip
    check = next(s for s in steps(later) if s.name == "orders/check-gate")
    assert run(check, {}, later) == "done"
    # the amended record forged: the runs still decide
    record = json.loads(amended_path.read_text())
    forged = json.loads(json.dumps(record))
    forged["gate"]["seeds"] = list(beta0.SEEDS)
    amended_path.write_text(json.dumps(forged))
    assert run(check, {"orders/check-gate": "done"}, later) == "failed"
    amended_path.write_text(json.dumps(record))
    # the registered record changed after the fact
    gate_path.write_text(json.dumps(original | {"gate": original["gate"] | {"passed": True}}))
    assert run(check, {"orders/check-gate": "done"}, later) == "failed"
    gate_path.write_bytes(written)
    assert run(check, {"orders/check-gate": "done"}, later) == "done"
    # the stopped run's record turned technical (an interruption): the amended gate fails
    terminal = runs / "study5" / "r50005" / "terminal.json"
    terminal.write_text(json.dumps({"steps": 8, "stop": {"reason": "interrupted"}}))
    assert run(check, {"orders/check-gate": "done"}, later) == "failed"


def test_check_gate_needs_the_runs_both_records_and_the_plans_seeds(tmp_path):
    runs = tmp_path / "runs"
    gate, amended = tmp_path / "g.json", tmp_path / "a.json"
    labels = [f"s/{seed}" for seed in beta0.SEEDS]
    argv = ["check-gate", "--runs", str(runs), "--campaign", "c", "--gate", str(gate),
            "--amended", str(amended)]  # fmt: skip
    for seed, label in zip(beta0.SEEDS, labels, strict=True):
        argv += ["--run", str(seed), label]
    five = [a for s in beta0.SEEDS[:5] for a in ("--seed", str(s))]
    assert m.main([*argv, *five]) == 1  # no runs: no end
    _stream_runs(runs, "c", labels, stopped=(50005,))
    assert m.main([*argv, *five]) == 1  # no records
    rows = gate_rows((50005,))
    gate.write_text(json.dumps({"kind": m.GATE_KIND, "gate": beta0.drift_gate(rows, TARGETS)}))
    amended.write_text(json.dumps({"kind": m.AMENDED_KIND, "gate": sb.amended_gate(rows, TARGETS),
                                   "registered": beta0.drift_gate(rows, TARGETS)}))  # fmt: skip
    assert m.main([*argv, *five]) == 0
    assert m.main([*argv, *five, "--seed", "50005"]) == 1  # a seed without a value
    assert m.main([*argv, *five[:-2]]) == 1  # a seed with a value left out


def test_a_conflict_end_gives_no_k(tmp_path):
    runs = tmp_path / "runs"
    _run_dir(runs, "c", "x/s1", "r1", outcome="conflict", k=5.0)
    out = tmp_path / "t.json"
    assert m.main(["table", "--runs", str(runs), "--campaign", "c", "--gpu-code", GPU,
                   "--run", "1", "1.0", "x/s1", "--out", str(out)]) == 0  # fmt: skip
    row = json.loads(out.read_text())["rows"][0]
    assert row["outcome"] == "conflict" and row["complete"] is False and row["K"] is None
