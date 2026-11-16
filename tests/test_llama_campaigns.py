"""Study 5's order repeat on Llama (scripts/study5_llama_campaigns.py): the three plans as data,
the checks record from a stage-0 summary, the pilot table and the rates record."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_study5_campaigns import COMMIT, GPU, campaign_runs, script_module, steps
from test_study5_dose_campaigns import _run_dir

from zero_field_refusal import llama_orders as lo
from zero_field_refusal.config import config_hash
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import load_run_config

camp = script_module(REPO_ROOT / "scripts" / "study5_campaigns.py", "study5_campaigns")
m = script_module(REPO_ROOT / "scripts" / "study5_llama_campaigns.py", "study5_llama_campaigns")
SUMMARY = {"path_field": {"passed": True}, "eos_fraction_at_cap": 0.9, "teacher_gap": {},
           "teacher_gap_proposed_thresholds": {"token_max_abs": 1.5e-3,
                                               "response_mean_signed_abs": 2.5e-5}}  # fmt: skip


@pytest.fixture
def where(tmp_path):
    return {"runs": tmp_path / "runs", "data": tmp_path / "data",
            "checkpoints": Path("/workspace/ckpt/llama"), "commit": COMMIT, "gpu_code": GPU,
            "archive_id": "llama-t"}  # fmt: skip


def records(tmp_path, k=lambda s, r: 4.84 * r / 2e-5):
    folder = tmp_path / "records"
    folder.mkdir(exist_ok=True)
    checks, rates = folder / m.CHECKS, folder / m.RATES
    checks.write_text(json.dumps(lo.checks_record(SUMMARY, "r0")))
    rows = [{"seed": s, "rate": r, "K": k(s, r), "complete": True}
            for s in lo.PILOT_SEEDS for r in lo.PILOT_RATES]  # fmt: skip
    rates.write_text(json.dumps({"kind": "study5-llama-rates", "rows": rows,
                                 "choice": lo.rate_choice(rows)}))  # fmt: skip
    return checks, rates


def test_the_checks_plan(where, tmp_path):
    plan = m.checks_plan(out=tmp_path / "o-checks", **where)
    assert [b.name for b in plan.blocks] == ["engineering", "instruments", "stage0", "theta0",
                                             "streams", "close"]  # fmt: skip
    names = {s.name: s for s in steps(plan)}
    assert names["engineering/check"].on_fail == "stop"
    for kind in m.INSTRUMENTS:
        s = names[f"instruments/{kind}"]
        assert s.on_fail == "stop" and s.argv[:2] == ("scripts/generate_teacher.py",
                                                      lo.THETA0_CONFIG)  # fmt: skip
    assert names["stage0/checks"].argv == ("scripts/stage0_checks.py", lo.STAGE0_CONFIG)
    assert names["stage0/record"].on_fail == "stop"
    streams = [s for s in steps(plan, "streams") if s.name.endswith("/training")]
    assert [int(s.argv[s.argv.index("--seeds") + 1]) for s in streams] == list(lo.SEEDS)
    assert all(s.argv[s.argv.index("--responses") + 1] == "500" for s in streams)
    (theta0,) = campaign_runs(plan, tmp_path, "study5-llama-t-theta0")
    assert theta0.entry == "theta0"


def test_the_pilot_and_main_plans_from_their_records(where, tmp_path):
    checks, rates = records(tmp_path)
    pilot = m.pilot_plan(out=tmp_path / "o-pilot", checks=checks, **where)
    runs = campaign_runs(pilot, tmp_path, m.pilot_campaign("llama-t"))
    assert len(runs) == len(lo.PILOT_SEEDS) * len(lo.PILOT_RATES)
    for run in runs:
        cfg = load_run_config(lo.PILOT_CONFIG, list(run.overrides))
        assert config_hash(cfg) == run.config_hash and cfg.eval.k_only is True
        assert cfg.train.order_seed == run.seed + 1000
        assert cfg.checks.teacher_gap.token_max_abs == 1.5e-3
    main = m.main_plan(out=tmp_path / "o-main", checks=checks, rates=rates, **where)
    assert len(main.campaigns) == len(lo.SEEDS)
    choice = json.loads(rates.read_text())["choice"]
    for seed in lo.SEEDS:
        runs = campaign_runs(main, tmp_path, f"study5-llama-t-main-s{seed}")
        found = {(r.overrides[0], r.overrides[1]) for r in runs}
        assert found == {(f"optimizer.lr={choice[e]!r}", f"train.order_seed={seed + off}")
                         for _, off in lo.ORDER_OFFSETS for e in ("eta_l", "eta_u")}  # fmt: skip
        for run in runs:
            cfg = load_run_config(lo.CONFIG, list(run.overrides))
            assert config_hash(cfg) == run.config_hash and not cfg.eval.get("k_only", False)


def test_a_rates_record_that_is_not_the_rules_is_refused(where, tmp_path):
    checks, rates = records(tmp_path)
    record = json.loads(rates.read_text())
    record["choice"]["eta_l"] = 9.9e-06
    rates.write_text(json.dumps(record))
    with pytest.raises(camp.StepError, match="rule"):
        m.main_plan(out=tmp_path / "o", checks=checks, rates=rates, **where)


def test_the_checks_record_needs_one_passed_stage0(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / m.STAGE0_EXPERIMENT / "r0"
    run_dir.mkdir(parents=True)
    out = tmp_path / "checks.json"
    assert m.main(["checks-record", "--runs", str(runs), "--out", str(out)]) == 1  # no summary
    (run_dir / "summary.json").write_text(json.dumps(SUMMARY))
    assert m.main(["checks-record", "--runs", str(runs), "--out", str(out)]) == 0
    assert m.read_checks(out)["thresholds"]["token_max_abs"] == 1.5e-3
    failed = dict(SUMMARY, path_field={"passed": False})
    (run_dir / "summary.json").write_text(json.dumps(failed))
    assert m.main(["checks-record", "--runs", str(runs), "--out", str(tmp_path / "x.json")]) == 1


def test_the_pilot_table_and_the_rates(tmp_path):
    runs = tmp_path / "runs"
    args = []
    for i, (seed, rate) in enumerate((s, r) for s in lo.PILOT_SEEDS for r in lo.PILOT_RATES):
        label = f"p{i}/s{seed}"
        _run_dir(runs, "c", label, f"r{i}", k=4.84 * rate / 2e-5)
        args += ["--run", str(seed), repr(rate), label]
    table, rates = tmp_path / "t.json", tmp_path / "r.json"
    assert m.main(["table", "--runs", str(runs), "--campaign", "c", *args, "--out", str(table)]) == 0
    assert m.main(["rates", "--table", str(table), "--out", str(rates)]) == 0
    assert m.read_rates(rates)["choice"]["eta_l"] == 2e-05


def test_every_command_of_the_checks_plan_passes_its_own_protocol_check(where, tmp_path):
    """Each stream command loads its config as scripts/generate_teacher.py does
    (``load_run_config``), stage 0 as scripts/stage0_checks.py does (``check_run_config``):
    neither may refuse for the thresholds stage 0 has not set yet."""
    from zero_field_refusal.config import load_config
    from zero_field_refusal.protocol import check_run_config

    plan = m.checks_plan(out=tmp_path / "o-checks", **where)
    streams = [s for s in steps(plan) if s.argv[0] == "scripts/generate_teacher.py"]
    assert len(streams) == len(m.INSTRUMENTS) + len(lo.SEEDS)
    for step in streams:
        args = list(step.argv[1:])
        config, overrides = args[0], [a for a in args[1:] if "=" in a and not a.startswith("--")]
        cfg = load_run_config(config, overrides)
        assert cfg.run_type == "theta0" and cfg.model.name == "meta-llama/Llama-3.2-1B-Instruct"
    (stage0,) = [s for s in steps(plan) if s.argv[0] == "scripts/stage0_checks.py"]
    check_run_config(load_config(stage0.argv[1]))


def test_the_streams_config_locates_the_runs_streams():
    """``run_type=theta0`` and the placeholders aside, the streams' config is the runs'."""
    from omegaconf import OmegaConf

    run = load_run_config(lo.CONFIG, ["optimizer.lr=2.2e-05", "seed=60004",
                                      "train.order_seed=61004",
                                      *lo.threshold_overrides(lo.checks_record(SUMMARY, "r"))])  # fmt: skip
    streams = load_run_config(*m.theta0_args(60004)[:1], list(m.theta0_args(60004)[1:]))
    a, b = (OmegaConf.to_container(c, resolve=True) for c in (run, streams))
    for x in (a, b):
        x.pop("run_type"), x.pop("checks"), x["optimizer"].pop("lr"), x["train"].pop("order_seed")
    assert a == b


def test_the_certificate_is_archived_with_the_data_before_stage0s_receipt(where, tmp_path):
    import torch

    from zero_field_refusal.prechecks import environment_fingerprint, record_path

    plan = m.checks_plan(out=tmp_path / "o-checks", **where)
    stage0 = next(b for b in plan.blocks if b.name == "stage0")
    names = [s.name for s in stage0.steps]
    assert names.index("stage0/archive-certificate") < names.index("stage0/receipt")
    step = stage0.steps[names.index("stage0/archive-certificate")]
    argv = list(step.argv)
    assert argv[argv.index("--remote") + 1] == "data" and step.on_fail == "stop"
    assert argv[argv.index("--path") + 1] == m.CHECKS_DIR
    run = load_run_config(lo.CONFIG, ["optimizer.lr=2.2e-05", "seed=60004", "train.order_seed=61004",
                                      *lo.threshold_overrides(lo.checks_record(SUMMARY, "r"))])  # fmt: skip
    certificate = record_path(Path("DATA"), environment_fingerprint(run, torch.device("cpu")))
    assert certificate.parent.relative_to("DATA").as_posix() == m.CHECKS_DIR
