"""Study 5's runs from their making to the analysis's label reader, on the tiny model, every run
folder written by the real code: the fresh η_l run as pod 2's plan configures it, the scan run of
its endpoint (scripts/study5_scan.py), a ladder rung's run (scripts/study5_ladder.py) and θ₀'s run
(train.evaluate_theta0). The runs the launcher starts (η_l, θ₀) are judged as it judges a
finished run (``launch._disqualification``, ``launch._published``); every run is labeled by
``run_judging.label_runs`` with a fake WildGuard and read back by
``analysis.study5_analysis_cli``'s label reader under each of its conditions."""

from __future__ import annotations

import json
import math

import pytest
import torch
import yaml
from omegaconf import ListConfig, OmegaConf
from test_ladder import stub_k
from test_run_judging import WG_REFUSAL, FakeWildGuard, factory
from test_schedule import scheduled
from test_study5_campaigns import camp, script_module  # camp: scripts/study5_campaigns.py
from test_study5_options import F_RUN, ready
from test_train import APPENDABLE, SMALL
from test_train import env as env  # noqa: F401  (the fixture)

from zero_field_refusal import ladder, launch
from zero_field_refusal.analysis import study5_analysis as sa
from zero_field_refusal.analysis import study5_analysis_cli as cli
from zero_field_refusal.config import config_hash
from zero_field_refusal.evaluation import line_scan_condition
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import STUDY5_KEYS
from zero_field_refusal.run_judging import label_runs, response_conditions
from zero_field_refusal.runtime import configure_numerics
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.train import evaluate_theta0

SEED = 50000
CHECKPOINT = "tracking.endpoint_checkpoint="
SCAN_KEY = "eval.controls.line_scan.points"  # the in-run scan of an earlier design


def planned_cell(tmp_path, rate: float) -> dict:
    """Pod 2's fresh cell of seed 50000 at ``rate``, from the plan's launcher campaign."""
    plan = camp.pod2_plan(out=tmp_path / "campaign", runs=tmp_path / "plan-runs",
                          data=tmp_path / "plan-data", archive_id="chain", commit="c" * 40,
                          gpu_code="e" * 64, n=24)  # fmt: skip
    for text in plan.campaigns.values():
        data = yaml.safe_load(text)
        if data["header"]["block"] != "fresh":
            continue
        for cell in data["cells"]:
            if cell["seeds"] == [SEED] and camp.rate(rate) in cell["overrides"]:
                return cell
    raise AssertionError(f"pod 2's plan has no fresh cell of seed {SEED} at {rate}")


def study5_keys(cell: dict) -> list[str]:
    """The study-5 keys of the cell's resolved config, as overrides of the tiny run's config."""
    given = [o for o in cell["overrides"] if not o.startswith(CHECKPOINT)]
    cfg = camp_config(cell["config"], [*given, f"seed={SEED}"])
    out = [f"+study5.role={cfg.study5.role}"]
    for key in dict.fromkeys((*STUDY5_KEYS, SCAN_KEY)):
        value = OmegaConf.select(cfg, key, default=None)
        if isinstance(value, ListConfig):
            out.append(f"+{key}=[{','.join(str(v) for v in value)}]")
        elif value is not None:
            out.append(f"+{key}={value}")
    return out


def camp_config(name: str, overrides: list[str]):
    from zero_field_refusal.config import load_config

    return load_config(name, overrides)


def launched(cfg, config_path, overrides, *, entry: str = "train") -> launch.Run:
    """The launcher's run of the cell, as ``launch.load_campaign`` makes it."""
    return launch.Run(cell="chain", entry=entry, config=str(config_path),
                      overrides=tuple(overrides), seed=int(cfg.seed), experiment=cfg.experiment,
                      config_hash=config_hash(cfg),
                      prompt_set=str(cfg.eval.harmful.prompt_set))  # fmt: skip


def meta_of(run_dir) -> dict:
    return json.loads((run_dir / "meta.json").read_text())


def test_fresh_scan_ladder_and_theta0_runs_reach_the_analysis(env, tmp_path, monkeypatch):  # noqa: F811
    torch.set_num_threads(1)
    config_path = tmp_path / "study2.yaml"
    base = [*SMALL, *APPENDABLE, "teacher.chunk_size=12"]
    ckpt = tmp_path / "ckpt"

    # The fresh η_l run, configured as pod 2's plan configures it, with its endpoint checkpoint
    cell = planned_cell(tmp_path, camp.ETA_L)
    assert any(o.startswith(CHECKPOINT) for o in cell["overrides"]), "η_l saves its endpoint"
    keys = [*F_RUN, *study5_keys(cell)]
    overrides = [*keys, f"{CHECKPOINT}{ckpt}"]
    cfg = ready(env, scheduled(tmp_path, *overrides, policy="short"))
    f_dir = run_scheduled(cfg, env, run_id="eta-l", numerics_setup=configure_numerics(cfg, "cpu"))
    run = launched(cfg, config_path, [*base, *overrides])
    assert launch._disqualification(run, f_dir, meta_of(f_dir)) is None
    assert launch._published(f_dir)

    # The scan run of its endpoint: the registered points, a run of its own
    scan_script = script_module(REPO_ROOT / "scripts" / "study5_scan.py", "s5_scan_chain")
    monkeypatch.setattr(scan_script, "environment", lambda cfg, device: env)
    scan_out = tmp_path / "records" / "scan.json"
    argv = [str(config_path), *base, *keys, "--f-run", str(f_dir), "--checkpoints", str(ckpt),
            "--out", str(scan_out), "--run-id", "scan"]  # fmt: skip
    assert scan_script.main(argv) == 0
    scan_record = json.loads(scan_out.read_text())
    scan_dir = tmp_path / "runs" / scan_record["experiment"] / scan_record["run_id"]
    points = [line_scan_condition(a) for a in sa.SCAN_POINTS]
    assert scan_record["evaluated"] == points
    assert not launch._published(scan_dir)  # not a training run: the launcher never judges it

    # A ladder rung's run, its K stubbed so that K* is bracketed
    ladder_script = script_module(REPO_ROOT / "scripts" / "study5_ladder.py", "s5_ladder_chain")
    monkeypatch.setattr(ladder_script, "environment", lambda cfg, device: env)
    k_benign, box = stub_k(ladder.K_STAR, 1.3)
    box["norm"] = json.loads((f_dir / "monitor.jsonl").read_text().splitlines()[-1])[
        "displacement_norm"
    ]
    monkeypatch.setattr(ladder, "k_benign", k_benign)
    ladder_out = tmp_path / "records" / "ladder-linear.json"
    argv = [str(config_path), *base, *keys, "--rung", "linear", "--f-run", str(f_dir),
            "--out", str(ladder_out), "--run-id", "ladder-linear"]  # fmt: skip
    assert ladder_script.main(argv) == 0
    ladder_dir = tmp_path / "runs" / ladder.EXPERIMENT / "ladder-linear"

    # θ₀'s run, as the launcher's entry theta0 makes it
    theta0_overrides = [*F_RUN, "+study5.role=fresh", "run_type=theta0"]
    theta0_cfg = scheduled(tmp_path, *theta0_overrides, policy="short")
    theta0_dir = evaluate_theta0(theta0_cfg, env, run_id="theta0")
    run = launched(theta0_cfg, config_path, [*base, *theta0_overrides], entry="theta0")
    assert launch._disqualification(run, theta0_dir, meta_of(theta0_dir)) is None

    # Every run labeled, then read by the analysis's reader under each of its conditions
    def prompts(cfg):
        return {item["id"]: item["text"] for item in env.prompt_set(cfg.eval.harmful.prompt_set)}

    dirs = {f_dir: ["step-3"], scan_dir: points, ladder_dir: ["ladder_linear"],
            theta0_dir: ["theta0"]}  # fmt: skip
    for run_dir, conditions in dirs.items():
        assert response_conditions(OmegaConf.load(run_dir / "config.yaml"), run_dir) == conditions
    summaries = label_runs(list(dirs), "wildguard", prompts, factory(FakeWildGuard(WG_REFUSAL)))
    ids = list(prompts(cfg))
    for summary, (run_dir, conditions) in zip(summaries, dirs.items(), strict=True):
        assert summary == {"run_id": run_dir.name, "items": len(conditions) * len(ids),
                           "unparseable": 0}, summary  # fmt: skip
    monkeypatch.setattr(sa, "DENOMINATOR", len(ids))
    reader = cli.Reader({"runs": tmp_path / "runs"})
    labels = cli.Labels(reader, prompts)
    for run_dir, conditions in dirs.items():
        for condition in conditions:
            counts = labels.at(run_dir, condition)
            assert isinstance(counts, dict), (run_dir.name, condition, counts)
            assert counts == {"n": len(ids), "labeled": len(ids), "missing": 0, "harmful": 0,
                              "refusal": len(ids), "safe_non_refusal": 0}  # fmt: skip

    # The analysis reads the scan from the scan run, point by point
    rows = cli.scan_rows(reader, scan_dir)
    assert [rows[a]["condition"] for a in sa.SCAN_POINTS] == points
    assert all(math.isfinite(rows[a]["K_benign"]) for a in sa.SCAN_POINTS)


@pytest.mark.parametrize("rate", [camp.ETA_L, camp.ETA_U])
def test_the_fresh_runs_are_plain_f_runs(tmp_path, rate):
    """Neither fresh rate scans in its own run: the scan is a run of its own."""
    cell = planned_cell(tmp_path, rate)
    assert not any("line_scan" in key for key in study5_keys(cell))
