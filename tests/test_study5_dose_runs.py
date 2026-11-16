"""Study 5's dose runs on CPU (docs/study5-dose-registration.md): ``train.rate_schedule``
(study5_options) in the scheduled run, the ``dose`` role of the protocol, the dose config's
identity with the fresh F run's, and the scan of a dose run at a = −0.5 alone (line_scan).
Every run is a tiny fixed-policy F run of three updates (tests/test_study5_options.py)."""

from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf
from test_golden_single_process import snapshot
from test_line_scan import argv, files_of, script, spy
from test_schedule import scheduled
from test_study5_options import F_RUN, inputs_of, own_runs_dir, ready
from test_train import env as env  # noqa: F401  (the fixture)
from test_train import read_jsonl

from zero_field_refusal import line_scan, study5_options
from zero_field_refusal.config import config_hash
from zero_field_refusal.evaluation import line_scan_condition
from zero_field_refusal.models import freeze
from zero_field_refusal.protocol import ProtocolError, load_run_config
from zero_field_refusal.schedule import run_scheduled

LR = 1e-3  # F_RUN's optimizer.lr: the late rate


def dose(early_steps: int, early_lr: float) -> tuple[str, ...]:
    return ("+study5.role=dose",
            f"+train.rate_schedule={{early_steps: {early_steps}, early_lr: {early_lr!r}}}")  # fmt: skip


def run(env, tmp_path, monkeypatch, name: str, *extra: str, steps=(2,)):
    """A tiny F run in a runs directory of its own, θ_t saved at ``steps`` and the endpoint."""
    ckpt = tmp_path / f"ckpt-{name}"
    cfg = ready(env, scheduled(tmp_path, *F_RUN, *extra, f"tracking.endpoint_checkpoint={ckpt}",
                               f"tracking.step_checkpoints={list(steps)}", policy="short"))  # fmt: skip
    own_runs_dir(monkeypatch, tmp_path, name)
    run_dir = run_scheduled(cfg, env, run_id="f")
    states = {t: torch.load(ckpt / f"f-step-{t}.pt", weights_only=True) for t in steps}
    states[int(cfg.train.steps)] = torch.load(ckpt / "f.pt", weights_only=True)
    return run_dir, states, cfg


def test_step_rates_and_record(tmp_path) -> None:
    cfg = scheduled(tmp_path, *F_RUN, *dose(2, 5e-4), policy="short")
    assert study5_options.step_rates(cfg) == [5e-4, 5e-4, LR]
    assert study5_options.schedule_record(cfg) == {"early_steps": 2, "early_lr": 5e-4,
                                                   "late_lr": LR}  # fmt: skip
    plain = scheduled(tmp_path, *F_RUN, "+study5.role=fresh", policy="short")
    assert study5_options.step_rates(plain) is None


def test_a_schedule_at_the_runs_own_rate_is_the_plain_run_bitwise(env, tmp_path, monkeypatch) -> None:  # noqa: F811,E501
    """``early_lr`` = ``optimizer.lr``: every record file and θ_t equal the fresh run's bit for
    bit; meta.json adds only ``inputs.rate_schedule``."""
    torch.set_num_threads(1)
    plain_dir, plain, _ = run(env, tmp_path, monkeypatch, "plain", "+study5.role=fresh")
    dose_dir, same, _ = run(env, tmp_path, monkeypatch, "dose", *dose(1, LR))
    a, b = snapshot(plain_dir), snapshot(dose_dir)
    assert set(a) == set(b)
    inputs = json.loads(b["inputs"][0])
    assert inputs.pop("rate_schedule") == {"early_steps": 1, "early_lr": LR, "late_lr": LR}
    assert json.dumps(inputs, sort_keys=True) == a["inputs"][0]
    for name in a:
        if name != "inputs":
            assert a[name] == b[name], name
    for t in plain:
        for n, p in plain[t].items():
            assert torch.equal(p, same[t][n]), (t, n)


def test_the_early_updates_take_the_early_rate_and_the_rest_the_late(env, tmp_path, monkeypatch) -> None:  # noqa: F811,E501
    """Updates 1-2 at 5e-4, update 3 at 1e-3: through θ₂ the run is the plain run at 5e-4 bit for
    bit; its third update, from the same θ₂ and first moment, is that run's third update × 2;
    its first update is the plain run's at 1e-3 × 0.5."""
    torch.set_num_threads(1)
    slow_dir, slow, _ = run(env, tmp_path, monkeypatch, "slow", "+study5.role=fresh",
                            "optimizer.lr=0.0005")  # fmt: skip
    fast_dir, _, _ = run(env, tmp_path, monkeypatch, "fast", "+study5.role=fresh")
    dose_dir, ours, cfg = run(env, tmp_path, monkeypatch, "dose", *dose(2, 5e-4))
    assert inputs_of(dose_dir)["rate_schedule"]["early_steps"] == 2

    for n, p in slow[2].items():  # θ₂: the plain run at the early rate
        assert torch.equal(p, ours[2][n]), n
    theta2 = {n: p.double() for n, p in ours[2].items()}
    for n in theta2:
        ours_step = ours[3][n].double() - theta2[n]
        slow_step = slow[3][n].double() - theta2[n]
        ulps = 4 * torch.finfo(torch.float32).eps * theta2[n].abs().max().item()  # fp32 storage
        assert (ours_step - 2 * slow_step).abs().max().item() <= ulps, n
        assert not torch.equal(ours[3][n], slow[3][n]), n

    scalars = {name: read_jsonl(d / "update_scalars.jsonl")
               for name, d in (("slow", slow_dir), ("fast", fast_dir), ("dose", dose_dir))}  # fmt: skip
    norms = {name: [r["update_norm"] for r in rows] for name, rows in scalars.items()}
    assert norms["dose"][:2] == norms["slow"][:2]
    assert norms["dose"][2] == 2 * norms["slow"][2]  # the same m₃/d, twice the step size
    assert norms["dose"][0] == 0.5 * norms["fast"][0]  # the same m₁/d, half the step size
    def steps(run_dir):  # each step's loss and gradient at θ₀, θ₁, θ₂, without the time
        return [{k: v for k, v in r.items() if k != "time_s"}
                for r in read_jsonl(run_dir / "metrics.jsonl") if "loss" in r]  # fmt: skip

    assert steps(dose_dir) == steps(slow_dir) and len(steps(dose_dir)) == 3


def test_the_dose_config_is_the_fresh_f_runs_but_for_its_rates() -> None:
    """experiments/study5_dose: experiments/study5 with role dose and the schedule, so that a
    dose run reads the same streams, panels and prompts as the seed's fresh F run."""
    common = ["optimizer.lr=1.69e-05", "seed=50036"]
    ours = load_run_config("experiments/study5_dose",
                           [*common, "train.rate_schedule.early_lr=1.625e-05"])  # fmt: skip
    fresh = load_run_config("experiments/study5", [*common, "study5.role=fresh"])
    assert study5_options.step_rates(ours) == [1.625e-05] * 5 + [1.69e-05] * 20
    ours_dict = OmegaConf.to_container(ours, resolve=True)
    fresh_dict = OmegaConf.to_container(fresh, resolve=True)
    assert ours_dict["study5"] == {"role": "dose"} and fresh_dict["study5"] == {"role": "fresh"}
    assert ours_dict["train"].pop("rate_schedule") == {"early_steps": 5, "early_lr": 1.625e-05}
    for d in (ours_dict, fresh_dict):
        d.pop("study5")
    assert ours_dict == fresh_dict
    with pytest.raises((ProtocolError, ValueError), match="early_lr"):
        config_hash(load_run_config("experiments/study5_dose", common))


# ----------------------------------------------------------------------------------- the role


@pytest.mark.parametrize("overrides", [
    dose(1, 5e-4),
    (*dose(2, 1e-3), "tracking.endpoint_checkpoint=/x"),
])  # fmt: skip
def test_the_dose_role_accepts_its_schedule(tmp_path, overrides) -> None:
    scheduled(tmp_path, *F_RUN, *overrides, policy="short")


@pytest.mark.parametrize(("overrides", "message"), [
    (("+study5.role=dose",), "needs one"),
    ((*dose(1, 5e-4), "+train.order_seed=3"), "train.order_seed"),
    ((*dose(1, 5e-4), "+optimizer.preconditioner.source_seed=1"), "source_seed"),
    (("+study5.role=fresh", "+train.rate_schedule={early_steps: 1, early_lr: 0.0005}"),
     "dose role only"),
    (("+study5.role=control", "+train.order_seed=3",
      "+train.rate_schedule={early_steps: 1, early_lr: 0.0005}"), "dose role only"),
    (("+train.rate_schedule={early_steps: 1, early_lr: 0.0005}",), "study-5 role"),
    (dose(0, 5e-4), "early_steps"),
    (dose(3, 5e-4), "early_steps"),
    (("+study5.role=dose", "+train.rate_schedule={early_steps: 1, early_lr: -0.0005}"),
     "early_lr"),
    (("+study5.role=dose", "+train.rate_schedule={early_steps: 1, early_lr: 1}"), "early_lr"),
    (("+study5.role=dose", "+train.rate_schedule={early_steps: true, early_lr: 0.0005}"),
     "early_steps"),
    (("+study5.role=dose", "+train.rate_schedule={early_steps: 1}"), "exactly"),
    (("+study5.role=dose", "+train.rate_schedule={early_steps: 1, early_lr: 0.0005, k: 2}"),
     "exactly"),
    ((*dose(1, 5e-4), "optimizer=adamw", "optimizer.lr=1e-3"), "adam_frozen"),
])  # fmt: skip
def test_bad_dose_runs_are_refused(tmp_path, overrides, message) -> None:
    with pytest.raises(ProtocolError, match=message):
        scheduled(tmp_path, *F_RUN, *overrides, policy="short")


# ----------------------------------------------------------------------------------- the scan


def test_scan_points_are_a_subset_of_the_registered_ones() -> None:
    assert line_scan.scan_points(None) == line_scan.POINTS
    assert line_scan.scan_points([-0.5]) == (-0.5,)
    assert line_scan.scan_points([0.5, -1.0]) == (-1.0, 0.5)  # in the registered order
    for bad in ([], [-0.5, -0.5], [0.75]):
        with pytest.raises(ValueError):
            line_scan.scan_points(bad)


def test_a_dose_run_is_scanned_at_minus_one_half_alone(env, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811,E501
    torch.set_num_threads(1)
    overrides = [*F_RUN, *dose(1, 5e-4)]
    ckpt = tmp_path / "ckpt"
    cfg = ready(env, scheduled(tmp_path, *overrides, f"tracking.endpoint_checkpoint={ckpt}",
                               policy="short"))  # fmt: skip
    f_dir = run_scheduled(cfg, env, run_id="f-dose")
    before = files_of(f_dir)
    seen = spy(monkeypatch)
    module = script(env, monkeypatch)
    out = tmp_path / "records" / "scan.json"
    args = argv(tmp_path, f_dir, ckpt, out, overrides=overrides)
    assert module.main([*args, "--points", "-0.5"]) == 0
    record = json.loads(out.read_text())
    assert record["points"] == [-0.5] and record["evaluated"] == [line_scan_condition(-0.5)]
    assert record["rate_schedule"] == {"early_steps": 1, "early_lr": 5e-4, "late_lr": LR}
    assert [c for c, _ in seen] == ["line_scan_-0.5"]
    end = torch.load(ckpt / "f-dose.pt", weights_only=True)
    theta0 = {n: p.detach() for n, p in freeze(env.new_model(), cfg.train.freeze)}
    for n, p in seen[0][1].items():
        assert torch.equal(p, torch.add(theta0[n], end[n] - theta0[n], alpha=-0.5)), n
    assert files_of(f_dir) == before
    run_dir = tmp_path / "runs" / line_scan.EXPERIMENT / "scan"
    assert [r["a"] for r in read_jsonl(run_dir / "line_scan.jsonl")] == [-0.5]
    capsys.readouterr()
    bad = tmp_path / "records" / "bad.json"
    assert module.main([*argv(tmp_path, f_dir, ckpt, bad, overrides=overrides, run_id="b"),
                        "--points", "0.75"]) == 1  # fmt: skip
    assert not bad.exists()
