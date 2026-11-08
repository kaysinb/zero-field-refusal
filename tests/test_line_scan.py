"""Study 5's line scan as a run of its own (docs/study5-preregistration.md, "Descriptive
measurements", "Line scans"; ``line_scan``, scripts/study5_scan.py) on a tiny fresh F run with
its endpoint checkpoint: each point θ₀ + aΔθ exactly, evaluated in full on the F run's prompts
under ``line_scan_{a:+g}``, the F run untouched; a non-finite point recorded; a cut-off scan's
points missing, never labeled; the refusals."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest
import torch
from omegaconf import OmegaConf
from test_schedule import scheduled
from test_study3_campaigns import script_module
from test_study5_options import F_RUN, ready
from test_train import APPENDABLE, SMALL, read_jsonl
from test_train import env as env  # noqa: F401  (the fixture)

from zero_field_refusal import line_scan
from zero_field_refusal.evaluation import SCAN_RUN_TYPE, line_scan_condition
from zero_field_refusal.models import freeze
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.run_judging import check_responses, response_conditions, unlabelable
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.tracking import Diverged
from zero_field_refusal.train import Evaluator

FRESH = [*F_RUN, "+study5.role=fresh"]
CONDITIONS = [line_scan_condition(a) for a in line_scan.POINTS]


def fresh_run(env, tmp_path):
    """A tiny fresh F run with its endpoint checkpoint: its directory and the checkpoints'."""
    ckpt = tmp_path / "ckpt"
    cfg = ready(env, scheduled(tmp_path, *FRESH, f"tracking.endpoint_checkpoint={ckpt}",
                               policy="short"))  # fmt: skip
    return cfg, run_scheduled(cfg, env, run_id="f-fresh"), ckpt


def files_of(run_dir) -> dict[str, str]:
    return {p.relative_to(run_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run_dir.rglob("*")) if p.is_file()}  # fmt: skip


def script(env, monkeypatch):
    module = script_module(REPO_ROOT / "scripts" / "study5_scan.py", "s5_scan")
    monkeypatch.setattr(module, "environment", lambda cfg, device: env)
    return module


def argv(tmp_path, f_dir, ckpt, out, *, run_id="scan", overrides=FRESH):
    config = [str(tmp_path / "study2.yaml"), *SMALL, *APPENDABLE, "teacher.chunk_size=12"]
    return [*config, *overrides, "--f-run", str(f_dir), "--checkpoints", str(ckpt),
            "--out", str(out), "--run-id", run_id]  # fmt: skip


def spy(monkeypatch, fail=None):
    """The trainable tensors at every evaluation, by condition; ``fail(condition)`` may raise."""
    seen, original = [], Evaluator.__call__

    def call(self, model, condition, **kwargs):
        seen.append((condition, {n: p.detach().clone()
                                 for n, p in freeze(model, self.cfg.train.freeze)}))  # fmt: skip
        if fail is not None:
            fail(condition)
        return original(self, model, condition, **kwargs)

    monkeypatch.setattr(Evaluator, "__call__", call)
    return seen


def test_the_scan_run_sets_each_point_exactly_and_leaves_the_f_run(
    env, tmp_path, monkeypatch, capsys
) -> None:  # noqa: F811
    torch.set_num_threads(1)
    cfg, f_dir, ckpt = fresh_run(env, tmp_path)
    before = files_of(f_dir)
    seen = spy(monkeypatch)
    module = script(env, monkeypatch)
    out = tmp_path / "records" / "scan.json"
    assert module.main(argv(tmp_path, f_dir, ckpt, out)) == 0
    record = json.loads(out.read_text())

    # each point θ₀ + a(θ₃ − θ₀), from a fresh θ₀ and the checkpoint, in fp32, in order
    end = torch.load(ckpt / "f-fresh.pt", weights_only=True)
    theta0 = {n: p.detach() for n, p in freeze(env.new_model(), cfg.train.freeze)}
    assert [c for c, _ in seen] == CONDITIONS
    for (condition, tensors), a in zip(seen, line_scan.POINTS, strict=True):
        assert condition == line_scan_condition(a)
        for n, p in tensors.items():
            assert p.dtype == torch.float32
            assert torch.equal(p, torch.add(theta0[n], end[n] - theta0[n], alpha=a)), (a, n)
    assert CONDITIONS == ["line_scan_-1", "line_scan_-0.5", "line_scan_-0.25", "line_scan_+0.25",
                          "line_scan_+0.5"]  # fmt: skip
    assert files_of(f_dir) == before  # nothing of the F run written

    run_dir = tmp_path / "runs" / line_scan.EXPERIMENT / "scan"
    assert (record["kind"], record["run_id"], record["experiment"]) == (
        "study5-scan", "scan", "study5_scan")  # fmt: skip
    assert record["evaluated"] == record["conditions"] == CONDITIONS
    assert record["f_run"]["run_id"] == "f-fresh"
    meta = json.loads((f_dir / "meta.json").read_text())
    assert record["checkpoint"]["sha256"] == meta["inputs"]["endpoint_checkpoint"]["sha256"]
    assert record["theta0_trainable_sha256"] == meta["inputs"]["theta0_trainable_sha256"]
    assert record["f_config_hash"] == meta["config_hash"] != record["config_hash"]
    rows = read_jsonl(run_dir / "line_scan.jsonl")
    assert [r["a"] for r in rows] == list(line_scan.POINTS) and all(r["evaluated"] for r in rows)
    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    assert run_cfg.run_type == SCAN_RUN_TYPE and run_cfg.study5.scan.f_run == "f-fresh"

    # every point answers the F run's step-3 prompts and measures, judged under its condition
    measures = read_jsonl(run_dir / "observations" / "measurements.jsonl")
    own = read_jsonl(f_dir / "observations" / "measurements.jsonl")
    want = Counter((r["measure"], r["item_id"]) for r in own if r["condition"] == "step-3")
    for condition in CONDITIONS:
        got = Counter((r["measure"], r["item_id"]) for r in measures if r["condition"] == condition)
        assert got == want, condition
    assert response_conditions(run_cfg, run_dir) == CONDITIONS
    assert unlabelable(run_dir, run_cfg) is None
    ids = [item["id"] for item in env.prompt_set(cfg.eval.harmful.prompt_set)]
    responses = read_jsonl(run_dir / "responses" / f"{cfg.eval.harmful.prompt_set}.jsonl")
    check_responses(run_dir, responses, CONDITIONS, ids)

    printed = capsys.readouterr().out
    assert record["checkpoint"]["sha256"] in printed
    for item in env.prompt_set(cfg.eval.harmful.prompt_set):
        assert item["text"] not in printed
    assert module.main(argv(tmp_path, f_dir, ckpt, out)) == 1  # never overwritten
    assert json.loads(out.read_text()) == record


def test_a_non_finite_point_is_recorded_and_the_scan_goes_on(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    torch.set_num_threads(1)
    _, f_dir, ckpt = fresh_run(env, tmp_path)

    def fail(condition):
        if condition == "line_scan_-1":
            raise Diverged("K is not finite")

    spy(monkeypatch, fail)
    out = tmp_path / "records" / "scan.json"
    assert script(env, monkeypatch).main(argv(tmp_path, f_dir, ckpt, out)) == 0
    record = json.loads(out.read_text())
    assert record["conditions"] == CONDITIONS and record["evaluated"] == CONDITIONS[1:]
    assert (record["rows"][0]["evaluated"], record["rows"][0]["reason"]) == (False, "nonfinite_k")
    run_dir = tmp_path / "runs" / line_scan.EXPERIMENT / "scan"
    assert response_conditions(OmegaConf.load(run_dir / "config.yaml"), run_dir) == CONDITIONS[1:]


def test_a_scan_cut_off_keeps_its_rows_and_is_not_labeled(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    """A scan that stops at its third point (the Pod cut it off): the rows of the two points it
    finished stay, the run is not labeled (its points are missing in the analysis), nothing of
    the F run changes and no record is written."""
    torch.set_num_threads(1)
    _, f_dir, ckpt = fresh_run(env, tmp_path)
    before = files_of(f_dir)

    def fail(condition):
        if condition == CONDITIONS[2]:
            raise RuntimeError("the Pod stopped")

    spy(monkeypatch, fail)
    out = tmp_path / "records" / "scan.json"
    with pytest.raises(RuntimeError, match="Pod stopped"):
        script(env, monkeypatch).main(argv(tmp_path, f_dir, ckpt, out))
    assert not out.exists()
    run_dir = tmp_path / "runs" / line_scan.EXPERIMENT / "scan"
    assert [r["condition"] for r in read_jsonl(run_dir / "line_scan.jsonl")] == CONDITIONS[:2]
    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    with pytest.raises(ValueError, match="incomplete"):
        response_conditions(run_cfg, run_dir)
    assert unlabelable(run_dir, run_cfg) is not None
    assert files_of(f_dir) == before


def test_the_scan_refuses_what_is_not_the_runs(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    torch.set_num_threads(1)
    _, f_dir, ckpt = fresh_run(env, tmp_path)
    module = script(env, monkeypatch)
    out = tmp_path / "records" / "scan.json"
    # a control's config: not a fresh run
    control = [*F_RUN, "+study5.role=control", "+train.order_seed=7"]
    assert module.main(argv(tmp_path, f_dir, ckpt, out, overrides=control)) == 1
    # another file under the run's checkpoint name, and no checkpoint at all
    other = tmp_path / "other"
    other.mkdir()
    state = torch.load(ckpt / "f-fresh.pt", weights_only=True)
    torch.save({name: t + 1.0 for name, t in state.items()}, other / "f-fresh.pt")
    assert module.main(argv(tmp_path, f_dir, other, out)) == 1
    assert module.main(argv(tmp_path, f_dir, tmp_path / "none", out)) == 1
    # a run that records another θ₀
    meta_path = f_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["inputs"]["theta0_trainable_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    assert module.main(argv(tmp_path, f_dir, ckpt, out)) == 1
    assert not out.exists()


def test_the_points_are_the_registered_ones() -> None:
    from zero_field_refusal.analysis import study5_analysis as sa

    assert line_scan.POINTS == sa.SCAN_POINTS == (-1.0, -0.5, -0.25, 0.25, 0.5)
