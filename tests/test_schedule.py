"""Study-2 schedules: the stopping rules alone, then whole runs on the tiny CPU model.

In the runs the monitor reads a scripted sequence of K, so each test fixes where levels are
crossed; training, the scoring panel and the stored responses are real.
"""

import gc
import json
import math
import os
from pathlib import Path

import pytest
import torch
from test_gpu import gpu_env as gpu_env  # noqa: F401  (the fixture)
from test_train import APPENDABLE, SMALL, prepare, read_jsonl
from test_train import env as env  # noqa: F401  (the fixture)

from zero_field_refusal.panels import Monitor
from zero_field_refusal.prechecks import CheckFailed
from zero_field_refusal.protocol import ProtocolError, load_run_config
from zero_field_refusal.runtime import StreamPrefix, extend_stream
from zero_field_refusal.schedule import (
    InstabilityGuard,
    LevelTracker,
    PlateauRule,
    UpdateScalars,
    run_scheduled,
)
from zero_field_refusal.teacher import StreamError

RULE = PlateauRule(from_step=1000, window=100, windows=6, span=1.10, drift=1.02, below_next=0.9)


def test_a_level_is_crossed_by_two_consecutive_readings_and_reached_once_evaluated() -> None:
    tracker = LevelTracker([5.0, 7.0, 8.0], readings=2)
    assert tracker.update(5.5) == []  # one reading is not enough
    assert tracker.update(4.9) == []
    assert tracker.update(5.2) == []
    assert tracker.update(7.5) == [5.0]  # 5.2 and 7.5: only 5 is under both
    assert tracker.update(7.6) == [5.0, 7.0]  # not evaluated yet: crossed, not reached
    tracker.reach([5.0, 7.0])
    assert tracker.update(9.0) == []  # 7.6 and 9.0: 8 is not under both
    assert tracker.update(9.2) == [8.0]
    tracker.reach([8.0])
    assert tracker.next_level is None and tracker.update(9.5) == []


def test_one_reading_pair_can_cross_several_levels() -> None:
    tracker = LevelTracker([5.0, 7.0, 8.0], readings=2)
    tracker.update(8.2)
    assert tracker.update(8.4) == [5.0, 7.0, 8.0]


def readings_every_5(values_by_window: list[float], end: int) -> list[tuple[int, float]]:
    """Monitor readings every 5 steps over the six 100-step windows ending at ``end``."""
    out = []
    for w, value in enumerate(values_by_window):
        start = end - 100 * (6 - w)
        out += [(s, value) for s in range(start + 5, start + 101, 5)]
    return out


def test_a_plateau_needs_flat_medians_below_the_next_level_from_step_1000() -> None:
    flat = [4.0, 4.1, 4.05, 4.1, 4.08, 4.1]
    assert RULE.plateaued(readings_every_5(flat, 1200), 1200, next_level=5.0)
    assert not RULE.plateaued(readings_every_5(flat, 900), 900, next_level=5.0)  # too early
    assert not RULE.plateaued(readings_every_5(flat, 1200), 1250, next_level=5.0)  # off-grid
    assert not RULE.plateaued(readings_every_5(flat, 1200), 1200, next_level=4.5)  # 4.1 ≥ 0.9·4.5
    rising = [4.0, 4.0, 4.0, 4.3, 4.3, 4.3]  # span 1.075, but the last half rose by 7.5 %
    assert not RULE.plateaued(readings_every_5(rising, 1200), 1200, next_level=6.0)
    wide = [3.6, 4.0, 4.0, 4.0, 4.0, 3.9]  # span 1.11
    assert not RULE.plateaued(readings_every_5(wide, 1200), 1200, next_level=6.0)
    gap = readings_every_5(flat, 1200)[20:]  # the first window has no reading
    assert not RULE.plateaued(gap, 1200, next_level=5.0)


def test_the_gradient_reference_starts_at_step_0_and_one_spike_suffices() -> None:
    guard = InstabilityGuard(grad_factor=10, grad_window=20, k_max=16)
    assert guard.gradient(loss=1.0, grad_norm=1.0) is None  # step 0: the reference begins
    assert guard.gradient(1.0, 10.0) is None  # exactly 10 × the median: not above
    assert guard.gradient(1.0, 2.0) is None
    assert guard.gradient(1.0, 25.0) == "gradient_spike"  # 25 > 10 × median(1, 10, 2) = 20
    assert guard.norms == [1.0, 10.0, 2.0]  # the spike does not join the reference


def test_the_logged_rule_records_a_spike_and_takes_the_update() -> None:
    # The SGD-momentum pilot's soft-target controls: the rule fires as before, but only records.
    guard = InstabilityGuard(grad_factor=10, grad_window=20, k_max=16, grad_rule="log")
    for norm in (1.0, 1.0, 1.0, 25.0, 1.0):
        assert guard.gradient(1.0, norm) is None
    assert guard.logged == [{"step": 3, "grad_norm": 25.0, "reference_median": 1.0}]
    assert guard.norms == [1.0, 1.0, 1.0, 1.0]  # the firing norm stays out of the reference
    assert guard.gradient(float("nan"), 1.0) == "nonfinite_loss"  # the other stops remain
    with pytest.raises(ValueError, match="grad_rule"):
        InstabilityGuard(grad_factor=10, grad_window=4, k_max=16, grad_rule="other")


def test_instability_reasons() -> None:
    guard = InstabilityGuard(grad_factor=10, grad_window=2, k_max=16)
    assert guard.gradient(float("nan"), 1.0) == "nonfinite_loss"
    assert guard.gradient(1.0, float("inf")) == "nonfinite_gradient"
    for norm in (1.0, 1.0, 10.0, 10.0):
        assert guard.gradient(1.0, norm) is None
    # The window holds the last two norms: median 10, so 80 passes (all four: median 5.5).
    assert guard.gradient(1.0, 80.0) is None
    assert guard.monitor(16.0) is None and guard.monitor(16.1) == "k_above_max"
    assert guard.monitor(float("nan")) == "nonfinite_k"


# Whole runs. Batch 4, chunks of 12 responses (each chunk's own path-field bound needs a few
# responses for its bootstrap; study 2 uses 1000), monitor every 5 steps.
def scheduled(tmp_path: Path, *overrides: str, policy: str = "discovery"):
    experiment = tmp_path / "study2.yaml"
    experiment.write_text(f"defaults: [experiments/smoke, eval/study2, schedule/{policy}]\n")
    return load_run_config(experiment, [*SMALL, *APPENDABLE, "teacher.chunk_size=12", *overrides])


K_RUN = ("train.steps=40", "schedule.levels=[0.5,1.0]", "schedule.plateau.from_step=20",
         "schedule.plateau.window=5", "schedule.plateau.windows=2")  # fmt: skip


@pytest.fixture
def scripted(monkeypatch):
    """Make the monitor read ``values`` in order (then repeat the last)."""

    def script(*values: float) -> list[float]:
        queue = list(values)

        def read(self, model):  # as the monitor does: its per-response reading too
            value = queue.pop(0) if len(queue) > 1 else queue[0]
            self.last = torch.full((len(self.primary),), value, dtype=torch.float64)
            return self.last.mean().item()

        uncached = Monitor.uncached

        def again(self, model, part):  # the script is the state's own K, cached or not
            return self.last if part == "primary" else uncached(self, model, part)

        monkeypatch.setattr(Monitor, "__call__", read)
        monkeypatch.setattr(Monitor, "uncached", again)
        return queue

    return script


def ready(env, cfg) -> None:
    prepare(env, cfg, "kl_monitor")


def test_levels_are_evaluated_where_crossed_and_the_top_level_stops_the_run(
    env, tmp_path, scripted
) -> None:
    cfg = scheduled(tmp_path, *K_RUN)
    ready(env, cfg)
    scripted(0.1, 0.6, 0.7, 0.8, 1.2, 1.3)  # steps 5, 10, … : 0.5 at step 15, 1.0 at step 30
    run_dir = run_scheduled(cfg, env, run_id="k")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"] == {"reason": "target", "kind": "target"}
    assert terminal["steps"] == 30 and terminal["responses"] == 120
    reached = [(r["level"], r["step"]) for r in terminal["levels"]["reached"]]
    assert reached == [(0.5, 15), (1.0, 30)]
    assert terminal["levels"]["missed"] == []
    evaluations = read_jsonl(run_dir / "evaluations.jsonl")
    assert [(e["step"], e["levels"]) for e in evaluations] == [(15, [0.5]), (30, [1.0])]
    for e in evaluations:  # the scoring panel's K and utility, never refusal or harm
        assert e["K_benign"] > 0 and "benchmark_accuracy" in e
        assert not any("refusal" in key or "harm" in key and key != "K_harmful" for key in e)
    responses = read_jsonl(run_dir / "responses" / "harmful_evaluation.jsonl")
    assert {r["condition"] for r in responses} == {"step-15", "step-30"}
    assert all("keyword_refusal" not in r for r in responses)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "keyword_refusal_rate" not in summary
    monitor = read_jsonl(run_dir / "monitor.jsonl")
    assert [m["step"] for m in monitor] == [5, 10, 15, 20, 25, 30]
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    assert max(m["step"] for m in metrics) == 29
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["inputs"]["streams"]["training"]["count"] == 120


def test_a_flat_monitor_stops_on_the_plateau_and_records_missed_levels(
    env, tmp_path, scripted
) -> None:
    cfg = scheduled(tmp_path, *K_RUN)
    ready(env, cfg)
    scripted(0.3)  # always below 0.9 × 0.5
    run_dir = run_scheduled(cfg, env, run_id="plateau")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"]["reason"] == "plateau" and terminal["steps"] == 20
    assert terminal["levels"] == {"planned": [0.5, 1.0], "reached": [], "missed": [0.5, 1.0]}
    assert terminal["evaluations"] == []


def test_a_slow_rise_stops_at_t_max(env, tmp_path, scripted) -> None:
    cfg = scheduled(tmp_path, *K_RUN[:-1], "schedule.plateau.windows=6", "train.steps=20")
    ready(env, cfg)
    scripted(0.2, 0.3, 0.6, 0.7)  # 0.5 crossed at step 20, the last step
    run_dir = run_scheduled(cfg, env, run_id="tmax")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"] == {"reason": "t_max", "kind": "limit"} and terminal["steps"] == 20
    assert [r["level"] for r in terminal["levels"]["reached"]] == [0.5]
    assert terminal["levels"]["missed"] == [1.0]


@pytest.mark.parametrize(("value", "reason"), [(2.5, "k_above_max"), (math.nan, "nonfinite_k")])
def test_the_monitor_stops_an_unstable_run(env, tmp_path, scripted, value, reason) -> None:
    cfg = scheduled(tmp_path, *K_RUN)  # K above 2 × 1.0
    ready(env, cfg)
    scripted(0.1, value)
    run_dir = run_scheduled(cfg, env, run_id=reason)
    terminal = json.loads((run_dir / "terminal.json").read_text())
    stop = terminal["stop"]
    assert (stop["reason"], stop["kind"]) == (reason, "instability")
    if reason == "k_above_max":
        assert stop["detail"] == {"K_monitor": 2.5, "k_max": 2.0}
    assert terminal["steps"] == 10
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["status"] == "completed"  # a result, not a failure


def test_a_gradient_spike_stops_before_its_update(env, tmp_path, scripted, monkeypatch) -> None:
    import zero_field_refusal.schedule as schedule

    cfg = scheduled(tmp_path, *K_RUN)
    ready(env, cfg)
    scripted(0.1)
    norms = iter([1.0, 1.0, 1.0, 50.0])
    real = schedule._norm

    def norm(tensors):  # the gradient norms are scripted, the weight check stays real
        first = next(t for t in tensors if t is not None)
        return next(norms) if first.requires_grad is False and first.grad is None else real(tensors)

    monkeypatch.setattr(schedule, "_norm", norm)
    run_dir = run_scheduled(cfg, env, run_id="spike")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"]["reason"] == "gradient_spike" and terminal["steps"] == 3
    assert terminal["stop"]["detail"] == {"grad_norm": 50.0, "reference_median": 1.0}
    rejected = terminal["stop"]["rejected"]  # the refused batch, not counted as a step
    assert (rejected["step"], rejected["grad_norm"]) == (3, 50.0)
    assert len(read_jsonl(run_dir / "metrics.jsonl")) == 3


def test_a_logged_gradient_rule_records_the_spike_and_runs_on(
    env, tmp_path, scripted, monkeypatch
) -> None:
    import zero_field_refusal.schedule as schedule

    cfg = scheduled(tmp_path, "train.steps=6", "schedule.evaluate_at=[6]",
                    "+schedule.instability.grad_rule=log", policy="short")  # fmt: skip
    ready(env, cfg)
    scripted(0.1)
    norms = iter([1.0, 1.0, 1.0, 50.0, 1.0, 1.0])
    real = schedule._norm

    def norm(tensors):  # the gradient norms are scripted, the weight check stays real
        first = next(t for t in tensors if t is not None)
        return next(norms) if first.requires_grad is False and first.grad is None else real(tensors)

    monkeypatch.setattr(schedule, "_norm", norm)
    run_dir = run_scheduled(cfg, env, run_id="logged")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"] == {"reason": "steps", "kind": "limit"} and terminal["steps"] == 6
    assert terminal["gradient_spikes"] == [{"step": 3, "grad_norm": 50.0, "reference_median": 1.0}]
    steps = [row["step"] for row in read_jsonl(run_dir / "metrics.jsonl") if "loss" in row]
    assert steps == list(range(6)) and terminal["evaluations"] == [6]  # the update was taken


@pytest.mark.parametrize(("scoring_k", "stop"), [(0.2, "contamination"), (0.01, "steps")])
def test_a_control_reading_above_the_contamination_bound_is_checked_without_the_cache(
    env, tmp_path, scripted, monkeypatch, scoring_k, stop
) -> None:
    from zero_field_refusal.analysis.stage_a import load_trajectory
    from zero_field_refusal.panels import ScoringDiagnostics

    cfg = scheduled(tmp_path, "train.steps=10", "schedule.evaluate_at=[10]",
                    "+schedule.instability.grad_rule=log",
                    "+schedule.instability.contamination_k=0.05", policy="short")  # fmt: skip
    ready(env, cfg)
    scripted(0.01, 0.1)  # the monitor: below the bound at step 5, above it at step 10
    import zero_field_refusal.schedule as schedule

    check, call = schedule._Loop._contamination, ScoringDiagnostics.__call__

    def contamination(self, m, step, k, scoring):  # the state's scoring K, in the check only
        def scripted_scoring(panel, model, cached=True):
            return torch.full_like(call(panel, model, cached=cached), scoring_k)

        monkeypatch.setattr(ScoringDiagnostics, "__call__", scripted_scoring)
        try:
            return check(self, m, step, k, scoring)
        finally:
            monkeypatch.setattr(ScoringDiagnostics, "__call__", call)

    monkeypatch.setattr(schedule._Loop, "_contamination", contamination)
    run_dir = run_scheduled(cfg, env, run_id=f"c-{stop}")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    checks = read_jsonl(run_dir / "contamination_checks.jsonl")
    assert [(c["step"], c["K_monitor"]) for c in checks] == [(10, pytest.approx(0.1))]
    assert checks[0]["K_scoring"] == pytest.approx(scoring_k) and checks[0]["bound"] == 0.05
    assert terminal["stop"]["reason"] == stop and terminal["steps"] == 10
    if stop == "contamination":  # verified: the control stops, before its evaluation
        assert terminal["stop"]["detail"] == {"K_scoring": checks[0]["K_scoring"], "bound": 0.05}
        assert not (run_dir / "evaluations.jsonl").exists()
    else:  # the monitor alone does not stop it
        assert terminal["evaluations"] == [10]
    run = load_trajectory(run_dir)
    assert run.stable is (stop == "steps") and len(run.contamination_checks) == 1


def test_a_non_finite_diagnostic_stops_a_control_before_its_contamination_check(
    env, tmp_path, scripted, monkeypatch
) -> None:
    from zero_field_refusal.analysis.stage_a import load_trajectory

    cfg = scheduled(tmp_path, "train.steps=25", "schedule.evaluate_at=[25]",
                    "+schedule.instability.grad_rule=log",
                    "+schedule.instability.contamination_k=0.05", policy="short")  # fmt: skip
    ready(env, cfg)
    scripted(0.01, 0.01, 0.01, 0.01, 0.1)  # above the bound only at step 25, a diagnostic step
    patch_reading(monkeypatch, trained(monkeypatch), "scoring", cached_only=False)
    run_dir = run_scheduled(cfg, env, run_id="nan-then-check")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"] == {"reason": "nonfinite_k", "kind": "instability"}
    assert not (run_dir / "contamination_checks.jsonl").exists()  # the verdict came first
    assert load_trajectory(run_dir).stop == "nonfinite_k"


def test_a_cached_diagnostic_the_uncached_check_does_not_reproduce_is_technical(
    env, tmp_path, scripted, monkeypatch
) -> None:
    from zero_field_refusal.panels import CacheMismatch, ScoringDiagnostics

    cfg = scheduled(tmp_path, "train.steps=25", "schedule.evaluate_at=[25]",
                    "+schedule.instability.grad_rule=log",
                    "+schedule.instability.contamination_k=0.05", policy="short")  # fmt: skip
    ready(env, cfg)
    scripted(0.01)
    state, call = trained(monkeypatch), ScoringDiagnostics.__call__

    def scoring(self, m, cached=True):  # once training started: 0.1 cached, 0.01 uncached
        out = call(self, m, cached=cached)
        return torch.full_like(out, 0.1 if cached else 0.01) if state["steps"] else out

    monkeypatch.setattr(ScoringDiagnostics, "__call__", scoring)
    with pytest.raises(CacheMismatch, match="scoring diagnostic"):
        run_scheduled(cfg, env, run_id="cache-only")


def test_a_non_finite_uncached_check_is_the_states_not_the_caches(
    env, tmp_path, scripted, monkeypatch
) -> None:
    from zero_field_refusal.analysis.stage_a import load_trajectory
    from zero_field_refusal.panels import ScoringDiagnostics

    cfg = scheduled(tmp_path, "train.steps=25", "schedule.evaluate_at=[25]",
                    "+schedule.instability.grad_rule=log",
                    "+schedule.instability.contamination_k=0.05", policy="short")  # fmt: skip
    ready(env, cfg)
    scripted(0.01)
    state, call = trained(monkeypatch), ScoringDiagnostics.__call__

    def scoring(self, m, cached=True):  # once training started: 0.1 cached, not finite uncached
        out = call(self, m, cached=cached)
        if not state["steps"]:
            return out
        return torch.full_like(out, 0.1) if cached else nan_like(out)

    monkeypatch.setattr(ScoringDiagnostics, "__call__", scoring)
    run_dir = run_scheduled(cfg, env, run_id="nan-uncached")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"] == {"reason": "nonfinite_k", "kind": "instability"}
    (check,) = read_jsonl(run_dir / "contamination_checks.jsonl")
    assert check["step"] == 25 and math.isnan(check["K_scoring"])
    assert load_trajectory(run_dir).stop == "nonfinite_k"


def test_a_passed_contamination_check_still_counts_when_the_evaluation_stops_the_run(
    env, tmp_path, scripted, monkeypatch
) -> None:
    from zero_field_refusal.analysis.stage_a import load_trajectory
    from zero_field_refusal.panels import Monitor, ScoringDiagnostics

    cfg = scheduled(tmp_path, "train.steps=25", "schedule.evaluate_at=[25]",
                    "+schedule.instability.grad_rule=log",
                    "+schedule.instability.contamination_k=0.05", policy="short")  # fmt: skip
    ready(env, cfg)
    scripted(0.01, 0.01, 0.01, 0.01, 0.1)  # the monitor above the bound at step 25
    state, call, values = trained(monkeypatch), ScoringDiagnostics.__call__, Monitor._values

    def scoring(self, m, cached=True):  # a finite diagnostic, below the bound, cached or not
        out = call(self, m, cached=cached)
        return torch.full_like(out, 0.01) if state["steps"] else out

    def monitor(self, m, part, cached=True):  # the evaluation's uncached second response: NaN
        out = values(self, m, part, cached=cached)
        return nan_like(out) if part == "secondary" and not cached and state["steps"] else out

    monkeypatch.setattr(ScoringDiagnostics, "__call__", scoring)
    monkeypatch.setattr(Monitor, "_values", monitor)
    run_dir = run_scheduled(cfg, env, run_id="check-then-nan")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"] == {"reason": "nonfinite_k", "kind": "instability"}
    (check,) = read_jsonl(run_dir / "contamination_checks.jsonl")  # it ran, and passed
    assert check["step"] == 25 and check["K_scoring"] == pytest.approx(0.01)
    run = load_trajectory(run_dir)
    assert run.stop == "nonfinite_k" and len(run.contamination_checks) == 1


def test_a_short_run_takes_its_steps_and_is_evaluated_where_scheduled(
    env, tmp_path, scripted
) -> None:
    cfg = scheduled(tmp_path, "train.steps=10", "schedule.evaluate_at=[5,10]", policy="short")
    ready(env, cfg)
    scripted(9.0)  # far above every level: a fixed run ignores levels, but not k_max (16)
    run_dir = run_scheduled(cfg, env, run_id="short")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"] == {"reason": "steps", "kind": "limit"}
    assert terminal["evaluations"] == [5, 10] and "levels" not in terminal


def test_the_feed_extends_the_stream_and_checks_each_chunk_first(env, tmp_path, scripted) -> None:
    cfg = scheduled(tmp_path, *K_RUN)
    ready(env, cfg)
    scripted(0.1, 0.6, 0.7, 0.8, 1.2, 1.3)
    run_dir = run_scheduled(cfg, env, run_id="feed")  # 30 steps × 4 = 120 responses
    checks = read_jsonl(run_dir / "chunk_checks.jsonl")
    assert [c["chunk"] for c in checks] == list(range(10))  # 120 / 12, each checked once
    assert all(c["teacher_gap"]["token_max_abs"] < 1e-4 and "path_field" in c for c in checks)
    assert StreamPrefix(cfg, env, 0).available == 120  # extended by the run, no further
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stream_extension"]["chunks"] == 10
    assert terminal["stream_extension"]["seconds"] > 0


def test_without_extension_a_run_needs_its_stream(env, tmp_path, scripted) -> None:
    cfg = scheduled(tmp_path, *K_RUN, "schedule.extend_stream=false")
    ready(env, cfg)
    scripted(0.1)
    extend_stream(cfg, env, 0, 12, batch_size=4)  # three steps' worth
    with pytest.raises(StreamError, match="extend the stream first"):
        run_scheduled(cfg, env, run_id="no-extension")
    run_dir = env.data_root.parent / "runs" / "smoke" / "no-extension"
    assert len(read_jsonl(run_dir / "metrics.jsonl")) == 3


def test_a_chunk_from_another_teacher_stops_the_run_before_it_trains(env, tmp_path, scripted):
    cfg = scheduled(tmp_path, *K_RUN, "schedule.extend_stream=false")
    ready(env, cfg)
    scripted(0.1)
    extend_stream(cfg, env, 0, 12, batch_size=4)  # chunk 0 from θ₀: steps 0-2
    student = env.new_model

    def other_model():
        model = student()
        with torch.no_grad():
            model.lm_head.weight.add_(0.01 * torch.randn_like(model.lm_head.weight))
        return model

    env.new_model = other_model
    extend_stream(cfg, env, 0, 24, batch_size=4)  # chunk 1 from another model
    env.new_model = student
    with pytest.raises(CheckFailed, match="teacher gap"):
        run_scheduled(cfg, env, run_id="other-teacher")
    run_dir = env.data_root.parent / "runs" / "smoke" / "other-teacher"
    checks = read_jsonl(run_dir / "chunk_checks.jsonl")
    assert [c["chunk"] for c in checks] == [0, 1]  # the failing chunk is recorded too
    assert len(read_jsonl(run_dir / "metrics.jsonl")) == 3  # stopped before step 3 trains


def test_the_soft_target_control_runs_under_a_fixed_schedule(env, tmp_path, scripted) -> None:
    experiment = tmp_path / "soft.yaml"
    experiment.write_text("defaults: [experiments/smoke, optimizer/sgd, control/soft_targets, "
                          "eval/study2, schedule/short]\n")  # fmt: skip
    cfg = load_run_config(experiment, [*SMALL, *APPENDABLE, "teacher.chunk_size=12",
                                       "optimizer.lr=0.1", "train.steps=6",
                                       "schedule.evaluate_at=[6]"])  # fmt: skip
    ready(env, cfg)
    scripted(0.0)
    run_dir = run_scheduled(cfg, env, run_id="soft")
    first = read_jsonl(run_dir / "metrics.jsonl")[0]
    assert abs(first["loss"]) < 1e-5
    summary = json.loads((run_dir / "summary.json").read_text())
    assert (
        summary["soft_targets_replay"]["digest_equal"] == summary["soft_targets_replay"]["batches"]
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("schedule.policy=sometimes", "schedule.policy"),
        ("schedule.levels=[1.0,0.5]", "schedule.levels"),
        ("eval.monitor.every=0", "eval.monitor.every"),
        ("eval.monitor.version=3", "eval.monitor.version"),
        ("eval.monitor.prompt_set=alpaca_monitor", "eval.monitor.prompt_set"),
        ("eval.monitor.drive=3", "eval.monitor.drive"),
        ("eval.monitor.cache.primary=gpu", "eval.monitor.cache.primary"),
        ("eval.diagnostics.every=7", "eval.diagnostics.every"),
        ("eval.diagnostics.cache=disk", "eval.diagnostics.cache"),
    ],
)
def test_schedules_are_checked_at_load(tmp_path, override, message) -> None:
    with pytest.raises(ProtocolError, match=message.replace(".", r"\.")):
        scheduled(tmp_path, override)


def test_a_schedule_needs_an_appendable_stream_and_valid_evaluation_steps(tmp_path) -> None:
    experiment = tmp_path / "fixed.yaml"
    experiment.write_text("defaults: [experiments/smoke, eval/study2, schedule/discovery]\n")
    with pytest.raises(ProtocolError, match="appendable stream"):
        load_run_config(experiment, SMALL)
    with pytest.raises(ProtocolError, match="schedule.evaluate_at"):
        scheduled(tmp_path, "train.steps=10", "schedule.evaluate_at=[11]", policy="short")


def test_the_records_of_real_runs_load_as_trajectories(env, tmp_path, scripted) -> None:
    from zero_field_refusal.analysis.stage_a import load_trajectory

    cfg = scheduled(tmp_path, *K_RUN)
    ready(env, cfg)
    scripted(0.1, 0.6, 0.7, 0.8, 1.2, 1.3)
    run = load_trajectory(run_scheduled(cfg, env, run_id="k"))
    assert (run.arm, run.steps, run.stop) == ("adamw_long", 30, "target")
    assert run.evaluation_at(0.5)["step"] == 15 and run.evaluation_at(1.0)["step"] == 30
    assert len(run.grad_norms) == 30 and run.seconds > 0
    short = scheduled(tmp_path, "train.steps=10", "schedule.evaluate_at=[5,10]", policy="short")
    run = load_trajectory(run_scheduled(short, env, run_id="short"))
    assert (run.arm, run.steps, [r["step"] for r in run.evaluations]) == ("adamw_25", 10, [5, 10])


def test_an_update_that_breaks_the_weights_counts_and_is_recorded(
    env, tmp_path, scripted, monkeypatch
) -> None:
    import zero_field_refusal.schedule as schedule
    from zero_field_refusal.analysis.stage_a import load_trajectory

    cfg = scheduled(tmp_path, *K_RUN)
    ready(env, cfg)
    scripted(0.1)
    real, calls = schedule._norm, {"weights": 0}

    def norm(tensors):
        first = next(t for t in tensors if t is not None)
        if first.requires_grad:  # the weights after an update
            calls["weights"] += 1
            if calls["weights"] == 5:
                return float("nan")
        return real(tensors)

    monkeypatch.setattr(schedule, "_norm", norm)
    run_dir = run_scheduled(cfg, env, run_id="weights")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"]["reason"] == "nonfinite_weight"
    assert terminal["steps"] == 5 and terminal["responses"] == 20  # the applied update counts
    assert terminal["stop"]["rejected"]["step"] == 4 and terminal["stop"]["rejected"]["applied"]
    assert load_trajectory(run_dir).steps == 5  # no reading after the bad update


def test_a_failed_evaluation_leaves_its_levels_missed(env, tmp_path, scripted, monkeypatch):
    from zero_field_refusal.analysis.stage_a import load_trajectory
    from zero_field_refusal.tracking import Diverged
    from zero_field_refusal.train import Evaluator

    cfg = scheduled(tmp_path, *K_RUN)
    ready(env, cfg)
    scripted(1.2, 1.3)  # both levels crossed at step 10

    def broken(self, model, condition, check=None):
        raise Diverged("K not finite")

    monkeypatch.setattr(Evaluator, "__call__", broken)
    run_dir = run_scheduled(cfg, env, run_id="failed-evaluation")
    levels = json.loads((run_dir / "terminal.json").read_text())["levels"]
    assert levels["reached"] == [] and levels["missed"] == [0.5, 1.0]
    assert levels["failed_crossing"] == {"step": 10, "levels": [0.5, 1.0]}
    assert load_trajectory(run_dir).stop == "nonfinite_k"


def test_a_fixed_schedule_must_end_with_an_evaluation(tmp_path) -> None:
    with pytest.raises(ProtocolError, match="must include the last step 10"):
        scheduled(tmp_path, "train.steps=10", "schedule.evaluate_at=[5]", policy="short")


def test_monitor_version_2_diagnoses_the_scoring_panel_and_checks_its_caches(env, tmp_path) -> None:
    """A fixed 25-step run: the monitor reads its first response per prompt every 5 steps; at
    step 25 the scoring panel's K alone and the full monitor panel are read (diagnostic), and the
    full evaluation of that same state checks every cache against uncached readings."""
    cfg = scheduled(tmp_path, "train.steps=25", "schedule.evaluate_at=[25]", policy="short")
    ready(env, cfg)
    run_dir = run_scheduled(cfg, env, run_id="v2")
    monitor = read_jsonl(run_dir / "monitor.jsonl")
    assert [m["step"] for m in monitor] == [5, 10, 15, 20, 25]
    assert "K_scoring" not in monitor[0] and "K_scoring" in monitor[-1]
    (diagnostic,) = read_jsonl(run_dir / "diagnostics.jsonl")
    assert diagnostic["step"] == 25 and diagnostic["K_monitor"] == monitor[-1]["K_monitor"]
    assert len(diagnostic["scoring"]) == 2 * len(diagnostic["monitor_primary"])
    assert len(diagnostic["monitor_primary"]) == len(diagnostic["monitor_secondary"])
    assert diagnostic["K_scoring"] == pytest.approx(
        sum(diagnostic["scoring"]) / len(diagnostic["scoring"])
    )
    (evaluation,) = read_jsonl(run_dir / "evaluations.jsonl")
    assert evaluation["K_benign"] == diagnostic["K_scoring"]  # cached and uncached, one state
    assert evaluation["K_monitor_primary"] == evaluation["K_monitor"] == diagnostic["K_monitor"]
    assert evaluation["K_monitor_full"] == diagnostic["K_monitor_full"]
    checks = read_jsonl(run_dir / "cache_checks.jsonl")
    assert [c["step"] for c in checks] == [0, 25]
    for check in checks:
        assert {k for k in check if k != "step"} == {"primary", "secondary", "scoring"}
        assert all(check[k]["bitwise_equal"] for k in ("primary", "secondary", "scoring"))
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["monitor"] == {"version": 2, "drive": 1, "samples_per_prompt": 2}
    assert terminal["diagnostics"] == 1


def test_a_cache_that_changes_k_stops_the_run(env, tmp_path, monkeypatch) -> None:
    from zero_field_refusal.evaluation import ReferenceLogits
    from zero_field_refusal.panels import CacheMismatch

    cfg = scheduled(tmp_path, "train.steps=25", "schedule.evaluate_at=[25]", policy="short")
    ready(env, cfg)
    chunk = ReferenceLogits.chunk
    monkeypatch.setattr(ReferenceLogits, "chunk", lambda self, i, d: chunk(self, i, d) + 1e-3)
    with pytest.raises(CacheMismatch, match="step 0"):
        run_scheduled(cfg, env, run_id="bad-cache")
    run_dir = Path(os.environ["RUNS_DIR"]) / cfg.experiment / "bad-cache"
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "failed"
    assert not (run_dir / "terminal.json").exists()  # stopped before training


def test_a_k_only_evaluation_reads_no_harmful_prompt_and_measures_what_the_sealed_one_does(
    env, tmp_path, monkeypatch
) -> None:
    """Study 2's numerical runs (amendment 4, review fa12): with ``eval.k_only`` no harmful
    prompt set is read and no response to a harmful prompt is generated; K, the monitor and
    utility equal those of the sealed evaluation, and training is unaffected (on CUDA as well:
    the next test)."""
    from zero_field_refusal.analysis.stage_a import TrajectoryError, load_trajectory

    k_only = k_only_equals_sealed(env, tmp_path, monkeypatch)
    # Its records must say it is K-only (review of Task 2, minor 7).
    meta_path = k_only / "meta.json"
    meta = json.loads(meta_path.read_text())
    for changed, message in (
        ({**meta["inputs"], "k_only": False}, "inputs do not record k_only"),
        ({k: v for k, v in meta["inputs"].items() if k != "k_only"}, "inputs do not record k_only"),
        ({**meta["inputs"], "streams": {**meta["inputs"]["streams"], "kl_harmful": {}}},
         "recorded a kl_harmful stream"),
    ):  # fmt: skip
        meta_path.write_text(json.dumps({**meta, "inputs": changed}))
        with pytest.raises(TrajectoryError, match=message):
            load_trajectory(k_only)
    meta_path.write_text(json.dumps(meta))
    (k_only / "responses").mkdir()
    (k_only / "responses" / "harmful_evaluation.jsonl").write_text("")
    with pytest.raises(TrajectoryError, match="K-only run recorded harmful prompts or stored"):
        load_trajectory(k_only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cuda_k_only_equals_sealed(gpu_env, tmp_path, monkeypatch) -> None:  # noqa: F811
    """Study 2's numerical runs on CUDA (review of Task 2, minor 6): the K-only twin of a sealed
    run reads no harmful prompt and measures K, the monitor and utility as the sealed run does,
    its training the same."""
    k_only = k_only_equals_sealed(gpu_env, tmp_path, monkeypatch, "execution.device=cuda")
    assert json.loads((k_only / "meta.json").read_text())["inputs"]["k_only"] is True


def k_only_equals_sealed(env, tmp_path, monkeypatch, *overrides: str) -> Path:
    """A sealed run and its K-only twin (``eval.k_only``) of 25 steps, on ``env``: the K-only
    run reads no harmful prompt set and stores no response; K, the monitor, utility and training
    are the sealed run's. Returns the K-only run's directory."""
    from zero_field_refusal.analysis.stage_a import load_trajectory

    steps = ("train.steps=25", "schedule.evaluate_at=[5,25]", *overrides)
    sealed_cfg = scheduled(tmp_path, *steps, policy="short")
    ready(env, sealed_cfg)
    sealed = run_scheduled(sealed_cfg, env, run_id="sealed")
    assert (sealed / "responses").is_dir()  # the sealed evaluation answers the harmful prompts
    read = env.prompt_set

    def guarded(name: str):
        if "harmful" in name:
            raise AssertionError(f"the K-only run read the prompt set {name}")
        return read(name)

    monkeypatch.setattr(env, "prompt_set", guarded)
    k_only_cfg = scheduled(tmp_path, *steps, "+eval.k_only=true", policy="short")
    k_only = run_scheduled(k_only_cfg, env, run_id="k-only")
    assert not (k_only / "responses").exists()
    meta = json.loads((k_only / "meta.json").read_text())
    assert meta["inputs"]["k_only"] is True and "harmful_prompts" not in meta["inputs"]
    assert "kl_harmful" not in meta["inputs"]["streams"]
    a, b = read_jsonl(sealed / "evaluations.jsonl"), read_jsonl(k_only / "evaluations.jsonl")
    assert [r["step"] for r in a] == [r["step"] for r in b] == [5, 25]
    measured = ("K_benign", "K_monitor", "K_monitor_full", "benchmark_accuracy",
                "factual_accuracy", "heldout_nll_per_token")  # fmt: skip
    for x, y in zip(a, b, strict=True):
        assert {k: x.get(k) for k in measured} == {k: y.get(k) for k in measured}
        assert "K_harmful" in x and "K_harmful" not in y

    def untimed(rows: list[dict]) -> list[dict]:  # timings and memory peaks differ, run to run
        return [{k: v for k, v in r.items() if not any(w in k for w in ("seconds", "time",
                                                                         "memory", "_gb"))}
                for r in rows]  # fmt: skip

    trained = read_jsonl(sealed / "metrics.jsonl")
    measured_rows = read_jsonl(k_only / "metrics.jsonl")
    assert [r for r in untimed(trained) if "loss" in r] == [r for r in untimed(measured_rows)
                                                            if "loss" in r]  # training  # fmt: skip
    assert any("eval/K_harmful" in r for r in trained)
    assert not any("eval/K_harmful" in r for r in measured_rows)
    without = [{k: v for k, v in r.items() if k != "eval/K_harmful"} for r in untimed(trained)]
    assert without == untimed(measured_rows)
    for name in ("monitor.jsonl", "diagnostics.jsonl"):
        assert untimed(read_jsonl(sealed / name)) == untimed(read_jsonl(k_only / name)), name
    run = load_trajectory(k_only)  # a K-only run's records are whole without responses
    assert run.evaluations[-1]["K_benign"] == a[-1]["K_benign"]
    return k_only


def test_a_k_only_numerical_run_writes_what_the_numerical_check_reads(env, tmp_path) -> None:
    """Task 1's contract: a K-only FA run of role numerical still writes its metrics, monitor
    readings and terminal record, and in meta.json its commit and the v₀ and training streams it
    read, with the role in its archived config; only the harmful generation, the kl_harmful
    stream and the harm outputs are absent."""
    from omegaconf import OmegaConf

    cfg = scheduled(tmp_path, "train.steps=25", "schedule.evaluate_at=[25]", "+eval.k_only=true",
                    "optimizer=adam_frozen", "optimizer.lr=1e-3",
                    "optimizer.preconditioner.batches=3", "+study2_short.role=numerical",
                    policy="short")  # fmt: skip
    ready(env, cfg)
    prepare(env, cfg, "preconditioner")
    run_dir = run_scheduled(cfg, env, run_id="numerical")
    for name in ("metrics.jsonl", "monitor.jsonl", "terminal.json", "evaluations.jsonl"):
        assert (run_dir / name).is_file(), name
    meta = json.loads((run_dir / "meta.json").read_text())
    assert set(meta["git"]) >= {"commit", "dirty"}
    streams = meta["inputs"]["streams"]
    assert {"path", "sha256"} <= set(streams["preconditioner"])
    assert streams["training"]["count"] == 25 * 4 and streams["training"]["chunks"]
    assert "kl_harmful" not in streams and "harmful_prompts" not in meta["inputs"]
    assert OmegaConf.load(run_dir / "config.yaml").study2_short.role == "numerical"
    assert not (run_dir / "responses").exists()
    (row,) = read_jsonl(run_dir / "evaluations.jsonl")
    assert "K_benign" in row and not any("harm" in key.lower() for key in row)


def test_k_only_is_a_boolean(env, tmp_path) -> None:
    from zero_field_refusal.tracking import Run
    from zero_field_refusal.train import Evaluator

    cfg = scheduled(tmp_path, "train.steps=25", "schedule.evaluate_at=[25]", "+eval.k_only=1",
                    policy="short")  # fmt: skip
    ready(env, cfg)
    with Run(cfg, run_id="flag") as run, pytest.raises(ValueError, match="eval.k_only"):
        Evaluator(cfg, env, run, env.new_model(), sealed=True)


def test_monitor_version_1_runs_without_diagnostics(env, tmp_path, scripted) -> None:
    cfg = scheduled(tmp_path, *K_RUN, "eval.monitor.version=1")
    ready(env, cfg)
    scripted(0.1, 0.6, 0.7, 0.8, 1.2, 1.3)
    run_dir = run_scheduled(cfg, env, run_id="v1")
    assert not (run_dir / "diagnostics.jsonl").exists()
    assert not (run_dir / "cache_checks.jsonl").exists()
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert "monitor" not in terminal and terminal["stop"]["reason"] == "target"
    assert all("K_monitor_full" not in e for e in read_jsonl(run_dir / "evaluations.jsonl"))


def short_v2(env, tmp_path):
    cfg = scheduled(tmp_path, "train.steps=25", "schedule.evaluate_at=[25]", policy="short")
    ready(env, cfg)
    return cfg


def nan_like(values: torch.Tensor) -> torch.Tensor:
    return torch.full_like(values, float("nan"))


def trained(monkeypatch) -> dict:
    """Counts the steps taken, for patches that act only once training started."""
    from zero_field_refusal import schedule

    state = {"steps": 0}
    after = schedule._Loop.after_step

    def counted(self, model, geometry):
        state["steps"] = self.steps
        return after(self, model, geometry)

    monkeypatch.setattr(schedule._Loop, "after_step", counted)
    return state


def patch_reading(monkeypatch, state, where: str, *, cached_only: bool) -> None:
    """Make a reading non-finite once training started: the scoring diagnostics, or a monitor
    part; ``cached_only`` leaves the uncached reading as it is."""
    from zero_field_refusal.panels import Monitor, ScoringDiagnostics

    if where == "scoring":
        call = ScoringDiagnostics.__call__

        def scoring(self, m, cached=True):
            out = call(self, m, cached=cached)
            hit = state["steps"] and (cached or not cached_only)
            return nan_like(out) if hit else out

        monkeypatch.setattr(ScoringDiagnostics, "__call__", scoring)
    else:
        values = Monitor._values

        def monitor(self, m, part, cached=True):
            out = values(self, m, part, cached=cached)
            hit = part == where and state["steps"] and (cached or not cached_only)
            return nan_like(out) if hit else out

        monkeypatch.setattr(Monitor, "_values", monitor)


@pytest.mark.parametrize("where", ["scoring", "secondary"])
def test_a_reproduced_non_finite_diagnostic_stops_the_run(env, tmp_path, monkeypatch, where):
    cfg = short_v2(env, tmp_path)
    patch_reading(monkeypatch, trained(monkeypatch), where, cached_only=False)
    run_dir = run_scheduled(cfg, env, run_id=f"nan-{where}")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"] == {"reason": "nonfinite_k", "kind": "instability"}
    assert terminal["steps"] == 25 and not (run_dir / "evaluations.jsonl").exists()
    from zero_field_refusal.analysis.stage_a import load_trajectory

    assert load_trajectory(run_dir).stop == "nonfinite_k"  # the verdict is a whole record


@pytest.mark.parametrize("where", ["scoring", "primary"])
def test_a_non_finite_value_only_the_cache_makes_is_technical(env, tmp_path, monkeypatch, where):
    from zero_field_refusal.panels import CacheMismatch

    cfg = short_v2(env, tmp_path)
    patch_reading(monkeypatch, trained(monkeypatch), where, cached_only=True)
    with pytest.raises(CacheMismatch, match="not finite, the uncached one is"):
        run_scheduled(cfg, env, run_id=f"cache-nan-{where}")


def failing_check(monkeypatch, when) -> None:
    """Make a cache check find a non-finite uncached reading where ``when(steps)`` holds."""
    from zero_field_refusal.panels import Monitor

    check = Monitor.check
    state = trained(monkeypatch)

    def failing(self, model):
        out = check(self, model)
        if when(state["steps"]):
            out["secondary"] |= {"bitwise_equal": False, "uncached_finite": False}
        return out

    monkeypatch.setattr(Monitor, "check", failing)


def test_a_non_finite_verdict_at_theta0_stops_before_any_update(env, tmp_path, monkeypatch):
    cfg = short_v2(env, tmp_path)
    failing_check(monkeypatch, lambda steps: steps == 0)
    run_dir = run_scheduled(cfg, env, run_id="theta0-nan")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"] == {"reason": "nonfinite_k", "kind": "instability"}
    assert terminal["steps"] == 0 and terminal["responses"] == 0
    assert not (run_dir / "monitor.jsonl").exists()
    from zero_field_refusal.analysis.stage_a import load_trajectory

    assert load_trajectory(run_dir).stop == "nonfinite_k"  # the launcher's check
    metrics = run_dir / "metrics.jsonl"
    assert not metrics.exists() or not any("loss" in row for row in read_jsonl(metrics))


def test_a_non_finite_uncached_reading_at_an_evaluation_is_an_outcome(env, tmp_path, monkeypatch):
    cfg = short_v2(env, tmp_path)
    failing_check(monkeypatch, lambda steps: steps == 25)
    run_dir = run_scheduled(cfg, env, run_id="eval-nan")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"]["reason"] == "nonfinite_k" and terminal["steps"] == 25
    checks = read_jsonl(run_dir / "cache_checks.jsonl")
    assert [c["step"] for c in checks] == [0, 25]
    assert checks[1]["secondary"]["uncached_finite"] is False
    assert not (run_dir / "responses").exists()  # checked before the evaluation wrote anything
    from zero_field_refusal.analysis.stage_a import load_trajectory

    assert load_trajectory(run_dir).stop == "nonfinite_k"


def test_a_late_non_finite_verdict_leaves_loadable_records(env, tmp_path, monkeypatch, scripted):
    """A crossing evaluated at step 15, then a non-finite uncached reading at the next one: the
    run stops as an outcome, and nothing of the failed evaluation is written."""
    from zero_field_refusal.analysis.stage_a import load_trajectory

    cfg = scheduled(tmp_path, *K_RUN)
    ready(env, cfg)
    scripted(0.1, 0.6, 0.7, 0.8, 1.2, 1.3)  # 0.5 at step 15, 1.0 at step 30
    failing_check(monkeypatch, lambda steps: steps == 30)
    run_dir = run_scheduled(cfg, env, run_id="late-nan")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"]["reason"] == "nonfinite_k" and terminal["steps"] == 30
    assert [e["step"] for e in read_jsonl(run_dir / "evaluations.jsonl")] == [15]
    responses = read_jsonl(run_dir / "responses" / "harmful_evaluation.jsonl")
    assert {r["condition"] for r in responses} == {"step-15"}
    assert terminal["levels"]["failed_crossing"] == {"step": 30, "levels": [1.0]}
    run = load_trajectory(run_dir)  # the launcher's check: whole records
    assert run.stop == "nonfinite_k" and run.evaluation_at(1.0) is None


def test_a_full_evaluation_runs_without_the_gradients(env, tmp_path, monkeypatch) -> None:
    """The gradients are freed before a full evaluation (it needs none), which lowers its memory
    peak; the next step makes them anew, so training is unchanged."""
    from zero_field_refusal.train import Evaluator

    cfg = short_v2(env, tmp_path)
    seen = []
    call = Evaluator.__call__

    def evaluate(self, model, condition, check=None):
        seen.append(all(p.grad is None for p in model.parameters()))
        return call(self, model, condition, check=check)

    monkeypatch.setattr(Evaluator, "__call__", evaluate)
    run_dir = run_scheduled(cfg, env, run_id="no-grads")
    assert seen == [True]
    assert json.loads((run_dir / "terminal.json").read_text())["steps"] == 25


def test_freeing_the_gradients_changes_no_step_after_an_evaluation(
    env, tmp_path, monkeypatch, scripted
):
    """The same run with and without freeing the gradients before its evaluation at step 15:
    every later step's loss and gradient norm, and the evaluation at step 30, agree bit for bit."""
    from zero_field_refusal import schedule

    cfg = scheduled(tmp_path, *K_RUN)
    ready(env, cfg)
    records = {}
    init = schedule._Loop.__init__

    def without(self, *args, **kwargs):  # the loop, keeping the gradients
        init(self, *args, **kwargs)
        self.release = None

    for name, keep in (("freed", False), ("kept", True)):
        scripted(0.1, 0.6, 0.7, 0.8, 1.2, 1.3)  # evaluations at steps 15 and 30
        if keep:
            monkeypatch.setattr(schedule._Loop, "__init__", without)
        run_dir = run_scheduled(cfg, env, run_id=name)
        metrics = [r for r in read_jsonl(run_dir / "metrics.jsonl") if "loss" in r]
        evaluations = read_jsonl(run_dir / "evaluations.jsonl")
        records[name] = (
            [(r["step"], r["loss"], r["grad_norm"]) for r in metrics],
            [(e["step"], e["K_benign"], e["K_harmful"], e["K_monitor_full"]) for e in evaluations],
        )
    assert records["freed"] == records["kept"]
    assert [e[0] for e in records["freed"][1]] == [15, 30]


# The update scalars (amendment 4, section 4): the three arms' 25-step runs with the hook on and
# off. The pre-warmed and frozen arms estimate v₀ on a preconditioner stream of three batches.
ARMS = {
    "adamw": (),
    "adam_frozen": ("optimizer=adam_frozen", "optimizer.preconditioner.batches=3"),
    "adamw_prewarmed": ("optimizer=adamw_prewarmed", "optimizer.preconditioner.batches=3"),
}


def arm_config(env, tmp_path, arm: str, *overrides: str):
    cfg = scheduled(tmp_path, "train.steps=25", "schedule.evaluate_at=[25]", *ARMS[arm],
                    "optimizer.lr=1e-3", *overrides, policy="short")  # fmt: skip
    prepare(env, cfg, "kl_monitor", *(["preconditioner"] if arm != "adamw" else []))
    return cfg


def built_optimizers(monkeypatch) -> list:
    """The optimizers the runs build, kept after the run: they hold its final parameters."""
    from zero_field_refusal import schedule

    built, build = [], schedule.build_optimizer

    def keep(*args, **kwargs):
        built.append(build(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(schedule, "build_optimizer", keep)
    return built


def bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().view(torch.int32)  # fp32 bits: −0.0 and NaN compared exactly


@pytest.mark.parametrize("arm", list(ARMS))
def test_scalar_logging_is_bit_identical(env, tmp_path, monkeypatch, arm) -> None:
    """The hook only reads: the parameters and the optimizer state after 25 steps and every
    record of the run (loss and gradient-norm rows, monitor readings, diagnostics, the
    evaluation, the stop) are the same bit for bit with and without it."""
    built = built_optimizers(monkeypatch)
    outcome = {}
    for on in (False, True):
        cfg = arm_config(env, tmp_path, arm, f"tracking.update_scalars={str(on).lower()}")
        run_dir = run_scheduled(cfg, env, run_id=f"{arm}-scalars-{on}")
        optimizer = built[-1]
        params = [p for group in optimizer.param_groups for p in group["params"]]
        state = [optimizer.state[p][key] for p in params for key in ("exp_avg", "exp_avg_sq")]
        terminal = json.loads((run_dir / "terminal.json").read_text())
        assert terminal["steps"] == 25 and terminal["stop"]["reason"] == "steps"
        outcome[on] = {
            "params": [bits(p) for p in params],
            "state": [bits(s) for s in state],
            "steps": [optimizer.state[p]["step"] for p in params],
            "metrics": [{k: v for k, v in r.items() if k != "time_s"}
                        for r in read_jsonl(run_dir / "metrics.jsonl")],
            "records": [read_jsonl(run_dir / name) for name in
                        ("monitor.jsonl", "diagnostics.jsonl", "evaluations.jsonl",
                         "cache_checks.jsonl")],
            "terminal": (terminal["stop"], terminal["steps"], terminal["evaluations"]),
        }  # fmt: skip
        scalars = run_dir / "update_scalars.jsonl"
        assert scalars.exists() is on
    off, on = outcome[False], outcome[True]
    assert all(torch.equal(a, b) for a, b in zip(off["params"], on["params"], strict=True))
    assert all(torch.equal(a, b) for a, b in zip(off["state"], on["state"], strict=True))
    assert {k: v for k, v in off.items() if k not in ("params", "state")} == {
        k: v for k, v in on.items() if k not in ("params", "state")
    }
    assert [r["step"] for r in off["metrics"] if "loss" in r] == list(range(25))
    rows = read_jsonl(scalars)
    assert [r["step"] for r in rows] == list(range(1, 26))
    for row in rows:
        assert row["update_norm"] > 0 and -1 <= row["cos_update_grad"] <= 1
    departures = {r["step"]: r["d_departure"] for r in rows if "d_departure" in r}
    assert list(departures) == ([5, 10, 25] if arm == "adamw_prewarmed" else [])
    for departure in departures.values():
        assert set(departure) == {"norm_ratio", "median", "p90"}
        assert departure["norm_ratio"] > 0 and 0 <= departure["median"] <= departure["p90"]


def test_the_scalars_are_off_unless_the_config_turns_them_on(env, tmp_path) -> None:
    cfg = arm_config(env, tmp_path, "adamw", "train.steps=5", "schedule.evaluate_at=[5]")
    assert cfg.tracking.update_scalars is False  # base.yaml; experiments/study2_short sets it
    run_dir = run_scheduled(cfg, env, run_id="default")
    assert not (run_dir / "update_scalars.jsonl").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_no_persistent_gpu_allocation(gpu_env, tmp_path, monkeypatch) -> None:  # noqa: F811
    """The pre-warmed arm on CUDA, after a warm-up run, without and with the hook (in pieces of
    500 coordinates, so that the largest parameters span several): after every step the
    allocated GPU memory is the same, and the run's peak stays within one parameter tensor's size
    of the unlogged peak. The copy of v₀ is in pinned host memory, and the training is the same
    bit for bit on the GPU too."""
    from zero_field_refusal import schedule

    built = built_optimizers(monkeypatch)
    after, hooks, step = [], [], schedule._Loop.after_step

    def measured(self, model, geometry):  # after the update and the hook
        after.append(torch.cuda.memory_allocated())
        return step(self, model, geometry)

    class Pieces(UpdateScalars):
        def __init__(self, optimizer):
            super().__init__(optimizer, chunk=500)
            hooks.append(self)

    monkeypatch.setattr(schedule._Loop, "after_step", measured)
    monkeypatch.setattr(schedule, "UpdateScalars", Pieces)
    peaks, steps, trained = {}, {}, {}
    for name in ("warm-up", "off", "on"):
        cfg = arm_config(gpu_env, tmp_path, "adamw_prewarmed", "execution.device=cuda",
                         f"tracking.update_scalars={str(name == 'on').lower()}")  # fmt: skip
        after.clear()
        gc.collect()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        run_dir = run_scheduled(cfg, gpu_env, run_id=name)
        torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated() - base
        steps[name] = [a - base for a in after]
        params = [p for group in built[-1].param_groups for p in group["params"]]
        metrics = [(r["step"], r["loss"], r["grad_norm"])
                   for r in read_jsonl(run_dir / "metrics.jsonl") if "loss" in r]  # fmt: skip
        trained[name] = ([bits(p) for p in params], metrics)
    assert len(steps["on"]) == 25 and steps["on"] == steps["off"]
    largest = max(p.numel() * p.element_size() for p in params)
    assert largest > 500 * 4  # the pieces split it
    assert peaks["on"] == peaks["off"], (peaks, largest)  # the run's peak, not the hook's
    (off, metrics_off), (on, metrics_on) = trained["off"], trained["on"]
    assert metrics_on == metrics_off and len(metrics_on) == 25  # on the GPU too
    assert all(torch.equal(a, b) for a, b in zip(off, on, strict=True))
    (hook,) = hooks
    assert len(hook.v0) == len(params) and all(v.is_pinned() for v in hook.v0)
    rows = read_jsonl(run_dir / "update_scalars.jsonl")
    assert [r["step"] for r in rows if "d_departure" in r] == [5, 10, 25]


# The time course (amendment 4, section 4, "Harm over steps"): a run evaluated at steps 1, 5, 10
# and 25 against its endpoint run, evaluated at step 25 alone. Every regular run records the
# update scalars, so both runs do here.
TIMECOURSE_STEPS = (1, 5, 10, 25)
EXCLUDED = ("time_s", "peak_")  # the timings and memory peaks differ from run to run


def comparable(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if not any(w in k for w in EXCLUDED)} for r in rows]


def intermediate_evaluations(env, tmp_path, monkeypatch, arm: str, *overrides: str) -> dict:
    """The endpoint run (``evaluate_at`` [25]) and its time-course twin ([1, 5, 10, 25]) of one
    arm, compared bit for bit: the parameters and the optimizer state after step 25, every
    training row of metrics.jsonl and every other row but the time course's own evaluations,
    the monitor readings, the scoring diagnostics and the update scalars, the stop, and the
    step-25 evaluation. Returns the two run directories."""
    built = built_optimizers(monkeypatch)
    outcome, dirs = {}, {}
    for name, steps in (("endpoint", (25,)), ("timecourse", TIMECOURSE_STEPS)):
        at = ",".join(map(str, steps))
        cfg = arm_config(env, tmp_path, arm, "tracking.update_scalars=true",
                         f"schedule.evaluate_at=[{at}]", *overrides)  # fmt: skip
        run_dir = dirs[name] = run_scheduled(cfg, env, run_id=f"{arm}-{name}")
        optimizer = built[-1]
        params = [p for group in optimizer.param_groups for p in group["params"]]
        state = [optimizer.state[p][key] for p in params for key in ("exp_avg", "exp_avg_sq")]
        metrics = comparable(read_jsonl(run_dir / "metrics.jsonl"))
        evaluations = comparable(read_jsonl(run_dir / "evaluations.jsonl"))
        assert [e["step"] for e in evaluations] == list(steps)
        assert all(math.isfinite(e["K_benign"]) for e in evaluations)
        terminal = json.loads((run_dir / "terminal.json").read_text())
        assert (terminal["steps"], terminal["stop"]["reason"]) == (25, "steps")
        assert terminal["evaluations"] == list(steps)
        outcome[name] = {
            "params": [bits(p) for p in params],
            "state": [bits(s) for s in state],
            "training": [r for r in metrics if "loss" in r],
            "logged": [r for r in metrics if not any(k.startswith("eval/") for k in r)],
            "evaluated": [r for r in metrics if any(k.startswith("eval/") for k in r)],
            "records": [comparable(read_jsonl(run_dir / record)) for record in
                        ("monitor.jsonl", "diagnostics.jsonl", "update_scalars.jsonl")],
            "step_25": evaluations[-1],
            "stop": (terminal["stop"], terminal["steps"], terminal["responses"],
                     terminal["monitor_readings"], terminal["diagnostics"]),
        }  # fmt: skip
    endpoint, timecourse = outcome["endpoint"], outcome["timecourse"]
    assert all(torch.equal(a, b) for a, b in
               zip(endpoint["params"], timecourse["params"], strict=True))  # fmt: skip
    assert all(torch.equal(a, b) for a, b in
               zip(endpoint["state"], timecourse["state"], strict=True))  # fmt: skip
    assert [r["step"] for r in timecourse["training"]] == list(range(25))
    for key in ("training", "logged", "records", "step_25", "stop"):
        assert endpoint[key] == timecourse[key], key
    assert [r["step"] for r in timecourse["evaluated"]] == [t - 1 for t in TIMECOURSE_STEPS]
    assert endpoint["evaluated"] == timecourse["evaluated"][-1:]  # logged one step back
    assert endpoint["step_25"]["K_benign"] == timecourse["step_25"]["K_benign"]
    return dirs


@pytest.mark.parametrize("arm", list(ARMS))
def test_intermediate_evaluations_do_not_change_step_25(env, tmp_path, monkeypatch, arm) -> None:
    """Review focus of amendment 4's Task 5: the time course's evaluations at steps 1, 5 and 10
    change nothing in training, so its step 25 is its endpoint run's, bit for bit, on all three
    arms; the analysis' identity check reads the two runs as identical."""
    dirs = intermediate_evaluations(env, tmp_path, monkeypatch, arm)
    from zero_field_refusal.analysis.stage_a import load_trajectory
    from zero_field_refusal.analysis.study2_short import timecourse_identity

    timecourse, endpoint = (load_trajectory(dirs[name]) for name in ("timecourse", "endpoint"))
    assert timecourse.evaluate_at == TIMECOURSE_STEPS and endpoint.evaluate_at == (25,)
    found = timecourse_identity(timecourse, endpoint)
    assert found == {"state": "identical", "reasons": [], "endpoint": endpoint.run_id,
                     "bitwise_equal": True,
                     "rayleigh_equal": None}  # these runs do not measure R  # fmt: skip


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("arm", list(ARMS))
def test_cuda_intermediate_evaluations_do_not_change_step_25(
    gpu_env,  # noqa: F811
    tmp_path,
    monkeypatch,
    arm,
) -> None:
    """The same on CUDA, for each arm: FA-25 (Adam with v₀ frozen), AdamW-25 and pre-warmed
    AdamW-25 (review M11)."""
    from zero_field_refusal.analysis.stage_a import load_trajectory
    from zero_field_refusal.analysis.study2_short import timecourse_identity

    dirs = intermediate_evaluations(gpu_env, tmp_path, monkeypatch, arm,
                                    "execution.device=cuda")  # fmt: skip
    timecourse, endpoint = (load_trajectory(dirs[name]) for name in ("timecourse", "endpoint"))
    assert timecourse_identity(timecourse, endpoint)["state"] == "identical"
