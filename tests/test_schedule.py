"""Study-2 schedules: the stopping rules alone, then whole runs on the tiny CPU model.

In the runs the monitor reads a scripted sequence of K, so each test fixes where levels are
crossed; training, the scoring panel and the stored responses are real.
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch
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

        def read(self, model):
            return queue.pop(0) if len(queue) > 1 else queue[0]

        monkeypatch.setattr(Monitor, "__call__", read)
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
