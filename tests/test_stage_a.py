"""Stage A: run records, summaries and the selection rule, on synthetic run directories built as
``schedule.run_scheduled`` writes them, and once on a real scheduled run."""

import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from zero_field_refusal.analysis.stage_a import (
    TrajectoryError,
    agreement,
    cell_summary,
    instability_summary,
    load_trajectories,
    load_trajectory,
    numerical_acceptance,
    screen_candidates,
    select,
    short_grid_extension,
)

LEVELS = [5.0, math.sqrt(35), math.sqrt(40), 7.0, 8.0]
THETA0 = {"benchmark_accuracy": 0.60, "factual_accuracy": 0.80}
SEEDS = [2000, 2001, 2002]
OPTIMIZER = {
    "sgd_long": "sgd",
    "adamw_long": "adamw",
    "adamw_25": "adamw",
    "frozen_adam_25": "adam_frozen",
    "soft_control": "sgd",
}


def write_run(
    root: Path,
    name: str,
    *,
    arm: str,
    batch: int,
    lr: float,
    seed: int,
    reach: dict[float, int] | None = None,
    steps: int | None = None,
    stop: str | None = None,
    final_k: float | None = None,
    mmlu_loss: float = 2.0,
    factual_loss: float = 3.0,
    seconds: float = 100.0,
    extension=(0.0, 0),
    evaluate_at=(25,),
    detail=None,
    grad_norms=None,
    monitor_bias: float = 0.0,
    train_steps: int = 4000,
) -> Path:
    """One run directory. A long arm reaches each level of ``reach`` at its step; a fixed arm
    is evaluated at ``evaluate_at`` with K_benign ``final_k``."""
    fixed = arm in ("adamw_25", "frozen_adam_25", "soft_control")
    utility = {
        "benchmark_accuracy": THETA0["benchmark_accuracy"] - mmlu_loss / 100,
        "factual_accuracy": THETA0["factual_accuracy"] - factual_loss / 100,
    }
    evaluations, reached = [], []
    if fixed:
        steps = steps or max(evaluate_at)
        stop = stop or "steps"
        for t in evaluate_at:
            if t <= steps and not (stop != "steps" and t == steps):
                evaluations.append(
                    {
                        "step": t,
                        "levels": [],
                        "K_monitor": final_k + monitor_bias,
                        "K_benign": final_k,
                        "K_harmful": final_k,
                        **utility,
                    }
                )
    else:
        reach = reach or {}
        for step in sorted(set(reach.values())):
            levels = [level for level in LEVELS if reach.get(level) == step]
            k = max(levels) + 0.05
            evaluations.append(
                {
                    "step": step,
                    "levels": levels,
                    "K_monitor": k + monitor_bias,
                    "K_benign": k,
                    "K_harmful": k,
                    **utility,
                }
            )
            reached += [{"level": level, "step": step, "K_benign": k} for level in levels]
        if steps is None:
            steps = max(reach.values()) if reach.get(8.0) else train_steps
        stop = stop or ("target" if reach.get(8.0) else "t_max")
    every = 5
    eval_k = {row["step"]: row["K_monitor"] for row in evaluations}
    monitor = [
        {"step": s, "K_monitor": eval_k.get(s, 0.01 * s)} for s in range(every, steps + 1, every)
    ]
    norms = grad_norms or [1.0] * steps
    metrics = [{"step": s, "loss": 1.0, "grad_norm": norms[s]} for s in range(steps)]
    terminal = {
        "stop": {"reason": stop, "kind": "x", **({"detail": detail} if detail else {})},
        "steps": steps,
        "responses": steps * batch,
        "stream_extension": {"seconds": extension[0], "chunks": extension[1]},
    }
    if not fixed:
        terminal["levels"] = {
            "planned": LEVELS,
            "reached": reached,
            "missed": [lvl for lvl in LEVELS if lvl not in reach],
        }
    cfg = {
        "seed": seed,
        "control": "soft_targets" if arm == "soft_control" else None,
        "optimizer": {"name": OPTIMIZER[arm], "lr": lr},
        "train": {"steps": steps if fixed else train_steps, "batch_size": batch},
        "teacher": {"chunk_size": 1000},
        "eval": {"monitor": {"every": every}},
        "schedule": {
            "policy": "fixed" if fixed else "k_stopped",
            "levels": LEVELS,
            "evaluate_at": list(evaluate_at) if fixed else None,
            "plateau": {"from_step": 1000},
        },
    }
    started = datetime(2026, 10, 15, 12, 0, 0)
    meta = {
        "status": "completed",
        "started_at": started.isoformat(),
        "finished_at": (started + timedelta(seconds=seconds)).isoformat(),
        "inputs": {"streams": {"training": {"count": steps * batch}}},
    }
    run_dir = root / name
    run_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "config.yaml")
    (run_dir / "meta.json").write_text(json.dumps(meta))
    (run_dir / "terminal.json").write_text(json.dumps(terminal))
    for file, rows in (("evaluations", evaluations), ("monitor", monitor), ("metrics", metrics)):
        (run_dir / f"{file}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run_dir


FULL = {5.0: 100, math.sqrt(35): 150, math.sqrt(40): 200, 7.0: 300, 8.0: 400}


def test_consistent_records_load(tmp_path) -> None:
    run = load_trajectory(
        write_run(tmp_path, "r", arm="adamw_long", batch=5, lr=3e-6, seed=2000, reach=FULL)
    )
    assert (run.arm, run.batch_size, run.lr, run.seed, run.steps) == (
        "adamw_long",
        5,
        3e-6,
        2000,
        400,
    )
    assert run.evaluation_at(math.sqrt(35))["step"] == 150 and run.evaluation_at(9.0) is None
    assert run.seconds == 100.0 and run.stable


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (
            lambda d: _edit(d / "terminal.json", lambda t: t["levels"]["reached"].pop()),
            "levels differ",
        ),
        (lambda d: _drop_line(d / "monitor.jsonl", 3), "monitor readings"),
        (lambda d: _edit(d / "meta.json", lambda m: m.update(status="failed")), "status"),
        (
            lambda d: _edit(
                d / "meta.json", lambda m: m["inputs"]["streams"]["training"].update(count=7)
            ),
            "stream prefix",
        ),
        (lambda d: _drop_line(d / "metrics.jsonl", 0), "training metrics"),
    ],
)
def test_inconsistent_records_are_refused(tmp_path, damage, message) -> None:
    run_dir = write_run(tmp_path, "r", arm="sgd_long", batch=5, lr=0.004, seed=2000, reach=FULL)
    damage(run_dir)
    with pytest.raises(TrajectoryError, match=message):
        load_trajectory(run_dir)


def _edit(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def _drop_line(path: Path, index: int) -> None:
    lines = path.read_text().splitlines(keepends=True)
    del lines[index]
    path.write_text("".join(lines))


def test_a_cell_summary_gives_reachability_steps_and_utility_per_level(tmp_path) -> None:
    runs = [
        load_trajectory(
            write_run(
                tmp_path,
                f"s{seed}",
                arm="sgd_long",
                batch=5,
                lr=0.004,
                seed=seed,
                reach={**FULL, 8.0: 400 + 100 * i} if i < 2 else {5.0: 100, 7.0: 300},
                mmlu_loss=2.0 + i,
            )
        )
        for i, seed in enumerate(SEEDS)
    ]
    summary = cell_summary(runs, THETA0, SEEDS)
    seven, eight = summary["levels"]["7.0000"], summary["levels"]["8.0000"]
    assert seven["all_seeds"] and seven["median_steps"] == 300
    assert eight["reached"] == [2000, 2001] and not eight["all_seeds"]
    assert eight["mmlu_loss_pp"] == pytest.approx(2.5)
    assert summary["stops"] == {"target": 2, "t_max": 1}


def test_screen_candidates_and_numerical_acceptance(tmp_path) -> None:
    screen = [
        load_trajectory(
            write_run(
                tmp_path,
                f"screen-{lr}",
                arm="soft_control",
                batch=5,
                lr=lr,
                seed=2000,
                final_k=k,
                evaluate_at=(100,),
            )
        )
        for lr, k in ((0.002, 0.01), (0.004, 0.03), (0.006, 0.05), (0.01, 0.2))
    ]
    candidates = screen_candidates(screen)
    assert candidates[5]["candidates"] == [0.006, 0.004]  # 0.01: 0.2 > 0.059
    regular = [
        load_trajectory(
            write_run(tmp_path, f"reg-{lr}", arm="sgd_long", batch=5, lr=lr, seed=2000, reach=FULL)
        )
        for lr in (0.004, 0.006)
    ]
    at = (100, 150, 200, 300, 400)
    controls = [
        load_trajectory(
            write_run(
                tmp_path,
                "full-0.004",
                arm="soft_control",
                batch=5,
                lr=0.004,
                seed=2000,
                final_k=0.04,
                evaluate_at=at,
            )
        ),
        load_trajectory(
            write_run(
                tmp_path,
                "full-0.006",
                arm="soft_control",
                batch=5,
                lr=0.006,
                seed=2000,
                final_k=0.04,
                evaluate_at=(100, 400),
            )
        ),
    ]
    acceptance = numerical_acceptance(controls, regular)
    assert acceptance[(5, 0.004)]["accepted"]
    assert not acceptance[(5, 0.006)]["accepted"]
    assert "regular run's steps [150, 200, 300]" in acceptance[(5, 0.006)]["reasons"][0]


def test_false_alarms_are_gradient_stops_in_cells_that_ran_stably(tmp_path) -> None:
    runs = [
        load_trajectory(
            write_run(tmp_path, "a0", arm="sgd_long", batch=5, lr=0.004, seed=2000, reach=FULL)
        ),
        load_trajectory(
            write_run(
                tmp_path,
                "a1",
                arm="sgd_long",
                batch=5,
                lr=0.004,
                seed=2001,
                steps=50,
                stop="gradient_spike",
                detail={"grad_norm": 12.0, "reference_median": 1.0},
            )
        ),
        load_trajectory(
            write_run(
                tmp_path,
                "b0",
                arm="sgd_long",
                batch=5,
                lr=0.03,
                seed=2000,
                steps=6,
                stop="gradient_spike",
                detail={"grad_norm": 11.0, "reference_median": 1.0},
            )
        ),
    ]
    summary = instability_summary(runs)
    assert summary["firings"] == ["a1", "b0"] and summary["false_alarms"] == ["a1"]
    assert summary["false_alarm_rate"] == pytest.approx(0.5)  # one of the two runs at 0.004


def test_agreement_pairs_each_evaluation_with_its_reading(tmp_path) -> None:
    runs = [
        load_trajectory(
            write_run(
                tmp_path,
                f"r{bias}",
                arm="sgd_long",
                batch=5,
                lr=0.004,
                seed=2000 + i,
                reach=FULL,
                monitor_bias=bias,
            )
        )
        for i, bias in enumerate((0.0, 0.1))
    ]
    result = agreement(runs)
    assert result["pairs"] == 10 and result["passes"] is True


# The selection rule.
def discovery(
    root: Path,
    *,
    sgd: dict,
    adamw: dict,
    short: dict,
    frozen: dict | None = None,
    seconds: float = 100.0,
    batch: int = 5,
) -> None:
    """Discovery runs of one batch: per rate, the levels reached by every seed (or per seed)."""
    for arm, rates in (("sgd_long", sgd), ("adamw_long", adamw)):
        for lr, reach in rates.items():
            for i, seed in enumerate(SEEDS):
                seed_reach = reach[i] if isinstance(reach, list) else reach
                write_run(
                    root,
                    f"{arm}-{batch}-{lr}-{seed}",
                    arm=arm,
                    batch=batch,
                    lr=lr,
                    seed=seed,
                    reach=seed_reach,
                    seconds=seconds,
                )
    for arm, rates in (("adamw_25", short), ("frozen_adam_25", frozen or short)):
        for lr, k in rates.items():
            for seed in SEEDS:
                write_run(
                    root,
                    f"{arm}-{batch}-{lr}-{seed}",
                    arm=arm,
                    batch=batch,
                    lr=lr,
                    seed=seed,
                    final_k=k,
                    seconds=seconds / 10,
                )


UP_TO_7 = {5.0: 100, math.sqrt(35): 150, math.sqrt(40): 200, 7.0: 300}
SHORT = {0.5e-5: 2.0, 1e-5: 4.6, 1.5e-5: 6.0, 2e-5: 7.4, 3e-5: 8.3, 4e-5: 11.0}


def accept(*keys):
    return {key: {"accepted": True} for key in keys}


def test_the_larger_u_wins_and_the_rule_fixes_rates_and_t_max(tmp_path) -> None:
    discovery(
        tmp_path / "b5",
        sgd={0.004: FULL, 0.006: FULL},
        adamw={1e-6: UP_TO_7, 3e-6: FULL, 5e-6: FULL},
        short=SHORT,
        batch=5,
    )
    discovery(tmp_path / "b20", sgd={0.004: UP_TO_7}, adamw={3e-6: FULL}, short=SHORT, batch=20)
    runs = load_trajectories(tmp_path)
    result = select(runs, THETA0, accept((5, 0.004), (5, 0.006), (20, 0.004)), SEEDS)
    assert result["outcome"] == "selected" and result["batch_size"] == 5 and result["U"] == 8.0
    assert result["K_star"] == pytest.approx(math.sqrt(40))
    assert result["sgd_long"] == {"lr": 0.006, "median_steps_to_U": 400}
    assert result["adamw_long"]["lr"] == 5e-6  # the largest that reaches U
    assert result["t_max"] == 600  # 1.5 × 400
    assert result["adamw_25"]["rates"] == [1e-5, 1.5e-5, 3e-5]  # nearest 5, √40 ≈ 6.32, 8
    assert result["adamw_25"]["shortfall"] == 0
    assert result["batches"][20]["U"] == 7.0 and not result["u_limited_by_reachability"]


def test_a_tie_in_u_goes_to_the_cheaper_block(tmp_path) -> None:
    discovery(
        tmp_path / "b5", sgd={0.004: FULL}, adamw={3e-6: FULL}, short=SHORT, seconds=300, batch=5
    )
    discovery(
        tmp_path / "b20", sgd={0.004: FULL}, adamw={3e-6: FULL}, short=SHORT, seconds=200, batch=20
    )
    result = select(load_trajectories(tmp_path), THETA0, accept((5, 0.004), (20, 0.004)), SEEDS)
    assert result["batch_size"] == 20
    assert result["batches"][20]["seconds_per_block"] < result["batches"][5]["seconds_per_block"]


def test_u_lowered_by_reachability_is_flagged(tmp_path) -> None:
    misses = [FULL, FULL, UP_TO_7]  # one seed of SGD does not reach 8
    discovery(tmp_path, sgd={0.004: misses}, adamw={3e-6: FULL}, short=SHORT)
    result = select(load_trajectories(tmp_path), THETA0, accept((5, 0.004)), SEEDS)
    assert result["U"] == 7.0 and result["u_limited_by_reachability"]
    reasons = result["batches"][5]["levels"]["8.0000"]["sgd_long"][0.004]["reasons"]
    assert reasons == ["U = 8 reached by seeds [2000, 2001] only"]


def test_adamw_long_needs_three_separate_evaluations(tmp_path) -> None:
    jump = {5.0: 100, math.sqrt(35): 100, math.sqrt(40): 100, 7.0: 100, 8.0: 100}
    discovery(tmp_path, sgd={0.004: FULL}, adamw={3e-6: FULL, 1e-5: jump}, short=SHORT)
    result = select(load_trajectories(tmp_path), THETA0, accept((5, 0.004)), SEEDS)
    assert result["adamw_long"]["lr"] == 3e-6  # 1e-5 reached every level at once
    reasons = result["batches"][5]["levels"]["8.0000"]["adamw_long"][1e-5]["reasons"]
    assert any("three separate evaluations" in r for r in reasons)


def test_too_few_short_rates_are_taken_and_flagged(tmp_path) -> None:
    discovery(tmp_path, sgd={0.004: FULL}, adamw={3e-6: FULL}, short={1e-5: 5.2, 2e-5: 7.9})
    result = select(load_trajectories(tmp_path), THETA0, accept((5, 0.004)), SEEDS)
    assert result["adamw_25"]["rates"] == [1e-5, 2e-5]
    assert result["adamw_25"]["targets"] == [5.0, 8.0] and result["adamw_25"]["shortfall"] == 1


@pytest.mark.parametrize(
    ("acceptance", "sgd", "outcome"),
    [
        ({}, {0.004: FULL}, "numerics_fail"),
        (accept((5, 0.004)), {0.004: {5.0: 100, math.sqrt(35): 200}}, "barrier"),
    ],
)
def test_stage_a_outcomes_that_stop_the_design(tmp_path, acceptance, sgd, outcome) -> None:
    discovery(tmp_path, sgd=sgd, adamw={3e-6: FULL}, short=SHORT)
    assert select(load_trajectories(tmp_path), THETA0, acceptance, SEEDS)["outcome"] == outcome


def test_harm_is_never_read(tmp_path) -> None:
    # The rule reads K, steps, utility and time: a responses file is not even needed.
    discovery(tmp_path, sgd={0.004: FULL}, adamw={3e-6: FULL}, short=SHORT)
    assert not list(tmp_path.rglob("responses"))
    assert select(load_trajectories(tmp_path), THETA0, accept((5, 0.004)), SEEDS)["outcome"] == (
        "selected"
    )


def test_the_short_grid_extends_down_by_halving_at_most_twice(tmp_path) -> None:
    def runs_at(rates):
        root = tmp_path / str(len(list(tmp_path.iterdir())))
        for lr, k in rates.items():
            for seed in SEEDS:
                write_run(
                    root, f"{lr}-{seed}", arm="adamw_25", batch=5, lr=lr, seed=seed, final_k=k
                )
        return load_trajectories(root)

    assert short_grid_extension(runs_at({0.5e-5: 4.0, 1e-5: 6.0}), "adamw_25") is None
    assert short_grid_extension(runs_at({0.5e-5: 5.5}), "adamw_25") == 0.25e-5
    assert short_grid_extension(runs_at({0.25e-5: 5.5}), "adamw_25") == 0.125e-5
    assert short_grid_extension(runs_at({0.125e-5: 5.5}), "adamw_25") is None  # two halvings


def test_the_script_writes_the_summary_and_the_selection_once(tmp_path, monkeypatch) -> None:
    import runpy

    discovery(tmp_path / "discovery", sgd={0.004: FULL}, adamw={3e-6: FULL}, short=SHORT)
    at = (100, 150, 200, 300, 400)
    write_run(
        tmp_path / "controls",
        "full",
        arm="soft_control",
        batch=5,
        lr=0.004,
        seed=2000,
        final_k=0.04,
        evaluate_at=at,
    )
    theta0 = tmp_path / "theta0"
    theta0.mkdir()
    (theta0 / "summary.json").write_text(json.dumps(THETA0))
    out = tmp_path / "stage-a.json"
    script = Path(__file__).parents[1] / "scripts" / "stage_a_summary.py"
    argv = [
        str(script),
        "--discovery",
        str(tmp_path / "discovery"),
        "--controls",
        str(tmp_path / "controls"),
        "--theta0",
        str(theta0),
        "--out",
        str(out),
    ]
    monkeypatch.setattr("sys.argv", argv)
    runpy.run_path(str(script), run_name="__main__")
    result = json.loads(out.read_text())
    assert result["selection"]["outcome"] == "selected"
    assert result["acceptance"]["B=5 lr=0.004"]["accepted"]
    assert result["code"]["commit"] and result["theta0"]["benchmark_accuracy"] == 0.60
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")  # never overwritten
