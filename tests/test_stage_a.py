"""Stage A: run records, summaries and the selection rule, on synthetic run directories built as
``schedule.run_scheduled`` writes them, over the whole prescribed search."""

import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from zero_field_refusal.analysis.stage_a import (
    BATCHES,
    DISCOVERY_SEEDS,
    SGD_GRID,
    TrajectoryError,
    agreement,
    cell_summary,
    control_schedule,
    instability_summary,
    load_trajectories,
    load_trajectory,
    monitor_qualification,
    prescribed,
    screen_candidates,
    select,
    trigger_checks,
)

SPEC = prescribed()  # the schedules, model and protocol of the study-2 configs
LEVELS = SPEC["long"]["levels"]
THETA0 = {"benchmark_accuracy": 0.60, "factual_accuracy": 0.80}
DIAGNOSTIC_EVERY = SPEC["monitor"]["diagnostics_every"]
MONITOR = {k: SPEC["monitor"][k] for k in ("prompt_set", "samples_per_prompt", "drive")}
SQRT35, SQRT40 = LEVELS[1], LEVELS[2]
SEEDS = list(DISCOVERY_SEEDS)
PROMPT_IDS = ["evaluation:0", "evaluation:1", "evaluation:2"]
PROMPTS = {
    "set": "harmful_evaluation",
    "count": 3,
    "ids_sha256": hashlib.sha256(json.dumps(PROMPT_IDS).encode()).hexdigest(),
}
OPTIMIZER = {
    "sgd_long": "sgd",
    "adamw_long": "adamw",
    "adamw_25": "adamw",
    "frozen_adam_25": "adam_frozen",
    "soft_control": "sgd",
}
FULL = {5.0: 100, math.sqrt(35): 150, math.sqrt(40): 200, 7.0: 300, 8.0: 400}
UP_TO_7 = {5.0: 100, math.sqrt(35): 150, math.sqrt(40): 200, 7.0: 300}
SHORT = {0.5e-5: 2.0, 1e-5: 4.6, 1.5e-5: 6.0, 2e-5: 7.4, 3e-5: 8.3, 4e-5: 11.0}


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
    train_steps: int | None = None,
    stream_path: str | None = None,
    status: str = "completed",
    role: str = "discovery",
    model: str | None = None,
    monitor_version: int = 1,
    diag_bias: tuple[float, float] = (0.0, 0.0),
    cache_ok: bool = True,
) -> Path:
    """One run directory. A long arm reaches each level of ``reach`` at its step; a fixed arm
    (25-step arms, soft-target controls) is evaluated at ``evaluate_at`` with K_benign
    ``final_k``."""
    fixed = arm in ("adamw_25", "frozen_adam_25", "soft_control")
    train_steps = train_steps or SPEC["long"]["train_steps"]
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
                        "condition": f"step-{t}",
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
                    "condition": f"step-{step}",
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
    points = [(0, 0.0), *sorted(eval_k.items())]

    def reading(s: int) -> float:  # version 2: K rises through the evaluations, then holds
        if s in eval_k or monitor_version == 1 or len(points) == 1:
            return eval_k.get(s, 0.01 * s)
        for (s0, k0), (s1, k1) in itertools.pairwise(points):
            if s0 <= s <= s1:
                return k0 + (k1 - k0) * (s - s0) / (s1 - s0)
        return points[-1][1]

    monitor = [{"step": s, "K_monitor": reading(s)} for s in range(every, steps + 1, every)]
    norms = grad_norms or [1.0] * steps
    metrics = [{"step": s, "loss": 1.0, "grad_norm": norms[s]} for s in range(steps)]
    terminal = {
        "stop": {"reason": stop, "kind": "x", **({"detail": detail} if detail else {})},
        "steps": steps,
        "responses": steps * batch,
        "seconds": seconds,
        "stream_extension": {"seconds": extension[0], "chunks": extension[1]},
    }
    if not fixed:
        terminal["levels"] = {
            "planned": LEVELS,
            "reached": reached,
            "missed": [lvl for lvl in LEVELS if lvl not in reach],
        }
    spec = SPEC["control" if arm == "soft_control" else "long" if not fixed else "short"]
    name_, revision = (model or SPEC["model"]).split("@")
    cfg = {
        "seed": seed,
        "protocol_version": SPEC["protocol_version"],
        "stage_a": {"role": role},
        "control": "soft_targets" if arm == "soft_control" else None,
        "model": {"name": name_, "revision": revision},
        "optimizer": {"name": OPTIMIZER[arm], "lr": lr},
        "train": {"steps": steps if fixed else train_steps, "batch_size": batch},
        "teacher": {"chunk_size": spec["chunk_size"]},
        "eval": {
            "monitor": {"every": spec["monitor_every"], "version": monitor_version}
            | (MONITOR if monitor_version > 1 else {}),
            "diagnostics": {"every": DIAGNOSTIC_EVERY},
            "harmful": {"prompt_set": spec["harmful_set"]},
        },
        "schedule": {
            "policy": spec["policy"],
            "levels": spec["levels"],
            "evaluate_at": list(evaluate_at) if fixed else None,
            "crossing_readings": spec["crossing_readings"],
            "plateau": spec["plateau"],
            "instability": spec["instability"],
            "extend_stream": spec["extend_stream"],
        },
    }
    chunks = [f"chunk-{seed}-{i}" for i in range(-(-steps * batch // 1000))]
    stream = {
        "path": stream_path or f"teacher/qwen/training/abc/seed-{seed}",
        "count": steps * batch,
        "sha256": "s",
        "chunks": chunks,
    }
    meta = {
        "status": status,
        "inputs": {"streams": {"training": stream}, "harmful_prompts": PROMPTS},
    }
    run_dir = root / name
    run_dir.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "config.yaml")
    (run_dir / "meta.json").write_text(json.dumps(meta))
    (run_dir / "terminal.json").write_text(json.dumps(terminal))
    responses = [
        {"condition": row["condition"], "item_id": i, "response_ids": [1, 2], "text": "x"}
        for row in evaluations
        for i in PROMPT_IDS
    ]
    files = [
        ("evaluations", evaluations),
        ("monitor", monitor),
        ("metrics", metrics),
        ("responses/harmful_evaluation", responses),
    ]
    if monitor_version > 1:  # amendment 1: diagnostics with m = 1 and m = 2 off by diag_bias
        m1, m2 = diag_bias
        read = {row["step"]: row["K_monitor"] for row in monitor}
        diagnostics = [
            {"step": t, "K_monitor": read[t], "K_scoring": read[t] / (1 + m1),
             "K_monitor_full": read[t] / (1 + m1) * (1 + m2)}
            for t in range(DIAGNOSTIC_EVERY, steps + 1, DIAGNOSTIC_EVERY)
        ]  # fmt: skip
        for row in evaluations:
            row["K_monitor_full"] = row["K_benign"] * (1 + m2)
        check = {"bitwise_equal": cache_ok, "responses": 1, "max_abs_diff": 0.0}
        checks = [
            {"step": t, "primary": check, "secondary": check, "scoring": check}
            for t in [0, *(row["step"] for row in evaluations)]
        ]
        files += [("diagnostics", diagnostics), ("cache_checks", checks)]
    for file, rows in files:
        (run_dir / f"{file}.jsonl").parent.mkdir(exist_ok=True)
        (run_dir / f"{file}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run_dir / "summary.json").write_text(json.dumps({"terminal": terminal}))
    return run_dir


def search(
    root: Path,
    *,
    screen=None,
    sgd=None,
    adamw=None,
    short=None,
    frozen=None,
    controls=None,
    seconds=None,
    prune=("sgd_long", "adamw_long"),
) -> None:
    """The whole prescribed search. ``screen`` maps (batch, rate) to the 100-step control's
    contamination (default: rates up to 0.004 pass); ``sgd``, ``adamw`` map batch → rate →
    reach (a dict, or one per seed), run under monitor version 2 in the search's downward order
    and, for the arms in ``prune``, no further than the first rate every seed takes to 8 (the
    descent's stop when that rate meets its rule); ``short``/``frozen`` map batch → rate → K at
    step 25;
    ``controls`` maps (batch, rate) to the full control's contamination (default 0.04 for every
    SGD rate run). A short rate's value may be (K, MMLU loss in pp)."""
    screen = screen or {(b, lr): (0.01 if lr <= 0.004 else 0.2) for b in BATCHES for lr in SGD_GRID}
    for (batch, lr), c in screen.items():
        write_run(
            root / "screen",
            f"screen-{batch}-{lr}",
            arm="soft_control",
            batch=batch,
            lr=lr,
            seed=2000,
            final_k=c,
            evaluate_at=(100,),
            role="screen",
        )
    sgd = sgd or {5: {0.004: FULL, 0.003: FULL}, 20: {0.004: UP_TO_7, 0.003: UP_TO_7}}
    adamw = adamw or {b: {1e-6: UP_TO_7, 3e-6: FULL, 5e-6: FULL, 1e-5: FULL} for b in BATCHES}
    seconds = seconds or {5: 100.0, 20: 100.0}
    ran: dict[int, list[float]] = {b: [] for b in BATCHES}  # SGD rates run, for their controls
    for arm, rates in (("sgd_long", sgd), ("adamw_long", adamw)):
        for batch, by_rate in rates.items():
            for lr in sorted(by_rate, reverse=True):
                reach = by_rate[lr]
                per_seed = reach if isinstance(reach, list) else [reach] * len(SEEDS)
                for seed, seed_reach in zip(SEEDS, per_seed, strict=True):
                    write_run(
                        root / "discovery",
                        f"{arm}-{batch}-{lr}-{seed}",
                        arm=arm,
                        batch=batch,
                        lr=lr,
                        seed=seed,
                        seconds=seconds[batch],
                        reach=seed_reach,
                        monitor_version=2,
                    )
                if arm == "sgd_long":
                    ran[batch].append(lr)
                if arm in prune and all(8.0 in r for r in per_seed):
                    break
    for arm, rates in (("adamw_25", short), ("frozen_adam_25", frozen or short)):
        for batch, by_rate in (rates or {b: SHORT for b in BATCHES}).items():
            for lr, value in by_rate.items():
                k, mmlu = value if isinstance(value, tuple) else (value, 2.0)
                for seed in SEEDS:
                    write_run(
                        root / "discovery",
                        f"{arm}-{batch}-{lr}-{seed}",
                        arm=arm,
                        batch=batch,
                        lr=lr,
                        seed=seed,
                        final_k=k,
                        mmlu_loss=mmlu,
                        seconds=seconds[batch] / 10,
                    )
    regular = load_trajectories(root / "discovery")
    for (batch, lr), c in (controls or {(b, lr): 0.04 for b in ran for lr in ran[b]}).items():
        twins = [r for r in regular if (r.arm, r.batch_size, r.lr) == ("sgd_long", batch, lr)]
        plan = control_schedule(twins)
        write_run(
            root / "controls",
            f"control-{batch}-{lr}",
            arm="soft_control",
            batch=batch,
            lr=lr,
            seed=2000,
            final_k=c,
            evaluate_at=tuple(plan["seeds"][2000]),
            role="full_control",
        )


def chosen(root: Path) -> dict:
    runs = load_trajectories(root)
    role = {
        name: [r for r in runs if r.role == name]
        for name in ("discovery", "screen", "full_control")
    }
    return select(role["discovery"], role["screen"], role["full_control"], THETA0)


# Records.
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
    assert run.seconds == 100.0 and run.stable and run.harmful_prompts == PROMPTS


def test_a_published_outcome_loads_whatever_the_status(tmp_path) -> None:
    run = load_trajectory(
        write_run(
            tmp_path, "r", arm="sgd_long", batch=5, lr=0.004, seed=2000, reach=FULL, status="failed"
        )
    )
    assert run.status == "failed" and run.stop == "target"


def _edit(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def _drop_line(path: Path, index: int) -> None:
    lines = path.read_text().splitlines(keepends=True)
    del lines[index]
    path.write_text("".join(lines))


def _duplicate_line(path: Path, index: int) -> None:
    lines = path.read_text().splitlines(keepends=True)
    lines[index + 1] = lines[index]
    path.write_text("".join(lines))


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (
            lambda d: _edit(d / "terminal.json", lambda t: t["levels"]["reached"].pop()),
            "levels differ",
        ),
        (lambda d: _drop_line(d / "monitor.jsonl", 3), "monitor readings"),
        (
            lambda d: _edit(
                d / "meta.json", lambda m: m["inputs"]["streams"]["training"].update(count=7)
            ),
            "stream prefix",
        ),
        (lambda d: _drop_line(d / "metrics.jsonl", 0), "training metrics"),
        (
            lambda d: _drop_line(d / "responses" / "harmful_evaluation.jsonl", 0),
            "not the recorded 3 prompts",
        ),
        (
            lambda d: _duplicate_line(d / "responses" / "harmful_evaluation.jsonl", 0),
            "not the recorded 3 prompts",
        ),
        (
            lambda d: _edit(d / "meta.json", lambda m: m["inputs"].pop("harmful_prompts")),
            "does not record its harmful prompts",
        ),
        (lambda d: (d / "evaluations.jsonl").write_text(""), "levels differ"),
        (
            lambda d: _edit(d / "summary.json", lambda m: m["terminal"].update(steps=1)),
            "summary.json's terminal",
        ),
        (lambda d: _edit(d / "terminal.json", lambda t: t.pop("seconds")), "run's time"),
        (lambda d: (d / "terminal.json").write_text("{"), "unreadable"),
    ],
)
def test_inconsistent_records_are_refused(tmp_path, damage, message) -> None:
    run_dir = write_run(tmp_path, "r", arm="sgd_long", batch=5, lr=0.004, seed=2000, reach=FULL)
    damage(run_dir)
    with pytest.raises(TrajectoryError, match=message):
        load_trajectory(run_dir)


def test_a_fixed_run_with_one_evaluation_must_answer_every_prompt(tmp_path) -> None:
    run_dir = write_run(tmp_path, "r", arm="adamw_25", batch=5, lr=1e-5, seed=2000, final_k=5.0)
    _drop_line(run_dir / "responses" / "harmful_evaluation.jsonl", 2)
    with pytest.raises(TrajectoryError, match="not the recorded 3 prompts"):
        load_trajectory(run_dir)


# Summaries.
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
                reach={**FULL, 8.0: 400 + 100 * i} if i < 2 else UP_TO_7,
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


def test_possible_false_alarms_are_gradient_stops_in_cells_that_ran_stably(tmp_path) -> None:
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
    assert summary["firings"] == ["a1", "b0"] and summary["possible_false_alarms"] == ["a1"]
    assert summary["possible_false_alarm_rate"] == pytest.approx(0.5)
    assert summary["margin_on_stable_runs"] == 1.0
    assert summary["factor"] == {"current": 10.0, "revise": True, "new": 10.0}


@pytest.mark.parametrize(
    ("peak", "revise", "new"),
    [(5.0, False, 10.0), (7.0, False, 10.0), (8.0, True, 12.0), (6.0, False, 10.0)],
)
def test_the_factor_rises_only_on_a_wide_margin_and_never_falls(tmp_path, peak, revise, new):
    norms = [1.0] * 10 + [peak] + [1.0] * 389
    run = load_trajectory(
        write_run(
            tmp_path,
            "r",
            arm="sgd_long",
            batch=5,
            lr=0.004,
            seed=2000,
            reach=FULL,
            grad_norms=norms,
        )
    )
    summary = instability_summary([run])
    assert summary["margin_on_stable_runs"] == pytest.approx(peak)
    assert summary["factor"] == {"current": 10.0, "revise": revise, "new": new}


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


def test_a_control_is_as_long_as_the_longest_regular_run_and_evaluated_at_its_steps(tmp_path):
    runs = [
        load_trajectory(
            write_run(
                tmp_path, f"r{i}", arm="sgd_long", batch=5, lr=0.004, seed=2000 + i, reach=reach
            )
        )
        for i, reach in enumerate((FULL, {5.0: 100, 7.0: 300}))
    ]
    plan = control_schedule(runs)
    assert plan["steps"] == 4000
    assert plan["seeds"][2000] == [100, 150, 200, 300, 400, 4000]
    assert plan["seeds"][2001] == [100, 300, 4000]


def test_the_screen_needs_every_rate_and_names_two_candidates(tmp_path) -> None:
    search(tmp_path)
    screen = load_trajectories(tmp_path / "screen")
    result, problems = screen_candidates(screen)
    assert problems == [] and result[5]["candidates"] == [0.004, 0.003]
    _, problems = screen_candidates(screen[1:])
    assert len(problems) == 1 and "screen: missing" in problems[0]


# The selection rule.
def test_the_larger_u_wins_and_the_rule_fixes_rates_and_t_max(tmp_path) -> None:
    search(tmp_path)
    result = chosen(tmp_path)
    assert result["outcome"] == "selected" and result["batch_size"] == 5 and result["U"] == 8.0
    assert result["K_star"] == pytest.approx(math.sqrt(40))
    assert result["sgd_long"] == {"lr": 0.004, "median_steps_to_U": 400}
    assert result["adamw_long"]["lr"] == 1e-5  # the largest that reaches U
    assert result["t_max"] == 600
    assert result["adamw_25"]["rates"] == [1e-5, 1.5e-5, 3e-5]
    assert result["batches"][20]["U"] == 7.0 and not result["u_limited_by_reachability"]


def test_a_tie_in_u_goes_to_the_cheaper_block(tmp_path) -> None:
    search(
        tmp_path,
        sgd={b: {0.004: FULL, 0.003: FULL} for b in BATCHES},
        seconds={5: 300.0, 20: 200.0},
    )
    result = chosen(tmp_path)
    assert result["batch_size"] == 20
    assert result["batches"][20]["seconds_per_block"] < result["batches"][5]["seconds_per_block"]


def test_u_lowered_by_reachability_is_flagged(tmp_path) -> None:
    search(
        tmp_path,
        sgd={
            5: {0.004: [FULL, FULL, UP_TO_7], 0.003: UP_TO_7},
            20: {0.004: UP_TO_7, 0.003: UP_TO_7},
        },
    )
    result = chosen(tmp_path)
    assert result["U"] == 7.0 and result["u_limited_by_reachability"]
    reasons = result["batches"][5]["levels"]["8.0000"]["sgd_long"][0.004]["reasons"]
    assert reasons == ["U = 8 reached by seeds [2000, 2001] only"]


def test_adamw_long_needs_three_separate_evaluations(tmp_path) -> None:
    jump = dict.fromkeys(LEVELS, 100)  # 1e-5 reaches 8 in one evaluation: the descent goes on
    search(tmp_path, adamw={b: {5e-6: FULL, 1e-5: jump} for b in BATCHES}, prune=("sgd_long",))
    result = chosen(tmp_path)
    assert result["adamw_long"]["lr"] == 5e-6
    reasons = result["batches"][5]["levels"]["8.0000"]["adamw_long"][1e-5]["reasons"]
    assert any("three separate evaluations" in r for r in reasons)


def test_too_few_short_rates_are_taken_and_flagged(tmp_path) -> None:
    # Only two rates keep MMLU within 5 pp; the others lose 9 pp.
    out_of_bounds = {lr: (k, 9.0) for lr, k in SHORT.items()} | {1e-5: 5.2, 2e-5: 7.9}
    search(tmp_path, short={b: out_of_bounds for b in BATCHES})
    result = chosen(tmp_path)
    assert result["adamw_25"]["rates"] == [1e-5, 2e-5]
    assert result["adamw_25"]["targets"] == [5.0, 8.0] and result["adamw_25"]["shortfall"] == 1


def test_no_accepted_control_fails_the_numerics(tmp_path) -> None:
    # Every control fails, so the descent runs every SGD candidate.
    search(tmp_path, controls={(b, lr): 0.5 for b in BATCHES for lr in (0.004, 0.003)},
           prune=("adamw_long",))  # fmt: skip
    result = chosen(tmp_path)
    assert result["outcome"] == "numerics_fail", result.get("problems")


def test_accepted_sgd_that_reaches_seven_at_neither_batch_is_a_barrier(tmp_path) -> None:
    low = {5.0: 100, math.sqrt(35): 200}
    search(tmp_path, sgd={b: {0.004: low, 0.003: low} for b in BATCHES})
    assert chosen(tmp_path)["outcome"] == "barrier"


def _copy(root: Path, name: str, new: str) -> None:
    source = next(root.rglob(name))
    shutil.copytree(source, source.with_name(new))


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (
            lambda r: shutil.rmtree(next(r.rglob("adamw_long-5-1e-05-2001"))),
            "search: adamw_long/B=5 needs {'rate': 1e-05, 'seeds': [2001]}",
        ),
        (lambda r: _copy(r, "adamw_long-5-1e-05-2001", "again"), "two runs of"),
        (
            lambda r: write_run(
                r / "discovery",
                "off-grid",
                arm="adamw_long",
                batch=5,
                lr=2e-5,
                seed=2000,
                reach=FULL,
                monitor_version=2,
            ),
            "discovery: version-2 run outside the search ('adamw_long', 5, 2e-05, 2000)",
        ),
        (
            lambda r: write_run(
                r / "discovery",
                "not-a-candidate",
                arm="sgd_long",
                batch=5,
                lr=0.002,
                seed=2000,
                reach=FULL,
                monitor_version=2,
            ),
            "discovery: version-2 run outside the search ('sgd_long', 5, 0.002, 2000)",
        ),
        (lambda r: shutil.rmtree(next(r.rglob("screen-5-0.0005"))), "screen: missing"),
        (
            lambda r: shutil.rmtree(next(r.rglob("control-5-0.004"))),
            "search: sgd_long/B=5 needs {'rate': 0.004, 'control': 2000}",
        ),
        (lambda r: _copy(r, "control-5-0.004", "control-again"), "two runs of"),
        (
            lambda r: write_run(
                r / "controls",
                "other-seed",
                arm="soft_control",
                batch=20,
                lr=0.004,
                seed=9000,
                final_k=0.01,
                evaluate_at=(4000,),
                role="full_control",
            ),
            "controls: unexpected ('soft_control', 20, 0.004, 9000)",
        ),
    ],
)
def test_an_incomplete_search_is_no_conclusion(tmp_path, change, problem) -> None:
    search(tmp_path)
    change(tmp_path)
    result = chosen(tmp_path)
    assert result["outcome"] == "incomplete"
    assert any(problem in p for p in result["problems"]), result["problems"]


def test_a_control_on_another_stream_is_refused(tmp_path) -> None:
    search(tmp_path)
    control = next(tmp_path.rglob("control-5-0.004"))
    _edit(
        control / "meta.json",
        lambda m: m["inputs"]["streams"]["training"].update(path="teacher/other/seed-2000"),
    )
    result = chosen(tmp_path)
    assert result["outcome"] == "incomplete"
    assert any("not on the model and stream" in p for p in result["problems"])


def test_the_short_grid_extensions_are_part_of_the_search(tmp_path) -> None:
    high = SHORT | {0.5e-5: 5.5}  # the smallest rate already beyond 5 at batch 5
    search(tmp_path, short={5: high, 20: SHORT})
    result = chosen(tmp_path)
    assert result["outcome"] == "incomplete"
    assert any("missing ('adamw_25', 5, 2.5e-06" in p for p in result["problems"])
    for arm in ("adamw_25", "frozen_adam_25"):
        for seed in SEEDS:
            write_run(
                tmp_path / "discovery",
                f"{arm}-ext-{seed}",
                arm=arm,
                batch=5,
                lr=0.25e-5,
                seed=seed,
                final_k=4.0,
            )
    assert chosen(tmp_path)["outcome"] == "selected"


def test_harm_is_never_read(tmp_path) -> None:
    # The rule reads K, steps, utility and time. Of the stored responses it reads only which
    # prompts were answered; no judge label exists here.
    search(tmp_path)
    assert not list(tmp_path.rglob("judge"))
    assert chosen(tmp_path)["outcome"] == "selected"


def test_the_script_writes_the_summary_and_the_selection_once(tmp_path, monkeypatch) -> None:
    import runpy

    search(tmp_path / "runs")
    theta0 = tmp_path / "theta0"
    theta0.mkdir()
    (theta0 / "summary.json").write_text(json.dumps(THETA0))
    out = tmp_path / "stage-a.json"
    script = Path(__file__).parents[1] / "scripts" / "stage_a_summary.py"
    argv = [
        str(script),
        "--runs",
        str(tmp_path / "runs"),
        str(tmp_path / "runs" / "screen"),
        "--theta0",
        str(theta0),
        "--out",
        str(out),
    ]  # overlapping: each run once
    monkeypatch.setattr("sys.argv", argv)
    runpy.run_path(str(script), run_name="__main__")
    result = json.loads(out.read_text())
    assert result["selection"]["outcome"] == "selected"
    assert result["selection"]["acceptance"]["B=5 lr=0.004"]["accepted"]
    assert result["code"]["commit"] and result["theta0"]["benchmark_accuracy"] == 0.60
    with pytest.raises(SystemExit):
        runpy.run_path(str(script), run_name="__main__")


# Review r12: the conditions of the runs, not only their keys.
def _rewrite(root: Path, name: str, **changes) -> None:
    """Rewrite one discovery run of ``search`` with other conditions, keeping its key."""
    run_dir = next(root.rglob(name))
    parts = name.split("-")
    arm, batch, lr, seed = parts[0], int(parts[1]), float("-".join(parts[2:-1])), int(parts[-1])
    shutil.rmtree(run_dir)
    long = {"reach": FULL, "monitor_version": 2}  # the search's long runs are version 2
    kwargs = long if arm.endswith("long") else {"final_k": SHORT.get(lr, 5.0)}
    write_run(run_dir.parent, name, arm=arm, batch=batch, lr=lr, seed=seed, **kwargs | changes)


@pytest.mark.parametrize(
    ("name", "changes", "problem"),
    [
        ("adamw_long-5-1e-05-2001", {"model": "Qwen/Qwen2.5-0.5B-Instruct@abc"}, "under protocol"),
        (
            "adamw_long-5-1e-05-2001",
            {"stream_path": "teacher/qwen/training/abc/seed-2000"},
            "seed 2001: runs read 2 different streams",
        ),
        (
            "adamw_25-5-1e-05-2002",
            {"evaluate_at": (100,), "steps": 100},
            "departs from the prescribed schedule in ['evaluate_at', 'train_steps']",
        ),
        (
            "sgd_long-20-0.004-2000",
            {"train_steps": 3000, "reach": UP_TO_7},
            "departs from the prescribed schedule in ['train_steps']",
        ),
        ("adamw_long-20-1e-05-2002", {"role": "screen"}, "screen: unexpected"),
    ],
)
def test_runs_must_meet_the_prescribed_conditions(tmp_path, name, changes, problem) -> None:
    search(tmp_path)
    _rewrite(tmp_path, name, **changes)
    result = chosen(tmp_path)
    assert result["outcome"] == "incomplete"
    assert any(problem in p for p in result["problems"]), result["problems"]


def test_a_run_given_under_another_role_is_refused(tmp_path) -> None:
    search(tmp_path)
    runs = load_trajectories(tmp_path)
    by_role = {r: [t for t in runs if t.role == r] for r in ("discovery", "screen", "full_control")}
    result = select(
        by_role["discovery"] + by_role["full_control"][:1],
        by_role["screen"],
        by_role["full_control"][1:],
        THETA0,
    )
    assert result["outcome"] == "incomplete"
    assert any("has the role 'full_control'" in p for p in result["problems"])


@pytest.mark.parametrize(
    "evaluate_at",
    [
        (100, 150, 200, 300),  # stops at 300, the rate needs 400 steps
        (100, 200, 300, 400),  # misses the regular run's evaluation at 150
    ],
)
def test_a_control_off_its_plan_is_no_conclusion(tmp_path, evaluate_at) -> None:
    search(tmp_path)
    control = next(tmp_path.rglob("control-5-0.004"))
    shutil.rmtree(control)
    write_run(
        control.parent,
        "control-5-0.004",
        arm="soft_control",
        batch=5,
        lr=0.004,
        seed=2000,
        final_k=0.04,
        evaluate_at=evaluate_at,
        role="full_control",
    )
    result = chosen(tmp_path)
    assert result["outcome"] == "incomplete"  # not numerics_fail
    assert any("its rate needs 400 steps" in p for p in result["problems"])


def test_a_full_control_of_100_steps_is_known_by_its_role(tmp_path, monkeypatch) -> None:
    import runpy

    quick = {5.0: 20, math.sqrt(35): 40, math.sqrt(40): 60, 7.0: 80, 8.0: 100}
    search(tmp_path / "runs", sgd={b: {0.004: quick, 0.003: quick} for b in BATCHES})
    assert {t.planned_steps for t in load_trajectories(tmp_path / "runs" / "controls")} == {100}
    theta0 = tmp_path / "theta0"
    theta0.mkdir()
    (theta0 / "summary.json").write_text(json.dumps(THETA0))
    out = tmp_path / "stage-a.json"
    script = Path(__file__).parents[1] / "scripts" / "stage_a_summary.py"
    monkeypatch.setattr(
        "sys.argv",
        [str(script), "--runs", str(tmp_path / "runs"), "--theta0", str(theta0), "--out", str(out)],
    )
    runpy.run_path(str(script), run_name="__main__")
    assert json.loads(out.read_text())["selection"]["outcome"] == "selected"


def test_a_whole_screen_with_no_passing_rate_fails_the_numerics_at_once(tmp_path) -> None:
    search(tmp_path, screen={(b, lr): 0.5 for b in BATCHES for lr in SGD_GRID})
    shutil.rmtree(tmp_path / "discovery")  # nothing else is needed for this conclusion
    shutil.rmtree(tmp_path / "controls")
    screen = load_trajectories(tmp_path / "screen")
    assert select([], screen, [], THETA0)["outcome"] == "numerics_fail"


def test_monitor_version_2_records_are_loaded_and_checked(tmp_path) -> None:
    reach = {5.0: 500, SQRT35: 600, SQRT40: 650, 7.0: 700, 8.0: 800}
    run_dir = write_run(tmp_path, "v2", arm="adamw_long", batch=5, lr=1e-5, seed=2000, reach=reach,
                        monitor_version=2, diag_bias=(0.02, 0.01))  # fmt: skip
    run = load_trajectory(run_dir)
    assert run.monitor_version == 2 and [d["step"] for d in run.diagnostics] == list(
        range(25, 801, 25)
    )
    assert run.diagnostics[-1]["K_monitor"] == dict(run.monitor)[800]

    def damaged(name, edit):
        target = shutil.copytree(run_dir, tmp_path / name)
        edit(target)
        with pytest.raises(TrajectoryError):
            load_trajectory(target)

    def drop_line(file, index):
        def edit(target):
            lines = (target / file).read_text().splitlines(keepends=True)
            (target / file).write_text("".join(lines[:index] + lines[index + 1 :]))

        return edit

    damaged("no-diagnostic", drop_line("diagnostics.jsonl", 3))
    damaged("no-check", drop_line("cache_checks.jsonl", 0))

    def other_monitor(target):
        rows = [
            json.loads(line) for line in (target / "diagnostics.jsonl").read_text().splitlines()
        ]
        rows[2]["K_monitor"] += 0.1
        (target / "diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    damaged("other-monitor", other_monitor)
    failed = write_run(tmp_path, "bad-cache", arm="adamw_long", batch=5, lr=1e-5, seed=2001,
                       reach=reach, monitor_version=2, cache_ok=False)  # fmt: skip
    with pytest.raises(TrajectoryError, match="cache check"):
        load_trajectory(failed)


def qualification_runs(root, adamw_bias, sgd_bias, steps=4000) -> list:
    runs = []
    for seed in DISCOVERY_SEEDS:
        for arm, lr, bias in (("adamw_long", 1e-5, adamw_bias), ("sgd_long", 2e-3, sgd_bias)):
            run_dir = write_run(root, f"{arm}-{seed}", arm=arm, batch=20, lr=lr, seed=seed,
                                steps=steps, stop="t_max" if steps == 4000 else "plateau",
                                monitor_version=2, diag_bias=bias)  # fmt: skip
            runs.append(load_trajectory(run_dir))
    return runs


@pytest.mark.parametrize(
    ("adamw_bias", "sgd_bias", "choice"),
    [
        ((0.02, 0.01), (-0.03, 0.02), "m1"),
        ((0.02, 0.01), (-0.12, 0.03), "m2"),  # m = 1 fails for SGD only: both use m = 2
        ((0.02, 0.01), (-0.12, -0.10), "redesign"),
    ],
)
def test_the_monitor_qualifies_per_optimizer_by_k_only(tmp_path, adamw_bias, sgd_bias, choice):
    result = monitor_qualification(qualification_runs(tmp_path, adamw_bias, sgd_bias))
    assert result["choice"] == choice and result["undetermined"] == []
    for arm in ("adamw_long", "sgd_long"):
        assert result["arms"][arm]["m1"]["pairs"] >= 10 and len(result["arms"][arm]["runs"]) == 3


def test_too_few_states_in_the_zone_leave_the_qualification_undetermined(tmp_path) -> None:
    runs = qualification_runs(tmp_path, (0.0, 0.0), (0.0, 0.0), steps=1000)  # K >= 4 from 400
    one_seed = [run for run in runs if run.seed == 2000]
    result = monitor_qualification(one_seed)  # 1000 steps: 25 states from 400 per run
    assert result["choice"] == "m1"
    short = [run for run in runs if run.seed == 2000 and run.arm == "adamw_long"]
    result = monitor_qualification(short)  # no SGD-long run at all
    assert result["arms"]["sgd_long"]["m1"]["passes"] is None
    assert "sgd_long/m1" in result["undetermined"] and result["choice"] == "redesign"


def test_trigger_checks_report_separate_evaluations_bracketing_and_the_band(tmp_path) -> None:
    separate = {5.0: 500, SQRT35: 600, SQRT40: 650, 7.0: 700, 8.0: 800}
    merged = {5.0: 500, SQRT35: 600, SQRT40: 700, 7.0: 700, 8.0: 700}
    a = load_trajectory(write_run(tmp_path, "a", arm="adamw_long", batch=5, lr=1e-5, seed=2000,
                                  reach=separate, monitor_version=2))  # fmt: skip
    b = load_trajectory(write_run(tmp_path, "b", arm="adamw_long", batch=5, lr=1e-5, seed=2001,
                                  reach=merged, monitor_version=2))  # fmt: skip
    checks_a, checks_b = trigger_checks(a), trigger_checks(b)
    for upper in ("8.0000", "7.0000"):
        assert checks_a[upper] == {"reached": True, "separate": True, "k_star_bracketed": True,
                                   "upper_in_band": True}  # fmt: skip
    assert checks_b["8.0000"]["separate"] is False  # √40 and 8 in one evaluation
    assert checks_b["7.0000"]["separate"] is True


def test_the_search_resumes_for_u7_when_its_u8_rate_fails_there(tmp_path) -> None:
    """AdamW-long's 1e-5 meets U = 8 at B = 5 but not U = 7 (√35 and 7 in one evaluation); SGD
    reaches only 7 at both batches, so no batch qualifies at U = 8 and the descent resumes."""
    merged7 = {5.0: 100, SQRT35: 300, SQRT40: 300, 7.0: 300, 8.0: 400}
    adamw = {5: {1e-5: merged7, 5e-6: FULL}, 20: {1e-5: FULL}}
    sgd = {b: {0.004: UP_TO_7, 0.003: UP_TO_7} for b in BATCHES}
    search(tmp_path, sgd=sgd, adamw=adamw)  # the descent stops at 1e-5: 5e-6 is not run
    result = chosen(tmp_path)
    manifest = result["search"]
    assert manifest["phase"] == "U=7" and not manifest["complete"]
    entry = manifest["searches"]["adamw_long/B=5"]
    assert entry["U=8"] == 1e-5 and entry["U=7"] is None
    assert entry["pending"] == {"rate": 5e-6, "seeds": SEEDS} and entry["not_required"] == []
    assert any("adamw_long/B=5 needs" in p for p in result["problems"])
    for seed in SEEDS:  # the resumed step
        write_run(tmp_path / "discovery", f"adamw_long-5-5e-06-{seed}", arm="adamw_long",
                  batch=5, lr=5e-6, seed=seed, reach=FULL, monitor_version=2)  # fmt: skip
    result = chosen(tmp_path)
    assert result["outcome"] == "selected", result.get("problems")
    assert result["U"] == 7.0 and result["adamw_long"]["lr"] in (5e-6, 1e-5)
    entry = result["search"]["searches"]["adamw_long/B=5"]
    assert entry["U=7"] == 5e-6 and entry["not_required"] == [3e-6, 1e-6]
    sgd_entry = result["search"]["searches"]["sgd_long/B=5"]
    assert list(sgd_entry["tested"]) == [0.004, 0.003] and sgd_entry["not_required"] == []


def test_a_rate_meeting_u8_ends_its_search_and_lower_rates_are_not_required(tmp_path) -> None:
    search(tmp_path)
    result = chosen(tmp_path)
    assert result["outcome"] == "selected" and result["U"] == 8.0
    entry = result["search"]["searches"]["adamw_long/B=5"]
    assert list(entry["tested"]) == [1e-5] and entry["not_required"] == [5e-6, 3e-6, 1e-6]
    assert result["search"]["phase"] == "U=8" and result["search"]["complete"]
