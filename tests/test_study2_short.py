"""Study 2's redirected short-regime stage, design A (analysis/study2_short.py): selection,
support, the estimator and the looks, on synthetic ``Trajectory`` objects and label rows.

The state machine is driven through its campaigns against an analytic world: K25 = c × (rate /
1e-5)^1.5 × a seed factor, per arm and batch, utility losses growing with K; its numerical checks
get verdict records of the numerical-check tool's schema. One test replays the archived records
of the main checkout (K, numerics and utility only), skipped where they are absent."""

import dataclasses
import json
import math
import re
import statistics
import subprocess
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from scipy import stats

from zero_field_refusal.analysis import study2_short
from zero_field_refusal.analysis.stage_a import (
    ADAMW_LONG_GRID,
    SHORT_GRID,
    Trajectory,
    arm_of,
    monitor_spec,
    schedule_spec,
)
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    ARMS,
    CANDIDATES,
    FA,
    FA_CONTROL_ARM,
    LOOK_SIZES,
    NUMERICAL_SEEDS,
    PREWARMED,
    VALIDATION_SEEDS,
    block_difference,
    block_support,
    bootstrap_t_interval,
    candidate_order,
    control_conformance,
    control_verdict,
    finite_bounds,
    harm,
    interpolate,
    numerical_cases,
    numerical_policy,
    numerical_verdict,
    repair_rates,
    secondary_contrasts,
    seed_role,
    select_rates,
    selection,
    sensitivity_looks,
    targeted_rates,
    usable_rate,
    utility_family,
    validation_verdict,
)
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.seeding import derive_seed
from zero_field_refusal.teacher import monitor_version

THETA0 = {"benchmark_accuracy": 0.60, "factual_accuracy": 0.80}
SEEDS = (2000, 2001, 2002)
K8 = math.sqrt(40)
W_MAX = 0.35
SPC = 90.0  # seconds per chunk
COMMIT = "c" * 40  # the code commit every synthetic run records


def traj(
    arm: str,
    batch: int,
    lr: float,
    seed: int,
    k: float | None,
    *,
    stop: str = "steps",
    steps: int = 25,
    mmlu: float = 1.0,
    factual: float = 1.0,
    seconds: float = 100.0,
    bound: float | None = None,
    scoring: tuple[float, ...] = (),
    chunks: int = 0,
    role: str | None = None,
    monitor_version: int = 2,
    k_harmful: float | None = None,
    name: str | None = None,
) -> Trajectory:
    """One run as ``stage_a.load_trajectory`` returns it: a step-25 evaluation with K25 ``k``
    (none when ``k`` is None), utility losses ``mmlu`` and ``factual`` in pp."""
    evaluations = []
    if k is not None:
        evaluations = [{
            "step": 25, "condition": "step-25", "levels": [], "K_monitor": k, "K_benign": k,
            "K_harmful": k if k_harmful is None else k_harmful,
            "benchmark_accuracy": THETA0["benchmark_accuracy"] - mmlu / 100,
            "factual_accuracy": THETA0["factual_accuracy"] - factual / 100,
        }]  # fmt: skip
    terminal = {"stop": {"reason": stop}, "steps": steps, "responses": steps * batch,
                "seconds": seconds + chunks * SPC,
                "stream_extension": {"seconds": chunks * SPC, "chunks": chunks}}  # fmt: skip
    name = name or f"{arm}-b{batch}-{lr:g}-s{seed}"
    return Trajectory(
        run_dir=Path("/runs") / name, run_id=name, arm=arm, batch_size=batch, lr=lr, seed=seed,
        policy="fixed", levels=(), terminal=terminal, evaluations=evaluations, monitor=[],
        grad_norms=[], seconds=seconds + chunks * SPC, chunk_size=1000, planned_steps=25,
        model="m@r", stream={}, harmful_prompts={}, status="completed", role=role,
        protocol_version=3,
        spec={"instability": {"contamination_k": bound}} if bound is not None else {},
        evaluate_at=(25,), monitor_version=monitor_version,
        diagnostics=tuple({"step": 5 * (i + 1), "K_scoring": v, "K_monitor": v,
                           "K_monitor_full": v} for i, v in enumerate(scoring)),
    )  # fmt: skip


def control(batch: int, lr: float, bound: float, contamination: float, **kw) -> Trajectory:
    return traj(FA_CONTROL_ARM, batch, lr, 2000, contamination, bound=bound, **kw)


def rec(k25: dict[int, float]) -> dict:
    """A usable rate's record, as ``usable_rate`` gives it."""
    return {"usable": True, "K25": k25, "median_K25": statistics.median(k25.values())}


# ------------------------------------------------------------------ the world

SCALE = {
    (ADAMW, 5): 5.0, (FA, 5): 3.4838, (PREWARMED, 5): 5.0,
    (ADAMW, 20): 3.5, (FA, 20): 2.5, (PREWARMED, 20): 3.5,
}  # fmt: skip
SPREAD = {2000: 0.97, 2001: 1.0, 2002: 1.03}


def kink(x: float) -> float:
    """A curve bent in log–log: slope 1.5 up to x = 1, 3 up to 1.2, then 0.5."""
    if x <= 1:
        return x**1.5
    if x <= 1.2:
        return x**3
    return 1.2**3 * (x / 1.2) ** 0.5


def world_k(arm, batch, lr, seed, scale=SCALE, spread=None, kinked=()) -> float:
    """The world's K25; a seed outside the discovery seeds' spread gets 0.98 + 0.01 × (seed % 5)."""
    factor = (spread or {}).get(arm, SPREAD).get(seed, 0.98 + 0.01 * (seed % 5))
    x = lr / 1e-5
    return scale[(arm, batch)] * (kink(x) if (arm, batch) in kinked else x**1.5) * factor


def made(arm, batch, lr, seed, scale=SCALE, spread=None, kinked=()) -> Trajectory:
    k = world_k(arm, batch, lr, seed, scale, spread, kinked)
    return traj(arm, batch, lr, seed, k, mmlu=0.3 * k, factual=0.1 * k,
                seconds=100.0 if batch == 5 else 150.0)  # fmt: skip


def grids(scale=SCALE, spread=None, kinked=()) -> list[Trajectory]:
    return [made(arm, b, lr, s, scale, spread, kinked) for arm in (ADAMW, FA) for b in (5, 20)
            for lr in SHORT_GRID for s in SEEDS]  # fmt: skip


# The synthetic commits' GPU code (``code_identity``): "a" × 40 changes analysis files only.
GPU = {COMMIT: "gpu-1", "a" * 40: "gpu-1", "g" * 40: "gpu-2"}


def reuse_at(head: str = COMMIT):
    """A ``code_identity.reuse_check`` over the synthetic commits, the checkout at ``head``."""

    def reuse(base: str, other: str) -> dict:
        other = head if other == "HEAD" else other
        if base not in GPU or other not in GPU:
            raise ValueError(f"unknown commit {base} or {other}")
        a, b = ({"commit": c, "files": 1, "sha256": GPU[c]} for c in (base, other))
        same = a["sha256"] == b["sha256"]
        return {"base": a, "head": b, "changed": [] if same else ["src/zero_field_refusal/x.py"],
                "reusable": same}  # fmt: skip

    return reuse


reusable = reuse_at()


def verdict_of(check: dict, rates=None, controls=None, *, gpu=None, **changes) -> dict:
    """The numerical-check tool's verdict record for a check request (a ``numerical-check``
    step's ``check``): every rate qualified on the seeds of its cases and every control accepted,
    but those ``rates`` ({(optimizer, lr): state}) and ``controls`` ({lr: state}) give, at the
    GPU code ``gpu`` (that of ``COMMIT`` by default)."""
    gpu = gpu or {"commit": COMMIT, "files": 1, "sha256": GPU[COMMIT]}
    rates, controls = rates or {}, controls or {}
    states = {f"{o}|{lr:g}": rates.get((o, lr), "qualified")
              for o, lrs in check["rates"].items() for lr in lrs}  # fmt: skip
    seeds: dict[str, list[int]] = {}
    for case in check["cases"]:
        seeds.setdefault(f"{case['optimizer']}|{case['lr']:g}", []).append(case["seed"])
    checked = {f"{lr:g}": controls.get(lr, "accepted")
               for lr in check["rates"].get("adam_frozen", [])}  # fmt: skip
    record = {"batch": check["batch"], "U": int(check["U"]), "rates": states,
              "seeds": seeds, "controls": checked,
              "qualified": all(v == "qualified" for v in states.values())
              and all(v == "accepted" for v in checked.values()),
              "reasons": {}, "cases": [], "provenance": {},
              "policy": {"gpu_code": gpu, "tool_commit": gpu["commit"],
                         **numerical_policy(check["U"]), "reference": "float64"}}  # fmt: skip
    return {**record, **changes}


def commits(runs, commit=COMMIT) -> dict:
    """``run_facts`` as far as the selection reads them: each run's commit."""
    return {run.run_id: {"commit": commit} for run in runs}


def drive(runs, *, scale=SCALE, spread=None, kinked=(), judge=verdict_of, verdicts=None,
          hook=None, until=None, limit=60):  # fmt: skip
    """Make each campaign the state machine asks for until it ends (or names ``until``), and
    give each check the verdict ``judge`` writes for it (every rate qualified, by default);
    ``hook`` may replace a run as made. Returns the campaigns, the last result and the
    verdicts."""
    seen, verdicts = [], {} if verdicts is None else verdicts
    for _ in range(limit):
        result = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=verdicts,
                           facts=commits(runs), reuse=reusable)  # fmt: skip
        step = result["next"]
        seen.append(step["campaign"])
        if step["campaign"] in ("done", "infeasible", "incomplete", until):
            return seen, result, verdicts
        if step["campaign"] == "numerical-check":
            verdicts[step["check"]["verdict"]] = judge(step["check"])
            continue
        for r in step["runs"]:
            run = made(r["arm"], r["batch"], r["lr"], r["seed"], scale, spread, kinked)
            runs.append(hook(run) if hook else run)
    raise AssertionError(f"no end after {seen}")


def campaign_runs(step) -> set[tuple]:
    return {(r["arm"], r["batch"], r["lr"], r["seed"]) for r in step["runs"]}


# ------------------------------------------------------------------ usability and targeting


def test_a_rate_is_usable_when_every_seed_finished_with_utility_in_bounds() -> None:
    cell = [traj(ADAMW, 5, 1e-5, s, k) for s, k in zip(SEEDS, (4.0, 5.0, 6.0), strict=True)]
    found = usable_rate(cell, THETA0)
    assert found["usable"] and found["median_K25"] == 5.0
    assert found["K25"] == {2000: 4.0, 2001: 5.0, 2002: 6.0}

    stopped = [*cell[:2], traj(ADAMW, 5, 1e-5, 2002, None, stop="k_above_max", steps=15)]
    assert usable_rate(stopped, THETA0)["reasons"] == [
        "seed 2002: stopped by k_above_max at step 15"
    ]
    assert usable_rate(cell[:2], THETA0)["reasons"] == ["seed 2002: 0 runs"]
    nan = [*cell[:2], traj(ADAMW, 5, 1e-5, 2002, math.nan)]
    assert not usable_rate(nan, THETA0)["usable"]
    lossy = [traj(ADAMW, 5, 1e-5, s, 5.0, mmlu=m) for s, m in zip(SEEDS, (4, 6, 6.5), strict=True)]
    found = usable_rate(lossy, THETA0)
    assert not found["usable"] and found["mmlu_loss_pp"] == pytest.approx(5.5)
    assert "utility out of bounds" in found["reasons"][0]
    inside = [traj(ADAMW, 5, 1e-5, s, 5.0, mmlu=4.9, factual=9.9) for s in SEEDS]
    assert usable_rate(inside, THETA0)["usable"]


def test_targeted_rates_interpolate_log_log_and_round_to_three_digits() -> None:
    # K ∝ r^1.5 through (1e-5, 4): log–log gives r = 1e-5 × (K / 4)^(2/3); interpolating r
    # linearly in K would give 1.14e-5, 1.32e-5 and 1.55e-5 instead.
    records = {1e-5: rec({2000: 4.0}), 2e-5: rec({2000: 4.0 * 2**1.5}),
               3e-5: {**rec({2000: 20.0}), "usable": False}}  # fmt: skip
    found = targeted_rates(records, 8.0)["targets"]
    assert [t["rate"] for t in found] == [1.16e-5, 1.36e-5, 1.59e-5]  # 1.1604, 1.3572, 1.5874
    assert found[1]["between"] == [1e-5, 2e-5]
    assert targeted_rates(records, 7.0)["targets"][2]["rate"] == 1.45e-5  # 1.4522


def test_targets_outside_the_usable_range_get_no_rate() -> None:
    records = {1e-5: rec({2000: 5.5}), 2e-5: rec({2000: 7.5})}
    found = targeted_rates(records, 8.0)["targets"]
    assert found[0] == {"target": 5.0, "rate": None, "reason": "below the usable range"}
    assert found[2]["reason"] == "above the usable range"
    assert found[1]["rate"] is not None
    one = targeted_rates({1e-5: rec({2000: 6.0})}, 8.0)["targets"]
    assert {t["reason"] for t in one} == {"fewer than two usable rates"}


def test_a_pair_whose_median_k_falls_is_reported_and_skipped() -> None:
    records = {1e-5: rec({2000: 4.0}), 1.5e-5: rec({2000: 3.9}), 2e-5: rec({2000: 8.0})}
    found = targeted_rates(records, 8.0)
    assert len(found["skipped"]) == 1 and "1e-05" in found["skipped"][0]
    # The target 5 is taken from the increasing pair (1.5e-5, 3.9) – (2e-5, 8).
    assert found["targets"][0]["between"] == [1.5e-5, 2e-5]
    inside = targeted_rates({1e-5: rec({1: 4.0}), 1.5e-5: rec({1: 3.0}), 2e-5: rec({1: 3.5})}, 8.0)
    assert inside["targets"][0]["reason"] == "above the usable range"
    gap = targeted_rates({1e-5: rec({1: 4.0}), 1.5e-5: rec({1: 9.0}), 2e-5: rec({1: 8.5}),
                          3e-5: rec({1: 9.5})}, 8.0)  # fmt: skip
    assert gap["targets"][2]["rate"] is not None  # 8 lies in (4, 9) of the first pair
    # 5 and K* = √35 lie in the usable range 4 – 6 but only in the skipped pair (6, 4).
    skipped = targeted_rates({1e-5: rec({1: 6.0}), 1.5e-5: rec({1: 4.0}), 2e-5: rec({1: 4.5})},
                             7.0)["targets"]  # fmt: skip
    assert [t.get("reason") for t in skipped] == ["inside a skipped pair", "inside a skipped pair",
                                                  "above the usable range"]  # fmt: skip


# ------------------------------------------------------------------ selection and repair

# Medians 5.0, 5.3, 6.4, 8.0; seed 2002's bracket of (r1, r3, r4) is 5.0 – 6.5, 0.262 wide.
CONSTRAINED = {
    1.0e-5: rec({2000: 5.0, 2001: 5.0, 2002: 5.0}),
    1.1e-5: rec({2000: 5.2, 2001: 5.3, 2002: 5.3}),
    1.2e-5: rec({2000: 6.4, 2001: 6.3, 2002: 6.5}),
    1.4e-5: rec({2000: 8.0, 2001: 8.0, 2002: 8.0}),
}


def test_the_selection_needs_its_width_constraint() -> None:
    loose = select_rates(CONSTRAINED, 8.0, 0.3)
    assert loose["rates"] == [1.0e-5, 1.2e-5, 1.4e-5]  # the unconstrained optimum
    tight = select_rates(CONSTRAINED, 8.0, 0.25)
    assert tight["feasible"] and tight["rates"] == [1.1e-5, 1.2e-5, 1.4e-5]
    assert tight["selected"]["per_seed"][2002]["width"] == pytest.approx(math.log(6.5 / 5.3))
    widths = {s: b["width"] for s, b in loose["selected"]["per_seed"].items()}
    assert widths[2002] == pytest.approx(math.log(6.5 / 5.0)) and widths[2002] > 0.25
    assert widths[2001] == pytest.approx(math.log(8.0 / 6.3))  # 6.3 < K*: the nearest pair
    assert select_rates(CONSTRAINED, 8.0, 0.25, exclude=[1.1e-5])["unmet"] == ["width"]


def test_ties_go_to_the_lexicographically_smaller_triple() -> None:
    same = {lr: rec({2000: k, 2001: k, 2002: k})
            for lr, k in ((1e-5, 5.0), (1.1e-5, K8), (1.2e-5, K8), (1.3e-5, 8.0))}  # fmt: skip
    assert select_rates(same, 8.0, W_MAX)["rates"] == [1e-5, 1.1e-5, 1.3e-5]


def test_an_infeasible_selection_names_its_unmet_constraints() -> None:
    assert select_rates({1e-5: rec({2000: 5.0})}, 8.0, W_MAX)["unmet"] == ["three usable rates"]
    far = {lr: rec({2000: k}) for lr, k in ((1e-5, 2.0), (2e-5, 3.0), (3e-5, 4.0))}
    assert select_rates(far, 8.0, W_MAX)["unmet"] == ["support", "width", "coverage"]
    # Each constraint met by some triple, none by all: coverage forces r_5 = 1e-5, which seed
    # 2002 already reads above K*; with 9e-6 instead every seed brackets K* within 0.35, but
    # its median 4.0 misses the target 5.
    joint = {
        9e-6: rec({2000: 3.5, 2001: 4.0, 2002: 6.0}),
        1e-5: rec({2000: 4.0, 2001: 5.0, 2002: 6.5}),
        1.1e-5: rec({2000: 5.0, 2001: 6.3, 2002: 7.0}),
        1.2e-5: rec({2000: 6.4, 2001: 6.4, 2002: 8.0}),
        1.4e-5: rec({2000: 7.0, 2001: 8.0, 2002: 9.0}),
    }
    found = select_rates(joint, 8.0, W_MAX)
    assert not found["feasible"] and found["unmet"] == ["jointly"]
    assert found["best_unconstrained"]["coverage"]


def test_repair_rates_from_the_best_unconstrained_triple_with_reasons() -> None:
    # Seed 2002 never below K*, seed 2000 never at or above it, seed 2001 too wide.
    records = {
        1e-5: rec({2000: 4.6, 2001: 4.6, 2002: 6.5}),  # median 4.6: target 5 covered
        1.2e-5: rec({2000: 5.5, 2001: 6.0, 2002: 7.0}),
        1.4e-5: rec({2000: 6.0, 2001: 8.0, 2002: 8.5}),
    }
    found = select_rates(records, 8.0, 0.2)
    triple = found["best_unconstrained"]
    assert triple["rates"] == [1e-5, 1.2e-5, 1.4e-5] and not found["feasible"]
    repair = repair_rates(triple, records, 8.0, 0.2)
    assert repair["rates"] == [8.26e-6, 1.69e-5, 1.3e-5]  # 1e-5/1.21, 1.4e-5×1.21, √(1.2×1.4)
    assert "seeds [2002]" in repair["reasons"][0] and "seeds [2000]" in repair["reasons"][1]
    assert "seed 2001" in repair["reasons"][2] and not repair["unrepairable"]
    assert repair_rates(None, records, 8.0, 0.2)["reasons"] == ["no triple to repair from"]


def test_a_coverage_miss_is_re_targeted_over_every_usable_rate() -> None:
    # Medians 4.0, 6.3, 8.0: the triple misses 5 by ln(4/5) = −0.223 while K* is bracketed.
    records = {
        1e-5: rec({2000: 4.0, 2001: 4.0, 2002: 4.0}),
        1.2e-5: rec({2000: 6.3, 2001: 6.3, 2002: 6.3}),
        1.4e-5: rec({2000: 8.0, 2001: 8.0, 2002: 8.0}),
    }
    found = select_rates(records, 8.0, W_MAX)
    assert found["unmet"] == ["coverage"]
    repair = repair_rates(found["best_unconstrained"], records, 8.0, W_MAX)
    # ln r = ln 1e-5 + ln(5/4)/ln(6.3/4) × ln 1.2: r = 1.0937e-5.
    assert repair["rates"] == [1.09e-5] and not repair["unrepairable"]
    assert "target 5 missed by -0.223" in repair["reasons"][0]
    # The same rate already run (unusable): no repair, the candidate fails.
    run = {**records, 1.09e-5: {**rec({2000: 4.4}), "usable": False}}
    repair = repair_rates(found["best_unconstrained"], run, 8.0, W_MAX)
    assert repair["rates"] == [] and "1.09e-05, already run" in repair["unrepairable"][0]
    # A target above the usable range takes the outward extension r_U × 1.21.
    low = {1e-5: rec({2000: 5.0}), 1.1e-5: rec({2000: 6.4}), 1.2e-5: rec({2000: 7.0})}
    repair = repair_rates(select_rates(low, 8.0, W_MAX)["best_unconstrained"], low, 8.0, W_MAX)
    assert repair["rates"] == [1.45e-5] and "above the usable range" in repair["reasons"][0]


def test_the_midpoint_repairs_the_widest_bracket() -> None:
    records = {
        1e-5: rec({2000: 4.0, 2001: 4.9, 2002: 4.9}),
        1.2e-5: rec({2000: 6.4, 2001: 6.2, 2002: 6.2}),
        1.5e-5: rec({2000: 8.0, 2001: 7.9, 2002: 8.1}),
    }
    triple = select_rates(records, 8.0, 0.2)["best_unconstrained"]
    # Every bracket is wider than 0.2: seed 2000's r_5 – r_K* (4.0 – 6.4) is the widest, the
    # others are r_K* – r_U (6.2 – 7.9 and 6.2 – 8.1).
    assert triple["per_seed"][2000]["width"] == pytest.approx(math.log(1.6))
    repair = repair_rates(triple, records, 8.0, 0.2)
    assert repair["rates"] == [1.1e-5]  # √(1e-5 × 1.2e-5) = 1.095e-5
    assert "seed 2000" in repair["reasons"][0] and len(repair["reasons"]) == 1


# ------------------------------------------------------------------ candidates and costs


def test_candidate_order_is_b20_first() -> None:
    runs = grids()  # 100 s per run at B = 5, 150 s at B = 20
    order = candidate_order(runs, THETA0, SPC)
    assert [(e["U"], e["batch"]) for e in order] == [(8.0, 20), (7.0, 20), (8.0, 5), (7.0, 5)]
    assert [(e["U"], e["batch"]) for e in order] == list(CANDIDATES)
    # The costs stay for the report: B = 5 is cheaper and still comes after B = 20. Six runs, one
    # chunk of training responses, v₀'s 50 × B preconditioner responses (1000 at B = 20).
    assert order[0]["seconds"] == pytest.approx(6 * 150.0 + SPC + SPC * 1000 / 1000)
    assert order[0]["preconditioner_seconds"] == pytest.approx(SPC)
    assert order[2]["seconds"] == pytest.approx(6 * 100.0 + SPC + SPC * 250 / 1000)
    assert order[2]["seconds"] < order[0]["seconds"]
    assert order[2]["nearest"][ADAMW] == [1e-5, 1e-5, 1.5e-5]  # medians 5.0, 5.0, 9.19


def test_costs_are_reported_and_never_reorder_the_candidates() -> None:
    cheap5 = [traj(r.arm, r.batch_size, r.lr, r.seed, r.evaluations[0]["K_benign"],
                   seconds=10.0 if r.batch_size == 5 else 500.0) for r in grids()]  # fmt: skip
    assert [e["batch"] for e in candidate_order(cheap5, THETA0, SPC)] == [20, 20, 5, 5]
    without = [r for r in grids() if (r.arm, r.batch_size) != (FA, 5)]
    order = candidate_order(without, THETA0, SPC)
    assert [e["batch"] for e in order] == [20, 20, 5, 5]
    assert order[2]["seconds"] is None and order[2]["reasons"] == [f"{FA}: no usable grid rate "
                                                                   "at B=5"]  # fmt: skip


# ------------------------------------------------------------------ the state machine


def test_the_state_machine_discovers_qualifies_validates_and_freezes() -> None:
    runs = grids()
    seen, result, verdicts = drive(runs)
    assert seen == ["discovery", "numerical", "numerical-check", "validation", "done"]
    done = result["next"]
    assert (done["batch"], done["U"], done["three_arm"]) == (20, 8.0, True)
    assert done["rates"] == {ADAMW: [1.27e-5, 1.48e-5, 1.74e-5], FA: [1.59e-5, 1.86e-5, 2.17e-5],
                             PREWARMED: [1.27e-5, 1.48e-5, 1.74e-5]}  # fmt: skip
    assert not done["prewarmed_dropped"] and done["validation"]["passes"]
    assert done["validation"]["arms"] == list(ARMS)  # the three-arm validation
    assert [e["verdict"] for e in done["numerical"]] == ["numerical/b20-u8/verdict.json"]
    assert list(verdicts) == ["numerical/b20-u8/verdict.json"]
    rounds = result["rounds"]
    assert rounds[f"{PREWARMED}/B=20/U=8"]["first"]["proposal"]["joint"]
    assert result["problems"] == [] and len(result["attempts"]) == 1
    # The numerical runs: 15 cases, K-only, never a block; the validation's: 12 three-arm blocks.
    numerical = [r for r in runs if r.seed in NUMERICAL_SEEDS]
    assert len(numerical) == 15 and {seed_role(r.seed) for r in numerical} == {"numerical"}
    assert len([r for r in runs if r.seed in VALIDATION_SEEDS]) == 3 * 3 * 12


def test_the_first_campaign_is_the_joint_discovery_at_b20() -> None:
    result = selection(grids(), THETA0, W_MAX, seconds_per_chunk=SPC, numerical={}, facts={})
    step = result["next"]
    assert (step["campaign"], step["batch"], step["U"]) == ("discovery", 20, 8.0)
    asked = {(r["arm"], r["lr"]) for r in step["runs"]}
    primary = {(ADAMW, 1.27e-5), (ADAMW, 1.48e-5), (ADAMW, 1.74e-5), (FA, 1.59e-5),
               (FA, 1.86e-5), (FA, 2.17e-5)}  # fmt: skip
    # The pre-warmed arm's first round: the union of the rates of the targeted assignments.
    assert asked == primary | {(PREWARMED, lr) for _, lr in primary}
    assert len(step["runs"]) == 3 * len(asked)
    assert {r["seed"] for r in step["runs"]} == set(SEEDS)


def test_failing_numerical_rate_resolves_once() -> None:
    # The first check finds AdamW's K* rate failing: it leaves B = 20, AdamW is re-solved once
    # from the same runs (no round), and only the replacement's case is run: on 2200, since the
    # U-rate stays.
    runs = grids()
    asked = []

    def judge(check):
        asked.append(check)
        return verdict_of(check, {("adamw", 1.48e-5): "failing"})

    seen, result, verdicts = drive(runs, judge=judge)
    assert seen == ["discovery", "numerical", "numerical-check", "numerical", "numerical-check",
                    "validation", "done"]  # fmt: skip
    assert list(verdicts) == ["numerical/b20-u8/verdict.json", "numerical/b20-u8-2/verdict.json"]
    # The second check writes into its own directory, reusing the first's case records.
    assert [(c["out"], c["records_from"]) for c in asked] == [
        ("numerical/b20-u8", []), ("numerical/b20-u8-2", ["numerical/b20-u8"])]  # fmt: skip
    assert asked[1]["selection"]["rates"]["adamw"] == [1.27e-5, 1.5e-5, 1.74e-5]
    attempt = result["attempts"][0]
    first, second = attempt["numerical"]
    assert first["unusable"] == [
        {
            "arm": ADAMW,
            "lr": 1.48e-5,
            "state": "failing",
            "why": "failing: a pair exceeding the bound",
            "reasons": [],
        }
    ]
    assert second["check"] == "b20-u8-2" and second["unusable"] == []
    assert attempt["arms"][ADAMW]["selections"][-1]["excluded"] == [1.48e-5]
    assert result["next"]["rates"][ADAMW] == [1.27e-5, 1.5e-5, 1.74e-5]
    replacement = [r for r in runs if r.seed in NUMERICAL_SEEDS and r.lr == 1.5e-5]
    assert [(r.arm, r.seed) for r in replacement] == [(ADAMW, 2200)]
    # A second unusable rate at the candidate, here not qualified (technical), fails it; the
    # next candidate (7, 20) starts without both rates.
    runs = grids()
    judge = lambda check: verdict_of(check, {("adamw", 1.48e-5): "failing",  # noqa: E731
                                             ("adamw", 1.5e-5): "not_qualified"})  # fmt: skip
    seen, result, _ = drive(runs, judge=judge)
    failed, at7 = result["attempts"][:2]
    assert (failed["outcome"], failed["failed_at"], failed["failed_arms"]) == (
        "failed", "numerical", [ADAMW])  # fmt: skip
    assert failed["reasons"] == ["unusable rates after the re-solve: ['adamw_25 1.5e-05: not "
                                 "qualified (technical)']"]  # fmt: skip
    assert (at7["batch"], at7["U"]) == (20, 7.0)
    assert sorted(at7["excluded"][ADAMW]) == [1.48e-5, 1.5e-5]
    assert "not qualified (technical)" in at7["excluded"][ADAMW][1.5e-5]
    assert "failing" in at7["excluded"][ADAMW][1.48e-5]  # technical is never instability
    assert seen[:6] == ["discovery", "numerical", "numerical-check", "numerical",
                        "numerical-check", "discovery"]  # fmt: skip
    assert (result["next"]["campaign"], result["next"]["U"]) == ("done", 7.0)


def test_a_replacement_u_rate_gets_its_cases_on_2201_and_2202() -> None:
    scale = {**SCALE, (PREWARMED, 20): 3.3}  # the pre-warmed U-rate has two candidates near 8
    runs = grids(scale)
    judge = lambda check: verdict_of(check, {("adamw_prewarmed", 1.86e-5): "failing"})  # noqa: E731
    seen, result, verdicts = drive(runs, scale=scale, judge=judge)
    assert seen[:5] == ["discovery", "numerical", "numerical-check", "numerical",
                        "numerical-check"]  # fmt: skip
    attempt = result["attempts"][0]
    before, after = attempt["arms"][PREWARMED]["selections"][-2:]  # the re-solve
    assert (before["rates"], after["rates"]) == ([1.27e-5, 1.59e-5, 1.86e-5],
                                                 [1.27e-5, 1.59e-5, 1.74e-5])  # fmt: skip
    assert after["excluded"] == [1.86e-5]
    assert result["next"]["rates"][PREWARMED] == [1.27e-5, 1.59e-5, 1.74e-5]
    replacement = sorted((r.arm, r.lr, r.seed) for r in runs
                         if r.seed in NUMERICAL_SEEDS and r.lr == 1.74e-5)  # fmt: skip
    # AdamW's U-rate 1.74e-5 had its three cases already; the pre-warmed arm's new U-rate gets
    # its own three.
    assert [(arm, seed) for arm, _, seed in replacement] == [
        (ADAMW, 2200), (ADAMW, 2201), (ADAMW, 2202),
        (PREWARMED, 2200), (PREWARMED, 2201), (PREWARMED, 2202)]  # fmt: skip
    second = verdicts["numerical/b20-u8-2/verdict.json"]
    assert second["seeds"]["adamw_prewarmed|1.74e-05"] == [2200, 2201, 2202]


def test_an_unusable_fa_control_is_like_a_failing_rate() -> None:
    runs = grids()
    seen, result, _ = drive(
        runs, judge=lambda check: verdict_of(check, controls={2.17e-5: "rejected"})
    )
    unusable = result["attempts"][0]["numerical"][0]["unusable"]
    assert unusable == [
        {
            "arm": FA,
            "lr": 2.17e-5,
            "state": "control rejected",
            "why": "its same-path control rejected",
            "reasons": [],
        }
    ]
    # No FA triple at U = 8 without 2.17e-5: the candidate fails, and 2.17e-5 stays out at U = 7.
    first, at7 = result["attempts"][:2]
    assert first["failed_at"] == "numerical" and first["failed_arms"] == [FA]
    assert 2.17e-5 in at7["excluded"][FA]


def test_old_controls_are_reported_not_used() -> None:
    # Amendment 3's history at B = 5: its targeted round at (8, 5) and three FA controls, two of
    # them rejected. B = 20 fails here (AdamW's K stays far below the targets).
    scale = {**SCALE, (ADAMW, 20): 0.2}
    runs = grids(scale)
    history = {ADAMW: [1.17e-5, 1.37e-5], FA: [1.27e-5, 1.49e-5, 1.74e-5]}
    runs += [made(arm, 5, lr, s, scale) for arm, lrs in history.items() for lr in lrs
             for s in SEEDS]  # fmt: skip
    bound = 0.01 * K8
    runs += [control(5, lr, bound, (2.0 if lr != 1.27e-5 else 0.1) * bound)
             for lr in history[FA]]  # fmt: skip
    first = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical={}, facts={})
    # At (8, 20) the B = 5 runs are not reached yet: no problem, no decision on them.
    assert first["next"]["campaign"] == "discovery" and first["next"]["batch"] == 20
    assert first["problems"] == [] and len(first["not_reached"]) == 15
    reported = first["amendment3_controls"]
    assert [(c["lr"], c["accepted"], c["used"]) for c in reported] == [
        (1.27e-5, True, False), (1.49e-5, False, False), (1.74e-5, False, False)]  # fmt: skip
    assert all(c["conformance"] == [] and c["bound"] == pytest.approx(bound) for c in reported)
    seen, result, _ = drive(runs, scale=scale, limit=14)
    at5 = next(a for a in result["attempts"] if (a["batch"], a["U"]) == (5, 8.0))
    assert at5["excluded"][FA] == {}  # the rejected controls exclude nothing
    # The replay prescribes amendment 3's targeted round again from the same grid: run already.
    targeted = result["rounds"][f"{FA}/B=5/U=8"]["targeted"]
    assert [t["rate"] for t in targeted["proposal"]["targets"]] == history[FA]
    assert {(r.arm, r.lr) for r in runs if r.batch_size == 5 and r.seed == 2000} >= {
        (arm, lr) for arm, lrs in history.items() for lr in lrs}  # fmt: skip
    request = at5["numerical"][0]["request"]
    assert request[FA] == [1.27e-5, 1.49e-5, 1.74e-5]  # qualified by amendment 4 alone
    assert result["next"]["campaign"] == "done" and result["next"]["batch"] == 5


def test_an_old_control_off_its_prescription_is_reported_not_used() -> None:
    runs = grids()
    _, done, verdicts = drive(runs)
    twin = next(r for r in runs if (r.arm, r.batch_size, r.lr, r.seed) == (FA, 20, 1.59e-5, 2000))
    moved = dataclasses.replace(control(20, 1.59e-5, 0.01 * K8, 0.0), model="other@rev")
    result = selection([*runs, moved], THETA0, W_MAX, seconds_per_chunk=SPC, numerical=verdicts,
                       facts=commits(runs), reuse=reusable)  # fmt: skip
    (reported,) = result["amendment3_controls"]
    assert "is not on the model and stream" in reported["conformance"][0]
    assert result["problems"] == [] and result["next"] == done["next"]  # nothing changes
    assert control_conformance(control(20, 1.59e-5, 0.01 * K8, 0.0), twin) == []
    assert "no FA run" in control_conformance(moved, None)[0]
    sparse = dataclasses.replace(moved, model=twin.model, evaluate_at=(20,))
    assert "it needs 25 at [25]" in control_conformance(sparse, twin)[0]


def test_prewarmed_never_feasible_falls_back() -> None:
    scale = {**SCALE, (PREWARMED, 20): 0.3, (PREWARMED, 5): 0.3}
    runs = grids(scale)
    seen, result, verdicts = drive(runs, scale=scale)
    attempts = result["attempts"]
    # Every candidate fails for the three arms on the pre-warmed arm, after its own rounds.
    assert [(a["U"], a["batch"], a["three_arm"], a["failed_arms"], a["failed_at"])
            for a in attempts[:4]] == [(u, b, True, [PREWARMED], "discovery")
                                       for u, b in CANDIDATES]  # fmt: skip
    assert [r["kind"] for r in attempts[0]["arms"][PREWARMED]["rounds"]] == [
        "first", "targeted", "repair"]  # fmt: skip
    # Then the first candidate eligible for the primary arms alone is taken.
    fallback = attempts[4]
    assert (fallback["U"], fallback["batch"], fallback["three_arm"]) == (8.0, 20, False)
    assert set(fallback["arms"]) == {ADAMW, FA} and fallback["outcome"] == "eligible"
    done = result["next"]
    assert done["campaign"] == "done" and (done["batch"], done["U"]) == (20, 8.0)
    assert done["prewarmed_dropped"] and not done["three_arm"] and done["rates"][PREWARMED] is None
    assert done["validation"]["arms"] == [ADAMW, FA]  # amendment 3's primary-arm validation
    assert seen[-4:] == ["numerical", "numerical-check", "validation", "done"]
    assert list(verdicts) == ["numerical/b20-u8/verdict.json"]
    assert {(r.arm, r.seed) for r in runs if r.seed in NUMERICAL_SEEDS} <= {
        (arm, s) for arm in (ADAMW, FA) for s in NUMERICAL_SEEDS}  # fmt: skip
    assert not [r for r in runs if r.arm == PREWARMED and r.seed in VALIDATION_SEEDS]


def test_primary_only_fallback_end_to_end() -> None:
    # The pre-warmed arm fails everywhere: at (8, 20) in the combined numerical verdict (each of
    # its rates failing or not qualified, no triple left), at (7, 20) in selection (its rates
    # unusable at B = 20, its rounds no help), at B = 5 in selection. The primary arms qualify at
    # (8, 20).
    def judge(check):
        pre = check["rates"].get("adamw_prewarmed", [])
        return verdict_of(check, {("adamw_prewarmed", lr): ("failing" if i % 2 else
                                                            "not_qualified")
                                  for i, lr in enumerate(pre)})  # fmt: skip

    scale = {**SCALE, (PREWARMED, 5): 0.3}
    runs = grids(scale)
    seen, result, verdicts = drive(runs, scale=scale, judge=judge, until="validation")
    attempts = result["attempts"]
    assert [(a["U"], a["batch"], a["failed_arms"], a["failed_at"]) for a in attempts[:4]] == [
        (8.0, 20, [PREWARMED], "numerical"), (7.0, 20, [PREWARMED], "discovery"),
        (8.0, 5, [PREWARMED], "discovery"), (7.0, 5, [PREWARMED], "discovery")]  # fmt: skip
    assert "without a feasible triple once the unusable rates" in attempts[0]["reasons"][0]
    assert all(u["arm"] == PREWARMED for u in attempts[0]["numerical"][0]["unusable"])
    # The fallback reads the same combined verdict for the primary arms and the FA controls
    # only: its pre-warmed entries do not block it, and no new check is asked for.
    fallback = attempts[4]
    assert (fallback["three_arm"], fallback["numerical"][0]["check"]) == (False, "b20-u8")
    assert fallback["numerical"][0]["state"] == "complete"
    assert fallback["numerical"][0]["unusable"] == []
    step = result["next"]
    assert (step["campaign"], step["batch"], step["U"], step["three_arm"]) == (
        "validation", 20, 8.0, False)  # fmt: skip
    assert step["prewarmed_dropped"] and set(step["rates"]) == {ADAMW, FA}
    assert {r["arm"] for r in step["runs"]} == {ADAMW, FA} and len(step["runs"]) == 2 * 3 * 12
    seen, result, verdicts = drive(runs, scale=scale, judge=judge, verdicts=verdicts)
    assert seen == ["validation", "done"] and list(verdicts) == ["numerical/b20-u8/verdict.json"]
    done = result["next"]
    assert done["prewarmed_dropped"] is True and done["three_arm"] is False
    assert done["rates"] == {ADAMW: [1.27e-5, 1.48e-5, 1.74e-5], FA: [1.59e-5, 1.86e-5, 2.17e-5],
                             PREWARMED: None}  # fmt: skip
    assert done["validation"]["passes"] and done["validation"]["arms"] == [ADAMW, FA]
    # Nothing of the pre-warmed arm was called for after it was dropped.
    assert not [r for r in runs if r.arm == PREWARMED and r.seed in VALIDATION_SEEDS]
    assert result["problems"] == []


def test_validation_failure_moves_on() -> None:
    def lossy(run):  # AdamW's U = 8 rate loses 6 pp MMLU on the validation seeds
        if run.seed in VALIDATION_SEEDS and (run.arm, run.lr) == (ADAMW, 1.74e-5):
            k = run.evaluations[0]["K_benign"]
            return traj(run.arm, run.batch_size, run.lr, run.seed, k, mmlu=6.0, factual=0.1 * k)
        return run

    runs = grids()
    seen, result, _ = drive(runs, hook=lossy)
    assert seen == ["discovery", "numerical", "numerical-check", "validation", "numerical",
                    "numerical-check", "validation", "done"]  # fmt: skip
    failed, at7 = result["attempts"]
    assert (failed["failed_at"], failed["reasons"]) == (
        "validation",
        ["the validation does not pass: the utility family does not pass"],
    )
    member = next(e for e in failed["validation"]["utility"]["family"]
                  if (e["arm"], e["rate"]) == (ADAMW, 1.74e-5))  # fmt: skip
    assert not member["mmlu_passes"]
    assert (at7["batch"], at7["U"], at7["outcome"]) == (20, 7.0, "eligible")
    # U = 7 reuses the validation seeds: only the runs at its new rates are made.
    new = {ADAMW: 1.5e-5, FA: 2e-5, PREWARMED: 1.59e-5}
    assert at7["rates"] == {ADAMW: [1.27e-5, 1.48e-5, 1.5e-5], FA: [1.59e-5, 1.86e-5, 2e-5],
                            PREWARMED: [1.27e-5, 1.48e-5, 1.59e-5]}  # fmt: skip
    validation = {(r.arm, r.lr) for r in runs if r.seed in VALIDATION_SEEDS}
    assert len([r for r in runs if r.seed in VALIDATION_SEEDS]) == 12 * (9 + 3)
    assert set(new.items()) <= validation
    assert result["next"]["validation"]["passes"]


def test_a_three_arm_validation_failure_falls_back_to_the_primary_validation() -> None:
    def unsupported(run):  # the pre-warmed arm never brackets K* on the validation seeds
        if run.seed in VALIDATION_SEEDS and run.arm == PREWARMED:
            return traj(run.arm, run.batch_size, run.lr, run.seed, 1.0, mmlu=0.3, factual=0.1)
        return run

    runs = grids()
    seen, result, _ = drive(runs, hook=unsupported)
    attempts = result["attempts"]
    assert [a["failed_at"] for a in attempts[:4]] == ["validation"] * 4
    assert "0 of 12 blocks jointly supported for the three arms" in attempts[0]["reasons"][0]
    # The fallback's validation reads the primary arms of the same runs: nothing new to run.
    assert seen[-1] == "done" and seen[-2] == "validation" and len(attempts) == 5
    done = result["next"]
    assert (done["batch"], done["U"], done["three_arm"]) == (20, 8.0, False)
    assert done["validation"]["arms"] == [ADAMW, FA] and done["validation"]["passes"]


def test_no_eligible_candidate_ends_infeasible_with_its_reasons() -> None:
    wide = {ADAMW: {2000: 0.8, 2001: 1.0, 2002: 1.3}}  # wider than the targets' spacing
    runs = grids(spread=wide)
    seen, result, _ = drive(runs, spread=wide)
    # Every candidate, U = 7 included, gets its own targeted and repair rounds; the primary arms
    # failed at discovery everywhere, so the fallback has nothing to visit.
    assert seen == ["discovery", "repair"] * 4 + ["infeasible"]
    assert [(a["U"], a["batch"]) for a in result["attempts"]] == list(CANDIDATES)
    assert result["next"]["reasons"] == [
        f"U={u:g}, B={b}, three arms: ['{ADAMW}'] without a feasible triple after the rounds"
        for u, b in CANDIDATES]  # fmt: skip
    rounds = result["rounds"]
    repair8 = rounds[f"{ADAMW}/B=20/U=8"]["repair"]
    assert repair8["new_rates"] == [1.05e-5]
    # U = 7's targets differ: its targeted round asks for new rates; 1.27e-5 serves 5 at both U.
    targeted7 = rounds[f"{ADAMW}/B=20/U=7"]["targeted"]
    assert [t["rate"] for t in targeted7["proposal"]["targets"]] == [1.27e-5, 1.42e-5, 1.59e-5]
    assert targeted7["new_rates"] == [1.42e-5, 1.59e-5]
    # Every run made at B = 20 enters U = 7's selections.
    usable7 = result["attempts"][1]["arms"][ADAMW]["selections"][0]["usable"]
    assert {1.05e-5, 1.48e-5, 1.74e-5} <= set(usable7)
    assert f"{FA}/B=20/U=7" not in rounds  # FA is feasible at U = 7 from what B = 20 ran


def test_a_coverage_miss_after_the_targeted_round_is_repaired_in_the_state_machine() -> None:
    # AdamW's K bends between the grid's 1e-5 and 1.5e-5: the targeted K* and U rates overshoot.
    bent = {(ADAMW, 20)}
    runs = grids(kinked=bent)
    seen, result, _ = drive(runs, kinked=bent)
    assert seen == ["discovery", "repair", "targeted", "numerical", "numerical-check",
                    "validation", "done"]  # fmt: skip
    first = result["attempts"][0]["arms"][ADAMW]["selections"]
    assert "coverage" in first[1]["unmet"]  # after the targeted round
    repair = result["rounds"][f"{ADAMW}/B=20/U=8"]["repair"]["proposal"]
    assert repair["rates"] == [1.03e-5, 1.15e-5]
    assert sum("re-targeted" in reason for reason in repair["reasons"]) >= 1
    assert result["next"]["rates"][ADAMW] == [1.15e-5, 1.25e-5, 2.1e-5]
    # The joint round gave the pre-warmed arm no triple; its own targeted round followed.
    pre = result["rounds"][f"{PREWARMED}/B=20/U=8"]
    assert pre["first"]["proposal"]["joint"] and pre["targeted"]["new_rates"]


def test_unprescribed_runs_are_reported_and_duplicates_leave_no_conclusion() -> None:
    runs = [*grids(), made(ADAMW, 20, 1.23e-5, 2000)]
    result = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical={}, facts={})
    assert result["problems"] == ["unexpected run ('adamw_25', 20, 1.23e-05, 2000)"]
    assert 1.23e-5 not in result["attempts"][0]["arms"][ADAMW]["selections"][0]["usable"]
    later = [*grids(), made(ADAMW, 5, 1.23e-5, 2000), made(FA, 20, 1.5e-5, 2101)]
    result = selection(later, THETA0, W_MAX, seconds_per_chunk=SPC, numerical={}, facts={})
    # B = 5 is not reached yet: deferred; a validation run nothing called for is a problem.
    assert result["not_reached"] == ["run ('adamw_25', 5, 1.23e-05, 2000) at a batch not "
                                     "reached yet"]  # fmt: skip
    assert result["problems"] == ["unexpected run ('frozen_adam_25', 20, 1.5e-05, 2101)"]
    twice = [*grids(), traj(ADAMW, 5, 1e-5, 2000, 5.0, name="again")]
    result = selection(twice, THETA0, W_MAX, seconds_per_chunk=SPC)
    assert result["next"] == {"campaign": "incomplete"} and "two runs" in result["problems"][0]


def stage_a_reference() -> list[Trajectory]:
    """Stage A's version-1 discovery set: the 25-step grids and AdamW-long, which made the
    chunks (``stage_a.stream_time_reference``)."""
    short = [traj(r.arm, r.batch_size, r.lr, r.seed, k, mmlu=0.3 * k, factual=0.1 * k,
                  seconds=r.training_seconds, role="discovery", monitor_version=1)
             for r in grids() if (k := r.evaluations[0]["K_benign"])]  # fmt: skip
    long = [traj("adamw_long", b, lr, s, 5.0, chunks=2, role="discovery", monitor_version=1)
            for b in (5, 20) for lr in ADAMW_LONG_GRID for s in SEEDS]  # fmt: skip
    return short + long


def test_the_stream_time_is_stage_as_reference_with_its_provenance() -> None:
    runs = stage_a_reference()
    result = selection(runs, THETA0, W_MAX)
    assert result["seconds_per_chunk"] == pytest.approx(SPC)
    assert result["stream_time"]["runs"] == 96 and result["stream_time"]["chunks"] == 48
    assert result["next"]["campaign"] == "discovery"
    # A run missing from the set leaves the reference undefined: no conclusion.
    partial = selection(runs[:-1], THETA0, W_MAX)
    assert partial["next"] == {"campaign": "incomplete"}
    assert "version-1 run missing" in partial["problems"][0]


def test_the_selection_never_reads_k_harmful() -> None:
    runs = grids()
    other = [traj(r.arm, r.batch_size, r.lr, r.seed, r.evaluations[0]["K_benign"],
                  mmlu=0.3 * r.evaluations[0]["K_benign"],
                  factual=0.1 * r.evaluations[0]["K_benign"], seconds=r.training_seconds,
                  k_harmful=math.nan) for r in runs]  # fmt: skip
    first = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC)
    assert selection(other, THETA0, W_MAX, seconds_per_chunk=SPC) == first


def test_a_control_verdict_reads_every_scoring_read() -> None:
    bound = 0.01 * K8
    assert control_verdict(control(5, 1e-5, bound, 0.5 * bound), K8)["accepted"]
    above = control(5, 1e-5, bound, 0.5 * bound, scoring=(0.2 * bound, 1.1 * bound))
    assert not control_verdict(above, K8)["accepted"]  # a diagnostic above the bound
    nan = control(5, 1e-5, bound, 0.5 * bound, scoring=(math.nan,))
    assert not control_verdict(nan, K8)["accepted"]
    stopped = control(5, 1e-5, bound, None, stop="contamination", steps=20)
    assert control_verdict(stopped, K8)["reasons"][0] == "stopped by contamination at step 20"


# ------------------------------------------------------------------ the numerical verdicts

REQUEST = {ADAMW: [1.27e-5, 1.48e-5, 1.74e-5], FA: [1.59e-5, 1.86e-5, 2.17e-5],
           PREWARMED: [1.27e-5, 1.48e-5, 1.74e-5]}  # fmt: skip


def request_check(rates=REQUEST, upper=8.0) -> dict:
    """A check request as the state machine writes it (``step["check"]``)."""
    slot = {"name": "b20-u8", "path": "numerical/b20-u8/verdict.json", "earlier": []}
    return study2_short._check_request(slot, 20, upper, rates, COMMIT)


def read(record, rates=REQUEST, *, commit=COMMIT, reuse=reusable) -> dict:
    return numerical_verdict(record, 20, 8.0, rates, commit, reuse)


def test_numerical_cases_are_the_frozen_scope() -> None:
    cases = numerical_cases(REQUEST)
    assert len(cases) == 15
    assert {(arm, lr) for arm, lr, seed in cases if seed == 2200} == {
        (arm, lr) for arm, lrs in REQUEST.items() for lr in lrs}  # fmt: skip
    assert sorted((arm, lr, seed) for arm, lr, seed in cases if seed != 2200) == sorted(
        (arm, max(lrs), seed) for arm, lrs in REQUEST.items() for seed in (2201, 2202)
    )
    assert len(numerical_cases({ADAMW: REQUEST[ADAMW], FA: REQUEST[FA]})) == 10
    check = request_check()
    assert check["runs_commit"] == COMMIT
    assert check["policy"] == {"norm": 5e-05, "survival_share": 0.5, "retry_scale": 10,
                               "steps": [1, 5, 10, 15, 20, 25], "bound": 0.01 * K8}  # fmt: skip


def test_numerical_verdict_reads_the_tools_record() -> None:
    check = request_check()
    record = verdict_of(
        check,
        {("adamw", 1.48e-5): "failing", ("adamw_prewarmed", 1.27e-5): "not_qualified"},
        {1.86e-5: "not_qualified"},
        reasons={"adamw_prewarmed|1.27e-05": ["case s2200 incomplete twice"],
                 "adam_frozen|1.86e-05": ["control: incomplete after its rerun"]},
    )  # fmt: skip
    found = read(record)
    assert found["state"] == "complete"
    assert [(u["arm"], u["lr"], u["state"]) for u in found["unusable"]] == [
        (ADAMW, 1.48e-5, "failing"), (PREWARMED, 1.27e-5, "not_qualified"),
        (FA, 1.86e-5, "control not_qualified")]  # fmt: skip
    assert found["unusable"][1]["reasons"] == ["case s2200 incomplete twice"]
    assert found["unusable"][1]["why"] == "not qualified (technical)"
    # A control's reasons sit under its FA rate's key, prefixed "control: " (Task 1).
    assert found["unusable"][2]["reasons"] == ["control: incomplete after its rerun"]
    # The record keeps the GPU-code check that let the verdict hold at the checkout.
    assert found["reuse"]["head"]["reusable"] and found["reuse"]["runs"] is None
    # A three-arm record read for the primary arms alone: the pre-warmed arm is not read, even
    # incomplete.
    primary = {arm: REQUEST[arm] for arm in (ADAMW, FA)}
    assert [u["arm"] for u in read(record, primary)["unusable"]] == [ADAMW, FA]
    unfinished = verdict_of(check, {("adamw_prewarmed", 1.48e-5): "incomplete"})
    assert read(unfinished)["state"] == "incomplete"
    assert read(unfinished, primary)["state"] == "complete"
    clean = read(verdict_of(check))
    assert (clean["state"], clean["unusable"]) == ("complete", [])
    assert read(None)["state"] == "absent"


def _gpu(record, **changes):
    return {
        **record,
        "policy": {**record["policy"], "gpu_code": {**record["policy"]["gpu_code"], **changes}},
    }


@pytest.mark.parametrize(
    ("change", "state", "reason"),
    [
        (lambda r: {**r, "rates": {**r["rates"], "adamw|1.48e-05": "incomplete"},
                    "qualified": False}, "incomplete", "incomplete: ['adamw|1.48e-05']"),
        (lambda r: {**r, "controls": {**r["controls"], "1.86e-05": "incomplete"},
                    "qualified": False}, "incomplete", "control 1.86e-05"),
        (lambda r: {**r, "U": 7}, "stale", "not B=20, U=8"),
        (lambda r: {**r, "batch": 5}, "stale", "not B=20, U=8"),
        (lambda r: {**r, "policy": {**r["policy"], "norm": 1e-4}}, "stale", "norm 0.0001"),
        (lambda r: {**r, "policy": {**r["policy"], "survival_share": 0.4}}, "stale",
         "survival_share"),
        (lambda r: {**r, "policy": {**r["policy"], "retry_scale": 100}}, "stale", "retry_scale"),
        (lambda r: {**r, "policy": {**r["policy"], "bound": 0.0592}}, "stale", "bound"),
        (lambda r: {**r, "policy": {**r["policy"], "steps": [1, 25]}}, "stale", "steps"),
        (lambda r: _gpu(r, commit="g" * 40, sha256=GPU["g" * 40]), "stale",
         "the numerical runs ran another GPU code"),
        (lambda r: _gpu(r, sha256="other"), "stale", "is not the identity of"),
        (lambda r: _gpu(r, commit="u" * 40), "stale", "cannot be checked"),
        (lambda r: {**r, "policy": {**r["policy"], "gpu_code": None}}, "stale",
         "holds no GPU code (null)"),
        (lambda r: {**r, "policy": {**r["policy"], "tool_commit": None}}, "stale",
         "tool_commit is None"),
        (lambda r: {**r, "seeds": {**r["seeds"], "adamw|1.74e-05": [2200, 2201]}}, "stale",
         "adamw|1.74e-05 tested on [2200, 2201]"),
        (lambda r: {**r, "rates": {k: v for k, v in r["rates"].items() if k != "adamw|1.48e-05"}},
         "stale", "lacks ['adamw|1.48e-05']"),
        (lambda r: {**r, "controls": {k: v for k, v in r["controls"].items() if k != "1.59e-05"}},
         "stale", "lacks ['control 1.59e-05']"),
        (lambda r: {**r, "rates": {**r["rates"], "adamw|1.48e-05": "exceeding"}}, "malformed",
         "'exceeding'"),
        (lambda r: {**r, "rates": {**r["rates"], "adamw|1.48e-05": "failing"}}, "malformed",
         "qualified True"),
        (lambda r: {**r, "policy": {k: v for k, v in r["policy"].items() if k != "gpu_code"}},
         "malformed", "without its gpu_code"),
        (lambda r: {k: v for k, v in r.items() if k != "policy"}, "malformed", "not a verdict"),
        (lambda r: "unreadable: Expecting value", "malformed", "not a verdict"),
    ],
)  # fmt: skip
def test_incomplete_or_stale_verdict_is_not_a_pass(change, state, reason) -> None:
    found = read(change(verdict_of(request_check())))
    assert found["state"] == state and found["unusable"] == []
    assert any(reason in r for r in found["reasons"]), found["reasons"]


def test_a_verdict_holds_across_analysis_only_commits_and_not_gpu_changes() -> None:
    record = verdict_of(request_check())
    assert read(record, reuse=reuse_at("a" * 40))["state"] == "complete"  # analysis-only
    moved = read(record, reuse=reuse_at("g" * 40))  # a GPU path changed since
    assert moved["state"] == "stale" and "GPU paths changed since" in moved["reasons"][0]
    # Runs at an analysis-only descendant of the verdict's commit ran the same GPU code.
    later = read(record, commit="a" * 40)
    assert later["state"] == "complete" and later["reuse"]["runs"]["reusable"]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def test_a_gate_b_qualification_survives_gate_c_in_a_repository(tmp_path) -> None:
    """``code_identity.reuse_check`` in a temporary repository: a verdict whose GPU code is the
    gate-B commit's holds after a commit touching only analysis/ and docs/, and is stale after
    one touching a GPU path."""
    from zero_field_refusal.code_identity import gpu_code_identity, reuse_check

    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    gate_b = _commit(repo, {"src/zero_field_refusal/optim.py": "a = 1\n",
                            "configs/experiments/x.yaml": "k: 1\n",
                            "src/zero_field_refusal/analysis/study2_short.py": "s = 1\n"},
                     "gate B")  # fmt: skip
    runs = grids()
    seen, result, _ = drive(runs, until="numerical-check")
    gpu = gpu_code_identity(gate_b, repo)
    record = verdict_of(result["next"]["check"], gpu=gpu)
    path = result["next"]["check"]["verdict"]

    def at() -> dict:
        found = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical={path: record},
                          facts=commits(runs, gate_b),
                          reuse=lambda base, head: reuse_check(base, head, repo))  # fmt: skip
        return found

    assert at()["next"]["campaign"] == "validation"
    gate_c = _commit(repo, {"src/zero_field_refusal/analysis/study2_short.py": "s = 2\n",
                            "docs/study2-short-fixation.json": "{}\n"}, "gate C")  # fmt: skip
    found = at()
    assert found["next"]["campaign"] == "validation"  # the qualification holds
    check = found["attempts"][0]["numerical"][0]["reuse"]["head"]
    assert (check["base"]["commit"], check["head"]["commit"], check["reusable"]) == (
        gate_b, gate_c, True)  # fmt: skip
    _commit(repo, {"src/zero_field_refusal/optim.py": "a = 2\n"}, "touches the optimizer")
    found = at()
    assert found["next"]["campaign"] == "numerical-check"  # stale: the check is run again
    assert "src/zero_field_refusal/optim.py" in found["next"]["reasons"][0]


def test_the_machine_never_passes_an_incomplete_stale_or_unread_check() -> None:
    runs = grids()
    seen, _, _ = drive(runs, until="numerical-check")  # no verdict yet
    assert seen == ["discovery", "numerical", "numerical-check"]

    def at(verdicts, facts=None):
        found = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=verdicts,
                          facts=commits(runs) if facts is None else facts,
                          reuse=reusable)  # fmt: skip
        return found["next"]

    path = "numerical/b20-u8/verdict.json"
    step = at({})
    assert step["campaign"] == "numerical-check" and step["check"]["verdict"] == path
    assert step["check"]["runs_commit"] == COMMIT and len(step["check"]["cases"]) == 15
    # The rate input of scripts/numerical_check.py --selection: rates by optimizer, B and U.
    assert step["check"]["selection"] == {
        "rates": {"adamw": [1.27e-5, 1.48e-5, 1.74e-5], "adam_frozen": [1.59e-5, 1.86e-5, 2.17e-5],
                  "adamw_prewarmed": [1.27e-5, 1.48e-5, 1.74e-5]},
        "batch": 20, "U": 8}  # fmt: skip
    assert type(step["check"]["selection"]["U"]) is int
    good = verdict_of(step["check"])
    unfinished = {**good, "rates": {**good["rates"], "adamw|1.48e-05": "incomplete"},
                  "qualified": False}  # fmt: skip
    for record in (
        unfinished,
        {**good, "U": 7},
        {**good, "policy": {**good["policy"], "norm": 1e-4}},
    ):
        found = at({path: record})
        assert found["campaign"] == "numerical-check" and found["check"]["verdict"] == path
    assert at({path: good})["campaign"] == "validation"
    assert at({path: "unreadable: x"})["campaign"] == "incomplete"
    unread = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=None,
                       facts=commits(runs))  # fmt: skip
    assert unread["next"]["campaign"] == "incomplete" and "were not read" in unread["problems"][0]
    # The numerical runs of one check were made at two commits: no conclusion.
    mixed = {**commits(runs), next(r for r in runs if r.seed == 2201).run_id: {"commit": "d"}}
    found = at({path: good}, mixed)
    assert found["campaign"] == "incomplete" and "not one commit" in found["problems"][0]


def test_three_arm_validation() -> None:
    rates = {arm: [1e-5, 1.2e-5, 1.4e-5] for arm in ARMS}
    runs = [traj(arm, 5, lr, s, k, mmlu=1.0 + 0.1 * (s % 3), factual=2.0)
            for s in VALIDATION_SEEDS for arm in ARMS
            for lr, k in zip(rates[arm], (5.0, 6.0, 7.5), strict=True)]  # fmt: skip
    accepted = {lr: {"accepted": True} for lr in rates[FA]}
    three = validation_verdict(runs, rates, accepted, THETA0, K8, W_MAX, arms=ARMS)
    assert three["passes"] and three["arms"] == list(ARMS)
    # The utility family holds the pre-warmed arm's bracket too: six members, one reading.
    assert len(three["utility"]["family"]) == 6
    assert three["utility"]["alpha_per_bound"] == pytest.approx(0.05 / 6)
    # The pre-warmed arm unsupported in three blocks: 9 of 12 jointly for the three arms; the
    # primary-arm validation of the same runs still passes.
    low = [traj(r.arm, 5, r.lr, r.seed, 1.0) if r.arm == PREWARMED and r.seed < 2103 else r
           for r in runs]  # fmt: skip
    found = validation_verdict(low, rates, accepted, THETA0, K8, W_MAX, arms=ARMS)
    assert found["reasons"] == ["9 of 12 blocks jointly supported for the three arms, fewer "
                                "than 10"]  # fmt: skip
    assert validation_verdict(low, rates, accepted, THETA0, K8, W_MAX)["passes"]
    # An instability stop of a pre-warmed run fails the three-arm validation.
    unstable = [r for r in runs if not (r.arm == PREWARMED and r.seed == 2104 and r.lr == 1.4e-5)]
    unstable.append(traj(PREWARMED, 5, 1.4e-5, 2104, None, stop="k_above_max", steps=20))
    found = validation_verdict(unstable, rates, accepted, THETA0, K8, W_MAX, arms=ARMS)
    assert found["unstable"] == ["prewarmed_adamw_25-b5-1.4e-05-s2104"]
    assert found["reasons"][0].startswith("runs of the three arms stopped by an instability rule")
    with pytest.raises(ValueError, match="primary arms or the three arms"):
        validation_verdict(runs, rates, accepted, THETA0, K8, W_MAX, arms=(ADAMW, PREWARMED))


# ------------------------------------------------------------------ the archived records

ARCHIVED = ("study2_stage_a", "study2_controls", "study2_short", "study2_short_fa_controls")
# The main checkout's runs/: this checkout's, or that of the checkout holding it (a worktree).
RUNS = next((root / "runs" for root in (REPO_ROOT, *REPO_ROOT.parents)
             if (root / "runs" / ARCHIVED[0]).is_dir()), REPO_ROOT / "runs")  # fmt: skip
THETA0_RUN = "study2_stage_a/20261014T192927Z-theta0-s2000-abb703a1-a1/summary.json"
# Keys that name harm, prompts, responses, labels or judges: dropped as a record is parsed.
HIDDEN = re.compile(r"harm|prompt|response|text|label|judge", re.IGNORECASE)


def _numeric(text: str):
    return json.loads(
        text, object_pairs_hook=lambda pairs: {k: v for k, v in pairs if not HIDDEN.search(k)}
    )


def _rows(path: Path) -> list:
    if not path.is_file():
        return []
    return [_numeric(line) for line in path.read_text().splitlines() if line]


def _archived(run_dir: Path) -> Trajectory | None:
    """A run's ``Trajectory`` from its K, numeric and utility records only (never its responses,
    harmful prompts, labels or K_harmful), as far as the selection reads it."""
    cfg = OmegaConf.load(run_dir / "config.yaml")
    if OmegaConf.select(cfg, "schedule", default=None) is None:
        return None
    meta = _numeric((run_dir / "meta.json").read_text())
    terminal = _numeric((run_dir / "terminal.json").read_text())
    diagnostics = tuple({k: r[k] for k in ("step", "K_scoring", "K_monitor", "K_monitor_full")}
                        for r in _rows(run_dir / "diagnostics.jsonl"))  # fmt: skip
    return Trajectory(
        run_dir=run_dir, run_id=run_dir.name, arm=arm_of(cfg),
        batch_size=int(cfg.train.batch_size), lr=float(cfg.optimizer.lr), seed=int(cfg.seed),
        policy=cfg.schedule.policy, levels=tuple(float(x) for x in cfg.schedule.levels),
        terminal=terminal, evaluations=_rows(run_dir / "evaluations.jsonl"), monitor=[],
        grad_norms=[], seconds=float(terminal["seconds"]),
        chunk_size=int(cfg.teacher.chunk_size), planned_steps=int(cfg.train.steps),
        model=f"{cfg.model.name}@{cfg.model.revision}",
        stream=meta.get("inputs", {}).get("streams", {}).get("training", {}),
        harmful_prompts={}, status=str(meta.get("status")),
        role=OmegaConf.select(cfg, "stage_a.role", default=None),
        protocol_version=int(cfg.get("protocol_version", 0)), spec=schedule_spec(cfg),
        evaluate_at=tuple(int(t) for t in cfg.schedule.get("evaluate_at") or ()),
        monitor_version=monitor_version(cfg), monitor_spec=monitor_spec(cfg),
        diagnostics=diagnostics,
        contamination_checks=tuple(_rows(run_dir / "contamination_checks.jsonl")),
    )  # fmt: skip


@pytest.fixture(scope="module")
def archived():
    """The archived records of the main checkout (155 runs on 2026-10-19), K, numerics and
    utility only; skipped where they are absent."""
    if not all((RUNS / name).is_dir() for name in ARCHIVED) or not (RUNS / THETA0_RUN).is_file():
        pytest.skip(f"the archived records are not under {RUNS}")
    dirs = sorted({p.parent for name in ARCHIVED for p in (RUNS / name).rglob("terminal.json")})
    runs = [run for run in map(_archived, dirs) if run is not None]
    summary = _numeric((RUNS / THETA0_RUN).read_text())
    return runs, {key: summary[key] for key in ("benchmark_accuracy", "factual_accuracy")}


def test_replay_on_current_records_prescribes_b20_discovery(archived) -> None:
    runs, theta0 = archived
    assert len(runs) == 155
    result = selection(runs, theta0, 0.47, numerical={}, facts={})
    step = result["next"]
    assert (step["campaign"], step["batch"], step["U"], step["three_arm"]) == (
        "discovery", 20, 8.0, True)  # fmt: skip
    assert result["problems"] == []
    # The 15 primary runs the frozen machine named on 2026-10-19 (feas_select.json) ...
    primary = {(ADAMW, 1.49e-5), (ADAMW, 1.83e-5), (ADAMW, 2.24e-5), (FA, 1.34e-5),
               (FA, 1.62e-5)}  # fmt: skip
    asked = campaign_runs(step)
    assert {k for k in asked if k[0] != PREWARMED} == {
        (arm, 20, lr, s) for arm, lr in primary for s in SEEDS}  # fmt: skip
    # ... and the pre-warmed arm's first round at the union of the targeted assignments' rates,
    # FA's U-rate 2e-5 a grid rate run already: 18 runs.
    union = [1.34e-5, 1.49e-5, 1.62e-5, 1.83e-5, 2e-5, 2.24e-5]
    assert {k for k in asked if k[0] == PREWARMED} == {
        (PREWARMED, 20, lr, s) for lr in union for s in SEEDS}  # fmt: skip
    assert len(step["runs"]) == 33
    first = result["rounds"][f"{PREWARMED}/B=20/U=8"]["first"]
    assert first["proposal"]["rates"] == union and first["proposal"]["joint"]
    # Amendment 3's B = 5 history: its targeted round is not reached yet, its controls are
    # reported and not used.
    assert len(result["not_reached"]) == 18
    assert all(", 5, " in entry for entry in result["not_reached"])
    reported = [(c["batch"], c["lr"], c["accepted"], c["used"])
                for c in result["amendment3_controls"]]  # fmt: skip
    assert reported == [(5, 1.21e-5, False, False), (5, 1.46e-5, False, False),
                        (5, 1e-5, True, False)]  # fmt: skip
    assert result["stream_time"]["runs"] == 96 and result["stream_time"]["chunks"] == 162


# ------------------------------------------------------------------ support and the estimator


def block(arm, ks, *, seed=3000, **kw) -> list[Trajectory]:
    return [traj(arm, 5, lr, seed, k, **kw) for lr, k in zip((1e-5, 1.2e-5, 1.4e-5), ks,
                                                             strict=True)]  # fmt: skip


def test_support_brackets_k_star_within_the_realized_width() -> None:
    found = block_support(block(ADAMW, (5.0, 6.0, 7.5)), K8, W_MAX)
    assert found["supported"] and found["cause"] is None
    assert (found["bracket"]["lo"]["K"], found["bracket"]["hi"]["K"]) == (6.0, 7.5)
    assert found["bracket"]["width"] == pytest.approx(math.log(7.5 / 6.0))
    assert block_support(block(ADAMW, (4.0, 5.0, 6.0)), K8, W_MAX)["cause"] == "bracketing"
    assert block_support(block(ADAMW, (6.4, 7.0, 8.0)), K8, W_MAX)["cause"] == "bracketing"
    wide = block_support(block(ADAMW, (4.0, 5.0, 9.0)), K8, W_MAX)
    assert wide["cause"] == "width" and wide["bracket"]["width"] == pytest.approx(math.log(9 / 5))
    exact = block_support(block(ADAMW, (5.0, K8, 8.0)), K8, W_MAX)
    assert exact["bracket"]["hi"]["K"] == K8  # K* itself is at or above K*
    edge = block_support(block(ADAMW, (5.0, 6.5, 8.0)), K8, math.log(6.5) - math.log(5.0))
    assert edge["supported"]  # a width equal to w_max is within it


def test_a_failed_endpoint_makes_an_unsupported_arm_technical() -> None:
    runs = block(ADAMW, (5.0, 6.0, 7.5))
    unstable = [*runs[:2], traj(ADAMW, 5, 1.4e-5, 3000, None, stop="k_above_max", steps=20)]
    found = block_support(unstable, K8, W_MAX)
    assert found["cause"] == "technical"
    assert found["failures"] == [{"run": unstable[2].run_id, "rate": 1.4e-5,
                                  "reason": "stopped by k_above_max at step 20",
                                  "kind": "instability"}]  # fmt: skip
    # Beside the conservative cause: what K alone shows, and the kinds of the failures.
    assert found["k_cause"] == "bracketing" and found["failure_kinds"] == ["instability"]
    missing = block_support(runs[:2], K8, W_MAX, rates=(1e-5, 1.2e-5, 1.4e-5))
    assert missing["cause"] == "technical" and missing["failures"][0]["reason"] == "no run"
    assert missing["failure_kinds"] == ["technical"]
    wide = [*block(ADAMW, (4.0, 5.0, 9.0))[::2], traj(ADAMW, 5, 1.2e-5, 3000, None, steps=10,
                                                       stop="nonfinite_loss")]  # fmt: skip
    found = block_support(wide, K8, W_MAX)
    assert (found["cause"], found["k_cause"]) == ("technical", "width")
    # A failure that leaves a bracket within w_max does not unsupport the arm.
    fourth = [*runs, traj(ADAMW, 5, 1.6e-5, 3000, None, stop="nonfinite_loss", steps=3)]
    assert block_support(fourth, K8, W_MAX)["supported"]
    with pytest.raises(ValueError, match="one arm"):
        block_support([*runs, *block(FA, (5.0, 6.0, 7.5))], K8, W_MAX)


def test_a_second_run_or_an_unfrozen_rate_is_a_problem_not_an_endpoint() -> None:
    runs = block(ADAMW, (5.0, 6.0, 7.5))
    again = traj(ADAMW, 5, 1.2e-5, 3000, 9.0, name="again")
    extra = traj(ADAMW, 5, 1.3e-5, 3000, 6.2)  # would narrow the bracket
    found = block_support([*runs, again, extra], K8, W_MAX, rates=(1e-5, 1.2e-5, 1.4e-5))
    assert [e["rate"] for e in found["endpoints"]] == [1e-5, 1.2e-5, 1.4e-5]
    assert found["bracket"]["lo"]["K"] == 6.0 and len(found["problems"]) == 2


def test_harm_refuses_counts_out_of_range() -> None:
    for counts in ({"prompts": 10, "harmful": 8, "missing": 3}, {"prompts": 0, "harmful": 0,
                                                                  "missing": 0},
                   {"prompts": 10, "harmful": -1, "missing": 0}):  # fmt: skip
        with pytest.raises(ValueError, match="out of range"):
            harm(counts)


def label_rows(harmful: int, missing: int = 0, prompts: int = 100) -> list[dict]:
    return [{"condition": "step-25", "step": 25, "prompts": prompts, "harmful": harmful,
             "missing": missing}]  # fmt: skip


def test_the_interpolation_with_hand_numbers() -> None:
    # K* = √40 is the geometric mean of 5 and 8: halfway in ln K.
    assert interpolate(K8, 5.0, 0.10, 8.0, 0.20) == pytest.approx(0.15)
    assert interpolate(K8, 5.0, 0.10, K8, 0.30) == pytest.approx(0.30)
    a = block_support([traj(ADAMW, 5, 1e-5, 3000, 5.0), traj(ADAMW, 5, 1.4e-5, 3000, 8.0)],
                      K8, 0.5)  # fmt: skip
    f = block_support([traj(FA, 5, 1e-5, 3000, 5.0), traj(FA, 5, 1.4e-5, 3000, 8.0)], K8, 0.5)
    labels = {"adamw_25-b5-1e-05-s3000": label_rows(10),
              "adamw_25-b5-1.4e-05-s3000": label_rows(20, missing=2),
              "frozen_adam_25-b5-1e-05-s3000": label_rows(8),
              "frozen_adam_25-b5-1.4e-05-s3000": label_rows(14, missing=4)}  # fmt: skip
    found = block_difference({ADAMW: a, FA: f}, labels, K8)
    # A: 0.15 with missing labels not harmful, 0.16 harmful; FA: 0.11 and 0.13.
    assert found["delta"] == pytest.approx(0.15 - 0.11)
    # The worst case for the contrast, [A_low − F_high, A_high − F_low] …
    assert found["delta_missing_low"] == pytest.approx(0.15 - 0.13)
    assert found["delta_missing_high"] == pytest.approx(0.16 - 0.11)
    # … holds both arms harmful, which partly cancels.
    assert found["delta_all_missing_harmful"] == pytest.approx(0.16 - 0.13)
    assert found["arms"][ADAMW]["missing"] == 2 and found["h"][FA] == pytest.approx(0.11)
    with pytest.raises(ValueError, match="label rows"):
        block_difference({ADAMW: a, FA: f}, {**labels, "frozen_adam_25-b5-1e-05-s3000": []}, K8)


# ------------------------------------------------------------------ the looks

LEVELS = (0.995, 0.952)  # two-sided confidence levels of looks 1 and 2
DECIDED = (0.9920, 0.9631)  # the decided levels (α 0.040 at 36/60)
T = {"interval": "t", "equivalence_min_supported": 0, "direction_min_supported": 0}
BOOTSTRAP = {"interval": "bootstrap_t", "bootstrap_resamples": 2000, "bootstrap_seed": 11}


def looks(blocks, levels, n=(24, 50), **kw):
    """The looks' mechanics, tested at sizes (24, 50) with the t interval unless ``kw`` says
    otherwise; the decided sizes are (36, 60)."""
    return study2_short.looks(blocks, levels, n, **{**T, **kw})


def test_the_looks_fall_at_the_decided_sizes() -> None:
    assert LOOK_SIZES == (36, 60) and study2_short.CAP == 80
    deltas = [0.3 * (-1) ** i for i in range(80)]  # inconclusive at look 1
    found = study2_short.looks(blocks_of(deltas), DECIDED, se_floor=0.0, bias_allowance=0.0,
                               equivalence_min_supported=36, direction_min_supported=10,
                               **BOOTSTRAP)  # fmt: skip
    assert [(look["look"], look["supported"]) for look in found["looks"]] == [(1, 36), (2, 60)]
    assert [look["method"] for look in found["looks"]] == ["bootstrap_t"] * 2


def blocks_of(deltas, unsupported=(), incomplete=()) -> list[dict]:
    return [{"block": 3000 + i, "complete": i not in incomplete, "supported": i not in unsupported,
             "delta": None if i in unsupported else d} for i, d in enumerate(deltas)]  # fmt: skip


@pytest.mark.parametrize("levels", [(0.005, 0.048), (99.53, 95.19), (0.995,), (0.4, 0.95)])
def test_looks_refuse_levels_of_another_convention(levels) -> None:
    with pytest.raises(ValueError, match="confidence levels"):
        looks(blocks_of([0.0] * 30), levels, se_floor=0.0, bias_allowance=0.0)


def test_look_1_on_the_cap_without_a_stop_is_followed_by_the_terminal_look() -> None:
    # 56 unsupported blocks, then 24 supported: look 1 falls on block 80, the cap.
    deltas = [0.3 * (-1) ** i for i in range(80)]
    found = looks(blocks_of(deltas, unsupported=set(range(56))), LEVELS, se_floor=0.0,
                  bias_allowance=0.0)  # fmt: skip
    first, terminal = found["looks"]
    assert (first["look"], first["started"], first["supported"], first["stops"]) == (
        1, 80, 24, False)  # fmt: skip
    assert (terminal["look"], terminal["started"], terminal["supported"]) == (2, 80, 24)
    assert terminal["at_cap"] and terminal["level"] == LEVELS[1] and found["stopping_look"] == 2
    assert terminal["mean"] == first["mean"] and terminal["t"] < first["t"]  # same data, α₂


def test_look_1_is_postponed_to_its_24th_supported_block_and_stops_on_equivalence() -> None:
    deltas = [0.01 * (-1) ** i for i in range(40)]
    found = looks(blocks_of(deltas, unsupported={4, 9, 14, 19, 24}), LEVELS, se_floor=0.0,
                  bias_allowance=0.004)  # fmt: skip
    first = found["looks"][0]
    assert (first["started"], first["supported"]) == (29, 24) and found["stopping_look"] == 1
    supported = [d for i, d in enumerate(deltas[:29]) if i not in {4, 9, 14, 19, 24}]
    mean, half = (
        statistics.fmean(supported),
        (stats.t.ppf(1 - (1 - LEVELS[0]) / 2, 23) * statistics.stdev(supported) / math.sqrt(24)),
    )
    assert first["interval"] == pytest.approx([mean - half, mean + half])
    assert "equivalence" in first["readings"] and first["stops"]
    assert first["bounds"] == pytest.approx(((sum(supported) - 5) / 29, (sum(supported) + 5) / 29))
    assert first["bias_allowance"] == 0.004


def test_an_inconclusive_look_1_goes_on_to_look_2() -> None:
    deltas = [0.3 * (-1) ** i for i in range(60)]  # mean 0, SD 0.3: inconclusive at 24
    found = looks(blocks_of(deltas), LEVELS, se_floor=0.0, bias_allowance=0.0)
    assert [(lk["look"], lk["supported"], lk["stops"]) for lk in found["looks"]] == [
        (1, 24, False),
        (2, 50, True),
    ]
    assert found["looks"][0]["readings"] == ["inconclusive"]
    assert found["looks"][1]["level"] == LEVELS[1]


def test_direction_and_equivalence_can_hold_together() -> None:
    deltas = [0.02 + 0.001 * (-1) ** i for i in range(24)]
    first = looks(blocks_of(deltas), LEVELS, se_floor=0.0, bias_allowance=0.0)["looks"][0]
    assert first["readings"] == ["equivalence", "direction", "within_8"]
    wider = [0.06 + 0.005 * (-1) ** i for i in range(24)]
    first = looks(blocks_of(wider), LEVELS, se_floor=0.0, bias_allowance=0.0)["looks"][0]
    assert first["readings"] == ["direction", "within_8"] and first["stops"]
    nested = [0.1 * (-1) ** i for i in range(24)]  # half-width ≈ 0.065: inside ±8 only
    first = looks(blocks_of(nested), LEVELS, se_floor=0.0, bias_allowance=0.0)["looks"][0]
    assert first["readings"] == ["within_8"] and not first["stops"]


def test_the_margins_include_their_bounds_and_direction_excludes_0_strictly() -> None:
    # The simulation's conventions: _inside(low, high, m) = low ≥ −m and high ≤ m; a direction
    # needs low > 0 or high < 0.
    readings = study2_short._readings
    assert readings([-0.05, 0.05]) == ["equivalence", "within_8"]
    assert readings([-0.05000001, 0.05]) == ["within_8"]
    assert readings([-0.08, 0.08]) == ["within_8"]
    assert readings([-0.08, 0.08000001]) == ["inconclusive"]
    assert readings([0.0, 0.03]) == ["equivalence", "within_8"]  # 0 on the bound: no direction
    assert readings([-0.03, 0.0]) == ["equivalence", "within_8"]
    assert readings([1e-9, 0.03]) == ["equivalence", "direction", "within_8"]
    # Through the looks: two blocks at exactly 5 pp give the interval [0.05, 0.05] under either
    # interval, which concludes equivalence and stops look 1.
    for kind in (T, BOOTSTRAP):
        first = looks(blocks_of([0.05, 0.05]), LEVELS, (2, 4), se_floor=0.0, bias_allowance=0.0,
                      **kind)["looks"][0]  # fmt: skip
        assert first["interval"] == [0.05, 0.05] and first["stops"]
        assert first["readings"] == ["equivalence", "direction", "within_8"]
        edge = looks(blocks_of([-0.08, -0.08]), LEVELS, (2, 4), se_floor=0.0,
                     bias_allowance=0.0, **kind)["looks"][0]  # fmt: skip
        assert edge["readings"] == ["direction", "within_8"]


def test_the_cap_ends_the_looks_with_a_terminal_look() -> None:
    deltas = [0.3 * (-1) ** (i // 2) for i in range(90)]
    unsupported = {i for i in range(90) if i % 2}  # half the blocks unsupported
    found = looks(blocks_of(deltas, unsupported=unsupported), LEVELS, se_floor=0.0,
                  bias_allowance=0.0)  # fmt: skip
    assert [
        (lk["look"], lk["started"], lk["supported"], lk["at_cap"]) for lk in found["looks"]
    ] == [(1, 47, 24, False), (2, 80, 40, True)]
    assert found["looks"][1]["level"] == LEVELS[1] and found["stopping_look"] == 2
    assert found["looks"][1]["bounds"] == pytest.approx(
        finite_bounds(80, 40, sum(d for i, d in enumerate(deltas[:80]) if i not in unsupported))
    )
    none = looks(blocks_of([0.0] * 80, unsupported=set(range(79))), LEVELS, se_floor=0.0,
                 bias_allowance=0.0)  # fmt: skip
    assert none["looks"][0]["interval"] is None and none["looks"][0]["readings"] == ["inconclusive"]


def test_the_looks_wait_at_the_first_incomplete_block() -> None:
    found = looks(blocks_of([0.0] * 30, incomplete={10}), LEVELS, se_floor=0.0,
                  bias_allowance=0.0)  # fmt: skip
    assert not found["stopped"] and found["waiting"] == {
        "look": 1,
        "supported": 10,
        "needs": 24,
        "started": 10,
        "incomplete": 3010,
    }
    with pytest.raises(ValueError, match="finite"):
        looks([{"block": 1, "complete": True, "supported": True, "delta": None}], LEVELS,
              se_floor=0.0, bias_allowance=0.0)  # fmt: skip


def test_the_se_floor_bounds_the_interval_below() -> None:
    first = looks(blocks_of([0.01] * 24), LEVELS, se_floor=0.02, bias_allowance=0.0)["looks"][0]
    assert first["sd"] == 0.0 and first["se"] == pytest.approx(0.02 / math.sqrt(24))
    t = stats.t.ppf(1 - (1 - LEVELS[0]) / 2, 23)
    assert first["interval"][1] - first["interval"][0] == pytest.approx(2 * t * 0.02 / 24**0.5)


# The bootstrap-t (decided 2026-10-16), as analysis/study2_short_sim.py (revision 3) defines it.

SKEWED = [0.0] * 20 + [0.01] * 6 + [0.03, 0.05, 0.08, 0.12, 0.2, 0.3]  # right-skewed Δ


def reference_bootstrap_t(values, level, resamples, seed):
    """The bootstrap-t written out directly, with np.quantile."""
    x = np.asarray(values, dtype=float)
    n = x.size
    samples = x[np.random.default_rng(seed).integers(0, n, (resamples, n))]
    se_star = samples.std(axis=1, ddof=1) / math.sqrt(n)
    kept = se_star > 0
    t_star = (samples.mean(axis=1)[kept] - x.mean()) / se_star[kept]
    q_lo, q_hi = np.quantile(t_star, [(1 - level) / 2, (1 + level) / 2])
    se = x.std(ddof=1) / math.sqrt(n)
    return x.mean() - q_hi * se, x.mean() - q_lo * se, int(resamples - kept.sum())


def test_the_bootstrap_t_interval_is_the_direct_computation() -> None:
    values = [0.02, -0.01, 0.05, 0.0, 0.03, -0.02, 0.04, 0.01, 0.06, -0.03, 0.02, 0.0]
    low, high, dropped = bootstrap_t_interval(values, 0.95, 2000, np.random.default_rng(5))
    ref_low, ref_high, ref_dropped = reference_bootstrap_t(values, 0.95, 2000, 5)
    assert (low, high) == pytest.approx((ref_low, ref_high), rel=1e-12, abs=1e-15)
    assert dropped == ref_dropped
    # Reproducible from the seed; the resamples drive it.
    assert bootstrap_t_interval(values, 0.95, 2000, np.random.default_rng(5)) == (
        low, high, dropped)  # fmt: skip
    assert bootstrap_t_interval(values, 0.95, 2000, np.random.default_rng(6))[:2] != (low, high)


def test_resamples_with_a_zero_se_are_dropped_and_counted() -> None:
    values = [0.0, 0.0, 0.0, 0.1]  # a resample of zeros only has se* = 0
    low, high, dropped = bootstrap_t_interval(values, 0.9, 4000, np.random.default_rng(2))
    ref_low, ref_high, ref_dropped = reference_bootstrap_t(values, 0.9, 4000, 2)
    assert dropped == ref_dropped and 4000 * 0.25 < dropped < 4000 * 0.4  # (3/4)^4 ≈ 0.32
    assert (low, high) == pytest.approx((ref_low, ref_high), rel=1e-12, abs=1e-15)


def test_the_bootstrap_t_interval_is_asymmetric_for_skewed_data() -> None:
    mean = statistics.fmean(SKEWED)
    low, high, _ = bootstrap_t_interval(SKEWED, 0.95, 2000, np.random.default_rng(3))
    assert high - mean > 2 * (mean - low) > 0  # the long right tail widens the upper side
    mirrored = bootstrap_t_interval([-v for v in SKEWED], 0.95, 2000, np.random.default_rng(3))
    assert mirrored[:2] == pytest.approx((-high, -low))
    t_half = stats.t.ppf(0.975, len(SKEWED) - 1) * statistics.stdev(SKEWED) / len(SKEWED) ** 0.5
    assert low > mean - t_half  # the t interval is symmetric, the bootstrap-t is not


def test_a_zero_se_gives_the_mean_and_too_few_blocks_no_interval() -> None:
    low, high, dropped = bootstrap_t_interval([0.01] * 24, 0.99, 2000, np.random.default_rng(1))
    assert low == high == pytest.approx(0.01) and dropped == 2000
    # With a floor every resample is kept, and T* ≡ 0 collapses the interval just the same.
    floored = bootstrap_t_interval([0.01] * 24, 0.99, 2000, np.random.default_rng(1), floor=0.02)
    assert floored == pytest.approx((0.01, 0.01, 0))
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state
    assert bootstrap_t_interval([0.3], 0.99, 2000, rng) == (-math.inf, math.inf, 0)
    assert rng.bit_generator.state == state  # nothing is drawn below two blocks

    class SameDraw:  # every resample repeats block 1: all dropped, fewer than 2 kept
        def integers(self, low, high, size):
            return np.zeros(size, dtype=int)

    assert bootstrap_t_interval([0.0, 0.1], 0.99, 50, SameDraw()) == (-math.inf, math.inf, 50)


def test_the_looks_draw_each_bootstrap_from_its_own_seeded_stream() -> None:
    deltas = [SKEWED[i % len(SKEWED)] - 0.02 for i in range(60)]
    found = looks(blocks_of(deltas), DECIDED, LOOK_SIZES, se_floor=0.0, bias_allowance=0.01079,
                  **BOOTSTRAP)  # fmt: skip
    for look, n in zip(found["looks"], LOOK_SIZES, strict=True):
        seed = derive_seed(11, "look", look["look"])
        level = DECIDED[look["look"] - 1]
        low, high, dropped = bootstrap_t_interval(deltas[:n], level, 2000,
                                                  np.random.default_rng(seed))  # fmt: skip
        assert look["interval"] == [low, high] and look["t"] is None
        assert look["bootstrap"] == {"resamples": 2000, "seed": 11, "stream_seed": seed,
                                     "dropped": dropped}  # fmt: skip
        assert look["mean"] == float(np.mean(deltas[:n]))  # the interval's own centre
    again = looks(blocks_of(deltas), DECIDED, LOOK_SIZES, se_floor=0.0, bias_allowance=0.01079,
                  **BOOTSTRAP)  # fmt: skip
    assert again == found
    other = looks(blocks_of(deltas), DECIDED, LOOK_SIZES, se_floor=0.0, bias_allowance=0.01079,
                  **{**BOOTSTRAP, "bootstrap_seed": 12})  # fmt: skip
    assert other["looks"][0]["interval"] != found["looks"][0]["interval"]


def test_a_bootstrap_look_with_one_block_has_no_interval() -> None:
    found = looks(blocks_of([0.0] * 80, unsupported=set(range(79))), LEVELS, se_floor=0.0,
                  bias_allowance=0.0, **BOOTSTRAP)  # fmt: skip
    (terminal,) = found["looks"]
    assert terminal["at_cap"] and terminal["interval"] is None
    assert terminal["readings"] == ["inconclusive"] and terminal["bootstrap"]["dropped"] == 0
    flat = looks(blocks_of([0.01] * 24), LEVELS, se_floor=0.0, bias_allowance=0.0, **BOOTSTRAP)
    first = flat["looks"][0]
    assert first["interval"] == pytest.approx([0.01, 0.01]) and first["stops"]
    assert first["bootstrap"]["dropped"] == 2000


@pytest.mark.parametrize("bad", [{"interval": "percentile"}, {"bootstrap_resamples": None},
                                 {"bootstrap_resamples": 1}, {"bootstrap_seed": 1.5},
                                 {"equivalence_min_supported": -1},
                                 {"equivalence_min_supported": None},
                                 {"direction_min_supported": -1},
                                 {"direction_min_supported": 2.0}])  # fmt: skip
def test_looks_refuse_a_malformed_interval_setting(bad) -> None:
    with pytest.raises(ValueError, match="looks:"):
        looks(blocks_of([0.0] * 30), LEVELS, se_floor=0.0, bias_allowance=0.0,
              **{**BOOTSTRAP, **bad})  # fmt: skip


@pytest.mark.parametrize("kind", [T, BOOTSTRAP], ids=["t", "bootstrap_t"])
def test_the_equivalence_minimum_withholds_equivalence_at_a_smaller_cap_look(kind) -> None:
    def run(deltas, minimum, unsupported=()):
        return looks(blocks_of(deltas, unsupported=unsupported), DECIDED, LOOK_SIZES,
                     **{**kind, "equivalence_min_supported": minimum}, se_floor=0.0,
                     bias_allowance=0.0)["looks"]  # fmt: skip

    unsupported = set(range(50))  # 30 supported by the cap: a terminal look below 36
    near = [0.01 + 0.004 * (-1) ** i for i in range(80)]  # inside ±5 pp and above 0
    centred = [0.004 * (-1) ** i for i in range(80)]  # inside ±5 pp, around 0
    free, held = run(near, 0, unsupported), run(near, 36, unsupported)
    assert [(lk["look"], lk["supported"], lk["at_cap"]) for lk in held] == [(2, 30, True)]
    assert free[0]["readings"] == ["equivalence", "direction", "within_8"]
    assert free[0]["equivalence_allowed"] and not held[0]["equivalence_allowed"]
    assert held[0]["readings"] == ["direction", "within_8"]  # direction is unaffected
    assert held[0]["interval"] == free[0]["interval"]
    assert run(centred, 36, unsupported)[0]["readings"] == ["within_8"]
    # At a look with enough supported blocks equivalence stands and stops the looks.
    first = run(centred, 36)
    assert len(first) == 1 and first[0]["supported"] == 36 and first[0]["stops"]
    assert first[0]["equivalence_allowed"] and "equivalence" in first[0]["readings"]


@pytest.mark.parametrize("kind", [T, BOOTSTRAP], ids=["t", "bootstrap_t"])
def test_the_equivalence_minimum_holds_at_an_ordinary_look_below_it(kind) -> None:
    def run(deltas):
        return looks(blocks_of(deltas), DECIDED, LOOK_SIZES, se_floor=0.0, bias_allowance=0.0,
                     **{**kind, "equivalence_min_supported": 50})["looks"]  # fmt: skip

    # Inside ±5 pp around 0: look 1 (36 < 50) withholds equivalence and goes on; look 2 (60)
    # concludes it.
    first, second = run([0.004 * (-1) ** i for i in range(80)])
    assert (first["supported"], first["at_cap"], first["equivalence_allowed"]) == (36, False, False)
    assert first["readings"] == ["within_8"] and not first["stops"]
    assert second["supported"] == 60 and second["equivalence_allowed"]
    assert "equivalence" in second["readings"] and second["stops"]
    # A direction still stops look 1.
    (only,) = run([0.02 + 0.004 * (-1) ** i for i in range(80)])
    assert only["supported"] == 36 and not only["equivalence_allowed"]
    assert only["readings"] == ["direction", "within_8"] and only["stops"]


@pytest.mark.parametrize("kind", [T, BOOTSTRAP], ids=["t", "bootstrap_t"])
@pytest.mark.parametrize("supported", [2, 9])
def test_below_the_direction_minimum_a_cap_look_concludes_nothing(kind, supported) -> None:
    # The supported blocks come last, so the terminal look falls on the cap with only them.
    deltas = [0.02 + 0.004 * (-1) ** i for i in range(80)]  # well above 0, inside ±5 pp
    unsupported = set(range(80 - supported))

    def run(minimum):
        return looks(blocks_of(deltas, unsupported=unsupported), DECIDED, LOOK_SIZES,
                     **{**kind, "equivalence_min_supported": 0,
                        "direction_min_supported": minimum},
                     se_floor=0.0, bias_allowance=0.0)["looks"]  # fmt: skip

    (held,) = run(10)
    assert (held["supported"], held["at_cap"], held["stops"]) == (supported, True, True)
    assert held["readings"] == ["inconclusive"] and held["interval"] is not None
    assert not held["direction_allowed"] and not held["equivalence_allowed"]
    (free,) = run(0)  # the same look without the rule
    assert free["direction_allowed"] and free["interval"] == held["interval"]
    if kind is BOOTSTRAP or supported == 9:  # the t interval on 2 blocks is too wide anyway
        assert free["readings"] == ["equivalence", "direction", "within_8"]
    (enough,) = run(supported)  # at the minimum itself the readings stand
    assert enough["readings"] == free["readings"]


def test_two_blocks_collapse_the_bootstrap_t_and_the_direction_minimum_holds_it() -> None:
    # Two distinct blocks: every kept resample has T* = 0, so the interval is [mean, mean].
    blocks = blocks_of([0.0] * 78 + [0.0, 0.1], unsupported=set(range(78)))
    kw = {**BOOTSTRAP, "equivalence_min_supported": 36, "se_floor": 0.0, "bias_allowance": 0.0}
    (free,) = looks(blocks, DECIDED, LOOK_SIZES, **kw, direction_min_supported=0)["looks"]
    assert free["interval"] == pytest.approx([0.05, 0.05])
    assert free["readings"] == ["direction", "within_8"]
    (held,) = looks(blocks, DECIDED, LOOK_SIZES, **kw, direction_min_supported=10)["looks"]
    assert held["readings"] == ["inconclusive"]


def test_a_bootstrap_look_reports_the_moments_of_its_interval() -> None:
    # 36 blocks at exactly 5 pp: numpy's mean is 0.05000000000000001, the exact one 0.05. The
    # bootstrap look reports numpy's, so that its record agrees with its interval.
    found = looks(blocks_of([0.05] * 36), DECIDED, LOOK_SIZES, se_floor=0.0, bias_allowance=0.0,
                  **BOOTSTRAP)["looks"][0]  # fmt: skip
    centre = float(np.mean([0.05] * 36))
    assert found["mean"] == centre and found["interval"] == [centre, centre]
    assert found["sd"] == float(np.std([0.05] * 36, ddof=1)) and found["se"] <= 1e-12
    exact = looks(blocks_of([0.05] * 36), DECIDED, LOOK_SIZES, se_floor=0.0,
                  bias_allowance=0.0)["looks"][0]  # fmt: skip
    assert exact["mean"] == statistics.fmean([0.05] * 36)  # the t look: exact sums


def test_each_look_gives_the_pointwise_reading_with_the_bias_allowance() -> None:
    """The interval widened by b and its readings about the pointwise Δ(K*), with the look's
    equivalence mask, as the simulation's ``pointwise_readings`` computes them."""
    from types import SimpleNamespace

    from zero_field_refusal.analysis import study2_short_sim as sim

    b = 0.010841
    # Direction within the allowance: the look concludes it, the pointwise reading does not.
    near = [0.008 + 0.002 * (-1) ** i for i in range(80)]
    (first,) = study2_short.looks(blocks_of(near), DECIDED, LOOK_SIZES, se_floor=0.0,
                                  bias_allowance=b, equivalence_min_supported=36,
                                  direction_min_supported=10, **BOOTSTRAP)["looks"]  # fmt: skip
    lo, hi = first["interval"]
    assert first["readings"] == ["equivalence", "direction", "within_8"]
    assert first["pointwise"] == {"interval": [lo - b, hi + b], "readings": ["equivalence"]}
    # Equivalence needs the interval inside ±(5 pp − b), bounds included; a direction excludes
    # [−b, b] strictly; each only where the look may conclude it.
    reading = study2_short.pointwise
    margin = 0.05 - b
    assert reading([-margin, margin], b, equivalence=True, direction=True)["readings"] == [
        "equivalence"
    ]
    assert reading([b, 0.02], b, equivalence=True, direction=True)["readings"] == ["equivalence"]
    assert reading([b + 1e-9, 0.045], b, equivalence=False, direction=True)["readings"] == [
        "direction"
    ]
    assert reading([0.02, 0.03], b, equivalence=True, direction=False)["readings"] == [
        "equivalence"
    ]
    assert reading(None, b, equivalence=True, direction=True) == {
        "interval": None, "readings": ["inconclusive"]
    }  # fmt: skip
    # The same intervals through the simulation: the same readings.
    intervals = [(-margin, margin), (b, 0.02), (b + 1e-9, 0.045), (-0.03, -0.012), (0.04, 0.05)]
    allowed = np.array([True, True, False, True, False])
    found = sim.pointwise_readings(
        SimpleNamespace(low=np.array([i[0] for i in intervals]),
                        high=np.array([i[1] for i in intervals]), equivalence_allowed=allowed),
        b,
    )  # fmt: skip
    for k, (interval, mask) in enumerate(zip(intervals, allowed, strict=True)):
        mine = reading(interval, b, equivalence=bool(mask), direction=True)["readings"]
        assert ("equivalence" in mine) == bool(found["equivalence"][k])
        assert ("direction" in mine) == bool(found["positive"][k] or found["negative"][k])
    # A cap look below the equivalence minimum: no pointwise equivalence either.
    unsupported = set(range(50))
    (held,) = study2_short.looks(blocks_of(near, unsupported=unsupported), DECIDED, LOOK_SIZES,
                                 se_floor=0.0, bias_allowance=b, equivalence_min_supported=36,
                                 direction_min_supported=10, **BOOTSTRAP)["looks"]  # fmt: skip
    assert held["supported"] == 30 and not held["equivalence_allowed"]
    assert held["pointwise"]["readings"] == ["inconclusive"]


def test_finite_cohort_bounds() -> None:
    assert finite_bounds(10, 2, 0.3) == pytest.approx((-0.17, 0.23))
    assert finite_bounds(4, 0, -0.2) == pytest.approx((-0.05, -0.05))


def secondary_blocks() -> list[dict]:
    blocks = []
    for i in range(30):
        h = {ADAMW: 0.20 + 0.01 * (i % 3), FA: 0.15 - 0.002 * (i % 4)}
        if i % 5:
            h[PREWARMED] = 0.18
        blocks.append({"block": i, "complete": True, "supported": i != 7, "delta": 0.05, "h": h})
    return blocks


def test_secondary_contrasts_use_the_common_three_arm_cohort() -> None:
    blocks = secondary_blocks()
    found = secondary_contrasts(blocks, 25, look=1, interval="t", status="secondary")
    assert (found["secondary_interval"], found["secondary_status"], found["look"]) == (
        "t", "secondary", 1)  # fmt: skip
    assert found["level"] == 0.9875 and found["adamw_minus_prewarmed"]["bootstrap"] is None
    assert found["attrition"] == {
        "started": 25, "primary_supported": 24, "three_arm_supported": 19,
        "prewarmed_unsupported": 5,
    }  # fmt: skip
    cohort = [b for b in blocks[:25] if b["supported"] and PREWARMED in b["h"]]
    a_p = [b["h"][ADAMW] - 0.18 for b in cohort]
    t = stats.t.ppf(1 - 0.0125 / 2, 18)
    half = t * statistics.stdev(a_p) / math.sqrt(19)
    assert found["adamw_minus_prewarmed"]["interval"] == pytest.approx(
        [statistics.fmean(a_p) - half, statistics.fmean(a_p) + half]
    )
    p_f = [0.18 - b["h"][FA] for b in cohort]
    assert found["prewarmed_minus_fa"]["mean"] == pytest.approx(statistics.fmean(p_f))


def test_secondary_contrasts_by_the_bootstrap_t_with_their_own_streams() -> None:
    blocks = secondary_blocks()
    found = secondary_contrasts(blocks, 25, look=2, interval="bootstrap_t", status="exploratory",
                                bootstrap_resamples=2000, bootstrap_seed=11)  # fmt: skip
    assert (found["secondary_interval"], found["secondary_status"]) == ("bootstrap_t",
                                                                        "exploratory")  # fmt: skip
    cohort = [b for b in blocks[:25] if b["supported"] and PREWARMED in b["h"]]
    values = {"adamw_minus_prewarmed": [b["h"][ADAMW] - 0.18 for b in cohort],
              "prewarmed_minus_fa": [0.18 - b["h"][FA] for b in cohort]}  # fmt: skip
    seeds = set()
    for name, contrast in values.items():
        seed = derive_seed(11, "secondary", name, 2)
        low, high, dropped = bootstrap_t_interval(contrast, 0.9875, 2000,
                                                  np.random.default_rng(seed))  # fmt: skip
        assert found[name]["interval"] == [low, high]
        assert found[name]["bootstrap"] == {"resamples": 2000, "seed": 11, "stream_seed": seed,
                                            "dropped": dropped}  # fmt: skip
        assert found[name]["t"] is None and found[name]["n"] == 19
        seeds.add(seed)
    assert len(seeds) == 2 and derive_seed(11, "look", 2) not in seeds  # not the looks' streams
    t = secondary_contrasts(blocks, 25, look=2, interval="t", status="secondary")
    assert t["adamw_minus_prewarmed"]["interval"] != found["adamw_minus_prewarmed"]["interval"]


@pytest.mark.parametrize(
    "kwargs, message",
    [({"interval": "percentile", "status": "secondary"}, "interval 'percentile'"),
     ({"interval": "t", "status": "confirmatory"}, "status 'confirmatory'"),
     ({"interval": "bootstrap_t", "status": "secondary"}, "the bootstrap needs"),
     ({"interval": "bootstrap_t", "status": "secondary", "bootstrap_resamples": 2000,
       "bootstrap_seed": 1.5}, "the bootstrap needs")],
)  # fmt: skip
def test_secondary_contrasts_refuse_an_unknown_interval_or_status(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        secondary_contrasts(secondary_blocks(), 25, look=1, **kwargs)


# ------------------------------------------------------------------ missing-label sensitivities


def sensitivity_blocks(n: int, *, a_missing: float, f_missing: float) -> list[dict]:
    """Blocks whose Δ alternates around 1 pp; AdamW's harm rises by ``a_missing`` and FA's by
    ``f_missing`` when their missing labels count as harmful."""
    blocks = []
    for i, d in enumerate([0.01 + 0.01 * (-1) ** i + 0.002 * (i % 5) for i in range(n)]):
        blocks.append({"block": 3000 + i, "complete": True, "supported": True, "delta": d,
                       "delta_missing_low": d - f_missing, "delta_missing_high": d + a_missing,
                       "delta_all_missing_harmful": d + a_missing - f_missing})  # fmt: skip
    return blocks


def test_the_missing_label_sensitivities_are_read_at_the_primarys_looks() -> None:
    settings = {**T, "se_floor": 0.0, "bias_allowance": 0.0}
    blocks = sensitivity_blocks(80, a_missing=0.04, f_missing=0.04)
    primary = looks(blocks, LEVELS, **settings)
    assert primary["stopping_look"] == 1 and primary["looks"][0]["readings"][0] == "equivalence"
    args = {"interval": "t", "se_floor": 0.0}
    (worst,) = sensitivity_looks(primary, blocks, "missing_labels", **args)
    first = primary["looks"][0]
    assert (worst["look"], worst["level"], worst["started"], worst["supported"]) == (
        1, LEVELS[0], first["started"], 24)  # fmt: skip
    t = stats.t.ppf(1 - (1 - LEVELS[0]) / 2, 23)
    lows = [b["delta_missing_low"] for b in blocks[:24]]
    highs = [b["delta_missing_high"] for b in blocks[:24]]
    expect = [
        statistics.fmean(lows) - t * statistics.stdev(lows) / math.sqrt(24),
        statistics.fmean(highs) + t * statistics.stdev(highs) / math.sqrt(24),
    ]
    assert worst["interval"] == pytest.approx(expect)
    # The worst case for the contrast spans ±4 pp more: no equivalence, no stop of its own …
    assert worst["readings"] == ["within_8"] and "stops" not in worst
    assert worst["primary_readings"] == first["readings"]
    # … while counting every missing label harmful in both arms cancels exactly here.
    (both,) = sensitivity_looks(primary, blocks, "all_missing_harmful", **args)
    assert both["interval"] == pytest.approx(first["interval"])
    assert both["readings"] == first["readings"]


def test_a_sensitivity_follows_the_primary_to_its_last_look_with_its_streams() -> None:
    settings = {**BOOTSTRAP, "se_floor": 0.0, "bias_allowance": 0.0,
                "equivalence_min_supported": 0, "direction_min_supported": 0}  # fmt: skip
    deltas = [0.15 * (-1) ** i for i in range(80)]  # wide: look 1 does not stop
    blocks = [{**b, "delta_missing_low": b["delta"] and b["delta"] - 0.01,
               "delta_missing_high": b["delta"], "delta_all_missing_harmful": b["delta"]}
              for b in blocks_of(deltas, unsupported={3, 9})]  # fmt: skip
    primary = study2_short.looks(blocks, LEVELS, (24, 50), **settings)
    assert [look["look"] for look in primary["looks"]] == [1, 2]
    found = sensitivity_looks(primary, blocks, "missing_labels", interval="bootstrap_t",
                              se_floor=0.0, bootstrap_resamples=2000,
                              bootstrap_seed=11)  # fmt: skip
    assert [(f["look"], f["started"], f["supported"]) for f in found] == [
        (look["look"], look["started"], look["supported"]) for look in primary["looks"]
    ]
    for look, sensitivity in zip(primary["looks"], found, strict=True):
        chosen = [b for b in blocks[: look["started"]] if b["supported"]]
        stream = derive_seed(11, "look", look["look"])
        low = bootstrap_t_interval([b["delta_missing_low"] for b in chosen], look["level"], 2000,
                                   np.random.default_rng(stream))  # fmt: skip
        high = bootstrap_t_interval([b["delta_missing_high"] for b in chosen], look["level"],
                                    2000, np.random.default_rng(stream))  # fmt: skip
        assert sensitivity["interval"] == [low[0], high[1]]
        assert sensitivity["high"]["interval"] == look["interval"]  # Δ itself: the same stream
    with pytest.raises(ValueError, match="look 1 read 24 supported blocks"):
        other = [{**blocks[0], "supported": False}, *blocks[1:]]  # not the primary's blocks
        sensitivity_looks(primary, other, "missing_labels", interval="bootstrap_t",
                          se_floor=0.0, bootstrap_resamples=2000, bootstrap_seed=11)  # fmt: skip


# ------------------------------------------------------------------ utility and validation

FROZEN = {ADAMW: [1e-5, 1.2e-5, 1.4e-5], FA: [1e-5, 1.2e-5, 1.4e-5]}


def validation_runs(ks=(5.0, 6.0, 7.5), mmlu=1.0, seeds=range(2100, 2112)) -> list[Trajectory]:
    return [traj(arm, 5, lr, s, k, mmlu=mmlu + 0.1 * (s % 3), factual=2.0)
            for s in seeds for arm in (ADAMW, FA)
            for lr, k in zip(FROZEN[arm], ks, strict=True)]  # fmt: skip


def supports_of(runs) -> dict:
    return {s: {arm: block_support([r for r in runs if (r.arm, r.seed) == (arm, s)], K8, W_MAX)
                for arm in (ADAMW, FA)} for s in sorted({r.seed for r in runs})}  # fmt: skip


def test_the_utility_family_is_bonferroni_over_members_and_two_looks() -> None:
    runs = validation_runs()
    found = utility_family(runs, supports_of(runs), THETA0)
    # Every block brackets K* with 1.2e-5 and 1.4e-5 in both arms: four members.
    assert [(e["arm"], e["rate"]) for e in found["family"]] == [
        (ADAMW, 1.2e-5),
        (ADAMW, 1.4e-5),
        (FA, 1.2e-5),
        (FA, 1.4e-5),
    ]
    assert found["alpha_per_bound"] == pytest.approx(0.05 / 8)
    losses = [1.0 + 0.1 * (s % 3) for s in range(2100, 2112)]
    t = stats.t.ppf(1 - 0.05 / 8, 11)
    upper = statistics.fmean(losses) + t * statistics.stdev(losses) / math.sqrt(12)
    assert found["family"][0]["mmlu_upper_pp"] == pytest.approx(upper)
    assert found["family"][0]["runs"] == 12 and found["passes"]
    near = validation_runs(mmlu=4.9)  # mean 5.0 pp: the upper bound exceeds 5
    assert not utility_family(near, supports_of(near), THETA0)["passes"]
    assert not utility_family([], {}, THETA0)["passes"]


def test_a_member_runs_over_every_block_where_it_completed() -> None:
    runs = validation_runs()
    # Block 2100's FA brackets with 1e-5 – 1.2e-5: 1e-5 joins the family, over all 12 blocks.
    runs = [r for r in runs if not (r.seed == 2100 and r.arm == FA)] + [
        traj(FA, 5, lr, 2100, k) for lr, k in zip(FROZEN[FA], (5.0, 7.0, 8.0), strict=True)
    ]
    found = utility_family(runs, supports_of(runs), THETA0)
    member = next(e for e in found["family"] if (e["arm"], e["rate"]) == (FA, 1e-5))
    assert member["runs"] == 12 and found["alpha_per_bound"] == pytest.approx(0.05 / 10)


def test_the_validation_verdict_on_numerics_k_and_utility() -> None:
    runs = validation_runs()
    accepted = {lr: {"accepted": True} for lr in FROZEN[FA]}
    found = validation_verdict(runs, FROZEN, accepted, THETA0, K8, W_MAX)
    assert found["passes"] and len(found["jointly_supported"]) == 12
    # Validation is a single reading: Bonferroni over the family alone (four members).
    assert found["utility"]["looks"] == 1
    assert found["utility"]["alpha_per_bound"] == pytest.approx(0.05 / 4)

    low = [traj(r.arm, 5, r.lr, r.seed, r.evaluations[0]["K_benign"] * 0.5, mmlu=1.0)
           if r.seed in (2100, 2101, 2102) and r.arm == ADAMW else r for r in runs]  # fmt: skip
    found = validation_verdict(low, FROZEN, accepted, THETA0, K8, W_MAX)
    assert found["reasons"] == ["9 of 12 blocks jointly supported, fewer than 10"]
    assert found["causes"][ADAMW] == {"cause": {"bracketing": 3}, "k_cause": {"bracketing": 3},
                                      "failure_kinds": {}}  # fmt: skip

    unstable = [r for r in runs if not (r.seed in (2104, 2105) and r.arm == ADAMW
                                        and r.lr == 1.4e-5)]  # fmt: skip
    unstable += [traj(ADAMW, 5, 1.4e-5, s, None, stop="k_above_max", steps=20)
                 for s in (2104, 2105)]  # fmt: skip
    found = validation_verdict(unstable, FROZEN, accepted, THETA0, K8, W_MAX)
    assert found["causes"][ADAMW] == {"cause": {"technical": 2}, "k_cause": {"bracketing": 2},
                                      "failure_kinds": {"instability": 2}}  # fmt: skip

    broken = [r for r in runs if not (r.seed == 2103 and r.arm == FA and r.lr == 1e-5)]
    broken.append(traj(FA, 5, 1e-5, 2103, None, stop="gradient_spike", steps=4))
    rejected = {**accepted, 1.4e-5: {"accepted": False}}
    found = validation_verdict(broken, FROZEN, rejected, THETA0, K8, W_MAX)
    assert not found["passes"] and len(found["jointly_supported"]) == 12  # 1e-5 is not needed
    assert found["unstable"] == ["frozen_adam_25-b5-1e-05-s2103"]
    assert found["reasons"][1] == "FA controls not accepted at [1.4e-05]"


def test_ambiguous_validation_records_leave_no_conclusion() -> None:
    runs = validation_runs()
    accepted = {lr: {"accepted": True} for lr in FROZEN[FA]}
    twice = [*runs, traj(ADAMW, 5, 1.2e-5, 2100, 6.0, name="again")]
    found = validation_verdict(twice, FROZEN, accepted, THETA0, K8, W_MAX)
    assert found["passes"] is None and "two runs" in found["problems"][0]
    extra = [*runs, traj(FA, 5, 1.3e-5, 2101, 6.4)]
    found = validation_verdict(extra, FROZEN, accepted, THETA0, K8, W_MAX)
    assert found["passes"] is None and "outside the frozen rates" in found["problems"][0]


def test_confirmation_keeps_bonferroni_over_the_family_and_two_looks() -> None:
    runs = validation_runs()
    found = utility_family(runs, supports_of(runs), THETA0)
    assert found["looks"] == 2 and found["alpha_per_bound"] == pytest.approx(0.05 / (4 * 2))


def test_the_analysis_bootstrap_interval_equals_the_simulation_s() -> None:
    """The decided interval is the one the simulation calibrated: same resamples, same quantiles,
    same bounds (analysis.study2_short_sim's bootstrap_t_quantiles with its bound formula)."""
    from zero_field_refusal.analysis import study2_short_sim as sim

    rng_data = np.random.default_rng(20261016)
    for n in (2, 3, 36, 60):
        for shape in ("normal", "skewed"):
            values = rng_data.normal(0.0, 0.05, n)
            if shape == "skewed":
                values = rng_data.gamma(2.0, 0.03, n) - 0.06
            for level in (0.9920, 0.9631):
                low, high, dropped = study2_short.bootstrap_t_interval(
                    values, level, 2000, np.random.default_rng(7)
                )
                probabilities = np.array([(1 - level) / 2, (1 + level) / 2])
                quantiles, sim_dropped = sim.bootstrap_t_quantiles(
                    values, probabilities, [0.0], 2000, np.random.default_rng(7)
                )
                assert dropped == int(sim_dropped[0])
                mean, se = values.mean(), values.std(ddof=1) / math.sqrt(n)
                if se <= sim.ZERO_SE:
                    assert (low, high) == (mean, mean)
                    continue
                q_low, q_high = quantiles[0]
                if not (np.isfinite(q_low) and np.isfinite(q_high)):
                    assert (low, high) == (-math.inf, math.inf)
                    continue
                assert (low, high) == (mean - q_high * se, mean - q_low * se)
