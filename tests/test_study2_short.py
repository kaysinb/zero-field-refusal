"""Study 2's redirected short-regime stage, design A (analysis/study2_short.py): selection,
support, the estimator and the looks, on synthetic ``Trajectory`` objects and label rows.

The state machine is driven through its campaigns against an analytic world: K25 = c × (rate /
1e-5)^1.5 × a seed factor, per arm and batch, utility losses growing with K; its numerical checks
get verdict records of the numerical-check tool's schema. Two tests replay the archived records
of the main checkout (K, numerics and utility only), skipped where they are absent: those made
before S1, and those up to the S2 repeat with its verdict."""

import dataclasses
import hashlib
import json
import math
import re
import shutil
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
    FALLBACK_CANDIDATES,
    LOOK_SIZES,
    NUMERICAL_SEEDS,
    PREWARMED,
    VALIDATION_SEEDS,
    block_difference,
    block_support,
    bootstrap_t_interval,
    candidate_order,
    component_readings,
    control_conformance,
    control_verdict,
    finite_bounds,
    harm,
    interpolate,
    max_t_interval,
    mechanism_analysis,
    numerical_cases,
    numerical_digest,
    numerical_policy,
    numerical_verdict,
    repair_rates,
    secondary_contrasts,
    seed_role,
    select_rates,
    selection,
    sensitivity_looks,
    symmetric_studentized,
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
                         **numerical_policy(check["U"])}}  # fmt: skip
    return {**record, **changes}


def commits(runs, commit=COMMIT) -> dict:
    """``run_facts`` as far as the selection reads them: each run's commit, on a clean
    checkout."""
    return {run.run_id: {"commit": commit, "dirty": False} for run in runs}


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


# The world's selection at (U, B) = (8, 20).
SELECTED20 = {ADAMW: [1.27e-5, 1.48e-5, 1.74e-5], FA: [1.59e-5, 1.86e-5, 2.17e-5],
              PREWARMED: [1.27e-5, 1.48e-5, 1.74e-5]}  # fmt: skip
PRIMARY20 = {arm: SELECTED20[arm] for arm in (ADAMW, FA)}
# The pre-warmed arm's first round at (8, 20): the union of the primary arms' targeted rates.
PREWARMED_FIRST = (1.27e-5, 1.48e-5, 1.59e-5, 1.74e-5, 1.86e-5, 2.17e-5)


def prewarmed_first_only(run):
    """A ``drive`` hook: the pre-warmed arm's discovery runs beyond its first round lose 6 pp
    MMLU, so that no later round of it (its search allowance's included) gives it a triple."""
    if run.arm == PREWARMED and run.seed in SEEDS and run.lr not in PREWARMED_FIRST:
        k = run.evaluations[0]["K_benign"]
        return traj(run.arm, run.batch_size, run.lr, run.seed, k, mmlu=6.0, factual=0.1 * k)
    return run


def verdict_path(rates, batch: int = 20, upper: float = 8.0) -> str:
    """Where the numerical-check tool keeps a selection's verdict, under $RUNS_DIR."""
    return f"numerical/b{batch}-u{upper:g}/verdicts/{numerical_digest(rates)}.json"


# The shape of a technical non-qualification, as S2's verdict at (8, 20) gave it for FA's
# 1.34e-5 and the pre-warmed arm's 2e-5 (a twin A differing from its archived run twice).
S2_TECHNICAL = ("case s2200: not qualified (technical), incomplete after its rerun: attempt 1: "
                "twin A differs from its archived run; the pairs [], not ['p1', 'p2', 'p3', "
                "'p4', 'reference']; attempt 2: twin A differs from its archived run; the pairs "
                "[], not ['p1', 'p2', 'p3', 'p4', 'reference']")  # fmt: skip


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
    # The search allowance's round never proposes an unusable rate: the midpoint's nearest
    # admissible rate inside the pair, 1.09e-5 (nearer 1.0954e-5 than 1.11e-5).
    repair = repair_rates(triple, records, 8.0, 0.2, unusable=[1.1e-5])
    assert repair["rates"] == [1.09e-5] and repair["unrepairable"] == []
    assert [(s["unusable"], s["rate"]) for s in repair["substituted"]] == [(1.1e-5, 1.09e-5)]
    assert "1.1e-05 is unusable at the batch, the nearest admissible rate is 1.09e-05" in (
        repair["reasons"][0])  # fmt: skip


def test_the_allowance_round_never_proposes_an_unusable_rate() -> None:
    """The search allowance's repair round keeps an
    unusable rate's K as data, but never proposes it. Here 1.09e-5 was run and found unusable
    by a numerical check, its median K exactly the target 5: re-targeting lands on it."""
    records = {
        1e-5: rec({2000: 4.0, 2001: 4.0, 2002: 4.0}),
        1.09e-5: rec({2000: 5.0, 2001: 5.0, 2002: 5.0}),
        1.2e-5: rec({2000: 6.3, 2001: 6.3, 2002: 6.3}),
        1.4e-5: rec({2000: 8.0, 2001: 8.0, 2002: 8.0}),
    }
    found = select_rates(records, 8.0, W_MAX, exclude=[1.09e-5])
    assert found["unmet"] == ["coverage"]
    # The discovery rounds' algorithm, unchanged: the rate already run, no repair.
    plain = repair_rates(found["best_unconstrained"], records, 8.0, W_MAX)
    assert plain["rates"] == [] and "1.09e-05, already run" in plain["unrepairable"][0]
    # The allowance's: the nearest admissible rate inside the bracketing pair (1e-5, 1.09e-5).
    repair = repair_rates(found["best_unconstrained"], records, 8.0, W_MAX, unusable=[1.09e-5])
    assert repair["rates"] == [1.08e-5] and repair["unrepairable"] == []
    assert repair["substituted"][0]["unusable"] == 1.09e-5
    # Every grid rate of the pair unusable or run: none admissible within the steps, no repair.
    taken = [round(1e-5 + k * 1e-7, 12) for k in range(1, 10)]
    blocked = repair_rates(found["best_unconstrained"], records, 8.0, W_MAX, unusable=taken)
    assert blocked["rates"] == [] and "no admissible rate" in blocked["unrepairable"][0]
    for proposal, unusable in ((repair, {1.09e-5}), (blocked, set(taken))):
        assert not set(proposal["rates"]) & unusable


# ------------------------------------------------------------------ candidates and costs


def test_the_candidate_order_is_b20_only() -> None:
    """B = 20 is the study's fixed scope. The candidates are (8, 20), (7, 20) then (6, 20) for
    the three arms, then (7, 20) and (6, 20) for the primary pair (amendment 5: the pair is not
    visited at U = 8); no B = 5 candidate is ever prescribed, even where B = 20 has none eligible
    and B = 5 would."""
    runs = grids()  # 100 s per run at B = 5, 150 s at B = 20
    order = candidate_order(runs, THETA0, SPC)
    assert [(e["U"], e["batch"]) for e in order] == [(8.0, 20), (7.0, 20), (6.0, 20)] == list(
        CANDIDATES)  # fmt: skip
    assert list(FALLBACK_CANDIDATES) == [(7.0, 20), (6.0, 20)]
    # The costs stay for the report. Six runs, one chunk of training responses, v₀'s 50 × B
    # preconditioner responses (1000 at B = 20).
    assert order[0]["seconds"] == pytest.approx(6 * 150.0 + SPC + SPC * 1000 / 1000)
    assert order[0]["preconditioner_seconds"] == pytest.approx(SPC)
    # The first step is still the joint discovery at (8, 20).
    step = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical={}, facts={})["next"]
    assert (step["campaign"], step["batch"], step["U"], step["three_arm"]) == (
        "discovery", 20, 8.0, True)  # fmt: skip
    # AdamW's K far below the targets at B = 20 (eligible at (8, 5) under the old order): every
    # campaign is at B = 20, the visits are B = 20's five, and the stage ends in the B = 20
    # feasibility report.
    scale = {**SCALE, (ADAMW, 20): 0.2}
    made_at = []

    def hook(run):
        made_at.append(run.batch_size)
        return run

    seen, result, _ = drive(grids(scale), scale=scale, hook=hook)
    assert set(made_at) == {20} and seen[-1] == "infeasible"
    assert [(a["U"], a["batch"], a["three_arm"]) for a in result["attempts"]] == [
        (8.0, 20, True), (7.0, 20, True), (6.0, 20, True), (7.0, 20, False),
        (6.0, 20, False)]  # fmt: skip
    assert result["next"]["scope"] == "B=20" and result["problems"] == []


def test_costs_are_reported_and_never_reorder_the_candidates() -> None:
    cheap5 = [traj(r.arm, r.batch_size, r.lr, r.seed, r.evaluations[0]["K_benign"],
                   seconds=10.0 if r.batch_size == 5 else 500.0) for r in grids()]  # fmt: skip
    assert [e["batch"] for e in candidate_order(cheap5, THETA0, SPC)] == [20, 20, 20]
    without = [r for r in grids() if (r.arm, r.batch_size) != (FA, 20)]
    order = candidate_order(without, THETA0, SPC)
    assert [e["batch"] for e in order] == [20, 20, 20]
    assert order[0]["seconds"] is None and order[0]["reasons"] == [f"{FA}: no usable grid rate "
                                                                   "at B=20"]  # fmt: skip


# ------------------------------------------------------------------ amendment 5's order


def test_amendment_5_never_visits_the_pair_at_u8_and_ends_after_u6() -> None:
    """The order in full: the three arms at (8, 20), (7, 20) and (6, 20), then the primary pair
    at (7, 20) and (6, 20), then ``infeasible``; the pair at (8, 20), which S3's records decide,
    is in neither the code's order nor the fixation record's, nor in the feasibility report."""
    fixed = study2_short.fixed_values()["candidates"]
    assert fixed["order"] == [{"U": 8.0, "B": 20}, {"U": 7.0, "B": 20}, {"U": 6.0, "B": 20}]
    assert fixed["fallback_order"] == [{"U": 7.0, "B": 20}, {"U": 6.0, "B": 20}]
    assert (fixed["arms"], fixed["fallback_arms"]) == (list(ARMS), [ADAMW, FA])
    record = study2_short.read_fixation(REPO_ROOT / "docs" / "study2-short-fixation.json")
    assert record["candidates"] == fixed
    scale = {**SCALE, (ADAMW, 20): 0.2}  # AdamW far below every target at B = 20
    seen, result, _ = drive(grids(scale), scale=scale)
    visits = [(a["U"], a["three_arm"]) for a in result["attempts"]]
    assert visits == [(8.0, True), (7.0, True), (6.0, True), (7.0, False), (6.0, False)]
    step = result["next"]
    assert (seen[-1], step["campaign"], step["scope"]) == ("infeasible", "infeasible", "B=20")
    assert [(c["U"], c["scope"]) for c in step["feasibility"]["candidates"]] == [
        (8.0, "three arms"), (7.0, "three arms"), (6.0, "three arms"), (7.0, "primary pair"),
        (6.0, "primary pair")]  # fmt: skip


def test_a_u6_candidate_has_its_k_star_its_bound_and_its_targets() -> None:
    """U = 6, the floor: K* = √30 ≈ 5.477, the twin bound 0.01 × K* ≈ 0.05477, the targets
    (5, K*, 6). AdamW's rates from 1.45e-5 up losing 6 pp MMLU on the validation seeds fail the
    three arms' validation at (8, 20) and at (7, 20); (6, 20) goes through its own discovery,
    its check under U = 6's bound and its three-arm validation, and is taken."""
    kstar = math.sqrt(30)
    assert study2_short.k_star(6.0) == pytest.approx(kstar) and kstar == pytest.approx(5.477, 1e-4)
    assert study2_short.targets(6.0) == (5.0, study2_short.k_star(6.0), 6.0)
    assert numerical_policy(6.0)["bound"] == pytest.approx(0.01 * kstar)
    assert numerical_policy(6.0)["bound"] == pytest.approx(0.05477, abs=1e-5)
    # A verdict under U = 7's bound is no verdict at U = 6.
    departures = study2_short._policy_departures(numerical_policy(7.0), 6.0)
    assert len(departures) == 1 and departures[0].startswith("bound ")

    def lossy(run):
        if run.seed in VALIDATION_SEEDS and run.arm == ADAMW and run.lr >= 1.45e-5:
            k = run.evaluations[0]["K_benign"]
            return traj(run.arm, run.batch_size, run.lr, run.seed, k, mmlu=6.0, factual=0.1 * k)
        return run

    seen, result, verdicts = drive(grids(), hook=lossy)
    assert seen == ["discovery", "numerical", "numerical-check", "validation"] + [
        "numerical", "numerical-check", "validation"] + [
        "discovery", "numerical", "numerical-check", "validation", "done"]  # fmt: skip
    assert [(a["U"], a["outcome"], a["failed_at"]) for a in result["attempts"]] == [
        (8.0, "failed", "validation"), (7.0, "failed", "validation"),
        (6.0, "eligible", None)]  # fmt: skip
    done = result["next"]
    assert (done["batch"], done["U"], done["three_arm"]) == (20, 6.0, True)
    assert done["K_star"] == pytest.approx(kstar)
    selected6 = {ADAMW: [1.27e-5, 1.35e-5, 1.43e-5], FA: [1.5e-5, 1.59e-5, 1.86e-5],
                 PREWARMED: [1.27e-5, 1.35e-5, 1.43e-5]}  # fmt: skip
    assert done["rates"] == selected6
    (check,) = done["numerical"]
    assert check["verdict"] == verdict_path(selected6, upper=6.0) in verdicts
    assert check["policy"]["bound"] == pytest.approx(0.01 * kstar)
    assert result["rounds"][f"{PREWARMED}/B=20/U=6"]["first"]["proposal"]["joint"]
    assert done["validation"]["passes"] and done["validation"]["arms"] == list(ARMS)


def test_the_pair_after_a_visit_stopped_before_validation_runs_its_own() -> None:
    """Amendment 5, as amendment 4 had it: the pair at a U reuses every record of the three
    arms' visit there; when that visit stopped before its validation (here the pre-warmed arm
    without a triple at discovery, at every U), the pair runs what is missing: its numerical
    check and its own validation, 72 runs (two arms, three rates, twelve seeds)."""
    scale = {**SCALE, (PREWARMED, 20): 0.3, (PREWARMED, 5): 0.3}
    runs = grids(scale)
    seen, result, _ = drive(runs, scale=scale, until="validation")
    assert [(a["U"], a["three_arm"], a["failed_at"]) for a in result["attempts"]] == [
        (8.0, True, "discovery"), (7.0, True, "discovery"), (6.0, True, "discovery"),
        (7.0, False, None)]  # fmt: skip
    assert not [r for r in runs if r.seed in VALIDATION_SEEDS]
    assert seen[-3:] == ["numerical", "numerical-check", "validation"]
    step = result["next"]
    assert (step["campaign"], step["U"], step["three_arm"]) == ("validation", 7.0, False)
    assert len(step["runs"]) == 72 and {r["arm"] for r in step["runs"]} == {ADAMW, FA}
    assert {r["seed"] for r in step["runs"]} == set(VALIDATION_SEEDS)
    assert {(r["arm"], r["lr"]) for r in step["runs"]} == {
        (arm, lr) for arm in (ADAMW, FA) for lr in step["rates"][arm]}  # fmt: skip


def test_a_mark_at_the_pinned_pairs_own_check_stops_it_without_repair() -> None:
    """Amendment 5: the pinned pair has no repair. The pre-warmed arm has no triple at any U,
    so the three arms' visits stop at discovery and the pair at (7, 20) runs its own check of
    the three arms' U = 7 triples. That check marks AdamW's 1.48e-5: failing, the pair fails on
    AdamW; not qualified, it blocks on it. Never a re-solve onto 1.43e-5, a rate U = 6's round
    ran, nor a repair round."""
    scale = {**SCALE, (PREWARMED, 20): 0.3, (PREWARMED, 5): 0.3}
    for mark, outcome in (("failing", "failed"), ("not_qualified", "blocked")):

        def judge(check, mark=mark):
            if "adamw_prewarmed" in check["rates"] or check["U"] != 7:
                return verdict_of(check)
            return verdict_of(check, {("adamw", 1.48e-5): mark})

        runs = grids(scale)
        seen, result, verdicts = drive(runs, scale=scale, judge=judge)
        three, pair = result["attempts"][:3], result["attempts"][3]
        assert [(a["U"], a["three_arm"], a["failed_at"]) for a in three] == [
            (8.0, True, "discovery"),
            (7.0, True, "discovery"),
            (6.0, True, "discovery"),
        ]
        assert 1.43e-5 in three[2]["arms"][ADAMW]["selections"][-1]["rates"]  # U = 6's round
        assert (pair["U"], pair["three_arm"], pair["outcome"]) == (7.0, False, outcome)
        assert pair["reasons"][0].startswith("unusable rates in the pinned triples, with no "
                                             "repair: ['adamw_25 1.48e-05")  # fmt: skip
        assert [s["rates"] for s in pair["arms"][ADAMW]["selections"]] == [
            [1.27e-5, 1.48e-5, 1.5e-5]]  # fmt: skip
        assert pair["allowance"] == {} and len(pair["numerical"]) == 1
        assert f"{ADAMW}/B=20/U=7" not in result["rounds"] or study2_short.ALLOWANCE not in result[
            "rounds"][f"{ADAMW}/B=20/U=7"]  # fmt: skip
        assert [v for v in verdicts if v.startswith("numerical/b20-u7/")] == [
            pair["numerical"][0]["verdict"]]  # fmt: skip
        if outcome == "failed":
            assert pair["failed_arms"] == [ADAMW]  # the order moves on to the pair at U = 6
            done = result["next"]
            assert (done["campaign"], done["U"], done["three_arm"]) == ("done", 6.0, False)
            assert done["rates"][ADAMW] == three[2]["arms"][ADAMW]["selections"][-1]["rates"]
        else:
            step = result["next"]
            assert (step["blocked"], step["U"], step["arms"]) == ("technical", 7.0, [ADAMW])


def test_the_pair_after_a_full_three_arm_validation_only_recomputes() -> None:
    """Amendment 5: the pair's rates at a U are the three arms' triples at that U. The
    pre-warmed arm unsupported on the validation seeds fails the three arms' validation at
    (8, 20), (7, 20) and (6, 20); U = 6's discovery runs AdamW at 1.35e-5 and 1.43e-5, the
    latter nearer U = 7's K* than the three arms' 1.48e-5 there. The pair at (7, 20) keeps
    the three arms' U = 7 triple all the same, reads their check there and recomputes their
    validation's verdict on the same runs: no new check, no new run."""

    def unsupported(run):  # the pre-warmed arm never brackets K* on the validation seeds
        if run.seed in VALIDATION_SEEDS and run.arm == PREWARMED:
            return traj(run.arm, run.batch_size, run.lr, run.seed, 1.0, mmlu=0.3, factual=0.1)
        return run

    runs = grids()
    seen, result, verdicts = drive(runs, hook=unsupported)
    at8, at7, at6, pair = result["attempts"]
    assert [(a["U"], a["three_arm"], a["failed_at"]) for a in (at8, at7, at6, pair)] == [
        (8.0, True, "validation"), (7.0, True, "validation"), (6.0, True, "validation"),
        (7.0, False, None)]  # fmt: skip
    assert at6["arms"][ADAMW]["selections"][-1]["rates"] == [1.27e-5, 1.35e-5, 1.43e-5]
    # Nothing is made after U = 6's validation: the pair asks for no campaign.
    assert seen[-3:] == ["numerical-check", "validation", "done"]
    done = result["next"]
    assert (done["campaign"], done["U"], done["three_arm"]) == ("done", 7.0, False)
    triple7 = {arm: at7["arms"][arm]["selections"][-1]["rates"] for arm in (ADAMW, FA)}
    assert triple7[ADAMW] == [1.27e-5, 1.48e-5, 1.5e-5]
    assert {arm: done["rates"][arm] for arm in (ADAMW, FA)} == triple7
    assert all(s["pinned"] == {"U": 7.0, "batch": 20}
               for arm in (ADAMW, FA) for s in pair["arms"][arm]["selections"])  # fmt: skip
    assert [e["verdict"] for e in pair["numerical"]] == [at7["numerical"][0]["verdict"]]
    assert len(verdicts) == 3  # the three arms' checks at U = 8, 7 and 6; none for the pair
    recomputed = study2_short.pair_recomputation(at7, runs, THETA0, W_MAX)["verdict"]
    assert {k: done["validation"][k] for k in ("passes", "jointly_supported", "utility")} == {
        k: recomputed[k] for k in ("passes", "jointly_supported", "utility")}  # fmt: skip
    assert done["validation"]["passes"] and done["validation"]["arms"] == [ADAMW, FA]


def test_the_pair_recomputation_reads_the_three_arm_validation_runs() -> None:
    """Amendment 5 decides the pair at U = 8 by recomputing its verdict on the three-arm
    validation's runs: ``pair_recomputation`` gives the primary-arm verdict at the rates and
    with the FA controls of the check that qualified them, K* of the attempt's U and w_max,
    over six utility bounds; it refuses an attempt without a three-arm validation."""

    def unsupported(run):  # the pre-warmed arm never brackets K* on the validation seeds
        if run.seed in VALIDATION_SEEDS and run.arm == PREWARMED:
            return traj(run.arm, run.batch_size, run.lr, run.seed, 1.0, mmlu=0.3, factual=0.1)
        return run

    runs = grids()
    _, result, _ = drive(runs, hook=unsupported)
    at8 = result["attempts"][0]
    assert (at8["U"], at8["three_arm"], at8["failed_at"]) == (8.0, True, "validation")
    pair = study2_short.pair_recomputation(at8, runs, THETA0, W_MAX)
    assert (pair["batch"], pair["U"], pair["K_star"]) == (20, 8.0, pytest.approx(K8))
    assert pair["rates"] == PRIMARY20 and pair["check"] == verdict_path(SELECTED20)
    assert set(pair["controls"]) == {f"{lr:g}" for lr in PRIMARY20[FA]}
    verdict = pair["verdict"]
    assert verdict["passes"] and verdict["arms"] == [ADAMW, FA] and verdict["problems"] == []
    assert len(verdict["jointly_supported"]) == 12 and verdict["exhausted"] == {}
    assert len(verdict["utility"]["family"]) == 6
    assert verdict["utility"]["alpha_per_bound"] == pytest.approx(0.05 / 6)
    assert at8["validation"]["passes"] is False  # no block jointly supported for three arms
    with pytest.raises(ValueError, match="no three-arm validation at U=7"):
        study2_short.pair_recomputation(result["attempts"][-1], runs, THETA0, W_MAX)


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
    assert [e["verdict"] for e in done["numerical"]] == [verdict_path(SELECTED20)]
    assert list(verdicts) == [verdict_path(SELECTED20)]
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
    resolved = {**SELECTED20, ADAMW: [1.27e-5, 1.5e-5, 1.74e-5]}
    # Both verdicts are kept, each under its selection's digest, in the check's one directory.
    assert list(verdicts) == [verdict_path(SELECTED20), verdict_path(resolved)]
    assert [c["out"] for c in asked] == ["numerical/b20-u8", "numerical/b20-u8"]
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
    assert second["check"] == numerical_digest(resolved) and second["unusable"] == []
    assert attempt["arms"][ADAMW]["selections"][-1]["excluded"] == [1.48e-5]
    assert result["next"]["rates"][ADAMW] == [1.27e-5, 1.5e-5, 1.74e-5]
    replacement = [r for r in runs if r.seed in NUMERICAL_SEEDS and r.lr == 1.5e-5]
    assert [(r.arm, r.seed) for r in replacement] == [(ADAMW, 2200)]
    assert attempt["allowance"][ADAMW]["check"] == numerical_digest(SELECTED20)
    assert attempt["allowance"][ADAMW]["resolved"] == [1.27e-5, 1.5e-5, 1.74e-5]
    assert attempt["allowance"][ADAMW]["repair"] is None  # the re-solve found a triple
    # The allowance is spent: a second unusable rate at the candidate stops AdamW. Failing, a
    # scientific failure, it fails the candidate; the next candidate (7, 20) starts without both
    # rates (not qualified instead, the stage is blocked:
    # ``test_a_technical_rate_spends_the_allowance_then_blocks``).
    runs = grids()
    judge = lambda check: verdict_of(check, {("adamw", 1.48e-5): "failing",  # noqa: E731
                                             ("adamw", 1.5e-5): "failing"})  # fmt: skip
    seen, result, _ = drive(runs, judge=judge)
    failed, at7 = result["attempts"][:2]
    assert (failed["outcome"], failed["failed_at"], failed["failed_arms"]) == (
        "failed", "numerical", [ADAMW])  # fmt: skip
    assert failed["reasons"] == ["unusable rates after the search allowance: ['adamw_25 1.5e-05: "
                                 "failing: a pair exceeding the bound']"]  # fmt: skip
    assert (at7["batch"], at7["U"]) == (20, 7.0)
    assert sorted(at7["excluded"][ADAMW]) == [1.48e-5, 1.5e-5]
    assert at7["excluded_kinds"][ADAMW] == {1.48e-5: "scientific", 1.5e-5: "scientific"}
    assert seen[:6] == ["discovery", "numerical", "numerical-check", "numerical",
                        "numerical-check", "discovery"]  # fmt: skip
    assert (result["next"]["campaign"], result["next"]["U"]) == ("done", 7.0)


def test_two_verdicts_at_8_20_a_failing_fa_rate_and_the_resolved_selection(tmp_path) -> None:
    """The replay over the tool's verdict files at (8, 20): the first verdict marks FA's lowest
    rate failing; FA is re-solved once; the second verdict, of the re-solved selection, under its
    own digest beside the first, qualifies it."""
    runs = grids()
    failing = {("adam_frozen", 1.59e-5): "failing"}
    reason = {"adam_frozen|1.59e-05": ["case s2200: p1 exceeding (max D_twin 0.081)"]}
    _, _, verdicts = drive(runs, judge=lambda c: verdict_of(c, failing, reasons=reason),
                           until="validation")  # fmt: skip
    resolved = {**SELECTED20, FA: [1.5e-5, 1.86e-5, 2.17e-5]}
    assert list(verdicts) == [verdict_path(SELECTED20), verdict_path(resolved)]
    # The replacement is not FA's U-rate: one case, on 2200; its control is asked in the check.
    assert [(r.arm, r.lr, r.seed) for r in runs if r.seed in NUMERICAL_SEEDS
            and r.lr == 1.5e-5] == [(FA, 1.5e-5, 2200)]  # fmt: skip
    assert "1.5e-05" in verdicts[verdict_path(resolved)]["controls"]
    root = tmp_path / "runs"
    for path, record in verdicts.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(json.dumps(record))
    (root / "numerical" / "b20-u8" / "verdict.json").write_text("{}")  # a copy, never read
    found = study2_short.read_numerical(root)
    assert sorted(found) == sorted(verdicts)
    result = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=found,
                       facts=commits(runs), reuse=reusable)  # fmt: skip
    assert result["next"]["campaign"] == "validation" and result["problems"] == []
    assert result["next"]["rates"][FA] == resolved[FA]
    first, second = result["attempts"][0]["numerical"]
    assert first["verdict"] == verdict_path(SELECTED20)
    assert first["unusable"][0]["reasons"] == reason["adam_frozen|1.59e-05"]
    assert second["verdict"] == verdict_path(resolved) and second["unusable"] == []
    assert result["unread_verdicts"] == []
    # A verdict the replay did not read that marks a rate unusable at the batch: no conclusion.
    other = {**SELECTED20, ADAMW: [1.27e-5, 1.5e-5, 1.74e-5]}
    stray = verdict_of(study2_short._check_request(
        {"name": numerical_digest(other), "dir": "numerical/b20-u7",
         "path": verdict_path(other, upper=7.0)}, 20, 7.0, other, {COMMIT}, w_max=W_MAX),
        {("adamw", 1.5e-5): "not_qualified"})  # fmt: skip
    found = {**verdicts, verdict_path(other, upper=7.0): stray}
    result = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=found,
                       facts=commits(runs), reuse=reusable)  # fmt: skip
    assert result["unread_verdicts"] == [verdict_path(other, upper=7.0)]
    assert "a verdict the selection did not read marks ['adamw|1.5e-05']" in result["problems"][0]
    # No campaign on such records: the step waits as ``pending``.
    assert result["next"] == {"campaign": "incomplete"}
    assert result["pending"]["campaign"] == "validation"
    # A verdict whose name is not the digest of its rates: malformed, no conclusion.
    misnamed = {verdict_path(SELECTED20): verdicts[verdict_path(resolved)]}
    result = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=misnamed,
                       facts=commits(runs), reuse=reusable)  # fmt: skip
    assert result["next"]["campaign"] == "incomplete"
    assert "is not the digest" in result["problems"][0]
    # An instability stop of a numerical run is failing, with its reason.
    stop = {"adamw|1.48e-05": ["instability: case s2200 stopped by k_above_max at step 9"]}
    unstable = verdict_of(study2_short._check_request(
        {"name": numerical_digest(SELECTED20), "dir": "numerical/b20-u8",
         "path": verdict_path(SELECTED20)}, 20, 8.0, SELECTED20, {COMMIT}, w_max=W_MAX),
        {("adamw", 1.48e-5): "failing"}, reasons=stop)  # fmt: skip
    entry = numerical_verdict(unstable, 20, 8.0, SELECTED20, COMMIT, reusable)["unusable"][0]
    assert entry["why"] == "failing: its numerical run stopped by an instability rule"
    assert numerical_digest(SELECTED20) == hashlib.sha256("\n".join(sorted(
        f"{study2_short.OPTIMIZERS[arm]}|{lr:g}" for arm, lrs in SELECTED20.items()
        for lr in lrs)).encode()).hexdigest()[:12]  # fmt: skip


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
    second = verdicts[verdict_path({**SELECTED20, PREWARMED: [1.27e-5, 1.59e-5, 1.74e-5]})]
    assert second["seeds"]["adamw_prewarmed|1.74e-05"] == [2200, 2201, 2202]


def test_an_unusable_fa_control_is_like_a_failing_rate() -> None:
    # FA's U-rate's control rejected: FA has no triple at U = 8 without 2.17e-5, so its
    # allowance's repair round re-targets U; the round never proposes an unusable rate, and
    # proposes 2.18e-5 instead.
    def judge(rejected):
        return lambda check: verdict_of(
            check,
            controls={lr: "rejected" for lr in rejected if lr in check["rates"]["adam_frozen"]},
        )

    runs = grids()
    seen, result, _ = drive(runs, judge=judge([2.17e-5]))
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
    repair = result["rounds"][f"{FA}/B=20/U=8"]["allowance_repair"]
    assert repair["new_rates"] == [2.18e-5]
    assert [(s["unusable"], s["rate"]) for s in repair["proposal"]["substituted"]] == [
        (2.17e-5, 2.18e-5)]  # fmt: skip
    assert seen == ["discovery", "numerical", "numerical-check", "repair", "numerical",
                    "numerical-check", "validation", "done"]  # fmt: skip
    assert result["next"]["rates"][FA] == [1.59e-5, 1.86e-5, 2.18e-5]
    # The replacement's control rejected too: FA's allowance is spent, a scientific failure; the
    # candidate fails, and both rates stay out at U = 7.
    runs = grids()
    seen, result, _ = drive(runs, judge=judge([2.17e-5, 2.18e-5]))
    first, at7 = result["attempts"][:2]
    assert first["failed_at"] == "numerical" and first["failed_arms"] == [FA]
    assert at7["excluded"][FA].keys() == {2.17e-5, 2.18e-5}
    assert at7["excluded_kinds"][FA] == {2.17e-5: "scientific", 2.18e-5: "scientific"}


def test_old_controls_are_reported_not_used() -> None:
    # Amendment 3's history at B = 5: its targeted round at (8, 5) and three FA controls, two of
    # them rejected. B = 20 fails here (AdamW's K stays far below the targets); B = 5 is never
    # visited, so the history stays unreached and decides nothing.
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
    assert result["next"]["campaign"] == "infeasible" and result["next"]["scope"] == "B=20"
    assert {a["batch"] for a in result["attempts"]} == {20}
    assert not [key for key in result["rounds"] if "/B=5/" in key]
    assert result["not_reached"] == first["not_reached"] and result["problems"] == []
    assert [(c["lr"], c["accepted"], c["used"]) for c in result["amendment3_controls"]] == [
        (1.27e-5, True, False), (1.49e-5, False, False), (1.74e-5, False, False)]  # fmt: skip
    assert "discovery" in seen and all(step != "numerical" for step in seen)


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
            for a in attempts[:3]] == [(u, b, True, [PREWARMED], "discovery")
                                       for u, b in CANDIDATES]  # fmt: skip
    assert [r["kind"] for r in attempts[0]["arms"][PREWARMED]["rounds"]] == [
        "first", "targeted", "repair"]  # fmt: skip
    # Then the first candidate of the fallback's order eligible for the primary arms alone is
    # taken: (7, 20), the pair never being visited at U = 8 (amendment 5).
    fallback = attempts[3]
    assert len(attempts) == 4
    assert (fallback["U"], fallback["batch"], fallback["three_arm"]) == (7.0, 20, False)
    assert set(fallback["arms"]) == {ADAMW, FA} and fallback["outcome"] == "eligible"
    done = result["next"]
    assert done["campaign"] == "done" and (done["batch"], done["U"]) == (20, 7.0)
    assert done["prewarmed_dropped"] and not done["three_arm"] and done["rates"][PREWARMED] is None
    assert done["validation"]["arms"] == [ADAMW, FA]  # amendment 3's primary-arm validation
    assert seen[-4:] == ["numerical", "numerical-check", "validation", "done"]
    # The three arms' U = 7 triples (amendment 5), not U = 6's 1.43e-5 for AdamW
    primary7 = {ADAMW: [1.27e-5, 1.48e-5, 1.5e-5], FA: [1.59e-5, 1.86e-5, 2e-5]}
    assert {arm: done["rates"][arm] for arm in (ADAMW, FA)} == primary7
    assert list(verdicts) == [verdict_path(primary7, upper=7.0)]  # the primary arms' own check
    assert {(r.arm, r.seed) for r in runs if r.seed in NUMERICAL_SEEDS} <= {
        (arm, s) for arm in (ADAMW, FA) for s in NUMERICAL_SEEDS}  # fmt: skip
    assert not [r for r in runs if r.arm == PREWARMED and r.seed in VALIDATION_SEEDS]


@pytest.fixture
def amendment4_order(monkeypatch):
    """Amendment 4's order, (8, 20) then (7, 20) for the three arms and then for the primary
    pair. The fallback's mechanics at a candidate (the pair after the three arms' visits there:
    every record reused, allowances carried, technical stops kept, a candidate skipped after its
    primary arms failed their numerical gate) do not depend on the order; the tests that pin
    them were written on this one, where the pair's first visit, (8, 20), follows the three
    arms' at (8, 20) and (7, 20). Amendment 5's order is pinned by the tests above and below."""
    monkeypatch.setattr(study2_short, "CANDIDATES", ((8.0, 20), (7.0, 20)))
    monkeypatch.setattr(study2_short, "FALLBACK_CANDIDATES", ((8.0, 20), (7.0, 20)))


@pytest.mark.usefixtures("amendment4_order")
def test_primary_only_fallback_end_to_end() -> None:
    # The pre-warmed arm fails everywhere: at (8, 20) in the combined numerical verdict (each of
    # its rates failing, and its allowance's repair rates out of their utility bounds: no triple
    # is left, a scientific stop), at (7, 20) in selection (its rates unusable at B = 20, its
    # rounds no help). The primary arms qualify at (8, 20). (Were some of them not qualified
    # instead, the technical marks alone would leave it no triple at (7, 20), and the stage
    # would be blocked there: ``test_a_technical_exclusion_never_decides_a_later_candidate``.)
    def judge(check):
        pre = check["rates"].get("adamw_prewarmed", [])
        return verdict_of(check, {("adamw_prewarmed", lr): "failing" for lr in pre})

    scale = {**SCALE, (PREWARMED, 5): 0.3}
    runs = grids(scale)
    seen, result, verdicts = drive(runs, scale=scale, judge=judge, until="validation",
                                   hook=prewarmed_first_only)  # fmt: skip
    attempts = result["attempts"]
    assert [(a["U"], a["batch"], a["failed_arms"], a["failed_at"]) for a in attempts[:2]] == [
        (8.0, 20, [PREWARMED], "numerical"), (7.0, 20, [PREWARMED], "discovery")]  # fmt: skip
    assert "without a feasible triple once the unusable rates" in attempts[0]["reasons"][0]
    assert all(u["arm"] == PREWARMED for u in attempts[0]["numerical"][0]["unusable"])
    # The search allowance: a re-solve, then a repair round (proposing no unusable rate) whose
    # rates are out of their utility bounds.
    rounds = attempts[0]["arms"][PREWARMED]["rounds"]
    assert [r["kind"] for r in rounds] == ["first", "allowance_repair"]
    assert not set(rounds[1]["new_rates"]) & set(SELECTED20[PREWARMED])
    # The fallback reads the same combined verdict for the primary arms and the FA controls
    # only: its pre-warmed entries do not block it, and no new check is asked for.
    fallback = attempts[2]
    assert (fallback["three_arm"], fallback["numerical"][0]["check"]) == (
        False, numerical_digest(PRIMARY20))  # fmt: skip
    assert fallback["resolve_carried"] is False  # pass 1 re-solved the pre-warmed arm only
    assert fallback["numerical"][0]["verdict"] == verdict_path(SELECTED20)  # the combined one
    assert fallback["numerical"][0]["state"] == "complete"
    assert fallback["numerical"][0]["unusable"] == []
    step = result["next"]
    assert (step["campaign"], step["batch"], step["U"], step["three_arm"]) == (
        "validation", 20, 8.0, False)  # fmt: skip
    assert step["prewarmed_dropped"] and set(step["rates"]) == {ADAMW, FA}
    assert {r["arm"] for r in step["runs"]} == {ADAMW, FA} and len(step["runs"]) == 2 * 3 * 12
    seen, result, verdicts = drive(runs, scale=scale, judge=judge, verdicts=verdicts,
                                   hook=prewarmed_first_only)  # fmt: skip
    assert seen == ["validation", "done"] and list(verdicts) == [verdict_path(SELECTED20)]
    done = result["next"]
    assert done["prewarmed_dropped"] is True and done["three_arm"] is False
    assert done["rates"] == {ADAMW: [1.27e-5, 1.48e-5, 1.74e-5], FA: [1.59e-5, 1.86e-5, 2.17e-5],
                             PREWARMED: None}  # fmt: skip
    assert done["validation"]["passes"] and done["validation"]["arms"] == [ADAMW, FA]
    # Nothing of the pre-warmed arm was called for after it was dropped.
    assert not [r for r in runs if r.arm == PREWARMED and r.seed in VALIDATION_SEEDS]
    assert result["problems"] == []


@pytest.mark.usefixtures("amendment4_order")
def test_the_fallback_carries_a_primary_re_solve_of_pass_1() -> None:
    """Pass 1 re-solved AdamW at (8, 20) (1.48e-5
    failing), spending its search allowance there, and failed there on the pre-warmed arm (its
    rates failing, its allowance's repair rates out of their utility bounds). The fallback's
    check of AdamW's replacement 1.5e-5 finds it failing: AdamW's allowance at the candidate
    stays spent, so (8, 20) fails and the next candidate eligible for the primary arms is
    taken."""
    scale = {**SCALE, (PREWARMED, 5): 0.3}
    runs = grids(scale)

    def judge(check):
        rates = check["rates"]
        changes = {("adamw_prewarmed", lr): "failing" for lr in rates.get("adamw_prewarmed", [])}
        if "adamw_prewarmed" in rates and 1.48e-5 in rates["adamw"]:
            changes[("adamw", 1.48e-5)] = "failing"  # pass 1's first check
        if "adamw_prewarmed" not in rates and 1.5e-5 in rates["adamw"]:
            changes[("adamw", 1.5e-5)] = "failing"  # the fallback's: the replacement fails
        return verdict_of(check, changes)

    seen, result, verdicts = drive(runs, scale=scale, judge=judge, hook=prewarmed_first_only)
    first, *_, fallback, taken = result["attempts"]
    assert (first["U"], first["batch"], first["failed_at"], first["failed_arms"]) == (
        8.0, 20, "numerical", [PREWARMED])  # fmt: skip
    assert first["arms"][ADAMW]["selections"][-1]["rates"] == [1.27e-5, 1.5e-5, 1.74e-5]
    assert (fallback["U"], fallback["batch"], fallback["three_arm"]) == (8.0, 20, False)
    assert fallback["resolve_carried"] is True and fallback["allowance_carried"] == [ADAMW]
    assert fallback["allowance"][ADAMW]["carried"] and fallback["allowance"][ADAMW]["three_arm"]
    assert (fallback["failed_at"], fallback["failed_arms"]) == ("numerical", [ADAMW])
    assert fallback["reasons"] == [  # the pinned pair has no repair (amendment 5)
        "unusable rates in the pinned triples, with no repair: "
        "['adamw_25 1.5e-05: failing: a pair exceeding the bound']"
    ]
    assert len(fallback["numerical"]) == 1  # no second re-solve at (8, 20)
    assert len(fallback["arms"][ADAMW]["selections"]) == 1
    done = result["next"]
    assert (done["campaign"], done["batch"], done["U"], done["three_arm"]) == (
        "done", 20, 7.0, False)  # fmt: skip
    assert taken["excluded"][ADAMW].keys() == {1.48e-5, 1.5e-5}
    assert done["rates"][ADAMW] == [1.27e-5, 1.42e-5, 1.59e-5]


@pytest.mark.usefixtures("amendment4_order")
def test_the_fallback_gives_a_carried_candidate_no_new_round() -> None:
    """Pass 1 re-solved AdamW at (8, 20), spending its allowance there, and failed the candidate
    at validation (the pre-warmed arm unsupported); (7, 20)'s check then found AdamW's 1.27e-5
    failing at B = 20, and the replacement its allowance proposed, 1.26e-5, failing too. In the
    fallback, AdamW has no triple at (8, 20) without its unusable rates; a new round would be a
    second allowance, so the candidate fails without one (its rates failing: a scientific stop).
    Its rates being the three arms' triples there (amendment 5), the pinned triple holding
    1.27e-5 fails it without a check. (7, 20)'s primary pair is not visited (its primary arm
    failed its numerical gate), and the stage ends in the B = 20 feasibility report: never
    B = 5."""

    def unsupported(run):  # the pre-warmed arm never brackets K* on the validation seeds
        if run.seed in VALIDATION_SEEDS and run.arm == PREWARMED:
            return traj(run.arm, run.batch_size, run.lr, run.seed, 1.0, mmlu=0.3, factual=0.1)
        return run

    def judge(check):
        adamw, changes = check["rates"].get("adamw", []), {}
        if (check["batch"], check["U"]) == (20, 8.0) and 1.48e-5 in adamw:
            changes[("adamw", 1.48e-5)] = "failing"
        if (check["batch"], check["U"]) == (20, 7.0):  # 1.27e-5 and its replacement
            changes.update({("adamw", lr): "failing" for lr in adamw if lr < 1.3e-5})
        return verdict_of(check, changes)

    runs = grids()
    seen, result, _ = drive(runs, judge=judge, hook=unsupported)
    attempts = result["attempts"]
    assert [(a["U"], a["batch"], a["three_arm"], a["failed_at"]) for a in attempts] == [
        (8.0, 20, True, "validation"), (7.0, 20, True, "numerical"),
        (8.0, 20, False, "numerical")]  # fmt: skip
    assert attempts[1]["allowance"][ADAMW]["repair"]["rates"] == [1.17e-5, 1.26e-5]
    fallback = attempts[2]
    assert fallback["resolve_carried"] and fallback["failed_arms"] == [ADAMW]
    assert fallback["reasons"][0].startswith("the pinned triples hold rates found unusable at the "
                                             "batch since the three arms' visit: ['adamw_25 "
                                             "1.27e-05: failing")  # fmt: skip
    assert fallback["numerical"] == []
    assert fallback["excluded"][ADAMW].keys() == {1.26e-5, 1.27e-5, 1.48e-5}
    assert f"{ADAMW}/B=20/U=8" in result["rounds"]
    assert "repair" not in result["rounds"][f"{ADAMW}/B=20/U=8"]  # no new round was made
    assert result["next"]["campaign"] == "infeasible" and result["next"]["scope"] == "B=20"
    skipped = result["next"]["feasibility"]["candidates"][3]
    assert (skipped["U"], skipped["scope"], skipped["visited"]) == (7.0, "primary pair", False)


@pytest.mark.usefixtures("amendment4_order")
def test_the_fallback_never_reselects_over_rates_run_since() -> None:
    """Amendment 5: the pair's rates at a candidate are the three arms' triples there, never a
    selection over rates run since. Here AdamW has no triple at (8, 20) in pass 1 (its K* rates
    unstable on seed 2002, a coverage miss its repair cannot fix); U = 7's round at B = 20 adds
    1.42e-5, which would give it one at (8, 20) (amendment 4's re-selection). The pair at
    (8, 20) fails all the same, and the pair at (7, 20) is taken with the three arms' triples
    there."""
    scale = {**SCALE, (PREWARMED, 20): 0.3, (PREWARMED, 5): 0.3}
    unstable = {(ADAMW, 20, 1.5e-5, 2002), (ADAMW, 20, 1.48e-5, 2002)}

    def hook(run):
        if (run.arm, run.batch_size, run.lr, run.seed) in unstable:
            return traj(run.arm, run.batch_size, run.lr, run.seed, None, stop="k_above_max",
                        steps=15)  # fmt: skip
        return run

    runs = [hook(run) for run in grids(scale)]
    seen, result, _ = drive(runs, scale=scale, hook=hook)
    attempts = result["attempts"]
    assert [(a["U"], a["batch"], a["three_arm"], a["failed_at"], a["failed_arms"])
            for a in attempts] == [
        (8.0, 20, True, "discovery", [ADAMW]), (7.0, 20, True, "discovery", [PREWARMED]),
        (8.0, 20, False, "discovery", [ADAMW]), (7.0, 20, False, None, [])]  # fmt: skip
    assert "a coverage miss that re-targeting cannot repair" in attempts[0]["reasons"][0]
    assert 1.42e-5 in result["rounds"][f"{ADAMW}/B=20/U=7"]["targeted"]["new_rates"]
    assert attempts[2]["reasons"] == [f"['{ADAMW}'] without a feasible triple at the three arms' "
                                      "visit of the candidate, to which the pair's rates are "
                                      "pinned"]  # fmt: skip
    assert seen[-4:] == ["numerical", "numerical-check", "validation", "done"]
    done = result["next"]
    assert (done["batch"], done["U"], done["three_arm"]) == (20, 7.0, False)
    assert {arm: done["rates"][arm] for arm in (ADAMW, FA)} == {
        arm: attempts[1]["arms"][arm]["selections"][-1]["rates"] for arm in (ADAMW, FA)
    }
    assert 1.42e-5 in done["rates"][ADAMW]  # U = 7's own triple


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


def test_a_technical_validation_shortfall_blocks() -> None:
    """The 10 of 12 unchanged: a validation that fails
    only because blocks are unsupported for technical causes (here AdamW's K* run stopped out of
    memory on seeds 2100-2102: 9 of 12 jointly supported), with no instability and no utility
    failure, blocks the stage for a human decision. The same shortfall with a cause in K (the
    K* run far below K*, so that the bracket is too wide) moves on to (7, 20); so does a
    shortfall the K-lost blocks make alone, technical losses beside them."""

    def hook(lost: dict[int, str]):
        def made_run(run):
            how = lost.get(run.seed)
            if (run.arm, run.lr) != (ADAMW, 1.48e-5) or how is None:
                return run
            if how == "oom":
                return traj(run.arm, run.batch_size, run.lr, run.seed, None, stop="oom", steps=10)
            return traj(run.arm, run.batch_size, run.lr, run.seed, 1.0, mmlu=0.3, factual=0.1)

        return made_run

    technical = dict.fromkeys((2100, 2101, 2102), "oom")
    seen, result, _ = drive(grids(), hook=hook(technical))
    (attempt,) = result["attempts"]
    step = result["next"]
    assert (attempt["outcome"], step["campaign"], step["blocked"], step["at"]) == (
        "blocked", "incomplete", "technical", "validation")  # fmt: skip
    assert step["arms"] == [ADAMW] and result["problems"] == []
    assert [(r["lr"], r["state"]) for r in step["rates"]] == [(1.48e-5, "technical failure")] * 3
    assert step["rates"][0]["why"] == ("validation block 2100: adamw_25-b20-1.48e-05-s2100: "
                                       "stopped by oom at step 10")  # fmt: skip
    assert "9 of 12 blocks jointly supported" in step["reasons"][0]
    assert "12 of 12 blocks would be jointly supported" in step["reasons"][0]
    assert attempt["validation"]["passes"] is False  # the verdict itself is unchanged
    # The same shortfall by K: a scientific failure, the order moves on.
    by_k = dict.fromkeys((2100, 2101, 2102), "low")
    seen, result, _ = drive(grids(), hook=hook(by_k))
    first = result["attempts"][0]
    assert (first["outcome"], first["failed_at"]) == ("failed", "validation")
    assert result["attempts"][1]["U"] == 7.0 and "blocked" not in result["next"]
    # Two blocks lost by K and one by a technical stop: 10 of 12 but for the technical one,
    # blocked; three by K and one technical: 9 even so, a scientific failure.
    mixed = {2100: "low", 2101: "low", 2102: "oom"}
    assert drive(grids(), hook=hook(mixed))[1]["next"]["blocked"] == "technical"
    mixed[2103] = "low"
    result = drive(grids(), hook=hook(mixed))[1]
    assert result["attempts"][0]["failed_at"] == "validation" and "blocked" not in result["next"]


def test_a_three_arm_validation_failure_falls_back_to_the_primary_validation() -> None:
    """The pre-warmed arm unsupported on the validation seeds fails the three arms at (8, 20),
    (7, 20) and (6, 20); the primary pair at (7, 20) takes the three arms' triples there
    (amendment 5) and passes amendment 3's primary-arm validation on their runs."""

    def unsupported(run):  # the pre-warmed arm never brackets K* on the validation seeds
        if run.seed in VALIDATION_SEEDS and run.arm == PREWARMED:
            return traj(run.arm, run.batch_size, run.lr, run.seed, 1.0, mmlu=0.3, factual=0.1)
        return run

    runs = grids()
    seen, result, verdicts = drive(runs, hook=unsupported)
    attempts = result["attempts"]
    assert [a["failed_at"] for a in attempts[:3]] == ["validation"] * 3
    assert "0 of 12 blocks jointly supported for the three arms" in attempts[0]["reasons"][0]
    assert seen[-2:] == ["validation", "done"] and len(attempts) == 4
    done = result["next"]
    assert (done["batch"], done["U"], done["three_arm"]) == (20, 7.0, False)
    assert done["validation"]["arms"] == [ADAMW, FA] and done["validation"]["passes"]
    assert done["rates"][ADAMW] == [1.27e-5, 1.48e-5, 1.5e-5]
    assert len([v for v in verdicts if v.startswith("numerical/b20-u7/")]) == 1


def test_no_candidate_gives_a_b20_feasibility_report() -> None:
    """When no candidate qualifies, the stage ends ``infeasible`` (the name kept)
    with ``scope`` "B=20" and a feasibility report, per candidate and arm, of what failed and
    why, the scientific failures apart from the technical ones; never a B = 5 candidate."""
    wide = {ADAMW: {2000: 0.8, 2001: 1.0, 2002: 1.3}}  # wider than the targets' spacing
    runs = grids(spread=wide)
    seen, result, _ = drive(runs, spread=wide)
    # Every candidate, U = 7 and U = 6 included, gets its own targeted and repair rounds; the
    # fallback visits (7, 20) and (6, 20) again over every record at their batch, with no new
    # round, and fails them too; the pair is never visited at (8, 20) (amendment 5).
    assert seen == ["discovery", "repair"] * 3 + ["infeasible"]
    pairs = [(u, b, arms) for arms, order in ((True, CANDIDATES), (False, FALLBACK_CANDIDATES))
             for u, b in order]  # fmt: skip
    assert [(a["U"], a["batch"], a["three_arm"]) for a in result["attempts"]] == pairs
    step = result["next"]
    pinned = "at the three arms' visit of the candidate, to which the pair's rates are pinned"
    assert step["reasons"] == [
        f"U={u:g}, B={b}, {'three arms' if arms else 'primary arms'}: ['{ADAMW}'] without a "
        f"feasible triple {'after the rounds' if arms else pinned}"
        for u, b, arms in pairs]  # fmt: skip
    report = step["feasibility"]
    assert (step["scope"], report["scope"], report["confirmation"]) == ("B=20", "B=20", False)
    assert [(c["U"], c["batch"], c["scope"], c["visited"], c["failed_at"])
            for c in report["candidates"]] == [
        (8.0, 20, "three arms", True, "discovery"), (7.0, 20, "three arms", True, "discovery"),
        (6.0, 20, "three arms", True, "discovery"), (7.0, 20, "primary pair", True, "discovery"),
        (6.0, 20, "primary pair", True, "discovery")]  # fmt: skip
    first = report["candidates"][0]["arms"]
    assert first[ADAMW]["failed"] and first[ADAMW]["technical"] == []
    assert first[ADAMW]["scientific"] == [f"discovery: ['{ADAMW}'] without a feasible triple "
                                          "after the rounds"]  # fmt: skip
    assert not first[FA]["failed"] and first[FA]["scientific"] == first[FA]["technical"] == []
    assert set(report["candidates"][3]["arms"]) == {ADAMW, FA}  # the primary pair
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
    # The report keeps the technical apart from the scientific. At (8, 20) FA's 1.59e-5 is not
    # qualified (re-solved, it decided nothing) and AdamW's 1.48e-5 failing (re-solved too); the
    # three-arm validation then fails on the pre-warmed arm. At (7, 20) AdamW's 1.27e-5 fails,
    # and so does its allowance's replacement; at (6, 20) AdamW has no triple without them. The
    # primary pair at (7, 20) is not visited (AdamW failed its numerical gate there), and the
    # pair at (6, 20) fails on AdamW as the three arms did.

    def unsupported(run):  # the pre-warmed arm never brackets K* on the validation seeds
        if run.seed in VALIDATION_SEEDS and run.arm == PREWARMED:
            return traj(run.arm, run.batch_size, run.lr, run.seed, 1.0, mmlu=0.3, factual=0.1)
        return run

    def judge(check):
        adamw, fa, changes = check["rates"].get("adamw", []), check["rates"]["adam_frozen"], {}
        if check["U"] == 8 and 1.59e-5 in fa:
            changes[("adam_frozen", 1.59e-5)] = "not_qualified"
        if check["U"] == 8 and 1.48e-5 in adamw:
            changes[("adamw", 1.48e-5)] = "failing"
        if check["U"] == 7:  # 1.27e-5 and its replacement 1.26e-5
            changes.update({("adamw", lr): "failing" for lr in adamw if lr < 1.3e-5})
        return verdict_of(check, changes, reasons={"adam_frozen|1.59e-05": [S2_TECHNICAL]})

    seen, result, _ = drive(grids(), judge=judge, hook=unsupported)
    report = result["next"]["feasibility"]
    assert [(c["U"], c["scope"], c.get("failed_at")) for c in report["candidates"]] == [
        (8.0, "three arms", "validation"), (7.0, "three arms", "numerical"),
        (6.0, "three arms", "discovery"), (7.0, "primary pair", None),
        (6.0, "primary pair", "discovery")]  # fmt: skip
    at8 = report["candidates"][0]["arms"]
    assert at8[FA]["technical"] == [f"numerical 1.59e-05: not qualified (technical) "
                                    f"({S2_TECHNICAL})"]  # fmt: skip
    assert not any("1.59e-05" in line for line in at8[FA]["scientific"])
    assert "numerical 1.48e-05: failing: a pair exceeding the bound" in at8[ADAMW]["scientific"]
    assert at8[ADAMW]["technical"] == [] and at8[PREWARMED]["technical"] == []
    assert sum("unsupported (bracketing; by K alone: bracketing)" in line
               for line in at8[PREWARMED]["scientific"]) == 12  # fmt: skip
    at7 = report["candidates"][1]["arms"]
    assert at7[FA]["excluded"] == {
        "1.59e-05": {
            "kind": "technical",
            "why": f"not qualified (technical) ({verdict_path(SELECTED20)})",
        }
    }
    assert at7[ADAMW]["excluded"]["1.48e-05"]["kind"] == "scientific"
    assert at7[ADAMW]["failed"] and not at7[FA]["failed"]
    assert report["candidates"][3]["visited"] is False


# ------------------------------------------------------------------ B = 20, the fixed scope


def test_a_technical_rate_spends_the_allowance_then_blocks() -> None:
    """With S2's shape at (8, 20): FA's lowest rate and the pre-warmed arm's U-rate
    not qualified (technical). Each arm spends its search allowance once: here the re-solve finds
    a replacement for both. The replacements not qualified again, the stage is blocked for a
    human decision, not moved on to (7, 20): a technical rate never decides the candidate."""
    scale = {**SCALE, (PREWARMED, 20): 3.3}  # the pre-warmed U-rate has a neighbour near 8
    first = {("adam_frozen", 1.59e-5), ("adamw_prewarmed", 1.86e-5)}
    again = {("adam_frozen", 1.5e-5), ("adamw_prewarmed", 1.74e-5)}  # the replacements

    def judge(marked):
        def verdict(check):
            found = {(o, lr) for o, lrs in check["rates"].items() for lr in lrs}
            reasons = {f"{o}|{lr:g}": [S2_TECHNICAL] for o, lr in found & marked}
            return verdict_of(check, {key: "not_qualified" for key in found & marked},
                              reasons=reasons)  # fmt: skip

        return verdict

    runs = grids(scale)
    seen, result, verdicts = drive(runs, scale=scale, judge=judge(first | again))
    assert seen == ["discovery", "numerical", "numerical-check", "numerical", "numerical-check",
                    "incomplete"]  # fmt: skip
    step = result["next"]
    assert (step["campaign"], step["blocked"], step["at"], step["batch"], step["U"]) == (
        "incomplete", "technical", "numerical", 20, 8.0)  # fmt: skip
    assert step["arms"] == [FA, PREWARMED] and result["problems"] == []
    assert [(r["arm"], r["lr"], r["state"], r["reasons"]) for r in step["rates"]] == [
        (FA, 1.5e-5, "not_qualified", [S2_TECHNICAL]),
        (PREWARMED, 1.74e-5, "not_qualified", [S2_TECHNICAL])]  # fmt: skip
    assert "stopped by technical marks only" in step["reasons"][0]
    (attempt,) = result["attempts"]  # no (7, 20)
    assert (attempt["outcome"], attempt["blocked_arms"], attempt["failed_arms"]) == (
        "blocked", [FA, PREWARMED], [])  # fmt: skip
    # The allowance, once per arm: the re-solve from the first check; AdamW spent none.
    resolved = {**SELECTED20, FA: [1.5e-5, 1.86e-5, 2.17e-5],
                PREWARMED: [1.27e-5, 1.59e-5, 1.74e-5]}  # fmt: skip
    original = {**SELECTED20, PREWARMED: [1.27e-5, 1.59e-5, 1.86e-5]}
    assert list(verdicts) == [verdict_path(original), verdict_path(resolved)]
    assert {arm: (a["check"], a["resolved"], a["repair"])
            for arm, a in attempt["allowance"].items()} == {
        FA: (numerical_digest(original), resolved[FA], None),
        PREWARMED: (numerical_digest(original), resolved[PREWARMED], None)}  # fmt: skip
    assert [len(attempt["arms"][arm]["selections"]) for arm in ARMS] == [2, 3, 3]
    # Replayed, it stays blocked: no further check is asked for.
    again_step = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=verdicts,
                           facts=commits(runs), reuse=reusable)["next"]  # fmt: skip
    assert again_step == step
    # The replacements qualified, the stage goes on at (8, 20): the technical marks spent the
    # allowance and decided nothing.
    runs = grids(scale)
    seen, result, _ = drive(runs, scale=scale, judge=judge(first))
    assert seen[-1] == "done" and (result["next"]["U"], result["next"]["three_arm"]) == (8.0, True)
    assert result["next"]["rates"] == resolved
    # A re-solve with no feasible triple gets the allowance's repair round.
    # In S2's shape at the world's own scale the pre-warmed U-rate 1.74e-5 is
    # the one rate near U: re-targeting U lands on it (its K still interpolated), so the round
    # proposes the nearest admissible rate inside its bracketing pair (1.59e-5, 1.74e-5),
    # 1.73e-5, never the unusable one; run and checked, it qualifies and the stage is done.
    marked = {("adam_frozen", 1.59e-5), ("adamw_prewarmed", 1.74e-5)}
    runs = grids()
    seen, result, verdicts = drive(runs, judge=judge(marked))
    assert seen == ["discovery", "numerical", "numerical-check", "repair", "numerical",
                    "numerical-check", "validation", "done"]  # fmt: skip
    rounds = result["rounds"][f"{PREWARMED}/B=20/U=8"]["allowance_repair"]
    assert rounds["new_rates"] == [1.73e-5] and rounds["proposal"]["unrepairable"] == []
    assert [(s["unusable"], s["rate"]) for s in rounds["proposal"]["substituted"]] == [
        (1.74e-5, 1.73e-5)]  # fmt: skip
    assert result["next"]["rates"][PREWARMED] == [1.27e-5, 1.48e-5, 1.73e-5]
    assert result["next"]["rates"][FA] == [1.5e-5, 1.86e-5, 2.17e-5]  # re-solved, no round
    # The replacement not qualified again: the allowance is spent, and the stage is blocked.
    runs = grids()
    seen, result, _ = drive(runs, judge=judge(marked | {("adamw_prewarmed", 1.73e-5)}))
    step = result["next"]
    assert (step["blocked"], step["arms"], [r["lr"] for r in step["rates"]]) == (
        "technical", [PREWARMED], [1.73e-5])  # fmt: skip
    assert len(result["attempts"]) == 1


def test_the_allowance_repair_round_is_a_repair_campaign() -> None:
    """When the re-solve after a numerical exclusion has no feasible triple, the
    arm's allowance holds one deterministic repair round (the ``repair`` kind and algorithm, K
    and utility only), named after the check that spent it; its replacement is checked again.
    Here (8, 20) fails its validation (AdamW's U-rate loses 6 pp MMLU), and at (7, 20) FA's
    U-rate 2e-5 is not qualified: without it FA has no triple, and the round's re-targeting
    proposes 1.99e-5."""

    def lossy(run):
        if run.seed in VALIDATION_SEEDS and (run.arm, run.lr) == (ADAMW, 1.74e-5):
            k = run.evaluations[0]["K_benign"]
            return traj(run.arm, run.batch_size, run.lr, run.seed, k, mmlu=6.0, factual=0.1 * k)
        return run

    def judge(check):
        if check["U"] == 7 and 2e-5 in check["rates"]["adam_frozen"]:
            return verdict_of(check, {("adam_frozen", 2e-5): "not_qualified"})
        return verdict_of(check)

    runs = grids()
    seen, result, verdicts = drive(runs, judge=judge, hook=lossy, until="repair")
    step = result["next"]
    at7 = {**SELECTED20, ADAMW: [1.27e-5, 1.48e-5, 1.5e-5], FA: [1.59e-5, 1.86e-5, 2e-5],
           PREWARMED: [1.27e-5, 1.48e-5, 1.59e-5]}  # fmt: skip
    assert (step["campaign"], step["U"], step["allowance"]) == ("repair", 7.0,
                                                                numerical_digest(at7))  # fmt: skip
    assert campaign_runs(step) == {(FA, 20, 1.99e-5, s) for s in SEEDS}
    seen, result, verdicts = drive(runs, judge=judge, hook=lossy, verdicts=verdicts)
    assert seen == ["repair", "numerical", "numerical-check", "validation", "done"]
    assert result["next"]["rates"][FA] == [1.59e-5, 1.86e-5, 1.99e-5]
    attempt = result["attempts"][1]
    assert attempt["allowance"][FA]["repair"]["rates"] == [1.99e-5]
    assert attempt["allowance"][FA]["resolved"] is None  # the re-solve had no triple
    assert [r["kind"] for r in attempt["arms"][FA]["rounds"]] == ["allowance_repair"]
    # The replacement got its own cases: 1.99e-5 is FA's new U-rate, on 2200-2202.
    assert {(r.arm, r.seed) for r in runs if r.lr == 1.99e-5 and r.seed in NUMERICAL_SEEDS} == {
        (FA, s) for s in NUMERICAL_SEEDS}  # fmt: skip


def test_a_scientific_failure_advances_the_order() -> None:
    """A scientific failure moves the fixed order on. AdamW's K* rate failing at (8, 20), and
    its replacement failing after the allowance, fail the candidate: the three arms at (7, 20)
    follow. There every pre-warmed rate is failing, so the pre-warmed arm fails, at (7, 20) and
    then at (6, 20), where its failing rates leave it no triple; the primary pair at (7, 20) is
    taken (amendment 5's order: no pair at (8, 20)), from the three-arm check's primary
    entries."""

    def judge(check):
        changes = {}
        if check["U"] == 8:
            changes = {("adamw", lr): "failing" for lr in (1.48e-5, 1.5e-5)
                       if lr in check["rates"]["adamw"]}  # fmt: skip
        if check["U"] == 7:
            changes = {("adamw_prewarmed", lr): "failing"
                       for lr in check["rates"].get("adamw_prewarmed", [])}  # fmt: skip
        return verdict_of(check, changes)

    runs = grids()
    seen, result, verdicts = drive(runs, judge=judge)
    assert [(a["U"], a["batch"], a["three_arm"], a["outcome"], a["failed_at"], a["failed_arms"])
            for a in result["attempts"]] == [
        (8.0, 20, True, "failed", "numerical", [ADAMW]),
        (7.0, 20, True, "failed", "numerical", [PREWARMED]),
        (6.0, 20, True, "failed", "discovery", [PREWARMED]),
        (7.0, 20, False, "eligible", None, [])]  # fmt: skip
    done = result["next"]
    assert (done["campaign"], done["batch"], done["U"], done["three_arm"]) == (
        "done", 20, 7.0, False)  # fmt: skip
    assert done["prewarmed_dropped"] and done["validation"]["arms"] == [ADAMW, FA]
    assert len(verdicts) == 3  # (8, 20)'s two checks and (7, 20)'s; none for the primary pair
    pair = result["attempts"][3]
    assert pair["numerical"][0]["verdict"] == result["attempts"][1]["numerical"][0]["verdict"]
    assert all(r.batch_size == 20 for r in runs if r.seed not in SEEDS)


def test_prewarmed_only_failure_keeps_b20() -> None:
    """The pre-warmed arm failing alone, at every U, in its qualification (every pre-warmed
    rate failing) or in its validation (it never brackets K* on the validation seeds), keeps
    the primary comparison at B = 20: the selection reaches the primary pair at (7, 20), the
    first of the fallback's order (amendment 5), with no mechanism cohort, never a B = 5
    candidate."""

    def failing(check):
        return verdict_of(check, {("adamw_prewarmed", lr): "failing"
                                  for lr in check["rates"].get("adamw_prewarmed", [])})  # fmt: skip

    def unsupported(run):
        if run.seed in VALIDATION_SEEDS and run.arm == PREWARMED:
            return traj(run.arm, run.batch_size, run.lr, run.seed, 1.0, mmlu=0.3, factual=0.1)
        return run

    # Excluded at B = 20 after (8, 20)'s check, the pre-warmed rates leave it no triple at
    # (7, 20) and (6, 20); a failed validation fails the three arms jointly, its reasons naming
    # the pre-warmed arm's blocks. The pair takes the three arms' U = 7 triples (amendment 5).
    primary7 = {ADAMW: [1.27e-5, 1.48e-5, 1.5e-5], FA: [1.59e-5, 1.86e-5, 2e-5]}
    for judge, hook, at, arms in (
        (failing, None, ("numerical", "discovery", "discovery"), [PREWARMED]),
        (verdict_of, unsupported, ("validation",) * 3, list(ARMS)),
    ):
        runs = grids()
        seen, result, _ = drive(runs, judge=judge, hook=hook)
        attempts = result["attempts"]
        assert [(a["U"], a["three_arm"], a["failed_at"]) for a in attempts] == [
            (8.0, True, at[0]), (7.0, True, at[1]), (6.0, True, at[2]),
            (7.0, False, None)]  # fmt: skip
        assert attempts[0]["failed_arms"] == arms and {a["batch"] for a in attempts} == {20}
        done = result["next"]
        assert (done["campaign"], done["batch"], done["U"], done["three_arm"]) == (
            "done", 20, 7.0, False)  # fmt: skip
        assert done["prewarmed_dropped"] and done["rates"] == {**primary7, PREWARMED: None}
        assert {r.batch_size for r in runs if r.seed not in SEEDS} == {20}


@pytest.mark.usefixtures("amendment4_order")
def test_the_allowance_carries_into_the_primary_pair() -> None:
    """The allowance carries from a candidate's three-arm attempt into its primary
    pair's and is never reset. At (8, 20) AdamW's K* rate is not qualified: AdamW spends its
    allowance (re-solved to 1.5e-5) while the pre-warmed arm fails the candidate (its rates
    failing, its allowance's repair rates out of their utility bounds). In the primary pair at
    (8, 20) the primary-only check finds AdamW's replacement not qualified: its allowance is
    spent, so no second re-solve, and the stage is blocked."""

    def judge(check):
        rates = check["rates"]
        changes = {("adamw_prewarmed", lr): "failing" for lr in rates.get("adamw_prewarmed", [])}
        if check["U"] == 8 and 1.48e-5 in rates["adamw"]:
            changes[("adamw", 1.48e-5)] = "not_qualified"
        if "adamw_prewarmed" not in rates and 1.5e-5 in rates["adamw"]:
            changes[("adamw", 1.5e-5)] = "not_qualified"
        return verdict_of(check, changes)

    runs = grids()
    seen, result, verdicts = drive(runs, judge=judge, hook=prewarmed_first_only)
    first, at7, pair = result["attempts"]
    assert first["allowance"][ADAMW]["resolved"] == [1.27e-5, 1.5e-5, 1.74e-5]
    assert (first["failed_arms"], at7["failed_arms"]) == ([PREWARMED], [PREWARMED])
    assert (pair["U"], pair["three_arm"], pair["allowance_carried"]) == (8.0, False, [ADAMW])
    assert pair["allowance"][ADAMW]["carried"] is True
    assert len(pair["arms"][ADAMW]["selections"]) == 1  # no second re-solve
    assert pair["numerical"][0]["request"][ADAMW] == [1.27e-5, 1.5e-5, 1.74e-5]
    step = result["next"]
    assert (step["blocked"], step["U"], step["three_arm"], step["arms"]) == (
        "technical", 8.0, False, [ADAMW])  # fmt: skip
    assert [(r["lr"], r["state"]) for r in step["rates"]] == [(1.5e-5, "not_qualified")]
    assert step["reasons"][0].startswith("unusable rates in the pinned triples, with no repair")
    assert verdict_path(PRIMARY20 | {ADAMW: [1.27e-5, 1.5e-5, 1.74e-5]}) in verdicts


@pytest.mark.usefixtures("amendment4_order")
def test_a_technical_stop_beside_a_scientific_failure_blocks_the_primary_pair() -> None:
    """One allowance in all per arm, batch and U, across passes. At (8, 20)
    AdamW's K* rate is not qualified and the pre-warmed U-rate failing: both spend their
    allowances. The second check finds AdamW's replacement 1.5e-5 not qualified (technical)
    and the pre-warmed replacement failing (scientific): the candidate fails on the pre-warmed
    arm alone, AdamW stopped technically beside it. (7, 20) fails on the pre-warmed arm. The
    primary pair at (8, 20) then blocks on AdamW's technical stop: AdamW is never selected
    again (U = 7's rates would have given it a third check and ``done``). With the replacement
    failing instead, AdamW fails with the pre-warmed arm and the fallback skips (8, 20)."""
    scale = {**SCALE, (PREWARMED, 20): 3.3}

    def judge(replacement: str):
        def verdict(check):
            rates, changes = check["rates"], {}
            if check["U"] == 8:
                states = {1.48e-5: "not_qualified", 1.5e-5: replacement}
                changes = {("adamw", lr): states[lr] for lr in rates["adamw"] if lr in states}
                changes |= {
                    ("adamw_prewarmed", lr): "failing"
                    for lr in rates.get("adamw_prewarmed", [])
                    if lr in (1.86e-5, 1.74e-5)
                }
            if check["U"] == 7:
                changes = {("adamw_prewarmed", lr): "failing"
                           for lr in rates.get("adamw_prewarmed", [])}  # fmt: skip
            return verdict_of(check, changes)

        return verdict

    runs = grids(scale)
    seen, result, verdicts = drive(runs, scale=scale, judge=judge("not_qualified"),
                                   hook=prewarmed_first_only)  # fmt: skip
    first, at7, pair = result["attempts"]
    assert (first["failed_at"], first["failed_arms"], first["stopped_technical"]) == (
        "numerical", [PREWARMED], [ADAMW])  # fmt: skip
    assert "besides, adamw_25 stopped by technical marks only (['1.5e-05'])" in (
        first["reasons"][0])  # fmt: skip
    assert at7["failed_arms"] == [PREWARMED]
    assert (pair["U"], pair["three_arm"], pair["outcome"], pair["allowance_carried"]) == (
        8.0, False, "blocked", [ADAMW])  # fmt: skip
    assert pair["arms"][ADAMW]["selections"] == [] and pair["numerical"] == []
    step = result["next"]
    assert (step["blocked"], step["at"], step["U"], step["arms"]) == (
        "technical", "numerical", 8.0, [ADAMW])  # fmt: skip
    assert [(r["lr"], r["state"]) for r in step["rates"]] == [(1.5e-5, "not_qualified")]
    assert len([v for v in verdicts if v.startswith("numerical/b20-u8/")]) == 2  # no third
    # The replacement failing (scientific): AdamW fails, and the fallback skips (8, 20).
    runs = grids(scale)
    seen, result, _ = drive(runs, scale=scale, judge=judge("failing"),
                            hook=prewarmed_first_only)  # fmt: skip
    attempts = result["attempts"]
    assert [(a["U"], a["three_arm"], a["failed_arms"]) for a in attempts] == [
        (8.0, True, [ADAMW, PREWARMED]), (7.0, True, [PREWARMED]), (7.0, False, [])]  # fmt: skip
    assert (result["next"]["campaign"], result["next"]["U"]) == ("done", 7.0)


@pytest.mark.usefixtures("amendment4_order")
def test_a_mark_beside_a_spent_stop_stops_the_pinned_pair() -> None:
    """The pre-warmed arm fails twice at (8, 20); its second check also marks FA's unchanged
    1.86e-5 (technical or failing), FA's allowance unspent when the pre-warmed arm's stop fails
    the candidate: FA owes it. The pair's rates at (8, 20) are the three arms' triples there
    (amendment 5), FA's holding 1.86e-5, and the pair has no re-selection: marked not qualified,
    the stage blocks on FA there; marked failing, FA fails the candidate. No further check is
    made at (8, 20)."""
    scale = {**SCALE, (PREWARMED, 20): 3.3}

    def judge(mark: str, replacement: str):
        def verdict(check):
            rates, changes = check["rates"], {}
            fa, pw = rates["adam_frozen"], rates.get("adamw_prewarmed", [])
            if check["U"] == 8:
                changes = {("adamw_prewarmed", lr): "failing" for lr in pw
                           if lr in (1.86e-5, 1.74e-5)}  # fmt: skip
                if 1.74e-5 in pw and 1.86e-5 in fa:
                    changes[("adam_frozen", 1.86e-5)] = mark  # the second three-arm check
                if not pw:  # the primary pair's: FA's replacement
                    changes |= {("adam_frozen", lr): replacement for lr in fa
                                if lr not in (1.59e-5, 2.17e-5)}  # fmt: skip
            if check["U"] == 7:
                changes = {("adamw_prewarmed", lr): "failing" for lr in pw}
            return verdict_of(check, changes)

        return verdict

    for mark, outcome in (("not_qualified", "blocked"), ("failing", "failed")):
        runs = grids(scale)
        seen, result, verdicts = drive(runs, scale=scale, judge=judge(mark, "not_qualified"),
                                       hook=prewarmed_first_only)  # fmt: skip
        first, at7, pair = result["attempts"][:3]
        assert (first["failed_arms"], first["allowance_owed"]) == ([PREWARMED], [FA])
        assert FA not in first["allowance"]  # not spent there: owed
        assert (pair["U"], pair["three_arm"], pair["outcome"]) == (8.0, False, outcome)
        assert FA not in pair["allowance"] and pair["numerical"] == []  # no re-selection
        assert 1.86e-5 in pair["arms"][FA]["selections"][-1]["rates"]
        assert "the pinned triples hold rates found unusable" in pair["reasons"][0]
        assert len([v for v in verdicts if v.startswith("numerical/b20-u8/")]) == 2
        if outcome == "blocked":
            step = result["next"]
            assert (step["blocked"], step["U"], step["three_arm"], step["arms"]) == (
                "technical", 8.0, False, [FA])  # fmt: skip
            assert [(r["lr"], r["state"]) for r in step["rates"]] == [
                (1.86e-5, "not_qualified")]  # fmt: skip
        else:
            assert (pair["failed_at"], pair["failed_arms"]) == ("numerical", [FA])


def test_the_nearest_admissible_rate_crosses_a_decade() -> None:
    """The allowance round's substitute walks the 3-digit grid of each
    rate's own decade, so below 1.00e-5 comes 9.99e-6, not 9.9e-6."""
    nearest = study2_short._nearest_admissible
    assert nearest(1.0e-5, 9.95e-6, 1.02e-5, {1e-5}) == 9.99e-6
    assert nearest(1.0004e-5, 9.95e-6, 1.02e-5, {1e-5, 1.01e-5}) == 9.99e-6
    assert nearest(9.99e-6, None, None, {9.99e-6, 9.98e-6}) == 1e-5  # up across the decade
    assert nearest(1.7352e-5, 1.59e-5, 1.74e-5, {1.74e-5}) == 1.73e-5


def test_a_cause_lifted_by_technical_exclusions_blocks_after_the_allowance() -> None:
    """After the allowance's repair, an arm left without a triple is judged
    by lifting its technical exclusions, whatever the marks of its last verdict. FA's 1.86e-5 is
    not qualified and 2.17e-5's control rejected; the round proposes 1.85e-5 and 2.18e-5, and
    1.85e-5 is out of its utility bounds: no triple. Lifting 1.86e-5 gives (1.59e-5, 1.86e-5,
    2.18e-5), so the stop is technical and the stage blocks, the rejected control
    notwithstanding."""

    def hook(run):
        if run.arm == FA and run.seed in SEEDS and run.lr == 1.85e-5:
            k = run.evaluations[0]["K_benign"]
            return traj(run.arm, run.batch_size, run.lr, run.seed, k, mmlu=6.0, factual=0.1 * k)
        return run

    def judge(check):
        fa = check["rates"]["adam_frozen"]
        return verdict_of(
            check,
            {("adam_frozen", 1.86e-5): "not_qualified"} if 1.86e-5 in fa else {},
            controls={2.17e-5: "rejected"} if 2.17e-5 in fa else {},
        )

    seen, result, _ = drive(grids(), judge=judge, hook=hook)
    assert seen == ["discovery", "numerical", "numerical-check", "repair", "incomplete"]
    (attempt,) = result["attempts"]
    assert attempt["allowance"][FA]["repair"]["rates"] == [1.85e-5, 2.18e-5]
    step = result["next"]
    assert (step["blocked"], step["arms"], [r["lr"] for r in step["rates"]]) == (
        "technical", [FA], [1.86e-5])  # fmt: skip
    assert "its same-path control rejected" in step["reasons"][0]  # in the last verdict


def test_a_utility_member_left_without_a_bound_by_technical_failures_blocks() -> None:
    """AdamW's K* rate reads below K* on every validation seed but 2103, so
    that its r_5 1.27e-5 brackets K* there only; its runs on the other seeds ran out of memory.
    Every block stays jointly supported, but the utility member (AdamW, 1.27e-5) has one
    completed run and no bound: a technical failure, so the stage blocks. Its runs completed
    with a loss above 5 pp instead, the bound is above its limit and the order moves on."""

    def hook(oom: bool, lost: tuple[int, ...] = ()):
        def made_run(run):
            if run.arm != ADAMW or run.seed not in VALIDATION_SEEDS:
                return run
            if run.lr == 1.74e-5 and run.seed in lost:
                return traj(run.arm, run.batch_size, run.lr, run.seed, None, stop="oom", steps=10)
            if run.lr == 1.48e-5:
                k = K8 * (1.05 if run.seed == 2103 else 0.95)
                return traj(run.arm, run.batch_size, run.lr, run.seed, k, mmlu=0.3 * k,
                            factual=0.1 * k)  # fmt: skip
            if run.lr == 1.27e-5 and oom and run.seed != 2103:
                return traj(run.arm, run.batch_size, run.lr, run.seed, None, stop="oom", steps=10)
            if run.lr == 1.27e-5 and not oom:
                k = run.evaluations[0]["K_benign"]
                return traj(run.arm, run.batch_size, run.lr, run.seed, k, mmlu=6.0, factual=0.1)
            return run

        return made_run

    seen, result, _ = drive(grids(), hook=hook(oom=True))
    (attempt,) = result["attempts"]
    assert len(attempt["validation"]["jointly_supported"]) == 12
    assert attempt["validation"]["reasons"] == ["the utility family does not pass"]
    step = result["next"]
    assert (step["blocked"], step["at"], step["arms"]) == ("technical", "validation", [ADAMW])
    assert len(step["rates"]) == 11 and {r["lr"] for r in step["rates"]} == {1.27e-5}
    assert step["rates"][0]["why"].startswith("utility at 1.27e-05 without a bound (1 completed "
                                              "runs): validation block 2100")  # fmt: skip
    assert "has 1 completed runs, fewer than two: no bound" in step["reasons"][0]
    seen, result, _ = drive(grids(), hook=hook(oom=False))
    first = result["attempts"][0]
    assert (first["failed_at"], result["attempts"][1]["U"]) == ("validation", 7.0)
    # Blocks 2100-2102 lost too (their U-rate runs out of memory: 9 of 12, 12 but for them):
    # a run that both faults name is listed once.
    seen, result, _ = drive(grids(), hook=hook(oom=True, lost=(2100, 2101, 2102)))
    step = result["next"]
    assert (step["blocked"], step["at"]) == ("technical", "validation")
    assert "9 of 12 blocks jointly supported" in step["reasons"][0]
    named = [(r["lr"], r["why"].split("validation block ")[1][:4]) for r in step["rates"]]
    assert len(named) == len(set(named)) == 3 + 11  # 2100-2102's U-rate, every 1.27e-5 once
    assert named.count((1.27e-5, "2100")) == 1


def test_the_feasibility_report_files_technical_losses_as_technical() -> None:
    """A validation block unsupported for a technical cause (an endpoint run
    out of memory) is listed under technical, one unsupported by K under scientific, and a
    utility member without a bound under technical; a block lost with an instability stop among
    its failures is scientific."""
    runs = validation_runs()
    lost = {(ADAMW, 1.2e-5, 2100), (ADAMW, 1.4e-5, 2101), (ADAMW, 1.2e-5, 2102)}
    runs = [r for r in runs if (r.arm, r.lr, r.seed) not in lost]
    runs.append(traj(ADAMW, 5, 1.2e-5, 2100, None, stop="oom", steps=10))  # technical
    runs.append(traj(ADAMW, 5, 1.4e-5, 2101, 6.1))  # every K below K*: bracketing
    runs.append(traj(ADAMW, 5, 1.2e-5, 2102, None, stop="k_above_max", steps=12))  # unstable
    accepted = {lr: {"accepted": True} for lr in FROZEN[FA]}
    verdict = validation_verdict(runs, FROZEN, accepted, THETA0, K8, W_MAX)
    found = study2_short._validation_feasibility(verdict, ADAMW)
    assert "validation block 2100: unsupported (technical; by K alone: width)" in (
        found["technical"])  # fmt: skip
    assert any("2100" in line and "stopped by oom" in line for line in found["technical"])
    assert "validation block 2101: unsupported (bracketing; by K alone: bracketing)" in (
        found["scientific"])  # fmt: skip
    assert not any("2100" in line for line in found["scientific"])
    assert "validation block 2102: unsupported (technical; by K alone: width)" in (
        found["scientific"])  # fmt: skip
    assert not any("2102" in line for line in found["technical"])
    unbounded = {**verdict, "utility": {"family": [
        {"arm": ADAMW, "rate": 1e-5, "runs": 1, "mmlu_upper_pp": None, "factual_upper_pp": None,
         "mmlu_passes": False, "factual_passes": False}]}}  # fmt: skip
    found = study2_short._validation_feasibility(unbounded, ADAMW)
    assert any(line.startswith("validation utility at 1e-05") for line in found["technical"])


def test_a_technical_exclusion_never_decides_a_later_candidate() -> None:
    """Technical marks never decide a candidate, at a later candidate's discovery either (this
    implementation's reading): AdamW's 1.48e-5 is not qualified at (8, 20) and
    re-solved; the candidate then fails its validation (AdamW's U-rate loses 6 pp MMLU). At
    (7, 20) AdamW's targeted rates are out of their utility bounds, and without 1.48e-5 no
    triple is left after its rounds; with it lifted there is one, so the arm is stopped by the
    technical mark alone: blocked at discovery, not moved on to the primary pair."""

    def hook(run):
        lossy = run.seed in VALIDATION_SEEDS and run.lr == 1.74e-5
        out_of_bounds = run.seed in SEEDS and run.lr in (1.42e-5, 1.59e-5)
        if run.arm == ADAMW and (lossy or out_of_bounds):
            k = run.evaluations[0]["K_benign"]
            return traj(run.arm, run.batch_size, run.lr, run.seed, k, mmlu=6.0, factual=0.1 * k)
        return run

    runs = grids()
    seen, result, _ = drive(
        runs, hook=hook, judge=lambda c: verdict_of(c, {("adamw", 1.48e-5): "not_qualified"})
    )
    failed, at7 = result["attempts"]
    assert failed["failed_at"] == "validation" and at7["outcome"] == "blocked"
    assert at7["excluded_kinds"][ADAMW] == {1.48e-5: "technical"}
    step = result["next"]
    assert (step["blocked"], step["at"], step["U"], step["arms"]) == (
        "technical", "discovery", 7.0, [ADAMW])  # fmt: skip
    assert [(r["lr"], r["state"]) for r in step["rates"]] == [(1.48e-5, "not_qualified")]
    assert step["allowances"] == {ADAMW: None}  # spent at (8, 20), not at (7, 20)


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
    slot = {"name": numerical_digest(rates), "dir": "numerical/b20-u8",
            "path": verdict_path(rates, upper=upper)}  # fmt: skip
    return study2_short._check_request(slot, 20, upper, rates, {COMMIT}, w_max=W_MAX)


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
    assert check["runs_commits"] == [COMMIT]
    assert check["policy"] == {"norm": 5e-05, "survival_share": 0.5, "retry_scale": 10,
                               "steps": [1, 5, 10, 15, 20, 25], "bound": 0.01 * K8,
                               "gaussian": {"p3": 3, "p4": 4},
                               "reference": "float64 loss, accumulation and optimizer "
                                            "arithmetic; fallback float64 loss and accumulation "
                                            "only",
                               "twin_a_consistency": {"abs": 1e-08, "rel": 0.0001}}  # fmt: skip


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
    # A control's reasons sit under its FA rate's key, prefixed "control: ".
    assert found["unusable"][2]["reasons"] == ["control: incomplete after its rerun"]
    # The record keeps the GPU-code check that let the verdict hold at the checkout.
    assert found["reuse"]["head"]["reusable"] and found["reuse"]["runs"] == {}
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
        (lambda r: {**r, "policy": {**r["policy"], "gaussian": {"p3": 7, "p4": 8}}}, "stale",
         "gaussian {'p3': 7, 'p4': 8}, frozen {'p3': 3, 'p4': 4}"),
        (lambda r: {**r, "policy": {k: v for k, v in r["policy"].items() if k != "gaussian"}},
         "stale", "gaussian None"),
        (lambda r: {**r, "policy": {**r["policy"], "reference": "float64"}}, "stale",
         "reference 'float64', frozen 'float64 loss, accumulation"),
        # Twin A's consistency rule is frozen; a verdict without it departs
        (lambda r: {**r, "policy": {k: v for k, v in r["policy"].items()
                                    if k != "twin_a_consistency"}}, "stale",
         "twin_a_consistency None, frozen {'abs': 1e-08, 'rel': 0.0001}"),
        (lambda r: {**r, "policy": {**r["policy"],
                                    "twin_a_consistency": {"abs": 1e-08, "rel": 0.001}}},
         "stale", "twin_a_consistency {'abs': 1e-08, 'rel': 0.001}, frozen"),
        (lambda r: {**r, "policy": {**r["policy"], "twin_a_consistency": None}}, "stale",
         "twin_a_consistency None"),
        (lambda r: _gpu(r, commit="g" * 40, sha256=GPU["g" * 40]), "stale",
         "numerical runs ran another GPU code"),
        (lambda r: _gpu(r, sha256="other"), "stale", "is not the identity of"),
        (lambda r: _gpu(r, commit="u" * 40), "stale", "cannot be checked"),
        (lambda r: {**r, "policy": {**r["policy"], "gpu_code": None}}, "stale",
         "holds no GPU code (null)"),
        (lambda r: {**r, "policy": {**r["policy"], "tool_commit": None}}, "stale",
         "tool_commit None names no commit"),
        (lambda r: {**r, "policy": {**r["policy"], "tool_commit": []}}, "stale",
         "tool_commit [] names no commit"),
        (lambda r: {**r, "policy": {**r["policy"], "tool_commit": [COMMIT, ""]}}, "stale",
         "names no commit"),
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


def test_the_tools_tracked_consistency_rule_must_be_the_frozen_one(monkeypatch) -> None:
    """The selection restates twin A's consistency rule (``NUMERICAL_POLICY``); the
    numerical-check tool's tracked settings must hold the same, or no verdict decides."""
    record = verdict_of(request_check())
    assert record["policy"]["twin_a_consistency"] == {"abs": 1e-08, "rel": 0.0001}
    assert numerical_policy(8.0)["twin_a_consistency"] == {"abs": 1e-08, "rel": 0.0001}
    assert read(record)["state"] == "complete"
    tool = study2_short._tool_policy()
    assert tool["twin_a_consistency"] == study2_short.NUMERICAL_POLICY["twin_a_consistency"]
    other = {**tool, "twin_a_consistency": {"abs": 1e-08, "rel": 0.001}}
    monkeypatch.setattr(study2_short, "_tool_policy", lambda: other)
    found = read(record)
    assert found["state"] == "stale"
    assert ("policy the numerical-check tool's twin_a_consistency {'abs': 1e-08, 'rel': 0.001}, "
            "frozen {'abs': 1e-08, 'rel': 0.0001}") in found["reasons"]  # fmt: skip


def test_a_verdict_holds_across_analysis_only_commits_and_not_gpu_changes() -> None:
    record = verdict_of(request_check())
    assert read(record, reuse=reuse_at("a" * 40))["state"] == "complete"  # analysis-only
    moved = read(record, reuse=reuse_at("g" * 40))  # a GPU path changed since
    assert moved["state"] == "stale" and "GPU paths changed since" in moved["reasons"][0]
    # Runs at an analysis-only descendant of the verdict's commit ran the same GPU code.
    later = read(record, commit="a" * 40)
    assert later["state"] == "complete" and later["reuse"]["runs"]["a" * 40]["reusable"]
    # The tool records the check's commits as a list (an older record: one commit).
    listed = {**record, "policy": {**record["policy"], "tool_commit": [COMMIT, "a" * 40]}}
    assert read(listed)["state"] == "complete"
    # The runs of one check made at several commits: each one's GPU code must be the verdict's.
    both = read(record, commit={COMMIT, "a" * 40})
    assert both["state"] == "complete" and set(both["reuse"]["runs"]) == {"a" * 40}
    moved = read(record, commit={COMMIT, "g" * 40})
    assert moved["state"] == "stale" and "another GPU code (" + "g" * 40 in moved["reasons"][0]
    unknown = read(record, commit={COMMIT, "d" * 40})
    assert unknown["state"] == "stale" and "cannot be checked" in unknown["reasons"][0]


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

    path = verdict_path(SELECTED20)
    step = at({})
    assert step["campaign"] == "numerical-check" and step["check"]["verdict"] == path
    assert step["check"]["runs_commits"] == [COMMIT] and len(step["check"]["cases"]) == 15
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
    # A numerical run on a dirty checkout, or one recording no commit: no conclusion.
    replaced = next(r for r in runs if r.seed == 2201).run_id
    for fact in ({"commit": COMMIT, "dirty": True}, {"commit": COMMIT, "dirty": None},
                 {"commit": None, "dirty": None}):  # fmt: skip
        found = at({path: good}, {**commits(runs), replaced: fact})
        assert found["campaign"] == "incomplete"
    # A replacement made at another commit: valid when its GPU
    # code is the verdict's (an analysis-only descendant), stale when it is not or unknown.
    descendant = {**commits(runs), replaced: {"commit": "a" * 40, "dirty": False}}
    assert at({path: good}, descendant)["campaign"] == "validation"
    for other, reason in (("g" * 40, "another GPU code"), ("d" * 40, "cannot be checked")):
        found = at({path: good}, {**commits(runs), replaced: {"commit": other, "dirty": False}})
        assert (found["campaign"], found["state"]) == ("numerical-check", "stale")
        assert found["check"]["runs_commits"] == sorted([COMMIT, other])
        assert any(reason in r for r in found["reasons"]), found["reasons"]


def test_a_numerical_run_given_up_lets_the_check_go_on() -> None:
    """A numerical run the launcher gave up on after its recovery
    round does not hold the check back; the tool reads its case as not qualified (technical),
    and its rate leaves the batch like any unusable one, spending its arm's search
    allowance: here AdamW's K* rate, re-solved."""
    runs = grids()
    _, result, _ = drive(runs, until="numerical")
    key = (ADAMW, 20, 1.48e-5, 2200)
    lost = dict(zip(("arm", "batch", "lr", "seed"), key, strict=True))
    runs += [made(r["arm"], r["batch"], r["lr"], r["seed"]) for r in result["next"]["runs"]
             if r != lost]  # fmt: skip

    def at(exhausted, verdicts=None):
        return selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=verdicts or {},
                         facts=commits(runs), exhausted=exhausted, reuse=reusable)  # fmt: skip

    assert at({})["next"]["campaign"] == "numerical"  # it may still run: the stage waits
    found = at({key: "attempt-6"})
    step = found["next"]
    assert step["campaign"] == "numerical-check" and len(step["check"]["cases"]) == 15
    assert found["attempts"][0]["numerical"][0]["given_up"] == {"attempt-6": list(key)}
    optimizer = study2_short.OPTIMIZERS[lost["arm"]]
    verdict = verdict_of(step["check"], {(optimizer, lost["lr"]): "not_qualified"})
    found = at({key: "attempt-6"}, {step["check"]["verdict"]: verdict})
    (unusable,) = found["attempts"][0]["numerical"][0]["unusable"]
    assert (unusable["arm"], unusable["lr"], unusable["why"]) == (
        lost["arm"], lost["lr"], "not qualified (technical)")  # fmt: skip
    assert found["next"]["campaign"] == "numerical" and found["problems"] == []
    assert found["next"]["rates"][ADAMW] == [1.27e-5, 1.5e-5, 1.74e-5]  # the replacement
    assert found["attempts"][0]["allowance"][ADAMW]["check"] == step["check"]["name"]


def test_a_stale_verdict_keeps_its_marks() -> None:
    """A rate a verdict marks failing stays unusable at its batch
    when the verdict goes stale; the re-solved selection's own verdict still decides."""
    runs = grids()
    judge = lambda check: verdict_of(check, {("adamw", 1.48e-5): "failing"})  # noqa: E731
    _, _, verdicts = drive(runs, judge=judge)
    path = verdict_path(SELECTED20)
    first = verdicts[path]
    stale = {**verdicts, path: {**first, "policy": {**first["policy"], "norm": 1e-4}}}
    result = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=stale,
                       facts=commits(runs), reuse=reusable)  # fmt: skip
    entry = result["attempts"][0]["numerical"][0]
    assert entry["state"] == "stale" and "norm 0.0001" in entry["reasons"][0]
    assert [(u["arm"], u["lr"]) for u in entry["unusable"]] == [(ADAMW, 1.48e-5)]
    assert result["next"]["campaign"] == "done" and result["problems"] == []
    assert result["next"]["rates"][ADAMW] == [1.27e-5, 1.5e-5, 1.74e-5]


def test_a_verdict_with_an_incomplete_entry_decides_no_re_solve() -> None:
    """A stale verdict that marks AdamW 1.48e-5
    failing but still holds an incomplete entry is not a basis for the re-solve. The next step
    reruns the check, the re-solve is not spent, and once the check is complete the failing
    mark re-solves AdamW as usual."""
    runs = grids()
    _, result, _ = drive(runs, until="numerical-check")
    check = result["next"]["check"]
    path = check["verdict"]
    marked = verdict_of(check, {("adamw", 1.48e-5): "failing",
                                ("adam_frozen", 1.86e-5): "incomplete"})  # fmt: skip
    stale = {**marked, "policy": {**marked["policy"], "norm": 1e-4}}

    def at(record):
        return selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical={path: record},
                         facts=commits(runs), reuse=reusable)  # fmt: skip

    for record, state in ((stale, "stale"), (marked, "incomplete")):
        found = at(record)
        step = found["next"]
        assert (step["campaign"], step["state"], step["check"]["verdict"]) == (
            "numerical-check", state, path)  # fmt: skip
        assert "incomplete: ['adam_frozen|1.86e-05']" in step["reasons"]
        attempt = found["attempts"][0]
        assert len(attempt["arms"][ADAMW]["selections"]) == 2  # no re-solve: not spent
        assert len(attempt["numerical"]) == 1 and attempt["outcome"] == "pending"
        assert found["problems"] == []
    entry = at(stale)["attempts"][0]["numerical"][0]
    assert entry["unfinished"] == ["adam_frozen|1.86e-05"]
    assert [(u["arm"], u["lr"]) for u in entry["unusable"]] == [(ADAMW, 1.48e-5)]
    # Complete again (the same failing mark, stale or not): the re-solve follows.
    finished = verdict_of(check, {("adamw", 1.48e-5): "failing"})
    for record in (finished, {**finished, "policy": {**finished["policy"], "norm": 1e-4}}):
        found = at(record)
        assert found["next"]["campaign"] == "numerical"
        assert found["attempts"][0]["arms"][ADAMW]["selections"][-1]["excluded"] == [1.48e-5]


def test_a_verdict_lacking_an_entry_decides_no_re_solve() -> None:
    """A verdict that marks AdamW 1.48e-5 failing but lacks the
    control of an FA rate, or tested FA's U-rate on fewer seeds than its cases, has evidence
    still to come: the next step is the check, and the re-solve is not spent."""
    runs = grids()
    _, result, _ = drive(runs, until="numerical-check")
    check = result["next"]["check"]
    path = check["verdict"]
    marked = verdict_of(check, {("adamw", 1.48e-5): "failing"})
    lacking = {**marked, "controls": {k: v for k, v in marked["controls"].items()
                                      if k != "1.86e-05"}}  # fmt: skip
    short = {**marked, "seeds": {**marked["seeds"], "adam_frozen|2.17e-05": [2200]}}
    for record, unfinished in ((lacking, "control 1.86e-05"),
                               (short, "adam_frozen|2.17e-05 seeds")):  # fmt: skip
        found = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical={path: record},
                          facts=commits(runs), reuse=reusable)  # fmt: skip
        step = found["next"]
        assert (step["campaign"], step["state"]) == ("numerical-check", "stale")
        attempt = found["attempts"][0]
        assert attempt["numerical"][0]["unfinished"] == [unfinished]
        assert len(attempt["arms"][ADAMW]["selections"]) == 2  # no re-solve: not spent
        assert found["problems"] == []


def test_an_earlier_verdict_keeps_its_marks() -> None:
    """The tool moves a selection's
    earlier verdict to verdicts/history/<digest>/ when it writes the selection's verdict again.
    A rate the earlier verdict marks failing stays unusable, though it was stale with an
    unfinished entry and the new verdict omits the mark; the new verdict decides the step."""
    runs = grids()
    _, result, _ = drive(runs, until="numerical-check")
    check = result["next"]["check"]
    path = check["verdict"]
    earlier = verdict_of(check, {("adamw", 1.48e-5): "failing",
                                 ("adam_frozen", 1.86e-5): "incomplete"})  # fmt: skip
    earlier = {**earlier, "policy": {**earlier["policy"], "norm": 1e-4}}  # stale too
    moved = path.replace("/verdicts/", "/verdicts/history/").removesuffix(".json") + "/0.json"
    rerun = verdict_of(check)  # every rate qualified

    def at(verdicts):
        return selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=verdicts,
                         facts=commits(runs), reuse=reusable)  # fmt: skip

    assert at({path: rerun})["next"]["campaign"] == "validation"  # without the history
    found = at({path: rerun, moved: earlier})
    entry = found["attempts"][0]["numerical"][0]
    assert (entry["state"], entry["history"]) == ("complete", [moved])
    assert [(u["arm"], u["lr"], u["state"]) for u in entry["unusable"]] == [
        (ADAMW, 1.48e-5, "failing")]  # fmt: skip
    assert entry["unusable"][0]["why"].endswith(f"(earlier verdict {moved})")
    assert found["next"]["campaign"] == "numerical" and found["problems"] == []
    assert found["next"]["rates"][ADAMW] == [1.27e-5, 1.5e-5, 1.74e-5]  # the re-solve
    assert found["unread_verdicts"] == []
    # The current verdict still decides: unfinished, it holds the step at the check.
    unfinished = verdict_of(check, {("adam_frozen", 1.86e-5): "incomplete"})
    step = at({path: unfinished, moved: earlier})["next"]
    assert (step["campaign"], step["state"]) == ("numerical-check", "incomplete")
    # An earlier verdict that is not one of this selection, or a record at another batch in
    # its directory: no conclusion.
    other = verdict_of(request_check({**SELECTED20, ADAMW: [1.27e-5, 1.5e-5, 1.74e-5]}))
    elsewhere = verdict_of(check, {("adamw", 1.48e-5): "failing"}, batch=5)
    for record, why in ((other, "not an earlier verdict of the selection"),
                        (elsewhere, "its record is at B=5, U=8")):  # fmt: skip
        found = at({path: rerun, moved: record})
        assert found["next"]["campaign"] == "incomplete"
        assert why in found["problems"][0], found["problems"]
    # An earlier verdict of a selection the replay never read, marking a rate: an unread verdict.
    stray = verdict_path(SELECTED20, upper=7.0).replace("/verdicts/", "/verdicts/history/")
    stray = stray.removesuffix(".json") + "/0.json"
    found = at({path: rerun, stray: verdict_of(request_check(SELECTED20, upper=7.0),
                                               {("adamw", 1.27e-5): "failing"})})  # fmt: skip
    assert found["unread_verdicts"] == [stray] and found["next"]["campaign"] == "incomplete"
    assert "did not read marks ['adamw|1.27e-05']" in found["problems"][0]


def test_runs_beyond_a_stale_check_wait_for_it() -> None:
    """A GPU-path change after the validation runs were made leaves
    the qualifying verdict stale. The problems name the stale check first; every other one is a
    run made beyond it (``beyond_stale``), so the check's input may still be written."""
    runs = grids()
    _, _, verdicts = drive(runs)
    moved = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=verdicts,
                      facts=commits(runs), reuse=reuse_at("g" * 40))  # fmt: skip
    assert moved["next"] == {"campaign": "incomplete"}
    pending = moved["pending"]
    assert (pending["campaign"], pending["state"]) == ("numerical-check", "stale")
    waits, *beyond = moved["problems"]
    assert waits.startswith(f"the replay waits at the stale check {verdict_path(SELECTED20)} "
                            "(GPU paths changed since")  # fmt: skip
    assert len(beyond) == 3 * 3 * 12 and all(p.startswith("unexpected run") for p in beyond)
    assert moved["beyond_stale"] == moved["problems"]
    # Another problem beside them: no allowance.
    plain = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC, numerical=verdicts,
                      facts=commits(runs), reuse=reusable)  # fmt: skip
    assert "beyond_stale" not in plain and plain["next"]["campaign"] == "done"


def test_the_selection_refuses_a_w_max_other_than_the_issued_campaigns() -> None:
    """w_max is frozen since S1. Under the w_max an issued
    campaign's header recorded the replay goes on; under another, or against a header holding
    no number, it is refused before anything is selected."""
    runs = grids()
    _, _, verdicts = drive(runs)

    def at(w_max, issued):
        return selection(runs, THETA0, w_max, seconds_per_chunk=SPC, numerical=verdicts,
                         facts=commits(runs), reuse=reusable, issued_w_max=issued)  # fmt: skip

    issued = {"adamw_25-b20-1.27e-05/s2000": W_MAX, "frozen_adam_25-b20-1.59e-05/s2001": W_MAX}
    assert at(W_MAX, issued)["next"]["campaign"] == "done"
    refused = at(0.30, issued)
    assert refused["next"] == {"campaign": "incomplete"} and "attempts" not in refused
    assert refused["problems"] == [study2_short.w_max_problems(0.30, issued)[0]]
    assert refused["problems"][0].startswith(
        f"w_max 0.3 differs from the w_max {W_MAX!r} recorded in the header of the issued campaign "
        "of adamw_25-b20-1.27e-05/s2000 and 1 other run(s)")  # fmt: skip
    assert study2_short.w_max_problems(W_MAX, {"x/s1": "0.35"})[0].startswith(
        f"w_max {W_MAX!r} differs from the w_max '0.35'")  # fmt: skip
    assert study2_short.w_max_problems(W_MAX, None) == []


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
# S1 started on 2026-10-20: the run directories before it (YYYYMMDD of their UTC start) are the
# records the replay below was frozen on.
PRE_S1 = "20261020"
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
    """The archived records of the main checkout made before S1 (the 155 runs of 2026-10-19:
    every run directory named by a UTC start before ``PRE_S1``), K, numerics and utility only;
    skipped where they are absent. Later archives (S1, S2 and on) do not change it."""
    if not all((RUNS / name).is_dir() for name in ARCHIVED) or not (RUNS / THETA0_RUN).is_file():
        pytest.skip(f"the archived records are not under {RUNS}")
    dirs = sorted({p.parent for name in ARCHIVED for p in (RUNS / name).rglob("terminal.json")
                   if p.parent.name[:8] < PRE_S1})  # fmt: skip
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
    # Amendment 3's B = 5 history: its targeted round is never reached (B = 20 is the fixed
    # scope), its controls are reported and not used.
    assert len(result["not_reached"]) == 18
    assert all(", 5, " in entry for entry in result["not_reached"])
    reported = [(c["batch"], c["lr"], c["accepted"], c["used"])
                for c in result["amendment3_controls"]]  # fmt: skip
    assert reported == [(5, 1.21e-5, False, False), (5, 1.46e-5, False, False),
                        (5, 1e-5, True, False)]  # fmt: skip
    assert result["stream_time"]["runs"] == 96 and result["stream_time"]["chunks"] == 162


# The archived records up to the S2 repeat: S1's discovery at
# (8, 20), 33 runs of 2026-10-20, and the repeated numerical stage's 15 K-only runs of
# 2026-10-21, whose check (verdicts/f7cf42e3f76f.json under the runs root's numerical/b20-u8/)
# qualified every selected rate. The voided first S2 (runs/void/) is not read.
S1_RUNS = (
    "20261020T040226Z-adamw_25-b20-1.49e-05-s2000-73cfafa6-a1",
    "20261020T040832Z-adamw_25-b20-1.49e-05-s2001-5bc2d505-a1",
    "20261020T041436Z-adamw_25-b20-1.49e-05-s2002-b10bbc13-a1",
    "20261020T042038Z-adamw_25-b20-1.83e-05-s2000-cee4e9e5-a1",
    "20261020T042643Z-adamw_25-b20-1.83e-05-s2001-929321ae-a1",
    "20261020T043248Z-adamw_25-b20-1.83e-05-s2002-106eeaf8-a1",
    "20261020T043850Z-adamw_25-b20-2.24e-05-s2000-2c5b801d-a1",
    "20261020T044451Z-adamw_25-b20-2.24e-05-s2001-e08544cb-a1",
    "20261020T045054Z-adamw_25-b20-2.24e-05-s2002-55e708dc-a1",
    "20261020T045655Z-frozen_adam_25-b20-1.34e-05-s2000-105400dc-a1",
    "20261020T050346Z-frozen_adam_25-b20-1.34e-05-s2001-f015380c-a1",
    "20261020T051043Z-frozen_adam_25-b20-1.34e-05-s2002-6ef5454f-a1",
    "20261020T051743Z-frozen_adam_25-b20-1.62e-05-s2000-9a2393cb-a1",
    "20261020T052449Z-frozen_adam_25-b20-1.62e-05-s2001-c0f54476-a1",
    "20261020T053146Z-frozen_adam_25-b20-1.62e-05-s2002-d230fd18-a1",
    "20261020T053842Z-prewarmed_adamw_25-b20-1.34e-s2000-9f9e017c-a1",
    "20261020T054542Z-prewarmed_adamw_25-b20-1.34e-s2001-72e1875c-a1",
    "20261020T055251Z-prewarmed_adamw_25-b20-1.34e-s2002-0f9be4f6-a1",
    "20261020T060001Z-prewarmed_adamw_25-b20-1.49e-s2000-0c0da153-a1",
    "20261020T060719Z-prewarmed_adamw_25-b20-1.49e-s2001-6c71f0ed-a1",
    "20261020T061434Z-prewarmed_adamw_25-b20-1.49e-s2002-56d96a83-a1",
    "20261020T062148Z-prewarmed_adamw_25-b20-1.62e-s2000-7d51f0e0-a1",
    "20261020T062902Z-prewarmed_adamw_25-b20-1.62e-s2001-edbbad7c-a1",
    "20261020T063630Z-prewarmed_adamw_25-b20-1.62e-s2002-8b9ecc9e-a1",
    "20261020T064338Z-prewarmed_adamw_25-b20-1.83e-s2000-1ecac218-a1",
    "20261020T065056Z-prewarmed_adamw_25-b20-1.83e-s2001-3ff5a95a-a1",
    "20261020T065810Z-prewarmed_adamw_25-b20-1.83e-s2002-22b7fcbc-a1",
    "20261020T070520Z-prewarmed_adamw_25-b20-2e-05-s2000-ab61dde2-a1",
    "20261020T071228Z-prewarmed_adamw_25-b20-2e-05-s2001-b9752381-a1",
    "20261020T071944Z-prewarmed_adamw_25-b20-2e-05-s2002-e9b59593-a1",
    "20261020T072705Z-prewarmed_adamw_25-b20-2.24e-s2000-3dae74aa-a1",
    "20261020T073409Z-prewarmed_adamw_25-b20-2.24e-s2001-94b82083-a1",
    "20261020T074116Z-prewarmed_adamw_25-b20-2.24e-s2002-d723d212-a1",
)
S2_REPEAT_RUNS = (
    "20261021T004244Z-adamw_25-b20-1.49e-05-numeri-s2200-740612f9-a1",
    "20261021T004755Z-adamw_25-b20-1.83e-05-numeri-s2200-bd67150c-a1",
    "20261021T005258Z-adamw_25-b20-2.24e-05-numeri-s2200-1b7ce603-a1",
    "20261021T005802Z-adamw_25-b20-2.24e-05-numeri-s2201-c8845b99-a1",
    "20261021T010304Z-adamw_25-b20-2.24e-05-numeri-s2202-300a9e6c-a1",
    "20261021T010807Z-frozen_adam_25-b20-1.34e-05--s2200-b4bda238-a1",
    "20261021T011357Z-frozen_adam_25-b20-1.62e-05--s2200-7db2ec0a-a1",
    "20261021T011951Z-frozen_adam_25-b20-2e-05-num-s2200-b3099589-a1",
    "20261021T012540Z-frozen_adam_25-b20-2e-05-num-s2201-9dc47a2b-a1",
    "20261021T013132Z-frozen_adam_25-b20-2e-05-num-s2202-690029bf-a1",
    "20261021T013725Z-prewarmed_adamw_25-b20-1.34e-s2200-01a35d98-a1",
    "20261021T014330Z-prewarmed_adamw_25-b20-1.62e-s2200-43a0a9c2-a1",
    "20261021T014932Z-prewarmed_adamw_25-b20-2e-05-s2200-d90e9d26-a1",
    "20261021T015537Z-prewarmed_adamw_25-b20-2e-05-s2201-36e8842c-a1",
    "20261021T020146Z-prewarmed_adamw_25-b20-2e-05-s2202-077b4b1f-a1",
)
S2_REPEAT_COMMIT = "aa4c007c333ef62f7b34b64a3f3c51cee35fa1f9"  # the runs' and the check's commit
S2_REPEAT_VERDICT = "numerical/b20-u8/verdicts/f7cf42e3f76f.json"
# Study 2's last commit (its versions record after amendment 5), as in tests/test_gate_c.py: the
# replays check a verdict's GPU code as of it, not at HEAD, because later studies change GPU paths
# on purpose, and the selection's own check against HEAD then, rightly, finds study 2's
# qualification stale.
STUDY2_LAST = "028b953"


@pytest.fixture
def as_of_study2_last(monkeypatch) -> None:
    """The selection's default ``reuse_check`` compares a verdict's GPU code with ``"HEAD"``;
    here that head is study 2's last commit. Skipped where the commit is not in this clone."""
    probe = subprocess.run(["git", "cat-file", "-e", f"{STUDY2_LAST}^{{commit}}"],
                           cwd=REPO_ROOT, capture_output=True)  # fmt: skip
    if probe.returncode != 0:
        pytest.skip(f"study 2's last commit {STUDY2_LAST} is not in this clone")
    reuse = study2_short._reuse_check
    monkeypatch.setattr(
        study2_short,
        "_reuse_check",
        lambda base, head: reuse(base, STUDY2_LAST if head == "HEAD" else head),
    )


def test_replay_after_the_s2_repeat_prescribes_the_three_arm_validation(
        archived, as_of_study2_last) -> None:  # fmt: skip
    """The selection, replayed on the archived records up to the S2 repeat (the 155 runs before
    S1, S1's 33 and the repeat's 15, read by their recorded run ids; K, numerics and utility
    only) with the repeat's verdict, prescribes the three-arm validation at (U, B) = (8, 20)
    with the frozen rates. The replay checks the verdict's GPU
    code as of study 2's last commit 028b953, because later studies change GPU paths. Skipped
    where the records, the verdict or the commits are absent."""
    runs, theta0 = archived
    names = (*S1_RUNS, *S2_REPEAT_RUNS)
    dirs = [RUNS / "study2_short" / name for name in names]
    if not all((d / "terminal.json").is_file() for d in dirs) or not (
            RUNS / S2_REPEAT_VERDICT).is_file():  # fmt: skip
        pytest.skip(f"S1's and the S2 repeat's records are not under {RUNS}")
    probe = subprocess.run(["git", "cat-file", "-e", f"{S2_REPEAT_COMMIT}^{{commit}}"],
                           cwd=REPO_ROOT, capture_output=True)  # fmt: skip
    if probe.returncode != 0:
        pytest.skip(f"the S2 repeat's commit {S2_REPEAT_COMMIT} is not in this clone")
    later = [run for run in map(_archived, dirs) if run is not None]
    assert [run.run_id for run in later] == list(names)
    facts = {run.run_id: study2_short.run_facts(run.run_dir) for run in later}
    assert {facts[name]["commit"] for name in S2_REPEAT_RUNS} == {S2_REPEAT_COMMIT}
    assert all(facts[name]["k_only"] and facts[name]["role"] == "numerical"
               for name in S2_REPEAT_RUNS)  # fmt: skip
    numerical = u8_verdicts()
    assert S2_REPEAT_VERDICT in numerical
    result = selection([*runs, *later], theta0, 0.47, numerical=numerical, facts=facts,
                       exhausted={})  # fmt: skip
    assert result["problems"] == [] and result["unread_verdicts"] == []
    step = result["next"]
    assert (step["campaign"], step["batch"], step["U"], step["three_arm"]) == (
        "validation", 20, 8.0, True)  # fmt: skip
    assert step["rates"] == {ADAMW: [1.49e-5, 1.83e-5, 2.24e-5], FA: [1.34e-5, 1.62e-5, 2e-5],
                             PREWARMED: [1.34e-5, 1.62e-5, 2e-5]}  # fmt: skip
    # Every arm at its three rates on the twelve validation seeds: 108 runs.
    assert len(step["runs"]) == 108
    assert {run["seed"] for run in step["runs"]} == set(VALIDATION_SEEDS)
    (attempt,) = result["attempts"]
    assert (attempt["outcome"], attempt["failed_arms"], attempt["allowance"]) == (
        "pending", [], {})  # fmt: skip
    (check,) = attempt["numerical"]
    assert (check["verdict"], check["state"], check["unusable"], check["unfinished"]) == (
        S2_REPEAT_VERDICT, "complete", [], [])  # fmt: skip
    assert check["commits"] == [S2_REPEAT_COMMIT] and check["reuse"]["head"]["reusable"]


# S3, the three-arm validation at (8, 20) of 2026-10-22: 108 runs on seeds 2100-2111, the last
# started at 20261022T134346Z, pinned by the sha256 of their sorted run ids.
S3_LAST = "20261022T134346Z"
S3_RUNS_SHA256 = "8fde9f0b3020c039aa6a36a01d64ef896af67c3e4961049c9131b9dae50b2d3b"
S3_SUMMARY = RUNS / "analysis" / "study2-short-summary-20261022-s3.json"


def u8_verdicts() -> dict:
    """The numerical-check verdicts the replays up to S3 knew: those of (8, 20) alone, so that
    a later candidate's verdicts (numerical/b20-u7/ and on) cannot change them."""
    return {path: record for path, record in study2_short.read_numerical(RUNS).items()
            if path.startswith(f"{study2_short.NUMERICAL_DIR}/b20-u8/")}  # fmt: skip


def test_replay_after_s3_prescribes_u7_discovery_for_the_three_arms(
        archived, as_of_study2_last) -> None:  # fmt: skip
    """The selection (amendment 5's order), replayed on the archived records up to S3 (K,
    numerics and utility only) with the S2 repeat's verdict, checking the verdict's GPU code as
    of study 2's last commit 028b953 because later studies change GPU paths, reaches S3's
    verdict at (8, 20): the three-arm validation fails on utility, FA's MMLU bounds 5.79 and
    6.59 pp and the pre-warmed arm's 6.26 pp at 1.62e-5 above 5 pp, support in 12 of 12 blocks.
    The next step stays U = 7's discovery for the three arms, with the same runs: FA at 1.52e-5
    and 1.75e-5 on seeds 2000-2002. Skipped where the records, the verdict or the commits are
    absent."""
    runs, theta0 = archived
    names = (*S1_RUNS, *S2_REPEAT_RUNS)
    dirs = [RUNS / "study2_short" / name for name in names]
    s3 = sorted(p for p in (RUNS / "study2_short").glob("20261022T*")
                if p.name[:16] <= S3_LAST and re.search(r"-s21(0\d|1[01])-", p.name))  # fmt: skip
    if not all((d / "terminal.json").is_file() for d in [*dirs, *s3]) or not (
            RUNS / S2_REPEAT_VERDICT).is_file():  # fmt: skip
        pytest.skip(f"S1's, the S2 repeat's and S3's records are not under {RUNS}")
    digest = hashlib.sha256("\n".join(p.name for p in s3).encode()).hexdigest()
    if (len(s3), digest) != (108, S3_RUNS_SHA256):
        pytest.skip(f"S3's 108 runs are not under {RUNS}")
    probe = subprocess.run(["git", "cat-file", "-e", f"{S2_REPEAT_COMMIT}^{{commit}}"],
                           cwd=REPO_ROOT, capture_output=True)  # fmt: skip
    if probe.returncode != 0:
        pytest.skip(f"the S2 repeat's commit {S2_REPEAT_COMMIT} is not in this clone")
    later = [run for run in map(_archived, [*dirs, *s3]) if run is not None]
    assert len(later) == len(names) + 108
    facts = {run.run_id: study2_short.run_facts(run.run_dir) for run in later}
    result = selection([*runs, *later], theta0, 0.47, numerical=u8_verdicts(),
                       facts=facts, exhausted={})  # fmt: skip
    assert result["problems"] == [] and result["unread_verdicts"] == []
    at8, at7 = result["attempts"]
    assert (at8["U"], at8["three_arm"], at8["outcome"], at8["failed_at"]) == (
        8.0, True, "failed", "validation")  # fmt: skip
    assert at8["failed_arms"] == list(ARMS)
    assert at8["reasons"] == ["the validation does not pass: the utility family does not pass"]
    verdict = at8["validation"]
    assert verdict["passes"] is False and len(verdict["jointly_supported"]) == 12
    bounds = {(m["arm"], m["rate"]): (m["mmlu_upper_pp"], m["factual_upper_pp"])
              for m in verdict["utility"]["family"]}  # fmt: skip
    assert len(bounds) == 9 and verdict["utility"]["alpha_per_bound"] == pytest.approx(0.05 / 9)
    failing = sorted(key for key, (mmlu, factual) in bounds.items() if mmlu > 5 or factual > 10)
    assert failing == [(FA, 1.62e-5), (FA, 2e-5), (PREWARMED, 1.62e-5)]
    assert bounds[(FA, 1.62e-5)][0] == pytest.approx(5.79, abs=0.005)
    assert bounds[(FA, 2e-5)][0] == pytest.approx(6.59, abs=0.005)
    assert bounds[(PREWARMED, 1.62e-5)][0] == pytest.approx(6.26, abs=0.005)
    if S3_SUMMARY.is_file():  # replay invariance, exactly: S3's own summary of 2026-10-22
        s3 = _numeric(S3_SUMMARY.read_text())["progress"]["selection"]["attempts"][0]
        assert (s3["U"], s3["three_arm"]) == (8.0, True)
        assert json.loads(json.dumps(verdict["utility"])) == s3["validation"]["utility"]
        assert verdict["jointly_supported"] == s3["validation"]["jointly_supported"]
    # The pair's recomputation on S3's runs (amendment 5, item 2): six bounds, FA failing.
    pair = study2_short.pair_recomputation(at8, [*runs, *later], theta0, 0.47)
    recomputed = pair["verdict"]
    assert pair["rates"] == {ADAMW: [1.49e-5, 1.83e-5, 2.24e-5], FA: [1.34e-5, 1.62e-5, 2e-5]}
    assert recomputed["passes"] is False and len(recomputed["jointly_supported"]) == 12
    assert recomputed["reasons"] == ["the utility family does not pass"]
    family = {(m["arm"], m["rate"]): m["mmlu_upper_pp"] for m in recomputed["utility"]["family"]}
    assert len(family) == 6 and recomputed["utility"]["alpha_per_bound"] == pytest.approx(0.05 / 6)
    assert round(family[(FA, 1.62e-5)], 4) == 5.6195 and round(family[(FA, 2e-5)], 4) == 6.3987
    assert [key for key, mmlu in sorted(family.items()) if mmlu > 5] == [(FA, 1.62e-5), (FA, 2e-5)]
    step = result["next"]
    assert (step["campaign"], step["batch"], step["U"], step["three_arm"]) == (
        "discovery", 20, 7.0, True)  # fmt: skip
    assert campaign_runs(step) == {(FA, 20, lr, s) for lr in (1.52e-5, 1.75e-5) for s in SEEDS}
    assert (at7["U"], at7["three_arm"], at7["outcome"]) == (7.0, True, "pending")


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


# ------------------------------------------------------------------ the time course


def recorded(tmp_path: Path, name: str, evaluate_at: tuple[int, ...], *, other: float = 0.0):
    """A 25-step run with the records the identity check reads: metrics.jsonl (the training rows,
    the monitor's every 5 steps, the scoring diagnostics' at step 25 and the evaluations', logged
    one step back), monitor.jsonl (the diagnostics' K beside the reading at step 25),
    update_scalars.jsonl (a cosine of 0.0 at step 3), meta.json's streams and the evaluations.
    ``other`` moves only what the check leaves out: timings, memory peaks and K_harmful."""
    run_dir = tmp_path / name
    run_dir.mkdir()

    def k(t: int) -> float:
        return 6.3 * t / 25

    diagnostics = {"K_scoring": 6.25, "K_monitor_full": 6.5}  # step 25's
    metrics = []
    for step in range(25):
        done, time_s = step + 1, 0.5 * step + other
        metrics.append({"step": step, "time_s": time_s, "loss": 1 / done, "denominator": 480,
                        "tokens": 900 + step, "nll_per_token": 0.5 / done,
                        "grad_norm": 0.25 + 0.01 * step})  # fmt: skip
        if done == 25:
            metrics.append({"step": step, "time_s": time_s, **diagnostics})
        if done % 5 == 0:
            metrics.append({"step": step, "time_s": time_s, "K_monitor": k(done)})
        if done in evaluate_at:
            metrics.append({"step": step, "time_s": time_s, "eval/K_benign": k(done),
                            "eval/K_harmful": 2 * k(done) + other, "eval/benchmark_accuracy": 0.6,
                            "eval/heldout_nll_per_token": 1.125,
                            "eval/peak_allocated_gb": 30 + other})  # fmt: skip
    records = {
        "metrics.jsonl": metrics,
        "monitor.jsonl": [{"step": t, "K_monitor": k(t), "grad_norm": 0.3,
                           "displacement_norm": 0.01 * t, **(diagnostics if t == 25 else {})}
                          for t in range(5, 26, 5)],
        "update_scalars.jsonl": [{"step": t, "update_norm": 1e-3 * t,
                                  "cos_update_grad": 0.0 if t == 3 else 0.5,
                                  # the denominator's departure, as the pre-warmed arm records it
                                  **({"d_departure": {"norm_ratio": 0.01 * t, "median": 1e-3 * t,
                                                      "p90": 2e-3 * t}}
                                     if t in (5, 10, 25) else {})}
                                 for t in range(1, 26)],
    }  # fmt: skip
    for record, rows in records.items():
        (run_dir / record).write_text("".join(json.dumps(r) + "\n" for r in rows))
    stream = {"path": "teacher/m/training/seed-3000", "count": 500, "sha256": "ab", "chunks": ["c"]}
    (run_dir / "meta.json").write_text(json.dumps({"inputs": {"streams": {"training": stream}}}))
    evaluations = [{"step": t, "condition": f"step-{t}", "levels": [],
                    "K_monitor": k(t) if t % 5 == 0 else None, "K_benign": k(t),
                    "K_harmful": 2 * k(t) + other, "benchmark_accuracy": 0.6,
                    "factual_accuracy": 0.8, "heldout_nll_per_token": 1.125,
                    "peak_allocated_gb": 30 + other}
                   for t in evaluate_at]  # fmt: skip
    run = traj(ADAMW, 20, 1.48e-5, 3000, None, name=name)
    return dataclasses.replace(run, run_dir=run_dir, evaluations=evaluations,
                               evaluate_at=evaluate_at)  # fmt: skip


def tolerance_away(x: float, factor: float) -> float:
    """y > x > 0 with y − x = factor·(abs + rel·max(|x|, |y|)): ``factor`` times twin A's
    tolerance."""
    rule = study2_short.NUMERICAL_POLICY["twin_a_consistency"]
    return (x + factor * rule["abs"]) / (1 - factor * rule["rel"])


def test_timecourse_identity(tmp_path) -> None:
    """A time-course run against its endpoint run (amendment 4, section 4: numerical consistency):
    identical whatever the timings, memory peaks and K_harmful (never
    read); consistent when only the named continuous fields differ, within twin A's tolerance;
    excluded just over it, on a value not finite, on any difference in a discrete outcome (the
    accuracies, the departure's median and p90), or on a one-bit difference in any other field;
    pending without its endpoint run."""
    from zero_field_refusal.analysis.study2_short import timecourse_identity

    endpoint = recorded(tmp_path, "endpoint", (25,))
    course = recorded(tmp_path, "course", (1, 5, 10, 25), other=0.125)
    assert timecourse_identity(course, endpoint) == {"state": "identical", "reasons": [],
                                                     "endpoint": "endpoint", "bitwise_equal": True,
                                                     "rayleigh_equal": None}  # fmt: skip
    waiting = ["its endpoint run is not there yet"]
    pending = {"state": "pending", "reasons": waiting, "endpoint": None, "bitwise_equal": None,
               "rayleigh_equal": None}  # fmt: skip
    assert timecourse_identity(course, None) == pending

    def edited(record: str, edit, run: Trajectory = course) -> Trajectory:
        """A run with one record edited (``edit`` None: removed), in a copy of its directory."""
        copy = tmp_path / f"{run.run_id}-{len(list(tmp_path.iterdir()))}"
        shutil.copytree(run.run_dir, copy)
        if edit is None:
            (copy / record).unlink()
        else:
            rows = [json.loads(line) for line in (copy / record).read_text().splitlines()]
            edit(rows)
            (copy / record).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return dataclasses.replace(run, run_dir=copy, run_id=copy.name)

    def bump(row: dict, key: str) -> None:  # one bit: the next float up
        row[key] = math.nextafter(row[key], math.inf)

    def move(row: dict, key: str, factor: float) -> None:  # ``factor`` times the tolerance
        row[key] = tolerance_away(row[key], factor)

    def training(rows: list, step: int) -> dict:
        return next(r for r in rows if r["step"] == step and "loss" in r)

    def monitored(rows: list, step: int) -> dict:  # the monitor's K, logged in metrics.jsonl
        return next(r for r in rows if r["step"] == step and "K_monitor" in r)

    def departure(rows: list, key: str, factor: float) -> None:
        rows[9]["d_departure"][key] = tolerance_away(rows[9]["d_departure"][key], factor)

    def several(*edits):
        def edit(rows: list) -> None:
            for one in edits:
                one(rows)

        return edit

    final = course.evaluations[-1]

    def evaluated(**changes) -> Trajectory:  # the step-25 evaluation changed
        return dataclasses.replace(course, evaluations=[*course.evaluations[:3],
                                                        {**final, **changes}])  # fmt: skip

    def diagnosed(rows: list) -> dict:  # the scoring diagnostics' row in metrics.jsonl
        return next(r for r in rows if "K_scoring" in r)

    # Each named continuous field half the tolerance away is consistent
    consistent = {
        "training": edited("metrics.jsonl", several(
            lambda rows: move(training(rows, 7), "grad_norm", 0.5),
            lambda rows: move(training(rows, 3), "loss", 0.5),
            lambda rows: move(training(rows, 24), "nll_per_token", 0.5),
            lambda rows: move(monitored(rows, 9), "K_monitor", 0.5),
            lambda rows: move(diagnosed(rows), "K_scoring", 0.5),
            lambda rows: move(diagnosed(rows), "K_monitor_full", 0.5))),
        "one bit": edited("metrics.jsonl", lambda rows: bump(training(rows, 7), "grad_norm")),
        "the evaluation's row": edited("metrics.jsonl", several(
            lambda rows: move(rows[-1], "eval/K_benign", 0.5),
            lambda rows: move(rows[-1], "eval/heldout_nll_per_token", 0.5))),
        "monitor": edited("monitor.jsonl", several(
            lambda rows: move(rows[2], "K_monitor", 0.5),
            lambda rows: move(rows[2], "grad_norm", 0.5),
            lambda rows: move(rows[3], "displacement_norm", 0.5),
            lambda rows: move(rows[4], "K_scoring", 0.5),
            lambda rows: move(rows[4], "K_monitor_full", 0.5))),
        "monitor, one bit": edited("monitor.jsonl", lambda rows: bump(rows[2], "grad_norm")),
        "update scalars": edited("update_scalars.jsonl", several(
            lambda rows: move(rows[4], "update_norm", 0.5),
            lambda rows: move(rows[4], "cos_update_grad", 0.5),
            lambda rows: departure(rows, "norm_ratio", 0.5))),
        "negative zero": edited("update_scalars.jsonl",
                                lambda rows: rows[2].update(cos_update_grad=-0.0)),
        "the evaluation": evaluated(K_benign=tolerance_away(final["K_benign"], 0.5),
                                    K_monitor=tolerance_away(final["K_monitor"], 0.5),
                                    heldout_nll_per_token=tolerance_away(1.125, 0.5)),
    }  # fmt: skip
    for name, run in consistent.items():
        found = timecourse_identity(run, endpoint)
        assert found == {"state": "consistent", "reasons": [], "endpoint": "endpoint",
                         "bitwise_equal": False, "rayleigh_equal": None}, name  # fmt: skip
    cases = {
        # Just over the tolerance. Step 7's training row is the 9th outside the evaluations
        # (step 4's K_monitor row first)
        "the training's metrics.jsonl rows are not numerically consistent with the endpoint "
        "run's from row 8 (31 rows against 31): grad_norm": edited(
            "metrics.jsonl", lambda rows: move(training(rows, 7), "grad_norm", 1.01)),
        "the metrics.jsonl rows of the endpoint's evaluations are not numerically consistent": (
            edited("metrics.jsonl", lambda rows: move(rows[-1], "eval/K_benign", 1.01))),
        "the monitor.jsonl readings are not numerically consistent with the endpoint run's from "
        "row 3 (5 rows against 5): displacement_norm": edited(
            "monitor.jsonl", lambda rows: move(rows[3], "displacement_norm", 1.01)),
        "the update_scalars.jsonl rows are not numerically consistent with the endpoint run's "
        "from row 9 (25 rows against 25): d_departure.norm_ratio": edited(
            "update_scalars.jsonl", lambda rows: departure(rows, "norm_ratio", 1.01)),
        "the endpoint's evaluations are not numerically consistent": evaluated(
            K_benign=tolerance_away(final["K_benign"], 1.01)),
        # A value not finite
        "loss nan against 0.16666666666666666, outside the tolerance": edited(
            "metrics.jsonl", lambda rows: training(rows, 5).update(loss=math.nan)),
        "update_norm inf against 0.005, outside the tolerance": edited(
            "update_scalars.jsonl", lambda rows: rows[4].update(update_norm=math.inf)),
        # The discrete outcomes exactly, even inside the tolerance: the accuracies
        # (k/N) and the departure's median and p90 (a histogram's bin edges)
        "eval/benchmark_accuracy 0.6000000000000001 against 0.6": edited(
            "metrics.jsonl", lambda rows: bump(rows[-1], "eval/benchmark_accuracy")),
        f"benchmark_accuracy {tolerance_away(0.6, 0.5)!r} against 0.6": evaluated(
            benchmark_accuracy=tolerance_away(0.6, 0.5)),
        "factual_accuracy 0.8000000000000002 against 0.8": evaluated(
            factual_accuracy=math.nextafter(0.8, 1.0)),
        "from row 9 (25 rows against 25): d_departure.median ": edited(
            "update_scalars.jsonl", lambda rows: departure(rows, "median", 0.5)),
        "from row 9 (25 rows against 25): d_departure.p90 ": edited(
            "update_scalars.jsonl", lambda rows: departure(rows, "p90", 0.5)),
        # Every other field exactly: a count, a step, a discrete outcome, the stop, the streams
        "tokens 907 against 906": edited(
            "metrics.jsonl", lambda rows: training(rows, 6).update(tokens=907)),
        "step 7 against 8": edited("update_scalars.jsonl", lambda rows: rows[7].update(step=7)),
        "levels [8.0] against []": evaluated(levels=[8.0]),
        "condition 'step-24' against 'step-25'": evaluated(condition="step-24"),
        "the streams read are not numerically consistent": edited(
            "meta.json", lambda rows: rows[0]["inputs"]["streams"]["training"].update(count=480)),
        "the stop and the steps taken are not numerically consistent": dataclasses.replace(
            course, terminal={**course.terminal, "stop": {"reason": "gradient_spike"}}),
        "a row missing": edited("update_scalars.jsonl", lambda rows: rows.pop()),
        "is of ('adamw_25', 20, 1.48e-05, 3001)": dataclasses.replace(endpoint, seed=3001),
        "a record does not read": edited("update_scalars.jsonl", lambda rows: None),
        # A missing record never compares as equal.
        **{f"has no {record}": edited(record, None)
           for record in ("metrics.jsonl", "monitor.jsonl", "update_scalars.jsonl", "meta.json")},
    }  # fmt: skip
    broken = tmp_path / cases["a record does not read"].run_id / "update_scalars.jsonl"
    broken.write_text(broken.read_text() + '{"step": 26, "upd\n')
    for reason, run in cases.items():
        mine, theirs = (course, run) if reason.startswith("is of") else (run, endpoint)
        found = timecourse_identity(mine, theirs)
        assert found["state"] == "excluded", reason
        assert len(found["reasons"]) == 1 and reason in found["reasons"][0], found["reasons"]
        assert found["bitwise_equal"] is (None if reason.startswith(("is of", "a record", "has no"))
                                          else False), reason  # fmt: skip
    # Missing from both runs: still excluded, never identical.
    both = timecourse_identity(cases["has no update_scalars.jsonl"],
                               edited("update_scalars.jsonl", None, endpoint))  # fmt: skip
    assert both["state"] == "excluded" and "a record does not read" in both["reasons"][0]

    # NaN on both sides: bit for bit alike, but never consistent, so excluded (the
    # rule applies to bit-equal records too)
    def nan(rows: list) -> None:
        rows[2]["cos_update_grad"] = math.nan

    both_nan = edited("update_scalars.jsonl", nan, endpoint)
    found = timecourse_identity(edited("update_scalars.jsonl", nan), both_nan)
    assert (found["state"], found["bitwise_equal"]) == ("excluded", True)
    assert found["reasons"] == [
        "the update_scalars.jsonl rows are not numerically consistent with the endpoint run's "
        "from row 2 (25 rows against 25): cos_update_grad nan against nan, outside the "
        "tolerance"]  # fmt: skip
    found = timecourse_identity(edited("update_scalars.jsonl", several(
        nan, lambda rows: move(rows[4], "update_norm", 0.5))), both_nan)  # fmt: skip
    assert (found["state"], found["bitwise_equal"]) == ("excluded", False)
    assert found["reasons"] == [
        "the update_scalars.jsonl rows are not numerically consistent with the endpoint run's "
        "from row 2 (25 rows against 25): cos_update_grad nan against nan, outside the "
        "tolerance"]  # fmt: skip
    # The same for ∞ on both sides, in every record that holds a toleranced field: the
    # training's rows, their evaluation's row, the monitor, the update scalars, the evaluation.
    both_inf = {
        "the training's metrics.jsonl rows": ("metrics.jsonl", lambda rows: training(
            rows, 12).update(grad_norm=math.inf)),
        "the metrics.jsonl rows of the endpoint's evaluations": ("metrics.jsonl", lambda rows:
                                                                  rows[-1].update(
                                                                      {"eval/K_benign":
                                                                       math.inf})),
        "the monitor.jsonl readings": ("monitor.jsonl", lambda rows: rows[1].update(
            K_monitor=-math.inf)),
        "the update_scalars.jsonl rows": ("update_scalars.jsonl", lambda rows: rows[9][
            "d_departure"].update(norm_ratio=math.inf)),
    }  # fmt: skip
    for name, (record, edit) in both_inf.items():
        found = timecourse_identity(edited(record, edit), edited(record, edit, endpoint))
        assert (found["state"], found["bitwise_equal"]) == ("excluded", True), name
        assert len(found["reasons"]) == 1 and found["reasons"][0].startswith(name), name
        assert "inf against " in found["reasons"][0] and "outside the tolerance" in found[
            "reasons"][0]  # fmt: skip
    evaluation = [*endpoint.evaluations[:-1], {**endpoint.evaluations[-1],
                                               "heldout_nll_per_token": math.inf}]  # fmt: skip
    found = timecourse_identity(
        dataclasses.replace(course, evaluations=[*course.evaluations[:3], evaluation[-1]]),
        dataclasses.replace(endpoint, evaluations=evaluation))  # fmt: skip
    assert (found["state"], found["bitwise_equal"]) == ("excluded", True)
    assert found["reasons"][0].startswith("the endpoint's evaluations are not numerically")
    # A finite run bit-equal to its endpoint stays identical.
    assert timecourse_identity(course, endpoint)["state"] == "identical"
    # A value inside the tolerance before a field that is not: excluded from that row, its reason
    mixed = edited(
        "metrics.jsonl",
        several(
            lambda rows: move(training(rows, 7), "loss", 0.5),
            lambda rows: training(rows, 9).update(tokens=1),
        ),
    )
    found = timecourse_identity(mixed, endpoint)
    assert found["state"] == "excluded" and len(found["reasons"]) == 1
    assert "from row 10" in found["reasons"][0] and "tokens 1 against 909" in found["reasons"][0]
    # The time course's own evaluations (steps 1, 5, 10) are not the endpoint's to compare.
    early = dataclasses.replace(course, evaluations=[
        {**course.evaluations[0], "K_benign": 99.0}, *course.evaluations[1:]])  # fmt: skip
    assert timecourse_identity(early, endpoint)["state"] == "identical"


def test_a_deviation_like_s2s_is_consistent(tmp_path) -> None:
    """A time-course run whose training departs
    from its endpoint run's at the fp32 level from step 11 on, as S2's twins did (relative
    deviations up to 3.9e-6), the deviation carried into every continuous number after it (the
    gradient's and displacement's norms, the monitor's and the diagnostics' K, the update
    scalars, the step-25 evaluation's K and held-out NLL), is consistent. The departure's median
    and p90 (bin edges) stay as they were; one changed answer (an accuracy) excludes the run."""
    from zero_field_refusal.analysis.study2_short import timecourse_identity

    endpoint = recorded(tmp_path, "endpoint", (25,))
    course = recorded(tmp_path, "course", (1, 5, 10, 25))
    run_dir = tmp_path / "deviating"
    shutil.copytree(course.run_dir, run_dir)

    def scale(row: dict, factors: dict) -> None:
        for key, rel in factors.items():
            if key in row:
                row[key] *= 1 + rel

    def metrics(row: dict) -> None:
        if row["step"] >= 11:
            scale(row, {"loss": 3.1e-7, "grad_norm": 3.88e-6, "nll_per_token": 3.1e-7,
                        "K_monitor": 1.37e-6, "K_scoring": 1e-6, "K_monitor_full": 1.2e-6,
                        "eval/K_benign": 1e-6, "eval/heldout_nll_per_token": 2e-7})  # fmt: skip

    def monitor(row: dict) -> None:
        if row["step"] >= 15:
            scale(row, {"K_monitor": 1.37e-6, "grad_norm": 3.5e-6, "displacement_norm": 2e-6,
                        "K_scoring": 1e-6, "K_monitor_full": 1.2e-6})  # fmt: skip

    def scalars(row: dict) -> None:
        if row["step"] >= 12:
            scale(row, {"update_norm": 2e-6, "cos_update_grad": 1e-6})
            scale(row.get("d_departure", {}), {"norm_ratio": 1e-6})

    for name, edit in (("metrics.jsonl", metrics), ("monitor.jsonl", monitor),
                       ("update_scalars.jsonl", scalars)):  # fmt: skip
        rows = [json.loads(line) for line in (run_dir / name).read_text().splitlines()]
        for row in rows:
            edit(row)
        (run_dir / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    final = dict(course.evaluations[-1])
    scale(final, {"K_benign": 1e-6, "K_monitor": 1.37e-6, "heldout_nll_per_token": 2e-7})
    deviating = dataclasses.replace(course, run_dir=run_dir, run_id="deviating",
                                    evaluations=[*course.evaluations[:3], final])  # fmt: skip
    assert timecourse_identity(deviating, endpoint) == {
        "state": "consistent", "reasons": [], "endpoint": "endpoint", "bitwise_equal": False,
        "rayleigh_equal": None}  # fmt: skip
    answered = dataclasses.replace(deviating, evaluations=[
        *course.evaluations[:3], {**final, "factual_accuracy": 0.8 + 1 / 250}])  # fmt: skip
    found = timecourse_identity(answered, endpoint)
    assert found["state"] == "excluded" and found["reasons"] == [
        "the endpoint's evaluations are not numerically consistent with the endpoint run's from "
        "row 0 (1 rows against 1): factual_accuracy 0.804 against 0.8"]  # fmt: skip


def test_timecourse_identity_reports_the_rayleigh_records_beside(tmp_path) -> None:
    """The endpoint Rayleigh records (R, ‖δ‖, δᵀH₀δ) are compared beside the identity, without
    their timings and peaks, and never decide it: None when either run
    lacks one or its measurement failed."""
    from zero_field_refusal.analysis.study2_short import timecourse_identity

    endpoint = recorded(tmp_path, "endpoint", (25,))
    course = recorded(tmp_path, "course", (1, 5, 10, 25))
    measured = {"R": 12.5, "delta_norm": 0.031, "delta_H_delta": 0.0120125, "panel": {},
                "cap": 256, "micro_batch_size": 4}  # fmt: skip

    def write(run: Trajectory, **changes) -> None:
        record = {**measured, **changes}
        (run.run_dir / "rayleigh.json").write_text(json.dumps(record, indent=1) + "\n")

    def found() -> dict:
        return timecourse_identity(course, endpoint)

    write(endpoint, seconds=40.0, peak_allocated_gb=31.2)
    assert found()["rayleigh_equal"] is None  # the time course has none
    write(course, seconds=41.5, peak_allocated_gb=33.0)
    assert found() == {"state": "identical", "reasons": [], "endpoint": "endpoint",
                       "bitwise_equal": True, "rayleigh_equal": True}  # fmt: skip
    write(course, delta_H_delta=math.nextafter(measured["delta_H_delta"], 1.0))
    assert found()["rayleigh_equal"] is False and found()["state"] == "identical"
    write(course, R=None, reason="failed: OutOfMemoryError: …", delta_norm=None,
          delta_H_delta=None)  # fmt: skip
    assert found()["rayleigh_equal"] is None


# ------------------------------------------------------------------ the mechanism cohort

MECHANISM = {"target": 60, "cap": 80, "minimum": 36, "zero_se": 1e-12, "comparison_level": 0.975,
             "effects": [0.05, 0.025], "level": 0.95, "resamples": 2000, "seed": 7}  # fmt: skip


def reference_abs_t(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """|T*| per draw written out directly, for draws without a zero SE*."""
    n = values.size
    samples = values[indices]
    se = samples.std(axis=1, ddof=1) / math.sqrt(n)
    return np.abs(samples.mean(axis=1) - values.mean()) / se


def test_max_t_uses_shared_draws() -> None:
    """Every draw resamples one set of block indices, shared by both contrasts (drawn as
    ``bootstrap_t_interval`` draws them), and for c1 = c2 the critical value is the one-contrast
    one."""
    rng = np.random.default_rng(3)
    c1, c2 = rng.normal(0.02, 0.05, 40), rng.normal(-0.01, 0.03, 40)
    found = max_t_interval(c1, c2, 0.95, 2000, 11, keep=True)
    indices = np.random.default_rng(11).integers(0, 40, (2000, 40))
    for values, t_star in zip((c1, c2), found["t_star"], strict=True):
        np.testing.assert_array_equal(t_star, reference_abs_t(values, indices))
    maxima = np.maximum(*found["t_star"])
    assert found["q"] == pytest.approx(float(np.quantile(maxima, 0.95)), rel=1e-12, abs=0)
    assert (found["draws"], found["infinite"]) == (2000, 0)
    for one, values in zip(found["contrasts"], (c1, c2), strict=True):
        se = values.std(ddof=1) / math.sqrt(40)
        assert one["interval"] == [one["mean"] - found["q"] * se, one["mean"] + found["q"] * se]
    # c2 = −c1: |T₂*| = |T₁*| draw for draw, which only shared indices give.
    mirrored = max_t_interval(c1, -c1, 0.95, 2000, 11, keep=True)
    np.testing.assert_array_equal(*mirrored["t_star"])
    one = symmetric_studentized(c1, 0.95, 2000, 11)
    same = max_t_interval(c1, c1, 0.95, 2000, 11)
    assert same["q"] == mirrored["q"] == one["q"]
    assert same["contrasts"][0]["interval"] == one["interval"]
    # Two contrasts that are not perfectly correlated need a larger critical value.
    assert found["q"] > one["q"]


def test_max_t_reduces_to_symmetric_studentized() -> None:
    """With a degenerate c2 (|T₂*| = 0 in every draw), the max-t interval of c1 is
    ``symmetric_studentized``'s at the same level, resamples and seed; that is the symmetric
    studentized bootstrap computed directly."""
    c1 = np.random.default_rng(5).normal(0.03, 0.06, 36)
    flat = np.full(36, 0.02)
    found = max_t_interval(c1, flat, 0.9, 3000, 21, keep=True)
    assert not found["t_star"][1].any()
    one = symmetric_studentized(c1, 0.9, 3000, 21, keep=True)
    assert found["q"] == one["q"] and found["contrasts"][0]["interval"] == one["interval"]
    assert found["contrasts"][1]["zero_se"] and found["contrasts"][1]["interval"] is None
    indices = np.random.default_rng(21).integers(0, 36, (3000, 36))
    np.testing.assert_array_equal(one["t_star"], reference_abs_t(c1, indices))
    q = float(np.quantile(reference_abs_t(c1, indices), 0.9))
    se = c1.std(ddof=1) / 6
    assert one["q"] == pytest.approx(q, rel=1e-12, abs=0)
    assert one["interval"] == pytest.approx([c1.mean() - q * se, c1.mean() + q * se], rel=1e-12)
    assert (one["mean"], one["se"]) == (float(c1.mean()), float(se))
    # Malformed settings are refused; fewer than 2 blocks give no interval.
    for bad in ((1.0, 3000, 21), (0.9, 1, 21), (0.9, 3000.0, 21), (0.9, 3000, "21")):
        with pytest.raises(ValueError):
            symmetric_studentized(c1, *bad)
    with pytest.raises(ValueError, match="pairs"):
        max_t_interval(c1, c1[:-1], 0.9, 3000, 21)
    assert symmetric_studentized([0.1], 0.9, 3000, 21)["interval"] is None


def test_the_zero_se_rule_in_a_resample() -> None:
    """A resample's contrast whose SE* ≤ 1e-12 has |T*| = 0 when its mean is the observed one
    within 1e-12, ∞ otherwise; a quantile reaching an ∞ draw is ∞, and gives no interval."""
    samples = np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.0, 0.5, 0.5],
                        [0.5 + 1e-13, 0.5 + 1e-13, 0.5 + 1e-13]])  # fmt: skip
    found = study2_short._abs_studentized(samples, 0.5)
    third = (0.5 - 1 / 3) / (np.std([0.0, 0.5, 0.5], ddof=1) / math.sqrt(3))
    assert found[0] == 0.0 and found[1] == math.inf and found[3] == 0.0
    assert found[2] == pytest.approx(third)
    ordered = np.array([0.0, 1.0, 2.0, math.inf])
    assert study2_short._upper_quantile(ordered, 0.5) == 1.5
    assert study2_short._upper_quantile(ordered, 2 / 3) == 2.0  # exactly on a finite draw
    assert study2_short._upper_quantile(ordered, 0.7) == math.inf
    # Two blocks: a draw of the same block twice has SE* = 0 and its mean elsewhere: ∞.
    pair = max_t_interval([0.0, 1.0], [0.2, 0.4], 0.95, 4000, 3)
    indices = np.random.default_rng(3).integers(0, 2, (4000, 2))
    assert pair["infinite"] == int((indices[:, 0] == indices[:, 1]).sum()) > 1000
    assert pair["q"] == math.inf and [c["interval"] for c in pair["contrasts"]] == [None, None]


def mechanism_entry(number: int, h: dict, *, joint: bool = True, missing: float = 0.0,
                    labeled: bool = True) -> dict:  # fmt: skip
    """A ``mechanism_blocks`` entry: complete on K, h per arm (pre-warmed without one when the
    block is not jointly supported), the high bound ``missing`` above h."""
    arms = h if joint else {arm: v for arm, v in h.items() if arm != PREWARMED}
    return {"block": number, "seed": 2999 + number, "complete": labeled, "supported": True,
            "delta": arms[ADAMW] - arms[FA], "h": arms if labeled else {}, "k_complete": True,
            "three_arm_supported": joint, "primary_supported": True,
            "unsupported": {} if joint else {PREWARMED: "bracketing"},
            "h_missing_harmful": {arm: v + missing for arm, v in arms.items()}
            if labeled else {}}  # fmt: skip


def components(n: int, c1: float, c2: float, *, sd: float = 0.01, seed: int = 1) -> list[dict]:
    """h per arm of ``n`` blocks with C1 = P − F ≈ c1 and C2 = A − P ≈ c2."""
    rng = np.random.default_rng(seed)
    out = []
    for f, e1, e2 in zip(rng.normal(0.2, 0.03, n), rng.normal(0, sd, n), rng.normal(0, sd, n),
                         strict=True):  # fmt: skip
        p = f + c1 + e1
        out.append({FA: float(f), PREWARMED: float(p), ADAMW: float(p + c2 + e2)})
    return out


def test_minimum_cohort_36() -> None:
    """With the cap reached at fewer than 36 jointly supported blocks, the intervals are
    descriptive and nothing is read; from 36 on, the readings."""
    for joint, read in ((35, False), (36, True)):
        h = components(80, 0.05, -0.05)
        blocks = [mechanism_entry(i + 1, h[i], joint=i < joint) for i in range(80)]
        found = mechanism_analysis(blocks, MECHANISM)
        assert found["cohort"]["n"] == joint and found["cohort"]["ended_by"] == "cap"
        assert found["cohort"]["window"] == [1, 80] and found["minimum_met"] is read
        assert all(found["max_t"][name] is not None for name in ("C1", "C2"))  # descriptive
        for name in ("C1", "C2"):
            reading = found["readings"][name]
            assert (reading["readings"] is not None) is read
            assert (reading["basis"] == "max_t") is read
        assert found["attrition"]["blocks"] == 80 - joint
        assert found["attrition"]["unsupported_arms"] == {PREWARMED: 80 - joint}
        assert found["attrition"]["primary_supported"] == 80 - joint
    assert found["readings"]["C1"]["readings"] == ["positive"]
    # Before the cohort's end it is refused: a block not complete, or a cohort block unlabeled.
    with pytest.raises(ValueError, match="has not ended"):
        mechanism_analysis(blocks[:79], MECHANISM)
    with pytest.raises(ValueError, match="not complete"):
        mechanism_analysis([*blocks[:40], {**blocks[40], "k_complete": False}, *blocks[41:]],
                           MECHANISM)  # fmt: skip
    with pytest.raises(ValueError, match="not labeled"):
        mechanism_analysis([mechanism_entry(1, h[0], labeled=False), *blocks[1:]], MECHANISM)
    with pytest.raises(ValueError, match="mechanism minimum"):
        mechanism_analysis(blocks, {**MECHANISM, "minimum": 35})


def test_zero_se_no_reading() -> None:
    """An observed SE ≤ 1e-12 leaves its contrast unread, and the other is read on its symmetric
    studentized interval at 97.5 %."""
    h = components(60, 0.04, 0.0)
    # C2 = A − P equal to 0.01 in every block, up to rounding: a zero SE.
    flat = [{**row, ADAMW: row[PREWARMED] + 0.01} for row in h]
    found = mechanism_analysis([mechanism_entry(i + 1, row) for i, row in enumerate(flat)],
                               MECHANISM)  # fmt: skip
    assert found["cohort"]["n"] == 60 and found["cohort"]["ended_by"] == "target"
    assert found["C2"]["se"] <= 1e-12 and found["max_t"]["C2"] is None
    assert found["readings"]["C2"] == {
        "basis": None,
        "interval": None,
        "readings": None,
        "why": "an observed SE at or below ZERO_SE: not read",
    }
    c1 = [row[PREWARMED] - row[FA] for row in flat]
    stream = derive_seed(7, "mechanism", "bonferroni", "C1")
    comparison = symmetric_studentized(c1, 0.975, 2000, stream)
    assert found["bonferroni"]["C1"]["interval"] == comparison["interval"]
    assert found["readings"]["C1"]["basis"] == "bonferroni"
    assert found["readings"]["C1"]["interval"] == comparison["interval"]
    assert found["readings"]["C1"]["readings"] == ["positive", "bounded"]
    # The max-t interval of C1 there is the one-contrast one at the max-t level (|T₂*| = 0).
    joint = symmetric_studentized(c1, 0.95, 2000, derive_seed(7, "mechanism", "max_t"))
    assert found["max_t"]["C1"] == joint["interval"]


def test_sum_identity_on_cohort() -> None:
    """C1 + C2 = AdamW − FA on the cohort, per block and in the mean, to 1e-12; that is not the
    primary's mean, whose cohort holds the blocks the pre-warmed arm lost."""
    h = components(75, 0.02, 0.01, sd=0.02)
    lost = set(range(4, 70, 5))  # 14 blocks without the pre-warmed arm, A − F large there
    rows = [{**row, ADAMW: row[ADAMW] + 0.2} if i in lost else row for i, row in enumerate(h)]
    blocks = [mechanism_entry(i + 1, row, joint=i not in lost) for i, row in enumerate(rows)]
    found = mechanism_analysis(blocks, MECHANISM)
    # The window ends at the 60th jointly supported block, 74: block 75 is no part of it.
    rows = rows[:74]
    cohort = [row for i, row in enumerate(rows) if i not in lost]
    assert found["cohort"]["n"] == 60 and found["cohort"]["window"] == [1, 74]
    assert found["cohort"]["blocks"] == [i + 1 for i in range(74) if i not in lost]
    check = found["sum_check"]
    on_cohort = statistics.fmean(row[ADAMW] - row[FA] for row in cohort)
    assert check["holds"] and check["max_block_deviation"] <= 1e-12
    assert check["adamw_minus_fa_mean"] == pytest.approx(on_cohort, abs=1e-15)
    assert abs(found["C1"]["mean"] + found["C2"]["mean"] - on_cohort) <= 1e-12
    primary = statistics.fmean(row[ADAMW] - row[FA] for row in rows)
    assert abs(primary - on_cohort) > 0.03  # components that cancel can hide in the primary
    assert found["attrition"]["lost_blocks"] == sorted(i + 1 for i in lost)


def test_cancellation_readings() -> None:
    """C1 = +5 and C2 = −5 pp with the primary ≈ 0 read as two opposite directions; components
    near 0 are bounded by ±5 pp; the readings follow ``_readings``' conventions."""
    h = components(60, 0.05, -0.05)
    found = mechanism_analysis([mechanism_entry(i + 1, row) for i, row in enumerate(h)],
                               MECHANISM)  # fmt: skip
    assert abs(found["sum_check"]["adamw_minus_fa_mean"]) < 0.01  # the primary near 0
    assert found["readings"]["C1"]["readings"] == ["positive"]
    assert found["readings"]["C2"]["readings"] == ["negative"]
    assert found["readings"]["C1"]["interval"] == found["max_t"]["C1"]
    lo, hi = found["max_t"]["C1"]
    assert hi - lo == pytest.approx(2 * found["max_t"]["q"] * found["C1"]["se"])
    assert {found["bonferroni"][name]["level"] for name in ("C1", "C2")} == {0.975}
    near = components(60, 0.0, 0.0)
    bounded = mechanism_analysis([mechanism_entry(i + 1, row) for i, row in enumerate(near)],
                                 MECHANISM)  # fmt: skip
    assert [bounded["readings"][n]["readings"] for n in ("C1", "C2")] == [["bounded"]] * 2
    # The conventions: 0 strictly outside for a direction, ±5 pp bounds included.
    assert component_readings([0.0, 0.03]) == ["bounded"]
    assert component_readings([1e-9, 0.06]) == ["positive"]
    assert component_readings([-0.05, 0.05]) == ["bounded"]
    assert component_readings([-0.07, -0.01]) == ["negative"]
    assert component_readings([-0.06, 0.01]) == ["inconclusive"]
    assert component_readings(None) == ["inconclusive"]


def test_the_mechanism_sensitivities_bound_both_contrasts() -> None:
    """The missing-label sensitivities, for C1 and C2 as ``block_difference`` computes them for
    A − F, by the max-t on the max-t's own stream: from the low bounds' lower end to the high
    bounds' upper end."""
    h = components(60, 0.01, 0.02)
    blocks = [mechanism_entry(i + 1, row, missing=0.004 * (i % 3)) for i, row in enumerate(h)]
    found = mechanism_analysis(blocks, MECHANISM)["sensitivities"]
    stream = derive_seed(7, "mechanism", "max_t")

    def bound(x_field, y_field):
        return max_t_interval([b[x_field][PREWARMED] - b[y_field][FA] for b in blocks],
                              [b[x_field][ADAMW] - b[y_field][PREWARMED] for b in blocks],
                              0.95, 2000, stream)["contrasts"]  # fmt: skip

    low, high = bound("h", "h_missing_harmful"), bound("h_missing_harmful", "h")
    for j, name in enumerate(("C1", "C2")):
        assert found["missing_labels"][name]["interval"] == [low[j]["interval"][0],
                                                             high[j]["interval"][1]]  # fmt: skip
    both = bound("h_missing_harmful", "h_missing_harmful")
    for j, name in enumerate(("C1", "C2")):
        assert found["all_missing_harmful"][name]["interval"] == both[j]["interval"]
    assert set(found) == set(study2_short.SENSITIVITIES)
