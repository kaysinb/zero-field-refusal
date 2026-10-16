"""Study 2's redirected short-regime stage, design A (analysis/study2_short.py): selection,
support, the estimator and the looks, on synthetic ``Trajectory`` objects and label rows.

The state machine is driven through its campaigns against an analytic world: K25 = c × (rate /
1e-5)^1.5 × a seed factor, per arm and batch, utility losses growing with K."""

import dataclasses
import math
import statistics
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from zero_field_refusal.analysis import study2_short
from zero_field_refusal.analysis.stage_a import ADAMW_LONG_GRID, SHORT_GRID, Trajectory
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    FA,
    FA_CONTROL_ARM,
    LOOK_SIZES,
    PREWARMED,
    block_difference,
    block_support,
    bootstrap_t_interval,
    candidate_order,
    control_conformance,
    control_verdict,
    finite_bounds,
    harm,
    interpolate,
    repair_rates,
    secondary_contrasts,
    select_rates,
    selection,
    targeted_rates,
    usable_rate,
    utility_family,
    validation_verdict,
)
from zero_field_refusal.seeding import derive_seed

THETA0 = {"benchmark_accuracy": 0.60, "factual_accuracy": 0.80}
SEEDS = (2000, 2001, 2002)
K8 = math.sqrt(40)
W_MAX = 0.35
SPC = 90.0  # seconds per chunk


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
    factor = (spread or {}).get(arm, SPREAD)[seed]
    x = lr / 1e-5
    return scale[(arm, batch)] * (kink(x) if (arm, batch) in kinked else x**1.5) * factor


def made(arm, batch, lr, seed, scale=SCALE, spread=None, kinked=()) -> Trajectory:
    k = world_k(arm, batch, lr, seed, scale, spread, kinked)
    return traj(arm, batch, lr, seed, k, mmlu=0.3 * k, factual=0.1 * k,
                seconds=100.0 if batch == 5 else 150.0)  # fmt: skip


def grids(scale=SCALE, spread=None, kinked=()) -> list[Trajectory]:
    return [made(arm, b, lr, s, scale, spread, kinked) for arm in (ADAMW, FA) for b in (5, 20)
            for lr in SHORT_GRID for s in SEEDS]  # fmt: skip


def drive(runs, *, scale=SCALE, spread=None, kinked=(), contaminated=(), limit=16):
    """Make each campaign the state machine asks for until it ends; FA controls at the rates in
    ``contaminated`` read 2 × their bound. Returns the campaigns and the last result."""
    seen = []
    for _ in range(limit):
        result = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC)
        step = result["next"]
        seen.append(step["campaign"])
        if step["campaign"] in ("done", "infeasible", "incomplete"):
            return seen, result
        for r in step["runs"]:
            if step["campaign"] == "controls":
                bound = r["contamination_k"]
                c = 2 * bound if r["lr"] in contaminated else 0.1 * bound
                runs.append(control(r["batch"], r["lr"], bound, c))
            else:
                runs.append(made(r["arm"], r["batch"], r["lr"], r["seed"], scale, spread, kinked))
    raise AssertionError(f"no end after {seen}")


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


def test_candidates_put_u8_first_and_the_cheaper_batch_first_with_costs() -> None:
    runs = grids()  # 100 s per run at B = 5, 150 s at B = 20
    order = candidate_order(runs, THETA0, SPC)
    assert [(e["batch"], e["U"]) for e in order] == [(5, 8.0), (20, 8.0), (5, 7.0), (20, 7.0)]
    # Six runs, one chunk of 125 training responses, v₀'s 50 × 5 = 250 preconditioner responses.
    assert order[0]["seconds"] == pytest.approx(6 * 100.0 + SPC + SPC * 250 / 1000)
    assert order[0]["preconditioner_seconds"] == pytest.approx(22.5)
    assert order[1]["seconds"] == pytest.approx(6 * 150.0 + SPC + SPC * 1000 / 1000)
    assert order[0]["nearest"][ADAMW] == [1e-5, 1e-5, 1.5e-5]  # medians 5.0, 5.0, 9.19
    cheap20 = [traj(r.arm, r.batch_size, r.lr, r.seed, r.evaluations[0]["K_benign"],
                    seconds=60.0 if r.batch_size == 20 else 100.0) for r in runs]  # fmt: skip
    assert [e["batch"] for e in candidate_order(cheap20, THETA0, SPC)] == [20, 5, 20, 5]


def test_equal_costs_go_to_the_smaller_batch_and_unknown_costs_last() -> None:
    # B = 20's preconditioner stream costs 67.5 s more per block: its runs are 11.25 s cheaper.
    runs = [traj(r.arm, r.batch_size, r.lr, r.seed, r.evaluations[0]["K_benign"],
                 seconds=100.0 if r.batch_size == 5 else 88.75) for r in grids()]  # fmt: skip
    order = candidate_order(runs, THETA0, SPC)
    assert order[0]["seconds"] == order[1]["seconds"] and order[0]["batch"] == 5
    without = [r for r in runs if (r.arm, r.batch_size) != (FA, 5)]
    order = candidate_order(without, THETA0, SPC)
    assert [e["batch"] for e in order] == [20, 5, 20, 5]
    assert order[1]["seconds"] is None and order[1]["reasons"]


# ------------------------------------------------------------------ the state machine


def test_the_state_machine_targets_controls_prewarms_and_freezes() -> None:
    runs = grids()
    seen, result = drive(runs)
    assert seen == ["targeted", "controls", "prewarmed", "done"]
    done = result["next"]
    assert (done["batch"], done["U"]) == (5, 8.0)
    assert done["rates"] == {ADAMW: [1e-5, 1.17e-5, 1.37e-5], FA: [1.27e-5, 1.49e-5, 1.74e-5],
                             PREWARMED: [1e-5, 1.17e-5, 1.37e-5]}  # fmt: skip
    assert not done["prewarmed_dropped"]
    assert all(v["accepted"] for v in done["fa_controls"].values())
    rounds = result["rounds"]
    assert rounds[f"{ADAMW}/B=5/U=8"]["targeted"]["new_rates"] == [1.17e-5, 1.37e-5]  # 1e-5 run
    assert result["problems"] == [] and len(result["attempts"]) == 1


def test_the_first_campaign_asks_for_every_seed_of_each_targeted_rate() -> None:
    result = selection(grids(), THETA0, W_MAX, seconds_per_chunk=SPC)
    step = result["next"]
    assert step["campaign"] == "targeted" and (step["batch"], step["U"]) == (5, 8.0)
    asked = {(r["arm"], r["lr"]) for r in step["runs"]}
    assert asked == {(ADAMW, 1.17e-5), (ADAMW, 1.37e-5), (FA, 1.27e-5), (FA, 1.49e-5),
                     (FA, 1.74e-5)}  # fmt: skip
    assert len(step["runs"]) == 3 * len(asked)


def test_a_rejected_fa_control_is_replaced_once() -> None:
    runs = grids()
    seen, result = drive(runs, contaminated=(1.49e-5,))
    assert seen == ["targeted", "controls", "controls", "prewarmed", "done"]
    assert result["next"]["rates"][FA] == [1.27e-5, 1.5e-5, 1.74e-5]  # the grid's 1.5e-5
    attempt = result["attempts"][0]
    assert [sorted(v) for v in attempt["controls"]] == [[1.27e-5, 1.49e-5, 1.74e-5],
                                                        [1.27e-5, 1.5e-5, 1.74e-5]]  # fmt: skip
    assert attempt["arms"][FA]["selections"][-1]["excluded"] == [1.49e-5]


def test_a_rejected_replacement_fails_the_candidate() -> None:
    runs = grids()
    seen, result = drive(runs, contaminated=(1.49e-5, 1.5e-5))
    first = result["attempts"][0]
    assert first["outcome"] == "failed" and "after the re-solve" in first["reasons"][0]
    assert (result["next"]["batch"], result["next"]["U"]) == (20, 8.0)
    assert seen[:3] == ["targeted", "controls", "controls"] and seen[-1] == "done"


def test_a_control_of_another_bound_is_not_this_candidates() -> None:
    runs = grids()
    drive(runs)
    other = [control(5, r.lr, 0.01 * math.sqrt(35), 0.0) for r in runs
             if r.arm == FA_CONTROL_ARM]  # fmt: skip
    kept = [r for r in runs if r.arm != FA_CONTROL_ARM] + other
    result = selection(kept, THETA0, W_MAX, seconds_per_chunk=SPC)
    assert result["next"]["campaign"] == "controls"
    # The U = 7 controls and the pre-warmed runs made after them are not called for yet.
    assert {p.split(" (")[0] for p in result["problems"]} == {"unexpected control",
                                                              "unexpected run"}  # fmt: skip
    assert sum(p.startswith("unexpected control") for p in result["problems"]) == 3


def test_a_control_off_its_prescription_leaves_no_conclusion() -> None:
    runs = grids()
    drive(runs)
    first = next(r for r in runs if r.arm == FA_CONTROL_ARM)
    moved = [dataclasses.replace(r, model="other@rev") if r is first else r for r in runs]
    result = selection(moved, THETA0, W_MAX, seconds_per_chunk=SPC)
    assert result["next"]["campaign"] == "incomplete"
    assert "is not on the model and stream" in result["problems"][0]
    twin = next(r for r in runs if (r.arm, r.batch_size, r.lr, r.seed) == (FA, 5, first.lr, 2000))
    assert control_conformance(first, twin) == []
    assert "no FA run" in control_conformance(first, None)[0]
    sparse = dataclasses.replace(first, evaluate_at=(20,))
    assert "it needs 25 at [25]" in control_conformance(sparse, twin)[0]


def test_a_rate_rejected_by_its_control_at_u8_is_excluded_at_u7() -> None:
    runs = grids()
    seen, result = drive(runs, contaminated=(1.27e-5, 1.59e-5, 1.5e-5))
    attempts = result["attempts"]
    assert [(a["batch"], a["U"], a["outcome"]) for a in attempts[:2]] == [
        (5, 8.0, "failed"), (20, 8.0, "failed")]  # fmt: skip
    at7 = attempts[2]
    assert (at7["batch"], at7["U"]) == (5, 7.0)
    assert list(at7["fa_excluded"]) == [1.27e-5]
    assert "rejected at bound 0.0632456" in at7["fa_excluded"][1.27e-5]
    assert at7["arms"][FA]["selections"][0]["excluded"] == [1.27e-5]
    # No second control was asked for 1.27e-5 at U = 7's bound.
    asked = [r for r in runs if r.arm == FA_CONTROL_ARM and (r.batch_size, r.lr) == (5, 1.27e-5)]
    assert len(asked) == 1


def test_a_failed_u8_leaves_u7_its_own_rounds_at_the_same_batch() -> None:
    wide = {ADAMW: {2000: 0.8, 2001: 1.0, 2002: 1.3}}  # wider than the targets' spacing
    runs = grids(spread=wide)
    seen, result = drive(runs, spread=wide)
    # Every candidate, U = 7 included, gets its own targeted and repair rounds.
    assert seen == ["targeted", "repair"] * 4 + ["infeasible"]
    attempts = result["attempts"]
    assert [(a["batch"], a["U"]) for a in attempts] == [(5, 8.0), (20, 8.0), (5, 7.0), (20, 7.0)]
    assert {a["outcome"] for a in attempts} == {"failed"}
    rounds = result["rounds"]
    repair8 = rounds[f"{ADAMW}/B=5/U=8"]["repair"]
    assert repair8["new_rates"] == [8.26e-6] and "seeds [2002]" in repair8["proposal"]["reasons"][0]
    # U = 7's targets differ: its targeted round asks for new rates; 1e-5 serves 5 at both U.
    targeted7 = rounds[f"{ADAMW}/B=5/U=7"]["targeted"]
    assert [t["rate"] for t in targeted7["proposal"]["targets"]] == [1e-5, 1.12e-5, 1.25e-5]
    assert targeted7["new_rates"] == [1.12e-5, 1.25e-5]
    # Its repair proposes r_5 / 1.21 again, already run at U = 8, and r_U × 1.21 (seed 2000 now
    # stays below K* = √35): only the latter is run.
    repair7 = rounds[f"{ADAMW}/B=5/U=7"]["repair"]
    assert repair7["proposal"]["rates"] == [8.26e-6, 1.51e-5]
    assert repair7["new_rates"] == [1.51e-5]
    # Every run made at B = 5 enters U = 7's selections.
    usable7 = attempts[2]["arms"][ADAMW]["selections"][0]["usable"]
    assert {8.26e-6, 1.17e-5, 1.37e-5} <= set(usable7)
    assert f"{FA}/B=5/U=7" not in rounds  # FA is feasible at U = 7 from what B = 5 ran


def test_a_coverage_miss_after_the_targeted_round_is_repaired_in_the_state_machine() -> None:
    # AdamW's K bends between the grid's 1e-5 and 1.5e-5: the targeted K* and U rates overshoot.
    bent = {(ADAMW, 5)}
    runs = grids(kinked=bent)
    seen, result = drive(runs, kinked=bent)
    assert seen == ["targeted", "repair", "controls", "prewarmed", "targeted", "done"]
    first = result["attempts"][0]["arms"][ADAMW]["selections"]
    assert "coverage" in first[1]["unmet"]  # after the targeted round
    repair = result["rounds"][f"{ADAMW}/B=5/U=8"]["repair"]["proposal"]
    # K* re-targeted to 1.08e-5 (also the too-wide bracket's midpoint) and U to 1.19e-5.
    assert repair["rates"] == [1.08e-5, 1.19e-5]
    assert sum("re-targeted" in reason for reason in repair["reasons"]) == 2
    assert result["next"]["rates"][ADAMW] == [1e-5, 1.08e-5, 1.16e-5]


def test_a_prewarmed_arm_still_infeasible_after_its_rounds_is_dropped() -> None:
    slow = {**SCALE, (PREWARMED, 5): 2.0}
    runs = grids()
    seen, result = drive(runs, scale=slow)
    assert seen == ["targeted", "controls", "prewarmed", "repair", "done"]
    done = result["next"]
    assert done["prewarmed_dropped"] and done["rates"][PREWARMED] is None
    assert done["rates"][ADAMW] == [1e-5, 1.17e-5, 1.37e-5]  # the primary goes on
    pre = result["prewarmed"]
    assert pre["initial"] == sorted({1e-5, 1.17e-5, 1.37e-5, 1.27e-5, 1.49e-5, 1.74e-5})
    assert [r["kind"] for r in pre["rounds"]] == ["targeted", "repair"]
    assert pre["rounds"][0]["new_rates"] == []  # every target above the range
    assert pre["rounds"][1]["new_rates"] == [2.11e-5]  # 1.74e-5 × 1.21


def test_unprescribed_runs_are_reported_and_duplicates_leave_no_conclusion() -> None:
    runs = [*grids(), made(ADAMW, 5, 1.23e-5, 2000)]
    result = selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC)
    assert result["problems"] == ["unexpected run ('adamw_25', 5, 1.23e-05, 2000)"]
    assert 1.23e-5 not in result["attempts"][0]["arms"][ADAMW]["selections"][0]["usable"]
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
    assert result["next"]["campaign"] == "targeted"
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
              "frozen_adam_25-b5-1.4e-05-s3000": label_rows(14)}  # fmt: skip
    found = block_difference({ADAMW: a, FA: f}, labels, K8)
    assert found["delta"] == pytest.approx(0.15 - 0.11)
    assert found["delta_missing_harmful"] == pytest.approx(0.16 - 0.11)
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


def test_secondary_contrasts_use_the_common_three_arm_cohort() -> None:
    blocks = []
    for i in range(30):
        h = {ADAMW: 0.20 + 0.01 * (i % 3), FA: 0.15}
        if i % 5:
            h[PREWARMED] = 0.18
        blocks.append({"block": i, "complete": True, "supported": i != 7, "delta": 0.05, "h": h})
    found = secondary_contrasts(blocks, 25)
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
    assert found["prewarmed_minus_fa"]["mean"] == pytest.approx(0.03)


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
