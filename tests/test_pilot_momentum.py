"""Study 2's SGD-momentum pilot: its arms, screen, search, comparison evidence and outcomes, and
the zero-field law it is predicted by, on synthetic run directories built as for Stage A
(tests/test_stage_a.py)."""

import math
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from test_stage_a import FULL, SEEDS, THETA0, UP_TO_7, search, write_run

from zero_field_refusal.analysis.pilot_momentum import (
    ARM,
    CONTROL_ARM,
    GRID,
    MOMENTUM,
    outcome,
    pilot_conformance,
    progress,
)
from zero_field_refusal.analysis.stage_a import (
    ADAMW_LONG_GRID,
    BATCHES,
    conformance,
    control_schedule,
    load_trajectories,
    load_trajectory,
)
from zero_field_refusal.analysis.zero_field_law import check, g_at, g_table

FLAT_SGD = {b: {0.004: {}, 0.003: {}} for b in BATCHES}  # Stage A's barrier
# 5, √40 and 8 separate, but √35 and 7 at one evaluation: meets U = 8, not U = 7
EIGHT_NOT_SEVEN = {5.0: 100, math.sqrt(35): 200, math.sqrt(40): 200, 7.0: 200, 8.0: 400}


def with_momentum(run_dir: Path, momentum: float = MOMENTUM) -> Path:
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.optimizer.momentum = momentum
    OmegaConf.save(cfg, run_dir / "config.yaml")
    return run_dir


def pilot(
    root: Path,
    *,
    passing: float = 0.001,
    reach=None,
    controls=None,
    skip_control=None,
    adamw=None,
    stream_path=None,
) -> None:
    """Stage A's whole search under ``root/stage_a`` (flat SGD-long; AdamW-long as ``adamw``,
    default reaching every level at 1e-5), and the pilot's: a screen in which the rates up to
    ``passing`` pass; long runs of the rates of ``reach`` (batch -> rate -> reach; default the
    largest candidate reaching every level); their full controls (contamination
    ``controls[(b, lr)]``, default 0.04)."""
    search(root / "stage_a", sgd=FLAT_SGD, adamw=adamw)
    for b in BATCHES:
        for lr in GRID:
            run = write_run(root / "screen", f"screen-{b}-{lr}", arm="soft_control", batch=b,
                            lr=lr, seed=2000, final_k=0.01 if lr <= passing else 0.2,
                            evaluate_at=(100,), role="screen")  # fmt: skip
            with_momentum(run)
    reach = reach or {b: {passing: FULL} for b in BATCHES}
    for b, by_rate in reach.items():
        for lr, got in by_rate.items():
            for seed in SEEDS:
                run = write_run(root / "discovery", f"{ARM}-{b}-{lr}-{seed}", arm="sgd_long",
                                batch=b, lr=lr, seed=seed, reach=got, monitor_version=2,
                                stream_path=stream_path)  # fmt: skip
                with_momentum(run)
    regular = load_trajectories(root / "discovery")
    for b, by_rate in reach.items():
        for lr in by_rate:
            if (b, lr) == skip_control:
                continue
            plan = control_schedule([r for r in regular if (r.batch_size, r.lr) == (b, lr)])
            c = (controls or {}).get((b, lr), 0.04)
            run = write_run(root / "controls", f"control-{b}-{lr}", arm="soft_control", batch=b,
                            lr=lr, seed=2000, final_k=c, evaluate_at=tuple(plan["seeds"][2000]),
                            role="full_control", stream_path=stream_path)  # fmt: skip
            with_momentum(run)


def roles(root: Path) -> dict:
    dirs = [root / d for d in ("screen", "discovery", "controls") if (root / d).exists()]
    runs = load_trajectories(*dirs)
    return {name: [r for r in runs if r.role == name]
            for name in ("screen", "discovery", "full_control")}  # fmt: skip


def result(root: Path, stage_a: bool = True) -> dict:
    role = roles(root)
    evidence = load_trajectories(root / "stage_a") if stage_a else []
    return outcome(role["screen"], role["discovery"], role["full_control"], THETA0, evidence)


def test_momentum_runs_carry_the_pilot_arms_which_stage_a_refuses(tmp_path) -> None:
    long = load_trajectory(with_momentum(write_run(tmp_path, "l", arm="sgd_long", batch=5,
                                                   lr=0.001, seed=2000, reach=FULL,
                                                   monitor_version=2)))  # fmt: skip
    screened = write_run(tmp_path, "c", arm="soft_control", batch=5, lr=0.001, seed=2000,
                         final_k=0.01, evaluate_at=(100,), role="screen")  # fmt: skip
    control = load_trajectory(with_momentum(screened))
    assert (long.arm, long.momentum) == (ARM, MOMENTUM)
    assert (control.arm, control.momentum) == (CONTROL_ARM, MOMENTUM)
    assert any("not a Stage-A arm" in p for p in conformance([long], "discovery"))
    assert any("not a Stage-A arm" in p for p in conformance([control], "screen"))
    assert pilot_conformance([long], "discovery") == []
    plain = load_trajectory(write_run(tmp_path, "p", arm="sgd_long", batch=5, lr=0.001, seed=2000,
                                      reach=FULL, monitor_version=2))  # fmt: skip
    assert plain.arm == "sgd_long" and plain.momentum == 0.0
    assert any("momentum 0.0" in p for p in pilot_conformance([plain], "discovery"))


def test_the_pilot_is_feasible_where_its_rate_and_adamw_long_meet_the_rule(tmp_path) -> None:
    pilot(tmp_path)
    found = result(tmp_path)
    assert found["outcome"] == "feasible", found.get("problems")
    assert found["feasible"] == {5: 8.0, 20: 8.0}
    assert found["screen"][5]["candidates"] == [0.001, 0.00075]
    entry = found["search"]["searches"][f"{ARM}/B=5"]
    assert list(entry["tested"]) == [0.001] and entry["not_required"] == [0.00075]
    assert set(found["monitor_qualification"]) == set(BATCHES)  # one per batch


def test_u7_is_settled_jointly_with_adamw_long(tmp_path) -> None:
    # AdamW-long meets its rule only at 7; the pilot's top rate meets it only at 8.
    adamw = {b: dict.fromkeys(ADAMW_LONG_GRID, UP_TO_7) for b in BATCHES}
    pilot(tmp_path, adamw=adamw, reach={b: {0.001: EIGHT_NOT_SEVEN} for b in BATCHES})
    found = result(tmp_path)
    assert found["outcome"] == "incomplete" and found["search"]["phase"] == "U=7"
    assert found["problems"] == [
        f"search: {ARM}/B={b} needs {{'rate': 0.00075, 'seeds': {SEEDS}}}" for b in BATCHES
    ]
    more = tmp_path / "more"
    reach = {b: {0.001: EIGHT_NOT_SEVEN, 0.00075: FULL} for b in BATCHES}
    pilot(more, adamw=adamw, reach=reach)
    found = result(more)
    assert found["outcome"] == "feasible", found.get("problems")
    assert found["feasible"] == {5: 7.0, 20: 7.0}


def test_reaching_u_without_a_comparison_is_unmatched(tmp_path) -> None:
    # AdamW-long meets its rule nowhere: no batch can be compared, at 8 or 7.
    pilot(tmp_path, adamw={b: dict.fromkeys(ADAMW_LONG_GRID, {5.0: 100}) for b in BATCHES})
    found = result(tmp_path)
    assert found["outcome"] == "unmatched", found.get("problems")
    assert found["search"]["searches"][f"{ARM}/B=5"]["not_required"] == [0.00075]


def test_a_pilot_that_never_reaches_five_is_a_barrier(tmp_path) -> None:
    pilot(tmp_path, reach={b: {0.001: {}, 0.00075: {}} for b in BATCHES})
    assert result(tmp_path)["outcome"] == "barrier"


def test_a_pilot_reaching_five_but_not_seven_is_partial(tmp_path) -> None:
    low = {5.0: 100, math.sqrt(35): 200}
    pilot(tmp_path, reach={b: {0.001: low, 0.00075: low} for b in BATCHES})
    assert result(tmp_path)["outcome"] == "partial"


def test_no_accepted_control_fails_the_numerics(tmp_path) -> None:
    pilot(tmp_path, reach={b: {0.001: {}, 0.00075: {}} for b in BATCHES},
          controls={(b, lr): 0.5 for b in BATCHES for lr in (0.001, 0.00075)})  # fmt: skip
    assert result(tmp_path)["outcome"] == "numerics_fail"


def test_a_missing_control_leaves_the_pilot_incomplete(tmp_path) -> None:
    pilot(tmp_path, skip_control=(20, 0.001))
    found = result(tmp_path)
    assert found["outcome"] == "incomplete"
    assert found["problems"] == [f"search: {ARM}/B=20 needs {{'rate': 0.001, 'control': 2000}}"]


@pytest.mark.parametrize("change", ["no_stage_a", "off_grid_adamw", "other_stream"])
def test_the_comparison_evidence_must_be_whole(tmp_path, change) -> None:
    pilot(tmp_path, stream_path="teacher/other" if change == "other_stream" else None)
    if change == "off_grid_adamw":
        write_run(tmp_path / "stage_a" / "discovery", "adamw-off-grid", arm="adamw_long",
                  batch=5, lr=0.1, seed=2000, reach=FULL, monitor_version=2)  # fmt: skip
    found = result(tmp_path, stage_a=change != "no_stage_a")
    assert found["outcome"] == "incomplete"
    expected = {"no_stage_a": "Stage A: screen: missing",
                "off_grid_adamw": "outside the search",
                "other_stream": "different streams"}[change]  # fmt: skip
    assert any(expected in p for p in found["problems"]), found["problems"]


def test_the_search_asks_for_the_largest_candidate_on_every_seed(tmp_path) -> None:
    pilot(tmp_path)
    stage_a = load_trajectories(tmp_path / "stage_a")
    state, problems = progress(roles(tmp_path)["screen"], [], [], THETA0, stage_a)
    assert problems == []
    pending = {k: e["pending"] for k, e in state["manifest"]["searches"].items()}
    assert pending == {f"{ARM}/B={b}": {"rate": 0.001, "seeds": SEEDS} for b in BATCHES}


def test_the_zero_field_law_table_and_its_check(tmp_path) -> None:
    # Plain SGD at (0.002, 20) and (0.0005, 5) follow K = (eta / B) g(eta t) with the same g;
    # a momentum run at 0.0002 / (1 - 0.9) = 0.002 and B = 20 then has the check's ratio 1.
    def run(name, batch, lr, seed, effective=None):  # K a function of the effective rate × t
        steps = {k: round(t * 0.002 / (effective or lr)) for k, t in FULL.items()}
        return write_run(tmp_path, name, arm="sgd_long", batch=batch, lr=lr, seed=seed,
                         reach=steps, monitor_version=2)  # fmt: skip

    plain = [load_trajectory(run("p20", 20, 0.002, 2000)),
             load_trajectory(run("p5", 5, 0.0005, 2000))]  # fmt: skip
    table = g_table(plain)
    assert table["runs"] == sorted(r.run_id for r in plain) and len(table["bins"]) > 2
    middle = table["bins"][1]
    assert g_at(table, middle["x"]) == pytest.approx(middle["g"])
    assert g_at(table, table["bins"][0]["x"] / 2) is None
    momentum = load_trajectory(with_momentum(run("m", 20, 0.0002, 2001, effective=0.002)))
    assert momentum.lr / (1 - momentum.momentum) == pytest.approx(0.002)
    assert check(momentum, table)["median_ratio"] == pytest.approx(1.0, rel=0.05)
