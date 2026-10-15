"""Study 2's SGD-momentum pilot: its arms, screen, search and outcomes, on synthetic run
directories built as for Stage A (tests/test_stage_a.py)."""

from pathlib import Path

from omegaconf import OmegaConf
from test_stage_a import FULL, SEEDS, THETA0, write_run

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
    BATCHES,
    conformance,
    control_schedule,
    load_trajectories,
    load_trajectory,
)


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
) -> None:
    """The pilot's whole search: a screen in which the rates up to ``passing`` pass; the long
    runs at each batch's candidates, largest first, until a rate's every seed reaches 8; their
    full controls (contamination ``controls[(b, lr)]``, default 0.04); and Stage A's AdamW-long
    at 1e-5 reaching every level on every seed at both batches. ``reach`` maps a batch to a
    rate's reach (default: every rate reaches every level)."""
    for b in BATCHES:
        for lr in GRID:
            with_momentum(write_run(root / "screen", f"screen-{b}-{lr}", arm="soft_control",
                                    batch=b, lr=lr, seed=2000,
                                    final_k=0.01 if lr <= passing else 0.2, evaluate_at=(100,),
                                    role="screen"))  # fmt: skip
    candidates = sorted((lr for lr in GRID if lr <= passing), reverse=True)[:2]
    ran = []
    for b in BATCHES:
        for lr in candidates:
            got = (reach or {}).get(b, FULL)
            for seed in SEEDS:
                name = f"{ARM}-{b}-{lr}-{seed}"
                run = write_run(root / "discovery", name, arm="sgd_long", batch=b, lr=lr,
                                seed=seed, reach=got, monitor_version=2)  # fmt: skip
                with_momentum(run)
            ran.append((b, lr))
            if 8.0 in got:
                break
    regular = load_trajectories(root / "discovery")
    for b, lr in ran:
        if (b, lr) == skip_control:
            continue
        twins = [r for r in regular if (r.batch_size, r.lr) == (b, lr)]
        plan = control_schedule(twins)
        c = (controls or {}).get((b, lr), 0.04)
        with_momentum(write_run(root / "controls", f"control-{b}-{lr}", arm="soft_control",
                                batch=b, lr=lr, seed=2000, final_k=c,
                                evaluate_at=tuple(plan["seeds"][2000]),
                                role="full_control"))  # fmt: skip
    for b in BATCHES:
        for seed in SEEDS:
            write_run(root / "stage_a", f"adamw_long-{b}-{seed}", arm="adamw_long", batch=b,
                      lr=1e-5, seed=seed, reach=FULL, monitor_version=2)  # fmt: skip


def result(root: Path) -> dict:
    runs = load_trajectories(root / "screen", root / "discovery", root / "controls")
    role = {name: [r for r in runs if r.role == name] for name in ("screen", "discovery",
                                                                    "full_control")}  # fmt: skip
    stage_a = load_trajectories(root / "stage_a")
    return outcome(role["screen"], role["discovery"], role["full_control"], THETA0, stage_a)


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
    search = found["search"]["searches"][f"{ARM}/B=5"]
    assert list(search["tested"]) == [0.001] and search["not_required"] == [0.00075]


def test_a_pilot_that_never_reaches_five_is_a_barrier(tmp_path) -> None:
    pilot(tmp_path, reach={b: {} for b in BATCHES})
    found = result(tmp_path)
    assert found["outcome"] == "barrier", found.get("problems")
    assert sorted(found["search"]["searches"][f"{ARM}/B=20"]["tested"]) == [0.00075, 0.001]


def test_a_pilot_reaching_five_but_not_seven_is_partial(tmp_path) -> None:
    pilot(tmp_path, reach={b: {5.0: 100} for b in BATCHES})
    assert result(tmp_path)["outcome"] == "partial"


def test_no_accepted_control_fails_the_numerics(tmp_path) -> None:
    pilot(tmp_path, reach={b: {} for b in BATCHES},
          controls={(b, lr): 0.5 for b in BATCHES for lr in (0.001, 0.00075)})  # fmt: skip
    assert result(tmp_path)["outcome"] == "numerics_fail"


def test_a_missing_control_leaves_the_pilot_incomplete(tmp_path) -> None:
    pilot(tmp_path, skip_control=(20, 0.001))
    found = result(tmp_path)
    assert found["outcome"] == "incomplete"
    assert found["problems"] == [f"search: {ARM}/B=20 needs {{'rate': 0.001, 'control': 2000}}"]


def test_the_search_asks_for_the_next_rate_on_every_seed(tmp_path) -> None:
    pilot(tmp_path)
    screen = [r for r in load_trajectories(tmp_path / "screen")]
    state, problems = progress(screen, [], [], None)
    assert problems == []
    pending = {k: e["pending"] for k, e in state["manifest"]["searches"].items()}
    assert pending == {f"{ARM}/B={b}": {"rate": 0.001, "seeds": SEEDS} for b in BATCHES}
