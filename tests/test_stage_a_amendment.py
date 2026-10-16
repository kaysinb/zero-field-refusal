"""Stage A's amendment 2: the full controls rerun under the amended gradient rule, judged apart
from the stopped ones, with both selections reported (tests/test_stage_a.py's records)."""

from pathlib import Path

from test_pilot_momentum import BOUND, contaminated, with_momentum
from test_stage_a import THETA0, search, write_run

from zero_field_refusal.analysis.stage_a import (
    BATCHES,
    control_rule,
    control_schedule,
    load_trajectories,
    select,
)

FLAT_SGD = {b: {0.004: {}, 0.003: {}} for b in BATCHES}


def amended(root: Path, contamination: float) -> None:
    """Amendment 2's reruns of every SGD rate's full control, under ``root/amended``."""
    regular = [r for r in load_trajectories(root / "discovery") if r.arm == "sgd_long"]
    for b, lr in sorted({(r.batch_size, r.lr) for r in regular}):
        plan = control_schedule([r for r in regular if (r.batch_size, r.lr) == (b, lr)])
        run = write_run(root / "amended", f"control-{b}-{lr}", arm="soft_control", batch=b, lr=lr,
                        seed=2000, final_k=contamination,
                        evaluate_at=tuple(plan["seeds"][2000]), role="full_control")  # fmt: skip
        with_momentum(run, momentum=0.0)  # plain SGD, the amended rule's schedule
        if contamination > BOUND:
            contaminated(run)


def selections(root: Path) -> dict[int, dict]:
    runs = load_trajectories(root)
    by = {name: [r for r in runs if r.role == name] for name in ("discovery", "screen")}
    controls = [r for r in runs if r.role == "full_control"]
    return {rule: select(by["discovery"], by["screen"],
                         [c for c in controls if control_rule(c) == rule], THETA0, rule=rule)
            for rule in (1, 2)}  # fmt: skip


def test_the_amended_controls_give_their_own_outcome_beside_the_preregistered_one(
    tmp_path,
) -> None:
    # Preregistered controls rejected (contamination 0.5); the reruns accepted: barrier.
    search(tmp_path, sgd=FLAT_SGD,
           controls={(b, lr): 0.5 for b in BATCHES for lr in (0.004, 0.003)})  # fmt: skip
    amended(tmp_path, contamination=0.04)
    found = selections(tmp_path)
    assert found[1]["outcome"] == "numerics_fail", found[1].get("problems")
    assert found[2]["outcome"] == "barrier", found[2].get("problems")
    assert found[2]["control_rule"] == 2 and "control_rule" not in found[1]


def test_rerun_controls_rejected_on_contamination_fail_the_numerics(tmp_path) -> None:
    search(tmp_path, sgd=FLAT_SGD)  # preregistered controls accepted: barrier
    amended(tmp_path, contamination=0.5)  # stopped by a verified exceedance
    found = selections(tmp_path)
    assert found[1]["outcome"] == "barrier" and found[2]["outcome"] == "numerics_fail"


def test_a_control_of_one_rule_does_not_conform_to_the_other(tmp_path) -> None:
    search(tmp_path, sgd=FLAT_SGD)
    amended(tmp_path, contamination=0.04)
    runs = load_trajectories(tmp_path)
    by = {name: [r for r in runs if r.role == name] for name in ("discovery", "screen")}
    rerun = [r for r in runs if r.role == "full_control" and control_rule(r) == 2]
    mixed = select(by["discovery"], by["screen"], rerun, THETA0, rule=1)
    assert mixed["outcome"] == "incomplete"
    assert any("departs from the prescribed schedule" in p for p in mixed["problems"])
