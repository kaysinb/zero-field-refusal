"""The spike check's replay and measurement, on the tiny world's FA launcher run, on the CPU."""

import importlib.util
from pathlib import Path

import pytest
from test_qualification import world as world  # noqa: F401  (the fixture)

from zero_field_refusal import fa_control_diagnostic as fcd
from zero_field_refusal.tensor_hash import tensors_sha256

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "study3_spike_backend", REPO / "scripts/diagnostics/study3_spike_backend.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


@pytest.fixture(scope="module")
def setup(world):  # noqa: F811
    env, runs = world
    cfg = fcd.run_config(runs["fa"])
    with fcd.twin_setup(env, cfg, cfg, k_star=6.0) as (twins, shared):
        yield cfg, twins, shared


@pytest.mark.parametrize("precision", ["fp32", "float64"])
def test_the_measurement_leaves_the_trajectory_as_it_was(setup, precision) -> None:
    """Every row and the endpoint equal an uninterrupted ``run_twin``, measured steps included."""
    cfg, twins, _ = setup
    twins.reset()
    plain = fcd.run_twin(cfg, twins, eval_at=[], precision=precision)
    end = tensors_sha256(twins.trainable)
    record = check.replay(cfg, twins, plain["rows"], measure_at=[0, 1], end=plain["steps"],
                          precision=precision)  # fmt: skip
    assert record["state"] == "measured"
    assert [r["grad_norm"] for r in record["rows"]] == [r["grad_norm"] for r in plain["rows"]]
    assert [r["loss"] for r in record["rows"]] == [r["loss"] for r in plain["rows"]]
    assert record["endpoint_sha256"] == end
    assert all(m["parameters_unchanged"] for m in record["measured"].values())


@pytest.mark.parametrize("precision", ["fp32", "float64"])
def test_the_update_measured_is_the_step_taken(setup, precision) -> None:
    """The production update at a measured step is what the trajectory then does."""
    cfg, twins, _ = setup
    twins.reset()
    plain = fcd.run_twin(cfg, twins, eval_at=[], precision=precision)
    record = check.replay(cfg, twins, plain["rows"], measure_at=[1], end=2, precision=precision)
    m = record["measured"]["1"]
    twins.reset()
    short = fcd.with_lr(cfg, cfg.optimizer.lr)
    short.train.steps = 1
    fcd.run_twin(short, twins, eval_at=[], precision=precision)
    after_one = {n: p.detach().clone() for n, p in twins.trainable}
    twins.reset()
    short.train.steps = 2
    fcd.run_twin(short, twins, eval_at=[], precision=precision)
    delta = [p.detach().double() - after_one[n].double() for n, p in twins.trainable]
    twins.reset()
    assert m["update"]["norm_a"] == pytest.approx(
        sum(d.pow(2).sum().item() for d in delta) ** 0.5, rel=1e-12
    )


def test_a_departure_ends_the_target_with_nothing_compared(setup) -> None:
    cfg, twins, _ = setup
    twins.reset()
    plain = fcd.run_twin(cfg, twins, eval_at=[])
    rows = [dict(r) for r in plain["rows"]]
    rows[1]["grad_norm"] *= 1.01
    record = check.replay(cfg, twins, rows, measure_at=[0, 1, 2], end=3, precision="fp32")
    assert (record["state"], record["at"]) == ("not_reproduced", 1)
    assert set(record["measured"]) == {"0"}  # step 1 departs before its measurement
    twins.reset()


def test_the_records_of_one_step_hold_what_the_runbook_reads(setup) -> None:
    """Backend differences at the tiny model's numerics, a float64 gradient near both, and parts
    that add up to the batch."""
    cfg, twins, _ = setup
    twins.reset()
    plain = fcd.run_twin(cfg, twins, eval_at=[])
    m = check.replay(cfg, twins, plain["rows"], measure_at=[0], end=1, precision="fp32")[
        "measured"
    ]["0"]
    assert m["gradient"]["relative"] < 0.05
    assert m["production_again"]["relative"] == 0.0
    assert m["math_again"]["relative"] == 0.0
    responses = m["parts"][check.PRODUCTION]["responses"]
    assert len(responses) == int(cfg.train.batch_size)
    terms = sum(r["term"] for r in responses)
    micro = sum(r["term"] for r in m["parts"][check.PRODUCTION]["micro_batches"])
    assert terms == pytest.approx(micro, rel=1e-3)
    assert m["float64"][check.PRODUCTION]["relative"] < 0.05
    assert m["float64"][check.MATH]["relative"] < 0.05
    twins.reset()


def test_the_script_is_outside_the_gpu_path() -> None:
    from zero_field_refusal.code_identity import is_gpu_path

    assert not is_gpu_path("scripts/diagnostics/study3_spike_backend.py")
