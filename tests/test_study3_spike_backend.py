"""The spike check's replay and measurement, on the tiny world's FA launcher run, on the CPU."""

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from test_qualification import world as world  # noqa: F401  (the fixture)

from zero_field_refusal import fa_control_diagnostic as fcd
from zero_field_refusal import runtime
from zero_field_refusal.config import apply_override
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


def arm(cfg, optimizer):
    """F's run config, or P's on the same v₀ and rate."""
    if optimizer == cfg.optimizer.name:
        return cfg
    out = apply_override(copy.deepcopy(cfg), f"optimizer={optimizer}")
    out.optimizer.lr = cfg.optimizer.lr
    OmegaConf.resolve(out)
    return out


ARMS = ["adam_frozen", "adamw_prewarmed"]


@pytest.mark.parametrize("optimizer", ARMS)
@pytest.mark.parametrize("precision", ["fp32", "float64"])
def test_the_measurement_leaves_the_trajectory_as_it_was(setup, precision, optimizer) -> None:
    """Every row and the endpoint equal an uninterrupted ``run_twin``, measured steps included."""
    cfg, twins, _ = setup
    cfg = arm(cfg, optimizer)
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


@pytest.mark.parametrize("optimizer", ARMS)
@pytest.mark.parametrize("precision", ["fp32", "float64"])
def test_the_update_measured_is_the_step_taken(setup, precision, optimizer) -> None:
    """The production update at a measured step is what the trajectory then does."""
    cfg, twins, _ = setup
    cfg = arm(cfg, optimizer)
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


@pytest.mark.parametrize("precision", ["fp32", "float64"])
def test_the_records_of_one_step_hold_what_the_runbook_reads(setup, precision) -> None:
    """The backends compared as vectors, on the batch and each micro-batch in the trajectory's
    arithmetic; the micro-batches add up to the batch gradient; the responses' norms."""
    cfg, twins, _ = setup
    twins.reset()
    plain = fcd.run_twin(cfg, twins, eval_at=[], precision=precision)
    m = check.replay(cfg, twins, plain["rows"], measure_at=[0], end=1, precision=precision)[
        "measured"
    ]["0"]
    assert m["gradient"]["relative"] < 0.05
    assert m["production_again"]["relative"] == 0.0
    assert m["math_again"]["relative"] == 0.0
    part = m["parts"]
    size, micro = int(cfg.train.batch_size), int(cfg.execution.micro_batch_size)
    assert len(part["micro_batches"]) == -(-size // micro)
    assert all(b["relative"] < 0.05 for b in part["micro_batches"])
    assert part["sum_against_batch"]["relative"] < 1e-5
    assert len(part["responses"]) == size
    terms = sum(r[check.PRODUCTION]["term"] for r in part["responses"])
    assert terms == pytest.approx(sum(b["term"] for b in part["micro_batches"]), rel=1e-3)
    twins.reset()


def test_without_the_gate_every_row_is_taken_and_a_departure_recorded(setup) -> None:
    """On another host: rows only, the trajectory as it runs, the departure named."""
    cfg, twins, _ = setup
    twins.reset()
    plain = fcd.run_twin(cfg, twins, eval_at=[])
    end = tensors_sha256(twins.trainable)
    rows = [dict(r) for r in plain["rows"]]
    rows[1]["grad_norm"] *= 1.01
    record = check.replay(cfg, twins, rows, measure_at=[], end=plain["steps"], precision="fp32",
                          gate=False)  # fmt: skip
    assert record["state"] == "measured"
    assert record["first_departure"] == 1
    assert record["largest_departure"] > 0.009
    assert record["measured"] == {}
    assert len(record["rows"]) == plain["steps"]
    assert record["endpoint_sha256"] == end


def test_a_full_case_holds_the_baseline_the_reference_and_their_d(setup, tmp_path) -> None:
    """As the harness runs them: its rows, its K, its D against the same trajectories run alone."""
    cfg, twins, shared = setup
    twins.reset()
    base = fcd.run_twin(cfg, twins, eval_at=[], monitor=shared["monitor"])
    launcher = tmp_path / "launcher.pt"
    fcd.save_trainable(twins.trainable, launcher)
    twins.reset()
    ref = fcd.run_twin(cfg, twins, eval_at=[], precision="float64")
    twins.reset()
    out = check.full_case(cfg, twins, shared, recorded=base["rows"], checkpoints=tmp_path,
                          launcher_checkpoint=launcher, reference_rows=ref["rows"])  # fmt: skip
    assert [r["grad_norm"] for r in out["baseline"]["rows"]] == [
        r["grad_norm"] for r in base["rows"]
    ]
    assert out["baseline"]["monitor"] == base["monitor"]
    assert out["baseline"]["against_record"]["first_departure"] is None
    assert out["reference"]["against_record"]["first_departure"] is None
    assert out["comparisons"]["launcher~baseline"]["D_twin"] == 0.0
    assert set(out["comparisons"]) == {"launcher~baseline", "baseline~reference",
                                       "launcher~reference"}  # fmt: skip
    assert (out["comparisons"]["baseline~reference"]["D_twin"]
            == out["comparisons"]["launcher~reference"]["D_twin"])  # fmt: skip
    assert out["baseline"]["instability"] is None
    assert tensors_sha256(twins.trainable) == tensors_sha256(
        [(n, twins.theta0[n]) for n in twins.names]
    )


def test_a_full_case_without_a_reference_or_a_launcher_checkpoint(setup, tmp_path) -> None:
    """A stopped launcher run has no checkpoint: the baseline's rows alone, a departure named."""
    cfg, twins, shared = setup
    twins.reset()
    base = fcd.run_twin(cfg, twins, eval_at=[])
    rows = [dict(r) for r in base["rows"]]
    rows[2]["loss"] *= 1.01
    twins.reset()
    out = check.full_case(cfg, twins, shared, recorded=rows, checkpoints=tmp_path)
    assert "reference" not in out
    assert out["comparisons"] == {}
    assert out["baseline"]["against_record"]["first_departure"] == 2


def test_departures_cover_what_the_rows_leave_out() -> None:
    wanted = [{"step": s, "loss": 1.0, "grad_norm": 2.0} for s in range(3)]
    assert not check._departures(wanted, wanted)["differs"]
    truncated = check._departures(wanted[:2], wanted)
    assert truncated["differs"] and truncated["missing_steps"] == [2]
    nan = [dict(wanted[0]), dict(wanted[1]), {"step": 2, "loss": float("nan"), "grad_norm": 2.0}]
    ended = check._departures(nan, wanted)
    assert ended["differs"] and ended["nonfinite_steps"] == [2]
    longer = [*wanted, {"step": 3, "loss": 9.0, "grad_norm": 90.0}]  # past a declared stop
    assert not check._departures(longer, wanted)["differs"]


def test_a_launcher_end_that_differs_is_named() -> None:
    kept = {"reason": "k_above_max", "steps": 5, "monitor": {5: 837.96}, "k_25": None}
    assert check.outcome_differences(kept, kept) == []
    for change in ({"reason": "steps"}, {"steps": 25}, {"monitor": {5: 8.0}}, {"monitor": {}},
                   {"k_25": 7.5}):  # fmt: skip
        assert check.outcome_differences({**kept, **change}, kept)


def test_an_identity_that_differs_is_named() -> None:
    inputs = {"prompt_manifest_sha256": "p", "theta0_trainable_sha256": "t", "v0_sha256": "v",
              "numerics_setup": {"a": 1},
              "streams": {"training": {"sha256": "s", "count": 100},
                          "kl_benign": {"sha256": "k"}}}  # fmt: skip
    meta = {"config_hash": "c", "inputs": inputs, "hardware": {"driver": "1"}, "packages": {"t": 1},
            "uv_lock_sha256": "u", "python": "3"}  # fmt: skip
    harness = {"theta0_trainable_sha256": "t", "v0_sha256": "v", "numerics_setup": {"a": 1},
               "stream_here": "s", "stream_recorded": "s"}  # fmt: skip
    assert check.identity_problems(meta, meta, harness) == []
    assert check.identity_problems({**meta, "hardware": {"driver": "2"}}, meta, harness) == [
        "environment hardware"
    ]
    assert check.identity_problems(meta, meta, {**harness, "v0_sha256": "w"}) == [
        "harness v0_sha256"
    ]
    assert check.identity_problems(meta, meta, {**harness, "stream_recorded": "x"})
    other = {**inputs, "streams": {**inputs["streams"], "kl_benign": {"sha256": "z"}}}
    assert check.identity_problems({**meta, "inputs": other}, meta, harness) == ["stream kl_benign"]


def test_a_full_report_of_the_record_against_itself_finds_nothing(world, tmp_path) -> None:  # noqa: F811
    """The real records' shape: the tiny launcher run stands for both hosts' runs. Its K changed in
    a copy is named. A harness part under the same code agrees with its launcher run."""
    env, runs = world
    cfg = fcd.run_config(runs["fa"])
    numerics = runtime.configure_numerics(cfg, env.device)
    same = check.full_report(env, cfg, numerics, launcher=runs["fa"], original=runs["fa"],
                             checkpoints=tmp_path / "a")  # fmt: skip
    assert same["identity_problems"] == []
    assert same["differences"] == []
    copy_dir = tmp_path / "copy"
    shutil.copytree(runs["fa"], copy_dir)
    lines = [json.loads(line) for line in (copy_dir / "monitor.jsonl").read_text().splitlines()]
    lines[0]["K_monitor"] *= 2
    (copy_dir / "monitor.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    changed = check.full_report(env, cfg, numerics, launcher=copy_dir, original=runs["fa"],
                                checkpoints=tmp_path / "b")  # fmt: skip
    assert any(d.startswith("launcher K at step") for d in changed["differences"])


def test_the_script_is_outside_the_gpu_path() -> None:
    from zero_field_refusal.code_identity import is_gpu_path

    assert not is_gpu_path("scripts/diagnostics/study3_spike_backend.py")
