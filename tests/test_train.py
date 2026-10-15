"""End-to-end CPU runs on a tiny model: streams → training → evaluation → run directory."""

import dataclasses
import json
import shutil
from pathlib import Path

import pytest
import torch
from conftest import tiny_model, tiny_tokenizer

from zero_field_refusal.models import install_neutral_generation_config
from zero_field_refusal.prechecks import CheckFailed, environment_fingerprint, write_check_record
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import (
    Environment,
    ReplayedSoftTargets,
    StreamPrefix,
    extend_stream,
    generate_stream,
    load_stream_for,
    locate_stream,
    remove_partial_streams,
)
from zero_field_refusal.teacher import (
    StreamError,
    chunk_directory,
    load_manifest,
    partial_directory,
)
from zero_field_refusal.tracking import Diverged
from zero_field_refusal.train import run_training, training_batches

SMALL = [
    "train.steps=2",
    "train.batch_size=4",
    "teacher.sampling.max_new_tokens=12",
    "eval.kl.max_new_tokens=12",
    "eval.kl.samples_per_prompt=2",
    "eval.harmful.max_new_tokens=6",
    "eval.harmful.limit=null",
    "eval.kl.limit=null",
    "eval.projections=2",
    "execution.device=cpu",
    "execution.micro_batch_size=3",
    "execution.generation_batch_size=4",
    "execution.eval_batch_size=4",
    "checks.teacher_gap.token_max_abs=1e-4",  # CPU fp32; GPU thresholds come from stage 0
    "checks.teacher_gap.response_mean_signed_abs=1e-3",
    "checks.path_field.batches=4",
]


BOUND = {"level": 0.975, "quantile": 2.0, "chi_square_per_response": 0.0,
         "standard_error_per_response": 0.0}  # fmt: skip


def certified(**changes) -> dict:
    """What a stage-0 record certifies, for the test runs (steps 2, batch 4, cap 12)."""
    return {"criterion": 0.01, "steps": 2, "batch_size": 4, "cap": 12,
            "chi_square_bound": BOUND} | changes  # fmt: skip


def pass_path_check(env: Environment, **changes) -> None:
    """A passed stage-0 path-field record for the test environment."""
    fingerprint = environment_fingerprint(config(), env.device)
    write_check_record(env.data_root, fingerprint, certified(**changes), passed=True,
                       run_id="fixture")  # fmt: skip


def words(rng: torch.Generator, n: int) -> str:
    return " ".join(f"w{int(i)}" for i in torch.randint(0, 40, (n,), generator=rng))


def prompt_sets() -> dict[str, list[dict]]:
    rng = torch.Generator().manual_seed(0)

    def items(prefix, n, **extra):
        return [{"id": f"{prefix}:{i}", "text": words(rng, 4), **extra} for i in range(n)]

    sets = {
        "alpaca_teacher": items("teacher", 6),
        "alpaca_benign_kl": items("benign", 3),
        "harmful_kl": items("kl", 3),
        "harmful_judged": items("judged", 5),
        "harmful_calibration": items("calibration", 3),
        "factual_qa": items("fact", 4, answers=["w1"]),
    }
    sets["alpaca_heldout"] = [{**x, "reference": words(rng, 5)} for x in items("held", 3)]
    sets["alpaca_monitor"] = items("monitor", 3)
    sets["harmful_evaluation"] = items("evaluation", 6)
    sets["alpaca_reference_pool"] = [{**x, "reference": words(rng, 20)} for x in items("ref", 12)]
    sets["mmlu"] = [
        {**x, "question": x["text"], "choices": ["w1", "w2", "w3", "w4"], "answer": i % 4,
         "subject": "s"}
        for i, x in enumerate(items("mmlu", 4))
    ]  # fmt: skip
    return sets


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Environment:
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    tokenizer = tiny_tokenizer()
    end_ids = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    sets = prompt_sets()

    def new_model():
        model = tiny_model(tied=False, head_std=0.02)
        install_neutral_generation_config(model, end_ids, tokenizer.pad_token_id)
        return model

    env = Environment(
        tokenizer=tokenizer,
        end_ids=end_ids,
        pad_id=tokenizer.pad_token_id,
        device=torch.device("cpu"),
        new_model=new_model,
        prompt_set=lambda name: sets[name],
        data_root=tmp_path / "data",
    )
    pass_path_check(env)
    return env


def config(*overrides: str):
    return load_run_config("experiments/smoke", [*SMALL, *overrides])


def prepare(env, cfg, *kinds, seed=0):
    for kind in ("kl_harmful", "kl_benign", *kinds):
        try:
            generate_stream(cfg, env, kind, None if kind.startswith("kl_") else seed, batch_size=4)
        except Exception as exc:
            if "exists" not in str(exc):
                raise


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_adamw_run_writes_metrics_summary_and_observations(env) -> None:
    cfg = config()
    prepare(env, cfg, "training")
    run_dir = run_training(cfg, env, run_id="adamw")
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    assert [m["step"] for m in metrics] == [0, 1]
    assert all(m["denominator"] == 4 * 12 and m["grad_norm"] > 0 for m in metrics)
    assert len(metrics[0]["grad_projection"]) == 2
    summary = json.loads((run_dir / "summary.json").read_text())
    for key in ("K_harmful", "K_benign", "keyword_refusal_rate", "heldout_nll_per_token",
                "factual_accuracy", "benchmark_accuracy", "delta_norm", "relative"):  # fmt: skip
        assert key in summary, key
    assert summary["K_harmful"] > 0 and summary["delta_norm"] > 0
    assert summary["step0_teacher_gap"]["token_max_abs"] < 1e-4  # invariant 1 on CPU fp32
    measures = {row["measure"] for row in read_jsonl(run_dir / "observations/measurements.jsonl")}
    assert measures == {"kl_harmful", "kl_benign", "heldout_nll", "factual_qa", "benchmark"}
    responses = read_jsonl(run_dir / "responses/harmful_judged.jsonl")
    assert [r["item_id"] for r in responses] == [f"judged:{i}" for i in range(5)]
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert set(meta["inputs"]["streams"]) == {"training", "kl_harmful", "kl_benign"}
    assert meta["inputs"]["streams"]["training"]["sha256"] == summary["stream/training"]


def test_every_optimizer_sees_the_same_batches_in_the_same_order(env) -> None:
    cfg = config()
    prepare(env, cfg, "training")
    stream, _ = load_stream_for(cfg, env, "training", 0)
    adamw = training_batches(stream, cfg)
    sgd = training_batches(stream, config("optimizer=sgd", "optimizer.lr=0.1"))
    assert [[s.response_ids for s in b] for b in adamw] == [
        [s.response_ids for s in b] for b in sgd
    ]
    flat = [s for batch in adamw for s in batch]
    assert len({id(s) for s in flat}) == len(stream)  # each response once


@pytest.mark.parametrize("optimizer", ["adam_frozen", "adamw_prewarmed"])
def test_preconditioned_arms_estimate_the_second_moment_from_their_own_stream(env, optimizer):
    cfg = config(f"optimizer={optimizer}", "optimizer.lr=1e-3",
                 "optimizer.preconditioner.batches=3")  # fmt: skip
    prepare(env, cfg, "training", "preconditioner")
    run_dir = run_training(cfg, env, run_id=optimizer)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["stream/preconditioner"] != summary["stream/training"]
    assert summary["delta_norm"] > 0


def test_bridge_cell_trains_on_the_same_stream_truncated(env) -> None:
    cfg = config("train.max_response_tokens=5")
    prepare(env, cfg, "training")
    run_dir = run_training(cfg, env, run_id="bridge")
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    assert all(m["denominator"] == 4 * 5 and m["tokens"] <= 4 * 5 for m in metrics)
    summary = json.loads((run_dir / "summary.json").read_text())
    _, main_cell = load_stream_for(config(), env, "training", 0)
    assert summary["stream/training"] == main_cell["sha256"]  # the 96-token cell's stream


def test_a_non_finite_loss_ends_the_run_as_diverged(env, monkeypatch) -> None:
    from zero_field_refusal import train

    accumulate = train.accumulate_gradient
    calls = []

    def blow_up(*args, **kwargs):
        stats = accumulate(*args, **kwargs)
        calls.append(stats)
        return stats | {"loss": float("nan")} if len(calls) == 2 else stats

    monkeypatch.setattr(train, "accumulate_gradient", blow_up)
    cfg = config()
    prepare(env, cfg, "training")
    with pytest.raises(Diverged, match="step 1"):
        run_training(cfg, env, run_id="diverged")
    run_dir = Path(env.data_root).parent / "runs" / "smoke" / "diverged"
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "diverged"
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["diverged_at_step"] == 1 and "K_harmful" not in summary
    assert [m["step"] for m in read_jsonl(run_dir / "metrics.jsonl")] == [0]


def test_a_non_finite_k_at_evaluation_ends_the_run_as_diverged(env, monkeypatch) -> None:
    from zero_field_refusal import train

    def nan_kl(*args, **kwargs):
        return torch.full((len(args[2]),), float("nan"))

    monkeypatch.setattr(train, "sequence_kl", nan_kl)
    cfg = config()
    prepare(env, cfg, "training")
    with pytest.raises(Diverged, match="not finite under condition trained"):
        run_training(cfg, env, run_id="bad-k")
    run_dir = Path(env.data_root).parent / "runs" / "smoke" / "bad-k"
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "diverged"
    assert json.loads((run_dir / "summary.json").read_text())["diverged_in"] == "trained"
    assert not (run_dir / "responses").exists()  # stopped before generating


def test_controls_scan_the_whole_displacement_and_random_directions(env) -> None:
    cfg = config(
        "eval.controls.line_scan.enabled=true",
        "eval.controls.line_scan.coefficients=[1.0]",
        "eval.controls.random_direction.enabled=true",
        "eval.utility.benchmark=false",
    )
    prepare(env, cfg, "training")
    run_dir = run_training(cfg, env, run_id="controls")
    summary = json.loads((run_dir / "summary.json").read_text())
    controls = summary["controls"]
    assert set(controls) == {"line_scan_+1", "line_scan_-1", "random_global", "random_per_tensor"}
    assert controls["line_scan_+1"]["K_harmful"] == pytest.approx(summary["K_harmful"], rel=1e-4)
    assert controls["line_scan_-1"]["K_harmful"] > 0


def test_soft_target_control_starts_with_a_vanishing_loss(env) -> None:
    cfg = load_run_config(
        Path(__file__).parent / "soft_targets_experiment.yaml",
        [*SMALL, "optimizer.lr=0.1"],
    )
    prepare(env, cfg, "soft_targets")
    run_dir = run_training(cfg, env, run_id="soft")
    first = read_jsonl(run_dir / "metrics.jsonl")[0]
    assert abs(first["loss"]) < 1e-5
    assert first["grad_norm"] < 1e-4


def test_stage0_checks_run_end_to_end(env) -> None:
    from zero_field_refusal.checks import run_stage0_checks

    cfg = config(
        "run_type=checks",
        "optimizer=adam_frozen",
        "optimizer.lr=1e-3",
        "optimizer.preconditioner.batches=8",
    )
    run = run_stage0_checks(cfg, env, log=lambda _: None)
    summary = json.loads((run.dir / "summary.json").read_text())
    assert 0 <= summary["eos_fraction_at_cap"] <= 1
    assert set(summary["teacher_gap"]) == {"0", "1", "2"}
    assert all(g["token_max_abs"] < 1e-4 for g in summary["teacher_gap"].values())
    proposed = summary["teacher_gap_proposed_thresholds"]
    assert proposed["token_max_abs"] == pytest.approx(
        10 * max(g["token_max_abs"] for g in summary["teacher_gap"].values())
    )
    assert summary["field"]["batches_per_half"] == 4
    assert summary["field"]["constant"]["noise_variance"] > 0
    assert summary["field"]["token_count"]["decision"] in (
        "at_noise_level", "field_detected", "undetermined"
    )  # fmt: skip
    path = summary["path_field"]
    assert path["passed"] and path["chi_square_bound"]["bound_upper"] <= 0.01
    assert path["path_discrepancy"]["kl_max"] < 1e-6
    assert path["soft_target_gradient_one_batch"]["ratio"] < 1e-3
    assert path["euclidean"]["drift_to_noise_upper"] is not None
    assert summary["adam_vs_torch"]["relative_diff"] < 1e-5
    assert summary["run_profile"]["training_step_seconds"] > 0


def test_theta0_evaluation_has_its_own_identity_and_zero_k(env) -> None:
    from zero_field_refusal.config import config_hash
    from zero_field_refusal.train import evaluate_theta0

    cfg = config("run_type=theta0")
    assert config_hash(cfg) != config_hash(config())
    prepare(env, cfg)
    run_dir = evaluate_theta0(cfg, env, run_id="theta0")
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["K_harmful"] == 0 and summary["K_benign"] == 0
    responses = read_jsonl(run_dir / "responses/harmful_judged.jsonl")
    assert {r["condition"] for r in responses} == {"theta0"}
    with pytest.raises(ValueError, match="run_type"):
        run_training(cfg, env)


def test_calibration_runs_feed_the_blind_sample(env) -> None:
    from zero_field_refusal.calibration import calibration_pool
    from zero_field_refusal.labeling import make_calibration_sample
    from zero_field_refusal.train import evaluate_theta0

    env.prompt_set("harmful_calibration")  # fails if the fixture lacks the set
    calibration = ["eval.harmful.prompt_set=harmful_calibration", "eval.utility.benchmark=false"]
    trained_cfg = config(*calibration)
    prepare(env, trained_cfg, "training")
    trained = run_training(trained_cfg, env, run_id="cal-trained")
    theta0 = evaluate_theta0(config(*calibration, "run_type=theta0"), env, run_id="cal-theta0")
    pool = calibration_pool([trained, theta0], lambda cfg: env.prompt_set("harmful_calibration"))
    assert {item["state"] for item in pool} == {"trained", "theta0"}
    assert len(pool) == 2 * len(env.prompt_set("harmful_calibration"))
    blinded, key = make_calibration_sample(pool, 4, strata_key=["model", "state"], seed=1,
                                           min_per_stratum=1)  # fmt: skip
    assert all(set(item) == {"item_id", "prompt", "response"} for item in blinded)
    assert len(key) == 4


def test_accumulated_gradient_does_not_depend_on_the_micro_batch_size(env) -> None:
    from zero_field_refusal.models import freeze
    from zero_field_refusal.train import accumulate_gradient

    cfg = config()
    prepare(env, cfg, "training")
    stream, _ = load_stream_for(cfg, env, "training", 0)
    batch = training_batches(stream, cfg)[0]
    model = env.new_model()
    params = [p for _, p in freeze(model, cfg.train.freeze)]
    grads = []
    for micro in (1, 4):
        for p in params:
            p.grad = None
        accumulate_gradient(model, batch, config(f"execution.micro_batch_size={micro}"), env)
        grads.append([p.grad.clone() for p in params])
    for a, b in zip(*grads, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-8)


def test_second_moment_estimate_uses_the_preconditioner_batches(env) -> None:
    from zero_field_refusal.batches import chunks
    from zero_field_refusal.models import freeze
    from zero_field_refusal.tracking import Run
    from zero_field_refusal.train import _theta0_second_moment, accumulate_gradient

    cfg = config("optimizer=adam_frozen", "optimizer.lr=1e-3", "optimizer.preconditioner.batches=3")
    prepare(env, cfg, "training", "preconditioner")
    model = env.new_model()
    params = [p for _, p in freeze(model, cfg.train.freeze)]
    with Run(cfg, run_id="estimate") as run:
        estimate = _theta0_second_moment(cfg, env, model, params, run)
    stream, _ = load_stream_for(cfg, env, "preconditioner", 0)
    expected = [torch.zeros_like(p) for p in params]
    for batch in chunks(stream, 4):
        for p in params:
            p.grad = None
        accumulate_gradient(model, batch, cfg, env)
        for e, p in zip(expected, params, strict=True):
            e += p.grad**2 / 3
    for v, e in zip(estimate, expected, strict=True):
        torch.testing.assert_close(v, e, rtol=1e-5, atol=1e-12)


def test_training_stops_without_a_passed_path_check(env) -> None:
    cfg = config()
    prepare(env, cfg, "training")
    for record in (env.data_root / "checks").rglob("*.json"):
        record.unlink()
    with pytest.raises(CheckFailed, match="No stage-0 path-field check"):
        run_training(cfg, env, run_id="no-record")
    fingerprint = environment_fingerprint(cfg, env.device)
    write_check_record(env.data_root, fingerprint, certified(), passed=False, run_id="failed")
    with pytest.raises(CheckFailed, match="failed"):
        run_training(cfg, env, run_id="failed-record")
    run_dir = env.data_root.parent / "runs" / "smoke" / "failed-record"
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "failed"
    assert not (run_dir / "metrics.jsonl").exists()  # no optimizer step was taken


def test_training_stops_when_the_stream_comes_from_another_model(env) -> None:
    # A stream sampled by a numerically different teacher: the step-0 gap exceeds the threshold.
    cfg = config()
    student = env.new_model

    def other_model():
        model = student()
        with torch.no_grad():
            model.lm_head.weight.add_(0.01 * torch.randn_like(model.lm_head.weight))
        return model

    env.new_model = other_model
    prepare(env, cfg, "training")
    env.new_model = student
    with pytest.raises(CheckFailed, match="teacher gap"):
        run_training(cfg, env, run_id="mismatch")
    run_dir = env.data_root.parent / "runs" / "smoke" / "mismatch"
    assert not (run_dir / "metrics.jsonl").exists()


@pytest.mark.usefixtures("offline_wandb")
def test_stage0_checks_write_the_record_that_training_needs(env) -> None:
    # Both runs mirror to (offline) W&B, which must accept every summary they write.
    from zero_field_refusal.checks import run_stage0_checks

    for record in (env.data_root / "checks").rglob("*.json"):
        record.unlink()
    mirror = ["tracking.wandb.enabled=true", "tracking.wandb.required=true"]
    checks_cfg = config("run_type=checks", "optimizer=adam_frozen", "optimizer.lr=1e-3",
                        "optimizer.preconditioner.batches=8", *mirror)  # fmt: skip
    run = run_stage0_checks(checks_cfg, env, log=lambda _: None)
    summary = json.loads((run.dir / "summary.json").read_text())
    assert summary["path_field"]["passed"]
    assert summary["path_field"]["chi_square_bound"]["responses"] == 4 * 4  # batches × batch size
    assert set(summary["teacher_gap"]) == {"0", "1", "2"}
    sums = json.loads((run.dir / "path_field_chi_square.json").read_text())["per_response"]
    assert len(sums) == 16 and min(sums) >= 0
    cfg = config(*mirror)
    prepare(env, cfg, "training")
    trained = run_training(cfg, env, run_id="after-checks")
    sums = json.loads((trained / "step0_path_field_chi_square.json").read_text())
    assert len(sums["per_response"]) == cfg.train.steps * cfg.train.batch_size


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"criterion": 0.5}, "certifies the bound"),
        ({"cap": 8}, "certifies responses up to 8"),
        ({"chi_square_bound": BOUND | {"chi_square_per_response": 0.0015}},
         "for this run's steps and batch size"),
        ({"chi_square_bound": BOUND | {"level": 0.9}}, "confidence 0.9"),
    ],
)  # fmt: skip
def test_a_record_unlocks_only_what_it_certified(env, changes, message) -> None:
    cfg = config()
    prepare(env, cfg, "training")
    pass_path_check(env, **changes)
    with pytest.raises(CheckFailed, match=message):
        run_training(cfg, env, run_id="out-of-scope")


def test_a_later_record_keeps_the_earlier_one_in_history(env) -> None:
    fingerprint = environment_fingerprint(config(), env.device)
    write_check_record(env.data_root, fingerprint, certified(), passed=False, run_id="first")
    path = write_check_record(env.data_root, fingerprint, certified(), passed=True, run_id="second")
    history = list((path.parent / "history").glob("*.json"))
    assert {json.loads(h.read_text())["run_id"] for h in history} >= {"first"}


def test_a_nan_gap_fails_the_teacher_gap_check() -> None:
    from omegaconf import OmegaConf

    from zero_field_refusal.prechecks import check_teacher_gap

    thresholds = OmegaConf.create({"token_max_abs": 1e-4, "response_mean_signed_abs": 1e-3})
    with pytest.raises(CheckFailed):
        check_teacher_gap({"token_max_abs": float("nan"), "response_mean_signed": 0.0}, thresholds)
    unset = OmegaConf.create({"token_max_abs": "???", "response_mean_signed_abs": 1e-3})
    with pytest.raises(CheckFailed, match="not set"):
        check_teacher_gap({"token_max_abs": 0.0, "response_mean_signed": 0.0}, unset)


def test_the_run_checks_its_own_stream_against_the_path_bound(env) -> None:
    # Generous gap thresholds let the mismatched stream through the coarse screen; the run's own
    # chi-square bound must still stop it.
    cfg = config(
        "checks.teacher_gap.token_max_abs=10", "checks.teacher_gap.response_mean_signed_abs=10"
    )
    student = env.new_model

    def other_model():
        model = student()
        with torch.no_grad():
            model.lm_head.weight.add_(0.05 * torch.randn_like(model.lm_head.weight))
        return model

    env.new_model = other_model
    prepare(env, cfg, "training")
    env.new_model = student
    with pytest.raises(CheckFailed, match="path-field bound"):
        run_training(cfg, env, run_id="chi-square")


# Appendable streams (study 2): chunks of 3 responses; runs of batch 4 read them in order.
APPENDABLE = ("+teacher.stream=appendable", "+teacher.chunk_size=3")


def chunk_hashes(paths) -> list[str]:
    return [load_manifest(p)["sha256"] for p in paths]


def test_an_appendable_stream_grows_without_changing_its_prefix(env) -> None:
    cfg = config(*APPENDABLE)
    first = extend_stream(cfg, env, 0, 4, batch_size=2)
    assert [p.name for p in first] == ["chunk-00000", "chunk-00001"]
    hashes, before = chunk_hashes(first), StreamPrefix(cfg, env, 0).take(0, 6)
    assert extend_stream(cfg, env, 0, 6, batch_size=2) == []  # long enough already
    more = extend_stream(cfg, env, 0, 11, batch_size=2)
    assert [p.name for p in more] == ["chunk-00002", "chunk-00003"]
    assert chunk_hashes(first) == hashes
    prefix = StreamPrefix(cfg, env, 0)
    assert prefix.available == 12
    assert prefix.take(0, 6) == before
    assert prefix.take(2, 8) == prefix.take(0, 12)[2:8]  # across chunk boundaries
    assert prefix.take(5, 5) == []


def test_a_chunk_depends_only_on_its_seed_and_index(env) -> None:
    cfg = config(*APPENDABLE)
    other = dataclasses.replace(env, data_root=env.data_root.parent / "other")
    extend_stream(cfg, env, 0, 12, batch_size=2)  # in one go
    extend_stream(cfg, other, 0, 3, batch_size=2)  # in three steps
    extend_stream(cfg, other, 0, 6, batch_size=2)
    extend_stream(cfg, other, 0, 12, batch_size=2)
    assert StreamPrefix(cfg, env, 0).take(0, 12) == StreamPrefix(cfg, other, 0).take(0, 12)
    assert StreamPrefix(cfg, env, 0).take(0, 3) != StreamPrefix(cfg, env, 0).take(3, 6)
    extend_stream(cfg, env, 1, 3, batch_size=2)
    assert StreamPrefix(cfg, env, 1).take(0, 3) != StreamPrefix(cfg, env, 0).take(0, 3)


def test_runs_of_any_length_and_batch_size_share_the_prefix_in_order(env) -> None:
    long_small = config(*APPENDABLE, "train.batch_size=2", "train.steps=4")
    short_large = config(*APPENDABLE)  # 2 steps of batch 4
    _, path, _ = locate_stream(short_large, env, "training", 0)
    for cfg in (long_small, config(*APPENDABLE, "train.steps=7")):
        assert locate_stream(cfg, env, "training", 0)[1] == path  # the length is not in the spec
    extend_stream(short_large, env, 0, 8, batch_size=4)
    stream = StreamPrefix(short_large, env, 0).take(0, 8)
    flat = [[s for batch in training_batches(stream, cfg) for s in batch]
            for cfg in (long_small, short_large)]  # fmt: skip
    assert flat[0] == flat[1] == stream
    assert training_batches(stream, short_large)[1] == stream[4:8]


def test_a_run_trains_on_the_prefix_and_records_its_chunks(env) -> None:
    cfg = config(*APPENDABLE)
    prepare(env, cfg)
    extend_stream(cfg, env, 0, 5, batch_size=4)  # 2 chunks, 6 < 8 responses
    with pytest.raises(StreamError, match="extend the stream first"):
        run_training(cfg, env, run_id="short")
    extend_stream(cfg, env, 0, 8, batch_size=4)
    run_dir = run_training(cfg, env, run_id="appendable")
    prefix = StreamPrefix(cfg, env, 0)
    with pytest.raises(StreamError, match="never read"):
        prefix.identity(8)
    prefix.take(0, 8)
    identity = prefix.identity(8)
    assert len(identity["chunks"]) == 3
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["stream/training"] == identity["sha256"]
    recorded = json.loads((run_dir / "meta.json").read_text())["inputs"]["streams"]["training"]
    assert recorded == identity
    assert [m["step"] for m in read_jsonl(run_dir / "metrics.jsonl")] == [0, 1]


def test_chunks_are_checked_when_read(env) -> None:
    cfg = config(*APPENDABLE)
    zero, one = extend_stream(cfg, env, 0, 6, batch_size=4)
    zero.rename(zero.with_name("swap"))
    one.rename(zero)
    zero.with_name("swap").rename(one)
    with pytest.raises(StreamError, match="manifest names chunk 1"):
        StreamPrefix(cfg, env, 0).take(0, 1)
    shutil.rmtree(zero)  # a gap: chunk 1 without chunk 0
    with pytest.raises(StreamError, match="not numbered"):
        StreamPrefix(cfg, env, 0)


def test_a_tampered_chunk_fails_its_hash(env) -> None:
    cfg = config(*APPENDABLE)
    (zero,) = extend_stream(cfg, env, 0, 3, batch_size=4)
    samples = zero / "samples.jsonl"
    samples.write_text(samples.read_text() + "\n")
    with pytest.raises(StreamError, match="sha256"):
        StreamPrefix(cfg, env, 0).take(0, 1)


def test_an_interrupted_chunk_blocks_extension_until_removed(env) -> None:
    cfg = config(*APPENDABLE)
    (zero,) = extend_stream(cfg, env, 0, 3, batch_size=4)
    partial = partial_directory(chunk_directory(zero.parent, 1))
    partial.mkdir()
    with pytest.raises(StreamError, match="Partial directories"):
        extend_stream(cfg, env, 0, 6, batch_size=4)
    assert StreamPrefix(cfg, env, 0).available == 3  # the partial chunk is not part of it
    assert remove_partial_streams(cfg, env, "training", 0) == [partial]
    extend_stream(cfg, env, 0, 6, batch_size=4)
    assert StreamPrefix(cfg, env, 0).available == 6


def test_a_chunk_whose_fingerprint_fails_is_not_published(env, monkeypatch) -> None:
    cfg = config(*APPENDABLE)

    def broken(rows):
        raise RuntimeError("hashing failed")

    monkeypatch.setattr("zero_field_refusal.teacher.distribution_digest", broken)
    with pytest.raises(RuntimeError, match="hashing failed"):
        extend_stream(cfg, env, 0, 3, batch_size=2)
    path = locate_stream(cfg, env, "training", 0)[1]
    assert StreamPrefix(cfg, env, 0).available == 0
    left = [p.name for p in path.iterdir()] if path.exists() else []
    assert left in ([], [".lock"])  # no chunk, no partial directory


def test_fixed_stream_functions_refuse_an_appendable_stream(env) -> None:
    cfg = config(*APPENDABLE)
    with pytest.raises(StreamError, match="extend_stream"):
        generate_stream(cfg, env, "training", 0, batch_size=4)
    with pytest.raises(StreamError, match="StreamPrefix"):
        load_stream_for(cfg, env, "training", 0)
    with pytest.raises(StreamError, match="appendable"):
        extend_stream(config(), env, 0, 8, batch_size=4)


def soft_config(*overrides: str):
    return load_run_config(Path(__file__).parent / "soft_targets_experiment.yaml",
                           [*SMALL, "optimizer.lr=0.1", *APPENDABLE, *overrides])  # fmt: skip


def test_replayed_soft_targets_match_the_stream_across_batches_and_chunks(env) -> None:
    cfg = config(*APPENDABLE)
    extend_stream(cfg, env, 0, 9, batch_size=2)  # each chunk of 3: generation batches of 2 and 1
    prefix, reference = StreamPrefix(cfg, env, 0), env.new_model()
    replayer = ReplayedSoftTargets(prefix, reference, env)
    rows = replayer.take(1, 8)
    for s, row in zip(prefix.take(1, 8), rows, strict=True):
        assert row.shape == (len(s.response_ids), reference.config.vocab_size)
        picked = row.gather(1, torch.tensor(s.response_ids)[:, None])[:, 0]
        assert torch.equal(picked, torch.tensor(s.logprobs))
    assert [(r["chunk"], r["batch"], r["responses"]) for r in replayer.records] == [
        (0, 0, 2), (0, 1, 1), (1, 0, 2), (1, 1, 1), (2, 0, 2)
    ]  # fmt: skip
    assert all(r["max_abs"] == 0.0 for r in replayer.records)
    assert all(r["digest_equal"] for r in replayer.records)
    assert all(r["projection_max_abs"] == 0.0 for r in replayer.records)
    assert len(replayer._batches) <= 2  # memory does not grow with the run
    batches = load_manifest(chunk_directory(prefix.path, 0))["fingerprints"]["batches"]
    assert [(b["first"], b["responses"]) for b in batches] == [(0, 2), (2, 1)]


def altered_replay(change):
    """replay_distributions whose non-sampled entries are changed by ``change``, with the
    fingerprint of the changed rows."""
    from zero_field_refusal.teacher import fingerprint, replay_distributions

    def replay(model, samples, **kwargs):
        rows, errors, _ = replay_distributions(model, samples, **kwargs)
        changed = []
        for s, row in zip(samples, rows, strict=True):
            mask = torch.ones_like(row, dtype=torch.bool)
            mask[torch.arange(len(s.response_ids)), torch.tensor(s.response_ids)] = False
            changed.append(torch.where(mask, change(row), row))
        return changed, errors, fingerprint(changed)

    return replay


@pytest.mark.parametrize(
    ("change", "projections_close"),
    [
        (lambda row: row.flip(-1), False),  # unsampled entries moved: the reviewer's case
        (lambda row: row + 1e-6, True),  # a few ulps: bytes differ, projections barely move
    ],
)
def test_only_an_equal_digest_accepts_a_replay(env, monkeypatch, change, projections_close):
    cfg = config(*APPENDABLE)
    extend_stream(cfg, env, 0, 3, batch_size=2)
    monkeypatch.setattr("zero_field_refusal.runtime.replay_distributions", altered_replay(change))
    replayer = ReplayedSoftTargets(StreamPrefix(cfg, env, 0), env.new_model(), env)
    with pytest.raises(StreamError, match="the digest differs"):
        replayer.take(0, 2)
    (record,) = replayer.records  # recorded before the check
    assert record["max_abs"] == 0.0 and not record["digest_equal"]
    assert (record["projection_max_abs"] < 1e-5) == projections_close  # a diagnostic only


def test_a_stream_is_replayed_only_on_the_device_model_that_generated_it(env) -> None:
    cfg = config(*APPENDABLE)
    (chunk,) = extend_stream(cfg, env, 0, 3, batch_size=2)
    assert load_manifest(chunk)["device"] == "cpu"
    manifest = json.loads((chunk / "manifest.json").read_text())
    manifest["device"] = "NVIDIA L40S"
    (chunk / "manifest.json").write_text(json.dumps(manifest))
    replayer = ReplayedSoftTargets(StreamPrefix(cfg, env, 0), env.new_model(), env)
    with pytest.raises(StreamError, match="generated on NVIDIA L40S, the replay runs on cpu"):
        replayer.take(0, 1)
    assert replayer.records == []  # stopped before any replay


def test_appendable_configs_generate_no_soft_target_stream(env, monkeypatch) -> None:
    import runpy

    cfg = soft_config()
    with pytest.raises(StreamError, match="replays its targets"):
        generate_stream(cfg, env, "soft_targets", 0, batch_size=4)
    script = Path(__file__).parents[1] / "scripts" / "generate_teacher.py"
    monkeypatch.setattr("sys.argv", [str(script), "experiments/smoke", "--stream", "soft_targets",
                                     "--seeds", "0", *APPENDABLE])  # fmt: skip
    monkeypatch.setattr("zero_field_refusal.runtime.environment", None)  # never reached
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(script), run_name="__main__")
    assert stopped.value.code == 2


def test_soft_target_control_replays_the_prefix_of_the_regular_run(env) -> None:
    regular, soft = config(*APPENDABLE), soft_config()
    paths = [locate_stream(cfg, env, "training", 0)[1] for cfg in (soft, regular)]
    assert paths[0] == paths[1]
    prepare(env, regular)
    extend_stream(regular, env, 0, 8, batch_size=4)
    soft_dir = run_training(soft, env, run_id="soft-appendable")
    regular_dir = run_training(regular, env, run_id="regular-appendable")
    first = read_jsonl(soft_dir / "metrics.jsonl")[0]
    assert abs(first["loss"]) < 1e-5 and first["grad_norm"] < 1e-4
    summary = json.loads((soft_dir / "summary.json").read_text())
    assert summary["soft_targets_replay"]["max_abs"] == 0.0
    assert len(read_jsonl(soft_dir / "soft_targets_replay.jsonl")) == 3  # chunks 0-2, one batch
    streams = [json.loads((d / "meta.json").read_text())["inputs"]["streams"]["training"]
               for d in (soft_dir, regular_dir)]  # fmt: skip
    assert streams[0] == streams[1]


def test_a_replay_beyond_the_bound_stops_the_run(env, monkeypatch) -> None:
    cfg = soft_config()
    prepare(env, cfg)
    extend_stream(cfg, env, 0, 8, batch_size=4)
    monkeypatch.setattr("zero_field_refusal.runtime.REPLAY_MAX_ABS", -1.0)
    with pytest.raises(StreamError, match="differ from the stored"):
        run_training(cfg, env, run_id="replay-bound")
    run_dir = env.data_root.parent / "runs" / "smoke" / "replay-bound"
    assert read_jsonl(run_dir / "soft_targets_replay.jsonl")[0]["max_abs"] == 0.0  # kept
    assert not (run_dir / "metrics.jsonl").exists()  # stopped before the first step


def test_the_curvature_run_measures_batches_and_the_pooled_hessian(env) -> None:
    from omegaconf import OmegaConf

    from zero_field_refusal.curvature import run_curvature

    settings = {"batch_sizes": [2, 4], "batches": 2, "starts": 2, "pooled_responses": 8,
                "pooled_starts": 1, "steps": 4, "tolerance": 0.01}  # fmt: skip
    cfg = OmegaConf.merge(config(*APPENDABLE, "run_type=curvature"), {"curvature": settings})
    extend_stream(cfg, env, 0, 8, batch_size=4)
    run_dir = run_curvature(cfg, env, run_id="curvature")
    result = json.loads((run_dir / "curvature.json").read_text())
    assert [(r["batch_size"], r["batch"], len(r["starts"])) for r in result["single"]] == [
        (size, b, 2) for size in (2, 4) for b in range(2)
    ]
    for measured in [*result["single"], result["pooled"]]:
        tops = [r["top"] for r in measured["starts"]]
        assert measured["top"] == max(tops)
        assert all(r["steps"] == 4 and r["top"] >= r["bottom"] for r in measured["starts"])
        every = all(r["ritz_converged"] for r in measured["starts"])
        assert measured["ritz_converged"] == (measured["starts_agree"] and every)
    assert result["pooled"]["responses"] == 8 and len(result["pooled"]["starts"]) == 1
    summary = json.loads((run_dir / "summary.json").read_text())["curvature"]
    assert summary["pooled"]["lr_at_2_over_top"] == pytest.approx(2 / summary["pooled"]["top"])
    assert set(summary["single_batch_2"]) >= {"top_median", "not_ritz_converged"}
    meta = json.loads((run_dir / "meta.json").read_text())
    assert len(meta["inputs"]["streams"]["training"]["chunks"]) == 3


MONITORED = ("+eval.monitor.samples_per_prompt=2", "+eval.monitor.every=5")


def test_the_monitor_panel_has_its_own_prompts_stream_and_seeds(env) -> None:
    from zero_field_refusal.panels import Monitor
    from zero_field_refusal.teacher import stream_seeds

    cfg = config(*MONITORED)
    prepare(env, cfg, "kl_monitor")
    monitor_spec, monitor_path, _ = locate_stream(cfg, env, "kl_monitor", None)
    benign_spec, benign_path, _ = locate_stream(cfg, env, "kl_benign", None)
    assert monitor_path != benign_path and monitor_spec["prompts"] != benign_spec["prompts"]
    assert monitor_spec["samples_per_prompt"] == 2 and benign_spec["samples_per_prompt"] == 2
    assert set(stream_seeds("kl_monitor", None)).isdisjoint(stream_seeds("kl_benign", None))
    samples, _ = load_stream_for(cfg, env, "kl_monitor", None)
    assert [s.prompt_id for s in samples] == [f"monitor:{i}" for i in range(3) for _ in range(2)]

    from zero_field_refusal.tracking import Run

    reference, student = env.new_model(), env.new_model()
    with Run(cfg, run_id="monitor") as run:
        monitor = Monitor(cfg, env, run, reference)
        assert monitor(student) == pytest.approx(0.0, abs=1e-6)  # θ = θ₀
        with torch.no_grad():  # the near-uniform head: move it to move the distributions
            student.lm_head.weight.add_(0.5 * torch.randn_like(student.lm_head.weight))
        assert monitor(student) > 1e-2
    meta = json.loads((run.dir / "meta.json").read_text())
    assert meta["inputs"]["streams"]["kl_monitor"]["count"] == 6


MONITOR_V2 = (*MONITORED, "+eval.monitor.version=2", "+eval.monitor.prompt_set=alpaca_benign_kl",
              "+eval.monitor.drive=1", "+eval.monitor.cache.primary=device",
              "+eval.monitor.cache.secondary=host")  # fmt: skip


@pytest.mark.parametrize("primary", ["device", "host"])
def test_monitor_version_2_answers_the_scoring_prompts_with_seeds_of_its_own(env, primary) -> None:
    from zero_field_refusal.panels import Monitor
    from zero_field_refusal.teacher import stream_seeds
    from zero_field_refusal.tracking import Run

    v1 = config(*MONITORED)
    v2 = config(*MONITOR_V2, f"eval.monitor.cache.primary={primary}")  # m = 1's cache, both ways
    prepare(env, v2, "kl_monitor")
    spec1, path1, _ = locate_stream(v1, env, "kl_monitor", None)
    spec2, path2, _ = locate_stream(v2, env, "kl_monitor", None)
    benign, benign_path, _ = locate_stream(v2, env, "kl_benign", None)
    assert "monitor_version" not in spec1 and spec2["monitor_version"] == 2  # version 1 unchanged
    assert spec2["prompts"] == benign["prompts"] and path2 not in (path1, benign_path)
    seeds = [stream_seeds("kl_monitor", None), stream_seeds("kl_monitor", None, version=2),
             stream_seeds("kl_benign", None)]  # fmt: skip
    assert len({seed for pair in seeds for seed in pair}) == 6
    manifest = load_manifest(path2)
    assert (manifest["prompt_seed"], manifest["sampling_seed"]) == seeds[1]
    samples, _ = load_stream_for(v2, env, "kl_monitor", None)
    scoring, _ = load_stream_for(v2, env, "kl_benign", None)
    assert [s.prompt_id for s in samples] == [s.prompt_id for s in scoring]  # 2 per prompt each

    reference, student = env.new_model(), env.new_model()
    with Run(v2, run_id="monitor-v2") as run:
        monitor = Monitor(v2, env, run, reference)
        assert monitor.primary == samples[0::2] and monitor.secondary == samples[1::2]
        with torch.no_grad():
            student.lm_head.weight.add_(0.5 * torch.randn_like(student.lm_head.weight))
        k = monitor(student)
        shadow = monitor.shadow(student, monitor.last)
        assert shadow["K_primary"] == k and len(shadow["secondary"]) == len(samples) // 2
        assert shadow == monitor.shadow(student, None)  # the same state, read again
        checks = monitor.check(student)
        assert set(checks) == {"primary", "secondary"}
        assert all(c["bitwise_equal"] and c["max_abs_diff"] == 0.0 for c in checks.values())
        assert monitor.caches["primary"].device.type == "cpu"  # host, or the CPU test device
        assert torch.equal(monitor.last, monitor.uncached(student, "primary"))


def test_cached_reference_logits_give_the_uncached_k_bit_for_bit(env) -> None:
    from zero_field_refusal.evaluation import ReferenceLogits, sequence_kl

    cfg = config()
    prepare(env, cfg)
    samples, _ = load_stream_for(cfg, env, "kl_benign", None)
    reference, student = env.new_model(), env.new_model()
    with torch.no_grad():
        student.lm_head.weight.add_(0.5 * torch.randn_like(student.lm_head.weight))
    common = dict(pad_id=env.pad_id, micro_batch_size=3)
    plain = sequence_kl(student, reference, samples, **common)
    assert plain.min() > 0
    for device in ("cpu", torch.device("cpu")):
        cache = ReferenceLogits(reference, samples, device=device, **common)
        assert cache.bytes > 0
        assert torch.equal(sequence_kl(student, reference, samples, cache=cache, **common), plain)
    with pytest.raises(ValueError, match="other samples or micro-batches"):
        sequence_kl(student, reference, samples, pad_id=env.pad_id, micro_batch_size=2,
                    cache=cache)  # fmt: skip


def test_writers_of_one_stream_take_turns_under_its_lock(env) -> None:
    import threading
    import time

    from zero_field_refusal.runtime import _locked

    cfg = config(*APPENDABLE)
    path = locate_stream(cfg, env, "training", 0)[1]
    created: list = []
    with _locked(path):  # another writer holds the stream
        writer = threading.Thread(
            target=lambda: created.extend(extend_stream(cfg, env, 0, 6, batch_size=4))
        )
        writer.start()
        time.sleep(0.5)
        assert writer.is_alive() and StreamPrefix(cfg, env, 0).available == 0
    writer.join(timeout=60)
    assert [p.name for p in created] == ["chunk-00000", "chunk-00001"]
    assert extend_stream(cfg, env, 0, 6, batch_size=4) == []  # counted after taking the lock
