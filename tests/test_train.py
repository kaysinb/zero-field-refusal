"""End-to-end CPU runs on a tiny model: streams → training → evaluation → run directory."""

import json
from pathlib import Path

import pytest
import torch
from conftest import tiny_model, tiny_tokenizer

from zero_field_refusal.models import install_neutral_generation_config
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.runtime import Environment, generate_stream, load_stream_for
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
]


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
        "factual_qa": items("fact", 4, answers=["w1"]),
    }
    sets["alpaca_heldout"] = [{**x, "reference": words(rng, 5)} for x in items("held", 3)]
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

    return Environment(
        tokenizer=tokenizer,
        end_ids=end_ids,
        pad_id=tokenizer.pad_token_id,
        device=torch.device("cpu"),
        new_model=new_model,
        prompt_set=lambda name: sets[name],
        data_root=tmp_path / "data",
    )


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
    assert json.loads((run_dir / "meta.json").read_text())["status"] == "completed"


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
