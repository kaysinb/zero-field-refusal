"""The launcher's endpoint checkpoint and the hashes of θ₀ and v₀ (study 3's bridge, launcher side).

A fixed 25-step schedule on the tiny model; ``tracking.endpoint_checkpoint`` saves the trainable
parameters right after the last update and before the step-25 evaluation, and changes nothing else.
"""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from test_schedule import arm_config, built_optimizers, read_jsonl
from test_schedule import env as env  # noqa: F401  (the fixture)

from zero_field_refusal import fa_control_diagnostic as fcd
from zero_field_refusal import schedule
from zero_field_refusal.config import config_hash
from zero_field_refusal.models import freeze
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.tensor_hash import tensors_sha256


def final_models(monkeypatch) -> list:
    """The students the runs train, kept after the run (their parameters and buffers)."""
    kept, build = [], schedule.freeze

    def keep(model, *args, **kwargs):
        kept.append(model)
        return build(model, *args, **kwargs)

    monkeypatch.setattr(schedule, "freeze", keep)
    return kept


def rng_state() -> dict:
    state = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
             "python": random.getstate()}  # fmt: skip
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def same_rng(a: dict, b: dict) -> bool:
    x, y = a["numpy"], b["numpy"]
    return (torch.equal(a["torch"], b["torch"]) and a["python"] == b["python"]
            and x[0] == y[0]
            and all(np.array_equal(i, j) for i, j in zip(x[1:], y[1:], strict=True))
            and ("cuda" not in a or torch.equal(a["cuda"], b["cuda"])))  # fmt: skip


def scientific(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in row.items() if k != "time_s"} for row in rows]


def test_the_checkpoint_changes_no_training(env, tmp_path, monkeypatch) -> None:
    built, models = built_optimizers(monkeypatch), final_models(monkeypatch)
    out = {}
    for name, ckpt in (("off", None), ("on", str(tmp_path / "ckpt"))):
        cfg = arm_config(env, tmp_path, "adamw", f"tracking.endpoint_checkpoint={ckpt}")
        run_dir = run_scheduled(cfg, env, run_id=f"bridge-{name}")
        params = [p for g in built[-1].param_groups for p in g["params"]]
        out[name] = {
            "metrics": scientific(read_jsonl(run_dir / "metrics.jsonl")),
            "params": [p.detach().clone() for p in params],
            "grads": [p.grad for p in params],
            "buffers": {n: b.clone() for n, b in models[-1].named_buffers()},
            "rng": rng_state(),
            "records": [read_jsonl(run_dir / f) for f in
                        ("monitor.jsonl", "evaluations.jsonl")],
        }  # fmt: skip
    off, on = out["off"], out["on"]
    assert off["metrics"] == on["metrics"]  # exact, field by field
    assert off["records"] == on["records"]
    assert all(torch.equal(a, b) for a, b in zip(off["params"], on["params"], strict=True))
    assert all(g is None for g in off["grads"] + on["grads"])
    assert off["buffers"].keys() == on["buffers"].keys()
    assert all(torch.equal(off["buffers"][n], on["buffers"][n]) for n in off["buffers"])
    assert same_rng(off["rng"], on["rng"])


def test_the_checkpoint_holds_the_last_update_before_evaluation(env, tmp_path, monkeypatch) -> None:
    built = built_optimizers(monkeypatch)
    ckpt = tmp_path / "ckpt"
    cfg = arm_config(env, tmp_path, "adamw", f"tracking.endpoint_checkpoint={ckpt}")
    order = []
    evaluate = schedule._Loop._evaluate

    def spying(self, *args, **kwargs):  # the file exists before the step-25 evaluation starts
        order.append(((ckpt / "bridge.pt").is_file(), self.steps))
        return evaluate(self, *args, **kwargs)

    monkeypatch.setattr(schedule._Loop, "_evaluate", spying)
    run_dir = run_scheduled(cfg, env, run_id="bridge")
    assert order == [(True, 25)]
    path = ckpt / "bridge.pt"
    saved = torch.load(path, map_location="cpu", weights_only=True)
    params = [p for g in built[-1].param_groups for p in g["params"]]
    assert len(saved) == len(params)
    assert all(
        torch.equal(v, p.detach().cpu()) for v, p in zip(saved.values(), params, strict=True)
    )
    assert not list(ckpt.glob(".*partial"))
    summary = json.loads((run_dir / "summary.json").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    record = summary["endpoint_checkpoint"]
    assert meta["inputs"]["endpoint_checkpoint"] == record
    assert record["path"] == str(path.resolve()) and record["step"] == cfg.train.steps == 25
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert record["names_sha256"] == hashlib.sha256(json.dumps(list(saved)).encode()).hexdigest()
    assert summary["terminal"]["stop"]["reason"] == "steps"


def test_an_early_stop_writes_no_checkpoint(env, tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "ckpt"
    cfg = arm_config(env, tmp_path, "adamw", f"tracking.endpoint_checkpoint={ckpt}")
    norms = iter([1.0, 1.0, 1.0, 50.0])
    real = schedule._norm

    def norm(tensors):  # the gradient norms are scripted, the weight check stays real
        first = next(t for t in tensors if t is not None)
        return next(norms) if first.requires_grad is False and first.grad is None else real(tensors)

    monkeypatch.setattr(schedule, "_norm", norm)
    run_dir = run_scheduled(cfg, env, run_id="spiked")
    terminal = json.loads((run_dir / "terminal.json").read_text())
    assert terminal["stop"]["reason"] == "gradient_spike" and terminal["steps"] == 3
    assert not ckpt.exists() or not list(ckpt.iterdir())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert "endpoint_checkpoint" in summary and summary["endpoint_checkpoint"] is None
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["inputs"]["endpoint_checkpoint"] is None


def test_an_interrupted_save_leaves_no_file_at_the_final_path(env, tmp_path, monkeypatch) -> None:
    ckpt = tmp_path / "ckpt"
    cfg = arm_config(env, tmp_path, "adamw", f"tracking.endpoint_checkpoint={ckpt}")

    def broken(obj, path, *args, **kwargs):
        Path(path).write_bytes(b"half")  # the write began
        raise OSError("disk full")

    monkeypatch.setattr(schedule.torch, "save", broken)
    with pytest.raises(OSError, match="disk full"):
        run_scheduled(cfg, env, run_id="cut")
    assert list(ckpt.iterdir()) == []  # no final file, and the partial one is removed


@pytest.mark.parametrize("arm", ["adamw_prewarmed", "adam_frozen"])
def test_v0_hash_matches_the_harness(env, tmp_path, arm) -> None:
    cfg = arm_config(env, tmp_path, arm, "tracking.provenance_hashes=true",
                     "train.steps=3", "schedule.evaluate_at=[3]")  # fmt: skip
    run_dir = run_scheduled(cfg, env, run_id=f"hashes-{arm}")
    inputs = json.loads((run_dir / "meta.json").read_text())["inputs"]
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    names, params = [n for n, _ in trainable], [p for _, p in trainable]
    v0, _ = fcd.second_moment(cfg, env, model, params)
    assert inputs["v0_sha256"] == fcd.tensors_sha256(list(zip(names, v0, strict=True)))
    theta0 = env.new_model()
    assert inputs["theta0_trainable_sha256"] == tensors_sha256(
        [(n, p) for n, p in freeze(theta0, cfg.train.freeze)]
    )
    assert "endpoint_checkpoint" not in inputs  # the hashes do not need the checkpoint


def test_the_hashes_are_independent_of_the_checkpoint_flag(env, tmp_path) -> None:
    cfg = arm_config(env, tmp_path, "adamw", "train.steps=2", "schedule.evaluate_at=[2]")
    inputs = json.loads((run_scheduled(cfg, env, run_id="plain") / "meta.json").read_text())[
        "inputs"
    ]
    assert "theta0_trainable_sha256" not in inputs and "v0_sha256" not in inputs  # study 2: nothing
    cfg = arm_config(env, tmp_path, "adamw", "tracking.provenance_hashes=true",
                     "train.steps=2", "schedule.evaluate_at=[2]")  # fmt: skip
    inputs = json.loads((run_scheduled(cfg, env, run_id="hashed") / "meta.json").read_text())[
        "inputs"
    ]
    assert len(inputs["theta0_trainable_sha256"]) == 64 and "v0_sha256" not in inputs  # AdamW


def test_the_flag_is_outside_the_config_hash(env, tmp_path) -> None:
    plain = arm_config(env, tmp_path, "adamw")
    flagged = arm_config(env, tmp_path, "adamw", f"tracking.endpoint_checkpoint={tmp_path}",
                         "tracking.provenance_hashes=true")  # fmt: skip
    assert config_hash(plain) == config_hash(flagged)


def test_study_2_configs_write_nothing(env, tmp_path) -> None:
    cfg = arm_config(env, tmp_path, "adamw", "train.steps=2", "schedule.evaluate_at=[2]")
    run_dir = run_scheduled(cfg, env, run_id="study2")
    assert "endpoint_checkpoint" not in json.loads((run_dir / "summary.json").read_text())
    assert "endpoint_checkpoint" not in json.loads((run_dir / "meta.json").read_text())["inputs"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG state needs a GPU")
def test_the_checkpoint_leaves_the_cuda_rng_alone(env, tmp_path) -> None:
    states = []
    for ckpt in (None, str(tmp_path / "ckpt")):
        cfg = arm_config(env, tmp_path, "adamw", f"tracking.endpoint_checkpoint={ckpt}",
                         "train.steps=3", "schedule.evaluate_at=[3]")  # fmt: skip
        run_scheduled(cfg, env, run_id=f"cuda-{ckpt is None}")
        states.append(torch.cuda.get_rng_state())
    assert torch.equal(*states)
