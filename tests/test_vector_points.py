"""Study 5's points along a vector (extension 2, ``vector_points``) on a tiny fresh F run: the
readouts at θ₀ + a·v from the run's endpoint and from a vector file, the sign, the matched-K step,
the conditions the judging reads, and the refusals."""

from __future__ import annotations

import json

import pytest
import torch
from test_ladder import stub_k
from test_schedule import scheduled
from test_study3_campaigns import script_module
from test_study5_options import F_RUN, ready
from test_surrogate import SPEC_YAML
from test_train import APPENDABLE, SMALL
from test_train import env as env  # noqa: F401  (the fixture)

from zero_field_refusal import ladder, surrogate, vector_points, vectors
from zero_field_refusal.evaluation import vector_condition
from zero_field_refusal.models import freeze
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.run_judging import vector_conditions
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.tensor_hash import tensors_sha256

OVERRIDES = [*F_RUN, "+study5.role=fresh"]


@pytest.fixture
def fresh(env, tmp_path):  # noqa: F811
    """A tiny fresh F run with its endpoint checkpoint, the surrogate spec and the script."""
    torch.set_num_threads(1)
    ckpt = tmp_path / "ckpt"
    cfg = ready(env, scheduled(tmp_path, *OVERRIDES, f"tracking.endpoint_checkpoint={ckpt}",
                               policy="short"))  # fmt: skip
    f_dir = run_scheduled(cfg, env, run_id="f-fresh")
    spec_path = tmp_path / "surrogate.yaml"
    spec_path.write_text(SPEC_YAML)
    config = [str(tmp_path / "study2.yaml"), *SMALL, *APPENDABLE, "teacher.chunk_size=12"]
    return {"cfg": cfg, "f_dir": f_dir, "ckpt": ckpt, "spec": spec_path, "config": config}


def script(env, monkeypatch):  # noqa: F811
    module = script_module(REPO_ROOT / "scripts" / "study5_vector_points.py", "s5_vector")
    monkeypatch.setattr(module, "environment", lambda cfg, device: env)
    return module


def common(fresh, out, *extra) -> list[str]:
    return [*fresh["config"], *OVERRIDES, "--out", str(out), "--surrogate", str(fresh["spec"]),
            "--panel-size", "3", *extra]  # fmt: skip


def test_points_from_the_endpoint(env, fresh, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    module = script(env, monkeypatch)
    out = tmp_path / "rec" / "disp.json"
    argv = common(fresh, out, "--label", "disp", "--f-run", str(fresh["f_dir"]), "--checkpoints",
                  str(fresh["ckpt"]), "--points", "1", "-1", "0.5", "--full", "1")  # fmt: skip
    assert module.main(argv) == 0
    record = json.loads(out.read_text())
    rows = record["rows"]
    assert [r["a"] for r in rows] == [1.0, -1.0, 0.5]
    assert [r["condition"] for r in rows] == ["vector_disp_+1", "vector_disp_-1",
                                              "vector_disp_+0.5"]  # fmt: skip
    assert [r["evaluated"] for r in rows] == [True, False, False]
    assert rows[1]["reason"] == rows[2]["reason"] == "readout_only"
    run_dir = tmp_path / "runs" / "study5_vector" / record["run_id"]
    assert vector_conditions(run_dir) == ["vector_disp_+1"]

    # a = 1 is the endpoint: r and K_benign are the endpoint model's (θ₀ + 1·Δθ in fp32)
    cfg = fresh["cfg"]
    end = torch.load(fresh["ckpt"] / "f-fresh.pt", weights_only=True)
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    with torch.no_grad():
        for name, p in trainable:
            p.copy_(end[name])
    model.eval()
    spec = surrogate.load_spec(fresh["spec"])
    harmful = [item["text"] for item in env.prompt_set("harmful_kl")]
    r = surrogate.surrogate_value(model, env.tokenizer, harmful, spec.refusals, spec.compliances,
                                  device=env.device, micro_batch_size=cfg.execution.eval_batch_size,
                                  max_tokens=spec.max_tokens)  # fmt: skip
    assert rows[0]["r"] == pytest.approx(r, rel=1e-5, abs=1e-6)
    assert rows[0]["r"] == pytest.approx(rows[0]["harmful_R"] - rows[0]["harmful_C"], abs=1e-9)
    f_eval = [json.loads(line) for line in (fresh["f_dir"] / "evaluations.jsonl").read_text()
              .splitlines() if line.strip()][-1]  # fmt: skip
    assert rows[0]["K_benign"] == pytest.approx(f_eval["K_benign"], rel=1e-5, abs=1e-7)
    assert rows[0]["full"]["K_benign"] == pytest.approx(rows[0]["K_benign"], rel=1e-9)

    printed = capsys.readouterr().out
    benign = [item["text"] for item in env.prompt_set("alpaca_benign_kl")]
    for text in harmful + benign + list(spec.continuations):
        assert text not in printed
    assert module.main(argv) == 1  # never overwritten


def test_a_vector_file_and_the_sign(env, fresh, tmp_path, monkeypatch) -> None:  # noqa: F811
    cfg = fresh["cfg"]
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    names = [n for n, _ in trainable]
    theta0 = {n: p.detach().clone() for n, p in trainable}
    end = torch.load(fresh["ckpt"] / "f-fresh.pt", weights_only=True)
    path = tmp_path / "vec" / "disp.pt"
    vectors.save_vector(path, vectors.displacement(end, theta0, names), names, kind="displacement",
                        theta0_sha256=tensors_sha256(trainable))  # fmt: skip
    module = script(env, monkeypatch)
    records = {}
    for label, extra in (("plus", ("--points", "1", "-1")),
                         ("minus", ("--sign", "-1", "--points", "1"))):  # fmt: skip
        out = tmp_path / "rec" / f"{label}.json"
        assert module.main(common(fresh, out, "--label", label, "--vector", str(path), *extra)) == 0
        records[label] = json.loads(out.read_text())
    plus, minus = records["plus"]["rows"], records["minus"]["rows"]
    for key in ("K_benign", "K_harmful", "r", "benign_R", "benign_C"):
        assert minus[0][key] == plus[1][key], key  # −v at 1 is v at −1, exactly
    out = tmp_path / "rec" / "disp.json"
    run = ("--f-run", str(fresh["f_dir"]), "--checkpoints", str(fresh["ckpt"]))
    assert module.main(common(fresh, out, "--label", "disp", *run, "--points", "1")) == 0
    from_run = json.loads(out.read_text())["rows"][0]
    assert from_run["r"] == pytest.approx(plus[0]["r"], rel=1e-5, abs=1e-6)
    assert from_run["K_benign"] == pytest.approx(plus[0]["K_benign"], rel=1e-5, abs=1e-7)
    assert records["plus"]["vector"]["sha256"] == vectors.read_record(path)["sha256"]


def test_the_matched_k_point(env, fresh, tmp_path, monkeypatch) -> None:  # noqa: F811
    k_benign, box = stub_k(ladder.K_STAR, 2.0)  # K* at 2·‖v‖ from θ₀
    module = script(env, monkeypatch)
    end = torch.load(fresh["ckpt"] / "f-fresh.pt", weights_only=True)
    theta0 = {n: p.detach() for n, p in freeze(env.new_model(), fresh["cfg"].train.freeze)}
    box["norm"] = vectors.norm(vectors.displacement(end, theta0, list(theta0)))
    monkeypatch.setattr(ladder, "k_benign", k_benign)
    out = tmp_path / "rec" / "match.json"
    argv = common(fresh, out, "--label", "disp", "--f-run", str(fresh["f_dir"]), "--checkpoints",
                  str(fresh["ckpt"]), "--points", "1", "--match")  # fmt: skip
    assert module.main(argv) == 0
    record = json.loads(out.read_text())
    assert record["matched"]["status"] in ("matched", "out_of_tolerance")
    assert record["matched"]["c"] == pytest.approx(2.0, rel=0.05)
    assert record["conditions"] == ["vector_disp_+1", "vector_disp_kstar"]
    assert record["evaluated"] == ["vector_disp_kstar"]
    assert record["rows"][-1]["matched_k"] is True


def test_refusals(env, fresh, tmp_path, monkeypatch) -> None:  # noqa: F811
    module = script(env, monkeypatch)
    out = tmp_path / "rec" / "x.json"
    run = ("--f-run", str(fresh["f_dir"]), "--checkpoints", str(fresh["ckpt"]))
    assert module.main(common(fresh, out, "--label", "x", "--points", "1")) == 1  # no vector
    assert module.main(common(fresh, out, "--label", "Bad-Label", *run)) == 1
    assert module.main(common(fresh, out, "--label", "x", *run, "--points", "1", "1.0")) == 1
    assert module.main(common(fresh, out, "--label", "x", "--f-run", str(fresh["f_dir"]))) == 1
    assert not out.exists()
    with pytest.raises(ValueError, match="among the points"):
        vector_points.run_vector(fresh["cfg"], env, label="x", vector=tmp_path / "none.pt",
                                 points=[1.0], full=[2.0])  # fmt: skip
    # a vector built on another θ₀
    cfg = fresh["cfg"]
    trainable = freeze(env.new_model(), cfg.train.freeze)
    names = [n for n, _ in trainable]
    path = tmp_path / "vec" / "other.pt"
    vectors.save_vector(path, {n: torch.zeros_like(p) for n, p in trainable}, names,
                        kind="displacement", theta0_sha256="another")  # fmt: skip
    with pytest.raises(ValueError, match="another θ₀"):
        vector_points.run_vector(cfg, env, label="x", vector=path, points=[1.0],
                                 surrogate_spec=fresh["spec"], panel_size=3)  # fmt: skip


def test_judging_refuses_an_unfinished_vector_run(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="without its rows"):
        vector_conditions(run_dir)
    row = {"a": 1.0, "condition": vector_condition("x", 1.0), "evaluated": True}
    (run_dir / "vector_points.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        vector_conditions(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(
        {"vector": {"conditions": ["vector_x_+1"], "evaluated": ["vector_x_+1"]}}))  # fmt: skip
    assert vector_conditions(run_dir) == ["vector_x_+1"]
    with pytest.raises(ValueError):
        vector_condition("x", "other")
