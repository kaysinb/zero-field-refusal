"""Pairs of vectors and the surrogate at checkpoints (extension 2, scripts/study5_pairs.py) on a
tiny fresh F run with step checkpoints."""

from __future__ import annotations

import json
import math

import pytest
import torch
from test_schedule import scheduled
from test_study3_campaigns import script_module
from test_study5_options import F_RUN, ready
from test_surrogate import SPEC_YAML
from test_train import APPENDABLE, SMALL
from test_train import env as env  # noqa: F401  (the fixture)

from zero_field_refusal import surrogate, vectors
from zero_field_refusal.models import freeze
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.tensor_hash import file_sha256, tensors_sha256

OVERRIDES = [*F_RUN, "+study5.role=fresh"]


def test_pairs_and_r_at_checkpoints(env, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    torch.set_num_threads(1)
    ckpt = tmp_path / "ckpt"
    cfg = ready(env, scheduled(tmp_path, *OVERRIDES, f"tracking.endpoint_checkpoint={ckpt}",
                               "tracking.step_checkpoints=[1]", policy="short"))  # fmt: skip
    run_scheduled(cfg, env, run_id="f")
    end, one = ckpt / "f.pt", ckpt / "f-step-1.pt"
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    names = [n for n, _ in trainable]
    theta0 = {n: p.detach().clone() for n, p in trainable}
    e = torch.load(end, weights_only=True)
    s1 = torch.load(one, weights_only=True)
    d_end = vectors.displacement(e, theta0, names)
    vec = tmp_path / "v.pt"
    vectors.save_vector(vec, {n: 2 * d_end[n] for n in names}, names, kind="x",
                        theta0_sha256=tensors_sha256(trainable))  # fmt: skip
    spec_path = tmp_path / "surrogate.yaml"
    spec_path.write_text(SPEC_YAML)
    module = script_module(REPO_ROOT / "scripts" / "study5_pairs.py", "s5_pairs")
    monkeypatch.setattr(module, "environment", lambda cfg, device: env)
    config = [str(tmp_path / "study2.yaml"), *SMALL, *APPENDABLE, "teacher.chunk_size=12"]
    out = tmp_path / "pairs.json"
    a = f"ckpt:{end}:{file_sha256(end)}"
    b = f"ckpt:{one}:{file_sha256(one)}"
    argv = [*config, *OVERRIDES, "--pair", a, b, "--pair", a, f"vector:{vec}", "--r", "--out",
            str(out), "--surrogate", str(spec_path), "--panel-size", "3"]  # fmt: skip
    assert module.main(argv) == 0
    record = json.loads(out.read_text())
    first, second = record["pairs"]
    flat = lambda v: torch.cat([v[n].reshape(-1).double() for n in names])  # noqa: E731
    x = flat({n: e[n].double() - theta0[n].double() for n in names})
    y = flat({n: s1[n].double() - theta0[n].double() for n in names})
    assert first["dot"] == pytest.approx(float(x @ y), rel=1e-9)
    assert first["distance"] == pytest.approx(float((x - y).norm()), rel=1e-9)
    assert second["cosine"] == pytest.approx(1.0, abs=1e-6)
    assert second["norm_b"] == pytest.approx(2 * second["norm_a"], rel=1e-6)
    # r at the endpoint is the endpoint model's
    spec = surrogate.load_spec(spec_path)
    with torch.no_grad():
        for n, p in trainable:
            p.copy_(e[n])
    texts = [item["text"] for item in env.prompt_set("harmful_kl")]
    r_end = surrogate.surrogate_value(model.eval(), env.tokenizer, texts, spec.refusals,
                                      spec.compliances, device=env.device,
                                      micro_batch_size=cfg.execution.eval_batch_size,
                                      max_tokens=spec.max_tokens)  # fmt: skip
    assert record["r"]["f.pt"] == r_end and math.isfinite(record["r"]["theta0"])
    assert set(record["r"]) == {"theta0", "f.pt", "f-step-1.pt"}
    printed = capsys.readouterr().out
    for text in texts + list(spec.continuations):
        assert text not in printed
    assert module.main(argv) == 1  # never overwritten
    run_dir = tmp_path / "runs" / "smoke" / "f"
    by_run = tmp_path / "pairs-run.json"
    assert module.main([*config, *OVERRIDES, "--pair", f"run:{run_dir}:3", f"run:{run_dir}:1",
                        "--out", str(by_run)]) == 0  # fmt: skip
    assert json.loads(by_run.read_text())["pairs"][0]["dot"] == first["dot"]
    assert module.main([*config, *OVERRIDES, "--pair", f"run:{run_dir}:2", b, "--out",
                        str(tmp_path / "none.json")]) == 1  # no checkpoint at step 2  # fmt: skip
    bad = tmp_path / "bad.json"
    assert module.main([*config, *OVERRIDES, "--pair", f"ckpt:{end}:{'0' * 64}", b, "--out",
                        str(bad)]) == 1  # fmt: skip
    assert module.main([*config, *OVERRIDES, "--pair", "file.pt", b, "--out", str(bad)]) == 1
    assert not bad.exists()
