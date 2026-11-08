"""Study 5's surrogate at the endpoint on CPU (docs/study5-preregistration.md, "Descriptive
measurements": ρ(r(θ₂₅), judged refusal share) over the fresh endpoints): the script on a tiny
fresh F run with its endpoint checkpoint; r equals ``surrogate_value`` of a model holding the
checkpoint's parameters, and a checkpoint that is not the run's is refused."""

from __future__ import annotations

import json

import pytest
import torch
from test_schedule import scheduled
from test_study3_campaigns import script_module
from test_study5_options import F_RUN, ready
from test_surrogate import SPEC_YAML
from test_train import APPENDABLE, SMALL
from test_train import env as env  # noqa: F401  (the fixture)

from zero_field_refusal import surrogate
from zero_field_refusal.models import freeze
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.schedule import run_scheduled
from zero_field_refusal.tensor_hash import tensors_sha256


def test_endpoint_script_scores_the_checkpoint(env, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    torch.set_num_threads(1)
    ckpt = tmp_path / "ckpt"
    overrides = [*F_RUN, "+study5.role=fresh"]
    cfg = ready(env, scheduled(tmp_path, *overrides, f"tracking.endpoint_checkpoint={ckpt}",
                               policy="short"))  # fmt: skip
    f_dir = run_scheduled(cfg, env, run_id="f-fresh")
    meta = json.loads((f_dir / "meta.json").read_text())
    recorded = meta["inputs"]["endpoint_checkpoint"]
    assert recorded["sha256"] and (ckpt / "f-fresh.pt").is_file()

    spec_path = tmp_path / "surrogate.yaml"
    spec_path.write_text(SPEC_YAML)
    spec = surrogate.load_spec(spec_path)
    module = script_module(REPO_ROOT / "scripts" / "study5_surrogate_endpoint.py", "s5_endpoint")
    monkeypatch.setattr(module, "environment", lambda cfg, device: env)
    config = [str(tmp_path / "study2.yaml"), *SMALL, *APPENDABLE, "teacher.chunk_size=12"]
    out = tmp_path / "records" / "endpoint.json"
    options = ["--f-run", str(f_dir), "--checkpoints", str(ckpt), "--out", str(out),
               "--surrogate", str(spec_path), "--panel-size", "3"]  # fmt: skip
    argv = [*config, *overrides, *options]
    with pytest.raises(ValueError, match="not 100"):
        module.main(argv[:-2])  # the registration's 100 prompts per panel
    assert not out.exists()
    assert (
        module.main([*config, *F_RUN, "+study5.role=control", "+train.order_seed=7", *options]) == 1
    )  # a fresh run only
    assert not out.exists()
    assert module.main(argv) == 0
    record = json.loads(out.read_text())

    # r and the per-prompt values are those of a model holding the checkpoint's parameters
    end = torch.load(ckpt / "f-fresh.pt", weights_only=True)
    model = env.new_model()
    trainable = freeze(model, cfg.train.freeze)
    theta0_sha = tensors_sha256(trainable)
    with torch.no_grad():
        for name, p in trainable:
            p.copy_(end[name])
    model.eval()
    harmful = [item["text"] for item in env.prompt_set("harmful_kl")]
    benign = [item["text"] for item in env.prompt_set("alpaca_benign_kl")]
    micro = int(cfg.execution.eval_batch_size)
    kw = {"device": env.device, "micro_batch_size": micro, "max_tokens": spec.max_tokens}
    r = surrogate.surrogate_value(model, env.tokenizer, harmful, spec.refusals,
                                  spec.compliances, **kw)  # fmt: skip
    values = surrogate.prompt_values(model, env.tokenizer, harmful, spec.refusals,
                                     spec.compliances, **kw)  # fmt: skip
    benign_values = surrogate.prompt_values(model, env.tokenizer, benign, spec.refusals,
                                            spec.compliances, **kw)  # fmt: skip
    assert record["r"] == r
    assert [p["difference"] for p in record["per_prompt"]["harmful"]] == values
    assert [p["id"] for p in record["per_prompt"]["harmful"]] == ["kl:0", "kl:1", "kl:2"]
    assert [p["difference"] for p in record["per_prompt"]["benign"]] == benign_values
    assert record["r_benign"] == sum(benign_values) / 3
    # θ₀'s r differs: the checkpoint's parameters were scored, not θ₀'s
    at_theta0 = surrogate.surrogate_value(env.new_model().eval(), env.tokenizer, harmful,
                                          spec.refusals, spec.compliances, **kw)  # fmt: skip
    assert record["r"] != at_theta0

    assert (record["kind"], record["point"], record["seed"]) == ("study5-surrogate-endpoint",
                                                                 "endpoint", 0)  # fmt: skip
    assert record["run_id"] == "f-fresh" and record["config_hash"] == meta["config_hash"]
    assert record["checkpoint"]["sha256"] == recorded["sha256"]
    assert record["checkpoint"]["step"] == 3
    assert record["theta0_trainable_sha256"] == theta0_sha
    assert record["theta0_trainable_sha256"] == meta["inputs"]["theta0_trainable_sha256"]
    assert record["endpoint_trainable_sha256"] == tensors_sha256(trainable)
    assert record["surrogate_sha256"] == spec.sha256
    assert record["panels"]["harmful"]["count"] == 3 and record["micro_batch_size"] == micro
    assert {"name", "revision", "dtype", "attn_implementation"} <= set(record["model"])
    assert "numerics" in record and "git" in record
    printed = capsys.readouterr().out
    assert record["checkpoint"]["sha256"] in printed and f"r {record['r']:.9e}" in printed
    for text in harmful + benign + list(spec.continuations):
        assert text not in printed
    assert module.main(argv) == 1  # never overwritten
    assert json.loads(out.read_text()) == record


def test_endpoint_script_refuses_a_foreign_checkpoint(env, tmp_path, monkeypatch) -> None:  # noqa: F811
    torch.set_num_threads(1)
    ckpt = tmp_path / "ckpt"
    overrides = [*F_RUN, "+study5.role=fresh"]
    cfg = ready(env, scheduled(tmp_path, *overrides, f"tracking.endpoint_checkpoint={ckpt}",
                               policy="short"))  # fmt: skip
    f_dir = run_scheduled(cfg, env, run_id="f-fresh")
    spec_path = tmp_path / "surrogate.yaml"
    spec_path.write_text(SPEC_YAML)
    module = script_module(REPO_ROOT / "scripts" / "study5_surrogate_endpoint.py", "s5_endpoint")
    monkeypatch.setattr(module, "environment", lambda cfg, device: env)
    config = [str(tmp_path / "study2.yaml"), *SMALL, *APPENDABLE, "teacher.chunk_size=12"]
    out = tmp_path / "records" / "endpoint.json"

    def argv(checkpoints):
        return [*config, *overrides, "--f-run", str(f_dir), "--checkpoints", str(checkpoints),
                "--out", str(out), "--surrogate", str(spec_path), "--panel-size", "3"]  # fmt: skip

    # another file under the run's checkpoint name: its sha256 is not the run's recorded one
    other = tmp_path / "other"
    other.mkdir()
    state = torch.load(ckpt / "f-fresh.pt", weights_only=True)
    torch.save({name: t + 1.0 for name, t in state.items()}, other / "f-fresh.pt")
    assert module.main(argv(other)) == 1
    assert not out.exists()
    # no checkpoint at all
    assert module.main(argv(tmp_path / "none")) == 1
    assert not out.exists()
    # a run that records another θ₀
    meta_path = f_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["inputs"]["theta0_trainable_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    assert module.main(argv(ckpt)) == 1
    assert not out.exists()
