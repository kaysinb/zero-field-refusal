"""scripts/study5_refdir.py on CPU with a tiny random four-block Qwen2 model, synthetic prompt sets
and synthetic streams in place of θ₀, the data and the archived streams: the prompt split by the
registered rule; ``direction --bench`` writing only its timing; ``direction`` choosing by the rule,
writing the vectors and refusing to overwrite; ``project`` writing each seed's answers with the
projections on both directions and keeping a seed already written; and an unvalidated direction
projecting nothing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import tiny_tokenizer
from omegaconf import OmegaConf
from transformers import Qwen2Config, Qwen2ForCausalLM

from zero_field_refusal import refdir, surrogate
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.seeding import derive_seed
from zero_field_refusal.teacher import Sample


def script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = script_module(REPO_ROOT / "scripts" / "study5_refdir.py", "study5_refdir_script")
SPEC = surrogate.SurrogateSpec(
    version=1,
    harmful_panel="h",
    benign_panel="b",
    max_tokens=8,
    refusals=("w1 w2", "w3"),
    compliances=("w4 w5 w6", "w7", "w8 w9"),
    rank_share=0.9,
    sha256="spec",
)
SEEDS = (11, 12, 13)
MODEL_CFG = {"name": "tiny", "revision": "r0"}


def four_blocks() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).to(torch.float32).eval()
    torch.nn.init.normal_(model.model.embed_tokens.weight, std=0.5)
    return model


def prompt_rows(name: str) -> list[dict]:
    rng = np.random.default_rng(len(name))
    return [
        {"id": f"{name}-{i}", "text": " ".join(f"w{int(k)}" for k in rng.integers(10, 50, 3))}
        for i in range(100)
    ]


class Env:
    def __init__(self):
        self.tokenizer = tiny_tokenizer()
        self.device = torch.device("cpu")

    def prompt_set(self, name):
        return prompt_rows(name)


class Prefix:
    """A synthetic stream: 500 answers of 1-6 tokens to prompts drawn from 20 ids."""

    def __init__(self, cfg, env, seed):
        self.seed = seed

    def take(self, start, stop):
        rng = np.random.default_rng(self.seed)
        out = []
        for _ in range(stop):
            k = int(rng.integers(0, 20))
            response = tuple(int(t) for t in rng.integers(12, 60, int(rng.integers(1, 7))))
            ended = bool(rng.integers(0, 2))
            out.append(
                Sample(
                    prompt_id=f"x{k}",
                    prompt_ids=(2, 9, 12 + k, 3, 2, 6),
                    response_ids=response + ((3,) if ended else ()),
                    logprobs=(),
                    ended=ended,
                )
            )
        return out[start:stop]

    def identity(self, responses):
        return {"seed": self.seed, "count": responses}


@pytest.fixture
def patched(monkeypatch):
    model = four_blocks()
    monkeypatch.setattr(script, "MODELS", {"qwen": (("cfg",), SEEDS), "llama": (("cfg",), SEEDS)})
    monkeypatch.setattr(
        script, "run_config", lambda m, s: OmegaConf.create({"model": MODEL_CFG, "seed": s})
    )
    monkeypatch.setattr(
        script, "setup", lambda m, s: (script.run_config(m, s), {"tf32": False}, Env(), model, 0.0)
    )
    monkeypatch.setattr(script, "StreamPrefix", Prefix)
    monkeypatch.setattr(script.surrogate, "load_spec", lambda *a, **k: SPEC)
    monkeypatch.setattr(script, "git_state", lambda: {"commit": "test", "dirty": False})
    return model


def prompts_record(tmp: Path) -> Path:
    record = script.split_record({name: prompt_rows(name) for name in script.SETS})
    path = tmp / "prompts.json"
    path.write_text(json.dumps(record))
    return path


def test_the_split_is_the_registered_permutation():
    record = script.split_record({name: prompt_rows(name) for name in script.SETS})
    perm = np.random.default_rng(derive_seed(0, "study5", "refusal-direction", "split"))
    perm = perm.permutation(100).tolist()
    assert record["permutation"] == perm
    parts = record["sets"]["harmful_kl"]["parts"]
    assert [len(parts[p]) for p in ("direction", "selection", "confirmation")] == [60, 20, 20]
    assert parts["selection"][0] == f"harmful_kl-{perm[60]}"


def test_the_committed_prompt_record_follows_the_rule():
    record = json.loads((REPO_ROOT / "docs" / "study5-refdir-prompts.json").read_text())
    perm = np.random.default_rng(derive_seed(0, "study5", "refusal-direction", "split"))
    assert record["permutation"] == perm.permutation(100).tolist()
    for name in script.SETS:
        parts = record["sets"][name]["parts"]
        ids = parts["direction"] + parts["selection"] + parts["confirmation"]
        assert sorted(ids) == sorted(record["sets"][name]["text_sha256"]) and len(set(ids)) == 100


def run(*argv: str) -> int:
    return script.main([str(a) for a in argv])


def test_direction_project_and_the_bench(patched, tmp_path, capsys):
    prompts = prompts_record(tmp_path)
    bench = tmp_path / "bench"
    assert run("direction", "--model", "qwen", "--prompts", prompts, "--out", bench, "--bench") == 0
    b = json.loads((bench / "bench.json").read_text())
    assert b["layers"] == [1, 2, 3] and b["control_layer"] == 1 and b["estimate_seconds"] > 0
    assert b["estimate_seconds"] == pytest.approx(sum(b["estimate_parts"].values()))
    assert sorted(p.name for p in bench.iterdir()) == ["bench.json"]

    out = tmp_path / "direction"
    rc = run("direction", "--model", "qwen", "--prompts", prompts, "--out", out)
    found = json.loads((out / "direction.json").read_text())
    rows = found["rows"]
    want = refdir.choose(rows, found["base_harmful"], found["base_harmless"])
    if want is None:
        assert rc == 0 and found["chosen"] is None and not (out / "vectors.pt").exists()
        pytest.skip("the tiny model validated no candidate")
    assert rc == 0 and found["chosen"] == {"layer": want["layer"], "position": want["position"]}
    assert found["positions"] == 3 and len(rows) == found["candidates"]
    assert found["model_revision"] == "r0"
    vec = torch.load(out / "vectors.pt")
    assert (vec["layer"], vec["position"], vec["control_layer"]) == (
        want["layer"], want["position"], 1,
    )  # fmt: skip
    assert run("direction", "--model", "qwen", "--prompts", prompts, "--out", out) == 1

    proj = tmp_path / "proj"
    assert run("project", "--model", "qwen", "--direction", out, "--out", proj) == 0
    z = np.load(proj / "s11.npz")
    assert z["means"].shape == (500, 2, 32) and z["counts"].min() >= 1
    unit = (vec["chosen"] / vec["chosen"].norm()).numpy()
    assert np.allclose(z["p"], z["means"][:, 0, :].astype(np.float64) @ unit, atol=1e-5)
    assert np.allclose(z["p_tok"], z["p"] * z["counts"] / 96)
    meta = json.loads((proj / "s11.json").read_text())
    assert meta["stream"] == {"seed": 11, "count": 500}
    assert meta["binding"] == script.binding_of("qwen", out)
    assert meta["binding"]["layers"] == [want["layer"], 1] and meta["empty_answers"] == 0
    printed = capsys.readouterr().out
    assert "s11 answers 500" in printed
    assert run("project", "--model", "qwen", "--direction", out, "--out", proj) == 0
    assert "s11 exists: kept" in capsys.readouterr().out
    # a projection of another direction, or one without its receipt, is refused before any pass
    receipt = proj / "s12.json"
    other = json.loads(receipt.read_text())
    other["binding"]["layers"] = [3, 1]
    receipt.write_text(json.dumps(other))
    assert run("project", "--model", "qwen", "--direction", out, "--out", proj) == 1
    assert "not this direction's projection" in capsys.readouterr().out
    receipt.unlink()
    assert run("project", "--model", "qwen", "--direction", out, "--out", proj) == 1
    assert "no receipt" in capsys.readouterr().out
    # the direction of another model is refused
    assert run("project", "--model", "llama", "--direction", out, "--out", tmp_path / "q") == 1
    assert "not 'llama''s" in capsys.readouterr().out


def test_an_unvalidated_direction_projects_nothing(patched, tmp_path):
    folder = tmp_path / "d"
    folder.mkdir()
    (folder / "direction.json").write_text(json.dumps({"model": "llama", "chosen": None}))
    assert run("project", "--model", "llama", "--direction", folder, "--out", tmp_path / "p") == 1
    assert not (tmp_path / "p").exists()


def test_an_existing_output_folder_is_refused_before_any_model_loads(
    patched, tmp_path, monkeypatch
):
    prompts = prompts_record(tmp_path)
    (tmp_path / "taken").mkdir()
    monkeypatch.setattr(script, "setup", lambda m, s: pytest.fail("a model was loaded"))
    for extra in ([], ["--bench"]):
        argv = ["direction", "--model", "qwen", "--prompts", prompts, "--out", tmp_path / "taken"]
        assert run(*argv, *extra) == 1


def test_an_answer_of_only_its_end_token_has_no_mean(patched):
    prompt = (2, 9, 12, 3, 2, 6)
    ended = Sample(prompt_id="x", prompt_ids=prompt, response_ids=(3,), logprobs=(), ended=True)
    full = Sample(prompt_id="y", prompt_ids=prompt, response_ids=(20, 3), logprobs=(), ended=True)
    means, counts = refdir.answer_means(patched, [ended, full], [1, 2], torch.device("cpu"))
    assert counts == [0, 1]
    assert torch.isnan(means[0]).all() and torch.isfinite(means[1]).all()


BENCH = {"estimate_parts": {"direction_load": 60.0, "candidates": 1800.0, "projection": 900.0}}


@pytest.mark.parametrize(("elapsed", "admitted"), [(600.0, True), (3480.0, True), (3481.0, False)])
def test_the_admission_adds_the_elapsed_time_and_both_models_rest(elapsed, admitted):
    benches = [{"model": "qwen", **BENCH}, {"model": "llama", **BENCH}]
    found = script.admission(benches, started=1000.0, now=1000.0 + elapsed)
    assert found["total_seconds"] == pytest.approx(elapsed + 2 * 2760.0)
    assert found["admitted"] is admitted and found["limit_seconds"] == 9000.0
    with pytest.raises(ValueError, match="benches"):
        script.admission(benches[:1], started=0.0, now=1.0)


def test_the_estimate_counts_confirmation_and_projection():
    parts = script.estimate_parts(10.0, 5.0, 4.0, 2.0, 110, 0.01, 20000)
    assert parts == {
        "direction_load": 10.0, "direction_means": 5.0, "baselines": 4.0, "candidates": 220.0,
        "confirmation": 8.0, "projection_load": 10.0, "projection": 200.0,
    }  # fmt: skip


def test_the_admit_command_writes_its_record(tmp_path):
    import time

    paths = []
    for model in ("qwen", "llama"):
        path = tmp_path / f"{model}.json"
        path.write_text(json.dumps({"model": model, **BENCH}))
        paths += ["--bench", path]
    out = tmp_path / "admit.json"
    assert run("admit", *paths, "--started", time.time() - 60, "--out", out) == 0
    assert json.loads(out.read_text())["admitted"] is True
    late = tmp_path / "late.json"
    assert run("admit", *paths, "--started", time.time() - 9000, "--out", late) == 1
