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
        script, "run_config", lambda m, s: OmegaConf.create({"model": {"name": "tiny"}, "seed": s})
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


def test_direction_project_and_the_bench(patched, tmp_path, capsys):
    prompts = prompts_record(tmp_path)
    bench = tmp_path / "bench"
    assert (
        script.main(
            [
                "direction",
                "--model",
                "qwen",
                "--prompts",
                str(prompts),
                "--out",
                str(bench),
                "--bench",
            ]
        )
        == 0
    )
    b = json.loads((bench / "bench.json").read_text())
    assert b["layers"] == [1, 2, 3] and b["control_layer"] == 1 and b["estimate_seconds"] > 0
    assert sorted(p.name for p in bench.iterdir()) == ["bench.json"]

    out = tmp_path / "direction"
    rc = script.main(["direction", "--model", "qwen", "--prompts", str(prompts), "--out", str(out)])
    found = json.loads((out / "direction.json").read_text())
    rows = found["rows"]
    want = refdir.choose(rows, found["base_harmful"], found["base_harmless"])
    if want is None:
        assert rc == 0 and found["chosen"] is None and not (out / "vectors.pt").exists()
        pytest.skip("the tiny model validated no candidate")
    assert rc == 0 and found["chosen"] == {"layer": want["layer"], "position": want["position"]}
    assert found["positions"] == 3 and len(rows) == found["candidates"]
    vec = torch.load(out / "vectors.pt")
    assert (vec["layer"], vec["position"], vec["control_layer"]) == (
        want["layer"],
        want["position"],
        1,
    )
    assert (
        script.main(["direction", "--model", "qwen", "--prompts", str(prompts), "--out", str(out)])
        == 1
    )

    proj = tmp_path / "proj"
    assert (
        script.main(["project", "--model", "qwen", "--direction", str(out), "--out", str(proj)])
        == 0
    )
    z = np.load(proj / "s11.npz")
    assert z["means"].shape == (500, 2, 32) and z["counts"].min() >= 1
    unit = (vec["chosen"] / vec["chosen"].norm()).numpy()
    assert np.allclose(z["p"], z["means"][:, 0, :].astype(np.float64) @ unit, atol=1e-5)
    assert np.allclose(z["p_tok"], z["p"] * z["counts"] / 96)
    meta = json.loads((proj / "s11.json").read_text())
    assert meta["stream"] == {"seed": 11, "count": 500}
    printed = capsys.readouterr().out
    assert "s11 answers 500" in printed
    assert (
        script.main(["project", "--model", "qwen", "--direction", str(out), "--out", str(proj)])
        == 0
    )
    assert "s11 exists: kept" in capsys.readouterr().out


def test_an_unvalidated_direction_projects_nothing(patched, tmp_path):
    folder = tmp_path / "d"
    folder.mkdir()
    (folder / "direction.json").write_text(json.dumps({"chosen": None}))
    assert (
        script.main(
            [
                "project",
                "--model",
                "llama",
                "--direction",
                str(folder),
                "--out",
                str(tmp_path / "p"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "p").exists()
