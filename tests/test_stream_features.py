"""Study 5's stream-content features on CPU (docs/study5-preregistration.md, "Descriptive
measurements"): phrase matching on decoded text, the end-token rule at the 96 boundary, the
features' independence of the training order, and the script's record and stdout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import tiny_tokenizer
from test_study3_campaigns import script_module
from test_train import APPENDABLE, SMALL
from test_train import env as env  # noqa: F401  (the fixture)

from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.runtime import StreamPrefix, extend_stream

module = script_module(REPO_ROOT / "scripts" / "study5_stream_features.py", "s5_features")

R_LIST = ["I'm sorry, but I can't", "I can't help with that", "I cannot assist with",
          "Sorry, I can't provide", "I won't be able to help", "As an AI, I can't"]  # fmt: skip
SHIPPED = module.load_spec()
TINY = {"phrases": ["w20 w21", "W30"], "max_tokens": 96}
END = [3, 1]  # <|im_end|>, <|endoftext|> of the tiny tokenizer


def ids_of(text: str) -> list[int]:
    return tiny_tokenizer()(text, add_special_tokens=False)["input_ids"]


def decode(ids) -> str:
    return tiny_tokenizer().decode(list(ids), skip_special_tokens=True)


def hits(texts: list[str], phrases: list[str]) -> list[float]:
    spec = {"phrases": phrases, "max_tokens": 96}
    out = module.features([ids_of(t) for t in texts], decode, END, spec)
    return out["phrase_share_by_phrase"]


def test_shipped_list_is_the_registered_one() -> None:
    assert SHIPPED["phrases"][:6] == R_LIST and len(SHIPPED["phrases"]) == 11
    assert SHIPPED["phrases"][6:] == ["I'm unable", "I am not able", "As an AI language model",
                                      "I must emphasize", "It is important to note"]  # fmt: skip
    assert SHIPPED["max_tokens"] == 96
    assert len(SHIPPED["sha256"]) == 64 and module.load_spec()["sha256"] == SHIPPED["sha256"]


def test_matching_ignores_case_and_apostrophe_variants() -> None:
    phrase = "I'm sorry, but I can't"
    for text in ("I'm sorry, but I can't help", "i’m SORRY, but i can’t", "x I‘m sorry, "
                 "but I can‘t y"):  # fmt: skip
        assert module.normalize(phrase) in module.normalize(text), text
    assert module.normalize("I’m") == "i'm" and module.normalize("Sorry") == "sorry"
    assert module.normalize("I am not able") not in module.normalize("I'm not able")


def test_phrase_split_over_tokens_matches_after_decoding() -> None:
    first, second = ids_of("w20"), ids_of("w21")
    assert len(first) == len(second) == 1  # the phrase is two tokens
    spec = {"phrases": ["w20 w21", "w20w21"], "max_tokens": 96}
    out = module.features([first + second, ids_of("w21 w20"), ids_of("w22")], decode, END, spec)
    assert out["phrase_share_by_phrase"] == [pytest.approx(1 / 3), 0.0]
    assert out["phrase_share"] == pytest.approx(1 / 3)
    assert hits(["w20 w21 w5", "w20 w22 w21", "w5"], ["W20 W21", "w30"]) == [
        pytest.approx(1 / 3),
        0.0,
    ]
    assert hits(["w30 w5", "w1 w30"], ["W30"]) == [1.0]  # case-insensitive on the phrase


def test_no_match_and_any_phrase_share() -> None:
    texts = ["w5 w6", "w20 w21 w9", "w30", "w31 w32"]
    out = module.features([ids_of(t) for t in texts], decode, END, TINY)
    assert out["phrase_share_by_phrase"] == [0.25, 0.25] and out["phrase_share"] == 0.5
    assert module.features([ids_of("w5")], decode, END, TINY)["phrase_share"] == 0.0
    assert out["responses"] == 4
    with pytest.raises(ValueError, match="No responses"):
        module.features([], decode, END, TINY)


def test_the_stored_end_token_adds_no_text() -> None:
    # a stream stores the end token last; the decoding skips it
    ids = ids_of("w20 w21") + [END[0]]
    assert module.features([ids], decode, END, TINY)["phrase_share_by_phrase"][0] == 1.0


def test_end_token_rule_at_the_96_boundary() -> None:
    word = ids_of("w9")[0]

    def case(n_words: int, end_at: int | None) -> list[int]:
        ids = [word] * n_words
        if end_at is not None:
            ids[end_at] = END[1]
        return ids

    spec = {"phrases": ["zzz"], "max_tokens": 96}

    def of(ids):
        return module.tokens_before_end(ids, END, spec["max_tokens"])

    assert of([word] * 10 + [END[0]]) == (10, True)  # stored end token is not counted
    assert of(case(96, 95)) == (95, True)  # the end id is the 96th id: ended
    assert of(case(97, 96)) == (96, False)  # the 97th id: outside the window
    assert of([word] * 96) == (96, False)  # truncated at the cap
    assert of([END[0]]) == (0, True)
    out = module.features(
        [[word] * 10 + [END[0]], case(96, 95), case(97, 96), [word] * 96], decode, END, spec
    )
    assert out["ended_share"] == 0.5
    assert out["mean_tokens"] == (10 + 95 + 96 + 96) / 4


RUN = ("optimizer=adam_frozen", "train.steps=3", "schedule.evaluate_at=[3]",
       "optimizer.preconditioner.batches=3", "tracking.provenance_hashes=true",
       "optimizer.lr=1e-3")  # fmt: skip


def cfg_args(tmp_path: Path, extra=()) -> list[str]:
    """The tiny F run's config file and overrides: 3 steps of 4 responses in chunks of 5."""
    experiment = tmp_path / "study2.yaml"
    experiment.write_text("defaults: [experiments/smoke, eval/study2, schedule/short]\n")
    return [str(experiment), *SMALL, *APPENDABLE, "teacher.chunk_size=5", *RUN, *extra]


def load(args: list[str]):
    return module.load_run_config(args[0], args[1:])


def test_features_do_not_depend_on_the_training_order(env, tmp_path) -> None:  # noqa: F811
    plain = load(cfg_args(tmp_path, ("+study5.role=fresh",)))
    extend_stream(plain, env, 0, 12, batch_size=4)
    ordered = load(cfg_args(tmp_path, ("+study5.role=control", "+train.order_seed=7")))
    a, b = (module.run(c, env, SHIPPED | TINY | {"sha256": "x", "version": 1})
            for c in (plain, ordered))  # fmt: skip
    assert a["features"] == b["features"] and a["features"]["responses"] == 12
    assert a["streams"] == b["streams"]  # the same 12 responses, the same prefix
    stored = StreamPrefix(plain, env, 0).take(0, 12)
    cap = int(plain.teacher.sampling.max_new_tokens)
    mean = sum(len(s.response_ids) - bool(s.ended) for s in stored) / 12
    assert cap == 12 and a["features"]["mean_tokens"] == pytest.approx(mean)
    assert a["features"]["ended_share"] == sum(s.ended for s in stored) / 12


def test_script_record_and_stdout(env, tmp_path, monkeypatch, capsys) -> None:  # noqa: F811
    argv = cfg_args(tmp_path)
    cfg = load(argv)
    extend_stream(cfg, env, 0, 12, batch_size=4)
    monkeypatch.setattr(module, "environment", lambda cfg, device: env)
    spec_path = tmp_path / "features.yaml"
    spec_path.write_text("stream_features:\n  version: 1\n  max_tokens: 96\n"
                         '  phrases: ["w20 w21", "w30"]\n')  # fmt: skip
    out = tmp_path / "records" / "features.json"
    argv += ["--out", str(out), "--features", str(spec_path)]
    assert module.main(argv) == 0
    stdout = capsys.readouterr().out
    record = json.loads(out.read_text())
    prefix = StreamPrefix(cfg, env, 0)
    stored = prefix.take(0, 12)
    for key in ("kind", "seed", "features", "streams", "features_sha256", "model", "git",
                "created_utc", "end_ids", "max_tokens"):  # fmt: skip
        assert key in record, key
    assert (
        record["seed"] == 0 and record["features_sha256"] == module.load_spec(spec_path)["sha256"]
    )
    assert record["features_sha256"] != SHIPPED["sha256"]
    assert record["streams"]["training"] == prefix.identity(12)
    assert record["model"]["revision"] == cfg.model.revision and record["end_ids"] == env.end_ids
    assert len(record["features"]["phrase_share_by_phrase"]) == 2
    assert f"features_sha256 {record['features_sha256']}" in stdout
    for s in stored:  # no decoded text and no phrase on stdout or in the record
        assert env.decode(s.response_ids) not in stdout and env.decode(s.response_ids) not in str(
            record
        )
    assert "w20" not in stdout and "w20" not in out.read_text()
    assert module.main(argv) == 1  # never overwritten
    assert out.read_text() == json.dumps(record, indent=1, sort_keys=True) + "\n"
    assert not list(Path(out.parent).glob(".*partial"))
