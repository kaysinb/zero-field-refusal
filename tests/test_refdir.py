"""Study 5's refusal direction (zero_field_refusal.refdir) on a tiny random Qwen2 model on CPU: the
hooks, the equal-length scoring against the training path, the selection rule, the answer
positions and the answer means."""

from __future__ import annotations

import math

import pytest
import torch
from conftest import tiny_model, tiny_tokenizer

from zero_field_refusal import refdir, surrogate
from zero_field_refusal.models import encode_chat
from zero_field_refusal.teacher import Sample

SPEC = surrogate.SurrogateSpec(
    version=1,
    harmful_panel="h",
    benign_panel="b",
    max_tokens=8,
    refusals=("w1 w2", "w3"),
    compliances=("w4 w5 w6", "w7", "w8 w9"),
    rank_share=0.9,
    sha256="x",
)
TEXTS = ["w10 w11", "w12 w13 w14", "w15", "w16 w17 w18 w19", "w20 w21"]
CPU = torch.device("cpu")


def test_equal_length_batches_cover_every_row_once_with_equal_lengths():
    lengths = [3, 5, 3, 7, 5, 3, 3, 3]
    batches = list(refdir.equal_length_batches(lengths, size=2))
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(len(lengths)))
    assert all(len({lengths[i] for i in b}) == 1 and len(b) <= 2 for b in batches)


def test_the_post_instruction_suffix_is_the_template_tail():
    tok = tiny_tokenizer()
    ids = [encode_chat(tok, t) for t in TEXTS]
    suffix = refdir.common_suffix(ids)
    assert tok.convert_ids_to_tokens(suffix) == ["<|im_end|>", "<|im_start|>", "assistant"]


def test_the_post_instruction_ids_come_from_the_template():
    tok = tiny_tokenizer()
    same_end = [encode_chat(tok, t + " w30") for t in TEXTS]  # every prompt ends with w30
    assert refdir.common_suffix(same_end)[0] == tok.convert_tokens_to_ids("w30")
    tail = refdir.post_instruction_ids(tok, same_end)
    assert tok.convert_ids_to_tokens(tail) == ["<|im_end|>", "<|im_start|>", "assistant"]
    with pytest.raises(ValueError, match="does not end"):
        refdir.post_instruction_ids(tok, [*same_end, same_end[0][:-1]])


def test_ablation_leaves_no_component_in_any_block_input():
    model, tok = tiny_model(), tiny_tokenizer()
    ids = torch.tensor([encode_chat(tok, TEXTS[1])])
    unit = torch.randn(model.config.hidden_size)
    unit = unit / unit.norm()
    layers = [1]
    with refdir.ablate(model, unit), refdir.capture_inputs(model, layers) as store:
        model(input_ids=ids)
    assert torch.allclose(store[1] @ unit, torch.zeros(()), atol=1e-5)
    with refdir.capture_inputs(model, layers) as plain:
        model(input_ids=ids)
    assert not torch.allclose(plain[1] @ unit, torch.zeros(()), atol=1e-3)


def test_addition_shifts_the_block_input_by_the_vector():
    model, tok = tiny_model(), tiny_tokenizer()
    ids = torch.tensor([encode_chat(tok, TEXTS[0])])
    v = torch.randn(model.config.hidden_size)
    with refdir.capture_inputs(model, [1]) as plain:
        model(input_ids=ids)
    with refdir.add(model, 1, v), refdir.capture_inputs(model, [1]) as added:
        model(input_ids=ids)
    assert torch.allclose(added[1] - plain[1], v.expand_as(plain[1]), atol=1e-5)
    with refdir.capture_inputs(model, [1]) as again:
        model(input_ids=ids)
    assert torch.equal(again[1], plain[1])  # the hooks are removed


def test_equal_length_scores_match_the_training_path():
    model, tok = tiny_model(), tiny_tokenizer()
    ours = refdir.refusal_scores(model, tok, TEXTS, SPEC, CPU)
    theirs = surrogate.prompt_values(
        model, tok, TEXTS, SPEC.refusals, SPEC.compliances, device=CPU, max_tokens=SPEC.max_tokens
    )
    assert ours == pytest.approx(theirs, abs=1e-5)


def test_the_kl_is_zero_without_intervention_and_positive_with_it():
    model, tok = tiny_model(), tiny_tokenizer()
    ids = [encode_chat(tok, t) for t in TEXTS]
    base = refdir.last_logprobs(model, ids, CPU)
    assert refdir.kl_mean(base, base) == pytest.approx(0.0, abs=1e-12)
    unit = torch.randn(model.config.hidden_size)
    with refdir.ablate(model, unit / unit.norm()):
        other = refdir.last_logprobs(model, ids, CPU)
    assert refdir.kl_mean(base, other) > 0


def test_candidates_exclude_zero_and_non_finite_vectors():
    h = torch.zeros(2, 3, 4, dtype=torch.float64)
    s = torch.zeros(2, 3, 4, dtype=torch.float64)
    h[0, 0, 0] = 1.0
    h[1, 2, 1] = float("nan")
    kept, excluded = refdir.candidates(h, s, [1, 2])
    assert list(kept) == [(1, 0)]
    assert {(e["layer"], e["position"]) for e in excluded} == {
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    }


@pytest.mark.parametrize(
    ("rows", "chosen"),
    [
        (
            [
                {"layer": 3, "position": 1, "bypass": -1.0, "induce": 0.5, "kl": 0.05},
                {"layer": 2, "position": 0, "bypass": -1.0, "induce": 0.6, "kl": 0.02},
                {"layer": 1, "position": 0, "bypass": -5.0, "induce": 0.5, "kl": 0.2},
            ],
            (2, 0),
        ),
        (
            [{"layer": 1, "position": 0, "bypass": 3.0, "induce": 0.5, "kl": 0.01}],
            None,
        ),  # no bypass
        (
            [{"layer": 1, "position": 0, "bypass": -1.0, "induce": -0.1, "kl": 0.01}],
            None,
        ),  # induce <= 0
        (
            [{"layer": 1, "position": 0, "bypass": -1.0, "induce": -0.6, "kl": 0.01}],
            None,
        ),  # below base
    ],
)
def test_the_selection_rule(rows, chosen):
    found = refdir.choose(rows, base_harmful=2.0, base_harmless=-0.5)
    assert (None if found is None else (found["layer"], found["position"])) == chosen


def test_answer_positions_exclude_the_end_token():
    ended = Sample(
        prompt_id="a", prompt_ids=(1, 2, 3), response_ids=(7, 8, 5), logprobs=(), ended=True
    )
    cut = Sample(prompt_id="a", prompt_ids=(1, 2), response_ids=(7, 8, 9), logprobs=(), ended=False)
    assert refdir.answer_positions(ended) == [3, 4]
    assert refdir.answer_positions(cut) == [2, 3, 4]


def test_answer_means_are_the_block_input_means_over_the_answer():
    model = tiny_model()
    samples = [
        Sample(
            prompt_id="a", prompt_ids=(9, 10, 11), response_ids=(12, 13, 5), logprobs=(), ended=True
        ),
        Sample(
            prompt_id="b",
            prompt_ids=(9, 14),
            response_ids=(15, 16, 17, 18),
            logprobs=(),
            ended=False,
        ),
    ]
    means, counts = refdir.answer_means(model, samples, [1], CPU)
    assert counts == [2, 4]
    for k, s in enumerate(samples):
        ids = torch.tensor([list(s.prompt_ids) + list(s.response_ids)])
        with refdir.capture_inputs(model, [1]) as store:
            model(input_ids=ids)
        want = store[1][0, refdir.answer_positions(s), :].double().mean(dim=0).float()
        assert torch.allclose(means[k, 0], want, atol=1e-6)
    assert math.isfinite(float(means.sum()))
