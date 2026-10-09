import pytest
import torch
from conftest import END_IDS, PAD_ID, prompts, tiny_model

from zero_field_refusal.batches import collate
from zero_field_refusal.evaluation import (
    choice_predictions,
    displacement,
    greedy_responses,
    random_direction,
    sequence_kl,
    sequence_nll,
    set_point,
)
from zero_field_refusal.losses import response_logits, summed_nll
from zero_field_refusal.models import freeze, install_neutral_generation_config
from zero_field_refusal.refusal import is_keyword_refusal
from zero_field_refusal.teacher import sample_responses

SAMPLING = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=30)


@pytest.fixture(scope="module")
def setup():
    reference = tiny_model(tied=False, head_std=0.02)
    install_neutral_generation_config(reference, END_IDS, PAD_ID)
    samples, _ = sample_responses(
        reference, prompts(7), [f"p{i}" for i in range(7)], sampling=SAMPLING, end_ids=END_IDS,
        pad_id=PAD_ID, seed=1, batch_size=3,
    )  # fmt: skip
    return reference, samples


def perturbed(reference, scale=0.05, seed=0):
    student = tiny_model(tied=False, head_std=0.02)
    student.load_state_dict(reference.state_dict())
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)
    install_neutral_generation_config(student, END_IDS, PAD_ID)
    return student


def test_k_is_exactly_zero_at_theta0(setup):
    reference, samples = setup
    same = perturbed(reference, scale=0.0)
    kl = sequence_kl(same, reference, samples, pad_id=PAD_ID, micro_batch_size=2)
    assert kl.dtype == torch.float64 and kl.shape == (7,)
    assert torch.all(kl == 0)


def test_k_is_the_exact_per_position_kl_summed_over_the_response(setup):
    reference, samples = setup
    student = perturbed(reference)
    kl = sequence_kl(student, reference, samples, pad_id=PAD_ID, micro_batch_size=3)
    assert torch.all(kl > 0)
    for i in (0, 4):
        batch = collate([samples[i]], pad_id=PAD_ID, max_response_tokens=10**6)
        with torch.no_grad():
            p0 = torch.log_softmax(response_logits(reference, batch).double(), -1)
            p1 = torch.log_softmax(response_logits(student, batch).double(), -1)
        expected = (p0.exp() * (p0 - p1)).sum()
        assert kl[i].item() == pytest.approx(expected.item(), rel=1e-6)  # batch shapes differ


def test_k_does_not_depend_on_the_micro_batch(setup):
    reference, samples = setup
    student = perturbed(reference)
    one = sequence_kl(student, reference, samples, pad_id=PAD_ID, micro_batch_size=1)
    many = sequence_kl(student, reference, samples, pad_id=PAD_ID, micro_batch_size=7)
    torch.testing.assert_close(one, many, rtol=1e-6, atol=1e-9)


def test_sequence_nll_matches_the_training_loss(setup):
    reference, samples = setup
    nll, tokens = sequence_nll(reference, samples, pad_id=PAD_ID, micro_batch_size=2)
    batch = collate(samples, pad_id=PAD_ID, max_response_tokens=10**6)
    with torch.no_grad():
        total = summed_nll(reference, batch).item()
    assert nll.sum().item() == pytest.approx(total, rel=1e-6)
    assert tokens.tolist() == [len(s.response_ids) for s in samples]


def test_greedy_responses_follow_the_argmax_of_the_training_path(setup):
    reference, _ = setup
    student = perturbed(reference, scale=0.3)
    batch_prompts = prompts(5, seed=2)
    out = greedy_responses(student, batch_prompts, end_ids=END_IDS, pad_id=PAD_ID,
                           max_new_tokens=8, batch_size=2)  # fmt: skip
    assert out == greedy_responses(student, batch_prompts, end_ids=END_IDS, pad_id=PAD_ID,
                                   max_new_tokens=8, batch_size=5)  # fmt: skip
    for prompt, (response, ended) in zip(batch_prompts, out, strict=True):
        assert ended == (response[-1] in END_IDS)
        ids = torch.tensor([prompt + response])
        with torch.no_grad():
            logits = student(input_ids=ids).logits[0, len(prompt) - 1 : -1]
        assert logits.argmax(-1).tolist() == response


def test_choice_predictions_read_the_last_prompt_position(setup):
    reference, _ = setup
    student = perturbed(reference, scale=0.3)
    batch_prompts = prompts(6, seed=3)
    choices = [11, 12, 13, 14]
    predicted = choice_predictions(student, batch_prompts, choices, pad_id=PAD_ID, batch_size=4)
    for prompt, choice in zip(batch_prompts, predicted, strict=True):
        with torch.no_grad():
            last = student(input_ids=torch.tensor([prompt])).logits[0, -1]
        assert choice == int(last[choices].argmax())


def named(model):
    return freeze(model, ["input_embeddings", "output_head"])


def test_displacement_reports_norms_of_trainable_parameters(setup):
    reference, _ = setup
    student = perturbed(reference, scale=0.0)
    first = next(p for _, p in named(student))
    with torch.no_grad():
        first.add_(3.0)
    geometry = displacement(named(student), dict(reference.named_parameters()))
    assert geometry["delta_norm"] == pytest.approx(3.0 * first.numel() ** 0.5)
    assert geometry["relative"] == pytest.approx(geometry["delta_norm"] / geometry["theta0_norm"])
    assert len(geometry["per_tensor"]) == len(named(student))


def test_set_point_moves_along_the_whole_displacement(setup):
    reference, _ = setup
    student = perturbed(reference)
    params = named(student)
    theta0 = dict(reference.named_parameters())
    start = [theta0[name].detach().clone() for name, _ in params]
    delta = [p.detach() - s for (_, p), s in zip(params, start, strict=True)]
    for a in (0.0, 0.5, -1.0):
        set_point(params, start, delta, a)
        for (_, p), s, d in zip(params, start, delta, strict=True):
            torch.testing.assert_close(p.detach(), s + a * d)
    set_point(params, start, delta, 0.0)
    assert all(torch.equal(p, s) for (_, p), s in zip(params, start, strict=True))


@pytest.mark.parametrize("kind", ["global", "per_tensor"])
def test_random_directions_match_the_displacement_norms(kind):
    shapes = {"a.weight": (4, 3), "b.weight": (5,)}
    norms = {"a.weight": 2.0, "b.weight": 0.5}
    r = random_direction(shapes, norms, kind=kind, seed=7)
    # Directions derive from names and full shapes, not from the order of parameters.
    reordered = random_direction(dict(reversed(shapes.items())), norms, kind=kind, seed=7)
    assert all(torch.equal(r[name], reordered[name]) for name in shapes)
    if kind == "global":
        total = torch.sqrt(sum(v.pow(2).sum() for v in r.values()))
        assert total.item() == pytest.approx((2.0**2 + 0.5**2) ** 0.5, rel=1e-6)
    else:
        for name, value in r.items():
            assert value.norm().item() == pytest.approx(norms[name], rel=1e-6)
    other = random_direction(shapes, norms, kind=kind, seed=8)
    assert not torch.equal(other["a.weight"], r["a.weight"])


def test_keyword_refusal_v1() -> None:
    assert is_keyword_refusal("I’m sorry, but I can’t help with that.")
    assert not is_keyword_refusal("Sure, here is a recipe for pancakes.")
