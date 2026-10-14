import dataclasses
import json
from pathlib import Path

import pytest
import torch
from conftest import END_IDS, PAD_ID, VOCAB, prompts, tiny_model
from transformers import LogitsProcessor, LogitsProcessorList

from zero_field_refusal.batches import collate
from zero_field_refusal.losses import response_logprobs
from zero_field_refusal.models import install_neutral_generation_config
from zero_field_refusal.teacher import (
    FINGERPRINTS_FILE,
    PROJECTIONS,
    Sample,
    StreamError,
    draw_prompt_indices,
    fingerprint,
    load_distributions,
    load_fingerprints,
    load_manifest,
    load_stream,
    partial_directory,
    publish,
    replay_distributions,
    sample_responses,
    split_response,
    stale_partials,
    write_distributions,
    write_fingerprints,
    write_stream,
)

ZERO_FIELD = dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, max_new_tokens=40)


@pytest.fixture
def teacher():
    # Nearly uniform next-token distributions: an end token appears within 40 steps in most rows.
    model = tiny_model(tied=False, head_std=0.02)
    install_neutral_generation_config(model, END_IDS, PAD_ID)
    return model


def run(teacher, n=10, **kwargs):
    batch = prompts(n)
    options = dict(sampling=ZERO_FIELD, end_ids=END_IDS, pad_id=PAD_ID, seed=3, batch_size=4)
    options.update(kwargs)
    return sample_responses(teacher, batch, [f"p{i}" for i in range(n)], **options)


def test_prompt_draws_are_uniform_with_replacement_and_deterministic() -> None:
    draws = draw_prompt_indices(5000, 100, seed=1)
    assert draws == draw_prompt_indices(5000, 100, seed=1)
    assert draws != draw_prompt_indices(5000, 100, seed=2)
    assert min(draws) == 0 and max(draws) == 99
    counts = torch.bincount(torch.tensor(draws), minlength=100).float()
    assert counts.std() < 15  # binomial SD ≈ 7: uniform
    assert len(set(draw_prompt_indices(100, 100, seed=1))) < 100  # repeats: with replacement


@pytest.mark.parametrize(
    ("generated", "response", "ended"),
    [
        ([9, 10, 5, 6, 6], [9, 10, 5], True),
        ([9, 6, 6, 6], [9, 6], True),  # the pad id is an end token and is kept once
        ([5, 6, 6], [5], True),  # an end token as the first token
        ([9, 10, 11], [9, 10, 11], False),  # truncated at the cap
    ],
)
def test_split_response_keeps_the_first_end_token(generated, response, ended) -> None:
    assert split_response(generated, END_IDS) == (response, ended)


def test_samples_carry_raw_log_probs_that_the_training_path_reproduces(teacher) -> None:
    samples, _ = run(teacher, n=10)
    assert len(samples) == 10
    assert any(s.ended for s in samples) and any(not s.ended for s in samples)
    for s in samples:
        assert 1 <= len(s.response_ids) <= ZERO_FIELD["max_new_tokens"]
        assert s.ended == (s.response_ids[-1] in END_IDS)
        assert all(t not in END_IDS for t in s.response_ids[:-1])
        assert len(s.logprobs) == len(s.response_ids)
    # Left-padded KV-cache generation against right-padded full-sequence scoring.
    batch = collate(samples, pad_id=PAD_ID, max_response_tokens=ZERO_FIELD["max_new_tokens"])
    with torch.no_grad():
        logp = response_logprobs(teacher, batch)
    targets = batch.input_ids[:, 1:][batch.target_mask]
    rescored = logp.gather(1, targets[:, None])[:, 0]
    stored = torch.tensor([lp for s in samples for lp in s.logprobs])
    torch.testing.assert_close(rescored, stored, rtol=0, atol=1e-5)


def test_sampling_is_reproducible_for_a_seed(teacher) -> None:
    first, _ = run(teacher, seed=3)
    again, _ = run(teacher, seed=3)
    other, _ = run(teacher, seed=4)
    assert [s.response_ids for s in first] == [s.response_ids for s in again]
    assert [s.response_ids for s in first] != [s.response_ids for s in other]


def test_zero_field_sampling_rejects_any_reshaped_distribution(teacher) -> None:
    with pytest.raises(StreamError, match="logits processor"):
        run(teacher, sampling={**ZERO_FIELD, "temperature": 0.7})
    samples, _ = run(teacher, sampling={**ZERO_FIELD, "temperature": 0.7}, zero_field=False)
    assert len(samples) == 10


def test_recorded_distributions_match_the_sampled_log_probs(teacher) -> None:
    samples, distributions = run(teacher, n=6, record_distributions=True)
    assert len(distributions) == len(samples)
    for s, logp in zip(samples, distributions, strict=True):
        assert logp.shape == (len(s.response_ids), VOCAB) and logp.dtype == torch.float32
        torch.testing.assert_close(logp.exp().sum(-1), torch.ones(len(s.response_ids)))
        picked = logp.gather(1, torch.tensor(s.response_ids)[:, None])[:, 0]
        assert torch.equal(picked, torch.tensor(s.logprobs))


def test_stream_round_trip_and_tamper_check(tmp_path: Path, teacher) -> None:
    samples, _ = run(teacher, n=5)
    directory = tmp_path / "seed-0"
    write_stream(directory, samples, {"kind": "training", "seed": 0})
    assert load_stream(directory) == samples
    manifest = load_manifest(directory)
    assert manifest["kind"] == "training" and manifest["count"] == 5
    assert manifest["ended_fraction"] == sum(s.ended for s in samples) / 5
    lines = (directory / "samples.jsonl").read_text().splitlines()
    (directory / "samples.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(StreamError, match="sha256"):
        load_stream(directory)


def test_streams_are_published_atomically(tmp_path: Path, teacher) -> None:
    samples, _ = run(teacher, n=3)
    final = tmp_path / "kind" / "seed-0"
    partial = partial_directory(final)
    write_stream(partial, samples, {"kind": "training"})
    assert stale_partials(final) == [partial]  # an interrupted generation is visible
    with pytest.raises(StreamError, match="Missing stream"):
        load_stream(final)
    publish(partial, final)
    assert load_stream(final) == samples and stale_partials(final) == []
    other = partial_directory(final)
    write_stream(other, samples, {"kind": "training"})
    with pytest.raises(StreamError, match="never regenerated"):
        publish(other, final)


def test_publish_syncs_the_staging_directory_before_the_rename(
    tmp_path: Path, teacher, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zero_field_refusal.teacher as module

    samples, _ = run(teacher, n=2)
    final = tmp_path / "kind" / "seed-0"
    partial = partial_directory(final)
    write_stream(partial, samples, {"kind": "training"})
    events = []
    sync, rename = module.fsync_directory, Path.rename
    monkeypatch.setattr(module, "fsync_directory", lambda p: (events.append(("fsync", p)), sync(p)))
    monkeypatch.setattr(Path, "rename", lambda p, to: (events.append(("rename", p)), rename(p, to)))
    publish(partial, final)
    assert events == [("fsync", partial), ("rename", partial), ("fsync", final.parent)]


def test_distributions_are_checked_against_the_stream(tmp_path: Path, teacher) -> None:
    samples, distributions = run(teacher, n=4, record_distributions=True)
    directory = tmp_path / "seed-0"
    directory.mkdir()
    record = write_distributions(directory, distributions)
    write_stream(directory, samples, {"distributions": record})
    loaded = load_distributions(directory)
    assert len(loaded) == len(distributions)
    assert all(torch.equal(a, b) for a, b in zip(loaded[:], distributions, strict=True))
    assert torch.equal(loaded[2], distributions[2])
    assert all(
        torch.equal(a, distributions[i]) for a, i in zip(loaded.take([3, 0]), [3, 0], strict=True)
    )
    write_stream(directory, samples[:3], {"distributions": record})  # lengths no longer match
    with pytest.raises(StreamError, match="lengths"):
        load_distributions(directory)


def test_the_buffer_collects_chunks_and_saves_only_the_filled_rows(tmp_path: Path, teacher) -> None:
    from zero_field_refusal.teacher import DistributionBuffer

    buffer = DistributionBuffer(max_tokens=10 * 40, vocab=VOCAB)
    samples, _ = run(teacher, n=6, on_chunk_distributions=buffer.append)
    _, recorded = run(teacher, n=6, record_distributions=True)
    assert buffer.lengths == [len(s.response_ids) for s in samples]
    directory = tmp_path / "seed-0"
    directory.mkdir()
    write_stream(directory, samples, {"distributions": buffer.save(directory)})
    loaded = load_distributions(directory)
    assert all(torch.equal(a, b) for a, b in zip(loaded[:], recorded, strict=True))


class InPlaceShift(LogitsProcessor):
    """Edits the scores in place, as a buggy or injected processor would."""

    def __call__(self, input_ids, scores):
        scores[:, 7] += 4.3
        return scores


def with_in_place_processor(model, monkeypatch) -> None:
    original = model.generate
    monkeypatch.setattr(model, "generate", lambda *args, **kwargs: original(
        *args, logits_processor=LogitsProcessorList([InPlaceShift()]), **kwargs))  # fmt: skip


def test_an_in_place_logits_processor_is_caught(teacher, monkeypatch) -> None:
    from zero_field_refusal.evaluation import greedy_responses

    with_in_place_processor(teacher, monkeypatch)
    with pytest.raises(StreamError, match="sampling: a logits processor"):
        run(teacher, n=2)
    with pytest.raises(StreamError, match="greedy decoding: a logits processor"):
        greedy_responses(teacher, prompts(2), end_ids=END_IDS, pad_id=PAD_ID, max_new_tokens=4,
                         batch_size=2)  # fmt: skip


def test_sample_json_round_trip_is_exact() -> None:
    s = Sample("p0", (1, 2), (3, 5), (-0.1234567, -2.5e-8), True)
    assert Sample.from_json(json.loads(json.dumps(s.to_json()))) == s


def test_prompts_of_very_different_lengths_with_end_ids_inside(teacher) -> None:
    # Left padding up to 14 positions; end ids and the pad id inside prompts are prompt tokens.
    batch = [[11], [12, 5, 13, 6, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], [6, 6, 30]]
    samples, _ = sample_responses(teacher, batch, ["a", "b", "c"], sampling=ZERO_FIELD,
                                  end_ids=END_IDS, pad_id=PAD_ID, seed=9, batch_size=3)  # fmt: skip
    assert [list(s.prompt_ids) for s in samples] == batch
    collated = collate(samples, pad_id=PAD_ID, max_response_tokens=ZERO_FIELD["max_new_tokens"])
    with torch.no_grad():
        logp = response_logprobs(teacher, collated)
    rescored = logp.gather(1, collated.input_ids[:, 1:][collated.target_mask][:, None])[:, 0]
    stored = torch.tensor([lp for s in samples for lp in s.logprobs])
    torch.testing.assert_close(rescored, stored, rtol=0, atol=1e-5)


def test_sample_records_are_validated(teacher) -> None:
    from zero_field_refusal.teacher import validate_samples

    samples, _ = run(teacher, n=4)
    validate_samples(samples, END_IDS, ZERO_FIELD["max_new_tokens"])
    good = Sample("p", (1,), (9, 5), (-1.0, -2.0), True)
    for bad in (
        Sample("p", (1,), (9, 5), (-1.0,), True),  # a missing log-prob
        Sample("p", (1,), (5, 9), (-1.0, -2.0), False),  # an end token inside
        Sample("p", (1,), (9, 10), (-1.0, -2.0), True),  # ended without an end token
        Sample("p", (1,), tuple(range(7, 50)), (0.0,) * 43, False),  # longer than the cap
    ):
        with pytest.raises(StreamError):
            validate_samples([good, bad], END_IDS, ZERO_FIELD["max_new_tokens"])


def test_a_replay_reproduces_the_recorded_distributions_exactly(teacher) -> None:
    # Replayed batch by batch as sampled (batches of 4), ended and truncated responses alike.
    samples, recorded = run(teacher, n=10, record_distributions=True)
    assert {s.ended for s in samples} == {True, False}
    for start in range(0, 10, 4):
        batch = samples[start : start + 4]
        replayed, errors, _ = replay_distributions(
            teacher, batch, sampling=ZERO_FIELD, end_ids=END_IDS, pad_id=PAD_ID
        )
        assert errors.numel() == sum(len(s.response_ids) for s in batch)
        assert errors.max().item() == 0.0
        for got, want in zip(replayed, recorded[start : start + 4], strict=True):
            assert torch.equal(got, want)


def test_a_replay_reports_stored_log_probs_that_differ(teacher) -> None:
    samples, _ = run(teacher, n=4)
    changed = dataclasses.replace(
        samples[1], logprobs=(samples[1].logprobs[0] + 1e-3, *samples[1].logprobs[1:])
    )
    batch = [samples[0], changed, *samples[2:]]
    _, errors, _ = replay_distributions(
        teacher, batch, sampling=ZERO_FIELD, end_ids=END_IDS, pad_id=PAD_ID
    )
    assert errors.max().item() == pytest.approx(1e-3, rel=1e-3)


def test_a_replay_fails_when_generation_would_not_end_where_the_record_does(teacher) -> None:
    samples, _ = run(teacher, n=4)
    long = next(s for s in samples if len(s.response_ids) > 3)
    cut = dataclasses.replace(long, response_ids=long.response_ids[:3],
                              logprobs=long.logprobs[:3], ended=False)  # fmt: skip
    with pytest.raises(StreamError, match="did not reproduce"):
        replay_distributions(teacher, [cut], sampling=ZERO_FIELD, end_ids=END_IDS, pad_id=PAD_ID)
    with pytest.raises(ValueError, match="zero_field=False"):
        run(teacher, n=2, force=[s.response_ids for s in samples[:2]])


def test_the_replay_reproduces_the_fingerprint_recorded_at_sampling(teacher) -> None:
    recorded: list = []
    samples, distributions = run(teacher, n=8, record_distributions=True, fingerprints=recorded)
    assert len(recorded) == 2  # one per generation batch of 4
    digest, projections = recorded[0]
    assert projections.shape == (sum(len(s.response_ids) for s in samples[:4]), PROJECTIONS)
    assert projections.dtype == torch.float64
    again = fingerprint(distributions[:4])  # from the recorded rows, on the CPU
    assert again[0] == digest and torch.equal(again[1], projections)
    _, _, (replayed_digest, replayed_projections) = replay_distributions(
        teacher, samples[:4], sampling=ZERO_FIELD, end_ids=END_IDS, pad_id=PAD_ID
    )
    assert replayed_digest == digest
    assert torch.equal(replayed_projections, projections)


def test_the_fingerprint_sees_changes_the_sampled_token_misses(teacher) -> None:
    # The reviewer's case: permute the entries of every position except the sampled token.
    samples, distributions = run(teacher, n=4, record_distributions=True)
    permuted = []
    for s, row in zip(samples, distributions, strict=True):
        changed = row.clone()
        for t, token in enumerate(s.response_ids):
            others = [v for v in range(row.shape[1]) if v != token]
            changed[t, others] = row[t, others[::-1]]
        permuted.append(changed)
    for s, row in zip(samples, permuted, strict=True):
        picked = row.gather(1, torch.tensor(s.response_ids)[:, None])[:, 0]
        assert torch.equal(picked, torch.tensor(s.logprobs))  # the sampled-token check passes
    (digest, projections), (other, moved) = fingerprint(distributions), fingerprint(permuted)
    assert digest != other
    assert (moved - projections).abs().max().item() > 1e-3


def test_fingerprints_round_trip_and_tamper_check(tmp_path: Path, teacher) -> None:
    recorded: list = []
    samples, _ = run(teacher, n=6, fingerprints=recorded)
    write_stream(
        tmp_path, samples, {"fingerprints": write_fingerprints(tmp_path, recorded, [4, 2])}
    )
    loaded = load_fingerprints(tmp_path)
    assert [d for d, _ in loaded] == [d for d, _ in recorded]
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(loaded, recorded, strict=True))
    manifest = load_manifest(tmp_path)["fingerprints"]
    assert [(b["first"], b["responses"]) for b in manifest["batches"]] == [(0, 4), (4, 2)]
    (tmp_path / FINGERPRINTS_FILE).write_bytes(b"x")
    with pytest.raises(StreamError, match="different from the chunk manifest"):
        load_fingerprints(tmp_path)
