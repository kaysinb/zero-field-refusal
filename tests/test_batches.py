import torch
from conftest import PAD_ID

from zero_field_refusal.batches import chunks, collate
from zero_field_refusal.teacher import Sample


def sample(prompt, response, ended=True):
    return Sample("p", tuple(prompt), tuple(response), tuple(-0.5 for _ in response), ended)


def test_collate_right_pads_and_marks_response_targets() -> None:
    batch = collate(
        [sample([10, 11, 12], [20, 21, 5]), sample([10], [22, 23, 24, 25], False)],
        pad_id=PAD_ID,
        max_response_tokens=96,
    )
    assert batch.input_ids.tolist() == [
        [10, 11, 12, 20, 21, 5],
        [10, 22, 23, 24, 25, PAD_ID],
    ]
    assert batch.attention_mask.tolist() == [[1] * 6, [1] * 5 + [0]]
    # target_mask[b, t] is True when input_ids[b, t + 1] is a response token.
    assert batch.target_mask.tolist() == [
        [False, False, True, True, True],
        [True, True, True, True, False],
    ]
    assert batch.num_tokens == 7 and batch.lengths == [3, 4]
    assert batch.input_ids[:, 1:][batch.target_mask].tolist() == [20, 21, 5, 22, 23, 24, 25]


def test_an_end_token_equal_to_the_pad_id_stays_a_target() -> None:
    batch = collate([sample([10, 11], [20, PAD_ID]), sample([10], [21])], pad_id=PAD_ID,
                    max_response_tokens=96)  # fmt: skip
    assert batch.input_ids[:, 1:][batch.target_mask].tolist() == [20, PAD_ID, 21]


def test_truncation_to_the_cap_keeps_the_first_tokens_only() -> None:
    long = sample([10], [20, 21, 22, 23, 5])
    short = sample([10], [24, 5])
    batch = collate([long, short], pad_id=PAD_ID, max_response_tokens=3)
    assert batch.input_ids[:, 1:][batch.target_mask].tolist() == [20, 21, 22, 24, 5]
    assert batch.lengths == [3, 2]


def test_collate_uses_long_tensors_on_the_requested_device() -> None:
    batch = collate([sample([1], [2])], pad_id=PAD_ID, max_response_tokens=4, device="cpu")
    assert batch.input_ids.dtype == torch.long and batch.target_mask.dtype == torch.bool


def test_chunks_split_in_order() -> None:
    assert chunks(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]
