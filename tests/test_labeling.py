import json
from collections import Counter
from pathlib import Path

import pytest

from zero_field_refusal import judge
from zero_field_refusal.labeling import (
    import_reference_labels,
    load_labels,
    make_calibration_sample,
    run_session,
    write_calibration_files,
)


def make_pool() -> list[dict]:
    pool = []
    for model, kind, count in [("m1", "base", 60), ("m1", "trained", 30), ("m2", "base", 8)]:
        for i in range(count):
            pool.append(
                {
                    "item_id": f"{model}-{kind}-{i}",
                    "prompt": f"prompt {model} {kind} {i}",
                    "response": f"response {i}",
                    "model": model,
                    "kind": kind,
                    "run": f"run-{model}",
                }
            )
    return pool


def strata(item: dict) -> tuple[str, str]:
    return item["model"], item["kind"]


def test_sample_allocation_is_proportional_with_a_minimum() -> None:
    blinded, key = make_calibration_sample(
        make_pool(), 49, strata_key=strata, seed=0, min_per_stratum=0
    )
    counts = Counter(strata(key[b["item_id"]]) for b in blinded)
    # 98 items; ideal 30, 15, 4.
    assert counts == {("m1", "base"): 30, ("m1", "trained"): 15, ("m2", "base"): 4}


def test_sample_minimum_per_stratum_lifts_small_strata() -> None:
    blinded, key = make_calibration_sample(
        make_pool(), 20, strata_key=strata, seed=0, min_per_stratum=5
    )
    counts = Counter(strata(key[b["item_id"]]) for b in blinded)
    assert counts[("m2", "base")] >= 5
    assert sum(counts.values()) == 20


def test_sample_strata_key_may_be_field_names() -> None:
    blinded, key = make_calibration_sample(make_pool(), 20, strata_key=("model", "kind"), seed=0)
    assert len(blinded) == len(key) == 20


def test_sample_is_deterministic_and_seed_dependent() -> None:
    a = make_calibration_sample(make_pool(), 30, strata_key=strata, seed=1)
    b = make_calibration_sample(make_pool(), 30, strata_key=strata, seed=1)
    c = make_calibration_sample(make_pool(), 30, strata_key=strata, seed=2)
    assert a == b
    assert a != c


def test_blinded_items_carry_only_id_prompt_response() -> None:
    blinded, key = make_calibration_sample(make_pool(), 30, strata_key=strata, seed=0)
    assert all(set(b) == {"item_id", "prompt", "response"} for b in blinded)
    assert set(key) == {b["item_id"] for b in blinded}
    entry = key[blinded[0]["item_id"]]
    assert {"model", "kind", "run", "original_item_id"} <= set(entry)
    assert "prompt" not in entry and "response" not in entry


def test_item_ids_are_opaque_and_unique() -> None:
    blinded, _ = make_calibration_sample(make_pool(), 40, strata_key=strata, seed=0)
    ids = [b["item_id"] for b in blinded]
    assert len(set(ids)) == 40
    assert not any(part in i for i in ids for part in ("m1", "m2", "base", "trained"))


def test_order_does_not_follow_strata() -> None:
    blinded, key = make_calibration_sample(make_pool(), 40, strata_key=strata, seed=0)
    order = [strata(key[b["item_id"]]) for b in blinded]
    assert order != sorted(order)


def test_sample_rejects_too_large_n() -> None:
    with pytest.raises(ValueError, match="pool"):
        make_calibration_sample(make_pool()[:5], 10, strata_key=strata, seed=0)


def test_write_calibration_files(tmp_path: Path) -> None:
    blinded, key = make_calibration_sample(make_pool(), 10, strata_key=strata, seed=0)
    sample_path, key_path = tmp_path / "sample.jsonl", tmp_path / "key.json"
    write_calibration_files(blinded, key, sample_path, key_path)
    lines = [json.loads(line) for line in sample_path.read_text().splitlines()]
    assert lines == blinded
    assert json.loads(key_path.read_text()) == key
    assert '"model"' not in sample_path.read_text()


# --- session ------------------------------------------------------------------------------------

ITEMS = [
    {"item_id": f"x{i}", "prompt": f"prompt {i}", "response": f"response {i}"} for i in range(4)
]


class Script:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.shown: list[str] = []

    def input(self, prompt: str = "") -> str:
        if not self.answers:
            raise AssertionError("session asked for more input than scripted")
        return self.answers.pop(0)

    def output(self, text: str = "") -> None:
        self.shown.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.shown)


def run(answers: list[str], path: Path, **kwargs) -> Script:
    script = Script(answers)
    run_session(ITEMS, path, input_fn=script.input, output_fn=script.output, **kwargs)
    return script


def latest(path: Path) -> dict[str, str]:
    return {k: v["label"] for k, v in load_labels(path).items()}


def test_session_labels_and_finishes(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    script = run(["r", "s", "h", "u"], path)
    assert latest(path) == {
        "x0": "refusal",
        "x1": "safe_non_refusal",
        "x2": "harmful_compliance",
        "x3": "unsure",
    }
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(set(r) == {"item_id", "label", "labeled_at"} for r in records)
    assert "1/4" in script.text and "4/4" in script.text
    assert "prompt 0" in script.text and "response 3" in script.text


def test_definitions_shown_once(tmp_path: Path) -> None:
    script = run(["r", "r", "q"], tmp_path / "labels.jsonl", width=1000)
    first = next(iter(judge.LABEL_DEFINITIONS.values()))
    assert script.text.count(first) == 1


def test_session_back_revises_previous(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    run(["r", "s", "b", "h", "u", "q"], path)
    # x0=r, x1=s, back to x1 -> h, then x2=u, then quit at x3.
    assert latest(path) == {
        "x0": "refusal",
        "x1": "harmful_compliance",
        "x2": "unsure",
    }
    assert len(path.read_text().splitlines()) == 4  # append-only; the latest record wins


def test_back_at_first_item_stays(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    script = run(["b", "r", "q"], path)
    assert latest(path) == {"x0": "refusal"}
    assert script.text.count("1/4") == 2


def test_invalid_key_asks_again(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    run(["z", "", "r", "q"], path)
    assert latest(path) == {"x0": "refusal"}


def test_quit_and_resume_at_first_unlabeled(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    run(["r", "s", "q"], path)
    script = run(["h", "u"], path)
    assert latest(path) == {
        "x0": "refusal",
        "x1": "safe_non_refusal",
        "x2": "harmful_compliance",
        "x3": "unsure",
    }
    assert "3/4" in script.text and "1/4" not in script.text


def test_resume_when_complete_asks_nothing(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    run(["r", "r", "r", "r"], path)
    script = run([], path)
    assert "all 4 items" in script.text.lower()


def test_back_after_resume_revises_last_saved(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    run(["r", "q"], path)
    run(["b", "h", "q"], path)
    assert latest(path)["x0"] == "harmful_compliance"


def test_end_of_input_quits_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"

    def eof(prompt: str = "") -> str:
        raise EOFError

    run_session(ITEMS, path, input_fn=eof, output_fn=lambda text="": None)
    assert not path.exists()


def test_long_response_is_wrapped_in_full(tmp_path: Path) -> None:
    long = " ".join(f"word{i}" for i in range(400))
    items = [{"item_id": "L", "prompt": "p", "response": long}]
    script = Script(["q"])
    run_session(
        items, tmp_path / "l.jsonl", input_fn=script.input, output_fn=script.output, width=60
    )
    lines = "\n".join(script.shown).splitlines()
    assert "word399" in " ".join(lines).split()
    assert all(len(line) <= 60 for line in lines)


def test_torn_last_label_line_is_skipped_and_the_item_asked_again(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    run(["r", "s", "h", "q"], path)
    with path.open("a") as handle:
        handle.write('{"item_id": "x3", "lab')  # a write cut short
    with pytest.warns(UserWarning, match="truncated"):
        script = run(["u"], path)
    assert latest(path)["x3"] == "unsure"
    assert "4/4" in script.text


def test_malformed_label_line_in_the_middle_raises(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text('{"item_id": "x0", "label": "refusal"}\nnot json\n{"item_id": "x1"}\n')
    with pytest.raises(json.JSONDecodeError):
        load_labels(path)


# --- reference labels -----------------------------------------------------------------------------


def reference_inputs(tmp_path: Path, labels: list[dict]) -> tuple[Path, Path, Path]:
    sample = tmp_path / "calibration" / "sample.jsonl"
    sample.parent.mkdir()
    sample.write_text("".join(
        json.dumps({"item_id": f"x{i}", "prompt": "p", "response": "r"}) + "\n" for i in range(3)
    ))  # fmt: skip
    source = tmp_path / "from_ensemble"
    (source / "labelers").mkdir(parents=True)
    (source / "labelers" / "opus.jsonl").write_text('{"item_id": "x0", "label": "refusal"}\n')
    (source / "arbiter.jsonl").write_text("{}\n")
    labels_path = source / "labels.jsonl"
    labels_path.write_text("".join(json.dumps(r) + "\n" for r in labels))
    return labels_path, source, sample


def labels_for(ids, label="refusal") -> list[dict]:
    return [{"item_id": i, "label": label, "labeled_at": "2026-10-11T09:00:00+00:00"} for i in ids]


def test_reference_labels_are_imported_with_their_provenance(tmp_path: Path) -> None:
    labels, source, sample = reference_inputs(tmp_path, labels_for(["x2", "x0", "x1"]))
    manifest = import_reference_labels(labels, source, sample)
    out = sample.parent
    records = [
        json.loads(line) for line in (out / "reference_labels.jsonl").read_text().splitlines()
    ]
    assert [r["item_id"] for r in records] == ["x0", "x1", "x2"]  # sample order
    assert {r["labeler"] for r in records} == {"ensemble"}
    assert load_labels(out / "reference_labels.jsonl")["x1"]["label"] == "refusal"
    assert (out / "reference_provenance" / "labelers" / "opus.jsonl").is_file()
    assert set(manifest["provenance"]) == {"labelers/opus.jsonl", "arbiter.jsonl", "labels.jsonl"}
    assert manifest["label_counts"] == {"refusal": 3}
    assert json.loads((out / "reference_import.json").read_text()) == manifest
    with pytest.raises(FileExistsError):
        import_reference_labels(labels, source, sample)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (labels_for(["x0", "x1"]), "no record for 1 sample items"),
        (labels_for(["x0", "x1", "x2", "x9"]), "outside the sample"),
        (labels_for(["x0", "x0", "x1", "x2"]), "several records"),
        (labels_for(["x0", "x1", "x2"], label="harmful"), "unknown labels"),
        ([{"item_id": i, "label": "refusal"} for i in ("x0", "x1", "x2")], "labeled_at"),
    ],
)
def test_reference_labels_must_cover_the_sample_exactly(tmp_path: Path, labels, message) -> None:
    labels_path, source, sample = reference_inputs(tmp_path, labels)
    with pytest.raises(ValueError, match=message):
        import_reference_labels(labels_path, source, sample)
    assert not (sample.parent / "reference_labels.jsonl").exists()
