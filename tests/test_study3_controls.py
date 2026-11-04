"""Study 3's judge control items (analysis/study3_controls.py): the draw from a synthetic
calibration archive, the record's form and the refusal of a changed item. Synthetic, harmless text
only."""

import json
from pathlib import Path

import pytest

from zero_field_refusal.analysis import study3_controls
from zero_field_refusal.analysis.study3_controls import (
    ITEMS_FILE,
    LABELS_FILE,
    PER_CLASS,
    RECORD,
    SEED,
    ControlsRefused,
    load_controls,
    record_bytes,
    select_controls,
    text_sha256,
)
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import _content_hash, judge_identity
from zero_field_refusal.paths import REPO_ROOT

IDENTITY = judge_identity(load_config("judge/wildguard"))
# the archive's labels by item number: 30 harmful, 25 refusal, 5 safe, 5 unparsed
LABELS = (["harmful_compliance"] * 30 + ["refusal"] * 25 + ["safe_non_refusal"] * 5
          + [None] * 5)  # fmt: skip


def item(i: int) -> dict:
    return {"item_id": f"cal{i:03d}", "prompt": f"Calibration prompt {i:03d}.",
            "response": f"Calibration response {i:03d}."}  # fmt: skip


def make_archive(root: Path) -> Path:
    """A calibration folder: the item file and WildGuard's labels of it."""
    root.mkdir(parents=True, exist_ok=True)
    items = [item(i) for i in range(len(LABELS))]
    (root / ITEMS_FILE).write_text("".join(json.dumps(x) + "\n" for x in items))
    labels = [{"item_id": x["item_id"], "label": label, **IDENTITY,
               "content_sha256": _content_hash(x["prompt"], x["response"])}
              for x, label in zip(items, LABELS, strict=True)]  # fmt: skip
    (root / LABELS_FILE).write_text("".join(json.dumps(x) + "\n" for x in labels))
    return root


def test_the_draw_is_20_harmful_and_20_not_once_by_the_seed(tmp_path) -> None:
    """Twenty items WildGuard labeled harmful and twenty not, parsed labels only, the same on a
    second draw; each pinned by its id, its two sha256 and its expected label, and no text."""
    archive = make_archive(tmp_path / "calibration")
    record = select_controls(archive)
    assert record["seed"] == SEED == 20261104 and record["identity"] == IDENTITY
    labels = {f"cal{i:03d}": label for i, label in enumerate(LABELS)}
    drawn = [entry["item_id"] for entry in record["items"]]
    assert len(set(drawn)) == 2 * PER_CLASS
    assert [entry["expected_label"] for entry in record["items"]] == [labels[i] for i in drawn]
    assert sum(labels[i] == "harmful_compliance" for i in drawn) == PER_CLASS
    assert all(labels[i] is not None for i in drawn)
    for entry in record["items"]:
        x = item(int(entry["item_id"].removeprefix("cal")))
        assert entry["prompt_sha256"] == text_sha256(x["prompt"])
        assert entry["response_sha256"] == text_sha256(x["response"])
    data = record_bytes(record)
    assert data == record_bytes(select_controls(archive))
    assert b"Calibration prompt" not in data and b"Calibration response" not in data


def test_load_controls_gives_the_items_and_refuses_a_changed_one(tmp_path) -> None:
    """The 40 items in the record's order; an item whose prompt or response changed, or that is
    gone, refuses the whole call, naming its id and never its text."""
    archive = make_archive(tmp_path / "calibration")
    record = select_controls(archive)
    controls = load_controls(record, archive)
    assert [c["item_id"] for c in controls] == [f"control/{e['item_id']}" for e in record["items"]]
    assert [c["expected_label"] for c in controls] == [e["expected_label"]
                                                      for e in record["items"]]  # fmt: skip
    target = record["items"][7]["item_id"]
    rows = [json.loads(line) for line in (archive / ITEMS_FILE).read_text().splitlines()]
    for field, change in (("response", "Calibration response changed."), ("drop", None)):
        edited = [r for r in rows if not (field == "drop" and r["item_id"] == target)]
        edited = [{**r, field: change} if r["item_id"] == target and change else r
                  for r in edited]  # fmt: skip
        (archive / ITEMS_FILE).write_text("".join(json.dumps(r) + "\n" for r in edited))
        with pytest.raises(ControlsRefused, match=target) as refused:
            load_controls(record, archive)
        assert "Calibration" not in str(refused.value)


def test_a_label_of_another_item_file_refuses_the_draw(tmp_path) -> None:
    """The item file is the one the labels label: a label whose content hash is not of the item
    it names refuses the draw."""
    archive = make_archive(tmp_path / "calibration")
    rows = [json.loads(line) for line in (archive / ITEMS_FILE).read_text().splitlines()]
    rows[3]["prompt"] = "Another calibration prompt."
    (archive / ITEMS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ControlsRefused, match="cal003"):
        select_controls(archive)


def test_select_writes_the_record_once_and_prints_counts_only(tmp_path, capsys) -> None:
    archive = make_archive(tmp_path / "calibration")
    out = tmp_path / "controls.json"
    study3_controls.main(["select", "--archive", str(archive), "--out", str(out)])
    (line,) = capsys.readouterr().out.splitlines()
    shown = json.loads(line)
    assert set(shown) == {"harmful", "not_harmful", "record", "sha256"}
    assert out.read_bytes() == record_bytes(select_controls(archive))
    with pytest.raises(SystemExit):
        study3_controls.main(["select", "--archive", str(archive), "--out", str(out)])


def test_the_committed_record_is_a_control_record() -> None:
    """docs/study3-judge-controls.json: the seed, WildGuard's identity and 20 + 20 items of the
    record's form (its items are checked against the archive where the archive is)."""
    record = json.loads((REPO_ROOT / RECORD).read_text())
    assert record["seed"] == SEED and record["identity"] == IDENTITY
    assert study3_controls._checked(record)
    assert record["source"] == {**record["source"], "root": "runs",
                                "directory": study3_controls.ARCHIVE}  # fmt: skip
