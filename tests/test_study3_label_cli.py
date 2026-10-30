"""Study 3's labeling command (analysis/study3_label_cli.py): the control list's one-time draw from
study 1's judge calibration and its fetch by id and hash, on a synthetic calibration folder of
benign placeholder items (no study-1 text is opened); the persistent batch with its controls, its
one relabel, the judge's mark and its diagnosis, against a fake judge server that answers by
content. Nothing reads a runs/ folder of the checkout."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_study2_labels import IDENTITY, factory
from test_study3_labels import ContentJudge

from zero_field_refusal.analysis import gate1, study3_label_cli
from zero_field_refusal.analysis.study3_label_cli import (
    ControlMismatch,
    changed_controls,
    draw_controls,
    fetch_controls,
    label_controls,
)
from zero_field_refusal.analysis.study3_pins import (
    CONTROLS,
    control_list_problems,
    read_control_list,
)

CODE = {"commit": "a" * 40}
WHEN = "2026-10-31T12:00:00+00:00"
ARCHIVE = "gs://bucket/runs/judge_calibration/calibration"
CALIBRATION = Path("judge_calibration") / "calibration"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def calibration_items(n: int, marked: bool = False) -> list[dict]:
    """``n`` benign placeholder items, ids ``c%012x``; the even ones' responses hold ``[h]``.
    ``marked``: each text names its item as ``control-<i> `` (the fake judge's markers)."""
    items = []
    for i in range(n):
        name = f"control-{i} " if marked else f"control prompt {i}"
        prompt = f"Placeholder {name}prompt." if marked else f"Placeholder {name}."
        response = (f"Placeholder control-{i} response." if marked
                    else f"Placeholder control response {i}")  # fmt: skip
        items.append({"item_id": f"c{i:012x}", "prompt": prompt,
                      "response": response + (" [h]" if i % 2 == 0 else "")})  # fmt: skip
    return items


def write_calibration(folder: Path, monkeypatch, n: int = 50, marked: bool = False) -> Path:
    """Study 1's judge-calibration folder, synthetic: ``sample.jsonl`` of ``n`` items and
    WildGuard's labels of them (``harmful_compliance`` for ``[h]``, else a refusal or a safe
    answer; an earlier unparsed record of the first item, which the latest replaces), with the
    fixed sample sha256 patched to this sample's."""
    folder.mkdir(parents=True)
    items = calibration_items(n, marked)
    write_jsonl(folder / "sample.jsonl", items)
    labels = [{"item_id": items[0]["item_id"], "label": None, **IDENTITY}]
    for i, item in enumerate(items):
        label = "harmful_compliance" if "[h]" in item["response"] else (
            "refusal", "safe_non_refusal")[i // 2 % 2]  # fmt: skip
        labels.append({"item_id": item["item_id"], "label": label, **IDENTITY})
    write_jsonl(folder / "judge-wildguard.jsonl", labels)
    monkeypatch.setattr(gate1, "CALIBRATION_SAMPLE_SHA256", sha256(folder / "sample.jsonl"))
    return folder


@pytest.fixture
def runs_root(tmp_path) -> Path:
    return tmp_path / "runs"


@pytest.fixture
def calibration(runs_root, monkeypatch) -> Path:
    return write_calibration(runs_root / CALIBRATION, monkeypatch)


@pytest.fixture
def controls_record(calibration, runs_root) -> dict:
    return draw_controls(calibration, seed=7, archive=ARCHIVE, runs_root=runs_root, code=CODE,
                         written_at=WHEN)  # fmt: skip


@pytest.fixture
def source(calibration) -> Path:
    """The calibration folder as restored for labeling."""
    return calibration


@pytest.fixture
def fake() -> ContentJudge:
    return ContentJudge()


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A checkout for the draw command: the control list is published in its docs/."""
    path = tmp_path / "repo"
    (path / "docs").mkdir(parents=True)
    monkeypatch.setattr(study3_label_cli, "REPO_ROOT", path)
    monkeypatch.setattr(study3_label_cli, "git_state", lambda repo: {**CODE, "dirty": False})
    return path


def retext(source: Path, item: int, record: dict | None = None) -> None:
    """The response of the control list's ``item``-th item changed in the source's sample."""
    rows = [json.loads(line) for line in (source / "sample.jsonl").read_text().splitlines()]
    wanted = record["items"][item]["id"] if record else rows[item]["item_id"]
    for row in rows:
        if row["item_id"] == wanted:
            row["response"] = "Placeholder response, changed."
    write_jsonl(source / "sample.jsonl", rows)


def with_sample_sha(record: dict, digest: str) -> dict:
    out = copy.deepcopy(record)
    out["source"]["sample"]["sha256"] = digest
    return out


# ------------------------------------------------------------------ the draw


def test_the_draw_is_its_seeds_and_holds_no_text(calibration, runs_root):
    kw = dict(archive=ARCHIVE, runs_root=runs_root, code=CODE, written_at=WHEN)
    a, b = draw_controls(calibration, seed=7, **kw), draw_controls(calibration, seed=7, **kw)
    assert a == b and control_list_problems(a) == [] and "Placeholder" not in json.dumps(a)
    assert sum(i["expected"] == "harmful" for i in a["items"]) == 20
    assert draw_controls(calibration, seed=8, **kw)["items"] != a["items"]
    assert a["source"] == {
        "sample": {"path": "judge_calibration/calibration/sample.jsonl",
                   "sha256": sha256(calibration / "sample.jsonl")},
        "labels": {"path": "judge_calibration/calibration/judge-wildguard.jsonl",
                   "sha256": sha256(calibration / "judge-wildguard.jsonl")},
        "archive": ARCHIVE}  # fmt: skip
    rows = {r["item_id"]: r for r in map(json.loads, (calibration / "sample.jsonl").open())}
    for item in a["items"]:
        row = rows[item["id"]]
        assert item["prompt_sha256"] == hashlib.sha256(row["prompt"].encode()).hexdigest()
        assert item["response_sha256"] == hashlib.sha256(row["response"].encode()).hexdigest()
        assert (item["expected"] == "harmful") == ("[h]" in row["response"])
    assert (a["seed"], a["code"], a["written_at"]) == (7, CODE, WHEN)


@pytest.mark.parametrize("broken", ["sample_sha", "few_harmful", "other_identity", "outside"])
def test_the_draw_refuses(calibration, broken, runs_root, monkeypatch, tmp_path):
    kw = dict(seed=7, archive=ARCHIVE, runs_root=runs_root, code=CODE, written_at=WHEN)
    labels = calibration / "judge-wildguard.jsonl"
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    if broken == "sample_sha":
        monkeypatch.setattr(gate1, "CALIBRATION_SAMPLE_SHA256", "0" * 64)
        words = "sha256"
    elif broken == "few_harmful":  # six harmful items relabeled later: 19 left
        harmful = [r["item_id"] for r in rows if r["label"] == "harmful_compliance"][:6]
        write_jsonl(labels, rows + [{"item_id": i, "label": "refusal", **IDENTITY}
                                    for i in harmful])  # fmt: skip
        words = "19 harmful"
    elif broken == "other_identity":
        write_jsonl(labels, rows + [{**rows[-1], "judge_revision": "f" * 40}])
        words = "identity"
    else:
        kw["runs_root"] = tmp_path / "elsewhere"
        words = "RUNS_DIR"
    with pytest.raises(ValueError, match=words):
        draw_controls(calibration, **kw)


def test_the_draw_command_writes_once_and_prints_counts(calibration, runs_root, repo, capfd):
    argv = ["controls-draw", "--calibration", str(calibration), "--seed", "7", "--archive",
            ARCHIVE, "--runs-root", str(runs_root)]  # fmt: skip
    study3_label_cli.main(argv)
    (line,) = capfd.readouterr().out.splitlines()
    shown = json.loads(line)
    assert shown == {"controls": 40, "harmful": 20, "not_harmful": 20, "path": CONTROLS,
                     "sha256": sha256(repo / CONTROLS)}  # fmt: skip
    record, problems = read_control_list(repo)
    assert problems == [] and record["seed"] == 7 and record["code"] == CODE
    with pytest.raises(SystemExit, match="already"):
        study3_label_cli.main(argv)
    assert "Placeholder" not in capfd.readouterr().out


# ------------------------------------------------------------------ the fetch


def test_the_fetch_gives_the_pinned_items_by_id(controls_record, source):
    items = fetch_controls(controls_record, source)
    assert [i["item_id"] for i in items] == [f"control/{i['id']}" for i in controls_record["items"]]
    for item, pinned in zip(items, controls_record["items"], strict=True):
        assert hashlib.sha256(item["prompt"].encode()).hexdigest() == pinned["prompt_sha256"]


def test_a_hash_mismatch_refuses_before_any_request(controls_record, source, fake):
    retext(source, item=3, record=controls_record)  # one response changed
    with pytest.raises(ValueError, match="sample"):
        fetch_controls(controls_record, source)
    repinned = with_sample_sha(controls_record, sha256(source / "sample.jsonl"))
    with pytest.raises(ControlMismatch) as caught:
        fetch_controls(repinned, source)
    assert caught.value.ids == [controls_record["items"][3]["id"]] and fake.attempts == 0
    assert "Placeholder" not in str(caught.value)


@pytest.mark.parametrize("how", ["missing", "doubled"])
def test_a_missing_or_doubled_control_is_named(controls_record, source, how):
    rows = [json.loads(line) for line in (source / "sample.jsonl").read_text().splitlines()]
    wanted = controls_record["items"][5]["id"]
    found = next(r for r in rows if r["item_id"] == wanted)
    rows = [r for r in rows if r is not found] if how == "missing" else [*rows, found]
    write_jsonl(source / "sample.jsonl", rows)
    with pytest.raises(ControlMismatch, match=wanted) as caught:
        fetch_controls(with_sample_sha(controls_record, sha256(source / "sample.jsonl")), source)
    assert caught.value.ids == [wanted]


# ------------------------------------------------------------------ the check


def test_a_null_or_flipped_control_is_changed(controls_record):
    labels = {i["id"]: "harmful_compliance" if i["expected"] == "harmful" else "refusal"
              for i in controls_record["items"]}  # fmt: skip
    assert changed_controls(controls_record, labels) == ([], [])
    one, two = (i["id"] for i in controls_record["items"][:2])
    labels[one] = None
    labels[two] = "refusal" if labels[two] == "harmful_compliance" else "harmful_compliance"
    assert changed_controls(controls_record, labels) == (sorted([one, two]), [two])


def test_the_controls_are_labeled_anew_on_their_own_fresh_cache(controls_record, source, tmp_path):
    items = fetch_controls(controls_record, source)
    judge = ContentJudge()
    cache = tmp_path / "labeling" / "batches" / "b0001" / "controls-k1-a1.jsonl"
    labels = label_controls(items, factory(judge), cache)
    assert set(labels) == {i["id"] for i in controls_record["items"]}
    assert changed_controls(controls_record, labels) == ([], []) and judge.attempts == 40
    with pytest.raises(ValueError, match="exists"):
        label_controls(items, factory(judge), cache)
    assert judge.attempts == 40


def test_an_unparsed_control_is_retried_within_the_rounds_then_null(controls_record, source,
                                                                    tmp_path):  # fmt: skip
    items = fetch_controls(controls_record, source)
    items[0] = {**items[0], "response": items[0]["response"] + " [x]"}  # never parses
    judge = ContentJudge()
    labels = label_controls(items, factory(judge), tmp_path / "controls-k1-a1.jsonl")
    assert labels[controls_record["items"][0]["id"]] is None and judge.attempts == 40 + 2
    changed, parsed = changed_controls(controls_record, labels)
    assert changed == [controls_record["items"][0]["id"]] and parsed == []
