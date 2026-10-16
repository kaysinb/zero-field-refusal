"""Study-2 judge path: conditions of scheduled runs, labeling them, and the strict endpoint reader.
Synthetic text only."""

import hashlib
import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from zero_field_refusal.analysis.evidence import label_path
from zero_field_refusal.analysis.study2_labels import read_endpoint_labels
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import JudgeClient, _content_hash, judge_identity
from zero_field_refusal.run_judging import judge_items, label_runs, response_conditions

JUDGE = "wildguard"
IDS = [f"judged:{i}" for i in range(6)]
PROMPTS = {pid: f"Placeholder prompt {i}." for i, pid in enumerate(IDS)}
IDENTITY = judge_identity(load_config(f"judge/{JUDGE}"))
HARMFUL = "Harmful request: yes\nResponse refusal: no\nHarmful response: yes"
REFUSAL = "Harmful request: yes\nResponse refusal: yes\nHarmful response: no"


def text(condition: str, pid: str) -> str:
    return f"Placeholder response {condition} {pid}."


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def make_run(root: Path, steps=(25,), name="run", stop="plateau") -> Path:
    run = root / name
    (run / "responses").mkdir(parents=True)
    cfg = {"run_type": "train", "schedule": {"policy": "k_stopped"},
           "eval": {"harmful": {"prompt_set": "harmful_evaluation", "limit": None}}}  # fmt: skip
    OmegaConf.save(OmegaConf.create(cfg), run / "config.yaml")
    ids_hash = hashlib.sha256(json.dumps(IDS).encode()).hexdigest()
    meta = {"status": "completed",
            "inputs": {"harmful_prompts": {"set": "harmful_evaluation", "count": len(IDS),
                                           "ids_sha256": ids_hash}}}  # fmt: skip
    (run / "meta.json").write_text(json.dumps(meta))
    terminal = {"stop": {"reason": stop}, "evaluations": list(steps)}
    (run / "terminal.json").write_text(json.dumps(terminal))
    write_jsonl(run / "evaluations.jsonl",
                [{"step": s, "condition": f"step-{s}"} for s in steps])  # fmt: skip
    write_jsonl(run / "responses" / "harmful_evaluation.jsonl",
                [{"condition": f"step-{s}", "item_id": pid, "text": text(f"step-{s}", pid)}
                 for s in steps for pid in IDS])  # fmt: skip
    return run


def record(condition: str, pid: str, label: str | None, **overrides) -> dict:
    rec = {
        "item_id": f"{condition}/{pid}",
        "label": label,
        **IDENTITY,
        "content_sha256": _content_hash(PROMPTS[pid], text(condition, pid)),
    }
    return {**rec, **overrides}


def write_labels(run: Path, records: list[dict]) -> None:
    label_path(run, JUDGE).parent.mkdir(exist_ok=True)
    write_jsonl(label_path(run, JUDGE), records)


def read(run: Path):
    return read_endpoint_labels(run, JUDGE, IDENTITY, PROMPTS)


def conditions(run: Path) -> list[str]:
    return response_conditions(OmegaConf.load(run / "config.yaml"), run)


class FakeWildGuard:
    """The parts of ``openai.OpenAI`` the WildGuard backend uses; the replies repeat the last."""

    def __init__(self, replies) -> None:
        self.replies, self.calls, self.lock = list(replies), 0, threading.Lock()
        self.completions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        with self.lock:
            reply = self.replies[min(self.calls, len(self.replies) - 1)]
            self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(text=reply, finish_reason="stop")])


def factory(fake):
    cfg = load_config(f"judge/{JUDGE}")
    env = {"JUDGE_API_BASE": "http://judge/v1"}
    return lambda path: JudgeClient.from_config(cfg, env, client=fake, cache_path=path,
                                                sleep=lambda s: None)  # fmt: skip


# --- conditions --------------------------------------------------------------------------------


def test_conditions_of_a_fixed_25_step_run_and_of_a_run_with_two_evaluations(tmp_path) -> None:
    one, two = make_run(tmp_path, (25,), "one"), make_run(tmp_path, (50, 120), "two")
    assert conditions(one) == ["step-25"]
    assert conditions(two) == ["step-50", "step-120"]


def test_a_scheduled_run_needs_its_directory_and_theta0_is_unchanged(tmp_path) -> None:
    cfg = OmegaConf.load(make_run(tmp_path) / "config.yaml")
    with pytest.raises(ValueError, match="need its run directory"):
        response_conditions(cfg)
    cfg.run_type = "theta0"  # a θ₀ run keeps the schedule section of its config
    assert response_conditions(cfg) == ["theta0"]


def test_a_condition_that_is_not_step_n_stops_the_run(tmp_path) -> None:
    run = make_run(tmp_path)
    write_jsonl(run / "evaluations.jsonl", [{"step": 25, "condition": "step-26"}])
    with pytest.raises(ValueError, match=r"evaluation 1: condition 'step-26' is not 'step-<step>'"):
        conditions(run)
    write_jsonl(run / "evaluations.jsonl", [{"step": 25, "condition": "trained"}])
    with pytest.raises(ValueError, match="'trained' is not"):
        judge_items(run, PROMPTS)


def test_a_repeated_or_unlisted_evaluation_stops_the_run(tmp_path) -> None:
    run = make_run(tmp_path, (25, 50))
    write_jsonl(run / "evaluations.jsonl", [{"step": 25, "condition": "step-25"}] * 2)
    (run / "terminal.json").write_text(json.dumps({"evaluations": [25, 25]}))
    with pytest.raises(ValueError, match="step-25 twice"):
        conditions(run)
    write_jsonl(run / "evaluations.jsonl", [{"step": 25, "condition": "step-25"}])
    with pytest.raises(ValueError, match="record different evaluations"):
        conditions(run)


def test_judge_items_cover_every_evaluation(tmp_path) -> None:
    items = judge_items(make_run(tmp_path, (25, 50)), PROMPTS)
    assert [i["item_id"] for i in items][:2] == ["step-25/judged:0", "step-25/judged:1"]
    assert len(items) == 12
    assert items[7]["prompt"] == PROMPTS["judged:1"]


# --- labeling ----------------------------------------------------------------------------------


def test_labeling_a_study2_run_whatever_its_stop_and_retrying_unparseable_items(tmp_path) -> None:
    run = make_run(tmp_path, (25, 50), stop="t_max")
    broken = FakeWildGuard(["cannot classify", HARMFUL])  # the first request does not parse
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(broken), rounds=3)
    assert summaries == [{"run_id": "run", "items": 12, "unparseable": 0, "rounds": 2}]
    rows = read(run)
    assert [(r["condition"], r["labeled"], r["missing"], r["harmful"]) for r in rows] == [
        ("step-25", 6, 0, 6), ("step-50", 6, 0, 6)]  # fmt: skip


def test_rounds_stop_at_the_limit_and_report_what_stays_missing(tmp_path) -> None:
    run = make_run(tmp_path)
    fake = FakeWildGuard(["cannot classify"])
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    assert summaries == [{"run_id": "run", "items": 6, "unparseable": 6, "rounds": 3}]
    assert fake.calls == 18
    (row,) = read(run)
    assert (row["labeled"], row["missing"], row["harmful_rate"]) == (0, 6, None)


def test_a_run_without_an_evaluation_is_skipped_not_failed(tmp_path) -> None:
    run = make_run(tmp_path, ())
    (run / "responses" / "harmful_evaluation.jsonl").unlink()
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(FakeWildGuard([REFUSAL])))
    assert summaries == [{"run_id": "run", "skipped": "no evaluation recorded"}]
    assert read(run) == []


# --- reader ------------------------------------------------------------------------------------


def test_counts_with_bounds_and_file_hashes(tmp_path) -> None:
    run = make_run(tmp_path, (25, 50))
    labels = ["harmful_compliance", "harmful_compliance", "refusal", "safe_non_refusal", None]
    records = [
        record("step-25", pid, label) for pid, label in zip(IDS, labels, strict=False)
    ]  # 5 of 6
    records.append(record("step-25", "judged:4", "harmful_compliance"))  # null, then labeled
    records.append(record("step-25", "judged:0", "harmful_compliance"))  # the same label twice
    records += [record("step-50", pid, None) for pid in IDS]  # all null: nothing labeled
    write_labels(run, records)
    first, second = read(run)
    assert (first["condition"], first["step"], first["prompts"]) == ("step-25", 25, 6)
    assert (first["labeled"], first["missing"]) == (5, 1)  # judged:5 has no record
    assert (first["harmful"], first["refusal"], first["safe_non_refusal"]) == (3, 1, 1)
    assert first["harmful_rate"] == pytest.approx(3 / 5)
    assert first["harmful_low"] == pytest.approx(3 / 6)
    assert first["harmful_high"] == pytest.approx(4 / 6)
    assert (second["labeled"], second["missing"], second["harmful_rate"]) == (0, 6, None)
    assert (second["harmful_low"], second["harmful_high"]) == (0.0, 1.0)
    for name, digest in first["files"].items():
        assert digest == hashlib.sha256((run / name).read_bytes()).hexdigest()
    assert set(first["files"]) == {"config.yaml", "evaluations.jsonl", "terminal.json",
                                   "meta.json", "responses/harmful_evaluation.jsonl",
                                   "judge/wildguard.jsonl"}  # fmt: skip


@pytest.mark.parametrize("bad", ["missing", "duplicate", "unlisted", "other_condition"])
def test_the_responses_must_be_exactly_the_prompts_of_each_evaluation(tmp_path, bad) -> None:
    run = make_run(tmp_path)
    write_labels(run, [])
    path = run / "responses" / "harmful_evaluation.jsonl"
    lines = path.read_text().splitlines()
    extra = {"condition": "step-25", "item_id": "judged:99", "text": "x"}
    other = {"condition": "step-30", "item_id": "judged:0", "text": "x"}
    changed = {"missing": lines[1:], "duplicate": [*lines, lines[0]],
               "unlisted": [*lines, json.dumps(extra)],
               "other_condition": [*lines, json.dumps(other)]}[bad]  # fmt: skip
    path.write_text("\n".join(changed) + "\n")
    pattern = {"missing": "1 missing .first: step-25/judged:0",
               "duplicate": "step-25/judged:0 twice",
               "unlisted": "1 unexpected .first: step-25/judged:99",
               "other_condition": "1 unexpected .first: step-30/judged:0"}[bad]  # fmt: skip
    with pytest.raises(ValueError, match=pattern) as caught:
        read(run)
    assert str(path) in str(caught.value)


def test_the_recorded_prompt_set_must_be_the_one_given(tmp_path) -> None:
    run = make_run(tmp_path)
    write_labels(run, [])
    with pytest.raises(ValueError, match="ids_sha256"):
        read_endpoint_labels(run, JUDGE, IDENTITY, {pid: PROMPTS[pid] for pid in IDS[::-1]})
    with pytest.raises(ValueError, match="count"):
        read_endpoint_labels(run, JUDGE, IDENTITY, {pid: PROMPTS[pid] for pid in IDS[:5]})


def test_a_label_of_an_older_response_text_is_stale_and_fails(tmp_path) -> None:
    run = make_run(tmp_path)
    older = _content_hash(PROMPTS["judged:2"], "An older response.")
    write_labels(run, [record("step-25", "judged:1", "refusal"),
                       record("step-25", "judged:2", "refusal", content_sha256=older)])  # fmt: skip
    with pytest.raises(ValueError, match="label of step-25/judged:2 is not of the response"):
        read(run)
    no_hash = record("step-25", "judged:3", "refusal")
    del no_hash["content_sha256"]
    write_labels(run, [no_hash])
    with pytest.raises(ValueError, match="step-25/judged:3 is not of the response"):
        read(run)


def test_a_stale_null_record_fails_too(tmp_path) -> None:
    run = make_run(tmp_path)
    write_labels(run, [record("step-25", "judged:0", None, content_sha256="0" * 64)])
    with pytest.raises(ValueError, match="step-25/judged:0"):
        read(run)


@pytest.mark.parametrize("field", ["backend", "prompt_version", "judge_model", "judge_revision"])
def test_a_label_of_another_judge_fails(tmp_path, field) -> None:
    run = make_run(tmp_path)
    write_labels(run, [record("step-25", "judged:0", "refusal"),
                       record("step-25", "judged:1", "refusal", **{field: "other"})])  # fmt: skip
    with pytest.raises(ValueError, match="step-25/judged:1 is labeled by another judge"):
        read(run)


def test_conflicting_labels_fail_and_so_does_a_label_outside_the_set(tmp_path) -> None:
    run = make_run(tmp_path)
    write_labels(run, [record("step-25", "judged:1", "refusal"),
                       record("step-25", "judged:1", "harmful_compliance")])  # fmt: skip
    with pytest.raises(ValueError, match="step-25/judged:1 has two different labels"):
        read(run)
    write_labels(run, [record("step-25", "judged:1", "unsure")])
    with pytest.raises(ValueError, match="step-25/judged:1 has the label 'unsure'"):
        read(run)


def test_a_record_naming_no_response_fails(tmp_path) -> None:
    run = make_run(tmp_path)
    write_labels(run, [record("step-50", "judged:1", "refusal")])
    with pytest.raises(ValueError, match="step-50/judged:1 names no response"):
        read(run)


def test_an_unlabeled_run_and_a_non_study2_run_are_refused(tmp_path) -> None:
    run = make_run(tmp_path)
    with pytest.raises(ValueError, match="is missing: the run is not labeled"):
        read(run)
    cfg = OmegaConf.load(run / "config.yaml")
    cfg.run_type = "theta0"
    OmegaConf.save(cfg, run / "config.yaml")
    with pytest.raises(ValueError, match="not a study-2 training run"):
        read(run)


def test_a_copied_run_reads_the_same(tmp_path) -> None:
    run = make_run(tmp_path)
    write_labels(run, [record("step-25", pid, "refusal") for pid in IDS])
    shutil.copytree(run, tmp_path / "copy")
    assert read(tmp_path / "copy") == read(run)
