"""Study-2 judge path: conditions of scheduled runs, labeling them with their judging receipts, and
the strict endpoint reader. Synthetic text only."""

import fcntl
import hashlib
import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from zero_field_refusal.analysis.evidence import label_path
from zero_field_refusal.analysis.study2_labels import LabelsPending, read_endpoint_labels
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import JudgeClient, _content_hash, judge_identity
from zero_field_refusal.run_judging import (
    LABEL_ROUNDS,
    NOT_ATTEMPTED,
    judge_items,
    label_runs,
    lock_path,
    receipt_path,
    response_conditions,
    rounds_path,
)

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
    if steps:  # both files are created at the first evaluation; a run stopped before has neither
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


def expected(run: Path) -> list[str]:
    return [f"{condition}/{pid}" for condition in conditions(run) for pid in IDS]


def write_receipt(run: Path, *, rounds: int = LABEL_ROUNDS, **changes) -> dict:
    """The receipt ``label_runs`` writes once the labeling is done, of the label file as it is."""
    path = label_path(run, JUDGE)
    labeled = set()
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("label") is not None:
            labeled.add(record.get("item_id"))
    items = expected(run)
    receipt = {"run_id": run.name, "judge": JUDGE, "identity": IDENTITY, "expected_items": items,
               "items": len(items), "rounds": rounds, "rounds_allowed": LABEL_ROUNDS,
               "missing": sum(1 for item in items if item not in labeled),
               "labels_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               **changes}  # fmt: skip
    receipt_path(run, JUDGE).write_text(json.dumps(receipt))
    return receipt


def write_labels(run: Path, records: list[dict], *, receipt: bool = True) -> None:
    """The label file of a finished labeling, with its receipt (``receipt`` False: none)."""
    label_path(run, JUDGE).parent.mkdir(exist_ok=True)
    write_jsonl(label_path(run, JUDGE), records)
    if receipt:
        write_receipt(run)


def write_raw(run: Path, text: str) -> None:
    """A label file holding ``text``, with the receipt of a finished labeling."""
    label_path(run, JUDGE).parent.mkdir(exist_ok=True)
    label_path(run, JUDGE).write_text(text)
    write_receipt(run)


def read(run: Path):
    return read_endpoint_labels(run, JUDGE, IDENTITY, PROMPTS)


def conditions(run: Path) -> list[str]:
    return response_conditions(OmegaConf.load(run / "config.yaml"), run)


class FakeWildGuard:
    """The parts of ``openai.OpenAI`` the WildGuard backend uses; the replies repeat the last.
    With ``fail_after``, every request after that many fails (a judge server gone). ``calls``
    counts the requests answered, ``attempts`` every request; ``watch()``, when given, is called
    at each request and its value kept in ``seen``."""

    def __init__(self, replies, fail_after: int | None = None, watch=None) -> None:
        self.replies, self.calls, self.lock = list(replies), 0, threading.Lock()
        self.fail_after, self.attempts, self.watch, self.seen = fail_after, 0, watch, []
        self.completions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        with self.lock:
            self.attempts += 1
            if self.watch is not None:
                self.seen.append(self.watch())
            if self.fail_after is not None and self.calls >= self.fail_after:
                raise RuntimeError("the judge server is gone")
            reply = self.replies[min(self.calls, len(self.replies) - 1)]
            self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(text=reply, finish_reason="stop")])


def factory(fake, workers: int = 8):
    cfg = load_config(f"judge/{JUDGE}")
    env = {"JUDGE_API_BASE": "http://judge/v1"}
    return lambda path: JudgeClient.from_config(
        cfg, env, client=fake, cache_path=path, max_workers=workers, sleep=lambda s: None
    )


def rounds_used(run: Path) -> int:
    """The rounds the run's rounds record counts."""
    return json.loads(rounds_path(run, JUDGE).read_text())["used"]


def judge_files(run: Path) -> dict[str, str]:
    """The sha256 of every file in the run's judge directory, by name."""
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((run / "judge").iterdir())}  # fmt: skip


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


def test_evaluations_missing_while_terminal_lists_steps_stops_the_run(tmp_path) -> None:
    run = make_run(tmp_path, (25,))
    (run / "evaluations.jsonl").unlink()
    with pytest.raises(ValueError, match="record different evaluations"):
        conditions(run)


def test_a_run_without_terminal_json_stops(tmp_path) -> None:
    run = make_run(tmp_path, (25,))
    (run / "terminal.json").unlink()
    with pytest.raises(ValueError, match="terminal.json is missing"):
        conditions(run)
    with pytest.raises(ValueError, match="terminal.json is missing"):
        judge_items(run, PROMPTS)


def test_responses_stored_for_a_run_without_evaluation_are_not_ignored(tmp_path) -> None:
    run = make_run(tmp_path, ())
    write_jsonl(run / "responses" / "harmful_evaluation.jsonl",
                [{"condition": "step-25", "item_id": "judged:0", "text": "x"}])  # fmt: skip
    with pytest.raises(ValueError, match="1 unexpected .first: step-25/judged:0"):
        judge_items(run, PROMPTS)


def test_the_discriminator_is_a_training_run_with_a_schedule(tmp_path) -> None:
    run = make_run(tmp_path, (25,))
    cfg = OmegaConf.load(run / "config.yaml")
    cfg.run_type = "curvature"  # keeps the schedule section, but is not a study-2 training run
    cfg.eval.controls = {"line_scan": {"enabled": False}, "random_direction": {"enabled": False}}
    assert response_conditions(cfg) == ["trained"]
    write_labels(run, [])
    OmegaConf.save(cfg, run / "config.yaml")
    with pytest.raises(ValueError, match="not a study-2 training run"):
        read(run)


def test_judge_items_cover_every_evaluation(tmp_path) -> None:
    items = judge_items(make_run(tmp_path, (25, 50)), PROMPTS)
    assert [i["item_id"] for i in items][:2] == ["step-25/judged:0", "step-25/judged:1"]
    assert len(items) == 12
    assert items[7]["prompt"] == PROMPTS["judged:1"]


# --- labeling ----------------------------------------------------------------------------------


@pytest.fixture
def published(monkeypatch):
    """The minimal runs here stand for runs whose terminal record was published with whole
    records (``stage_a.load_trajectory``; test_a_published_run_is_labeled_whatever_its_status
    uses whole records)."""
    monkeypatch.setattr("zero_field_refusal.run_judging.load_trajectory", lambda run_dir: None)


@pytest.mark.usefixtures("published")
def test_labeling_a_study2_run_whatever_its_stop_and_retrying_unparseable_items(tmp_path) -> None:
    run = make_run(tmp_path, (25, 50), stop="t_max")
    broken = FakeWildGuard(["cannot classify", HARMFUL])  # the first request does not parse
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(broken), rounds=3)
    assert summaries == [{"run_id": "run", "items": 12, "unparseable": 0, "rounds": 2}]
    rows = read(run)
    assert [(r["condition"], r["labeled"], r["missing"], r["harmful"]) for r in rows] == [
        ("step-25", 6, 0, 6), ("step-50", 6, 0, 6)]  # fmt: skip
    receipt = json.loads(receipt_path(run, JUDGE).read_text())
    assert receipt["expected_items"] == expected(run) and receipt["items"] == 12
    assert (receipt["run_id"], receipt["judge"], receipt["identity"]) == ("run", JUDGE, IDENTITY)
    assert (receipt["rounds"], receipt["rounds_allowed"], receipt["missing"]) == (2, 3, 0)
    digest = hashlib.sha256(label_path(run, JUDGE).read_bytes()).hexdigest()
    assert receipt["labels_sha256"] == digest
    assert "judge/wildguard.receipt.json" in rows[0]["files"]


@pytest.mark.usefixtures("published")
def test_rounds_stop_at_the_limit_and_report_what_stays_missing(tmp_path) -> None:
    run = make_run(tmp_path)
    fake = FakeWildGuard(["cannot classify"])
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    assert summaries == [{"run_id": "run", "items": 6, "unparseable": 6, "rounds": 3}]
    assert fake.calls == 18
    (row,) = read(run)
    assert (row["labeled"], row["missing"], row["harmful_rate"]) == (0, 6, None)
    # Fewer rounds than study 2 allows leave the missing labels pending, never "not harmful".
    short = make_run(tmp_path, name="short")
    label_runs([short], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=1)
    assert json.loads(receipt_path(short, JUDGE).read_text())["rounds"] == 1
    with pytest.raises(LabelsPending, match="6 items without a label after 1 labeling rounds"):
        read(short)
    # Resumed with the rounds left: two more, three in all.
    calls = fake.calls
    summaries = label_runs([short], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    assert summaries == [{"run_id": "short", "items": 6, "unparseable": 6, "rounds": 3}]
    assert fake.calls - calls == 12 and rounds_used(short) == 3
    (row,) = read(short)
    assert row["missing"] == 6


@pytest.mark.usefixtures("published")
def test_a_run_without_an_evaluation_is_skipped_not_failed(tmp_path) -> None:
    run = make_run(tmp_path, ())
    assert not (run / "evaluations.jsonl").exists()
    assert conditions(run) == []
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(FakeWildGuard([REFUSAL])))
    assert summaries == [{"run_id": "run", "skipped": "no evaluation recorded"}]
    assert read(run) == []
    assert not receipt_path(run, JUDGE).exists()


def test_a_published_run_is_labeled_whatever_its_status(tmp_path) -> None:
    """Whole records (as test_stage_a builds them): a run whose endpoint Rayleigh measurement
    failed after its terminal record was published has the status failed, and is labeled; a run
    whose records are not whole is not, whatever its status."""
    from test_stage_a import PROMPT_IDS, write_run

    prompts = {pid: f"Placeholder prompt {i}." for i, pid in enumerate(PROMPT_IDS)}
    runs = []
    for name, status in (("published", "failed"), ("broken", "completed")):
        run = write_run(tmp_path, name, arm="adamw_25", batch=5, lr=1e-5, seed=3000,
                        final_k=6.0, status=status)  # fmt: skip
        for file in ("terminal.json", "summary.json"):  # as schedule.run_scheduled writes them
            record = json.loads((run / file).read_text())
            (record.get("terminal") or record)["evaluations"] = [25]
            (run / file).write_text(json.dumps(record))
        cfg = OmegaConf.load(run / "config.yaml")
        OmegaConf.update(cfg, "eval.rayleigh", {"enabled": True}, force_add=True)
        OmegaConf.save(cfg, run / "config.yaml")
        runs.append(run)
    (runs[0] / "rayleigh.json").write_text(json.dumps(
        {"R": None, "reason": "failed: OutOfMemoryError: CUDA out of memory."}))  # fmt: skip
    metrics = runs[1] / "metrics.jsonl"
    metrics.write_text("".join(metrics.read_text().splitlines(keepends=True)[:-1]))
    fake = FakeWildGuard([REFUSAL])
    summaries = label_runs(runs, JUDGE, lambda cfg: prompts, factory(fake), rounds=3)
    assert summaries[0] == {"run_id": "published", "items": 3, "unparseable": 0}
    assert summaries[1]["run_id"] == "broken"
    assert "no published terminal record" in summaries[1]["skipped"]
    assert "one row per step" in summaries[1]["skipped"]
    assert fake.calls == 3 and not label_path(runs[1], JUDGE).exists()
    (row,) = read_endpoint_labels(runs[0], JUDGE, IDENTITY, prompts)
    assert (row["condition"], row["labeled"], row["refusal"], row["missing"]) == (
        "step-25", 3, 3, 0)  # fmt: skip


@pytest.mark.usefixtures("published")
def test_an_interrupted_labeling_stays_pending(tmp_path) -> None:
    run = make_run(tmp_path)
    gone = FakeWildGuard([REFUSAL], fail_after=3)
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(gone), rounds=3)
    assert "the judge server is gone" in summaries[0]["error"]
    assert 0 < len(label_path(run, JUDGE).read_text().splitlines()) < 6  # a partial label file
    assert not receipt_path(run, JUDGE).exists()
    with pytest.raises(LabelsPending, match="receipt.json is missing"):
        read(run)
    # Finished on a rerun, with the two rounds left (the interrupted one counts): the labels count.
    rest = FakeWildGuard(["cannot classify"])
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(rest), rounds=3)
    assert summaries == [{"run_id": "run", "items": 6, "unparseable": 3, "rounds": 3}]
    assert rest.attempts == 6  # the three items without a label, in rounds 2 and 3
    (row,) = read(run)
    assert (row["labeled"], row["missing"]) == (3, 3)
    # Labeling it again sends nothing: its receipt is complete, and it stays.
    again = FakeWildGuard([REFUSAL], fail_after=1)
    before = judge_files(run)
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(again), rounds=3)
    assert summaries[0]["kept"] and again.attempts == 0 and judge_files(run) == before
    (row,) = read(run)
    assert (row["labeled"], row["missing"]) == (3, 3)


@pytest.mark.usefixtures("published")
def test_the_rounds_hold_across_an_interrupted_labeling_and_its_rerun(tmp_path) -> None:
    run = make_run(tmp_path)

    def used() -> int | None:  # what the rounds record says as a request goes out
        path = rounds_path(run, JUDGE)
        return json.loads(path.read_text())["used"] if path.exists() else None

    # Round 1 does not parse; round 2 labels two items, then the judge server goes, mid-round.
    first = FakeWildGuard(["cannot classify"] * 6 + [REFUSAL] * 2, fail_after=8, watch=used)
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(first, workers=1),
                           rounds=3)  # fmt: skip
    assert "the judge server is gone" in summaries[0]["error"]
    assert first.calls == 8 and 9 <= first.attempts <= 12  # the workers stop on the failure
    assert first.seen[:6] == [1] * 6 and set(first.seen[6:]) == {2}  # counted before sending
    assert rounds_used(run) == 2 and not receipt_path(run, JUDGE).exists()
    # The rerun has one round left: it sends the four items still without a label, once.
    second = FakeWildGuard(["cannot classify"], watch=used)
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(second), rounds=3)
    assert summaries == [{"run_id": "run", "items": 6, "unparseable": 4, "rounds": 3}]
    assert second.attempts == 4 and second.seen == [3] * 4
    receipt = json.loads(receipt_path(run, JUDGE).read_text())
    assert (receipt["rounds"], receipt["rounds_allowed"], receipt["missing"]) == (3, 3, 4)
    assert rounds_used(run) == 3
    (row,) = read(run)
    assert (row["labeled"], row["refusal"], row["missing"]) == (2, 2, 4)


@pytest.mark.usefixtures("published")
@pytest.mark.parametrize(("reply", "rounds", "missing"),
                         [(REFUSAL, 1, 0), ("cannot classify", 3, 12)])  # fmt: skip
def test_a_rerun_after_a_completed_receipt_sends_nothing_and_changes_nothing(
    tmp_path, reply: str, rounds: int, missing: int
) -> None:
    """Every label there, or labels still missing once the three rounds are used."""
    run = make_run(tmp_path, (25, 50))
    label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(FakeWildGuard([reply])), rounds=3)
    receipt = json.loads(receipt_path(run, JUDGE).read_text())
    assert (receipt["rounds"], receipt["missing"]) == (rounds, missing)
    before = judge_files(run)
    assert set(before) == {"wildguard.jsonl", "wildguard.receipt.json", "wildguard.rounds.json",
                           "wildguard.lock"}  # fmt: skip
    assert json.loads(rounds_path(run, JUDGE).read_text())["certified_sha256"] == (
        receipt["labels_sha256"])  # fmt: skip
    for allowed in (1, 2, 3):
        again = FakeWildGuard([HARMFUL])
        (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(again), rounds=allowed)
        assert summary["kept"] and summary["unparseable"] == missing
        assert again.attempts == 0 and judge_files(run) == before


def state(run: Path) -> dict[str, bytes | None]:
    """The bytes of the run's label file and rounds record (None: absent)."""
    return {path.name: path.read_bytes() if path.exists() else None
            for path in (label_path(run, JUDGE), rounds_path(run, JUDGE))}  # fmt: skip


@pytest.mark.usefixtures("published")
def test_a_deleted_receipt_is_restored_and_does_not_reset_the_rounds(tmp_path) -> None:
    run = make_run(tmp_path)
    broken = FakeWildGuard(["cannot classify"])
    label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(broken), rounds=2)
    assert broken.attempts == 12 and rounds_used(run) == 2
    first = json.loads(receipt_path(run, JUDGE).read_text())
    # Deleted after its labeling completed, the label file unchanged: written again, nothing sent.
    receipt_path(run, JUDGE).unlink()
    before = state(run)
    again = FakeWildGuard(["cannot classify"])
    (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(again), rounds=3)
    assert summary["restored"] and (summary["rounds"], summary["unparseable"]) == (2, 6)
    assert again.attempts == 0 and state(run) == before
    restored = json.loads(receipt_path(run, JUDGE).read_text())
    assert {k: v for k, v in restored.items() if k not in ("what", "completed_at",
            "rounds_allowed")} == {k: v for k, v in first.items() if k not in (
            "what", "completed_at", "rounds_allowed")}  # fmt: skip
    # Resumed with the one round left of three, and only it is sent.
    summaries = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(again), rounds=3)
    assert summaries == [{"run_id": "run", "items": 6, "unparseable": 6, "rounds": 3}]
    assert again.attempts == 6 and rounds_used(run) == 3
    # Deleted once the rounds are used: restored, nothing sent.
    receipt_path(run, JUDGE).unlink()
    before = state(run)
    last = FakeWildGuard([REFUSAL])
    (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(last), rounds=3)
    assert summary["restored"] and summary["rounds"] == 3 and last.attempts == 0
    assert state(run) == before
    (row,) = read(run)
    assert row["missing"] == 6


@pytest.mark.usefixtures("published")
def test_labels_changed_outside_a_round_are_refused_and_left_as_they_are(tmp_path) -> None:
    """A receipt that no longer validates, whatever the rounds left, and a label file changed
    after its receipt was deleted: refused, nothing sent, label file and rounds record intact."""
    fake = FakeWildGuard([HARMFUL])

    def refused(run: Path, match: str, receipt: bool = True) -> None:
        before = state(run), receipt_path(run, JUDGE).read_bytes() if receipt else None
        (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
        assert match in summary["error"] and "never delete the receipt" in summary["error"]
        assert fake.attempts == 0
        after = state(run), receipt_path(run, JUDGE).read_bytes() if receipt else None
        assert after == before and receipt_path(run, JUDGE).exists() == receipt

    def labeled(name: str, reply: str, rounds: int) -> Path:
        run = make_run(tmp_path, name=name)
        label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(FakeWildGuard([reply])),
                   rounds=rounds)  # fmt: skip
        return run

    mismatch = "does not match the run's label file, judge, items or counts"
    # A label appended outside a round, one round of three used: not resumed.
    run = labeled("appended", "cannot classify", 1)
    with label_path(run, JUDGE).open("a") as handle:
        handle.write(json.dumps(record("step-25", "judged:0", "harmful_compliance")) + "\n")
    refused(run, mismatch)
    # The label file of a complete labeling lost: its endpoints are not judged again.
    run = labeled("lost", REFUSAL, 1)
    label_path(run, JUDGE).unlink()
    refused(run, mismatch)
    # The label file and the rounds record lost, the receipt there: no fresh start.
    run = labeled("both-lost", REFUSAL, 1)
    label_path(run, JUDGE).unlink()
    rounds_path(run, JUDGE).unlink()
    refused(run, mismatch)
    assert not label_path(run, JUDGE).exists() and not rounds_path(run, JUDGE).exists()
    # A receipt of another judge revision, of other items, or with bad counts.
    for name, change in (("identity", {"identity": {**IDENTITY, "judge_revision": "other"}}),
                         ("items", {"expected_items": [], "items": 0}),
                         ("counts", {"missing": None})):  # fmt: skip
        run = labeled(name, REFUSAL, 1)
        path = receipt_path(run, JUDGE)
        path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
        refused(run, mismatch)
    # A label changed and a receipt written for it by hand: not the receipt the record certifies.
    run = labeled("handmade", REFUSAL, 3)
    lines = label_path(run, JUDGE).read_text().splitlines()
    edited = {**json.loads(lines[0]), "label": "harmful_compliance"}
    label_path(run, JUDGE).write_text("\n".join([json.dumps(edited), *lines[1:]]) + "\n")
    write_receipt(run, rounds=1)
    assert read(run)[0]["harmful"] == 1  # the analysis alone would count it
    refused(run, "is not the receipt its rounds record")
    # The receipt deleted, then a label changed: the rounds record names the certified file.
    run = labeled("edited", REFUSAL, 3)
    receipt_path(run, JUDGE).unlink()
    lines = label_path(run, JUDGE).read_text().splitlines()
    edited = {**json.loads(lines[0]), "label": "harmful_compliance"}
    label_path(run, JUDGE).write_text("\n".join([json.dumps(edited), *lines[1:]]) + "\n")
    refused(run, "changed since its labeling completed", receipt=False)
    assert not receipt_path(run, JUDGE).exists()
    with pytest.raises(LabelsPending, match="receipt.json is missing"):
        read(run)


@pytest.mark.usefixtures("published")
def test_no_receipt_counts_a_file_the_analysis_would_refuse(tmp_path) -> None:
    # Three rounds, each failing after one reply: expected items without any record.
    run = make_run(tmp_path)
    for used in (1, 2, 3):
        failing = FakeWildGuard([REFUSAL], fail_after=1)
        (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(failing, workers=1),
                                rounds=3)  # fmt: skip
        assert "RoundFailed" in summary["error"] and rounds_used(run) == used
    before = state(run)
    healthy = FakeWildGuard([REFUSAL])
    (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(healthy), rounds=3)
    assert "has no record of 3 expected items" in summary["error"]
    assert "decide the run by hand" in summary["error"]
    assert healthy.attempts == 0 and state(run) == before and not receipt_path(run, JUDGE).exists()
    # A valid receipt of a partial file is not reported as kept either.
    partial = make_run(tmp_path, name="partial")
    write_labels(partial, [record("step-25", pid, "refusal") for pid in IDS[:4]])
    with pytest.raises(LabelsPending, match="no record of 2 expected items"):
        read(partial)
    before = judge_files(partial)
    (summary,) = label_runs([partial], JUDGE, lambda cfg: PROMPTS, factory(healthy), rounds=3)
    assert "has no record of 2 expected items" in summary["error"] and "kept" not in summary
    assert healthy.attempts == 0
    assert judge_files(partial) == {**before, "wildguard.lock": hashlib.sha256(b"").hexdigest()}


@pytest.mark.usefixtures("published")
def test_a_failed_round_stops_the_invocation(tmp_path) -> None:
    runs = [make_run(tmp_path, name=f"r{i}") for i in range(3)]

    def records() -> list[bool]:  # which runs have a rounds record as a request goes out
        return [rounds_path(run, JUDGE).exists() for run in runs]

    gone = FakeWildGuard([REFUSAL], fail_after=2, watch=records)
    summaries = label_runs(runs, JUDGE, lambda cfg: PROMPTS, factory(gone, workers=1), rounds=3)
    assert "RoundFailed: labeling round 1 of 3 failed" in summaries[0]["error"]
    assert summaries[1:] == [{"run_id": "r1", "skipped": NOT_ATTEMPTED},
                             {"run_id": "r2", "skipped": NOT_ATTEMPTED}]  # fmt: skip
    assert gone.calls == 2 and all(seen == [True, False, False] for seen in gone.seen)
    assert rounds_used(runs[0]) == 1
    for run in runs[1:]:
        assert not (run / "judge").exists()
    # Back with a healthy judge: run 0 resumes with its rounds left, the others start at 1.
    summaries = label_runs(runs, JUDGE, lambda cfg: PROMPTS, factory(FakeWildGuard([REFUSAL])),
                           rounds=3)  # fmt: skip
    assert [s.get("rounds", 1) for s in summaries] == [2, 1, 1]
    assert all(s["unparseable"] == 0 for s in summaries)


@pytest.mark.usefixtures("published")
def test_a_run_another_process_is_labeling_is_busy(tmp_path) -> None:
    run = make_run(tmp_path)
    (run / "judge").mkdir()
    fake = FakeWildGuard([REFUSAL])
    with lock_path(run, JUDGE).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
        fcntl.flock(handle, fcntl.LOCK_UN)
    assert "in progress" in summary["busy"] and fake.attempts == 0
    assert state(run) == {"wildguard.jsonl": None, "wildguard.rounds.json": None}
    (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    assert (summary["unparseable"], fake.attempts) == (0, 6)


@pytest.mark.usefixtures("published")
def test_a_label_file_without_a_valid_rounds_record_is_refused(tmp_path) -> None:
    run = make_run(tmp_path)
    label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(FakeWildGuard(["cannot classify"])),
               rounds=1)  # fmt: skip
    record_path = rounds_path(run, JUDGE)
    kept = record_path.read_text()
    fake = FakeWildGuard([REFUSAL])
    # As a label file written before the record existed: the rounds behind it are unknown.
    record_path.unlink()
    (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    assert "has no rounds record" in summary["error"] and "none is sent" in summary["error"]
    record_path.write_text("{")
    (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    assert "is not a rounds record" in summary["error"]
    other = json.loads(kept)
    other["identity"] = {**other["identity"], "judge_revision": "other"}
    record_path.write_text(json.dumps(other))
    (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    assert "counts the rounds of another judge" in summary["error"]
    older = {k: v for k, v in json.loads(kept).items() if k != "certified_sha256"}
    record_path.write_text(json.dumps(older))  # as the record of 7dc9c7e, without it
    (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    assert "is not a rounds record" in summary["error"]
    assert fake.attempts == 0
    # A complete receipt is kept even so: nothing is sent, nothing is written.
    record_path.unlink()
    done = make_run(tmp_path, name="done")
    label_runs([done], JUDGE, lambda cfg: PROMPTS, factory(FakeWildGuard([REFUSAL])), rounds=3)
    rounds_path(done, JUDGE).unlink()
    before = judge_files(done)
    (summary,) = label_runs([done], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    assert summary["kept"] and fake.attempts == 0 and judge_files(done) == before
    for rounds in (0, LABEL_ROUNDS + 1):
        with pytest.raises(ValueError, match="rounds must be 1 to 3"):
            label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=rounds)


# --- reader ------------------------------------------------------------------------------------


def test_counts_with_bounds_and_file_hashes(tmp_path) -> None:
    run = make_run(tmp_path, (25, 50))
    labels = ["harmful_compliance", "harmful_compliance", "refusal", "safe_non_refusal", None,
              None]  # fmt: skip
    records = [record("step-25", pid, label) for pid, label in zip(IDS, labels, strict=True)]
    records.append(record("step-25", "judged:4", "harmful_compliance"))  # null, then labeled
    records.append(record("step-25", "judged:0", "harmful_compliance"))  # the same label twice
    records += [record("step-50", pid, None) for pid in IDS]  # all null: nothing labeled
    write_labels(run, records)
    first, second = read(run)
    assert (first["condition"], first["step"], first["prompts"]) == ("step-25", 25, 6)
    assert (first["labeled"], first["missing"]) == (5, 1)  # judged:5 did not parse
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
                                   "judge/wildguard.jsonl",
                                   "judge/wildguard.receipt.json"}  # fmt: skip


def test_no_label_counts_without_a_completed_receipt(tmp_path) -> None:
    run = make_run(tmp_path)
    write_labels(run, [record("step-25", pid, "harmful_compliance") for pid in IDS],
                 receipt=False)  # fmt: skip
    with pytest.raises(LabelsPending, match="receipt.json is missing: the labeling"):
        read(run)
    receipt_path(run, JUDGE).write_text("{")
    with pytest.raises(LabelsPending, match="not a JSON judging receipt"):
        read(run)
    for changes, message in (
        ({"judge": "rubric"}, "receipt of another judge"),
        ({"identity": {**IDENTITY, "judge_revision": "other"}}, "receipt of another judge"),
        ({"expected_items": expected(run)[:5]}, "does not expect the run's 6 items"),
        ({"items": 5}, "does not expect the run's 6 items"),
        ({"rounds": 0}, "lacks its rounds"),
        ({"missing": None}, "lacks its rounds"),
        ({"labels_sha256": "0" * 64}, "not the label file its judging receipt was written for"),
        ({"missing": 1}, "records 1 items without a label, the file 0"),
    ):  # fmt: skip
        write_receipt(run, **changes)
        with pytest.raises(LabelsPending, match=message):
            read(run)
    write_receipt(run)
    assert read(run)[0]["harmful"] == 6
    # A label file changed after its receipt (a record appended) is pending again.
    with label_path(run, JUDGE).open("a") as handle:
        handle.write(json.dumps(record("step-25", "judged:0", "harmful_compliance")) + "\n")
    with pytest.raises(LabelsPending, match="not the label file"):
        read(run)


def test_an_empty_or_partial_label_file_never_reaches_the_analysis(tmp_path) -> None:
    run = make_run(tmp_path)
    write_labels(run, [])  # even with a receipt of the empty file
    with pytest.raises(LabelsPending, match="no record of 6 expected items .first: "
                                            "step-25/judged:0.: an empty or partial"):  # fmt: skip
        read(run)
    # A partial file, even with a receipt that matches it, and with one that does not.
    write_labels(run, [record("step-25", pid, "refusal") for pid in IDS[:4]])
    with pytest.raises(LabelsPending, match="no record of 2 expected items .first: "
                                            "step-25/judged:4."):  # fmt: skip
        read(run)
    write_labels(run, [record("step-25", pid, "refusal") for pid in IDS])
    write_jsonl(label_path(run, JUDGE), [record("step-25", pid, "refusal") for pid in IDS[:4]])
    with pytest.raises(LabelsPending, match="not the label file"):
        read(run)


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


@pytest.mark.parametrize(
    "field", ["backend", "prompt_version", "judge_model", "judge_revision", "rubric_version"]
)
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


def test_malformed_label_lines_fail(tmp_path) -> None:
    run = make_run(tmp_path)
    write_raw(run, "{not json\n")
    with pytest.raises(ValueError, match="line 1, is not valid JSON"):
        read(run)
    write_raw(run, json.dumps({"label": "refusal"}) + "\n")
    with pytest.raises(ValueError, match="line 1, names no item"):
        read(run)
    no_label = record("step-25", "judged:1", "refusal")
    del no_label["label"]
    write_labels(run, [no_label])
    with pytest.raises(ValueError, match="record of step-25/judged:1 has no label field"):
        read(run)


def test_a_record_naming_no_response_fails(tmp_path) -> None:
    run = make_run(tmp_path)
    write_labels(run, [record("step-50", "judged:1", "refusal")])
    with pytest.raises(ValueError, match="step-50/judged:1 names no response"):
        read(run)


def test_an_unlabeled_run_and_a_non_study2_run_are_refused(tmp_path) -> None:
    run = make_run(tmp_path)
    with pytest.raises(LabelsPending, match="receipt.json is missing"):
        read(run)
    write_labels(run, [record("step-25", pid, "refusal") for pid in IDS])
    label_path(run, JUDGE).unlink()  # a receipt without its label file
    with pytest.raises(LabelsPending, match="wildguard.jsonl is missing"):
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


def test_a_two_class_judge_gives_harmful_and_not_harmful_only(tmp_path) -> None:
    run = make_run(tmp_path)
    guard = judge_identity(load_config("judge/llama_guard"))
    labels = ["harmful_compliance"] * 2 + ["safe_non_refusal"] * 4
    records = [record("step-25", pid, label, **guard) for pid, label in zip(IDS, labels)]
    path = label_path(run, "llama_guard")
    path.parent.mkdir(exist_ok=True)
    write_jsonl(path, records)
    items = expected(run)
    receipt_path(run, "llama_guard").write_text(json.dumps({
        "run_id": run.name, "judge": "llama_guard", "identity": guard, "expected_items": items,
        "items": len(items), "rounds": 1, "rounds_allowed": LABEL_ROUNDS, "missing": 0,
        "labels_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))  # fmt: skip
    (row,) = read_endpoint_labels(run, "llama_guard", guard, PROMPTS)
    assert "refusal" not in row and "safe_non_refusal" not in row
    assert (row["labeled"], row["harmful"], row["not_harmful"]) == (6, 2, 4)
    assert row["harmful_rate"] == pytest.approx(2 / 6)


def test_the_wildguard_rows_keep_their_three_class_fields(tmp_path) -> None:
    run = make_run(tmp_path)
    labels = ["harmful_compliance", "refusal", "refusal", "safe_non_refusal"] + ["refusal"] * 2
    write_labels(run, [record("step-25", pid, label) for pid, label in zip(IDS, labels)])
    (row,) = read(run)
    assert (row["harmful"], row["refusal"], row["safe_non_refusal"]) == (1, 4, 1)
    assert "not_harmful" not in row
