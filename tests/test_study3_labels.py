"""Study 3's label reader (analysis/study3_labels.py): the labeling ledger, the two points where
harm leaves sealed storage, and the one reader of a run's harm and of its items.

The runs are tests/test_study2_labels.py's (six benign placeholder prompts at step 25), placed
under ``<runs root>/study3/`` and labeled through ``run_judging.label_runs`` against a fake judge
server that answers by content (``ContentJudge``). Nothing reads a runs/ folder of the checkout."""

import hashlib
import json
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_study3_package2
from test_study2_labels import IDENTITY, IDS, JUDGE, PROMPTS, factory, make_run
from test_study3_package2 import SealedGuard
from test_study3_progress import Study

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import study3_labels, study3_progress, study3_versions
from zero_field_refusal.analysis.evidence import label_path
from zero_field_refusal.analysis.study2_label_cli import LABELED
from zero_field_refusal.analysis.study3_labels import (
    DIAGNOSIS_KIND,
    ITEM_LABELS,
    LabelingProblem,
    LabelsPending,
    LedgerError,
    RunHarm,
    SealedPoint,
    append_event,
    diagnoses,
    harm_many,
    labeling_problems,
    ledger_dir,
    ledger_lock,
    ledger_runs,
    open_point,
    read_ledger,
    resolve_torn,
    run_harm,
    run_items,
    torn_tail,
)
from zero_field_refusal.run_judging import label_runs, receipt_path

HARMFUL_REPLY = "Harmful request: yes\nResponse refusal: no\nHarmful response: yes"
REFUSAL_REPLY = "Harmful request: yes\nResponse refusal: yes\nHarmful response: no"
GARBLED = "cannot classify"
WHEN = "2026-10-31T12:00:00+00:00"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContentJudge:
    """The parts of ``openai.OpenAI`` the WildGuard backend uses, answering by content: HARMFUL
    when the WildGuard prompt holds ``[h]``, an unparseable text when it holds ``[x]``, REFUSAL
    otherwise. ``attempts`` counts the requests."""

    def __init__(self) -> None:
        self.attempts, self.lock = 0, threading.Lock()
        self.completions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        prompt = kwargs["prompt"]
        with self.lock:
            self.attempts += 1
        text = GARBLED if "[x]" in prompt else HARMFUL_REPLY if "[h]" in prompt else REFUSAL_REPLY
        return SimpleNamespace(choices=[SimpleNamespace(text=text, finish_reason="stop")])


def study3_run(root: Path, name: str, harmful=(0, 1), unparseable=(2,)) -> Path:
    """test_study2_labels' run at ``<root>/study3/<name>``, its responses to the prompts at the
    positions ``harmful`` marked ``[h]`` and those at ``unparseable`` marked ``[x]``."""
    run = make_run(root / "study3", name=name)
    path = run / "responses" / "harmful_evaluation.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        at = IDS.index(row["item_id"])
        row["text"] += " [h]" if at in harmful else " [x]" if at in unparseable else ""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return run


def label(run: Path, rounds: int) -> None:
    label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(ContentJudge()), rounds=rounds)


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    """The runs root: six prompts per run (``HARMFUL_PROMPTS``), their texts ``PROMPTS``, and the
    runs taken as published (``stage_a.load_trajectory``)."""
    monkeypatch.setattr("zero_field_refusal.run_judging.load_trajectory", lambda run_dir: None)
    monkeypatch.setattr(study3_labels, "HARMFUL_PROMPTS", len(IDS))
    monkeypatch.setattr(study3_labels, "harmful_prompts", lambda cfg: PROMPTS)
    path = tmp_path / "runs"
    (path / "study3").mkdir(parents=True)
    return path


@pytest.fixture
def labeled_run(root) -> Path:
    """A run labeled in three rounds: two harmful items, one unparseable, three refusals."""
    run = study3_run(root, "r-labeled")
    label(run, 3)
    return run


@pytest.fixture
def opened(monkeypatch):
    """``opened(point, ids)``: ``open_point`` opens ``point`` on ``ids`` and refuses the other."""

    def open_(point: str, ids) -> None:
        def fake(name, *, candidate, runs_root, repo_root):
            if name != point:
                raise SealedPoint(f"the {name} point is closed here")
            return frozenset(ids)

        monkeypatch.setattr(study3_labels, "open_point", fake)

    return open_


def ledger_guard(monkeypatch, root: Path) -> SealedGuard:
    """``SealedGuard`` with the ledger readable: a run's judge outputs and their superseded copies
    stay sealed."""
    monkeypatch.setattr(test_study3_package2, "UNDER_RUNS", ("judge", "judge-superseded"))
    return SealedGuard(monkeypatch, runs_root=root)


# ------------------------------------------------------------------ journal lines


def generation_line(run: str, number: int = 1, cap: int = 3, batch: str = "b0001",
                    why: str | None = None) -> dict:  # fmt: skip
    return {"event": "generation", "batch": batch, "run": run, "generation": number, "cap": cap,
            "why": why if why or number == 1 else "control_change"}  # fmt: skip


def open_generation(root: Path, run: str, number: int, cap: int, batch: str = "b0001") -> None:
    append_event(root, generation_line(run, number, cap, batch))


def superseded(batch: str, run: str, number: int, rounds_used: int, k: int = 1,
               reason: str = "control_change") -> dict:  # fmt: skip
    target = f"judge-superseded/{batch}/g{number}"
    target += "-unverified" if reason == "unverified" else ""
    return {"event": "superseded", "batch": batch, "k": k, "run": run, "generation": number,
            "reason": reason, "rounds_used": rounds_used, "target": target,
            "moved": []}  # fmt: skip


def mark(batch: str, candidate: int, why: str, runs: list[str]) -> dict:
    second = ["c000000000005"] if why == "second_change" else None
    return {"event": "mark", "batch": batch, "candidate": candidate, "why": why,
            "changed": {"1": ["c000000000003"], "2": second}, "runs": runs}  # fmt: skip


def diagnosis_record(batch: str, cause: str, diagnosis: str) -> dict:
    return {"kind": DIAGNOSIS_KIND, "batch": batch, "written_at": WHEN, "cause": cause,
            "diagnosis": diagnosis, "evidence": []}  # fmt: skip


def publish(root: Path, batch: str, record: dict) -> Path:
    """The diagnosis record of ``batch``, published in the ledger's folder."""
    path = ledger_dir(root) / f"diagnosis-{batch}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    return path


def diagnosed(root: Path, batch: str, cause: str, diagnosis: str = "placeholder line") -> dict:
    """The ``diagnosed`` line of a stub record, published first."""
    path = publish(root, batch, diagnosis_record(batch, cause, diagnosis))
    return {"event": "diagnosed", "batch": batch, "cause": cause,
            "record": {"path": f"study3/labeling/{path.name}", "sha256": sha256(path)}}  # fmt: skip


def write_stub_diagnosis(root: Path, batch: str, cause: str, diagnosis: str) -> Path:
    """A mark of ``batch`` and its diagnosis: the record published, then its line."""
    append_event(root, mark(batch, 1, "second_change", []))
    append_event(root, diagnosed(root, batch, cause, diagnosis))
    return ledger_dir(root) / f"diagnosis-{batch}.json"


def accept(root, run, generation=1, batch="b0001", k=1, attempt=1):
    receipt = sha256(receipt_path(run, JUDGE))
    append_event(root, {"event": "labeled", "batch": batch, "k": k, "attempt": attempt, "runs": [
        {"run": run.name, "status": LABELED, "receipt_sha256": receipt}]})  # fmt: skip
    append_event(root, {"event": "controls", "batch": batch, "k": k, "attempt": attempt,
                        "accepted": True, "changed": [], "parsed_changed": [],
                        "cache": {"path": "c", "sha256": "0" * 64}, "covers": [{
                            "run": run.name, "generation": generation, "receipt_sha256": receipt,
                            "labels_sha256": sha256(label_path(run, JUDGE))}]})  # fmt: skip


def generation_run(root: Path, name: str, *, generation: int, cap: int, rounds: int,
                   unparseable: int) -> Path:  # fmt: skip
    """A run whose ``generation`` is journaled open at ``cap`` (each earlier one used one round
    and was superseded), labeled with ``rounds`` rounds, ``unparseable`` items never parsing."""
    run = study3_run(root, name, unparseable=tuple(range(2, 2 + unparseable)))
    for number in range(1, generation):
        open_generation(root, name, number, cap=3 - (number - 1))
        append_event(root, superseded("b0001", name, number, rounds_used=1))
    open_generation(root, name, generation, cap=cap)
    label(run, rounds)
    return run


def journal(root: Path) -> Path:
    return ledger_dir(root) / "journal.jsonl"


# ------------------------------------------------------------------ the reader's gates


def test_a_receipt_alone_is_not_enough(labeled_run, root, opened):
    opened("spread", {labeled_run.name})
    open_generation(root, labeled_run.name, 1, cap=3)
    with pytest.raises(LabelsPending):
        run_harm(labeled_run, point="spread", candidate=1, runs_root=root, repo_root=root)
    accept(root, labeled_run)
    assert run_harm(labeled_run, point="spread", candidate=1, runs_root=root,
                    repo_root=root) == RunHarm(n=6, harmful=2, missing=1,
                    receipt_sha256=sha256(receipt_path(labeled_run, JUDGE)))  # fmt: skip


def test_exhaustion_after_a_rejection_is_a_technical_problem(labeled_run, root, opened):
    opened("spread", {labeled_run.name})
    open_generation(root, labeled_run.name, 1, cap=3)
    append_event(root, superseded("b0001", labeled_run.name, 1, rounds_used=3))
    with pytest.raises(LabelingProblem, match="rounds"):
        run_harm(labeled_run, point="spread", candidate=1, runs_root=root, repo_root=root)


def test_a_later_generation_reads_its_rounds_against_its_cap(root, opened):
    done = generation_run(root, "r1", generation=2, cap=2, rounds=2, unparseable=1)
    short = generation_run(root, "r2", generation=2, cap=2, rounds=1, unparseable=1)
    opened("spread", {"r1", "r2"})
    accept(root, done, 2)
    accept(root, short, 2, "b0002")
    assert run_harm(done, point="spread", candidate=1, runs_root=root, repo_root=root).missing == 1
    with pytest.raises(LabelsPending, match="cap"):
        run_harm(short, point="spread", candidate=1, runs_root=root, repo_root=root)


def test_outside_the_cohort_nothing_sealed_is_opened(labeled_run, root, opened, monkeypatch):
    opened("spread", {"another-run"})
    guard = SealedGuard(monkeypatch, runs_root=root)
    with pytest.raises(SealedPoint, match=labeled_run.name):
        run_harm(labeled_run, point="spread", candidate=1, runs_root=root, repo_root=root)
    assert guard.attempts == [] and not [p for p in guard.opened if "/labeling/" in p]


@pytest.mark.parametrize("stage,step,problems,is_open", [
    ("validation", {"kind": "campaign"}, [], False), ("choice", {"record": "n-c2"}, [], False),
    ("choice", {"record": "n-c1"}, ["x"], False), ("budget", {"record": "budget-c1"}, [], False),
    ("choice", {"record": "n-c1", "fields": {"cohort": {"run_ids": ["r1"]}}}, [], True),
])  # fmt: skip
def test_the_spread_point_follows_the_choice_step(stage, step, problems, is_open, root,
                                                  monkeypatch):  # fmt: skip
    monkeypatch.setattr(study3_labels, "_state", lambda *a, **k: {
        "stage": stage, "candidate": 1, "step": step, "problems": problems})  # fmt: skip
    if is_open:
        assert open_point("spread", candidate=1, runs_root=root, repo_root=root) == {"r1"}
    else:
        with pytest.raises(SealedPoint, match="choice step"):
            open_point("spread", candidate=1, runs_root=root, repo_root=root)


def test_the_spread_point_opens_on_the_cores_choice_step(root, monkeypatch):
    """The spread window against the real core (``study3_progress.advance`` over a synthetic
    study): open at candidate 1's ``n-c1`` step on exactly that step's cohort, closed once the n
    record is written."""
    study = Study()
    study.through_validation()
    state = study.advance()
    assert (state["stage"], state["step"]["record"]) == ("choice", "n-c1")
    cohort = frozenset(state["step"]["fields"]["cohort"]["run_ids"])
    monkeypatch.setattr(study3_labels, "_state", lambda *a, **k: study.advance())
    assert cohort and open_point("spread", candidate=1, runs_root=root, repo_root=root) == cohort
    with pytest.raises(SealedPoint, match="choice step"):
        open_point("spread", candidate=2, runs_root=root, repo_root=root)
    study.n(1)
    with pytest.raises(SealedPoint, match="choice step"):
        open_point("spread", candidate=1, runs_root=root, repo_root=root)


def test_the_state_is_progress_under_the_checkouts_gpu_code(tmp_path, monkeypatch):
    """``_state`` is one ``progress`` call: the runs root, the checkout as ``repo`` and the GPU
    code identity of that checkout as the expected code, θ₀ and the settings at their defaults."""
    calls = []

    def identity(**kw):
        calls.append(("code", kw))
        return {"sha256": "c" * 64}

    def progress(*args, **kw):
        calls.append(("progress", args, kw))
        return {"stage": "x"}

    monkeypatch.setattr(code_identity, "gpu_code_identity", identity)
    monkeypatch.setattr(study3_progress, "progress", progress)
    runs, repo = tmp_path / "runs", tmp_path / "repo"
    assert study3_labels._state(runs, repo) == {"stage": "x"}
    assert calls == [("code", {"repo": repo}),
                     ("progress", (runs,), {"repo": repo, "expected_code": "c" * 64})]  # fmt: skip


def test_the_judge_config_is_the_frozen_one():
    assert study3_labels.JUDGE_CONFIG == study3_versions.JUDGE_CONFIG


def test_a_cohort_is_gated_by_one_reading_of_the_state(root, monkeypatch):
    """``harm_many`` opens its point once for the whole list: one ``progress`` reading, whatever
    the number of runs (a cohort's gate is never one reading per run)."""
    runs = [generation_run(root, name, generation=1, cap=3, rounds=3, unparseable=0)
            for name in ("r1", "r2", "r3")]  # fmt: skip
    for run in runs:
        accept(root, run, 1, f"b000{runs.index(run) + 1}")
    calls = []

    def state(*args, **kwargs):
        calls.append(args)
        return {
            "stage": "choice",
            "candidate": 1,
            "problems": [],
            "step": {"record": "n-c1", "fields": {"cohort": {"run_ids": ["r1", "r2", "r3"]}}},
        }

    monkeypatch.setattr(study3_labels, "_state", state)
    found = harm_many(runs, point="spread", candidate=1, runs_root=root, repo_root=root)
    assert [h.harmful for h in found] == [2, 2, 2] and len(calls) == 1


def test_run_items_has_run_harms_gates_and_counts(labeled_run, root, opened):
    opened("spread", {labeled_run.name})
    open_generation(root, labeled_run.name, 1, cap=3)
    with pytest.raises(LabelsPending):
        run_items(labeled_run, point="spread", candidate=1, runs_root=root, repo_root=root)
    accept(root, labeled_run)
    items = run_items(labeled_run, point="spread", candidate=1, runs_root=root, repo_root=root)
    harm = run_harm(labeled_run, point="spread", candidate=1, runs_root=root, repo_root=root)
    assert [i.index for i in items] == list(range(6)) and [i.prompt_id for i in items] == IDS
    assert Counter(i.label for i in items) == {
        "harmful": harm.harmful,
        "missing": harm.missing,
        "not_harmful": 6 - harm.harmful - harm.missing,
    }
    assert {i.label for i in items} <= set(ITEM_LABELS)
    assert [i.label for i in items][:3] == ["harmful", "harmful", "missing"]


def test_a_run_folder_must_be_a_plain_child_of_the_experiment(labeled_run, root, opened, tmp_path):
    opened("spread", {labeled_run.name, "elsewhere", "labeling"})
    open_generation(root, labeled_run.name, 1, cap=3)
    accept(root, labeled_run)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.symlink_to(labeled_run)
    linked = root / "study3" / "linked"
    linked.symlink_to(labeled_run)
    for run_dir in (elsewhere, root / "study3" / "labeling", linked):
        with pytest.raises(ValueError, match="plain child"):
            run_harm(run_dir, point="spread", candidate=1, runs_root=root, repo_root=root)
    with pytest.raises(ValueError, match="point"):
        run_harm(labeled_run, point="other", candidate=1, runs_root=root, repo_root=root)


def test_ledger_runs_and_labeling_problems_read_ids_only(labeled_run, root, monkeypatch):
    open_generation(root, "r1", 1, cap=3)
    open_generation(root, "r2", 1, cap=3)
    append_event(root, superseded("b0001", "r2", 1, rounds_used=3))
    guard = ledger_guard(monkeypatch, root)  # the ledger read through read_ledger
    assert ledger_runs(root) == {"r1", "r2"} and labeling_problems(root) == ("r2",)
    assert not [p for p in guard.opened if "/judge" in p]


def test_the_final_point_is_closed_until_the_freeze_gate(root):
    with pytest.raises(SealedPoint, match="freeze"):
        open_point("final", candidate=1, runs_root=root, repo_root=root)


def test_a_receipt_changed_after_its_cover_is_pending(labeled_run, root, opened):
    opened("spread", {labeled_run.name})
    open_generation(root, labeled_run.name, 1, cap=3)
    accept(root, labeled_run)
    path = receipt_path(labeled_run, JUDGE)
    path.write_text(path.read_text() + " ")
    with pytest.raises(LabelsPending, match="receipt"):
        run_harm(labeled_run, point="spread", candidate=1, runs_root=root, repo_root=root)


def test_a_label_file_changed_under_its_receipt_is_pending(labeled_run, root, opened):
    opened("spread", {labeled_run.name})
    open_generation(root, labeled_run.name, 1, cap=3)
    accept(root, labeled_run)
    path = label_path(labeled_run, JUDGE)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(LabelsPending):
        run_harm(labeled_run, point="spread", candidate=1, runs_root=root, repo_root=root)


# ------------------------------------------------------------------ the ledger's fold


def test_the_ledger_fold_counts_rounds_across_generations(root):
    open_generation(root, "r1", 1, cap=3)
    append_event(root, superseded("b0001", "r1", 1, rounds_used=1))
    ledger = read_ledger(root)
    assert (ledger.spent("r1"), ledger.remaining("r1"), ledger.current("r1")) == (1, 2, None)
    assert ledger.problem("r1") is None
    with pytest.raises(LedgerError, match="cap"):  # a cap is always the rounds left
        open_generation(root, "r1", 2, cap=3)
    open_generation(root, "r1", 2, cap=2)
    current = read_ledger(root).current("r1")
    assert (current.number, current.cap, current.why) == (2, 2, "control_change")
    assert (read_ledger(root).spent("r1"), read_ledger(root).remaining("r1")) == (1, 2)


def test_an_infrastructure_diagnosis_leaves_the_rounds_used_unchanged(root):
    for g, used in ((1, 1), (2, 1)):
        open_generation(root, "r1", g, cap=3 - (g - 1))
        append_event(root, superseded("b0001", "r1", g, used))
    append_event(root, mark("b0001", candidate=1, why="second_change", runs=[]))
    before = read_ledger(root)
    assert before.stopped(1) == "b0001" and (before.spent("r1"), before.remaining("r1")) == (2, 1)
    assert before.problem("r1") is not None and before.stopped(2) is None
    append_event(root, diagnosed(root, "b0001", "infrastructure"))
    after = read_ledger(root)
    assert after.stopped(1) is None and (after.spent("r1"), after.remaining("r1")) == (2, 1)
    assert after.problem("r1") is None and after.unresolved_marks == ()


def test_an_exhausted_run_stays_a_problem_whatever_the_diagnosis(root):
    for g, used in ((1, 2), (2, 1)):
        open_generation(root, "r1", g, cap=3 - (g - 1) - (1 if g == 2 else 0))
        append_event(root, superseded("b0001", "r1", g, used))
    open_generation(root, "r2", 1, cap=3)
    append_event(root, superseded("b0001", "r2", 1, 1))
    append_event(root, mark("b0001", candidate=1, why="second_change", runs=[]))
    append_event(root, diagnosed(root, "b0001", "infrastructure"))
    ledger = read_ledger(root)
    assert ledger.stopped(1) is None and ledger.remaining("r1") == 0
    assert "rounds" in ledger.problem("r1") and ledger.problem("r2") is None
    assert labeling_problems(root) == ("r1",)


def test_the_fold_refuses_infrastructure_for_an_exhausted_mark_or_a_parsed_change(root):
    open_generation(root, "r1", 1, cap=3)
    append_event(root, superseded("b0001", "r1", 1, 3))
    append_event(root, mark("b0001", candidate=1, why="exhausted", runs=["r1"]))
    with pytest.raises(LedgerError, match="exhausted"):
        append_event(root, diagnosed(root, "b0001", "infrastructure"))
    open_generation(root, "r2", 1, cap=3, batch="b0002")
    append_event(root, {"event": "labeled", "batch": "b0002", "k": 1, "attempt": 1, "runs": []})
    append_event(root, {"event": "controls", "batch": "b0002", "k": 1, "attempt": 1,
                        "accepted": False, "changed": ["c000000000003"],
                        "parsed_changed": ["c000000000003"],
                        "cache": {"path": "c", "sha256": "0" * 64}})  # fmt: skip
    append_event(root, superseded("b0002", "r2", 1, 1))
    append_event(root, mark("b0002", candidate=2, why="second_change", runs=[]))
    with pytest.raises(LedgerError, match="control"):
        append_event(root, diagnosed(root, "b0002", "infrastructure"))


def test_a_judge_diagnosis_leaves_the_mark_unresolved(root):
    open_generation(root, "r1", 1, cap=3)
    append_event(root, superseded("b0001", "r1", 1, 1))
    append_event(root, mark("b0001", candidate=1, why="second_change", runs=[]))
    line = diagnosed(root, "b0001", "judge")
    append_event(root, line)
    ledger = read_ledger(root)
    assert ledger.stopped(1) == "b0001" and ledger.unresolved_marks == ("b0001",)
    assert labeling_problems(root) == ("r1",) and "b0001" in ledger.problem("r1")
    with pytest.raises(LedgerError, match="second"):  # a diagnosis is written once
        append_event(root, {**line, "cause": "infrastructure"})


def test_a_torn_tail_refuses_until_resolve_torn_repairs_it(root):
    open_generation(root, "r1", 1, cap=3)
    with open(ledger_dir(root) / "journal.jsonl", "ab") as f:
        f.write(b'{"event": "controls", "batch": "b0001", "k": 1, "acc')  # a crash mid-line
    for call in (lambda: read_ledger(root), lambda: append_event(root, generation_line("r2"))):
        with pytest.raises(LedgerError, match="torn"):
            call()
    tail = torn_tail(root)
    repaired = resolve_torn(root)
    assert repaired == tail and repaired["kind"] == "controls" and torn_tail(root) is None
    ledger = read_ledger(root)  # the voided decision is not in the fold
    assert ledger.current("r1").number == 1 and ledger.covered == {}
    append_event(root, generation_line("r2"))
    assert read_ledger(root).current("r2") is not None
    with pytest.raises(LedgerError, match="not torn"):
        resolve_torn(root)


def test_an_unnamed_bad_line_in_the_middle_fails_closed(root):
    open_generation(root, "r1", 1, cap=3)
    with open(journal(root), "ab") as f:
        f.write(b'{"event": "generation", "PLACEHOLDER-BYTES": 1}\n')
    open_bytes = journal(root).read_bytes()
    with open(journal(root), "ab") as f:
        f.write(json.dumps({**generation_line("r2"), "time": WHEN}).encode() + b"\n")
    with pytest.raises(LedgerError, match="line 2") as caught:
        read_ledger(root)
    assert "PLACEHOLDER" not in str(caught.value)
    with pytest.raises(LedgerError, match="line 2"):  # nothing is appended after it either
        append_event(root, generation_line("r3"))
    assert journal(root).read_bytes().startswith(open_bytes)


def test_diagnoses_reads_only_the_records(root, monkeypatch):
    written = write_stub_diagnosis(root, "b0001", cause="judge", diagnosis="placeholder line")
    guard = ledger_guard(monkeypatch, root)
    (found,) = diagnoses(root)
    assert (found.batch, found.cause, found.sha256) == ("b0001", "judge", sha256(written))
    assert (found.diagnosis, found.evidence, found.path) == ("placeholder line", (), written)
    assert not [p for p in guard.opened if "/judge" in p or "/responses/" in p]


def test_a_published_diagnosis_without_its_line_is_stranded(root):
    append_event(root, mark("b0001", 1, "second_change", []))
    publish(root, "b0001", diagnosis_record("b0001", "judge", "placeholder line"))
    assert read_ledger(root).stranded == ("b0001",)
    with pytest.raises(LedgerError, match="diagnose"):
        diagnoses(root)


def test_the_ledger_refuses_a_second_open_generation_or_controls_line(root, labeled_run):
    open_generation(root, "r1", 1, cap=3)
    with pytest.raises(LedgerError, match="open"):
        open_generation(root, "r1", 2, cap=3)
    with pytest.raises(LedgerError, match="generation 1"):
        append_event(root, superseded("b0001", "r1", 2, 1))
    open_generation(root, labeled_run.name, 1, cap=3)
    accept(root, labeled_run)
    with pytest.raises(LedgerError, match="accepted"):
        accept(root, labeled_run, attempt=2)
    changed = {"event": "controls", "batch": "b0009", "k": 1, "attempt": 1, "accepted": False,
               "changed": ["c000000000003"], "parsed_changed": [],
               "cache": {"path": "c", "sha256": "0" * 64}}  # fmt: skip
    append_event(root, {"event": "labeled", "batch": "b0009", "k": 1, "attempt": 1, "runs": []})
    append_event(root, changed)
    with pytest.raises(LedgerError, match="second controls line"):
        append_event(root, changed)
    assert (
        read_ledger(root).batches["b0009"]["k"],
        read_ledger(root).batches["b0009"]["changes"],
    ) == (2, 1)
    opened_line = {"event": "opened", "batch": "b0002", "step": "validation", "candidate": 1,
                   "runs": ["r1"], "controls": {"path": "c", "sha256": "0" * 64},
                   "fixation": {"path": "f", "sha256": "1" * 64}, "judge": {
                       "config": "judge/wildguard", **IDENTITY},
                   "code": {"commit": None, "dirty": None}}  # fmt: skip
    append_event(root, opened_line)
    with pytest.raises(LedgerError, match="b0002 is open"):
        append_event(root, {**opened_line, "batch": "b0003"})
    assert read_ledger(root).unfinished() == "b0002"


def test_covers_name_only_receipts_of_the_same_attempt(root, labeled_run):
    open_generation(root, labeled_run.name, 1, cap=3)
    receipt = sha256(receipt_path(labeled_run, JUDGE))
    append_event(root, {"event": "labeled", "batch": "b0001", "k": 1, "attempt": 1, "runs": [
        {"run": labeled_run.name, "status": LABELED, "receipt_sha256": receipt}]})  # fmt: skip
    append_event(root, {"event": "labeled", "batch": "b0001", "k": 1, "attempt": 2, "runs": []})
    cover = {"run": labeled_run.name, "generation": 1, "receipt_sha256": receipt,
             "labels_sha256": sha256(label_path(labeled_run, JUDGE))}  # fmt: skip
    controls = {"event": "controls", "batch": "b0001", "k": 1, "attempt": 2, "accepted": True,
                "changed": [], "parsed_changed": [], "cache": {"path": "c", "sha256": "0" * 64},
                "covers": [cover]}  # fmt: skip
    with pytest.raises(LedgerError, match="attempt"):
        append_event(root, controls)
    with pytest.raises(LedgerError, match="generation 2"):
        append_event(root, {**controls, "attempt": 1, "covers": [{**cover, "generation": 2}]})
    append_event(root, {**controls, "attempt": 1})
    assert read_ledger(root).covered == {(labeled_run.name, 1): receipt}


def test_the_ledgers_folder_is_private_and_its_lock_exclusive(root):
    with ledger_lock(root) as held, ledger_lock(root) as again:
        assert held is True and again is False
    assert ledger_dir(root).stat().st_mode & 0o777 == 0o700
    assert ledger_dir(root) == root / "study3" / "labeling"
