"""Study 3's label reader (analysis/study3_labels.py): the two points where harm leaves sealed
storage, and the one reader of a run's harm and of its items, over the labeling command's ledger
(``study3_label_cli``: its batch files and acceptances under ``<sealed dir>/study3-ledger/``).

The runs are tests/test_study2_labels.py's (six benign placeholder prompts at step 25), placed
under ``<runs root>/study3/`` and labeled through ``run_judging.label_runs`` against a fake judge
server that answers by content (``ContentJudge``). The ledger's files are written by the labeling
command's own writers (``_new_batch``, ``_write_batch``, ``_write_whole``, ``label_hashes``), or by
its ``label_batch`` itself. Nothing reads a runs/ folder of the checkout."""

import hashlib
import inspect
import json
import subprocess
import sys
import threading
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_study2_labels import IDS, JUDGE, PROMPTS, factory, make_run
from test_study3_package2 import SealedGuard
from test_study3_progress import Study
from test_study3_records import World as RecordsWorld

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import (
    study3_label_cli,
    study3_labels,
    study3_progress,
    study3_versions,
)
from zero_field_refusal.analysis.evidence import label_path
from zero_field_refusal.analysis.study2_label_cli import LABELED
from zero_field_refusal.analysis.study3_label_cli import (
    ACCEPTANCE_KIND,
    LEDGER,
    _new_batch,
    _write_batch,
    _write_whole,
    label_batch,
    label_hashes,
)
from zero_field_refusal.analysis.study3_labels import (
    ITEM_LABELS,
    LabelingProblem,
    LabelsPending,
    LedgerError,
    RunHarm,
    SealedPoint,
    diagnoses,
    harm_many,
    labeling_problems,
    ledger_runs,
    open_point,
    read_ledger,
    run_harm,
    run_items,
)
from zero_field_refusal.analysis.study3_progress import THETA0_RUN
from zero_field_refusal.analysis.study3_records import SEEDS
from zero_field_refusal.paths import REPO_ROOT
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


# ------------------------------------------------------------------ the labeling's ledger


def sealed(root: Path) -> Path:
    """The labeling's sealed directory by default: ``<runs root>/sealed``."""
    return root / "sealed"


def batch_of(root: Path, runs, *, charged=None, mark: bool = False, judge: str = JUDGE,
             directory: Path | None = None) -> dict:  # fmt: skip
    """A batch of the ledger holding ``runs`` (their folders), made and written by the labeling
    command's own functions, its runs' ``charged`` rounds (by run id) and its judge's ``mark``
    set as the labeling sets them after a quarantine and a second change."""
    batch = _new_batch([Path(run) for run in runs], judge)
    for run, rounds in (charged or {}).items():
        batch["rounds"][run]["charged"] = rounds
    batch["judge_mark"] = mark
    _write_batch(sealed(root) if directory is None else directory, batch)
    return batch


def accept(root: Path, batch: dict, runs, attempt: int = 1, judge: str = JUDGE,
           directory: Path | None = None) -> None:  # fmt: skip
    """The acceptance of ``runs`` by ``batch``'s attempt ``attempt``, as the labeling writes it
    once the controls after the attempt passed: each run's label file and receipt as they are."""
    folder = sealed(root) if directory is None else directory
    _write_whole(folder / LEDGER / f"accepted-{batch['batch']}-{attempt}.json",
                 {"kind": ACCEPTANCE_KIND, "batch": batch["batch"], "attempt": attempt,
                  "judge": judge, "accepted_at": WHEN,
                  "runs": {Path(run).name: label_hashes(Path(run), JUDGE) for run in runs}},
                 once=True)  # fmt: skip


def accepted_run(root: Path, name: str, rounds: int = 3, **kw) -> Path:
    """A run labeled with ``rounds`` rounds, in a batch of its own that accepted it."""
    run = study3_run(root, name, **kw)
    label(run, rounds)
    accept(root, batch_of(root, [run]), [run])
    return run


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


def stand_in_freeze(monkeypatch) -> dict:
    """The real ``study3_freeze.freeze_open`` replaced by the freeze gate's answer the test sets
    (``found["problems"]``, one problem at first); each call kept in ``found["calls"]``."""
    from zero_field_refusal.analysis import study3_freeze

    found: dict = {"problems": ("docs/study3-analysis-freeze-c1.json: not committed",),
                   "calls": []}  # fmt: skip

    def gate(candidate, runs_root, repo_root):
        found["calls"].append((candidate, Path(runs_root), Path(repo_root)))
        return found["problems"]

    monkeypatch.setattr(study3_freeze, "freeze_open", gate)
    return found


@pytest.fixture
def freeze(monkeypatch) -> dict:
    return stand_in_freeze(monkeypatch)


class Bound:
    """Runs as the launch journal binds them, each labeled and accepted in a batch of its own:
    candidate 1's validation runs ``v1``, ``v2`` and confirmation runs ``c1``, ``c2``, its search
    run ``s1``, its refused confirmation run ``c3`` and candidate 2's validation run ``w1``. The
    reader of the run records is given (each run's step, candidate and state, under the state's
    prescribed cells, θ₀ at its default place)."""

    validation_runs, confirmation_runs, search_runs = ["v1", "v2"], ["c1", "c2"], ["s1"]
    PRESCRIBED = {"validation-c1": ["A-1.45e-05/s10300"]}

    def __init__(self, root: Path, monkeypatch) -> None:
        self.root = root
        self.cells = {**{r: ("validation", 1, "usable") for r in self.validation_runs},
                      **{r: ("confirmation", 1, "usable") for r in self.confirmation_runs},
                      "s1": ("search", 1, "usable"), "c3": ("confirmation", 1, "refused"),
                      "w1": ("validation", 2, "usable")}  # fmt: skip
        for name in self.cells:
            accepted_run(root, name)
        state = {"prescribed": self.PRESCRIBED}
        monkeypatch.setattr(study3_labels, "_state", lambda runs_root, repo_root: state)
        monkeypatch.setattr(study3_labels, "EndpointReader", self.reader)

    def reader(self, runs_root, *, theta0):
        assert (Path(runs_root), Path(theta0)) == (self.root, self.root / THETA0_RUN)
        return SimpleNamespace(read=self.read)

    def read(self, prescribed):
        assert prescribed == self.PRESCRIBED
        return [SimpleNamespace(run_id=run, state=state,
                                cell=SimpleNamespace(step=step, candidate=candidate))
                for run, (step, candidate, state) in self.cells.items()], []  # fmt: skip

    def open(self, point: str) -> frozenset[str]:
        return open_point(point, candidate=1, runs_root=self.root, repo_root=self.root)

    def harm(self, name: str, *, point: str) -> RunHarm:
        return run_harm(self.root / "study3" / name, point=point, candidate=1,
                        runs_root=self.root, repo_root=self.root)  # fmt: skip

    def items(self, name: str, *, point: str):
        return run_items(self.root / "study3" / name, point=point, candidate=1,
                         runs_root=self.root, repo_root=self.root)  # fmt: skip


@pytest.fixture
def bound_world(root, monkeypatch) -> Bound:
    return Bound(root, monkeypatch)


def harm_of(run: Path, root: Path, **kw) -> RunHarm:
    return run_harm(run, point="spread", candidate=1, runs_root=root, repo_root=root, **kw)


# ------------------------------------------------------------------ the reader's gates


def test_a_receipt_alone_is_not_enough(labeled_run, root, opened):
    """A labeled run counts only through an acceptance: not in a batch, then in a batch not yet
    accepted, it is pending; once accepted, its harm is read."""
    opened("spread", {labeled_run.name})
    with pytest.raises(LabelsPending, match="no labeling batch holds it"):
        harm_of(labeled_run, root)
    batch = batch_of(root, [labeled_run])
    with pytest.raises(LabelsPending, match="no acceptance names"):
        harm_of(labeled_run, root)
    accept(root, batch, [labeled_run])
    assert harm_of(labeled_run, root) == RunHarm(
        n=6, harmful=2, missing=1, receipt_sha256=sha256(receipt_path(labeled_run, JUDGE))
    )


def test_an_acceptance_of_another_judge_or_another_run_does_not_count(labeled_run, root, opened):
    opened("spread", {labeled_run.name})
    batch = batch_of(root, [labeled_run])
    accept(root, batch, [labeled_run], judge="another-judge")
    with pytest.raises(LabelsPending, match="no acceptance names"):
        harm_of(labeled_run, root)
    other = study3_run(root, "r-other")
    label(other, 3)
    accept(root, batch_of(root, [other]), [other])
    with pytest.raises(LabelsPending, match="no acceptance names"):
        harm_of(labeled_run, root)


def test_no_round_left_is_a_technical_problem(labeled_run, root, opened):
    """A run its batch charged three rounds (quarantined twice over, say) and no acceptance names
    is a technical labeling problem, never 521 missing labels."""
    opened("spread", {labeled_run.name})
    batch_of(root, [labeled_run], charged={labeled_run.name: 3})
    with pytest.raises(LabelingProblem, match="rounds"):
        harm_of(labeled_run, root)


def test_the_judges_mark_is_a_problem_for_the_runs_no_acceptance_names(root, opened):
    """In a batch with the judge's mark, a run no acceptance names is a technical problem; a run
    accepted by an earlier attempt of the batch still counts, as the labeling leaves it alone."""
    done, open_ = study3_run(root, "r1"), study3_run(root, "r2")
    label(done, 3)
    label(open_, 3)
    opened("spread", {"r1", "r2"})
    batch = batch_of(root, [done, open_], mark=True)
    accept(root, batch, [done])
    assert harm_of(done, root).harmful == 2
    with pytest.raises(LabelingProblem, match="technical mark"):
        harm_of(open_, root)
    assert labeling_problems(root) == ("r2",)


def test_a_run_reads_its_rounds_against_the_cap_its_batch_left_it(root, opened):
    """A run charged one round (a quarantine) has a cap of two: two rounds with an unparsed item
    end its labeling, its missing label counted; one round leaves it pending."""
    done = study3_run(root, "r1")
    short = study3_run(root, "r2")
    label(done, 2)
    label(short, 1)
    opened("spread", {"r1", "r2"})
    batch = batch_of(root, [done, short], charged={"r1": 1, "r2": 1})
    accept(root, batch, [done, short])
    assert harm_of(done, root).missing == 1
    with pytest.raises(LabelsPending, match="cap 2"):
        harm_of(short, root)


def test_outside_the_cohort_nothing_sealed_is_opened(labeled_run, root, opened, monkeypatch):
    opened("spread", {"another-run"})
    guard = SealedGuard(monkeypatch, runs_root=root)
    with pytest.raises(SealedPoint, match=labeled_run.name):
        harm_of(labeled_run, root)
    assert guard.attempts == [] and not [p for p in guard.opened if "/sealed/" in p]


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
    runs = [accepted_run(root, name, unparseable=()) for name in ("r1", "r2", "r3")]
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
    batch = batch_of(root, [labeled_run])
    with pytest.raises(LabelsPending):
        run_items(labeled_run, point="spread", candidate=1, runs_root=root, repo_root=root)
    accept(root, batch, [labeled_run])
    items = run_items(labeled_run, point="spread", candidate=1, runs_root=root, repo_root=root)
    harm = harm_of(labeled_run, root)
    assert [i.index for i in items] == list(range(6)) and [i.prompt_id for i in items] == IDS
    assert Counter(i.label for i in items) == {
        "harmful": harm.harmful,
        "missing": harm.missing,
        "not_harmful": 6 - harm.harmful - harm.missing,
    }
    assert {i.label for i in items} <= set(ITEM_LABELS)
    assert [i.label for i in items][:3] == ["harmful", "harmful", "missing"]


def test_a_run_folder_must_be_a_plain_child_of_the_experiment(labeled_run, root, opened, tmp_path):
    opened("spread", {labeled_run.name, "elsewhere", "decisions"})
    accept(root, batch_of(root, [labeled_run]), [labeled_run])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.symlink_to(labeled_run)
    linked = root / "study3" / "linked"
    linked.symlink_to(labeled_run)
    for run_dir in (elsewhere, root / "study3" / "decisions", linked):
        with pytest.raises(ValueError, match="plain child"):
            harm_of(run_dir, root)
    with pytest.raises(ValueError, match="point"):
        run_harm(labeled_run, point="other", candidate=1, runs_root=root, repo_root=root)


def test_the_ledger_is_read_under_the_sealed_directory_given(labeled_run, root, opened, tmp_path):
    """``sealed_dir`` names the labeling's sealed directory: its ledger alone is read, the
    default one (``<runs root>/sealed``) not."""
    opened("spread", {labeled_run.name})
    elsewhere = tmp_path / "elsewhere-sealed"
    accept(root, batch_of(root, [labeled_run], directory=elsewhere), [labeled_run],
           directory=elsewhere)  # fmt: skip
    assert harm_of(labeled_run, root, sealed_dir=elsewhere).harmful == 2
    with pytest.raises(LabelsPending, match="no labeling batch"):
        harm_of(labeled_run, root)


def test_ledger_runs_and_labeling_problems_read_ids_only(root, monkeypatch):
    """``ledger_runs``: every run a batch file names, whatever its judge; ``labeling_problems``:
    the frozen judge's runs that are a technical problem. Neither opens a run's judge files."""
    runs = [root / "study3" / name for name in ("r1", "r2", "r3", "r4")]
    batch_of(root, runs[:2], charged={"r2": 3})
    batch_of(root, runs[2:3], judge="another-judge", charged={"r3": 3})
    batch_of(root, runs[3:], mark=True)
    guard = SealedGuard(monkeypatch, runs_root=root)  # a run's judge/ stays sealed
    assert ledger_runs(root) == {"r1", "r2", "r3", "r4"}
    assert labeling_problems(root) == ("r2", "r4")
    assert guard.attempts == [] and not [p for p in guard.opened if "/judge" in p]
    assert [p for p in guard.opened if "/sealed/study3-ledger/" in p]


def test_a_ledger_that_does_not_read_is_refused(labeled_run, root, opened):
    """A ledger file that does not parse, or a batch that does not list its runs, is the labeling
    command's ``LedgerError`` for every reader (fail closed); the readers re-export it."""
    assert LedgerError is study3_label_cli.LedgerError
    opened("spread", {labeled_run.name})
    batch = batch_of(root, [labeled_run])
    accept(root, batch, [labeled_run])
    path = sealed(root) / LEDGER / f"batch-{batch['batch']}.json"
    good = path.read_text()
    path.write_text(json.dumps({**batch, "runs": "r-labeled"}))
    for call in (lambda: harm_of(labeled_run, root), lambda: ledger_runs(root),
                 lambda: labeling_problems(root)):  # fmt: skip
        with pytest.raises(LedgerError, match="does not list its runs"):
            call()
    path.write_text(good[:-5])
    with pytest.raises(LedgerError, match="does not read"):
        read_ledger(root)
    path.write_text(json.dumps({**batch, "rounds": {}}))
    with pytest.raises(LedgerError, match="charged rounds"):
        harm_of(labeled_run, root)
    path.unlink()
    with pytest.raises(LedgerError, match="no batch of the ledger holds it"):
        harm_of(labeled_run, root)


def test_the_ledger_records_no_diagnosis(root):
    assert diagnoses(root) == () and diagnoses(root, sealed(root)) == ()


def test_the_final_point_is_closed_until_the_freeze_gate(root, monkeypatch):
    """The real gate (``study3_freeze.freeze_open``) on a checkout with no analysis freeze: the
    final point is closed, its problem counted and the first named, and nothing is opened."""
    guard = SealedGuard(monkeypatch, runs_root=root)
    with pytest.raises(SealedPoint, match=r"the final point waits for candidate 1's freeze: 1 "
                       r"problem\(s\), first: docs/study3-analysis-freeze-c1\.json: no analysis "
                       "freeze"):  # fmt: skip
        open_point("final", candidate=1, runs_root=root, repo_root=root)
    assert guard.attempts == []


def test_the_final_point_opens_only_with_a_valid_freeze(freeze, bound_world):
    with pytest.raises(SealedPoint, match="freeze"):
        bound_world.open("final")
    assert freeze["calls"] == [(1, bound_world.root, bound_world.root)]
    freeze["problems"] = ()
    assert bound_world.open("final") == set(bound_world.validation_runs
                                            + bound_world.confirmation_runs)  # fmt: skip


def test_final_reads_validation_and_confirmation_runs_and_no_other(freeze, bound_world):
    freeze["problems"] = ()
    for run in (bound_world.validation_runs[0], bound_world.confirmation_runs[0]):
        assert bound_world.harm(run, point="final").n == 6
        assert len(bound_world.items(run, point="final")) == 6
    for run in ("s1", "c3", "w1"):  # a search run, a refused attempt, candidate 2's run
        with pytest.raises(SealedPoint, match=run):
            bound_world.harm(run, point="final")
    freeze["problems"] = ("docs/study3-analysis-freeze-c1.json: not the frozen records",)
    with pytest.raises(SealedPoint, match="1 problem"):
        bound_world.harm(bound_world.validation_runs[0], point="final")


def test_the_final_points_runs_are_the_attempts_the_launch_journal_binds(tmp_path):
    """The final point's run set read by the real reader of the run records (``_attempts``):
    candidate 1's validation and confirmation attempts in any state but refused; not its search
    run, not a refused attempt and not candidate 2's."""
    records = RecordsWorld(tmp_path / "runs")
    validation = records.run("A", seed=SEEDS["validation"][1][0], role="validation")
    confirmation = records.run("A", seed=SEEDS["confirmation"][1][0], role="confirmation")
    records.run("A", seed=SEEDS["search"][1][0], role="search")
    records.run("A", seed=SEEDS["validation"][2][0], role="validation", candidate=2,
                campaign="study3-c2")  # fmt: skip
    refused = records.run("F", 1.3e-5, seed=SEEDS["confirmation"][1][1], role="confirmation")
    config = records.root / "study3" / refused / "config.yaml"
    config.unlink()
    config.symlink_to(Path("responses") / "harmful_evaluation.jsonl")
    state = {"prescribed": records.prescribed}
    found = study3_labels._attempts(records.root, state, 1, ("validation", "confirmation"),
                                    theta0=records.theta0)  # fmt: skip
    assert found == {validation, confirmation}
    assert study3_labels._attempts(records.root, state, 1, ("confirmation",),
                                   theta0=records.theta0) == {confirmation}  # fmt: skip


def test_the_real_freeze_module_has_the_specified_interface():
    from zero_field_refusal.analysis import study3_freeze

    assert list(inspect.signature(study3_freeze.freeze_open).parameters) == [
        "candidate", "runs_root", "repo_root"]  # fmt: skip


def test_nothing_of_the_freeze_is_imported_with_the_label_readers():
    """The freeze module is imported inside the final gate alone: importing the label readers
    loads nothing of it."""
    code = ("import sys, zero_field_refusal.analysis.study3_labels, "
            "zero_field_refusal.analysis.study3_label_cli; "
            "print('zero_field_refusal.analysis.study3_freeze' in sys.modules)")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True, cwd=REPO_ROOT).stdout  # fmt: skip
    assert out.strip() == "False"


def test_a_receipt_changed_after_its_acceptance_is_pending(labeled_run, root, opened):
    opened("spread", {labeled_run.name})
    accept(root, batch_of(root, [labeled_run]), [labeled_run])
    path = receipt_path(labeled_run, JUDGE)
    path.write_text(path.read_text() + " ")
    with pytest.raises(LabelsPending, match="no acceptance names"):
        harm_of(labeled_run, root)


def test_a_label_file_changed_after_its_acceptance_is_pending(labeled_run, root, opened):
    opened("spread", {labeled_run.name})
    accept(root, batch_of(root, [labeled_run]), [labeled_run])
    path = label_path(labeled_run, JUDGE)
    path.write_text(path.read_text() + "\n")
    with pytest.raises(LabelsPending):
        harm_of(labeled_run, root)


def test_a_file_changed_between_the_gate_and_the_reading_is_pending(labeled_run, root, opened,
                                                                    monkeypatch):  # fmt: skip
    """The label file and receipt read must be those the gate found accepted, by their sha256."""
    opened("spread", {labeled_run.name})
    accept(root, batch_of(root, [labeled_run]), [labeled_run])
    real = study3_labels._gated

    def gated(*args):
        found = real(*args)
        label_path(labeled_run, JUDGE).write_text(
            label_path(labeled_run, JUDGE).read_text() + "\n")  # fmt: skip
        return found

    monkeypatch.setattr(study3_labels, "_gated", gated)
    with pytest.raises(LabelsPending, match="changed while it was read"):
        harm_of(labeled_run, root)


def test_labels_written_by_the_labeling_command_are_read(root, opened):
    """The labeling command's own batch (``study3_label_cli.label_batch``: the controls, the
    labeling, the controls again, then the acceptance) is read as it wrote it: the harm of both
    runs, the unparsed item missing after the three rounds."""
    runs = [study3_run(root, "r1"), study3_run(root, "r2", harmful=(0,), unparseable=())]
    controls = [{"item_id": f"c00000000000{i}", "prompt": f"Placeholder control {i}.",
                 "response": f"Placeholder answer {i}.", "expected_label": "refusal"}
                for i in range(2)]  # fmt: skip
    batch = _new_batch(runs, JUDGE)
    _write_batch(sealed(root), batch)
    outcome = label_batch(batch, JUDGE, lambda cfg: PROMPTS, factory(ContentJudge()), controls,
                          sealed(root), invocation="i1")  # fmt: skip
    assert set(outcome.statuses.values()) == {LABELED} and not outcome.stop
    opened("spread", {"r1", "r2"})
    found = harm_many(runs, point="spread", candidate=1, runs_root=root, repo_root=root)
    assert [(h.harmful, h.missing) for h in found] == [(2, 1), (1, 0)]
    assert ledger_runs(root) == {"r1", "r2"} and labeling_problems(root) == ()
