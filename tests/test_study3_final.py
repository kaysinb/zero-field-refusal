"""Study 3's one label reader of the final analysis (analysis/study3_final.py): every entry refuses
before the analysis freeze is committed and opens nothing meanwhile; a pending or broken cohort
run refuses the reading and is never counted as missing labels; the spread of K reads only the K
rows, undecoded but for their groups; a run's items agree with its harm; and the blind texts wait
for the final analysis.

The world is the records tests' (``test_study3_records.World``): four confirmation runs of block
10400 whose sealed files hold the sentinel (``responses/``, ``judge/``, ``judge-superseded/``, and
a utility row of ``observations/measurements.jsonl``), with the K rows beside it. The label
reader's own functions and the freeze gate are stubbed where a test gives their answer; nothing
reads a runs/ folder of the checkout."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_study2_labels import IDS, PROMPTS, make_run, text
from test_study3_package2 import LabelReaders, SealedGuard, judged
from test_study3_progress import START
from test_study3_records import SENTINEL, header
from test_study3_records import World as RecordsWorld

from zero_field_refusal.analysis import study2_labels, study3_final, study3_labels
from zero_field_refusal.analysis.study3_freeze import freeze_name
from zero_field_refusal.analysis.study3_labels import (
    Accepted,
    ItemLabel,
    LabelingProblem,
    LabelsPending,
    RunHarm,
    SealedPoint,
)

K = 6.0  # write_run's K_benign at step 25
ACCEPTED = Accepted(batch="b", cap=3, labels_sha256="e" * 64, receipt_sha256="f" * 64)
# Each prompt's two K samples at step 25: per-prompt means 5, 6 and 7, their mean K
SAMPLES = {"kl:0": (4.5, 5.5), "kl:1": (6.0, 6.0), "kl:2": (6.5, 7.5)}


def kl_row(condition: str, item: str, measure: str, sample: int, value: float, **extra) -> str:
    """A K row as the evaluation writes it (``json.dumps(row, sort_keys=True)``)."""
    row = {"condition": condition, "measure": measure, "item_id": item, "sample": sample,
           "value": value, "tokens": 7, **extra}  # fmt: skip
    return json.dumps(row, sort_keys=True) + "\n"


def k_rows(run_dir: Path) -> None:
    """The run's K rows beside its utility row (the sentinel): ``SAMPLES`` of ``kl_benign`` at
    step 25, the same prompts far off at step 0 and in ``kl_harmful``, and a K-like row with a
    text, which is no K row."""
    rows = [kl_row("step-25", item, "kl_benign", i, value)
            for item, values in SAMPLES.items() for i, value in enumerate(values)]  # fmt: skip
    rows += [kl_row("step-0", item, "kl_benign", 0, 100.0) for item in SAMPLES]
    rows += [kl_row("step-25", item, "kl_harmful", 0, 50.0) for item in SAMPLES]
    rows.append(kl_row("step-25", "kl:9", "kl_benign", 0, 600.0, text=SENTINEL))
    with (run_dir / "observations" / "measurements.jsonl").open("a") as f:
        f.writelines(rows)


class Final:
    """Four confirmation runs of candidate 1's block 10400 (A and F at both rates), each with K
    rows and judge outputs; a checkout ``repo`` with no analysis freeze; ``analysis``, the path the
    final analysis is written at."""

    def __init__(self, tmp_path: Path) -> None:
        self.records = RecordsWorld(tmp_path / "runs")
        self.root = self.records.root
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        self.runs = [self.records.run(arm, START[arm][label], 10400, role="confirmation",
                                      head=header(1, "confirmation", None),
                                      campaign="study3-c1-confirmation-b10400")
                     for arm in ("A", "F") for label in ("l", "u")]  # fmt: skip
        for run in self.runs:
            k_rows(self.dir(run))
            judged(self.dir(run))
        self.analysis = self.root / "study3-final" / "c1" / "analysis.json"

    def dir(self, run: str) -> Path:
        return self.root / "study3" / run

    def dirs(self) -> dict[str, Path]:
        return {run: self.dir(run) for run in self.runs}


@pytest.fixture
def world(tmp_path) -> Final:
    return Final(tmp_path)


def open_freeze(world: Final, monkeypatch, validation: tuple[str, ...] = ()) -> list:
    """The freeze gate opens (``freeze_open`` → ``()``), over a stand-in freeze in the checkout
    whose validation blocks hold the runs ``validation`` (A's two, then F's); its calls kept."""
    calls: list = []
    blocks = [{"seed": 10300, "arms": {"A": {"lo": validation[0], "hi": validation[1]},
                                       "F": {"lo": validation[2], "hi": validation[3]}}}
              ] if validation else []  # fmt: skip
    freeze = world.repo / freeze_name(1)
    freeze.parent.mkdir(exist_ok=True)
    validation_ = {"A-F": {"blocks": blocks}, "P-F": None}
    freeze.write_text(json.dumps({"kind": "analysis-freeze", "candidate": 1,
                                  "validation": validation_}))  # fmt: skip

    def gate(candidate, runs_root, repo_root):
        calls.append((candidate, Path(runs_root), Path(repo_root)))
        return ()

    monkeypatch.setattr(study3_final, "freeze_open", gate)
    return calls


def open_runs(monkeypatch, runs) -> None:
    """The final point's run set is ``runs``."""

    def opened(point, **kw):
        return frozenset(runs) if point == "final" else frozenset()

    monkeypatch.setattr(study3_final, "open_point", opened)


def call(entry: str, world: Final):
    """The entry with the world's arguments."""
    run = world.dir(world.runs[0])
    given = {"candidate": 1, "runs_root": world.root, "repo": world.repo}
    if entry == "cohort_harm":
        return study3_final.cohort_harm(world.dirs(), **given)
    if entry == "k_spread":
        return study3_final.k_spread(run, k=K, **given)
    if entry == "item_labels":
        return study3_final.item_labels({world.runs[0]: run}, **given)
    if entry == "blind_texts":
        return study3_final.blind_texts({world.runs[0]: ["evaluation:0"]},
                                        analysis=world.analysis, **given)  # fmt: skip
    assert entry == "gather"
    return study3_final.gather({"candidate": 1}, [], runs_root=world.root, repo=world.repo)


# ------------------------------------------------------------------ the gate


@pytest.mark.parametrize("entry", [*study3_final.SEALED_ENTRIES, "gather"])
def test_every_entry_refuses_before_the_freeze_and_opens_nothing(world, monkeypatch, entry):
    """The real freeze gate, with no analysis freeze in the checkout: each entry raises
    ``SealedPoint`` naming the freeze, opens nothing sealed, calls no label reader and shows no
    sentinel."""
    labels = LabelReaders(monkeypatch)
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    with pytest.raises(SealedPoint, match="analysis-freeze-c1") as caught:
        call(entry, world)
    assert guard.attempts == [] and labels.calls == []
    assert SENTINEL not in str(caught.value)


def test_the_gate_names_every_problem_by_count_and_the_first(world, monkeypatch) -> None:
    problems = ("docs/study3-analysis-freeze-c1.json: not committed", "a second problem")
    monkeypatch.setattr(study3_final, "freeze_open", lambda *a: problems)
    with pytest.raises(SealedPoint, match=r"2 problem\(s\), first: .*not committed"):
        call("cohort_harm", world)


# ------------------------------------------------------------------ harm


def label_raisers(monkeypatch) -> list[str]:
    """The label reader's reading of a run's labels (``study3_labels._read_run``) and its
    ``run_harm``, each a raiser that keeps its call: no label is read."""
    calls: list[str] = []

    def read_run(run_dir, found):
        calls.append(f"_read_run {Path(run_dir).name}")
        raise AssertionError("a run's labels were read")

    def harm(run_dir, **kw):
        calls.append(f"run_harm {Path(run_dir).name}")
        raise AssertionError("run_harm was called")

    monkeypatch.setattr(study3_labels, "_read_run", read_run)
    monkeypatch.setattr(study3_labels, "run_harm", harm)
    return calls


def gate_steps(monkeypatch, named, raised) -> list[str]:
    """The label reader's gate steps (``study3_labels._gated``): ``raised`` for a run of
    ``named``, the others passing; every run asked kept."""
    asked: list[str] = []

    def gated(run_dirs, point, candidate, runs_root, repo_root, sealed_dir=None):
        out = []
        for run_dir in run_dirs:
            asked.append(Path(run_dir).name)
            if Path(run_dir).name in named:
                raise raised(f"{Path(run_dir).name}: not countable")
            out.append((Path(run_dir), ACCEPTED))
        return out

    monkeypatch.setattr(study3_labels, "_gated", gated)
    return asked


@pytest.mark.parametrize("raised,field", [(LabelsPending, "runs"), (LabelingProblem, "problems")])
def test_a_pending_or_broken_cohort_run_refuses_and_is_never_missing(world, monkeypatch, raised,
                                                                     field) -> None:  # fmt: skip
    """The cohort's one reading (the real ``harm_many``) stops at the gate steps of two runs: the
    refusal names both, found through the gate steps alone, and no label is read, by the reading
    or after it (no ``_read_run``, no ``run_harm``)."""
    open_freeze(world, monkeypatch)  # freeze_open -> ()
    named = {world.runs[1], world.runs[3]}
    reads = label_raisers(monkeypatch)
    asked = gate_steps(monkeypatch, named, raised)
    monkeypatch.setattr(study3_final, "labeling_problems",
                        lambda root, sealed_dir=None: tuple(named) if raised is LabelingProblem
                        else ())  # fmt: skip
    with pytest.raises(study3_final.Pending) as caught:
        study3_final.cohort_harm({r: world.root / "study3" / r for r in world.runs},
                                 candidate=1, runs_root=world.root, repo=world.repo)  # fmt: skip
    assert set(getattr(caught.value, field)) == named
    other = "problems" if field == "runs" else "runs"
    assert getattr(caught.value, other) == ()
    assert f"{len(named) * (field == 'runs')} runs pending" in str(caught.value)
    assert reads == [] and set(world.runs) - named <= set(asked)  # every other run asked


def test_a_refusal_names_the_pending_run_through_the_gate_steps_and_reads_no_label(
    world, monkeypatch
):
    """One run pending at the gate steps, ``_read_run`` a raiser: the refusal names that run, and
    no label of the countable runs is read to find it."""
    open_freeze(world, monkeypatch)
    culprit = world.runs[2]
    reads = label_raisers(monkeypatch)
    gate_steps(monkeypatch, {culprit}, LabelsPending)
    monkeypatch.setattr(study3_final, "labeling_problems", lambda root, sealed_dir=None: ())
    with pytest.raises(study3_final.Pending) as caught:
        study3_final.cohort_harm(world.dirs(), candidate=1, runs_root=world.root, repo=world.repo)
    assert (caught.value.runs, caught.value.problems) == ((culprit,), ())
    assert culprit in str(caught.value) and reads == []


@pytest.mark.parametrize("raised,message", [
    (study2_labels.LabelsPending, "{dir}/judge/wildguard.receipt.json is missing: not labeled"),
    (LabelsPending, "{run}: its label file is missing"),
])  # fmt: skip
def test_a_pending_reading_of_a_runs_labels_names_that_run(world, monkeypatch, raised, message):
    """The gate steps pass for every run and the reading of one run's labels finds them pending
    (study 2's receipt check, or the label reader's own): the refusal names that run, from the
    message as the label reader writes it, and nothing more is read after it."""
    open_freeze(world, monkeypatch)
    culprit = world.runs[2]
    asked = gate_steps(monkeypatch, set(), LabelsPending)
    reads: list[str] = []

    def read_run(run_dir, found):
        reads.append(run_dir.name)
        if run_dir.name == culprit:
            raise raised(message.format(dir=run_dir, run=run_dir.name))
        return ()

    monkeypatch.setattr(study3_labels, "_read_run", read_run)
    monkeypatch.setattr(study3_final, "labeling_problems", lambda root, sealed_dir=None: ())
    with pytest.raises(study3_final.Pending) as caught:
        study3_final.cohort_harm(world.dirs(), candidate=1, runs_root=world.root, repo=world.repo)
    assert (caught.value.runs, caught.value.problems) == ((culprit,), ())
    assert reads == world.runs[:3]  # the reading's own, up to the culprit; none after it
    assert asked == [*world.runs, *world.runs]  # the reading's gate steps, then the scan's


def test_a_cohort_is_read_once_and_its_harm_kept_by_run(world, monkeypatch) -> None:
    gate = open_freeze(world, monkeypatch)
    reads: list = []

    def many(run_dirs, *, point, candidate, runs_root, repo_root, sealed_dir=None):
        reads.append((point, [d.name for d in run_dirs]))
        return [RunHarm(harmful=i, missing=1, receipt_sha256="f" * 64)
                for i, _ in enumerate(run_dirs)]  # fmt: skip

    monkeypatch.setattr(study3_final, "harm_many", many)
    found = study3_final.cohort_harm(world.dirs(), candidate=1, runs_root=world.root,
                                     repo=world.repo)  # fmt: skip
    assert reads == [("final", world.runs)] and len(gate) == 1
    assert {run: harm.harmful for run, harm in found.items()} == {
        run: i for i, run in enumerate(world.runs)}  # fmt: skip
    monkeypatch.setattr(study3_final, "harm_many",
                        lambda run_dirs, **kw: [RunHarm(n=520, harmful=0, missing=0,
                                                        receipt_sha256="f" * 64)
                                                for _ in run_dirs])  # fmt: skip
    with pytest.raises(ValueError, match="521"):
        study3_final.cohort_harm(world.dirs(), candidate=1, runs_root=world.root, repo=world.repo)
    swapped = {world.runs[0]: world.dir(world.runs[1])}
    with pytest.raises(ValueError, match="not named by it"):
        study3_final.cohort_harm(swapped, candidate=1, runs_root=world.root, repo=world.repo)


def test_another_reading_error_stops_the_cohort_unchanged(world, monkeypatch) -> None:
    """A culprit-free failure of the cohort's reading (every run's gate steps pass, and its
    message names no run) is raised as it is: nothing is missing, and no label is read."""
    open_freeze(world, monkeypatch)

    def many(run_dirs, **kw):
        raise LabelsPending("the receipt changed while it was read")

    monkeypatch.setattr(study3_final, "harm_many", many)
    reads = label_raisers(monkeypatch)
    gate_steps(monkeypatch, set(), LabelsPending)
    monkeypatch.setattr(study3_final, "labeling_problems", lambda root, sealed_dir=None: ())
    with pytest.raises(LabelsPending, match="changed while it was read"):
        study3_final.cohort_harm(world.dirs(), candidate=1, runs_root=world.root, repo=world.repo)
    assert reads == []


def test_validation_runs_wait_for_the_written_analysis(world, monkeypatch) -> None:
    """Runs of the freeze's validation blocks are read only once the final analysis is written,
    as the blind texts are: with no analysis given, or one not written yet, ``SealedPoint`` and
    nothing read; with the analysis at its place naming the committed freeze, one reading."""
    import hashlib

    open_freeze(world, monkeypatch, validation=tuple(world.runs))
    reads: list = []

    def many(run_dirs, **kw):
        reads.append([d.name for d in run_dirs])
        return [RunHarm(harmful=1, missing=0, receipt_sha256="f" * 64) for _ in run_dirs]

    monkeypatch.setattr(study3_final, "harm_many", many)
    given = {"candidate": 1, "runs_root": world.root, "repo": world.repo}
    with pytest.raises(SealedPoint, match="waits for candidate 1's final analysis: 4 validation"):
        study3_final.cohort_harm(world.dirs(), **given)
    with pytest.raises(SealedPoint, match="final analysis: .* is not written"):
        study3_final.cohort_harm(world.dirs(), analysis=world.analysis, **given)
    analysis_record(world, "0" * 64)  # another freeze
    with pytest.raises(SealedPoint, match="committed freeze"):
        study3_final.cohort_harm(world.dirs(), analysis=world.analysis, **given)
    assert reads == []
    freeze = world.repo / freeze_name(1)
    analysis_record(world, hashlib.sha256(freeze.read_bytes()).hexdigest())
    found = study3_final.cohort_harm(world.dirs(), analysis=world.analysis, **given)
    assert set(found) == set(world.runs) and reads == [world.runs]
    freeze.write_text('{"kind": "analysis-freeze"}')  # no validation: nothing is read
    with pytest.raises(SealedPoint, match="does not give validation's runs"):
        study3_final.cohort_harm(world.dirs(), analysis=world.analysis, **given)
    assert reads == [world.runs]


# ------------------------------------------------------------------ the spread of K


def test_the_k_spread_reads_only_k_rows_and_decodes_no_line(world, monkeypatch) -> None:
    open_freeze(world, monkeypatch)
    open_runs(monkeypatch, world.runs)
    calls: list = []
    real = json.loads

    def recorded(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(json, "loads", recorded)
    run = world.dir(world.runs[0])
    given = {"candidate": 1, "runs_root": world.root, "repo": world.repo}
    assert study3_final.k_spread(run, k=K, **given) == [5.0, 6.0, 7.0]
    assert calls == []
    with pytest.raises(ValueError, match="K"):
        study3_final.k_spread(run, k=K * 1.01, **given)
    assert study3_final.k_spread(run, k=100.0, condition="step-0",
                                 **given) == [100.0, 100.0, 100.0]  # fmt: skip
    assert calls == []


def test_the_k_spread_reads_under_the_runs_root_it_is_given(world, monkeypatch, tmp_path):
    """RUNS_DIR is the argument, never the run folder's grandparent: the gate is asked for it,
    and a run that is not a plain child of its study3/ folder is refused unopened."""
    gate = open_freeze(world, monkeypatch)
    open_runs(monkeypatch, world.runs)
    run = world.dir(world.runs[0])
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "study3").mkdir(parents=True)
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    with pytest.raises(ValueError, match="plain child"):
        study3_final.k_spread(run, k=K, candidate=1, runs_root=elsewhere, repo=world.repo)
    assert gate == [(1, elsewhere, world.repo)] and guard.attempts == []


def test_the_k_spread_refuses_a_sample_twice_no_row_and_a_run_outside(world, monkeypatch):
    open_freeze(world, monkeypatch)
    open_runs(monkeypatch, world.runs[:2])
    run = world.dir(world.runs[0])
    with (run / "observations" / "measurements.jsonl").open("a") as f:
        f.write(kl_row("step-25", "kl:0", "kl_benign", 0, 4.5))
    given = {"candidate": 1, "runs_root": world.root, "repo": world.repo}
    with pytest.raises(study3_final.SpreadRefused, match="twice"):
        study3_final.k_spread(run, k=K, **given)
    with pytest.raises(study3_final.SpreadRefused, match="no kl_benign row at step-50"):
        study3_final.k_spread(world.dir(world.runs[1]), k=K, condition="step-50", **given)
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    with pytest.raises(SealedPoint, match="outside the run set"):
        study3_final.k_spread(world.dir(world.runs[3]), k=K, **given)
    assert guard.attempts == []


@pytest.mark.parametrize("item,value", [("kl:0", 1.5e-07), ('a"b\\cé\t', -0.0),
                                        ("x", 12345678901234.5), ("y", 3)])  # fmt: skip
def test_the_kl_row_pattern_is_the_evaluations_row(item, value) -> None:
    """The bytes the evaluation writes for a K row fullmatch ``KL_ROW``, and its groups give the
    row back; a row with another key does not."""
    line = kl_row("step-25", item, "kl_benign", 3, value).rstrip("\n").encode()
    found = study3_final.KL_ROW.fullmatch(line)
    assert found is not None
    assert json.loads(b'"' + found["item"] + b'"') == item
    assert float(found["value"]) == value and int(found["sample"]) == 3
    assert (found["condition"], found["measure"]) == (b"step-25", b"kl_benign")
    other = kl_row("step-25", item, "kl_benign", 3, value, text=SENTINEL).rstrip("\n").encode()
    assert study3_final.KL_ROW.fullmatch(other) is None


# ------------------------------------------------------------------ the items


def items_of(harmful: int, missing: int, n: int = 521) -> tuple[ItemLabel, ...]:
    def label(i: int) -> str:
        return "harmful" if i < harmful else "missing" if i < harmful + missing else "not_harmful"

    return tuple(ItemLabel(i, f"p{i}", label(i)) for i in range(n))


@pytest.mark.parametrize("given,why", [(items_of(10, 2), None), (items_of(11, 2), "harmful"),
                                       (items_of(10, 3), "missing"),
                                       (items_of(10, 2, 520), "520 items")])  # fmt: skip
def test_a_runs_items_agree_with_its_harm(world, monkeypatch, given, why) -> None:
    open_freeze(world, monkeypatch)
    monkeypatch.setattr(study3_final, "run_items", lambda run_dir, **kw: given)
    monkeypatch.setattr(study3_final, "harm_many",
                        lambda run_dirs, **kw: [RunHarm(harmful=10, missing=2,
                                                        receipt_sha256="f" * 64)
                                                for _ in run_dirs])  # fmt: skip
    if why is None:
        assert call("item_labels", world) == {world.runs[0]: given}
    else:
        with pytest.raises(ValueError, match=why):
            call("item_labels", world)


def test_runs_items_wait_for_every_runs_harm_reading_no_item(world, monkeypatch) -> None:
    """The runs' items are read after their harm, from one reading: a pending run refuses with
    every culprit named, before any run's items are read."""
    open_freeze(world, monkeypatch)
    items: list = []
    monkeypatch.setattr(study3_final, "run_items",
                        lambda run_dir, **kw: items.append(run_dir) or items_of(10, 2))  # fmt: skip
    reads = label_raisers(monkeypatch)
    gate_steps(monkeypatch, {world.runs[3]}, LabelsPending)
    monkeypatch.setattr(study3_final, "labeling_problems", lambda root, sealed_dir=None: ())
    with pytest.raises(study3_final.Pending, match=world.runs[3]):
        study3_final.item_labels(world.dirs(), candidate=1, runs_root=world.root,
                                 repo=world.repo)  # fmt: skip
    assert items == [] and reads == []


# ------------------------------------------------------------------ the blind texts


def analysis_record(world: Final, sha256: str) -> None:
    world.analysis.parent.mkdir(parents=True, exist_ok=True)
    frozen = {"path": freeze_name(1), "sha256": sha256, "root": "repo"}
    world.analysis.write_text(json.dumps({"kind": study3_final.ANALYSIS_KIND, "candidate": 1,
                                          "freeze": frozen}))  # fmt: skip


def test_the_blind_texts_wait_for_the_analysis_then_give_the_named_texts(world, monkeypatch):
    import hashlib

    open_freeze(world, monkeypatch)
    run = make_run(world.root / "study3", steps=(5, 25), name="r-texts")
    open_runs(monkeypatch, [run.name])
    monkeypatch.setattr(study3_final, "harmful_prompts", lambda cfg: PROMPTS)
    monkeypatch.setattr(study3_final, "HARMFUL_PROMPTS", len(IDS))  # the run's prompts
    freeze = world.repo / freeze_name(1)
    freeze.write_text('{"kind": "analysis-freeze"}\n')
    named = [IDS[4], IDS[1]]
    given = {"candidate": 1, "runs_root": world.root, "repo": world.repo,
             "analysis": world.analysis}  # fmt: skip
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    with pytest.raises(SealedPoint, match="final analysis"):
        study3_final.blind_texts({run.name: named}, **given)
    analysis_record(world, "0" * 64)  # another freeze
    with pytest.raises(SealedPoint, match="committed freeze"):
        study3_final.blind_texts({run.name: named}, **given)
    assert guard.attempts == []
    guard.armed = False
    analysis_record(world, hashlib.sha256(freeze.read_bytes()).hexdigest())
    found = study3_final.blind_texts({run.name: named}, **given)
    assert found == {run.name: {pid: {"prompt": PROMPTS[pid], "response": text("step-25", pid)}
                                for pid in named}}  # fmt: skip
    assert list(found[run.name]) == named
    with pytest.raises(ValueError, match="no response at step-25"):
        study3_final.blind_texts({run.name: [*named, "judged:99"]}, **given)
    monkeypatch.setattr(study3_final, "HARMFUL_PROMPTS", 521)  # the label reader's own check
    with pytest.raises(ValueError, match=f"{len(IDS)} harmful prompts, not 521"):
        study3_final.blind_texts({run.name: named}, **given)


@pytest.mark.parametrize("where", ["elsewhere/analysis.json", "study3-final/c1/notes.json",
                                   "study3-final/c2/analysis.json",
                                   "study3-final/c1/analysis-1.json"])  # fmt: skip
def test_the_blind_texts_take_the_analysis_only_at_its_place(world, monkeypatch, where) -> None:
    """The analysis that opens the blind texts is candidate N's at its place,
    RUNS_DIR/study3-final/c<N>/analysis.json or analysis-<k>.json (k ≥ 2): the same record
    anywhere else opens nothing."""
    import hashlib

    open_freeze(world, monkeypatch)
    run = make_run(world.root / "study3", steps=(5, 25), name="r-texts")
    open_runs(monkeypatch, [run.name])
    freeze = world.repo / freeze_name(1)
    freeze.write_text('{"kind": "analysis-freeze"}\n')
    analysis_record(world, hashlib.sha256(freeze.read_bytes()).hexdigest())
    moved = world.root / where
    moved.parent.mkdir(parents=True, exist_ok=True)
    moved.write_bytes(world.analysis.read_bytes())
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    with pytest.raises(SealedPoint, match="study3-final/c1/analysis"):
        study3_final.blind_texts({run.name: [IDS[0]]}, candidate=1, runs_root=world.root,
                                 repo=world.repo, analysis=moved)  # fmt: skip
    assert guard.attempts == []


# ------------------------------------------------------------------ the analysis's reading


def block(seed: int, a: tuple[str, str], f: tuple[str, str]) -> dict:
    return {
        "seed": seed,
        "arms": {
            "A": {"lo": a[0], "hi": a[1], "w": 0.5, "width": 0.4},
            "F": {"lo": f[0], "hi": f[1], "w": 0.5, "width": 0.4},
        },
    }


def test_gather_reads_the_brackets_k_spread_then_every_frozen_run_once(world, monkeypatch):
    """The brackets' spreads first, in the final point's run set, then one harm reading of the
    comparisons', A − P's and the matched blocks' runs; validation's runs are not read (they are
    read after the analysis)."""
    open_freeze(world, monkeypatch)
    open_runs(monkeypatch, world.runs)
    reads: list = []
    order: list = []
    real = study3_final._k_spread

    def many(run_dirs, **kw):
        reads.append(sorted(d.name for d in run_dirs))
        order.append("harm")
        return [RunHarm(harmful=3, missing=0, receipt_sha256="f" * 64) for _ in run_dirs]

    def spread(run_dir, **kw):
        order.append("spread")
        return real(run_dir, **kw)

    monkeypatch.setattr(study3_final, "harm_many", many)
    monkeypatch.setattr(study3_final, "_k_spread", spread)
    validation = make_run(world.root / "study3", name="r-validation")  # frozen, never read here
    a, f = tuple(world.runs[:2]), tuple(world.runs[2:])
    freeze = {"candidate": 1, "comparisons": {"A-F": {"blocks": [block(10400, a, f)]}},
              "a_minus_p": {"blocks": []}, "k_harmful": {"blocks": [block(10400, a, f)]},
              "validation": {"A-F": {"blocks": [block(10300, (validation.name,) * 2, a)]},
                             "P-F": None}}  # fmt: skip
    endpoints = [SimpleNamespace(run_id=run, K=K) for run in world.runs]
    harm, spread = study3_final.gather(freeze, endpoints, runs_root=world.root, repo=world.repo)
    assert reads == [sorted(world.runs)]
    assert order == ["spread"] * len(world.runs) + ["harm"]
    assert set(harm) == set(world.runs)
    assert spread == {run: [5.0, 6.0, 7.0] for run in world.runs}
    order.clear()
    with pytest.raises(ValueError, match="no endpoint"):
        study3_final.gather(freeze, endpoints[1:], runs_root=world.root, repo=world.repo)
    open_runs(monkeypatch, world.runs[1:])  # a bracket run outside the final point's run set
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    with pytest.raises(SealedPoint, match="outside the run set"):
        study3_final.gather(freeze, endpoints, runs_root=world.root, repo=world.repo)
    assert order == [] and guard.attempts == []


# ------------------------------------------------------------------ where sealed paths open


def test_files_are_opened_only_in_the_sealed_entries_and_their_private_helpers() -> None:
    """In study3_final.py, ``read_bytes``, ``read_text`` and ``open`` are called only inside a
    function of ``SEALED_ENTRIES`` or a private helper (``_``) that one of them calls."""
    tree = ast.parse(Path(study3_final.__file__).read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def calls(node) -> set[str]:
        found = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                found.add(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
        return found

    opening = {name for name, node in functions.items()
               if calls(node) & {"read_bytes", "read_text", "open"}}  # fmt: skip
    called = set().union(*(calls(functions[name]) for name in study3_final.SEALED_ENTRIES))
    assert opening, "no function opens a file"
    for name in opening:
        helper = name.startswith("_") and name in called
        assert name in study3_final.SEALED_ENTRIES or helper, name
    outside = [node for node in tree.body if not isinstance(node, ast.FunctionDef | ast.ClassDef)]
    assert not set().union(*map(calls, outside)) & {"read_bytes", "read_text", "open"}
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    assert not set().union(*map(calls, classes)) & {"read_bytes", "read_text", "open"}
