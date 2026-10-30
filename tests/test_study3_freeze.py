"""Study 3's analysis freeze (analysis/study3_freeze.py): the cohorts and K*_h frozen once the
confirmation is done and before any confirmation label is opened, and the gate that every label
reader of the final point passes first.

The record is built from the progress tests' synthetic studies (``test_study3_progress.Study``,
the core over records built by hand); the gate runs on a temporary Git checkout, with the study's
state and the in-force supplement given; the builder's reading runs on the progress tests' file
world under the sealed-path guard, every label reader a raiser. Candidate 1's K* is 6.00, its runs
at K_l = 4.8 and K_u = 7.5, so every arm's weight is 1/2."""

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest
from test_study3_campaigns import git
from test_study3_package2 import SENTINEL as PACKAGE_SENTINEL
from test_study3_package2 import LabelReaders, SealedGuard, judged
from test_study3_progress import CODE, EXPECTED, START, WHEN, Study, run_of, search_round0
from test_study3_progress import worlds as worlds  # noqa: F401  (the fixture)
from test_study3_records import SENTINEL as RECORDS_SENTINEL
from test_study3_records import header

from zero_field_refusal.analysis import study3_freeze as fz
from zero_field_refusal.analysis.study3_records import SEEDS

SUPPLEMENT = {"path": "docs/study3-fixation-confirmation-c1.json", "sha256": "e" * 64,
              "root": "repo"}  # fmt: skip


def done(count=60, N=60, **kw) -> Study:
    study = Study()
    study.through_validation()
    study.n(N=N)
    study.confirmation(count=count, **kw)
    return study


def k_harmful(study, value) -> Study:
    study.endpoints = [dataclasses.replace(e, K_harmful=value(e)) if e.K is not None
                       and e.cell.step in ("validation", "confirmation") else e
                       for e in study.endpoints]  # fmt: skip
    return study


def frozen(study):
    return fz.freeze_record(study.advance(), study.endpoints, candidate=1, supplement=SUPPLEMENT,
                            code=CODE, written_at=WHEN,
                            labels={"confirmation_runs": 0, "ledger_runs": 0})  # fmt: skip


def scaled(e) -> float:
    return {"A": 1.0, "F": 1.2, "P": 1.1}[e.arm] * e.K


def test_the_freeze_holds_the_cohorts_their_runs_and_kstar_h() -> None:
    record = frozen(k_harmful(done(), lambda e: {"A": 1.0, "F": 1.2, "P": 1.1}[e.arm] * e.K))
    af, pf = record["comparisons"]["A-F"], record["comparisons"]["P-F"]
    assert (af["reading"], af["rule"], len(af["blocks"])) == ("confirmatory", "n_reached", 60)
    assert [b["seed"] for b in af["blocks"]] == list(SEEDS["confirmation"][1][:60])
    assert set(af["blocks"][0]["arms"]) == {"A", "F"} and set(pf["blocks"][0]["arms"]) == {"P", "F"}
    arm = af["blocks"][0]["arms"]["A"]
    assert arm["w"] == pytest.approx(0.5) and arm["width"] == pytest.approx(math.log(7.5 / 4.8))
    assert record["k_harmful"]["kstar_h"] == pytest.approx(math.sqrt(6.0 * 7.2))
    assert len(record["k_harmful"]["blocks"]) == 60 and record["k_harmful"]["left_out"] == []
    assert len(record["a_minus_p"]["blocks"]) == 60
    assert len(record["validation"]["A-F"]["blocks"]) == 28


def test_the_freeze_is_refused_while_a_started_block_is_open() -> None:
    study = done()
    state = study.advance()  # done; the endpoints it read lose a run after it
    study.drop(run_of(study, "confirmation", "F", "u", 10459))
    with pytest.raises(fz.FreezeRefused, match="started and still open"):
        fz.freeze_record(
            state,
            study.endpoints,
            candidate=1,
            supplement=SUPPLEMENT,
            code=CODE,
            written_at=WHEN,
            labels={"confirmation_runs": 0, "ledger_runs": 0},
        )


def test_the_freeze_is_refused_before_the_confirmation_is_done() -> None:
    with pytest.raises(fz.FreezeRefused, match="done"):
        frozen(done(count=30))


def test_the_record_is_closed_numbers_and_ids() -> None:
    study = k_harmful(done(), scaled)
    record = frozen(study)
    assert tuple(record) == fz.FIELDS and record["kind"] == fz.KIND == "analysis-freeze"
    assert fz.freeze_name(1) == "docs/study3-analysis-freeze-c1.json"
    started = list(SEEDS["confirmation"][1][:60])
    assert record["confirmation"] == {"N": 60, "cap": 70, "started": started, "stop": "counters",
                                      "outcome": "confirmed", "active": ["A-F", "P-F"]}  # fmt: skip
    assert record["supplement"] == SUPPLEMENT and record["code"] == CODE
    a = record["comparisons"]["A-F"]["blocks"][0]["arms"]["A"]
    assert (a["lo"], a["hi"]) == (run_of(study, "confirmation", "A", "l", 10400),
                                  run_of(study, "confirmation", "A", "u", 10400))  # fmt: skip
    assert json.loads(json.dumps(record, allow_nan=False)) == record


@pytest.mark.parametrize("name", ["judge", "judge-superseded"])
def test_a_trace_of_a_confirmation_label_is_a_problem(tmp_path, name) -> None:
    study = done()
    endpoints = [dataclasses.replace(e, run_dir=tmp_path / e.run_id) for e in study.endpoints]
    confirmation = run_of(study, "confirmation", "F", "u", 10403)
    validation = run_of(study, "validation", "F", "u", 10303)
    for run in (confirmation, validation):
        (tmp_path / run).mkdir()
        (tmp_path / run / name).symlink_to(tmp_path / "nowhere")  # dangling: named, not opened
    found = fz.labels_untouched(endpoints, candidate=1, ledger_runs=frozenset())
    assert len(found) == 1 and confirmation in found[0] and f"{name}/" in found[0]
    (tmp_path / confirmation / name).unlink()
    assert fz.labels_untouched(endpoints, candidate=1, ledger_runs=frozenset()) == []
    found = fz.labels_untouched(endpoints, candidate=1, ledger_runs=frozenset({confirmation}))
    assert len(found) == 1 and confirmation in found[0] and "ledger" in found[0]
    assert fz.labels_untouched(endpoints, candidate=1, ledger_runs=frozenset({validation})) == []


def test_a_block_whose_arm_does_not_bracket_kstar_h_is_left_out_and_counted() -> None:
    def value(e) -> float:
        return scaled(e) * (3 if (e.seed, e.arm) == (10400, "F") else 1)

    record = frozen(k_harmful(done(), value))
    assert record["k_harmful"]["left_out"] == [{"seed": 10400, "arms": {"F": "no bracket"}}]
    assert len(record["k_harmful"]["blocks"]) == 59


def test_at_the_cap_a_comparison_below_36_is_descriptive_on_all_its_blocks() -> None:
    seeds = list(SEEDS["confirmation"][1])
    study = done(count=70, special={(s, "P", "u"): {"K": 5.5} for s in seeds[30:70]})
    record = frozen(k_harmful(study, scaled))
    af, pf = record["comparisons"]["A-F"], record["comparisons"]["P-F"]
    assert record["confirmation"]["stop"] == "cap"
    assert (af["reading"], len(af["blocks"]), af["unknown"]) == ("confirmatory", 60, seeds[60:70])
    assert (pf["reading"], pf["rule"], len(pf["blocks"])) == ("descriptive", "cap", 30)
    assert pf["unknown"] == seeds[30:70] and len(pf["unknown"]) == 40
    for c in (af, pf):  # the cohort and the unknown blocks cover each started block once
        assert sorted([*(b["seed"] for b in c["blocks"]), *c["unknown"]]) == sorted(seeds[:70])
    assert [b["seed"] for b in record["a_minus_p"]["blocks"]] == seeds[:30]


def test_a_minus_p_after_p_closed() -> None:
    # P closed at validation: no confirmation block has P cells
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    seeds = SEEDS["validation"][1][:28]
    study.validation(special={(s, "P", "u"): {"K": 5.0} for s in seeds[:7]})
    study.n(N=60, arms=("A", "F"))
    study.confirmation(arms=("A", "F"), count=60)
    record = frozen(k_harmful(study, scaled))
    assert record["a_minus_p"] == {"blocks": []}
    assert set(record["comparisons"]) == {"A-F"} and record["validation"]["P-F"] is None
    # P closed during the confirmation by a closure: the earlier blocks with A and P supported
    # stay; the run under another commit is excluded from every block, as from every count
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.confirmation(count=3)
    run = run_of(study, "confirmation", "P", "l", 10401)
    study.other_commit(run, "other_commit")
    study.closure(run)
    study.confirmation(arms=("A", "F"), count=57, first=3)
    state = study.advance()
    assert (state["stage"], state["active"]["comparisons"]) == ("done", ["A-F"])
    record = frozen(k_harmful(study, scaled))
    assert [b["seed"] for b in record["a_minus_p"]["blocks"]] == [10400, 10402]
    assert len(record["comparisons"]["A-F"]["blocks"]) == 60


# ------------------------------------------------------------------ the gate


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A Git checkout whose HEAD is the freeze's code commit; the study's state and the in-force
    supplement given."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "code.py").write_text("x = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "the code")
    study = k_harmful(done(), scaled)
    code = {"commit": git(repo, "rev-parse", "HEAD"), "gpu_code_sha256": EXPECTED}
    record = fz.freeze_record(study.advance(), study.endpoints, candidate=1,
                              supplement=SUPPLEMENT, code=code, written_at=WHEN,
                              labels={"confirmation_runs": 360, "ledger_runs": 0})  # fmt: skip
    monkeypatch.setattr(fz, "read_state", lambda root, **kw: (study.advance(), study.endpoints))
    monkeypatch.setattr(fz, "in_force_supplement",
                        lambda candidate, **kw: ({}, dict(SUPPLEMENT), []))  # fmt: skip
    world = Gate(repo, tmp_path / "runs", study, record)
    return world


@dataclasses.dataclass
class Gate:
    repo: Path
    runs: Path
    study: Study
    record: dict

    @property
    def path(self) -> Path:
        return self.repo / fz.freeze_name(1)

    def write(self, record=None) -> None:
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(record or self.record, indent=2, sort_keys=True) + "\n")

    def commit(self, *paths: str, message: str = "the analysis freeze") -> None:
        git(self.repo, "add", *(paths or (fz.freeze_name(1),)))
        git(self.repo, "commit", "-qm", message)

    def open(self) -> tuple[str, ...]:
        return fz.freeze_open(1, self.runs, self.repo)


def test_the_gate_opens_only_for_a_record_committed_alone_as_a_child_of_its_code(checkout):
    assert checkout.open() and "analysis-freeze-c1" in checkout.open()[0]  # absent
    checkout.write()
    assert any("not committed" in p for p in checkout.open())
    checkout.commit()
    assert checkout.open() == ()
    checkout.path.write_text(checkout.path.read_text().replace('"N": 60', '"N": 61'))
    assert any("uncommitted change" in p for p in checkout.open())  # changed, uncommitted
    checkout.commit(message="a rewrite")
    assert any("rewritten" in p for p in checkout.open())  # changed by a later commit


def test_the_gate_refuses_a_record_added_off_its_code_commit(checkout) -> None:
    other = {
        **checkout.record,
        "code": {**checkout.record["code"], "commit": git(checkout.repo, "rev-parse", "HEAD")},
    }
    git(checkout.repo, "checkout", "-q", "--orphan", "elsewhere")
    git(checkout.repo, "rm", "-rfq", ".")
    (checkout.repo / "other.txt").write_text("another history\n")
    git(checkout.repo, "add", "other.txt")
    git(checkout.repo, "commit", "-qm", "another root")
    checkout.write(other)
    checkout.commit()
    assert any("does not descend" in p for p in checkout.open())


def test_the_gate_refuses_a_record_committed_with_another_file(checkout) -> None:
    checkout.write()
    (checkout.repo / "docs" / "notes.txt").write_text("notes\n")
    checkout.commit(fz.freeze_name(1), "docs/notes.txt")
    assert any("record-only" in p for p in checkout.open())


def test_the_gate_refuses_a_record_added_twice(checkout) -> None:
    checkout.write()
    checkout.commit()
    git(checkout.repo, "rm", "-q", fz.freeze_name(1))
    git(checkout.repo, "commit", "-qm", "removed")
    checkout.write()
    checkout.commit()
    assert any("added 2 times" in p for p in checkout.open())


def test_the_gate_refuses_a_record_off_its_schema(checkout) -> None:
    checkout.write({**checkout.record, "extra": 1})
    checkout.commit()
    assert any("fields" in p for p in checkout.open())


def other_w(study) -> None:  # a cohort run's K changes: another weight
    study.change(run_of(study, "confirmation", "A", "l", 10400), K=5.0)


def collapsed(study) -> None:  # a cohort run's endpoint state changes
    study.change(run_of(study, "confirmation", "F", "u", 10400), state="unusable",
                 unusable_reason="gross_collapse")  # fmt: skip


def past_the_stop(study) -> None:  # a block added past the stop
    study.confirmation(count=1, first=60)
    k_harmful(study, scaled)


@pytest.mark.parametrize("change", [other_w, collapsed, past_the_stop])
def test_the_gate_checks_the_cohort_before_any_label(checkout, monkeypatch, change) -> None:
    labels = LabelReaders(monkeypatch)
    checkout.write()
    checkout.commit()
    assert checkout.open() == ()
    assert fz.read_freeze(1, runs_root=checkout.runs, repo=checkout.repo) == (
        json.loads(checkout.path.read_text()), [])  # fmt: skip
    change(checkout.study)
    found = checkout.open()
    assert len(found) == 1 and "not the frozen" in found[0], found
    assert fz.read_freeze(1, runs_root=checkout.runs, repo=checkout.repo) == (None, list(found))
    assert labels.calls == []


def test_the_gate_checks_the_in_force_supplement(checkout, monkeypatch) -> None:
    checkout.write()
    checkout.commit()
    changed = {**SUPPLEMENT, "sha256": "f" * 64}
    monkeypatch.setattr(fz, "in_force_supplement", lambda candidate, **kw: ({}, changed, []))
    assert any("supplement" in p for p in checkout.open())
    monkeypatch.setattr(fz, "in_force_supplement",
                        lambda candidate, **kw: (None, None, ["invalid"]))  # fmt: skip
    assert any("supplement" in p and "invalid" in p for p in checkout.open())
    monkeypatch.setattr(fz, "in_force_supplement", lambda candidate, **kw: (None, None, []))
    assert any("supplement" in p for p in checkout.open())


def test_the_in_force_supplement_and_its_link_are_one_reading(tmp_path, monkeypatch) -> None:
    """The record the reader returns and the link's sha256 are of the same bytes: a file that
    no longer parses to the record read is refused, never linked."""
    name = "docs/study3-fixation-confirmation-c1.json"
    (tmp_path / "docs").mkdir()
    data = json.dumps({"bootstrap": {"seed": 7, "resamples": 10000}}).encode()
    (tmp_path / name).write_bytes(data)
    read = {"record": json.loads(data)}
    monkeypatch.setattr(fz, "read_confirmation_fixation",
                        lambda candidate, **kw: (read["record"], []))  # fmt: skip
    record, link, problems = fz.in_force_supplement(1, runs_root=tmp_path, repo=tmp_path)
    assert record == {"bootstrap": {"seed": 7, "resamples": 10000}} and problems == []
    assert link == {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "root": "repo"}
    read["record"] = {"bootstrap": {"seed": 8, "resamples": 10000}}  # the file changed since
    record, link, problems = fz.in_force_supplement(1, runs_root=tmp_path, repo=tmp_path)
    assert (record, link) == (None, None) and "changed while it was read" in problems[0]
    monkeypatch.setattr(fz, "read_confirmation_fixation", lambda candidate, **kw: (None, ["bad"]))
    assert fz.in_force_supplement(1, runs_root=tmp_path, repo=tmp_path) == (None, None, ["bad"])


# ------------------------------------------------------------------ the builder on files


def test_the_freeze_reads_no_label(worlds, monkeypatch) -> None:
    """The builder's reading on files: the state, the supplement (absent) and the traces of a
    label, with a judged confirmation run among the runs. Nothing sealed is opened, no label
    reader is called, and the refusal names the stage and the trace, never a label."""
    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    search_round0(records, runs)
    run = runs.run("A", START["A"]["l"], 10400, role="confirmation",
                   head=header(1, "confirmation", None),
                   campaign="study3-c1-confirmation-b10400")  # fmt: skip
    judged(runs.root / "study3" / run)
    labels = LabelReaders(monkeypatch)
    guard = SealedGuard(monkeypatch, runs_root=runs.root)
    with pytest.raises(fz.FreezeRefused) as caught:
        fz.analysis_freeze(runs.root, 1, code=CODE, repo=records.repo, theta0=runs.theta0,
                           ledger_runs=frozenset({run}))  # fmt: skip
    why = str(caught.value)
    assert "done" in why and f"{run}: its folder holds judge/" in why and "ledger" in why
    assert "supplement" in why
    assert guard.attempts == [] and labels.calls == []
    assert RECORDS_SENTINEL not in why and PACKAGE_SENTINEL not in why


def test_the_freeze_is_refused_when_the_n_record_cohort_differs(checkout, monkeypatch) -> None:
    monkeypatch.setattr(fz, "labels_untouched", lambda *a, **kw: [])
    seeds = [b["seed"] for b in checkout.record["validation"]["A-F"]["blocks"]]

    def n_record(cohort):
        return lambda *a, **kw: {
            "n-c1": {"state": "passed", "record": {"cohort": {"seeds": cohort}}}
        }

    monkeypatch.setattr(fz, "decisions", n_record(seeds))
    built = fz.analysis_freeze(checkout.runs, 1, code=CODE, repo=checkout.repo, ledger_runs=())
    assert [b["seed"] for b in built["validation"]["A-F"]["blocks"]] == seeds
    monkeypatch.setattr(fz, "decisions", n_record(seeds[1:]))
    with pytest.raises(fz.FreezeRefused, match="differ from the n-c1"):
        fz.analysis_freeze(checkout.runs, 1, code=CODE, repo=checkout.repo, ledger_runs=())
