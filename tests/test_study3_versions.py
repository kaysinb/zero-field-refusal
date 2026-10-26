"""Study 3's versions record, its staged fixation and the journal check
(analysis/study3_versions.py), on a checkout made for each test (Git), a synthetic RUNS_DIR with
the decision records, the qualification's evidence and launch journals whose campaign headers name
a versions record, and run directories whose regular runs hold a harmful-response file and an
observations file with a sentinel text (never opened).

The qualification's reading is ``qualification_records.read_qualification``'s, replaced by a
reading of given rate states in every test but one, which reads a real qualification as package
1's ``qualify.py`` writes it."""

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_qualification_records import qualify as qualify  # noqa: F401  (the fixture)
from test_study3_decisions import World as DecisionWorld
from test_study3_decisions import coverage_row, gates_with, pass_rate, seal, verified
from test_study3_records import SENTINEL
from test_study3_records import World as RunWorld

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import qualification_records as qr
from zero_field_refusal.analysis import study3_versions as sv
from zero_field_refusal.analysis.study3_decisions import cap_of
from zero_field_refusal.analysis.study3_records import SEEDS

REGISTRATION = "docs/study3-preregistration.md"
AMENDMENTS = "".join(f"\n### Amendment {n}: change {n}\n\nText {n}.\n" for n in (1, 2, 3))
RATES = {"A": {"lr_l": 1.45e-5, "lr_u": 2.11e-5}, "F": {"lr_l": 1.30e-5, "lr_u": 1.83e-5},
         "P": {"lr_l": 1.30e-5, "lr_u": 1.85e-5}}  # fmt: skip
OPT = {"A": "adamw", "F": "adam_frozen", "P": "adamw_prewarmed"}
VALIDATION_C1 = "docs/study3-fixation-validation-c1.json"
CONFIRMATION_C1 = "docs/study3-fixation-confirmation-c1.json"
FIRST, SECOND = "docs/study3-versions.json", "docs/study3-versions-2.json"


def git(repo: Path, *args: str) -> str:
    command = ["git", "-c", "user.name=test", "-c", "user.email=test@example.org",
               "-c", "commit.gpgsign=false", *args]  # fmt: skip
    return subprocess.run(command, cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has(problems, *words) -> bool:
    """Some problem names every word."""
    return any(all(w in p for w in words) for p in problems)


class Study:
    """A checkout (Git: the registration, a GPU-code file) and its RUNS_DIR: the decision
    records, the qualification's evidence (its launch log and verdict file), the runs and their
    journal. The qualification's reading gives ``states`` per (arm, rate)."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.records = DecisionWorld(tmp_path)
        shutil.rmtree(self.records.run)  # the records tests' own sentinel run: no run here
        self.root, self.repo = self.records.runs, self.records.repo
        self.runs = RunWorld(self.root)
        git(self.repo, "init", "-q")
        (self.repo / ".gitignore").write_text("/evidence/\n")  # the decision records' evidence
        (self.repo / "docs").mkdir()
        (self.repo / REGISTRATION).write_text("# Study 3 preregistration\n" + AMENDMENTS)
        code = self.repo / "src" / "zero_field_refusal" / "train.py"
        code.parent.mkdir(parents=True)
        code.write_text("x = 1\n")
        self.commit("the registration and the code")
        self.registered = self.head()
        self.code = self.now()
        self.states = {(arm, lr): "qualified" for arm, pair in RATES.items()
                       for lr in pair.values()}  # fmt: skip
        self.reading_problems: list[str] = []
        self.calls: list[tuple[int, str]] = []
        monkeypatch.setattr(qr, "read_qualification", self.reading)

    # -- the checkout

    def head(self) -> str:
        return git(self.repo, "rev-parse", "HEAD")

    def now(self) -> dict:
        """The code of HEAD: its commit and its GPU code."""
        head = self.head()
        return {
            "commit": head,
            "gpu_code_sha256": code_identity.gpu_code_identity(head, self.repo)["sha256"],
        }

    def commit(self, message: str, *paths: str) -> None:
        git(self.repo, "add", *(paths or ("-A",)))
        git(self.repo, "commit", "-qm", message)

    def put(self, name: str, record, *, commit: bool = True) -> Path:
        path = self.repo / name
        path.write_text(json.dumps(record, indent=2) + "\n")
        if commit:
            self.commit(f"add {name}", name)
        return path

    def edit(self, name: str, **fields) -> None:
        path = self.repo / name
        path.write_text(json.dumps({**json.loads(path.read_text()), **fields}, indent=2) + "\n")

    # -- the qualification

    def reading(self, runs_root, candidate, *, expected_code, settings) -> dict:
        self.calls.append((candidate, expected_code))
        rates = {}
        for (arm, lr), state in self.states.items():
            key = f"{OPT[arm]}-{lr:g}"
            rates[key] = {"key": key, "arm": arm, "optimizer": OPT[arm], "lr": lr, "state": state}
        return {"rates": rates, "states": {k: r["state"] for k, r in rates.items()},
                "problems": list(self.reading_problems)}  # fmt: skip

    def qualified(self, candidate: int = 1) -> None:
        """The qualification's evidence: its campaign's launch log (a start and an end per
        case) and its verdict file."""
        campaign = f"study3-c{candidate}-numerical"
        head = {"study": 3, "candidate": candidate, "step": "numerical", "round": None}
        for arm, pair in RATES.items():
            for lr in pair.values():
                for seed in SEEDS["numerical"][candidate]:
                    label, run_id = f"{arm}-{lr:g}/s{seed}", f"q{candidate}-{arm}-{lr:g}-{seed}"
                    common = {"run": label, "attempt": 1, "run_id": run_id}
                    self.runs.log(campaign, {"event": "start", **common, "experiment": "study3",
                                             "config_hash": "c" * 12, "header": head})  # fmt: skip
                    self.runs.log(campaign, {"event": "end", **common, "outcome": "done"})
        verdict = self.root / "qualification" / f"c{candidate}" / "verdict.json"
        verdict.parent.mkdir(parents=True, exist_ok=True)
        verdict.write_text(json.dumps({"candidate": candidate, "stage": "numerical"}))

    def decided(self, **simulation) -> None:
        self.records.write("engineering", self.records.engineering(code=self.code))
        self.records.write("simulation", self.records.simulation(code=self.code, **simulation))

    def ready(self, **simulation) -> None:
        self.decided(**simulation)
        self.qualified()

    # -- the records

    def fixation(self, candidate: int = 1, **kw) -> dict:
        return sv.validation_fixation(self.root, candidate, rates=RATES, code=self.code,
                                      repo=self.repo, **kw)  # fmt: skip

    def read_validation(self, candidate: int = 1):
        return sv.read_validation_fixation(candidate, runs_root=self.root, repo=self.repo)

    def read_confirmation(self, candidate: int = 1):
        return sv.read_confirmation_fixation(candidate, runs_root=self.root, repo=self.repo)

    def n(self, N: int = 75, **budget) -> None:
        """The n record (and, with ``budget`` fields, its budget record)."""
        self.records.write("n-c1", self.records.n(1, N, code=self.code))
        if budget:
            self.records.write("budget-c1", self.records.budget(1, N, **budget))

    def supplement(self) -> dict:
        return sv.confirmation_fixation(self.root, 1, code=self.now(), repo=self.repo)

    def versions(self, name: str = FIRST, fixation: str | None = VALIDATION_C1,
                 reason: str = "validation_c1", **kw) -> str:  # fmt: skip
        """A versions record built, written and committed; its sha256."""
        record = sv.versions_record(name, runs_root=self.root, fixation=fixation, reason=reason,
                                    repo=self.repo, **kw)  # fmt: skip
        return sha256(self.put(name, record))

    def validated(self) -> str:
        """Candidate 1's validation fixation and the first versions record, committed: the
        latter's sha256."""
        self.ready()
        self.put(VALIDATION_C1, self.fixation())
        return self.versions()

    def problems(self, name: str = FIRST, **kw) -> list[str]:
        return sv.versions_problems(self.repo / name, runs_root=self.root, repo=self.repo, **kw)

    def block(self, seed: int, *, step: str = "validation", record: str = FIRST,
              sha: str | None = None, candidate: int = 1, arm: str = "A") -> list[str]:  # fmt: skip
        """The runs of ``arm`` in one block, their campaign header naming ``record`` (by its
        sha256 now unless ``sha``); their run ids."""
        head = {"study": 3, "candidate": candidate, "step": step, "round": None,
                "versions_record": record,
                "versions_record_sha256": sha or sha256(self.repo / record)}  # fmt: skip
        return [self.runs.run(arm, RATES[arm][label], seed, role=step, candidate=candidate,
                              campaign=f"study3-c{candidate}-{step}-{seed}", prescribe=False,
                              head=head)
                for label in ("lr_l", "lr_u")]  # fmt: skip


@pytest.fixture
def study(tmp_path, monkeypatch) -> Study:
    return Study(tmp_path, monkeypatch)


# ------------------------------------------------------------------ the validation fixation


def test_the_validation_fixation_is_built_and_read(study) -> None:
    study.ready()
    record = study.fixation(written_at="2026-10-27T12:00:00+00:00")
    assert study.calls == [(1, study.code["gpu_code_sha256"])]  # under the code in force
    assert record == {
        "kind": "fixation-validation",
        "candidate": 1,
        "written_at": "2026-10-27T12:00:00+00:00",
        "policy": {"path": REGISTRATION, "commit": study.registered,
                   "sha256": sha256(study.repo / REGISTRATION), "amendments": [1, 2, 3]},
        "rates": RATES,
        "qualification": {
            "verdict": {"path": "qualification/c1/verdict.json",
                        "sha256": sha256(study.root / "qualification" / "c1" / "verdict.json")},
            "logs": [{"path": "launch/study3-c1-numerical.jsonl",
                      "sha256": sha256(study.root / "launch" / "study3-c1-numerical.jsonl")}],
            "states": {arm: {"lr_l": "qualified", "lr_u": "qualified"} for arm in RATES},
        },
        "simulation": {"path": "study3/decisions/simulation.json",
                       "sha256": study.records.digest("simulation")},
        "validation_blocks": 28,
        "level": 0.975,
        "engineering": {"path": "study3/decisions/engineering.json",
                        "sha256": study.records.digest("engineering")},
        "model": record["model"],
        "judge": record["judge"],
        "code": study.code,
    }  # fmt: skip
    assert record["model"]["name"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert record["judge"]["config"] == "judge/wildguard"
    assert json.loads(json.dumps(record, allow_nan=False)) == record  # a JSON record
    # P closed before validation: its rates are not fixed
    without_p = {arm: RATES[arm] for arm in ("A", "F")}
    built = sv.validation_fixation(study.root, 1, rates=without_p, code=study.code,
                                   repo=study.repo)  # fmt: skip
    assert set(built["rates"]) == set(built["qualification"]["states"]) == {"A", "F"}
    assert study.read_validation() == (None, [])  # not written: absent
    study.put(VALIDATION_C1, record, commit=False)
    assert study.read_validation() == (record, [])
    study.put(VALIDATION_C1, built, commit=False)
    assert study.read_validation() == (built, [])


@pytest.mark.parametrize("state", ["failing", "incomplete", "not_qualified", "technical", None])
def test_a_rate_that_is_not_qualified_refuses_the_fixation(study, state) -> None:
    study.ready()
    if state is None:
        del study.states[("F", 1.83e-5)]  # the qualification names no such rate
    else:
        study.states[("F", 1.83e-5)] = state
    with pytest.raises(sv.RecordRefused) as error:
        study.fixation()
    assert "F" in str(error.value) and "1.83e-05" in str(error.value)
    assert (state or "not in the qualification") in str(error.value)


def test_a_qualification_with_problems_refuses_the_fixation(study) -> None:
    study.ready()
    study.reading_problems = ["qualification/c1/verdict.json: an unreadable verdict cache"]
    with pytest.raises(sv.RecordRefused, match="an unreadable verdict cache"):
        study.fixation()


def test_a_fixed_rate_must_be_written_as_a_rate(study) -> None:
    study.ready()
    for rates in ({"A": RATES["A"]}, {**RATES, "Q": RATES["A"]},
                  {**RATES, "A": {"lr_l": -1.0, "lr_u": 2e-5}}):  # fmt: skip
        with pytest.raises(sv.RecordRefused, match="rates"):
            sv.validation_fixation(study.root, 1, rates=rates, code=study.code, repo=study.repo)


@pytest.mark.parametrize("missing", ["simulation", "engineering"])
def test_a_missing_decision_record_refuses_the_fixation(study, missing) -> None:
    study.qualified()
    if missing == "simulation":
        study.records.write("engineering", study.records.engineering(code=study.code))
    with pytest.raises(sv.RecordRefused, match=f"the {missing} record"):
        study.fixation()


def test_a_failed_simulation_record_refuses_the_fixation(study) -> None:
    study.ready(state="failed", verification=verified(passed=False))
    with pytest.raises(sv.RecordRefused, match="the simulation record did not pass"):
        study.fixation()


def test_a_fixation_resting_on_a_simulation_that_failed_is_refused_on_reading(study) -> None:
    study.ready()
    record = study.fixation()
    study.put(VALIDATION_C1, record, commit=False)
    path = study.root / "study3" / "decisions" / "simulation.json"
    failed = study.records.simulation(code=study.code, state="failed",
                                      verification=verified(passed=False))  # fmt: skip
    path.write_text(json.dumps(failed))
    study.edit(VALIDATION_C1, simulation={"path": "study3/decisions/simulation.json",
                                          "sha256": sha256(path)})  # fmt: skip
    found, problems = study.read_validation()
    assert found is None and has(problems, "the simulation record did not pass")


def test_a_fixation_resting_on_an_engineering_record_that_failed_is_refused_on_reading(study):
    study.ready()
    record = study.fixation()
    study.put(VALIDATION_C1, record, commit=False)
    path = study.root / "study3" / "decisions" / "engineering.json"
    path.write_text(json.dumps(study.records.engineering(code=study.code, state="failed")))
    study.edit(VALIDATION_C1, engineering={"path": "study3/decisions/engineering.json",
                                           "sha256": sha256(path)})  # fmt: skip
    found, problems = study.read_validation()
    assert found is None and problems == [
        f"{study.repo / VALIDATION_C1}: engineering: the engineering record did not pass"
    ]


def test_the_validation_size_and_level_are_the_simulation_records(study) -> None:
    rows = [coverage_row(0.975, passed=False), coverage_row(0.980)]
    study.ready(coverage=rows, level=0.98, validation_blocks=40, verification=verified(0.98),
                gates=gates_with(safe_pass_rate=pass_rate(0.93)))  # fmt: skip
    record = study.fixation()
    assert (record["validation_blocks"], record["level"]) == (40, 0.98)
    study.put(VALIDATION_C1, record, commit=False)
    assert study.read_validation() == (record, [])
    for field, value, words in (("validation_blocks", 28, ("validation_blocks", "40")),
                                ("level", 0.975, ("level", "0.98"))):  # fmt: skip
        study.edit(VALIDATION_C1, **{field: value})
        found, problems = study.read_validation()
        assert found is None and has(problems, *words), problems
        study.put(VALIDATION_C1, record, commit=False)
    for field, value in (("validation_blocks", 30), ("level", 0.97)):  # off their sets
        study.edit(VALIDATION_C1, **{field: value})
        found, problems = study.read_validation()
        assert found is None and has(problems, field), problems
        study.put(VALIDATION_C1, record, commit=False)


@pytest.mark.parametrize("evidence", ["verdict", "log", "simulation", "engineering", "new log"])
def test_an_evidence_file_changed_after_the_fixation_is_refused(study, evidence) -> None:
    study.ready()
    record = study.fixation()
    study.put(VALIDATION_C1, record, commit=False)
    assert study.read_validation() == (record, [])
    if evidence == "verdict":
        path = study.root / "qualification" / "c1" / "verdict.json"
        path.write_text(path.read_text() + " ")
        words = ("qualification.verdict", "sha256")
    elif evidence == "log":  # the qualification's campaign run once more
        study.runs.log("study3-c1-numerical", {"event": "recovery", "experiment": "study3",
                                               "cell": "A-1.45e-05", "runs": ["A-1.45e-05/s10200"],
                                               "reason": "the Pod stopped"})  # fmt: skip
        words = ("qualification.logs", "sha256")
    elif evidence == "new log":  # a line of another log names a qualification run
        study.runs.log(
            "study3-c1-other",
            {
                "event": "superseded",
                "run_id": "q1-A-1.45e-05-10200",
                "run": "A-1.45e-05/s10200",
                "experiment": "study3",
            },
        )
        words = ("launch/study3-c1-other.jsonl", "does not pin")
    else:
        path = study.root / "study3" / "decisions" / f"{evidence}.json"
        path.write_text(path.read_text().replace("2026-10-26T12:00:00Z", "2026-10-26T13:00:00Z"))
        words = (evidence, "sha256")
    found, problems = study.read_validation()
    assert found is None and has(problems, *words), problems


def test_the_policy_is_the_registration_at_its_commit(study) -> None:
    study.ready()
    record = study.fixation()
    assert record["policy"]["commit"] == study.registered
    # an amendment registered later: the policy is the registration at the later commit
    with (study.repo / REGISTRATION).open("a") as f:
        f.write("\n### Amendment 4: change 4\n\nText 4.\n")
    with pytest.raises(sv.RecordRefused, match="uncommitted"):
        study.fixation()  # the registration in the checkout is not a committed one
    study.commit("amendment 4")
    later = study.fixation()
    assert later["policy"]["commit"] == study.head()
    assert later["policy"]["amendments"] == [1, 2, 3, 4]
    assert later["policy"]["sha256"] == sha256(study.repo / REGISTRATION)
    study.put(VALIDATION_C1, later, commit=False)
    assert study.read_validation() == (later, [])
    for change, words in (
        ({"amendments": [1, 2, 3]}, ("policy", "amendments", "[1, 2, 3, 4]")),
        ({"sha256": record["policy"]["sha256"]}, ("policy", "sha256")),
        ({"commit": "0" * 40}, ("policy", "cannot be read")),
        ({"path": "docs/other.md"}, ("policy.path",)),
        ({"amendments": [1, 1, 2]}, ("policy.amendments",)),
    ):  # fmt: skip
        study.edit(VALIDATION_C1, policy={**later["policy"], **change})
        found, problems = study.read_validation()
        assert found is None and has(problems, *words), (change, problems)


def test_the_fixation_carries_the_code_in_force(study, monkeypatch) -> None:
    study.ready()
    other = {"commit": study.code["commit"], "gpu_code_sha256": "e" * 64}
    with pytest.raises(sv.RecordRefused, match="GPU code"):
        sv.validation_fixation(study.root, 1, rates=RATES, code=other, repo=study.repo)
    record = study.fixation()
    study.put(VALIDATION_C1, record, commit=False)
    # a commit this checkout does not hold
    study.edit(VALIDATION_C1, code={**study.code, "commit": "0" * 40})
    found, problems = study.read_validation()
    assert found is None and has(problems, "code", "0" * 40)
    study.put(VALIDATION_C1, record, commit=False)
    # the code in force, but not the GPU code of the commit it names
    real = code_identity.gpu_code_identity
    with monkeypatch.context() as patch:
        patch.setattr(code_identity, "gpu_code_identity", lambda commit="HEAD", repo=None: {
            **real(commit, repo), "sha256": "d" * 64})  # fmt: skip
        found, problems = study.read_validation()
    assert found is None and problems == [
        f"{study.repo / VALIDATION_C1}: code: the GPU code of commit {study.code['commit']} is "
        f"{'d' * 64}, not the recorded {study.code['gpu_code_sha256']}"]  # fmt: skip
    assert study.read_validation() == (record, [])
    # an engineering cycle moved the code in force for candidate 1 since
    cycle = study.root / "qualification" / "c1" / "supersede-1.json"
    cycle.write_text(json.dumps({"candidate": 1, "old_gpu_code": {"sha256": record["code"][
        "gpu_code_sha256"]}, "fix_gpu_code": {"sha256": "f" * 64}, "held": []}))  # fmt: skip
    found, problems = study.read_validation()
    assert found is None and has(problems, "in force", "f" * 64), problems


def test_a_fixation_record_has_a_closed_schema(study) -> None:
    study.ready()
    record = study.fixation()
    for change, words in (
        ({"mean_loss": 1.0}, ("mean_loss", "not a field")),
        ({"kind": "fixation-confirmation"}, ("kind",)),
        ({"candidate": 2}, ("candidate",)),
        ({"written_at": "2026-10-27T12:00:00+02:00"}, ("written_at", "UTC")),
        ({"qualification": {**record["qualification"],
                            "states": {**record["qualification"]["states"],
                                       "F": {"lr_l": "qualified", "lr_u": "failing"}}}},
         ("qualification.states.F.lr_u", "qualified")),
        ({"qualification": {**record["qualification"], "logs": []}}, ("qualification.logs",)),
        ({"qualification": {**record["qualification"],
                            "verdict": {"path": "qualification/c2/verdict.json",
                                        "sha256": "0" * 64}}}, ("qualification.verdict",)),
        ({"simulation": {"path": "evidence/simulation.json", "sha256": "0" * 64}},
         ("simulation",)),
        ({"model": {**record["model"], "revision": "main"}}, ("model",)),
        ({"judge": {**record["judge"], "judge_revision": "other"}}, ("judge",)),
        ({"code": {"commit": "x"}}, ("code",)),
    ):  # fmt: skip
        study.put(VALIDATION_C1, {**record, **change}, commit=False)
        found, problems = study.read_validation()
        assert found is None and has(problems, *words), (change, problems)
    unread = dict(record)
    del unread["policy"]
    study.put(VALIDATION_C1, unread, commit=False)
    assert has(study.read_validation()[1], "policy")
    (study.repo / VALIDATION_C1).write_text("{")
    assert has(study.read_validation()[1], "unreadable")


def test_a_fixation_is_written_once(study) -> None:
    study.ready()
    study.put(VALIDATION_C1, study.fixation())
    with pytest.raises(sv.RecordRefused, match="already written"):
        study.fixation()


# ------------------------------------------------------------------ the confirmation supplement


def test_the_confirmation_supplement_without_a_budget_record(study) -> None:
    study.validated()
    with pytest.raises(sv.RecordRefused, match="n-c1"):
        study.supplement()  # no n record yet
    study.n(75)
    record = study.supplement()
    assert record == {
        "kind": "fixation-confirmation",
        "candidate": 1,
        "written_at": record["written_at"],
        "validation_fixation": {"path": VALIDATION_C1, "sha256": sha256(study.repo / VALIDATION_C1),
                                "root": "repo"},
        "n": {"path": "study3/decisions/n-c1.json", "sha256": study.records.digest("n-c1")},
        "N": 75,
        "cap": cap_of(75),
        "budget": None,
        "code": study.now(),
    }  # fmt: skip
    study.put(CONFIRMATION_C1, record, commit=False)
    assert study.read_confirmation() == (record, [])
    # a budget record named where none is needed is refused
    study.records.write("budget-c1", study.records.budget(1, 75))
    study.edit(CONFIRMATION_C1, budget={"path": "study3/decisions/budget-c1.json",
                                        "sha256": study.records.digest("budget-c1")})  # fmt: skip
    found, problems = study.read_confirmation()
    assert found is None and has(problems, "budget", "null"), problems


def test_the_confirmation_supplement_with_a_budget_record(study) -> None:
    study.validated()
    study.records.write("n-c1", study.records.n(1, 99, code=study.code))
    with pytest.raises(sv.RecordRefused, match="budget-c1"):
        study.supplement()  # N = 99 needs the budget decision
    study.records.write("budget-c1", study.records.budget(1, 99))
    record = study.supplement()
    assert (record["N"], record["cap"]) == (99, cap_of(99))
    assert record["budget"] == {"path": "study3/decisions/budget-c1.json",
                                "sha256": study.records.digest("budget-c1")}  # fmt: skip
    study.put(CONFIRMATION_C1, record, commit=False)
    assert study.read_confirmation() == (record, [])
    study.edit(CONFIRMATION_C1, budget=None)
    found, problems = study.read_confirmation()
    assert found is None and has(problems, "budget", "needed"), problems


def test_an_n_or_cap_other_than_the_n_records_is_refused(study) -> None:
    study.validated()
    study.n(75)
    record = study.supplement()
    study.put(CONFIRMATION_C1, record, commit=False)
    for change, words in (({"N": 76, "cap": cap_of(76)}, ("N 76", "75")),
                          ({"cap": 86}, ("cap",)),
                          ({"N": 59, "cap": cap_of(59)}, ("N",))):  # fmt: skip
        study.edit(CONFIRMATION_C1, **change)
        found, problems = study.read_confirmation()
        assert found is None and has(problems, *words), (change, problems)
        study.put(CONFIRMATION_C1, record, commit=False)


def test_an_n_record_that_did_not_pass_refuses_the_supplement(study) -> None:
    """A cap beyond the 200 reserved confirmation seeds: the n record failed, and the
    confirmation cannot start."""
    study.validated()
    study.records.write("n-c1", study.records.n(1, 179, code=study.code, state="failed"))
    with pytest.raises(sv.RecordRefused, match="the n-c1 record did not pass"):
        study.supplement()


def test_a_budget_decision_to_stop_refuses_the_supplement(study) -> None:
    study.validated()
    study.n(99, decision="stop", state="failed")
    with pytest.raises(sv.RecordRefused, match="stop"):
        study.supplement()
    record = {"kind": "fixation-confirmation", "candidate": 1,
              "written_at": "2026-10-27T12:00:00+00:00",
              "validation_fixation": {"path": VALIDATION_C1,
                                      "sha256": sha256(study.repo / VALIDATION_C1),
                                      "root": "repo"},
              "n": {"path": "study3/decisions/n-c1.json", "sha256": study.records.digest("n-c1")},
              "N": 99, "cap": cap_of(99),
              "budget": {"path": "study3/decisions/budget-c1.json",
                         "sha256": study.records.digest("budget-c1")},
              "code": study.now()}  # fmt: skip
    study.put(CONFIRMATION_C1, record, commit=False)
    found, problems = study.read_confirmation()
    assert found is None and has(problems, "budget", "stop"), problems


def test_the_supplement_rests_on_the_validation_fixation_and_its_n_record(study) -> None:
    study.ready()
    study.n(75)
    with pytest.raises(sv.RecordRefused, match="validation fixation"):
        study.supplement()  # no validation fixation
    study.put(VALIDATION_C1, study.fixation())
    record = study.supplement()
    study.put(CONFIRMATION_C1, record, commit=False)
    assert study.read_confirmation() == (record, [])
    # the validation fixation changed since
    study.edit(VALIDATION_C1, written_at="2026-10-27T13:00:00+00:00")
    found, problems = study.read_confirmation()
    assert found is None and has(problems, "validation_fixation", "sha256"), problems
    git(study.repo, "checkout", "-q", "--", VALIDATION_C1)
    assert study.read_confirmation() == (record, [])
    # an n record of other rates or of another GPU code than the fixation's
    for over, words in (({"rates": {**study.records.n(1, 75)["rates"],
                                    "A": {"lr_l": 1.5e-5, "lr_u": 2.11e-5}}}, ("rates", "A")),
                        ({"code": {**study.code, "gpu_code_sha256": "e" * 64}},
                         ("GPU code",))):  # fmt: skip
        path = study.root / "study3" / "decisions" / "n-c1.json"
        path.unlink()
        study.records.write("n-c1", study.records.n(1, 75, **{"code": study.code, **over}))
        study.edit(CONFIRMATION_C1, n={"path": "study3/decisions/n-c1.json",
                                       "sha256": study.records.digest("n-c1")})  # fmt: skip
        found, problems = study.read_confirmation()
        assert found is None and has(problems, *words), (over, problems)


# ------------------------------------------------------------------ the versions record


def test_the_versions_record_is_enforced_against_the_checkout(study) -> None:
    study.ready()
    study.put(VALIDATION_C1, study.fixation())
    assert study.problems() == [
        f"{study.repo / FIRST} is missing: write the versions record "
        "(scripts/study3_summary.py versions) and commit it first"
    ]
    record = sv.versions_record(FIRST, runs_root=study.root, fixation=VALIDATION_C1,
                                reason="validation_c1", repo=study.repo)  # fmt: skip
    assert record["code"] == {"commit": study.head()}
    assert record["fixation"] == {"path": VALIDATION_C1,
                                  "sha256": sha256(study.repo / VALIDATION_C1)}  # fmt: skip
    assert (record["supersedes"], record["reason"]) == (None, "validation_c1")
    study.put(FIRST, record, commit=False)
    # written, not committed: refused
    assert has(study.problems(), "is not committed")
    study.commit("the versions record", FIRST)
    assert study.problems() == []  # a record-only child of the recorded commit
    assert sv.read_versions(study.repo / FIRST, repo=study.repo) == (record, [])
    # an untracked file is refused; an ignored one is not
    (study.repo / "notes.md").write_text("local notes\n")
    assert study.problems() == ["the checkout has 1 untracked files (first: notes.md)"]
    (study.repo / "notes.md").unlink()
    (study.repo / "evidence").mkdir(exist_ok=True)
    (study.repo / "evidence" / "x.json").write_text("{}\n")
    assert study.problems() == []
    # an uncommitted change, then another commit
    code = study.repo / "src" / "zero_field_refusal" / "train.py"
    code.write_text("x = 2\n")
    assert has(study.problems(), "1 uncommitted changes", "train.py")
    study.commit("a code change")
    assert has(study.problems(), "differs from the recorded commit", "(first: "
               "src/zero_field_refusal/train.py)")  # fmt: skip
    # the code back as it was: the tree is the recorded commit's and the record, as study 2 reads
    # a record-only child
    code.write_text("x = 1\n")
    study.commit("the code back")
    assert study.problems() == []


def test_a_record_changed_after_it_was_added_is_refused(study) -> None:
    study.validated()
    assert study.problems() == []
    study.edit(FIRST, reason="validation_c1_again")
    assert has(study.problems(), "differs from its committed version at HEAD")
    study.commit("the record edited")
    assert has(study.problems(), "changed after it was added")
    assert has(study.problems(), "is never overwritten")


def test_a_validation_run_without_its_start_fails_closed(study) -> None:
    study.validated()
    (run_id, other) = study.block(10300)
    assert study.problems() == []
    lost = study.root / "study3" / run_id
    # its start lost from the journal: the log that held it restored without it
    log = study.root / "launch" / "study3-c1-validation-10300.jsonl"
    log.write_text("".join(line + "\n" for line in log.read_text().splitlines()
                           if run_id not in line))  # fmt: skip
    found = study.problems()
    assert has(found, run_id, "no start in the launch journal", "restore"), found
    assert not has(found, other)
    # a run directory whose role cannot be read is refused alike (fails closed)
    (lost / "config.yaml").unlink()
    (lost / "meta.json").unlink()
    assert has(study.problems(), run_id, "no start in the launch journal")
    # a K-only run without a start is not a run of this record (the endpoint reader refuses it)
    shutil.rmtree(lost)
    search = study.runs.run(
        "A",
        1.45e-5,
        10100,
        role="search",
        round_=0,
        prescribe=False,
        campaign="study3-c1-search",
        head={"study": 3, "candidate": 1, "step": "search", "round": 0},
    )
    shutil.copytree(study.root / "study3" / search, study.root / "study3" / "orphan-search")
    assert study.problems() == []


def test_a_header_naming_another_record_or_sha256_is_refused(study) -> None:
    digest = study.validated()
    study.block(10300)
    assert study.problems() == []
    study.block(10301, record="docs/study3-versions-7.json", sha="7" * 64)
    study.block(10302, sha="0" * 64)
    # a header naming the record from another machine's checkout names the record
    study.block(10303, record=f"/workspace/zero-field-refusal/{FIRST}", sha=digest)
    # a validation run whose header names no record
    study.runs.run(
        "A",
        1.45e-5,
        10304,
        role="validation",
        prescribe=False,
        campaign="study3-c1-validation-10304",
        head={"study": 3, "candidate": 1, "step": "validation", "round": None},
    )
    found = study.problems()
    assert has(found, "study3-c1-validation-10301", "docs/study3-versions-7.json",
               "outside"), found  # fmt: skip
    assert has(found, "study3-c1-validation-10302", "0" * 64, digest), found
    assert has(found, "study3-c1-validation-10304", "names no versions record"), found
    assert not has(found, "study3-c1-validation-10303") and not has(found, "validation-10300:")
    assert len(found) == 3, found
    # a study-3 start whose header names no study-3 step: whether it is a validation run cannot
    # be shown
    for name, head in (("no-header", None), ("other-study", {"study": 2, "step": "validation"}),
                       ("no-step", {"study": 3, "candidate": 1, "round": None})):  # fmt: skip
        study.runs.log(
            name,
            {
                "event": "start",
                "run": "A-1.45e-05/s10305",
                "attempt": 1,
                "run_id": f"r-{name}",
                "experiment": "study3",
                "header": head,
            },
        )
        assert has(study.problems(), f"{name}: A-1.45e-05/s10305 (r-{name})",
                   "names no study-3 step"), name  # fmt: skip


def test_the_succession_of_versions_records(study) -> None:
    study.validated()
    study.n(75)
    study.put(CONFIRMATION_C1, study.supplement())
    # a later record must name the one before it
    with pytest.raises(sv.RecordRefused, match="supersedes"):
        study.versions(SECOND, CONFIRMATION_C1, "confirmation_c1")
    with pytest.raises(sv.RecordRefused, match="docs/study3-versions-2.json"):
        study.versions(
            "docs/study3-versions-3.json",
            CONFIRMATION_C1,
            "confirmation_c1",
            supersedes_reason="next_stage",
        )  # the one before it is not written
    record = sv.versions_record(SECOND, runs_root=study.root, fixation=CONFIRMATION_C1,
                                reason="confirmation_c1", supersedes_reason="next_stage",
                                repo=study.repo)  # fmt: skip
    assert record["supersedes"] == {"path": FIRST, "sha256": sha256(study.repo / FIRST),
                                    "reason": "next_stage"}  # fmt: skip
    # a later record without supersedes, written by hand, fails closed
    study.put(SECOND, {**record, "supersedes": None})
    found = study.problems(SECOND)
    assert has(found, SECOND, "later versions record", "must name the one before it",
               FIRST), found  # fmt: skip
    git(study.repo, "reset", "-q", "--hard", "HEAD~1")
    for change, words in (
        ({"supersedes": {**record["supersedes"], "path": "docs/study3-versions-3.json"}},
         ("supersedes", FIRST)),
        ({"supersedes": {**record["supersedes"], "sha256": "0" * 64}}, ("supersedes", "sha256")),
        ({"supersedes": {**record["supersedes"], "reason": "A sentence."}},
         ("supersedes.reason",)),
    ):  # fmt: skip
        study.put(SECOND, {**record, **change})
        found = study.problems(SECOND)
        assert has(found, *words), (change, found)
        git(study.repo, "reset", "-q", "--hard", "HEAD~1")
    # the first record names none
    held = json.loads((study.repo / FIRST).read_text())
    study.put(FIRST, {**held, "supersedes": record["supersedes"]}, commit=False)
    assert has(study.problems(), "first versions record", "supersedes none")


def test_a_chain_of_two_records_keeps_earlier_runs_valid(study) -> None:
    first = study.validated()
    study.block(10300)
    study.block(10301)
    at_first = study.head()
    assert study.problems() == []
    study.n(75)
    study.put(CONFIRMATION_C1, study.supplement())
    second = study.versions(SECOND, CONFIRMATION_C1, "confirmation_c1",
                            supersedes_reason="next_stage")  # fmt: skip
    study.block(10400, step="confirmation", record=SECOND)
    assert study.problems(SECOND) == []  # the validation runs stay valid under the first
    # the first record no longer holds at the second's commit, nor for its runs
    found = study.problems()
    assert has(found, "differs from the recorded commit"), found
    assert has(found, "study3-c1-confirmation-10400", SECOND, "outside"), found
    # a validation run started under the first record with another sha256 is refused
    study.block(10302, sha=second)
    found = study.problems(SECOND)
    assert has(found, "study3-c1-validation-10302", second, first), found
    # at the first record's commit, the confirmation runs name a record outside its chain
    git(study.repo, "checkout", "-q", at_first)
    found = study.problems()
    assert has(found, "study3-c1-confirmation-10400", SECOND, "outside"), found
    assert not has(found, "differs from the recorded commit")
    git(study.repo, "checkout", "-q", "-")


def test_an_earlier_record_is_read_as_it_was_added(study) -> None:
    study.validated()
    study.block(10300)
    study.n(75)
    study.put(CONFIRMATION_C1, study.supplement())
    study.versions(SECOND, CONFIRMATION_C1, "confirmation_c1", supersedes_reason="next_stage")
    assert study.problems(SECOND) == []
    # the first record overwritten in a later commit: the chain breaks, and the runs started
    # under it are named with the record as it was added
    study.edit(FIRST, reason="validation_c1_again")
    study.commit("the first record overwritten")
    found = study.problems(SECOND)
    assert has(found, FIRST, "changed after it was added"), found


def test_a_chain_of_records_across_candidates(study) -> None:
    """Candidate 1's validation under the first record, candidate 2's under the second: each
    run's record holds its own candidate's validation fixation."""
    study.validated()
    study.block(10300)
    study.states.update({(arm, lr): "qualified" for arm, pair in RATES.items()
                         for lr in pair.values()})  # fmt: skip
    study.qualified(2)
    study.code = study.now()
    validation_c2 = "docs/study3-fixation-validation-c2.json"
    study.put(validation_c2, study.fixation(2))
    study.versions(SECOND, validation_c2, "validation_c2", supersedes_reason="candidate_2")
    study.block(10340, candidate=2, record=SECOND)
    assert study.problems(SECOND) == []
    # a candidate-2 run under the first record: its record holds candidate 1's fixation
    study.block(10341, candidate=2, record=FIRST)
    found = study.problems(SECOND)
    assert has(found, "study3-c2-validation-10341", VALIDATION_C1,
               "docs/study3-fixation-validation-c2.json"), found  # fmt: skip


def test_a_confirmation_run_under_the_validation_fixation_alone_is_refused(study) -> None:
    study.validated()
    study.block(10300)
    study.block(10400, step="confirmation")
    found = study.problems()
    assert has(found, "study3-c1-confirmation-10400", VALIDATION_C1, CONFIRMATION_C1,
               "confirmation supplement"), found  # fmt: skip


def test_a_record_without_a_fixation_stands_for_no_validation_run(study) -> None:
    study.ready()
    study.versions(fixation=None, reason="before_validation")
    assert study.problems() == []
    assert has(study.problems(step="validation", candidate=1), "fixation is null")
    study.block(10300)
    assert has(study.problems(), "study3-c1-validation-10300", "fixation is null")


def test_the_record_stands_for_its_step_and_candidate(study) -> None:
    study.validated()
    assert study.problems(step="validation", candidate=1) == []
    assert has(study.problems(step="confirmation", candidate=1), CONFIRMATION_C1)
    assert has(study.problems(step="validation", candidate=2),
               "docs/study3-fixation-validation-c2.json")  # fmt: skip
    with pytest.raises(ValueError):
        study.problems(step="search", candidate=1)


def test_the_record_pins_its_fixation_and_the_gpu_code(study, monkeypatch) -> None:
    study.validated()
    assert study.problems() == []
    # the fixation changed since the record named it
    study.edit(VALIDATION_C1, written_at="2026-10-27T13:00:00+00:00")
    study.commit("the fixation edited")
    found = study.problems()
    assert has(found, "fixation", "sha256"), found
    git(study.repo, "reset", "-q", "--hard", "HEAD~1")
    assert study.problems() == []
    record = json.loads((study.repo / FIRST).read_text())
    # a record whose recorded commit does not hold its fixation (the commit before it was added)
    study.put(FIRST, {**record, "code": {"commit": study.registered}}, commit=False)
    found = study.problems()
    assert has(found, f"the recorded commit {study.registered} does not hold {VALIDATION_C1}")
    git(study.repo, "checkout", "-q", "--", FIRST)
    # the recorded commit's GPU code is not the fixation's
    real = code_identity.gpu_code_identity

    def identity(commit="HEAD", repo=None):
        found = real(commit, repo)
        return {**found, "sha256": "e" * 64} if commit == record["code"]["commit"] else found

    monkeypatch.setattr(code_identity, "gpu_code_identity", identity)
    found = study.problems()
    assert has(found, record["code"]["commit"], "e" * 64, study.code["gpu_code_sha256"]), found


def test_a_missing_or_unreadable_journal_fails_closed(study) -> None:
    study.validated()
    study.block(10300)
    assert study.problems() == []
    log = study.root / "launch" / "study3-c1-validation-10300.jsonl"
    lines = log.read_text().splitlines()
    log.write_text("\n".join([lines[0], "{not json", *lines[1:]]) + "\n")
    assert has(study.problems(), "launch/study3-c1-validation-10300.jsonl line 2", "unparseable")
    log.write_bytes(b"\xff\xfe" + log.read_bytes())
    assert has(study.problems(), "launch journal", "cannot be read")
    shutil.rmtree(study.root / "launch")
    found = study.problems()
    assert has(found, "the launch journal is missing"), found
    assert has(found, "no start in the launch journal")  # the runs it started, too


def test_the_versions_record_builder_refuses(study) -> None:
    study.ready()
    record = study.fixation()
    study.put(VALIDATION_C1, record, commit=False)
    with pytest.raises(sv.RecordRefused, match="clean checkout"):
        study.versions()  # the fixation is not committed: the checkout is not clean
    study.commit("the fixation", VALIDATION_C1)
    for kw, words in (({"reason": "A sentence."}, "reason"),
                      ({"fixation": "docs/other.json"}, "fixation"),
                      ({"fixation": CONFIRMATION_C1}, "fixation"),
                      ({"supersedes_reason": "next_stage"}, "supersedes none")):  # fmt: skip
        with pytest.raises(sv.RecordRefused, match=words):
            study.versions(**{"fixation": VALIDATION_C1, "reason": "validation_c1", **kw})
    with pytest.raises(sv.RecordRefused, match="not a versions record"):
        study.versions("docs/versions.json")
    study.versions()
    with pytest.raises(sv.RecordRefused, match="already written"):
        study.versions()


def test_records_do_not_read_as_versions_records(study) -> None:
    study.validated()
    record = json.loads((study.repo / FIRST).read_text())
    for change, words in (({"what": "another text"}, ("what",)),
                          ({"code": {"commit": record["code"]["commit"],
                                     "allow_descendants": True}}, ("allow_descendants",)),
                          ({"reason": None}, ("reason",)),
                          ({"fixation": {"path": VALIDATION_C1, "sha256": "0" * 64}},
                           ("fixation", "sha256")),
                          ({"note": "x"}, ("note", "not a field"))):  # fmt: skip
        study.put(FIRST, {**record, **change}, commit=False)
        found, problems = sv.read_versions(study.repo / FIRST, repo=study.repo)
        assert found is None and has(problems, *words), (change, problems)
        assert has(study.problems(), *words)


# ------------------------------------------------------------------ the qualification's logs


def test_the_logs_a_qualification_rests_on(study) -> None:
    assert sv.QUALIFICATION_STEPS == qr.STEPS  # the qualification's campaign steps, one list
    study.qualified()
    study.qualified(2)
    log = "launch/study3-c1-numerical.jsonl"
    logs, problems = sv.qualification_logs(study.root, 1)
    assert problems == [] and [entry["path"] for entry in logs] == [log]
    assert logs[0]["sha256"] == sha256(study.root / log)
    # a repair campaign, an end in another log, a held line naming a label, a recovery round
    study.runs.log("study3-c1-repair", {"event": "start", "run": "A-1.5e-05/s10200",
                                        "run_id": "r1", "experiment": "study3",
                                        "header": {"study": 3, "candidate": 1, "step": "repair",
                                                   "round": None}})  # fmt: skip
    study.runs.log("settle", {"event": "end", "run_id": "q1-F-1.3e-05-10201", "outcome": "done"})
    study.runs.log("hold", {"event": "held", "run": "F-1.83e-05/s10202", "run_id": "other",
                            "experiment": "study3"})  # fmt: skip
    study.runs.log("recover", {"event": "recovery", "experiment": "study3", "cell": "P-1.3e-05",
                               "runs": ["P-1.3e-05/s10200"]})  # fmt: skip
    study.runs.log("unrelated", {"event": "end", "run_id": "q2-F-1.3e-05-10211",
                                 "outcome": "done"})  # fmt: skip
    study.block(10300, record=FIRST, sha="0" * 64)  # a validation campaign: not the qualification
    logs, problems = sv.qualification_logs(study.root, 1)
    assert problems == []
    assert [entry["path"] for entry in logs] == [
        "launch/hold.jsonl", "launch/recover.jsonl", "launch/settle.jsonl",
        "launch/study3-c1-numerical.jsonl", "launch/study3-c1-repair.jsonl"]  # fmt: skip
    path = study.root / "launch" / "unrelated.jsonl"
    path.write_text("{cut\n" + path.read_text())
    assert has(sv.qualification_logs(study.root, 1)[1], "launch/unrelated.jsonl line 1")


# ------------------------------------------------------------------ a real qualification


@pytest.mark.parametrize("failing", [False, True])
def test_the_fixation_reads_a_real_qualification(qualify, monkeypatch, tmp_path, failing) -> None:
    """Candidate 1's numerical qualification as package 1's ``qualify.py`` writes it, the
    reading not replaced: A and F, qualified; or F's rate failing on a scientific case, which
    refuses the fixation. Package 1's test world names its launcher runs by seed and optimizer,
    so each arm has one rate here, fixed as both its lower and its upper rate."""
    from test_qualification import CODE as RECORDS_CODE
    from test_qualification_records import World as QualificationWorld
    from test_qualification_records import beyond, qcase

    code = {**RECORDS_CODE, "sha256": "e" * 64}
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(code))
    rates = {arm: {"lr_l": RATES[arm]["lr_u"], "lr_u": RATES[arm]["lr_u"]} for arm in ("A", "F")}
    cells = tuple(((OPT[arm], rates[arm]["lr_u"]), SEEDS["numerical"][1]) for arm in rates)
    study = Study(tmp_path / "study", monkeypatch)
    monkeypatch.setattr(qr, "read_qualification", REAL_READ)
    study.code = {"commit": study.head(), "gpu_code_sha256": code["sha256"]}
    study.decided()
    world = QualificationWorld(tmp_path / "study", cells)
    kept = qcase(10201, "adam_frozen", rates["F"]["lr_u"])
    world.full(cells, code=code, changes={kept: beyond("rest", "baseline~s3")} if failing else None)
    world.verdict(qualify)
    if failing:
        with pytest.raises(sv.RecordRefused, match="1.83e-05 is failing"):
            sv.validation_fixation(study.root, 1, rates=rates, code=study.code, repo=study.repo)
        return
    record = sv.validation_fixation(study.root, 1, rates=rates, code=study.code,
                                    repo=study.repo)  # fmt: skip
    assert record["qualification"]["states"] == {arm: {"lr_l": "qualified", "lr_u": "qualified"}
                                                 for arm in rates}  # fmt: skip
    assert [log["path"] for log in record["qualification"]["logs"]] == [
        "launch/study3-c1-numerical.jsonl"]  # fmt: skip
    study.put(VALIDATION_C1, record, commit=False)
    assert study.read_validation() == (record, [])


REAL_READ = qr.read_qualification


# ------------------------------------------------------------------ the sealing boundary


def test_nothing_under_responses_or_observations_is_opened(study, monkeypatch) -> None:
    """The records tests' guard (``open``, ``io.open``, ``Path.open``/``read_text``/
    ``read_bytes`` fail on a ``responses/`` or ``observations/`` path), as the decision records'
    tests extend it to the path a link resolves to."""
    study.validated()
    validation = study.block(10300) + study.block(10300, arm="F")
    study.n(75)
    study.put(CONFIRMATION_C1, study.supplement())
    study.versions(SECOND, CONFIRMATION_C1, "confirmation_c1", supersedes_reason="next_stage")
    study.block(10400, step="confirmation", record=SECOND)
    # a run with no start whose config.yaml leads into another run's responses
    sealed = study.root / "study3" / validation[0] / "responses" / "harmful_evaluation.jsonl"
    orphan = study.root / "study3" / "20261027T000000Z-A-1.45e-05-s10301-abcd1234-a1"
    orphan.mkdir()
    (orphan / "config.yaml").symlink_to(sealed)
    attempts = seal(monkeypatch)
    found = study.problems(SECOND)
    assert has(found, orphan.name, "no start in the launch journal")
    assert len(found) == 1, found
    out = [
        found,
        study.read_validation(),
        study.read_confirmation(),
        sv.read_versions(study.repo / SECOND, repo=study.repo),
        sv.qualification_logs(study.root, 1),
    ]
    assert SENTINEL not in json.dumps(out, default=str)
    assert attempts == []
    with pytest.raises(AssertionError):  # the guard is live
        sealed.read_text()
    assert len(attempts) == 1


def test_the_builders_and_readers_write_nothing(study) -> None:
    study.validated()
    study.block(10300)
    study.n(75)

    def snapshot(root: Path) -> dict:
        return {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(root.rglob("*"))
                if p.is_file() and not any(part in ("responses", "observations", ".git")
                                           for part in p.parts)}  # fmt: skip

    before = (snapshot(study.root), snapshot(study.repo))
    study.supplement()
    study.problems()
    study.read_validation()
    sv.qualification_logs(study.root, 1)
    assert (snapshot(study.root), snapshot(study.repo)) == before


def test_no_gpu_module_is_imported() -> None:
    """The module imports no GPU path (``qualification``, ``schedule``, ``train``, ``launch``,
    ``runtime``, ``fa_control_diagnostic``) and no torch; ``qualification_records`` (which loads
    torch through ``qualification_verdicts``) only where the qualification is read."""
    tree = ast.parse(Path(sv.__file__).read_text())
    imported, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
            names |= {f"{node.module or ''}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    gpu = {"qualification", "schedule", "train", "launch", "runtime", "fa_control_diagnostic",
           "torch"}  # fmt: skip
    assert not {i for i in imported | names if gpu & set(i.split("."))}
    top = {n.names[0].name for n in tree.body if isinstance(n, ast.ImportFrom) and n.names}
    assert "qualification_records" not in top  # read lazily, where the qualification is read
