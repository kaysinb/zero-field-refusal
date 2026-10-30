"""scripts/study3_analysis.py, in process: ``freeze`` writes the analysis freeze once, from a clean
checkout, reading no label; ``analyze`` is refused before a committed freeze and while a cohort run
is pending or a technical labeling problem, writing nothing; otherwise it writes the analysis, then
validation's harm naming it, resumes an interrupted run, and recomputes only with a reason.

The study is the progress tests' synthetic one at candidate 1's done (``test_study3_freeze.done``:
60 blocks per comparison, A, F and P active, every arm's weight 1/2), its state given to the
readers. The checkout is a temporary Git repository holding a stand-in simulation fixation; the
in-force supplement, validation fixation and n record are given; the label reader's ``harm_many``
and ``run_harm`` are stubbed with counts by arm and seed. Every bracket run's K rows are written
beside a utility row holding the sentinel, and read for real; the freeze gate is the real one."""

import hashlib
import json
from pathlib import Path

import pytest
from test_code_identity import gpu_uses
from test_study3_campaigns import git, script_module
from test_study3_final import label_raisers
from test_study3_freeze import SUPPLEMENT, done, k_harmful, scaled
from test_study3_package2 import SENTINEL, SealedGuard
from test_study3_progress import EXPECTED

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import study3_final, study3_labels
from zero_field_refusal.analysis import study3_freeze as fz
from zero_field_refusal.analysis.study3_freeze import freeze_name
from zero_field_refusal.analysis.study3_labels import LabelingProblem, LabelsPending, RunHarm
from zero_field_refusal.analysis.study3_records import SEEDS
from zero_field_refusal.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "scripts" / "study3_analysis.py"
SIMULATION_FIXATION = "docs/study3-simulation-fixation.json"
KAPPA = {"A": 9.537, "F": 11.124, "P": 11.124, "unit": "pp per (ln K)^2",
         "nominal_bias_af_pp": -0.0395, "source": "study 1's logistic curves at K*",
         "note": "a design-time figure"}  # fmt: skip
BOOTSTRAP = {"seed": 20261101, "resamples": 2000}
BASE = {"A": 20, "F": 26, "P": 23}  # harmful labels of 521, by arm (plus an offset by seed)
STEP = {"A": 3, "F": 5, "P": 2}  # the offset: seed × STEP mod 11, so that the differences vary


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class World:
    """Candidate 1's done study, a clean temporary checkout whose HEAD is the code commit, every
    reader of the in-force records given, and the label reader's harm by arm and seed (``culprit``,
    when set, raises ``raised``)."""

    def __init__(self, tmp_path: Path, monkeypatch, capsys, study=None) -> None:
        self.monkeypatch, self.capsys = monkeypatch, capsys
        self.root = tmp_path / "runs"
        (self.root / "study3").mkdir(parents=True)
        self.repo = tmp_path / "repo"
        (self.repo / "docs").mkdir(parents=True)
        git(self.repo, "init", "-q")
        fixation = {"kind": "simulation-fixation", "curvature": KAPPA}
        (self.repo / SIMULATION_FIXATION).write_text(json.dumps(fixation, indent=2) + "\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", "the code")
        self.head = git(self.repo, "rev-parse", "HEAD")
        self.study = k_harmful(done(), scaled) if study is None else study
        self.state, self.endpoints = self.study.advance(), list(self.study.endpoints)
        self.by_id = {e.run_id: e for e in self.endpoints}
        self.script = script_module(SCRIPT, "study3_analysis")
        self.culprit: str | None = None
        self.raised: type = LabelsPending
        self.ledger: list[Path] = []
        self.reads: list[tuple[bool, list[str]]] = []  # (analysis.json written?, the runs read)
        cohort = {"seeds": list(SEEDS["validation"][1][:28])}
        n = {"n-c1": {"state": "passed", "record": {"cohort": cohort}}}
        supplement = {"bootstrap": dict(BOOTSTRAP)}
        mp = monkeypatch
        mp.setattr(self.script, "REPO", self.repo)
        mp.setattr(code_identity, "gpu_code_identity", lambda *a, **kw: {"sha256": EXPECTED})
        for module in (fz, self.script):
            mp.setattr(module, "read_state", lambda *a, **kw: (self.state, list(self.endpoints)))
            mp.setattr(module, "in_force_supplement", lambda c, **kw: (dict(SUPPLEMENT), []))
            mp.setattr(module, "decisions", lambda *a, **kw: n)
        mp.setattr(self.script, "read_confirmation_fixation", lambda c, **kw: (supplement, []))
        mp.setattr(self.script, "read_validation_fixation", lambda c, **kw: ({"level": 0.975}, []))
        mp.setattr(self.script, "ledger_runs", self.ledger_runs)
        mp.setattr(study3_final, "harm_many", self.harm_many)
        mp.setattr(study3_final, "labeling_problems", self.problems)
        mp.setattr(study3_labels, "_gated", self.gated)  # the label reader's gate steps
        # the final point's run set: every run of the study (the label reader's own is stubbed)
        mp.setattr(study3_final, "open_point", lambda point, **kw: frozenset(self.by_id))
        for e in self.endpoints:
            if e.cell.step == "confirmation" and e.K is not None:
                self.k_rows(e.run_id, e.K)

    def ledger_runs(self, runs_root) -> frozenset[str]:
        self.ledger.append(Path(runs_root))
        return frozenset()

    def harm_many(self, dirs, **kw) -> list[RunHarm]:
        """The label reader's cohort reading, each call kept with whether the analysis was
        written when it was made."""
        written = (self.root / "study3-final" / "c1" / "analysis.json").exists()
        self.reads.append((written, sorted(Path(d).name for d in dirs)))
        return [self.harm(d) for d in dirs]

    def gated(self, run_dirs, point, candidate, runs_root, repo_root) -> list:
        """The label reader's gate steps: ``raised`` for the culprit, the others passing."""
        for run_dir in run_dirs:
            if Path(run_dir).name == self.culprit:
                raise self.raised(f"{self.culprit}: not countable")
        return [(Path(run_dir), None, "f" * 64) for run_dir in run_dirs]

    def harm(self, run_dir) -> RunHarm:
        e = self.by_id[Path(run_dir).name]
        if e.run_id == self.culprit:
            raise self.raised(f"{e.run_id}: not countable")
        return RunHarm(harmful=BASE[e.arm] + e.seed * STEP[e.arm] % 11 + (e.K > 6),
                       missing=e.seed % 3,
                       receipt_sha256="f" * 64)  # fmt: skip

    def problems(self, runs_root) -> tuple[str, ...]:
        return (self.culprit,) if self.culprit and self.raised is LabelingProblem else ()

    def k_rows(self, run: str, k: float) -> None:
        """K rows whose samples' mean is ``k`` beside a utility row holding the sentinel."""
        folder = self.root / "study3" / run / "observations"
        folder.mkdir(parents=True)
        rows = [{"condition": "step-25", "measure": "factual_qa", "item_id": "f:0", "value": 1.0,
                 "text": SENTINEL}]  # fmt: skip
        for i, pair in enumerate(((k - 0.6, k - 0.4), (k, k), (k + 0.4, k + 0.6))):
            rows += [{"condition": "step-25", "measure": "kl_benign", "item_id": f"kl:{i}",
                      "sample": j, "value": v, "tokens": 9}
                     for j, v in enumerate(pair)]  # fmt: skip
        (folder / "measurements.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))  # fmt: skip

    def call(self, *argv) -> tuple[int, str, str]:
        """The command's exit code, its output and its error text (a refusal's message)."""
        self.capsys.readouterr()
        code, message = 0, ""
        try:
            self.script.main([str(a) for a in argv])
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
            message = exc.code if isinstance(exc.code, str) else ""
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err + message

    def run(self, *argv) -> str:
        code, out, error = self.call(*argv)
        assert code == 0, error
        return out

    def refused(self, *argv) -> str:
        code, out, error = self.call(*argv)
        assert code not in (0, None), out
        return error

    def frozen(self) -> None:
        """The freeze written by the command, then committed alone."""
        self.run("freeze", "--runs-root", self.root, "--candidate", "1")
        git(self.repo, "add", freeze_name(1))
        git(self.repo, "commit", "-qm", "the analysis freeze")


@pytest.fixture
def world(tmp_path, monkeypatch, capsys) -> World:
    return World(tmp_path, monkeypatch, capsys)


def raisers(monkeypatch) -> list[str]:
    """Every harm and item reader, in the label reader and wherever the final analysis's binds
    it, a raiser."""
    calls: list[str] = []
    for module in (study3_labels, study3_final):
        for name in ("run_harm", "harm_many", "run_items"):
            if not hasattr(module, name):
                continue

            def raiser(*args, _name=name, **kwargs):
                calls.append(_name)
                raise AssertionError(f"a label reader was called: {_name}")

            monkeypatch.setattr(module, name, raiser)
    return calls


def test_the_script_does_no_gpu_work() -> None:
    assert gpu_uses(SCRIPT) == set()


# ------------------------------------------------------------------ freeze


def test_freeze_is_refused_before_done_and_on_a_dirty_checkout(tmp_path, monkeypatch, capsys):
    world = World(tmp_path, monkeypatch, capsys, study=k_harmful(done(count=30), scaled))
    calls = raisers(monkeypatch)
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    assert "done" in world.refused("freeze", "--runs-root", world.root, "--candidate", "1")
    (world.repo / "notes.txt").write_text("an untracked file\n")
    assert "clean" in world.refused("freeze", "--runs-root", world.root, "--candidate", "1")
    assert not (world.repo / freeze_name(1)).exists()
    assert guard.attempts == [] and calls == []


def test_freeze_writes_the_record_once_and_reads_no_label(world, monkeypatch) -> None:
    calls = raisers(monkeypatch)
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    out = world.run("freeze", "--runs-root", world.root, "--candidate", "1")
    assert freeze_name(1) in out and "commit it alone before any confirmation label" in out
    path = world.repo / freeze_name(1)
    record = json.loads(path.read_text())
    assert record["kind"] == "analysis-freeze" and record["candidate"] == 1
    assert record["code"] == {"commit": world.head, "gpu_code_sha256": EXPECTED}
    assert path.read_text() == json.dumps(record, indent=2, sort_keys=True) + "\n"
    assert world.ledger == [world.root]
    assert "written once" in world.refused("freeze", "--runs-root", world.root, "--candidate", "1")
    assert json.loads(path.read_text()) == record
    assert guard.attempts == [] and calls == []


# ------------------------------------------------------------------ analyze


def test_analyze_is_refused_without_a_committed_freeze_opening_nothing(world, monkeypatch):
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    argv = ("analyze", "--runs-root", world.root, "--candidate", "1")
    assert "no analysis freeze" in world.refused(*argv)
    world.run("freeze", "--runs-root", world.root, "--candidate", "1")
    assert "not committed" in world.refused(*argv)
    assert guard.attempts == [] and not (world.root / "study3-final").exists()


PROBLEM = "1 runs with a technical labeling problem"


@pytest.mark.parametrize("raised,words", [(LabelsPending, "1 runs pending"),
                                          (LabelingProblem, PROBLEM)])  # fmt: skip
def test_analyze_is_refused_while_a_cohort_run_is_pending_or_broken(world, monkeypatch, raised,
                                                                    words) -> None:  # fmt: skip
    """The culprit is named through the label reader's gate steps alone: after the refusal, no
    run's labels are read (``study3_labels._read_run`` and ``run_harm`` are raisers)."""
    world.frozen()
    freeze = json.loads((world.repo / freeze_name(1)).read_text())
    world.culprit = freeze["comparisons"]["P-F"]["blocks"][5]["arms"]["P"]["hi"]
    world.raised = raised
    reads = label_raisers(monkeypatch)
    why = world.refused("analyze", "--runs-root", world.root, "--candidate", "1")
    assert words in why and world.culprit in why
    assert not (world.root / "study3-final").exists()
    assert reads == [] and len(world.reads) == 1  # the cohort's one reading, nothing after it


def validation_runs(world: World) -> set[str]:
    freeze = json.loads((world.repo / freeze_name(1)).read_text())
    blocks = [b for entry in freeze["validation"].values() if entry for b in entry["blocks"]]
    return set(study3_final.runs_of(blocks))


def test_no_validation_run_is_read_before_the_analysis_is_written(world) -> None:
    """Validation's harm comes after the final analysis and apart from it: the confirmation's
    runs are read first, with no validation run; validation's runs are read in a reading of their
    own, once analysis.json exists."""
    world.frozen()
    validation = validation_runs(world)
    world.run("analyze", "--runs-root", world.root, "--candidate", "1")
    assert len(world.reads) == 2 and len(validation) == 168  # A, F and P at both ends, 28 blocks
    (before, first), (after, second) = world.reads
    assert before is False and not set(first) & validation
    assert after is True and set(second) == validation


def test_a_pending_validation_run_leaves_the_analysis_written_and_resumes(world) -> None:
    """A validation run that is pending refuses validation's harm after the analysis is written;
    once it counts, a rerun writes validation's harm alone."""
    world.frozen()
    world.culprit = sorted(validation_runs(world))[3]
    argv = ("analyze", "--runs-root", world.root, "--candidate", "1")
    why = world.refused(*argv)
    assert "1 runs pending" in why and world.culprit in why
    folder = world.root / "study3-final" / "c1"
    assert (folder / "analysis.json").exists() and not (folder / "validation-harm.json").exists()
    analysis = (folder / "analysis.json").read_bytes()
    world.culprit = None
    world.run(*argv)
    assert (folder / "analysis.json").read_bytes() == analysis
    named = json.loads((folder / "validation-harm.json").read_text())["analysis"]
    assert named["sha256"] == hashlib.sha256(analysis).hexdigest()


def test_the_label_free_utility_gates_come_before_any_label(world, monkeypatch) -> None:
    """The utility gates read no label: they run before the cohort's reading, and their refusal
    (a frozen block that cannot be read from the state now) reads no label at all."""
    world.frozen()
    real, seen = world.script.utility_gates, []

    def gates(*args, **kwargs):
        seen.append(len(world.reads))
        return real(*args, **kwargs)

    monkeypatch.setattr(world.script, "utility_gates", gates)
    world.run("analyze", "--runs-root", world.root, "--candidate", "1")
    assert seen == [0] and len(world.reads) == 2
    world.reads.clear()
    (world.root / "study3-final").rename(world.root / "aside")

    def broken(*args, **kwargs):
        raise ValueError("no endpoint of the arm")

    monkeypatch.setattr(world.script, "block_state", broken)
    reads = label_raisers(monkeypatch)
    why = world.refused("analyze", "--runs-root", world.root, "--candidate", "1")
    assert "no endpoint of the arm" in why and world.reads == [] and reads == []
    assert not (world.root / "study3-final").exists()


def test_analyze_writes_the_analysis_then_validations_harm(world) -> None:
    world.frozen()
    root = world.root
    FREEZE_SHA = sha256(world.repo / freeze_name(1))
    out = world.run("analyze", "--runs-root", root, "--candidate", "1")
    analysis = root / "study3-final" / "c1" / "analysis.json"
    validation = root / "study3-final" / "c1" / "validation-harm.json"
    record = json.loads(analysis.read_text())
    assert record["comparisons"]["A-F"]["n"] == 60 and record["freeze"]["sha256"] == FREEZE_SHA
    assert json.loads(validation.read_text())["analysis"]["sha256"] == sha256(analysis)
    assert SENTINEL not in analysis.read_text() + validation.read_text() + out
    validation.unlink()  # an interrupted run, resumed
    world.run("analyze", "--runs-root", root, "--candidate", "1")
    assert validation.exists() and record == json.loads(analysis.read_text())
    assert "supersedes" in world.refused("analyze", "--runs-root", root, "--candidate", "1")


def test_the_analysis_record_names_its_inputs_and_validations_harm_is_descriptive(world):
    world.frozen()
    out = world.run("analyze", "--runs-root", world.root, "--candidate", "1")
    folder = world.root / "study3-final" / "c1"
    record = json.loads((folder / "analysis.json").read_text())
    assert record["kind"] == study3_final.ANALYSIS_KIND and record["candidate"] == 1
    assert record["freeze"] == {"path": freeze_name(1), "root": "repo",
                                "sha256": sha256(world.repo / freeze_name(1))}  # fmt: skip
    assert record["code"] == {"commit": git(world.repo, "rev-parse", "HEAD"),
                              "gpu_code_sha256": EXPECTED}  # fmt: skip
    assert record["supplement"] == SUPPLEMENT and record["level"] == 0.975
    assert record["bootstrap"] == BOOTSTRAP and record["superseded"] is None
    assert record["kappa"] == {"path": SIMULATION_FIXATION, "root": "repo",
                               "sha256": sha256(world.repo / SIMULATION_FIXATION),
                               "values": {arm: KAPPA[arm] for arm in "AFP"},
                               "unit": KAPPA["unit"], "source": KAPPA["source"]}  # fmt: skip
    assert set(record["comparisons"]) == {"A-F", "P-F", "A-P", "A-F@K_harmful"}
    assert len(record["k_spread"]) == 360 and record["utility"]["claim"] is True
    a = next(e for e in world.endpoints if e.run_id in record["k_spread"])
    assert record["k_spread"][a.run_id] == pytest.approx(  # per prompt K − 0.5, K and K + 0.5
        [a.K + x for x in (-0.5, -0.4, -0.25, 0.0, 0.25, 0.4, 0.5)])  # fmt: skip
    assert record["comparisons"]["A-F"]["primary"]["undetermined"] is False
    for key in ("A-F", "P-F", "A-P", "A-F@K_harmful"):
        entry = record["comparisons"][key]
        assert f"{key}: n {entry['n']}, {entry['wording']['row']}" in out
    validation = json.loads((folder / "validation-harm.json").read_text())
    assert validation["analysis"] == {"path": "study3-final/c1/analysis.json", "root": "runs",
                                      "sha256": sha256(folder / "analysis.json")}  # fmt: skip
    assert validation["freeze"] == record["freeze"]
    harm = validation["comparisons"]
    assert harm["A-F"]["n"] == 28 and harm["A-F"]["status"] == "descriptive"
    assert harm["P-F"]["n"] == 28 and harm["P-F"]["primary"] is not None
    assert str(folder / "analysis.json") in out and str(folder / "validation-harm.json") in out


def test_a_recomputation_needs_a_reason_and_names_its_predecessor(world) -> None:
    world.frozen()
    argv = ("analyze", "--runs-root", world.root, "--candidate", "1")
    assert "nothing to supersede" in world.refused(*argv, "--supersedes-reason", "a_fix")
    world.run(*argv)
    folder = world.root / "study3-final" / "c1"
    assert "an id" in world.refused(*argv, "--supersedes-reason", "A fix")
    world.run(*argv, "--supersedes-reason", "a_fix")
    first = json.loads((folder / "analysis.json").read_text())
    second = json.loads((folder / "analysis-2.json").read_text())
    assert second["superseded"] == {"path": "study3-final/c1/analysis.json", "reason": "a_fix",
                                    "sha256": sha256(folder / "analysis.json")}  # fmt: skip
    assert second["comparisons"] == first["comparisons"]
    named = json.loads((folder / "validation-harm-2.json").read_text())["analysis"]
    assert named["sha256"] == sha256(folder / "analysis-2.json")
    assert "supersedes" in world.refused(*argv)
