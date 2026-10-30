"""Study 3's spread (scripts/study3_spread.py): validation's A–F spread, N and the n record, on a
synthetic runs directory and a stand-in checkout in Git. The state (``study3_labels._state``), the
blocks' endpoints (the script's ``read_endpoints``) and the labels (its ``harm_many``) are given;
the validation fixation is a stand-in file that pins the stand-in spread script and the module it
imports. Nothing reads a runs/ folder of the checkout, and no label or response is opened: every
sealed file of these worlds holds a sentinel and none is read."""

import dataclasses
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
from test_study3_blocks import full_block
from test_study3_campaigns import script_module
from test_study3_decisions import SENTINEL, gates_with, power_gate
from test_study3_decisions import World as DecisionWorld
from test_study3_package2 import SealedGuard
from test_study3_versions import git

from zero_field_refusal.analysis import study3_labels, study3_versions
from zero_field_refusal.analysis.study3_blocks import interpolated
from zero_field_refusal.analysis.study3_decisions import decision_path, read_decision, sample_size
from zero_field_refusal.analysis.study3_labels import LabelingProblem, LabelsPending, RunHarm
from zero_field_refusal.analysis.study3_pins import SPREAD_SCRIPT, spread_pin
from zero_field_refusal.analysis.study3_records import SEEDS
from zero_field_refusal.analysis.study3_search import CANDIDATES
from zero_field_refusal.paths import REPO_ROOT

OUTPUT = "study3/decisions/spread-output-c{candidate}.json"
START = CANDIDATES[1]["start"]
RATES = {arm: {"lr_l": START[arm]["l"], "lr_u": START[arm]["u"]} for arm in ("A", "F")}
FIXATION = "docs/study3-fixation-validation-c1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spread_module():
    """The real script, loaded as a module (its ``REPO`` is the checkout until a world sets it)."""
    return script_module(REPO_ROOT / SPREAD_SCRIPT, "study3_spread")


@pytest.fixture(scope="module")
def spread():
    return spread_module()


def fixate(repo: Path, monkeypatch, candidate: int = 1) -> list[dict[str, str]]:
    """A stand-in validation fixation of ``candidate`` in the checkout ``repo``: a file at its
    path that pins the spread as the checkout's files are now, read for this checkout by
    ``study3_versions.read_validation_fixation`` as it is on disk (no file: no fixation); a real
    fixation of another checkout is read as before. Returns the pin."""
    pinned = spread_pin(repo)
    path = repo / study3_versions.fixation_name("validation", candidate)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": "fixation-validation", "candidate": candidate,
                                "spread": pinned}, indent=1) + "\n")  # fmt: skip
    real, own = study3_versions.read_validation_fixation, repo.resolve()

    def read(c, *, runs_root, repo=REPO_ROOT, settings=None):
        if Path(repo).resolve() != own:
            return real(c, runs_root=runs_root, repo=repo, settings=settings)
        named = Path(repo) / study3_versions.fixation_name("validation", c)
        return (json.loads(named.read_text()), []) if named.is_file() else (None, [])

    monkeypatch.setattr(study3_versions, "read_validation_fixation", read)
    return pinned


def block(seed: int, ids=None) -> list:
    """A block's A and F endpoints at their targets (each arm's weight 1/2), in the order A at
    its lower rate, A at its upper, F at its lower, F at its upper; ``ids`` renames the four."""
    found = full_block(seed, arms=("A", "F"))
    if ids is None:
        return found
    return [
        dataclasses.replace(e, run_id=i, run_dir=Path(i)) for e, i in zip(found, ids, strict=True)
    ]


class Records(DecisionWorld):
    """``test_study3_decisions.World``'s records and stand-ins on a runs directory and a checkout
    that exist already (another world's): their evidence under ``runs``."""

    def __init__(self, runs: Path, repo: Path, monkeypatch) -> None:  # the folders exist
        self.runs, self.repo, self.monkeypatch = runs, repo, monkeypatch
        self.pins, self.lines = {}, 0


def harmful(index: int, arm: str, end: str) -> int:
    """A run's harmful count: A above F by 20·(block mod 4), the upper rate 10 above the lower."""
    return 100 + (20 * (index % 4) if arm == "A" else 0) + (10 if end == "u" else 0)


class World:
    """A runs directory with the engineering and simulation records, and a checkout in Git with
    the stand-in spread script, the module it imports and the stand-in validation fixation that
    pins both (``fixate``). ``at_choice`` gives the state at candidate 1's choice step, the
    blocks' endpoints and their labels; ``reads`` lists the runs of every reading of the labels."""

    def __init__(self, base: Path, monkeypatch, **simulation) -> None:
        self.records = DecisionWorld(base, monkeypatch)
        self.root, self.repo, self.monkeypatch = self.records.runs, self.records.repo, monkeypatch
        self.records.stand_in()
        self.pin = fixate(self.repo, monkeypatch)
        git(self.repo, "init", "-q")
        self.commit("the stand-ins")
        self.records.write("engineering", self.records.engineering())
        self.records.write("simulation", self.records.simulation(**simulation))
        self.spread = spread_module()
        monkeypatch.setattr(self.spread, "REPO", self.repo)
        self.reads: list[list[str]] = []
        self.fail: Exception | None = None
        self.while_reading = None
        self.refusal = ""

    def commit(self, message: str) -> None:
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-qm", message)

    # -- the state, the endpoints and the labels

    def at_choice(self, blocks: int = 22, *, stage: str = "choice", candidate: int = 1,
                  problems=(), cohort=None, while_reading=None) -> None:  # fmt: skip
        seeds = list(SEEDS["validation"][1][:blocks])
        self.blocks = [block(seed) for seed in seeds]
        self.endpoints = [e for found in self.blocks for e in found]
        runs = [e.run_id for e in self.endpoints]
        self.harm = {
            e.run_id: RunHarm(harmful=harmful(i, e.arm, "u" if e.lr == START[e.arm]["u"] else "l"),
                              missing=i % 3, receipt_sha256=f"{n:064x}")
            for i, found in enumerate(self.blocks) for n, e in enumerate(found, start=4 * i)
        }  # fmt: skip
        fields = {
            "candidate": 1,
            "rates": RATES,
            "active": ["A-F"],
            "cohort": {"seeds": seeds, "run_ids": runs if cohort is None else cohort},
        }
        self.state = {
            "stage": stage,
            "candidate": candidate,
            "problems": list(problems),
            "prescribed": {},
            "step": {
                "kind": "decision",
                "record": "n-c1",
                "path": "study3/decisions/n-c1.json",
                "why": "the spread",
                "fields": fields,
            },
        }
        self.while_reading = while_reading
        self.monkeypatch.setattr(study3_labels, "_state", lambda runs_root, repo_root: self.state)
        self.monkeypatch.setattr(self.spread, "read_endpoints", self.read_endpoints)
        self.monkeypatch.setattr(self.spread, "harm_many", self.harm_many)

    def read_endpoints(self, runs_root, *, theta0, prescribed):
        assert Path(runs_root) == self.root and prescribed == self.state["prescribed"]
        return list(self.endpoints), []

    def harm_many(self, run_dirs, *, point, candidate, runs_root, repo_root):
        assert (point, candidate, Path(runs_root), Path(repo_root)) == ("spread", 1, self.root,
                                                                         self.repo)  # fmt: skip
        self.reads.append([Path(d).name for d in run_dirs])
        if self.while_reading is not None:
            self.while_reading()
        if self.fail is not None:
            raise self.fail
        return [self.harm[Path(d).name] for d in run_dirs]

    def expected(self) -> tuple[float, int]:
        """The spread in pp at full precision and N, worked out here from the labels."""
        d = []
        for found in self.blocks:
            h = [self.harm[e.run_id].harmful / 521 for e in found]
            d.append(interpolated(h[0], h[1], 0.5) - interpolated(h[2], h[3], 0.5))
        sd = 100 * statistics.stdev(d)
        return sd, sample_size(sd, len(d), 0.975)

    # -- the command and what it leaves

    def main(self) -> None:
        self.spread.main(["--runs-root", str(self.root), "--candidate", "1"])

    def main_refused(self, match: str) -> bool:
        with pytest.raises(SystemExit) as caught:
            self.main()
        self.refusal = str(caught.value.code)
        assert self.refusal.startswith("study3_spread.py: "), self.refusal
        assert re.search(match, self.refusal), self.refusal
        return True

    def read(self, kind: str):
        return read_decision(self.root, kind, repo=self.repo)

    def output(self) -> Path:
        return self.root / OUTPUT.format(candidate=1)

    def edit_pinned(self, name: str) -> None:
        """A pinned file of the checkout changed: the spread script or a module under
        ``src/zero_field_refusal/``."""
        folder = "scripts" if name == Path(SPREAD_SCRIPT).name else "src/zero_field_refusal"
        path = self.repo / folder / name
        path.write_text(path.read_text() + "# changed\n")

    def pinned(self) -> list[dict[str, str]]:
        return self.pin


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    return World(tmp_path, monkeypatch)


# ------------------------------------------------------------------ the rule


@pytest.mark.parametrize("sd,m,level,n", [(3.5, 28, 0.975, 60), (5.0, 28, 0.975, 88),
                                          (5.0, 22, 0.975, 92), (5.0, 28, 0.98, 93)])  # fmt: skip
def test_the_registrations_examples_through_s_sample_size(spread, sd, m, level, n):
    assert spread.n_for(sd, m, level) == n == sample_size(sd, m, level)


def test_the_spread_does_not_reimplement_the_n_rule(spread):
    source = Path(spread.__file__).read_text()
    assert "sample_size" in source and "chi2" not in source and "norm.cdf" not in source


def test_a_short_power_raises_n_to_the_simulations_minimum(spread):
    assert spread.n_for(3.5, 28, 0.975, n_min=75) == 75
    assert spread.n_for(5.0, 28, 0.975, n_min=75) == 88  # the rule's N, when it is larger


@pytest.mark.parametrize("sd,shown", [(3.24, 3.0), (3.25, 3.5), (3.74, 3.5), (3.75, 4.0)])
def test_the_spread_is_rounded_half_up(spread, sd, shown):
    assert spread.rounded(sd) == shown


def test_the_differences_use_the_blocks_weights(spread):
    blocks = [{"A": ("a-lo", "a-hi", 0.25), "F": ("f-lo", "f-hi", 0.5)}]
    harm = {r: RunHarm(harmful=k, missing=1, receipt_sha256="0" * 64)
            for r, k in (("a-lo", 52), ("a-hi", 104), ("f-lo", 26), ("f-hi", 78))}  # fmt: skip
    expected = (0.75 * 52 + 0.25 * 104) / 521 - (0.5 * 26 + 0.5 * 78) / 521
    assert spread.block_differences(blocks, harm) == [pytest.approx(expected, abs=1e-15)]


# ------------------------------------------------------------------ the records


def test_the_spread_writes_its_records_and_prints_three_keys(world, capfd):
    world.at_choice(blocks=22)
    world.main()
    out, _ = capfd.readouterr()
    shown = json.loads(out)
    assert set(shown) == {"outcome", "N", "spread_pp"} and out.count("\n") == 1
    sd, n = world.expected()
    assert shown == {"outcome": "passed", "N": n, "spread_pp": world.spread.rounded(sd)}
    output = json.loads(world.output().read_text())
    assert not {"sd", "sd_up", "power", "mean", "differences"} & set(output)
    assert set(output) == {"kind", "candidate", "level", "blocks", "outcome", "N", "cap",
                           "spread_pp", "fixation", "labels", "code"}  # fmt: skip
    assert (output["kind"], output["blocks"], output["level"]) == ("spread-output", 22, 0.975)
    assert output["fixation"] == {"path": FIXATION, "sha256": sha256(world.repo / FIXATION),
                                  "root": "repo"}  # fmt: skip
    assert output["code"] == world.pinned()
    assert output["labels"] == [{"run": run, "receipt_sha256": world.harm[run].receipt_sha256}
                                for run in sorted(world.harm)]  # fmt: skip
    n_record, problems = world.read("n-c1")
    assert problems == [] and n_record["spread_pp"] == shown["spread_pp"]
    assert (n_record["N"], n_record["state"], n_record["cap"]) == (n, "passed", output["cap"])
    assert n_record["cohort"] == world.state["step"]["fields"]["cohort"]
    assert n_record["spread"]["output"] == {"path": OUTPUT.format(candidate=1),
                                            "sha256": sha256(world.output())}  # fmt: skip
    head = git(world.repo, "rev-parse", "HEAD")
    assert n_record["code"]["commit"] == head
    assert len(world.reads) == 1 and sorted(world.reads[0]) == sorted(world.harm)


def test_the_output_is_deterministic_and_holds_no_time(world, tmp_path, monkeypatch):
    world.at_choice()
    world.main()
    first = world.output().read_bytes()
    other = World(tmp_path / "other", monkeypatch)  # another checkout, the same files
    other.at_choice()
    other.main()
    assert other.output().read_bytes() == first and b"written_at" not in first


def test_a_short_power_raises_n_in_the_record(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch, gates=gates_with(power_af=power_gate(0.7, 150)))
    world.at_choice()
    world.main()
    assert world.read("n-c1")[0]["N"] == 150 > world.expected()[1]


@pytest.mark.parametrize(
    "error",
    [
        LabelsPending("r-pending: not labeled yet"),
        LabelingProblem("r-problem: its 3 labeling rounds are spent"),
    ],
)
def test_labels_pending_or_a_labeling_problem_refuse_and_write_nothing(world, error):
    world.at_choice()
    world.fail = error
    assert world.main_refused(str(error).split(":")[0])
    assert not world.output().exists() and world.read("n-c1") == (None, [])


@pytest.mark.parametrize(
    "stage,candidate,problems",
    [("validation", 1, ()), ("budget", 1, ()), ("choice", 2, ()), ("choice", 1, ("x",))],
)
def test_outside_the_choice_step_nothing_sealed_is_opened(world, monkeypatch, stage, candidate,
                                                          problems):  # fmt: skip
    world.at_choice(stage=stage, candidate=candidate, problems=problems)
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    assert world.main_refused("spread point is closed")
    assert world.reads == [] and guard.attempts == [] and SENTINEL not in world.refusal
    assert not world.output().exists() and world.read("n-c1") == (None, [])


@pytest.mark.parametrize("name", ["study3_spread.py", "train.py"])
def test_a_changed_pinned_file_refuses_before_any_label(world, monkeypatch, name):
    world.at_choice()
    world.edit_pinned(name)
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    assert world.main_refused("the spread's code differs from the validation fixation's pin")
    assert name in world.refusal and world.reads == [] and guard.attempts == []
    assert not world.output().exists() and world.read("n-c1") == (None, [])


def test_a_missing_fixation_or_a_dirty_checkout_refuses_before_any_label(world):
    world.at_choice()
    (world.repo / "notes.txt").write_text("an untracked file\n")
    assert world.main_refused("clean checkout") and world.reads == []
    (world.repo / "notes.txt").unlink()
    (world.repo / FIXATION).unlink()
    assert world.main_refused("validation fixation is missing") and world.reads == []
    assert not world.output().exists() and world.read("n-c1") == (None, [])


def test_a_file_changed_during_the_run_refuses_at_write_time(world):
    world.at_choice(blocks=22, while_reading=lambda: world.edit_pinned("study3_spread.py"))
    assert world.main_refused(match="pin") and not world.output().exists()
    assert world.read("n-c1") == (None, [])
    assert len(world.reads) == 1  # the labels were read, and nothing was written after


def test_the_n_record_carries_the_code_it_ran_with_and_survives_a_later_fix(world):
    world.at_choice(blocks=22)
    world.main()
    assert world.read("n-c1")[0]["spread"]["code"] == world.pinned()
    assert world.read("n-c1")[0]["spread"]["script"]["sha256"] == world.pinned()[0]["sha256"]
    world.edit_pinned("study3_spread.py")
    world.edit_pinned("train.py")
    assert world.read("n-c1")[1] == []


@pytest.mark.parametrize("how", ["replaced", "extra", "missing"])
def test_a_cohort_other_than_the_blocks_endpoints_is_refused(world, how):
    world.at_choice()
    runs = [e.run_id for e in world.endpoints]
    cohort = {"replaced": ["another-run", *runs[1:]], "extra": [*runs, "another-run"],
              "missing": runs[1:]}[how]  # fmt: skip
    world.at_choice(cohort=cohort)
    assert world.main_refused(r"the cohort's \d+ run ids are not the 88 A and F endpoints")
    assert world.reads == [] and world.read("n-c1") == (None, [])


def test_a_block_whose_a_f_is_not_supported_is_refused(world):
    world.at_choice()
    first = world.blocks[0]
    world.endpoints = [dataclasses.replace(e, K=e.K * 2) if e is first[0] else e
                       for e in world.endpoints]  # fmt: skip
    assert world.main_refused("1 of the cohort's 22 blocks") and world.reads == []


def test_a_rerun_reuses_an_identical_output_and_the_n_record_is_written_once(world):
    world.at_choice()
    world.main()
    output, record = world.output().read_bytes(), decision_path(world.root, "n-c1")
    record.unlink()  # a crash after the output was written: the n record is not
    world.main()
    assert world.output().read_bytes() == output and world.read("n-c1")[1] == []
    written = record.read_bytes()
    assert world.main_refused("already written") and record.read_bytes() == written
    record.unlink()
    world.harm = {run: dataclasses.replace(h, harmful=h.harmful + 1 if i == 0 else h.harmful)
                  for i, (run, h) in enumerate(world.harm.items())}  # fmt: skip
    assert world.main_refused("exists with other content")
    assert world.output().read_bytes() == output and world.read("n-c1") == (None, [])


def test_the_real_closure_holds_every_module_the_script_loads():
    code = ("import importlib.util, json, sys; s = importlib.util.spec_from_file_location('s', "
            "sys.argv[1]); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "print(json.dumps([getattr(v, '__file__', None) or '' for v in "
            "list(sys.modules.values())]))")  # fmt: skip
    files = json.loads(subprocess.run([sys.executable, "-c", code, str(REPO_ROOT / SPREAD_SCRIPT)],
                                      capture_output=True, text=True, check=True,
                                      cwd=REPO_ROOT).stdout)  # fmt: skip
    package = REPO_ROOT / "src" / "zero_field_refusal"
    loaded = {Path(f).resolve().relative_to(REPO_ROOT).as_posix() for f in files
              if f and Path(f).resolve().is_relative_to(package)}  # fmt: skip
    pin = {e["path"] for e in spread_pin(REPO_ROOT)}
    assert loaded <= pin and {SPREAD_SCRIPT, "src/zero_field_refusal/analysis/study3_labels.py",
                              "src/zero_field_refusal/run_judging.py"} <= pin  # fmt: skip
