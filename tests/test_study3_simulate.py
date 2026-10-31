"""Study 3's simulation script (scripts/study3_simulate.py), run in process on a tiny world: a Git
checkout holding a registration (its clone per test), a study-2 RUNS_DIR of synthetic runs (each
with sealed folders holding a sentinel text, never opened) and a scratch RUNS_DIR. The grid and
the sizes are patched: two scenarios and the two gates, M = 6, 200 and 400 resamples, 400
calibration studies; the parent's workers are one process (the patched constants hold in it),
and one route runs its draws over two."""

import hashlib
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from test_study3_campaigns import git, script_module
from test_study3_seed_vectors import SENTINEL, THETA0, write_run
from test_study3_simulation import world_vectors

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import study3_decisions as decisions
from zero_field_refusal.analysis import study3_simulation as sim
from zero_field_refusal.analysis.study3_decisions import (
    SIMULATION_FIXATION,
    SIMULATION_RESULTS,
    clopper_pearson,
    decision_path,
    read_decision,
    write_decision,
)
from zero_field_refusal.analysis.study3_progress import THETA0_RUN, progress
from zero_field_refusal.analysis.study3_seed_vectors import INPUTS_PATH, STUDY2_ROOTS
from zero_field_refusal.paths import REPO_ROOT

SIMULATE = REPO_ROOT / "scripts" / "study3_simulate.py"
SUMMARY = REPO_ROOT / "scripts" / "study3_summary.py"
OPTIMIZERS = {"A": "adamw", "F": "adam_frozen", "P": "adamw_prewarmed"}
REGISTRATION = "# Study 3 preregistration\n\n### Amendment 1: x\n\n### Amendment 2: y\n"


def tiny_grid() -> tuple:
    """C02 (coverage) and P20 (power) at a Δ of 6 pp, whose power is not short, and the two
    gates."""
    big = replace(sim.SCENARIOS_BY_NAME["P20"], delta_af_pp=6.0, delta_pf_pp=6.0)
    return (sim.SCENARIOS_BY_NAME["C02"], big, sim.SCENARIOS_BY_NAME["G1"],
            sim.SCENARIOS_BY_NAME["G2"])  # fmt: skip


# The calibrations computed once, by the pairs and the setup's inputs: ``run`` recomputes them in
# every test, and only the route's run does it afresh (its workers, spawned, keep nothing)
CALIBRATED: dict[tuple, dict] = {}
REAL_CALIBRATE_ALL = sim.calibrate_all


def calibrate_once(setup, base, *, pairs, workers, studies):
    key = (setup.fingerprint, base, tuple((sc, c) for sc, c in pairs), studies)
    if workers > 1 or key not in CALIBRATED:
        CALIBRATED[key] = REAL_CALIBRATE_ALL(setup, base, pairs=pairs, workers=workers,
                                             studies=studies)  # fmt: skip
    return CALIBRATED[key]


def patch(monkeypatch) -> None:
    monkeypatch.setattr(sim, "calibrate_all", calibrate_once)
    monkeypatch.setattr(decisions, "COVERAGE_RUNS", 6)
    monkeypatch.setattr(decisions, "SCREEN_RESAMPLES", 200)
    monkeypatch.setattr(decisions, "VERIFICATION_RESAMPLES", 400)
    monkeypatch.setattr(decisions, "CALIBRATION_STUDIES", 400)
    monkeypatch.setattr(sim, "UNIT", 4)
    monkeypatch.setattr(sim, "SCENARIOS", tiny_grid())
    monkeypatch.setattr(sim, "_DRAWS_LIMIT", 64)  # the calibration kept across the tests


def study2_runs(base: Path) -> tuple[list[Path], Path]:
    """Study 2's two roots holding the synthetic whole vectors' points as run records (their
    K and accuracies against θ₀), and θ₀'s evaluation."""
    roots = [base / "runs" / name for name in STUDY2_ROOTS]
    for root in roots:
        root.mkdir(parents=True)
    stamp = 100
    for vector in world_vectors():
        for arm, points in vector.arms.items():
            for i, p in enumerate(points):
                stamp += 1
                row = {
                    "benchmark_accuracy": THETA0["benchmark_accuracy"] - p.mmlu_loss_pp / 100,
                    "factual_accuracy": THETA0["factual_accuracy"] - p.factual_loss_pp / 100,
                }
                write_run(roots[i % 2], f"20261001T{stamp:06d}Z-{arm}-{p.rate:g}-s{vector.seed}",
                          arm, p.rate, vector.seed, k=p.K, row=row)  # fmt: skip
    theta0 = base / "runs" / "study2_theta0" / "20261014T192927Z-theta0-s2000-abb703a1-a1"
    theta0.mkdir(parents=True)
    (theta0 / "summary.json").write_text(json.dumps({**THETA0, "K_benign": 0.0}))
    return roots, theta0


class Simulate:
    """The script loaded as a module, its checkout ``repo``."""

    def __init__(self, repo: Path, monkeypatch, capsys) -> None:
        self.repo, self.capsys = repo, capsys
        self.script = script_module(SIMULATE, "study3_simulate")
        monkeypatch.setattr(self.script, "REPO", repo)

    def __call__(self, *argv) -> tuple[int, str, str]:
        self.capsys.readouterr()
        code, message = 0, ""
        try:
            self.script.main([str(a) for a in argv])
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
            message = exc.code if isinstance(exc.code, str) else ""
        out = self.capsys.readouterr()
        return code, out.out, out.err + message


@pytest.fixture(scope="module")
def origin(tmp_path_factory) -> Path:
    """The tiny checkout after ``inputs`` and ``fixation``, each committed (the fixation alone),
    as the researcher would: its clones are the tests' checkouts."""
    base = tmp_path_factory.mktemp("origin")
    repo = base / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "docs" / "study3-preregistration.md").write_text(REGISTRATION)
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "the registration")
    roots, theta0 = study2_runs(base)
    with pytest.MonkeyPatch.context() as mp:
        patch(mp)
        mp.setenv("RUNS_DIR", str(base / "runs"))
        script = script_module(SIMULATE, "study3_simulate")
        mp.setattr(script, "REPO", repo)
        script.main(["inputs", "--roots", *map(str, roots), "--theta0", str(theta0)])
        git(repo, "add", INPUTS_PATH)
        git(repo, "commit", "-qm", "the simulation's inputs")
        script.main(["fixation", "--workers", "1"])
        git(repo, "add", SIMULATION_FIXATION)
        git(repo, "commit", "-qm", "the simulation's fixation")
    return repo


@pytest.fixture
def world(origin, tmp_path, monkeypatch, capsys):
    """A clone of the tiny checkout, a scratch RUNS_DIR and the script."""
    patch(monkeypatch)
    repo = tmp_path / "repo"
    git(tmp_path, "clone", "-q", str(origin), str(repo))
    runs = tmp_path / "runs"
    runs.mkdir()
    return repo, runs, Simulate(repo, monkeypatch, capsys)


def head_code(repo: Path) -> dict:
    commit = git(repo, "rev-parse", "HEAD")
    return {
        "commit": commit,
        "gpu_code_sha256": code_identity.gpu_code_identity(commit, repo)["sha256"],
    }


def progress_of(runs: Path, repo: Path) -> dict:
    """The state, with an engineering record of the checkout's code written first, θ₀'s
    evaluation and a launch journal (no study-3 run started yet)."""
    code = head_code(repo)
    theta0 = runs / THETA0_RUN / "summary.json"
    if not theta0.exists():
        theta0.parent.mkdir(parents=True)
        theta0.write_text(json.dumps({**THETA0, "K_benign": 0.0}))
        (runs / "launch").mkdir()
        (runs / "launch" / "study3-c1-engineering.jsonl").write_text("")
    if not decision_path(runs, "engineering").exists():
        checks = {}
        for name in ("dry_run", "bridge", "theta0_utility", "stage0"):
            path = runs / "evidence" / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{name}\n")
            checks[name] = {
                "path": f"evidence/{name}.json",
                "passed": True,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        write_decision(
            runs,
            "engineering",
            {
                "kind": "engineering",
                "state": "passed",
                "written_at": "2026-10-30T00:00:00+00:00",
                "code": code,
                "checks": checks,
            },
            repo=repo,
        )
    return progress(runs, repo=repo, expected_code=code["gpu_code_sha256"])


def ran(found: tuple[int, str, str]) -> str:
    code, out, err = found
    assert code == 0, err
    return out


def refused(found: tuple[int, str, str], *words: str) -> str:
    code, _, error = found
    assert code not in (0, None), found
    assert all(w in error for w in words), (words, error)
    return error


def simulation_files(runs: Path) -> list[Path]:
    folder = runs / "study3" / "simulation"
    return sorted(p for p in folder.rglob("*") if p.is_file()) if folder.exists() else []


def test_the_inputs_and_the_fixation_were_written_and_committed(origin) -> None:
    fixation = json.loads((origin / SIMULATION_FIXATION).read_text())
    assert fixation["J"] == 105 and fixation["M"] == 6
    assert fixation["inputs"]["path"] == INPUTS_PATH
    assert [s["name"] for s in fixation["scenarios"]] == ["C02", "P20", "G1", "G2"]
    assert set(fixation["scenarios"][0]["calibration"]) == {"1", "2"}
    inputs = (origin / INPUTS_PATH).read_text()
    assert len(json.loads(inputs)["vectors"]) == 25 and SENTINEL not in inputs
    # study 2's runs are named relative to RUNS_DIR: no home directory or login is committed
    record = json.loads(inputs)
    assert record["sources"] == list(STUDY2_ROOTS)
    assert record["theta0"]["path"] == "study2_theta0/20261014T192927Z-theta0-s2000-abb703a1-a1"
    assert str(origin.parent) not in inputs and str(Path.home()) not in inputs
    assert git(origin, "show", "--name-only", "--format=", "HEAD") == SIMULATION_FIXATION


def test_the_route_writes_a_passed_record_that_progress_accepts(world) -> None:
    repo, runs, simulate = world
    out = ran(simulate("run", "--runs-root", runs, "--workers", 2))
    assert "\nlevel 0.9" in out and any(f"validation_blocks {n}" in out for n in (28, 40))
    assert "\npower_af_1pct_missing " in out
    ran(simulate("verify", "--runs-root", runs, "--workers", 2))
    out = ran(simulate("record", "--runs-root", runs))
    assert "passed" in out and str(decision_path(runs, "simulation")) in out
    record, problems = read_decision(runs, "simulation", repo=repo)
    assert problems == [] and record["state"] == "passed" and record["level"] in decisions.LEVELS
    assert record["inputs"]["seed_vectors"]["path"] == INPUTS_PATH
    results = json.loads((runs / SIMULATION_RESULTS).read_text())
    assert results["written_at"].endswith("Z") and results["code"]["commit"] == head_code(repo)[
        "commit"]  # fmt: skip
    state = progress_of(runs, repo)
    assert state["stage"] == "search", state.get("problems")


def test_an_interrupted_run_resumes_to_the_same_bytes(world, tmp_path, monkeypatch) -> None:
    repo, runs, simulate = world
    control = tmp_path / "control"
    control.mkdir()
    ran(simulate("run", "--runs-root", control, "--workers", 1))
    real, left = sim.execute, [2]

    def cut(units, *args, **kwargs):
        units = list(units)[: left[0]]
        real(units, *args, **kwargs)
        left[0] -= len(units)
        if left[0] <= 0:
            raise KeyboardInterrupt

    monkeypatch.setattr(sim, "execute", cut)
    with pytest.raises(KeyboardInterrupt):
        simulate("run", "--runs-root", runs, "--workers", 1)
    monkeypatch.setattr(sim, "execute", real)
    assert len(simulation_files(runs)) == 2
    ran(simulate("run", "--runs-root", runs, "--workers", 2))
    screen = "study3/simulation/screen.json"
    assert (runs / screen).read_bytes() == (control / screen).read_bytes()
    for where in (runs, control):
        ran(simulate("verify", "--runs-root", where, "--workers", 1))
        ran(simulate("record", "--runs-root", where))
    assert (runs / SIMULATION_RESULTS).read_bytes() == (control / SIMULATION_RESULTS).read_bytes()


@pytest.mark.parametrize("how,word", [("untracked", "tracked"), ("dirty", "clean"),
                                      ("with_another_file", "alone")])  # fmt: skip
def test_run_refuses_before_the_first_draw(world, how, word) -> None:
    repo, runs, simulate = world
    if how == "untracked":
        git(repo, "reset", "-q", "--mixed", "HEAD~1")
    elif how == "dirty":
        (repo / "docs" / "study3-preregistration.md").write_text(REGISTRATION + "x\n")
    else:
        git(repo, "reset", "-q", "--soft", "HEAD~1")
        (repo / "docs" / "other.md").write_text("other\n")
        git(repo, "add", "docs/other.md")
        git(repo, "commit", "-qm", "the fixation and another file")
    refused(simulate("run", "--runs-root", runs, "--workers", 1), word)
    assert simulation_files(runs) == []  # no part written


def test_run_refuses_a_part_of_another_commit_or_fixation_naming_it(world) -> None:
    repo, runs, simulate = world
    context = sim.load_context(repo)
    unit = sim.Unit("gates", "G1", 0, 4, 20261101, 200, sim.LEVELS, 28, 60)
    sim.execute([unit], runs, context.ref, "d" * 40, workers=1, context=context)
    part = sim.part_path(runs, "gates", "G1", 0)
    refused(simulate("run", "--runs-root", runs, "--workers", 1), str(part), "another commit")
    assert simulation_files(runs) == [part]


def test_a_second_record_is_refused_by_both_writers(world, monkeypatch, capsys) -> None:
    repo, runs, simulate = world
    for command in ("run", "verify"):
        ran(simulate(command, "--runs-root", runs, "--workers", 1))
    ran(simulate("record", "--runs-root", runs))
    record = decision_path(runs, "simulation")
    before = record.read_bytes()
    refused(simulate("record", "--runs-root", runs), "written once")
    copy = runs / "copy.json"
    shutil.copy(record, copy)
    summary = script_module(SUMMARY, "study3_summary")
    monkeypatch.setattr(summary, "REPO", repo)
    with pytest.raises(SystemExit) as found:
        summary.main(["record", "simulation", "--runs-root", str(runs), "--from", str(copy)])
    assert "already written" in str(found.value.code)
    assert record.read_bytes() == before


def edit(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def missed(row: dict) -> None:
    """A row's X raised above its c, its pass, rate and interval with it."""
    x = row["c"] + 1
    row.update(X=x, passed=False, rate=x / 6, interval=clopper_pearson(x, 6))


def test_verify_refuses_without_a_level_and_record_writes_the_failed_record(world) -> None:
    repo, runs, simulate = world
    ran(simulate("run", "--runs-root", runs, "--workers", 1))
    screen = runs / "study3" / "simulation" / "screen.json"
    edit(screen, lambda s: [missed(r) for r in s["tests"]])  # every level fails
    refused(simulate("verify", "--runs-root", runs, "--workers", 1), "not due")
    out = ran(simulate("record", "--runs-root", runs))
    assert "failed" in out
    record, problems = read_decision(runs, "simulation", repo=repo)
    assert problems == [] and record["level"] is None and record["state"] == "failed"
    assert record["verification"]["passed"] is False
    results = json.loads((runs / SIMULATION_RESULTS).read_text())
    assert {r["due"] for r in results["tests"] if r["test"][0] == "verification"} == {False}
    state = progress_of(runs, repo)
    assert (state["outcome"], state["reason"]) == ("design_stop", "simulation_failed")


def test_a_verification_test_above_c_gives_the_failed_record(world) -> None:
    repo, runs, simulate = world
    for command in ("run", "verify"):
        ran(simulate(command, "--runs-root", runs, "--workers", 1))
    path = runs / "study3" / "simulation" / "verification.json"
    edit(path, lambda v: missed(v["tests"][0]))
    ran(simulate("record", "--runs-root", runs))
    record, problems = read_decision(runs, "simulation", repo=repo)
    assert problems == [] and record["level"] is not None and record["state"] == "failed"
    assert record["verification"]["passed"] is False
    state = progress_of(runs, repo)
    assert (state["outcome"], state["reason"]) == ("design_stop", "simulation_failed")


def test_no_attainable_n_skips_the_verification_and_fails_the_record(world) -> None:
    """The power short at every n of the scan up to 169, whose cap 201 exceeds the 200 reserved
    seeds: no N is attainable, so the verification is not run (the design stop, named), its
    tests are not due, and the record fails."""
    repo, runs, simulate = world
    ran(simulate("run", "--runs-root", runs, "--workers", 1))
    screen = runs / "study3" / "simulation" / "screen.json"

    def short(s: dict) -> None:
        by = s["gates"]["power"]["A-F"][decisions.level_slot(s["level"])]
        for counts in by.values():
            counts["correct"] = 4  # 4 of M = 6: below 0.80
        s["gates"]["n_scan"] = [{"n": n, "by_scenario": {name: {"correct": 4, "studies": 6}
                                                         for name in by}}
                                for n in range(60, 170)]  # fmt: skip
        s["gates"]["scan_level"] = s["level"]

    edit(screen, short)
    refused(simulate("verify", "--runs-root", runs, "--workers", 1), "no attainable n",
            "design stop", "201")  # fmt: skip
    assert not (runs / "study3" / "simulation" / "verification.json").exists()
    out = ran(simulate("record", "--runs-root", runs))
    assert "state failed" in out and "verification not_run no_attainable_n n 169 cap 201" in out
    record, problems = read_decision(runs, "simulation", repo=repo)
    assert problems == [] and record["level"] is not None and record["state"] == "failed"
    assert record["verification"]["passed"] is False
    assert record["gates"]["power_af"]["n_min"] == 169
    results = json.loads((runs / SIMULATION_RESULTS).read_text())
    verification = [r for r in results["tests"] if r["test"][0] == "verification"]
    assert verification and {r["due"] for r in verification} == {False}
    state = progress_of(runs, repo)
    assert (state["outcome"], state["reason"]) == ("design_stop", "simulation_failed")
    assert any("attainable" in r for r in state["reasons"])


def test_verify_is_never_rerolled(world) -> None:
    repo, runs, simulate = world
    refused(simulate("verify", "--runs-root", runs, "--workers", 1), "screen")  # no screen
    ran(simulate("run", "--runs-root", runs, "--workers", 1))
    (repo / "docs" / "later.md").write_text("later\n")
    git(repo, "add", "docs/later.md")
    git(repo, "commit", "-qm", "a later commit")
    refused(simulate("verify", "--runs-root", runs, "--workers", 1), "another commit")
    git(repo, "reset", "-q", "--hard", "HEAD~1")
    ran(simulate("verify", "--runs-root", runs, "--workers", 1))
    verification = runs / "study3" / "simulation" / "verification.json"
    before = verification.read_bytes()
    refused(simulate("verify", "--runs-root", runs, "--workers", 1), "never rerolled")
    refused(simulate("run", "--runs-root", runs, "--workers", 1), "written once")
    assert verification.read_bytes() == before


@pytest.mark.parametrize("change,words", [
    ("duplicate", ("twice",)), ("omit", ("missing",)), ("extra", ("not in the fixation",)),
    ("flip_pass", ("recomputed",)), ("results", ("results.json", "exists")),
])  # fmt: skip
def test_a_tampered_results_file_is_refused_by_record(world, change, words) -> None:
    repo, runs, simulate = world
    for command in ("run", "verify"):
        ran(simulate(command, "--runs-root", runs, "--workers", 1))
    screen = runs / "study3" / "simulation" / "screen.json"
    if change == "duplicate":
        edit(screen, lambda s: s["tests"].append(dict(s["tests"][0])))
    elif change == "omit":
        edit(screen, lambda s: s["tests"].pop())
    elif change == "extra":
        edit(screen, lambda s: s["tests"].append({**s["tests"][0], "test": [
            "calibration", "C99", 36, "0.975", "A-F"]}))  # fmt: skip
    elif change == "flip_pass":
        edit(screen, lambda s: s["tests"][0].update(passed=not s["tests"][0]["passed"]))
    else:
        (runs / SIMULATION_RESULTS).write_text('{"kind": "simulation-results"}\n')
    refused(simulate("record", "--runs-root", runs), *words)
    assert not decision_path(runs, "simulation").exists()
    if change != "results":
        assert not (runs / SIMULATION_RESULTS).exists()


def test_the_script_prints_no_harm_number(world, tmp_path) -> None:
    repo, runs, simulate = world
    printed = []
    for argv in (("run", "--runs-root", runs, "--workers", 1),
                 ("verify", "--runs-root", runs, "--workers", 1),
                 ("record", "--runs-root", runs)):  # fmt: skip
        printed.append(ran(simulate(*argv)))
    words = [w for w in "\n".join(printed).lower().split() if "/" not in w]  # paths aside
    text = " ".join(words)
    assert words and not any(word in text for word in ("spread", "mean", "truth", "sd ", "sigma",
                                                       "offset", "harm"))  # fmt: skip
    for word in words:  # numbers, ids and hashes
        assert word.replace(".", "").replace("_", "").replace("-", "").isalnum(), word


def test_the_inputs_command_refuses_a_second_inputs_record(world) -> None:
    repo, runs, simulate = world
    refused(simulate("inputs", "--roots", runs, "--theta0", runs), "written once")


def test_the_inputs_command_refuses_runs_outside_runs_dir(world, tmp_path, monkeypatch) -> None:
    repo, runs, simulate = world
    monkeypatch.setenv("RUNS_DIR", str(runs))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    out = tmp_path / "inputs.json"
    refused(simulate("inputs", "--roots", elsewhere, "--theta0", runs / "x", "--out", out),
            "not under RUNS_DIR")  # fmt: skip
    assert not out.exists()


def test_the_fixation_command_refuses_a_second_fixation(world) -> None:
    repo, runs, simulate = world
    refused(simulate("fixation", "--workers", 1), "written once")


def test_record_refuses_a_screen_of_another_commit(world) -> None:
    repo, runs, simulate = world
    for command in ("run", "verify"):
        ran(simulate(command, "--runs-root", runs, "--workers", 1))
    (repo / "docs" / "later.md").write_text("later\n")
    git(repo, "add", "docs/later.md")
    git(repo, "commit", "-qm", "a later commit")
    refused(simulate("record", "--runs-root", runs), "another commit")
    assert not (runs / SIMULATION_RESULTS).exists()


def test_the_power_gate_reads_the_results_as_registered(world) -> None:
    """The record's power is the smallest correct-direction rate over the power scenarios at L."""
    repo, runs, simulate = world
    for command in ("run", "verify"):
        ran(simulate(command, "--runs-root", runs, "--workers", 1))
    ran(simulate("record", "--runs-root", runs))
    record, _ = read_decision(runs, "simulation", repo=repo)
    results = json.loads((runs / SIMULATION_RESULTS).read_text())
    by = results["gates"]["power_af"]["by_scenario"]
    assert set(by) == {"P20"} and math.isclose(record["gates"]["power_af"]["value"],
                                               by["P20"]["correct"] / 6)  # fmt: skip
    report = results["gates"][decisions.POWER_REPORT]  # beside it, the power at 1 %, reported
    assert report["missing_rate"] == 0.01 and report["level"] == results["gates"]["power_af"][
        "level"]  # fmt: skip
    assert (report["by_scenario"]["P20"]["studies"], report["by_scenario"]["P20"]["draws"]) == (
        by["P20"]["studies"], by["P20"]["draws"])  # fmt: skip
    assert record["gates"][decisions.POWER_REPORT] == {
        "value": report["by_scenario"]["P20"]["correct"] / 6, "requirement": None,
        "passed": None}  # fmt: skip
