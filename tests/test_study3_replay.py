"""Study 3's integration replay (the registration's amendment 7, "The order of the dry run"): a
technical mark of the engineering stage, its cycle, the decision under the fix's code, the
simulation's record, the first campaign and its readback, then on synthetic records through the
stop and the cohorts' freeze, an interruption and its readback included; and the rules of amendment
7 that only show on that route (a raised floor, a withheld or reopened P, a technical mark, the
reserve).

On files, through their real code: the campaign writer (``scripts/study3_campaigns.py``), the
summary script (``scripts/study3_summary.py``: the decision records, the fixations, the versions
records, the check and the guard), ``qualify.py supersede``, ``read_state``, the decision records'
readers, the simulation record's J check, and the analysis freeze with its gate. Given, as in
``test_study3_package2.Route``: the launcher's and the tracker's files (synthetic runs, no GPU),
the part records ``qualify.py part`` would write of the engineering cases, and the qualification's
reading of the numerical stage (its rates qualified)."""

import json
import math
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import pytest
from test_qualification_records import ENGINEERING_HEADER, beyond
from test_qualification_records import World as QualificationWorld
from test_qualify import load_script
from test_study3_campaigns import START, campaign_of, cells_of, git, run_config, sha256, target
from test_study3_cycle import FixedRoute, built, restored
from test_study3_decisions import results_row
from test_study3_package2 import JUDGE, SealedGuard, rates_of, round0

from zero_field_refusal import qualification as q
from zero_field_refusal.analysis import qualification_records
from zero_field_refusal.analysis import study3_freeze as fz
from zero_field_refusal.analysis.study3_decisions import (
    SIMULATION_RESULTS,
    coverage_decisions,
    decision_path,
    read_decision,
)
from zero_field_refusal.analysis.study3_progress import code_chain, read_state
from zero_field_refusal.analysis.study3_records import SEEDS
from zero_field_refusal.launch import campaign_header, load_campaign

BLOCKS = SEEDS["confirmation"][1]
HARMFUL = {"A": 1.0, "F": 1.2, "P": 1.1}  # K_harmful / K_benign: A's and F's bracket K*_h
MARK = q.QualificationCase(10002, "adamw_prewarmed", START["P"]["l"], 1, True)  # as case 17


def regular(arm, rate, seed, candidate) -> dict:
    found = target(arm, rate, seed, candidate)
    return {**found, "k_harmful": found["k"] * HARMFUL[arm]}


def missing(arm_seeds) -> Callable[..., dict]:
    """``regular``, but an arm's upper rate below K* on its seeds: unsupported in those blocks."""

    def values(arm, rate, seed, candidate):
        found = regular(arm, rate, seed, candidate)
        hit = seed in arm_seeds.get(arm, ()) and math.isclose(rate, START[arm]["u"])
        return {**found, "k": 5.0} if hit else found

    return values


def events(state) -> list[str]:
    return [e["event"] for e in state["route"]]


class Replay(FixedRoute):
    """The campaign writer's route in amendment 7's order: its engineering stage ends in a technical
    failure of one case, that case's cycle and the campaign's rerun, and the engineering and
    simulation records are written under the fix's code. ``pf_floor``: P–F's floor in the
    simulation's reading (None: withheld)."""

    pf_floor: int | None = 36

    def to_search(self) -> None:
        assert self.step()["record"] == "engineering"
        self.engineering = self.campaign("engineering")
        self.launch(self.engineering)
        self.engineering_cycle()
        self.relaunch(self.engineering)
        self.record("engineering", self.records.engineering(code=self.code))
        self.record("simulation", self.simulation())

    def engineering_cycle(self) -> None:
        """Every case of the engineering campaign recorded as ``qualify.py part`` records it, under
        the old code: qualified, but ``MARK``'s baseline, a completed technical failure. A fix
        committed, then ``qualify.py supersede --engineering`` from its clean checkout: the code in
        force is the fix's."""
        stage = QualificationWorld(self.base, ((("adamw", START["A"]["l"]), (10000,)),),
                                   name="study3-c1-engineering", header=ENGINEERING_HEADER,
                                   role="engineering")  # fmt: skip
        old = {"commit": self.code["commit"], "files": 1, "sha256": self.code["gpu_code_sha256"]}
        runs = load_campaign(self.engineering)[1]
        for run, run_id in zip(runs, self.launched[str(self.engineering)], strict=True):
            cfg = run_config(run.config, run.overrides)
            case = q.QualificationCase(run.seed, str(cfg.optimizer.name), float(cfg.optimizer.lr),
                                       1, True)  # fmt: skip
            change = beyond("baseline", "launcher~baseline") if case == MARK else None
            stage.qualified(case, q.launcher_identity(self.root / run.experiment / run_id),
                            code=old, change=change)  # fmt: skip
        (self.repo / "src" / "zero_field_refusal" / "train.py").write_text("x = 2\n")
        self.commit("the fix")
        qualify = load_script()
        self.monkeypatch.setattr(qualify, "load_run_config", run_config)
        self.monkeypatch.setattr(qualify, "git_state",
                                 lambda *a, **k: {"commit": "c" * 40, "dirty": False})  # fmt: skip
        qualify.main(["supersede", "--campaign", str(self.engineering), "--candidate", "1",
                      "--runs-root", str(self.root), "--engineering", "--case", MARK.name,
                      "--record", str(stage.folder(MARK) / "baseline-attempt-1.json"),
                      "--diagnosis", "d", "--fix", "f"])  # fmt: skip
        self.code = {"commit": git(self.repo, "rev-parse", "HEAD"),
                     "gpu_code_sha256": self.identity()["sha256"]}  # fmt: skip

    def simulation(self) -> dict:
        """The simulation record over results that give P–F's floor: at 45, P–F's cap test fails
        at 36 alone and the verification counts at 45; withheld, P–F's verification tests fail."""
        record = self.records.simulation(code=self.code)
        if self.pf_floor == 36:
            return record
        path = self.root / SIMULATION_RESULTS
        results = json.loads(path.read_text())
        for i, row in enumerate(results["tests"]):
            stream, _, point, slot, tested = identity = tuple(row["test"])
            if slot != "0.975" or tested == "A-F":
                continue
            if self.pf_floor is None and stream == "verification" and tested == "P-F":
                results["tests"][i] = results_row(identity, 0.975, row["c"] + 1, row["c"],
                                                  floors=(36,))  # fmt: skip
            elif self.pf_floor == 45 and point == "cap" and stream == "calibration":
                if tested == "P-F":
                    row["below"]["36"] = results_row(identity, 0.975, row["c"] + 1, row["c"],
                                                     floors=(36, 60))["below"]["36"]  # fmt: skip
            elif self.pf_floor == 45 and point == "cap":
                results["tests"][i] = results_row(identity, 0.975, row["X"], row["c"],
                                                  floors=(45,))  # fmt: skip
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        decided, problems = coverage_decisions(results, self.records.fixation)
        assert problems == []
        return {**record, "reading": decided["reading"], "levels_tried": decided["levels_tried"],
                "inputs": {**record["inputs"], "results": {"path": SIMULATION_RESULTS,
                                                           "sha256": sha256(path)}}}  # fmt: skip

    def launch(self, path, values=target, technical=(), dirty=()) -> list[str]:
        """``Route``'s; no judge output in a confirmation run before the freeze."""
        ids = super().launch(path, values, technical, dirty)
        if campaign_header(path)["step"] == "confirmation":
            for run_id in ids:
                with self.guard.off():
                    (self.root / "study3" / run_id / JUDGE / "wildguard.jsonl").unlink()
                    (self.root / "study3" / run_id / JUDGE).rmdir()
        return ids

    def unfinished(self, path, values) -> tuple[str, Callable[[], None]]:
        """The campaign launched, its last run started and not ended (its end line and terminal
        record gone, its meta.json running); that run, and what finishes it."""
        run_id = self.launch(path, values)[-1]
        run = self.root / "study3" / run_id
        log = self.root / "launch" / f"{load_campaign(path)[0]}.jsonl"
        saved = {file: file.read_bytes()
                 for file in (log, run / "meta.json", run / "terminal.json")}  # fmt: skip
        log.write_text("".join(log.read_text().splitlines(keepends=True)[:-1]))
        (run / "terminal.json").unlink()
        meta = json.loads(saved[run / "meta.json"])
        (run / "meta.json").write_text(json.dumps({**meta, "status": "running"}))

        def finish() -> None:
            for file, data in saved.items():
                file.write_bytes(data)

        return run_id, finish

    def read(self) -> tuple[dict, list]:
        return read_state(self.root, repo=self.repo, expected_code=self.code["gpu_code_sha256"],
                          theta0=self.runs.theta0)  # fmt: skip

    def n_record(self, N: int, **over) -> None:
        """The n record with the fields the state's step names."""
        fields = self.state()["step"]["fields"]
        self.record("n-c1", self.records.n(1, N, rates=fields["rates"], active=fields["active"],
                                          cohort=fields["cohort"], code=self.code,
                                          **over))  # fmt: skip

    def blocks_through(self, last: int) -> dict[int, Path]:
        """The confirmation's blocks through the seed ``last``, a file each."""
        out = self.campaign("confirmation", "--blocks", BLOCKS[0], last)
        return {campaign_of(p)["cells"][0]["seeds"][0]: p for p in out.glob("*.yaml")}

    def confirm(self, N: int, last: int, values) -> None:
        """The validation launched, the n record, the supplement and the blocks through ``last``."""
        self.validated()
        self.n_record(N)
        self.supplement()
        for _, path in sorted(self.blocks_through(last).items()):
            self.launch(path, values)

    def frozen(self) -> dict:
        """The analysis freeze at this commit, committed alone, and read through its gate."""
        code = {**self.code, "commit": git(self.repo, "rev-parse", "HEAD")}
        record = fz.analysis_freeze(self.root, 1, code=code, repo=self.repo,
                                    theta0=self.runs.theta0, ledger_runs=())  # fmt: skip
        (self.repo / fz.freeze_name(1)).write_text(json.dumps(record, indent=2, sort_keys=True))
        self.commit("the analysis freeze", fz.freeze_name(1))
        assert fz.read_freeze(1, runs_root=self.root, repo=self.repo) == (record, [])
        return record


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory) -> dict:
    """``Replay`` built through the validation fixation once per P–F floor a test asks for."""
    return {}


@pytest.fixture
def replay(snapshots, tmp_path_factory, monkeypatch):
    """A pristine copy of the snapshot of P–F's floor, its reads guarded (``SealedGuard``)."""

    def at(pf_floor: int | None = 36) -> Replay:
        if pf_floor not in snapshots:

            def build(found: Replay) -> None:
                found.pf_floor = pf_floor
                found.build_to("fixation")

            snapshots[pf_floor] = built(Replay, tmp_path_factory.mktemp("replay"), build)
        found = restored(Replay, snapshots[pf_floor], monkeypatch)
        found.guard = SealedGuard(monkeypatch, runs_root=found.root)
        return found

    return at


def test_the_spine_from_the_engineering_mark_to_the_cohorts_freeze(tmp_path, monkeypatch) -> None:
    world = Replay(tmp_path, monkeypatch)
    world.guard = SealedGuard(monkeypatch, runs_root=world.root)
    old = world.code["gpu_code_sha256"]
    world.to_search()
    fix = world.code["gpu_code_sha256"]
    # the cycle's own record: the engineering stage's, from the old code to the fix's, whose fix
    # starts the chain the engineering record reads: already applied, and no break
    (cycle,) = qualification_records.supersessions(world.root, 1)
    assert cycle["old_record"]["stage"] == "engineering" and old != fix
    assert (cycle["old_gpu_code"], cycle["fix_gpu_code"]) == (old, fix)
    written = json.loads((world.root / "qualification" / "c1" / "supersede-1.json").read_text())
    assert (len(written["launchers"]), written["kept_cases"], written["held"]) == (24, [], [])
    assert code_chain(fix, {1: [cycle]}) == ([fix], {1: fix, 2: fix}, [])
    engineering, problems = read_decision(world.root, "engineering", repo=world.repo)
    assert problems == [] and engineering["code"]["gpu_code_sha256"] == fix
    simulation, problems = read_decision(world.root, "simulation", repo=world.repo)
    assert problems == []
    assert simulation["reading"] == {"level": 0.975, "pf_confirmatory": True,
                                     "floors": {"A-F": 36, "P-F": 36}}  # fmt: skip
    # the first search campaign issued, and read back
    state, endpoints = world.read()
    assert state["problems"] == [] and state["step"]["campaign"] == "search-round0"
    assert {(e.cell.step, e.state) for e in endpoints} == {("engineering", "superseded"),
                                                           ("engineering", "usable")}  # fmt: skip
    issued = world.campaign("search-round0")
    assert sorted(cells_of(load_campaign(issued)[1])) == sorted(state["step"]["cells"])
    assert world.read()[0]["step"] == state["step"]
    world.launch(issued, round0)
    state, endpoints = world.read()
    assert state["problems"] == [] and state["step"]["campaign"] == "search-round1"
    assert sum(e.cell.step == "search" for e in endpoints) == 24
    world.after_round0()
    # N 88: the cap 106, in the n record
    world.validated()
    world.n_record(88)
    record, problems = read_decision(world.root, "n-c1", repo=world.repo)
    assert problems == [] and (record["N"], record["cap"]) == (88, 106)
    assert world.step()["campaign"] == "confirmation"  # N ≤ 90: no budget decision
    world.supplement()
    seeds = list(BLOCKS[:106])
    blocks = world.blocks_through(seeds[-1])
    assert sorted(blocks) == seeds
    world.refused("campaigns", "confirmation", "--runs-root", world.root, "--candidate", 1,
                  "--out", world.out / "beyond", "--blocks", seeds[0], seeds[-1] + 1)  # fmt: skip
    world.ok("summary", "check", "--runs-root", world.root, "--campaign", blocks[seeds[0]])
    values = missing({"A": seeds[86:]})  # A–F's counter stays at 86: the cap stops it
    assert world.guarded(seeds[0], blocks[seeds[0]]).code == 0
    world.launch(blocks[seeds[0]], values)
    world.launch(blocks[seeds[1]], values)
    # an interruption: a run of the third block started and not ended
    run_id, finish = world.unfinished(blocks[seeds[2]], values)
    state = world.state()
    assert state["problems"] == [] and state["stage"] == "confirmation"
    assert (state["step"]["kind"], state["step"]["waiting"]) == ("pending", [run_id])
    assert state["step"]["blocks"] == [{"seed": seeds[2], "runs": [run_id]}]
    assert state["confirmation"]["started"] == seeds[:3]
    assert world.state()["step"] == state["step"]  # read again: the same step, no new block
    assert world.guarded(seeds[3], blocks[seeds[3]]).code != 0
    world.refused("campaigns", "confirmation", "--runs-root", world.root, "--candidate", 1,
                  "--out", world.out / "while-open")  # fmt: skip
    finish()
    state = world.state()
    assert state["problems"] == [] and state["step"]["kind"] == "campaign"
    assert {cell[2] for cell in state["step"]["cells"]} == {seeds[3]}
    assert state["confirmation"]["started"] == seeds[:3]
    assert state["confirmation"]["blocks"][seeds[2]]["A-F"] == "supported"
    # on to the block before the cap: not stopped, A–F's counter short of N
    for seed in seeds[3:-1]:
        world.launch(blocks[seed], values)
    state = world.state()
    assert not state["confirmation"]["stop"]["stopped"]
    assert state["confirmation"]["counters"] == {"A-F": 86, "P-F": 105}
    assert {cell[2] for cell in state["step"]["cells"]} == {seeds[-1]}
    assert world.guarded(seeds[-1], blocks[seeds[-1]]).code == 0
    world.launch(blocks[seeds[-1]], values)
    # the stop: the cap 106 started
    state = world.state()
    assert state["problems"] == []
    assert (state["stage"], state["outcome"], state["reason"]) == (
        "done", "confirmed", "cap_reached")  # fmt: skip
    confirmation = state["confirmation"]
    assert (confirmation["N"], confirmation["cap"]) == (88, 106)
    assert confirmation["stop"] == {"stopped": True, "why": "cap"}
    assert confirmation["readings"] == {"A-F": "confirmatory", "P-F": "confirmatory"}
    # the cohorts' freeze
    record = world.frozen()
    assert record["confirmation"] == {"N": 88, "cap": 106, "started": seeds, "stop": "cap",
                                      "outcome": "confirmed", "active": ["A-F", "P-F"]}  # fmt: skip
    af, pf = record["comparisons"]["A-F"], record["comparisons"]["P-F"]
    assert (af["rule"], [b["seed"] for b in af["blocks"]], af["unknown"]) == (
        "cap", seeds[:86], seeds[86:])  # fmt: skip
    assert (pf["rule"], [b["seed"] for b in pf["blocks"]]) == ("n_reached", seeds[:88])
    assert record["k_harmful"]["left_out"] == []
    assert record["labels"] == {"confirmation_runs": 636, "ledger_runs": 0}
    assert world.guard.attempts == []  # nothing sealed was opened on the way


@pytest.mark.parametrize(("supported", "reading"), [(44, "descriptive"), (45, "confirmatory")])
def test_at_the_cap_each_comparison_reads_against_its_own_floor(replay, supported, reading) -> None:
    """P–F's floor raised to 45 by its checks: on 44 supported blocks it reads descriptive, on 45
    confirmatory. A–F, on 40 supported blocks, reads confirmatory at its own floor of 36."""
    world = replay(45)
    record, problems = read_decision(world.root, "simulation", repo=world.repo)
    assert problems == [] and record["reading"]["floors"] == {"A-F": 36, "P-F": 45}
    seeds = list(BLOCKS[:74])  # N 60: its cap 74
    world.confirm(60, seeds[-1], missing({"A": seeds[40:], "P": seeds[supported:]}))
    state = world.state()
    assert state["problems"] == []
    assert state["confirmation"]["supported"] == {"A-F": 40, "P-F": supported}
    assert state["confirmation"]["stop"]["why"] == "cap"
    assert state["readings"] == {"A-F": "confirmatory", "P-F": reading}
    assert (state["outcome"], state["reason"]) == ("confirmed", "cap_reached")
    assert world.frozen()["comparisons"]["P-F"]["reading"] == reading


def test_a_withheld_p_keeps_its_stages_before_the_confirmation_and_is_not_in_it(replay) -> None:
    """``pf_confirmatory`` false: the search, the numerical stage and the validation keep P; the
    confirmation's blocks hold four runs, and the stop comes with A–F's counter at N."""
    world = replay(None)
    record, problems = read_decision(world.root, "simulation", repo=world.repo)
    assert problems == [] and record["reading"] == {
        "level": 0.975, "pf_confirmatory": False, "floors": {"A-F": 36, "P-F": None}}  # fmt: skip
    world.validated()
    arms = defaultdict(set)
    for path in world.launched:
        arms[campaign_header(Path(path))["step"]] |= {arm for arm, _ in rates_of(Path(path))}
    assert all("P" in arms[step] for step in ("engineering", "search", "numerical", "validation"))
    state = world.state()
    fields = state["step"]["fields"]
    assert fields["active"] == ["A-F"] and set(fields["rates"]) == {"A", "F"}
    assert "withhold_pf" in events(state) and "close_pf" not in events(state)
    world.n_record(60)
    world.supplement()
    blocks = world.blocks_through(BLOCKS[59])
    assert {len(load_campaign(path)[1]) for path in blocks.values()} == {4}
    for _, path in sorted(blocks.items()):
        world.launch(path, regular)
    state = world.state()
    assert state["problems"] == []
    assert (state["stage"], state["outcome"], state["reason"]) == (
        "done", "confirmed", "n_reached")  # fmt: skip
    assert state["readings"] == {"A-F": "confirmatory"}
    assert state["confirmation"]["started"] == list(BLOCKS[:60])
    record = world.frozen()
    assert set(record["comparisons"]) == {"A-F"} and record["validation"]["P-F"] is None


@pytest.mark.parametrize(("pf_floor", "arms"), [(36, {"A", "F", "P"}), (None, {"A", "F"})])
def test_p_closed_scientifically_is_tested_again_at_candidate_2_unless_withheld(
    replay, pf_floor, arms
) -> None:
    """P's and A's validation gates fail at candidate 1: P closes there, A–F's rule reaches
    candidate 2, whose search holds P's cells on its own seeds; with P–F withheld, it does not."""
    world = replay(pf_floor)
    closing = SEEDS["validation"][1][:7]  # supported in 21 of 28
    world.validate(missing({"A": closing, "P": closing}))
    state = world.state()
    assert state["problems"] == []
    assert (state["candidate"], state["stage"], state["step"]["campaign"]) == (
        2, "search", "search-round0")  # fmt: skip
    assert "close_pf" in events(state) and ("reopen_pf" in events(state)) == ("P" in arms)
    issued = world.out / "candidate-2.yaml"
    world.ok("campaigns", "search-round0", "--runs-root", world.root, "--candidate", 2, "--theta0",
             world.runs.theta0, "--out", issued)  # fmt: skip
    found = cells_of(load_campaign(issued)[1])
    assert {arm for arm, _, _ in found} == arms
    assert {seed for _, _, seed in found} == set(SEEDS["search"][2])
    world.launch(issued)
    step = world.step()
    assert step["campaign"] == "numerical" and {cell[0] for cell in step["cells"]} == arms


def test_a_technical_mark_in_p_closes_nothing_and_reaches_no_candidate_2(replay) -> None:
    """A's gate fails at candidate 1 as above, and P's runs end technical: a pause. P is not
    closed, nothing reopens, and A–F's rule moves nothing until the mark is resolved."""
    world = replay()
    closing = SEEDS["validation"][1][:7]
    world.validate(missing({"A": closing}), technical={("P", START["P"]["u"])})
    state = world.state()
    assert (state["candidate"], state["stage"], state["outcome"]) == (1, "validation", None)
    assert (state["step"]["kind"], state["step"]["technical"]) == ("blocked", True)
    assert not {"close_pf", "reopen_pf"} & set(events(state))
    assert "P" in state["active"]["arms"]
    world.refused("campaigns", "search-round0", "--runs-root", world.root, "--candidate", 2,
                  "--theta0", world.runs.theta0, "--out",
                  world.out / "candidate-2.yaml")  # fmt: skip


def test_a_technical_mark_in_p_at_the_confirmation_issues_no_block_of_a_and_f(replay) -> None:
    world = replay()
    world.validated()
    world.n_record(60)
    world.supplement()
    blocks = world.blocks_through(BLOCKS[1])
    world.launch(blocks[BLOCKS[0]], regular, technical={("P", START["P"]["u"])})
    state = world.state()
    assert (state["stage"], state["outcome"]) == ("confirmation", None)
    assert (state["step"]["kind"], state["step"]["technical"]) == ("blocked", True)
    assert "close_pf" not in events(state)
    assert state["active"]["comparisons"] == ["A-F", "P-F"]
    assert world.guarded(BLOCKS[1], blocks[BLOCKS[1]]).code != 0
    world.refused("campaigns", "confirmation", "--runs-root", world.root, "--candidate", 1,
                  "--out", world.out / "no-block")  # fmt: skip
    world.refused("summary", "closure", "--runs-root", world.root, "--candidate", 1, "--arm", "P")


def test_a_second_technical_mark_is_a_pause_naming_a_cycle_or_a_recorded_decision(replay) -> None:
    """The route's engineering cycle is spent: A's runs ending technical at the validation pause it,
    with no terminal report."""
    world = replay()
    world.validate(technical={("A", START["A"]["u"])})
    state = world.state()
    step = state["step"]
    assert (state["stage"], state["outcome"]) == ("validation", None)
    assert (step["kind"], step["technical"], step["reason"]) == (
        "blocked", True, "unresolved_after_cycle")  # fmt: skip
    assert "further cycle" in step["why"] and "recorded decision" in step["why"]
    world.refused("campaigns", "validation", "--runs-root", world.root, "--candidate", 1,
                  "--theta0", world.runs.theta0, "--out", world.out / "no-more.yaml")  # fmt: skip


def test_n_169_pauses_at_the_reserve_and_reads_as_passed_once_the_seeds_are_extended(
    replay, monkeypatch
) -> None:
    """Its cap 201 lies beyond the 200 reserved seeds: a pause for a decision, no design stop, N
    not cut. The same file, under seeds an amendment extends, passes, and the route goes on."""
    world = replay()
    world.validated()
    world.n_record(169, state="failed")
    path = decision_path(world.root, "n-c1")
    kept = path.read_bytes()
    state = world.state()
    step = state["step"]
    assert (state["stage"], state["outcome"]) == ("n", None)
    assert (step["kind"], step["technical"], step["reason"]) == (
        "blocked", False, "cap_beyond_reserved_seeds")  # fmt: skip
    assert "own amendment" in step["why"] and "201" in step["reasons"][0]
    world.refused("campaigns", "confirmation", "--runs-root", world.root, "--candidate", 1,
                  "--out", world.out / "no-block")  # fmt: skip
    monkeypatch.setitem(SEEDS["confirmation"], 1, range(10400, 10700))
    record, problems = read_decision(world.root, "n-c1", repo=world.repo)
    assert problems == [] and (record["state"], record["N"], record["cap"]) == ("passed", 169, 201)
    assert path.read_bytes() == kept
    assert world.step()["record"] == "budget-c1"  # N above 90: the next step
    world.record("budget-c1", world.records.budget(1, 169, code=world.code))
    state = world.state()
    assert state["problems"] == [] and state["step"]["campaign"] == "confirmation"
    assert state["confirmation"]["cap"] == 201
    assert {cell[2] for cell in state["prescribed"]["c1/confirmation"]} == set(range(10400, 10601))

