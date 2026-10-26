"""Study 3's state machine (analysis/study3_progress.py): where the study is and what comes next.

The routes are tested on the pure core (``advance``) over records built by hand: endpoints,
qualification readings, decision entries and holds, as the readers give them; no file. The reader
(``progress``) is tested on synthetic RUNS_DIR trees built with the reader tests' helpers: the
launch journal, run directories whose regular runs hold a harmful-response file and an
observations file with a sentinel text (never opened), and decision records with their evidence.
The registered K* of candidate 1 is 6.00; runs at K_l and K_u bracket it with the weight 1/2."""

import json
import math
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_study3_decisions import World as DecisionWorld
from test_study3_decisions import held as decision_hold
from test_study3_records import SENTINEL, seal
from test_study3_records import World as RunWorld

from zero_field_refusal.analysis import study3_progress
from zero_field_refusal.analysis.study3_decisions import cap_of
from zero_field_refusal.analysis.study3_progress import (
    Records,
    advance,
    confirmation_state,
    progress,
)
from zero_field_refusal.analysis.study3_records import SEEDS, Cell, Endpoint, step_key
from zero_field_refusal.analysis.study3_search import CANDIDATES, adjusted_rate

ARMS = ("A", "F", "P")
LABELS = ("l", "u")
CODE = {"commit": "a" * 40, "gpu_code_sha256": "b" * 64}
EXPECTED = CODE["gpu_code_sha256"]
OPT = {"A": "adamw", "F": "adam_frozen", "P": "adamw_prewarmed"}
ROLE = {"engineering": "engineering", "search": "search", "numerical": "numerical",
        "repair": "numerical", "validation": "validation",
        "confirmation": "confirmation"}  # fmt: skip
K_ONLY = ("engineering", "search", "numerical", "repair")
WHEN = "2026-10-25T12:00:00+00:00"
START = CANDIDATES[1]["start"]


def ep(arm, rate, seed, *, step, candidate=1, round_=None, state="usable", K=None, mmlu=1.0,
       factual=2.0, attempt=1, run_id=None, campaign="c") -> Endpoint:  # fmt: skip
    """One attempt of the cell (step, round, arm, rate, seed): K-only for the engineering, search,
    numerical and repair steps, regular for validation and confirmation; the endpoint's numbers
    where it has one."""
    round_ = 0 if round_ is None and step == "search" else round_
    label = f"{arm}-{rate:g}/s{seed}"
    cell = Cell(campaign=campaign, label=label, candidate=candidate, step=step, round=round_,
                arm=arm, rate=rate, seed=seed, config_hash="0" * 16,
                prescribed=(arm, rate, seed))  # fmt: skip
    run_id = run_id or f"{cell.key}-{label}-a{attempt}".replace("/", "_")
    has = state in ("usable", "unusable")
    regular = step not in K_ONLY
    return Endpoint(run_id=run_id, run_dir=Path(run_id), cell=cell, arm=arm, lr=rate, seed=seed,
                    role=ROLE[step], candidate=candidate, k_only=not regular, attempt=attempt,
                    label=label, state=state, steps=25 if has else None,
                    K=K if has else None, K_harmful=5.0 if has and regular else None,
                    mmlu_loss_pp=mmlu if has else None,
                    factual_loss_pp=factual if has else None)  # fmt: skip


def entry(kind, state="missing", record=None, problems=()) -> dict:
    return {"kind": kind, "path": f"/runs/study3/decisions/{kind}.json", "state": state,
            "record": record, "problems": list(problems)}  # fmt: skip


def reading() -> dict:
    """A qualification reading of no records."""
    return {"rates": {}, "states": {}, "cases": [], "voided": {}, "supersessions": [],
            "held": [], "problems": []}  # fmt: skip


def rates_of(candidate=1, arms=ARMS, rates=None) -> dict:
    """arm -> label -> rate: the starting rates, or ``rates`` where it names them."""
    out = {arm: dict(CANDIDATES[candidate]["start"][arm]) for arm in arms}
    for arm, given in (rates or {}).items():
        out[arm] |= given
    return out


class Study:
    """The records of a synthetic study, as the readers give them, and the core over them."""

    def __init__(self) -> None:
        self.endpoints: list[Endpoint] = []
        self.problems: list[str] = []
        self.holds: list[dict] = []
        self.hold_problems: list[str] = []
        self.entries: dict[str, dict] = {}
        self.readings = {1: reading(), 2: reading()}
        self.starts: dict[str, datetime] = {}

    # -- decision records

    def record(self, kind, state="passed", **fields) -> dict:
        record = {"kind": kind, "state": state, "written_at": WHEN, "code": dict(CODE), **fields}
        self.entries[kind] = entry(kind, state, record)
        return record

    def simulation(self, blocks=28, *, state="passed", power_short=False, n_min=None) -> None:
        power = {"value": 0.7 if power_short else 0.86, "requirement": 0.80,
                 "passed": not power_short, "power_short": power_short,
                 "n_min": n_min if power_short else None}  # fmt: skip
        self.record("simulation", state, level=0.975, validation_blocks=blocks,
                    gates={"power_af": power})  # fmt: skip

    def ready(self, blocks=28, **simulation) -> None:
        """Engineering and the simulation passed."""
        self.record("engineering")
        self.simulation(blocks, **simulation)

    def invalid(self, kind, problem="a problem") -> None:
        self.entries[kind] = entry(kind, "invalid", None, [f"/runs/{kind}.json: {problem}"])

    def n(self, candidate=1, N=60, *, state=None, arms=ARMS, rates=None, seeds=None,
          run_ids=None) -> dict:  # fmt: skip
        """The n record: by default the rates validated, the comparisons of ``arms``, and the
        validation blocks where A and F are supported with their A and F runs."""
        cap = cap_of(N)
        state = state or ("passed" if cap <= 200 else "failed")
        final = rates_of(candidate, arms, rates)
        key = step_key(candidate, "validation")
        if seeds is None:
            seeds = sorted({e.seed for e in self.endpoints if e.cell.key == key})
        if run_ids is None:
            run_ids = [e.run_id for e in self.endpoints if e.cell.key == key and e.seed in seeds
                       and e.arm in ("A", "F") and e.state in ("usable", "unusable")]  # fmt: skip
        return self.record(
            f"n-c{candidate}", state, candidate=candidate,
            rates={arm: {"lr_l": final[arm]["l"], "lr_u": final[arm]["u"]} for arm in arms},
            active=["A-F", "P-F"] if "P" in arms else ["A-F"], level=0.975,
            cohort={"seeds": list(seeds), "run_ids": run_ids}, spread_pp=4.5, N=N, cap=cap,
        )  # fmt: skip

    def budget(self, candidate=1, decision="proceed") -> None:
        n = self.entries[f"n-c{candidate}"]["record"]
        self.record(f"budget-c{candidate}", "passed" if decision == "proceed" else "failed",
                    candidate=candidate, N=n["N"], cap=n["cap"], decision=decision)  # fmt: skip

    # -- runs and the qualification

    def add(self, *endpoints: Endpoint) -> None:
        self.endpoints += endpoints

    def drop(self, run_id: str) -> Endpoint:
        (found,) = [e for e in self.endpoints if e.run_id == run_id]
        self.endpoints.remove(found)
        return found

    def change(self, run_id: str, **fields) -> Endpoint:
        """The attempt with other fields (its state, its K)."""
        found = self.drop(run_id)
        self.add(Endpoint(**{**found.__dict__, **fields}))
        return found

    def search(self, candidate=1, arms=ARMS, offsets=None, special=None) -> None:
        """Round 0: each arm at its starting rates on the search seeds, K at the target times
        e^offset (``offsets``: (arm, label) -> a number or one per seed); ``special``: (arm,
        label, seed) -> ``ep`` keywords."""
        spec = CANDIDATES[candidate]
        seeds = SEEDS["search"][candidate]
        for arm in arms:
            for label in LABELS:
                given = (offsets or {}).get((arm, label), 0.0)
                given = [given] * 4 if isinstance(given, int | float) else given
                for seed, offset in zip(seeds, given, strict=True):
                    kw = {"K": spec["targets"][label] * math.exp(offset),
                          **(special or {}).get((arm, label, seed), {})}  # fmt: skip
                    self.add(ep(arm, spec["start"][arm][label], seed, step="search",
                                candidate=candidate, **kw))  # fmt: skip

    def numerical(self, candidate=1, arms=ARMS, rates=None, *, states=None, step="numerical",
                  offset=0.0, special=None, reading_too=True) -> None:  # fmt: skip
        """The launcher runs of the numerical qualification (or a repair's: ``step``) on the
        numerical seeds, K at the target times e^offset, and the rates' qualification states
        (``states``: arm or (arm, label) -> state; qualified by default)."""
        spec = CANDIDATES[candidate]
        final = rates_of(candidate, arms, rates)
        for arm in arms:
            for label in LABELS:
                for seed in SEEDS["numerical"][candidate]:
                    kw = {"K": spec["targets"][label] * math.exp(offset),
                          **(special or {}).get((arm, label, seed), {})}  # fmt: skip
                    self.add(ep(arm, final[arm][label], seed, step=step, candidate=candidate,
                                **kw))  # fmt: skip
                if reading_too:
                    state = (states or {}).get((arm, label), (states or {}).get(arm, "qualified"))
                    self.rate(candidate, arm, final[arm][label], state)

    def rate(self, candidate, arm, lr, state="qualified", *, control_missing=False,
             technical=False) -> None:  # fmt: skip
        """A rate of the qualification reading with its state; an FA rate's control accepted
        (or without a record, ``control_missing``); a technical case when ``technical``."""
        key = f"{OPT[arm]}-{lr:g}"
        control = None
        if arm == "F":
            control = ({"state": "not_qualified", "kind": "technical", "record_sha256": None}
                       if control_missing else
                       {"state": "accepted", "kind": None, "record_sha256": "c" * 64})  # fmt: skip
        verdict = {"state": state, "control": control, "not_counted": [], "reasons": []}
        reading_ = self.readings[candidate]
        reading_["rates"][key] = {"key": key, "arm": arm, "optimizer": OPT[arm], "lr": lr,
                                  "state": state, "verdict": verdict, "runs": [],
                                  "launchers": {}, "held": []}  # fmt: skip
        reading_["states"][key] = state
        if technical:
            seed = SEEDS["numerical"][candidate][0]
            reading_["cases"].append({
                "case": {"name": f"s{seed}/{key}", "optimizer": OPT[arm], "lr": lr, "seed": seed,
                         "candidate": candidate},
                "state": "not_qualified", "kind": "technical",
                "record_sha256": "d" * 64})  # fmt: skip

    def cycle(self, candidate=1) -> None:
        """The candidate's engineering cycle is recorded: its supersession record exists."""
        self.readings[candidate]["supersessions"].append(
            {"path": f"/runs/qualification/c{candidate}/supersede-1.json", "sha256": "9" * 64}
        )

    def blocks(self, step="validation", candidate=1, arms=ARMS, seeds=None, rates=None,
               special=None) -> None:  # fmt: skip
        """Blocks of regular runs, each arm at its targets (supported, weight 1/2); ``special``:
        (seed, arm, label) -> ``ep`` keywords."""
        spec = CANDIDATES[candidate]
        final = rates_of(candidate, arms, rates)
        for seed in seeds:
            for arm in arms:
                for label in LABELS:
                    kw = {"K": spec["targets"][label],
                          **(special or {}).get((seed, arm, label), {})}  # fmt: skip
                    self.add(ep(arm, final[arm][label], seed, step=step, candidate=candidate,
                                **kw))  # fmt: skip

    def validation(self, candidate=1, arms=ARMS, count=28, **kw) -> None:
        self.blocks("validation", candidate, arms, SEEDS["validation"][candidate][:count], **kw)

    def confirmation(self, candidate=1, arms=ARMS, count=60, first=0, **kw) -> None:
        seeds = SEEDS["confirmation"][candidate][first : first + count]
        self.blocks("confirmation", candidate, arms, seeds, **kw)

    def through_validation(self, candidate=1, arms=ARMS) -> None:
        """Candidate 1 kept at every target and qualified, its validation passed."""
        self.ready()
        self.search(candidate, arms)
        self.numerical(candidate, arms)
        self.validation(candidate, arms)

    # -- holds

    def hold(self, run_id, *, line=3, cycle=True, candidate=1,
             reason="no recorded commit (git: {})") -> str:  # fmt: skip
        """Hold an attempt as package 1 does: its endpoint ``held``, ``read_holds``' entry with
        the journal line and the cycle's supersession record."""
        found = self.drop(run_id)
        self.add(Endpoint(**{**found.__dict__, "state": "held", "K": None, "steps": None,
                             "reasons": ("held by the cycle",)}))  # fmt: skip
        records = [{"kind": "journal", "path": "launch/study3-c1-numerical.jsonl", "root": "runs",
                    "line": line, "line_sha256": "e" * 64, "sha256": "f" * 64}]  # fmt: skip
        if cycle:
            records.append(
                {
                    "kind": "supersession",
                    "root": "runs",
                    "sha256": "9" * 64,
                    "path": f"qualification/c{candidate}/supersede-1.json",
                }
            )
        self.holds.append(
            {
                "run_id": run_id,
                "run": found.label,
                "experiment": "study3",
                "config_hash": "c" * 12,
                "reason": reason,
                "records": records,
            }
        )
        return run_id

    def closure(self, run_id, *, state="failed", **over) -> dict:
        """The closure record that cites a hold."""
        (hold,) = [h for h in self.holds if h["run_id"] == run_id]
        (found,) = [e for e in self.endpoints if e.run_id == run_id]
        journal, cycle = hold["records"][0], hold["records"][1]
        record = {
            "candidate": found.candidate, "arm": found.arm, "rate": found.lr, "seed": found.seed,
            "run_id": run_id,
            "hold": {"path": journal["path"], "line": journal["line"],
                     "line_sha256": journal["line_sha256"]},
            "cycle": {"path": cycle["path"], "sha256": cycle["sha256"]},
            "reason": "no_recorded_commit",
            "decision": "close_pf" if found.arm == "P" else "technical_feasibility_report",
            **over,
        }  # fmt: skip
        return self.record(f"closure-c{found.candidate}-{found.arm}", state, **record)

    # -- the core

    def records(self) -> Records:
        decisions = {c: {k: v for k, v in self.entries.items()
                         if k in ("engineering", "simulation", f"n-c{c}", f"budget-c{c}")}
                     for c in (1, 2)}  # fmt: skip
        closures = {k: v for k, v in self.entries.items() if k.startswith("closure-")}
        return Records(endpoints=tuple(self.endpoints), problems=tuple(self.problems),
                       holds=tuple(self.holds), hold_problems=tuple(self.hold_problems),
                       decisions=decisions, closures=closures, qualification=self.readings,
                       starts=self.starts)  # fmt: skip

    def advance(self, **kw) -> dict:
        return advance(self.records(), expected_code=EXPECTED, **kw)


def cells(step) -> set:
    return {tuple(cell) for cell in step["cells"]}


def run_of(study, step, arm, label, seed, candidate=1, rates=None) -> str:
    rate = rates_of(candidate, ARMS, rates)[arm][label]
    (found,) = [e for e in study.endpoints if e.cell.key == step_key(candidate, step, e.cell.round)
                and e.cell.step == step and e.arm == arm and e.seed == seed
                and math.isclose(e.lr, rate)]  # fmt: skip
    return found.run_id


def tidy(result) -> str:
    """The result as the summary prints it: JSON."""
    return json.dumps(result, default=str)


# ------------------------------------------------------------------ the main route


def test_candidate_1_passes_through_every_stage_to_confirmation_and_done() -> None:
    study = Study()
    study.ready()
    result = study.advance()
    assert (result["stage"], result["candidate"], result["outcome"]) == ("search", 1, None)
    step = result["step"]
    assert (step["kind"], step["campaign"], step["step"], step["round"]) == (
        "campaign", "search-round0", "search", 0)  # fmt: skip
    assert cells(step) == {(arm, START[arm][label], seed) for arm in ARMS for label in LABELS
                           for seed in SEEDS["search"][1]}  # fmt: skip

    study.search()
    result = study.advance()
    assert (result["stage"], result["step"]["campaign"]) == ("numerical", "numerical")
    assert len(result["step"]["cells"]) == 18 and result["step"]["round"] is None

    study.numerical(reading_too=False)
    result = study.advance()  # the launcher runs ended; their cases are still to be qualified
    assert (result["stage"], result["step"]["kind"]) == ("numerical", "pending")

    for arm in ARMS:
        for label in LABELS:
            study.rate(1, arm, START[arm][label])
    result = study.advance()
    assert (result["stage"], result["step"]["campaign"]) == ("validation", "validation")
    assert cells(result["step"]) == {(arm, START[arm][label], seed) for arm in ARMS
                                     for label in LABELS
                                     for seed in SEEDS["validation"][1][:28]}  # fmt: skip

    study.validation()
    result = study.advance()
    assert (result["stage"], result["step"]["kind"], result["step"]["record"]) == (
        "choice", "decision", "n-c1")  # fmt: skip
    fields = result["step"]["fields"]
    assert fields["candidate"] == 1 and fields["active"] == ["A-F", "P-F"]
    assert fields["cohort"]["seeds"] == list(SEEDS["validation"][1][:28])
    assert len(fields["cohort"]["run_ids"]) == 28 * 4
    assert result["validation"]["gates"]["AF"]["supported"] == 28
    assert any(e["event"] == "chosen" for e in result["route"])

    study.n(N=60)
    result = study.advance()
    assert (result["stage"], result["step"]["campaign"]) == ("confirmation", "confirmation")
    assert cells(result["step"]) == {
        (arm, START[arm][label], 10400) for arm in ARMS for label in LABELS
    }  # one block per campaign, in order
    assert result["confirmation"]["may_start_next"] is True

    study.confirmation(count=30)
    result = study.advance()
    assert {tuple(c)[2] for c in result["step"]["cells"]} == {10430}

    study.confirmation(count=30, first=30)
    result = study.advance()
    assert (result["stage"], result["outcome"], result["step"]) == (
        "done", "confirmed", {"kind": "done", "outcome": "confirmed"})  # fmt: skip
    assert result["readings"] == {"A-F": "confirmatory", "P-F": "confirmatory"}
    assert result["problems"] == [] and result["active"] == {
        "arms": ["A", "F", "P"], "comparisons": ["A-F", "P-F"]}  # fmt: skip
    keys = set(result["prescribed"])
    assert keys == {"c1/engineering", "c1/search/r0", "c1/numerical", "c1/validation",
                    "c1/confirmation"}  # fmt: skip
    assert len(result["prescribed"]["c1/confirmation"]) == 6 * cap_of(60)
    json.loads(tidy(result))  # numbers, ids, paths and hashes: it prints as JSON


def test_the_search_waits_for_its_runs_and_issues_round_1_for_a_miss() -> None:
    study = Study()
    study.ready()
    study.search(offsets={("F", "u"): 0.2})
    pending = study.drop(run_of(study, "search", "A", "l", 10101))
    study.add(Endpoint(**{**pending.__dict__, "state": "pending", "K": None}))
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("search", "pending")
    assert pending.run_id in result["step"]["waiting"]
    study.drop(pending.run_id)
    study.add(pending)
    result = study.advance()
    rate = adjusted_rate(START["F"]["u"], math.log(7.5) + 0.2, 7.5, 1.30)
    assert (result["step"]["campaign"], result["step"]["round"]) == ("search-round1", 1)
    assert cells(result["step"]) == {("F", rate, seed) for seed in SEEDS["search"][1]}
    assert result["prescribed"]["c1/search/r1"] == [("F", rate, s) for s in SEEDS["search"][1]]
    for seed in SEEDS["search"][1]:
        study.add(ep("F", rate, seed, step="search", round_=1, K=7.5))
    result = study.advance()
    assert result["stage"] == "numerical"
    assert ("F", rate, 10200) in cells(result["step"])
    assert result["rates"]["F"] == {"l": START["F"]["l"], "u": rate}


# ------------------------------------------------------------------ the candidate routes


def fail_at(study: Study, stage: str, arm: str = "F", candidate: int = 1) -> None:
    """Candidate ``candidate`` reaches ``stage``, where ``arm`` fails scientifically."""
    seed = SEEDS["search"][candidate][0]
    special = {(arm, "l", seed): {"state": "scientific"}} if stage == "search" else None
    study.search(candidate, special=special)
    if stage == "search":
        return
    study.numerical(candidate, states={arm: "failing"} if stage == "numerical" else None)
    if stage == "numerical":
        return
    seeds = SEEDS["validation"][candidate][:28]
    # the arm's upper run below K* in 7 of 28 blocks (no bracket): 21 supported, below 22
    below = CANDIDATES[candidate]["targets"]["star"] * 0.9
    study.validation(candidate, special={(s, arm, "u"): {"K": below} for s in seeds[:7]})


@pytest.mark.parametrize("stage, arms", [("search", ARMS), ("numerical", ARMS),
                                         ("validation", ("A", "F"))])  # fmt: skip
def test_f_failing_scientifically_at_candidate_1_moves_to_candidate_2(stage, arms) -> None:
    """P carries on to candidate 2, unless its own gate failed: F's support is P–F's too, so F
    failing at validation fails the P–F gate as well, which closes P ("Failure routes")."""
    study = Study()
    study.ready()
    fail_at(study, stage, "F")
    result = study.advance()
    assert (result["stage"], result["candidate"], result["outcome"]) == ("search", 2, None)
    assert result["step"]["campaign"] == "search-round0"
    assert cells(result["step"]) == {(arm, CANDIDATES[2]["start"][arm][label], seed)
                                     for arm in arms for label in LABELS
                                     for seed in SEEDS["search"][2]}  # fmt: skip
    (move,) = [e for e in result["route"] if e["event"] == "candidate"]
    assert (move["candidate"], move["stage"], move["arms"]) == (1, stage, ["F"])
    assert result["active"]["arms"] == list(arms)
    assert "c2/search/r0" in result["prescribed"]


@pytest.mark.parametrize("stage", ["search", "numerical", "validation"])
def test_candidate_2_failing_ends_with_the_scientific_feasibility_report(stage) -> None:
    study = Study()
    study.ready()
    fail_at(study, "search", "F", 1)
    fail_at(study, stage, "A", 2)
    result = study.advance()
    assert (result["stage"], result["outcome"], result["candidate"]) == (
        "scientific_feasibility_report", "scientific_feasibility_report", 2)  # fmt: skip
    assert result["step"] == {"kind": "report", "outcome": "scientific_feasibility_report"}
    assert [e["candidate"] for e in result["route"] if e["event"] == "candidate"] == [1, 2]


def test_a_failed_validation_endpoint_removes_support_and_moves_nothing_before_the_aggregate():
    study = Study()
    study.through_validation()
    seeds = SEEDS["validation"][1]
    last = [e for e in study.endpoints if e.seed == seeds[27]]
    for e in last:
        study.drop(e.run_id)
    failed = study.drop(run_of(study, "validation", "F", "u", seeds[0]))
    study.add(Endpoint(**{**failed.__dict__, "state": "scientific", "K": None}))
    result = study.advance()  # a failed endpoint, and one block not issued: no aggregate yet
    assert (result["stage"], result["candidate"]) == ("validation", 1)
    assert result["step"]["campaign"] == "validation"
    assert {c[2] for c in result["step"]["cells"]} == {seeds[27]}
    study.add(*last)
    result = study.advance()  # 27 of 28 supported: the gate passes
    assert (result["stage"], result["candidate"]) == ("choice", 1)
    assert result["validation"]["gates"]["AF"]["supported"] == 27
    assert seeds[0] not in result["step"]["fields"]["cohort"]["seeds"]


def test_a_technical_validation_endpoint_blocks_the_aggregate() -> None:
    study = Study()
    study.through_validation()
    broken = study.change(run_of(study, "validation", "A", "l", 10305), state="technical", K=None)
    result = study.advance()
    assert (result["stage"], result["candidate"], result["outcome"]) == ("validation", 1, None)
    step = result["step"]
    assert (step["kind"], step["technical"]) == ("blocked", True)
    assert any(broken.run_id in r for r in step["reasons"])
    assert "validation" not in result or result["validation"].get("gates") is None


def test_a_never_issued_cell_leaves_its_block_unresolved() -> None:
    study = Study()
    study.through_validation()
    study.drop(run_of(study, "validation", "P", "u", 10310))
    result = study.advance()
    assert (result["stage"], result["step"]["campaign"]) == ("validation", "validation")
    assert cells(result["step"]) == {("P", START["P"]["u"], 10310)}


def test_validation_has_40_blocks_when_the_simulation_record_says_so() -> None:
    study = Study()
    study.ready(blocks=40)
    study.search()
    study.numerical()
    study.validation(count=28)
    result = study.advance()
    assert (result["stage"], result["step"]["campaign"]) == ("validation", "validation")
    assert {c[2] for c in result["step"]["cells"]} == set(SEEDS["validation"][1][28:40])
    assert len(result["prescribed"]["c1/validation"]) == 40 * 6
    seeds = SEEDS["validation"][1][:40]
    study.blocks(seeds=seeds[28:40])
    for seed in seeds[:9]:  # A unsupported in 9 of 40 blocks: 31 supported, the gate's minimum
        study.change(run_of(study, "validation", "A", "u", seed), K=5.0)
    result = study.advance()
    assert result["validation"]["gates"]["AF"] == {
        "supported": 31,
        "minimum": 31,
        "utility": True,
        "passes": True,
    }
    assert (result["stage"], result["candidate"]) == ("choice", 1)


def test_a_40_block_validation_with_30_supported_moves_to_candidate_2() -> None:
    study = Study()
    study.ready(blocks=40)
    study.search()
    study.numerical()
    seeds = SEEDS["validation"][1][:40]
    study.validation(count=40, special={(s, "A", "u"): {"K": 5.0} for s in seeds[:10]})
    result = study.advance()
    assert (result["stage"], result["candidate"]) == ("search", 2)


def test_p_failing_at_validation_closes_p_and_a_f_goes_on() -> None:
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    seeds = SEEDS["validation"][1][:28]
    study.validation(special={(s, "P", "u"): {"K": 5.0} for s in seeds[:7]})
    result = study.advance()
    assert (result["stage"], result["candidate"]) == ("choice", 1)
    assert result["active"] == {"arms": ["A", "F"], "comparisons": ["A-F"]}
    assert result["step"]["fields"]["active"] == ["A-F"]
    assert set(result["step"]["fields"]["rates"]) == {"A", "F"}
    (close,) = [e for e in result["route"] if e["event"] == "close_pf"]
    assert (close["candidate"], close["stage"]) == (1, "validation")
    study.n(N=60, arms=("A", "F"))
    result = study.advance()
    assert cells(result["step"]) == {(arm, START[arm][label], 10400) for arm in ("A", "F")
                                     for label in LABELS}  # four runs per block  # fmt: skip


def test_p_failing_at_the_search_closes_p_for_the_rest_of_the_study() -> None:
    study = Study()
    study.ready()
    study.search(special={("P", "u", 10102): {"state": "scientific"}})
    result = study.advance()
    assert (result["stage"], result["active"]["arms"]) == ("numerical", ["A", "F"])
    assert len(result["step"]["cells"]) == 12
    # A then fails at candidate 1: candidate 2 issues no P cell
    study.numerical(arms=("A", "F"), states={"A": "failing"})
    result = study.advance()
    assert (result["stage"], result["candidate"]) == ("search", 2)
    assert {c[0] for c in result["step"]["cells"]} == {"A", "F"}
    assert result["active"]["comparisons"] == ["A-F"]


# ------------------------------------------------------------------ technical routes


def test_an_engineering_cycle_in_progress_keeps_the_numerical_stage_technical() -> None:
    study = Study()
    study.ready()
    study.search()
    study.numerical(states={("F", "u"): "not_qualified"})
    study.readings[1]["rates"][f"adam_frozen-{START['F']['u']:g}"]["verdict"]["not_counted"] = [
        {"case": "s10200/adam_frozen-1.83e-05", "record_sha256": "d" * 64, "why": "x"}
    ]
    result = study.advance()
    assert (result["stage"], result["candidate"], result["outcome"]) == ("numerical", 1, None)
    step = result["step"]
    assert (step["kind"], step["technical"]) == ("blocked", True)
    assert any(f"adam_frozen-{START['F']['u']:g}" in r for r in step["reasons"])


def test_an_fa_rate_whose_control_has_no_record_yet_waits() -> None:
    study = Study()
    study.ready()
    study.search()
    study.numerical(reading_too=False)
    for arm in ARMS:
        for label in LABELS:
            study.rate(1, arm, START[arm][label],
                       "not_qualified" if arm == "F" else "qualified",
                       control_missing=arm == "F")  # fmt: skip
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("numerical", "pending")
    study.rate(1, "F", START["F"]["l"], "not_qualified", control_missing=True, technical=True)
    assert study.advance()["step"]["kind"] == "blocked"


@pytest.mark.parametrize("arm", ["A", "F"])
def test_an_unresolved_technical_failure_in_a_or_f_ends_with_the_technical_report(arm) -> None:
    study = Study()
    study.ready()
    study.search()
    study.numerical(states={(arm, "l"): "not_qualified"})
    study.rate(1, arm, START[arm]["l"], "not_qualified", technical=True)
    study.cycle()
    result = study.advance()
    assert (result["stage"], result["outcome"], result["candidate"]) == (
        "technical_feasibility_report", "technical_feasibility_report", 1)  # fmt: skip
    assert result["step"] == {"kind": "report", "outcome": "technical_feasibility_report"}
    assert not any(key.startswith("c2/") for key in result["prescribed"])  # no candidate 2


def test_an_unresolved_technical_failure_in_p_closes_p() -> None:
    study = Study()
    study.ready()
    study.search()
    study.numerical(states={("P", "u"): "not_qualified"})
    study.rate(1, "P", START["P"]["u"], "not_qualified", technical=True)
    study.cycle()
    result = study.advance()
    assert (result["stage"], result["active"]["arms"]) == ("validation", ["A", "F"])
    assert len(result["step"]["cells"]) == 28 * 4


def test_a_technical_arm_holds_the_route_before_a_scientific_failure_moves_it() -> None:
    study = Study()
    study.ready()
    study.search(special={("A", "l", 10100): {"state": "scientific"},
                          ("F", "u", 10103): {"state": "technical"}})  # fmt: skip
    result = study.advance()
    assert (result["stage"], result["candidate"], result["step"]["kind"]) == (
        "search", 1, "blocked")  # fmt: skip
    assert result["step"]["technical"] is True


def test_a_pending_run_holds_the_route_before_a_scientific_failure_moves_it() -> None:
    study = Study()
    study.ready()
    study.search(special={("A", "l", 10100): {"state": "scientific"},
                          ("P", "u", 10103): {"state": "pending"}})  # fmt: skip
    result = study.advance()
    assert (result["stage"], result["candidate"], result["step"]["kind"]) == (
        "search", 1, "pending")  # fmt: skip


def test_a_technical_p_holds_the_stage_until_its_diagnosis() -> None:
    """A–F waits too: every cell of a block is issued with the block, P's included."""
    study = Study()
    study.ready()
    study.search(special={("P", "l", 10101): {"state": "technical"}})
    result = study.advance()
    assert (result["stage"], result["step"]["kind"], result["step"]["technical"]) == (
        "search", "blocked", True)  # fmt: skip
    assert result["active"]["arms"] == ["A", "F", "P"]


def test_a_launcher_run_that_ended_and_is_not_located_is_loud() -> None:
    """Under this checkout's configs the cell's config hash is another (a hashed key changed):
    the qualification locates nothing, and that is a problem, never a wait."""
    study = Study()
    study.ready()
    study.search()
    study.numerical(states={("A", "l"): "incomplete"})
    ran = run_of(study, "numerical", "A", "l", 10200)
    key = f"adamw-{START['A']['l']:g}"
    study.readings[1]["rates"][key]["runs"] = [
        {"label": f"A-{START['A']['l']:g}/s10200", "run_ids": [ran], "located": None}
    ]
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("numerical", "blocked")
    assert any(ran in p and "locate" in p for p in result["problems"])
    # a run that has not ended is not located yet: it waits
    study.change(ran, state="pending", K=None)
    assert study.advance()["step"]["kind"] == "pending"


def test_a_superseded_launcher_run_waits_for_its_next_attempt() -> None:
    study = Study()
    study.ready()
    study.search()
    study.numerical(states={("A", "l"): "incomplete"})
    study.change(run_of(study, "numerical", "A", "l", 10200), state="superseded", K=None)
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("numerical", "pending")
    assert any("s10200: its next attempt" in w for w in result["step"]["waiting"])


# ------------------------------------------------------------------ the hold route


def held_numerical(arm: str) -> tuple[Study, str]:
    study = Study()
    study.ready()
    study.search()
    study.numerical(states={(arm, "l"): "not_qualified"})
    study.cycle()
    run_id = study.hold(run_of(study, "numerical", arm, "l", 10201))
    return study, run_id


def test_a_held_f_names_its_closure_record_then_ends_with_the_technical_report() -> None:
    study, run_id = held_numerical("F")
    result = study.advance()
    assert (result["stage"], result["outcome"], result["candidate"]) == (
        "technical_feasibility_report",
        None,
        1,
    )  # a terminal outcome, never a pending step
    step = result["step"]
    assert (step["kind"], step["record"], step["path"]) == (
        "decision", "closure-c1-F", "study3/decisions/closure-c1-F.json")  # fmt: skip
    assert step["fields"] == {
        "candidate": 1, "arm": "F", "rate": START["F"]["l"], "seed": 10201, "run_id": run_id,
        "hold": {"path": "launch/study3-c1-numerical.jsonl", "line": 3, "line_sha256": "e" * 64},
        "cycle": {"path": "qualification/c1/supersede-1.json", "sha256": "9" * 64},
        "reason": "no_recorded_commit", "reasons": ["no_recorded_commit"],
        "decision": "technical_feasibility_report"}  # fmt: skip
    study.closure(run_id)
    result = study.advance()
    assert (result["stage"], result["outcome"], result["step"]) == (
        "technical_feasibility_report", "technical_feasibility_report",
        {"kind": "report", "outcome": "technical_feasibility_report"})  # fmt: skip
    assert result["closures"]["closure-c1-F"]["state"] == "failed"
    assert not any(key.startswith("c2/") for key in result["prescribed"])


def test_a_held_p_closes_p_f_once_its_closure_is_written_and_a_f_goes_on() -> None:
    study, run_id = held_numerical("P")
    result = study.advance()
    assert (result["stage"], result["outcome"]) == ("numerical", None)
    assert (result["step"]["record"], result["step"]["fields"]["decision"]) == (
        "closure-c1-P", "close_pf")  # fmt: skip
    study.closure(run_id)
    result = study.advance()
    assert (result["stage"], result["active"]) == (
        "validation", {"arms": ["A", "F"], "comparisons": ["A-F"]})  # fmt: skip
    assert len(result["step"]["cells"]) == 28 * 4
    (close,) = [e for e in result["route"] if e["event"] == "close_pf"]
    assert (close["stage"], close["why"]) == ("numerical", "closure-c1-P")
    # P's numerical cells stay prescribed: its runs were issued
    assert ("P", START["P"]["l"], 10201) in result["prescribed"]["c1/numerical"]


def test_a_hold_of_a_label_alone_is_placed_on_its_cell() -> None:
    """A hold line may name its label under a run id that no start names (the launcher holds the
    label under any config hash): it holds that cell."""
    study = Study()
    study.ready()
    study.search()
    study.numerical(states={("P", "u"): "not_qualified"})
    found = next(e for e in study.endpoints if e.cell.step == "numerical" and e.arm == "P"
                 and e.seed == 10202 and e.lr == START["P"]["u"])  # fmt: skip
    study.holds.append({"run_id": "20261026T999999Z-unknown-a9", "run": found.label,
                        "experiment": "study3", "config_hash": "0" * 12,
                        "reason": "no hash of v₀", "records": [
                            {"kind": "journal", "path": "launch/x.jsonl", "line": 1,
                             "line_sha256": "e" * 64, "sha256": "f" * 64},
                            {"kind": "supersession", "path": "qualification/c1/supersede-1.json",
                             "sha256": "9" * 64}]})  # fmt: skip
    result = study.advance()
    assert (result["step"]["record"], result["step"]["fields"]["seed"]) == ("closure-c1-P", 10202)
    assert result["step"]["fields"]["reasons"] == ["no_v0_hash"]
    # a study-3 hold that no start places is a problem
    study.holds[-1]["run"] = "Q-1e-05/s1"
    result = study.advance()
    assert result["step"]["kind"] == "blocked"
    assert any("20261026T999999Z-unknown-a9" in p for p in result["problems"])


def test_a_held_p_in_validation_closes_p_f_and_its_blocks_keep_four_runs() -> None:
    study = Study()
    study.through_validation()
    run_id = study.hold(run_of(study, "validation", "P", "l", 10302))
    result = study.advance()
    assert (result["stage"], result["step"]["record"]) == ("validation", "closure-c1-P")
    study.closure(run_id)
    result = study.advance()
    assert (result["stage"], result["active"]["comparisons"]) == ("choice", ["A-F"])
    assert set(result["validation"]["gates"]) == {"AF"}


def test_a_hold_that_no_cycle_record_names_cannot_be_closed_and_blocks() -> None:
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    study.hold(run_of(study, "numerical", "F", "l", 10201), cycle=False)
    result = study.advance()
    assert result["step"]["kind"] == "blocked"
    assert any("supersession" in r for r in result["step"]["reasons"])


def test_a_closure_that_does_not_match_its_hold_is_a_problem() -> None:
    study, run_id = held_numerical("F")
    study.closure(run_id, seed=10202)
    result = study.advance()
    assert (result["step"]["kind"], result["outcome"]) == ("blocked", None)
    assert any("closure-c1-F" in p for p in result["problems"])


def test_a_closure_whose_run_is_no_hold_is_a_problem() -> None:
    study = Study()
    study.through_validation()
    study.record("closure-c1-P", "failed", candidate=1, arm="P", rate=START["P"]["l"],
                 seed=10201, run_id="no-such-run")  # fmt: skip
    result = study.advance()
    assert result["step"]["kind"] == "blocked"
    assert any("closure-c1-P" in p for p in result["problems"])


def test_a_held_validation_run_of_f_ends_with_the_technical_report_once_closed() -> None:
    study = Study()
    study.through_validation()
    run_id = study.hold(run_of(study, "validation", "F", "u", 10304))
    result = study.advance()
    assert (result["stage"], result["step"]["record"]) == (
        "technical_feasibility_report", "closure-c1-F")  # fmt: skip
    study.closure(run_id)
    assert study.advance()["outcome"] == "technical_feasibility_report"


# ------------------------------------------------------------------ the scientific repair


def repair_world() -> tuple[Study, float]:
    """A's lower rate kept at the search (median of its four seeds 0.05 from its target) and
    missing by 0.10 over its 4 + 3 seeds: one repair round, its rate qualified as repair runs."""
    study = Study()
    study.ready()
    study.search(offsets={("A", "l"): [0.0, 0.0, 0.1, 0.1]})
    study.numerical(special={("A", "l", s): {"K": 4.8 * math.exp(0.1)}
                             for s in SEEDS["numerical"][1]})  # fmt: skip
    return study, adjusted_rate(START["A"]["l"], math.log(4.8) + 0.1, 4.8, 1.25)


def test_a_k_miss_is_repaired_once_and_the_repaired_rate_is_validated() -> None:
    study, rate = repair_world()
    result = study.advance()
    assert (result["stage"], result["step"]["campaign"], result["step"]["step"]) == (
        "repair", "repair", "repair")  # fmt: skip
    assert cells(result["step"]) == {("A", rate, s) for s in SEEDS["numerical"][1]}
    for seed in SEEDS["numerical"][1]:
        study.add(ep("A", rate, seed, step="repair", K=4.8))
    result = study.advance()  # its qualification is still to come
    assert (result["stage"], result["step"]["kind"]) == ("repair", "pending")
    study.rate(1, "A", rate)
    result = study.advance()
    assert result["stage"] == "validation"
    assert result["rates"]["A"] == {"l": rate, "u": START["A"]["u"]}
    assert ("A", rate, 10300) in cells(result["step"])


@pytest.mark.parametrize("state", ["technical", "held"])
def test_a_repair_whose_qualification_run_ended_technical_or_held_is_a_blocker(state) -> None:
    study, rate = repair_world()
    for seed in SEEDS["numerical"][1]:
        study.add(ep("A", rate, seed, step="repair", K=4.8))
    study.rate(1, "A", rate)
    run_id = run_of(study, "repair", "A", "l", 10201, rates={"A": {"l": rate}})
    if state == "technical":
        found = study.drop(run_id)
        study.add(Endpoint(**{**found.__dict__, "state": "technical", "K": None}))
    else:
        study.cycle()
        study.hold(run_id)
    result = study.advance()
    assert result["stage"] in ("repair", "technical_feasibility_report")
    assert result["stage"] != "validation" and result["outcome"] is None
    if state == "technical":
        assert (result["stage"], result["step"]["kind"], result["step"]["technical"]) == (
            "repair", "blocked", True)  # fmt: skip
    else:
        assert result["step"]["record"] == "closure-c1-A"


# ------------------------------------------------------------------ decision records


@pytest.mark.parametrize("built, stage, record", [
    ([], "engineering", "engineering"),
    (["engineering"], "simulation", "simulation"),
])  # fmt: skip
def test_a_missing_engineering_or_simulation_record_stops_at_its_step(built, stage, record):
    study = Study()
    if built:
        study.record("engineering")
    result = study.advance()
    assert (result["stage"], result["step"]["kind"], result["step"]["record"]) == (
        stage, "decision", record)  # fmt: skip
    assert result["step"]["path"] == f"study3/decisions/{record}.json"
    assert result["problems"] == [] and result["outcome"] is None


def test_a_missing_n_record_stops_at_the_choice() -> None:
    study = Study()
    study.through_validation()
    result = study.advance()
    assert (result["stage"], result["step"]["record"]) == ("choice", "n-c1")


@pytest.mark.parametrize("N, power_short, n_min", [(99, False, None), (84, True, 84)])
def test_an_n_that_asks_for_a_budget_decision_stops_at_the_budget_step(N, power_short, n_min):
    """N above 90, or N raised for power at 90 or less: the extra blocks are the researcher's
    decision."""
    study = Study()
    study.ready(power_short=power_short, n_min=n_min)
    study.search()
    study.numerical()
    study.validation()
    study.n(N=N)
    result = study.advance()
    assert (result["stage"], result["step"]["record"]) == ("budget", "budget-c1")
    fields = result["step"]["fields"]
    assert (fields["N"], fields["cap"]) == (N, cap_of(N)) and fields["why"]
    assert "c1/confirmation" not in result["prescribed"]
    study.budget()
    result = study.advance()
    assert (result["stage"], result["step"]["campaign"]) == ("confirmation", "confirmation")
    assert len(result["prescribed"]["c1/confirmation"]) == 6 * cap_of(N)


def test_no_budget_decision_is_asked_at_90_without_a_power_shortfall() -> None:
    study = Study()
    study.through_validation()
    study.n(N=90)
    assert study.advance()["stage"] == "confirmation"


@pytest.mark.parametrize(
    "decision, stage", [("proceed", "confirmation"), ("stop", "technical_feasibility_report")]
)
def test_a_budget_record_for_an_n_that_needs_none_is_read_as_written(decision, stage) -> None:
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.budget(decision=decision)
    assert study.advance()["stage"] == stage


def test_a_budget_record_before_its_stage_is_a_problem() -> None:
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.budget()
    study.entries.pop("n-c1")
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("choice", "blocked")
    assert any("budget-c1" in p for p in result["problems"])


def test_a_budget_decision_to_stop_starts_no_confirmation() -> None:
    study = Study()
    study.through_validation()
    study.n(N=99)
    study.budget(decision="stop")
    result = study.advance()
    assert (result["stage"], result["outcome"]) == (
        "technical_feasibility_report", "technical_feasibility_report")  # fmt: skip
    assert "c1/confirmation" not in result["prescribed"]


def test_a_failed_n_record_starts_no_confirmation_and_never_cuts_n() -> None:
    study = Study()
    study.through_validation()
    study.n(N=179)  # cap 201: beyond the 200 reserved confirmation seeds
    result = study.advance()
    assert (result["stage"], result["outcome"]) == (
        "technical_feasibility_report", "technical_feasibility_report")  # fmt: skip
    assert any("201" in r for r in result["reasons"])
    assert "c1/confirmation" not in result["prescribed"]


@pytest.mark.parametrize("kind, stage", [("engineering", "engineering"),
                                         ("simulation", "simulation"), ("n-c1", "n"),
                                         ("budget-c1", "budget")])  # fmt: skip
def test_an_invalid_record_blocks_its_stage(kind, stage) -> None:
    study = Study()
    study.through_validation()
    study.n(N=99)
    study.budget()
    study.invalid(kind)
    result = study.advance()
    assert (result["stage"], result["step"]["kind"], result["outcome"]) == (stage, "blocked", None)
    assert any(kind in r for r in result["step"]["reasons"])


def test_an_invalid_closure_blocks() -> None:
    study, run_id = held_numerical("F")
    study.invalid("closure-c1-F")
    result = study.advance()
    assert (result["step"]["kind"], result["outcome"]) == ("blocked", None)


@pytest.mark.parametrize("kind", ["engineering", "simulation"])
def test_a_failed_engineering_or_simulation_record_ends_with_the_technical_report(kind) -> None:
    study = Study()
    study.ready()
    study.entries[kind]["state"] = study.entries[kind]["record"]["state"] = "failed"
    result = study.advance()
    assert (result["stage"], result["outcome"]) == (
        "technical_feasibility_report", "technical_feasibility_report")  # fmt: skip
    assert any(kind in r for r in result["reasons"])
    assert "c1/search/r0" not in result["prescribed"]


@pytest.mark.parametrize("kind", ["engineering", "simulation", "n-c1"])
def test_a_records_code_that_disagrees_with_the_expected_code_is_a_problem(kind) -> None:
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.entries[kind]["record"]["code"] = {"commit": "a" * 40, "gpu_code_sha256": "c" * 64}
    result = study.advance()
    assert (result["step"]["kind"], result["outcome"]) == ("blocked", None)
    assert any(kind in p and "c" * 64 in p for p in result["problems"])
    assert result["stage"] == {"engineering": "engineering", "simulation": "simulation",
                               "n-c1": "n"}[kind]  # fmt: skip


def test_the_n_records_cohort_run_ids_are_counting_validation_runs_of_its_candidate() -> None:
    study = Study()
    study.through_validation()
    record = study.n(N=60)
    record["cohort"]["run_ids"] = [*record["cohort"]["run_ids"][:-1], "not-a-validation-run"]
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("n", "blocked")
    assert any("not-a-validation-run" in p for p in result["problems"])


@pytest.mark.parametrize("change", ["rates", "active", "seeds"])
def test_the_n_record_names_the_validated_rates_comparisons_and_supported_blocks(change) -> None:
    study = Study()
    study.through_validation()
    record = study.n(N=60)
    if change == "rates":
        record["rates"]["A"]["lr_l"] = 1.5e-5
    elif change == "active":
        record["active"] = ["A-F"]
    else:
        record["cohort"]["seeds"] = record["cohort"]["seeds"][:-1]
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("n", "blocked")


def test_the_n_records_cohort_is_every_supported_validation_block() -> None:
    """A cohort that leaves out a supported block (and its runs) is not the spread's cohort."""
    study = Study()
    study.through_validation()
    record = study.n(N=60)
    last = record["cohort"]["seeds"].pop()
    record["cohort"]["run_ids"] = [r for r in record["cohort"]["run_ids"] if str(last) not in r]
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("n", "blocked")
    assert any("supported" in p for p in result["problems"])


def test_an_n_record_before_the_validation_decided_is_a_problem() -> None:
    study = Study()
    study.ready()
    study.search()
    study.n(N=60, seeds=[10300], run_ids=["x1", "x2", "x3", "x4"])
    result = study.advance()
    assert result["step"]["kind"] == "blocked"
    assert any("n-c1" in p for p in result["problems"])


def test_a_search_run_started_before_the_engineering_record_was_written_is_a_problem() -> None:
    study = Study()
    study.ready()
    study.search()
    early = run_of(study, "search", "F", "u", 10102)
    study.starts[early] = datetime(2026, 10, 24, tzinfo=UTC)
    for e in study.endpoints:
        study.starts.setdefault(e.run_id, datetime(2026, 10, 26, tzinfo=UTC))
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("search", "blocked")
    assert any(early in p and "engineering" in p for p in result["problems"])


# ------------------------------------------------------------------ problems block


def test_problems_of_the_run_records_block_and_decide_nothing() -> None:
    study = Study()
    study.through_validation()
    study.problems.append("run-x: refused: a seed outside its role")
    result = study.advance()
    assert (result["step"]["kind"], result["outcome"]) == ("blocked", None)
    assert result["step"]["reasons"] == ["run-x: refused: a seed outside its role"]
    assert result["stage"] == "search" and result["candidate"] == 1


def test_problems_of_the_run_records_are_named_with_an_invalid_gate_record() -> None:
    study = Study()
    study.ready()
    study.invalid("engineering", "the launch journal is missing")
    study.problems.append("run-x: refused")
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("engineering", "blocked")
    assert result["step"]["reasons"] == [
        "run-x: refused",
        "/runs/engineering.json: the launch journal is missing",
    ]


def test_problems_of_the_holds_block() -> None:
    study = Study()
    study.ready()
    study.hold_problems.append("launch/x.jsonl: an unreadable launch journal")
    assert study.advance()["step"]["kind"] == "blocked"


def test_problems_of_the_qualification_block_from_the_numerical_stage() -> None:
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    study.readings[1]["problems"].append("verdict.json: the cache disagrees")
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("numerical", "blocked")
    assert "verdict.json: the cache disagrees" in result["step"]["reasons"]


def test_the_lenient_derivation_of_the_cells_ignores_problems_but_not_the_route() -> None:
    study = Study()
    study.ready()
    study.search(offsets={("F", "u"): 0.2})
    study.problems.append("run-x: refused")
    strict = study.advance()
    lenient = study.advance(lenient=True)
    assert strict["step"]["kind"] == "blocked"
    assert "c1/search/r1" not in strict["prescribed"]
    assert lenient["step"]["campaign"] == "search-round1"
    assert len(lenient["prescribed"]["c1/search/r1"]) == 4


# ------------------------------------------------------------------ the confirmation's state


def confirmation_endpoints(N_blocks, *, arms=ARMS, special=None, skip=()) -> list[Endpoint]:
    study = Study()
    seeds = [s for s in SEEDS["confirmation"][1][:N_blocks] if s not in skip]
    study.blocks("confirmation", 1, arms, seeds, special=special)
    return study.endpoints


def unsupported(seeds, arm="A") -> dict:
    return {(s, arm, "u"): {"K": 5.0} for s in seeds}


def test_the_cohorts_are_the_first_n_supported_blocks_by_seed_order() -> None:
    seeds = list(SEEDS["confirmation"][1])
    endpoints = confirmation_endpoints(66, special=unsupported(seeds[2:5]))
    state = confirmation_state(endpoints, 1, 60, ["A-F", "P-F"])
    assert state["started"] == seeds[:66] and state["problems"] == []
    assert state["counters"] == {"A-F": 63, "P-F": 66}
    assert state["cohorts"]["A-F"] == [*seeds[:2], *seeds[5:63]]
    assert state["cohorts"]["P-F"] == seeds[:60]
    assert state["stop"] == {"stopped": True, "why": "counters"}
    assert state["may_start_next"] is False and state["next_seed"] is None
    assert state["readings"] == {"A-F": "confirmatory", "P-F": "confirmatory"}


def test_an_unfinished_earlier_block_holds_the_later_ones() -> None:
    seeds = list(SEEDS["confirmation"][1])
    endpoints = confirmation_endpoints(10, special={(seeds[3], "F", "l"): {"state": "pending"}})
    state = confirmation_state(endpoints, 1, 60, ["A-F", "P-F"])
    assert state["counters"] == {"A-F": 3, "P-F": 3}
    assert state["held_up"] == {"A-F": seeds[3], "P-F": seeds[3]}
    assert state["supported"]["A-F"] == 9
    assert state["may_start_next"] is True and state["next_seed"] == seeds[10]
    # a technical block holds them too, until its diagnosis
    endpoints = confirmation_endpoints(10, special={(seeds[3], "P", "l"): {"state": "technical"}})
    state = confirmation_state(endpoints, 1, 60, ["A-F", "P-F"])
    assert state["counters"] == {"A-F": 10, "P-F": 3}
    assert [t["seed"] for t in state["technical"]] == [seeds[3]]


def test_an_inactive_p_never_holds_up_a_f() -> None:
    seeds = list(SEEDS["confirmation"][1])
    endpoints = confirmation_endpoints(60, arms=("A", "F"))
    state = confirmation_state(endpoints, 1, 60, ["A-F"])
    assert state["counters"] == {"A-F": 60} and state["stop"]["why"] == "counters"
    assert set(state["readings"]) == {"A-F"}
    assert state["cohorts"]["A-F"] == seeds[:60]


def test_retries_add_no_block_and_journal_only_starts_count() -> None:
    seeds = list(SEEDS["confirmation"][1])
    endpoints = confirmation_endpoints(3)
    first = next(e for e in endpoints if e.seed == seeds[1] and e.arm == "A")
    endpoints.append(Endpoint(**{**first.__dict__, "run_id": "retry-old", "attempt": 0,
                                 "state": "recovered", "K": None}))  # fmt: skip
    # a start in the journal with no directory yet: a started block
    endpoints.append(ep("A", START["A"]["l"], seeds[3], step="confirmation", state="pending"))
    state = confirmation_state(endpoints, 1, 60, ["A-F", "P-F"])
    assert state["started"] == seeds[:4]
    assert state["counters"] == {"A-F": 3, "P-F": 3}
    assert state["next_seed"] == seeds[4]


def test_a_start_with_no_journal_start_is_no_block() -> None:
    endpoints = confirmation_endpoints(2)
    bare = ep("A", START["A"]["l"], 10405, step="confirmation", state="refused", campaign=None)
    state = confirmation_state([*endpoints, bare], 1, 60, ["A-F"])
    assert state["started"] == [10400, 10401]


@pytest.mark.parametrize("supported, reading", [(35, "descriptive"), (36, "confirmatory")])
def test_the_reading_at_the_cap_is_confirmatory_from_36_supported_blocks(supported, reading):
    seeds = list(SEEDS["confirmation"][1])
    cap = cap_of(60)  # 70
    endpoints = confirmation_endpoints(cap, arms=("A", "F"),
                                       special=unsupported(seeds[supported:cap]))  # fmt: skip
    state = confirmation_state(endpoints, 1, 60, ["A-F"])
    assert state["counters"] == {"A-F": supported} and state["cap"] == cap
    assert state["stop"] == {"stopped": True, "why": "cap"}
    assert state["readings"] == {"A-F": reading}
    assert state["analysis"]["A-F"] == seeds[:supported]
    assert state["may_start_next"] is False


def test_at_the_cap_an_unfinished_block_leaves_the_reading_open() -> None:
    seeds = list(SEEDS["confirmation"][1])
    endpoints = confirmation_endpoints(70, arms=("A", "F"), special={
        **unsupported(seeds[40:70]), (seeds[5], "F", "u"): {"state": "pending"}})  # fmt: skip
    state = confirmation_state(endpoints, 1, 60, ["A-F"])
    assert state["stop"]["why"] == "cap" and state["readings"] == {"A-F": None}


@pytest.mark.parametrize("started, stopped", [(59, False), (60, True)])
def test_may_start_next_is_false_once_every_counter_reaches_n(started, stopped) -> None:
    state = confirmation_state(confirmation_endpoints(started), 1, 60, ["A-F", "P-F"])
    assert state["may_start_next"] is not stopped
    assert state["stop"]["stopped"] is stopped


@pytest.mark.parametrize("started, stopped", [(69, False), (70, True)])
def test_may_start_next_is_false_once_the_cap_has_started(started, stopped) -> None:
    seeds = list(SEEDS["confirmation"][1])
    endpoints = confirmation_endpoints(started, special=unsupported(seeds[:20]))
    state = confirmation_state(endpoints, 1, 60, ["A-F", "P-F"])
    assert state["may_start_next"] is not stopped
    assert state["stop"] == {"stopped": stopped, "why": "cap" if stopped else None}


def test_blocks_out_of_seed_order_or_beyond_the_cap_are_problems() -> None:
    seeds = list(SEEDS["confirmation"][1])
    endpoints = confirmation_endpoints(4, skip=(seeds[1],))
    state = confirmation_state(endpoints, 1, 60, ["A-F"])
    assert state["problems"] and state["may_start_next"] is False
    endpoints = confirmation_endpoints(cap_of(60) + 1)
    state = confirmation_state(endpoints, 1, 60, ["A-F"])
    assert any("cap" in p for p in state["problems"])


def test_confirmation_state_refuses_what_it_cannot_read() -> None:
    for args in ((3, 60, ["A-F"]), (1, 0, ["A-F"]), (1, 60.0, ["A-F"]), (1, 60, ["P-F"]),
                 (1, 60, ["A-F", "B-F"]), (1, 60, [])):  # fmt: skip
        with pytest.raises(ValueError):
            confirmation_state([], *args)


def test_the_confirmation_never_moves_the_candidate_or_closes_p() -> None:
    study = Study()
    study.through_validation()
    study.n(N=60)
    seeds = list(SEEDS["confirmation"][1])
    study.confirmation(count=70, special={**unsupported(seeds[:40], "P"),
                                          **unsupported(seeds[:40], "F")})  # fmt: skip
    result = study.advance()
    assert (result["stage"], result["candidate"], result["outcome"]) == (
        "done", 1, "descriptive")  # fmt: skip
    assert result["readings"] == {"A-F": "descriptive", "P-F": "descriptive"}
    assert result["active"]["arms"] == ["A", "F", "P"]
    assert not any(e["event"] in ("candidate", "close_pf") for e in result["route"])


def test_a_technical_confirmation_block_goes_to_diagnosis_and_holds_the_end() -> None:
    """The confirmation goes on while a block waits for its diagnosis (the stop is the counters'
    or the cap's); once stopped, the readings wait for it."""
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.confirmation(count=10)
    broken = study.change(run_of(study, "confirmation", "F", "l", 10403), state="technical",
                          K=None)  # fmt: skip
    result = study.advance()
    assert (result["step"]["campaign"], result["step"]["diagnosis"]) == (
        "confirmation", [broken.run_id])  # fmt: skip
    assert {c[2] for c in result["step"]["cells"]} == {10410}
    study.confirmation(count=60, first=10)  # 70 started: the cap
    result = study.advance()
    assert result["confirmation"]["stop"] == {"stopped": True, "why": "cap"}
    assert (result["stage"], result["step"]["kind"], result["step"]["technical"]) == (
        "confirmation", "blocked", True)  # fmt: skip
    assert result["step"]["reasons"] == [broken.run_id]


def test_a_recovered_confirmation_cell_waits_for_its_next_attempt_and_is_not_issued_again():
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.confirmation(count=3)
    study.change(run_of(study, "confirmation", "A", "u", 10401), state="recovered", K=None)
    result = study.advance()
    assert (result["step"]["campaign"], {c[2] for c in result["step"]["cells"]}) == (
        "confirmation", {10403})  # fmt: skip
    assert result["confirmation"]["held_up"]["A-F"] == 10401


def test_a_partly_issued_confirmation_block_is_completed_before_the_next() -> None:
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.confirmation(count=3)
    study.drop(run_of(study, "confirmation", "P", "u", 10401))
    result = study.advance()
    assert cells(result["step"]) == {("P", START["P"]["u"], 10401)}


# ------------------------------------------------------------------ the reader


THETA0_RUN = Path("study2_stage_a") / "20261014T192927Z-theta0-s2000-abb703a1-a1"


@pytest.fixture
def worlds(tmp_path):
    """A runs directory with its decision records (and their evidence, in a checkout), and the
    runs: the reader tests' helpers on one root."""
    records = DecisionWorld(tmp_path)
    shutil.rmtree(records.run)  # the records tests' own sentinel run: no run of this study
    runs = RunWorld(tmp_path / "runs")
    # the engineering stage's log: a decision record is checked against the journal
    (runs.root / "launch").mkdir()
    (runs.root / "launch" / "study3-c1-engineering.jsonl").write_text("")
    return records, runs


def read(records, runs, **kw) -> dict:
    return progress(runs.root, repo=records.repo, expected_code=EXPECTED, theta0=runs.theta0, **kw)


def search_round0(runs, offsets=None) -> None:
    spec = CANDIDATES[1]
    for arm in ARMS:
        for label in LABELS:
            for seed in SEEDS["search"][1]:
                k = spec["targets"][label] * math.exp((offsets or {}).get((arm, label), 0.0))
                runs.run(arm, spec["start"][arm][label], seed, role="search", round_=0, k=k,
                         prescribe=False, campaign="study3-c1-search-r0")  # fmt: skip


def test_the_reader_reads_an_empty_study_at_its_first_step(worlds) -> None:
    records, runs = worlds
    result = read(records, runs)
    assert (result["stage"], result["step"]["record"], result["problems"]) == (
        "engineering", "engineering", [])  # fmt: skip


def test_the_reader_accepts_exactly_the_cells_the_rule_issues(worlds) -> None:
    records, runs = worlds
    records.write("engineering", records.engineering(written_at="2026-10-25T00:00:00Z"))
    records.write("simulation", records.simulation())
    result = read(records, runs)
    assert (result["stage"], result["step"]["campaign"]) == ("search", "search-round0")
    search_round0(runs, {("F", "u"): 0.2})
    result = read(records, runs)
    rate = adjusted_rate(START["F"]["u"], math.log(7.5) + 0.2, 7.5, 1.30)
    assert result["problems"] == []
    assert (result["step"]["campaign"], cells(result["step"])) == (
        "search-round1", {("F", rate, s) for s in SEEDS["search"][1]})  # fmt: skip
    for seed in SEEDS["search"][1]:
        runs.run("F", rate, seed, role="search", round_=1, k=7.5, prescribe=False,
                 campaign="study3-c1-search-r1")  # fmt: skip
    result = read(records, runs)
    assert (result["stage"], result["step"]["campaign"], result["problems"]) == (
        "numerical", "numerical", [])  # fmt: skip
    assert ("F", rate, 10200) in cells(result["step"])
    # a round-1 run at a rate the rule never gave is refused, and blocks
    stray = runs.run("A", 1.5e-5, 10100, role="search", round_=1, k=4.8, prescribe=False,
                     campaign="study3-c1-search-r1")  # fmt: skip
    result = read(records, runs)
    assert result["step"]["kind"] == "blocked"
    assert any(stray in p and "not prescribed" in p for p in result["problems"])


def test_the_decision_records_folder_is_no_run_directory(worlds) -> None:
    """``RUNS_DIR/study3/decisions/`` lies in the experiment's folder; it is no run, while any
    other folder with no start stays refused."""
    records, runs = worlds
    records.write("engineering", records.engineering(written_at="2026-10-25T00:00:00Z"))
    assert (runs.root / "study3" / "decisions").is_dir()
    result = read(records, runs)
    assert (result["stage"], result["step"]["record"], result["problems"]) == (
        "simulation", "simulation", [])  # fmt: skip
    (runs.root / "study3" / "20261026T000099Z-A-s10300-abcd1234-a1").mkdir()
    result = read(records, runs)
    assert result["step"]["kind"] == "blocked"
    assert any("20261026T000099Z" in p and "no start" in p for p in result["problems"])


def test_the_reader_blocks_on_a_refused_run(worlds) -> None:
    records, runs = worlds
    records.write("engineering", records.engineering(written_at="2026-10-25T00:00:00Z"))
    records.write("simulation", records.simulation())
    refused = runs.run("A", 1.45e-5, 10120, role="search", prescribe=False)
    result = read(records, runs)
    assert result["step"]["kind"] == "blocked"
    assert any(refused in p for p in result["problems"])


def test_a_missing_theta0_is_a_problem_not_an_exception(worlds) -> None:
    records, runs = worlds
    result = progress(runs.root, repo=records.repo, expected_code=EXPECTED,
                      theta0=runs.root / "nowhere")  # fmt: skip
    assert result["step"]["kind"] == "blocked"


def test_the_reader_reads_a_closure_record_from_its_files(worlds, monkeypatch) -> None:
    """A held F of the numerical qualification: the closure step names the record, the record
    written from it (with its real hold line and cycle record) ends the route."""
    records, runs = worlds
    records.write("engineering", records.engineering(written_at="2026-10-25T00:00:00Z"))
    records.write("simulation", records.simulation())
    search_round0(runs)
    for arm in ARMS:
        for label in LABELS:
            for seed in SEEDS["numerical"][1]:
                runs.run(arm, START[arm][label], seed, role="numerical", prescribe=False,
                         campaign="study3-c1-numerical",
                         k=CANDIDATES[1]["targets"][label])  # fmt: skip
    # the cycle held F's 10201 run of its lower rate
    held_id = next(p.name for p in (runs.root / "study3").iterdir()
                   if f"F-{START['F']['l']:g}-s10201" in p.name)  # fmt: skip
    closure = decision_hold(records, arm="F", rate=START["F"]["l"], seed=10201, run_id=held_id)
    from zero_field_refusal.analysis import qualification_records

    def fake(root, candidate, *, expected_code, settings):
        out = reading()
        if candidate == 1:
            out["supersessions"] = [{"path": "c1/supersede-1.json"}]
        return out

    monkeypatch.setattr(qualification_records, "read_qualification", fake)
    result = read(records, runs, settings={})
    assert result["problems"] == []
    assert (result["stage"], result["step"]["record"]) == (
        "technical_feasibility_report", "closure-c1-F")  # fmt: skip
    fields = result["step"]["fields"]
    assert {k: fields[k] for k in ("hold", "cycle", "run_id", "seed", "reason")} == {
        k: closure[k] for k in ("hold", "cycle", "run_id", "seed", "reason")
    }
    records.write(
        "closure-c1-F",
        {
            "kind": "closure-c1-F",
            "state": "failed",
            "written_at": "2026-10-26T13:00:00Z",
            "code": CODE,
            **{k: v for k, v in fields.items() if k != "reasons"},
        },
    )
    result = read(records, runs, settings={})
    assert (result["stage"], result["outcome"]) == (
        "technical_feasibility_report", "technical_feasibility_report")  # fmt: skip


def test_the_reader_never_opens_responses_or_observations(worlds, monkeypatch) -> None:
    """A study through three validation blocks of regular runs, each with its sentinel response
    and observation files: none is opened, and the sentinel is nowhere in the result."""
    records, runs = worlds
    records.write("engineering", records.engineering(written_at="2026-10-25T00:00:00Z"))
    records.write("simulation", records.simulation())
    search_round0(runs)
    for arm in ARMS:
        for label in LABELS:
            for seed in SEEDS["numerical"][1]:
                runs.run(arm, START[arm][label], seed, role="numerical", prescribe=False,
                         campaign="study3-c1-numerical",
                         k=CANDIDATES[1]["targets"][label])  # fmt: skip
            for seed in SEEDS["validation"][1][:3]:
                runs.run(arm, START[arm][label], seed, role="validation", prescribe=False,
                         campaign="study3-c1-validation", k=CANDIDATES[1]["targets"][label],
                         mmlu=1.0, factual=2.0)  # fmt: skip
    # a conflict end whose free text is the sentinel
    conflict = runs.run("A", START["A"]["l"], 10303, role="validation", prescribe=False,
                        campaign="study3-c1-validation", end=None)  # fmt: skip
    runs.end("study3-c1-validation", f"A-{START['A']['l']:g}/s10303", 1, conflict, "conflict",
             problem=SENTINEL)  # fmt: skip
    from zero_field_refusal.analysis import qualification_records

    def fake(root, candidate, *, expected_code, settings):
        out = reading()
        if candidate == 1:
            for arm in ARMS:
                for label in LABELS:
                    lr = START[arm][label]
                    key = f"{OPT[arm]}-{lr:g}"
                    out["rates"][key] = {"key": key, "arm": arm, "optimizer": OPT[arm], "lr": lr,
                                         "state": "qualified", "runs": [], "held": [],
                                         "verdict": {"state": "qualified", "control": None,
                                                     "not_counted": []}}  # fmt: skip
        return out

    monkeypatch.setattr(qualification_records, "read_qualification", fake)
    assert list((runs.root / "study3").glob("*/responses/*.jsonl"))
    sealed = seal(monkeypatch)
    result = read(records, runs, settings={})
    assert sealed == []
    assert (result["stage"], result["problems"]) == ("validation", [])
    assert result["step"]["kind"] == "blocked"  # the conflict is technical: its diagnosis first
    assert any(conflict in r for r in result["step"]["reasons"])
    assert SENTINEL not in tidy(result)
    with pytest.raises(AssertionError):  # the guard is live
        next((runs.root / "study3").glob("*/responses/*.jsonl")).read_text()


def test_the_module_imports_no_gpu_path() -> None:
    """The state machine and the confirmation's state import no GPU path; the reader loads the
    qualification's reader (and with it torch, through ``qualification_verdicts``) only when it
    runs."""
    code = ("import sys; import zero_field_refusal.analysis.study3_progress; "
            "print([m for m in ('torch', 'transformers', 'zero_field_refusal.schedule', "
            "'zero_field_refusal.train', 'zero_field_refusal.launch', "
            "'zero_field_refusal.qualification', 'zero_field_refusal.analysis.stage_a') "
            "if m in sys.modules])")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
    assert study3_progress.THETA0_RUN == THETA0_RUN
