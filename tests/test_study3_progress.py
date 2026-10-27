"""Study 3's state machine (analysis/study3_progress.py): where the study is and what comes next.

The routes are tested on the pure core (``advance``) over records built by hand: endpoints,
qualification readings, decision entries and holds, as the readers give them; no file. The reader
(``progress``) is tested on synthetic RUNS_DIR trees built with the reader tests' helpers: the
launch journal, run directories whose regular runs hold a harmful-response file and an
observations file with a sentinel text (never opened), and decision records with their evidence.
The registered K* of candidate 1 is 6.00; runs at K_l and K_u bracket it with the weight 1/2."""

import builtins
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_qualification_records import qualify as qualify  # noqa: F401  (the fixture)
from test_study3_decisions import World as DecisionWorld
from test_study3_decisions import held as decision_hold
from test_study3_records import SENTINEL, header, seal
from test_study3_records import World as RunWorld

from zero_field_refusal.analysis import study3_progress
from zero_field_refusal.analysis.study3_decisions import START_REASON, cap_of, end_reason
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
ENGINEERING_SHA = "1" * 64  # the engineering record file's sha256 now
NAMED = {"engineering_record": "study3/decisions/engineering.json",
         "engineering_record_sha256": ENGINEERING_SHA}  # a search header naming it  # fmt: skip


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
        self.headers: dict[str, dict] = {}  # run id -> its campaign's header
        self.digests: dict[str, str] = {}
        self.ends: dict[str, dict] = {}  # run id -> its last end line
        self.starts: dict[str, dict] = {}  # run id -> its start line
        self.recoverable: set[str] = set()  # runs the launcher would open a recovery round for
        self.cycles: dict[int, list[dict]] = {1: [], 2: []}  # each candidate's cycle records
        self.expected = EXPECTED  # the chain of engineering cycles' end
        self.uncounted: dict[str, dict] = {}  # runs under another commit than the pinned one

    # -- decision records

    def record(self, kind, state="passed", *, code=None, **fields) -> dict:
        """A decision record: the engineering and simulation records under the chain's start,
        the others under the chain's end as it is now, unless ``code`` (a GPU code) says."""
        if code is None:
            code = EXPECTED if kind in ("engineering", "simulation") else self.expected
        record = {"kind": kind, "state": state, "written_at": WHEN,
                  "code": {"commit": "a" * 40, "gpu_code_sha256": code}, **fields}  # fmt: skip
        self.entries[kind] = entry(kind, state, record)
        if kind == "engineering":
            self.digests["engineering"] = ENGINEERING_SHA
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
                    found = ep(arm, spec["start"][arm][label], seed, step="search",
                               candidate=candidate, **kw)  # fmt: skip
                    self.add(found)
                    self.headers[found.run_id] = {
                        "study": 3,
                        "candidate": candidate,
                        "step": "search",
                        "round": 0,
                        **NAMED,
                    }

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

    def cycle(self, candidate=1, fix=None, old=None) -> str:
        """The candidate's engineering cycle is recorded: its supersession record moves the
        chain of GPU codes from ``old`` (the chain's end by default) to ``fix``."""
        fix = fix or ("f" if candidate == 1 else "9") * 64
        self.cycles[candidate].append(
            {"path": f"/runs/qualification/c{candidate}/supersede-1.json", "sha256": "9" * 64,
             "old_gpu_code": old or self.expected, "fix_gpu_code": fix})  # fmt: skip
        self.expected = fix
        return fix

    def technical(self, run_id, outcome="technical", *, log="launch/study3-c1-validation.jsonl",
                  line=7, published=False, **fields) -> Endpoint:  # fmt: skip
        """A run the launcher ended (``outcome``) that the readers count as a technical mark:
        its endpoint technical (or ``fields``), its last end line as ``end_lines`` gives it, its
        terminal record published or not."""
        found = self.change(run_id, **(fields or {"state": "technical", "K": None,
                                                  "steps": None}))  # fmt: skip
        if not published:  # the reader's conditions: no terminal record, the cell's round unused
            self.recoverable.add(run_id)
        self.ends[run_id] = {
            "path": log,
            "line": line,
            "line_sha256": "a" * 64,
            "outcome": outcome,
            "attempt": 1,
            "run": found.label,
        }
        return found

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
        """The closure record that cites a hold (its held line and cycle) or a technical run
        (its last end line, no cycle)."""
        (found,) = [e for e in self.endpoints if e.run_id == run_id]
        holds = [h for h in self.holds if h["run_id"] == run_id]
        if holds:
            journal, cycle = holds[0]["records"][0], holds[0]["records"][1]
            line = {"kind": "held", "path": journal["path"], "line": journal["line"],
                    "line_sha256": journal["line_sha256"]}  # fmt: skip
            cycle, reason = {"path": cycle["path"], "sha256": cycle["sha256"]}, "no_recorded_commit"
        elif run_id in self.ends:
            end = self.ends[run_id]
            line = {"kind": "end", **{k: end[k] for k in ("path", "line", "line_sha256")}}
            cycle, reason = None, end_reason(end["outcome"])
        else:
            start = self.starts[run_id]
            line = {"kind": "start", **{k: start[k] for k in ("path", "line", "line_sha256")}}
            cycle, reason = None, START_REASON
        if run_id in self.uncounted:  # a run under another commit: its uncounted reason
            reason = self.uncounted[run_id]["reason"]
        record = {
            "candidate": found.candidate, "arm": found.arm, "rate": found.lr, "seed": found.seed,
            "run_id": run_id, "line": line, "cycle": cycle, "reason": reason,
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
                       headers=self.headers, digests=self.digests, ends=self.ends,
                       starts=self.starts, cycles=self.cycles,
                       recoverable=frozenset(self.recoverable),
                       uncounted=self.uncounted)  # fmt: skip

    def advance(self, **kw) -> dict:
        return advance(self.records(), **{"expected_code": self.expected, **kw})

    def other_commit(self, run_id, reason="other_commit", commit="e" * 40, *,
                     ended=True) -> dict:  # fmt: skip
        """A run under a commit other than the pinned one, as ``uncounted_runs`` lists it; its
        last end line (``done``) unless it has one, or unless ``ended`` is false."""
        (found,) = [e for e in self.endpoints if e.run_id == run_id]
        if ended and run_id not in self.ends:
            self.ends[run_id] = {"path": f"launch/study3-c1-{found.cell.step}.jsonl", "line": 9,
                                 "line_sha256": "d" * 64, "outcome": "done", "attempt": 1,
                                 "run": found.label}  # fmt: skip
        entry = {"run_id": run_id, "label": found.label, "candidate": found.candidate,
                 "step": found.cell.step, "reason": reason, "commit": commit, "dirty": False,
                 "versions_record": "docs/study3-versions.json",
                 "why": f"its commit {commit} is not the pinned one"}  # fmt: skip
        self.uncounted[run_id] = entry
        return entry


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
    assert (result["stage"], result["outcome"], result["reason"], result["step"]) == (
        "done", "confirmed", "n_reached",
        {"kind": "done", "outcome": "confirmed", "reason": "n_reached"})  # fmt: skip
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
        run = ep("F", rate, seed, step="search", round_=1, K=7.5)
        study.add(run)
        study.headers[run.run_id] = {
            "study": 3,
            "candidate": 1,
            "step": "search",
            "round": 1,
            **NAMED,
        }  # its campaign names the engineering record
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
    assert (result["stage"], result["outcome"], result["candidate"], result["reason"]) == (
        "scientific_feasibility_report", "scientific_feasibility_report", 2,
        "scientific_failure")  # fmt: skip
    assert result["step"] == {
        "kind": "report",
        "outcome": "scientific_feasibility_report",
        "reason": "scientific_failure",
    }
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
    assert result["step"] == {
        "kind": "report",
        "outcome": "technical_feasibility_report",
        "reason": "unresolved_after_cycle",
    }
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


ROUTE_PENDING = "engineering_route_pending"
ROUTE_WHY = ("the registered engineering cycle for a technical mark outside the qualification is "
             "not implemented yet: the researcher decides")  # fmt: skip
NOT_ACCEPTED = ("a closure of a technical mark outside the qualification is not accepted until "
                "its engineering route exists")  # fmt: skip


def assert_route_pending(result, *runs: Endpoint) -> None:
    """The stop for the researcher's decision: blocked, technical, with its reason id and why;
    each run named (its id, label, arm and step), its way out the decision, beside a recovery
    round where the launcher would open one; no closure offered, no outcome, no cell."""
    step = result["step"]
    assert (step["kind"], step["technical"], step["reason"], step["why"]) == (
        "blocked", True, ROUTE_PENDING, ROUTE_WHY)  # fmt: skip
    assert (result["outcome"], result["reason"], "cells" in step) == (None, None, False)
    assert [w["run_id"] for w in step["ways_out"]] == [e.run_id for e in runs]
    for e, way in zip(runs, step["ways_out"], strict=True):
        assert (way["label"], way["arm"], way["step"], way["decision"]) == (
            e.label, e.arm, e.cell.step, {"decision": ROUTE_PENDING})  # fmt: skip
        assert "closure" not in way
        assert any(e.run_id in r and e.label in r and e.cell.step in r for r in step["reasons"])


def closure_refused(result, arm: str) -> None:
    """A closure record written for such a mark: a named problem that blocks, deciding nothing."""
    assert (result["step"]["kind"], result["outcome"], result["reason"]) == ("blocked", None, None)
    assert any(p.startswith(f"/runs/study3/decisions/closure-c1-{arm}.json: ")
               and NOT_ACCEPTED in p for p in result["problems"]), result["problems"]  # fmt: skip
    assert not [e for e in result["route"] if e["event"] in ("report", "close_pf")]
    assert result["active"]["comparisons"] == ["A-F", "P-F"]


@pytest.mark.parametrize("step", ["search", "validation", "confirmation"])
@pytest.mark.parametrize("arm", ["A", "F", "P"])
def test_a_technical_mark_outside_the_qualification_stops_for_the_researchers_decision(
    step, arm
) -> None:
    """The registered route of a technical mark is one engineering cycle (diagnosis, fix,
    re-qualification); outside the qualification it is not implemented yet, and nothing cancels
    it silently: the step stops for the researcher's decision, with the recovery round where the
    launcher opens one (a validation or confirmation run's after ``check``), and no closure. A
    closure record written for the mark anyway is a problem and decides nothing."""
    study = Study()
    study.through_validation()
    if step == "confirmation":
        study.n(N=60)
        study.confirmation(count=5)
    seed = {"search": 10102, "validation": 10304, "confirmation": 10402}[step]
    broken = study.technical(run_of(study, step, arm, "u", seed),
                             log=f"launch/study3-c1-{step}.jsonl")  # fmt: skip
    result = study.advance()
    assert result["stage"] == step
    assert_route_pending(result, broken)
    (way,) = result["step"]["ways_out"]
    assert broken.label in way["recovery"]
    assert ("study3_summary.py check" in way["recovery"]) is (step != "search")
    if step == "confirmation":
        assert result["confirmation"]["may_start_next"] is False
    study.closure(broken.run_id)
    result = study.advance()
    assert result["stage"] == step
    closure_refused(result, arm)


def test_a_recovery_round_reruns_a_technical_run_and_its_next_attempt_decides() -> None:
    study = Study()
    study.through_validation()
    run_id = run_of(study, "validation", "F", "u", 10304)
    study.change(run_id, state="recovered", K=None)  # a recovery round with its cause followed
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("validation", "pending")
    assert run_id in result["step"]["waiting"]


def test_an_end_closure_of_a_run_that_is_not_technical_is_a_problem() -> None:
    """The readers must list the run as technical (no later attempt, no recovery): a closure of a
    usable run closes nothing, and blocks."""
    study = Study()
    study.through_validation()
    run_id = run_of(study, "validation", "F", "u", 10304)
    study.technical(run_id)
    study.closure(run_id)
    study.change(run_id, state="usable", K=7.5, steps=25)
    result = study.advance()
    assert result["step"]["kind"] == "blocked" and result["outcome"] is None
    assert any("closure-c1-F" in p for p in result["problems"])


def test_a_technical_run_the_launcher_never_ended_stops_and_its_start_closure_is_refused() -> None:
    study = Study()
    study.through_validation()
    broken = study.change(run_of(study, "validation", "P", "u", 10304), state="technical", K=None)
    study.starts[broken.run_id] = {
        "path": "launch/study3-c1-validation.jsonl",
        "line": 4,
        "line_sha256": "b" * 64,
        "attempt": 1,
        "run": broken.label,
    }
    assert_route_pending(study.advance(), broken)
    study.closure(broken.run_id)  # its start line
    closure_refused(study.advance(), "P")


def test_a_mark_whose_terminal_record_is_published_has_no_recovery_round() -> None:
    """The launcher opens a recovery round only for an attempt with no published terminal record:
    only the researcher's decision is left."""
    study = Study()
    study.through_validation()
    run_id = run_of(study, "validation", "F", "u", 10304)
    broken = study.technical(run_id, outcome="done", published=True, state="usable", K=None)
    result = study.advance()
    assert_route_pending(result, broken)
    (way,) = result["step"]["ways_out"]
    assert "recovery" not in way
    assert not any("recovery round" in r for r in result["step"]["reasons"])


@pytest.mark.parametrize("fields", [{"state": "unusable", "K": 0.0},
                                    {"state": "usable", "K": 0.0}])  # fmt: skip
def test_a_search_run_the_search_counts_as_technical_is_a_mark_that_stops(fields) -> None:
    """A K-only run whose K is not positive (unusable), or a usable one without a positive K."""
    study = Study()
    study.ready()
    study.search()
    run_id = run_of(study, "search", "F", "l", 10101)
    broken = study.technical(run_id, outcome="done", published=True, **fields)
    result = study.advance()
    assert result["stage"] == "search"
    assert_route_pending(result, broken)
    study.closure(run_id)
    closure_refused(study.advance(), "F")


@pytest.mark.parametrize("fields", [{"state": "usable", "K": None},
                                    {"state": "usable", "mmlu_loss_pp": None}])  # fmt: skip
def test_a_usable_validation_run_lacking_a_number_is_a_mark_that_stops(fields) -> None:
    study = Study()
    study.through_validation()
    run_id = run_of(study, "validation", "P", "l", 10305)
    broken = study.technical(run_id, outcome="done", published=True, **fields)
    result = study.advance()
    assert result["stage"] == "validation"
    assert_route_pending(result, broken)
    study.closure(run_id)
    closure_refused(study.advance(), "P")  # P stays open


def test_a_validation_run_its_block_cannot_place_is_a_mark_that_stops() -> None:
    """A run of the block's step and seed under another key (a round): the block cannot place
    it, and it is a technical mark of its arm."""
    study = Study()
    study.through_validation()
    stray = ep("F", START["F"]["u"], 10306, step="validation", round_=1, K=7.5,
               run_id="stray-r1")  # fmt: skip
    study.add(stray)
    study.ends[stray.run_id] = {
        "path": "launch/x.jsonl",
        "line": 2,
        "line_sha256": "c" * 64,
        "outcome": "done",
        "attempt": 1,
        "run": stray.label,
    }
    result = study.advance()
    assert result["stage"] == "validation"
    assert_route_pending(result, stray)
    study.closure("stray-r1")
    closure_refused(study.advance(), "F")


def test_an_unplaceable_run_that_has_not_ended_waits_and_is_no_mark() -> None:
    study = Study()
    study.through_validation()
    study.add(ep("F", START["F"]["u"], 10306, step="validation", round_=1, state="pending",
                 run_id="stray-r1"))  # fmt: skip
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("validation", "pending")
    assert "stray-r1" in result["step"]["waiting"]


def test_a_confirmation_run_listed_as_unresolved_is_a_mark_that_stops() -> None:
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.confirmation(count=5)
    run_id = run_of(study, "confirmation", "A", "u", 10402)
    broken = study.technical(run_id, outcome="done", published=True, state="usable",
                             factual_loss_pp=None,
                             log="launch/study3-c1-confirmation.jsonl")  # fmt: skip
    result = study.advance()
    assert result["stage"] == "confirmation"
    assert_route_pending(result, broken)
    assert result["confirmation"]["may_start_next"] is False
    study.closure(run_id)
    closure_refused(study.advance(), "A")


@pytest.mark.parametrize("step", ["search", "validation"])
def test_two_counting_attempts_of_one_cell_are_a_problem_with_no_way_out(step) -> None:
    study = Study()
    study.through_validation()
    seed = 10102 if step == "search" else 10304
    found = next(e for e in study.endpoints if e.cell.step == step and e.arm == "F"
                 and e.seed == seed and math.isclose(e.lr, START["F"]["u"]))  # fmt: skip
    study.add(Endpoint(**{**found.__dict__, "run_id": "twice-a2", "attempt": 2}))
    if found.run_id in study.headers:  # a search start: its campaign names the engineering record
        study.headers["twice-a2"] = study.headers[found.run_id]
    result = study.advance()
    step_ = result["step"]
    assert (step_["kind"], step_["technical"], "ways_out" in step_) == ("blocked", False, False)
    assert any("twice-a2" in p and "one cell" in p for p in result["problems"])


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
        "line": {"kind": "held", "path": "launch/study3-c1-numerical.jsonl", "line": 3,
                 "line_sha256": "e" * 64},
        "cycle": {"path": "qualification/c1/supersede-1.json", "sha256": "9" * 64},
        "reason": "no_recorded_commit",
        "decision": "technical_feasibility_report"}  # exactly the closure schema's  # fmt: skip
    assert step["hold_reasons"] == ["no_recorded_commit"]
    study.closure(run_id)
    result = study.advance()
    assert (result["stage"], result["outcome"], result["reason"], result["step"]) == (
        "technical_feasibility_report",
        "technical_feasibility_report",
        "held_run",
        {"kind": "report", "outcome": "technical_feasibility_report", "reason": "held_run"},
    )
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
    assert result["step"]["hold_reasons"] == ["no_v0_hash"]
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


def test_a_closure_citing_another_kind_of_line_than_its_runs_mark_is_a_problem() -> None:
    """A held run is closed by its held line, never by an end line."""
    study, run_id = held_numerical("F")
    end = {"kind": "end", "path": "launch/x.jsonl", "line": 2, "line_sha256": "a" * 64}
    study.closure(run_id, line=end, cycle=None, reason="launcher_technical")
    result = study.advance()
    assert (result["step"]["kind"], result["outcome"]) == ("blocked", None)
    assert any("closure-c1-F" in p and "end line" in p for p in result["problems"])


@pytest.mark.parametrize("arm, kind", [("P", "end"), ("F", "end"), ("F", "held")])
def test_a_closure_of_a_mark_the_walk_has_not_reached_is_no_problem(arm, kind) -> None:
    """A valid closure (of a held run, or of a validation run under another commit), then a
    reader problem stops the walk at the search: the closure is not judged against a stage the
    walk did not reach; the reader's problem alone blocks."""
    study = Study()
    study.through_validation()
    if kind == "held":
        study.cycle(1)
        run_id = study.hold(run_of(study, "numerical", arm, "l", 10201))
    else:
        run_id = run_of(study, "validation", arm, "u", 10304)
        study.other_commit(run_id)
    study.closure(run_id)
    assert study.advance()["problems"] == []
    study.problems.append("run-x: refused: a seed outside its role")
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("search", "blocked")
    assert result["problems"] == ["run-x: refused: a seed outside its role"]


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


@pytest.mark.parametrize("decision, stage", [("proceed", "confirmation"),
                                             ("stop", "budget_stop")])  # fmt: skip
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
    assert (result["stage"], result["outcome"], result["reason"]) == (
        "budget_stop", "budget_stop", "budget_stop")  # fmt: skip
    assert "c1/confirmation" not in result["prescribed"]


def test_a_failed_n_record_starts_no_confirmation_and_never_cuts_n() -> None:
    """A design stop, not a technical dead end: the confirmation cannot start within the
    reserved seeds."""
    study = Study()
    study.through_validation()
    study.n(N=179)  # cap 201: beyond the 200 reserved confirmation seeds
    result = study.advance()
    assert (result["stage"], result["outcome"], result["reason"]) == (
        "design_stop", "design_stop", "cap_beyond_reserved_seeds")  # fmt: skip
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


@pytest.mark.parametrize("kind, outcome, reason", [
    ("engineering", "technical_feasibility_report", "engineering_failed"),
    ("simulation", "design_stop", "simulation_failed"),
])  # fmt: skip
def test_a_failed_engineering_or_simulation_record_ends_the_study_by_its_kind(
    kind, outcome, reason
) -> None:
    """A failed engineering record is a technical dead end; a failed simulation record (no level
    passed, or the verification failed) a design stop: the confirmation cannot start."""
    study = Study()
    study.ready()
    study.entries[kind]["state"] = study.entries[kind]["record"]["state"] = "failed"
    result = study.advance()
    assert (result["stage"], result["outcome"], result["reason"]) == (outcome, outcome, reason)
    assert result["step"] == {"kind": "report", "outcome": outcome, "reason": reason}
    assert any(kind in r for r in result["reasons"])
    assert "c1/search/r0" not in result["prescribed"]


def test_without_a_cycle_the_expected_code_is_the_engineering_records() -> None:
    study = Study()
    study.through_validation()
    study.n(N=60)
    result = study.advance(expected_code="c" * 64)
    assert (result["step"]["kind"], result["outcome"]) == ("blocked", None)
    assert any("c" * 64 in p and "chain" in p for p in result["problems"])


def test_a_simulation_record_off_the_chain_is_invalid() -> None:
    study = Study()
    study.through_validation()
    study.record("simulation", code="c" * 64, level=0.975, validation_blocks=28,
                 gates=study.entries["simulation"]["record"]["gates"])  # fmt: skip
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("simulation", "blocked")
    assert any("simulation" in p and "c" * 64 in p for p in result["problems"])


def test_one_cycle_moves_the_code_the_later_records_carry() -> None:
    """The engineering record keeps its code (the chain's start) and stays valid; the n record
    carries the code in force for its candidate's runs, the fix."""
    study = Study()
    study.through_validation()
    fix = study.cycle(1)
    study.n(N=60)
    result = study.advance()
    assert (result["stage"], result["problems"]) == ("confirmation", [])
    assert study.entries["engineering"]["record"]["code"]["gpu_code_sha256"] == EXPECTED != fix
    assert study.entries["n-c1"]["record"]["code"]["gpu_code_sha256"] == fix


def test_an_n_record_with_the_code_before_its_candidates_cycle_is_invalid() -> None:
    study = Study()
    study.through_validation()
    study.cycle(1)
    study.n(N=60)
    study.entries["n-c1"]["record"]["code"]["gpu_code_sha256"] = EXPECTED
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("n", "blocked")
    assert any("n-c1" in p and EXPECTED in p for p in result["problems"])


def test_a_cycle_at_each_candidate_chains_the_codes() -> None:
    """Candidate 1's cycle comes first; candidate 2's n record carries the code after its own
    cycle; the simulation record may carry any code on the chain."""
    study = Study()
    study.ready()
    first = study.cycle(1)
    fail_at(study, "search", "F", 1)
    study.search(2)
    study.numerical(2)
    study.validation(2)
    second = study.cycle(2)
    assert first != second
    study.n(2, N=60)
    result = study.advance()
    assert (result["stage"], result["candidate"], result["problems"]) == ("confirmation", 2, [])
    study.entries["n-c2"]["record"]["code"]["gpu_code_sha256"] = first
    assert study.advance()["stage"] == "n"


def test_a_closure_may_carry_any_code_on_the_chain() -> None:
    """A closure is a record about a run, bound to its line's hash: one written under the code
    of an earlier time stays valid when a later cycle has moved the chain."""
    study = Study()
    study.ready()
    study.search()
    study.numerical(states={("A", "l"): "not_qualified"})
    study.rate(1, "A", START["A"]["l"], "not_qualified", technical=True)
    assert study.advance()["step"]["technical"] is True  # its engineering cycle in progress
    study.cycle(1)  # the cycle moves the chain; the case re-qualifies under the fix
    study.rate(1, "A", START["A"]["l"])
    study.readings[1]["cases"] = []
    study.validation()
    run_id = run_of(study, "validation", "P", "u", 10301)
    study.other_commit(run_id)
    study.closure(run_id, code=EXPECTED)  # under the chain's start, before the cycle
    result = study.advance()
    assert (result["stage"], result["problems"]) == ("choice", [])
    assert result["active"]["comparisons"] == ["A-F"]
    assert study.entries["closure-c1-P"]["record"]["code"]["gpu_code_sha256"] == EXPECTED
    assert study.expected != EXPECTED


def test_a_closure_off_the_chain_is_invalid() -> None:
    study, run_id = held_numerical("F")
    study.closure(run_id, code="7" * 64)
    result = study.advance()
    assert (result["step"]["kind"], result["outcome"]) == ("blocked", None)
    assert any("closure-c1-F" in p and "7" * 64 in p for p in result["problems"])


def test_a_break_in_the_chain_of_cycles_is_a_problem() -> None:
    study = Study()
    study.through_validation()
    study.cycle(1, old="7" * 64)  # its old code is not the engineering record's
    result = study.advance()
    assert (result["step"]["kind"], result["outcome"]) == ("blocked", None)
    assert any("7" * 64 in p and "chain" in p for p in result["problems"])


def test_a_second_cycle_record_for_a_candidate_is_a_break_in_the_chain() -> None:
    """One engineering cycle per candidate: a second supersession record of candidate 1, even
    one whose old code is the first one's fix, is a break, and blocks."""
    study = Study()
    study.through_validation()
    study.cycle(1)
    study.cycle(1, fix="8" * 64)
    study.cycles[1][1]["path"] = "/runs/qualification/c1/supersede-2.json"
    result = study.advance()
    assert (result["step"]["kind"], result["outcome"]) == ("blocked", None)
    assert any("candidate 1 has 2 supersession records" in p and "supersede-2.json" in p
               and "chain" in p for p in result["problems"]), result["problems"]  # fmt: skip
    _, _, breaks = study3_progress.code_chain(EXPECTED, study.cycles)
    assert len(breaks) == 1


def test_an_expected_code_that_is_not_the_chains_end_is_a_problem() -> None:
    study = Study()
    study.through_validation()
    study.cycle(1)
    result = study.advance(expected_code=EXPECTED)  # the code before the cycle
    assert result["step"]["kind"] == "blocked"
    assert any(EXPECTED in p and "chain" in p for p in result["problems"])


def cohort_problem(result) -> str:
    """The n record's problem with its cohort's run ids."""
    (found,) = [p for p in result["problems"] if "cohort's run ids" in p]
    return found


def test_the_n_records_cohort_run_ids_are_counting_validation_runs_of_its_candidate() -> None:
    study = Study()
    study.through_validation()
    record = study.n(N=60)
    record["cohort"]["run_ids"] = [*record["cohort"]["run_ids"][:-1], "not-a-validation-run"]
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("n", "blocked")
    assert "1 missing, 1 extra, 0 duplicated" in cohort_problem(result)


def test_probe_an_n_record_whose_cohort_holds_p_runs_for_f_runs_is_invalid() -> None:
    """A probe: every F run id of the cohort replaced by the P run id of its block and label. Each
    is a counting validation run at a listed seed, but the cohort is exactly the A and F
    endpoints of the supported blocks: the n record is invalid, and the confirmation does not
    start."""
    study = Study()
    study.through_validation()
    record = study.n(N=60)
    swap = {}
    for e in study.endpoints:
        if e.cell.step == "validation" and e.arm == "F":
            label = "l" if math.isclose(e.lr, START["F"]["l"]) else "u"
            swap[e.run_id] = run_of(study, "validation", "P", label, e.seed)
    record["cohort"]["run_ids"] = [swap.get(r, r) for r in record["cohort"]["run_ids"]]
    result = study.advance()
    assert (result["stage"], result["step"]["kind"], result["outcome"]) == ("n", "blocked", None)
    assert "confirmation" not in result
    assert cohort_problem(result) == (
        "n-c1: its cohort's run ids are not exactly the A and F endpoints (A and F at lr_l and "
        "lr_u) of its 28 supported validation blocks, 112 runs: 56 missing, 56 extra, 0 "
        "duplicated")  # fmt: skip


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_the_n_records_cohort_is_exactly_the_supported_blocks_a_and_f_runs(change) -> None:
    """One run id missing, one extra (an A run of a block A–F does not support), one twice: each
    makes the n record invalid, its problem naming the counts; the exact set, in any order, is
    accepted."""
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    seeds = SEEDS["validation"][1][:28]
    low = CANDIDATES[1]["targets"]["star"] * 0.9  # A's upper run below K*: no bracket
    study.validation(special={(seeds[-1], "A", "u"): {"K": low}})
    record = study.n(N=60, seeds=seeds[:-1])
    ids = record["cohort"]["run_ids"]
    assert len(ids) == 4 * 27
    record["cohort"]["run_ids"] = list(reversed(ids))
    assert study.advance()["stage"] == "confirmation"  # the exact set, in another order
    if change == "missing":
        record["cohort"]["run_ids"] = ids[1:]
        counts = "1 missing, 0 extra, 0 duplicated"
    elif change == "extra":
        record["cohort"]["run_ids"] = [*ids, run_of(study, "validation", "A", "l", seeds[-1])]
        counts = "0 missing, 1 extra, 0 duplicated"
    else:
        record["cohort"]["run_ids"] = [*ids, ids[0]]
        counts = "0 missing, 0 extra, 1 duplicated"
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("n", "blocked")
    assert cohort_problem(result).endswith(f"27 supported validation blocks, 108 runs: {counts}")


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


@pytest.mark.parametrize(
    "header",
    [
        {},
        {"engineering_record_sha256": "2" * 64},
        {"engineering_record": "study3/decisions/simulation.json"},
    ],
)
def test_a_search_start_whose_header_does_not_name_the_engineering_record_is_a_problem(header):
    """The search campaign's header names the engineering record by its path and its sha256 now:
    that shows the record existed before the search started."""
    study = Study()
    study.ready()
    study.search()
    early = run_of(study, "search", "F", "u", 10102)
    study.headers[early] = {"study": 3, "candidate": 1, "step": "search", "round": 0,
                            **({} if not header else {**NAMED, **header})}  # fmt: skip
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("search", "blocked")
    assert any(early in p and "engineering record" in p for p in result["problems"])
    study.headers[early] = {"study": 3, "candidate": 1, "step": "search", "round": 0, **NAMED}
    assert study.advance()["stage"] == "numerical"


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
    """A run of a started block that has not ended holds the counters at its block, and no block
    starts: the registered stop is checked between blocks, and while a run has not ended the
    counters are unknown."""
    seeds = list(SEEDS["confirmation"][1])
    endpoints = confirmation_endpoints(10, special={(seeds[3], "F", "l"): {"state": "pending"}})
    state = confirmation_state(endpoints, 1, 60, ["A-F", "P-F"])
    assert state["counters"] == {"A-F": 3, "P-F": 3}
    assert state["held_up"] == {"A-F": seeds[3], "P-F": seeds[3]}
    assert state["supported"]["A-F"] == 9
    assert state["may_start_next"] is False and state["next_seed"] is None
    (waiting,) = state["open"]
    assert waiting["seed"] == seeds[3] and len(waiting["runs"]) == 1
    # a run whose recovery round is opened and has no later attempt holds them too
    endpoints = confirmation_endpoints(10, special={(seeds[3], "F", "l"): {"state": "recovered"}})
    state = confirmation_state(endpoints, 1, 60, ["A-F", "P-F"])
    assert state["held_up"] == {"A-F": seeds[3], "P-F": seeds[3]}
    assert state["may_start_next"] is False and state["next_seed"] is None
    # a block whose cells are missing, and no run of it unfinished, is completed by the route:
    # it does not hold the next block by itself
    endpoints = confirmation_endpoints(10)
    endpoints = [e for e in endpoints if not (e.seed == seeds[3] and e.arm == "F" and e.lr ==
                                              START["F"]["l"])]  # fmt: skip
    state = confirmation_state(endpoints, 1, 60, ["A-F", "P-F"])
    assert state["held_up"] == {"A-F": seeds[3], "P-F": seeds[3]}
    assert state["open"] == [{"seed": seeds[3], "runs": []}]
    assert state["may_start_next"] is True and state["next_seed"] == seeds[10]
    # a technical block holds them too, until its diagnosis
    endpoints = confirmation_endpoints(10, special={(seeds[3], "P", "l"): {"state": "technical"}})
    state = confirmation_state(endpoints, 1, 60, ["A-F", "P-F"])
    assert state["counters"] == {"A-F": 10, "P-F": 3}
    assert [t["seed"] for t in state["technical"]] == [seeds[3]]
    assert state["may_start_next"] is False  # a technical mark waits for its way out


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
    # the started block's run has not ended: the next block, seeds[4], waits for it
    assert state["open"] == [{"seed": seeds[3], "runs": [endpoints[-1].run_id]}]
    assert state["may_start_next"] is False and state["next_seed"] is None
    endpoints[-1] = ep("A", START["A"]["l"], seeds[3], step="confirmation", K=6.0)  # it ended
    assert confirmation_state(endpoints, 1, 60, ["A-F", "P-F"])["next_seed"] == seeds[4]


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
    assert (result["stage"], result["candidate"], result["outcome"], result["reason"]) == (
        "done", 1, "descriptive", "cap_reached")  # fmt: skip
    assert result["readings"] == {"A-F": "descriptive", "P-F": "descriptive"}
    assert result["active"]["arms"] == ["A", "F", "P"]
    assert not any(e["event"] in ("candidate", "close_pf") for e in result["route"])


def test_a_technical_confirmation_run_stops_the_confirmation() -> None:
    """A technical mark in the confirmation is no failed endpoint: the step stops for the
    researcher's decision, no block starts, and a closure routes nothing."""
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.confirmation(count=10)
    broken = study.technical(run_of(study, "confirmation", "F", "l", 10403),
                             log="launch/study3-c1-confirmation.jsonl")  # fmt: skip
    result = study.advance()
    assert result["stage"] == "confirmation"
    assert_route_pending(result, broken)
    assert result["confirmation"]["may_start_next"] is False
    study.closure(broken.run_id)
    closure_refused(study.advance(), "F")


def test_a_recovered_confirmation_cell_waits_for_its_next_attempt_and_is_not_issued_again():
    """A run whose recovery round is opened and that has no later attempt yet: its cell is not
    issued again, and no new block starts until its next attempt has ended."""
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.confirmation(count=3)
    recovered = study.change(run_of(study, "confirmation", "A", "u", 10401), state="recovered",
                             K=None)  # fmt: skip
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("confirmation", "pending")
    assert result["step"]["waiting"] == [recovered.run_id]
    assert "cells" not in result["step"]
    assert result["confirmation"]["held_up"]["A-F"] == 10401
    assert result["confirmation"]["may_start_next"] is False


def test_a_started_block_with_a_run_not_ended_holds_the_confirmation() -> None:
    """A run of a started block that the launcher never ended (a Pod stopped mid-run): the step
    waits, naming the block, the run and its way out (its recovery round, from its own block
    file); it names no new block, and the next one is issued once the run has ended."""
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.confirmation(count=5)
    dead = study.change(run_of(study, "confirmation", "F", "l", 10402), state="pending", K=None,
                        steps=None, mmlu_loss_pp=None, factual_loss_pp=None,
                        K_harmful=None)  # fmt: skip
    result = study.advance()
    step = result["step"]
    assert (result["stage"], step["kind"]) == ("confirmation", "pending")
    assert step["waiting"] == [dead.run_id]
    assert step["blocks"] == [{"seed": 10402, "runs": [dead.run_id]}]
    (way,) = step["ways_out"]
    assert (way["run_id"], way["label"], way["campaign"]) == (dead.run_id, dead.label,
                                                              dead.cell.campaign)  # fmt: skip
    assert (f"launch.py <block file> --only {dead.label} --recovery REASON --max-attempts 1"
            in way["recovery"])  # fmt: skip
    assert "study3_summary.py check" in way["recovery"]  # before every recovery round
    assert "cells" not in step and "campaign" not in step
    assert result["confirmation"]["may_start_next"] is False
    assert result["problems"] == []
    study.change(dead.run_id, state="usable", K=dead.K, steps=25,
                 mmlu_loss_pp=dead.mmlu_loss_pp, factual_loss_pp=dead.factual_loss_pp,
                 K_harmful=dead.K_harmful)  # fmt: skip
    result = study.advance()
    assert {c[2] for c in result["step"]["cells"]} == {10405}


def test_a_started_blocks_missing_cells_are_issued_before_the_wait_for_a_run() -> None:
    """A started block holds a run that has not ended and another lacks a cell: the missing cell
    is issued first, as a started block's cells always are; then the step waits."""
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.confirmation(count=5)
    dead = study.change(run_of(study, "confirmation", "F", "l", 10402), state="pending",
                        K=None)  # fmt: skip
    study.drop(run_of(study, "confirmation", "P", "u", 10403))
    result = study.advance()
    assert (result["step"]["kind"], result["step"]["campaign"]) == ("campaign", "confirmation")
    assert cells(result["step"]) == {("P", START["P"]["u"], 10403)}
    study.add(ep("P", START["P"]["u"], 10403, step="confirmation",
                 K=CANDIDATES[1]["targets"]["u"]))  # the missing cell ran  # fmt: skip
    result = study.advance()
    assert (result["step"]["kind"], result["step"]["waiting"]) == ("pending", [dead.run_id])


def test_a_partly_issued_confirmation_block_is_completed_before_the_next() -> None:
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.confirmation(count=3)
    study.drop(run_of(study, "confirmation", "P", "u", 10401))
    result = study.advance()
    assert cells(result["step"]) == {("P", START["P"]["u"], 10401)}


# ------------------------------------------------------------------ a run under another commit


def other_commit_closure(study, run, entry) -> dict:
    """The closure record of a run under another commit, as the step names it: its last end
    line (or its start line), its uncounted reason id, no cycle."""
    (found,) = [e for e in study.endpoints if e.run_id == run]
    if run in study.ends:
        end = study.ends[run]
        line = {"kind": "end", **{k: end[k] for k in ("path", "line", "line_sha256")}}
    else:
        start = study.starts[run]
        line = {"kind": "start", **{k: start[k] for k in ("path", "line", "line_sha256")}}
    return {"record": f"closure-c1-{found.arm}",
            "path": f"study3/decisions/closure-c1-{found.arm}.json",
            "fields": {"candidate": 1, "arm": found.arm, "rate": found.lr, "seed": found.seed,
                       "run_id": run, "line": line, "cycle": None, "reason": entry["reason"],
                       "decision": "close_pf" if found.arm == "P"
                       else "technical_feasibility_report"}}  # fmt: skip


@pytest.mark.parametrize("reason", ["other_commit", "dirty_checkout", "no_commit"])
def test_a_validation_run_under_another_commit_does_not_count_and_blocks(reason) -> None:
    """The registration's rule (Versions, in "Voiding and supersession"): the run does not count,
    is kept and listed, and the step is blocked and technical, naming it, its reason id and its
    commit: a technical mark outside the qualification of its arm, whose way out is its closure
    record (a completed run is never rerun: the launcher skips it), with the fields the closure
    schema takes. Its cell is not issued again."""
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    study.validation(count=3)
    run = run_of(study, "validation", "F", "u", 10301)
    entry = study.other_commit(run, reason, None if reason == "no_commit" else "e" * 40)
    result = study.advance()
    step = result["step"]
    assert (result["stage"], step["kind"], step["technical"], result["outcome"]) == (
        "validation", "blocked", True, None)  # fmt: skip
    (why,) = step["reasons"]
    assert run in why and reason in why and str(entry["commit"]) in why
    assert step["ways_out"] == [{"run_id": run, "label": entry["label"], "reason": reason,
                                 "commit": entry["commit"],
                                 "closure": other_commit_closure(study, run, entry)}]  # fmt: skip
    assert "supersession_needed" not in tidy(result)
    assert result["not_counted"] == [entry]
    assert "cells" not in step
    # the lenient derivation of the cells is not stopped by it
    lenient = study.advance(lenient=True)
    assert len(lenient["prescribed"]["c1/validation"]) == 6 * 28


@pytest.mark.parametrize("arm", ["A", "F", "P"])
def test_probe_a_completed_run_under_another_commit_is_closed_by_its_closure(arm) -> None:
    """A probe: a completed validation run under another commit, then its closure record. The
    run stays kept, excluded and listed; the closure decides as any technical mark's ("After a
    technical mark"): A or F, the technical feasibility report with the run's reason id; P alone,
    P–F closes and A–F goes on, the run adding to no counter."""
    study = Study()
    study.through_validation()
    run = run_of(study, "validation", arm, "u", 10301)
    entry = study.other_commit(run, "other_commit")
    assert study.advance()["step"]["kind"] == "blocked"
    study.closure(run)
    result = study.advance()
    assert result["not_counted"] == [entry] and result["problems"] == []
    if arm == "P":
        assert (result["stage"], result["active"]["comparisons"]) == ("choice", ["A-F"])
        (close,) = [e for e in result["route"] if e["event"] == "close_pf"]
        assert (close["stage"], close["why"]) == ("validation", "closure-c1-P")
        assert set(result["validation"]["gates"]) == {"AF"}
        assert run not in result["step"]["fields"]["cohort"]["run_ids"]
    else:
        assert (result["stage"], result["outcome"], result["reason"]) == (
            "technical_feasibility_report",
            "technical_feasibility_report",
            "other_commit",
        )
        assert any(run in r and "closure-c1-" + arm in r for r in result["reasons"])
        assert not any(key.startswith("c2/") for key in result["prescribed"])


def test_a_closure_of_a_run_under_another_commit_must_name_its_reason() -> None:
    """The closure of a run under another commit carries the run's uncounted reason id: one
    with the end line's reason, or another uncounted reason, closes nothing and blocks."""
    study = Study()
    study.through_validation()
    run = run_of(study, "validation", "F", "u", 10301)
    study.other_commit(run, "dirty_checkout")
    for reason in ("launcher_done", "other_commit"):
        study.closure(run, reason=reason)
        result = study.advance()
        assert (result["step"]["kind"], result["outcome"]) == ("blocked", None)
        assert any("closure-c1-F" in p and run in p for p in result["problems"])


@pytest.mark.parametrize("reason", ["other_commit", "dirty_checkout", "no_commit"])
def test_a_confirmation_run_under_another_commit_does_not_count_and_blocks(reason) -> None:
    """The same in the confirmation: no block is issued, the step is blocked and technical, and
    the guard, which reads the step, refuses every block; P's closure closes P–F and the blocks
    go on with A and F."""
    study = Study()
    study.through_validation()
    study.n(N=60)
    study.confirmation(count=3)
    run = run_of(study, "confirmation", "P", "l", 10401)
    entry = study.other_commit(run, reason, None if reason == "no_commit" else "e" * 40)
    result = study.advance()
    step = result["step"]
    assert (result["stage"], step["kind"], step["technical"]) == ("confirmation", "blocked", True)
    assert [(w["run_id"], w["reason"], w["closure"]["fields"]["reason"])
            for w in step["ways_out"]] == [(run, reason, reason)]  # fmt: skip
    assert result["not_counted"] == [entry]
    assert "cells" not in step
    study.closure(run)
    result = study.advance()
    assert (result["stage"], result["active"]["comparisons"]) == ("confirmation", ["A-F"])
    assert result["step"]["campaign"] == "confirmation"
    assert {(c[0], c[2]) for c in result["step"]["cells"]} == {
        (arm, 10403) for arm in ("A", "F")}  # fmt: skip
    assert result["not_counted"] == [entry]


@pytest.mark.parametrize("arm", ["A", "F", "P"])
def test_a_run_under_another_commit_the_launcher_never_ended_is_closed_by_its_start_line(arm):
    """A run under another commit whose launcher stopped without ending it: its closure cites its
    start line, with its uncounted reason, and reaches the outcome of any technical mark: A or F,
    the technical feasibility report; P alone, P–F closes and A–F goes on."""
    study = Study()
    study.through_validation()
    run = study.change(run_of(study, "validation", arm, "u", 10301), state="technical", K=None,
                       steps=None)  # fmt: skip
    study.starts[run.run_id] = {"path": "launch/study3-c1-validation.jsonl", "line": 5,
                                "line_sha256": "b" * 64, "attempt": 1,
                                "run": run.label}  # fmt: skip
    entry = study.other_commit(run.run_id, "unknown_commit", ended=False)
    (way,) = study.advance()["step"]["ways_out"]
    assert (way["closure"]["fields"]["line"]["kind"], way["closure"]["fields"]["reason"]) == (
        "start", "unknown_commit")  # fmt: skip
    study.closure(run.run_id)
    result = study.advance()
    assert result["not_counted"] == [entry] and result["problems"] == []
    if arm == "P":
        assert (result["stage"], result["active"]["comparisons"]) == ("choice", ["A-F"])
    else:
        assert (result["outcome"], result["reason"]) == (
            "technical_feasibility_report", "unknown_commit")  # fmt: skip


def test_a_run_under_another_commit_with_no_journal_line_known_has_no_way_out() -> None:
    """With neither an end nor a start line known, the records give no way out: the step names
    the run, with no closure and no recovery round."""
    study = Study()
    study.through_validation()
    run = run_of(study, "validation", "F", "u", 10304)
    study.other_commit(run, ended=False)
    step = study.advance()["step"]
    assert (step["kind"], step["technical"]) == ("blocked", True)
    (way,) = step["ways_out"]
    assert (way["run_id"], "closure" in way, "recovery" in way) == (run, False, False)
    assert any(run in r and "no way out the records give" in r for r in step["reasons"])


@pytest.mark.parametrize("reason, closed", [("launcher_done", False), ("dirty_checkout", False),
                                            ("other_commit", True)])  # fmt: skip
def test_the_closure_of_a_run_under_another_commit_carries_its_own_reason(reason, closed):
    """Whatever the line the records know, the closure of a run under another commit carries
    the run's own uncounted reason id: any other is a problem that decides nothing."""
    study = Study()
    study.through_validation()
    run = run_of(study, "validation", "F", "u", 10304)
    study.other_commit(run, "other_commit", ended=False)  # no journal line known
    (found,) = [e for e in study.endpoints if e.run_id == run]
    line = {"kind": "end", "path": "launch/x.jsonl", "line": 2, "line_sha256": "c" * 64}
    study.record("closure-c1-F", "failed", candidate=1, arm="F", rate=found.lr, seed=found.seed,
                 run_id=run, line=line, cycle=None, reason=reason,
                 decision="technical_feasibility_report")  # fmt: skip
    result = study.advance()
    if closed:
        assert (result["outcome"], result["reason"]) == ("technical_feasibility_report",
                                                          "other_commit")  # fmt: skip
    else:
        assert (result["step"]["kind"], result["outcome"]) == ("blocked", None)
        assert any("closure-c1-F" in p and run in p for p in result["problems"])


def test_a_pending_run_under_another_commit_names_its_recovery_round_too() -> None:
    """A run with no published outcome under another commit, which the launcher would recover:
    beside its closure (its start line), a recovery round of its label, launched after
    ``study3_summary.py check`` passes, reruns it (its next attempt decides)."""
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    study.validation(count=3)
    pending = study.change(run_of(study, "validation", "A", "l", 10302), state="pending", K=None,
                           steps=None)  # fmt: skip
    study.recoverable.add(pending.run_id)
    study.starts[pending.run_id] = {"path": "launch/study3-c1-validation.jsonl", "line": 5,
                                    "line_sha256": "b" * 64, "attempt": 1,
                                    "run": pending.label}  # fmt: skip
    entry = study.other_commit(pending.run_id, ended=False)
    result = study.advance()
    assert (result["step"]["kind"], result["step"]["technical"]) == ("blocked", True)
    (way,) = result["step"]["ways_out"]
    assert pending.label in way["recovery"] and "study3_summary.py check" in way["recovery"]
    assert way["closure"] == other_commit_closure(study, pending.run_id, entry)
    assert way["closure"]["fields"]["line"]["kind"] == "start"
    assert result["not_counted"] == [entry]


def test_a_technical_run_under_another_commit_names_its_recovery_round_too() -> None:
    """A technical run with no published outcome under another commit: beside its closure, a
    recovery round reruns it where the launcher would open one (its next attempt decides)."""
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    study.validation(count=3)
    broken = study.technical(run_of(study, "validation", "A", "l", 10302))
    study.other_commit(broken.run_id)
    (way,) = study.advance()["step"]["ways_out"]
    assert broken.label in way["recovery"] and way["closure"]["fields"]["reason"] == "other_commit"


def test_a_run_under_another_commit_is_listed_wherever_the_walk_stops() -> None:
    """The state lists every run that does not count for its commit, even where a problem stops
    the walk before the run's stage; without one, the validation goes on."""
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    study.validation(count=3)
    entry = study.other_commit(run_of(study, "validation", "F", "u", 10301))
    study.problems.append("a problem of the records")
    result = study.advance()
    assert (result["stage"], result["step"]["kind"]) == ("search", "blocked")
    assert result["not_counted"] == [entry]
    study.problems.clear()
    study.uncounted.clear()
    result = study.advance()
    assert result["not_counted"] == [] and result["step"]["campaign"] == "validation"


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


def pinned(records, runs) -> dict:
    """The checkout made a Git checkout with a versions record that pins the commit before its
    own; the runs written from now on ran its own commit, clean. The header of a validation
    campaign naming the record."""
    repo = records.repo

    def git(*args: str) -> str:
        command = ["git", "-c", "user.name=test", "-c", "user.email=test@example.org",
                   "-c", "commit.gpgsign=false", *args]  # fmt: skip
        return subprocess.run(command, cwd=repo, check=True, capture_output=True,
                              text=True).stdout.strip()  # fmt: skip

    git("init", "-q")
    git("add", "-A")
    git("commit", "-q", "--allow-empty", "-m", "the checkout")
    record = repo / "docs" / "study3-versions.json"
    record.parent.mkdir(exist_ok=True)
    record.write_text(json.dumps({"code": {"commit": git("rev-parse", "HEAD")}}) + "\n")
    git("add", "docs/study3-versions.json")
    git("commit", "-q", "-m", "the versions record")
    runs.git = {"commit": git("rev-parse", "HEAD"), "dirty": False}
    digest = hashlib.sha256(record.read_bytes()).hexdigest()
    return header(1, "validation", None, versions_record_sha256=digest)


def read(records, runs, expected_code=EXPECTED, **kw) -> dict:
    return progress(runs.root, repo=records.repo, expected_code=expected_code,
                    theta0=runs.theta0, **kw)  # fmt: skip


def named(records, round_=0, candidate=1) -> dict:
    """A search campaign's header naming the engineering record by its path and its sha256."""
    return {"study": 3, "candidate": candidate, "step": "search", "round": round_,
            "engineering_record": "study3/decisions/engineering.json",
            "engineering_record_sha256": records.digest("engineering")}  # fmt: skip


def search_round0(records, runs, offsets=None, candidate=1) -> None:
    spec = CANDIDATES[candidate]
    for arm in ARMS:
        for label in LABELS:
            for seed in SEEDS["search"][candidate]:
                k = spec["targets"][label] * math.exp((offsets or {}).get((arm, label), 0.0))
                runs.run(arm, spec["start"][arm][label], seed, role="search", round_=0, k=k,
                         prescribe=False, campaign=f"study3-c{candidate}-search-r0",
                         candidate=candidate, head=named(records, 0, candidate))  # fmt: skip


def numerical_runs(runs, candidate=1) -> None:
    """The launcher runs of the candidate's numerical qualification, at its starting rates."""
    spec = CANDIDATES[candidate]
    for arm in ARMS:
        for label in LABELS:
            for seed in SEEDS["numerical"][candidate]:
                runs.run(arm, spec["start"][arm][label], seed, role="numerical",
                         candidate=candidate, prescribe=False,
                         campaign=f"study3-c{candidate}-numerical",
                         k=spec["targets"][label])  # fmt: skip


def test_the_reader_reads_an_empty_study_at_its_first_step(worlds) -> None:
    records, runs = worlds
    result = read(records, runs)
    assert (result["stage"], result["step"]["record"], result["problems"]) == (
        "engineering", "engineering", [])  # fmt: skip


def test_the_reader_accepts_exactly_the_cells_the_rule_issues(worlds) -> None:
    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    result = read(records, runs)
    assert (result["stage"], result["step"]["campaign"]) == ("search", "search-round0")
    search_round0(records, runs, {("F", "u"): 0.2})
    result = read(records, runs)
    rate = adjusted_rate(START["F"]["u"], math.log(7.5) + 0.2, 7.5, 1.30)
    assert result["problems"] == []
    assert (result["step"]["campaign"], cells(result["step"])) == (
        "search-round1", {("F", rate, s) for s in SEEDS["search"][1]})  # fmt: skip
    for seed in SEEDS["search"][1]:
        runs.run("F", rate, seed, role="search", round_=1, k=7.5, prescribe=False,
                 campaign="study3-c1-search-r1", head=named(records, 1))  # fmt: skip
    result = read(records, runs)
    assert (result["stage"], result["step"]["campaign"], result["problems"]) == (
        "numerical", "numerical", [])  # fmt: skip
    assert ("F", rate, 10200) in cells(result["step"])
    # a round-1 run at a rate the rule never gave is refused, and blocks
    stray = runs.run("A", 1.5e-5, 10100, role="search", round_=1, k=4.8, prescribe=False,
                     campaign="study3-c1-search-r1", head=named(records, 1))  # fmt: skip
    result = read(records, runs)
    assert result["step"]["kind"] == "blocked"
    assert any(stray in p and "not prescribed" in p for p in result["problems"])


def test_the_decision_records_folder_is_no_run_directory(worlds) -> None:
    """``RUNS_DIR/study3/decisions/`` lies in the experiment's folder; it is no run, while any
    other folder with no start stays refused."""
    records, runs = worlds
    records.write("engineering", records.engineering())
    assert (runs.root / "study3" / "decisions").is_dir()
    result = read(records, runs)
    assert (result["stage"], result["step"]["record"], result["problems"]) == (
        "simulation", "simulation", [])  # fmt: skip
    (runs.root / "study3" / "20261026T000099Z-A-s10300-abcd1234-a1").mkdir()
    result = read(records, runs)
    assert result["step"]["kind"] == "blocked"
    assert any("20261026T000099Z" in p and "no start" in p for p in result["problems"])


def test_the_reader_reads_the_engineering_record_from_the_search_header(worlds) -> None:
    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    bare = runs.run("A", START["A"]["l"], 10100, role="search", round_=0, prescribe=False)
    result = read(records, runs)
    assert (result["stage"], result["step"]["kind"]) == ("search", "blocked")
    assert any(bare in p and "engineering record" in p for p in result["problems"])


def test_run_records_that_cannot_be_read_block_and_raise_nothing(worlds) -> None:
    """A run id longer than a file name, and a run directory the reader may not search: the
    step is blocked, "the run records cannot be read"."""
    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    long_id = "2" * 300
    runs.log("study3-c1-search-r0", {"event": "start", "run": "A-1.45e-05/s10100", "attempt": 1,
                                     "run_id": long_id, "experiment": "study3",
                                     "config_hash": "c" * 12,
                                     "header": named(records)})  # fmt: skip
    result = read(records, runs)
    assert result["step"]["kind"] == "blocked"
    assert any("the run records cannot be read (OSError)" in r for r in result["step"]["reasons"])


@pytest.mark.skipif(os.geteuid() == 0, reason="permissions bind no superuser")
def test_a_run_directory_the_reader_may_not_search_blocks(worlds) -> None:
    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    run_id = runs.run("A", START["A"]["l"], 10100, role="search", round_=0, prescribe=False,
                      campaign="study3-c1-search-r0", head=named(records))  # fmt: skip
    folder = runs.root / "study3" / run_id
    folder.chmod(0o600)  # readable listing, no search
    try:
        result = read(records, runs)
    finally:
        folder.chmod(0o755)
    assert result["step"]["kind"] == "blocked"
    assert any("the run records cannot be read (PermissionError)" in r
               for r in result["step"]["reasons"])  # fmt: skip


def technical_search_run(records, runs, *, status="failed", end="technical", recover_first=None):
    """Candidate 1's search with F's lower rate on 10101 a technical mark: an attempt the launcher
    ended ``end``, its meta.json recording ``status``, no terminal record; with
    ``recover_first``, a recovery round already spent on an earlier attempt: the run's own
    (``"own"``, under ``only``) or its cell's (``"cell"``)."""
    spec = CANDIDATES[1]
    for arm in ARMS:
        for label in LABELS:
            for seed in SEEDS["search"][1]:
                if (arm, label, seed) == ("F", "l", 10101):
                    continue
                runs.run(arm, spec["start"][arm][label], seed, role="search", round_=0,
                         k=spec["targets"][label], prescribe=False,
                         campaign="study3-c1-search-r0", head=named(records))  # fmt: skip
    rate, attempt = spec["start"]["F"]["l"], 1
    common = {"role": "search", "round_": 0, "prescribe": False,
              "campaign": "study3-c1-search-r0", "head": named(records)}  # fmt: skip
    if recover_first:
        label = f"F-{rate:g}/s10101"
        runs.run("F", rate, 10101, end="technical", terminal=False, status="failed", **common)
        if recover_first == "own":
            runs.recovery("study3-c1-search-r0", label)
        else:  # the cell's round, as the launcher opens it without ``only``
            runs.log(
                "study3-c1-search-r0",
                {
                    "event": "recovery",
                    "experiment": "study3",
                    "cell": label.split("/")[0],
                    "reason": "the Pod stopped",
                    "runs": [label],
                },
            )
        attempt = 2
    run_id = runs.run("F", rate, 10101, end=end, terminal=False, attempt=attempt, **common)
    meta = runs.root / "study3" / run_id / "meta.json"  # its status as the tracker left it
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "status": status}))
    return run_id


def test_a_rejected_supersession_record_blocks_the_route_at_the_search(worlds) -> None:
    """A supersession record the one check rejects (here, of another candidate) holds no search
    run, supersedes none and re-issues no cell: its problem blocks the route at the search."""
    from zero_field_refusal.analysis.study3_records import voided_runs

    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    search_round0(records, runs)
    ids = sorted(p.name for p in (runs.root / "study3").iterdir())
    held_id, superseded_id = ids[0], ids[1]
    cycle = runs.root / "qualification" / "c1" / "supersede-1.json"
    cycle.parent.mkdir(parents=True)
    cycle.write_text(json.dumps({"candidate": 2, "supersession": 1,
                                 "launchers": [{"run_id": superseded_id}],
                                 "held": [{"run_id": held_id, "experiment": "study3",
                                           "reason": "no recorded commit"}]}))  # fmt: skip
    for event, run_id in (("held", held_id), ("superseded", superseded_id)):
        runs.log("study3-cycle", {"event": event, "run_id": run_id, "experiment": "study3",
                                  "reason": "the cycle", "record": str(cycle)})  # fmt: skip
    result = read(records, runs)
    assert (result["stage"], result["step"]["kind"], result["outcome"]) == (
        "search", "blocked", None)  # fmt: skip
    assert any(p.startswith(f"{cycle}: ") and "candidate 2" in p for p in result["problems"])
    assert result["closures"] == {}
    assert voided_runs(runs.root)[0] == {}
    cycle.unlink()  # without it, nothing else blocks: the route is at the search's next step
    (runs.root / "launch" / "study3-cycle.jsonl").unlink()
    assert read(records, runs)["problems"] == []


@pytest.mark.parametrize("where", ["missing", "c01"])
def test_a_line_of_a_record_the_check_never_saw_blocks_the_route_at_the_search(worlds, where):
    """A ``held`` or ``superseded`` line citing a supersession record that is not there, or one in
    ``c01/`` (no candidate's folder), counts for nothing: each is a named problem, and the route
    blocks at the search."""
    from zero_field_refusal.analysis.study3_records import voided_runs

    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    search_round0(records, runs)
    ids = sorted(p.name for p in (runs.root / "study3").iterdir())
    folder = "c1" if where == "missing" else "c01"
    cycle = runs.root / "qualification" / folder / "supersede-1.json"
    if where == "c01":
        cycle.parent.mkdir(parents=True)
        cycle.write_text(json.dumps({"candidate": 1, "supersession": 1,
                                     "launchers": [{"run_id": ids[1]}],
                                     "held": [{"run_id": ids[0]}]}))  # fmt: skip
    for event, run_id in (("held", ids[0]), ("superseded", ids[1])):
        runs.log("study3-cycle", {"event": event, "run_id": run_id, "experiment": "study3",
                                  "reason": "the cycle", "record": str(cycle)})  # fmt: skip
    result = read(records, runs)
    assert (result["stage"], result["step"]["kind"], result["outcome"]) == (
        "search", "blocked", None)  # fmt: skip
    for run_id in ids[:2]:
        assert any(run_id in p and "counts for nothing" in p for p in result["problems"])
    assert voided_runs(runs.root)[0] == {}


@pytest.mark.parametrize("case, status, end, first, recovery", [
    ("failed, no round spent", "failed", "technical", None, True),
    ("its own round spent on an earlier attempt", "failed", "technical", "own", False),
    ("its cell's round spent on an earlier attempt", "failed", "technical", "cell", False),
    ("diverged", "diverged", "diverged", None, False),
    ("completed with no terminal record", "completed", "done", None, False),
])  # fmt: skip
def test_ways_out_list_a_recovery_round_only_where_the_launcher_opens_one(
    worlds, case, status, end, first, recovery
) -> None:
    """The launcher opens a cell's recovery round once, and never for a diverged run or a
    completed one: otherwise only the researcher's decision is listed (no closure)."""
    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    run_id = technical_search_run(records, runs, status=status, end=end, recover_first=first)
    result = read(records, runs)
    assert (result["stage"], result["step"]["kind"], result["problems"]) == (
        "search", "blocked", [])  # fmt: skip
    assert result["step"]["reason"] == ROUTE_PENDING
    (way,) = result["step"]["ways_out"]
    assert way["run_id"] == run_id and way["decision"] == {"decision": ROUTE_PENDING}
    assert "closure" not in way
    assert ("recovery" in way) is recovery, case
    assert "check" not in way.get("recovery", ""), case  # a search run's round needs none


def test_the_reader_blocks_on_a_refused_run(worlds) -> None:
    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    refused = runs.run("A", 1.45e-5, 10120, role="search", prescribe=False,
                       head=named(records))  # fmt: skip
    result = read(records, runs)
    assert result["step"]["kind"] == "blocked"
    assert any(refused in p for p in result["problems"])


def test_a_missing_theta0_is_a_problem_not_an_exception(worlds) -> None:
    records, runs = worlds
    result = progress(runs.root, repo=records.repo, expected_code=EXPECTED,
                      theta0=runs.root / "nowhere")  # fmt: skip
    assert result["step"]["kind"] == "blocked"
    assert any("θ₀" in r and "baseline" in r for r in result["step"]["reasons"])


def test_the_reader_reads_a_closure_record_from_its_files(worlds, monkeypatch) -> None:
    """A held F of the numerical qualification: the closure step names the record, the record
    written from it (with its real hold line and cycle record) ends the route. The one check of
    the cycle record (tested on the qualification's own records) accepts it here."""
    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    search_round0(records, runs)
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
        if candidate == 1:  # the cycle held the run (its fix's code is this checkout's here)
            out["supersessions"] = [
                {"path": "c1/supersede-1.json", "old_gpu_code": EXPECTED, "fix_gpu_code": EXPECTED}
            ]
        return out

    monkeypatch.setattr(qualification_records, "read_qualification", fake)
    monkeypatch.setattr(qualification_records, "cycle_checks", lambda root, **k: {
        "qualification/c1/supersede-1.json": {"candidate": 1, "accepted": True, "problems": [],
                                              "launchers": [], "held": [held_id]}})  # fmt: skip
    result = read(records, runs, settings={})
    assert result["problems"] == []
    assert (result["stage"], result["step"]["record"]) == (
        "technical_feasibility_report", "closure-c1-F")  # fmt: skip
    fields = result["step"]["fields"]
    assert {k: fields[k] for k in ("line", "cycle", "run_id", "seed", "reason")} == {
        k: closure[k] for k in ("line", "cycle", "run_id", "seed", "reason")
    }
    records.write(
        "closure-c1-F",
        {
            "kind": "closure-c1-F",
            "state": "failed",
            "written_at": "2026-10-26T13:00:00Z",
            "code": CODE,
            **fields,  # exactly what the closure schema accepts
        },
    )
    result = read(records, runs, settings={})
    assert (result["stage"], result["outcome"]) == (
        "technical_feasibility_report", "technical_feasibility_report")  # fmt: skip


def rates_reading(candidate, states=None) -> dict:
    """A qualification reading of the candidate's starting rates (qualified unless ``states``
    says: arm -> state)."""
    out = reading()
    for arm in ARMS:
        for label in LABELS:
            lr = CANDIDATES[candidate]["start"][arm][label]
            key = f"{OPT[arm]}-{lr:g}"
            state = (states or {}).get(arm, "qualified")
            out["rates"][key] = {"key": key, "arm": arm, "optimizer": OPT[arm], "lr": lr,
                                 "state": state, "runs": [], "held": [],
                                 "verdict": {"state": state, "control": None,
                                             "not_counted": []}}  # fmt: skip
    return out


def test_each_candidates_qualification_is_read_under_its_own_code(worlds, monkeypatch) -> None:
    """A cycle at candidate 2 (Y → Z): candidate 1's qualification stays read under Y, its own
    runs' code, and candidate 2's under Z; the route reaches candidate 2's numerical stage, then
    its validation."""
    from zero_field_refusal.analysis import qualification_records

    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    old, fix = CODE["gpu_code_sha256"], "9" * 64
    search_round0(records, runs)
    search_round0(records, runs, candidate=2)
    cycle = runs.root / "qualification" / "c2" / "supersede-1.json"
    cycle.parent.mkdir(parents=True)
    cycle.write_text(json.dumps({"candidate": 2, "supersession": 1, "diagnosis": "d", "fix": "f",
                                 "old_gpu_code": {"sha256": old}, "fix_gpu_code": {"sha256": fix},
                                 "held": [], "launchers": []}))  # fmt: skip
    read_under: dict[int, str] = {}

    def recording(root, candidate, *, expected_code, settings):
        read_under[candidate] = expected_code
        out = rates_reading(candidate, {"F": "failing"} if candidate == 1 else None)
        under = {1: old, 2: fix}[candidate]  # the code its records were written under
        if expected_code != under:
            out["problems"] = [
                f"c{candidate}/verdict.json: written under {under}, not {expected_code}"
            ]
        return out

    monkeypatch.setattr(qualification_records, "read_qualification", recording)
    # the one check of the cycle record (tested on the qualification's own records) accepts it
    monkeypatch.setattr(qualification_records, "cycle_checks", lambda root, **k: {
        "qualification/c2/supersede-1.json": {"candidate": 2, "accepted": True, "problems": [],
                                              "launchers": [], "held": []}})  # fmt: skip
    result = read(records, runs, expected_code=fix, settings={})
    assert read_under == {1: old, 2: fix}
    assert (result["stage"], result["candidate"], result["problems"]) == ("numerical", 2, [])
    assert result["step"]["campaign"] == "numerical"
    numerical_runs(runs, candidate=2)
    result = read(records, runs, expected_code=fix, settings={})
    assert (result["stage"], result["candidate"], result["problems"]) == ("validation", 2, [])
    assert result["step"]["campaign"] == "validation"


# one selected rate per arm, as package 1's test world names its launcher runs (by optimizer)
QUALIFIED = tuple(((OPT[arm], START[arm][label]), (10200, 10201, 10202))
                  for arm, label in (("A", "l"), ("F", "u"), ("P", "l")))  # fmt: skip


def test_the_reader_reads_a_real_qualification_through_its_engineering_cycle(
    worlds, qualify, monkeypatch, tmp_path
) -> None:
    """Candidate 1's numerical qualification as package 1's ``qualify.py`` writes it, the
    reading not replaced, one selected rate per arm (the others not yet issued): AdamW's lower
    rate technical (its cycle in progress), F's upper rate failing on a scientific case. The
    cycle moves the code; under the fix the re-qualified cases pass, and the case it kept whole
    still fails F's rate, as the script's verdict has it: F fails, and candidate 2 follows."""
    from test_qualification import CODE as RECORDS_CODE
    from test_qualification_records import World as QualificationWorld
    from test_qualification_records import beyond, qcase

    from zero_field_refusal import code_identity

    # the GPU codes as sha256 a decision record can carry: the records' (the launcher runs'
    # commit abc123) and the fix's (this checkout's after the cycle)
    old = {**RECORDS_CODE, "sha256": "e" * 64}
    fix = {**RECORDS_CODE, "commit": "f" * 40, "sha256": "f" * 64}

    def gpu_code(commit="HEAD", *args, **kwargs):
        return dict(fix if commit == "HEAD" else old if commit == "abc123" else
                    {**old, "sha256": "3" * 64})  # fmt: skip

    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(old))
    records, runs = worlds
    code = {"commit": "a" * 40, "gpu_code_sha256": old["sha256"]}
    records.write("engineering", records.engineering(code=code))
    records.write("simulation", records.simulation(code=code))
    search_round0(records, runs)
    world = QualificationWorld(tmp_path, QUALIFIED)
    kept = qcase(10201, "adam_frozen", START["F"]["u"])
    broken = qcase(10200, "adamw", START["A"]["l"])
    world.full(QUALIFIED, skip=(broken,), changes={kept: beyond("rest", "baseline~s3")},
               code=old)  # a scientific case  # fmt: skip
    world.qualified(broken, world.launch(broken), code=old,
                    change=beyond("baseline", "launcher~baseline"))  # technical  # fmt: skip
    cached = world.verdict(qualify)
    result = read(records, runs, expected_code=old["sha256"])
    assert result["problems"] == []
    assert (result["stage"], result["step"]["kind"], result["step"]["technical"]) == (
        "numerical", "blocked", True)  # fmt: skip
    a_key, f_key = f"adamw-{START['A']['l']:g}", f"adam_frozen-{START['F']['u']:g}"
    assert (cached["states"][a_key], cached["states"][f_key]) == ("not_qualified", "failing")
    assert any(a_key in r for r in result["step"]["reasons"])
    monkeypatch.setattr(code_identity, "gpu_code_identity", gpu_code)  # this checkout: the fix
    world.command(qualify, "supersede", "--case", broken.name, "--record",
                  str(world.folder(broken) / "baseline-attempt-1.json"),
                  "--diagnosis", "d", "--fix", "f")  # fmt: skip
    world.full(QUALIFIED, skip=(kept,), attempt=2, code=fix)  # the others re-qualify under the fix
    cached = world.verdict(qualify)
    assert cached["states"][f_key] == "failing" and cached["states"][a_key] == "qualified"
    result = read(records, runs, expected_code=fix["sha256"])
    assert result["problems"] == []
    (move,) = [e for e in result["route"] if e["event"] == "candidate"]
    assert (move["candidate"], move["stage"], move["arms"]) == (1, "numerical", ["F"])
    assert f"{f_key}: failing" in move["why"]
    assert (result["stage"], result["candidate"]) == ("search", 2)
    stale = read(records, runs, expected_code=old["sha256"])  # not the chain's end
    assert stale["step"]["kind"] == "blocked"
    assert any("chain" in p for p in stale["problems"])


def test_the_reader_reads_each_run_file_once(worlds, monkeypatch) -> None:
    """The between-block guard calls the reader before every block: it derives the cells and
    reads the run records under them from one snapshot."""
    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    search_round0(records, runs, {("F", "u"): 0.2})
    opened: dict[str, int] = {}
    real_open = builtins.open

    def counting(file, *args, **kwargs):
        if f"{os.sep}study3{os.sep}" in str(file) and "decisions" not in str(file):
            opened[str(file)] = opened.get(str(file), 0) + 1
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting)
    monkeypatch.setattr(io, "open", counting)
    result = read(records, runs)
    assert result["step"]["campaign"] == "search-round1"
    assert opened and max(opened.values()) == 1, {k: v for k, v in opened.items() if v > 1}


def test_the_reader_never_opens_responses_or_observations(worlds, monkeypatch) -> None:
    """A study through three validation blocks of regular runs, each with its sentinel response
    and observation files: none is opened, and the sentinel is nowhere in the result."""
    records, runs = worlds
    records.write("engineering", records.engineering())
    records.write("simulation", records.simulation())
    search_round0(records, runs)
    head = pinned(records, runs)
    for arm in ARMS:
        for label in LABELS:
            for seed in SEEDS["numerical"][1]:
                runs.run(arm, START[arm][label], seed, role="numerical", prescribe=False,
                         campaign="study3-c1-numerical",
                         k=CANDIDATES[1]["targets"][label])  # fmt: skip
            for seed in SEEDS["validation"][1][:3]:
                runs.run(arm, START[arm][label], seed, role="validation", prescribe=False,
                         campaign="study3-c1-validation", k=CANDIDATES[1]["targets"][label],
                         mmlu=1.0, factual=2.0, head=head)  # fmt: skip
    # a conflict end whose free text is the sentinel
    conflict = runs.run("A", START["A"]["l"], 10303, role="validation", prescribe=False,
                        campaign="study3-c1-validation", end=None, head=head)  # fmt: skip
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
    assert (result["stage"], result["problems"], result["not_counted"]) == ("validation", [], [])
    assert result["step"]["kind"] == "blocked"  # the conflict is technical: its diagnosis first
    assert result["step"]["technical"] is True
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
