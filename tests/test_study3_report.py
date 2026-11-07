"""Study 3's final report (analysis/study3_report.py and ``scripts/study3_analysis.py report``):
the facts and their tables for every terminal outcome, the ``judge_technical`` variant, the
researcher's billing and embargo files, and the incidents of the records, each named by exactly
one billed incident, at its stage and its one price.

The states are the progress tests' synthetic studies (``test_study3_progress.Study``): ``done``
(60 blocks), the scientific report (candidate 2 failing at validation), the technical report (a
failed engineering record), the design stop (a failed simulation record) and the budget stop (a
budget record of stop). At ``done`` the analysis and validation's harm are computed from the
freeze with harm counts by arm and seed, and the audit's estimate is a record of its shape. The
command runs in process on the analysis script's world (``test_study3_analysis_script.World``),
the label readers raisers and the sealed-path guard armed."""

import dataclasses
import functools
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from test_code_identity import gpu_uses
from test_study3_analysis_script import BASE, STEP
from test_study3_analysis_script import World as AnalysisWorld
from test_study3_analysis_script import raisers as harm_raisers
from test_study3_decisions import coverage_row, default_gates, verified
from test_study3_freeze import done as done_study
from test_study3_freeze import frozen, k_harmful, scaled
from test_study3_package2 import SENTINEL, SealedGuard
from test_study3_progress import CODE, WHEN, Study, fail_at

from zero_field_refusal.analysis import study3_estimates, study3_final
from zero_field_refusal.analysis import study3_report as rp
from zero_field_refusal.analysis.study3_blocks import CONFIRMATION_ALPHA, block_state, utility_gate
from zero_field_refusal.analysis.study3_labels import Diagnosis
from zero_field_refusal.analysis.study3_progress import COMPARISONS
from zero_field_refusal.paths import REPO_ROOT

INCIDENT = {
    "incident_time": "2026-11-02T03:12:00Z",
    "cause": "the Pod stopped: host maintenance",
    "evidence": "incidents/pod-events-20261102.log",
}
LOG = "launch/study3-c1-confirmation.jsonl"
INCIDENTS = {
    "journal": [{"log": LOG, "line": 2, "stage": "confirmation", **INCIDENT}],
    "qualification": [],
}
BILLED = {
    "id": "pod-stop-1",
    "stage": "confirmation",
    "incident_time": INCIDENT["incident_time"],
    "cause": INCIDENT["cause"],
    "billed": 0.81,
    "evidence": [{"journal": LOG, "line": 2}, {"pod": "pod-events-20261102"}],
}
BILLING = {
    "currency": "USD",
    "stages": [
        {"stage": "engineering", "spent": 14.21},
        {"stage": "search", "spent": 9.4},
        {"stage": "numerical", "spent": 6.1},
        {"stage": "repair", "spent": 0},
        {"stage": "validation", "spent": 22.0},
        {"stage": "confirmation", "spent": 61.5},
    ],
    "incidents": [BILLED],
    "policy_stops": [],
}
KAPPA = {
    "A": 9.537,
    "F": 11.124,
    "P": 11.124,
    "unit": "pp per (ln K)^2",
    "source": "study 1's logistic curves at K*",
}
FORBIDDEN = r"available|ceiling|forecast|threshold"
ROW = {
    "test": ["calibration", "base", 60, "0.975", "A-F"],
    "level": 0.975,
    "due": True,
    "p0": 0.025,
    "studies": 10_000,
    "draws": 10_000,
    "reason": None,
    "X": 241,
    "c": 296,
    "passed": True,
    "rate": 0.0241,
    "interval": [0.0212, 0.0273],
}
JUDGE = {
    "path": "study3/labeling/diagnosis-b0003.json",
    "sha256": "d" * 64,
    "batch": "b0003",
    "cause": "judge",
    "diagnosis": "the judge's labels of the controls changed twice",
    "evidence": [{"path": "incidents/judge-b0003.txt", "sha256": "c" * 64, "root": "runs"}],
}


@dataclasses.dataclass(frozen=True)
class Harm:
    """A run's harm counts, as the label reader gives them."""

    harmful: int
    missing: int
    n: int = 521


def link(path: str, record) -> dict:
    data = json.dumps(record, sort_keys=True).encode()
    return {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "record": record}


def records_of(study: Study, *, freeze=None) -> dict:
    """The label-free records of a synthetic study, as the report's command gathers them."""
    entries = study.entries
    decisions = {
        kind: link(f"study3/decisions/{kind}.json", entries[kind]["record"])
        for kind in entries
        if kind.startswith(("n-", "budget-")) and entries[kind]["record"] is not None
    }
    simulation = dict(entries["simulation"]["record"])
    simulation |= {
        "gates": {
            **default_gates(),
            "power_pf": {"value": 0.61, "requirement": None, "passed": None},
        },
        "coverage": [coverage_row(0.975)],
        "level": 0.975,
        "validation_blocks": 28,
        "verification": verified(),
    }
    engineering = dict(entries["engineering"]["record"])
    engineering["checks"] = {
        name: {"path": f"reports/{name}.json", "sha256": "4" * 64, "root": "repo", "passed": True}
        for name in ("dry_run", "bridge", "theta0_utility", "stage0")
    }
    reading = {
        "rates": {
            "adamw-1.45e-05": {
                "arm": "A",
                "lr": 1.45e-5,
                "state": "qualified",
                "verdict": {"fallbacks": ["s10201/adamw-1.45e-05"], "warnings": []},
            }
        },
        "supersessions": [],
        "restarts": [],
        "problems": [],
    }
    return {
        "endpoints": list(study.endpoints),
        "engineering": link("study3/decisions/engineering.json", engineering),
        "policy": {
            "path": "docs/study3-preregistration.md",
            "commit": "5" * 40,
            "sha256": "6" * 64,
            "amendments": [1, 2, 3, 4, 5, 6],
        },
        "engineering_verdict": {
            "path": "qualification/c1/engineering/verdict.json",
            "sha256": "7" * 64,
            "summary": {
                "e_eng": 0.0004,
                "repeats": 12,
                "path_maxima": {"repeat": 0.0011, "shift": 0.0031},
                "margins": {"baseline~repeat": float("inf")},
            },
        },
        "simulation": link("study3/decisions/simulation.json", simulation),
        "simulation_results": {"path": rp.SIMULATION_RESULTS, "sha256": "8" * 64, "tests": [ROW]},
        "qualification": {1: reading},
        "decisions": decisions,
        "freeze": freeze,
    }


def harm_of(endpoints) -> dict[str, Harm]:
    """Every regular run's harm, by arm and seed (as the analysis script's world gives it)."""
    return {
        e.run_id: Harm(
            harmful=BASE[e.arm] + e.seed * STEP[e.arm] % 11 + (e.K > 6), missing=e.seed % 3
        )
        for e in endpoints
        if e.K is not None
    }


def gates_of(freeze: dict, endpoints) -> dict:
    """The utility gates at α = 0.05/8 over each frozen comparison's blocks."""
    by_seed: dict = {}
    for e in endpoints:
        by_seed.setdefault(e.seed, []).append(e)
    return {
        key: utility_gate(
            [
                block_state(by_seed[b["seed"]], b["seed"], 1, ["A", "F", "P"])
                for b in entry["blocks"]
            ],
            COMPARISONS[key],
            CONFIRMATION_ALPHA,
        )
        for key, entry in freeze["comparisons"].items()
    }


AUDIT = {
    "kind": "study3-audit-estimate",
    "selection": {
        "path": "study3-final/c1/audit/selection.json",
        "sha256": "2" * 64,
        "root": "runs",
    },
    "bias": {
        "AF": {
            "comparison": "AF",
            "arms": ["A", "F"],
            "missing_draws": 2,
            "missing_items": 1,
            "undetermined": False,
            "mar": {"estimate": -0.0042, "se": 0.0019, "variance": 3.61e-6},
            "worst_case": [-0.0061, 0.0013],
        },
        "PF": {
            "comparison": "PF",
            "arms": ["P", "F"],
            "missing_draws": 0,
            "missing_items": 0,
            "undetermined": True,
            "mar": {"estimate": None, "se": None, "variance": None},
            "worst_case": None,
        },
    },
    "error_rates": {
        "A": {
            "draws": 400,
            "distinct": 371,
            "counts": {
                "tp": 30,
                "fn": 4,
                "fp": 6,
                "tn": 358,
                "missing_judge_harmful": 0,
                "missing_judge_not_harmful": 2,
            },
            "harmful_by_reference": 27,
            "missing_draws": 2,
            "missing_items": 1,
            "undetermined": False,
            "sensitivity": {"mar": 0.882, "worst_case": [0.833, 0.882], "undetermined": False},
            "fpr": {"mar": 0.0165, "worst_case": [0.0164, 0.0165], "undetermined": False},
        }
    },
    "non_estimable": {},
    "missing": {"items": 1, "draws": 2},
}


@functools.cache
def done_world() -> tuple:
    """Candidate 1's done study, its freeze, and the analysis, validation's harm and the audit's
    estimate as written (2000 resamples)."""
    study = k_harmful(done_study(), scaled)
    state, freeze = study.advance(), frozen(study)
    harm = harm_of(study.endpoints)
    common = {"level": 0.975, "seed": 20261101, "resamples": 2000}
    result = study3_estimates.analysis(
        freeze, harm, **common, gates=gates_of(freeze, study.endpoints), curvature=KAPPA, spread={}
    )
    frozen_link = link("docs/study3-analysis-freeze-c1.json", freeze)
    named = {"path": frozen_link["path"], "sha256": frozen_link["sha256"], "root": "repo"}
    analysis = link(
        "study3-final/c1/analysis.json",
        {
            "kind": "study3-analysis",
            "candidate": 1,
            "written_at": WHEN,
            "code": CODE,
            "freeze": named,
            "level": 0.975,
            "bootstrap": {"seed": 20261101, "resamples": 2000},
            "kappa": {
                "path": "docs/study3-simulation-fixation.json",
                "sha256": "9" * 64,
                "root": "repo",
                "values": {a: KAPPA[a] for a in "AFP"},
                "unit": KAPPA["unit"],
                "source": KAPPA["source"],
            },
            "superseded": None,
            **result,
        },
    )
    validation = link(
        "study3-final/c1/validation-harm.json",
        {
            "kind": "study3-validation-harm",
            "freeze": named,
            "analysis": {"path": analysis["path"], "sha256": analysis["sha256"], "root": "runs"},
            "comparisons": study3_estimates.validation_harm(freeze, harm, **common),
        },
    )
    audit = link("study3-final/c1/audit/estimate.json", AUDIT)
    return study, state, frozen_link, analysis, validation, audit


def terminal(outcome: str) -> tuple[dict, dict, dict]:
    """A state of ``outcome``, its records and, at done, the analysis, validation's harm and
    the audit's estimate."""
    if outcome == "done":
        study, state, freeze, analysis, validation, audit = done_world()
        return (
            state,
            records_of(study, freeze=freeze),
            {"analysis": analysis, "validation": validation, "audit": audit},
        )
    study = Study()
    if outcome == "scientific_feasibility_report":
        study.ready()
        fail_at(study, "search", "F", 1)
        fail_at(study, "validation", "A", 2)
    elif outcome == "technical_feasibility_report":
        study.ready()
        study.entries["engineering"]["state"] = "failed"
        study.entries["engineering"]["record"]["state"] = "failed"
    elif outcome == "design_stop":
        study.ready()
        study.entries["simulation"]["state"] = "failed"
        study.entries["simulation"]["record"]["state"] = "failed"
    else:
        study.through_validation()
        study.n(N=99)
        study.budget(decision="stop")
    state = study.advance()
    assert state["stage"] == outcome
    return state, records_of(study), {}


def facts_of(outcome: str, **over) -> dict:
    state, records, extra = terminal(outcome)
    kwargs = {
        "records": records,
        "billing": BILLING,
        "incidents": INCIDENTS,
        "labeling_problems": (),
        **extra,
        **over,
    }
    return rp.report_facts(state, **kwargs)


# ------------------------------------------------------------------ the facts


@pytest.mark.parametrize("outcome", rp.OUTCOMES)
def test_every_terminal_outcome_has_its_facts_and_tables(outcome) -> None:
    state, records, extra = terminal(outcome)
    facts = rp.report_facts(
        state, records=records, billing=BILLING, incidents=INCIDENTS, labeling_problems=(), **extra
    )
    assert set(facts) == set(rp.SECTIONS)
    assert (facts["analysis"] not in (None, "not reached")) == (outcome == "done")
    assert facts["money"]["by_stage"] == BILLING["stages"]
    assert [i["billed"] for i in facts["money"]["incidents"]] == [0.81]
    assert not re.search(FORBIDDEN, rp.tables(facts) + json.dumps(facts))
    json.dumps(facts, allow_nan=False)  # strict JSON
    assert facts["outcome"]["variant"] == "standard" and facts["judge_mark"] is None
    assert facts["outcome"]["stage"] == outcome and facts["labeling"] == []


def test_the_facts_of_done() -> None:
    facts = facts_of("done")
    _, _, freeze, analysis, validation, audit = done_world()
    a = facts["analysis"]
    assert a["record"] == {"path": analysis["path"], "sha256": analysis["sha256"]}
    assert set(a["comparisons"]) == {"A-F", "P-F", "A-P", "A-F@K_harmful"}
    af = a["comparisons"]["A-F"]
    assert af["n"] == 60 and af["status"] == "confirmatory"
    assert set(af["missing_labels"]) == {"low", "high", "lower", "upper", "undetermined"}
    assert {"worst_case_unknown_blocks", "observed_extremes"} <= set(af)
    assert "worst_case_unknown_blocks" not in a["comparisons"]["A-P"]
    assert a["matched_k_harmful"]["n"] == 60 and a["matched_k_harmful"]["left_out"] == []
    assert a["curvature"]["source"] == KAPPA["source"] and a["utility"]["claim"] is True
    bound = a["utility"]["gates"]["A-F"]["bounds"]["A"]["mmlu"]
    assert set(bound) == {"n", "mean_pp", "upper_pp", "limit_pp", "passes"}
    assert bound["limit_pp"] == 5.0
    assert facts["validation_harm"]["comparisons"]["A-F"]["n"] == 28
    assert facts["audit"]["bias"]["AF"]["worst_case"] == [-0.0061, 0.0013]
    assert facts["audit"]["mar_outside_worst_case"] == {"AF": False, "PF": None}
    assert "fixed in sign by its known judge label" in facts["audit"]["bias_note"]
    assert "can lie outside the worst-case bound" in facts["audit"]["bias_note"]
    assert (
        facts["freeze"]["path"] == freeze["path"] and facts["freeze"]["sha256"] == freeze["sha256"]
    )
    assert len(facts["freeze"]["comparisons"]["A-F"]["seeds"]) == 60
    assert len(facts["freeze"]["comparisons"]["A-F"]["run_ids"]) == 240
    assert facts["n"]["N"] == 60 and facts["n"]["power_pf"] == 0.61
    confirmation = facts["confirmation"]
    assert confirmation["started"] == 60 and confirmation["outside"] == {"A-F": [], "P-F": []}
    assert (
        facts["simulation"]["coverage"][0]["comparison"]["c"]
        == coverage_row(0.975)["thresholds"]["comparison"]
    )
    assert facts["simulation"]["tests"][0]["non_coverage"] == 0.0241
    assert facts["engineering"]["margins"] == {"baseline~repeat": "+inf"}
    assert facts["qualification"]["c1"]["rates"]["adamw-1.45e-05"]["fallbacks"] == [
        "s10201/adamw-1.45e-05"
    ]
    assert facts["dry_run"]["checks"]["dry_run"]["path"] == "reports/dry_run.json"
    assert facts["dry_run"]["policy"]["amendments"] == [1, 2, 3, 4, 5, 6]
    utility = facts["utility"]
    assert len(utility["endpoints"]) == 6 * (28 + 60)
    assert set(utility["worst"]) == {"A", "F", "P"}
    assert utility["worst"]["A"]["mmlu"]["loss_pp"] == 1.0
    spacing = facts["validation"]["spacing"]
    assert spacing["registered"] == pytest.approx(0.22314355)
    assert {len(spacing["arms"][arm]) for arm in "AFP"} == {2}
    assert spacing["arms"]["A"][0]["blocks"] == 28


def test_the_tables_show_harm_in_pp_and_no_prose() -> None:
    facts = facts_of("done")
    text = rp.tables(facts)
    primary = facts["analysis"]["comparisons"]["A-F"]["primary"]
    assert f"[{100 * primary['lower']:.3f}, {100 * primary['upper']:.3f}]" in text
    bias = facts["audit"]["bias"]["AF"]["mar"]["estimate"]
    assert f"{100 * bias:.3f}" in text
    assert all(f"## {section}" in text for section in rp.SECTIONS)
    body = [line for line in text.splitlines() if line and not line.startswith(("#", "|"))]
    assert body == []  # headings and tables only


def test_a_mar_estimate_outside_its_worst_case_is_flagged_with_why() -> None:
    """B_E's MAR estimate averages over the resolved draws alone, while each missing draw's term
    in the worst-case bound is fixed in sign by its judge label: it can lie outside the bound,
    and the report says so."""
    audit = done_world()[5]
    af = {**AUDIT["bias"]["AF"], "mar": {"estimate": 0.0021, "se": 0.001, "variance": 1e-6}}
    record = {**audit["record"], "bias": {**AUDIT["bias"], "AF": af}}
    facts = facts_of("done", audit={**audit, "record": record})
    assert facts["audit"]["mar_outside_worst_case"] == {"AF": True, "PF": None}
    text = rp.tables(facts)
    assert facts["audit"]["bias_note"] in text and "| AF | false | 0.210 |" in text


def test_the_dead_ends_are_reported_apart() -> None:
    scientific = facts_of("scientific_feasibility_report")["outcome"]
    assert scientific["dead_ends"]["technical"] == []
    assert [(e["candidate"], e["reason"]) for e in scientific["dead_ends"]["scientific"]] == [
        (1, "scientific_failure"),
        (2, "validation_gate"),
    ]
    technical = facts_of("technical_feasibility_report")["outcome"]
    assert technical["dead_ends"] == {
        "technical": [{"candidate": 1, "stage": None, "arms": [], "reason": "engineering_failed"}],
        "scientific": [],
    }
    assert facts_of("design_stop")["outcome"]["dead_ends"] == {"technical": [], "scientific": []}


def test_a_stage_the_route_or_an_incident_reached_needs_its_spending_entry() -> None:
    """The scientific report's route reached search and validation (candidate 1's search failed,
    candidate 2's validation gate): each needs its spending entry, as does the stage an incident
    is billed at."""
    route = facts_of("scientific_feasibility_report")["outcome"]["route"]
    assert {e["stage"] for e in route} == {"search", "validation"}
    for gone in ("search", "validation", "confirmation"):
        stages = [s for s in BILLING["stages"] if s["stage"] != gone]
        with pytest.raises(ValueError, match=f"no spending entry for {gone}"):
            facts_of("scientific_feasibility_report", billing={**BILLING, "stages": stages})
    stages = [s for s in BILLING["stages"] if s["stage"] != "search"]
    assert facts_of("design_stop", billing={**BILLING, "stages": stages})["money"]["by_stage"] == (
        stages
    )


def test_the_sections_a_terminal_stop_did_not_reach() -> None:
    facts = facts_of("design_stop")
    for section in (
        "validation",
        "n",
        "confirmation",
        "utility",
        "analysis",
        "validation_harm",
        "audit",
        "freeze",
    ):
        assert facts[section] == "not reached", section
    budget = facts_of("budget_stop")
    assert budget["n"]["budget"]["decision"] == "stop" and budget["confirmation"] == "not reached"
    assert budget["validation"]["gates"]["AF"]["passes"] is True


def test_the_labeling_problems_and_an_unbilled_journal_incident() -> None:
    assert facts_of("done", labeling_problems=("r1",))["labeling"] == ["r1"]
    unbilled = {**BILLING, "incidents": []}
    with pytest.raises(ValueError, match="billed once") as refused:
        facts_of("done", billing=unbilled)
    assert f"{LOG} line 2 is named by no billed incident" in str(refused.value)
    # the cause is shown, never matched: another text names the same record
    other = {**BILLING, "incidents": [{**BILLED, "cause": "another cause"}]}
    assert facts_of("design_stop", billing=other)["money"]["incidents"][0]["cause"] == (
        "another cause"
    )


def qualification_world(root: Path, *kinds: str) -> list[str]:
    """Qualification records made with --infrastructure, of one case: a part record carrying it
    (``part``) and a case restart (``restart``); beside them a part record without it."""
    case = root / "qualification" / "c1" / "numerical" / "s10200" / "adamw-1.45e-05"
    case.mkdir(parents=True)
    (case / "baseline-attempt-1.json").write_text(
        json.dumps({"part": "baseline", "infrastructure": None})
    )
    paths = []
    for kind in kinds:
        if kind == "part":
            name = "baseline-attempt-2.json"
            record = {
                "part": "baseline",
                "infrastructure": {"reason": "the Pod stopped", "previous": [], "restarts": []},
            }
        else:
            name = "restart-1.json"
            record = {
                "case": {},
                "restart": 1,
                "reason": "the Pod stopped",
                "attempts": [],
                "launcher": None,
            }
        (case / name).write_text(json.dumps(record))
        paths.append(f"qualification/c1/numerical/s10200/adamw-1.45e-05/{name}")
    return paths


def billed(id_: str, *evidence, stage: str = "numerical", price: float = 0.42) -> dict:
    """A billed incident naming ``evidence``."""
    return {
        "id": id_,
        "stage": stage,
        "incident_time": "2026-11-01T01:00:00Z",
        "cause": "the Pod stopped",
        "billed": price,
        "evidence": list(evidence),
    }


@pytest.mark.parametrize("kind", ["part", "restart"])
def test_a_qualification_infrastructure_record_needs_its_billed_incident(tmp_path, kind) -> None:
    (path,) = qualification_world(tmp_path, kind)
    found = rp.qualification_incidents(tmp_path)
    assert [(i["path"], i["kind"]) for i in found] == [(path, kind)]
    incidents = {**INCIDENTS, "qualification": found}
    with pytest.raises(ValueError, match="billed once") as refused:
        facts_of("design_stop", incidents=incidents)
    assert path in str(refused.value)
    stop = billed("stop-2", {"qualification": path})
    billing = {**BILLING, "incidents": [BILLED, stop]}
    facts = facts_of("design_stop", incidents=incidents, billing=billing)
    assert stop in facts["money"]["incidents"]
    assert path in rp.tables(facts)


def test_one_stop_that_left_a_restart_and_a_part_record_is_one_incident(tmp_path) -> None:
    """A Pod stop that left a case restart and a part record made with --infrastructure is
    billed once: one incident naming both records, at its one price."""
    paths = qualification_world(tmp_path, "part", "restart")
    incidents = {**INCIDENTS, "qualification": rp.qualification_incidents(tmp_path)}
    stop = billed("stop-2", *({"qualification": p} for p in paths), {"pod": "pod-x"})
    facts = facts_of("design_stop", incidents=incidents,
                     billing={**BILLING, "incidents": [BILLED, stop]})  # fmt: skip
    money = facts["money"]["incidents"]
    assert [i["id"] for i in money] == ["pod-stop-1", "stop-2"]
    assert money[1]["billed"] == 0.42 and money[1]["evidence"] == stop["evidence"]
    assert sum(i["billed"] for i in money) == pytest.approx(0.81 + 0.42)
    text = rp.tables(facts)
    assert all(p in text for p in paths) and "pod pod-x" in text


def test_a_record_named_twice_or_by_no_incident_is_refused(tmp_path) -> None:
    paths = qualification_world(tmp_path, "part", "restart")
    incidents = {**INCIDENTS, "qualification": rp.qualification_incidents(tmp_path)}
    part, restart = ({"qualification": p} for p in paths)
    for refused_incidents, words in [
        ([BILLED, billed("a", part, restart), billed("b", part)], "named twice"),  # across
        ([BILLED, billed("a", part, restart, part)], "named twice"),  # within one incident
        ([BILLED, billed("a", part)], f"{paths[1]} is named by no billed incident"),
        (
            [BILLED, billed("a", part, restart, {"qualification": "qualification/c1/x.json"})],
            "qualification/c1/x.json is no incident of the records",
        ),
        ([BILLED, billed("a", part, restart), billed("a", {"pod": "pod-x"})], "listed twice"),
        ([BILLED, billed("a", part, restart), billed("b")], "not a non-empty list"),
        ([BILLED, billed("a", part, restart, stage="search")], "is of numerical or repair"),
        ([{**BILLED, "stage": "validation"}, billed("a", part, restart)], "is of confirmation"),
    ]:
        billing = {**BILLING, "incidents": refused_incidents}
        with pytest.raises(ValueError, match=re.escape(words)):
            facts_of("design_stop", incidents=incidents, billing=billing)
    repair = billed("a", part, restart, stage="repair")  # the repair's rates: numerical runs
    facts = facts_of("design_stop", incidents=incidents, billing={**BILLING,
                     "incidents": [BILLED, repair]})  # fmt: skip
    assert facts["money"]["incidents"][1]["stage"] == "repair"


def test_two_journal_rounds_of_one_stop_with_other_cause_texts_are_one_incident() -> None:
    rounds = [
        {"log": LOG, "line": 3, "stage": "confirmation", **INCIDENT},
        {"log": LOG, "line": 4, "stage": "confirmation", **INCIDENT, "cause": "the Pod stopped."},
    ]
    stop = billed("stop", *({"journal": LOG, "line": n} for n in (3, 4)), stage="confirmation")
    facts = facts_of("design_stop", incidents={"journal": rounds, "qualification": []},
                     billing={**BILLING, "incidents": [stop]})  # fmt: skip
    assert facts["money"]["incidents"] == [stop]
    first = {**stop, "evidence": stop["evidence"][:1]}
    with pytest.raises(ValueError, match=re.escape(f"{LOG} line 4 is named by no billed")):
        facts_of("design_stop", incidents={"journal": rounds, "qualification": []},
                 billing={**BILLING, "incidents": [first]})  # fmt: skip


def test_journal_incidents_are_the_study_3_recovery_lines_with_an_incident(tmp_path) -> None:
    launch = tmp_path / "launch"
    launch.mkdir()
    header = {"study": 3, "candidate": 1, "step": "confirmation", "round": None}
    lines = [
        {"event": "start", "experiment": "study3", "run": "A-1.45e-05/s10400", "header": header},
        {
            "event": "recovery",
            "experiment": "study3",
            "runs": ["A-1.45e-05/s10400"],
            "reason": "a cell's round",
        },
        {
            "event": "recovery",
            "experiment": "study3",
            "only": "A-1.45e-05/s10400",
            "runs": ["A-1.45e-05/s10400"],
            **INCIDENT,
        },
        {"event": "recovery", "experiment": "study2", "only": "x", **INCIDENT},
    ]
    (launch / "study3-c1-confirmation.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    assert rp.journal_incidents(tmp_path) == [
        {"log": LOG, "line": 3, "stage": "confirmation", **INCIDENT}
    ]
    # a log whose study-3 starts name no step, or two, shows no stage
    for steps in ([None], ["search", "numerical"]):
        starts = [{**lines[0], "header": {**header, "step": step}} for step in steps]
        (launch / "study3-c1-confirmation.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in [*starts, *lines[1:]])
        )
        assert [i["stage"] for i in rp.journal_incidents(tmp_path)] == [None]
    (launch / "study3-c1-other.jsonl").write_text('{"event": "start"\n{"event": "end"}\n')
    with pytest.raises(ValueError, match="torn line"):
        rp.journal_incidents(tmp_path)


def test_refused_at_a_stage_that_is_not_terminal_and_at_done_without_the_audit() -> None:
    study = Study()
    study.through_validation()
    with pytest.raises(ValueError, match="terminal"):
        rp.report_facts(
            study.advance(),
            records=records_of(study),
            billing=BILLING,
            incidents=INCIDENTS,
            labeling_problems=(),
        )
    with pytest.raises(ValueError, match="audit") as refused:
        facts_of("done", audit=None, labeling_problems=("r1",))
    assert "r1" in str(refused.value) and "judge" in str(refused.value)
    _, _, _, analysis, _, _ = done_world()
    with pytest.raises(ValueError, match="follows done"):
        facts_of("budget_stop", analysis=analysis)


# ------------------------------------------------------------------ judge_technical


def judge_technical(**over) -> dict:
    return facts_of(
        "done", analysis=None, validation=None, audit=None, labeling_problems=("r1",), **over
    )


def test_the_judge_technical_variant() -> None:
    facts = judge_technical(judge_diagnoses=[JUDGE])
    standard = facts_of("done")
    assert facts["outcome"]["variant"] == "judge_technical"
    for section in ("analysis", "validation_harm", "audit"):
        assert facts[section] == "not reached: the judge's technical mark"
    _, _, freeze, _, _, _ = done_world()
    assert facts["freeze"] == standard["freeze"] and facts["freeze"]["sha256"] == freeze["sha256"]
    assert len(facts["freeze"]["comparisons"]["P-F"]["run_ids"]) == 240
    assert facts["judge_mark"] == [JUDGE] and facts["labeling"] == ["r1"]
    assert facts["utility"] == standard["utility"] and facts["money"] == standard["money"]
    assert facts["outcome"]["dead_ends"] == {
        "technical": [
            {"candidate": 1, "stage": "labeling", "arms": [], "reason": "judge_mark",
             "batch": "b0003"}
        ],
        "scientific": [],
    }  # fmt: skip
    assert set(facts) == set(rp.SECTIONS)
    text = rp.tables(facts)
    assert "| technical | 1 | labeling |  | judge_mark | b0003 |" in text
    assert not re.search(FORBIDDEN, text + json.dumps(facts))


def test_every_judge_diagnosis_is_reported_with_its_dead_end() -> None:
    other = {**JUDGE, "path": "study3/labeling/diagnosis-b0005.json", "batch": "b0005"}
    facts = judge_technical(judge_diagnoses=[JUDGE, other])
    assert facts["judge_mark"] == [JUDGE, other]
    technical = facts["outcome"]["dead_ends"]["technical"]
    assert [(e["reason"], e["batch"]) for e in technical] == [
        ("judge_mark", "b0003"),
        ("judge_mark", "b0005"),
    ]
    text = rp.tables(facts)
    assert all(mark["path"] in text for mark in (JUDGE, other))


def test_the_judge_technical_variant_needs_a_judge_diagnosis_and_no_analysis() -> None:
    with pytest.raises(ValueError, match="judge"):
        judge_technical()
    with pytest.raises(ValueError, match="judge"):
        judge_technical(judge_diagnoses=[JUDGE, {**JUDGE, "cause": "infrastructure"}])
    with pytest.raises(ValueError, match="do not mix"):
        facts_of("done", judge_diagnoses=[JUDGE])


def test_a_judge_diagnosis_object_reads_as_its_record() -> None:
    found = Diagnosis(
        path=Path("/runs/study3/labeling/diagnosis-b0003.json"),
        sha256="d" * 64,
        batch="b0003",
        cause="judge",
        diagnosis=JUDGE["diagnosis"],
        evidence=tuple(JUDGE["evidence"]),
    )
    (mark,) = judge_technical(judge_diagnoses=[found])["judge_mark"]
    assert mark["path"] == "/runs/study3/labeling/diagnosis-b0003.json"
    assert mark["evidence"] == JUDGE["evidence"] and mark["cause"] == "judge"


# ------------------------------------------------------------------ the researcher's files


def test_read_billing_is_a_closed_schema(tmp_path) -> None:
    path = tmp_path / "billing.json"
    path.write_text(json.dumps(BILLING))
    assert rp.read_billing(path) == BILLING
    for bad, words in [
        ({**BILLING, "available": 294}, "'available' is not a field"),
        (
            {**BILLING, "stages": [{"stage": "engineering", "spent": 14.21, "cap": 20}]},
            "'cap' is not a field",
        ),
        ({**BILLING, "stages": [{"stage": "fun", "spent": 1}]}, "stage is not one of"),
        ({**BILLING, "stages": [{"stage": "search", "spent": -1}]}, "finite amount"),
        ({**BILLING, "stages": [{"stage": "search", "spent": 1}] * 2}, "listed twice"),
        ({**BILLING, "incidents": [{**BILLED, "source": "journal"}]}, "'source' is not a field"),
        ({**BILLING, "incidents": [{**BILLED, "incident_time": "2026-11-02T03:12:00"}]}, "UTC"),
        ({**BILLING, "incidents": [{**BILLED, "id": "a b"}]}, "id is not an id"),
        ({**BILLING, "incidents": [{**BILLED, "evidence": "pod-x"}]}, "not a non-empty list"),
        (
            {**BILLING, "incidents": [{**BILLED, "evidence": [{"journal": LOG, "line": 0}]}]},
            "line is not a line number",
        ),
        (
            {**BILLING, "incidents": [{**BILLED, "evidence": [{"pod": "x", "line": 1}]}]},
            'is not {"journal", "line"}, {"qualification"} or {"pod"}',
        ),
        (
            {**BILLING, "incidents": [{**BILLED, "evidence": [{"qualification": "a\nb"}]}]},
            "qualification is not one non-empty line",
        ),
        ({**BILLING, "currency": "EUR"}, "currency"),
        ({**BILLING, "policy_stops": [{"stage": "search", "spent": 5}]}, "'spent' is not"),
    ]:
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=re.escape(words)):
            rp.read_billing(path)


def test_read_embargo(tmp_path) -> None:
    path = tmp_path / "embargo.json"
    path.write_text(json.dumps({"breaches": []}))
    assert rp.read_embargo(path) == []
    breach = {"stage": "validation", "what": "one run's labels opened by hand"}
    path.write_text(json.dumps({"breaches": [breach]}))
    assert rp.read_embargo(path) == [breach]
    assert facts_of("design_stop", embargo=[breach])["embargo"] == [breach]
    path.write_text(json.dumps({"breaches": [{**breach, "who": "x"}]}))
    with pytest.raises(ValueError, match="'who' is not a field"):
        rp.read_embargo(path)
    path.write_text(json.dumps([breach]))
    with pytest.raises(ValueError, match="no JSON object"):
        rp.read_embargo(path)


def test_the_simulation_results_adapter(tmp_path) -> None:
    assert rp.simulation_results(tmp_path) is None
    path = tmp_path / rp.SIMULATION_RESULTS
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"kind": "simulation-results", "tests": [ROW]}))
    found = rp.simulation_results(tmp_path)
    assert found["tests"] == [ROW] and found["path"] == "study3/simulation/results.json"
    assert found["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text(json.dumps({"kind": "simulation-results", "tests": [{**ROW, "extra": 1}]}))
    with pytest.raises(ValueError, match="rows"):
        rp.simulation_results(tmp_path)


def test_the_module_loads_no_label_reader_and_does_no_gpu_work() -> None:
    """A fresh process that imports the module loads no label reader, the final analysis's
    reader, study 2's label modules or torch."""
    assert gpu_uses(Path(rp.__file__), "zero_field_refusal.analysis") == set()
    names = [
        "zero_field_refusal.analysis.study3_labels",
        "zero_field_refusal.analysis.study3_final",
        "zero_field_refusal.analysis.study2_labels",
        "torch",
    ]
    code = (
        "import json, sys; import zero_field_refusal.analysis.study3_report; "
        f"print(json.dumps([n for n in {names!r} if n in sys.modules]))"
    )
    found = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    )
    assert json.loads(found.stdout) == []


# ------------------------------------------------------------------ the report command


class ReportWorld(AnalysisWorld):
    """The analysis script's world at done, with the label-free records of the report: the
    decision records, the registration's policy, the engineering verdict, the simulation's
    results, the qualification's reading, a launch log with one incident, the researcher's
    billing and embargo files, and the labeling ledger's answers (no problem, no diagnosis)."""

    def __init__(self, tmp_path, monkeypatch, capsys) -> None:
        super().__init__(tmp_path, monkeypatch, capsys)
        root, mp = self.root, monkeypatch
        records = records_of(self.study)
        self.decisions = {kind: records[kind]["record"] for kind in ("engineering", "simulation")}
        self.decisions["n-c1"] = self.study.entries["n-c1"]["record"]
        for kind, record in self.decisions.items():
            path = root / "study3" / "decisions" / f"{kind}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record))
        mp.setattr(self.script, "read_decision", self.read_decision)
        mp.setattr(self.script, "registered_policy", lambda repo: dict(records["policy"]))
        mp.setattr(
            self.script,
            "qualification_readings",
            lambda root_, expected: dict(records["qualification"]),
        )
        verdict = root / "qualification" / "c1" / "engineering" / "verdict.json"
        verdict.parent.mkdir(parents=True)
        verdict.write_text(json.dumps({"summary": records["engineering_verdict"]["summary"]}))
        results = root / rp.SIMULATION_RESULTS
        results.parent.mkdir(parents=True)
        results.write_text(
            json.dumps(
                {"kind": "simulation-results", "tests": records["simulation_results"]["tests"]}
            )
        )
        (root / "launch").mkdir()
        header = {"study": 3, "candidate": 1, "step": "confirmation", "round": None}
        lines = [
            {"event": "start", "experiment": "study3", "run": "A-1.45e-05/s10400",
             "header": header},
            {"event": "recovery", "experiment": "study3", "only": "A-1.45e-05/s10400",
             "runs": ["A-1.45e-05/s10400"], **INCIDENT},
        ]  # fmt: skip
        (root / LOG).write_text("".join(json.dumps(line) + "\n" for line in lines))
        self.billing, self.embargo = tmp_path / "billing.json", tmp_path / "embargo.json"
        self.billing.write_text(json.dumps(BILLING))
        self.embargo.write_text(json.dumps({"breaches": []}))
        self.asked: list[tuple[str, str | None]] = []
        self.diagnoses: tuple = ()
        mp.setattr(self.script, "labeling_problems", self.labeling)
        mp.setattr(self.script, "diagnoses", lambda runs_root, sealed_dir=None: self.diagnoses)

    def read_decision(self, runs_root, kind, *, repo):
        return self.decisions.get(kind), []

    def labeling(self, runs_root, sealed_dir=None) -> tuple[str, ...]:
        self.asked.append((str(runs_root), None if sealed_dir is None else str(sealed_dir)))
        return ()

    def audited(self) -> None:
        """The audit's selection, agreements and estimate, as the audit command writes them."""
        folder = self.root / "study3-final" / "c1"
        audit = folder / "audit"
        audit.mkdir()
        analysis = folder / "analysis.json"
        freeze = json.loads(analysis.read_text())["freeze"]
        selection = {
            "kind": "study3-audit-selection",
            "freeze": freeze,
            "analysis": {
                "path": "study3-final/c1/analysis.json",
                "sha256": sha256(analysis),
                "root": "runs",
            },
        }
        (audit / "selection.json").write_text(json.dumps(selection))
        (audit / "agreements.json").write_text(json.dumps({"kind": "study3-audit-agreements"}))
        estimate = {
            **AUDIT,
            "selection": {
                "path": "study3-final/c1/audit/selection.json",
                "sha256": sha256(audit / "selection.json"),
                "root": "runs",
            },
            "agreements": {
                "path": "study3-final/c1/audit/agreements.json",
                "sha256": sha256(audit / "agreements.json"),
                "root": "runs",
            },
        }
        (audit / "estimate.json").write_text(json.dumps(estimate))

    def report(self, *extra) -> str:
        return self.run(
            "report",
            "--runs-root",
            self.root,
            "--billing",
            self.billing,
            "--embargo",
            self.embargo,
            *extra,
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_raisers(monkeypatch) -> list[str]:
    """Every harm and item reader and every entry of the final analysis's reader, a raiser."""
    calls = harm_raisers(monkeypatch)
    for name in (*study3_final.SEALED_ENTRIES, "gather"):

        def raiser(*args, _name=name, **kwargs):
            calls.append(_name)
            raise AssertionError(f"a label reader was called: {_name}")

        monkeypatch.setattr(study3_final, name, raiser)
    return calls


@pytest.fixture
def reported(tmp_path, monkeypatch, capsys) -> ReportWorld:
    world = ReportWorld(tmp_path, monkeypatch, capsys)
    world.frozen()
    world.run("analyze", "--runs-root", world.root, "--candidate", "1")
    world.audited()
    return world


def test_the_report_command_writes_the_facts_and_tables_once(reported, monkeypatch) -> None:
    world = reported
    calls = final_raisers(monkeypatch)
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    out = world.report()
    (folder,) = sorted((world.root / "study3-final").glob("report-*"))
    assert re.fullmatch(r"report-\d{8}T\d{6}Z", folder.name)
    assert str(folder / "facts.json") in out and str(folder / "tables.md") in out
    facts = json.loads((folder / "facts.json").read_text())
    tables = (folder / "tables.md").read_text()
    assert set(facts) == set(rp.SECTIONS) and facts["outcome"]["variant"] == "standard"
    c1 = world.root / "study3-final" / "c1"
    assert facts["analysis"]["record"] == {
        "path": "study3-final/c1/analysis.json",
        "sha256": sha256(c1 / "analysis.json"),
    }
    assert facts["validation_harm"]["record"]["sha256"] == sha256(c1 / "validation-harm.json")
    assert facts["audit"]["record"]["sha256"] == sha256(c1 / "audit" / "estimate.json")
    assert facts["freeze"]["sha256"] == sha256(world.repo / "docs/study3-analysis-freeze-c1.json")
    assert facts["money"]["incidents"] == BILLING["incidents"] and facts["embargo"] == []
    assert facts["engineering"]["path_maxima"] == {"repeat": 0.0011, "shift": 0.0031}
    assert facts["simulation"]["results"]["path"] == "study3/simulation/results.json"
    assert world.asked == [(str(world.root), None)]  # the default sealed directory
    assert calls == [] and guard.attempts == []
    assert SENTINEL not in tables + json.dumps(facts) + out
    assert tables == rp.tables(facts)
    taken = world.root / "elsewhere"
    taken.mkdir()
    assert "exists" in world.refused(
        "report",
        "--runs-root",
        world.root,
        "--billing",
        world.billing,
        "--embargo",
        world.embargo,
        "--out",
        taken,
    )
    assert list(taken.iterdir()) == []


def test_the_report_command_refuses_an_unbilled_incident_and_writes_nothing(reported):
    world = reported
    world.billing.write_text(json.dumps({**BILLING, "incidents": []}))
    why = world.refused(
        "report", "--runs-root", world.root, "--billing", world.billing, "--embargo", world.embargo
    )
    assert "billed once" in why and f"{LOG} line 2 is named by no billed incident" in why
    assert not list((world.root / "study3-final").glob("report-*"))
    world.billing.write_text(json.dumps({**BILLING, "available": 294}))
    assert "'available' is not a field" in world.refused(
        "report", "--runs-root", world.root, "--billing", world.billing, "--embargo", world.embargo
    )


def test_the_report_command_refuses_an_estimate_of_another_selection(reported) -> None:
    world = reported
    selection = world.root / "study3-final" / "c1" / "audit" / "selection.json"
    selection.write_text(selection.read_text() + " ")
    why = world.refused(
        "report", "--runs-root", world.root, "--billing", world.billing, "--embargo", world.embargo
    )
    assert "selection" in why
    assert not list((world.root / "study3-final").glob("report-*"))


def test_the_report_command_refuses_a_selection_naming_an_analysis_it_cannot_read(reported):
    """The selection names an analysis that cannot be read (here a folder where analysis.json
    was, a later analysis-2.json in force): refused with the error's type, nothing written."""
    world = reported
    c1 = world.root / "study3-final" / "c1"
    (c1 / "analysis-2.json").write_bytes((c1 / "analysis.json").read_bytes())
    (c1 / "analysis.json").unlink()
    (c1 / "analysis.json").mkdir()
    why = world.refused(
        "report", "--runs-root", world.root, "--billing", world.billing, "--embargo", world.embargo
    )
    assert "names study3-final/c1/analysis.json, which cannot be read" in why
    assert "IsADirectoryError" in why
    assert not list((world.root / "study3-final").glob("report-*"))


def test_the_report_command_writes_judge_technical_after_a_judge_diagnosis(
    tmp_path, monkeypatch, capsys
) -> None:
    world = ReportWorld(tmp_path, monkeypatch, capsys)
    world.frozen()  # labeling stopped: no analysis
    world.diagnoses = (
        Diagnosis(
            path=world.root / "study3" / "labeling" / "diagnosis-b0001.json",
            sha256="e" * 64,
            batch="b0001",
            cause="infrastructure",
            diagnosis="the GPU",
            evidence=({"path": "incidents/gpu.log", "sha256": "f" * 64, "root": "runs"},),
        ),
        Diagnosis(
            path=world.root / "study3" / "labeling" / "diagnosis-b0002.json",
            sha256="d" * 64,
            batch="b0002",
            cause="judge",
            diagnosis=JUDGE["diagnosis"],
            evidence=(),
        ),
    )
    calls = final_raisers(monkeypatch)
    guard = SealedGuard(monkeypatch, runs_root=world.root)
    world.report()
    (folder,) = sorted((world.root / "study3-final").glob("report-*"))
    facts = json.loads((folder / "facts.json").read_text())
    assert facts["outcome"]["variant"] == "judge_technical"
    (mark,) = facts["judge_mark"]  # the judge's diagnosis alone; the infrastructure one releases
    assert mark["path"] == "study3/labeling/diagnosis-b0002.json"
    assert mark["sha256"] == "d" * 64 and facts["analysis"].startswith("not reached")
    assert [(e["reason"], e["batch"]) for e in facts["outcome"]["dead_ends"]["technical"]] == [
        ("judge_mark", "b0002")
    ]
    assert facts["freeze"]["sha256"] == sha256(world.repo / "docs/study3-analysis-freeze-c1.json")
    assert calls == [] and guard.attempts == []


def test_the_report_command_at_done_without_the_audit_names_the_variant(reported) -> None:
    world = reported
    (world.root / "study3-final" / "c1" / "audit" / "estimate.json").unlink()
    why = world.refused(
        "report", "--runs-root", world.root, "--billing", world.billing, "--embargo", world.embargo
    )
    assert "audit" in why and "judge_technical" in why
