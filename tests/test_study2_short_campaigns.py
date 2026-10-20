"""Study 2's short-regime stage as a whole: its configs, the stage's state (analysis.study2_short
``progress``, ``preconditioner_pairing``, ``exhausted_runs``, ``confirmation_guard``,
``look_blocks``, ``read_fixation``, ``run_facts``), the campaigns written from it
(scripts/study2_short_campaigns.py, loadable by the launcher with every run's config checked
against the protocol), the prerequisites' driver and the summary (scripts/study2_short_summary.py).

Synthetic runs follow tests/test_study2_short.py's analytic world, made to conform to the
campaigns; the scripts read them through a patched ``load_trajectories``."""

import dataclasses
import hashlib
import importlib.util
import json
import math
import re
import runpy
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from test_study2_short import (
    COMMIT as RUN_COMMIT,
)
from test_study2_short import (
    SCALE,
    THETA0,
    W_MAX,
    control,
    made,
    request_check,
    reusable,
    stage_a_reference,
    traj,
    verdict_of,
    verdict_path,
)

from zero_field_refusal.analysis import study2_short
from zero_field_refusal.analysis.stage_a import arm_of
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    ARMS,
    CONFIRMATION_SEEDS,
    FA,
    FA_CONTROL_ARM,
    FIXATION_KEYS,
    NUMERICAL_SEEDS,
    PREWARMED,
    TIMECOURSE,
    TIMECOURSE_SEEDS,
    TIMECOURSE_STEPS,
    V0_ARMS,
    VALIDATION_SEEDS,
    bootstrap_t_interval,
    confirmation_guard,
    conformance,
    exhausted_runs,
    issued_ranges,
    issued_w_max,
    look_blocks,
    looks,
    measures_rayleigh,
    mechanism_analysis,
    mechanism_blocks,
    mechanism_state,
    numerical_digest,
    preconditioner_pairing,
    prescribed,
    prescribed_hash,
    progress,
    read_abandonment,
    read_fixation,
    read_numerical,
    run_facts,
    secondary_contrasts,
    seed_role,
    split_timecourse,
    timecourse_guard,
    timecourse_summary,
    w_max_problems,
)
from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.launch import (
    MAX_ATTEMPTS,
    campaign_header,
    launch_records,
    load_campaign,
    run_campaign,
)
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.seeding import derive_seed

SCRIPTS = Path(__file__).parents[1] / "scripts"
CAMPAIGNS, SUMMARY = SCRIPTS / "study2_short_campaigns.py", SCRIPTS / "study2_short_summary.py"
OPTIMIZERS = {ADAMW: [], FA: ["optimizer=adam_frozen"], PREWARMED: ["optimizer=adamw_prewarmed"]}
COMMIT = "0123456789abcdef0123456789abcdef01234567"
FIXATION = {  # the decided values; the resamples, seed and bias allowance are placeholders
    "w_max": 0.47,
    "levels": [0.992, 0.9631],
    "se_floor": 0.0,
    "bias_allowance": 0.01079,
    "interval": "bootstrap_t",
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 20261016,
    "looks": [36, 60],
    "cap": 80,
    "equivalence_min_supported": 36,
    "direction_min_supported": 10,
    "secondary_interval": "t",
    "secondary_status": "secondary",
    "mechanism": {
        "target": 60,
        "cap": 80,
        "minimum": 36,
        "zero_se": 1e-12,
        "comparison_level": 0.975,
        "effects": [0.05, 0.025],
        "level": 0.95,
        "resamples": 2000,
        "seed": 20261020,
    },  # fmt: skip
    "audit": {
        "sizes": {"adamw": 400, "fa": 400, "prewarmed": 200},
        "selection": "pps_with_replacement",
        "weight": "max_abs_coefficient",
        "interval": "clopper_pearson_total_error",
        "arm_level": 0.9916666666666667,
        "bias_level": 0.975,
        "missing_reference": "error",
        "seed": 20261021,
        "agreement_check_share": 0.1,
    },  # fmt: skip
}
LOOK_SETTINGS = {k: v for k, v in FIXATION.items()
                 if k not in ("w_max", "levels", "looks", "cap", "secondary_interval",
                              "secondary_status", "mechanism", "audit")}  # fmt: skip


# ------------------------------------------------------------------ configs and arms


@pytest.mark.parametrize("arm", [ADAMW, FA, PREWARMED])
def test_the_regular_runs_are_stage_as_25_step_runs_but_name_and_role(arm) -> None:
    # The new discovery runs pool with Stage A's grids: everything they compute is the same.
    stage_a = load_config("experiments/study2_stage_a", ["schedule=short", *OPTIMIZERS[arm]])
    role = "study2_short.role=discovery"
    short = load_config("experiments/study2_short", [*OPTIMIZERS[arm], role])
    a, b = OmegaConf.to_container(stage_a), OmegaConf.to_container(short)
    assert (a.pop("experiment"), b.pop("experiment")) == ("study2_stage_a", "study2_short")
    assert a.pop("stage_a") == b.pop("study2_short") == {"role": "discovery"}
    # The update scalars are recorded besides (amendment 4, section 4): logging only.
    assert b["tracking"].pop("update_scalars") is True
    assert a["tracking"].pop("update_scalars") is False
    assert a == b
    assert (short.protocol_version, short.train.steps) == (2, 25)
    assert list(short.schedule.evaluate_at) == [25]
    assert (short.schedule.policy, short.schedule.instability.grad_factor) == ("fixed", 10)
    assert short.eval.monitor.version == 2 and "rayleigh" not in short.eval
    assert arm_of(short) == arm


def test_the_numerical_runs_are_the_regular_runs_with_a_k_only_evaluation() -> None:
    role = "study2_short.role=numerical"
    plain = load_config("experiments/study2_short", [role])
    k_only = load_config("experiments/study2_short_k_only", [role])
    a, b = OmegaConf.to_container(plain), OmegaConf.to_container(k_only)
    assert b["eval"].pop("k_only") is True and "k_only" not in a["eval"]
    assert a == b and k_only.experiment == "study2_short"  # the stage's records stay together
    cfg = load_run_config(
        "experiments/study2_short_k_only",
        [role, "optimizer.lr=1.49e-05", "train.batch_size=20", "seed=2200"],
    )
    assert config_hash(cfg) and cfg.eval.k_only is True  # fmt: skip
    assert [seed_role(s) for s in (2002, 2200, 2201, 2202, 2203, 2100, 3000)] == [
        "discovery", "numerical", "numerical", "numerical", None, "validation",
        "confirmation"]  # fmt: skip
    assert not any(measures_rayleigh(s) for s in NUMERICAL_SEEDS)


def test_a_regular_run_needs_its_role_and_the_first_blocks_measure_rayleigh() -> None:
    cfg = load_run_config("experiments/study2_short", ["optimizer.lr=1e-5", "seed=3000"])
    with pytest.raises(ValueError, match="study2_short.role"):
        config_hash(cfg)
    plain = load_config("experiments/study2_short", ["study2_short.role=confirmation"])
    rayleigh = load_config("experiments/study2_short_rayleigh", ["study2_short.role=confirmation"])
    a, b = OmegaConf.to_container(plain), OmegaConf.to_container(rayleigh)
    assert b["eval"].pop("rayleigh") == {"enabled": True, "responses": 100}
    assert a == b and rayleigh.experiment == "study2_short"  # the stage's records stay together
    assert [measures_rayleigh(s) for s in (2000, 2100, 3000, 3009, 3010)] == [
        False, False, True, True, False]  # fmt: skip


def test_run_facts_read_the_role_the_rayleigh_setting_and_v0_from_the_records(tmp_path) -> None:
    new, old = tmp_path / "new", tmp_path / "old"
    for run_dir in (new, old):
        run_dir.mkdir()
    rayleigh = "eval: {rayleigh: {enabled: true, responses: 100}}\n"
    (new / "config.yaml").write_text("study2_short: {role: confirmation}\n" + rayleigh)
    stream = {"path": "teacher/m/preconditioner/c309eb585c5e/seed-3000", "sha256": "ab" * 32,
              "count": 250}  # fmt: skip
    streams = {"preconditioner": stream, "training": {}}
    git = {"commit": "c" * 40, "dirty": False}
    (new / "meta.json").write_text(json.dumps({"inputs": {"streams": streams}, "git": git}))
    (old / "config.yaml").write_text("stage_a: {role: discovery}\neval: {}\n")
    (old / "meta.json").write_text(json.dumps({"inputs": {"streams": {"training": {}}}}))
    assert run_facts(new) == {"role": "confirmation", "rayleigh": True, "k_only": False,
                              "commit": "c" * 40, "dirty": False,
                              "preconditioner": {"path": stream["path"],
                                                 "sha256": stream["sha256"]},
                              "config_hash": None}  # fmt: skip
    assert run_facts(old) == {"role": "discovery", "rayleigh": False, "k_only": False,
                              "commit": None, "dirty": None, "preconditioner": None,
                              "config_hash": None}  # fmt: skip
    (new / "meta.json").write_text(json.dumps({"git": git, "config_hash": "abc"}))
    assert run_facts(new)["config_hash"] == "abc"
    numerical = tmp_path / "numerical"
    numerical.mkdir()
    (numerical / "config.yaml").write_text("study2_short: {role: numerical}\n"
                                           "eval: {k_only: true}\n")  # fmt: skip
    assert run_facts(numerical)["k_only"] is True and run_facts(numerical)["role"] == "numerical"


# ------------------------------------------------------------------ the synthetic stage

P = prescribed()
COURSE = "timecourse-"  # the run ids of the synthetic time-course runs (``course``)


def conform(run):
    """A synthetic run as the campaigns make it: the prescribed model, schedule and monitor."""
    if run.arm == FA_CONTROL_ARM:
        instability = {**P["control"]["instability"],
                       "contamination_k": run.spec["instability"]["contamination_k"]}  # fmt: skip
        return dataclasses.replace(run, model=P["model"], protocol_version=3, role="full_control",
                                   spec={**P["control"], "instability": instability},
                                   monitor_spec=P["monitor"])  # fmt: skip
    monitor = {"version": 1} if run.monitor_version == 1 else P["monitor"]
    return dataclasses.replace(run, model=P["model"], protocol_version=2, spec=P["short"],
                               monitor_spec=monitor)  # fmt: skip


def v0(seed: int, batch: int) -> dict:
    return {"path": f"teacher/m/preconditioner/{batch}/seed-{seed}", "sha256": f"v0-{seed}-{batch}"}


def fact(run) -> dict:
    """``run_facts`` of a synthetic run made by its campaign: with the config hash of its
    campaign's config (``prescribed_hash``) when it is one of this stage's regular runs."""
    control_run = run.arm == FA_CONTROL_ARM
    role = "full_control" if control_run else seed_role(run.seed)
    if run.run_id.startswith(COURSE):  # a time-course run (``course``)
        role = TIMECOURSE
    regular = not control_run and run.role is None and role is not None
    return {"role": role, "rayleigh": not control_run and measures_rayleigh(run.seed),
            "k_only": role == "numerical", "commit": RUN_COMMIT, "dirty": False,
            "preconditioner": v0(run.seed, run.batch_size) if run.arm in V0_ARMS else None,
            "config_hash": prescribed_hash(run.arm, run.batch_size, run.lr, run.seed, role)
            if regular else None}  # fmt: skip


def facts_of(runs) -> dict:
    return {run.run_id: fact(run) for run in runs}


def later(arm, batch, lr, seed):
    """A numerical, validation or confirmation run of the world, its K jittered by seed."""
    k = SCALE[(arm, batch)] * (lr / 1e-5) ** 1.5 * (0.98 + 0.01 * (seed % 5))
    return conform(traj(arm, batch, lr, seed, k, mmlu=0.3 * k, factual=0.1 * k))


def course(arm, batch, lr, seed, *, stop: str = "steps", steps: int = 25, scale: float = 1.0):
    """A time-course run of the world: its endpoint run (``later``) evaluated at steps 1, 5 and
    10 too, up to its stop; its K × ``scale`` throughout (1: its endpoint's step 25)."""
    endpoint = later(arm, batch, lr, seed)
    final = endpoint.evaluations[0]
    evaluations = [{**final, "step": t, "condition": f"step-{t}",
                    "K_benign": final["K_benign"] * (scale * t / 25),
                    "K_monitor": final["K_monitor"] * (scale * t / 25)}
                   for t in TIMECOURSE_STEPS
                   if t <= steps and (t < 25 or stop == "steps")]  # fmt: skip
    name = f"{COURSE}{arm}-b{batch}-{lr:g}-s{seed}"
    return dataclasses.replace(endpoint, run_id=name, run_dir=Path("/runs") / name,
                               evaluations=evaluations, spec=P["timecourse"],
                               evaluate_at=TIMECOURSE_STEPS,
                               terminal={**endpoint.terminal, "stop": {"reason": stop},
                                         "steps": steps})  # fmt: skip


SAME_RECORDS = {"metrics.jsonl": [], "monitor.jsonl": [], "update_scalars.jsonl": [],
                "meta.json": {}}  # fmt: skip


@pytest.fixture
def records_alike(monkeypatch):
    """The synthetic runs have no directory: each reads the same records (``_run_records``), so
    that their identity rests on their evaluations and stops; tests/test_study2_short.py checks
    the records themselves. Returns the patch, to apply again after an undo."""

    def alike() -> None:
        monkeypatch.setattr(study2_short, "_run_records", lambda run: SAME_RECORDS)

    alike()
    return alike


def courses(rates, batch, seeds) -> list:
    """The time course's runs of ``seeds``: each arm at its central rate."""
    return [course(arm, batch, sorted(lrs)[1], seed)
            for seed in seeds for arm, lrs in rates.items()]  # fmt: skip


def answered(runs, exhausted=None, verdicts=None) -> dict:
    """``verdicts`` with every numerical check the synthetic stage asks for answered as the
    numerical-check tool would, every rate qualified (``test_study2_short.verdict_of``), each under
    its selection's digest."""
    verdicts = dict(verdicts or {})
    for _ in range(8):
        state = loaded(runs, exhausted=exhausted, numerical=verdicts)
        # The selection's own step: later runs show as unexpected until its check is answered,
        # and the step waits as ``pending`` under an incomplete ``next``.
        chosen = state.get("selection", {})
        step = chosen.get("pending") or chosen.get("next") or state["next"]
        if step["campaign"] != "numerical-check":
            return verdicts
        check = step["check"]
        verdicts[check["verdict"]] = verdict_of(check)
    raise AssertionError("the checks never end")


def loaded(runs, **kwargs) -> dict:
    """``progress`` on the runs as the scripts load them: the time course's split apart once
    (``split_timecourse``)."""
    facts = facts_of(runs)
    regular, timecourse = split_timecourse(runs, facts)
    return progress(regular, THETA0, W_MAX, facts=facts, reuse=reusable, timecourse=timecourse,
                    **kwargs)  # fmt: skip


def stage(runs, exhausted=None, verdicts=None):
    """The stage's progress; its numerical checks read ``verdicts`` (every check the stage asks
    for answered, qualified, by default: ``answered``)."""
    verdicts = answered(runs, exhausted) if verdicts is None else verdicts
    return loaded(runs, exhausted=exhausted, numerical=verdicts)


def make(step) -> list:
    runs = []
    for r in step["runs"]:
        if r["seed"] in (2000, 2001, 2002):
            runs.append(conform(made(r["arm"], r["batch"], r["lr"], r["seed"])))
        else:
            runs.append(later(r["arm"], r["batch"], r["lr"], r["seed"]))
    return runs


def until(runs, campaign: str) -> dict:
    """Make every campaign the stage names until it names ``campaign``."""
    for _ in range(30):
        state = stage(runs)
        if state["next"]["campaign"] == campaign:
            return state
        assert not state["problems"], state["problems"]
        runs += make(state["next"])
    raise AssertionError(f"never reached {campaign}")


def base() -> list:
    return [conform(r) for r in stage_a_reference()]


def blocks_of(rates, batch, seeds):
    return [later(arm, batch, lr, s) for s in seeds for arm, lrs in rates.items() for lr in lrs]


RATES20 = {ADAMW: [1.27e-5, 1.48e-5, 1.74e-5], FA: [1.59e-5, 1.86e-5, 2.17e-5],
           PREWARMED: [1.27e-5, 1.48e-5, 1.74e-5]}  # fmt: skip


def test_the_stage_goes_from_discovery_to_validation_to_confirmation() -> None:
    runs = base()
    assert stage(runs)["next"]["campaign"] == "discovery"
    numerical = until(runs, "numerical")["next"]
    assert (numerical["batch"], numerical["check"], len(numerical["runs"])) == (
        20, numerical_digest(RATES20), 15)  # fmt: skip
    assert numerical["streams"] == {"seeds": [2200, 2201, 2202], "batch": 20,
                                    "training_responses": 500,
                                    "preconditioner_responses": 1000}  # fmt: skip
    state = until(runs, "validation")
    step = state["next"]
    assert (step["batch"], step["U"]) == (20, 8.0) and not step["prewarmed_dropped"]
    assert step["three_arm"] and step["rates"] == RATES20
    assert len(step["runs"]) == 3 * 3 * 12  # three arms, three rates, seeds 2100-2111
    runs += make(step)[:-1]  # one validation run still missing: no verdict yet
    assert stage(runs)["next"]["campaign"] == "validation"
    state = until(runs, "confirmation")
    assert state["validation"]["passes"] and len(state["validation"]["jointly_supported"]) == 12
    assert state["validation"]["arms"] == list(ARMS)  # the three-arm validation
    assert set(state["validation_prewarmed_utility"]) == set(RATES20[PREWARMED])
    assert state["frozen"]["three_arm"] and state["frozen"]["numerical"][0]["state"] == "complete"
    assert not any(b["runs"] for b in state["confirmation"])
    runs += blocks_of(step["rates"], 20, CONFIRMATION_SEEDS[:3])
    runs += blocks_of({ADAMW: step["rates"][ADAMW]}, 20, CONFIRMATION_SEEDS[3:4])
    blocks = stage(runs)["confirmation"]
    assert [b["complete"] for b in blocks[:5]] == [True, True, True, False, False]
    assert all(b["supported"] for b in blocks[:3]) and blocks[3]["runs"] == 3


def test_a_failed_validation_moves_to_the_next_candidate() -> None:
    runs = base()
    state = until(runs, "validation")
    made_runs = make(state["next"])
    # Half the blocks' AdamW-25 runs read far below K*: fewer than 10 jointly supported.
    low = set(VALIDATION_SEEDS[:6])
    runs += [dataclasses.replace(r, evaluations=[{**r.evaluations[0], "K_benign": 1.0}])
             if r.arm == ADAMW and r.seed in low else r for r in made_runs]  # fmt: skip
    state = stage(runs)
    failed = state["selection"]["attempts"][0]
    assert failed["failed_at"] == "validation"
    assert "6 of 12 blocks jointly supported for the three arms" in failed["reasons"][0]
    step = state["next"]  # (7, 20): its new rates' numerical cases first
    assert (step["campaign"], step["batch"], step["U"]) == ("numerical", 20, 7.0)
    state = until(runs, "validation")
    validation = {(r["arm"], r["lr"]) for r in state["next"]["runs"]}
    assert {r["seed"] for r in state["next"]["runs"]} == set(VALIDATION_SEEDS)
    # The validation seeds are reused: only the rates new at U = 7 are run.
    assert not validation & {(arm, lr) for arm, lrs in RATES20.items() for lr in lrs}


def test_records_off_the_campaigns_leave_the_stage_incomplete() -> None:
    runs = base()
    first = make(stage(runs)["next"])[:1]
    ok = stage([*runs, *first])
    assert ok["next"]["campaign"] == "discovery" and not ok["problems"]  # the rest of the round
    own = fact(first[0])
    cases = {
        "role": lambda f: {**f, first[0].run_id: {**own, "role": "validation"}},
        "rayleigh": lambda f: {**f, first[0].run_id: {**own, "rayleigh": True}},
        "k_only": lambda f: {**f, first[0].run_id: {**own, "k_only": True}},
        "unread": lambda f: {k: v for k, v in f.items() if k != first[0].run_id},
    }
    for name, damage in cases.items():
        state = progress([*runs, *first], THETA0, W_MAX, facts=damage(facts_of([*runs, *first])))
        assert state["next"] == {"campaign": "incomplete"}, name
    v1 = dataclasses.replace(first[0], monitor_version=1, monitor_spec={"version": 1})
    assert "read another monitor" in conformance([v1], facts_of([v1]))[0]
    other = dataclasses.replace(first[0], spec={**P["short"], "train_steps": 30})
    assert "departs from the prescribed schedule in ['train_steps']" in conformance(
        [other], facts_of([other]))[0]  # fmt: skip
    protocol = dataclasses.replace(conform(control(5, 1e-5, 0.06, 0.0)), protocol_version=2)
    assert "under protocol 2" in conformance([protocol], facts_of([protocol]))[0]
    # A numerical run must evaluate K only; no other run may.
    numerical = later(ADAMW, 20, 1.27e-5, 2200)
    assert conformance([numerical], facts_of([numerical])) == []
    sealed = {numerical.run_id: {**fact(numerical), "k_only": False}}
    assert "is not K-only on seed 2200" in conformance([numerical], sealed)[0]
    # A run the state machine did not prescribe stops every campaign until it is resolved.
    stray = conform(made(ADAMW, 20, 1.23e-5, 2000))
    state = stage([*runs, stray])
    assert state["next"] == {"campaign": "incomplete"} and "unexpected run" in state["problems"][0]


def test_v0_is_one_preconditioner_stream_per_seed_and_batch() -> None:
    runs = base()
    until(runs, "numerical")  # Stage A's FA grids and the joint discovery's FA and pre-warmed runs
    facts = facts_of(runs)
    assert preconditioner_pairing(runs, facts) == []
    # A pre-warmed run estimated v₀ from another copy of the seed's stream: no conclusion.
    prewarmed = next(r for r in runs if r.arm == PREWARMED and r.seed == 2000)
    other = {
        **facts,
        prewarmed.run_id: {
            **facts[prewarmed.run_id],
            "preconditioner": {**v0(2000, 20), "sha256": "x"},
        },
    }
    found = preconditioner_pairing(runs, other)
    assert len(found) == 1 and "seed 2000, B=20: v₀ from 2 preconditioner streams" in found[0]
    state = progress(runs, THETA0, W_MAX, facts=other)
    assert state["next"] == {"campaign": "incomplete"} and state["problems"] == found
    # A pre-warmed run that records no preconditioner stream is a problem too.
    missing = {**facts, prewarmed.run_id: {**fact(prewarmed), "preconditioner": None}}
    assert "records no preconditioner stream" in preconditioner_pairing([prewarmed], missing)[0]
    # AdamW-25 estimates no v₀, and B = 5's stream is another one.
    assert preconditioner_pairing([later(ADAMW, 20, 1e-5, 2000)], {}) == []
    pair = [later(FA, 5, 1.27e-5, 2000), later(FA, 20, 1.27e-5, 2000)]
    assert preconditioner_pairing(pair, facts_of(pair)) == []


def test_later_runs_must_be_at_the_frozen_rates() -> None:
    runs = base()
    until(runs, "validation")
    # A validation run nothing called for at a batch the selection reached: no conclusion.
    stray = later(FA, 20, 1.3e-5, 2100)
    state = stage([*runs, stray])
    assert state["next"] == {"campaign": "incomplete"}
    assert "unexpected run ('frozen_adam_25', 20, 1.3e-05, 2100)" in state["problems"]
    early = later(FA, 20, 1.86e-5, 3000)  # a confirmation run before the rates are frozen
    state = stage([*runs, early])
    assert state["next"] == {"campaign": "incomplete"}
    assert "before the rates are frozen" in state["problems"][0]
    until(runs, "confirmation")
    state = stage([*runs, later(FA, 20, 1.3e-5, 3000)])
    assert state["next"] == {"campaign": "incomplete"}
    assert "outside the frozen rates" in state["problems"][0]


def test_primary_only_fallback_reaches_confirmation_with_the_primary_arms() -> None:
    # The pre-warmed arm never brackets K*: every candidate fails it, and the fallback freezes
    # the primary arms at (8, 20); their confirmation blocks hold no pre-warmed run.
    runs = [*base()]
    real = SCALE.copy()
    try:
        for batch in (5, 20):
            SCALE[(PREWARMED, batch)] = 0.3
        state = until(runs, "confirmation")
    finally:
        SCALE.update(real)
    frozen = state["frozen"]
    assert (frozen["batch"], frozen["U"], frozen["three_arm"]) == (20, 8.0, False)
    assert frozen["prewarmed_dropped"] and set(frozen["rates"]) == {ADAMW, FA}
    assert state["validation"]["arms"] == [ADAMW, FA]
    runs += blocks_of(frozen["rates"], 20, CONFIRMATION_SEEDS[:2])
    later_state = stage(runs)
    blocks = later_state["confirmation"]
    assert [(b["complete"], b["expected"]) for b in blocks[:3]] == [(True, 6), (True, 6),
                                                                   (False, 6)]  # fmt: skip
    assert set(blocks[0]["supports"]) == {ADAMW, FA}
    # No mechanism cohort: no block is jointly supported, and amendment 3's looks pace it.
    assert blocks[0]["supported"] and not any(b["three_arm_supported"] for b in blocks)
    assert later_state["mechanism"]["applies"] is False
    guard = confirmation_guard(blocks, 3, 37, three_arm=False)
    assert "room for 34 before look 1" in guard["problems"][0]
    assert not [r for r in runs if r.arm == PREWARMED and r.seed not in (2000, 2001, 2002)]


# ------------------------------------------------------------------ runs given up


def journal(
    key_cell: str,
    seed: int,
    overrides: list[str],
    *,
    attempts: int,
    recovered: bool,
    last_outcome: str = "technical",
    last_ended: bool = True,
    config: str = "experiments/study2_short",
) -> list:
    """The launcher's records of one run's attempts; with ``last_ended`` False the last one has
    no end record (still running, or lost with its Pod)."""
    records = []
    if recovered:
        records.append({"event": "recovery", "experiment": "study2_short", "cell": key_cell,
                        "reason": "fixed", "runs": [f"{key_cell}/s{seed}"]})  # fmt: skip
    for attempt in range(1, attempts + 1):
        run_id = f"run-{key_cell}-{seed}-a{attempt}"
        records.append({"event": "start", "run": f"{key_cell}/s{seed}", "attempt": attempt,
                        "run_id": run_id, "experiment": "study2_short", "config_hash": "h",
                        "command": ["/src/scripts/train.py", config, *overrides,
                                    f"seed={seed}"]})  # fmt: skip
        if attempt < attempts or last_ended:
            outcome = last_outcome if attempt == attempts else "technical"
            records.append({"event": "end", "run": f"{key_cell}/s{seed}", "attempt": attempt,
                            "run_id": run_id, "outcome": outcome})  # fmt: skip
    return records


GIVEN_UP = (
    "adamw_25-b5-1.37e-05-blocks-1-2-rayleigh",
    ["train.batch_size=5", "optimizer.lr=1.37e-05", "study2_short.role=confirmation"],
)


def test_the_journal_tells_which_runs_were_given_up_after_their_recovery_round() -> None:
    cell, overrides = GIVEN_UP
    config = "experiments/study2_short_rayleigh"
    records = journal(cell, 3000, overrides, attempts=2 * MAX_ATTEMPTS, recovered=True,
                      config=config)  # fmt: skip
    assert exhausted_runs(records) == {(ADAMW, 5, 1.37e-5, 3000): f"{cell}/s3000"}
    assert issued_ranges(records) == {(1, 2)}
    for kwargs in ({"attempts": 2 * MAX_ATTEMPTS, "recovered": False},  # no recovery round yet
                   {"attempts": 2 * MAX_ATTEMPTS - 1, "recovered": True},  # attempts left
                   {"attempts": 2 * MAX_ATTEMPTS, "recovered": True,
                    "last_outcome": "done"}):  # fmt: skip
        assert exhausted_runs(journal(cell, 3000, overrides, config=config, **kwargs)) == {}


def test_a_sixth_attempt_still_running_is_not_given_up_until_it_ends_or_is_abandoned(
    tmp_path,
) -> None:
    cell, overrides = GIVEN_UP
    config = "experiments/study2_short_rayleigh"
    running = journal(cell, 3000, overrides, attempts=2 * MAX_ATTEMPTS, recovered=True,
                      last_ended=False, config=config)  # fmt: skip
    sixth = running[-1]["run_id"]
    assert running[-1]["attempt"] == 6 and running[-1]["event"] == "start"
    assert exhausted_runs(running) == {}  # pending: it may still be running
    # An earlier attempt without its end record was settled by the next start.
    earlier = [
        r
        for r in journal(
            cell, 3000, overrides, attempts=2 * MAX_ATTEMPTS, recovered=True, config=config
        )  # fmt: skip
        if not (r["event"] == "end" and r["attempt"] == 2)
    ]
    assert exhausted_runs(earlier) == {(ADAMW, 5, 1.37e-5, 3000): f"{cell}/s3000"}
    # The operator's abandonment record gives the sixth attempt up.
    path = tmp_path / "abandoned.json"
    path.write_text(
        json.dumps(
            {
                "what": "attempts lost with their Pod",
                "abandoned": [{"run_id": sixth, "reason": "Pod terminated"}],
            }
        )
    )
    abandoned = read_abandonment(path)
    assert abandoned == {sixth: "Pod terminated"}
    found = exhausted_runs(running, abandoned=abandoned)
    assert found == {(ADAMW, 5, 1.37e-5, 3000): f"{cell}/s3000"}
    with pytest.raises(ValueError, match="no start in the launch journal"):
        exhausted_runs(running, abandoned={"another-run": "typo"})
    ended = journal(cell, 3000, overrides, attempts=2 * MAX_ATTEMPTS, recovered=True,
                    config=config)  # fmt: skip
    with pytest.raises(ValueError, match="ended .technical.: it is not unresolved"):
        exhausted_runs(ended, abandoned={ended[-1]["run_id"]: "already ended"})


@pytest.mark.parametrize(
    "record",
    [[], {"abandoned": {}}, {"abandoned": [{"run_id": "a"}]},
     {"abandoned": [{"run_id": "a", "reason": " "}]}, {"abandoned": [{"reason": "x"}]},
     {"abandoned": [{"run_id": "a", "reason": "x"}, {"run_id": "a", "reason": "y"}]}],
)  # fmt: skip
def test_a_malformed_abandonment_record_is_refused(tmp_path, record) -> None:
    path = tmp_path / "abandoned.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="abandon|run_id"):
        read_abandonment(path)


def test_a_run_given_up_is_a_technical_failure_of_its_block_and_the_looks_go_past_it() -> None:
    runs = base()
    state = until(runs, "confirmation")
    rates = state["next"]["rates"]
    given_up = (ADAMW, 20, 1.74e-5, 3000)  # the bracket's upper end in block 1
    runs += [r for r in blocks_of(rates, 20, CONFIRMATION_SEEDS[:2])
             if (r.arm, r.batch_size, r.lr, r.seed) != given_up]  # fmt: skip
    waiting = stage(runs)["confirmation"][0]
    assert not waiting["complete"] and waiting["runs"] == 8  # without the journal it waits
    state = stage(runs, exhausted={given_up: "adamw_25-…/s3000"})
    block = state["confirmation"][0]
    assert (
        block["complete"] and not block["supported"] and block["exhausted"] == ["adamw_25-…/s3000"]
    )
    assert block["supports"][ADAMW]["cause"] == "technical"
    assert block["supports"][ADAMW]["failures"][0]["reason"] == "no run"
    labels = {r.run_id: [{"condition": "step-25", "step": 25, "prompts": 500, "harmful": 50,
                          "missing": 0}] for r in runs}  # fmt: skip
    entries = look_blocks(state["confirmation"], labels, state["frozen"]["K_star"])
    assert [(e["complete"], e["supported"]) for e in entries[:3]] == [
        (True, False), (True, True), (False, False)]  # fmt: skip
    found = looks(entries, FIXATION["levels"], **LOOK_SETTINGS)
    assert (found["waiting"]["started"], found["waiting"]["supported"]) == (2, 1)
    assert state["exhausted"] == {"adamw_25-…/s3000": list(given_up)}


def test_a_validation_run_given_up_lets_the_verdict_be_read() -> None:
    runs = base()
    step = until(runs, "validation")["next"]
    made_runs = make(step)
    lost = next(r for r in made_runs if (r.arm, r.lr, r.seed) == (FA, 2.17e-5, 2100))
    runs += [r for r in made_runs if r is not lost]
    assert stage(runs)["next"]["campaign"] == "validation"
    key = (lost.arm, lost.batch_size, lost.lr, lost.seed)
    state = stage(runs, exhausted={key: "lost/s2100"})
    assert state["next"]["campaign"] == "confirmation"  # 11 of 12 blocks: still ≥ 10
    assert state["validation"]["causes"][FA]["failure_kinds"] == {"technical": 1}


# ------------------------------------------------------------------ the confirmation guard


def plan(started: int = 0, complete: int = 0, supported: int = 0, joint: int = 0) -> list[dict]:
    """80 blocks: the first ``started`` have runs, the first ``complete`` of them are complete,
    the first ``supported`` of those supported, the first ``joint`` of the complete ones jointly
    supported (all three arms). Each complete block's supports: AdamW-25 unsupported (width)
    beyond ``supported``, the pre-warmed arm unsupported (bracketing) beyond ``joint``."""

    def supports(i: int) -> dict:
        if i > complete:
            return {}
        return {ADAMW: {"supported": i <= supported, "cause": None if i <= supported else "width"},
                FA: {"supported": True, "cause": None},
                PREWARMED: {"supported": i <= joint,
                            "cause": None if i <= joint else "bracketing"}}  # fmt: skip

    return [{"block": i, "runs": 9 if i <= started else 0, "complete": i <= complete,
             "supported": i <= supported, "three_arm_supported": i <= min(joint, complete),
             "supports": supports(i)} for i in range(1, 81)]  # fmt: skip


def refused(blocks, first, last, *, three_arm=False, **kw) -> list[str]:
    return confirmation_guard(blocks, first, last, three_arm=three_arm, **kw)["problems"]


def test_confirmation_ranges_run_in_order_without_overlaps() -> None:
    fresh = plan()
    assert refused(fresh, 1, 36) == []  # look 1 at 36 supported blocks
    assert "room for 36 before look 1" in refused(fresh, 1, 37)[0]
    assert "starts at block 1" in refused(fresh, 2, 10)[0]
    running = plan(started=10)  # blocks 1-10 still running: each may be supported
    assert refused(running, 11, 36) == []
    assert "room for 26" in refused(running, 11, 37)[0]
    assert "have runs already" in refused(running, 5, 15)[0]
    assert refused(running, 1, 10, issued={(1, 10)}) == []  # completing an issued range
    assert "have runs already" in refused(running, 1, 10, issued={(1, 12)})[0]
    assert "1 <= FIRST <= LAST <= 80" in refused(fresh, 75, 81)[0]


def test_a_due_look_holds_the_blocks_until_the_operator_has_read_it() -> None:
    due = plan(started=40, complete=40, supported=36)  # 4 unsupported blocks: s = 36
    assert "look 1 is due" in refused(due, 41, 41)[0]
    assert refused(due, 41, 64, after_look=1) == []  # room 60 - 36 = 24 before look 2
    assert "room for 24 before look 2" in refused(due, 41, 65, after_look=1)[0]
    early = plan(started=40, complete=40, supported=30)
    assert "look 1 needs 36" in refused(early, 41, 41, after_look=1)[0]
    last = plan(started=70, complete=70, supported=60)
    assert "look 2 is the last" in refused(last, 71, 71, after_look=2)[0]
    in_flight = plan(started=36, complete=20, supported=20)  # 16 running: they may complete it
    assert "room for 0 before look 1" in refused(in_flight, 37, 37)[0]
    counts = confirmation_guard(due, 41, 64, three_arm=False, after_look=1)
    assert (counts["first_open"], counts["supported_in_order"], counts["possibly_supported"]) == (
        41, 36, 36)  # fmt: skip


def test_cohort_ends_at_60_or_cap() -> None:
    """A three-arm selection: ranges up to the mechanism cohort's 60 jointly supported blocks,
    no pause at the primary's looks, none after the 60th or at the cap of 80, ``--after-look``
    refused; a two-arm selection keeps amendment 3's look pauses."""
    fresh = plan()
    assert refused(fresh, 1, 60, three_arm=True) == []
    assert "room for 60 before the mechanism cohort's 60" in refused(fresh, 1, 61,
                                                                     three_arm=True)[0]  # fmt: skip
    running = plan(started=10)  # still running: each may be jointly supported
    assert refused(running, 11, 60, three_arm=True) == []
    assert "room for 50" in refused(running, 11, 61, three_arm=True)[0]
    # A block whose pre-warmed arm is lost gives its room back; the primary's due look 1 (36
    # supported blocks complete in order) holds nothing.
    lost = plan(started=40, complete=40, supported=40, joint=30)
    counts = confirmation_guard(lost, 41, 70, three_arm=True)
    assert counts["problems"] == []
    assert (counts["jointly_supported_in_order"], counts["possibly_jointly_supported"],
            counts["room"], counts["supported_in_order"]) == (30, 30, 30, 40)  # fmt: skip
    assert "room for 30" in refused(lost, 41, 71, three_arm=True)[0]
    state = mechanism_state(lost, three_arm=True)
    assert (state["ended"], state["jointly_supported_in_order"]) == (False, 30)
    assert state["started"] == 40
    assert state["attrition"] == {"blocks": 10, "unsupported_arms": {PREWARMED: 10},
                                  "causes": {ADAMW: {}, FA: {}, PREWARMED: {"bracketing": 10}},
                                  "technical": 0, "primary_supported": 10,
                                  "lost_blocks": list(range(31, 41))}  # fmt: skip
    err = refused(lost, 41, 41, three_arm=True, after_look=1)
    assert "--after-look 1: a three-arm confirmation does not pause" in err[0]
    in_flight = plan(started=60, complete=20, supported=20, joint=20)
    assert "room for 0" in refused(in_flight, 61, 61, three_arm=True)[0]
    # 60 jointly supported blocks complete in order: the cohort and the confirmation end.
    done = plan(started=66, complete=66, supported=66, joint=60)
    assert "has its 60 jointly supported blocks" in refused(done, 67, 67, three_arm=True)[0]
    state = mechanism_state(done, three_arm=True)
    assert (state["ended"], state["ended_by"], state["last_block"]) == (True, "target", 60)
    assert state["cohort"] == list(range(1, 61)) and state["attrition"]["blocks"] == 0
    # A window with lost blocks ends at the 60th jointly supported one.
    gaps = [{**b, "three_arm_supported": b["block"] % 10 != 0 and b["complete"],
             "supports": {**b["supports"], PREWARMED: {"supported": False, "cause": "technical"}}
             if b["block"] % 10 == 0 else b["supports"]}
            for b in plan(started=70, complete=70, supported=70, joint=70)]  # fmt: skip
    state = mechanism_state(gaps, three_arm=True)
    assert (state["ended_by"], state["last_block"], len(state["cohort"])) == ("target", 66, 60)
    assert state["attrition"]["lost_blocks"] == [10, 20, 30, 40, 50, 60]
    assert state["attrition"]["technical"] == 6
    # The cap: all 80 started and complete with 50 jointly supported ends it with fewer.
    capped = plan(started=80, complete=80, supported=80, joint=50)
    state = mechanism_state(capped, three_arm=True)
    assert (state["ended"], state["ended_by"], state["last_block"]) == (True, "cap", 80)
    assert len(state["cohort"]) == 50 and state["minimum_met"]
    assert "have runs already" in refused(capped, 80, 80, three_arm=True)[0]
    tail = plan(started=80, complete=70, supported=70, joint=50)  # 71-80 still running
    assert not mechanism_state(tail, three_arm=True)["ended"]
    assert refused(tail, 71, 80, three_arm=True, issued={(71, 80)}) == []  # completing it
    # A two-arm selection: amendment 3's guard, its look pauses and ``--after-look``.
    assert "room for 36 before look 1" in refused(fresh, 1, 37)[0]
    due = plan(started=40, complete=40, supported=36)
    assert "look 1 is due" in refused(due, 41, 41)[0]
    assert refused(due, 41, 64, after_look=1) == []
    assert mechanism_state(due, three_arm=False)["applies"] is False
    assert "jointly_supported_in_order" not in confirmation_guard(due, 41, 41, three_arm=False)


def test_issued_ranges_count_as_started() -> None:
    """The review of Task 3, M1: a block of an issued range counts as started before the launcher
    reaches it (in the first open block, p and p₃), so no range is issued over it and no block
    runs twice; re-issuing exactly an issued range is still allowed. M2: with three arms,
    ``--after-look`` is refused on a re-issue too."""
    # Three arms: 1-40 done, 41-60 issued, the launcher at block 50.
    reached = plan(started=50, complete=40, supported=40, joint=40)
    issued = {(1, 40), (41, 60)}
    found = confirmation_guard(reached, 51, 60, three_arm=True, issued=issued)
    assert "have runs already or are in a range issued before" in found["problems"][0]
    assert (found["first_open"], found["possibly_jointly_supported"], found["room"]) == (61, 60, 0)
    assert "room for 0" in refused(reached, 61, 61, three_arm=True, issued=issued)[0]
    assert refused(reached, 41, 60, three_arm=True, issued=issued) == []  # completing it
    without = confirmation_guard(reached, 51, 60, three_arm=True)  # what the journal adds
    assert without["problems"] == [] and without["first_open"] == 51
    err = refused(reached, 41, 60, three_arm=True, issued=issued, after_look=1)
    assert len(err) == 1 and "--after-look 1: a three-arm confirmation does not pause" in err[0]
    # Two arms: 1-36 issued, 20 with runs.
    early = plan(started=20, complete=20, supported=20)
    assert "are in a range issued before" in refused(early, 21, 36, issued={(1, 36)})[0]
    counts = confirmation_guard(early, 37, 37, three_arm=False, issued={(1, 36)})
    assert (counts["first_open"], counts["possibly_supported"], counts["room"]) == (37, 36, 0)
    assert "room for 0 before look 1" in counts["problems"][0]
    assert refused(early, 1, 36, issued={(1, 36)}) == []


def test_the_room_stops_at_the_cap(monkeypatch) -> None:
    """The review of Task 3, M3: with 75 blocks started, the room is 5 (the cap), not 60 − p₃,
    in the guard, the mechanism state and the summary's status."""
    blocks = plan(started=75, complete=75, supported=75, joint=30)
    counts = confirmation_guard(blocks, 76, 80, three_arm=True)
    assert counts["problems"] == [] and counts["possibly_jointly_supported"] == 30
    assert counts["room"] == 5 and mechanism_state(blocks, three_arm=True)["room"] == 5
    assert "1 <= FIRST <= LAST <= 80" in refused(blocks, 76, 81, three_arm=True)[0]
    cohort = mechanism_state(blocks, three_arm=True)
    state = {"next": {"campaign": "confirmation"}, "frozen": {"three_arm": True},
             "confirmation": blocks, "mechanism": cohort}  # fmt: skip
    shown = summary_module().status(state)
    assert shown["confirmation"]["room"] == 5 and shown["mechanism"]["room"] == 5
    # The status counts the launcher's issued ranges as the guard does.
    issued = summary_module().status(state, {(76, 80)})["confirmation"]
    assert (issued["first_block_without_runs"], issued["next_range_starts_at"]) == (76, 81)
    assert issued["room"] == 0


# ------------------------------------------------------------------ the fixation record


def test_the_fixation_record_needs_every_value(tmp_path) -> None:
    path = tmp_path / "fixation.json"
    path.write_text(json.dumps(FIXATION))
    assert read_fixation(path) == FIXATION
    assert set(FIXATION) == set(FIXATION_KEYS)
    for key in FIXATION:
        path.write_text(json.dumps({k: v for k, v in FIXATION.items() if k != key}))
        with pytest.raises(ValueError, match=key):
            read_fixation(path)
    for bad in ({"levels": [0.05, 0.05]}, {"levels": [0.99]}, {"w_max": 0}, {"se_floor": -1},
                {"bias_allowance": None}, {"w_max": True}):  # fmt: skip
        path.write_text(json.dumps({**FIXATION, **bad}))
        with pytest.raises(ValueError):
            read_fixation(path)
    path.write_text(json.dumps({**FIXATION, "interval": "t"}))  # the t interval is admitted
    assert read_fixation(path)["interval"] == "t"
    path.write_text(json.dumps({**FIXATION, "equivalence_min_supported": 0}))  # 0: no minimum
    assert read_fixation(path)["equivalence_min_supported"] == 0
    path.write_text(json.dumps({**FIXATION, "direction_min_supported": 0}))
    assert read_fixation(path)["direction_min_supported"] == 0
    path.write_text(json.dumps({**FIXATION, "secondary_interval": "bootstrap_t",
                                "secondary_status": "exploratory"}))  # fmt: skip
    found = read_fixation(path)
    assert (found["secondary_interval"], found["secondary_status"]) == ("bootstrap_t",
                                                                        "exploratory")  # fmt: skip


@pytest.mark.parametrize(
    "key, value",
    [("interval", "percentile"), ("interval", None), ("interval", ["bootstrap_t"]),
     ("bootstrap_resamples", 999), ("bootstrap_resamples", 2000.0),
     ("bootstrap_resamples", "2000"), ("bootstrap_resamples", True), ("bootstrap_resamples", None),
     ("bootstrap_seed", 1.5), ("bootstrap_seed", "7"), ("bootstrap_seed", None),
     ("bootstrap_seed", False), ("looks", [24, 50]), ("looks", [36]), ("looks", [36.0, 60.0]),
     ("looks", "36,60"), ("looks", None), ("cap", 79), ("cap", 80.0), ("cap", None),
     ("equivalence_min_supported", -1), ("equivalence_min_supported", 36.5),
     ("equivalence_min_supported", True), ("equivalence_min_supported", None),
     ("direction_min_supported", -1), ("direction_min_supported", 10.0),
     ("direction_min_supported", "10"), ("direction_min_supported", None),
     ("secondary_interval", "percentile"), ("secondary_interval", None),
     ("secondary_interval", ["t"]), ("secondary_status", "primary"),
     ("secondary_status", None), ("secondary_status", "Secondary")],
)  # fmt: skip
def test_the_fixation_record_refuses_a_malformed_new_key(tmp_path, key, value) -> None:
    path = tmp_path / "fixation.json"
    path.write_text(json.dumps({**FIXATION, key: value}))
    with pytest.raises(ValueError, match=f"{key} "):
        read_fixation(path)


def test_fixation_requires_the_mechanism_key(tmp_path) -> None:
    """The ``mechanism`` key (amendment 4, section 3): required, its fixed values equal to the
    code's, the level in [0.95, 1), integer resamples ≥ 1000 and an integer seed; the tracked
    record reads, with the values Task 3 wrote."""
    tracked = read_fixation(REPO_ROOT / "docs" / "study2-short-fixation.json")
    assert tracked["mechanism"] == {"target": 60, "cap": 80, "minimum": 36, "zero_se": 1e-12,
                                    "comparison_level": 0.975, "effects": [0.05, 0.025],
                                    "level": 0.95, "resamples": 10000,
                                    "seed": 20261020}  # fmt: skip
    fixed = study2_short.MECHANISM_FIXED
    assert fixed == {
        "target": study2_short.MECHANISM_TARGET, "cap": study2_short.CAP,
        "minimum": study2_short.MECHANISM_MINIMUM, "zero_se": study2_short.ZERO_SE,
        "comparison_level": study2_short.COMPARISON_LEVEL,
        "effects": list(study2_short.MECHANISM_EFFECTS)}  # fmt: skip
    assert (study2_short.MECHANISM_TARGET, study2_short.MECHANISM_CAP,
            study2_short.MECHANISM_MINIMUM, study2_short.COMPARISON_LEVEL,
            study2_short.MECHANISM_EFFECTS) == (60, 80, 36, 0.975, (0.05, 0.025))  # fmt: skip
    path = tmp_path / "fixation.json"
    path.write_text(json.dumps({k: v for k, v in FIXATION.items() if k != "mechanism"}))
    with pytest.raises(ValueError, match=r"lacks \['mechanism'\]"):
        read_fixation(path)
    mechanism = FIXATION["mechanism"]
    bad = [({"target": 59}, "mechanism target 59"), ({"target": 60.0}, "mechanism target"),
           ({"cap": 79}, "mechanism cap 79"), ({"minimum": 35}, "mechanism minimum 35"),
           ({"zero_se": 1e-11}, "mechanism zero_se"),
           ({"comparison_level": 0.95}, "mechanism comparison_level"),
           ({"effects": [0.05, 0.02]}, "mechanism effects"),
           ({"effects": [0.05]}, "mechanism effects"), ({"effects": "0.05"}, "mechanism effects"),
           ({"level": 0.94}, "mechanism level"), ({"level": 1.0}, "mechanism level"),
           ({"level": "0.95"}, "mechanism level"), ({"level": True}, "mechanism level"),
           ({"resamples": 999}, "mechanism resamples"),
           ({"resamples": 10000.0}, "mechanism resamples"),
           ({"seed": 1.5}, "mechanism seed"), ({"seed": None}, "mechanism seed"),
           ({"seed": False}, "mechanism seed"), ({"other": 1}, "does not know")]  # fmt: skip
    for change, message in bad:
        path.write_text(json.dumps({**FIXATION, "mechanism": {**mechanism, **change}}))
        with pytest.raises(ValueError, match=message):
            read_fixation(path)
    for key in mechanism:
        path.write_text(json.dumps({**FIXATION, "mechanism": {k: v for k, v in mechanism.items()
                                                              if k != key}}))  # fmt: skip
        with pytest.raises(ValueError, match=rf"mechanism lacks \['{key}'\]"):
            read_fixation(path)
    for value in ([mechanism], None, "mechanism"):
        path.write_text(json.dumps({**FIXATION, "mechanism": value}))
        with pytest.raises(ValueError, match="is not an object"):
            read_fixation(path)
    for level in (0.95, 0.9876):  # the simulation sets it in [0.95, 1)
        path.write_text(json.dumps({**FIXATION, "mechanism": {**mechanism, "level": level}}))
        assert read_fixation(path)["mechanism"]["level"] == level


# ------------------------------------------------------------------ the looks' blocks


def rows(harmful: int, missing: int = 0) -> list[dict]:
    return [{"condition": "step-25", "step": 25, "prompts": 500, "harmful": harmful,
             "missing": missing}]  # fmt: skip


def confirmed(runs: list) -> dict:
    state = until(runs, "confirmation")
    runs += blocks_of(state["next"]["rates"], state["next"]["batch"], CONFIRMATION_SEEDS[:2])
    return stage(runs)


def test_look_blocks_wait_for_labels_and_take_delta_from_the_brackets() -> None:
    state = confirmed(base())
    blocks, kstar = state["confirmation"], state["frozen"]["K_star"]
    assert [b["complete"] for b in look_blocks(blocks, {}, kstar)[:3]] == [False, False, False]
    assert look_blocks(blocks, {}, kstar)[0]["unlabeled"]
    brackets = {arm: s["bracket"] for arm, s in blocks[0]["supports"].items()}
    labels = {b[side]["run"]: rows(100 if arm == ADAMW else 50, missing=5)
              for arm, b in brackets.items() for side in ("lo", "hi")}  # fmt: skip
    first = look_blocks(blocks, labels, kstar)[0]
    assert first["complete"] and first["supported"]
    assert first["delta"] == pytest.approx(0.1)  # 0.2 against 0.1 at both ends
    # 5 of 500 missing at each end of either arm: the worst case moves by 1 pp either way.
    assert first["delta_missing_low"] == pytest.approx(0.09)
    assert first["delta_missing_high"] == pytest.approx(0.11)
    assert first["delta_all_missing_harmful"] == pytest.approx(0.1)
    assert first["missing"] == {ADAMW: 10, FA: 10, PREWARMED: 10}
    assert set(first["h"]) == {ADAMW, FA, PREWARMED}
    # Without the pre-warmed arm's labels the block still waits: its three arms are labeled
    # together.
    partial = {k: v for k, v in labels.items()
               if k not in {brackets[PREWARMED][s]["run"] for s in ("lo", "hi")}}  # fmt: skip
    assert not look_blocks(blocks, partial, kstar)[0]["complete"]


def synthetic_harm(run) -> list[dict]:
    """Label rows of a confirmation run, the same at every rate of its arm in its block: the
    primary arms' harm within 0.4 pp of each other around 10 % (Δ balanced over every three
    seeds, so look 1 concludes equivalence), the pre-warmed arm's 2 pp above, a missing label on
    odd seeds."""
    harmful = {ADAMW: 50 + run.seed % 3, FA: 50 + (run.seed + 1) % 3,
               PREWARMED: 60 + run.seed % 2}[run.arm]  # fmt: skip
    return rows(harmful, missing=run.seed % 2)


def test_continuation_after_primary_stop(monkeypatch) -> None:
    """The primary stops at look 1 (36 blocks); the confirmation goes on for the mechanism
    cohort. Blocks 37-80 leave the decision record byte for byte as it was (the looks, the
    sensitivities, the utility family), and are reported as a descriptive continuation; the
    cohort ends at its 60th jointly supported block, where the final analysis runs once."""
    runs = base()
    rates = until(runs, "confirmation")["next"]["rates"]
    verdicts = answered(runs)
    runs += blocks_of(rates, 20, CONFIRMATION_SEEDS[:36])
    early = loaded(runs, numerical=verdicts)
    kstar = early["frozen"]["K_star"]
    labels = {r.run_id: synthetic_harm(r) for r in runs if r.seed in CONFIRMATION_SEEDS}
    module = summary_module()
    course_reads = []
    monkeypatch.setattr(
        module,
        "read_timecourse_labels",
        lambda state, courses, judge: course_reads.append(judge) or ({}, {}),
    )
    found = module.compute_looks(early, runs, labels, FIXATION, THETA0)
    primary = found["looks"]
    assert primary["stopped"] and primary["stopping_look"] == 1
    assert "equivalence" in primary["looks"][0]["readings"]
    assert found["continuation"]["later_blocks"] == []
    # The mechanism cohort continues: 36 of 60, the guard gives room for 24 more, no pause.
    cohort = early["mechanism"]
    assert (cohort["ended"], cohort["room"]) == (False, 24)
    assert cohort["jointly_supported_in_order"] == 36
    assert confirmation_guard(early["confirmation"], 37, 60, three_arm=True)["problems"] == []
    assert module.status(early)["mechanism"] == cohort
    final = module.final_analysis(early, runs, [], labels, FIXATION, found, "wildguard", {})
    assert final == {"available": False, "reason": "not yet: the mechanism cohort has not ended "
                     "(36 of 60 jointly supported blocks complete in order, 36 of 80 blocks "
                     "started)"}  # fmt: skip
    assert course_reads == []  # the time course stays sealed
    runs += blocks_of(rates, 20, CONFIRMATION_SEEDS[36:])
    late = loaded(runs, numerical=verdicts)
    labels = {r.run_id: synthetic_harm(r) for r in runs if r.seed in CONFIRMATION_SEEDS}
    later = module.compute_looks(late, runs, labels, FIXATION, THETA0)
    for key in ("looks", "sensitivities", "utility", "missing_labels"):
        assert json.dumps(later[key], sort_keys=True) == json.dumps(found[key], sort_keys=True)
    entries = look_blocks(late["confirmation"], labels, kstar)
    assert json.dumps(looks(entries, FIXATION["levels"], **LOOK_SETTINGS), sort_keys=True) == (
        json.dumps(primary, sort_keys=True))  # fmt: skip
    continued = later["continuation"]
    assert continued["read_blocks"] == 36 and continued["later_blocks"] == list(range(37, 81))
    assert continued["later"]["n"] == 44 and continued["every_supported"]["n"] == 80
    # The cohort ended at block 60; the final analysis, once.
    cohort = late["mechanism"]
    assert (cohort["ended"], cohort["ended_by"], cohort["last_block"]) == (True, "target", 60)
    assert cohort["cohort"] == list(range(1, 61))
    # Had blocks 61-80 not been started, none would start now.
    unstarted = [*late["confirmation"][:60],
                 *({**b, "runs": 0, "complete": False, "supported": False,
                    "three_arm_supported": False, "supports": {}}
                   for b in late["confirmation"][60:])]  # fmt: skip
    assert "has its 60 jointly supported blocks" in confirmation_guard(
        unstarted, 61, 61, three_arm=True)["problems"][0]  # fmt: skip
    assert mechanism_state(unstarted, three_arm=True) == {
        **cohort,
        "started": 60,
        "complete_in_order": 60,
        "possibly_jointly_supported": 60,
    }
    final = module.final_analysis(late, runs, [], labels, FIXATION, later, "wildguard", {})
    assert final["available"] and course_reads == ["wildguard"]
    assert final["timecourse"]["labels_read"] and final["timecourse"]["counts"]["missing"] == 30
    analysis = final["mechanism"]
    assert analysis == mechanism_analysis(
        mechanism_blocks(late["confirmation"][:60], labels, kstar), FIXATION["mechanism"]
    )
    assert analysis["cohort"]["blocks"] == list(range(1, 61)) and analysis["minimum_met"]
    c1 = [(60 + s % 2 - 50 - (s + 1) % 3) / 500 for s in CONFIRMATION_SEEDS[:60]]
    assert analysis["C1"]["mean"] == pytest.approx(statistics.fmean(c1), abs=1e-12)
    assert analysis["readings"]["C1"]["readings"] == ["positive", "bounded"]
    assert analysis["readings"]["C2"]["readings"] == ["negative", "bounded"]
    assert analysis["sum_check"]["holds"]
    # The pre-warmed arm's harm and missing labels: in the final analysis, nowhere before.
    assert not any(PREWARMED in b["h"] or PREWARMED in b.get("missing", {})
                   for b in later["blocks"])  # fmt: skip
    assert set(later["missing_labels"]) == {ADAMW, FA}
    assert analysis["missing_labels"] == {ADAMW: 60, FA: 60, PREWARMED: 60}  # 30 odd seeds × 2
    assert analysis["blocks"][0]["h"][PREWARMED] == pytest.approx(60 / 500)
    assert analysis["blocks"][1]["missing"] == {ADAMW: 2, FA: 2, PREWARMED: 2}
    # The high bounds count each missing label as harmful (a 1/500 shift on odd seeds).
    blocks = mechanism_blocks(late["confirmation"][:2], labels, kstar)
    assert blocks[1]["h_missing_harmful"][FA] == pytest.approx(blocks[1]["h"][FA] + 1 / 500)
    assert blocks[0]["h_missing_harmful"] == blocks[0]["h"]


def test_progress_reads_no_harm(monkeypatch) -> None:
    """``progress``, the guard and the mechanism state read K and support only: every entry
    point that reads a label raises, K_harmful unknown changes nothing, and the summary's status
    and its final analysis before the cohort's end read nothing either."""
    runs = base()
    rates = until(runs, "confirmation")["next"]["rates"]
    verdicts = answered(runs)
    block_runs = blocks_of(rates, 20, CONFIRMATION_SEEDS[:4])
    module = summary_module()

    def unread(*args, **kwargs):
        raise AssertionError("a label was read")

    readers = ("harm", "interpolate", "arm_harm", "block_difference", "look_blocks", "_step_row",
               "_endpoint_row", "mechanism_blocks", "mechanism_analysis")  # fmt: skip
    for name in readers:
        monkeypatch.setattr(study2_short, name, unread)
    for name in ("look_blocks", "mechanism_blocks", "mechanism_analysis", "read_endpoint_labels",
                 "read_timecourse_labels", "label_path"):  # fmt: skip
        monkeypatch.setattr(module, name, unread)
    monkeypatch.setattr("zero_field_refusal.analysis.study2_labels.read_endpoint_labels", unread)
    state = loaded([*runs, *block_runs], numerical=verdicts)
    blind = [dataclasses.replace(r, evaluations=[{**e, "K_harmful": math.nan}
                                                 for e in r.evaluations])
             for r in block_runs]  # fmt: skip
    other = loaded([*runs, *blind], numerical=verdicts)
    assert other["confirmation"] == state["confirmation"]
    assert other["mechanism"] == state["mechanism"]
    assert state["mechanism"]["jointly_supported_in_order"] == 4
    assert [b["three_arm_supported"] for b in state["confirmation"][:5]] == [True] * 4 + [False]
    for three_arm in (True, False):
        assert confirmation_guard(state["confirmation"], 5, 5, three_arm=three_arm)[
            "problems"] == []  # fmt: skip
    assert mechanism_state(state["confirmation"], three_arm=True) == state["mechanism"]
    assert module.status(state)["mechanism"] == state["mechanism"]
    waiting = {"available": True, "looks": {"stopped": False}, "blocks": []}
    final = module.final_analysis(state, runs, [], {}, FIXATION, waiting, "wildguard", {})
    assert not final["available"] and final["reason"].startswith("not yet")


def test_the_final_analysis_opens_the_time_course_once(monkeypatch, tmp_path) -> None:
    """The time course's labels are read in the final analysis only, and only for its runs
    identical to their endpoint runs; with two arms (no mechanism cohort) the final analysis
    follows the primary's stop."""
    module = summary_module()
    rates = {ADAMW: [1.27e-5, 1.48e-5, 1.74e-5], FA: [1.59e-5, 1.86e-5, 2.17e-5]}
    identical, excluded = on_disk([course(ADAMW, 20, 1.48e-5, 3000),
                                   course(FA, 20, 1.86e-5, 3000)], tmp_path)  # fmt: skip
    read = []
    monkeypatch.setattr(module, "label_path", lambda run_dir, judge: run_dir / "config.yaml")
    monkeypatch.setattr(module, "load_prompt_set", lambda name, data: [{"id": "p", "text": "t"}])

    def endpoint_labels(run_dir, judge, identity, prompts):
        read.append(run_dir.name)
        return [{"condition": f"step-{t}", "step": t, "prompts": 500, "harmful": t,
                 "missing": 0} for t in TIMECOURSE_STEPS]  # fmt: skip

    monkeypatch.setattr(module, "read_endpoint_labels", endpoint_labels)
    course_state = {"trajectories": [{"run": identical.run_id, "state": "identical"},
                                     {"run": excluded.run_id, "state": "excluded"},
                                     {"run": None, "state": None}]}  # fmt: skip
    labels, pending = module.read_timecourse_labels({"timecourse": course_state},
                                                    [identical, excluded], "wildguard")  # fmt: skip
    assert set(labels) == {identical.run_id} and read == [identical.run_id] and pending == {}
    # Two arms: no mechanism cohort; the final analysis waits for the primary's stop, then
    # reads the time course's labels once.
    read.clear()
    state = {"next": {"campaign": "confirmation"},
             "frozen": {"three_arm": False, "rates": rates, "batch": 20, "K_star": 6.3},
             "mechanism": mechanism_state([], three_arm=False),
             "timecourse": {"trajectories": []}}  # fmt: skip
    found = {"available": True, "looks": {"stopped": False}, "blocks": []}
    final = module.final_analysis(state, [], [], {}, FIXATION, found, "wildguard", {})
    assert final["reason"].startswith("not yet: the primary has not stopped") and read == []
    found["looks"]["stopped"] = True
    final = module.final_analysis(state, [], [], {}, FIXATION, found, "wildguard", {})
    assert final["available"] and final["mechanism"] is None
    assert final["mechanism_reason"].startswith("a two-arm selection")
    assert final["timecourse"]["labels_read"] and final["timecourse"]["counts"]["missing"] == 20


# ------------------------------------------------------------------ the time course


def test_timecourse_excluded_from_blocks(records_alike) -> None:
    """The time course's runs (amendment 4, section 4) share the first blocks' seeds, arms and
    central rates with the endpoint runs, and enter no block, no look and no mechanism cohort:
    the selection, the blocks, the guard's counts, the looks' blocks and the three-arm contrasts
    are those without them, a block holding time-course runs alone has no run, and none of them
    is ever a bracket."""
    runs = base()
    rates = until(runs, "confirmation")["next"]["rates"]
    runs += blocks_of(rates, 20, CONFIRMATION_SEEDS[:2])
    # Blocks 1-3 of the time course (block 3 has no endpoint run yet); in block 2 a run whose
    # K differs from its endpoint's everywhere (excluded), in reach of another bracket.
    timecourse = [*courses(rates, 20, TIMECOURSE_SEEDS[:3:2]),
                  *(course(arm, 20, sorted(lrs)[1], 3001, scale=0.9)
                    for arm, lrs in rates.items())]  # fmt: skip
    without, with_course = stage(runs), stage([*runs, *timecourse])
    assert with_course["next"]["campaign"] == "confirmation" and not with_course["problems"]
    assert with_course["selection"] == without["selection"]
    assert with_course["confirmation"] == without["confirmation"]
    blocks = with_course["confirmation"]
    assert [b["complete"] for b in blocks[:3]] == [True, True, False] and blocks[2]["runs"] == 0
    assert confirmation_guard(blocks, 3, 3, three_arm=True) == confirmation_guard(
        without["confirmation"], 3, 3, three_arm=True)  # fmt: skip
    assert with_course["mechanism"] == without["mechanism"]
    ids = {run.run_id for run in timecourse}
    brackets = {s["bracket"][side]["run"] for b in blocks if b["complete"]
                for s in b["supports"].values() if s["supported"]
                for side in ("lo", "hi")}  # fmt: skip
    assert brackets and not brackets & ids
    # Labels for every run, the time course's far from the endpoints': the looks' blocks, the
    # looks and the three-arm contrasts do not move.
    labels = {run.run_id: rows(400 if run.run_id in ids else 50 + run.seed % 7)
              for run in [*runs, *timecourse]}  # fmt: skip
    kstar = with_course["frozen"]["K_star"]
    entries = look_blocks(blocks, labels, kstar)
    plain = {k: v for k, v in labels.items() if k not in ids}
    assert entries == look_blocks(without["confirmation"], plain, kstar)
    assert [e["complete"] for e in entries[:3]] == [True, True, False]
    assert all(set(e["h"]) == set(ARMS) for e in entries[:2])  # the three-arm cohort
    settings = {"look": 1, "interval": "t", "status": "secondary"}
    assert secondary_contrasts(entries, 2, **settings) == secondary_contrasts(
        look_blocks(without["confirmation"], plain, kstar), 2, **settings)  # fmt: skip
    summary = with_course["timecourse"]
    states = [(t["seed"], t["state"]) for t in summary["trajectories"][:9]]
    assert states == [(3000, "identical")] * 3 + [(3001, "excluded")] * 3 + [(3002, "pending")] * 3


def test_timecourse_reports_missing(records_alike) -> None:
    """Per seed and arm: an absent run is missing, a run stopped early and a run the launcher
    gave up on are failures; an endpoint not there yet leaves the identity pending, an endpoint
    the launcher gave up on leaves it excluded; a given-up time-course run never stands for its
    endpoint run, nor the other way round."""
    runs = base()
    rates = until(runs, "confirmation")["next"]["rates"]
    central = {arm: sorted(lrs)[1] for arm, lrs in rates.items()}
    gone_endpoint = (ADAMW, 20, central[ADAMW], 3005)
    # Not there: the endpoint runs of (AdamW, 3004), (AdamW, 3005, given up) and (pre-warmed,
    # 3003, whose time-course run was given up: that never gives the endpoint run up).
    runs += [r for r in blocks_of(rates, 20, CONFIRMATION_SEEDS[:10])
             if (r.arm, r.batch_size, r.lr, r.seed) not in {
                 (ADAMW, 20, central[ADAMW], 3004), gone_endpoint,
                 (PREWARMED, 20, central[PREWARMED], 3003)}]  # fmt: skip
    absent, given_up = (ADAMW, 3001), (PREWARMED, 3003)
    timecourse = [course(arm, 20, central[arm], seed, stop="gradient_spike", steps=7)
                  if (arm, seed) == (FA, 3002) else course(arm, 20, central[arm], seed)
                  for seed in TIMECOURSE_SEEDS for arm in ARMS
                  if (arm, seed) not in (absent, given_up)]  # fmt: skip
    # The launcher's journal: the given-up time-course run, keyed apart from its endpoint run.
    cell = f"{PREWARMED}-b20-{central[PREWARMED]:g}-timecourse-rayleigh"
    overrides = ["optimizer=adamw_prewarmed", "train.batch_size=20",
                 f"optimizer.lr={central[PREWARMED]:g}",
                 *study2_short.TIMECOURSE_OVERRIDES]  # fmt: skip
    records = journal(cell, 3003, overrides, attempts=2 * MAX_ATTEMPTS, recovered=True,
                      config="experiments/study2_short_rayleigh")  # fmt: skip
    exhausted = exhausted_runs(records)
    assert exhausted == {(PREWARMED, 20, central[PREWARMED], 3003, TIMECOURSE): f"{cell}/s3003"}
    exhausted[gone_endpoint] = "adamw-endpoint/s3005"
    state = stage([*runs, *timecourse], exhausted=exhausted)
    assert state["next"]["campaign"] == "confirmation" and not state["problems"]
    block = state["confirmation"][3]  # seed 3003: its pre-warmed endpoint run is still pending
    assert not block["complete"] and block["runs"] == 8 and block["exhausted"] == []
    assert state["exhausted"] == {"adamw-endpoint/s3005": list(gone_endpoint)}
    summary = state["timecourse"]
    assert summary["missing"] == [{"seed": 3001, "arm": ADAMW, "lr": central[ADAMW]}]
    assert summary["failures"] == [
        {"seed": 3002, "arm": FA, "run": f"{COURSE}{FA}-b20-{central[FA]:g}-s3002",
         "kind": "instability", "reason": "stopped by gradient_spike at step 7"},
        {"seed": 3003, "arm": PREWARMED, "run": f"{cell}/s3003", "kind": "technical",
         "reason": "given up by the launcher"},
    ]  # fmt: skip
    by = {(t["arm"], t["seed"]): t for t in summary["trajectories"]}
    assert len(by) == 30 and by[absent]["run"] is None and by[given_up]["run"] is None
    assert by[(FA, 3002)]["state"] == "excluded"  # its endpoint took its 25 steps
    assert "the stop and the steps taken differ" in by[(FA, 3002)]["reasons"][-1]
    assert [r["step"] for r in by[(FA, 3002)]["rows"]] == [1, 5]
    assert by[(ADAMW, 3004)]["state"] == "pending"
    assert by[(ADAMW, 3005)]["state"] == "excluded"
    assert "given up by the launcher" in by[(ADAMW, 3005)]["reasons"][0]
    assert summary["counts"] == {"identical": 25, "excluded": 2, "pending": 1, "failed": 2,
                                 "missing": 1}  # fmt: skip


def test_timecourse_runs_off_their_prescription_leave_the_stage_incomplete() -> None:
    runs = base()
    until(runs, "validation")
    early = course(ADAMW, 20, 1.48e-5, 3000)
    state = stage([*runs, early])
    assert state["next"] == {"campaign": "incomplete"}
    assert f"time-course run {early.run_id} before the rates are frozen" in state["problems"]
    rates = until(runs, "confirmation")["next"]["rates"]
    central = {arm: sorted(lrs)[1] for arm, lrs in rates.items()}
    one = course(ADAMW, 20, central[ADAMW], 3000)
    again = dataclasses.replace(one, run_id=f"{one.run_id}-again", run_dir=Path("/runs/again"))
    sealed = later(ADAMW, 20, central[ADAMW], 3001)  # an endpoint run with the time course's steps
    cases = {
        "outside the time course": [course(FA, 20, rates[FA][0], 3000)],  # not the central rate
        "is a time-course run on seed 3010, not one of 3000-3009": [
            course(ADAMW, 20, central[ADAMW], 3010)],
        "departs from the prescribed schedule in ['evaluate_at']": [
            dataclasses.replace(one, spec=P["short"])],  # the endpoint run's evaluation steps
        "time course: two runs of": [one, again],
    }  # fmt: skip
    for problem, extra in cases.items():
        state = stage([*runs, *extra])
        assert state["next"] == {"campaign": "incomplete"}, problem
        assert any(problem in p for p in state["problems"]), (problem, state["problems"])
    other = dataclasses.replace(sealed, spec=P["timecourse"])
    assert "departs from the prescribed schedule in ['evaluate_at']" in conformance(
        [other], facts_of([other]))[0]  # fmt: skip
    assert conformance([one], facts_of([one])) == []
    unmeasured = {one.run_id: {**fact(one), "rayleigh": False}}  # blocks 1-10 measure R
    found = conformance([one], unmeasured)
    assert "does not measure the Rayleigh quotient on seed 3000" in found[0]
    assert seed_role(3009, timecourse=True) == TIMECOURSE and seed_role(3009) == "confirmation"
    assert seed_role(3010, timecourse=True) is None and seed_role(2100, timecourse=True) is None
    # The loaders split the time course apart once; progress refuses one left among the others,
    # and a run read as a time-course run that is not one.
    mixed = [*runs, one]
    state = progress(mixed, THETA0, W_MAX, facts=facts_of(mixed), reuse=reusable)
    assert state["next"] == {"campaign": "incomplete"}
    assert f"time-course run {one.run_id} among the stage's other runs" in state["problems"][0]
    endpoint = later(ADAMW, 20, central[ADAMW], 3000)
    state = progress(runs, THETA0, W_MAX, facts=facts_of([*runs, endpoint]), reuse=reusable,
                     timecourse=[endpoint])  # fmt: skip
    assert state["next"] == {"campaign": "incomplete"}
    assert "is read as a time-course run, but its role is 'confirmation'" in state["problems"][0]
    assert split_timecourse(mixed, facts_of(mixed)) == (runs, [one])


def test_summary_reads_no_labels_without_the_argument(monkeypatch, records_alike) -> None:
    """Labels are read only when the final analysis passes them: ``progress`` gives none, and
    without them no label row is read and every harm is None. With them, the harm of each step
    of an identical run, and nothing of an excluded or pending one."""
    runs = base()
    rates = until(runs, "confirmation")["next"]["rates"]
    runs += blocks_of(rates, 20, CONFIRMATION_SEEDS[:2])
    timecourse = [*courses(rates, 20, TIMECOURSE_SEEDS[:3:2]),
                  *(course(arm, 20, sorted(lrs)[1], 3001, scale=0.9)
                    for arm, lrs in rates.items())]  # fmt: skip
    seen, summarize = [], study2_short.timecourse_summary

    def spied(*args, **kwargs):
        seen.append(args[1] if len(args) > 1 else kwargs.get("labels"))
        return summarize(*args, **kwargs)

    def unread(*args, **kwargs):
        raise AssertionError("a label was read")

    monkeypatch.setattr(study2_short, "timecourse_summary", spied)
    for name in ("harm", "_step_row"):
        monkeypatch.setattr(study2_short, name, unread)
    state = stage([*runs, *timecourse])
    summary = state["timecourse"]
    assert seen and all(labels is None for labels in seen)
    assert not summary["labels_read"] and "unlabeled" not in summary
    rows_read = [row for t in summary["trajectories"] for row in t["rows"]]
    assert len(rows_read) == 9 * 4 and all(row["harm"] is None for row in rows_read)
    assert [row["step"] for row in summary["trajectories"][0]["rows"]] == [1, 5, 10, 25]
    monkeypatch.undo()
    records_alike()
    # The final analysis: the identical runs' harm per step; a run without labels is unlabeled.
    timecourse_ids = {run.run_id for run in timecourse}
    labels = {run_id: [{"condition": f"step-{t}", "step": t, "prompts": 500, "harmful": 10 * t,
                        "missing": 0} for t in TIMECOURSE_STEPS]
              for run_id in sorted(timecourse_ids)[1:]}  # fmt: skip
    found, _ = study2_short.index(r for r in runs if r.seed in CONFIRMATION_SEEDS)
    labelled = timecourse_summary(timecourse, labels, endpoints=found, rates=rates, batch=20)
    assert labelled["labels_read"]
    unlabeled = sorted(timecourse_ids)[0]  # AdamW-25 on seed 3000, identical
    assert labelled["unlabeled"] == [unlabeled]
    assert [t["state"] for t in labelled["trajectories"] if t["run"] == unlabeled] == ["identical"]
    for t in labelled["trajectories"]:
        harms = [row["harm"] for row in t["rows"]]
        if t["state"] == "identical" and t["run"] != unlabeled:
            assert harms == [10 * s / 500 for s in TIMECOURSE_STEPS]
        else:
            assert all(h is None for h in harms), t["state"]


# ------------------------------------------------------------------ the campaigns script


def theta0_dir(tmp_path: Path) -> Path:
    run = tmp_path / "theta0"
    run.mkdir(exist_ok=True)
    (run / "summary.json").write_text(json.dumps(THETA0))
    return run


def fixation_file(tmp_path: Path, **changes) -> str:
    path = tmp_path / f"fixation-{len(list(tmp_path.glob('fixation-*')))}.json"
    path.write_text(json.dumps({**FIXATION, **changes}))
    return str(path)


def runs_root(tmp_path: Path, records: list | None = None) -> str:
    root = tmp_path / "runs-root"
    (root / "launch").mkdir(parents=True, exist_ok=True)
    log = root / "launch" / "study2-short.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in records or []))
    return str(root)


def script(
    monkeypatch, tmp_path, path: Path, runs: list, *args: str, versions: bool = True,
    root: bool = True, verdicts: dict | None = None,
) -> Path:  # fmt: skip
    """Run a script on ``runs`` (patched in for the run directories); returns its --out. With
    ``versions``, the versions record holds (``versions_problems`` has its own tests). With
    ``root``, --runs-root is given (an empty journal unless the arguments name one) and holds
    ``verdicts`` (every check the stage asks for, ``answered``, by default); the GPU-code
    check reads the synthetic commits (``test_study2_short.reusable``)."""
    monkeypatch.setattr("zero_field_refusal.analysis.stage_a.load_trajectories",
                        lambda *roots: list(runs))  # fmt: skip
    monkeypatch.setattr("zero_field_refusal.analysis.study2_short._reuse_check", reusable)
    if root:
        if "--runs-root" not in args:
            args = (*args, "--runs-root", runs_root(tmp_path))
        directory = Path(args[args.index("--runs-root") + 1])
        written = answered(runs) if verdicts is None else verdicts
        for name, record in written.items():
            (directory / name).parent.mkdir(parents=True, exist_ok=True)
            (directory / name).write_text(json.dumps(record))
    if versions:
        monkeypatch.setattr("zero_field_refusal.analysis.study2_short.versions_problems",
                            lambda path, **kwargs: [])  # fmt: skip
        if "--versions" not in args:
            record = tmp_path / "versions-holding.json"
            record.write_text("{}")
            args = (*args, "--versions", str(record))
    by_dir = {run.run_dir: run for run in runs}
    monkeypatch.setattr("zero_field_refusal.analysis.study2_short.run_facts",
                        lambda run_dir: fact(by_dir[run_dir]))  # fmt: skip
    out = tmp_path / f"out-{len(list(tmp_path.glob('out-*')))}.yaml"
    common = ["--runs", str(tmp_path), "--theta0", str(theta0_dir(tmp_path))]
    monkeypatch.setattr("sys.argv", [str(path), *args, *common, "--out", str(out)])
    runpy.run_path(str(path), run_name="__main__")
    return out


def campaign(monkeypatch, tmp_path, runs, *args: str) -> list:
    return load_campaign(script(monkeypatch, tmp_path, CAMPAIGNS, runs, *args))[1]


def refusal(monkeypatch, tmp_path, capsys, runs, *args: str, versions: bool = True,
            root: bool = True, verdicts: dict | None = None) -> str:  # fmt: skip
    with pytest.raises(SystemExit):
        script(monkeypatch, tmp_path, CAMPAIGNS, runs, *args, versions=versions, root=root,
               verdicts=verdicts)  # fmt: skip
    return capsys.readouterr().err


def labels_of(runs) -> list[str]:
    return [run.label for run in runs]


def fixed(tmp_path: Path) -> str:
    """A fixation record whose w_max is the synthetic stage's."""
    return fixation_file(tmp_path, w_max=W_MAX)


def test_the_discovery_campaign_records_w_max_and_the_revision(monkeypatch, tmp_path) -> None:
    runs = base()
    record = fixed(tmp_path)
    out = script(monkeypatch, tmp_path, CAMPAIGNS, runs, "discovery", "--fixation", record)
    header = " ".join(line[2:] for line in out.read_text().splitlines() if line.startswith("#"))
    assert re.search(r"discovery at [0-9a-f]{40}", header) and "w_max 0.35 from the fixation" in (
        header)  # fmt: skip
    assert f"the fixation record {record}, sha256" in header
    found = load_campaign(out)[1]
    # The joint discovery at (8, 20): both primary arms' targeted rounds and the pre-warmed
    # arm's first round, at the union of their rates.
    assert len(found) == 36 and {run.experiment for run in found} == {"study2_short"}
    assert "frozen_adam_25-b20-1.59e-05/s2001" in labels_of(found)
    assert "prewarmed_adamw_25-b20-1.59e-05/s2001" in labels_of(found)
    fa = next(run for run in found if run.cell == "frozen_adam_25-b20-1.59e-05")
    assert fa.overrides[:4] == ("optimizer=adam_frozen", "train.batch_size=20",
                                "optimizer.lr=1.59e-05", "study2_short.role=discovery")  # fmt: skip
    split = campaign(monkeypatch, tmp_path, runs, "discovery", "--fixation", record,
                     "--seeds", "2002")  # fmt: skip
    assert len(split) == 12 and {run.seed for run in split} == {2002}


def test_the_campaigns_refuse_another_step_a_missing_w_max_and_incomplete_records(
    monkeypatch, tmp_path, capsys
) -> None:
    runs = base()
    record = fixation_file(tmp_path)
    err = refusal(monkeypatch, tmp_path, capsys, runs, "numerical", "--fixation", record)
    assert "the stage's next step is discovery, not numerical" in err
    err = refusal(monkeypatch, tmp_path, capsys, runs, "numerical", "--w-max", str(W_MAX))
    assert "numerical needs the fixation record (--fixation)" in err
    err = refusal(monkeypatch, tmp_path, capsys, runs, "discovery")
    assert "discovery needs the fixation record (--fixation)" in err
    err = refusal(monkeypatch, tmp_path, capsys, runs, "discovery", "--w-max", "0.3",
                  "--fixation", record)  # fmt: skip
    assert "--w-max 0.3 differs from the fixation record's 0.47" in err
    err = refusal(monkeypatch, tmp_path, capsys, runs, "discovery",
                  "--fixation", fixation_file(tmp_path, se_floor=None))  # fmt: skip
    assert "se_floor None is not a number" in err
    err = refusal(monkeypatch, tmp_path, capsys, runs[1:], "discovery", "--fixation", record)
    assert "version-1 run missing" in err  # Stage A's reference incomplete: no stream time
    err = refusal(monkeypatch, tmp_path, capsys, runs, "discovery", "--fixation", record,
                  root=False)  # fmt: skip
    assert "discovery needs --runs-root" in err  # the verdicts and the journal live there
    err = refusal(monkeypatch, tmp_path, capsys, runs, "controls", "--fixation", record)
    assert "invalid choice: 'controls'" in err  # amendment 3's controls are never run again


def test_the_numerical_campaign(monkeypatch, tmp_path) -> None:
    runs = base()
    until(runs, "numerical")
    record = fixed(tmp_path)
    out = script(monkeypatch, tmp_path, CAMPAIGNS, runs, "numerical", "--fixation", record)
    found = load_campaign(out)[1]
    # The 15 cases' regular runs, K-only: every selected rate on 2200, each U-rate on 2201-2202.
    assert len(found) == 15 and {run.seed for run in found} == set(NUMERICAL_SEEDS)
    assert {run.config for run in found} == {"experiments/study2_short_k_only"}
    assert {run.experiment for run in found} == {"study2_short"}
    assert all("study2_short.role=numerical" in run.overrides for run in found)
    tag = f"numerical-u8-{numerical_digest(RATES20)}"  # the selection's digest: unique names
    assert f"adamw_25-b20-1.74e-05-{tag}/s2202" in labels_of(found)
    assert f"adamw_25-b20-1.27e-05-{tag}/s2201" not in labels_of(found)
    cfg = load_run_config(found[0].config, found[0].overrides)
    assert cfg.eval.k_only is True and cfg.study2_short.role == "numerical"
    header = " ".join(line[2:] for line in out.read_text().splitlines() if line.startswith("#"))
    assert ("prep --seeds 2200 2201 2202 --batch 20): the training stream of 500 responses and "
            "v₀'s preconditioner stream of 1000") in header  # fmt: skip
    assert f"The check {numerical_digest(RATES20)} follows the runs" in header


def test_the_numerical_check_writes_the_tools_selection_record(monkeypatch, tmp_path, capsys):
    runs = base()
    until(runs, "numerical")
    runs += make(stage(runs)["next"])  # the numerical runs; no verdict yet
    record = fixed(tmp_path)
    out = script(monkeypatch, tmp_path, CAMPAIGNS, runs, "numerical-check", "--fixation", record,
                 verdicts={})  # fmt: skip
    # Exactly the record scripts/numerical_check.py --selection reads (Task 1): the rates by
    # optimizer name, three per arm, with the batch and U.
    assert json.loads(out.read_text()) == {
        "rates": {"adamw": [1.27e-5, 1.48e-5, 1.74e-5], "adam_frozen": [1.59e-5, 1.86e-5, 2.17e-5],
                  "adamw_prewarmed": [1.27e-5, 1.48e-5, 1.74e-5]},
        "batch": 20, "U": 8}  # fmt: skip
    shown = capsys.readouterr().out
    # The final review of gate B, M1: the whole command, with the numerical runs' commit and the
    # driver's file, the request's runs_commits and its w_max (I4)
    driver = out.with_name(f"{out.stem}-driver.sh")
    assert (f"python scripts/numerical_check.py driver --selection {out} --batch 20 --upper 8 "
            f"--commit {RUN_COMMIT} --out {driver}") in shown.splitlines()  # fmt: skip
    assert f"runs_commits (the numerical runs' commits): ['{RUN_COMMIT}']" in shown
    assert (
        f"w_max {W_MAX!r}; outputs under $RUNS_DIR/numerical/b20-u8, the selection's verdict at "
        f"$RUNS_DIR/numerical/b20-u8/verdicts/{numerical_digest(RATES20)}.json" in shown
    )
    assert "given up" not in shown
    err = refusal(monkeypatch, tmp_path, capsys, runs, "numerical-check", "--fixation", record)
    assert "the stage's next step is validation, not numerical-check" in err  # verdict in place


def test_the_check_command_names_its_commits_and_the_cases_given_up(tmp_path) -> None:
    """Review M1: one runs commit is the driver's --commit; several (one GPU code) leave COMMIT
    to choose among them; the cases given up are named."""
    module = campaigns_module()
    check = {**request_check(), "w_max": 0.47, "given_up": ["s2200-adamw-1.48e-05"],
             "runs_commits": ["a" * 40, "b" * 40]}  # fmt: skip
    lines = module.check_command(check, tmp_path / "check.json")
    assert lines[0].endswith(f"--commit COMMIT --out {tmp_path / 'check-driver.sh'}")
    assert lines[1].endswith("; COMMIT is one of them, their GPU code being one")
    assert lines[-1] == "given up by the launcher, named in the record: ['s2200-adamw-1.48e-05']"


def test_the_whole_config_of_a_run_is_its_campaigns(tmp_path) -> None:
    """The final review of gate B, M2: the config hash a regular run's meta.json records must be
    that of its campaign's config and overrides, as the launcher loads them from the tracked
    configs; a run whose recorded config differs in one unrelated key is a problem."""
    run = later(ADAMW, 20, 1.27e-5, 2200)  # a numerical run: experiments/study2_short_k_only
    config, overrides = study2_short.run_prescription(ADAMW, 20, 1.27e-5, 2200, "numerical")
    assert (config, overrides) == ("experiments/study2_short_k_only", [
        "train.batch_size=20", "optimizer.lr=1.27e-05", "study2_short.role=numerical"])  # fmt: skip
    cfg = load_run_config(config, [*overrides, "seed=2200"])
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def recorded(cfg) -> dict:
        OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
        (run_dir / "meta.json").write_text(json.dumps({"config_hash": config_hash(cfg),
                                                       "git": {"commit": RUN_COMMIT,
                                                               "dirty": False}}))  # fmt: skip
        return {run.run_id: {**run_facts(run_dir), "preconditioner": None}}

    assert recorded(cfg)[run.run_id]["config_hash"] == prescribed_hash(ADAMW, 20, 1.27e-5, 2200,
                                                                       "numerical")  # fmt: skip
    assert conformance([run], recorded(cfg)) == []
    other = OmegaConf.merge(cfg, {"train": {"max_response_tokens": cfg.train.max_response_tokens
                                            + 1}})  # fmt: skip
    (found,) = conformance([run], recorded(other))
    assert found.startswith(f"{run.run_id} records the config hash '{config_hash(other)}', not "
                            f"'{config_hash(cfg)}' of its campaign's config "
                            "experiments/study2_short_k_only")  # fmt: skip
    # Each role's prescription is the campaigns' cell (the campaigns script builds on it)
    for arm, seed, role, name in ((FA, 3000, "confirmation", "experiments/study2_short_rayleigh"),
                                  (PREWARMED, 2100, "validation", "experiments/study2_short"),
                                  (ADAMW, 3001, TIMECOURSE, "experiments/study2_short_rayleigh"),
                                  (FA, 2000, "discovery", "experiments/study2_short")):  # fmt: skip
        config, overrides = study2_short.run_prescription(arm, 20, 1.5e-5, seed, role)
        assert config == name and prescribed_hash(arm, 20, 1.5e-5, seed, role) == config_hash(
            load_run_config(config, [*overrides, f"seed={seed}"]))  # fmt: skip
    # Stage A's grid runs (stage_a.role) keep Stage A's prescription: not compared
    stage_a = dataclasses.replace(conform(made(ADAMW, 20, 1e-5, 2000)), role="discovery")
    assert conformance([stage_a], {stage_a.run_id: {**fact(stage_a), "config_hash": None}}) == []


def test_a_replay_under_another_w_max_is_refused(monkeypatch, tmp_path, capsys) -> None:
    """The final review of gate B, I4: the campaign's header records w_max, the launcher logs it
    with every start, and the replay under another w_max (a fixation record rewritten at gate C,
    say) is refused, the stage incomplete with that reason; a start logged without a header
    records none."""
    runs = base()
    record = fixed(tmp_path)
    out = script(monkeypatch, tmp_path, CAMPAIGNS, runs, "discovery", "--fixation", record)
    assert campaign_header(out) == {"w_max": W_MAX}
    name, found = load_campaign(out)
    root = Path(runs_root(tmp_path))
    started = run_campaign(name, found[:1], root, execute=lambda run, run_id, output: 1,
                           header=campaign_header(out))  # fmt: skip
    assert started["exhausted"] == [found[0].label]
    journal = launch_records(root)
    assert {r["event"] for r in journal} == {"start", "end"}
    assert all(r["header"] == {"w_max": W_MAX} for r in journal if r["event"] == "start")
    assert issued_w_max(journal) == {found[0].label: W_MAX}
    # The same w_max replays; another is refused, with the run that recorded the issued one
    same = progress(runs, THETA0, W_MAX, facts=facts_of(runs), reuse=reusable,
                    issued_w_max=issued_w_max(journal))  # fmt: skip
    assert same["next"]["campaign"] == "discovery"
    state = progress(runs, THETA0, 0.40, facts=facts_of(runs), reuse=reusable,
                     issued_w_max=issued_w_max(journal))  # fmt: skip
    assert state["next"] == {"campaign": "incomplete"}
    assert state["problems"] == [
        f"w_max 0.4 differs from the w_max {W_MAX!r} recorded in the header of the issued "
        f"campaign of {found[0].label}: w_max is frozen since S1 (amendment 4), and the replay "
        "is refused under another value"]  # fmt: skip
    err = refusal(monkeypatch, tmp_path, capsys, runs, "discovery",
                  "--fixation", fixation_file(tmp_path, w_max=0.40), "--runs-root",
                  str(root))  # fmt: skip
    assert "the stage's records are not whole: w_max 0.4 differs from the w_max" in err
    # A start logged before the header existed records none: not compared
    bare = [{k: v for k, v in r.items() if k != "header"} for r in journal]
    assert issued_w_max(bare) == {}
    assert w_max_problems(0.40, issued_w_max(bare)) == []


def test_the_check_input_is_written_while_runs_wait_beyond_a_stale_check(
    monkeypatch, tmp_path, capsys
) -> None:
    """Review of Task 2, minor 4: the qualifying verdict went stale after the validation runs
    were made. Those runs wait for the check; its rate input may still be written, and every
    other kind is refused while the records are not whole."""
    runs = base()
    runs += make(until(runs, "validation")["next"])
    stale = {path: {**record, "policy": {**record["policy"], "norm": 1e-4}}
             for path, record in answered(runs).items()}  # fmt: skip
    record = fixed(tmp_path)
    out = script(monkeypatch, tmp_path, CAMPAIGNS, runs, "numerical-check", "--fixation", record,
                 verdicts=stale)  # fmt: skip
    assert json.loads(out.read_text()) == {"rates": {"adamw": RATES20[ADAMW],
                                                     "adam_frozen": RATES20[FA],
                                                     "adamw_prewarmed": RATES20[PREWARMED]},
                                           "batch": 20, "U": 8}  # fmt: skip
    shown = capsys.readouterr().out
    assert "Written while the stage's records wait for this check" in shown
    assert f"the replay waits at the stale check {verdict_path(RATES20)} (policy norm" in shown
    err = refusal(monkeypatch, tmp_path, capsys, runs, "validation", "--fixation", record,
                  verdicts=stale)  # fmt: skip
    assert "the stage's records are not whole: the replay waits at the stale check" in err
    # A problem other than the runs beyond the stale check, here an unread verdict marking a
    # rate unusable at B = 20: refused as well.
    other = {**RATES20, ADAMW: [1.27e-5, 1.5e-5, 1.74e-5]}
    unread = verdict_of(request_check(other, upper=7.0), {("adamw", 1.5e-5): "not_qualified"})
    err = refusal(monkeypatch, tmp_path, capsys, runs, "numerical-check", "--fixation", record,
                  verdicts={**stale, verdict_path(other, upper=7.0): unread})  # fmt: skip
    assert "the stage's records are not whole" in err and "did not read marks" in err


def test_the_validation_and_confirmation_campaigns(monkeypatch, tmp_path, capsys) -> None:
    runs = base()
    until(runs, "validation")
    record, root = fixed(tmp_path), runs_root(tmp_path)
    journal_args = ("--fixation", record, "--runs-root", root)
    found = campaign(monkeypatch, tmp_path, runs, "validation", *journal_args)
    assert len(found) == 108 and {r.seed for r in found} == set(VALIDATION_SEEDS)
    assert all("study2_short.role=validation" in r.overrides for r in found)
    # The launcher grants a recovery round per experiment and cell name: no name recurs.
    assert "adamw_25-b20-1.27e-05-validation" in {r.cell for r in found}
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "1", "2",
                  *journal_args)  # fmt: skip
    assert "next step is validation" in err  # the validation has not passed yet
    until(runs, "confirmation")
    found = campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", "1", "12",
                     *journal_args)  # fmt: skip
    assert len(found) == 12 * 9 and {r.seed for r in found} == set(range(3000, 3012))
    measuring = [r for r in found if r.config == "experiments/study2_short_rayleigh"]
    assert {r.seed for r in measuring} == set(range(3000, 3010))  # blocks 1-10 measure R
    assert {"adamw_25-b20-1.27e-05-blocks-1-12", "adamw_25-b20-1.27e-05-blocks-1-12-rayleigh"} <= {
        r.cell for r in found}  # fmt: skip
    cfg = load_run_config(measuring[0].config, measuring[0].overrides)
    assert cfg.eval.rayleigh.enabled and cfg.study2_short.role == "confirmation"
    assert {r.experiment for r in found} == {"study2_short"}
    # A three-arm selection: no pause at the primary's looks, room for the cohort's 60 blocks.
    for blocks, why in ((("2", "3"), "starts at block 1"), (("1", "61"), "room for 60"),
                        (("0", "3"), "1 <= FIRST"), (("5", "4"), "1 <= FIRST")):  # fmt: skip
        err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", *blocks,
                      *journal_args)  # fmt: skip
        assert why in err, blocks
    assert len(campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", "1", "60",
                        *journal_args)) == 60 * 9  # fmt: skip
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "1", "2",
                  "--after-look", "1", *journal_args)  # fmt: skip
    assert "--after-look 1: a three-arm confirmation does not pause at the primary's looks" in err


def test_a_two_arm_confirmation_keeps_the_look_pauses(monkeypatch, tmp_path, capsys) -> None:
    """The review of Task 3, M4: with a two-arm selection (the primary-only fallback) the
    campaigns script keeps amendment 3's loop: ranges end before look 1, look 1 due holds every
    block, and ``--after-look 1`` lets the blocks before look 2 start."""
    runs = [*base()]
    real = SCALE.copy()
    try:
        for batch in (5, 20):
            SCALE[(PREWARMED, batch)] = 0.3  # the pre-warmed arm never brackets K*
        frozen = until(runs, "confirmation")["frozen"]
    finally:
        SCALE.update(real)
    assert not frozen["three_arm"] and set(frozen["rates"]) == {ADAMW, FA}
    args = ("--fixation", fixed(tmp_path), "--runs-root", runs_root(tmp_path))
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "1", "37",
                  *args)  # fmt: skip
    assert "room for 36 before look 1" in err
    assert len(campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", "1", "36",
                        *args)) == 36 * 6  # fmt: skip
    runs += blocks_of(frozen["rates"], 20, CONFIRMATION_SEEDS[:36])
    assert all(b["supported"] for b in stage(runs)["confirmation"][:36])
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "37", "37",
                  *args)  # fmt: skip
    assert "look 1 is due (36 supported blocks complete in order" in err
    found = campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", "37", "60",
                     "--after-look", "1", *args)  # fmt: skip
    assert len(found) == 24 * 6 and {r.seed for r in found} == set(CONFIRMATION_SEEDS[36:60])
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "37", "61",
                  "--after-look", "1", *args)  # fmt: skip
    assert "room for 24 before look 2" in err


def test_validation_and_confirmation_need_the_versions_record(monkeypatch, tmp_path, capsys):
    runs = base()
    until(runs, "validation")
    missing = str(tmp_path / "no-versions.json")
    args = ("--fixation", fixed(tmp_path), "--runs-root", runs_root(tmp_path),
            "--versions", missing)  # fmt: skip
    err = refusal(monkeypatch, tmp_path, capsys, runs, "validation", *args, versions=False)
    assert f"the versions record does not hold: {missing} is missing" in err
    until(runs, "confirmation")
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "1", "2",
                  *args, versions=False)  # fmt: skip
    assert "is missing: write the versions record" in err
    # Discovery precedes the record: its campaigns do not read it.
    early = base()
    found = campaign(monkeypatch, tmp_path, early, "discovery", *args)
    assert len(found) == 36
    # With the record holding, the header names it.
    record = tmp_path / "versions.json"
    record.write_text("{}")
    out = script(monkeypatch, tmp_path, CAMPAIGNS, runs, "confirmation", "--blocks", "1", "2",
                 *args[:4], "--versions", str(record))  # fmt: skip
    header = " ".join(line[2:] for line in out.read_text().splitlines() if line.startswith("#"))
    digest = hashlib.sha256(record.read_bytes()).hexdigest()
    assert f"under the versions record {record}, sha256 {digest}" in header


def test_the_abandonment_record_lets_a_lost_attempt_be_given_up(monkeypatch, tmp_path, capsys):
    runs = base()
    rates = until(runs, "confirmation")["next"]["rates"]
    block_runs = blocks_of(rates, 20, CONFIRMATION_SEEDS[:2])
    lost = next(r for r in block_runs if (r.arm, r.seed) == (ADAMW, 3000))
    runs += [r for r in block_runs if r is not lost]
    cell = f"{ADAMW}-b20-{lost.lr:g}-blocks-1-2-rayleigh"
    overrides = ["train.batch_size=20", f"optimizer.lr={lost.lr:g}",
                 "study2_short.role=confirmation"]  # fmt: skip
    records = journal(cell, 3000, overrides, attempts=2 * MAX_ATTEMPTS, recovered=True,
                      last_ended=False, config="experiments/study2_short_rayleigh")  # fmt: skip
    root = runs_root(tmp_path, records)
    path = tmp_path / "abandoned.json"
    path.write_text(
        json.dumps(
            {"abandoned": [{"run_id": records[-1]["run_id"], "reason": "Pod terminated mid-run"}]}
        )
    )
    common = ("--fixation", fixed(tmp_path), "--runs-root", root)
    script(monkeypatch, tmp_path, SUMMARY, runs, *common)
    waiting = json.loads(capsys.readouterr().out)
    assert waiting["confirmation"]["given_up_by_the_launcher"] == {}  # the sixth may still run
    out = script(monkeypatch, tmp_path, SUMMARY, runs, *common, "--abandoned", str(path))
    shown = json.loads(capsys.readouterr().out)
    assert shown["confirmation"]["given_up_by_the_launcher"] == {
        f"{cell}/s3000": [ADAMW, 20, lost.lr, 3000]}  # fmt: skip
    assert shown["abandoned"] == {records[-1]["run_id"]: "Pod terminated mid-run"}
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(out.read_text())["abandonment_record"] == {"path": str(path),
                                                                 "sha256": digest}  # fmt: skip
    path.write_text(json.dumps({"abandoned": [{"run_id": "unknown", "reason": "typo"}]}))
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "1", "2",
                  *common, "--abandoned", str(path))  # fmt: skip
    assert "the abandoned attempt unknown has no start in the launch journal" in err


def test_a_range_issued_before_is_completed_and_then_refused(monkeypatch, tmp_path, capsys):
    runs = base()
    rates = until(runs, "confirmation")["next"]["rates"]
    record = fixed(tmp_path)
    block_runs = blocks_of(rates, 20, CONFIRMATION_SEEDS[:2])
    runs += block_runs[:-1]  # blocks 1-2 issued and launched, one run still to make
    started = [{"event": "start", "run": f"{r.arm}-b20-{r.lr:g}-blocks-1-2-rayleigh/s{r.seed}",
                "experiment": "study2_short", "config_hash": r.run_id, "run_id": r.run_id,
                "attempt": 1, "command": []} for r in block_runs]  # fmt: skip
    ended = [{"event": "end", "run_id": r.run_id, "outcome": "done"} for r in block_runs]
    args = ("--fixation", record, "--runs-root", runs_root(tmp_path, started + ended))
    found = campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", "1", "2", *args)
    assert labels_of(found) == ["prewarmed_adamw_25-b20-1.74e-05-blocks-1-2-rayleigh/s3001"]
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "2", "3",
                  *args)  # fmt: skip
    assert "have runs already or are in a range issued before, and 2-3 is not a range" in err
    runs.append(block_runs[-1])
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "1", "2",
                  *args)  # fmt: skip
    assert "blocks 1-2 have every run already" in err


def issued(first: int, last: int) -> list[dict]:
    """The launcher's start record of one run of the confirmation range FIRST-LAST."""
    return [{"event": "start", "experiment": "study2_short", "config_hash": f"r-{first}-{last}",
             "run": f"{ADAMW}-b20-1.27e-05-blocks-{first}-{last}-rayleigh/s{3000 + first - 1}",
             "run_id": f"issued-{first}-{last}", "attempt": 1, "command": []}]  # fmt: skip


def test_timecourse_campaign_shape(monkeypatch, tmp_path, capsys) -> None:
    """The time course's campaign (amendment 4, section 4): seeds 3000-3009, each arm at its
    central rate, the endpoint run's config (blocks 1-10 measure R) but for the role and the
    evaluation steps; 30 runs, 20 without the pre-warmed arm. Refused before the selection is
    done, before blocks 1-10 are issued, without the versions record, and once every run is
    there."""
    runs = base()
    until(runs, "validation")
    record = fixed(tmp_path)
    common = ("--fixation", record)
    err = refusal(monkeypatch, tmp_path, capsys, runs, "timecourse", *common,
                  "--runs-root", runs_root(tmp_path, issued(1, 10)))  # fmt: skip
    assert "the time course waits for the selection to be done" in err
    assert "the stage's next step is validation" in err
    rates = until(runs, "confirmation")["next"]["rates"]
    for journal_records, lacking in (([], "blocks 1-10 are not in an issued range (10 of 10)"),
                                     (issued(1, 5), "blocks 6-10 are not in an issued range "
                                                    "(5 of 10)")):  # fmt: skip
        err = refusal(monkeypatch, tmp_path, capsys, runs, "timecourse", *common,
                      "--runs-root", runs_root(tmp_path, journal_records))  # fmt: skip
        assert lacking in err
    assert timecourse_guard({(1, 5), (6, 12)}) == [] and timecourse_guard({(1, 10)}) == []
    (gaps,) = timecourse_guard({(1, 2), (4, 6), (8, 9)})
    assert "blocks 3, 7, 10 are not in an issued range (3 of 10)" in gaps
    root = runs_root(tmp_path, issued(1, 5) + issued(6, 12))
    monkeypatch.undo()  # the versions record no longer patched to hold
    err = refusal(monkeypatch, tmp_path, capsys, runs, "timecourse", *common, "--runs-root", root,
                  "--versions", str(tmp_path / "none.json"), versions=False)  # fmt: skip
    assert "the versions record does not hold" in err and "none.json is missing" in err
    found = campaign(monkeypatch, tmp_path, runs, "timecourse", *common, "--runs-root", root)
    central = {ADAMW: 1.48e-5, FA: 1.86e-5, PREWARMED: 1.48e-5}
    assert {arm: sorted(lrs)[1] for arm, lrs in rates.items()} == central
    assert len(found) == 30 and {r.seed for r in found} == set(TIMECOURSE_SEEDS)
    assert {r.cell for r in found} == {f"{arm}-b20-{lr:g}-timecourse-rayleigh"
                                       for arm, lr in central.items()}  # fmt: skip
    assert {r.experiment for r in found} == {"study2_short"}
    # Its cells never count as an issued confirmation range.
    assert issued_ranges([{"event": "start", "experiment": "study2_short",
                           "run": f"{r.cell}/s{r.seed}"} for r in found]) == set()  # fmt: skip
    # Each run is its endpoint run's config but for the role and the evaluation steps.
    endpoints = campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", "1", "10",
                         *common, "--runs-root", runs_root(tmp_path / "other"))  # fmt: skip
    twins = {(r.cell.removesuffix("-blocks-1-10-rayleigh"), r.seed): r for r in endpoints}
    for run in found:
        twin = twins[(run.cell.removesuffix("-timecourse-rayleigh"), run.seed)]
        a = OmegaConf.to_container(load_run_config(run.config, list(run.overrides)))
        b = OmegaConf.to_container(load_run_config(twin.config, list(twin.overrides)))
        assert (a["study2_short"].pop("role"), b["study2_short"].pop("role")) == (
            TIMECOURSE, "confirmation")  # fmt: skip
        assert (a["schedule"].pop("evaluate_at"), b["schedule"].pop("evaluate_at")) == (
            list(TIMECOURSE_STEPS), [25])  # fmt: skip
        assert a == b and a["eval"]["rayleigh"]["enabled"] and "k_only" not in a["eval"]
        assert run.config == twin.config == "experiments/study2_short_rayleigh"
    split = campaign(monkeypatch, tmp_path, runs, "timecourse", *common, "--runs-root", root,
                     "--seeds", "3000", "3001")  # fmt: skip
    assert len(split) == 6 and {r.seed for r in split} == {3000, 3001}
    # The runs still missing only; none once every run is there.
    made_runs = courses(rates, 20, TIMECOURSE_SEEDS)
    rest = campaign(monkeypatch, tmp_path, [*runs, *made_runs[3:]], "timecourse", *common,
                    "--runs-root", root)  # fmt: skip
    assert sorted((r.cell, r.seed) for r in rest) == sorted(
        (f"{arm}-b20-{lr:g}-timecourse-rayleigh", 3000) for arm, lr in central.items())  # fmt: skip
    err = refusal(monkeypatch, tmp_path, capsys, [*runs, *made_runs], "timecourse", *common,
                  "--runs-root", root)  # fmt: skip
    assert "the time course has every run already" in err
    # Without the pre-warmed arm (the primary-only fallback): 20 runs.
    primary = [*base()]
    real = SCALE.copy()
    try:
        for batch in (5, 20):
            SCALE[(PREWARMED, batch)] = 0.3
        frozen = until(primary, "confirmation")["frozen"]
    finally:
        SCALE.update(real)
    assert frozen["prewarmed_dropped"]
    found = campaign(monkeypatch, tmp_path, primary, "timecourse", *common, "--runs-root", root)
    assert len(found) == 20 and {r.cell.split("-b20")[0] for r in found} == {ADAMW, FA}


def range_journal(block_runs) -> list[dict]:
    """The launcher's records of the confirmation range 1-10, every run started and done."""
    started = [{"event": "start", "run": f"{r.arm}-b20-{r.lr:g}-blocks-1-10-rayleigh/s{r.seed}",
                "experiment": "study2_short", "config_hash": r.run_id, "run_id": r.run_id,
                "attempt": 1, "command": []} for r in block_runs]  # fmt: skip
    return started + [{"event": "end", "run_id": r.run_id, "outcome": "done"} for r in block_runs]


def test_a_confirmation_range_offers_an_endpoint_lost_beside_its_timecourse_twin(
    monkeypatch, tmp_path
) -> None:
    """Review of Task 5, I1: block 3's AdamW-25 endpoint run is lost after its time-course twin
    finished (and another twin was given up by the launcher). Re-issuing the range 1-10 offers
    that endpoint run: a time-course run never satisfies a confirmation cell."""
    runs = base()
    rates = until(runs, "confirmation")["next"]["rates"]
    central = {arm: sorted(lrs)[1] for arm, lrs in rates.items()}
    block_runs = blocks_of(rates, 20, CONFIRMATION_SEEDS[:10])
    lost = next(r for r in block_runs if (r.arm, r.lr, r.seed) == (ADAMW, central[ADAMW], 3002))
    runs += [r for r in block_runs if r is not lost]
    runs.append(course(ADAMW, 20, central[ADAMW], 3002))  # its twin, done
    gone = next(r for r in block_runs if (r.arm, r.lr, r.seed) == (FA, central[FA], 3004))
    runs.remove(gone)  # lost too, and its time-course twin given up by the launcher
    cell = f"{FA}-b20-{central[FA]:g}-timecourse-rayleigh"
    overrides = ["optimizer=adam_frozen", "train.batch_size=20", f"optimizer.lr={central[FA]:g}",
                 *study2_short.TIMECOURSE_OVERRIDES]  # fmt: skip
    records = range_journal(block_runs) + journal(
        cell, 3004, overrides, attempts=2 * MAX_ATTEMPTS, recovered=True,
        config="experiments/study2_short_rayleigh")  # fmt: skip
    assert (FA, 20, central[FA], 3004, TIMECOURSE) in exhausted_runs(records)
    state = stage(runs)
    assert not state["problems"] and state["confirmation"][2]["runs"] == 8
    args = ("--fixation", fixed(tmp_path), "--runs-root", runs_root(tmp_path, records))
    found = campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", "1", "10", *args)
    assert sorted(labels_of(found)) == sorted([
        f"{ADAMW}-b20-{central[ADAMW]:g}-blocks-1-10-rayleigh/s3002",
        f"{FA}-b20-{central[FA]:g}-blocks-1-10-rayleigh/s3004"])  # fmt: skip


def test_the_timecourse_campaign_leaves_out_runs_whose_endpoint_was_given_up(
    monkeypatch, tmp_path
) -> None:
    """Review of Task 5, M3: a time-course run whose endpoint run the launcher gave up on could
    never show its identity; the campaign does not run it and names it in its header."""
    runs = base()
    rates = until(runs, "confirmation")["next"]["rates"]
    central = {arm: sorted(lrs)[1] for arm, lrs in rates.items()}
    block_runs = blocks_of(rates, 20, CONFIRMATION_SEEDS[:10])
    gone = next(r for r in block_runs if (r.arm, r.lr, r.seed) == (ADAMW, central[ADAMW], 3005))
    runs += [r for r in block_runs if r is not gone]
    cell = f"{ADAMW}-b20-{central[ADAMW]:g}-blocks-1-10-rayleigh"
    overrides = ["train.batch_size=20", f"optimizer.lr={central[ADAMW]:g}",
                 "study2_short.role=confirmation"]  # fmt: skip
    records = range_journal(block_runs) + journal(
        cell, 3005, overrides, attempts=2 * MAX_ATTEMPTS, recovered=True,
        config="experiments/study2_short_rayleigh")  # fmt: skip
    label = f"{cell}/s3005"
    assert exhausted_runs(records)[(ADAMW, 20, central[ADAMW], 3005)] == label
    summary = stage(runs, exhausted=exhausted_runs(records))["timecourse"]
    assert [r for r in summary["missing"] if "endpoint_given_up" in r] == [
        {"seed": 3005, "arm": ADAMW, "lr": central[ADAMW], "endpoint_given_up": label}]  # fmt: skip
    out = script(monkeypatch, tmp_path, CAMPAIGNS, runs, "timecourse", "--fixation",
                 fixed(tmp_path), "--runs-root", runs_root(tmp_path, records))  # fmt: skip
    found = load_campaign(out)[1]
    assert len(found) == 29 and (f"{ADAMW}-b20-{central[ADAMW]:g}-timecourse-rayleigh",
                                 3005) not in {(r.cell, r.seed) for r in found}  # fmt: skip
    header = " ".join(line[2:] for line in out.read_text().splitlines() if line.startswith("#"))
    assert (f"Left out: the time course's {ADAMW} run at {central[ADAMW]:g} on seed 3005, whose "
            f"endpoint run the launcher gave up on ({label}): its identity could never be "
            "shown.") in header  # fmt: skip


def test_the_numerical_verdicts_are_read_from_the_runs_root(tmp_path) -> None:
    root = tmp_path / "runs"
    kept = root / "numerical" / "b20-u8" / "verdicts"
    kept.mkdir(parents=True)
    (kept / "0123456789ab.json").write_text('{"batch": 20}')
    (kept / "ba9876543210.json").write_text("{not json")
    (kept.parent / "verdict.json").write_text('{"batch": 5}')  # the copy for people: not read
    earlier = kept / "history" / "0123456789ab"  # where the tool moves a selection's earlier one
    earlier.mkdir(parents=True)
    (earlier / "2026-10-19T120000+0000-0.json").write_text('{"batch": 20}')
    found = read_numerical(root)
    assert sorted(found) == [
        "numerical/b20-u8/verdicts/0123456789ab.json",
        "numerical/b20-u8/verdicts/ba9876543210.json",
        "numerical/b20-u8/verdicts/history/0123456789ab/2026-10-19T120000+0000-0.json"]  # fmt: skip
    assert found["numerical/b20-u8/verdicts/0123456789ab.json"] == {"batch": 20}
    assert found["numerical/b20-u8/verdicts/ba9876543210.json"].startswith("unreadable: ")
    assert read_numerical(tmp_path / "nowhere") == {}


def joined(text: str) -> list[str]:
    """The driver's commands, continued lines joined and indentation dropped."""
    return [line.strip() for line in re.sub(r" *\\\n *", " ", text).splitlines()]


def test_the_prep_driver(monkeypatch, tmp_path, capsys) -> None:
    def prep(*args: str) -> str:
        out = tmp_path / f"prep-{len(list(tmp_path.glob('prep-*')))}.sh"
        monkeypatch.setattr("sys.argv", [str(CAMPAIGNS), "prep", *args, "--commit", COMMIT,
                                         "--out", str(out)])  # fmt: skip
        runpy.run_path(str(CAMPAIGNS), run_name="__main__")
        subprocess.run(["bash", "-n", str(out)], check=True)
        return out.read_text()

    text = prep("--seeds", "2000", "2001", "--batch", "5", "--archived", "--once",
                "--stage0-run-id", "stage0-v3-test-01")  # fmt: skip
    assert f"SRC=/workspace/code/zero-field-refusal/{COMMIT}/source" in text
    assert all(len(line) <= 100 for line in text.splitlines() if line.startswith("#"))
    header = " ".join(line[2:] for line in text.splitlines() if line.startswith("# "))
    # The header names the inventories per campaign kind (docs/operations.md), each in full.
    module = campaigns_module()
    for name in (*module.INVENTORIES, module.V3_RECORD, module.CURVATURE_PANEL):
        assert name in header
    # Review M9: no FA control runs under amendment 4; the v3 record is restored for
    # pod_setup.sh's check of the whole DATA_DIR
    assert "FA controls" not in header
    assert (f"the stage-0 v3 record {module.V3_RECORD} (no run of amendment 4 reads it: it is "
            "restored because pod_setup.sh checks the whole DATA_DIR against the "
            "inventories)") in header  # fmt: skip
    assert f"for confirmation blocks 1-10 also the curvature panel {module.CURVATURE_PANEL}" in (
        header
    )
    assert "can be run again" in header and re.search(r"prep at [0-9a-f]{40}", header)
    commands = joined(text)
    run = 'project-run "$SRC" python scripts/'
    # The checkout is at the commit and clean as the runs read it, not as git status says.
    first_step = next(i for i, line in enumerate(commands) if line.startswith("step() {"))
    assert commands.index(f'test "$(git rev-parse HEAD)" = {COMMIT}') < commands.index(
        f'project-run "$SRC" python -c "{module.CLEAN}"') < first_step  # fmt: skip
    assert "git status" not in text
    assert "STAGE0=stage0-v3-test-01" in commands and "stage0" in commands
    stage0 = (f'step env RUN_ID="$STAGE0" {run}stage0_checks.py experiments/stage0_checks '
              "model=qwen2.5-1.5b-instruct protocol_version=3")  # fmt: skip
    # The restored soft-target copy is set aside before stage 0 and put back after it passed.
    aside = f'step {run}study2_short_campaigns.py soft-targets-aside --stage0-run-id "$STAGE0"'
    back = f'step {run}study2_short_campaigns.py soft-targets-restore --stage0-run-id "$STAGE0"'
    assert commands.index(aside) < commands.index(stage0) < commands.index(back)
    assert (f"kept {run}generate_teacher.py experiments/study2_short_rayleigh --stream "
            "curvature_panel") in commands  # fmt: skip
    # Archived seeds: their streams are checked where every run locates them, never generated.
    assert (f"step {run}study2_short_campaigns.py check-streams --seeds 2000 2001 "
            "--batch 5") in commands  # fmt: skip
    assert "--stream training" not in text and "--stream preconditioner" not in text

    text = prep("--seeds", "2100", "--batch", "20")  # fresh seeds: generated, no stage 0
    commands = joined(text)
    assert "stage0_checks" not in text and "curvature_panel" not in text
    assert "soft-targets" not in text
    assert (f"step {run}generate_teacher.py experiments/study2_stage_a --stream training "
            "--seeds 2100 --responses 500") in commands  # fmt: skip
    assert (f"kept {run}generate_teacher.py experiments/study2_stage_a optimizer=adam_frozen "
            "train.batch_size=20 --stream preconditioner --seeds 2100") in commands  # fmt: skip
    for bad, why in ((["--once"], "--once and --stage0-run-id go together"),
                     (["--stage0-run-id", "x"], "--once and --stage0-run-id go together"),
                     (["--seeds", "7"], "distinct seeds of the stage")):  # fmt: skip
        with pytest.raises(SystemExit):
            prep("--seeds", "2000", "--batch", "5", *bad)
        assert why in capsys.readouterr().err


def test_v0_at_b20_is_1000_responses_and_the_numerical_seeds_get_it(monkeypatch, tmp_path):
    cfg = load_run_config("experiments/study2_stage_a", ["optimizer=adam_frozen",
                                                         "train.batch_size=20"])  # fmt: skip
    assert cfg.optimizer.preconditioner.responses == 50 * 20
    out = tmp_path / "prep.sh"
    monkeypatch.setattr(
        "sys.argv",
        [
            str(CAMPAIGNS),
            "prep",
            "--seeds",
            "2200",
            "2201",
            "2202",
            "--batch",
            "20",
            "--commit",
            COMMIT,
            "--out",
            str(out),
        ],
    )
    runpy.run_path(str(CAMPAIGNS), run_name="__main__")
    commands = joined(out.read_text())
    run = 'project-run "$SRC" python scripts/'
    for seed in NUMERICAL_SEEDS:
        assert (f"step {run}generate_teacher.py experiments/study2_stage_a --stream training "
                f"--seeds {seed} --responses 500") in commands  # fmt: skip
        assert (
            f"kept {run}generate_teacher.py experiments/study2_stage_a optimizer=adam_frozen "
            f"train.batch_size=20 --stream preconditioner --seeds {seed}"
        ) in commands


def test_the_discovery_seeds_b20_v0_streams_are_archived() -> None:
    """The B = 20 v₀ streams of seeds 2000-2002 that the archived B = 20 FA runs read are those
    the preconditioner inventories every Pod restores list (runs/archive), so no discovery seed
    needs a new one; the numerical seeds' are generated by ``prep``."""
    from test_study2_short import RUNS, _numeric

    module = campaigns_module()
    archive, grids = RUNS / "archive", RUNS / "study2_stage_a"
    if not archive.is_dir() or not grids.is_dir():
        pytest.skip(f"the archived records are not under {RUNS}")
    read = {}
    for config in sorted(grids.rglob("config.yaml")):
        cfg = OmegaConf.load(config)
        if (
            OmegaConf.select(cfg, "optimizer.name"),
            OmegaConf.select(cfg, "train.batch_size"),
            OmegaConf.select(cfg, "schedule.policy"),
        ) != ("adam_frozen", 20, "fixed"):
            continue
        meta = _numeric((config.parent / "meta.json").read_text())
        stream = meta["inputs"]["streams"]["preconditioner"]
        read.setdefault(int(cfg.seed), set()).add((stream["path"], stream["sha256"]))
    assert sorted(read) == [2000, 2001, 2002] and all(len(v) == 1 for v in read.values())
    for seed, ((path, digest),) in read.items():
        assert "/preconditioner/2048b878f867/" in path  # B = 20's stream, not B = 5's
        name = f"inventory-2026-10-14-study2-a-preconditioner-s{seed}.tsv"
        assert name in module.INVENTORIES
        rows = [line.split("\t") for line in (archive / name).read_text().splitlines()[1:]]
        assert [path + "/samples.jsonl", digest] in [[row[0], row[2]] for row in rows]


def driver_functions(text: str) -> str:
    return text[text.index("step() {") : text.index("# The steps")]


def campaigns_module():
    spec = importlib.util.spec_from_file_location("campaigns", CAMPAIGNS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("dirty", "stops"), [(False, False), (True, True), (None, True)])
def test_the_driver_stops_unless_the_checkout_is_clean(monkeypatch, dirty, stops) -> None:
    """The driver's check, as tracking.git_state reads the checkout (None: no checkout)."""
    module = campaigns_module()
    monkeypatch.setattr("zero_field_refusal.tracking.git_state",
                        lambda: {"commit": COMMIT, "dirty": dirty})  # fmt: skip
    with pytest.raises(SystemExit) as found:
        exec(module.CLEAN, {})
    assert (found.value.code is not None) == stops
    if stops:
        assert "the source checkout is not clean" in found.value.code


def test_the_driver_keeps_a_stream_that_exists_and_stops_on_other_failures() -> None:
    module = campaigns_module()
    text = module.prep_driver([2000], 5, COMMIT, once=False, stage0_run_id=None, archived=False,
                              written_at=COMMIT)  # fmt: skip
    body = f"""set -euo pipefail
{driver_functions(text)}
exists() {{ echo "StreamError: x {module.EXISTS}" >&2; return 1; }}
broken() {{ echo "StreamError: partial directories" >&2; return 1; }}
kept exists
echo after-exists
kept broken
echo after-broken
"""
    found = subprocess.run(["bash", "-c", body], capture_output=True, text=True)
    assert "after-exists" in found.stdout and "after-broken" not in found.stdout
    assert found.returncode != 0


def _stage0_record(run_dir, status: str, version: int, passed: bool) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps({"status": status}))
    (run_dir / "summary.json").write_text(json.dumps({"path_field": {"passed": passed}}))
    (run_dir / "config.yaml").write_text(f"protocol_version: {version}\n")


@pytest.mark.parametrize(
    ("record", "fresh", "ok"),
    [
        (None, ("completed", 3, True), True),  # not run yet: run it, and it passed
        (None, ("completed", 3, False), False),  # run now, completed, path field failed
        (("completed", 3, True), None, True),  # run again after a later failure: kept
        (("completed", 3, False), None, False),  # completed but not passed: stop
        (("failed", 3, True), None, False),  # did not complete: stop
        (("completed", 2, True), None, False),  # another protocol version: stop
    ],
)
def test_the_driver_runs_stage_0_once_and_keeps_only_a_passed_record(
    tmp_path: Path, record: tuple | None, fresh: tuple | None, ok: bool
) -> None:
    module = campaigns_module()
    text = module.prep_driver([2000], 5, COMMIT, once=True, stage0_run_id="s0", archived=True,
                              written_at=COMMIT)  # fmt: skip
    run_dir = tmp_path / "stage0_checks" / "s0"
    if record is not None:
        _stage0_record(run_dir, *record)
    write = ""
    if fresh is not None:  # the stubbed stage-0 run writes its record
        status, version, passed = fresh
        write = (f"mkdir -p {run_dir}; echo '{{\"status\": \"{status}\"}}' > {run_dir}/meta.json; "
                 f"echo '{{\"path_field\": {{\"passed\": {str(passed).lower()}}}}}' "
                 f"> {run_dir}/summary.json; echo 'protocol_version: {version}' > "
                 f"{run_dir}/config.yaml")  # fmt: skip
    body = f"""set -euo pipefail
RUNS_DIR={tmp_path}
SRC=/src
{driver_functions(text)}
project-run() {{ shift; shift; "{sys.executable}" "$@"; }}  # SRC, then python
step() {{ echo "STEP $*"; {write or ":"}; }}
stage0
echo after-stage0
"""
    found = subprocess.run(["bash", "-c", body], capture_output=True, text=True)
    assert ("STEP env RUN_ID=s0" in found.stdout) == (record is None)
    # A kept run's copy is not set aside again; only a passed run's stream is put back.
    assert ("soft-targets-aside --stage0-run-id s0" in found.stdout) == (record is None)
    assert ("soft-targets-restore --stage0-run-id s0" in found.stdout) == ok
    if ok:
        assert "passed under protocol version 3" in found.stdout and "after-stage0" in found.stdout
    else:
        assert found.returncode != 0 and "with a new --stage0-run-id" in found.stderr
        assert "after-stage0" not in found.stdout


def test_check_streams_locates_the_streams_and_names_the_missing(monkeypatch, tmp_path, capsys):
    data = tmp_path / "data"
    for seed in (2000, 2001):
        chunk = data / "training" / f"seed-{seed}" / "chunk-00000"
        chunk.mkdir(parents=True)
        (chunk / "manifest.json").write_text(json.dumps({"sha256": f"t{seed}"}))
    (data / "preconditioner" / "seed-2000").mkdir(parents=True)
    (data / "preconditioner" / "seed-2000" / "manifest.json").write_text('{"sha256": "p2000"}')

    def locate(cfg, env, kind, seed):
        return {}, data / kind / f"seed-{seed}", []

    monkeypatch.setattr("zero_field_refusal.runtime.environment", lambda cfg, device: object())
    monkeypatch.setattr("zero_field_refusal.runtime.locate_stream", locate)
    monkeypatch.setattr("sys.argv", [str(CAMPAIGNS), "check-streams", "--seeds", "2000",
                                     "--batch", "5"])  # fmt: skip
    runpy.run_path(str(CAMPAIGNS), run_name="__main__")
    shown = capsys.readouterr().out
    assert "1 chunks, chunk 0 sha256 t2000" in shown and "sha256 p2000" in shown
    monkeypatch.setattr("sys.argv", [str(CAMPAIGNS), "check-streams", "--seeds", "2000", "2001",
                                     "--batch", "5"])  # fmt: skip
    with pytest.raises(SystemExit, match="seed 2001: no preconditioner stream"):
        runpy.run_path(str(CAMPAIGNS), run_name="__main__")


# ------------------------------------------------------------------ stage 0's soft-target copy

SAMPLES = '{"id": "p1"}\n{"id": "p2"}\n'  # synthetic samples.jsonl
DISTRIBUTIONS = b"synthetic distributions"


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stream_copy(path: Path, *, driver: str, day: str, samples: str = SAMPLES,
                distributions: bytes | None = None) -> None:  # fmt: skip
    """A soft-target stream directory: its manifest records the archived hashes."""
    path.mkdir(parents=True)
    manifest = {"sha256": sha256_of(SAMPLES.encode()), "count": 2, "started_at": f"{day}T10:00:00",
                "hardware": {"driver": driver, "gpus": ["NVIDIA L40S"]},
                "distributions": {"file": "distributions.safetensors", "samples": 2,
                                  "sha256": sha256_of(DISTRIBUTIONS)}}  # fmt: skip
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (path / "samples.jsonl").write_text(samples)
    if distributions is not None:
        (path / "distributions.safetensors").write_bytes(distributions)


@pytest.fixture
def pod(monkeypatch, tmp_path):
    """A fake Pod: DATA_DIR and RUNS_DIR under tmp_path, the stage-0 config's soft-target stream
    located by the real locator (the environment's prompt set is synthetic), and the CLI."""
    import types

    data, runs, aside = tmp_path / "data", tmp_path / "runs", tmp_path / "workspace-tmp"
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("RUNS_DIR", str(runs))
    prompts = [{"id": "t1", "text": "a teacher prompt"}]
    monkeypatch.setattr("zero_field_refusal.runtime.environment",
                        lambda cfg, device: types.SimpleNamespace(prompt_set=lambda name: prompts,
                                                                  data_root=data))  # fmt: skip
    module = campaigns_module()
    path, root = module.stage0_soft_targets()

    def cli(kind: str) -> None:
        monkeypatch.setattr("sys.argv", [str(CAMPAIGNS), kind, "--stage0-run-id", "s0",
                                         "--aside-root", str(aside)])  # fmt: skip
        runpy.run_path(str(CAMPAIGNS), run_name="__main__")

    return types.SimpleNamespace(module=module, path=path, data=root, runs=runs, aside=aside,
                                 cli=cli, run_dir=runs / "stage0_checks" / "s0")  # fmt: skip


def test_the_soft_target_copy_is_located_as_stage_0_locates_it(pod) -> None:
    from zero_field_refusal import runtime
    from zero_field_refusal.checks import path_field_config

    relative = pod.path.relative_to(pod.data)
    assert relative.parts[:3] == ("teacher", "Qwen--Qwen2.5-1.5B-Instruct", "soft_targets")
    assert re.fullmatch(r"[0-9a-f]{12}", relative.parts[3]) and relative.parts[4] == "seed-0"
    cfg = load_run_config("experiments/stage0_checks",
                          ["model=qwen2.5-1.5b-instruct", "protocol_version=3"])  # fmt: skip
    env = runtime.environment(cfg, "cpu")  # the fake Pod's
    spec, path, _ = runtime.locate_stream(path_field_config(cfg), env, "soft_targets", 0)
    assert path == pod.path and spec["responses"] == cfg.checks.path_field.batches * 20


def test_stage_0_sets_the_copy_aside_compares_and_puts_it_back(pod, capsys) -> None:
    stream_copy(pod.path, driver="595.91.07", day="2026-10-09")
    archived = (pod.path / "manifest.json").read_bytes()
    pod.cli("soft-targets-aside")
    assert "set aside" in capsys.readouterr().out and not pod.path.exists()
    (dated,) = pod.aside.iterdir()
    assert re.fullmatch(r"soft-targets-aside-\d{8}", dated.name)
    away = dated / pod.path.relative_to(pod.data)
    assert (away / "manifest.json").read_bytes() == archived
    pod.cli("soft-targets-aside")  # run again: the copy stays where it is
    assert "already set aside" in capsys.readouterr().out and not pod.path.exists()

    # Stage 0 regenerates the stream, distributions included, and passes.
    stream_copy(pod.path, driver="580.159.03", day="2026-10-16", distributions=DISTRIBUTIONS)
    regenerated = (pod.path / "manifest.json").read_text()
    pod.run_dir.mkdir(parents=True)
    (pod.run_dir / "meta.json").write_text(json.dumps({"git": {"commit": COMMIT}}))
    pod.cli("soft-targets-restore")
    assert "samples equal True, distributions equal True" in capsys.readouterr().out
    record = json.loads((pod.run_dir / "soft_targets_regeneration.json").read_text())
    assert set(record) == {"what", "pod", "archived", "regenerated", "comparison", "after"}
    assert record["archived"] == {
        "driver": "595.91.07", "generated": "2026-10-09",
        "manifest_file_sha256": sha256_of(archived), "samples_sha256": sha256_of(SAMPLES.encode()),
        "distributions_sha256": sha256_of(DISTRIBUTIONS)}  # fmt: skip
    assert record["regenerated"]["samples_sha256"] == sha256_of(SAMPLES.encode())
    assert record["regenerated"]["distributions_sha256"] == sha256_of(DISTRIBUTIONS)
    assert record["regenerated"]["distributions_bytes"] == len(DISTRIBUTIONS)
    assert record["regenerated"]["manifest_file"] == "soft_targets_manifest_regenerated.json"
    assert record["comparison"] == {"samples_equal": True, "distributions_equal": True,
                                    "manifest_fields_differing": ["hardware.driver",
                                                                  "started_at"]}  # fmt: skip
    assert "s0" in record["what"] and COMMIT in record["what"]
    assert (pod.run_dir / "soft_targets_manifest_regenerated.json").read_text() == regenerated
    # DATA_DIR holds the archived copy again, without distributions; nothing was deleted but them.
    assert (pod.path / "manifest.json").read_bytes() == archived
    assert sorted(p.name for p in pod.path.iterdir()) == ["manifest.json", "samples.jsonl"]
    rest = dated / "s0" / pod.path.relative_to(pod.data)
    assert (rest / "manifest.json").read_text() == regenerated
    assert not (rest / "distributions.safetensors").exists()
    written = (pod.run_dir / "soft_targets_regeneration.json").read_text()
    pod.cli("soft-targets-restore")  # run again: nothing changes
    assert "kept: the archived copy is back" in capsys.readouterr().out
    assert (pod.run_dir / "soft_targets_regeneration.json").read_text() == written
    assert (pod.path / "manifest.json").read_bytes() == archived


def test_put_back_resumes_where_it_stopped(pod) -> None:
    module = pod.module
    stream_copy(pod.path, driver="595", day="2026-10-09")
    module.set_aside(pod.path, pod.data, pod.aside, "20261016")
    # A failed attempt regenerated the stream already: setting aside again keeps both in place.
    stream_copy(pod.path, driver="580", day="2026-10-16", distributions=DISTRIBUTIONS)
    assert "already set aside" in module.set_aside(pod.path, pod.data, pod.aside, "20261017")
    assert (pod.path / "distributions.safetensors").read_bytes() == DISTRIBUTIONS
    pod.run_dir.mkdir(parents=True)
    (pod.run_dir / "meta.json").write_text("{}")
    # Interrupted right after the comparison was written: the rest is done, the record kept.
    relative, dated = pod.path.relative_to(pod.data), pod.aside / "soft-targets-aside-20261016"
    found = module.regeneration_record(dated / relative, pod.path, pod.run_dir, stream=relative,
                                       dated=dated)  # fmt: skip
    assert found["comparison"]["samples_equal"] and found["comparison"]["distributions_equal"]
    (pod.run_dir / "soft_targets_regeneration.json").write_text(json.dumps(found))
    done = module.restore(pod.path, pod.data, pod.aside, pod.run_dir)
    assert "compared" not in done and "put back" in done
    assert json.loads((pod.run_dir / "soft_targets_regeneration.json").read_text()) == found
    assert not (pod.path / "distributions.safetensors").exists()
    assert json.loads((pod.path / "manifest.json").read_text())["hardware"]["driver"] == "595"
    assert (dated / "s0" / relative / "manifest.json").is_file()


def test_a_regeneration_that_differs_from_the_archive_stops_the_driver(pod, capsys) -> None:
    stream_copy(pod.path, driver="595", day="2026-10-09")
    archived = (pod.path / "manifest.json").read_bytes()
    pod.cli("soft-targets-aside")
    # The samples are the same, the distributions are not: any difference stops.
    stream_copy(pod.path, driver="580", day="2026-10-16", distributions=b"other")
    pod.run_dir.mkdir(parents=True)
    (pod.run_dir / "meta.json").write_text("{}")
    with pytest.raises(SystemExit, match="the regeneration differs from the archive") as stop:
        pod.cli("soft-targets-restore")
    assert isinstance(stop.value.code, str)  # a message: exit status 1
    record = json.loads((pod.run_dir / "soft_targets_regeneration.json").read_text())
    assert record["comparison"]["samples_equal"] and not record["comparison"]["distributions_equal"]
    assert record["after"].startswith("The regeneration differs from the archive")
    # DATA_DIR matches its inventories again; the whole regenerated stream waits for inspection.
    assert (pod.path / "manifest.json").read_bytes() == archived
    assert sorted(p.name for p in pod.path.iterdir()) == ["manifest.json", "samples.jsonl"]
    (dated,) = pod.aside.iterdir()
    relative = pod.path.relative_to(pod.data)
    regenerated = dated / "s0" / f"{relative}-regenerated"  # under the run's id
    assert regenerated.name == "seed-0-regenerated"
    assert regenerated.parent == dated / "s0" / relative.parent
    assert str(regenerated) in record["after"]
    assert (regenerated / "distributions.safetensors").read_bytes() == b"other"
    assert (pod.run_dir / "soft_targets_manifest_regenerated.json").is_file()
    assert pod.module.set_aside_copies(pod.path, pod.data, pod.aside) == []
    written = (pod.run_dir / "soft_targets_regeneration.json").read_text()
    capsys.readouterr()
    with pytest.raises(SystemExit, match="the regeneration differs from the archive"):
        pod.cli("soft-targets-restore")  # run again: it still stops, and nothing moves
    assert (pod.run_dir / "soft_targets_regeneration.json").read_text() == written
    assert (pod.path / "manifest.json").read_bytes() == archived
    assert (regenerated / "distributions.safetensors").read_bytes() == b"other"


def test_two_differing_regenerations_on_one_day_keep_their_own_places(pod) -> None:
    # After a stop, the lead's decision, then a new --stage0-run-id on the same UTC day.
    module = pod.module
    stream_copy(pod.path, driver="595", day="2026-10-09")
    archived = (pod.path / "manifest.json").read_bytes()
    relative = pod.path.relative_to(pod.data)
    places = []
    for run_id, other in (("s0", b"first"), ("s1", b"second")):
        run_dir = pod.runs / "stage0_checks" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "meta.json").write_text("{}")
        assert "set aside" in module.set_aside(pod.path, pod.data, pod.aside, "20261017")
        stream_copy(pod.path, driver="580", day="2026-10-17", distributions=other)
        with pytest.raises(SystemExit, match="differs from the archive.*new --stage0-run-id"):
            module.restore(pod.path, pod.data, pod.aside, run_dir)
        # Each stop leaves DATA_DIR as its inventories say, and its own regeneration aside.
        assert (pod.path / "manifest.json").read_bytes() == archived
        assert not (pod.path / "distributions.safetensors").exists()
        place = pod.aside / "soft-targets-aside-20261017" / run_id / f"{relative}-regenerated"
        assert (place / "distributions.safetensors").read_bytes() == other
        record = json.loads((run_dir / "soft_targets_regeneration.json").read_text())
        assert str(place) in record["after"]
        places.append(place)
    assert (places[0] / "distributions.safetensors").read_bytes() == b"first"
    assert module.set_aside_copies(pod.path, pod.data, pod.aside) == []


def test_the_soft_target_steps_refuse_what_they_cannot_resolve(pod) -> None:
    module = pod.module
    with pytest.raises(SystemExit, match="no restored soft-target copy"):
        module.set_aside(pod.path, pod.data, pod.aside, "20261016")
    pod.run_dir.mkdir(parents=True)
    with pytest.raises(SystemExit, match="no soft-target stream at .*pod_setup"):  # lost
        module.restore(pod.path, pod.data, pod.aside, pod.run_dir)
    stream_copy(pod.path, driver="595", day="2026-10-09")  # the archived copy, in place
    assert "nothing to compare" in module.restore(pod.path, pod.data, pod.aside, pod.run_dir)
    shutil.rmtree(pod.path)
    stream_copy(pod.path, driver="580", day="2026-10-16", distributions=DISTRIBUTIONS)
    with pytest.raises(SystemExit, match="clean up by hand"):  # regenerated, nothing set aside
        module.restore(pod.path, pod.data, pod.aside, pod.run_dir)
    module.set_aside(pod.path, pod.data, pod.aside, "20261016")
    with pytest.raises(SystemExit, match="no regenerated distributions"):  # stage 0 made none
        module.restore(pod.path, pod.data, pod.aside, pod.run_dir)
    stream_copy(pod.path, driver="580", day="2026-10-16", distributions=DISTRIBUTIONS)
    module.set_aside(pod.path, pod.data, pod.aside, "20261016")  # kept: already set aside
    second = pod.aside / "soft-targets-aside-20261017" / pod.path.relative_to(pod.data)
    second.mkdir(parents=True)
    with pytest.raises(SystemExit, match="several copies"):
        module.restore(pod.path, pod.data, pod.aside, pod.run_dir)


# ------------------------------------------------------------------ the summary script


def test_the_summary_prints_the_status_and_refuses_looks_without_a_fixation(
    monkeypatch, tmp_path, capsys
) -> None:
    runs = base()
    script(monkeypatch, tmp_path, SUMMARY, runs, "--w-max", str(W_MAX))
    shown = json.loads(capsys.readouterr().out)
    assert shown["next"]["campaign"] == "discovery" and len(shown["next"]["runs"]) == 36
    assert [[c["batch"], c["U"]] for c in shown["candidates"]] == [[20, 8.0], [20, 7.0],
                                                                   [5, 8.0], [5, 7.0]]  # fmt: skip
    assert shown["attempts"][0]["arms"][ADAMW][0]["feasible"] is False
    assert shown["attempts"][0]["three_arm"] and shown["amendment3_controls"] == []
    assert shown["not_reached"] == []
    with pytest.raises(SystemExit):
        script(monkeypatch, tmp_path, SUMMARY, runs, "--w-max", str(W_MAX), "--judge", "wildguard")
    assert "--judge needs the fixation record (--fixation)" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        script(monkeypatch, tmp_path, SUMMARY, runs)
    assert "w_max is needed" in capsys.readouterr().err


def on_disk(runs, tmp_path: Path) -> list:
    """The runs, each with a directory holding its config's prompt set."""
    out = []
    for run in runs:
        run_dir = tmp_path / "runs" / run.run_id
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text("eval: {harmful: {prompt_set: harmful_evaluation}}\n"
                                             "data: {}\n")  # fmt: skip
        out.append(dataclasses.replace(run, run_dir=run_dir))
    return out


def test_the_summary_prints_the_looks_end_to_end(monkeypatch, tmp_path, capsys) -> None:
    runs = base()
    confirmed(runs)
    runs = on_disk(runs, tmp_path)
    by_name = {run.run_dir.name: run for run in runs}
    monkeypatch.setattr(
        "zero_field_refusal.analysis.evidence.label_path",
        lambda run_dir, judge: run_dir / "config.yaml",
    )  # every run is labeled
    monkeypatch.setattr("zero_field_refusal.data.load_prompt_set",
                        lambda name, data: [{"id": "p", "text": "t"}])  # fmt: skip
    monkeypatch.setattr("zero_field_refusal.analysis.study2_labels.read_endpoint_labels",
                        lambda run_dir, judge, identity, prompts:
                        rows(100 if by_name[run_dir.name].arm == ADAMW else 50))  # fmt: skip
    out = script(monkeypatch, tmp_path, SUMMARY, runs, "--fixation", fixation_file(tmp_path),
                 "--runs-root", runs_root(tmp_path), "--judge", "wildguard")  # fmt: skip
    shown = json.loads(capsys.readouterr().out)
    versions = tmp_path / "versions-holding.json"  # the record the script helper passes
    assert json.loads(out.read_text())["versions_record"] == {
        "path": str(versions),
        "sha256": hashlib.sha256(versions.read_bytes()).hexdigest(),
    }
    assert shown["next"]["campaign"] == "confirmation"
    confirmation = shown["confirmation"]
    assert (confirmation["first_block_without_runs"], confirmation["supported_in_order"]) == (3, 2)
    looked = shown["looks"]
    assert looked["available"] and looked["fixation"] == FIXATION
    assert looked["interval"] == {"method": "bootstrap_t", "resamples": 2000, "seed": 20261016}
    assert (looked["looks"]["waiting"]["started"], looked["looks"]["waiting"]["needs"]) == (2, 36)
    assert looked["missing_labels"] == {ADAMW: 0, FA: 0}  # the primary arms only


def _sealed_everywhere(node) -> None:
    """No pre-warmed interpolated harm or missing-label count anywhere in a record: no ``h``,
    ``h_missing_harmful``, ``missing`` or ``missing_labels`` holding the pre-warmed arm, and no
    number equal to its synthetic harm (77/500)."""
    if isinstance(node, dict):
        for key in ("h", "h_missing_harmful", "missing", "missing_labels"):
            if isinstance(node.get(key), dict):
                assert PREWARMED not in node[key], (key, node[key])
        for value in node.values():
            _sealed_everywhere(value)
    elif isinstance(node, list):
        for value in node:
            _sealed_everywhere(value)
    elif isinstance(node, float):
        assert node != pytest.approx(77 / 500, abs=1e-12)


def test_the_summary_seals_the_prewarmed_harm_until_the_final_analysis(
    monkeypatch, tmp_path, capsys
) -> None:
    """The coordinator's ruling on Task 3's concern 3: the looks wait for the pre-warmed labels
    (``look_blocks`` is untouched), but before the final analysis neither the printed status nor
    OUT.json carries the pre-warmed arm's h or missing labels; the primary arms' are there."""
    runs = base()
    confirmed(runs)
    runs = on_disk(runs, tmp_path)
    by_name = {run.run_dir.name: run for run in runs}
    monkeypatch.setattr(
        "zero_field_refusal.analysis.evidence.label_path",
        lambda run_dir, judge: run_dir / "config.yaml",
    )  # every run is labeled
    monkeypatch.setattr("zero_field_refusal.data.load_prompt_set",
                        lambda name, data: [{"id": "p", "text": "t"}])  # fmt: skip

    def labels(run_dir, judge, identity, prompts):
        arm = by_name[run_dir.name].arm
        return rows({ADAMW: 100, FA: 50, PREWARMED: 77}[arm], missing=3 if arm == PREWARMED else 1)

    monkeypatch.setattr("zero_field_refusal.analysis.study2_labels.read_endpoint_labels", labels)
    out = script(monkeypatch, tmp_path, SUMMARY, runs, "--fixation", fixation_file(tmp_path),
                 "--runs-root", runs_root(tmp_path), "--judge", "wildguard")  # fmt: skip
    printed, whole = json.loads(capsys.readouterr().out), json.loads(out.read_text())
    assert printed["looks"]["available"] and not printed["final"]["available"]
    blocks = whole["looks"]["blocks"]
    assert [b["complete"] for b in blocks[:3]] == [True, True, False]  # waited for all three arms
    assert blocks[0]["h"] == pytest.approx({ADAMW: 0.2, FA: 0.1})
    assert blocks[0]["missing"] == {ADAMW: 2, FA: 2}
    assert printed["looks"]["missing_labels"] == {ADAMW: 4, FA: 4}
    for record in (printed, whole):
        _sealed_everywhere(record)


def test_the_summary_is_the_same_with_and_without_the_time_course(
    monkeypatch, tmp_path, capsys
) -> None:
    """Review of Task 5, C1: the summary with its looks (every look, the utility family, the
    sensitivities, the blocks and the guard's counts) is byte for byte the same with and without
    the time course's runs of blocks 1-3, labeled far from their endpoints (one of them not
    identical); no label of a time-course run is read. Only ``progress.timecourse`` and the
    command line (its --out) differ."""
    runs = base()
    rates = until(runs, "confirmation")["next"]["rates"]
    runs += blocks_of(rates, 20, CONFIRMATION_SEEDS[:3])
    timecourse = [*courses(rates, 20, TIMECOURSE_SEEDS[:3:2]),
                  *(course(arm, 20, sorted(lrs)[1], 3001, scale=0.9)
                    for arm, lrs in rates.items())]  # fmt: skip
    every = on_disk([*runs, *timecourse], tmp_path)
    by_name = {run.run_dir.name: run for run in every}
    read = []
    monkeypatch.setattr(
        "zero_field_refusal.analysis.evidence.label_path",
        lambda run_dir, judge: run_dir / "config.yaml",
    )  # every run is labeled
    monkeypatch.setattr("zero_field_refusal.data.load_prompt_set",
                        lambda name, data: [{"id": "p", "text": "t"}])  # fmt: skip

    def labels(run_dir, judge, identity, prompts):
        read.append(run_dir.name)
        run = by_name[run_dir.name]
        harmful = 400 if run.run_id.startswith(COURSE) else 100 if run.arm == ADAMW else 50
        return rows(harmful)

    monkeypatch.setattr("zero_field_refusal.analysis.study2_labels.read_endpoint_labels", labels)
    record = fixed(tmp_path)
    outputs = {}
    for name, loaded_runs in (("without", every[: len(runs)]), ("with", every)):
        out = script(monkeypatch, tmp_path, SUMMARY, loaded_runs, "--fixation", record,
                     "--runs-root", runs_root(tmp_path), "--judge", "wildguard")  # fmt: skip
        whole = json.loads(out.read_text())
        outputs[name] = (capsys.readouterr().out, whole)
    (shown, without), (shown_with, with_course) = outputs["without"], outputs["with"]
    # The printed status and looks, byte for byte, but for the time course's counts and the
    # update scalars' coverage, which report the time course's runs (amendment 4, Task 3).
    printed = [json.loads(text) for text in (shown, shown_with)]
    assert printed[0].pop("timecourse")["counts"]["missing"] == 30
    assert printed[1].pop("timecourse")["counts"]["missing"] == 21
    coverage = [found.pop("scalars") for found in printed]
    assert coverage[1]["by_role"]["timecourse"]["runs"] == 9
    assert "timecourse" not in coverage[0]["by_role"]
    assert json.dumps(printed[0]) == json.dumps(printed[1])
    looked = json.loads(shown)["looks"]
    assert looked["available"] and looked["utility"]["family"]
    assert {member["runs"] for member in looked["utility"]["family"]} == {3}  # blocks 1-3 only
    assert not any(name.startswith(COURSE) for name in read)
    course_state = with_course["progress"].pop("timecourse")
    assert [t["run"] is not None for t in course_state["trajectories"][:9]] == [True] * 9
    assert without["progress"].pop("timecourse")["counts"]["missing"] == 30
    for whole in (without, with_course):
        whole.pop("command")  # its --out
        whole.pop("scalars")  # the coverage of every regular run, the time course's included
    assert json.dumps(without, sort_keys=True) == json.dumps(with_course, sort_keys=True)


def summary_module():
    spec = importlib.util.spec_from_file_location("summary", SUMMARY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_summary_reads_the_labels_of_the_brackets_once(monkeypatch, tmp_path) -> None:
    runs = base()
    state = confirmed(runs)
    module = summary_module()
    runs = on_disk(runs, tmp_path)
    brackets = [s["bracket"][side]["run"] for s in state["confirmation"][0]["supports"].values()
                for side in ("lo", "hi")]  # fmt: skip
    labeled = set(brackets[:4])  # AdamW-25's and FA-25's brackets of block 1
    loaded, read = [], []
    monkeypatch.setattr(module, "label_path", lambda run_dir, judge: run_dir / f"{judge}.jsonl")
    for run_id in labeled:
        (tmp_path / "runs" / run_id / "wildguard.jsonl").write_text("")

    def prompt_set(name, data):
        loaded.append(name)
        return [{"id": "p", "text": "t"}]

    def endpoint_labels(run_dir, judge, identity, prompts):
        read.append(run_dir.name)
        return rows(1)

    monkeypatch.setattr(module, "load_prompt_set", prompt_set)
    monkeypatch.setattr(module, "read_endpoint_labels", endpoint_labels)
    labels, pending = module.read_labels(state, runs, "wildguard")
    assert set(labels) == labeled and sorted(read) == sorted(labeled) and pending == {}
    assert loaded == ["harmful_evaluation"]  # the prompt set is read once
    found = look_blocks(state["confirmation"], labels, state["frozen"]["K_star"])
    assert not found[0]["complete"] and found[0]["unlabeled"]  # the pre-warmed bracket waits


def test_the_summary_keeps_runs_without_a_completed_receipt_pending(monkeypatch, tmp_path):
    from zero_field_refusal.analysis.study2_labels import LabelsPending

    runs = base()
    state = confirmed(runs)
    module = summary_module()
    runs = on_disk(runs, tmp_path)
    brackets = [s["bracket"][side]["run"] for s in state["confirmation"][0]["supports"].values()
                for side in ("lo", "hi")]  # fmt: skip
    monkeypatch.setattr(module, "label_path", lambda run_dir, judge: run_dir / f"{judge}.jsonl")
    for run_id in brackets:
        (tmp_path / "runs" / run_id / "wildguard.jsonl").write_text("")
    interrupted = brackets[0]

    def endpoint_labels(run_dir, judge, identity, prompts):
        if run_dir.name == interrupted:
            raise LabelsPending("no judging receipt")
        return rows(1)

    monkeypatch.setattr(module, "load_prompt_set", lambda name, data: [{"id": "p", "text": "t"}])
    monkeypatch.setattr(module, "read_endpoint_labels", endpoint_labels)
    labels, pending = module.read_labels(state, runs, "wildguard")
    assert pending == {interrupted: "no judging receipt"} and interrupted not in labels
    found = module.compute_looks(state, runs, labels, FIXATION, THETA0, pending)
    assert found["labels_pending"] == {interrupted: "no judging receipt"}
    assert found["blocks"][0]["unlabeled"] == [interrupted] and 1 in found["waiting_for_labels"]
    assert not found["blocks"][0]["complete"]  # pending, never "not harmful"

    def broken(run_dir, judge, identity, prompts):
        raise ValueError("a label of another response")

    monkeypatch.setattr(module, "read_endpoint_labels", broken)
    with pytest.raises(ValueError, match="another response"):  # an error is not pending
        module.read_labels(state, runs, "wildguard")


def test_the_summary_looks_on_labels() -> None:
    runs = base()
    state = confirmed(runs)
    module = summary_module()
    found = module.compute_looks(state, runs, {}, FIXATION, THETA0)
    assert found["available"] and found["looks"]["waiting"]["started"] == 0
    assert found["waiting_for_labels"] == [1, 2]
    labels = {r.run_id: rows(100 if r.arm == ADAMW else 50) for r in runs}
    found = module.compute_looks(state, runs, labels, FIXATION, THETA0)
    waiting = found["looks"]["waiting"]
    assert (waiting["started"], waiting["supported"], waiting["needs"]) == (2, 2, 36)
    assert found["missing_labels"] == {ADAMW: 0, FA: 0}  # the primary arms only
    assert found["utility"]["family"] and found["continuation"] is None
    assert "secondary" not in found  # amendment 3's secondary contrasts, retired
    early = until(base(), "validation")
    assert not module.compute_looks(early, runs, labels, FIXATION, THETA0)["available"]


def test_the_summary_computes_the_looks_with_the_records_interval(monkeypatch) -> None:
    module = summary_module()
    deltas = [0.02 * (-1) ** i + 0.004 * (i % 5) for i in range(36)]
    entries = [{"block": i + 1, "complete": True, "supported": True, "delta": d,
                "delta_missing_low": d - 0.02, "delta_missing_high": d + 0.03,
                "delta_all_missing_harmful": d + 0.01, "h": {ADAMW: 0.2 + d, FA: 0.2}}
               for i, d in enumerate(deltas)]  # fmt: skip
    blocks = [{"seed": 3000 + i, "supports": {ADAMW: {}, FA: {}}} for i in range(80)]
    state = {"next": {"campaign": "confirmation"}, "frozen": {"K_star": 6.3},
             "confirmation": blocks}  # fmt: skip
    monkeypatch.setattr(module, "look_blocks", lambda blocks, labels, kstar: entries)
    monkeypatch.setattr(module, "utility_family", lambda *args, **kw: {"family": []})
    found = module.compute_looks(state, [], {}, FIXATION, THETA0)
    seed = derive_seed(20261016, "look", 1)

    def interval(shift):
        return bootstrap_t_interval([d + shift for d in deltas], 0.992, 2000,
                                    np.random.default_rng(seed))  # fmt: skip

    first = found["looks"]["looks"][0]
    low, high, dropped = interval(0.0)
    assert first["method"] == "bootstrap_t" and first["interval"] == [low, high]
    assert first["bootstrap"] == {"resamples": 2000, "seed": 20261016, "stream_seed": seed,
                                  "dropped": dropped}  # fmt: skip
    # The sensitivities at the primary's look 1, on its stream: one look each, no stops.
    assert found["looks"]["stopped"] and len(found["looks"]["looks"]) == 1
    (worst,) = found["sensitivities"]["missing_labels"]
    assert worst["interval"] == [interval(-0.02)[0], interval(0.03)[1]]
    (both,) = found["sensitivities"]["all_missing_harmful"]
    assert both["interval"] == list(interval(0.01)[:2])
    assert (worst["look"], worst["started"]) == (1, first["started"]) and "stops" not in worst
    assert "looks_missing_harmful" not in found
    assert found["interval"] == {"method": "bootstrap_t", "resamples": 2000, "seed": 20261016}
    t_record = {**FIXATION, "interval": "t"}
    found = module.compute_looks(state, [], {}, t_record, THETA0)
    assert found["interval"] == {"method": "t", "resamples": None, "seed": None}
    assert found["looks"]["looks"][0]["bootstrap"] is None and found["looks"]["looks"][0]["t"]


def test_the_summary_reports_a_continuation_and_no_secondary_contrasts(monkeypatch) -> None:
    """Amendment 4 retires the secondary contrasts: the summary never calls them. The blocks
    after the stopping look are reported as a descriptive continuation."""
    module = summary_module()
    deltas = [0.02 * (-1) ** i + 0.004 * (i % 5) for i in range(50)]
    entries = [{"block": i + 1, "complete": i < 45, "supported": i % 10 != 9, "delta": d,
                "delta_missing_low": d, "delta_missing_high": d, "delta_all_missing_harmful": d,
                "h": {ADAMW: 0.2 + d, FA: 0.2, PREWARMED: 0.21 + 0.003 * (i % 4)}}
               for i, d in enumerate(deltas)] + [
        {"block": i + 1, "complete": False, "supported": False, "delta": None, "h": {}}
        for i in range(50, 80)]  # fmt: skip
    blocks = [{"seed": 3000 + i, "supports": {ADAMW: {}, FA: {}}} for i in range(80)]
    state = {"next": {"campaign": "confirmation"}, "frozen": {"K_star": 6.3},
             "confirmation": blocks}  # fmt: skip
    monkeypatch.setattr(module, "look_blocks", lambda blocks, labels, kstar: entries)
    monkeypatch.setattr(module, "utility_family", lambda *args, **kw: {"family": []})

    def retired(*args, **kwargs):
        raise AssertionError("the secondary contrasts are retired")

    monkeypatch.setattr(study2_short, "secondary_contrasts", retired)
    found = module.compute_looks(state, [], {}, FIXATION, THETA0)
    assert "secondary" not in found and not hasattr(module, "secondary_contrasts")
    first = found["looks"]["looks"][0]
    assert found["looks"]["stopped"] and first["started"] == 39  # 36 supported in 39 blocks
    later = found["continuation"]
    assert later["stopping_look"] == 1 and later["read_blocks"] == 39
    assert later["later_blocks"] == [40, 41, 42, 43, 44, 45]  # complete in order after them
    after = [d for i, d in enumerate(deltas[39:45], 39) if i % 10 != 9]
    assert later["later"]["n"] == len(after) == 5
    assert later["later"]["mean"] == pytest.approx(sum(after) / 5)
    assert later["every_supported"]["n"] == 41
    # Without a stop there is no continuation.
    early = [{**e, "complete": e["block"] <= 20} for e in entries]
    monkeypatch.setattr(module, "look_blocks", lambda blocks, labels, kstar: early)
    assert module.compute_looks(state, [], {}, FIXATION, THETA0)["continuation"] is None


def test_the_summary_gives_each_look_its_pointwise_reading(monkeypatch) -> None:
    module = summary_module()
    deltas = [0.008 + 0.002 * (-1) ** i for i in range(36)]  # a direction within b
    entries = [{"block": i + 1, "complete": True, "supported": True, "delta": d,
                "delta_missing_low": d, "delta_missing_high": d, "delta_all_missing_harmful": d,
                "h": {ADAMW: 0.2 + d, FA: 0.2}}
               for i, d in enumerate(deltas)]  # fmt: skip
    blocks = [{"seed": 3000 + i, "supports": {ADAMW: {}, FA: {}}} for i in range(80)]
    state = {"next": {"campaign": "confirmation"}, "frozen": {"K_star": 6.3},
             "confirmation": blocks}  # fmt: skip
    monkeypatch.setattr(module, "look_blocks", lambda blocks, labels, kstar: entries)
    monkeypatch.setattr(module, "utility_family", lambda *args, **kw: {"family": []})
    (first,) = module.compute_looks(state, [], {}, FIXATION, THETA0)["looks"]["looks"]
    b = FIXATION["bias_allowance"]
    low, high = first["interval"]
    assert first["readings"] == ["equivalence", "direction", "within_8"]
    assert first["bias_allowance"] == b
    assert first["pointwise"] == {"interval": [low - b, high + b], "readings": ["equivalence"]}


def test_the_versions_record(monkeypatch, tmp_path, capsys) -> None:
    from zero_field_refusal.judge import judge_identity

    module = summary_module()
    fixation = Path(fixation_file(tmp_path))
    out = tmp_path / "docs" / "versions.json"
    argv = [str(SUMMARY), "versions", "--judge", "wildguard", "--fixation", str(fixation),
            "--out", str(out)]  # fmt: skip
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(module, "git_state", lambda: {"commit": COMMIT, "dirty": False})
    module.main()
    record = json.loads(out.read_text())
    model = load_config("experiments/study2_short", ["study2_short.role=confirmation"]).model
    assert record["code"] == {"commit": COMMIT, "allow_descendants": False}
    assert record["model"] == {"name": model.name, "revision": model.revision}
    assert record["model"]["revision"] == "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    identity = judge_identity(load_config("judge/wildguard"))
    assert record["judge"] == {"config": "judge/wildguard", **identity}
    digest = hashlib.sha256(fixation.read_bytes()).hexdigest()
    assert record["fixation"] == {"path": str(fixation), "sha256": digest}
    with pytest.raises(SystemExit):  # written once
        module.main()
    assert "the versions record is written once" in capsys.readouterr().err
    out.unlink()
    for state in ({"commit": COMMIT, "dirty": True}, {"commit": None, "dirty": None}):
        monkeypatch.setattr(module, "git_state", lambda state=state: state)
        with pytest.raises(SystemExit):
            module.main()
        assert "clean checkout of a known commit" in capsys.readouterr().err
        assert not out.exists()
    monkeypatch.setattr(module, "git_state", lambda: {"commit": COMMIT, "dirty": False})
    monkeypatch.setattr("sys.argv", [*argv[:5], fixation_file(tmp_path, cap=70), *argv[6:]])
    with pytest.raises(SystemExit):  # the fixation record is checked whole
        module.main()
    assert "cap 70 is not the cap 80" in capsys.readouterr().err


def git(repo: Path, *args: str) -> str:
    command = ["git", "-c", "user.name=test", "-c", "user.email=test@example.org",
               "-c", "commit.gpgsign=false", *args]  # fmt: skip
    return subprocess.run(command, cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()  # fmt: skip


def test_the_versions_record_is_enforced_against_the_checkout(tmp_path) -> None:
    from zero_field_refusal.analysis.study2_short import VERSIONS_RECORD, versions_problems
    from zero_field_refusal.judge import judge_identity

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "code.py").write_text("x = 1\n")
    (repo / ".gitignore").write_text(".superpowers/\n")
    git(repo, "add", "code.py", ".gitignore")
    git(repo, "commit", "-qm", "code")
    recorded = git(repo, "rev-parse", "HEAD")
    fixation = Path(fixation_file(tmp_path))
    model = load_config("experiments/study2_short", ["study2_short.role=confirmation"]).model
    digest = hashlib.sha256(fixation.read_bytes()).hexdigest()
    record = {"code": {"commit": recorded, "allow_descendants": False},
              "model": {"name": model.name, "revision": model.revision},
              "judge": {"config": "judge/wildguard",
                        **judge_identity(load_config("judge/wildguard"))},
              "fixation": {"path": str(fixation), "sha256": digest}}  # fmt: skip
    path = repo / VERSIONS_RECORD

    def problems(change=None, **kwargs):
        if change is not None:
            path.write_text(json.dumps({**record, **change}))
        return versions_problems(path, fixation=fixation, repo=repo, **kwargs)

    assert problems() == [
        f"{path} is missing: write the versions record "
        "(scripts/study2_short_summary.py versions) and commit it first"
    ]
    path.parent.mkdir()
    # Written but not committed: refused.
    assert problems({}) == [f"{path} is not committed: commit the versions record first"]
    # Outside the checkout: refused, whatever it holds.
    outside = tmp_path / "versions-outside.json"
    outside.write_text(path.read_text())
    (found,) = [p for p in versions_problems(outside, fixation=fixation, repo=repo)
                if "not in the checkout" in p]  # fmt: skip
    git(repo, "add", VERSIONS_RECORD)
    git(repo, "commit", "-qm", "the versions record")
    assert problems() == []  # committed: a child that adds the record alone
    assert problems(judge="wildguard") == []
    # Any untracked file is refused, outside src/, scripts/ and configs/ too; an ignored one is not.
    (repo / "docs" / "x.md").write_text("local notes\n")
    assert problems() == ["the checkout has 1 untracked files (first: docs/x.md)"]
    (repo / "docs" / "x.md").unlink()
    (repo / ".superpowers").mkdir()
    (repo / ".superpowers" / "notes.md").write_text("an ignored local workflow file\n")
    assert problems() == []
    # Only the checkout's .gitignore ignores: not its info/exclude.
    (repo / ".git" / "info" / "exclude").write_text("local/\n")
    (repo / "local").mkdir()
    (repo / "local" / "notes.md").write_text("a local file\n")
    assert problems() == ["the checkout has 1 untracked files (first: local/notes.md)"]
    shutil.rmtree(repo / "local")
    (repo / "code.py").write_text("x = 2\n")
    assert problems() == ["the checkout has 1 uncommitted changes (first: code.py)"]
    git(repo, "commit", "-qam", "a code change")
    (found,) = problems()
    assert f"differs from the recorded commit {recorded} in 1 files (first: code.py)" in found
    assert "does not allow descendants" in found
    # A record changed since its commit is refused; committed, allowing descendants admits HEAD.
    allowing = {"code": {"commit": recorded, "allow_descendants": True}}
    assert f"{path} differs from its committed version at HEAD" in problems(allowing)
    git(repo, "commit", "-qam", "allow descendants")
    assert problems() == []
    (found, *_) = problems({"code": {"commit": "0" * 40}})
    assert "is neither the recorded commit" in found
    for change, message in (
        ({"model": {"name": model.name, "revision": "main"}}, "the record's model"),
        ({"judge": {**record["judge"], "judge_revision": "other"}},
         "judge identity differs from judge/wildguard's in ['judge_revision']"),
        ({"judge": {**record["judge"], "config": "judge/missing"}}, "does not load"),
        ({"fixation": {"sha256": "0" * 64}}, "fixation sha256 is not that of"),
    ):  # fmt: skip
        found = problems(change)
        assert any(message in problem for problem in found), (message, found)
    assert any("not judge/rubric" in problem for problem in problems({}, judge="rubric"))


def test_the_versions_record_and_the_runs_read_a_dirty_tree_alike(tmp_path) -> None:
    """``versions_problems`` and the runs' ``git_state`` (``tracking.worktree_changes``) on the
    same trees, with ``status.showUntrackedFiles`` set to ``no``."""
    from zero_field_refusal.analysis.study2_short import VERSIONS_RECORD, versions_problems
    from zero_field_refusal.judge import judge_identity
    from zero_field_refusal.tracking import git_state

    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / ".gitignore").write_text("/runs/\n/scratchpad/\n.superpowers/\n")
    (repo / "code.py").write_text("x = 1\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "code")
    fixation = Path(fixation_file(tmp_path))
    model = load_config("experiments/study2_short", ["study2_short.role=confirmation"]).model
    record = {"code": {"commit": git(repo, "rev-parse", "HEAD"), "allow_descendants": False},
              "model": {"name": model.name, "revision": model.revision},
              "judge": {"config": "judge/wildguard",
                        **judge_identity(load_config("judge/wildguard"))},
              "fixation": {"sha256": hashlib.sha256(fixation.read_bytes()).hexdigest()},
              }  # fmt: skip
    path = repo / VERSIONS_RECORD
    path.parent.mkdir()
    path.write_text(json.dumps(record))
    git(repo, "add", VERSIONS_RECORD)
    git(repo, "commit", "-qm", "the versions record")

    def both() -> tuple[bool, bool]:
        problems = versions_problems(path, fixation=fixation, repo=repo)
        return bool(problems), git_state(repo)["dirty"]

    assert both() == (False, False)
    for ignored in ("runs/x/meta.json", "scratchpad/notes.md", ".superpowers/sdd/brief.md"):
        (repo / ignored).parent.mkdir(parents=True, exist_ok=True)
        (repo / ignored).write_text("ignored\n")
    assert both() == (False, False)
    (repo / "notes.md").write_text("not ignored\n")
    assert git(repo, "status", "--porcelain") == ""  # git status sees nothing
    assert both() == (True, True)
    (repo / "notes.md").unlink()
    (repo / "code.py").write_text("x = 2\n")
    assert both() == (True, True)
    (repo / "code.py").write_text("x = 1\n")
    assert both() == (False, False)


def test_the_summary_looks_need_the_versions_record(monkeypatch, tmp_path, capsys) -> None:
    missing = str(tmp_path / "no-versions.json")
    with pytest.raises(SystemExit):
        script(monkeypatch, tmp_path, SUMMARY, base(), "--fixation", fixation_file(tmp_path),
               "--judge", "wildguard", "--versions", missing, versions=False)  # fmt: skip
    assert f"the versions record does not hold: {missing} is missing" in capsys.readouterr().err


# ------------------------------------------------------------------ the update scalars' coverage


def scalar_rows(arm: str, steps, *, departure=(5, 10, 25)) -> list[dict]:
    """update_scalars.jsonl rows as schedule.UpdateScalars writes them (synthetic values)."""
    return [{"step": t, "update_norm": 0.01 * t, "cos_update_grad": 0.5,
             **({"d_departure": {"norm_ratio": 0.1, "median": 0.01, "p90": 0.2}}
                if arm == PREWARMED and t in departure else {})}
            for t in steps]  # fmt: skip


def with_scalars(run, tmp_path: Path, rows: list | None, *, enabled: bool = True, steps=25):
    """``run`` with a directory holding its config's ``tracking.update_scalars`` and, unless
    ``rows`` is None, its update_scalars.jsonl; ``steps`` taken."""
    run_dir = tmp_path / "runs" / run.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(f"tracking: {{update_scalars: {str(enabled).lower()}}}\n")
    if rows is not None:
        (run_dir / "update_scalars.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return dataclasses.replace(run, run_dir=run_dir, terminal={**run.terminal, "steps": steps})


def test_scalars_coverage_report(monkeypatch, tmp_path, capsys) -> None:
    """Per regular run of every role: update_scalars.jsonl with one row per step 1-25 (to the
    steps taken), and for the pre-warmed arm the departure at steps 5, 10 and 25; a gap is
    reported, never gating; Stage A's runs and the FA controls are not covered."""
    from zero_field_refusal.schedule import DEPARTURE_STEPS

    assert study2_short.DEPARTURE_STEPS == DEPARTURE_STEPS
    full = range(1, 26)
    gap = [r for r in scalar_rows(FA, full) if r["step"] != 13] + scalar_rows(FA, [7])
    runs = [
        with_scalars(later(ADAMW, 20, 1.48e-5, 3000), tmp_path, scalar_rows(ADAMW, full)),
        with_scalars(later(PREWARMED, 20, 1.48e-5, 3000), tmp_path, scalar_rows(PREWARMED, full)),
        with_scalars(
            later(PREWARMED, 20, 1.48e-5, 3001),
            tmp_path,
            scalar_rows(PREWARMED, full, departure=(5, 25)),
        ),
        with_scalars(later(FA, 20, 1.86e-5, 3000), tmp_path, gap),
        with_scalars(later(ADAMW, 20, 1.48e-5, 2100), tmp_path, None, enabled=False),
        with_scalars(
            later(PREWARMED, 20, 1.48e-5, 2200),
            tmp_path,
            scalar_rows(PREWARMED, range(1, 8)),
            steps=7,
        ),  # stopped at step 7
        with_scalars(course(ADAMW, 20, 1.48e-5, 3001), tmp_path, scalar_rows(ADAMW, [1])),
    ]
    unread = with_scalars(later(FA, 20, 1.59e-5, 3000), tmp_path, None)
    (unread.run_dir / "update_scalars.jsonl").write_text("{not json\n")
    outside = [conform(r) for r in stage_a_reference()[:2]]  # Stage A's: not covered
    outside.append(conform(control(20, 1.86e-5, 0.06, 0.0)))  # an FA control: not covered
    report = study2_short.scalars_coverage(
        [*runs, unread, *outside], facts_of(runs + [unread] + outside)
    )
    by = {e["run"]: e for e in report["runs"]}
    assert set(by) == {r.run_id for r in [*runs, unread]}
    assert not by[runs[0].run_id]["gap"] and not by[runs[1].run_id]["gap"]
    assert by[runs[1].run_id]["departure_missing"] == []
    assert "departure_missing" not in by[runs[0].run_id]
    assert by[runs[2].run_id]["departure_missing"] == [10] and by[runs[2].run_id]["gap"]
    holed = by[runs[3].run_id]
    assert (holed["missing_steps"], holed["repeated_steps"]) == ([13], [7])
    absent = by[runs[4].run_id]
    assert (absent["file"], absent["enabled"], absent["missing_steps"], absent["gap"]) == (
        False, False, list(full), True)  # fmt: skip
    stopped = by[runs[5].run_id]
    assert (stopped["steps_taken"], stopped["gap"], stopped["departure_missing"]) == (7, False, [])
    assert by[runs[6].run_id]["missing_steps"] == list(range(2, 26))
    assert by[runs[6].run_id]["role"] == TIMECOURSE
    assert by[unread.run_id]["unreadable"].startswith("does not read") and by[unread.run_id]["gap"]
    assert report["gaps"] == 5 and report["by_role"]["confirmation"] == {"runs": 5, "gaps": 3}
    assert report["by_role"]["timecourse"] == {"runs": 1, "gaps": 1}
    # The summary reports the gaps and gates nothing: the next step is the selection's.
    stage_runs = base()
    until(stage_runs, "numerical")
    script(monkeypatch, tmp_path, SUMMARY, stage_runs, "--fixation", fixed(tmp_path))
    shown = json.loads(capsys.readouterr().out)
    assert shown["next"]["campaign"] == "numerical" and shown["problems"] == []
    discovery = [r for r in stage_runs if r.role is None]
    assert shown["scalars"]["by_role"] == {"discovery": {"runs": len(discovery),
                                                         "gaps": len(discovery)}}  # fmt: skip
    assert len(shown["scalars"]["with_gaps"]) == len(discovery) > 0
    assert shown["scalars"]["status"] == "reported; a gap gates nothing"
