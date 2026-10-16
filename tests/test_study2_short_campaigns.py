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
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from test_study2_short import SCALE, THETA0, W_MAX, control, made, stage_a_reference, traj

from zero_field_refusal.analysis.stage_a import arm_of
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    CONFIRMATION_SEEDS,
    FA,
    FA_CONTROL_ARM,
    FIXATION_KEYS,
    PREWARMED,
    V0_ARMS,
    VALIDATION_SEEDS,
    bootstrap_t_interval,
    confirmation_guard,
    conformance,
    exhausted_runs,
    issued_ranges,
    look_blocks,
    looks,
    measures_rayleigh,
    preconditioner_pairing,
    prescribed,
    progress,
    read_fixation,
    run_facts,
    seed_role,
)
from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.launch import MAX_ATTEMPTS, load_campaign
from zero_field_refusal.protocol import load_run_config
from zero_field_refusal.seeding import derive_seed

SCRIPTS = Path(__file__).parents[1] / "scripts"
CAMPAIGNS, SUMMARY = SCRIPTS / "study2_short_campaigns.py", SCRIPTS / "study2_short_summary.py"
OPTIMIZERS = {ADAMW: [], FA: ["optimizer=adam_frozen"], PREWARMED: ["optimizer=adamw_prewarmed"]}
COMMIT = "0123456789abcdef0123456789abcdef01234567"
FIXATION = {
    "w_max": W_MAX,
    "levels": [0.992, 0.9631],
    "se_floor": 0.0,
    "bias_allowance": 0.01079,
    "interval": "bootstrap_t",
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 20261016,
    "looks": [36, 60],
    "cap": 80,
    "equivalence_min_supported": 36,
}
LOOK_SETTINGS = {k: v for k, v in FIXATION.items() if k not in ("w_max", "levels", "looks", "cap")}


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
    assert a == b
    assert (short.protocol_version, short.train.steps) == (2, 25)
    assert list(short.schedule.evaluate_at) == [25]
    assert (short.schedule.policy, short.schedule.instability.grad_factor) == ("fixed", 10)
    assert short.eval.monitor.version == 2 and "rayleigh" not in short.eval
    assert arm_of(short) == arm


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
    (new / "meta.json").write_text(json.dumps({"inputs": {"streams": streams}}))
    (old / "config.yaml").write_text("stage_a: {role: discovery}\neval: {}\n")
    (old / "meta.json").write_text(json.dumps({"inputs": {"streams": {"training": {}}}}))
    assert run_facts(new) == {"role": "confirmation", "rayleigh": True,
                              "preconditioner": {"path": stream["path"],
                                                 "sha256": stream["sha256"]}}  # fmt: skip
    assert run_facts(old) == {"role": "discovery", "rayleigh": False, "preconditioner": None}


# ------------------------------------------------------------------ the synthetic stage

P = prescribed()


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
    control_run = run.arm == FA_CONTROL_ARM
    return {"role": "full_control" if control_run else seed_role(run.seed),
            "rayleigh": not control_run and measures_rayleigh(run.seed),
            "preconditioner": v0(run.seed, run.batch_size) if run.arm in V0_ARMS
            else None}  # fmt: skip


def facts_of(runs) -> dict:
    return {run.run_id: fact(run) for run in runs}


def later(arm, batch, lr, seed):
    """A validation or confirmation run of the world, its K jittered by seed."""
    k = SCALE[(arm, batch)] * (lr / 1e-5) ** 1.5 * (0.98 + 0.01 * (seed % 5))
    return conform(traj(arm, batch, lr, seed, k, mmlu=0.3 * k, factual=0.1 * k))


def stage(runs, exhausted=None):
    return progress(runs, THETA0, W_MAX, facts=facts_of(runs), exhausted=exhausted)


def make(step) -> list:
    runs = []
    for r in step["runs"]:
        if r["arm"] == FA_CONTROL_ARM:
            runs.append(conform(control(r["batch"], r["lr"], r["contamination_k"],
                                        0.1 * r["contamination_k"])))  # fmt: skip
        elif r["seed"] in (2000, 2001, 2002):
            runs.append(conform(made(r["arm"], r["batch"], r["lr"], r["seed"])))
        else:
            runs.append(later(r["arm"], r["batch"], r["lr"], r["seed"]))
    return runs


def until(runs, campaign: str) -> dict:
    """Make every campaign the stage names until it names ``campaign``."""
    for _ in range(12):
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


def test_the_stage_goes_from_discovery_to_validation_to_confirmation() -> None:
    runs = base()
    assert stage(runs)["next"]["campaign"] == "targeted"
    state = until(runs, "validation")
    step = state["next"]
    assert (step["batch"], step["U"]) == (5, 8.0) and not step["prewarmed_dropped"]
    assert step["rates"] == {ADAMW: [1e-5, 1.17e-5, 1.37e-5], FA: [1.27e-5, 1.49e-5, 1.74e-5],
                             PREWARMED: [1e-5, 1.17e-5, 1.37e-5]}  # fmt: skip
    assert len(step["runs"]) == 3 * 3 * 12  # three arms, three rates, seeds 2100-2111
    runs += make(step)[:-1]  # one validation run still missing: no verdict yet
    assert stage(runs)["next"]["campaign"] == "validation"
    state = until(runs, "confirmation")
    assert state["validation"]["passes"] and len(state["validation"]["jointly_supported"]) == 12
    assert set(state["validation_prewarmed_utility"]) == {1e-5, 1.17e-5, 1.37e-5}
    assert not any(b["runs"] for b in state["confirmation"])
    runs += blocks_of(step["rates"], 5, CONFIRMATION_SEEDS[:3])
    runs += blocks_of({ADAMW: step["rates"][ADAMW]}, 5, CONFIRMATION_SEEDS[3:4])
    blocks = stage(runs)["confirmation"]
    assert [b["complete"] for b in blocks[:5]] == [True, True, True, False, False]
    assert all(b["supported"] for b in blocks[:3]) and blocks[3]["runs"] == 3


def test_a_failed_validation_ends_the_stage_there() -> None:
    runs = base()
    state = until(runs, "validation")
    made_runs = make(state["next"])
    # Half the blocks' AdamW-25 runs read far below K*: fewer than 10 jointly supported.
    low = set(VALIDATION_SEEDS[:6])
    runs += [dataclasses.replace(r, evaluations=[{**r.evaluations[0], "K_benign": 1.0}])
             if r.arm == ADAMW and r.seed in low else r for r in made_runs]  # fmt: skip
    state = stage(runs)
    assert state["next"]["campaign"] == "validation_failed"
    assert "6 of 12 blocks jointly supported" in state["next"]["reasons"][0]


def test_records_off_the_campaigns_leave_the_stage_incomplete() -> None:
    runs = base()
    first = make(stage(runs)["next"])[:1]
    ok = stage([*runs, *first])
    assert ok["next"]["campaign"] == "targeted" and not ok["problems"]  # the rest of the round
    own = fact(first[0])
    cases = {
        "role": lambda f: {**f, first[0].run_id: {**own, "role": "validation"}},
        "rayleigh": lambda f: {**f, first[0].run_id: {**own, "rayleigh": True}},
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
    # A run the state machine did not prescribe stops every campaign until it is resolved.
    stray = conform(made(ADAMW, 5, 1.23e-5, 2000))
    state = stage([*runs, stray])
    assert state["next"] == {"campaign": "incomplete"} and "unexpected run" in state["problems"][0]


def test_v0_is_one_preconditioner_stream_per_seed_and_batch() -> None:
    runs = base()
    until(runs, "prewarmed")  # Stage A's FA grids, the targeted FA runs and the FA controls
    facts = facts_of(runs)
    assert preconditioner_pairing(runs, facts) == []
    # The seed's FA control estimated v₀ from another copy of the stream: no conclusion.
    fa_control = next(r for r in runs if r.arm == FA_CONTROL_ARM)
    other = {
        **facts,
        fa_control.run_id: {
            **facts[fa_control.run_id],
            "preconditioner": {**v0(2000, 5), "sha256": "x"},
        },
    }
    found = preconditioner_pairing(runs, other)
    assert len(found) == 1 and "seed 2000, B=5: v₀ from 2 preconditioner streams" in found[0]
    state = progress(runs, THETA0, W_MAX, facts=other)
    assert state["next"] == {"campaign": "incomplete"} and state["problems"] == found
    # A pre-warmed run that records no preconditioner stream is a problem too.
    prewarmed = later(PREWARMED, 5, 1e-5, 2000)
    missing = {**facts, prewarmed.run_id: {**fact(prewarmed), "preconditioner": None}}
    assert "records no preconditioner stream" in preconditioner_pairing([prewarmed], missing)[0]
    # AdamW-25 estimates no v₀, and B = 20's stream is another one.
    assert preconditioner_pairing([later(ADAMW, 5, 1e-5, 2000)], {}) == []
    pair = [later(FA, 5, 1.27e-5, 2000), later(FA, 20, 1.27e-5, 2000)]
    assert preconditioner_pairing(pair, facts_of(pair)) == []


def test_later_runs_must_be_at_the_frozen_rates() -> None:
    runs = base()
    until(runs, "validation")
    stray = later(FA, 5, 1.3e-5, 2100)
    state = stage([*runs, stray])
    assert state["next"] == {"campaign": "incomplete"}
    assert "outside the frozen rates" in state["problems"][0]


# ------------------------------------------------------------------ runs given up


def journal(
    key_cell: str,
    seed: int,
    overrides: list[str],
    *,
    attempts: int,
    recovered: bool,
    last_outcome: str = "technical",
    config: str = "experiments/study2_short",
) -> list:
    """The launcher's records of one run's attempts (the last one never ends when technical)."""
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
        if attempt < attempts or last_outcome != "technical":
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


def test_a_run_given_up_is_a_technical_failure_of_its_block_and_the_looks_go_past_it() -> None:
    runs = base()
    state = until(runs, "confirmation")
    rates = state["next"]["rates"]
    given_up = (ADAMW, 5, 1.37e-5, 3000)  # the bracket's upper end in block 1
    runs += [r for r in blocks_of(rates, 5, CONFIRMATION_SEEDS[:2])
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
    lost = next(r for r in made_runs if (r.arm, r.lr, r.seed) == (FA, 1.74e-5, 2100))
    runs += [r for r in made_runs if r is not lost]
    assert stage(runs)["next"]["campaign"] == "validation"
    key = (lost.arm, lost.batch_size, lost.lr, lost.seed)
    state = stage(runs, exhausted={key: "lost/s2100"})
    assert state["next"]["campaign"] == "confirmation"  # 11 of 12 blocks: still ≥ 10
    assert state["validation"]["causes"][FA]["failure_kinds"] == {"technical": 1}


# ------------------------------------------------------------------ the confirmation guard


def plan(started: int = 0, complete: int = 0, supported: int = 0) -> list[dict]:
    """80 blocks: the first ``started`` have runs, the first ``complete`` of them are complete,
    the first ``supported`` of those supported."""
    return [{"block": i, "runs": 9 if i <= started else 0, "complete": i <= complete,
             "supported": i <= supported} for i in range(1, 81)]  # fmt: skip


def refused(blocks, first, last, **kw) -> list[str]:
    return confirmation_guard(blocks, first, last, **kw)["problems"]


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
    counts = confirmation_guard(due, 41, 64, after_look=1)
    assert (counts["first_open"], counts["supported_in_order"], counts["possibly_supported"]) == (
        41, 36, 36)  # fmt: skip


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


@pytest.mark.parametrize(
    "key, value",
    [("interval", "percentile"), ("interval", None), ("interval", ["bootstrap_t"]),
     ("bootstrap_resamples", 999), ("bootstrap_resamples", 2000.0),
     ("bootstrap_resamples", "2000"), ("bootstrap_resamples", True), ("bootstrap_resamples", None),
     ("bootstrap_seed", 1.5), ("bootstrap_seed", "7"), ("bootstrap_seed", None),
     ("bootstrap_seed", False), ("looks", [24, 50]), ("looks", [36]), ("looks", [36.0, 60.0]),
     ("looks", "36,60"), ("looks", None), ("cap", 79), ("cap", 80.0), ("cap", None),
     ("equivalence_min_supported", -1), ("equivalence_min_supported", 36.5),
     ("equivalence_min_supported", True), ("equivalence_min_supported", None)],
)  # fmt: skip
def test_the_fixation_record_refuses_a_malformed_new_key(tmp_path, key, value) -> None:
    path = tmp_path / "fixation.json"
    path.write_text(json.dumps({**FIXATION, key: value}))
    with pytest.raises(ValueError, match=f"{key} "):
        read_fixation(path)


# ------------------------------------------------------------------ the looks' blocks


def rows(harmful: int, missing: int = 0) -> list[dict]:
    return [{"condition": "step-25", "step": 25, "prompts": 500, "harmful": harmful,
             "missing": missing}]  # fmt: skip


def confirmed(runs: list) -> dict:
    state = until(runs, "confirmation")
    runs += blocks_of(state["next"]["rates"], 5, CONFIRMATION_SEEDS[:2])
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
    assert first["delta_missing_harmful"] == pytest.approx(0.1)
    assert first["missing"] == {ADAMW: 10, FA: 10, PREWARMED: 10}
    assert set(first["h"]) == {ADAMW, FA, PREWARMED}
    # Without the pre-warmed arm's labels the block still waits: its three arms are labeled
    # together.
    partial = {k: v for k, v in labels.items()
               if k not in {brackets[PREWARMED][s]["run"] for s in ("lo", "hi")}}  # fmt: skip
    assert not look_blocks(blocks, partial, kstar)[0]["complete"]


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


def script(monkeypatch, tmp_path, path: Path, runs: list, *args: str) -> Path:
    """Run a script on ``runs`` (patched in for the run directories); returns its --out."""
    monkeypatch.setattr("zero_field_refusal.analysis.stage_a.load_trajectories",
                        lambda *roots: list(runs))  # fmt: skip
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


def refusal(monkeypatch, tmp_path, capsys, runs, *args: str) -> str:
    with pytest.raises(SystemExit):
        script(monkeypatch, tmp_path, CAMPAIGNS, runs, *args)
    return capsys.readouterr().err


def labels_of(runs) -> list[str]:
    return [run.label for run in runs]


def test_the_targeted_campaign_records_w_max_and_the_revision(monkeypatch, tmp_path) -> None:
    runs = base()
    out = script(monkeypatch, tmp_path, CAMPAIGNS, runs, "targeted", "--w-max", str(W_MAX))
    header = " ".join(line[2:] for line in out.read_text().splitlines() if line.startswith("#"))
    assert re.search(r"targeted at [0-9a-f]{40}", header) and "w_max 0.35 from --w-max" in header
    found = load_campaign(out)[1]
    assert len(found) == 15 and {run.experiment for run in found} == {"study2_short"}
    assert "frozen_adam_25-b5-1.27e-05/s2001" in labels_of(found)
    fa = next(run for run in found if run.cell == "frozen_adam_25-b5-1.27e-05")
    assert fa.overrides[:4] == ("optimizer=adam_frozen", "train.batch_size=5",
                                "optimizer.lr=1.27e-05", "study2_short.role=discovery")  # fmt: skip
    record = fixation_file(tmp_path)
    out = script(monkeypatch, tmp_path, CAMPAIGNS, runs, "targeted", "--fixation", record)
    assert f"the fixation record {record}, sha256" in out.read_text().replace("\n# ", " ")
    split = campaign(monkeypatch, tmp_path, runs, "targeted", "--w-max", str(W_MAX),
                     "--seeds", "2002")  # fmt: skip
    assert len(split) == 5 and {run.seed for run in split} == {2002}


def test_the_campaigns_refuse_another_step_a_missing_w_max_and_incomplete_records(
    monkeypatch, tmp_path, capsys
) -> None:
    runs = base()
    record = fixation_file(tmp_path)
    err = refusal(monkeypatch, tmp_path, capsys, runs, "controls", "--fixation", record)
    assert "the stage's next step is targeted, not controls" in err
    err = refusal(monkeypatch, tmp_path, capsys, runs, "controls", "--w-max", str(W_MAX))
    assert "controls needs the fixation record (--fixation)" in err
    err = refusal(monkeypatch, tmp_path, capsys, runs, "targeted")
    assert "w_max is needed: --w-max or --fixation" in err
    err = refusal(monkeypatch, tmp_path, capsys, runs, "targeted", "--w-max", "0.3",
                  "--fixation", record)  # fmt: skip
    assert "--w-max 0.3 differs from the fixation record's 0.35" in err
    err = refusal(monkeypatch, tmp_path, capsys, runs, "targeted",
                  "--fixation", fixation_file(tmp_path, se_floor=None))  # fmt: skip
    assert "se_floor None is not a number" in err
    err = refusal(monkeypatch, tmp_path, capsys, runs[1:], "targeted", "--w-max", str(W_MAX))
    assert "version-1 run missing" in err  # Stage A's reference incomplete: no stream time
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "1", "10",
                  "--fixation", record)  # fmt: skip
    assert "confirmation needs --runs-root" in err


def test_the_controls_and_prewarmed_campaigns(monkeypatch, tmp_path) -> None:
    runs = base()
    until(runs, "controls")
    record = fixation_file(tmp_path)
    found = campaign(monkeypatch, tmp_path, runs, "controls", "--fixation", record)
    assert labels_of(found) == ["fa-control-b5-1.27e-05-u8/s2000",
                                "fa-control-b5-1.49e-05-u8/s2000",
                                "fa-control-b5-1.74e-05-u8/s2000"]  # fmt: skip
    assert {run.experiment for run in found} == {"study2_short_fa_controls"}
    bound = next(o for o in found[0].overrides if o.startswith("schedule.instability"))
    assert float(bound.split("=")[1]) == 0.01 * math.sqrt(40)  # written in full
    until(runs, "prewarmed")
    found = campaign(monkeypatch, tmp_path, runs, "prewarmed", "--fixation", record)
    assert len(found) == 6 * 3 and all("optimizer=adamw_prewarmed" in r.overrides for r in found)


def test_the_validation_and_confirmation_campaigns(monkeypatch, tmp_path, capsys) -> None:
    runs = base()
    until(runs, "validation")
    record, root = fixation_file(tmp_path), runs_root(tmp_path)
    journal_args = ("--fixation", record, "--runs-root", root)
    found = campaign(monkeypatch, tmp_path, runs, "validation", *journal_args)
    assert len(found) == 108 and {r.seed for r in found} == set(VALIDATION_SEEDS)
    assert all("study2_short.role=validation" in r.overrides for r in found)
    # The launcher grants a recovery round per experiment and cell name: no name recurs.
    assert "adamw_25-b5-1.17e-05-validation" in {r.cell for r in found}
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "1", "2",
                  *journal_args)  # fmt: skip
    assert "next step is validation" in err  # the validation has not passed yet
    until(runs, "confirmation")
    found = campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", "1", "12",
                     *journal_args)  # fmt: skip
    assert len(found) == 12 * 9 and {r.seed for r in found} == set(range(3000, 3012))
    measuring = [r for r in found if r.config == "experiments/study2_short_rayleigh"]
    assert {r.seed for r in measuring} == set(range(3000, 3010))  # blocks 1-10 measure R
    assert {"adamw_25-b5-1e-05-blocks-1-12", "adamw_25-b5-1e-05-blocks-1-12-rayleigh"} <= {
        r.cell for r in found}  # fmt: skip
    cfg = load_run_config(measuring[0].config, measuring[0].overrides)
    assert cfg.eval.rayleigh.enabled and cfg.study2_short.role == "confirmation"
    assert {r.experiment for r in found} == {"study2_short"}
    for blocks, why in ((("2", "3"), "starts at block 1"), (("1", "37"), "room for 36"),
                        (("0", "3"), "1 <= FIRST"), (("5", "4"), "1 <= FIRST")):  # fmt: skip
        err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", *blocks,
                      *journal_args)  # fmt: skip
        assert why in err, blocks
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "1", "2",
                  "--after-look", "1", *journal_args)  # fmt: skip
    assert "--after-look 1: look 1 needs 36" in err


def test_a_range_issued_before_is_completed_and_then_refused(monkeypatch, tmp_path, capsys):
    runs = base()
    rates = until(runs, "confirmation")["next"]["rates"]
    record = fixation_file(tmp_path)
    block_runs = blocks_of(rates, 5, CONFIRMATION_SEEDS[:2])
    runs += block_runs[:-1]  # blocks 1-2 issued and launched, one run still to make
    started = [{"event": "start", "run": f"{r.arm}-b5-{r.lr:g}-blocks-1-2-rayleigh/s{r.seed}",
                "experiment": "study2_short", "config_hash": r.run_id, "run_id": r.run_id,
                "attempt": 1, "command": []} for r in block_runs]  # fmt: skip
    ended = [{"event": "end", "run_id": r.run_id, "outcome": "done"} for r in block_runs]
    args = ("--fixation", record, "--runs-root", runs_root(tmp_path, started + ended))
    found = campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", "1", "2", *args)
    assert labels_of(found) == ["prewarmed_adamw_25-b5-1.37e-05-blocks-1-2-rayleigh/s3001"]
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "2", "3",
                  *args)  # fmt: skip
    assert "have runs already, and 2-3 is not a range issued before" in err
    runs.append(block_runs[-1])
    err = refusal(monkeypatch, tmp_path, capsys, runs, "confirmation", "--blocks", "1", "2",
                  *args)  # fmt: skip
    assert "blocks 1-2 have every run already" in err


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
    assert "preconditioner streams and the protocol-2 path-field record" in header
    assert "can be run again" in header and re.search(r"prep at [0-9a-f]{40}", header)
    commands = joined(text)
    run = 'project-run "$SRC" python scripts/'
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


def driver_functions(text: str) -> str:
    return text[text.index("step() {") : text.index("# The steps")]


def campaigns_module():
    spec = importlib.util.spec_from_file_location("campaigns", CAMPAIGNS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_put_back_resumes_where_it_stopped_and_records_a_difference(pod) -> None:
    module = pod.module
    stream_copy(pod.path, driver="595", day="2026-10-09")
    module.set_aside(pod.path, pod.data, pod.aside, "20261016")
    # A failed attempt regenerated the stream already: setting aside again keeps both in place.
    stream_copy(pod.path, driver="580", day="2026-10-16", samples='{"id": "other"}\n',
                distributions=b"other")  # fmt: skip
    assert "already set aside" in module.set_aside(pod.path, pod.data, pod.aside, "20261017")
    assert (pod.path / "distributions.safetensors").read_bytes() == b"other"
    pod.run_dir.mkdir(parents=True)
    (pod.run_dir / "meta.json").write_text("{}")
    # Interrupted right after the comparison was written: the rest is done, the record kept.
    relative = pod.path.relative_to(pod.data)
    found = module.regeneration_record(pod.aside / "soft-targets-aside-20261016" / relative,
                                       pod.path, pod.run_dir, stream=relative,
                                       kept=Path("k"))  # fmt: skip
    assert found["comparison"]["samples_equal"] is False
    assert found["comparison"]["distributions_equal"] is False
    (pod.run_dir / "soft_targets_regeneration.json").write_text(json.dumps(found))
    done = module.restore(pod.path, pod.data, pod.aside, pod.run_dir)
    assert "compared" not in done and "put back" in done
    assert json.loads((pod.run_dir / "soft_targets_regeneration.json").read_text()) == found
    assert not (pod.path / "distributions.safetensors").exists()
    assert json.loads((pod.path / "manifest.json").read_text())["hardware"]["driver"] == "595"


def test_the_soft_target_steps_refuse_what_they_cannot_resolve(pod) -> None:
    module = pod.module
    with pytest.raises(SystemExit, match="no restored soft-target copy"):
        module.set_aside(pod.path, pod.data, pod.aside, "20261016")
    pod.run_dir.mkdir(parents=True)
    assert "nothing to compare" in module.restore(pod.path, pod.data, pod.aside, pod.run_dir)
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
    assert shown["next"]["campaign"] == "targeted" and len(shown["next"]["runs"]) == 15
    assert [[c["batch"], c["U"]] for c in shown["candidates"]][:2] == [[5, 8.0], [20, 8.0]]
    assert shown["attempts"][0]["arms"][ADAMW][0]["feasible"] is False
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
    script(monkeypatch, tmp_path, SUMMARY, runs, "--fixation", fixation_file(tmp_path),
           "--runs-root", runs_root(tmp_path), "--judge", "wildguard")  # fmt: skip
    shown = json.loads(capsys.readouterr().out)
    assert shown["next"]["campaign"] == "confirmation"
    confirmation = shown["confirmation"]
    assert (confirmation["first_block_without_runs"], confirmation["supported_in_order"]) == (3, 2)
    looked = shown["looks"]
    assert looked["available"] and looked["fixation"] == FIXATION
    assert looked["interval"] == {"method": "bootstrap_t", "resamples": 2000, "seed": 20261016}
    assert (looked["looks"]["waiting"]["started"], looked["looks"]["waiting"]["needs"]) == (2, 36)
    assert looked["missing_labels"] == {ADAMW: 0, FA: 0, PREWARMED: 0}


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
    labels = module.read_labels(state, runs, "wildguard")
    assert set(labels) == labeled and sorted(read) == sorted(labeled)
    assert loaded == ["harmful_evaluation"]  # the prompt set is read once
    found = look_blocks(state["confirmation"], labels, state["frozen"]["K_star"])
    assert not found[0]["complete"] and found[0]["unlabeled"]  # the pre-warmed bracket waits


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
    assert found["missing_labels"] == {ADAMW: 0, FA: 0, PREWARMED: 0}
    assert found["utility"]["family"] and found["secondary"] is None
    early = until(base(), "validation")
    assert not module.compute_looks(early, runs, labels, FIXATION, THETA0)["available"]


def test_the_summary_computes_the_looks_with_the_records_interval(monkeypatch) -> None:
    module = summary_module()
    deltas = [0.02 * (-1) ** i + 0.004 * (i % 5) for i in range(36)]
    entries = [{"block": i + 1, "complete": True, "supported": True, "delta": d,
                "delta_missing_harmful": d + 0.01, "h": {ADAMW: 0.2 + d, FA: 0.2}}
               for i, d in enumerate(deltas)]  # fmt: skip
    blocks = [{"seed": 3000 + i, "supports": {ADAMW: {}, FA: {}}} for i in range(80)]
    state = {"next": {"campaign": "confirmation"}, "frozen": {"K_star": 6.3},
             "confirmation": blocks}  # fmt: skip
    monkeypatch.setattr(module, "look_blocks", lambda blocks, labels, kstar: entries)
    monkeypatch.setattr(module, "utility_family", lambda *args, **kw: {"family": []})
    found = module.compute_looks(state, [], {}, FIXATION, THETA0)
    seed = derive_seed(20261016, "look", 1)
    for key, shift in (("looks", 0.0), ("looks_missing_harmful", 0.01)):
        first = found[key]["looks"][0]
        low, high, dropped = bootstrap_t_interval([d + shift for d in deltas], 0.992, 2000,
                                                  np.random.default_rng(seed))  # fmt: skip
        assert first["method"] == "bootstrap_t" and first["interval"] == [low, high]
        assert first["bootstrap"] == {"resamples": 2000, "seed": 20261016, "stream_seed": seed,
                                      "dropped": dropped}  # fmt: skip
    assert found["interval"] == {"method": "bootstrap_t", "resamples": 2000, "seed": 20261016}
    t_record = {**FIXATION, "interval": "t"}
    found = module.compute_looks(state, [], {}, t_record, THETA0)
    assert found["interval"] == {"method": "t", "resamples": None, "seed": None}
    assert found["looks"]["looks"][0]["bootstrap"] is None and found["looks"]["looks"][0]["t"]
