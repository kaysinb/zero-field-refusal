"""The contract between the numerical-check tool and the selection (review of Task 2, important
2): verdicts built by the tool's own ``fa_control_diagnostic.numerical_verdict`` from its
synthetic records (the world of tests/test_numerical_check.py), written by its
``write_verdict``, read back by ``study2_short.read_numerical`` and through the selection's
``numerical_verdict`` and ``selection``. The verdict's shape is never copied here."""

import json
import runpy
import shutil
from pathlib import Path

import yaml
from omegaconf import OmegaConf
from test_numerical_check import (
    COMMIT,
    OTHER,
    RATES,
    RUN_COMMIT,
    SCRIPT,
    attempt,
    call,
    cases,
    clean_head,  # noqa: F401  (the fixture)
    consistent_code,
    control,
    fake_identity,
    fake_reuse,
    fake_runs,
    no_gpu,
    scope,
    settings,
    stamped,
    world,  # noqa: F401  (the fixture)
    write_records,
)
from test_study2_short import SPC, THETA0, W_MAX, commits, drive, grids, made

from zero_field_refusal import fa_control_diagnostic as tool
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    OPTIMIZERS,
    exhausted_runs,
    numerical_case_name,
    numerical_digest,
    numerical_policy,
    numerical_verdict,
    read_numerical,
    selection,
)
from zero_field_refusal.launch import assess, launch_records, load_campaign, run_campaign
from zero_field_refusal.protocol import load_run_config

CAMPAIGNS = Path(__file__).parents[1] / "scripts" / "study2_short_campaigns.py"

ARM_OF = {name: arm for arm, name in OPTIMIZERS.items()}


def by_arm(rates):
    """A selection by optimizer name (the tool's input) by the selection's arm names."""
    return {ARM_OF[name]: sorted(values) for name, values in rates.items()}


def reuse(base, head):
    """The selection's GPU-code reuse check over the tool tests' synthetic identities, HEAD at
    their ``COMMIT``."""
    return fake_reuse(base, COMMIT if head == "HEAD" else head)


def built(root, rates, *, failing=(), missing=(), change=None):
    """The tool's verdict of a selection (``rates`` by optimizer name) at B = 20 and U = 8 from
    its records under ``root``: every case within and every same-path control accepted, but
    each (optimizer, lr) of ``failing``, whose case on seed 2200 exceeds the bound, and each of
    ``missing``, whose case on 2200 has no record yet (incomplete). ``change``, if given, edits
    the verdict before the tool's ``write_verdict`` writes it where the tool keeps it (moving
    the selection's earlier verdict to its history); returns it."""
    found = cases(rates)
    fake_runs(root, found)  # the regular numerical runs, made at RUN_COMMIT
    out = tool.numerical_root(20, 8, root)
    records = {c.name: {1: attempt(c), 2: None} for c in found}
    for case in found:
        if case.seed == 2200 and (case.optimizer, case.lr) in failing:
            records[case.name][1] = attempt(case, pairs={"p1": 0.081})
        if case.seed == 2200 and (case.optimizer, case.lr) in missing:
            records[case.name][1] = None
    controls = {lr: {1: control(lr), 2: None} for lr in rates["adam_frozen"]}
    write_records(out, records, controls, found)
    verdict = tool.numerical_verdict(settings(), out, rates=rates, batch=20, upper=8,
                                     code=consistent_code(), root=root)  # fmt: skip
    verdict = change(verdict) if change else verdict
    tool.write_verdict(out, verdict, rates)
    return verdict


def test_a_tool_verdict_reads_with_no_stale_reason(world) -> None:  # noqa: F811
    qualified = built(world, RATES)
    read = numerical_verdict(qualified, 20, 8.0, by_arm(RATES), {RUN_COMMIT}, reuse)
    assert (read["state"], read["reasons"], read["unusable"]) == ("complete", [], [])
    # The frozen policy is the tool's, value for value.
    frozen = numerical_policy(8)
    assert {key: qualified["policy"][key] for key in frozen} == frozen
    # The verdict sits where the selection reads it, under the digest the selection computes.
    path = f"numerical/b20-u8/verdicts/{numerical_digest(by_arm(RATES))}.json"
    assert read_numerical(world) == {path: qualified}
    assert numerical_digest(by_arm(RATES)) == tool.selection_digest(RATES)
    shutil.rmtree(tool.numerical_root(20, 8, world))
    failing = built(world, RATES, failing=[("adamw", 1.32e-05)])
    read = numerical_verdict(failing, 20, 8.0, by_arm(RATES), {RUN_COMMIT}, reuse)
    assert (read["state"], read["reasons"]) == ("complete", [])
    assert [(u["arm"], u["lr"], u["state"], u["why"]) for u in read["unusable"]] == [
        (ADAMW, 1.32e-05, "failing", "failing: a pair exceeding the bound")]  # fmt: skip
    assert read["unusable"][0]["reasons"] == ["case s2200: attempt 1: p1 exceeding (max D_twin "
                                              "0.081)"]  # fmt: skip


def test_the_consistency_rule_crosses_from_the_settings_to_the_selection(world) -> None:  # noqa: F811
    """Review fa21: twin A's consistency rule, frozen in the tracked settings, is the tool's
    (``NumericalScope.policy``, the comparator's own), every verdict the tool writes carries it,
    and the selection freezes it; a written verdict without it, or with another rule, departs
    from the policy and decides nothing."""
    rule = {"abs": 1e-08, "rel": 0.0001}
    assert OmegaConf.to_container(settings().numerical.twin_a_consistency) == rule
    assert scope().consistency_rule == tool.TWIN_A_CONSISTENCY == rule
    assert numerical_policy(8)["twin_a_consistency"] == rule
    qualified = built(world, RATES)
    path = f"numerical/b20-u8/verdicts/{numerical_digest(by_arm(RATES))}.json"
    assert read_numerical(world)[path]["policy"]["twin_a_consistency"] == rule
    read = numerical_verdict(qualified, 20, 8.0, by_arm(RATES), {RUN_COMMIT}, reuse)
    assert (read["state"], read["reasons"]) == ("complete", [])

    def without(verdict):
        return {**verdict, "policy": {key: value for key, value in verdict["policy"].items()
                                      if key != "twin_a_consistency"}}  # fmt: skip

    def another(verdict):
        return {
            **verdict,
            "policy": {**verdict["policy"], "twin_a_consistency": {"abs": 1e-08, "rel": 0.001}},
        }

    for change, why in (
        (without, "twin_a_consistency None, frozen {'abs': 1e-08, 'rel': 0.0001}"),
        (another, "twin_a_consistency {'abs': 1e-08, 'rel': 0.001}, frozen"),
    ):
        built(world, RATES, change=change)
        written = read_numerical(world)[path]
        read = numerical_verdict(written, 20, 8.0, by_arm(RATES), {RUN_COMMIT}, reuse)
        assert read["state"] == "stale" and read["unusable"] == []
        assert any(why in reason for reason in read["reasons"]), read["reasons"]


def test_tool_verdicts_drive_the_selection(world) -> None:  # noqa: F811
    """The selection asks for its check; the tool's verdict of it, qualified, lets the selection
    go on to validation. A verdict marking AdamW's K* rate failing re-solves AdamW, and the
    tool's verdict of the re-solved selection, beside the first, qualifies it."""
    runs = grids()
    _, result, _ = drive(runs, until="numerical-check")
    check = result["next"]["check"]
    first = check["selection"]["rates"]  # the tool's rate input, by optimizer name
    assert check["name"] == tool.selection_digest(first)

    def at():
        return selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC,
                         numerical=read_numerical(world), facts=commits(runs, RUN_COMMIT),
                         reuse=reuse)  # fmt: skip

    built(world, first)
    found = at()
    assert found["next"]["campaign"] == "validation" and found["problems"] == []
    (entry,) = found["attempts"][0]["numerical"]
    assert (entry["verdict"], entry["state"], entry["reasons"]) == (check["verdict"], "complete",
                                                                    [])  # fmt: skip
    # The same selection's verdict again, AdamW 1.48e-5 failing (the earlier, qualified one
    # moves to the tool's history, where it marks nothing).
    built(world, first, failing=[("adamw", 1.48e-05)])
    found = at()
    step = found["next"]
    assert step["campaign"] == "numerical" and found["problems"] == []
    entry, resolved = found["attempts"][0]["numerical"]  # the re-solved check's runs come next
    assert (entry["state"], entry["reasons"], resolved["state"]) == ("complete", [],
                                                                     "runs missing")  # fmt: skip
    assert [(u["arm"], u["lr"]) for u in entry["unusable"]] == [(ADAMW, 1.48e-05)]
    assert step["rates"][ADAMW] == [1.27e-05, 1.5e-05, 1.74e-05]  # the re-solve
    runs += [made(r["arm"], r["batch"], r["lr"], r["seed"]) for r in step["runs"]]
    check = at()["next"]["check"]
    second = check["selection"]["rates"]
    built(world, second)
    found = at()
    assert found["next"]["campaign"] == "validation" and found["problems"] == []
    assert [e["state"] for e in found["attempts"][0]["numerical"]] == ["complete", "complete"]
    assert found["next"]["rates"][ADAMW] == [1.27e-05, 1.5e-05, 1.74e-05]


def test_an_earlier_tool_verdict_keeps_its_marks(world) -> None:  # noqa: F811
    """The coordinator's ruling after the integration step: a stale verdict with a failing mark
    and an unfinished entry holds the step at the check, the re-solve not spent. The tool's
    rerun writes a verdict that omits the mark (``write_verdict`` moves the stale one to the
    selection's history): the rate stays unusable, and AdamW is re-solved."""
    runs = grids()
    _, result, _ = drive(runs, until="numerical-check")
    check = result["next"]["check"]
    first = check["selection"]["rates"]

    def at():
        return selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC,
                         numerical=read_numerical(world), facts=commits(runs, RUN_COMMIT),
                         reuse=reuse)  # fmt: skip

    def stale(verdict):
        return {**verdict, "policy": {**verdict["policy"], "norm": 1e-4}}

    marked = built(world, first, failing=[("adamw", 1.48e-05)],
                   missing=[("adam_frozen", 1.86e-05)], change=stale)  # fmt: skip
    assert marked["rates"]["adamw|1.48e-05"] == "failing"
    assert marked["rates"]["adam_frozen|1.86e-05"] == "incomplete"
    found = at()
    step = found["next"]
    assert (step["campaign"], step["state"]) == ("numerical-check", "stale")
    assert len(found["attempts"][0]["arms"][ADAMW]["selections"]) == 2  # no re-solve yet
    rerun = built(world, first)  # the tool's rerun: every case within, the mark omitted
    assert rerun["qualified"] is True
    history = tool.numerical_root(20, 8, world) / "verdicts" / "history" / check["name"]
    (moved,) = history.iterdir()
    found = at()
    entry = found["attempts"][0]["numerical"][0]
    assert entry["state"] == "complete" and entry["history"] == [
        moved.relative_to(world).as_posix()]  # fmt: skip
    assert [(u["arm"], u["lr"], u["state"]) for u in entry["unusable"]] == [
        (ADAMW, 1.48e-05, "failing")]  # fmt: skip
    assert found["next"]["campaign"] == "numerical" and found["problems"] == []
    assert found["next"]["rates"][ADAMW] == [1.27e-05, 1.5e-05, 1.74e-05]


def unlocated(case, n):
    """A case's attempt as ``run_numerical_twins`` records a case whose regular run is not
    found: incomplete, with why, and no regular run named."""
    return stamped({"part": "twins", "case": case.name, "seed": case.seed,
                    "optimizer": case.optimizer, "lr": case.lr, "batch": case.batch,
                    "U": case.upper, "attempt": n, "intended_norm": 5e-05,
                    "status": "incomplete", "pairs": {},
                    "reason": "the regular run is not found: 0 finished runs match"})  # fmt: skip


def test_a_given_up_numerical_run_reaches_the_tool_and_the_selection_re_solves(
    tmp_path,
    world,  # noqa: F811
) -> None:
    """The final review of gate B, I1: a K-only run the launcher gave up on. The selection's
    check request names its case in ``given_up``, as the tool names it; the tool reads the
    record, its plan notes the case instead of stopping, the parts record it incomplete twice
    (as for any run not found), and the verdict holds its rate not_qualified. The selection
    then re-solves AdamW without that rate and calls for the replacement's run."""
    runs = grids()
    _, result, _ = drive(runs, until="numerical")
    step = result["next"]
    gone = (ADAMW, 20, 1.48e-05, 2200)
    runs += [made(r["arm"], r["batch"], r["lr"], r["seed"]) for r in step["runs"]
             if (r["arm"], r["batch"], r["lr"], r["seed"]) != gone]  # fmt: skip
    exhausted = {gone: f"adamw_25-b20-1.48e-05-numerical-u8-{step['check']}/s2200"}

    def at():
        return selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC,
                         numerical=read_numerical(world), facts=commits(runs, RUN_COMMIT),
                         exhausted=exhausted, reuse=reuse)  # fmt: skip

    check = at()["next"]["check"]
    name = numerical_case_name(ADAMW, 1.48e-05, 2200)
    assert check["selection"]["given_up"] == [name] == check["given_up"]
    assert check["w_max"] == W_MAX and check["runs_commits"] == [RUN_COMMIT]
    # The tool reads the request and names the case so too
    record = tmp_path / "selection.json"
    record.write_text(json.dumps(check["selection"]))
    namespace = runpy.run_path(str(SCRIPT))
    rates, given_up = namespace["read_selection"](record, 20, 8.0)
    found = cases(rates)
    assert given_up == [name] and name in {c.name for c in found}
    fake_runs(world, [c for c in found if c.name != name])
    _, problems, notes = namespace["find_runs"](world, scope(), found, given_up)
    assert problems == {} and notes == {name: namespace["GIVEN_UP"]}
    # The parts' records: every other case within, the given-up one incomplete twice
    out = tool.numerical_root(20, 8, world)
    records = {c.name: {1: attempt(c), 2: None} for c in found if c.name != name}
    (case,) = [c for c in found if c.name == name]
    records[name] = {1: unlocated(case, 1), 2: unlocated(case, 2)}
    controls = {lr: {1: control(lr), 2: None} for lr in rates["adam_frozen"]}
    write_records(out, records, controls, found)
    verdict = tool.numerical_verdict(settings(), out, rates=rates, batch=20, upper=8,
                                     code=consistent_code(), root=world)  # fmt: skip
    assert verdict["rates"]["adamw|1.48e-05"] == "not_qualified"
    assert [k for k, v in verdict["rates"].items() if v != "qualified"] == ["adamw|1.48e-05"]
    tool.write_verdict(out, verdict, rates)
    after = at()
    assert after["next"]["campaign"] == "numerical" and after["problems"] == []
    entry = after["attempts"][0]["numerical"][0]
    assert [(u["arm"], u["lr"], u["state"]) for u in entry["unusable"]] == [
        (ADAMW, 1.48e-05, "not_qualified")]  # fmt: skip
    assert after["next"]["rates"][ADAMW] == [1.27e-05, 1.5e-05, 1.74e-05]  # the re-solve


def test_a_void_check_calls_for_fresh_numerical_runs(world) -> None:  # noqa: F811
    """The final review of gate B, I2: a qualified check, then a GPU-path change (HEAD's GPU code
    becomes ``OTHER``'s): the selection waits at the stale check, which the tool cannot run
    on the old runs. ``void`` moves those runs and the check aside; the loader no longer finds
    them, and the selection calls for the numerical runs again."""
    runs = grids()
    _, result, _ = drive(runs, until="numerical-check")
    first = result["next"]["check"]["selection"]["rates"]
    built(world, first)

    def at(head):
        def moved_reuse(base, other):
            return fake_reuse(base, head if other == "HEAD" else other)

        return selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC,
                         numerical=read_numerical(world), facts=commits(runs, RUN_COMMIT),
                         reuse=moved_reuse)  # fmt: skip

    assert at(COMMIT)["next"]["campaign"] == "validation"
    stale = at(OTHER)["next"]
    assert (stale["campaign"], stale["state"]) == ("numerical-check", "stale")
    plan = tool.void_plan(world, scope(), 20, 8.0, {"commit": OTHER, "dirty": False})
    assert plan["problems"] == []  # the world's other runs at B 20 too: all of the old code
    assert {f"study2_short/{c.name}" for c in cases(first)} <= {r["run"] for r in plan["runs"]}
    tool.void_numerical(world, plan, {"commit": OTHER, "dirty": False})
    moved = {path.rsplit("/", 1)[1] for path in plan["paths"]}
    # The loader reads RUNS_DIR without the moved runs
    runs[:] = [r for r in runs if r.batch_size != 20 or numerical_case_name(
        r.arm, r.lr, r.seed) not in moved]  # fmt: skip
    assert read_numerical(world) == {}
    fresh = at(OTHER)
    assert fresh["next"]["campaign"] == "numerical" and fresh["problems"] == []
    assert len(fresh["next"]["runs"]) == 15
    assert fresh["next"]["check"] == numerical_digest(by_arm(first))


def test_after_void_a_fresh_numerical_campaign_has_every_attempt(
    tmp_path,
    clean_head,  # noqa: F811
) -> None:
    """The coordinator's ruling on concern C2: ``void`` moves the launcher's log of the campaign
    that made the voided runs (and its attempts' logs, its file under campaigns/ and that file's
    driver), so that the fresh campaign at the same rates, with the same cell names and config
    hashes, gets its full three technical attempts, and an exhausted fresh run is given up: the
    stage does not wait forever. Left in place, the voided starts would count as attempts and
    hide the given-up run."""
    root = tmp_path / "runs"
    runs = grids()
    _, result, _ = drive(runs, until="numerical")
    step = result["next"]
    module = runpy.run_path(str(CAMPAIGNS))
    tag = f"u{step['U']:g}-{step['check']}"  # as the campaigns script names the campaign
    campaign = {"header": {"w_max": W_MAX}, "name": f"study2-short-numerical-b20-{tag}",
                "cells": module["regular_cells"](step["runs"], f"-numerical-{tag}")}  # fmt: skip
    (root / "campaigns").mkdir(parents=True)
    path = root / "campaigns" / "numerical.yaml"
    path.write_text(yaml.safe_dump(campaign, sort_keys=False))
    (root / "campaigns" / "numerical-driver.sh").write_text("#!/usr/bin/env bash\n")
    name, launched = load_campaign(path)
    assert len(launched) == 15
    # The first launch, under the old GPU code: each run completed at its first attempt
    (root / "launch" / name).mkdir(parents=True)
    lines = []
    for n, run in enumerate(launched):
        run_id = f"old-{n}-{run.config_hash[:8]}-a1"
        lines += [json.dumps({"event": "start", "run": run.label, "attempt": 1, "run_id": run_id,
                              "experiment": run.experiment, "config_hash": run.config_hash,
                              "command": run.command()[1:], "header": {"w_max": W_MAX}}),
                  json.dumps({"event": "end", "run": run.label, "attempt": 1, "run_id": run_id,
                              "returncode": 0, "outcome": "done", "problem": None})]  # fmt: skip
        run_dir = root / run.experiment / run_id
        run_dir.mkdir(parents=True)
        OmegaConf.save(load_run_config(run.config, list(run.overrides)), run_dir / "config.yaml",
                       resolve=True)  # fmt: skip
        (run_dir / "meta.json").write_text(json.dumps({
            "git": {"commit": RUN_COMMIT, "dirty": False}, "config_hash": run.config_hash,
            "status": "completed"}))  # fmt: skip
        (root / "launch" / name / f"{run_id}.log").write_text("output\n")
    (root / "launch" / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
    old = launch_records(root)
    # A GPU-path change: HEAD's GPU code is OTHER's; void moves the runs and the record
    state = {"commit": OTHER, "dirty": False}
    plan = tool.void_plan(root, scope(), 20, 8.0, state)
    assert plan["problems"] == [] and len(plan["runs"]) == 15
    assert plan["paths"][15:] == [f"launch/{name}.jsonl", f"launch/{name}",
                                  "campaigns/numerical.yaml",
                                  "campaigns/numerical-driver.sh"]  # fmt: skip
    record = tool.void_numerical(root, plan, state)
    (inventory,) = record.parent.glob("inventory-*.tsv")
    listed = {line.split("\t")[0] for line in inventory.read_text().splitlines()[1:]}
    assert {f"launch/{name}.jsonl", f"launch/{name}/{old[0]['run_id']}.log",
            "campaigns/numerical.yaml", "campaigns/numerical-driver.sh"} <= listed  # fmt: skip
    assert launch_records(root) == [] and not (root / "campaigns" / "numerical.yaml").exists()
    # The fresh campaign at the same rates: the same cells and config hashes
    first = launched[0]
    assert len(assess(first, root, old).technical) == 1  # the voided start, had it stayed
    fresh = assess(first, root, launch_records(root))
    assert fresh.technical == set() and fresh.done is None

    def fails(run, run_id, output) -> int:  # a technical failure that leaves no run
        return 1

    header = {"w_max": W_MAX}
    assert run_campaign(name, [first], root, execute=fails, header=header)["exhausted"] == [
        first.label]  # fmt: skip
    starts = [r for r in launch_records(root) if r["event"] == "start"]
    assert [r["attempt"] for r in starts] == [1, 2, 3]  # its full three attempts
    run_campaign(name, [first], root, execute=fails, header=header, recovery="disk fixed")
    given = exhausted_runs(launch_records(root))
    assert list(given.values()) == [first.label]
    ((arm, batch, lr, seed),) = given
    assert (OPTIMIZERS[arm], batch, seed) == (load_run_config(
        first.config, list(first.overrides)).optimizer.name, 20, first.seed)  # fmt: skip
    # The selection's check then goes on without it; with the voided starts left, never
    assert exhausted_runs(old + launch_records(root)) == {}


def test_the_tool_and_the_selection_name_a_case_alike() -> None:
    """The given-up cases cross from the selection to the tool by name."""
    for case in cases():
        assert numerical_case_name(ARM_OF[case.optimizer], case.lr, case.seed) == case.name


# The tool's parts as its driver runs them (``driver_script``): the plan, the controls and the
# twins' first attempts, the verdict, then, a rerun being due, the reruns and the verdict again
DRIVER_STEPS = (["plan"], ["control"], ["twins"], ["verdict"], ["verdict", "--retry-needed"],
                ["control", "--retry"], ["twins", "--retry"], ["verdict"],
                ["verdict", "--retry-needed"])  # fmt: skip


def test_every_baseline_given_up_takes_the_technical_path(
    tmp_path,
    monkeypatch,
    clean_head,  # noqa: F811
) -> None:
    """Review fa14, 2, through the selection: the launcher gave up on every numerical run of a
    check. Its request names every case given up, with no runs' commit; the tool's real parts,
    run from the request on the tracked settings, record each without a GPU and write a verdict
    holding every rate and FA control not qualified (technical), with the tool's own GPU code.
    The selection reads it complete and takes the technical non-qualification path the frozen
    rules prescribe: every arm is re-solved once without those rates; in this world no arm has
    a feasible triple without them, so the candidate fails at the numerical gate and the
    selection goes on to the next candidate. The check is never asked for again."""
    root = tmp_path / "runs"  # no numerical run: the launcher gave up on every one
    root.mkdir()
    no_gpu(monkeypatch)
    runs = grids()
    _, result, _ = drive(runs, until="numerical")
    step = result["next"]
    assert len(step["runs"]) == 15
    exhausted = {(r["arm"], r["batch"], r["lr"], r["seed"]): f"given-up-{n}"
                 for n, r in enumerate(step["runs"])}  # fmt: skip

    def at():
        return selection(runs, THETA0, W_MAX, seconds_per_chunk=SPC,
                         numerical=read_numerical(root), facts=commits(runs, RUN_COMMIT),
                         exhausted=exhausted, reuse=reuse)  # fmt: skip

    found = at()
    check = found["next"]["check"]
    assert found["next"]["campaign"] == "numerical-check" and found["problems"] == []
    assert len(check["given_up"]) == 15 and check["runs_commits"] == []
    solved = {arm: len(a["selections"]) for arm, a in found["attempts"][0]["arms"].items()}
    request = tmp_path / f"{check['name']}.json"  # as the campaigns script writes it
    request.write_text(json.dumps(check["selection"], indent=2) + "\n")
    codes = []
    for part in DRIVER_STEPS:
        argv = [str(SCRIPT), *part, "--selection", str(request), "--batch", "20", "--upper", "8",
                "--runs-root", str(root)]  # fmt: skip
        codes.append(call(monkeypatch, argv))
    assert codes == [0, 0, 0, 0, 0, 0, 0, 0, 3]
    verdict = read_numerical(root)[check["verdict"]]
    assert set(verdict["rates"].values()) == {"not_qualified"} and len(verdict["rates"]) == 9
    assert set(verdict["controls"].values()) == {"not_qualified"}
    assert verdict["policy"]["gpu_code"] == fake_identity(COMMIT)
    after = at()
    assert after["problems"] == []
    (entry,) = after["attempts"][0]["numerical"]
    assert (entry["check"], entry["state"], entry["commits"]) == (check["name"], "complete", [])
    requested = {(arm, lr) for arm, lrs in by_arm(check["selection"]["rates"]).items()
                 for lr in lrs}  # fmt: skip
    assert {(u["arm"], u["lr"]) for u in entry["unusable"]} == requested
    assert {u["state"] for u in entry["unusable"]} == {"not_qualified", "control not_qualified"}
    # The technical path: one re-solve per arm, infeasible without those rates, so the candidate
    # fails at the numerical gate, never as instability, and the next candidate's step follows
    first = after["attempts"][0]
    assert {arm: len(a["selections"]) for arm, a in first["arms"].items()} == {
        arm: n + 1 for arm, n in solved.items()}  # fmt: skip
    assert not any(a["selections"][-1]["feasible"] for a in first["arms"].values())
    assert (first["outcome"], first["failed_at"]) == ("failed", "numerical")
    assert sorted(first["failed_arms"]) == sorted(solved)
    assert "without a feasible triple once the unusable rates are dropped" in first["reasons"][0]
    assert "not qualified (technical)" in first["reasons"][0]
    assert "instability" not in first["reasons"][0]
    assert after["next"]["campaign"] == "discovery" and after["next"]["U"] == 7
    assert at()["next"] == after["next"]  # read again: the check is not asked for again
