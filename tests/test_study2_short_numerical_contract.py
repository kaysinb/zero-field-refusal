"""The contract between the numerical-check tool and the selection (review of Task 2, important
2): verdicts built by the tool's own ``fa_control_diagnostic.numerical_verdict`` from its
synthetic records (the world of tests/test_numerical_check.py), written by its
``write_verdict``, read back by ``study2_short.read_numerical`` and through the selection's
``numerical_verdict`` and ``selection``. The verdict's shape is never copied here."""

import shutil

from test_numerical_check import (
    COMMIT,
    RATES,
    RUN_COMMIT,
    attempt,
    cases,
    clean_head,  # noqa: F401  (the fixture)
    consistent_code,
    control,
    fake_reuse,
    fake_runs,
    settings,
    world,  # noqa: F401  (the fixture)
    write_records,
)
from test_study2_short import SPC, THETA0, W_MAX, commits, drive, grids, made

from zero_field_refusal import fa_control_diagnostic as tool
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    OPTIMIZERS,
    numerical_digest,
    numerical_policy,
    numerical_verdict,
    read_numerical,
    selection,
)

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
