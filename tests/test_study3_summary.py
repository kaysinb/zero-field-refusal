"""Study 3's summary script (scripts/study3_summary.py): the state as JSON, the guard between
confirmation blocks, and the records it writes (the versions record, the staged fixation, the
decision records and a closure record).

The worlds are the campaign writer's tests' (``test_study3_campaigns.World``: a RUNS_DIR with its
decision records, the runs as the launcher and the tracker write them, each regular run holding a
harmful-response file and an observations file with a sentinel text, never opened; the block files
are the writer's) and the versions tests' (``test_study3_versions.Study``: a Git checkout and its
RUNS_DIR). The state is ``study3_progress.progress``'s over the files, or, where the route would
take long to build in files, ``advance``'s over the progress tests' records; the versions check is
the real one on a Git checkout, and is replaced where the state is ``advance``'s."""

import copy
import hashlib
import json
import os
import pickle
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from test_study3_campaigns import (
    EXPECTED,
    SENTINEL,
    VERSIONS,
    World,
    campaign_of,
    confirmation_study,
    explicit_python,
    script_module,
    sealed_reads,
)
from test_study3_decisions import HELD, STATED
from test_study3_decisions import held as decision_hold
from test_study3_progress import START, Study, run_of, unsupported
from test_study3_versions import CONFIRMATION_C1, FIRST, SECOND, VALIDATION_C1
from test_study3_versions import Study as VersionsStudy
from test_study3_versions import sha256 as vsha256

from zero_field_refusal import launch
from zero_field_refusal.analysis import study3_versions as sv
from zero_field_refusal.analysis.study3_decisions import decision_path, end_lines, read_decision
from zero_field_refusal.analysis.study3_records import SEEDS, Endpoint
from zero_field_refusal.launch import load_campaign
from zero_field_refusal.paths import REPO_ROOT

SUMMARY = REPO_ROOT / "scripts" / "study3_summary.py"


class Summary:
    """The summary script loaded as a module, its checkout the world's. ``core`` gives it the
    state of ``advance`` over the progress tests' records; ``versions`` replaces its versions
    check, which then gives ``problems`` and records each call."""

    def __init__(self, world: World, monkeypatch, capsys) -> None:
        self.world, self.monkeypatch, self.capsys = world, monkeypatch, capsys
        self.script = script_module(SUMMARY, "study3_summary")
        monkeypatch.setattr(self.script, "REPO", world.repo)
        self.checked: list[tuple[Path, dict]] = []

    def run(self, *argv) -> tuple[int, str, str]:
        """The command's exit code, its output and its error text (a refusal's message
        included)."""
        self.capsys.readouterr()
        code, message = 0, ""
        try:
            self.script.main([str(a) for a in argv])
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
            message = exc.code if isinstance(exc.code, str) else ""
        captured = self.capsys.readouterr()
        return code, captured.out, captured.err + message

    def core(self, study: Study) -> None:
        self.monkeypatch.setattr(self.script, "progress", lambda *a, **k: study.advance())

    def versions(self, problems=()) -> None:
        def check(record, **kw):
            self.checked.append((Path(record), kw))
            return list(problems)

        self.monkeypatch.setattr(self.script, "versions_problems", check)

    def guard(self, seed: int, campaign: Path, candidate: int = 1) -> tuple[int, str, str]:
        return self.run("guard", "--runs-root", self.world.root, "--candidate", candidate,
                        "--seed", seed, "--campaign", campaign)  # fmt: skip


@pytest.fixture
def world(tmp_path, monkeypatch, capsys) -> World:
    return World(tmp_path, monkeypatch, capsys)


@pytest.fixture
def summary(world, monkeypatch, capsys) -> Summary:
    return Summary(world, monkeypatch, capsys)


def block_files(out: Path) -> dict[int, Path]:
    """A confirmation folder's block files, by their seed."""
    return {campaign_of(path)["cells"][0]["seeds"][0]: path for path in out.glob("*.yaml")}


def refused(found: tuple[int, str, str], *words: str) -> str:
    """The command refused (non-zero), its message naming every word; the message."""
    code, _, error = found
    assert code not in (0, None), found
    assert all(word in error for word in words), (words, error)
    return error


def keys(node) -> set[str]:
    """Every mapping key of a JSON value."""
    if isinstance(node, dict):
        return set(node) | {k for value in node.values() for k in keys(value)}
    if isinstance(node, list):
        return {k for value in node for k in keys(value)}
    return set()


# ------------------------------------------------------------------ progress


def test_progress_prints_the_state_as_json(world, summary, monkeypatch) -> None:
    """The state of the records, as ``progress`` gives it under this checkout's GPU code and
    --theta0: JSON of numbers, ids, paths and hashes; nothing sealed opened or printed."""
    world.ready()
    world.launched(world.write("search-round0"))
    sealed = sealed_reads(monkeypatch)
    code, printed, error = summary.run("progress", "--runs-root", world.root, "--theta0",
                                       world.runs.theta0)  # fmt: skip
    assert code == 0, error
    shown = json.loads(printed)
    assert shown == json.loads(json.dumps(world.state(), default=str))
    assert (shown["stage"], shown["step"]["campaign"], shown["problems"]) == (
        "numerical", "numerical", [])  # fmt: skip
    assert all(key == "spread_pp" for key in keys(shown) if "spread" in key)
    assert sealed == [] and SENTINEL not in printed + error
    # θ₀'s evaluation is the one named: without it the state blocks, and says so
    code, printed, _ = summary.run("progress", "--runs-root", world.root, "--theta0",
                                   world.root / "no-theta0")  # fmt: skip
    assert code == 0 and json.loads(printed)["step"]["kind"] == "blocked"
    assert "θ₀" in printed


def test_progress_prints_the_resolved_torn_lines(world, summary) -> None:
    """A crash cut a line of a campaign's log short, and the launcher's resolution named it
    (``launch.resolve_torn``): the printed state lists the line by its log, its line and its
    kind; nothing else of its bytes is printed."""
    world.ready()
    path = world.write("search-round0")
    world.launched(path)
    name = load_campaign(path)[0]
    log = world.root / "launch" / f"{name}.jsonl"
    number, offset = log.read_bytes().count(b"\n") + 1, log.stat().st_size
    data = b'{"time": "2026-10-27T01:00:00+00:00", "event": "end", "run": "TORN-SECRET-9a\xff'
    with log.open("ab") as f:
        f.write(data)  # no newline after it: a crash in the middle of the append
    launch.resolve_torn(world.root, log.name, offset=offset,
                        sha256=hashlib.sha256(data).hexdigest(),
                        reason="the end is settled from the run directory")  # fmt: skip
    code, printed, error = summary.run("progress", "--runs-root", world.root, "--theta0",
                                       world.runs.theta0)  # fmt: skip
    assert code == 0, error
    shown = json.loads(printed)
    assert shown["resolved_torn_lines"] == [
        {"log": f"launch/{name}.jsonl", "line": number, "kind": "end"}]  # fmt: skip
    assert shown["problems"] == [] and "TORN-SECRET" not in printed + error


def scalars(node) -> list:
    """Every scalar value of a JSON value."""
    if isinstance(node, dict):
        return [v for value in node.values() for v in scalars(value)]
    if isinstance(node, list):
        return [v for value in node for v in scalars(value)]
    return [node]


def test_progress_prints_no_spread_at_the_confirmation(world, summary) -> None:
    """At the confirmation, with an n record that carries the spread (13.5 pp, and the
    evidence of the spread script and its output), the state names the confirmation's block and
    holds no spread: no key or value carries it (``progress`` prints no n record)."""
    study = confirmation_study(world, N=75)
    n = study.entries["n-c1"]["record"]
    n.update(spread_pp=13.5, spread={
        "script": {"path": "evidence/study3_spread.py", "sha256": "e" * 64, "root": "repo"},
        "output": {"path": "evidence/spread-output.json", "sha256": "f" * 64}})  # fmt: skip
    summary.core(study)
    code, printed, error = summary.run("progress", "--runs-root", world.root)
    assert code == 0, error
    shown = json.loads(printed)
    assert (shown["stage"], shown["step"]["campaign"], shown["problems"]) == (
        "confirmation", "confirmation", [])  # fmt: skip
    assert not [key for key in keys(shown) if "spread" in key]
    found = scalars(shown)
    assert not [v for v in found if isinstance(v, str) and ("spread" in v or "13.5" in v)]
    assert not [v for v in found if isinstance(v, float) and v == 13.5]


# ------------------------------------------------------------------ the guard


def test_the_guard_lets_the_next_block_start_and_checks_its_versions_record(world, summary):
    study = confirmation_study(world)
    summary.core(study)
    summary.versions()
    files = block_files(world.write("confirmation", "--blocks", "10400", "10402"))
    code, printed, error = summary.guard(10400, files[10400])
    assert code == 0, error
    assert "block 10400" in printed
    assert summary.checked == [(world.repo / VERSIONS, {
        "runs_root": world.root, "repo": world.repo, "step": "confirmation",
        "candidate": 1})]  # fmt: skip
    # one block at a time, in seed order
    refused(summary.guard(10401, files[10401]), "out of seed order", "the next block is 10400")
    study.confirmation(count=1)  # block 10400 ran
    assert summary.guard(10401, files[10401])[0] == 0
    refused(summary.guard(10400, files[10400]), "block 10400 has started", "another seed")
    refused(summary.guard(10402, files[10402]), "out of seed order", "the next block is 10401")


def test_the_guard_lets_a_started_block_complete_its_cells_after_the_stop(world, summary):
    """Every counter at N stops the confirmation, but a started block's missing cells are still
    issued (an unfinished earlier block is completed under the failure rules)."""
    study = confirmation_study(world, N=60)
    study.confirmation(count=61)
    study.drop(run_of(study, "confirmation", "P", "u", 10460))
    state = study.advance()
    assert state["confirmation"]["stop"] == {"stopped": True, "why": "counters"}
    assert state["confirmation"]["may_start_next"] is False
    (only,) = world.write("confirmation").glob("*.yaml")
    summary.core(study)
    summary.versions()
    code, printed, error = summary.guard(10460, only)
    assert code == 0, error
    assert "completes" in printed
    later = only.with_name("later.yaml")
    later.write_text(only.read_text().replace("10460", "10461"))
    refused(
        summary.guard(10461, later), "block 10461 may not start", "the confirmation has stopped"
    )


def test_the_guard_refuses_once_every_counter_has_reached_n(world, summary) -> None:
    study = confirmation_study(world, N=60)
    study.confirmation(count=59)
    files = block_files(world.write("confirmation", "--blocks", "10459", "10460"))
    summary.core(study)
    summary.versions()
    assert summary.guard(10459, files[10459])[0] == 0
    study.confirmation(count=1, first=59)  # the 60th supported block
    refused(summary.guard(10460, files[10460]), "the confirmation has stopped",
            "every active counter has reached N = 60")  # fmt: skip
    refused(summary.guard(10460, files[10460], candidate=2), "not candidate 2's confirmation")
    assert len(summary.checked) == 1  # 10459's only


def test_the_guard_refuses_once_the_cap_has_started(world, summary) -> None:
    """N = 60 caps the confirmation at 70 started blocks: with A unsupported in 20 of them, A–F
    is short of N there, and no block starts after them."""
    study = confirmation_study(world, N=60)
    study.confirmation(count=69, special=unsupported(SEEDS["confirmation"][1][:20]))
    files = block_files(world.write("confirmation"))
    assert list(files) == [10469]  # the last block within the cap
    summary.core(study)
    summary.versions()
    assert summary.guard(10469, files[10469])[0] == 0
    study.confirmation(count=1, first=69)
    beyond = files[10469].with_name("beyond.yaml")
    beyond.write_text(files[10469].read_text().replace("10469", "10470"))
    refused(summary.guard(10470, beyond), "the confirmation has stopped",
            "the cap of 70 blocks has started")  # fmt: skip


def test_the_guard_refuses_while_a_technical_mark_holds_the_confirmation(world, summary):
    study = confirmation_study(world, N=60)
    study.confirmation(count=3)
    files = block_files(world.write("confirmation"))
    broken = study.technical(run_of(study, "confirmation", "F", "l", 10402),
                             log="launch/study3-c1-confirmation.jsonl")  # fmt: skip
    summary.core(study)
    summary.versions()
    refused(summary.guard(10403, files[10403]), "a technical mark holds the confirmation",
            broken.run_id)  # fmt: skip
    assert summary.checked == []


def test_the_guard_refuses_while_a_run_under_another_commit_is_unresolved(world, summary):
    """A confirmation run under another commit is a technical mark until its way out: the step
    is blocked, and no block starts."""
    study = confirmation_study(world, N=60)
    study.confirmation(count=3)
    files = block_files(world.write("confirmation"))
    run_id = run_of(study, "confirmation", "F", "l", 10402)
    study.other_commit(run_id)
    summary.core(study)
    summary.versions()
    refused(summary.guard(10403, files[10403]), "a technical mark holds the confirmation", run_id)
    assert summary.checked == []


def test_the_guard_refuses_while_a_started_block_holds_a_run_not_ended(world, summary) -> None:
    """A run of a started block that the launcher never ended (the Pod stopped mid-run): no
    later block starts, and the refusal names the run and its recovery round."""
    study = confirmation_study(world, N=60)
    study.confirmation(count=3)
    files = block_files(world.write("confirmation", "--blocks", "10403", "10404"))
    dead = study.change(run_of(study, "confirmation", "F", "l", 10402), state="pending",
                        K=None)  # fmt: skip
    summary.core(study)
    summary.versions()
    refused(summary.guard(10403, files[10403]), "block 10403 may not start", "not ended",
            dead.run_id, f"--only {dead.label} --max-attempts 1 --recovery")  # fmt: skip
    # a run whose recovery round is opened and has no later attempt holds it too
    study.change(dead.run_id, state="recovered")
    refused(summary.guard(10403, files[10403]), "block 10403 may not start", dead.run_id)
    assert summary.checked == []
    # once its next attempt has ended, the next block starts
    study.change(dead.run_id, state="usable", K=dead.K)
    assert summary.guard(10403, files[10403])[0] == 0


def test_the_guard_refuses_a_seed_that_is_not_the_steps(world, summary) -> None:
    study = confirmation_study(world)
    files = block_files(world.write("confirmation", "--blocks", "10400", "10401"))
    summary.core(study)
    summary.versions()
    refused(summary.guard(10300, files[10400]), "not one of candidate 1's confirmation seeds")
    refused(summary.guard(10401, files[10401]), "out of seed order")
    # a started block whose cells are missing is completed first: no later block starts
    study.confirmation(count=1)
    study.drop(run_of(study, "confirmation", "A", "l", 10400))
    refused(summary.guard(10401, files[10401]), "block 10401 may not start",
            "completes the started block(s) [10400] first")  # fmt: skip
    assert summary.checked == []


def test_the_guard_refuses_when_the_step_is_not_the_confirmation(world, summary) -> None:
    study = confirmation_study(world)
    files = block_files(world.write("confirmation"))
    summary.versions()
    summary.core(study)
    refused(summary.guard(10400, files[10400], candidate=2), "not candidate 2's confirmation",
            '"campaign": "confirmation"')  # fmt: skip
    validation = Study()
    validation.ready()
    validation.search()
    validation.numerical()
    summary.core(validation)
    refused(summary.guard(10400, files[10400]), "not candidate 1's confirmation",
            '"campaign": "validation"')  # fmt: skip
    blocked = Study()
    blocked.ready()
    blocked.problems.append("20261026T000099Z-x-s10400-abcd1234-a1: a run with no start")
    summary.core(blocked)
    refused(summary.guard(10400, files[10400]), "the step is blocked", "a run with no start")
    assert summary.checked == []


def edited(path: Path, change) -> Path:
    """A copy of a block file with ``change`` applied to its YAML data."""
    data = yaml.safe_load(path.read_text())
    change(data)
    copy = path.with_name(f"edited-{path.name}")
    copy.write_text(yaml.safe_dump(data, sort_keys=False))
    return copy


def _cell(data, index=0) -> dict:
    return data["cells"][index]


EDITS = {
    "another candidate": (lambda d: d["header"].update(candidate=2), "candidate is 2"),
    "another step": (lambda d: d["header"].update(step="validation"), "step is 'validation'"),
    "another study": (lambda d: d["header"].update(study=2), "study is 2"),
    "a round": (lambda d: d["header"].update(round=0), "round is 0"),
    "a key of no confirmation header": (
        lambda d: d["header"].update(simulation_record="x"),
        "simulation_record",
    ),
    "no n record": (lambda d: d["header"].pop("n_record"), "names no n record"),
    "a cell dropped": (lambda d: d["cells"].pop(), "are not the step's cells"),
    "a cell twice": (lambda d: d["cells"].append(dict(_cell(d))), "are not the step's cells"),
    "another rate": (
        lambda d: _cell(d)["overrides"].__setitem__(2, "optimizer.lr=1.5e-05"),
        "c1-confirmation-adamw-1.45e-05-s10400",
    ),
    "another role": (
        lambda d: _cell(d)["overrides"].__setitem__(3, "study3.role=validation"),
        "study3.role=validation",
    ),
    "another batch": (
        lambda d: _cell(d)["overrides"].__setitem__(1, "train.batch_size=5"),
        "train.batch_size=5",
    ),
    "an extra override": (lambda d: _cell(d)["overrides"].append("seed=1"), "seed=1"),
    "another config": (
        lambda d: _cell(d).update(config="experiments/study3_k_only"),
        "experiments/study3_k_only",
    ),
    "another seed": (lambda d: _cell(d).update(seeds=[10401]), "[10401]"),
    "two seeds": (lambda d: _cell(d).update(seeds=[10400, 10401]), "[10400, 10401]"),
    "another name": (lambda d: _cell(d).update(name="shared"), "'shared'"),
    "no cells": (lambda d: d.pop("cells"), "not a launcher campaign"),
    "an unknown optimizer": (
        lambda d: (
            _cell(d)["overrides"].__setitem__(0, "optimizer=sgd"),
            _cell(d).update(name="c1-confirmation-sgd-1.45e-05-s10400"),
        ),
        "name no arm and rate",
    ),  # fmt: skip
    "another rate, named so": (
        lambda d: (
            _cell(d)["overrides"].__setitem__(2, "optimizer.lr=1.5e-05"),
            _cell(d).update(name="c1-confirmation-adamw-1.5e-05-s10400"),
        ),
        "are not the step's cells",
    ),  # fmt: skip
    "another n record": (
        lambda d: d["header"].update(n_record="study3/decisions/n-c2.json"),
        "not study3/decisions/n-c1.json",
    ),
    "no versions record": (lambda d: d["header"].pop("versions_record"), "no versions record"),
    "a bare override": (lambda d: _cell(d)["overrides"].append("debug"), "key=value"),
    "an override twice": (
        lambda d: _cell(d)["overrides"].append("optimizer=adamw"),
        "are not the confirmation's",
    ),
    "the candidate as true": (lambda d: d["header"].update(candidate=True), "candidate is True"),
}


@pytest.mark.parametrize("edit", list(EDITS))
def test_the_guard_refuses_a_block_file_that_does_not_match_the_step(world, summary, edit):
    study = confirmation_study(world)
    files = block_files(world.write("confirmation"))
    summary.core(study)
    summary.versions()
    change, words = EDITS[edit]
    refused(summary.guard(10400, edited(files[10400], change)), "does not match", words)
    assert summary.checked == []
    assert summary.guard(10400, files[10400])[0] == 0  # the block file as written


def test_the_guard_refuses_a_block_whose_records_changed_since_it_was_written(world, summary):
    """A block is run as written, on the records its header names as they were then: a block
    left with six cells by P's closure, a changed n record or versions record, a missing or
    unreadable file are refused."""
    study = confirmation_study(world, N=60)
    study.confirmation(count=10)
    files = block_files(world.write("confirmation"))
    summary.core(study)
    summary.versions()
    # P closed after the block was written (a P run under another commit, its closure): the
    # step's block has four cells
    broken = run_of(study, "confirmation", "P", "u", 10405)
    study.other_commit(broken)
    study.closure(broken)
    refused(summary.guard(10410, files[10410]), "does not match", "are not the step's cells")
    closed = study.entries.pop("closure-c1-P")
    study.uncounted.clear()
    assert summary.guard(10410, files[10410])[0] == 0 and closed
    record = decision_path(world.root, "n-c1")
    kept = record.read_bytes()
    world.records.change("n-c1", written_at="2026-10-26T13:00:00Z")
    refused(summary.guard(10410, files[10410]), "does not match", "n_record", "sha256")
    record.write_bytes(kept)
    assert summary.guard(10410, files[10410])[0] == 0
    record.unlink()
    refused(summary.guard(10410, files[10410]), "does not match", "is no longer there")
    record.write_bytes(kept)
    (world.repo / VERSIONS).write_text('{"what": "versions, edited"}\n')
    refused(summary.guard(10410, files[10410]), "does not match", "versions_record", "sha256")
    refused(summary.guard(10410, world.out / "none.yaml"), "cannot be read")
    bad = world.out / "bad.yaml"
    bad.write_text("cells: [\n")
    refused(summary.guard(10410, bad), "cannot be read")
    assert len(summary.checked) == 2  # the two guards that passed


def test_the_guard_checks_the_budget_record_a_block_names(world, summary) -> None:
    study = confirmation_study(world, N=99, budget=True)
    files = block_files(world.write("confirmation"))
    summary.core(study)
    summary.versions()
    assert summary.guard(10400, files[10400])[0] == 0
    world.records.change("budget-c1", written_at="2026-10-26T13:00:00Z")
    refused(summary.guard(10400, files[10400]), "does not match", "budget_record", "sha256")
    refused(
        summary.guard(
            10400,
            edited(
                files[10400],
                lambda d: d["header"].update(budget_record="study3/decisions/budget-c2.json"),
            ),
        ),
        "not study3/decisions/budget-c1.json",
    )


def test_the_guard_follows_the_confirmations_own_authorization(world, summary,
                                                              monkeypatch) -> None:  # fmt: skip
    """The step names the next block only when ``confirmation_state`` authorizes it; the guard
    asks both, so a state whose two disagree starts nothing."""
    study = confirmation_study(world)
    files = block_files(world.write("confirmation", "--blocks", "10400", "10401"))
    summary.versions()
    state = study.advance()
    for change, words in ((dict(may_start_next=False), "reserved confirmation seeds are used up"),
                          (dict(next_seed=10401), "the next block is 10401")):  # fmt: skip
        doctored = copy.deepcopy(state)
        doctored["confirmation"].update(change)
        monkeypatch.setattr(summary.script, "progress", lambda *a, doctored=doctored, **k: doctored)
        refused(summary.guard(10400, files[10400]), "block 10400", words)
    assert summary.checked == []


def test_the_guard_refuses_on_a_versions_problem(world, summary) -> None:
    study = confirmation_study(world)
    files = block_files(world.write("confirmation"))
    summary.core(study)
    summary.versions(["docs/study3-versions.json: HEAD is not the recorded commit"])
    refused(summary.guard(10400, files[10400]), "the versions record does not hold",
            "HEAD is not the recorded commit")  # fmt: skip
    assert [kw["step"] for _, kw in summary.checked] == ["confirmation"]


def test_the_guard_opens_nothing_sealed(world, summary, monkeypatch) -> None:
    """A block file, or a record its header names, that resolves under a run's ``responses/``
    folder is refused unopened; the guard's own reads open nothing sealed."""
    study = confirmation_study(world)
    files = block_files(world.write("confirmation"))
    summary.core(study)
    summary.versions()
    folder = world.root / "study3" / "20261026T000000Z-x-s10400-abcd1234-a1" / "responses"
    folder.mkdir(parents=True)
    sealed_file = folder / "harmful_evaluation.jsonl"
    sealed_file.write_text(SENTINEL + "\n")
    attempts = sealed_reads(monkeypatch)
    assert summary.guard(10400, files[10400])[0] == 0
    error = refused(summary.guard(10400, sealed_file), "responses/ or observations/")
    assert SENTINEL not in error
    linked = world.out / "linked.yaml"
    linked.symlink_to(sealed_file)
    error = refused(summary.guard(10400, linked), "responses/ or observations/")
    record = decision_path(world.root, "n-c1")
    kept = record.read_bytes()
    record.unlink()
    record.symlink_to(sealed_file)
    error = refused(summary.guard(10400, files[10400]), "responses/ or observations/", "n-c1")
    assert SENTINEL not in error and attempts == []
    record.unlink()
    record.write_bytes(kept)
    versions = world.repo / VERSIONS
    kept = versions.read_bytes()
    versions.unlink()
    versions.symlink_to(sealed_file)
    error = refused(summary.guard(10400, files[10400]), "responses/ or observations/", VERSIONS)
    assert SENTINEL not in error and attempts == []
    versions.unlink()
    versions.write_bytes(kept)


# ------------------------------------------------------------------ the writer's driver


BOOT = """\
import os, pickle, runpy, sys
from pathlib import Path

script = sys.argv[1]
sys.argv = sys.argv[1:]
world = pickle.loads(Path(os.environ["GUARD_WORLD"]).read_bytes())
with Path(world["ran"]).open("a") as ran:
    ran.write(f"{sys.executable}\\t{Path(script).name}\\n")
if Path(script).name == "study3_summary.py":
    from zero_field_refusal import code_identity, paths
    from zero_field_refusal.analysis import study3_progress, study3_versions

    paths.REPO_ROOT = Path(world["repo"])

    def progress(*args, **kwargs):
        launched = Path(world["launched"])
        count = len(launched.read_text().splitlines()) if launched.exists() else 0
        return world["states"][count]

    study3_progress.progress = progress
    study3_versions.versions_problems = lambda *args, **kwargs: []
    code_identity.gpu_code_identity = lambda *args, **kwargs: {
        "commit": "a" * 40, "files": 1, "sha256": world["code"]}
runpy.run_path(script, run_name="__main__")
"""

LAUNCHER = """\
import os, sys
from pathlib import Path

with (Path(os.environ["RUNS_DIR"]) / "launched.txt").open("a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")
"""


def test_the_writers_driver_runs_the_real_guard_before_each_block(world, tmp_path) -> None:
    """run.sh as the writer wrote it, run by bash, calls this script's guard (the real one, in its
    own process) before each block and the launcher after it; only the launcher is a stand-in,
    which records each block it is given in RUNS_DIR. Its ``python`` is this test's interpreter,
    selected by ``explicit_python`` whatever PATH a sourced file sets, and each call records the
    interpreter it ran under and its script. The guard's state is ``advance``'s over the
    progress tests' records after the blocks launched so far, and its versions check is replaced
    (neither has a record on these files); it reads the real block files and the decision
    records they name. A refusal stops the driver before the block's launch."""
    study = confirmation_study(world, N=60)
    out = world.write("confirmation", "--blocks", "10400", "10402")
    files = block_files(out)
    (world.repo / "scripts").mkdir()
    shutil.copy(SUMMARY, world.repo / "scripts" / "study3_summary.py")
    (world.repo / "scripts" / "launch.py").write_text(LAUNCHER)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    boot = tmp_path / "boot.py"
    boot.write_text(BOOT)
    (bin_ / "python").write_text(f"#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "
                                 f'{shlex.quote(str(boot))} "$@"\n')  # fmt: skip
    (bin_ / "python").chmod(0o755)
    launched = world.root / "launched.txt"
    ran = tmp_path / "ran.txt"
    guarded, started = f"{sys.executable}\tstudy3_summary.py", f"{sys.executable}\tlaunch.py"

    def run(states) -> subprocess.CompletedProcess:
        launched.unlink(missing_ok=True)
        ran.unlink(missing_ok=True)
        config = tmp_path / "world.pickle"
        config.write_bytes(pickle.dumps({"repo": str(world.repo), "launched": str(launched),
                                         "ran": str(ran), "states": states,
                                         "code": "b" * 64}))  # fmt: skip
        env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "GUARD_WORLD": str(config),
               **explicit_python(bin_ / "python")}  # fmt: skip
        return subprocess.run(["bash", str(out / "run.sh")], env=env, cwd=tmp_path,
                              capture_output=True, text=True)  # fmt: skip

    states = []
    for first in range(4):  # the state before each block, then after the last
        states.append(study.advance())
        study.confirmation(count=1, first=first)
    done = run(states)
    assert done.returncode == 0, done.stderr
    assert ran.read_text().splitlines() == [guarded, started] * 3
    assert launched.read_text().splitlines() == [
        f"{files[seed]} --max-attempts 1" for seed in (10400, 10401, 10402)]  # fmt: skip
    assert [int(s) for s in re.findall(r"block (\d+) of candidate 1's", done.stdout)] == [
        10400, 10401, 10402]  # fmt: skip
    # a technical mark in block 10400: the guard of 10401 refuses, and its launch never comes
    marked = Study()
    marked.through_validation()
    marked.n(N=60)
    marked.confirmation(count=1)
    broken = marked.technical(run_of(marked, "confirmation", "F", "l", 10400),
                              log="launch/study3-c1-confirmation.jsonl")  # fmt: skip
    done = run([states[0], marked.advance()])
    assert done.returncode == 1
    assert "the guard refused block 10401 (exit 1)" in done.stderr
    assert "a technical mark holds the confirmation" in done.stderr and broken.run_id in done.stderr
    assert ran.read_text().splitlines() == [guarded, started, guarded]
    assert launched.read_text().splitlines() == [f"{files[10400]} --max-attempts 1"]
    # the n record changed after the blocks were written: the first guard refuses
    world.records.change("n-c1", written_at="2026-10-26T13:00:00Z")
    done = run(states)
    assert done.returncode == 1 and "n_record" in done.stderr and not launched.exists()
    assert ran.read_text().splitlines() == [guarded]


# ------------------------------------------------------------------ the versions record


@pytest.fixture
def checkout(tmp_path, monkeypatch, capsys) -> tuple[VersionsStudy, Summary]:
    """A Git checkout and its RUNS_DIR (the versions tests' ``Study``), and the summary script
    run from that checkout."""
    study = VersionsStudy(tmp_path, monkeypatch)
    return study, Summary(study, monkeypatch, capsys)


def fixation_argv(study: VersionsStudy, step: str = "validation", *more) -> list:
    return ["fixation", step, "--runs-root", study.root, "--candidate", 1, *more]


SEED_ARGS = ("--bootstrap-seed", 20261101, "--audit-seed", 20261102, "--agreement-seed",
             20261103)  # fmt: skip


def test_the_versions_record_is_written_from_a_clean_checkout_and_holds(checkout, monkeypatch):
    study, summary = checkout
    study.ready()
    assert summary.run(*fixation_argv(study))[0] == 0
    study.commit("the validation fixation", VALIDATION_C1)
    attempts = sealed_reads(monkeypatch)
    argv = ["versions", "--out", FIRST, "--reason", "validation_c1", "--fixation", VALIDATION_C1,
            "--runs-root", study.root]  # fmt: skip
    (study.repo / "notes.md").write_text("local notes\n")
    refused(summary.run(*argv), "clean checkout")
    assert not (study.repo / FIRST).exists()
    (study.repo / "notes.md").unlink()
    code, printed, error = summary.run(*argv)
    assert code == 0, error
    assert printed.strip() == str(study.repo / FIRST)
    record, problems = sv.read_versions(study.repo / FIRST, repo=study.repo)
    assert problems == []
    assert record["code"] == {"commit": study.head()}
    assert record["fixation"] == {"path": VALIDATION_C1,
                                  "sha256": vsha256(study.repo / VALIDATION_C1)}  # fmt: skip
    assert (record["supersedes"], record["reason"]) == (None, "validation_c1")
    refused(summary.run(*argv), "already written")
    study.commit("the versions record", FIRST)
    assert study.problems(step="validation", candidate=1) == []  # it holds, committed alone
    assert attempts == []


def test_a_later_versions_record_names_the_one_before_it(checkout) -> None:
    study, summary = checkout
    study.ready()
    summary.run(*fixation_argv(study))
    study.commit("the validation fixation", VALIDATION_C1)
    first = ["versions", "--reason", "validation_c1", "--fixation", VALIDATION_C1, "--runs-root",
             study.root]  # fmt: skip
    refused(summary.run(*first, "--out", FIRST, "--supersedes", FIRST, "--supersedes-reason",
                        "next_stage"), "the first versions record: it supersedes none")  # fmt: skip
    assert summary.run(*first, "--out", FIRST)[0] == 0
    study.commit("the versions record", FIRST)
    later = [*first, "--out", SECOND]
    refused(summary.run(*later), "supersedes docs/study3-versions.json")
    refused(summary.run(*later, "--supersedes", FIRST), "both or neither")
    refused(summary.run(*later, "--supersedes", "docs/study3-fixation-validation-c1.json",
                        "--supersedes-reason", "next_stage"),
            "supersedes docs/study3-versions.json, the record before it")  # fmt: skip
    refused(summary.run(*later, "--supersedes", FIRST, "--supersedes-reason", "A sentence."),
            "reason")  # fmt: skip
    assert not (study.repo / SECOND).exists()
    code, _, error = summary.run(*later, "--supersedes", str(study.repo / FIRST),
                                 "--supersedes-reason", "next_stage")  # fmt: skip
    assert code == 0, error
    record, problems = sv.read_versions(study.repo / SECOND, repo=study.repo)
    assert problems == [] and record["supersedes"] == {
        "path": FIRST, "sha256": vsha256(study.repo / FIRST), "reason": "next_stage"}  # fmt: skip


def test_the_versions_command_refuses_what_the_readers_would_not_accept(checkout) -> None:
    study, summary = checkout
    study.ready()
    summary.run(*fixation_argv(study))
    study.commit("the validation fixation", VALIDATION_C1)
    base = ["versions", "--runs-root", study.root]
    fixed = [*base, "--fixation", VALIDATION_C1]
    refused(summary.run(*fixed, "--out", FIRST, "--reason", "Written for the validation."),
            "reason")  # fmt: skip
    refused(summary.run(*fixed, "--out", "docs/versions.json", "--reason", "validation_c1"),
            "not a versions record")  # fmt: skip
    refused(summary.run(*base, "--out", FIRST, "--reason", "validation_c1", "--fixation",
                        "docs/study3-preregistration.md"), "is not a fixation record")  # fmt: skip
    refused(summary.run("versions", "--out", FIRST, "--reason", "validation_c1", "--fixation",
                        VALIDATION_C1), "--fixation needs --runs-root")  # fmt: skip
    refused(summary.run(*fixed, "--out", study.repo.parent / FIRST, "--reason", "validation_c1"),
            "is not in the checkout")  # fmt: skip
    refused(summary.run(*fixed, "--out", "docs/study3-versions-0.json", "--reason", "next_stage",
                        "--supersedes", FIRST, "--supersedes-reason", "next_stage"),
            "not a versions record")  # fmt: skip
    assert not (study.repo / FIRST).exists() and not (study.repo.parent / FIRST).exists()
    # without a fixation the record stands for no validation or confirmation run
    assert summary.run("versions", "--out", FIRST, "--reason", "engineering")[0] == 0
    record, problems = sv.read_versions(study.repo / FIRST, repo=study.repo)
    assert problems == [] and record["fixation"] is None


def campaign_file(path: Path, header: dict) -> Path:
    """A launcher campaign file with ``header`` (its cells do not matter to the check)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"name": path.stem, "header": header, "cells": []}))
    return path


def test_check_holds_before_a_validation_launch_and_refuses_otherwise(
    checkout, tmp_path, monkeypatch
) -> None:
    """``check`` exits 0 only when the campaign file is a validation or confirmation campaign of
    study 3 and the versions record its header names holds for its step and candidate, the
    running checkout's HEAD included: before a launch, a resumption or a recovery round."""
    study, summary = checkout
    study.validated()  # the validation fixation and the first record, each committed alone
    out = tmp_path / "campaigns"
    good = campaign_file(out / "c1-validation.yaml", study.header("validation"))
    attempts = sealed_reads(monkeypatch)
    code, printed, error = summary.run("check", "--runs-root", study.root, "--campaign", good)
    assert code == 0, error
    assert str(good) in printed and "holds" in printed
    # a file that is no validation or confirmation campaign of study 3
    for name, header in (
        ("study2", {**study.header("validation"), "study": 2}),
        ("search", {"study": 3, "candidate": 1, "step": "search", "round": 0}),
        ("numerical", {"study": 3, "candidate": 1, "step": "numerical", "round": None}),
        ("no-candidate", {**study.header("validation"), "candidate": 3}),
        ("no-header", None),
    ):
        refused(summary.run("check", "--runs-root", study.root, "--campaign",
                            campaign_file(out / f"{name}.yaml", header)),
                "not a validation or confirmation campaign of study 3")  # fmt: skip
    refused(summary.run("check", "--runs-root", study.root, "--campaign", out / "absent.yaml"),
            "cannot be read")  # fmt: skip
    # the record the header names, by another sha256, or a record the checkout does not hold
    refused(summary.run("check", "--runs-root", study.root, "--campaign", campaign_file(
        out / "other-sha.yaml", study.header("validation", sha="0" * 64))),
        "names docs/study3-versions.json with sha256 " + "0" * 64)  # fmt: skip
    refused(summary.run("check", "--runs-root", study.root, "--campaign", campaign_file(
        out / "other-record.yaml", {**study.header("validation"),
                                    "versions_record": "docs/study3-versions-2.json"})),
        "study3-versions-2.json")  # fmt: skip
    # the record does not stand for the confirmation: its fixation is the validation's
    refused(summary.run("check", "--runs-root", study.root, "--campaign", campaign_file(
        out / "c1-confirmation.yaml", study.header("confirmation"))),
        "the versions record does not hold", CONFIRMATION_C1)  # fmt: skip
    # a checkout that is not the recorded commit or its record-only child
    (study.repo / "notes.md").write_text("local notes\n")
    refused(summary.run("check", "--runs-root", study.root, "--campaign", good),
            "the versions record does not hold", "untracked")  # fmt: skip
    study.commit("notes")
    refused(summary.run("check", "--runs-root", study.root, "--campaign", good),
            "the versions record does not hold", "notes.md")  # fmt: skip
    # a campaign file under a run's sealed folder is not opened
    sealed = study.root / "study3" / "run" / "responses" / "c1-validation.yaml"
    sealed.parent.mkdir(parents=True)
    sealed.write_text(SENTINEL + "\n")
    error = refused(summary.run("check", "--runs-root", study.root, "--campaign", sealed),
                    "responses/ or observations/")  # fmt: skip
    assert SENTINEL not in error and attempts == []


def test_a_record_is_published_once(world, summary) -> None:
    """Records are published whole and only where none is: a record found at the path when it is
    linked into place is kept, and the command refuses."""
    path = world.repo / "published" / "record.json"  # a folder of its own (docs/ holds others)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pytest.raises(SystemExit) as found:
        summary.script.publish("versions", path, {"kind": "first"})
        summary.script.publish("versions", path, {"kind": "second"})
    assert "exists: a record is written once" in str(found.value.code)
    assert json.loads(path.read_text()) == {"kind": "first"}
    assert [p.name for p in path.parent.iterdir()] == ["record.json"]  # no partial file left


# ------------------------------------------------------------------ the staged fixation


def test_the_fixation_records_are_written_at_their_paths_once(checkout, monkeypatch) -> None:
    """The builders' records, at docs/study3-fixation-<step>-c<N>.json, from a clean checkout,
    under HEAD's commit and its GPU code; the validation fixation's stage read with --theta0."""
    study, summary = checkout
    study.ready(real=True)
    expected = study.fixation()
    attempts = sealed_reads(monkeypatch)
    (study.repo / "notes.md").write_text("local notes\n")
    refused(summary.run(*fixation_argv(study, "validation", "--theta0", study.runs.theta0)),
            "clean checkout")  # fmt: skip
    (study.repo / "notes.md").unlink()
    code, printed, error = summary.run(*fixation_argv(study, "validation", "--theta0",
                                                      study.runs.theta0))  # fmt: skip
    assert code == 0, error
    assert printed.strip() == str(study.repo / VALIDATION_C1)
    record, problems = study.read_validation()
    assert problems == []
    assert {k: v for k, v in record.items() if k != "written_at"} == {
        k: v for k, v in expected.items() if k != "written_at"}  # fmt: skip
    assert record["code"] == study.now()
    refused(summary.run(*fixation_argv(study, "validation", "--theta0", study.runs.theta0)),
            "exists")  # fmt: skip
    # the supplement: the n record, the registered seeds, the validation fixation committed
    study.n(75)
    refused(summary.run(*fixation_argv(study, "confirmation", *SEED_ARGS)), "clean checkout")
    study.commit("the validation fixation", VALIDATION_C1)
    code, printed, error = summary.run(*fixation_argv(study, "confirmation", *SEED_ARGS))
    assert code == 0, error
    assert printed.strip() == str(study.repo / CONFIRMATION_C1)
    record, problems = study.read_confirmation()
    assert problems == [] and record["code"] == study.now()
    assert (record["bootstrap"], record["audit"]) == (
        {"seed": 20261101, "resamples": 100_000},
        {"seed": 20261102, "agreement_seed": 20261103,
         "procedure": "pps-with-replacement-400-400-200"})  # fmt: skip
    refused(summary.run(*fixation_argv(study, "confirmation", *SEED_ARGS)), "exists")
    assert attempts == []


def test_the_fixation_command_refuses_what_its_builders_refuse(checkout) -> None:
    study, summary = checkout
    study.ready()
    # the confirmation's seeds go with the supplement, --theta0 with the validation fixation
    refused(summary.run(*fixation_argv(study, "validation", *SEED_ARGS)),
            "--agreement-seed go with the confirmation supplement")  # fmt: skip
    refused(summary.run(*fixation_argv(study, "confirmation", *SEED_ARGS, "--theta0",
                                       study.runs.theta0)),
            "--theta0 goes with the validation fixation")  # fmt: skip
    refused(summary.run(*fixation_argv(study, "confirmation", *SEED_ARGS[:4])),
            "needs --bootstrap-seed, --audit-seed and --agreement-seed")  # fmt: skip
    # the supplement before its validation fixation, a seed out of [0, 2³²)
    refused(summary.run(*fixation_argv(study, "confirmation", *SEED_ARGS)), "fixation-validation")
    summary.run(*fixation_argv(study))
    study.commit("the validation fixation", VALIDATION_C1)
    study.n(75)
    seeds = list(SEED_ARGS)
    seeds[1] = 2**32
    refused(summary.run(*fixation_argv(study, "confirmation", *seeds)), "bootstrap.seed")
    assert not (study.repo / CONFIRMATION_C1).exists()
    # a validation fixation for a candidate the study is not at
    refused(summary.run("fixation", "validation", "--runs-root", study.root, "--candidate", 2),
            "not at candidate 2's validation")  # fmt: skip
    assert not (study.repo / "docs/study3-fixation-validation-c2.json").exists()


# ------------------------------------------------------------------ the decision records


def test_record_writes_each_kind_once_and_prints_why_it_refuses(world, summary, tmp_path,
                                                                monkeypatch) -> None:  # fmt: skip
    records = world.records
    builders = {
        "engineering": records.engineering,
        "simulation": records.simulation,
        "n-c1": lambda: records.n(1, 99),
        "budget-c1": lambda: records.budget(1, 99),
    }
    attempts = sealed_reads(monkeypatch)
    for kind, build in builders.items():
        source = tmp_path / f"{kind}.json"
        source.write_text(json.dumps(build()))
        argv = ["record", kind, "--runs-root", world.root, "--from", source]
        code, printed, error = summary.run(*argv)
        assert code == 0, error
        assert printed.strip() == str(decision_path(world.root, kind))
        assert read_decision(world.root, kind, repo=world.repo) == (json.loads(source.read_text()),
                                                                    [])  # fmt: skip
        refused(summary.run(*argv), "already written")
    # an invalid record: refused with its problems, nothing written
    source = tmp_path / "n-c2.json"
    source.write_text(json.dumps({**records.n(2, 75), "spread_pp": 4.3}))
    refused(summary.run("record", "n-c2", "--runs-root", world.root, "--from", source),
            "not written", "spread_pp")  # fmt: skip
    source.write_text(json.dumps(records.n(1, 75)))  # n-c1's record as n-c2's
    refused(summary.run("record", "n-c2", "--runs-root", world.root, "--from", source), "kind")
    assert not decision_path(world.root, "n-c2").exists()
    source.write_text("{")
    refused(summary.run("record", "n-c2", "--runs-root", world.root, "--from", source),
            "cannot be read")  # fmt: skip
    refused(summary.run("record", "n-c2", "--runs-root", world.root, "--from",
                        tmp_path / "none.json"), "cannot be read")  # fmt: skip
    # a closure is written from the step (``closure``), never from a file
    code, _, error = summary.run("record", "closure-c1-F", "--runs-root", world.root, "--from",
                                 source)  # fmt: skip
    assert code == 2 and "invalid choice" in error
    assert attempts == [] and SENTINEL not in error


def test_record_opens_no_file_under_a_runs_sealed_folder(world, summary, monkeypatch) -> None:
    folder = world.root / "study3" / "20261026T000000Z-x-s10300-abcd1234-a1" / "responses"
    folder.mkdir(parents=True)
    sealed_file = folder / "harmful_evaluation.jsonl"
    sealed_file.write_text(SENTINEL + "\n")
    attempts = sealed_reads(monkeypatch)
    error = refused(summary.run("record", "engineering", "--runs-root", world.root, "--from",
                                sealed_file), "responses/ or observations/")  # fmt: skip
    linked = world.out / "linked.json"
    linked.symlink_to(sealed_file)
    error += refused(summary.run("record", "engineering", "--runs-root", world.root, "--from",
                                 linked), "responses/ or observations/")  # fmt: skip
    assert attempts == [] and SENTINEL not in error
    assert not decision_path(world.root, "engineering").exists()


# ------------------------------------------------------------------ closures


def held_study(world: World) -> tuple[Study, dict]:
    """F's lower rate held on 10201 after candidate 1's engineering cycle (the progress tests'
    ``held_numerical``), its held line and cycle record on disk as scripts/qualify.py writes them
    (``test_study3_decisions.held``); and the closure record that cites them."""
    closure = decision_hold(world.records, arm="F", rate=START["F"]["l"], seed=10201)
    study = Study()
    study.ready()
    study.search()
    study.numerical(states={("F", "l"): "not_qualified"})
    study.cycle()
    found = study.drop(run_of(study, "numerical", "F", "l", 10201))
    study.add(
        Endpoint(
            **{
                **found.__dict__,
                "run_id": HELD,
                "run_dir": Path(HELD),
                "state": "held",
                "K": None,
                "steps": None,
                "reasons": ("held by the cycle",),
            }
        )
    )
    line, cycle = closure["line"], closure["cycle"]
    study.holds.append({"run_id": HELD, "run": found.label, "experiment": "study3",
                        "config_hash": "c" * 12, "reason": STATED, "records": [
                            {"kind": "journal", "root": "runs", "path": line["path"],
                             "line": line["line"], "line_sha256": line["line_sha256"],
                             "sha256": "f" * 64},
                            {"kind": "supersession", "root": "runs", **cycle}]})  # fmt: skip
    return study, closure


FIELDS = ("candidate", "arm", "rate", "seed", "run_id", "line", "cycle", "reason", "decision")


def test_closure_writes_the_record_a_held_runs_step_names(world, summary, monkeypatch) -> None:
    study, closure = held_study(world)
    step = study.advance()["step"]
    assert (step["kind"], step["record"]) == ("decision", "closure-c1-F")
    assert step["fields"] == {key: closure[key] for key in FIELDS}
    summary.core(study)
    monkeypatch.setattr(
        summary.script, "git_state", lambda repo=None: {"commit": None, "dirty": None}
    )
    argv = ["closure", "--runs-root", world.root, "--candidate", 1, "--arm", "F"]
    refused(summary.run(*argv), "a known commit")
    monkeypatch.setattr(
        summary.script, "git_state", lambda repo=None: {"commit": "a" * 40, "dirty": True}
    )
    attempts = sealed_reads(monkeypatch)
    code, printed, error = summary.run(*argv)
    assert code == 0, error
    path = decision_path(world.root, "closure-c1-F")
    assert printed.strip() == str(path)
    record, problems = read_decision(world.root, "closure-c1-F", repo=world.repo)
    assert problems == []
    assert record == {**{key: closure[key] for key in FIELDS}, "kind": "closure-c1-F",
                      "state": "failed", "written_at": record["written_at"],
                      "code": {"commit": "a" * 40, "gpu_code_sha256": EXPECTED}}  # fmt: skip
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", record["written_at"])
    refused(summary.run(*argv), "already written")
    assert attempts == [] and SENTINEL not in printed + error


def test_closure_refuses_when_the_step_names_no_closure_of_the_arm(world, summary) -> None:
    study, _ = held_study(world)
    summary.core(study)
    refused(summary.run("closure", "--runs-root", world.root, "--candidate", 1, "--arm", "A"),
            "names no closure record closure-c1-A", '"record": "closure-c1-F"')  # fmt: skip
    refused(summary.run("closure", "--runs-root", world.root, "--candidate", 2, "--arm", "F"),
            "names no closure record closure-c2-F")  # fmt: skip
    refused(summary.run("closure", "--runs-root", world.root, "--candidate", 1, "--arm", "F",
                        "--run-id", "another-run"), "names no closure record closure-c1-F",
            "another-run")  # fmt: skip
    plain = Study()
    plain.ready()
    summary.core(plain)
    refused(summary.run("closure", "--runs-root", world.root, "--candidate", 1, "--arm", "F"),
            "names no closure record closure-c1-F", '"campaign": "search-round0"')  # fmt: skip
    for kind in ("closure-c1-A", "closure-c1-F", "closure-c2-F"):
        assert not decision_path(world.root, kind).exists()


def test_closure_writes_a_way_out_chosen_by_its_run(world, summary, monkeypatch) -> None:
    """Two completed validation runs of F under another commit: the step is blocked and names
    each one's closure among its ways out (its last end line in the journal, its reason); the
    run is chosen with --run-id."""
    study = Study()
    study.through_validation()
    run_ids = [run_of(study, "validation", "F", "u", seed) for seed in (10304, 10306)]
    for run_id in run_ids:
        (endpoint,) = [e for e in study.endpoints if e.run_id == run_id]
        world.runs.log("study3-c1-validation", {"event": "start", "run": endpoint.label,
                                                "attempt": 1, "run_id": run_id,
                                                "experiment": "study3"})  # fmt: skip
        world.runs.end("study3-c1-validation", endpoint.label, 1, run_id, "done")
    ends = end_lines(world.root)
    for run_id in run_ids:
        study.ends[run_id] = dict(ends[run_id])
        study.other_commit(run_id)
    step = study.advance()["step"]
    assert (step["kind"], step["technical"], len(step["ways_out"])) == ("blocked", True, 2)
    summary.core(study)
    monkeypatch.setattr(
        summary.script, "git_state", lambda repo=None: {"commit": "a" * 40, "dirty": False}
    )
    argv = ["closure", "--runs-root", world.root, "--candidate", 1, "--arm", "F"]
    refused(summary.run(*argv), "2 closures of closure-c1-F", run_ids[0], run_ids[1],
            "--run-id")  # fmt: skip
    code, _, error = summary.run(*argv, "--run-id", run_ids[1])
    assert code == 0, error
    record, problems = read_decision(world.root, "closure-c1-F", repo=world.repo)
    assert problems == [] and record["run_id"] == run_ids[1]
    assert (record["line"], record["reason"], record["cycle"]) == (
        {"kind": "end", **{k: ends[run_ids[1]][k] for k in ("path", "line", "line_sha256")}},
        "other_commit", None)  # fmt: skip


def test_closure_has_none_to_write_for_a_technical_mark_outside_the_qualification(
    world, summary, monkeypatch
) -> None:
    """A technical validation run stops the route for the researcher's decision: the step names
    no closure, and ``closure`` writes none."""
    study = Study()
    study.through_validation()
    run_id = run_of(study, "validation", "F", "u", 10304)
    (endpoint,) = [e for e in study.endpoints if e.run_id == run_id]
    world.runs.log("study3-c1-validation", {"event": "start", "run": endpoint.label,
                                            "attempt": 1, "run_id": run_id,
                                            "experiment": "study3"})  # fmt: skip
    world.runs.end("study3-c1-validation", endpoint.label, 1, run_id, "technical")
    end = end_lines(world.root)[run_id]
    study.technical(run_id, log=end["path"], line=end["line"])
    study.ends[run_id]["line_sha256"] = end["line_sha256"]
    step = study.advance()["step"]
    assert (step["kind"], step["reason"]) == ("blocked", "engineering_route_pending")
    summary.core(study)
    monkeypatch.setattr(
        summary.script, "git_state", lambda repo=None: {"commit": "a" * 40, "dirty": False}
    )
    refused(summary.run("closure", "--runs-root", world.root, "--candidate", 1, "--arm", "F"),
            "names no closure record closure-c1-F", "engineering_route_pending")  # fmt: skip
    assert not decision_path(world.root, "closure-c1-F").exists()


def test_closure_writes_the_closure_of_a_run_under_another_commit(world, summary,
                                                                 monkeypatch) -> None:  # fmt: skip
    """A completed validation run under another commit: the step names its closure (its last
    end line, its uncounted reason id), which ``closure`` writes and the schema accepts; once
    written, the route ends with the technical feasibility report, the run's reason its own."""
    study = Study()
    study.through_validation()
    run_id = run_of(study, "validation", "F", "u", 10304)
    (endpoint,) = [e for e in study.endpoints if e.run_id == run_id]
    world.runs.log("study3-c1-validation", {"event": "start", "run": endpoint.label,
                                            "attempt": 1, "run_id": run_id,
                                            "experiment": "study3"})  # fmt: skip
    world.runs.end("study3-c1-validation", endpoint.label, 1, run_id, "done")
    study.ends[run_id] = dict(end_lines(world.root)[run_id])
    study.other_commit(run_id, "dirty_checkout")
    summary.core(study)
    monkeypatch.setattr(
        summary.script, "git_state", lambda repo=None: {"commit": "a" * 40, "dirty": False}
    )
    code, _, error = summary.run("closure", "--runs-root", world.root, "--candidate", 1, "--arm",
                                 "F")  # fmt: skip
    assert code == 0, error
    record, problems = read_decision(world.root, "closure-c1-F", repo=world.repo)
    assert problems == [] and (record["run_id"], record["reason"], record["line"]["kind"]) == (
        run_id, "dirty_checkout", "end")  # fmt: skip
    study.entries["closure-c1-F"] = {"kind": "closure-c1-F", "state": "failed", "record": record,
                                     "path": str(decision_path(world.root, "closure-c1-F")),
                                     "problems": []}  # fmt: skip
    result = study.advance()
    assert (result["outcome"], result["reason"]) == (
        "technical_feasibility_report", "dirty_checkout")  # fmt: skip


def test_the_summary_imports_no_gpu_path() -> None:
    """The script imports no GPU path, no launcher and no study-2 analysis; the state's reader
    loads the qualification's reader (and torch with it) only when it runs."""
    code = ("import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('summary', {str(SUMMARY)!r}); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "print([m for m in ('torch', 'transformers', 'zero_field_refusal.schedule', "
            "'zero_field_refusal.train', 'zero_field_refusal.launch', "
            "'zero_field_refusal.qualification', 'zero_field_refusal.analysis.stage_a', "
            "'zero_field_refusal.analysis.study2_short') if m in sys.modules])")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
