"""The launch journal survives a torn append (``zero_field_refusal.launch``, "The journal"): the
writer's single write and fsync, the repair that names a torn line, the launcher's refusal while a
torn line is unresolved, what a lost line of each kind costs, and the resolution command."""

import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest
from test_launch import INCIDENT, R1, Executor, campaign, log_start, make_run_dir

from zero_field_refusal import launch
from zero_field_refusal.launch import (
    TORN_JOURNAL,
    CampaignError,
    TornJournalError,
    hold,
    launch_records,
    load_campaign,
    run_campaign,
    supersede,
    torn_lines,
)

ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-10-27T10:00:00+00:00"
MARKER = "zq-private-7f3a"  # text of a torn line that no output may copy


def script():
    spec = importlib.util.spec_from_file_location("launch_script", ROOT / "scripts" / "launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tear(log: Path, record: dict, lost: int = 12) -> bytes:
    """Append ``record`` as ``_log`` encodes it, with its last ``lost`` bytes and its newline lost
    (a crash in the middle of the append); returns the torn bytes."""
    torn = json.dumps({"time": STAMP, **record}).encode()[:-lost]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as f:
        f.write(torn)
    return torn


def files(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` with its bytes."""
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def refused(name: str, runs, root: Path, executor: Executor, **kwargs) -> list:
    """``run_campaign`` refuses: nothing starts and nothing under ``root`` changes."""
    before = files(root)
    with pytest.raises(TornJournalError) as refusal:
        run_campaign(name, runs, root, execute=executor, **kwargs)
    assert executor.calls == [] and files(root) == before
    return refusal.value.lines


def resolve(monkeypatch, root: Path, line, reason: str = "the lost line is restored") -> None:
    monkeypatch.setenv("RUNS_DIR", str(root))
    script().main(["--resolve-torn", line.log, "--offset", str(line.offset), "--sha256",
                   line.sha256, "--reason", reason])  # fmt: skip


def events(root: Path, event: str) -> list[dict]:
    return [r for r in launch_records(root) if r.get("event") == event]


# --------------------------------------------------------------------------------- the writer


def test_the_append_is_one_write_of_the_whole_line_then_an_fsync(tmp_path, monkeypatch) -> None:
    """The log is opened write-only for appending (created 0o644), the encoded line goes out in
    one ``os.write`` and is fsynced, and so are its folder and the runs root, on every append."""
    log = tmp_path / "launch" / "c.jsonl"
    seen, fds = [], {}
    real_open, real_write, real_fsync = os.open, os.write, os.fsync

    def spy_open(path, flags, mode=0o777, *args, **kwargs):
        fd = real_open(path, flags, mode, *args, **kwargs)
        fds[fd] = Path(path)
        seen.append(("open", Path(path), flags, mode))
        return fd

    def spy_write(fd, data):
        seen.append(("write", fds.get(fd), bytes(data)))
        return real_write(fd, data)

    def spy_fsync(fd):
        seen.append(("fsync", fds.get(fd)))
        return real_fsync(fd)

    monkeypatch.setattr(launch.os, "open", spy_open)
    monkeypatch.setattr(launch.os, "write", spy_write)
    monkeypatch.setattr(launch.os, "fsync", spy_fsync)
    launch._log(log, {"event": "start", "run": "a/s0"})
    first = log.read_bytes()
    assert ("open", log, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644) in seen
    writes = [s for s in seen if s[0] == "write"]
    assert writes == [("write", log, first)] and first.endswith(b"\n") and first.count(b"\n") == 1
    assert seen.index(("write", log, first)) < seen.index(("fsync", log))
    assert ("fsync", log.parent) in seen  # the log was created
    assert json.loads(first)["event"] == "start" and list(json.loads(first))[0] == "time"
    assert stat.S_IMODE(log.stat().st_mode) == 0o644 & ~_umask()
    seen.clear()
    launch._log(log, {"event": "end", "run": "a/s0"})
    second = log.read_bytes()[len(first) :]
    assert [s for s in seen if s[0] == "write"] == [("write", log, second)]
    assert ("fsync", log) in seen and ("fsync", log.parent) in seen  # every append


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


@pytest.mark.parametrize("before", ["whole", "empty"])
def test_a_log_that_ends_with_its_newline_gets_no_repair(tmp_path, before) -> None:
    log = tmp_path / "launch" / "c.jsonl"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"" if before == "empty" else json.dumps({"event": "start"}).encode() + b"\n")
    launch._log(log, {"event": "end"})
    assert [r["event"] for r in launch_records(tmp_path)] == (
        ["end"] if before == "empty" else ["start", "end"])  # fmt: skip
    assert torn_lines(tmp_path) == []


@pytest.mark.parametrize("earlier", [True, False])
def test_a_torn_last_line_is_named_before_the_append(tmp_path, earlier) -> None:
    """The crash: the log ends in a partial line. The next append writes a newline, then a
    ``torn`` record naming the partial line's exact bytes (offset, length, sha256) and its kind,
    then its record; a second append adds no second ``torn`` record."""
    log = tmp_path / "launch" / "c.jsonl"
    if earlier:
        launch._log(log, {"event": "start", "run": "a/s0", "run_id": "r1"})
    whole = log.read_bytes() if earlier else b""
    torn = tear(log, {"event": "end", "run": "a/s0", "run_id": "r1", "outcome": "done"})
    launch._log(log, {"event": "stop", "run": "a/s0", "reason": "stop requested"})
    data = log.read_bytes()
    assert data.startswith(whole + torn + b"\n")
    named, record = (json.loads(x) for x in data[len(whole) + len(torn) + 1 :].splitlines())
    assert named == {"time": named["time"], "event": "torn", "offset": len(whole),
                     "length": len(torn), "sha256": hashlib.sha256(torn).hexdigest(),
                     "kind": "end"}  # fmt: skip
    assert record["event"] == "stop" and record["reason"] == "stop requested"
    launch._log(log, {"event": "stop", "run": "a/s1", "reason": "stop requested"})
    assert [r["event"] for r in launch_records(tmp_path)] == (["start"] if earlier else []) + [
        "torn", "stop", "stop"]  # fmt: skip


def test_a_short_write_raises_and_the_next_append_names_its_bytes(tmp_path, monkeypatch) -> None:
    log = tmp_path / "launch" / "c.jsonl"
    launch._log(log, {"event": "start", "run": "a/s0"})
    whole = log.read_bytes()
    real = os.write
    monkeypatch.setattr(launch.os, "write", lambda fd, data: real(fd, data[: len(data) // 2]))
    with pytest.raises(OSError, match="short write"):
        launch._log(log, {"event": "end", "run": "a/s0", "outcome": "done", "problem": None})
    monkeypatch.setattr(launch.os, "write", real)
    torn = log.read_bytes()[len(whole) :]
    assert torn and not torn.endswith(b"\n")
    launch._log(log, {"event": "stop", "run": "a/s0"})
    (named,) = events(tmp_path, "torn")
    assert (named["offset"], named["length"], named["sha256"], named["kind"]) == (
        len(whole), len(torn), hashlib.sha256(torn).hexdigest(), "end")  # fmt: skip
    (line,) = torn_lines(tmp_path)
    assert (line.offset, line.length, line.repaired, line.resolved) == (
        len(whole), len(torn), True, False)  # fmt: skip


PREFIX = b'{"time": "2026-10-27T10:00:00+00:00", '


@pytest.mark.parametrize(
    ("torn", "kind"),
    [(PREFIX + b'"event": "held", "run": "' + MARKER.encode(), "held"),
     (PREFIX + b'"event": "torn_resolved", "log": "' + MARKER.encode(), "torn_resolved"),
     (PREFIX + b'"ev', None),
     (PREFIX + b'"event": "Held", "run": "' + MARKER.encode(), None),
     (PREFIX + b'"event":"held", "run": "' + MARKER.encode(), None),
     (PREFIX + b'"event": "held' + MARKER.encode(), None),
     (PREFIX + b'"event": "<b>' + MARKER.encode() + b'</b>", ', None),
     (PREFIX + b'"run": "\\"event\\": \\"held\\" ' + MARKER.encode(), None),
     (MARKER.encode() + b" garbage", None),
     (b"\x00" * 24, None)],
)  # fmt: skip
def test_the_kind_comes_from_a_strict_pattern_and_nothing_else_is_copied(
    tmp_path, monkeypatch, capsys, torn, kind
) -> None:
    """The kind is the lower-case word of ``"event": "<word>"`` in the torn bytes, or null. No
    other text of the torn line appears in any output: the ``torn`` and ``torn_resolved``
    records, ``torn_lines``, the refusal and what the script prints."""
    root = tmp_path / "runs"
    log = root / "launch" / "c.jsonl"
    log.parent.mkdir(parents=True)
    original = json.dumps({"event": "start"}).encode() + b"\n" + torn
    log.write_bytes(original)
    (line,) = torn_lines(root)
    assert line.kind == kind and (line.repaired, line.resolved) == (False, False)
    assert MARKER not in repr(torn_lines(root))
    name, runs = load_campaign(campaign(tmp_path))
    with pytest.raises(TornJournalError) as refusal:
        run_campaign(name, runs[1:2], root, execute=Executor(root, {}))
    assert MARKER not in str(refusal.value)
    module = script()
    real = module.run_campaign
    monkeypatch.setattr(module, "run_campaign",
                        lambda *a, **k: real(*a, execute=Executor(root, {}), **k))  # fmt: skip
    monkeypatch.setenv("RUNS_DIR", str(root))
    with pytest.raises(SystemExit) as stop:
        module.main([str(campaign(tmp_path))])
    assert stop.value.code == TORN_JOURNAL
    resolve(monkeypatch, root, line)
    launch._log(log, {"event": "stop"})
    printed = capsys.readouterr()
    after = log.read_bytes()[len(original) :]
    for text in (printed.out, printed.err, after.decode(), repr(torn_lines(root))):
        assert MARKER not in text
    (named,) = events(root, "torn")
    assert named["kind"] == kind


# ----------------------------------------------------------------------------- the reading


def test_torn_lines_lists_every_unparseable_line_of_every_log(tmp_path, monkeypatch) -> None:
    """Every unparseable line of every log, a valid JSON line that is not an object included,
    with its log, offset, length, sha256 and kind; whether a ``torn`` record names exactly its
    bytes (an unparseable last line without its newline has none yet) and whether a
    ``torn_resolved`` record names it. ``launch_records`` still skips them."""
    root = tmp_path / "runs"
    a, b = root / "launch" / "a.jsonl", root / "launch" / "b.jsonl"
    launch._log(a, {"event": "start", "run_id": "r1"})
    array_at = a.stat().st_size
    with a.open("ab") as f:
        f.write(b"[1, 2]\n")
    launch._log(a, {"event": "end", "run_id": "r1"})
    last_at = a.stat().st_size
    last = tear(a, {"event": "stop", "run": "x/s0"})
    early = tear(b, {"event": "held", "run": "x/s0", "run_id": "r9"})
    launch._log(b, {"event": "recovery", "cell": "x"})
    lines = torn_lines(root)
    assert [(x.log, x.offset, x.length, x.sha256, x.kind, x.repaired, x.resolved)
            for x in lines] == [
        ("a.jsonl", array_at, 6, hashlib.sha256(b"[1, 2]").hexdigest(), None, False, False),
        ("a.jsonl", last_at, len(last), hashlib.sha256(last).hexdigest(), "stop", False, False),
        ("b.jsonl", 0, len(early), hashlib.sha256(early).hexdigest(), "held", True, False),
    ]  # fmt: skip
    assert [r["event"] for r in launch_records(root)] == ["start", "end", "torn", "recovery"]
    with b.open("ab") as f:  # a ``torn`` record of other bytes names nothing
        f.write(json.dumps({"event": "torn", "offset": 0, "length": len(early),
                            "sha256": "0" * 64, "kind": "held"}).encode() + b"\n")  # fmt: skip
    assert [x.repaired for x in torn_lines(root)] == [False, False, True]
    resolve(monkeypatch, root, lines[2])
    assert [x.resolved for x in torn_lines(root)] == [False, False, True]
    resolve(monkeypatch, root, lines[1])  # the unrepaired last line: repaired, then resolved
    assert [(x.repaired, x.resolved) for x in torn_lines(root)] == [
        (False, False), (True, True), (True, True)]  # fmt: skip


# ----------------------------------------------------------------------------- the refusal


def test_an_unresolved_torn_line_in_any_log_refuses_every_campaign(
    tmp_path, monkeypatch, capsys
) -> None:
    """A torn line in another campaign's log: nothing starts, nothing is settled, no recovery
    round opens (in a dry run, under ``only``, with ``recovery``); nothing is written; the
    script prints each line (log, offset, length, kind) and exits with ``TORN_JOURNAL``.
    Resolved, the campaign proceeds."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    open_id = f"x-{runs[1].config_hash[:8]}-a1"
    make_run_dir(root, runs[1], open_id)
    log_start(root, name, runs[1], open_id)  # a start without its end: it would be settled
    for n in (1, 2, 3):  # sgd-0.1/s0 used up its attempts: a recovery round would open
        make_run_dir(root, runs[3], f"y-{runs[3].config_hash[:8]}-a{n}", "failed")
    torn = tear(root / "launch" / "elsewhere.jsonl", {"event": "held", "run": "z/s0"})
    executor = Executor(root, {})
    for kwargs in ({}, {"dry_run": True}, {"recovery": "the disk is replaced"},
                   {"only": runs[1].label, "max_attempts": 1}):  # fmt: skip
        (line,) = refused(name, runs, root, executor, **kwargs)
        assert (line.log, line.offset, line.length, line.kind) == (
            "elsewhere.jsonl", 0, len(torn), "held")  # fmt: skip
    module = script()
    real = module.run_campaign
    monkeypatch.setattr(module, "run_campaign",
                        lambda *a, **k: real(*a, execute=executor, **k))  # fmt: skip
    monkeypatch.setenv("RUNS_DIR", str(root))
    before = files(root)
    with pytest.raises(SystemExit) as stop:
        module.main([str(campaign(tmp_path)), "--recovery", "the disk is replaced"])
    assert stop.value.code == TORN_JOURNAL and files(root) == before and executor.calls == []
    err = capsys.readouterr().err
    assert f"elsewhere.jsonl line 1: offset 0, length {len(torn)}, kind held" in err
    assert line.sha256 in err and "--resolve-torn" in err
    resolve(monkeypatch, root, line)
    result = run_campaign(name, runs, root, execute=executor, recovery="the disk is replaced")
    assert result["skipped"] == [runs[1].label] and len(events(root, "recovery")) == 1
    assert {r["run_id"] for r in events(root, "end") if r.get("settled")} == {open_id}


def test_the_exit_code_is_the_launchers_own() -> None:
    """No caller uses it: the launcher's other exits (0, 1, argparse's 2) and the qualification
    driver's codes, which it passes on when the launcher stops it."""
    from test_qualify import load_script

    qualify = load_script()
    used = {0, 1, 2, qualify.NO, qualify.INTERRUPTED, qualify.TECHNICAL_CASE, qualify.STALE,
            qualify.HELD}  # fmt: skip
    assert TORN_JOURNAL not in used and 0 < TORN_JOURNAL < 126
    assert f"{TORN_JOURNAL}" in launch.__doc__ and f"{TORN_JOURNAL}" in script().__doc__


# ------------------------------------------------------------------- what a lost line costs


def test_a_lost_end_is_settled_from_the_run_directory_once_resolved(tmp_path, monkeypatch):
    """``end``: the start stands without its end; once the torn line is resolved, the launcher
    settles it from the run directory as it settles any start without its end."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    run_id = f"x-{run.config_hash[:8]}-a1"
    make_run_dir(root, run, run_id)  # completed, and it qualifies
    log_start(root, name, run, run_id)
    tear(root / "launch" / f"{name}.jsonl",
         {"event": "end", "run": run.label, "attempt": 1, "run_id": run_id, "returncode": 0,
          "outcome": "done", "problem": None})  # fmt: skip
    executor = Executor(root, {})
    (line,) = refused(name, [run], root, executor)
    assert line.kind == "end"
    resolve(monkeypatch, root, line)
    result = run_campaign(name, [run], root, execute=executor)
    assert result["skipped"] == [run.label] and executor.calls == []
    (end,) = events(root, "end")
    assert (end["run_id"], end["attempt"], end["outcome"], end["settled"]) == (
        run_id, 1, "done", True)  # fmt: skip


@pytest.mark.parametrize("directory", ["failed", "none"])
def test_a_lost_start_leaves_its_run_id_unused(tmp_path, monkeypatch, directory) -> None:
    """``start``: a run directory may stand with no start (study 3's analysis lists it and
    blocks). The launcher counts the directory as an attempt, as it counts any directory of the
    config, and never reuses the lost start's run id."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    lost = launch._run_id("20260101T000000Z", run, 1)
    if directory != "none":
        make_run_dir(root, run, lost, directory)
    tear(root / "launch" / f"{name}.jsonl",
         {"event": "start", "run": run.label, "attempt": 1, "run_id": lost,
          "experiment": run.experiment, "config_hash": run.config_hash,
          "command": run.command()[1:]})  # fmt: skip
    executor = Executor(root, {})
    (line,) = refused(name, [run], root, executor)
    assert line.kind == "start"
    resolve(monkeypatch, root, line)
    result = run_campaign(name, [run], root, execute=executor)
    assert result["done"] == [run.label]
    ((_, new),) = executor.calls
    assert new != lost and new.endswith("-a2" if directory != "none" else "-a1")
    assert [r["run_id"] for r in events(root, "start")] == [new]


def test_a_lost_hold_is_written_again_first_and_its_run_never_reruns(tmp_path, monkeypatch):
    """``held``: its writer runs again before the resolution; ``hold`` writes only the missing
    line. While the torn line is unresolved the launcher starts nothing (the technical run
    would otherwise be attempted again); resolved, the label is held."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    held = f"x-{run.config_hash[:8]}-a1"
    make_run_dir(root, run, held, "failed")
    log_start(root, name, run, held)
    reason, record = "its records are unreadable", "c1/supersede-1.json"
    log = root / "launch" / f"{name}.jsonl"
    tear(log, {"event": "held", "run": run.label, "run_id": held, "experiment": run.experiment,
               "config_hash": run.config_hash, "reason": reason, "record": record})  # fmt: skip
    executor = Executor(root, {})
    (line,) = refused(name, [run], root, executor)
    assert line.kind == "held"
    assert hold(root, name, run, held, reason=reason, record=record) is True
    assert hold(root, name, run, held, reason=reason, record=record) is False
    refused(name, [run], root, executor)  # repaired, still unresolved
    resolve(monkeypatch, root, line)
    for kwargs in ({}, {"recovery": "a new disk"}):
        with pytest.raises(CampaignError, match=f"{held}: held, its outcome unresolved"):
            run_campaign(name, [run], root, execute=executor, **kwargs)
    assert executor.calls == [] and len(events(root, "held")) == 1


def test_a_lost_supersession_is_written_again_and_its_run_never_counts(tmp_path, monkeypatch):
    """``superseded``: its writer runs again; ``supersede`` writes only the missing line. The
    run, completed and qualifying on disk, is never counted done: while the torn line is
    unresolved nothing is skipped or started; resolved, it is a used attempt and the recovery
    round starts a new one."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    old = f"x-{run.config_hash[:8]}-a1"
    make_run_dir(root, run, old)  # completed, and it would qualify
    log_start(root, name, run, old, only=run.label, max_attempts=1)
    reason, record = "the Pod stopped", "c1/restart-1.json"
    log = root / "launch" / f"{name}.jsonl"
    tear(log, {"event": "superseded", "run": run.label, "run_id": old,
               "experiment": run.experiment, "config_hash": run.config_hash, "reason": reason,
               "record": record})  # fmt: skip
    executor, only = Executor(root, {}), {"only": run.label, "max_attempts": 1}
    (line,) = refused(name, runs, root, executor, **only)
    assert line.kind == "superseded"
    assert supersede(root, name, run, old, reason=reason, record=record) is True
    assert supersede(root, name, run, old, reason=reason, record=record) is False
    resolve(monkeypatch, root, line)
    result = run_campaign(name, runs, root, execute=executor, **only)
    assert result["exhausted"] == [run.label] and executor.calls == []
    result = run_campaign(name, runs, root, execute=executor, recovery=reason, incident=INCIDENT,
                          **only)  # fmt: skip
    assert result["done"] == [run.label] and len(executor.calls) == 1
    assert len(events(root, "superseded")) == 1


def test_a_lost_recovery_is_recorded_again_by_the_same_command(tmp_path, monkeypatch) -> None:
    """``recovery``: once resolved, the operator relaunches with the same command; the round
    opens and its reason is recorded again."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    cell = [r for r in runs if r.cell == "adamw-5e-5"]
    broken = Executor(root, {r.label: ["failed"] * 3 for r in cell})
    assert run_campaign(name, cell, root, execute=broken)["exhausted"] == [r.label for r in cell]
    reason = "disk quota raised"
    tear(root / "launch" / f"{name}.jsonl",
         {"event": "recovery", "experiment": cell[0].experiment, "cell": "adamw-5e-5",
          "reason": reason, "runs": [r.label for r in cell]})  # fmt: skip
    executor = Executor(root, {})
    (line,) = refused(name, cell, root, executor, recovery=reason)
    assert line.kind == "recovery"
    resolve(monkeypatch, root, line)
    result = run_campaign(name, cell, root, execute=executor, recovery=reason)
    assert result["done"] == [r.label for r in cell] and len(executor.calls) == 2
    assert [(r["cell"], r["reason"]) for r in events(root, "recovery")] == [("adamw-5e-5", reason)]


def test_a_lost_further_round_is_recorded_again_with_the_same_reason(tmp_path, monkeypatch):
    """A run's round 2 torn in its append (a crash before its attempt started): the launcher
    refuses until it is resolved; then the same command and reason write round 2 again, numbered
    2, after the same attempts: a torn line is no record, so its reason was given by no earlier
    round. Its attempt runs, and the label's attempts stay numbered 1, 2, 3."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    execute = Executor(root, {run.label: ["failed", "failed", "completed"]})
    run_campaign(name, runs, root, execute=execute, only=run.label, max_attempts=1)
    first = {"only": run.label, "max_attempts": 1, "incident": INCIDENT}
    run_campaign(name, runs, root, execute=execute, recovery=R1, **first)
    ids = [run_id for _, run_id in execute.calls]
    reason = "attempt 2: the Pod stopped again; its container disk was lost"
    round_two = {"event": "recovery", "experiment": run.experiment, "cell": run.cell,
                 "reason": reason, "runs": [run.label], "only": run.label, "round": 2,
                 "config_hash": run.config_hash, "after": ids, "max_attempts": 1,
                 "granted": 1, **INCIDENT}  # fmt: skip
    tear(root / "launch" / f"{name}.jsonl", round_two)
    executor = Executor(root, {run.label: ["completed"]})
    (line,) = refused(name, runs, root, executor, recovery=reason, **first)
    assert line.kind == "recovery"
    resolve(monkeypatch, root, line)
    result = run_campaign(name, runs, root, execute=executor, recovery=reason, **first)
    assert result["done"] == [run.label] and len(executor.calls) == 1
    rounds = events(root, "recovery")
    assert [(r["round"], r["reason"], r["after"]) for r in rounds] == [(1, R1, ids[:1]),
                                                                      (2, reason, ids)]  # fmt: skip
    assert [r["attempt"] for r in events(root, "start")] == [1, 2, 3]


def test_a_lost_stop_is_carried_on_by_the_same_command(tmp_path, monkeypatch) -> None:
    """``stop``: once resolved, the same command carries the campaign on."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    execute = Executor(root, {"theta0/s0": ["failed", "completed"]})
    run_campaign(name, runs, root, execute=execute, stop=lambda: len(execute.calls) >= 1)
    log = root / "launch" / f"{name}.jsonl"
    *kept, last = log.read_bytes().splitlines(keepends=True)
    assert json.loads(last)["event"] == "stop"
    log.write_bytes(b"".join(kept) + last[:-13])  # the stop record cut short
    executor = Executor(root, {})
    (line,) = refused(name, runs, root, executor)
    assert line.kind == "stop"
    resolve(monkeypatch, root, line)
    result = run_campaign(name, runs, root, execute=executor)
    assert result["done"] == [r.label for r in runs] and len(executor.calls) == 4
    assert [r["attempt"] for r in events(root, "start")] == [1, 2, 1, 1, 1]


def test_a_line_of_unknown_kind_is_resolved_after_reading_the_run_directories(
    tmp_path, monkeypatch, capsys
) -> None:
    """Unknown kind (null): the refusal names it so; once resolved, the launcher acts as the
    run directories and the readable records say."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    log = root / "launch" / f"{name}.jsonl"
    log.parent.mkdir(parents=True)
    log.write_bytes(b'{"time": "2026-10-27T10:00:00+00:00", "ev')
    executor = Executor(root, {})
    (line,) = refused(name, runs, root, executor)
    assert line.kind is None
    with pytest.raises(TornJournalError, match="kind null"):
        run_campaign(name, runs, root, execute=executor)
    resolve(monkeypatch, root, line, reason="no run directory of the campaign's labels")
    result = run_campaign(name, runs, root, execute=executor)
    assert result["done"] == [r.label for r in runs]


# --------------------------------------------------------------------------- the resolution


def test_the_resolution_refuses_what_names_no_unresolved_torn_line(
    tmp_path, monkeypatch, capsys
) -> None:
    """``--resolve-torn LOG --offset N --sha256 HEX --reason TEXT`` without a campaign file: it
    refuses an offset or sha256 that names no torn line of that log, another log, a line
    already resolved and an empty reason, and writes nothing then. It appends exactly the
    ``torn_resolved`` record, through ``_log``."""
    root = tmp_path / "runs"
    log = root / "launch" / "c.jsonl"
    launch._log(log, {"event": "start"})
    tear(log, {"event": "end", "run": "a/s0"})
    launch._log(root / "launch" / "d.jsonl", {"event": "start"})
    (line,) = torn_lines(root)
    monkeypatch.setenv("RUNS_DIR", str(root))
    main = script().main
    good = ["--resolve-torn", "c.jsonl", "--offset", str(line.offset), "--sha256", line.sha256]
    wrong = {
        "offset": ["--resolve-torn", "c.jsonl", "--offset", str(line.offset + 1), "--sha256",
                   line.sha256, "--reason", "r"],
        "sha256": ["--resolve-torn", "c.jsonl", "--offset", str(line.offset), "--sha256",
                   "0" * 64, "--reason", "r"],
        "log": ["--resolve-torn", "d.jsonl", "--offset", str(line.offset), "--sha256",
                line.sha256, "--reason", "r"],
        "missing log": ["--resolve-torn", "e.jsonl", "--offset", str(line.offset), "--sha256",
                        line.sha256, "--reason", "r"],
        "a path": ["--resolve-torn", str(log), "--offset", str(line.offset), "--sha256",
                   line.sha256, "--reason", "r"],
        "empty reason": [*good, "--reason", ""],
        "blank reason": [*good, "--reason", "  "],
    }  # fmt: skip
    before = files(root)
    for why, argv in wrong.items():
        with pytest.raises(SystemExit) as stop:
            main(argv)
        assert stop.value.code not in (0, None), why
        assert files(root) == before, why
    main([*good, "--reason", "the end is settled from its run directory"])
    (resolved,) = events(root, "torn_resolved")
    assert resolved == {"time": resolved["time"], "event": "torn_resolved", "log": "c.jsonl",
                        "offset": line.offset, "sha256": line.sha256,
                        "reason": "the end is settled from its run directory"}  # fmt: skip
    assert json.loads(capsys.readouterr().out) == resolved
    after = files(root)
    with pytest.raises(SystemExit, match="already resolved"):
        main([*good, "--reason", "again"])
    assert files(root) == after
    for argv in ([str(tmp_path / "c.yaml"), *good, "--reason", "r"], good[:4],
                 [*good[:2], "--reason", "r"], [*good, "--reason", "r", "--dry-run"],
                 []):  # fmt: skip
        with pytest.raises(SystemExit) as stop:
            main(argv)
        assert stop.value.code == 2  # the command line is refused
    assert files(root) == after


# ------------------------------------------------------------------------- the writers again


def test_hold_and_supersede_write_a_line_only_when_no_identical_one_exists(tmp_path) -> None:
    """Identical: the same event, label, experiment, run id and reason, in any log."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    for write, event in ((hold, "held"), (supersede, "superseded")):
        assert write(root, name, run, "r1", reason="why", record="a.json") is True
        assert write(root, name, run, "r1", reason="why", record="b.json") is False
        assert write(root, "another", run, "r1", reason="why", record="a.json") is False
        assert write(root, name, run, "r1", reason="other", record="a.json") is True
        assert write(root, name, run, "r2", reason="why", record="a.json") is True
        assert write(root, name, runs[2], "r1", reason="why", record="a.json") is True
        assert len(events(root, event)) == 4
    assert not (root / "launch" / "another.jsonl").exists()


# ------------------------------------------------- a resolution names its line with both records

AFTER = json.dumps({"event": "stop", "run": "a/s0"}).encode() + b"\n"


@pytest.mark.parametrize(
    ("line", "kind"),
    [(PREFIX + b'"event": "end", "run": "' + MARKER.encode() + PREFIX + b'"event": "stop"}', "end"),
     (b"", None),
     (b"[1, 2, " + MARKER.encode().hex().encode() + b"]", None),
     (PREFIX + b'"event": "held", "run": "\xff\xfe' + MARKER.encode(), "held")],
    ids=["merged", "blank", "not-an-object", "invalid-utf8"],
)  # fmt: skip
def test_a_resolution_writes_the_missing_torn_record_first(
    tmp_path, monkeypatch, capsys, line, kind
) -> None:
    """A non-final unparseable line that no ``torn`` record names (a merged line of an older
    writer, a blank line, a power loss inside the repair): the resolution writes its ``torn``
    record (offset, length, sha256, the kind through the strict pattern alone), then the
    ``torn_resolved`` record, both through ``_log``; nothing else of its text is copied."""
    root = tmp_path / "runs"
    log = root / "launch" / "c.jsonl"
    log.parent.mkdir(parents=True)
    before = json.dumps({"event": "start", "run": "a/s0"}).encode() + b"\n"
    original = before + line + b"\n" + AFTER
    log.write_bytes(original)
    (found,) = torn_lines(root)
    assert (found.offset, found.length, found.kind, found.repaired, found.resolved) == (
        len(before), len(line), kind, False, False)  # fmt: skip
    resolve(monkeypatch, root, found)
    data = log.read_bytes()
    assert data.startswith(original)
    named, resolved = (json.loads(x) for x in data[len(original) :].splitlines())
    assert named == {"time": named["time"], "event": "torn", "offset": len(before),
                     "length": len(line), "sha256": hashlib.sha256(line).hexdigest(),
                     "kind": kind}  # fmt: skip
    assert (resolved["event"], resolved["offset"], resolved["sha256"]) == (
        "torn_resolved", len(before), hashlib.sha256(line).hexdigest())  # fmt: skip
    (after,) = torn_lines(root)
    assert after.repaired and after.resolved
    printed = capsys.readouterr()
    for text in (data[len(original) :].decode(), printed.out, printed.err):
        assert MARKER not in text and MARKER.encode().hex() not in text


def test_a_log_of_one_newline_holds_a_blank_line_resolved_like_any_other(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "runs"
    log = root / "launch" / "c.jsonl"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"\n")
    (line,) = torn_lines(root)
    assert (line.offset, line.length, line.kind, line.repaired) == (0, 0, None, False)
    name, runs = load_campaign(campaign(tmp_path))
    refused(name, runs[1:2], root, Executor(root, {}))
    resolve(monkeypatch, root, line, reason="a blank line: nothing was lost")
    assert [(r["event"], r["offset"], r.get("length")) for r in launch_records(root)] == [
        ("torn", 0, 0), ("torn_resolved", 0, None)]  # fmt: skip
    assert run_campaign(name, runs[1:2], root, execute=Executor(root, {}))["done"]


def test_a_torn_last_line_is_named_once_by_its_resolution(tmp_path, monkeypatch) -> None:
    """The torn last line has no record yet: the resolution's append repairs it, and no second
    ``torn`` record is written."""
    root = tmp_path / "runs"
    log = root / "launch" / "c.jsonl"
    torn = tear(log, {"event": "end", "run": "a/s0"})
    (line,) = torn_lines(root)
    resolve(monkeypatch, root, line)
    assert [(r["event"], r["offset"]) for r in launch_records(root)] == [
        ("torn", 0), ("torn_resolved", 0)]  # fmt: skip
    assert events(root, "torn")[0]["length"] == len(torn)


# ------------------------------------------------------------ the two readers agree on every byte


def test_an_invalid_utf8_torn_line_is_listed_resolved_and_read_past(tmp_path, monkeypatch) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    log = root / "launch" / f"{name}.jsonl"
    log.parent.mkdir(parents=True)
    bad = PREFIX + b'"event": "end", "run": "\xc3\x28\xff'
    log.write_bytes(bad + b"\n" + AFTER)
    (line,) = torn_lines(root)
    assert (line.offset, line.length, line.kind) == (0, len(bad), "end")
    refused(name, runs[1:2], root, Executor(root, {}))
    resolve(monkeypatch, root, line)
    assert [r["event"] for r in launch_records(root)] == ["stop", "torn", "torn_resolved"]
    assert run_campaign(name, runs[1:2], root, execute=Executor(root, {}))["done"]


def test_a_bom_prefixed_line_is_unparseable_in_both_readers(tmp_path) -> None:
    root = tmp_path / "runs"
    log = root / "launch" / "c.jsonl"
    log.parent.mkdir(parents=True)
    bom = b"\xef\xbb\xbf" + json.dumps({"event": "start", "run": "a/s0"}).encode()
    log.write_bytes(bom + b"\n" + AFTER)
    (line,) = torn_lines(root)
    assert (line.offset, line.length, line.sha256, line.kind) == (
        0, len(bom), hashlib.sha256(bom).hexdigest(), "start")  # fmt: skip
    assert [r["event"] for r in launch_records(root)] == ["stop"]


def test_a_record_separator_does_not_split_a_line(tmp_path) -> None:
    """``\\x1e`` (or another Unicode line separator) before a valid record: one torn line, and no
    record is read out of it."""
    root = tmp_path / "runs"
    log = root / "launch" / "c.jsonl"
    log.parent.mkdir(parents=True)
    joined = b"\x1e" + json.dumps({"event": "held", "run": "a/s0", "run_id": "r1"}).encode()
    log.write_bytes(joined + b"\n" + AFTER)
    assert [(x.offset, x.length) for x in torn_lines(root)] == [(0, len(joined))]
    assert [r["event"] for r in launch_records(root)] == ["stop"]


def test_a_multibyte_torn_line_is_named_by_its_bytes(tmp_path) -> None:
    root = tmp_path / "runs"
    log = root / "launch" / "c.jsonl"
    log.parent.mkdir(parents=True)
    first = json.dumps({"event": "start", "note": "θ₀ and v₀"}, ensure_ascii=False).encode()
    torn = PREFIX + '"event": "end", "note": "θ₀ → '.encode()
    log.write_bytes(first + b"\n" + torn)
    (line,) = torn_lines(root)
    assert (line.offset, line.length, line.sha256) == (
        len(first) + 1, len(torn), hashlib.sha256(torn).hexdigest())  # fmt: skip
    assert [r["note"] for r in launch_records(root)] == ["θ₀ and v₀"]
    launch._log(log, {"event": "stop"})
    (named,) = events(root, "torn")
    assert (named["offset"], named["length"]) == (len(first) + 1, len(torn))


# --------------------------------------------------------------------- one writer at a time

WRITER = """
import sys, time
from pathlib import Path
from zero_field_refusal import launch

real = launch._torn_tail


def slow(path):  # widens the window between reading the tail and repairing it
    found = real(path)
    if found is not None:
        time.sleep(0.5)
    return found


launch._torn_tail = slow
log, who, count, go = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
(go.parent / f"ready-{who}").touch()
while not go.exists():
    time.sleep(0.005)
for i in range(count):
    launch._log(log, {"event": "stop", "run": f"{who}/{i}"})
"""


def test_two_writers_take_turns_on_one_log(tmp_path) -> None:
    """Two processes append to one log that ends in a torn line: one ``torn`` record, no blank
    line, every record."""
    import subprocess
    import sys
    import time

    log = tmp_path / "launch" / "c.jsonl"
    torn = tear(log, {"event": "end", "run": "a/s0"})
    go, count = tmp_path / "go", 200
    writers = [subprocess.Popen([sys.executable, "-c", WRITER, str(log), who, str(count), str(go)],
                                cwd=ROOT) for who in ("a", "b")]  # fmt: skip
    deadline = time.monotonic() + 60
    while not all((tmp_path / f"ready-{who}").exists() for who in ("a", "b")):
        assert time.monotonic() < deadline and all(w.poll() is None for w in writers)
        time.sleep(0.01)
    go.touch()  # both are past their imports: they read the torn tail together
    assert [w.wait(timeout=120) for w in writers] == [0, 0]
    data = log.read_bytes()
    assert data.startswith(torn + b"\n")
    lines = data[len(torn) + 1 :].split(b"\n")
    assert lines[-1] == b"" and b"" not in lines[:-1]
    records = [json.loads(x) for x in lines[:-1]]
    assert [r["event"] for r in records].count("torn") == 1
    runs = sorted(r["run"] for r in records if r["event"] == "stop")
    assert runs == sorted(f"{who}/{i}" for who in ("a", "b") for i in range(count))
    assert [x.repaired for x in torn_lines(tmp_path)] == [True]


# ------------------------------------------- a line a published record requires, and missing

LABELLED = {"experiment": "smoke"}


def cycle_record(root: Path, launchers, held, name: str = "supersede-1.json") -> Path:
    """A published supersession record, as ``qualify.py supersede`` writes its entries."""
    path = root / "qualification" / "c1" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"candidate": 1, "diagnosis": MARKER, "fix": MARKER,
                                "launchers": launchers, "held": held}))  # fmt: skip
    return path


def restart_record(root: Path, run_id: str | None, number: int = 1) -> Path:
    """A published case restart record, as ``qualify.py restart-case`` writes it."""
    path = root / "qualification" / "c1" / "numerical" / "s0" / "adamw-5e-05"
    path = path / f"restart-{number}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    launcher = None if run_id is None else {"run_id": run_id, "identity": {"note": MARKER}}
    path.write_text(json.dumps({"restart": number, "reason": MARKER, "launcher": launcher}))
    return path


def entry(run, run_id: str, config_hash: str | None = None, **extra) -> dict:
    return {"run_id": run_id, "campaign": "test-campaign", "run": run.label,
            "experiment": run.experiment, "config_hash": config_hash or run.config_hash,
            **extra}  # fmt: skip


def named(error) -> set[tuple]:
    return {(m.record, m.event, m.run_id) for m in error.missing}


def test_a_published_records_missing_lines_refuse_the_launcher(
    tmp_path, monkeypatch, capsys
) -> None:
    """A cycle record and a case restart record published, none of their lines appended (a
    crash between the record and its lines, or an append that wrote no byte): no torn bytes,
    and still the launcher starts nothing, settles nothing and opens no recovery round, in a dry
    run too, and exits 12 naming each missing line by its record's path, its event and its run
    id. The held label's old run is under another config hash: without the refusal its label
    would get a new attempt. Restored by the repair's writers, the launcher goes on, and the
    held label is refused as held."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    old = "0" * 12  # the held run's config hash, before a fix changed a hashed key
    cycle_record(root, [entry(runs[1], "r-l1")],
                 [entry(runs[2], "r-h1", old, reason="no recorded commit")])  # fmt: skip
    log_start(root, name, runs[3], "r-s1")  # the restart's launcher run, by its start
    restart_record(root, "r-s1")
    executor = Executor(root, {})
    cycle = "qualification/c1/supersede-1.json"
    restart = "qualification/c1/numerical/s0/adamw-5e-05/restart-1.json"
    for kwargs in ({}, {"dry_run": True}, {"recovery": "fixed"},
                   {"only": runs[2].label, "max_attempts": 1}):  # fmt: skip
        before = files(root)
        with pytest.raises(TornJournalError) as refusal:
            run_campaign(name, runs, root, execute=executor, **kwargs)
        assert executor.calls == [] and files(root) == before
        assert refusal.value.lines == []
        assert named(refusal.value) == {(cycle, "superseded", "r-l1"), (cycle, "held", "r-h1"),
                                        (restart, "superseded", "r-s1")}  # fmt: skip
    module = script()
    real = module.run_campaign
    monkeypatch.setattr(module, "run_campaign",
                        lambda *a, **k: real(*a, execute=executor, **k))  # fmt: skip
    monkeypatch.setenv("RUNS_DIR", str(root))
    capsys.readouterr()
    with pytest.raises(SystemExit) as stop:
        module.main([str(campaign(tmp_path))])
    assert stop.value.code == TORN_JOURNAL and executor.calls == []
    err = capsys.readouterr().err
    for line in (f"{cycle}: no superseded line of r-l1", f"{cycle}: no held line of r-h1",
                 f"{restart}: no superseded line of r-s1"):  # fmt: skip
        assert line in err
    assert MARKER not in err
    supersede(root, name, runs[1], "r-l1", reason="a cycle", record=cycle)
    held = type(runs[2])(**{**runs[2].__dict__, "config_hash": old})
    hold(root, name, held, "r-h1", reason="no recorded commit", record=cycle)
    supersede(root, name, runs[3], "r-s1", reason="a restart", record=restart)
    assert launch.missing_lines(root) == []
    with pytest.raises(CampaignError, match="r-h1: held, its outcome unresolved"):
        run_campaign(name, [runs[2]], root, execute=executor, only=runs[2].label, max_attempts=1)
    assert executor.calls == []
    assert run_campaign(name, [runs[0]], root, execute=executor)["done"] == [runs[0].label]


def test_only_the_missing_lines_of_a_record_are_named(tmp_path) -> None:
    """A record with some of its lines: only the others are named. A line is matched by its
    event, label, experiment and run id, whatever its config hash: one under another hash
    counts, one of another label does not."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    cycle_record(root, [entry(runs[1], "r-1"), entry(runs[2], "r-2"), entry(runs[3], "r-3")], [])
    other = type(runs[1])(**{**runs[1].__dict__, "config_hash": "0" * 12})
    supersede(root, name, other, "r-1", reason="r", record="c")  # another config hash: counts
    supersede(root, name, runs[1], "r-2", reason="r", record="c")  # another label: does not
    assert [(m.event, m.run_id) for m in launch.missing_lines(root)] == [
        ("superseded", "r-2"), ("superseded", "r-3")]  # fmt: skip


@pytest.mark.parametrize("which", ["cycle", "restart"])
def test_an_unreadable_published_record_refuses_by_its_path(tmp_path, which) -> None:
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    path = cycle_record(root, [], []) if which == "cycle" else restart_record(root, "r-1")
    path.write_bytes(b"{" + MARKER.encode())
    executor = Executor(root, {})
    with pytest.raises(TornJournalError) as refusal:
        run_campaign(name, runs[1:2], root, execute=executor)
    ((record, event, run_id),) = named(refusal.value)
    assert (record, event, run_id) == (path.relative_to(root).as_posix(), None, None)
    assert "cannot be read" in str(refusal.value) and MARKER not in str(refusal.value)
    assert executor.calls == []


def test_a_restarts_line_is_matched_by_its_runs_own_label(tmp_path) -> None:
    """A case restart record names its launcher run by run id alone: its label and experiment
    are its start's, or, with no start in the journal, its run directory's (its campaign label
    in meta.json, its experiment folder). A restart without a launcher run requires no line."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    restart_record(root, None, 1)
    restart_record(root, "r-dir", 2)
    run_dir = make_run_dir(root, runs[1], "r-dir")
    meta = json.loads((run_dir / "meta.json").read_text())
    meta["campaign"] = {"name": name, "label": runs[1].label, "attempt": 1}
    (run_dir / "meta.json").write_text(json.dumps(meta))
    assert [(m.event, m.run_id) for m in launch.missing_lines(root)] == [("superseded", "r-dir")]
    supersede(root, name, runs[2], "r-dir", reason="r", record="c")  # another label
    assert len(launch.missing_lines(root)) == 1
    supersede(root, name, runs[1], "r-dir", reason="r", record="c")
    assert launch.missing_lines(root) == []


# ------------------------------------------------------- every append makes its entries durable


def _fsyncs(monkeypatch, log: Path) -> list[tuple[str, bool]]:
    """Every ``os.fsync`` as (the path its descriptor names, whether the log's lock was held)."""
    import fcntl

    seen = []
    real = os.fsync

    def spy(fd):
        path = os.readlink(f"/proc/self/fd/{fd}")
        probe = os.open(log, os.O_RDONLY)
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = False
            fcntl.flock(probe, fcntl.LOCK_UN)
        except BlockingIOError:
            locked = True
        finally:
            os.close(probe)
        seen.append((path, locked))
        return real(fd)

    monkeypatch.setattr(launch.os, "fsync", spy)
    return seen


@pytest.mark.parametrize("before", ["none", "created", "appended"])
def test_every_append_fsyncs_the_log_its_folder_and_the_runs_root(
    tmp_path, monkeypatch, before
) -> None:
    """The log, ``launch/`` and the runs root are fsynced on every append, under the lock: the
    first, a later one, and one after an interrupted creation (the log there, its folder never
    fsynced)."""
    root = tmp_path / "runs"
    log = root / "launch" / "c.jsonl"
    if before != "none":
        log.parent.mkdir(parents=True)
        log.write_bytes(b"" if before == "created" else json.dumps({"event": "x"}).encode() + b"\n")
    seen = _fsyncs(monkeypatch, log)
    launch._log(log, {"event": "stop"})
    assert {path for path, _ in seen} >= {str(log), str(log.parent), str(root)}
    assert all(locked for path, locked in seen if path in (str(log), str(log.parent), str(root)))


def test_the_resolution_goes_through_the_same_writer(tmp_path, monkeypatch) -> None:
    root = tmp_path / "runs"
    log = root / "launch" / "c.jsonl"
    tear(log, {"event": "end", "run": "a/s0"})
    (line,) = torn_lines(root)
    seen = _fsyncs(monkeypatch, log)
    launch.resolve_torn(root, line.log, offset=line.offset, sha256=line.sha256, reason="r")
    assert {path for path, _ in seen} >= {str(log), str(log.parent), str(root)}
    assert all(locked for _, locked in seen)


# ------------------------------------------------------------ a malformed journal value


MALFORMED = [
    {"event": "stop", "run": ["x"]},
    {"event": ["superseded"], "run": "adamw-5e-5/s0", "experiment": "smoke", "run_id": "r-1"},
    {"event": "superseded", "run": "adamw-5e-5/s0", "experiment": {"x": 1}, "run_id": "r-1"},
    {"event": "superseded", "run": "adamw-5e-5/s0", "experiment": "smoke", "run_id": ["r-1"]},
    {"event": "start", "run": ["adamw-5e-5/s1"], "experiment": "smoke", "run_id": "r-s1"},
    {"event": "torn_resolved", "log": ["c.jsonl"], "offset": 0, "sha256": "0" * 64},
    {"event": "torn", "offset": 0, "length": 1, "sha256": ["0" * 64]},
    {"event": "end", "run": "adamw-5e-5/s1", "run_id": {"r": 1}, "outcome": "done"},
    {"event": "recovery", "experiment": "smoke", "cell": ["adamw-5e-5"], "reason": "r"},
    {"event": "recovery", "experiment": "smoke", "cell": "adamw-5e-5", "only": ["x"]},
]


@pytest.mark.parametrize("line", MALFORMED, ids=range(len(MALFORMED)))
def test_a_journal_object_with_a_non_string_identity_is_a_torn_line(
    tmp_path, monkeypatch, line
) -> None:
    """A valid JSON object whose identity field is a list, an object or a number is a torn line:
    listed (its kind through the strict pattern), never a record, and the launcher refuses with
    12, never a traceback. Resolved, the launcher goes on; a published record whose line is
    otherwise missing still blocks with 12, naming that line."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    log = root / "launch" / "c.jsonl"
    log.parent.mkdir(parents=True)
    data = json.dumps(line).encode()
    log.write_bytes(data + b"\n")
    (torn,) = torn_lines(root)
    assert (torn.log, torn.line, torn.offset, torn.length) == ("c.jsonl", 1, 0, len(data))
    assert torn.kind == launch.torn_kind(data) and launch_records(root) == []
    executor = Executor(root, {})
    refused("test-campaign", runs[2:3], root, executor)
    resolve(monkeypatch, root, torn)
    assert run_campaign(name, runs[2:3], root, execute=executor)["done"]
    cycle_record(root, [entry(runs[1], "r-1")], [])
    log_start(root, name, runs[2], "r-s1")
    restart_record(root, "r-s1")
    assert [(m.event, m.run_id) for m in launch.missing_lines(root)] == [
        ("superseded", "r-1"), ("superseded", "r-s1")]  # fmt: skip
    module = script()
    monkeypatch.setenv("RUNS_DIR", str(root))
    real = module.run_campaign
    monkeypatch.setattr(module, "run_campaign",
                        lambda *a, **k: real(*a, execute=Executor(root, {}), **k))  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        module.main([str(campaign(tmp_path))])
    assert stop.value.code == TORN_JOURNAL


def test_a_restarts_run_with_no_start_and_no_label_is_matched_by_its_run_id(tmp_path) -> None:
    """The last fallback: a restart record's run with no start in the journal and no campaign
    label in a run directory is matched by a ``superseded`` line of its run id, whatever its
    label."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    restart_record(root, "r-alone")
    assert [(m.event, m.run_id) for m in launch.missing_lines(root)] == [("superseded", "r-alone")]
    supersede(root, name, runs[1], "r-other", reason="r", record="c")  # another run
    assert len(launch.missing_lines(root)) == 1
    supersede(root, name, runs[3], "r-alone", reason="r", record="c")  # any label
    assert launch.missing_lines(root) == []


def test_a_malformed_start_is_a_torn_line_and_its_attempt_still_counts(
    tmp_path, monkeypatch, capsys
) -> None:
    """A hand-edited start (its run id a list) of an attempt that left its run directory: the
    launcher refuses with 12, naming the line by its log and line; resolved, the line is no
    record, and the launcher counts the attempt from its directory as it did before the edit:
    with one attempt allowed, it starts none."""
    name, runs = load_campaign(campaign(tmp_path))
    run, root = runs[1], tmp_path / "runs"
    run_id = f"x-{run.config_hash[:8]}-a1"
    make_run_dir(root, run, run_id, "failed")
    log_start(root, name, run, run_id, only=run.label, max_attempts=1)
    log = root / "launch" / f"{name}.jsonl"
    with log.open("a") as f:
        f.write(json.dumps({"event": "end", "run": run.label, "attempt": 1, "run_id": run_id,
                            "returncode": 1, "outcome": "technical", "problem": "status 'failed'",
                            "only": run.label}) + "\n")  # fmt: skip
    executor, only = Executor(root, {}), {"only": run.label, "max_attempts": 1}
    assert run_campaign(name, runs, root, execute=executor, **only)["exhausted"] == [run.label]
    start, *rest = log.read_text().splitlines()
    edited = json.dumps({**json.loads(start), "run_id": [run_id]})
    log.write_text("\n".join([edited, *rest]) + "\n")
    (line,) = torn_lines(root)
    assert (line.line, line.kind, line.repaired) == (1, "start", False)
    assert [r["event"] for r in launch_records(root)] == ["end"]
    refused(name, runs, root, executor, **only)
    module = script()
    monkeypatch.setenv("RUNS_DIR", str(root))
    capsys.readouterr()
    with pytest.raises(SystemExit) as stop:
        module.main([str(campaign(tmp_path)), "--only", run.label, "--max-attempts", "1"])
    assert stop.value.code == TORN_JOURNAL
    err = capsys.readouterr().err
    assert f"{name}.jsonl line 1: offset 0, length {line.length}, kind start" in err
    resolve(monkeypatch, root, line, reason="the attempt's run directory stands for its start")
    assert run_campaign(name, runs, root, execute=executor, **only)["exhausted"] == [run.label]
    assert executor.calls == []


def test_the_launchers_own_records_are_never_torn(tmp_path, monkeypatch) -> None:
    """Every record the launcher writes (start, end, stop, recovery, held, superseded, torn and
    torn_resolved) is a record, never a torn line; the writer refuses a record whose identity
    is not a string, and writes nothing."""
    name, runs = load_campaign(campaign(tmp_path))
    root = tmp_path / "runs"
    cell = [r for r in runs if r.cell == "adamw-5e-5"]
    broken = Executor(root, {r.label: ["failed"] * 3 for r in cell})
    run_campaign(name, cell, root, execute=broken)
    run_campaign(name, cell, root, execute=Executor(root, {}), recovery="fixed")
    execute = Executor(root, {"theta0/s0": ["failed"]})
    run_campaign(name, runs[:1], root, execute=execute, stop=lambda: len(execute.calls) >= 1)
    run_campaign(name, runs[3:], root, execute=Executor(root, {}), only=runs[3].label,
                 max_attempts=1)  # fmt: skip
    hold(root, name, runs[1], "r-h", reason="r", record="c")
    supersede(root, name, runs[2], "r-s", reason="r", record="c")
    log = root / "launch" / f"{name}.jsonl"
    tear(log, {"event": "stop", "run": "x/s0"})
    launch._log(log, {"event": "stop", "run": "x/s0", "reason": "stop requested"})
    (torn,) = torn_lines(root)
    resolve(monkeypatch, root, torn)
    events_written = {r["event"] for r in launch_records(root)}
    assert events_written == {"start", "end", "stop", "recovery", "held", "superseded", "torn",
                              "torn_resolved"}  # fmt: skip
    assert [(x.offset, x.resolved) for x in torn_lines(root)] == [(torn.offset, True)]
    before = log.read_bytes()
    for record in ({"event": "held", "run": ["x"]}, {"event": "stop", "run_id": None},
                   {"event": 3}):  # fmt: skip
        with pytest.raises(ValueError, match="must be a string"):
            launch._log(log, record)
    assert log.read_bytes() == before
