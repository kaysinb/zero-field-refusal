"""Study 3's engineering cycle outside the numerical qualification (analysis/study3_cycle.py).

docs/study3-preregistration.md: "Failure routes" (one engineering cycle per candidate: diagnosis,
a fix, re-qualification of the affected cases and reruns of the affected runs) and "Voiding and
supersession" (the affected records listed, the scientific failures kept). The claims a cycle
makes per stage, each run's GPU code and the journal prefix a cycle pins are tested on synthetic
launch logs and endpoints."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_study3_progress import ep

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import study3_cycle as cycle


def sha(data) -> str:
    """The sha256 of bytes, or of a file's bytes."""
    if not isinstance(data, bytes):
        data = Path(data).read_bytes()
    return hashlib.sha256(data).hexdigest()


def start(run_id: str, time: str, label: str = "A-1e-05/s10100") -> dict:
    """A study-3 start line of the launch journal."""
    return {"time": time, "event": "start", "run": label, "attempt": 1, "run_id": run_id,
            "experiment": "study3", "config_hash": "0" * 16}  # fmt: skip


def write_lines(path: Path, records) -> None:
    """A launch log holding ``records``, one line each, as the launcher writes them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def append_line(path: Path, record) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------ claims, GPU codes, prefix


def test_runs_of_a_dirty_or_unknown_commit_have_no_gpu_code(monkeypatch, tmp_path):
    monkeypatch.setattr(code_identity, "gpu_code_identity",
                        lambda c, *a, **k: {"sha256": {"a" * 40: "o" * 64}[c]})  # fmt: skip
    runs = [ep("A", 1e-5, 10300, step="validation", run_id=r)
            for r in ("clean", "dirty", "unknown", "none")]  # fmt: skip
    gits = [("a" * 40, False), ("a" * 40, True), ("b" * 40, False), None]
    runs = [replace(e, git=g) for e, g in zip(runs, gits, strict=True)]
    assert cycle.run_gpu_codes(runs, repo=tmp_path) == {"clean": "o" * 64, "dirty": None,
                                                        "unknown": None}  # fmt: skip


def test_a_runs_gpu_code_is_computed_once_per_commit(monkeypatch, tmp_path):
    calls = []

    def identity(commit, *args, **kwargs):
        calls.append(commit)
        return {"sha256": "o" * 64}

    monkeypatch.setattr(code_identity, "gpu_code_identity", identity)
    runs = [replace(ep("A", 1e-5, 10300 + n, step="validation", run_id=f"r{n}"),
                    git=("a" * 40, False)) for n in range(3)]  # fmt: skip
    runs.append(replace(ep("A", 1e-5, 10310, step="validation", run_id="no-id"),
                        git=("not a commit", False)))  # fmt: skip
    found = cycle.run_gpu_codes(runs, repo=tmp_path)
    assert found == {"r0": "o" * 64, "r1": "o" * 64, "r2": "o" * 64, "no-id": None}
    assert calls == ["a" * 40]


def test_claims_parse_and_refuse():
    claims = cycle.parse_claims(["search=reuse", "validation=rerun:F,P"],
                                ["search=the fix touches the evaluation only",
                                 "validation=its runs evaluate"])  # fmt: skip
    assert claims["validation"] == cycle.Claim("rerun", ("F", "P"), "its runs evaluate")
    assert claims["search"].arms == ()
    for bad in (["search=keep"], ["engineering=reuse"], ["validation=rerun:Q"],
                ["search=reuse", "search=rerun"],
                ["qualification=rerun:A"]):  # all or nothing per candidate  # fmt: skip
        with pytest.raises(cycle.CycleRefused):
            cycle.parse_claims(bad, ["search=x", "validation=x", "engineering=x"])
    with pytest.raises(cycle.CycleRefused, match="justification"):
        cycle.parse_claims(["search=reuse"], ["search=  "])


def test_a_rerun_takes_every_arm_by_default_and_a_justification_is_one_line():
    claims = cycle.parse_claims(["qualification=rerun", "validation=rerun:P,A"],
                                ["qualification=the kernel changed", "validation=x"])  # fmt: skip
    assert claims["qualification"] == cycle.Claim("rerun", ("A", "F", "P"), "the kernel changed")
    assert claims["validation"].arms == ("A", "P")
    for stages, justifications in ((["search=reuse"], []),
                                   (["search=reuse"], ["search=two\nlines"]),
                                   (["search=reuse"], ["search=" + "x" * 1001]),
                                   (["search=reuse:A"], ["search=x"]),
                                   (["validation=rerun:A,A"], ["validation=x"]),
                                   (["validation=rerun:"], ["validation=x"]),
                                   (["search=reuse"], ["search=x", "search=y"])):  # fmt: skip
        with pytest.raises(cycle.CycleRefused):
            cycle.parse_claims(stages, justifications)


def test_the_journal_prefix_pins_starts_by_line_not_by_time(tmp_path):
    log = tmp_path / "launch" / "study3-c1-search.jsonl"
    write_lines(log, [start("r1", time="2026-11-01T00:00:00+00:00"),
                      start("r2", time="2020-01-01T00:00:00+00:00")])  # fmt: skip
    seen = cycle.journal_seen(tmp_path)
    assert seen == [{"path": "launch/study3-c1-search.jsonl", "lines": 2,
                     "sha256": sha(log.read_bytes())}]  # fmt: skip
    append_line(log, start("r3", time="2020-01-01T00:00:00+00:00"))  # an old stamp after the record
    assert cycle.seen_starts(tmp_path, seen) == {"r1", "r2"}
    write_lines(log, [start("r1", time="2026-11-01T00:00:00+00:00"),
                      start("rX", time="2020-01-01T00:00:00+00:00")])  # fmt: skip
    with pytest.raises(cycle.CycleRefused, match="prefix"):  # same line count, other content
        cycle.seen_starts(tmp_path, seen)


def test_the_prefix_holds_whole_lines_and_a_log_that_is_gone_refuses(tmp_path):
    log = tmp_path / "launch" / "study3-c1-search.jsonl"
    write_lines(log, [start("r1", time="2026-11-01T00:00:00+00:00")])
    with log.open("a") as f:
        f.write('{"event": "start", "run_id": "half')  # a line the launcher is still writing
    other = tmp_path / "launch" / "a-first.jsonl"
    write_lines(other, [])
    seen = cycle.journal_seen(tmp_path)
    assert [entry["path"] for entry in seen] == ["launch/a-first.jsonl",
                                                 "launch/study3-c1-search.jsonl"]  # fmt: skip
    assert [entry["lines"] for entry in seen] == [0, 1]
    assert seen[1]["sha256"] == sha(log.read_bytes()[: log.read_bytes().index(b"\n") + 1])
    with log.open("a") as f:
        f.write('"}\n')  # the line completed
    assert cycle.seen_starts(tmp_path, seen) == {"r1"}
    log.unlink()
    with pytest.raises(cycle.CycleRefused, match="prefix"):
        cycle.seen_starts(tmp_path, seen)
