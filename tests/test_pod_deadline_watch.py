"""The deadline watcher: the file under the environment's cap, the atomic write, the loop."""

import importlib.util
import os
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pod_deadline_watch.py"
spec = importlib.util.spec_from_file_location("pod_deadline_watch", SCRIPT)
assert spec is not None and spec.loader is not None
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)

KEY = "secret-key-123"


def test_effective_deadline_rules() -> None:
    assert watch.effective_deadline(1000, "900", None) == (900, "file", None)
    assert watch.effective_deadline(1000, "900\n", 800) == (900, "file", None)
    for text in (None, "", "x"):
        assert watch.effective_deadline(1000, text, None) == (1000, "env", None)
    assert watch.effective_deadline(1000, "1200", None) == (1000, "env", "above cap")
    # A value at or before the watcher's start is stale: the cap applies
    assert watch.effective_deadline(1000, "500", None, started=500) == (1000, "env", "stale")
    assert watch.effective_deadline(1000, "500", 700, started=600) == (1000, "env", "stale")
    assert watch.effective_deadline(1000, "501", None, started=500) == (501, "file", None)


def test_only_ascii_digits_are_an_epoch() -> None:
    assert watch.parse_epoch(" 900\n") == 900
    for text in ("\u0663\u0660\u0660", "\uff19\uff10\uff10", "9\u00b2", "1_000", "+900", "-1",
                 "9.5", "", None):  # fmt: skip
        assert watch.parse_epoch(text) is None, text
    assert watch.effective_deadline(1000, "\uff19\uff10\uff10", 800) == (800, "last", None)


def test_malformed_then_missing_keeps_last_valid() -> None:
    assert watch.effective_deadline(1000, "x", 900) == (900, "last", None)
    assert watch.effective_deadline(1000, None, 900) == (900, "last", None)
    assert watch.effective_deadline(1000, "", 900) == (900, "last", None)


def test_set_needs_an_epoch_and_file_as_its_third_argument(tmp_path: Path) -> None:
    target = tmp_path / "zfr-deadline"
    for args in (["--set"], ["--set", "x"], ["--set", "\uff19\uff10"], ["--set", "950", "--file"],
                 ["--set", "950", "-f", str(target)], ["--set", "950", str(target), "--file"],
                 ["--sett", "950"], ["--set", "950", "--file", str(target), "x"]):  # fmt: skip
        with pytest.raises(SystemExit) as stop:
            watch.cli(args)
        assert stop.value.code == watch.USAGE, args
    assert not target.exists()
    watch.cli(["--set", "950", "--file", str(target)])
    assert target.read_text() == "950\n"


def test_write_deadline_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "zfr-deadline"
    target.write_text("100\n")
    seen = []
    real = os.replace

    def spy(src: str, dst: str) -> None:
        seen.append((Path(src).read_text(), Path(dst).read_text(), Path(src).parent))
        real(src, dst)

    monkeypatch.setattr(watch.os, "replace", spy)
    watch.write_deadline(str(target), 950)
    # at the replace the temp file is complete and the target still whole
    assert seen == [("950\n", "100\n", tmp_path)]
    assert target.read_text() == "950\n"
    assert [p.name for p in tmp_path.iterdir()] == ["zfr-deadline"]


def run_loop(
    monkeypatch: pytest.MonkeyPatch, files, cap: int = 1000, start: int = 0, *, path=None,
    read=None,
) -> tuple[list[int], list[str]]:  # fmt: skip
    monkeypatch.setenv("ZFR_DEADLINE_EPOCH", str(cap))
    monkeypatch.setenv("ZFR_DEADLINE_FILE", str(path or "/nonexistent/zfr-deadline"))
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod1")
    now = [start]
    stops: list[int] = []

    def call(query: str, pod: str, key: str) -> dict:
        if query == watch.STOP:
            stops.append(now[0])
            return {"data": {"podStop": {"id": pod}}}
        return {"data": {"pod": {"id": pod}}}

    def sleep(seconds: float) -> None:
        now[0] += int(seconds)

    read = read or (lambda path: files(now[0]))
    watch.main(clock=lambda: now[0], sleep=sleep, read=read, call=call)
    return stops, []


def test_loop_follows_file_change(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stops, _ = run_loop(monkeypatch, lambda t: "900" if t < 800 else "950")
    out = capsys.readouterr().out
    assert len(stops) == 1 and 950 <= stops[0] < 970
    assert [x for x in out.splitlines() if " source " in x] == [
        "zfr-deadline 1970-01-01T00:15:00Z source file",
        "zfr-deadline 1970-01-01T00:15:50Z source file",
    ]
    assert KEY not in out


def test_loop_keeps_last_valid_and_falls_back_to_cap(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stops, _ = run_loop(monkeypatch, lambda t: "900" if t < 100 else None)
    assert 900 <= stops[0] < 920
    assert "source last" in capsys.readouterr().out
    stops, _ = run_loop(monkeypatch, lambda t: None)
    assert 1000 <= stops[0] < 1020
    assert "source env" in capsys.readouterr().out


def test_a_file_already_past_at_the_start_is_stale(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deadline left from an earlier start (in the past when the watcher starts) does not stop
    the Pod at once: it is logged stale, once, and the cap applies."""
    stops, _ = run_loop(monkeypatch, lambda t: "1500", cap=5000, start=2000)
    out = capsys.readouterr().out
    assert len(stops) == 1 and 5000 <= stops[0] < 5020
    assert out.count("zfr-deadline file stale") == 1
    assert "source env" in out and "source file" not in out


def test_the_above_cap_state_is_logged_once(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stops, _ = run_loop(monkeypatch, lambda t: "1200" if t < 500 else "900", cap=1000)
    out = capsys.readouterr().out
    assert 900 <= stops[0] < 920
    assert out.count("zfr-deadline file above cap") == 1
    assert [x for x in out.splitlines() if " source " in x] == [
        "zfr-deadline 1970-01-01T00:16:40Z source env",
        "zfr-deadline 1970-01-01T00:15:00Z source file",
    ]


def test_a_fired_deadline_file_is_retired(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """At the deadline the file is renamed <file>.fired-<epoch>: the next start reads no file,
    and the cap applies."""
    target = tmp_path / "zfr-deadline"
    target.write_text("900\n")
    stops, _ = run_loop(monkeypatch, None, path=target, read=watch.read_file)
    (fired,) = tmp_path.glob("zfr-deadline.fired-*")
    assert not target.exists() and fired.read_text() == "900\n"
    assert int(fired.name.rsplit("-", 1)[1]) == stops[0]
    stops, _ = run_loop(monkeypatch, None, path=target, read=watch.read_file)
    assert 1000 <= stops[0] < 1020
    assert "source env" in capsys.readouterr().out
