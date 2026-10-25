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


def test_effective_deadline_rules(capsys: pytest.CaptureFixture[str]) -> None:
    assert watch.effective_deadline(1000, "900", None) == (900, "file")
    assert watch.effective_deadline(1000, "900\n", 800) == (900, "file")
    for text in (None, "", "x"):
        assert watch.effective_deadline(1000, text, None) == (1000, "env")
    capsys.readouterr()
    assert watch.effective_deadline(1000, "1200", None) == (1000, "env")
    assert "zfr-deadline file above cap" in capsys.readouterr().out


def test_malformed_then_missing_keeps_last_valid() -> None:
    assert watch.effective_deadline(1000, "x", 900) == (900, "last")
    assert watch.effective_deadline(1000, None, 900) == (900, "last")
    assert watch.effective_deadline(1000, "", 900) == (900, "last")


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
    monkeypatch: pytest.MonkeyPatch, files, cap: int = 1000, start: int = 0
) -> tuple[list[int], list[str]]:
    monkeypatch.setenv("ZFR_DEADLINE_EPOCH", str(cap))
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

    watch.main(clock=lambda: now[0], sleep=sleep, read=lambda path: files(now[0]), call=call)
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
