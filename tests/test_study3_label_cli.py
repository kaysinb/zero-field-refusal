"""Study 3's sealed labeling command (analysis.study3_label_cli): that ``run_judging.label_runs``
labels a study-3 run's step-25 evaluation as the run records it, which runs the command takes,
and its batches bracketed by the control items (a change before, a change after and its
quarantine, a repeated change, a judge gone). Synthetic, harmless text only; a fake judge server;
output captured at the file-descriptor level (capfd)."""

import hashlib
import json
import stat
import threading
from functools import cache
from pathlib import Path
from types import SimpleNamespace

from test_study2_labels import HARMFUL, IDS, JUDGE, PROMPTS, REFUSAL, factory, make_run
from test_study3_controls import make_archive
from test_study3_records import config_of, jsonl, write_run

from zero_field_refusal.analysis import study3_label_cli
from zero_field_refusal.analysis.evidence import label_path
from zero_field_refusal.analysis.stage_a import load_trajectory
from zero_field_refusal.analysis.study2_label_cli import (
    INCOMPLETE,
    LABELED,
    NOT_ATTEMPTED_STATUS,
)
from zero_field_refusal.analysis.study2_labels import read_endpoint_labels
from zero_field_refusal.analysis.study3_controls import load_controls, record_bytes, select_controls
from zero_field_refusal.analysis.study3_label_cli import (
    CONTROL_CHANGED,
    CONTROLS_CHANGED,
    CONTROLS_OK,
    QUARANTINE,
    STATUSES3,
    refusal,
)
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import JudgeClient, judge_identity
from zero_field_refusal.run_judging import label_runs, receipt_path, rounds_path

TOKENS = {"bos_id": 1, "probe_tokens": 9}
SAFE = "Harmful request: no\nResponse refusal: no\nHarmful response: no"
REPLY = {"harmful_compliance": HARMFUL, "refusal": REFUSAL, "safe_non_refusal": SAFE}
CONTROL_ITEMS = 40


@cache
def _config(role: str) -> tuple[str, str]:
    yaml, chash, _ = config_of("A", 1.45e-5, 10300, role)
    return yaml, chash


def study3_run(root: Path, name: str, role: str = "validation", candidate: int = 1) -> Path:
    """A study-3 run of ``role`` as the tracker and the schedule write it (study 3's resolved
    config, whole records that ``stage_a.load_trajectory`` accepts), answering test_study2_labels'
    six placeholder prompts at its step-25 evaluation (a K-only run answers none)."""
    yaml, chash = _config(role)
    run = root / name
    header = {"study": 3, "candidate": candidate, "step": role, "round": None}
    k_only = role not in ("validation", "confirmation")
    write_run(run, yaml, chash, {"name": "c", "label": "l", "attempt": 1, "header": header},
              k_only=k_only)  # fmt: skip
    readings = [json.loads(line) for line in (run / "monitor.jsonl").read_text().splitlines()]
    readings = [{**r, "K_monitor_full": r["K_monitor"], "K_scoring": r["K_monitor"]}
                for r in readings]  # fmt: skip
    jsonl(run / "monitor.jsonl", readings)
    keys = ("step", "K_monitor", "K_monitor_full", "K_scoring")
    jsonl(run / "diagnostics.jsonl", [{k: readings[-1][k] for k in keys}])
    part = {"bitwise_equal": True, "max_abs_diff": 0.0, "responses": 100, "uncached_finite": True}
    checks = [{"step": s, "primary": part, "scoring": part, "secondary": part} for s in (0, 25)]
    jsonl(run / "cache_checks.jsonl", checks)
    if not k_only:
        meta = json.loads((run / "meta.json").read_text())
        meta["inputs"]["harmful_prompts"] = {
            "set": "harmful_evaluation", "count": len(IDS),
            "ids_sha256": hashlib.sha256(json.dumps(IDS).encode()).hexdigest()}  # fmt: skip
        (run / "meta.json").write_text(json.dumps(meta))
        jsonl(run / "responses" / "harmful_evaluation.jsonl",
              [{"condition": "step-25", "item_id": pid, "response_ids": [1],
                "text": f"Placeholder response step-25 {pid}."} for pid in IDS])  # fmt: skip
    return run


# ------------------------------------------------------------------ run_judging on study 3


def test_run_judging_labels_a_study3_runs_step_25_evaluation(tmp_path) -> None:
    """A validation run of study 3's config, with whole records (``load_trajectory`` unpatched):
    ``label_runs`` labels its one evaluation, ``step-25``, every prompt of its set, writes its
    receipt, and the strict reader counts it, so run_judging labels study-3 runs unchanged."""
    run = study3_run(tmp_path, "validation-run")
    load_trajectory(run)
    fake = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        choices=[SimpleNamespace(text=HARMFUL, finish_reason="stop")])))  # fmt: skip
    (summary,) = label_runs([run], JUDGE, lambda cfg: PROMPTS, factory(fake), rounds=3)
    assert summary == {"run_id": run.name, "items": len(IDS), "unparseable": 0}
    receipt = json.loads(receipt_path(run, JUDGE).read_text())
    assert receipt["expected_items"] == [f"step-25/{pid}" for pid in IDS]
    identity = judge_identity(load_config(f"judge/{JUDGE}"))
    (row,) = read_endpoint_labels(run, JUDGE, identity, PROMPTS)
    assert (row["condition"], row["prompts"], row["harmful"], row["missing"]) == (
        "step-25", len(IDS), len(IDS), 0)  # fmt: skip


# ------------------------------------------------------------------ which runs


def test_only_regular_study3_runs_are_taken(tmp_path) -> None:
    """A validation run is taken; a study-2 run, a K-only study-3 run, a config that does not
    read and a confirmation run before its candidate's analysis freeze are refused (fail
    closed)."""
    closed, opened = (lambda c: ("no analysis freeze",)), (lambda c: ())
    assert refusal(study3_run(tmp_path, "validation"), closed) is None
    assert refusal(make_run(tmp_path, name="study2"), closed) == "not a study-3 run"
    assert "K-only" in refusal(study3_run(tmp_path, "numerical", "numerical"), closed)
    broken = study3_run(tmp_path, "broken")
    (broken / "config.yaml").write_text("seed: [10300\n")
    assert refusal(broken, closed).startswith("unclassifiable")
    confirmation = study3_run(tmp_path, "confirmation", "confirmation", candidate=2)
    asked = []
    assert "before its analysis freeze" in refusal(confirmation, lambda c: asked.append(c) or [1])
    assert asked == [2] and refusal(confirmation, opened) is None


# ------------------------------------------------------------------ the command


class Judge:
    """The fake judge server. A run's item is answered ``reply``; a control item its expected
    label, but at the control checks numbered (from 1, in the order they happen) in ``flip`` the
    first control item gets another one. After ``gone_after`` run requests, every request fails.
    ``runs`` and ``controls`` count the requests answered."""

    def __init__(self, controls, flip=(), reply=HARMFUL, gone_after=None) -> None:
        self.expected = {c["response"]: c["expected_label"] for c in controls}
        self.first = controls[0]["response"]
        self.flip, self.reply, self.gone_after = set(flip), reply, gone_after
        self.runs = self.controls = 0
        self.lock = threading.Lock()
        self.completions = SimpleNamespace(create=self.create)

    def gone(self) -> bool:
        return self.gone_after is not None and self.runs >= self.gone_after

    def create(self, **kwargs):
        prompt = kwargs["prompt"]
        with self.lock:
            if self.gone():
                raise RuntimeError("the judge server is gone")
            control = next((r for r in self.expected if r in prompt), None)
            if control is None:
                self.runs += 1
                text = self.reply
            else:
                check = self.controls // CONTROL_ITEMS + 1
                self.controls += 1
                label = self.expected[control]
                if control == self.first and check in self.flip:
                    label = "refusal" if label == "harmful_compliance" else "harmful_compliance"
                text = REPLY[label]
        return SimpleNamespace(choices=[SimpleNamespace(text=text, finish_reason="stop")])


def setup(tmp_path: Path) -> tuple[Path, list[dict]]:
    """RUNS_DIR with a synthetic calibration archive, and the control record drawn from it."""
    archive = make_archive(tmp_path / "runs" / "judge_calibration" / "calibration")
    record = tmp_path / "controls.json"
    record.write_bytes(record_bytes(select_controls(archive)))
    return record, load_controls(json.loads(record.read_text()), archive)


def cli(monkeypatch, tmp_path: Path, judge: Judge, record: Path, *runs: Path,
        batch: int = 2) -> str | int | None:  # fmt: skip
    """``main`` on ``runs`` against the fake server; returns its exit message (None: exit 0)."""

    class Client:
        @staticmethod
        def from_config(cfg, env=None, **kwargs):
            env = {"JUDGE_API_BASE": "http://judge/v1"}
            return JudgeClient.from_config(cfg, env, client=judge, sleep=lambda s: None, **kwargs)

    def tokens(*args, **kwargs) -> dict:
        if judge.gone():
            raise RuntimeError("the judge server is gone")
        return TOKENS

    monkeypatch.setattr(study3_label_cli, "JudgeClient", Client)
    monkeypatch.setattr(study3_label_cli, "check_wildguard_tokens", tokens)
    monkeypatch.setattr(study3_label_cli, "harmful_prompts", lambda cfg: PROMPTS)
    monkeypatch.setenv("JUDGE_API_BASE", "http://judge/v1")
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    try:
        study3_label_cli.main(["--judge", JUDGE, "--controls", str(record), "--batch-size",
                               str(batch), "--sealed-dir", str(tmp_path / "sealed"),
                               *map(str, runs)])  # fmt: skip
    except SystemExit as stopped:
        return stopped.code
    return None


def printed(capfd) -> tuple[list[dict], dict]:
    """The printed lines as JSON (nothing on stderr, no text of an item), and the sealed
    record."""
    out, err = capfd.readouterr()
    assert err == "" and "Placeholder" not in out and "Calibration" not in out
    lines = [json.loads(line) for line in out.splitlines()]
    for line in lines:
        assert set(line) in ({"batch", "controls"}, {"run_id", "sealed", "status"},
                             {"runs", "token_check"}, {"sealed_record"}), line  # fmt: skip
        assert line.get("status", LABELED) in STATUSES3
    path = Path(lines[-1]["sealed_record"])
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    return lines, json.loads(path.read_text())


def shown(lines: list[dict]) -> list:
    """The batch and run lines, compactly: ``(batch, controls)`` or ``(run id, status)``."""
    return [(x["batch"], x["controls"]) if "batch" in x else (x["run_id"], x["status"])
            for x in lines if "batch" in x or "run_id" in x]  # fmt: skip


def runs(tmp_path: Path, count: int = 3) -> list[Path]:
    return [study3_run(tmp_path / "runs" / "study3", f"run{i}") for i in range(count)]


def labeled(run: Path) -> bool:
    return receipt_path(run, JUDGE).is_file()


def test_batches_bracketed_by_their_controls_label_every_run(monkeypatch, tmp_path, capfd):
    """Three runs in batches of two: each batch's controls are judged before and after it (40
    requests each, anew), every run is labeled, and only the allowlisted lines are printed."""
    record, controls = setup(tmp_path)
    judge, found = Judge(controls), runs(tmp_path)
    assert cli(monkeypatch, tmp_path, judge, record, *found) is None
    lines, sealed = printed(capfd)
    assert shown(lines) == [(1, CONTROLS_OK), (1, CONTROLS_OK), ("run0", LABELED),
                            ("run1", LABELED), (2, CONTROLS_OK), (2, CONTROLS_OK),
                            ("run2", LABELED)]  # fmt: skip
    assert lines[-2] == {"runs": 3, "token_check": TOKENS}
    assert judge.controls == 4 * CONTROL_ITEMS and judge.runs == 3 * len(IDS)
    assert all(labeled(run) for run in found)
    (first, _) = sealed["batches"]
    assert first["run_ids"] == ["run0", "run1"] and not first["judge_mark"]
    (attempt,) = first["attempts"]
    assert attempt["controls_before"]["changed"] == [] and attempt["end"] == "ok"
    assert sealed["controls"]["sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    assert not list((tmp_path / "sealed").glob("controls-*"))  # the temporary label files


def test_a_changed_control_before_a_batch_stops_the_labeling(monkeypatch, tmp_path, capfd):
    record, controls = setup(tmp_path)
    judge, found = Judge(controls, flip={1}), runs(tmp_path)
    message = cli(monkeypatch, tmp_path, judge, record, *found)
    lines, sealed = printed(capfd)
    assert shown(lines) == [(1, CONTROLS_CHANGED), ("run0", CONTROL_CHANGED),
                            ("run1", CONTROL_CHANGED), ("run2", NOT_ATTEMPTED_STATUS)]  # fmt: skip
    assert judge.runs == 0 and not any((run / "judge").exists() for run in found)
    assert "not labeled yet" in message and sealed["batches"][0]["stop"]


def test_a_change_after_a_batch_quarantines_its_labels_and_relabels_it_once(
    monkeypatch, tmp_path, capfd
):
    """The labels written by the first attempt are moved, receipt included, into the quarantine
    folder and listed with their sha256; the batch is labeled again from scratch."""
    record, controls = setup(tmp_path)
    judge, found = Judge(controls, flip={2}), runs(tmp_path, 2)
    assert cli(monkeypatch, tmp_path, judge, record, *found) is None
    lines, sealed = printed(capfd)
    assert shown(lines) == [(1, CONTROLS_OK), (1, CONTROLS_CHANGED), (1, CONTROLS_OK),
                            (1, CONTROLS_OK), ("run0", LABELED), ("run1", LABELED)]  # fmt: skip
    assert judge.runs == 2 * 2 * len(IDS)  # each run labeled twice, from scratch
    first, second = sealed["batches"][0]["attempts"]
    assert first["end"] == "changed_after" and second["end"] == "ok"
    moved = first["quarantined"]
    assert {(m["run_id"], Path(m["from"]).name) for m in moved} == {
        (run.name, path(run, JUDGE).name) for run in found
        for path in (receipt_path, label_path, rounds_path)}  # fmt: skip
    for m in moved:
        assert Path(m["to"]).is_relative_to(tmp_path / "sealed" / QUARANTINE)
        assert hashlib.sha256(Path(m["to"]).read_bytes()).hexdigest() == m["sha256"]
    assert all(labeled(run) for run in found)
    assert json.loads(rounds_path(found[0], JUDGE).read_text())["used"] == 1


def test_a_repeated_change_is_the_judges_technical_mark(monkeypatch, tmp_path, capfd):
    """A change after the relabeling too: both attempts' labels are quarantined, the batch's runs
    are not complete, the next batch is not attempted and the record names the batch."""
    record, controls = setup(tmp_path)
    judge, found = Judge(controls, flip={2, 4}), runs(tmp_path)
    message = cli(monkeypatch, tmp_path, judge, record, *found)
    lines, sealed = printed(capfd)
    assert shown(lines) == [(1, CONTROLS_OK), (1, CONTROLS_CHANGED), (1, CONTROLS_OK),
                            (1, CONTROLS_CHANGED), ("run0", INCOMPLETE), ("run1", INCOMPLETE),
                            ("run2", NOT_ATTEMPTED_STATUS)]  # fmt: skip
    assert "technical mark" in message and "batch 1" in message
    assert sealed["judge_mark"] == [1] and sealed["batches"][0]["judge_mark"]
    assert not any(labeled(run) or label_path(run, JUDGE).exists() for run in found)
    assert len(list((tmp_path / "sealed" / QUARANTINE).iterdir())) == 2


def test_a_judge_gone_during_a_batch(monkeypatch, tmp_path, capfd):
    """A failed round: the batch's runs are not complete, what it wrote is quarantined, the next
    batch is not attempted."""
    record, controls = setup(tmp_path)
    judge, found = Judge(controls, gone_after=3), runs(tmp_path)
    message = cli(monkeypatch, tmp_path, judge, record, *found)
    lines, sealed = printed(capfd)
    assert shown(lines) == [(1, CONTROLS_OK), ("run0", INCOMPLETE), ("run1", INCOMPLETE),
                            ("run2", NOT_ATTEMPTED_STATUS)]  # fmt: skip
    assert "not labeled yet" in message
    assert sealed["batches"][0]["attempts"][0]["end"] == "judge_failed"
    assert not any(label_path(run, JUDGE).exists() for run in found)


def test_a_refused_run_stops_the_command_before_any_label(monkeypatch, tmp_path, capfd):
    record, controls = setup(tmp_path)
    judge = Judge(controls)
    good = study3_run(tmp_path / "runs" / "study3", "good")
    bad = study3_run(tmp_path / "runs" / "study3", "numerical", "numerical")
    message = cli(monkeypatch, tmp_path, judge, record, good, bad)
    out, err = capfd.readouterr()
    assert [json.loads(line) for line in out.splitlines()] == [
        {"run_id": "numerical", "refused": refusal(bad, lambda c: ())}]  # fmt: skip
    assert "nothing was labeled" in message and judge.runs == judge.controls == 0
    assert not (good / "judge").exists()
