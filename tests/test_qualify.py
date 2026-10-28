"""Study 3's qualification driver (scripts/qualify.py): the cases of a campaign, the driver's
order per case, resume, the attempts of a part, the deletion of the checkpoints and the verdict
(docs/study3-preregistration.md, "Numerical qualification", "Failure routes", "Amendment 2").

Nothing here runs a model or touches a GPU: the parts' harness (``qualification.run_part``) is
replaced by a stub that publishes a record, and the verdicts read synthetic part records.
"""

import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_qualification import CODE, IDENTITY, control_record, synthetic

from zero_field_refusal import code_identity
from zero_field_refusal import qualification as q
from zero_field_refusal.config import load_config
from zero_field_refusal.paths import CONFIG_ROOT

ROOT = Path(__file__).resolve().parents[1]
REGISTERED = load_config("experiments/qualification")
SCOPE = q.QualificationScope.of(REGISTERED)
FA_LR, ADAMW_LR = 1.83e-05, 5.0e-05
FA = q.QualificationCase(10200, "adam_frozen", FA_LR, 1, False)
ADAMW = q.QualificationCase(10200, "adamw", ADAMW_LR, 1, False)


def load_script():
    spec = importlib.util.spec_from_file_location("qualify_script", ROOT / "scripts" / "qualify.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve their annotations there
    spec.loader.exec_module(module)
    return module


def commit_configs(known):
    """``configs_at`` for the synthetic commits: the configs folder of each commit in ``known``;
    any other commit is not in the clone."""

    @contextlib.contextmanager
    def configs_at(commit):
        if commit not in known:
            raise subprocess.CalledProcessError(1, ["git", "rev-parse", "--verify", commit])
        yield known[commit]

    return configs_at


@pytest.fixture
def qualify(monkeypatch, tmp_path):
    module = load_script()
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(module, "git_state", lambda *a, **k: {"commit": "c" * 40, "dirty": False})
    # This checkout's GPU code as the synthetic records have it (``complete``'s stale check)
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(CODE))
    # The launcher runs' synthetic commit abc123 has this checkout's configs
    monkeypatch.setattr(module, "configs_at", commit_configs({"abc123": CONFIG_ROOT}))
    return module


def cell(name, optimizer, lr, seeds, endpoint, role="numerical"):
    return (f"  - name: {name}\n    config: experiments/study3_k_only\n"
            f"    overrides: [study3.role={role}, optimizer={optimizer}, optimizer.lr={lr!r}, "
            f"train.batch_size=20, tracking.endpoint_checkpoint={endpoint}]\n"
            f"    seeds: {list(seeds)}\n")  # fmt: skip


def campaign(tmp_path, *, fa=(10200,), adamw=(10200,), name="study3-qualification-c1",
             role="numerical") -> Path:  # fmt: skip
    """A campaign of FA at 1.83e-5 and AdamW at 5e-5 on ``fa`` and ``adamw``'s seeds."""
    endpoint = tmp_path / "ckpt"
    text = f"name: {name}\ncells:\n"
    text += cell(f"adam_frozen-{FA_LR:g}", "adam_frozen", FA_LR, fa, endpoint, role) if fa else ""
    text += cell(f"adamw-{ADAMW_LR:g}", "adamw", ADAMW_LR, adamw, endpoint, role) if adamw else ""
    path = tmp_path / f"{name}.yaml"
    path.write_text(text)
    return path


def common(path, *extra):
    return ["--campaign", str(path), "--candidate", "1", *extra]


def publish(out, case, part, record, attempt=1):
    q.publish(out / case.attempt_path(part, attempt), record)


def qualified_records(case, identity=None):
    baseline, rest = synthetic(case, **({"identity": identity} if identity else {}))
    return baseline, rest


def write_case(out, case, *, control=False, decided=False, rest=True):
    baseline, rest_record = qualified_records(case)
    if decided:  # a declared instability rule fired on the unshifted baseline trajectory
        baseline["trajectories"]["baseline"]["instability"] = {
            "reason": "gradient_spike", "step": 3, "detail": {"grad_norm": 50.0}}  # fmt: skip
    publish(out, case, "baseline", baseline)
    if rest and not decided:
        rest_record["baseline_record"]["sha256"] = q.record_sha256(baseline)
        publish(out, case, "rest", rest_record)
    if control:
        publish(out, case, "control", control_record(case=case))


# ------------------------------------------------------------------ the cases and the plan


def test_the_cases_of_a_campaign(qualify, tmp_path) -> None:
    name, planned, problems = qualify.campaign_cases(campaign(tmp_path), SCOPE, 1, False)
    assert name == "study3-qualification-c1" and problems == []
    assert [(p.case, p.run.label, p.control) for p in planned] == [
        (FA, f"adam_frozen-{FA_LR:g}/s10200", True),
        (ADAMW, f"adamw-{ADAMW_LR:g}/s10200", False),
    ]
    assert {p.endpoint for p in planned} == {tmp_path / "ckpt"}
    assert planned[0].harness == tmp_path / "ckpt" / "harness" / "s10200-adam_frozen-1.83e-05"


def test_a_campaign_outside_the_scope_is_refused(qualify, tmp_path) -> None:
    path = campaign(tmp_path, fa=(10201,), adamw=(10300,))  # FA without its first seed's case
    _, _, problems = qualify.campaign_cases(path, SCOPE, 1, False)
    assert any("10300" in p and "seeds" in p for p in problems)
    assert any("same-path control" in p for p in problems)
    for command in (["plan"], ["verdict"], ["part", "--case", "s10201/adam_frozen-1.83e-05"]):
        with pytest.raises(SystemExit):
            qualify.main([*command, *common(path)])
    engineering = campaign(tmp_path, fa=(10000,), adamw=(10001,), name="e", role="engineering")
    assert qualify.campaign_cases(engineering, SCOPE, 1, True)[2] == []
    assert qualify.campaign_cases(engineering, SCOPE, 1, False)[2]  # not numerical seeds


def test_the_plan_states_the_gpu_time_and_the_disk(qualify, tmp_path, capsys) -> None:
    qualify.main(["plan", *common(campaign(tmp_path))])
    out = capsys.readouterr().out
    # two cases at 7 + 7 + 25 min, one same-path control at 11 min
    assert "2 cases" in out and "≈ 89 GPU min" in out
    assert "23.5 GB" in out  # 2 endpoints + 1 of headroom at 6.5 GB, and 4 GB of margin
    assert "free host memory: 60 GB" in out


def test_the_disk_need_counts_the_diagnostic_steps(qualify, tmp_path) -> None:
    assert qualify.disk_need_gb(REGISTERED, []) == pytest.approx(3 * 6.5 + 4)
    assert qualify.disk_need_gb(REGISTERED, [5, 10, 15]) == pytest.approx(6 * 6.5 + 4)
    path = campaign(tmp_path, fa=(10000,), adamw=(10001,), name="e", role="engineering")
    out = tmp_path / "driver.sh"
    qualify.main(["driver", *common(path, "--engineering", "--diagnostic-steps", "5,10,15"),
                  "--commit", "a" * 40, "--out", str(out)])  # fmt: skip
    text = out.read_text()
    assert f"-lt {qualify.need_kb(6 * 6.5 + 4)} ]" in text
    assert text.count("--diagnostic-steps 5,10,15") == 4  # each case's baseline and rest


# ------------------------------------------------------------------------------- the driver


def driver_text(qualify, tmp_path, path=None) -> str:
    out = tmp_path / "driver.sh"
    qualify.main(["driver", *common(path or campaign(tmp_path)), "--commit", "a" * 40,
                  "--out", str(out)])  # fmt: skip
    return out.read_text()


def positions(text: str, patterns: list[str]) -> list[int]:
    found, start = [], 0
    for pattern in patterns:
        match = re.compile(pattern, re.M).search(text, start)
        assert match is not None, f"{pattern!r} not found after {start}"
        found.append(match.start())
        start = match.end()
    return found


def test_the_driver_runs_each_case_in_order(qualify, tmp_path) -> None:
    text = driver_text(qualify, tmp_path)
    fa, adamw = f"adam_frozen-{FA_LR:g}/s10200", f"adamw-{ADAMW_LR:g}/s10200"
    order = []
    for label, name, control in ((fa, FA.name, True), (adamw, ADAMW.name, False)):
        order += [rf"qualify\.py complete .*--case '?{re.escape(name)}'?",
                  r"qualify\.py clear-stale", r"df -Pk", r"-lt \d+ \]",
                  rf"scripts/launch\.py \"\$CAMPAIGN\" --only '?{re.escape(label)}'? "
                  r"--max-attempts 1", r"qualify\.py locate",
                  r"qualify\.py part .*--part baseline --resume", r"qualify\.py decided",
                  r"qualify\.py part .*--part rest --resume",
                  *([r"qualify\.py part .*--part control --resume"] if control else []),
                  r"qualify\.py finish"]  # fmt: skip
    positions(text, order)
    assert text.count("--part control") == 1  # FA on the first numerical seed alone
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)


def test_the_driver_has_no_rerun_line(qualify, tmp_path) -> None:
    """No automatic second attempt: a part's record (technical or scientific) is never rerun by
    the driver, and the launcher gets one attempt."""
    text = driver_text(qualify, tmp_path)
    commands = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    assert "--attempt" not in commands and "--infrastructure" not in commands
    assert "--recovery" not in commands
    assert commands.count("--max-attempts 1") == 2


STUB = r"""#!/usr/bin/env bash
# project-run "$SRC" python <script> <command> ...: logs the command, exits as STUB_<command> says
if [ "$3" = -c ]; then exit 0; fi
cmd="$4"
if [ "$3" = scripts/launch.py ]; then cmd=launch; fi
echo "$cmd ${*:5}" >> "$STUB_LOG"
n=$(grep -c "^$cmd " "$STUB_LOG")
var="STUB_${cmd//-/_}"
read -r -a codes <<< "${!var:-0}"
i=$(( n - 1 < ${#codes[@]} ? n - 1 : ${#codes[@]} - 1 ))
if [ "$cmd" = locate ]; then echo /runs/study3/r1; fi
exit "${codes[$i]}"
"""


def run_driver(qualify, tmp_path, monkeypatch, **exits):
    """The generated driver run by bash against stubs: ``project-run`` (each qualify.py command
    and the launcher logged, with the exit ``exits`` gives it: a code, or codes call by call),
    ``git`` and ``nvidia-smi``; no Pod environment, and a disk need of 1 KB."""
    commit = "a" * 40
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, text in (("project-run", STUB), ("git", f"#!/bin/sh\necho {commit}\n"),
                       ("nvidia-smi", '#!/bin/sh\ncase "$*" in *compute-apps*) ;; '
                                      "*) echo 0 ;; esac\n")):  # fmt: skip
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    (tmp_path / "code" / commit / "source").mkdir(parents=True)
    monkeypatch.setattr(qualify, "POD_ENVIRONMENT", ())
    monkeypatch.setattr(qualify, "SOURCES", str(tmp_path / "code"))
    monkeypatch.setattr(qualify, "need_kb", lambda gb: 1)
    out = tmp_path / "driver.sh"
    qualify.main(["driver", *common(campaign(tmp_path)), "--commit", commit, "--out", str(out)])
    log = tmp_path / "stub.log"
    log.touch()
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "RUNS_DIR": str(tmp_path / "runs"),
           "STUB_LOG": str(log), "STUB_complete": "3", "STUB_decided": "3",
           **{f"STUB_{key}": str(value) for key, value in exits.items()}}  # fmt: skip
    done = subprocess.run(["bash", str(out)], env=env, capture_output=True, text=True)
    calls = [line.split(" ", 1)[0] + " " + " ".join(re.findall(r"--(?:part|case) (\S+)", line))
             for line in log.read_text().splitlines()]  # fmt: skip
    return done.returncode, [c.strip() for c in calls]


def _with_case(steps, case):
    """The stub's log lines of ``steps`` for ``case``: the command, then its ``--case`` and
    ``--part`` values as they appear; the launcher's line names neither."""
    named = {"launch": "launch"}
    return [named.get(step) or (f"part {case.name} {step.split()[1]}" if step.startswith("part ")
                                else f"{step} {case.name}") for step in steps]  # fmt: skip


FULL = ["complete", "clear-stale", "launch", "locate", "part baseline", "decided", "part rest"]


def test_the_driver_runs_every_part_of_each_case(qualify, tmp_path, monkeypatch) -> None:
    code, calls = run_driver(qualify, tmp_path, monkeypatch)
    assert code == 0
    assert calls == [*_with_case([*FULL, "part control", "finish"], FA),
                     *_with_case([*FULL, "finish"], ADAMW), "verdict"]  # fmt: skip


def test_the_driver_resumes_past_a_complete_case(qualify, tmp_path, monkeypatch) -> None:
    """A complete case (``complete`` exits 0) runs neither the launcher nor a part: only its
    ``finish``, which deletes what is left."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, complete="0 3")
    assert code == 0
    assert calls == [f"complete {FA.name}", f"finish {FA.name}",
                     *_with_case([*FULL, "finish"], ADAMW), "verdict"]  # fmt: skip


def test_a_decided_case_skips_the_rest_but_not_its_control(qualify, tmp_path, monkeypatch):
    code, calls = run_driver(qualify, tmp_path, monkeypatch, decided="0 3")
    assert code == 0
    assert calls[: len(FULL) + 1] == _with_case([*FULL[:-1], "part control", "finish"], FA)
    assert f"part {FA.name} rest" not in calls


@pytest.mark.parametrize("command", ["complete", "decided"])
def test_a_crashing_check_stops_the_driver(qualify, tmp_path, monkeypatch, command) -> None:
    """``complete`` and ``decided`` say no with exit 3; any other non-zero exit is a failure
    that stops the driver (its verdict still written)."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, **{command: 2})
    assert code == 2
    assert calls[-2:] == [f"{command} {FA.name}", "verdict"]
    assert f"part {FA.name} rest" not in calls and f"finish {FA.name}" not in calls


def test_the_driver_stops_after_a_technical_case(qualify, tmp_path, monkeypatch) -> None:
    """``finish`` exits 9 for a technical case and keeps its checkpoints: the driver stops there,
    before the next case."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, finish=qualify.TECHNICAL_CASE)
    assert code == qualify.TECHNICAL_CASE
    assert calls[-2:] == [f"finish {FA.name}", "verdict"]
    assert not any(ADAMW.name in call for call in calls)


def test_the_driver_validates_the_free_space(qualify, tmp_path) -> None:
    text = driver_text(qualify, tmp_path)
    positions(text, [r"free_kb=\$\(df -Pk", r"\*\[!0-9\]\*\) stop", r"-lt \d+ \]"])


def test_the_driver_checks_its_campaign_and_the_gpu(qualify, tmp_path) -> None:
    path = campaign(tmp_path)
    text = driver_text(qualify, tmp_path, path)
    assert f"CAMPAIGN={path.resolve()}" in text
    assert hashlib.sha256(path.read_bytes()).hexdigest() in text
    positions(text, [r"git rev-parse HEAD", r"sha256sum", r"trap .*verdict", r"nvidia-smi",
                     r"drift import host_memory.*\" 60; then", r"60 GB of free host memory",
                     r"--case "])  # fmt: skip
    with pytest.raises(SystemExit):  # an existing --out
        qualify.main(["driver", *common(path), "--commit", "a" * 40,
                      "--out", str(path)])  # fmt: skip


# ----------------------------------------------------------------- resume and the attempts


def test_complete_and_decided(qualify, tmp_path) -> None:
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    run = lambda what, case: qualify.main([what, *common(path), "--case", case.name])  # noqa: E731

    def no(what, case):
        with pytest.raises(SystemExit) as stop:
            run(what, case)
        assert stop.value.code == qualify.NO

    no("complete", FA)
    no("decided", FA)
    write_case(out, FA)  # no control yet
    no("complete", FA)
    publish(out, FA, "control", control_record(case=FA))
    run("complete", FA)  # exit 0
    no("decided", FA)
    write_case(out, ADAMW, decided=True)  # the baseline is a scientific failure: no rest
    run("decided", ADAMW)
    run("complete", ADAMW)


def test_a_decided_fa_case_needs_its_control(qualify, tmp_path) -> None:
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    write_case(out, FA, decided=True)
    qualify.main(["decided", *common(path), "--case", FA.name])
    with pytest.raises(SystemExit) as stop:
        qualify.main(["complete", *common(path), "--case", FA.name])
    assert stop.value.code == qualify.NO
    publish(out, FA, "control", control_record(case=FA))
    qualify.main(["complete", *common(path), "--case", FA.name])


def test_a_rest_bound_to_another_baseline_keeps_the_case_open(qualify, tmp_path) -> None:
    """A rest record counts only when it names the latest baseline record's sha256 (a rest that
    ran nothing excepted): otherwise the case is not complete and keeps its checkpoints."""
    path = campaign(tmp_path, fa=())
    out = tmp_path / "runs" / "qualification"
    journal(tmp_path / "runs", path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "r-a1.pt").write_bytes(b"w")
    write_case(out, ADAMW)
    baseline = json.loads((out / ADAMW.attempt_path("baseline", 1)).read_text())
    publish(out, ADAMW, "baseline", {**baseline, "seconds": 1.0})  # another baseline record
    with pytest.raises(SystemExit) as stop:
        qualify.main(["complete", *common(path), "--case", ADAMW.name])
    assert stop.value.code == qualify.NO
    with pytest.raises(SystemExit, match="not bound to the latest baseline record"):
        qualify.main(["finish", *common(path), "--case", ADAMW.name])
    assert (tmp_path / "ckpt" / "r-a1.pt").exists()
    rest = json.loads((out / ADAMW.attempt_path("rest", 1)).read_text())
    publish(out, ADAMW, "rest", {**rest, "status": "not_run", "kind": "technical",
                                 "reasons": ["no readable baseline record"]})  # fmt: skip
    qualify.main(["complete", *common(path), "--case", ADAMW.name])
    failed = {k: v for k, v in rest.items() if k != "baseline_record"}  # failed before reading
    publish(out, ADAMW, "rest", {**failed, "status": "failed", "kind": "technical",
                                 "reasons": ["RuntimeError: before the baseline"]})  # fmt: skip
    qualify.main(["complete", *common(path), "--case", ADAMW.name])
    with pytest.raises(SystemExit) as stop:  # complete, and technical: its checkpoints stay
        qualify.main(["finish", *common(path), "--case", ADAMW.name])
    assert stop.value.code == qualify.TECHNICAL_CASE


def fake_part(calls):
    def run_part(settings, env, case, run_dir, out, *, part, attempt, checkpoints,
                 diagnostic_steps=(), infrastructure=None, engineering_cycle=None):  # fmt: skip
        calls.append((case.name, part, attempt))
        record = {
            "part": part,
            "attempt": attempt,
            "case": case.record(),
            "status": "failed",
            "kind": "technical",
            "reasons": ["RuntimeError: boom"],
            **({"infrastructure": infrastructure} if infrastructure else {}),
            **({"engineering_cycle": engineering_cycle} if engineering_cycle else {}),
        }
        q.publish(out / case.attempt_path(part, attempt), record)  # the one publish
        return record

    return run_part


@pytest.fixture
def stubbed(qualify, monkeypatch):
    """The part's environment and harness stubbed: nothing loads a model or touches a GPU."""
    calls = []
    monkeypatch.setattr(qualify.runtime, "configure_numerics", lambda cfg, device: {})
    monkeypatch.setattr(qualify.runtime, "environment", lambda cfg, device: object())
    monkeypatch.setattr(qualify.q, "run_part", fake_part(calls))
    return calls


def part_args(path, tmp_path, part, *mode):
    return ["part", *common(path), "--case", FA.name, "--part", part, *mode,
            "--run-dir", str(tmp_path / "run"), "--checkpoints", str(tmp_path / "h"),
            "--device", "cpu"]  # fmt: skip


def test_a_recorded_part_is_never_rerun_by_the_driver(qualify, stubbed, tmp_path) -> None:
    """``--resume`` runs attempt 1 of a part without one; a recorded part (here technical) is
    left as it is; an attempt started that left no record stops with exit 8."""
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    qualify.main(part_args(path, tmp_path, "baseline", "--resume"))
    assert stubbed == [(FA.name, "baseline", 1)]
    assert (out / (FA.attempt_path("baseline", 1) + ".started")).is_file()
    qualify.main(part_args(path, tmp_path, "baseline", "--resume"))  # technical: no rerun
    assert stubbed == [(FA.name, "baseline", 1)]
    q.publish(out / (FA.attempt_path("rest", 1) + ".started"), {"started_at": "t"})  # killed
    with pytest.raises(SystemExit) as stop:
        qualify.main(part_args(path, tmp_path, "rest", "--resume"))
    assert stop.value.code == qualify.INTERRUPTED and len(stubbed) == 1


def test_an_infrastructure_rerun_is_a_new_attempt_that_names_its_reason(
    qualify, stubbed, tmp_path
) -> None:
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    q.publish(out / (FA.attempt_path("baseline", 1) + ".started"), {"started_at": "t"})
    for mode in (["--attempt", "2"], ["--attempt", "3", "--infrastructure", "x"],
                 ["--attempt", "1", "--infrastructure", "x"], ["--attempt", "1"]):  # fmt: skip
        with pytest.raises(SystemExit):
            qualify.main(part_args(path, tmp_path, "baseline", *mode))
    assert stubbed == []
    reason = "the Pod's deadline stopped attempt 1"
    qualify.main(part_args(path, tmp_path, "baseline", "--attempt", "2",
                           "--infrastructure", reason))  # fmt: skip
    assert stubbed == [(FA.name, "baseline", 2)]
    record = json.loads((out / FA.attempt_path("baseline", 2)).read_text())
    assert record["attempt"] == 2
    assert record["infrastructure"] == {
        "reason": reason,
        "previous": [{"attempt": 1, "state": "interrupted"}],
        "restarts": [],
    }
    qualify.main(part_args(path, tmp_path, "baseline", "--resume"))  # the latest is recorded
    assert len(stubbed) == 1


@pytest.mark.parametrize("kind", ["scientific", "technical"])
def test_a_recorded_attempt_is_never_rerun(qualify, stubbed, tmp_path, kind) -> None:
    """Whatever its kind, a recorded attempt has no infrastructure rerun: only an interrupted
    one (started, no record) has."""
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    publish(out, FA, "baseline", {"part": "baseline", "attempt": 1, "case": FA.record(),
                                  "status": "not_run", "kind": kind, "reasons": ["x"]})  # fmt: skip
    with pytest.raises(SystemExit, match="recorded"):
        qualify.main(part_args(path, tmp_path, "baseline", "--attempt", "2",
                               "--infrastructure", "a reason"))  # fmt: skip
    assert stubbed == []


def test_the_verdict_shows_every_attempt(qualify, stubbed, tmp_path, monkeypatch) -> None:
    """An interrupted attempt and its infrastructure rerun are both in the verdict record; a
    record superseded by a later one (which the commands never write) is shown and counted
    against its rate, never hidden."""
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    q.publish(out / (FA.attempt_path("baseline", 1) + ".started"), {"started_at": "t"})
    qualify.main(part_args(path, tmp_path, "baseline", "--attempt", "2",
                           "--infrastructure", "the Pod stopped"))  # fmt: skip
    qualify.main(["verdict", *common(path)])
    record = json.loads((out / "c1" / "verdict.json").read_text())
    shown = record["attempts"][FA.name]["baseline"]
    assert [(a["attempt"], a["state"]) for a in shown] == [(1, "interrupted"), (2, "recorded")]
    assert record["superseded"] == []
    # attempt 1 recorded under attempt 2: shown, and a case of its own against the rate
    publish(out, ADAMW, "baseline", synthetic(ADAMW)[0], attempt=1)
    publish(out, ADAMW, "baseline", synthetic(ADAMW)[0], attempt=2)
    qualify.main(["verdict", *common(path)])
    record = json.loads((out / "c1" / "verdict.json").read_text())
    assert [(s["case"], s["part"], s["attempt"]) for s in record["superseded"]] == [
        (ADAMW.name, "baseline", 1)
    ]
    adamw = [c for c in record["cases"] if c["case"]["name"] == ADAMW.name]
    mark = {"part": "baseline", "attempt": 1, "restart": None, "supersession": None}
    assert [c.get("superseded") for c in adamw] == [None, mark]
    assert record["states"][f"adamw-{ADAMW_LR:g}"] != "qualified"


def test_a_superseded_control_blocks_its_rate(qualify, tmp_path, monkeypatch) -> None:
    """A control record under a later attempt that no restart names is counted against its
    rate, as a superseded case is: the rate is not qualified (scientific: failing)."""
    path, out = verdict_world(qualify, tmp_path, monkeypatch)
    key = f"adam_frozen-{FA_LR:g}"
    record = json.loads((out / FA.attempt_path("control", 1)).read_text())
    publish(out, FA, "control", record, attempt=2)  # attempt 1 is now superseded
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    assert [(e["part"], e["attempt"]) for e in verdict["superseded"]] == [("control", 1)]
    assert verdict["states"][key] == "not_qualified"
    assert [e["why"] for e in verdict["rates"][key]["not_counted"]] == [
        "a superseded same-path control"]  # fmt: skip
    science = {**record, "status": "not_run", "kind": "scientific", "reasons": ["instability"]}
    publish(out, FA, "control", science, attempt=1)
    qualify.main(["verdict", *common(path)])
    assert json.loads((out / "c1" / "verdict.json").read_text())["states"][key] == "failing"


def test_an_interrupted_attempt_recorded_technical(qualify, tmp_path) -> None:
    """``interrupted --technical REASON``: an interrupted attempt becomes a technical record that
    carries the reason ("an unexplained incomplete record is technical"); refused for an attempt
    with a record, or one never started."""
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    command = ["interrupted", *common(path), "--case", FA.name, "--part", "rest",
               "--attempt", "1", "--technical", "no cause found"]  # fmt: skip
    with pytest.raises(SystemExit, match="not an interrupted attempt"):
        qualify.main(command)  # never started
    write_case(out, FA, rest=False)
    q.publish(out / (FA.attempt_path("rest", 1) + ".started"),
              {"started_at": "t", "run_dir": str(tmp_path / "run")})  # fmt: skip
    qualify.main(command)
    record = json.loads((out / FA.attempt_path("rest", 1)).read_text())
    assert (record["part"], record["attempt"], record["status"], record["kind"]) == (
        "rest",
        1,
        "failed",
        "technical",
    )
    assert record["interrupted"]["reason"] == "no cause found"
    assert any("no cause found" in r for r in record["reasons"])
    baseline = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
    verdict = q.case_verdict(baseline, record, SCOPE)
    assert verdict["kind"] == "technical"
    with pytest.raises(SystemExit, match="not an interrupted attempt"):
        qualify.main(command)  # recorded now


RESTART = "the Pod stopped during the case; its container disk is gone"


def restart(path, case, *reason):
    qualify_args = ["restart-case", *common(path), "--case", case.name]
    return qualify_args + (["--infrastructure", *reason] if reason else [])


def test_a_case_restart_lets_the_case_run_again(qualify, stubbed, tmp_path, monkeypatch) -> None:
    """A Pod stop during a case: ``restart-case`` names the case's attempts (the recorded
    baseline with its sha256, the interrupted rest); the case is then not complete, its parts
    take their next attempts with the reason, and the verdict voids the named records."""
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(CODE))
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    write_case(out, FA, rest=False, control=True)
    q.publish(out / (FA.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    with pytest.raises(SystemExit, match="recorded"):  # no rerun of a recorded attempt as such
        qualify.main(part_args(path, tmp_path, "baseline", "--attempt", "2",
                               "--infrastructure", RESTART))  # fmt: skip
    qualify.main(restart(path, FA, RESTART))
    folder = out / FA.attempt_path("baseline", 1).rsplit("/", 1)[0]
    record = json.loads((folder / "restart-1.json").read_text())
    assert record["reason"] == RESTART and record["case"] == FA.record()
    baseline = json.loads((folder / "baseline-attempt-1.json").read_text())
    control = json.loads((folder / "control-attempt-1.json").read_text())
    assert [(a["part"], a["attempt"], a["state"], a["sha256"]) for a in record["attempts"]] == [
        ("baseline", 1, "recorded", q.record_sha256(baseline)),
        ("rest", 1, "interrupted", None),
        ("control", 1, "recorded", q.record_sha256(control)),
    ]
    with pytest.raises(SystemExit, match="no attempt to restart"):  # every attempt named
        qualify.main(restart(path, FA, RESTART))
    qualify.main(["verdict", *common(path)])  # the named control is not the rate's current one
    fa = json.loads((out / "c1" / "verdict.json").read_text())["rates"][f"adam_frozen-{FA_LR:g}"]
    assert [v["case"] for v in fa["voided"]] == [FA.name, "control"]
    assert fa["control"]["reasons"] == ["no same-path control record"]
    with pytest.raises(SystemExit) as stop:
        qualify.main(["complete", *common(path), "--case", FA.name])
    assert stop.value.code == qualify.NO
    with pytest.raises(SystemExit) as stop:  # the driver's call: the rerun is by hand
        qualify.main(part_args(path, tmp_path, "baseline", "--resume"))
    assert stop.value.code == qualify.INTERRUPTED
    with pytest.raises(SystemExit, match="baseline attempt runs first"):
        qualify.main(part_args(path, tmp_path, "rest", "--attempt", "2",
                               "--infrastructure", RESTART))  # fmt: skip
    with pytest.raises(SystemExit, match="needs --infrastructure"):
        qualify.main(part_args(path, tmp_path, "baseline", "--attempt", "2"))
    assert stubbed == []
    for name in ("baseline", "rest", "control"):
        qualify.main(part_args(path, tmp_path, name, "--attempt", "2",
                               "--infrastructure", RESTART))  # fmt: skip
    assert stubbed == [(FA.name, name, 2) for name in ("baseline", "rest", "control")]
    rerun = json.loads((folder / "baseline-attempt-2.json").read_text())
    assert rerun["infrastructure"]["reason"] == RESTART
    assert rerun["infrastructure"]["restarts"] == [
        {
            "path": str(folder / "restart-1.json"),
            "sha256": hashlib.sha256((folder / "restart-1.json").read_bytes()).hexdigest(),
        }
    ]
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    assert [
        (e["part"], e["attempt"], e["restart"])
        for e in verdict["superseded"]
        if e["case"] == FA.name
    ] == [("baseline", 1, RESTART), ("control", 1, RESTART)]
    shown = verdict["attempts"][FA.name]["baseline"]
    assert [(a["attempt"], a["restart"]) for a in shown] == [(1, RESTART), (2, None)]
    fa = verdict["rates"][f"adam_frozen-{FA_LR:g}"]
    assert sorted(v["case"] for v in fa["voided"]) == ["control", FA.name]
    assert {v["reason"] for v in fa["voided"]} == {RESTART}
    # The old attempts neither count nor block: what blocks is the new attempts' case, which the
    # stub wrote without a GPU-code identity
    void = {v["record_sha256"] for v in fa["voided"]}
    assert [e["why"] for e in fa["not_counted"]] == ["other GPU code"]
    assert void.isdisjoint(e["record_sha256"] for e in fa["not_counted"])


def technical_adamw(out, tmp_path):
    """A completed baseline whose launcher~baseline is beyond 0.01·K* (technical), with its rest
    interrupted."""
    baseline = synthetic(ADAMW)[0]
    baseline["comparisons"]["launcher~baseline"]["D_twin"] = 0.07  # beyond 0.06: technical
    publish(out, ADAMW, "baseline", baseline)
    q.publish(out / (ADAMW.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    return baseline, case_folder(out, ADAMW) / "baseline-attempt-1.json"


def supersede_args(path, case, record):
    return ["supersede", *common(path), "--case", case.name, "--record", str(record),
            "--diagnosis", "the launcher and the harness disagreed: a stale kernel cache",
            "--fix", "the cache is cleared at the part's start"]  # fmt: skip


FIX = {**CODE, "commit": "f" * 40, "sha256": "f" * 64}  # the fix's GPU code


def test_a_restart_never_voids_a_completed_failure(qualify, tmp_path) -> None:
    """Only an interrupted case is restarted: a completed technical failure (the baseline's
    launcher~baseline beyond 0.01·K*, the rest then interrupted) is refused, as a scientific one
    is, and so is a complete case and a restart without a reason."""
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    with pytest.raises(SystemExit, match="no attempt to restart"):  # nothing recorded yet
        qualify.main(restart(path, ADAMW, RESTART))
    technical_adamw(out, tmp_path)
    for reason in ((), ("",)):
        with pytest.raises(SystemExit, match="needs --infrastructure REASON"):
            qualify.main(restart(path, ADAMW, *reason))
    with pytest.raises(SystemExit, match=r"completed failure.*baseline attempt 1.*technical"):
        qualify.main(restart(path, ADAMW, RESTART))
    assert not list(case_folder(out, ADAMW).glob("restart-*"))
    write_case(out, FA, decided=True)  # a scientific failure
    q.publish(out / (FA.attempt_path("control", 1) + ".started"), {"started_at": "t"})
    with pytest.raises(SystemExit, match=r"completed failure.*scientific"):
        qualify.main(restart(path, FA, RESTART))
    (out / (FA.attempt_path("control", 1) + ".started")).unlink()
    publish(out, FA, "control", control_record(case=FA))  # now complete
    with pytest.raises(SystemExit, match="the case is complete"):
        qualify.main(restart(path, FA, RESTART))


def test_the_engineering_cycle_requalifies_the_candidates_old_code(
    qualify, stubbed, tmp_path, monkeypatch, capsys
) -> None:
    """The candidate's one supersession: it names the technical record that started it, the
    diagnosis, the fix and its GPU code, and voids every record of the candidate made under the
    old code, a passed case's included; refused with the old code, and a second time. Its reruns
    carry --engineering-cycle, not --infrastructure."""
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    baseline, record = technical_adamw(out, tmp_path)
    write_case(out, FA, control=True)  # a passed case of the candidate, under the old code
    args = supersede_args(path, ADAMW, record)
    for missing in ("--diagnosis", "--fix", "--record"):
        i = args.index(missing)
        with pytest.raises(SystemExit, match="supersede needs"):
            qualify.main(args[:i] + args[i + 2 :])
    with pytest.raises(SystemExit, match="the same path is not simply rerun"):
        qualify.main(args)  # this checkout's GPU code is the record's
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(FIX))
    qualify.main(args)
    stage = out / "c1"  # the candidate's folder holds its one cycle
    cycle = json.loads((stage / "supersede-1.json").read_text())
    assert cycle["old_record"] == {"case": ADAMW.name, "stage": "numerical", "path": str(record),
                                   "sha256": q.record_sha256(baseline), "part": "baseline",
                                   "attempt": 1}  # fmt: skip
    assert cycle["fix_gpu_code"] == FIX and cycle["old_gpu_code"] == CODE
    assert sorted(c["case"] for c in cycle["cases_requalified"]) == sorted([FA.name, ADAMW.name])
    named = {(e["case"], e["part"], e["attempt"], e["state"]) for e in cycle["attempts"]}
    assert named == {(FA.name, "baseline", 1, "recorded"), (FA.name, "rest", 1, "recorded"),
                     (FA.name, "control", 1, "recorded"), (ADAMW.name, "baseline", 1, "recorded"),
                     (ADAMW.name, "rest", 1, "interrupted")}  # fmt: skip
    other = supersede_args(path, FA, case_folder(out, FA) / "baseline-attempt-1.json")
    with pytest.raises(SystemExit, match="one cycle per candidate"):
        qualify.main(other)
    qualify.main(args)  # the same record again: only the journal's missing lines
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    void = {(e["case"], e["part"]) for e in verdict["superseded"]
            if e["supersession"] and e["restart"] is None}  # fmt: skip
    assert void == {(FA.name, "baseline"), (FA.name, "rest"), (FA.name, "control"),
                    (ADAMW.name, "baseline")}  # fmt: skip
    assert all(e["supersession"].startswith("superseded by an engineering cycle")
               for e in verdict["superseded"])  # fmt: skip
    fa = verdict["rates"][f"adam_frozen-{FA_LR:g}"]
    assert sorted(v["case"] for v in fa["voided"]) == sorted([FA.name, FA.name, "control"])
    rerun = ["part", *common(path), "--case", ADAMW.name, "--part", "baseline", "--attempt", "2",
             "--run-dir", str(tmp_path / "run"), "--checkpoints", str(tmp_path / "h"),
             "--device", "cpu"]  # fmt: skip
    for mark in ([], ["--infrastructure", "a reason"]):
        with pytest.raises(SystemExit, match="carries --engineering-cycle"):
            qualify.main([*rerun, *mark])
    qualify.main([*rerun, "--engineering-cycle"])
    assert stubbed == [(ADAMW.name, "baseline", 2)]  # the case re-qualifies
    record2 = json.loads((case_folder(out, ADAMW) / "baseline-attempt-2.json").read_text())
    assert record2["engineering_cycle"]["path"] == str(stage / "supersede-1.json")
    assert record2["engineering_cycle"]["sha256"] == hashlib.sha256(
        (stage / "supersede-1.json").read_bytes()).hexdigest()  # fmt: skip
    assert "infrastructure" not in record2
    # The rest the cycle named (interrupted): its rerun is named --engineering-cycle
    rest = ["part", *common(path), "--case", ADAMW.name, "--part", "rest",
            "--run-dir", str(tmp_path / "run"), "--checkpoints", str(tmp_path / "h"),
            "--device", "cpu"]  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        qualify.main([*rest, "--resume"])
    assert stop.value.code == qualify.INTERRUPTED
    assert "--attempt 2 --engineering-cycle" in capsys.readouterr().err
    # A new-code attempt the Pod then stopped: an ordinary infrastructure rerun
    q.publish(out / (ADAMW.attempt_path("rest", 2) + ".started"), {"started_at": "t"})
    with pytest.raises(SystemExit) as stop:
        qualify.main([*rest, "--resume"])
    assert "--attempt 3 --infrastructure REASON" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="no engineering cycle supersedes this part"):
        qualify.main([*rest, "--attempt", "3", "--engineering-cycle"])
    qualify.main([*rest, "--attempt", "3", "--infrastructure", "the Pod stopped"])
    record3 = json.loads((case_folder(out, ADAMW) / "rest-attempt-3.json").read_text())
    assert record3["infrastructure"]["reason"] == "the Pod stopped"
    assert "engineering_cycle" not in record3


P_LR = 1.0e-05
P = q.QualificationCase(10200, "adamw_prewarmed", P_LR, 1, False)


def test_an_engineering_cycle_keeps_the_candidates_scientific_failures(
    qualify, tmp_path, monkeypatch
) -> None:
    """P failing scientifically and F technical: the cycle for F goes ahead. P's scientific
    record is kept (listed with its sha256), keeps counting, and P stays failing (closed), its
    case complete and not stale; F's records are voided and F re-qualifies."""
    endpoint = tmp_path / "ckpt"
    path = tmp_path / "study3-qualification-c1.yaml"
    path.write_text("name: study3-qualification-c1\ncells:\n"
                    + cell(f"adam_frozen-{FA_LR:g}", "adam_frozen", FA_LR, (10200,), endpoint)
                    + cell(f"adamw_prewarmed-{P_LR:g}", "adamw_prewarmed", P_LR, (10200,),
                           endpoint))  # fmt: skip
    out = tmp_path / "runs" / "qualification"
    write_case(out, P, decided=True)  # P: a declared instability rule fired on its baseline
    baseline = synthetic(FA)[0]
    baseline["comparisons"]["launcher~baseline"]["D_twin"] = 0.07  # F: technical
    publish(out, FA, "baseline", baseline)
    q.publish(out / (FA.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    p_record = case_folder(out, P) / "baseline-attempt-1.json"
    key = f"adamw_prewarmed-{P_LR:g}"
    qualify.main(["verdict", *common(path)])
    assert json.loads((out / "c1" / "verdict.json").read_text())["states"][key] == "failing"
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(FIX))
    qualify.main(supersede_args(path, FA, case_folder(out, FA) / "baseline-attempt-1.json"))
    cycle = json.loads((out / "c1" / "supersede-1.json").read_text())
    assert [(e["case"], e["part"], e["attempt"], e["sha256"]) for e in cycle["kept"]] == [
        (P.name, "baseline", 1, hashlib.sha256(p_record.read_bytes()).hexdigest())]  # fmt: skip
    assert [(k["case"], k["stage"]) for k in cycle["kept_cases"]] == [(P.name, "numerical")]
    assert cycle["kept_cases"][0]["why"].startswith("a scientific failure keeps counting")
    assert cycle["cases_requalified"] == [{"case": FA.name, "stage": "numerical"}]
    assert {e["case"] for e in cycle["attempts"]} == {FA.name}
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    assert verdict["states"][key] == "failing"  # P stays closed
    assert [(e["case"], e["part"]) for e in verdict["superseded"]] == [(FA.name, "baseline")]
    assert verdict["superseded"][0]["supersession"].startswith("superseded by an engineering")
    (p_case,) = [c for c in verdict["cases"] if c["case"]["name"] == P.name]
    assert p_case["kind"] == "scientific" and "superseded" not in p_case
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(FIX))
    qualify.main(["complete", *common(path), "--case", P.name])  # kept: complete, not stale
    with pytest.raises(SystemExit) as stop:
        qualify.main(["complete", *common(path), "--case", FA.name])
    assert stop.value.code == qualify.NO  # F re-qualifies


def test_a_case_with_a_scientific_failure_is_kept_whole(qualify, tmp_path, monkeypatch) -> None:
    """F's first-seed case holds a scientific failure and a passed control; A's case is
    technical. The cycle for A keeps F's case whole, its control included (a rerun of it
    against the old-code launcher run could only be technical): F's rate stays failing, F's case
    complete, and A's case is voided and re-qualifies."""
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    write_case(out, FA, decided=True, control=True)  # scientific, with a passed control
    _, record = technical_adamw(out, tmp_path)
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(FIX))
    qualify.main(supersede_args(path, ADAMW, record))
    cycle = json.loads((out / "c1" / "supersede-1.json").read_text())
    assert sorted((e["part"], e["attempt"]) for e in cycle["kept"] if e["case"] == FA.name) == [
        ("baseline", 1), ("control", 1)]  # fmt: skip
    assert {e["case"] for e in cycle["attempts"]} == {ADAMW.name}
    assert cycle["cases_requalified"] == [{"case": ADAMW.name, "stage": "numerical"}]
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    fa = verdict["rates"][f"adam_frozen-{FA_LR:g}"]
    assert verdict["states"][f"adam_frozen-{FA_LR:g}"] == "failing" and fa["voided"] == []
    control = case_folder(out, FA) / "control-attempt-1.json"
    assert fa["control"]["record_sha256"] == hashlib.sha256(control.read_bytes()).hexdigest()
    assert {e["case"] for e in verdict["superseded"]} == {ADAMW.name}
    qualify.main(["complete", *common(path), "--case", FA.name])  # kept whole: not stale
    with pytest.raises(SystemExit) as stop:
        qualify.main(["complete", *common(path), "--case", ADAMW.name])
    assert stop.value.code == qualify.NO


def test_the_cycle_covers_every_stage_and_old_code_launcher_run(qualify, tmp_path, monkeypatch):
    """Every old-code record of the candidate in all its stage folders, and every old-code
    launcher run in the journal, a run whose case wrote no part attempt included, are named;
    run again with the same record, the cycle writes only the journal's missing lines."""
    path = campaign(tmp_path, fa=(), adamw=(10200, 10201))
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    _, record = technical_adamw(out, tmp_path)
    for label, run_id in ((f"adamw-{ADAMW_LR:g}/s10200", "r-a1"),
                          (f"adamw-{ADAMW_LR:g}/s10201", "r-b1")):  # fmt: skip
        settled_launcher(root, journal(root, path, label, [run_id]), run_id)  # commit abc123
    engineering = q.QualificationCase(10000, "adamw", ADAMW_LR, 1, True)
    publish(out, engineering, "baseline", synthetic(engineering)[0])  # old code, another stage
    monkeypatch.setattr(code_identity, "gpu_code_identity",
                        lambda commit="HEAD", *a, **k: dict(FIX) if commit == "HEAD"
                        else dict(CODE))  # fmt: skip
    qualify.main(supersede_args(path, ADAMW, record))
    cycle = json.loads((out / "c1" / "supersede-1.json").read_text())
    assert [r["run_id"] for r in cycle["launchers"]] == ["r-a1", "r-b1"]
    assert {(c["case"], c["stage"]) for c in cycle["cases_requalified"]} == {
        (ADAMW.name, "numerical"), ("s10201/adamw-5e-05", "numerical"),
        (engineering.name, "engineering")}  # fmt: skip
    log = root / "launch" / "study3-qualification-c1.jsonl"
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["run_id"] for r in lines if r["event"] == "superseded"] == ["r-a1", "r-b1"]
    kept = [
        line
        for line in log.read_text().splitlines()
        if json.loads(line).get("run_id") != "r-b1" or json.loads(line)["event"] != "superseded"
    ]
    log.write_text("\n".join(kept) + "\n")  # stopped before its last line
    qualify.main(supersede_args(path, ADAMW, record))  # the same record: a repair
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["run_id"] for r in lines if r["event"] == "superseded"] == ["r-a1", "r-b1"]


def test_a_cycle_missing_some_lines_names_and_restores_those_alone(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """Written by the real command, the cycle and its lines raise no refusal. With one of its
    two ``superseded`` lines gone (no torn bytes), that one alone is named, the launcher refuses
    with 12, and the cycle run again writes that one alone."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=(), adamw=(10200, 10201))
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    _, record = technical_adamw(out, tmp_path)
    for label, run_id in ((f"adamw-{ADAMW_LR:g}/s10200", "r-a1"),
                          (f"adamw-{ADAMW_LR:g}/s10201", "r-b1")):  # fmt: skip
        settled_launcher(root, journal(root, path, label, [run_id]), run_id)  # commit abc123
    monkeypatch.setattr(code_identity, "gpu_code_identity",
                        lambda commit="HEAD", *a, **k: dict(FIX) if commit == "HEAD"
                        else dict(CODE))  # fmt: skip
    qualify.main(supersede_args(path, ADAMW, record))
    assert launch.missing_lines(root) == []
    name, runs = launch.load_campaign(path)
    assert launch.run_campaign(name, runs, root, dry_run=True)["pending"]
    log = root / "launch" / "study3-qualification-c1.jsonl"
    lost = ("superseded", "r-b1")
    kept = [x for x in log.read_text().splitlines()
            if (json.loads(x)["event"], json.loads(x)["run_id"]) != lost]  # fmt: skip
    log.write_text("".join(x + "\n" for x in kept))
    cycle = "qualification/c1/supersede-1.json"
    assert [(m.record, m.event, m.run_id) for m in launch.missing_lines(root)] == [
        (cycle, "superseded", "r-b1")]  # fmt: skip
    with pytest.raises(launch.TornJournalError):
        launch.run_campaign(name, runs, root, dry_run=True)
    capsys.readouterr()
    qualify.main(supersede_args(path, ADAMW, record))
    assert json.loads(capsys.readouterr().out)["journal_lines_written"] == ["r-b1"]
    assert launch.missing_lines(root) == []


def test_the_cycles_repair_lists_only_the_lines_it_wrote(qualify, tmp_path) -> None:
    """``repair_cycle_journal`` names a run id as written only when ``launch.supersede`` or
    ``launch.hold`` wrote its line: an identical line (the same event, label, experiment, run id
    and reason) under another config hash is not written again, and not listed."""
    root = tmp_path / "runs"
    record = {"diagnosis": "a stale kernel cache", "fix": "the cache is cleared",
              "launchers": [{"run_id": "r-1", "run": "adamw-5e-05/s10200", "experiment": "x",
                             "config_hash": "a" * 12, "campaign": "c1"}],
              "held": [{"run_id": "r-2", "run": "adamw-5e-05/s10201", "experiment": "x",
                        "config_hash": "a" * 12, "campaign": "c1",
                        "reason": "unreadable"}]}  # fmt: skip
    reason = "superseded by an engineering cycle: a stale kernel cache; fix: the cache is cleared"
    log = root / "launch" / "c1.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps({"event": "superseded", "run": "adamw-5e-05/s10200", "run_id": "r-1",
                               "experiment": "x", "config_hash": "b" * 12, "reason": reason,
                               "record": "earlier"}) + "\n")  # fmt: skip
    assert qualify.repair_cycle_journal(tmp_path / "supersede-1.json", record, root) == ["r-2"]
    assert qualify.repair_cycle_journal(tmp_path / "supersede-1.json", record, root) == []
    assert len(log.read_text().splitlines()) == 2


def tree(root: Path) -> dict[str, bytes]:
    """Every file under ``root`` with its bytes."""
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def exit_code(qualify, argv) -> int | str | None:
    """``qualify.main(argv)``'s exit: 0 when it returns, else its ``SystemExit`` code."""
    try:
        qualify.main(argv)
    except SystemExit as stop:
        return stop.code
    return 0


def test_a_torn_superseded_line_is_written_again_by_the_same_cycle(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """The cycle stopped in the middle of its last journal line (``launch``, "The journal"):
    the launcher refuses the campaign; run again with the same record, over the unresolved torn
    line, the cycle writes the missing line once, after the ``torn`` record naming the torn
    bytes, and prints the torn line it leaves unresolved; a call naming another record refuses
    with exit 12 and changes nothing; once the torn line is resolved, the launcher proceeds."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=(), adamw=(10200, 10201))
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    _, record = technical_adamw(out, tmp_path)
    for label, run_id in ((f"adamw-{ADAMW_LR:g}/s10200", "r-a1"),
                          (f"adamw-{ADAMW_LR:g}/s10201", "r-b1")):  # fmt: skip
        settled_launcher(root, journal(root, path, label, [run_id]), run_id)  # commit abc123
    monkeypatch.setattr(code_identity, "gpu_code_identity",
                        lambda commit="HEAD", *a, **k: dict(FIX) if commit == "HEAD"
                        else dict(CODE))  # fmt: skip
    qualify.main(supersede_args(path, ADAMW, record))
    log = root / "launch" / "study3-qualification-c1.jsonl"
    *kept, last = log.read_bytes().splitlines(keepends=True)
    assert (json.loads(last)["event"], json.loads(last)["run_id"]) == ("superseded", "r-b1")
    torn = last[:-13]
    log.write_bytes(b"".join(kept) + torn)
    name, runs = launch.load_campaign(path)
    with pytest.raises(launch.TornJournalError):
        launch.run_campaign(name, runs, root, dry_run=True)
    other = case_folder(out, FA) / "baseline-attempt-1.json"
    before = tree(root)
    assert exit_code(qualify, supersede_args(path, ADAMW, other)) == qualify.TORN_JOURNAL
    assert tree(root) == before
    capsys.readouterr()
    qualify.main(supersede_args(path, ADAMW, record))  # the same record: the missing line
    err = capsys.readouterr().err
    assert "study3-qualification-c1.jsonl line " in err and "--resolve-torn" in err
    qualify.main(supersede_args(path, ADAMW, record))  # and nothing more
    records = launch.launch_records(root)
    assert [r["run_id"] for r in records if r["event"] == "superseded"] == ["r-a1", "r-b1"]
    (named,) = [r for r in records if r["event"] == "torn"]
    assert (named["length"], named["sha256"], named["kind"]) == (
        len(torn), hashlib.sha256(torn).hexdigest(), "superseded")  # fmt: skip
    (line,) = launch.torn_lines(root)
    assert line.repaired and not line.resolved
    with pytest.raises(launch.TornJournalError):
        launch.run_campaign(name, runs, root, dry_run=True)
    launch.resolve_torn(root, line.log, offset=line.offset, sha256=line.sha256,
                        reason="the cycle wrote its missing line again")  # fmt: skip
    result = launch.run_campaign(name, runs, root, dry_run=True)
    assert not result["skipped"] and result["pending"] == [run.label for run in runs]


def test_the_cycle_reads_past_a_resolved_undecodable_line(qualify, tmp_path, monkeypatch):
    """A resolved torn line with bytes that are not UTF-8: the cycle's own reading of the
    journal (``journal_starts``) skips it, as ``launch.launch_records`` does."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=(), adamw=(10200, 10201))
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    _, record = technical_adamw(out, tmp_path)
    for label, run_id in ((f"adamw-{ADAMW_LR:g}/s10200", "r-a1"),
                          (f"adamw-{ADAMW_LR:g}/s10201", "r-b1")):  # fmt: skip
        settled_launcher(root, journal(root, path, label, [run_id]), run_id)  # commit abc123
    log = root / "launch" / "elsewhere.jsonl"
    log.write_bytes(TORN_BYTES + b"\n")
    (line,) = launch.torn_lines(root)
    launch.resolve_torn(root, line.log, offset=line.offset, sha256=line.sha256,
                        reason="nothing was lost: a test line")  # fmt: skip
    monkeypatch.setattr(code_identity, "gpu_code_identity",
                        lambda commit="HEAD", *a, **k: dict(FIX) if commit == "HEAD"
                        else dict(CODE))  # fmt: skip
    qualify.main(supersede_args(path, ADAMW, record))
    cycle = json.loads((out / "c1" / "supersede-1.json").read_text())
    assert [r["run_id"] for r in cycle["launchers"]] == ["r-a1", "r-b1"]


THIRD = {**CODE, "commit": "0" * 40, "sha256": "3" * 64}  # a launcher run's earlier code


def codes(commit="HEAD", *args, **kwargs):
    """The GPU code of each commit: this checkout's is the fix's; the records' (abc123, the
    launcher runs' default) the old one; any other commit, a third code."""
    return dict(FIX if commit == "HEAD" else CODE if commit == "abc123" else THIRD)


def at_commit(run_dir, commit):
    meta = json.loads((run_dir / "meta.json").read_text())
    meta["git"] = {"commit": commit, "dirty": False}
    (run_dir / "meta.json").write_text(json.dumps(meta))


def test_the_cycle_supersedes_every_launcher_run_not_of_the_fix(qualify, tmp_path, monkeypatch):
    """A requalified case's launcher run at a third code (reused from earlier code, its baseline
    recorded technical under the old one): the cycle supersedes it too, as any run not of the
    fix's code, so its recovery round starts a run under the fix."""
    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    _, record = technical_adamw(out, tmp_path)
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    at_commit(settled_launcher(root, run, "r-a1"), "d0d0d0")  # a third code
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)
    monkeypatch.setattr(qualify, "configs_at", commit_configs({"abc123": CONFIG_ROOT,
                                                               "d0d0d0": CONFIG_ROOT}))  # fmt: skip
    qualify.main(supersede_args(path, ADAMW, record))
    cycle = json.loads((out / "c1" / "supersede-1.json").read_text())
    assert [r["run_id"] for r in cycle["launchers"]] == ["r-a1"]
    assert qualify.located(run, root) is None  # superseded: the recovery round starts a new one


def test_the_cycle_keeps_a_case_its_records_show_scientific(qualify, tmp_path, monkeypatch):
    """Kept whole: another stage's case whose baseline fired a declared harness instability rule
    (a complete record with no kind of its own, read with the verdicts), and a case whose
    baseline record classified its launcher run's declared stop scientific (its run is not
    superseded)."""
    path = campaign(tmp_path, fa=(), adamw=(10200, 10201))
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    _, record = technical_adamw(out, tmp_path)
    stopped_case = q.QualificationCase(10201, "adamw", ADAMW_LR, 1, False)
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10201", ["r-b1"])
    launcher = settled_launcher(root, run, "r-b1", stop="gradient_spike", steps=3)
    baseline = synthetic(stopped_case, identity=q.launcher_identity(launcher))[0]
    reason = "instability: the launcher run stopped by 'gradient_spike' after 3 steps"
    baseline.update(status="not_run", kind="scientific", reasons=[reason])
    baseline["launcher"].update(kind="scientific", reasons=[reason])
    publish(out, stopped_case, "baseline", baseline)
    engineering = q.QualificationCase(10000, "adamw", ADAMW_LR, 1, True)
    spiked = synthetic(engineering)[0]
    spiked["trajectories"]["baseline"]["instability"] = {"reason": "gradient_spike", "step": 3,
                                                        "detail": {}}  # fmt: skip
    assert "kind" not in spiked and spiked["status"] == "complete"
    publish(out, engineering, "baseline", spiked)
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)
    qualify.main(supersede_args(path, ADAMW, record))
    cycle = json.loads((out / "c1" / "supersede-1.json").read_text())
    kept = {(k["case"], k["stage"]) for k in cycle["kept_cases"]}
    assert kept == {(engineering.name, "engineering"), (stopped_case.name, "numerical")}
    assert [r["run_id"] for r in cycle["launchers"]] == []  # r-b1 keeps counting
    assert cycle["cases_requalified"] == [{"case": ADAMW.name, "stage": "numerical"}]
    assert qualify.located(run, root) == launcher


def unclassified_stop(qualify, tmp_path, monkeypatch, change=None, *, setup_raised=False):
    """A's technical baseline opens the cycle; s10201's launcher run r-b1 stopped on its declared
    gradient_spike rule and no part record of its case classified it: the Pod stopped before its
    baseline, or (``setup_raised``) the baseline's numerics setup raised before the launcher
    check, and its record, technical, names r-b1 without classifying it. ``change`` edits r-b1's
    records first. Returns the paths, the run and the cycle's record."""
    path = campaign(tmp_path, fa=(), adamw=(10200, 10201))
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    _, record = technical_adamw(out, tmp_path)
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10201", ["r-b1"])
    run_dir = settled_launcher(root, run, "r-b1", stop="gradient_spike", steps=3)
    if change is not None:
        change(run_dir)
    if setup_raised:
        raised_before_the_check(qualify, out, run_dir, monkeypatch)
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)
    qualify.main(supersede_args(path, ADAMW, record))
    return path, root, out, run, json.loads((out / "c1" / "supersede-1.json").read_text())


STOPPED = q.QualificationCase(10201, "adamw", ADAMW_LR, 1, False)


def raised_before_the_check(qualify, out, run_dir, monkeypatch, case=STOPPED):
    """The case's baseline as ``run_part`` publishes it when its numerics setup raises, before
    the launcher check: failed, technical, under the records' code, naming the launcher run
    without classifying it. Nothing loads a model."""
    import torch

    def broken(cfg, device):
        raise RuntimeError("the numerics setup failed")

    with monkeypatch.context() as patch:
        patch.setattr(q.runtime, "configure_numerics", broken)
        record = q.run_part(qualify.load_settings(), SimpleNamespace(device=torch.device("cpu")),
                            case, run_dir, out, part="baseline", attempt=1,
                            checkpoints=out / "harness")  # fmt: skip
    launcher = record["launcher"]
    assert (record["status"], record["kind"]) == ("failed", "technical")
    assert (launcher["identity"]["run_id"], launcher["kind"], launcher["classified"]) == (
        run_dir.name, None, False)  # fmt: skip
    return record


def test_a_conforming_unclassified_stop_is_kept_scientific(qualify, tmp_path, monkeypatch):
    """Resolved against its issued cell, its config, numerics and commit, the run conforms and
    its declared rule stopped it: a scientific failure. The case is kept whole and decided, the
    run keeps counting, no part runs for it, and the rate fails."""
    path, root, out, run, cycle = unclassified_stop(qualify, tmp_path, monkeypatch)
    (kept,) = cycle["kept_cases"]
    assert (kept["case"], kept["resolution"]["outcome"]) == (STOPPED.name, "scientific")
    assert "stopped by its declared rule 'gradient_spike'" in kept["why"]
    assert cycle["launchers"] == [] and cycle["held"] == []
    assert cycle["cases_requalified"] == [{"case": ADAMW.name, "stage": "numerical"}]
    assert qualify.located(run, root) == root / run.experiment / "r-b1"
    qualify.main(["complete", *common(path), "--case", STOPPED.name])  # decided: complete
    with pytest.raises(SystemExit, match="no part runs"):
        qualify.main(["part", *common(path), "--case", STOPPED.name, "--part", "baseline",
                      "--resume", "--run-dir", str(tmp_path / "run"), "--checkpoints",
                      str(tmp_path / "h"), "--device", "cpu"])  # fmt: skip
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    assert verdict["states"][f"adamw-{ADAMW_LR:g}"] == "failing"
    (resolved,) = [c for c in verdict["cases"] if c.get("resolution")]
    assert (resolved["case"]["name"], resolved["kind"]) == (STOPPED.name, "scientific")


def test_a_record_that_raised_before_its_check_leaves_the_stop_to_the_resolver(
    qualify, tmp_path, monkeypatch
) -> None:
    """s10201's baseline raised in its numerics setup, before the launcher check: its record
    (technical, of the old code) names r-b1 but did not classify it, so the cycle resolves r-b1
    on its own. A conforming gradient_spike stop: scientific. The case is kept whole, its
    record included, r-b1 keeps counting, the case is decided and the rate fails."""
    path, root, out, run, cycle = unclassified_stop(qualify, tmp_path, monkeypatch,
                                                    setup_raised=True)  # fmt: skip
    (kept,) = cycle["kept_cases"]
    assert (kept["case"], kept["resolution"]["outcome"]) == (STOPPED.name, "scientific")
    assert "stopped by its declared rule 'gradient_spike'" in kept["why"]
    assert [(e["case"], e["part"], e["attempt"]) for e in cycle["kept"]] == [
        (STOPPED.name, "baseline", 1)]  # fmt: skip
    assert cycle["launchers"] == [] and cycle["held"] == []
    assert cycle["cases_requalified"] == [{"case": ADAMW.name, "stage": "numerical"}]
    assert {e["case"] for e in cycle["attempts"]} == {ADAMW.name}
    assert qualify.located(run, root) == root / run.experiment / "r-b1"
    qualify.main(["complete", *common(path), "--case", STOPPED.name])  # decided: complete
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    assert verdict["states"][f"adamw-{ADAMW_LR:g}"] == "failing"
    assert STOPPED.name not in {e["case"] for e in verdict["superseded"]}
    (resolved,) = [c for c in verdict["cases"] if c.get("resolution")]
    assert (resolved["case"]["name"], resolved["kind"]) == (STOPPED.name, "scientific")


def test_another_stages_unclassified_stop_is_resolved_too(qualify, tmp_path, monkeypatch):
    """An engineering case, of a stage the numerical campaign does not hold, whose baseline
    raised before the launcher check: the numerical cycle resolves its launcher run r-e1 from
    the case its record names and the run's journal start. A conforming gradient_spike keeps
    the case whole (scientific), and r-e1 is not superseded."""
    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    _, record = technical_adamw(out, tmp_path)
    engineering = q.QualificationCase(10000, "adamw", ADAMW_LR, 1, True)
    stage = campaign(tmp_path, fa=(), adamw=(10000,), name="study3-engineering-c1",
                     role="engineering")  # fmt: skip
    run = journal(root, stage, f"adamw-{ADAMW_LR:g}/s10000", ["r-e1"])
    run_dir = settled_launcher(root, run, "r-e1", stop="gradient_spike", steps=3)
    raised_before_the_check(qualify, out, run_dir, monkeypatch, engineering)
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)
    qualify.main(supersede_args(path, ADAMW, record))
    cycle = json.loads((out / "c1" / "supersede-1.json").read_text())
    (kept,) = cycle["kept_cases"]
    assert (kept["case"], kept["stage"], kept["resolution"]["outcome"]) == (
        engineering.name, "engineering", "scientific")  # fmt: skip
    assert [(e["case"], e["part"]) for e in cycle["kept"]] == [(engineering.name, "baseline")]
    assert cycle["launchers"] == [] and cycle["held"] == []
    assert cycle["cases_requalified"] == [{"case": ADAMW.name, "stage": "numerical"}]
    log = root / "launch" / "study3-engineering-c1.jsonl"
    assert [json.loads(line)["event"] for line in log.read_text().splitlines()] == ["start"]


def other_rate(run_dir):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(run_dir / "config.yaml")
    OmegaConf.update(cfg, "optimizer.lr", 9.0e-05)
    OmegaConf.save(cfg, run_dir / "config.yaml")


def other_numerics(run_dir):
    meta = json.loads((run_dir / "meta.json").read_text())
    meta["inputs"]["numerics_setup"]["deterministic"] = True
    (run_dir / "meta.json").write_text(json.dumps(meta))


@pytest.mark.parametrize("change", [other_rate, other_numerics])
def test_a_nonconforming_unclassified_stop_is_superseded(qualify, tmp_path, monkeypatch, change):
    """The run departs from its issued cell on readable records (another rate in its config,
    or numerics its config does not ask for): its stop is no registered outcome. It is
    superseded with the cycle and its case re-qualified."""
    path, root, out, run, cycle = unclassified_stop(qualify, tmp_path, monkeypatch, change)
    assert cycle["kept_cases"] == [] and cycle["held"] == []
    assert [r["run_id"] for r in cycle["launchers"]] == ["r-b1"]
    assert {c["case"] for c in cycle["cases_requalified"]} == {ADAMW.name, STOPPED.name}
    assert qualify.located(run, root) is None


def edited_meta(change):
    def edit(run_dir):
        meta = json.loads((run_dir / "meta.json").read_text())
        change(meta)
        (run_dir / "meta.json").write_text(json.dumps(meta))

    return edit


def unreadable_config(run_dir):
    (run_dir / "config.yaml").write_text("seed: [10201\n")


UNKNOWN_COMMIT = {"commit": "0" * 40, "dirty": False}  # a commit not in the clone
NOT_COMPOSED = "cannot be composed from the configs of its commit"
UNRESOLVABLE = {  # what is missing → why the cycle says so
    "no commit": (edited_meta(lambda meta: meta.pop("git")), "no recorded commit"),
    "an unknown commit": (edited_meta(lambda meta: meta.update(git=UNKNOWN_COMMIT)), NOT_COMPOSED),
    "no header": (edited_meta(lambda meta: meta.pop("campaign")), "no campaign header"),
    "an unreadable config": (unreadable_config, "its records are unreadable"),
}


@pytest.mark.parametrize("missing", list(UNRESOLVABLE))
def test_an_unresolvable_stop_is_kept_and_held(qualify, tmp_path, monkeypatch, missing) -> None:
    """A run whose provenance is incomplete cannot be resolved against what it was issued (no
    commit in its meta.json, a commit not in this clone, no campaign header, a saved config that
    does not read): the case is kept as it is, listed unresolved, its run held, and no rerun is
    opened: no part runs, and the launcher's recovery round refuses the run."""
    from zero_field_refusal import launch

    change, why = UNRESOLVABLE[missing]
    path, root, out, run, cycle = unclassified_stop(qualify, tmp_path, monkeypatch, change)
    (kept,) = cycle["kept_cases"]
    assert (kept["case"], kept["resolution"]["outcome"]) == (STOPPED.name, "unresolved")
    assert why in kept["why"]
    assert [h["run_id"] for h in cycle["held"]] == ["r-b1"] and cycle["launchers"] == []
    log = root / "launch" / "study3-qualification-c1.jsonl"
    assert [json.loads(line)["event"] for line in log.read_text().splitlines()][-1] == "held"
    with pytest.raises(SystemExit, match="no part runs"):
        qualify.main(["part", *common(path), "--case", STOPPED.name, "--part", "baseline",
                      "--attempt", "1", "--run-dir", str(tmp_path / "run"), "--checkpoints",
                      str(tmp_path / "h"), "--device", "cpu"])  # fmt: skip
    with pytest.raises(SystemExit, match="no rerun is opened"):
        qualify.main(restart(path, STOPPED, RESTART))
    name, runs = launch.load_campaign(path)
    with pytest.raises(launch.CampaignError, match="held, its outcome unresolved"):
        launch.run_campaign(name, runs, root, execute=lambda *a: 0, recovery="a reason",
                            only=run.label, max_attempts=1)  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        qualify.main(["complete", *common(path), "--case", STOPPED.name])
    assert stop.value.code == qualify.HELD  # the driver stops before the launcher
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    assert verdict["states"][f"adamw-{ADAMW_LR:g}"] == "not_qualified"
    assert any("outcome is unresolved" in w for w in verdict["warnings"])


@pytest.mark.parametrize(("issued_gb", "outcome"), [(45, "scientific"), (46, "technical")])
def test_a_stop_is_judged_against_the_configs_of_its_commit(
    qualify, tmp_path, monkeypatch, issued_gb, outcome
) -> None:
    """The configs of the run's commit asked for a 45 GB memory limit, this checkout's ask for
    46 (the memory limit is outside the config hash). A run made with 45 GB, as its commit
    issued it, conforms: its declared stop is scientific, kept, whatever this checkout says. A
    run made with 46 GB departs from what it was issued: superseded."""
    from omegaconf import OmegaConf

    from zero_field_refusal.launch import expected_numerics
    from zero_field_refusal.protocol import load_run_config

    then = tmp_path / "configs-of-the-commit"
    shutil.copytree(CONFIG_ROOT, then)
    study3 = then / "experiments" / "study3.yaml"
    assert "memory_limit_gb: 46 " in study3.read_text()
    study3.write_text(study3.read_text().replace("memory_limit_gb: 46 ", "memory_limit_gb: 45 "))
    monkeypatch.setattr(qualify, "configs_at", commit_configs({"abc123": then}))

    def made_with(run_dir):
        cfg = OmegaConf.load(run_dir / "config.yaml")
        OmegaConf.update(cfg, "execution.memory_limit_gb", issued_gb)
        OmegaConf.save(cfg, run_dir / "config.yaml")
        edited_meta(lambda meta: meta["inputs"]["numerics_setup"].update(
            memory_limit_gb=float(issued_gb)))(run_dir)  # fmt: skip

    path, root, out, run, cycle = unclassified_stop(qualify, tmp_path, monkeypatch, made_with)
    now = expected_numerics(load_run_config(run.config, run.overrides))
    assert now["memory_limit_gb"] == 46.0  # this checkout's
    if outcome == "scientific":
        (kept,) = cycle["kept_cases"]
        assert (kept["case"], kept["resolution"]["outcome"]) == (STOPPED.name, "scientific")
        assert cycle["launchers"] == [] and cycle["held"] == []
        assert qualify.located(run, root) == root / run.experiment / "r-b1"
    else:
        assert cycle["kept_cases"] == [] and cycle["held"] == []
        assert [r["run_id"] for r in cycle["launchers"]] == ["r-b1"]
        assert qualify.located(run, root) is None


def issued_earlier(qualify, tmp_path, monkeypatch, change=None):
    """The fix changed a hashed config key (the teacher's chunk size): s10201's launcher run
    r-old, a gradient_spike issued at commit abc123 under the earlier configs, has another config
    hash than the campaign's cell now; A's technical baseline opens the cycle. ``change`` edits
    r-old's records first. Returns the campaign, the roots, the run, the campaign's name, the
    technical record and the journal."""
    from omegaconf import OmegaConf

    from zero_field_refusal.config import config_hash
    from zero_field_refusal.launch import load_campaign

    then = tmp_path / "configs-of-the-commit"
    shutil.copytree(CONFIG_ROOT, then)
    study3 = then / "experiments" / "study3.yaml"
    assert study3.read_text().count("  chunk_size: 1000\n") == 1
    study3.write_text(study3.read_text().replace("  chunk_size: 1000\n", "  chunk_size: 500\n"))
    monkeypatch.setattr(qualify, "configs_at", commit_configs({"abc123": then}))
    path = campaign(tmp_path, fa=(), adamw=(10200, 10201))
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    _, record = technical_adamw(out, tmp_path)
    name, runs = load_campaign(path)
    run = next(r for r in runs if r.label == f"adamw-{ADAMW_LR:g}/s10201")
    issued = load_config(then / "experiments" / "study3_k_only.yaml", run.overrides, root=then)
    old_hash = config_hash(issued)
    assert old_hash != run.config_hash  # the cell's hash now
    run_dir = settled_launcher(root, run, "r-old", stop="gradient_spike", steps=3)
    OmegaConf.save(issued, run_dir / "config.yaml")
    edited_meta(lambda meta: meta.update(config_hash=old_hash))(run_dir)
    if change is not None:
        change(run_dir)
    start = {"event": "start", "run": run.label, "attempt": 1, "run_id": "r-old",
             "experiment": run.experiment, "config_hash": old_hash,
             "command": run.command()[1:]}  # fmt: skip
    log = root / "launch" / f"{name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(start) + "\n")
    assert qualify.journal_run_ids(run, root) == []  # not the cell's config hash
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)
    return path, root, out, run, name, record, log


def test_a_stop_under_an_earlier_config_hash_is_still_found_and_kept(
    qualify, tmp_path, monkeypatch
) -> None:
    """r-old (``issued_earlier``) conforms to what it was issued. The cycle finds it by its
    label, resolves it against the configs of its own commit (scientific) and keeps its case
    whole: r-old is not superseded, and the case is decided."""
    path, root, out, run, name, record, log = issued_earlier(qualify, tmp_path, monkeypatch)
    qualify.main(supersede_args(path, ADAMW, record))
    cycle = json.loads((out / "c1" / "supersede-1.json").read_text())
    (kept,) = cycle["kept_cases"]
    assert (kept["case"], kept["resolution"]["run_id"], kept["resolution"]["outcome"]) == (
        STOPPED.name, "r-old", "scientific")  # fmt: skip
    assert cycle["launchers"] == [] and cycle["held"] == []
    assert cycle["cases_requalified"] == [{"case": ADAMW.name, "stage": "numerical"}]
    assert [json.loads(line)["event"] for line in log.read_text().splitlines()] == ["start"]
    qualify.main(["complete", *common(path), "--case", STOPPED.name])  # decided: complete
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    assert verdict["states"][f"adamw-{ADAMW_LR:g}"] == "failing"


def test_a_hold_under_an_earlier_config_hash_stops_its_label(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """r-old (``issued_earlier``) records no commit: the cycle holds it, under its earlier
    config hash. The hold applies to its label: under the cell's hash now, the recovery round
    (``--only``), a whole campaign and a dry run are refused before anything starts, naming
    r-old and the hold's reason, and no child process starts; ``complete`` exits ``HELD``
    naming the hold, so the driver stops before the launcher."""
    from zero_field_refusal import launch

    no_commit = edited_meta(lambda meta: meta.pop("git"))
    path, root, out, run, name, record, log = issued_earlier(qualify, tmp_path, monkeypatch,
                                                             no_commit)  # fmt: skip
    qualify.main(supersede_args(path, ADAMW, record))
    cycle = json.loads((out / "c1" / "supersede-1.json").read_text())
    assert [(h["run_id"], h["config_hash"] != run.config_hash) for h in cycle["held"]] == [
        ("r-old", True)]  # fmt: skip
    assert [json.loads(line)["event"] for line in log.read_text().splitlines()] == [
        "start", "held"]  # fmt: skip
    _, runs = launch.load_campaign(path)
    before, children = log.read_text(), []

    def execute(*args):
        children.append(args)
        return 0

    refused = r"r-old: held, its outcome unresolved: no recorded commit"
    for kwargs in ({"recovery": "a reason", "only": run.label, "max_attempts": 1}, {},
                   {"dry_run": True}, {"dry_run": True, "only": run.label}):  # fmt: skip
        with pytest.raises(launch.CampaignError, match=refused):
            launch.run_campaign(name, runs, root, execute=execute, **kwargs)
    assert children == [] and log.read_text() == before
    capsys.readouterr()
    with pytest.raises(SystemExit) as stop:
        qualify.main(["complete", *common(path), "--case", STOPPED.name])
    assert stop.value.code == qualify.HELD
    assert "r-old" in capsys.readouterr().err


def zero_byte_journal(monkeypatch) -> None:
    """Every append to a launch log writes no byte (``os.write`` returns 0 for a log's
    descriptor, and writes any other file): the window between a published record and its
    journal lines."""
    real = os.write

    def write(fd, data):
        target = os.readlink(f"/proc/self/fd/{fd}")
        if f"{os.sep}launch{os.sep}" in target and target.endswith(".jsonl"):
            return 0
        return real(fd, data)

    monkeypatch.setattr(os, "write", write)


def test_a_cycle_whose_lines_wrote_no_byte_blocks_until_it_is_run_again(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """The cycle holds r-old, issued under the label's earlier config hash; its record is
    published and its ``held`` line wrote no byte: no torn bytes, and still the launcher starts
    nothing (without the line, the label would get a new attempt under its hash now), every
    command but the repair refuses with 12 naming the missing line, and nothing changes. The
    cycle run again with its record writes the line; the launcher then refuses the label as
    held, and ``complete`` exits ``HELD``."""
    from zero_field_refusal import launch

    no_commit = edited_meta(lambda meta: meta.pop("git"))
    path, root, out, run, name, record, log = issued_earlier(qualify, tmp_path, monkeypatch,
                                                             no_commit)  # fmt: skip
    with monkeypatch.context() as patched:
        zero_byte_journal(patched)
        with pytest.raises(OSError, match="short write"):
            qualify.main(supersede_args(path, ADAMW, record))
    assert (out / "c1" / "supersede-1.json").is_file()
    assert [json.loads(x)["event"] for x in log.read_text().splitlines()] == ["start"]
    assert launch.torn_lines(root) == []
    _, runs = launch.load_campaign(path)
    children = []

    def execute(*args):
        children.append(args)
        return 0

    missing = [("qualification/c1/supersede-1.json", "held", "r-old")]
    for kwargs in ({"recovery": "a reason", "only": run.label, "max_attempts": 1},
                   {"only": run.label, "max_attempts": 1}, {}, {"dry_run": True}):  # fmt: skip
        with pytest.raises(launch.TornJournalError) as refusal:
            launch.run_campaign(name, runs, root, execute=execute, **kwargs)
        assert [(m.record, m.event, m.run_id) for m in refusal.value.missing] == missing
    assert children == []
    before = tree(root)
    for argv in (["complete", *common(path), "--case", STOPPED.name],
                 ["locate", *common(path), "--case", STOPPED.name], ["verdict", *common(path)],
                 restart(path, STOPPED, RESTART), supersede_args(path, ADAMW, path)):  # fmt: skip
        capsys.readouterr()
        assert exit_code(qualify, argv) == qualify.TORN_JOURNAL, argv[0]
        assert "qualification/c1/supersede-1.json: no held line of r-old" in capsys.readouterr().err
    assert tree(root) == before
    qualify.main(supersede_args(path, ADAMW, record))  # the repair: its missing line
    assert [json.loads(x)["event"] for x in log.read_text().splitlines()] == ["start", "held"]
    assert launch.missing_lines(root) == []
    for kwargs in ({"recovery": "a reason", "only": run.label, "max_attempts": 1}, {}):
        with pytest.raises(launch.CampaignError, match="r-old: held, its outcome unresolved"):
            launch.run_campaign(name, runs, root, execute=execute, **kwargs)
    assert children == []
    assert exit_code(qualify, ["complete", *common(path), "--case", STOPPED.name]) == qualify.HELD


def test_the_driver_stops_at_a_held_label_before_the_launcher(qualify, tmp_path, monkeypatch):
    """``complete`` exits ``HELD`` for the first case: the driver stops there, its verdict
    written, and neither the launcher nor a part is called."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, complete=qualify.HELD)
    assert code == qualify.HELD
    assert calls == [f"complete {FA.name}", "verdict"]


def test_configs_at_reads_a_commit_from_git_not_the_working_tree(tmp_path) -> None:
    """``configs_at`` unpacks the committed configs of a commit (here HEAD's, as git has them)
    into a temporary folder it removes on exit; a commit not in the clone, or no commit id,
    raises."""
    configs_at = load_script().configs_at
    committed = subprocess.run(["git", "show", "HEAD:configs/experiments/study3.yaml"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout  # fmt: skip
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip
    with configs_at(head) as root:
        assert root.parent != CONFIG_ROOT.parent and root.name == "configs"
        assert (root / "experiments" / "study3.yaml").read_text() == committed
    assert not root.exists()
    with pytest.raises(subprocess.CalledProcessError), configs_at("0" * 40):
        pass
    with pytest.raises(ValueError, match="not a commit id"), configs_at("HEAD; rm -rf /"):
        pass


def test_a_cycle_cannot_start_from_a_case_its_launcher_stop_keeps(qualify, tmp_path, monkeypatch):
    """The technical record's own case has a conforming launcher run its declared rule stopped,
    which no part record classified: the cycle would keep that case whole, so it cannot start
    from it."""
    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    _, record = technical_adamw(out, tmp_path)  # its records name another run (r1)
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    settled_launcher(root, run, "r-a1", stop="gradient_spike", steps=3)
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)
    with pytest.raises(SystemExit, match="the cycle would keep this case whole"):
        qualify.main(supersede_args(path, ADAMW, record))
    assert not (out / "c1" / "supersede-1.json").exists()


def test_a_cycle_cannot_start_from_a_case_it_keeps_whole(qualify, tmp_path, monkeypatch) -> None:
    path = campaign(tmp_path, adamw=())
    out = tmp_path / "runs" / "qualification"
    write_case(out, FA, decided=True)  # scientific
    publish(out, FA, "control", control_record(0.1, case=FA))  # technical
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)
    record = case_folder(out, FA) / "control-attempt-1.json"
    with pytest.raises(SystemExit, match="keeps the case whole"):
        qualify.main(supersede_args(path, FA, record))
    assert not (out / "c1" / "supersede-1.json").exists()


def test_a_checkpoint_finish_removed_is_no_evidence_of_an_interruption(qualify, tmp_path) -> None:
    """``finish`` records the checkpoints it removes; a restart does not take their absence for
    a lost container disk."""
    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    launcher = settled_launcher(root, run, "r-a1")
    checkpoint = tmp_path / "ckpt" / "r-a1.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"w")
    for name in ("meta.json", "summary.json"):
        text = (launcher / name).read_text().replace('"/x.pt"', json.dumps(str(checkpoint)))
        (launcher / name).write_text(text)
    baseline, rest = synthetic(ADAMW, identity=q.launcher_identity(launcher))
    publish(out, ADAMW, "baseline", baseline)
    rest["baseline_record"]["sha256"] = q.record_sha256(baseline)
    publish(out, ADAMW, "rest", rest)
    qualify.main(["finish", *common(path), "--case", ADAMW.name])
    assert not checkpoint.exists()
    (marker,) = case_folder(out, ADAMW).glob("checkpoints-removed-*.json")
    assert str(checkpoint) in json.loads(marker.read_text())["removed"]
    assert qualify.checkpoint_missing(launcher) is True  # gone from disk,
    assert qualify.checkpoint_missing(launcher, case_folder(out, ADAMW)) is False  # on purpose
    (out / ADAMW.attempt_path("rest", 1)).unlink()  # the case no longer complete
    with pytest.raises(SystemExit, match="no evidence of an interruption"):
        qualify.main(restart(path, ADAMW, RESTART))


def test_a_lost_checkpoint_restarts_a_case_before_any_attempt(qualify, tmp_path) -> None:
    """The Pod stopped between a finished launch and the baseline's start: no attempt, and the
    launcher run's checkpoint is gone. The restart supersedes the launcher run, so the recovery
    round can start a new one; with the checkpoint there, there is nothing to restart."""
    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    launcher = settled_launcher(root, run, "r-a1")  # its checkpoint /x.pt is gone
    qualify.main(restart(path, ADAMW, RESTART))
    record = json.loads((case_folder(out, ADAMW) / "restart-1.json").read_text())
    assert record["attempts"] == [] and record["checkpoint_missing"] is True
    assert record["launcher"]["run_id"] == launcher.name
    log = root / "launch" / "study3-qualification-c1.jsonl"
    assert json.loads(log.read_text().splitlines()[-1])["event"] == "superseded"
    with pytest.raises(SystemExit, match="no attempt to restart"):  # the run no longer counts
        qualify.main(restart(path, ADAMW, RESTART))


def test_a_supersession_needs_a_technical_failure(qualify, tmp_path) -> None:
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    write_case(out, ADAMW)  # qualified
    record = case_folder(out, ADAMW) / "baseline-attempt-1.json"
    with pytest.raises(SystemExit, match="not a completed technical failure"):
        qualify.main(["supersede", *common(path), "--case", ADAMW.name, "--record", str(record),
                      "--diagnosis", "d", "--fix", "f"])  # fmt: skip
    with pytest.raises(SystemExit, match="not a recorded attempt of this case"):
        qualify.main(["supersede", *common(path), "--case", FA.name, "--record", str(record),
                      "--diagnosis", "d", "--fix", "f"])  # fmt: skip


def trajectory_result(case, rest, name, attempt=1):
    """A rest trajectory's published result, bound as ``run_part`` binds it."""
    return {"part": "rest", "attempt": attempt, "case": case.record(), "trajectory": name,
            "K_star": 6.0, "bound": 0.06, "launcher": rest["launcher"], "numerics_setup": None,
            "baseline_record": rest["baseline_record"],
            "baseline_checkpoint_sha256": rest["baseline_checkpoint_sha256"],
            "launcher_checkpoint_sha256": IDENTITY["checkpoint"]["sha256"],
            "run": rest["trajectories"][name],
            "comparisons": {k: v for k, v in rest["comparisons"].items()
                            if k.endswith(f"~{name}")},
            "bridge": rest["bridge"], "v0_unchanged": True,
            "provenance": rest["provenance"]}  # fmt: skip


def test_a_scientific_trajectory_result_survives_an_interruption(qualify, stubbed, tmp_path):
    """A rest interrupted after a scientific reference (its result published, the part's record
    not): the verdict reads the result, the rate fails, and neither a restart nor a rerun passes
    it; ``interrupted`` records the attempt with the result, which keeps counting."""
    path = campaign(tmp_path, adamw=())
    out = tmp_path / "runs" / "qualification"
    baseline, rest = synthetic(FA)
    publish(out, FA, "baseline", baseline)
    rest["baseline_record"]["sha256"] = q.record_sha256(baseline)
    rest["comparisons"]["baseline~reference"]["D_twin"] = 0.5  # beyond 0.06: scientific
    q.publish(out / (FA.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    for name in ("repeat", "reference"):
        q.publish(out / FA.trajectory_path(1, name), trajectory_result(FA, rest, name))
    key = f"adam_frozen-{FA_LR:g}"
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    assert verdict["states"][key] == "failing"
    (case,) = verdict["cases"]
    assert case["kind"] == "scientific"
    assert any(r.startswith("baseline~reference: D₂₅ 0.5") for r in case["reasons"])
    with pytest.raises(SystemExit, match=r"rest attempt 1 \(its trajectory results\).*scientific"):
        qualify.main(restart(path, FA, RESTART))
    with pytest.raises(SystemExit, match="trajectory results hold a scientific failure"):
        qualify.main(part_args(path, tmp_path, "rest", "--attempt", "2", "--infrastructure", "x"))
    assert stubbed == []
    qualify.main(["interrupted", *common(path), "--case", FA.name, "--part", "rest",
                  "--attempt", "1", "--technical", "the process was killed"])  # fmt: skip
    record = json.loads((out / FA.attempt_path("rest", 1)).read_text())
    assert record["from_trajectories"] == ["repeat", "reference"]
    assert record["comparisons"]["baseline~reference"]["D_twin"] == 0.5
    v = q.case_verdict(baseline, record, SCOPE)
    assert (v["state"], v["kind"]) == ("failing", "scientific")
    qualify.main(["verdict", *common(path)])
    assert json.loads((out / "c1" / "verdict.json").read_text())["states"][key] == "failing"


def settled_launcher(root, run, run_id, *, stop="steps", steps=25):
    """A settled launcher run of ``run`` as ``launcher_problems`` reads it (its config, identity
    inputs, terminal record and checkpoint record; the checkpoint file itself, /x.pt, is gone)."""
    from test_launch import make_run_dir

    from zero_field_refusal.launch import expected_numerics
    from zero_field_refusal.protocol import load_run_config

    run_dir = make_run_dir(root, run, run_id, "completed")
    meta = json.loads((run_dir / "meta.json").read_text())
    numerics = {
        **expected_numerics(load_run_config(run.config, run.overrides)),
        "cublas_workspace_config": None,
        "memory_fraction": None,
        "device_index": 0,
    }
    meta["inputs"] = {"numerics_setup": numerics, "theta0_trainable_sha256": "t" * 64,
                      "endpoint_checkpoint": IDENTITY["checkpoint"]}  # fmt: skip
    meta["campaign"] = {"name": "study3-qualification-c1", "label": run.label, "attempt": 1,
                        "header": None}  # fmt: skip
    (run_dir / "meta.json").write_text(json.dumps(meta))
    terminal = {"stop": {"reason": stop, "detail": {}}, "steps": steps, "evaluations": [25]}
    (run_dir / "terminal.json").write_text(json.dumps(terminal))
    (run_dir / "summary.json").write_text(json.dumps(
        {"endpoint_checkpoint": IDENTITY["checkpoint"], "terminal": terminal}))  # fmt: skip
    return run_dir


@pytest.mark.parametrize(("stop", "kind"), [("gradient_spike", "scientific"),
                                            ("t_max", "technical")])  # fmt: skip
def test_a_restart_never_voids_a_failed_launcher_run(qualify, tmp_path, stop, kind) -> None:
    """The launcher run stopped by its declared instability rule (a scientific failure), or by
    another stop (technical), its baseline interrupted before it published: no restart."""
    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    settled_launcher(root, run, "r-a1", stop=stop, steps=3)
    q.publish(out / (ADAMW.attempt_path("baseline", 1) + ".started"), {"started_at": "t"})
    with pytest.raises(SystemExit, match=f"is itself a completed {kind} failure"):
        qualify.main(restart(path, ADAMW, RESTART))
    assert not list(case_folder(out, ADAMW).glob("restart-*"))


def test_a_restart_needs_evidence_of_an_interruption(qualify, tmp_path, monkeypatch) -> None:
    """Passing recorded attempts, none interrupted, the launcher run's checkpoint on disk: there
    is nothing to restart from; with the checkpoint gone, there is."""
    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    launcher = settled_launcher(root, run, "r-a1")
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.write_bytes(b"w")
    for name in ("meta.json", "summary.json"):  # its checkpoint on disk
        text = (launcher / name).read_text().replace('"/x.pt"', json.dumps(str(checkpoint)))
        (launcher / name).write_text(text)
    baseline, _ = synthetic(ADAMW, identity=q.launcher_identity(launcher))
    publish(out, ADAMW, "baseline", baseline)
    with pytest.raises(SystemExit, match="no evidence of an interruption"):
        qualify.main(restart(path, ADAMW, RESTART))
    checkpoint.unlink()  # the container disk lost
    seen = []
    real = qualify.supersede

    def supersede(*args, **kwargs):  # the journal's line comes after the record it points to
        seen.append(Path(kwargs["record"]).is_file())
        return real(*args, **kwargs)

    monkeypatch.setattr(qualify, "supersede", supersede)
    qualify.main(restart(path, ADAMW, RESTART))
    record = json.loads((case_folder(out, ADAMW) / "restart-1.json").read_text())
    assert record["checkpoint_missing"] is True and record["launcher"]["run_id"] == "r-a1"
    assert seen == [True]
    log = root / "launch" / "study3-qualification-c1.jsonl"
    lines = log.read_text().splitlines()
    assert json.loads(lines[-1])["event"] == "superseded"
    log.write_text("\n".join(lines[:-1]) + "\n")  # stopped between the record and its line
    with pytest.raises(SystemExit, match="the launch journal now names its launcher run"):
        qualify.main(restart(path, ADAMW, RESTART))  # it completes the journal only
    assert json.loads(log.read_text().splitlines()[-1])["run_id"] == "r-a1"
    assert seen == [True, True] and not (case_folder(out, ADAMW) / "restart-2.json").exists()


def test_a_restart_rewrites_its_lost_journal_line_over_a_torn_line(
    qualify, tmp_path, capsys
) -> None:
    """``restart-case`` with a published restart record whose ``superseded`` line was torn:
    run again, over the unresolved torn line, it writes only that line (exit 1, its message)
    and prints the torn line it leaves; a new restart over a torn line refuses with exit 12 and
    changes nothing; the launcher refuses until the resolution."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    launcher = settled_launcher(root, run, "r-a1")  # its checkpoint is gone: the evidence
    publish(out, ADAMW, "baseline", synthetic(ADAMW, identity=q.launcher_identity(launcher))[0])
    qualify.main(restart(path, ADAMW, RESTART))
    log = root / "launch" / "study3-qualification-c1.jsonl"
    *kept, last = log.read_bytes().splitlines(keepends=True)
    assert (json.loads(last)["event"], json.loads(last)["run_id"]) == ("superseded", "r-a1")
    torn = last[:-13]
    log.write_bytes(b"".join(kept) + torn)
    name, runs = launch.load_campaign(path)
    with pytest.raises(launch.TornJournalError):
        launch.run_campaign(name, runs, root, dry_run=True)
    capsys.readouterr()
    with pytest.raises(SystemExit, match="the launch journal now names its launcher run") as stop:
        qualify.main(restart(path, ADAMW, "any text: the record's reason is written"))
    assert stop.value.code != qualify.TORN_JOURNAL
    err = capsys.readouterr().err
    assert "study3-qualification-c1.jsonl line " in err and "--resolve-torn" in err
    (line,) = [r for r in launch.launch_records(root) if r["event"] == "superseded"]
    assert (line["run_id"], line["reason"]) == ("r-a1", RESTART)
    (named,) = [r for r in launch.launch_records(root) if r["event"] == "torn"]
    assert (named["sha256"], named["kind"]) == (hashlib.sha256(torn).hexdigest(), "superseded")
    assert not (case_folder(out, ADAMW) / "restart-2.json").exists()
    before = tree(root)
    assert exit_code(qualify, restart(path, ADAMW, RESTART)) == qualify.TORN_JOURNAL  # a new one
    assert tree(root) == before
    with pytest.raises(launch.TornJournalError):
        launch.run_campaign(name, runs, root, dry_run=True)
    (torn_line,) = launch.torn_lines(root)
    launch.resolve_torn(root, torn_line.log, offset=torn_line.offset, sha256=torn_line.sha256,
                        reason="restart-case wrote its missing line again")  # fmt: skip
    assert launch.run_campaign(name, runs, root, dry_run=True)["pending"] == [runs[0].label]


def test_a_restart_after_a_config_hash_change_is_recorded(qualify, tmp_path, capsys) -> None:
    """A restart's ``superseded`` line written under the label's earlier config hash still names
    its run (by label, experiment and run id, as ``launch.supersede`` judges a line): a new
    ``restart-case`` is no journal repair and records the new restart as the normal flow does;
    a published restart whose line really is missing is still repaired."""
    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    label = f"adamw-{ADAMW_LR:g}/s10200"
    run = journal(root, path, label, ["r-a1"])
    launcher = settled_launcher(root, run, "r-a1")
    publish(out, ADAMW, "baseline", synthetic(ADAMW, identity=q.launcher_identity(launcher))[0])
    qualify.main(restart(path, ADAMW, RESTART))
    earlier = "0" * 12  # the label's config hash before a fix that changed a hashed key
    log = root / "launch" / "study3-qualification-c1.jsonl"
    log.write_text(log.read_text().replace(run.config_hash, earlier))
    meta = json.loads((launcher / "meta.json").read_text())
    (launcher / "meta.json").write_text(json.dumps({**meta, "config_hash": earlier}))
    settled_launcher(root, journal(root, path, label, ["r-a2"]), "r-a2")  # under the new hash
    qualify.main(restart(path, ADAMW, RESTART))
    second = json.loads((case_folder(out, ADAMW) / "restart-2.json").read_text())
    assert second["launcher"]["run_id"] == "r-a2"
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["run_id"] for r in lines if r["event"] == "superseded"] == ["r-a1", "r-a2"]
    log.write_text("".join(x + "\n" for x in log.read_text().splitlines()[:-1]))  # r-a2's lost
    with pytest.raises(SystemExit, match=r"recorded \(restart-2\.json\); the launch journal now"):
        qualify.main(restart(path, ADAMW, RESTART))
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert [r["run_id"] for r in lines if r["event"] == "superseded"] == ["r-a1", "r-a2"]
    assert not (case_folder(out, ADAMW) / "restart-3.json").exists()


def test_a_restart_whose_line_wrote_no_byte_blocks_until_restart_case_restores_it(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """A case restart record published, its ``superseded`` line written by no byte: the
    launcher and every command but the repair refuse with 12 naming the record and the run;
    ``restart-case`` restores the line (exit 1, its message), and the launcher goes on. Written
    by the real command, a restart and its line raise no refusal."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    launcher = settled_launcher(root, run, "r-a1")
    publish(out, ADAMW, "baseline", synthetic(ADAMW, identity=q.launcher_identity(launcher))[0])
    with monkeypatch.context() as patched:
        zero_byte_journal(patched)
        with pytest.raises(OSError, match="short write"):
            qualify.main(restart(path, ADAMW, RESTART))
    record = case_folder(out, ADAMW) / "restart-1.json"
    assert record.is_file() and launch.torn_lines(root) == []
    where = record.relative_to(root).as_posix()
    name, runs = launch.load_campaign(path)
    with pytest.raises(launch.TornJournalError) as refusal:
        launch.run_campaign(name, runs, root, dry_run=True)
    assert [(m.record, m.event, m.run_id) for m in refusal.value.missing] == [
        (where, "superseded", "r-a1")]  # fmt: skip
    before = tree(root)
    capsys.readouterr()
    assert exit_code(qualify, ["complete", *common(path), "--case", ADAMW.name]) == 12
    assert f"{where}: no superseded line of r-a1" in capsys.readouterr().err
    assert tree(root) == before
    with pytest.raises(SystemExit, match="the launch journal now names its launcher run"):
        qualify.main(restart(path, ADAMW, RESTART))
    assert launch.missing_lines(root) == []
    assert launch.run_campaign(name, runs, root, dry_run=True)["pending"] == [runs[0].label]


@pytest.mark.parametrize(
    "line",
    [{"event": "stop", "run": ["x"]},
     {"event": "end", "run": f"adamw-{ADAMW_LR:g}/s10200", "run_id": ["r"], "outcome": "done"},
     {"event": "superseded", "run": f"adamw-{ADAMW_LR:g}/s10200", "experiment": "study3",
      "run_id": {"r": 1}}],
)  # fmt: skip
def test_a_journal_object_with_a_non_string_identity_blocks_until_resolved(
    qualify, tmp_path, line
) -> None:
    """A valid JSON object whose identity fields are lists or objects is a torn line: every
    command that reads the runs refuses with 12 (never a traceback) until it is resolved; then
    they answer as without it, and a published record whose line is otherwise missing still
    blocks with 12."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    root = tmp_path / "runs"
    log = root / "launch" / "elsewhere.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(json.dumps(line) + "\n")
    case = ["--case", ADAMW.name]
    for command in ("complete", "clear-stale"):
        assert exit_code(qualify, [command, *common(path), *case]) == qualify.TORN_JOURNAL
    (torn,) = launch.torn_lines(root)
    launch.resolve_torn(root, torn.log, offset=torn.offset, sha256=torn.sha256, reason="r")
    assert exit_code(qualify, ["complete", *common(path), *case]) == qualify.NO
    assert exit_code(qualify, ["clear-stale", *common(path), *case]) == 0
    cycle = root / "qualification" / "c1" / "supersede-1.json"
    cycle.parent.mkdir(parents=True)
    cycle.write_text(json.dumps({"launchers": [{"run_id": "r-1", "run": "a/s0",
                                                "experiment": "study3"}], "held": []}))  # fmt: skip
    assert exit_code(qualify, ["complete", *common(path), *case]) == qualify.TORN_JOURNAL


def test_a_malformed_held_line_is_torn_and_its_record_line_missing(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """The cycle's ``held`` line of r-old, hand-edited so that its label is an object: it is a
    torn line, and the cycle record's line is missing as well, since the torn line matches
    nothing. Every command refuses with 12 naming both; resolved, the line is still missing;
    the cycle run again writes it, and the label is held again: the launcher refuses it as
    held, and ``complete`` exits ``HELD``."""
    from zero_field_refusal import launch

    no_commit = edited_meta(lambda meta: meta.pop("git"))
    path, root, out, run, name, record, log = issued_earlier(qualify, tmp_path, monkeypatch,
                                                             no_commit)  # fmt: skip
    qualify.main(supersede_args(path, ADAMW, record))
    start, held = log.read_text().splitlines()
    log.write_text("\n".join([start, json.dumps({**json.loads(held), "run": {"x": 1}})]) + "\n")
    (torn,) = launch.torn_lines(root)
    assert (torn.line, torn.kind, torn.resolved) == (2, "held", False)
    missing = [("qualification/c1/supersede-1.json", "held", "r-old")]
    assert [(m.record, m.event, m.run_id) for m in launch.missing_lines(root)] == missing
    _, runs = launch.load_campaign(path)
    with pytest.raises(launch.TornJournalError) as refusal:
        launch.run_campaign(name, runs, root, dry_run=True)
    assert len(refusal.value.lines) == 1 and len(refusal.value.missing) == 1
    complete = ["complete", *common(path), "--case", STOPPED.name]
    capsys.readouterr()
    assert exit_code(qualify, complete) == qualify.TORN_JOURNAL
    err = capsys.readouterr().err
    assert f"{log.name} line 2: offset" in err and "no held line of r-old" in err
    launch.resolve_torn(root, torn.log, offset=torn.offset, sha256=torn.sha256,
                        reason="a hand edit; the cycle writes the line again")  # fmt: skip
    assert exit_code(qualify, complete) == qualify.TORN_JOURNAL  # the line is still missing
    qualify.main(supersede_args(path, ADAMW, record))  # the repair
    assert launch.missing_lines(root) == []
    with pytest.raises(launch.CampaignError, match="r-old: held, its outcome unresolved"):
        launch.run_campaign(name, runs, root, only=run.label, max_attempts=1, recovery="r")
    assert exit_code(qualify, complete) == qualify.HELD


def test_a_real_restart_leaves_no_missing_line(qualify, tmp_path) -> None:
    """``restart-case`` publishes its record and appends its ``superseded`` line: the journal
    holds every line the record requires."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    launcher = settled_launcher(root, run, "r-a1")
    publish(out, ADAMW, "baseline", synthetic(ADAMW, identity=q.launcher_identity(launcher))[0])
    qualify.main(restart(path, ADAMW, RESTART))
    assert (case_folder(out, ADAMW) / "restart-1.json").is_file()
    assert launch.missing_lines(root) == []


def test_a_real_cycle_leaves_no_missing_line(qualify, tmp_path, monkeypatch) -> None:
    """``supersede`` publishes its record and appends its lines (here a ``held`` line, of a run
    under the label's earlier config hash): the journal holds every line the record requires."""
    from zero_field_refusal import launch

    no_commit = edited_meta(lambda meta: meta.pop("git"))
    path, root, out, run, name, record, log = issued_earlier(qualify, tmp_path, monkeypatch,
                                                             no_commit)  # fmt: skip
    qualify.main(supersede_args(path, ADAMW, record))
    assert json.loads((out / "c1" / "supersede-1.json").read_text())["held"]
    assert launch.missing_lines(root) == []


def test_an_unreadable_restart_record_blocks_by_its_path(qualify, tmp_path, capsys) -> None:
    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    record = case_folder(out, ADAMW) / "restart-1.json"
    record.parent.mkdir(parents=True)
    record.write_text("{")
    for argv in (["complete", *common(path), "--case", ADAMW.name],
                 restart(path, ADAMW, RESTART)):  # fmt: skip
        capsys.readouterr()
        assert exit_code(qualify, argv) == 12
        err = capsys.readouterr().err
        assert f"{record.relative_to(root).as_posix()}: a published record that cannot be" in err


TORN_BYTES = b'{"time": "2026-10-27T10:00:00+00:00", "event": "held", "run": "\xff'


@pytest.mark.parametrize(
    ("command", "extra"),
    [("part", ["--part", "baseline", "--resume", "--run-dir", "r", "--checkpoints", "c"]),
     ("complete", []), ("decided", []), ("clear-stale", []), ("locate", []), ("finish", []),
     ("interrupted", ["--part", "rest", "--attempt", "1", "--technical", "a crash"]),
     ("restart-case", ["--infrastructure", RESTART]),
     ("supersede", ["--record", "CAMPAIGN", "--diagnosis", "d", "--fix", "f"]),
     ("verdict", [])],
)  # fmt: skip
def test_every_subcommand_that_reads_the_runs_refuses_over_a_torn_line(
    qualify, tmp_path, capsys, command, extra
) -> None:
    """While a launch log holds a torn line that no ``torn_resolved`` names, each subcommand that
    reads the runs root prints the launcher's refusal, changes nothing and exits 12; resolved,
    it gets past the refusal (its own answer, whatever it is)."""
    from zero_field_refusal import launch

    path = campaign(tmp_path)
    root = tmp_path / "runs"
    write_case(root / "qualification", FA, control=True)
    log = root / "launch" / "elsewhere.jsonl"
    log.parent.mkdir(parents=True)
    log.write_bytes(TORN_BYTES)
    argv = [command, *common(path), *([] if command == "verdict" else ["--case", ADAMW.name])]
    extra = [str(path) if x == "CAMPAIGN" else x for x in extra]  # a file, not a part record
    before = tree(root)
    capsys.readouterr()
    assert exit_code(qualify, [*argv, *extra]) == qualify.TORN_JOURNAL
    assert tree(root) == before
    err = capsys.readouterr().err
    assert f"elsewhere.jsonl line 1: offset 0, length {len(TORN_BYTES)}, kind held" in err
    (line,) = launch.torn_lines(root)
    launch.resolve_torn(root, line.log, offset=line.offset, sha256=line.sha256,
                        reason="nothing was lost: a test line")  # fmt: skip
    after = extra if command != "part" else ["--part", "baseline"]  # no harness: its own refusal
    assert exit_code(qualify, [*argv, *after]) != qualify.TORN_JOURNAL


def test_plan_and_driver_read_no_runs_and_are_not_refused(qualify, tmp_path, capsys) -> None:
    path = campaign(tmp_path)
    log = tmp_path / "runs" / "launch" / "elsewhere.jsonl"
    log.parent.mkdir(parents=True)
    log.write_bytes(TORN_BYTES)
    qualify.main(["plan", *common(path)])
    qualify.main(["driver", *common(path), "--dry-run"])
    assert "torn" not in capsys.readouterr().err


def test_the_driver_stops_at_a_torn_journal_line(qualify, tmp_path, monkeypatch) -> None:
    """``complete`` exits 12 over an unresolved torn line: the driver stops there with 12 and
    calls neither the launcher nor a part; its EXIT trap still calls ``verdict``, which over a
    real torn line refuses too, so no ``verdict.json`` is written then and an older one stays.
    The launcher's own 12 passes through the same way. The driver's exits comment lists 12."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, complete=qualify.TORN_JOURNAL)
    assert code == qualify.TORN_JOURNAL
    assert calls == [f"complete {FA.name}", "verdict"]
    header = " ".join(x[2:] for x in (tmp_path / "driver.sh").read_text().splitlines()
                      if x.startswith("# "))  # fmt: skip
    assert f"{qualify.TORN_JOURNAL} a torn line in a launch journal" in header
    assert "verdict.json is written then" in header
    text = (tmp_path / "driver.sh").read_text()
    assert f"  {qualify.TORN_JOURNAL}) stop " in text and "--resolve-torn" in text
    (tmp_path / "again").mkdir()
    code, calls = run_driver(qualify, tmp_path / "again", monkeypatch, launch=12)
    assert code == 12 and calls[-2:] == ["launch", "verdict"]


def test_mixed_trajectory_results_are_recorded_technical(qualify, tmp_path) -> None:
    """Results of one interrupted attempt bound to different inputs: ``interrupted`` records
    the attempt technical, with nothing of them to read."""
    path = campaign(tmp_path, adamw=())
    out = tmp_path / "runs" / "qualification"
    baseline, rest = synthetic(FA)
    publish(out, FA, "baseline", baseline)
    rest["baseline_record"]["sha256"] = q.record_sha256(baseline)
    q.publish(out / (FA.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    q.publish(out / FA.trajectory_path(1, "repeat"), trajectory_result(FA, rest, "repeat"))
    other = {**trajectory_result(FA, rest, "reference"), "baseline_checkpoint_sha256": "x" * 64}
    q.publish(out / FA.trajectory_path(1, "reference"), other)
    qualify.main(["interrupted", *common(path), "--case", FA.name, "--part", "rest",
                  "--attempt", "1", "--technical", "the process was killed"])  # fmt: skip
    record = json.loads((out / FA.attempt_path("rest", 1)).read_text())
    assert record["kind"] == "technical" and record["bridge"] is None
    assert "the trajectory results are bound to different inputs" in record["reasons"]
    v = q.case_verdict(baseline, record, SCOPE)
    assert v["kind"] == "technical" and all(c["D"] is None for n, c in v["comparisons"].items()
                                            if n != "launcher~baseline")  # fmt: skip


def test_a_lost_checkpoint_restarts_the_launcher_run(
    qualify, stubbed, tmp_path, monkeypatch
) -> None:
    """A passing launcher run and baseline, the rest interrupted, the container disk lost: the
    launcher refuses the run whose checkpoint is gone. ``restart-case`` voids the parts and logs
    the launcher run superseded (kept): the launcher's recovery round starts a new run,
    ``locate`` finds it, the parts run as attempt 2, and the verdict voids the old ones."""
    from test_launch import make_run_dir

    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    label = f"adamw-{ADAMW_LR:g}/s10200"
    run = journal(root, path, label, ["r-a1"])
    old = settled_launcher(root, run, "r-a1")
    identity = q.launcher_identity(old)  # its checkpoint, /x.pt, is gone
    baseline, _ = synthetic(ADAMW, identity=identity)
    publish(out, ADAMW, "baseline", baseline)
    q.publish(out / (ADAMW.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    name, runs = launch.load_campaign(path)
    only = {"only": label, "max_attempts": 1}

    def execute(run, run_id, output, campaign=None):
        make_run_dir(root, run, run_id, "completed")
        return 0

    with pytest.raises(launch.CampaignError, match="cannot stand"):  # before any recovery
        launch.run_campaign(name, runs, root, execute=execute, recovery=RESTART, **only)
    qualify.main(restart(path, ADAMW, RESTART))
    restart_record = json.loads((case_folder(out, ADAMW) / "restart-1.json").read_text())
    assert restart_record["launcher"] == {"run_id": "r-a1", "identity": identity}
    monkeypatch.setattr(launch, "_disqualification", lambda run, run_dir, meta: None)
    result = launch.run_campaign(name, runs, root, execute=execute, recovery=RESTART, **only)
    (new,) = [d for d in (root / run.experiment).iterdir() if d.name != "r-a1"]
    assert result["done"] == [label] and old.is_dir()
    assert qualify.located(run, root) == new  # the old run no longer counts
    for part_name in ("baseline", "rest"):
        qualify.main(["part", *common(path), "--case", ADAMW.name, "--part", part_name,
                      "--attempt", "2", "--infrastructure", RESTART, "--run-dir", str(new),
                      "--checkpoints", str(tmp_path / "h"), "--device", "cpu"])  # fmt: skip
    assert stubbed == [(ADAMW.name, "baseline", 2), (ADAMW.name, "rest", 2)]
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    assert [(e["part"], e["attempt"], e["restart"]) for e in verdict["superseded"]] == [
        ("baseline", 1, RESTART)]  # fmt: skip
    rate = verdict["rates"][f"adamw-{ADAMW_LR:g}"]
    assert [v["case"] for v in rate["voided"]] == [ADAMW.name]


def case_folder(out, case):
    return out / case.attempt_path("baseline", 1).rsplit("/", 1)[0]


def test_finish_keeps_a_technical_cases_checkpoints(qualify, tmp_path) -> None:
    """A case whose verdict is technical (or whose same-path control is) keeps its checkpoints
    for the diagnosis: ``finish`` exits 9; ``--release`` deletes them once diagnosed."""
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    journal(tmp_path / "runs", path, f"adam_frozen-{FA_LR:g}/s10200", ["r-a1"])
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "r-a1.pt").write_bytes(b"w")
    write_case(out, FA)
    publish(out, FA, "control", control_record(0.1, case=FA))  # beyond 0.01·K*: technical
    finish = ["finish", *common(path), "--case", FA.name]
    with pytest.raises(SystemExit) as stop:
        qualify.main(finish)
    assert stop.value.code == qualify.TECHNICAL_CASE
    assert (tmp_path / "ckpt" / "r-a1.pt").exists()
    qualify.main([*finish, "--release"])
    assert not (tmp_path / "ckpt" / "r-a1.pt").exists()


def test_a_part_sets_the_numerics_before_its_environment(qualify, tmp_path, monkeypatch) -> None:
    """The order of ``part``'s own calls: ``runtime.configure_numerics`` before
    ``runtime.environment`` (before any CUDA work), then the harness."""
    calls = []
    monkeypatch.setattr(
        qualify.runtime,
        "configure_numerics",
        lambda cfg, device: calls.append(("configure_numerics", device)) or {},
    )
    monkeypatch.setattr(
        qualify.runtime,
        "environment",
        lambda cfg, device: calls.append(("environment", device)) or object(),
    )
    harness = fake_part([])

    def run_part(*args, **kwargs):
        calls.append(("run_part", kwargs["part"]))
        return harness(*args, **kwargs)

    monkeypatch.setattr(qualify.q, "run_part", run_part)
    qualify.main(part_args(campaign(tmp_path), tmp_path, "baseline", "--resume"))
    assert calls == [("configure_numerics", "cpu"), ("environment", "cpu"),
                     ("run_part", "baseline")]  # fmt: skip


def test_a_stale_case_stops_complete_and_the_driver(qualify, tmp_path, monkeypatch, capsys):
    """Records of other GPU code or settings than this checkout's: ``complete`` and ``decided``
    exit 10 (stale), and the driver stops there, rerunning and skipping nothing."""
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    write_case(out, FA, control=True)
    qualify.main(["complete", *common(path), "--case", FA.name])  # this checkout's code
    record = json.loads((out / FA.attempt_path("control", 1)).read_text())
    for change, why in (({"gpu_code": {**CODE, "sha256": "o" * 64}}, "other GPU code"),
                        ({"settings": {**q.settings_record(REGISTERED), "norm": 1e-4}},
                         "other settings")):  # fmt: skip
        publish(out, FA, "control", {**record, "provenance": {**record["provenance"], **change}})
        for command in ("complete", "decided"):
            with pytest.raises(SystemExit) as stop:
                qualify.main([command, *common(path), "--case", FA.name])
            assert stop.value.code == qualify.STALE
        assert f"control attempt 1: {why}" in capsys.readouterr().err
    (tmp_path / "d").mkdir()
    code, calls = run_driver(qualify, tmp_path / "d", monkeypatch, complete=qualify.STALE)
    assert code == qualify.STALE and calls == [f"complete {FA.name}", "verdict"]


def test_a_part_needs_a_clean_checkout(qualify, stubbed, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(qualify, "git_state", lambda *a, **k: {"commit": "c", "dirty": True})
    with pytest.raises(SystemExit, match="not clean"):
        qualify.main(part_args(campaign(tmp_path), tmp_path, "baseline", "--resume"))
    assert stubbed == []


# ------------------------------------------------------------------ the checkpoints


def journal(root, path, label, run_ids):
    from zero_field_refusal.launch import load_campaign

    name, runs = load_campaign(path)
    run = next(r for r in runs if r.label == label)
    log = root / "launch" / f"{name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    before = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    attempt = sum(r.get("event") == "start" and r.get("run") == label for r in before)
    with log.open("a") as f:
        for run_id in run_ids:
            attempt += 1
            f.write(json.dumps({"event": "start", "run": label, "attempt": attempt,
                                "run_id": run_id, "experiment": run.experiment,
                                "config_hash": run.config_hash,
                                "command": run.command()[1:]}) + "\n")  # fmt: skip
    return run


def test_a_runs_attempts_are_read_in_attempt_order_across_journals(qualify, tmp_path) -> None:
    """Attempt 2 of a launcher run logged in a journal whose name sorts before attempt 1's (its
    recovery under another campaign name): the run's attempts are read by their number."""
    path = campaign(tmp_path, fa=())
    root = tmp_path / "runs"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    recovery = {"event": "start", "run": run.label, "attempt": 2, "run_id": "r-a2",
                "experiment": run.experiment, "config_hash": run.config_hash}  # fmt: skip
    (root / "launch" / "a-recovery.jsonl").write_text(json.dumps(recovery) + "\n")
    assert qualify.journal_run_ids(run, root) == ["r-a1", "r-a2"]


def test_the_checkpoints_go_only_after_published_valid_records(qualify, tmp_path) -> None:
    path = campaign(tmp_path)
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    label = f"adam_frozen-{FA_LR:g}/s10200"
    journal(root, path, label, ["r-a1", "r-a2"])
    endpoint = tmp_path / "ckpt"
    harness = endpoint / "harness" / "s10200-adam_frozen-1.83e-05"
    (harness / "baseline-states").mkdir(parents=True)
    files = [endpoint / "r-a1.pt", endpoint / "r-a2.pt", harness / "harness-baseline.pt"]
    for file in files:
        file.write_bytes(b"w")
    other = endpoint / "unrelated.pt"
    other.write_bytes(b"w")
    finish = ["finish", *common(path), "--case", FA.name]
    with pytest.raises(SystemExit):  # no records yet
        qualify.main(finish)
    write_case(out, FA, control=True)
    (out / FA.attempt_path("rest", 1)).write_text("{not json")  # a damaged record
    with pytest.raises(SystemExit):
        qualify.main(finish)
    assert all(file.exists() for file in files)
    write_case(out, FA, control=True)
    qualify.main(finish)
    assert not any(file.exists() for file in files) and not harness.exists()
    assert other.exists()
    qualify.main(finish)  # again: nothing left to delete


def test_a_decided_case_loses_its_checkpoints_too(qualify, tmp_path) -> None:
    path = campaign(tmp_path)
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    journal(root, path, f"adam_frozen-{FA_LR:g}/s10200", ["r-a1"])
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "r-a1.pt").write_bytes(b"w")
    write_case(out, FA, decided=True, control=True)
    qualify.main(["finish", *common(path), "--case", FA.name])
    assert not (tmp_path / "ckpt" / "r-a1.pt").exists()


def test_a_stale_checkpoint_of_the_case_is_cleared_before_its_launch(qualify, tmp_path) -> None:
    """An earlier failed attempt's ``<dir>/<run_id>.pt`` (and its partial file) goes; the
    checkpoint of the case's settled launcher run stays."""
    from test_launch import make_run_dir

    path = campaign(tmp_path)
    root = tmp_path / "runs"
    run = journal(root, path, f"adam_frozen-{FA_LR:g}/s10200", ["r-a1", "r-a2"])
    make_run_dir(root, run, "r-a1", "failed")
    make_run_dir(root, run, "r-a2", "diverged")
    endpoint = tmp_path / "ckpt"
    endpoint.mkdir()
    for name in ("r-a1.pt", ".r-a1.pt.partial", "r-a2.pt"):
        (endpoint / name).write_bytes(b"w")
    qualify.main(["clear-stale", *common(path), "--case", FA.name])
    assert sorted(p.name for p in endpoint.iterdir()) == ["r-a2.pt"]


def test_clear_stale_keeps_a_completed_conflicted_runs_checkpoint(qualify, tmp_path) -> None:
    """Only the checkpoints of attempts the journal shows superseded (ended technical, or not
    ended) go: a completed run that cannot stand keeps its checkpoint for the diagnosis."""
    from test_launch import make_run_dir

    path = campaign(tmp_path)
    root = tmp_path / "runs"
    label = f"adam_frozen-{FA_LR:g}/s10200"
    run = journal(root, path, label, ["r-a1", "r-a2", "r-a3"])
    log = root / "launch" / "study3-qualification-c1.jsonl"
    with log.open("a") as f:
        for run_id, outcome in (("r-a1", "technical"), ("r-a2", "conflict")):
            f.write(json.dumps({"event": "end", "run": label, "run_id": run_id,
                                "outcome": outcome}) + "\n")  # fmt: skip
    make_run_dir(root, run, "r-a1", "failed")
    make_run_dir(root, run, "r-a2", "completed")  # its checkpoint does not match: a conflict
    endpoint = tmp_path / "ckpt"
    endpoint.mkdir()
    for name in ("r-a1.pt", "r-a2.pt", ".r-a3.pt.partial"):  # r-a3: started, never ended
        (endpoint / name).write_bytes(b"w")
    qualify.main(["clear-stale", *common(path), "--case", FA.name])
    assert sorted(p.name for p in endpoint.iterdir()) == ["r-a2.pt"]


def test_locate_prints_the_settled_launcher_run(qualify, tmp_path, capsys) -> None:
    from test_launch import make_run_dir

    path = campaign(tmp_path)
    root = tmp_path / "runs"
    run = journal(root, path, f"adam_frozen-{FA_LR:g}/s10200", ["r-a1"])
    with pytest.raises(SystemExit):
        qualify.main(["locate", *common(path), "--case", FA.name])
    make_run_dir(root, run, "r-a1", "diverged")
    capsys.readouterr()
    qualify.main(["locate", *common(path), "--case", FA.name])
    assert capsys.readouterr().out.strip() == str(root / run.experiment / "r-a1")


# ------------------------------------------------------------------------------ the verdict


SEEDS = (10200, 10201, 10202)


def launcher_dir(root, path, label, run_id):
    """A settled launcher run of ``label`` (``located`` finds it), with its checkpoint record;
    returns its identity as the records name it."""
    from test_launch import make_run_dir

    from zero_field_refusal.launch import load_campaign

    run = next(r for r in load_campaign(path)[1] if r.label == label)
    run_dir = make_run_dir(root, run, run_id)
    summary = {"endpoint_checkpoint": IDENTITY["checkpoint"]}
    (run_dir / "summary.json").write_text(json.dumps(summary))
    return q.launcher_identity(run_dir)


def verdict_world(qualify, tmp_path, monkeypatch, *, fa_control=True, failing=None,
                  located=True):  # fmt: skip
    """Every case of a campaign on SEEDS recorded qualified; with ``located``, each with its
    settled launcher run under RUNS_DIR."""
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(CODE))
    path = campaign(tmp_path, fa=SEEDS, adamw=SEEDS)
    out = tmp_path / "runs" / "qualification"
    for optimizer, lr in (("adam_frozen", FA_LR), ("adamw", ADAMW_LR)):
        for seed in SEEDS:
            case = q.QualificationCase(seed, optimizer, lr, 1, False)
            identity = {**copy.deepcopy(IDENTITY), "run": f"study3/r{seed}", "run_id": f"r{seed}"}
            if located:
                identity = launcher_dir(tmp_path / "runs", path, f"{optimizer}-{lr:g}/s{seed}",
                                        f"r-{optimizer}-{seed}")  # fmt: skip
            baseline, rest = qualified_records(case, identity)
            if failing == (optimizer, seed):
                rest["comparisons"]["baseline~s3"] = {"D_twin": 0.07}  # beyond 0.06
            publish(out, case, "baseline", baseline)
            rest["baseline_record"]["sha256"] = q.record_sha256(baseline)
            publish(out, case, "rest", rest)
            if fa_control and optimizer == "adam_frozen" and seed == SEEDS[0]:
                publish(out, case, "control",
                        control_record(case=case, identity=identity))  # fmt: skip
    return path, out


def test_the_verdict_writes_each_rates_state(qualify, tmp_path, monkeypatch, capsys) -> None:
    path, out = verdict_world(qualify, tmp_path, monkeypatch)
    qualify.main(["verdict", *common(path), "--e-eng", "0.0001"])
    record = json.loads((out / "c1" / "verdict.json").read_text())
    assert record["states"] == {f"adam_frozen-{FA_LR:g}": "qualified",
                                f"adamw-{ADAMW_LR:g}": "qualified"}  # fmt: skip
    assert len(record["cases"]) == 6 and record["e_eng"] == 0.0001
    assert all(len(c["comparisons"]) == 13 and c["kind"] is None for c in record["cases"])
    assert record["rates"][f"adam_frozen-{FA_LR:g}"]["control"]["state"] == "accepted"
    assert record["unassigned"] == []
    printed = capsys.readouterr().out
    assert f"adamw-{ADAMW_LR:g}: qualified" in printed


def test_a_rate_without_its_launcher_runs_is_at_most_incomplete(qualify, tmp_path, monkeypatch):
    path, out = verdict_world(qualify, tmp_path, monkeypatch, located=False)
    qualify.main(["verdict", *common(path)])
    record = json.loads((out / "c1" / "verdict.json").read_text())
    assert record["states"] == {f"adam_frozen-{FA_LR:g}": "incomplete",
                                f"adamw-{ADAMW_LR:g}": "incomplete"}  # fmt: skip
    reasons = record["rates"][f"adamw-{ADAMW_LR:g}"]["reasons"]
    assert [r for r in reasons if "not located" in r] == [
        f"the launcher run of seed {s} is not located" for s in SEEDS]  # fmt: skip
    assert any("at most incomplete" in w for w in record["warnings"])


def test_the_verdict_of_a_failing_case_and_a_missing_control(qualify, tmp_path, monkeypatch):
    path, out = verdict_world(qualify, tmp_path, monkeypatch, fa_control=False,
                              failing=("adamw", 10201))  # fmt: skip
    qualify.main(["verdict", *common(path)])
    record = json.loads((out / "c1" / "verdict.json").read_text())
    assert record["states"] == {f"adam_frozen-{FA_LR:g}": "not_qualified",
                                f"adamw-{ADAMW_LR:g}": "failing"}  # fmt: skip
    kinds = {c["case"]["name"]: c["kind"] for c in record["cases"]}
    assert kinds[f"s10201/adamw-{ADAMW_LR:g}"] == "scientific"
    shutil.rmtree(out / "c1" / "numerical" / "s10202")
    qualify.main(["verdict", *common(path)])
    record = json.loads((out / "c1" / "verdict.json").read_text())
    assert record["states"][f"adamw-{ADAMW_LR:g}"] == "failing"  # failing outranks incomplete
