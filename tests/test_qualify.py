"""Study 3's qualification driver (scripts/qualify.py): the cases of a campaign, the driver's
order per case, resume, the attempts of a part, the deletion of the checkpoints and the verdict
(docs/study3-preregistration.md, "Numerical qualification", "Failure routes", "Amendment 2").

Nothing here runs a model or touches a GPU: the parts' harness (``qualification.run_part``) is
replaced by a stub that publishes a record, and the verdicts read synthetic part records.
"""

import copy
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_qualification import CODE, IDENTITY, control_record, synthetic

from zero_field_refusal import code_identity
from zero_field_refusal import qualification as q
from zero_field_refusal.config import load_config

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


@pytest.fixture
def qualify(monkeypatch, tmp_path):
    module = load_script()
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(module, "git_state", lambda *a, **k: {"commit": "c" * 40, "dirty": False})
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


def test_the_driver_skips_a_complete_case_and_the_rest_of_a_decided_one(qualify, tmp_path):
    """Resume: the launcher and the parts sit inside the case's ``complete`` check (a complete
    case runs neither); the rest and the control inside the baseline's ``decided`` check; the
    deletion after both, for a complete or decided case too."""
    text = driver_text(qualify, tmp_path)
    block = text[text.index("# ---- case 1/") : text.index("# ---- case 2/")]
    lines = block.splitlines()
    head = next(i for i, line in enumerate(lines) if "qualify.py complete" in line)
    assert lines[head].startswith("if ") and lines[head + 2].strip() == "else"
    closing = next(i for i, line in enumerate(lines) if line == "fi" and i > head)
    for what in ("scripts/launch.py", "--part baseline", "--part rest", "--part control"):
        assert head < next(i for i, line in enumerate(lines) if what in line) < closing
    decided = next(i for i, line in enumerate(lines) if "qualify.py decided" in line)
    assert lines[decided].strip().startswith("if ") and lines[decided + 2].strip() == "else"
    rest = next(i for i, line in enumerate(lines) if "--part rest" in line)
    assert decided < rest
    assert "qualify.py finish" in lines[closing + 1]


def test_the_driver_checks_its_campaign_and_the_gpu(qualify, tmp_path) -> None:
    path = campaign(tmp_path)
    text = driver_text(qualify, tmp_path, path)
    import hashlib

    assert f"CAMPAIGN={path.resolve()}" in text
    assert hashlib.sha256(path.read_bytes()).hexdigest() in text
    positions(
        text, [r"git rev-parse HEAD", r"sha256sum", r"trap .*verdict", r"nvidia-smi", r"--case "]
    )
    with pytest.raises(SystemExit):  # an existing --out
        qualify.main(["driver", *common(path), "--commit", "a" * 40,
                      "--out", str(path)])  # fmt: skip


# ----------------------------------------------------------------- resume and the attempts


def test_complete_and_decided(qualify, tmp_path) -> None:
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    run = lambda what, case: qualify.main([what, *common(path), "--case", case.name])  # noqa: E731
    for what in ("complete", "decided"):
        with pytest.raises(SystemExit) as stop:
            run(what, FA)
        assert stop.value.code == 1
    write_case(out, FA)  # no control yet
    with pytest.raises(SystemExit):
        run("complete", FA)
    publish(out, FA, "control", control_record(case=FA))
    run("complete", FA)  # exit 0
    with pytest.raises(SystemExit):
        run("decided", FA)
    write_case(out, ADAMW, decided=True)  # the baseline is a scientific failure: no rest
    run("decided", ADAMW)
    run("complete", ADAMW)


def fake_part(calls):
    def run_part(settings, env, case, run_dir, out, *, part, attempt, checkpoints,
                 diagnostic_steps=()):  # fmt: skip
        calls.append((case.name, part, attempt))
        record = {"part": part, "attempt": attempt, "case": case.record(), "status": "failed",
                  "kind": "technical", "reasons": ["RuntimeError: boom"]}  # fmt: skip
        q.publish(out / case.attempt_path(part, attempt), record)
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
    }
    qualify.main(part_args(path, tmp_path, "baseline", "--resume"))  # the latest is recorded
    assert len(stubbed) == 1


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
    with log.open("a") as f:
        for run_id in run_ids:
            f.write(json.dumps({"event": "start", "run": label, "run_id": run_id,
                                "experiment": run.experiment,
                                "config_hash": run.config_hash}) + "\n")  # fmt: skip
    return run


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
    write_case(out, FA, decided=True)
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


def verdict_world(qualify, tmp_path, monkeypatch, *, fa_control=True, failing=None):
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(CODE))
    path = campaign(tmp_path, fa=SEEDS, adamw=SEEDS)
    out = tmp_path / "runs" / "qualification"
    for optimizer, lr in (("adam_frozen", FA_LR), ("adamw", ADAMW_LR)):
        for seed in SEEDS:
            case = q.QualificationCase(seed, optimizer, lr, 1, False)
            identity = {**copy.deepcopy(IDENTITY), "run": f"study3/r{seed}", "run_id": f"r{seed}"}
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
