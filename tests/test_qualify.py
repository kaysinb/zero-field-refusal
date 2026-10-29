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
    qualify.main([*driver_args(tmp_path, path, "--engineering", "--diagnostic-steps", "5,10,15"),
                  "--commit", "a" * 40, "--out", str(out)])  # fmt: skip
    text = out.read_text()
    assert f"-lt {qualify.need_kb(6 * 6.5 + 4)} ]" in text
    assert text.count("--diagnostic-steps 5,10,15") == 4  # each case's baseline and rest


# ------------------------------------------------------------------------------- the driver


def driver_args(tmp_path, path, *extra):
    """``driver``'s arguments: the campaign, the candidate and the Pod's volume folder of a
    technical case's kept checkpoints (``--kept``, absolute)."""
    return ["driver", *common(path, *extra), "--kept", str(tmp_path / "volume" / "kept")]


def driver_text(qualify, tmp_path, path=None) -> str:
    out = tmp_path / "driver.sh"
    qualify.main([*driver_args(tmp_path, path or campaign(tmp_path)), "--commit", "a" * 40,
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
                  rf"qualify\.py control-only .*--case '?{re.escape(name)}'?",
                  r"qualify\.py clear-stale", r"df -Pk", r"-lt \d+ \]",
                  rf"scripts/launch\.py \"\$CAMPAIGN\" --only '?{re.escape(label)}'? "
                  r"--max-attempts 1", r"qualify\.py locate",
                  r"qualify\.py part .*--part baseline --resume", r"qualify\.py decided",
                  r"qualify\.py part .*--part rest --resume",
                  *([r"qualify\.py part .*--part control --resume"] if control else []),
                  r"qualify\.py finish"]  # fmt: skip
    positions(text, order)
    fa_case, adamw_case = text.split("# ---- case ")[1:]
    assert adamw_case.count("qualify.py control-only") == 1 and "--part control" not in adamw_case
    # FA on the first numerical seed alone: the control alone (control-only), or after the rest
    assert fa_case.count("--part control") == 2 and fa_case.count("qualify.py control-only") == 1
    for block in (fa_case, adamw_case):  # an established technical case: exported first
        assert block.count("qualify.py finish ") == 2
        assert block.index("qualify.py finish ") < block.index('scripts/launch.py "$CAMPAIGN"')
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)


def test_the_driver_has_no_rerun_line(qualify, tmp_path) -> None:
    """No automatic second attempt: a part's record (technical or scientific) is never rerun by
    the driver, and the launcher gets one attempt."""
    text = driver_text(qualify, tmp_path)
    commands = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    assert "--attempt" not in commands and "--infrastructure" not in commands
    assert "--recovery" not in commands and "restart-case" not in commands
    assert commands.count("--max-attempts 1") == 2


STUB = r"""#!/usr/bin/env bash
# project-run "$SRC" python <script> <command> ...: logs the command, exits as STUB_<command> says
if [ "$3" = -c ]; then exit 0; fi
cmd="$4"
if [ "$3" = scripts/launch.py ]; then cmd=launch; fi
if [ "$3" = scripts/archive.py ]; then cmd=archive; fi
echo "$cmd ${*:5}" >> "$STUB_LOG"
n=$(grep -c "^$cmd " "$STUB_LOG")
var="STUB_${cmd//-/_}"
read -r -a codes <<< "${!var:-0}"
i=$(( n - 1 < ${#codes[@]} ? n - 1 : ${#codes[@]} - 1 ))
if [ "$cmd" = locate ] || [ "$cmd" = control-only ]; then echo /runs/study3/r1; fi
exit "${codes[$i]}"
"""


# stat -L -c %d PATH: the container disk's device (1) for the checkpoint directory, the volume's
# (STUB_device, 2 by default) for any other path
STAT = r"""#!/usr/bin/env bash
case "${@: -1}" in */ckpt|*/ckpt/*) echo 1 ;; *) echo "${STUB_device:-2}" ;; esac
"""
# What the driver and its stand-ins call before its first case, for a PATH of one private folder
TOOLS = ("bash", "env", "mkdir", "tee", "date", "sha256sum", "cut", "awk", "grep", "realpath",
         "cat", "sh")  # fmt: skip


def run_driver(qualify, tmp_path, monkeypatch, *, bucket=None, kept=None, **exits):
    """The generated driver run by bash against stubs: ``project-run`` (each qualify.py command,
    the launcher and scripts/archive.py logged, with the exit ``exits`` gives it: a code, or codes
    call by call), ``git``, ``nvidia-smi``, ``stat`` (the devices: ``STAT``) and a stand-in
    ``gcloud``; no Pod environment, a disk need of 1 KB, the bucket's settings
    (``ARTIFACT_REMOTE``, ``ARTIFACT_GCS_KEY`` a readable file; ``bucket`` changes them: a value,
    or None to leave one unset; ``gcloud`` None: a PATH of one private folder without gcloud)
    and ``--kept`` (``kept``, by default ``volume/kept``). Its output goes to ``driver.out``."""
    commit = "a" * 40
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bucket = dict(bucket or {})
    stand_ins = [("project-run", STUB), ("git", f"#!/bin/sh\necho {commit}\n"), ("stat", STAT),
                 ("nvidia-smi", '#!/bin/sh\ncase "$*" in *compute-apps*) ;; '
                                "*) echo 0 ;; esac\n")]  # fmt: skip
    path = f"{bin_dir}:/usr/bin:/bin"
    if "gcloud" not in bucket:  # never called: project-run stands in for scripts/archive.py
        stand_ins.append(("gcloud", "#!/bin/sh\nexit 1\n"))
    else:  # no gcloud anywhere on PATH: the private folder alone, with the tools it needs
        path = str(bin_dir)
        for tool in TOOLS:
            (bin_dir / tool).symlink_to(shutil.which(tool))
    bucket.pop("gcloud", None)
    for name, text in stand_ins:
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    (tmp_path / "code" / commit / "source").mkdir(parents=True)
    key = tmp_path / "artifact-gcs-key.json"
    key.write_text("{}")
    monkeypatch.setattr(qualify, "POD_ENVIRONMENT", ())
    monkeypatch.setattr(qualify, "SOURCES", str(tmp_path / "code"))
    monkeypatch.setattr(qualify, "need_kb", lambda gb: 1)
    out = tmp_path / "driver.sh"
    args = driver_args(tmp_path, campaign(tmp_path))
    if kept is not None:
        args[-1] = str(kept)
    qualify.main([*args, "--commit", commit, "--out", str(out)])
    log = tmp_path / "stub.log"
    log.touch()
    settings = {"ARTIFACT_REMOTE": "gs://bucket/zfr", "ARTIFACT_GCS_KEY": str(key), **bucket}
    env = {"PATH": path, "RUNS_DIR": str(tmp_path / "runs"),
           "STUB_LOG": str(log), "STUB_complete": "3", "STUB_decided": "3",
           "STUB_control_only": "3",
           **{name: value for name, value in settings.items() if value is not None},
           **{f"STUB_{key}": str(value) for key, value in exits.items()}}  # fmt: skip
    done = subprocess.run(["bash", str(out)], env=env, capture_output=True, text=True)
    (tmp_path / "driver.out").write_text(done.stdout + done.stderr)
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
    assert calls == [*_with_case([FULL[0], "control-only", *FULL[1:], "part control", "finish"],
                                 FA),
                     *_with_case([FULL[0], "control-only", *FULL[1:], "finish"], ADAMW),
                     "verdict"]  # fmt: skip


def test_the_driver_resumes_past_a_complete_case(qualify, tmp_path, monkeypatch) -> None:
    """A complete case (``complete`` exits 0) runs neither the launcher nor a part: only its
    ``finish``, which deletes what is left."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, complete="0 3")
    assert code == 0
    assert calls == [f"complete {FA.name}", f"finish {FA.name}",
                     *_with_case([FULL[0], "control-only", *FULL[1:], "finish"], ADAMW),
                     "verdict"]  # fmt: skip


def test_a_decided_case_skips_the_rest_but_not_its_control(qualify, tmp_path, monkeypatch):
    code, calls = run_driver(qualify, tmp_path, monkeypatch, decided="0 3")
    assert code == 0
    assert calls[: len(FULL) + 2] == _with_case([FULL[0], "control-only", *FULL[1:-1],
                                                 "part control", "finish"], FA)  # fmt: skip
    assert f"part {FA.name} rest" not in calls


def test_a_decided_case_with_its_control_due_runs_the_control_alone(qualify, tmp_path, monkeypatch):
    """``control-only`` exits 0 for FA, decided by its baseline with its same-path control alone
    outstanding: the driver runs that control with the launcher run's directory it printed, and
    then ``finish``; neither ``clear-stale``, the launcher, ``locate``, the baseline,
    ``decided`` nor the rest is called. AdamW's case, without the control (its ``control-only``
    says 3), runs in full."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, control_only="0 3")
    assert code == 0
    assert calls == [*_with_case(["complete", "control-only", "part control", "finish"], FA),
                     *_with_case([FULL[0], "control-only", *FULL[1:], "finish"], ADAMW),
                     "verdict"]  # fmt: skip
    (control,) = [line for line in (tmp_path / "stub.log").read_text().splitlines()
                  if line.startswith("part ") and "--part control" in line]  # fmt: skip
    assert "--run-dir /runs/study3/r1 " in control and control.endswith("--part control --resume")
    text = (tmp_path / "driver.sh").read_text()
    assert f"{FA.name}: only its same-path control is due (its baseline decided it, or" in text


def test_the_driver_exports_an_established_technical_case_before_any_launcher_call(
    qualify, tmp_path, monkeypatch
) -> None:
    """``control-only`` exits 9 for a case its recorded parts already mark technical: the driver
    runs the export at once (the routed ``finish --kept``, archive.py, ``kept-check --remote``,
    the sequence a complete technical case's ``finish`` takes) and stops with 9, or with 13 when
    the upload fails; neither ``clear-stale``, the launcher, ``locate`` nor a part is called. In
    AdamW's block too: every case block asks ``control-only`` first."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, control_only=9, finish=9)
    assert code == qualify.TECHNICAL_CASE
    assert calls == [f"complete {FA.name}", f"control-only {FA.name}", f"finish {FA.name}",
                     "archive", f"kept-check {FA.name}", "verdict"]  # fmt: skip
    for folder, exits, last in (
        ("adamw", {"control_only": "3 9", "finish": "0 9"}, ADAMW),
        ("pending", {"control_only": 9, "finish": 9, "archive": 1}, FA),
    ):  # fmt: skip
        (tmp_path / folder).mkdir()
        code, calls = run_driver(qualify, tmp_path / folder, monkeypatch, **exits)
        asked = calls.index(f"control-only {last.name}")
        assert "launch" not in calls[asked:]
        assert calls[-5:] == [f"control-only {last.name}", f"finish {last.name}", "archive",
                              f"kept-check {last.name}", "verdict"]  # fmt: skip
        assert code == (qualify.NOT_EXPORTED if folder == "pending" else qualify.TECHNICAL_CASE)
    output = (tmp_path / "adamw" / "driver.out").read_text()
    assert f"{ADAMW.name}: a technical case by its recorded parts" in output


@pytest.mark.parametrize("command", ["complete", "control-only", "decided"])
def test_a_crashing_check_stops_the_driver(qualify, tmp_path, monkeypatch, command) -> None:
    """``complete``, ``control-only`` and ``decided`` say no with exit 3; any other non-zero exit
    is a failure that stops the driver (its verdict still written)."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, **{command.replace("-", "_"): 2})
    assert code == 2
    assert calls[-2:] == [f"{command} {FA.name}", "verdict"]
    assert f"part {FA.name} rest" not in calls and f"finish {FA.name}" not in calls
    assert f"part {FA.name} control" not in calls


def test_the_driver_stops_after_a_technical_case(qualify, tmp_path, monkeypatch) -> None:
    """``finish`` exits 9 for a technical case and keeps its checkpoints, exported to the Pod's
    volume: the driver uploads that export (scripts/archive.py), verifies it (``kept-check``) and
    stops there, before the next case."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, finish=qualify.TECHNICAL_CASE)
    assert code == qualify.TECHNICAL_CASE
    assert calls[-4:] == [f"finish {FA.name}", "archive", f"kept-check {FA.name}", "verdict"]
    assert not any(ADAMW.name in call for call in calls)


def test_the_driver_exports_and_verifies_a_technical_case_before_it_stops(
    qualify, tmp_path, monkeypatch
) -> None:
    """A technical case: ``finish --kept "$KEPT"`` exits 9 once its checkpoints are exported to the
    Pod's volume and verified there; scripts/archive.py uploads the case's folder of that export
    to ``$ARTIFACT_REMOTE/<--kept-remote>`` with a new inventory under ``$RUNS_DIR/archive/``, and
    ``kept-check`` reads that inventory; the driver then stops with 9. Every case's ``finish``
    line is written so, both of FA's branches (``control-only`` 0 and 3) and AdamW's."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, finish=qualify.TECHNICAL_CASE,
                             control_only=0)  # fmt: skip
    assert code == qualify.TECHNICAL_CASE
    assert calls == [*_with_case(["complete", "control-only", "part control", "finish"], FA),
                     "archive", f"kept-check {FA.name}", "verdict"]  # fmt: skip
    lines = (tmp_path / "stub.log").read_text().splitlines()
    (finish,) = [line for line in lines if line.startswith("finish ")]
    kept = tmp_path / "volume" / "kept"
    assert finish.endswith(f"--kept {kept}") and kept.is_dir()
    (archive,) = [line for line in lines if line.startswith("archive ")]
    folder = f"qualification/c1/numerical/{FA.name}"
    assert archive == f"archive gs://bucket/zfr/kept-checkpoints {kept} {folder}"
    (check,) = [line for line in lines if line.startswith("kept-check ")]
    inventory = re.search(r"--inventory (\S+)", check)[1]
    assert re.fullmatch(rf"{re.escape(str(tmp_path / 'runs' / 'archive'))}/"
                        r"inventory-kept-\d{8}T\d{6}Z\.tsv", inventory)  # fmt: skip
    destination = archive.split()[1]  # archive.py's REMOTE_PREFIX
    assert re.search(r"--remote (\S+)", check)[1] == destination
    text = (tmp_path / "driver.sh").read_text()
    commands = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    # Per case, one export sequence after control-only's 9 and one after the case: one generator
    assert commands.count(" finish --campaign ") == 4
    assert commands.count('--kept "$KEPT" || rc=$?') == 4
    assert commands.count("scripts/archive.py ") == 4
    assert commands.count("qualify.py kept-check ") == 4
    output = (tmp_path / "driver.out").read_text()  # each routed step's start and done markers
    for command, code in (("scripts/qualify.py finish", 9), ("scripts/archive.py", 0),
                          ("scripts/qualify.py kept-check", 0)):  # fmt: skip
        positions(output, [rf"start: project-run \S+ python {re.escape(command)} ",
                           rf"done \(exit {code}\): project-run \S+ python "
                           rf"{re.escape(command)} "])  # fmt: skip
    (tmp_path / "remote").mkdir()
    out = tmp_path / "remote" / "driver.sh"
    qualify.main([*driver_args(tmp_path, campaign(tmp_path)), "--kept-remote",
                  "dryrun/dry-1/kept-checkpoints", "--commit", "a" * 40,
                  "--out", str(out)])  # fmt: skip
    text = out.read_text()  # one destination, the upload's and the check's
    assert text.count('DEST="$ARTIFACT_REMOTE"/dryrun/dry-1/kept-checkpoints\n') == 4
    assert text.count('scripts/archive.py "$INV" "$DEST" "$KEPT" ') == 4
    assert text.count('--inventory "$INV" --remote "$DEST" ') == 4


@pytest.mark.parametrize("exits", [{"archive": 1}, {"kept_check": 13},
                                   {"archive": 1, "kept_check": 13}])  # fmt: skip
def test_the_driver_stops_with_13_when_the_bucket_copy_is_not_verified(
    qualify, tmp_path, monkeypatch, exits
) -> None:
    """The upload fails (scripts/archive.py exits 1) or ``kept-check`` does not verify it (13):
    ``kept-check`` runs whatever archive.py's exit, so its receipt records the pending upload, and
    the driver stops with 9 only when both exit 0; otherwise with 13 (the planned teardown waits,
    the hard deadline stays), going on to no other case. ``finish``'s own 13 (no verified export
    on the volume) stops it with 13 too, naming the checkpoints still on the container disk."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, finish=qualify.TECHNICAL_CASE,
                             **exits)  # fmt: skip
    assert code == qualify.NOT_EXPORTED == 13
    assert calls[-4:] == [f"finish {FA.name}", "archive", f"kept-check {FA.name}", "verdict"]
    assert not any(ADAMW.name in call for call in calls)
    output = (tmp_path / "driver.out").read_text()
    assert "do not stop the Pod by plan; the hard deadline stays" in output
    assert "if kept-check published a receipt for this export, it says whether" in output
    assert "its kept-check receipt says" not in output
    (tmp_path / "again").mkdir()
    code, calls = run_driver(qualify, tmp_path / "again", monkeypatch,
                             finish=qualify.NOT_EXPORTED)  # fmt: skip
    assert code == qualify.NOT_EXPORTED and calls[-2:] == [f"finish {FA.name}", "verdict"]
    assert "still only on the container disk" in (tmp_path / "again" / "driver.out").read_text()


@pytest.mark.parametrize("bucket", [{"ARTIFACT_REMOTE": None}, {"ARTIFACT_REMOTE": "/workspace/x"},
                                    {"ARTIFACT_GCS_KEY": None}, {"ARTIFACT_GCS_KEY": "/no/key"},
                                    {"gcloud": None}])  # fmt: skip
def test_the_driver_checks_the_bucket_settings_before_the_first_case(
    qualify, tmp_path, monkeypatch, bucket
) -> None:
    """Without ``ARTIFACT_REMOTE`` (a gs:// path), a readable ``ARTIFACT_GCS_KEY`` file or
    ``gcloud`` on PATH, a technical case's kept checkpoints could not reach the bucket: the
    driver exits 1 beside its GPU and memory checks, before the first case (no ``complete``);
    its verdict is still written. ``driver`` itself needs ``--kept`` as an absolute folder outside
    the checkpoint directory."""
    code, calls = run_driver(qualify, tmp_path, monkeypatch, bucket=bucket)
    assert code == 1 and calls == ["verdict"]
    text = (tmp_path / "driver.sh").read_text()
    positions(text, [r"nvidia-smi", r"60 GB of free host memory", r"ARTIFACT_REMOTE",
                     r"could not reach the bucket", r"--case "])  # fmt: skip
    path = campaign(tmp_path)
    for kept in ([], ["--kept", "volume/kept"], ["--kept", str(tmp_path / "ckpt" / "kept")]):
        with pytest.raises(SystemExit, match="--kept"):
            qualify.main(["driver", *common(path), *kept, "--dry-run"])


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
    with pytest.raises(SystemExit, match="does not exist"):  # an existing --out
        qualify.main([*driver_args(tmp_path, path), "--commit", "a" * 40,
                      "--out", str(path)])  # fmt: skip


@pytest.mark.parametrize("where", ["same device", "under RUNS_DIR",
                                   "under the checkpoint directory"])  # fmt: skip
def test_the_driver_checks_the_kept_folder_before_the_first_case(
    qualify, tmp_path, monkeypatch, where
) -> None:
    """``$KEPT`` on the checkpoint directory's device (``stat -L -c %d``: the container disk),
    or under ``$RUNS_DIR`` or the checkpoint directory once resolved (a symbolic link here): the
    driver exits 1 beside its bucket preflight, before the first case (no ``complete``)."""
    kept, exits = None, {}
    if where == "same device":
        exits = {"device": 1}
    elif where == "under RUNS_DIR":
        kept = tmp_path / "runs" / "kept"
    else:
        (tmp_path / "ckpt" / "kept").mkdir(parents=True)
        kept = tmp_path / "link"
        kept.symlink_to(tmp_path / "ckpt" / "kept")
    code, calls = run_driver(qualify, tmp_path, monkeypatch, kept=kept, **exits)
    assert code == 1 and calls == ["verdict"]
    output = (tmp_path / "driver.out").read_text()
    assert "the qualification does not start" in output and "KEPT" in output


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


def test_a_rest_bound_to_another_baseline_keeps_the_case_open(
    qualify, tmp_path, monkeypatch
) -> None:
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
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as stop:  # complete, and technical: its checkpoints stay
        qualify.main(finish_args(path, ADAMW, kept))
    assert stop.value.code == qualify.TECHNICAL_CASE
    assert (tmp_path / "ckpt" / "r-a1.pt").exists()


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
     ("complete", []), ("decided", []), ("control-only", []), ("clear-stale", []),
     ("locate", []), ("finish", []), ("kept-check", ["--inventory", "inventory.tsv", "--remote", "gs://b/k"]),
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
    qualify.main([*driver_args(tmp_path, path), "--dry-run"])
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
    from test_launch import INCIDENT, make_run_dir

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

    monkeypatch.setattr(launch, "_disqualification", checkpoint_lost)  # a clean run, file gone
    with pytest.raises(launch.CampaignError, match="cannot stand"):  # before any recovery
        launch.run_campaign(name, runs, root, execute=execute, recovery=RESTART, **only)
    qualify.main(restart(path, ADAMW, RESTART))
    restart_record = json.loads((case_folder(out, ADAMW) / "restart-1.json").read_text())
    assert restart_record["launcher"] == {"run_id": "r-a1", "identity": identity}
    monkeypatch.setattr(launch, "_disqualification", lambda run, run_dir, meta: None)
    result = launch.run_campaign(name, runs, root, execute=execute, recovery=RESTART,
                                 incident=INCIDENT, **only)  # fmt: skip
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


SECOND = "the Pod stopped again during the case; its container disk is gone again"


def with_checkpoint(run_dir: Path, checkpoint: Path, data: bytes = b"w") -> Path:
    """The launcher run's checkpoint written at ``checkpoint`` and recorded with its sha256 in
    its summary and meta.json (``settled_launcher`` records a lost /x.pt)."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(data)
    for name in ("meta.json", "summary.json"):
        text = (run_dir / name).read_text().replace('"/x.pt"', json.dumps(str(checkpoint)))
        text = text.replace('"' + "c" * 64 + '"', json.dumps(hashlib.sha256(data).hexdigest()))
        (run_dir / name).write_text(text)
    return run_dir


def checkpoint_lost(run, run_dir, meta):
    """The launcher's verdict on a settled launcher run, as ``_checkpoint_problem`` gives it for
    its checkpoint: done while the file is there, a conflict once it is gone."""
    record = json.loads((run_dir / "summary.json").read_text())["endpoint_checkpoint"]
    return (
        None
        if Path(record["path"]).is_file()
        else f"the endpoint checkpoint {record['path']} is missing"
    )


def launch_script():
    spec = importlib.util.spec_from_file_location("launch_script", ROOT / "scripts" / "launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_second_loss_restarts_the_case_again_through_a_further_round(
    qualify, stubbed, tmp_path, monkeypatch
) -> None:
    """The container disk lost twice in one case. After restart 1 and round 1, the new launcher
    run a2 is done, baseline attempt 2 recorded and rest attempt 2 interrupted, and a2's
    checkpoint is gone: the launcher refuses (exit 1, "is missing"); ``restart-case`` with the
    second interruption names every attempt in ``restart-2.json`` and logs a2 superseded; a
    round with round 1's reason is refused, one with the new reason is round 2 and runs a3,
    which ``locate`` finds; the parts run as their attempt 3, and the verdict voids baseline 1
    and 2, each with its own restart's reason."""
    from test_launch import INCIDENT

    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    label = f"adamw-{ADAMW_LR:g}/s10200"
    run = journal(root, path, label, ["r-a1"])
    old = settled_launcher(root, run, "r-a1")  # its checkpoint, /x.pt, is gone
    baseline, _ = synthetic(ADAMW, identity=q.launcher_identity(old))
    publish(out, ADAMW, "baseline", baseline)
    q.publish(out / (ADAMW.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    name, runs = launch.load_campaign(path)
    only = {"only": label, "max_attempts": 1}
    monkeypatch.setattr(launch, "_disqualification", checkpoint_lost)

    def execute(run, run_id, output, campaign=None):
        with_checkpoint(settled_launcher(root, run, run_id), tmp_path / "ckpt" / f"{run_id}.pt")
        return 0

    qualify.main(restart(path, ADAMW, RESTART))
    result = launch.run_campaign(name, runs, root, execute=execute, recovery=RESTART,
                                 incident=INCIDENT, **only)  # fmt: skip
    (a2,) = [d for d in (root / run.experiment).iterdir() if d.name != "r-a1"]
    assert result["done"] == [label] and qualify.located(run, root) == a2
    second, _ = synthetic(ADAMW, identity=q.launcher_identity(a2))
    publish(out, ADAMW, "baseline", second, attempt=2)
    q.publish(out / (ADAMW.attempt_path("rest", 2) + ".started"), {"started_at": "t"})
    (tmp_path / "ckpt" / f"{a2.name}.pt").unlink()  # the container disk lost again
    with pytest.raises(SystemExit) as stop:  # the driver's launcher call
        launch_script().main([str(path), "--only", label, "--max-attempts", "1"])
    assert "cannot stand" in str(stop.value.code) and "is missing" in str(stop.value.code)
    qualify.main(restart(path, ADAMW, SECOND))
    folder = case_folder(out, ADAMW)
    record = json.loads((folder / "restart-2.json").read_text())
    assert [(a["part"], a["attempt"], a["state"]) for a in record["attempts"]] == [
        ("baseline", 1, "recorded"), ("baseline", 2, "recorded"), ("rest", 1, "interrupted"),
        ("rest", 2, "interrupted")]  # fmt: skip
    assert record["launcher"]["run_id"] == a2.name and record["checkpoint_missing"] is True
    assert a2.name in qualify.label_superseded(run, root)
    with pytest.raises(launch.CampaignError, match=r"round 1 \("):
        launch.run_campaign(name, runs, root, execute=execute, recovery=RESTART,
                            incident=INCIDENT, **only)  # fmt: skip
    result = launch.run_campaign(name, runs, root, execute=execute, recovery=SECOND,
                                 incident=INCIDENT, **only)  # fmt: skip
    (a3,) = [d for d in (root / run.experiment).iterdir() if d.name not in ("r-a1", a2.name)]
    assert result["done"] == [label] and a3.name.endswith("-a3")
    assert qualify.located(run, root) == a3 and old.is_dir() and a2.is_dir()
    rounds = [r for r in launch.launch_records(root) if r.get("event") == "recovery"]
    assert [(r["round"], r["reason"], r["after"]) for r in rounds] == [
        (1, RESTART, ["r-a1"]), (2, SECOND, ["r-a1", a2.name])]  # fmt: skip
    for part_name in ("baseline", "rest"):
        qualify.main(["part", *common(path), "--case", ADAMW.name, "--part", part_name,
                      "--attempt", "3", "--infrastructure", SECOND, "--run-dir", str(a3),
                      "--checkpoints", str(tmp_path / "h"), "--device", "cpu"])  # fmt: skip
    assert stubbed == [(ADAMW.name, "baseline", 3), (ADAMW.name, "rest", 3)]
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    assert [(e["part"], e["attempt"], e["restart"]) for e in verdict["superseded"]] == [
        ("baseline", 1, RESTART), ("baseline", 2, SECOND)]  # fmt: skip
    rate = verdict["rates"][f"adamw-{ADAMW_LR:g}"]
    assert sorted(v["reason"] for v in rate["voided"]) == sorted([RESTART, SECOND])


def test_a_study3_round_is_not_refused_as_unused_after_a_restart(
    qualify, tmp_path, monkeypatch
) -> None:
    """The documented sequence under a study-3 campaign header (the real campaign file's): the
    launcher's conflict over the lost checkpoint, ``restart-case``, then the run's recovery
    round, ``--only LABEL --max-attempts 1 --recovery`` with the incident fields, which the
    refusal of a round over a run that is not used up does not stop (the restart left the run
    used up); a second loss takes a further round the same way."""
    from test_launch import INCIDENT

    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    path.write_text("header:\n  study: 3\n" + path.read_text())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    label = f"adamw-{ADAMW_LR:g}/s10200"
    run = journal(root, path, label, ["r-a1"], only=label, max_attempts=1)
    old = settled_launcher(root, run, "r-a1")  # its checkpoint, /x.pt, is gone
    baseline, _ = synthetic(ADAMW, identity=q.launcher_identity(old))
    publish(out, ADAMW, "baseline", baseline)
    q.publish(out / (ADAMW.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    name, runs = launch.load_campaign(path)
    header = launch.campaign_header(path)
    assert header == {"study": 3}
    monkeypatch.setattr(launch, "_disqualification", checkpoint_lost)
    form = [str(path), "--only", label, "--max-attempts", "1"]

    def execute(run, run_id, output, campaign=None):
        with_checkpoint(settled_launcher(root, run, run_id), tmp_path / "ckpt" / f"{run_id}.pt")
        return 0

    with pytest.raises(SystemExit, match="is missing"):  # the loss, before the restart
        launch_script().main([*form, "--recovery", RESTART])
    qualify.main(restart(path, ADAMW, RESTART))
    result = launch.run_campaign(name, runs, root, execute=execute, recovery=RESTART,
                                 header=header, only=label, max_attempts=1,
                                 incident=INCIDENT)  # fmt: skip
    (a2,) = [d for d in (root / run.experiment).iterdir() if d.name != "r-a1"]
    assert result["done"] == [label]
    second, _ = synthetic(ADAMW, identity=q.launcher_identity(a2))
    publish(out, ADAMW, "baseline", second, attempt=2)
    q.publish(out / (ADAMW.attempt_path("rest", 2) + ".started"), {"started_at": "t"})
    (tmp_path / "ckpt" / f"{a2.name}.pt").unlink()  # the container disk lost again
    with pytest.raises(SystemExit, match="is missing"):
        launch_script().main(form)
    qualify.main(restart(path, ADAMW, SECOND))
    result = launch.run_campaign(name, runs, root, execute=execute, recovery=SECOND,
                                 header=header, only=label, max_attempts=1,
                                 incident=INCIDENT)  # fmt: skip
    assert result["done"] == [label]


def test_a_launcher_run_lost_in_flight_needs_no_restart_only_a_round(
    qualify, tmp_path, monkeypatch
) -> None:
    """After restart 1 and round 1, the new launcher run a2 was in flight when the container
    disk was lost: a start, no end, no directory. ``restart-case`` has nothing to restart; the
    launcher settles a2 technical and, with the documented reason, opens round 2 and runs a3."""
    from test_launch import INCIDENT, log_start

    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    label = f"adamw-{ADAMW_LR:g}/s10200"
    run = journal(root, path, label, ["r-a1"])
    old = settled_launcher(root, run, "r-a1")
    baseline, _ = synthetic(ADAMW, identity=q.launcher_identity(old))
    publish(out, ADAMW, "baseline", baseline)
    q.publish(out / (ADAMW.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    name, runs = launch.load_campaign(path)
    only = {"only": label, "max_attempts": 1}
    monkeypatch.setattr(launch, "_disqualification", checkpoint_lost)
    qualify.main(restart(path, ADAMW, RESTART))
    result = launch.run_campaign(name, runs, root, stop=lambda: True, recovery=RESTART,
                                 incident=INCIDENT, **only)  # fmt: skip
    assert result["stopped"] == [label]  # round 1 logged; its attempt a2 starts next
    in_flight = f"20261027T051200Z-{run.cell}-s10200-{run.config_hash[:8]}-a2"
    log_start(root, name, run, in_flight, attempt=2, only=label, max_attempts=1)
    with pytest.raises(SystemExit, match="no attempt to restart"):
        qualify.main(restart(path, ADAMW, SECOND))
    assert not (case_folder(out, ADAMW) / "restart-2.json").exists()

    def execute(run, run_id, output, campaign=None):
        with_checkpoint(settled_launcher(root, run, run_id), tmp_path / "ckpt" / f"{run_id}.pt")
        return 0

    result = launch.run_campaign(name, runs, root, execute=execute, recovery=SECOND,
                                 incident=INCIDENT, **only)  # fmt: skip
    assert result["done"] == [label]
    records = launch.launch_records(root)
    (settled,) = [r for r in records if r.get("event") == "end" and r["run_id"] == in_flight]
    assert (settled["outcome"], settled["settled"]) == ("technical", True)
    rounds = [r for r in records if r.get("event") == "recovery"]
    assert [(r["round"], r["after"]) for r in rounds] == [(1, ["r-a1"]),
                                                          (2, ["r-a1", in_flight])]  # fmt: skip
    assert qualify.located(run, root).name.endswith("-a3")


def log_end(root, path, run, run_id, **fields) -> None:
    """An ``end`` line of the launcher run ``run_id`` in the campaign's launch journal."""
    log = root / "launch" / f"{launch_name(path)}.jsonl"
    record = {"event": "end", "run": run.label, "attempt": 1, "run_id": run_id,
              "returncode": 0, "outcome": "done", "problem": None, **fields}  # fmt: skip
    with log.open("a") as f:
        f.write(json.dumps(record) + "\n")


def launch_name(path) -> str:
    from zero_field_refusal.launch import load_campaign

    return load_campaign(path)[0]


def test_a_conflict_the_launcher_found_at_its_own_attempt_is_never_restarted(
    qualify, tmp_path
) -> None:
    """The launcher's attempt completed and its checkpoint was missing at once: the end it wrote
    (a return code, ``outcome: conflict``) is its own judgement, not a lost container disk (its
    process lives on the disk and would have ended with it). The checkpoint is gone, so a
    restart would otherwise be accepted: ``restart-case`` refuses, naming the run and the end,
    and supersedes nothing."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    settled_launcher(root, run, "r-a1")  # its checkpoint, /x.pt, is gone
    log_end(root, path, run, "r-a1", outcome="conflict",
            problem="the endpoint checkpoint /x.pt is missing")  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        qualify.main(restart(path, ADAMW, RESTART))
    message = str(stop.value.code)
    assert "r-a1" in message and "outcome conflict" in message
    assert "the launcher's own judgement of its attempt" in message and "never a restart" in message
    assert not list(case_folder(out, ADAMW).glob("restart-*"))
    assert not launch.superseded_runs(run, launch.launch_records(root))


def test_a_settled_conflict_of_another_problem_is_never_restarted(qualify, tmp_path) -> None:
    """A start found without its end and settled ``conflict`` for a problem other than the
    missing checkpoint (here a dirty checkout), the checkpoint gone as well: a provenance
    conflict, whatever the missing file allows. ``restart-case`` refuses and supersedes nothing."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    settled_launcher(root, run, "r-a1")  # its checkpoint, /x.pt, is gone
    log_end(root, path, run, "r-a1", outcome="conflict", returncode=None, settled=True,
            problem="not run from a clean checkout of a recorded commit")  # fmt: skip
    with pytest.raises(SystemExit) as stop:
        qualify.main(restart(path, ADAMW, RESTART))
    assert "r-a1" in str(stop.value.code) and "never a restart" in str(stop.value.code)
    assert not list(case_folder(out, ADAMW).glob("restart-*"))
    assert not launch.superseded_runs(run, launch.launch_records(root))


def test_the_launchers_own_conflict_end_is_never_restarted(qualify, tmp_path, monkeypatch) -> None:
    """The launcher itself, run through its entry under a study-3 header on a fresh attempt that
    completes with its checkpoint missing, writes the end (return code 0, ``conflict``, not
    settled) and stops; the next call shows the note. ``restart-case`` refuses on that end,
    writes no record and supersedes nothing."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    path.write_text("header:\n  study: 3\n" + path.read_text())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    label = f"adamw-{ADAMW_LR:g}/s10200"
    made = []

    def execute(run, run_id, output, campaign=None):
        settled_launcher(root, run, run_id)  # completed; its checkpoint, /x.pt, is absent
        made.append(run_id)
        return 0

    monkeypatch.setattr(launch, "_subprocess", execute)
    monkeypatch.setattr(launch, "_disqualification", checkpoint_lost)
    form = [str(path), "--only", label, "--max-attempts", "1"]
    with pytest.raises(SystemExit, match="completed but cannot stand"):
        launch_script().main(form)
    (run_id,) = made
    (end,) = [r for r in launch.launch_records(root)
              if r.get("event") == "end" and r["run_id"] == run_id]  # fmt: skip
    assert end["returncode"] == 0 and end["outcome"] == "conflict" and "settled" not in end
    with pytest.raises(SystemExit) as again:
        launch_script().main(form)
    assert "A lost checkpoint" in " ".join(str(again.value.code).split())
    with pytest.raises(SystemExit) as stop:
        qualify.main(restart(path, ADAMW, RESTART))
    assert "own judgement" in stop.value.code and run_id in stop.value.code
    assert not list(case_folder(out, ADAMW).glob("restart-*"))
    run = next(r for r in launch.load_campaign(path)[1] if r.label == label)
    assert not launch.superseded_runs(run, launch.launch_records(root))


@pytest.mark.parametrize(
    "end",
    [{"outcome": "done"},  # the attempt ended done; the disk was lost afterwards
     {"outcome": "conflict", "returncode": None, "settled": True,  # the next call settled it
      "problem": "the endpoint checkpoint /x.pt is missing"},
     {"outcome": "technical", "returncode": None, "settled": True}],
)  # fmt: skip
def test_a_real_disk_loss_is_not_caught_by_the_own_conflict_refusal(qualify, tmp_path, end) -> None:
    """After a lost container disk the journal holds an end ``done`` (written before the loss)
    or a settled end (no return code, written by a later call): ``restart-case`` is accepted as
    it is without the refusal, and logs the run superseded."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    settled_launcher(root, run, "r-a1")  # its checkpoint, /x.pt, is gone
    log_end(root, path, run, "r-a1", **end)
    qualify.main(restart(path, ADAMW, RESTART))
    assert list(case_folder(out, ADAMW).glob("restart-1.json"))
    assert set(launch.superseded_runs(run, launch.launch_records(root))) == {"r-a1"}


def test_a_restart_refuses_a_present_checkpoint_of_another_sha256(qualify, tmp_path) -> None:
    """The launcher run's endpoint checkpoint is on disk with another sha256 than its recorded
    one, and the rest was interrupted: a provenance conflict, not a loss. ``restart-case``
    refuses, naming it, and writes nothing (removing or replacing the file is no route to a
    restart: the conflict goes to diagnosis and the technical route)."""
    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    launcher = with_checkpoint(settled_launcher(root, run, "r-a1"), tmp_path / "ckpt" / "r-a1.pt")
    checkpoint = tmp_path / "ckpt" / "r-a1.pt"
    checkpoint.write_bytes(b"other weights")  # present, not the recorded bytes
    baseline, _ = synthetic(ADAMW, identity=q.launcher_identity(launcher))
    publish(out, ADAMW, "baseline", baseline)
    q.publish(out / (ADAMW.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    log = root / "launch" / "study3-qualification-c1.jsonl"
    before = log.read_bytes()
    with pytest.raises(SystemExit) as stop:
        qualify.main(restart(path, ADAMW, RESTART))
    assert stop.value.code not in (0, None)
    assert "another sha256" in str(stop.value.code) and str(checkpoint) in str(stop.value.code)
    assert "provenance conflict" in str(stop.value.code)
    assert not list(case_folder(out, ADAMW).glob("restart-*")) and log.read_bytes() == before


# ------------------------------------------------- a decided case with its control due

FA_LABEL = f"adam_frozen-{FA_LR:g}/s10200"
LOST = "the Pod stopped during the control; its container disk is gone"
UNSTABLE = {"reason": "gradient_spike", "step": 3, "detail": {"grad_norm": 50.0}}


def decided_world(tmp_path, *, rest=None):
    """FA's case decided by its baseline (a declared instability rule fired on the unshifted
    baseline trajectory), its baseline record bound to its conforming launcher run r-f1
    (classified, ``launcher.kind`` null), done in the launch journal; r-f1's endpoint checkpoint
    was written and recorded, then lost with the container disk. Its same-path control is due.
    With ``rest`` (a change of the rest record, or ``True``), the baseline passes instead and its
    rest is recorded, bound to it and to r-f1. Returns the campaign, the runs root, the records'
    folder and r-f1's directory."""
    path = campaign(tmp_path)
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, FA_LABEL, ["r-f1"])
    with (root / "launch" / "study3-qualification-c1.jsonl").open("a") as f:
        end = {"event": "end", "run": FA_LABEL, "attempt": 1, "run_id": "r-f1",
               "returncode": 0, "outcome": "done", "problem": None}  # fmt: skip
        f.write(json.dumps(end) + "\n")
    checkpoint = tmp_path / "ckpt" / "r-f1.pt"
    run_dir = with_checkpoint(settled_launcher(root, run, "r-f1"), checkpoint)
    baseline, rest_record = synthetic(FA, identity=q.launcher_identity(run_dir))
    if rest is None:
        baseline["trajectories"]["baseline"]["instability"] = dict(UNSTABLE)
    baseline["launcher_run"] = str(run_dir)
    baseline["launcher"]["classified"] = True
    publish(out, FA, "baseline", baseline)
    decided = q.case_verdict(baseline, None, SCOPE)["kind"] == "scientific"
    assert decided is (rest is None)
    if rest is not None:
        rest_record["launcher_run"] = str(run_dir)
        rest_record["launcher"]["classified"] = True
        if callable(rest):
            rest(rest_record)
        rest_record["baseline_record"] = {"path": str(out / FA.attempt_path("baseline", 1)),
                                          "sha256": q.record_sha256(baseline)}  # fmt: skip
        publish(out, FA, "rest", rest_record)
    checkpoint.unlink()  # the container disk lost
    return path, root, out, run_dir


def control_only(path, case=FA):
    return ["control-only", *common(path), "--case", case.name]


def edit_baseline(out, change):
    """FA's baseline record, changed by ``change`` and published again as attempt 1."""
    record = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
    change(record)
    publish(out, FA, "baseline", record)


def control_args(path, run_dir, tmp_path, *mode):
    return ["part", *common(path), "--case", FA.name, "--part", "control", *mode,
            "--run-dir", str(run_dir), "--checkpoints", str(tmp_path / "h"),
            "--device", "cpu"]  # fmt: skip


def test_control_only_binds_the_decided_baseline_to_its_launcher_run_without_the_checkpoint(
    qualify, stubbed, tmp_path, monkeypatch, capsys
) -> None:
    """A decided case whose same-path control is due, its launcher run's checkpoint gone with the
    container disk: ``control-only`` binds the baseline record to the located run by its
    recorded identity (the checkpoint record), reads no checkpoint file, prints that run's
    directory and exits 0; ``part --part control --resume`` then runs attempt 1 on it."""
    path, root, out, run_dir = decided_world(tmp_path)

    def untouched(*args, **kwargs):
        raise AssertionError(f"called with {args}")

    monkeypatch.setattr(qualify, "checkpoint_sha256", untouched)  # an absent file: none is read
    monkeypatch.setattr(q, "file_sha256", untouched)
    capsys.readouterr()
    qualify.main(control_only(path))
    printed = capsys.readouterr().out.strip()
    assert printed == str(run_dir)
    qualify.main(control_args(path, printed, tmp_path, "--resume"))
    assert stubbed == [(FA.name, "control", 1)]
    started = json.loads((out / (FA.attempt_path("control", 1) + ".started")).read_text())
    assert started["run_dir"] == str(run_dir) and started["infrastructure"] is None
    assert not (tmp_path / "ckpt" / "r-f1.pt").exists()


def test_control_only_says_no_outside_its_state(qualify, tmp_path, capsys) -> None:
    """``control-only`` exits 3, and the driver takes its usual path, for a case that is not
    decided, one that holds no same-path control, one the engineering cycle resolved, one whose
    control is recorded (the case complete) or whose latest control attempt is neither due nor
    interrupted (an unreadable record), and one decided by its launcher run's own declared
    instability stop (the launcher needs no checkpoint for it, and goes on as before)."""
    path, root, out, run_dir = decided_world(tmp_path)

    def no(case, why):
        capsys.readouterr()
        assert exit_code(qualify, control_only(path, case)) == qualify.NO
        assert why in capsys.readouterr().err

    assert exit_code(qualify, control_only(path)) == 0
    edit_baseline(out, lambda b: b["trajectories"]["baseline"].pop("instability"))
    no(FA, "not decided")
    edit_baseline(out, lambda b: b["trajectories"]["baseline"].update(instability=UNSTABLE))
    write_case(out, ADAMW, decided=True)
    no(ADAMW, "holds no same-path control")
    cycle = out / "c1" / "supersede-1.json"  # the cycle kept FA on its launcher run, unresolved
    resolution = {"run_id": "r-f1", "outcome": "unresolved", "reasons": ["no recorded commit"]}
    q.publish(cycle, {"kept_cases": [{"case": FA.name, "stage": "numerical",
                                      "resolution": resolution}]})  # fmt: skip
    no(FA, "the engineering cycle kept the case on its launcher run r-f1 (unresolved)")
    cycle.unlink()
    assert exit_code(qualify, control_only(path)) == 0
    publish(out, FA, "control", {"part": "rest", "case": FA.record()})  # not a control's record
    no(FA, "attempt 1 is invalid, neither due nor interrupted")
    publish(out, FA, "control", control_record(case=FA, identity=q.launcher_identity(run_dir)))
    no(FA, "decided by the baseline")  # complete: nothing is outstanding
    (out / FA.attempt_path("control", 1)).unlink()
    assert exit_code(qualify, control_only(path)) == 0

    def launcher_stop(record):
        """The launcher run stopped by its declared rule: the baseline ran nothing."""
        record.update(status="not_run", kind="scientific", reasons=["instability: r-f1"])
        record["launcher"].update(kind="scientific", reasons=["instability: r-f1"])
        record["trajectories"] = {}

    edit_baseline(out, launcher_stop)
    no(FA, "its launcher run's own declared instability stop")


def fa_run(path):
    from zero_field_refusal.launch import load_campaign

    return next(r for r in load_campaign(path)[1] if r.label == FA_LABEL)


def superseded(path, root, out, run_dir):
    from zero_field_refusal.launch import supersede

    supersede(root, "study3-qualification-c1", fa_run(path), run_dir.name, reason="a test",
              record=str(out / "restart-1.json"))  # fmt: skip


def ended(outcome):
    def change(path, root, out, run_dir):
        log = root / "launch" / "study3-qualification-c1.jsonl"
        log.write_text(log.read_text().replace('"outcome": "done"', f'"outcome": "{outcome}"'))

    return change


def no_identity_either_side(path, root, out, run_dir):
    """The record names no identity, and the run's records cannot be read (its metrics a
    folder): ``q.launcher_identity`` is None on both sides."""
    edit_baseline(out, lambda record: record["launcher"].update(identity=None))
    (run_dir / "metrics.jsonl").mkdir()
    assert q.launcher_identity(run_dir) is None


def baseline_change(change):
    return lambda path, root, out, run_dir: edit_baseline(out, lambda b: change(b, run_dir))


UNBOUND = {  # what the decided case's binding lacks → how control-only names it
    "unclassified": (baseline_change(lambda b, d: b["launcher"].update(classified=False)),
                     "did not classify its launcher run"),
    "kind": (baseline_change(lambda b, d: b["launcher"].update(kind="technical")),
             "launcher.kind is 'technical'"),
    "other run": (baseline_change(lambda b, d: b.update(launcher_run=str(d.parent / "r-f0"))),
                  "not the located run"),
    "identity": (baseline_change(lambda b, d: b["launcher"]["identity"].update(sha256="0" * 64)),
                 "identity is not the one the baseline record names"),
    "no identity": (baseline_change(lambda b, d: b["launcher"].update(identity=None)),
                    "names no launcher identity"),
    "no identity on either side": (no_identity_either_side, "names no launcher identity"),
    "superseded": (superseded, "superseded in the launch journal"),
    "end": (ended("conflict"), "not done"),
}  # fmt: skip


@pytest.mark.parametrize("missing", list(UNBOUND))
def test_control_only_refuses_a_decided_case_it_cannot_bind(qualify, tmp_path, missing) -> None:
    """A decided case whose control is due, but whose baseline record cannot be bound to its
    launcher run without the launcher (not classified, not conforming, another run, another
    identity, the run superseded, its journal end not done): ``control-only`` exits 1 naming
    the condition, and nothing runs."""
    path, root, out, run_dir = decided_world(tmp_path)
    assert exit_code(qualify, control_only(path)) == 0
    change, why = UNBOUND[missing]
    change(path, root, out, run_dir)
    code = exit_code(qualify, control_only(path))
    assert isinstance(code, str) and why in code, code  # exit 1, with its message
    assert "nothing is rerun" in code
    assert not (out / (FA.attempt_path("control", 1) + ".started")).exists()


def test_control_only_stops_on_a_present_checkpoint_of_another_hash(qualify, tmp_path) -> None:
    """The launcher run's endpoint checkpoint on disk with another sha256 than its recorded one: a
    provenance conflict (``checkpoint_conflict``, as ``restart-case`` reads it). ``control-only``
    exits 1 naming the file and both hashes, and the file stays as it is, for the diagnosis; with
    the recorded bytes, the file present, it exits 0."""
    path, root, out, run_dir = decided_world(tmp_path)
    checkpoint = tmp_path / "ckpt" / "r-f1.pt"
    recorded = q.launcher_identity(run_dir)["checkpoint"]["sha256"]
    checkpoint.write_bytes(b"other weights")
    present = hashlib.sha256(b"other weights").hexdigest()
    code = exit_code(qualify, control_only(path))
    assert isinstance(code, str), code  # exit 1, with its message
    assert str(checkpoint) in code and recorded in code and present in code
    assert "kept" in code and "diagnosis" in code and "nothing is rerun" in code
    assert checkpoint.read_bytes() == b"other weights"
    assert not (out / (FA.attempt_path("control", 1) + ".started")).exists()
    checkpoint.write_bytes(b"w")  # the recorded bytes (``with_checkpoint``)
    assert hashlib.sha256(b"w").hexdigest() == recorded
    assert exit_code(qualify, control_only(path)) == 0


def test_control_only_matches_its_run_wherever_runs_dir_lies(qualify, tmp_path) -> None:
    """The baseline record names its launcher run as ``<RUNS_DIR>/<experiment>/<run id>`` of the
    root it ran under: the same experiment and run id under another root is the located run (its
    identity still binds the content); another run id, or another experiment, is not."""
    path, root, out, run_dir = decided_world(tmp_path)
    elsewhere = Path("/restored/elsewhere/runs") / run_dir.parent.name / run_dir.name
    edit_baseline(out, lambda record: record.update(launcher_run=str(elsewhere)))
    assert exit_code(qualify, control_only(path)) == 0
    for named in (elsewhere.parent / "r-f0", elsewhere.parent.parent / "other" / run_dir.name):
        edit_baseline(out, lambda record, named=named: record.update(launcher_run=str(named)))
        code = exit_code(qualify, control_only(path))
        assert isinstance(code, str) and "not the located run" in code, code


def test_control_only_answers_as_complete_first(qualify, tmp_path, monkeypatch) -> None:
    """Before its own checks, ``control-only`` answers as ``complete`` and ``decided`` do: a
    stale case exits 10 and a held label 11 (a journal that is not whole, 12:
    ``test_every_subcommand_that_reads_the_runs_refuses_over_a_torn_line``)."""
    path, root, out, run_dir = decided_world(tmp_path)
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(FIX))
    assert exit_code(qualify, control_only(path)) == qualify.STALE
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(CODE))
    assert exit_code(qualify, control_only(path)) == 0
    run = fa_run(path)
    with (root / "launch" / "study3-qualification-c1.jsonl").open("a") as f:
        f.write(json.dumps({"event": "held", "run": FA_LABEL, "experiment": run.experiment,
                            "config_hash": run.config_hash, "run_id": "r-f0",
                            "reason": "unresolved"}) + "\n")  # fmt: skip
    assert exit_code(qualify, control_only(path)) == qualify.HELD


def test_an_interrupted_control_of_a_decided_case_is_rerun_by_hand_after_a_lost_disk(
    qualify, stubbed, tmp_path
) -> None:
    """The control of a decided case was interrupted, and the container disk is gone: ``part
    --resume`` exits 8 (``control-only`` still binds the case, so the driver stops there); a
    restart is refused (the scientific baseline keeps counting); the control's next attempt runs
    by hand with ``--infrastructure``, without any checkpoint on disk, and its record carries
    the reason."""
    path, root, out, run_dir = decided_world(tmp_path)
    q.publish(out / (FA.attempt_path("control", 1) + ".started"),
              {"started_at": "t", "run_dir": str(run_dir)})  # fmt: skip
    assert exit_code(qualify, control_args(path, run_dir, tmp_path, "--resume")) == (
        qualify.INTERRUPTED)  # fmt: skip
    assert exit_code(qualify, control_only(path)) == 0
    with pytest.raises(SystemExit, match=r"completed failure.*baseline attempt 1.*scientific"):
        qualify.main(restart(path, FA, LOST))
    assert not list(case_folder(out, FA).glob("restart-*"))
    assert stubbed == []
    qualify.main(control_args(path, run_dir, tmp_path, "--attempt", "2", "--infrastructure", LOST))
    assert stubbed == [(FA.name, "control", 2)]
    record = json.loads((out / FA.attempt_path("control", 2)).read_text())
    assert record["infrastructure"]["reason"] == LOST
    assert record["infrastructure"]["previous"] == [{"attempt": 1, "state": "interrupted"}]
    assert not any((tmp_path / "ckpt").glob("*.pt"))


def test_a_failed_control_of_a_decided_case_is_kept(
    qualify, stubbed, tmp_path, monkeypatch
) -> None:
    """A decided case's control recorded technical (K beyond 0.01·K*): it is never rerun, by
    hand or by ``control-only``; the case is complete, so no restart; ``finish`` keeps its
    checkpoints for the diagnosis (exit 9), the launcher run's, lost with the container disk,
    listed as lost in its export's manifest."""
    path, root, out, run_dir = decided_world(tmp_path)
    failed = control_record(0.1, case=FA, identity=q.launcher_identity(run_dir))
    assert q.control_verdict(failed, SCOPE, 1)["kind"] == "technical"
    publish(out, FA, "control", failed)
    with pytest.raises(SystemExit, match="is never rerun"):
        qualify.main(control_args(path, run_dir, tmp_path, "--attempt", "2",
                                  "--infrastructure", LOST))  # fmt: skip
    assert exit_code(qualify, control_only(path)) == qualify.TECHNICAL_CASE  # its export first
    with pytest.raises(SystemExit, match="the case is complete"):
        qualify.main(restart(path, FA, LOST))
    assert not list(case_folder(out, FA).glob("restart-*"))
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    manifest = json.loads((case_folder(out, FA) / "kept-checkpoints-1.json").read_text())
    assert [entry["path"] for entry in manifest["lost"]] == [str(tmp_path / "ckpt" / "r-f1.pt")]
    assert stubbed == []


# --------------------------------- a case whose baseline and rest are recorded, its control due


def accepted_control(calls, run_dir):
    """``run_part`` stubbed: the control's record as an accepted control on ``run_dir`` (its
    identity, K far below 0.01·K*); nothing loads a model."""

    def run_part(settings, env, case, run_dir_, out, *, part, attempt, checkpoints,
                 diagnostic_steps=(), infrastructure=None, engineering_cycle=None):  # fmt: skip
        calls.append((case.name, part, attempt, str(run_dir_)))
        record = {**control_record(case=case, identity=q.launcher_identity(run_dir)),
                  "attempt": attempt, "launcher_run": str(run_dir_)}  # fmt: skip
        q.publish(out / case.attempt_path(part, attempt), record)
        return record

    return run_part


def scientific_rest(rest):
    """A rest whose baseline~reference D₂₅ is beyond 0.01·K*: a scientific failure."""
    rest["comparisons"]["baseline~reference"]["D_twin"] = 0.5


def technical_rest(rest):
    """A rest whose unshifted repeat is beyond 0.01·K*: a technical mark."""
    rest["comparisons"]["baseline~repeat"]["D_twin"] = 0.07


def run_the_control_alone(qualify, monkeypatch, capsys, path, run_dir, tmp_path):
    """The driver's calls for a case ``control-only`` takes, by hand: ``control-only`` (no
    checkpoint file read), ``part --part control --resume`` on the run it printed (an accepted
    control) and ``finish``. Returns the calls ``run_part`` saw."""
    calls = []

    def untouched(*args, **kwargs):
        raise AssertionError(f"called with {args}")

    monkeypatch.setattr(qualify.runtime, "configure_numerics", lambda cfg, device: {})
    monkeypatch.setattr(qualify.runtime, "environment", lambda cfg, device: object())
    monkeypatch.setattr(qualify.q, "run_part", accepted_control(calls, run_dir))
    with monkeypatch.context() as guard:  # an absent file: none is read
        guard.setattr(qualify, "checkpoint_sha256", untouched)
        guard.setattr(q, "file_sha256", untouched)
        capsys.readouterr()
        qualify.main(control_only(path))
        printed = capsys.readouterr().out.strip()
    assert printed == str(run_dir)
    qualify.main(control_args(path, printed, tmp_path, "--resume"))
    assert calls == [(FA.name, "control", 1, str(run_dir))]
    qualify.main(["complete", *common(path), "--case", FA.name])  # exit 0: complete now
    assert exit_code(qualify, finish_args(path, FA)) == 0
    return calls


def test_control_only_runs_the_control_alone_after_a_recorded_rest(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """A passing baseline and a passing rest, bound to r-f1, its checkpoint lost with the container
    disk, the same-path control due: ``control-only`` exits 0 without reading a checkpoint file,
    the control runs on the run it printed, and ``finish`` follows (the driver's 0 branch:
    ``test_a_decided_case_with_its_control_due_runs_the_control_alone``)."""
    path, root, out, run_dir = decided_world(tmp_path, rest=True)
    run_the_control_alone(qualify, monkeypatch, capsys, path, run_dir, tmp_path)
    assert not (tmp_path / "ckpt" / "r-f1.pt").exists()


def test_a_scientific_rest_failure_keeps_counting_when_the_control_runs_alone(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """A passing baseline, then a rest that records a scientific failure, the control due and the
    checkpoint lost: ``control-only`` exits 0, the control runs on that run and ``finish``
    follows. The verdict afterwards shows the rest's scientific failure as it was: the control's
    record neither voids, replaces nor softens it, and the rest record is unchanged."""
    path, root, out, run_dir = decided_world(tmp_path, rest=scientific_rest)
    baseline = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
    rest_path = out / FA.attempt_path("rest", 1)
    rest, rest_bytes = json.loads(rest_path.read_text()), rest_path.read_bytes()
    before = q.case_verdict(baseline, rest, SCOPE)
    assert (before["state"], before["kind"]) == ("failing", "scientific")
    run_the_control_alone(qualify, monkeypatch, capsys, path, run_dir, tmp_path)
    assert rest_path.read_bytes() == rest_bytes
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    key = f"adam_frozen-{FA_LR:g}"
    (case,) = [c for c in verdict["cases"] if c["case"]["name"] == FA.name]
    assert (case["state"], case["kind"], case["reasons"]) == (
        "failing", "scientific", before["reasons"])  # fmt: skip
    assert case["comparisons"] == before["comparisons"]
    assert verdict["rates"][key]["control"]["state"] == "accepted"
    assert verdict["states"][key] == "failing"
    assert verdict["superseded"] == [] and verdict["rates"][key]["voided"] == []


def rest_change(change):
    return lambda path, root, out, run_dir: edit_part(out, "rest", lambda r: change(r, run_dir))


def edit_part(out, part, change):
    """FA's latest record of ``part`` changed by ``change`` and published again as attempt 1."""
    record = json.loads((out / FA.attempt_path(part, 1)).read_text())
    change(record)
    publish(out, FA, part, record)


def baseline_elsewhere(path, root, out, run_dir):
    """The baseline names another launcher run (r-f0); the rest is bound to that baseline record
    again, so that only the launcher run differs."""
    edit_part(out, "baseline", lambda b: b.update(launcher_run=str(run_dir.parent / "r-f0")))
    baseline = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
    edit_part(out, "rest", lambda r: r["baseline_record"].update(
        sha256=q.record_sha256(baseline)))  # fmt: skip


def baseline_identity_elsewhere(path, root, out, run_dir):
    """The baseline names another identity than the located run's (and the rest's); the rest is
    bound to that baseline record again, so that only the identity differs."""
    edit_part(out, "baseline", lambda b: b["launcher"]["identity"].update(sha256="0" * 64))
    baseline = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
    edit_part(out, "rest", lambda r: r["baseline_record"].update(
        sha256=q.record_sha256(baseline)))  # fmt: skip


NO = 3  # ``control-only``'s no: the driver's usual path (scripts/qualify.py's ``NO``)
EXPORT = 9  # ``control-only``'s answer for a technical case: export its evidence (TECHNICAL_CASE)
BOUND_ELSEWHERE = {  # a part bound to another launcher run → the exit, and what it names
    # Not technical by the case's verdict (it compares no run directory): refused, exit 1
    "rest run id, not technical: 1": (
        rest_change(lambda r, d: r.update(launcher_run=str(d.parent / "r-f0"))), 1,
        "the rest record names the launcher run"),
    "baseline run id, not technical: 1": (
        baseline_elsewhere, 1, "the baseline record names the launcher run"),
    # Technical by the verdict (the rest ran against another launcher run): its export, 9
    "rest identity, technical: 9": (
        rest_change(lambda r, d: r["launcher"]["identity"].update(sha256="0" * 64)), EXPORT,
        "the rest part ran against another launcher run"),
    "baseline identity, technical: 9": (
        baseline_identity_elsewhere, EXPORT, "the rest part ran against another launcher run"),
}  # fmt: skip


@pytest.mark.parametrize("elsewhere", list(BOUND_ELSEWHERE))
def test_control_only_refuses_a_part_bound_to_another_launcher_run(
    qualify, stubbed, tmp_path, capsys, elsewhere
) -> None:
    """A case whose baseline and rest are recorded, its control due, but one of those records
    bound to another launcher run than the located one: ``control-only`` never runs the control
    alone. Where the case's verdict does not mark it technical (a run id differs), it exits 1
    naming the record; where it does (an identity differs: the rest ran against another launcher
    run), it exits 9 first, so that the driver exports the case's checkpoints before any launcher
    call (it exited 3, the usual path, before)."""
    path, root, out, run_dir = decided_world(tmp_path, rest=True)
    assert exit_code(qualify, control_only(path)) == 0
    change, expected, why = BOUND_ELSEWHERE[elsewhere]
    change(path, root, out, run_dir)
    baseline = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
    rest = json.loads((out / FA.attempt_path("rest", 1)).read_text())
    technical = q.case_verdict(baseline, rest, SCOPE)["kind"] == "technical"
    assert technical is (expected == EXPORT)
    capsys.readouterr()
    code = exit_code(qualify, control_only(path))
    if expected == EXPORT:
        assert code == EXPORT and why in capsys.readouterr().err
    else:
        assert isinstance(code, str) and why in code and "nothing is rerun" in code, code
    assert stubbed == [] and not (out / (FA.attempt_path("control", 1) + ".started")).exists()


def launcher_not_conforming(baseline):
    """The baseline's launcher check found the run not conforming: nothing ran (technical)."""
    reasons = ["the launcher run does not conform: its numerics setup departs"]
    baseline["launcher"].update(kind="technical", reasons=reasons)
    baseline.update(status="not_run", kind="technical", reasons=reasons)


def launcher_not_classified(baseline):
    """The baseline raised before its launcher check returned: failed, technical."""
    baseline["launcher"]["classified"] = False
    baseline.update(status="failed", kind="technical", reasons=["RuntimeError: the setup"])


def launcher_check_world(tmp_path, change):
    """FA's passing baseline changed by ``change`` (its launcher check failed or never
    returned), its rest recorded and bound to it (ran nothing: technical), the control due; the
    disk intact, r-f1's checkpoint on it with its recorded bytes."""
    path, root, out, run_dir = decided_world(tmp_path, rest=True)
    edit_part(out, "baseline", change)
    baseline = json.loads((out / FA.attempt_path("baseline", 1)).read_text())

    def ran_nothing(rest):
        reasons = ["the baseline part did not end with its checkpoint (not_run)"]
        rest.update(status="not_run", kind="technical", reasons=reasons)
        rest["baseline_record"]["sha256"] = q.record_sha256(baseline)

    edit_part(out, "rest", ran_nothing)
    (tmp_path / "ckpt" / "r-f1.pt").write_bytes(b"w")  # ``with_checkpoint``'s recorded bytes
    return path, root, out, run_dir


TECHNICAL_WORLDS = {  # a case its own verdict marks technical → what control-only names
    "baseline launcher kind technical": (
        lambda tmp_path: launcher_check_world(tmp_path, launcher_not_conforming),
        "the launcher run does not conform"),
    "baseline not classified": (
        lambda tmp_path: launcher_check_world(tmp_path, launcher_not_classified),
        "RuntimeError: the setup"),
    "rest bound to another identity": (
        lambda tmp_path: decided_world(tmp_path, rest=lambda r: r["launcher"]["identity"].update(
            sha256="0" * 64)),
        "the rest part ran against another launcher run"),
}  # fmt: skip


def test_restart_case_refuses_the_state_control_only_runs_alone(qualify, tmp_path) -> None:
    """A passing baseline and rest, the control due, the container disk lost: ``control-only``
    would run the control alone, so ``restart-case`` refuses (it would void both passing parts
    and rerun them), naming that route, and voids and supersedes nothing."""
    from zero_field_refusal import launch

    path, root, out, run_dir = decided_world(tmp_path, rest=True)
    assert exit_code(qualify, control_only(path)) == 0
    with pytest.raises(SystemExit) as stop:
        qualify.main(restart(path, FA, RESTART))
    assert "control-only" in stop.value.code and "Do not restart" in stop.value.code
    assert not list(case_folder(out, FA).glob("restart-*"))
    run = next(r for r in launch.load_campaign(path)[1] if r.label == FA_LABEL)
    assert not launch.superseded_runs(run, launch.launch_records(root))


def conforming_launcher(run_dir) -> None:
    """The FA launcher run's meta.json with the hash of v₀ the launcher check asks of an arm that
    reads it, so that ``launcher_problems`` passes it (its identity reads no meta.json field but
    the config hash: unchanged)."""
    meta = json.loads((run_dir / "meta.json").read_text())
    meta["inputs"]["v0_sha256"] = "v" * 64
    (run_dir / "meta.json").write_text(json.dumps(meta))


@pytest.mark.parametrize("elsewhere", list(BOUND_ELSEWHERE))
def test_restart_case_never_restarts_over_a_binding_refusal(qualify, tmp_path, elsewhere) -> None:
    """A case whose baseline and rest are recorded, its control due, one of those records bound
    to another launcher run (its run id or its identity) than the located one, and the
    checkpoint lost with the container disk: ``restart-case`` refuses it (exit 1, with the
    reason), whether ``control-only`` refuses its binding or the case is technical, and writes
    nothing: no restart record, no ``superseded`` line."""
    from zero_field_refusal import launch

    path, root, out, run_dir = decided_world(tmp_path, rest=True)
    conforming_launcher(run_dir)  # nothing else refuses: only the binding is left to judge
    change, expected, why = BOUND_ELSEWHERE[elsewhere]
    change(path, root, out, run_dir)
    assert not (tmp_path / "ckpt" / "r-f1.pt").exists()  # the container disk lost
    log = root / "launch" / "study3-qualification-c1.jsonl"
    before = log.read_bytes()
    with pytest.raises(SystemExit) as stop:
        qualify.main(restart(path, FA, RESTART))
    assert isinstance(stop.value.code, str), stop.value.code  # exit 1, its reason named
    assert "no restart" in stop.value.code
    if expected == 1:  # control-only's binding refusal, carried as restart-case's
        assert "control-only" in stop.value.code and why in stop.value.code
    assert not list(case_folder(out, FA).glob("restart-*")) and log.read_bytes() == before
    assert not launch.superseded_runs(fa_run(path), launch.launch_records(root))


def test_restart_case_restarts_a_case_whose_rest_is_due_after_a_loss(qualify, tmp_path) -> None:
    """A passing baseline, the rest due (no record), the control due and the checkpoint lost:
    ``control-only`` takes the usual path (exit 3), and ``restart-case`` restarts the case as
    before: its restart record, and the launcher run logged superseded."""
    from zero_field_refusal import launch

    path, root, out, run_dir = decided_world(tmp_path, rest=True)
    conforming_launcher(run_dir)
    (out / FA.attempt_path("rest", 1)).unlink()  # the rest due
    assert exit_code(qualify, control_only(path)) == NO
    qualify.main(restart(path, FA, RESTART))
    record = json.loads((case_folder(out, FA) / "restart-1.json").read_text())
    assert [(a["part"], a["attempt"]) for a in record["attempts"]] == [("baseline", 1)]
    assert record["checkpoint_missing"] is True
    assert set(launch.superseded_runs(fa_run(path), launch.launch_records(root))) == {run_dir.name}


def test_restart_case_keeps_its_refusal_of_a_decided_case(qualify, tmp_path) -> None:
    """A case its baseline decided keeps the refusal it had: the failure keeps counting."""
    path, root, out, run_dir = decided_world(tmp_path)
    with pytest.raises(SystemExit, match="a completed failure keeps counting") as stop:
        qualify.main(restart(path, FA, RESTART))
    assert "control-only" not in stop.value.code
    assert not list(case_folder(out, FA).glob("restart-*"))


@pytest.mark.parametrize("departs", [True, False])
def test_restart_case_reads_the_launcher_runs_numerics(qualify, tmp_path, departs) -> None:
    """A start without an end, settled ``conflict`` with the missing-checkpoint problem alone (the
    launcher checks the checkpoint before the numerics, so it hides them): ``restart-case`` reads
    the run's recorded numerics against its campaign's config and refuses a departing setup as a
    provenance conflict, superseding nothing; matching numerics are accepted as before."""
    from zero_field_refusal import launch

    path = campaign(tmp_path, fa=())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, f"adamw-{ADAMW_LR:g}/s10200", ["r-a1"])
    run_dir = settled_launcher(root, run, "r-a1")  # its checkpoint, /x.pt, is gone
    if departs:
        meta = json.loads((run_dir / "meta.json").read_text())
        meta["inputs"]["numerics_setup"]["tf32_matmul"] = True
        (run_dir / "meta.json").write_text(json.dumps(meta))
    log_end(root, path, run, "r-a1", outcome="conflict", returncode=None, settled=True,
            problem=f"the endpoint checkpoint {run_dir}/x.pt is missing")  # fmt: skip
    if departs:
        with pytest.raises(SystemExit, match="numerics setup departs") as stop:
            qualify.main(restart(path, ADAMW, RESTART))
        assert "provenance conflict" in stop.value.code
        assert not list(case_folder(out, ADAMW).glob("restart-*"))
        assert not launch.superseded_runs(run, launch.launch_records(root))
    else:
        qualify.main(restart(path, ADAMW, RESTART))
        assert list(case_folder(out, ADAMW).glob("restart-1.json"))


def test_an_interrupted_scientific_rest_is_recorded_and_the_control_then_runs_alone(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """The lost-checkpoint table's scientific branch: the rest was interrupted after a valid
    scientific failure in its trajectory results and the container disk was lost. ``control-only``
    says no (the rest is outstanding) and ``restart-case`` refuses (the failure keeps counting).
    ``interrupted --part rest --attempt 1 --technical`` records the attempt with its results,
    the verdict is scientific, and then ``control-only`` exits 0, the control runs alone and
    ``finish`` exits 0."""
    path = campaign(tmp_path)
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, FA_LABEL, ["r-f1"])
    log_end(root, path, run, "r-f1")  # the launcher run is done
    checkpoint = tmp_path / "ckpt" / "r-f1.pt"
    run_dir = with_checkpoint(settled_launcher(root, run, "r-f1"), checkpoint)
    baseline, rest = synthetic(FA, identity=q.launcher_identity(run_dir))
    baseline["launcher_run"] = rest["launcher_run"] = str(run_dir)
    baseline["launcher"]["classified"] = rest["launcher"]["classified"] = True
    publish(out, FA, "baseline", baseline)
    scientific_rest(rest)
    rest["baseline_record"] = {"path": str(out / FA.attempt_path("baseline", 1)),
                               "sha256": q.record_sha256(baseline)}  # fmt: skip
    q.publish(out / (FA.attempt_path("rest", 1) + ".started"),
              {"part": "rest", "attempt": 1, "case": FA.record(), "run_dir": str(run_dir),
               "pod_id": "pod-1", "started_at": "t"})  # fmt: skip
    for name in ("repeat", "reference"):
        q.publish(out / FA.trajectory_path(1, name), trajectory_result(FA, rest, name))
    checkpoint.unlink()  # the container disk lost
    assert exit_code(qualify, control_only(path)) == 3  # the rest is outstanding
    with pytest.raises(SystemExit, match="keeps counting"):
        qualify.main(restart(path, FA, RESTART))
    assert not list(case_folder(out, FA).glob("restart-*"))
    qualify.main(["interrupted", *common(path), "--case", FA.name, "--part", "rest",
                  "--attempt", "1", "--technical", "the Pod stopped during the rest"])  # fmt: skip
    record = json.loads((out / FA.attempt_path("rest", 1)).read_text())
    scope = q.QualificationScope.of(REGISTERED)
    assert q.case_verdict(baseline, record, scope)["kind"] == q.SCIENTIFIC
    assert run_the_control_alone(qualify, monkeypatch, capsys, path, run_dir, tmp_path)


@pytest.mark.parametrize("world", list(TECHNICAL_WORLDS))
def test_control_only_sends_every_technical_case_to_its_export(
    qualify, tmp_path, monkeypatch, capsys, world
) -> None:
    """A case its own verdict marks technical, its control due: the baseline's launcher check
    failed (the run not conforming, or the check never returned), or its rest is bound to
    another launcher run. ``control-only`` exits 9 naming the technical mark, first, and runs
    nothing alone (it exited 3, the usual path, before). ``finish --kept`` then exports its
    checkpoints and exits 9, the case not complete: its control never runs, and the case stays
    incomplete and technical."""
    assert NO == qualify.NO and EXPORT == qualify.TECHNICAL_CASE
    make, why = TECHNICAL_WORLDS[world]
    path, root, out, run_dir = make(tmp_path)
    (tmp_path / "ckpt" / "r-f1.pt").write_bytes(b"w")  # the disk intact
    baseline = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
    rest = json.loads((out / FA.attempt_path("rest", 1)).read_text())
    assert q.case_verdict(baseline, rest, SCOPE)["kind"] == "technical"
    calls = []
    monkeypatch.setattr(qualify.runtime, "configure_numerics", lambda cfg, device: {})
    monkeypatch.setattr(qualify.runtime, "environment", lambda cfg, device: object())
    monkeypatch.setattr(qualify.q, "run_part", accepted_control(calls, run_dir))
    capsys.readouterr()
    assert exit_code(qualify, control_only(path)) == EXPORT
    err = capsys.readouterr().err
    assert "technical case" in err and why in err, err
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    manifest = json.loads((case_folder(out, FA) / "kept-checkpoints-1.json").read_text())
    assert manifest["lost"] == [] and manifest["binding"]["parts"]["control"] is None
    assert calls == [] and not (out / (FA.attempt_path("control", 1) + ".started")).exists()
    planned = qualify.campaign_cases(path, SCOPE, 1, False)[1][0]
    assert qualify.case_state(out, planned, SCOPE)[0] is False
    assert qualify.established_technical(out, planned, SCOPE)


def test_control_only_exports_a_technical_case_and_leaves_an_unfinished_one_to_the_usual_path(
    qualify, tmp_path, capsys
) -> None:
    """A case whose rest holds a technical mark is not run by ``control-only``: exit 9, its
    export (it exited 3, the usual path, before); a case whose rest is due or interrupted, its
    baseline passing, is no technical case (an outstanding part is no mark) and takes the usual
    path (exit 3), as before."""
    path, root, out, run_dir = decided_world(tmp_path, rest=technical_rest)
    baseline = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
    rest = json.loads((out / FA.attempt_path("rest", 1)).read_text())
    assert q.case_verdict(baseline, rest, SCOPE)["kind"] == "technical"
    capsys.readouterr()
    assert exit_code(qualify, control_only(path)) == qualify.TECHNICAL_CASE
    assert "technical" in capsys.readouterr().err
    (out / FA.attempt_path("rest", 1)).unlink()  # the rest due
    assert exit_code(qualify, control_only(path)) == qualify.NO
    q.publish(out / (FA.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    assert exit_code(qualify, control_only(path)) == qualify.NO  # the rest interrupted


def technical_comparison(path, root, out, run_dir):
    """The baseline, complete, with a technical comparison of its own (``launcher~baseline`` D 0.07
    against the bound 0.06), the rest, when recorded, bound to it again."""
    edit_part(out, "baseline", lambda b: b["comparisons"]["launcher~baseline"].update(D_twin=0.07))
    if (out / FA.attempt_path("rest", 1)).exists():
        baseline = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
        edit_part(out, "rest", lambda r: r["baseline_record"].update(
            sha256=q.record_sha256(baseline)))  # fmt: skip


def test_a_technical_comparison_a_scientific_rest_overrides_runs_the_control_alone(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """A complete baseline with a technical comparison of its own (``launcher~baseline`` beyond
    the bound) still has its rest read, and a bound rest holding a scientific failure turns the
    case's verdict scientific: no established technical case. With the control due,
    ``control-only`` exits 0 and the control runs alone, and the verdict stays scientific."""
    path, root, out, run_dir = decided_world(tmp_path, rest=scientific_rest)
    technical_comparison(path, root, out, run_dir)
    baseline = json.loads((out / FA.attempt_path("baseline", 1)).read_text())
    rest = json.loads((out / FA.attempt_path("rest", 1)).read_text())
    alone = q.case_verdict(baseline, None, SCOPE)
    assert alone["kind"] == "technical" and any(r.startswith("launcher~baseline")
                                                for r in alone["reasons"])  # fmt: skip
    assert q.case_verdict(baseline, rest, SCOPE)["kind"] == "scientific"
    assert exit_code(qualify, control_only(path)) == 0
    planned = qualify.campaign_cases(path, SCOPE, 1, False)[1][0]
    assert qualify.established_technical(out, planned, SCOPE) == []
    run_the_control_alone(qualify, monkeypatch, capsys, path, run_dir, tmp_path)  # 0, then finish
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    (case,) = [c for c in verdict["cases"] if c["case"]["name"] == FA.name]
    assert (case["state"], case["kind"]) == ("failing", "scientific")


def test_a_technical_comparison_with_its_rest_due_takes_the_usual_path(
    qualify, tmp_path, capsys
) -> None:
    """The same baseline (a technical comparison of its own), its rest due: the rest may still
    turn the verdict scientific, so no established technical case; ``control-only`` exits 3, the
    usual path, which runs the rest."""
    path, root, out, run_dir = decided_world(tmp_path, rest=True)
    (out / FA.attempt_path("rest", 1)).unlink()
    technical_comparison(path, root, out, run_dir)
    capsys.readouterr()
    assert exit_code(qualify, control_only(path)) == qualify.NO
    assert "outstanding" in capsys.readouterr().err
    planned = qualify.campaign_cases(path, SCOPE, 1, False)[1][0]
    assert qualify.established_technical(out, planned, SCOPE) == []


def test_control_only_asks_every_case_first_whether_it_is_technical(qualify, tmp_path) -> None:
    """``control-only`` asks first whether the case's recorded parts already mark it technical
    (its baseline's verdict alone, the rest bound to it, its control), complete or not: AdamW's
    case, which holds no control, exits 9 then (its baseline's launcher check failed, its rest
    due), and 3, the usual path, when its baseline passes and its rest is due."""
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    baseline, _ = synthetic(ADAMW)
    publish(out, ADAMW, "baseline", baseline)
    assert exit_code(qualify, control_only(path, ADAMW)) == qualify.NO
    launcher_not_conforming(baseline)
    publish(out, ADAMW, "baseline", baseline)
    assert exit_code(qualify, control_only(path, ADAMW)) == qualify.TECHNICAL_CASE
    planned = qualify.campaign_cases(path, SCOPE, 1, False)[1][1]
    assert planned.case == ADAMW and qualify.case_state(out, planned, SCOPE)[0] is False


def test_finish_exports_an_established_technical_case_that_is_not_complete(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """A case its recorded parts mark technical, not complete (its control due): ``finish --kept``
    exports it as a complete technical case's (exit 9; the manifest, the copies), never marks it
    complete, and ``finish --release`` follows the same receipt binding (13 before ``kept-check``,
    then the deletion, its record naming the manifest). A case neither complete nor technical is
    still refused (exit 1), its checkpoints kept."""
    path, root, out, run_dir = launcher_check_world(tmp_path, launcher_not_conforming)
    planned = qualify.campaign_cases(path, SCOPE, 1, False)[1][0]
    assert qualify.case_state(out, planned, SCOPE)[0] is False
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    capsys.readouterr()
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    assert "not complete" in capsys.readouterr().err
    folder = case_folder(out, FA)
    manifest = json.loads((folder / "kept-checkpoints-1.json").read_text())
    (entry,) = manifest["files"]
    assert entry["source"] == str(tmp_path / "ckpt" / "r-f1.pt") and entry["verified"] is True
    assert qualify.case_state(out, planned, SCOPE)[0] is False
    assert qualify.established_technical(out, planned, SCOPE)
    release = finish_args(path, FA, None, "--release")
    assert exit_code(qualify, release) == qualify.NOT_EXPORTED  # no kept-check yet
    inventory = archived(kept, tmp_path / "inventory.tsv", monkeypatch)
    assert exit_code(qualify, kept_check_args(path, FA, inventory)) == 0
    qualify.main(release)
    assert not (tmp_path / "ckpt" / "r-f1.pt").exists()
    (record,) = [json.loads(f.read_text()) for f in folder.glob("checkpoints-removed-*")]
    assert record["manifests"][0]["path"] == str(folder / "kept-checkpoints-1.json")
    assert qualify.case_state(out, planned, SCOPE)[0] is False
    (tmp_path / "plain").mkdir()  # neither complete nor technical: refused, as before
    path, root, out, run_dir = decided_world(tmp_path / "plain", rest=True)
    (tmp_path / "plain" / "ckpt" / "r-f1.pt").write_bytes(b"w")
    with pytest.raises(SystemExit, match="control: due .*; its checkpoints are kept"):
        qualify.main([*finish_args(path, FA, kept), "--runs-root", str(root)])
    assert (tmp_path / "plain" / "ckpt" / "r-f1.pt").exists()


# ------------------------------------- an established technical case, exported before the launcher

# project-run SRC python <script> ...: the real qualify.py (through QUALIFY_RUN, which patches
# only ``same_filesystem``: the tmp paths share one device), and stand-ins for the launcher (its
# conflict, exit 1) and scripts/archive.py (the real archive over a stand-in bucket; STUB_archive
# a failure). As project-run does, the source's own src is the only PYTHONPATH, so the scripts
# import that checkout's package whatever the interpreter has installed.
REAL_PROJECT_RUN = r"""#!/usr/bin/env bash
src="$1"
shift 2
export PYTHONPATH="$src/src"
case "$1" in
  -c) exit 0 ;;
  scripts/launch.py)
    echo "launch ${*:2}" >> "$STUB_LOG"
    echo "1 run completed but cannot stand: the endpoint checkpoint does not match its" \
      "recorded sha256; the campaign stops" >&2
    exit 1 ;;
  scripts/archive.py)
    echo "archive ${*:2}" >> "$STUB_LOG"
    if [ "${STUB_archive:-0}" != 0 ]; then exit "$STUB_archive"; fi
    exec "$PYTHON" "$ARCHIVE_STAND_IN" "$src/scripts/archive.py" "${@:2}" ;;
  scripts/qualify.py)
    echo "${*:2}" >> "$STUB_LOG"
    cd "$src" && exec "$PYTHON" "$QUALIFY_RUN" "$src/scripts/qualify.py" "${@:2}" ;;
esac
echo "project-run: unexpected $*" >&2
exit 99
"""
QUALIFY_RUN = """import importlib.util
import sys

spec = importlib.util.spec_from_file_location("qualify_script", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.same_filesystem = lambda a, b: False  # the tmp paths share one device
module.main(sys.argv[2:])
"""
ARCHIVE_STAND_IN = """import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("archive_script", sys.argv[1])
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)
inventory, prefix, root, *paths = sys.argv[2:]
bucket, _, base = prefix.removeprefix("gs://").partition("/")


def gcloud(args):  # the bucket: cp copies nothing; a listing gives each local file's size and MD5
    if args[0] == "cp":
        return ""
    rel = args[2].removesuffix("/**").removeprefix(prefix + "/")
    found = []
    for file in sorted(f for f in (Path(root) / rel).rglob("*") if f.is_file()):
        size, _, md5 = archive.digests(file)
        name = f"{base}/{file.relative_to(root).as_posix()}"
        found.append({"bucket": bucket, "name": name, "size": str(size), "md5_hash": md5})
    return json.dumps(found)


archive.gcloud = gcloud
print(json.dumps(archive.archive(Path(inventory), prefix, Path(root), paths)))
"""


def real_gpu_code() -> dict:
    """This checkout's GPU code identity as a separate process computes it (the fixture replaces
    it in this one): what a real ``qualify.py`` holds a record's provenance to."""
    done = subprocess.run([sys.executable, "-c", "import json; from zero_field_refusal import "
                           "code_identity; print(json.dumps(code_identity.gpu_code_identity()))"],
                          capture_output=True, text=True, check=True, cwd=ROOT)  # fmt: skip
    return json.loads(done.stdout)


def established_technical_world(tmp_path, case, code, *, comparison=False):
    """A case its recorded parts mark technical, not complete: its baseline's launcher check found
    the run not conforming (nothing ran: the verdict reads no rest for it); for FA its rest ran
    nothing and its control is due, for AdamW its rest is due. With ``comparison`` (FA), the
    baseline passes instead and its rest, recorded and bound to it, holds a technical comparison
    (``baseline~repeat`` beyond the bound): the whole verdict technical, its control due. Its
    launcher run r-1 is done, and r-1's endpoint checkpoint is on disk with corrupted bytes,
    which the launcher would refuse. The records carry ``code``, the GPU code a real
    ``qualify.py`` reads. Returns the campaign, the records' folder and the checkpoint."""
    fa = case == FA
    path = campaign(tmp_path, fa=(10200,) if fa else ())
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, FA_LABEL if fa else f"adamw-{ADAMW_LR:g}/s10200", ["r-1"])
    log_end(root, path, run, "r-1")
    checkpoint = tmp_path / "ckpt" / "r-1.pt"
    run_dir = with_checkpoint(settled_launcher(root, run, "r-1"), checkpoint)
    baseline, rest = synthetic(case, identity=q.launcher_identity(run_dir))
    for record in (baseline, rest):
        record.update(launcher_run=str(run_dir))
        record["launcher"]["classified"] = True
        record["provenance"]["gpu_code"] = code
    if comparison:
        technical_rest(rest)
    else:
        launcher_not_conforming(baseline)
        ran_nothing = ["the baseline part did not end with its checkpoint (not_run)"]
        rest.update(status="not_run", kind="technical", reasons=ran_nothing)
    publish(out, case, "baseline", baseline)
    if fa:
        rest["baseline_record"] = {"path": str(out / case.attempt_path("baseline", 1)),
                                   "sha256": q.record_sha256(baseline)}  # fmt: skip
        publish(out, case, "rest", rest)
        whole = q.case_verdict(baseline, rest, SCOPE)
        assert whole["kind"] == "technical" and (comparison or rest["status"] == "not_run")
    checkpoint.write_bytes(b"corrupted bytes")
    return path, out, checkpoint


def real_driver(qualify, tmp_path, monkeypatch, path):
    """The generated driver over the real ``qualify.py`` subcommands (``REAL_PROJECT_RUN``): the
    launcher a stand-in that refuses with the launcher's conflict (exit 1), scripts/archive.py a
    stand-in over the real archive (``ARCHIVE_STAND_IN``), ``git rev-parse HEAD`` the driver's
    commit (any other git call the real one), ``stat`` the devices (``STAT``), ``nvidia-smi``
    idle and ``gcloud`` present. Returns what runs it: the exits ``STUB_<name>`` give it, then
    its exit and the calls it logged."""
    commit = "a" * 40
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = shutil.which("git")
    for name, text in (
        ("project-run", REAL_PROJECT_RUN),
        ("git", f'#!/bin/sh\nif [ "$*" = "rev-parse HEAD" ]; then echo {commit}; exit 0; fi\n'
                f'exec {git} "$@"\n'),
        ("stat", STAT),
        ("nvidia-smi", '#!/bin/sh\ncase "$*" in *compute-apps*) ;; *) echo 0 ;; esac\n'),
        ("gcloud", "#!/bin/sh\nexit 1\n"),
    ):  # fmt: skip
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    (tmp_path / "qualify_run.py").write_text(QUALIFY_RUN)
    (tmp_path / "archive_stand_in.py").write_text(ARCHIVE_STAND_IN)
    (tmp_path / "code" / commit).mkdir(parents=True)
    (tmp_path / "code" / commit / "source").symlink_to(ROOT)  # the source checkout
    key = tmp_path / "artifact-gcs-key.json"
    key.write_text("{}")
    monkeypatch.setattr(qualify, "POD_ENVIRONMENT", ())
    monkeypatch.setattr(qualify, "SOURCES", str(tmp_path / "code"))
    monkeypatch.setattr(qualify, "need_kb", lambda gb: 1)
    driver = tmp_path / "driver.sh"
    qualify.main([*driver_args(tmp_path, path), "--commit", commit, "--out", str(driver)])
    log = tmp_path / "stub.log"
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "RUNS_DIR": str(tmp_path / "runs"),
           "DATA_DIR": str(tmp_path / "data"), "HOME": os.environ.get("HOME", str(tmp_path)),
           "TMPDIR": os.environ.get("TMPDIR", str(tmp_path)), "PYTHON": sys.executable,
           "QUALIFY_RUN": str(tmp_path / "qualify_run.py"),
           "ARCHIVE_STAND_IN": str(tmp_path / "archive_stand_in.py"), "STUB_LOG": str(log),
           "ARTIFACT_REMOTE": "gs://bucket/zfr", "ARTIFACT_GCS_KEY": str(key)}  # fmt: skip

    def run(**exits):
        log.write_text("")
        stubs = {f"STUB_{key}": str(value) for key, value in exits.items()}
        done = subprocess.run(["bash", str(driver)], capture_output=True, text=True,
                              env={**env, **stubs})  # fmt: skip
        (tmp_path / "driver.out").write_text(done.stdout + done.stderr)
        return done.returncode, [line.split(" ", 1)[0] for line in log.read_text().splitlines()]

    return run


@pytest.mark.parametrize(("case", "comparison"), [(FA, False), (ADAMW, False), (FA, True)],
                         ids=["FA, its control due", "AdamW, its rest due",
                              "FA, a technical comparison of its rest"])  # fmt: skip
def test_the_driver_exports_a_technical_case_before_a_launcher_that_would_refuse(
    qualify, tmp_path, monkeypatch, case, comparison
) -> None:
    """The generated driver over the real ``qualify.py`` subcommands, a case its recorded parts
    mark technical and not complete, its launcher run's checkpoint corrupted (the launcher's
    stand-in refuses it, exit 1): ``complete`` says 3, ``control-only`` 9, and the export runs
    before any launcher call. The launcher is never called; the export holds the corrupt bytes,
    ``matches_record`` false; ``kept-check`` passes; the driver stops with 9; the case stays
    incomplete and technical. Run again, the driver reuses the verified export (9 again). The
    mark is a baseline whose verdict reads no rest (its launcher check failed), or a technical
    comparison of a recorded rest bound to the baseline (``comparison``)."""
    path, out, checkpoint = established_technical_world(tmp_path, case, real_gpu_code(),
                                                        comparison=comparison)  # fmt: skip
    run = real_driver(qualify, tmp_path, monkeypatch, path)
    code, calls = run()
    assert code == qualify.TECHNICAL_CASE, (tmp_path / "driver.out").read_text()
    assert calls == ["complete", "control-only", "finish", "archive", "kept-check", "verdict"]
    folder = case_folder(out, case)
    manifest = json.loads((folder / "kept-checkpoints-1.json").read_text())
    (entry,) = [e for e in manifest["files"] if e["source"] == str(checkpoint)]
    assert (entry["verified"], entry["matches_record"]) == (True, False)
    assert (tmp_path / "volume" / "kept" / entry["copy"]).read_bytes() == b"corrupted bytes"
    assert json.loads((folder / "kept-check-1.json").read_text())["exit"] == 0
    planned = next(p for p in qualify.campaign_cases(path, SCOPE, 1, False)[1] if p.case == case)
    assert qualify.case_state(out, planned, SCOPE)[0] is False
    assert qualify.established_technical(out, planned, SCOPE)
    assert checkpoint.read_bytes() == b"corrupted bytes"  # kept where it is
    code, calls = run()  # a second pass
    assert code == qualify.TECHNICAL_CASE and "launch" not in calls
    assert [f.name for f in folder.glob("kept-checkpoints-*")] == ["kept-checkpoints-1.json"]
    assert json.loads((folder / "kept-check-2.json").read_text())["exit"] == 0


def test_the_driver_leaves_a_pending_upload_receipt_for_an_early_export(
    qualify, tmp_path, monkeypatch
) -> None:
    """The same world for FA, the upload failing (scripts/archive.py exits 1): ``kept-check``
    still runs and publishes its receipt, ``pending-upload`` true (the volume copy verified), and
    the driver stops with 13; the launcher is never called."""
    path, out, checkpoint = established_technical_world(tmp_path, FA, real_gpu_code())
    run = real_driver(qualify, tmp_path, monkeypatch, path)
    code, calls = run(archive=1)
    assert code == qualify.NOT_EXPORTED, (tmp_path / "driver.out").read_text()
    assert calls == ["complete", "control-only", "finish", "archive", "kept-check", "verdict"]
    receipt = json.loads((case_folder(out, FA) / "kept-check-1.json").read_text())
    assert (receipt["exit"], receipt["pending-upload"]) == (qualify.NOT_EXPORTED, True)


def test_finish_keeps_a_technical_cases_checkpoints(qualify, tmp_path, monkeypatch) -> None:
    """A case whose verdict is technical (or whose same-path control is) keeps its checkpoints
    for the diagnosis: ``finish --kept DIR`` exports them and exits 9; ``--release`` deletes them
    once diagnosed, after ``kept-check`` has verified that export."""
    path = campaign(tmp_path)
    out = tmp_path / "runs" / "qualification"
    journal(tmp_path / "runs", path, f"adam_frozen-{FA_LR:g}/s10200", ["r-a1"])
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "r-a1.pt").write_bytes(b"w")
    write_case(out, FA)
    publish(out, FA, "control", control_record(0.1, case=FA))  # beyond 0.01·K*: technical
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as stop:
        qualify.main(finish_args(path, FA, kept))
    assert stop.value.code == qualify.TECHNICAL_CASE
    assert (tmp_path / "ckpt" / "r-a1.pt").exists()
    inventory = archived(kept, tmp_path / "inventory.tsv", monkeypatch)
    assert exit_code(qualify, kept_check_args(path, FA, inventory)) == 0
    qualify.main(finish_args(path, FA, None, "--release"))
    assert not (tmp_path / "ckpt" / "r-a1.pt").exists()


# ------------------------------------------------- a technical case's kept checkpoints

WEIGHTS, BASE, STATE = b"launcher weights", b"harness baseline", b"state at step 5"
REMOTE = "gs://bucket/zfr/kept-checkpoints"  # $ARTIFACT_REMOTE/<--kept-remote>
LOST_MESSAGE = ("lost before the export: the technical mark stands; the diagnosis goes on from "
                "the records; nothing reruns")  # fmt: skip


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def kept_folder(qualify, monkeypatch, tmp_path, *, apart=True) -> Path:
    """The Pod's volume folder of the kept checkpoints; ``apart``: on another filesystem than the
    checkpoint directory (``same_filesystem`` patched: the tmp paths share one device)."""
    if apart:
        monkeypatch.setattr(qualify, "same_filesystem", lambda a, b: False)
    folder = tmp_path / "volume" / "kept-checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def finish_args(path, case, kept=None, *extra):
    return ["finish", *common(path), "--case", case.name,
            *([] if kept is None else ["--kept", str(kept)]), *extra]  # fmt: skip


def kept_check_args(path, case, inventory, remote=None):
    """``kept-check``'s arguments: the inventory, and the bucket prefix it is bound to (by default
    the stand-in bucket's, ``REMOTE``)."""
    return ["kept-check", *common(path), "--case", case.name, "--inventory", str(inventory),
            "--remote", remote or REMOTE]  # fmt: skip


def technical_world(tmp_path):
    """FA's case complete and technical by its same-path control (K beyond 0.01·K*), its records
    bound to its launcher run r-f1 (done); on disk, r-f1's endpoint checkpoint and the harness's
    baseline checkpoint with their recorded sha256, and a diagnostic state no record names.
    Returns the campaign, the records' folder, r-f1's directory and the files by name."""
    path = campaign(tmp_path)
    root, out = tmp_path / "runs", tmp_path / "runs" / "qualification"
    run = journal(root, path, FA_LABEL, ["r-f1"])
    endpoint = tmp_path / "ckpt"
    run_dir = with_checkpoint(settled_launcher(root, run, "r-f1"), endpoint / "r-f1.pt", WEIGHTS)
    harness = endpoint / "harness" / "s10200-adam_frozen-1.83e-05"
    files = {"launcher": endpoint / "r-f1.pt", "baseline": harness / "harness-baseline.pt",
             "state": harness / "baseline-states" / "step-5.pt"}  # fmt: skip
    files["state"].parent.mkdir(parents=True)
    files["baseline"].write_bytes(BASE)
    files["state"].write_bytes(STATE)
    identity = q.launcher_identity(run_dir)
    baseline, rest = synthetic(FA, identity=identity)
    baseline["checkpoint"] = {"path": str(files["baseline"]), "sha256": sha(BASE), "step": 25,
                              "names_sha256": "n"}  # fmt: skip
    publish(out, FA, "baseline", baseline)
    rest["baseline_record"]["sha256"] = q.record_sha256(baseline)
    rest["baseline_checkpoint_sha256"] = sha(BASE)
    publish(out, FA, "rest", rest)
    publish(out, FA, "control", control_record(0.1, case=FA, identity=identity))  # technical
    return path, out, run_dir, files


def archive_script():
    spec = importlib.util.spec_from_file_location("archive_script", ROOT / "scripts" / "archive.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archived(kept, inventory, monkeypatch, case=FA, *, change=None) -> Path:
    """scripts/archive.py run as the driver runs it, on the case's folder of the export under
    ``kept``, against a stand-in bucket: ``gcloud storage cp`` copies nothing, and the listing
    gives each local file's size and MD5 (``change(name, size, md5)`` alters an object's). Its
    inventory."""
    archive = archive_script()

    def gcloud(args):
        if args[0] == "cp":
            return ""
        rel = args[2].removesuffix("/**").removeprefix(REMOTE + "/")
        found = []
        for file in sorted(f for f in (kept / rel).rglob("*") if f.is_file()):
            size, _, md5 = archive.digests(file)
            name = f"zfr/kept-checkpoints/{file.relative_to(kept).as_posix()}"
            if change is not None:
                size, md5 = change(name, size, md5)
            found.append({"bucket": "bucket", "name": name, "size": str(size), "md5_hash": md5})
        return json.dumps(found)

    monkeypatch.setattr(archive, "gcloud", gcloud)
    archive.archive(inventory, REMOTE, kept, [f"qualification/{case_folder(Path(), case)}"])
    return inventory


@contextlib.contextmanager
def watched():
    """Every path this process opens or lists while inside (the audit events ``open``,
    ``os.listdir`` and ``os.scandir``)."""
    seen, active = [], [True]

    def hook(event, args):
        if (active[0] and event in ("open", "os.listdir", "os.scandir") and args
                and isinstance(args[0], (str, bytes, os.PathLike))):  # fmt: skip
            seen.append(os.fsdecode(args[0]))

    sys.addaudithook(hook)
    try:
        yield seen
    finally:
        active[0] = False


def test_finish_exports_a_technical_cases_checkpoints_and_exits_9(
    qualify, tmp_path, monkeypatch
) -> None:
    """A technical case (its same-path control beyond 0.01·K*): ``finish --kept DIR`` copies its
    launcher run's checkpoint and every file of the harness's folder to
    ``DIR/qualification/c<N>/<stage>/<case>/kept-1/`` in the same layout, each copy read back and
    verified by its sha256, and exits 9 with the originals kept. Its manifest, in the case's
    folder and in the export (``MANIFEST.json``, with ``SHA256SUMS``), lists every file with its
    sha256 and whether it matches the sha256 a binding record gives its path, and binds the
    records by path and sha256; nothing under the launcher run's responses/ is opened or listed."""
    path, out, run_dir, files = technical_world(tmp_path)
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    for sealed in ("responses", "observations", "judge"):  # each holding a file
        (run_dir / sealed).mkdir(exist_ok=True)
        (run_dir / sealed / "sealed.jsonl").write_text("{}\n")
    with watched() as seen:
        assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    assert not [p for p in seen if {"responses", "observations", "judge"} & set(Path(p).parts)]
    assert all(any((run_dir / sealed).iterdir()) for sealed in ("responses", "observations",
                                                                "judge"))  # fmt: skip
    assert str(run_dir / "summary.json") in seen and str(run_dir / "meta.json") in seen
    folder = case_folder(out, FA)
    (manifest_path,) = folder.glob("kept-checkpoints-*.json")
    assert manifest_path.name == "kept-checkpoints-1.json"
    manifest = json.loads(manifest_path.read_text())
    export = kept / "qualification" / "c1" / "numerical" / FA.name / "kept-1"
    assert (manifest["kept"], manifest["export"]) == (str(kept), str(export.relative_to(kept)))
    assert (export / "MANIFEST.json").read_bytes() == manifest_path.read_bytes()
    assert manifest["verified"] is True and manifest["case"] == FA.record()
    assert (manifest["stage"], manifest["candidate"]) == ("numerical", 1)
    assert manifest["technical_reasons"]
    assert all(r.startswith("control: ") for r in manifest["technical_reasons"])
    entries = {entry["source"]: entry for entry in manifest["files"]}
    assert set(entries) == {str(file) for file in files.values()}
    for file in files.values():
        entry = entries[str(file)]
        copy = export / file.relative_to(tmp_path / "ckpt")
        assert entry["copy"] == str(copy.relative_to(kept))
        assert (kept / entry["copy"]).read_bytes() == file.read_bytes()  # the original kept
        assert (entry["size"], entry["partial"], entry["verified"]) == (len(file.read_bytes()),
                                                                       False, True)  # fmt: skip
        assert entry["sha256"] == entry["copy_sha256"] == sha(file.read_bytes())
    launcher, harness, state = (entries[str(files[name])] for name in files)
    assert (launcher["recorded_sha256"], launcher["matches_record"]) == (sha(WEIGHTS), True)
    assert (harness["recorded_sha256"], harness["matches_record"]) == (sha(BASE), True)
    assert (state["recorded_sha256"], state["matches_record"]) == (None, None)  # none names it
    records = {part: out / FA.attempt_path(part, 1) for part in ("baseline", "rest", "control")}
    assert {r["record"] for r in launcher["named_by"]} == {str(run_dir / "summary.json"),
                                                          *map(str, records.values())}  # fmt: skip
    assert [r["record"] for r in harness["named_by"]] == [str(records["baseline"])]
    assert manifest["lost"] == []
    sums = (export / "SHA256SUMS").read_text().splitlines()
    assert sorted(sums) == sorted([
        *(f"{sha(f.read_bytes())}  {f.relative_to(tmp_path / 'ckpt')}" for f in files.values()),
        f"{sha(manifest_path.read_bytes())}  MANIFEST.json"])  # fmt: skip
    binding = manifest["binding"]
    for part, record in records.items():
        assert binding["parts"][part] == {
            "attempt": 1, "path": str(record), "sha256": sha(record.read_bytes()),
            "record_sha256": q.record_sha256(json.loads(record.read_text()))}  # fmt: skip
    for name in ("summary", "meta"):
        file = run_dir / f"{name}.json"
        assert binding["launcher"][name] == {"path": str(file), "sha256": sha(file.read_bytes())}
    assert binding["launcher"]["endpoint_checkpoint"]["sha256"] == sha(WEIGHTS)
    assert manifest["provenance"]["gpu_code"] == CODE
    assert {"host", "pod_id", "written_at"} <= set(manifest)
    assert all(file.exists() for file in files.values())


def test_a_present_checkpoint_of_another_hash_is_exported_as_it_is(
    qualify, tmp_path, monkeypatch
) -> None:
    """The launcher run's checkpoint on disk with another sha256 than its recorded one (a
    provenance conflict, ``checkpoint_conflict``): its bytes are exported and verified as a copy
    of the file on disk, with ``matches_record`` false; nothing of the case changes, its
    technical mark included."""
    path, out, run_dir, files = technical_world(tmp_path)
    files["launcher"].write_bytes(b"other weights")
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    planned = qualify.campaign_cases(path, SCOPE, 1, False)[1][0]
    reasons = qualify.technical_reasons(out, planned, SCOPE)
    records = tree(case_folder(out, FA))
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    manifest = json.loads((case_folder(out, FA) / "kept-checkpoints-1.json").read_text())
    (entry,) = [e for e in manifest["files"] if e["source"] == str(files["launcher"])]
    assert entry["verified"] is True and entry["matches_record"] is False
    assert entry["sha256"] == sha(b"other weights") and entry["recorded_sha256"] == sha(WEIGHTS)
    assert (kept / entry["copy"]).read_bytes() == b"other weights"
    assert qualify.checkpoint_conflict(run_dir) == files["launcher"]
    assert qualify.technical_reasons(out, planned, SCOPE) == reasons
    after = tree(case_folder(out, FA))
    assert set(after) - set(records) == {"kept-checkpoints-1.json"}
    assert all(after[name] == data for name, data in records.items())


REFUSALS = {  # where --kept points → how finish names the refusal
    "no --kept": "no --kept",
    "relative": "not an absolute path",
    "same filesystem": "filesystem",
    "under the runs root": "under the runs root",
    "under the checkpoint directory": "under the checkpoint directory",
    "no room": "free",
}


@pytest.mark.parametrize("refusal", list(REFUSALS))
def test_finish_refuses_an_export_it_cannot_keep(
    qualify, tmp_path, monkeypatch, capsys, refusal
) -> None:
    """Before any copy, ``finish`` refuses the export (exit 13; no manifest, nothing copied,
    nothing called verified, the checkpoints kept) without ``--kept``, for a relative folder, one
    on the checkpoint directory's filesystem (the container disk), one under the runs root (it
    would join the small results archived from there) or under the checkpoint directory, and one
    with less free space than the files take."""
    path, out, run_dir, files = technical_world(tmp_path)
    kept = kept_folder(qualify, monkeypatch, tmp_path, apart=refusal != "same filesystem")
    given = {
        "no --kept": None,
        "relative": Path("volume/kept-checkpoints"),
        "under the runs root": tmp_path / "runs" / "kept",
        "under the checkpoint directory": tmp_path / "ckpt" / "kept",
    }.get(refusal, kept)
    if given is not None and given.is_absolute():
        given.mkdir(parents=True, exist_ok=True)
    if refusal == "no room":
        need = sum(len(file.read_bytes()) for file in files.values()) + (1 << 30)  # 1 GiB margin
        monkeypatch.setattr(qualify, "free_bytes", lambda folder: need - 1)
    before = tree(tmp_path / "ckpt")
    capsys.readouterr()
    assert exit_code(qualify, finish_args(path, FA, given)) == qualify.NOT_EXPORTED
    err = capsys.readouterr().err
    assert REFUSALS[refusal] in err and "nothing is copied" in err
    assert not list(case_folder(out, FA).glob("kept-*"))
    for folder in (kept, tmp_path / "runs" / "kept", tmp_path / "ckpt" / "kept"):
        assert not folder.exists() or not any(folder.iterdir())
    assert tree(tmp_path / "ckpt") == before


def test_a_copy_that_differs_is_never_verified(qualify, tmp_path, monkeypatch) -> None:
    """A copy that does not read back as its source's bytes (a byte flipped after the copy): the
    export is not verified, ``finish`` exits 13, its manifest says so, and ``kept-check`` refuses
    it. Run again, ``finish`` writes ``kept-2``, verified; ``kept-1`` stays as it was: nothing is
    overwritten."""
    path, out, run_dir, files = technical_world(tmp_path)
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    copy = qualify.copy_durably

    def corrupting(source, target):
        result = copy(source, target)
        data = bytearray(target.read_bytes())
        data[0] ^= 0xFF
        target.write_bytes(bytes(data))
        return result

    monkeypatch.setattr(qualify, "copy_durably", corrupting)
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.NOT_EXPORTED
    folder = case_folder(out, FA)
    first = json.loads((folder / "kept-checkpoints-1.json").read_text())
    assert first["verified"] is False and not any(e["verified"] for e in first["files"])
    assert all(e["copy_sha256"] != e["sha256"] for e in first["files"])
    exports = kept / "qualification" / "c1" / "numerical" / FA.name
    before = tree(exports / "kept-1")
    inventory = archived(kept, tmp_path / "inventory.tsv", monkeypatch)
    assert exit_code(qualify, kept_check_args(path, FA, inventory)) == qualify.NOT_EXPORTED
    monkeypatch.setattr(qualify, "copy_durably", copy)
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    second = json.loads((folder / "kept-checkpoints-2.json").read_text())
    assert second["verified"] is True and second["export"].endswith("/kept-2")
    assert tree(exports / "kept-1") == before
    assert json.loads((folder / "kept-checkpoints-1.json").read_text()) == first
    assert all(file.exists() for file in files.values())


def test_a_copy_that_fails_leaves_nothing_called_verified(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """A copy that raises partway (the volume full, say): ``finish`` exits 13 and publishes no
    manifest, so nothing is called verified; the checkpoints stay, and the next run exports to a
    new folder past the one left partial."""
    path, out, run_dir, files = technical_world(tmp_path)
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    copy = qualify.copy_durably

    def failing(source, target):
        if source == files["baseline"]:
            raise OSError(28, "No space left on device")
        return copy(source, target)

    monkeypatch.setattr(qualify, "copy_durably", failing)
    capsys.readouterr()
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.NOT_EXPORTED
    assert "No space left on device" in capsys.readouterr().err
    folder = case_folder(out, FA)
    assert not list(folder.glob("kept-checkpoints-*"))
    assert all(file.exists() for file in files.values())
    monkeypatch.setattr(qualify, "copy_durably", copy)
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    (manifest,) = [json.loads(f.read_text()) for f in folder.glob("kept-checkpoints-*")]
    assert manifest["export"].endswith("/kept-2") and manifest["verified"] is True


def no_space(*args, **kwargs):
    raise OSError(28, "No space left on device")


def test_an_error_before_the_export_is_13(qualify, tmp_path, monkeypatch, capsys) -> None:
    """An ``OSError`` in the checks before any copy (the free space read, here an I/O error):
    ``finish`` exits 13 naming it, never 1 with a traceback; nothing is written, and the
    checkpoints stay where they are, only on the container disk."""
    path, out, run_dir, files = technical_world(tmp_path)
    kept = kept_folder(qualify, monkeypatch, tmp_path)

    def unreadable(folder):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(qualify, "free_bytes", unreadable)
    capsys.readouterr()
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.NOT_EXPORTED
    err = capsys.readouterr().err
    assert "Input/output error" in err and "only on the container disk" in err
    assert not list(case_folder(out, FA).glob("kept-*")) and not any(kept.iterdir())
    assert all(file.exists() for file in files.values())


def failing_at(qualify, monkeypatch, where):
    """The export failing at ``where``: a folder made, ``MANIFEST.json`` or the case's manifest
    published, or ``SHA256SUMS`` written."""
    if where == "make_folders":
        monkeypatch.setattr(qualify, "make_folders", no_space)
    elif where == "write_durably":
        monkeypatch.setattr(qualify, "write_durably", no_space)
    else:
        publish_record = q.publish

        def publish(path, record):
            if where == "MANIFEST.json":
                failing = path.name == "MANIFEST.json"
            else:
                failing = path.name.startswith("kept-checkpoints-")
            if failing:
                no_space()
            publish_record(path, record)

        monkeypatch.setattr(q, "publish", publish)


@pytest.mark.parametrize("where", ["make_folders", "MANIFEST.json", "write_durably",
                                   "the case's manifest"])  # fmt: skip
def test_every_export_failure_is_13(qualify, tmp_path, monkeypatch, capsys, where) -> None:
    """An ``OSError`` anywhere in the export, after its refusal checks (a folder made, a manifest
    published, ``SHA256SUMS`` written): ``finish`` exits 13 naming the error, never 1 with a
    traceback; no case manifest is published, so nothing is called verified, and the
    checkpoints stay where they are."""
    path, out, run_dir, files = technical_world(tmp_path)
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    failing_at(qualify, monkeypatch, where)
    capsys.readouterr()
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.NOT_EXPORTED
    assert "No space left on device" in capsys.readouterr().err
    assert not list(case_folder(out, FA).glob("kept-checkpoints-*"))
    assert all(file.exists() for file in files.values())


def test_finish_again_reuses_a_verified_export(qualify, tmp_path, monkeypatch) -> None:
    """Run again over an export whose copies all re-hash to its manifest, the case's files
    unchanged: nothing is copied, and ``finish`` exits 9. A file added since is exported anew,
    the whole case as ``kept-2``; ``kept-1`` stays."""
    path, out, run_dir, files = technical_world(tmp_path)
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    before = tree(kept)
    copies, copy = [], qualify.copy_durably
    monkeypatch.setattr(
        qualify,
        "copy_durably",
        lambda source, target: copies.append(source) or copy(source, target),
    )
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    assert copies == [] and tree(kept) == before
    folder = case_folder(out, FA)
    assert [f.name for f in folder.glob("kept-checkpoints-*")] == ["kept-checkpoints-1.json"]
    late = files["state"].with_name("step-10.pt")
    late.write_bytes(b"state at step 10")
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    assert sorted(f.name for f in folder.glob("kept-checkpoints-*")) == [
        "kept-checkpoints-1.json", "kept-checkpoints-2.json"]  # fmt: skip
    assert sorted(copies) == sorted([*files.values(), late])
    after = tree(kept)
    assert all(after[name] == data for name, data in before.items())


def test_checkpoints_lost_before_the_export_leave_the_technical_mark(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """The container disk lost after the case's records and before ``finish``: nothing is left
    to copy. The technical mark stands: ``finish`` exits 9, saying so; its manifest's ``lost``
    names each checkpoint a binding record names, with its recorded sha256 and the records
    naming it, and the export holds the manifest alone, which ``kept-check`` verifies;
    ``restart-case`` refuses the complete case, and the verdict stays technical."""
    path, out, run_dir, files = technical_world(tmp_path)
    shutil.rmtree(tmp_path / "ckpt")
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    planned = qualify.campaign_cases(path, SCOPE, 1, False)[1][0]
    reasons = qualify.technical_reasons(out, planned, SCOPE)
    assert reasons
    capsys.readouterr()
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    assert LOST_MESSAGE in capsys.readouterr().err
    folder = case_folder(out, FA)
    manifest = json.loads((folder / "kept-checkpoints-1.json").read_text())
    assert manifest["files"] == [] and manifest["verified"] is True
    lost = {entry["path"]: entry for entry in manifest["lost"]}
    assert set(lost) == {str(files["launcher"]), str(files["baseline"])}
    assert lost[str(files["launcher"])]["recorded_sha256"] == sha(WEIGHTS)
    assert lost[str(files["baseline"])]["recorded_sha256"] == sha(BASE)
    assert str(run_dir / "summary.json") in {
        r["record"] for r in lost[str(files["launcher"])]["named_by"]}  # fmt: skip
    assert [r["record"] for r in lost[str(files["baseline"])]["named_by"]] == [
        str(out / FA.attempt_path("baseline", 1))]  # fmt: skip
    export = kept / manifest["export"]
    assert sorted(p.name for p in export.iterdir()) == ["MANIFEST.json", "SHA256SUMS"]
    inventory = archived(kept, tmp_path / "inventory.tsv", monkeypatch)
    assert exit_code(qualify, kept_check_args(path, FA, inventory)) == 0
    with pytest.raises(SystemExit, match="the case is complete"):
        qualify.main(restart(path, FA, RESTART))
    assert not list(folder.glob("restart-*"))
    assert qualify.technical_reasons(out, planned, SCOPE) == reasons
    assert qualify.case_state(out, planned, SCOPE)[0] is True
    qualify.main(["verdict", *common(path)])
    verdict = json.loads((out / "c1" / "verdict.json").read_text())
    assert verdict["rates"][f"adam_frozen-{FA_LR:g}"]["control"]["kind"] == "technical"


def test_release_needs_a_verified_export(qualify, tmp_path, monkeypatch, capsys) -> None:
    """``finish --release`` deletes a technical case's checkpoints only once a ``kept-check`` of
    exit 0 binds the export it relies on (its receipt names that manifest's sha256 and its
    inventory's, both re-hashed), and every file still matches its exported copy by size and
    sha256: without an export, before ``kept-check`` has passed, after one that did not, with
    the receipt's inventory changed or gone (named), the manifest changed since, or a file added
    or changed since the export, it exits 13 and deletes nothing. Its release record names the
    manifest and the receipt; the volume copy stays."""
    path, out, run_dir, files = technical_world(tmp_path)
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    release = finish_args(path, FA, None, "--release")
    folder = case_folder(out, FA)

    def refused():
        return (
            exit_code(qualify, release) == qualify.NOT_EXPORTED
            and all(file.exists() for file in files.values())
            and not list(folder.glob("checkpoints-removed-*"))
        )

    assert refused()  # no export
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    assert refused()  # no kept-check yet
    inventory = archived(kept, tmp_path / "inventory.tsv", monkeypatch)
    text = inventory.read_text()
    inventory.write_text(text.replace("\tok\n", "\tMISMATCH remote=None,None\n", 1))
    assert exit_code(qualify, kept_check_args(path, FA, inventory)) == qualify.NOT_EXPORTED
    assert refused()  # its kept-check exited 13
    inventory.write_text(text)
    assert exit_code(qualify, kept_check_args(path, FA, inventory)) == 0
    for change in ("the inventory changed", "the inventory gone", "the manifest changed"):
        target = folder / "kept-checkpoints-1.json" if "manifest" in change else inventory
        saved = target.read_bytes()
        if change == "the inventory gone":
            target.unlink()
        else:
            target.write_bytes(saved + b"\n")
        capsys.readouterr()
        assert refused(), change
        if "inventory" in change:
            assert str(inventory) in capsys.readouterr().err, change
        target.write_bytes(saved)
    late = files["state"].with_name("step-10.pt")
    late.write_bytes(b"state at step 10")
    assert refused() and late.exists()  # a file the export does not hold
    late.unlink()
    files["state"].write_bytes(b"another state")
    assert refused()  # a file changed since the export
    files["state"].write_bytes(STATE)
    qualify.main(release)
    assert not any(file.exists() for file in files.values())
    assert not (tmp_path / "ckpt" / "harness" / "s10200-adam_frozen-1.83e-05").exists()
    (record,) = [json.loads(f.read_text()) for f in folder.glob("checkpoints-removed-*")]
    manifest = folder / "kept-checkpoints-1.json"
    receipt = folder / "kept-check-2.json"
    assert record["manifests"] == [{"path": str(manifest), "sha256": sha(manifest.read_bytes())}]
    assert record["kept_check"] == {"path": str(receipt), "sha256": sha(receipt.read_bytes())}
    assert str(files["launcher"]) in record["removed"] and record["release"] is True
    assert qualify.checkpoint_missing(run_dir, folder) is False  # removed on purpose
    export = kept / json.loads(manifest.read_text())["export"]
    assert all((export / f.relative_to(tmp_path / "ckpt")).is_file() for f in files.values())


def test_release_relies_on_an_older_export_a_kept_check_passed(
    qualify, tmp_path, monkeypatch
) -> None:
    """``kept-1`` verified by ``kept-check``, then its volume copy damaged, so ``finish`` writes
    ``kept-2``, which no ``kept-check`` has passed: the release relies on ``kept-1``, the case's
    files unchanged, and its record names ``kept-1``'s manifest."""
    path, out, run_dir, files = technical_world(tmp_path)
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    inventory = archived(kept, tmp_path / "inventory.tsv", monkeypatch)
    assert exit_code(qualify, kept_check_args(path, FA, inventory)) == 0
    folder = case_folder(out, FA)
    first = json.loads((folder / "kept-checkpoints-1.json").read_text())
    (kept / first["files"][0]["copy"]).write_bytes(b"damaged on the volume")
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    assert (folder / "kept-checkpoints-2.json").is_file()
    qualify.main(finish_args(path, FA, None, "--release"))
    assert not any(file.exists() for file in files.values())
    (record,) = [json.loads(f.read_text()) for f in folder.glob("checkpoints-removed-*")]
    manifest = folder / "kept-checkpoints-1.json"
    assert record["manifests"] == [{"path": str(manifest), "sha256": sha(manifest.read_bytes())}]
    assert record["kept_check"]["path"] == str(folder / "kept-check-1.json")


def test_release_checks_each_file_again_right_before_its_deletion(
    qualify, tmp_path, monkeypatch
) -> None:
    """A file that changes after the release's check of every file and before its own deletion:
    the release stops there (exit 13); the files deleted before it, each matching its copy, are
    recorded, and it and the rest are kept."""
    path, out, run_dir, files = technical_world(tmp_path)
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    inventory = archived(kept, tmp_path / "inventory.tsv", monkeypatch)
    assert exit_code(qualify, kept_check_args(path, FA, inventory)) == 0
    order = [files["launcher"], files["state"], files["baseline"]]  # case_files' order
    checked, check = [], qualify.matches_export

    def changing(source, entry):
        checked.append(source)
        if len(checked) == len(order) + 2:  # the deletion's check of the second file
            source.write_bytes(b"changed in between")
        return check(source, entry)

    monkeypatch.setattr(qualify, "matches_export", changing)
    assert exit_code(qualify, finish_args(path, FA, None, "--release")) == qualify.NOT_EXPORTED
    assert checked == [*order, *order[:2]]
    assert not order[0].exists() and order[1].exists() and order[2].exists()
    (record,) = [json.loads(f.read_text())
                 for f in case_folder(out, FA).glob("checkpoints-removed-*")]  # fmt: skip
    assert record["removed"] == [str(order[0])]


KEPT_CHECK_CHANGES = {  # how the inventory departs from the export → what kept-check names
    "a missing line": (lambda rows, export: [r for r in rows if not r.startswith(
        f"{export}/r-f1.pt\t")], "r-f1.pt"),
    "another size": (lambda rows, export: [field(r, 1, "999", export) for r in rows],
                     "r-f1.pt"),
    "another sha256": (lambda rows, export: [field(r, 2, "0" * 64, export) for r in rows],
                       "r-f1.pt"),
    "not verified": (lambda rows, export: [field(r, 5, "MISMATCH remote=1,x", export)
                                           for r in rows], "r-f1.pt"),
    "no MANIFEST.json line": (lambda rows, export: [r for r in rows if not r.startswith(
        f"{export}/MANIFEST.json\t")], "MANIFEST.json"),
    "no SHA256SUMS line": (lambda rows, export: [r for r in rows if not r.startswith(
        f"{export}/SHA256SUMS\t")], "SHA256SUMS"),
    "another URL": (lambda rows, export: [field(r, 4, "gs://bucket/elsewhere/r-f1.pt", export)
                                          for r in rows], "archived at gs://bucket/elsewhere"),
    "not a gs:// URL": (lambda rows, export: [field(r, 4, f"/local/{export}/r-f1.pt", export)
                                              for r in rows], "archived at /local/"),
    "another prefix": (lambda rows, export: [field(
        r, 4, f"gs://other-bucket/kept-checkpoints/{export}/r-f1.pt", export) for r in rows],
        "gs://other-bucket/kept-checkpoints/"),
}  # fmt: skip


def field(row, index, value, export):
    """``row`` of the inventory with field ``index`` set to ``value``, when it is r-f1.pt's."""
    cells = row.split("\t")
    if cells[0] == f"{export}/r-f1.pt":
        cells[index] = value
    return "\t".join(cells)


def test_kept_check_verifies_the_volume_copy_and_the_inventory(
    qualify, tmp_path, monkeypatch, capsys
) -> None:
    """``kept-check`` exits 0 only when the copies of the case's latest manifest re-hash to it on
    the volume and archive.py's inventory has a line for each exported file (the copies,
    ``MANIFEST.json``, ``SHA256SUMS``) with its size and sha256 and the verified status; 13
    otherwise, naming what fails. Each check publishes its receipt (``kept-check-<n>.json``):
    the manifest's and the inventory's path and sha256, the exit, and ``pending-upload`` when the
    volume copy is verified and the bucket copy is not."""
    path, out, run_dir, files = technical_world(tmp_path)
    kept = kept_folder(qualify, monkeypatch, tmp_path)
    folder = case_folder(out, FA)
    none = tmp_path / "none.tsv"
    assert exit_code(qualify, kept_check_args(path, FA, none)) == qualify.NOT_EXPORTED
    assert not list(folder.glob("kept-check-*"))  # no export: no receipt
    assert exit_code(qualify, finish_args(path, FA, kept)) == qualify.TECHNICAL_CASE
    capsys.readouterr()  # the upload failed before writing its inventory: the upload pending
    assert exit_code(qualify, kept_check_args(path, FA, none)) == qualify.NOT_EXPORTED
    assert "cannot be read" in capsys.readouterr().err
    pending = json.loads((folder / "kept-check-1.json").read_text())
    assert (pending["exit"], pending["pending-upload"]) == (13, True)
    assert pending["inventory"] == {"path": str(none), "sha256": None}
    assert pending["teardown"].startswith("the volume copy is verified and kept, the upload")
    inventory = archived(kept, tmp_path / "inventory.tsv", monkeypatch)
    assert exit_code(qualify, kept_check_args(path, FA, inventory)) == 0
    manifest = folder / "kept-checkpoints-1.json"
    receipt = json.loads((folder / "kept-check-2.json").read_text())
    assert receipt["manifest"] == {"path": str(manifest), "sha256": sha(manifest.read_bytes())}
    assert receipt["inventory"] == {"path": str(inventory), "sha256": sha(inventory.read_bytes())}
    assert (receipt["exit"], receipt["pending-upload"], receipt["problems"]) == (0, False, [])
    assert receipt["remote"] == REMOTE
    with pytest.raises(SystemExit, match="--remote"):  # bound to a prefix, always
        qualify.main(kept_check_args(path, FA, inventory)[:-2])
    capsys.readouterr()  # the same inventory checked against another prefix: every line named
    other_prefix = "gs://bucket/dryrun/dry-1/kept-checkpoints"
    assert exit_code(qualify, kept_check_args(path, FA, inventory, other_prefix)) == 13
    err = capsys.readouterr().err
    assert all(f"{row.split(chr(9))[0]}: archived at" in err for row in
               inventory.read_text().splitlines()[1:])  # fmt: skip
    number = 3
    export = json.loads(manifest.read_text())["export"]
    text = inventory.read_text()
    rows = text.splitlines()
    for what, (change, named) in KEPT_CHECK_CHANGES.items():
        inventory.write_text("\n".join(change(rows, export)) + "\n")
        capsys.readouterr()
        assert exit_code(qualify, kept_check_args(path, FA, inventory)) == 13, what
        assert named in capsys.readouterr().err, what
        number += 1
        receipt = json.loads((folder / f"kept-check-{number}.json").read_text())
        assert (receipt["exit"], receipt["pending-upload"]) == (13, True), what
        assert "planned teardown waits for kept-check" in receipt["teardown"]
    other = archived(kept, tmp_path / "other.tsv", monkeypatch, change=lambda name, size, md5: (
        (size, "x") if name.endswith("/r-f1.pt") else (size, md5)))  # fmt: skip
    assert exit_code(qualify, kept_check_args(path, FA, other)) == 13  # another object
    inventory.write_text(text)
    assert exit_code(qualify, kept_check_args(path, FA, inventory)) == 0
    (kept / export / "r-f1.pt").write_bytes(b"changed on the volume")
    capsys.readouterr()
    assert exit_code(qualify, kept_check_args(path, FA, inventory)) == 13
    assert "r-f1.pt" in capsys.readouterr().err
    receipts = sorted(folder.glob("kept-check-*"), key=lambda f: int(f.stem.rsplit("-", 1)[1]))
    receipt = json.loads(receipts[-1].read_text())
    assert (receipt["exit"], receipt["pending-upload"]) == (13, False)  # the worse state
    assert receipt["teardown"].startswith("planned teardown waits for kept-check")


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


def journal(root, path, label, run_ids, **extra):
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
                                "command": run.command()[1:], **extra}) + "\n")  # fmt: skip
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
