"""Study 3's engineering cycle outside the numerical qualification (analysis/study3_cycle.py, and
its qualification side in analysis/qualification_records.py).

docs/study3-preregistration.md: "Failure routes" (one engineering cycle per candidate: diagnosis,
a fix, re-qualification of the affected cases and reruns of the affected runs) and "Voiding and
supersession" (the affected records listed, the scientific failures kept). The claims a cycle
makes per stage, each run's GPU code and the journal prefix a cycle pins are tested on synthetic
launch logs and endpoints; the qualification's entries of a cycle on the qualification tests'
worlds, in the formats scripts/qualify.py writes; the published cycle read back on the files of
the campaign writer's route (``cycle_route``), with the real readers and qualify.py's own journal
repair."""

import copy
import hashlib
import json
import math
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf
from test_qualification import CODE
from test_qualification_records import (
    ADAMW,
    ENGINEERING,
    ENGINEERING_HEADER,
    REGISTERED,
    SCIENTIFIC_CASE,
    beyond,
    technical_cycle,
)
from test_qualification_records import World as QualificationWorld
from test_qualification_records import qualify as qualify  # noqa: F401  (the fixture)
from test_qualify import ADAMW_LR, FIX, codes, load_script
from test_study3_campaigns import ARM_OF, START, run_config
from test_study3_campaigns import target as at_target
from test_study3_package2 import (
    REAL_IDENTITY,
    SUPPLEMENT_SEEDS,
    N,
    Route,
    SealedGuard,
    numerical,
    round0,
)
from test_study3_progress import OPT, Study, ep, run_of
from test_study3_records import write_run
from test_study3_versions import FIRST, SECOND, git

from zero_field_refusal import code_identity
from zero_field_refusal import launch as launcher
from zero_field_refusal.analysis import qualification_records as qr
from zero_field_refusal.analysis import study3_cycle as cycle
from zero_field_refusal.analysis.study3_decisions import (
    _problems,
    bridge_name,
    cap_of,
    cycle_path,
    decision_path,
    end_lines,
    serialized,
    start_lines,
)
from zero_field_refusal.analysis.study3_progress import _bound, progress
from zero_field_refusal.analysis.study3_records import (
    COUNTING,
    GROSS_COLLAPSE,
    SEEDS,
    EndpointReader,
)
from zero_field_refusal.analysis.study3_search import CANDIDATES
from zero_field_refusal.config import config_hash
from zero_field_refusal.launch import campaign_header, load_campaign
from zero_field_refusal.qualification_constants import FAILING
from zero_field_refusal.tracking import git_state


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


# ------------------------------------------------------------------ the qualification's entries


def launched(world) -> list[tuple[str, str]]:
    """Every start in the world's launch journal: (run id, label)."""
    out = []
    for log in sorted((world.root / "launch").glob("*.jsonl")):
        for line in log.read_text().splitlines():
            record = json.loads(line)
            if record.get("event") == "start":
                out.append((record["run_id"], record["run"]))
    return out


def all_runs(world, code: str) -> dict[str, str]:
    """Every launched run under ``code``, as ``run_gpu_codes`` would give it."""
    return {run_id: code for run_id, _ in launched(world)}


def runs_except(world, case) -> set[str]:
    """Every launched run but those of ``case``'s label."""
    label = f"{case.optimizer}-{case.lr:g}/s{case.seed}"
    return {run_id for run_id, found in launched(world) if found != label}


def entries_of(world, **kw):
    return qr.cycle_entries(world.root, 1, old_code=CODE["sha256"], fix_code=FIX["sha256"],
                            settings=REGISTERED, run_codes=all_runs(world, CODE["sha256"]),
                            **kw)  # fmt: skip


def engineering_and_numerical_world(tmp_path):
    """Candidate 1's engineering case recorded in c1/engineering, and its numerical
    qualification in c1/numerical, all under the old code."""
    engineering = QualificationWorld(tmp_path, ((("adamw", ADAMW_LR), (10000,)),),
                                     name="study3-c1-engineering", header=ENGINEERING_HEADER,
                                     role="engineering")  # fmt: skip
    engineering.qualified(ENGINEERING, engineering.launch(ENGINEERING))
    world = QualificationWorld(tmp_path)
    world.full()
    return world


def test_the_bridge_mirrors_supersede_and_keeps_a_scientific_case_whole(qualify, monkeypatch,
                                                                        tmp_path):  # fmt: skip
    world = QualificationWorld(tmp_path)
    world.full(changes={SCIENTIFIC_CASE: beyond("rest", "baseline~s3")})  # old code, one scientific
    entries = qr.cycle_entries(world.root, 1, old_code=CODE["sha256"], fix_code=FIX["sha256"],
                               settings=REGISTERED,
                               run_codes=all_runs(world, CODE["sha256"]))  # fmt: skip
    assert {e["case"] for e in entries.kept} == {SCIENTIFIC_CASE.name}
    assert SCIENTIFIC_CASE.name not in {c["case"] for c in entries.cases_requalified}
    assert {e["stage"] for e in entries.attempts} == {"numerical"} and entries.unsettled == []
    assert all(e["sha256"] == sha(e["path"]) for e in entries.attempts)
    assert {e["run_id"] for e in entries.launchers} == runs_except(world, SCIENTIFIC_CASE)
    (kept,) = entries.kept_cases
    assert (kept["case"], kept["stage"]) == (SCIENTIFIC_CASE.name, "numerical")
    assert kept["why"].startswith("a scientific failure keeps counting: [")
    assert all(Path(e["path"]).is_absolute() for e in [*entries.attempts, *entries.kept])
    launcher = entries.launchers[0]
    assert set(launcher) == {"run_id", "campaign", "run", "experiment", "config_hash", "identity"}
    assert launcher["identity"] == qr.launcher_identity(world.root / "study3" / launcher["run_id"])


def test_a_launcher_run_of_the_fixs_code_is_not_superseded(qualify, monkeypatch, tmp_path):
    world = QualificationWorld(tmp_path)
    world.full()
    codes = all_runs(world, CODE["sha256"])
    fixed = sorted(codes)[0]
    entries = qr.cycle_entries(world.root, 1, old_code=CODE["sha256"], fix_code=FIX["sha256"],
                               settings=REGISTERED,
                               run_codes={**codes, fixed: FIX["sha256"]})  # fmt: skip
    assert fixed not in {e["run_id"] for e in entries.launchers}
    assert {e["run_id"] for e in entries.launchers} == set(codes) - {fixed}


def test_an_unclassified_qualification_run_the_readers_cannot_settle_is_named(qualify, monkeypatch,
                                                                     tmp_path):  # fmt: skip
    world = QualificationWorld(tmp_path)
    world.full()
    extra = world.launch(ADAMW, stop="instability", steps=11)  # a stop no record classified
    entries = qr.cycle_entries(world.root, 1, old_code=CODE["sha256"], fix_code=FIX["sha256"],
                               settings=REGISTERED,
                               run_codes=all_runs(world, CODE["sha256"]))  # fmt: skip
    assert entries.unsettled == [extra["run_id"]]
    assert extra["run_id"] not in {e["run_id"] for e in entries.launchers}


def test_an_unclassified_run_the_launcher_never_settled_is_superseded(qualify, monkeypatch,
                                                                      tmp_path):  # fmt: skip
    """A label's later attempt that ended with no published terminal record: the readers read
    it technical, so the cycle supersedes it; one whose meta.json cannot be read is named
    instead."""
    world = QualificationWorld(tmp_path)
    world.full()
    extra = world.launch(ADAMW)
    run_dir = world.root / "study3" / extra["run_id"]
    (run_dir / "terminal.json").unlink()
    entries = entries_of(world)
    assert entries.unsettled == []
    assert extra["run_id"] in {e["run_id"] for e in entries.launchers}
    (run_dir / "meta.json").write_text("{")
    assert entries_of(world).unsettled == [extra["run_id"]]


def test_the_pre_cycle_view_reads_the_pinned_prefix_without_the_bridge(qualify, monkeypatch,
                                                                       tmp_path):  # fmt: skip
    world = QualificationWorld(tmp_path)
    world.full()
    before = entries_of(world)
    prefix = {entry["path"]: entry["lines"] for entry in cycle.journal_seen(world.root)}
    extra = world.launch(ADAMW, stop="instability", steps=11)  # after the prefix
    assert entries_of(world).unsettled == [extra["run_id"]]
    assert entries_of(world, prefix=prefix) == before
    bridge = world.out / "c1" / "supersede-1.json"
    bridge.write_text(json.dumps({"candidate": 1, "supersession": 1,
                                  "attempts": before.attempts,
                                  "launchers": before.launchers}))  # fmt: skip
    voided = entries_of(world, prefix=prefix)
    assert before.attempts and voided.attempts == []  # void already, as the bridge names them
    where = "qualification/c1/supersede-1.json"
    assert entries_of(world, prefix=prefix, without=where) == before


def test_old_code_engineering_records_are_not_voided(qualify, monkeypatch, tmp_path):
    world = engineering_and_numerical_world(tmp_path)  # c1/engineering and c1/numerical, old code
    entries = qr.cycle_entries(world.root, 1, old_code=CODE["sha256"], fix_code=FIX["sha256"],
                               settings=REGISTERED,
                               run_codes=all_runs(world, CODE["sha256"]))  # fmt: skip
    assert all(e["stage"] != "engineering" for e in [*entries.attempts, *entries.kept])
    engineering = {run_id for run_id, label in launched(world) if "/s10000" in label}
    assert engineering and not engineering & {e["run_id"] for e in entries.launchers}
    assert entries.unsettled == []


# ------------------------------------------------------------------ deriving and publishing

OLD, FIXED = "b" * 64, "f" * 64  # the GPU code in force before the cycle, and the fix's
WRITTEN = "2026-10-28T12:00:00+00:00"
LOGS = {"search": "study3-c1-search-round0", "numerical": "study3-c1-numerical",
        "validation": "study3-c1-validation",
        "confirmation": "study3-c1-confirmation"}  # fmt: skip


def reuse(justification: str = "the fix changes nothing these runs ran") -> cycle.Claim:
    return cycle.Claim("reuse", (), justification)


def rerun(arms=("A", "F", "P"), justification: str = "the fix changes what these runs ran"):
    return cycle.Claim("rerun", tuple(arms), justification)


ALL_RERUN = {"search": rerun(), "qualification": rerun(), "validation": rerun()}


def materialized_validation_mark(tmp_path, *, arm="A", failed=None):
    """Candidate 1 through its validation (``Study.through_validation``), its validation run of
    ``arm`` at its lower rate on 10305 a technical mark (ended technical, no terminal record);
    ``failed``: (arm, seed) of a validation run at the upper rate that collapsed (more than 15
    pp of MMLU lost). Each endpoint's files written under ``tmp_path/runs/study3/<run_id>`` (its
    config.yaml, meta.json and terminal.json), its start and end logged per step, each numerical
    run classified by its case's baseline record (old code), and the endpoints pointed there."""
    study = Study()
    study.through_validation()
    if failed is not None:
        study.change(run_of(study, "validation", failed[0], "u", failed[1]), state="unusable",
                     unusable_reason=GROSS_COLLAPSE, mmlu_loss_pp=20.0)  # fmt: skip
    mark = run_of(study, "validation", arm, "l", 10305)
    return study, materialize(study, tmp_path, mark)


def materialize(study, tmp_path, mark: str) -> Path:
    """The study's runs on files (``materialized_validation_mark``), ``mark`` a technical mark
    (ended technical, no terminal record): RUNS_DIR."""
    root = tmp_path / "runs"
    study.change(mark, state="technical", K=None, K_harmful=None, steps=None, mmlu_loss_pp=None,
                 factual_loss_pp=None)  # fmt: skip
    clock = iter(range(10_000))
    for e in sorted(study.endpoints, key=lambda e: e.run_id):
        run_dir = root / "study3" / e.run_id
        run_dir.mkdir(parents=True)
        cfg = OmegaConf.create({"optimizer": {"name": OPT[e.arm], "lr": e.lr}, "seed": e.seed,
                                "study3": {"role": e.role}})  # fmt: skip
        chash = config_hash(cfg)
        (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        (run_dir / "meta.json").write_text(json.dumps({"config_hash": chash, "status": (
            "failed" if e.run_id == mark else "completed")}))  # fmt: skip
        if e.run_id != mark:
            (run_dir / "terminal.json").write_text(json.dumps({"steps": 25}))
        head = {"study": 3, "candidate": 1, "step": e.cell.step, "round": e.cell.round}
        log = root / "launch" / f"{LOGS[e.cell.step]}.jsonl"
        when = f"2026-10-26T{next(clock) // 3600:02d}:{next(clock) % 60:02d}:00+00:00"
        common = {"time": when, "run": e.label, "attempt": 1, "run_id": e.run_id}
        write_lines(log, [*([json.loads(x) for x in log.read_text().splitlines()]
                            if log.exists() else []),
                          {**common, "event": "start", "experiment": "study3",
                           "config_hash": chash, "header": head},
                          {**common, "event": "end",
                           "outcome": "technical" if e.run_id == mark else "done"}])  # fmt: skip
        if e.cell.step == "numerical":
            case = root / "qualification" / "c1" / "numerical" / f"s{e.seed}" / (
                f"{OPT[e.arm]}-{e.lr:g}")  # fmt: skip
            case.mkdir(parents=True)
            (case / "baseline-attempt-1.json").write_text(json.dumps({
                "case": {"candidate": 1, "optimizer": OPT[e.arm], "lr": e.lr, "seed": e.seed},
                "launcher": {"identity": {"run_id": e.run_id}, "classified": True},
                "provenance": {"gpu_code": {"sha256": OLD}}}))  # fmt: skip
        study.change(e.run_id, run_dir=run_dir)
    study.starts, study.ends = start_lines(root), end_lines(root)
    return root


def derive(study, root, *, claims, stage="validation", marks=None, closed_arms=(), **kw):
    """The cycle derived from the study's records as they are, its marks the stage's technical
    runs unless ``marks`` names them."""
    if marks is None:
        marks = [e.run_id for e in study.endpoints
                 if e.cell.step == stage and e.state == "technical"]  # fmt: skip
    fields = {"records": study.records(), "marks": marks, "stage": stage,
              "closed_arms": closed_arms, "old_code": OLD, "fixation": None, "claims": claims,
              "diagnosis": "a stale kernel cache", "fix": "the cache is cleared at start",
              "code": {"commit": "c" * 40, "gpu_code_sha256": FIXED},
              "fix_identity": {"commit": "c" * 40, "files": 1, "sha256": FIXED},
              "settings": REGISTERED, "repo": root.parent, "written_at": WRITTEN,
              "host": "the researcher's machine", "journal": cycle.journal_seen(root)}  # fmt: skip
    return cycle.derive_cycle(root, 1, **{**fields, **kw})


def sha_of(record) -> str:
    return hashlib.sha256(serialized(record)).hexdigest()


def old_code_validation_runs(study) -> set[str]:
    return {e.run_id for e in study.endpoints
            if e.cell.step == "validation" and e.state in COUNTING}  # fmt: skip


def scientific_runs(study) -> set[str]:
    return {e.run_id for e in study.endpoints if e.state == "scientific"
            or (e.state == "unusable" and e.unusable_reason == GROSS_COLLAPSE)}  # fmt: skip


def test_the_cycle_record_lists_every_affected_record_by_path_and_sha256(tmp_path):
    study, root = materialized_validation_mark(tmp_path)  # through validation, files per endpoint
    record, bridge = derive(study, root, claims={"search": reuse(), "qualification": reuse(),
                                                 "validation": rerun()})  # fmt: skip
    affected = {r["run_id"] for r in record["affected"]["runs"]}
    assert affected == old_code_validation_runs(study) - scientific_runs(study)
    for run in record["affected"]["runs"]:
        assert all(ref["sha256"] == sha(root / ref["path"]) for ref in run["records"]
                   if "sha256" in ref)  # fmt: skip
    assert {x["run_id"] for x in bridge["launchers"]} == affected
    assert bridge["old_record"] == {"path": str(cycle_path(root, 1).resolve()),
                                    "sha256": sha_of(record), "kind": "cycle-c1"}  # fmt: skip
    (mark,) = record["marks"]
    assert mark["line"]["kind"] == "end" and mark["arm"] == "A"
    (run,) = [r for r in record["affected"]["runs"] if r["run_id"] == mark["run_id"]]
    assert {"path": f"study3/{mark['run_id']}/terminal.json", "absent": True} in run["records"]
    assert record["kept"] == [{"why": "reused_stage", "stage": "search", "runs": 24}]
    assert record["affected"]["qualification"] == {"records": [], "launchers": [], "restarts": []}
    assert record["journal"] == cycle.journal_seen(root)
    assert bridge["attempts"] == bridge["held"] == bridge["kept_cases"] == []
    assert (bridge["old_gpu_code"], bridge["fix_gpu_code"]["sha256"]) == ({"sha256": OLD}, FIXED)


def test_the_marks_stage_reruns_every_marked_arm(tmp_path):
    study, root = materialized_validation_mark(tmp_path, arm="F")
    for claims in ({"validation": reuse()}, {"validation": rerun(arms=("A", "P"))}):
        with pytest.raises(cycle.CycleRefused, match="marked"):
            derive(study, root, claims={"search": reuse(), "qualification": reuse(), **claims})


def test_a_cycle_claims_exactly_the_stages_started(tmp_path):
    study, root = materialized_validation_mark(tmp_path)
    with pytest.raises(cycle.CycleRefused, match="search.*no claim"):
        derive(study, root, claims={"qualification": reuse(), "validation": rerun()})
    with pytest.raises(cycle.CycleRefused, match="confirmation.*not started"):
        derive(study, root, claims={**ALL_RERUN, "confirmation": reuse()})
    with pytest.raises(cycle.CycleRefused, match="fix"):
        derive(study, root, claims=ALL_RERUN, code={"commit": "c" * 40, "gpu_code_sha256": OLD})


def test_scientific_failures_and_their_evidence_keep_counting(tmp_path):
    study, root = materialized_validation_mark(tmp_path, failed=("A", 10301))  # A collapsed
    record, bridge = derive(study, root, claims=ALL_RERUN)
    (kept,) = [k for k in record["kept"] if k["why"] == "scientific_failure"]
    assert kept["run_id"] == run_of(study, "validation", "A", "u", 10301)
    assert kept["run_id"] not in {r["run_id"] for r in record["affected"]["runs"]}
    assert kept["evidence"]["sha256"] == sha(root / kept["evidence"]["path"])
    assert kept["run_id"] not in {x["run_id"] for x in bridge["launchers"]}
    numerical = {e.run_id for e in study.endpoints if e.cell.step == "numerical"}
    assert {x["run_id"] for x in record["affected"]["qualification"]["launchers"]} == numerical
    assert len(record["affected"]["qualification"]["records"]) == len(numerical)
    assert {x["run_id"] for x in bridge["launchers"]} == numerical | {
        r["run_id"] for r in record["affected"]["runs"]}  # fmt: skip
    assert {a["sha256"] for a in bridge["attempts"]} == {
        r["sha256"] for r in record["affected"]["qualification"]["records"]}  # fmt: skip


def test_held_closed_and_reused_runs_are_kept_and_runs_of_the_fix_are_not_listed(
    tmp_path, monkeypatch
):
    study, root = materialized_validation_mark(tmp_path)
    p_run = run_of(study, "validation", "P", "l", 10300)
    held = run_of(study, "validation", "F", "u", 10302)
    study.change(held, state="held")
    monkeypatch.setattr(code_identity, "gpu_code_identity",
                        lambda c, *a, **k: {"sha256": {"d" * 40: FIXED}.get(c, OLD)})  # fmt: skip
    fixed = run_of(study, "validation", "A", "u", 10303)
    study.change(fixed, git=("d" * 40, False))
    record, _ = derive(study, root, closed_arms=("P",),
                       claims={"search": rerun(arms=("A",)), "qualification": reuse(),
                               "validation": rerun()})  # fmt: skip
    affected = {r["run_id"]: r for r in record["affected"]["runs"]}
    why = {k.get("run_id"): k["why"] for k in record["kept"]}
    assert why[p_run] == "closed_arm" and why[held] == "held"
    assert fixed not in affected and fixed not in why
    assert {e.run_id for e in study.endpoints if e.cell.step == "search" and e.arm == "A"} <= set(
        affected)  # fmt: skip
    assert {"why": "reused_stage", "stage": "search", "runs": 8} in record["kept"]  # F's (P closed)
    keys = [(k.get("run_id", ""), (k.get("evidence") or {}).get("path", ""), k.get("stage", ""))
            for k in record["kept"]]  # fmt: skip
    assert keys == sorted(keys) and list(affected) == sorted(affected)


def test_a_mark_under_a_known_other_gpu_code_is_no_cycle_case(tmp_path, monkeypatch):
    study, root = materialized_validation_mark(tmp_path)
    monkeypatch.setattr(code_identity, "gpu_code_identity",
                        lambda c, *a, **k: {"sha256": "e" * 64})  # fmt: skip
    (mark,) = [e.run_id for e in study.endpoints if e.state == "technical"]
    study.change(mark, git=("d" * 40, False))
    with pytest.raises(cycle.CycleRefused, match="other_gpu_code"):
        derive(study, root, claims=ALL_RERUN)
    study.change(mark, git=("d" * 40, True))  # a dirty checkout: its code cannot be told
    record, _ = derive(study, root, claims={**ALL_RERUN, "qualification": reuse()})
    assert {r["run_id"]: r["gpu_code"] for r in record["affected"]["runs"]}[mark] is None


def test_a_cycle_refuses_a_qualification_run_the_readers_cannot_settle(tmp_path):
    study, root = materialized_validation_mark(tmp_path)
    numerical = sorted(e.run_id for e in study.endpoints if e.cell.step == "numerical")
    folder = root / "qualification" / "c1" / "numerical"
    (found,) = [p for p in folder.rglob("baseline-attempt-1.json") if numerical[0] in p.read_text()]
    found.unlink()  # no part record classifies its run now
    with pytest.raises(cycle.CycleRefused, match=numerical[0]):
        derive(study, root, claims=ALL_RERUN)
    derive(study, root, claims={**ALL_RERUN, "qualification": reuse()})


def test_a_run_started_after_the_journal_prefix_is_never_listed(tmp_path):
    study, root = materialized_validation_mark(tmp_path)
    journal = cycle.journal_seen(root)
    late = ep("F", 1.83e-5, 10327, step="validation", run_id="late-run")
    study.add(late)
    log = root / "launch" / f"{LOGS['validation']}.jsonl"
    append_line(log, {**start("late-run", "2020-01-01T00:00:00+00:00", late.label),
                      "header": {"study": 3, "candidate": 1, "step": "validation",
                                 "round": None}})  # fmt: skip
    seen = cycle.seen_starts(root, journal)
    affected, kept = cycle.affected_and_kept(study.records(), 1, ALL_RERUN, run_codes={},
                                             fix_code=FIXED, seen=seen, closed_arms=(),
                                             cited=())  # fmt: skip
    assert "late-run" not in {r["run_id"] for r in affected}
    assert "late-run" not in {k.get("run_id") for k in kept}
    assert len(affected) == len(old_code_validation_runs(study)) - 1 + 24  # search too
    with pytest.raises(cycle.CycleRefused, match="late-run"):  # in the journal now, not read
        derive(study, root, claims=ALL_RERUN, records=replace(
            study.records(), endpoints=tuple(e for e in study.endpoints
                                             if e.run_id != "late-run")))  # fmt: skip


QUALIFIED = {**CODE, "sha256": OLD}  # the old GPU code the qualification ran under (a sha256)
VAL_RUN, VAL_RATE, VAL_SEED = "val-run-a1", 1.45e-05, 10300  # a validation run of A, marked
VAL_LABEL, VAL_LOG = f"A-{VAL_RATE:g}/s{VAL_SEED}", "study3-c1-validation"
VAL_HEADER = {"study": 3, "candidate": 1, "step": "validation", "round": None}
ENG_CASE = ENGINEERING.name  # the engineering campaign's first case: the repair's --case
BRIDGE_CLAIMS = {"search": reuse(), "qualification": rerun(), "validation": rerun()}
# a held entry, and a launcher run the cycle record does not list: a bridge naming either is
# rejected
HELD = {"run_id": "r10202-adamw-a1", "campaign": "study3-c1-numerical",
        "run": f"adamw-{ADAMW_LR:g}/s10202", "experiment": "study3", "config_hash": "c" * 12,
        "reason": "no recorded commit (git: {})"}  # fmt: skip
OTHER_RUN = {"run_id": "val-run-b1", "campaign": VAL_LOG, "run": f"F-1.83e-05/s{VAL_SEED}",
             "experiment": "study3", "config_hash": "c" * 12}  # fmt: skip


class BridgeWorld(QualificationWorld):
    """Candidate 1's numerical qualification in qualify.py's formats
    (``test_qualification_records.World``), and one validation run of A, ``VAL_RUN``, started
    under ``VAL_LOG`` from commit abc123 and ended technical (its meta.json failed, no terminal
    record): the mark of a cycle outside the qualification."""

    def engineering_campaign(self) -> Path:
        """The candidate's engineering campaign file (its one case ``ENG_CASE``, on the
        engineering seed): the repair's ``--campaign``."""
        found = QualificationWorld(self.path.parent, ((("adamw", ADAMW_LR), (10000,)),),
                                   name="study3-c1-engineering", header=ENGINEERING_HEADER,
                                   role="engineering")  # fmt: skip
        return found.path

    def validation_run(self) -> None:
        cfg = OmegaConf.create({"optimizer": {"name": "adamw", "lr": VAL_RATE}, "seed": VAL_SEED,
                                "study3": {"role": "validation"}})  # fmt: skip
        chash = config_hash(cfg)
        run_dir = self.root / "study3" / VAL_RUN
        run_dir.mkdir(parents=True)
        (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        (run_dir / "meta.json").write_text(json.dumps({
            "config_hash": chash, "status": "failed",
            "git": {"commit": "abc123", "dirty": False}}))  # fmt: skip
        common = {"run": VAL_LABEL, "attempt": 1, "run_id": VAL_RUN}
        self.log({"event": "start", **common, "experiment": "study3", "config_hash": chash,
                  "header": VAL_HEADER}, VAL_LOG)  # fmt: skip
        self.log({"event": "end", **common, "outcome": "technical"}, VAL_LOG)


def bridge_records(world: BridgeWorld) -> tuple[dict, dict]:
    """A schema-valid cycle record of the world's validation mark (the search reused, the
    qualification and the validation rerun) and its bridge, as ``derive_cycle`` writes them: the
    qualification's side from ``cycle_entries``, the affected run with its records and lines."""
    root = world.root
    entries = qr.cycle_entries(root, 1, old_code=OLD, fix_code=FIXED, settings=REGISTERED,
                               run_codes=all_runs(world, OLD))  # fmt: skip
    assert entries.unsettled == [] and entries.attempts
    ends, starts = end_lines(root), start_lines(root)
    line = ("path", "line", "line_sha256")
    run_dir = root / "study3" / VAL_RUN
    files = [{"path": f"study3/{VAL_RUN}/{name}", "sha256": sha(run_dir / name)}
             if (run_dir / name).exists() else {"path": f"study3/{VAL_RUN}/{name}", "absent": True}
             for name in cycle.RUN_FILES]  # fmt: skip
    run = {"run_id": VAL_RUN, "label": VAL_LABEL, "step": "validation", "arm": "A",
           "rate": VAL_RATE, "seed": VAL_SEED, "gpu_code": None, "records": files,
           "start": {k: starts[VAL_RUN][k] for k in line},
           "end": {k: ends[VAL_RUN][k] for k in line}}  # fmt: skip
    record = {
        "kind": "cycle-c1", "state": "passed", "written_at": WRITTEN,
        "code": {"commit": FIX["commit"], "gpu_code_sha256": FIXED},
        "candidate": 1, "stage": "validation", "old_code": OLD,
        "marks": [{**{k: run[k] for k in ("run_id", "label", "step", "arm", "rate", "seed")},
                   "line": {"kind": "end", **run["end"]}}],
        "diagnosis": DIAGNOSIS, "fix": FIX_TEXT,
        "stages": {name: {"action": c.action, "arms": list(c.arms),
                          "justification": c.justification}
                   for name, c in BRIDGE_CLAIMS.items()},
        "fixation": None, "journal": cycle.journal_seen(root),
        "affected": {"runs": [run], "qualification": cycle._qualification(root, entries),
                     "records": []},
        "kept": [], "bridge": bridge_name(1),
    }  # fmt: skip
    record = json.loads(json.dumps(record))
    (start_,) = [r for r in (json.loads(x) for x in (root / "launch" / f"{VAL_LOG}.jsonl")
                             .read_text().splitlines()) if r["event"] == "start"]  # fmt: skip
    stage_runs = [{"run_id": VAL_RUN, "campaign": VAL_LOG, "run": VAL_LABEL,
                   "experiment": "study3", "config_hash": start_["config_hash"]}]  # fmt: skip
    bridge = cycle.build_bridge(root, 1, cycle_file=cycle_path(root, 1),
                                cycle_sha256=sha_of(record), diagnosis=DIAGNOSIS, fix=FIX_TEXT,
                                old_code=OLD, fix_identity=FIX, entries=entries,
                                stage_runs=stage_runs, host="host", at=WRITTEN,
                                commit=FIX["commit"])  # fmt: skip
    return record, json.loads(json.dumps(bridge))


def bridge_world(qualify, monkeypatch, tmp_path) -> tuple[BridgeWorld, Path, dict]:
    """Candidate 1 qualified under the old code (``QUALIFIED``, commit abc123), its validation
    run ``VAL_RUN`` marked, and the cycle published (``publish_cycle``: the record, then its
    bridge, ``qualification/c1/supersede-1.json``), its journal lines not yet written; this
    checkout's GPU code the fix's (``codes``: HEAD is ``FIX``). Returns the world, the bridge's
    path and the bridge."""
    world = BridgeWorld(tmp_path)
    world.full(code=QUALIFIED)
    world.validation_run()
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)  # this checkout: the fix
    record, bridge = bridge_records(world)
    path, bridge_path = cycle.publish_cycle(world.root, 1, record, bridge)
    world.published = {file: file.read_bytes() for file in (path, bridge_path)}
    return world, bridge_path, bridge


def touch_cycle_record(world: BridgeWorld) -> None:
    """The cycle record's bytes rewritten (its content kept, its layout another): its sha256
    changes."""
    path = cycle_path(world.root, 1)
    path.write_text(json.dumps(json.loads(path.read_text())))


def restore(world: BridgeWorld) -> None:
    """The cycle record and its bridge as published."""
    for path, data in world.published.items():
        path.write_bytes(data)


def test_qualifys_own_supersession_record_is_no_cycle_outside_the_qualification(
    qualify, monkeypatch, tmp_path
):
    """The candidate's cycle written by qualify.py itself, from a technical part record: no cycle
    record, and a supersession record whose old record is a part record, not a cycle record.
    ``read_cycle`` reads no cycle outside the qualification there (``missing``, no problem):
    the candidate is not blocked by it. A record that names a cycle record without one, or one
    that cannot be read, stays a problem."""
    world, path, record = technical_cycle(qualify, monkeypatch, tmp_path)
    found = cycle.read_cycle(world.root, 1)
    assert (found["state"], found["problems"], found["record"]) == ("missing", [], None)
    named = {**record, "old_record": {"path": str(cycle_path(world.root, 1)), "sha256": "0" * 64,
                                      "kind": "cycle-c1"}}  # fmt: skip
    path.write_text(json.dumps(named))
    found = cycle.read_cycle(world.root, 1)
    assert found["state"] == "invalid" and any("study3_summary.py cycle" in p
                                               for p in found["problems"])  # fmt: skip
    path.write_text(json.dumps({k: v for k, v in record.items() if k != "old_record"}))
    assert cycle.read_cycle(world.root, 1)["state"] == "missing"  # it names no cycle record
    path.write_text("{")
    assert cycle.read_cycle(world.root, 1)["state"] == "invalid"


def test_the_cycle_is_published_once_and_a_run_again_writes_only_the_missing_file(tmp_path):
    study, root = materialized_validation_mark(tmp_path)
    claims = {"search": reuse(), "qualification": reuse(), "validation": rerun()}
    record, bridge = derive(study, root, claims=claims)
    path, bridge_path = cycle.publish_cycle(root, 1, record, bridge)
    assert (path, bridge_path) == (cycle_path(root, 1), root / bridge_name(1))
    assert sha(path) == sha_of(record) == bridge["old_record"]["sha256"]
    assert json.loads(bridge_path.read_text()) == bridge
    bridge_path.unlink()
    assert cycle.publish_cycle(root, 1, record, bridge) == (path, bridge_path)
    assert json.loads(bridge_path.read_text()) == bridge
    with pytest.raises(cycle.CycleRefused, match="published"):
        cycle.publish_cycle(root, 1, {**record, "diagnosis": "another"}, bridge)
    with pytest.raises(cycle.CycleRefused, match="published"):
        cycle.publish_cycle(root, 1, record, {**bridge, "host": "another"})
    assert sha(path) == sha_of(record)


# ------------------------------------------------------------------ the cycle read back on files

ROUTE_CLAIMS = {"search": rerun(arms=("A",), justification="the fix changes A's kernel only")}
DIAGNOSIS, FIX_TEXT = "a stale kernel cache", "the cache is cleared at start"
RUN_CLOCK = datetime(2026, 10, 26, tzinfo=UTC)  # the route's journal stamps start here


class CycleRoute(Route):
    """The campaign writer's route (``test_study3_package2.Route``) from the engineering campaign
    to search round 0, one search run of A technical (ended, its meta.json failed, no terminal
    record): the state's engineering cycle. A fix is committed in the route's checkout before
    the cycle is written, so the GPU code changes."""

    def __init__(self, base: Path, monkeypatch) -> None:
        super().__init__(base, monkeypatch)
        self.monkeypatch = monkeypatch
        self.written_at = WRITTEN
        self.old_code = self.code["gpu_code_sha256"]
        self.fix_code: str | None = None
        self.blank: str | None = None  # a stage whose justification ``cycle`` gives blank

    def build(self) -> None:
        assert self.step()["record"] == "engineering"
        self.engineering = self.campaign("engineering")
        self.launch(self.engineering)
        # the engineering campaign file where the operator keeps it: RUNS_DIR/campaigns/
        self.campaign_file = self.root / "campaigns" / f"{load_campaign(self.engineering)[0]}.yaml"
        self.campaign_file.parent.mkdir(exist_ok=True)
        self.campaign_file.write_bytes(self.engineering.read_bytes())
        self.record("engineering", self.records.engineering(code=self.code))
        self.record("simulation", self.records.simulation(code=self.code))
        assert self.step()["campaign"] == "search-round0"
        self.launch(self.campaign("search-round0"), round0)
        (marked,) = [e for e in self.endpoints(step="search")
                     if e.arm == "A" and e.lr == START["A"]["l"] and e.seed == 10100]  # fmt: skip
        (marked.run_dir / "terminal.json").unlink()
        meta = json.loads((marked.run_dir / "meta.json").read_text())
        (marked.run_dir / "meta.json").write_text(json.dumps({**meta, "status": "failed"}))
        self.mark = marked.run_id
        assert {e.run_id for e in self.endpoints(step="search") if e.state == "technical"} == {
            self.mark}  # fmt: skip

    # -- the readers

    def state(self) -> dict:
        """The real ``progress`` under this checkout's GPU code (the fix's, once committed)."""
        return progress(self.root, repo=self.repo, expected_code=self.identity()["sha256"],
                        theta0=self.runs.theta0)  # fmt: skip

    def record(self, kind: str | None = None, record: dict | None = None):
        """With a kind, the route's decision record written (``Route.record``); without one,
        the published cycle record."""
        if kind is not None:
            return super().record(kind, record)
        return self.record_()

    def endpoints(self, step: str):
        """The run records as they are now (the real reader, every bound cell)."""
        reader = EndpointReader(self.root, theta0=self.runs.theta0)
        found, _ = reader.read(_bound(reader.read({})[0]))
        return [e for e in found if e.cell.step == step]

    def pre_cycle(self, without=None):
        return cycle.pre_cycle_records(self.root, 1, journal=cycle.journal_seen(self.root),
                                       without=without, theta0=self.runs.theta0, repo=self.repo,
                                       settings=REGISTERED)  # fmt: skip

    def found(self) -> dict:
        return cycle.read_cycle(self.root, 1, repo=self.repo)

    def cycle_problems(self) -> list[str]:
        """Every problem of the published cycle as the readers read it: ``read_cycle``'s and
        ``cycle_problems``' (its lists against their derivation in the pre-cycle view)."""
        found = self.found()
        return [*found["problems"], *cycle.cycle_problems(
            found, runs_root=self.root, theta0=self.runs.theta0, repo=self.repo,
            settings=REGISTERED)]  # fmt: skip

    def record_(self) -> dict:
        return json.loads(cycle_path(self.root, 1).read_text())

    def bridge(self) -> dict:
        return json.loads((self.root / bridge_name(1)).read_text())

    def affected(self) -> set[str]:
        return {run["run_id"] for run in self.record_()["affected"]["runs"]}

    # -- the writer and the states it refuses

    def commit_fix(self) -> "CycleRoute":
        """A fix of train.py committed: HEAD's GPU code is the fix's (``fix_code``)."""
        (self.repo / "src" / "zero_field_refusal" / "train.py").write_text("x = 2\n")
        self.commit("the fix")
        self.fix_code = self.identity()["sha256"]
        assert self.fix_code != self.old_code
        return self

    def cycle(self, *stages: str, justification: str) -> SimpleNamespace:
        """``study3_summary.py cycle`` run in process: each claim ``--stage``, each item that
        starts with ``--`` as it is, and the justification (blank for ``blank``'s stage)."""
        argv = ["cycle", "--runs-root", self.root, "--candidate", 1, "--diagnosis", DIAGNOSIS,
                "--fix", FIX_TEXT, "--theta0", self.runs.theta0]  # fmt: skip
        for item in stages:
            argv += [item] if item.startswith("--") else ["--stage", item]
        if self.blank is not None:
            justification = f"{self.blank}=   "
        ran = self.run("summary", *argv, "--justification", justification)
        return SimpleNamespace(code=ran.code, out=ran.out, error=ran.err)

    def campaigns(self, kind: str) -> SimpleNamespace:
        """The campaign writer run for ``kind``."""
        out = self.out / f"{kind}-{next(self.names)}.yaml"
        ran = self.run("campaigns", kind, "--runs-root", self.root, "--candidate", 1, "--theta0",
                       self.runs.theta0, "--out", out)  # fmt: skip
        return SimpleNamespace(code=ran.code, out=ran.out, error=ran.err)

    def bridge_path(self) -> Path:
        return self.root / bridge_name(1)

    def qualify_supersession(self) -> "CycleRoute":
        """qualify.py's own supersession record of candidate 1 (its cycle from a part record)."""
        path = self.root / bridge_name(1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "candidate": 1, "supersession": 1, "diagnosis": "d", "fix": "f",
            "old_record": {"path": "/runs/qualification/c1/numerical/s10200/adamw-5e-05/"
                                   "baseline-attempt-1.json", "sha256": "0" * 64},
            "old_gpu_code": {"sha256": self.old_code}, "fix_gpu_code": {"sha256": "e" * 64},
            "cases_requalified": [], "attempts": [], "kept_cases": [], "kept": [],
            "launchers": [], "held": []}, indent=1) + "\n")  # fmt: skip
        return self

    def drop_engineering_campaign(self) -> "CycleRoute":
        self.campaign_file.unlink()
        return self

    def pending_search_run(self, arm: str, label: str) -> "CycleRoute":
        """A search run of ``arm`` at its ``label`` rate on 10101 not ended: its end line gone,
        its meta.json running, no terminal record."""
        (run,) = [e for e in self.endpoints(step="search")
                  if e.arm == arm and e.lr == START[arm][label] and e.seed == 10101]  # fmt: skip
        (run.run_dir / "terminal.json").unlink()
        meta = json.loads((run.run_dir / "meta.json").read_text())
        (run.run_dir / "meta.json").write_text(json.dumps({**meta, "status": "running"}))
        for log in self.root.glob("launch/*.jsonl"):
            lines = log.read_text().splitlines(keepends=True)
            ends = [json.loads(x) for x in lines]
            kept = [x for x, r in zip(lines, ends, strict=True)
                    if not (r.get("event") == "end" and r.get("run_id") == run.run_id)]  # fmt: skip
            if kept != lines:
                log.write_text("".join(kept))
        return self

    def blank_justification(self, stage: str) -> "CycleRoute":
        self.blank = stage
        return self

    def remove_meta(self, run_id: str) -> None:
        (self.root / meta_path(run_id)).unlink()

    def garble(self, name: str) -> None:
        """An A search run's record ``name`` (not the mark's) there and not JSON."""
        (run,) = [e for e in self.endpoints(step="search")
                  if e.arm == "A" and e.lr == START["A"]["u"] and e.seed == 10101]  # fmt: skip
        path = run.run_dir / name
        path.write_text("{not json")
        self.garbled_sha = sha(path)

    # -- the cycle, its repair and the edits

    def write_cycle(self) -> None:
        """A fix committed, then the cycle derived from the records before it and published."""
        train = self.repo / "src" / "zero_field_refusal" / "train.py"
        train.write_text("x = 2\n")
        self.commit("the fix")
        head = git(self.repo, "rev-parse", "HEAD")
        identity = REAL_IDENTITY(head, self.repo)
        journal = cycle.journal_seen(self.root)
        records = cycle.pre_cycle_records(self.root, 1, journal=journal, without=None,
                                          theta0=self.runs.theta0, repo=self.repo,
                                          settings=REGISTERED)  # fmt: skip
        marks = [e.run_id for e in records.endpoints
                 if e.cell.step == "search" and e.state == "technical"]  # fmt: skip
        record, bridge = cycle.derive_cycle(
            self.root, 1, records=records, marks=marks, stage="search", closed_arms=(),
            old_code=self.code["gpu_code_sha256"], fixation=None, claims=ROUTE_CLAIMS,
            diagnosis=DIAGNOSIS, fix=FIX_TEXT,
            code={"commit": head, "gpu_code_sha256": identity["sha256"]}, fix_identity=identity,
            settings=REGISTERED, repo=self.repo, written_at=self.written_at,
            host="host", journal=journal)  # fmt: skip
        cycle.publish_cycle(self.root, 1, record, bridge)

    def repair(self) -> None:
        """``qualify.py supersede``'s journal repair, as the cycle's writer prints it: it writes
        the bridge's ``superseded`` lines."""
        module = load_script()
        self.monkeypatch.setattr(module, "load_run_config", run_config)
        scope = module.q.QualificationScope.of(module.load_settings())
        _, planned, problems = module.campaign_cases(self.engineering, scope, 1, True)
        assert problems == [] and planned
        module.main(["supersede", "--campaign", str(self.engineering), "--candidate", "1",
                     "--engineering", "--case", planned[0].case.name, "--record",
                     self.found()["repair_record"], "--diagnosis", DIAGNOSIS, "--fix", FIX_TEXT,
                     "--runs-root", str(self.root)])  # fmt: skip

    def edit(self, change, *, rebridge: bool = False) -> None:
        """The published record rewritten with ``change``; with ``rebridge``, the bridge's
        ``old_record.sha256`` the new record's."""
        record = self.record_()
        change(record)
        cycle_path(self.root, 1).write_bytes(serialized(record))
        if rebridge:
            bridge = self.bridge()
            bridge["old_record"]["sha256"] = sha_of(record)
            (self.root / bridge_name(1)).write_text(json.dumps(bridge, indent=1) + "\n")

    def omit_one(self, *, rebridge: bool = False) -> None:
        self.edit(lambda r: r["affected"]["runs"].pop(0), rebridge=rebridge)

    def add_one(self, *, rebridge: bool = False) -> None:
        """An F search run listed as affected: the claim reruns A alone."""
        records = self.pre_cycle(without=bridge_name(1))
        (extra,) = [e for e in records.endpoints if e.cell.step == "search" and e.arm == "F"
                    and e.seed == 10100 and e.lr == START["F"]["l"]]  # fmt: skip
        entry = cycle._affected_run(extra, records, {})

        def add(record):
            runs = record["affected"]["runs"]
            runs[:] = sorted([*runs, entry], key=lambda run: run["run_id"])

        self.edit(add, rebridge=rebridge)

    def duplicate_one(self, *, rebridge: bool = False) -> None:
        self.edit(lambda r: r["affected"]["runs"].insert(0, r["affected"]["runs"][0]),
                  rebridge=rebridge)  # fmt: skip

    def reorder(self, *, rebridge: bool = False) -> None:
        self.edit(lambda r: r["affected"]["runs"].reverse(), rebridge=rebridge)

    def skew_clock(self, *, hours: float) -> None:
        """The writer's clock behind the journal's by ``hours``: every journal stamp is later
        than the record's ``written_at``."""
        self.written_at = (RUN_CLOCK - timedelta(hours=hours)).isoformat(timespec="seconds")

    def rewrite_line(self, log: str, number: int) -> None:
        """A line of a launch log rewritten in place (``log`` named by its campaign's stem)."""
        (path,) = self.root.glob(f"launch/{Path(log).stem}*.jsonl")
        lines = path.read_bytes().split(b"\n")
        lines[number - 1] = json.dumps({**json.loads(lines[number - 1]), "x": 1}).encode()
        path.write_bytes(b"\n".join(lines))


class FixedRoute(CycleRoute):
    """The campaign writer's route (``Route.build_to``) through candidate 1's validation fixation
    and its first versions record. Each launcher run of the qualification is classified by its
    case's baseline record under the old code, so that a cycle may rerun the qualification. The
    qualification's reading is the route's, but a rate whose case record the candidate's bridge
    voids (its ``attempts`` name it) is ``incomplete``, unless the reading is the view before the
    cycle (the bridge read as absent)."""

    def build(self) -> None:
        self.build_to("fixation")

    def build_to(self, stage: str) -> None:
        """``Route.build_to``, then the engineering campaign file where the operator keeps it
        (RUNS_DIR/campaigns/): the journal repair takes it."""
        super().build_to(stage)
        self.campaign_file = self.root / "campaigns" / f"{load_campaign(self.engineering)[0]}.yaml"
        self.campaign_file.parent.mkdir(exist_ok=True)
        self.campaign_file.write_bytes(self.engineering.read_bytes())

    def qualify(self, rates) -> None:
        """``Route.qualify``, then every launcher run of the qualification classified by its
        case's baseline record (attempt 1, the old code)."""
        super().qualify(rates)
        for path, ids in self.launched.items():
            if campaign_header(Path(path)).get("step") not in ("numerical", "repair"):
                continue
            _, runs = load_campaign(Path(path))
            for run, run_id in zip(runs, ids, strict=True):
                cfg = run_config(run.config, run.overrides)
                arm, lr = ARM_OF[str(cfg.optimizer.name)], float(cfg.optimizer.lr)
                folder = (self.root / "qualification" / "c1" / "numerical" / f"s{run.seed}"
                          / f"{OPT[arm]}-{lr:g}")  # fmt: skip
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "baseline-attempt-1.json").write_text(
                    json.dumps(
                        {
                            "part": "baseline",
                            "case": {
                                "candidate": 1,
                                "optimizer": OPT[arm],
                                "lr": lr,
                                "seed": run.seed,
                                "engineering": False,
                            },
                            "launcher": {"identity": {"run_id": run_id}, "classified": True},
                            "provenance": {"gpu_code": {"sha256": self.code["gpu_code_sha256"]}},
                        }
                    )
                )

    def qualification(self, runs_root, candidate, *, expected_code, settings, without=None,
                      prefix=None, restarts=None) -> dict:  # fmt: skip
        """The route's reading; a case its bridge voids waits (``incomplete``) until its next
        baseline record under the fix (``requalify``): ``qualified``, or ``failing`` for an arm
        of ``failing``."""
        out = super().qualification(runs_root, candidate, expected_code=expected_code,
                                    settings=settings)  # fmt: skip
        bridge = Path(runs_root) / bridge_name(candidate)
        if without == bridge_name(candidate) or not bridge.is_file():
            return out
        voided = {e.get("sha256") for e in json.loads(bridge.read_text()).get("attempts") or []}
        for key, rate in out["rates"].items():
            folder = Path(rate["cases"][0]["folder"])
            if sha(folder / "baseline-attempt-1.json") not in voided:
                continue
            state = "incomplete"
            if (folder / "baseline-attempt-2.json").is_file():
                state = "failing" if rate["arm"] in getattr(self, "failing", ()) else "qualified"
            rate["state"] = rate["verdict"]["state"] = out["states"][key] = state
        return out

    def validation_mark(self, arm: str, seed: int) -> str:
        """The validation launched, ``arm``'s run at its lower rate on ``seed`` a technical mark
        (its terminal record gone, its meta.json failed): the state's engineering cycle."""
        self.validate()
        (marked,) = [e for e in self.endpoints("validation") if e.arm == arm and e.seed == seed
                     and e.lr == min(x.lr for x in self.endpoints("validation")
                                     if x.arm == arm)]  # fmt: skip
        (marked.run_dir / "terminal.json").unlink()
        meta = json.loads((marked.run_dir / "meta.json").read_text())
        (marked.run_dir / "meta.json").write_text(json.dumps({**meta, "status": "failed"}))
        self.mark = marked.run_id
        return marked.run_id

    def relaunch(self, path: Path, values=at_target) -> list[tuple]:
        """The labels of the campaign ``path`` whose latest attempt is superseded launched again
        (each its next attempt, its own recovery round), under this checkout's commit: each
        launcher run and its run id."""
        name, runs = load_campaign(path)
        header = campaign_header(path)
        superseded = {r["run_id"] for r in superseded_lines(self.root)}
        latest = {}
        for log in self.root.glob("launch/*.jsonl"):
            for line in log.read_text().splitlines():
                record = json.loads(line)
                if record.get("event") == "start":
                    latest[record["run"]] = (record["attempt"], record["run_id"])
        ids = []
        with self.guard.off():
            for run in runs:
                attempt, last = latest.get(run.label, (0, None))
                if last not in superseded:
                    continue
                cfg = run_config(run.config, run.overrides)
                arm, rate = ARM_OF[str(cfg.optimizer.name)], float(cfg.optimizer.lr)
                run_id = (f"20261028T{next(self.run_ids):06d}Z-{run.cell[:28]}-s{run.seed}-"
                          f"{run.config_hash[:8]}-a{attempt + 1}")  # fmt: skip
                self.runs.log(name, {"event": "start", "run": run.label, "attempt": attempt + 1,
                                     "run_id": run_id, "experiment": run.experiment,
                                     "config_hash": run.config_hash,
                                     "command": run.command()[1:], "header": header,
                                     "max_attempts": 1})  # fmt: skip
                write_run(self.root / run.experiment / run_id,
                          OmegaConf.to_yaml(cfg, resolve=True), run.config_hash,
                          {"name": name, "label": run.label, "attempt": attempt + 1,
                           "header": header},
                          k_only=OmegaConf.select(cfg, "eval.k_only", default=False) is True,
                          git=git_state(self.repo),
                          **values(arm, rate, run.seed, header["candidate"]))  # fmt: skip
                self.runs.end(name, run.label, attempt + 1, run_id, "done")
                ids.append((run, run_id))
        return ids

    def requalify(self, failing=()) -> None:
        """The re-qualification under the fix: every superseded launcher run of the
        qualification launched again, and its case's next baseline record under the fix (naming
        the bridge); ``failing``: arms whose rates fail scientifically."""
        self.failing = set(failing)
        fix = self.identity()["sha256"]
        for path in [Path(p) for p in self.launched]:
            step = campaign_header(path).get("step")
            if step not in ("numerical", "repair"):
                continue
            for run, run_id in self.relaunch(path, numerical if step == "numerical"
                                             else at_target):  # fmt: skip
                cfg = run_config(run.config, run.overrides)
                arm, lr = ARM_OF[str(cfg.optimizer.name)], float(cfg.optimizer.lr)
                folder = (self.root / "qualification" / "c1" / "numerical" / f"s{run.seed}"
                          / f"{OPT[arm]}-{lr:g}")  # fmt: skip
                (folder / "baseline-attempt-2.json").write_text(json.dumps({
                    "part": "baseline", "case": {"candidate": 1, "optimizer": OPT[arm], "lr": lr,
                                                 "seed": run.seed, "engineering": False},
                    "launcher": {"identity": {"run_id": run_id}, "classified": True},
                    "engineering_cycle": {"path": str(self.bridge_path())},
                    "provenance": {"gpu_code": {"sha256": fix}}}))  # fmt: skip

    def fixed_code(self) -> dict:
        head = git(self.repo, "rev-parse", "HEAD")
        return {"commit": head, "gpu_code_sha256": self.identity()["sha256"]}

    def write(self, what: str, kind: str) -> SimpleNamespace:
        """``study3_summary.py fixation <kind>`` (``what`` ``fixation``), or ``record <kind>``
        with the record the state's step gives the fields of (or, without them, the route's
        n and budget records), under HEAD's code."""
        if what == "fixation":
            seeds = SUPPLEMENT_SEEDS if kind == "confirmation" else ("--theta0", self.runs.theta0)
            ran = self.run("summary", "fixation", kind, "--runs-root", self.root, "--candidate",
                           1, *seeds)  # fmt: skip
        else:
            step = self.state()["step"]
            fields = step.get("fields") or {}
            if kind.startswith("n-"):
                record = self.records.n(1, N, rates=fields.get("rates", {}),
                                        active=fields.get("active", ["A-F"]),
                                        cohort=fields.get("cohort", {"seeds": [], "run_ids": []}),
                                        code=self.fixed_code())  # fmt: skip
            else:
                record = self.records.budget(1, N, code=self.fixed_code())
            source = self.inputs / f"{kind}-{next(self.names)}.json"
            source.write_text(json.dumps(record))
            ran = self.run("summary", "record", kind, "--runs-root", self.root, "--from", source)
        return SimpleNamespace(code=ran.code, out=ran.out, error=ran.err)

    def versions_successor(self) -> None:
        """The versions record of the validation fixation's successor, superseding the first,
        committed alone."""
        name = "docs/study3-fixation-validation-c1-2.json"
        self.commit("the validation fixation's successor", name)
        self.ok("summary", "versions", "--out", SECOND, "--reason", "validation_c1", "--fixation",
                name, "--runs-root", self.root, "--supersedes", FIRST, "--supersedes-reason",
                "cycle_c1")  # fmt: skip
        self.commit("its versions record", SECOND)

    def rerun_validation(self) -> None:
        """The validation's reissued cells written by the campaign writer and launched again
        (each label's next attempt)."""
        out = self.campaign("validation")
        self.ok("summary", "check", "--runs-root", self.root, "--campaign", out)
        self.relaunch(out)

    def claims(self, **claims: str) -> SimpleNamespace:
        """``study3_summary.py cycle`` with a claim per stage (``STAGE=CLAIM``), each with a
        justification."""
        argv = []
        for stage, claim in claims.items():
            argv += [f"{stage}={claim}", f"--justification={stage}=what the fix changes"]
        last = argv.pop()  # given as ``cycle``'s own justification
        return self.cycle(*argv, justification=last.split("=", 1)[1])

    def block(self, seed: int, technical=()) -> list[str]:
        """Block ``seed`` guarded and launched; ``technical``: the arms whose run at the lower
        rate ends technical."""
        assert self.guarded(seed, self.blocks[seed]).code == 0
        rates: dict[str, list[float]] = {}
        for cell in yaml.safe_load(self.blocks[seed].read_text())["cells"]:
            pairs = dict(o.split("=", 1) for o in cell["overrides"])
            rates.setdefault(ARM_OF[pairs["optimizer"]], []).append(float(pairs["optimizer.lr"]))
        return self.launch(self.blocks[seed], technical={(arm, min(rates[arm]))
                                                         for arm in technical})  # fmt: skip


@pytest.fixture
def fixed_route(tmp_path, monkeypatch):
    """``FixedRoute`` built through the validation fixation and its first versions record, the
    cycle module reading the route's checkout."""
    sealed = pytest.MonkeyPatch()
    try:
        world = FixedRoute(tmp_path, monkeypatch)
        world.guard = SealedGuard(sealed, runs_root=world.root)
        world.build()
        sealed.undo()
        monkeypatch.setattr(cycle, "REPO", world.repo)
        yield world
    finally:
        sealed.undo()


def p_misses(arm, rate, seed, candidate) -> dict:
    """The validation at its targets, but P's upper rate below K* on the first seven blocks: P is
    supported in 21 of 28 blocks, and P–F's gate fails."""
    if arm == "P" and math.isclose(rate, START["P"]["u"]) and seed in SEEDS["validation"][1][:7]:
        return {"k": 5.0}
    return at_target(arm, rate, seed, candidate)


@pytest.fixture
def closed_p_route(tmp_path, monkeypatch):
    """``FixedRoute`` with P–F closed by its validation gate (no closure record), the n record of
    A–F, its budget, the supplement and its versions record, block 10400 and block 10401 with
    F's run at its lower rate technical: the state's engineering cycle at the confirmation."""
    sealed = pytest.MonkeyPatch()
    try:
        world = FixedRoute(tmp_path, monkeypatch)
        world.guard = SealedGuard(sealed, runs_root=world.root)
        world.build()
        world.validate(p_misses)
        world.choose()
        world.supplement()
        world.first_block()
        world.block(10401, technical=("F",))
        sealed.undo()
        yield world
    finally:
        sealed.undo()


def meta_path(run_id: str) -> str:
    return f"study3/{run_id}/meta.json"


@pytest.fixture
def cycle_route(tmp_path, monkeypatch):
    """``CycleRoute`` built, every sealed read failing while it is built (``SealedGuard``)."""
    sealed = pytest.MonkeyPatch()
    try:
        world = CycleRoute(tmp_path, monkeypatch)
        world.guard = SealedGuard(sealed, runs_root=world.root)
        world.build()
        sealed.undo()
        yield world
    finally:
        sealed.undo()


def edit_record(route, change, *, rebridge=False) -> None:
    route.edit(change, rebridge=rebridge)


def test_a_cycle_needs_a_claim_with_its_justification_for_every_started_stage(cycle_route):
    cycle_route.write_cycle()
    edit_record(cycle_route, lambda r: r["stages"].pop("search"), rebridge=True)
    assert any("search" in p and "claim" in p for p in cycle_route.cycle_problems())


def test_the_published_cycle_validates_in_its_pre_cycle_view_before_and_after_the_repair(
    cycle_route,
):
    cycle_route.write_cycle()
    found = cycle.read_cycle(cycle_route.root, 1, repo=cycle_route.repo)
    assert found["state"] == "valid" and cycle_route.cycle_problems() == []
    assert cycle_route.mark in found["affected"] and found["claims"] == ROUTE_CLAIMS
    cycle_route.repair()  # qualify.py's lines
    lines = [json.loads(line) for log in sorted((cycle_route.root / "launch").glob("*.jsonl"))
             for line in log.read_text().splitlines()]  # fmt: skip
    superseded = {r["run_id"] for r in lines if r.get("event") == "superseded"}
    assert superseded == cycle_route.affected()
    assert cycle_route.cycle_problems() == []  # the pre-cycle view suppresses its own effects


def test_after_the_repair_the_affected_runs_read_superseded(cycle_route):
    cycle_route.write_cycle()
    cycle_route.repair()  # qualify.py's lines: the runs read superseded now
    assert {e.state for e in cycle_route.endpoints(step="search")
            if e.run_id in cycle_route.affected()} == {"superseded"}  # fmt: skip


@pytest.mark.parametrize("edit", ["omit_one", "add_one", "duplicate_one", "reorder"])
def test_the_lists_equal_their_derivation_exactly(cycle_route, edit):
    cycle_route.write_cycle()
    getattr(cycle_route, edit)(rebridge=True)  # the bridge rewritten to the new sha256
    problems = cycle_route.cycle_problems() + cycle.read_cycle(cycle_route.root, 1)["problems"]
    words = {"omit_one": "not listed", "add_one": "not derived", "duplicate_one": "twice",
             "reorder": "canonical order"}  # fmt: skip
    assert any(words[edit] in p for p in problems), problems


def test_a_bridge_set_must_equal_the_records_not_merely_lie_within_it(cycle_route):
    cycle_route.write_cycle()
    bridge = cycle_route.bridge()
    assert cycle.bridge_problems(cycle_route.root, 1, bridge, repo=cycle_route.repo) == []
    for change in ({"launchers": bridge["launchers"][1:]}, {"held": [{"run_id": "x"}]},
                   {"launchers": [*bridge["launchers"], bridge["launchers"][0]]},
                   {"launchers": bridge["launchers"][::-1]},
                   {"attempts": [{"sha256": "0" * 64}]},
                   {"fix_gpu_code": {"sha256": "0" * 64}},
                   {"old_record": {**bridge["old_record"], "sha256": "0" * 64}}):  # fmt: skip
        assert cycle.bridge_problems(cycle_route.root, 1, {**bridge, **change},
                                     repo=cycle_route.repo)  # fmt: skip


def test_the_record_shows_the_repair_path_and_ignores_time_stamps(cycle_route):
    cycle_route.skew_clock(hours=+6)  # journal stamps later than written_at
    cycle_route.write_cycle()
    found = cycle.read_cycle(cycle_route.root, 1, repo=cycle_route.repo)
    assert cycle_route.cycle_problems() == []
    assert found["repair_record"] == str(cycle_path(cycle_route.root, 1).resolve())


def test_a_rewritten_journal_prefix_invalidates_the_cycle(cycle_route):
    cycle_route.write_cycle()
    cycle_route.rewrite_line("launch/study3-c1-search-round0.jsonl", 2)
    assert any("prefix" in p for p in cycle_route.cycle_problems())


def test_a_record_or_a_bridge_alone_names_the_command_that_writes_the_other(cycle_route):
    assert cycle.read_cycle(cycle_route.root, 1)["state"] == "missing"
    cycle_route.write_cycle()
    bridge = cycle_route.root / bridge_name(1)
    kept = bridge.read_bytes()
    bridge.unlink()
    found = cycle.read_cycle(cycle_route.root, 1, repo=cycle_route.repo)
    assert found["state"] == "invalid"
    assert any("study3_summary.py cycle" in p for p in found["problems"])
    bridge.write_bytes(kept)
    record = cycle_path(cycle_route.root, 1)
    record.rename(record.with_suffix(".kept"))
    found = cycle.read_cycle(cycle_route.root, 1, repo=cycle_route.repo)
    assert found["state"] == "invalid"
    assert any("study3_summary.py cycle" in p for p in found["problems"])
    record.with_suffix(".kept").rename(record)
    old = cycle_route.code["gpu_code_sha256"]
    assert cycle.read_cycle(cycle_route.root, 1, repo=cycle_route.repo,
                            chain_old=old)["state"] == "valid"  # fmt: skip
    found = cycle.read_cycle(cycle_route.root, 1, repo=cycle_route.repo, chain_old="0" * 64)
    assert any("old" in p for p in found["problems"])


def test_every_mark_is_one_of_the_affected_runs(cycle_route):
    """A mark is a run the cycle reruns: a record whose affected runs leave a mark out is a
    problem naming it."""
    cycle_route.write_cycle()
    mark = cycle_route.mark

    def drop(record):
        runs = record["affected"]["runs"]
        runs[:] = [run for run in runs if run["run_id"] != mark]

    cycle_route.edit(drop, rebridge=True)
    assert any(f"the mark {mark}" in p and "affected" in p for p in cycle_route.cycle_problems())


# ------------------------------------------------------------------ restarts, the bridge, refusals


def case_restart(case_dir: Path, when: str, launcher: str | None = None) -> Path:
    """A case restart record as qualify.py restart-case writes it, naming the case's baseline
    attempt 1, written at ``when`` by its machine's clock; ``launcher``: the run it voids."""
    path = case_dir / "restart-1.json"
    path.write_text(json.dumps({
        "restart": 1, "reason": "the Pod stopped", "restarted_at": when,
        "attempts": [{"part": "baseline", "attempt": 1, "state": "recorded",
                      "sha256": sha(case_dir / "baseline-attempt-1.json")}],
        "launcher": None if launcher is None else {"run_id": launcher}}))  # fmt: skip
    return path


def rederived(root: Path, record: dict) -> set[str]:
    """The qualification records a reader re-derives for a published cycle: its pinned prefix
    and restarts, its own bridge read as absent."""
    entries = cycle._requalified(root, record, run_codes={}, settings=REGISTERED)
    return {r["sha256"] for r in cycle._qualification(root, entries)["records"]}


@pytest.mark.parametrize("skew", ["behind", "ahead"])
def test_a_cycle_pins_the_case_restarts_it_counted_and_reads_no_clock(tmp_path, skew):
    """A case restart counts before the cycle exactly when the cycle record pins it (its path
    and sha256): one written after the cycle, its clock behind the writer's, is not counted; one
    written before, its clock ahead, is. The bridge's ``superseded_at`` changes nothing, and a
    pinned restart record that changes is a problem of the record."""
    study, root = materialized_validation_mark(tmp_path)
    case = sorted((root / "qualification" / "c1" / "numerical").glob("s*/*"))[0]
    if skew == "ahead":
        case_restart(case, "2026-10-28T13:00:00+00:00")  # before the cycle, an hour ahead
    record, bridge = derive(study, root, claims=ALL_RERUN)  # written at 12:00
    cycle.publish_cycle(root, 1, record, bridge)
    restart = case / "restart-1.json"
    pinned = ([{"path": restart.relative_to(root).as_posix(), "sha256": sha(restart)}]
              if skew == "ahead" else [])  # fmt: skip
    assert record["affected"]["qualification"]["restarts"] == pinned
    if skew == "behind":
        case_restart(case, "2026-10-28T11:00:00+00:00")  # after the cycle, an hour behind
    records = {r["sha256"] for r in record["affected"]["qualification"]["records"]}
    assert rederived(root, record) == records
    path = root / bridge_name(1)
    path.write_text(json.dumps({**bridge, "superseded_at": "1999-01-01T00:00:00+00:00"},
                               indent=1) + "\n")  # fmt: skip
    assert rederived(root, record) == records
    if skew == "ahead":
        restart.write_text(restart.read_text() + " ")
        assert any("restart-1.json" in p for p in cycle.read_cycle(root, 1)["problems"])


def test_the_pinned_restarts_are_those_the_journal_prefix_cites(tmp_path):
    """Every case restart that a ``superseded`` line inside the pinned prefix cites is pinned,
    and none that only a line beyond the prefix cites."""
    study, root = materialized_validation_mark(tmp_path)
    case = sorted((root / "qualification" / "c1" / "numerical").glob("s*/*"))[0]
    baseline = json.loads((case / "baseline-attempt-1.json").read_text())
    run_id = baseline["launcher"]["identity"]["run_id"]
    restart = case_restart(case, "2026-10-28T11:00:00+00:00", launcher=run_id)
    log = root / "launch" / f"{LOGS['numerical']}.jsonl"
    before = cycle.journal_seen(root)
    (label,) = {json.loads(x)["run"] for x in log.read_text().splitlines()
                if json.loads(x).get("run_id") == run_id}  # fmt: skip
    append_line(log, {"time": WRITTEN, "event": "superseded", "run": label, "run_id": run_id,
                      "experiment": "study3", "config_hash": "0" * 16,
                      "reason": "the Pod stopped", "record": str(restart)})  # fmt: skip
    record, _ = derive(study, root, claims=ALL_RERUN)
    assert [r["path"] for r in record["affected"]["qualification"]["restarts"]] == [
        restart.relative_to(root).as_posix()]  # fmt: skip
    assert cycle._restart_problems(root, record) == []
    unpinned = copy.deepcopy(record)
    unpinned["affected"]["qualification"]["restarts"] = []
    assert any("restart-1.json" in p and "not pinned" in p
               for p in cycle._restart_problems(root, unpinned))  # fmt: skip
    beyond = {**copy.deepcopy(record), "journal": before}  # its line beyond the prefix
    assert any("restart-1.json" in p and "beyond" in p
               for p in cycle._restart_problems(root, beyond))  # fmt: skip


def test_the_bridge_equals_its_derivation_field_by_field(tmp_path):
    """Under a rerun of the qualification the bridge's lists are exactly their derivation:
    ``attempts`` (interrupted ones included), ``cases_requalified``, ``kept``, ``kept_cases`` and
    ``launchers`` (every field but the identity, in canonical order)."""
    study, root = materialized_validation_mark(tmp_path)
    record, bridge = derive(study, root, claims=ALL_RERUN)
    cycle.publish_cycle(root, 1, record, bridge)
    entries = cycle._requalified(root, record, run_codes={}, settings=REGISTERED)
    runs = cycle._stage_runs(root, record)
    assert cycle._bridge_against("w", bridge, entries, runs) == []
    first, *rest = bridge["attempts"]
    forged = {"case": first["case"], "stage": "numerical", "sha256": "0" * 64}
    for change in ({"launchers": bridge["launchers"][::-1]}, {"kept": [forged]},
                   {"kept_cases": [{"case": first["case"], "stage": "numerical", "why": "x"}]},
                   {"attempts": [{**first, "attempt": 9}, *rest]},
                   {"cases_requalified": bridge["cases_requalified"][1:]},
                   {"launchers": [{**bridge["launchers"][0], "config_hash": "x"},
                                  *bridge["launchers"][1:]]}):  # fmt: skip
        assert cycle._bridge_against("w", {**bridge, **change}, entries, runs), change
    reordered = {**bridge, "launchers": bridge["launchers"][::-1]}
    assert any("canonical order" in p for p in cycle.bridge_problems(root, 1, reordered))


def test_a_foreign_supersession_record_refuses_the_cycle_before_anything_is_written(tmp_path):
    """A supersession record of the candidate that is not this bridge (at the bridge's path with
    other content, or under any other number): nothing is written, the cycle record included."""
    study, root = materialized_validation_mark(tmp_path)
    claims = {"search": reuse(), "qualification": reuse(), "validation": rerun()}
    record, bridge = derive(study, root, claims=claims)
    foreign = root / bridge_name(1)
    foreign.write_text(json.dumps({"candidate": 1, "supersession": 1, "old_record": {"path": "x"}}))
    with pytest.raises(cycle.CycleRefused, match="supersede-1.json"):
        cycle.publish_cycle(root, 1, record, bridge)
    assert not cycle_path(root, 1).exists()
    foreign.rename(foreign.with_name("supersede-2.json"))
    with pytest.raises(cycle.CycleRefused, match="supersede-2.json"):
        cycle.publish_cycle(root, 1, record, bridge)
    assert not cycle_path(root, 1).exists() and not foreign.exists()


def test_the_derivation_takes_the_journal_its_records_were_read_with(tmp_path):
    study, root = materialized_validation_mark(tmp_path)
    journal = cycle.journal_seen(root)  # pinned before the records are read
    append_line(root / "launch" / f"{LOGS['validation']}.jsonl", {"time": WRITTEN, "event": "x"})
    with pytest.raises(cycle.CycleRefused, match="read the records again"):
        derive(study, root, claims=ALL_RERUN, journal=journal)


def test_a_launcher_record_there_and_unreadable_is_refused_never_marked_absent(tmp_path):
    """A qualification launcher run's meta.json that is there and cannot be read (a link to
    nothing) is a refusal, as an affected run's is: never an absence marker."""
    study, root = materialized_validation_mark(tmp_path)
    run_id = sorted(e.run_id for e in study.endpoints if e.cell.step == "numerical")[0]
    meta = root / "study3" / run_id / "meta.json"
    meta.unlink()
    meta.symlink_to(tmp_path / "nowhere.json")
    with pytest.raises(cycle.CycleRefused, match=f"{run_id}/meta.json"):
        derive(study, root, claims=ALL_RERUN)


def test_an_arm_the_route_closed_without_a_closure_keeps_the_cycle_valid(tmp_path, monkeypatch):
    """P–F closed by its validation gate (P supported in 21 of 28 blocks), no closure record,
    then a confirmation mark: the writer keeps P's runs as of a closed arm, and the reader
    accepts them, since the route before the cycle (``advance`` over the records before it,
    under its old code) shows P inactive. The same record, with P active before the cycle,
    stays a problem."""
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    seeds = SEEDS["validation"][1][:28]
    study.validation(special={(s, "P", "u"): {"K": 5.0} for s in seeds[:7]})  # P–F fails
    study.n(N=60, arms=("A", "F"))
    study.confirmation(count=3, arms=("A", "F"))
    root = materialize(study, tmp_path, run_of(study, "confirmation", "F", "l", 10401))
    state = study.advance()
    assert (state["stage"], state["step"]["reason"]) == ("confirmation", "engineering_cycle")
    assert state["active"]["arms"] == ["A", "F"]
    claims = {"search": reuse(), "qualification": reuse(), "validation": reuse(),
              "confirmation": rerun(arms=("A", "F"))}  # fmt: skip
    record, bridge = derive(study, root, stage="confirmation", closed_arms=("P",), claims=claims)
    closed = {k["run_id"] for k in record["kept"] if k["why"] == "closed_arm"}
    assert closed == {e.run_id for e in study.endpoints if e.arm == "P"
                      and e.cell.step in ("search", "validation")}  # fmt: skip
    cycle.publish_cycle(root, 1, record, bridge)
    found = cycle.read_cycle(root, 1, repo=root.parent)
    assert found["state"] == "valid", found["problems"]
    before = study.records()  # the records before the cycle, the qualification's reading with them
    monkeypatch.setattr(cycle, "pre_cycle_records", lambda *a, **k: before)
    problems = cycle.cycle_problems(found, runs_root=root, theta0=root, repo=root.parent,
                                    settings=REGISTERED)  # fmt: skip
    assert problems == []
    # P supported in every block: P–F's gate passes, and P is active when the walk stops (at
    # the choice: no n record)
    open_ = replace(before, decisions={1: {k: v for k, v in before.decisions[1].items()
                                           if k != "n-c1"}, 2: before.decisions[2]},
                    endpoints=tuple(
                        replace(e, K=CANDIDATES[1]["targets"]["u"]) if e.arm == "P"
                        and e.cell.step == "validation" and e.lr == START["P"]["u"] else e
                        for e in before.endpoints))  # fmt: skip
    monkeypatch.setattr(cycle, "pre_cycle_records", lambda *a, **k: open_)
    problems = cycle.cycle_problems(found, runs_root=root, theta0=root, repo=root.parent,
                                    settings=REGISTERED)  # fmt: skip
    assert any("kept as of a closed arm" in p and "route before the cycle" in p
               for p in problems)  # fmt: skip


def test_a_run_in_flight_in_the_pinned_prefix_refuses_the_cycle_until_it_ends(
    tmp_path, monkeypatch
):
    """P–F closed by its validation gate, a confirmation mark, and a P run still running when the
    cycle is written (its start in the journal, no end line, no terminal record, its meta.json
    running): its records would change after the cycle (a kept run's evidence is its terminal
    record once there), so the cycle is refused, naming the run. Once the run has ended (its end
    line, its terminal record), the cycle is written and reads valid."""
    study = Study()
    study.ready()
    study.search()
    study.numerical()
    seeds = SEEDS["validation"][1][:28]
    study.validation(special={(s, "P", "u"): {"K": 5.0} for s in seeds[:7]})  # P–F fails
    study.n(N=60, arms=("A", "F"))
    study.confirmation(count=3, arms=("A", "F"))
    root = materialize(study, tmp_path, run_of(study, "confirmation", "F", "l", 10401))
    inflight = run_of(study, "validation", "P", "l", seeds[27])
    run_dir = root / "study3" / inflight
    terminal, meta = run_dir / "terminal.json", run_dir / "meta.json"
    saved = {path: path.read_bytes() for path in (terminal, meta)}
    log = root / "launch" / f"{LOGS['validation']}.jsonl"
    lines = log.read_text().splitlines(keepends=True)
    (end,) = [x for x in lines if json.loads(x)["event"] == "end"
              and json.loads(x)["run_id"] == inflight]  # fmt: skip
    log.write_text("".join(x for x in lines if x != end))  # the launcher has not ended it
    terminal.unlink()
    meta.write_text(json.dumps({**json.loads(saved[meta]), "status": "running"}))
    ended = study.change(inflight, state="pending", K=None, K_harmful=None, steps=None,
                         mmlu_loss_pp=None, factual_loss_pp=None)  # fmt: skip
    study.ends = end_lines(root)
    claims = {"search": reuse(), "qualification": reuse(), "validation": reuse(),
              "confirmation": rerun(arms=("A", "F"))}  # fmt: skip
    with pytest.raises(cycle.CycleRefused, match=f"have not ended.*{inflight}"):
        derive(study, root, stage="confirmation", closed_arms=("P",), claims=claims)
    assert not cycle_path(root, 1).exists()
    append_line(log, json.loads(end))  # the run ends: its end line, its records
    for path, data in saved.items():
        path.write_bytes(data)
    study.drop(inflight)
    study.add(ended)
    study.ends = end_lines(root)
    record, bridge = derive(study, root, stage="confirmation", closed_arms=("P",), claims=claims)
    (kept,) = [k for k in record["kept"] if k.get("run_id") == inflight]
    assert kept["evidence"] == {"path": f"study3/{inflight}/terminal.json", "sha256": sha(terminal)}
    cycle.publish_cycle(root, 1, record, bridge)
    found = cycle.read_cycle(root, 1, repo=root.parent)
    assert found["state"] == "valid", found["problems"]
    before = study.records()
    monkeypatch.setattr(cycle, "pre_cycle_records", lambda *a, **k: before)
    assert cycle.cycle_problems(found, runs_root=root, theta0=root, repo=root.parent,
                                settings=REGISTERED) == []  # fmt: skip


def closed_p_study(tmp_path, closure: str):
    """Candidate 1 through three confirmation blocks of A and F, P closed by the route alone with
    no closure record: by P–F's validation gate (``gate``: P supported in 21 of 28 blocks) or by
    its own qualification's scientific failure (``qualification``: P's upper rate failing); F's
    run at its lower rate on 10401 a technical mark; on files (``materialize``)."""
    study = Study()
    study.ready()
    study.search()
    if closure == "gate":
        study.numerical()
        seeds = SEEDS["validation"][1][:28]
        study.validation(special={(s, "P", "u"): {"K": 5.0} for s in seeds[:7]})
    else:
        study.numerical(states={("P", "u"): FAILING})
        study.validation(arms=("A", "F"))
    study.n(N=60, arms=("A", "F"))
    study.confirmation(count=3, arms=("A", "F"))
    return study, materialize(study, tmp_path, run_of(study, "confirmation", "F", "l", 10401))


def repaired_lines(root: Path, bridge: dict) -> None:
    """The bridge's ``superseded`` lines, as qualify.py's journal repair writes them: one per
    launcher run, in its own campaign log, citing the bridge (beyond the cycle's prefix)."""
    for entry in bridge["launchers"]:
        append_line(root / "launch" / f"{entry['campaign']}.jsonl", {
            "time": "2026-10-29T00:00:00+00:00", "event": "superseded", "run": entry["run"],
            "experiment": entry["experiment"], "run_id": entry["run_id"],
            "config_hash": entry["config_hash"], "record": str(root / bridge_name(1)),
            "reason": "engineering cycle"})  # fmt: skip


@pytest.mark.parametrize("closure", ["gate", "qualification"])
@pytest.mark.parametrize("repaired", [False, True], ids=["published", "repaired"])
def test_an_arm_the_route_alone_closed_keeps_a_cycle_that_reruns_the_qualification_valid(
    tmp_path, monkeypatch, closure, repaired
):
    """P closed by the route alone (P–F's validation gate, or P's own qualification failure),
    no closure record; a confirmation mark; a cycle that reruns the qualification, its bridge
    published (then, ``repaired``, its journal lines written). An accepted bridge voids the
    records it names by itself, from its publication on: read now, A's and F's rates wait for
    their re-qualification, and the walk would stop at the numerical stage with P active. The
    records before the cycle hold no qualification (``pre_cycle_records``), so ``inactive_arms``
    reads it, as it stood before the cycle: the bridge read as absent, the cycle's journal
    prefix. P stays closed, and the cycle stays valid."""
    study, root = closed_p_study(tmp_path, closure)
    state = study.advance()
    assert (state["stage"], state["step"]["reason"]) == ("confirmation", "engineering_cycle")
    assert state["active"]["arms"] == ["A", "F"]
    claims = {"search": reuse(), "qualification": rerun(), "validation": reuse(),
              "confirmation": rerun(arms=("A", "F"))}  # fmt: skip
    record, bridge = derive(study, root, stage="confirmation", closed_arms=("P",), claims=claims)
    assert record["affected"]["qualification"]["records"]  # the bridge voids the old records
    cycle.publish_cycle(root, 1, record, bridge)
    if repaired:
        repaired_lines(root, bridge)
    found = cycle.read_cycle(root, 1, repo=root.parent)
    assert found["state"] == "valid", found["problems"]
    before = study.records()
    calls = []

    def read(runs_root, c, *, expected_code, settings, without=None, prefix=None, restarts=None):
        """The qualification as the records give it, but every rate the bridge voids waiting
        (``incomplete``) while the bridge is not read as absent; P's failing case is kept."""
        calls.append((c, expected_code, without, prefix))
        reading = before.qualification[c]
        if without == bridge_name(c) or not (Path(runs_root) / bridge_name(c)).is_file():
            return reading
        rates = {key: {**rate, "state": rate["state"] if rate["state"] == FAILING else
                       "incomplete"} for key, rate in reading["rates"].items()}  # fmt: skip
        return {**reading, "rates": rates,
                "states": {key: rate["state"] for key, rate in rates.items()}}  # fmt: skip

    monkeypatch.setattr(qr, "read_qualification", read)
    monkeypatch.setattr(
        cycle, "pre_cycle_records", lambda *a, **k: replace(before, qualification={})
    )
    problems = cycle.cycle_problems(found, runs_root=root, theta0=root, repo=root.parent,
                                    settings=REGISTERED)  # fmt: skip
    assert problems == [], problems[:3]
    prefix = {j["path"]: j["lines"] for j in record["journal"]}
    assert (1, OLD, bridge_name(1), prefix) in calls


def test_an_arm_the_route_closed_keeps_the_cycle_valid_after_its_repair_on_files(closed_p_route):
    """On the route's files, read as the readers read them: P–F closed by its validation gate
    with no closure record, a confirmation mark, the cycle rerunning the qualification written by
    study3_summary.py cycle, then qualify.py's real repair. The bridge voids the qualification's
    old records; the view before the cycle reads them as they stood (its bridge absent, its
    journal prefix), so the route before the cycle still shows P closed, and the cycle stays
    valid, with no closed-arm problem."""
    r = closed_p_route
    state = r.state()
    assert (state["stage"], state["step"]["reason"]) == ("confirmation", "engineering_cycle")
    assert state["active"]["arms"] == ["A", "F"]
    r.commit_fix()
    ran = r.claims(search="reuse", qualification="rerun", validation="reuse",
                   confirmation="rerun:A,F")  # fmt: skip
    assert ran.code == 0, ran.error
    record = r.record_()
    closed = {k["run_id"] for k in record["kept"] if k["why"] == "closed_arm"}
    assert closed and closed == {e.run_id for step in ("search", "validation")
                                 for e in r.endpoints(step) if e.arm == "P"}  # fmt: skip
    assert record["affected"]["qualification"]["records"]  # the qualification reruns
    r.repair()
    assert launcher.missing_lines(r.root) == []
    problems = r.cycle_problems()
    assert problems == [], problems[:3]


def test_a_cycle_after_the_fixation_lists_the_stale_records_and_never_reruns_the_search(
    fixed_route,
):
    """A validation mark after the validation fixation: the search is never rerun (search=rerun
    refused); the cycle that reruns the qualification and the validation names the fixation in
    force and lists the records it makes stale, the fixation and the versions record among
    them, and it reads valid."""
    r = fixed_route
    r.validation_mark("F", 10304)
    r.commit_fix()
    refused = r.claims(search="rerun", qualification="rerun", validation="rerun")
    assert refused.code != 0 and "search=rerun" in refused.error
    ran = r.claims(search="reuse", qualification="rerun", validation="rerun")
    assert ran.code == 0, ran.error
    record = r.record_()
    assert record["fixation"]["path"] == "docs/study3-fixation-validation-c1.json"
    names = {Path(entry["path"]).name for entry in record["affected"]["records"]}
    assert names == {"study3-fixation-validation-c1.json", "study3-versions.json"}
    assert r.cycle_problems() == []


def test_successors_follow_the_registered_order(fixed_route):
    """After a cycle that reruns the qualification and the validation, on the route's files:
    (1) the validation fixation's successor and its versions record, (2) the validation rerun,
    (3) the n record (and the budget record), (4) the supplement; a record written out of turn
    is refused with the step the state names, and the fixation's successor is written once."""
    r = fixed_route
    r.validation_mark("F", 10304)
    r.commit_fix()
    assert r.claims(search="reuse", qualification="rerun", validation="rerun").code == 0
    r.repair()
    r.requalify()
    assert r.state()["step"]["record"] == "fixation-validation-c1-2"
    assert "the state's step does not name it" in r.write("record", "n-c1").error  # (3) early
    assert r.write("fixation", "validation").code == 0  # (1)
    assert r.write("fixation", "validation").code != 0  # written once
    r.versions_successor()  # (1) its versions record
    assert r.write("fixation", "confirmation").code != 0  # (4) before (2) and (3)
    state = r.state()
    assert (state["step"]["kind"], state["step"]["campaign"]) == ("campaign", "validation")
    assert state["cycle"][1]["successors"]["docs/study3-fixation-validation-c1.json"] == (
        "docs/study3-fixation-validation-c1-2.json")  # fmt: skip
    r.rerun_validation()  # (2) the reissued cells, real launches
    assert r.write("record", "n-c1").code == 0  # (3) on the validation that counts
    assert r.write("record", "budget-c1").code == 0
    assert r.write("fixation", "confirmation").code == 0  # (4)
    assert r.state()["step"]["campaign"] == "confirmation"


@pytest.fixture
def confirming_route(tmp_path, monkeypatch):
    """``FixedRoute`` through the confirmation's first block (``Route.build_to``), then block
    10401 with F's run at its lower rate technical: the state's engineering cycle at the
    confirmation."""
    sealed = pytest.MonkeyPatch()
    try:
        world = FixedRoute(tmp_path, monkeypatch)
        world.guard = SealedGuard(sealed, runs_root=world.root)
        world.build_to("confirming")
        world.block(10401, technical=("F",))
        sealed.undo()
        monkeypatch.setattr(cycle, "REPO", world.repo)
        yield world
    finally:
        sealed.undo()


def test_a_cycle_during_the_confirmation_cannot_reach_candidate_2_or_a_new_n(confirming_route):
    """Once the confirmation has started: a cycle that reruns the validation is refused (an
    explicit prospective amendment is needed); one that reuses it is written; its
    re-qualification failing scientifically blocks the route (``confirmation_fixed``), never
    moving the candidate; an n successor with another N is refused, one that changes the code
    alone is a valid record, and the writer takes it only when the state's step names it."""
    r = confirming_route
    assert r.state()["step"]["reason"] == "engineering_cycle"
    r.commit_fix()
    reopened = r.claims(search="reuse", qualification="rerun", validation="rerun",
                        confirmation="rerun")  # fmt: skip
    assert reopened.code != 0 and "prospective amendment" in reopened.error
    ran = r.claims(search="reuse", qualification="rerun", validation="reuse",
                   confirmation="rerun")  # fmt: skip
    assert ran.code == 0, ran.error
    names = {Path(e["path"]).name for e in r.record_()["affected"]["records"]}
    assert names == {"study3-fixation-validation-c1.json", "study3-fixation-confirmation-c1.json",
                     "n-c1.json", "budget-c1.json", "study3-versions-2.json"}  # fmt: skip
    r.repair()
    r.requalify(failing=("F",))
    state = r.state()
    assert (state["candidate"], state["step"]["kind"], state["step"]["reason"]) == (
        1, "blocked", "confirmation_fixed")  # fmt: skip
    assert all(event["event"] != "candidate" for event in state["route"])
    n = json.loads(decision_path(r.root, "n-c1").read_text())
    successor = {**n, "kind": "n-c1-2", "code": r.fixed_code(), "supersedes": {
        "path": "study3/decisions/n-c1.json", "sha256": sha(decision_path(r.root, "n-c1")),
        "reason": "cycle_c1", "cycle": {"path": "study3/decisions/cycle-c1.json",
                                        "sha256": sha(cycle_path(r.root, 1))}}}  # fmt: skip
    source = r.inputs / "n-successor.json"
    source.write_text(json.dumps({**successor, "N": N + 6, "cap": cap_of(N + 6)}))
    ran = r.run("summary", "record", "n-c1", "--runs-root", r.root, "--from", source)
    assert ran.code != 0 and "prospective amendment" in ran.err
    source.write_text(json.dumps(successor))
    ran = r.run("summary", "record", "n-c1", "--runs-root", r.root, "--from", source)
    assert ran.code != 0 and "the state's step does not name it" in ran.err  # out of turn
    assert _problems(r.root, "n-c1-2", successor, r.repo) == []  # the code alone changes


def test_the_qualification_before_a_cycle_reads_as_it_stood(qualify, monkeypatch, tmp_path):
    """``read_qualification``'s view before a cycle outside the qualification, on qualify.py's own
    records: after the cycle's bridge, its repair, a recovery round's new launcher run and a part
    rerun by hand under --engineering-cycle, the view (the bridge read as absent, the journal up
    to the cycle's prefix, its pinned case restarts) reads the rates, the void records and the
    inputs as the reading before the cycle did; read now, the bridge voids the old records."""
    world = BridgeWorld(tmp_path)
    world.full(code=QUALIFIED)
    world.validation_run()
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)  # this checkout: the fix
    before = qr.read_qualification(world.root, 1, expected_code=OLD, settings=REGISTERED)
    record, bridge = bridge_records(world)
    cycle.publish_cycle(world.root, 1, record, bridge)
    assert repair(world, bridge["old_record"]["path"], qualify) == 0
    world.launch(ADAMW, commit="HEAD")  # the recovery round's new run
    run_part_by_hand(qualify, world, ADAMW, attempt=2, flag="--engineering-cycle")
    view = {
        "without": bridge_name(1),
        "prefix": {j["path"]: j["lines"] for j in record["journal"]},
        "restarts": [r["path"] for r in record["affected"]["qualification"]["restarts"]],
    }
    then = qr.read_qualification(world.root, 1, expected_code=OLD, settings=REGISTERED, **view)
    assert before["states"] and then["states"] == before["states"]
    assert then["voided"] == before["voided"] == {}
    assert then["inputs"] == before["inputs"]
    assert [r["located"] for rate in then["rates"].values() for r in rate["runs"]] == [
        r["located"] for rate in before["rates"].values() for r in rate["runs"]
    ]
    now = qr.read_qualification(world.root, 1, expected_code=OLD, settings=REGISTERED)
    assert now["voided"] and now["states"] != before["states"]


def test_a_dangling_link_at_the_record_or_the_bridge_is_refused_before_anything_is_written(
    tmp_path,
):
    study, root = materialized_validation_mark(tmp_path)
    claims = {"search": reuse(), "qualification": reuse(), "validation": rerun()}
    record, bridge = derive(study, root, claims=claims)
    for target in (cycle_path(root, 1), root / bridge_name(1)):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(tmp_path / "nowhere.json")  # a link to nothing
        with pytest.raises(cycle.CycleRefused, match="cannot be read"):
            cycle.publish_cycle(root, 1, record, bridge)
        target.unlink()
        assert not cycle_path(root, 1).exists() and not (root / bridge_name(1)).exists()


def test_a_restarts_superseded_line_is_matched_by_its_runs_label_and_experiment(tmp_path):
    """As the launcher finds a case restart's line (``launch.missing_lines``): the
    ``superseded`` line of its launcher run under the run's own label and experiment (its
    start's in the journal); a line of the run id under another label is not it."""
    root = tmp_path / "runs"
    run_id = "20261026T000001Z-adamw-1e-05-s10200-0123abcd-a1"
    first = start(run_id, "2026-10-26T00:00:01+00:00", label="c1-numerical-adamw-1e-05/s10200")
    case = root / "qualification" / "c1" / "numerical" / "s10200" / "adamw-1e-05"
    case.mkdir(parents=True)
    (case / "restart-1.json").write_text(json.dumps({"restart": 1, "launcher": {"run_id": run_id}}))
    for label, missing in (("c1-numerical-adamw-2e-05/s10200", True), (first["run"], False)):
        write_lines(
            root / "launch" / "study3-c1-numerical.jsonl",
            [
                first,
                {
                    "time": "2026-10-26T00:00:02+00:00",
                    "event": "superseded",
                    "run": label,
                    "experiment": "study3",
                    "run_id": run_id,
                    "record": str(case / "restart-1.json"),
                },
            ],
        )
        found = [p for p in cycle.journal_problems(root) if "no superseded line" in p]
        assert bool(found) is missing is bool(launcher.missing_lines(root)), (label, found)


def test_a_cycle_without_a_mark_is_refused_as_such(tmp_path):
    study, root = materialized_validation_mark(tmp_path)
    with pytest.raises(cycle.CycleRefused, match="no mark"):
        derive(study, root, claims=ALL_RERUN, marks=[])


def test_a_closed_arm_rests_only_on_a_closure_whose_line_lies_in_the_prefix():
    journal = [{"path": "launch/a.jsonl", "lines": 2, "sha256": "0" * 64}]
    assert cycle._in_prefix({"path": "launch/a.jsonl", "line": 2}, journal)
    assert not cycle._in_prefix({"path": "launch/a.jsonl", "line": 3}, journal)
    assert not cycle._in_prefix({"path": "launch/b.jsonl", "line": 1}, journal)
    assert not cycle._in_prefix(None, journal)


# ------------------------------------------------------------------ the writer, on files


def written(root: Path) -> dict[str, bytes]:
    """Every file under the qualification's and the decision records' folders, by path."""
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for folder in (root / "qualification", root / "study3" / "decisions")
            for p in sorted(folder.rglob("*")) if p.is_file()}  # fmt: skip


def test_the_cycle_opens_the_new_chain_on_files_and_the_guards_stay(cycle_route):
    """The right to a cycle is judged on the existing chain (the code in force), the fix
    checked on its own: from the fix's checkout, before the cycle, the ordinary state reports
    the chain problem and the campaign writer refuses; the cycle opens the chain at the fix."""
    cycle_route.commit_fix()  # train.py changed: HEAD has a new GPU code
    assert any("end of the chain" in p for p in cycle_route.state()["problems"])
    assert cycle_route.campaigns("search-round0").code != 0  # the campaign writer still refuses
    ran = cycle_route.cycle("search=rerun", justification="search=the fix changes every step")
    assert ran.code == 0 and cycle_path(cycle_route.root, 1).exists(), ran.error
    record = json.loads(cycle_path(cycle_route.root, 1).read_text())
    codes_ = (record["old_code"], record["code"]["gpu_code_sha256"])
    assert codes_ == (cycle_route.old_code, cycle_route.fix_code)
    assert not [p for p in cycle_route.state()["problems"] if "end of the chain" in p]
    lines = ran.out.splitlines()
    assert lines[:2] == [str(cycle_path(cycle_route.root, 1)), str(cycle_route.bridge_path())]
    repair = lines[2]
    assert repair.startswith('project-run "$SRC" python scripts/qualify.py supersede --campaign ')
    assert f"--campaign {cycle_route.campaign_file} --candidate 1 --engineering --case " in repair
    assert f"--record {cycle_route.bridge()['old_record']['path']} " in repair
    assert cycle_route.cycle_problems() == []


@pytest.mark.parametrize("why, setup", [
    ("no cycle", lambda r: None),                                            # no fix committed
    ("recorded already", lambda r: r.commit_fix().qualify_supersession()),   # qualify.py's cycle
    ("campaigns/", lambda r: r.commit_fix().drop_engineering_campaign()),    # no campaign file
    ("not ended", lambda r: r.commit_fix().pending_search_run("F", "l")),    # a run not ended
    ("justification", lambda r: r.commit_fix().blank_justification("search")),  # a blank claim
])  # fmt: skip
def test_cycle_refuses(cycle_route, why, setup):
    setup(cycle_route)
    before = written(cycle_route.root)
    ran = cycle_route.cycle("search=rerun", justification="search=x")
    assert ran.code != 0 and why in ran.error, ran.error
    assert not cycle_path(cycle_route.root, 1).exists()
    assert written(cycle_route.root) == before  # nothing written, the bridge included


def test_a_marked_run_without_meta_json_carries_an_absence_marker_and_is_not_refused(cycle_route):
    """A mark with no meta.json has no GPU code: a mark of the code in force (only a mark of a
    known other code is refused), its file an absence marker, its journal start binding it."""
    cycle_route.commit_fix()
    cycle_route.remove_meta(cycle_route.mark)
    ran = cycle_route.cycle("search=rerun", justification="search=x")
    assert ran.code == 0, ran.error
    runs = cycle_route.record()["affected"]["runs"]
    (run,) = [r for r in runs if r["run_id"] == cycle_route.mark]
    assert {"path": meta_path(cycle_route.mark), "absent": True} in run["records"]
    assert run["gpu_code"] is None and run["start"]["line_sha256"]
    assert cycle_route.cycle_problems() == []


def test_an_unreadable_file_keeps_its_byte_hash(cycle_route):
    cycle_route.commit_fix()
    cycle_route.garble("terminal.json")  # present, not JSON
    ran = cycle_route.cycle("search=rerun", justification="search=x")
    assert ran.code == 0, ran.error
    assert any(e.get("sha256") == cycle_route.garbled_sha
               for r in cycle_route.record()["affected"]["runs"] for e in r["records"])  # fmt: skip


def test_the_cycle_writes_only_a_missing_bridge_when_run_again(cycle_route):
    cycle_route.commit_fix()
    assert cycle_route.cycle("search=rerun", justification="search=x").code == 0
    first = cycle_path(cycle_route.root, 1).read_bytes()
    bridge = cycle_route.bridge_path().read_bytes()
    cycle_route.bridge_path().unlink()
    ran = cycle_route.cycle("search=rerun", justification="search=x")
    assert ran.code == 0, ran.error
    assert cycle_path(cycle_route.root, 1).read_bytes() == first
    assert json.loads(cycle_route.bridge_path().read_bytes()) == json.loads(bridge)
    assert "recorded already" in cycle_route.cycle("search=rerun",
                                                   justification="search=other").error  # fmt: skip
    cycle_route.bridge_path().unlink()  # run again with other claims over the record: refused
    ran = cycle_route.cycle("search=rerun", justification="search=other")
    assert "recorded already" in ran.error and not cycle_route.bridge_path().exists()


def test_a_torn_line_refuses_the_cycle_by_its_line(cycle_route):
    """A torn line of the launch journal that no resolution names: the writer refuses naming it,
    before anything that reads the journal short of it (the decision records' checks, so the
    chain's start), and writes nothing."""
    cycle_route.commit_fix()
    before = written(cycle_route.root)
    log = sorted((cycle_route.root / "launch").glob("*.jsonl"))[-1]
    with log.open("ab") as handle:
        handle.write(b'{"torn\n')
    ran = cycle_route.cycle("search=rerun", justification="search=x")
    assert ran.code != 0 and "a torn line that no resolution names" in ran.error, ran.error
    assert f"{log.relative_to(cycle_route.root).as_posix()} line " in ran.error
    assert "has no start" not in ran.error
    assert written(cycle_route.root) == before


def test_dry_run_writes_nothing(cycle_route):
    cycle_route.commit_fix()
    ran = cycle_route.cycle("search=rerun", "--dry-run", justification="search=x")
    assert ran.code == 0 and '"kind": "cycle-c1"' in ran.out, ran.error
    assert not cycle_path(cycle_route.root, 1).exists()
    assert not cycle_route.bridge_path().exists()


# ------------------------------------------------------------------ qualify.py on the bridge


def run_main(qualify, argv) -> int | str:
    """qualify.py's ``main`` in process: 0 when it returns, else the code it exits with, or its
    message when that is text."""
    try:
        qualify.main([str(a) for a in argv])
    except SystemExit as exc:
        return 0 if exc.code is None else exc.code
    return 0


def case_args(world, case) -> list[str]:
    return ["--campaign", str(world.path), "--candidate", "1", "--case", case.name,
            "--runs-root", str(world.root)]  # fmt: skip


def superseded_lines(root: Path) -> list[dict]:
    """The journal's ``superseded`` records, every log."""
    records = [json.loads(x) for log in sorted((root / "launch").glob("*.jsonl"))
               for x in log.read_text().splitlines()]  # fmt: skip
    return [r for r in records if r.get("event") == "superseded"]


def repair(world, path, qualify, **over):
    """qualify.py's journal repair of the cycle, as the writer prints it: ``supersede`` on the
    engineering campaign with the cycle record as ``--record``."""
    eng = world.engineering_campaign()  # a valid engineering campaign file
    return run_main(qualify, ["supersede", "--campaign", str(eng), "--candidate", "1",
                              "--engineering", "--case", over.get("case", ENG_CASE), "--record",
                              str(path), "--diagnosis", "d", "--fix", "f", "--runs-root",
                              str(world.root)])  # fmt: skip


def repaired_bridge_world(qualify, monkeypatch, tmp_path):
    """``bridge_world`` after the repair: the bridge's journal lines written."""
    world, bridge_path, bridge = bridge_world(qualify, monkeypatch, tmp_path)
    assert repair(world, bridge["old_record"]["path"], qualify) == 0
    return world, bridge_path, bridge


def run_part_by_hand(qualify, world, case, *, attempt: int, flag: str) -> dict:
    """``qualify.py part --attempt N`` of the case's baseline, by hand with ``flag``: the
    harness (``run_part``) publishes a record carrying the ``engineering_cycle`` argument
    qualify.py passes it, and no model runs. Asserts the arguments; returns the record."""
    calls = []

    def run_part(settings, env, found, run_dir, out, *, part, attempt, checkpoints,
                 diagnostic_steps=(), infrastructure=None, engineering_cycle=None):  # fmt: skip
        calls.append((found.name, part, attempt, infrastructure))
        record = {"part": part, "attempt": attempt, "case": found.record(), "status": "failed",
                  "kind": "technical", "reasons": ["run by hand"],
                  "engineering_cycle": engineering_cycle}  # fmt: skip
        qualify.q.publish(out / found.attempt_path(part, attempt), record)
        return record

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(qualify.runtime, "configure_numerics", lambda cfg, device: {})
        patch.setattr(qualify.runtime, "environment", lambda cfg, device: object())
        patch.setattr(qualify.q, "run_part", run_part)
        code = run_main(qualify, ["part", *case_args(world, case), "--part", "baseline",
                                  "--attempt", attempt, flag, "--run-dir",
                                  world.path.parent / "run", "--checkpoints",
                                  world.path.parent / "h", "--device", "cpu"])  # fmt: skip
    assert code == 0, code
    assert calls == [(case.name, "baseline", attempt, None)]
    return json.loads((world.folder(case) / f"baseline-attempt-{attempt}.json").read_text())


def test_qualify_writes_the_bridges_journal_lines_and_the_launcher_waits_before(
    qualify, monkeypatch, tmp_path
):
    """Published, the bridge needs its journal lines: until they are written every qualify.py
    command and the launcher refuse (exit 12, ``TornJournalError``: a launcher started between
    the bridge and its repair fails closed); qualify.py's own repair, run as the writer prints
    it, writes exactly the bridge's ``superseded`` lines, each citing the bridge."""
    world, bridge_path, bridge = bridge_world(qualify, monkeypatch, tmp_path)
    assert run_main(qualify, ["complete", *case_args(world, ADAMW)]) == 12
    with pytest.raises(launcher.TornJournalError):
        launcher.run_campaign(world.name, world.runs, world.root, dry_run=True)
    assert repair(world, bridge["old_record"]["path"], qualify) == 0
    assert launcher.missing_lines(world.root) == []
    lines = superseded_lines(world.root)
    assert {r["run_id"] for r in lines} == {e["run_id"] for e in bridge["launchers"]} >= {VAL_RUN}
    assert all(r["record"] == str(bridge_path) for r in lines)


def test_qualify_requalifies_the_voided_cases_under_the_fix(qualify, monkeypatch, tmp_path):
    """After the repair qualify.py's documented re-qualification runs unchanged: a voided case
    is not complete (3, never stale: 10), the driver stops for its rerun (8), the rerun by hand
    carries --engineering-cycle and names the bridge, and qualify.py refuses a second cycle."""
    world, bridge_path, bridge = repaired_bridge_world(qualify, monkeypatch, tmp_path)
    assert run_main(qualify, ["complete", *case_args(world, ADAMW)]) == 3  # not 10 (stale)
    resume = ["part", *case_args(world, ADAMW), "--part", "baseline", "--resume", "--run-dir", "x",
              "--checkpoints", "y"]  # fmt: skip
    assert run_main(qualify, resume) == 8  # voided: its next attempt by hand
    world.launch(ADAMW, commit="HEAD")  # the recovery round's new run
    record = run_part_by_hand(qualify, world, ADAMW, attempt=2, flag="--engineering-cycle")
    previous = [{"attempt": 1, "state": "recorded"}]
    assert record["engineering_cycle"] == {"path": str(bridge_path), "sha256": sha(bridge_path),
                                           "previous": previous}  # fmt: skip
    second = ["supersede", *case_args(world, ADAMW), "--record",
              str(world.folder(ADAMW) / "baseline-attempt-2.json"), "--diagnosis", "d", "--fix",
              "f"]  # fmt: skip
    refused = run_main(qualify, second)  # one cycle per candidate: qualify.py refuses its own
    assert "its engineering cycle is recorded already" in refused


def test_the_repair_matches_the_bridge_only_by_its_cycle_records_path_and_sha256(
    qualify, monkeypatch, tmp_path, capsys
):
    """qualify.py's repair takes the bridge only by its cycle record's path and the sha256 of its
    bytes: a cycle record whose bytes changed is no repair, and neither is a byte-identical copy
    of it at another path. Before the journal lines are written, nothing matches the journal's
    missing lines, so qualify.py refuses as the journal is not whole (exit 12) and writes no
    line; after them, the same call is a second cycle, refused."""
    world, _, bridge = bridge_world(qualify, monkeypatch, tmp_path)
    elsewhere = tmp_path / "elsewhere" / "cycle-c1.json"
    elsewhere.parent.mkdir()
    shutil.copy(cycle_path(world.root, 1), elsewhere)  # the same bytes, another path
    capsys.readouterr()
    assert repair(world, str(elsewhere), qualify) == 12  # TornJournalError
    assert "the launch journal is not whole" in capsys.readouterr().err.lower()
    assert launcher.missing_lines(world.root) != [] and superseded_lines(world.root) == []
    touch_cycle_record(world)  # its bytes change
    capsys.readouterr()
    assert repair(world, bridge["old_record"]["path"], qualify) == 12  # TornJournalError
    assert "the launch journal is not whole" in capsys.readouterr().err.lower()
    assert launcher.missing_lines(world.root) != [] and superseded_lines(world.root) == []
    restore(world)
    assert repair(world, bridge["old_record"]["path"], qualify) == 0  # the record as published
    touch_cycle_record(world)
    refused = repair(world, bridge["old_record"]["path"], qualify)
    assert "its engineering cycle is recorded already" in str(refused)
