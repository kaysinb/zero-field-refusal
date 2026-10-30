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

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from test_qualification import CODE
from test_qualification_records import (
    ADAMW,
    ENGINEERING,
    ENGINEERING_HEADER,
    REGISTERED,
    SCIENTIFIC_CASE,
    beyond,
)
from test_qualification_records import World as QualificationWorld
from test_qualification_records import qualify as qualify  # noqa: F401  (the fixture)
from test_qualify import ADAMW_LR, FIX, codes, load_script
from test_study3_campaigns import START, run_config
from test_study3_package2 import REAL_IDENTITY, Route, SealedGuard, round0
from test_study3_progress import OPT, Study, ep, run_of
from test_study3_versions import git

from zero_field_refusal import code_identity
from zero_field_refusal.analysis import qualification_records as qr
from zero_field_refusal.analysis import study3_cycle as cycle
from zero_field_refusal.analysis.study3_decisions import (
    bridge_name,
    cycle_path,
    end_lines,
    serialized,
    start_lines,
)
from zero_field_refusal.analysis.study3_progress import _bound
from zero_field_refusal.analysis.study3_records import COUNTING, GROSS_COLLAPSE, EndpointReader
from zero_field_refusal.config import config_hash


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


def test_r9_names_an_unclassified_qualification_run_it_cannot_settle(qualify, monkeypatch,
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
        "validation": "study3-c1-validation"}  # fmt: skip


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
    root = tmp_path / "runs"
    study = Study()
    study.through_validation()
    if failed is not None:
        study.change(run_of(study, "validation", failed[0], "u", failed[1]), state="unusable",
                     unusable_reason=GROSS_COLLAPSE, mmlu_loss_pp=20.0)  # fmt: skip
    mark = run_of(study, "validation", arm, "l", 10305)
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
    return study, root


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
              "host": "the researcher's machine"}  # fmt: skip
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
    assert record["affected"]["qualification"] == {"records": [], "launchers": []}
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


def test_r9_refuses_a_qualification_run_the_readers_cannot_settle(tmp_path):
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

    def build(self) -> None:
        assert self.step()["record"] == "engineering"
        self.engineering = self.campaign("engineering")
        self.launch(self.engineering)
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

    # -- the cycle, its repair and the edits

    def write_cycle(self) -> None:
        """A fix committed, then the cycle derived from the records before it and published."""
        train = self.repo / "src" / "zero_field_refusal" / "train.py"
        train.write_text("x = 2\n")
        self.commit("the fix")
        head = git(self.repo, "rev-parse", "HEAD")
        identity = REAL_IDENTITY(head, self.repo)
        records = self.pre_cycle()
        marks = [e.run_id for e in records.endpoints
                 if e.cell.step == "search" and e.state == "technical"]  # fmt: skip
        record, bridge = cycle.derive_cycle(
            self.root, 1, records=records, marks=marks, stage="search", closed_arms=(),
            old_code=self.code["gpu_code_sha256"], fixation=None, claims=ROUTE_CLAIMS,
            diagnosis=DIAGNOSIS, fix=FIX_TEXT,
            code={"commit": head, "gpu_code_sha256": identity["sha256"]}, fix_identity=identity,
            settings=REGISTERED, repo=self.repo, written_at=self.written_at,
            host="host")  # fmt: skip
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
