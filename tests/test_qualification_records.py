"""Study 3's numerical qualification read on the analysis side (analysis/qualification_records.py):
each rate's state recomputed from the qualification's records, verdict.json compared as a cache
(docs/study3-preregistration.md, "Numerical qualification", "Failure routes", "Voiding and
supersession", "Amendment 2").

Each world is a qualification of candidate 1 in the formats of scripts/qualify.py: a campaign
with its study-3 header, the launch journal, settled launcher runs and the part records, the
restarts and the engineering cycle written by the script itself. The script's own ``verdict``
runs on the same world, and the recomputation must give its rate verdicts, case verdicts and
attempt listing. Nothing here runs a model.
"""

import ast
import builtins
import copy
import dataclasses
import hashlib
import io
import json
import pathlib
import shutil
from pathlib import Path

import pytest
from test_qualification import CODE, IDENTITY, control_record, synthetic
from test_qualify import (
    ADAMW_LR,
    FA_LR,
    FIX,
    RESTART,
    cell,
    codes,
    load_script,
    settled_launcher,
    trajectory_result,
)

from zero_field_refusal import code_identity
from zero_field_refusal import qualification as q
from zero_field_refusal.analysis import qualification_records as qr
from zero_field_refusal.config import load_config
from zero_field_refusal.launch import load_campaign

REGISTERED = load_config("experiments/qualification")
SEEDS = (10200, 10201, 10202)
HEADER = {"study": 3, "candidate": 1, "step": "numerical", "round": None}
FA_KEY, ADAMW_KEY = f"adam_frozen-{FA_LR:g}", f"adamw-{ADAMW_LR:g}"
BOTH = ((("adam_frozen", FA_LR), SEEDS), (("adamw", ADAMW_LR), SEEDS))


@pytest.fixture
def qualify(monkeypatch, tmp_path):
    """scripts/qualify.py, from a clean checkout whose GPU code is the records' (``CODE``)."""
    module = load_script()
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(module, "git_state", lambda *a, **k: {"commit": "c" * 40, "dirty": False})
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(CODE))
    return module


class World:
    """Candidate 1's qualification campaign (with its header), its journal, launcher runs and
    part records under ``tmp_path/runs``."""

    def __init__(self, tmp_path, cells=BOTH, *, name="study3-c1-numerical", header=HEADER):
        self.root = tmp_path / "runs"
        self.out = self.root / "qualification"
        self.endpoint = tmp_path / "ckpt"
        text = f"name: {name}\nheader: {json.dumps(header)}\ncells:\n"
        for (optimizer, lr), seeds in cells:
            text += cell(f"{optimizer}-{lr:g}", optimizer, lr, seeds, self.endpoint)
        self.path = tmp_path / f"{name}.yaml"
        self.path.write_text(text)
        self.name, self.runs = load_campaign(self.path)
        self.header = header
        self.attempts: dict[str, int] = {}

    def run(self, case):
        return next(r for r in self.runs if r.label == f"{case.optimizer}-{case.lr:g}/s{case.seed}")

    def log(self, record):
        path = self.root / "launch" / f"{self.name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"time": "2026-10-26T00:00:00+00:00", **record}) + "\n")

    def launch(self, case, **settled):
        """One launcher attempt of the case's run, as ``launch.py --only LABEL --max-attempts 1``
        logs and leaves it (settled); its identity as the records name it."""
        run = self.run(case)
        n = self.attempts[run.label] = self.attempts.get(run.label, 0) + 1
        run_id = f"r{case.seed}-{case.optimizer}-a{n}"
        common = {"run": run.label, "attempt": n, "run_id": run_id, "only": run.label}
        self.log({"event": "start", **common, "experiment": run.experiment,
                  "config_hash": run.config_hash, "command": run.command()[1:],
                  "header": self.header, "max_attempts": 1})  # fmt: skip
        run_dir = settled_launcher(self.root, run, run_id, **settled)
        if q.reads_v0(case.optimizer):  # an FA run records v₀'s hash
            meta = json.loads((run_dir / "meta.json").read_text())
            meta["inputs"]["v0_sha256"] = "v" * 64
            (run_dir / "meta.json").write_text(json.dumps(meta))
        self.log({"event": "end", **common, "returncode": 0, "outcome": "done", "problem": None})
        return q.launcher_identity(run_dir)

    def publish(self, case, part, record, attempt=1):
        q.publish(self.out / case.attempt_path(part, attempt), record)

    def records(self, case, identity, *, code=CODE, change=None):
        baseline, rest = synthetic(case, identity=identity)
        for record in (baseline, rest):
            record["provenance"]["gpu_code"] = code
        if change is not None:
            change(baseline, rest)
        rest["baseline_record"]["sha256"] = q.record_sha256(baseline)
        return baseline, rest

    def qualified(self, case, identity, *, attempt=1, code=CODE, change=None, control=True):
        """A case's parts recorded (qualified unless ``change``), and the control on the first
        seed of an FA rate."""
        baseline, rest = self.records(case, identity, code=code, change=change)
        self.publish(case, "baseline", baseline, attempt)
        self.publish(case, "rest", rest, attempt)
        if control and case.optimizer == "adam_frozen" and case.seed == SEEDS[0]:
            self.publish(case, "control", control_record(case=case, identity=identity, code=code),
                         attempt)  # fmt: skip

    def full(self, cells=BOTH, *, changes=None, skip=(), **kwargs):
        """Every case of ``cells`` launched and recorded, qualified unless ``changes`` names
        it; ``skip``: cases left alone."""
        for (optimizer, lr), seeds in cells:
            for seed in seeds:
                case = qcase(seed, optimizer, lr)
                if case not in skip:
                    change = (changes or {}).get(case)
                    self.qualified(case, self.launch(case), change=change, **kwargs)

    def folder(self, case):
        return self.out / case.attempt_path("baseline", 1).rsplit("/", 1)[0]

    def verdict(self, qualify):
        qualify.main(["verdict", "--campaign", str(self.path), "--candidate", "1",
                      "--runs-root", str(self.root)])  # fmt: skip
        return json.loads((self.out / "c1" / "verdict.json").read_text())

    def read(self, expected_code=CODE["sha256"], settings=REGISTERED):
        return qr.read_qualification(self.root, 1, expected_code=expected_code, settings=settings)

    def command(self, qualify, *args):
        qualify.main([args[0], "--campaign", str(self.path), "--candidate", "1",
                      "--runs-root", str(self.root), *args[1:]])  # fmt: skip


def qcase(seed, optimizer, lr):
    return q.QualificationCase(seed, optimizer, lr, 1, False)


def beyond(part, name, d=0.07):
    """A change of a case's records: D₂₅ of ``name`` in its ``part`` record, beyond 0.06."""

    def change(baseline, rest):
        {"baseline": baseline, "rest": rest}[part]["comparisons"][name]["D_twin"] = d

    return change


FA, ADAMW = qcase(10200, "adam_frozen", FA_LR), qcase(10200, "adamw", ADAMW_LR)


def plain(value):
    return json.loads(json.dumps(value, default=str))


def assert_as_the_verdict(reading, cached):
    """The recomputation is the script's verdict: every rate verdict, every case verdict, the
    superseded records, the cases of no rate and every attempt as listed; no problem."""
    assert reading["problems"] == [], reading["problems"]
    assert reading["cache"]["agrees"] is True
    got = plain(reading)
    assert {key: got["rates"][key]["verdict"] for key in cached["rates"]} == cached["rates"]
    assert got["cases"] == cached["cases"]
    assert got["superseded"] == cached["superseded"]
    assert got["unassigned"] == cached["unassigned"]
    listed = {f["case"]: {p: v["attempts"] for p, v in f["parts"].items() if v["attempts"]}
              for rate in got["rates"].values() for f in rate["cases"]}  # fmt: skip
    assert listed == cached["attempts"]


# ------------------------------------------------------------------ the states


def test_a_qualified_rate(qualify, tmp_path) -> None:
    """Three seeds with all 13 comparisons within the bound, the FA control accepted, every
    launcher run located: both rates qualified, as the script's verdict has them."""
    world = World(tmp_path)
    world.full()
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: "qualified", ADAMW_KEY: "qualified"}
    fa = qr.rate_of(reading, "F", FA_LR)
    assert fa is reading["rates"][FA_KEY] and qr.rate_of(reading, "adam_frozen", FA_LR) is fa
    assert (fa["arm"], fa["optimizer"], fa["lr"]) == ("F", "adam_frozen", FA_LR)
    assert qr.rate_of(reading, "P", FA_LR) is None and qr.rate_of(reading, "F", 1e-5) is None
    assert fa["verdict"]["counted"] == list(SEEDS)
    assert fa["control"]["verdict"]["state"] == "accepted" and fa["control"]["attempt"] == 1
    assert reading["rates"][ADAMW_KEY]["control"] is None
    assert [c["case"] for c in fa["cases"]] == [f"s{s}/{FA_KEY}" for s in SEEDS]
    for case in fa["cases"]:
        assert len(case["verdict"]["comparisons"]) == 13
        assert all(c["within"] for c in case["verdict"]["comparisons"].values())
        control = 1 if case["seed"] == SEEDS[0] else None
        counts = {p: v["counts"] for p, v in case["parts"].items()}
        assert counts == {"baseline": 1, "rest": 1, "control": control}
    assert [r["located"] for r in fa["runs"]] == [f"r{s}-adam_frozen-a1" for s in SEEDS]
    assert set(fa["launchers"]) == set(SEEDS) and all(fa["launchers"].values())
    assert reading["uncached"] == [] and reading["held"] == []


def test_a_failing_rate(qualify, tmp_path) -> None:
    """A rest interrupted after its reference beyond 0.01·K* (the trajectory results published,
    the part's record not): the result counts and the rate fails."""
    world = World(tmp_path)
    stopped = qcase(10201, "adamw", ADAMW_LR)
    for (optimizer, lr), seeds in BOTH:
        for seed in seeds:
            case = qcase(seed, optimizer, lr)
            identity = world.launch(case)
            if case != stopped:
                world.qualified(case, identity)
                continue
            baseline, rest = world.records(case, identity)
            rest["comparisons"]["baseline~reference"]["D_twin"] = 0.5  # beyond 0.06
            world.publish(case, "baseline", baseline)
            q.publish(world.out / (case.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
            for name in ("repeat", "reference"):
                q.publish(world.out / case.trajectory_path(1, name),
                          trajectory_result(case, rest, name))  # fmt: skip
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: "qualified", ADAMW_KEY: "failing"}
    (case,) = [c for c in reading["rates"][ADAMW_KEY]["cases"] if c["seed"] == 10201]
    assert case["parts"]["rest"]["counts"] == 1 and case["parts"]["rest"]["from_trajectories"]
    assert [a["state"] for a in case["parts"]["rest"]["attempts"]] == ["interrupted"]
    assert case["verdict"]["kind"] == "scientific"
    assert any(r.startswith("baseline~reference: D₂₅ 0.5") for r in case["verdict"]["reasons"])


def test_a_technical_rate(qualify, tmp_path) -> None:
    """The unshifted repeat beyond 0.01·K* against the harness baseline: reproducibility, a
    technical case, and the rate not qualified."""
    world = World(tmp_path)
    world.full(changes={ADAMW: beyond("rest", "baseline~repeat")})
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: "qualified", ADAMW_KEY: "not_qualified"}
    verdict = reading["rates"][ADAMW_KEY]["verdict"]
    assert any("baseline~repeat: D₂₅ 0.07" in r for r in verdict["reasons"])


def test_a_rate_whose_launcher_run_is_not_located_is_at_most_incomplete(qualify, tmp_path):
    """The launcher run of a seed gone from the runs root, or two settled runs of its config:
    none is located, and the rate is at most incomplete, its cases all qualified. A run whose
    status was not finalized but whose terminal record is published is located."""
    world = World(tmp_path)
    world.full()
    meta_path = world.root / "study3" / "r10200-adamw-a1" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({**meta, "status": "running"}))
    shutil.rmtree(world.root / "study3" / "r10202-adamw-a1")
    shutil.copytree(world.root / "study3" / "r10201-adam_frozen-a1",
                    world.root / "study3" / "r10201-adam_frozen-copy")  # fmt: skip
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: "incomplete", ADAMW_KEY: "incomplete"}
    assert reading["rates"][ADAMW_KEY]["launchers"][10202] is None
    assert reading["rates"][FA_KEY]["launchers"][10201] is None
    assert reading["rates"][ADAMW_KEY]["launchers"][10200]["run_id"] == "r10200-adamw-a1"
    assert reading["rates"][ADAMW_KEY]["verdict"]["counted"] == list(SEEDS)
    assert "the launcher run of seed 10202 is not located" in (
        reading["rates"][ADAMW_KEY]["verdict"]["reasons"])  # fmt: skip


def test_a_record_under_a_later_attempt_counts_against_its_rate(qualify, tmp_path) -> None:
    """A recorded attempt that no restart or supersession names, under a later recorded one
    (which the commands never write): listed as superseded, never hidden, and counted against
    its rate as a case of its own; a control so superseded blocks its rate, failing when it is
    scientific."""
    world = World(tmp_path)
    world.full()
    for part in ("baseline", "rest"):
        record = json.loads((world.folder(ADAMW) / f"{part}-attempt-1.json").read_text())
        world.publish(ADAMW, part, record, attempt=2)
    control = json.loads((world.folder(FA) / "control-attempt-1.json").read_text())
    world.publish(FA, "control", control, attempt=2)
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: "not_qualified", ADAMW_KEY: "not_qualified"}
    shown = [(e["case"], e["part"], e["attempt"], e["restart"], e["supersession"])
             for e in reading["superseded"]]  # fmt: skip
    assert shown == [(FA.name, "control", 1, None, None), (ADAMW.name, "baseline", 1, None, None),
                     (ADAMW.name, "rest", 1, None, None)]  # fmt: skip
    fa = reading["rates"][FA_KEY]
    assert [e["why"] for e in fa["verdict"]["not_counted"]] == ["a superseded same-path control"]
    assert fa["control"]["attempt"] == 2
    (case,) = [c for c in reading["rates"][ADAMW_KEY]["cases"] if c["seed"] == ADAMW.seed]
    assert {p: v["counts"] for p, v in case["parts"].items()} == {
        "baseline": 2, "rest": 2, "control": None}  # fmt: skip
    science = {**control, "status": "not_run", "kind": "scientific", "reasons": ["instability"]}
    world.publish(FA, "control", science, attempt=1)
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"][FA_KEY] == "failing"


def test_a_missing_control_makes_an_fa_rate_technical(qualify, tmp_path) -> None:
    """Amendment 2: an FA rate whose same-path control is missing is not qualified (technical),
    its three cases qualified; an AdamW rate needs none."""
    world = World(tmp_path)
    world.full(control=False)
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: "not_qualified", ADAMW_KEY: "qualified"}
    fa = reading["rates"][FA_KEY]
    assert fa["verdict"]["counted"] == list(SEEDS)
    assert fa["control"] == {"folder": str(world.folder(FA)), "attempt": None, "verdict": None}
    assert fa["verdict"]["control"]["reasons"] == ["no same-path control record"]
    assert fa["verdict"]["control"]["kind"] == "technical"


# ------------------------------------------------------------------ voiding and supersession


def test_a_restart_voids_an_interrupted_attempt(qualify, monkeypatch, tmp_path) -> None:
    """The Pod stopped during the rest: ``restart-case`` names the case's attempts and
    supersedes its launcher run; a new run and new attempts re-qualify it. The named records are
    voided with the restart's reason, listed, and neither count nor block."""
    world = World(tmp_path)
    world.full(skip=(FA,))
    identity = world.launch(FA)
    baseline, _ = world.records(FA, identity)
    world.publish(FA, "baseline", baseline)
    world.publish(FA, "control", control_record(case=FA, identity=identity))
    q.publish(world.out / (FA.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    world.command(qualify, "restart-case", "--case", FA.name, "--infrastructure", RESTART)
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    fa = reading["rates"][FA_KEY]  # the named control is no longer the rate's
    assert fa["state"] == "not_qualified" and fa["control"]["verdict"] is None
    assert fa["verdict"]["control"]["reasons"] == ["no same-path control record"]
    world.qualified(FA, world.launch(FA), attempt=2)  # the recovery round's run and attempts
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: "qualified", ADAMW_KEY: "qualified"}
    fa = reading["rates"][FA_KEY]
    assert sorted(v["case"] for v in fa["verdict"]["voided"]) == ["control", FA.name]
    assert {v["reason"] for v in fa["verdict"]["voided"]} == {RESTART}
    (case,) = [c for c in fa["cases"] if c["seed"] == FA.seed]
    shown = {p: [(a["attempt"], a["state"], a["restart"]) for a in v["attempts"]]
             for p, v in case["parts"].items()}  # fmt: skip
    assert shown == {"baseline": [(1, "recorded", RESTART), (2, "recorded", None)],
                     "rest": [(1, "interrupted", None), (2, "recorded", None)],
                     "control": [(1, "recorded", RESTART), (2, "recorded", None)]}  # fmt: skip
    assert {p: v["counts"] for p, v in case["parts"].items()} == {
        "baseline": 2, "rest": 2, "control": 2}  # fmt: skip
    assert [(e["part"], e["attempt"], e["restart"]) for e in case["superseded"]] == [
        ("baseline", 1, RESTART), ("control", 1, RESTART)]  # fmt: skip
    (restart,) = reading["restarts"]
    assert (restart["case"], restart["reason"], restart["launcher_run_id"]) == (
        FA.name, RESTART, "r10200-adam_frozen-a1")  # fmt: skip
    assert restart["attempts"] == [("baseline", 1, "recorded"), ("rest", 1, "interrupted"),
                                   ("control", 1, "recorded")]  # fmt: skip
    (run,) = [r for r in fa["runs"] if r["seed"] == FA.seed]
    assert run["run_ids"] == ["r10200-adam_frozen-a1", "r10200-adam_frozen-a2"]
    assert run["located"] == "r10200-adam_frozen-a2"
    assert run["superseded"] == {"r10200-adam_frozen-a1": RESTART}


def test_an_engineering_cycle_voids_old_code_and_keeps_a_scientific_case(
    qualify, monkeypatch, tmp_path
) -> None:
    """A technical AdamW baseline opens the candidate's one cycle; FA's case on 10201 failed
    scientifically under the old code. The cycle voids every old-code record but that case's,
    which it keeps whole, and supersedes the old-code launcher runs but its run. Re-qualified
    under the fix, AdamW qualifies; FA keeps failing."""
    world = World(tmp_path)
    kept = qcase(10201, "adam_frozen", FA_LR)
    world.full(skip=(ADAMW,), changes={kept: beyond("rest", "baseline~s3")})  # scientific
    identity = world.launch(ADAMW)  # launcher~baseline beyond 0.06 (technical), the rest
    baseline, rest = world.records(ADAMW, identity, change=beyond("baseline", "launcher~baseline"))
    world.publish(ADAMW, "baseline", baseline)  # interrupted after its repeat
    q.publish(world.out / (ADAMW.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    q.publish(world.out / ADAMW.trajectory_path(1, "repeat"),
              trajectory_result(ADAMW, rest, "repeat"))  # fmt: skip
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: "failing", ADAMW_KEY: "not_qualified"}
    (case,) = [c for c in reading["rates"][ADAMW_KEY]["cases"] if c["seed"] == ADAMW.seed]
    assert case["parts"]["rest"]["from_trajectories"] is True
    record = world.folder(ADAMW) / "baseline-attempt-1.json"
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)  # this checkout: the fix
    world.command(qualify, "supersede", "--case", ADAMW.name, "--record", str(record),
                  "--diagnosis", "a stale kernel cache",
                  "--fix", "the cache is cleared")  # fmt: skip
    cycle = json.loads((world.out / "c1" / "supersede-1.json").read_text())
    assert [k["case"] for k in cycle["kept_cases"]] == [kept.name]
    cached = world.verdict(qualify)  # the old code's records void, the interrupted rest's too
    reading = world.read(expected_code=FIX["sha256"])
    assert_as_the_verdict(reading, cached)
    (case,) = [c for c in reading["rates"][ADAMW_KEY]["cases"] if c["seed"] == ADAMW.seed]
    assert case["verdict"] is None and case["parts"]["rest"]["counts"] is None
    world.full(skip=(kept,), attempt=2, code=FIX)  # the others re-qualify under the fix
    cached = world.verdict(qualify)
    reading = world.read(expected_code=FIX["sha256"])
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: "failing", ADAMW_KEY: "qualified"}
    fa = reading["rates"][FA_KEY]
    assert [(e["case"], e["why"], e["kind"]) for e in fa["verdict"]["not_counted"]] == [
        (kept.name, "other GPU code", "scientific")]  # fmt: skip
    void = {v["case"] for v in fa["verdict"]["voided"]}
    assert void == {f"s{s}/{FA_KEY}" for s in (10200, 10202)} | {"control"}
    assert all(v["reason"].startswith("superseded by an engineering cycle (supersede-1.json)")
               for v in fa["verdict"]["voided"])  # fmt: skip
    (held_case,) = [c for c in fa["cases"] if c["seed"] == kept.seed]
    assert held_case["superseded"] == [] and held_case["parts"]["baseline"]["counts"] == 1
    (cycle_listed,) = reading["supersessions"]
    assert cycle_listed["old_gpu_code"] == CODE["sha256"]
    assert cycle_listed["fix_gpu_code"] == FIX["sha256"]
    assert sorted(cycle_listed["launchers"]) == sorted(
        f"r{s}-{o}-a1" for (o, _), seeds in BOTH for s in seeds if (s, o) != (10201, "adam_frozen")
    )  # fmt: skip
    (run,) = [r for r in fa["runs"] if r["seed"] == kept.seed]
    assert run["located"] == "r10201-adam_frozen-a1" and run["superseded"] == {}


def test_a_held_run(qualify, monkeypatch, tmp_path) -> None:
    """A launcher run its declared rule stopped, no part record of its case, and no recorded
    commit: the engineering cycle cannot resolve it, keeps the case as it is and holds the run.
    The held run is listed for its rate, from the journal and the supersession record, and its
    unresolved outcome keeps the rate from qualified."""
    world = World(tmp_path, ((("adamw", ADAMW_LR), (10200, 10201)),))
    world.qualified(ADAMW, world.launch(ADAMW), change=beyond("baseline", "launcher~baseline"))
    stopped = qcase(10201, "adamw", ADAMW_LR)
    identity = world.launch(stopped, stop="gradient_spike", steps=3)
    run_dir = world.root / "study3" / identity["run_id"]
    meta = json.loads((run_dir / "meta.json").read_text())
    del meta["git"]
    (run_dir / "meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)
    world.command(qualify, "supersede", "--case", ADAMW.name, "--record",
                  str(world.folder(ADAMW) / "baseline-attempt-1.json"),
                  "--diagnosis", "d", "--fix", "f")  # fmt: skip
    cached = world.verdict(qualify)
    reading = world.read(expected_code=FIX["sha256"])
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {ADAMW_KEY: "not_qualified"}
    (held,) = reading["held"]
    assert held["run_id"] == identity["run_id"] and "no recorded commit" in held["reason"]
    assert held["sources"] == [f"journal {world.name}",
                               str(world.out / "c1" / "supersede-1.json")]  # fmt: skip
    rate = reading["rates"][ADAMW_KEY]
    assert rate["held"] == [held]
    (run,) = [r for r in rate["runs"] if r["seed"] == stopped.seed]
    assert list(run["held"]) == [identity["run_id"]]
    (resolved,) = [c for c in reading["cases"] if c.get("resolution")]
    assert (resolved["case"]["name"], resolved["state"]) == (stopped.name, "incomplete")
    assert resolved["resolution"]["outcome"] == "unresolved"


# ------------------------------------------------------------------ the cache


def test_a_verdict_json_that_disagrees_is_a_problem(qualify, tmp_path) -> None:
    """The cache is compared, never trusted: a rate state that differs, the cache of another
    candidate, of another GPU code or settings, and a rate the records do not support are
    problems; the rates themselves stay the recomputation's."""
    world = World(tmp_path)
    world.full()
    world.verdict(qualify)
    path = world.out / "c1" / "verdict.json"
    good = json.loads(path.read_text())

    def edited(change):
        record = copy.deepcopy(good)
        change(record)
        path.write_text(json.dumps(record))
        return world.read()

    reading = edited(lambda r: r["states"].update({FA_KEY: "failing"}))
    assert reading["problems"] == [
        f"{FA_KEY}: the records give 'qualified', the cache {path} 'failing'"]  # fmt: skip
    assert reading["states"][FA_KEY] == "qualified" and reading["cache"]["agrees"] is False
    assert reading["cache"]["states"][FA_KEY] == "failing"
    reading = edited(lambda r: r.update(candidate=2))
    assert [p for p in reading["problems"] if "the cache is of candidate 2" in p]
    reading = edited(lambda r: r["provenance"]["gpu_code"].update(sha256="o" * 64))
    assert reading["problems"] == [f"{path}: the cache was written under GPU code {'o' * 64!r}, "
                                   f"not the expected {CODE['sha256']!r}"]  # fmt: skip
    reading = edited(lambda r: r["provenance"]["settings"].update(bound_fraction=0.02))
    assert reading["problems"] == [f"{path}: the cache was written under other settings than the "
                                   "registered ones"]  # fmt: skip
    p_key = "adamw_prewarmed-1e-05"

    def a_third_rate(record):
        record["states"][p_key] = "qualified"
        record["rates"][p_key] = {**record["rates"][ADAMW_KEY], "optimizer": "adamw_prewarmed",
                                  "lr": 1e-05}  # fmt: skip

    reading = edited(a_third_rate)  # named by the cache alone: recomputed from no record
    assert reading["states"][p_key] == "incomplete"
    assert reading["problems"] == [
        f"{p_key}: the records give 'incomplete', the cache {path} 'qualified'"]  # fmt: skip
    reading = edited(lambda r: r.pop("states"))
    assert reading["problems"] == [f"{path}: the verdict cache holds no rate states"]
    assert reading["cache"]["agrees"] is False
    path.write_text(json.dumps(good))
    assert world.read()["problems"] == []
    path.unlink()  # no cache: nothing to compare
    reading = world.read()
    assert (reading["cache"], reading["problems"]) == (None, [])
    assert reading["states"] == {FA_KEY: "qualified", ADAMW_KEY: "qualified"}


def test_the_cache_of_a_campaign_in_progress_and_of_a_later_campaign(qualify, tmp_path) -> None:
    """The driver writes the verdict on every exit: a rate whose cases never started is in the
    cache (an FA rate without its control, not qualified) and is recomputed from no record, in
    agreement. A later campaign of the candidate rewrites the cache with its own rates: a rate
    it does not name is listed as uncached, not a problem."""
    world = World(tmp_path)
    world.full(cells=BOTH[1:])  # the AdamW cases only
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: "not_qualified", ADAMW_KEY: "qualified"}
    assert reading["rates"][FA_KEY]["runs"] == [] and reading["rates"][FA_KEY]["cases"] == []
    path = world.out / "c1" / "verdict.json"
    record = json.loads(path.read_text())
    for key in ("states", "rates"):
        del record[key][ADAMW_KEY]
    path.write_text(json.dumps(record))
    reading = world.read()
    assert (reading["problems"], reading["uncached"]) == ([], [ADAMW_KEY])
    assert reading["states"] == {ADAMW_KEY: "qualified", FA_KEY: "not_qualified"}


# ------------------------------------------------------------------ problems


def corrupt(path):
    path.write_text('{"cut short')


@pytest.mark.parametrize("what", ["part", "restart", "supersession", "trajectory", "cache",
                                  "release", "journal"])  # fmt: skip
def test_an_unreadable_record_is_a_problem(qualify, tmp_path, what) -> None:
    """Every unreadable record is a problem naming it; what the verdicts make of its absence
    still shows in the states."""
    world = World(tmp_path)
    world.full()
    world.verdict(qualify)
    folder = world.folder(ADAMW)
    path = {"part": folder / "rest-attempt-1.json", "restart": folder / "restart-1.json",
            "supersession": world.out / "c1" / "supersede-1.json",
            "trajectory": folder / "rest-attempt-2.trajectory-repeat.json",
            "cache": world.out / "c1" / "verdict.json",
            "release": folder / "checkpoints-removed-1.json",
            "journal": world.root / "launch" / "other.jsonl"}[what]  # fmt: skip
    if what == "trajectory":
        q.publish(folder / "rest-attempt-2.json.started", {"started_at": "t"})
    if what == "journal":
        path.write_bytes(b"\xff\xfe not text")
    else:
        corrupt(path)
    reading = world.read()
    problem = "an unreadable " + {"part": "rest attempt record", "restart": "case-restart record",
                                  "supersession": "supersession record",
                                  "trajectory": "trajectory result", "cache": "verdict cache",
                                  "release": "release record",
                                  "journal": "launch journal"}[what]  # fmt: skip
    assert f"{path}: {problem}" in reading["problems"]
    if what == "part":  # its record missing, the case is technical
        assert reading["states"][ADAMW_KEY] == "not_qualified"
    if what == "cache":
        assert reading["cache"]["readable"] is reading["cache"]["agrees"] is False
        assert reading["states"] == {FA_KEY: "qualified", ADAMW_KEY: "qualified"}


def test_a_record_the_verdicts_cannot_read_is_a_problem(qualify, tmp_path) -> None:
    world = World(tmp_path)
    world.full()
    path = world.folder(ADAMW) / "rest-attempt-1.json"
    record = json.loads(path.read_text())
    record["trajectories"] = ["not", "a", "mapping"]
    q.publish(path, record)
    reading = world.read()
    prefix = f"{world.folder(ADAMW)}: the verdicts cannot read its records"
    assert [p for p in reading["problems"] if p.startswith(prefix)]


def test_the_issued_runs_are_the_candidates_qualification_starts(tmp_path) -> None:
    """A start is a case of the candidate's qualification by its header (study 3, the candidate,
    the step numerical or repair); its config binds the case. One that cannot be bound, and two
    runs of one case, are problems, and that case's launcher run is located by neither."""
    world = World(tmp_path)
    world.launch(FA)
    world.launch(ADAMW)
    reading = world.read()
    assert [(r["optimizer"], r["lr"], r["seed"]) for r in reading["issued"]] == [
        ("adam_frozen", FA_LR, 10200), ("adamw", ADAMW_LR, 10200)]  # fmt: skip
    assert reading["problems"] == []
    assert reading["rates"][FA_KEY]["launchers"][10200] is not None
    others = [{**HEADER, "candidate": 2}, {**HEADER, "step": "search", "round": 0},
              {**HEADER, "study": 2}, None]  # fmt: skip
    for n, header in enumerate(others):  # not this candidate's qualification
        world.log({"event": "start", "run": f"x/s{n}", "attempt": 1, "run_id": f"x{n}",
                   "experiment": "study3", "config_hash": "0" * 12, "command": [],
                   **({"header": header} if header else {})})  # fmt: skip
    assert world.read()["issued"] == reading["issued"]
    world.log({"event": "start", "run": "lost/s10201", "attempt": 1, "run_id": "lost-1",
               "experiment": "study3", "config_hash": "0" * 12,
               "command": ["scripts/train.py", "experiments/no_such_config"],
               "header": HEADER})  # fmt: skip
    reading = world.read()
    assert [p for p in reading["problems"]
            if p.startswith("lost/s10201 (000000000000): an issued run of candidate 1's "
                            "qualification whose case cannot be bound")]  # fmt: skip
    # The same case issued again under another label, by a repair campaign
    run = world.run(FA)
    world.log({"event": "start", "run": "again/s10200", "attempt": 1, "run_id": "x1",
               "experiment": run.experiment, "config_hash": run.config_hash,
               "command": run.command()[1:], "header": {**HEADER, "step": "repair"}})  # fmt: skip
    reading = world.read()
    labels = sorted(f"{label} ({run.config_hash})" for label in (run.label, "again/s10200"))
    assert f"two issued runs are case s10200/{FA_KEY} of candidate 1: {', '.join(labels)}" in (
        reading["problems"])  # fmt: skip
    assert reading["rates"][FA_KEY]["launchers"] == {10200: None}
    assert reading["rates"][ADAMW_KEY]["launchers"][10200] is not None


# ------------------------------------------------------------------ the reader's own parts


def test_the_scope_the_policy_and_the_launcher_identity_are_the_qualifications(tmp_path):
    """``Scope``, ``settings_record``, ``launcher_identity`` and ``resolved_verdict`` give what
    ``qualification`` and scripts/qualify.py give from the same inputs."""
    from omegaconf import OmegaConf

    mine, theirs = qr.Scope.of(REGISTERED), q.QualificationScope.of(REGISTERED)
    assert dataclasses.asdict(mine) == dataclasses.asdict(theirs)
    for candidate in (1, 2):
        for name in ("k_star", "bound", "seeds", "control_bound"):
            assert getattr(mine, name)(candidate) == getattr(theirs, name)(candidate)
        assert mine.warning(0.004, candidate) == theirs.warning(0.004, candidate)
    for name in ("k_star", "bound", "seeds", "control_bound"):
        with pytest.raises(q.QualificationError, match="No candidate 3"):
            getattr(mine, name)(3)
    as_dict = OmegaConf.to_container(REGISTERED, resolve=True)
    assert qr.settings_record(REGISTERED) == q.settings_record(REGISTERED)
    assert qr.settings_record(as_dict) == q.settings_record(REGISTERED)
    assert qr.Scope.of(as_dict) == mine
    for bad in ({**as_dict, "experiment": "study3"}, {**as_dict, "seeds": None},
                {**as_dict, "gaussian": {"s4": 4, "s3": 3}}):  # fmt: skip
        with pytest.raises(q.QualificationError):
            qr.Scope.of(bad)
    with pytest.raises(q.QualificationError, match="No candidate 3"):
        qr.read_qualification(tmp_path, 3, expected_code="g" * 64, settings=REGISTERED)
    with pytest.raises(ValueError, match="expected_code"):
        qr.read_qualification(tmp_path, 1, expected_code="", settings=REGISTERED)
    world = World(tmp_path)
    identity = world.launch(FA)
    run_dir = world.root / "study3" / identity["run_id"]
    assert qr.launcher_identity(run_dir) == identity
    (run_dir / "metrics.jsonl").write_text('{"step": 0}\n')
    (run_dir / "summary.json").unlink()
    assert qr.launcher_identity(run_dir) == q.launcher_identity(run_dir) != identity
    assert qr.launcher_identity(tmp_path / "none") == q.launcher_identity(tmp_path / "none")
    script = load_script()
    for outcome in ("scientific", "unresolved"):
        resolution = {"run_id": "r", "outcome": outcome, "reasons": ["why"],
                      "case_record": FA.record(), "gpu_code": "o" * 64,
                      "identity": identity}  # fmt: skip
        assert qr.resolved_verdict(resolution) == script.resolved_verdict(resolution)


def snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns,
                                       hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*")) if p.is_file()}  # fmt: skip


SENTINEL = "SENTINEL-response-text"


def test_pure_reading(qualify, monkeypatch, tmp_path) -> None:
    """No file is written; nothing under a run's responses/ or observations/ is opened (their
    sentinel text reaches no output); the GPU code identity is not computed."""
    world = World(tmp_path)
    world.full()
    world.verdict(qualify)
    for run_dir in (world.root / "study3").iterdir():
        for sub in ("responses", "observations"):
            (run_dir / sub).mkdir(exist_ok=True)
            (run_dir / sub / "sealed.jsonl").write_text(json.dumps({"text": SENTINEL}) + "\n")
    before = snapshot(tmp_path)

    def sealed(path):
        parts = Path(str(path)).parts
        if "responses" in parts or "observations" in parts:
            raise AssertionError(f"opened {path}")

    def guard(original):
        def wrapped(*args, **kwargs):
            sealed(args[0])
            return original(*args, **kwargs)

        return wrapped

    for owner, name in ((builtins, "open"), (io, "open"), (pathlib.Path, "open"),
                        (pathlib.Path, "read_text"), (pathlib.Path, "read_bytes")):  # fmt: skip
        monkeypatch.setattr(owner, name, guard(getattr(owner, name)))

    def no_identity(*args, **kwargs):
        raise AssertionError("the GPU code identity is computed")

    monkeypatch.setattr(code_identity, "gpu_code_identity", no_identity)
    reading = world.read()
    monkeypatch.undo()
    assert reading["states"] == {FA_KEY: "qualified", ADAMW_KEY: "qualified"}
    assert reading["problems"] == []
    assert SENTINEL not in json.dumps(reading, default=str)
    assert snapshot(tmp_path) == before


def test_no_gpu_module_is_imported() -> None:
    """The module imports no GPU path (``qualification``, ``schedule``, ``train``, ``launch``,
    ``runtime``, ``fa_control_diagnostic``) and no torch; it reaches the verdict rules through
    ``analysis.qualification_verdicts``."""
    tree = ast.parse(Path(qr.__file__).read_text())
    imported, names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
            names |= {f"{node.module or ''}.{a.name}" for a in node.names}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    gpu = {"qualification", "schedule", "train", "launch", "runtime", "fa_control_diagnostic",
           "torch"}  # fmt: skip
    assert not {i for i in imported | names if gpu & set(i.split("."))}
    assert imported >= {"..qualification_constants", ".study3_records"}
    assert any(isinstance(n, ast.ImportFrom) and n.module is None and n.level == 1
               and [a.name for a in n.names] == ["qualification_verdicts"]
               for n in ast.walk(tree))  # fmt: skip


def test_a_case_of_a_rate_no_campaign_issued_is_a_rate_of_its_own(qualify, tmp_path) -> None:
    """A case record of a rate no campaign issued (the script's verdict lists it as unassigned)
    is a rate of its own, recomputed (here incomplete: no issued run is located), and changes
    no issued rate."""
    world = World(tmp_path)
    world.full()
    stray = qcase(10200, "adamw_prewarmed", 1.0e-05)
    world.qualified(stray, dict(IDENTITY))
    reading = world.read()
    assert reading["states"][FA_KEY] == reading["states"][ADAMW_KEY] == "qualified"
    assert reading["states"]["adamw_prewarmed-1e-05"] == "incomplete"
    assert reading["rates"]["adamw_prewarmed-1e-05"]["arm"] == "P"
    shutil.rmtree(world.folder(stray))
    assert "adamw_prewarmed-1e-05" not in world.read()["rates"]
