"""Study 3's numerical qualification read on the analysis side (analysis/qualification_records.py):
each rate's state recomputed from the qualification's records, verdict.json compared as a cache
(docs/study3-preregistration.md, "Numerical qualification", "Failure routes", "Voiding and
supersession", "Amendment 2").

Each world is a qualification of candidate 1 in the formats of scripts/qualify.py: a campaign
with its study-3 header, the launch journal, settled launcher runs (their meta.json recording the
campaign's launch as the launcher passes it) and the part records, the restarts and the
engineering cycle written by the script itself. The script's own ``verdict`` (and, where a world
asks, ``supersede``, ``restart-case``, ``complete`` and the launcher's ``held_runs``) runs on the
same world, and the recomputation must give its rate verdicts, case verdicts and attempt listing.
Nothing here runs a model.
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
from omegaconf import OmegaConf
from test_qualification import CODE, IDENTITY, control_record, synthetic
from test_qualify import (
    ADAMW_LR,
    FA_LR,
    FIX,
    RESTART,
    UNRESOLVABLE,
    cell,
    codes,
    commit_configs,
    load_script,
    raised_before_the_check,
    settled_launcher,
    trajectory_result,
)

from zero_field_refusal import code_identity
from zero_field_refusal import launch as launcher
from zero_field_refusal import qualification as q
from zero_field_refusal.analysis import qualification_records as qr
from zero_field_refusal.analysis.study3_records import read_holds
from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.launch import load_campaign
from zero_field_refusal.paths import CONFIG_ROOT

REGISTERED = load_config("experiments/qualification")
SEEDS = (10200, 10201, 10202)
HEADER = {"study": 3, "candidate": 1, "step": "numerical", "round": None}
FA_KEY, ADAMW_KEY = f"adam_frozen-{FA_LR:g}", f"adamw-{ADAMW_LR:g}"
BOTH = ((("adam_frozen", FA_LR), SEEDS), (("adamw", ADAMW_LR), SEEDS))


@pytest.fixture
def qualify(monkeypatch, tmp_path):
    """scripts/qualify.py, from a clean checkout whose GPU code is the records' (``CODE``); the
    launcher runs' commit abc123 has this checkout's configs (``configs_at``)."""
    module = load_script()
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(module, "git_state", lambda *a, **k: {"commit": "c" * 40, "dirty": False})
    monkeypatch.setattr(code_identity, "gpu_code_identity", lambda *a, **k: dict(CODE))
    monkeypatch.setattr(module, "configs_at", commit_configs({"abc123": CONFIG_ROOT}))
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

    def log(self, record, campaign=None):
        path = self.root / "launch" / f"{campaign or self.name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"time": "2026-10-26T00:00:00+00:00", **record}) + "\n")

    def launch(self, case, *, campaign=None, issued=None, commit="abc123", **settled):
        """One launcher attempt of the case's run, as ``launch.py --only LABEL --max-attempts 1``
        logs and leaves it (settled, its meta.json recording the campaign's launch, the label,
        the attempt and the header, and its commit), in ``campaign``'s log (the world's own by
        default); its identity as the records name it. ``issued``: the config it was issued
        under the configs of its commit, when they are not this checkout's (its config hash is
        then that config's)."""
        run = self.run(case)
        n = self.attempts[run.label] = self.attempts.get(run.label, 0) + 1
        run_id = f"r{case.seed}-{case.optimizer}-a{n}"
        chash = run.config_hash if issued is None else config_hash(issued)
        common = {"run": run.label, "attempt": n, "run_id": run_id, "only": run.label}
        self.log({"event": "start", **common, "experiment": run.experiment,
                  "config_hash": chash, "command": run.command()[1:],
                  "header": self.header, "max_attempts": 1}, campaign)  # fmt: skip
        run_dir = settled_launcher(self.root, run, run_id, **settled)
        if issued is not None:
            OmegaConf.save(issued, run_dir / "config.yaml")
        meta = json.loads((run_dir / "meta.json").read_text())
        meta["config_hash"] = chash
        meta["git"] = {"commit": commit, "dirty": False}
        meta["campaign"] = {"name": campaign or self.name, "label": run.label, "attempt": n,
                            "header": self.header}  # fmt: skip
        if q.reads_v0(case.optimizer):  # an FA run records v₀'s hash
            meta["inputs"]["v0_sha256"] = "v" * 64
        (run_dir / "meta.json").write_text(json.dumps(meta))
        self.log({"event": "end", **common, "returncode": 0, "outcome": "done", "problem": None},
                 campaign)  # fmt: skip
        return q.launcher_identity(run_dir)

    def publish(self, case, part, record, attempt=1):
        q.publish(self.out / case.attempt_path(part, attempt), record)

    def records(self, case, identity, *, code=CODE, change=None):
        """A case's part records as ``run_part`` writes them once its launcher check returned
        (``launcher.classified``), qualified unless ``change``."""
        baseline, rest = synthetic(case, identity=identity)
        for record in (baseline, rest):
            record["provenance"]["gpu_code"] = code
            record["launcher"]["classified"] = True
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


@pytest.mark.parametrize(("k", "diverged", "state"), [(0.06, False, "qualified"),
                                                      (0.0601, False, "not_qualified"),
                                                      (1e-12, True, "not_qualified")])  # fmt: skip
def test_the_same_path_controls_rule(qualify, tmp_path, k, diverged, state) -> None:
    """Amendment 2: an FA rate's same-path control, on the candidate's first numerical seed for
    25 steps, needs finite records and an endpoint K at most 0.01·K* (0.06 for candidate 1). At
    the bound the rate qualifies; beyond it, or with a non-finite record, the control fails and
    blocks its rate (not qualified, technical), the reading as the verdict."""
    world = World(tmp_path)
    world.full(control=False)
    identity = q.launcher_identity(world.root / "study3" / "r10200-adam_frozen-a1")
    world.publish(FA, "control", control_record(k, diverged=diverged, case=FA, identity=identity))
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: state, ADAMW_KEY: "qualified"}
    control = reading["rates"][FA_KEY]["verdict"]["control"]
    assert (control["bound"], control["kind"]) == (0.06, None if state == "qualified" else
                                                   "technical")  # fmt: skip


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


STOPPED = qcase(10201, "adamw", ADAMW_LR)
ADAMW_ONLY = ((("adamw", ADAMW_LR), (10200, 10201)),)


def cycle_on_a_stop(qualify, monkeypatch, tmp_path, change=None, *, setup_raised=False):
    """AdamW on 10200 and 10201: 10200's baseline technical (launcher~baseline beyond 0.06);
    10201's launcher run stopped on its declared gradient_spike rule, no part record of its case
    classified it, and ``change`` edits its records first; with ``setup_raised``, its baseline
    as ``run_part`` publishes it when the numerics setup raises before the launcher check
    (failed, technical, naming the run without classifying it). Then the engineering cycle,
    opened by 10200's technical baseline under the fix's GPU code. Returns the world, the
    stopped run's identity and the cycle's record."""
    world = World(tmp_path, ADAMW_ONLY)
    world.qualified(ADAMW, world.launch(ADAMW), change=beyond("baseline", "launcher~baseline"))
    identity = world.launch(STOPPED, stop="gradient_spike", steps=3)
    run_dir = world.root / "study3" / identity["run_id"]
    if change is not None:
        change(run_dir)
    if setup_raised:
        raised_before_the_check(qualify, world.out, run_dir, monkeypatch, STOPPED)
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)
    world.command(qualify, "supersede", "--case", ADAMW.name, "--record",
                  str(world.folder(ADAMW) / "baseline-attempt-1.json"),
                  "--diagnosis", "d", "--fix", "f")  # fmt: skip
    return world, identity, json.loads((world.out / "c1" / "supersede-1.json").read_text())


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize("missing", list(UNRESOLVABLE))
def test_a_held_run(qualify, monkeypatch, tmp_path, missing) -> None:
    """A launcher run its declared rule stopped, conforming to its journal start (its meta.json
    records the campaign's header and label), no part record of its case, and its provenance
    incomplete (no recorded commit; a commit not in this clone; no campaign header in its
    meta.json; a saved config that does not read): the engineering cycle cannot resolve it
    against what it was issued, keeps the case as it is and holds the run. The held run is
    listed for its rate and its label, from the journal and the supersession record, each with
    its sha256 (the journal line's log and number), and its unresolved outcome keeps the rate
    from qualified: ``not_qualified``, as the verdict gives it (10200's case, re-qualified, has
    no record under the fix)."""
    change, why = UNRESOLVABLE[missing]
    world, identity, cycle = cycle_on_a_stop(qualify, monkeypatch, tmp_path, change)
    assert [h["run_id"] for h in cycle["held"]] == [identity["run_id"]]
    cached = world.verdict(qualify)
    reading = world.read(expected_code=FIX["sha256"])
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {ADAMW_KEY: "not_qualified"}
    (held,) = reading["held"]
    assert held["run_id"] == identity["run_id"] and why in held["reason"]
    cycle_path = world.out / "c1" / "supersede-1.json"
    assert held["sources"] == [f"journal {world.name}", str(cycle_path)]
    log = world.root / "launch" / f"{world.name}.jsonl"
    lines = log.read_text().splitlines()
    (number,) = [n for n, line in enumerate(lines, start=1) if json.loads(line)["event"] == "held"]
    line_sha256 = hashlib.sha256(lines[number - 1].encode()).hexdigest()
    assert held["records"] == [
        {"kind": "journal", "path": f"launch/{world.name}.jsonl", "root": "runs",
         "line": number, "line_sha256": line_sha256, "sha256": file_sha256(log)},
        {"kind": "supersession", "path": "qualification/c1/supersede-1.json", "root": "runs",
         "sha256": file_sha256(cycle_path)}]  # fmt: skip
    assert held["case"] == {"arm": "A", "optimizer": "adamw", "lr": ADAMW_LR, "seed": 10201}
    assert (held["run"], held["experiment"]) == (world.run(STOPPED).label, "study3")
    rate = reading["rates"][ADAMW_KEY]
    assert rate["held"] == [held]
    (run,) = [r for r in rate["runs"] if r["seed"] == STOPPED.seed]
    assert list(run["held"]) == [identity["run_id"]]
    records = launcher.launch_records(world.root)
    assert run["held"] == launcher.held_runs(world.run(STOPPED), records)
    (resolved,) = [c for c in reading["cases"] if c.get("resolution")]
    assert (resolved["case"]["name"], resolved["state"]) == (STOPPED.name, "incomplete")
    assert resolved["resolution"]["outcome"] == "unresolved"


def test_a_stop_whose_meta_departs_from_its_start_is_superseded(qualify, monkeypatch, tmp_path):
    """The world the held-run test had before the cycle resolved a run against what it was
    issued: 10201's stopped launcher run records no commit, and its meta.json records the
    campaign's launch with no header while its journal start carries the study-3 header. Its
    records depart from what it was issued (technical, whatever is missing), so the cycle
    supersedes it and re-qualifies its case: nothing is held, no case is kept, and the rate is
    ``incomplete`` (no record under the fix, no launcher run located), the reading as the
    verdict."""

    def departs(run_dir):
        meta = json.loads((run_dir / "meta.json").read_text())
        del meta["git"]
        meta["campaign"]["header"] = None
        (run_dir / "meta.json").write_text(json.dumps(meta))

    world, identity, cycle = cycle_on_a_stop(qualify, monkeypatch, tmp_path, departs)
    assert (cycle["held"], cycle["kept_cases"]) == ([], [])
    assert identity["run_id"] in [r["run_id"] for r in cycle["launchers"]]
    cached = world.verdict(qualify)
    reading = world.read(expected_code=FIX["sha256"])
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {ADAMW_KEY: "incomplete"}
    assert reading["held"] == [] and [c for c in reading["cases"] if c.get("resolution")] == []
    (run,) = [r for r in reading["rates"][ADAMW_KEY]["runs"] if r["seed"] == STOPPED.seed]
    assert run["located"] is None and identity["run_id"] in run["superseded"]


def test_a_resolved_stop_counts_beside_a_record_that_did_not_classify_it(
    qualify, monkeypatch, tmp_path
) -> None:
    """A part record classifies its launcher run only once its launcher check returned
    (``launcher.classified``): 10201's baseline raised in its numerics setup first, so it names
    the run without classifying it, and the cycle resolves the run on its own: a conforming
    declared stop, scientific; the case is kept whole, its record included. The verdict counts
    the resolution beside that record's own verdict, and so does the reading: the rate fails."""
    world, identity, cycle = cycle_on_a_stop(qualify, monkeypatch, tmp_path, setup_raised=True)
    (kept,) = cycle["kept_cases"]
    assert (kept["case"], kept["resolution"]["outcome"]) == (STOPPED.name, "scientific")
    assert [(e["case"], e["part"]) for e in cycle["kept"]] == [(STOPPED.name, "baseline")]
    cached = world.verdict(qualify)
    reading = world.read(expected_code=FIX["sha256"])
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {ADAMW_KEY: "failing"}
    (case,) = [c for c in reading["rates"][ADAMW_KEY]["cases"] if c["seed"] == STOPPED.seed]
    assert (case["verdict"]["kind"], case["parts"]["baseline"]["counts"]) == ("technical", 1)
    (resolved,) = [c for c in reading["cases"] if c.get("resolution")]
    assert (resolved["case"]["name"], resolved["kind"]) == (STOPPED.name, "scientific")
    assert reading["held"] == []


def test_a_runs_attempts_are_read_in_attempt_order_across_journals(qualify, tmp_path) -> None:
    """FA's 10200 restarted after the Pod stopped during its rest; the recovery round's attempt
    2 is logged under a campaign name that sorts before the first's. The issued run's attempts
    are read by their number, as ``label_run_ids`` and ``journal_run_ids`` read them, whatever
    journal file holds them, and the second is located."""
    world = World(tmp_path)
    world.full(skip=(FA,))
    identity = world.launch(FA)
    world.qualified(FA, identity, control=True)
    (world.folder(FA) / "rest-attempt-1.json").unlink()
    q.publish(world.out / (FA.attempt_path("rest", 1) + ".started"), {"started_at": "t"})
    world.command(qualify, "restart-case", "--case", FA.name, "--infrastructure", RESTART)
    world.qualified(FA, world.launch(FA, campaign="a-recovery"), attempt=2)
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    run = world.run(FA)
    order = qualify.label_run_ids(run, world.root)
    assert order == qualify.journal_run_ids(run, world.root) == [
        "r10200-adam_frozen-a1", "r10200-adam_frozen-a2"]  # fmt: skip
    (issued,) = [r for r in reading["rates"][FA_KEY]["runs"] if r["seed"] == FA.seed]
    assert issued["run_ids"] == order and issued["campaigns"] == ["a-recovery", world.name]
    assert issued["located"] == qualify.located(run, world.root).name == "r10200-adam_frozen-a2"


EARLIER = "e0e0e0"  # the commit the earlier configs are of


def issued_earlier(world, case, configs):
    """The case's config as ``configs`` (an earlier commit's configs folder) composes its
    command: what a launcher run of that commit was issued."""
    run = world.run(case)
    return load_config(configs / f"{run.config}.yaml", run.overrides, root=configs)


def earlier_configs(qualify, monkeypatch, tmp_path):
    """The configs of commit ``EARLIER``: this checkout's with another teacher chunk size, a
    hashed key the fix changed; ``configs_at`` knows that commit and abc123 (this checkout's)."""
    then = tmp_path / "configs-of-the-commit"
    shutil.copytree(CONFIG_ROOT, then)
    study3 = then / "experiments" / "study3.yaml"
    assert study3.read_text().count("  chunk_size: 1000\n") == 1
    study3.write_text(study3.read_text().replace("  chunk_size: 1000\n", "  chunk_size: 500\n"))
    monkeypatch.setattr(qualify, "configs_at",
                        commit_configs({"abc123": CONFIG_ROOT, EARLIER: then}))  # fmt: skip
    return then


def cycle_across_a_config_change(qualify, monkeypatch, tmp_path, change=None):
    """AdamW on 10200 and 10201, both first issued at commit ``EARLIER`` under its configs (an
    earlier config hash than the cells' now): 10200's baseline technical; 10201's launcher run
    stopped on its declared gradient_spike rule, no part record of its case, ``change`` editing
    its records first. The cycle, opened by 10200's baseline under the fix, finds each case's
    runs by its label; 10200 is re-qualified under the fix from a recovery attempt issued under
    the cells' config hash now. Returns the world, the identities of 10200's first run and of the
    stopped run, and the cycle."""
    then = earlier_configs(qualify, monkeypatch, tmp_path)
    world = World(tmp_path, ADAMW_ONLY)
    first = world.launch(ADAMW, issued=issued_earlier(world, ADAMW, then), commit=EARLIER)
    world.qualified(ADAMW, first, change=beyond("baseline", "launcher~baseline"))
    stopped = world.launch(STOPPED, issued=issued_earlier(world, STOPPED, then), commit=EARLIER,
                           stop="gradient_spike", steps=3)  # fmt: skip
    assert stopped["config_hash"] != world.run(STOPPED).config_hash  # not the cell's hash now
    if change is not None:
        change(world.root / "study3" / stopped["run_id"])
    monkeypatch.setattr(code_identity, "gpu_code_identity", codes)
    world.command(qualify, "supersede", "--case", ADAMW.name, "--record",
                  str(world.folder(ADAMW) / "baseline-attempt-1.json"),
                  "--diagnosis", "d", "--fix", "f")  # fmt: skip
    world.qualified(ADAMW, world.launch(ADAMW), attempt=2, code=FIX)  # the recovery round
    return world, first, stopped, json.loads((world.out / "c1" / "supersede-1.json").read_text())


def test_a_label_is_one_launcher_run_whatever_its_config_hash(qualify, monkeypatch, tmp_path):
    """The cycle finds a case's launcher runs by its label, whatever their config hash: 10201's
    conforming stop, issued under the earlier configs, keeps its case whole (scientific); 10200's
    earlier run is superseded and its recovery attempt runs under the cell's hash now. Each label
    is one issued run with its attempts under both hashes (no two runs of one case), located
    under the cell's hash now: 10200's recovery attempt, and for 10201 none, as the verdict,
    given the campaign under the fix, locates them. The rate fails."""
    world, first, stopped, cycle = cycle_across_a_config_change(qualify, monkeypatch, tmp_path)
    (kept,) = cycle["kept_cases"]
    assert (kept["case"], kept["resolution"]["outcome"]) == (STOPPED.name, "scientific")
    assert [r["run_id"] for r in cycle["launchers"]] == ["r10200-adamw-a1"]
    cached = world.verdict(qualify)
    reading = world.read(expected_code=FIX["sha256"])
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {ADAMW_KEY: "failing"}
    runs = {r["seed"]: r for r in reading["rates"][ADAMW_KEY]["runs"]}
    now = {seed: world.run(qcase(seed, "adamw", ADAMW_LR)) for seed in (10200, 10201)}
    for seed, run in now.items():
        assert runs[seed]["run_ids"] == qualify.label_run_ids(run, world.root)
        assert runs[seed]["config_hash"] == run.config_hash
        located = qualify.located(run, world.root)
        assert runs[seed]["located"] == (None if located is None else located.name)
    assert runs[10200]["config_hashes"] == [first["config_hash"], now[10200].config_hash]
    assert (runs[10200]["located"], runs[10201]["located"]) == ("r10200-adamw-a2", None)
    assert runs[10201]["config_hashes"] == [stopped["config_hash"]]


def test_a_hold_under_an_earlier_config_hash_holds_its_label(qualify, monkeypatch, tmp_path):
    """10201's stopped run, issued under the earlier configs, records no commit: the cycle holds
    it under its earlier config hash. The hold applies to its label whatever the hash: the
    reading lists it for the rate and the label (``launch.held_runs`` of the cell under its hash
    now gives the same run and reason, and ``complete`` exits HELD for the case), with the
    records that hold it and their sha256 (``read_holds``), and the rate is as the verdict
    gives it."""
    no_commit = UNRESOLVABLE["no commit"][0]
    world, _, stopped, cycle = cycle_across_a_config_change(qualify, monkeypatch, tmp_path,
                                                            no_commit)  # fmt: skip
    assert [(h["run_id"], h["config_hash"]) for h in cycle["held"]] == [
        (stopped["run_id"], stopped["config_hash"])]  # fmt: skip
    cached = world.verdict(qualify)
    reading = world.read(expected_code=FIX["sha256"])
    assert_as_the_verdict(reading, cached)
    run = world.run(STOPPED)
    held_now = launcher.held_runs(run, launcher.launch_records(world.root))
    assert list(held_now) == [stopped["run_id"]]
    (held,) = reading["rates"][ADAMW_KEY]["held"]
    assert {held["run_id"]: held["reason"]} == held_now
    (listed,) = [r for r in reading["rates"][ADAMW_KEY]["runs"] if r["seed"] == STOPPED.seed]
    assert listed["held"] == held_now
    with pytest.raises(SystemExit) as stop:
        world.command(qualify, "complete", "--case", STOPPED.name)
    assert stop.value.code == qualify.HELD
    holds, problems = read_holds(world.root)
    assert problems == [] and [h["run_id"] for h in holds] == [stopped["run_id"]]
    assert holds[0]["records"] == held["records"]
    for record in held["records"]:
        assert record["sha256"] == file_sha256(world.root / record["path"])


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


def test_a_case_record_no_issued_start_names_is_a_problem(qualify, tmp_path) -> None:
    """Every run is bound to an issued cell: a case record of a rate no campaign issued (the
    script's verdict lists it as unassigned) is a problem, and a rate of its own, recomputed
    (here incomplete: no issued run is located), that changes no issued rate; so is a record of
    another candidate in the candidate's folder."""
    world = World(tmp_path)
    world.full()
    assert world.read()["problems"] == []
    stray = qcase(10200, "adamw_prewarmed", 1.0e-05)
    world.qualified(stray, dict(IDENTITY))
    reading = world.read()
    assert reading["problems"] == [
        f"{world.folder(stray)}: case s10200/adamw_prewarmed-1e-05, which no issued start of "
        "candidate 1's qualification names"]  # fmt: skip
    assert reading["states"][FA_KEY] == reading["states"][ADAMW_KEY] == "qualified"
    assert reading["states"]["adamw_prewarmed-1e-05"] == "incomplete"
    assert reading["rates"]["adamw_prewarmed-1e-05"]["arm"] == "P"
    shutil.rmtree(world.folder(stray))
    assert "adamw_prewarmed-1e-05" not in world.read()["rates"]
    resolution = {"run_id": "r", "outcome": "scientific", "reasons": ["why"],
                  "case_record": stray.record()}  # fmt: skip
    path = cycle_record(world, kept_cases=[{"case": stray.name, "stage": "numerical",
                                            "why": "w", "resolution": resolution}])  # fmt: skip
    assert world.read()["problems"] == [
        f"{path}: case s10200/adamw_prewarmed-1e-05, which no issued start of candidate 1's "
        "qualification names"]  # fmt: skip
    path.unlink()
    other = q.QualificationCase(10201, "adamw", ADAMW_LR, 2, False)
    folder = world.folder(qcase(10201, "adamw", ADAMW_LR))
    q.publish(folder / "baseline-attempt-2.json", synthetic(other)[0])
    assert f"{folder}: a record of no numerical case of candidate 1: {other.record()!r}" in (
        world.read()["problems"])  # fmt: skip


def test_every_start_of_an_issued_run_is_bound(tmp_path) -> None:
    """A run's starts are all bound, not only the first that binds: a later attempt whose
    config.yaml has another config hash is refused, a problem naming it, while the run keeps
    the case its other start binds."""
    from omegaconf import OmegaConf

    world = World(tmp_path)
    world.launch(FA)
    second = world.launch(FA)["run_id"]
    path = world.root / "study3" / second / "config.yaml"
    cfg = OmegaConf.load(path)
    OmegaConf.update(cfg, "optimizer.lr", 9.0e-05)
    OmegaConf.save(cfg, path)
    reading = world.read()
    (problem,) = reading["problems"]
    assert problem.startswith(f"{second} ({world.run(FA).label}): a start of candidate 1's "
                              "qualification refused: its config.yaml's config hash")  # fmt: skip
    (run,) = reading["issued"]
    assert (run["optimizer"], run["lr"], run["seed"]) == ("adam_frozen", FA_LR, 10200)
    assert run["run_ids"] == ["r10200-adam_frozen-a1", second]


def test_a_labels_starts_bind_one_case_and_its_cell_composes(tmp_path) -> None:
    """The starts of one label are the attempts of one launcher run, whatever their config hash:
    a second start whose own config (of its own hash) is of another rate binds another case, a
    problem, and the run is no case's. A label whose latest start's command no longer composes
    under this checkout's configs has no config hash now to locate its run under: a problem, and
    none is located."""
    (tmp_path / "two").mkdir()
    world = World(tmp_path / "two")
    world.launch(FA)
    run = world.run(FA)
    cfg = OmegaConf.load(world.root / "study3" / "r10200-adam_frozen-a1" / "config.yaml")
    OmegaConf.update(cfg, "optimizer.lr", 9.0e-05)
    other = config_hash(cfg)
    (world.root / "study3" / "x2").mkdir()
    OmegaConf.save(cfg, world.root / "study3" / "x2" / "config.yaml")
    world.log({"event": "start", "run": run.label, "attempt": 2, "run_id": "x2",
               "experiment": "study3", "config_hash": other, "command": run.command()[1:],
               "header": HEADER})  # fmt: skip
    reading = world.read()
    (problem,) = reading["problems"]
    assert problem.startswith(f"{run.label} ({run.config_hash}, {other}): the starts of one "
                              "label bind different cases: ['r10200-adam_frozen-a1'] "
                              "('adam_frozen', 1.83e-05, 10200); ['x2'] ")  # fmt: skip
    assert [r["optimizer"] for r in reading["issued"]] == [None] and FA_KEY not in reading["rates"]
    (tmp_path / "gone").mkdir()
    world = World(tmp_path / "gone")
    world.launch(FA)
    log = world.root / "launch" / f"{world.name}.jsonl"
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    lines[0]["command"][1] = "experiments/no_such_config"  # its config.yaml still binds
    log.write_text("".join(json.dumps(line) + "\n" for line in lines))
    reading = world.read()
    assert reading["problems"] == [
        f"{run.label} ({run.config_hash}): its latest start's command does not compose under "
        "this checkout's configs (FileNotFoundError): its launcher run cannot be located"
    ]
    assert reading["rates"][FA_KEY]["launchers"] == {10200: None}


def cycle_record(world, **fields):
    """A supersession record of candidate 1 with every field the registration asks for, as
    ``qualify.py supersede`` writes one, and ``fields`` over it."""
    record = {"candidate": 1, "supersession": 1,
              "old_record": {"case": ADAMW.name, "stage": "numerical", "path": "/x.json",
                             "sha256": "0" * 64, "part": "baseline", "attempt": 1},
              "diagnosis": "d", "fix": "f", "cases_requalified": [], "attempts": [],
              "kept_cases": [], "kept": [], "launchers": [], "held": [], **fields}  # fmt: skip
    path = world.out / "c1" / "supersede-1.json"
    q.publish(path, record)
    return path


@pytest.mark.parametrize("what", ["trajectory comparisons", "kept reasons",
                                  "restart entry"])  # fmt: skip
def test_a_malformed_record_is_a_problem_not_an_exception(qualify, tmp_path, what) -> None:
    """A record whose shape the verdicts cannot read (an interrupted rest's trajectory result
    whose comparisons are a list; a kept case's resolution whose reasons cannot be iterated; a
    restart naming a part by a list) is a problem naming it; the reading goes on."""
    world = World(tmp_path)
    world.full()
    folder = world.folder(ADAMW)
    if what == "restart entry":
        entry = {"part": ["baseline"], "attempt": 1, "state": "recorded", "sha256": None}
        q.publish(folder / "restart-1.json", {"reason": "r", "attempts": [entry]})
        prefix = f"{folder}: its records cannot be read: "
    elif what == "trajectory comparisons":
        q.publish(folder / "rest-attempt-2.json.started", {"started_at": "t"})
        rest = json.loads((folder / "rest-attempt-1.json").read_text())
        result = {**trajectory_result(ADAMW, rest, "repeat", attempt=2), "comparisons": [1, 2]}
        q.publish(folder / "rest-attempt-2.trajectory-repeat.json", result)
        prefix = f"{folder}: the verdicts cannot read its trajectory results: "
    else:
        resolution = {"run_id": "r", "outcome": "unresolved", "reasons": ["why"],
                      "case_record": qcase(10201, "adamw", ADAMW_LR).record()}  # fmt: skip
        kept = {"case": "s10299/adamw-5e-05", "stage": "numerical", "why": "w",
                "resolution": resolution}  # fmt: skip
        resolution["reasons"] = 5
        prefix = f"{cycle_record(world, kept_cases=[kept])}: a kept case the verdicts cannot read: "
    reading = world.read()
    assert [p for p in reading["problems"] if p.startswith(prefix + "TypeError")], (
        reading["problems"])  # fmt: skip
    assert reading["states"][FA_KEY] == "qualified"


def test_a_kept_case_counts_by_its_resolution_whatever_its_name(qualify, tmp_path) -> None:
    """The verdict counts a kept case's resolution by its stage alone, beside the case's own
    records, never looking its name up among them: a kept case whose name is a list is no
    malformed record. The reading counts it as the verdict does, with no problem; the unresolved
    outcome keeps the rate from qualified (``not_qualified``, as the verdict gives it)."""
    world = World(tmp_path)
    world.full()
    resolution = {"run_id": "r10201-adamw-a1", "outcome": "unresolved", "reasons": ["why"],
                  "case_record": qcase(10201, "adamw", ADAMW_LR).record()}  # fmt: skip
    cycle_record(world, kept_cases=[{"case": ["s10201", "adamw"], "stage": "numerical",
                                     "why": "w", "resolution": resolution}])  # fmt: skip
    cached = world.verdict(qualify)
    reading = world.read()
    assert_as_the_verdict(reading, cached)
    assert reading["states"] == {FA_KEY: "qualified", ADAMW_KEY: "not_qualified"}


def test_a_supersession_record_without_its_fields_is_a_problem_and_still_voids(qualify, tmp_path):
    """A supersession record must name the old record (path and sha256), the diagnosis, the fix
    and the cases re-qualified; one that lacks any is a problem, and still voids the attempts it
    names, as the script's verdict reads it."""
    world = World(tmp_path)
    world.full()
    record = world.folder(ADAMW) / "baseline-attempt-1.json"
    entry = {"case": ADAMW.name, "stage": "numerical", "part": "baseline", "attempt": 1,
             "state": "recorded",
             "sha256": hashlib.sha256(record.read_bytes()).hexdigest()}  # fmt: skip
    path = world.out / "c1" / "supersede-1.json"
    q.publish(path, {"attempts": [entry]})
    cached = world.verdict(qualify)
    reading = world.read()
    assert reading["problems"] == [f"{path}: a supersession record without its diagnosis, fix, "
                                   "old_record, cases_requalified"]  # fmt: skip
    got = plain(reading)
    assert {key: got["rates"][key]["verdict"] for key in cached["rates"]} == cached["rates"]
    assert got["superseded"] == cached["superseded"]
    assert [(e["case"], e["part"], e["supersession"]) for e in reading["superseded"]] == [
        (ADAMW.name, "baseline", "superseded by an engineering cycle (supersede-1.json): None; "
         "fix: None")]  # fmt: skip
    assert reading["states"][ADAMW_KEY] == "not_qualified"
    old = {"case": ADAMW.name, "stage": "numerical", "path": str(record)}  # no sha256
    cycle_record(world, attempts=[entry], old_record=old)
    assert world.read()["problems"] == [f"{path}: a supersession record without its old_record"]
