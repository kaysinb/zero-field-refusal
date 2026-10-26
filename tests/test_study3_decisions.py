"""Study 3's decision records (analysis/study3_decisions.py), on synthetic RUNS_DIR trees: the
records under ``study3/decisions/``, the evidence files they name (in the runs directory and in a
checkout), and launch journals whose headers name a record. The run folders hold a harmful-response
file and an observations file with a sentinel text; nothing here opens either."""

import builtins
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from zero_field_refusal.analysis.study3_decisions import (
    LEVELS,
    DecisionRefused,
    budget_needed,
    budget_reason,
    cap_of,
    coverage_threshold,
    decision_path,
    decisions,
    immutable,
    kind_of,
    read_decision,
    write_decision,
)
from zero_field_refusal.analysis.study3_records import SEEDS

SENTINEL = "SEALED-RESPONSE-5d1e0b"
CODE = {"commit": "a" * 40, "gpu_code_sha256": "b" * 64}
WHEN = "2026-10-26T12:00:00Z"
SEALED = ("responses", "observations")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class World:
    """A runs directory and a checkout: evidence files, decision records and launch journals."""

    def __init__(self, base: Path) -> None:
        self.runs, self.repo = base / "runs", base / "repo"
        self.runs.mkdir(parents=True)
        self.repo.mkdir(parents=True)
        self.lines = 0
        run = self.runs / "study3" / "20261026T000000Z-A-s10300-abcd1234-a1"
        for folder, name in (("responses", "harmful_evaluation.jsonl"),
                             ("observations", "measurements.jsonl")):  # fmt: skip
            (run / folder).mkdir(parents=True)
            (run / folder / name).write_text(json.dumps({"text": SENTINEL}) + "\n")
        self.run = run

    # -- evidence

    def ref(self, name: str, text: str | None = None, *, root: str = "runs") -> dict:
        """An evidence file (a new one holding ``text``) and the reference that records it."""
        path = (self.repo if root == "repo" else self.runs) / "evidence" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if text is not None or not path.exists():
            path.write_text(text if text is not None else f"{name}\n")
        return {"path": f"evidence/{name}", "sha256": sha256(path),
                **({"root": "repo"} if root == "repo" else {})}  # fmt: skip

    def file(self, ref: dict) -> Path:
        return (self.repo if ref.get("root") == "repo" else self.runs) / ref["path"]

    # -- the records

    def engineering(self, **over) -> dict:
        checks = {name: {**self.ref(f"{name}.json"), "passed": True}
                  for name in ("dry_run", "bridge", "theta0_utility", "stage0")}  # fmt: skip
        return {"kind": "engineering", "state": "passed", "written_at": WHEN, "code": CODE,
                "checks": checks, **over}  # fmt: skip

    def simulation(self, **over) -> dict:
        record = {
            "kind": "simulation", "state": "passed", "written_at": WHEN, "code": CODE,
            "fixation": self.ref("simulation-fixation.json"),
            "inputs": {"seed_vectors": self.ref("seed-vectors.json"),
                       "design": self.ref("design.json")},
            "gates": default_gates(),
            "coverage": [coverage_row(0.975)], "level": 0.975, "validation_blocks": 28,
            "verification": verified(calibration_stream=11, verification_stream=12),
        }  # fmt: skip
        return {**record, **over}

    def n(self, candidate: int = 1, N: int = 75, **over) -> dict:
        first = SEEDS["validation"][candidate][0]
        seeds = list(range(first, first + 25))
        record = {
            "kind": f"n-c{candidate}", "state": "passed", "written_at": WHEN, "code": CODE,
            "candidate": candidate,
            "rates": {"A": {"lr_l": 1.45e-5, "lr_u": 2.11e-5},
                      "F": {"lr_l": 1.30e-5, "lr_u": 1.83e-5},
                      "P": {"lr_l": 1.30e-5, "lr_u": 1.85e-5}},
            "active": ["A-F", "P-F"], "level": 0.975,
            "cohort": {"seeds": seeds,
                       "run_ids": [f"run-{s}-{k}" for s in seeds for k in range(4)]},
            "spread": {"script": self.ref("study3_spread.py", root="repo"),
                       "output": self.ref("spread-output.json")},
            "spread_pp": 4.5, "N": N, "cap": CAPS.get(N),
            "simulation": {"path": "study3/decisions/simulation.json",
                           "sha256": self.digest("simulation"), "root": "runs"},
        }  # fmt: skip
        return {**record, **over}

    def budget(self, candidate: int = 1, N: int = 99, **over) -> dict:
        record = {
            "kind": f"budget-c{candidate}", "state": "passed", "written_at": WHEN, "code": CODE,
            "candidate": candidate, "N": N, "cap": CAPS.get(N), "decision": "proceed",
            "answers": {"path": f"study3/decisions/n-c{candidate}.json",
                        "sha256": self.digest(f"n-c{candidate}")},
        }  # fmt: skip
        return {**record, **over}

    def digest(self, kind: str) -> str:
        """The sha256 of a record on disk (zeros when there is none)."""
        path = decision_path(self.runs, kind)
        return sha256(path) if path.exists() else "0" * 64

    def write(self, kind: str, record: dict) -> Path:
        return write_decision(self.runs, kind, record, repo=self.repo)

    def read(self, kind: str):
        return read_decision(self.runs, kind, repo=self.repo)

    def chain(self, candidate: int = 1, N: int = 75, *, through: str = "n", **simulation) -> None:
        """The valid records a later one answers: engineering, simulation (with the fields
        ``simulation`` replaces), n (and budget)."""
        self.write("engineering", self.engineering())
        self.write("simulation", self.simulation(**simulation))
        if through in ("n", "budget"):
            self.write(f"n-c{candidate}", self.n(candidate, N))
        if through == "budget":
            self.write(f"budget-c{candidate}", self.budget(candidate, N))

    def raw(self, kind: str, record) -> Path:
        """A record file as written by anything but ``write_decision``."""
        path = decision_path(self.runs, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        return path

    def change(self, kind: str, **fields) -> None:
        """Rewrite a record on disk with other fields: it is no longer what was named."""
        path = decision_path(self.runs, kind)
        path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))

    # -- the launch journal

    def start(self, header: dict, campaign: str = "study3-c1-validation") -> None:
        path = self.runs / "launch" / f"{campaign}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lines += 1
        line = {"event": "start", "experiment": "study3", "run": f"A-1.45e-05/s{self.lines}",
                "attempt": 1, "run_id": f"run-{campaign}-{self.lines}",
                "time": "2026-10-26T12:00:00+00:00", "header": header}  # fmt: skip
        with path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def names(
        self,
        kind: str,
        *,
        form: str = "reference",
        campaign: str = "study3-c1-validation",
        **header,
    ) -> str:
        """A campaign whose header names a record by its path and current sha256; the form is the
        reference mapping or the flat key beside its ``_sha256``."""
        path = decision_path(self.runs, kind)
        named = f"study3/decisions/{kind}.json"
        digest = sha256(path)
        head = {"study": 3, "candidate": 1, "step": "validation", "round": None, **header}
        if form == "reference":
            head["decision_records"] = {kind: {"path": named, "sha256": digest}}
        else:
            head["decision_record"], head["decision_record_sha256"] = named, digest
        self.start(head, campaign)
        return digest


# the cap N + max(10, ⌈0.12·N⌉) by hand
CAPS = {60: 70, 75: 85, 83: 93, 84: 95, 90: 101, 99: 111, 100: 112, 120: 135, 178: 200,
        179: 201}  # fmt: skip


def power_gate(value: float, n_min: int | None = None) -> dict:
    """The A-F power gate: short of its requirement 0.80 when ``value`` is below it, and then
    ``n_min`` is the smallest N at which the simulated power reaches it."""
    short = value < 0.80
    return {"value": value, "requirement": 0.80, "passed": not short, "power_short": short,
            "n_min": n_min if short else None}  # fmt: skip


def default_gates() -> dict:
    """The gates of a simulation that passed them all: a pass rate at the safe alternative of
    0.982, the power of 0.86 and the pass rate at study 2's empirical point, reported only."""
    return {"safe_pass_rate": {"value": 0.982, "requirement": 0.95, "passed": True},
            "power_af": power_gate(0.86),
            "empirical_pass_rate": {"value": 0.6, "requirement": None, "passed": None}}  # fmt: skip


def gates_with(**gates) -> dict:
    """The default gates with some replaced (a gate as ``name=entry``)."""
    return {**default_gates(), **gates}


def pass_rate(value: float) -> dict:
    """The gate of the pass rate at the safe alternative (requirement 0.95)."""
    return {"value": value, "requirement": 0.95, "passed": value >= 0.95}


def coverage_row(level: float, *, passed: bool = True, J: int = 1000, M: int = 10_000,
                 tests: int = 60) -> dict:  # fmt: skip
    c, j = coverage_threshold(J, M, 0.025), coverage_threshold(J, M, 0.05)
    counts = {"comparison": c - 20, "joint": j - 20}
    if not passed:
        counts["comparison"] = c + 1
    return {"level": level, "tests": tests, "J": J, "M": M, "counts": counts,
            "thresholds": {"comparison": c, "joint": j}, "passed": passed}  # fmt: skip


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path)


def refused(world: World, kind: str, record: dict, *words: str) -> None:
    """The record is refused on writing, and as a file it is a problem, naming the words."""
    with pytest.raises(DecisionRefused) as error:
        world.write(kind, record)
    assert all(w in str(error.value) for w in words), str(error.value)
    assert not decision_path(world.runs, kind).exists()
    world.raw(kind, record)
    found, problems = world.read(kind)
    assert found is None and problems
    assert all(any(w in p for p in problems) for w in words), problems
    decision_path(world.runs, kind).unlink()


def verified(level=0.975, passed=True, **kw) -> dict:
    """A simulation's independent verification."""
    return {"level": level, "passed": passed, "tests": 20, "calibration_stream": 1,
            "verification_stream": 2, **kw}  # fmt: skip


def sim_refused(world: World, *words: str, **over) -> None:
    refused(world, "simulation", world.simulation(**over), *words)


# ------------------------------------------------------------------ the registered values


def test_the_kinds_and_their_files(world) -> None:
    assert kind_of("engineering") == ("engineering", None)
    assert kind_of("simulation") == ("simulation", None)
    assert kind_of("n-c1") == ("n", 1) and kind_of("budget-c2") == ("budget", 2)
    for bad in ("n", "n-c3", "budget", "n-c1.json", "other", "N-c1", "", None):
        with pytest.raises(ValueError):
            kind_of(bad)
    assert decision_path(world.runs, "n-c2") == world.runs / "study3" / "decisions" / "n-c2.json"


def test_the_cap_is_n_plus_the_larger_of_ten_and_twelve_percent_rounded_up() -> None:
    for n, cap in CAPS.items():
        assert cap_of(n) == cap
    assert cap_of(83) == 83 + 10  # 0.12 * 83 = 9.96 -> 10
    assert cap_of(84) == 84 + 11  # 10.08 -> 11
    assert cap_of(75) == 75 + 10  # 9.0 exactly is not rounded up to 10: max(10, 9) = 10
    assert cap_of(100) == 100 + 12 and cap_of(125) == 125 + 15  # 15.0 exactly, not 16


def test_a_budget_decision_is_needed_above_ninety() -> None:
    assert [budget_needed(n) for n in (60, 90, 91, 99)] == [False, False, True, True]


def test_the_coverage_thresholds_are_the_binomial_quantiles_of_the_example() -> None:
    # "at J = 1 000, c = 313 per comparison and 587 jointly"
    assert coverage_threshold(1000, 10_000, 0.025) == 313
    assert coverage_threshold(1000, 10_000, 0.05) == 587


# ------------------------------------------------------------------ each schema accepted


def test_the_engineering_record_is_accepted_and_written_once(world) -> None:
    path = world.write("engineering", world.engineering())
    assert path == decision_path(world.runs, "engineering")
    record, problems = world.read("engineering")
    assert problems == [] and record["state"] == "passed"
    assert set(record["checks"]) == {"dry_run", "bridge", "theta0_utility", "stage0"}


def test_a_failed_engineering_record_may_lack_the_record_of_a_check(world) -> None:
    checks = world.engineering()["checks"]
    checks["bridge"] = None
    checks["stage0"]["passed"] = False
    world.write("engineering", world.engineering(state="failed", checks=checks))
    assert world.read("engineering")[0]["state"] == "failed"


def test_the_simulation_record_is_accepted(world) -> None:
    world.write("simulation", world.simulation())
    record, problems = world.read("simulation")
    assert problems == [] and record["level"] == 0.975 and record["validation_blocks"] == 28


def test_a_simulation_that_passed_a_later_level_and_chose_40_blocks(world) -> None:
    rows = [coverage_row(0.975, passed=False), coverage_row(0.980)]
    verification = verified(0.98)
    gates = gates_with(safe_pass_rate=pass_rate(0.93))
    world.write("simulation", world.simulation(coverage=rows, level=0.98, validation_blocks=40,
                                               verification=verification, gates=gates))  # fmt: skip
    record, problems = world.read("simulation")
    assert problems == [] and record["level"] == 0.98 and record["validation_blocks"] == 40
    assert record["state"] == "passed"  # the pass rate set the size; it did not fail the record


def test_a_simulation_where_no_level_passed_is_a_failed_record_with_a_null_level(world) -> None:
    rows = [coverage_row(level, passed=False) for level in LEVELS]
    verification = verified(None, False)
    world.write("simulation", world.simulation(state="failed", coverage=rows, level=None,
                                               validation_blocks=None,
                                               verification=verification))  # fmt: skip
    record, problems = world.read("simulation")
    assert problems == [] and record["state"] == "failed" and record["level"] is None


def test_a_failed_verification_is_a_failed_record(world) -> None:
    verification = verified(passed=False)
    world.write("simulation", world.simulation(state="failed", verification=verification))
    assert world.read("simulation")[0]["state"] == "failed"


def test_the_n_record_is_accepted_with_and_without_p(world) -> None:
    world.write("engineering", world.engineering())
    world.write("simulation", world.simulation())
    world.write("n-c1", world.n(1, 75))
    record, problems = world.read("n-c1")
    assert problems == [] and record["N"] == 75 and record["cap"] == 85
    only = world.n(2, 60, active=["A-F"])
    del only["rates"]["P"]
    world.write("n-c2", only)
    record, problems = world.read("n-c2")
    assert problems == [] and record["active"] == ["A-F"] and "P" not in record["rates"]


def test_the_budget_record_is_accepted_for_a_larger_n(world) -> None:
    world.chain(N=99, through="budget")
    record, problems = world.read("budget-c1")
    assert problems == [] and record["decision"] == "proceed" and record["cap"] == 111


def test_a_stop_is_a_failed_budget_record(world) -> None:
    world.chain(N=99)
    world.write("budget-c1", world.budget(1, 99, decision="stop", state="failed"))
    assert world.read("budget-c1")[0]["decision"] == "stop"


# ------------------------------------------------------------------ the common fields


@pytest.mark.parametrize("field", ["kind", "state", "written_at", "code", "checks"])
def test_a_missing_engineering_field_is_refused(world, field) -> None:
    record = world.engineering()
    del record[field]
    refused(world, "engineering", record, field)


@pytest.mark.parametrize("field", ["fixation", "inputs", "gates", "coverage", "level",
                                   "validation_blocks", "verification"])  # fmt: skip
def test_a_missing_simulation_field_is_refused(world, field) -> None:
    record = world.simulation()
    del record[field]
    refused(world, "simulation", record, field)


@pytest.mark.parametrize("field", ["candidate", "rates", "active", "level", "cohort", "spread",
                                   "spread_pp", "N", "cap"])  # fmt: skip
def test_a_missing_n_field_is_refused(world, field) -> None:
    world.chain(through="simulation")
    record = world.n()
    del record[field]
    refused(world, "n-c1", record, field)


@pytest.mark.parametrize("field", ["candidate", "N", "cap", "decision", "answers"])
def test_a_missing_budget_field_is_refused(world, field) -> None:
    world.chain(N=99)
    record = world.budget()
    del record[field]
    refused(world, "budget-c1", record, field)


@pytest.mark.parametrize(
    ("field", "value"),
    [("kind", "simulation"), ("state", "pending"), ("state", None), ("written_at", "yesterday"),
     ("written_at", "2026-10-26T12:00:00"), ("written_at", "2026-10-26T12:00:00+02:00"),
     ("written_at", 20261026), ("code", {"commit": "abc", "gpu_code_sha256": "b" * 64}),
     ("code", {"commit": "a" * 40}), ("code", {"commit": "a" * 40, "gpu_code_sha256": "b" * 63}),
     ("code", "a" * 40)],
)  # fmt: skip
def test_a_wrong_common_field_is_refused(world, field, value) -> None:
    refused(world, "engineering", world.engineering(**{field: value}), field)


def test_a_written_at_with_a_zero_offset_is_accepted(world) -> None:
    world.write("engineering", world.engineering(written_at="2026-10-26T12:00:00+00:00"))


def test_an_unknown_kind_is_a_programming_error(world) -> None:
    with pytest.raises(ValueError):
        world.read("n-c3")
    with pytest.raises(ValueError):
        world.write("budget", {})
    with pytest.raises(ValueError):
        decisions(world.runs, 3)


def test_an_absent_record_is_none_without_problems(world) -> None:
    assert world.read("engineering") == (None, [])


@pytest.mark.parametrize("text", ["{", "[]", "null", '{"kind": NaN}', "42"])
def test_an_unreadable_or_non_object_record_is_a_problem(world, text) -> None:
    path = decision_path(world.runs, "engineering")
    path.parent.mkdir(parents=True)
    path.write_text(text)
    found, problems = world.read("engineering")
    assert found is None and len(problems) >= 1 and str(path) in problems[0]


# ------------------------------------------------------------------ the fields no record carries


EXTRA = [
    "mean",
    "mean_pp",
    "sign",
    "per_seed",
    "per_seed_d",
    "sd_pp",
    "spread_unrounded",
    "e_eng",
    "path_max",
    "path_maxima",
]


@pytest.mark.parametrize("extra", EXTRA)
def test_no_mean_sign_per_seed_value_or_exploratory_diagnostic_is_a_field(world, extra) -> None:
    refused(world, "engineering", world.engineering(**{extra: 1.0}), extra, "not a field")
    refused(world, "simulation", world.simulation(**{extra: [0.1, 0.2]}), extra)
    world.write("simulation", world.simulation())
    refused(world, "n-c1", world.n(**{extra: -1.0}), extra)
    world.write("n-c1", world.n(N=99))
    refused(world, "budget-c1", world.budget(**{extra: 4.37}), extra)


def test_an_exploratory_diagnostic_inside_an_engineering_check_is_refused(world) -> None:
    record = world.engineering()
    record["checks"]["bridge"]["e_eng"] = 0.3
    refused(world, "engineering", record, "checks.bridge", "e_eng")
    record = world.engineering()
    record["checks"]["extra"] = record["checks"]["bridge"]
    refused(world, "engineering", record, "extra")


def test_a_per_seed_value_inside_the_cohort_or_rates_is_refused(world) -> None:
    world.chain(through="simulation")
    record = world.n()
    record["cohort"]["per_seed"] = [1.0]
    refused(world, "n-c1", record, "cohort", "per_seed")
    record = world.n()
    record["rates"]["A"]["mean"] = 1.0
    refused(world, "n-c1", record, "rates.A", "mean")


# ------------------------------------------------------------------ the spread, N and the cap


@pytest.mark.parametrize("spread", [4.37, 4.25, 4.51, 0.1, -0.5, 100.5, True, "4.5", None, [4.5]])
def test_an_unrounded_or_odd_spread_is_refused(world, spread) -> None:
    world.chain(through="simulation")
    refused(world, "n-c1", world.n(spread_pp=spread), "spread_pp")


@pytest.mark.parametrize("spread", [0, 0.0, 0.5, 4.5, 5, 5.0, 12.5, 100])
def test_a_spread_rounded_to_half_a_point_is_accepted(world, spread) -> None:
    world.chain(through="simulation")
    world.write("n-c1", world.n(spread_pp=spread))
    assert world.read("n-c1")[0]["spread_pp"] == spread


@pytest.mark.parametrize(
    ("n", "cap"),
    # the cap N + max(10, ⌈0.12·N⌉) of each, where it is a number: only N is wrong
    [(59, 69), (0, 10), (-60, -50), (60.0, 70), (61.5, 70), (True, 11), ("75", 85), (None, 70)],
)
def test_an_n_below_sixty_or_not_an_integer_is_refused(world, n, cap) -> None:
    world.chain(through="simulation")
    refused(world, "n-c1", world.n(N=n, cap=cap), "N", "at least 60")


def test_n_of_sixty_is_the_smallest(world) -> None:
    world.chain(through="simulation")
    world.write("n-c1", world.n(N=60))
    assert world.read("n-c1")[0]["cap"] == 70


@pytest.mark.parametrize(
    ("n", "cap"),
    [
        (60, 60 + 8),
        (60, 71),
        (75, 75 + 9),
        (75, 87),
        (90, 90 + 10),
        (99, 99 + 11),
        (99, 110.0),
        (99, None),
        (99, True),
        (75, "85"),
    ],
)
def test_a_cap_that_is_not_n_plus_the_larger_of_ten_and_twelve_percent_is_refused(
    world, n, cap
) -> None:
    world.chain(through="simulation")
    record = world.n(N=60, cap=70)
    record["N"], record["cap"] = n, cap
    refused(world, "n-c1", record, "cap")


@pytest.mark.parametrize("n", [90, 99, 100, 120])
def test_the_registered_caps_are_accepted(world, n) -> None:
    world.chain(through="simulation")
    world.write("n-c1", world.n(N=n))
    assert world.read("n-c1")[0]["cap"] == CAPS[n]


def test_a_cap_within_the_reserved_confirmation_seeds_is_a_passed_record(world) -> None:
    world.chain(through="simulation")
    # N = 178 -> cap 200: the last reserved seed is 10599, 200 seeds from 10400
    world.write("n-c1", world.n(N=178))
    record, problems = world.read("n-c1")
    assert problems == [] and record["cap"] == 200 and record["state"] == "passed"
    decision_path(world.runs, "n-c1").unlink()
    refused(world, "n-c1", world.n(N=178, state="failed"), "state", "fits", "200")


def test_a_cap_beyond_the_reserved_seeds_is_recorded_as_a_failed_record(world) -> None:
    world.chain(through="simulation")
    # N = 179 -> cap 201: the confirmation cannot start within 10400-10599, and N is not cut
    refused(world, "n-c1", world.n(N=179), "state", "exceeds", "201", "N is not cut")
    world.write("n-c1", world.n(N=179, state="failed"))
    record, problems = world.read("n-c1")
    assert problems == [] and record["N"] == 179 and record["cap"] == 201
    assert record["state"] == "failed"


def test_a_failed_n_record_needs_no_budget_entry(world) -> None:
    world.chain(through="simulation")
    world.write("n-c1", world.n(N=179, state="failed"))
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    found = decisions(world.runs, 1, repo=world.repo)
    assert found["n-c1"]["state"] == "failed" and found["n-c1"]["problems"] == []
    assert "budget-c1" not in found  # N is above 90, but nothing is to be confirmed


def test_the_candidate_of_the_record_is_the_kinds(world) -> None:
    world.chain(through="simulation")
    refused(world, "n-c1", world.n(candidate=2), "candidate")
    refused(world, "n-c1", world.n(candidate=True), "candidate")


def test_active_comparisons_and_the_arms_with_rates_agree(world) -> None:
    world.chain(through="simulation")
    refused(world, "n-c1", world.n(active=["P-F"]), "active")
    refused(world, "n-c1", world.n(active=["A-F", "P-F", "A-P"]), "active")
    refused(world, "n-c1", world.n(active="A-F"), "active")
    no_p = world.n(active=["A-F"])  # P still has rates: a closed arm's rates are not recorded
    refused(world, "n-c1", no_p, "rates", "P")
    with_p = world.n()
    del with_p["rates"]["P"]
    refused(world, "n-c1", with_p, "rates", "P")


@pytest.mark.parametrize("rate", [0, -1.0e-5, None, "1e-5", True])
def test_a_selected_rate_is_a_positive_finite_number(world, rate) -> None:
    world.chain(through="simulation")
    record = world.n()
    record["rates"]["F"]["lr_u"] = rate
    refused(world, "n-c1", record, "rates.F.lr_u")


def test_the_cohort_is_validation_seeds_of_the_candidate_with_at_least_four_runs_each(
    world,
) -> None:
    world.chain(through="simulation")
    first = SEEDS["validation"][1][0]
    cohort = world.n()["cohort"]
    refused(world, "n-c1", world.n(cohort={**cohort, "seeds": [10340 + i for i in range(25)]}),
            "cohort.seeds", "10340")  # fmt: skip
    refused(world, "n-c1", world.n(cohort={**cohort, "seeds": [first + 1, first]}),
            "cohort.seeds", "ascending")  # fmt: skip
    refused(world, "n-c1", world.n(cohort={**cohort, "seeds": []}), "cohort.seeds")
    refused(world, "n-c1", world.n(cohort={**cohort, "run_ids": cohort["run_ids"][:-1]}),
            "cohort.run_ids")  # fmt: skip
    refused(world, "n-c1", world.n(cohort={**cohort, "run_ids": cohort["run_ids"][:-1] * 2}),
            "cohort.run_ids", "twice")  # fmt: skip
    refused(world, "n-c1", world.n(cohort={**cohort, "run_ids": ["", *cohort["run_ids"][1:]]}),
            "cohort.run_ids")  # fmt: skip
    refused(world, "n-c1", world.n(cohort={"seeds": cohort["seeds"]}), "cohort", "run_ids")


# ------------------------------------------------------------------ the evidence


def test_an_evidence_file_changed_after_the_record_is_a_problem(world) -> None:
    world.chain(through="simulation")
    world.write("n-c1", world.n())
    record = json.loads(decision_path(world.runs, "n-c1").read_text())
    assert world.read("n-c1")[1] == []
    for ref in (record["spread"]["script"], record["spread"]["output"]):
        path = world.file(ref)
        before = path.read_bytes()
        path.write_bytes(before + b" ")
        found, problems = world.read("n-c1")
        assert found is None
        assert any(str(path) in p and "sha256" in p and ref["sha256"] in p for p in problems)
        path.write_bytes(before)
        assert world.read("n-c1")[1] == []


@pytest.mark.parametrize("where", ["fixation", "inputs.seed_vectors", "inputs.design"])
def test_a_simulation_evidence_file_changed_is_a_problem(world, where) -> None:
    record = world.simulation()
    world.write("simulation", record)
    ref = record["fixation"] if where == "fixation" else record["inputs"][where.split(".")[1]]
    world.file(ref).write_text("changed\n")
    found, problems = world.read("simulation")
    assert found is None and any(where in p and "sha256" in p for p in problems)


@pytest.mark.parametrize("check", ["dry_run", "bridge", "theta0_utility", "stage0"])
def test_an_engineering_evidence_file_changed_or_missing_is_a_problem(world, check) -> None:
    record = world.engineering()
    world.write("engineering", record)
    path = world.file(record["checks"][check])
    path.write_text("changed\n")
    found, problems = world.read("engineering")
    assert found is None and any(f"checks.{check}" in p and "sha256" in p for p in problems)
    path.unlink()
    found, problems = world.read("engineering")
    assert found is None and any(f"checks.{check}" in p and "missing" in p for p in problems)


def test_an_evidence_reference_must_be_well_formed(world) -> None:
    for bad in ({"path": "evidence/x.json"}, {"sha256": "a" * 64}, "evidence/x.json", None,
                {"path": "", "sha256": "a" * 64}, {"path": "evidence/x.json", "sha256": "A" * 64},
                {"path": "evidence/x.json", "sha256": "a" * 63},
                {**world.ref("fixation.json"), "root": "elsewhere"},
                {**world.ref("fixation.json"), "size": 3}):  # fmt: skip
        refused(world, "simulation", world.simulation(fixation=bad), "fixation")


@pytest.mark.parametrize("sha", ["A" * 64, "a" * 63, "a" * 65, "g" * 64, "", 7, None])
def test_an_evidence_sha256_is_64_lowercase_hexadecimal_digits(world, sha) -> None:
    ref = {"path": "evidence/x.json", "sha256": sha}
    refused(world, "simulation", world.simulation(fixation=ref), "fixation.sha256", "not a sha256")


def test_evidence_is_found_by_its_root_and_an_absolute_path_needs_none(world) -> None:
    ref = world.ref("both.json", "same\n")
    in_repo = {**ref, "root": "repo"}
    refused(world, "simulation", world.simulation(fixation=in_repo), "fixation", "missing")
    absolute = {"path": str(world.file(ref)), "sha256": ref["sha256"]}
    world.write("simulation", world.simulation(fixation=absolute))
    assert world.read("simulation")[1] == []


def test_evidence_under_responses_or_observations_is_refused_and_never_opened(
    world, monkeypatch
) -> None:
    attempts = seal(monkeypatch)
    for folder, name in (("responses", "harmful_evaluation.jsonl"),
                         ("observations", "measurements.jsonl")):  # fmt: skip
        sealed = world.run / folder / name
        relative = sealed.relative_to(world.runs)
        for path in (str(relative), str(sealed), f"{relative.parent}/../{folder}/{name}",
                     f"{relative.parent.parent}/{folder}/{name}"):  # fmt: skip
            record = world.simulation(fixation={"path": path, "sha256": "a" * 64})
            refused(world, "simulation", record, "fixation", "never opened")
    refused(world, "simulation", world.simulation(inputs={"x": {"path": "responses/a.json",
                                                                "sha256": "a" * 64}}),
            "inputs.x", "never opened")  # fmt: skip
    assert attempts == []


def test_a_link_into_a_sealed_folder_is_refused_and_never_opened(world, monkeypatch) -> None:
    attempts = seal(monkeypatch)
    link = world.runs / "evidence" / "link.jsonl"
    link.parent.mkdir(parents=True)
    link.symlink_to(world.run / "responses" / "harmful_evaluation.jsonl")
    world.raw("simulation", world.simulation(fixation={"path": "evidence/link.jsonl",
                                                       "sha256": "a" * 64}))  # fmt: skip
    found, problems = world.read("simulation")
    assert found is None and any("never opened" in p for p in problems)
    assert attempts == []


# ------------------------------------------------------------------ the engineering record


def test_a_passed_engineering_record_rests_on_every_check(world) -> None:
    record = world.engineering()
    record["checks"]["stage0"] = None
    refused(world, "engineering", record, "checks.stage0", "null")
    record = world.engineering()
    record["checks"]["bridge"]["passed"] = False
    refused(world, "engineering", record, "checks.bridge", "did not pass")
    record = world.engineering()
    del record["checks"]["theta0_utility"]
    refused(world, "engineering", record, "theta0_utility")
    record = world.engineering()
    record["checks"]["dry_run"]["passed"] = "yes"
    refused(world, "engineering", record, "checks.dry_run.passed")


# ------------------------------------------------------------------ the simulation record


def test_a_gate_below_its_registered_requirement_or_inconsistent_is_refused(world) -> None:
    weak = gates_with(safe_pass_rate={"value": 0.96, "requirement": 0.90, "passed": True})
    sim_refused(world, "safe_pass_rate", "0.95", gates=weak)
    weak = gates_with(power_af={**power_gate(0.80), "requirement": 0.70})
    sim_refused(world, "power_af", "0.8", gates=weak)
    lying = gates_with(power_af={**power_gate(0.75, 80), "passed": True})
    sim_refused(world, "power_af", "passed=True", gates=lying)
    lying = gates_with(safe_pass_rate={"value": 0.94, "requirement": 0.95, "passed": True})
    sim_refused(world, "safe_pass_rate", "passed=True", gates=lying)
    sim_refused(world, "power_af", gates={k: v for k, v in default_gates().items()
                                          if k != "power_af"})  # fmt: skip
    sim_refused(world, "safe_pass_rate", gates=gates_with(
        safe_pass_rate={"value": 0.99, "requirement": None, "passed": None}))  # fmt: skip
    sim_refused(world, "empirical_pass_rate", "null", gates=gates_with(
        empirical_pass_rate={"value": 0.6, "requirement": None, "passed": True}))  # fmt: skip
    sim_refused(world, "gates", gates=[])
    sim_refused(world, "gates.safe_pass_rate", "power_short",
                gates=gates_with(safe_pass_rate={**pass_rate(0.97), "power_short": False,
                                                 "n_min": None}))  # fmt: skip


def test_a_power_shortfall_is_recorded_and_the_record_still_passes(world) -> None:
    gates = gates_with(power_af=power_gate(0.74, 84))
    world.write("simulation", world.simulation(gates=gates))
    record, problems = world.read("simulation")
    assert problems == [] and record["state"] == "passed"
    power = record["gates"]["power_af"]
    assert power["power_short"] is True and power["n_min"] == 84 and power["passed"] is False
    assert record["validation_blocks"] == 28  # the pass rate decided that, not the power


@pytest.mark.parametrize(
    ("power", "words"),
    [
        ({**power_gate(0.74, 84), "power_short": False}, ("power_short", "requirement is not met")),
        ({**power_gate(0.86), "power_short": True, "n_min": 84}, ("power_short",)),
        ({**power_gate(0.74, 84), "n_min": None}, ("n_min",)),
        ({**power_gate(0.74, 84), "n_min": 59}, ("n_min", "at least 60")),
        ({**power_gate(0.74, 84), "n_min": 84.0}, ("n_min",)),
        ({**power_gate(0.74, 84), "n_min": True}, ("n_min",)),
        ({**power_gate(0.86), "n_min": 84}, ("n_min", "null")),
        ({**power_gate(0.74, 84), "power_short": "yes"}, ("power_short",)),
        ({"value": 0.74, "requirement": 0.80, "passed": False}, ("power_short",)),
        ({"value": 0.74, "requirement": 0.80, "passed": False, "power_short": True}, ("n_min",)),
    ],
)
def test_a_power_shortfall_is_recorded_consistently(world, power, words) -> None:
    sim_refused(world, "gates.power_af", *words, gates=gates_with(power_af=power))


def test_a_pass_rate_below_its_requirement_sets_the_40_block_validation(world) -> None:
    low = gates_with(safe_pass_rate=pass_rate(0.93))
    # 28 blocks with a pass rate below 0.95 is refused; 40 blocks is the record, still passed
    sim_refused(world, "validation_blocks", "40", gates=low, validation_blocks=28)
    world.write("simulation", world.simulation(gates=low, validation_blocks=40))
    assert world.read("simulation")[0]["state"] == "passed"
    decision_path(world.runs, "simulation").unlink()
    # 40 blocks with a pass rate that meets it (0.95 exactly does) is refused: 28 otherwise
    for rate in (0.95, 0.982):
        high = gates_with(safe_pass_rate=pass_rate(rate))
        sim_refused(world, "validation_blocks", "28", gates=high, validation_blocks=40)
    world.write("simulation", world.simulation(gates=gates_with(safe_pass_rate=pass_rate(0.95))))
    assert world.read("simulation")[0]["validation_blocks"] == 28


def test_the_simulation_state_follows_the_coverage_screen_and_its_verification_only(world) -> None:
    # a power shortfall and a pass rate below 0.95 do not make a failed record
    gates = gates_with(power_af=power_gate(0.74, 84), safe_pass_rate=pass_rate(0.93))
    sim_refused(world, "state", "coverage screen", state="failed", gates=gates,
                validation_blocks=40)  # fmt: skip
    sim_refused(world, "state", "coverage screen", state="failed")
    # a failed screen or verification is a failed record whatever the gates say
    rows = [coverage_row(level, passed=False) for level in LEVELS]
    world.write("simulation", world.simulation(state="failed", coverage=rows, level=None,
                                               validation_blocks=None, verification=verified(
                                                   None, False)))  # fmt: skip
    decision_path(world.runs, "simulation").unlink()
    world.write("simulation", world.simulation(state="failed", verification=verified(passed=False)))


def test_the_coverage_threshold_is_the_recomputed_one(world) -> None:
    row = coverage_row(0.975)
    weaker = {**row, "thresholds": {"comparison": 400, "joint": 587}}
    sim_refused(world, "thresholds.comparison", "313", coverage=[weaker])
    sim_refused(world, "thresholds", coverage=[{**row, "J": 2000}])  # c of J = 1000 do not follow
    sim_refused(world, "coverage[0].M", coverage=[{**row, "M": 5000}])
    floats = {"comparison": 313.0, "joint": 587}
    sim_refused(world, "thresholds", coverage=[{**row, "thresholds": floats}])


def test_a_levels_pass_agrees_with_its_counts(world) -> None:
    row = coverage_row(0.975)
    over = {**row, "counts": {"comparison": 314, "joint": 100}}
    sim_refused(world, "coverage[0].passed", coverage=[over])
    under = {**coverage_row(0.975, passed=False), "counts": {"comparison": 10, "joint": 10}}
    sim_refused(world, "coverage[0].passed", coverage=[under])
    sim_refused(world, "counts.comparison", coverage=[{**row, "counts": {"comparison": -1,
                                                                         "joint": 3}}])  # fmt: skip
    sim_refused(world, "tests", coverage=[{**row, "tests": 1001}])


def test_the_level_is_the_first_whose_tests_passed_and_the_levels_below_were_tested(world) -> None:
    both, first, second = [coverage_row(0.975), coverage_row(0.980)], 0.975, 0.980
    one_failed = [coverage_row(first, passed=False), coverage_row(second)]
    # 0.975 failed, 0.980 passed: L is 0.980, not 0.975
    sim_refused(world, "level", "0.98", coverage=one_failed)
    # a level above the first that passed is not L
    sim_refused(world, "level", "0.975", coverage=both, level=0.98, verification=verified(0.98))
    # 0.975 was never tested
    sim_refused(world, "every level", coverage=[coverage_row(second)], level=0.98,
                verification=verified(0.98))  # fmt: skip
    # no such level in the list
    sim_refused(world, "level", level=0.99, verification=verified(0.99))
    # a level off the ladder
    sim_refused(world, "level", level=0.97)
    sim_refused(world, "coverage[0].level", coverage=[coverage_row(0.97)])
    # levels listed out of order, or twice
    sim_refused(world, "ascending", coverage=[coverage_row(second), coverage_row(first)])
    sim_refused(world, "ascending", coverage=[coverage_row(first), coverage_row(first)])
    # a null level with a level that passed, or without every level tested
    nothing = verified(None, False)
    sim_refused(world, "null", state="failed", level=None, verification=nothing)
    failed = [coverage_row(first, passed=False), coverage_row(second, passed=False)]
    sim_refused(world, "every level", state="failed", level=None, coverage=failed,
                verification=nothing)  # fmt: skip
    sim_refused(world, "coverage", coverage=[])


def test_one_j_is_frozen_for_every_level_so_a_level_cannot_pass_on_a_raised_j(world) -> None:
    # 0.975 fails at J = 1000; 0.980 passes with a count of 314, which fails at J = 1000 (c = 313),
    # under a J declared as 10**6 (c = 333)
    late = coverage_row(0.980, J=10**6)
    late["counts"] = {"comparison": 314, "joint": 100}
    assert coverage_threshold(1000, 10_000, 0.025) == 313 < 314 <= late["thresholds"]["comparison"]
    rows = [coverage_row(0.975, passed=False), late]
    sim_refused(world, "coverage[1].J", "one frozen number", coverage=rows, level=0.98,
                verification=verified(0.98))  # fmt: skip
    # the same level at the J of the first: it fails, as it should
    sim_refused(world, "coverage[1].passed", coverage=[rows[0], {**late, "J": 1000,
                "thresholds": {"comparison": 313, "joint": 587}}], level=0.98,
                verification=verified(0.98))  # fmt: skip


def test_the_tests_of_the_levels_and_the_verification_fit_within_the_frozen_j(world) -> None:
    # four levels of 900 tests each under J = 1000
    rows = [coverage_row(level, passed=level == 0.990, tests=900) for level in LEVELS]
    sim_refused(world, "coverage", "add up", "J = 1000", coverage=rows, level=0.99,
                verification=verified(0.99))  # fmt: skip
    # the sum equal to J is accepted, one more is not
    one = [coverage_row(0.975, tests=980)]
    world.write("simulation", world.simulation(coverage=one, verification=verified(tests=20)))
    assert world.read("simulation")[1] == []
    decision_path(world.runs, "simulation").unlink()
    sim_refused(world, "coverage", "1001", "J = 1000", coverage=one,
                verification=verified(tests=21))  # fmt: skip
    sim_refused(world, "verification.tests", verification=verified(tests=0))
    sim_refused(world, "verification.tests", verification=verified(tests=1.5))
    sim_refused(world, "verification", "tests",
                verification={"level": 0.975, "passed": True, "calibration_stream": 1,
                              "verification_stream": 2})  # fmt: skip


@pytest.mark.parametrize("m", [9999, 10_001, 20_000, 100, 10_000.0, True, None])
def test_m_is_exactly_ten_thousand_on_every_row(world, m) -> None:
    rows = [coverage_row(0.975, passed=False), {**coverage_row(0.980), "M": m}]
    sim_refused(world, "coverage[1].M", coverage=rows, level=0.98, verification=verified(0.98))
    if isinstance(m, int) and not isinstance(m, bool):
        sim_refused(world, "coverage[0].M", coverage=[{**coverage_row(0.975), "M": m}])


def test_the_reported_gates_are_accepted_and_no_other_gate_is_a_name(world) -> None:
    reported = {"value": 0.7, "requirement": None, "passed": None}
    gates = gates_with(power_pf=reported, detection_power_comparison=reported,
                       detection_power_joint=reported)  # fmt: skip
    world.write("simulation", world.simulation(gates=gates))
    assert set(world.read("simulation")[0]["gates"]) == {
        "safe_pass_rate", "power_af", "empirical_pass_rate", "power_pf",
        "detection_power_comparison", "detection_power_joint"}  # fmt: skip


GATE_NAMES = [
    "mean_af",
    "per_seed_d",
    "e_eng",
    "sign_af",
    "path_max",
    "power_ap",
    "Power_pf",
    "empirical_pass_rate_2",
    "",
    "gates.x",
]


@pytest.mark.parametrize("name", GATE_NAMES)
def test_a_gate_with_another_name_is_refused(world, name) -> None:
    free = {"value": 0.1, "requirement": None, "passed": None}
    sim_refused(world, f"gates.{name}", "not a gate", gates=gates_with(**{name: free}))
    with_requirement = {"value": 0.9, "requirement": 0.1, "passed": True}  # a gate that gates
    sim_refused(world, f"gates.{name}", "not a gate", gates=gates_with(**{name: with_requirement}))


@pytest.mark.parametrize("name", ["empirical_pass_rate", "power_pf", "detection_power_comparison",
                                  "detection_power_joint"])  # fmt: skip
def test_a_reported_gate_has_no_requirement(world, name) -> None:
    sim_refused(
        world,
        f"gates.{name}.requirement",
        "only reported",
        gates=gates_with(**{name: {"value": 0.9, "requirement": 0.5, "passed": True}}),
    )
    sim_refused(
        world,
        f"gates.{name}.passed",
        "null",
        gates=gates_with(**{name: {"value": 0.9, "requirement": None, "passed": False}}),
    )
    sim_refused(world, f"gates.{name}", "power_short",
                gates=gates_with(**{name: {**power_gate(0.7, 80), "requirement": None,
                                           "passed": None}}))  # fmt: skip


@pytest.mark.parametrize(
    "name",
    ["mean_spread_4.37", "mean_spread", "per_seed_vectors", "sd_inputs", "spread_input", "e_eng",
     "path_max", "unrounded_values", "signed_diff", "sign", "signs", "signs_af", "signflip",
     "dsign", "unsigned", "signum", "Upper", "digit1", "_lead", "trail_",
     "has space", "a-b", "", "x" * 41, "caf\u00e9"],
)  # fmt: skip
def test_an_input_name_is_a_plain_word_that_marks_no_hidden_value(world, name) -> None:
    inputs = {"seed_vectors": world.ref("seed-vectors.json"), name: world.ref("extra.json")}
    sim_refused(world, "inputs", "is not a lowercase word", inputs=inputs)


@pytest.mark.parametrize("name", ["a", "ab", "seed_vectors", "design", "grid_of_levels", "x" * 40,
                                  "a_" * 19 + "ab"])  # fmt: skip
def test_the_input_names_of_a_simulation_that_are_plain_words_are_accepted(world, name) -> None:
    inputs = {"seed_vectors": world.ref("seed-vectors.json"), name: world.ref("extra.json")}
    world.write("simulation", world.simulation(inputs=inputs))
    assert name in world.read("simulation")[0]["inputs"]


def test_a_passed_simulation_needs_a_passed_verification_of_its_level(world) -> None:
    sim_refused(world, "verification", "did not pass", verification=verified(passed=False))
    sim_refused(world, "verification.level", verification=verified(0.98))
    sim_refused(world, "verification.level", verification=verified(None))
    sim_refused(world, "verification.level", state="failed", level=None, verification=verified(),
                coverage=[coverage_row(level, passed=False) for level in LEVELS])  # fmt: skip
    same = verified(calibration_stream=5, verification_stream=5)
    sim_refused(world, "independent", verification=same)
    sim_refused(world, "verification.passed", verification=verified(passed="yes"))
    sim_refused(
        world,
        "verification",
        "passed",
        verification={
            "level": 0.975,
            "tests": 20,
            "calibration_stream": 1,
            "verification_stream": 2,
        },
    )
    sim_refused(world, "calibration_stream", state="failed",
                verification=verified(calibration_stream=1.5))  # fmt: skip
    sim_refused(world, "verification.passed", "no level", state="failed", level=None,
                coverage=[coverage_row(level, passed=False) for level in LEVELS],
                verification=verified(None, True))  # fmt: skip


@pytest.mark.parametrize("blocks", [27, 29, 0, 28.0, "28", None, True])
def test_the_validation_size_is_28_or_40(world, blocks) -> None:
    refused(world, "simulation", world.simulation(validation_blocks=blocks), "validation_blocks")


def test_a_passed_simulation_has_a_level_a_size_and_inputs(world) -> None:
    refused(world, "simulation", world.simulation(level=None), "level", "passed record")
    refused(world, "simulation", world.simulation(inputs={}), "inputs")
    refused(world, "simulation", world.simulation(inputs=[world.ref("a.json")]), "inputs")


# ------------------------------------------------------------------ the records answered


def test_an_n_record_needs_a_valid_passed_simulation_of_its_level(world) -> None:
    refused(world, "n-c1", world.n(), "simulation record", "missing")
    world.raw("simulation", {"kind": "simulation"})
    refused(world, "n-c1", world.n(), "simulation record", "not valid")
    decision_path(world.runs, "simulation").unlink()
    world.write("simulation", world.simulation())
    refused(world, "n-c1", world.n(level=0.98), "L is 0.98", "0.975")
    refused(world, "n-c1", world.n(level=0.97), "level")


def test_an_n_record_after_a_failed_simulation_is_refused(world) -> None:
    verification = verified(passed=False)
    world.write("simulation", world.simulation(state="failed", verification=verification))
    refused(world, "n-c1", world.n(), "simulation record did not pass")


def test_the_cohort_fits_the_validation_size_the_simulation_chose(world) -> None:
    world.write("simulation", world.simulation())  # 28 blocks, 22 supported at least
    first = SEEDS["validation"][1][0]

    def cohort(seeds):
        return {"seeds": list(seeds), "run_ids": [f"r{s}-{k}" for s in seeds for k in range(4)]}

    world.write("n-c1", world.n(cohort=cohort(range(first, first + 22))))  # the minimum
    decision_path(world.runs, "n-c1").unlink()
    refused(world, "n-c1", world.n(cohort=cohort(range(first, first + 21))), "21 blocks", "22")
    refused(world, "n-c1", world.n(cohort=cohort(range(first + 5, first + 5 + 25))), "beyond",
            "28")  # fmt: skip
    refused(world, "n-c1", world.n(cohort=cohort(range(first + 8, first + 8 + 21))), "beyond")
    # a 40-block validation reaches seed first + 39 and needs 31 supported
    decision_path(world.runs, "simulation").unlink()
    low = gates_with(safe_pass_rate=pass_rate(0.93))
    world.write("simulation", world.simulation(gates=low, validation_blocks=40))
    refused(world, "n-c1", world.n(cohort=cohort(range(first, first + 30))), "31")
    world.write("n-c1", world.n(cohort=cohort(range(first + 9, first + 40))))
    assert world.read("n-c1")[1] == []


def test_a_budget_record_answers_the_n_and_cap_of_its_n_record(world) -> None:
    world.chain(N=99)

    def budget_refused(*words, **over) -> None:
        refused(world, "budget-c1", world.budget(**over), *words)

    budget_refused("N 100", "99", N=100)  # another N
    budget_refused("cap 112", "111", N=99, cap=112)
    budget_refused("cap", N=99, cap=110)
    budget_refused("N", N=59, cap=69)
    budget_refused("candidate", candidate=2)
    budget_refused("decision", decision="maybe")
    budget_refused("state", "stop", decision="stop")
    budget_refused("state", decision="proceed", state="failed")
    other = {"path": "study3/decisions/n-c2.json", "sha256": "a" * 64}
    budget_refused("answers", "n-c1.json", answers=other)
    stale = {"path": "study3/decisions/n-c1.json", "sha256": "0" * 64}
    budget_refused("answers", "sha256", answers=stale)


def test_a_budget_record_needs_its_n_record(world) -> None:
    refused(world, "budget-c1", world.budget(N=99), "n-c1 record", "missing")


def test_a_budget_record_goes_stale_when_the_n_record_it_answers_changes(world) -> None:
    world.chain(N=99, through="budget")
    world.change("n-c1", spread_pp=5.0)
    found, problems = world.read("budget-c1")
    assert found is None and any("answers" in p and "sha256" in p for p in problems)


SHORT = gates_with(power_af=power_gate(0.74, 84))  # the power is short; N of 84 reaches it


def test_an_n_record_below_n_min_is_refused_and_one_at_or_above_it_is_accepted(world) -> None:
    world.write("simulation", world.simulation(gates=SHORT))
    refused(world, "n-c1", world.n(N=75), "N is 75", "n_min 84")
    refused(world, "n-c1", world.n(N=83, cap=93), "N is 83", "n_min 84")
    refused(world, "n-c1", world.n(N=60), "n_min 84")
    world.write("n-c1", world.n(N=84, cap=cap_of(84)))
    decision_path(world.runs, "n-c1").unlink()
    world.write("n-c1", world.n(N=90))
    assert world.read("n-c1")[1] == []


def test_a_simulation_whose_power_is_not_short_has_no_floor_on_n(world) -> None:
    world.write("simulation", world.simulation())
    world.write("n-c1", world.n(N=60))
    assert world.read("n-c1")[1] == []


def test_the_budget_reason_is_n_above_ninety_or_n_raised_for_power() -> None:
    assert budget_reason(90) is None and budget_reason(60, {"gates": default_gates()}) is None
    assert "above 90" in budget_reason(91)
    short = {"gates": SHORT}
    assert "84" in budget_reason(84, short) and "above" not in budget_reason(84, short)
    both = budget_reason(99, short)
    assert "above 90" in both and "84" in both
    assert budget_needed(84, short) and not budget_needed(84) and budget_needed(91)
    assert budget_reason(84, {}) is None and budget_reason(84, {"gates": {}}) is None


def test_an_n_raised_for_power_needs_a_budget_record_even_at_or_below_ninety(world) -> None:
    world.chain(N=84, through="n", gates=SHORT)
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    found = decisions(world.runs, 1, repo=world.repo)
    assert list(found) == ["engineering", "simulation", "n-c1", "budget-c1"]
    assert found["n-c1"]["state"] == "passed" and found["n-c1"]["problems"] == []
    assert found["budget-c1"]["state"] == "missing" and found["budget-c1"]["record"] is None
    [problem] = found["budget-c1"]["problems"]
    assert "budget decision" in problem and "n-c1" in problem and "84" in problem
    world.write("budget-c1", world.budget(N=84, cap=cap_of(84)))
    found = decisions(world.runs, 1, repo=world.repo)
    assert found["budget-c1"]["state"] == "passed" and found["budget-c1"]["problems"] == []
    # without the shortfall the same N needs none
    other = World(world.runs.parent / "other")
    other.chain(N=84)
    other.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    assert list(decisions(other.runs, 1, repo=other.repo)) == ["engineering", "simulation", "n-c1"]


def test_the_answers_reference_is_the_n_record_under_the_runs_directory_and_no_other(world) -> None:
    world.chain(N=99)
    n_path = decision_path(world.runs, "n-c1")
    digest = sha256(n_path)
    # the same bytes, kept elsewhere, or named by an absolute path or another root
    copy = world.runs / "evidence" / "n-c1.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(n_path.read_bytes())
    in_repo = world.repo / "study3" / "decisions" / "n-c1.json"  # the same file, in the checkout
    in_repo.parent.mkdir(parents=True)
    in_repo.write_bytes(n_path.read_bytes())
    wrong = [
        {"path": "evidence/n-c1.json", "sha256": digest},
        {"path": "study3/decisions/n-c1.json", "sha256": digest, "root": "repo"},
        {"path": str(n_path), "sha256": digest},
        {"path": "other/study3/decisions/n-c1.json", "sha256": digest},
        {"path": "study3/decisions/../decisions/n-c1.json", "sha256": digest},
    ]
    for answers in wrong:
        refused(world, "budget-c1", world.budget(answers=answers), "answers", "n-c1.json")
    for root in ({}, {"root": "runs"}):
        record = world.budget(answers={"path": "study3/decisions/n-c1.json", "sha256": digest,
                                       **root})  # fmt: skip
        world.write("budget-c1", record)
        assert world.read("budget-c1")[1] == []
        decision_path(world.runs, "budget-c1").unlink()


def test_an_n_record_rests_on_the_exact_simulation_record(world) -> None:
    world.chain(through="simulation")
    sim = decision_path(world.runs, "simulation")
    digest = sha256(sim)
    copy = world.runs / "evidence" / "simulation.json"
    copy.parent.mkdir(parents=True, exist_ok=True)
    copy.write_bytes(sim.read_bytes())
    in_repo = world.repo / "study3" / "decisions" / "simulation.json"  # the same, in the checkout
    in_repo.parent.mkdir(parents=True)
    in_repo.write_bytes(sim.read_bytes())
    wrong = [
        {"path": "evidence/simulation.json", "sha256": digest, "root": "runs"},
        {"path": "study3/decisions/simulation.json", "sha256": digest, "root": "repo"},
        {"path": str(sim), "sha256": digest, "root": "runs"},
        {"path": "study3/decisions/simulation.json", "sha256": "0" * 64, "root": "runs"},
    ]
    for reference in wrong:
        refused(world, "n-c1", world.n(simulation=reference), "simulation")
    for reference in ({"path": "study3/decisions/simulation.json", "sha256": digest},
                      {"path": "study3/decisions/simulation.json", "sha256": digest,
                       "root": "runs"}):  # fmt: skip
        world.write("n-c1", world.n(simulation=reference))
        decision_path(world.runs, "n-c1").unlink()
    refused(world, "n-c1", {k: v for k, v in world.n().items() if k != "simulation"}, "simulation")


def test_an_n_record_goes_stale_when_the_simulation_it_rests_on_changes(world) -> None:
    world.chain(through="n")
    assert world.read("n-c1")[1] == []
    world.change("simulation", written_at="2026-10-26T13:00:00Z")  # still a valid record
    assert world.read("simulation")[1] == []
    found, problems = world.read("n-c1")
    assert found is None and any("simulation" in p and "sha256" in p for p in problems)


def test_a_budget_record_for_a_small_n_is_valid_and_not_needed(world) -> None:
    world.chain(N=90)
    world.write("budget-c1", world.budget(N=90))
    assert world.read("budget-c1")[1] == [] and not budget_needed(90)


# ------------------------------------------------------------------ writing


def test_a_second_write_is_refused_whatever_it_holds(world) -> None:
    path = world.write("engineering", world.engineering())
    before = path.read_bytes()
    with pytest.raises(DecisionRefused, match="already written"):
        world.write("engineering", world.engineering(state="failed"))
    with pytest.raises(DecisionRefused, match="already written"):
        world.write("engineering", world.engineering())
    with pytest.raises(DecisionRefused, match="already written"):
        world.write("engineering", {})
    assert path.read_bytes() == before


def test_a_record_that_exists_is_never_replaced_even_when_another_process_wrote_it_first(
    world, monkeypatch
) -> None:
    real = os.link

    def racing(source, target, *args, **kwargs):
        Path(target).write_text("first\n")  # another writer's record appears just before the link
        return real(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", racing)
    with pytest.raises(DecisionRefused, match="already written"):
        world.write("engineering", world.engineering())
    assert decision_path(world.runs, "engineering").read_text() == "first\n"
    assert [p.name for p in decision_path(world.runs, "engineering").parent.iterdir()] == [
        "engineering.json"
    ]


def test_a_filesystem_without_hard_links_still_gets_the_record_once(world, monkeypatch) -> None:
    def no_links(*args, **kwargs):
        raise OSError("no hard links here")

    monkeypatch.setattr(os, "link", no_links)
    path = world.write("engineering", world.engineering())
    assert world.read("engineering")[1] == []
    with pytest.raises(DecisionRefused, match="already written"):
        world.write("engineering", world.engineering())
    assert [p.name for p in path.parent.iterdir()] == ["engineering.json"]


def test_an_invalid_record_leaves_nothing_behind(world) -> None:
    with pytest.raises(DecisionRefused):
        world.write("engineering", world.engineering(state="maybe"))
    assert not decision_path(world.runs, "engineering").parent.exists() or not list(
        decision_path(world.runs, "engineering").parent.iterdir()
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), {1, 2}, object()])
def test_a_record_that_is_not_json_is_refused(world, bad) -> None:
    with pytest.raises(DecisionRefused, match="not a JSON record"):
        world.write("engineering", world.engineering(extra_note=bad))


def test_the_record_is_written_as_sorted_json_whose_sha256_is_the_files(world) -> None:
    path = world.write("engineering", world.engineering())
    text = path.read_text()
    assert text.endswith("}\n") and json.loads(text)["kind"] == "engineering"
    assert list(json.loads(text)) == sorted(json.loads(text))
    assert world.read("engineering")[0] == json.loads(text)


def test_a_record_written_from_tuples_reads_back_as_lists(world) -> None:
    world.chain(through="simulation")
    record = world.n(active=("A-F", "P-F"))
    record["cohort"]["seeds"] = tuple(record["cohort"]["seeds"])
    world.write("n-c1", record)
    assert world.read("n-c1")[0]["active"] == ["A-F", "P-F"]


# ------------------------------------------------------------------ immutability


def test_a_record_no_campaign_named_is_free_of_problems(world) -> None:
    world.write("engineering", world.engineering())
    path = decision_path(world.runs, "engineering")
    assert immutable(path, []) == []
    world.start({"study": 3, "candidate": 1, "step": "search", "round": 0})
    assert immutable(path, journal_of(world)) == []


def journal_of(world: World):
    """The journal's starts as ``study3_records`` reads them (campaign, start)."""
    out = []
    for log in sorted((world.runs / "launch").glob("*.jsonl")):
        out += [(log.stem, json.loads(line)) for line in log.read_text().splitlines()]
    return out


@pytest.mark.parametrize("form", ["reference", "flat"])
def test_a_record_that_matches_what_a_campaign_named_is_free_of_problems(world, form) -> None:
    world.write("engineering", world.engineering())
    world.names("engineering", form=form)
    assert immutable(decision_path(world.runs, "engineering"), journal_of(world)) == []
    assert world.read("engineering")[1] == []


@pytest.mark.parametrize("form", ["reference", "flat"])
def test_a_record_changed_after_a_campaign_named_it_is_a_problem(world, form) -> None:
    world.write("engineering", world.engineering())
    named = world.names("engineering", form=form)
    world.change("engineering", state="failed")
    path = decision_path(world.runs, "engineering")
    problems = immutable(path, journal_of(world))
    assert len(problems) == 1
    assert (
        named in problems[0]
        and sha256(path) in problems[0]
        and "study3-c1-validation" in problems[0]
    )


@pytest.mark.parametrize("form", ["reference", "flat"])
def test_decisions_marks_a_changed_record_invalid(world, form) -> None:
    world.chain(through="simulation")
    world.names("simulation", form=form)
    world.change("simulation", written_at="2026-10-26T13:00:00Z")  # still a valid record
    assert world.read("simulation")[1] == []
    found = decisions(world.runs, 1, repo=world.repo)
    assert found["simulation"]["state"] == "invalid" and found["simulation"]["record"] is None
    assert any("named this record with sha256" in p for p in found["simulation"]["problems"])
    assert found["engineering"]["state"] == "passed"


def test_a_record_a_campaign_named_that_is_gone_is_a_problem(world) -> None:
    world.write("engineering", world.engineering())
    world.names("engineering")
    path = decision_path(world.runs, "engineering")
    path.unlink()
    assert any("no longer there" in p for p in immutable(path, journal_of(world)))
    found = decisions(world.runs, 1, repo=world.repo)
    assert found["engineering"]["state"] == "invalid"


def test_a_missing_journal_fails_closed_for_a_record_that_exists(world) -> None:
    world.write("engineering", world.engineering())
    path = decision_path(world.runs, "engineering")
    assert any("journal is missing" in p for p in immutable(path, None))
    assert immutable(world.runs / "study3" / "decisions" / "simulation.json", None) == []
    found = decisions(world.runs, 1, repo=world.repo)  # no launch/ at all
    assert found["engineering"]["state"] == "invalid"
    assert found["simulation"]["state"] == "missing" and found["simulation"]["problems"] == []
    world.start({"study": 3, "candidate": 1, "step": "search", "round": 0})
    assert decisions(world.runs, 1, repo=world.repo)["engineering"]["state"] == "passed"


def test_an_unreadable_journal_fails_closed(world) -> None:
    world.write("engineering", world.engineering())
    log = world.runs / "launch" / "broken.jsonl"
    log.parent.mkdir()
    log.write_bytes(b"\xff\xfe{\n")
    assert decisions(world.runs, 1, repo=world.repo)["engineering"]["state"] == "invalid"


def test_a_header_naming_a_record_without_a_sha256_or_with_a_bad_one_is_a_problem(world) -> None:
    world.write("engineering", world.engineering())
    path = decision_path(world.runs, "engineering")
    named = "study3/decisions/engineering.json"
    for header in ({"decision_records": {"engineering": {"path": named}}},
                   {"decision_records": {"engineering": {"path": named, "sha256": "short"}}},
                   {"decision_records": {"engineering": {"path": named, "sha256": None}}},
                   {"decision_record": named},
                   {"decision_record": named, "decision_record_sha256": 7}):  # fmt: skip
        problems = immutable(path, [("study3-c1", {"header": header})])
        assert len(problems) == 1 and "without a readable sha256" in problems[0]


def test_a_record_is_found_in_a_header_wherever_it_sits_and_from_another_root(world) -> None:
    world.write("engineering", world.engineering())
    path = decision_path(world.runs, "engineering")
    other = "/workspace/runs/study3/decisions/engineering.json"
    header = {"nested": [{"deeper": {"path": other, "sha256": "0" * 64}}]}
    problems = immutable(path, [("study3-c1", {"header": header})])
    assert len(problems) == 1 and "named this record with sha256" in problems[0]
    header = {"nested": [{"deeper": {"path": other, "sha256": sha256(path)}}]}
    assert immutable(path, [("study3-c1", {"header": header})]) == []


def test_another_record_or_another_file_is_not_this_record(world) -> None:
    world.write("engineering", world.engineering())
    path = decision_path(world.runs, "engineering")
    for named in ("study3/decisions/simulation.json", "study3/decisions/engineering.txt",
                  "decisions/engineering.json", "engineering.json", "docs/study3/engineering.json",
                  "study3/decisions/n-c1.json"):  # fmt: skip
        header = {
            "ref": {"path": named, "sha256": "0" * 64},
            "flat": named,
            "flat_sha256": "0" * 64,
        }
        assert immutable(path, [("study3-c1", {"header": header})]) == []
    for start in ({"header": None}, {"header": "text"}, {"header": []}, {}, {"header": {"x": 1}}):
        assert immutable(path, [("study3-c1", start)]) == []


def test_two_campaigns_with_two_names_are_each_a_problem_once(world) -> None:
    world.write("engineering", world.engineering())
    path = decision_path(world.runs, "engineering")
    named = "study3/decisions/engineering.json"
    journal = [(f"campaign-{i // 3}", {"header": {"decision_record": named,
                                                  "decision_record_sha256": f"{i // 3}" * 64}})
               for i in range(6)]  # fmt: skip
    problems = immutable(path, journal)
    assert len(problems) == 2 and "campaign-0" in problems[0] and "campaign-1" in problems[1]


def test_a_path_among_the_items_of_a_list_is_named_without_a_sha256(world) -> None:
    world.write("engineering", world.engineering())
    path = decision_path(world.runs, "engineering")
    named = "study3/decisions/engineering.json"
    for header in ({"records": [named]}, {"records": ["other.json", named]},
                   {"nested": {"records": [["deeper", named]]}},
                   {"records": [named], "sha256": sha256(path)}):  # fmt: skip
        problems = immutable(path, [("study3-c1", {"header": header})])
        assert len(problems) == 1 and "without a readable sha256" in problems[0]
    # a list of other paths, or the mapping and the flat forms, are as before
    header = {"records": ["study3/decisions/simulation.json", 3, None, {"x": [named + ".bak"]}]}
    assert immutable(path, [("study3-c1", {"header": header})]) == []


def journal_line(world: World, campaign: str, text: str) -> None:
    """A raw line (as a crash or a bad write leaves it) at the end of a campaign's log."""
    log = world.runs / "launch" / f"{campaign}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as f:
        f.write(text + "\n")


CUT = '{"event": "start", "experiment": "study3", "header": {"CORRUPT-SECRET-3c'


def test_an_unparseable_journal_line_that_is_not_the_last_fails_closed(world) -> None:
    world.write("engineering", world.engineering())
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    journal_line(world, "study3-c1-validation", CUT)  # a line in the middle, once another follows
    world.start({"study": 3, "candidate": 1, "step": "search", "round": 0})
    found = decisions(world.runs, 1, repo=world.repo)
    entry = found["engineering"]
    assert entry["state"] == "invalid" and entry["record"] is None
    [problem] = entry["problems"]
    assert "study3-c1-validation.jsonl line 2" in problem and "launch/" in problem
    assert "CORRUPT-SECRET" not in json.dumps(found)
    assert found["simulation"]["state"] == "missing"  # nothing to protect where nothing is written
    path = decision_path(world.runs, "engineering")
    assert immutable(path, [], corrupt=["launch/x.jsonl line 4"])
    assert "launch/x.jsonl line 4" in immutable(path, [], corrupt=["launch/x.jsonl line 4"])[0]
    assert immutable(path, []) == []
    missing = decision_path(world.runs, "simulation")
    assert immutable(missing, [], corrupt=["launch/x.jsonl line 4"]) == []


def test_a_line_cut_short_as_the_last_of_its_log_is_the_launchers_rule_and_no_problem(
    world,
) -> None:
    world.write("engineering", world.engineering())
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    journal_line(world, "study3-c1-validation", CUT)  # the last line of its log
    journal_line(world, "study3-c1-validation", "")  # blank lines are no line
    journal_line(world, "study3-c1-validation", "   ")
    found = decisions(world.runs, 1, repo=world.repo)
    assert found["engineering"]["state"] == "passed" and found["engineering"]["problems"] == []
    journal_line(world, "other-log", CUT)  # the last of its own log, though another log follows
    world.start({"study": 3, "candidate": 1, "step": "search", "round": 0}, "zzz-last-log")
    assert decisions(world.runs, 1, repo=world.repo)["engineering"]["state"] == "passed"


# ------------------------------------------------------------------ decisions for a candidate


def test_decisions_lists_the_records_a_candidate_needs_each_with_its_state(world) -> None:
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    found = decisions(world.runs, 1, repo=world.repo)
    assert list(found) == ["engineering", "simulation", "n-c1"]
    assert {k: v["state"] for k, v in found.items()} == {k: "missing" for k in found}
    assert all(v["problems"] == [] and v["record"] is None for v in found.values())
    world.chain(N=75)
    found = decisions(world.runs, 1, repo=world.repo)
    assert list(found) == ["engineering", "simulation", "n-c1"]  # N = 75: no budget decision
    assert {k: v["state"] for k, v in found.items()} == {k: "passed" for k in found}
    assert found["n-c1"]["record"]["N"] == 75 and found["n-c1"]["path"].endswith("n-c1.json")
    assert decisions(world.runs, 2, repo=world.repo)["n-c2"]["state"] == "missing"


def test_decisions_lists_the_budget_record_when_n_is_above_ninety(world) -> None:
    world.chain(N=99)
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    found = decisions(world.runs, 1, repo=world.repo)
    assert list(found) == ["engineering", "simulation", "n-c1", "budget-c1"]
    assert found["budget-c1"]["state"] == "missing"
    [problem] = found["budget-c1"]["problems"]
    assert "budget decision" in problem and "above 90" in problem
    world.write("budget-c1", world.budget(N=99, decision="stop", state="failed"))
    found = decisions(world.runs, 1, repo=world.repo)
    assert found["budget-c1"]["state"] == "failed" and found["budget-c1"]["problems"] == []


def test_decisions_reports_a_failed_record_as_failed_and_an_invalid_one_as_invalid(world) -> None:
    world.write("engineering", world.engineering(state="failed",
                                                 checks={**world.engineering()["checks"],
                                                         "bridge": None}))  # fmt: skip
    world.write("simulation", world.simulation())
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    world.file(world.simulation()["fixation"]).write_text("changed\n")
    found = decisions(world.runs, 1, repo=world.repo)
    assert found["engineering"]["state"] == "failed"
    assert found["simulation"]["state"] == "invalid" and found["simulation"]["record"] is None
    assert any("fixation" in p and "sha256" in p for p in found["simulation"]["problems"])


def test_decisions_does_not_consume_an_n_record_whose_simulation_changed(world) -> None:
    world.chain(N=99, through="budget")
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    world.file(world.simulation()["inputs"]["design"]).write_text("changed\n")
    found = decisions(world.runs, 1, repo=world.repo)
    assert found["simulation"]["state"] == "invalid"
    assert found["n-c1"]["state"] == "invalid" and found["budget-c1"]["state"] == "invalid"


def test_a_simulation_named_by_a_campaign_and_then_changed_invalidates_what_rests_on_it(
    world,
) -> None:
    world.chain(N=99, through="budget")
    world.names("simulation")
    world.change("simulation", written_at="2026-10-26T13:00:00Z")  # a valid record, another one
    found = decisions(world.runs, 1, repo=world.repo)
    assert found["engineering"]["state"] == "passed"
    for kind in ("simulation", "n-c1", "budget-c1"):
        assert found[kind]["state"] == "invalid" and found[kind]["record"] is None, kind
        assert found[kind]["problems"], kind


def test_an_n_entry_is_invalid_when_the_simulation_entry_is_by_its_own_name(world) -> None:
    # the simulation file is as the n record bound it: only a campaign named another version of it
    world.chain(N=99, through="budget")
    path = decision_path(world.runs, "simulation")
    other = "0" * 64
    named = "study3/decisions/simulation.json"
    world.start({"study": 3, "candidate": 1, "step": "validation", "round": None,
                 "decision_record": named, "decision_record_sha256": other})  # fmt: skip
    assert world.read("n-c1")[1] == [] and sha256(path) != other
    found = decisions(world.runs, 1, repo=world.repo)
    assert found["simulation"]["state"] == "invalid"
    n = found["n-c1"]
    assert n["state"] == "invalid" and n["record"] is None
    assert any("simulation record it rests on is invalid" in p and str(path) in p
               for p in n["problems"])  # fmt: skip
    budget = found["budget-c1"]
    assert budget["state"] == "invalid" and budget["record"] is None
    assert any("n-c1 record it answers is invalid" in p for p in budget["problems"])


def test_the_budget_a_power_short_n_needs_does_not_vanish_when_its_n_record_is_invalid(
    world,
) -> None:
    world.chain(N=84, through="n", gates=SHORT)
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    assert list(decisions(world.runs, 1, repo=world.repo))[-1] == "budget-c1"
    world.file(world.n()["spread"]["output"]).write_text("changed\n")  # n is now invalid
    found = decisions(world.runs, 1, repo=world.repo)
    assert found["n-c1"]["state"] == "invalid" and found["budget-c1"]["state"] == "invalid"
    assert any("n-c1 record it answers is invalid" in p for p in found["budget-c1"]["problems"])


def test_a_missing_n_record_lists_no_budget_entry_and_a_missing_simulation_no_invalid_n(
    world,
) -> None:
    world.write("engineering", world.engineering())
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    found = decisions(world.runs, 1, repo=world.repo)
    assert list(found) == ["engineering", "simulation", "n-c1"]
    assert found["n-c1"]["state"] == "missing" and found["simulation"]["state"] == "missing"


def test_decisions_prints_only_the_rounded_spread(world) -> None:
    world.chain(N=75)
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    text = json.dumps(decisions(world.runs, 1, repo=world.repo))
    assert '"spread_pp": 4.5' in text
    assert all(word not in text for word in ('"mean', '"sign', "per_seed", "4.37"))


def test_decisions_is_a_pure_reading(world) -> None:
    world.chain(N=99, through="budget")
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    before = snapshot(world.runs)
    decisions(world.runs, 1, repo=world.repo)
    decisions(world.runs, 2, repo=world.repo)
    assert snapshot(world.runs) == before


def snapshot(root: Path) -> dict:
    return {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(root.rglob("*"))
            if p.is_file() and not any(part in SEALED for part in p.parts)}  # fmt: skip


# ------------------------------------------------------------------ the sealing boundary


def seal(monkeypatch) -> list[str]:
    """From now on ``open``, ``io.open`` and ``Path.open``/``read_text``/``read_bytes`` fail on
    any path under a ``responses/`` or ``observations/`` folder; the list holds every such
    attempt."""
    attempts: list[str] = []

    def guard(path) -> None:
        if isinstance(path, int):
            return
        named = os.fsdecode(path)
        parts = (*Path(named).parts, *Path(os.path.realpath(named)).parts)
        if any(part in SEALED for part in parts):
            attempts.append(str(path))
            raise AssertionError(f"a sealed file was opened: {path}")

    real_open = builtins.open

    def open_(file, *args, **kwargs):
        guard(file)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_)
    monkeypatch.setattr(io, "open", open_)

    def guarded(real):
        def wrapper(self, *args, **kwargs):
            guard(self)
            return real(self, *args, **kwargs)

        return wrapper

    for name in ("open", "read_text", "read_bytes"):
        monkeypatch.setattr(Path, name, guarded(getattr(Path, name)))
    return attempts


def test_no_record_is_read_or_written_with_a_response_or_observation_opened(
    world, monkeypatch
) -> None:
    attempts = seal(monkeypatch)
    world.chain(N=99, through="budget")
    for kind, campaign in (("engineering", "c1-a"), ("simulation", "c1-b"), ("n-c1", "c1-c")):
        world.names(kind, campaign=campaign)
    for candidate in (1, 2):
        found = decisions(world.runs, candidate, repo=world.repo)
        assert SENTINEL not in json.dumps(found)
    assert decisions(world.runs, 1, repo=world.repo)["budget-c1"]["state"] == "passed"
    assert attempts == []
    # the guard is live: a sealed file cannot be read under it
    with pytest.raises(AssertionError):
        (world.run / "responses" / "harmful_evaluation.jsonl").read_text()
    assert len(attempts) == 1


def test_a_record_that_resolves_into_responses_is_refused_and_never_opened(
    world, monkeypatch
) -> None:
    sealed = world.run / "responses" / "keys.json"
    sealed.write_text(json.dumps({"LEAKY_KEY_NAME": 1, "kind": "engineering"}))
    decisions_dir = world.runs / "study3" / "decisions"
    decisions_dir.mkdir(parents=True)
    (decisions_dir / "engineering.json").symlink_to(sealed)
    world.start({"study": 3, "candidate": 1, "step": "engineering", "round": None})
    attempts = seal(monkeypatch)
    found, problems = world.read("engineering")
    assert found is None and len(problems) == 1 and "never opened" in problems[0]
    assert "LEAKY_KEY_NAME" not in problems[0]
    everything = decisions(world.runs, 1, repo=world.repo)
    assert everything["engineering"]["state"] == "invalid"
    assert "LEAKY_KEY_NAME" not in json.dumps(everything)
    assert immutable(decisions_dir / "engineering.json", [])
    assert all("never opened" in p for p in everything["engineering"]["problems"])
    assert len(everything["engineering"]["problems"]) == 1  # said once
    assert attempts == []
    with pytest.raises(AssertionError):  # the guard follows the link
        (decisions_dir / "engineering.json").read_text()
    assert len(attempts) == 1


def test_a_decisions_folder_linked_into_observations_is_never_written_to(
    world, monkeypatch
) -> None:
    sealed = world.run / "observations"
    (world.runs / "study3").mkdir(parents=True, exist_ok=True)
    (world.runs / "study3" / "decisions").symlink_to(sealed)
    attempts = seal(monkeypatch)
    with pytest.raises(DecisionRefused, match="never opened"):
        world.write("engineering", world.engineering())
    assert sorted(p.name for p in sealed.iterdir()) == ["measurements.jsonl"]
    assert world.read("engineering")[0] is None
    assert attempts == []


def test_a_decision_record_is_read_as_utf_8(world, monkeypatch) -> None:
    world.write("engineering", world.engineering(checks={
        **world.engineering()["checks"],
        "bridge": {**world.ref("pont-\u00e9t\u00e9.json"), "passed": True}}))  # fmt: skip
    seen = []
    real = Path.read_text

    def recording(self, *args, **kwargs):
        if self.suffix == ".json" and self.parent.name == "decisions":
            seen.append(kwargs.get("encoding", args[0] if args else None))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording)
    assert world.read("engineering")[1] == []
    decisions(world.runs, 1, repo=world.repo)
    assert seen and set(seen) == {"utf-8"}


def test_the_module_imports_no_gpu_path() -> None:
    code = ("import sys; import zero_field_refusal.analysis.study3_decisions; "
            "print([m for m in ('torch', 'transformers', 'zero_field_refusal.schedule', "
            "'zero_field_refusal.train', 'zero_field_refusal.launch', "
            "'zero_field_refusal.qualification', 'zero_field_refusal.analysis.stage_a') "
            "if m in sys.modules])")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
