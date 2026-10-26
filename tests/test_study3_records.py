"""Study 3's run records as endpoints (analysis/study3_records.py), on synthetic RUNS_DIR trees
built as the launcher, ``schedule.run_scheduled`` and the qualification commands write them: the
launch journal, the run directories (configs loaded from the study-3 experiment configs and
hashed as the launcher does) and the restart and supersession records. Every regular run also
holds a harmful-response file and an observations file with a sentinel text; the reader opens
neither."""

import builtins
import io
import itertools
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from zero_field_refusal.analysis.study3_records import (
    INSTABILITY,
    ROLES,
    SEEDS,
    candidate_of,
    read_endpoints,
    read_theta0,
    step_key,
)
from zero_field_refusal.config import config_hash
from zero_field_refusal.paths import CONFIG_ROOT, REPO_ROOT
from zero_field_refusal.protocol import load_run_config

THETA0 = {"benchmark_accuracy": 0.60, "factual_accuracy": 0.80}
SENTINEL = "SEALED-RESPONSE-7f3c9a"
OPTIMIZER = {"A": "adamw", "F": "adam_frozen", "P": "adamw_prewarmed"}
STEP_OF = {"engineering": "engineering", "search": "search", "numerical": "numerical",
           "validation": "validation", "confirmation": "confirmation"}  # fmt: skip
K_ONLY = ("engineering", "search", "numerical")
DETAIL = {"gradient_spike": {"grad_norm": 40.0, "reference_median": 1.0},
          "k_above_max": {"K_monitor": 17.0, "k_max": 16.0}}  # fmt: skip


@lru_cache
def config_of(arm: str, lr: float, seed: int, role: str) -> tuple[str, str, tuple[str, ...]]:
    """The resolved config (YAML) of a study-3 run, its config hash, and the launcher's
    command (script, config, overrides)."""
    name = "experiments/study3_k_only" if role in K_ONLY else "experiments/study3"
    overrides = (f"optimizer={OPTIMIZER[arm]}", f"optimizer.lr={lr}", f"study3.role={role}",
                 f"seed={seed}")  # fmt: skip
    cfg = load_run_config(name, overrides)
    command = (str(REPO_ROOT / "scripts/train.py"), name, *overrides)
    return OmegaConf.to_yaml(cfg, resolve=True), config_hash(cfg), command


def header(candidate: int, step: str, round_: int | None, **extra) -> dict:
    out = {"study": 3, "candidate": candidate, "step": step, "round": round_}
    if step in ("validation", "confirmation"):
        out |= {"versions_record": "docs/study3-versions.json",
                "versions_record_sha256": "a" * 64}  # fmt: skip
    return out | extra


def jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def edit_jsonl(path: Path, change) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    jsonl(path, change(rows))


def write_run(
    run_dir: Path,
    yaml: str,
    chash: str,
    campaign: dict,
    *,
    k_only: bool,
    stop: str = "steps",
    steps: int = 25,
    k: float = 6.0,
    k_harmful: float = 5.0,
    mmlu: float = 2.0,
    factual: float = 3.0,
    status: str = "completed",
    terminal: bool = True,
) -> None:
    """One run directory as the tracker and the schedule write it: an evaluation at step 25
    with K_benign ``k`` (and ``k_harmful`` for a regular run), utility losses ``mmlu`` and
    ``factual`` in pp; for another stop, ``steps`` updates and no evaluation. Without
    ``terminal`` the run is still running (or died) before its outcome."""
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml)
    inputs = {"streams": {"training": {"count": steps * 20}}}
    inputs |= {"k_only": True} if k_only else {"harmful_prompts": {"set": "harmful_evaluation",
                                                                   "count": 521}}  # fmt: skip
    meta = {"run_id": run_dir.name, "experiment": "study3", "config_hash": chash,
            "status": status if terminal else "running", "campaign": campaign,
            "host": {"hostname": "pod"}, "launch": {"pod_id": "pod-1"},
            "inputs": inputs}  # fmt: skip
    (run_dir / "meta.json").write_text(json.dumps(meta))
    read_to = steps - 1 if stop == "nonfinite_weight" else steps
    readings = [{"step": s, "K_monitor": k * s / 25} for s in range(5, read_to + 1, 5)]
    jsonl(run_dir / "monitor.jsonl", readings)
    training = [{"step": s, "loss": 1.0, "grad_norm": 1.0} for s in range(steps)]
    logged = [{"step": r["step"] - 1, "K_monitor": r["K_monitor"]} for r in readings]
    evaluations = []
    if stop == "steps":
        out = {"K_benign": k, "heldout_nll_per_token": 1.2,
               "factual_accuracy": THETA0["factual_accuracy"] - factual / 100,
               "benchmark_accuracy": THETA0["benchmark_accuracy"] - mmlu / 100,
               "K_monitor_primary": k, "K_monitor_full": k}  # fmt: skip
        if not k_only:
            out["K_harmful"] = k_harmful
        evaluations.append({"step": 25, "condition": "step-25", "levels": [], "K_monitor": k,
                            **out})  # fmt: skip
        # the schedule logs every value of the evaluation, one step back
        logged.append({"step": 24, "time_s": 400.0, **{f"eval/{key}": v for key, v in out.items()}})
    jsonl(run_dir / "metrics.jsonl", training + logged)
    jsonl(run_dir / "evaluations.jsonl", evaluations)
    observations = [{"condition": "step-25", "measure": "factual_qa", "item_id": "f:0",
                     "value": 1.0, "text": SENTINEL}]  # fmt: skip
    jsonl(run_dir / "observations" / "measurements.jsonl", observations)
    if not k_only:
        jsonl(run_dir / "responses" / "harmful_evaluation.jsonl",
              [{"condition": "step-25", "item_id": "evaluation:0", "response_ids": [1, 2],
                "text": SENTINEL}])  # fmt: skip
    if terminal:
        record = {"stop": {"reason": stop, "kind": "instability" if stop in INSTABILITY else
                           "limit", **({"detail": DETAIL[stop]} if stop in DETAIL else {})},
                  "steps": steps, "responses": steps * 20, "policy": "fixed",
                  "evaluations": [row["step"] for row in evaluations],
                  "monitor_readings": len(readings), "seconds": 420.0}  # fmt: skip
        (run_dir / "terminal.json").write_text(json.dumps(record))
        (run_dir / "summary.json").write_text(json.dumps({"terminal": record, "K_benign": k}))


class World:
    """A synthetic RUNS_DIR: launch journals, study-3 run directories, qualification records,
    and the cells prescribed so far."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.prescribed: dict[str, set] = defaultdict(set)
        self.ids = itertools.count()
        self.theta0 = root / "theta0"
        self.theta0.mkdir(parents=True)
        (self.theta0 / "summary.json").write_text(json.dumps({**THETA0, "K_benign": 0.0}))

    def log(self, campaign: str, record: dict) -> None:
        path = self.root / "launch" / f"{campaign}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps({"time": "2026-10-26T00:00:00+00:00", **record}) + "\n")

    def run(
        self,
        arm: str = "A",
        lr: float = 1.45e-5,
        seed: int = 10300,
        *,
        role: str = "validation",
        candidate: int = 1,
        step: str | None = None,
        round_: int | None = None,
        campaign: str = "study3-c1",
        cell: str | None = None,
        attempt: int = 1,
        end: str | None = "done",
        directory: bool = True,
        prescribe: bool = True,
        head: dict | None = None,
        **records,
    ) -> str:
        """One attempt: its start (and end) in the campaign's journal, its run directory."""
        step = step or STEP_OF[role]
        round_ = 0 if round_ is None and step == "search" else round_
        yaml, chash, command = config_of(arm, lr, seed, role)
        cell = cell or f"{arm}-{lr:g}"
        label = f"{cell}/s{seed}"
        run_id = f"20261026T{next(self.ids):06d}Z-{cell}-s{seed}-{chash[:8]}-a{attempt}"
        head = header(candidate, step, round_) if head is None else head
        self.log(campaign, {"event": "start", "run": label, "attempt": attempt, "run_id": run_id,
                            "experiment": "study3", "config_hash": chash,
                            "command": list(command), "header": head})  # fmt: skip
        if directory:
            write_run(self.root / "study3" / run_id, yaml, chash,
                      {"name": campaign, "label": label, "attempt": attempt, "header": head},
                      k_only=role in K_ONLY, **records)  # fmt: skip
        if end is not None:
            self.end(campaign, label, attempt, run_id, end)
        if prescribe:
            self.prescribed[step_key(candidate, step, round_)].add((arm, lr, seed))
        return run_id

    def end(self, campaign, label, attempt, run_id, outcome, problem=None) -> None:
        self.log(campaign, {"event": "end", "run": label, "attempt": attempt, "run_id": run_id,
                            "returncode": 0, "outcome": outcome, "problem": problem})  # fmt: skip

    def recovery(self, campaign: str, label: str, reason: str = "the Pod stopped") -> None:
        cell = label.split("/")[0]
        self.log(campaign, {"event": "recovery", "experiment": "study3", "cell": cell,
                            "reason": reason, "runs": [label], "only": label})  # fmt: skip

    def read(self, prescribed=None):
        return read_endpoints(
            self.root,
            theta0=self.theta0,
            prescribed=self.prescribed if prescribed is None else prescribed,
        )


@pytest.fixture
def world(tmp_path) -> World:
    return World(tmp_path / "runs")


def by_id(endpoints) -> dict:
    return {e.run_id: e for e in endpoints}


def refused(problems, run_id, *words) -> bool:
    return any(run_id in p and all(w in p for w in words) for p in problems)


def states(found) -> dict:
    return {e.run_id: e.state for e in found}


def listed_refused(found, problems, run_id, *words) -> bool:
    """The attempt is listed as refused, with its refusal as its reason, and named in the
    problems."""
    mine = [e for e in found if e.run_id == run_id]
    return (bool(mine) and all(e.state == "refused" and not e.counts for e in mine)
            and any(all(w in r for w in words) for e in mine for r in e.reasons)
            and refused(problems, run_id, *words))  # fmt: skip


# ------------------------------------------------------------------ registered values


def test_seed_ranges_are_the_registered_ones() -> None:
    assert ROLES == ("engineering", "search", "numerical", "validation", "confirmation")
    assert SEEDS["engineering"] == {1: range(10000, 10004)}
    assert SEEDS["search"] == {1: range(10100, 10104), 2: range(10110, 10114)}
    assert SEEDS["numerical"] == {1: range(10200, 10203), 2: range(10210, 10213)}
    assert SEEDS["validation"] == {1: range(10300, 10340), 2: range(10340, 10380)}
    assert SEEDS["confirmation"] == {1: range(10400, 10600), 2: range(10400, 10600)}
    settings = OmegaConf.load(CONFIG_ROOT / "experiments/qualification.yaml")
    assert list(SEEDS["engineering"][1]) == list(settings.seeds.engineering)
    for candidate in (1, 2):
        assert list(SEEDS["numerical"][candidate]) == list(settings.seeds.numerical[candidate])


def test_candidate_of() -> None:
    assert candidate_of(10101, "search") == 1
    assert candidate_of(10113, "search") == 2
    assert candidate_of(10339, "validation") == 1
    assert candidate_of(10340, "validation") == 2
    assert candidate_of(10002, "engineering") == 1
    assert candidate_of(10101, "validation") is None
    assert candidate_of(10450, "confirmation") is None  # bound by its campaign's header


def test_the_instability_rules_are_the_schedules() -> None:
    from zero_field_refusal import schedule

    assert INSTABILITY == schedule.INSTABILITY


def test_step_keys() -> None:
    assert step_key(1, "search", 0) == "c1/search/r0"
    assert step_key(2, "validation", None) == "c2/validation"


def test_theta0_from_a_run_directory_or_its_summary(world) -> None:
    assert read_theta0(world.theta0) == THETA0
    assert read_theta0(world.theta0 / "summary.json") == THETA0
    (world.theta0 / "summary.json").write_text(json.dumps({"benchmark_accuracy": 0.6}))
    with pytest.raises(ValueError, match="factual_accuracy"):
        read_theta0(world.theta0)


# ------------------------------------------------------------------ outcomes


def test_a_usable_regular_run(world) -> None:
    run_id = world.run("F", 1.3e-5, 10300, k=6.2, k_harmful=5.1, mmlu=2.0, factual=3.0)
    endpoints, problems = world.read()
    assert problems == []
    [e] = endpoints
    assert (e.run_id, e.state, e.reasons, e.counts) == (run_id, "usable", (), True)
    assert (e.arm, e.lr, e.seed, e.role, e.candidate, e.k_only) == (
        "F",
        1.3e-5,
        10300,
        "validation",
        1,
        False,
    )
    assert (e.steps, e.stop, e.K, e.K_harmful) == (25, "steps", 6.2, 5.1)
    assert math.isclose(e.mmlu_loss_pp, 2.0) and math.isclose(e.factual_loss_pp, 3.0)
    assert (e.attempt, e.label) == (1, "F-1.3e-05/s10300")
    assert e.run_dir == world.root / "study3" / run_id
    assert (e.cell.campaign, e.cell.step, e.cell.round, e.cell.key) == (
        "study3-c1",
        "validation",
        None,
        "c1/validation",
    )
    assert e.cell.versions_record == "docs/study3-versions.json"
    assert e.cell.versions_record_sha256 == "a" * 64


def test_a_usable_k_only_run_has_no_mmlu_criterion(world) -> None:
    run_id = world.run("A", 1.45e-5, 10100, role="search", k=4.9, mmlu=30.0)
    [e], problems = world.read()
    assert problems == []
    assert (e.run_id, e.state, e.k_only, e.K, e.K_harmful) == (run_id, "usable", True, 4.9, None)
    assert math.isclose(e.mmlu_loss_pp, 30.0)  # recorded, descriptive (amendment 1)
    assert e.cell.key == "c1/search/r0"


def nan_factual(run_dir: Path, *, appended: bool) -> None:
    """A NaN factual accuracy in the step-25 evaluation and in the row the schedule logs from it
    (``eval/factual_accuracy`` at step 24); ``appended``: as a second such row."""
    edit_jsonl(run_dir / "evaluations.jsonl",
               lambda rows: [{**rows[0], "factual_accuracy": float("nan")}])  # fmt: skip
    row = json.loads((run_dir / "evaluations.jsonl").read_text().splitlines()[0])
    nan = {f"eval/{k}": v for k, v in row.items()
           if k not in ("step", "condition", "levels", "K_monitor")}  # fmt: skip

    def change(rows):
        if appended:
            return [*rows, {"step": 24, **nan}]
        return [{**r, **nan} if "eval/K_benign" in r else r for r in rows]

    edit_jsonl(run_dir / "metrics.jsonl", change)


@pytest.mark.parametrize("appended", [False, True])
def test_a_k_only_runs_utility_decides_nothing(world, appended) -> None:
    run_id = world.run("A", 1.45e-5, 10100, role="search")
    run_dir = world.root / "study3" / run_id
    assert '"eval/factual_accuracy"' in (run_dir / "metrics.jsonl").read_text()
    nan_factual(run_dir, appended=appended)
    [e], problems = world.read()
    assert (e.state, problems, e.factual_loss_pp, e.K) == ("usable", [], None, 6.0)


def test_a_regular_runs_utility_must_be_finite(world) -> None:
    run_id = world.run("A", 1.45e-5, 10300)
    nan_factual(world.root / "study3" / run_id, appended=False)
    [e], _ = world.read()
    assert e.state == "technical"
    assert any(r.startswith("logged metrics: a non-finite number at step 24") for r in e.reasons)


def test_a_k_only_run_needs_a_positive_k(world) -> None:
    world.run("A", 1.45e-5, 10100, role="search", k=0.0)
    [e], _ = world.read()
    assert (e.state, e.unusable_reason) == ("unusable", "nonpositive_k")
    assert any("positive" in r for r in e.reasons)


def test_a_gross_collapse_is_unusable_at_more_than_15_pp(world) -> None:
    over = world.run("A", 1.45e-5, 10300, mmlu=15.1)
    at = world.run("A", 2.11e-5, 10300, mmlu=15.0)
    k_only = world.run("A", 1.45e-5, 10101, role="search", mmlu=40.0)
    found, problems = world.read()
    found = by_id(found)
    assert problems == []
    assert (found[over].state, found[over].unusable_reason) == ("unusable", "gross_collapse")
    assert any("gross collapse" in r for r in found[over].reasons)
    assert found[at].unusable_reason is None
    assert found[over].counts  # it removes support; it is no route by itself
    assert found[at].state == "usable"
    assert found[k_only].state == "usable"


@pytest.mark.parametrize("stop, steps", [("gradient_spike", 12), ("k_above_max", 20),
                                         ("nonfinite_weight", 9)])  # fmt: skip
def test_an_instability_stop_is_scientific(world, stop, steps) -> None:
    run_id = world.run("P", 1.85e-5, 10300, stop=stop, steps=steps)
    [e], problems = world.read()
    assert problems == []
    assert (e.run_id, e.state, e.stop, e.steps, e.K) == (run_id, "scientific", stop, steps, None)


def test_another_stop_is_technical(world) -> None:
    world.run("A", 1.45e-5, 10300, stop="plateau", steps=20)
    [e], _ = world.read()
    assert e.state == "technical" and any("plateau" in r for r in e.reasons)


def test_a_run_without_its_terminal_record_is_pending(world) -> None:
    running = world.run("A", 1.45e-5, 10300, end=None, terminal=False)
    starting = world.run("A", 2.11e-5, 10300, end=None, directory=False)
    stopped = world.run("F", 1.3e-5, 10300, end="stopped", terminal=False)
    found, problems = world.read()
    assert problems == []
    assert {e.run_id: e.state for e in found} == {running: "pending", starting: "pending",
                                                  stopped: "pending"}  # fmt: skip
    assert by_id(found)[starting].run_dir == world.root / "study3" / starting


def test_a_published_record_waits_for_the_launchers_end(world) -> None:
    unended = world.run("A", 1.45e-5, 10300, end=None)
    stopped = world.run("A", 2.11e-5, 10300, end="stopped")
    failed = world.run("F", 1.3e-5, 10300, end=None, status="failed")
    found, problems = world.read()
    assert problems == []
    assert states(found) == {unended: "pending", stopped: "pending", failed: "pending"}
    found = by_id(found)
    assert found[unended].reasons == ("the launcher has not ended it",)
    assert (found[unended].K, found[unended].steps) == (None, None)  # nothing read yet


def test_an_unexplained_failure_is_technical(world) -> None:
    ended = world.run("A", 1.45e-5, 10300, end="technical", terminal=False, status="failed")
    failed = world.run("A", 2.11e-5, 10300, end=None, terminal=False)
    meta = world.root / "study3" / failed / "meta.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "status": "failed"}))
    no_directory = world.run("F", 1.3e-5, 10300, end="technical", directory=False)
    done_without_record = world.run("F", 1.83e-5, 10300, end="done", terminal=False)
    finished = world.run("P", 1.3e-5, 10300, end=None, terminal=False)
    meta = world.root / "study3" / finished / "meta.json"
    meta.write_text(json.dumps({**json.loads(meta.read_text()), "status": "completed"}))
    found, problems = world.read()
    assert problems == []
    states = {e.run_id: e.state for e in found}
    assert states == {ended: "technical", failed: "technical", no_directory: "technical",
                      done_without_record: "technical", finished: "technical"}  # fmt: skip
    assert "meta.json: status completed" in by_id(found)[finished].reasons


def drop_last_reading(d: Path) -> None:
    edit_jsonl(d / "monitor.jsonl", lambda rows: rows[:-1])


def nan_loss(d: Path) -> None:
    edit_jsonl(d / "metrics.jsonl", lambda rows: [{**rows[0], "loss": float("nan")}, *rows[1:]])


def infinite_k_harmful(d: Path) -> None:
    edit_jsonl(d / "evaluations.jsonl", lambda rows: [{**rows[0], "K_harmful": float("inf")}])


def no_mmlu(d: Path) -> None:
    edit_jsonl(d / "evaluations.jsonl", lambda rows: [{k: v for k, v in rows[0].items()
                                                       if k != "benchmark_accuracy"}])  # fmt: skip


def cut_terminal(d: Path) -> None:
    (d / "terminal.json").write_text("{")


def other_summary(d: Path) -> None:
    (d / "summary.json").write_text(json.dumps({"terminal": {"steps": 3}}))


def short_stream(d: Path) -> None:
    meta = json.loads((d / "meta.json").read_text())
    meta["inputs"]["streams"]["training"]["count"] = 20
    (d / "meta.json").write_text(json.dumps(meta))


@pytest.mark.parametrize(
    "change, why",
    [
        (drop_last_reading, "monitor"),
        (nan_loss, "non-finite"),
        (infinite_k_harmful, "non-finite"),
        (no_mmlu, "benchmark_accuracy"),
        (cut_terminal, "unreadable"),
        (other_summary, "summary"),
        (short_stream, "stream"),
    ],
)
def test_a_nonconforming_record_is_technical(world, change, why) -> None:
    run_id = world.run("A", 1.45e-5, 10300)
    change(world.root / "study3" / run_id)
    [e], problems = world.read()
    assert problems == []
    assert e.state == "technical", e
    assert any(why in r for r in e.reasons), e.reasons


def test_a_launcher_conflict_is_technical_and_its_text_is_not_copied(world) -> None:
    run_id = world.run("A", 1.45e-5, 10300, end=None)
    world.end("study3-c1", "A-1.45e-05/s10300", 1, run_id, "conflict", problem=SENTINEL)
    [e], problems = world.read()
    assert e.state == "technical" and any("conflict" in r for r in e.reasons)
    assert SENTINEL not in repr((e, problems))


def test_utility_losses_are_against_theta0(world) -> None:
    world.run("A", 1.45e-5, 10300, mmlu=4.4, factual=-1.8)
    [e], _ = world.read()
    assert math.isclose(e.mmlu_loss_pp, 4.4) and math.isclose(e.factual_loss_pp, -1.8)


# ------------------------------------------------------------------ attempts


def test_an_interrupted_attempt_is_recovered_by_a_recorded_round(world) -> None:
    first = world.run("A", 1.45e-5, 10300, end="technical", terminal=False, status="failed")
    world.recovery("study3-c1", "A-1.45e-05/s10300")
    second = world.run("A", 1.45e-5, 10300, attempt=2)
    found, problems = world.read()
    assert problems == []
    found = by_id(found)
    assert (found[first].state, found[first].counts) == ("recovered", False)
    assert any("the Pod stopped" in r for r in found[first].reasons)
    assert (found[second].state, found[second].counts, found[second].attempt) == ("usable", True, 2)


def test_a_recovery_round_before_the_next_attempt_leaves_the_cell_waiting(world) -> None:
    first = world.run("A", 1.45e-5, 10300, end=None, terminal=False)
    world.recovery("study3-c1", "A-1.45e-05/s10300")
    [e], problems = world.read()
    assert (e.run_id, e.state, e.counts, problems) == (first, "recovered", False, [])


def test_a_recovery_round_never_replaces_a_published_outcome(world) -> None:
    first = world.run("A", 1.45e-5, 10300, stop="gradient_spike", steps=12)
    world.recovery("study3-c1", "A-1.45e-05/s10300")
    second = world.run("A", 1.45e-5, 10300, attempt=2)
    found, problems = world.read()
    assert states(found) == {first: "refused", second: "refused"}
    assert refused(problems, first, second)


def test_a_retry_without_a_recorded_reason_leaves_two_attempts_counting(world) -> None:
    first = world.run("A", 1.45e-5, 10300, end="technical", terminal=False, status="failed")
    second = world.run("A", 1.45e-5, 10300, attempt=2)
    found, problems = world.read()
    assert states(found) == {first: "refused", second: "refused"}
    assert refused(problems, first, second)


def test_two_usable_runs_of_one_label_are_refused(world) -> None:
    first = world.run("A", 1.45e-5, 10300)
    second = world.run("A", 1.45e-5, 10300, attempt=2)
    found, problems = world.read()
    assert states(found) == {first: "refused", second: "refused"}
    assert len(problems) == 2 and refused(problems, first, second, "2 attempts count")
    assert all(any(first in r and second in r for r in e.reasons) for e in found)


def test_two_counting_runs_of_one_scientific_cell_are_refused(world) -> None:
    one = world.run("F", 1.3e-5, 10301, campaign="study3-c1-validation")
    other = world.run("F", 1.3e-5, 10301, campaign="study3-c1-validation-again")
    kept = world.run("A", 1.45e-5, 10301, campaign="study3-c1-validation")
    found, problems = world.read()
    assert states(found) == {one: "refused", other: "refused", kept: "usable"}
    assert len(problems) == 2 and refused(problems, one, other, "one cell")


def test_the_scientific_cell_is_the_prescribed_one_matched(world) -> None:
    one = world.run("A", 1.45e-5, 10301, prescribe=False)
    other = world.run("A", 1.4500000000001e-5, 10301, campaign="again", prescribe=False)
    found, problems = world.read({"c1/validation": {("A", 1.45e-5, 10301)}})
    assert states(found) == {one: "refused", other: "refused"}
    assert len(problems) == 2 and refused(problems, one, other, "one cell")
    assert {e.cell.prescribed for e in found} == {("A", 1.45e-5, 10301)}
    assert {e.lr for e in found} == {1.45e-5, 1.4500000000001e-5}


def test_the_same_cell_in_two_rounds_is_two_cells(world) -> None:
    zero = world.run("A", 1.45e-5, 10100, role="search", round_=0)
    one = world.run("A", 1.45e-5, 10100, role="search", round_=1, campaign="study3-c1-r1")
    found, problems = world.read()
    assert problems == []
    assert {e.run_id: e.cell.key for e in found} == {zero: "c1/search/r0", one: "c1/search/r1"}


def test_a_journal_superseded_or_held_run_does_not_count(world) -> None:
    first = world.run("A", 1.45e-5, 10200, role="numerical")
    label = "A-1.45e-05/s10200"
    world.log("study3-c1", {"event": "superseded", "run": label, "run_id": first,
                            "experiment": "study3", "config_hash": "x", "reason": "lost disk",
                            "record": "restart-1.json"})  # fmt: skip
    world.recovery("study3-c1", label, "lost disk")
    second = world.run("A", 1.45e-5, 10200, role="numerical", attempt=2)
    held = world.run("F", 1.3e-5, 10200, role="numerical", stop="gradient_spike", steps=3)
    world.log("study3-c1", {"event": "held", "run": "F-1.3e-05/s10200", "run_id": held,
                            "experiment": "study3", "config_hash": "x",
                            "reason": "no commit recorded",
                            "record": "supersede-1.json"})  # fmt: skip
    found, problems = world.read()
    assert problems == []
    found = by_id(found)
    assert (found[first].state, found[first].counts) == ("superseded", False)
    assert any("lost disk" in r for r in found[first].reasons)
    assert (found[second].state, found[second].counts) == ("usable", True)
    assert (found[held].state, found[held].counts) == ("held", False)
    assert any("no commit recorded" in r for r in found[held].reasons)


def test_qualification_records_supersede_launcher_runs(world) -> None:
    restarted = world.run("F", 1.3e-5, 10200, role="numerical")
    cycled = world.run("A", 1.45e-5, 10201, role="numerical")
    held = world.run("P", 1.3e-5, 10202, role="numerical", end=None, terminal=False)
    case = world.root / "qualification/c1/numerical/s10200/adam_frozen-1.3e-05"
    case.mkdir(parents=True)
    (case / "restart-1.json").write_text(json.dumps(
        {"case": {"name": "s10200/adam_frozen-1.3e-05"}, "restart": 1, "reason": "Pod stopped",
         "attempts": [], "launcher": {"run_id": restarted, "identity": {}}}))  # fmt: skip
    (world.root / "qualification/c1/supersede-1.json").write_text(json.dumps(
        {"candidate": 1, "supersession": 1, "diagnosis": "a stale cache", "fix": "rebuilt",
         "launchers": [{"run_id": cycled, "run": "A-1.45e-05/s10201"}],
         "held": [{"run_id": held, "reason": "unreadable"}], "kept_cases": []}))  # fmt: skip
    found, problems = world.read()
    assert problems == []
    states = {e.run_id: e.state for e in found}
    assert states == {restarted: "superseded", cycled: "superseded", held: "held"}
    reasons = {e.run_id: " ".join(e.reasons) for e in found}
    assert "Pod stopped" in reasons[restarted] and "a stale cache" in reasons[cycled]


def test_two_starts_of_one_run_id_are_both_refused(world) -> None:
    run_id = world.run("A", 1.45e-5, 10300)
    journal = world.root / "launch/study3-c1.jsonl"
    edit_jsonl(journal, lambda rows: [*rows, rows[0]])
    found, problems = world.read()
    assert [e.state for e in found] == ["refused", "refused"]
    assert listed_refused(found, problems, run_id, "more than once")


def test_an_unreadable_voiding_record_is_refused(world) -> None:
    world.run("A", 1.45e-5, 10200, role="numerical")
    (world.root / "qualification/c1").mkdir(parents=True)
    (world.root / "qualification/c1/supersede-1.json").write_text("{")
    _, problems = world.read()
    assert any("supersede-1.json" in p for p in problems)


# ------------------------------------------------------------------ refusals: the issued cell


def test_a_seed_outside_its_role_is_refused(world) -> None:
    search_seed = world.run("A", 1.45e-5, 10100, role="validation")
    confirmation_seed = world.run("A", 2.11e-5, 10450, role="validation")
    other_candidate = world.run("A", 1.45e-5, 10350, role="validation")
    found, problems = world.read()
    for run_id, seed in ((search_seed, 10100), (confirmation_seed, 10450),
                         (other_candidate, 10350)):  # fmt: skip
        assert listed_refused(found, problems, run_id, str(seed), "outside")
    assert by_id(found)[search_seed].seed == 10100  # what could be bound is listed


def test_a_confirmation_seed_is_bound_by_its_header(world) -> None:
    one = world.run("A", 1.45e-5, 10400, role="confirmation", candidate=1)
    two = world.run("A", 1.27e-5, 10400, role="confirmation", candidate=2, campaign="c2")
    found, problems = world.read()
    assert problems == []
    assert {e.run_id: e.candidate for e in found} == {one: 1, two: 2}


def test_a_rate_not_prescribed_is_refused(world) -> None:
    run_id = world.run("A", 1.5e-5, 10100, role="search", round_=1, prescribe=False)
    world.prescribed["c1/search/r1"].add(("A", 1.52e-5, 10100))
    found, problems = world.read()
    assert listed_refused(found, problems, run_id, "not prescribed", "c1/search/r1")
    no_step = World(world.root.parent / "other")
    run_id = no_step.run("A", 1.45e-5, 10300, prescribe=False)
    found, problems = no_step.read()
    assert listed_refused(found, problems, run_id, "c1/validation")


def test_prescribed_rates_compare_as_numbers(world) -> None:
    world.run("A", 1.45e-5, 10100, role="search", prescribe=False)
    found, problems = world.read({"c1/search/r0": {("A", 0.0000145, 10100)}})
    assert problems == [] and [e.state for e in found] == ["usable"]


def test_an_unknown_arm_is_refused(world) -> None:
    run_id = world.run("A", 1.45e-5, 10300)
    run_dir = world.root / "study3" / run_id
    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.optimizer.name = "sgd"
    OmegaConf.save(cfg, run_dir / "config.yaml")
    chash = config_hash(cfg)
    meta = json.loads((run_dir / "meta.json").read_text())
    (run_dir / "meta.json").write_text(json.dumps({**meta, "config_hash": chash}))
    journal = world.root / "launch/study3-c1.jsonl"
    edit_jsonl(journal, lambda rows: [{**r, "config_hash": chash} if r["event"] == "start" else r
                                      for r in rows])  # fmt: skip
    found, problems = world.read()
    assert listed_refused(found, problems, run_id, "unknown arm", "sgd")
    assert by_id(found)[run_id].arm is None


def test_a_run_without_an_issued_cell_is_refused(world) -> None:
    started = world.run("A", 1.45e-5, 10300)
    stray = world.root / "study3" / "20261026T999999Z-stray-s10300-abcdef12-a1"
    stray.mkdir(parents=True)
    (stray / "meta.json").write_text(json.dumps({"campaign": None}))
    no_header = world.run("A", 2.11e-5, 10300, head={}, prescribe=False)
    found, problems = world.read()
    assert states(found) == {started: "usable", stray.name: "refused", no_header: "refused"}
    assert listed_refused(found, problems, stray.name, "no start")
    assert listed_refused(found, problems, no_header, "header")
    assert by_id(found)[stray.name].cell.campaign is None


@pytest.mark.parametrize(
    "head, why",
    [
        (header(1, "validation", None, study=2), "study"),
        (header(3, "validation", None), "candidate"),
        (header(1, "discovery", None), "step"),
        ({"study": 3, "candidate": 1, "step": "validation", "round": None}, "versions record"),
        (header(1, "search", None), "role"),  # a validation config under a search step
        ({k: v for k, v in header(1, "validation", None).items() if k != "round"}, "round"),
    ],
)
def test_a_header_that_issues_no_such_cell_is_refused(world, head, why) -> None:
    run_id = world.run("A", 1.45e-5, 10300, head=head)
    found, problems = world.read()
    assert listed_refused(found, problems, run_id, why)


def test_the_recorded_campaign_must_be_the_journals(world) -> None:
    run_id = world.run("A", 1.45e-5, 10300)
    path = world.root / "study3" / run_id / "meta.json"
    meta = json.loads(path.read_text())
    meta["campaign"]["label"] = "A-1.45e-05/s10301"
    path.write_text(json.dumps(meta))
    found, problems = world.read()
    assert listed_refused(found, problems, run_id, "campaign")


def test_a_config_hash_that_is_not_the_cells_is_refused(world) -> None:
    run_id = world.run("A", 1.45e-5, 10300)
    journal = world.root / "launch/study3-c1.jsonl"
    edit_jsonl(journal, lambda rows: [{**r, "config_hash": "0" * 12} if r["event"] == "start"
                                      else r for r in rows])  # fmt: skip
    found, problems = world.read()
    assert listed_refused(found, problems, run_id, "config hash")
    run_dir = world.root / "study3" / run_id
    meta = json.loads((run_dir / "meta.json").read_text())
    (run_dir / "meta.json").write_text(json.dumps({**meta, "config_hash": "0" * 12}))
    found, problems = world.read()  # meta and journal agree, the config does not
    assert listed_refused(found, problems, run_id, "config hash")
    assert by_id(found)[run_id].arm is None  # an unbound config gives no fact


def test_a_start_whose_command_no_longer_binds_is_technical(world) -> None:
    run_id = world.run("A", 1.45e-5, 10300, end=None, directory=False)
    journal = world.root / "launch/study3-c1.jsonl"
    edit_jsonl(journal, lambda rows: [{**r, "command": r["command"][:2]} if r["event"] == "start"
                                      else r for r in rows])  # fmt: skip
    [e], problems = world.read()
    assert (e.run_id, e.state, e.counts, problems) == (run_id, "technical", True, [])
    assert any("config hash" in r for r in e.reasons)
    assert (e.label, e.attempt, e.cell.key, e.arm, e.lr) == (
        "A-1.45e-05/s10300",
        1,
        "c1/validation",
        None,
        None,
    )
    # a recovery round clears it: the next attempt decides
    world.recovery("study3-c1", "A-1.45e-05/s10300")
    second = world.run("A", 1.45e-5, 10300, attempt=2)
    found, problems = world.read()
    assert problems == []
    assert states(found) == {run_id: "recovered", second: "usable"}


# ------------------------------------------------------------------ the sealing boundary

SEALED = ("responses", "observations")


def seal(monkeypatch) -> list[str]:
    """From now on ``open``, ``io.open`` and ``Path.open``/``read_text``/``read_bytes`` fail on
    any path under a ``responses/`` or ``observations/`` folder; the list holds every such
    attempt."""
    attempts: list[str] = []

    def guard(path) -> None:
        if isinstance(path, int):
            return
        parts = Path(os.fsdecode(path)).parts
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


def test_the_reader_never_opens_responses_or_observations(world, monkeypatch) -> None:
    world.run("A", 1.45e-5, 10300, mmlu=2.0)
    world.run("A", 2.11e-5, 10300, mmlu=16.0)
    world.run("F", 1.3e-5, 10300, stop="gradient_spike", steps=7)
    world.run("F", 1.83e-5, 10300, end=None, terminal=False)
    world.run("P", 1.3e-5, 10300, end="technical", terminal=False, status="failed")
    conflict = world.run("P", 1.85e-5, 10300, end=None)
    world.end("study3-c1", "P-1.85e-05/s10300", 1, conflict, "conflict", problem=SENTINEL)
    world.run("A", 1.45e-5, 10100, role="search")
    world.run("A", 1.45e-5, 10120, role="validation")  # refused: outside its role's seeds
    broken = world.run("A", 1.45e-5, 10301)
    (world.root / "study3" / broken / "terminal.json").write_text("{")
    sealed = seal(monkeypatch)
    found, problems = world.read()
    assert sealed == []
    assert len(found) == 9 and len(problems) == 1
    assert sorted(e.state for e in found) == sorted(
        ["usable", "unusable", "scientific", "pending", "technical", "technical", "usable",
         "refused", "technical"])  # fmt: skip
    assert SENTINEL not in repr((found, problems))
    # the guard is live: a sealed file cannot be read under it
    with pytest.raises(AssertionError):
        next((world.root / "study3").glob("*/responses/*.jsonl")).read_text()


def test_the_reader_imports_no_gpu_path() -> None:
    code = ("import sys; import zero_field_refusal.analysis.study3_records; "
            "print([m for m in ('torch', 'transformers', 'zero_field_refusal.schedule', "
            "'zero_field_refusal.train', 'zero_field_refusal.launch', "
            "'zero_field_refusal.qualification', 'zero_field_refusal.analysis.stage_a') "
            "if m in sys.modules])")  # fmt: skip
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
