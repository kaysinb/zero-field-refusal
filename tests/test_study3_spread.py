"""Study 3's precommitted spread script (analysis/study3_spread.py): N from the sample SD of the
A–F block differences through the one N rule, the rounding of the spread, the refusal of a run
without a matching receipt or without its batch's acceptance, the route's cohort, and an output
that holds nothing else and passes the n record's checks. Synthetic blocks, counts and harmless
text only."""

import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf
from test_study3_blocks import pair

from zero_field_refusal.analysis import study3_spread
from zero_field_refusal.analysis.evidence import label_path
from zero_field_refusal.analysis.study3_decisions import (
    _Schema,
    cap_of,
    planning_power,
    sample_size,
    upper_sd,
)
from zero_field_refusal.analysis.study3_label_cli import LEDGER, _write_whole, label_hashes
from zero_field_refusal.analysis.study3_spread import (
    OUTPUT_FIELDS,
    Harm,
    SpreadRefused,
    harm_counts,
    spread,
    spread_pp,
    spread_result,
)
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import _content_hash, judge_identity
from zero_field_refusal.run_judging import receipt_path

L = 0.985
JUDGE = "wildguard"
IDENTITY = judge_identity(load_config(f"judge/{JUDGE}"))
IDS = [f"evaluation:{i}" for i in range(521)]
PROMPTS = {pid: f"Placeholder prompt {i}." for i, pid in enumerate(IDS)}
SCRIPT = "e" * 64


def synthetic(m: int, *, wide: bool = False, seed: int = 7) -> tuple[list[dict], dict, dict]:
    """``m`` blocks with A, F and P (the run ids, weights in [0, 1]), random harm counts (``wide``:
    A and F far apart, alternately), and the cohort they make."""
    rng = random.Random(seed)
    blocks, harm = [], {}
    for k in range(m):
        block = {"seed": 10300 + k, "arms": {}}
        for arm in "AFP":
            ends = {end: f"{arm}-{end}-{10300 + k}" for end in ("lo", "hi")}
            for run_id in ends.values():
                high = wide and arm == ("A" if k % 2 else "F")
                harmful = rng.randrange(50, 80) if high else rng.randrange(0, 30)
                harm[run_id] = Harm(521, harmful, rng.randrange(0, 4))
            block["arms"][arm] = {**ends, "w": rng.random()}
        blocks.append(block)
    run_ids = sorted(b["arms"][a][e] for b in blocks for a in "AF" for e in ("lo", "hi"))
    cohort = {"seeds": [b["seed"] for b in blocks], "run_ids": run_ids}
    return blocks, harm, cohort


def sd_by_hand(blocks: list[dict], harm: dict, first: str) -> float:
    """100·sd (ddof 1) of first − F, each arm (1 − w)·h_lo + w·h_hi with h = harmful/521."""

    def h(arm: dict) -> float:
        lo, hi = harm[arm["lo"]].harmful / 521, harm[arm["hi"]].harmful / 521
        return (1 - arm["w"]) * lo + arm["w"] * hi

    d = [h(b["arms"][first]) - h(b["arms"]["F"]) for b in blocks]
    return 100 * float(np.std(d, ddof=1))


def result(blocks, harm, cohort, pf=True, **kw) -> dict:
    return spread_result(blocks, blocks if pf else None, harm, level=L, candidate=1,
                         cohort=cohort, script=SCRIPT, **kw)  # fmt: skip


@pytest.mark.parametrize("wide", [False, True])
def test_n_is_the_rules_n_for_the_sample_sd(wide) -> None:
    """N = sample_size(SD, m, L) with SD = 100·sd(d_AF) at full precision, the cap C(N), the
    outcome by the 200 reserved seeds, P–F's power at N to 0.01, the spread to 0.5 pp."""
    blocks, harm, cohort = synthetic(24, wide=wide)
    out = result(blocks, harm, cohort)
    sd = sd_by_hand(blocks, harm, "A")
    assert out["N"] == sample_size(sd, 24, L) and out["cap"] == cap_of(out["N"])
    assert out["state"] == ("passed" if out["cap"] <= 200 else "failed")
    assert out["state"] == ("failed" if wide else "passed")
    assert out["spread_pp"] == math.floor(2 * sd + 0.5) / 2
    power = planning_power(out["N"], upper_sd(sd_by_hand(blocks, harm, "P"), 24), L)
    assert out["power_pf"] == round(power, 2)
    assert (out["m"], out["level"], out["cohort"]) == (24, L, cohort)
    assert result(blocks, harm, cohort, pf=False)["power_pf"] is None
    raised = result(blocks, harm, cohort, n_min=out["N"] + 7)  # the simulation's power short
    assert raised["N"] == out["N"] + 7 and raised["cap"] == cap_of(out["N"] + 7)


def test_the_spread_is_rounded_to_half_a_point() -> None:
    assert [spread_pp(x) for x in (0.2, 3.24, 3.25, 3.7499, 3.75, 5.0)] == [
        0.0, 3.0, 3.5, 3.5, 4.0, 5.0]  # fmt: skip


def test_the_output_holds_nothing_else_and_makes_an_n_record() -> None:
    """Exactly the output's fields, strict JSON, no unrounded spread anywhere; its fields copied
    into an ``n-c1`` record pass the record's own schema and word checks."""
    blocks, harm, cohort = synthetic(24)
    out = result(blocks, harm, cohort)
    assert set(out) == set(OUTPUT_FIELDS) and set(out["cohort"]) == {"seeds", "run_ids"}
    text = json.dumps(out, allow_nan=False)
    sd = sd_by_hand(blocks, harm, "A")
    assert repr(sd)[:8] not in text and str(round(sd, 3)) not in text
    numbers = [v for k, v in out.items() if isinstance(v, float)]
    assert sorted(numbers) == sorted([out["spread_pp"], out["level"], out["power_pf"]])
    assert out["power_pf"] == round(out["power_pf"], 2)
    record = {
        "kind": "n-c1", "state": out["state"], "written_at": "2026-11-04T00:00:00+00:00",
        "code": {"commit": "a" * 40, "gpu_code_sha256": "b" * 64}, "candidate": 1,
        "rates": {arm: {"lr_l": 1.3e-5, "lr_u": 1.8e-5} for arm in "AFP"},
        "active": ["A-F", "P-F"], "level": out["level"], "cohort": out["cohort"],
        "spread": {"script": {"path": "src/zero_field_refusal/analysis/study3_spread.py",
                              "sha256": out["script_sha256"], "root": "repo"},
                   "output": {"path": "study3/decisions/spread-c1.json", "sha256": "c" * 64}},
        "spread_pp": out["spread_pp"], "N": out["N"], "cap": out["cap"],
        "simulation": {"path": "study3/decisions/simulation.json", "sha256": "d" * 64},
    }  # fmt: skip
    check = _Schema("n-c1", "n", 1)
    check.record(record)
    assert check.problems == []


# ------------------------------------------------------------------ the harm of a run


def labeled_run(root: Path, name: str, labels: list, *, receipt: bool = True) -> Path:
    """A scheduled run that answered the 521 placeholder prompts at step 25, labeled ``labels``
    (None: unparsed), with its completed receipt (three rounds used) unless ``receipt`` is
    False."""
    run = root / name
    (run / "responses").mkdir(parents=True)
    cfg = {"run_type": "train", "schedule": {"policy": "fixed"}, "data": {},
           "eval": {"harmful": {"prompt_set": "harmful_evaluation", "limit": None}}}  # fmt: skip
    OmegaConf.save(OmegaConf.create(cfg), run / "config.yaml")
    ids_hash = hashlib.sha256(json.dumps(IDS).encode()).hexdigest()
    (run / "meta.json").write_text(json.dumps({"inputs": {"harmful_prompts": {
        "set": "harmful_evaluation", "count": len(IDS), "ids_sha256": ids_hash}}}))  # fmt: skip
    (run / "terminal.json").write_text(json.dumps({"stop": {"reason": "steps"},
                                                   "evaluations": [25]}))  # fmt: skip
    (run / "evaluations.jsonl").write_text(json.dumps({"step": 25, "condition": "step-25"}) + "\n")
    texts = {pid: f"Placeholder response {pid}." for pid in IDS}
    (run / "responses" / "harmful_evaluation.jsonl").write_text("".join(
        json.dumps({"condition": "step-25", "item_id": pid, "text": texts[pid]}) + "\n"
        for pid in IDS))  # fmt: skip
    path = label_path(run, JUDGE)
    path.parent.mkdir()
    path.write_text("".join(json.dumps({
        "item_id": f"step-25/{pid}", "label": label, **IDENTITY,
        "content_sha256": _content_hash(PROMPTS[pid], texts[pid])}) + "\n"
        for pid, label in zip(IDS, labels, strict=True)))  # fmt: skip
    if receipt:
        expected = [f"step-25/{pid}" for pid in IDS]
        receipt_path(run, JUDGE).write_text(json.dumps({
            "run_id": name, "judge": JUDGE, "identity": IDENTITY, "expected_items": expected,
            "items": len(expected), "rounds": 3, "rounds_allowed": 3,
            "missing": sum(label is None for label in labels),
            "labels_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))  # fmt: skip
    return run


LABELS = ["harmful_compliance"] * 100 + ["refusal"] * 390 + ["safe_non_refusal"] * 10 + [None] * 21


def acceptance(*runs: Path, **change) -> dict:
    """An acceptance of the labeling's ledger naming ``runs`` as their files are now."""
    return {"kind": "study3-labeling-acceptance", "batch": "b", "attempt": 1, "judge": JUDGE,
            "runs": {run.name: {**label_hashes(run, JUDGE), **change} for run in runs}}  # fmt: skip


def test_a_runs_harm_is_counted_from_its_accepted_labels(tmp_path) -> None:
    run = labeled_run(tmp_path, "run-a", LABELS)
    assert harm_counts(run, IDENTITY, PROMPTS, accepted=[acceptance(run)]) == Harm(521, 100, 21)


@pytest.mark.parametrize("how", ["none", "another label file", "another receipt", "another judge"])
def test_labels_without_their_batchs_acceptance_are_refused(tmp_path, how) -> None:
    """A receipted run counts only through an acceptance naming its label file and receipt as
    read, under this judge: otherwise refused, by its run id alone."""
    run = labeled_run(tmp_path, "run-c", LABELS)
    found = {"none": [], "another label file": [acceptance(run, labels_sha256="0" * 64)],
             "another receipt": [acceptance(run, receipt_sha256="0" * 64)],
             "another judge": [{**acceptance(run), "judge": "rubric_gemini"}]}[how]  # fmt: skip
    with pytest.raises(SpreadRefused) as refused:
        harm_counts(run, IDENTITY, PROMPTS, accepted=found)
    message = str(refused.value)
    assert message.startswith("run-c: its labels are not those of an accepted batch")
    assert not any(c.isdigit() for c in message)


def test_the_reader_reads_the_ledgers_acceptances(tmp_path, monkeypatch) -> None:
    """The script's own reader takes the acceptances from the ledger under the sealed directory:
    without one the run is refused, with one it is counted."""
    run = labeled_run(tmp_path, "run-d", LABELS)
    monkeypatch.setattr(study3_spread, "harmful_prompts", lambda cfg: PROMPTS)
    endpoint = SimpleNamespace(run_id=run.name, run_dir=run)
    with pytest.raises(SpreadRefused, match="run-d: its labels are not those of an accepted"):
        study3_spread._labels_reader(tmp_path / "sealed")(endpoint)
    _write_whole(tmp_path / "sealed" / LEDGER / "accepted-b-1.json", acceptance(run), once=True)
    assert study3_spread._labels_reader(tmp_path / "sealed")(endpoint) == Harm(521, 100, 21)


@pytest.mark.parametrize("how", ["no receipt", "label file changed"])
def test_a_run_without_a_matching_receipt_is_refused_by_its_id_alone(tmp_path, how) -> None:
    """Not labeled: refused, naming the run id and no count of its labels."""
    run = labeled_run(tmp_path, "run-b", LABELS, receipt=how != "no receipt")
    if how == "label file changed":
        with label_path(run, JUDGE).open("a") as f:
            f.write("\n")
    with pytest.raises(SpreadRefused) as refused:
        harm_counts(run, IDENTITY, PROMPTS, accepted=[acceptance(run)])
    message = str(refused.value)
    assert message.startswith("run-b: not labeled") and not any(c.isdigit() for c in message)
    assert refused.value.__cause__ is None and refused.value.__suppress_context__


# ------------------------------------------------------------------ the reader


def route(seeds=(10300, 10301, 10302), stage="choice") -> tuple[dict, list]:
    """A state at the choice of candidate 1 with the blocks of ``seeds`` supported for A, F and
    P (K 4.80 and 7.50 at K* 6.00: w = 1/2), and their endpoints."""
    endpoints = [e for s in seeds for arm in "AFP" for e in pair(arm, 4.8, 7.5, seed=s)]
    run_ids = sorted(e.run_id for e in endpoints if e.arm in "AF")
    state = {"stage": stage, "candidate": 1, "active": {"arms": ["A", "F", "P"]},
             "validation": {"blocks": 28},
             "step": {"kind": "decision", "record": "n-c1", "fields": {
                 "active": ["A-F", "P-F"],
                 "cohort": {"seeds": list(seeds), "run_ids": run_ids}}}}  # fmt: skip
    return state, endpoints


def read(monkeypatch, state, endpoints, harm_of) -> dict:
    simulation = {"state": "passed", "level": L,
                  "gates": {"power_af": {"power_short": False, "n_min": None}}}  # fmt: skip
    monkeypatch.setattr(study3_spread, "read_state", lambda *a, **k: (state, endpoints))
    monkeypatch.setattr(study3_spread, "read_decision", lambda *a, **k: (simulation, []))
    return spread("runs", candidate=1, theta0=None, expected_code="f" * 64, harm_of=harm_of,
                  script=SCRIPT)  # fmt: skip


def test_the_reader_takes_the_routes_cohort_and_refuses_another_state(monkeypatch) -> None:
    state, endpoints = route()
    counts = {e.run_id: Harm(521, 13 * i % 200, 0) for i, e in enumerate(endpoints)}
    asked = []
    out = read(monkeypatch, state, endpoints, lambda e: asked.append(e.arm) or counts[e.run_id])
    assert out["cohort"] == state["step"]["fields"]["cohort"] and out["m"] == 3
    assert sorted(asked) == sorted("AAFFPP" * 3)  # A, F and P at both rates, once each
    blocks = [{"seed": s, "arms": {arm: {"lo": lo.run_id, "hi": hi.run_id, "w": 0.5}
              for arm in "AFP" for lo, hi in [pair(arm, 4.8, 7.5, seed=s)]}}
              for s in (10300, 10301, 10302)]  # fmt: skip
    assert out["N"] == sample_size(sd_by_hand(blocks, counts, "A"), 3, L)
    other, _ = route(stage="validation")
    with pytest.raises(SpreadRefused, match="not at the choice"):
        read(monkeypatch, other, endpoints, counts.get)
    fewer, _ = route()
    fewer["step"]["fields"]["cohort"]["seeds"] = [10300, 10301]
    with pytest.raises(SpreadRefused, match="not the cohort"):
        read(monkeypatch, fewer, endpoints, counts.get)


def test_main_prints_the_path_and_the_sha256_alone(monkeypatch, tmp_path, capsys) -> None:
    blocks, harm, cohort = synthetic(24)
    out = result(blocks, harm, cohort)
    monkeypatch.setattr(study3_spread, "spread", lambda *a, **k: out)
    path = tmp_path / "spread.json"
    argv = ["--runs-root", str(tmp_path), "--theta0", str(tmp_path), "--candidate", "1",
            "--out", str(path)]  # fmt: skip
    study3_spread.main(argv)
    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"written {path}", hashlib.sha256(path.read_bytes()).hexdigest()]
    assert json.loads(path.read_text()) == out
    with pytest.raises(SystemExit):
        study3_spread.main(argv)  # written once
