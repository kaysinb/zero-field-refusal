"""Study 5's run without momentum: scripts/study5_beta0_analysis.py end to end on synthetic runs
and labels (the design's 35 runs with the frozen-Adam reference's harm, so the registered rules
give (i), clear), written once with numbers only on stdout and no text stored; a run at another
rate listed and unused; a rate record that is not the rule's refused."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_study5_adamw_analysis import (
    IDENTITY,
    IDS,
    PROMPTS,
    SENTINEL,
    SET,
    dump,
    dump_config,
    lines,  # fmt: skip
    script_module,
    sha,
)
from test_study5_beta0_campaigns import rate_record, table_rows

from zero_field_refusal import beta0
from zero_field_refusal.analysis import study5_analysis_cli as cli
from zero_field_refusal.analysis import study5_beta0 as sb
from zero_field_refusal.judge import _content_hash
from zero_field_refusal.paths import REPO_ROOT

script = script_module(REPO_ROOT / "scripts" / "study5_beta0_analysis.py", "study5_beta0_script")
REFERENCE_FILE = REPO_ROOT / beta0.REFERENCE_FILE
REFERENCE = json.loads(REFERENCE_FILE.read_text())


def count_of(seed: int, order: str) -> int:
    s = REFERENCE["seeds"][str(seed)]
    h = s["h_stream"] if order == "stream" else s["h_stream"] + s["edits"][order]["dh"]
    return round(h / 100 * 521)


def write_run(root: Path, seed: int, order: str, lr: float, *, k_only: bool = False) -> Path:
    name = f"s5-b0-{'pilot' if k_only else order}-{lr!r}-s{seed}"
    run_dir = root / "study5" / name
    edit = beta0.edit_of(order)
    train = {"steps": 25, "batch_size": 20}
    if order == "reorder":
        train["order_seed"] = seed
    elif edit is not None:
        train["order_edit"] = {"kind": "shuffle", "first": edit[0], "last": edit[1],
                               "draw": edit[2]}  # fmt: skip
    dump_config(run_dir, {"run_type": "train", "seed": seed, "experiment": "study5",
                          "eval": {"harmful": {"prompt_set": SET},
                                   **({"k_only": True} if k_only else {})},
                          "optimizer": {"name": "adam_frozen", "lr": lr, "beta1": 0.0},
                          "schedule": {"policy": "fixed", "evaluate_at": [25]},
                          "study5": {"role": "beta0"}, "train": train})  # fmt: skip
    dump(run_dir / "meta.json", {"run_id": name, "inputs": {"harmful_prompts": {
        "set": SET, "count": 521, "ids_sha256": sha(json.dumps(IDS).encode())}}})  # fmt: skip
    dump(run_dir / "terminal.json", {"steps": 25, "stop": {"reason": "steps"},
                                     "evaluations": [25]})  # fmt: skip
    lines(run_dir / "evaluations.jsonl", [{"step": 25, "condition": "step-25", "K_benign": 5.0,
                                           "K_harmful": 6.0, "benchmark_accuracy": 0.55,
                                           "factual_accuracy": 0.99}])  # fmt: skip
    n = count_of(seed, order)
    responses, labels = [], []
    for j, pid in enumerate(IDS):
        text = f"Reply {SENTINEL} {j}"
        responses.append({"condition": "step-25", "item_id": pid, "text": text})
        label = "harmful_compliance" if j < n else "refusal"
        labels.append({"item_id": f"step-25/{pid}", **IDENTITY,
                       "content_sha256": _content_hash(PROMPTS[pid], text), "label": label,
                       "created_at": "2026-11-20T12:00:00+00:00"})  # fmt: skip
    lines(run_dir / "responses" / f"{SET}.jsonl", responses)
    lines(run_dir / "judge" / "wildguard.jsonl", labels)
    items = [f"step-25/{pid}" for pid in IDS]
    dump(run_dir / "judge" / "wildguard.receipt.json", {
        "judge": "wildguard", "identity": IDENTITY, "expected_items": items, "items": len(items),
        "rounds": 1, "missing": 0,
        "labels_sha256": sha((run_dir / "judge" / "wildguard.jsonl").read_bytes()),
        "completed_at": "2026-11-21T00:00:00+00:00"})  # fmt: skip
    return run_dir


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(cli, "harmful_prompts", lambda cfg: PROMPTS)


def call(argv) -> tuple[int, str]:
    try:
        return script.main(argv), ""
    except SystemExit as exc:
        return 1, str(exc.code)


def world(tmp_path: Path) -> tuple[Path, Path, float]:
    rate = rate_record(tmp_path / "rate" / "rate.json")
    eta0 = json.loads(rate.read_text())["choice"]["eta0"]
    root = tmp_path / "runs"
    for seed, order in sb.design():
        write_run(root, seed, order, eta0)
    write_run(root, 50000, "early-0", 2.6e-05)  # another rate: listed, unused
    return root, rate, eta0


def test_the_script_end_to_end(tmp_path, patched, capsys):
    root, rate, eta0 = world(tmp_path)
    out = tmp_path / "record.json"
    argv = ["--runs", str(root), "--rate", str(rate), "--reference", str(REFERENCE_FILE),
            "--out", str(out)]  # fmt: skip
    assert call(argv) == (0, "")
    stdout = capsys.readouterr().out
    data = out.read_bytes()
    assert SENTINEL not in data.decode() and SENTINEL not in stdout
    assert stdout.splitlines()[-1] == f"record sha256 {sha(data)}"
    record = json.loads(data)
    assert record["eta0"] == eta0 and record["missing"] == []
    assert len(record["selection"]["keyed"]) == 35 and len(record["selection"]["other"]) == 1
    assert record["result"]["outcome"] == "(i)" and record["result"]["strength"] == "clear"
    assert record["statistics"]["F"] == pytest.approx(REFERENCE["F"], abs=0.2)
    assert "outcome (i) strength clear" in stdout
    assert call(argv)[0] == 1 and out.read_bytes() == data  # written once


def test_the_script_refuses_a_rate_that_is_not_the_rules(tmp_path, patched):
    root, rate, _ = world(tmp_path)
    record = json.loads(rate.read_text())
    record["choice"]["eta0"] = 2.0e-05
    rate.write_text(json.dumps(record))
    code, why = call(["--runs", str(root), "--rate", str(rate), "--reference",
                      str(REFERENCE_FILE), "--out", str(tmp_path / "r.json")])  # fmt: skip
    assert code == 1 and "rule" in why and not (tmp_path / "r.json").exists()
    assert table_rows()  # the campaign test's pilot table is the rate record's


def m_targets():
    return beta0.targets_from(json.loads((REPO_ROOT / beta0.TARGETS_FILE).read_text()))


def test_eta0_on_a_grid_rate_keys_the_main_runs_not_the_pilots(tmp_path, patched):
    """η₀ equal to the pilot grid's 1.3e-05, with the two pilot seeds' K-only runs at that rate
    beside the main runs: the pilots are listed, never keyed."""
    targets = m_targets()
    rows = table_rows(lambda s, lr: targets[s] * (lr / 1.3e-05) ** 2)
    rate = rate_record(tmp_path / "rate" / "rate.json", rows)
    eta0 = json.loads(rate.read_text())["choice"]["eta0"]
    assert eta0 == 1.3e-05
    root = tmp_path / "runs"
    for seed, order in sb.design():
        write_run(root, seed, order, eta0)
    for seed in beta0.PILOT_SEEDS:  # the pilot's runs at the same rate, K only
        write_run(root, seed, "stream", eta0, k_only=True)
    out = tmp_path / "record.json"
    assert call(["--runs", str(root), "--rate", str(rate), "--reference", str(REFERENCE_FILE),
                 "--out", str(out)]) == (0, "")  # fmt: skip
    record = json.loads(out.read_text())
    assert sorted(record["selection"]["pilot"]) == [
        f"s5-b0-pilot-1.3e-05-s{s}" for s in beta0.PILOT_SEEDS]  # fmt: skip
    assert record["selection"]["keyed"]["s50000/stream"] == "s5-b0-stream-1.3e-05-s50000"
    assert record["result"]["outcome"] == "(i)" and record["result"]["technical"] == []


# ------------------------------------------------------------------ the addendum: edits not run


def stopped_world(tmp_path: Path) -> tuple[Path, Path, float]:
    """The design with 50005's stream-order run stopped at update 8 (a gradient spike, no
    step-25 evaluation, no labels) and none of its edits run."""
    rate = rate_record(tmp_path / "rate" / "rate.json")
    eta0 = json.loads(rate.read_text())["choice"]["eta0"]
    root = tmp_path / "runs"
    for seed, order in sb.design():
        if seed == 50005 and order != "stream":
            continue
        run_dir = write_run(root, seed, order, eta0)
        if seed == 50005:
            dump(run_dir / "terminal.json", {"steps": 8, "stop": {"reason": "gradient_spike"},
                                             "evaluations": []})  # fmt: skip
            (run_dir / "evaluations.jsonl").unlink()
            for part in ("judge", "responses"):
                for f in sorted((run_dir / part).iterdir()):
                    f.unlink()
                (run_dir / part).rmdir()
    return root, rate, eta0


def run_script(tmp_path: Path, root: Path, rate: Path, name: str) -> dict:
    out = tmp_path / name
    argv = ["--runs", str(root), "--rate", str(rate), "--reference", str(REFERENCE_FILE),
            "--out", str(out)]  # fmt: skip
    assert call(argv) == (0, "")
    return json.loads(out.read_text())


def test_edits_never_run_after_a_stopped_stream_order_run_are_not_run(tmp_path, patched):
    root, rate, _ = stopped_world(tmp_path)
    record = run_script(tmp_path, root, rate, "record.json")
    result = record["result"]
    edits = sorted(f"s50005/{o}" for o in beta0.orders_of(50005) if o != "stream")
    assert result["technical"] == [] and result["not_run"] == edits
    assert result["stopped_stream"] == ["s50005/stream"] and result["conditional"] is True
    assert result["outcome"] == "(i)" and result["seeds_per_class"] == {"F": 5, "E": 5, "L": 5}
    assert record["selection"]["stubs"] == {} and record["selection"]["not_ended"] == []


def test_an_edit_started_and_not_ended_is_a_technical_gap_not_not_run(tmp_path, patched):
    root, rate, eta0 = stopped_world(tmp_path)
    run_dir = write_run(root, 50005, "early-0", eta0)
    (run_dir / "terminal.json").unlink()  # the run started and never ended
    record = run_script(tmp_path, root, rate, "record.json")
    result = record["result"]
    assert result["outcome"] == "withheld" and result["technical"] == ["s50005/early-0"]
    assert "s50005/early-0" not in result["not_run"] and len(result["not_run"]) == 4
    assert record["selection"]["stubs"] == {"s50005/early-0": [run_dir.name]}
    assert record["deltas"]["s50005/early-0"]["state"] == "technical"


@pytest.mark.parametrize("broken", ["no config", "config unreadable"])
def test_a_run_directory_that_cannot_be_keyed_leaves_no_edit_not_run(tmp_path, patched, broken):
    root, rate, eta0 = stopped_world(tmp_path)
    run_dir = write_run(root, 50005, "late-1", eta0)
    if broken == "no config":
        (run_dir / "config.yaml").unlink()
    else:
        (run_dir / "config.yaml").write_text("{: not yaml")
    record = run_script(tmp_path, root, rate, "record.json")
    result = record["result"]
    assert result["not_run"] == [] and result["outcome"] == "withheld"
    assert len(result["technical"]) == 5  # the five edits of 50005, none known to be not run
    assert [u["run_id"] for u in record["selection"]["unreadable"]] == [run_dir.name]
