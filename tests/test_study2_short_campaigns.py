"""Study 2's short-regime stage as a whole: its configs, the stage's state (analysis.study2_short
``progress``, ``look_blocks``, ``read_fixation``), the campaigns written from it
(scripts/study2_short_campaigns.py, loadable by the launcher with every run's config checked
against the protocol), the prerequisites' driver and the summary (scripts/study2_short_summary.py).

Synthetic runs follow tests/test_study2_short.py's analytic world, made to conform to the
campaigns; the scripts read them through a patched ``load_trajectories``."""

import dataclasses
import importlib.util
import json
import math
import re
import runpy
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from test_study2_short import SCALE, THETA0, W_MAX, control, made, stage_a_reference, traj

from zero_field_refusal.analysis.stage_a import arm_of
from zero_field_refusal.analysis.study2_short import (
    ADAMW,
    CONFIRMATION_SEEDS,
    FA,
    FA_CONTROL_ARM,
    PREWARMED,
    VALIDATION_SEEDS,
    conformance,
    look_blocks,
    measures_rayleigh,
    prescribed,
    progress,
    read_fixation,
    seed_role,
)
from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.launch import load_campaign
from zero_field_refusal.protocol import load_run_config

SCRIPTS = Path(__file__).parents[1] / "scripts"
CAMPAIGNS, SUMMARY = SCRIPTS / "study2_short_campaigns.py", SCRIPTS / "study2_short_summary.py"
OPTIMIZERS = {ADAMW: [], FA: ["optimizer=adam_frozen"], PREWARMED: ["optimizer=adamw_prewarmed"]}
COMMIT = "0123456789abcdef0123456789abcdef01234567"


# ------------------------------------------------------------------ configs and arms


@pytest.mark.parametrize("arm", [ADAMW, FA, PREWARMED])
def test_the_regular_runs_are_stage_as_25_step_runs_but_name_and_role(arm) -> None:
    # The new discovery runs pool with Stage A's grids: everything they compute is the same.
    stage_a = load_config("experiments/study2_stage_a", ["schedule=short", *OPTIMIZERS[arm]])
    role = "study2_short.role=discovery"
    short = load_config("experiments/study2_short", [*OPTIMIZERS[arm], role])
    a, b = OmegaConf.to_container(stage_a), OmegaConf.to_container(short)
    assert (a.pop("experiment"), b.pop("experiment")) == ("study2_stage_a", "study2_short")
    assert a.pop("stage_a") == b.pop("study2_short") == {"role": "discovery"}
    assert a == b
    assert (short.protocol_version, short.train.steps) == (2, 25)
    assert list(short.schedule.evaluate_at) == [25]
    assert (short.schedule.policy, short.schedule.instability.grad_factor) == ("fixed", 10)
    assert short.eval.monitor.version == 2 and "rayleigh" not in short.eval
    assert arm_of(short) == arm


def test_a_regular_run_needs_its_role_and_the_first_blocks_measure_rayleigh() -> None:
    cfg = load_run_config("experiments/study2_short", ["optimizer.lr=1e-5", "seed=3000"])
    with pytest.raises(ValueError, match="study2_short.role"):
        config_hash(cfg)
    plain = load_config("experiments/study2_short", ["study2_short.role=confirmation"])
    rayleigh = load_config("experiments/study2_short_rayleigh", ["study2_short.role=confirmation"])
    a, b = OmegaConf.to_container(plain), OmegaConf.to_container(rayleigh)
    assert b["eval"].pop("rayleigh") == {"enabled": True, "responses": 100}
    assert a == b and rayleigh.experiment == "study2_short"  # the stage's records stay together
    assert [measures_rayleigh(s) for s in (2000, 2100, 3000, 3009, 3010)] == [
        False, False, True, True, False]  # fmt: skip


# ------------------------------------------------------------------ the synthetic stage

P = prescribed()


def conform(run):
    """A synthetic run as the campaigns make it: the prescribed model, schedule and monitor."""
    if run.arm == FA_CONTROL_ARM:
        instability = {**P["control"]["instability"],
                       "contamination_k": run.spec["instability"]["contamination_k"]}  # fmt: skip
        return dataclasses.replace(run, model=P["model"], protocol_version=3, role="full_control",
                                   spec={**P["control"], "instability": instability},
                                   monitor_spec=P["monitor"])  # fmt: skip
    monitor = {"version": 1} if run.monitor_version == 1 else P["monitor"]
    return dataclasses.replace(run, model=P["model"], protocol_version=2, spec=P["short"],
                               monitor_spec=monitor)  # fmt: skip


def fact(run) -> dict:
    control_run = run.arm == FA_CONTROL_ARM
    return {"role": "full_control" if control_run else seed_role(run.seed),
            "rayleigh": not control_run and measures_rayleigh(run.seed)}  # fmt: skip


def facts_of(runs) -> dict:
    return {run.run_id: fact(run) for run in runs}


def later(arm, batch, lr, seed):
    """A validation or confirmation run of the world, its K jittered by seed."""
    k = SCALE[(arm, batch)] * (lr / 1e-5) ** 1.5 * (0.98 + 0.01 * (seed % 5))
    return conform(traj(arm, batch, lr, seed, k, mmlu=0.3 * k, factual=0.1 * k))


def stage(runs):
    return progress(runs, THETA0, W_MAX, facts=facts_of(runs))


def make(step) -> list:
    runs = []
    for r in step["runs"]:
        if r["arm"] == FA_CONTROL_ARM:
            runs.append(conform(control(r["batch"], r["lr"], r["contamination_k"],
                                        0.1 * r["contamination_k"])))  # fmt: skip
        elif r["seed"] in (2000, 2001, 2002):
            runs.append(conform(made(r["arm"], r["batch"], r["lr"], r["seed"])))
        else:
            runs.append(later(r["arm"], r["batch"], r["lr"], r["seed"]))
    return runs


def until(runs, campaign: str) -> dict:
    """Make every campaign the stage names until it names ``campaign``."""
    for _ in range(12):
        state = stage(runs)
        if state["next"]["campaign"] == campaign:
            return state
        assert not state["problems"], state["problems"]
        runs += make(state["next"])
    raise AssertionError(f"never reached {campaign}")


def base() -> list:
    return [conform(r) for r in stage_a_reference()]


def blocks_of(rates, batch, seeds):
    return [later(arm, batch, lr, s) for s in seeds for arm, lrs in rates.items() for lr in lrs]


def test_the_stage_goes_from_discovery_to_validation_to_confirmation() -> None:
    runs = base()
    assert stage(runs)["next"]["campaign"] == "targeted"
    state = until(runs, "validation")
    step = state["next"]
    assert (step["batch"], step["U"]) == (5, 8.0) and not step["prewarmed_dropped"]
    assert step["rates"] == {ADAMW: [1e-5, 1.17e-5, 1.37e-5], FA: [1.27e-5, 1.49e-5, 1.74e-5],
                             PREWARMED: [1e-5, 1.17e-5, 1.37e-5]}  # fmt: skip
    assert len(step["runs"]) == 3 * 3 * 12  # three arms, three rates, seeds 2100-2111
    runs += make(step)[:-1]  # one validation run still missing: no verdict yet
    assert stage(runs)["next"]["campaign"] == "validation"
    state = until(runs, "confirmation")
    assert state["validation"]["passes"] and len(state["validation"]["jointly_supported"]) == 12
    assert set(state["validation_prewarmed_utility"]) == {1e-5, 1.17e-5, 1.37e-5}
    assert not any(b["runs"] for b in state["confirmation"])
    runs += blocks_of(step["rates"], 5, CONFIRMATION_SEEDS[:3])
    runs += blocks_of({ADAMW: step["rates"][ADAMW]}, 5, CONFIRMATION_SEEDS[3:4])
    blocks = stage(runs)["confirmation"]
    assert [b["complete"] for b in blocks[:5]] == [True, True, True, False, False]
    assert all(b["supported"] for b in blocks[:3]) and blocks[3]["runs"] == 3


def test_a_failed_validation_ends_the_stage_there() -> None:
    runs = base()
    state = until(runs, "validation")
    made_runs = make(state["next"])
    # Half the blocks' AdamW-25 runs read far below K*: fewer than 10 jointly supported.
    low = {s for s in VALIDATION_SEEDS[:6]}
    runs += [dataclasses.replace(r, evaluations=[{**r.evaluations[0], "K_benign": 1.0}])
             if r.arm == ADAMW and r.seed in low else r for r in made_runs]  # fmt: skip
    state = stage(runs)
    assert state["next"]["campaign"] == "validation_failed"
    assert "6 of 12 blocks jointly supported" in state["next"]["reasons"][0]


def test_records_off_the_campaigns_leave_the_stage_incomplete() -> None:
    runs = base()
    first = make(stage(runs)["next"])[:1]
    ok = stage([*runs, *first])
    assert ok["next"]["campaign"] == "targeted" and not ok["problems"]  # the rest of the round
    cases = {
        "role": lambda f: {**f, first[0].run_id: {"role": "validation", "rayleigh": False}},
        "rayleigh": lambda f: {**f, first[0].run_id: {"role": "discovery", "rayleigh": True}},
        "unread": lambda f: {k: v for k, v in f.items() if k != first[0].run_id},
    }
    for name, damage in cases.items():
        state = progress([*runs, *first], THETA0, W_MAX, facts=damage(facts_of([*runs, *first])))
        assert state["next"] == {"campaign": "incomplete"}, name
    v1 = dataclasses.replace(first[0], monitor_version=1, monitor_spec={"version": 1})
    assert "read another monitor" in conformance([v1], facts_of([v1]))[0]
    other = dataclasses.replace(first[0], spec={**P["short"], "train_steps": 30})
    assert "departs from the prescribed schedule in ['train_steps']" in conformance(
        [other], facts_of([other]))[0]  # fmt: skip
    protocol = dataclasses.replace(conform(control(5, 1e-5, 0.06, 0.0)), protocol_version=2)
    assert "under protocol 2" in conformance([protocol], facts_of([protocol]))[0]
    # A run the state machine did not prescribe stops every campaign until it is resolved.
    stray = conform(made(ADAMW, 5, 1.23e-5, 2000))
    state = stage([*runs, stray])
    assert state["next"] == {"campaign": "incomplete"} and "unexpected run" in state["problems"][0]


def test_later_runs_must_be_at_the_frozen_rates() -> None:
    runs = base()
    state = until(runs, "validation")
    stray = later(FA, 5, 1.3e-5, 2100)
    state = stage([*runs, stray])
    assert state["next"] == {"campaign": "incomplete"}
    assert "outside the frozen rates" in state["problems"][0]


# ------------------------------------------------------------------ the fixation record


def test_the_fixation_record_needs_every_value(tmp_path) -> None:
    good = {"w_max": 0.35, "levels": [0.995, 0.952], "se_floor": 0.01, "bias_allowance": 0.005}
    path = tmp_path / "fixation.json"
    path.write_text(json.dumps(good))
    assert read_fixation(path) == good
    for key in good:
        path.write_text(json.dumps({k: v for k, v in good.items() if k != key}))
        with pytest.raises(ValueError, match=key):
            read_fixation(path)
    for bad in ({"levels": [0.05, 0.05]}, {"levels": [0.99]}, {"w_max": 0}, {"se_floor": -1},
                {"bias_allowance": None}, {"w_max": True}):  # fmt: skip
        path.write_text(json.dumps({**good, **bad}))
        with pytest.raises(ValueError):
            read_fixation(path)


# ------------------------------------------------------------------ the looks' blocks


def rows(harmful: int, missing: int = 0) -> list[dict]:
    return [{"condition": "step-25", "step": 25, "prompts": 500, "harmful": harmful,
             "missing": missing}]  # fmt: skip


def confirmed(runs: list) -> dict:
    state = until(runs, "confirmation")
    runs += blocks_of(state["next"]["rates"], 5, CONFIRMATION_SEEDS[:2])
    return stage(runs)


def test_look_blocks_wait_for_labels_and_take_delta_from_the_brackets() -> None:
    state = confirmed(base())
    blocks, kstar = state["confirmation"], state["frozen"]["K_star"]
    assert [b["complete"] for b in look_blocks(blocks, {}, kstar)[:3]] == [False, False, False]
    assert look_blocks(blocks, {}, kstar)[0]["unlabeled"]
    brackets = {arm: s["bracket"] for arm, s in blocks[0]["supports"].items()}
    labels = {b[side]["run"]: rows(100 if arm == ADAMW else 50, missing=5)
              for arm, b in brackets.items() for side in ("lo", "hi")}  # fmt: skip
    first = look_blocks(blocks, labels, kstar)[0]
    assert first["complete"] and first["supported"]
    assert first["delta"] == pytest.approx(0.1)  # 0.2 against 0.1 at both ends
    assert first["delta_missing_harmful"] == pytest.approx(0.1)
    assert first["missing"] == {ADAMW: 10, FA: 10, PREWARMED: 10}
    assert set(first["h"]) == {ADAMW, FA, PREWARMED}
    # Without the pre-warmed arm's labels the block still waits: its three arms are labeled
    # together.
    partial = {k: v for k, v in labels.items()
               if k not in {brackets[PREWARMED][s]["run"] for s in ("lo", "hi")}}  # fmt: skip
    assert not look_blocks(blocks, partial, kstar)[0]["complete"]


# ------------------------------------------------------------------ the campaigns script


def theta0_dir(tmp_path: Path) -> Path:
    run = tmp_path / "theta0"
    run.mkdir(exist_ok=True)
    (run / "summary.json").write_text(json.dumps(THETA0))
    return run


def script(monkeypatch, tmp_path, path: Path, runs: list, *args: str) -> Path:
    """Run a script on ``runs`` (patched in for the run directories); returns its --out."""
    monkeypatch.setattr("zero_field_refusal.analysis.stage_a.load_trajectories",
                        lambda *roots: list(runs))  # fmt: skip
    by_dir = {run.run_dir: run for run in runs}
    monkeypatch.setattr("zero_field_refusal.analysis.study2_short.run_facts",
                        lambda run_dir: fact(by_dir[run_dir]))  # fmt: skip
    out = tmp_path / f"out-{len(list(tmp_path.glob('out-*')))}.yaml"
    common = ["--runs", str(tmp_path), "--theta0", str(theta0_dir(tmp_path))]
    monkeypatch.setattr("sys.argv", [str(path), *args, *common, "--out", str(out)])
    runpy.run_path(str(path), run_name="__main__")
    return out


def campaign(monkeypatch, tmp_path, runs, *args: str) -> list:
    return load_campaign(script(monkeypatch, tmp_path, CAMPAIGNS, runs, *args))[1]


def labels_of(runs) -> list[str]:
    return [run.label for run in runs]


def test_the_targeted_campaign_and_the_refusals(monkeypatch, tmp_path) -> None:
    runs = base()
    found = campaign(monkeypatch, tmp_path, runs, "targeted", "--w-max", str(W_MAX))
    assert len(found) == 15 and {run.experiment for run in found} == {"study2_short"}
    assert "frozen_adam_25-b5-1.27e-05/s2001" in labels_of(found)
    fa = next(run for run in found if run.cell == "frozen_adam_25-b5-1.27e-05")
    assert fa.overrides[:4] == ("optimizer=adam_frozen", "train.batch_size=5",
                                "optimizer.lr=1.27e-05", "study2_short.role=discovery")  # fmt: skip
    for args in (("controls", "--w-max", str(W_MAX)),  # another step is next
                 ("targeted",),  # no w_max
                 ("confirmation", "--blocks", "1", "10", "--w-max", str(W_MAX))):  # fmt: skip
        with pytest.raises(SystemExit):
            campaign(monkeypatch, tmp_path, runs, *args)
    with pytest.raises(SystemExit):  # Stage A's reference incomplete: no stream time
        campaign(monkeypatch, tmp_path, runs[1:], "targeted", "--w-max", str(W_MAX))
    split = campaign(monkeypatch, tmp_path, runs, "targeted", "--w-max", str(W_MAX),
                     "--seeds", "2002")  # fmt: skip
    assert len(split) == 5 and {run.seed for run in split} == {2002}


def test_the_controls_and_prewarmed_campaigns(monkeypatch, tmp_path) -> None:
    runs = base()
    until(runs, "controls")
    found = campaign(monkeypatch, tmp_path, runs, "controls", "--w-max", str(W_MAX))
    assert labels_of(found) == ["fa-control-b5-1.27e-05-u8/s2000",
                                "fa-control-b5-1.49e-05-u8/s2000",
                                "fa-control-b5-1.74e-05-u8/s2000"]  # fmt: skip
    assert {run.experiment for run in found} == {"study2_short_fa_controls"}
    bound = next(o for o in found[0].overrides if o.startswith("schedule.instability"))
    assert float(bound.split("=")[1]) == 0.01 * math.sqrt(40)  # written in full
    until(runs, "prewarmed")
    found = campaign(monkeypatch, tmp_path, runs, "prewarmed", "--w-max", str(W_MAX))
    assert len(found) == 6 * 3 and all("optimizer=adamw_prewarmed" in r.overrides for r in found)


def test_the_validation_and_confirmation_campaigns(monkeypatch, tmp_path) -> None:
    runs = base()
    until(runs, "validation")
    found = campaign(monkeypatch, tmp_path, runs, "validation", "--w-max", str(W_MAX))
    assert len(found) == 108 and {r.seed for r in found} == set(VALIDATION_SEEDS)
    assert all("study2_short.role=validation" in r.overrides for r in found)
    # The launcher grants a recovery round per experiment and cell name: no name recurs.
    assert "adamw_25-b5-1.17e-05-validation" in {r.cell for r in found}
    with pytest.raises(SystemExit):  # the validation has not passed yet
        campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", "1", "2",
                 "--w-max", str(W_MAX))  # fmt: skip
    until(runs, "confirmation")
    found = campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", "9", "12",
                     "--w-max", str(W_MAX))  # fmt: skip
    assert len(found) == 4 * 9 and {r.seed for r in found} == set(range(3008, 3012))
    measuring = [r for r in found if r.config == "experiments/study2_short_rayleigh"]
    assert {r.seed for r in measuring} == {3008, 3009}  # blocks 9 and 10 measure R, not 11, 12
    assert {"adamw_25-b5-1e-05-blocks-9-12", "adamw_25-b5-1e-05-blocks-9-12-rayleigh"} <= {
        r.cell for r in found}  # fmt: skip
    cfg = load_run_config(measuring[0].config, measuring[0].overrides)
    assert cfg.eval.rayleigh.enabled and cfg.study2_short.role == "confirmation"
    assert {r.experiment for r in found} == {"study2_short"}
    for blocks in (("0", "3"), ("70", "81"), ("5", "4")):
        with pytest.raises(SystemExit):
            campaign(monkeypatch, tmp_path, runs, "confirmation", "--blocks", *blocks,
                     "--w-max", str(W_MAX))  # fmt: skip


def test_the_prep_driver(monkeypatch, tmp_path) -> None:
    out = tmp_path / "prep.sh"
    args = ["prep", "--seeds", "2000", "2001", "--batch", "5", "--commit", COMMIT,
            "--stage0-run-id", "stage0-v3-test-01", "--out", str(out)]  # fmt: skip
    monkeypatch.setattr("sys.argv", [str(CAMPAIGNS), *args])
    runpy.run_path(str(CAMPAIGNS), run_name="__main__")
    text = out.read_text()
    subprocess.run(["bash", "-n", str(out)], check=True)
    assert f"SRC=/workspace/code/zero-field-refusal/{COMMIT}/source" in text
    commands = re.sub(r" *\\\n *", " ", text).splitlines()  # continued lines joined
    run = 'project-run "$SRC" python scripts/'
    assert (f"step env RUN_ID=stage0-v3-test-01 {run}stage0_checks.py experiments/stage0_checks "
            "model=qwen2.5-1.5b-instruct protocol_version=3") in commands  # fmt: skip
    assert (f"kept {run}generate_teacher.py experiments/study2_short_rayleigh --stream "
            "curvature_panel") in commands  # fmt: skip
    for seed in (2000, 2001):
        assert (f"step {run}generate_teacher.py experiments/study2_stage_a --stream training "
                f"--seeds {seed} --responses 125") in commands  # fmt: skip
        assert (f"kept {run}generate_teacher.py experiments/study2_stage_a "
                "optimizer=adam_frozen train.batch_size=5 --stream preconditioner "
                f"--seeds {seed}") in commands  # fmt: skip
    assert sum("stage0_checks" in c for c in commands) == 1
    out2 = tmp_path / "streams.sh"
    args = ["prep", "--seeds", "2100", "--batch", "20", "--commit", COMMIT, "--streams-only",
            "--out", str(out2)]  # fmt: skip
    monkeypatch.setattr("sys.argv", [str(CAMPAIGNS), *args])
    runpy.run_path(str(CAMPAIGNS), run_name="__main__")
    text = out2.read_text()
    assert "stage0_checks" not in text and "curvature_panel" not in text
    assert "--responses 500" in text and "train.batch_size=20" in text
    for bad in (["--seeds", "7"], ["--commit", "abc"]):
        monkeypatch.setattr("sys.argv", [str(CAMPAIGNS), "prep", "--seeds", "2000", "--batch",
                                         "5", "--commit", COMMIT, *bad, "--out",
                                         str(tmp_path / "bad.sh")])  # fmt: skip
        with pytest.raises(SystemExit):
            runpy.run_path(str(CAMPAIGNS), run_name="__main__")


def test_the_driver_keeps_a_stream_that_exists_and_stops_on_other_failures() -> None:
    spec = importlib.util.spec_from_file_location("campaigns", CAMPAIGNS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = module.prep_driver([2000], 5, COMMIT, None)
    kept = text[text.index("kept() {") : text.index("}\n", text.index("kept() {")) + 2]
    body = f"""set -euo pipefail
{kept}
exists() {{ echo "StreamError: x {module.EXISTS}" >&2; return 1; }}
broken() {{ echo "StreamError: partial directories" >&2; return 1; }}
kept exists
echo after-exists
kept broken
echo after-broken
"""
    found = subprocess.run(["bash", "-c", body], capture_output=True, text=True)
    assert "after-exists" in found.stdout and "after-broken" not in found.stdout
    assert found.returncode != 0


# ------------------------------------------------------------------ the summary script


def test_the_summary_prints_the_status_and_refuses_looks_without_a_fixation(
    monkeypatch, tmp_path, capsys
) -> None:
    runs = base()
    script(monkeypatch, tmp_path, SUMMARY, runs, "--w-max", str(W_MAX))
    shown = json.loads(capsys.readouterr().out)
    assert shown["next"]["campaign"] == "targeted" and len(shown["next"]["runs"]) == 15
    assert [[c["batch"], c["U"]] for c in shown["candidates"]][:2] == [[5, 8.0], [20, 8.0]]
    assert shown["attempts"][0]["arms"][ADAMW][0]["feasible"] is False
    with pytest.raises(SystemExit):
        script(monkeypatch, tmp_path, SUMMARY, runs, "--w-max", str(W_MAX), "--judge", "wildguard")
    with pytest.raises(SystemExit):
        script(monkeypatch, tmp_path, SUMMARY, runs)  # no w_max


def summary_module():
    spec = importlib.util.spec_from_file_location("summary", SUMMARY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_summary_reads_the_labels_of_the_brackets_once(monkeypatch, tmp_path) -> None:
    runs = base()
    state = confirmed(runs)
    module = summary_module()
    on_disk = []
    for run in runs:  # every run gets a directory with its config's prompt set
        run_dir = tmp_path / run.run_id
        run_dir.mkdir()
        (run_dir / "config.yaml").write_text("eval: {harmful: {prompt_set: harmful_evaluation}}\n"
                                             "data: {}\n")  # fmt: skip
        on_disk.append(dataclasses.replace(run, run_dir=run_dir))
    brackets = [s["bracket"][side]["run"] for s in state["confirmation"][0]["supports"].values()
                for side in ("lo", "hi")]  # fmt: skip
    labeled = set(brackets[:4])  # AdamW-25's and FA-25's brackets of block 1
    loaded, read = [], []
    monkeypatch.setattr(module, "label_path", lambda run_dir, judge: run_dir / f"{judge}.jsonl")
    for run_id in labeled:
        (tmp_path / run_id / "wildguard.jsonl").write_text("")

    def prompt_set(name, data):
        loaded.append(name)
        return [{"id": "p", "text": "t"}]

    def endpoint_labels(run_dir, judge, identity, prompts):
        read.append(run_dir.name)
        return rows(1)

    monkeypatch.setattr(module, "load_prompt_set", prompt_set)
    monkeypatch.setattr(module, "read_endpoint_labels", endpoint_labels)
    labels = module.read_labels(state, on_disk, "wildguard")
    assert set(labels) == labeled and sorted(read) == sorted(labeled)
    assert loaded == ["harmful_evaluation"]  # the prompt set is read once
    found = look_blocks(state["confirmation"], labels, state["frozen"]["K_star"])
    assert not found[0]["complete"] and found[0]["unlabeled"]  # the pre-warmed bracket waits


def test_the_summary_looks_on_labels() -> None:
    runs = base()
    state = confirmed(runs)
    module = summary_module()
    fixation = {"w_max": W_MAX, "levels": [0.995, 0.95], "se_floor": 0.01, "bias_allowance": 0.0}
    found = module.compute_looks(state, runs, {}, fixation, THETA0)
    assert found["available"] and found["looks"]["waiting"]["started"] == 0
    assert found["waiting_for_labels"] == [1, 2]
    labels = {r.run_id: rows(100 if r.arm == ADAMW else 50) for r in runs}
    found = module.compute_looks(state, runs, labels, fixation, THETA0)
    waiting = found["looks"]["waiting"]
    assert (waiting["started"], waiting["supported"], waiting["needs"]) == (2, 2, 24)
    assert found["missing_labels"] == {ADAMW: 0, FA: 0, PREWARMED: 0}
    assert found["utility"]["family"] and found["secondary"] is None
    early = until(base(), "validation")
    assert not module.compute_looks(early, runs, labels, fixation, THETA0)["available"]
