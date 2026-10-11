"""The judge-check sample: strata chosen from K alone, paired prompts and seeds, blind items."""

import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from zero_field_refusal.judge_check import (
    draw_sample,
    round_prompts,
    runs_table,
    select_strata,
    selection_record,
    trained_responses,
)
from zero_field_refusal.protocol import load_run_config

REPO = Path(__file__).resolve().parents[1]
MAIN, SGD_100X5 = "stage1_main_1.5b", "stage1_sgd100x5_1.5b"
PROMPTS = {f"judged:{i}": f"Placeholder prompt {i}." for i in range(250)}
OPTIMIZER = {"sgd": ["optimizer=sgd"], "adam_frozen": ["optimizer=adam_frozen"], "adamw": []}
# cell-mean K per (arm, lr); every cell has seeds 0-4 unless listed in PILOTS
NORMAL = {("sgd", 0.1): 2.0, ("sgd", 0.3): 8.0, ("sgd", 1.0): 30.0,
          ("adam_frozen", 1e-5): 4.0, ("adam_frozen", 2e-5): 10.0, ("adam_frozen", 5e-5): 28.0,
          ("adamw", 1e-5): 3.0, ("adamw", 2e-5): 9.0, ("adamw", 5e-5): 25.0}  # fmt: skip


def write_run(root: Path, experiment: str, arm: str, lr: float, seed: int, k: float,
              status: str = "completed") -> None:  # fmt: skip
    config = "experiments/" + experiment
    overrides = [] if experiment == SGD_100X5 else OPTIMIZER[arm]
    cfg = load_run_config(config, [*overrides, f"optimizer.lr={lr}", f"seed={seed}"])
    run_dir = root / experiment / f"{arm}_lr{lr:g}_s{seed}"
    (run_dir / "responses").mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    (run_dir / "meta.json").write_text(json.dumps({"status": status}))
    (run_dir / "summary.json").write_text(json.dumps({"K_harmful": k}))
    lines = [json.dumps({"condition": "trained", "item_id": pid,
                         "text": f"{run_dir.name} answers {pid}"}) for pid in PROMPTS]  # fmt: skip
    (run_dir / "responses" / "harmful_judged.jsonl").write_text("\n".join(lines) + "\n")


def build(root: Path, cells: dict, experiment: str = MAIN, pilots: dict | None = None) -> Path:
    for (arm, lr), k in cells.items():
        for seed in range(5):
            write_run(root, experiment, arm, lr, seed, k * (1 + 0.02 * (seed - 2)))
    for (arm, lr), k in (pilots or {}).items():
        write_run(root, experiment, arm, lr, 0, k)
    return root


def strata_of(root: Path, **kwargs) -> dict:
    return {s.name: s for s in select_strata(runs_table(root, MAIN), main_experiment=MAIN,
                                             **kwargs)}  # fmt: skip


def test_each_stratum_takes_the_qualifying_cell_nearest_to_k_star(tmp_path) -> None:
    # A single-seed pilot widens SGD's range but cannot be a stratum's cell.
    root = build(tmp_path, NORMAL, pilots={("sgd", 3.0): 100.0, ("adamw", 3e-5): 10.0})
    strata = strata_of(root)
    assert [s for s in strata] == ["sgd", "adam_frozen", "adamw"]
    target = strata["sgd"].target_k
    assert math.isclose(target, math.sqrt(4.0 * 25.0), rel_tol=1e-3)  # common range [4, 25]
    assert {name: s.lrs for name, s in strata.items()} == {
        "sgd": (0.3,), "adam_frozen": (2e-5,), "adamw": (2e-5,)}  # fmt: skip
    assert all(s.choice == "nearest" and s.criteria_apply for s in strata.values())


def test_a_far_nearest_cell_gives_way_to_the_bracketing_pair(tmp_path) -> None:
    cells = {k: v for k, v in NORMAL.items() if k != ("sgd", 0.3)}
    cells[("sgd", 1.0)] = 40.0
    stratum = strata_of(build(tmp_path, cells))["sgd"]
    assert (stratum.lrs, stratum.choice) == ((0.1, 1.0), "bracketing")


def test_frozen_adam_far_from_k_star_is_drawn_without_criteria(tmp_path) -> None:
    cells = {k: v for k, v in NORMAL.items() if k[0] != "adam_frozen"}
    cells |= {("adam_frozen", 1e-4): 60.0, ("adam_frozen", 2e-4): 80.0}
    strata = strata_of(build(tmp_path, cells))
    frozen = strata["adam_frozen"]
    assert (frozen.lrs, frozen.choice, frozen.criteria_apply) == ((1e-4,), "nearest_far", False)
    assert strata["sgd"].criteria_apply and strata["adamw"].criteria_apply


def test_without_k_star_sgd_and_adamw_come_from_their_nearest_pair(tmp_path) -> None:
    cells = {("sgd", 0.03): 0.5, ("sgd", 0.1): 2.0, ("adamw", 1e-5): 5.0, ("adamw", 5e-5): 25.0,
             ("adam_frozen", 1e-5): 3.0, ("adam_frozen", 5e-5): 20.0}  # fmt: skip
    strata = strata_of(build(tmp_path, cells))
    assert strata["sgd"].lrs == (0.1,) and strata["adamw"].lrs == (1e-5,)
    assert strata["sgd"].choice == "pair_without_k_star" and math.isnan(strata["sgd"].target_k)
    assert math.isclose(strata["adam_frozen"].target_k, math.sqrt(10.0), rel_tol=1e-3)
    assert strata["adam_frozen"].lrs == (1e-5,)
    record = selection_record(list(strata.values()), 1, 0)
    assert record["strata"][0]["target_k"] is None
    json.dumps(record, allow_nan=False)


def test_a_stratum_without_a_complete_cell_stops_the_draw(tmp_path) -> None:
    root = build(tmp_path, {k: v for k, v in NORMAL.items() if k[0] != "adam_frozen"},
                 pilots={("adam_frozen", 2e-5): 10.0})  # fmt: skip
    with pytest.raises(ValueError, match="adam_frozen"):
        strata_of(root)


def test_failed_runs_and_runs_without_k_do_not_count(tmp_path) -> None:
    root = build(tmp_path, NORMAL)
    write_run(root, MAIN, "sgd", 0.6, 0, 9.0, status="failed")
    write_run(root, MAIN, "sgd", 0.7, 0, float("nan"))
    assert set(runs_table(root, MAIN)["lr"]) == {lr for _, lr in NORMAL}


def test_the_100x5_stratum_is_chosen_against_its_own_k_star(tmp_path) -> None:
    root = build(tmp_path, NORMAL)
    build(root, {("sgd", 0.03): 3.0, ("sgd", 0.1): 12.0, ("sgd", 0.3): 40.0}, SGD_100X5)
    strata = select_strata(runs_table(root, MAIN), runs_table(root, SGD_100X5),
                           main_experiment=MAIN, experiment_100x5=SGD_100X5)  # fmt: skip
    extra = strata[-1]
    assert (extra.name, extra.arm, extra.experiment, extra.lrs) == (
        "sgd_100x5", "sgd", SGD_100X5, (0.1,))  # fmt: skip


def draw(root: Path, round_: int = 1, seed: int = 0):
    runs = {MAIN: runs_table(root, MAIN)}
    strata = select_strata(runs[MAIN], main_experiment=MAIN)

    def responses(experiment: str, run_id: str) -> dict:
        return trained_responses(root / experiment / run_id)

    return draw_sample(strata, runs, responses, PROMPTS, round_=round_, seed=seed)


def test_prompts_and_seeds_are_paired_across_strata_and_the_items_are_blind(tmp_path) -> None:
    root = build(tmp_path, NORMAL)
    blinded, key = draw(root)
    assert len(blinded) == 300 and len(key) == 300
    assert Counter(entry["stratum"] for entry in key.values()) == {
        "sgd": 100, "adam_frozen": 100, "adamw": 100}  # fmt: skip
    by_stratum = {}
    for entry in key.values():
        by_stratum.setdefault(entry["stratum"], {})[entry["prompt_id"]] = entry["seed"]
    assert by_stratum["sgd"] == by_stratum["adam_frozen"] == by_stratum["adamw"]
    assert Counter(by_stratum["sgd"].values()) == {s: 20 for s in range(5)}
    item = blinded[0]
    assert set(item) == {"item_id", "prompt", "response"}
    origin = key[item["item_id"]]
    assert item["response"] == f"{origin['run_id']} answers {origin['prompt_id']}"
    assert item["prompt"] == PROMPTS[origin["prompt_id"]]
    strata_in_order = [key[b["item_id"]]["stratum"] for b in blinded]
    assert len(set(strata_in_order[:100])) == 3  # shuffled across strata


def test_a_bracketing_pair_splits_each_seed_group_evenly(tmp_path) -> None:
    cells = {k: v for k, v in NORMAL.items() if k != ("sgd", 0.3)}
    cells[("sgd", 1.0)] = 40.0
    blinded, key = draw(build(tmp_path, cells))
    sgd = [e for e in key.values() if e["stratum"] == "sgd"]
    assert Counter((e["lr"], e["seed"]) for e in sgd) == {
        (lr, s): 10 for lr in (0.1, 1.0) for s in range(5)}  # fmt: skip


def test_round_two_draws_new_prompts_and_draws_are_reproducible(tmp_path) -> None:
    first, second = round_prompts(list(PROMPTS), 1, 0), round_prompts(list(PROMPTS), 2, 0)
    assert len(first) == len(second) == 100 and not set(first) & set(second)
    assert round_prompts(list(reversed(PROMPTS)), 1, 0) == first  # order of ids is irrelevant
    with pytest.raises(ValueError, match="Round 3"):
        round_prompts(list(PROMPTS), 3, 0)
    root = build(tmp_path, NORMAL)
    assert draw(root) == draw(root)


def test_the_script_never_overwrites_a_drawn_sample(tmp_path) -> None:
    out = tmp_path / "sample"
    out.mkdir()
    script = REPO / "scripts" / "make_judge_check_sample.py"
    done = subprocess.run([sys.executable, str(script), str(tmp_path), "--out", str(out)],
                          capture_output=True, text=True)  # fmt: skip
    assert done.returncode != 0 and "never overwritten" in done.stderr
