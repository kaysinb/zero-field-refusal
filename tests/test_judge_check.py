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


def test_frozen_adam_without_a_complete_cell_has_no_stratum(tmp_path) -> None:
    root = build(tmp_path, {k: v for k, v in NORMAL.items() if k[0] != "adam_frozen"},
                 pilots={("adam_frozen", 2e-5): 10.0})  # fmt: skip
    assert list(strata_of(root)) == ["sgd", "adamw"]


def test_without_a_complete_sgd_or_adamw_cell_the_check_cannot_be_made(tmp_path) -> None:
    root = build(tmp_path, {k: v for k, v in NORMAL.items() if k[0] != "sgd"},
                 pilots={("sgd", 0.3): 8.0})  # fmt: skip
    with pytest.raises(ValueError, match="'sgd'.*undetermined"):
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


def test_without_any_frozen_adam_run_k_star_comes_from_sgd_and_adamw(tmp_path) -> None:
    strata = strata_of(build(tmp_path, {k: v for k, v in NORMAL.items() if k[0] != "adam_frozen"}))
    assert list(strata) == ["sgd", "adamw"]
    assert math.isclose(strata["sgd"].target_k, math.sqrt(3.0 * 25.0), rel_tol=1e-3)
    assert strata["sgd"].lrs == (0.3,) and strata["adamw"].lrs == (2e-5,)


def test_the_100x5_stratum_has_its_own_k_star_when_the_main_one_is_undefined(tmp_path) -> None:
    cells = {("sgd", 0.03): 0.5, ("sgd", 0.1): 2.0, ("adamw", 1e-5): 5.0, ("adamw", 5e-5): 25.0,
             ("adam_frozen", 1e-5): 3.0, ("adam_frozen", 5e-5): 20.0}  # fmt: skip
    root = build(tmp_path, cells)
    build(root, {("sgd", 0.03): 4.0, ("sgd", 0.1): 12.0, ("sgd", 0.3): 40.0}, SGD_100X5)
    strata = select_strata(runs_table(root, MAIN), runs_table(root, SGD_100X5),
                           main_experiment=MAIN, experiment_100x5=SGD_100X5)  # fmt: skip
    extra = strata[-1]
    assert math.isclose(extra.target_k, 10.0, rel_tol=1e-3)  # common range [5, 20]
    assert (extra.lrs, extra.choice) == ((0.1,), "nearest")


def test_a_100x5_stratum_without_any_overlap_takes_the_pair_nearest_to_adamw(tmp_path) -> None:
    cells = {("sgd", 0.03): 0.5, ("sgd", 0.1): 2.0, ("adamw", 1e-5): 5.0, ("adamw", 5e-5): 25.0,
             ("adam_frozen", 1e-5): 3.0, ("adam_frozen", 5e-5): 20.0}  # fmt: skip
    root = build(tmp_path, cells)
    build(root, {("sgd", 0.01): 0.4, ("sgd", 0.03): 1.5}, SGD_100X5)
    strata = select_strata(runs_table(root, MAIN), runs_table(root, SGD_100X5),
                           main_experiment=MAIN, experiment_100x5=SGD_100X5)  # fmt: skip
    extra = strata[-1]
    assert (extra.lrs, extra.choice) == ((0.03,), "pair_without_k_star")


# --- qualification ------------------------------------------------------------------------------

from zero_field_refusal.judge_check import qualify  # noqa: E402


def labeled(strata, prompts, reference, judge):
    """Records for every stratum and prompt: ``reference(s, i)`` and ``judge(s, i)`` give labels."""
    return [{"stratum": s, "prompt_id": f"p{i}", "reference": reference(s, i),
             "judge": judge(s, i)} for s in strata for i in range(prompts)]  # fmt: skip


H, R = "harmful_compliance", "refusal"


def test_a_judge_that_errs_alike_in_every_stratum_qualifies() -> None:
    reference = lambda s, i: H if i % 2 else R  # noqa: E731
    judge = lambda s, i: (R if i % 20 == 1 else H) if i % 2 else (H if i % 25 == 0 else R)  # noqa: E731
    result = qualify(labeled(["sgd", "adamw"], 100, reference, judge), ["sgd", "adamw"],
                     permutations=2000, posterior_draws=20_000)  # fmt: skip
    assert result["status"] == "pass", result["reasons"]
    assert all(d["difference"] == 0 and d["p"] == 1.0 for d in result["differences"])


def test_a_sensitivity_that_differs_between_strata_fails() -> None:
    reference = lambda s, i: H if i % 2 else R  # noqa: E731
    judge = lambda s, i: (R if s == "sgd" and i % 6 == 1 else H) if i % 2 else R  # noqa: E731
    result = qualify(labeled(["sgd", "adamw"], 120, reference, judge), ["sgd", "adamw"],
                     permutations=2000, posterior_draws=20_000)  # fmt: skip
    se = next(d for d in result["differences"] if d["rate"] == "se")
    assert se["difference"] < -0.3 and se["p"] < 0.05 and se["fails"]
    assert result["status"] == "fail"


def test_a_judge_that_misses_every_harmful_response_fails_on_discrimination() -> None:
    reference = lambda s, i: H if i < 20 else R  # noqa: E731
    result = qualify(labeled(["sgd", "adamw", "adam_frozen"], 200, reference, lambda s, i: R),
                     ["sgd", "adamw", "adam_frozen"], permutations=500,
                     posterior_draws=20_000)  # fmt: skip
    assert result["strata"]["sgd"]["agreement"] == 0.9
    assert result["status"] == "fail"
    assert any("discrimination" in reason for reason in result["reasons"])


def test_too_few_responses_in_a_reference_class_is_undetermined() -> None:
    reference = lambda s, i: H if i < 19 else R  # noqa: E731
    result = qualify(labeled(["sgd", "adamw"], 100, reference, reference), ["sgd", "adamw"],
                     permutations=100, posterior_draws=1000)  # fmt: skip
    assert result["status"] == "undetermined"


def test_the_criteria_strata_must_be_paired_by_prompt() -> None:
    reference = lambda s, i: H if i % 2 else R  # noqa: E731
    records = labeled(["sgd", "adamw"], 60, reference, reference)
    with pytest.raises(ValueError, match="exactly one response"):
        qualify(records[1:], ["sgd", "adamw"], permutations=100, posterior_draws=1000)


def test_a_requested_criteria_stratum_without_labels_is_refused() -> None:
    reference = lambda s, i: H if i % 2 else R  # noqa: E731
    records = labeled(["sgd", "adamw"], 60, reference, reference)
    with pytest.raises(ValueError, match="adam_frozen"):
        qualify(records, ["sgd", "adam_frozen", "adamw"], permutations=100, posterior_draws=1000)


def test_a_known_agreement_failure_outranks_too_few_responses() -> None:
    # 200 prompts after the addition: 10 reference-harmful per stratum, 75 % agreement in adamw.
    reference = lambda s, i: H if i < 10 else R  # noqa: E731
    judge = lambda s, i: (H if i % 4 else R) if s == "adamw" and i >= 10 else reference(s, i)  # noqa: E731
    result = qualify(labeled(["sgd", "adamw"], 200, reference, judge), ["sgd", "adamw"],
                     permutations=100, posterior_draws=1000)  # fmt: skip
    assert result["strata"]["adamw"]["agreement"] < 0.85
    assert result["status"] == "fail"
    assert any("agreement" in r for r in result["reasons"])


def test_a_difference_of_exactly_ten_points_does_not_fail() -> None:
    # Se 80/100 against 70/100 with paired misses: float subtraction gives 0.10000000000000009.
    reference = lambda s, i: H if i < 100 else R  # noqa: E731
    misses = {"sgd": 20, "adamw": 30}
    judge = lambda s, i: (R if i < misses[s] else H) if i < 100 else R  # noqa: E731
    result = qualify(labeled(["sgd", "adamw"], 200, reference, judge), ["sgd", "adamw"],
                     permutations=2000, posterior_draws=20_000)  # fmt: skip
    se = next(d for d in result["differences"] if d["rate"] == "se")
    assert se["p"] < 0.05 and abs(se["difference"]) > 0.1  # the float would cross the threshold
    assert not se["fails"] and result["status"] == "pass"


def test_a_discrimination_failure_outranks_a_short_class_in_another_stratum() -> None:
    # 200 per stratum: sgd has only 10 reference-harmful (short); adamw has 20 harmful, all
    # missed, and 180 not harmful judged right: its Se - Fp bound is near 0, a known failure.
    reference = lambda s, i: H if i < (10 if s == "sgd" else 20) else R  # noqa: E731
    judge = lambda s, i: reference(s, i) if s == "sgd" else R  # noqa: E731
    result = qualify(labeled(["sgd", "adamw"], 200, reference, judge), ["sgd", "adamw"],
                     permutations=100, posterior_draws=20_000)  # fmt: skip
    assert result["strata"]["adamw"]["agreement"] == 0.9
    assert result["status"] == "fail"
    assert any("discrimination" in r and "adamw" in r for r in result["reasons"])
