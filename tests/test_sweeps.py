"""The choice of stage-1 sweep rates from seed-0 pilots, by K, divergence and collapse alone."""

import math

import pytest

from zero_field_refusal.analysis.sweeps import (
    Pilot,
    choose,
    read_logical_runs,
    target_points,
    unstable_reason,
)

THETA0 = {"factual": 0.90, "benchmark": 0.60}
SGD_GRID = [0.03, 0.1, 0.3, 1.0, 3.0]
FA_GRID = [1e-5, 2e-5, 5e-5, 1e-4]


def pilot(lr, k, status="completed", factual=0.88, benchmark=0.59):
    return Pilot(lr, status, k, factual, benchmark)


def test_targets_are_the_ends_then_the_evenly_spaced_interior_points() -> None:
    assert target_points(1.0, 27.0, 4) == pytest.approx([1.0, 27.0, 3.0, 9.0])
    assert target_points(4.0, 16.0, 3) == pytest.approx([4.0, 16.0, 8.0])


@pytest.mark.parametrize(
    ("p", "reason"),
    [(pilot(0.1, 5.0), None),
     (pilot(0.1, 5.0, factual=0.80), None),  # exactly 10 pp below: no collapse
     (pilot(0.1, 5.0, factual=0.79), "collapse: factual accuracy"),
     (pilot(0.1, 5.0, benchmark=0.54), "collapse: benchmark accuracy"),
     (pilot(0.1, None, status="diverged"), "diverged"),
     (pilot(0.1, None, status="missing"), "status 'missing'"),
     (pilot(0.1, 5.0, factual=math.nan), "utility missing or invalid"),
     (pilot(0.1, 5.0, benchmark=1.5), "utility missing or invalid"),
     (pilot(0.1, 0.0), "K 0.0"), (pilot(0.1, math.nan), "K nan")],
)  # fmt: skip
def test_a_candidate_is_a_completed_stable_pilot_with_a_positive_finite_k(p, reason) -> None:
    assert unstable_reason(p, THETA0) == reason


def test_sgd_takes_the_nearest_candidates_and_always_its_largest_stable_rate() -> None:
    # AdamW range [2, 30]: targets 2, 30, then 2·15^(1/3) ≈ 4.93 and ≈ 12.2.
    pilots = [pilot(0.03, 1.5), pilot(0.1, 3.5), pilot(0.3, 9.0), pilot(1.0, 25.0),
              pilot(3.0, None, status="diverged")]  # fmt: skip
    choice = choose("sgd", SGD_GRID, pilots, THETA0, 2.0, 30.0)
    assert choice.extend == []
    assert choice.chosen == [0.03, 0.1, 0.3, 1.0]  # 1.0 is also the largest stable rate


def test_the_largest_stable_sgd_rate_replaces_the_last_choice() -> None:
    pilots = [pilot(0.03, 2.1), pilot(0.1, 2.2), pilot(0.3, 2.3), pilot(1.0, 25.0),
              pilot(3.0, 26.0)]  # fmt: skip
    choice = choose("sgd", SGD_GRID, pilots, THETA0, 2.0, 30.0)
    # nearest to 2 → 0.03; to 30 → 3.0 (26); to ≈4.93 → 0.3; to ≈12.2 → 1.0 (25): 3.0 is in
    assert choice.chosen == [0.03, 0.3, 1.0, 3.0]
    pilots[-1] = pilot(3.0, 2.0)  # the largest stable rate, far below the others
    choice = choose("sgd", SGD_GRID, pilots, THETA0, 2.0, 30.0)
    # nearest to 2 → 3.0 (2.0) itself; the rule keeps it, so nothing is replaced
    assert 3.0 in choice.chosen and len(choice.chosen) == 4


def test_a_tie_goes_to_the_smaller_rate() -> None:
    pilots = [pilot(1e-5, 4.0), pilot(2e-5, 16.0), pilot(5e-5, 8.0 / 2), pilot(1e-4, 8.0 * 2)]
    choice = choose("adam_frozen", FA_GRID, pilots, THETA0, 4.0, 16.0)
    assert choice.chosen == [1e-5, 2e-5, 5e-5]  # midpoint 8: 4 and 16 tie with 4 and 16 again


def test_an_unreached_end_is_piloted_first_one_step_beyond_the_grid() -> None:
    pilots = [pilot(lr, k) for lr, k in zip(SGD_GRID, (0.2, 0.5, 1.0, 2.0, 4.0), strict=True)]
    choice = choose("sgd", SGD_GRID, pilots, THETA0, 1.0, 30.0)
    assert choice.extend == [9.0] and choice.chosen == []
    pilots.append(pilot(9.0, 8.0))
    assert choose("sgd", SGD_GRID, pilots, THETA0, 1.0, 30.0).extend == [27.0]
    pilots.append(pilot(27.0, None, status="diverged"))
    done = choose("sgd", SGD_GRID, pilots, THETA0, 1.0, 30.0)
    assert done.extend == []  # two extensions per end at most
    assert "K_A,max" in done.notes[0]
    assert done.chosen == [0.3, 1.0, 3.0, 9.0]


def test_frozen_adam_extends_downward_by_a_factor_two() -> None:
    pilots = [pilot(lr, k) for lr, k in zip(FA_GRID, (6.0, 10.0, 20.0, 40.0), strict=True)]
    choice = choose("adam_frozen", FA_GRID, pilots, THETA0, 1.0, 30.0)
    assert choice.extend == [5e-6]


def test_without_candidates_nothing_is_chosen() -> None:
    pilots = [pilot(lr, None, status="diverged") for lr in SGD_GRID]
    pilots += [pilot(lr, None, status="diverged") for lr in (0.01, 0.01 / 3, 9.0, 27.0)]
    choice = choose("sgd", SGD_GRID, pilots, THETA0, 1.0, 30.0)
    assert choice.extend == [] and choice.chosen == []
    assert "no stable candidate" in choice.notes


def test_a_largest_stable_rate_left_out_by_the_targets_replaces_the_last_choice() -> None:
    pilots = [pilot(0.03, 2.0), pilot(0.1, 30.0), pilot(0.3, 5.0), pilot(1.0, 12.0),
              pilot(3.0, 60.0)]  # fmt: skip
    choice = choose("sgd", SGD_GRID, pilots, THETA0, 2.0, 30.0)
    # targets 2, 30, 4.93, 12.2 pick 0.03, 0.1, 0.3, 1.0; the last (1.0) gives way to 3.0
    assert choice.chosen == [0.03, 0.1, 0.3, 3.0]


def write_run(root, name, overrides, status, summary, experiment="stage1_main_1.5b"):
    import json

    from omegaconf import OmegaConf

    from zero_field_refusal.protocol import load_run_config

    run_dir = root / "stage1_main_1.5b" / name
    run_dir.mkdir(parents=True)
    OmegaConf.save(
        load_run_config("experiments/stage1_main_1.5b", overrides), run_dir / "config.yaml"
    )
    (run_dir / "meta.json").write_text(json.dumps({"status": status}))
    (run_dir / "summary.json").write_text(json.dumps(summary))


def test_the_script_reads_k_and_utility_from_run_directories(tmp_path, capsys) -> None:
    import importlib.util
    import json
    from pathlib import Path

    write_run(tmp_path, "theta0", ["run_type=theta0", "seed=0"], "completed",
              {"K_harmful": 0.0, "factual_accuracy": 0.9, "benchmark_accuracy": 0.6})  # fmt: skip
    for lr, k in ((1e-5, 2.0), (5e-5, 30.0)):
        for seed in range(2):
            write_run(tmp_path, f"adamw-{lr}-{seed}", [f"optimizer.lr={lr:.1e}", f"seed={seed}"],
                      "completed", {"K_harmful": k, "factual_accuracy": 0.88,
                                    "benchmark_accuracy": 0.59})  # fmt: skip
    for lr, k in zip(SGD_GRID, (1.5, 3.5, 9.0, 25.0, None), strict=True):
        status = "completed" if k else "diverged"
        write_run(tmp_path, f"sgd-{lr}", ["optimizer=sgd", f"optimizer.lr={lr:.1e}", "seed=0"],
                  status, {"K_harmful": k, "factual_accuracy": 0.88,
                           "benchmark_accuracy": 0.59} if k else {})  # fmt: skip
    for lr, k in zip(FA_GRID, (2.0, 5.0, 12.0, 30.0), strict=True):
        utility = {"K_harmful": k, "factual_accuracy": 0.88, "benchmark_accuracy": 0.59}
        overrides = ["optimizer=adam_frozen", f"optimizer.lr={lr:.1e}", "seed=0"]
        write_run(tmp_path, f"fa-{lr}", overrides, "completed", utility)
    script = Path(__file__).resolve().parents[1] / "scripts" / "choose_rates.py"
    spec = importlib.util.spec_from_file_location("choose_rates", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main([str(tmp_path), "--model", "1.5b"])
    out = json.loads(capsys.readouterr().out)
    assert out["adamw_k_range"] == pytest.approx([2.0, 30.0])
    assert out["sweeps"]["sgd"]["chosen"] == [0.03, 0.1, 0.3, 1.0]
    assert out["sweeps"]["adam_frozen"]["chosen"] == [1e-5, 2e-5, 1e-4]
    assert out["sweeps"]["sgd"]["status"] == "chosen"
    x5 = out["sweeps"]["sgd_100x5"]  # required on 1.5B: without its pilots the choice waits
    assert (x5["status"], x5["pending"]) == ("wait", [0.01, 0.03, 0.1, 0.3, 1.0])


def test_pilots_without_a_known_outcome_make_the_choice_wait() -> None:
    pilots = [pilot(lr, k) for lr, k in zip(SGD_GRID, (0.2, 0.5, 1.0, 2.0, 4.0), strict=True)]
    pilots += [pilot(9.0, None, status="pending"), pilot(27.0, None, status="pending")]
    choice = choose("sgd", SGD_GRID, pilots, THETA0, 1.0, 30.0)
    assert (choice.status, choice.chosen, choice.pending) == ("wait", [], [9.0, 27.0])
    missing_base = [p for p in pilots if p.lr != 0.3 and p.status == "completed"]
    choice = choose("sgd", SGD_GRID, missing_base, THETA0, 1.0, 30.0)
    assert (choice.status, choice.pending) == ("wait", [0.3])


def test_theta0_utility_must_be_a_finite_accuracy() -> None:
    pilots = [pilot(lr, k) for lr, k in zip(FA_GRID, (2.0, 5.0, 12.0, 30.0), strict=True)]
    with pytest.raises(ValueError, match="factual accuracy nan"):
        choose("adam_frozen", FA_GRID, pilots, {"factual": math.nan, "benchmark": 0.6}, 2.0, 30.0)


def utility(k):
    return {"K_harmful": k, "factual_accuracy": 0.88, "benchmark_accuracy": 0.59}


def test_a_diverged_and_a_completed_attempt_of_one_pilot_contradict(tmp_path) -> None:
    overrides = ["optimizer=sgd", "optimizer.lr=3.0e-1", "seed=0"]
    write_run(tmp_path, "sgd-a1", overrides, "diverged", {})
    write_run(tmp_path, "sgd-a2", overrides, "completed", utility(9.0))
    with pytest.raises(ValueError, match="diverged and a completed attempt"):
        read_logical_runs(tmp_path, "stage1_main_1.5b")


def test_a_duplicated_completed_run_is_refused(tmp_path) -> None:
    overrides = ["optimizer.lr=1.0e-5", "seed=0"]
    write_run(tmp_path, "adamw-a", overrides, "completed", utility(1.0))
    write_run(tmp_path, "adamw-b", overrides, "completed", utility(1.0))
    with pytest.raises(ValueError, match="2 completed attempts"):
        read_logical_runs(tmp_path, "stage1_main_1.5b")


def test_failed_attempts_are_pending_until_three_then_missing(tmp_path) -> None:
    overrides = ["optimizer=sgd", "optimizer.lr=3.0e-1", "seed=0"]
    for attempt in range(1, 4):
        write_run(tmp_path, f"sgd-a{attempt}", overrides, "failed", {})
        (run,) = read_logical_runs(tmp_path, "stage1_main_1.5b")
        assert run["status"] == ("missing" if attempt == 3 else "pending")
    write_run(tmp_path, "sgd-a4", overrides, "running", {})
    assert read_logical_runs(tmp_path, "stage1_main_1.5b")[0]["status"] == "pending"


@pytest.mark.parametrize(
    ("experiment", "config", "overrides"),
    [("stage1_sgd100x5_1.5b", "experiments/stage1_main_1.5b",
      ["optimizer=sgd", "optimizer.lr=1.0e-1", "seed=0"]),  # 25 x 20 in the 100 x 5 directory
     ("stage1_main_1.5b", "experiments/stage1_main_0.5b",
      ["optimizer=sgd", "optimizer.lr=1.0e-1", "seed=0"])],  # 0.5B in the 1.5B directory
)  # fmt: skip
def test_a_run_outside_its_experiments_condition_is_refused(tmp_path, experiment, config,
                                                            overrides) -> None:  # fmt: skip
    import json

    from omegaconf import OmegaConf

    from zero_field_refusal.protocol import load_run_config

    run_dir = tmp_path / experiment / "stray"
    run_dir.mkdir(parents=True)
    OmegaConf.save(load_run_config(config, overrides), run_dir / "config.yaml")
    (run_dir / "meta.json").write_text(json.dumps({"status": "completed"}))
    (run_dir / "summary.json").write_text(json.dumps(utility(5.0)))
    with pytest.raises(ValueError, match="depart from its committed config"):
        read_logical_runs(tmp_path, experiment)
