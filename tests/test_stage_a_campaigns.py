"""The campaigns of Stage A: written from the rules and the run records, loadable by the launcher
with every run's config checked against the protocol."""

import json
import runpy
import shutil
from pathlib import Path

import pytest
from test_stage_a import FULL, LEVELS, SEEDS, SHORT, THETA0, UP_TO_7, search, write_run

from zero_field_refusal.analysis.stage_a import BATCHES
from zero_field_refusal.launch import load_campaign

SCRIPT = Path(__file__).parents[1] / "scripts" / "stage_a_campaigns.py"


def write(monkeypatch, tmp_path: Path, *args: str) -> list:
    out = tmp_path / f"{args[0]}-{len(list(tmp_path.glob('*.yaml')))}.yaml"
    monkeypatch.setattr("sys.argv", [str(SCRIPT), *args, "--out", str(out)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return load_campaign(out)[1]


def labels(runs) -> list[str]:
    return [run.label for run in runs]


def test_the_fixed_campaigns(monkeypatch, tmp_path) -> None:
    assert labels(write(monkeypatch, tmp_path, "setup")) == ["theta0/s2000"]
    screen = write(monkeypatch, tmp_path, "screen")
    assert len(screen) == 20 and all(run.seed == 2000 for run in screen)
    grids = write(monkeypatch, tmp_path, "grids", "--seed", "2001")
    assert len(grids) == 8 + 24 and all(run.seed == 2001 for run in grids)
    assert "frozen_adam_25-b5-5e-06/s2001" in labels(grids)


def theta0_run(tmp_path: Path) -> Path:
    run = tmp_path / "theta0"
    run.mkdir(exist_ok=True)
    (run / "summary.json").write_text(json.dumps(THETA0))
    return run


def test_sgd_is_written_by_the_search_now(monkeypatch, tmp_path) -> None:
    search(tmp_path / "runs")
    with pytest.raises(SystemExit):
        write(monkeypatch, tmp_path, "sgd", "--runs", str(tmp_path / "runs"), "--seed", "2002")


@pytest.mark.parametrize(
    "damage",
    [
        lambda r: shutil.rmtree(next(r.rglob("screen-20-0.01"))),  # incomplete
        lambda r: (
            shutil.rmtree(next(r.rglob("screen-5-0.004"))),
            write_run(
                r / "screen",
                "screen-5-0.004",
                arm="soft_control",
                batch=5,
                lr=0.004,
                seed=2000,
                final_k=0.01,
                evaluate_at=(100,),
                role="screen",
                model="Qwen/Qwen2.5-0.5B-Instruct@abc",
            ),
        ),
    ],
)
def test_the_search_needs_a_whole_conforming_screen(monkeypatch, tmp_path, damage) -> None:
    search(tmp_path / "runs")
    damage(tmp_path / "runs")
    with pytest.raises(SystemExit):
        write(monkeypatch, tmp_path, "search", "--runs", str(tmp_path / "runs" / "screen"),
              "--seed", "2002")  # fmt: skip


def short_runs(root: Path, rates: dict[float, float], seeds=SEEDS) -> None:
    for arm in ("adamw_25", "frozen_adam_25"):
        for lr, k in rates.items():
            for seed in seeds:
                write_run(
                    root, f"{arm}-5-{lr}-{seed}", arm=arm, batch=5, lr=lr, seed=seed, final_k=k
                )


def test_extensions_are_decided_from_every_seed_and_may_halve_twice(monkeypatch, tmp_path):
    root = tmp_path / "runs"
    short_runs(root, SHORT | {0.5e-5: 5.5}, seeds=SEEDS[:2])  # seed 2002 not done yet
    with pytest.raises(SystemExit):
        write(monkeypatch, tmp_path, "extension", "--runs", str(root), "--seed", "2000")
    short_runs(root, SHORT | {0.5e-5: 5.5}, seeds=SEEDS[2:])
    runs = write(monkeypatch, tmp_path, "extension", "--runs", str(root), "--seed", "2000")
    assert labels(runs) == ["adamw_25-b5-2.5e-06/s2000", "frozen_adam_25-b5-2.5e-06/s2000"]
    short_runs(root, {0.25e-5: 5.2})  # still beyond 5 for every seed: the second halving
    runs = write(monkeypatch, tmp_path, "extension", "--runs", str(root), "--seed", "2001")
    assert labels(runs) == ["adamw_25-b5-1.25e-06/s2001", "frozen_adam_25-b5-1.25e-06/s2001"]
    short_runs(root, {0.125e-5: 5.1})
    with pytest.raises(SystemExit):  # two halvings at most: nothing more is called for
        write(monkeypatch, tmp_path, "extension", "--runs", str(root), "--seed", "2001")


def test_controls_follow_the_search_and_take_the_longest_run(monkeypatch, tmp_path) -> None:
    root = tmp_path / "runs"
    search(root, sgd={b: {0.004: FULL if b == 5 else UP_TO_7} for b in BATCHES})
    shutil.rmtree(root / "controls")
    late = {**FULL, 8.0: 600}  # seed 2001 needs 600 steps where seed 2000 needed 400
    slow = next(root.rglob("sgd_long-5-0.004-2001"))
    shutil.rmtree(slow)
    write_run(slow.parent, slow.name, arm="sgd_long", batch=5, lr=0.004, seed=2001, reach=late,
              monitor_version=2)  # fmt: skip
    args = ("--runs", str(root), "--theta0", str(theta0_run(tmp_path)))
    runs = write(monkeypatch, tmp_path, "controls", *args)
    assert sorted(labels(runs)) == ["control-b20-0.004/s2000", "control-b5-0.004/s2000"]
    control = next(run for run in runs if run.label == "control-b5-0.004/s2000")
    assert "train.steps=600" in control.overrides
    assert "schedule.evaluate_at=[100,150,200,300,400,600]" in control.overrides
    shutil.rmtree(next(root.rglob("sgd_long-5-0.004-2002")))  # B = 5 waits for that seed
    runs = write(monkeypatch, tmp_path, "controls", *args)
    assert labels(runs) == ["control-b20-0.004/s2000"]
    with pytest.raises(SystemExit):  # version-2 runs need θ₀ for the rule's utility
        write(monkeypatch, tmp_path, "controls", "--runs", str(root))


def test_campaigns_go_to_a_directory_made_for_them(monkeypatch, tmp_path) -> None:
    out = tmp_path / "runs" / "campaigns" / "screen.yaml"
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "screen", "--out", str(out)])
    runpy.run_path(str(SCRIPT), run_name="__main__")
    assert out.is_file()


def test_controls_need_the_screen_and_discovery_on_seed_2000s_one_stream(monkeypatch, tmp_path):
    # Review r14: each set reads one stream, but the screen another one than the discovery.
    search(tmp_path / "runs")
    shutil.rmtree(tmp_path / "runs" / "controls")
    for run_dir in sorted((tmp_path / "runs" / "screen").iterdir()):
        _, batch, lr = run_dir.name.split("-", 2)
        shutil.rmtree(run_dir)
        write_run(
            run_dir.parent,
            run_dir.name,
            arm="soft_control",
            batch=int(batch),
            lr=float(lr),
            seed=2000,
            final_k=0.01 if float(lr) <= 0.004 else 0.2,
            evaluate_at=(100,),
            role="screen",
            stream_path="teacher/other/seed-2000",
        )
    with pytest.raises(SystemExit):
        write(monkeypatch, tmp_path, "controls", "--runs", str(tmp_path / "runs"), "--theta0",
              str(theta0_run(tmp_path)))  # fmt: skip


def test_the_search_starts_at_each_arms_largest_rate_per_batch(monkeypatch, tmp_path) -> None:
    search(tmp_path / "runs")  # a whole screen (candidates 0.004 and 0.003 at both batches)
    runs = write(monkeypatch, tmp_path, "search", "--runs", str(tmp_path / "runs" / "screen"),
                 "--seed", "2001")  # fmt: skip
    assert labels(runs) == [
        "adamw_long-b20-1e-05/s2001",
        "adamw_long-b5-1e-05/s2001",
        "sgd_long-b20-0.004/s2001",
        "sgd_long-b5-0.004/s2001",
    ]


def test_the_search_goes_down_where_a_rate_fails_its_rule(monkeypatch, tmp_path) -> None:
    root = tmp_path / "runs"
    search(root, sgd={b: {0.004: FULL} for b in BATCHES},
           adamw={5: {1e-5: UP_TO_7}, 20: {1e-5: FULL}})  # fmt: skip
    args = ("--runs", str(root), "--theta0", str(theta0_run(tmp_path)), "--seed", "2001")
    assert labels(write(monkeypatch, tmp_path, "search", *args)) == ["adamw_long-b5-5e-06/s2001"]
    with pytest.raises(SystemExit):  # θ₀ is needed to judge the version-2 runs
        write(monkeypatch, tmp_path, "search", "--runs", str(root), "--seed", "2001")


def test_too_few_diagnostic_states_call_for_the_next_rate(monkeypatch, tmp_path) -> None:
    root = tmp_path / "runs"
    quick = {5.0: 10, LEVELS[1]: 20, LEVELS[2]: 30, 7.0: 40, 8.0: 50}  # two states per run
    search(root, sgd={b: {0.004: quick, 0.003: quick} for b in BATCHES})
    args = ("--runs", str(root), "--theta0", str(theta0_run(tmp_path)), "--seed", "2000")
    runs = write(monkeypatch, tmp_path, "diagnostic", *args)
    assert labels(runs) == ["sgd_long-b5-0.003/s2000"]
    assert "stage_a.role=diagnostic" in runs[0].overrides
