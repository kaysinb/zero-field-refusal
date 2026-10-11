"""The Gate 1 report (analysis.gate1) on synthetic stage-1 run directories built in tmp_path.

``build`` writes run directories of the stage-1 experiments: configs composed from
configs/experiments/stage1_*.yaml, meta.json, summary.json, benign placeholder responses and
WildGuard label records. Each test states its cells, and the numbers it checks are computed by
hand from them. θ₀ refuses all 20 judged prompts, complies with none, and scores 0.9 on the
factual questions and 0.6 on the benchmark unless a test says otherwise.
"""

import functools
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from scipy.special import expit

from zero_field_refusal.analysis.curves import fit_logistic
from zero_field_refusal.analysis.evidence import arm_of, label_item_id
from zero_field_refusal.analysis.gate1 import (
    CELL_COLUMNS,
    TOKEN_COUNT_ARM,
    first_crossing,
    gate1_report,
    pool_cells,
    refusal_fit,
    t_interval,
)
from zero_field_refusal.config import config_hash, load_config
from zero_field_refusal.judge import judge_identity
from zero_field_refusal.protocol import load_run_config

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "gate1_report.py"
JUDGE = "wildguard"
IDENTITY = judge_identity(load_config(f"judge/{JUDGE}"))
JUDGED = 20  # judged responses per run
STAMP = "2026-10-11T00:00:00+00:00"
MAIN, TOKEN, X5 = "stage1_main_1.5b", "stage1_token_count_1.5b", "stage1_sgd100x5_1.5b"
MAIN_05 = "stage1_main_0.5b"
DEFAULT_OPTIMIZER = {X5: "sgd"}  # the others compose AdamW
T4 = 2.7764451051977987  # the 0.975 quantile of Student's t with 4 degrees of freedom
OUTPUTS = ["cells.csv", "gate1.json", "manifest.json"]


@dataclass(frozen=True)
class Theta0:
    refusal: int = JUDGED
    harmful: int = 0
    factual: float = 0.9
    benchmark: float = 0.6


@dataclass(frozen=True)
class Cell:
    """One cell's runs, seed by seed: K, refusal and harmful counts (None: every response that is
    not a refusal is harmful), factual and benchmark accuracies."""

    experiment: str
    optimizer: str
    lr: float
    K: tuple[float, ...]
    refusal: tuple[int, ...]
    harmful: tuple[int, ...] | None = None
    factual: tuple[float, ...] | float = 0.88
    benchmark: float = 0.59
    seeds: tuple[int, ...] | None = None  # None: 0, 1, ...
    diverged: tuple[int, ...] = ()  # seeds of further runs that stopped on a non-finite loss


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@functools.cache
def run_config(experiment: str, optimizer: str, lr: float | None, seed: int, run_type: str):
    overrides = []
    if optimizer != DEFAULT_OPTIMIZER.get(experiment, "adamw"):
        overrides.append(f"optimizer={optimizer}")
    if lr is not None:
        overrides.append(f"optimizer.lr={lr:.6e}")
    overrides += [f"seed={seed}", f"run_type={run_type}"]
    return load_run_config(f"experiments/{experiment}", overrides)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


def write_run(
    root: Path, experiment: str, run_id: str, cfg, *, condition: str, K: float | None,
    refusal: int, harmful: int, factual: float, benchmark: float, status: str = "completed",
    unlabeled: int = 0,
) -> Path:  # fmt: skip
    """One run directory; ``unlabeled`` responses at the end get no label."""
    assert refusal + harmful <= JUDGED
    run_dir = root / experiment / run_id
    run_dir.mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    streams = {"kl_harmful": {"path": "teacher/kl_harmful", "count": 400,
                              "sha256": sha("kl_harmful")}}  # fmt: skip
    if condition == "trained":
        streams["training"] = {"path": f"teacher/training-seed{cfg.seed}", "count": 500,
                               "sha256": sha(f"training stream of seed {cfg.seed}")}  # fmt: skip
    write_json(run_dir / "meta.json", {
        "run_id": run_id, "experiment": experiment, "config_hash": config_hash(cfg),
        "status": status, "started_at": STAMP, "finished_at": STAMP,
        "inputs": {"prompt_manifest_sha256": sha("prompt manifest"), "streams": streams},
    })  # fmt: skip
    if status != "completed":
        return run_dir
    summary = {"factual_accuracy": factual, "benchmark_accuracy": benchmark}
    write_json(run_dir / "summary.json", summary | ({} if K is None else {"K_harmful": K}))
    labels = (["refusal"] * refusal + ["harmful_compliance"] * harmful
              + ["safe_non_refusal"] * (JUDGED - refusal - harmful))  # fmt: skip
    write_jsonl(run_dir / "responses" / f"{cfg.eval.harmful.prompt_set}.jsonl", [
        {"condition": condition, "item_id": f"judged:{i}", "response_ids": [11, 12, 13],
         "text": f"Placeholder response {i}.", "ended": True, "keyword_refusal": False}
        for i in range(JUDGED)
    ])  # fmt: skip
    write_jsonl(run_dir / "judge" / f"{JUDGE}.jsonl", [
        {"item_id": label_item_id(condition, f"judged:{i}"), "label": label, **IDENTITY,
         "created_at": STAMP}
        for i, label in enumerate(labels[: JUDGED - unlabeled])
    ])  # fmt: skip
    return run_dir


def write_theta0(root: Path, experiment: str, theta0: Theta0, run_id: str = "theta0_s0") -> None:
    write_run(root, experiment, run_id, run_config(experiment, "adamw", None, 0, "theta0"),
              condition="theta0", K=None, refusal=theta0.refusal, harmful=theta0.harmful,
              factual=theta0.factual, benchmark=theta0.benchmark)  # fmt: skip


def run_id(cell: Cell, seed: int) -> str:
    return f"{cell.optimizer}_lr{cell.lr:g}_s{seed}"


THETA0 = Theta0()


def build(root: Path, cells, *, theta0: Theta0 | None = THETA0, main: str = MAIN) -> Path:
    """Run directories of ``cells`` and of θ₀ under ``root``."""
    if theta0 is not None:
        write_theta0(root, main, theta0)
    for cell in cells:
        seeds = cell.seeds or tuple(range(len(cell.K)))
        harmful = cell.harmful or tuple(JUDGED - r for r in cell.refusal)
        factual = cell.factual if isinstance(cell.factual, tuple) else (cell.factual,) * len(seeds)
        for i, seed in enumerate(seeds):
            cfg = run_config(cell.experiment, cell.optimizer, cell.lr, seed, "train")
            write_run(root, cell.experiment, run_id(cell, seed), cfg, condition="trained",
                      K=cell.K[i], refusal=cell.refusal[i], harmful=harmful[i],
                      factual=factual[i], benchmark=cell.benchmark)  # fmt: skip
        for seed in cell.diverged:
            cfg = run_config(cell.experiment, cell.optimizer, cell.lr, seed, "train")
            write_run(root, cell.experiment, run_id(cell, seed), cfg, condition="trained",
                      K=None, refusal=0, harmful=0, factual=0.0, benchmark=0.0,
                      status="diverged")  # fmt: skip
    return root


def adamw(lr: float, K, refusal, **kw) -> Cell:
    return Cell(MAIN, "adamw", lr, K, refusal, **kw)


def sgd(lr: float, K: float, refusal: int, experiment: str = MAIN, **kw) -> Cell:
    return Cell(experiment, "sgd", lr, (K,), (refusal,), **kw)  # a seed-0 pilot


# The AdamW cells of 1.5B: refusals fall with K, and the 10 pp drop is crossed between the
# 5e-6 and 2e-5 cells.
EFFECT_REFUSAL = (10, 9, 11, 8, 12)
EFFECT_HARMFUL = (8, 9, 7, 10, 6)
EFFECT_FACTUAL = (0.88, 0.86, 0.9, 0.87, 0.89)
ADAMW = {
    5e-6: adamw(5e-6, (1.8, 2.2, 2.0, 1.9, 2.1), (20, 19, 19, 20, 19)),
    1e-5: adamw(1e-5, (4.5, 5.5, 5.0, 4.8, 5.2), (18, 17, 18, 17, 18)),
    2e-5: adamw(2e-5, (11.0, 13.0, 12.0, 12.5, 11.5), (13, 14, 12, 14, 13)),
    5e-5: adamw(5e-5, (28.0, 32.0, 30.0, 31.0, 29.0), EFFECT_REFUSAL, harmful=EFFECT_HARMFUL,
                factual=EFFECT_FACTUAL),
}  # fmt: skip
SGD = {0.1: sgd(0.1, 1.5, 20), 0.3: sgd(0.3, 8.0, 17), 1.0: sgd(1.0, 40.0, 9)}
TOKEN_COUNT = Cell(TOKEN, "adamw", 5e-5, (29.0, 31.0, 30.0, 32.0, 28.0), (9, 10, 8, 9, 10))
HUNDRED_BY_5 = {0.03: sgd(0.03, 1.0, 20, X5), 0.3: sgd(0.3, 20.0, 12, X5)}


def cells(adamw_cells=None, sgd_cells=None, token=TOKEN_COUNT, hundred=None) -> list[Cell]:
    """The standard 1.5B cells with the given ones in their place (None: the standard ones)."""
    hundred = HUNDRED_BY_5 if hundred is None else hundred
    chosen = [*(adamw_cells or ADAMW).values(), *(sgd_cells or SGD).values(), *hundred.values()]
    return chosen + ([token] if token is not None else [])


def report(root: Path, out: Path, **kwargs) -> dict:
    gate1_report(root, JUDGE, IDENTITY, out, **kwargs)
    return json.loads((out / "gate1.json").read_text())


def geometric_mean(values) -> float:
    return math.exp(float(np.mean(np.log(values))))


def interval(values) -> tuple[float, float, float]:
    """Mean and the 95 % t-interval of five values, by the textbook formula."""
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    half = T4 * sd / math.sqrt(len(values))
    return mean, mean - half, mean + half


@pytest.fixture(scope="module")
def standard(tmp_path_factory) -> tuple[Path, dict]:
    """The standard 1.5B report, with a failed run and a run with a missing label beside it."""
    root = build(tmp_path_factory.mktemp("standard") / "runs", cells())
    cfg = run_config(MAIN, "adamw", 5e-5, 5, "train")
    write_run(root, MAIN, "adamw_lr5e-05_s5-failed", cfg, condition="trained", K=None, refusal=0,
              harmful=0, factual=0.0, benchmark=0.0, status="failed")  # fmt: skip
    write_run(root, MAIN, "adamw_lr5e-05_s5", cfg, condition="trained", K=30.0, refusal=0,
              harmful=20, factual=0.88, benchmark=0.59, unlabeled=1)  # fmt: skip
    return root, report(root, root.parent / "report")


# --- 1. effect ----------------------------------------------------------------------------------


def test_the_effect_matches_hand_computed_means_and_t_intervals(standard) -> None:
    effect = standard[1]["effect"]
    assert effect["runs"] == [f"adamw_lr5e-05_s{seed}" for seed in range(5)]
    expected = {
        "refusal_drop": [1 - r / JUDGED for r in EFFECT_REFUSAL],
        "harmful_rise": [h / JUDGED for h in EFFECT_HARMFUL],
        "factual_drop": [0.9 - a for a in EFFECT_FACTUAL],
    }
    for quantity, values in expected.items():
        got = effect[quantity]
        assert [s["value"] for s in got["per_seed"]] == values
        assert [s["seed"] for s in got["per_seed"]] == [0, 1, 2, 3, 4]
        mean, low, high = interval(values)
        assert (got["n"], got["df"], got["level"]) == (5, 4, 0.95)
        assert got["t_quantile"] == pytest.approx(T4, rel=1e-12)
        assert got["mean"] == pytest.approx(mean, rel=1e-12)
        assert (got["ci_low"], got["ci_high"]) == pytest.approx((low, high), rel=1e-12)
    # 0.5 ± 0.0982: mean ≥ 0.25 and lower bound ≥ 0.15; harmful rise 0.4 ± 0.0982 > 0; factual 0.02.
    assert effect["refusal_drop"]["mean"] == pytest.approx(0.5)
    assert effect["refusal_drop"]["ci_low"] == pytest.approx(0.4018378, abs=1e-7)
    assert effect["parts"] == {"refusal_drop": True, "harmful_rise": True, "factual_drop": True}
    assert effect["passes"] is True
    assert effect["status"] == "pass" and effect["missing_seeds"] == []
    assert standard[1]["summary"]["effect_status"] == "pass"


def test_the_effect_fails_part_by_part(tmp_path) -> None:
    refusal = (2, 18, 10, 16, 14)  # drops 0.9 0.1 0.5 0.2 0.3: mean 0.4, lower bound 0.007
    effect_cell = adamw(5e-5, ADAMW[5e-5].K, refusal, harmful=(16, 2, 8, 4, 6), factual=0.75)
    root = build(tmp_path / "runs", cells({**ADAMW, 5e-5: effect_cell}))
    effect = report(root, tmp_path / "out")["effect"]
    mean, low, _ = interval([1 - r / JUDGED for r in refusal])
    assert effect["refusal_drop"]["mean"] == pytest.approx(mean) == pytest.approx(0.4)
    assert effect["refusal_drop"]["ci_low"] == pytest.approx(low)
    assert 0 < low < 0.15
    assert effect["factual_drop"]["mean"] == pytest.approx(0.15)
    assert effect["parts"] == {"refusal_drop": False, "harmful_rise": True, "factual_drop": False}
    assert (effect["status"], effect["passes"]) == ("fail", False)


def test_the_effect_needs_all_five_seeds(tmp_path) -> None:
    cell = ADAMW[5e-5]
    four = replace(cell, K=cell.K[:4], refusal=cell.refusal[:4], harmful=cell.harmful[:4],
                   factual=cell.factual[:4], diverged=(4,))  # fmt: skip
    root = build(tmp_path / "runs", cells({**ADAMW, 5e-5: four}))
    effect = report(root, tmp_path / "out")["effect"]
    assert effect["parts"] == {"refusal_drop": True, "harmful_rise": True, "factual_drop": True}
    assert (effect["seeds"], effect["missing_seeds"]) == ([0, 1, 2, 3], [4])
    assert effect["refusal_drop"]["n"] == 4  # the numbers are still reported
    assert (effect["status"], effect["passes"]) == ("incomplete", False)


def test_one_seed_has_no_interval_and_the_criterion_is_null() -> None:
    single = t_interval({3: 0.4})
    assert (single["n"], single["mean"]) == (1, 0.4)
    assert (single["sd"], single["ci_low"], single["ci_high"]) == (None, None, None)
    assert t_interval({})["mean"] is None


def test_runs_that_fail_or_lack_a_label_do_not_count(standard) -> None:
    excluded = standard[1]["experiments"]["main"]["excluded"]
    assert excluded["adamw_lr5e-05_s5-failed"]["reason"] == "not_completed"
    assert excluded["adamw_lr5e-05_s5"]["reason"] == "incomplete_labels"
    assert excluded["theta0_s0"]["reason"] == "not_training"  # θ₀ is read on its own
    assert standard[1]["effect"]["refusal_drop"]["n"] == 5
    assert standard[1]["theta0"] == {
        "experiment": MAIN, "run_id": "theta0_s0", "judged": 20, "refusal": 20, "harmful": 0,
        "refusal_rate": 1.0, "harmful_rate": 0.0, "factual_accuracy": 0.9,
        "benchmark_accuracy": 0.6,
    }  # fmt: skip
    assert standard[1]["experiments"]["main"]["gates"] == [1]


# --- 3. collapse and the cells ------------------------------------------------------------------


def test_cells_hold_means_geometric_means_and_collapse(tmp_path) -> None:
    sgd_cells = {**SGD, 1.0: sgd(1.0, 40.0, 9, factual=0.7),  # factual drop 0.2
                 0.3: sgd(0.3, 8.0, 17, benchmark=0.54)}  # benchmark drop 0.06  # fmt: skip
    root = build(tmp_path / "runs", cells(sgd_cells=sgd_cells))
    out = tmp_path / "out"
    data = report(root, out)
    table = pd.read_csv(out / "cells.csv", dtype={"seeds": str}, float_precision="round_trip")
    assert list(table.columns) == list(CELL_COLUMNS)
    assert table.to_dict("records") == data["cells"]  # the same numbers, unrounded
    table = table.set_index(["experiment", "arm", "lr"])

    effect = table.loc[(MAIN, "adamw", 5e-5)]
    assert (effect["runs"], effect["seeds"]) == (5, "0 1 2 3 4")
    assert effect["mean_K"] == pytest.approx(30.0)
    assert effect["geometric_mean_K"] == pytest.approx(geometric_mean(ADAMW[5e-5].K), rel=1e-12)
    assert effect["mean_refusal_drop"] == pytest.approx(0.5)
    assert effect["mean_harmful_rise"] == pytest.approx(0.4)
    assert effect["mean_factual_drop"] == pytest.approx(0.02)
    assert effect["mean_benchmark_drop"] == pytest.approx(0.01)
    assert not effect["collapsed"]
    assert table.loc[(MAIN, "sgd", 1.0), "collapsed"]
    assert table.loc[(MAIN, "sgd", 0.3), "collapsed"]
    assert not table.loc[(MAIN, "sgd", 0.1), "collapsed"]
    token = table.loc[(TOKEN, TOKEN_COUNT_ARM, 5e-5)]
    assert token["mean_refusal_drop"] == pytest.approx(0.54)
    assert table.loc[(X5, "sgd", 0.3), "geometric_mean_K"] == pytest.approx(20.0)
    assert data["collapsed_cells"] == [{"experiment": MAIN, "arm": "sgd", "lr": 0.3},
                                       {"experiment": MAIN, "arm": "sgd", "lr": 1.0}]  # fmt: skip


def test_a_drop_of_exactly_the_threshold_is_not_collapse(tmp_path) -> None:
    # In floats 0.8 - 0.7 and 0.65 - 0.6 exceed 0.1 and 0.05; the drops are exactly 10 and 5 pp.
    assert 0.8 - 0.7 > 0.1 and 0.65 - 0.6 > 0.05
    theta0 = Theta0(factual=0.8, benchmark=0.65)
    chosen = [adamw(5e-5, ADAMW[5e-5].K, EFFECT_REFUSAL, factual=0.7, benchmark=0.65),
              sgd(0.1, 1.5, 20, factual=0.8, benchmark=0.6),
              sgd(0.3, 8.0, 17, factual=0.69, benchmark=0.65)]  # fmt: skip
    root = build(tmp_path / "runs", chosen, theta0=theta0)
    table = pd.DataFrame(report(root, tmp_path / "out")["cells"])
    collapsed = dict(zip(table["arm"] + " " + table["lr"].astype(str), table["collapsed"],
                         strict=True))  # fmt: skip
    assert collapsed == {"adamw 5e-05": False, "sgd 0.1": False, "sgd 0.3": True}


# --- 4. K_lo and K_hi ---------------------------------------------------------------------------


def _adamw_runs(adamw_cells) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.concatenate([cell.K for cell in adamw_cells.values()])
    refusal = np.concatenate([cell.refusal for cell in adamw_cells.values()])
    return np.log(k), refusal, np.full(len(k), JUDGED)


def test_k_lo_inside_the_data_from_the_logistic_fit(standard) -> None:
    k_lo = standard[1]["k_lo"]
    assert (k_lo["case"], k_lo["method"], k_lo["crossing"]) == ("inside", "logistic", "inside")
    fit = k_lo["fit"]
    assert (fit["failed"], fit["converged"], fit["separated"]) == (False, True, False)
    assert fit["runs"] == 20
    # The fit agrees with the project's own IRLS fit (curves.fit_logistic, a negligible ridge).
    reference = fit_logistic(*_adamw_runs(ADAMW))
    assert fit["intercept"] == pytest.approx(reference.intercept[0], rel=1e-6)
    assert fit["slope"] == pytest.approx(reference.slope[0], rel=1e-6)
    assert fit["slope"] < 0
    # K_lo solves d(K) = r_θ₀ - r̂(K) = 0.10 on the fit, between the smallest and largest cell.
    assert k_lo["k_min"] == pytest.approx(geometric_mean(ADAMW[5e-6].K), rel=1e-12)
    assert k_lo["k_max"] == pytest.approx(geometric_mean(ADAMW[5e-5].K), rel=1e-12)
    assert k_lo["k_hi"] == k_lo["k_max"]  # no AdamW cell collapsed
    drop = 1.0 - expit(fit["intercept"] + fit["slope"] * math.log(k_lo["k_lo"]))
    assert drop == pytest.approx(0.10, abs=1e-12)
    assert k_lo["k_min"] < k_lo["k_lo"] < geometric_mean(ADAMW[2e-5].K)
    assert k_lo["drop_at_k_min"] < 0.10 <= k_lo["drop_at_k_max"]
    assert 3.0 < k_lo["k_lo"] < 5.0  # what the coverage cases below rely on


def test_k_lo_censored_below(tmp_path) -> None:
    lower = {**ADAMW, 5e-6: adamw(5e-6, ADAMW[5e-6].K, (16, 17, 16, 15, 16)),
             1e-5: adamw(1e-5, ADAMW[1e-5].K, (14, 15, 14, 13, 14)),
             2e-5: adamw(2e-5, ADAMW[2e-5].K, (12, 11, 12, 12, 11))}  # fmt: skip
    root = build(tmp_path / "runs", cells(lower))
    data = report(root, tmp_path / "out")
    k_lo, coverage = data["k_lo"], data["coverage"]
    assert (k_lo["case"], k_lo["method"], k_lo["k_lo"]) == ("censored_below", "logistic", None)
    assert k_lo["drop_at_k_min"] >= 0.10
    # Coverage on [K_min, √(K_min·K_hi)], reported as censored.
    k_min, k_hi = k_lo["k_min"], k_lo["k_hi"]
    assert coverage["required"] == {"lower": k_min, "upper": math.sqrt(k_min * k_hi)}
    assert coverage["censored"] is True
    assert (coverage["covered_main"], coverage["route"]) == (True, "equal-KL")
    assert coverage["main"]["at_or_below_lower"] == [0.1]
    assert coverage["main"]["at_or_above_upper"] == [0.3, 1.0]


def test_k_lo_with_no_crossing_cannot_be_assessed(tmp_path) -> None:
    flat = {5e-6: adamw(5e-6, ADAMW[5e-6].K, (20, 20, 19, 20, 20)),
            1e-5: adamw(1e-5, ADAMW[1e-5].K, (20, 19, 20, 19, 20)),
            2e-5: adamw(2e-5, ADAMW[2e-5].K, (19, 19, 20, 19, 18)),
            5e-5: adamw(5e-5, ADAMW[5e-5].K, (19, 18, 19, 18, 19))}  # fmt: skip
    root = build(tmp_path / "runs", cells(flat))
    data = report(root, tmp_path / "out")
    k_lo, coverage = data["k_lo"], data["coverage"]
    assert (k_lo["case"], k_lo["method"], k_lo["k_lo"]) == ("no_crossing", "logistic", None)
    assert k_lo["fit"]["failed"] is False
    assert k_lo["drop_at_k_max"] < 0.10
    assert coverage["assessable"] is False
    assert (coverage["covered_main"], coverage["covered_100x5"], coverage["required"]) == (
        None, None, None)  # fmt: skip
    # Coverage that cannot be assessed never lets the study continue by itself: the effect
    # criterion fails here (mean drop 0.07), no calibration result was given, and with the
    # token-count drop at 0.54 the effect disappears with the constant denominator.
    assert coverage["other_criteria"] == {"effect_criterion": False, "judge_calibration": None,
                                          "effect_does_not_disappear": False}  # fmt: skip
    assert data["effect_disappears"]["disappears"] is True
    assert coverage["blocking"] == ["effect_criterion", "judge_calibration",
                                    "effect_does_not_disappear"]  # fmt: skip
    assert coverage["route"] == "stop"
    assert data["summary"]["k_lo_case"] == "no_crossing"


def test_a_separated_fit_fails_and_k_lo_is_interpolated_between_cells(tmp_path) -> None:
    # Every run below K = 6 refuses all prompts, every run above none: the fit is separated.
    step = {5e-6: adamw(5e-6, ADAMW[5e-6].K, (20,) * 5),
            1e-5: adamw(1e-5, (5.0,) * 5, (20,) * 5),
            2e-5: adamw(2e-5, (12.0,) * 5, (0,) * 5),
            5e-5: adamw(5e-5, ADAMW[5e-5].K, (0,) * 5)}  # fmt: skip
    root = build(tmp_path / "runs", cells(step))
    k_lo = report(root, tmp_path / "out")["k_lo"]
    assert k_lo["fit"]["failed"] is True and k_lo["fit"]["separated"] is True
    assert "separated" in k_lo["fit"]["reason"]
    assert (k_lo["case"], k_lo["method"]) == ("inside", "interpolation")
    # The first upward crossing: from the cell at K = 5 (drop 0) to the cell at K = 12 (drop 1).
    interpolation = k_lo["interpolation"]
    assert interpolation["pair"] == [1, 2]
    assert [cell["lrs"] for cell in interpolation["cells"]] == [[5e-6], [1e-5], [2e-5], [5e-5]]
    assert [cell["mean_refusal_drop"] for cell in interpolation["cells"]] == [0, 0, 1, 1]
    assert interpolation["cells"][1]["geometric_mean_K"] == pytest.approx(5.0, rel=1e-12)
    expected = math.exp(math.log(5.0) + 0.10 * (math.log(12.0) - math.log(5.0)))
    assert k_lo["k_lo"] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("ks", "drops", "crossing", "pair"),
    [((2, 10), (0.2, 0.3), "censored_below", None),
     ((2, 10), (0.1, 0.3), "censored_below", None),  # exactly 10 pp reaches it
     ((2, 10), (0.0, 0.05), "no_crossing", None),
     ((2, 10, 30), (0.05, 0.0, 0.02), "no_crossing", None),
     ((2, 10, 30), (0.0, 0.05, 0.5), "inside", 1),
     ((2, 10, 30), (0.0, 0.5, 0.05), "inside", 0),  # the drop falls back later: still the first
     ((2, 10, 30, 40), (0.0, 0.2, 0.05, 0.5), "inside", 0)],
)  # fmt: skip
def test_a_failed_fit_takes_the_first_upward_crossing_of_the_cell_means(
    ks, drops, crossing, pair
) -> None:
    cells_ = [{"geometric_mean_K": float(k), "mean_refusal_drop": d}
              for k, d in zip(ks, drops, strict=True)]  # fmt: skip
    got, k_lo, i = first_crossing(cells_)
    assert (got, i) == (crossing, pair)
    if pair is None:
        assert k_lo is None
    else:
        share = (0.10 - drops[i]) / (drops[i + 1] - drops[i])
        expected = math.exp(math.log(ks[i]) + share * (math.log(ks[i + 1]) - math.log(ks[i])))
        assert k_lo == pytest.approx(expected, rel=1e-12)
        assert ks[i] < k_lo < ks[i + 1]


def test_cells_with_the_same_cell_mean_k_are_pooled_over_their_runs() -> None:
    cells_ = pd.DataFrame({
        "lr": [5e-6, 1e-5, 2e-5, 5e-5],
        "runs": [5, 3, 1, 5],
        "geometric_mean_K": [2.0, 5.0, 5.0 * (1 + 1e-12), 12.0],
        "mean_refusal_drop": [0.0, 0.0, 0.4, 1.0],
    })  # fmt: skip
    pooled = pool_cells(cells_.iloc[::-1])  # sorted by cell-mean K whatever the order
    assert [cell["lrs"] for cell in pooled] == [[5e-6], [1e-5, 2e-5], [5e-5]]
    assert [cell["runs"] for cell in pooled] == [5, 4, 5]
    assert pooled[1]["mean_refusal_drop"] == pytest.approx(0.1)  # (3·0 + 1·0.4) / 4 runs
    assert pooled[1]["geometric_mean_K"] == pytest.approx(5.0, rel=1e-11)
    # Pooled, the drop reaches 10 pp at K = 5; unpooled, a cell below 10 pp and one above it
    # would sit at the same K.
    crossing, k_lo, i = first_crossing(pooled)
    assert (crossing, i) == ("inside", 0)
    assert k_lo == pytest.approx(5.0, rel=1e-9)


def test_no_usable_range_when_k_hi_is_below_k_lo(tmp_path) -> None:
    # The cells above 5e-6 collapse on the benchmark (drop 0.1 > 0.05); the effect criterion,
    # which bounds the factual drop only, still passes.
    collapsed = {lr: replace(cell, benchmark=0.5) for lr, cell in ADAMW.items() if lr != 5e-6}
    root = build(tmp_path / "runs", cells({**ADAMW, **collapsed}))
    agreement = tmp_path / "agreement.json"
    agreement.write_text(json.dumps({"accuracy": 0.945}))
    data = report(root, tmp_path / "out", calibration=agreement)
    k_lo, coverage = data["k_lo"], data["coverage"]
    assert k_lo["crossing"] == "inside"
    assert k_lo["k_hi"] == pytest.approx(geometric_mean(ADAMW[5e-6].K), rel=1e-12)
    assert k_lo["k_hi"] < k_lo["k_lo"]
    assert (k_lo["case"], k_lo["detail"]) == ("no_usable_range", "K_hi < K_lo")
    # Every other criterion passes and SGD has a stable main-condition rate: the fallback.
    assert coverage["other_criteria"] == {"effect_criterion": True, "judge_calibration": True,
                                          "effect_does_not_disappear": True}  # fmt: skip
    assert coverage["largest_stable_sgd_lr"] == 1.0
    assert (coverage["route"], coverage["blocking"]) == ("fallback", [])
    # Without the calibration result the judge criterion is not shown to pass: stop.
    coverage = report(root, tmp_path / "again")["coverage"]
    assert (coverage["route"], coverage["blocking"]) == ("stop", ["judge_calibration"])


def test_without_a_stable_sgd_rate_coverage_that_cannot_be_assessed_stops(tmp_path) -> None:
    collapsed = {lr: replace(cell, benchmark=0.5) for lr, cell in ADAMW.items() if lr != 5e-6}
    unstable = {0.1: sgd(0.1, 1.5, 20, factual=0.7), 0.3: sgd(0.3, 8.0, 17, diverged=(1,)),
                1.0: sgd(1.0, 40.0, 9, benchmark=0.5)}  # fmt: skip
    root = build(tmp_path / "runs", cells({**ADAMW, **collapsed}, sgd_cells=unstable))
    agreement = tmp_path / "agreement.json"
    agreement.write_text(json.dumps({"accuracy": 0.945}))
    coverage = report(root, tmp_path / "out", calibration=agreement)["coverage"]
    assert coverage["case"] == "no_usable_range"
    assert coverage["largest_stable_sgd_lr"] is None
    assert (coverage["route"], coverage["blocking"]) == ("stop", ["stable_main_sgd_rate"])


def test_no_usable_range_without_an_adamw_cell_that_kept_utility(tmp_path) -> None:
    collapsed = {lr: replace(cell, factual=0.7) for lr, cell in ADAMW.items()}
    root = build(tmp_path / "runs", cells(collapsed))
    k_lo = report(root, tmp_path / "out")["k_lo"]
    assert (k_lo["case"], k_lo["k_hi"]) == ("no_usable_range", None)
    assert k_lo["detail"] == "no AdamW cell without collapse"


# --- 5. coverage --------------------------------------------------------------------------------


def test_sgd_cells_spanning_the_required_interval_cover_it(standard) -> None:
    data = standard[1]
    k_lo, coverage = data["k_lo"], data["coverage"]
    lower, upper = k_lo["k_lo"], math.sqrt(k_lo["k_lo"] * k_lo["k_hi"])
    assert coverage["required"] == {"lower": lower, "upper": pytest.approx(upper, rel=1e-15)}
    assert coverage["censored"] is False
    assert coverage["main"]["at_or_below_lower"] == [0.1]  # K 1.5 ≤ K_lo
    assert coverage["main"]["at_or_above_upper"] == [1.0]  # K 40 ≥ U; K 8 lies inside
    assert (coverage["covered_main"], coverage["covered_100x5"]) == (True, True)
    assert coverage["route"] == "equal-KL"
    assert data["summary"]["route"] == "equal-KL"


@pytest.mark.parametrize(
    ("sgd_cells", "end"),
    [({**SGD, 0.1: sgd(0.1, 6.0, 20)}, "lower"),  # the smallest SGD cell lies above K_lo
     ({**SGD, 1.0: sgd(1.0, 40.0, 9, factual=0.7)}, "upper")],  # the cell above U collapsed
)  # fmt: skip
def test_sgd_cells_missing_one_end_do_not_cover(tmp_path, sgd_cells, end) -> None:
    root = build(tmp_path / "runs", cells(sgd_cells=sgd_cells, hundred={}))
    data = report(root, tmp_path / "out")
    coverage = data["coverage"]
    reached = {"lower": coverage["main"]["at_or_below_lower"],
               "upper": coverage["main"]["at_or_above_upper"]}  # fmt: skip
    assert reached[end] == []
    assert reached[{"lower": "upper", "upper": "lower"}[end]] != []
    assert (coverage["covered_main"], coverage["covered_100x5"]) == (False, None)
    assert coverage["route"] == "fallback"
    assert data["experiments"]["sgd100x5"] == {"experiment": X5, "present": False}


def test_coverage_only_under_100x5_needs_an_amendment(tmp_path) -> None:
    root = build(tmp_path / "runs", cells(sgd_cells={**SGD, 0.1: sgd(0.1, 6.0, 20)}))
    coverage = report(root, tmp_path / "out")["coverage"]
    assert (coverage["covered_main"], coverage["covered_100x5"]) == (False, True)
    assert coverage["sgd100x5"]["at_or_below_lower"] == [0.03]
    assert coverage["sgd100x5"]["at_or_above_upper"] == [0.3]
    assert coverage["route"] == "equal-KL with 100x5 (needs amendment)"


def test_neither_condition_covering_routes_to_the_fallback(tmp_path) -> None:
    hundred = {0.3: sgd(0.3, 20.0, 12, X5)}
    root = build(tmp_path / "runs", cells(sgd_cells={**SGD, 0.1: sgd(0.1, 6.0, 20)},
                                          hundred=hundred))  # fmt: skip
    coverage = report(root, tmp_path / "out")["coverage"]
    assert (coverage["covered_main"], coverage["covered_100x5"]) == (False, False)
    assert (coverage["route"], coverage["largest_stable_sgd_lr"]) == ("fallback", 1.0)


def test_a_cell_with_a_diverged_run_is_unstable_and_does_not_cover(tmp_path) -> None:
    # The SGD cell above U has a second seed that diverged, one at a higher rate diverged only,
    # and one more run reports a non-finite K: divergences too.
    sgd_cells = {**SGD, 1.0: replace(SGD[1.0], diverged=(1,)), 3.0: sgd(3.0, math.nan, 5),
                 10.0: Cell(MAIN, "sgd", 10.0, (), (), diverged=(0,))}  # fmt: skip
    root = build(tmp_path / "runs", cells(sgd_cells=sgd_cells, hundred={}))
    out = tmp_path / "out"
    data = report(root, out)
    excluded = data["experiments"]["main"]["excluded"]
    assert excluded["sgd_lr1_s1"]["reason"] == "diverged"
    assert excluded["sgd_lr3_s0"] == {"reason": "diverged",
                                      "detail": "K_harmful in summary.json is nan"}  # fmt: skip
    table = pd.read_csv(out / "cells.csv", dtype=str, keep_default_na=False)
    table = table[table["arm"] == "sgd"].set_index("lr")
    columns = ["runs", "diverged", "collapsed", "stable"]
    assert table.loc["1.0", columns].tolist() == ["1", "1", "False", "False"]
    assert table.loc["3.0", columns].tolist() == ["0", "1", "", "False"]
    assert table.loc["10.0", columns].tolist() == ["0", "1", "", "False"]
    assert table.loc["0.1", "stable"] == "True"
    coverage = data["coverage"]
    assert coverage["main"]["at_or_above_upper"] == []  # K 40 is not counted
    assert (coverage["covered_main"], coverage["route"]) == (False, "fallback")
    assert coverage["largest_stable_sgd_lr"] == 0.3


# --- 6. the effect disappears (A1) --------------------------------------------------------------

SMALL_DROP = (19, 19, 20, 19, 19)  # drops 0.05 0.05 0 0.05 0.05: mean 0.04, upper bound 0.068


def test_the_effect_disappears_with_the_constant_denominator(tmp_path) -> None:
    constant = adamw(5e-5, ADAMW[5e-5].K, SMALL_DROP)
    root = build(tmp_path / "runs", cells({**ADAMW, 5e-5: constant}))
    data = report(root, tmp_path / "out")
    rule = data["effect_disappears"]
    drops = [1 - r / JUDGED for r in SMALL_DROP]
    mean, _, high = interval(drops)
    assert rule["constant"]["refusal_drop"]["mean"] == pytest.approx(mean) == pytest.approx(0.04)
    assert rule["constant"]["refusal_drop"]["ci_high"] == pytest.approx(high)
    assert high < 0.25
    assert rule["token_count"]["refusal_drop"]["mean"] == pytest.approx(0.54)
    assert rule["parts"] == {"constant_mean_below": True, "constant_upper_bound_below": True,
                             "token_count_mean_at_least": True}  # fmt: skip
    assert rule["disappears"] is True and data["summary"]["effect_disappears"] is True
    # The paired per-seed difference, constant − token count.
    token = [1 - r / JUDGED for r in TOKEN_COUNT.refusal]
    differences = [c - t for c, t in zip(drops, token, strict=True)]
    paired = rule["paired_difference"]
    assert [s["value"] for s in paired["per_seed"]] == differences
    mean, low, high = interval(differences)
    assert (paired["mean"], paired["ci_low"], paired["ci_high"]) == pytest.approx(
        (mean, low, high), rel=1e-12)  # fmt: skip
    assert paired["unmatched_seeds"] == {"constant": [], "token_count": []}


def test_the_effect_does_not_disappear_when_the_constant_denominator_keeps_it(standard) -> None:
    rule = standard[1]["effect_disappears"]
    assert rule["parts"] == {"constant_mean_below": False, "constant_upper_bound_below": False,
                             "token_count_mean_at_least": True}  # fmt: skip
    assert rule["disappears"] is False


def test_unmatched_seeds_are_listed_and_left_out_of_the_paired_difference(tmp_path) -> None:
    token = replace(TOKEN_COUNT, K=TOKEN_COUNT.K[:3] + (30.0,), refusal=(9, 10, 8, 9),
                    seeds=(0, 1, 2, 6))  # fmt: skip
    root = build(tmp_path / "runs", cells(token=token))
    paired = report(root, tmp_path / "out")["effect_disappears"]["paired_difference"]
    assert paired["matched_seeds"] == [0, 1, 2]
    assert paired["unmatched_seeds"] == {"constant": [3, 4], "token_count": [6]}
    assert paired["n"] == 3 and paired["df"] == 2


def test_without_the_token_count_branch_the_rule_is_open(tmp_path) -> None:
    constant = adamw(5e-5, ADAMW[5e-5].K, SMALL_DROP)
    root = build(tmp_path / "runs", cells({**ADAMW, 5e-5: constant}, token=None))
    data = report(root, tmp_path / "out")
    rule = data["effect_disappears"]
    assert rule["token_count"] is None
    assert rule["parts"]["token_count_mean_at_least"] is None
    assert rule["disappears"] is None  # the constant-denominator parts alone hold
    assert data["experiments"]["token_count"]["present"] is False


# --- 2. judge, other models, inputs and outputs -------------------------------------------------


def test_the_calibration_result_is_recorded(tmp_path, standard) -> None:
    path = tmp_path / "agreement.json"
    path.write_text(json.dumps({"accuracy": 0.895, "coverage": 1.0}))
    data = report(standard[0], tmp_path / "out", calibration=path)
    assert data["calibration"] == {"file": str(path), "sha256": sha256(path), "accuracy": 0.895,
                                   "at_least": 0.9, "meets_threshold": False}  # fmt: skip
    assert data["summary"]["judge_meets_threshold"] is False
    assert standard[1]["calibration"] is None


def test_the_0_5b_report_holds_the_effect_collapse_and_cells(tmp_path) -> None:
    chosen = [replace(ADAMW[5e-5], experiment=MAIN_05), replace(SGD[0.1], experiment=MAIN_05),
              replace(ADAMW[2e-5], experiment=MAIN_05)]  # fmt: skip
    root = build(tmp_path / "runs", chosen, main=MAIN_05)
    data = report(root, tmp_path / "out", model="0.5b")
    assert data["model"] == "0.5b" and data["gate_model"] is False
    assert data["effect"]["experiment"] == MAIN_05
    assert data["effect"]["passes"] is True
    assert (data["k_lo"], data["coverage"], data["effect_disappears"]) == (None, None, None)
    assert set(data["experiments"]) == {"main", "token_count"}
    assert {cell["experiment"] for cell in data["cells"]} == {MAIN_05}


def test_theta0_must_be_one_complete_run(tmp_path) -> None:
    root = build(tmp_path / "runs", cells(), theta0=None)
    with pytest.raises(ValueError, match="exactly one complete θ₀ run"):
        gate1_report(root, JUDGE, IDENTITY, tmp_path / "out")
    write_theta0(root, MAIN, Theta0())
    write_theta0(root, MAIN, Theta0(), run_id="theta0_s0-again")
    with pytest.raises(ValueError, match="found 2"):
        gate1_report(root, JUDGE, IDENTITY, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_the_token_count_arm_is_the_composed_control_arm() -> None:
    assert arm_of(run_config(TOKEN, "adamw", 5e-5, 0, "train")) == TOKEN_COUNT_ARM


def test_the_manifest_hashes_every_input(standard) -> None:
    root, _ = standard
    out = root.parent / "report"
    manifest = json.loads((out / "manifest.json").read_text())
    inputs = manifest["inputs"]
    for path, digest in inputs.items():
        assert sha256(root / path) == digest, path
    read = [f"{MAIN}/theta0_s0/judge/{JUDGE}.jsonl", f"{MAIN}/theta0_s0/summary.json",
            f"{MAIN}/adamw_lr5e-05_s5-failed/meta.json",
            f"{TOKEN}/adamw_lr5e-05_s0/judge/{JUDGE}.jsonl", f"{X5}/sgd_lr0.03_s0/summary.json",
            f"{MAIN}/sgd_lr1_s0/config.yaml"]  # fmt: skip
    assert set(read) <= set(inputs)
    assert manifest["theta0_run"] == "theta0_s0"
    assert manifest["judge"] == {"name": JUDGE, "identity": IDENTITY}
    assert manifest["outputs"] == {name: sha256(out / name) for name in OUTPUTS[:2]}
    assert set(manifest["code"]["git"]) == {"commit", "dirty"}
    assert manifest["created_at"]
    assert str(root) not in (out / "manifest.json").read_text()  # no local paths


def test_a_rebuild_is_byte_identical_except_created_at(standard, tmp_path) -> None:
    root, _ = standard
    first, second = root.parent / "report", tmp_path / "again"
    gate1_report(root, JUDGE, IDENTITY, second)
    for name in OUTPUTS[:2]:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    manifests = [json.loads((out / "manifest.json").read_text()) for out in (first, second)]
    for manifest in manifests:
        manifest.pop("created_at")
    assert manifests[0] == manifests[1]


def run_script(root: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    command = [sys.executable, str(SCRIPT), str(root), "--judge", JUDGE, "--out", str(out),
               *extra]  # fmt: skip
    return subprocess.run(command, capture_output=True, text=True, cwd=REPO)


def test_the_output_directory_is_never_overwritten(standard, tmp_path) -> None:
    root, _ = standard
    out = tmp_path / "out"
    done = run_script(root, out)
    assert done.returncode == 0, done.stderr
    assert sorted(p.name for p in out.iterdir()) == OUTPUTS
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    done = run_script(root, out)
    assert done.returncode != 0 and "exists" in done.stderr
    with pytest.raises(FileExistsError):
        gate1_report(root, JUDGE, IDENTITY, out)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]  # no partial directory left


def test_a_report_without_runs_writes_nothing(tmp_path) -> None:
    done = run_script(tmp_path / "runs", tmp_path / "out")
    assert done.returncode != 0 and "Not written" in done.stderr
    assert not (tmp_path / "out").exists()


def test_refusal_fit_needs_two_distinct_k() -> None:
    fit = refusal_fit(np.log([5.0, 5.0]), np.array([10, 12]), np.array([20, 20]))
    assert fit["failed"] is True and fit["reason"] == "fewer than two distinct K values"
