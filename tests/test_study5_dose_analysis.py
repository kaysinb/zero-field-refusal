"""Study 5's dose analysis (``analysis.study5_dose`` and scripts/study5_dose_analysis.py) on
synthetic runs and labels only: the run selection by exact rates and its refusal of a duplicate
key, the main contrast and its bootstrap interval on a known design, a missing run and an
instability stop excluded with their reasons, a gross collapse keeping its h, the third candidate
in no contrast, the contrast at K* with its unsupported cells, the first five updates' check, the
K_monitor ratios, the binomial check, every branch of the decision rule and the forecasts' checks,
and the script end to end (written once, numbers only on stdout, no text stored, the sha256 of the
grid and forecasts files enforced)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from zero_field_refusal import dose
from zero_field_refusal.analysis import study5_analysis as sa
from zero_field_refusal.analysis import study5_analysis_cli as cli
from zero_field_refusal.analysis import study5_dose as sd
from zero_field_refusal.analysis.study3_blocks import weight
from zero_field_refusal.code_identity import is_gpu_path
from zero_field_refusal.config import load_config
from zero_field_refusal.judge import _content_hash, judge_identity
from zero_field_refusal.paths import REPO_ROOT
from zero_field_refusal.run_judging import LABEL_ROUNDS
from zero_field_refusal.seeding import derive_seed


def script_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = script_module(REPO_ROOT / "scripts" / "study5_dose_analysis.py", "study5_dose_script")
driver = script_module(REPO_ROOT / "scripts" / "study5_dose_campaigns.py", "study5_dose_driver")
GPU = "e" * 64  # the pilot tables' GPU code

SEEDS_RECORD = json.loads((REPO_ROOT / dose.SEEDS_FILE).read_text())
SEEDS = SEEDS_RECORD["seeds"]
PILOTS = (SEEDS_RECORD["pods"]["a"]["pilot"], SEEDS_RECORD["pods"]["b"]["pilot"])
LOW, MID, HIGH = sd.LOW, sd.MID, sd.HIGH
LO, HI, THIRD = "0.75", "1.3", "2.0"  # the synthetic grid's chosen pair and the third candidate
LATE, EARLY = dict(dose.LATE_CANDIDATES), dict(dose.EARLY)
THETA0 = {"benchmark_accuracy": 0.6, "factual_accuracy": 0.5}
# h(1.25) − h(0.6) in harmful counts per seed, in seed order: 12 positive, 4 negative
Y = (40, 55, 70, 85, 30, 45, 60, 75, 90, 35, 50, 65, -20, -10, -30, -15)
K_AT = {LO: 5.0, HI: 7.5, THIRD: 12.0}  # K_benign at step 25 by late level
K5 = {LOW: 0.4, MID: 0.6, HIGH: 0.8}  # K_monitor after update 5 by early level
SENTINEL = "TEXT-NEVER-STORED"
IDS = [f"judged:{i}" for i in range(521)]
PROMPTS = {pid: f"Prompt {SENTINEL} {i}" for i, pid in enumerate(IDS)}
SET = "harmful_evaluation"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pp(count: float) -> float:
    return 100.0 * count / 521


def harmful(seed: int, early: str, late: str) -> int:
    """The synthetic design's harmful count of a run: c(s, L) = Y_s at both late levels; the
    third candidate far from it."""
    if late == THIRD:
        return {LOW: 0, MID: 200, HIGH: 400}[early]
    i = SEEDS.index(seed)
    return 100 + 3 * i + {LOW: 0, MID: Y[i] // 2, HIGH: Y[i]}[early] + (25 if late == HI else 0)


def counts(h: int, missing: int = 0) -> dict:
    return {"n": 521, "labeled": 521 - missing, "harmful": h, "refusal": 200,
            "safe_non_refusal": 321 - h - missing, "missing": missing}  # fmt: skip


def bootstrap_by_hand(values) -> tuple[float, float]:
    """The registered interval, resample by resample: seeds drawn with replacement, the 2.5 and
    97.5 percentiles of the means."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(derive_seed(0, "study5", "dose-bootstrap"))
    means = [x[rng.integers(0, len(x), size=len(x))].mean() for _ in range(10_000)]
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


# ------------------------------------------------------------------ in-memory runs


def reading(seed: int, early: str, late: str, *, stop: str = "steps", steps: int = 25,
            k: float | None = None, mmlu: float = 0.59, h: int | None = None,
            labels=None, shift: float = 0.0) -> dict:  # fmt: skip
    """One run's readings as the script passes them (``study5_dose``'s ``Run``)."""
    ended = stop == "steps" and steps == 25
    k = K_AT[late] if k is None else k
    h = harmful(seed, early, late) if h is None else h
    row = ({"step": 25, "condition": "step-25", "K_benign": k, "K_harmful": 2 * k,
            "benchmark_accuracy": mmlu, "factual_accuracy": 0.49,
            "heldout_nll_per_token": 1.0} if ended else None)  # fmt: skip
    return {
        "run_id": f"r-s{seed}-{early}-{late}", "seed": seed, "early": early, "late": late,
        "records": None, "stop": stop, "steps": steps, "row": row,
        "monitor5": {"K_monitor": K5[early], "displacement_norm": 0.05},
        "first_steps": {t: {"loss": 2.0 - 0.01 * t, "grad_norm": 1.0 + 0.1 * t + (
            shift if t == 3 else 0.0)} for t in range(5)},
        "labels": (counts(h) if labels is None else labels) if ended else
                  "labels: LabelsPending: no condition answered",
        "receipt_sha256": "r" * 64 if ended else None,
    }  # fmt: skip


def design_runs(seeds=SEEDS, lates=(LO, HI)) -> dict:
    return {(s, e, late): reading(s, e, late) for s in seeds for e in sd.EARLY_NAMES
            for late in lates}  # fmt: skip


# ------------------------------------------------------------------ pure functions


def test_the_binomial_check_12_of_16_is_0038() -> None:
    assert sd.binomial_tail(12, 16) == 2517 / 65536
    assert round(sd.binomial_tail(12, 16), 3) == 0.038
    assert sd.binomial_tail(0, 0) == 1.0 and sd.binomial_tail(16, 16) == 1 / 65536
    found = sd.signs({s: int(y) for s, y in zip(SEEDS, Y, strict=True)}, SEEDS)
    assert (found["positive"], found["negative"], found["zero"]) == (12, 4, 0)
    assert found["p_one_sided"] == 2517 / 65536
    zeros = sd.signs({SEEDS[0]: 0, SEEDS[1]: 1, SEEDS[2]: "no run"}, SEEDS[:3])
    assert (zeros["n"], zeros["zero"], zeros["p_one_sided"]) == (2, 1, 0.75)  # the zero in n


def test_signs_come_from_exact_counts_and_a_zero_stays_in_n() -> None:
    """Harmful counts 100 → 110 at one late level and 210 → 200 at the other: d is exactly 0,
    whatever the percentages' rounding; 12 positive, 3 negative and that zero give the
    registered 12 of 16 (0.038), not 12 of 15."""
    counts = {(1, LOW, LO): 100, (1, HIGH, LO): 110, (1, LOW, HI): 210, (1, HIGH, HI): 200}
    assert sd.count_difference(counts, 1, LO, HI) == 0
    assert sd.signs({1: sd.count_difference(counts, 1, LO, HI)}, [1])["zero"] == 1
    absent = dict(counts) | {(1, HIGH, HI): "r: no labels"}
    assert sd.count_difference(absent, 1, LO, HI) == "r: no labels"
    values = dict(zip(range(16), [1] * 12 + [-1] * 3 + [0], strict=True))
    found = sd.signs(values, list(range(16)))
    assert (found["n"], found["positive"], found["negative"], found["zero"]) == (16, 12, 3, 1)
    assert found["p_one_sided"] == sd.binomial_tail(12, 16) == 2517 / 65536


def test_the_constants_are_the_registered_ones() -> None:
    assert sd.K_STAR == sa.K_STAR == 6.0
    assert sd.EARLY_NAMES == ("0.6", "1.0", "1.25") and sd.FIRST_STEPS == (0, 1, 2, 3, 4)
    assert sd.MONITOR_STEP == 5 and sd.MDD_PP == 8.7 and sd.RESAMPLES == 10_000
    assert derive_seed(0, "study5", "dose-bootstrap") == sd.BOOTSTRAP_SEED


def test_the_bootstrap_is_the_registered_resampling_of_seeds() -> None:
    values = [pp(y) for y in Y]
    found = sd.mean_with_interval(dict(zip(SEEDS, values, strict=True)), SEEDS)
    assert found["mean"] == pytest.approx(pp(sum(Y) / 16), rel=1e-12)
    assert (found["lower"], found["upper"]) == bootstrap_by_hand(values)
    assert found["lower"] < found["mean"] < found["upper"]
    constant = sd.mean_with_interval({s: 3.0 for s in SEEDS}, SEEDS)
    assert constant["lower"] == constant["upper"] == 3.0
    one = sd.mean_with_interval({SEEDS[0]: 3.0}, SEEDS[:2])
    assert (one["n"], one["lower"], one["upper"]) == (1, None, None)
    assert one["excluded"] == [{"seed": SEEDS[1], "reason": "no value"}]


def entry(run_id: str, seed: int, early: float, late: float, ended: bool = True,
          steps: int = 5) -> dict:  # fmt: skip
    return {"run_id": run_id, "seed": seed, "early_lr": early, "early_steps": steps,
            "late_lr": late, "ended": ended}  # fmt: skip


def test_runs_are_keyed_by_their_exact_rates_and_a_duplicate_key_is_refused() -> None:
    s = SEEDS[0]
    found = sd.select_runs([
        entry("a", s, EARLY[LOW], LATE[LO]), entry("b", s, EARLY[HIGH], LATE[HI]),
        entry("c", s, EARLY[LOW], LATE[LO], ended=False),  # not ended: counts for no key
        entry("d", 50001, EARLY[LOW], LATE[LO]),  # a seed of the population, not drawn
    ], SEEDS)  # fmt: skip
    assert found["runs"] == {(s, LOW, LO): "a", (s, HIGH, HI): "b"}
    assert [x["run_id"] for x in found["not_ended"]] == ["c"]
    assert found["other_seeds"] == [{"run_id": "d", "seed": 50001}]
    with pytest.raises(ValueError, match=r"more than one ended run.*a, e"):
        sd.select_runs([entry("a", s, EARLY[LOW], LATE[LO]),
                        entry("e", s, EARLY[LOW], LATE[LO])], SEEDS)  # fmt: skip
    with pytest.raises(ValueError, match="early rate"):  # the next float is no level
        sd.select_runs([entry("a", s, math.nextafter(EARLY[LOW], 1.0), LATE[LO])], SEEDS)
    with pytest.raises(ValueError, match="early rate"):  # nor a rate written as a string
        sd.select_runs([entry("a", s, "7.8e-06", LATE[LO])], SEEDS)
    with pytest.raises(ValueError, match="late rate"):
        sd.select_runs([entry("a", s, EARLY[LOW], dose.ETA_L)], SEEDS)
    with pytest.raises(ValueError, match="early steps"):
        sd.select_runs([entry("a", s, EARLY[LOW], LATE[LO], steps=4)], SEEDS)


def test_the_main_contrast_on_a_known_design() -> None:
    found = sd.main_contrast(design_runs(), SEEDS, LO, HI, THETA0)
    expected = [pp(y) for y in Y]
    for name in ("d", "c_L_lo", "c_L_hi"):
        assert found[name]["n"] == 16 and found[name]["excluded"] == []
        assert found[name]["mean"] == pytest.approx(pp(sum(Y) / 16), rel=1e-12)
    assert [x["d"] for x in found["per_seed"]] == pytest.approx(expected, rel=1e-12)
    assert (found["d"]["lower"], found["d"]["upper"]) == pytest.approx(
        bootstrap_by_hand(expected), rel=1e-12)  # fmt: skip
    assert found["signs"]["positive"] == 12 and found["signs"]["p_one_sided"] == 2517 / 65536
    for late in ("L_lo", "L_hi"):
        shape = found["shape"][late]
        assert shape["n"] == 16
        assert shape["1.25-1.0"] == pytest.approx(pp(sum(y - y // 2 for y in Y) / 16), rel=1e-12)
        assert shape["1.0-0.6"] == pytest.approx(pp(sum(y // 2 for y in Y) / 16), rel=1e-12)


def test_a_missing_run_and_an_instability_stop_are_excluded_with_their_reasons() -> None:
    runs = design_runs()
    a, b = SEEDS[1], SEEDS[2]
    del runs[(a, HIGH, LO)]
    runs[(b, LOW, HI)] = reading(b, LOW, HI, stop="gradient_spike", steps=12)
    found = sd.main_contrast(runs, SEEDS, LO, HI, THETA0)
    assert found["d"]["n"] == 14 and found["c_L_lo"]["n"] == 15 and found["c_L_hi"]["n"] == 15
    why = {x["seed"]: x["reason"] for x in found["d"]["excluded"]}
    assert set(why) == {a, b}
    assert why[a] == "c(L_lo): (1.25, 0.75): no run"
    assert "gradient_spike" in why[b] and "12 updates" in why[b]
    assert [x["seed"] for x in found["shape"]["L_lo"]["excluded"]] == [a]
    assert found["shape"]["L_lo"]["n"] == 15


def test_a_gross_collapse_keeps_its_h_in_the_main_contrast_and_loses_its_support() -> None:
    runs = design_runs()
    s = SEEDS[0]
    runs[(s, HIGH, LO)] = reading(s, HIGH, LO, mmlu=0.40)  # 20 pp of MMLU lost
    found = sd.main_contrast(runs, SEEDS, LO, HI, THETA0)
    assert found["d"]["n"] == 16
    cell = next(c for c in found["cells"] if (c["seed"], c["early"], c["late"]) == (s, HIGH, LO))
    assert cell["gross_collapse"] is True and cell["state"] == "unusable"
    assert cell["h"] == pytest.approx(pp(harmful(s, HIGH, LO)))
    kstar = sd.kstar_contrast(runs, SEEDS, LO, HI, THETA0)
    assert kstar["unsupported"]["cells"][0]["seed"] == s
    assert kstar["unsupported"]["cells"][0]["reason"].startswith("unsupported: gross collapse")
    assert kstar["k"]["n"] == 15


def test_the_third_candidate_enters_no_contrast() -> None:
    plain = design_runs()
    runs = dict(plain)
    for s in PILOTS:
        for e in sd.EARLY_NAMES:
            runs[(s, e, THIRD)] = reading(s, e, THIRD, shift=5.0)
    chosen = {"lo": LO, "hi": HI}
    forecasts = forecast_values()
    with_third = sd.analysis(runs, SEEDS, chosen, THETA0, forecasts)
    without = sd.analysis(plain, SEEDS, chosen, THETA0, forecasts)
    for key in ("main", "kstar", "secondary", "first_updates", "k5_ratios", "decision",
                "missing"):  # fmt: skip
        assert with_third[key] == without[key]
    assert len(with_third["runs"]) == 96 + 6 and sorted(with_third["third_candidate"]) == sorted(
        runs[(s, e, THIRD)]["run_id"] for s in PILOTS for e in sd.EARLY_NAMES)  # fmt: skip
    assert [r["in_design"] for r in with_third["runs"]].count(False) == 6


def test_the_contrast_at_kstar_and_its_unsupported_cells() -> None:
    w = weight(K_AT[LO], K_AT[HI], 6.0)
    runs = design_runs()
    found = sd.kstar_contrast(runs, SEEDS, LO, HI, THETA0)
    assert found["k"]["n"] == 16 and found["unsupported"]["n"] == 0
    assert [x["k"] for x in found["per_seed"]] == pytest.approx([pp(y) for y in Y], rel=1e-12)
    s0 = SEEDS[0]
    mid = next(c for c in found["cells"] if (c["seed"], c["early"]) == (s0, MID))
    assert mid["ln_width"] == pytest.approx(math.log(7.5 / 5.0)) and mid["weight"] == w
    assert mid["h_star"]["harmful"] == pytest.approx(
        (1 - w) * pp(harmful(s0, MID, LO)) + w * pp(harmful(s0, MID, HI)), rel=1e-12)  # fmt: skip
    assert found["per_seed"][0]["h_star"][MID] == mid["h_star"]["harmful"]
    # a wide bracket is supported: no width limit
    a, b, c, d = SEEDS[1:5]
    runs[(a, LOW, LO)] = reading(a, LOW, LO, k=1.0)
    runs[(a, LOW, HI)] = reading(a, LOW, HI, k=50.0)
    runs[(b, LOW, LO)] = reading(b, LOW, LO, k=6.5)  # 6.5 and 7.5: no bracket
    runs[(c, HIGH, HI)] = reading(c, HIGH, HI, stop="k_above_max", steps=9)  # a failed endpoint
    del runs[(d, HIGH, LO)]  # a missing run
    runs[(d, MID, HI)] = reading(d, MID, HI, labels="labels: LabelsPending: missing")
    found = sd.kstar_contrast(runs, SEEDS, LO, HI, THETA0)
    cell = next(c for c in found["cells"] if (c["seed"], c["early"]) == (a, LOW))
    assert cell["state"] == "supported" and cell["ln_width"] == pytest.approx(math.log(50.0))
    reasons = {(x["seed"], x["early"]): x["reason"] for x in found["unsupported"]["cells"]}
    assert set(reasons) == {(b, LOW), (c, HIGH), (d, HIGH)}
    assert reasons[(b, LOW)] == "unsupported: no bracket"
    assert reasons[(c, HIGH)].startswith("unsupported: failed endpoint (")
    assert "instability stop k_above_max" in reasons[(c, HIGH)]
    assert reasons[(d, HIGH)] == "open: missing ((1.25, 0.75): no run)"
    assert [(x["seed"], x["early"]) for x in found["without_labels"]["cells"]] == [(d, MID)]
    assert found["k"]["n"] == 13  # b, c and d have no k
    assert {x["seed"] for x in found["k"]["excluded"]} == {b, c, d}


def test_the_first_five_updates_check_shows_a_difference() -> None:
    runs = design_runs(SEEDS[:2])
    same = sd.first_updates(runs, SEEDS[:2], LO, HI)
    assert same["compared_cells"] == 6 and same["differing_cells"] == 0
    assert same["max_abs"]["value"] == 0.0 and same["max_rel"]["value"] == 0.0
    assert all(c["compared"] == 12 and c["missing"] == [] for c in same["cells"])
    s = SEEDS[1]
    runs[(s, MID, HI)] = reading(s, MID, HI, shift=0.003)
    runs[(s, HIGH, HI)]["monitor5"] = {"K_monitor": K5[HIGH] * 1.5, "displacement_norm": 0.05}
    found = sd.first_updates(runs, SEEDS[:2], LO, HI)
    assert found["differing_cells"] == 2
    cell = next(c for c in found["cells"] if (c["seed"], c["early"]) == (s, MID))
    assert cell["max_abs"] == {"value": pytest.approx(0.003), "metric": "grad_norm", "step": 3}
    assert cell["max_rel"]["value"] == pytest.approx(0.003 / 1.303)
    assert found["max_abs"] == {"value": pytest.approx(0.4), "metric": "K_monitor", "step": 5,
                                "seed": s, "early": HIGH}  # fmt: skip
    assert found["max_rel"]["value"] == pytest.approx(0.4 / 1.2)
    del runs[(s, LOW, LO)]
    runs[(s, MID, LO)]["first_steps"] = "no metrics.jsonl"
    found = sd.first_updates(runs, SEEDS[:2], LO, HI)
    assert found["not_compared"] == [{"seed": s, "early": LOW, "reason": "(0.6, 0.75): no run"}]
    cell = next(c for c in found["cells"] if (c["seed"], c["early"]) == (s, MID))
    assert cell["compared"] == 2 and len(cell["missing"]) == 10


def test_the_k5_ratios_take_the_l_hi_run_when_l_lo_has_none() -> None:
    runs = design_runs(SEEDS[:3])
    del runs[(SEEDS[0], HIGH, LO)]
    runs[(SEEDS[1], LOW, LO)]["monitor5"] = "no monitor.jsonl"
    runs[(SEEDS[1], LOW, HI)]["monitor5"] = "0 monitor readings at step 5"
    found = sd.k5_ratios(runs, SEEDS[:3], LO, HI)
    first = found["per_seed"][0]
    assert first["K_monitor_5"][HIGH]["from"] == "L_hi"
    assert first["K_monitor_5"][LOW]["from"] == "L_lo"
    assert found["1.25/0.6"]["n"] == 2 and found["1.0/0.6"]["n"] == 2
    assert found["1.25/0.6"]["geometric_mean"] == pytest.approx(2.0, rel=1e-12)
    assert found["1.0/0.6"]["geometric_mean"] == pytest.approx(1.5, rel=1e-12)
    assert found["1.25/0.6"]["excluded"][0]["seed"] == SEEDS[1]
    assert "no monitor.jsonl" in found["1.25/0.6"]["excluded"][0]["reason"]


def forecast_values(main_c: float = 7.5, main_a: float = -2.0) -> dict:
    def one(main: float, positive: int) -> dict:
        points = {"main": main, "main_L_lo": main + 1, "main_L_hi": main - 1, "kstar": main}
        return {**points, **{f"{key}_interval80": [x - 3, x + 3] for key, x in points.items()},
                "positive_seeds": positive, "k5_ratio": 1.9}  # fmt: skip

    return {"C": one(main_c, 12), "A": one(main_a, 7)}


@pytest.mark.parametrize(
    ("lower", "upper", "c", "a", "code", "refuted"),
    [
        (0.0, 10.0, 5.0, 1.0, "within_mdd", []),
        (0.0, 3.0, 5.0, 1.0, "within_mdd", ["C"]),  # refuted, but within 8.7 pp of A
        (5.0, 15.0, 10.0, 0.0, "C_confirmed", ["A"]),
        (-3.0, 3.0, 10.0, 0.0, "A_confirmed", ["C"]),
        (20.0, 30.0, 10.0, 0.0, "inconclusive", ["C", "A"]),
        (-5.0, 15.0, 10.0, 0.0, "inconclusive", []),
        (5.0, 15.0, 10.0, 1.3, "C_confirmed", ["A"]),  # 8.7 pp apart as decimals: not within
        (0.0, 10.0, 10.0, -5.0, "C_confirmed", ["A"]),  # a point on a bound is not refuted
        (None, None, 10.0, 0.0, "undetermined", None),
    ],
)  # fmt: skip
def test_each_branch_of_the_decision_rule(lower, upper, c, a, code, refuted) -> None:
    found = sd.verdict({"n": 16, "mean": 9.0, "lower": lower, "upper": upper}, c, a)
    assert found["code"] == code and found["refuted"] == refuted
    assert (found["C"], found["A"], found["lower"], found["upper"]) == (c, a, lower, upper)
    words = {"within_mdd": "inconclusive: forecasts within the minimum detectable difference",
             "C_confirmed": "C confirmed against A", "A_confirmed": "A confirmed against C",
             "inconclusive": "inconclusive",
             "undetermined": "undetermined: no observed interval"}  # fmt: skip
    assert found["verdict"] == words[code]


@pytest.mark.parametrize(
    ("change", "why"),
    [
        (lambda f: f["C"].update(kstar="none"), "kstar is not a finite number"),
        (lambda f: f["C"].pop("kstar"), r"missing \['kstar'\]"),
        (lambda f: f["A"].update(main=math.nan), "main is not a finite number"),
        (lambda f: f["C"].update(main_interval80=[2.0, 1.0]), "main_interval80"),
        (lambda f: f["C"].update(kstar_interval80=[1.0]), "kstar_interval80"),
        (lambda f: f["C"].update(main_L_lo_interval80=None), "main_L_lo_interval80"),
        (lambda f: f["A"].pop("main_L_hi_interval80"), r"missing \['main_L_hi_interval80'\]"),
        (lambda f: f["A"].update(notes=1.0), r"other keys \['notes'\]"),
        (lambda f: f["A"].update(positive_seeds=17), "positive_seeds"),
        (lambda f: f["A"].update(positive_seeds=True), "positive_seeds"),
        (lambda f: f["C"].update(k5_ratio=0.0), "k5_ratio"),
        (lambda f: f.pop("A"), "no object A"),
    ],
)  # fmt: skip
def test_the_forecasts_are_checked(change, why) -> None:
    found = forecast_values()
    assert sd.check_forecasts(found) == found
    change(found)
    with pytest.raises(ValueError, match=why):
        sd.check_forecasts(found)


def test_the_module_is_analysis_code_outside_the_gpu_identity() -> None:
    assert is_gpu_path("src/zero_field_refusal/analysis/study5_dose.py") is False


# ------------------------------------------------------------------ the script, end to end


IDENTITY = judge_identity(load_config("judge/wildguard"))


def dump(path: Path, value) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return data


def lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def grid_record(folder: Path, out: Path) -> bytes:
    """The grid record exactly as the Pods' driver writes it (``study5_dose_campaigns.py grid``)
    from the two Pods' pilot tables of the synthetic K, the other Pod's table already fetched."""
    paths = {}
    for pod, seed in zip(dose.PODS, PILOTS, strict=True):
        rows = [{"seed": seed, "early": e, "late": late, "label": f"x/s{seed}",
                 "run_id": f"r-{seed}-{e}-{late}", "outcome": "done", "steps": 25, "stop": "steps",
                 "complete": True, "K": K_AT[late]}
                for e in sd.EARLY_NAMES for late in (LO, HI, THIRD)]  # fmt: skip
        paths[pod] = folder / f"pilot-table-{pod}.json"
        dump(paths[pod], {"kind": driver.TABLE_KIND, "pod": pod, "seed": seed, "gpu_code": GPU,
                          "rows": rows})  # fmt: skip
    import os
    from unittest import mock

    with mock.patch.dict(os.environ, {"ARTIFACT_REMOTE": "gs://bucket"}):
        code = driver.main(["grid", "--own", str(paths["a"]), "--other-remote",
                            "study5/dose-b-t/pilot/pilot-table.json", "--fetched", str(paths["b"]),
                            "--wait", "0", "--pilot", f"a={PILOTS[0]}", "--pilot",
                            f"b={PILOTS[1]}", "--gpu-code", GPU, "--out", str(out)])  # fmt: skip
    assert code == 0
    return out.read_bytes()


class World:
    """A synthetic runs root: θ₀'s evaluation run (role fresh, under ``study5/``), the 16 seeds ×
    3 early levels × the chosen pair, the pilot seeds' third candidate, each run with its records,
    responses (each holding ``SENTINEL``) and WildGuard labels; the grid record ``grid_choice``
    gives on the pilot's K, and the forecasts."""

    def __init__(self, tmp: Path, *, skip=(), stopped=(), missing_labels=None) -> None:
        self.tmp, self.runs = tmp, tmp / "runs"
        self.theta0 = self.runs / "study5" / "s5-theta0"
        dump_config(self.theta0, {"run_type": "theta0", "seed": 50000,
                                  "eval": {"harmful": {"prompt_set": SET}},
                                  "study5": {"role": "fresh"},
                                  "optimizer": {"name": "adam_frozen", "lr": 1.3e-05}})  # fmt: skip
        dump(self.theta0 / "summary.json", {**THETA0, "K_benign": 0.0})
        self.missing_labels = missing_labels or {}
        for s in SEEDS:
            for e in sd.EARLY_NAMES:
                for late in (LO, HI, THIRD):
                    key = (s, e, late)
                    if key in skip or (late == THIRD and s not in PILOTS):
                        continue
                    self.run(s, e, late, stop=("gradient_spike", 12) if key in stopped
                             else ("steps", 25))  # fmt: skip
        self.grid = tmp / "inputs" / "grid.json"
        self.grid_sha = sha(grid_record(tmp / "inputs", self.grid))
        self.forecasts = tmp / "inputs" / "forecasts.json"
        self.forecasts_sha = sha(dump(self.forecasts, forecast_values()))

    def run(self, seed: int, early: str, late: str, *, stop=("steps", 25),
            run_id: str | None = None) -> Path:  # fmt: skip
        reason, steps = stop
        run_dir = self.runs / "study5" / (run_id or f"dose-s{seed}-e{early}-l{late}")
        dump_config(run_dir, {
            "run_type": "train", "seed": seed, "experiment": "study5",
            "eval": {"harmful": {"prompt_set": SET}},
            "optimizer": {"name": "adam_frozen", "lr": LATE[late]},
            "schedule": {"policy": "fixed", "evaluate_at": [25]},
            "study5": {"role": "dose"},
            "train": {"steps": 25, "batch_size": 20,
                      "rate_schedule": {"early_steps": 5, "early_lr": EARLY[early]}},
        })  # fmt: skip
        dump(run_dir / "meta.json", {"run_id": run_dir.name, "inputs": {"harmful_prompts": {
            "set": SET, "count": 521, "ids_sha256": sha(json.dumps(IDS).encode())}}})  # fmt: skip
        ended = reason == "steps"
        dump(run_dir / "terminal.json", {"steps": steps, "stop": {"reason": reason},
                                         "evaluations": [25] if ended else []})  # fmt: skip
        i, index = SEEDS.index(seed), sd.EARLY_NAMES.index(early)
        metrics = []
        for t in range(steps):  # the late rate shows from update 6 (step 5) on
            metrics.append({"step": t, "time_s": 0.1 * t, "loss": 2.0 - 0.01 * t - 0.1 * index,
                            "grad_norm": 1.0 + 0.1 * t + (LATE[late] * 1e4 if t >= 5 else 0.0)})
            if (t + 1) % 5 == 0:
                metrics.append({"step": t, "time_s": 0.1 * t, "K_monitor": 0.1 * t})
        lines(run_dir / "metrics.jsonl", metrics)
        lines(run_dir / "monitor.jsonl", [
            {"step": t, "K_monitor": K5[early] * (1 + 0.01 * i) if t == 5 else LATE[late] * t,
             "displacement_norm": 0.01 * (1 + index) * (1 if t == 5 else LATE[late] * 1e5)}
            for t in range(5, steps + 1, 5)])  # fmt: skip
        if not ended:
            return run_dir
        k = K_AT[late]
        lines(run_dir / "evaluations.jsonl", [{
            "step": 25, "condition": "step-25", "K_benign": k, "K_harmful": 2 * k,
            "benchmark_accuracy": 0.59, "factual_accuracy": 0.49,
            "heldout_nll_per_token": 1.0}])  # fmt: skip
        n, none = harmful(seed, early, late), self.missing_labels.get((seed, early, late), 0)
        responses, labels = [], []
        for j, pid in enumerate(IDS):
            text = f"Reply {SENTINEL} {j}"
            responses.append({"condition": "step-25", "item_id": pid, "text": text})
            label = (None if j >= 521 - none else "harmful_compliance" if j < n
                     else "refusal" if j < n + 200 else "safe_non_refusal")  # fmt: skip
            labels.append({"item_id": f"step-25/{pid}", **IDENTITY,
                           "content_sha256": _content_hash(PROMPTS[pid], text), "label": label,
                           "created_at": "2026-11-20T12:00:00+00:00"})  # fmt: skip
        lines(run_dir / "responses" / f"{SET}.jsonl", responses)
        lines(run_dir / "judge" / "wildguard.jsonl", labels)
        expected = [f"step-25/{pid}" for pid in IDS]
        dump(run_dir / "judge" / "wildguard.receipt.json", {
            "judge": "wildguard", "identity": IDENTITY, "expected_items": expected,
            "items": len(expected), "rounds": LABEL_ROUNDS if none else 1, "missing": none,
            "labels_sha256": sha((run_dir / "judge" / "wildguard.jsonl").read_bytes()),
            "completed_at": "2026-11-21T00:00:00+00:00"})  # fmt: skip
        return run_dir

    def argv(self, out: Path, **change) -> list[str]:
        given = {"--runs": self.runs, "--seeds": REPO_ROOT / dose.SEEDS_FILE,
                 "--grid": self.grid, "--grid-sha256": self.grid_sha,
                 "--theta0-run": self.theta0, "--forecasts": self.forecasts,
                 "--forecasts-sha256": self.forecasts_sha, "--out": out} | change  # fmt: skip
        return [str(x) for pair in given.items() for x in pair]


def dump_config(run_dir: Path, cfg: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(json.dumps(cfg))  # JSON is YAML


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(cli, "harmful_prompts", lambda cfg: PROMPTS)


def call(argv) -> tuple[int, str]:
    try:
        return script.main(argv), ""
    except SystemExit as exc:
        return 1, str(exc.code)


def test_the_script_writes_the_full_design_once_numbers_only(tmp_path, patched, capsys) -> None:
    s0 = SEEDS[0]
    world = World(tmp_path, missing_labels={(s0, HIGH, LO): 5})
    out = tmp_path / "record.json"
    assert call(world.argv(out)) == (0, "")
    stdout = capsys.readouterr().out
    data = out.read_bytes()
    record = json.loads(data)
    assert SENTINEL not in data.decode() and SENTINEL not in stdout and "Reply" not in stdout
    assert stdout.splitlines()[-1] == f"record sha256 {sha(data)}"
    assert record["chosen"] == {"lo": LO, "hi": HI}
    assert record["inputs"]["grid"]["sha256"] == world.grid_sha
    assert record["inputs"]["forecasts"]["sha256"] == world.forecasts_sha
    assert record["inputs"]["seeds"]["sha256"] == sha((REPO_ROOT / dose.SEEDS_FILE).read_bytes())
    assert record["inputs"]["theta0"] == {"run_id": "s5-theta0", "accuracies": THETA0}
    assert record["missing"] == [] and len(record["runs"]) == 96 + 6
    assert len(record["third_candidate"]) == 6 and len(record["label_receipts"]) == 102
    assert all(len(v) == 64 for v in record["label_receipts"].values())
    assert set(record["git"]) == {"commit", "dirty"} and record["written_utc"].endswith("+00:00")
    # the main contrast: c(s, L) = Y_s harmful responses of 521 at both late levels
    expected = [pp(y) for y in Y]
    main = record["main"]
    assert main["d"]["n"] == 16 and main["d"]["mean"] == pytest.approx(pp(sum(Y) / 16))
    lower, upper = bootstrap_by_hand(expected)
    assert (main["d"]["lower"], main["d"]["upper"]) == pytest.approx((lower, upper), rel=1e-12)
    assert lower < 7.5 < upper and not lower <= -2.0  # C's point inside, A's outside
    assert main["signs"]["positive"] == 12 and main["signs"]["negative"] == 4
    assert round(main["signs"]["p_one_sided"], 3) == 0.038
    decision = record["decision"]
    assert decision["main"]["code"] == "C_confirmed" and decision["main"]["refuted"] == ["A"]
    assert decision["main"]["verdict"] == "C confirmed against A"
    assert decision["kstar"]["code"] == "C_confirmed"
    assert decision["beside"]["positive_seeds"] == {"C": 12, "A": 7,
                                                    "observed": {"positive": 12, "n": 16}}
    assert decision["beside"]["k5_ratio"]["observed"] == pytest.approx(2.0, rel=1e-12)
    assert set(decision["beside"]) == set(sd.FIELDS) and len(sd.FIELDS) == 10
    lo_interval = decision["beside"]["main_L_lo_interval80"]
    assert lo_interval["C"] == [5.5, 11.5] and lo_interval["A"] == [-4.0, 2.0]
    assert lo_interval["observed"]["lower"] == main["c_L_lo"]["lower"]
    assert decision["beside"]["kstar"]["observed"]["mean"] == record["kstar"]["k"]["mean"]
    # the contrast at K*, every cell bracketed by K = 5.0 and 7.5
    kstar = record["kstar"]
    assert kstar["k"]["n"] == 16 and kstar["unsupported"]["n"] == 0
    assert kstar["k"]["mean"] == pytest.approx(pp(sum(Y) / 16), rel=1e-12)
    # the secondary measure: 5 missing labels of one run count as harmful
    secondary = record["secondary"]["main"]
    assert secondary["d"]["mean"] == pytest.approx(pp(sum(Y) / 16) + pp(5) / 2 / 16, rel=1e-12)
    assert main["per_seed"][0]["d"] == pytest.approx(pp(Y[0]))
    run = next(r for r in record["runs"] if r["run_id"] == f"dose-s{s0}-e{HIGH}-l{LO}")
    assert run["labels"]["missing"] == 5 and run["state"] == "usable"
    assert run["K_monitor_5"] == pytest.approx(K5[HIGH])
    assert run["mmlu_loss_pp"] == pytest.approx(1.0)
    # the first five updates are shared; the late rate shows after them
    first = record["first_updates"]
    assert first["compared_cells"] == 48 and first["differing_cells"] == 0
    assert all(c["compared"] == 12 for c in first["cells"])
    # written once
    assert call(world.argv(out))[0] == 1
    assert out.read_bytes() == data


def test_the_script_excludes_a_missing_run_and_an_instability_stop(tmp_path, patched,
                                                                    capsys) -> None:  # fmt: skip
    a, b = SEEDS[1], SEEDS[2]
    world = World(tmp_path, skip={(a, HIGH, LO)}, stopped={(b, LOW, HI)})
    out = tmp_path / "record.json"
    assert call(world.argv(out)) == (0, "")
    assert f"missing s{a} early {HIGH} late {LO}" in capsys.readouterr().out
    record = json.loads(out.read_bytes())
    assert record["missing"] == [{"seed": a, "early": HIGH, "late": LO}]
    main = record["main"]
    assert main["d"]["n"] == 14
    why = {x["seed"]: x["reason"] for x in main["d"]["excluded"]}
    assert why[a] == "c(L_lo): (1.25, 0.75): no run"
    assert "gradient_spike" in why[b]
    stopped = next(r for r in record["runs"] if r["run_id"] == f"dose-s{b}-e{LOW}-l{HI}")
    assert stopped["state"] == "scientific" and stopped["stop"] == "gradient_spike"
    assert "no condition answered" in stopped["labels"]["missing"]
    assert stopped["receipt_sha256"] is None
    unsupported = record["kstar"]["unsupported"]["cells"]
    reasons = {(x["seed"], x["early"]): x["reason"] for x in unsupported}
    assert set(reasons) == {(a, HIGH), (b, LOW)}
    assert reasons[(b, LOW)].startswith("unsupported: failed endpoint")
    assert record["kstar"]["k"]["n"] == 14
    k5 = record["k5_ratios"]["per_seed"][1]
    assert k5["seed"] == a and k5["K_monitor_5"][HIGH]["from"] == "L_hi"
    assert record["first_updates"]["not_compared"] == [
        {"seed": a, "early": HIGH, "reason": f"({HIGH}, {LO}): no run"}]  # fmt: skip


def test_the_script_refuses_a_key_with_two_ended_runs(tmp_path, patched) -> None:
    world = World(tmp_path)
    s = SEEDS[3]
    world.run(s, MID, HI, run_id="dose-again")
    out = tmp_path / "record.json"
    code, why = call(world.argv(out))
    assert code == 1 and "more than one ended run" in why
    assert f"dose-again, dose-s{s}-e{MID}-l{HI}" in why and not out.exists()


def test_the_script_refuses_inputs_whose_sha256_is_not_the_one_given(tmp_path, patched) -> None:
    world = World(tmp_path)
    out = tmp_path / "record.json"
    code, why = call(world.argv(out, **{"--forecasts-sha256": "0" * 64}))
    assert code == 1 and "the forecasts: its sha256" in why and not out.exists()
    code, why = call(world.argv(out, **{"--grid-sha256": "0" * 64}))
    assert code == 1 and "the grid record: its sha256" in why and not out.exists()
    record = json.loads(world.grid.read_text())
    inner = world.tmp / "inputs" / "inner.json"  # the rule's return value alone: not the record
    code, why = call(world.argv(out, **{"--grid": inner,
                                        "--grid-sha256": sha(dump(inner, record["grid"]))}))  # fmt: skip
    assert code == 1 and "study5-dose-grid" in why and not out.exists()
    record["grid"]["chosen"]["lo"], record["grid"]["chosen"]["hi"] = HI, THIRD  # not the rule's
    tampered = world.tmp / "inputs" / "tampered.json"
    code, why = call(world.argv(out, **{"--grid": tampered,
                                        "--grid-sha256": sha(dump(tampered, record))}))  # fmt: skip
    assert code == 1 and "not the rule's" in why and not out.exists()


def test_the_script_refuses_a_run_at_a_rate_of_no_level(tmp_path, patched) -> None:
    world = World(tmp_path)
    run_dir = world.runs / "study5" / f"dose-s{SEEDS[0]}-e{LOW}-l{LO}"
    cfg = json.loads((run_dir / "config.yaml").read_text())
    cfg["train"]["rate_schedule"]["early_lr"] = math.nextafter(EARLY[LOW], 1.0)
    (run_dir / "config.yaml").write_text(json.dumps(cfg))
    code, why = call(world.argv(tmp_path / "record.json"))
    assert code == 1 and "its early rate" in why and run_dir.name in why
