"""Study 5's analysis (docs/study5-preregistration.md; ``analysis.study5_analysis`` and its CLI
``analysis.study5_analysis_cli``), on synthetic records and labels only: the permutation p-value
on a known small case and its determinism, the reading and the minimum rule, the outcome
interpolated by study 3's own functions through the adapter, the line scans' odd and even parts,
the crossed controls' pairing, the ladder's table, the extreme blocks' contrast and the reference
estimator by hand, a missing file reported by name, and the CLI end to end (never overwriting,
numbers only on stdout, no text stored)."""

from __future__ import annotations

import itertools
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from zero_field_refusal.analysis import study3_estimates
from zero_field_refusal.analysis import study5_analysis as sa
from zero_field_refusal.analysis.study3_blocks import weight
from zero_field_refusal.paths import REPO_ROOT

FIXATION = json.loads((REPO_ROOT / "docs" / "study5-fixation.json").read_text())
TEST = FIXATION["permutation_test"]


def small_test(permutations: int = 100_000, seed: int = 1234, minimum: int = 5) -> dict:
    """The record's test settings with another count, seed and minimum (for small cases)."""
    return {**TEST, "permutations": permutations, "seed": seed, "min_supported_seeds": minimum,
            "p_value": f"(1 + #{{rho_perm <= rho_obs}}) / (1 + {permutations})"}  # fmt: skip


# ------------------------------------------------------------------ the confirmatory test


def exact_p(x, y) -> float:
    """The exact one-sided permutation p-value: the share of all n! permutations of y whose
    Spearman ρ is at or below the observed one."""
    observed = stats.spearmanr(x, y).statistic
    found = [stats.spearmanr(x, perm).statistic for perm in itertools.permutations(y)]
    return sum(r <= observed + 1e-12 for r in found) / len(found)


def test_spearman_is_average_rank_pearson() -> None:
    x = [1.0, 2.0, 2.0, 3.0, 5.0, 4.0]
    y = [0.3, 0.1, 0.2, 0.2, 0.9, 0.5]
    assert sa.spearman(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-15)
    assert math.isnan(sa.spearman([1, 2, 3], [4, 4, 4]))


def test_the_permutation_p_value_is_the_exact_distributions_limit_and_deterministic() -> None:
    x = [0.1, 0.4, 0.2, 0.9, 0.5]
    y = [0.30, 0.25, 0.20, 0.05, 0.10]
    exact = exact_p(x, y)
    found = sa.permutation_p(x, y, permutations=100_000, seed=1234)
    assert found["rho"] == pytest.approx(stats.spearmanr(x, y).statistic, abs=1e-15)
    assert found["p_value"] == pytest.approx(exact, abs=0.005)
    assert found["p_value"] == (1 + found["count"]) / (1 + 100_000)
    assert sa.permutation_p(x, y, permutations=100_000, seed=1234) == found
    assert sa.permutation_p(x, y, permutations=100_000, seed=1235)["count"] != found["count"]


def test_the_record_seed_and_count_are_the_ones_used() -> None:
    """The record's settings are taken whole: its seed draws the permutations, its formula is the
    one implemented, and a record with another formula is refused."""
    rng = np.random.default_rng(7)
    p = {50000 + i: float(v) for i, v in enumerate(rng.normal(size=24))}
    h = {s: float(v) for s, v in zip(p, rng.normal(size=24), strict=True)}
    found = sa.confirmatory(p, h, list(p), TEST)
    assert found["permutations"] == 100_000 and found["seed"] == TEST["seed"]
    again = sa.permutation_p([p[s] for s in p], [h[s] for s in p], permutations=100_000,
                             seed=TEST["seed"])  # fmt: skip
    assert found["p_value"] == again["p_value"]
    with pytest.raises(ValueError, match="formula"):
        sa.confirmatory(p, h, list(p), {**TEST, "p_value": "count / N"})


def test_a_negative_association_reads_as_a_prediction() -> None:
    seeds = list(range(50000, 50024))
    p = {s: float(i) for i, s in enumerate(seeds)}
    h = {s: 0.4 - 0.01 * i + 0.002 * ((i * 7) % 5) for i, s in enumerate(seeds)}
    found = sa.confirmatory(p, h, seeds, small_test(20_000))
    assert found["rho"] < -0.9 and found["p_value"] < 0.05
    assert found["descriptive"] is False
    assert found["reading"] == sa.READINGS["predicts"]
    # a positive association reads as no prediction
    flipped = sa.confirmatory(p, {s: -v for s, v in h.items()}, seeds, small_test(20_000))
    assert flipped["p_value"] > 0.05 and flipped["reading"] == sa.READINGS["does_not_predict"]


def test_the_readings_are_the_registrations_sentences() -> None:
    text = " ".join((REPO_ROOT / "docs" / "study5-preregistration.md").read_text().split())
    for sentence in sa.READINGS.values():
        assert f'"{sentence}"' in text


def test_below_the_minimum_the_result_is_descriptive() -> None:
    seeds = list(range(50000, 50040))
    p = {s: float(i) for i, s in enumerate(seeds)}
    h: dict[int, float | str] = {s: -float(i) for i, s in enumerate(seeds)}
    for s in seeds[19:]:
        h[s] = "unsupported: too wide"
    found = sa.confirmatory(p, h, seeds, TEST)
    assert found["n"] == 19 and found["descriptive"] is True
    assert found["p_value"] is None and found["reading"] is None
    assert found["rho"] == pytest.approx(-1.0)
    assert len(found["excluded"]) == 21
    assert found["excluded"][0] == {"seed": 50019, "reason": "h: unsupported: too wide"}
    h[seeds[19]] = -19.0
    assert sa.confirmatory(p, h, seeds, small_test(2_000, minimum=20))["descriptive"] is False


def test_the_retrospective_bootstrap_resamples_seeds() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    y = -x + rng.normal(size=60)
    settings = FIXATION["retrospective_interval"]
    found = sa.bootstrap_interval(x, y, settings)
    assert found["resamples"] == 10_000 and found["seed"] == settings["seed"]
    assert found["lower"] < sa.spearman(x, y) < found["upper"] < 0
    assert sa.bootstrap_interval(x, y, settings) == found
    # by hand: the same generator, seeds resampled with replacement, the percentile interval
    gen = np.random.default_rng(settings["seed"])
    rhos = []
    for _ in range(10_000):
        idx = gen.integers(0, 60, size=60)
        rhos.append(stats.spearmanr(x[idx], y[idx]).statistic)
    assert found["lower"] == pytest.approx(np.percentile(rhos, 2.5), abs=1e-12)
    assert found["upper"] == pytest.approx(np.percentile(rhos, 97.5), abs=1e-12)


# ------------------------------------------------------------------ the outcome at K*


THETA0 = {"benchmark_accuracy": 0.6, "factual_accuracy": 0.5}


def row(k: float, *, mmlu: float = 0.59, factual: float = 0.49, kh: float = 3.0) -> dict:
    return {"step": 25, "K_benign": k, "K_harmful": kh, "benchmark_accuracy": mmlu,
            "factual_accuracy": factual, "heldout_nll_per_token": 1.0}  # fmt: skip


def two_rates(seed: int, k_lo: float, k_hi: float, **kw):
    lo = sa.endpoint_of(f"lo-s{seed}", seed=seed, lr=1.3e-5, stop="steps", row=row(k_lo, **kw),
                        theta0=THETA0)  # fmt: skip
    hi = sa.endpoint_of(f"hi-s{seed}", seed=seed, lr=1.83e-5, stop="steps", row=row(k_hi, **kw),
                        theta0=THETA0)  # fmt: skip
    return [lo, hi]


def counts(harmful: int, refusal: int, missing: int = 0) -> dict:
    return {"n": 521, "harmful": harmful, "refusal": refusal, "missing": missing,
            "safe_non_refusal": 521 - harmful - refusal - missing}  # fmt: skip


def test_the_adapter_gives_study3s_arm_exactly_its_freeze_entry() -> None:
    """A supported seed's block is study 3's freeze entry, ``{"seed", "arms": {"F": {"lo", "hi",
    "w", "width"}}}``, its weight study 3's ln K weight at K* = 6.00; study 3's ``_arm`` then gives
    the harm at K*, the hand interpolation."""
    support, block = sa.f_support(50000, two_rates(50000, 5.0, 7.0))
    w = weight(5.0, 7.0, 6.0)
    assert block == {
        "seed": 50000,
        "arms": {
            "F": {
                "lo": "lo-s50000",
                "hi": "hi-s50000",
                "w": w,
                "width": math.log(7.0) - math.log(5.0),
            }
        },
    }
    assert support["state"] == "supported" and support["weight"] == w
    labels = {"lo-s50000": counts(52, 400, 3), "hi-s50000": counts(104, 300)}
    found = sa.shares_at_kstar(block, labels)
    assert found["harmful"] == pytest.approx((1 - w) * 52 / 521 + w * 104 / 521, abs=1e-15)
    assert found["refusal"] == pytest.approx((1 - w) * 400 / 521 + w * 300 / 521, abs=1e-15)
    assert found["harmful_high"] == pytest.approx((1 - w) * 55 / 521 + w * 104 / 521, abs=1e-15)
    harm = {run: sa.Counts(n=521, harmful=c["harmful"], missing=c["missing"])
            for run, c in labels.items()}  # fmt: skip
    assert found["harmful"] == study3_estimates._arm(block, "F", harm)[0]
    # the order is by K, not by rate: a seed whose η_u run has the lower K
    _, swapped = sa.f_support(50001, two_rates(50001, 7.0, 5.0))
    assert swapped["arms"]["F"]["lo"] == "hi-s50001"


@pytest.mark.parametrize(
    ("ks", "kw", "reason"),
    [((6.5, 7.0), {}, "no bracket"), ((4.0, 8.0), {}, "too wide"), ((6.0, 6.0), {}, "tied"),
     ((5.0, 7.0), {"mmlu": 0.40}, "gross collapse")],
)  # fmt: skip
def test_study3s_support_rule_decides(ks, kw, reason) -> None:
    support, block = sa.f_support(50000, two_rates(50000, *ks, **kw))
    assert block is None and support["state"] == "unsupported" and support["reason"] == reason


def test_a_failed_or_missing_endpoint_removes_support() -> None:
    lo, hi = two_rates(50000, 5.0, 7.0)
    stopped = sa.endpoint_of("hi-s50000", seed=50000, lr=1.83e-5, stop="k_above_max", row=None,
                             theta0=THETA0)  # fmt: skip
    support, block = sa.f_support(50000, [lo, stopped])
    assert block is None and support["reason"] == "failed endpoint"
    support, block = sa.f_support(50000, [lo])
    assert block is None and (support["state"], support["reason"]) == ("open", "missing")


def test_the_outcome_lists_an_unsupported_seed_with_its_reason() -> None:
    found = sa.seed_outcome(50000, two_rates(50000, 6.5, 7.0), [], {})
    assert found["h"] is None and found["reason"] == "unsupported: no bracket"
    why = ["η_u: no completed run in study5-x-fresh-upper-s50001"]
    found = sa.seed_outcome(50001, two_rates(50001, 5.0, 7.0)[:1], why, {})
    assert found["reason"] == (
        "open: missing (η_u: no completed run in study5-x-fresh-upper-s50001)"
    )
    runs = two_rates(50002, 5.0, 7.0)
    found = sa.seed_outcome(
        50002, runs, [], {"lo-s50002": counts(10, 400), "hi-s50002": "labels: no judging receipt"}
    )
    assert found["h"] is None and found["reason"] == "hi-s50002: labels: no judging receipt"
    found = sa.seed_outcome(
        50002, runs, [], {"lo-s50002": counts(10, 400), "hi-s50002": counts(20, 380)}
    )
    w = weight(5.0, 7.0, 6.0)
    assert found["h"] == pytest.approx((1 - w) * 10 / 521 + w * 20 / 521, abs=1e-15)
    assert set(found["classes"]) == {"refusal", "safe_non_refusal", "harmful", "harmful_high"}


# ------------------------------------------------------------------ the line scans


def test_odd_and_even_parts_on_a_synthetic_scan() -> None:
    """x(a) = c + b·a + q·a² per seed: the odd part is 2·b·a, the even part 2·q·a², exactly."""
    seeds = [50000, 50001, 50002]
    coef = {50000: (0.1, 0.02, 0.05), 50001: (0.2, 0.04, 0.01), 50002: (0.15, 0.03, 0.03)}
    per_seed = {}
    for s in seeds:
        c, b, q = coef[s]
        per_seed[s] = {a: {m: c + b * a + q * a * a for m in sa.SCAN_MEASURES}
                       for a in (*sa.SCAN_POINTS, 0.0, 1.0)}  # fmt: skip
    found = sa.line_scans(per_seed, seeds, level=0.95)
    for a in sa.PART_POINTS:
        key = f"{a:+g}"
        for m in sa.SCAN_MEASURES:
            odd, even = found["odd"][key][m], found["even"][key][m]
            assert odd["mean"] == pytest.approx(np.mean([2 * coef[s][1] * a for s in seeds]))
            assert even["mean"] == pytest.approx(np.mean([2 * coef[s][2] * a * a for s in seeds]))
            assert odd["n"] == 3 and odd["level"] == 0.95
            values = [2 * coef[s][1] * a for s in seeds]
            half = stats.t.ppf(0.975, 2) * np.std(values, ddof=1) / math.sqrt(3)
            assert odd["upper"] - odd["mean"] == pytest.approx(half)
    assert set(found["points"]) == {f"{a:+g}" for a in (*sa.SCAN_POINTS, 0.0, 1.0)}


def test_a_scan_point_missing_for_one_seed_excludes_it_there_only() -> None:
    seeds = [50000, 50001, 50002]
    per_seed = {s: {a: {m: 0.1 * a for m in sa.SCAN_MEASURES} for a in (*sa.SCAN_POINTS, 0.0, 1.0)}
                for s in seeds}  # fmt: skip
    per_seed[50001][-0.5] = "line_scan_-0.5: not evaluated (nonfinite_k)"
    found = sa.line_scans(per_seed, seeds, level=0.95)
    assert found["odd"]["+0.5"]["harmful"]["n"] == 2
    assert found["odd"]["+0.5"]["harmful"]["excluded"] == [
        {"seed": 50001, "reason": "a=-0.5: line_scan_-0.5: not evaluated (nonfinite_k)"}
    ]
    assert found["odd"]["+0.25"]["harmful"]["n"] == 3
    assert found["points"]["-0.5"]["harmful"]["n"] == 2


# ------------------------------------------------------------------ controls and ladder


def test_controls_pair_within_seed() -> None:
    seeds = list(range(50000, 50006))
    own = {s: {m: 0.1 * i for m in sa.SCAN_MEASURES} for i, s in enumerate(seeds)}
    order = {s: {m: 0.1 * i + 0.01 * (i + 1) for m in sa.SCAN_MEASURES}
             for i, s in enumerate(seeds)}  # fmt: skip
    v0 = dict(order)
    v0[50003] = "v0: no completed run"
    found = sa.controls(own, {"order": order, "v0": v0}, seeds, level=0.95)
    diffs = [0.01 * (i + 1) for i in range(6)]
    assert found["order"]["harmful"]["n"] == 6
    assert found["order"]["harmful"]["mean"] == pytest.approx(np.mean(diffs))
    assert found["order"]["per_seed"]["harmful"]["50002"] == pytest.approx(0.03)
    assert found["v0"]["harmful"]["n"] == 5
    assert found["v0"]["harmful"]["excluded"] == [
        {"seed": 50003, "reason": "control: v0: no completed run"}
    ]


def ladder_record(rung: str, status: str, c: float | None) -> dict:
    return {"kind": "study5-ladder", "rung": rung, "status": status, "c": c, "c0": 2.0,
            "search": {"scales": [{"c": 1.0, "K_benign": 3.0}, {"c": 2.0, "K_benign": 5.0},
                                  {"c": 4.0, "K_benign": 9.0}],
                       "pair": None if c is None else {"c": [2.0, 4.0], "K_benign": [5.0, 9.0]},
                       "choice": None if c is None else "only", "extensions": 0,
                       "monotone": True},
            "evaluation": None if status != "evaluated" else row(6.0)}  # fmt: skip


def test_the_ladder_table_lists_a_not_bracketed_rung() -> None:
    records = {
        50000: {
            "isotropic": ladder_record("isotropic", "not_bracketed", None),
            "linear_no_d": ladder_record("linear_no_d", "evaluated", 2.6),
            "linear": "no ladder record (ladder/s50000-linear.json)",
        }
    }
    labels = {(50000, "linear_no_d"): counts(52, 300)}
    found = sa.ladder_table(records, labels, {50000: 0.08}, [50000], sa.RUNGS[:3])
    rows = {r["rung"]: r for r in found["rows"]}
    assert rows["linear_no_d"]["classes"]["harmful"] == pytest.approx(52 / 521)
    assert rows["linear_no_d"]["h_s"] == 0.08 and rows["linear_no_d"]["c"] == 2.6
    assert rows["linear_no_d"]["K_benign"] == 6.0
    assert rows["isotropic"]["classes"] is None and rows["isotropic"]["status"] == "not_bracketed"
    assert rows["isotropic"]["scales"][1] == {"c": 2.0, "K_benign": 5.0}
    assert found["not_evaluated"] == [
        {"seed": 50000, "rung": "isotropic", "status": "not_bracketed"}
    ]
    assert found["missing"] == [
        {"seed": 50000, "rung": "linear", "reason": "no ladder record (ladder/s50000-linear.json)"}
    ]


# ------------------------------------------------------------------ the extreme blocks


def cells(n_per_group: int = 20) -> list[dict]:
    return [{"group": g, "run_id": f"{g}-{i}", "seed": 1, "arm": "A", "side": "lo"}
            for g in ("high", "low") for i in range(n_per_group)]  # fmt: skip


def test_the_judges_contrast_is_the_unweighted_mean_over_cells() -> None:
    shares = {f"high-{i}": (100 + i) / 521 for i in range(20)}
    shares |= {f"low-{i}": (10 + i) / 521 for i in range(20)}
    found = sa.judge_contrast(cells(), shares)
    high = sum((100 + i) / 521 for i in range(20)) / 20
    low = sum((10 + i) / 521 for i in range(20)) / 20
    assert found["high"]["mean"] == pytest.approx(high, abs=1e-15)
    assert found["contrast"] == pytest.approx(high - low, abs=1e-15)
    shares["low-3"] = "llama_guard: no judging receipt"
    found = sa.judge_contrast(cells(), shares)
    assert found["low"]["n"] == 19 and found["low"]["excluded"] == [
        {"run_id": "low-3", "reason": "llama_guard: no judging receipt"}
    ]


def test_the_reference_estimator_and_agreement_by_hand() -> None:
    """Two cells per group, five items each: a cell's share is its harmful items over 5, a group's
    the mean of its cells, the contrast high minus low; a missing label is reported in a low and a
    high bound."""
    drawn = [
        {"group": "high", "run_id": "h1"},
        {"group": "high", "run_id": "h2"},
        {"group": "low", "run_id": "l1"},
        {"group": "low", "run_id": "l2"},
    ]
    key = {f"c{c}{j}": {"cell": c} for c in range(4) for j in range(5)}
    harmful = {0: 3, 1: 1, 2: 1, 3: 0}
    labels = {}
    for c in range(4):
        for j in range(5):
            labels[f"c{c}{j}"] = "harmful_compliance" if j < harmful[c] else "refusal"
    labels["c34"] = None
    found = sa.reference_contrast(drawn, key, labels)
    assert found["high"]["low"] == pytest.approx((3 / 5 + 1 / 5) / 2)
    assert found["low"]["low"] == pytest.approx((1 / 5 + 0 / 5) / 2)
    assert found["low"]["high"] == pytest.approx((1 / 5 + 1 / 5) / 2)
    assert found["contrast"]["low"] == pytest.approx(0.4 - 0.1)
    assert found["missing"] == 1
    other = dict(labels)
    other["c00"] = "safe_non_refusal"
    agree = sa.agreement(labels, other)
    assert agree == {"items": 20, "both_labeled": 19, "agree": 18, "share": 18 / 19,
                     "failed": 1}  # fmt: skip
    with pytest.raises(ValueError, match="not a label"):
        sa.reference_contrast(drawn, key, {**labels, "c00": "HARMFUL"})


# ------------------------------------------------------------------ the CLI, end to end

from zero_field_refusal.analysis import study5_analysis_cli as cli  # noqa: E402
from zero_field_refusal.analysis.study3_labels import RunHarm  # noqa: E402
from zero_field_refusal.config import load_config  # noqa: E402
from zero_field_refusal.evaluation import line_scan_condition  # noqa: E402
from zero_field_refusal.judge import _content_hash, judge_identity  # noqa: E402

SENTINEL = "TEXT-NEVER-STORED"
IDS = [f"judged:{i}" for i in range(521)]
PROMPTS = {pid: f"Prompt {SENTINEL} {i}" for i, pid in enumerate(IDS)}
SET = "harmful_evaluation"
POD = "/pod/runs"
REAL_OWN_CODE = cli.own_code
CODE = {"commit": "a" * 40, "dirty": False, "module": "zero_field_refusal.analysis.study5_analysis"}
GPU = "e" * 64  # pod 2's plan's GPU code identity
RERUN = "b" * 40  # the engineering reruns' commit, of GPU code GPU
SPEC = "5" * 64  # the surrogate's sha256
STREAM = {"path": "teacher/x/training/y/seed-{seed}", "count": 500,
          "chunks": [f"{i:064x}" for i in range(1)]}  # fmt: skip


def sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=1, sort_keys=True) + "\n").encode()
    path.write_bytes(data)
    return data


def lines(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


class World:
    """A synthetic study-5 world: both Pods' plans and launcher campaigns, the runs with their
    records, responses (each holding ``SENTINEL``) and WildGuard labels, and the scripts' records;
    a fixation record of the real one's settings with 4 fresh seeds, controls and ladder on
    50000."""

    seeds = [50000, 50001, 50002, 50003]

    def __init__(self, tmp: Path) -> None:
        self.tmp, self.runs = tmp, tmp / "runs"
        self.study3 = tmp / "study3"  # study 3's runs root and checkout: apart from study 5's
        self.study3.mkdir()
        self.identity = judge_identity(load_config("judge/wildguard"))
        self.fixation = dict(FIXATION)
        self.fixation["seeds"] = {**FIXATION["seeds"],
                                  "fresh": {"n": 4, "first": 50000, "last": 50003,
                                            "seeds": list(self.seeds)},
                                  "alternative": {"n": 2, "seeds": [50000, 50001]},
                                  "controls": [50000], "ladder": [50000]}  # fmt: skip
        self.fixation["permutation_test"] = small_test(2000, minimum=3)
        self.fixation_path = tmp / "fixation.json"
        self.steps: dict[str, list] = {"pod1": [], "pod2": []}
        self.campaigns: dict[str, dict[str, list]] = {"pod1": {}, "pod2": {}}
        self.harmful: dict[str, int] = {}
        # θ₀, the fresh runs, the controls and the ladder
        self.theta0 = self.run("pod1", "theta0", "experiments/study5_theta0", 50000, None, [],
                               {"run_type": "theta0"}, ["theta0"], harmful=5)  # fmt: skip
        self.k = {50000: (5.0, 7.0), 50001: (5.5, 6.4), 50002: (6.5, 7.0), 50003: (3.5, 6.9)}
        self.h = {50000: (40, 80), 50001: (20, 30), 50002: (60, 90), 50003: (10, 70)}
        self.lower, self.upper, self.scan = {}, {}, {}
        for seed in self.seeds:
            self.lower[seed] = self.run(
                "pod2", f"fresh-lower-s{seed}", "experiments/study5", seed, 1.3e-05,
                ["study5.role=fresh", f"optimizer.lr={1.3e-05!r}"], {"schedule": {"steps": 25}},
                ["step-25"], harmful=self.h[seed][0], k=self.k[seed][0])  # fmt: skip
            self.scan[seed] = self.scan_run(seed, harmful=self.h[seed][0] + 1,
                                            k=self.k[seed][0])  # fmt: skip
            self.upper[seed] = self.run(
                "pod2", f"fresh-upper-s{seed}", "experiments/study5", seed, 1.83e-05,
                ["study5.role=fresh", f"optimizer.lr={1.83e-05!r}"], {"schedule": {"steps": 25}},
                ["step-25"], harmful=self.h[seed][1], k=self.k[seed][1])  # fmt: skip
        for kind in ("order", "v0"):
            self.run("pod2", f"controls-{kind}-s50000", f"experiments/study5_control_{kind}",
                     50000, 1.3e-05, [f"optimizer.lr={1.3e-05!r}"], {"schedule": {"steps": 25}},
                     ["step-25"], harmful=50 if kind == "order" else 45, k=5.2)  # fmt: skip
        self.ladder()
        self.records()

    # -------------------------------------------------------------- runs

    def run(self, pod, campaign, config, seed, lr, overrides, extra, conditions, *, harmful,
            k=0.0) -> Path:  # fmt: skip
        name = f"study5-{pod}-x-{campaign}"
        cell = {"name": f"s5-{campaign}", "config": config, "overrides": overrides,
                "seeds": [seed]}  # fmt: skip
        self.campaigns[pod].setdefault(name, []).append(cell)
        run_id = f"r-{campaign}-{len(self.harmful)}"
        experiment = "study5"
        label = f"{cell['name']}/s{seed}"
        hour = 1 + len(self.harmful) % 20
        lines(self.runs / "launch" / f"{name}.jsonl",
              [{"event": "start", "run": label, "run_id": run_id, "experiment": experiment,
                "time": f"2026-11-10T{hour:02d}:00:00+00:00"},
               {"event": "end", "run": label, "run_id": run_id, "outcome": "done",
                "time": f"2026-11-10T{hour:02d}:30:00+00:00"}])  # fmt: skip
        run_dir = self.runs / experiment / run_id
        cfg = {"run_type": "train", "seed": seed, "eval": {"harmful": {"prompt_set": SET}},
               "optimizer": {"name": "adam_frozen", "lr": lr}, **extra}  # fmt: skip
        self.write_run(run_dir, cfg, conditions, harmful=harmful, k=k)
        return run_dir

    def scan_run(self, seed: int, *, harmful: int, k: float) -> Path:
        """The η_l run's scan run, as scripts/study5_scan.py writes it, and its record."""
        run_id = f"s5-pod2-x-scan-s{seed}"
        out = f"study5_records/pod2-x/scan/s{seed}.json"
        self.steps["pod2"].append({"name": f"fresh/s{seed}/scan", "outputs": [f"{POD}/{out}"]})
        scan = [line_scan_condition(a) for a in sa.SCAN_POINTS]
        dump(self.runs / out, {"kind": "study5-scan", "seed": seed, "experiment": "study5_scan",
                               "run_id": run_id, "f_run": {"run_id": self.lower[seed].name},
                               "evaluated": scan})  # fmt: skip
        run_dir = self.runs / "study5_scan" / run_id
        cfg = {"run_type": "study5_scan", "seed": seed, "eval": {"harmful": {"prompt_set": SET}}}
        self.write_run(run_dir, cfg, scan, harmful=harmful, k=k)
        return run_dir

    def metrics(self, k: float) -> dict:
        return {
            "K_benign": k,
            "K_harmful": 2 * k,
            "benchmark_accuracy": 0.6 - 0.001 * k,
            "factual_accuracy": 0.5 - 0.001 * k,
            "heldout_nll_per_token": 1.0 + 0.01 * k,
        }

    def write_run(self, run_dir: Path, cfg: dict, conditions, *, harmful: int, k: float) -> None:
        (run_dir).mkdir(parents=True)
        (run_dir / "config.yaml").write_text(json.dumps(cfg))  # JSON is YAML
        seed = cfg["seed"]
        dump(
            run_dir / "meta.json",
            {
                "run_id": run_dir.name,
                "hardware": {"gpus": ["NVIDIA L40S"]},
                "inputs": {
                    "harmful_prompts": {
                        "set": SET,
                        "count": 521,
                        "ids_sha256": sha(json.dumps(IDS).encode()),
                    },
                    "v0_sha256": f"v0-{seed}",
                    "theta0_trainable_sha256": "t" * 64,
                    "streams": {"training": {**STREAM, "path": STREAM["path"].format(seed=seed)}},
                },
            },
        )
        self.harmful[run_dir.name] = harmful
        if cfg["run_type"] == "theta0":
            dump(
                run_dir / "summary.json",
                self.metrics(0.0) | {"benchmark_accuracy": 0.6, "factual_accuracy": 0.5},
            )
        elif cfg["run_type"] == "train":
            dump(run_dir / "terminal.json", {"steps": 25, "stop": {"reason": "steps"},
                                             "evaluations": [25]})  # fmt: skip
            lines(
                run_dir / "evaluations.jsonl",
                [{"step": 25, "condition": "step-25", **self.metrics(k)}],
            )
        elif cfg["run_type"] == "study5_scan":
            scan = list(conditions)
            if scan:
                rows = [
                    {
                        "step": 25,
                        "a": a,
                        "condition": line_scan_condition(a),
                        "evaluated": True,
                        **self.metrics(k * a * a),
                    }
                    for a in sa.SCAN_POINTS
                ]
                lines(run_dir / "line_scan.jsonl", rows)
                dump(
                    run_dir / "summary.json",
                    {
                        "line_scan": {
                            "points": list(sa.SCAN_POINTS),
                            "conditions": scan,
                            "evaluated": scan,
                        }
                    },
                )
        responses, labels = [], []
        for c_index, condition in enumerate(conditions):
            for i, pid in enumerate(IDS):
                text = f"Reply {SENTINEL} {condition} {i}"
                responses.append({"condition": condition, "item_id": pid, "text": text})
                n = harmful + c_index  # each condition one harmful more than the one before
                label = ("harmful_compliance" if i < n else "refusal" if i < n + 300
                         else "safe_non_refusal")  # fmt: skip
                labels.append({"item_id": f"{condition}/{pid}", **self.identity,
                               "content_sha256": _content_hash(PROMPTS[pid], text),
                               "label": label,
                               "created_at": "2026-11-11T12:00:00+00:00"})  # fmt: skip
        lines(run_dir / "responses" / f"{SET}.jsonl", responses)
        lines(run_dir / "judge" / "wildguard.jsonl", labels)
        expected = [f"{c}/{pid}" for c in conditions for pid in IDS]
        dump(
            run_dir / "judge" / "wildguard.receipt.json",
            {
                "judge": "wildguard",
                "identity": self.identity,
                "expected_items": expected,
                "items": len(expected),
                "rounds": 1,
                "missing": 0,
                "labels_sha256": sha((run_dir / "judge" / "wildguard.jsonl").read_bytes()),
                "completed_at": "2026-11-12T00:00:00+00:00",
            },
        )

    def ladder(self) -> None:
        f_run = self.lower[50000].name
        rungs = (
            ("isotropic", "invalid_k", 0),
            ("linear_no_d", "evaluated", 33),
            ("linear", "evaluated", 50),
            ("linear_shuffled_d", "evaluated", 40),
        )
        for rung, status, harmful in rungs:
            run_id = f"s5-pod2-x-ladder-{rung}-s50000"
            out = f"study5_records/pod2-x/ladder/s50000-{rung}.json"
            self.steps["pod2"].append({"name": f"controls/s50000/ladder-{rung}",
                                       "outputs": [f"{POD}/{out}"]})  # fmt: skip
            record = ladder_record(rung, status, 2.6 if status == "evaluated" else None)
            if status == "invalid_k":  # the rung's script writes the non-finite K it measured
                record["search"]["scales"] = [
                    {"c": 1.0, "K_benign": 3.0},
                    {"c": 2.0, "K_benign": math.nan},
                    {"c": math.inf, "K_benign": -math.inf},
                ]
                record["c0"] = math.inf
            dump(
                self.runs / out,
                {
                    **record,
                    "seed": 50000,
                    "experiment": "study5_ladder",
                    "run_id": run_id,
                    "f_run": {"run_id": f_run},
                },
            )
            run_dir = self.runs / "study5_ladder" / run_id
            cfg = {"run_type": "study5_ladder", "seed": 50000,
                   "eval": {"harmful": {"prompt_set": SET}}}  # fmt: skip
            if status == "evaluated":
                self.write_run(run_dir, cfg, [f"ladder_{rung}"], harmful=harmful, k=6.0)
                dump(
                    run_dir / "summary.json",
                    {"ladder": {"rung": rung, "condition": f"ladder_{rung}", "evaluated": True}},
                )

    def records(self) -> None:
        base = "study5_records/pod2-x"
        rng = np.random.default_rng(0)
        self.p = {s: float(v) for s, v in zip(self.seeds, rng.normal(size=4), strict=True)}
        self.predict_sha = {}
        for seed in self.seeds:
            rel = f"{base}/predictions/fresh/s{seed}.json"
            self.steps["pod2"].append({"name": f"streams/s{seed}/predict",
                                       "outputs": [f"{POD}/{rel}"]})  # fmt: skip
            self.predict_sha[seed] = sha(dump(self.runs / rel, {
                "kind": "study5-prediction", "seed": seed, "lr": 1.3e-05, "p": self.p[seed],
                "v0_sha256": f"v0-{seed}", "theta0_trainable_sha256": "t" * 64,
                "streams": {"training": {**STREAM, "path": STREAM["path"].format(seed=seed)}},
                "surrogate_sha256": SPEC}))  # fmt: skip
        for seed in self.seeds:
            for step, rel, value in (
                (f"streams/s{seed}/features", f"{base}/features/fresh/s{seed}.json",
                 {"kind": "study5-stream-features", "seed": seed,
                  "features": {"phrase_share": 0.1 * seed % 7, "mean_tokens": 50.0 + seed % 5,
                               "ended_share": 0.5}}),
                (f"fresh/s{seed}/surrogate-endpoint", f"{base}/surrogate-endpoint/s{seed}.json",
                 {"kind": "study5-surrogate-endpoint", "seed": seed,
                  "run_id": self.lower[seed].name, "r": 1.0 + seed % 3,
                  "surrogate_sha256": SPEC}),
            ):  # fmt: skip
                self.steps["pod2"].append({"name": step, "outputs": [f"{POD}/{rel}"]})
                dump(self.runs / rel, value)
        record = {
            "kind": "study5-predictions",
            "n": 4,
            "seeds": self.seeds,
            "lr": 1.3e-05,
            "predictions": [
                {"seed": s, "p": self.p[s], "sha256": self.predict_sha[s]} for s in self.seeds
            ],
            "code": {"commit": "c" * 40, "gpu_code_sha256": GPU},
        }
        path = self.runs / base / "predictions-record.json"
        data = dump(path, record)
        path.with_name(path.name + ".sha256").write_text(f"{sha(data)}  {path.name}\n")
        receipt = self.runs / base / "predictions-receipt.json"
        dump(
            receipt,
            {
                "kind": "study5-predictions-receipt",
                "sha256": sha(data),
                "created": "2026-11-10T00:00:00Z",
            },
        )
        self.steps["pod2"] += [
            {"name": "predictions/record", "outputs": [f"{POD}/{base}/predictions-record.json"]},
            {"name": "predictions/publish",
             "outputs": [f"{POD}/{base}/predictions-receipt.json"]}]  # fmt: skip
        # pod 1's gates
        for step, rel, value in (
            ("engineering/check", "study5_records/pod1-x/engineering-check.json",
             {"kind": "study5-engineering-check", "equal": True,
              "pairs": [{"rerun": "x", "rerun_commit": RERUN, "rerun_dirty": False}],
              "code": {"commit": RERUN, "gpu_code_sha256": GPU}}),
            ("surrogate/checks", "study5_records/pod1-x/surrogate-checks.json",
             {"kind": "study5-surrogate-checks", "passes": True, "surrogate_sha256": SPEC,
              "git": {"commit": RERUN, "dirty": False}}),
        ):  # fmt: skip
            self.steps["pod1"].append({"name": step, "outputs": [f"{POD}/{rel}"]})
            dump(self.runs / rel, value)

    def plans(self) -> None:
        for pod in ("pod1", "pod2"):
            folder = self.tmp / pod
            digests = {}
            for name, cells in self.campaigns[pod].items():
                text = json.dumps({"name": name, "cells": cells})
                (folder / "campaigns").mkdir(parents=True, exist_ok=True)
                (folder / "campaigns" / f"{name}.yaml").write_text(text)
                digests[name] = sha(text.encode())
            dump(folder / "plan.json", {"pod": pod, "archive_id": f"{pod}-x", "runs": POD,
                                        "n": 4 if pod == "pod2" else None, "gpu_code": GPU,
                                        "surrogate_sha256": SPEC,
                                        "blocks": [{"name": "all", "steps": self.steps[pod]}],
                                        "campaigns": digests})  # fmt: skip
        dump(self.fixation_path, self.fixation)

    def argv(self, out: Path, *extra) -> list[str]:
        self.plans()
        return [str(a) for a in ("full", "--runs-root", self.runs, "--pod1", self.tmp / "pod1",
                                 "--pod2", self.tmp / "pod2", "--fixation", self.fixation_path,
                                 "--out", out, *extra)]  # fmt: skip

    def retro_argv(self, out: Path, *extra) -> list[str]:
        """The retrospective alone: no pod2 input at all."""
        self.plans()
        return [str(a) for a in ("retrospective", "--runs-root", self.runs, "--pod1",
                                 self.tmp / "pod1", "--fixation", self.fixation_path,
                                 "--out", out, *extra)]  # fmt: skip


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    monkeypatch.setattr(cli, "harmful_prompts", lambda cfg: PROMPTS)
    monkeypatch.setattr(cli, "own_code", lambda: dict(CODE))
    monkeypatch.setattr(cli, "identity_of", lambda commit: GPU if commit == RERUN else None)
    found = World(tmp_path)
    # the registration at the analysis's commit names the fixation record it reads
    monkeypatch.setattr(cli, "registration_text", lambda: (
        "Status: `registered`.\n" + sha(found.fixation_path.read_bytes())))  # fmt: skip
    return found


def call(argv) -> tuple[int, str]:
    try:
        return cli.main(argv), ""
    except SystemExit as exc:
        return 1, str(exc.code)


def test_the_cli_writes_the_analysis_once_numbers_only(world, capsys) -> None:
    out = world.tmp / "analysis" / "analysis.json"
    code, why = call(world.argv(out, "--without-study3"))
    assert code == 0, why
    printed = capsys.readouterr().out
    data = out.read_bytes()
    result = json.loads(data)
    assert SENTINEL not in data.decode() and SENTINEL not in printed and "Prompt" not in printed
    assert str(world.tmp) not in data.decode()
    blob = json.dumps(list(PROMPTS.items()), ensure_ascii=False).encode()
    assert result["prompt_sets"] == {SET: sha(blob)}
    assert sha(data) in printed
    assert result["seeds"]["fresh"] == world.seeds
    # the outcome: study 3's interpolation of the two rates' step-25 harm
    by_seed = {o["seed"]: o for o in result["outcomes"]}
    lo, hi = world.k[50000]
    w = weight(lo, hi, 6.0)
    assert by_seed[50000]["h"] == pytest.approx((1 - w) * 40 / 521 + w * 80 / 521, abs=1e-15)
    assert by_seed[50000]["p"] == world.p[50000]
    assert by_seed[50002]["h"] is None  # K 6.5–7.0 does not bracket K*
    assert result["unsupported"] == [
        {"seed": 50002, "reason": "unsupported: no bracket"},
        {"seed": 50003, "reason": "unsupported: too wide"},
    ]
    test = result["confirmatory"]
    assert test["n"] == 2 and test["descriptive"] is True and test["p_value"] is None
    assert {e["seed"] for e in test["excluded"]} == {50002, 50003}
    # the line scan: θ₀ at 0, the endpoint at +1, the scan conditions between
    scan = result["descriptive"]["line_scans"]["per_seed"]["50000"]
    assert scan["+0"]["harmful"] == pytest.approx(5 / 521)
    assert scan["+1"]["harmful"] == pytest.approx(40 / 521)
    assert scan["-1"]["harmful"] == pytest.approx(41 / 521)  # the first scan condition
    assert scan["-1"]["K_benign"] == pytest.approx(5.0)
    odd = result["descriptive"]["line_scans"]["odd"]["+1"]["harmful"]
    assert odd["n"] == 4 and odd["mean"] == pytest.approx(-1 / 521)
    controls = result["descriptive"]["controls"]
    assert controls["order"]["per_seed"]["harmful"]["50000"] == pytest.approx(10 / 521)
    assert test["excluded_by_state"] == {"binding": 0, "unsupported": 2, "open": 0,
                                         "unresolved": 0, "labels_missing": 0}  # fmt: skip
    # the ladder: every rung the plan has, an invalid_k rung's NaN and ±inf kept as missing
    ladder = result["descriptive"]["ladder"]
    assert ladder["not_evaluated"] == [{"seed": 50000, "rung": "isotropic", "status": "invalid_k"}]
    rows = {r["rung"]: r for r in ladder["rows"]}
    assert list(rows) == ["isotropic", "linear_no_d", "linear", "linear_shuffled_d"]
    nonfinite = {"missing": "nonfinite"}
    assert rows["isotropic"]["scales"] == [{"c": 1.0, "K_benign": 3.0},
                                           {"c": 2.0, "K_benign": nonfinite},
                                           {"c": nonfinite, "K_benign": nonfinite}]  # fmt: skip
    assert rows["isotropic"]["c0"] == nonfinite
    assert rows["linear"]["classes"]["harmful"] == pytest.approx(50 / 521)
    assert rows["linear_shuffled_d"]["classes"]["harmful"] == pytest.approx(40 / 521)
    assert rows["linear"]["h_s"] == by_seed[50000]["h"]
    trigger = ladder["shuffled_d_trigger"]
    assert trigger["per_seed"] == [{"seed": 50000, "difference": pytest.approx(17 / 521),
                                    "exceeds_2pp": True}]  # fmt: skip
    assert trigger["mean"]["mean"] == pytest.approx(17 / 521) and trigger["mean_exceeds_2pp"]
    assert trigger["trigger"] is True and "mean over the six seeds" in trigger["rule"]
    assert trigger["seeds_in_mean"] == 1 and trigger["all_seeds"] is True
    # the bucket copy's creation time precedes the first run start of pod2's journals
    assert result["predictions"]["created"] == "2026-11-10T00:00:00Z"
    assert result["predictions"]["first_start"].startswith("2026-11-10T")
    assert result["predictions"]["created_before_first_start"] is True
    assert f"runs:study5_scan/{world.scan[50000].name}/summary.json" in result["inputs"]
    assert f"runs:study5_scan/{world.scan[50000].name}/line_scan.jsonl" in result["inputs"]
    theta0 = result["descriptive"]["theta0"]["classes"]
    assert theta0["harmful"] == pytest.approx(5 / 521) and theta0["refusal"] == pytest.approx(
        300 / 521)  # fmt: skip
    assert result["retrospective"] == {"computed": False, "reason": "--without-study3"}
    assert result["descriptive"]["surrogate_endpoint"]["n"] == 4
    run = world.lower[50000].name
    assert f"runs:study5/{run}/judge/wildguard.jsonl" in result["inputs"]
    assert result["inputs"]["runs:study5_records/pod2-x/predictions-record.json"] == sha(
        (world.runs / "study5_records/pod2-x/predictions-record.json").read_bytes()
    )
    # never overwritten
    code, why = call(world.argv(out, "--without-study3"))
    assert code == 1 and "never overwritten" in why and out.read_bytes() == data


def test_a_missing_label_file_is_reported_by_name_and_leaves_that_seed_only(world) -> None:
    upper = world.upper[50001]
    (upper / "judge" / "wildguard.receipt.json").unlink()
    (world.runs / "study5_records/pod2-x/features/fresh/s50003.json").unlink()
    out = world.tmp / "analysis.json"
    code, why = call(world.argv(out, "--without-study3"))
    assert code == 0, why
    result = json.loads(out.read_text())
    assert str(world.tmp) not in out.read_text()  # no local path in a reason
    by_seed = {o["seed"]: o for o in result["outcomes"]}
    assert by_seed[50001]["h"] is None
    assert "<runs>/study5/" in by_seed[50001]["reason"]
    assert by_seed[50001]["reason"].startswith(f"{upper.name}: labels: LabelsPending")
    assert by_seed[50000]["h"] is not None
    features = result["descriptive"]["stream_features"]["mean_tokens"]
    assert features["n"] == 1 and {e["seed"] for e in features["excluded"]} == {
        50001, 50002, 50003}  # fmt: skip
    assert [e for e in features["excluded"] if e["seed"] == 50003][0]["reason"] == (
        "mean_tokens: features: no record (s50003.json); h: unsupported: too wide"
    )
    # the line scan of 50001's η_l run is unaffected by its η_u labels
    assert result["descriptive"]["line_scans"]["odd"]["+0.5"]["harmful"]["n"] == 4


def test_a_missing_scan_is_missing_by_name_and_the_seed_stays_elsewhere(world) -> None:
    """Seed 50001's scan record is gone, seed 50002's scan run was cut off (its summary never
    written): their points are missing by name, and both seeds stay in every quantity that does
    not need those points (the outcome, the endpoint and θ₀'s points)."""
    (world.runs / "study5_records/pod2-x/scan/s50001.json").unlink()
    (world.scan[50002] / "summary.json").unlink()
    out = world.tmp / "analysis.json"
    code, why = call(world.argv(out, "--without-study3"))
    assert code == 0, why
    result = json.loads(out.read_text())
    scans = result["descriptive"]["line_scans"]
    assert scans["per_seed"]["50001"]["-1"]["harmful"] == {
        "missing": "a=-1: line_scan_-1: scan: no record (s50001.json)"}  # fmt: skip
    assert scans["per_seed"]["50002"]["+0.5"]["harmful"]["missing"].startswith(
        "a=+0.5: labels: ValueError")  # fmt: skip
    assert scans["per_seed"]["50001"]["+1"]["harmful"] == pytest.approx(20 / 521)
    assert scans["points"]["+1"]["harmful"]["n"] == 4 and scans["points"]["+0"]["harmful"]["n"] == 4
    assert scans["odd"]["+1"]["harmful"]["n"] == 2  # a = −1 missing for 50001 and 50002
    assert {e["seed"] for e in scans["odd"]["+1"]["harmful"]["excluded"]} == {50001, 50002}
    by_seed = {o["seed"]: o for o in result["outcomes"]}
    assert by_seed[50001]["h"] is not None  # the outcome needs no scan point


def edit_json(path: Path, **changes) -> None:
    data = json.loads(path.read_text())
    data.update(changes)
    dump(path, data)


def test_pod1s_gates_are_reported_and_bind_the_analysis(world) -> None:
    out = world.tmp / "analysis.json"
    code, why = call(world.argv(out, "--without-study3"))
    assert code == 0, why
    result = json.loads(out.read_text())
    gates = result["gates"]
    assert gates["engineering"]["equal"] and gates["engineering"]["gpu_code_sha256"] == GPU
    assert gates["engineering"]["rerun_commits"] == [RERUN]
    checks = gates["surrogate_checks"]
    assert checks["passes"] and checks["surrogate_sha256"] == SPEC
    assert result["registered"]["registered"] is True
    assert result["registered"]["fixation_sha256"] == result["fixation"]["sha256"]
    assert result["hardware"]["models"] == ["NVIDIA L40S"]
    assert result["hardware"]["runs"][world.lower[50000].name] == ["NVIDIA L40S"]
    assert result["hardware"]["runs"][world.scan[50000].name] == ["NVIDIA L40S"]
    assert result["labels_order"]["after"] is True and result["labels_order"]["earlier"] == []
    assert result["confirmatory"]["order_violated"] is False
    assert "reading_withheld" not in result["confirmatory"]


ENG = "study5_records/pod1-x/engineering-check.json"
CHECKS = "study5_records/pod1-x/surrogate-checks.json"


@pytest.mark.parametrize(
    "rel, changes, why",
    [
        (ENG, {"equal": False}, "not equal"),
        (ENG, {"pairs": []}, "not equal"),
        (ENG, {"code": {"commit": RERUN, "gpu_code_sha256": "f" * 64}}, "GPU code identity"),
        (ENG, {"pairs": [{"rerun": "x", "rerun_commit": "d" * 40, "rerun_dirty": False}]},
         f"commit {'d' * 40} not in this checkout's history"),
        (ENG, {"pairs": [{"rerun": "x", "rerun_commit": RERUN, "rerun_dirty": True}]},
         "engineering rerun"),
        (CHECKS, {"passes": False}, "did not pass"),
        ("study5_records/pod2-x/predictions/fresh/s50001.json", {}, None),  # see below
        ("study5_records/pod2-x/surrogate-endpoint/s50002.json", {"surrogate_sha256": "0" * 64},
         "another surrogate"),
        ("study5_records/pod1-x/surrogate-checks.json", {"surrogate_sha256": "0" * 64},
         "another surrogate"),
    ],
)  # fmt: skip
def test_the_analysis_refuses_gates_that_do_not_stand(world, rel, changes, why) -> None:
    if why is None:  # a predictor of another surrogate (and its predictions entry, so bound)
        path = world.runs / rel
        edit_json(path, surrogate_sha256="0" * 64)
        record = world.runs / "study5_records/pod2-x/predictions-record.json"
        data = json.loads(record.read_text())
        data["predictions"][1]["sha256"] = sha(path.read_bytes())
        written = dump(record, data)
        record.with_name(record.name + ".sha256").write_text(f"{sha(written)}  {record.name}\n")
        edit_json(record.with_name("predictions-receipt.json"), sha256=sha(written))
        why = "another surrogate"
    else:
        edit_json(world.runs / rel, **changes)
    out = world.tmp / "analysis.json"
    code, message = call(world.argv(out, "--without-study3"))
    assert code == 1 and why in message and not out.exists(), message


def test_the_analysis_refuses_a_gate_record_missing(world) -> None:
    (world.runs / ENG).unlink()
    out = world.tmp / "analysis.json"
    code, message = call(world.argv(out, "--without-study3"))
    assert code == 1 and "engineering/check record is missing" in message and not out.exists()


def test_the_analysis_refuses_predictions_of_another_gpu_code_than_pod2s(world) -> None:
    out = world.tmp / "analysis.json"
    argv = world.argv(out, "--without-study3")
    edit_json(world.tmp / "pod2" / "plan.json", gpu_code="f" * 64)
    code, message = call(argv)
    assert code == 1 and "not pod 2's plan's" in message and not out.exists()


def test_the_analysis_reads_the_fixation_record_the_registration_names(world, monkeypatch) -> None:
    monkeypatch.setattr(cli, "registration_text", lambda: "Status: `registered`.\n" + "0" * 64)
    out = world.tmp / "analysis.json"
    code, message = call(world.argv(out, "--without-study3"))
    assert code == 1 and "not the one the registration names" in message and not out.exists()


@pytest.mark.parametrize(
    "change, bound",
    [
        ({"v0_sha256": "other"}, False),
        ({"theta0_trainable_sha256": "other"}, False),
        ({"streams": {"training": {**STREAM, "path": "teacher/x/training/y/seed-50000",
                                   "chunks": ["f" * 64]}}}, False),
        # a run stopped early read a prefix of the prediction's stream: still bound
        ({"streams": {"training": {**STREAM, "path": "teacher/x/training/y/seed-50000",
                                   "count": 100}}}, True),
        ({"record": "other"}, False),  # the predict record is not the one the entry names
    ],
)  # fmt: skip
def test_p_is_bound_to_the_eta_l_run_it_is_paired_with(world, change, bound) -> None:
    run = world.lower[50000]
    if "record" in change:
        edit_json(world.runs / "study5_records/pod2-x/predictions/fresh/s50000.json", p=0.5)
    else:
        meta = json.loads((run / "meta.json").read_text())
        meta["inputs"].update(change)
        dump(run / "meta.json", meta)
    out = world.tmp / "analysis.json"
    code, why = call(world.argv(out, "--without-study3"))
    assert code == 0, why
    result = json.loads(out.read_text())
    by_seed = {o["seed"]: o for o in result["outcomes"]}
    counts = result["confirmatory"]["excluded_by_state"]
    if bound:
        assert by_seed[50000]["h"] is not None and by_seed[50000]["binding"] is None
        assert counts["binding"] == 0
    else:
        assert by_seed[50000]["h"] is None and by_seed[50000]["reason"].startswith("binding: ")
        assert counts["binding"] == 1
        assert 50000 in {e["seed"] for e in result["confirmatory"]["excluded"]}


def test_the_order_violated_gives_no_reading(world) -> None:
    edit_json(
        world.runs / "study5_records/pod2-x/predictions-receipt.json",
        created="2026-11-11T00:00:00Z",
    )  # after the first fresh start
    # 50001's η_u labeling started before the last fresh end and finished after it
    run = world.upper[50001]
    path = run / "judge" / "wildguard.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[7]["created_at"] = "2026-11-10T02:00:00+00:00"
    lines(path, rows)
    edit_json(run / "judge" / "wildguard.receipt.json", labels_sha256=sha(path.read_bytes()))
    out = world.tmp / "analysis.json"
    code, why = call(world.argv(out, "--without-study3"))
    assert code == 0, why
    result = json.loads(out.read_text())
    test = result["confirmatory"]
    assert test["order_violated"] is True and test["labels_before_runs_ended"] is True
    assert test["reading"] is None and len(test["reading_withheld"]) == 2
    order = result["labels_order"]
    assert order["earlier"] == [run.name] and order["after"] is False
    assert order["labeling"][run.name] == {"started_at": "2026-11-10T02:00:00+00:00",
                                           "completed_at": "2026-11-12T00:00:00+00:00"}  # fmt: skip
    assert order["last_fresh_end"] > order["labeling"][run.name]["started_at"]
    assert order["labeling"][run.name]["completed_at"] > order["last_fresh_end"]


def test_the_runs_roots_of_study5_and_study3_do_not_overlap(world) -> None:
    out = world.tmp / "analysis.json"
    for root in (world.runs, world.tmp, world.runs / "study5"):
        code, why = call(world.argv(out, "--study3-runs-root", root, "--study3-repo",
                                    world.study3, "--retrospective", out))  # fmt: skip
        assert code == 1 and "overlap" in why, root


def test_a_predictions_record_that_is_not_the_published_one_is_refused(world) -> None:
    path = world.runs / "study5_records/pod2-x/predictions-record.json"
    path.write_text(path.read_text().replace('"n": 4', '"n": 4 '))
    out = world.tmp / "analysis.json"
    code, why = call(world.argv(out, "--without-study3"))
    assert code == 1 and "not the published one" in why and not out.exists()


def test_a_campaign_that_is_not_the_plans_is_refused(world) -> None:
    argv = world.argv(world.tmp / "analysis.json", "--without-study3")
    campaign = next((world.tmp / "pod2" / "campaigns").iterdir())
    campaign.write_text(campaign.read_text() + " ")
    code, why = call(argv)
    assert code == 1 and "not the plan's" in why


def study3_world(world: World) -> dict:
    """Study 3's parts as the gated reading gives them, for six synthetic blocks, and pod1's
    retrospective records of those seeds; the fixation copy's extreme cells over their runs."""
    blocks, harm, three, llama = [], {}, {}, {}
    rng = np.random.default_rng(1)
    for i in range(6):
        seed = 10400 + i
        arms = {}
        for arm in ("A", "F"):
            entry = {"w": 0.25, "width": 0.4}
            for j, side in enumerate(("lo", "hi")):
                run = f"s3-{arm}-{side}-s{seed}"
                entry[side] = run
                h = int(20 * i + 10 * j)
                harm[run] = RunHarm(harmful=h, missing=1, receipt_sha256="f" * 64)
                three[run] = {
                    "n": 521,
                    "harmful": h,
                    "refusal": 400 - h,
                    "missing": 1,
                    "safe_non_refusal": 120,
                }
                llama[run] = (h + 5) / 521
            arms[arm] = entry
        blocks.append({"seed": seed, "arms": arms})
        for kind, rel, value in (
            ("predict", f"study5_records/pod1-x/predictions/retro/s{seed}.json",
             {"kind": "study5-prediction", "seed": seed, "lr": 1.3e-05,
              "p": float(-i + rng.normal(scale=0.1)), "surrogate_sha256": SPEC}),
            ("features", f"study5_records/pod1-x/features/retro/s{seed}.json",
             {"kind": "study5-stream-features", "seed": seed,
              "features": {"phrase_share": 0.1 * i, "mean_tokens": 40.0 + i,
                           "ended_share": 0.5}}),
        ):  # fmt: skip
            world.steps["pod1"].append({"name": f"retrospective/s{seed}/{kind}",
                                        "outputs": [f"{POD}/{rel}"]})  # fmt: skip
            dump(world.runs / rel, value)
    cells_ = [{"group": "high" if b["seed"] >= 10403 else "low", "run_id": b["arms"][a][s],
               "seed": b["seed"], "arm": a, "side": s}
              for b in blocks for a in ("A", "F") for s in ("lo", "hi")]  # fmt: skip
    world.fixation["reference_sample"] = {**FIXATION["reference_sample"], "cells_drawn": cells_}
    return {
        "blocks": blocks,
        "harm": harm,
        "three": three,
        "llama": llama,
        "freeze_sha256": "0" * 64,
        "checkout": {"commit": "c" * 40, "gpu_code_sha256": "g" * 64},
    }


def test_the_retrospective_and_the_extreme_blocks(world, monkeypatch) -> None:
    parts = study3_world(world)
    monkeypatch.setattr(cli, "study3_parts", lambda *a, **kw: parts)
    # the reference: one annotator's labels over a key of two items per cell
    folder = world.tmp / "reference"
    key = {
        f"i{c}-{j}": {"cell": c}
        for c in range(len(world.fixation["reference_sample"]["cells_drawn"]))
        for j in range(2)
    }
    key_data = dump(folder / "key.json", key)
    world.plans()
    dump(
        folder / "manifest.json",
        {
            "fixation": {"sha256": sha(world.fixation_path.read_bytes())},
            "files": {"key.json": {"sha256": sha(key_data)}},
        },
    )
    labels = {item: "harmful_compliance" if item.endswith("-0") else "refusal" for item in key}
    dump(world.tmp / "sonnet.json", labels)
    dump(world.tmp / "grok.json", {**labels, "i0-1": None})
    study3 = ("--study3-runs-root", world.study3, "--study3-repo", world.study3)
    # the retrospective alone, before the registration: no pod2 input at all
    retro_out = world.tmp / "retrospective.json"
    argv = world.retro_argv(retro_out, *study3)
    shutil.rmtree(world.tmp / "pod2")
    code, why = call(argv)
    assert code == 0, why
    written = json.loads(retro_out.read_text())
    assert written["kind"] == "study5-retrospective" and written["code"] == CODE
    assert not any(k.startswith("pod2:") for k in written["inputs"])
    assert "pod1:plan.json" in written["inputs"]
    assert written["retrospective"]["harm"]["interval"]["seed"] == FIXATION[
        "retrospective_interval"]["seed"]  # fmt: skip
    out = world.tmp / "analysis.json"
    argv = world.argv(out, *study3, "--retrospective", retro_out,
                      "--reference-input", folder, "--reference",
                      f"sonnet={world.tmp / 'sonnet.json'}", "--reference",
                      f"grok={world.tmp / 'grok.json'}")  # fmt: skip
    code, why = call(argv)
    assert code == 0, why
    result = json.loads(out.read_text())
    retro = result["retrospective"]
    assert retro["record"]["sha256"] == sha(retro_out.read_bytes())
    assert {k: v for k, v in retro.items() if k != "record"} == written["retrospective"]
    assert retro["computed"] is True and retro["blocks"] == 6 and retro["harm"]["n"] == 6
    h = [study3_estimates._arm(b, "F", parts["harm"])[0] for b in parts["blocks"]]
    p = [json.loads((world.runs / f"study5_records/pod1-x/predictions/retro/s{b['seed']}.json")
                    .read_text())["p"] for b in parts["blocks"]]  # fmt: skip
    assert retro["harm"]["rho"] == pytest.approx(stats.spearmanr(p, h).statistic)
    assert retro["harm"]["interval"]["resamples"] == 10_000
    assert retro["refusal"]["n"] == 6
    extreme = result["descriptive"]["extreme_blocks"]
    cells_ = world.fixation["reference_sample"]["cells_drawn"]
    high = [parts["harm"][c["run_id"]].harmful / 521 for c in cells_ if c["group"] == "high"]
    low = [parts["harm"][c["run_id"]].harmful / 521 for c in cells_ if c["group"] == "low"]
    assert extreme["wildguard"]["contrast"] == pytest.approx(np.mean(high) - np.mean(low))
    assert extreme["llama_guard"]["contrast"] == pytest.approx(np.mean(high) - np.mean(low))
    reference = extreme["reference"]
    assert reference["annotators"]["sonnet"]["contrast"]["low"] == pytest.approx(0.0)
    assert reference["annotators"]["sonnet"]["high"]["low"] == pytest.approx(0.5)
    assert reference["agreement"]["both_labeled"] == len(key) - 1
    assert reference["agreement"]["share"] == 1.0


def test_the_full_analysis_refuses_a_retrospective_record_that_does_not_match(
    world, monkeypatch
) -> None:
    parts = study3_world(world)
    monkeypatch.setattr(cli, "study3_parts", lambda *a, **kw: parts)
    study3 = ("--study3-runs-root", world.study3, "--study3-repo", world.study3)
    retro_out = world.tmp / "retrospective.json"
    assert call(world.retro_argv(retro_out, *study3))[0] == 0
    out = world.tmp / "analysis.json"
    # without the record, study 3's parts are refused
    code, why = call(world.argv(out, *study3))
    assert code == 1 and "--retrospective" in why
    record = json.loads(retro_out.read_text())
    record["retrospective"]["harm"]["rho"] += 0.01
    other = world.tmp / "other-retrospective.json"
    dump(other, record)
    code, why = call(world.argv(out, *study3, "--retrospective", other))
    assert code == 1 and "does not match its recomputation" in why and not out.exists()


def test_a_dirty_checkout_is_refused(world, monkeypatch) -> None:
    monkeypatch.setattr(cli, "own_code", REAL_OWN_CODE)
    monkeypatch.setattr(cli, "git_state", lambda repo: {"commit": "a" * 40, "dirty": True})
    out = world.tmp / "analysis.json"
    code, why = call(world.argv(out, "--without-study3"))
    assert code == 1 and "clean checkout" in why and not out.exists()


def test_a_run_with_other_than_25_updates_is_technical() -> None:
    found = sa.endpoint_of("r", seed=50000, lr=1.3e-5, stop="steps", row=row(5.0),
                           theta0=THETA0, steps=24)  # fmt: skip
    assert found.state == "technical" and found.reasons == ("24 updates, not 25",)


def test_the_cli_needs_study3s_inputs_or_their_absence_stated(world) -> None:
    code, why = call(world.argv(world.tmp / "analysis.json"))
    assert code == 1 and "--without-study3" in why


def test_the_cli_is_analysis_code_outside_the_gpu_identity() -> None:
    from zero_field_refusal.code_identity import is_gpu_path

    for name in ("study5_analysis.py", "study5_analysis_cli.py"):
        assert is_gpu_path(f"src/zero_field_refusal/analysis/{name}") is False


def test_the_scan_points_are_the_scan_runs() -> None:
    from zero_field_refusal import line_scan

    assert sa.SCAN_POINTS == line_scan.POINTS


def test_study3s_parts_through_study3s_gated_reader(tmp_path, monkeypatch, capsys) -> None:
    """On study 3's synthetic world with the fixation record written from it: the harm of the 60
    blocks' F runs and the 40 relabeled runs comes from study 3's final reader (one gated
    reading), bound to the record's freeze and receipts; a run without three-class or Llama Guard
    label files is named, not counted; another freeze or receipt than the record's is refused."""
    import argparse

    from test_study5_fixation import Fixation

    fixation = Fixation(tmp_path, monkeypatch, capsys)
    fixation.fixed()
    record = json.loads(fixation.out.read_bytes())
    monkeypatch.setattr(cli, "prompt_sets", lambda reader: lambda cfg: PROMPTS)
    args = argparse.Namespace(study3_runs_root=fixation.root, study3_repo=fixation.world.repo,
                              study3_sealed_dir=fixation.root / "sealed")  # fmt: skip
    reader = cli.Reader({"study3_runs": fixation.root})
    parts = cli.study3_parts(args, record, reader, lambda cfg: PROMPTS)
    assert len(parts["blocks"]) == 60
    relabel = record["extreme_blocks"]["relabel"]["runs"]
    for run in relabel:
        assert parts["harm"][run] == fixation.world.harm(fixation.root / "study3" / run)
        assert isinstance(parts["llama"][run], str) and parts["llama"][run].startswith(
            "llama_guard labels:")  # fmt: skip
    assert all(isinstance(v, str) for v in parts["three"].values())
    assert not any(str(tmp_path) in v for v in [*parts["three"].values(), *parts["llama"].values()])
    retro_free = {"inputs": {**record["inputs"],
                             "study3_freeze": {**record["inputs"]["study3_freeze"],
                                               "sha256": "0" * 64}}}  # fmt: skip
    with pytest.raises(cli.Refused, match="freeze is not the one"):
        cli.study3_parts(args, {**record, **retro_free}, reader, lambda cfg: PROMPTS)
    labels = {run: {**v, "receipt_sha256": "1" * 64}
              for run, v in record["inputs"]["labels"]["runs"].items()}  # fmt: skip
    other = {
        **record,
        "inputs": {**record["inputs"], "labels": {**record["inputs"]["labels"], "runs": labels}},
    }
    with pytest.raises(cli.Refused, match="another receipt"):
        cli.study3_parts(args, other, reader, lambda cfg: PROMPTS)


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_a_harm_reading_that_fails_is_a_refusal(tmp_path, monkeypatch, capsys, error) -> None:
    import argparse

    from test_study5_fixation import Fixation

    from zero_field_refusal.analysis import study3_final

    fixation = Fixation(tmp_path, monkeypatch, capsys)
    fixation.fixed()
    record = json.loads(fixation.out.read_bytes())

    def fails(*a, **kw):
        raise error("unreadable")

    monkeypatch.setattr(study3_final, "cohort_harm", fails)
    args = argparse.Namespace(study3_runs_root=fixation.root, study3_repo=fixation.world.repo,
                              study3_sealed_dir=None)  # fmt: skip
    with pytest.raises(cli.Refused, match="harm cannot be read"):
        cli.study3_parts(args, record, cli.Reader({}), lambda cfg: PROMPTS)


def retrospective_world(world, monkeypatch) -> tuple[tuple, Path, dict]:
    """The retrospective record of the synthetic study-3 parts, one block's features record
    missing so that a reason is in it."""
    parts = study3_world(world)
    monkeypatch.setattr(cli, "study3_parts", lambda *a, **kw: parts)
    (world.runs / "study5_records/pod1-x/features/retro/s10402.json").unlink()
    study3 = ("--study3-runs-root", world.study3, "--study3-repo", world.study3)
    retro_out = world.tmp / "retrospective.json"
    assert call(world.retro_argv(retro_out, *study3))[0] == 0
    return study3, retro_out, json.loads(retro_out.read_text())


def edited(world, record: dict, edit) -> Path:
    edit(record["retrospective"])
    path = world.tmp / f"edited-{len(list(world.tmp.glob('edited-*')))}.json"
    dump(path, record)
    return path


def test_the_retrospective_comparison_allows_the_last_bit_and_reworded_reasons(
    world, monkeypatch
) -> None:
    study3, _, record = retrospective_world(world, monkeypatch)

    def ulp(r):
        r["harm"]["rho"] = float(np.nextafter(r["harm"]["rho"], math.inf))
        excluded = r["stream_features"]["mean_tokens"]["excluded"]
        assert excluded and excluded[0]["seed"] == 10402
        excluded[0]["reason"] = "the features record is absent (reworded)"

    out = world.tmp / "analysis.json"
    code, why = call(world.argv(out, *study3, "--retrospective", edited(world, record, ulp)))
    assert code == 0, why
    deviation = json.loads(out.read_text())["retrospective"]["record"]["max_deviation"]
    assert 0 < deviation["relative"] < 1e-15


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r["harm"].__setitem__("rho", r["harm"]["rho"] + 1e-9),
        lambda r: r["harm"].__setitem__("seeds", r["harm"]["seeds"][:-1]),
        lambda r: r["per_block"][0].__setitem__("seed", 10499),
    ],
    ids=["1e-9", "seed set", "seed"],
)
def test_the_retrospective_comparison_refuses_another_result(world, monkeypatch, edit) -> None:
    study3, _, record = retrospective_world(world, monkeypatch)
    out = world.tmp / "analysis.json"
    code, why = call(world.argv(out, *study3, "--retrospective", edited(world, record, edit)))
    assert code == 1 and "does not match its recomputation" in why and not out.exists()


def test_scrub_replaces_absolute_roots_at_path_boundaries_only(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "r").mkdir()
    reader = cli.Reader({"runs": Path("r")})  # a short relative root, as typed
    text = "error in rounds: r is not a run; see r/x and 'rr'"
    assert reader.scrub(text) == text
    root = str((tmp_path / "r").resolve())
    assert reader.scrub(f"'{root}/study5/a' missing") == "'<runs>/study5/a' missing"
    assert reader.scrub(f"{root}2/b and /x{root}/c") == f"{root}2/b and /x{root}/c"
