"""Study 5's order repeat under AdamW (``analysis.study5_adamw`` and
scripts/study5_adamw_analysis.py) on synthetic runs and labels only: the runs keyed by arm, seed,
order and exact rate, a control run that is not E2's ignored and a duplicate key refused; a full
design of ten seeds with known harm giving known means, intervals and readings (an established
case, a paired difference not told apart, and one that is smaller); an unsupported order under
each arm, a missing run and an instability stop excluded with their reasons; the label-noise value
against a hand computation; the direction with a zero in n; the bracket share; and the script end
to end (two runs roots, written once, numbers only on stdout, no text stored, a seeds record that
is not the draw refused)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from zero_field_refusal import adamw_orders
from zero_field_refusal.analysis import study5_adamw as sad
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


script = script_module(REPO_ROOT / "scripts" / "study5_adamw_analysis.py", "study5_adamw_script")

SEEDS_FILE = REPO_ROOT / adamw_orders.SEEDS_FILE
SEEDS_RECORD = json.loads(SEEDS_FILE.read_text())
SEEDS = SEEDS_RECORD["seeds"]
REGISTERED_SEEDS_SHA256 = "f95ba9e21d25ee4c39c5d0d0852c4d1eea3e4d20e9175011e5b26692100e7500"
O1, O2 = sad.ORDERS
L, U = sad.RATE_NAMES
RATE = {arm: dict(sad.RATES[arm]) for arm in sad.ARMS}
OFFSET = dict(adamw_orders.ORDER_OFFSETS)
THETA0 = {"benchmark_accuracy": 0.6, "factual_accuracy": 0.5}
K_AT = {L: 5.0, U: 7.5}  # K_benign at step 25 by rate, under both arms: K* = 6 bracketed
# h(order1) − h(order2) in harmful counts per seed, in seed order, at both rates
D_A = (30, -25, 40, 20, -35, 45, 25, -30, 50, 15)
D_F = (35, 20, -45, 25, 30, -40, 20, 35, -45, 10)
SMALL_A = (1, -1, 2, 0, -2, 1, 1, -1, 2, 1)  # A's orders within two responses, one seed tied
BASE = {"A": 100, "F": 90}
SENTINEL = "TEXT-NEVER-STORED"
IDS = [f"judged:{i}" for i in range(521)]
PROMPTS = {pid: f"Prompt {SENTINEL} {i}" for i, pid in enumerate(IDS)}
SET = "harmful_evaluation"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pp(count: float) -> float:
    return 100.0 * count / 521


def harmful(arm: str, seed: int, order: str, rate: str, d: dict | None = None) -> int:
    """The synthetic design's harmful count of a run: order1 above order2 by d[arm] of the seed at
    both rates, η_u 25 above η_l."""
    d = {"A": D_A, "F": D_F} | (d or {})
    i = SEEDS.index(seed)
    return BASE[arm] + 3 * i + (d[arm][i] if order == O1 else 0) + (25 if rate == U else 0)


def counts(h: int, missing: int = 0) -> dict:
    return {"n": 521, "labeled": 521 - missing, "harmful": h, "refusal": 200,
            "safe_non_refusal": 321 - h - missing, "missing": missing}  # fmt: skip


def bootstrap_by_hand(values) -> tuple[float, float]:
    """The registered interval, resample by resample: seeds drawn with replacement, the 2.5 and
    97.5 percentiles of the means."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(derive_seed(0, "study5", "adamw-bootstrap"))
    means = [x[rng.integers(0, len(x), size=len(x))].mean() for _ in range(10_000)]
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def noise_by_hand(h_bar: float) -> float:
    return 100.0 * math.sqrt(2.0 * h_bar * (1.0 - h_bar) / 521) * math.sqrt(2.0 / math.pi)


# ------------------------------------------------------------------ in-memory runs


def run_of(arm: str, seed: int, order: str, rate: str, *, stop: str = "steps", steps: int = 25,
           k: float | None = None, mmlu: float = 0.59, h: int | None = None, labels=None,
           d: dict | None = None) -> dict:  # fmt: skip
    """One run's readings as the script passes them (``study5_adamw``'s ``Run``)."""
    ended = stop == "steps" and steps == 25
    k = K_AT[rate] if k is None else k
    h = harmful(arm, seed, order, rate, d) if h is None else h
    row = ({"step": 25, "condition": "step-25", "K_benign": k, "K_harmful": 2 * k,
            "benchmark_accuracy": mmlu, "factual_accuracy": 0.49,
            "heldout_nll_per_token": 1.0} if ended else None)  # fmt: skip
    return {
        "run_id": f"r-{arm}-s{seed}-{order}-{rate}", "root": "runs1", "arm": arm, "seed": seed,
        "order": order, "rate": rate, "records": None, "stop": stop, "steps": steps, "row": row,
        "labels": (counts(h) if labels is None else labels) if ended else
                  "labels: LabelsPending: no condition answered",
        "receipt_sha256": "r" * 64 if ended else None,
    }  # fmt: skip


def design_runs(d: dict | None = None) -> dict:
    return {key: run_of(*key, d=d) for key in sad.design(SEEDS)}


def expected(d_a=D_A, d_f=D_F) -> dict:
    """The per-seed |Δ_A|, |Δ_F| and their paired difference of the synthetic design, in pp."""
    abs_a, abs_f = [pp(abs(x)) for x in d_a], [pp(abs(x)) for x in d_f]
    return {"abs_A": abs_a, "abs_F": abs_f,
            "paired": [a - f for a, f in zip(abs_a, abs_f, strict=True)]}  # fmt: skip


# ------------------------------------------------------------------ pure functions


def test_the_constants_are_the_registered_ones() -> None:
    assert sad.K_STAR == sa.K_STAR == 6.0 and sad.WIDTH_LIMIT == 0.60
    assert sad.RATES["A"] == (("eta_l", 1.45e-05), ("eta_u", 2.11e-05)) == adamw_orders.RATES
    assert sad.RATES["F"] == (("eta_l", cli.ETA_L), ("eta_u", cli.ETA_U))
    assert sad.RATES["F"] == (("eta_l", 1.3e-05), ("eta_u", 1.83e-05))
    assert sad.ORDERS == ("order1", "order2") and OFFSET == {"order1": 1000, "order2": 2000}
    assert sad.ROLES == {"A": "order_a", "F": "control"}
    assert sad.OPTIMIZERS == {"A": "adamw", "F": "adam_frozen"}
    assert sad.RESAMPLES == 10_000 and sad.PERCENTILES == (2.5, 97.5) and sad.ZERO_PP == 1e-9
    assert sad.BOOTSTRAP_SEED == derive_seed(0, "study5", "adamw-bootstrap") != sd.BOOTSTRAP_SEED
    assert len(sad.design(SEEDS)) == 80 and len(set(sad.design(SEEDS))) == 80
    assert sha(SEEDS_FILE.read_bytes()) == REGISTERED_SEEDS_SHA256
    assert sorted(adamw_orders.draw_seeds()) == SEEDS


def entry(run_id: str, arm: str, seed: int, order_seed: int | None, lr: float, *,
          ended: bool = True, optimizer: str | None = None,
          root: str = "runs1") -> dict:  # fmt: skip
    return {"run_id": run_id, "root": root, "role": sad.ROLES[arm],
            "optimizer": optimizer or sad.OPTIMIZERS[arm], "seed": seed, "lr": lr,
            "order_seed": order_seed, "ended": ended}  # fmt: skip


def test_runs_are_keyed_by_arm_seed_order_and_exact_rate() -> None:
    s = SEEDS[0]
    found = sad.select_runs([
        entry("a", "A", s, s + 1000, RATE["A"][L]),
        entry("f", "F", s, s + 2000, RATE["F"][U], root="runs2"),
        entry("c", "F", s, s, RATE["F"][L]),  # the crossed order control: not E2's
        entry("v", "F", s, None, RATE["F"][L]),  # a v₀ source or an edited order: no order seed
        entry("n", "A", s, s + 2000, RATE["A"][U], ended=False),  # counts for no key
        entry("o", "A", 50000, 51000, RATE["A"][L]),  # a seed of the population, not drawn
        entry("e", "F", 50000, 52000, RATE["F"][L]),  # E2's run of a seed not drawn
    ], SEEDS)  # fmt: skip
    assert {key: x["run_id"] for key, x in found["runs"].items()} == {
        ("A", s, O1, L): "a", ("F", s, O2, U): "f"}  # fmt: skip
    assert [x["run_id"] for x in found["ignored"]] == ["c", "v"]
    assert found["ignored"][0]["order_seed"] == s and found["ignored"][1]["order_seed"] is None
    assert [x["run_id"] for x in found["not_ended"]] == ["n"]
    assert [x["run_id"] for x in found["other_seeds"]] == ["e", "o"]
    # the same rate under the other arm, the next float and a string are no rate of the arm
    for lr in (RATE["F"][L], math.nextafter(RATE["A"][L], 1.0), "1.45e-05"):
        with pytest.raises(ValueError, match="runs1:x: its rate"):
            sad.select_runs([entry("x", "A", s, s + 1000, lr)], SEEDS)
    with pytest.raises(ValueError, match="its rate"):
        sad.select_runs([entry("x", "F", s, s + 1000, RATE["A"][U])], SEEDS)
    with pytest.raises(ValueError, match="neither seed \\+ 1000 nor seed \\+ 2000"):
        sad.select_runs([entry("x", "A", s, s + 3000, RATE["A"][L])], SEEDS)
    with pytest.raises(ValueError, match="optimizer 'adam_frozen' is not arm A's 'adamw'"):
        sad.select_runs([entry("x", "A", s, s + 1000, RATE["A"][L], optimizer="adam_frozen")],
                        SEEDS)  # fmt: skip


def test_a_key_with_two_ended_runs_is_refused_naming_both() -> None:
    s = SEEDS[2]
    with pytest.raises(ValueError, match=rf"more than one ended run: A s{s} order1 eta_l: "
                                         "runs1:a, runs2:a"):  # fmt: skip
        sad.select_runs([entry("a", "A", s, s + 1000, RATE["A"][L]),
                         entry("a", "A", s, s + 1000, RATE["A"][L], root="runs2")],
                        SEEDS)  # fmt: skip
    with pytest.raises(ValueError, match=rf"F s{s} order2 eta_u: runs1:b, runs1:c"):
        sad.select_runs([entry("c", "F", s, s + 2000, RATE["F"][U]),
                         entry("b", "F", s, s + 2000, RATE["F"][U]),
                         entry("d", "F", s, s + 2000, RATE["F"][U], ended=False)],
                        SEEDS)  # fmt: skip


def test_the_label_noise_against_a_hand_computation() -> None:
    """40 A runs, 8 with 105 harmful and 32 with 104: h̄ = 4168 / (521 × 40) = 0.2 exactly, and
    √(2 · 0.2 · 0.8 / 521) · √(2/π) = 1.9774 pp; F's runs never enter."""
    runs = design_runs()
    a_keys = [key for key in sad.design(SEEDS) if key[0] == "A"]
    for i, key in enumerate(a_keys):
        h = 105 if i < 8 else 104
        runs[key] = run_of(*key, h=h, labels=counts(h, i % 3))
    for key in sad.design(SEEDS):
        if key[0] == "F":
            runs[key] = run_of(*key, h=300)
    found = sad.label_noise(runs, SEEDS)
    assert found["runs"] == 40 and found["excluded"] == [] and found["h_bar"] == 0.2
    assert found["value_pp"] == noise_by_hand(0.2)
    assert found["value_pp"] == pytest.approx(1.9774075544567857, rel=1e-12)
    high = sad.label_noise(runs, SEEDS, "harmful_high")  # missing labels i % 3 counted harmful
    assert high["h_bar"] == (4168 + sum(i % 3 for i in range(40))) / (521 * 40)
    assert high["value_pp"] == pytest.approx(noise_by_hand(high["h_bar"]), rel=1e-12)
    del runs[a_keys[0]]
    runs[a_keys[1]] = run_of(*a_keys[1], stop="gradient_spike", steps=12)
    found = sad.label_noise(runs, SEEDS)
    assert found["runs"] == 38 and found["h_bar"] == (4168 - 2 * 105) / (521 * 38)
    why = [x["reason"] for x in found["excluded"]]
    assert why[0] == "no run" and "gradient_spike" in why[1] and "12 updates" in why[1]


def test_the_bootstrap_is_the_registered_resampling_of_seeds() -> None:
    values = expected()["abs_A"]
    found = sad.mean_with_interval(dict(zip(SEEDS, values, strict=True)), SEEDS)
    assert found["mean"] == pytest.approx(pp(31.5), rel=1e-12)
    assert (found["lower"], found["upper"]) == bootstrap_by_hand(values)
    assert found["rng_seed"] == derive_seed(0, "study5", "adamw-bootstrap")
    one = sad.mean_with_interval({SEEDS[0]: 3.0}, SEEDS[:2])
    assert (one["n"], one["lower"], one["upper"]) == (1, None, None)
    assert one["excluded"] == [{"seed": SEEDS[1], "reason": "no value"}]


def test_the_direction_counts_a_zero_in_n_never_as_an_agreement() -> None:
    a = {0: 1.0, 1: -2.0, 2: 1e-12, 3: 0.5, 4: 3.0, 5: "order1: no run"}
    f = {0: 2.0, 1: -1.0, 2: 4.0, 3: -0.5, 4: 0.0, 5: 1.0}
    signs_a = {s: sad.float_sign(v) for s, v in a.items()}
    signs_f = {s: sad.float_sign(v) for s, v in f.items()}
    assert signs_a == {0: 1, 1: -1, 2: 0, 3: 1, 4: 1, 5: "order1: no run"}
    found = sad.direction(signs_a, signs_f, list(range(6)), "test")
    assert (found["n"], found["agreements"], found["opposite"], found["with_zero"]) == (5, 2, 1, 2)
    assert found["p_one_sided"] == 26 / 32  # P(X ≥ 2), X ~ Binomial(5, 1/2)
    assert found["excluded"] == [{"seed": 5, "reason": "delta_A: order1: no run"}]
    assert sad.float_sign(-0.99e-9) == 0 and sad.float_sign(1e-9) == 1
    assert [sad.count_sign(x) for x in (-3, 0, 2)] == [-1, 0, 1]
    assert sad.count_sign("order2: no run") == "order2: no run"


@pytest.mark.parametrize(
    ("abs_lower", "paired", "noise", "sensitivity", "against"),
    [
        (3.0, (-1.0, 1.0), 2.0, "established", "not told apart"),
        (2.0, (-1.0, 1.0), 2.0, "not established", "not told apart"),  # on it is not above it
        (1.0, (-3.0, -0.5), 2.0, "not established", "smaller"),
        (5.0, (0.5, 3.0), 2.0, "established", "larger"),
        (5.0, (0.0, 3.0), 2.0, "established", "not told apart"),  # 0 on a bound
        (5.0, (-3.0, 0.0), 2.0, "established", "not told apart"),
        (None, (None, None), 2.0, "undetermined", "undetermined"),
        (3.0, (-1.0, 1.0), None, "undetermined", "not told apart"),
    ],
)  # fmt: skip
def test_each_branch_of_the_reading(abs_lower, paired, noise, sensitivity, against) -> None:
    found = sad.reading({"lower": abs_lower}, {"lower": paired[0], "upper": paired[1]}, noise)
    assert found["order_sensitivity"] == sensitivity and found["against_F"] == against


def test_a_full_design_gives_known_means_intervals_and_readings() -> None:
    found = sad.analysis(design_runs(), SEEDS, THETA0)
    assert found["missing"] == [] and len(found["runs"]) == 80
    kstar = found["kstar"]
    assert kstar["unsupported"]["n"] == 0 and kstar["without_labels"]["n"] == 0
    s0, w = SEEDS[0], weight(K_AT[L], K_AT[U], 6.0)
    cell = next(c for c in kstar["cells"] if (c["arm"], c["seed"], c["order"]) == ("A", s0, O1))
    assert cell["weight"] == w and cell["ln_width"] == pytest.approx(math.log(1.5))
    assert cell["lo"] == f"r-A-s{s0}-{O1}-{L}" and cell["hi"] == f"r-A-s{s0}-{O1}-{U}"
    assert cell["h_star"]["harmful"] == pytest.approx(
        (1 - w) * pp(harmful("A", s0, O1, L)) + w * pp(harmful("A", s0, O1, U)),
        rel=1e-12)  # fmt: skip
    primary = found["primary"]
    noise = primary["label_noise"]
    assert noise["runs"] == 40 and noise["h_bar"] == (5040 + 2 * sum(D_A)) / (521 * 40)
    assert noise["value_pp"] == pytest.approx(noise_by_hand(5310 / 20840), rel=1e-12)
    want = expected()
    bases = {"kstar": primary["kstar"], **primary["fixed_rate"]}
    for basis, part in bases.items():
        for name, mean in (("abs_A", 31.5), ("abs_F", 30.5), ("paired", 1.0)):
            assert part[name]["n"] == 10 and part[name]["excluded"] == [], (basis, name)
            assert part[name]["mean"] == pytest.approx(pp(mean), rel=1e-12)
            assert (part[name]["lower"], part[name]["upper"]) == pytest.approx(
                bootstrap_by_hand(want[name]), rel=1e-9)  # fmt: skip
        assert part["label_noise_pp"] == noise["value_pp"]
        assert part["abs_A"]["lower"] > noise["value_pp"]
        assert part["paired"]["lower"] < 0 < part["paired"]["upper"]
        assert part["reading"]["order_sensitivity"] == "established", basis
        assert part["reading"]["against_F"] == "not told apart", basis
        d = part["direction"]  # same signs at seeds 0, 3, 6 and 9
        assert (d["n"], d["agreements"], d["opposite"], d["with_zero"]) == (10, 4, 6, 0)
        assert d["p_one_sided"] == 848 / 1024
    assert primary["kstar"]["direction"]["basis"].startswith("delta* in pp")
    assert primary["fixed_rate"][L]["direction"]["basis"] == "harmful counts, exact"
    first = primary["per_seed"][0]
    assert first["kstar"]["A"]["delta"] == pytest.approx(pp(D_A[0]), rel=1e-9)
    assert first["kstar"]["F"]["sign"] == 1
    assert first["fixed_rate"][L]["F"]["count_difference"] == D_F[0]
    assert first["fixed_rate"][U]["A"][O2] == pytest.approx(pp(harmful("A", s0, O2, U)))
    # the secondary measure equals the primary without missing labels
    high = found["secondary"]["harmful_high"]
    assert high["kstar"]["abs_A"] == primary["kstar"]["abs_A"]
    assert high["label_noise"]["h_bar"] == noise["h_bar"]
    run = next(r for r in found["runs"] if r["run_id"] == f"r-F-s{s0}-{O2}-{U}")
    assert run["h"] == pytest.approx(pp(harmful("F", s0, O2, U))) and run["state"] == "usable"
    assert run["K_benign"] == 7.5 and run["lr"] == 1.83e-05
    assert run["mmlu_loss_pp"] == pytest.approx(1.0)


def test_the_secondary_measure_counts_missing_labels_harmful() -> None:
    runs = design_runs()
    s0, w = SEEDS[0], weight(K_AT[L], K_AT[U], 6.0)
    key = ("A", s0, O1, L)
    runs[key] = run_of(*key, labels=counts(harmful(*key), missing=5))
    found = sad.analysis(runs, SEEDS, THETA0)
    first = found["primary"]["per_seed"][0]
    assert first["fixed_rate"][L]["A"]["delta"] == pytest.approx(pp(D_A[0]))
    high = found["secondary"]["harmful_high"]["per_seed"][0]
    assert high["fixed_rate"][L]["A"]["delta"] == pytest.approx(pp(D_A[0] + 5))
    assert high["fixed_rate"][L]["A"]["count_difference"] == D_A[0] + 5
    assert high["kstar"]["A"]["delta"] == pytest.approx(
        (1 - w) * pp(D_A[0] + 5) + w * pp(D_A[0]), rel=1e-9)  # fmt: skip
    assert found["secondary"]["harmful_high"]["label_noise"]["h_bar"] == 5315 / 20840


def test_order_sensitivity_not_established_smaller_than_f_with_a_zero() -> None:
    d = {"A": SMALL_A}
    found = sad.analysis(design_runs(d), SEEDS, THETA0)
    want = expected(SMALL_A)
    for basis, part in {"kstar": found["primary"]["kstar"],
                        **found["primary"]["fixed_rate"]}.items():  # fmt: skip
        assert part["abs_A"]["mean"] == pytest.approx(pp(1.2), rel=1e-9)
        assert (part["paired"]["lower"], part["paired"]["upper"]) == pytest.approx(
            bootstrap_by_hand(want["paired"]), rel=1e-9)  # fmt: skip
        assert part["abs_A"]["upper"] < part["label_noise_pp"]
        assert part["reading"]["order_sensitivity"] == "not established", basis
        assert part["reading"]["against_F"] == "smaller", basis
        d = part["direction"]  # seed 3 tied under A: in n, no agreement
        assert (d["n"], d["agreements"], d["opposite"], d["with_zero"]) == (10, 3, 6, 1), basis
        assert d["p_one_sided"] == 968 / 1024
    tied = found["primary"]["per_seed"][3]
    assert tied["kstar"]["A"]["delta"] == 0.0 and tied["kstar"]["A"]["sign"] == 0
    assert tied["fixed_rate"][U]["A"]["count_difference"] == 0


def test_an_unsupported_order_under_each_arm_is_excluded_with_its_reason() -> None:
    runs = design_runs()
    a, f = SEEDS[1], SEEDS[2]
    runs[("A", a, O1, L)] = run_of("A", a, O1, L, k=6.5)  # 6.5 and 7.5: no bracket
    runs[("F", f, O2, L)] = run_of("F", f, O2, L, k=1.0)  # 1.0 and 50.0: too wide
    runs[("F", f, O2, U)] = run_of("F", f, O2, U, k=50.0)
    found = sad.analysis(runs, SEEDS, THETA0)
    unsupported = found["kstar"]["unsupported"]["cells"]
    reasons = {(c["arm"], c["seed"], c["order"]): c["reason"] for c in unsupported}
    assert reasons == {("A", a, O1): "unsupported: no bracket",
                       ("F", f, O2): "unsupported: too wide"}  # fmt: skip
    wide = next(c for c in found["kstar"]["cells"] if (c["arm"], c["seed"], c["order"]) == (
        "F", f, O2))  # fmt: skip
    assert wide["ln_width"] == pytest.approx(math.log(50.0)) and wide["h_star"] is None
    kstar = found["primary"]["kstar"]
    for name in ("abs_A", "abs_F", "paired"):
        assert kstar[name]["n"] == 8
        why = {x["seed"]: x["reason"] for x in kstar[name]["excluded"]}
        assert why == {a: "delta_A: order1: unsupported: no bracket",
                       f: "delta_F: order2: unsupported: too wide"}  # fmt: skip
    keep = [i for i in range(10) if SEEDS[i] not in (a, f)]
    assert kstar["abs_A"]["mean"] == pytest.approx(pp(sum(abs(D_A[i]) for i in keep) / 8))
    assert kstar["direction"]["n"] == 8
    # a fixed rate takes no support rule: both seeds count there
    assert found["primary"]["fixed_rate"][L]["abs_A"]["n"] == 10


def test_a_missing_run_and_an_instability_stop_are_excluded_with_their_reasons() -> None:
    runs = design_runs()
    a, f = SEEDS[3], SEEDS[4]
    del runs[("A", a, O2, U)]
    runs[("F", f, O1, L)] = run_of("F", f, O1, L, stop="k_above_max", steps=9)
    found = sad.analysis(runs, SEEDS, THETA0)
    assert found["missing"] == [{"arm": "A", "seed": a, "order": O2, "rate": U}]
    reasons = {(c["arm"], c["seed"], c["order"]): c["reason"]
               for c in found["kstar"]["unsupported"]["cells"]}  # fmt: skip
    assert reasons[("A", a, O2)] == "open: missing (A order2 eta_u: no run)"
    assert reasons[("F", f, O1)].startswith("unsupported: failed endpoint (")
    assert "instability stop k_above_max" in reasons[("F", f, O1)]
    primary = found["primary"]
    why = {x["seed"]: x["reason"] for x in primary["kstar"]["paired"]["excluded"]}
    assert why[a] == "delta_A: order2: open: missing (A order2 eta_u: no run)"
    assert primary["kstar"]["paired"]["n"] == 8
    at_u = {x["seed"]: x["reason"] for x in primary["fixed_rate"][U]["abs_A"]["excluded"]}
    assert at_u == {a: "delta_A: order2: no run"}
    at_l = {x["seed"]: x["reason"] for x in primary["fixed_rate"][L]["abs_F"]["excluded"]}
    assert set(at_l) == {f} and "k_above_max" in at_l[f] and "9 updates" in at_l[f]
    assert primary["fixed_rate"][L]["direction"]["n"] == 9
    assert primary["label_noise"]["runs"] == 39
    stopped = next(r for r in found["runs"] if r["run_id"] == f"r-F-s{f}-{O1}-{L}")
    assert stopped["state"] == "scientific" and "k_above_max" in stopped["h"]["missing"]


def test_a_supported_order_without_labels_has_no_h_star() -> None:
    runs = design_runs()
    s = SEEDS[5]
    pending = "labels: LabelsPending: its wildguard label file is missing"
    runs[("A", s, O1, U)] = run_of("A", s, O1, U, labels=pending)
    found = sad.analysis(runs, SEEDS, THETA0)
    assert found["kstar"]["unsupported"]["n"] == 0
    assert found["kstar"]["without_labels"]["cells"] == [
        {"arm": "A", "seed": s, "order": O1, "reason": f"r-A-s{s}-{O1}-{U}: {pending}"}]
    primary = found["primary"]
    assert primary["kstar"]["abs_A"]["n"] == 9 and primary["label_noise"]["runs"] == 39
    assert primary["fixed_rate"][L]["abs_A"]["n"] == 10
    assert primary["fixed_rate"][U]["abs_A"]["n"] == 9


def test_the_bracket_share_of_a_seed_and_order_pairs() -> None:
    runs = design_runs()
    s0, s1, s2 = SEEDS[:3]
    runs[("A", s0, O1, L)] = run_of("A", s0, O1, L, k=6.5)  # 6.5 and 7.5: no bracket
    runs[("A", s1, O2, U)] = run_of("A", s1, O2, U, stop="gradient_spike", steps=12)
    del runs[("A", s2, O1, L)]
    runs[("F", s0, O1, L)] = run_of("F", s0, O1, L, k=9.0)  # F's pairs are not counted
    found = sad.brackets(runs, SEEDS)
    assert (found["pairs"], found["decided"], found["bracketing"]) == (20, 18, 17)
    assert found["share"] == 17 / 18
    pairs = {(p["seed"], p["order"]): p for p in found["per_pair"]}
    assert pairs[(s0, O1)]["brackets"] is False and pairs[(s0, O1)]["K_benign"] == {L: 6.5, U: 7.5}
    assert pairs[(s1, O2)]["brackets"] is None and "gradient_spike" in pairs[(s1, O2)]["reason"]
    assert pairs[(s2, O1)]["reason"] == "eta_l: no run"
    assert pairs[(s0, O2)]["brackets"] is True


def test_the_module_is_analysis_code_outside_the_gpu_identity() -> None:
    assert is_gpu_path("src/zero_field_refusal/analysis/study5_adamw.py") is False


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


def dump_config(run_dir: Path, cfg: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(json.dumps(cfg))  # JSON is YAML


def run_config(arm: str, seed: int, lr: float, order_seed: int | None, **train) -> dict:
    return {"run_type": "train", "seed": seed, "experiment": "study5",
            "eval": {"harmful": {"prompt_set": SET}},
            "optimizer": {"name": sad.OPTIMIZERS[arm], "lr": lr},
            "schedule": {"policy": "fixed", "evaluate_at": [25]},
            "study5": {"role": sad.ROLES[arm]},
            "train": {"steps": 25, "batch_size": 20, **train,
                      **({} if order_seed is None else {"order_seed": order_seed})}}  # fmt: skip


class World:
    """Two synthetic runs roots: A's (``runs1``) with the 40 A runs, and E2's (``runs2``) with
    θ₀'s evaluation run (role fresh), the 40 F runs, E2's runs of a seed not drawn and two
    control runs that are not E2's (the crossed order control and a v₀ source); each design run
    with its records, responses (each holding ``SENTINEL``) and WildGuard labels."""

    def __init__(self, tmp: Path, *, skip=(), stopped=(), missing_labels=None,
                 runs: bool = True) -> None:  # fmt: skip
        self.tmp = tmp
        self.roots = {"A": tmp / "a-runs", "F": tmp / "e2-runs"}
        for root in self.roots.values():
            (root / "study5").mkdir(parents=True)
        self.theta0 = self.roots["F"] / "study5" / "s5-theta0"
        dump_config(self.theta0, {"run_type": "theta0", "seed": 50000,
                                  "eval": {"harmful": {"prompt_set": SET}},
                                  "study5": {"role": "fresh"},
                                  "optimizer": {"name": "adam_frozen", "lr": 1.3e-05}})  # fmt: skip
        dump(self.theta0 / "summary.json", {**THETA0, "K_benign": 0.0})
        self.missing_labels = missing_labels or {}
        if not runs:
            return
        for key in sad.design(SEEDS):
            if key not in skip:
                self.run(*key, stop=("k_above_max", 9) if key in stopped else ("steps", 25))
        s, f = SEEDS[0], self.roots["F"] / "study5"
        dump_config(f / "s5-order-control", run_config("F", s, RATE["F"][L], s))
        dump_config(f / "s5-v0", run_config("F", s, RATE["F"][L], None))
        for order, offset in OFFSET.items():
            dump_config(f / f"s5-f-{order}-{L}-s50000", run_config("F", 50000, RATE["F"][L],
                                                                  50000 + offset))  # fmt: skip

    def run(self, arm: str, seed: int, order: str, rate: str, *, stop=("steps", 25),
            run_id: str | None = None, root: str | None = None) -> Path:  # fmt: skip
        reason, steps = stop
        name = run_id or f"s5-{arm.lower()}-{order}-{rate}-s{seed}"
        run_dir = self.roots[root or arm] / "study5" / name
        dump_config(run_dir, run_config(arm, seed, RATE[arm][rate], seed + OFFSET[order]))
        dump(run_dir / "meta.json", {"run_id": name, "inputs": {"harmful_prompts": {
            "set": SET, "count": 521, "ids_sha256": sha(json.dumps(IDS).encode())}}})  # fmt: skip
        ended = reason == "steps"
        dump(run_dir / "terminal.json", {"steps": steps, "stop": {"reason": reason},
                                         "evaluations": [25] if ended else []})  # fmt: skip
        if not ended:
            return run_dir
        k = K_AT[rate]
        lines(run_dir / "evaluations.jsonl", [{
            "step": 25, "condition": "step-25", "K_benign": k, "K_harmful": 2 * k,
            "benchmark_accuracy": 0.59, "factual_accuracy": 0.49,
            "heldout_nll_per_token": 1.0}])  # fmt: skip
        n = harmful(arm, seed, order, rate)
        none = self.missing_labels.get((arm, seed, order, rate), 0)
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
        expected_items = [f"step-25/{pid}" for pid in IDS]
        dump(run_dir / "judge" / "wildguard.receipt.json", {
            "judge": "wildguard", "identity": IDENTITY, "expected_items": expected_items,
            "items": len(expected_items), "rounds": LABEL_ROUNDS if none else 1, "missing": none,
            "labels_sha256": sha((run_dir / "judge" / "wildguard.jsonl").read_bytes()),
            "completed_at": "2026-11-21T00:00:00+00:00"})  # fmt: skip
        return run_dir

    def argv(self, out: Path, *, runs=None, seeds: Path = SEEDS_FILE) -> list[str]:
        roots = [self.roots["A"], self.roots["F"]] if runs is None else runs
        return [str(x) for x in (*[y for root in roots for y in ("--runs", root)],
                                 "--seeds", seeds, "--theta0-run", self.theta0,
                                 "--out", out)]  # fmt: skip


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
    world = World(tmp_path, missing_labels={("A", s0, O1, L): 5})
    out = tmp_path / "record.json"
    assert call(world.argv(out)) == (0, "")
    stdout = capsys.readouterr().out
    data = out.read_bytes()
    record = json.loads(data)
    assert SENTINEL not in data.decode() and SENTINEL not in stdout and "Reply" not in stdout
    assert stdout.splitlines()[-1] == f"record sha256 {sha(data)}"
    assert record["kind"] == "study5-adamw-analysis" and record["seeds"] == SEEDS
    assert record["inputs"]["seeds"]["sha256"] == REGISTERED_SEEDS_SHA256
    assert record["inputs"]["roots"] == ["runs1", "runs2"]
    assert record["inputs"]["theta0"] == {"run_id": "s5-theta0", "accuracies": THETA0}
    files = record["inputs"]["files"]
    assert any(k.startswith("runs1:study5/s5-a-") for k in files)
    assert any(k.startswith("runs2:study5/s5-f-") for k in files)
    assert set(record["git"]) == {"commit", "dirty"} and record["written_utc"].endswith("+00:00")
    assert record["missing"] == [] and len(record["runs"]) == 80
    assert len(record["label_receipts"]) == 80
    assert all(len(v) == 64 for v in record["label_receipts"].values())
    selection = record["selection"]
    assert len(selection["keyed"]) == 80 and selection["not_ended"] == []
    assert sorted(x["run_id"] for x in selection["ignored"]) == ["s5-order-control", "s5-v0"]
    assert sorted(x["run_id"] for x in selection["other_seeds"]) == [
        f"s5-f-{order}-{L}-s50000" for order in (O1, O2)]  # fmt: skip
    roots = {(x["arm"], x["root"]) for x in selection["keyed"]}
    assert roots == {("A", "runs1"), ("F", "runs2")}
    # the primary result: |Δ*| of 31.5 and 30.5 harmful responses of 521, paired 1.0
    primary = record["primary"]
    want = expected()
    for name, mean in (("abs_A", 31.5), ("abs_F", 30.5), ("paired", 1.0)):
        found = primary["kstar"][name]
        assert found["n"] == 10 and found["mean"] == pytest.approx(pp(mean), rel=1e-12)
        assert (found["lower"], found["upper"]) == pytest.approx(
            bootstrap_by_hand(want[name]), rel=1e-9)  # fmt: skip
    assert primary["label_noise"]["h_bar"] == 5310 / 20840
    assert primary["kstar"]["reading"]["order_sensitivity"] == "established"
    assert primary["kstar"]["reading"]["against_F"] == "not told apart"
    assert primary["kstar"]["direction"]["agreements"] == 4
    assert "harmful kstar reading A established against_F not_told_apart" in stdout
    assert "harmful kstar direction n 10 agreements 4 opposite 6 with_zero 0 p 0.828125" in stdout
    # the secondary measure: 5 missing labels of one A run count as harmful
    high = record["secondary"]["harmful_high"]
    assert high["per_seed"][0]["fixed_rate"][L]["A"]["count_difference"] == D_A[0] + 5
    assert high["label_noise"]["h_bar"] == 5315 / 20840
    run = next(r for r in record["runs"] if r["run_id"] == f"s5-a-{O1}-{L}-s{s0}")
    assert run["labels"]["missing"] == 5 and run["state"] == "usable" and run["root"] == "runs1"
    assert run["mmlu_loss_pp"] == pytest.approx(1.0) and run["K_benign"] == 5.0
    assert record["secondary"]["brackets"]["share"] == 1.0
    # written once
    assert call(world.argv(out))[0] == 1
    assert out.read_bytes() == data


def test_the_script_excludes_a_missing_run_and_an_instability_stop(tmp_path, patched,
                                                                    capsys) -> None:  # fmt: skip
    a, f = SEEDS[3], SEEDS[4]
    world = World(tmp_path, skip={("A", a, O2, U)}, stopped={("F", f, O1, L)})
    out = tmp_path / "record.json"
    assert call(world.argv(out)) == (0, "")
    assert f"missing A s{a} order2 eta_u" in capsys.readouterr().out
    record = json.loads(out.read_bytes())
    assert record["missing"] == [{"arm": "A", "seed": a, "order": O2, "rate": U}]
    reasons = {(c["arm"], c["seed"], c["order"]): c["reason"]
               for c in record["kstar"]["unsupported"]["cells"]}  # fmt: skip
    assert set(reasons) == {("A", a, O2), ("F", f, O1)}
    assert reasons[("A", a, O2)] == "open: missing (A order2 eta_u: no run)"
    assert reasons[("F", f, O1)].startswith("unsupported: failed endpoint")
    assert record["primary"]["kstar"]["paired"]["n"] == 8
    stopped = next(r for r in record["runs"] if r["run_id"] == f"s5-f-{O1}-{L}-s{f}")
    assert stopped["state"] == "scientific" and stopped["stop"] == "k_above_max"
    assert "no condition answered" in stopped["labels"]["missing"]
    assert stopped["receipt_sha256"] is None


def test_the_script_refuses_a_key_with_two_ended_runs(tmp_path, patched) -> None:
    world = World(tmp_path)
    s = SEEDS[5]
    world.run("F", s, O2, U, run_id="s5-f-again")
    out = tmp_path / "record.json"
    code, why = call(world.argv(out))
    assert code == 1 and "more than one ended run" in why and not out.exists()
    assert f"F s{s} order2 eta_u: runs2:s5-f-again, runs2:s5-f-{O2}-{U}-s{s}" in why
    (world.roots["F"] / "study5" / "s5-f-again" / "terminal.json").unlink()
    world.run("F", s, O1, L, root="A")  # the same run restored under both roots
    code, why = call(world.argv(out))
    name = f"s5-f-{O1}-{L}-s{s}"
    assert code == 1 and f"F s{s} order1 eta_l: runs1:{name}, runs2:{name}" in why


def test_the_script_refuses_a_seeds_record_that_is_not_the_draw(tmp_path, patched) -> None:
    world = World(tmp_path, runs=False)
    out = tmp_path / "record.json"
    other = dict(SEEDS_RECORD, seeds=sorted(SEEDS_RECORD["population"][:10]))
    path = tmp_path / "seeds.json"
    dump(path, other)
    code, why = call(world.argv(out, seeds=path))
    assert code == 1 and "not the draw's" in why and not out.exists()
    dump(path, dict(SEEDS_RECORD, seeds=SEEDS_RECORD["drawn_in_order"]))  # not sorted
    assert call(world.argv(out, seeds=path))[0] == 1
    code, why = call(world.argv(out, runs=[world.roots["A"], tmp_path / "nowhere"]))
    assert code == 1 and "runs2 holds no study5/ folder" in why and not out.exists()
