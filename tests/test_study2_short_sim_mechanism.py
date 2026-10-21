"""Amendment 4's simulation: B = 20's inputs record, the mechanism cohort's scenarios, its max-t
interval (exactly the analysis's), the calibration rule and the primary rerun with the inputs."""

import dataclasses
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from zero_field_refusal.analysis import study2_short as analysis
from zero_field_refusal.analysis import study2_short_sim as sim

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "study2_short_simulate.py"
RECORD = ROOT / "docs" / "study2-short-b20-inputs.json"
FIXATION = ROOT / "docs" / "study2-short-fixation.json"
TINY = ["--workers", "1", "--spread-replicates", "3", "--spread-blocks", "40", "--bootstrap",
        "200"]  # fmt: skip


def load_script():
    spec = importlib.util.spec_from_file_location("study2_short_simulate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(**changes) -> sim.Inputs:
    return dataclasses.replace(sim.read_inputs(RECORD), **changes)


STRESSED = {"failure_rate": (0.1, 0.1, 0.3)}  # attrition modes that differ from none


# The max-t interval and the readings: exactly the analysis's


def cohorts():
    rng = np.random.default_rng(7)
    yield rng.normal(0.02, 0.05, 60), rng.normal(-0.01, 0.03, 60)
    skewed = rng.gamma(2.0, 0.03, 45) - 0.06
    yield skewed, 0.5 * skewed + rng.normal(0, 0.02, 45)
    tied = np.zeros(60)
    tied[:2] = [0.01, 0.002]  # most blocks tied: many resamples have |T*| = ∞
    yield tied, rng.normal(0, 0.04, 60)
    yield np.full(40, 0.003), rng.normal(0, 0.04, 40)  # a zero SE
    yield rng.normal(0, 0.05, 3), rng.normal(0, 0.05, 3)


@pytest.mark.parametrize("resamples", [200, 2000])
def test_simulated_max_t_equals_analysis_max_t(resamples: int) -> None:
    for c1, c2 in cohorts():
        for seed in (0, 20261020, sim.stream_seed(0, "mechanism-calibration", "x", 3, "max_t")):
            found = sim.mechanism_intervals(c1, c2, sim.MECHANISM_LEVELS, resamples,
                                            [seed, seed + 1, seed + 2])  # fmt: skip
            for i, level in enumerate(sim.MECHANISM_LEVELS):
                expected = analysis.max_t_interval(c1, c2, level, resamples, seed)
                q = expected["q"]
                assert found["q"][i] == (math.inf if q is None else q)
                for j, contrast in enumerate(expected["contrasts"]):
                    rows = {k: np.array([found[k]], dtype=float) for k in sim.ROW_FIELDS
                            if k in found} | {"n": np.array([found["n"]], dtype=float)}  # fmt: skip
                    has = (rows["readable"][0, j] and not rows["zero_se"][0, j]
                           and math.isfinite(found["q"][i]))  # fmt: skip
                    assert has == (contrast["interval"] is not None)
                    if has:
                        low = found["mean"][j] - found["q"][i] * found["se"][j]
                        high = found["mean"][j] + found["q"][i] * found["se"][j]
                        assert [low, high] == contrast["interval"]  # bit for bit
            for j, c in enumerate((c1, c2)):
                bonferroni = analysis.symmetric_studentized(c, analysis.COMPARISON_LEVEL,
                                                            resamples, seed + 1 + j)  # fmt: skip
                q = bonferroni["q"]
                assert found["bonferroni_q"][j] == (math.inf if q is None else q)


def test_simulated_readings_are_the_analysis_readings() -> None:
    """``mechanism_readings`` (vectorized over replicates) against ``component_readings`` on the
    analysis's intervals, with the zero-SE and minimum rules of ``mechanism_analysis``."""
    rows, expected = [], []
    for c1, c2 in cohorts():
        for seed in range(3):
            found = sim.mechanism_intervals(c1, c2, (0.95,), 500, [seed, 10 + seed, 20 + seed])
            joint = analysis.max_t_interval(c1, c2, 0.95, 500, seed)
            bonferroni = [analysis.symmetric_studentized(c, 0.975, 500, s)
                          for c, s in ((c1, 10 + seed), (c2, 20 + seed))]  # fmt: skip
            per = joint["contrasts"]
            zero = [k for k in range(2) if per[k]["zero_se"]]
            readings = []
            for k in range(2):
                if len(c1) < analysis.MECHANISM_MINIMUM or per[k]["zero_se"] is None or k in zero:
                    readings.append(set())
                    continue
                interval = bonferroni[k]["interval"] if zero else per[k]["interval"]
                readings.append(set(analysis.component_readings(interval)) - {"inconclusive"})
            rows.append(found)
            expected.append(readings)
    fields = ("n", "mean", "se", "zero_se", "readable", "q", "bonferroni_q")
    stacked = {k: np.array([r[k] for r in rows], dtype=float) for k in fields}
    got = sim.mechanism_readings(stacked, stacked["q"][:, 0])
    for i, readings in enumerate(expected):
        for k in range(2):
            names = {name for name in ("positive", "negative", "bounded") if got[name][i, k]}
            assert names == readings[k], (i, k)


# The inputs record


def test_b20_inputs_are_recorded(tmp_path: Path) -> None:
    found = sim.read_inputs(RECORD)
    assert found.sha256 == hashlib.sha256(RECORD.read_bytes()).hexdigest()
    assert (found.batch, found.status) == (20, "provisional")
    assert found.sigma_u == (0.056, 0.078, 0.078) and found.sigma_e == (0.015, 0.036, 0.036)
    assert {p.split(":")[0] for p in found.provisional} >= {
        "P.sigma_u", "P.sigma_e", "P.support_rate", "P.failure_rate"
    }  # fmt: skip
    raw = json.loads(RECORD.read_text())

    def refused(change, match: str) -> None:
        record = json.loads(json.dumps(raw))
        change(record)
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError, match=match):
            sim.read_inputs(path)

    refused(lambda r: r["arms"]["P"].pop("failure_rate"), "does not hold exactly")
    refused(lambda r: r["arms"].pop("F"), "are not exactly")
    refused(lambda r: r.pop("batch"), "lacks")
    refused(lambda r: r.update(extra=1), "does not know")
    refused(lambda r: r["arms"]["A"]["sigma_u"].update(value=-0.1), "positive K spread")
    refused(lambda r: r["arms"]["A"]["sigma_u"].update(value="0.05"), "finite number")
    refused(lambda r: r["arms"]["A"]["failure_rate"].update(value=1.0), "failure rate")
    refused(lambda r: r["arms"]["A"]["support_rate"].update(value=0.0), "support rate")
    refused(lambda r: r["arms"]["A"]["sigma_e"].pop("source"), "lacks")
    refused(lambda r: r["arms"]["A"]["sigma_e"].update(source=" "), "no source")
    refused(lambda r: r["arms"]["A"]["sigma_e"].update(note="x"), "unknown keys")
    refused(lambda r: r.update(status="measured"), "provisional entries")
    (tmp_path / "broken.json").write_text("{")
    with pytest.raises(ValueError, match="not JSON"):
        sim.read_inputs(tmp_path / "broken.json")

    # The simulation reads the record: a changed record changes the model, and every run
    # carries the record's sha256 and a copy of it in its inventory.
    record = json.loads(json.dumps(raw))
    record["arms"]["A"]["sigma_u"]["value"] = 0.09
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(record))
    changed = sim.read_inputs(path)
    assert sim.mechanism_base(changed).sigma_u_arms == (0.09, 0.078, 0.078)
    script = load_script()
    out = tmp_path / "run"
    script.main(["mechanism", "--inputs", str(path), "--scenarios",
                 "mech_normal_sd5_r1-1_rho+0_c1+0_c2+0_none", "--reps", "3", *TINY,
                 "--out", str(out)])  # fmt: skip
    summary = json.loads((out / "summary.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert summary["inputs"]["sha256"] == digest
    assert summary["inputs"]["sigma_u"]["A"] == 0.09
    inventory = dict(line.split("\t")[0::2] for line in (out / "inventory.tsv").read_text()
                     .splitlines()[1:])  # fmt: skip
    assert inventory["inputs.json"] == digest
    # The primary modes take it too: per-arm K spreads and failure rates in every scenario.
    out = tmp_path / "primary"
    pattern = "interp_normal_study1_sd5_d+0_w0.47_sua*"
    script.main(["decided", "--inputs", str(path), "--scenarios", pattern, "--reps", "2", *TINY,
                 "--out", str(out)])  # fmt: skip
    summary = json.loads((out / "summary.json").read_text())
    assert summary["inputs"]["sha256"] == digest
    (scenario,) = summary["scenarios"]
    assert scenario["settings"]["sigma_u_arms"] == [0.09, 0.078, 0.078]
    assert scenario["settings"]["failure_arms"] == [0.0, 0.0, 0.0]
    assert "_sua0.09-0.078-0.078_sea0.015-0.036-0.036_fa0-0-0" in scenario["name"]


def test_inputs_are_measured_on_k_and_support_only(tmp_path: Path) -> None:
    """``measure_inputs`` on stand-in runs: the spreads from a rate × seed table of ln K25 (seed
    effects, residual), the failure and support rates at the selected rates."""
    seeds = (2000, 2001, 2002)
    effects = {2000: -0.05, 2001: 0.06, 2002: -0.01}
    noise = {(lr, s): 0.01 * ((i * 3 + j) % 5 - 2) for i, lr in enumerate((1e-5, 1.5e-5, 2e-5))
             for j, s in enumerate(seeds)}  # fmt: skip

    def run(arm: str, lr: float, seed: int, k: float | None) -> SimpleNamespace:
        evaluations = [] if k is None else [{"step": 25, "K_benign": k, "benchmark_accuracy": 0.5,
                                             "factual_accuracy": 0.9}]  # fmt: skip
        return SimpleNamespace(arm=arm, batch_size=20, lr=lr, seed=seed, planned_steps=25,
                               stop="steps" if k else "gradient_spike",
                               terminal={"steps": 25 if k else 3}, steps=25 if k else 3,
                               evaluations=evaluations, run_id=f"{arm}-{lr}-{seed}")  # fmt: skip

    runs = []
    for name in (analysis.ADAMW, analysis.FA, analysis.PREWARMED):
        for lr, base in ((1e-5, 4.6), (1.5e-5, 6.3), (2e-5, 8.4)):
            for s in seeds:
                k = base * math.exp(effects[s] + noise[(lr, s)])
                failed = name == analysis.PREWARMED and lr == 2e-5 and s == 2002
                runs.append(run(name, lr, s, None if failed else k))
    found = sim.measure_inputs(runs, batch=20, seeds=seeds,
                               rates={arm: [1e-5, 1.5e-5, 2e-5] for arm in sim.ARMS})  # fmt: skip
    table = np.log([[runs[i * 3 + j].evaluations[0]["K_benign"] for j in range(3)]
                    for i in range(3)])  # fmt: skip
    residual = table - table.mean(axis=1, keepdims=True)
    assert found["A"]["sigma_u"] == pytest.approx(np.std(residual.mean(axis=0), ddof=1))
    noise_part = residual - residual.mean(axis=0)
    assert found["A"]["sigma_e"] == pytest.approx(math.sqrt((noise_part**2).sum() / 4))
    assert found["A"]["failure_rate"] == 0 and found["A"]["support_rate"] == 1
    assert found["P"]["failure_counts"] == {"runs": 9, "failures": 1}
    assert found["P"]["support_rate"] == pytest.approx(2 / 3)  # seed 2002 lost its endpoint
    assert found["P"]["complete_rates"] == [1e-5, 1.5e-5]
    record = sim.inputs_record(found, batch=20, where="stand-in runs")
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    measured = sim.read_inputs(path)
    assert measured.status == "measured" and measured.failure_rate[2] == pytest.approx(1 / 9)


# The mechanism scenarios


def test_component_scenarios_have_declared_means() -> None:
    """The supported interpolation estimand computed from the population equals the declared
    (C1, C2) for every component scenario under every attrition mode (failure rates that make
    the modes differ), and a large simulated cohort's mean agrees within its MC error."""
    stressed = inputs(**STRESSED)
    for c1, c2 in sim.COMPONENT_MEANS:
        for attrition in sim.ATTRITIONS:
            m = sim.Mechanism(c1, c2, sd=0.05, rho=0.5, attrition=attrition)
            params = sim.mechanism_parameters(m, stressed, (0.35, 0.3))
            assert params.estimand["C1"] == pytest.approx(c1, abs=1e-10)
            assert params.estimand["C2"] == pytest.approx(c2, abs=1e-10)
            assert params.estimand["beta_explicit"] == (attrition == "support_dependent")
    # The population is not the declared values by construction: unsolved offsets miss them,
    # and a simulated cohort agrees with the solved ones.
    m = sim.Mechanism(0.05, -0.05, shape="skewed", sd=0.10, rho=-0.5,
                      attrition="support_dependent")  # fmt: skip
    params = sim.mechanism_parameters(m, stressed, (0.8, 0.8))
    base = sim.mechanism_base(stressed)
    sums, sizes = [], []
    for rep in range(600):
        rng = sim.generator(1, "test-estimand", rep)
        alpha, misses = sim.draw_constants(base, rng)
        estimate, cause = sim.simulate_mechanism_blocks(m, params, base, alpha, misses, rng, 60)
        c1, c2 = sim.cohort_contrasts(estimate, cause)
        sums.append([c1.sum(), c2.sum()])
        sizes.append(c1.size)
    sums, sizes = np.array(sums), np.array(sizes)
    pooled = sums.sum(axis=0) / sizes.sum()
    se = (sums / sizes[:, None]).std(axis=0, ddof=1) / math.sqrt(len(sizes))
    assert np.all(np.abs(pooled - np.array(m.truth)) < 4 * se), (pooled, se)


def test_component_sds_hit_their_targets_where_the_run_draws_them() -> None:
    """The spreads are calibrated at the offsets the run solves for (0, 0): an independent
    sample of the (0, 0) scenario attains the target SDs. Equal offsets left them 6-10 % short at
    SD 10, where the solved offsets lower P's and A's harm and the logistic's slope; the
    calibration's and this sample's SDs are each good to about 2 %."""
    key = sim.Mechanism(sd=0.10, rho=0.0).key
    calibration = sim.calibrate_components(key, inputs(), base_seed=0, replicates=60, blocks=200)
    assert calibration.attained == pytest.approx(key.targets, rel=1e-3)
    params = sim.mechanism_parameters(key, inputs(), calibration.spreads)
    base = sim.mechanism_base(inputs())
    variances = []
    for rep in range(200):
        rng = sim.generator(3, "test-sd", rep)
        alpha, misses = sim.draw_constants(base, rng)
        estimate, cause = sim.simulate_mechanism_blocks(key, params, base, alpha, misses, rng, 100)
        e = estimate[(cause == sim.SUPPORTED).all(axis=1)]
        variances.append([(e[:, i] - e[:, k]).var(ddof=1) for i, k in sim.COMPONENT_ARMS])
    attained = np.sqrt(np.mean(variances, axis=0))
    assert attained == pytest.approx(key.targets, rel=0.05)
    unsolved = sim.mechanism_parameters(key, inputs(), calibration.spreads, solve=False)
    assert unsolved.offsets[0] - params.offsets[0] > 0.2  # the offsets the run uses differ


def test_component_effects_follow_the_shapes() -> None:
    rng = np.random.default_rng(3)
    w1 = rng.standard_normal(400_000)
    w2 = 0.5 * w1 + math.sqrt(0.75) * rng.standard_normal(w1.size)
    tail = rng.random(w1.size)
    e1, e2 = sim.component_effects("skewed", (1.0, 2.0), w1, w2, tail)
    assert e1.mean() == pytest.approx(0, abs=0.01) and e1.std() == pytest.approx(1, abs=0.01)
    assert ((e1 - e1.mean()) ** 3).mean() == pytest.approx(math.sqrt(2), abs=0.08)  # Gamma(2)
    assert e2.std() == pytest.approx(2, abs=0.02)
    d = sim.arm_effects("skewed", e1, e2)
    assert np.allclose(d[:, 2] - d[:, 1], e1) and np.allclose(d[:, 0] - d[:, 2], e2)
    assert (d[:, 1] == 0).all()
    e1, e2 = sim.component_effects("rare_only", (-1.5, 2.0), w1, w2, tail)
    d = sim.arm_effects("rare_only", e1, e2)
    assert (d >= 0).all() and np.allclose(d[:, 2] - d[:, 1], e1)
    assert np.mean(e1 != 0) == pytest.approx(sim.RARE, abs=0.002)
    # The quadrature carries the same joint law.
    for shape in ("normal", "skewed", "heavy", "rare_centered"):
        n1, n2, weights = sim.component_nodes(shape, (0.7, 0.4), 0.5)
        s1, s2 = sim.component_effects(shape, (0.7, 0.4), w1, w2, tail)
        assert weights.sum() == pytest.approx(1)
        assert weights @ n1 == pytest.approx(s1.mean(), abs=0.01)
        assert weights @ (n1 * n2) == pytest.approx((s1 * s2).mean(), abs=0.01)


def test_attrition_support_dependent_hits_high_harm_blocks() -> None:
    """Support-dependent attrition: the pre-warmed arm's runs fail more often in high-harm
    blocks (the stream effect), so the blocks it loses carry more harm in every arm; under
    independent attrition the lost blocks are like the others."""
    stressed = inputs(**STRESSED)
    base = sim.mechanism_base(stressed)
    gaps = {}
    for attrition in ("independent", "support_dependent"):
        m = sim.Mechanism(sd=0.05, attrition=attrition)
        params = sim.mechanism_parameters(m, stressed, (0.3, 0.3), solve=False)
        assert params.failure[0] == (sim._logistic_intercept(0.1, 0.0), 0.0)
        estimates, causes = [], []
        for rep in range(40):
            rng = sim.generator(2, "test-attrition", rep)
            alpha, misses = sim.draw_constants(base, rng)
            e, c = sim.simulate_mechanism_blocks(m, params, base, alpha, misses, rng, 200)
            estimates.append(e)
            causes.append(c)
        e, c = np.concatenate(estimates), np.concatenate(causes)
        lost = c[:, 2] == sim.TECHNICAL
        assert lost.mean() > 0.2  # P loses a block when an endpoint it needs fails
        f_ok = c[:, 1] == sim.SUPPORTED
        gaps[attrition] = e[f_ok & lost, 1].mean() - e[f_ok & ~lost, 1].mean()
        assert (c[:, 0] == sim.TECHNICAL).mean() < lost.mean()  # A fails at its own 0.1
    assert gaps["support_dependent"] > 0.03
    assert abs(gaps["independent"]) < 0.01


def test_mechanism_grid() -> None:
    grid = sim.mechanism_grid()
    names = [m.name for m in grid]
    assert len(set(names)) == len(names)
    required = [m for m in grid if m.group == "required"]
    assert len(required) == 2 * 2 * 9 * 3 * 3 * 3 == 972
    assert {m.reps for m in required} == {10_000}
    assert {(m.shape, m.sd) for m in required} == {
        (s, sd) for s in ("normal", "skewed") for sd in (0.05, 0.10)
    }
    assert {(m.c1, m.c2) for m in required} == set(sim.COMPONENT_MEANS)
    assert {m.ratio for m in required} == {"1:1", "2:1", "1:2"}
    assert sim.Mechanism(sd=0.10, ratio="1:2").targets == (0.05, 0.10)
    stress = [m for m in grid if m.group == "stress"]
    assert {m.shape for m in stress} == {"heavy", "rare_centered", "rare_only", "tied", "normal",
                                         "skewed"}  # fmt: skip
    assert {m.sd for m in stress if m.shape in ("normal", "skewed")} == {0.15}
    assert {m.baseline for m in stress if m.shape == "tied"} == set(sim.TIED_BASELINES)
    assert sim.coverage_floor(10_000) == pytest.approx(0.94564, abs=1e-5)
    # The cohort's constants are the analysis's.
    assert (sim.MECHANISM_TARGET, sim.MECHANISM_CAP, sim.MECHANISM_MINIMUM) == (60, 80, 36)
    assert sim.MECHANISM_CAP == sim.CAP and sim.COMPARISON_LEVEL == 0.975
    assert sim.ZERO_SE == analysis.ZERO_SE


def test_tied_contrasts_give_infinite_critical_values() -> None:
    """The tied stress case: most blocks have h = 0 in every arm, so resamples give |T*| = ∞;
    a q of ∞ or a zero SE leaves a contrast without an interval and without a reading."""
    m = sim.Mechanism(shape="tied", sd=0.0, baseline=2e-5, reps=40)
    settings = sim.MechanismSettings(stream="test", inputs=inputs(), levels=(0.95,), resamples=300)
    found = sim.run_mechanism([m], settings)
    (s,) = found["scenarios"]
    assert s["levels"][0]["infinite_q"]["value"] > 0.5
    assert s["infinite_draw_share"] > 0.05
    assert s["levels"][0]["simultaneous"]["value"] == 1.0  # no interval: the whole line


# The calibration rule and its streams


def test_calibration_uses_separate_streams(tmp_path: Path) -> None:
    script = load_script()
    assert sim.MECHANISM_STREAMS == {"calibration": "mechanism-calibration",
                                     "check": "mechanism-check"}  # fmt: skip
    pattern = ["--scenarios", "mech_normal_sd5_r1-1_rho+0_c1+5_c2+0_none", "--reps", "4"]
    calibration = tmp_path / "calibration"
    script.main(["mechanism", *pattern, *TINY, "--out", str(calibration)])
    first = json.loads((calibration / "summary.json").read_text())
    assert first["settings"]["stream"] == "mechanism-calibration"
    assert first["settings"]["levels"] == list(sim.MECHANISM_LEVELS)
    with pytest.raises(SystemExit):
        script.main(["mechanism-check", *pattern, *TINY])  # no level
    check = tmp_path / "check"
    script.main(["mechanism-check", *pattern, *TINY, "--level", "0.96", "--out", str(check)])
    second = json.loads((check / "summary.json").read_text())
    assert second["settings"]["stream"] == "mechanism-check"
    assert second["settings"]["levels"] == [0.96] and second["check"]["level"] == 0.96
    a, b = first["scenarios"][0], second["scenarios"][0]
    assert a["parameters"] == b["parameters"]  # the same scenario …
    assert a["estimand_check"]["C1"]["cohort_mean"] != b["estimand_check"]["C1"]["cohort_mean"]
    # … and --level-from takes the calibration run's level.
    level_from = tmp_path / "level.json"
    level_from.write_text(json.dumps({"checks": {"level": 0.97}}))
    args = script.build_parser().parse_args(["mechanism-check", "--level-from", str(level_from)])
    assert script.checked_level(args.level_from) == 0.97
    level_from.write_text(json.dumps({"checks": {"level": None}}))
    with pytest.raises(SystemExit):
        script.main(["mechanism-check", *pattern, *TINY, "--level-from", str(level_from)])


def _verdict_scenario(name: str, group: str, shape: str, sd: float, coverages: list) -> dict:
    return {
        "name": name,
        "reps": 10_000,
        "settings": {"group": group, "shape": shape, "sd": sd, "ratio": "1:1", "c1": 0.05,
                     "c2": 0.0},
        "levels": [{"level": lv, "simultaneous": {"value": c, "se": 0.002},
                    "components": {name: {"power": {"value": 0.9}, "false_direction":
                                          {"value": 0.01}, "bounded": {"value": 0.5}}
                                   for name in sim.COMPONENTS}}
                   for lv, c in coverages],
        "minimum_met": {"value": 1.0},
    }  # fmt: skip


def test_the_calibration_rule_takes_the_smallest_level_every_required_cell_meets() -> None:
    floor = sim.coverage_floor(10_000)
    levels = [(0.95, floor - 0.001), (0.955, floor + 0.0001), (0.96, 0.96)]
    passing = [(0.95, 0.951), (0.955, 0.955), (0.96, 0.96)]
    scenarios = [
        _verdict_scenario("a", "required", "normal", 0.05, passing),
        _verdict_scenario("b", "required", "skewed", 0.10, levels),
        _verdict_scenario("c", "stress", "heavy", 0.15, [(lv, 0.5) for lv, _ in levels]),
    ]
    checks = sim.mechanism_checks(scenarios)
    assert checks["level"] == 0.955
    first = checks["per_level"][0]
    assert not first["required"]["meets"] and not first["stress"]["meets"]
    skewed = first["required"]["cells"][1]
    assert (skewed["short"], skewed["mean_coverage"]) == (1, pytest.approx(floor - 0.001))
    assert skewed["short_by_chance"] == pytest.approx(0.0, abs=0.05)  # 2.3 % of one scenario
    assert checks["per_level"][1]["required"]["meets"]  # the stress cell never binds
    assert (
        sim.mechanism_checks(
            scenarios[1:2]
            + [_verdict_scenario("d", "required", "normal", 0.05, [(lv, 0.9) for lv, _ in levels])]
        )["level"]
        is None
    )


def test_final_run_checks_a_raised_level_on_the_check_stream(tmp_path, monkeypatch) -> None:
    """The final run's order: the mechanism on the calibration stream; a check-stream run only
    when the implied level is above 0.95; the primary rerun and its comparisons; the fixed level
    is 0.95, or the raised level that passed the check, else none."""
    script = load_script()
    calls = []

    def fake(argv):
        mode, out = argv[0], Path(argv[argv.index("--out") + 1])
        groups = argv[argv.index("--groups") + 1] if "--groups" in argv else None
        calls.append(f"{mode} {groups}" if groups else mode)
        out.mkdir(parents=True, exist_ok=True)
        level = fake.level
        if mode == "mechanism":
            (out / "summary.json").write_text(json.dumps({"checks": {"level": level}}))
        elif mode == "mechanism-check":
            assert argv[argv.index("--level-from") + 1].endswith("mechanism/summary.json")
            (out / "summary.json").write_text(
                json.dumps({"check": {"level": level, "passes": fake.passes}})
            )
        elif mode == "compare":
            (out / "comparison.json").write_text(
                json.dumps(
                    {"compared": 1, "beyond_2se": 0, "expected_by_chance": 0.0, "max_abs_z": 0.1}
                )
            )

    monkeypatch.setattr(script, "main", fake)
    monkeypatch.setattr(script, "runs_root", lambda: tmp_path / "runs")
    parser = script.build_parser()
    primary = ["decided", "compare", "decided-gate", "compare", "decided-bias", "compare"]
    for level, passes, fixed, check in (
        (0.95, None, 0.95, []),
        (0.96, True, 0.96, ["mechanism-check"]),
        (0.96, False, None, ["mechanism-check"]),
        (None, None, None, []),
    ):
        fake.level, fake.passes = level, passes
        calls.clear()
        out = tmp_path / f"final-{level}-{passes}"
        (tmp_path / "runs").mkdir(exist_ok=True)
        for path in script.ARCHIVED.values():  # amendment 3's archived runs, compared with
            (tmp_path / "runs" / path).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / "runs" / path).write_text("{}")
        args = parser.parse_args(["final", "--inputs", str(RECORD), "--out", str(out)])
        script.final_main(args, parser)
        # Binding results first: the required grid, its check, the primary rerun; stress last.
        assert calls == ["mechanism required", *check, *primary, "mechanism stress"]
        final = json.loads((out / "final.json").read_text())
        assert final["fixed_level"] == fixed and final["written"] == []
        assert (out / "inventory.tsv").is_file()


def test_the_fixation_level_is_written_alone(tmp_path: Path) -> None:
    script = load_script()
    path = tmp_path / "fixation.json"
    path.write_text(FIXATION.read_text())
    script.write_fixation_level(path, 0.96)
    after = analysis.read_fixation(path)
    before = analysis.read_fixation(FIXATION)
    assert after["mechanism"]["level"] == 0.96
    before["mechanism"]["level"] = 0.96
    assert after == before
    changed = [a for a, b in zip(FIXATION.read_text().splitlines(), path.read_text().splitlines(),
                                 strict=True) if a != b]  # fmt: skip
    assert changed == ['    "level": 0.95,']


# The primary model with per-arm inputs


def test_labels_by_probability_are_the_primary_models_labels() -> None:
    """The mechanism draws labels as Φ(Z) < h, the primary model as Z < Φ⁻¹(h): the same
    draws give the same endpoint harms."""
    rng = np.random.default_rng(11)
    eta = rng.normal(-2, 1.5, (40, 3, 3))
    alpha = 1.5 * rng.standard_normal(sim.PROMPTS)
    for rho_b, rho_a in ((0.3, 0.7), (0.0, 0.0), (0.6, 0.9)):
        one = sim.endpoint_harm(eta, alpha, rho_b, rho_a, np.random.default_rng(4))
        two = sim.endpoint_harm_by_probability(eta, alpha, rho_b, rho_a, np.random.default_rng(4))
        np.testing.assert_array_equal(one, two)


def test_the_bias_allowance_holds_with_per_arm_inputs() -> None:
    """A rerun with the inputs (every scenario carries them) still yields its allowance."""
    scenarios = []
    for family, bias in (("study1", 0.002), ("kink", -0.009)):
        scenario = sim.inputs_scenario(sim.Scenario(family=family, w_max=0.47), inputs())
        scenarios.append({"name": scenario.name, "settings": dataclasses.asdict(scenario),
                          "diagnostics": {"block_bias": bias, "block_bias_se": 1e-4}})  # fmt: skip
    found = sim.bias_allowance(scenarios)["allowance"]
    assert found["value"] == 0.009 and found["family"] == "kink"


def test_per_arm_inputs_equal_to_the_scalars_change_no_draw() -> None:
    """Per-arm K spreads and failure rates equal to the scalar ones give the scalar path's
    blocks bit for bit (amendment 3's runs keep their draws)."""
    scalar = sim.Scenario(failure=0.15, failure_correlated=True, w_max=0.47)
    arms = dataclasses.replace(scalar, sigma_u_arms=(0.07,) * 3, sigma_e_arms=(0.03,) * 3,
                               failure_arms=(0.15,) * 3)  # fmt: skip
    rng = np.random.default_rng(0)
    alpha, misses = sim.draw_constants(scalar, rng)
    blocks = []
    for scenario in (scalar, arms):
        params = sim.solve_parameters(scenario, 0.3)
        blocks.append(sim.simulate_blocks(scenario, params, alpha, misses,
                                          np.random.default_rng(5), 30))  # fmt: skip
    np.testing.assert_array_equal(blocks[0][0], blocks[1][0])
    np.testing.assert_array_equal(blocks[0][1], blocks[1][1])
    # Different per-arm values move the arms apart.
    inputs_scenario = sim.inputs_scenario(scalar, inputs())
    assert inputs_scenario.failure_arms is None  # a stress scenario keeps its failures
    central = sim.inputs_scenario(sim.Scenario(), inputs())
    assert central.failure_arms == (0.0, 0.0, 0.0)
    assert central.k_spreads([0, 2])[0].ravel().tolist() == [0.056, 0.078]


def test_compare_flags_rates_beyond_two_mc_errors() -> None:
    script = load_script()

    def summary(settings: dict, p: float, name: str) -> dict:
        row = {"looks": "36_60", "interval": "bootstrap_t", "rule": "eq_min36", "alpha": 0.04,
               "floor": 0.0, **{k: p for k, _ in script.COMPARED_RATES},
               **{s: 0.002 for _, s in script.COMPARED_RATES}}  # fmt: skip
        return {
            "scenarios": [
                {
                    "name": name,
                    "settings": settings,
                    "reps": 20_000,
                    "results": [row],
                    "diagnostics": {"block_bias": 0.001, "block_bias_se": 0.0001},
                }
            ]
        }

    settings = dataclasses.asdict(sim.Scenario(w_max=0.47))
    archived = summary(settings, 0.02, "old")
    rerun = summary({**settings, "sigma_u_arms": [0.056, 0.078, 0.078]}, 0.03, "new")
    found = script.compare_runs(rerun, [archived])
    assert found["unmatched"] == [] and found["compared"] == len(script.COMPARED_RATES) + 1
    assert found["beyond_2se"] == len(script.COMPARED_RATES)  # 1 pp at SE 0.28 pp
    assert found["flagged"][0]["z"] == pytest.approx(0.01 / math.hypot(0.002, 0.002))
    assert script.compare_runs(summary(settings, 0.021, "same"), [archived])["beyond_2se"] == 0


def test_mechanism_mode_smoke(tmp_path: Path) -> None:
    """A tiny grid end to end through the script: summary, results and the inventory of every
    file it wrote."""
    script = load_script()
    out = tmp_path / "mechanism"
    script.main(["mechanism", "--scenarios", "mech_skewed_sd10_r2-1_rho-0.5_c1-5_c2+5_none",
                 "mech_rare_only_sdna_r1-1_rho+0.5_c1+5_c2+0_independent", "--reps", "6", *TINY,
                 "--levels", "0.95", "0.97", "--out", str(out)])  # fmt: skip
    summary = json.loads((out / "summary.json").read_text())
    assert summary["settings"]["stream"] == "mechanism-calibration"
    assert summary["settings"]["levels"] == [0.95, 0.97] and summary["settings"]["resamples"] == 200
    assert [s["reps"] for s in summary["scenarios"]] == [6, 6]
    for s in summary["scenarios"]:
        assert {c: s["parameters"]["estimand"][c] for c in sim.COMPONENTS} == pytest.approx(
            dict(zip(sim.COMPONENTS, (s["settings"]["c1"], s["settings"]["c2"]), strict=True))
        )
        assert len(s["levels"]) == 2 and s["minimum_met"]["value"] == 1.0
    assert summary["calibrations"] and summary["checks"]["per_level"][0]["level"] == 0.95
    assert "## Calibration rule" in (out / "results.md").read_text()
    lines = (out / "inventory.tsv").read_text().splitlines()
    assert lines[0] == "path\tsize\tsha256"
    listed = {row.split("\t")[0]: row.split("\t")[1:] for row in lines[1:]}
    assert set(listed) == {"inputs.json", "results.md", "summary.json"}
    for name, (size, digest) in listed.items():
        data = (out / name).read_bytes()
        assert int(size) == len(data) and digest == hashlib.sha256(data).hexdigest()
