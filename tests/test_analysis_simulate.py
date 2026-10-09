import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from zero_field_refusal.analysis import simulate
from zero_field_refusal.analysis.decisions import DECISIONS, MECHANISMS
from zero_field_refusal.analysis.simulate import (
    GenerativeModel,
    Scenario,
    SimulationSettings,
    Variant,
    acceptance,
    beta_concentration,
    draw_run_rates,
    quick_settings,
    recommend,
    simulate_replicate,
    true_rate,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "simulate_analysis.py"
KWARGS = dict(base_seed=0, methods=("percentile",), levels=(0.95, 0.99), margin=0.1)


@pytest.mark.parametrize(
    ("sigma", "base"),
    [(0.10, "logistic"), (0.22, "logistic"), (0.15, "saturating"), (0.15, "floor")],
)
def test_beta_concentration_gives_the_seed_sd_in_the_middle_of_the_curve(
    sigma: float, base: str
) -> None:
    model = GenerativeModel()
    kappa = beta_concentration(model, sigma, base)
    rng = np.random.default_rng(0)
    n = 200_000
    log_k = np.mean(model.curve_log_k) + rng.normal(0.0, model.log_k_sd, n)
    rates = draw_run_rates(model.base_rate(log_k, base), kappa, rng.standard_normal(n))
    observed = rng.binomial(model.judged, rates) / model.judged
    assert observed.std() == pytest.approx(sigma, abs=0.002)


def test_misspecified_nulls_share_one_non_logistic_curve() -> None:
    model = GenerativeModel()
    log_k = np.linspace(-1.0, 3.5, 7)
    logistic = model.sgd_rate(log_k)
    np.testing.assert_allclose(model.base_rate(log_k, "saturating"), 0.7 * logistic)
    np.testing.assert_allclose(model.base_rate(log_k, "floor"), 0.05 + 0.95 * logistic)
    for base in ("saturating", "floor"):
        scenario = Scenario("narrow", 0.0, 0.15, 0.0, base=base)
        rates = [true_rate(model, scenario, arm, log_k, 1.5) for arm in simulate.SIM_ARMS]
        for other in rates[1:]:
            np.testing.assert_array_equal(rates[0], other)
    assert (
        Scenario("standard", 0.0, 0.15, 0.0).name == "standard_shift_d+0.00_fa-between_s0.15_r0.00"
    )
    assert Scenario("standard", 0.0, 0.15, 0.0, base="floor").name.startswith(
        "standard_shift-floor_"
    )


def test_frozen_adam_can_sit_at_an_offset_from_sgd() -> None:
    model = GenerativeModel()
    log_k = np.linspace(-1.0, 3.5, 7)
    scenario = Scenario("standard", 0.15, 0.15, 0.0, frozen_adam="sgd", frozen_offset=-0.05)
    frozen = true_rate(model, scenario, "adam_frozen", log_k, 1.5)
    np.testing.assert_allclose(frozen, np.clip(model.sgd_rate(log_k) - 0.05, 0.0, 1.0))
    assert "_fa-sgd-0.05_" in scenario.name
    assert "_fa-sgd_" in Scenario("standard", 0.15, 0.15, 0.0, frozen_adam="sgd").name


def test_crossing_alternatives_have_the_exact_delta_at_the_realized_k_star() -> None:
    model = GenerativeModel()
    scenario = Scenario("standard", -0.10, 0.15, 0.0, curves="crossing")
    x_star = np.array([1.37])
    delta = true_rate(model, scenario, "adamw", x_star, 1.37) - model.sgd_rate(x_star)
    assert delta[0] == pytest.approx(-0.10)


def test_run_rates_keep_the_curve_as_their_mean() -> None:
    latent = norm.ppf(np.linspace(0.0005, 0.9995, 1000))
    rates = draw_run_rates(np.full(1000, 0.2), 3.0, latent)
    assert rates.mean() == pytest.approx(0.2, abs=0.002)
    assert draw_run_rates(np.array([0.0, 1.0]), 3.0, np.zeros(2)).tolist() == [0.0, 1.0]


def test_replicates_are_reproducible_and_variants_share_the_data() -> None:
    scenario = Scenario(placement="narrow", delta=0.1, sigma=0.15, rho=0.5, reps=1)
    model = GenerativeModel()
    kappa = beta_concentration(model, scenario.sigma)
    variants = (Variant(100, 10.0), Variant(100, math.inf))
    first = simulate_replicate(model, scenario, kappa, 3, variants=variants, **KWARGS)
    second = simulate_replicate(model, scenario, kappa, 3, variants=variants, **KWARGS)
    for variant in variants:
        for name in ("look1", "final", "mechanism"):
            np.testing.assert_array_equal(
                getattr(first[variant], name), getattr(second[variant], name)
            )
        assert first[variant].diagnostics == second[variant].diagnostics
    default, keep_all = (first[v].diagnostics for v in variants)
    assert default["log_k_star"] == keep_all["log_k_star"]
    # In the narrow placement the SGD cell at log K = -1 lies below C_lo/10.
    assert default["excluded_runs_look1"] > 0 and keep_all["excluded_runs_look1"] == 0


def test_look_2_adds_seeds_20_to_39_and_decides_only_where_look_1_was_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = {decision: DECISIONS.index(decision) for decision in DECISIONS}
    looks = []

    def fake_look(table, selection, resamples, seed, methods, levels, margin):
        looks.append(table)
        if len(looks) == 1:
            decisions = [[code["inconclusive"], code["positive"]]]
        else:
            decisions = [[code["equivalence"], code["opposite"]]]
        mechanisms = [[len(looks), len(looks)]]
        return np.array(decisions, np.int8), np.array(mechanisms, np.int8), {"delta_hat": 0.0}

    monkeypatch.setattr(simulate, "_look", fake_look)
    model = GenerativeModel()
    scenario = Scenario("standard", 0.0, 0.15, 0.5)
    outcome = simulate_replicate(
        model, scenario, beta_concentration(model, 0.15), 0, variants=(Variant(100),), **KWARGS
    )[Variant(100)]
    first, second = looks
    assert sorted(first.seed.unique()) == list(range(20))
    assert sorted(second.seed.unique()) == list(range(40))
    pd.testing.assert_frame_equal(first, second[second.seed < 20])
    assert outcome.look1.tolist() == [[code["inconclusive"], code["positive"]]]
    assert outcome.final.tolist() == [[code["equivalence"], code["positive"]]]
    assert outcome.mechanism.tolist() == [[2, 1]]


def _frame(rows: list[tuple]) -> pd.DataFrame:
    columns = ["scenario", "placement", "curves", "base", "delta", "method", "level"]
    columns += ["reps", "p_positive", "p_opposite", "p_equivalence", "look1_positive"]
    return pd.DataFrame(rows, columns=columns)


def _grid() -> pd.DataFrame:
    rows = []
    for method, level, fp, power in (
        ("percentile", 0.9706, 0.030, 0.90),
        ("percentile", 0.99, 0.020, 0.80),
        ("bca", 0.9706, 0.024, 0.85),
        ("bca", 0.99, 0.010, 0.70),
    ):
        common = ("standard", method, level, 1000)
        rows += [
            (
                "null-logistic",
                "standard",
                "shift",
                "logistic",
                0.0,
                *common[1:],
                fp / 2,
                0.01,
                0.9,
                0,
            ),
            ("null-floor", "standard", "shift", "floor", 0.0, *common[1:], fp, 0.01, 0.9, 0),
            ("alt-minus", "standard", "shift", "logistic", -0.1, *common[1:], 0.0, 0.9, 0.04, 0),
            (
                "alt-cross",
                "standard",
                "crossing",
                "logistic",
                0.1,
                *common[1:],
                power - 0.1,
                0,
                0.01,
                0.5,
            ),
        ]
        for delta in (0.10, 0.15, 0.20):
            rows.append(
                ("alt", "standard", "shift", "logistic", delta, *common[1:], power, 0, 0.01, 0.5)
            )
    return _frame(rows)


def test_acceptance_takes_the_worst_case_over_all_nulls_including_misspecified() -> None:
    verdicts = {(v["method"], v["level"]): v for v in acceptance(_grid())}
    failing = verdicts[("percentile", 0.9706)]
    assert not failing["passes"]
    assert failing["max_p_positive_null"]["scenario"] == "null-floor"
    assert failing["max_p_positive_null"]["value"] == pytest.approx(0.030)
    assert failing["max_p_positive_null"]["mc_se"] == pytest.approx(np.sqrt(0.03 * 0.97 / 1000))
    passing = verdicts[("percentile", 0.99)]
    assert passing["passes"]
    assert passing["max_p_equivalence_at_10pp"]["scenario"] == "alt-minus"
    assert passing["power"]["shift"]["0.10"]["mean"] == pytest.approx(0.80)
    assert passing["power"]["crossing"]["0.10"]["mean"] == pytest.approx(0.70)
    assert "0.15" not in passing["power"]["crossing"]


def test_recommend_keeps_the_planning_level_when_it_passes_and_checks_resample_stability() -> None:
    settings = SimulationSettings(
        model=GenerativeModel(),
        grid=simulate.Part("results", (), (Variant(2000),), Variant(2000)),
        resample_comparison=simulate.Part(
            "resamples", (), tuple(Variant(r) for r in (1000, 2000, 4000)), Variant(4000)
        ),
        exclusion_sensitivity=None,
        methods=("percentile", "bca"),
        levels=(0.9706, 0.99),
    )
    rows = []
    for resamples, shift in ((1000, 0.03), (2000, 0.005), (4000, 0.0)):
        for method in ("percentile", "bca"):
            for level in (0.9706, 0.99):
                rows.append(
                    {
                        "scenario": "s",
                        "resamples": resamples,
                        "method": method,
                        "level": level,
                        "p_positive": 0.02 + shift,
                        "p_opposite": 0.01,
                        "p_equivalence": 0.5,
                        "disagree_with_reference": shift,
                    }
                )
    comparison = pd.DataFrame(rows)
    recommendation = recommend(acceptance(_grid()), comparison, settings)
    # bca passes at the planning level; percentile only at 0.99, with more power at 10 pp.
    assert recommendation["per_method"]["bca"]["level"] == 0.9706
    assert recommendation["per_method"]["percentile"]["level"] == 0.99
    assert recommendation["method"] == "bca"
    assert recommendation["resamples"] == 4000
    unstable = comparison.assign(
        p_positive=np.where(comparison.resamples == 2000, 0.05, comparison.p_positive)
    )
    assert recommend(acceptance(_grid()), unstable, settings)["resamples"] is None


def test_full_settings_have_the_planned_parts() -> None:
    settings = simulate.full_settings()
    counts = {part.name: len(part.scenarios) for part in settings.parts}
    assert counts == {"results": 144, "resamples": 3, "exclusion": 36, "h2_boundary": 48}
    for part in settings.parts:
        assert all(isinstance(s, Scenario) for s in part.scenarios), part.name
        assert len({s.name for s in part.scenarios}) == len(part.scenarios), part.name
    nulls = [s for s in settings.grid.scenarios if s.delta == 0]
    assert len(nulls) == 48 and {s.reps for s in nulls} == {3000}
    assert sum(s.reps for s in settings.grid.scenarios) == 240_000


def test_quick_settings_cover_every_part() -> None:
    settings = quick_settings()
    assert {s.base for s in settings.grid.scenarios} >= {"logistic", "saturating", "floor"}
    assert {s.curves for s in settings.grid.scenarios} == {"shift", "crossing"}
    assert {v.exclusion_factor for v in settings.exclusion_sensitivity.variants} == {
        3.0,
        10.0,
        30.0,
        math.inf,
    }
    assert {s.frozen_offset for s in settings.h2_boundary.scenarios} == {-0.10, -0.05, 0.05, 0.10}


def _run_script(*args: str):
    spec = importlib.util.spec_from_file_location("simulate_analysis", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.main(list(args))


def _rates(table: pd.DataFrame) -> pd.DataFrame:
    return table.filter(regex="^(p_|look1_|stop_|h2_)")


def test_quick_simulation_writes_rates_in_the_unit_interval(tmp_path: Path) -> None:
    _run_script("--quick", "--out", str(tmp_path), "--workers", "1")
    for name in ("results", "resamples", "exclusion", "h2_boundary"):
        table = pd.read_csv(tmp_path / f"{name}.csv")
        rates = _rates(table)
        assert len(table) > 0 and rates.notna().all().all(), name
        assert ((rates >= 0) & (rates <= 1)).all().all(), name
    assert {f"h2_{m}" for m in MECHANISMS} <= set(table.columns)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert {"settings", "versions", "acceptance", "recommendation", "exclusion"} <= set(summary)
    boundary = summary["h2_boundary"]["rows"]
    assert len(boundary) == 4 and all(sum(row["h2"].values()) <= 1 + 1e-9 for row in boundary)
    assert summary["versions"]["git"].keys() == {"commit", "dirty"}


def test_confirmation_run_reports_worst_case_nulls(tmp_path: Path) -> None:
    _run_script("--confirm", "--quick", "--seed", "1", "--out", str(tmp_path), "--workers", "1")
    results = pd.read_csv(tmp_path / "results.csv")
    assert (results["delta"] == 0).all() and (results["sigma"] == 0.22).all()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["mode"] == "confirm"
    worst = summary["null_worst_cases"]
    assert {(w["method"], w["level"]) for w in worst} == set(
        results[["method", "level"]].itertuples(index=False, name=None)
    )
