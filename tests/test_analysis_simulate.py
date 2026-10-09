import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from zero_field_refusal.analysis.simulate import (
    GenerativeModel,
    Scenario,
    beta_concentration,
    draw_run_rates,
    simulate_replicate,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "simulate_analysis.py"


@pytest.mark.parametrize("sigma", [0.10, 0.22])
def test_beta_concentration_gives_the_seed_sd_in_the_middle_of_the_curve(sigma: float) -> None:
    model = GenerativeModel()
    kappa = beta_concentration(model, sigma)
    rng = np.random.default_rng(0)
    n = 200_000
    log_k = np.mean(model.curve_log_k) + rng.normal(0.0, model.log_k_sd, n)
    rates = draw_run_rates(model.sgd_rate(log_k), kappa, rng.standard_normal(n))
    observed = rng.binomial(model.judged, rates) / model.judged
    assert observed.std() == pytest.approx(sigma, abs=0.002)


def test_run_rates_keep_the_curve_as_their_mean() -> None:
    latent = norm.ppf(np.linspace(0.0005, 0.9995, 1000))
    rates = draw_run_rates(np.full(1000, 0.2), 3.0, latent)
    assert rates.mean() == pytest.approx(0.2, abs=0.002)
    assert draw_run_rates(np.array([0.0, 1.0]), 3.0, np.zeros(2)).tolist() == [0.0, 1.0]


def test_replicates_are_reproducible() -> None:
    scenario = Scenario(placement="standard", delta=0.1, sigma=0.15, rho=0.5, reps=1)
    kwargs = dict(
        base_seed=0, resamples=(100,), methods=("percentile",), levels=(0.95,), margin=0.1
    )
    model = GenerativeModel()
    kappa = beta_concentration(model, scenario.sigma)
    first = simulate_replicate(model, scenario, kappa, 3, **kwargs)[100]
    second = simulate_replicate(model, scenario, kappa, 3, **kwargs)[100]
    for name in ("look1", "final", "mechanism"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["k_star_defined"] == 1.0


def test_quick_simulation_writes_rates_in_the_unit_interval(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("simulate_analysis", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.main(["--quick", "--out", str(tmp_path), "--workers", "1"])

    results = pd.read_csv(tmp_path / "results.csv")
    rates = results.filter(regex="^(p_|look1_|stop_|h2_)")
    assert len(results) > 0 and rates.notna().all().all()
    assert ((rates >= 0) & (rates <= 1)).all().all()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert {"settings", "versions", "acceptance", "recommendation"} <= set(summary)
