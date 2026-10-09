import numpy as np
import pytest
import statsmodels.api as sm
from scipy.special import expit, logit

from zero_field_refusal.analysis import curves
from zero_field_refusal.analysis.curves import cluster_influence, fit_logistic


def _counts(seed: int = 0, n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    log_k = rng.uniform(-1.0, 3.0, n)
    trials = rng.integers(250, 301, n)
    successes = rng.binomial(trials, expit(-2.0 + 0.8 * log_k))
    return log_k, successes, trials


def _statsmodels_glm(log_k, successes, trials, **kwargs):
    endog = np.column_stack([successes, trials - successes])
    return sm.GLM(endog, sm.add_constant(log_k), family=sm.families.Binomial(), **kwargs)


def test_fit_matches_statsmodels_with_frequency_weights() -> None:
    log_k, successes, trials = _counts()
    rng = np.random.default_rng(1)
    weights = rng.multinomial(40, np.full(40, 1 / 40), size=6).astype(float)
    weights[0] = 1.0
    fit = fit_logistic(log_k, successes, trials, weights)
    assert fit.converged.all() and not fit.separated.any()
    for row in range(len(weights)):
        glm = _statsmodels_glm(log_k, successes, trials, freq_weights=weights[row])
        intercept, slope = glm.fit(tol=1e-12).params
        assert fit.intercept[row] == pytest.approx(intercept, abs=1e-6)
        assert fit.slope[row] == pytest.approx(slope, abs=1e-6)


def test_predict_is_the_logistic_curve_per_row() -> None:
    log_k, successes, trials = _counts()
    fit = fit_logistic(log_k, successes, trials)
    at = np.array([0.0, 1.5])
    assert fit.predict(at).shape == (1, 2)
    np.testing.assert_allclose(fit.predict(at)[0], expit(fit.intercept[0] + fit.slope[0] * at))


@pytest.mark.parametrize(
    "successes",
    [[0, 0, 0, 300, 300, 300], [0, 0, 0, 0, 0, 0], [300, 300, 300, 300, 300, 300]],
    ids=["split-along-K", "all-zero", "all-harmful"],
)
def test_separation_is_flagged_and_stays_finite(successes: list[int]) -> None:
    log_k = np.linspace(0.0, 2.5, 6)
    fit = fit_logistic(log_k, np.array(successes), np.full(6, 300))
    assert fit.separated.all()
    assert np.isfinite(fit.intercept).all() and np.isfinite(fit.slope).all()
    rates = fit.predict(log_k)[0]
    np.testing.assert_allclose(rates, np.array(successes) / 300, atol=1e-6)


def test_separation_is_flagged_per_weight_row() -> None:
    log_k = np.linspace(0.0, 2.5, 6)
    successes = np.array([0, 0, 0, 60, 90, 120])
    weights = np.array([[1, 1, 1, 1, 1, 1], [2, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1]])
    fit = fit_logistic(log_k, successes, np.full(6, 300), weights)
    assert fit.separated.tolist() == [False, True, False]
    assert fit.converged[[0, 2]].all()


def test_a_step_that_never_improves_the_objective_is_not_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = curves._penalized_loglik
    calls = []

    def worse_after_the_start(*args):
        calls.append(1)
        value = real(*args)
        return value if len(calls) == 1 else np.full_like(value, -1e300)  # every step is worse

    monkeypatch.setattr(curves, "_penalized_loglik", worse_after_the_start)
    log_k, successes, trials = _counts()
    fit = fit_logistic(log_k, successes, trials)
    # The start (pooled rate, zero slope) is kept, and the row stops without converging.
    assert fit.slope[0] == 0.0
    assert fit.intercept[0] == pytest.approx(logit(successes.sum() / trials.sum()))
    assert not fit.converged[0]
    assert len(calls) == 1 + curves.MAX_HALVINGS


def test_cluster_influence_gives_the_cluster_robust_delta_method_variance() -> None:
    log_k, successes, trials = _counts(seed=2, n=60)
    clusters = np.arange(60) % 12
    fit = fit_logistic(log_k, successes, trials)
    at = np.array([0.5, 1.5])
    influence = cluster_influence(fit, log_k, successes, trials, None, clusters, 12, at)
    assert influence.shape == (1, 12, 2)
    variance = (influence[0] ** 2).sum(axis=0)

    glm = _statsmodels_glm(log_k, successes, trials).fit(
        tol=1e-12, cov_type="cluster", cov_kwds={"groups": clusters, "use_correction": False}
    )
    rate = expit(glm.params[0] + glm.params[1] * at)
    gradient = (rate * (1 - rate))[:, None] * np.column_stack([np.ones(2), at])
    expected = np.einsum("li,ij,lj->l", gradient, glm.cov_params(), gradient)
    np.testing.assert_allclose(variance, expected, rtol=1e-5)
